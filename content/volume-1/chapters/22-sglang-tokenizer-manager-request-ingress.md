# 22장. SGLang TokenizerManager에서 scheduler까지 요청이 건너가는 법

HTTP server가 JSON을 받았다고 GPU scheduler가 곧바로 request를 아는 것은 아니다. OpenAI chat request는 message와 tool·sampling 계약을 검증하고 chat template를 거쳐 text 또는 token IDs가 된다. multimodal request라면 image/video/audio 전처리 state도 만들어진다. `TokenizerManager`는 이 결과를 IPC message로 보내고 request ID별 async state를 보유하다가 detokenizer/output 경로에서 돌아오는 batch output을 원 HTTP coroutine에 배달한다.

이 경계를 tokenizer 호출 한 번으로 줄이면 queue와 ownership이 사라진다. tokenizer가 느려 GPU가 굶는 문제, request ID가 IPC 전에 abort되어 취소가 놓치는 race, client가 느려 stream queue가 쌓이는 문제, worker가 죽었는데 scheduler request가 남는 문제를 설명하지 못한다. 이 장은 SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`의 source로 대표 text 요청을 먼저 완주한다.

## 22.1 HTTP ingress에서 R17의 첫 identity를 만든다

사용자가 `/v1/chat/completions`에 messages와 stream=true를 보낸다. FastAPI route는 protocol object로 validation한 뒤 OpenAI serving handler에 넘긴다. handler는 model name, chat template, tools/response format와 sampling fields를 SGLang `GenerateReqInput`으로 적응시킨다. 이 객체가 아직 scheduler `Req`는 아니다.

HTTP 경로를 읽을 때는 [chat endpoint](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/http_server.py#L1700-L1740)를 첫 기준점으로 삼는다. 가공하지 않은 생성 요청의 입구는 [generate_request](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/http_server.py#L894-L935)다.

그다음 프로토콜의 [ChatCompletionRequest](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/protocol.py#L758-L930)와 서빙 핸들러를 함께 읽는다. 그래야 외부 필드가 내부 요청으로 바뀌는 지점을 놓치지 않는다.

request identity ledger에는 HTTP trace ID, OpenAI response ID, SGLang rid, batch index와 incarnation을 둔다. 한 API request가 n generations로 여러 rid를 만들 수 있다. client retry는 같은 payload여도 새 logical request다. rid가 할당되기 전 disconnect가 오면 무엇을 abort할지 애매하다.

non-stream은 generator를 끝까지 소비해 JSON 하나를 만들고 stream은 각 output을 SSE chunk로 바꾼다. 첫 byte 뒤 error는 HTTP status를 바꾸기 어렵다. route가 background abort task를 붙이는 [response construction](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/http_server.py#L894-L935)을 읽을 때 generator cleanup까지 본다.

### 사건으로 first divergence를 찾는 순서

GPU utilization 20%인데 scheduler queue도 짧으면 tokenization queue와 template/multimodal preprocessing을 본다. client disconnect 뒤 GPU memory가 남으면 disconnect→abort enqueue→scheduler removal→KV release→manager cleanup timestamps를 잇는다. abort API 성공을 release 증거로 쓰지 않는다.

mixed stream에서 A chunk가 B response에 들어가면 batch rids, state lookup, mailbox dispatch와 incarnation을 본다. text-only는 맞고 image decode부터 틀리면 rendered IDs, placeholder/grid/mRoPE metadata를 ingress에서 비교한다. worker restart 뒤 request가 영원히 기다리면 registered pending state와 IPC send/failure broadcast를 본다.

slow client 하나로 모든 ITL이 나빠지면 handle loop가 client send를 기다리는지와 output mailbox bounds를 본다. provider/model token timestamp와 manager dispatch, client write를 분리한다. 같은 현상을 scheduler kernel 탓으로 돌리지 않는다.

이제 R17의 정상 수명을 기준으로, 뒤의 변형과 실험을 하나의 workbook 안에서 닫아 보자.

TokenizerManager는 tokenizer 호출뿐 아니라 API input을 model-ready contract로 만들고 process boundary를 건너며 여러 output을 원 coroutine으로 돌리고 abort를 전달한다. 이 분리는 HTTP/client 지연과 scheduler receive loop를 떼어 놓는다.

대가는 pending state, IPC queue, rid demux, abort race와 worker failure다. scale-out은 throughput을 늘리지만 ordering과 duplicate dispatch를 관리해야 한다. 중앙 output loop는 효율적이지만 head-of-line blocking을 막아야 한다.

최종 invariant는 이렇다. **한 API request의 rendered/tokenized/multimodal input과 rid가 정확히 한 scheduler request incarnation으로 전달되고, output·finish·abort event가 같은 rid의 client mailbox와 engine state에 한 번씩 귀속되어야 한다.**

이 invariant가 맞아도 model 품질은 scheduler/model/kernel 증거가 필요하다. ingress가 증명하는 것은 입력 계약과 수명이다. 다음 장에서 scheduler가 이 Req를 waiting/running batch와 KV capacity로 바꾸는 과정을 본다.

### 대표 text 요청을 실제 시간표로 다시 걷는다

t=0에 client가 chat request를 보낸다. HTTP body parse와 protocol validation이 2ms, chat template가 3ms, tokenizer가 5ms를 썼다고 하자. TokenizerManager가 rid R17을 할당하고 state mailbox를 등록한 뒤 t=12ms에 tokenized message를 IPC로 보낸다. scheduler receiver가 t=14ms에 받고 Req를 만든다. waiting queue admission은 t=16ms다.

GPU prefill이 끝나 첫 output token이 t=60ms에 나왔고 detokenizer batch가 manager handle loop에 t=62ms 도착한다. manager는 batch index에서 R17을 찾아 mailbox에 event를 넣는다. OpenAI stream generator가 이를 SSE chunk로 바꾸고 client write가 t=65ms에 끝난다. provider/model TTFT 44ms와 user-observed TTFT 65ms 사이에 ingress와 output delivery 21ms가 있다.

이 숫자는 실제 측정 주장이 아니라 timestamp owner를 설명하는 예다. API received, template start/end, tokenize end, state registered, IPC sent/received, Req created/admitted, model first token, detokenizer output, manager dispatch, first client byte를 별도 field로 둔다. 한 timestamp를 여러 단계에 재사용하지 않는다.

두 번째 token부터 manager output과 client write가 반복된다. request finish는 scheduler finish reason, detokenizer final output, mailbox finished, HTTP generator completion과 background cleanup 순으로 보일 수 있다. scheduler가 끝났다는 사실과 client connection이 안전히 닫혔다는 사실은 다르다.

### trace context가 process boundary에서 끊기는 사건

API span은 tokenization done까지 보이고 scheduler span은 별 root로 시작한다. operator는 같은 request인지 수동으로 시간만 맞춘다. TokenizedGenerateReqInput 또는 side-channel에 trace context/rid correlation이 전달되지 않은 경우다.

full tracing object를 pickle하는 것이 아니라 stable trace/span IDs와 sampling decision을 전달할 수 있다. security와 version compatibility를 본다. scheduler output이 manager로 돌아올 때 context를 다시 찾을 수도 있다.

clock source와 unit이 process마다 다른지 확인한다. wall clock skew가 음수 stage duration을 만들 수 있다. monotonic timestamps는 process 간 직접 비교가 어려울 수 있어 calibration/central event를 사용한다. source가 제공하는 req time stats field를 실제 owner에 매핑한다.

## 22.2 messages와 token IDs의 identity를 고정한다

`TokenizerManager`는 tokenizer object만 감싼 class가 아니다. scheduler socket, detokenizer receive loop, request state mapping, multimodal receiver, metrics와 abort control을 구성한다. [TokenizerManager initialization](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L386-L620)에서 owner를 찾는다.

대표 request는 [generate_request](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L765-L890)로 들어온다. handle loop를 보장하고 input을 tokenize/normalize하며 request state를 등록하고 IPC로 보낸 뒤 output을 기다리거나 yield한다. await 지점마다 owner가 바뀐다.

rid별 state/queue/event는 handle loop가 받은 batch output을 원 coroutine에 전달하는 mailbox다. 단일 receive loop가 batch를 rid별로 demultiplex한다. state는 rid, output event/queue, finished, accumulation과 created/dispatch timestamps를 가진다. IPC send 전에 등록해야 빠른 output을 놓치지 않고, 등록 뒤 send failure면 orphan을 cleanup해야 한다.

text가 들어오면 tokenizer와 context validation이 필요하다. input IDs를 직접 받으면 tokenization은 생략할 수 있지만 vocab range와 special token semantics 책임이 caller 쪽으로 이동한다. [GenerateReqInput](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/io_struct.py#L160-L360)과 [TokenizedGenerateReqInput](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/io_struct.py#L941-L1054)의 field 차이를 읽는다.

### template를 두 번 적용하는 사건

OpenAI serving handler가 messages에 chat template를 적용해 rendered prompt를 만들었다. 그런데 downstream raw path가 이 문자열을 다시 messages처럼 처리하거나 tokenizer가 add_special_tokens를 중복 적용했다고 하자. BOS 또는 assistant generation marker가 두 번 들어간다. request는 성공하고 shape도 정상인데 output 품질과 prefix cache hit가 달라진다.

input ledger에는 original messages digest, rendered text/IDs owner, add_special_tokens flag, final IDs digest를 둔다. template before/after token count와 first/last special IDs를 bounded하게 본다. TokenizerManager가 받은 field가 raw messages인지 text인지 IDs인지 확인한다.

OpenAI route와 raw `/generate` route는 다른 trust boundary다. raw text를 chat message로 자동 해석하지 않고 caller contract를 따른다. same visible string이 same token IDs를 뜻하지 않는다. tokenizer/template revision과 option provenance를 고정한다.

수정 후 system+user, tool call, assistant prefix, already-tokenized input을 별도 fixture로 검증한다. output text parity만 보지 않고 exact input IDs와 prompt token usage, prefix identity를 본다.

### tokenizer worker가 exception을 던지는 순간

malformed tool schema가 template 함수에서 exception을 낸다. 이 시점이 rid/state 등록 전이면 route가 validation error를 바로 반환할 수 있다. 등록 후라면 mailbox terminal error와 mapping cleanup이 필요하다. IPC send 뒤라면 scheduler abort도 필요할 수 있다.

exception mapper가 user 4xx와 server 5xx를 구분한다. invalid messages/content는 client error이고 tokenizer process crash는 server error다. raw exception에 prompt content가 포함될 수 있어 redaction한다. trace에는 stage와 safe error category를 둔다.

worker pool에서 한 task exception이 loop 전체를 죽이는지, supervisor가 restart하는지 본다. restart 동안 queued futures가 fail되는지 재전송되는지, duplicate dispatch 가능성이 있는지 확인한다. worker PID alive metric만으로 outstanding futures health를 증명하지 않는다.

### multimodal remote fetch는 별도 deadline과 보안 경계다

image URL을 받으면 DNS/connect/read, size/type validation과 decode가 필요하다. model request timeout 전체를 remote fetch가 소비할 수 있다. fetch deadline, maximum bytes/pixels, allowed scheme/host와 SSRF 방어를 ingress contract로 둔다.

download는 성공했지만 decode가 CPU/memory를 과도하게 쓰는 decompression bomb일 수 있다. content-length만 믿지 않고 decoded dimension을 제한한다. cache가 있으면 tenant/privacy와 URL freshness identity를 본다.

preprocessing 결과 image tensor/features가 IPC로 직접 가는지 shared memory/encoder process handle로 가는지 ownership을 추적한다. large data serialization이 event loop를 막을 수 있다. handle lifetime은 request abort/worker crash에서 회수되어야 한다.

placeholder token count와 image feature/grid count invariant를 확인한다. mismatch를 scheduler/model까지 보내 늦게 실패시키기보다 가능한 ingress에서 검증한다. 그러나 model-specific merge 규칙을 gateway 일반성으로 추측하지 않고 processor source를 따른다.

### request ingress의 byte·CPU 원장을 손으로 만든다

평균 rendered prompt 8 KiB, token IDs 2048개라고 하자. Python int list는 raw int32 tensor보다 훨씬 큰 object overhead를 가질 수 있다. IPC serialization이 list를 compact array로 바꾸는지 pickle/msgspec 형태인지에 따라 payload와 CPU가 달라진다. 단순 `2048×4=8 KiB`로 process memory를 확정하지 않는다.

sampling params와 strings, stop lists, logprob fields, trace metadata가 붙는다. batch 128이면 payload bytes와 serialization pause가 커질 수 있다. long prompt 한 개와 short 127개의 batch가 함께 serialized될 때 tail과 fairness를 본다. message batching이 syscall overhead를 줄이지만 한 large item이 나머지를 지연할 수 있다.

text char/byte length, token count, rendered size, tokenized IPC serialized bytes, serialization/deserialization time를 stage별로 관측한다. prompt 내용을 label로 넣지 않는다. length bucket과 request trace exemplar를 사용한다.

multimodal data가 image raw bytes 5 MiB, decoded tensor 50 MiB라면 어떤 representation이 process boundary를 넘는지가 결정적이다. shared memory handle이면 IPC message는 작지만 shared allocation lifetime이 있다. 직접 serialize하면 CPU와 copy, socket limits가 커진다. source에서 actual field와 receiver를 본다.

### CPU batching은 GPU continuous batching과 다른 층이다

TokenizerManager가 여러 input을 batch tokenize하는 것은 CPU tokenizer throughput 최적화다. scheduler continuous batch는 서로 다른 sequence의 prefill/decode tokens를 GPU step에 섞는다. 두 batch의 membership과 timing은 같지 않다.

CPU batch를 기다려 모으면 tokenizer throughput은 오르지만 낮은 load에서 queue delay가 TTFT를 늘린다. scheduler는 tokenized request가 도착해야 후보로 볼 수 있다. CPU batching window와 GPU scheduling iteration을 별도 tuning한다.

batch tokenization failure가 한 item 때문에 전체 batch를 실패시키는지 item-level error를 지원하는지 본다. 한 malformed request가 127 healthy requests를 지연/실패시키면 isolation이 약하다. error demux와 parent mailbox를 설계한다.

### tokenizer cache의 identity를 조심한다

같은 text를 반복 tokenize하면 cache를 둘 수 있지만 key에는 tokenizer revision, add_special_tokens, truncation, template/rendered bytes가 필요하다. raw messages와 rendered prompt를 같은 key로 쓰지 않는다. adapter/model group별 tokenizer가 다르면 분리한다.

cache hit은 CPU work를 줄이지만 request semantic을 바꾸면 안 된다. mutable list를 shared return해 caller가 수정하면 이후 hit가 오염될 수 있다. immutable/copy contract를 본다. cache memory와 sensitive prompt retention 정책도 필요하다.

prefix KV cache hit와 tokenizer result cache hit는 다른 saved work다. 전자는 GPU model prefill을 줄이고 후자는 CPU tokenization만 줄인다. metric 이름과 identity를 나눈다. tokenizer cache hit인데 TTFT가 그대로일 수 있다.

**사건 J — direct input_ids가 잘못된 tokenizer revision이다.**

client가 tokenizer A로 만든 IDs를 model B endpoint에 보낸다. IDs는 vocab range 안이라 validation을 통과하지만 다른 token 의미다. SGLang ingress는 bytes 원문을 몰라 이를 자동 복원할 수 없다. direct-ID API의 trust contract를 문서화한다.

request metadata에 tokenizer revision digest를 선택적으로 요구/검증할 수 있지만 모든 client가 제공하지 않는다. strict environment에서는 gateway/client tokenizer를 고정하고 canary exact IDs를 확인한다. invalid semantic output을 scheduler bug로 분류하지 않는다.

**chat template failure를 친절하게 설명한다**

model tokenizer에 chat template가 없거나 request tools/roles를 지원하지 않을 수 있다. serving handler가 default/override template를 선택하는 순서와 error를 본다. 빈 template로 raw concatenation한다고 추측하지 않는다.

template render는 text를 만들지만 tokenization과 special token addition이 뒤에 있다. `add_generation_prompt` 또는 assistant prefix가 빠지면 model이 continuation할 위치가 달라진다. exact rendered representation과 final IDs를 stage로 나눈다.

template override는 cache identity와 semantic deployment identity를 바꾼다. version rollout에서 old/new requests가 같은 prefix cache를 공유할 수 있는지 downstream key를 본다. ingress trace에 template digest를 둔다.

## 22.3 TokenizerManager가 mailbox와 dispatch를 소유한다

OpenAI messages는 chat template가 문자열 또는 token IDs로 compile한다. system/user/assistant/tool role, generation prompt와 special token semantics는 7·8장이 소유한다. 여기서는 serving handler가 template를 적용했는지와 manager가 언제 tokenization하는지 본다.

batch/text tokenize는 [_tokenize_texts](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L891-L994), 단일 request는 [_tokenize_one_request](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L995-L1180)에서 추적한다. rendered bytes digest, tokenizer/template revision, token count와 bounded IDs를 evidence로 남긴다.

```text
messages + sampling params
→ rendered prompt 또는 input_ids
→ rid + normalized sampling params
→ TokenizedGenerateReqInput
→ IPC send
```

context length validation이 input tokens와 max new tokens, server limit을 어떻게 비교하는지 본다. truncation이 있으면 어느 쪽을 자르고 cache identity가 어떻게 변하는지 말한다. option accepted와 실제 IDs 변경을 분리한다.

`TokenizedGenerateReqInput`은 process boundary contract다. field를 추가하면 producer/consumer와 batch wrapper를 함께 바꿔야 한다. tokenized object 생성/send는 [construction path](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L1330-L1660), batch wrapper는 [BatchTokenizedGenerateReqInput](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/io_struct.py#L1055-L1070)에서 본다.

IPC send 완료는 scheduler admission이 아니다. socket enqueue, peer receive, Req 생성, waiting queue admission은 다른 event다. API received, tokenization done, IPC sent, scheduler received, admitted timestamp를 나눈다.

### mailbox를 send 전에 등록하는 이유를 race로 본다

manager가 IPC send를 먼저 하고 state mapping을 나중에 등록한다고 가정하자. 아주 짧은 cached request가 scheduler와 detokenizer를 빠르게 지나 handle loop에 돌아온다. output R17을 받았지만 mapping에 R17이 없어 unknown으로 버린다. 이후 state가 등록되고 HTTP coroutine은 영원히 기다린다.

반대 순서로 state를 먼저 등록하면 이 lost wakeup을 피한다. 대신 IPC serialization/send가 실패할 때 mapping을 제거하고 waiting coroutine에 exception을 전달해야 한다. state 등록 성공을 request dispatch 성공으로 metric에 세면 orphan이 생긴다.

registration, send begin/end, send failure cleanup을 transaction처럼 본다. 실제 DB transaction은 아니지만 invariant는 output producer가 가능해지기 전에 consumer mailbox가 존재하는 것이다. failure path는 mailbox를 terminal error로 완료하고 mapping을 pop한다.

### rid collision은 평균 latency가 아니라 cross-request correctness 문제다

R17이 active인데 새 request가 같은 rid를 사용하면 state mapping assignment가 old mailbox를 덮을 수 있다. old output이 new client로 가거나 new output이 old generator로 갈 수 있다. UUID 사용은 collision 확률을 낮추지만 explicit user-provided rid, restart와 test counter를 확인한다.

rid uniqueness scope가 process lifetime인지 cluster-wide인지, P/D stages에서 같은 rid를 공유하는지 기록한다. numeric rid가 재사용될 수 있으면 incarnation/generation을 IPC message와 output에 포함한다. output batch가 rid만 가진다면 reuse 전 old pipeline drain을 보장해야 한다.

stale abort R17도 같은 문제다. old client disconnect task가 늦게 실행되어 new R17 scheduler Req를 abort할 수 있다. abort message가 incarnation을 갖거나 rid reuse를 금지한다. cleanup task의 cancellation과 join을 본다.

### batch input이 여러 rid로 갈리는 경우

한 HTTP request가 prompts 세 개 또는 n=2 generations를 요청하면 TokenizerManager가 여러 tokenized objects와 rid를 만들 수 있다. client response는 하나지만 scheduler requests는 여럿이다. 하나가 validation/tokenization에 실패했을 때 all-or-nothing인지 partial result인지 API contract를 정한다.

batch wrapper가 objects를 한 IPC message로 묶어 serialization overhead를 줄일 수 있다. scheduler는 각 item을 Req로 만든다. batch message receive success와 각 Req admission success를 나눈다. abort HTTP request는 child rids 모두를 찾아야 한다.

output은 child rid와 choice index를 원 OpenAI choices order로 재조립한다. completion order가 input order와 다를 수 있다. stream이면 choice index별 delta/finish를 보낸다. rid→parent request→choice mapping을 state에 둔다.

child A는 끝났고 B는 running, C는 waiting일 때 client disconnect가 오면 남은 B/C를 abort하고 A output mailbox도 cleanup한다. 이미 delivered된 A를 abort metric에 세지 않는다. parent와 child lifecycle을 분리한다.

### queue를 세 곳으로 나눈다

첫째 API/tokenizer work queue다. CPU workers가 template/tokenization/multimodal preprocess를 기다린다. 둘째 IPC/socket queue다. tokenized messages가 scheduler receiver를 기다린다. 셋째 scheduler waiting queue다. Req가 token/KV budget과 policy를 기다린다. 모두 TTFT를 늘리지만 owner와 해결책이 다르다.

tokenizer queue가 크면 CPU parallelism, tokenizer implementation, long input distribution을 본다. IPC queue가 크면 receiver throughput, serialization, control message와 backpressure를 본다. scheduler queue가 크면 GPU capacity와 scheduling/admission을 본다. queue length만 비교하지 않고 arrival/service rate와 age를 본다.

네 번째로 output mailbox/client queue가 있다. model은 빠르지만 client가 느리면 output pending이 쌓인다. 이 queue는 ingress queue와 반대 방향이다. 같은 `queue_size` metric 이름을 쓰지 않고 stage를 붙인다.

대기열 평균 관계 `L≈λW`를 적용할 때 안정 상태와 stage별 arrival을 확인한다. tokenizer queue L=100, λ=200 req/s면 평균 wait 약 0.5초라는 직관을 얻지만 heavy-tail과 batching, drop은 별도다. 정확한 p99를 평균식으로 예측하지 않는다.

### backpressure가 없다면 pending state가 memory queue가 된다

API가 초당 1000 request를 받아 tokenization하고 scheduler가 500만 처리하면 초당 500 pending이 늘어난다. socket send가 무한 buffer처럼 보이면 HTTP handler는 빠르게 accepted 상태가 되고 memory/latency가 폭발한다. admission limit 또는 bounded channel이 필요하다.

bounded send가 await로 block하면 request deadline과 client disconnect를 감지해야 한다. send 대기 중 abort는 scheduler에 Req가 아직 없으므로 local state만 취소할 수 있다. send가 막혔다 풀린 직후 stale request를 보내는 race를 막는다.

drop policy가 있다면 어떤 request를 거부하는지와 client status를 명시한다. 이미 state 등록/비용 accounting을 시작했다면 cleanup한다. silent drop은 HTTP coroutine hang으로 나타난다.

scheduler control messages가 generate flood 뒤에 갇히면 abort/health/update가 늦어진다. receiver priority 또는 별도 channel이 있는지 source를 본다. priority가 없다면 운영 limitation을 측정한다.

### option을 queue와 state 변화로 읽는다

tokenizer worker 수를 늘리면 CPU service capacity와 동시에 처리할 preprocessing이 늘어난다. 각 worker가 tokenizer/model processor memory를 복제하는지, request order와 state mapping이 router에서 어떻게 유지되는지, worker crash scope가 달라진다. 숫자 하나를 올리면 항상 TTFT가 줄지 않는다. contention과 IPC overhead가 생길 수 있다.

skip-tokenizer-init 또는 input IDs direct mode가 있다면 startup/memory는 줄 수 있지만 text API capability와 detokenization owner가 달라진다. raw strings를 받을 수 없는지, output text를 누가 만드는지, OpenAI chat route가 지원되는지 effective validation을 본다. tokenizer가 없는데 chat template를 기대하면 startup보다 request에서 늦게 실패할 수 있다.

max request/input length는 validation/admission과 actual IDs shape를 바꾼다. truncate 옵션은 rendered prompt와 cache key, logprobs alignment를 바꾼다. reject는 CPU/GPU work 전에 끝낼 수 있지만 client contract를 바꾼다. field→validation branch→token IDs→scheduler effect를 쓴다.

stream interval 또는 output coalescing 설정은 model token 생성 수학을 바꾸지 않고 manager/detokenizer가 client event를 내보내는 빈도와 queue load를 바꾼다. interval을 키우면 chunk overhead는 줄지만 user ITL과 cancel detection/delivery가 늦어질 수 있다. usage/finish event가 정확히 한 번 나오는지 본다.

IPC channel high-water/buffer 설정은 producer blocking과 memory를 바꾼다. buffer를 크게 하면 burst를 흡수하지만 overload를 늦게 드러내고 pending latency가 커진다. 작게 하면 early backpressure가 오지만 abort/control message와 fairness를 설계해야 한다.

### queue saturation을 client error로 번역한다

tokenizer/IPC pending cap에 도달하면 request를 429/503 등으로 거부할 수 있다. error가 retryable인지 `Retry-After`와 admission source를 명시한다. client가 즉시 retry해 overload를 악화시키지 않게 한다.

이미 body parse와 multimodal download를 끝낸 뒤 reject하면 CPU 비용을 썼다. 가능한 early token/queue estimate와 exact post-tokenization check를 나눈다. early estimate가 틀릴 수 있어 conservative headroom을 둔다.

priority request가 일반 queue를 앞설 수 있지만 starvation과 tenant fairness가 생긴다. ingress priority가 scheduler priority로 전달되는지, 한쪽만 적용되는지 본다. field를 받아도 consumer가 무시하면 effective state가 아니다.

**tokenizer throughput 실험을 설계한다**

workload는 raw text length와 resulting token length를 독립 축으로 둔다. ASCII repetition, Korean/Unicode, code/JSON/tool schema가 chars/token ratio와 pretokenization cost가 다르다. chat messages count와 template complexity도 둔다. 실제 사용자 text를 그대로 수집하지 않고 synthetic cohort를 만든다.

worker count 1,2,4를 바꾸며 throughput, p50/p99 queue/service, CPU/memory, IPC bytes와 downstream idle을 본다. arrival은 open-loop로 주어 saturation을 찾고 closed-loop만으로 queue를 숨기지 않는다. warm tokenizer/cache state를 고정한다.

결과가 worker 2 이후 늘지 않으면 GIL/native parallelism, memory bandwidth, shared lock, IPC receiver를 경쟁 가설로 둔다. CPU utilization 합계만으로 원인을 확정하지 않는다. per-stage profile/source path를 본다.

worker 증가로 median은 좋아지고 p99가 나빠질 수 있다. large request가 multiple workers memory를 압박하거나 OS scheduling이 흔들릴 수 있다. length cohort별로 본다. GPU model을 실행하지 않는 이 집필에서는 experiment contract만 남기고 결과를 주장하지 않는다.

**IPC serialization failure를 설계한다**

TokenizedGenerateReqInput에 serialize할 수 없는 custom object가 들어가거나 oversized message가 생긴다고 하자. state register 후 send에서 exception이 난다. expected는 scheduler Req 없음, manager state terminal error/cleanup, client error 한 번, abort message 불필요 또는 안전한 no-op다.

send가 부분 성공했는지 알 수 없는 transport failure는 어렵다. at-most-once dispatch를 위해 message ID/dedupe가 있는지, retry가 duplicate Req를 만들 수 있는지 본다. 무조건 resend하지 않는다. receiver acknowledgement가 있으면 state를 나눈다.

batch message에서 item 하나 serialization failure가 전체 batch를 막는지 확인한다. producer가 batch를 만들기 전 item validate하거나 batch failure를 parent requests에 모두 배달한다. 일부 items만 scheduler에 갔다면 child-level dispatch evidence가 필요하다.

## 22.4 scheduler가 message를 받고 Req로 바꾼다
### source walk를 producer와 consumer 양쪽에서 닫는다

첫 field는 `rid`다. GenerateReqInput에서 언제 생성/검증되는지, TokenizedGenerateReqInput constructor가 그대로 복사하는지, batch wrapper가 순서를 보존하는지 본다. scheduler handler가 Req.rid로 옮기고 output batch rids가 다시 반환되는 edge까지 닫는다. 한 파일에서 검색 결과가 많다는 것은 증거가 아니라 이 edge 각각의 owner를 찾는 출발점이다.

둘째는 `input_ids`다. template/render와 tokenize producer, context/truncation validation, IPC serialization, Req.origin_input_ids와 prefix/cache consumer를 잇는다. mutable list가 send 뒤 변경될 가능성과 serialization copy timing을 본다. prompt logprobs나 echo가 original IDs를 필요로 하면 truncated/current IDs와 구분한다.

셋째는 sampling parameters다. API field가 normalized struct로 바뀌고 scheduler Req/sampler가 소비한다. temperature/top-k/p, max new tokens, stop IDs/strings, grammar가 서로 다른 owner를 가질 수 있다. ingress validation이 끝났다고 sampler behavior가 증명되는 것은 아니지만 row와 request mapping은 ingress가 보존해야 한다.

넷째는 multimodal state다. raw URLs/content, processor output, placeholder IDs, grid/feature handle, modality token count가 message를 건넌다. large tensor가 pickle/copy되는지 shared handle인지 확인한다. cleanup owner와 security boundary를 붙인다.

다섯째는 timing/trace context다. API server stage, tokenizer stage, scheduler stage가 같은 clock/unit과 request identity로 합쳐지는지 본다. object serialization에서 trace context가 빠지면 spans가 분리된다. timestamp가 None인 path와 observability disabled path도 request semantic을 바꾸면 안 된다.

여섯째는 output state다. detokenizer BatchStrOutput/BatchTokenIDOutput의 rids, delta/cumulative fields, finish reasons와 usage를 manager state가 어떻게 소비하는지 본다. state.finished를 set한 뒤 mailbox notification과 mapping pop 순서를 확인한다. consumer가 마지막 event를 읽기 전에 state를 삭제해도 queue reference가 살아 있는지 contract를 본다.

일곱째는 abort다. API/task가 AbortReq를 만들고 scheduler handler가 waiting/running state를 변경하며 finish/cleanup output이 돌아오는지 잇는다. manager가 local state만 지우고 output을 무시하는 path와 scheduler acknowledgement를 기다리는 path를 구분한다.



[SchedulerRequestReceiver](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler_components/request_receiver.py#L49-L298)는 tokenized generate/batch/control message의 ingress owner다. scheduler [message dispatch](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L1510-L1560)가 handler를 고른다.

[handle_generate_request](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L2370-L2510)는 message를 scheduler [Req](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_batch.py#L803-L980)로 만든다. origin IDs, output IDs, sampling state, cache/prefix와 finish/scheduling metadata가 실행 수명을 얻는다.

Req 생성은 admission이 아니다. waiting queue에서 policy와 memory를 기다린다. tokenizer latency와 scheduler queue latency를 합쳐 TTFT로만 보면 ingress 병목을 못 찾는다. request receiver backlog, IPC high-water mark와 producer backpressure를 본다.

rid와 batch index, trace/timing context가 message→Req로 이동한다. duplicate rid가 active state와 충돌하면 output 귀속이 깨진다. manager state와 scheduler Req는 같은 object가 아니다. scheduler는 GPU/cache state, manager는 client mailbox를 소유한다. 한쪽 cleanup이 다른 쪽 cleanup을 자동 의미하지 않는다.

**scheduler rejection이 HTTP에 돌아오는 경로**

Req가 context/KV/admission 조건에서 reject되면 scheduler가 finish/error output을 rid와 함께 보낸다. manager demux가 state를 찾고 serving adapter가 non-stream status 또는 stream error로 번역한다. reject가 output 없이 local drop되면 HTTP hang이 된다.

error owner를 scheduler admission과 model execution으로 나눈다. retryability와 status도 다를 수 있다. invalid request는 4xx, overload는 429/503 등 contract를 따른다. raw scheduler exception을 그대로 노출하지 않는다.

tokenization preflight와 scheduler exact validation이 중복될 수 있다. 둘의 limit/config가 다르면 ingress에서는 통과하고 scheduler에서 reject되거나 반대가 된다. effective model length와 truncation policy를 한 source of truth에 맞추거나 version을 trace에 남긴다.

**request pause/continue 같은 control도 동일 identity를 쓴다**

SGLang io structs에는 generation pause/continue와 여러 control messages가 있을 수 있다. 이 장의 대표 path는 generate/abort지만 control이 scheduler receiver channel을 공유하면 ordering과 starvation에 영향을 준다. rid/incarnation과 state transition을 동일하게 유지한다.

pause는 client disconnect와 다르다. request state와 KV를 유지하며 scheduling만 중단할 수 있다. abort는 reclaim/terminal이다. 두 message를 같은 cancel flag로 처리하지 않는다. continue가 old incarnation을 살리지 않게 한다.

control endpoint authorization과 audit도 필요하다. abort-all/pause-all이 public user에게 노출되면 다른 tenant requests를 건드릴 수 있다. HTTP route와 manager method의 trust boundary를 본다.

## 22.5 output loop가 원 HTTP stream을 복원한다

scheduler/runner output은 detokenizer를 거쳐 문자열 delta 또는 token batch로 manager에 온다. [handle_loop](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L2174-L2215)는 receive socket을 읽고 output type을 dispatch한다.

[_handle_batch_output](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L2216-L2510)은 batch rids를 순회해 state를 찾고 delta text/IDs, finish reason, usage, logprobs와 timing을 조립한다. 한 batch output이 여러 HTTP generators로 갈라진다.

output이 cumulative text인지 delta인지 확인한다. previous length를 state에 저장해 delta를 자를 수 있고 UTF-8/token decode boundary 때문에 token 하나가 즉시 text를 만들지 않을 수 있다. OpenAI layer는 manager output을 SSE로 변환한다. [completion streaming](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_completions.py#L180-L380)을 manager output과 연결한다.

handle loop는 한 느린 client의 socket write를 직접 기다리면 안 된다. rid mailbox로 dispatch하고 consumer가 따로 send해야 head-of-line blocking을 피한다. queue가 unbounded면 slow client memory가 늘고 bounded이면 backpressure/coalescing/abort policy가 필요하다. gRPC의 [chunk backpressure](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/grpc_bridge.py#L100-L160)는 이 문제를 드러낸다.

### output demux가 빠르게 다음 receive로 돌아가야 하는 이유

detokenizer batch가 64 rid를 가진다. handle loop가 각 rid의 client JSON serialization과 network send까지 수행하면 느린 한 request가 나머지 63개의 dispatch를 지연한다. manager는 mailbox event를 enqueue하고 다음 socket receive로 돌아가는 것이 자연스럽다.

mailbox put 자체가 bounded queue full로 block할 수 있다. nonblocking put+policy, per-request task, shared backpressure 중 무엇인지 source를 본다. unbounded이면 memory cap/slow-client abort를 다른 층에서 둔다. 모든 선택에는 대가가 있다.

batch output processing에서 meta_info/logprobs/hidden states가 크면 Python CPU 시간이 커질 수 있다. logprobs 요청 cohort와 normal stream을 분리 측정한다. output hidden states 같은 debug field는 reference lifetime과 serialization을 늘린다.

manager receive rate, batch size, dispatch duration, unknown rid, mailbox depth를 관측한다. client write latency는 route/transport에 둔다. 두 timestamp를 이어 provider/model ITL과 user ITL을 나눈다.

### delta text와 token commit을 혼동하지 않는다

token ID 하나가 detokenizer buffer에서 아직 완전한 Unicode text를 만들지 못할 수 있다. manager output_ids는 진행했지만 delta_text는 빈 문자열일 수 있다. empty delta를 finish로 해석하면 안 된다. 반대로 special/stop token은 ID가 있지만 client text에 emit되지 않을 수 있다.

usage completion_tokens는 committed model tokens인지 emitted text chunks인지 source를 확인한다. speculative rejected tokens는 client completion token이 아니다. reasoning token과 cached prompt token도 별도 field다.

stream chunk count를 token count로 쓰지 않는다. coalescing과 tool-call fragments, heartbeat가 다르다. observability에는 generated/accepted IDs와 emitted bytes/chunks를 다른 metric으로 둔다.

**사건 A — HTTP 200인데 빈 stream으로 끝난다.**

route가 StreamingResponse를 시작했지만 TokenizerManager tokenization exception이 generator 첫 iteration에서 난다. headers는 이미 200으로 commit되었고 표준 error event 없이 connection이 닫힌다. client는 success status와 empty body를 본다.

validation/template/tokenization을 response commit 전에 가능한 범위까지 완료하는지 본다. generator creation과 first iteration 중 실제 work 위치를 확인한다. preflight가 너무 무거우면 TTFT와 event loop를 막지만 error contract는 선명해진다.

stream protocol이 error chunk를 지원하면 post-commit exception을 표준 event로 변환한다. 그렇지 않으면 connection close와 trace finish reason을 기록한다. HTTP status success와 stream success를 별도 metric으로 둔다.

**사건 D — scheduler는 output을 보냈는데 HTTP가 hang한다.**

detokenizer output batch에 rid가 있고 handle loop도 받았지만 state mapping에서 찾지 못하거나 mailbox notification이 누락됐다. registration race, premature cleanup, rid mismatch와 handle loop task death를 본다. scheduler/model을 재실행할 필요가 없다.

unknown rid counter와 output batch safe digest, state lifecycle events를 보존한다. handle loop exception이 task를 종료했는데 server health는 HTTP process alive로 green일 수 있다. background loop liveness와 last receive timestamp를 health에 연결한다.

mapping은 있는데 consumer event wait가 깨어나지 않으면 queue/event API 순서와 cancellation을 본다. finished flag만 set하고 notify를 빼거나 consumer가 old event object를 기다릴 수 있다. 작은 unit state machine으로 검증한다.

**사건 E — batch choices 순서가 때때로 바뀐다.**

세 child rid가 scheduler에서 B,C,A 순으로 완료되고 serving adapter가 completion order로 choices를 append하면 API input order A,B,C와 달라진다. choice index를 rid state에 보존해 final response에서 원 order로 정렬한다. stream은 index field로 interleave할 수 있다.

batch wrapper index와 scheduler output batch index는 같은 개념이 아니다. scheduler batch는 매 step 재편된다. origin choice index를 request metadata로 유지한다. rid numeric order로 정렬하지 않는다.

한 child가 error/abort면 parent contract가 partial choices를 허용하는지 결정한다. 모든 child를 abort할지 error choice를 넣을지 endpoint semantics를 따른다. cleanup과 usage aggregation도 parent/child로 나눈다.

**사건 F — Unicode output이 중복되거나 깨진다.**

detokenizer가 cumulative output string을 보내는데 manager가 delta라고 그대로 emit하면 앞 text가 매 chunk 중복된다. 반대로 delta를 cumulative라 생각하고 previous length로 자르면 문자가 빠진다. BatchStrOutput field contract와 state counter를 확인한다.

UTF-8 multibyte와 byte fallback token이 chunk boundary를 가를 수 있다. Python string으로 이미 valid text인지 raw bytes fragment인지 owner를 본다. replacement character가 한 번 emit되면 나중에 복구할 수 없을 수 있다. detokenizer buffer가 incomplete bytes를 보존한다.

token IDs stream과 text stream을 동시에 제공하면 둘의 delta length가 1:1이 아닐 수 있다. stop/special token도 다르다. exact IDs와 emitted bytes를 별도 checkpoint로 둔다.

### output finish reason이 mailbox를 닫는 방식

stop, length, abort, error 등 finish reason이 output batch item에 있다. manager는 `state.finished=True`로 만들고 final event를 consumer에게 준다. mapping을 즉시 pop할지 consumer acknowledgement 뒤 pop할지 lifetime을 본다.

immediate pop 후 late duplicate output이 오면 unknown rid로 무시된다. consumer가 final event를 아직 queue에 갖고 있으면 괜찮다. 하지만 event object가 state 안에만 있고 mapping pop과 함께 reference가 사라지면 hang할 수 있다. data structure contract를 확인한다.

finish reason이 None인 chunk와 final chunk 순서가 IPC에서 보존되는지, multiple producer가 있는지 본다. ZMQ/channel ordering은 socket/peer별 보장 범위가 있다. P/D나 multiple detokenizer가 같은 rid output을 보낼 때 sequence index가 필요할 수 있다.

### error를 output으로 돌려보낼지 exception으로 완성할지

scheduler가 invalid grammar 또는 model error를 finish reason/error object로 보내면 manager가 mailbox event로 client handler에 전달한다. tokenizer 단계 exception은 local coroutine에서 바로 raise될 수 있다. 두 error path가 OpenAI 표준 error와 status/stream event로 일관되게 변환되는지 본다.

non-stream은 아직 response commit 전이면 status code를 선택할 수 있다. stream은 이미 200일 수 있어 error chunk/finish와 close를 쓴다. 같은 underlying abort가 API mode에 따라 wire 표현이 다르지만 metrics category는 같아야 한다.

exception text에 prompt/tool schema가 들어갈 수 있어 redaction한다. scheduler traceback 전체를 client에 보내지 않고 trace ID를 준다. 운영 evidence에는 fixed revision과 safe stage/category를 남긴다.

**사건 I — stop token 뒤 output loop가 state를 일찍 지운다.**

scheduler final output과 usage/logprobs metadata가 별 batch/event로 올 수 있는 구조에서 첫 finish reason만 보고 state를 pop하면 뒤 metadata가 unknown rid로 버려질 수 있다. output protocol이 final event에 모든 fields를 atomic하게 포함하는지 확인한다.

OpenAI stream은 final finish chunk와 optional usage chunk 순서를 요구할 수 있다. mailbox lifetime이 usage event까지 이어져야 한다. disconnect가 finish 직후 오면 background abort가 이미 completed state에 안전해야 한다.

**사건 K — output queue가 unbounded라 process가 OOM한다.**

client 1000개가 느리고 각 mailbox에 10 MiB logprobs/hidden data가 쌓이면 10 GiB가 된다. GPU memory는 정상인데 API process OOM이다. output queue depth/bytes와 request options를 본다.

bounded queue와 slow-client timeout/abort를 두고 large debug fields를 streaming에서 제한한다. backpressure가 handle loop 전체를 막지 않게 per-request 정책을 쓴다. aborted model work와 memory release를 연결한다.

**output backpressure 실험을 설계한다**

fast client와 인위적으로 느린 client를 섞고 distinct rids를 사용한다. scheduler/detokenizer output fixture를 mock해 manager demux와 HTTP write만 검증할 수 있다. slow client mailbox depth가 늘어도 fast client dispatch latency가 유지되어야 한다.

bounded queue가 full일 때 expected policy를 정한다. slow request만 abort하는지, producer를 block하는지, chunks를 coalesce하는지 확인한다. logprobs/tool-call fragments는 arbitrary coalescing이 semantic을 깨뜨릴 수 있다. policy별 test를 둔다.

disconnect를 queue full 직전/후에 주어 abort와 mailbox cleanup race를 검증한다. queue item references가 release되고 pending gauge가 내려가는지 본다. scheduler abort acknowledgement가 늦어도 API memory는 bounded해야 한다.

## 22.6 abort·disconnect·restart를 원자적으로 닫는다

client는 rid 할당 전, state 등록 후 send 전, IPC 뒤 admission 전, running 중 어느 때든 끊길 수 있다. abort를 한 번 enqueue하고 끝내면 dispatch 뒤 생긴 Req가 살아남을 수 있다. 정상 finish 뒤 stale abort가 같은 numeric rid의 새 incarnation을 죽여서도 안 된다.

[abort_request](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L1991-L2045), [create_abort_task](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L2160-L2174), IPC [AbortReq](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/io_struct.py#L2005-L2018)을 하나의 handshake로 읽는다.

state는 cancel requested, abort enqueued, scheduler removed/finished, output mailbox cleaned로 나눈다. abort API return은 enqueue만 의미할 수 있다. running batch가 token을 더 실행할 수 있고 client delivery가 끝난 뒤 compute가 남을 수 있다.

마지막 output과 disconnect가 동시에 오면 finish와 abort task가 state를 둘 다 제거한다. cleanup은 idempotent해야 하고 abort/success metric을 이중 count하지 않는다. unknown rid output과 이미 정상 정리된 rid를 구분한다.

### abort message가 scheduler에서 처리되는 세 위치

Req가 아직 receiver queue에 없다면 abort가 먼저 도착할 수 있다. scheduler가 tombstone/abort set을 보관해 이후 동일 rid request를 reject하는지, 순서 보장 channel을 사용하는지 확인한다. 그렇지 않으면 early abort가 no-op이고 request가 나중에 실행된다.

Req가 waiting이면 queue에서 제거하고 allocated prefix/cache refs를 release한다. running이면 current batch completion boundary에서 mark finished/abort하고 KV/request slots를 해제한다. output/finish event를 manager로 보내 mailbox를 닫을 수 있다.

Req가 이미 finished이면 abort는 idempotent no-op이어야 한다. metrics가 abort로 바뀌지 않는다. same rid incarnation 문제를 피한다. scheduler handler source에서 waiting/running/finished search order를 본다.

abort-all은 개별 rid와 위험이 다르다. admin shutdown/update 용도라면 active requests snapshot과 new admission gate를 조정해야 한다. abort-all 메시지 뒤 새 request가 들어오는 race를 정의한다.

### disconnect detection은 polling과 event timing을 가진다

HTTP framework의 `is_disconnected`를 polling하면 poll interval만큼 abort가 늦을 수 있다. background task가 response 종료 뒤 실행되면 정상 finish에도 abort 호출이 생길 수 있다. generator cancellation/finally와 background response task의 역할을 source에서 구분한다.

client TCP disconnect가 proxy/load balancer를 거쳐 application에 늦게 전달될 수 있다. app timestamp만으로 user disconnect 정확 시점을 모를 수 있다. front proxy trace와 correlation한다. downstream compute가 계속된 시간을 upper bound로 기록한다.

disconnect 뒤 client가 retry하면 old/new requests가 동시에 실행할 수 있다. idempotency 또는 operation ID가 없으면 둘은 독립이다. old request abort가 늦어 new rid에 영향을 주지 않게 한다.

**사건 C — abort metric은 늘지만 GPU work가 줄지 않는다.**

client disconnect detector가 abort_request를 자주 호출해 manager metric은 증가한다. 그러나 scheduler receiver가 generate flood 뒤 AbortReq를 늦게 받거나 abort가 rid dispatch 전에 도착해 no-op일 수 있다. abort count와 removal latency, executed tokens after cancel을 비교한다.

manager-side `abort enqueued`를 완료 metric으로 이름 붙이지 않는다. scheduler `Req marked/removed`, KV release, last executed token을 연결한다. control channel priority 또는 dispatch tombstone이 필요할 수 있다.

fix 후 cancel-to-removal histogram과 wasted scheduled/executed tokens가 줄어야 한다. client disconnect rate 자체는 network 특성이므로 그대로일 수 있다. 잘못된 metric 개선을 요구하지 않는다.

### health check가 handle loop와 worker liveness를 포함해야 하는 이유

HTTP process와 route는 살아 있지만 TokenizerManager handle loop task가 exception으로 종료되면 새 requests가 IPC로 가고 scheduler는 output을 보내도 client는 영원히 기다린다. 단순 `/health` 200은 거짓 green이다.

handle loop task alive, last receive/dispatch timestamp, pending mailbox oldest age, scheduler channel connectivity를 health/readiness에 연결한다. tokenizer workers registration/heartbeat와 queue도 본다. GPU runner health만으로 ingress를 증명하지 않는다.

readiness failure에서 새 requests를 받지 않되 existing requests cleanup/abort를 시도한다. process restart가 quickest recovery여도 pending client responses와 scheduler Req를 어떻게 끝내는지 문서화한다. load balancer가 retry해 duplicate work를 만들 수 있다.

### graceful shutdown의 request ownership

server shutdown이 시작되면 new admission을 닫고 tokenizing/pending/running requests를 drain 또는 abort한다. TokenizerManager handle loop를 scheduler/detokenizer output이 모두 끝나기 전에 취소하면 final events와 usage가 사라진다.

drain deadline을 정하고 이후 abort-all을 보낼 수 있다. IPC sockets를 close하는 순서, background abort tasks와 worker pools join을 본다. shutdown exception을 provider/model error로 count하지 않는다.

deployment rollout에서 old/new process가 같은 external rid namespace를 공유하는지와 client connection drain을 본다. old request output이 new manager로 라우팅되지 않도록 socket topology를 고정한다.

**cancellation race를 시간 순서 네 개로 검산한다**

순서 A는 disconnect→local cancel→rid assignment다. abort task가 rid None에서 끝나면 이후 dispatch가 살아난다. dispatch completion을 기다리거나 cancelled flag를 tokenization path가 확인해야 한다.

순서 B는 rid assignment/state register→disconnect→IPC send다. local state cancelled면 send를 중단하고 cleanup한다. send가 이미 in-flight라면 scheduler abort도 enqueue한다.

순서 C는 IPC receive/Req waiting→disconnect다. AbortReq가 waiting queue에서 제거하고 finish/ack를 돌린다. prefix refs와 metrics를 release한다. 순서 D는 running/output race다. last token output과 abort가 교차할 때 정확히 하나의 terminal outcome을 고른다.

각 순서에 expected manager mapping, scheduler Req existence, KV allocation, client bytes, terminal metric을 표 대신 서술형 state test로 쓴다. 한 fixture가 네 race를 모두 증명하지 않는다.

**장말 종합 사건: restart와 disconnect가 겹쳤다**

tokenizer worker가 restart되는 순간 client R17이 disconnect했다. old worker는 tokenization을 끝냈지만 result delivery가 늦었다. manager abort task는 rid가 아직 없어 local cancel만 했다. new worker가 result를 받아 rid R17을 만들고 IPC로 보내 scheduler가 실행했다. client는 없고 GPU work와 mailbox state가 남았다.

first divergence는 disconnect state와 dispatch eligibility 사이이다. tokenization future/GenerateReqInput에 cancellation/incarnation flag를 유지하고 result completion 직전에 확인해야 한다. rid assignment 후에는 scheduler abort도 보낸다. old worker result가 current request state에 속하는지 attempt ID로 검증한다.

source evidence는 abort task timing, rid assignment, worker result receive와 send path다. runtime evidence는 disconnect, worker generation, state registered, IPC sent와 scheduler received timestamps다. worker restart 자체를 원인으로 끝내지 않고 stale completion이 dispatch된 edge를 찾는다.

수정 후 disconnect가 tokenization 전/중/완료 직후, worker restart 유무 네 boundary에서 scheduler Req가 생기지 않거나 생겼다면 bounded abort되는지 검증한다. pending mailbox와 worker future, IPC message, scheduler/KV state가 모두 회수되어야 한다.

이 사건은 ingress의 본질을 압축한다. CPU preprocessing은 pure function처럼 보이지만 결과를 scheduler로 보내는 순간 side effect가 된다. cancellation은 HTTP task만 끝내는 것이 아니라 아직 미래에 도착할 preprocessing result의 dispatch 권한도 취소해야 한다.

## 22.7 overlap·P/D·speculative가 수명에 더하는 상태

긴 text, tool schema와 template 처리는 CPU queue를 만든다. GPU utilization과 scheduler waiting이 낮은데 TTFT가 높다면 API accepted→tokenization done 구간과 worker backlog를 본다. malformed Unicode/worker exception에서 IPC가 보내졌는지, pending state가 정리되고 client error가 반환되는지 확인한다.

multimodal은 image/video/audio fetch/decode/resize, placeholder와 grid state를 만든다. manager의 [multimodal receive path](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L1040-L1110)를 tokenized message와 잇는다. text IDs가 같아도 grid/mRoPE metadata가 다르면 output이 갈린다.

worker timeout을 API가 retry하면 첫 worker가 늦게 완료해 duplicate IPC message를 보낼 수 있다. tokenization은 pure해 보여도 dispatch side effect와 결합되면 attempt/incarnation이 필요하다. restart가 완료되어도 old futures가 자동 완료되지는 않는다.


overlap scheduling은 CPU가 다음 batch를 준비하는 동안 GPU가 현재 batch를 실행한다. tokenized request→Req 의미는 유지되지만 buffer lifetime과 abort 위치가 늘어난다. future batch와 current running batch 양쪽에서 rid를 찾는다.

P/D 분리는 prefill/decode stage identity와 KV transfer handle를 추가한다. HTTP mailbox는 하나지만 request는 prefill scheduler와 decode scheduler를 건넌다. rid, logical position과 transfer state가 이어져야 한다. 자세한 transfer는 60~65장이 맡는다.

speculative는 proposed/verified/accepted metrics와 여러 accepted token delta를 output에 넣는다. manager의 [spec metrics](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L2760-L2805)를 client-committed token과 나눈다. rejected token은 emit하지 않는다.

특수 mode부터 설명하지 않는 이유는 모든 mode의 공통 invariant가 rid, tokenized input, IPC ownership, scheduler admission, output demux와 abort cleanup이기 때문이다. 특수 mode는 handoff를 더할 뿐 기준 수명을 없애지 않는다.

### overlap mode에서 stale buffer를 찾는다

CPU가 step s+1 metadata를 준비하는 동안 GPU가 s를 실행한다. R17 abort가 오면 current s output, prepared s+1 input과 scheduler state 모두에서 제거되어야 한다. future buffer가 이미 compact/permuted됐다면 row mapping을 갱신한다.

abort를 current batch에만 mark하면 next buffer가 R17을 다시 실행한다. future에만 제거하면 current output이 mailbox에 도착할 수 있다. request incarnation과 step generation을 buffer metadata에 둔다.

overlap 성능 이득은 CPU preparation과 GPU work를 겹치는 데서 온다. synchronization을 무조건 추가하면 race는 숨지만 이득을 잃는다. event/dependency와 state ownership을 고치고 regression에서 overlap timeline을 본다.

### P/D 분리에서 ingress가 잃기 쉬운 auxiliary state

prefill scheduler가 token IDs와 positions로 KV를 만들고 transfer handle을 decode scheduler에 준다. decode는 selected first token, logical committed length, block/transfer metadata를 받아야 한다. multimodal mRoPE delta와 model-specific state도 필요할 수 있다.

TokenizerManager mailbox는 prefill progress를 client에 직접 emit하지 않을 수 있고 decode output만 받는다. prefill failure, transfer failure와 decode failure를 같은 rid 아래 다른 stage로 기록한다. abort는 두 stages와 transfer manager에 전파된다.

prefill 완료 뒤 decode admission 전에 disconnect가 오면 transferred KV가 orphan될 수 있다. ownership handoff와 abort acknowledgement를 state machine으로 둔다. 60~65장에서 네트워크/transfer를 확대하되 ingress correlation을 유지한다.

### speculative output에서 accepted count를 지킨다

draft가 네 token을 제안하고 target이 두 token을 accept했다. scheduler output은 accepted IDs 두 개와 verification metrics를 보낸다. manager가 proposed count를 completion usage나 output delta에 쓰면 ghost tokens가 생긴다.

output message fields의 `spec_verify_ct`, correct drafts와 actual output IDs를 구분한다. client stream은 accepted order만 emit한다. abort가 verification 중 오면 어느 accepted prefix가 commit됐는지 scheduler finish output을 따른다.

spec metrics는 performance observability이고 client semantic state가 아니다. mailbox state의 cumulative output length는 accepted IDs로 진행한다. detokenizer buffer도 rejected drafts를 보지 않는다.

**사건 B — prompt logprobs request만 tokenizer CPU가 폭증한다.**

prompt logprobs는 input token alignment와 offsets/top logprob fields를 추가 처리할 수 있다. 동일 text generation과 비교해 tokenization/metadata construction, scheduler LM head와 output serialization 어느 구간이 늘었는지 나눈다. API total만 보고 tokenizer를 탓하지 않는다.

manager tokenized object가 flat token_ids_logprob 목록과 per-token fields를 만들고 IPC payload가 커질 수 있다. scheduler output도 큰 logprob arrays를 돌려준다. handle loop Python object construction과 mailbox memory가 늘어난다.

input length sweep에서 tokenize time, IPC bytes, model LM-head, output handling을 각각 측정한다. logprobs top-n과 prompt length를 축으로 둔다. option→field count/shape→CPU/IPC/GPU/output effect를 닫는다.

**사건 G — multimodal fetch가 event loop를 멈춘다.**

remote image download/decode가 synchronous call로 async request loop에서 실행되면 한 large image가 다른 text requests의 route/tokenization을 지연한다. API process CPU profile과 event-loop lag, multimodal stage duration을 본다. model GPU는 idle할 수 있다.

blocking work를 thread/process pool로 옮길 수 있지만 cancellation과 resource limit, context propagation이 필요하다. client disconnect 뒤 decode task가 계속되고 large buffer를 만든다면 worker cancel/cleanup을 본다. pool queue가 새 병목이 된다.

text-only cohort latency가 image load에 따라 나빠지는 cross-traffic fixture를 설계한다. worker concurrency를 무한히 늘리지 않고 decoded memory upper bound를 둔다.

**사건 H — P/D transfer 완료 뒤 response가 다른 request로 간다.**

prefill controller가 rid R17 transfer handle H9를 decode stage에 보낸다. slot reuse로 new R17이 생겼고 old transfer completion이 new mailbox/Req에 붙는다. rid만 correlation한 incarnation bug다.

transfer message에 request incarnation, model/cache identity와 logical length를 둔다. manager state, prefill Req, decode Req, transfer handle의 composite key를 맞춘다. old completion은 drop하고 resources를 release한다.

P/D stage별 trace를 root API request 아래 잇는다. prefill output first token과 decode stream handoff가 한번만 client mailbox로 들어가는지 본다. transfer retry가 duplicate decode admission을 만들지 않는다.

**마지막으로 옵션 변경의 승인선을 그린다**

tokenizer worker를 2에서 8로 늘리는 변경을 생각하자. 예상 state 변화는 동시에 처리하는 CPU requests와 worker-local tokenizer/processor memory, result reorder 폭이 늘어나는 것이다. 기대 효과는 CPU queue 감소다. 경쟁 가설은 IPC receiver나 template single-thread section이 병목이라 worker 증가가 효과 없다는 것이다.

승인 metric은 tokenization queue/service p99, total API TTFT, CPU/memory, IPC backlog와 duplicate/orphan state다. rollback 조건은 memory pressure, worker crash 증가, rid/result mapping 오류다. throughput만 보고 승인하지 않는다. same workload length/modality distribution을 고정한다.

IPC buffer를 키우는 변경은 short burst drop/block을 줄일 수 있지만 overload에서 pending age와 memory를 늘린다. 승인에는 buffer occupancy와 oldest age, scheduler receive rate가 필요하다. queue가 커져 metric이 조용해졌지만 user TTFT가 늘면 개선이 아니다.

stream interval을 1에서 4 token으로 늘리면 output event/serialization은 줄 수 있지만 user-visible ITL과 disconnect 반응이 늦을 수 있다. tool/stop/usage event ordering과 exact token/text를 검증한다. model decode ITL은 그대로일 수 있다. provider/model metric과 client metric을 분리한다.

multimodal worker pool을 분리하면 text requests 격리가 좋아질 수 있지만 image queue와 shared memory lifetime, worker failure domain이 추가된다. text/image cross-traffic와 cancel cleanup을 함께 본다. pool 생성만으로 security/size validation이 해결되지는 않는다.

## 22.8 R17 사건을 종합하고 scheduler로 넘긴다
증상에서 다음 owner로 이동할 때는 방향을 먼저 고른다. 요청이 GPU에 도달하지 않는다면
render/tokenize 완료에서 IPC receive 쪽으로 내려간다. tokenization 전에 멈췄으면
API·template·worker가, send 뒤 receive 전에 멈췄으면 channel과 backpressure가, `Req` 생성 뒤에
멈췄으면 scheduler가 다음 owner다. 반대로 응답이 돌아오지 않는다면 scheduler output rid에서
manager receive·demux, mailbox, serving chunk와 client write를 거슬러 올라간다. model output이
있다는 사실은 HTTP delivery를 증명하지 않고, output stage가 없다는 사실도 곧바로 model
failure를 증명하지 않는다.

취소 사건은 같은 수명을 반대 방향으로 닫는다. rid와 incarnation이 정해진 뒤 abort가
enqueue되고 receiver ordering을 거쳐 waiting/running membership과 resource가 해제되는지
시간축으로 본다. abort 호출 횟수보다 취소 뒤 실행된 token과 최종 state release가 결과다. 이
세 경로를 source와 trace로 연결하면 `TokenizerManager`는 CPU 전처리 helper가 아니라 API와
scheduler 사이에서 입력 의미, identity, backpressure와 output·abort lifetime을 소유하는
manager로 보인다.

**최종 source 지도와 다음 장 handoff**

입구는 HTTP [generate route](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/http_server.py#L894-L935)와 OpenAI [chat route](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/http_server.py#L1700-L1740)다.

요청 수명 주기의 소유자는 TokenizerManager다. [generate](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L765-L890)와 [tokenize](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L891-L1180)를 이어 읽으면 소유권의 범위가 드러난다.

프로세스 경계는 [tokenized message construction](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L1330-L1660)과 스케줄러의 [request receiver](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler_components/request_receiver.py#L49-L298) 사이에 있다.

실행 상태는 [handle_generate_request](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L2370-L2510)와 `Req`에서 시작한다.

결과의 귀환은 매니저 [handle loop](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L2174-L2215)와 [batch output demux](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L2216-L2510)를 따라간다.

취소 경로는 [abort_request](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L1991-L2045)와 `AbortReq`를 연결해 읽는다.

다음 scheduler 장은 여기서 확정된 Req를 받는다. 그 장에서는 waiting/running, token budget, prefix/KV memory와 preemption을 본다. 이 장은 scheduler policy를 미리 설명하지 않고 정확히 어떤 input/state/timestamp가 ingress에서 넘어왔는지 보장한다.

**한 줄 로그를 인과 trace로 바꾸는 법**

`received request R17`만 남기면 어느 stage가 받았는지 모른다. API route, TokenizerManager registration, IPC receiver, scheduler Req creation이 모두 receive라고 말할 수 있다. event 이름에 owner와 transition을 붙인다. `api.validated`, `tokenizer.state_registered`, `tokenizer.ipc_sent`, `scheduler.ipc_received`, `scheduler.req_created`처럼 쓴다.

각 event에는 rid/incarnation, parent API ID와 monotonic timestamp를 넣는다. payload 내용 대신 input chars/tokens, modality count, serialized bytes와 mode를 쓴다. config/model/tokenizer revision은 root span/resource에 둔다. request마다 거대한 중복 label을 넣지 않는다.

output도 `token generated` 하나로 끝내지 않는다. scheduler accepted token, detokenizer batch sent, manager batch received/demuxed, mailbox enqueued, HTTP chunk written을 나눈다. user ITL과 model ITL 차이를 설명할 수 있다.

abort log는 reason과 stage를 붙인다. client disconnect, explicit abort endpoint, timeout, scheduler/model error, shutdown을 구분한다. `abort enqueued`와 `scheduler removed`, `manager cleaned`를 별 event로 둔다. 마지막 terminal outcome은 하나만 count한다.

**Prometheus metric과 trace의 역할을 나눈다**

metric에는 API RPS/error, tokenization duration/queue, IPC send/receive lag, pending manager states, scheduler ingress queue, output dispatch duration, abort/removal count와 lag를 둔다. rid를 label로 넣지 않는다. model group, endpoint mode, modality와 bounded error category 정도를 사용한다.

trace는 개별 rid의 stage timeline과 rare race를 보여 준다. histogram p99가 나빠지면 exemplar trace로 내려간다. logs는 source revision/effective config와 bounded errors를 제공한다. 세 도구가 같은 질문을 반복하지 않고 단계적으로 좁힌다.

pending states gauge가 높을 때 input과 output 방향을 나눈다. scheduler waiting 때문에 오래 pending인지, model running인지, output mailbox/client write 때문인지 age/state breakdown을 본다. 하나의 pending count로 capacity를 판단하지 않는다.

unknown rid output counter는 0이 이상적이지만 정상 cleanup 뒤 late duplicate를 무조건 fatal로 보지 않는다. reason/state history를 sample trace로 연결한다. 지속 증가하면 premature cleanup, collision, duplicate output을 조사한다.

**증거가 없는 설명을 걷어 낸다**

“TokenizerManager가 tokenization을 병렬화한다”는 문장은 구성에 따라 달라질 수 있다. source에서 worker/router mode와 effective args를 확인한 뒤 쓴다. class가 async라고 CPU tokenization이 자동 parallel인 것도 아니다. blocking function이면 event loop를 막을 수 있다.

“abort하면 즉시 GPU memory가 해제된다”는 문장도 걷어 낸다. source가 message enqueue를 보여 줄 뿐 scheduler batch boundary와 KV allocator release timing은 다음 owner다. trace에서 acknowledgement와 release를 봐야 한다.

“stream은 token마다 chunk를 보낸다”도 보편적이지 않다. stream interval, detokenizer buffering과 tool/reasoning protocol이 chunk granularity를 바꾼다. token IDs, text delta와 chunks를 나눈다.

“input IDs를 주면 tokenizer가 필요 없다”도 출력 detokenization과 chat/template/API capability를 누락한다. pretokenized ingress와 output text path를 각각 본다. tokenizer object initialization option이 어느 endpoint를 disable하는지 확인한다.

**사건 증거 묶음은 최소화한다.**

필수 evidence는 SGLang revision/config, endpoint/request mode, safe rendered/token IDs digest와 lengths, rid/incarnation, stage timestamps, IPC message type, scheduler Req state, output/abort terminal events다. multimodal이면 grid/placeholder/handle digest를 추가한다.

raw prompt/image와 full logprobs/hidden을 기본 포함하지 않는다. synthetic fixture로 대체하고 bounded slice를 승인받아 사용한다. source link는 각 stage가 어떤 field를 생산/소비하는지 증명한다.

negative evidence도 남긴다. tokenization done이 빠르고 IPC received도 빠르면 CPU ingress 병목 가능성이 낮다. scheduler output은 있는데 manager state가 없으면 model correctness보다 output demux를 본다. rid/state가 맞고 client write만 늦으면 transport다.

**first divergence matrix를 말로 읽는다**

HTTP adapted request부터 다르면 protocol/default/template owner다. rendered text는 같고 IDs가 다르면 tokenizer revision/options다. IDs는 같고 TokenizedGenerateReqInput이 다르면 sampling/multimodal/logprob normalization이다. message는 같고 Req부터 다르면 scheduler ingress conversion이다.

Req는 같고 scheduler model output부터 다르면 다음 장 이하로 내려간다. output batch는 맞고 manager event가 다르면 demux/delta/usage owner다. manager event는 맞고 SSE/text가 다르면 OpenAI serving/transport다. selected IDs는 같고 visible text만 다르면 detokenization/output buffer다.

이 matrix의 가치는 모든 stage를 항상 dump하는 데 있지 않다. coarse digest와 timestamps로 first divergence interval을 찾고 그 경계만 bounded하게 확대한다. 개인정보와 observer overhead를 줄인다.

**수직 요청 trace를 독자 스스로 복원한다**

source를 열어 HTTP chat route에서 serving handler call을 찾는다. handler가 만드는 GenerateReqInput field 중 model/input/sampling/stream/rid를 기록한다. TokenizerManager.generate_request caller와 yield/return contract를 읽는다.

manager 안에서 tokenization function, state mapping registration, tokenized object constructor와 IPC send를 순서대로 표시한다. exception/finally와 abort task를 옆에 둔다. scheduler request receiver가 message type을 어디서 dispatch하고 Req constructor에 어떤 fields를 넘기는지 잇는다.

귀환은 scheduler output struct producer까지 깊게 갈 필요 없이 detokenizer→manager receive socket부터 시작할 수 있다. handle_loop batch output rid loop, state update/mailbox notification, serving generator의 chunk conversion과 final cleanup을 잇는다.

각 edge에 `producer owns until`, `consumer owns after`, `failure cleanup`, `identity key` 네 주석을 붙인다. source line 목록보다 수명을 설명하는 지도다. branch가 많으면 대표 text stream=false부터 닫고 stream=true, batch, multimodal을 추가한다.

**R17의 수명을 마지막으로 대조한다**

첫 번째 화면은 OpenAI request다. 아직 messages와 provider-facing semantics다. 두 번째는 adapted GenerateReqInput이다. model/server-specific generation fields가 정규화되었지만 text일 수 있다. 세 번째는 TokenizedGenerateReqInput이다. exact IDs와 rid, sampling/multimodal metadata가 process boundary를 건널 준비가 됐다.

네 번째는 scheduler Req다. waiting/running/cache와 output state를 가질 engine object다. 다섯 번째는 detokenizer output batch다. 여러 Req 결과를 효율적으로 묶는다. 여섯 번째는 manager rid mailbox event다. 다시 개별 client request가 된다. 일곱 번째는 OpenAI SSE/JSON response다.

각 화면 사이에는 변환뿐 아니라 ownership handoff가 있다. API handler→TokenizerManager, manager→IPC, receiver→scheduler, scheduler/detokenizer→manager, manager→serving coroutine, coroutine→client다. failure는 edge와 양쪽 cleanup을 가져야 한다.

request identity가 이 모든 화면을 관통하고 input meaning은 messages→rendered→IDs→Req로 구체화된다. output meaning은 accepted token IDs→detokenized delta→API chunk로 변한다. abort는 반대 방향으로 client→manager→scheduler/cache로 내려간다.

**22.8의 중간 회고: R17의 수명은 한 번만 닫힌다.**

대표 text request가 route에서 exact tokenized message, scheduler Req, 첫/final output과 cleanup까지 끊김 없이 이어졌는가. 각 단계의 owner와 queue, timestamp를 말할 수 있어야 한다. 특수 mode 없이 이 경로가 먼저 이해되어야 한다.

multimodal은 raw media가 아니라 placeholder/grid/processor state가 어디서 만들어지고 회수되는지 설명해야 한다. overlap은 future/current buffer abort, P/D는 stage/transfer identity, speculative는 proposed와 accepted output을 기준 경로에 추가해야 한다.

failure 설명은 tokenizer exception, IPC send/receive, scheduler reject, output loop, slow client와 disconnect race를 서로 구분해야 한다. 모든 현상을 TTFT 또는 abort 하나로 합치면 미완성이다. source line과 관측 field가 같은 handoff를 가리켜야 한다.

마지막으로 no-runtime 범위를 정직하게 남긴다. 이 장은 v0.5.18 source에서 가능한 path와 invariants를 고정했으며 특정 deployment가 어떤 worker/overlap/P-D branch를 실제 선택했는지는 effective config와 runtime evidence가 필요하다. 측정값을 꾸며내지 않고 test와 trace 계약을 제공한다.

이 판정을 통과하면 독자는 다음 scheduler 장에서 Req가 어디서 왔는지 되묻지 않아도 된다. exact IDs, rid/incarnation, sampling/multimodal state와 API timestamps를 입구 증거로 받아 GPU scheduling과 KV ownership을 파고들 수 있다.

실제 조사 메모는 이 장의 흐름을 한 문장으로 압축해 시작할 수 있다. “API request R17은 template T와 tokenizer revision Z로 2048 IDs가 되었고, manager state generation G에 등록되어 tokenized message M으로 보내졌으며 scheduler Req Q가 되었고, output batch O가 같은 R17/G mailbox로 돌아왔다.” 어느 절이 비어 있는지가 바로 다음 probe다.

예를 들어 token count는 있는데 rendered/template identity가 없으면 입력 의미를 재현할 수 없다. IPC sent만 있고 scheduler received가 없으면 channel과 receiver를 본다. Req admitted까지 있는데 first output이 없으면 scheduler/model로 내려간다. output batch는 있는데 mailbox event가 없으면 manager demux다. mailbox event는 있는데 client byte가 없으면 serving/transport다.

abort 조사도 같은 문장에 terminal edge를 붙인다. “disconnect D 뒤 abort A가 generation G로 enqueue되었고 scheduler remove S와 KV release K가 완료됐으며 mailbox cleanup C가 한 번 실행됐다.” A까지만 있으면 취소 요청이지 취소 완료가 아니다. S/K가 있고 C가 없으면 API memory가 남고, C만 있고 S/K가 없으면 GPU work가 남는다.

queue 사건에서는 stage와 단위를 붙인다. tokenizer queue는 requests와 input length, IPC는 messages/bytes, scheduler waiting은 Req/tokens, output mailbox는 events/bytes로 잰다. 서로 다른 단위를 하나의 queue depth로 합산하지 않는다. arrival와 service rate, oldest age가 있어야 포화와 순간 burst를 구분한다.

이 수직 수명이 완성되면 다음 장의 scheduler는 막연한 GPU 대기열이 아니다. R17이 입력 identity와 소유권을 잃지 않은 채 admission을 요청하는 정확한 다음 상태다. 어느 handoff가 비어 있다면 평균 latency가 정상이어도 ingress 검토는 끝나지 않았다.

이 규율은 문서를 친절하게 만든다. 독자는 class 목록을 외우지 않고 자신의 증상을 시간축의 빈 edge에 놓는다. source 링크는 그 edge의 producer와 consumer를 보여 주고 수치 원장은 병목의 크기를 가늠하며 사건 fixture는 경쟁 가설을 반증한다. 체크 항목을 반복하지 않아도 조사 순서가 자연스럽게 나온다.

마지막으로 운영 변경은 이 수직 문장을 깨지 않는 범위에서 승인한다. worker scaling은 R17/G와 result mapping을 보존해야 하고, output coalescing은 accepted IDs와 emitted delta order를 보존해야 하며, overlap/P-D는 stage가 늘어도 하나의 client mailbox와 terminal outcome을 보존해야 한다. 성능 이득은 이 불변식을 통과한 뒤에만 유효하다.

따라서 ingress 최적화의 목표는 무조건 tokenization을 빠르게 만드는 것이 아니다. CPU preprocessing과 IPC가 GPU를 충분히 공급하면서도 overload에서는 일찍 backpressure를 걸고, disconnect에서는 future dispatch 권한까지 취소하며, output에서는 느린 client를 다른 requests와 격리하는 것이다. throughput, latency, memory와 correctness가 같은 ownership 설계에서 만난다.

이 관점을 갖고 다음 장으로 가면 scheduler의 waiting queue가 책 전체 요청의 첫 queue가 아니라는 사실도 선명해진다. 이미 API, tokenizer와 IPC queue를 통과했다. TTFT를 설명하려면 이 앞단 시간을 빼먹지 않고 scheduler token/KV 시간과 이어야 한다.

독자가 source revision을 갱신할 때도 동일한 수직 문장을 사용한다. class나 파일이 이동했더라도 route가 어떤 adapted input을 만들고, manager가 어떤 exact message를 보내며, scheduler가 어떤 Req를 만들고, 어느 output struct가 mailbox로 돌아오는지 다시 고정한다. line number가 유지되었다는 사실보다 producer/consumer 의미가 유지되었는지가 중요하다.

새 field가 message에 추가되면 default가 old clients와 어떤 의미를 갖는지, batch wrapper와 P/D path가 함께 전달하는지, abort/cleanup에서 resource를 더 회수해야 하는지 본다. source diff 한 줄이 request lifetime 여러 edge를 바꿀 수 있다. 그래서 schema 변화는 constructor만 아니라 귀환과 failure path까지 감사한다.

최종적으로 R17이 성공했는지 묻는 대신 입력 의미, scheduler incarnation, accepted output, client delivery와 cleanup이 모두 같은 R17/G에 귀속됐는지 묻는다. 이 질문이 정상 path와 race, 성능과 correctness를 하나의 이해 가능한 이야기로 묶는다.

그 일관된 수명이 다음 scheduler 분석의 단단한 출발점이 된다.

독자는 이제 그 경계를 스스로 추적한다.

## 22.9 TokenizerManager가 request mailbox와 IPC payload를 만드는 순간

SGLang ingress의 핵심은 tokenization 자체보다 ownership 전이다. HTTP/OpenAI serving이 adapted request를 TokenizerManager에 넘기면 manager는 request identity와 result mailbox/future를 먼저 준비하고 scheduler로 보낼 exact IPC object를 만든다. send 실패 전에 local state를 등록했다면 rollback이 필요하다.

### 22.9.1 payload identity

payload에는 request ID/generation, input IDs 또는 text/processor state, sampling/output params, stream flag, priority, LoRA/grammar/multimodal identity와 return routing stamp가 있다. 실제 DTO field는 pinned source에서 확인한다. 모든 API field가 그대로 scheduler에 가는 것은 아니다.

batch request는 child IDs와 parent/result ordering을 가진다. manager local mailbox mapping과 scheduler accepted Req mapping이 같은 순서를 보존해야 한다. rank/batch wrapper가 field를 누락하지 않는지 본다.

### 22.9.2 register-before-send와 rollback

result가 매우 빨리 돌아와도 mailbox가 있어야 한다. local state를 send 전에 등록하는 이유다. 그러나 ZMQ/SHM send/serialization이 실패하면 mailbox와 tokenizer/multimodal temporary, metrics/span을 제거해야 한다.

send 성공은 scheduler receive/admission이 아니다. `registered`, `serialized`, `sent`, `received`, `Req constructed`, `waiting/admitted`를 별 event로 둔다. client timeout이 어느 단계인지에 따라 abort owner가 달라진다.

### 22.9.3 queue/serialization 수치

초당2,000 requests, IPC payload 평균8KiB면 raw ingress가 약16MiB/s다. multimodal metadata/embedded features가 평균256KiB인 request가10%면 추가 약50MiB/s다. 실제 copy/SHM/zero-copy와 overhead는 source/trace가 결정한다.

queue cap1,000에서 arrival2,000/s, drain1,500/s이면 순증500/s로 2초에 찬다. cap 전 backpressure/reject가 어디서 발생하고 local mailbox를 만들기 전인지 후인지 본다. 늦은 reject는 CPU/메모리와 cleanup을 더 쓴다.

## 22.10 scheduler receive와 DTO→Req state transition

scheduler event loop가 IPC message를 receive하고 rank/order/control kind를 확인한 뒤 internal Req를 만든다. DTO parsing, request validation, dependency preparation과 queue insertion을 한 단계로 합치지 않는다. 실패 시 어디까지 state가 생겼는지 rollback을 본다.

### 22.10.1 receive rank/order

TP/DP 또는 tokenizer workers가 여러 channel을 쓰면 같은 logical request의 payload가 어느 scheduler/rank로 가는지 routing stamp가 필요하다. rank mismatch나 duplicate receive를 bounded error로 처리한다. shape가 맞아도 wrong scheduler incarnation에 가면 output routing이 깨진다.

message sequence와 generation으로 add/abort ordering을 보존한다. abort-before-add가 가능하면 tombstone/ordered channel이 필요하다. request ID만으로 old generation을 cancel하지 않는다.

### 22.10.2 Req construction

Req는 input IDs length, sampling params, origin/stream routing, state status와 cache/token bookkeeping을 소유한다. grammar/LoRA/multimodal dependency가 준비되지 않았다면 blocked/waiting reason을 둔다. constructor가 resource를 할당한 뒤 validation 실패하면 reject cleanup을 수행한다.

original IDs와 truncated/effective IDs가 둘 다 존재할 수 있다. usage/cache/observability consumer가 어느 것을 읽는지 표시한다. manager와 scheduler가 length를 다르게 계산하면 admission/budget이 갈린다.

### 22.10.3 admission response와 local mailbox

scheduler rejection/exception이 manager output loop로 돌아와 해당 mailbox를 terminal 처리해야 한다. HTTP writer가 기다리는 future가 영원히 남지 않게 한다. reject reason과 cleanup evidence를 보존한다.

accepted acknowledgment가 별도로 없고 첫 output/terminal로만 알 수 있다면 local state machine에 unknown/pending을 둔다. send 성공을 accepted로 metric하지 않는다.

## 22.11 overlap event loop와 output ordering을 incident로 닫는다

normal loop는 batch result를 생성한 뒤 manager로 보내고, overlap 모드는 다음 scheduler/worker step과 output processing을 겹칠 수 있다. 성능 이득은 compute/communication/CPU overlap이지만 buffer generation과 result ordering 책임이 늘어난다.

### 22.11.1 overlap lifetime

step n output buffer를 manager send/serialization이 읽는 동안 worker가 step n+1에 같은 storage를 재사용하면 stale/wrong delta가 나올 수 있다. stable copy, event dependency 또는 ring-buffer incarnation이 필요하다. Python object reference만으로 device/SHM consumer completion을 추정하지 않는다.

request별 token delta sequence는 monotonic해야 한다. batch result ordering이 step completion 순서와 다를 수 있어도 mailbox routing과 per-request cursor가 복원한다. late step n output이 terminal n+1 뒤 오면 drop/protocol violation을 generation으로 처리한다.

### 22.11.2 overlap incident O22

O22는 overlap mode에서만 두 요청의 first token이 간헐적으로 뒤바뀌었다. scheduler selected IDs는 맞고 output DTO batch shape도 맞았다. manager receive에서 request IDs와 token arrays mapping이 previous batch order를 사용했다.

경쟁 가설은 scheduler wrong row, serializer buffer reuse, routing order stamp stale, mailbox map collision이다. scheduler output request-ID/token pair snapshot이 맞고 serialized bytes부터 다르면 buffer/order producer를 본다. bytes는 맞고 manager demux가 다르면 receive consumer다.

root가 overlap ring slot의 routing stamp generation 갱신 누락이었다고 하자. 수정은 payload/result에 step/batch generation을 넣고 slot last consumer completion 뒤 reuse한다. `[A,B]`/`[B,A]`, cancel between steps, slow manager, normal/overlap matrix를 검증한다.

### 22.11.3 throughput과 latency

normal step compute10ms+send2ms이면 단순12ms, perfect overlap이면 steady state 약max(10,2)=10ms 상한이다. 실제 synchronization, queue와 batching을 포함한다. overlap 개선20%를 보편적으로 주장하지 않는다.

send가8ms로 늘면 overlap이 일부 숨겨도 manager queue/backpressure가 누적될 수 있다. kernel time만 빨라지고 client ITL이 나빠질 수 있다. scheduler step, serialization/send, manager receive/demux, socket delivery를 나눈다.

## 22.12 grammar와 multimodal state는 별 dependency identity다

grammar compilation과 multimodal preprocessing/encoder feature는 token IDs 외에 request가 scheduler에서 실행 가능해지는 조건이다. input shape만 맞다고 준비된 것이 아니다. artifact generation, tenant/request binding과 lifecycle을 보존한다.

### 22.12.1 grammar identity

grammar/schema bytes digest, compiler/backend/version, tokenizer vocabulary, compiled object generation을 key로 둔다. manager가 async compilation future를 소유하는지 scheduler가 blocked Req와 waiter를 소유하는지 source를 읽는다.

cancel된 request의 compile callback이 뒤늦게 scheduler queue에 재삽입하지 않게 generation guard를 둔다. shared compiled grammar cache와 request waiter cleanup을 구분한다. grammar failure는 explicit terminal/reject로 mailbox에 돌아온다.

### 22.12.2 multimodal identity

media content digest, processor revision, grid/feature shape, placeholder binding, encoder feature generation과 request ordinal을 둔다. manager에서 processor를 실행하는지 separate worker/scheduler path인지 확인한다. raw text IDs만 IPC에 보내고 feature routing을 별 channel로 보내면 두 message ordering이 필요하다.

request A image1, B image3인 batch 순서를 바꿔 feature ranges와 placeholders가 request identity에 유지되는지 본다. count total equality만으로 cross-request mix를 잡지 못한다. cancellation/reorder에서 late feature result를 reject한다.

### 22.12.3 blocked queue와 fairness

grammar/media 준비 중 request가 scheduler waiting count에 들어가도 schedulable하지 않을 수 있다. ready waiting work, blocked count/oldest age/reason을 나눈다. 긴 compilation/encoder가 ready requests fairness를 막지 않게 wake/queue policy를 본다.

dependency 완료 후 wake가 exactly once이고 cancelled/terminal generation을 건너뛰는지 fixture를 둔다. duplicate wake가 duplicate Req admission을 만들지 않게 한다.

## 22.13 abort가 manager·IPC·scheduler·dependency를 닫는 길

client disconnect는 manager mailbox/writer, pending send, scheduler Req, grammar/media waiter, worker/KV state를 닫아야 한다. abort request를 보냈다는 사실과 각 resource terminal을 분리한다.

### 22.13.1 단계별 abort

tokenization/preprocess 중에는 local task/future가 owner다. registered-before-send에서는 local rollback과 abort tombstone이 필요할 수 있다. sent-before-receive는 IPC reorder를 고려한다. waiting/running은 scheduler abort cleanup이다. output terminal 후 disconnect는 writer/local stream cleanup이다.

### 22.13.2 abort conservation 계산

disconnect100, manager abort created99, IPC sent98, scheduler received96, Req terminal95, resources released94라면 gaps1/1/2/1/1이다. propagation delay/cohort를 맞추고 oldest pending을 본다. single cancelled counter로 뭉치지 않는다.

grammar blocked10 중 cancel9, callback late3이면 late callbacks가 cancelled generation을 wake하지 않는지 본다. multimodal features2가 late arrive하면 shared cache admission과 request binding을 구분한다.

### 22.13.3 restart and abort

scheduler restart generation이 바뀌면 manager active mailboxes에 failure terminal을 broadcast하고 new scheduler readiness까지 admission을 fence한다. old outputs/abort acknowledgments를 new request IDs와 섞지 않는다. manager restart에서는 scheduler가 client lease/heartbeat loss로 orphan Req를 정리하는지 본다.

### 22.13.4 incident A22

grammar compile 중 disconnect됐지만 callback이 old request ID를 ready queue에 넣어 ghost request가 실행됐다. manager mailbox는 없어 output이 dropped됐고 KV만 소비됐다. first divergence는 callback generation guard 누락이다.

수정은 waiter에 request/scheduler generation과 terminal state를 확인하고 cancel 시 unregister한다. cancel-before/after-compile, scheduler restart, shared cache hit/miss와 duplicate callback fixture를 통과한다. output drop이 아니라 Req/KV terminal까지 확인한다.

## 22.14 pinned source·metrics·최종 handoff

source walk는 HTTP/OpenAI serving→TokenizerManager method→local state/mailbox register→send/IPC payload→scheduler receive/DTO→Req→queue/worker→output DTO→manager receive/demux→HTTP stream, 그리고 abort/dependency/restart path를 잇는다.

각 edge card에는 caller/callee, message class/fields, request/generation, state mutation, queue/channel, failure rollback, next consumer를 둔다. exact pinned revision/symbol을 사용한다. source에 overlap branch가 있다는 사실과 실제 enabled path를 구분한다.

metric에는 preprocessing, mailbox count/oldest, IPC serialize/send/receive, scheduler receive/admission, blocked dependencies, output queue/demux, abort propagation와 resource terminal을 둔다. request/media/schema는 high-cardinality label로 넣지 않는다. generation/reason/mode는 bounded하게 둔다.

matrix는 normal/overlap, single/batch reorder, send failure, abort-before/after-add, grammar compile success/fail/cancel, multimodal order/late result, slow output, manager/scheduler restart, P/D/speculative representative path를 포함한다.

P/D에서는 prefill/decode stage bootstrap/transfer/import와 one client mailbox terminal을 보존한다. speculative는 draft/target child state와 accepted output cursor를 root request에 귀속한다. 상세 수학은 후속 장으로 넘기되 ingress identity/abort가 모든 child/stage를 닫는지 본다.

최종 terminal은 manager local, IPC, scheduler Req, dependency, output/external, abort/resource, readiness/observability다. incident 문장은 “Overlap ring slot G41 routing stamp가 old `[A,B]` order를 유지해 G42 `[B,A]` token arrays를 wrong mailbox로 demux했다. step generation과 last-consumer fencing을 추가하고 reorder/cancel/slow-output matrix를 통과했다”처럼 쓴다.

독자는 이제 TokenizerManager를 tokenizer wrapper로만 보지 않는다. client mailbox, IPC dispatch, dependency와 output/abort ownership을 가진 ingress coordinator로 읽는다. 이 상태를 다음 scheduler 장에 넘기면 waiting/running/KV budget을 앞단 queue와 끊김 없이 연결할 수 있다.

**current source function trace를 독자 순서로 배열한다.**

HTTP/OpenAI handler에서 TokenizerManager의 generation method를 호출하는 exact span을 찾는다. adapted request와 raw HTTP disconnect handle이 어떻게 전달되는지 본다. manager method가 single/batch, text/input_ids, stream path를 어디서 분기하는지 표시한다.

manager 내부에서 request object validation/preprocessing, local future/event/mailbox registration, send object construction과 IPC method 호출을 순서대로 적는다. `try/finally`와 exception rollback을 놓치지 않는다. send 후 결과 loop가 어느 map/key로 response를 찾는지 잇는다.

scheduler process entry/event loop에서 message receive dispatcher와 type branch를 찾는다. generation request를 DTO에서 Req로 만드는 constructor/helper, add queue와 failure response를 연결한다. abort/control message가 같은 dispatcher인지 별 channel인지 확인한다.

output loop에서는 scheduler/worker result DTO, manager receive task, request-specific detokenizer/state, future/queue와 HTTP generator를 잇는다. normal/overlap 함수가 서로 다른 result readiness/lifetime을 갖는지 표로 둔다.

**IPC message schema를 compatibility contract로 읽는다.**

message class field 추가는 sender/receiver version을 동시에 바꾼다. default가 없는 새 required field는 mixed rollout을 깨뜨릴 수 있다. optional default가 있어도 semantics가 old receiver에서 silent loss되는지 본다. protocol version/handshake 또는 atomic rollout을 검토한다.

serialization은 Python object/pickle, structured bytes, shared-memory handle 등 실제 방식을 source에서 확인한다. multimodal tensor/large payload가 copy되는지 handle로 전달되는지에 따라 lifetime과 memory가 달라진다. 이름만으로 zero-copy라고 추정하지 않는다.

SHM handle이라면 producer가 buffer를 반환하기 전 scheduler consumer가 attach/copy 완료해야 한다. send exception과 receiver crash에서 segment cleanup owner를 둔다. stale handle generation이 다른 request data를 가리키지 않게 한다.

**manager event loop starvation incident.**

tokenizer CPU 작업이나 tool/grammar parsing이 manager event loop에서 synchronous하게 오래 실행되면 output receive/demux와 disconnect handling이 지연될 수 있다. scheduler GPU는 output을 만들었지만 client ITL과 abort propagation이 나빠진다.

trace에서 scheduler output timestamp는 정상인데 manager receive/demux gap이 늘고 mailbox queue age가 상승하면 event-loop owner를 본다. GPU/scheduler를 조정하지 않는다. CPU executor/offload, bounded concurrency와 ordering을 검토한다.

offload해도 request cancel이 CPU task를 중단/결과 drop하고 temporary memory를 정리해야 한다. task completion callback이 terminal generation을 mailbox에 쓰지 않게 guard한다.

**batch parent-child conservation.**

batch API 하나가 child N개를 만들면 external parent terminal과 child scheduler terminals를 분리한다. child 하나 reject/fail/cancel일 때 partial response policy, ordering과 parent cleanup을 정한다. parent cancel은 모든 active children과 shared artifacts를 닫는다.

parent request order와 scheduler batch packing order가 달라질 수 있다. output은 child identity로 원래 API order를 복원한다. 단순 array index를 physical batch position으로 사용하지 않는다. `[A,B,C]`→`[C,A,B]` metamorphic fixture를 둔다.

child accepted count+rejected+cancelled+inflight=N 보존식을 본다. parent terminal 뒤 late child output가 mailbox를 재생성하지 않게 generation을 확인한다.

**grammar compilation 비용과 cache 계산.**

grammar compile이 요청당20ms이고 같은 schema hit율80%, 초당1,000 grammar requests라면 miss200건×20ms=4 CPU-seconds/s의 compile work가 필요하다. 병렬 core/실제 분포를 고려하지만 cache identity가 CPU 병목에 중요한 이유를 보여 준다.

cache key에서 tokenizer vocab/compiler revision을 빼면 hit율은 높아도 wrong allowed-token state를 재사용한다. 정확성 key를 먼저 닫는다. tenant/schema 민감 정보와 cache sharing 정책을 고려한다.

compile future coalescing은 같은 key concurrent requests가 one producer를 기다리게 할 수 있다. producer failure/cancel이 모든 waiters에 적절한 terminal을 주고, 한 waiter cancel이 shared producer를 잘못 취소하지 않게 reference/ownership을 둔다.

**multimodal payload와 memory backpressure.**

이미지 feature 하나가 `[256,4096]` BF16이면 payload만2MiB다. 100 pending requests면200MiB이고 raw decode/temporary는 별도다. feature를 IPC copy하면 순간 duplicate memory가 생길 수 있다. metadata-only request와 같은 cap으로 세지 않는다.

admission은 request count뿐 아니라 payload bytes/encoder work와 mailbox memory를 본다. overload reject는 큰 preprocessing을 완료하기 전에 가능하면 일찍 한다. 그러나 raw request만으로 정확한 feature size를 모르면 staged reservation과 rollback이 필요하다.

cross-request feature binding은 보안 incident다. request ID/content ordinal/range generation assertion을 hard fail로 둔다. output 품질 metric으로만 찾지 않는다. cancellation/reorder/SHM reuse fixture를 유지한다.

**overlap option의 mutation chain.**

overlap enable option이 parser/config에서 어느 scheduler loop class/function을 선택하고, result queue/buffer count, event synchronization과 output path를 바꾸는지 걷는다. option flag가 true여도 capability/graph/model 때문에 fallback할 수 있다. effective loop mode를 trace한다.

normal과 overlap은 같은 semantic results를 내야 하지만 timestamps/order of internal completion은 다를 수 있다. request-token cursor와 terminal ordering으로 비교한다. global batch result ordering bitwise equality를 강제하지 않는다.

성능은 scheduler step, worker compute, result serialize/send, manager receive와 external ITL을 함께 본다. overlap이 worker utilization을 높여도 manager bottleneck/queue memory가 늘면 SLO goodput이 악화할 수 있다.

**P/D disaggregation ingress identity.**

prefill/decode가 다른 scheduler/worker로 나뉘면 manager 또는 router가 stage identity와 transfer/bootstrap state를 소유한다. one external request 아래 prefill Req, KV publish/transfer/import, decode Req가 있다. stage IDs와 root generation을 둔다.

client cancel이 prefill running, transfer pending, decode waiting 어느 stage에서든 모든 future stage dispatch 권한을 취소해야 한다. prefill abort만 하고 transfer callback이 decode를 만들면 ghost request가 된다. generation tombstone을 stage callbacks에 적용한다.

output은 decode stage에서 주로 오지만 prefill error/terminal이 manager mailbox로 돌아와야 한다. stage success를 external success로 혼동하지 않는다. detailed P/D protocol은 후속 장으로 넘기되 ingress terminal conservation을 닫는다.

**speculative child state.**

draft/target 요청과 accepted token state가 root request에 귀속된다. manager는 external token cursor만 보고 internal draft tokens를 그대로 stream하지 않는다. abort가 draft/target workers, temporary buffers와 root mailbox를 닫는지 확인한다.

speculative fallback/disable이 execution mode를 바꿔도 external identity와 usage/finish contract를 보존한다. selected mode와 draft model generation을 trace한다. accepted/generated draft count를 billing/output count와 구분한다.

**restart matrix를 구체화한다.**

manager만 restart, scheduler만 restart, both rolling restart, IPC partition을 나눈다. manager restart에서 scheduler active Req를 lease/heartbeat로 정리하는지, scheduler restart에서 manager mailboxes를 error terminal하는지 본다. 자동 retry는 external commit/idempotency를 고려한다.

mixed version sender/receiver message compatibility와 old SHM/output queues를 격리한다. readiness는 manager HTTP, IPC send/receive, scheduler generation/model ready와 output loop health를 포함한다. shallow health만으로 admission하지 않는다.

**metrics와 cardinality.**

gauges는 manager registered/pending, IPC queue bytes/oldest, scheduler receive pending, blocked grammar/media, output pending과 abort/resource pending을 둔다. counters는 send/deserialize/reject, stale generation, late output/callback와 restart terminal을 둔다.

histograms은 tokenize/process, serialize/send/receive, DTO→Req, scheduler wait, first output, demux/delivery, abort propagation/cleanup이다. mode/reason/dependency type/generation cohort는 bounded label, request/schema/media는 trace다.

conservation dashboard는 ingress accepted, manager registered, scheduler received/admitted/terminal, external terminal과 resources released를 같은 time/generation cohort로 맞춘다. queue snapshot과 event totals를 섞지 않는다.

**option review 질문.**

tokenizer workers 수를 늘리면 어느 queue, process/channel과 ordering이 바뀌는가. overlap은 buffer/event/loop를 어떻게 바꾸는가. grammar backend는 compile/cache identity와 scheduler allowed-token state를 어떻게 바꾸는가. multimodal limit는 preprocessing/IPC/admission을 어디서 거절하는가.

P/D/speculative option은 child/stage state와 cancel fan-out을 어떻게 바꾸는가. 각 option을 parser→constructed component→message/state→consumer→effect→falsifier로 쓴다. “빠름”, “지원”만 쓰지 않는다.

**incident dossier의 최종 형식.**

R22/G의 timeline, manager mailbox/current state, send message sequence/generation, scheduler receive/Req status, dependency state, output cursor, abort/resources와 source spans를 둔다. passing neighbor와 batch neighbors를 보존한다.

first divergence가 manager payload면 scheduler를 고치지 않는다. payload bytes는 맞고 Req construction부터 다르면 receiver를 본다. scheduler selected result는 맞고 demux부터 다르면 output/overlap owner다. abort received 뒤 resource만 남으면 scheduler cleanup이다.

수정 후 original failure, normal/overlap, batch reorder, cancel/restart와 dependency boundary를 통과한다. service, resource, telemetry terminal을 별도로 닫는다. old generation pending이 관측 window 뒤0인지 확인한다.

22장의 final artifact는 TokenizerManager→Scheduler의 한 방향 payload와 Scheduler→manager→client의 역방향 output/terminal, 그리고 abort fan-out이다. 이 세 화살표가 같은 request generation과 resource ledger를 보존하면 scheduler 내부를 더 깊게 파도 ingress 의미를 잃지 않는다.

**처음 소스를 펼친 독자를 위한 90분 조사 순서.**

처음 15분에는 서버 실행 옵션을 읽지 말고 요청 하나의 이름이 바뀌는 지점만 표시한다. HTTP 요청의 식별자, manager가 만든 내부 식별자, scheduler가 받는 식별자, 출력 객체의 식별자가 같은 값인지 별도 값인지 적는다. 값이 같다는 사실만으로 같은 수명이라는 결론을 내리지 않는다. 재시작 세대와 batch child ordinal이 빠지면 문자열이 같아도 다른 실행일 수 있다. 이 단계의 산출물은 네 칸짜리 identity 표와 각 변환을 수행한 함수 span이다.

다음 15분에는 `send` 앞뒤만 읽는다. send object를 만들기 전에 등록되는 mailbox/future가 무엇인지, serialization 실패 시 어떤 등록을 되돌리는지, send 성공이 scheduler admission을 뜻하는지 확인한다. 이 경계에서 가장 흔한 오독은 “메시지를 보냈으므로 요청이 존재한다”이다. 실제로는 manager에는 요청이 있고 scheduler에는 아직 없거나, 반대로 scheduler에는 생겼지만 응답 경로가 끊긴 구간이 있다. 두 소유자의 상태를 한 열에 합치지 않는다.

세 번째 15분에는 receive dispatcher부터 `Req` 생성까지 역방향으로 읽는다. queue 삽입 함수를 먼저 찾고 caller를 거슬러 올라가면 어떤 message type과 validation이 그 함수에 도달하는지 빨리 알 수 있다. constructor가 토큰 배열, sampling parameter, grammar handle, multimodal metadata를 복사하는지 참조하는지도 본다. 복사 여부는 메모리 최적화 trivia가 아니라 producer buffer를 언제 재사용할 수 있는지를 결정하는 수명 계약이다.

네 번째 15분에는 정상 출력 한 조각을 따라간다. worker 결과가 scheduler output DTO가 되는 지점, IPC에 넣는 지점, manager receive loop가 꺼내는 지점, detokenizer가 visible text를 갱신하는 지점, HTTP generator가 yield하는 지점을 연결한다. 각 지점에 `generated`, `received`, `detokenized`, `delivered` cursor를 적으면 중복 출력과 누락 출력이 어느 경계에서 생겼는지 구분할 수 있다. finish reason과 usage가 마지막 content와 같은 frame인지 별 frame인지도 API 계약과 함께 확인한다.

마지막 30분에는 정상 경로와 나란히 cancel 하나, send failure 하나, scheduler restart 하나를 겹쳐 그린다. 예외 처리 코드를 모아서 읽는 방식보다 정상 경로의 각 소유권 획득 직후 실패를 주입하는 방식이 빠뜨림이 적다. 최종 산출물에는 실행하지 않았다는 사실, 정적 소스에서 확인한 것, 운영 trace로 검증해야 할 가설을 서로 다른 표시로 남긴다. 정적 분석만으로 queue 지연 분포나 실제 fallback 비율을 단정하지 않는다.

**작은 수치 예제로 backpressure를 계산한다.**

manager가 초당 1,200건을 받고 scheduler가 초당 1,000건을 안정적으로 받아들인다고 하자. 차이 200건/s가 manager 또는 IPC 대기열에 쌓인다. 평균 payload가 64KiB이면 payload만 초당 약 12.5MiB 증가한다. 30초 동안 admission 제어가 없다면 약 375MiB이며 Python 객체, 토큰 배열, multimodal 임시 버퍼와 응답 mailbox는 제외한 값이다. request count만 보면 6,000건이지만 bytes 관점에서는 이미 프로세스 안정성을 위협할 수 있다.

반대로 작은 텍스트 요청 5,000건과 2MiB feature를 가진 multimodal 요청 500건은 count만으로 전자가 더 커 보인다. payload 기준으로 후자는 약 1GiB다. 따라서 `pending_requests` 하나로 cap을 구현하면 workload에 따라 너무 일찍 거절하거나 너무 늦게 죽는다. request slots, estimated bytes, preprocessing work, scheduler token budget을 서로 다른 reservation으로 두고 어느 단계에서 확정·반환되는지 적는다.

대기열이 5,000건이고 처리율이 1,000건/s라면 새 요청은 서비스 시간과 별개로 대략 5초의 queue delay를 만날 수 있다. 이 수치는 Little의 법칙을 적용하기 전의 직관적 근사이며 arrival과 service가 비정상적이면 그대로 예측값으로 쓰면 안 된다. 핵심은 GPU execution latency가 정상이어도 ingress queue가 TTFT를 망칠 수 있다는 것이다. dashboard는 manager oldest age와 scheduler waiting age를 분리해 어느 queue가 5초를 소유하는지 보여줘야 한다.

overload 정책은 “queue가 차면 503”으로 끝나지 않는다. 이미 비싼 image preprocessing을 끝낸 뒤 거절하면 GPU는 보호해도 CPU와 메모리를 낭비한다. 너무 이른 추정은 실제 토큰 수와 feature 크기를 모르므로 false reject를 늘린다. cheap validation 뒤 provisional reservation, preprocessing 뒤 actual adjustment, scheduler admission 뒤 manager reservation release처럼 단계별 계약을 둔다. adjustment 실패 시 temporary artifact와 mailbox가 함께 닫혀야 한다.

**실패 주입 카드를 소유권 경계마다 만든다.**

카드 첫 줄은 주입 지점이다. `mailbox 등록 직후`, `serialize 직후`, `IPC send 반환 직전`, `scheduler receive 직후`, `Req queue 삽입 직후`, `첫 output 생성 직후`, `manager demux 직후`처럼 모호하지 않게 쓴다. 둘째 줄은 기대 terminal이다. external error/cancel만 쓰지 말고 manager registration 제거, IPC buffer 반환, scheduler Req terminal, KV/resource release, dependency waiter unregister를 열거한다.

셋째 줄은 관측 증거다. bounded reason counter, stage latency, trace event와 generation별 pending gauge를 둔다. 로그 한 줄은 terminal의 증거가 아니다. 넷째 줄은 반증 조건이다. 예를 들어 send failure 뒤 scheduler receive가 0이고 manager pending도 0이어야 한다. scheduler receive가 1이면 “send가 실패했다”는 API 반환 의미와 실제 전달 여부 사이에 모호성이 있으므로 idempotent reconciliation이 필요하다.

다섯째 줄은 이웃 요청이다. 실패 요청만 terminal됐다고 성공으로 판정하면 shared batch/slot corruption을 놓친다. 앞뒤 요청의 token sequence, output cursor, grammar/media binding과 latency를 비교한다. `[A,B,C]`에서 B를 취소한 뒤 `[A,C]`의 identity가 보존되는지, overlap buffer가 B의 slot을 재사용한 뒤 늦은 event가 C를 건드리지 않는지 확인한다.

여섯째 줄은 재시작 변형이다. 같은 주입을 scheduler generation 전환 직전과 직후에 반복한다. old generation output가 drop되는 것만 보지 말고 old Req/KV/SHM이 회수되는지 본다. manager가 오류를 client에 보냈더라도 resource terminal이 남으면 복구가 아니다. 반대로 자원을 회수했지만 client stream이 무한 대기하면 service terminal이 빠졌다.

**코드 리뷰에서 바로 쓸 수 있는 질문.**

요청 객체 필드마다 producer, first consumer, last consumer를 말할 수 있는가. token IDs와 multimodal handles는 어느 함수에서 immutable해지는가. batch reorder 뒤 원래 client order를 복원하는 key는 무엇인가. optional field가 없는 구버전 sender를 receiver가 어떻게 해석하는가. unknown field를 구버전 receiver가 무시해도 안전한가. 이 질문에 답하지 못하면 message schema 변경을 단순 dataclass 수정으로 승인하지 않는다.

manager map에 등록한 뒤 모든 exit가 제거를 보장하는가. exception, disconnect, timeout, scheduler reject, restart broadcast와 normal finish가 동일 cleanup helper로 수렴하는가. helper가 멱등적인가. cleanup 두 번이 다른 요청의 재사용 slot을 지우지 않는가. map 크기만 감소하는지 보지 말고 generation key와 object identity assertion을 본다.

scheduler의 `Req`는 언제 token/KV budget을 예약하는가. validation 실패가 reservation 이전인지 이후인지, 이후라면 rollback이 있는지 본다. grammar/media blocked request가 waiting count에는 들어가지만 scheduling budget을 점유하는지 구분한다. blocked oldest가 ready oldest를 가리는 metric 설계도 피한다. ready queue와 dependency queue의 fairness를 따로 설명할 수 있어야 한다.

output DTO가 delta인지 snapshot인지, manager가 중복 receive를 견디는지 묻는다. delta에 sequence number가 없다면 retry/duplicate channel에서 중복 전달을 찾기 어렵다. snapshot은 payload가 커지지만 cursor reconciliation이 쉬울 수 있다. 실제 구현의 선택을 확인하고 trade-off를 기록한다. 프로토콜을 바꾸라는 결론보다 현재 불변식과 실패 조건을 먼저 적는다.

abort acknowledgment가 무엇을 보장하는가. scheduler가 control message를 받았다는 뜻인지, ready/running queue에서 제거했다는 뜻인지, worker와 KV까지 회수됐다는 뜻인지 구분한다. acknowledgment가 없다면 manager는 client terminal 후 resource completion을 어떻게 관측하는가. background cleanup을 허용한다면 최대 지연과 stale-generation alarm을 둔다.

**독자가 재현 가능한 incident 판정 연습.**

사건은 “stream이 멈췄다”로 시작한다. 먼저 external last-delivered timestamp와 manager last-demux timestamp를 비교한다. 둘 다 멈췄지만 scheduler output counter가 증가하면 IPC/receive owner를 본다. manager demux는 증가하는데 delivery만 멈추면 writer backpressure와 disconnect detection을 본다. scheduler output 자체가 멈추면 Req state와 dependency/worker path로 내려간다. 동일 증상을 queue 하나의 문제라고 부르지 않는다.

예를 들어 G52의 요청 1,000건 중 12건이 멈췄고 모두 grammar miss이며 manager mailbox age는 40초, scheduler에는 해당 Req가 없다고 하자. HTTP와 GPU를 먼저 의심할 근거가 약하다. compile future 완료 callback, send 여부, callback generation과 cancellation state를 연결한다. compile completed 12, callback invoked 12, send attempted 0이면 first divergence는 callback에서 dispatch로 넘어가는 조건이다.

다른 예로 scheduler terminal 1,000, manager terminal 988, stale output drop 12라면 drop 자체는 generation fencing이 작동한 증거일 수 있다. 그러나 왜 old generation 요청 12개가 client terminal을 받지 못했는지가 남는다. restart broadcast 대상 snapshot과 mailbox 등록 시점의 경쟁을 본다. broadcast snapshot 이후 old scheduler로 send된 요청이 있다면 admission fence 순서가 잘못됐을 수 있다.

수정은 한 줄 lock 추가로 설명하지 않는다. readiness false 전환, new admission 차단, active mailbox snapshot/terminal, old channel drain 또는 discard, resource reconciliation, new generation ready의 순서를 명시한다. 그리고 send-before-fence, register-before-broadcast, broadcast-before-send의 세 interleaving fixture를 둔다. 잠금 범위가 tokenization이나 slow writer까지 포함돼 event loop를 막지 않는지도 검토한다.

이 장의 완료 조건은 모든 함수를 암기하는 것이 아니다. 임의의 SGLang ingress 장애를 받았을 때 독자가 request identity, forward payload, reverse output, abort fan-out의 세 흐름을 그리고 첫 불일치 경계를 고를 수 있어야 한다. 근거 span, 수치 보존식, 실패 주입, terminal ledger가 함께 있을 때만 그 판단은 재현 가능하다. 다음 장의 scheduler 정책은 이 ingress 계약 위에서 시작한다.

**배포 전 마지막 대조표.**

변경 전후의 effective configuration을 먼저 저장한다. CLI 문자열만 비교하지 않고 manager process 수, tokenizer mode, overlap loop 선택, grammar backend, multimodal processor, IPC endpoint와 scheduler generation으로 해석된 값을 비교한다. 기본값 변경은 사용자가 옵션을 쓰지 않은 배포에도 영향을 준다. capability 검사 때문에 요청한 값과 실제 값이 다르면 요청값과 선택값을 둘 다 남긴다.

동일한 pinned fixture를 구버전과 신버전에 넣어 forward payload의 schema, token IDs, request/generation, sampling/grammar/media fields를 비교한다. byte equality가 필수인 필드와 semantic equality만 필요한 필드를 구분한다. random ID와 timestamp 차이를 무작정 제외하지 말고 consumer가 그것을 routing이나 deduplication에 쓰는지 먼저 확인한다. batch fixture는 single fixture의 반복이 아니라 reorder와 partial cancel을 포함한다.

reverse path에서는 첫 output까지의 event sequence와 마지막 terminal sequence를 비교한다. content가 같아도 finish reason, usage, logprob alignment, tool/grammar state와 cancel behavior가 달라질 수 있다. slow consumer에서 manager queue bytes, scheduler output pending과 external delivered cursor를 함께 본다. latency 평균이 좋아졌어도 p99 mailbox age나 orphan resource가 악화하면 승격하지 않는다.

canary는 정상 요청만 흘리지 않는다. grammar cache miss, 큰 multimodal payload, disconnect-before-send, disconnect-after-admission, overlap reorder와 scheduler restart를 작은 비율로 포함한다. 실패 주입 트래픽은 사용자 트래픽과 분리 표식하되 request ID를 metric label로 만들지 않는다. canary generation의 conservation gap과 stale callback/output 수가 0으로 수렴하는지 확인한다.

rollback 역시 프로세스 이미지만 되돌리는 작업이 아니다. mixed sender/receiver schema가 남는 시간, old shared-memory segment와 IPC queue, active scheduler Req, manager mailbox를 어떻게 drain할지 정한다. 즉시 종료가 요청을 잃는다면 admission을 먼저 막고 bounded drain 뒤 generation을 폐기한다. drain timeout 뒤 남은 요청에는 명시적 external terminal과 resource reconciliation을 수행한다.

마지막으로 source revision, build artifact, effective config, message schema fingerprint, fixture 결과, metric query, trace와 rollback 판정을 한 dossier에 묶는다. 다음 사람이 동일 commit에서 같은 caller→consumer 경로를 다시 찾을 수 있어야 한다. “문제 없음”이라는 결론만 남기지 말고 어떤 경계까지 확인했고 실행 검증을 하지 않아 남은 가설이 무엇인지 쓴다. 이것이 정적 소스 검토를 운영 가능한 지식으로 바꾸는 마지막 단계다.

리뷰 승인 문장도 검증 가능하게 쓴다. “ingress 호환성이 유지된다”가 아니라 “고정 fixture의 request/generation과 token·grammar·media binding이 manager send부터 scheduler `Req`까지 보존됐고, normal/overlap에서 delivered cursor와 terminal이 일치했으며, cancel/restart 뒤 generation별 pending resource가 관측 창 안에 0이 됐다”라고 쓴다. 아직 실행하지 않았다면 완료형 대신 “소스상 보존되며 다음 fixture로 확인해야 한다”고 제한한다.

이 구분은 겸손한 표현을 위한 장식이 아니다. source proof는 가능한 branch와 cleanup 코드를 보여 주지만 production configuration이 그 branch를 택했는지, race가 실제 timing에서 재현되는지, queue가 어떤 분포를 갖는지는 알려 주지 못한다. 반대로 trace는 한 실행을 보여 줄 뿐 unreachable-looking branch의 안전성을 증명하지 못한다. 코드 span, configuration provenance, controlled fixture, production telemetry가 같은 request generation을 가리킬 때 비로소 강한 판정이 된다.

따라서 독자의 노트에는 `확정`, `조건부`, `미검증` 세 상태가 필요하다. 확정은 고정 revision의 실제 caller와 consumer가 뒷받침한다. 조건부는 option이나 capability에 따라 선택되는 branch다. 미검증은 실행 timing, 배포 topology, workload 분포처럼 소스만으로 닫을 수 없는 항목이다. 이 표식이 있으면 후속 실험은 이미 확인한 코드를 반복하지 않고 가장 위험한 빈칸부터 채운다.

각 빈칸에는 담당자, 재현 fixture, 판정 metric과 중단 조건까지 붙여야 실제 검증 과제로 전환된다.
