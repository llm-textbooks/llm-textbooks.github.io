# 21장. vLLM ingress에서 AsyncLLM까지 요청 수명

client가 text completion 요청 하나를 보냈다. HTTP route는 즉시 request object를 받았지만 GPU는 아직 아무 일도 하지 않았다. prompt validation과 tokenization을 통과하고 AsyncLLM이 request ID를 등록하고 EngineCore client가 work를 받아야 비로소 engine 수명이 시작된다. 생성 token이 나와도 API stream이 commit하기 전에는 client에게 전달되지 않았다.

이 장은 vLLM `v0.27.1`, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`의 대표 text request 하나를 OpenAI entrypoint→preprocessing/tokenization→`AsyncLLM.generate`와 `add_request`→EngineCore client→output handler와 stream→abort·shutdown까지 따라간다. V1/V2나 selector 목록부터 외우지 않는다. 요청 하나의 identity, queue와 output ownership을 먼저 닫는다.

source만 읽고 model/server/CUDA를 실행하지 않는다. source에 branch가 있다는 사실과 현재 deployment가 그 branch를 선택했다는 사실을 분리한다. 실제 queue latency와 backpressure는 관찰 field로 정의하되 측정 결과를 만들지 않는다.

## 21.1 HTTP 요청 하나가 engine request가 되기 전까지

### 21.1.1 wire request와 engine request의 identity를 분리한다

문제 장면부터 보자. `/v1/completions`에 prompt와 `max_tokens=32`, `stream=true`를 보냈는데 400 error가 즉시 왔다. GPU utilization은 0이다. 이를 engine failure라 부르면 조사 시작점이 너무 아래다. route·schema·preprocessor 어느 곳에서 거절됐는지 먼저 찾는다.

OpenAI completion route는 [`completion/api_router.py`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/completion/api_router.py#L1-L70)에서 serving object에 request와 raw HTTP context를 넘긴다.

실제 completion lifecycle은 [`completion/serving.py:120-330`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/completion/serving.py#L120-L330)에서 model validation, prompt preparation, sampling conversion과 engine generator 준비를 잇는다.

route가 body를 parse했다는 사실은 request가 scheduler에 들어갔다는 뜻이 아니다. model alias가 loaded model과 맞는지, prompt type이 허용되는지, token length가 context 안인지, sampling fields가 valid한지 검사한다. 이 단계의 error는 request ID가 API에만 존재하고 EngineCore에는 없을 수 있다.

대표 prompt를 `Hello`로 두자. raw request에는 text와 API options가 있다. typed request는 defaults를 가진다. preprocessing은 model config와 tokenizer/template policy를 보고 engine prompt representation을 만든다. tokenizer는 text를 integer IDs로 바꾼다. max model length와 requested output budget을 함께 검사한다.

vLLM preprocessing의 공통 owner는 [`entrypoints/openai/serving_engine.py:1-320`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/completion/serving.py#L1-L320)와 input preprocessing 경로다. [`inputs/preprocess.py:1-300`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/inputs/preprocess.py#L1-L291)에서 text/token prompt, multimodal inputs와 tokenization 결과가 model input이 되는 source symbol을 따라간다.

tokenizer 호출은 CPU work다. request receive 시각과 tokenize start/end, engine enqueue를 나누면 queue 이전 지연을 설명할 수 있다. prompt가 길거나 tokenizer worker가 밀리면 GPU가 한가해도 TTFT가 늦을 수 있다. API latency를 모두 scheduler queue로 귀속하지 않는다.

request ID는 이때 만들어지거나 client-provided ID와 결합된다. 같은 ID가 logs, output queue와 abort에 쓰이므로 uniqueness가 중요하다. retry가 같은 external ID를 보낼 때 physical attempt를 구분할 incarnation이나 internal ID가 필요하다. ID collision이 있으면 output stream과 cancel target이 섞일 수 있다.

preprocess checkpoint는 raw request hash, typed effective fields, resolved model/tokenizer revision, prompt IDs와 count, sampling params, request ID, arrival timestamp다. prompt 내용은 민감하므로 synthetic fixture와 hash를 쓴다.

first divergence는 명확하다. raw body에서 typed fields가 다르면 protocol/validation, rendered or token IDs부터 다르면 preprocessing/tokenizer, engine params부터 다르면 request conversion이다. EngineCore를 보기 전에 이 경계를 닫는다.

### 전처리 사건: GPU가 놀지만 TTFT가 길다

현장에서 queue time이 길다는 신고를 받았다고 하자. scheduler waiting metric은 낮고 GPU도 한동안 idle인데 request receive부터 first token까지 800ms가 걸린다. 이때 scheduler를 튜닝하기 전에 API receive→preprocess enqueue→tokenizer start→tokenizer finish→core add 구간을 나눈다.

긴 prompt 하나가 tokenizer CPU를 점유하거나 process pool의 앞 작업이 느릴 수 있다. chat이 아니라 text completion이어도 normalization과 encode 비용이 있다. multimodal request가 같은 preprocessor resource를 공유하면 text request가 간접 대기할 수 있다. event 이름을 모두 queue time으로 합치지 않는다.

prompt가 이미 token IDs인 request와 text prompt를 비교하면 tokenizer cost를 격리할 수 있다. 그러나 두 API paths가 validation과 special token policy까지 동일하다는 보장은 없다. source에서 pretokenized input branch가 BOS/EOS와 max-length check를 어떻게 처리하는지 확인한다.

tokenization 결과가 cache될 수 있다면 cache key를 본다. raw text만이 아니라 tokenizer revision, special-token flags와 preprocessing options가 identity에 포함되어야 한다. 잘못된 hit는 빠른 오답이고 miss는 느린 정답이다. latency 개선을 correctness보다 먼저 승인하지 않는다.

preprocessor backpressure도 owner를 가진다. API coroutine이 tokenizer worker slot을 await하면 HTTP task count가 늘 수 있다. unbounded work queue면 memory와 tail latency가 늘어난다. reject/admission policy가 있다면 scheduler admission 이전 429 또는 503으로 보일 수 있다.

source 감사에서는 serving function이 preprocessing coroutine을 어디서 await하고 exception을 어떤 error response로 바꾸는지, tokenizer executor가 bounded인지, result가 어느 request ID로 돌아오는지 본다. 실제 CPU duration은 실행 없이는 주장하지 않는다.

first divergence가 token IDs 이전이면 model runner profile을 열지 않는다. raw prompt 길이 cohort, tokenizer service/queue time, process utilization을 관찰한다. token IDs와 add timestamp가 정상인데 first output만 늦으면 그때 core/scheduler로 넘어간다.

복구 검증은 짧은/긴 text, pretokenized IDs, concurrent burst를 분리한다. average만 줄지 않고 P99 queue와 error policy, output IDs parity를 본다. tokenizer worker 수를 늘렸다면 CPU contention으로 각 service time이 늘 수 있어 queue와 service를 함께 본다.

## 21.2 AsyncLLM.generate는 generator와 engine 수명을 묶는다

preprocessing 뒤 serving layer는 `AsyncLLM.generate`를 호출해 async iterator를 얻는다. 겉으로는 token stream을 순회하는 generator지만 내부에는 request 등록, output queue와 cancellation cleanup이 연결된다.

고정 source의 [`vllm/v1/engine/async_llm.py:150-360`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/async_llm.py#L150-L360)에서 `generate`, `add_request`, abort와 output processor 연결을 읽는다. exact line은 symbol 선언과 caller를 고정 commit tree에서 검산한다.

직관적으로 `generate`는 request를 넣고 output을 기다린다. 그러나 generator body가 언제 실행되는지 주의한다. async generator는 object를 만들었을 때와 첫 iteration을 시작했을 때 side effect 시점이 다를 수 있다. serving layer가 iterator를 반환했지만 client disconnect로 iteration이 시작되지 않으면 request가 등록됐는지 source ordering을 확인한다.

request 등록에는 request ID와 prompt, params뿐 아니라 output kind, adapters, priority, trace headers와 arrival time이 들어갈 수 있다. `add_request`가 preprocessing을 다시 수행하는지 이미 prepared prompt를 받는지 caller/callee contract를 본다. 같은 validation을 두 번 한다고 가정하지 않는다.

AsyncLLM은 request별 output stream/queue를 소유한다. EngineCore에서 온 batched outputs가 request ID로 demultiplex되어 해당 iterator에 들어간다. request가 terminal이면 stream을 닫고 map에서 제거해야 한다. ID map insertion과 removal이 정확히 한 번 이뤄져야 한다.

ownership을 세 줄로 적자. API coroutine은 transport lifetime을 소유한다. AsyncLLM은 request-to-output-stream mapping과 core client 요청 lifetime을 소유한다. EngineCore scheduler는 running/waiting state와 KV/resource lifetime을 소유한다. 한 owner의 종료가 다음 owner에게 자동 전파된다고 가정하지 않는다.

engine epoch도 필요하다. EngineCore process가 restart됐는데 API process가 살아 있으면 이전 request IDs와 output queue가 새 core outputs와 섞여서는 안 된다. client object, process generation, connection identity가 restart boundary를 어떻게 표현하는지 source에서 찾는다. epoch field가 명시적이지 않다면 connection 재생성과 pending request failure가 그 역할을 할 수 있다.

add failure도 분리한다. API validation은 통과했지만 EngineCore client가 unavailable하거나 queue put이 실패할 수 있다. request stream map에 먼저 insert한 뒤 core send가 실패하면 rollback이 필요하다. 반대로 core가 request를 받았는데 local output stream creation이 실패하면 orphan work가 생길 수 있다.

transaction 질문은 순서다. local stream 생성, map 등록, core add 전송 중 어느 것이 먼저인가. 중간 exception에서 어떤 cleanup과 abort가 실행되는가. retry 전에 같은 request ID state가 남는가. source의 `try/finally`와 exception mapping을 읽는다.

대표 request가 성공하면 `generate` iterator는 zero or more intermediate outputs와 one terminal output을 낸다. empty output이 terminal을 운반할 수 있으므로 token list가 비었다고 버리지 않는다. output object의 finished property와 finish reason을 source에서 확인한다.

### request ID는 네 map을 관통하는 capability다

request ID를 log correlation string으로만 생각하면 collision의 위험을 과소평가한다. serving layer는 ID로 generator를 식별하고 AsyncLLM은 output stream map을 찾으며 core는 scheduler request를 찾고 abort는 제거할 target을 찾는다. 같은 문자열이 관찰뿐 아니라 제어 capability다.

외부 client가 ID를 지정할 수 있다면 trust boundary를 본다. 다른 tenant의 active ID를 추측해 abort할 수 없어야 한다. API auth context와 internal ID namespace를 분리하거나 server-generated unique ID를 쓴다. core utility method가 ID를 받는다는 사실만으로 network에 그대로 노출된다고 단정하지 않지만 route validation을 확인한다.

ID uniqueness는 시간축도 포함한다. request `r1`이 finish해 map에서 제거된 직후 retry가 같은 `r1`을 쓰고 late output이 도착하면 새 stream에 들어갈 수 있다. ID를 영구 unique하게 만들거나 epoch/incarnation과 함께 match해야 한다. cleanup이 늦은 output을 drop할 때 stale event metric을 남긴다.

한 API request가 `n=4` candidates를 만들면 parent ID와 child sequence identity가 있을 수 있다. output object는 choice index를 보존하고 parent abort는 모든 child work를 제거해야 한다. child 하나 finish가 parent stream terminal이라는 뜻은 아니다.

beam이나 speculative 내부 candidates는 public choices와도 다르다. client가 보는 choice index와 scheduler sequence ID를 같은 것으로 쓰지 않는다. output processor가 mapping을 소유한다. retry가 parent ID를 재사용할 때 internal child namespace를 새로 만든다.

engine epoch는 core process lifetime을 구분한다. explicit integer가 없더라도 IPC connection, client instance와 pending request failure가 epoch 역할을 한다. old core가 죽고 new core에 reconnect하면 old pending IDs를 성공 상태로 carry하지 않는다. replay policy가 있다면 prompt와 params를 새 incarnation으로 명시적으로 재등록해야 한다.

epoch 없이 reconnection만 하면 위험한 장면을 생각하자. old core output queue에 `r7` final이 남아 있고 new request도 `r7`이다. API handler가 queue를 drain하며 new stream에 old result를 전달할 수 있다. transport endpoint 재생성과 output source identity가 stale batch를 막는지 본다.

관찰 로그에는 external request ID, internal request ID, attempt/incarnation, core client instance와 process generation을 둔다. 어느 값이 source에 명시적으로 없는지는 inference로 표시한다. 한 string을 request lifetime 전체의 완전한 identity라고 가정하지 않는다.

ID collision fixture는 active duplicate, finish 직후 duplicate, abort-before-add, restart 후 reuse를 포함한다. expected behavior는 reject, safe namespace 또는 stale output drop 중 source contract로 고정한다. silent overwrite는 허용하지 않는다.

### add_request의 transaction을 상태로 펼친다

`add_request` 한 call을 원자적 사건으로 그리면 중간 failure를 놓친다. 상태를 `NEW`, `LOCAL_STREAM_REGISTERED`, `CORE_ADD_SENT`, `CORE_ACCEPTED`, `ACTIVE`, `TERMINAL`, `CLEANED`로 나눌 수 있다. 실제 source가 같은 enum을 쓰지 않아도 의미 상태는 존재한다.

NEW에는 preprocessed prompt와 params가 있지만 output stream map과 core state가 없다. local registered 뒤에는 iterator가 outputs를 받을 주소가 생겼다. core send 뒤에는 command가 transport에 들어갔지만 scheduler accepted 여부는 모를 수 있다. active는 output이나 explicit acknowledgment로 core ownership을 확인한 상태다.

local register가 core send보다 먼저라면 send failure 때 map rollback이 필요하다. core send가 먼저라면 local registration failure 때 abort가 필요하다. 두 side effect 사이 exception을 source의 `try`와 cleanup으로 확인한다. 편리한 ordering 하나가 모든 race를 없애지는 않는다.

acknowledgment가 없다면 core accepted를 정확히 아는 시점이 첫 output일 수 있다. send exception도 core가 message 일부를 받았는지 불명확할 수 있다. best-effort abort와 idempotent add/abort가 필요하다. exactly-once transport를 source 근거 없이 가정하지 않는다.

priority나 adapter validation 일부가 core에서만 일어나면 API-side add 성공 뒤 asynchronous error가 stream에 올 수 있다. HTTP headers가 이미 나간 streaming path에서는 status code 대신 error event/close로 표현될 수 있다. add validation을 가능한 앞에서 하는 이유다.

state transition마다 owner를 적는다. serving coroutine은 NEW prompt를 만들고, AsyncLLM은 local stream과 send를 소유하며, core client는 transport, EngineCore는 accepted/active, scheduler는 terminal/resource, output handler는 local clean을 소유한다. cleanup responsibility가 겹치는 지점은 idempotency를 본다.

정적 fixture는 core send 직전 exception, send 직후 connection close, duplicate ID, output stream allocation failure를 fault point로 둔다. 실행하지 않고 source의 expected final state와 cleanup calls를 표로 전개한다. unhandled state가 있으면 review issue다.

## 21.3 EngineCore client는 process 경계에서 명령과 결과를 운반한다

### 21.3.1 command enqueue와 core admission은 다른 사건이다

AsyncLLM은 직접 scheduler method를 호출하지 않을 수 있다. EngineCore client가 IPC 또는 in-process transport를 통해 add, abort와 utility requests를 보낸다. 동일 Python function call처럼 보여도 process boundary와 serialization, queue ordering이 있다.

client abstraction과 concrete implementations는 [`vllm/v1/engine/core_client.py:1-360`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/core_client.py#L1-L360)에서 add_request, abort_requests, get_output과 shutdown을 읽는다. selector는 deployment에 맞는 client를 고르지만 이 장의 중심은 공통 request command와 output semantics다.

add command가 enqueue됐다는 사실과 core가 처리했다는 사실은 다르다. input queue put 시각, core receive, scheduler add를 나눈다. queue가 bounded인지, full일 때 await·block·error 중 무엇인지가 ingress backpressure를 정한다.

serialization은 request state snapshot을 만든다. API-side object를 이후 mutate해도 core가 보는 값이 바뀌지 않아야 한다. shared memory나 zero-copy를 쓴다면 lifetime과 ownership이 더 중요하다. source에서 message type과 serialization boundary를 확인한다.

command ordering도 본다. add 직후 disconnect로 abort가 들어오면 core가 add보다 abort를 먼저 볼 수 있는가. 같은 channel FIFO인지, 별도 queues인지, unknown ID abort를 저장하거나 drop하는지 확인한다. abort가 add보다 먼저 도착해 drop되고 add가 나중에 실행되면 ghost request가 생긴다.

core output에는 request ID와 token/logprob/finish/state updates가 있다. batched output 한 개가 여러 requests를 포함할 수 있다. API output handler는 ID별 stream으로 나눈다. core connection epoch와 output generation이 stale batch를 배제하는지 본다.

core failure는 pending requests 전체에 terminal error를 전달해야 한다. output handler task만 죽고 request iterators가 열린 채 남으면 clients가 무한 대기한다. client connection error, handler exception, shutdown이 pending streams를 어떻게 fail/close하는지 source를 잇는다.

backpressure는 양방향이다. ingress command queue가 차면 new requests admission이 늦어진다. output queue가 차면 core results drain이 늦어 scheduler progress나 memory에 영향을 줄 수 있다. slow HTTP client 하나가 shared output handler를 막지 않도록 request-local buffering과 global drain 구조를 확인한다.

관찰 field는 client type, core process/epoch identity, add enqueue/receive/schedule timestamps, abort enqueue/receive, output batch receive, per-request stream put, queue sizes와 terminal close다. source-only 단계에서는 hook 위치만 정의한다.

### add와 abort가 교차하는 세 시간표

첫 시간표는 client가 preprocessing 중 disconnect하는 경우다. 아직 local stream과 core request가 없다면 preprocessing coroutine만 cancel하면 된다. abort를 core로 보내도 unknown ID일 뿐이다. 하지만 tokenizer worker가 별도 process에서 계속 작업하면 CPU waste cleanup을 확인한다.

둘째는 local stream map에 등록했지만 add command를 보내기 전 disconnect하는 경우다. local entry를 지우고 iterator를 닫는다. core command가 아직 없다는 ordering이 증명되면 abort는 불필요하다. add send와 disconnect handler가 concurrent하면 lock이나 state check가 필요하다.

셋째는 add와 abort commands가 모두 transport에 들어간 경우다. same FIFO queue면 core가 add 뒤 abort를 본다. separate channels면 abort가 먼저 도착할 수 있다. core가 unknown-ID abort를 drop하고 나중 add를 받으면 ghost request가 된다. tombstone, ordering sequence 또는 same channel이 이를 막을 수 있다.

네 번째 변형은 core가 request를 active로 만들고 첫 output batch가 API로 오는 동시에 disconnect하는 경우다. output handler는 request stream에 put하려 하지만 consumer가 사라졌다. local map 제거와 put 순서가 race한다. put failure가 shared handler를 죽이지 않아야 하고 core abort는 여전히 전달되어야 한다.

다섯 번째는 final output과 abort 교차다. core가 정상 terminal을 만들었지만 API가 이를 받기 직전 client close를 감지한다. normal finish와 cancelled outcome 중 metrics가 무엇을 세는지 policy가 필요하다. resource free는 exactly once여야 한다.

source review에서는 request state를 보호하는 lock/event loop serialization과 map operations, core client queue semantics를 잇는다. Python event loop single-threaded라는 일반론만으로 awaits 사이 race가 없다고 말하지 않는다. await 지점에서 다른 coroutine이 map을 바꿀 수 있다.

race fixture는 deterministic barriers를 가정해 each boundary에서 disconnect를 삽입한다. expected command sequence, local map final state, core state와 output terminal을 적는다. runtime fault injection은 별도 승인이 필요하지만 source로 missing cleanup branch를 찾을 수 있다.

good outcome은 abort call success가 아니다. client output 없음, core request 제거, blocks free, local stream/map clean, handler alive와 subsequent request 정상이라는 다섯 조건이다. stale output count가 늘면 epoch/cleanup timing을 다시 본다.

### core command queue가 ingress admission을 바꾸는 순간

API는 request를 받아 tokenization까지 끝냈지만 core input queue가 가득 찰 수 있다. 이때 coroutine이 await하면 HTTP connections와 preprocessed prompts가 memory에 머문다. 즉시 reject하면 client가 retry할 수 있다. queue policy는 scheduler admission 이전의 admission layer다.

unbounded queue는 burst를 흡수하지만 tail과 memory를 제한하지 않는다. bounded queue는 capacity를 드러내지만 timeout/error mapping이 필요하다. queue size 하나만 보지 말고 oldest command age와 enqueue wait를 본다.

abort commands가 same queue 뒤에 줄 서면 overload 때 cancellation도 늦어진다. stale work가 계속 실행되어 pressure를 키운다. control command priority 또는 separate path가 있다면 ordering race를 함께 해결해야 한다. 빠른 abort와 add-before-abort consistency가 모두 필요하다.

utility/health requests가 same core channel을 쓰면 overload 때 health가 timeout되어 process를 재시작시키고 load를 악화할 수 있다. command types와 priority, timeout을 source에서 확인한다. health failure와 engine death를 구분한다.

input backpressure가 API에 전달되는 방식도 본다. await latency로만 나타나는지, explicit server-busy error인지, request timeout cancellation인지 다르다. client retry policy와 결합해 open-loop offered load를 키울 수 있다.

preprocessed prompt가 큰 request는 queue memory를 더 많이 차지한다. count bound만으로 byte bound가 되지 않는다. prompt IDs와 multimodal payload가 message에 복사되는지 shared reference인지 serialization source를 본다.

관찰은 API active tasks, preprocess-complete waiting-core, core command queue depth/age, scheduler waiting을 별도 둔다. 셋을 한 waiting count로 합치면 admission point를 못 찾는다. source hook이 없으면 request timestamps에서 구간을 파생할 수 있도록 ID를 보존한다.

overload recovery는 throughput만 보지 않는다. bounded memory, reject latency, cancel propagation, normal request goodput과 client retry rate를 본다. queue capacity를 늘려 error를 숨기면 tail과 memory가 악화될 수 있다.

## 21.4 output handler는 batched engine output을 request stream으로 되돌린다

EngineCore는 여러 requests를 함께 step으로 처리한다. output handler는 core output을 받아 request ID별 state와 stream에 전달한다. 이 층이 멈추면 GPU는 일하고 scheduler도 outputs를 내지만 API stream은 멈출 수 있다.

AsyncLLM initialization과 output handler loop는 [`async_llm.py:360-620`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/async_llm.py#L360-L620)에서 core client output read, output processor와 stream put을 연결한다. exact symbols와 task startup/shutdown을 고정 source에서 확인한다.

handler는 request-local output processor state를 갱신할 수 있다. incremental detokenization, logprobs와 finish, metrics가 intermediate output에서 누적된다. core token IDs와 API-visible text는 이 processor를 지나며 달라진다. stop trim과 stream delta는 상위 OpenAI serving과 역할을 나눌 수 있다.

한 core batch에 A와 B outputs가 있다. A stream consumer가 느리고 B는 빠르다고 하자. handler가 A queue put에서 block되면 B 전달도 늦는다. request-local queue가 unbounded면 handler는 빠르지만 A memory가 늘어난다. drop은 text correctness를 깨뜨린다. backpressure policy는 명시적 trade-off다.

output stream queue가 bounded인지, `put_nowait`인지, async await인지 source를 본다. API generator가 queue를 얼마나 빨리 drain하는지, client socket backpressure가 generator까지 전파되는지 잇는다. queue depth와 oldest output age가 관찰 field다.

terminal output은 finish reason, final metrics와 usage를 운반할 수 있다. token delta가 empty여도 stream close를 수행해야 한다. handler가 finished request를 map에서 먼저 삭제한 뒤 final output put에 실패하면 consumer가 terminal을 못 받는다. put·close·remove 순서와 exception cleanup을 확인한다.

unknown request ID output이 오면 stale epoch, early local cleanup, duplicate finish 또는 core bug 후보다. 단순 ignore가 안전할 수 있지만 왜 생겼는지 metric과 epoch를 남겨야 한다. stale output을 새 same-ID request에 전달하는 것은 더 위험하다.

handler task exception은 process-wide event다. 특정 request malformed output이 shared loop를 죽일 수 있는지, per-request error로 격리하는지 본다. loop가 죽으면 health/readiness와 pending streams가 함께 실패해야 한다. listener만 살아 있으면 new requests가 계속 들어와 hang한다.

API streaming assembler는 AsyncLLM output을 SSE chunks로 바꾼다. engine output receive와 generator yield, transport write를 별도 timestamp로 둔다. TTFT와 ITL이 model step이 아니라 output handler나 client backpressure에서 늘 수 있다.

### output queue backpressure 사건: A가 느리자 B도 멈춘다

동시에 두 requests A와 B가 생성 중이다. A client는 network read를 멈췄고 B는 정상이다. GPU trace에는 decode steps가 계속 있지만 B의 visible ITL이 갑자기 커졌다. scheduler fairness만 의심하기 전에 output delivery graph를 본다.

core는 A와 B outputs를 한 batch message로 보낼 수 있다. handler가 A request-local queue에 blocking put을 하고 queue가 full이면 그 coroutine은 B output까지 진행하지 못한다. head-of-line blocking이다. 반대로 nonblocking unbounded put이면 B는 정상이나 A queue memory가 계속 증가한다.

request-local queue size 1은 strong backpressure를 주지만 shared handler 구조에서는 위험하다. queue를 크게 하면 burst를 흡수하지만 slow client가 오래되면 결국 찬다. handler와 transport 사이 per-request relay task를 두면 shared drain을 분리할 수 있지만 task와 buffer 수명이 늘어난다.

어느 설계든 명시적 limit과 timeout, cancel policy가 필요하다. output을 drop하면 text와 logprobs가 깨지므로 허용하기 어렵다. slow client를 abort한다면 error/connection close 의미와 engine cleanup을 연결한다. memory를 허용하면 global/tenant quota가 필요하다.

first divergence timing을 나눈다. core output batch receive는 정상인데 B stream put이 늦으면 handler이다. B stream get은 정상인데 API yield가 늦으면 assembler이다. yield는 정상인데 client receive가 늦으면 transport/proxy이다. GPU step gap이 먼저 늘면 scheduler/runner로 돌아간다.

output processor 자체가 CPU-heavy할 수도 있다. top logprobs JSON, detokenization, tool parser가 event loop를 오래 점유하면 모든 request output가 늦는다. queue blocking과 CPU service time을 나눈다. processor별 duration과 event loop lag가 필요하다.

usage/metrics aggregation이 shared lock을 잡는지, callback이 sync logging을 하는지도 본다. debug log를 켠 뒤 ITL이 악화되면 observer effect다. full token text logging은 개인정보와 CPU/IO 비용을 만든다.

terminal의 우선순위도 중요하다. A queue가 full인데 abort terminal을 넣지 못하면 cleanup이 막힌다. queue를 clear하고 terminal을 넣는지, out-of-band close signal이 있는지 source를 확인한다. data order와 terminal exactly-once를 보존해야 한다.

negative fixture는 A queue가 full인 상태에서 B intermediate와 terminal, A abort, core shutdown을 순서대로 넣는다고 생각한다. shared loop가 계속되고 B가 terminal을 받으며 A map이 cleanup되는지 source state를 전개한다.

성능 최적화는 backpressure를 없애는 것이 아니라 어디에 둘지 선택한다. network보다 생성이 빠르면 어딘가 buffer나 속도 제어가 필요하다. service는 memory, latency, fairness와 cancel semantics를 함께 설계해야 한다.

### output handler exception은 request-local인가 engine-wide인가

request A의 malformed logprob object가 output processor assertion을 일으켰다고 하자. handler loop가 batch 전체를 처리하는 중이라 B outputs도 같은 call stack에 있다. exception boundary가 batch 밖이면 shared task가 죽고 모든 pending streams가 hang할 수 있다.

per-request `try`로 A를 error/abort 처리하고 B를 계속할 수 있는지 source를 본다. 그러나 corrupted core batch나 connection decode error처럼 global failure는 engine-wide가 맞다. error scope를 원인과 일치시킨다.

handler task가 죽었을 때 done callback이나 health monitor가 exception을 관찰하는지 확인한다. task exception이 event loop log에만 남고 readiness가 green이면 new requests가 계속 들어온다. pending queue와 map이 memory leak된다.

engine-wide failure handler는 pending request iterators 각각에 terminal error를 보내야 한다. consumer가 queue get에서 영원히 기다리지 않아야 한다. map cleanup과 error delivery 순서는 정상 final과 같은 ownership 규칙을 따른다.

per-request failure는 core abort까지 이어져야 한다. API output serialization이 불가능해도 scheduler work를 계속할 이유가 없다. abort send 자체가 core failure 때문에 실패할 수 있으므로 best-effort cleanup과 process teardown을 구분한다.

error object에 request ID와 core epoch, first bad field를 남기되 raw text/logprob 전체를 기록하지 않는다. malformed external data인지 internal invariant인지 분류한다. security-sensitive cross-request ID mismatch는 process-wide stop을 고려할 수 있다.

## 21.5 abort는 HTTP disconnect에서 KV 반환까지 이어져야 한다

### 21.5.1 disconnect signal과 resource retirement를 잇는다

client가 첫 content 뒤 연결을 끊었다. API coroutine은 disconnect 또는 generator cancellation을 보고 AsyncLLM abort를 호출한다. 하지만 abort method 반환이 scheduler removal과 KV free 완료를 뜻하는지는 source contract를 읽어야 한다.

vLLM OpenAI serving generator의 disconnect handling과 AsyncLLM abort를 [completion stream path](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/completion/serving.py#L330-L700), [`async_llm.py`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/async_llm.py#L200-L420), core client abort command에서 잇는다.

event chain은 client close, API detection, abort call, core command enqueue, core receive, scheduler remove, in-flight drain, block free, local output stream close다. 어느 두 사건도 자동으로 같은 timestamp가 아니다.

abort idempotency가 필요하다. disconnect poll과 generator `finally`가 둘 다 abort할 수 있고 timeout manager도 호출할 수 있다. unknown/finished ID abort가 harmless여야 한다. 중복 command가 새 same-ID request를 취소하지 않도록 incarnation 또는 uniqueness를 유지한다.

add/abort race를 다시 보자. local stream은 등록됐지만 add command가 아직 core queue에 있을 때 abort한다. 같은 FIFO channel이면 order가 보존될 수 있다. 별도 path면 early abort tombstone이 필요할 수 있다. source에서 실제 ordering guarantee를 찾는다.

running request abort는 현재 model step을 즉시 중단하지 못할 수 있다. scheduler가 next safe point에서 제거하고 runner output을 버릴 수 있다. cancelled tokens executed metric은 waste를 보여 준다. abort latency가 길다고 무조건 bug는 아니지만 bound와 owner를 알아야 한다.

waiting request는 GPU work 없이 제거할 수 있다. running과 waiting abort latency를 같은 histogram에 합치면 해석이 흐린다. preempted, connector transfer waiting, structured state 같은 lifecycle도 분류한다.

local output stream cleanup이 너무 이르면 core의 late output이 unknown ID가 된다. 너무 늦으면 map과 buffers가 leak된다. terminal/abort ack가 있는지, best-effort timeout으로 제거하는지 source를 확인한다. API disconnect만으로 local consumer는 사라졌지만 core cleanup은 별도다.

비정상 client가 느리게 읽을 뿐 disconnect하지 않았다면 abort하면 안 된다. slow consumer timeout policy가 있다면 transport timeout과 engine timeout을 구분한다. buffer pressure가 다른 requests에 퍼지지 않게 한다.

취소 incident는 access log만 보지 않는다. request ID로 모든 events와 scheduler/KV metrics를 묶는다. 실행하지 않는 이 장에서는 event schema와 source hook을 정의한다.

## 21.6 failure와 cleanup은 add 전·중·후가 서로 다르다

첫 failure는 preprocessing 전후 validation error다. core request가 없으므로 abort를 보낼 필요가 없다. local metrics와 HTTP error만 닫는다. prompt tokenization이 부분적으로 resource를 잡는다면 worker cleanup은 별도다.

둘째는 local stream 등록 뒤 core add 전송 실패다. stream map을 제거하고 iterator에 terminal error를 전달해야 한다. core가 request를 못 받았다는 증거가 불확실하면 best-effort abort도 필요할 수 있다.

셋째는 core add 성공 뒤 API assembler failure다. model work가 존재하므로 abort해야 한다. tool parser나 JSON serialization error라고 engine이 자동 멈추지 않는다. generator exception cleanup을 본다.

넷째는 core process failure다. 모든 pending requests에 error terminal을 보내고 readiness를 내리며 reconnect/restart policy를 실행해야 한다. output handler task가 silently 죽어 listener만 살아 있으면 worst hang이 된다.

### cancel flood가 정상 request까지 굶기는 과정

많은 clients가 짧은 timeout으로 끊으면 abort commands가 폭증한다. 각 request는 output stream cleanup, core command, scheduler removal과 block free를 요구한다. cancel은 공짜 control event가 아니다.

abort와 add가 같은 queue를 쓰면 normal admission이 늦을 수 있다. scheduler finished-ID lookup, late output drop, metrics와 logging도 CPU를 쓴다. cancelled work가 device에서 더 실행되면 compute pressure도 남는다.

logical cancels와 physical abort commands를 나눈다. disconnect poll, generator finally와 timeout이 같은 request에 중복 abort를 보낼 수 있다. core idempotency가 correctness를 지켜도 command load는 늘어난다. API deduplication과 core idempotency는 다른 보호다.

unknown ID abort가 많으면 add/abort race, duplicate cleanup, stale epoch를 의심한다. restart 직후 expected stale IDs와 steady-state anomaly를 구분한다. request incarnation과 command sequence가 필요하다.

관찰에는 disconnects, unique cancelled IDs, abort commands, core receives, scheduler removals, cancelled executed tokens, KV free latency와 normal add wait를 둔다. cancel success 하나로는 영향 범위를 알 수 없다.

회복은 timeout을 무조건 늘리는 일이 아니다. client deadline을 SLO에 맞추고 overload admission을 빠르게 reject하며 abort path를 bounded하게 만든다. normal requests fairness와 safe free를 함께 검증한다.

다섯째는 duplicate request ID다. 기존 active stream을 덮으면 두 consumers와 one core request의 identity가 깨진다. reject, namespace, generated suffix 중 policy를 확인한다. retry correlation과 physical uniqueness를 나눈다.

여섯째는 output processor exception이다. malformed logprobs나 detokenization state가 한 request를 실패시킬 수 있다. shared handler loop가 계속되는지, request abort와 stream error가 exactly-once인지 본다.

failure matrix는 exception 이름 목록이 아니라 state 기준이다. local registered 여부, core accepted 여부, scheduler state, output committed 여부, terminal delivered 여부와 resources freed 여부다. 같은 exception도 발생 시점에 따라 cleanup이 다르다.

rollback order는 new admission 차단, pending stream failure, core abort/drain, handler stop, client/process teardown으로 생각할 수 있다. 실제 shutdown source가 어떤 순서를 쓰는지 다음 절에서 확인한다.

### core crash 사건: listener는 살아 있고 stream은 멈춘다

EngineCore process가 fatal exception으로 종료됐다고 하자. API process와 HTTP listener는 살아 있다. active clients는 마지막 delta 뒤 기다리고 new requests도 route validation을 통과한다. handler가 receive error를 만났지만 readiness가 이를 반영하지 않으면 겉으로만 healthy하다.

첫 관찰은 core process/connection liveness와 output-handler task다. scheduler metrics와 output batches가 동시에 멈췄는지 본다. GPU idle만으로 no-work와 core-dead를 구분할 수 없다.

local stream map에는 pending IDs가 남아 있다. 이 iterators에 terminal engine error를 넣고 map을 cleanup해야 한다. partial stream은 HTTP status를 바꿀 수 없으므로 error event나 close가 보일 수 있고, non-stream은 아직 headers 전이면 5xx를 만들 수 있다.

automatic restart가 있더라도 old pending requests를 무심코 replay하지 않는다. partial output, sampler/constraint state와 cache가 사라졌다. exact resume state가 없다면 transparent continuation을 약속하기 어렵다. fail-and-retry가 더 명확할 수 있다.

new core가 ready되기 전 admission을 열지 않는다. model load와 worker health, client connection을 확인한다. old pending IDs는 terminal이 되고 new epoch namespace와 격리되어야 한다.

crash evidence는 exit code, last command/output sequence, pending counts, worker health를 남긴다. raw prompts는 필요하지 않을 수 있다. restart가 evidence를 덮어쓰지 않게 한다.

source에서는 core receive exception이 AsyncLLM handler task와 pending requests로 전파되는 길, supervisor와 shutdown owner를 잇는다. listener health만 검사하는 endpoint라면 engine readiness signal을 추가로 찾는다.

negative fixture는 zero/partial stream, non-stream wait, add in-flight, abort in-flight 상태에서 connection close를 삽입한다고 생각한다. expected는 all pending terminal, maps empty, readiness false, stale outputs 없음과 safe new epoch다.

### output commit과 engine state는 직교한다

non-stream은 headers/body 전까지 error status로 바꿀 수 있다. stream은 first SSE 뒤 success headers가 이미 나갔다. output handler failure의 public 표현은 transport commit state에 따라 달라진다.

commit state를 `NOT_STARTED`, `HEADERS_SENT`, `PARTIAL_BODY`, `TERMINAL_SENT`, `CLOSED`로 생각하자. engine은 동시에 queued, active, finished, aborted, failed 중 하나다. engine finished+transport partial, engine active+transport closed가 모두 가능하다.

transport closed와 engine active면 abort가 필요하다. 둘 다 terminal이면 normal cleanup이다. transport error와 engine terminal이면 delivery failure를 기록하되 work cleanup은 정상일 수 있다. engine error와 headers sent면 stream error/close policy가 필요하다.

metrics는 engine completed와 client-delivered success를 분리한다. core throughput이 정상인데 API success가 낮으면 output/transport owner다. partial streams를 completed request로만 세면 goodput을 과장한다.

TCP close 전에 bytes 일부가 전달됐는지 server가 정확히 모를 수 있다. retry는 duplicate prefix를 만들 수 있으므로 application idempotency가 필요하다. server request ID가 exactly-once external effect를 보장하지 않는다.

## 21.7 shutdown은 listener 종료보다 긴 engine lifecycle이다

### 21.7.1 관측되지 않은 teardown edge는 근거 공백으로 남긴다

배포가 SIGTERM을 받았다. HTTP listener만 닫으면 new requests는 막히지만 active streams, output handler와 EngineCore process가 남을 수 있다. 반대로 core를 먼저 죽이면 active clients가 terminal 없이 hang한다. shutdown ordering이 서비스 계약이다.

AsyncLLM shutdown과 context manager는 [`async_llm.py:620-820`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/async_llm.py#L620-L820), core client shutdown은 [`core_client.py:300-520`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/core_client.py#L300-L520)에서 handler task, pending requests와 process teardown 순서를 읽는다.

graceful shutdown은 readiness를 내려 new traffic을 막고 load balancer drain을 기다린다. active requests를 일정 deadline까지 완료하거나 abort한다. output buffers와 terminal을 flush하고 EngineCore/worker를 닫는다. process와 IPC resources를 정리한다.

deadline이 지나면 forced abort로 전환한다. client에게 어떤 error/connection close가 보이는지, KV와 shared memory가 process exit로 정리되는지 확인한다. graceful과 forced metrics를 분리한다.

output handler를 너무 일찍 cancel하면 core의 final outputs를 못 읽고 streams가 hang한다. 너무 늦게 두면 core client가 닫힌 뒤 read loop가 exception을 반복할 수 있다. task cancellation을 await하고 exception을 consume하는지 본다.

shutdown 중 new add race도 있다. readiness는 내려갔지만 이미 accepted HTTP request가 preprocessing 중일 수 있다. AsyncLLM이 shutting-down flag로 add를 reject하는지, caller가 명확한 error를 만드는지 확인한다.

restart 뒤 old output과 new request를 격리해야 한다. IPC endpoint와 core epoch가 새로 만들어지고 pending old streams는 terminal error가 되어야 한다. old request ID를 retry해도 stale output이 전달되지 않는다.

health endpoint가 listener만 보는지 engine handler/core liveness까지 보는지 확인한다. output handler가 죽었는데 health가 green이면 requests가 hang한다. readiness와 liveness의 owner가 다를 수 있다.

shutdown audit은 signal receive, admission close, active count, abort/drain, final stream, handler stop, core exit와 resource cleanup timestamps를 정의한다. source-only 단계에서는 ordering을 증명한다.

### graceful deadline을 실제 state로 계산한다

grace 30초를 설정했다고 하자. 단순히 sleep한 뒤 process를 죽이는 것은 graceful contract가 아니다. new admission을 막고 active engine requests와 active transports가 어떻게 줄어드는지 보며 deadline에 남은 work를 abort해야 한다.

readiness를 먼저 내려도 load balancer propagation 동안 새 connections가 들어올 수 있다. listener가 accepted했지만 preprocessing 중이거나 AsyncLLM add 직전인 requests를 포함해야 한다. admission gate가 어느 boundary에 있는지 본다.

slow clients가 output queues를 막으면 engine은 finish해도 transport-active count가 남는다. generation deadline과 terminal flush deadline을 분리할 수 있다. slow consumer policy 없이 무한 drain을 약속하지 않는다.

deadline 직전 normal finish와 forced abort가 race한다. block free와 map removal, final chunk가 exactly once여야 한다. shutdown task와 handler가 같은 request map을 수정하는 ordering을 확인한다.

core shutdown 뒤 handler가 읽을 final batches가 있는지 contract를 본다. EOF/sentinel로 loop를 끝내는지 task cancellation인지 다르다. EOF 전 pending streams를 success로 가정하지 않는다.

completion evidence는 listener closed 하나가 아니다. admission 0, preprocessor pending 0, stream map 0, core pending 0, handler stopped, core exited와 IPC cleanup을 본다. hard requirements는 deployment policy로 고정한다.

rolling deploy에서는 old instance가 drain하며 new instance가 ready된다. IDs와 metrics에 instance/epoch를 넣어 두 세대 events를 섞지 않는다. partial stream을 load balancer가 새 instance로 resume할 수 있다고 가정하지 않는다.

## 21.8 대표 request workbook과 다음 scheduler handoff

이 workbook의 목적은 모든 필드를 한번에 채우는 것이 아니다. `Hello` 요청 하나가 API-only state에서 engine-owned state로 넘어가고, token output이 다시 stream으로 돌아오는 정상 경로를 먼저 닫는다. 그 다음 같은 identity에 cancel, failure, backpressure를 하나씩 대입한다.

### 21.8.1 정상 request와 output 경로

대표 fixture는 text `Hello`, deterministic sampling, output cap 4, stream true다. raw body부터 prompt IDs와 params, request ID를 적는다. AsyncLLM stream map과 core add command, scheduler admission 전 handoff까지 잇는다.

output fixture는 intermediate token 두 개와 final empty delta를 포함한다고 생각한다. core output IDs, output processor delta, request queue put, API generator yield, finish chunk를 적는다. empty terminal을 버리지 않는다.

#### cancel·failure로 owner 회수를 검증한다

cancel fixture는 first content 뒤 disconnect다. API detection부터 abort enqueue, scheduler remove와 stream cleanup owner를 source로 표시한다. 실제 times는 비워 둔다.

failure fixture는 core add send exception과 output handler exception 두 개다. local map과 pending iterator가 어떻게 닫히는지 확인한다. exception이 shared handler 전체를 죽이는지 본다.

#### backpressure로 공유 owner의 영향을 검증한다

backpressure fixture는 A slow consumer와 B normal consumer다. shared handler가 A 때문에 B를 막는지 queue semantics로 판정한다. unbounded queue라면 memory bound와 cancellation policy를 찾는다.

관찰 장부는 request ID/incarnation, API arrival, preprocess start/end, add enqueue/core receive, scheduler state, output batch receive, stream put/get, transport yield, abort/remove/free와 epoch를 갖는다. request content는 synthetic 또는 hash다.

#### 사건 분기 뒤에 여는 소스 지도

소스 지도는 다음 다섯 의미 경계를 순서대로 연결한다.

- HTTP 요청의 도착점은 [completion route](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/completion/api_router.py#L1-L70)다.
- 프로토콜을 내부 요청으로 바꾸는 과정은 [serving conversion](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/completion/serving.py#L120-L330)과 [preprocess](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/inputs/preprocess.py#L1-L291)에서 확인한다.
- 비동기 수명 주기의 소유자는 [AsyncLLM](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/async_llm.py#L150-L820)이다.
- 엔진 프로세스와 맞닿는 경계는 [core client](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/core_client.py#L1-L520)에서 닫힌다.

### 21.8.2 scheduler로 넘기기 전의 출구 질문

이 장의 출구 질문은 다섯 가지다. request는 어느 시점에 API-only에서 engine-owned가 되는가. request ID와 core epoch는 output과 abort를 어떻게 격리하는가. output handler가 batched output을 어느 stream에 commit하는가. disconnect가 언제 scheduler resource free로 닫히는가. shutdown이 pending requests를 어떻게 terminal로 만드는가.

다음 scheduler 장으로 넘길 것은 core가 accepted한 normalized request, arrival/priority, prompt/output token budgets와 request ID다. API validation 전에 실패하면 19장 owner다. core add/stream lifecycle이면 이 장 owner다. scheduler waiting/running/preemption부터는 다음 장 owner다. 이 경계가 있어야 ingress 문제를 CUDA나 scheduler 탓으로 돌리지 않는다.

### 함수 산책 1: route에서 첫 core command까지

독자가 source를 실제로 열었다고 하자. route decorator와 endpoint function은 짧다. 여기서 오래 머물 필요는 없다. typed request와 raw HTTP request를 어느 serving object method에 넘기는지, response가 error object인지 generator인지 분기하는 지점만 표시한다.

serving method로 내려가면 resolved model, request ID와 sampling conversion을 찾는다. prompt가 list인지 text인지, multiple prompts가 one/many engine requests가 되는지 확인한다. representative single text request path를 먼저 따라가고 batch prompt variation은 그 뒤에 둔다.

prompt preparation helper에서 tokenizer 호출 consumer를 찾는다. raw text, explicit token IDs와 multimodal inputs의 union을 어떻게 나누는지, tokenizer group이나 engine client를 await하는지 본다. prompt length validation이 tokenization 전 character length가 아니라 final IDs를 기준으로 하는지 확인한다.

sampling conversion에서 raw API fields를 effective `SamplingParams` 같은 engine object로 옮긴다. EOS, stop, logprobs, output kind와 detokenization flags를 적는다. params object가 immutable snapshot인지 caller가 이후 mutate할 수 있는지 본다.

AsyncLLM `generate` caller에서 request ID와 engine prompt, params, adapters와 priority가 어떻게 넘는지 적는다. 반환 async iterator를 stream/full response code가 어떻게 consume하는지 연결한다. iterator creation과 first iteration side-effect timing을 확인한다.

`generate` 내부에서는 local request state 또는 stream creation, `add_request`, iteration loop와 `finally` abort/cleanup을 순서대로 본다. 함수가 짧더라도 helper로 위임된 state mutation을 따라간다. public method signature만으로 ownership을 추측하지 않는다.

core client `add_request`에서는 message object와 transport send를 본다. in-process와 multiprocess implementations가 공통 semantic을 지키는지 본다. selector exception catalogue보다 add command의 identity, ordering과 failure cleanup을 먼저 고정한다.

이 산책의 결과는 call graph 그림보다 state ledger다. API-only ID, prepared prompt, local stream registered, core command sent가 어느 line에서 바뀌는지 적는다. 다음 함수로 넘어갈 때 owner와 failure rollback을 한 문장으로 남긴다.

### 함수 산책 2: core output에서 API iterator까지

반대 방향으로 읽으면 output ownership이 보인다. core client가 batched engine output을 받는 method를 찾고 AsyncLLM handler loop가 await하는 곳을 찾는다. message가 empty/sentinel/error일 수 있는지 type definition을 본다.

handler는 output processor에게 batch를 넘기거나 request별로 반복한다. processor가 token IDs와 finish, metrics를 어떻게 stateful output으로 바꾸는지 확인한다. request ID가 local stream map lookup key가 되는 line을 표시한다.

stream put과 map removal ordering을 본다. final output을 consumer가 볼 수 있게 한 뒤 close하는지, close signal이 final object와 별도인지 확인한다. consumer iteration은 queue get 결과에서 terminal을 언제 판단하는지 본다.

OpenAI serving generator는 Engine output iterator를 `async for`로 소비한다. intermediate output에서 previous text/token offset을 유지해 delta를 만들고 final에서 finish/usage를 붙인다. disconnect poll이 loop 안인지 exception/finally인지 찾는다.

non-stream path는 outputs를 aggregate한다. client disconnect를 같은 방식으로 관찰하는지, result count와 choices를 어떻게 완성하는지 비교한다. stream path만 봐서는 full request ghost work를 놓칠 수 있다.

output error가 어느 exception class로 API까지 올라오는지 본다. headers 전에는 structured error response가 가능하고 generator 안에서는 stream close/error가 될 수 있다. core error type을 public error로 직접 노출하는지 sanitize하는지 확인한다.

이 역방향 산책을 request ID로 정방향 산책과 연결하면 수명이 닫힌다. add command의 ID와 output map ID, abort ID, log correlation이 같은 logical request를 가리켜야 한다. child/choice identity가 있다면 mapping을 별도 적는다.

### lifecycle invariant를 수식처럼 고정한다

첫 invariant는 active ID uniqueness다. 한 AsyncLLM instance와 core epoch 안에서 active internal request IDs는 중복되지 않는다. map overwrite를 허용하지 않는다. external correlation ID는 중복될 수 있어도 internal incarnation은 달라야 한다.

둘째는 terminal exactly once다. 정상 finish, abort와 error 중 하나만 request terminal owner가 된다. duplicate cleanup calls는 idempotent할 수 있지만 consumer-visible terminal과 resource free는 한 번이다.

셋째는 no output after cleanup이다. local stream map에서 request를 제거한 뒤 같은 epoch output을 새 request에 전달하지 않는다. late output은 stale로 식별하고 drop/metric 처리한다.

넷째는 no orphan core work다. API consumer가 영구히 사라지고 request가 terminal이 아니라면 abort가 core에 전달되어야 한다. transport error 때문에 확정할 수 없으면 process failure policy가 pending work를 닫는다.

다섯째는 no hanging consumer다. core/handler failure와 shutdown은 pending iterators에 terminal error 또는 close를 전달한다. queue get이 무한히 기다리지 않는다.

여섯째는 bounded buffering이다. 명시적 bound가 없다면 admission/slow-client policy로 effective bound를 제공한다. output correctness를 위해 drop하지 않는다. shared handler의 progress가 한 slow consumer에 영구 종속되지 않는다.

일곱째는 epoch isolation이다. core restart 전 outputs와 pending state가 new epoch requests에 연결되지 않는다. old streams는 명시적으로 fail하거나 supported replay protocol을 따른다.

여덟째는 cleanup ordering이다. scheduler/runner가 더 이상 buffer와 KV를 읽지 않는 safe point 뒤 resource를 재사용한다. API map cleanup 시점과 device resource free 시점은 다르지만 각각 owner가 명확하다.

이 invariants는 runtime test 이름이 아니다. source review 질문과 incident branch를 만든다. violation 하나가 보이면 해당 owner의 map, queue, exception과 ordering으로 좁힌다.

### 장애 사건: 첫 요청은 끝났는데 같은 ID retry가 옛 답을 받는다

request `x`가 timeout으로 client에서 사라졌고 바로 같은 external ID로 retry했다. new request 첫 chunk가 old request tail처럼 보인다고 하자. model nondeterminism보다 identity reuse와 late output을 먼저 본다.

old attempt의 internal ID와 new attempt internal ID가 같은지 확인한다. local map removal timestamp, old core final receive와 new registration 순서를 적는다. 같은 key로 map이 재등록된 뒤 old output이 도착하면 contamination이 가능하다.

경쟁 가설은 실제 old output misrouting, client가 old buffered chunk를 retry UI에 붙임, deterministic model의 coincidental suffix다. server event sequence와 client attempt IDs로 나눈다. core output의 epoch/attempt tag가 strongest evidence다.

server-generated unique internal ID를 쓰고 external ID는 metadata로만 유지하면 충돌을 줄일 수 있다. 그렇더라도 client UI가 logical ID로 streams를 merge하면 application bug가 남는다. 양쪽 identity를 보존한다.

late output drop은 정상 방어일 수 있지만 빈도가 높으면 abort/cleanup latency 문제다. drop count를 숨기지 않고 request state와 epoch별 metric으로 둔다. new request에 deliver하지 않는 것이 최우선이다.

복구 fixture는 old request를 add 후 abort하고 core final을 지연시킨 뒤 same external ID retry를 등록하는 timeline이다. expected는 internal IDs distinct, old output stale drop, new stream은 new outputs만 받고 both attempts cleanup이다.

### 장애 사건: handler는 살았지만 stream map이 계속 커진다

traffic가 끝나도 AsyncLLM request stream map size가 내려가지 않는다고 하자. GPU와 scheduler active는 0이다. resource leak은 engine보다 local output lifecycle에 있다.

정상 finish path에서 final output이 map removal을 유발하는지, consumer가 iterator를 끝까지 읽지 않았을 때 generator close/finally가 cleanup하는지 본다. client가 first result만 읽고 iterator를 버릴 수 있는 library use도 고려한다.

error path와 cancel path에서 같은 removal helper를 쓰는지, duplicate removal exception 때문에 cleanup이 중단되는지 본다. weak reference나 garbage collection에 의존하면 시점이 비결정적일 수 있다.

map entry가 stream queue와 large output history를 참조하면 작은 ID leak이 memory leak으로 커진다. entry age, terminal flag, queue depth와 last event를 관찰한다. active/finished/orphan categories를 나눈다.

forced timeout cleanup은 safety를 확인한다. core가 아직 output을 보낼 수 있는데 map key를 새 request가 재사용하면 안 된다. stale epoch/unique ID가 방어한다. core abort와 local removal의 ordering을 기록한다.

복구는 map size가 0이라는 순간 snapshot만 보지 않는다. 정상, error, disconnect, consumer early-break, core crash와 shutdown paths 모두 eventual cleanup되고 handler progress가 유지되는지 본다.

### 1.2초 TTFT를 ingress 구간으로 해부한다

대표 incident를 숫자로 풀어 보자. request receive가 0ms, schema validation 완료 3ms, tokenizer work enqueue 5ms, tokenizer start 205ms, finish 265ms, core add enqueue 270ms, core receive 300ms, scheduler run 330ms, first core output 520ms, handler stream put 525ms, API yield 530ms, client receive 1,200ms라고 하자. 이 값은 설명을 위한 예시이지 실측 결과가 아니다.

200ms tokenizer queue, 60ms service, 30ms core command queue, 30ms scheduler wait, 190ms first execution/output, 5ms handler, 670ms transport/proxy가 보인다. 1.2초를 모두 model prefill로 부르면 가장 큰 원인인 transport를 놓친다. GPU kernel 최적화로 줄일 수 있는 범위도 190ms 구간 일부뿐이다.

같은 timestamp를 어느 clock에서 얻었는지 주의한다. API process와 core process clocks가 monotonic but not synchronized일 수 있다. IPC message에 sender timestamp를 넣고 receive local timestamp를 함께 기록하거나 clock offset을 교정한다. wall clock jumps를 latency 계산에 쓰지 않는다.

first core output이 token 없는 metadata update일 수 있다면 first token timestamp와 구분한다. handler stream put도 API-visible content가 아닐 수 있다. role-only chunk와 first content를 나눈다. TTFT 정의를 dashboard와 client benchmark에 맞춘다.

transport gap은 proxy buffering, client slow read, network와 API coroutine scheduling으로 다시 나눈다. server yield timestamp가 530ms인데 socket write timestamp가 없다면 gap attribution은 아직 indirect다. access log end time만으로 first-byte flush를 알 수 없다.

이 예시는 lifecycle ledger가 latency ledger이기도 함을 보여 준다. request identity가 process boundaries를 지나야 timestamps를 잇고, output commit state가 있어야 first visible event를 정한다. correctness 소유권과 performance 분석이 같은 state machine을 공유한다.

### queue가 많다는 말의 네 가지 다른 뜻

API active tasks는 HTTP request를 받았지만 아직 response가 닫히지 않은 수다. preprocessing queue는 tokenization/processing을 기다리는 work다. core command queue는 prepared request가 EngineCore에 전달되기를 기다리는 곳이다. scheduler waiting queue는 core가 accepted했지만 resources/turn을 기다리는 requests다.

네 counts가 같은 request를 동시에 포함할 수도 있다. API active는 전체 lifecycle 동안 남고 scheduler waiting은 일부 phase다. 따라서 합해서 total queued라고 하면 double count한다. phase-exclusive gauge와 inclusive active gauge를 구분한다.

API active만 높고 preprocessor/core/scheduler가 낮으면 slow clients와 output aggregation을 본다. preprocess waiting만 높으면 CPU worker, core command만 높으면 IPC/core input drain, scheduler waiting만 높으면 token/KV/resource admission을 본다.

output queue는 fifth pressure point다. request가 scheduler active/finished인데 client delivery를 기다릴 수 있다. generation work와 delivery work가 서로 다른 방향으로 flow한다. input/output backpressure를 같은 queue utilization로 표시하지 않는다.

queue age distribution이 count보다 중요할 때가 많다. 신규 requests가 계속 빠르게 통과하지만 하나가 orphan되어 count 1이 영원히 남을 수 있다. oldest age, percentiles와 state reason을 둔다.

boundedness도 단위가 다르다. request count, token count, bytes, queue items가 있다. 큰 prompt 하나와 짧은 prompt 백 개가 같은 count trade-off가 아니다. core scheduler token budget과 API task count를 직접 비교하지 않는다.

## 21.9 readiness·근거 범위·scheduler handoff를 닫는다

HTTP process가 event loop를 돌고 있다는 liveness와 new request를 safely accept할 readiness는 다르다. core client disconnected, output handler dead, model loading/shutdown 중이면 listener가 살아도 ready가 아니다.

readiness 조건은 core connection, handler task, engine health와 shutdown/admission flag를 포함할 수 있다. scheduler overloaded이지만 healthy한 경우는 ready를 유지하고 admission reject를 할 수도 있다. overload와 component failure를 구분한다.

health check가 core command queue를 왕복한다면 overload 때 느려질 수 있다. timeout이 process restart를 일으켜 worse cascade를 만들 수 있다. lightweight local state와 deeper engine probe를 나누는 설계가 가능하다.

handler task exception을 done callback이 관찰하는지, exception state가 readiness에 반영되는지 source에서 본다. pending streams만 fail하고 health는 green으로 돌아오는 restart mechanism이 있다면 epoch transition을 확인한다.

shutdown은 readiness false로 먼저 시작한다. new add가 reject되어도 already accepted preprocessing tasks가 core add로 들어갈 수 있으므로 AsyncLLM 자체 admission flag가 필요할 수 있다. route-level gate만으로 race를 막는다고 단정하지 않는다.

readiness incident report는 endpoint response뿐 아니라 core epoch, handler task state, pending streams와 last successful output receive를 포함한다. green/false 한 bit보다 왜 accept 가능한지 설명한다.

### 21.9.1 source 좌표를 넓은 링크로 끝내지 않는다

이 초안의 링크 범위는 독자가 고정 source를 찾아갈 입구다. 최종 source note에서는 각 주장에 exact symbol과 좁은 lines를 다시 고정한다. route delegation, prompt preprocessing call, `generate`, `add_request`, `abort`, handler loop, stream class와 shutdown이 각각 owner다.

route file은 endpoint가 serving object에 위임한다는 사실만 증명한다. prompt tokenization은 preprocessing/helper source가 증명한다. AsyncLLM source는 local stream map과 core client calls를, core client는 transport semantics를 증명한다. scheduler removal과 KV free는 이후 scheduler chapter source가 owner다.

output handler가 존재한다는 사실로 backpressure 결과를 단정하지 않는다. queue implementation과 put/get semantics를 확인하고 실제 queue wait는 runtime 관찰이 필요하다. source-only 장은 가능한 block path와 hooks를 증명한다.

core client selector가 여러 implementation을 지원한다는 사실은 대표 request lifecycle을 흐리지 않는다. 공통 abstract contract를 먼저 쓰고 current deployment의 concrete type은 runtime config와 initialization source로 확인한다. 예외 catalog는 source note로 보낸다.

source version이 바뀌면 file path만 search하지 않는다. route literal, serving call, preprocessing consumer, `generate/add_request`, output receive, abort와 shutdown의 의미 좌표를 다시 찾는다. 이름이 이동해도 request state chain을 재구성할 수 있다.

### 21.9.2 최종 incident handoff의 문장 형태

나쁜 인계는 “vLLM stream이 가끔 멈춘다”다. 좋은 인계는 “core epoch E에서 request R의 core outputs는 14:03:02.120까지 수신됐고 handler task는 alive지만 request-local queue put이 30초 blocked되어 같은 core batch의 request S stream put도 지연된다”다. state, identity와 first divergence가 있다.

또 다른 좋은 문장은 “client disconnect는 100ms에 관찰됐고 AsyncLLM abort command는 103ms에 같은 ID로 enqueue됐지만 core receive가 add command보다 앞서 unknown-ID로 drop되며 이후 add가 active가 된다”다. add/abort ordering owner가 보인다.

core crash 인계는 “listener와 event loop는 alive지만 core connection epoch E가 closed되고 output handler task가 exception terminal이다. pending stream map 42 entries에는 terminal이 전달되지 않았고 readiness는 여전히 true다”처럼 쓴다. health propagation과 cleanup이 owner다.

source-only evidence에는 line links와 expected transition을 붙이고 runtime 미관찰은 분명히 한다. 예시 timeline을 실제 incident 수치로 오해하지 않게 label한다. 재현 command나 model execution은 이 장에서 수행하지 않는다.

수정 검증도 같은 문장을 뒤집는다. queue put이 request-local relay로 격리되고 S delivery가 A consumer 속도에 불변이며 buffers가 bound 안에 있고 A cancel이 cleanup된다. add/abort sequence가 ordered 또는 tombstone으로 ghost request를 만들지 않는다. core failure가 readiness false와 pending terminal로 전파된다.

이렇게 쓰면 다른 팀이 issue를 올바른 owner에게 보낸다. API route, preprocessing, AsyncLLM, core client, output handler, scheduler 가운데 first divergence가 있는 repository module에서 시작한다. GPU라는 넓은 범주로 보내지 않는다.

### 21.9.3 독자가 남겨야 하는 하나의 그림

복잡한 class diagram 대신 request와 engine의 두 수명을 평행하게 그린다. request 수명은 HTTP receive→validated→preprocessed→stream registered→outputs committed→transport terminal이다. engine 수명은 core add→scheduler accepted→running→finished/aborted→resources freed다.

두 선은 add와 output/abort에서 교차한다. request line이 transport close로 먼저 끝나면 engine line에 abort를 보내야 한다. engine line이 error로 먼저 끝나면 request line에 terminal error를 보내야 한다. shutdown은 두 선 모두를 닫는다.

ID와 epoch는 두 선의 사건이 같은 request incarnation임을 증명한다. queue와 output stream은 사건을 운반한다. backpressure는 두 선 사이 전달이 늦어진 상태다. cleanup은 두 선의 마지막 owner가 references를 놓는 과정이다.

이 그림이 있으면 `generate`가 단순 함수 호출이 아니라 lifecycle bridge라는 점이 보인다. API generator close, handler exception, core restart가 왜 서로에게 terminal을 전달해야 하는지도 보인다.

다음 장 scheduler는 engine line의 accepted 이후를 확대한다. waiting/running, token budget, KV allocation과 preemption을 설명한다. 이 장은 그 scheduler에 정확한 request를 넘기고 결과를 정확한 client stream으로 돌려보내는 경계를 책임진다.

### 21.9.4 한 요청을 다시 천천히 완주한다

마지막에는 함수 이름을 잠시 내려놓고 `Hello` 요청으로 돌아가자. HTTP server가 body를 받으면 아직 prompt text와 API options만 있다. validation은 이 요청이 처리 가능한지 판정하고 preprocessing은 model이 이해할 token IDs와 engine params를 만든다. 여기까지 실패하면 scheduler와 GPU는 owner가 아니다.

AsyncLLM이 request-local stream을 만들고 core client에 add를 보낼 때 두 수명이 연결된다. local stream은 client로 돌아갈 길이고 core command는 scheduler로 들어갈 길이다. request ID는 두 길을 묶는다. 어느 한쪽만 만들어지고 다른 쪽이 실패하면 rollback이나 abort가 필요하다.

core가 request를 받아 scheduler에 넘기면 engine이 work를 소유한다. API coroutine이 여전히 client connection을 소유하지만 token 계산 시점은 core가 정한다. outputs는 여러 requests가 섞인 batch로 돌아오며 handler가 ID를 보고 각 local stream으로 나눈다.

API generator는 그 stream을 읽어 client protocol로 commit한다. core가 token을 만들었다는 사실과 client가 text를 읽었다는 사실 사이에 output processor, request queue, generator와 transport가 있다. 이 구간이 느리면 GPU는 정상이어도 stream이 끊겨 보인다.

client가 떠나면 request line이 먼저 끝난다. generator가 abort를 보내고 core/scheduler가 engine line을 닫아야 한다. core가 먼저 죽으면 engine line이 먼저 끝나므로 handler가 pending client streams에 error terminal을 보내야 한다. shutdown은 new lines를 막고 두 종류의 existing lines를 모두 정리한다.

이 설명은 API와 engine을 두 개의 black box로 두지 않는다. 각 bridge가 state와 identity를 전달하며 failure를 반대편에 전파한다. queue가 단순 performance detail이 아닌 이유도 여기 있다. queue는 수명의 사건을 보존하고 순서를 만든다.

### 21.9.5 reader-first debugging: 첫 10분에 무엇을 묻는가

응답이 오지 않는다고 GPU trace부터 열지 않는다. request ID가 생성됐는지, preprocessing이 끝나 final IDs가 생겼는지, core add가 enqueue/receive됐는지, output handler가 살아 있는지 네 질문으로 시작한다. 이 네 답이 ingress·core·delivery를 빠르게 가른다.

ID가 없으면 route/validation, ID는 있고 prompt IDs가 없으면 preprocessing, add send가 없으면 AsyncLLM transaction, core receive는 있는데 scheduler state가 없으면 core input, core output은 있는데 stream put이 없으면 handler, yield는 있는데 receive가 없으면 transport다.

timeout 후에도 GPU가 돈다면 abort chain을 본다. disconnect detection, abort enqueue, core receive, scheduler removal과 block free 중 첫 누락을 찾는다. abort method가 호출됐다는 log 하나로 완료를 선언하지 않는다.

재시작 뒤 이상하다면 epoch를 묻는다. pending old streams가 terminal됐는지, new core readiness가 확인됐는지, stale output channel이 닫혔는지, IDs가 new namespace인지 본다. model reload 성공만으로 lifecycle 복구를 증명하지 않는다.

한 slow client 때문에 모두 느리다면 shared handler와 per-request queue를 본다. core batch receive, A/B stream put, API yield timestamps로 head-of-line을 찾는다. queue 크기를 무작정 늘리기 전에 memory bound와 cancel policy를 확인한다.

이 첫 10분 질문들은 source chain과 일치한다. route→preprocess→AsyncLLM→core client→handler→transport 순서다. 독자는 symptom이 어느 함수 owner로 가는지 알고 deeper scheduler/kernel 조사 여부를 결정할 수 있다.

### 21.9.6 source-only 결론의 정확한 범위

고정 source로 우리는 request state가 어느 function과 map/queue를 지나도록 구현됐는지, 어떤 error와 cleanup branch가 존재하는지 증명할 수 있다. source에서 absent ordering이나 missing cleanup 후보도 찾을 수 있다. 그러나 특정 배포의 queue wait와 실제 selected client type, crash frequency는 실행 evidence가 필요하다.

따라서 이 장의 시간표와 queue 수치는 fixture 설명이다. 실측값처럼 인용하지 않는다. deployment report에는 vLLM revision, client/core topology, server flags, process epoch와 workload를 붙인다. source claim과 observation을 다른 열에 둔다.

source에 graceful shutdown method가 존재한다고 zero dropped stream을 주장하지 않는다. disconnect poll이 있다고 cancel latency가 짧다고 주장하지 않는다. bounded queue class가 있다고 memory가 충분하다고 주장하지 않는다. 각각 runtime workload와 timing이 필요하다.

반대로 runtime symptom만으로 source owner를 추측하지 않는다. GPU idle과 긴 TTFT가 tokenizer queue일 수도 transport buffering일 수도 있다. request timestamps와 state transitions로 first divergence를 만든다. source와 관찰이 만나는 지점에서만 결론을 쓴다.

21장의 성과는 모든 selector와 exception 이름을 외우는 것이 아니다. 대표 request의 정확한 수명을 설명하고, 새로운 branch를 만났을 때 어느 의미 상태를 보존해야 하는지 아는 것이다. 이 기준이 있으면 version이 바뀌어도 source를 다시 읽을 수 있다.

실제 code review에서도 이 기준을 쓴다. preprocessing을 병렬화하는 change라면 prompt와 request ID가 엇갈리지 않는지, cancellation이 worker task까지 닫는지 본다. output queue를 바꾸는 change라면 ordering과 terminal, boundedness와 slow-client 격리를 본다. core client reconnect change라면 old pending failure와 epoch isolation을 본다.

성능 개선 claim도 lifecycle 위에서 읽는다. add latency를 줄였지만 failure rollback이 빠졌다면 안전한 개선이 아니다. output handler throughput을 높였지만 unbounded memory로 옮겼다면 tail risk를 이동했을 뿐이다. abort를 빠르게 했지만 add-before-abort ordering을 깨면 ghost request가 생긴다.

request trace를 보존하는 이유도 단순 observability가 아니다. identity와 timestamps는 cleanup safety, latency attribution과 retry accounting을 동시에 증명한다. 다만 raw prompt와 output을 남기지 않고도 충분한 state evidence를 만들 수 있다. IDs, lengths, enums, hashes와 queue transitions를 우선한다.

운영 handoff에는 current state를 포함한다. request가 API preprocessing, local registered, core sent, scheduler active, transport partial 중 어디에 있는지 적는다. finished라고만 쓰면 engine finish와 client terminal을 혼동한다. cancelled라고만 쓰면 abort requested와 resources freed를 혼동한다.

문제가 해결된 뒤에도 competing hypotheses를 기록한다. 예를 들어 handler blocking을 고쳤다면 tokenizer queue와 scheduler stall이 아니었던 evidence를 남긴다. 다음 비슷한 incident에서 같은 조사를 반복하지 않는다. 수정이 first divergence를 이동시키지 않고 제거했는지 인접 checkpoints를 확인한다.

이렇게 요청 수명을 끝까지 붙잡으면 vLLM의 많은 내부 class가 하나의 이야기로 읽힌다. route와 preprocessor는 engine input을 준비하고, AsyncLLM과 core client는 두 process의 ownership을 잇고, handler와 stream은 결과를 client에게 commit하며, abort와 shutdown은 남은 state를 안전하게 닫는다. 독자가 다음 scheduler 장에 도착할 때 request는 더 이상 추상적인 HTTP 한 건이 아니라 명확한 identity와 budget, lifetime을 가진 engine object다.

최종 source review에서는 대표 request의 각 화살표에 실제 caller와 callee를 하나씩 붙인다. 함수가 이동했거나 wrapper가 추가됐다면 고정 commit의 concrete symbol을 따른다. 넓은 file link만으로 세부 ordering을 주장하지 않고 exact mutation과 cleanup line을 source note에 좁힌다. 특정 runtime client와 topology는 initialization config 없이는 확정하지 않는다.

독자는 이 장의 fixture를 새 release에도 반복할 수 있어야 한다. text prompt 하나, terminal 없는 slow consumer, add 직후 abort, handler exception과 core restart를 의미 좌표로 적용한다. function name이 바뀌어도 stream map, core command, output demultiplex와 cleanup owner를 찾으면 behavior change를 판정할 수 있다. 이것이 예외 이름을 외우는 것보다 오래가는 디깅 방법이다.

그리고 모든 판정에는 request identity, core generation, expected terminal과 미검증 runtime 항목을 함께 남긴다. 그래야 다음 조사자가 같은 경계를 다시 증명할 수 있다.

22장은 같은 외부 요청을 SGLang의 `TokenizerManager`와 scheduler IPC 경계에 놓는다. 이름을 억지로
일치시키지 않고, 이 장에서 사용한 ingress identity, queue admission, stream terminal, abort와 cleanup
좌표를 유지해 두 구현이 책임을 나누는 위치를 비교한다.

## 21.10 current vLLM request를 함수 경계로 한 번에 걷는다

대표 요청 V21은 OpenAI route에서 schema validation과 preprocessing을 거쳐 engine prompt/params를 만들고, `AsyncLLM.generate`가 request stream을 등록한 뒤 engine-core client에 add command를 보낸다. core process는 scheduler queue에 넣고 step 결과를 output handler가 request stream으로 돌려준다. generator 종료/finally는 abort와 cleanup을 소유한다.

함수 이름은 pinned revision에서 확인하되 의미 checkpoint를 유지한다. `route accepted`, `preprocess complete`, `local stream registered`, `core add sent`, `scheduler admitted`, `first output`, `engine terminal`, `external terminal`, `abort requested`, `resources freed`다.

### 21.10.1 add ordering이 ghost request를 막는다

local request stream/map을 등록하기 전에 core add를 보내면 fast output이 돌아왔을 때 consumer를 찾지 못할 수 있다. 반대로 local 등록 후 core send가 실패하면 local residue를 rollback해야 한다. source에서 exact ordering과 exception cleanup을 읽는다.

V21 generation key는 request ID만이 아니다. core restart 뒤 동일 ID가 old output과 충돌하지 않게 client/core generation을 둔다. duplicate add를 reject/replace하는 policy와 scheduler state를 확인한다.

### 21.10.2 Async generator lifetime

client가 iteration을 끝까지 소비하지 않거나 disconnect/exception으로 generator를 닫으면 `finally`가 abort를 호출해야 한다. generator object가 GC될 때까지 cleanup을 미루지 않는다. route/task cancellation이 generator close로 전달되는지 본다.

engine finish가 stream queue에 terminal output을 넣어도 HTTP writer가 이를 consume하지 못할 수 있다. engine terminal과 external terminal을 분리한다. output queue와 result objects의 last consumer가 끝나야 local cleanup이 닫힌다.

### 21.10.3 client transport와 core command

in-process/multiprocess client가 add/abort command를 어떤 queue/socket로 보내고 buffer/reference lifetime을 어떻게 보존하는지 읽는다. send 성공은 scheduler admission이 아니다. command accepted/decoded/core handling을 별 event로 둔다.

timeout/cancel 시 add와 abort가 다른 channels/queues에서 reorder될 수 있다. scheduler가 abort-before-add tombstone을 보존하거나 client가 ordering/ack를 보장해야 ghost request를 막는다. exact design은 source predicate로 확인한다.

### 21.10.4 scheduler ingress

core add handler가 EngineCoreRequest를 scheduler add_request에 넘기는 지점을 찾는다. waiting queue entry, token/KV budget과 structured/adapter dependencies는 scheduler 장의 owner다. 여기서는 ingress event와 rejection/exception이 output stream terminal로 돌아오는 경로를 닫는다.

queue에 들어갔다는 사실과 running/scheduled를 구분한다. cancel이 waiting과 running 어느 collection을 제거하는지, freed resources와 terminal output을 누가 만드는지 확인한다.

## 21.11 queue·output·cancel conservation을 숫자로 검산한다

시각 t0 accepted, t1 stream register, t2 add send, t3 core receive, t4 waiting, t5 running, t6 first output, t7 disconnect, t8 abort receive, t9 scheduler finish/free, t10 local stream cleanup이라고 하자. 각 clock owner와 request generation을 둔다.

### 21.11.1 request conservation

`registered = pre_core + core_pending + scheduler_waiting + scheduler_running + output_pending + terminal + residue`처럼 mutually exclusive snapshot을 만들 수 있다. 실제 구현 state가 겹치면 conservation event ledger로 쓴다. accepted logical request는 exactly one engine terminal과 cleanup terminal을 가져야 한다.

batch output 하나가 여러 request 결과를 담으면 output handler가 request ID로 demultiplex한다. unknown/stale ID는 drop만 하지 말고 core generation mismatch와 residue를 기록한다. 한 request handler exception이 다른 requests output loop를 멈추지 않게 isolation을 본다.

### 21.11.2 queue latency 계산

ingress preprocessing `t1-t0`, local/core submit `t3-t1`, scheduler wait `t5-t4`, model first output `t6-t5`, delivery `client-t6`를 나눈다. TTFT 하나로 tokenizer, IPC, queue와 compute를 합치지 않는다.

request 1,000개 중 scheduler waiting 100, core-pending 200이면 “queue=300” 하나보다 owner별 oldest age/work를 본다. core IPC stall과 scheduler capacity 부족의 수정 owner가 다르다.

### 21.11.3 abort conservation

disconnect requests 100, abort commands sent98, core received95, scheduler terminal93, resources freed92라면 first gap은 route/generator2, IPC3, scheduler2, free1이다. 동일 time window/settlement delay를 고려하고 oldest pending generation을 본다.

abort latency가 길어도 이미 engine finish된 race일 수 있다. finish owner와 abort owner가 exactly-one terminal을 만들고 cleanup에 합류한다. not-found abort는 정상 race인지 stale generation인지 reason을 둔다.

### 21.11.4 output backpressure

engine output rate가 client consumption보다 크면 per-request stream queue가 커진다. bounded queue/backpressure/cancel policy와 core generation pause 여부를 확인한다. queue bytes/oldest age와 request count를 관측한다.

slow client가 output objects를 붙잡아 KV cleanup까지 늦추는지 ownership을 본다. engine terminal 후 KV는 해제돼도 serialized output memory는 남을 수 있다. resource 종류별 terminal을 분리한다.

## 21.12 cancel race와 core restart incident

사건 V21-C는 client가 add 직후 취소해 local abort command가 add보다 먼저 다른 queue로 core에 도착했다. scheduler는 request를 찾지 못해 abort를 no-op했고, 뒤늦은 add가 waiting queue에 들어가 client 없는 generation을 계속 실행했다.

### 21.12.1 경쟁 가설

route가 abort를 안 보냈다는 가설, transport reorder, scheduler waiting removal bug, stale output cleanup을 나눈다. trace에서 abort sent/received가 add receive보다 앞이고 scheduler not-found 뒤 add admission이 보이면 reorder/tombstone 가설이 강하다.

### 21.12.2 tombstone 또는 ordering protocol

core가 `(request_id,generation)` abort tombstone을 짧은 lifetime 유지하면 뒤 add를 reject/terminal 처리할 수 있다. 또는 같은 ordered channel/sequence number와 ack로 add-before-abort를 보장할 수 있다. tombstone TTL은 max reorder/retry window보다 길고 ID reuse를 generation으로 막아야 한다.

abort를 무한 tombstone으로 보존하면 memory가 늘므로 cleanup/expiry와 metric을 둔다. add retry가 동일 sequence인지 새 attempt인지 구분한다. duplicate add와 cancel race fixture를 넣는다.

### 21.12.3 core restart generation

core process restart 시 client의 local streams와 old core requests가 어떤 terminal을 받는지 본다. new core가 old request ID output을 만들 수 없으므로 failure broadcast와 abort/cleanup을 수행한다. client가 자동 resubmit하면 external commit/idempotency를 고려한다.

old ZMQ/buffer/output handler task가 살아 새 generation output과 섞이지 않게 generation fence를 둔다. restart readiness는 socket connect만 아니라 output handler, core health, scheduler accepting을 포함한다.

### 21.12.4 incident 종료

add/abort reorder, duplicate add, cancel-after-finish, core restart during add/running/output의 matrix를 통과한다. accepted requests, engine terminals, cleanup과 KV/request residue 보존식을 본다. client response recovery만으로 닫지 않는다.

## 21.13 pinned current source를 caller→callee→next consumer로 고정한다

OpenAI route에서 serving object가 async generator를 만들고 response가 이를 소비하는 span, serving preprocessing이 engine prompt/params를 만드는 span을 찾는다. `AsyncLLM.generate`에서 add/stream register/finally abort를 연결한다. EngineCore client의 add/abort command transport와 output handler를 잇는다.

core side에서는 command handler, add request construction, scheduler add/abort/finish와 output serialization을 잇는다. 넓은 file 링크 하나로 ordering과 cleanup을 모두 증명하지 않는다. 각 mutation과 next consumer에 pinned span을 둔다.

### 21.13.1 함수 trace 표

열은 `stage`, `caller`, `callee`, `input identity`, `state mutation`, `output/ack`, `failure rollback`, `next consumer`다. wrapper/helper가 추가돼도 의미 edge를 유지한다. current revision에서 exact symbol을 확인한다.

### 21.13.2 source가 증명하지 않는 것

queue/socket send code는 실제 latency나 selected topology를 증명하지 않는다. abort handler existence는 provider/client receipt나 KV free completion을 자동 증명하지 않는다. runtime trace/metrics가 필요한 칸을 표시한다.

### 21.13.3 option mutation

priority, structured output/grammar, multimodal/LoRA, output mode, scheduler config가 preprocessing/request state와 dependency/admission을 바꾼다. option parse에서 scheduler consumer까지 잇되 상세 scheduling은 뒤 장으로 넘긴다.

**수명주기 통제 실험.** 실험 A는 enqueue 직후 cancel해 waiting entry와 block allocation이 남지 않는지 본다. 실험 B는 schedule과 output 사이 cancel race를 주입해 late output이 stream에 commit되지 않는지 확인한다. 실험 C는 core restart 뒤 old generation output을 보내 새 request와 섞이지 않는지 본다. 실험 D는 slow consumer로 output queue를 채워 backpressure가 engine completion과 연결 cleanup을 어떻게 바꾸는지 추적한다. 실험 E는 같은 request ID 재사용을 거부하거나 generation으로 구분하는지 검증한다.

## 21.14 regression·monitoring·handoff terminal

matrix는 preprocess error, add send failure, abort-before/after-add, waiting/running cancel, finish/disconnect race, slow consumer, output handler exception, core restart/shutdown을 포함한다. 각 cell은 local stream/core/scheduler/output/resource terminal을 판정한다.

metric에는 preprocessing, core submit/ack, scheduler wait, first output, delivery, abort propagation, cleanup latency와 state counts/oldest age를 둔다. request ID는 trace에, state/reason/generation은 bounded labels로 둔다.

readiness는 ingress accepted 여부, core health/generation, command/output channel, scheduler admission과 model readiness를 나눈다. core restart 중 route가 200/stream을 먼저 commit하지 않게 admission fence를 둔다.

rollback은 API process/client/core/scheduler protocol generation을 맞추고 in-flight를 drain한다. local stream map과 old output queues, tombstones를 잊지 않는다. binary rollback이 external queues/state를 자동 삭제하지 않는다.

최종 terminal은 ingress, core command, scheduler, output, cancellation, resource와 observability다. incident 문장은 “V21 add seq31보다 abort seq32가 별 channel에서 먼저 도착해 not-found 처리됐고 뒤 add가 ghost waiting이 됐다. generation tombstone/ordered protocol과 add-abort fixture로 scheduler terminal·KV free까지 닫았다”처럼 쓴다.

독자는 이제 vLLM을 route→AsyncLLM→client→core→scheduler→output stream의 함수와 state로 읽는다. 22장에는 동일한 identity/terminal 좌표를 가져가 SGLang의 TokenizerManager와 Scheduler IPC ownership을 비교한다.

**한 요청의 identity envelope를 구체화한다.**

envelope에는 external logical ID, gateway attempt ID, vLLM request ID, client/core generation, arrival/admission deadline, model/adapter/grammar/multimodal identity, output/stream mode가 있다. 모든 field를 scheduler key에 넣는 것은 아니지만 trace와 cleanup correlation에 필요하다.

request ID만 재사용되면 old output/abort와 충돌할 수 있다. generation이 process restart, client reconnect, retry attempt를 구분한다. transport message와 scheduler request, output event가 같은 envelope subset을 갖는지 확인한다.

prompt 원문은 보안상 envelope에 넣지 않고 token length/digest/artifact generation을 둔다. multimodal/tool data는 bounded identity와 lifecycle owner를 둔다. detailed content는 승인된 incident store에 제한한다.

**preprocessing CPU와 core queue를 구분한다.**

route accepted 뒤 tokenizer/template/multimodal preprocessing이 오래 걸리면 core request는 아직 없다. scheduler waiting count는 정상인데 TTFT가 늘 수 있다. preprocessing inflight/oldest age와 worker pool saturation을 관측한다.

preprocess 완료 후 local stream registration과 core send 사이 blocking도 별 구간이다. serialization/IPC buffer와 client backpressure를 본다. core queue와 scheduler waiting을 하나의 queue로 합치지 않는다.

preprocess 중 disconnect면 engine abort가 아니라 local task cancellation/temporary artifact cleanup이 owner다. core add 전/후 cancel 분기와 resource를 따로 테스트한다.

**output handler fairness와 failure isolation.**

batched engine output이 request 여러 개를 포함할 때 한 request의 formatter/parser가 CPU를 오래 쓰면 다른 request demultiplex가 지연될 수 있다. handler가 sequential인지 task 분리인지 source를 보고, per-batch handling duration과 slow request를 관측한다.

unknown request ID output은 core/client state divergence 신호다. 이미 cancelled/cleaned generation이면 expected late output로 count할 수 있지만 new request와 collision하면 심각하다. reason/generation을 기록한다. 무조건 ignore해 evidence를 잃지 않는다.

handler exception이 main loop를 종료하면 모든 request streams가 orphan될 수 있다. outer loop error broadcast/failure cleanup과 restart를 확인한다. 개별 malformed output를 격리할 수 있는지도 본다.

**bounded stream queue 계산.**

동시 requests 2,000, per-request queue cap32, output object 평균4KiB면 payload 상한 약256MiB다. Python/object overhead와 logprobs 큰 output은 별도다. unbounded queue면 slow clients가 memory를 계속 늘린다.

cap 도달 정책은 engine pause, drop, cancel/error 중 하나다. token delta drop은 protocol corruption이므로 조용히 버리지 않는다. engine pause가 scheduler backpressure와 연결되지 않으면 output handler만 막히고 core가 계속 생성할 수 있다.

queue entry count와 bytes, oldest age, client write wait, engine generated-delivered gap을 본다. output queue 개선이 GPU throughput/ITL과 external delivery를 어떻게 바꾸는지 함께 평가한다.

**scheduler add acknowledgment의 의미.**

core command receive가 scheduler add 성공을 뜻하는지, rejection/exception이 output channel로 언제 돌아오는지 본다. client가 submit 성공을 admission으로 기록하면 queue latency와 failure attribution이 틀린다. ack 단계가 없으면 trace events로 간접 구분한다.

model/adapter/grammar dependency 때문에 request가 blocked state에 있을 수 있다. waiting queue count만으로 schedulable work를 알 수 없다. ingress는 dependency identity를 정확히 전달하고 scheduler 장은 readiness/wake를 소유한다.

**cancel이 KV free까지 내려가는 경로.**

running request abort는 scheduler running state에서 제거하고 sequence/KV blocks, encoder/multimodal/structured state와 output bookkeeping을 정리해야 한다. waiting request는 아직 KV가 없을 수 있지만 dependency waiter와 local stream이 있다. 상태별 cleanup 항목을 표로 둔다.

speculative/parallel sampling에서 하나의 external request가 child sequences를 가질 수 있다. abort가 root/children 모두 닫는지 확인한다. child terminal count를 external terminal과 혼합하지 않는다.

prefix cache blocks는 shared라 reference/ownership 정책에 따라 request cancel이 physical page를 바로 free하지 않을 수 있다. resource terminal은 request-owned references가 release됐다는 뜻이지 cache 전체 byte가 감소한다는 뜻이 아닐 수 있다.

**shutdown incident.**

process drain 중 new ingress가 여전히 accepted돼 local stream을 등록했지만 core command channel은 닫힌 사건을 생각하자. add send가 실패하고 rollback이 stream map을 제거하지 않아 readiness false 뒤에도 residue가 쌓였다.

shutdown ordering은 ingress admission fence→new add stop→in-flight drain/terminal→output handler stop→channels/core close→local maps cleanup이다. exact implementation은 source에서 확인한다. listener close만으로 충분하지 않다.

send failure fixture는 stream registered 직후 channel close를 넣고 local rollback/terminal을 판정한다. shutdown deadline에서 remaining running requests abort와 KV free를 확인한다. core crash와 planned shutdown reason을 구분한다.

**source diff에서 위험한 이동.**

`AsyncLLM.generate`에서 add/abort ordering이 바뀌거나 output handler가 별 task/process로 이동하면 lifecycle contract를 재감사한다. request map 등록 시점, generator finally, exception broadcast와 queue ownership이 핵심이다.

client transport가 ZMQ/MP queue/in-process로 바뀌면 buffer copy/lifetime, ordering/ack와 failure model이 바뀐다. class 이름이 유지돼도 effective topology는 config에 따라 다를 수 있다. initialized client type을 runtime artifact로 요구한다.

scheduler API가 add/abort/finish 반환 semantics를 바꾸면 core/output cleanup을 갱신한다. changelog가 아니라 pinned caller-callee와 state transition diff를 본다.

**function trace를 실제 리뷰 질문으로 바꾼다.**

route는 언제 response header를 commit하는가. serving handler는 generator를 누가 close하는가. preprocessing failure는 stream map 등록 전/후 어느 terminal을 만드는가. AsyncLLM은 local stream을 add send 전 등록하는가. send failure를 어떻게 rollback하는가.

client는 command ordering과 buffer lifetime을 어떻게 보장하는가. core는 duplicate/stale generation을 어떻게 처리하는가. scheduler add rejection은 어디로 돌아오는가. output handler는 unknown/terminal request를 어떻게 reconcile하는가. abort는 waiting/running/resource를 어떻게 닫는가.

답은 option 문서가 아니라 pinned symbol과 next consumer에서 찾는다. source가 증명하지 않는 actual latency/branch는 runtime trace TODO다.

**latency와 goodput 계산.**

logical 1,000 requests 중 preprocessing p99 100ms, core submit p99 20ms, scheduler wait p99 500ms, model TTFT 200ms, delivery 30ms라면 end-to-end p99를 단순 합할 수는 없지만 각 owner 후보를 보여 준다. correlated cohort/trace에서 first divergence를 찾는다.

handler가 10ms마다 batch outputs를 처리하고 slow formatter 한 건이 100ms를 막으면 다른 requests delivery gap이 늘 수 있다. GPU ITL이 정상이어도 client ITL이 나빠진다. engine vs output-handler clocks를 나눈다.

cancel 100건 중 20건이 engine terminal까지1초 넘게 걸리고 각자 200 token/s로 계속 생성하면 wasted token 상한이 약4,000 tokens다. actual step/batch를 측정하되 abort propagation 개선의 비용 의미를 설명한다.

**observability schema.**

span events는 ingress_accepted, preprocess_done, stream_registered, core_add_sent/received, scheduler_wait/running, first_output, engine_terminal, external_terminal, abort_sent/received, resources_released, stream_removed다. generation과 bounded reason을 둔다.

metrics는 stage duration histograms, state gauges/oldest age, abort gaps, output queue bytes, unknown output, core restart generation과 residue counters다. request ID는 label로 쓰지 않는다. trace sampling이 terminal events를 잃지 않게 한다.

**regression matrix sensitivity.**

abort-before-add mutation이 ghost request assertion을 실패시키는지, output handler stop mutation이 orphan stream을 검출하는지, send failure rollback 제거가 residue counter를 올리는지 확인한다. positive path만으로 probe를 승인하지 않는다.

same request ID/new generation fixture, duplicate add, late output, core restart와 slow consumer를 넣는다. model/CUDA 실행 없이 fake core/scheduler transport로 lifecycle state machine을 검증할 수 있는 부분과 실제 integration이 필요한 부분을 구분한다.

**최종 source-backed incident record.**

레코드에는 pinned revision, initialized client topology, request/generation, stage timeline, core commands sequence, scheduler state, output/abort/resource terminals, first divergence와 source spans를 둔다. raw prompt는 넣지 않는다.

수정 후 original failing fixture와 passing neighbor, adjacent race를 통과한다. service success, resource cleanup, telemetry 회복을 별 terminal로 닫는다. old generation queues/maps가 settlement window 뒤 0인지 본다.

21장의 final artifact는 API request가 scheduler object가 되는 화살표와 그 반대 output/cleanup 화살표다. 독자는 어느 함수가 단순 전달이고 어느 함수가 ownership을 새로 만드는지, 실패가 어느 side에서 rollback돼야 하는지 알 수 있다. 이 상태에서 22장의 SGLang IPC와 overlap ownership을 공정하게 비교할 수 있다.

**OpenAI entrypoint에서 current callable을 찾는 순서.**

router decorator와 endpoint function에서 request model, raw HTTP request와 serving object 호출을 찾는다. chat/completion path가 공유 base를 쓰는지, streaming/non-stream response branch가 generator를 어떻게 소비하는지 확인한다. API path 문자열만으로 handler를 추정하지 않는다.

serving object에서 request preprocessing, request ID/deadline, sampling/output params와 engine client 호출을 잇는다. helper가 여러 층이면 실제 `generate`/`encode` call까지 내려간다. unsupported/error branch와 disconnect callback도 표시한다.

AsyncLLM class가 wrapper/factory 뒤에서 constructed된다면 실제 implementation과 engine-core client type을 config/init source에서 찾는다. 같은 public method라도 in-process/multiprocess가 다를 수 있다. current deployment가 어느 path인지 runtime 미검증으로 구분한다.

**request object 변환 ledger.**

API request는 messages/text, tools/options를 갖는다. preprocessing 후 engine prompt/token IDs와 multimodal/LoRA/grammar input이 된다. AsyncLLM add 단계에서는 EngineCoreRequest와 local RequestStream/state가 생긴다. core/scheduler에서는 request state, token/KV budget과 status가 생긴다.

각 변환에 identity, immutable fields, mutated/default fields, owner, rejection/rollback을 둔다. 같은 request ID 아래 객체가 여러 개 있으므로 class 이름과 generation을 적는다. raw API object가 scheduler까지 그대로 전달된다고 설명하지 않는다.

**client/core protocol message taxonomy.**

add, abort, utility/profile/reset/shutdown 같은 command가 같은 channel을 쓸 수 있다. request lifecycle과 control plane command를 구분한다. add/abort message에는 request/generation/sequence가 있는지, batch command가 여러 IDs를 담는지 본다.

output message는 per-step batched outputs, engine stats, failure/health와 terminal을 포함할 수 있다. output handler가 message kind별로 dispatch하고 request streams를 reconcile하는 source를 읽는다. failure broadcast가 모든 active streams를 terminal로 만드는지 확인한다.

serialization은 copy/zero-copy buffer와 lifetime을 바꾼다. send가 반환된 뒤 producer가 buffer를 재사용해도 되는지 transport contract를 본다. reference retention claim이 있는 경우 exact source와 next consumer를 연결한다.

**scheduler queue에서 cancel이 찾는 집합.**

waiting, running, finished, preempted/blocked 등 state가 있을 수 있다. abort handler가 모든 active collection과 auxiliary dependency waiters를 검색하는지 본다. 이미 terminal이면 idempotent no-op과 reason을 기록한다.

request가 scheduler add 전 core pending에 있으면 scheduler abort만으로 부족하다. core command layer tombstone/ordering이 필요하다. local stream 전 단계면 route task cancellation이 owner다. cancel API 하나가 모든 단계에 같은 방식으로 작동한다고 쓰지 않는다.

**KV/resource free acknowledgment.**

scheduler finish/free 함수 호출은 block allocator mutation과 output terminal ordering을 가진다. output이 client로 전달되기 전에 KV를 free해도 output object가 KV를 참조하지 않는다면 가능하다. 정확한 last consumer를 source contract로 확인한다.

freed block count가 바로 allocator reserved memory 감소를 뜻하지 않는다. logical free와 pool reuse 가능, physical CUDA memory release를 구분한다. request residue는 logical ownership으로 판정한다.

adapter/grammar/multimodal temporary state와 encoder cache도 cleanup ledger에 넣는다. KV free counter 하나로 전체 resource terminal을 증명하지 않는다.

**core busy와 scheduler busy를 분리하는 incident.**

TTFT가 늘고 scheduler waiting이 낮은 사건에서 core input queue oldest age가 상승했다고 하자. route/preprocess는 정상, add sent와 core received 사이가 길다. scheduler option을 바꾸기 전에 client/core transport/handler를 본다.

반대로 core receive는 빠르고 scheduler waiting work/oldest가 늘면 admission/scheduling owner다. queue count가 같아도 prompt token work와 schedulability가 다를 수 있다. 26장 이후 scheduler로 handoff한다.

output queue가 막히면 core processing loop가 input/add도 늦추는 구조인지 source/thread/task topology를 본다. 하나의 event loop에서 serialization이 command receive를 starve할 수 있다. topology를 추정하지 않고 initialized tasks를 확인한다.

**priority와 deadline handoff.**

gateway attempt의 remaining deadline/priority가 vLLM request에 전달되는지, server가 자체 timeout/abort를 지원하는지 확인한다. 전달되지 않으면 gateway client disconnect가 primary cancellation signal일 수 있다. scheduler가 API deadline을 직접 안다고 가정하지 않는다.

priority option은 request field validation에서 scheduler ordering consumer까지 걷는다. unsupported scheduler에서 ignore/reject되는지 본다. priority가 output stream fairness까지 보장하지 않는다.

**multiprocess failure matrix.**

API process crash, engine client task crash, core process crash, scheduler exception, output handler crash를 분리한다. 어느 side가 active requests에 failure terminal을 broadcast하고 local/core resources를 정리하는지 표로 둔다.

API process crash에서 core가 disconnect를 감지하고 requests를 abort하는지, 아니면 generation lease/heartbeat가 필요한지 본다. core crash에서 API streams는 error terminal과 cleanup을 받아야 한다. restart가 readiness 이전에 old messages를 drain/격리한다.

network/IPC partition에서는 양쪽이 상대 terminal을 모를 수 있다. timeout/heartbeat와 generation fencing을 확인한다. split-brain new core/old core output을 request ID만으로 섞지 않는다.

**backpressure 정책 수치 fixture.**

per-request cap16, engine이 초당 100 outputs, client가 20 consume하면 queue는 초당 80 증가해 0.2초에 cap을 채운다. cap 도달 뒤 engine pause가 100ms 지연되면 추가 10 outputs가 생긴다. overflow handling과 generated-delivered gap을 판정한다.

batch handler가 request100개 output을 10ms에 처리하지만 한 parser가100ms를 쓰면 batch delivery가90ms 이상 지연될 수 있다. per-request isolation/tasking이 latency와 ordering을 어떻게 보존하는지 본다. task 폭발도 guardrail이다.

**cleanup idempotency fixture.**

finish, disconnect, output error와 shutdown이 같은 request cleanup을 호출한다. stream map delete, scheduler abort/free, trace/metrics close가 여러 번 호출돼도 negative counts/double free를 만들지 않아야 한다. terminal compare-and-set과 resource generation을 둔다.

known finished request abort, unknown stale generation abort, duplicate terminal output을 넣는다. expected reason과 residue0을 판정한다. exception을 swallow해 test가 crash하지 않는 것만으로 idempotency를 선언하지 않는다.

**현재 source pin을 책에 친절하게 배치한다.**

본문은 V21 요청을 먼저 걷고 각 stage에서 exact function link를 붙인다. 장말 source note에는 revision/file/symbol과 claim을 모은다. repository atlas를 opening에 두지 않는다. 독자는 왜 그 함수를 열어야 하는지 알고 link를 클릭해야 한다.

코드 일부 인용은 add/register/finally abort와 scheduler add/abort 같은 핵심 predicate만 짧게 사용한다. 긴 boilerplate를 넣지 않는다. 한국어 설명은 입력 state, mutation, next consumer와 실패 rollback을 풀어 쓴다.

**최종 acceptance 질문.**

route가 언제 engine request를 만들고 외부 response를 commit하는가. AsyncLLM은 local stream과 core request 중 무엇을 먼저 만든다. client/core message는 ordering과 generation을 어떻게 보존하는가. scheduler add/abort/finish는 어느 collection/resource를 mutate하는가.

output handler는 batched result를 request stream으로 어떻게 돌리고 unknown/stale output를 처리하는가. disconnect/exception/shutdown은 generator finally와 core abort를 거쳐 KV/resource cleanup까지 도달하는가. 각 질문에 source와 fixture가 있어야 한다.

답하지 못한 runtime topology/latency는 unknown으로 남기고 필요한 trace를 적는다. 정적 source가 증명한 branch를 실제 selected path로 꾸미지 않는다. 이 품질 기준이 함수 이름 나열을 운영 가능한 설명으로 바꾼다.

21장의 최종 문장은 명확하다. vLLM request는 API 객체에서 local stream과 core command, scheduler state, output frames로 여러 번 소유권이 이동한다. 각 이동의 generation·rollback·terminal을 닫아야 client cancel이 실제 GPU/KV resource cleanup과 같은 사건이 된다.

**배포 option을 lifecycle mutation으로 쓴다.**

engine multiprocess option은 client/core process 경계, command/output transport, failure/restart와 buffer lifetime을 바꾼다. 단순 성능 option이 아니다. initialized topology와 core generation을 trace한다. in-process와 multiprocess fixture를 같은 의미 checkpoint로 비교한다.

max concurrent/queue 관련 option은 ingress rejection과 scheduler admission, memory/capacity를 바꾼다. API server가 먼저 제한하는지 core/scheduler가 reject하는지 확인한다. rejection이 response commit 전 명확한 error로 돌아오고 local stream residue가 없어야 한다.

output/stream 관련 option은 aggregation, logprobs/payload, queue pressure와 external commit을 바꾼다. engine output mode와 OpenAI formatter가 같은 contract를 보는지 확인한다. option 변경이 handler CPU/bytes와 cancel lifetime에 미치는 효과를 측정한다.

**readiness와 liveness를 state로 분리한다.**

HTTP process가 listening이어도 model/core가 ready하지 않을 수 있다. readiness는 new request를 안전하게 preprocess/submit/admit할 수 있는 상태다. liveness는 process loop가 응답하는 상태다. core restart/drain에서 readiness를 먼저 내린다.

health ping이 core round trip을 하는지 local flag만 보는지 source를 읽는다. scheduler/model load/worker failure가 health에 어떻게 반영되는지 본다. shallow health를 admission proof로 과장하지 않는다.

readiness flip race에서 request가 accepted된 뒤 core unavailable이면 before-commit error/rollback을 수행한다. 이미 stream commit한 request는 structured terminal error와 cleanup을 낸다. 19장 contract와 잇는다.

**request ID 생성과 충돌.**

client supplied ID를 허용하는지 server generated인지, 중복 시 replace/reject인지 본다. prefix/trace용 ID와 scheduler key가 같을 수 있다. malicious collision이나 retry가 다른 user's request를 abort하지 않게 tenant/generation scope를 둔다.

ID를 metrics label로 쓰지 않는다. trace correlation에는 provider/gateway attempt와 mapping한다. logs에서 민감한 user IDs를 노출하지 않는다.

**structured/grammar preparation blocked state.**

grammar compilation이 async라면 request는 local/core/scheduler 어느 층에서 기다리는지 확인한다. cancel이 compiler waiter와 compiled artifact ref를 정리해야 한다. 준비 완료 callback이 cancelled generation을 scheduler에 재삽입하지 않게 generation guard를 둔다.

multimodal encoder input/LoRA loading도 비슷한 dependency state를 만든다. waiting count가 flat해도 oldest dependency age가 늘 수 있다. scheduler chapter로 넘길 ready/blocked reason을 보존한다.

**preprocessing cache identity와 lifecycle.**

tokenization/template/multimodal preprocessing cache가 있으면 request cancel 후 shared entry는 남을 수 있다. request-owned future/waiter와 shared artifact를 구분한다. cleanup이 cache 전체를 지우지 않아도 residue가 아닐 수 있다.

artifact generation이 바뀌면 old preprocess result를 new engine request로 보내지 않는다. request envelope에 tokenizer/template/processor generation을 둔다. input IDs가 같아도 multimodal feature/adapter identity가 다를 수 있다.

**metrics로 conservation을 계산한다.**

window에서 ingress accepted 10,000, preprocess errors100, core send failures20, scheduler terminal9,700, in-flight100, residue80이면 보존 gap을 조사한다. 서로 다른 cohort/time window를 섞지 않고 generation별로 맞춘다. retries logical/attempt를 분리한다.

abort requested500, core received490, scheduler terminal480, resources released478이면 gaps10/10/2다. propagation delay envelope와 oldest pending을 본다. counter equality만 기다리지 않고 stale generation을 찾는다.

**수정의 성능 부작용.**

ordered add/abort ack를 강화하면 round trip/latency가 늘 수 있다. correctness를 보존하면서 batching/sequence number로 비용을 줄일 수 있는지 본다. global lock/synchronize로 race를 없애고 throughput을 크게 잃은 수정은 다음 최적화 대상이다.

bounded output queue는 memory를 지키지만 slow client cancel이 늘 수 있다. backpressure upstream 전달과 SLO policy를 함께 평가한다. correctness terminal과 capacity terminal을 별도로 둔다.

**장말 handoff artifact.**

한 장 표에 V21의 request/generation, current state, owner, timestamp, queue/work, output cursor, abort/resource terminal을 둔다. source links는 각 transition 옆에 붙인다. unknown runtime field에는 probe를 적는다.

passing neighbor는 fast normal request, failing은 add/abort reorder나 slow output/core restart다. 수정 뒤 두 fixture와 boundary race를 재검증한다. first divergence 직전 정상 edge도 보존한다.

이 artifact를 22장에 복사하되 vLLM class 이름은 버린다. SGLang의 TokenizerManager/Scheduler가 같은 의미 ownership을 어디서 나누는지 새 source로 채운다. 구현 비교가 이름 대응표가 아니라 lifecycle contract 비교가 된다.

최종 release audit에서는 public `AsyncLLM` API signature뿐 아니라 내부 client/core protocol version을 본다. add/abort/output message schema가 달라지면 mixed binary rollout을 막거나 compatibility shim을 검증한다. API process와 core를 독립 배포할 수 있다면 handshake가 필요하다.

request stream implementation이 바뀌면 queue boundedness, terminal sentinel, exception propagation과 iterator close를 다시 확인한다. 사용자-visible chunk가 같아도 cleanup/lifetime이 바뀔 수 있다. slow/cancel fixture를 회귀에 유지한다.

scheduler add path가 queue 자료구조나 dependency state를 바꾸면 ingress event 이름도 갱신한다. waiting admission과 schedulable/running을 혼합하지 않는다. metric migration에서 old/new state를 같은 label로 합치기 전에 의미 호환성을 확인한다.

debugging 가이드는 넓은 repository 검색보다 request ID의 마지막 확인 transition부터 시작한다. route까지 보이면 preprocessing, core add sent까지만 보이면 transport/core, scheduler waiting이면 scheduling, engine terminal 뒤 client gap이면 output/transport, abort sent 뒤 residue면 cleanup owner다.

이 stop rule은 조사 시간을 줄인다. downstream evidence가 없는 상태에서 CUDA kernel을 의심하거나, scheduler terminal이 확인된 뒤 tokenizer를 다시 보지 않는다. first divergence 바로 앞의 passing transition과 next missing/incorrect consumer를 연다.

장애가 intermittent라면 batch neighbors와 process/core generation, queue ordering을 함께 보존한다. 단독 replay 성공은 race/overlap 조건이 사라졌다는 단서이지 원 사건의 반증이 아니다. failing cohort와 가까운 passing cohort를 비교한다.

보안상 request prompt를 수집하지 않고도 lifecycle은 조사할 수 있다. token counts/digests, state enums, timestamps, generation과 bounded reason을 사용한다. 필요한 synthetic fixture로 content-dependent formatter/grammar를 재현한다.

최종 acceptance는 source link 개수보다 reader action을 본다. 독자가 새 vLLM revision에서 entrypoint를 찾아 `generate`, client/core message와 scheduler add/abort를 연결하고, ghost request·orphan output·KV residue를 검출하는 fixture를 설계할 수 있어야 한다.

책의 배열도 이 행동을 돕는다. 정상 요청 하나를 먼저 route에서 scheduler와 output까지 걷고, 그 화살표 위에 add-before-abort·slow consumer·core restart 사건을 겹친다. source reference 표는 뒤에 두고 각 함수가 어느 state를 소유하는지 먼저 설명한다.

독자가 option을 만날 때만 해당 mutation을 소개한다. multiprocess는 client/core 경계에서, priority와 grammar는 request/scheduler handoff에서, streaming output은 handler/queue에서 설명한다. 무관한 option inventory로 흐름을 끊지 않는다.

모니터링 장과의 연결은 metric 이름이 아니라 conservation으로 한다. accepted, core sent/received, scheduler terminal, external terminal, resources released가 generation/cohort 안에서 닫히는지 본다. gap이 생긴 edge가 probe와 source owner를 결정한다.

마지막 미확인 항목은 selected runtime topology와 실제 stage latency다. static source에서 가능한 client classes와 ordering을 확인하되 배포 config/trace 없이는 선택됐다고 쓰지 않는다. 후속 검증에 필요한 initialized class, channel IDs와 timestamps를 명시한다.

21장을 빠져나갈 때 request는 scheduler에 “들어갔다”는 한 단어가 아니다. 어느 함수와 channel을 지나 어떤 generation으로 admission됐고, output/cancel이 역방향으로 어느 owner를 닫는지 추적 가능한 상태다.

장 종료 레코드에는 confirmed source transition, expected hand calculation과 runtime unknown을 분리한다. 특히 send 성공과 core receive, scheduler admission, engine terminal과 KV/resource free를 같은 event로 합치지 않는다.

수정 뒤에는 원래 failing race와 normal passing request를 같은 generation ledger로 재검증한다. terminal count뿐 아니라 oldest residue, output queue와 tombstone expiry까지 확인해 cleanup이 단지 지연된 것을 성공으로 오판하지 않는다.

이 최종 기준이 충족되면 22장에서 두 구현의 이름을 비교하는 대신 IPC ownership과 cancel conservation을 동일 좌표로 비교할 수 있다.

동일 fixture와 generation ledger를 다음 revision 정밀 감사에도 그대로 재사용한다.
