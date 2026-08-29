# 19장. OpenAI 호환 API의 약속과 경계

같은 JSON을 vLLM과 SGLang에 보냈는데 한 서버는 tool call을 만들고 다른 서버는 평문을 내놓았다고 하자. 둘 다 HTTP 200이고 응답에는 `choices`가 있다. 어느 쪽이 호환인가? endpoint 이름과 JSON key만 비교해서는 답할 수 없다. 메시지가 어느 chat template로 렌더됐는지, 어떤 token IDs와 stop 조건이 engine request가 됐는지, stream의 delta와 finish가 어떤 상태 전이를 나타내는지까지 따라가야 한다.

OpenAI 호환 API는 흔히 client SDK가 요청을 보내고 응답을 parse할 수 있다는 뜻으로 쓰인다. 그 말이 같은 tokenizer, prompt bytes, sampling order, tool grammar, logprob 의미, disconnect 이후 자원 수명까지 보장하지는 않는다. 호환성의 층을 분리하지 않으면 schema 호환을 모델 동작 동등성으로 과장한다.

이 장은 request schema에서 validation·normalization, rendered prompt, engine request, output collector, streaming chunks, finish·error·cancel까지 한 요청을 걷는다. 고정 source는 vLLM `v0.27.1` commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang `v0.5.18` commit `71de97b264b04dcd514cf904003028aefe9775c8`, llama.cpp `v0.2.0` commit `bb4caa7540188872173c44d161602d9271386413`, Transformers `v5.15.1` commit `550d7b3834670483a4df436541272c055dc364bf`다. source만 읽고 실제 server나 model은 실행하지 않는다.

## 19.1 한 streaming request의 JSON에서 terminal까지 걷는다

Python client가 `/v1/chat/completions`에 `model`, `messages`, `temperature`, `stream`을 보낸다고 하자. 네 field를 server가 받아들이고 familiar한 response object를 돌려주면 schema 표면은 호환된다. 그러나 assistant 답이 같을 것이라는 약속은 아직 없다.

첫째는 transport 계약이다. HTTP method와 path, content type, authentication, status code, SSE framing과 연결 종료 규칙이다. SDK가 연결하고 JSON을 parse할 수 있는가를 결정한다. proxy timeout과 buffering도 이 층에서 stream 체감을 바꾼다.

둘째는 schema 계약이다. 어떤 request fields를 허용하고 required·default·null을 어떻게 해석하는지, unknown field를 reject하거나 무시하는지, response object가 어떤 keys와 types를 갖는지다. 같은 이름도 허용 범위와 default가 다를 수 있다.

셋째는 semantic normalization 계약이다. messages가 어느 template로 렌더되는지, special tokens와 generation marker를 어떻게 붙이는지, output cap과 stop, tools, response format을 engine constraints로 어떻게 바꾸는지다. 이 층부터 같은 JSON이 다른 token program이 된다.

넷째는 engine 계약이다. token IDs, sampling params, request ID, priority, adapters, cache identity와 output mode가 scheduler request가 되는 과정이다. API는 continuous batching과 paged cache를 숨기지만 latency와 cancellation에는 영향을 준다.

다섯째는 lifecycle 계약이다. 첫 chunk 이전 오류, partial output 뒤 오류, normal finish, length stop, tool finish, client disconnect가 각각 어떤 terminal state가 되는지다. finish reason을 생성한 시각, socket에 쓴 시각, client가 읽은 시각, engine 자원을 반환한 시각은 다르다.

API 표면이 보장하지 않는 것은 model revision, tokenizer files, chat template, random stream, floating reduction, batch composition, cache warm state와 backend kernel이다. `temperature=0`이어도 tie breaking과 logits processor가 같다는 보장은 없다. 같은 model alias가 같은 artifact를 가리킨다는 보장도 deployment가 별도로 해야 한다.

반대로 API가 자기 계약 안에서 지켜야 할 것도 있다. validation error를 성공 response로 숨기지 않고, stream choice index와 request ID를 일관되게 유지하며, terminal reason 뒤 content를 계속 보내지 않아야 한다. usage unit과 disconnect policy도 명시해야 한다.

호환성 사고는 어느 층이 달랐는지 말한다. 먼저 body와 schema, 다음 rendered prompt와 IDs, sampling·constraints, stream assembly와 cleanup 순으로 본다. model weight와 numeric backend는 그 뒤다.

이 구분이 왜 필요한지 첫 요청을 따라가 보자. client가 user message 하나와 `stream=true`를 보냈다. gateway는 body를 parse했고 API process는 request ID를 만들었다. chat compiler는 assistant marker를 붙였고 tokenizer는 27개 IDs를 만들었다. engine은 output cap 64와 stop IDs를 가진 request를 받았다. 세 token을 생성한 뒤 API는 role chunk, content chunk, finish chunk를 보냈다. 이 과정에서 호환성을 주장할 수 있는 지점은 하나가 아니다.

다른 server가 같은 body를 받아 29개 prompt tokens를 만들었다면 transport와 schema는 호환되지만 prompt semantic은 다르다. 27개 IDs까지 같고 max output이 128이라면 normalization이 다르다. engine params까지 같은데 first chunk에 role을 넣지 않았다면 stream state machine이 다르다. content까지 같지만 disconnect 뒤 generation이 계속된다면 lifecycle이 다르다. 한 단어 대신 이렇게 좌표를 붙이면 논쟁이 검증 가능한 질문으로 바뀐다.

API가 숨기는 내부 구조도 약속의 범위를 좁힌다. continuous batching은 같은 request를 다른 neighbors와 실행할 수 있다. paged cache hit는 prompt 계산량을 줄일 수 있다. tensor parallel과 quantization은 numeric order를 바꾼다. 이 차이가 response schema를 깨뜨리지는 않지만 deterministic output이나 latency parity를 자동으로 보장하지 않는다.

client 관점에서는 두 가지 호환성이 특히 자주 섞인다. 코드 호환은 SDK 호출을 바꾸지 않아도 된다는 뜻이다. 행동 호환은 same input이 same semantics와 lifecycle을 갖는다는 뜻이다. 전자는 migration의 시작점이고 후자는 별도 differential test의 결과다.

책임 경계도 명확히 한다. server는 accepted fields와 defaults, rendered/tokenized 정책, finish·usage 의미를 공개해야 한다. deployment owner는 model/tokenizer/template revision을 고정해야 한다. client는 stream state machine과 retry/idempotency를 구현해야 한다. proxy는 SSE buffering과 timeout policy를 소유한다. 어느 한 component가 OpenAI-compatible이라는 label로 나머지 계약을 대신하지 않는다.

## 19.2 JSON을 normalized request와 engine state로 낮춘다

사용자는 JSON 한 덩어리를 보지만 server는 typed request로 바꾼다. required field, union type, range, mutual exclusion과 server flags를 검사한다. route가 body를 받았다는 사실과 engine request가 생겼다는 사실은 다르다. 4xx는 GPU를 호출하지 않은 정상적인 거절일 수 있다.

vLLM chat endpoint의 입구는 [`create_chat_completion`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/api_router.py#L53-L68)이다. 실제 validation·render·sampling conversion은 [`OpenAIServingChat._create_chat_completion`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L235-L388)에서 이어진다.

schema class는 data holder만이 아니다. validator와 conversion method가 default와 mutual exclusion을 정한다. vLLM request protocol과 sampling conversion은 [`protocol.py:350-850`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/protocol.py#L350-L850)에서 실제 symbol을 확인한다.

default는 생략에만 관계하지 않는다. `null`, empty list, zero가 생략과 같은지 다를 수 있다. `stop` 미지정, empty, null이 같은 engine state가 되는지 본다. `tools=[]`와 tools field 자체가 없는 경우 parser나 template selection이 달라질 수 있다.

unknown fields를 엄격 reject하면 typo를 빨리 찾지만 extension을 가진 client를 깨뜨린다. ignore하면 forward compatibility가 있지만 사용자가 option이 적용됐다고 오해한다. silent ignore를 호환성의 장점으로만 설명하지 않는다.

`model`은 routing alias이지 artifact hash가 아니다. 두 deployment가 같은 alias를 받아도 revision, quantization, tokenizer가 다를 수 있다. fixture에는 resolved model identity를 별도 기록한다.

길이 field에는 input tokens, requested output, model context, scheduler cap이 함께 작동한다. `max_tokens=128`을 max new tokens로 옮길 때 prompt length 포함 여부와 context overflow reject/truncate policy를 본다. response 길이가 같아도 prompt를 자른 server는 의미가 다르다.

sampling fields도 일대일 복사가 아니다. temperature 0은 greedy branch가 될 수 있고 top-p, top-k, repetition penalty, seed, logit bias 적용 순서가 engine마다 다르다. field 이름이 같아도 processor order는 다를 수 있다.

SGLang protocol과 chat serving은 [`protocol.py:1-300`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/protocol.py#L1-L300)과 [`serving_chat.py:1-360`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_chat.py#L1-L360)에서 typed request와 generation conversion을 잇는다.

llama.cpp server는 C++ route와 JSON parser가 같은 역할을 나눠 갖는다. [`server-context.cpp:1-620`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1-L620)과 endpoint literal에서 request parser와 task queue까지 잇는다.

Transformers는 OpenAI HTTP lifecycle 자체보다 generation config와 tokenizer/template 경계를 제공한다. adapter가 [`GenerationConfig`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/configuration_utils.py#L100-L360)에 무엇을 넘기는지 연결한다.

request checkpoint에는 raw body hash, parsed fields, defaults 적용 뒤 fields, warning/reject reason, resolved model, request ID를 둔다. 개인정보 messages 전체를 운영 log에 남기지 않고 synthetic fixture와 길이·hash를 쓴다.

### validation 사건: 받아들였지만 적용하지 않은 option

한 팀이 `logprobs=true`, `top_logprobs=5`를 보냈고 response에 logprobs가 없어 장애를 신고했다고 하자. HTTP 200이고 text는 정상이다. 첫 가설은 model backend가 logprobs를 지원하지 않는다는 것이다. 하지만 source walk는 더 앞에서 시작한다.

raw body에 두 fields가 실제로 있었는지, gateway가 allowlist로 body를 다시 만들며 제거하지 않았는지 본다. typed request가 fields를 보존했는지, validator가 조합을 허용했는지, sampling conversion이 engine logprob count를 만들었는지 본다. engine output이 logprobs를 가졌는지, response assembler가 serialize했는지까지 단계별로 간다.

네 개의 first divergence가 가능하다. gateway에서 사라지면 transport adapter owner다. typed model에서 default로 덮이면 schema owner다. engine params가 0이면 normalization owner다. raw output에는 있는데 JSON에 없으면 response owner다. backend를 교체하기 전에 이 네 checkpoint를 닫는다.

unknown field를 ignore하는 server에서는 typo `top_logprob`가 accepted response를 만들 수 있다. strict server는 422로 빨리 실패한다. 어느 행동이 더 compatible한지는 client 기대와 문서 계약에 달렸지만, silent non-application은 반드시 관찰 가능해야 한다. effective params나 warning이 없다면 사용자는 품질과 비용 실험을 잘못 해석한다.

범위 validation도 단순하다가 복잡해진다. negative temperature는 명백히 reject할 수 있지만 top-p 0, max tokens 0, empty stop, empty tools는 의미가 구현마다 다르다. zero output cap을 empty successful completion으로 볼지 invalid로 볼지, empty stop string이 즉시 stop인지 reject인지 source를 읽는다. boundary fixture가 필요하다.

mutual exclusion은 field 둘의 존재가 아니라 normalized semantics를 본다. tools와 response format grammar를 동시에 지원할지, stream과 best-of를 함께 허용할지, prompt logprobs와 chat multimodal을 함께 허용할지 engine capability가 결정한다. validator가 허용했다고 downstream path가 실제 구현됐다는 증거는 아니다.

resolved model과 adapter도 validation 이후 state다. model alias가 없으면 404 계열 error를 내는지, adapter name을 model처럼 해석하는지, tokenizer를 base와 adapter 중 어디서 가져오는지 확인한다. 같은 request body가 다른 resolved artifacts로 가는 것을 schema differential로 잡을 수 없다.

validation error response 자체도 contract다. HTTP status, error object type, message, parameter field, code가 client retry policy를 바꾼다. 4xx를 500으로 내면 client가 retry storm을 만들 수 있고, transient admission error를 400으로 내면 복구 기회를 잃는다. server 내부 exception type과 public error mapping을 연결한다.

vLLM route가 serving method 결과로 error response 또는 stream generator를 받는 분기를 보고, protocol model validator에서 field 범위와 conversion method를 잇는다. SGLang도 Pydantic request와 serving exception mapping을 연결한다. llama.cpp는 JSON helper가 missing/type/range error를 어떻게 response로 바꾸는지 route handler까지 본다. Transformers adapter는 generation config validation exception을 HTTP error로 번역하는 wrapper가 owner다.

실제 조사 장부에는 raw value, parsed value, effective value, consumer와 evidence를 둔다. `temperature`가 raw 0, parsed 0.0, effective greedy mode, consumer sampler branch라면 적용을 증명할 수 있다. raw field가 존재한다는 사실만으로 option effect를 주장하지 않는다.

복구 검증은 성공 response 하나가 아니다. typo, null, empty, min/max boundary, mutually exclusive pair, unsupported capability를 fixture로 만든다. 각 server가 동일하게 행동해야 한다고 미리 요구하지 않고, 문서화한 자기 contract와 일치하는지 먼저 본다. migration이 행동 parity를 요구하면 차이를 shim이나 client capability branch로 닫는다.

**messages를 token compiler 입력으로 정규화한다.**

두 server의 답이 다르면 model보다 rendered prompt를 먼저 비교한다. chat API는 structured messages를 model control-token sequence로 컴파일한다. role markers, BOS/EOS, assistant generation marker, tool schema와 multimodal placeholder가 모두 token program을 바꾼다.

vLLM은 [`_effective_chat_template_kwargs`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L180-L191)에서 template kwargs를 정리하고 [`render_chat_request`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L192-L218)로 messages를 engine prompt로 바꾼다.

Transformers canonical template application은 [`apply_chat_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L1540-L1840)에서 conversation, tools, documents, tokenize, generation prompt와 assistant mask를 처리한다. 같은 template string도 kwargs와 special-token policy가 다르면 IDs가 달라진다.

기본 template의 source를 적는다. tokenizer artifact, server override, registry fallback, request override가 precedence를 가질 수 있다. messages만 보존하고 effective template hash를 기록하지 않으면 재현이 안 된다.

role validation도 engine 이전 의미다. system/user/assistant/tool 순서, null content, tool call ID 매칭을 server가 얼마나 엄격히 검사하는지 다르다. 느슨한 server는 이상한 prompt를 만들고 엄격한 server는 4xx로 거절할 수 있다.

tool calling은 tools JSON을 model이 자동으로 이해하는 기능이 아니다. definitions를 template에 넣고, model output을 parser가 tool calls로 복원하며, optional grammar가 output을 제한한다. template, parser, constraint owner를 분리한다.

`tool_choice=auto`, 특정 function 강제, none은 서로 다른 prompt와 constraint가 된다. parser plugin이 없으면 tools를 받아도 평문 JSON을 content로 낼 수 있다. schema acceptance를 tool semantic 지원으로 해석하지 않는다.

SGLang chat path는 [`serving_chat.py`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_chat.py#L1-L420), tokenize boundary는 [`serving_tokenize.py:20-154`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_tokenize.py#L20-L154)에서 plain text와 chat을 나눈다.

llama.cpp는 [`llama-chat.cpp:1-320`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-chat.cpp#L1-L320)에서 template detection·application과 tool formatting을 읽고 server adapter가 messages를 이 구조로 바꾸는 곳을 잇는다.

tokenization parity는 visible text parity보다 강하다. 동일 text도 BOS 자동 추가, special recognition, normalization으로 IDs가 달라진다. checkpoint는 raw messages, effective template identity, rendered bytes, token IDs, special-token mask, prompt count다.

multimodal content는 fetch policy, placeholder count, processor output과 embedding splice가 추가된다. OpenAI-style content schema를 받는다는 사실과 같은 image preprocessing을 한다는 사실은 다르다.

render incident는 tool 하나, system 한 줄, user 한 줄인 synthetic conversation으로 만든다. 첫 다른 byte가 role marker, tool schema order, generation marker 중 무엇인지 찾는다. logits를 보기 전에 compiler 차이를 닫는다.

**tool calling 사건: JSON처럼 보이는 text와 tool call은 다르다**

사용자가 weather tool 하나를 정의하고 특정 function을 강제했다고 하자. server A는 `finish_reason=tool_calls`와 structured arguments를 반환한다. server B는 assistant content에 JSON text를 반환하고 `finish_reason=stop`을 붙인다. visible characters만 보면 arguments가 같을 수 있지만 API lifecycle은 다르다.

먼저 tools schema가 prompt에 어떻게 렌더됐는지 본다. function name, description, JSON schema property order, required fields와 tool selection marker가 token stream에 들어갔는지 확인한다. model이 tool protocol을 학습한 template와 다른 marker를 쓰면 parser 이전부터 behavior가 갈린다.

다음은 constraint다. 특정 tool 강제가 grammar를 생성해 model이 function name과 arguments grammar 밖 token을 고르지 못하게 하는지, prompt instruction만으로 유도하는지 구분한다. 두 server가 같은 template여도 constraint 유무가 raw IDs를 바꾼다. 강제 선택이라는 public field가 얼마나 강한 보장을 의미하는지 문서화해야 한다.

세 번째는 parser다. model raw output이 special tool tokens와 JSON fragment를 포함한다면 parser가 content와 tool call을 분리하고 call ID, function name, arguments를 만든다. parser가 model family에 맞지 않으면 raw output은 맞아도 structured response가 틀린다. raw generated IDs/text checkpoint가 parser 전 oracle이다.

stream에서는 parser state가 token 사이에 살아 있다. function name이 여러 chunks로 쪼개질 수 있고 arguments JSON string은 중간에 닫히지 않았다. parser는 아직 content인지 tool call인지 결정하지 못한 prefix를 보류할 수 있다. 성급히 content로 commit하면 나중에 tool delta로 되돌릴 수 없다.

call index와 ID도 lifecycle identity다. 두 parallel tool calls가 interleave되면 fragments를 올바른 call에 붙여야 한다. finish 전에 call ID가 바뀌거나 같은 index를 재사용하면 client assembler가 arguments를 섞는다. `n>1`이면 choice index와 tool index의 두 축이 있다.

tool result message가 다음 request에 들어올 때 call ID matching을 validation하는지도 본다. 이전 assistant tool call에 없는 ID를 user가 tool result로 보내면 strict reject, lenient render가 가능하다. lenient path가 model에게 혼란스러운 protocol을 만들 수 있다.

SGLang과 vLLM에서 parser selection option, model-specific parser registry, serving chat의 tool response conversion을 이어 읽는다. llama.cpp는 chat template tool formatting과 server output parser가 어느 형식을 지원하는지 본다. Transformers는 template에 tools를 넣을 수 있지만 OpenAI tool response parser와 SSE state machine을 library core가 자동 제공한다고 가정하지 않는다.

tool parity fixture는 raw messages와 tools를 canonical JSON으로 고정한다. effective template bytes, prompt IDs, constraint identity, raw output IDs, parser events, final chunks를 비교한다. schema key 순서가 template output을 바꾼다면 canonicalization policy도 기록한다.

first divergence가 prompt라면 model/backend를 바꾸지 않는다. prompt는 같고 raw output이 다르면 constraint·sampling·model path를 본다. raw output까지 같고 structured response만 다르면 parser와 stream assembler다. 이 세 갈래를 닫으면 tool calling을 하나의 magic feature로 부르지 않게 된다.

보안 측면에서는 tool arguments가 model-generated untrusted data다. parser가 JSON을 만들었다고 실행 권한이 생기지 않는다. application은 tool allowlist, schema validation, authorization과 side-effect confirmation을 소유한다. OpenAI-compatible response shape는 안전한 tool execution을 보장하지 않는다.

복구는 특정 weather example만 통과하는 것으로 끝내지 않는다. escaped Unicode, nested object, parallel calls, empty arguments, invalid JSON recovery, content와 tool call 혼합, stream fragmentation을 본다. parser가 malformed output을 error, content fallback, partial tool 중 무엇으로 만드는지 계약을 고정한다.

**normalized option을 실행 가능한 engine field로 바꾼다.**

rendered IDs 뒤 serving layer는 request ID와 sampling/constraint params를 붙인다. API request 하나가 engine sequence 하나인지, `n`이나 best-of로 여러 candidates를 만드는지, prompt logprobs가 별도 work를 요구하는지 결정한다.

vLLM conversion은 [고정 source `serving.py:235-388`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L235-L388)에서 request ID, priority, adapters, trace headers와 engine generate를 구분한다.

stop에는 EOS ID, additional stop IDs, decoded stop strings, max tokens, grammar accept와 tool parser terminal이 있다. `stop` 문자열을 token matcher로 처리하는지 streaming text suffix로 처리하는지 확인한다. stop text를 response에 포함할지도 별도 policy다.

stop string은 token boundary를 가로지를 수 있어 stream이 가능성 있는 suffix를 보류한다. 이미 보낸 bytes는 되돌릴 수 없다. API option과 engine detokenizer가 같은 include/exclude policy를 써야 한다.

logprobs는 response key 하나의 비용이 아니다. top alternatives를 보존하고 device에서 CPU로 옮기며 token bytes와 offsets를 조립한다. prompt logprobs는 prompt positions score를 요구한다. generation-only fast path와 memory가 달라질 수 있다.

logprob 의미는 natural log 여부, sampled token 포함, top-k stage, `-inf` serialization, byte representation, 첫 prompt token의 null policy를 포함한다. shape만 같아도 값 semantic은 다를 수 있다.

seed는 continuous batching에서 request-local RNG인지 batch-global RNG인지 묻는다. 다른 requests arrival가 같은 request output을 바꾸지 않는다는 범위를 문서화해야 한다.

structured constraint는 automaton state를 request 또는 candidate별로 유지한다. cancel/retry 뒤 state가 다른 request에 재사용되지 않아야 한다. API response format이 어느 constraint object가 되는지 source를 연결한다.

Transformers adapter는 fields를 logits processors, stopping criteria, generation config와 streamer로 바꾼다. [`generation/utils.py:2100-2500`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2100-L2500)에서 config validation, processor preparation과 mode dispatch를 읽는다.

SGLang은 [`serving_chat.py`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_chat.py#L180-L420)와 [`tokenizer_manager.py:1-300`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L1-L300)에서 internal request와 output queue 경계를 찾는다.

llama.cpp는 parsed params를 task와 sampler chain으로 바꾼다. [`server-task.cpp:1-420`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-task.cpp#L1-L420)에서 task ID, slot, stream과 cancel을 잇는다.

engine checkpoint는 final IDs, request incarnation, max new tokens, stop IDs/strings, processor order, RNG identity, constraints, candidate count, logprob mode, priority와 arrival time이다. raw fields만 기록하면 normalization 차이를 놓친다.

**logprobs 사건: 같은 token인데 확률 값이 다르다**

두 server가 greedy로 같은 token IDs를 냈지만 selected token logprob가 다르다고 하자. 첫 반응은 low-precision kernel 차이일 수 있다. 그러나 API가 반환하는 logprob가 어느 distribution에서 계산됐는지 먼저 정해야 한다.

raw model logits에 temperature, repetition/frequency/presence penalty, logit bias, bad-word나 grammar mask가 순서대로 적용될 수 있다. logprob를 raw logits softmax에서 계산하는지, processors 적용 뒤 sampling distribution에서 계산하는지에 따라 값이 다르다. token 선택은 같아도 probability는 달라진다.

temperature가 0인 greedy path도 주의한다. 실제 sampling distribution을 만들지 않고 argmax할 수 있다. API가 logprobs를 요청하면 별도로 log-softmax를 계산할 수 있다. top alternatives가 processor 전인지 후인지 source conversion과 output processor를 읽는다.

작은 numeric fixture를 보자. raw logits `[2,1,0]`에서 token 0 logprob는 `2-log(e²+e¹+1)`이다. token 2에 logit bias 3을 더하면 adjusted logits `[2,1,3]`이고 선택 token은 2로 바뀐다. API logprob가 bias 전 raw distribution을 보고하면 selected token 2가 낮은 logprob처럼 보인다. bias 후라면 가장 높다. 어느 값이 public contract인지 명시해야 한다.

top-logprobs count도 response 배열 길이만의 문제가 아니다. sampled token이 requested top-k 밖이면 별도로 포함해 k+1개가 될지, sampled token을 포함하도록 alternatives 하나를 밀어낼지, exact k를 유지할지 server마다 다를 수 있다. client는 shape assumption보다 documented semantics를 따라야 한다.

token text와 bytes는 tokenizer boundary다. 한 Unicode character가 여러 token bytes로 나뉠 수 있고 invalid partial UTF-8가 replacement text로 보일 수 있다. logprob entry의 token string과 bytes array, text offset을 어떻게 만드는지 detokenizer와 잇는다. visible token string만 key로 alternatives를 합치면 서로 다른 IDs를 잃는다.

prompt logprobs에서는 첫 input token에 preceding context가 없을 수 있어 null을 반환할 수 있다. chat template가 BOS를 붙였다면 first user-visible token의 conditional context는 system markers를 포함한다. raw user text 기준 offset과 rendered prompt token 기준 position을 혼동하지 않는다.

`echo`나 prompt inclusion option이 있으면 completion response text와 logprob sequence alignment가 달라진다. stop string으로 visible suffix를 trim해도 sampled stop tokens의 logprobs를 usage나 response에 포함할지 정책이 필요하다. finish reason과 logprob array 길이를 함께 본다.

성능 사건에서는 logprobs on/off workload를 분리한다. logits 보존, top-k selection, CPU transfer, JSON serialization이 ITL과 network bytes를 늘린다. model kernel time이 같아도 API output handler가 병목일 수 있다. stream client가 느리면 큰 logprob chunks가 backpressure를 키운다.

first divergence 순서는 prompt IDs, selected raw logits slice, processors 적용 뒤 logits, selected token ID, engine logprob object, API serialized entry다. raw logits부터 다르면 model/backend, processor 뒤부터면 normalization/order, engine object는 같고 JSON만 다르면 serializer다.

비실행 source 감사는 sampling params가 logprob count와 prompt logprob flag를 어떻게 만들고 model runner output이 output processor로 전달되는지 잇는다. 실제 numeric tolerance는 hardware execution이 필요하므로 미검증으로 남긴다. fixture와 observation field는 준비할 수 있다.

복구 검증에는 bias 전후가 다른 fixture, top-k 경계, sampled token outside top alternatives, Unicode bytes, first prompt token null, stop trim, stream chunks를 넣는다. text가 같다는 이유로 logprob semantic parity를 주장하지 않는다.

## 19.3 SSE frame은 한 choice의 commit state를 운반한다

non-stream은 output을 모아 한 JSON을 만든다. stream은 append-only chunks를 client에 commit한다. 아직 확정되지 않은 UTF-8, stop prefix, tool JSON fragment를 얼마나 보류할지 결정해야 한다. chunk boundary는 token boundary와 같지 않다.

SSE event가 server generator에서 yield되어도 proxy buffering 때문에 client TTFT가 늦을 수 있다. generator yield, ASGI write, proxy flush, client read timestamps를 나눈다.

vLLM [`chat_completion_stream_generator`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L422-L843)는 engine outputs를 deltas로 바꾼다. full generator는 [같은 파일 `:844-1010`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L844-L1010)에서 aggregate한다. 둘은 serializer만 다른 것이 아니라 partial commit과 error lifecycle이 다르다.

첫 chunk는 role만 담고 content가 비어 있을 수 있다. protocol first chunk와 사용자가 읽을 first content를 나눈다. tool stream에서는 name과 arguments가 여러 deltas로 나뉜다.

choice chunks가 interleave되어도 같은 index delta order는 보존되어야 한다. 한 choice가 finish해도 다른 choice는 계속될 수 있다. response-level terminal과 choice-level finish를 구분한다.

tool arguments는 partial JSON fragment다. chunk마다 parse하지 않고 call index와 ID별로 누적해 finish 뒤 parse한다. server parser와 client assembler가 같은 state machine을 가져야 한다.

usage는 terminal 근처 별도 chunk로 올 수 있다. prompt tokens는 rendered IDs를 세는지, completion은 emitted·accepted·sampled 중 무엇인지, cached details를 어떻게 표시하는지 확인한다. disconnect 전에 usage를 못 받았다고 계산 work가 0이었던 것은 아니다.

finish reason은 empty delta와 함께 올 수 있다. content 없는 chunk를 버리면 terminal을 잃는다. stop, length, tool calls와 internal enum mapping을 source에서 읽는다.

partial output 뒤 error는 HTTP status를 500으로 되돌릴 수 없다. error event나 connection close policy가 필요하다. retry하면 duplicate prefix가 생길 수 있다.

SGLang stream conversion은 [`serving_chat.py:360-700`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_chat.py#L360-L700)에서 delta, finish와 usage를 잇는다. llama.cpp는 [`server-context.cpp:620-1040`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L620-L1040)에서 result queue와 response formatter를 잇는다.

Transformers [`TextIteratorStreamer`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/streamers.py#L160-L320)는 token outputs를 queue와 text chunks로 넘기지만 SSE, usage, finish, disconnect를 완성하지 않는다. server wrapper가 lifecycle adapter를 소유한다.

stream parity는 visible text만 비교하지 않는다. role, content/tool fragments, choice index, finish, usage, sentinel과 error를 state machine으로 검사한다.

### stream state machine을 한 choice로 그린다

한 choice는 `INIT→ROLE_SENT→CONTENT_OR_TOOL→FINISHED`로 생각할 수 있다. error와 cancellation은 어느 state에서도 terminal로 갈 수 있다. INIT에서 content가 먼저 와도 protocol이 허용할 수 있지만 client가 role default를 어떻게 정하는지 확인한다. FINISHED 뒤 delta는 오류다.

server 전체는 여러 choices를 multiplex한다. request-level stream은 모든 choices가 terminal이 되고 optional usage와 sentinel을 보낸 뒤 닫힌다. choice 0이 stop되고 choice 1이 생성 중일 수 있다. 첫 finish chunk를 보고 socket을 닫는 client는 나머지를 잃는다.

role chunk가 비어 있다는 표현은 부정확하다. content delta는 비어도 role이라는 protocol state를 commit한다. TTFT를 first SSE, first semantic delta, first visible content로 세 가지 정의할 수 있다. dashboard와 client benchmark가 어느 것을 재는지 맞춘다.

text streaming은 offline decode의 prefix가 항상 단조로 증가한다는 가정이 필요하다. incomplete UTF-8와 cleanup, stop-prefix 보류가 이를 깨뜨릴 수 있다. server는 streaming-safe detokenizer로 확정 suffix만 보내거나 수정 event를 지원해야 한다. OpenAI-style delta는 보통 append semantics라 이전 text를 되돌리기 어렵다.

예를 들어 stop string이 `END`이고 생성 suffix가 `EN`까지 왔다면 아직 content로 보낼지 보류할지 결정한다. 다음 token이 `D`면 stop이 완성되어 세 글자를 제외할 수 있다. 이미 `EN`을 보냈다면 되돌릴 수 없다. 다음이 `X`면 보류한 `ENX`를 commit한다. matcher의 pending buffer가 stream state다.

tool parser도 같은 commit 문제를 가진다. raw prefix가 ordinary JSON content인지 tool envelope인지 늦게 확정될 수 있다. server가 role/content를 먼저 보내고 나중에 tool call로 바꾸면 client state가 모순된다. parser start marker가 확정되기 전 text를 보류하는 이유다.

usage chunk는 choices가 empty일 수 있다. client schema가 모든 chunk에 choice 하나 이상을 요구하면 실패한다. include-usage option이 typed request에서 assembler branch로 가는 source를 따라가고, terminal finish와 usage/sentinel order를 고정한다.

network backpressure는 engine output과 delivered output을 갈라놓는다. generator가 queue에서 tokens를 소비하지 못하면 output queue가 차고 scheduler나 detokenizer에 영향을 줄 수 있다. 또는 server가 engine outputs를 계속 drain하고 socket buffer에 쌓아 memory가 늘 수 있다. slow client policy와 buffer bound를 확인한다.

partial error retry는 application 문제를 만든다. client가 이미 20 characters를 사용자에게 보여 준 뒤 connection이 끊겼다. 자동 retry가 새 response 전체를 이어 붙이면 prefix가 중복된다. request idempotency와 resume offset이 없다면 안전한 transparent resume를 약속할 수 없다.

stream recorder는 event sequence number, choice index, delta kind, payload length, finish reason, usage presence, generator yield와 transport timestamps를 남긴다. content 자체는 민감하므로 synthetic test나 approved hash를 쓴다. 마지막 event가 어디서 사라졌는지 이 기록으로 찾는다.

### streaming 사건: 마지막 문장은 왔는데 요청이 닫히지 않는다

client UI에는 완전한 문장이 보이지만 loading spinner가 멈추지 않는다고 하자. model 생성은 끝났고 scheduler도 request를 제거했다. 이때 model이나 GPU를 조사할 이유가 없다. terminal protocol path를 걷는다.

engine output의 final flag와 finish enum이 output generator에 도착했는지 확인한다. assembler가 finish chunk를 만들었는지, SSE encoder가 yield했는지, ASGI와 proxy가 flush했는지, client parser가 event를 받았는지 본다. 각 timestamp가 first divergence를 만든다.

assembler가 empty delta를 optimize away하면서 finish reason까지 버릴 수 있다. usage-only chunk를 malformed로 판단해 exception이 나고 sentinel을 못 보낼 수도 있다. proxy가 response buffering을 켜 마지막 작은 frame을 connection close까지 보류할 수도 있다. client가 `[DONE]`만 terminal로 보고 finish reason만 받은 stream을 기다릴 수도 있다.

negative fixture는 content 없는 finish, finish와 content가 같은 chunk, usage-only chunk, error after partial, two choices with staggered finish를 포함한다. server state machine과 client assembler를 함께 검증한다. 한쪽만 통과하면 실제 interoperability를 증명하지 못한다.

복구는 spinner가 사라졌다는 UI observation으로 끝내지 않는다. engine final→generator finish yield→transport flush→client terminal latency를 보고, 선택별 terminal exactly-once, finish 뒤 delta 0, stream-level close를 검사한다. proxy 설정도 artifact로 고정한다.

**frame 종류를 server state로 해석한다.**

streaming 응답을 `data:` 문자열 연속으로만 보면 어떤 frame이 상태를 바꾸는지 놓친다. 최소한 role/content delta, tool-call delta, logprobs, usage, terminal finish, error를 구분한다. transport heartbeat나 `[DONE]` sentinel은 application payload와 별 층이다.

서버 state는 `accepted → running → externally_committed → terminal → cleaned`로 둔다. 첫 사용자-visible delta를 내보내면 externally committed다. 그 뒤 다른 provider로 fallback해 처음부터 다시 보내면 중복 content/tool side effect가 생긴다. terminal frame을 보냈다고 scheduler/KV cleanup이 끝난 것은 아니다.

**content delta와 UTF-8/token 경계를 분리한다**

한 token이 완전한 Unicode 문자와 일치하지 않을 수 있고 incremental detokenizer가 byte를 보류할 수 있다. frame boundary는 token boundary, character boundary와 다를 수 있다. client가 delta를 단순 concat했을 때 최종 non-stream text와 같아야 하지만 각 frame이 valid 독립 문장일 필요는 없다.

fixture는 multibyte Korean, emoji sequence, combining mark, leading-space token을 포함한다. server의 emitted text cursor와 token cursor를 따로 기록한다. retry/reconnect에서 character count를 token offset으로 사용하지 않는다.

**usage frame은 누적·증분·terminal 중 무엇인가**

usage가 매 chunk 누적값인지 마지막에만 오는지, include_usage option에서 별 empty-choice frame인지 구현 계약을 확인한다. prompt, completion, total tokens의 source도 tokenizer estimate인지 engine committed tokens인지 구분한다.

completion token을 chunk별 delta로 더하는 client가 누적 usage를 매번 합하면 과금이 중복된다. 반대로 terminal usage frame이 network error로 유실되면 성공 content가 전달됐지만 accounting이 비게 된다. logical request/attempt ledger와 server terminal state를 연결한다.

**logprobs는 content와 같은 commit 단위를 가져야 한다**

한 emitted token에 token text, token ID, logprob, top alternatives와 byte representation이 대응한다. detokenizer가 여러 token을 하나의 text delta로 합칠 수 있으므로 text frame 수와 logprob item 수를 같다고 가정하지 않는다.

fixture는 two-token one-text와 one-token delayed-text를 포함한다. logprob cursor, token cursor, text cursor를 별도로 두고 final join에서 일관성을 확인한다. stop token을 visible text에서 제거하더라도 raw selected token과 logprob, usage에는 남을 수 있다.

**tool call은 index별 incremental JSON state다**

tool name과 arguments가 여러 frame에 걸쳐 올 수 있다. arguments delta `{"city":`와 `"Seoul"}`은 중간에는 valid JSON이 아니다. client는 tool-call index/id별 buffer를 유지하고 finish reason 또는 terminal에서 parse한다. frame 하나마다 JSON parse error를 내면 정상 stream을 실패로 처리한다.

두 tool call이 interleave될 수 있다면 index와 stable tool-call ID가 필요하다. content delta와 tool delta가 같은 choice에 섞일 때 허용 contract를 확인한다. application은 terminal/validated tool call 전에 side effect를 실행하지 않는 편이 안전하다.

**error frame과 transport close를 분리한다**

HTTP status를 이미 200으로 commit한 뒤 engine error가 나면 status code를 바꿀 수 없다. server는 stream 내 error event를 보내거나 connection을 끊을 수 있다. 두 경우 client-visible semantics가 다르다. structured error에는 request/attempt identity, code, retryability와 terminal 여부가 필요하다.

connection reset은 server error, proxy timeout, client cancel을 구분하지 못할 수 있다. server trace의 engine terminal과 transport write result, client evidence를 함께 본다. error frame 뒤 content/usage delta가 더 오면 terminal ordering 위반이다.

**cancel은 client intent와 engine cleanup 두 terminal이다**

disconnect 감지는 abort intent를 만든다. request stream/engine queue/scheduler running state/KV pages가 정리돼야 resource terminal이다. client는 connection이 닫힌 순간 완료됐지만 server는 cleanup을 계속할 수 있다. cancellation latency와 orphan work를 관측한다.

race는 finish와 disconnect가 동시에 오는 경우다. exactly one external terminal과 idempotent cleanup을 보장한다. abort가 이미 finished request를 찾지 못해도 cleanup ledger는 terminal을 기록해야 한다. request ID 재사용을 막고 generation을 붙인다.

**frame conservation과 partial-stream incident를 검증한다.**

요청 S19가 role frame, content 세 frame, tool-call 두 frame, usage frame, finish frame을 낸다고 하자. sequence number 0..7과 event kind를 저장한다. 각 choice/tool index별 monotonic cursor와 server request generation을 둔다.

**정상 event conservation**

accepted logical request 1개는 external terminal exactly one, engine terminal exactly one, cleanup terminal exactly one을 가져야 한다. content/token delta 수는 가변이지만 committed token count와 usage completion count가 일관되어야 한다. tool-call buffer는 terminal에서 complete 또는 explicit invalid error다.

`accepted = success + error + cancelled + in_flight` 보존식을 window별로 본다. transport disconnect count와 cancelled engine count가 같은 순간에 일치할 필요는 없지만 장기적으로 orphan gap이 닫혀야 한다. retry attempt가 있으면 logical request와 physical attempt를 분리한다.

**partial stream 뒤 error incident**

S19는 content “서울의 날”까지 12 token을 commit한 뒤 tool parser error가 났다. server는 structured error를 보내고 finish frame도 보냈다. client A는 error에서 terminal 처리했고 뒤 finish를 duplicate terminal로 기록했다. client B는 finish만 보고 partial content를 성공으로 저장했다.

root edge는 engine error 자체와 별개로 stream terminal contract가 두 번 발생한 것이다. 상태기계는 error를 terminal로 정했다면 finish를 추가로 보내지 않거나, finish frame 안에 error semantics를 표준적으로 담아야 한다. client도 first terminal 이후 delta를 protocol violation으로 기록한다.

usage는 committed 12 tokens인지 generated 13 including hidden stop/error token인지 정책을 명시한다. billing과 visible text를 같은 count로 가정하지 않는다. server ledger에 generated, committed-to-client, billed counts를 둔다.

**disconnect-write race incident**

engine이 finish output을 만들었지만 proxy가 connection close를 먼저 알렸다. output loop는 finish frame write와 abort를 동시에 시작했다. scheduler는 abort로 pages를 반환했고 writer는 stale output reference를 serialize했다. 이 race는 request state generation과 output ownership으로 막는다.

writer가 terminal ownership을 획득했으면 cleanup이 output serialization의 last consumer를 기다리거나 stable copy를 가져야 한다. abort가 terminal owner면 writer는 더 이상 frame을 내보내지 않는다. global synchronize가 아니라 request-local event/state transition이 필요하다.

**replay와 idempotency**

client가 terminal을 못 보고 동일 idempotency key로 재요청할 수 있다. server가 이전 committed stream을 replay할지 새 generation을 만들지 contract가 필요하다. streaming bytes 전체를 보존하지 않으면 exact replay를 약속하지 않는다. tool side effect가 있었다면 application-level idempotency가 더 중요하다.

request ID와 idempotency key, attempt ID를 분리한다. 같은 logical request의 새 attempt가 old output queue/KV state와 충돌하지 않게 generation namespace를 둔다. usage/billing은 logical/attempt ledger에서 중복을 reconcile한다.

**한 요청의 frame timeline을 운영 상태로 복원한다.**

요청 ID R, attempt A0가 시각 t0에 accepted되고 t1에 engine queued, t2에 first token selected, t3에 first frame write, t4에 client receive, t5에 engine finish, t6에 usage/finish write, t7에 cleanup됐다고 하자. TTFT도 어느 clock인지 구분한다. server first-frame은 t3-t0, client TTFT는 t4-client-send다. engine compute TTFT는 t2-queue/admit 기준일 수 있다.

**frame conservation table**

ledger 행은 `(logical_request,attempt,choice,frame_seq)`다. 열은 engine token range, text cursor before/after, tool cursor, logprob range, usage snapshot, frame kind, write result, external commit state다. terminal frame은 terminal owner와 reason을 가진다.

content delta가 빈 문자열이어도 role/tool/finish 정보를 운반할 수 있으므로 empty payload를 무조건 drop하지 않는다. 반대로 heartbeat는 choice state를 mutate하지 않는다. serializer가 empty choice usage frame을 만드는 contract도 구분한다.

choice n>1이면 각 choice가 finish reason과 token cursor를 별도로 갖고 전체 response terminal은 모든 choice terminal 또는 global error에서 결정된다. choice 하나가 끝났다고 request cleanup을 시작하지 않는다. streaming multiple choices 지원 여부를 schema/implementation에서 확인한다.

**backpressure가 state lifetime을 늘리는 경로**

slow client에서 socket write가 막히면 engine output queue가 쌓인다. engine generation을 계속할지 pause/backpressure할지, queue limit에서 cancel/error할지 정책이 필요하다. output buffer가 KV와 request state cleanup을 붙잡으면 GPU work가 끝난 뒤에도 memory가 남는다.

frame bytes, queued frames, oldest frame age와 engine pending output을 관측한다. prompt/response content를 metric label로 넣지 않는다. serialization CPU와 socket wait, proxy buffer를 분리한다. ITL spike가 model decode가 아니라 write flush에서 생길 수 있다.

client disconnect detection이 늦으면 orphan generation이 계속된다. polling interval, ASGI cancellation, writer exception과 engine abort 전달 지연을 timeline에 둔다. abort latency와 wasted generated tokens를 계산한다. retry로 같은 logical request가 새 attempt를 시작하면 old orphan과 동시 실행될 수 있다.

**usage와 billing의 세 수치**

generated token은 engine이 선택/commit한 token, delivered token은 client-visible frame write가 성공한 token, billed token은 정책 ledger에 반영한 token이다. 정상 완료에서는 셋이 같을 수 있지만 disconnect/error에서는 갈린다. 정책은 어떤 count를 과금하는지 명시하고 감사 가능해야 한다.

prompt usage도 cache hit가 compute를 줄였다고 token count가 0이 되는 것은 아니다. API usage, internal scheduled/computed token과 비용 원장을 구분한다. gateway/provider billing은 provider reported usage와 local estimate를 reconcile한다.

예를 들어 generated20, successful writes12, client receive는 불명, billed20이면 사용자 이의 처리에 delivered evidence가 부족하다. write 성공이 client consumption을 보장하지 않지만 server commitment는 보여 준다. proxy/client receipt가 필요하면 end-to-end acknowledgment contract가 별도로 필요하다.

**logprobs payload 비용 계산**

completion token 100, top_logprobs 5일 때 최소 500 alternative entries와 selected token metadata를 serialize한다. 각 entry가 token string, logprob, bytes를 갖으면 payload가 text-only보다 크게 늘어난다. 정확한 byte는 token 길이/JSON serializer에 따라 달라지므로 representative fixture로 측정한다.

CPU serialization과 network bandwidth, client parse가 ITL을 늘릴 수 있다. GPU sampler가 top-k values를 준비하는 비용도 있다. option 효과를 “확률을 보여 준다”로 끝내지 않고 sampler state→host transfer→serialization→backpressure lifetime으로 잇는다.

logprobs off/on A/B는 같은 generated IDs를 고정하거나 teacher-forced fixture로 output workload를 맞춘다. 자연 생성 token이 달라지면 payload 길이와 decode step이 달라져 비용 비교가 흐려진다.

**tool-call incremental parser의 실패 경계**

tool arguments가 10 frames에 걸쳐 온다면 parser는 append state와 nesting/string escape를 보존할 수 있다. 단순 concat 후 terminal parse는 안전하지만 early validation/streaming tool UI를 원하면 incremental parser contract가 필요하다. incomplete와 invalid를 구분한다.

cancel 시 partial tool buffer를 application에 실행 가능한 call로 넘기지 않는다. error terminal 뒤 buffer를 폐기하고 audit digest/length만 남긴다. tool-call ID가 다른 attempt에서 재사용되지 않게 generation을 붙인다. parallel calls는 index reorder를 검증한다.

tool side effect는 server stream terminal과 별 application terminal이다. client가 complete tool call을 받아 외부 API를 호출한 뒤 response stream이 끊기면 retry가 side effect를 중복할 수 있다. gateway/agent layer에 idempotency key를 전달하는 정책이 필요하다. 20장의 retry 경계로 넘긴다.

**structured error taxonomy**

validation error는 header commit 전에 4xx로 반환할 수 있다. admission/overload도 first frame 전이면 status와 retry hint를 줄 수 있다. engine error가 commit 후 발생하면 in-stream terminal error가 필요하다. serializer/socket failure는 error frame조차 보내지 못할 수 있다.

client cancel은 provider failure로 계산하지 않는다. server shutdown/drain은 retryable 여부와 partial commit을 함께 본다. timeout은 gateway/client/server/engine 중 어느 clock이 만료됐는지 owner를 적는다. 모두 `500` 하나로 접으면 retry storm과 잘못된 SLA가 생긴다.

error metric은 bounded class/stage/commit state를 label로 둔다. raw message와 request ID는 trace에 둔다. error-before-commit과 after-commit을 분리하면 fallback 안전성을 판단할 수 있다.

**cancellation race의 상태 전이표**

states를 `RUNNING`, `COMMITTING_TERMINAL`, `ABORTING`, `TERMINAL`, `CLEANED`로 둔다. finish와 disconnect가 compare-and-set으로 terminal ownership을 경쟁한다. winner가 external terminal policy를 결정하고 loser는 idempotent cleanup에 합류한다.

`RUNNING→COMMITTING_TERMINAL` 뒤 disconnect가 오면 writer가 terminal write를 완료할지 abort로 전환할지 contract를 정한다. 이미 content가 committed됐으므로 client가 close했다면 write는 실패할 수 있다. engine state는 finish로, external state는 disconnected로 기록할 수 있다. 하나의 status 문자열로 합치지 않는다.

cleanup은 output queue, parser buffer, metrics/span, scheduler request, KV/cache handle을 각각 terminal로 만든다. 일부 cleanup 실패가 request result를 뒤집지는 않아도 residue alert와 retry cleanup이 필요하다.

**regression fixture가 실제 오류를 검출하는지 확인한다**

terminal double-emit mutation을 넣으면 exactly-one assertion이 실패해야 한다. usage 누적/증분 혼동 mutation은 billed conservation이 실패해야 한다. tool frame index swap은 reconstructed calls가 달라져야 한다. abort drop mutation은 orphan age/KV residue가 실패해야 한다.

positive test만 통과시키면 probe가 무감각할 수 있다. 알려진 잘못된 state transition을 가정해 negative expected를 작성한다. source unit/integration test가 어느 invariant를 실제 assertion하는지 읽는다. test 이름만 믿지 않는다.

**deployment checklist를 상태기계로 압축한다**

배포 전에는 schema revision, frame ordering, usage/logprobs/tool fixtures, before/after-commit errors, disconnect race와 slow-client backpressure를 확인한다. client SDK 대표 버전이 unknown frame/error를 어떻게 처리하는지도 compatibility matrix에 둔다.

canary에서 TTFF/ITL, frame queue age, generated/delivered/billed gap, terminal duplicates, cancel cleanup latency, orphan count를 generation별로 본다. 새 formatter가 payload를 키워 proxy buffer/timeout을 바꾸는지 확인한다.

rollback은 formatter/API process와 engine output contract generation을 맞춘다. in-flight stream은 old handler에서 drain하거나 명시적으로 terminal 처리한다. 중간에 handler를 바꿔 frame schema/order가 섞이지 않게 한다.

최종 handoff는 logical request, attempt, external commit/terminal, engine terminal, cleanup terminal과 usage ledger다. 20장은 이 상태를 받아 retry/fallback이 새 attempt를 만들 수 있는 시점과 duplicate billing/idempotency를 결정한다. external commit 이후의 fallback은 기본적으로 새 logical semantics가 필요하다.

## 19.4 finish·error·disconnect·usage를 서로 다른 terminal로 닫는다

stop token 선택 뒤 engine finish, detokenize·trim·tool parse, final chunk 생성, socket write와 resource free가 이어진다. 한 `finished_at`으로 뭉개지 않는다.

client disconnect도 GPU work를 즉시 멈추지 않는다. HTTP layer가 감지하고 request ID로 abort를 보내며 engine queue가 받고 scheduler가 request를 제거한다. 이미 enqueue된 step은 drain될 수 있고 KV는 safe point 뒤 반환된다.

vLLM stream path와 async engine abort는 [stream generator](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L422-L843)와 [`async_llm.py:200-420`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/async_llm.py#L200-L420)에서 잇는다.

disconnect count, abort 발행, scheduler removal, in-flight drain, KV free를 별도 계수한다. access log의 close는 자원 반환 증거가 아니다.

non-stream socket도 끊어진다. full result를 모으는 동안 abort하는지 background generation을 계속하는지 확인한다. stream path에만 disconnect check가 있는 비대칭이 있을 수 있다.

client, proxy, API server, engine deadline은 다른 owner다. retry 후 첫 attempt가 계속 실행되면 physical work가 중복된다. idempotency가 없다면 두 답이 함께 생성될 수 있다.

error를 validation, admission, execution, output processing, transport로 나눈다. validation은 engine request 없음, execution은 partial state cleanup, output parse는 model work 성공 뒤 API 실패, transport는 전달만 실패할 수 있다.

public finish reason과 internal cause를 같이 기록한다. `length`가 model cap, requested cap, server cap을 합칠 수 있다. abort와 error를 stop으로 숨기지 않는다.

SGLang abort는 [`tokenizer_manager.py:300-680`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L300-L680)에서 request state와 abort message를 보고 scheduler removal까지 잇는다.

llama.cpp는 task ID와 slot ID를 구분한다. slot 재사용 때 이전 callback/result가 섞이지 않도록 incarnation을 본다. [`server-task.cpp`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-task.cpp#L1-L420)와 context routing을 잇는다.

Transformers `generate`는 HTTP disconnect를 알지 못한다. wrapper가 stopping criterion이나 worker cancellation을 관리한다. caller thread 취소가 device work와 cache cleanup을 자동 보장하지 않는다.

clients가 100ms timeout 후 retry해 usage가 오르는 사건에서는 abort 미발행, queue delay, in-flight drain, duplicate attempts 가설을 request timeline으로 나눈다. 복구는 abort→remove→KV free latency와 cancelled tokens executed까지 본다.

### disconnect 사건: socket은 닫혔지만 decode는 계속된다

구체적인 시간표를 만들자. 0ms에 request를 받고 40ms에 first content를 보냈다. 100ms에 client timeout으로 socket이 닫혔다. API process는 130ms에 disconnect를 관찰했고 135ms에 abort message를 enqueue했다. engine은 180ms에 message를 받았지만 current decode step이 210ms에 끝난 뒤 scheduler에서 제거했다. KV blocks는 215ms에 free됐다.

이 요청의 cancel propagation은 115ms 걸렸고 100~210ms 사이 generated tokens는 delivered되지 않은 waste다. HTTP access log에는 100ms close만 보일 수 있고 engine metric에는 210ms finish만 보일 수 있다. 두 timestamp를 correlation ID로 연결하지 않으면 waste를 설명하지 못한다.

client timeout이 100ms인데 정상 TTFT p99가 120ms라면 retry storm은 server 내부 cancel bug가 없어도 생긴다. 첫 attempt가 abort되기 전에 retry attempt가 admission된다. logical task 하나가 physical requests 둘이 되고 cache와 scheduler capacity를 차지한다. timeout policy와 service SLO를 함께 본다.

API generator의 exception handling도 중요한 branch다. normal generator close, framework cancellation exception, explicit disconnect poll이 모두 cleanup으로 가는지 확인한다. `finally`에서 abort하는 경우 request가 정상 finish한 뒤 중복 abort를 보내지 않는지 idempotency가 필요하다.

abort message가 enqueue됐다는 사실은 scheduler가 제거했다는 뜻이 아니다. IPC queue가 밀릴 수 있고 request ID mapping이 틀릴 수 있으며 이미 finish한 ID일 수 있다. parent request와 `n` candidates가 child IDs를 가질 때 모두 제거해야 한다.

scheduler removal도 resource free와 같지 않다. in-flight runner output이 reference를 갖고 있거나 deferred block free가 step boundary에 실행될 수 있다. connector나 prefix cache refcount가 남을 수 있다. cancel latency를 API·engine·scheduler·KV 네 구간으로 나눈다.

non-stream path에서는 response aggregation coroutine이 disconnect를 poll하지 않을 수 있다. client가 사라져도 full generation을 끝내고 write에서 실패한다. 이것이 의도된 policy인지 구현 누락인지 문서와 source로 판정한다. stream과 full 두 paths를 fixture로 둔다.

slow consumer와 disconnect도 구분한다. socket은 살아 있지만 client read가 느리면 backpressure가 걸린다. abort해서는 안 되지만 buffer limit과 timeout policy가 필요하다. queue pressure 때문에 engine output drain이 늦는지, transport buffer만 늘어나는지 본다.

복구 후에는 cancellation flood에서 normal requests가 starvation되지 않는지도 본다. abort 처리 queue가 normal engine inputs와 같은 channel을 쓰면 flood 자체가 병목이 될 수 있다. cancelled request blocks가 즉시 재사용되는지보다 안전한 free ordering이 먼저다.

source-only 단계에서는 event hooks와 state transitions, idempotent cleanup 구조를 증명한다. 실제 milliseconds와 waste tokens는 실행 없이는 주장하지 않는다. 위 시간표는 관찰 schema를 설명하는 예시이며 측정 결과가 아니다.

## 19.5 first divergence와 exactly-once 경계로 호환성 사건을 푼다

tools를 보냈는데 한 server만 tool call을 반환한다고 하자. raw schema acceptance는 같다. rendered prompt를 비교해 A는 definitions와 tool marker를 넣고 B는 plain template를 썼다면 first divergence는 model이 아니라 compiler다.

경쟁 가설은 template, parser, grammar다. rendered IDs가 다르면 template가 먼저다. prompt와 raw text가 같고 response object만 다르면 parser다. prompt가 같지만 grammar가 한쪽에만 있다면 constraint가 raw token path를 바꾼다.

visible text는 같은데 usage가 120과 124라면 BOS/EOS, generation marker, tokenizer revision을 비교한다. completion usage는 stop token과 trimmed tokens 포함 여부를 본다. cached detail과 total을 혼동하지 않는다.

stream content를 다 받았는데 client가 무한 대기하면 engine finish, generator yield, write, proxy flush, client receive를 나눈다. empty terminal delta를 server나 client가 버렸을 수 있다.

disconnect 뒤 ghost generation은 route→generator→engine abort→scheduler remove→KV free로 걷는다. API request ID와 engine ID mapping, parent choice와 child requests 전파를 보존한다.

logprobs shape는 같고 값이 다르면 prompt IDs와 logits를 먼저 비교한다. penalties·temperature 전후 어느 distribution의 logprob인지, top alternatives가 어느 stage인지 source로 고정한다.

checkpoint 순서는 raw request, typed/defaulted request, rendered bytes, IDs, engine params, raw token outputs, parsed output, chunks, terminal, cleanup이다. 처음 다른 경계가 owner다.

### stop 사건: 같은 문장에서 서로 다른 finish reason이 나온다

두 server가 화면에는 똑같이 `완료했습니다.`를 보여 주지만 하나는 `finish_reason=stop`, 다른 하나는 `length`를 반환한다고 하자. visible text parity만 보면 성공이지만 agent orchestration은 length를 불완전 응답으로 보고 retry할 수 있다. terminal cause는 application behavior를 바꾼다.

먼저 요청의 stop strings, tokenizer EOS와 model generation config, output cap을 적는다. raw sampled IDs에서 EOS가 선택됐는지, stop string이 decoded suffix에서 발견됐는지, cap에 도달했는지 시간 순서로 본다. 같은 step에서 stop string과 cap이 동시에 만족되면 precedence가 public reason을 정한다.

예를 들어 output cap이 10이고 열 번째 token을 decode한 text가 stop string을 완성했다고 하자. engine이 token count cap을 먼저 검사하면 length, detokenizer가 stop match를 먼저 commit하면 stop이 될 수 있다. 어느 reason이 contract인지 source의 stopping/order와 response mapping을 이어야 한다.

EOS는 token ID 수준이고 stop string은 decoded byte/text 수준이다. special token skipping 때문에 EOS가 visible text에 없을 수 있다. stop string은 여러 IDs를 가로지르고 Unicode byte boundary를 가로지를 수 있다. 하나의 stop list를 token IDs로만 바꾸면 모든 문자열을 정확히 표현할 수 없다.

stop text 포함 policy도 user-visible parity를 바꾼다. server A는 matched `END`를 제거하고 B는 포함한다. streaming에서 A가 `EN`을 이미 commit했다면 완전한 제거가 불가능하다. matcher pending buffer와 maximum stop-prefix length를 본다. 여러 stop strings가 prefix 관계일 때 longest/first match rule도 필요하다.

additional stop token IDs가 template special markers와 겹치면 generation이 즉시 끝날 수 있다. tokenizer revision이 달라 같은 textual marker가 다른 ID가 되면 behavior가 갈린다. request conversion이 stop IDs를 model defaults와 merge하는지 replace하는지 확인한다.

`ignore_eos` 같은 extension은 public stop semantic을 크게 바꾼다. true여도 max cap과 stop strings는 남을 수 있다. API compatibility test가 EOS만 보고 무한 generation이라 판단하지 않도록 effective stopping state를 기록한다.

first divergence는 raw sampled IDs, detokenized pending buffer, matched stop identity, engine internal finish enum, public reason, trimmed text다. raw IDs부터 다르면 sampling/model, IDs는 같고 match가 다르면 detokenizer/stop automaton, internal enum은 같고 public reason만 다르면 response mapping이다.

negative fixture는 single-token EOS, multi-token string, overlapping strings `END`와 `END!`, Unicode boundary, exact cap과 stop simultaneous, stream fragment boundary를 포함한다. user-visible text뿐 아니라 finish reason, usage count와 raw completion IDs를 비교한다.

### usage 사건: token 수는 어느 상태를 세는가

usage는 단순 `len(encode(text))`가 아니다. prompt count는 rendered chat IDs를 세야 하고 multimodal placeholders나 virtual tokens를 어떻게 포함하는지 policy가 있다. completion count는 sampled IDs, accepted IDs, visible emitted IDs 중 어느 상태인지 정해야 한다.

stop string이 trim되면 sampled completion IDs 수가 visible text 재tokenization보다 많을 수 있다. speculative decoding에서는 draft sampled tokens 중 rejected tokens는 compute work였지만 final completion usage에 보통 포함되지 않을 수 있다. prefix cache hit는 prompt semantic tokens를 줄이지 않지만 computed prompt tokens는 줄인다.

따라서 service accounting에는 semantic usage와 physical work를 나눈다. API `prompt_tokens`, `completion_tokens`, `total_tokens`는 semantic request accounting이고, scheduler metrics의 executed/rejected/cached tokens는 performance accounting이다. 두 값을 같다고 요구하면 cache와 speculation을 오류로 오인한다.

chat template가 다르면 prompt usage도 다르다. server A가 BOS와 assistant marker 두 IDs를 더 넣으면 같은 messages가 두 tokens 길어질 수 있다. independent count tool도 A의 exact template와 tokenizer를 써야 oracle이다. raw content를 generic tokenizer로 encode한 수는 비교 기준이 아니다.

stream usage 전달과 engine accounting도 분리한다. engine이 terminal usage를 계산했어도 disconnect 전에 usage chunk가 client에 전달되지 않을 수 있다. delivered usage absence를 zero work로 저장하면 비용 reconciliation이 틀린다. server metric과 response receipt를 request ID로 연결한다.

`n=2`에서 prompt tokens를 한 번 세고 completion을 choices 합으로 셀지, choice별 usage를 어떻게 노출할지 확인한다. best-of가 hidden candidates를 계산하면 public usage에 모든 sampled candidate work가 포함되는지 별도 계약이다. API 표면만 보고 physical cost를 역산하지 않는다.

cached token details가 있다면 total prompt tokens의 subset인지 별도 추가량인지 명확히 한다. reasoning tokens, audio tokens 같은 detail도 total과 관계를 source schema에서 확인한다. unsupported details를 0으로 보내는 것과 field를 생략하는 것은 client에게 다를 수 있다.

usage differential은 rendered IDs count, accepted output IDs, trimmed IDs, cache hit count, speculative accepted/rejected, serialized usage를 나란히 둔다. first divergence가 semantic IDs인지 accounting policy인지 serializer인지 구분한다.

비용 incident에서는 provider invoice나 quota와 local server usage를 직접 동일시하지 않는다. contract version과 detail fields를 고정하고, rounding·batching·cached discounts 같은 billing policy가 API token counts 밖에 있을 수 있음을 명시한다.

## 19.6 하나의 request record로 source·구현 비교·배포 판정을 합친다

**구현은 행이 아니라 같은 책임을 담은 열로 비교한다.**

한 구현씩 길게 읽고 기억으로 맞추지 않는다. 동일한 request record의 stage를 행에 두고 구현을 열에 둔다. 빈칸은 동일하다는 뜻이 아니라 `unsupported`, `wrapper-owned`, `not observed` 중 하나로 명시한다.

Disconnect row는 transport 신호에서 시작해 안쪽으로 읽는다. 먼저 client가 끊긴 시각과 마지막으로 전달된 chunk를 고정하고, server의 cancel observer가 같은 request ID로 abort를 만들었는지 확인한다. 이어 engine 또는 task queue가 그 ID의 future dispatch를 막았는지, 이미 실행 중인 work가 terminal state로 전환됐는지, scheduler slot과 KV/cache ref가 반환됐는지를 본다. 표의 한 셀에 `abort 지원`이라고만 쓰지 말고 `observer→abort message→engine/task terminal→resource free` 가운데 확인한 마지막 경계를 적는다.

| request stage | vLLM | SGLang | llama.cpp | Transformers |
|---|---|---|---|---|
| raw JSON·validation | OpenAI route와 typed protocol | OpenAI protocol/serving | server JSON parser | HTTP wrapper 소유 |
| normalized request | serving conversion | serving→tokenizer manager DTO | server task params | generation config와 wrapper 변환 |
| render·tokenize | chat serving renderer | serving render/encode와 manager | GGUF/Jinja·legacy chat path | tokenizer/processor template API |
| engine submission | AsyncLLM request | tokenizer manager→scheduler IPC | task queue→slot | `generate()` 호출 |
| frame assembly | streaming response generator | OpenAI response assembler | partial-result formatter | streamer; SSE는 wrapper 소유 |
| disconnect·terminal | abort와 engine cleanup | abort message와 scheduler cleanup | task/slot cancellation | wrapper cancel과 generation cleanup |

이 표의 셀은 소스 카드의 축약어다. 지원 여부를 판정할 때는 해당 셀의 parser, mutation, consumer와 terminal을 같은 열에서 이어야 한다. 예를 들어 Transformers의 streamer가 있다고 해서 SSE·disconnect 계약까지 library가 소유한다고 쓰지 않는다.

source walk는 route literal에서 시작해 request model, validators, rendering, engine generate, output generator, abort를 잇는다. helper 목록 대신 같은 request identity를 따라간다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- vLLM 지도는 [route](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/api_router.py#L53-L68), [render](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L180-L218), [conversion](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L219-L388), [stream](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L422-L843), [full](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L844-L1010), [engine](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/async_llm.py#L200-L420)다.

- SGLang은 [protocol](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/protocol.py#L1-L300), [chat](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_chat.py#L1-L700), [tokenize](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_tokenize.py#L20-L154), [manager](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L1-L680)을 잇는다.

llama.cpp는 endpoint literal에서 JSON parser, [`llama-chat.cpp`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-chat.cpp#L1-L320), [`server-task.cpp`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-task.cpp#L1-L420), [`server-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1-L1040)을 잇는다.

- Transformers는 [template](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L1540-L1840), [config](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/configuration_utils.py#L100-L360), [generation](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2100-L2500), [streamer](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/streamers.py#L160-L320)을 잇는다.
- HTTP lifecycle은 wrapper owner다.

fixture는 plain message, system+user, stop boundary, forced tool, `n=2` stream, first content 뒤 disconnect를 쓴다. request hash, template hash, rendered IDs, params, raw outputs, finish, chunks, usage와 abort/free events를 기록한다.

호환 판정은 transport/schema, prompt semantic, engine params, output state machine, lifecycle로 나눈다. 한 층의 pass를 전체 호환으로 확대하지 않는다. product가 약속할 층을 공개한다.

**네 stack은 같은 책임을 서로 다른 위치에 둔다**

vLLM은 OpenAI route, typed protocol, serving class, async engine client가 비교적 명시적으로 분리되어 있다. chat serving이 template rendering과 sampling conversion, stream response를 소유하고 engine client가 request lifecycle을 core로 전달한다. source walk에서 route만 읽고 호환을 판정하면 핵심 변환을 놓친다.

SGLang은 OpenAI entrypoint와 tokenizer manager, scheduler process 사이 IPC가 중요하다. API serving에서 만든 internal request가 tokenizer manager의 async state와 scheduler message가 된다. stream result가 다시 manager를 거쳐 response delta가 된다. process boundary 때문에 request identity와 abort message를 이어야 한다.

llama.cpp server는 C++ server context, task queue와 slot lifecycle이 중심이다. JSON parse와 chat format, sampling params, slot assignment, partial result formatting이 Python serving class처럼 한 모듈 계층에 보이지 않을 수 있다. endpoint literal에서 task ID와 slot ID를 따라가는 것이 유용하다.

Transformers는 generation library이므로 HTTP route, OpenAI error object, SSE와 disconnect를 완성하지 않는다. tokenizer/template, generation config, processors·criteria, streamer는 제공하지만 API wrapper가 schema와 lifecycle을 추가한다. Transformers-backed server를 비교할 때 wrapper source를 빼고 library만 보는 것은 계약 절반을 잃는 일이다.

이 차이는 우열표가 아니다. owner가 다르면 first divergence를 찾는 위치가 달라진다. vLLM에서 response assembler bug를 scheduler source에서 찾지 않고, Transformers streamer의 queue behavior를 OpenAI protocol이라고 부르지 않는다. llama.cpp slot cleanup을 Pydantic validator 관점으로 설명하지 않는다.

네 stack 공통 좌표는 만들 수 있다. raw body owner, typed/effective request owner, render/tokenize owner, engine task owner, raw output owner, response assembler, transport cancel observer, scheduler/resource cleanup owner다. 함수 이름은 달라도 이 여덟 의미 경계를 채운다.

source card에는 함수 signature뿐 아니라 소비자와 state mutation을 적는다. validator는 어떤 field를 바꾸고 누가 effective 값을 읽는가. renderer는 bytes/IDs와 template identity를 누구에게 넘기는가. generator는 request ID와 finish state를 언제 commit하는가. abort는 어느 queue에 어떤 ID를 넣는가.

async generator lifetime을 특히 조심한다. Python generator close가 `finally` cleanup을 실행할 수 있지만 framework가 언제 close하는지, exception이 어떤 type인지 확인한다. C++ callback과 task queue는 다른 cleanup pattern을 쓴다. 언어가 다르다고 lifecycle 의미가 달라져야 하는 것은 아니다.

**capability matrix는 실제 fixture에서 나온다**

문서에 supported라고 적힌 field 목록만 옮기지 않는다. capability 하나마다 positive fixture, boundary fixture, unsupported fixture와 source consumer를 둔다. tool calling이면 template/parser/constraint, logprobs면 engine output/serializer, streaming usage면 assembler/client parsing을 함께 본다.

plain chat fixture는 system·user messages와 deterministic output cap을 쓴다. 목적은 answer quality가 아니라 rendered IDs와 terminal response shape를 고정하는 것이다. chat template가 없는 model에서 server가 error인지 fallback인지 기록한다.

sampling fixture는 temperature, top-p, seed와 logit bias를 한 번에 켜지 않는다. 하나씩 effective params를 확인한 뒤 pairwise order가 중요한 조합을 추가한다. 실행 없는 source 감사에서는 conversion과 processor order를 증명하고 output은 미검증으로 남긴다.

tool fixture는 auto와 forced, no-tools, parallel calls를 나눈다. schema acceptance, prompt inclusion, parser selection, finish mapping을 각각 pass/fail로 둔다. tool call text가 보였다는 사실만으로 structured support pass를 주지 않는다.

logprobs fixture는 selected token, top alternatives, prompt first token, Unicode bytes와 stream을 포함한다. response key 존재가 아니라 어느 score stage를 serialize하는지 source로 설명해야 한다.

stop fixture는 token ID, multi-token string, overlapping strings, exact cap collision을 다룬다. visible text, raw IDs, reason, usage가 함께 맞아야 한다. include/exclude policy가 다르면 documented difference로 기록한다.

stream fixture는 role-only first event, content, tool fragments, usage-only, staggered multi-choice finish와 partial error다. state-machine validity를 검사한다. chunk 크기나 event grouping이 달라도 append 결과와 terminal order가 같으면 호환 가능한 범위일 수 있다.

disconnect fixture는 headers 전, first chunk 전, partial content 뒤, terminal 직전과 non-stream wait 중 disconnect를 나눈다. abort event와 resource cleanup source path를 확인한다. 실제 timing은 runtime 승인 뒤 측정한다.

capability matrix의 cell에는 supported/unsupported만 넣지 않는다. accepted, normalized, executed, serialized, lifecycle-safe의 다섯 상태를 둔다. field가 accepted됐지만 ignored인 상태가 드러난다. tool request가 rendered됐지만 parser가 없는 상태도 드러난다.

**schema version과 extension을 다루는 태도**

OpenAI API 표면은 시간이 지나며 새 fields와 object variants가 생길 수 있다. 호환 server가 어느 schema snapshot을 목표로 하는지 고정해야 한다. 최신 SDK가 보내는 새 field를 오래된 server가 reject할 수 있다. silently ignore하면 기능이 적용되지 않은 채 성공할 수 있다.

server-specific extensions는 유용하지만 namespace와 discovery가 필요하다. priority, guided decoding, adapter, cache control을 standard field처럼 보이게 만들면 client migration이 어려워진다. extension이 engine state를 어떻게 바꾸는지 문서와 effective response metadata로 확인한다.

client SDK도 변환 layer다. deprecated field를 새 field로 바꾸거나 null을 생략하고, response object validation을 수행할 수 있다. wire capture와 application object를 구분한다. server에 도착한 body가 source-level client call과 같다고 가정하지 않는다.

gateway가 provider-neutral schema를 OpenAI schema로 바꾸는 경우 conversion이 하나 더 생긴다. model alias, tools, stop, usage와 error mapping을 gateway source에서도 감사한다. 이 장의 다섯 계약이 hop마다 반복된다.

version upgrade audit는 새 field 목록만 비교하지 않는다. defaults 변경, enum 추가, finish mapping, stream usage order, validation strictness를 본다. 동일 fixture를 old/new source conversion에 적용해 effective request와 event sequence diff를 만든다.

compatibility shim은 차이를 숨길 수도 있고 명시적으로 닫을 수도 있다. field rename을 무손실 변환하면 유용하다. unsupported option을 drop하고 성공시키면 위험하다. shim은 변환 전후 state와 warning을 기록해야 한다.

**reader workbook: 한 request를 종이에 완주한다**

synthetic request를 하나 정한다. system은 짧은 instruction, user는 weather 질문, tool 하나, forced choice, stream true, top logprobs 3, max output 32, stop string 하나다. 실제 content보다 control fields가 중요하다.

첫 칸에는 raw wire body와 headers, path, schema version을 적는다. 둘째에는 typed/defaulted fields와 resolved model을 적는다. 셋째에는 effective template hash, rendered bytes length, token IDs와 prompt count를 적는다.

넷째에는 engine params를 적는다. output cap, greedy/sampling mode, stop IDs와 strings, tool grammar/parser, logprob count, candidate count, RNG, priority다. 다섯째에는 raw output event schema를 적는다. token IDs, logprobs, finish enum, usage counters다.

여섯째에는 stream state를 적는다. role event, tool call index/ID, argument fragments, finish reason, usage, sentinel이다. 일곱째에는 disconnect 시 abort ID와 scheduler/cache cleanup owner를 적는다.

각 칸 옆에는 고정 source link와 아직 runtime에서만 확인 가능한 field를 구분한다. source가 option을 consumer에 전달한다고 증명해도 실제 model output을 주장하지 않는다. runtime result가 있어도 revision link 없이 일반화하지 않는다.

이 request를 four stacks에 나란히 놓으면 differences가 드러난다. template와 tokenizer가 달라 IDs부터 다를 수 있고, tool parser support가 다를 수 있으며, stream usage order와 disconnect policy가 다를 수 있다. 모두 HTTP-compatible이면서 semantic parity는 아닐 수 있다.

workbook의 목적은 모든 stack을 강제로 동일하게 만드는 것이 아니다. application이 요구하는 contract를 고르고 차이를 조기에 발견하는 것이다. tool agent라면 structured tool lifecycle이 필수이고 plain text batch job이라면 transport/schema와 deterministic prompt가 더 중요할 수 있다.

**migration 사건: SDK는 그대로인데 품질과 비용이 함께 변했다**

한 서비스를 provider API에서 self-hosted vLLM으로 옮겼다고 하자. client code와 request JSON은 바꾸지 않았다. 배포 첫날 답변이 장황해지고 prompt usage가 평균 여섯 tokens 늘었으며 일부 stop strings가 response 끝에 보였다. 팀은 model quantization을 원인으로 지목했다.

하지만 first divergence audit는 더 앞에서 시작한다. provider의 exact model/tokenizer/template revision을 알 수 있는 범위와 self-hosted artifact를 기록한다. synthetic conversation의 rendered prompt를 양쪽에서 직접 얻을 수 없다면 provider usage와 known template를 간접 증거로 두고, self-hosted path는 rendered bytes와 IDs를 source/runtime에서 고정한다. 모르는 provider 내부를 같다고 가정하지 않는다.

self-hosted template가 assistant generation marker 뒤 newline을 하나 더 넣고 BOS를 명시적으로 추가했다고 하자. prompt IDs가 여섯 늘어난 usage와 맞는다. model은 다른 prompt 조건을 받았으므로 장황함을 quantization으로 단정할 수 없다. 먼저 template를 intended artifact contract에 맞춘 뒤 selected logits와 output behavior를 재평가한다.

stop text 노출은 또 다른 owner다. provider는 matched string을 제외했고 self-hosted conversion은 include policy를 켰거나 stream matcher가 prefix를 너무 일찍 commit했을 수 있다. prompt 차이와 stop serialization 차이는 같은 migration에서 동시에 존재해도 별도 fixes를 요구한다.

비용도 API usage만 보지 않는다. prompt semantic tokens가 늘었고 prefix cache hit policy와 batch scheduler가 바뀌었다. public usage 증가 여섯 tokens와 GPU work 증가가 정확히 같지 않을 수 있다. cached prompt tokens, executed tokens와 request latency를 별도 본다.

migration 검증은 client unit test보다 넓다. request/response schema fixture, rendered IDs golden, engine params snapshot, stream event state machine, finish/usage, disconnect cleanup을 둔다. model quality set은 이 계약들이 맞은 뒤 비교한다. 그래야 infra 변화와 prompt 변화가 평가를 오염하지 않는다.

rollback 기준도 층별이다. schema breaking이면 traffic을 즉시 되돌릴 수 있다. prompt semantic difference가 intentional이라면 model evaluation이 필요하다. disconnect cleanup leak은 capacity risk라 rollout을 멈춘다. minor chunk grouping 차이는 client state machine이 허용하면 blocker가 아닐 수 있다.

이 사건은 같은 SDK 호출이 end-to-end 동등성을 보장하지 않는다는 핵심을 보여 준다. migration checklist가 field 목록에 머무르면 template, stop automaton과 cleanup을 놓친다. request가 상태를 바꾸며 지나가는 전체 길을 golden artifact로 둬야 한다.

**retry 사건: error mapping이 중복 side effect를 만든다**

tool agent가 stream 중 network error를 만나 자동 retry했다고 하자. 첫 attempt는 이미 tool call arguments를 모두 전달했지만 terminal sentinel 전에 연결이 끊겼다. client는 실패로 판단해 같은 messages를 다시 보냈고 두 번째 attempt도 같은 tool call을 만들었다. application이 두 calls를 모두 실행하면 side effect가 중복된다.

OpenAI-compatible stream shape는 exactly-once tool execution을 보장하지 않는다. server request ID가 달라졌고 resume token이나 idempotency key가 없다면 retry는 새 generation이다. deterministic seed가 있어도 batch context와 parser chunking이 달라질 수 있다.

client는 partial stream state를 보존해야 한다. choice와 tool call ID, arguments fragments, finish 수신 여부를 기록한다. terminal 이전 tool call을 실행해도 되는지 application policy를 정한다. 보통 완전한 structured call과 authorization을 확인하고 application idempotency key로 side effect를 보호한다.

server error mapping도 retry behavior를 바꾼다. validation 4xx는 retry하지 않아야 하고 transient 5xx는 가능할 수 있다. partial output 뒤 transport close에는 HTTP status를 바꾸기 어려우므로 client가 event sequence를 해석해야 한다. error object type만으로 모든 경우를 포괄할 수 없다.

첫 attempt가 server에서 계속 실행되는 ghost request라면 두 engine attempts가 동시에 tool text를 생성한다. abort propagation과 application side-effect idempotency는 별도 방어다. server cancel을 고쳐도 client가 이미 받은 first tool call을 두 번 실행하는 문제는 남는다.

fixture는 partial content 뒤 close, partial tool name, complete arguments but no finish, finish received but no sentinel, usage 뒤 close를 포함한다. client가 어느 state를 success·retry·manual review로 분류하는지 검증한다. server는 terminal exactly-once와 abort cleanup을 검증한다.

correlation에는 logical operation ID, HTTP attempt ID, server request ID, engine child IDs, tool call ID를 둔다. 하나의 ID로 모두 대체하면 retry와 choices를 구분하지 못한다. 개인정보 arguments 대신 hash와 schema validation status를 남긴다.

복구 종료 조건은 단순 error rate 감소가 아니다. retry policy가 error class와 partial commit state를 반영하고, duplicate attempts가 application idempotency로 side effect를 한 번만 만들며, abandoned engine work가 cleanup되어야 한다. usage accounting도 logical task와 physical attempts를 나눈다.

**source claim과 실제 행동을 섞지 않는 법**

고정 source에서 route가 field를 받는 것은 schema path 존재를 증명한다. validator가 field를 sampling params로 바꾸면 effective conversion을 증명한다. backend output code가 logprobs를 serialize하면 구현 branch 존재를 증명한다. 이 세 사실만으로 모든 model/backend combination에서 field가 작동한다고 단정하지 않는다.

capability는 build flags, optional parser, tokenizer template와 launch config에 의존할 수 있다. source의 conditional branch와 deployment configuration을 함께 봐야 한다. runtime을 실행하지 않는 이 장은 condition과 관찰 방법을 제시하고 실측 result는 만들지 않는다.

문서가 지원을 약속하고 source가 reject한다면 version mismatch나 defect 후보다. source가 extension을 제공하지만 문서에 없으면 unstable/internal contract일 수 있다. public compatibility claim은 문서와 release contract를 포함해야 한다.

고정 line link는 독자가 주장 바로 옆에서 구현을 확인하게 한다. route link로 abort cleanup을 증명하지 않고, template link로 tool parser를 증명하지 않는다. 각 state transition에 가장 가까운 owner link를 둔다.

source file이 revision에서 이동하면 의미 좌표를 다시 찾는다. endpoint literal, request class, render call, engine generate, stream generator, abort method를 순서로 검색한다. 예전 line number를 최신 code에 억지로 적용하지 않는다.

실제 behavior report에는 source revision, build identity, server flags, resolved model/tokenizer/template, client/proxy versions와 fixture를 포함한다. 하나라도 없으면 다른 팀이 같은 결과를 재현하기 어렵다.

**독자가 가져갈 최종 판단법**

새 server를 만났을 때 supported options 표부터 외우지 않는다. 가장 작은 message 하나를 route에서 engine request까지 따라간다. raw body가 typed fields가 되고, template와 tokenizer가 IDs를 만들고, parameters가 sampler/stop/constraint state가 되는 과정을 적는다.

그다음 output 한 token을 반대 방향으로 따라온다. engine token과 logprob가 detokenizer와 parser를 지나 delta가 되고, finish와 usage가 붙고, transport가 terminal을 전달하는 과정을 적는다. disconnect를 한 지점에 넣어 abort와 resource free owner를 찾는다.

두 방향의 trace가 만나면 API contract가 닫힌다. input 쪽에서는 어떤 의미가 engine에 전달됐는지 알고 output 쪽에서는 어떤 engine state가 client에게 commit됐는지 안다. 가운데 scheduler와 model은 다음 장들의 owner다.

호환성을 주장할 때 scope를 문장에 넣는다. SDK와 schema가 호환되는가, rendered prompt와 generation semantics가 호환되는가, stream/lifecycle이 호환되는가를 말한다. 동일 model output을 요구한다면 artifacts와 numeric conditions를 추가한다.

문제가 생기면 가장 먼저 다른 checkpoint를 찾는다. body부터 다르면 gateway, typed fields부터면 validation, IDs부터면 template/tokenizer, params부터면 normalization, raw outputs부터면 engine/model, chunks부터면 parser/assembler, cleanup부터면 lifecycle owner다.

이 순서가 독자에게 주는 가장 큰 이득은 불필요한 GPU 조사를 줄이는 것이다. 많은 OpenAI compatibility 장애는 GPU를 호출하기 전 또는 output을 받은 뒤 생긴다. source 경계를 정확히 나누면 고가의 profiler보다 먼저 몇 줄의 conversion과 state machine에서 원인을 찾을 수 있다.

마지막으로 운영자가 실제 새벽 장애에서 이 장을 쓰는 장면을 그려 보자. 호출자는 응답이 가끔 비어 있다고 말하고, dashboard에는 HTTP 200과 GPU activity가 모두 보인다. 이 두 신호만 보면 model이 empty token을 냈다고 생각하기 쉽다. 그러나 200은 headers가 이미 나갔다는 뜻일 수 있고 GPU activity는 같은 batch의 다른 requests일 수 있다.

운영자는 failing request ID 하나를 고른다. raw body hash와 typed request가 존재하는지 확인한다. rendered prompt length가 0이 아니고 engine request가 admission됐는지 본다. raw engine output에 token ID가 있었는지, detokenizer가 visible suffix를 만들었는지, assembler가 content chunk를 yield했는지, transport가 client에 flush했는지 차례로 본다.

engine output이 없고 finish가 length라면 output cap normalization을 본다. raw token은 있는데 text가 비면 special-token skipping과 incomplete byte, stop trim을 본다. text는 있는데 chunk가 없으면 parser pending state와 assembler를 본다. chunk yield는 있는데 client가 못 받으면 proxy와 client parser를 본다. 같은 증상인 empty response가 다섯 owner로 갈라진다.

이때 큰 debug log를 무조건 켜지 않는다. request body와 output text는 민감할 수 있고 모든 token log는 latency를 바꾼다. 먼저 길이, hashes, state enums와 timestamps를 기록한다. synthetic fixture로 재현되면 승인된 환경에서 IDs와 chunks를 자세히 본다. 관찰 비용과 개인정보 경계를 함께 지킨다.

설정 변경도 원인 장부를 쓴다. chat template override를 바꾸면 rendered bytes와 IDs가 달라진다. include usage를 바꾸면 terminal stream shape가 달라진다. logprobs를 켜면 engine output와 serialization 비용이 달라진다. disconnect poll interval을 바꾸면 cancel propagation이 달라진다. 네 option을 한 번에 바꾸고 response가 정상이라고 결론 내리지 않는다.

각 변경에는 기대 state를 적는다. template 변경은 prompt hash가 바뀌되 schema와 transport는 같아야 한다. usage option은 token generation을 바꾸지 않고 terminal events만 바꿔야 한다. logprobs는 selected IDs가 deterministic fixture에서 같되 auxiliary output가 늘어야 한다. cancel 변경은 normal finish를 바꾸지 않고 abort latency만 줄여야 한다.

기대 밖 변화는 새로운 first divergence다. usage option을 켰는데 selected IDs가 바뀐다면 batch pressure나 output mode가 engine path를 바꿨는지 본다. logprobs 때문에 latency가 늘어난 것은 예상 가능하지만 prompt IDs가 달라졌다면 conversion 결함이다. template 변경 뒤 usage가 달라지는 것은 자연스럽지만 cached-prefix identity가 의도대로 invalidated됐는지 확인한다.

source 비교에서 네 stack을 똑같은 class diagram으로 만들 필요도 없다. vLLM의 serving class, SGLang의 manager IPC, llama.cpp의 task/slot, Transformers wrapper의 streamer는 구조가 다르다. 중요한 것은 typed request, prompt program, engine work, output commit, cleanup이라는 의미 상태가 모두 존재하고 request identity로 이어지는가다.

이 의미 상태가 닫히면 문서도 친절해진다. 독자에게 수십 option을 먼저 던지지 않고 한 요청을 따라가며 option이 등장하는 지점에서만 설명할 수 있다. `stop`은 matcher가 필요할 때, `logprobs`는 output state가 커질 때, `stream`은 commit protocol이 갈릴 때, usage는 terminal accounting에서 설명한다. field가 왜 존재하고 무엇을 바꾸는지가 자연스럽게 연결된다.

OpenAI 호환이라는 표지는 유용하다. 공통 client ecosystem과 익숙한 request shape를 제공한다. 다만 표지가 약속하지 않는 것까지 기대하지 않아야 한다. model의 token protocol, server의 engine semantics, stream과 cancellation은 구현과 deployment가 별도로 증명해야 한다.

그래서 이 장의 최종 문장은 간단하다. 호환 여부를 yes/no 하나로 기록하지 않는다. 어느 schema revision을 받고, 어느 prompt·generation semantics를 보존하며, 어떤 streaming·error·cancel lifecycle을 제공하는지 적는다. 그리고 그 문장을 고정 source와 fixture로 재현 가능하게 만든다. 그때 API compatibility는 marketing label이 아니라 운영 가능한 계약이 된다.

인계 문서도 같은 구조를 따른다. raw request sample만 첨부하지 않고 effective fields와 resolved artifact, rendered prompt hash, engine params, event sequence와 cleanup evidence를 묶는다. 수신자는 HTTP client부터 GPU까지 전부 다시 조사하지 않고 first divergence가 있는 owner에서 시작할 수 있다.

아직 확인하지 못한 사실은 빈칸으로 남긴다. source에 branch가 있어도 현재 build에서 선택됐는지 모르면 runtime 미검증이라고 쓴다. 외부 provider의 internal template를 알 수 없으면 동일하다고 추정하지 않는다. partial stream 뒤 client가 tool을 실행했는지 모르면 application evidence가 필요하다고 쓴다. 빈칸은 약점이 아니라 과장된 결론을 막는 경계다.

완료 판정에는 positive와 negative evidence가 모두 필요하다. expected field가 consumer까지 도달했고 unsupported 조합은 문서화한 error로 거절되며, terminal 뒤 추가 delta가 없고 disconnect request가 cleanup 경로로 들어가야 한다. 단지 test가 crash하지 않았다는 사실은 의미 보존을 증명하지 않는다.

독자가 이후 scheduler 장으로 내려갈 때 API terminology를 그대로 들고 가지 않는 이유도 여기에 있다. scheduler는 messages나 tool choice를 직접 보지 않고 normalized token budget, sequence state, priority와 output mode를 본다. API field가 어느 engine state가 됐는지 번역표가 있어야 option 변화가 batch·cache·latency에 미치는 효과를 설명할 수 있다.

반대로 scheduler event를 API success로 바로 올리지 않는다. engine이 finish했어도 stream terminal이 client에게 전달되지 않을 수 있고 output parser가 error를 낼 수 있다. 내부 완료와 외부 완료 사이의 마지막 구간을 이 장이 소유한다. 이 양방향 번역이 닫힐 때 요청 한 건의 수명주기가 비로소 끝난다.

최종 review에서는 synthetic request 하나를 소스 좌표만으로 처음부터 끝까지 다시 읽는다. 중간 state를 추측으로 메우지 않고 각 consumer를 확인한다. 빠진 owner가 없다면 독자는 같은 절차를 새 release와 다른 wrapper에도 적용할 수 있다. 그것이 이 장이 목표로 하는 실용적인 호환성 독법이다.

이 장을 빠져나갈 때 독자는 JSON이 어느 validator와 defaults를 지나고, messages가 어느 template/tokenizer로 IDs가 되며, stop·tools·logprobs가 어느 engine state가 되고, chunks가 어떤 commit을 나타내며, disconnect가 언제 scheduler와 KV cleanup으로 닫히는지 설명할 수 있어야 한다.

20~25장에서 gateway와 각 engine의 ingress 수명을 비교한 뒤 scheduler로 내려갈 때 가져갈 것은
normalized engine request와 lifecycle identity다. rendered IDs가 다르면 앞 장으로 돌아간다. raw
outputs는 같고 chunks만 다르면 이 장에 남는다. engine request가 같고 latency가 다르면 scheduler
owner로 넘어간다.

**호환성 통제 실험.** 동일 request ID로 non-stream과 stream을 실행해 최종 choice text, finish reason과 usage를 비교한다. 이어 client disconnect를 첫 frame 전과 후에 각각 주입하고 engine cancel, terminal frame 금지와 resource cleanup을 확인한다. 실험의 핵심은 HTTP 200 여부가 아니라 accepted request가 exactly one terminal 또는 명시적 disconnect terminal로 닫히는지다.

## 19.7 장말 source note와 deployment 판정을 같은 request record에 붙인다

source walk는 response model 정의에서 끝나지 않는다. route handler가 request schema를 validate하고 async generator를 만들며 disconnect/cancel을 감지하는 지점, serving layer가 engine outputs를 chat/completion chunks로 변환하는 지점, final usage/finish/error를 emit하는 지점, generator finalizer가 abort/cleanup을 호출하는 지점을 잇는다.

vLLM에서는 OpenAI serving route, chat/completion serving generator, output protocol, AsyncLLM generate/abort 경계를 연결한다. SGLang에서는 HTTP/OpenAI serving, TokenizerManager output loop와 abort message를 잇는다. Transformers streamers는 token→text buffering reference를 제공하지만 server lifecycle을 대신하지 않는다. llama.cpp server의 SSE chunk와 slot cancellation도 같은 의미 좌표로 비교한다.

**source card의 열**

각 frame kind에 `producer`, `input state`, `sequence/choice index`, `commit effect`, `terminal?`, `next consumer`, `cleanup dependency`를 둔다. usage producer가 engine output인지 local counter인지, logprobs가 raw token output과 어디서 결합하는지, tool delta parser가 어느 state를 보존하는지 적는다.

링크는 특정 frame 생성 branch를 증명하지만 실제 proxy buffering이나 client receipt를 증명하지 않는다. transport writer와 runtime trace가 필요하다. source에 abort call이 있다는 사실과 scheduler/KV cleanup 완료도 구분한다.

**option을 mutation chain으로 읽는다**

`stream=true`는 response type, generator lifetime, header commit 시점과 fallback 가능성을 바꾼다. `stream_options.include_usage`는 usage accumulator와 terminal frame shape를 바꾼다. `logprobs/top_logprobs`는 sampler output state, serialization payload와 bandwidth를 늘린다.

tools/tool_choice/parallel tool calls는 prompt compile, parser state, delta schema와 finish reason을 바꾼다. stop/max tokens는 engine terminal과 visible suffix/filter를 바꾼다. option 이름만 나열하지 않고 parser→engine state→output accumulator→frame→client effect를 잇는다.

**source revision과 protocol contract**

OpenAI-compatible shape는 revision마다 field/ordering/error behavior가 달라질 수 있다. pinned server revision과 API schema를 기록한다. 외부 OpenAI service의 undocumented 내부 동작을 local server와 동일하다고 주장하지 않는다. compatibility matrix에 supported/unsupported/conditional을 둔다.

**회귀·모니터링·종료 조건을 terminal 열로 붙인다.**

matrix는 non-stream/stream, content/tool, logprobs off/on, usage off/on, normal/error/disconnect/cancel, error-before-first-frame/after-commit, single/multiple choices를 포함한다. 전체 Cartesian product 대신 terminal/commit 경계를 가르는 cell을 고른다.

각 cell은 ordered frame kinds, per-choice/tool sequence, concatenated text, token/logprob alignment, usage semantics, exactly-one external terminal, engine/cleanup terminal과 orphan residue를 판정한다. HTTP status와 stream error event를 별 열로 둔다.

fleet metric에는 accepted/running/committed/terminal/cleanup counts, time-to-first-frame, inter-frame gap, cancel-to-engine-terminal, orphan age, frame serialization error와 bounded frame kind를 둔다. request ID/tool arguments/prompt를 label로 넣지 않는다. detailed timeline은 trace/incident store에 둔다.

performance는 TTFT와 ITL만 보지 않는다. logprobs/top-k와 tool JSON serialization이 CPU/output queue/backpressure를 늘릴 수 있다. usage terminal이 slow client에 막히면 engine은 끝났지만 request object가 오래 살아 있을 수 있다. engine time, serialization, socket write, proxy/client receive를 분리한다.

rollback은 stream formatter만 되돌리지 않는다. API schema, output parser, engine output contract와 client compatibility generation을 묶는다. partial committed streams를 새 formatter로 이어 쓰지 않도록 in-flight를 drain한다. known-good non-stream/fallback lane의 의미 지원 범위를 확인한다.

최종 terminal은 protocol, accounting, cancellation, resource 네 개다. protocol은 ordered frames와 exactly-one external terminal, accounting은 committed/generated/billed usage reconciliation, cancellation은 client intent가 engine abort로 도달, resource는 output queue와 KV/request state cleanup이다. 하나를 다른 것으로 대신하지 않는다.

이 장의 incident 문장은 구체적이어야 한다. “Content 12 token commit 뒤 parser error가 error와 finish 두 terminal을 emit해 client별 성공 판정이 갈렸고, usage는 generated 13을 billed했지만 committed count를 기록하지 않았다. error terminal을 단일 owner로 만들고 three-count ledger와 abort cleanup generation을 추가해 partial-error/disconnect race fixture를 통과했다.”

이제 streaming 설명은 dry한 chunk field 목록이 아니다. 어떤 frame이 어떤 server state를 mutate하고, 언제 외부 commit이 생기며, error/cancel 뒤 무엇이 남아서는 안 되는지 독자가 추적할 수 있다. 20장에서는 gateway retry/fallback이 이 commit boundary와 billing/idempotency를 어떻게 다뤄야 하는지 이어 간다.

**reader 감사와 deployment 판정도 같은 request record를 쓴다.**

첫 15분에는 public schema와 effective defaults를 고정한다. text/chat endpoint, stream/non-stream, usage/logprobs/tools/error fields를 표로 옮긴다. unsupported와 silently ignored를 구분할 fixture를 정한다. API 문서만 읽지 않고 request model validation과 route handler를 찾는다.

다음 15분에는 raw request가 template/tokenizer와 engine params로 변환되는 호출을 걷는다. stop, max tokens, logprobs, tool choice와 streaming flag가 어느 internal state를 바꾸는지 적는다. field가 parse되지만 consumer가 없으면 지원으로 표시하지 않는다.

세 번째 15분에는 engine output object와 output formatter 사이를 걷는다. token IDs/text delta, finish reason, logprobs, usage, tool parser state가 어디서 합쳐지는지 본다. per-request accumulator와 choice/tool index lifetime을 기록한다. formatter 함수 이름보다 입력/출력 mutation을 적는다.

네 번째 15분에는 transport writer와 generator finalizer를 읽는다. header commit, first frame, write exception, disconnect detection, abort call과 cleanup을 연결한다. engine finish와 socket terminal이 같은 함수에 있지 않을 수 있다. process/task 경계도 표시한다.

다섯 번째 15분에는 작은 정상 fixture를 종이에 실행한다. role, Korean content 두 delta, tool argument 세 delta, logprob 두 token, usage, finish를 ordered event로 만든다. client concat/parser의 expected state와 server cursor를 나란히 둔다. non-stream final response와 의미 parity를 확인한다.

여섯 번째 15분에는 negative fixture를 넣는다. first frame 전 validation/admission error, content commit 후 engine error, terminal write 중 disconnect, slow-client queue overflow, finish/abort race다. 각각 expected external/engine/cleanup terminal과 retryability를 적는다.

일곱 번째 15분에는 metric/trace coverage를 확인한다. accepted→commit→terminal→cleanup conservation을 복원할 수 있는지, generated/delivered/billed를 구분하는지, selected schema/formatter generation을 아는지 본다. prompt/tool arguments를 고 cardinality label로 노출하지 않는다.

마지막 15분에는 compatibility와 rollback 문장을 쓴다. 어떤 field/ordering/error를 지원하고 어떤 SDK/version에서 검증했는지, partial stream과 idempotency를 어떻게 다루는지, known-good fallback lane과 in-flight drain이 무엇인지 적는다. 빈 관측은 TODO probe로 남긴다.

**배열과 표를 읽기 쉽게 배치한다**

독자에게 처음부터 모든 frame field 표를 던지지 않는다. 먼저 정상 content stream 한 건으로 commit/terminal 직관을 만든다. 다음에 usage와 logprobs가 별 cursor임을 보여 주고, tool incremental JSON을 추가한다. 마지막에 error/cancel race로 상태기계를 확장한다.

reference table은 본문 뒤에 둔다. 열은 field name이 아니라 독자의 질문을 따른다. “언제 나타나는가”, “무슨 state를 바꾸는가”, “terminal인가”, “누가 소비하는가”, “실패 시 무엇이 남는가”다. dry schema inventory를 인과 표로 바꾼다.

source atlas도 호출 순서로 배치한다. schema/route→normalize/engine submit→output accumulator→formatter→transport/finalizer→abort/cleanup이다. repository별 파일 목록을 먼저 나열하지 않는다. 같은 의미 stage에 vLLM/SGLang/llama.cpp/Transformers reference를 대응한다.

**option 효과를 독자가 반증하게 한다**

`stream=false→true` 변경의 예측은 first byte가 빨라질 가능성뿐 아니라 external commit이 앞당겨져 safe fallback window가 줄고 generator/transport state가 오래 산다는 것이다. falsifier는 first frame 전/후 provider error와 cleanup timeline이다.

`include_usage=true`는 final accounting frame과 accumulator state/payload를 추가한다. falsifier는 empty-choice usage frame, error/disconnect에서 usage 유실/중복, client 누적 해석이다. latency 개선을 기대하는 option이 아니다.

`logprobs=5`는 sampler output/top alternatives와 serialization bytes를 늘린다. falsifier는 token/logprob/text cursor alignment와 payload/ITL A/B다. 최종 text가 같다고 logprobs contract가 맞는 것은 아니다.

tools는 prompt와 parser, incremental frame schema, finish reason을 바꾼다. falsifier는 interleaved two-call index, incomplete JSON cancel, schema validation error다. tool success rate만으로 frame correctness를 대체하지 않는다.

**server와 client 책임을 나눈다**

server는 ordered frames와 declared terminal, stable identifiers, cleanup을 제공한다. client는 frame parser와 per-index accumulator, unknown/error handling, cancellation intent를 소유한다. proxy는 buffering/timeout/retry로 observable semantics를 바꿀 수 있다. 한 층의 success를 end-to-end receipt로 확장하지 않는다.

SDK가 malformed sequence를 조용히 무시한다면 server test가 통과해도 다른 SDK에서 실패할 수 있다. raw frame fixture와 client matrix를 함께 보존한다. network capture는 민감 정보를 제거하고 synthetic fixture를 우선한다.

application tool side effect와 billing idempotency는 API server 혼자 exactly-once로 보장할 수 없다. stable logical/attempt IDs와 committed evidence를 제공하고 gateway/application ledger가 조정한다. 다음 장과의 ownership 경계다.

**source에서 의도를 읽는 질문**

왜 first chunk 전까지 error를 HTTP status로 돌려주는가. header가 아직 commit되지 않아 더 명확한 protocol을 쓸 수 있기 때문이다. 왜 after-commit error가 별 event인가. 이미 200/status와 content bytes가 나갔기 때문이다. 왜 abort가 generator finalizer에 있는가. 정상·exception·disconnect 모두에서 cleanup을 실행하기 위해서다.

왜 usage를 terminal에 모으는가. 최종 committed/generated counts가 정해지는 시점이기 때문이다. 중간 cumulative usage를 지원한다면 client semantics와 duplication 위험을 명시해야 한다. 왜 tool arguments를 delta로 보내는가. 긴 structured output의 latency/streaming을 허용하지만 incomplete state 책임이 생긴다.

이 “왜”는 source branch와 state lifetime에 연결돼야 한다. 단순 설계 미학이 아니다. commit 이전/이후 가능한 오류 표현, buffer 비용, backpressure와 cancellation이라는 제약에서 선택이 나온다.

**최종 품질 판정**

좋은 장은 독자가 OpenAI-compatible이라는 라벨만 기억하게 하지 않는다. raw SSE/frame을 보고 role/content/tool/logprob/usage/terminal을 분류하고, server state와 cursor를 복원하며, double terminal·usage 중복·orphan cleanup을 찾게 해야 한다.

또한 모든 edge case를 지원한다고 과장하지 않는다. pinned revision과 static source가 증명하는 범위, runtime 미검증, external provider/client unknown을 분리한다. 직접 인용은 핵심 predicate 일부로 제한하고 나머지는 정확한 한국어로 설명한다.

19장의 final artifact는 ordered frame fixture, state-transition table, three-count usage ledger, cancel/cleanup timeline, pinned producer-consumer map과 compatibility matrix다. 이 자료가 있으면 20장의 router가 언제 재시도할 수 있고 언제 이미 외부 commit 때문에 멈춰야 하는지 정확히 판단할 수 있다.

배포 전 마지막 계산은 queue와 payload 상한이다. 동시 streaming 1,000개, 요청당 buffered frame 32개, frame 평균 2KiB라면 payload만 약 64MiB다. Python/object/allocator overhead와 tool/logprobs 큰 frame은 별도다. slow client에서 limit에 도달했을 때 pause, spill, cancel, error 중 어떤 정책인지 source와 config를 확인한다.

logprobs frame이 평균 10KiB로 커지고 같은 32개를 보존하면 해당 request 하나가 320KiB다. 1,000개면 320MiB 상한이다. 실제 분포와 backpressure를 측정해야 하지만 option이 output memory와 CPU를 바꾸는 이유를 보여 준다. payload limit과 request limit을 분리한다.

tool arguments는 단일 JSON 문자열이 매우 길 수 있다. frame count 제한만 두고 cumulative bytes를 제한하지 않으면 parser buffer가 커진다. per-tool/per-request byte limit과 incomplete terminal 처리, error code를 명시한다. limit 초과 후 engine abort와 cleanup까지 이어져야 한다.

stream timeout도 한 숫자가 아니다. time-to-first-frame, idle inter-frame, total request, downstream write, client deadline이 있다. first frame 후 provider를 retry하기 어려운 반면 idle timeout은 partial content 뒤 발생한다. 어느 clock이 어떤 terminal/error와 abort를 만드는지 option consumer를 걷는다.

proxy buffering은 server가 frame을 즉시 write해도 client TTFT를 늦출 수 있다. server write timestamp, proxy flush, client receive를 분리한다. 서버 source만으로 end-to-end streaming을 증명하지 않는다. 콘텐츠 타입과 buffering-disable header, ingress 설정은 배포 artifact다.

HTTP/2나 connection pooling에서 한 stream cancel이 다른 request connection을 끊지 않는지 client/proxy contract를 본다. server request identity와 transport stream identity를 분리한다. connection-level error를 모든 engine requests abort로 확대하지 않게 scope를 확인한다.

shutdown/drain은 새 request admission을 막고 기존 streams를 terminal 또는 deadline까지 처리한다. deadline 초과 시 structured error를 보낼 수 있는지, connection close만 가능한지 commit state에 따라 다르다. engine abort와 resource cleanup을 drain terminal에 포함한다. process exit만으로 정상 종료를 선언하지 않는다.

worker crash recovery에서는 external committed streams의 exact continuation이 가능한지 솔직히 말한다. output cursor와 engine/cache state가 복제되지 않았다면 같은 connection에서 이어 쓰기 어렵다. gateway가 새 attempt를 시작하면 duplicate prefix와 tool side effect 위험이 있다. “자동 복구”라는 표현을 evidence 없이 쓰지 않는다.

multi-choice stream은 usage가 choice별인지 request aggregate인지 확인한다. 한 choice가 cancel/finish되고 다른 choice가 계속될 때 request terminal과 cleanup을 잘못 앞당기지 않는다. `n` 지원이 제한되면 명시적 reject가 silent coercion보다 낫다.

seed/determinism option도 retry semantics에 영향을 준다. 같은 seed라도 provider/backend/batch가 달라 exact continuation을 보장하지 않는다. partial stream 뒤 재시작해 같은 suffix가 나온다는 가정은 위험하다. idempotency는 random determinism이 아니라 logical ledger와 side-effect key로 다룬다.

보안 관점에서 frame trace에는 tool arguments, logprob alternatives와 content가 민감할 수 있다. 일반 metric에는 size/count/state만 남기고 raw synthetic fixture를 사용한다. 실제 incident payload는 암호화·접근 통제·짧은 보존 기간을 둔다. digest도 반복 입력을 드러낼 수 있어 tenant scope를 고려한다.

문서의 troubleshooting은 증상→첫 관측→분기 순서를 따른다. “화면이 중간에 끊김”이면 raw frames와 external terminal, server writer/engine terminal을 본다. “과금만 남음”이면 generated/delivered/billed ledger를 본다. “취소해도 GPU가 돎”이면 disconnect→abort→scheduler/KV cleanup timeline을 본다. 관련 없는 option 목록을 먼저 제시하지 않는다.

최종 독자 연습은 failing timeline을 한 문장으로 압축한다. “Frame seq4 content commit 뒤 socket idle timeout이 발생했으나 abort가 output formatter에서 멈춰 scheduler request가 18초 더 running했고 generated 40/delivered 12/billed 40으로 갈렸다.” 이 문장은 owner, 낭비와 accounting gap을 동시에 보여 준다.

수정 뒤에는 timeout 자체가 사라졌는지가 아니라 abort propagation과 queue/KV residue가 닫혔는지, partial-stream error contract와 billing reconciliation이 맞는지 본다. 네 terminal을 다시 판정하면 19장의 설명이 실제 운영 행동으로 이어진다.

release diff에서도 같은 state machine을 사용한다. response schema에 새 field가 추가됐는지보다 formatter가 emit 순서를 바꾸는지, usage/finish terminal ownership이 이동하는지, generator exception/finalizer가 abort를 놓치는지 본다. SDK가 unknown field를 허용해도 ordering과 terminal 변화는 호환성 문제일 수 있다.

새 reasoning/content channel이 도입되면 cursor를 별도로 둔다. hidden/reasoning text가 visible content와 어떤 frame/usage 정책을 가지는지, parser가 channel transition을 어떻게 표현하는지 확인한다. 한 문자열로 합치면 stop, billing과 redaction 경계가 흐려질 수 있다.

audio/image 같은 multimodal output도 frame payload와 final artifact lifecycle이 다르다. binary/reference chunk, checksum, ordering, cancellation cleanup을 별 schema로 정의한다. text delta의 concat 규칙을 그대로 적용하지 않는다. 지원하지 않는 modality는 capability gate에서 거절한다.

server-side content filtering이 stream 중간에 개입하면 already committed content와 final moderation reason을 구분한다. 필터가 이전 frame을 회수할 수 없으므로 early buffering과 latency tradeoff가 생긴다. 정책을 숨기지 않고 external commit 지점과 visible partial risk를 문서화한다.

관측 샘플링도 terminal event는 잃지 않게 설계한다. 모든 content frame을 저장하지 않아도 accepted, first commit, error/finish, disconnect, engine terminal, cleanup은 보존한다. frame count/bytes와 cursor range로 중간을 요약한다. incident 때 필요한 chronology를 복원할 수 있어야 한다.

테스트 fixture에는 proxy/client consumer도 최소 하나 포함한다. raw server frames가 맞아도 SDK가 usage-only frame을 종료로 오해하거나 tool index를 잘못 합칠 수 있다. server contract와 client interpretation의 first divergence를 나눈다.

마지막으로 API compatibility score 하나를 만들지 않는다. schema, prompt semantics, generation options, stream frame, error/cancel, accounting을 별 axis로 보고한다. 어떤 축이 partial인지 알아야 사용자가 안전한 조합과 추가 검증 지점을 선택할 수 있다.

19장의 완료 판정은 독자가 endpoint 호출 예제를 복사하는 데 있지 않다. frame 하나를 보고 server state mutation과 consumer를 설명하고, partial commit 이후 retry 위험과 cancel cleanup을 찾아내며, source revision과 fixture로 그 설명을 반증할 수 있어야 한다.

최종 source walk에서 확인하지 못한 proxy·SDK·external provider 동작은 local server 사실과 분리한다. 필요한 network/client trace와 책임자를 적고, 호환이라고 추정하지 않는다. 반대로 local code가 명시적으로 reject하거나 ignore하는 option은 문서의 실제 지원표에 반영한다.

승인 회의에서는 정상 stream보다 실패 stream을 먼저 한 건 읽는다. ordered frame, external commit, engine terminal, cleanup과 three-count usage가 모두 설명되면 정상 경로는 그 부분집합으로 이해할 수 있다. 실패 경로를 설명하지 못하는 API 문서는 운영 계약으로 부족하다.

이 장의 artifact는 다음 장에서 attempt 생성 여부를 결정하는 입력이 된다. `committed=false`인 retryable error와 `committed=true` partial stream을 같은 fallback rule에 넣지 않는다. usage/tool side-effect evidence도 함께 넘긴다.

마지막 미확인 칸에는 owner와 확인 기한을 붙인다. 특히 proxy flush, client receipt와 external provider billing은 local source만으로 확정할 수 없다. 필요한 trace/statement가 없으면 unknown으로 남긴다.

이 정직한 경계가 있어야 다음 장의 router가 안전한 retry를 과장하지 않는다. 서버가 증명한 commit·terminal·usage까지만 입력으로 사용하고, application side effect와 end-to-end receipt는 별 ledger에서 조정한다.

장 종료 시 동일 failure timeline을 source revision 변경 뒤 다시 재생할 수 있도록 frame fixture와 state-transition expectation을 version control에 보존한다.

passing neighbor와 rollback generation도 함께 기록한다.
