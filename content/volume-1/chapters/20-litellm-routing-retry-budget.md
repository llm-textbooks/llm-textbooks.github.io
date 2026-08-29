# 20장. LiteLLM routing·retry·budget가 만드는 요청의 두 번째 수명

사용자는 요청 한 건을 보냈지만 gateway 뒤에서는 물리 시도가 둘이나 셋 생길 수 있다. 첫 deployment가 rate limit을 반환해 같은 model group의 다른 deployment로 retry하고, context window 오류가 나서 더 긴 model group으로 fallback할 수 있다. client는 최종 성공 하나만 받지만 비용 장부와 latency, provider request ID, streaming fragment는 모든 attempt를 기억해야 한다.

이 층을 “OpenAI 호환 JSON을 다른 provider JSON으로 바꾸는 프록시”라고만 보면 retry storm과 이중 과금, timeout 뒤 살아 있는 downstream 요청, stream 중간 fallback 불가능, budget race를 이해할 수 없다. 반대로 gateway가 model server라고 생각하면 tokenizer·chat template·logit·KV cache 정답까지 보장한다고 오해한다. LiteLLM은 provider contract와 요청 수명을 정규화하지만 downstream model의 수학을 통일하지 않는다.

이 장은 LiteLLM v1.97.0, commit `ef84494d52c6708e4e9f4a54ce551a265995ad8f`를 고정점으로 삼는다. 실행 결과를 만들지 않고 source에서 request→router→deployment→provider adapter→HTTP→표준 응답/stream→usage·cost·error의 handoff를 추적한다. 옵션 이름을 나열하는 대신 각 field가 어느 state를 바꾸고 어떤 실패를 새로 만드는지 본다.

## 20.1 retry와 fallback 사건에서 one logical request와 three attempts를 복원한다

한 사용자가 `model=chat-prod`로 요청했다. router는 `chat-prod` model group 아래 deployment A를 골랐고 A가 429를 반환했다. retry가 B를 골랐으나 connect timeout이 났다. fallback이 `chat-long`의 C를 골라 성공했다. 사용자 관점의 logical request는 하나지만 provider 관점의 physical attempt는 셋이다.

장부의 key는 `logical_request_id`, `attempt_id`, `model_group`, `deployment_id`, `provider`, `provider_request_id`, `start/end`, `outcome`, `retry/fallback reason`, `usage/cost confidence`다. attempt 번호만으로는 fallback tree를 표현하기 부족할 수 있어 parent attempt와 transition type도 둔다. 같은 deployment retry와 다른 model fallback을 구분한다.

세 attempt를 설명하는 canonical 표는 다음 하나다. 시간표, 비용표와 reservation 표를 따로 만들면 timeout 뒤 late usage가 서로 다른 요청처럼 갈라진다. 모든 금액은 예시이며 실제 판정에서는 provider price revision과 usage confidence를 같은 행에 붙인다.

| logical ID | attempt | 전이·deployment | 시작→관측 시각 | deadline 잔여 | token reservation | 실제·추정 usage/cost | terminal과 다음 상태 |
|---|---:|---|---|---:|---|---|---|
| L20 | A0 | initial route→A | 0.025s→8.025s | 약 31.98s | input/output 상한 0.02 | 429 전 일부 처리, 0~0.02 unknown | retryable 429, 2s backoff; reservation은 정산 전 유지 |
| L20 | B0 | same-group retry→B | 10.025s→25.025s | 약 14.98s | 추가 상한 0.03 | local read timeout, late usage 가능 | cancel requested; provider terminal 전에는 비용 0으로 닫지 않음 |
| L20 | C0 | semantic fallback→C | 25.025s→34.025s | 약 5.98s | 성공 가능 상한과 남은 budget | 성공 usage 0.07 예시 | client response를 한 번 commit하고 A0/B0를 별도 reconcile |

이 표에서 logical success는 C0 한 행이지만 total cost와 amplification은 세 행의 합이다. Token reservation은 예상 token 수와 단가를 분리해 보존하고, provider가 usage를 늦게 보고하면 `unknown`을 0으로 바꾸지 않는다. B0의 late success도 client response를 다시 쓰지는 못하지만 비용 원장은 갱신한다.

**gateway의 상태 전이**

```text
RECEIVED → AUTHORIZED/BUDGET_CHECKED → ROUTED
→ ATTEMPT_STARTED → HEADERS/STREAMING/SUCCEEDED/FAILED/TIMED_OUT
→ RETRY_WAIT → ATTEMPT_STARTED ...
→ FALLBACK_SELECTED → ATTEMPT_STARTED ...
→ RESPONSE_COMMITTED 또는 TERMINAL_ERROR
```

실제 code가 이 enum을 그대로 쓰지 않아도 의미 상태는 필요하다. response body가 client에 아직 전달되지 않았으면 다른 attempt로 전환할 수 있지만 첫 stream chunk가 전달된 뒤에는 이미 response contract가 commit되었다. timeout exception을 만들었다고 downstream socket/task가 종료되었다는 뜻도 아니다.

LiteLLM [Router class](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/router.py#L354-L520)는 deployment/model-list와 routing state의 중심이다. async entry들은 router 안에서 [completion과 fallback wrapper](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/router.py#L1980-L2220)로 이어진다. 최종 source span은 symbol과 함께 고정한다.

**logical success가 attempt success 합계와 다르다**

attempt C가 성공해 logical request가 성공하더라도 A가 provider에서 token을 일부 생성했다면 비용이 발생할 수 있다. B의 connect timeout은 provider에 도달하지 않았을 수 있고 read timeout은 이미 처리 중일 수 있다. gateway가 받은 usage만 합산하면 unknown spend가 남는다.

metric은 logical request success/latency와 attempt count/outcome/provider latency를 나눈다. raw request throughput이 100이고 attempt throughput이 160이면 retry amplification 1.6이다. provider 429가 늘 때 logical success가 유지되어 dashboard가 정상처럼 보여도 cost와 tail latency, downstream load가 악화될 수 있다.

**40초 deadline의 세 attempt를 시간·비용·reservation으로 계산한다.**

routing은 healthy capacity와 정책에 맞는 deployment를 고르기 위해 존재한다. retry는 transient failure를 사용자 실패로 확정하기 전에 한 번 더 기회를 준다. fallback은 한 model group이 실패했을 때 availability를 다른 semantic/cost domain으로 확장한다. timeout은 무한 대기를 끊고 자원을 회수하기 위한 deadline이다. budget은 미래 비용을 admission 시점에 제한하고 사후 usage로 reconciliation한다.

각 장치에는 대가가 있다. routing state가 stale하면 hot spot이 생긴다. retry는 amplification과 duplicate cost를 만든다. fallback은 semantic drift를 만든다. timeout은 orphan work를 만들 수 있다. budget estimate는 false reject/overspend 사이 오차를 갖는다. streaming은 response commit 뒤 선택지를 줄인다.

따라서 option 효과는 `field → normalized policy → logical/attempt state transition → provider wire action → client/cost/metric effect → counterexample`로 설명한다. retry 횟수를 올리면 성공률이 오른다고만 쓰지 않는다. 어떤 error가 retry되고 total deadline과 attempt cap, cost가 어떻게 바뀌는지 쓴다.

최종 invariant는 이렇다. **하나의 logical request가 여러 attempt로 갈려도 client response는 한 번만 commit되고, 각 attempt의 provider identity·error·usage·cost가 누락 없이 귀속되며, timeout/cancel/budget 상태가 다음 attempt와 downstream work의 lifetime에 일관되게 적용되어야 한다.**

gateway가 이 invariant를 지켜도 downstream model의 token semantics와 정답 품질은 별도 증거가 필요하다. 그 경계를 인정할 때 LiteLLM은 얕은 부록이 아니라 API contract와 여러 serving engine을 연결하는 운영 제어면으로 정확히 자리 잡는다.

**한 요청을 40초 deadline 안에서 끝까지 걷는다**

구체적인 시간표를 만들자. client는 t=0에 40초 deadline과 `model=chat-prod`, stream=false를 보낸다. proxy authentication과 budget reservation에 20ms가 걸리고 router가 deployment A를 5ms에 선택한다. A attempt는 connect 50ms 뒤 시작하지만 8초에 429를 반환한다. `Retry-After=2`를 존중해 2초 기다리고 B를 선택한다. B는 15초 read timeout이 난다. 남은 시간은 약 15초다. fallback C가 9초에 성공하면 logical latency는 약 34초다.

사용자 응답에는 C의 성공만 보이지만 root trace에는 admission, A 429, backoff, B timeout/cancel, fallback selection, C success가 모두 있어야 한다. provider latency C=9초만 기록하면 25초가 사라진다. router overhead라고 한 덩어리로 적어도 어느 policy가 시간을 썼는지 알 수 없다.

각 attempt timeout을 고정 15초로 다시 적용하면 C가 client deadline 뒤까지 실행될 수 있다. C를 시작할 때 남은 deadline에서 response serialization headroom을 빼 effective timeout을 정해야 한다. retry count가 남았다는 이유만으로 새 attempt를 시작하지 않는다. next attempt가 성공할 최소 시간보다 remaining budget이 작으면 terminal timeout을 선택할 수 있다.

이 계산의 직관은 wall-clock budget이 attempt 횟수를 제약한다는 것이다. 한계는 provider latency를 사전에 정확히 모른다는 점이다. historical quantile, model class, region을 이용해 “시작할 가치”를 판단할 수 있지만 availability와 tail tradeoff가 있다. source가 deadline-aware인지 단순 per-attempt timeout인지 확인한 뒤 실제 보장을 쓴다.

**exception history는 fallback 판단과 사용자 오류를 동시에 떠받친다**

A의 429에는 status, provider code, retry-after, request ID가 있다. B timeout에는 connect/read phase와 elapsed가 있다. C가 최종 실패하면 client-facing error는 어느 exception을 대표로 할지 결정해야 한다. 마지막 error만 보내면 최초 rate limit을 잃고, 모든 detail을 보내면 deployment 이름과 endpoint가 노출될 수 있다.

router kwargs 또는 metadata에 exception history를 누적하는 경로를 읽고 redaction이 어느 경계에서 적용되는지 본다. 운영 trace에는 normalized category와 안전한 deployment ID, 원본 provider request ID를 권한 보호 하에 남긴다. client에는 stable error type, status, retry guidance와 logical request ID를 준다.

context-window fallback에서는 원본 exception category가 routing edge를 선택한다. mapper가 provider message 문자열에 의존하면 provider wording 변화로 분류가 깨질 수 있다. typed status/code가 있으면 우선 사용하고 문자열 heuristic은 evidence와 test를 갖는다. content policy와 context length를 같은 invalid request로 뭉개지 않는다.

**retry storm을 수치로 계산한다**

초당 logical request 100개가 들어오고 A가 모두 429라서 평균 2 retries 후 B가 성공한다고 하자. attempt rate는 초당 300개다. B가 감당 가능한 capacity가 150이면 retry가 B도 overload시켜 429를 늘린다. logical arrival이 그대로인데 downstream load가 두 배 이상 된다.

backoff 없이 즉시 retry하면 동기화된 burst가 생긴다. exponential backoff와 jitter는 attempt를 시간에 분산하지만 client latency를 늘린다. cooldown이 A를 빠르게 후보에서 제외하면 불필요한 첫 실패를 줄일 수 있다. 그러나 false-positive cooldown은 healthy capacity를 버린다.

metric은 logical RPS, attempt RPS, attempts/logical, retry reason, cooldown entries, candidate count를 함께 본다. logical error rate가 낮아도 amplification이 rising이면 조기 경고다. provider별 성공 request만 세어 load를 추정하지 않는다. timeout/cancel late completion도 attempt load에 포함한다.

circuit breaker와 cooldown은 이름보다 state scope가 중요하다. gateway process local이면 replica 10개가 각자 A를 한 번씩 실패한 뒤 배운다. distributed shared state면 빠르게 전파되지만 stale/coordination overhead가 있다. 배포 topology와 failure domain을 source/config에 연결한다.

**hedging은 retry와 다른 동시성 비용을 가진다**

tail latency를 줄이려고 일정 지연 뒤 두 번째 provider attempt를 동시에 시작하는 hedged request를 생각하자. 먼저 성공한 response를 client에 commit하고 loser를 cancel한다. 순차 retry와 달리 latency는 줄 수 있지만 두 provider가 동시에 계산해 비용과 load가 늘어난다.

stream에서는 어느 attempt가 first byte를 먼저 보냈는지 winner를 정할 수 있지만 quality/complete latency와 다를 수 있다. tool call fragment가 시작된 뒤 winner를 바꾸면 안 된다. budget reservation도 최대 두 attempt 비용을 고려해야 한다. 이 장의 고정 source가 hedging을 지원한다고 단정하지 않고, retry와 혼동하지 않기 위한 대비로 사용한다.

**deployment identity가 semantic identity인지 검증한다**

`chat-prod` 아래 A는 vLLM model revision r1, B는 SGLang r1, C는 provider managed model alias라고 하자. A/B는 같은 artifact/tokenizer/template를 고정할 수 있지만 backend numeric과 sampling 구현이 다를 수 있다. C는 provider alias가 언제 revision을 바꾸는지 모를 수 있다.

availability group과 semantic-equivalence group을 분리할 수 있다. strict request는 A/B만 허용하고 best-effort request는 C까지 fallback한다. structured JSON, logprobs, tool use, multimodal capability도 deployment metadata로 검증한다. capability가 없으면 request transform에서 조용히 field를 drop하지 않고 policy에 따라 reject/fallback한다.

gateway differential은 같은 rendered payload만으로 끝나지 않는다. self-hosted server에는 tokenizer/model revision과 chat template override를 고정하고 provider managed endpoint에는 API model version과 documented behavior를 기록한다. gateway는 downstream hidden/logits를 볼 수 없으므로 semantic parity를 별도 test suite와 output contract로 관리한다.

**provider adapter 번역을 tool-call 예로 해부한다**

표준 request에 system/user messages, tools schema, tool_choice가 있다. provider A는 tool schema를 별도 field로 받고, B는 content block 형식을 요구할 수 있다. adapter는 role/content/tool IDs를 변환한다. unsupported JSON Schema keyword를 제거하거나 오류로 만들 수 있다.

response에서 provider는 tool name과 arguments를 여러 streaming chunk로 나눠 보낼 수 있다. 표준 chunk로 변환할 때 동일 tool call index/ID에 arguments 문자열을 순서대로 붙인다. UTF-8 또는 JSON token boundary 중간 fragment도 가능하다. 각 chunk를 독립 JSON으로 parse하면 실패한다.

fallback이 provider A에서 B로 바뀌면 tool call ID format과 argument serialization이 달라질 수 있다. client는 표준화된 shape를 받더라도 exact ID stability를 기대해서는 안 될 수 있다. gateway가 생성한 stable ID를 쓰는지 provider ID를 보존하는지 source contract를 본다.

usage에서 tool schema/input token counting도 provider마다 다를 수 있다. 같은 standard request라고 prompt token cost가 같지 않다. budget reservation은 provider candidate별 estimate range를 사용할 수 있고 최종 usage로 reconcile한다.

**response normalization이 정보 손실을 관리하는 법**

provider A는 stop reason `end_turn`, B는 `stop`, C는 `content_filter`를 반환한다. 표준 finish reason으로 mapping하면서 원본 reason과 provider code를 protected metadata에 남긴다. client portability를 위해 공통 enum을 쓰되 incident에는 원인이 필요하다.

reasoning token, cached input token, safety ratings, provider latency headers도 공통 response에 완전히 들어맞지 않을 수 있다. extra fields를 extension으로 보존하거나 callback/spend record에 둔다. 알 수 없는 field를 무조건 버리면 billing과 품질 조사에서 근거가 사라진다.

표준 response ID와 provider request ID를 구분한다. retry/fallback logical response는 하나지만 attempt마다 provider ID가 있다. client support ticket에는 logical ID로 시작해 attempt IDs로 내려갈 수 있어야 한다. provider support에는 해당 provider ID를 전달한다.

**budget reservation을 세 attempt에 적용한다**

logical request 예상 prompt cost 0.02, max output cost 0.08이라 총 0.10을 reserve했다고 하자. A가 0.01을 쓰고 timeout, B가 0.03을 쓰고 실패, C가 0.07에 성공하면 actual aggregate attempt cost는 0.11이다. logical final usage C만 보면 0.07이라 retry waste 0.04를 잃는다.

reservation 하나 0.10으로 세 concurrent attempt 가능성을 충분히 막지 못한다. policy가 retry cost를 tenant budget에 포함한다면 attempt 시작 전에 추가 reserve 또는 remaining headroom을 확인해야 한다. provider error가 usage를 주지 않으면 unknown estimate를 둔다. gateway 장애 비용을 사용자에게 청구할지 platform overhead로 분류할지도 제품 정책이다.

budget DB update가 async batch라면 admission read가 stale할 수 있다. reservation store가 atomic하게 pending spend를 반영해야 한다. reservation ID는 logical request와 attempt에 연결하고 성공/실패/cancel/crash에서 release/reconcile한다. 만료시간만으로 orphan을 회수하면 긴 request와 충돌할 수 있어 heartbeat/state를 본다.

**timeout 뒤 late success를 회계에 넣는다**

B가 gateway에서는 15초 timeout이었지만 provider가 25초에 success와 usage를 callback/webhook 또는 billing export로 남겼다고 하자. client는 C의 response를 받았고 B output은 사용하지 않았다. B의 cost는 wasted attempt이지만 실제 spend다.

attempt state를 `TIMED_OUT_CLIENT_SIDE`에서 terminal로 닫아 버리면 late event를 중복/unknown으로 처리할 수 있다. logical response commitment와 provider 정산을 다른 state dimension으로 둔다. late success는 client response를 바꾸지 않지만 cost/reliability metric을 갱신한다.

idempotency key가 provider에 전달되었다면 retry가 동일 operation으로 deduplicate되는지 contract를 확인한다. completion generation은 provider가 idempotency를 지원하지 않거나 cached response 정책이 다를 수 있다. gateway가 local cache key만으로 exactly-once provider billing을 보장한다고 하지 않는다.

**stream first-byte와 response commit을 timestamp로 본다**

attempt A가 t=1에 headers, t=2에 role chunk, t=3에 content 첫 bytes를 보냈다고 하자. gateway가 role chunk를 client에 전달한 시점부터 response status/headers와 provider/model metadata는 사실상 commit될 수 있다. content가 아직 없다는 이유로 자유롭게 B로 fallback하면 duplicate role/ID가 생길 수 있다.

commit 기준을 first downstream byte, first semantic content, headers 중 어디에 둘지 구현 contract로 정한다. 안전한 기본은 client response가 시작되기 전만 fallback하는 것이다. buffer를 두어 provider 첫 몇 chunks를 검증한 뒤 client에 보내면 fallback window는 늘지만 TTFT와 memory가 늘어난다.

stream error는 표준 error event를 보낼 수 있는 protocol인지, 단순 connection close인지 확인한다. HTTP status는 이미 200으로 commit되었을 수 있다. observability에는 logical status 200과 stream terminal error를 둘 다 기록해 성공률 왜곡을 막는다.

**duplicate billing과 idempotency race도 같은 attempt 기록에서 닫는다.**

사건 G20은 client timeout 뒤 gateway가 fallback B를 성공시켰지만 provider A도 background에서 완료돼 두 usage가 청구된 경우다. client는 B response 하나만 받았고 cost dashboard는 logical request 하나로 aggregate해 중복 attempt를 숨겼다.

**세 원장과 reservation 계산**

logical ledger는 client contract와 final chosen attempt를, attempt ledger는 provider별 reservation/usage/terminal을, budget ledger는 reserved·reported·estimated·released를 가진다. 하나의 `cost` 필드로 접지 않는다.

예상 input/output 비용이 A $0.02, B $0.03이면 A0 admission에서 $0.02를 reserve한다. timeout 후 A cancel completion이 확인되지 않은 상태에서 B0를 시작하려면 worst-case reserved는 $0.05다. logical budget $0.04라면 B fallback을 시작하면 ceiling을 넘을 수 있다.

cancel intent만으로 A reservation을 즉시 release하면 A가 실제 완료/청구될 때 overspend가 생긴다. provider terminal/usage confidence에 따라 `pending_cancel` reservation을 유지한다. hard ceiling과 availability tradeoff를 정책으로 명시한다.

**budget race timeline**

t0 A reserve 0.02, t1 A submit, t2 local timeout/cancel intent, t3 B reserve0.03, t4 B success/commit, t5 A late usage0.02, t6 reconciliation이다. t2에서 A reserve를 0으로 만들면 t3 admission은 0.03만 보고 통과하지만 final cost0.05다.

atomic operation은 “현재 reserved+new worst-case<=limit이면 reserve”다. concurrent logical requests도 같은 tenant/model key에서 compare-and-update가 필요하다. read-check-write가 분리되면 두 요청이 같은 남은 budget을 동시에 소비한다.

**duplicate billing과 response dedup은 다르다**

client response를 하나만 commit해도 provider attempts 두 개가 bill될 수 있다. response dedup은 externally selected result를, billing dedup/reconciliation은 attempt charges를 다룬다. provider가 서로 다르면 동일 idempotency key로 과금을 자동 dedup해 주지 않을 수 있다.

gateway는 logical ID와 provider attempt IDs를 보존해 cost를 모두 귀속한다. final success rate 분모에서 attempt amplification을 숨기지 않는다. `attempts/logical`, `cost/logical`, cancelled-late-usage를 metric으로 둔다.

**idempotency와 tool side effect**

non-stream request라도 provider A가 tool call을 생성/commit했으나 response가 유실되고 B retry가 다른 tool call을 만들 수 있다. gateway가 model output만 dedup해 application side effect를 막을 수 없다. logical idempotency key를 downstream agent/tool executor에 전달한다.

stream first chunk 뒤 fallback은 duplicate visible prefix와 tool buffer를 만든다. default rule은 external commit 후 transparent retry/fallback 금지다. explicit resume protocol이나 application consent가 있을 때만 별 semantics로 지원한다.

**incident first divergence**

G20에서 router selection 자체는 정상이다. first divergence는 A cancel intent를 provider terminal로 간주해 reservation을 release한 budget transition이다. B success는 downstream 결과다. retry count를 줄이는 것은 완화일 수 있지만 root contract는 pending attempt cost 보존이다.

수정은 attempt state `RUNNING→CANCEL_REQUESTED→CANCEL_CONFIRMED|LATE_COMPLETED|UNKNOWN_EXPIRED`와 reservation policy를 둔다. provider usage report가 늦으면 confidence와 settlement window를 갖는다. 유효시간 만료로 unknown을 0으로 만들지 않고 audit write-off/estimate 정책을 둔다.

## 20.2 routing은 model 이름을 deployment와 credential로 해석한다

client의 `model=chat-prod`는 반드시 provider의 실제 model ID가 아니다. gateway model group, alias 또는 policy 이름일 수 있다. router는 후보 deployment 목록에서 model, api base, credential reference, region과 rate/cooldown state를 반영해 하나를 선택한다. provider adapter는 선택된 target을 provider-specific request로 변환한다.

routing의 직관은 load balancer지만 한계가 있다. LLM deployment는 단순히 같은 HTTP backend replica가 아닐 수 있다. tokenizer revision, context window, quantization, adapter, safety policy와 가격이 다르면 fallback 뒤 semantic output과 usage가 달라진다. “동일 model group”이라는 운영 이름이 bitwise model identity를 보장하지 않는다.

**selection 전에 후보 eligibility가 바뀐다**

deployment가 cooldown, rate limit, unhealthy state, allowed region, budget와 capability 조건 때문에 후보에서 빠질 수 있다. routing strategy는 남은 후보의 RPM/TPM, latency, usage 또는 cost 신호를 사용할 수 있다. field→candidate filter→selected deployment state→downstream latency/cost effect→selection log로 사슬을 쓴다.

Router의 deployment 선택과 cooldown 처리는 [routing core](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/router.py#L520-L900)와 [cooldown handlers](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/router_utils/cooldown_handlers.py#L1-L270)를 함께 읽는다. exception이 retryable인지와 deployment를 cooldown할지는 같은 결정이 아닐 수 있다.

**provider adapter는 표준 request의 손실 없는 번역을 약속하지 않는다**

OpenAI-style `messages`, tools, response format, stream options가 provider마다 다른 field와 의미로 매핑된다. unsupported parameter를 drop, transform, error 중 무엇으로 처리하는지 adapter contract를 확인한다. tool role이나 multimodal content block, reasoning token field가 provider별로 다르다.

gateway는 JSON shape를 표준화할 수 있지만 provider tokenizer가 같은 bytes를 같은 token IDs로 만드는지 보장할 수 없다. `max_tokens` 의미, stop 처리, logprobs 지원과 usage counting도 다를 수 있다. fallback이 다른 provider로 가면 동일 prompt string은 유지되어도 model input과 output 분포가 달라질 수 있다.

**credential과 endpoint identity는 관측에서 비밀을 노출하지 않는다**

attempt ledger에는 deployment/credential identity가 필요하지만 raw API key를 넣지 않는다. stable secret reference 또는 hash, region과 api base의 안전한 alias를 쓴다. provider error body에 secret/header가 섞일 수 있어 client-facing error와 운영 증거를 분리한다.

model alias만 metric label로 쓰면 어느 deployment가 실패했는지 모른다. 반대로 full endpoint URL과 key ID를 공개 label로 쓰면 보안과 cardinality 문제가 생긴다. bounded deployment ID와 trace exemplar를 사용한다.

## 20.3 retry와 fallback은 같은 재시도가 아니라 다른 상태 전이다

retry는 흔히 같은 logical model group에서 다른 deployment 또는 같은 operation을 다시 시도한다. fallback은 exception category나 policy에 따라 다른 model group/provider로 이동한다. 둘 다 새 physical attempt를 만들지만 semantic drift와 budget, max attempt 계산이 다르다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- LiteLLM async 경로는 [async_function_with_fallbacks](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/router.py#L6406-L6499)가 retry wrapper를 호출하고 exception에서 fallback common utility로 이동한다.
- [async_function_with_retries](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/router.py#L6501-L6787)는 attempt 반복, retry count와 delay의 owner다.
- sync wrapper는 [function_with_fallbacks](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/router.py#L6788-L6805)에서 async 수명을 감싼다.

**재시도 가능성은 HTTP status 하나로 결정되지 않는다**

429와 일부 5xx는 retry 후보가 될 수 있지만 401 credential 오류, invalid request, content policy, context window는 다른 처리가 필요하다. exception mapper가 provider-specific error를 표준 category/status로 바꾸고 router가 retry/fallback policy를 적용한다. 원본 cause, normalized class, selected transition을 모두 보존한다.

context window error는 같은 deployment retry로 해결되지 않고 longer-context fallback 후보가 될 수 있다. content policy fallback은 다른 provider/policy로 이동할 수 있지만 안전 의미가 달라질 수 있다. auth error를 무한 retry하면 credential 문제를 load spike로 바꾼다.

**retry budget은 retry 횟수와 wall-clock deadline을 함께 가져야 한다**

`num_retries=3`, fallback 2개라면 최대 physical attempts가 직관과 다를 수 있다. 전체 request attempt cap과 model별 retries/fallback cap의 우선순위를 source에서 확인한다. LiteLLM global/router 관련 field는 [retry/fallback globals](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/__init__.py#L490-L512)에 나타난다.

각 attempt timeout 30초를 세 번 허용한다고 client deadline 40초를 지킬 수 없다. 남은 deadline을 다음 attempt timeout과 backoff에 반영해야 한다. retry delay 뒤 남은 시간이 model 응답에 충분하지 않으면 새 attempt를 시작하지 않는 편이 낫다. 횟수 budget과 시간 budget, 비용 budget을 함께 본다.

**backoff와 cooldown은 서로 다른 시간축이다**

backoff는 한 logical request가 다음 attempt를 시작하기 전에 기다리는 시간이다. cooldown은 deployment를 여러 request의 후보에서 잠시 제외하는 shared state다. 둘을 같은 timer로 설명하면 장애 회복과 thundering herd를 이해하기 어렵다.

429 `Retry-After`를 존중하는지, jitter가 있는지, exception category별 cooldown이 어떻게 정해지는지 본다. 모든 gateway replica가 cooldown state를 공유하지 않으면 각각 같은 unhealthy deployment를 찌를 수 있다. local cache와 distributed coordination의 범위를 기록한다.

## 20.4 timeout은 예외를 반환하는 순간과 downstream 작업이 끝나는 순간을 나눈다

client timeout, gateway total timeout, attempt connect/read/write/pool timeout, provider server timeout이 겹친다. 가장 바깥 deadline이 끝나면 사용자에게 504를 줄 수 있지만 내부 task가 계속 provider response를 기다리면 connection과 비용이 남는다. Python task cancel이 HTTP request cancel 또는 provider generation abort까지 전파되는지도 별도다.

LiteLLM의 generic [timeout decorator](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/timeout.py#L24-L88)는 sync future와 async wait path를 보여 준다. [async worker cleanup](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/timeout.py#L90-L117)에서 cancel/join lifetime을 확인한다. decorator의 exception과 provider HTTP client's timeout contract를 혼동하지 않는다.

**timeout 이름의 우선순위를 effective value까지 추적한다**

`force_timeout`, `request_timeout`, router timeout, HTTP client timeout이 모두 존재할 수 있다. parser에 field가 있다는 사실보다 어느 wrapper가 최종 seconds를 소비하는지 본다. source에서 async branch가 계산한 local value와 실제 `wait_for` argument가 같은지도 검토한다.

effective deadline ledger에는 client disconnect/deadline, proxy admitted time, router remaining budget, attempt timeout, provider header/first-byte/complete time을 둔다. TTFT가 timeout보다 작아 stream이 시작됐지만 전체 generation이 timeout을 넘는 경우와 첫 byte도 못 받은 경우를 분리한다.

**retry-after-timeout은 duplicate work를 만들 수 있다**

read timeout이 났지만 provider가 요청을 계속 처리한다면 retry B와 A가 동시에 생성할 수 있다. provider idempotency key나 cancel API가 없으면 두 attempt 모두 과금될 수 있다. non-stream final response는 먼저 성공한 하나만 client에 commit할 수 있지만 spend reconciliation에는 둘을 남겨야 한다.

tool side effect가 provider/model 밖에서 실행되는 agent workflow라면 retry semantics는 더 위험하다. 이 장의 chat completion gateway는 tool call suggestion을 반환할 뿐 외부 tool 실행을 자동으로 exactly-once 보장하지 않는다. 3권 agent 메커니즘에서 확대한다.

## 20.5 budget은 admission estimate와 사후 usage reconciliation 사이의 원장이다

budget check를 “현재 spend < max_budget이면 통과”로만 구현하면 concurrency에서 초과한다. spend 90, max 100일 때 예상 cost 8인 요청 두 개가 동시에 check하면 둘 다 통과해 106이 된다. reservation 또는 atomic counter, concurrency headroom이 필요하다.

예산 초과를 가르는 판정의 상태 owner는 다음 좌표에서 확인한다.

- LiteLLM의 budget model은 [canonical budget fields](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/models/budget.py#L1-L53), user의 [is_over_budget](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/models/user.py#L20-L68), proxy의 [budget reservation](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/proxy/spend_tracking/budget_reservation.py#L1-L260)에서 상태 owner를 찾는다.
- key/user/team/tag/model/session budget이 겹치면 우선순위와 atomic boundary를 확인한다.

**estimate는 tokenizer와 provider pricing에 의존한다**

request admission 전에는 output token 수를 모르므로 max tokens 또는 historical estimate로 reserve할 수 있다. prompt token estimate도 provider tokenizer와 다를 수 있다. gateway tokenizer count가 downstream usage와 다르면 reservation과 actual cost가 갈린다.

model price table과 provider-reported usage, cached/input/output/reasoning token categories를 매핑해야 한다. fallback이 다른 provider/model로 가면 단가가 바뀐다. logical request 예상 cost 한 개가 아니라 attempt별 reserved/actual/unknown cost를 기록한다.

**soft budget과 hard budget은 사용자 경험이 다르다**

soft budget은 alert나 속도 제한/cooldown policy를 촉발할 수 있고 hard max는 admission을 거부할 수 있다. reset duration과 timezone/clock, DB update lag가 effective state에 영향을 준다. request가 이미 진행 중인데 budget을 넘었을 때 stream을 중단할지 완료시킬지도 policy다.

budget exceeded를 429/403/other status로 표준화하는 방식은 client retry behavior를 바꾼다. budget error를 transient 429로 보내 client가 재시도하면 gateway load만 늘 수 있다. retryable category와 user-visible error contract를 맞춘다.

**usage 누락과 delayed reconciliation을 숨기지 않는다**

stream이 중간 취소되거나 provider error가 usage를 반환하지 않으면 exact cost를 즉시 모를 수 있다. estimate, provider final usage, later billing export의 confidence를 나눈다. zero로 기록하면 budget을 과소평가하고 maximum으로 확정하면 사용자를 과도하게 막을 수 있다.

reservation은 completion 성공/실패/timeout에서 release 또는 actual로 convert되어야 한다. process crash 뒤 orphan reservation을 어떻게 회수하는지 본다. duplicate spend event의 idempotency key와 DB batch writer도 중요하다.

## 20.6 streaming은 첫 chunk에서 fallback 가능성을 response commitment로 바꾼다

non-stream 요청은 provider A가 실패하면 client에 아무 것도 보내기 전 B로 fallback할 수 있다. streaming은 A의 첫 content delta를 client에 보낸 순간 provider/model identity와 response sequence 일부가 공개된다. 이후 A가 실패했을 때 B의 새 stream을 이어 붙이면 두 model의 출력이 한 response에 섞인다.

따라서 pre-first-byte fallback과 post-first-byte failure를 구분한다. 첫 chunk 전 error는 새 attempt로 전환할 수 있다. 첫 chunk 후에는 terminal stream error를 보내거나 연결을 닫는 편이 semantic하게 정직할 수 있다. 제품이 resume protocol을 정의하지 않았다면 투명 fallback이라고 부르지 않는다.

**provider chunk를 표준 delta로 바꾸는 동안 상태가 누적된다**

provider마다 content, tool call arguments, reasoning, usage final chunk, finish reason 표현이 다르다. adapter는 partial fragments를 표준 chunk로 바꾸고 tool call index/ID, UTF-8 boundary를 유지해야 한다. chunk 하나의 변환과 전체 stream state machine을 분리한다.

usage가 final chunk에만 오면 client disconnect 전에는 cost를 확정하기 어렵다. `include_usage` 같은 option과 provider 지원을 확인한다. partial output bytes와 generated token usage가 같지 않다. buffering과 backpressure 때문에 provider에서 받은 시점과 client에 보낸 시점도 다르다.

**client disconnect가 downstream cancel로 이어지는지 증명한다**

ASGI/request task cancel, async generator close, HTTP stream context close, provider connection cancel이 순서대로 전파되어야 downstream work를 줄일 수 있다. 한 단계가 `CancelledError`를 삼키거나 background callback이 reference를 유지하면 provider generation이 계속될 수 있다.

cancel이 provider에 전파돼도 이미 사용한 token 비용은 남는다. cancel latency, downstream connection closed, final usage known/unknown을 기록한다. logical request status `CLIENT_DISCONNECTED`와 attempt status `CANCEL_REQUESTED/CANCELLED/COMPLETED_LATE`를 나눈다.

### slow client는 model latency처럼 보일 수 있다

provider delta는 빠르게 오지만 client network가 느려 send buffer가 막히면 gateway memory와 stream duration이 늘어난다. provider read와 client write를 같은 coroutine에서 직렬 처리하면 backpressure가 upstream read를 늦출 수 있다. bounded queue가 있으면 overflow policy를 본다.

TTFT, provider inter-chunk latency, gateway transform time, client write wait를 나눈다. 사용자 ITL이 느린데 provider ITL이 정상이면 model server를 최적화할 일이 아니다. 반대로 gateway buffer가 chunks를 합쳐 늦게 flush하면 provider와 client timestamp가 갈린다.

## 20.7 gateway가 보장할 수 있는 것과 없는 것을 경계로 쓴다

LiteLLM은 provider별 request/response/error를 공통 API에 가깝게 변환하고 routing·retry·fallback·budget·observability policy를 적용할 수 있다. 하지만 서로 다른 provider model이 같은 tokenizer, chat template, context truncation, tool grammar, safety filter, logits 또는 deterministic seed semantics를 갖게 만들지는 못한다.

같은 `model=chat-prod`, temperature 0, seed 42라도 deployment A와 B의 artifact/tokenizer가 다르면 output이 달라질 수 있다. gateway는 routing을 sticky하게 하거나 semantic identity가 같은 deployment만 group으로 묶을 수 있지만 downstream truth를 자동 증명하지 않는다. model group inventory에 artifact/tokenizer/template/capability digest를 연결해야 한다.

### usage 표준화는 측정 정의를 동일하게 만들지 않는다

provider가 prompt/completion tokens를 반환해도 cached token, reasoning token, audio/image token과 billing unit이 다를 수 있다. standardized fields는 공통 column을 제공하지만 비교 가능한 단위인지 설명해야 한다. gateway-side token estimate와 provider final usage를 구분한다.

finish_reason도 provider 원인을 완전히 보존하지 못할 수 있다. length, stop, content filter, tool call, provider-specific safety를 mapping할 때 original reason을 보호된 운영 metadata에 남긴다. client-facing portability와 incident evidence의 상세도를 분리한다.

### fallback은 availability와 semantic consistency를 교환한다

fallback으로 success rate는 높아질 수 있지만 model 품질, latency, 가격, data residency, safety policy가 바뀔 수 있다. model group마다 허용 가능한 fallback edge와 금지 edge를 정의한다. context-window fallback은 longer model로 가더라도 tokenizer/template가 달라 prompt 의미가 바뀔 수 있다.

high-stakes structured output에서 provider를 바꾸기보다 동일 semantic deployment retry만 허용할 수 있다. casual chat은 broader fallback을 허용할 수 있다. 하나의 전역 fallback list가 모든 endpoint/request class에 맞지 않는다.

**source walk는 표준 API에서 provider wire까지 왕복한다**

입구에서 proxy가 인증·key/team/user metadata와 budget/rate limits를 확인한다. request model alias와 body를 정규화하고 router kwargs에 운영 metadata를 붙인다. router가 deployment를 고른 뒤 provider adapter와 HTTP client가 wire request를 만든다. response는 provider parser→standard object→callback/spend logging→proxy response로 돌아온다.

- router의 retry/fallback core는 앞에서 본 [fallback common utility](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/router.py#L6153-L6405), [fallback entry](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/router.py#L6406-L6499), [retry loop](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/router.py#L6501-L6787)을 의미 좌표로 삼는다.
- exception history가 kwargs/metadata에 어떻게 쌓이는지와 client 노출 redaction을 확인한다.

provider adapter는 각 integration package의 config/handler에서 supported params, transform request, async completion/stream parse를 찾는다. 모든 provider를 한 장에 나열하지 않고 OpenAI-compatible/vLLM target 하나와 Anthropic 등 shape가 다른 하나를 수직 추적한다. 공통 core와 provider-specific branch를 구분한다.

- spend와 budget은 callback 성공/실패 시점, usage extraction, DB batching을 잇는다.
- [spend tracking utility](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/proxy/spend_tracking/spend_tracking_utils.py#L1-L260), [spend log batching](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/proxy/db/spend_log_batching.py#L1-L87), [budget limiter hook](https://github.com/BerriAI/litellm/blob/ef84494d52c6708e4e9f4a54ce551a265995ad8f/litellm/proxy/hooks/max_budget_limiter.py#L1-L84)을 request attempt identity로 연결한다.

**incident laboratory는 gateway와 downstream 원인을 갈라낸다**

첫 사건은 logical success rate 99.9%인데 provider attempts가 두 배가 된 경우다. logical metric만 보면 정상이다. attempt outcome을 보면 deployment A 429, B success가 반복된다. cooldown이 gateway replica 사이 공유되지 않거나 capacity weight가 stale할 수 있다. retry amplification, per-deployment selection과 cooldown state를 본다.

두 번째 사건은 client p99가 70초인데 provider p99는 25초다. routing wait, backoff, 여러 attempt 합계, gateway queue와 client write를 timeline으로 붙인다. 최종 성공 provider latency 하나만 trace에 남기면 앞 실패 40초가 사라진다. attempt span을 logical root span 아래 둔다.

세 번째 사건은 timeout response 뒤 provider bill이 계속 늘어나는 경우다. timeout exception timestamp, task cancel, HTTP close, provider request complete와 usage arrival을 비교한다. gateway가 response를 끝낸 것과 downstream work 종료를 구분한다. idempotency/cancel 지원이 없다면 unknown/late cost를 reconciliation한다.

네 번째 사건은 stream이 중간에 끊긴 뒤 fallback model 문장이 이어지는 경우다. 첫-byte commit 이후 fallback을 허용한 path를 찾는다. response ID/model/finish reason과 tool-call fragment가 두 attempt에서 섞였는지 본다. pre-commit fallback만 허용하도록 state boundary를 고친다.

다섯 번째 사건은 budget 100인데 spend가 120까지 올라가는 경우다. concurrent admission reservation, retries와 fallback의 multi-attempt estimate, delayed usage, DB atomicity를 본다. budget check timestamp와 reservation/actual/release event를 logical/attempt ID로 연결한다.

여섯 번째 사건은 fallback 뒤 JSON schema failure가 늘어난 경우다. gateway routing은 성공했지만 provider/model의 structured-output semantics가 다르다. selected deployment capability와 transformed response format, model identity를 본다. gateway가 HTTP success를 semantic success로 간주한 범위를 명시한다.

## 20.8 router option을 physical attempt state까지 추적한다

### 옵션 하나를 상태 변화 여섯 단계로 읽는다

`num_retries=2`를 예로 들자. 첫 단계는 parser와 precedence다. request override, Router constructor, global default 중 어느 값이 effective인지 확인한다. None, 0, 음수, 문자열 변환과 endpoint별 override도 본다. 설정 파일에 2가 있다는 사실만으로 모든 request가 두 번 retry된다고 쓰지 않는다.

둘째 단계는 counter scope다. initial attempt를 retry 수에 포함하는지, 같은 deployment retry와 다른 deployment retry를 함께 세는지, fallback 뒤 counter를 reset하는지 확인한다. `num_retries_per_request`처럼 전체 cap이 별도면 local loop와 global cap의 최소값이 될 수 있다. source의 decrement 위치를 따라 최대 attempt tree를 그린다.

셋째는 eligibility다. 모든 exception이 counter를 소비하는지, retryable exception만 소비하는지 본다. invalid request가 즉시 fallback/terminal로 가고 429만 retry될 수 있다. retry count가 남아도 cooldown 때문에 후보 deployment가 없으면 종료될 수 있다.

넷째는 time state다. retry delay와 attempt timeout, total deadline을 붙인다. field 2는 최대 두 번의 추가 기회를 요청하지만 wall-clock이 충분하다는 보장은 아니다. backoff로 deadline을 소진하면 실제 retries는 0이나 1일 수 있다.

다섯째는 비용과 load다. 최대 physical attempts와 reservation, provider RPS amplification을 계산한다. success rate가 오를 가능성과 p99/cost가 오르는 조건을 함께 쓴다. 여섯째는 관측이다. effective retry cap, actual attempt count, skip reason, exception category, remaining deadline을 trace에 남긴다.

`fallbacks`도 같은 방식으로 읽는다. model group→ordered/eligible fallback targets, exception category별 edge, recursive fallback와 cycle 방지, max fallback cap, semantic capability, budget/price를 연결한다. A→B와 B→A cycle이 config에 있으면 visited set 또는 cap이 무한 loop를 막는지 본다.

`context_window_fallbacks`는 provider error classification 또는 preflight token estimate가 trigger가 될 수 있다. preflight estimate가 부정확하면 긴 model로 불필요하게 보내거나 작은 model에 보냈다 실패한다. fallback target이 정말 더 긴 context와 같은 input capability를 갖는지 inventory를 본다.

`request_timeout`은 sync decorator, async wait, provider HTTP client 중 어디에 쓰이는지 분리한다. connect/read/write/pool timeout object와 단일 seconds가 다르다. streaming에서 first-byte 이후 전체 duration에도 같은 timeout을 적용하는지 idle/read timeout인지 확인한다.

`max_budget`은 entity scope와 spend source, reset window, pending reservation 포함 여부를 연결한다. 0이 unlimited인지 zero budget인지 default semantics를 source에서 본다. soft budget과 hard rejection, budget fallback이 서로 다른 transition이다. 설정 하나가 요청 중단까지 자동 의미한다고 가정하지 않는다.

### 실패 주입을 실행하지 않고 test source로 설계한다

429 fixture는 provider mock가 status와 Retry-After를 반환하고 router가 다른 deployment를 선택하는지 본다. 기대 evidence는 attempt count, backoff 범위, cooldown state, final response identity다. 실제 provider를 공격하거나 서버를 실행할 필요 없이 unit/integration test contract를 읽을 수 있다.

connect timeout과 read timeout을 분리한다. connect failure는 provider가 work를 시작하지 않았을 가능성이 높고 read timeout은 시작했을 수 있다. mock가 cancel signal을 받았는지, late response callback이 spend를 기록하는지 test case를 설계한다. exception class만 같게 만들면 lifetime 차이를 놓친다.

stream fixture는 first chunk 전 failure와 두 chunk 후 failure를 나눈다. 전자는 fallback success를 기대할 수 있고 후자는 response mixing을 금지한다. chunk tool-call fragments와 usage final chunk를 포함한다. async generator close 때 HTTP context와 callback이 정리되는지 본다.

budget concurrency fixture는 spend 90/max 100에서 estimate 8인 두 request가 barrier를 동시에 통과하도록 한다. atomic reservation이면 하나만 허용하거나 policy-defined headroom을 적용한다. DB update 결과만 나중에 보는 test는 admission race를 잡지 못한다.

fallback semantic fixture는 A/B adapter가 같은 standard request를 서로 다른 wire payload로 변환하는 expected snapshots를 가진다. unsupported tool schema가 drop되는지 error인지 명시한다. response standard fields와 original provider metadata 보존을 비교한다.

### observability를 논리 요청과 attempt tree로 그린다

root span은 client-facing logical request다. admission/auth/budget, routing selection, 각 provider attempt, backoff, fallback decision, stream transform, response commit, spend reconciliation을 child span/event로 둔다. attempt span은 provider/deployment 안전 ID, category, timeout phase, usage confidence를 가진다.

logical latency histogram에는 client가 기다린 전체 시간을 넣는다. provider attempt histogram은 개별 wire latency를 넣는다. 두 histogram을 혼합하면 평균이 의미를 잃는다. retries count, fallbacks count, attempts per logical, response committed after attempts를 별도 metric으로 둔다.

status code label만으로 failure를 분류하지 않는다. 429가 provider rate limit인지 gateway budget/rate limit인지 source label을 둔다. 400도 context, unsupported parameter, content policy mapping일 수 있다. bounded normalized category와 original detail trace를 나눈다.

stream에는 request accepted, provider headers, provider first chunk, gateway first client byte, last provider chunk, last client byte, disconnect/cancel timestamps를 둔다. 이로써 provider TTFT와 gateway transform/queue, network delivery를 나눈다. chunk마다 Prometheus event를 만들지 않고 trace 또는 summary histogram을 사용한다.

cost metric은 estimated reserved, provider reported actual, calculated actual, unknown/late, retry waste를 구분한다. logical final cost만 집계하면 reliability mechanism의 비용을 숨긴다. attempt cost를 사용자/team/provider와 연결하되 중복 spend event를 idempotency key로 막는다.

### routing algorithm을 바꾸기 전 실험 계약을 만든다

latency-based routing을 평가하려면 provider latency 측정이 queue, network, model generation length를 공정하게 반영하는지 본다. 짧은 요청만 받은 deployment가 빠르게 보일 수 있다. token-normalized latency와 request class, streaming TTFT/total을 분리한다.

least-busy 또는 usage-based routing은 local in-flight count가 실제 provider queue와 다를 수 있다. gateway replica별 counter scope, cancel/timeout 뒤 decrement, long stream weight를 확인한다. stale counter가 deployment를 영구 회피하거나 overload시킬 수 있다.

cost-based routing은 단가가 싼 provider로 보내지만 semantic quality와 latency, cached token discount, region egress가 다르다. request capability와 SLO를 먼저 filter하고 cost를 secondary objective로 쓴다. 단가 table revision을 evidence에 남긴다.

random/weighted routing은 단순하지만 weight가 capacity 비율을 반영해야 한다. failure/cooldown 뒤 remaining weights를 renormalize한다. canary weight 1%가 retries/fallback 때문에 실제 attempt의 1%와 달라질 수 있다. logical selection과 physical attempt exposure를 모두 측정한다.

실험은 동일 workload arrival와 prompt/output distribution, warm state, provider quotas를 고정한다. success, p50/p99, attempt amplification, cost, semantic contract failure를 함께 본다. 최종 success만 최적화하면 retry로 provider를 태울 수 있고 provider latency만 최적화하면 budget/quality를 잃는다.

**사건 1 — 같은 deployment가 retry에서 다시 선택된다.**

model group에 A와 B가 있는데 A의 429 뒤 retry가 다시 A를 고른다. 운영자는 “retry가 고장”이라고 말하지만 여러 원인이 있다. failed deployment를 attempt-local exclusion set에 넣지 않았을 수 있고, deployment ID normalization이 달라 동일 endpoint를 다른 ID로 볼 수 있다. B가 cooldown/budget/capability filter에서 빠져 A만 남았을 수도 있다.

selection trace에 candidate before/after filters, exclusion reason, selected deployment와 previous attempt identity를 남긴다. api base가 같고 credential만 다른 deployment를 동일 endpoint로 retry하는 것이 원하는지 정책을 정한다. 429가 credential quota라면 다른 key가 유효할 수 있고 regional outage라면 같은 base는 무의미하다.

fix는 무조건 previous deployment를 금지하는 것이 아니다. pool에 하나뿐이고 retryable transient network error라면 same deployment retry가 유일한 선택일 수 있다. exception scope와 available alternatives, backoff를 반영한다. test는 두 deployment, 한 deployment, 동일 endpoint 다른 credential을 나눈다.

**사건 2 — fallback cycle가 cap을 소진한다.**

config가 A→B, B→C, C→A edge를 만들었다고 하자. A context error, B content policy, C timeout이 차례로 발생하면 visited group 없이 A로 돌아갈 수 있다. 전체 attempt cap이 있어도 사용자 latency와 cost를 낭비한다.

fallback path metadata에 visited model groups와 transition reason을 둔다. 같은 group 재방문을 금지할지 exception category가 달라지면 허용할지 policy를 정한다. graph validation에서 cycle을 reject할 수도 있다. dynamic budget fallback이 edge를 추가하면 runtime guard도 필요하다.

client error에는 “fallback exhausted”와 stable category를 주고 보호된 evidence에는 path A(context)→B(policy)→C(timeout)를 남긴다. 마지막 timeout만 보면 config cycle과 최초 context 문제를 잃는다.

**사건 3 — gateway timeout은 10초인데 60초 뒤 connection pool이 고갈된다.**

logical requests는 빠르게 timeout response를 받지만 downstream HTTP tasks가 계속 살아 socket을 잡는다. 다음 requests는 pool acquisition에서 기다리고 timeout한다. provider latency dashboard는 완료된 요청만 집계해 정상처럼 보일 수 있다.

timeout exception timestamp와 task state, HTTP stream/context close, pool checked-out count를 연결한다. cancel이 coroutine에 전달됐지만 provider adapter가 broad `except Exception`으로 잡아 retry하거나 cleanup을 건너뛰는지 본다. callback background task가 response reference를 유지하는지도 확인한다.

수정 후 timeout rate뿐 아니라 outstanding attempts, pool acquire wait, cancelled task completion latency를 본다. provider가 cancel을 지원하지 않아 server work는 계속해도 local connection을 안전히 close/reuse할 수 있는지 HTTP client contract를 따른다. 무조건 socket abort가 항상 안전/저렴하다고 단정하지 않는다.

**사건 4 — client disconnect 뒤 retries가 계속된다.**

stream client가 모바일 network 단절로 사라졌지만 router retry loop는 provider error를 받아 새 fallback을 시작한다. logical consumer가 없으므로 availability 이득이 없고 비용만 생긴다. request cancellation token이 router state까지 전파되는지 본다.

ASGI disconnect event, response generator close, router task cancel, retry loop의 cancellation handling을 잇는다. Python cancellation exception을 normal provider failure로 normalize해 retry하면 안 된다. cleanup callback은 실행하되 새 attempt는 시작하지 않는다.

다만 client disconnect 뒤 audit/logging과 provider attempt cancel/reconciliation은 계속해야 할 수 있다. request task를 통째로 버리면 reservation이 orphan된다. serving work cancel과 accounting cleanup을 별도 task/lifetime으로 설계한다.

**사건 5 — stream usage가 두 번 과금된다.**

provider final chunk에 usage가 있고 success callback도 response object usage를 처리한다. adapter가 둘을 같은 attempt event로 deduplicate하지 않으면 spend log가 두 번 올라간다. retry/fallback까지 있으면 logical ID만 dedupe key로 쓰는 것도 틀리다. 여러 attempt 실제 비용이 있기 때문이다.

spend event key는 provider/attempt request ID와 event kind/version을 포함한다. final chunk와 success callback이 같은 provider usage record라면 하나로 합친다. late billing reconciliation은 기존 estimate/actual을 update하지 새 charge를 무조건 append하지 않는다.

사용자 response의 usage field와 billing ledger가 같은 timing일 필요는 없다. client에 usage를 한 번 보냈어도 DB write retry는 idempotent해야 한다. spend batching queue 재전송과 process crash를 포함한 test를 설계한다.

**사건 6 — soft budget alert가 retry storm을 만든다.**

soft budget 초과 policy가 deployment를 cooldown하고 budget fallback group으로 보낸다. fallback group도 같은 user budget을 확인해 다시 원 group으로 가거나 429를 반환하고 client SDK가 retry한다. policy 층들이 서로의 error를 transient로 해석한다.

gateway budget rejection, provider 429, router capacity 429를 normalized source field로 나눈다. client retry guidance와 router retryability를 다르게 설정할 수 있다. budget exhausted는 시간이 지나 reset되기 전까지 retry해도 해결되지 않는다. `Retry-After`를 reset time과 연결하거나 non-retryable status contract를 사용한다.

soft budget은 alert만인지 traffic 속도 제한인지 명확히 한다. cooldown 대상이 deployment인지 key/user인지 혼동하지 않는다. user budget 때문에 shared deployment를 cooldown하면 다른 tenant capacity까지 잃는다.

**사건 7 — context fallback이 prompt를 조용히 잘라낸다.**

A가 context error를 내고 longer model B로 fallback했지만 adapter B가 provider max input 또는 gateway transform 과정에서 prompt를 truncate한다. HTTP success와 schema success는 있지만 중요한 system message가 사라진다. availability metric은 좋아지고 품질은 나빠진다.

gateway가 preflight token count와 truncation policy를 갖는지, provider가 implicit truncation하는지, response metadata가 알려 주는지 본다. fallback target의 tokenizer가 달라 A 기준 token count와 B actual count가 다를 수 있다. 문자열 길이와 token budget을 분리한다.

strict request는 truncation을 금지하고 explicit error 또는 known longer target만 허용할 수 있다. best-effort summarization/truncation은 사용자 opt-in과 observability가 필요하다. gateway가 prompt를 바꾸면 transformed payload digest와 rule을 evidence에 남긴다.

**사건 8 — provider 성공인데 표준 parser가 실패한다.**

HTTP 200과 valid provider response가 왔지만 adapter가 예상하지 못한 field/stream order 때문에 표준 object 변환에서 exception을 낸다. router가 이를 provider failure로 분류해 fallback하면 A는 과금됐고 B도 실행된다. 원인은 provider availability가 아니라 normalization layer다.

attempt outcome을 wire success와 normalization success로 나눈다. provider request ID와 raw schema version, parser exception을 보존한다. raw body에는 민감 정보가 있을 수 있어 bounded/redacted sample과 schema digest를 사용한다.

parser failure가 retryable인지 신중히 정한다. 동일 adapter/deployment retry는 같은 response shape로 다시 실패할 가능성이 높다. 다른 provider fallback은 user availability를 회복할 수 있지만 duplicate cost와 semantic drift가 있다. max cap과 error category를 별도로 둔다.

**사건 9 — health check는 정상인데 실제 request만 실패한다.**

health endpoint는 짧은 text prompt, no tools, no stream으로 성공한다. production은 긴 multimodal tools stream이라 unsupported parameter와 timeout이 난다. deployment health를 단일 boolean로 표현한 한계다.

capability별 health와 canary request class를 둔다. text, tool, image, stream, long-context를 모두 매번 실행할 필요는 없지만 deployment metadata와 sampled probes로 eligible filter를 만든다. health check 자체가 비용/quota를 쓰므로 frequency와 cheap model endpoint를 고려한다.

실제 failure가 특정 capability에만 있으면 deployment 전체 cooldown 대신 capability edge를 제외할 수 있다. source router가 그런 granularity를 지원하지 않으면 운영 limitation으로 기록한다. health green을 semantic support 증거로 사용하지 않는다.

**사건 10 — gateway cache와 fallback이 오래된 응답을 돌려준다.**

동일 logical cache key가 model alias만 포함하고 selected deployment/model revision, tool schema와 semantic options를 충분히 포함하지 않았다고 하자. A 실패 뒤 B fallback response가 cache되고 다음 A-target request에도 반환된다. HTTP는 빠르지만 model identity와 정책이 섞인다.

cache key에 normalized messages, semantic params, model/deployment identity policy, adapter/template revision을 넣는다. availability alias 수준에서 B 결과를 A와 공유할지 명시한다. user/tenant isolation과 private content도 포함한다.

stream cache replay는 chunk timing/usage/response ID를 어떻게 다루는지 별도다. 이 장의 중심은 routing이지만 gateway cache가 attempt를 생략하면서 budget/usage metric을 바꿀 수 있음을 기록한다. cache hit은 provider attempt 0개인 logical success다.

### API client retry와 gateway retry를 곱하지 않는다

gateway가 최대 3 attempts, client SDK도 최대 3 logical requests를 보낸다면 최악에 9 provider attempts가 생긴다. load balancer나 service mesh retry까지 있으면 더 곱해진다. 각 층은 자기 retry만 보고 안전하다고 생각할 수 있다.

end-to-end retry budget을 request header/metadata로 전달하거나 idempotency/logical parent ID를 연결한다. gateway response에 retryable category와 Retry-After를 정확히 주어 client가 불필요하게 재시도하지 않게 한다. mesh는 streaming/non-idempotent response retry를 피한다.

observability에서 client attempt ID와 gateway logical ID, provider attempts를 tree로 잇는다. 동일 user operation의 여러 logical requests를 묶을 correlation이 없으면 amplification root를 못 찾는다. privacy와 cardinality를 고려한 stable operation key를 쓴다.

### cancellation이 비용을 얼마나 줄였는지 정직하게 계산한다

client가 token 20에서 disconnect했고 provider가 token 25에 cancel을 반영했다고 하자. 5 token의 cancel lag가 있다. provider가 final usage를 주면 actual을 쓰고 없으면 last observed chunk/token estimate와 upper bound를 둔다. “cancel 성공으로 비용 0”이라고 하지 않는다.

cancel request가 wire에 전달된 시간, provider acknowledgement, last chunk, connection close를 기록한다. cancellation latency 분포는 streaming concurrency capacity와 비용에 직접 연결된다. provider별 cancel 지원/behavior가 다르면 routing capability로 사용할 수 있다.

timeout도 cancellation의 한 source다. client deadline, gateway policy, admin revoke와 deploy shutdown을 cancel reason으로 나눈다. provider reliability metric에서 user disconnect를 provider failure로 세지 않는다.

**한 사건의 최초 분기에서 원인을 좁힌다**

### reader-first 디깅 경로를 한 사건으로 닫는다

사용자는 “가끔 65초 뒤 성공하고 bill이 두 배”라고 신고했다. 첫 관측은 logical trace 한 건이다. root에는 A read timeout 30초, B fallback success 20초, client serialization 1초가 있어 51초여야 하는데 65초다. 14초 backoff/candidate wait를 찾는다. spend에는 A late success와 B success가 둘 다 있다.

첫 분기는 latency와 cost다. latency 원인은 retry-after/backoff와 sequential attempts, cost 원인은 A cancellation 미전파 또는 provider 지연 정산이다. 두 원인이 연결되어도 수정은 다를 수 있다. total deadline propagation과 A cancel/비용 정합화를 각각 검증한다.

source에서는 retry loop delay 계산, remaining deadline 소비, timeout task cleanup, spend event dedupe를 찾는다. 관측에서는 attempt timestamps, cancel state, provider request IDs와 usage confidence를 본다. 설정을 무작정 `num_retries=0`으로 내려 availability를 버리기 전에 인과를 닫는다.

수정 후 A timeout fixture에서 B가 남은 deadline 안에 시작/완료하고, A task/connection이 bounded time에 정리되며, late usage가 중복 없이 cost ledger에 반영되는지 본다. logical success, p99, amplification, aggregate actual cost를 함께 검증한다.

**장을 나갈 때 남는 세 원장.**

첫째는 request tree다. logical ID 아래 attempt, transition reason, deployment/provider, response commit을 둔다. 둘째는 deadline/lifetime 원장이다. admission, backoff, attempt timeout, stream commit, cancel, late completion을 둔다. 셋째는 money 원장이다. reservation, estimate, actual, unknown, release와 retry waste를 둔다.

세 원장을 서로 연결해야 한다. attempt가 tree에는 있는데 cost에 없으면 과소 회계다. cost가 있는데 attempt evidence가 없으면 중복/late event를 조사한다. timeout이 있는데 task lifetime이 없으면 orphan 가능성을 모른다. response commit 뒤 fallback edge가 있으면 stream mixing을 조사한다.

이 구조는 LiteLLM의 특정 옵션 이름이 바뀌어도 유지된다. router function, provider adapter, budget backend가 리팩터링되어도 logical/physical identity, deadline, response commitment, cost reconciliation은 gateway가 풀어야 할 근본 문제다. 독자는 source에서 이 owner들을 다시 찾을 수 있다.

**gateway와 downstream의 계약을 함께 검증한다**

**logical request와 attempt tree에서 실제 mutation을 확인한다.**

LiteLLM router의 option은 이름보다 consumer가 중요하다. routing strategy는 candidate ordering/selection state를, retries는 동일 deployment 재시도 edge를, fallbacks는 다른 model/deployment edge를, timeout은 attempt deadline을, budget limiter는 admission reservation과 reconciliation을 바꾼다.

**logical request와 attempt tree**

logical request L0가 deployment A attempt A0을 시작하고 timeout 뒤 A1 retry, 그 뒤 B0 fallback을 시작했다고 하자. tree edge에는 reason, remaining deadline, remaining retry/fallback budget, capability predicate와 commit state를 둔다. flat attempt count만으로 retry와 fallback 의미를 잃지 않는다.

logical ID는 client contract와 idempotency/billing owner다. attempt ID는 provider request, timeout/cancel, usage와 latency owner다. 동일 logical request의 attempt들이 concurrent하게 살아 있을 수 있으므로 `active_attempts`, terminal과 cancellation generation을 둔다.

**routing strategy의 실제 mutation**

least-busy, latency, usage-based, random 같은 strategy는 후보 set이 이미 capability/credential/region/health로 필터된 뒤 ordering/choice를 만든다. strategy가 unsupported tool/model을 고르지 않게 hard constraints와 soft preference를 분리한다.

parser의 문자열 option이 어느 router class/function을 선택하고 deployment stats/cache를 읽는지 걷는다. 선택 결과에 deployment ID와 reason/snapshot generation을 기록한다. option이 invalid해 default로 떨어지는지 reject하는지도 본다.

**retry와 fallback consumer**

retry classifier는 exception/status/commit state와 policy를 읽어 same deployment 또는 equivalent attempt를 만든다. fallback mapping은 model/group/error class에서 다른 candidate group으로 edge를 만든다. 모든 error를 동일 retryable로 처리하지 않는다.

validation/auth 같은 deterministic error는 retry amplification만 만든다. overload/transport error도 partial stream commit 뒤에는 transparent fallback이 안전하지 않을 수 있다. 19장의 `external_committed`를 retry predicate 입력으로 둔다.

**deadline은 attempt마다 새로 시작하지 않는다**

logical deadline 1,000ms에서 A0가 700ms를 썼다면 A1/B0가 각각 새 1,000ms를 받으면 client SLO를 넘는다. remaining logical deadline에서 routing/queue/network reserve를 빼 attempt timeout을 정한다.

예를 들어 total1000, elapsed700, response reserve50이면 usable250ms다. provider B p50이400ms라면 fallback을 시작해도 성공 가능성이 낮고 비용만 늘 수 있다. admission이 edge를 거절하는 이유를 `insufficient_remaining_deadline`으로 기록한다.

## 20.9 downstream 경계에서 gateway 책임을 판정한다

LiteLLM deployment A가 vLLM OpenAI-compatible endpoint, B가 SGLang endpoint라고 하자. gateway는 `/chat/completions` request와 표준 response를 다룬다. A와 B 내부 scheduler의 request ID, tokenization, continuous batch, KV cache와 kernel은 각각 downstream server가 소유한다. gateway request ID와 server request ID를 correlation하되 같은 ID namespace라고 가정하지 않는다.

gateway timeout이 30초이고 vLLM request는 queue에서 20초, prefill 15초를 쓴다면 첫 token 전 timeout이 난다. gateway가 HTTP connection을 닫아도 vLLM abort path까지 request가 전달되는지 확인해야 GPU work를 줄일 수 있다. SGLang도 동일하다. OpenAI-compatible transport가 cancel propagation까지 동일하다는 뜻은 아니다.

fallback A→B에서 prompt string과 sampling JSON이 같아도 chat template/tokenizer default와 option interpretation이 다를 수 있다. gateway가 request를 보존했다는 evidence와 두 servers가 같은 token IDs/model artifact를 썼다는 evidence를 분리한다. strict semantic pool이면 deployment inventory와 canary differential로 이를 관리한다.

usage가 server에서 오면 gateway는 표준 usage와 cost로 변환한다. vLLM/SGLang이 prompt/completion token을 세는 위치와 speculative/rejected token 회계를 비교해야 provider bill/compute metric과 맞출 수 있다. gateway usage는 scheduler executed token과 같은 metric이 아니다. logical billable tokens와 physical compute tokens를 나눈다.

gateway load balancing은 downstream continuous batching을 직접 보지 못할 수 있다. HTTP in-flight count가 같아도 한 server는 긴 prefill, 다른 server는 짧은 decode로 GPU 상태가 다르다. exported queue/TTFT/capacity signal을 routing에 쓰려면 sampling lag와 failure mode를 이해한다. 단순 health와 latency history만으로 충분한지 workload에서 검증한다.

### timeout·retry 예산을 downstream SLO와 합성한다

client SLO가 TTFT 10초, total 60초라고 하자. gateway route/backoff에 2초, network에 1초 headroom을 둔다면 first attempt TTFT budget은 7초다. provider의 p99 TTFT가 9초인 deployment를 선택하면 retry가 없어도 SLO를 지키기 어렵다. router가 total timeout만 알면 first-byte SLO를 놓친다.

stream이 시작되면 total generation timeout과 idle inter-chunk timeout을 구분한다. long reasoning request가 60초를 정상적으로 쓸 수 있고, provider가 20초 동안 chunk를 전혀 주지 않는 stall은 별도다. timeout option 하나를 두 의미에 재사용하면 정상 long request를 끊거나 stalled stream을 오래 붙잡는다.

retry를 허용할 TTFT headroom도 계산한다. first attempt를 7초까지 기다린 뒤 retry하면 이미 SLO를 소진한다. 빠른 connect/429 failure에는 retry가 유용하지만 slow timeout에는 fallback 성공 가능 시간이 없다. exception 발생 시간과 remaining budget이 transition eligibility다.

SLO-qualified goodput은 최종 HTTP success만 세지 않는다. 60초 안에 성공했는지, first byte와 stream cadence가 계약을 지켰는지, semantic fallback이 허용 범위인지 포함한다. retry로 성공률이 올라도 SLO를 넘긴 late success가 늘면 goodput은 낮아질 수 있다.

### budget과 rate limit을 순서대로 적용할 때 생기는 차이

auth 뒤 budget reservation, rate limit, routing 순서를 생각하자. budget을 먼저 reserve하고 rate limit에서 reject하면 즉시 release해야 한다. release가 늦으면 사용하지 않은 capacity가 pending spend로 막힌다. rate limit을 먼저 적용하면 budget DB load를 줄일 수 있지만 entity policy와 error 우선순위가 달라진다.

routing 뒤 provider별 cost estimate로 reserve하려면 deployment selection이 budget check보다 먼저 필요하다. 그러나 budget 때문에 싼 fallback을 골라야 한다면 selection과 budget이 상호 의존한다. candidate별 estimated cost로 eligibility를 filter하고 selected candidate에 atomic reservation을 할 수 있다.

concurrency limit은 cost budget과 다르다. long cheap request가 slot을 오래 잡고 short expensive request가 budget을 쓴다. RPM/TPM/concurrency/spend를 하나의 “rate limit”으로 부르면 어떤 state가 request를 막았는지 모른다. normalized rejection source와 reset/retryability를 나눈다.

### provider price revision과 reconciliation을 관리한다

request 시점 price table과 billing export 시점 price가 다를 수 있다. attempt record에 pricing model/version과 region, cached/reasoning token categories를 남긴다. 현재 table로 과거 usage를 다시 계산하면 spend가 바뀔 수 있다. provider invoice를 authority로 쓸지 gateway estimate를 쓸지 정책을 정한다.

fallback C가 더 비싼 model이면 admission reservation을 최초 A price만으로 잡아 budget을 넘을 수 있다. fallback edge별 maximum 또는 remaining budget check를 사용한다. availability를 위해 비싼 fallback을 허용할 tenant와 금지할 tenant를 나눈다.

unknown usage attempt는 나중에 invoice로 나타날 수 있다. reconciliation event가 original attempt를 찾도록 provider request ID와 time/model/key를 보존한다. 찾지 못한 cost를 silently drop하지 않고 unmatched bucket과 alert를 둔다.

### 개인정보와 observability의 균형

retry/fallback incident를 조사하려고 raw messages와 provider error body를 무제한 log하면 민감 정보가 복제된다. logical/attempt IDs, payload digest, token estimate, schema/capability, bounded redacted error를 기본으로 한다. 재현은 공개 가능한 synthetic fixture로 만든다.

stream chunk를 모두 저장하지 않아도 first/last timestamp, chunk count/bytes, finish reason, tool-call state digest로 lifetime을 볼 수 있다. content correctness가 필요한 승인된 case에만 최소 slice를 보존한다. provider headers의 request ID와 rate-limit fields도 allowlist한다.

metric label에는 user ID, endpoint URL, error message를 직접 넣지 않는다. team/model group/deployment safe ID와 normalized category를 사용하고 개별 trace로 내려간다. budget label에 email을 넣는 설정은 privacy/cardinality tradeoff를 명시한다.

### source revision을 올릴 때 회귀 감사한다

LiteLLM version을 올리면 router retry/fallback precedence, exception mapping, provider adapter와 budget schema가 바뀔 수 있다. changelog만 읽지 않고 pinned old/new source에서 handoff symbol을 비교한다. function move보다 state transition 변화가 중요하다.

고정 fixture는 429 retry, context fallback, first-byte stream failure, post-commit failure, client cancel, concurrent budget reservation, usage dedupe를 포함한다. provider mock expected wire payload와 standardized output snapshots를 version별로 비교한다. 실제 server를 실행하지 않는 source audit와 CI evidence 범위를 구분한다.

config field가 deprecated되어 새 field와 동시에 존재하면 precedence를 확인한다. old setting이 조용히 무효가 되어 retries가 기본값으로 돌아가는 위험이 있다. startup effective config와 warning을 구조화해 기록한다.

provider adapter가 supported params를 새로 drop/accept하면 semantic behavior가 바뀐다. strict mode에서 unknown field error, permissive mode에서 drop log를 검증한다. API 호환성이 HTTP schema 통과만 의미하지 않게 한다.

**최종 독자 경로를 정확히 닫는다.**

HTTP status와 provider 선택이 궁금하면 routing/attempt tree를 본다. timeout 뒤 자원이 남으면 deadline/lifetime 원장을 본다. bill과 usage가 맞지 않으면 reservation/attempt cost/reconciliation을 본다. stream이 섞이거나 끊기면 response commitment와 cancel propagation을 본다. output 의미가 달라지면 gateway를 지나 downstream tokenizer/model identity로 내려간다.

이 경로는 문제를 gateway 탓 또는 model server 탓으로 성급히 나누지 않는다. first divergence가 wire transform 전인지, provider execution인지, response normalization인지, retry/fallback state인지 증거로 결정한다. 경계를 건너는 correlation ID와 timestamp가 핵심이다.

마지막으로 설정 변경 기록에는 이전/새 effective value, 영향을 받는 request class, 예상 attempt tree와 deadline, cost upper bound, rollback condition을 쓴다. retry를 2에서 4로 올린다면 provider load와 client p99, budget reserve가 어떻게 바뀌는지 먼저 계산한다. fallback target 추가라면 semantic capability와 data policy를 검증한다.

책의 다른 장과 연결하면 gateway는 API ingress 위의 제어면이고 vLLM/SGLang scheduler는 GPU work의 제어면이다. 둘 다 요청을 고르고 실패를 처리하지만 상태와 증거가 다르다. gateway의 deployment attempt와 engine의 sequence/request state를 같은 “retry”나 “queue”로 뭉개지 않는다.

이 경계를 닫을 때는 성공률보다 사용 가능한 성공을 판정한다. logical request 하나가 deadline,
response commitment와 비용 reconciliation까지 통과해야 분자에 넣을 수 있다.

### 20.9.1 마지막 사례: 성공률을 올린 변경이 goodput을 낮춘 이유

변경 전에는 logical requests 1000건 중 930건이 20초 안에 성공하고 70건이 실패했다. 변경 후 retry와 broad fallback을 늘려 990건이 최종 성공했다. 표면 success rate는 93%에서 99%로 올랐다. 그러나 100건은 client SLO 60초를 넘겨 성공했고, semantic strict cohort 20건은 다른 provider fallback으로 schema quality gate를 실패했다. SLO와 semantic contract를 모두 통과한 goodput은 870건이다.

physical attempts도 1000에서 1800으로 늘었다. provider A wasted cost 30, B/C actual cost 100이 발생해 cost per SLO-qualified request가 올랐다. retry가 availability를 올린 사실과 serving 효율을 낮춘 사실이 동시에 참이다. 어떤 metric을 최적화했는지 말하지 않고 “개선”이라고 쓰면 안 된다.

원인을 request tree에서 찾는다. rate limit 429는 빠르게 실패해 B retry가 SLO 안에 들어왔지만 read timeout 30초 뒤 fallback은 거의 모두 늦었다. content policy fallback은 HTTP success였지만 strict schema cohort에서 금지된 semantic edge였다. error category와 elapsed/remaining deadline, request class를 transition eligibility에 넣는다.

새 policy는 빠른 transient failure만 retry하고 slow timeout은 남은 budget과 predicted latency가 충분할 때만 fallback한다. strict cohort는 semantic-equivalent pool만 허용한다. best-effort cohort는 broad fallback을 허용하되 response metadata와 cost ceiling을 둔다. 한 전역 숫자 대신 이유 있는 edge가 생긴다.

변경을 검증할 때 success rate, SLO-qualified success, attempts/logical, provider load, aggregate cost, schema/tool contract failure를 함께 본다. p50만 좋아지고 p99와 cost가 악화되는지 cohort별로 본다. retry cap이 effective config에 적용되었고 cycle/cooldown이 예상대로 작동하는지 trace로 확인한다.

이 사례는 gateway optimization의 핵심을 보여 준다. 실패를 숨기는 것이 목적이 아니라 실패의 위치와 비용을 제어하면서 사용자 계약을 최대한 지키는 것이다. retry와 fallback은 오류를 없애지 않고 어느 failure domain에서 누가 비용을 부담할지 바꾼다.

이 사건을 source에서 다시 확인할 때는 route 선택, physical attempt 실행, 표준 response 조립의
세 층을 한 함수처럼 합치지 않는다.

router loop의 `try/except`만 보면 retry가 단순해 보인다. 그러나 첫째 exception mapper가 provider error를 어떤 class/status로 바꾸는지가 앞에 있다. 둘째 cooldown/candidate selector가 다음 deployment를 고르는 shared state가 옆에 있다. 셋째 callback과 spend logger가 attempt outcome을 사후 처리하는 수명이 뒤에 있다.

함수 한 개를 인용할 때 이 세 handoff를 함께 설명한다. `async_function_with_retries`는 반복 owner지만 provider adapter cancel과 proxy budget reservation 전체를 소유하지 않는다. `async_function_with_fallbacks`는 edge를 선택하지만 downstream model semantics를 검증하지 않는다. 책임 범위를 좁혀야 source 인용이 설계 설명이 된다.

동기 wrapper도 단순히 같은 함수의 blocking 버전이라고 넘기지 않는다. async loop를 별도 thread/event loop에서 실행할 때 cancellation, timeout, context propagation이 달라질 수 있다. sync client disconnect를 어떻게 알 수 있는지, future timeout 뒤 coroutine cleanup이 되는지 본다. source-only 단계에서는 가능한 path와 test를 기록하고 실제 효과를 주장하지 않는다.

global default와 router instance, per-request override가 같은 이름을 쓰면 effective config가 추적하기 어렵다. startup snapshot과 request trace에 resolved value/source를 남기는 이유다. 값 2만 기록하지 않고 `request.num_retries=2 overrides router default 1`처럼 provenance를 둔다.

여기서 경계도 분명히 한다. 이 장은 gateway가 소유하지 않는 tokenizer·model forward·KV와
sampling correctness를 대신 설명하지 않는다.

provider SDK의 모든 parameter 표와 LiteLLM management endpoint 전수 목록은 본문에서 제외했다. 독자가 요청 수명을 이해하는 데 필요한 owner만 연결했다. 전수 옵션은 source note/부록에서 version별로 관리하는 편이 낫다. 본문에 나열하면 왜와 failure boundary가 묻힌다.

LiteLLM이 호출하는 vLLM/SGLang/CUDA kernel의 내부도 이 장에서 반복하지 않았다. gateway가 보는 것은 HTTP/stream/usage와 cancel correlation이다. engine scheduler, KV cache와 kernels는 해당 편의 owner다. 다만 timeout/cancel이 그 상태로 전달되는 handoff는 남겼다.

model quality benchmark와 provider 추천도 하지 않았다. fallback semantic pool을 설계하려면 별도 evaluation evidence가 필요하다. 동일 API schema와 model alias는 quality equivalence의 증거가 아니다. 이 경계를 명시하는 것이 부실함이 아니라 정확성이다.

**20.10의 중간 회고.**

처음에는 user request 하나만 보였다. 이제 그 뒤에 logical identity, attempt tree, deployment selection, provider wire request, exception transition, response commitment, deadline/cancel, reservation/actual cost가 보인다. 이 좌표가 있으면 “gateway가 느리다”를 backoff, pool wait, provider attempt, stream write로 나눌 수 있다.

동시에 gateway의 한계도 보인다. 요청과 응답 envelope를 정규화하고 availability policy를 적용할 수 있지만, 서로 다른 model의 token IDs와 hidden/logits, safety와 tool semantics를 동일하게 만들지는 못한다. 그 영역은 downstream artifact와 serving engine evidence가 필요하다.

독자는 이제 LiteLLM option을 볼 때 무엇이 바뀌는지 묻는다. candidate set인가, attempt cap인가, exception edge인가, deadline인가, budget reservation인가, response commit인가. 그리고 효과를 success 하나가 아니라 SLO goodput, amplification, cost, semantic contract로 검증한다.

이렇게 읽으면 LiteLLM은 책 전체에서 잠깐 스쳐 가는 제품 목록이 아니다. API client와 여러 inference server 사이에서 failure와 돈, 시간, streaming ownership을 조정하는 얇지만 중요한 층이다. 얇다는 말은 책임이 적다는 뜻이 아니라 model tensor를 직접 소유하지 않는다는 뜻이다.

장애가 다시 발생하면 첫 질문은 “어느 provider가 실패했나”가 아니다. client response가 commit되기 전인지, logical request의 몇 번째 physical attempt인지, 남은 deadline과 budget이 얼마였는지부터 묻는다. 이 세 값이 있어야 retry가 합리적이었는지와 fallback이 허용된 edge였는지 판단할 수 있다.

두 번째 질문은 실패 attempt가 정말 끝났는지다. timeout response, coroutine cancel, HTTP connection close, provider generation 중단, usage 정산은 서로 다른 event다. 하나의 `TimeoutError`로 다섯 event가 자동 완료되었다고 보지 않는다. task와 비용 수명이 닫힐 때까지 attempt ledger를 유지한다.

세 번째 질문은 최종 성공이 같은 사용자 계약을 지켰는지다. schema와 tool capability, data region, model/tokenizer identity, SLO, cost ceiling을 본다. HTTP 200만으로 fallback success를 정의하면 availability 숫자는 좋아져도 사용 가능한 응답은 줄 수 있다.

이 세 질문을 trace와 source handoff로 답하고, 변경 후 경쟁 가설을 기각할 fixture까지 통과시키면 gateway incident가 닫힌다. 그렇지 않으면 retry 횟수나 timeout 숫자를 바꾼 것은 원인 수정이 아니라 다음 failure shape를 바꾼 것뿐이다.

최종 handoff에서는 logical request ID와 attempt tree, response commit, 비용 confidence를 운영 기록에 남긴다. downstream server에는 provider request ID와 cancel 상관관계를 전달하고, client에는 안전한 logical ID와 명확한 retry guidance를 돌려준다. 각 당사자가 자기 수명에서 같은 사건을 찾을 수 있어야 gateway가 진정한 연결 계층이 된다.

21장은 이 downstream 가운데 vLLM 하나를 골라 provider request ID가 `AsyncLLM`과 engine-core command,
output stream으로 바뀌는 과정을 따라간다. gateway attempt가 끝났다는 사실과 vLLM의 request state가
실제로 정리됐다는 사실을 같은 사건으로 오해하지 않도록, 이 장의 cancel 상관관계를 그대로 넘긴다.

그 연결이 관측과 책임을 함께 완성한다.

## 20.10 같은 attempt 기록으로 metrics·regression·rollback을 검증한다

metric에는 logical requests, physical attempts, retry/fallback edge, commit-state retry rejection, attempt deadline, cancel pending age, provider late usage, reserved/reported/released budget, attempt amplification과 cost/logical을 둔다. provider request ID와 user key는 trace에 둔다.

**numeric regression matrix**

deadline은 error at 0/400/900ms, total1000ms에서 remaining edge를 계산한다. budget은 limit0.04에서 A0.02+B0.03 pending/cancel-confirmed를 비교한다. concurrency는 두 logical requests가 remaining0.03을 동시에 reserve하려는 race를 넣는다.

stream은 error before first chunk, buffered first chunk, after external commit을 나눈다. tool/idempotency는 side effect before response loss와 duplicate attempt를 포함한다. retryable/nonretryable status와 capability mismatch fallback을 넣는다.

각 cell은 attempt tree, selected route, deadline, commit state, reservation transitions, final response owner, provider charges, cleanup terminal을 판정한다. HTTP success 하나로 PASS하지 않는다.

**SLO goodput와 amplification**

success rate가 올라가도 p99가 deadline을 넘거나 duplicate cost가 크면 usable goodput은 나빠질 수 있다. `logical successes within SLO and semantic contract / logical requests`를 본다. attempts/success, cost/success, wasted tokens와 orphan duration을 guardrail로 둔다.

retry storm은 provider failure가 attempt rate를 배수로 늘려 overload를 악화한다. retry budget, backoff/jitter, global concurrency/circuit breaker가 같은 state를 보도록 한다. 각 logical request cap만으로 fleet amplification을 막지 못할 수 있다.

**rollback generation**

routing policy, fallback map, retry classifier, price/budget model과 client commit contract를 version한다. in-flight attempt는 시작 generation을 유지하거나 안전하게 drain한다. mid-request policy swap으로 budget/route semantics가 바뀌지 않게 한다.

known-good config rollback은 provider tasks/cancel callbacks와 budget reservations를 잊지 않는다. old generation pending attempts가 settlement될 때까지 ledger consumer를 유지한다. config 되돌림이 external state를 지우지 않는다.

**종료 terminal**

routing terminal은 capability를 만족하는 candidate 선택 또는 명시적 no-route다. attempt terminal은 provider task/connection/generation과 usage confidence가 닫힌 상태다. response terminal은 exactly-one external result와 commit state다. budget terminal은 모든 attempt reservation이 report/estimate/release로 reconcile된 상태다. idempotency terminal은 application side effect key와 logical request가 연결된 상태다.

incident 종료 문장은 “A timeout cancel intent에서 $0.02 reserve를 release해 B $0.03 fallback이 $0.04 ceiling을 통과했으나 A late usage로 $0.05가 됐다. pending-cancel reservation과 atomic check-reserve를 도입하고 commit-aware fallback, two-request race와 delayed usage fixture를 통과했다”처럼 쓴다.

이제 LiteLLM 장은 option catalogue가 아니다. 독자는 config 한 값이 candidate/attempt/deadline/budget/commit state 중 무엇을 바꾸는지, 성공이 왜 중복 비용과 orphan을 숨길 수 있는지, 어떤 source consumer와 ledger를 확인할지 알 수 있다. 21장에는 선택된 vLLM attempt ID와 cancel/commit/budget 상관관계를 넘긴다.

**Retry 비용을 손으로 계산한다.**

logical request 10,000개, base attempt success 90%, 실패 10% 중 절반이 retryable이라고 하자. retry 한 번의 success가 60%라면 추가 attempts는 500개, 추가 successes는 300개다. logical success는 9,300, attempt 수는 10,500이다. success rate는 90%→93%지만 attempt amplification은 1.05다.

평균 base cost $0.01, retry cost도 $0.01이면 추가 $5, 성공 한 건당 전체 cost는 $105/9300≈$0.01129다. retry 전 $100/9000≈$0.01111보다 높다. latency deadline 내 성공이 retry successes 중 150개뿐이면 SLO goodput은 9,150이고 비용 효율은 더 나쁘다. 성공률 하나로 정책을 승인하지 않는다.

provider outage에서 retryable 실패가 50%로 늘면 같은 one-retry 정책은 5,000 extra attempts를 만든다. failing provider capacity를 더 압박한다. circuit breaker와 global retry budget이 없으면 각 request의 합리적 retry가 fleet에서는 폭주가 된다. 시간 window별 attempt amplification을 alert한다.

**Fallback semantic compatibility를 평가한다.**

model A에서 B로 fallback할 때 API schema가 같아도 context length, tool calling, JSON schema, multimodal, region, data retention, tokenizer/template와 output quality가 다를 수 있다. candidate filter에 hard contract를 넣고 routing strategy는 그 뒤에 적용한다.

fallback matrix 열은 capability, max context, tool/response format, streaming frame, logprobs/usage, region/compliance, cost/SLO, model semantic class다. unsupported field를 silently drop해 HTTP200을 만드는 것을 success로 보지 않는다. transform이 필요하면 effective request와 lost semantics를 기록한다.

긴 prompt가 B context를 넘으면 retry로 성공할 수 없다. gateway가 truncate하면 사용자 계약이 바뀐다. explicit policy/consent 없이 fallback transform을 availability로 포장하지 않는다. rejection reason을 bounded하게 노출한다.

**Provider rate limit과 local concurrency를 구분한다.**

429가 provider quota인지 deployment concurrency인지, retry-after가 있는지 본다. 같은 credential로 다른 deployment를 골라도 quota를 공유할 수 있다. candidate별 limit과 credential/account shared limit을 모델링한다. fallback이 같은 quota domain이면 폭주를 옮기지 못한다.

local queue가 이미 deadline을 소모했다면 provider retry-after를 기다릴 수 있는지 remaining deadline으로 판정한다. backoff/jitter timer도 active attempt/reservation lifetime이다. client cancel이 timer를 취소하고 reservation을 reconcile하는지 본다.

**Timeout task lifecycle을 닫는다.**

`asyncio.wait_for`나 유사 wrapper가 timeout exception을 냈다고 underlying HTTP/provider 작업이 끝났다는 보장은 library contract와 cancellation handling에 달렸다. shield/background callback이 있으면 late result/usage가 올 수 있다. task reference와 provider request ID를 ledger에 남긴다.

timeout 전 DNS/connect/write/read/first-byte/total 단계 중 어디였는지 알면 retry risk가 달라진다. request body가 provider에 도달하지 않았다면 side effect 위험이 낮고, response read timeout이면 provider compute/charge 가능성이 높다. transport phase를 error class에 포함한다.

cancel API가 있으면 best-effort 호출과 acknowledgment를 구분한다. cancel request 성공 status가 generation stopped/usage final을 보장하는지 provider contract를 확인한다. unknown provider는 settlement window까지 reserve를 유지한다.

**Streaming buffer와 fallback window를 계산한다.**

gateway가 first N bytes 또는 first frame을 buffer해 provider early failure 시 fallback할 수 있다. buffer 100ms는 client TTFT를 최소100ms 늦출 수 있고, 그 안에 provider가 content/tool delta를 얼마나 생성하는지에 따라 memory/비용이 생긴다.

buffer가 complete semantic frame을 보장해야 한다. UTF-8 bytes 중간, incomplete tool JSON, usage/logprob cursor를 버리고 다른 provider로 갈 때 client에게 아무 것도 commit하지 않았더라도 provider A cost는 남는다. buffered attempt ledger를 유지한다.

buffer timeout 뒤 first frame을 commit하면 fallback gate를 닫는다. 동시에 provider error가 오면 compare-and-set으로 commit/error owner를 정한다. double response를 막는다. 이 state는 19장의 formatter와 LiteLLM wrapper 양쪽 source를 연결한다.

**Budget estimate 오차와 가격 revision.**

admission estimate는 prompt tokens와 max output, model price를 사용한다. actual usage는 더 작거나 provider-specific cached/reasoning token 가격이 다를 수 있다. estimate price revision과 provider report currency/unit을 기록한다. price table update가 in-flight reservation을 소급 변경하지 않게 generation을 둔다.

hard budget은 worst-case reserve로 안전하지만 utilization을 낮출 수 있다. expected reserve는 overspend 위험이 있다. 정책 선택과 confidence를 문서화한다. max output가 매우 크면 partial reserve/streaming cutoff를 지원할 수 있지만 external commit 뒤 budget exhaustion terminal semantics를 정의해야 한다.

tenant budget과 model budget이 중첩되면 한 attempt가 여러 ledger에 atomic reservation을 해야 한다. 일부 key만 reserve되고 다른 key가 실패하면 rollback한다. distributed transaction/ordered lock/atomic script 등 source의 실제 보장을 확인한다.

**Idempotency record의 lifetime.**

idempotency key record에는 request digest, logical state, committed response metadata, attempt IDs, usage/budget와 유효시간을 둔다. 같은 key에 다른 request body가 오면 conflict로 거절한다. 유효시간이 provider late usage/side-effect window보다 짧으면 duplicate를 다시 허용할 수 있다.

streaming 전체 bytes를 저장하지 않는다면 completed response replay 범위를 명시한다. “idempotent”가 provider submit dedup인지 client response replay인지 tool side-effect dedup인지 축을 나눈다. 하나의 key가 모든 시스템을 자동으로 exactly-once로 만들지 않는다.

**Metrics cardinality와 incident trace.**

Prometheus에는 router strategy, error class, retry/fallback edge kind, commit state, attempt ordinal bucket, deadline outcome, budget state와 provider group 같은 bounded label을 둔다. deployment/request/idempotency key는 trace에 둔다. 동적 model string도 cardinality 정책을 적용한다.

histogram은 logical latency와 attempt latency, cancel pending, settlement delay, reserved/actual ratio를 분리한다. counter reset/multiprocess 합산을 고려한다. cost는 currency/model price revision을 명시하고 aggregate 단위를 혼합하지 않는다.

incident trace는 selection snapshot, attempt tree, transport phase/error, commit event, provider cancel/usage, budget transitions를 시간순으로 가진다. raw credentials/prompt는 저장하지 않는다. logical/attempt correlation이 끊기면 duplicate cost 원인을 찾을 수 없다.

**회귀 종료와 reader checklist.**

독자는 옵션 하나를 골라 parser, validation, consumer, state mutation, next attempt, metric effect를 추적한다. `num_retries=2`가 실제로 어떤 exception에서 몇 physical attempts를 허용하는지, nested SDK/provider retry와 합쳐 최대 amplification이 얼마인지 계산한다.

예를 들어 gateway 2 retries와 provider SDK 2 retries가 곱으로 구성되면 최대 provider calls가 9일 수 있다. 각 layer semantics에 따라 다르므로 source에서 loop nesting을 확인한다. 단순 합 5로 보고 capacity를 설계하지 않는다. fallback group 2개까지 있으면 상한이 더 커질 수 있다.

incident fix 뒤에는 success뿐 아니라 attempts/logical, cost/logical, deadline goodput, late usage, pending cancel, budget invariant와 duplicate side effect를 본다. 두 관측 window와 outage/control cohort를 통과한다. performance 회복과 accounting settlement가 다른 시각에 끝날 수 있다.

이 모든 디테일의 목적은 gateway를 거대한 제품 설명으로 만드는 것이 아니다. router가 실패를 다른 attempt로 바꾸는 순간 시간, 돈, external commit과 side-effect 책임이 어떻게 이동하는지 독자가 판단하게 하는 것이다. 그 판단이 가능하면 다음 downstream engine의 실제 cleanup까지 추적할 준비가 된다.

**처음 보는 LiteLLM incident를 한 시간 안에 분류한다.**

첫 10분에는 logical request ID와 provider attempt IDs를 모은다. client error/success, external commit 시각, selected deployment와 routing generation을 적는다. 한 provider request만 보고 전체 logical lifecycle을 추정하지 않는다.

다음 10분에는 attempt tree와 각 edge reason을 복원한다. retry인지 fallback인지, 누가 exception을 분류했는지, remaining deadline/retry cap은 얼마였는지 본다. SDK 내부 retry가 gateway trace에 숨었는지도 provider request count로 확인한다.

세 번째 10분에는 provider tasks terminal을 확인한다. local timeout/cancel intent, connection close, provider cancel acknowledgment, late result/usage를 나눈다. final client success가 old attempts cleanup을 증명하지 않는다.

네 번째 10분에는 budget transitions를 시간순으로 둔다. reserve, release, report/estimate reconcile을 attempt별로 더한다. logical ceiling과 tenant/model shared ceiling을 확인한다. concurrent reservation race가 있었는지 atomic store operation을 본다.

다섯 번째 10분에는 semantic compatibility를 확인한다. fallback provider가 tools/schema/context/region/stream usage를 보존했는지 effective request와 response를 비교한다. HTTP200을 usable success로 자동 분류하지 않는다.

마지막 10분에는 first divergence와 owner를 쓴다. selection, classifier, timeout lifecycle, commit guard, budget ledger, provider cancel 중 최초 잘못된 transition을 지정한다. mitigation, fix, passing neighbor와 regression을 붙인다.

**Concurrency race를 더 작은 수치로 재현한다.**

tenant remaining budget $0.04에서 logical L1과 L2가 각각 $0.03 reserve를 시도한다. 둘 다 read 단계에서 0.04를 보고 check를 통과한 뒤 write하면 reserved total0.06이다. atomic compare-and-reserve라면 하나만 성공하고 다른 하나는 budget rejection 또는 cheaper route를 선택한다.

release race도 있다. attempt A가 cancel callback과 late usage callback을 동시에 받는다. cancel handler가 reserve0.02를 release하고 usage handler가 actual0.018을 charge한다면 최종 net이 맞을 수 있지만 두 callback 순서/중복에 안전해야 한다. idempotent transition ID와 compare state를 둔다.

usage callback이 두 번 오면 attempt usage를 overwrite/reconcile해야지 두 번 더하지 않는다. provider event ID나 final flag, monotonic cumulative semantics를 확인한다. incremental usage면 sequence를, cumulative면 max/final snapshot을 사용한다.

**Hedging과 retry를 구분한다.**

latency hedging은 A가 아직 running일 때 B를 의도적으로 병렬 시작하고 first acceptable result를 선택한다. retry는 실패/timeout 뒤 새 attempt다. hedging은 cost/reservation과 cancel-loser를 처음부터 두 attempt로 계획한다. retry metric에 숨기지 않는다.

A p95가800ms이고 SLO1000ms, hedge delay300ms, B latency400ms라면 B가700ms에 끝나 tail을 줄일 수 있다. 그러나 두 provider cost가 겹치고 A cancel이 늦으면 amplification이 커진다. quality/semantic arbitration과 external commit owner가 필요하다.

hard budget이 한 attempt만 허용하면 hedge를 시작할 수 없다. expected-cost policy를 쓰면 overspend probability를 문서화한다. tool/non-idempotent request는 hedging에서 제외할 수 있다. option이 존재한다고 모든 request에 안전하지 않다.

**Circuit breaker와 retry budget의 상호작용.**

circuit breaker는 deployment health state를 candidate filter에 반영하고 retry budget은 fleet amplification을 제한한다. breaker가 열리기 전 burst에서 retry storm이 생길 수 있어 rolling error/latency와 half-open probes를 설계한다. breaker state가 모든 process/router instance에 일관되는지 본다.

half-open probe는 일반 user request를 여러 개 동시에 보내지 않게 lease/limit를 둔다. probe 성공이 tool/multimodal 등 모든 capability 회복을 뜻하지 않는다. capability별 health 또는 conservative reopen policy를 둔다.

breaker open으로 fallback traffic이 B에 몰려 B도 overload될 수 있다. destination capacity와 global admission을 함께 본다. route success만 최적화하면 cascading failure를 만든다.

**Priority와 fairness가 budget/routing에 미치는 영향.**

premium tenant가 더 큰 retry budget이나 preferred deployment를 가질 수 있다. policy는 명시적이고 budget key/queue priority와 일관돼야 한다. starvation과 noisy-neighbor를 metric으로 본다. credential pool selection이 tenant isolation을 깨지 않게 한다.

budget rejection이 반복되는 tenant를 fallback cheaper model로 보낼 때 semantic contract/consent를 확인한다. 비용 최적화가 품질/region 요구를 위반하지 않는다. rejection reason과 offered alternative를 client에 명확히 한다.

**Cache와 retry identity.**

gateway response cache가 logical request를 hit하면 provider attempt가 없을 수 있다. cache key가 tools, prompt/template/model semantics와 policy를 포함하는지 별 장에서 다루지만, budget ledger는 cache hit 비용/usage 정책을 구분해야 한다. provider usage를 꾸며내지 않는다.

failed/partial response를 cache하지 않는다. stream external commit 뒤 error인 response를 성공 cache로 저장하면 이후 client에 partial을 완전 응답처럼 돌려준다. terminal state와 completeness가 admission predicate다.

retry attempt의 provider cache/prefix hit는 compute cost를 줄일 수 있어도 provider billed usage contract는 다를 수 있다. local estimate와 reported charge를 reconcile한다. cache hit율을 cost proof로 쓰지 않는다.

**Security와 data governance fallback.**

deployment 후보는 credential scope, data region, retention/BAA와 model capability를 hard constraint로 필터한다. primary failure 때 금지 region provider로 fallback하면 availability는 올라가도 계약 위반이다. routing source에서 filter가 strategy 이전에 적용되는지 확인한다.

trace에는 credential secret 대신 deployment/region/policy generation을 둔다. prompt를 여러 provider에 보내는 retry/hedging은 data exposure surface를 늘린다. policy와 audit를 cost/latency와 함께 보고한다.

tool schema나 user data를 error log에 그대로 남기지 않는다. attempt correlation은 pseudonymous IDs와 digest/size로 가능하다. incident vault 접근을 통제한다.

**Release diff와 source freshness.**

LiteLLM upstream revision이 바뀌면 router option default, retry classifier, fallback wrapper, timeout behavior와 budget callbacks를 다시 확인한다. source anchor가 이동했더라도 semantic consumer를 찾는다. changelog만으로 state 호환성을 선언하지 않는다.

새 exception class가 retryable set에 들어가면 amplification과 semantic safety가 바뀐다. streaming wrapper buffering/commit guard 변경은 partial fallback을 바꾼다. budget price update는 in-flight generation과 reconciliation을 바꾼다. diff를 regression matrix에 연결한다.

**최종 인계 문장.**

“Logical L0는 policy G3에서 A0을 선택했고 700ms read timeout 후 external commit 없이 cancel intent가 났다. A0 reserve $0.02는 cancel confirmation 전 유지됐으므로 remaining ceiling $0.02에서 B0 $0.03 fallback을 거절했다. A late usage $0.018이 reconcile됐고 cleanup terminal 뒤 reserve가 닫혔다.”처럼 쓴다.

또는 “첫 tool delta가 client에 commit된 뒤 provider error가 발생했으므로 transparent fallback을 금지하고 partial terminal error를 반환했다. application idempotency key로 이미 실행된 tool side effect를 조회했다.”처럼 commit과 side effect를 연결한다.

이 문장이 있으면 다음 vLLM 담당자는 gateway timeout이 downstream abort로 도달했는지 provider request ID에서 시작할 수 있다. gateway success/error와 engine cleanup을 혼합하지 않는다. 20장의 책임은 attempt를 정확히 만들고 닫으며 비용과 external semantics를 보존하는 것이다.

**운영자가 자주 내리는 잘못된 결론을 반증한다.**

“최종 응답이 성공했으니 retry는 성공적이었다”는 결론은 deadline, semantics, duplicate cost와 orphan을 보지 않는다. final success를 logical SLO goodput과 cost ledger로 다시 평가한다. 늦은 성공이나 capability-lost fallback은 별 result다.

“timeout이 났으니 provider는 과금하지 않는다”도 틀릴 수 있다. request가 provider에 도달했고 generation이 진행됐다면 local timeout 뒤 usage가 올 수 있다. transport phase와 cancel acknowledgment, settlement를 본다.

“idempotency key가 있으니 exactly-once다”도 범위를 묻는다. gateway response dedup, provider submit, billing, tool side effect 중 어느 층이 key를 소비하는가. consumer가 없는 header/field는 효과가 없다.

“budget limiter가 있으니 overspend가 없다”는 atomicity와 distributed scope를 확인해야 한다. process-local limiter, delayed usage, cancel release race, price revision이 ceiling을 깨뜨릴 수 있다. reservation invariant를 수치로 검산한다.

“fallback model이 API-compatible하니 안전하다”는 prompt/template/tool/region/quality 차이를 무시한다. hard capability filter와 semantic fixture를 본다. transform으로 field를 drop했다면 effective request와 사용자 consent를 기록한다.

**Provider SDK 내부 retry를 source에서 찾는다.**

LiteLLM router가 `num_retries=0`이어도 provider SDK/HTTP adapter가 connect retry를 할 수 있다. gateway logical attempt 하나가 여러 network sends를 만들 수 있다. SDK config와 transport adapter, status retry를 확인하고 attempt ledger에 network subattempt를 필요하면 둔다.

반대로 LiteLLM과 SDK 양쪽 retry가 중첩되면 maximum call 상한과 deadline을 계산한다. 각 loop가 새 timeout을 부여하는지 remaining deadline을 공유하는지 본다. capacity planning과 cost estimate에 worst-case/observed amplification을 넣는다.

**Fallback graph cycle과 cap을 검증한다.**

fallback mapping A→B, B→A 같은 cycle이 config merge로 생길 수 있다. visited model/deployment set과 global attempt cap이 cycle을 끊는지 source를 확인한다. error type별 fallback list가 중첩될 때 같은 candidate를 반복하지 않게 한다.

graph fixture는 A auth error, B overload, C success와 cycle A↔B를 넣는다. expected path와 stop reason을 판정한다. 단순 recursion depth error에 의존하지 않는다. capability filter로 후보가 0일 때 명시적 no-route terminal을 낸다.

**Backoff가 deadline과 queue를 소비하는 방식.**

exponential backoff가 100,200,400ms이고 logical deadline 500ms라면 세 번째 sleep은 성공 시도 시간을 남기지 않는다. sleep 전에 remaining deadline과 minimum attempt budget을 확인한다. jitter는 herd를 줄이지만 SLO를 보장하지 않는다.

sleeping retries도 logical request concurrency와 budget reservation을 점유할 수 있다. queue/admission에서 active로 세는지 정책을 정한다. client cancel이 timer를 즉시 깨우고 cleanup하는지 본다. timer leak은 메모리/attempt residue를 만든다.

**Budget fail-open/fail-closed incident.**

external budget store가 timeout됐을 때 fail-open하면 availability는 유지하지만 ceiling을 보장하지 못한다. fail-closed면 비용은 지키지만 요청을 거절한다. tenant tier와 emergency policy, max local safety reserve를 문서화한다. store outage를 zero remaining budget이나 unlimited로 조용히 해석하지 않는다.

degraded mode metric과 response reason을 둔다. store recovery 후 local reservations/usage를 reconcile한다. split brain에서 두 router가 각각 reserve한 금액을 합친다. budget correctness terminal은 store availability와 별개다.

**가격과 token estimate의 불확실성.**

chat template/tool schema 때문에 gateway tokenizer estimate가 provider usage와 다를 수 있다. reasoning/cache/audio token처럼 별 단가가 있을 수 있다. estimate error distribution과 safety margin을 model/revision별로 둔다. unknown model price에서 fail-open/closed를 명시한다.

actual usage가 estimate보다 작으면 unused reserve를 release하고, 크면 overage를 ledger에 남긴다. estimate를 actual로 덮어써 원래 admission 판단 근거를 잃지 않는다. 두 값을 보존해야 margin을 조정할 수 있다.

**최종 reader artifact의 배열.**

책에서는 먼저 logical/attempt 두 원장을 설명하고, 그 다음 route/retry/fallback state를 넣는다. deadline 계산과 external commit을 이해한 뒤 budget/idempotency race를 보여 준다. option reference는 뒤에 둔다. 이 순서가 dry한 config 목록을 사건 해결법으로 바꾼다.

표는 attempt tree, time/budget ledger, compatibility matrix, terminal checklist 네 개로 제한한다. 중복되는 option inventory를 만들지 않는다. source link는 각 state mutation과 바로 붙인다. 독자는 링크를 열지 않아도 인과를 이해하고, 열면 consumer를 확인할 수 있어야 한다.

20장의 최종 완료 기준은 router를 “여러 모델 중 하나를 고르는 함수”로 설명하지 않는 것이다. 하나의 실패가 새 physical attempt, 새 deadline/cost exposure와 commit 위험을 만드는 과정을 계산하고, 모든 attempt와 reservation이 terminal로 닫혔는지 증명할 수 있어야 한다.

변경 review에서는 option default 하나의 파급을 끝까지 쓴다. retry 1→2는 단순 한 번 증가가 아니라 worst-case attempts, backoff/deadline, reservations, provider load와 duplicate exposure를 바꾼다. fallback list 추가는 capability/data region과 price table, cache/idempotency namespace를 바꾼다.

timeout 30s→10s는 빠른 실패만 만드는 것이 아니라 late provider tasks와 cancel/usage settlement 비율을 늘릴 수 있다. remaining deadline과 provider latency cohort를 보고 정한다. streaming first-frame timeout과 total timeout을 같은 값으로 바꾸지 않는다.

router strategy 변경은 historical stats warm-up과 feedback loop를 고려한다. latency-based route가 실패/timeout samples를 어떻게 포함하는지, cold deployment가 unfair하게 선택/배제되는지 본다. selection metric generation과 observation window를 trace한다.

budget limit 변경은 in-flight reservations를 새 limit에 어떻게 평가하는지 정한다. limit을 낮췄을 때 이미 reserved가 초과하면 새 admission만 막을지 running attempts도 cancel할지 정책이 필요하다. running cancel은 partial commit/side-effect 위험을 갖는다.

회귀 canary는 단일 provider 정상만 돌리지 않는다. primary early failure, late failure, commit 후 stream error, slow/cancel, budget boundary, concurrent reservation, fallback capability mismatch와 late usage를 포함한다. provider를 실제 호출하지 않는 static/fake transport fixture도 state machine을 검증할 수 있다.

소스 증거는 config schema, router selection, retry/fallback wrapper, timeout task, budget limiter와 callback의 pinned span을 갖는다. runtime 선택과 provider behavior는 후속 trace다. source에 branch가 있다고 실제 실행됐다고 쓰지 않는다.

운영 종료는 client error rate가 회복된 순간이 아니다. pending cancel/reservation과 delayed usage가 settlement window 안에서 닫히고 attempt amplification/cost가 baseline envelope로 돌아오며 semantic fallback rejection이 의도대로 작동해야 한다. telemetry coverage도 별 terminal이다.

마지막으로 다음 장에 넘기는 provider attempt ID는 gateway 내부 숫자만이 아니다. downstream request ID, submit 시각, deadline/cancel generation, external commit과 budget state를 함께 전달한다. 그래야 vLLM ingress에서 queue/abort/cleanup을 같은 사건으로 찾을 수 있다.

최종 독자 연습은 config 한 줄을 mutation chain으로 쓰는 것이다. 예를 들어 `num_retries=2`를 parser validation, retryable classifier, attempt loop/backoff, logical deadline, budget reserve, commit guard, provider cancel과 metrics까지 연결한다. 중간 consumer가 없으면 option 효과를 확정하지 않는다.

그다음 실패 하나를 넣고 예상 tree를 그린다. first attempt가 400ms 뒤 503, second가 300ms 뒤 timeout, fallback이 remaining deadline 부족으로 거절되는 경우다. 각 edge의 reason과 reservation, final client error를 계산한다. 실제 source가 다른 tree를 만들면 최초 분기를 찾는다.

이 연습을 통과하면 사용자는 retry 횟수를 무작정 올리지 않는다. 성공 가능성, remaining SLO, semantic compatibility, concurrent cost와 orphan cleanup을 함께 보며 변경 후 어떤 metric과 fixture로 효과를 판정할지 안다.

장말 ledger에는 confirmed source fact, hand calculation, runtime unknown을 다른 열로 둔다. external provider cancel/usage처럼 local code가 증명하지 못하는 항목은 필요한 관측과 settlement owner를 적는다. 이 정직한 빈칸이 다음 운영 검증의 출발점이다.

마지막 acceptance에서는 failing attempt tree와 바로 인접한 passing tree를 함께 보존한다. commit 전 timeout은 fallback이 가능하지만 first content commit 뒤 같은 timeout은 terminal error가 되어야 한다. 두 fixture가 정책 경계를 실제로 검출한다.

수정 후에는 late provider usage settlement까지 기다린다. client response 회복만 보고 budget terminal을 닫지 않는다. 모든 reservation과 orphan task가 종료되고 SLO goodput·cost guardrail이 두 window 동안 유지돼야 한다.

그때 logical request와 모든 physical attempt의 ledger를 최종 보관하고 다음 downstream owner에게 정확히 전달한다.

**세 retry 통제 실험.** 첫 실험은 upstream connect 전에 실패시켜 같은 logical request의 새 attempt가 안전한지 본다. 둘째 실험은 첫 token commit 뒤 timeout을 주입해 자동 retry가 중복 응답을 만들지 않는지 확인한다. 셋째 실험은 전체 deadline 직전 provider fallback을 선택하게 해 남은 budget보다 connect timeout이 큰 시도를 거부하는지 본다. 각 실험은 route decision, attempt generation, spent budget과 terminal owner를 기록한다.

## 20.11 장말 source note — pinned LiteLLM parser와 consumer

source walk는 router initialization/config parse, deployment selection, retry decision, fallback mapping, streaming wrapper, timeout task/cancel, budget limiter admission/reconciliation을 잇는다. option definition만 인용하지 않고 실제 state mutation과 next attempt 생성 지점을 찾는다.

### 20.11.1 selection source card

card에는 input model/group, filtered candidates, strategy class/function, stats generation, selected deployment, credential/region/capability, fallback reason을 둔다. health/cache가 stale할 수 있으면 유효시간과 generation을 적는다. selection source는 provider 실제 availability를 증명하지 않는다.

### 20.11.2 retry/fallback source card

exception mapper와 `should_retry` predicate, retry cap/backoff, fallback list/group, commit-aware streaming wrapper를 연결한다. same attempt coroutine이 정말 취소/종료되는지 task lifecycle을 본다. exception을 반환한 함수와 provider HTTP/generation terminal은 다를 수 있다.

fallback wrapper가 first chunk를 buffer해 commit 전 provider error에 다른 stream으로 갈아탈 수 있다면 buffer size/time과 semantics를 적는다. first chunk를 client에 emit한 뒤 fallback하지 않는 guard를 찾는다. 없다면 incident risk로 표시한다.

### 20.11.3 budget limiter source card

budget key가 tenant/user/team/model 중 무엇인지, estimate formula와 currency/model price revision, atomic store operation, reserve/release/reconcile consumer를 적는다. provider-reported usage가 없을 때 estimate 정책과 delayed callback을 본다.

in-memory limiter는 multiprocess/distributed deployment에서 global ceiling을 보장하지 않을 수 있다. external store/lock 사용 여부와 failure behavior를 확인한다. limiter unavailable 때 fail-open/closed를 명시한다.

### 20.11.4 current revision과 evidence 범위

pinned LiteLLM revision의 exact file/symbol/span을 source note와 claim에 둔다. upstream 변화가 빠르므로 option 이름이 같아도 consumer/state가 이동할 수 있다. 최신 revision에서는 parser→consumer→attempt transition을 다시 감사한다.

정적 source는 가능한 branch와 state mutation을 증명한다. 실제 selected route, provider cancel/usage와 latency는 trace가 필요하다. 실행하지 않은 결과를 source fact로 쓰지 않는다.
