# 29장. Chunked prefill과 decode 우선순위

긴 prompt P가 들어오는 순간 D1과 D2가 decode 중이라고 하자. 한 iteration의 query-token budget은 8이고 P는 아직 20 tokens를 계산해야 한다. 누가 먼저 여덟 자리를 가져가는지, chunk 뒤 무엇을 commit하는지, 남은 suffix가 어디에 보존되는지가 TTFT와 ITL을 바꾼다.

Chunked prefill은 tokenizer가 prompt를 파일처럼 분할하는 기능이 아니다. scheduler가 한 request의 computed prefix와 uncomputed suffix를 여러 model iterations에 걸쳐 진전시키는 실행 정책이다. 27장의 budget 정의와 30장의 장기 fairness를 반복하지 않고 priority, chunk boundary와 commit correctness에 집중한다.

## 29.1 같은 여덟 자리를 누가 먼저 쓰는가

decode-first에서 step 0은 D1=1, D2=1, P=6이다. P의 computed prefix는 6, suffix는 14다. step 1 뒤 prefix 12, step 2 뒤 prefix 18, step 3의 P=2로 prompt가 끝난다. D1/D2는 네 steps 모두 진전하지만 P는 네 chunk completion 뒤 first output 후보를 얻는다.

prefill-first는 P를 8, 8, 4로 처리한다. strict prefill-only라면 D1/D2는 세 iterations 동안 grant 0이다. P TTFT 후보는 한 step 빨라지지만 D streams의 ITL은 정상 한 step에서 네 step 가까이 벌어진다. 같은 계산량 20이어도 priority가 사용자별 latency를 교환한다.

이 산술에는 중요한 숨은 질문이 있다. P의 마지막 prompt chunk 4가 step 2에서 끝났을 때 그 step의 remaining balance 4를 누구에게 줄 수 있는가다. mixed execution이 가능하면 D1과 D2에 하나씩 주고 두 자리는 비워 둘 수 있다. scheduler가 prefill batch를 확정한 뒤 decode candidates를 다시 보지 않거나 backend가 두 phase를 섞지 못하면 네 자리 모두 비어도 D는 다음 iteration까지 기다린다. `prefill-first`라는 이름 하나로는 어느 결과인지 알 수 없다.

### 논리 step을 wall time으로 바꾸기

이제 각 path에 시간을 부여하자. prefill-only eight-row iteration은 12ms, mixed two-decode-plus-six-prefill iteration은 14ms, decode-only two-row iteration은 4ms라고 가정한다. 이 숫자는 측정값이 아니라 계산 방법을 드러내기 위한 fixture다. strict prefill-first에서 P의 세 chunks는 36ms에 끝난다. D1/D2는 0ms 직전에 이전 token을 받았다고 하면 다음 token을 적어도 40ms 부근에 받으므로 ITL은 약 40ms다.

decode-first에서는 세 번의 full mixed step이 각각 14ms다. P가 마지막 2 tokens만 남은 네 번째 mixed step은 physical implementation에 따라 여전히 graph capacity 8을 쓰거나 smaller path로 8ms에 끝날 수 있다. 전자를 쓰면 prompt completion은 56ms, 후자를 쓰면 50ms다. D1/D2 output은 14, 28, 42, 50 또는 56ms에 이어진다. P TTFT를 14~20ms 양보해 streaming cadence를 지킨 셈이다.

P의 첫 output 시점은 prompt completion과 같다고 단정하지 않는다. 마지막 prompt position hidden state에서 logits를 만들고 같은 iteration output processor가 sample할 수 있다면 completion 직후 first token 후보가 생긴다. runner가 prompt completion을 scheduler에 돌려준 뒤 next iteration에 decode token을 별도로 넣어야 한다면 decode-only 또는 mixed step 하나가 더 붙는다. source에서 logits selection index와 sampling commit 경계를 찾아 계산식에 반영한다.

wall-time fixture는 priority와 kernel path를 분리한다. prefill-first가 세 steps이고 decode-first가 네 steps라 해도 prefill-only path가 20ms, mixed path가 8ms라면 순위가 달라질 수 있다. 반대로 mixed path가 graph fallback과 metadata rebuild로 느리면 D cadence를 지키려 한 정책이 P뿐 아니라 전체 goodput도 낮출 수 있다. 그래서 plan grant와 runner duration을 같은 event에 둔다.

### D1과 D2의 output length를 바꾸면 priority가 어떻게 변하는가

D1이 step 1에서 EOS를 내고 D2만 남는다고 하자. decode-first의 다음 balance는 7이므로 P grants는 `6,7,7`이 될 수 있다. P는 세 steps에 끝나 prefill-first와 논리 step 수가 같아지고 D2도 계속 진전한다. active decode count가 줄어드는 membership transition이 effective chunk를 자동으로 키운 것이다.

반대로 D3~D6까지 여섯 decode streams가 더 있으면 eight decode tokens만으로 budget이 찬다. strict decode-first는 P grant 0을 반복할 수 있다. 이 장에서는 그 장기 starvation을 해결하지 않지만, P가 `chunked enabled`인데도 시작하지 못하는 이유를 설명할 수 있어야 한다. option이 no-op이 아니라 priority 앞선 consumers가 balance를 모두 쓴 것이다. 30장의 fairness는 이 관측에서 출발한다.

decode requests가 매 step 모두 query token 하나만 요구한다는 가정도 speculative decoding이나 multi-token scheduling에서는 달라질 수 있다. D1 grant가 여러 tokens라면 P에 남는 balance가 더 작다. 이 장의 fixture는 one-token decode로 고정해 priority를 분리하고, 실제 source에서는 request별 `num_new_tokens` 단위를 확인한다.

### TTFT와 ITL을 평균으로 합치지 않기

P의 TTFT 개선 20ms와 D1/D2 ITL 악화 26ms를 평균 latency 하나로 더하면 어느 사용자가 손해를 봤는지 사라진다. prefill requests는 arrival부터 first accepted output, decode requests는 직전 accepted output부터 다음 accepted output을 잰다. scheduler iteration duration은 둘의 causal bridge이지 사용자 metric의 대체물이 아니다.

P arrival가 D의 직전 output 직후인지 직전인지도 ITL 계산에 영향을 준다. iteration이 이미 submit된 뒤 도착했다면 다음 scheduling boundary까지 기다리는 baseline queue delay가 있다. priority 비교는 같은 arrival phase를 사용한다. captured trace를 replay할 때 request arrival offset을 iteration boundary에 고정하는 이유다.

## 29.2 chunk boundary는 계산 경계이자 commit 경계다

첫 chunk [0,6)이 성공하면 scheduler progress, cache logical length와 KV write가 모두 6을 가리켜야 한다. scheduler만 전진하면 존재하지 않는 prefix를 읽고, KV만 전진하면 같은 chunk를 다시 계산한다. plan epoch, interval, expected prefix와 commit generation을 묶는다.

남은 suffix는 새 request가 아니다. sampling state, prompt logprob accumulator, multimodal feature, adapter와 cache ownership이 같은 request lifetime에 남는다. 중간 chunk logits도 전체 prompt 끝의 next-token logits로 commit해서는 안 된다.

### commit ordering을 한 epoch씩 펼치기

epoch 10 시작 전에 P의 committed prefix는 6이라고 하자. scheduler는 plan C10에 expected prefix 6, grant `[6,12)`, cache write slots와 request generation 3을 넣는다. allocator가 필요한 block을 reserve하고 runner가 input rows를 만든다. plan이 device에 submit된 뒤 host가 P offset을 12로 먼저 바꾸면 안 된다. device failure나 cancellation이 생길 경우 12라는 숫자가 실제 cache보다 앞선다.

device completion event가 signal되면 runner output은 C10 epoch를 동반한다. cache owner는 writes가 visible하다는 전제 아래 logical length를 12로 commit하고 scheduler progress도 12로 옮긴다. prompt logprob가 요청됐다면 accountable score interval을 accumulator에 exactly once 반영한다. 마지막으로 remaining suffix view를 `[12,20)`으로 갱신하고 P를 next candidate owner에게 넘긴다.

이 순서를 하나의 mutex 안에서 실행해야 한다는 뜻은 아니다. cache commit, result processing과 next plan preparation이 pipeline될 수 있다. 그러나 next plan C11은 C10 cache write completion과 progress commit을 happens-before로 가져야 한다. immutable plan epoch와 request generation이 비동기 구성요소 사이의 합의점이다.

retry가 C10 completion을 두 번 전달하는 경우를 생각하자. 단순히 `computed += 6`을 하면 progress가 18로 뛰고 logprob도 두 번 더해진다. idempotent commit은 request generation과 interval end가 이미 committed인지 확인한다. same epoch duplicate는 무시하거나 assertion으로 잡고, 다른 epoch가 같은 interval을 쓰려 하면 plan construction 오류로 본다.

### cache block과 logical interval을 동시에 기록하기

block size 4에서 prefix 6은 block 0 positions 0…3과 block 1 positions 4…5를 사용한다. `[6,12)`는 block 1의 slots 2…3과 block 2 전체를 쓴다. logical new tokens는 6, new physical block allocation은 1일 수 있다. metrics가 allocation count만 보면 chunk work 6을 4로 오해하고, token count만 보면 allocator pressure를 과대평가할 수 있다.

block 1이 prefix sharing으로 다른 request와 shared라면 positions 6…7 write 전에 copy-on-write가 필요할 수 있다. 이때 new physical blocks는 2가 된다. scheduler cache estimate가 shared-tail 상태를 몰라 optimistic하면 plan은 query budget을 통과하고 allocator에서 실패한다. first limiting reason을 `KV capacity` 하나로만 남기지 않고 shared-tail copy와 fresh suffix allocation을 구분할 가치가 있다.

sliding-window cache는 logical prefix length 12와 physically retained positions가 다를 수 있다. scheduler progress는 prompt에서 12 tokens가 계산됐음을 뜻하지만 cache는 window 밖 old K/V를 버렸을 수 있다. next chunk가 요구하는 attention context와 model semantics가 window policy에 맞는지 cache implementation이 책임진다. physical block count를 computed length로 역산하면 안 된다.

### prompt logprob의 shifted domain

prompt tokens `p0…p19`의 conditional scores는 보통 `p_i`를 예측한 이전 position logits와 맞물린다. `p0` score는 BOS나 이전 context 정의에 따라 없을 수 있다. 따라서 expected count가 언제나 prompt length 20이라고 단정하지 않고 API contract의 score domain을 먼저 정한다. 중요한 invariant는 그 domain의 각 logical position을 정확히 한 번 commit하는 것이다.

첫 chunk physical inputs가 `[p0…p7]`이고 score domain이 `p1…p7`이면 accountable interval은 `[1,8)`다. continuation이 cache API 때문에 `p7`을 boundary context로 다시 포함해 physical inputs `[p7…p15]`가 되더라도 새 accountable scores는 `p8…p15`, 즉 `[8,16)`이어야 한다. physical input start 7을 그대로 accumulator start로 쓰면 p7 score를 중복한다.

마지막 chunk `[p15…p19]`도 같은 원리로 accountable `[16,20)`이다. 세 intervals `[1,8),[8,16),[16,20)`은 score domain `[1,20)`을 빈틈 없이 덮는다. count뿐 아니라 position IDs를 검사해야 동일 count의 gap/overlap 교환도 잡는다.

prefix cache hit가 `[0,12)`인데 cached artifact가 KV만 보존하고 prompt logits를 보존하지 않았다고 하자. generation은 suffix 8만 계산하면 되지만 prompt logprob API는 prefix scores를 반환할 증거가 없다. cache hit를 포기하고 recompute하거나 unsupported/partial contract를 명시해야 한다. KV progress와 logprob progress를 한 offset으로 합치면 이 차이를 표현할 수 없다.

### multimodal state의 commit

image feature span이 prompt positions 5…12에 매핑되면 chunk `[0,5)` 뒤 text/KV prefix만 commit할 수 있다. 다음 plan이 span 전체를 `[5,13)`으로 처리한다면 encoder output handle, placeholder-to-feature offsets와 adapter/model generation이 함께 보존돼야 한다. request가 parked되는 동안 feature buffer를 scratch처럼 재사용하면 suffix tokens는 정상이어도 embedding inputs가 오염된다.

backend가 span partial processing을 허용하면 `[0,8)`과 `[8,16)`처럼 나눌 수 있지만 feature offset도 chunk local로 재계산해야 한다. 첫 chunk가 feature rows 0…2, 둘째가 3…7을 소비했다는 progress를 token offset 하나로 충분히 표현할 수 있는지 확인한다. 그렇지 않으면 multimodal progress가 별도 commit state다.

encoder가 asynchronous라면 P가 scheduler candidate여도 features ready event가 없어서 grant 0일 수 있다. 이를 decode priority 결과로 오진하지 않는다. iteration record의 first limiting reason이 readiness인지 balance ordering인지 구분한다. feature ready 뒤 첫 eligible epoch까지의 delay만 priority latency다.

### cancellation이 chunk 사이에 들어올 때

P가 C10을 submit한 뒤 client cancellation이 온 경우 current device work를 request 하나만 취소하기 어려울 수 있다. C10 completion을 drain하되 next suffix를 schedule하지 않고 committed cache를 release하는 path가 필요하다. cancellation timestamp만 보고 C10 write를 stale work로 오인하지 않는다.

반대로 cancellation이 submit 전에 받아들여졌다면 plan에서 P rows를 제거하거나 plan 전체를 rebuild해야 한다. 이미 mapping을 freeze했다면 dummy/discard로 실행할 수 있지만 output/logprob를 user commit해서는 안 된다. 어느 safe boundary를 택하든 suffix owner와 reserved blocks가 정확히 한 번 해제돼야 한다.

terminal finalizer는 prompt complete와 request cancelled를 구분한다. prompt completion은 decode lifetime으로 state를 넘기지만 cancellation은 cache, suffix, feature와 logprob accumulator를 닫는다. 둘을 같은 `prefill done` boolean으로 표현하면 cancelled P가 decode candidate로 되살아나는 race가 생긴다.

## 29.3 effective chunk는 여러 조건의 교집합이다

requested size 8도 remaining tokens, step balance, threshold, KV capacity, block boundary, multimodal span, graph eligibility와 workspace가 다시 줄인다. requested value→validated effective state→scheduler consumer→state mutation→latency/correctness effect를 이어 읽는다.

block size 4에서 committed prefix 6, next grant 6이면 마지막 block 두 slots와 새 block 하나를 쓴다. cached prefix를 다시 reserve하면 이중 계산이고 partially filled block ownership을 놓치면 collision이다.

### requested option이 latency까지 도달하는 다섯 관문

첫 관문은 parsing과 type domain이다. 문자열 `8`, integer 8, 0과 unset이 서로 어떤 뜻인지 확인한다. 0이 disable인지 automatic인지 구현마다 다를 수 있다. CLI help만 읽지 않고 constructor default와 config serialization을 본다. deployment wrapper가 option을 전달하지 않거나 다른 이름으로 덮어쓰면 여기서 이미 requested value가 사라진다.

둘째 관문은 validation과 capability다. model maximum length, attention backend, pipeline mode, multimodal path와 incompatibility가 size를 거부하거나 chunking을 비활성화할 수 있다. error로 종료하는지 warning 뒤 fallback하는지 구별한다. process가 올라왔다는 사실은 requested feature가 active라는 증거가 아니다.

셋째 관문은 finalized effective state다. automatic default가 workload/model config에서 계산될 수 있고 dynamic mode가 static requested size를 upper bound로만 쓸 수 있다. startup log 또는 config snapshot에서 `enabled`, effective upper bound, mixed capability와 policy type을 함께 읽는다. 단일 boolean으로는 priority를 알 수 없다.

넷째 관문은 scheduler predicate다. P remaining 20, effective upper 8이어도 D1/D2가 두 budget을 먼저 쓰면 grant 6이다. KV allocator가 두 자리만 허용하면 2, multimodal boundary가 앞에 있으면 5가 될 수 있다. candidate가 priority에서 선택되지 않으면 0이다. effective config와 actual interval histogram을 분리하는 이유다.

다섯째 관문은 runner와 commit effect다. scheduler interval 6이 graph capacity 8, workspace chunk 4×2 suboperations 또는 backend micro-batches로 바뀔 수 있다. cache/progress가 6 전진하고 prompt completion 및 D output timestamps가 움직였는지 확인한다. config와 plan이 바뀌었어도 physical path와 latency가 그대로면 기대한 최적화 지점이 아니었다.

이 관문을 표가 아니라 인과 문장으로 남긴다. “사용자는 upper bound 8을 요청했다. model/backend validation 뒤 chunking과 mixed mode가 active다. epoch 12에 D1/D2가 먼저 두 tokens를 받아 P interval은 `[6,12)`였다. runner는 total rows 8 graph를 replay했고 cache와 logprob progress가 12로 commit됐다. D ITL은 14ms, P remaining은 8이 됐다.” 이 문장은 option이 실제 state와 latency에 도달했음을 증명한다.

반대로 no-op도 설명할 수 있다. “upper bound를 8에서 4로 바꿨지만 workspace predicate가 이전에도 actual grants를 4로 제한했다. scheduler interval histogram, graph capacity와 latency가 변하지 않았다.” 이때 option 자체가 고장난 것이 아니다. first limiting condition이 이미 더 작은 값을 갖고 있었다.

### chunk size를 줄였는데 memory가 늘 수 있는 이유

P=20을 8,8,4로 처리할 때 live prefix blocks는 각 completion 뒤 8,16,20에 해당한다. 4씩 다섯 chunks로 나누면 final persistent KV는 같지만 intermediate scheduling과 workspace가 더 자주 생긴다. double buffering이 next plan metadata를 미리 준비하면 두 chunk의 temporary buffers가 겹칠 수 있다. peak가 단순 chunk size에 비례하지 않는다.

작은 chunks가 graph captured shape를 벗어나 eager fallback하면 allocator temporary lifetime이 달라질 수 있다. graph replay용 persistent buffers는 이미 reserved돼 있고 eager workspace가 추가로 생기면 memory가 오히려 증가한다. metric은 KV persistent blocks, graph pools와 per-iteration workspace를 나눠야 한다.

반대로 큰 chunk는 context gather나 logits temporary를 키우지만 launch 횟수를 줄인다. prompt logprobs가 필요하면 각 row logits를 보존하는 범위가 일반 last-row-only prefill보다 클 수 있다. 동일 chunk option도 logprob request mix에 따라 memory curve가 달라진다.

### dynamic chunk가 재현성을 흐리는 방식

같은 P라도 concurrent D count와 free cache, graph eligibility가 달라지면 intervals가 `8,8,4`가 아니라 `6,7,7` 또는 `4,4,4,4,4`가 된다. output semantics는 같아야 하지만 floating-point kernel path나 sampling boundary가 달라질 수 있다. greedy parity와 tolerance-based logits 비교, request-local RNG를 사용한다.

incident 재현 시 requested size만 고정하고 actual plan을 기록하지 않으면 같은 chunk boundaries를 재현하지 못한다. captured trace에는 arrivals, active memberships, cache/prefix hit, effective config와 plan intervals를 포함한다. graph warmup state도 path를 바꾸므로 replay 조건을 명시한다.

dynamic decision에 latency predictor가 있다면 prediction input과 selected size를 함께 남긴다. prediction이 틀려 size가 나빴는지, prediction은 맞았지만 downstream workspace/graph path가 달랐는지 구분한다. `dynamic chose 8`만으로는 model과 consumer 중 어디를 고칠지 모른다.

### priority와 commit을 하나의 트랜잭션으로 보지 않는 이유

scheduler가 P에 grant 6을 주는 결정과 cache가 `[6,12)`를 commit하는 것은 서로 다른 단계다. priority는 plan construction의 ordering이고 commit은 execution 성공 뒤 state advance다. grant됐다고 즉시 progress를 올리면 실패 rollback이 어렵고, commit됐는데 parked owner를 만들지 않으면 사건 1이 생긴다.

관측에서는 `planned`, `submitted`, `device-complete`, `state-committed`, `next-owner`를 나눈다. 정상 fast path에서는 가까워도 fault injection에서 갈라진다. OOM before submit, device error, cancellation after submit, duplicate completion과 result processor exception을 각각 어느 단계에서 멈추는지 source로 추적한다.

rollback도 resource별이다. submit 전 실패는 reserved cache slots를 반환하고 progress는 유지한다. device write 뒤 result processing 실패는 cache contents가 존재하지만 logical commit 여부를 결정해야 한다. 무조건 retry하면 duplicate write/logprob가 생길 수 있다. commit record와 idempotency가 필요한 이유다.

three-stack 비교에서 이 단계들을 담당하는 object가 다르다. vLLM scheduler/runner/cache state, SGLang scheduler/PrefillAdder/result processor, Transformers scheduler/request/cache가 나눠 소유한다. class 대응표보다 각 단계의 owner와 전달 payload를 기록하면 cross-file race를 찾기 쉽다.

## 29.4 vLLM의 option에서 실제 grant까지

고정 source는 vLLM v0.27.1 commit 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac이다. config validation, arg effective state, scheduler running/waiting grant, input accounting과 MLA workspace를 순서대로 추적한다.

[`vllm/config/scheduler.py:70-281`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/config/scheduler.py#L70-L281)에서 `enable_chunked_prefill`과 `long_prefill_token_threshold`의 선언, default와 validation을 읽는다. threshold가 model length보다 클 수 없는 것은 scheduler가 의미 없는 domain 밖 상한을 갖지 않게 하기 위해서다. user가 8을 입력했다는 사실과 validator 뒤 effective value가 8이라는 사실은 다르다.

[`vllm/engine/arg_utils.py:2598-2664`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/engine/arg_utils.py#L2598-L2664)는 model capability와 default selection, incompatible condition에서 요청값을 확정하거나 비활성화하는 경계다. 운영 record에는 CLI value와 finalized scheduler config를 둘 다 남긴다. validation error, warning 뒤 fallback과 silent effective state를 구별한다.

실제 consumer는 [`vllm/v1/core/sched/scheduler.py:404-637`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L404-L637)이다. candidate `num_new_tokens`가 long-prefill threshold로 잘리고 current step balance로 다시 잘린다. running requests 순회가 D1/D2와 partially computed P의 priority를 물질화한다. config 8이 있어도 D rows 두 개가 먼저 grant되면 P effective interval은 six tokens다.

waiting admission은 [`scheduler.py:687-913`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L687-L913)에서 본다. chunking이 off이고 remaining prompt가 balance보다 크면 whole prefill이 들어오지 못한다. on이면 current balance만큼 잘라 admission할 수 있다. long prompt가 queue head에서 전체 batch를 막는지, partial progress를 시작하는지가 여기서 갈린다.

runner accounting은 [`vllm/v1/worker/gpu/input_batch.py:445-456`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu/input_batch.py#L445-L456)에서 확인한다. 아직 prefill 중인 row의 sampled/rejected token count를 decode output commit처럼 취급하지 않는 경계다. scheduler가 six query tokens를 grant한 것과 user-visible output six 개가 생긴 것은 전혀 다른 사건이다.

MLA path는 [`mla_attention.py:1600-1831`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/attention/mla_attention.py#L1600-L1831)의 context gather workspace와 최대 context chunk를 연결한다. scheduler threshold를 줄여도 peak memory가 그대로면 query chunk 외 context workspace가 지배하는지 확인한다. option→grant 변화는 보였지만 physical allocation이 안 바뀌었다면 source walk를 backend까지 더 내려가야 한다.

## 29.5 SGLang과 Transformers의 parked chunk

SGLang은 effective chunk size와 mixed 가능 상태, PrefillAdder, parked chunk owner와 prompt logprob result를 잇는다. Transformers는 FIFO/PrefillFirst ordering, remaining prefill tokens와 cache read/write boundary를 잇는다.

SGLang 고정 source는 v0.5.18 commit 71de97b264b04dcd514cf904003028aefe9775c8이다. [`scheduler.py:1160-1181`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L1160-L1181)에서 effective `chunked_prefill_size`와 mixed-chunk 가능 여부가 초기화된다. requested size만 metric에 남기면 mixed disabled나 dynamic adjustment를 놓친다.

[`scheduler.py:3241-3278`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L3241-L3278)은 batch마다 static/dynamic chunk size를 선택하고 `PrefillAdder`를 만든 뒤 이전 iteration의 `chunked_req`를 다시 넣는다. parked P가 ordinary waiting tail로 가는지 별도 owner로 먼저 복원되는지가 실제 priority다.

[`schedule_policy.py:504-760`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_policy.py#L504-L760)의 `PrefillAdder`는 total-token capacity, running reserve와 current budget을 함께 소비한다. [`schedule_policy.py:997-1120`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_policy.py#L997-L1120)에서 request 일부를 materialize하고 remainder를 반환한다.

반환 object가 scheduler persistent owner로 이어지지 않으면 P suffix는 cache만 남기고 사라진다.

prompt logprob는 [`batch_result_processor.py:317-363`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L317-L363), completion time은 [`batch_result_processor.py:560-620`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L560-L620)을 잇는다.

KV length만 맞아도 score interval이 겹치거나 prompt complete timestamp가 첫 chunk에 찍힐 수 있다.

Transformers 고정 source는 v5.15.1 commit 550d7b3834670483a4df436541272c055dc364bf이다. [`continuous_batching/scheduler.py:1-280`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/scheduler.py#L1-L280)에서 request token list를 current budget으로 자르고 remaining prefill tokens에 suffix를 보존하는 경계를 찾는다. FIFO와 PrefillFirst `schedule_batch`는 같은 chunk representation을 다른 ordering으로 소비하므로 chunk feature와 priority policy를 분리해 비교한다.

[`continuous_batching_architecture.md:41-77`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/docs/source/en/continuous_batching_architecture.md#L41-L77)은 admission에 token budget과 cache space가 모두 필요하고 PrefillFirst가 prompt completion과 decode 재개를 어떻게 의도하는지 설명한다. 문서가 design intent를 주고 source가 exact mutation을 준다.

cache continuation은 [`cache.py:401-430`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache.py#L401-L430)에서 읽는다. non-chunked first prefill과 달리 later chunk는 cached prefix를 읽고 suffix를 쓴다. sliding window가 있으면 read-before-overwrite ordering이 logical prefix 의미를 보존하는지 확인한다.

### 같은 옵션표 대신 같은 질문표로 비교하기

세 구현은 option 이름과 object topology가 다르다. `chunk size` 숫자만 나란히 놓으면 실제 차이를 놓친다. 첫 질문은 requested size가 어디에서 validate되는가다. vLLM은 scheduler config와 argument finalization을 왕복하고, SGLang은 scheduler initialization에서 mixed capability와 effective size를 함께 만든다. Transformers는 선택한 scheduler policy가 remaining tokens를 어떻게 자르는지부터 읽는다.

둘째 질문은 effective state를 누가 소비하는가다. vLLM scheduler는 running candidates의 new-token count에 threshold와 balance를 적용한다. SGLang `PrefillAdder`는 running reserve와 total-token condition을 함께 본다. Transformers FIFO와 PrefillFirst는 같은 remaining prompt representation을 서로 다른 순서로 선택한다. 같은 requested 8이어도 consumer에 도달했을 때 P grant가 6, 8 또는 0일 수 있다.

셋째는 mutation의 모양이다. vLLM에서는 scheduled count와 runner cached request state, cache blocks의 전진을 잇는다. SGLang에서는 materialized chunk와 returned `chunked_req`, result accumulator를 잇는다. Transformers에서는 scheduled slice와 remaining tokens, cache read/write positions를 잇는다. 구현별 class 이름보다 committed prefix와 suffix가 정확히 한 owner에 남는지 확인한다.

넷째는 user-visible effect다. option을 8에서 4로 줄였다는 사실만으로 D ITL 개선을 주장하지 않는다. actual grants가 줄었는지, mixed path가 늘었는지, graph fallback과 host overhead가 어떻게 변했는지, P prompt complete와 D outputs가 움직였는지 본다. effective grant가 이미 workspace limit 4였다면 config 8→4는 no-op이다.

다섯째는 correctness다. prompt completion 전 sampling 차단, cache/progress commit, prompt score interval과 parked suffix owner를 찾는다. 성능 option도 state mutation을 바꾸므로 parity fixture를 동반한다. output text 하나뿐 아니라 positions, usage와 logprob count도 비교한다.

**vLLM source를 P 하나로 왕복하기**

P가 waiting이고 prompt 20이 남았다고 하자. finalized config에서 chunking true와 threshold 8을 확인한다. scheduler waiting path가 P를 고를 때 balance 6이면 grant `[0,6)`을 만든다. schedule output이 runner request state에 들어가 positions 0…5와 cache slots를 만드는 지점을 잇는다.

runner completion 뒤 scheduler progress가 6인지 역방향으로 올라온다. P가 아직 prefill이므로 sampled output count가 decode처럼 증가하지 않는지도 확인한다. next epoch running ordering에서 P가 `[6,12)` candidate가 되는지, D1/D2 뒤 balance 6으로 잘리는지 본다. source 왕복이 이어져야 threshold가 실제 timeline을 만들었다고 말할 수 있다.

MLA workspace는 옆의 physical branch다. grant 6이 context gather chunk와 workspace shape로 어떻게 변하는지 본다. threshold 8과 effective grant 6을 알아도 workspace는 context length로 larger bucket을 잡을 수 있다. query grant만으로 peak memory를 예측하지 않는다.

**SGLang source를 parked P로 왕복하기**

P 첫 chunk가 materialize되면 execution view와 남은 logical request가 갈릴 수 있다. `add_chunked_req` 반환을 따라 scheduler field가 remainder를 소유하는지 본다. result processor는 current interval의 KV/logprob만 commit하고 remainder를 terminal로 만들지 않아야 한다.

next iteration에서 parked P를 adder에 다시 넣는다. 이미 cached prefix가 total-token capacity에 중복 reserve되지 않는지 확인한다. previous chunk length, cache-hit length와 new extension을 같은 length로 섞으면 P가 실제보다 비싸 보여 계속 park될 수 있다.

dynamic size가 첫 epoch 8, 다음 epoch 4여도 progress는 8→12다. execution view local length 4를 global computed length로 덮으면 suffix start가 4로 되돌아간다. local materialized tokens와 original request offset을 구별한다.

**Transformers에서 policy와 mechanism을 분리하기**

remaining tokens를 budget만큼 slice하는 것은 mechanism이다. FIFO와 PrefillFirst가 어느 slice를 먼저 만드는지는 policy다. 두 scheduler를 비교할 때 schedule result ordering과 intervals부터 비교한다.

P=20, D1/D2 fixture를 대입한다. 하나가 D1/D2를 먼저 넣고 P six를 자르면 decode-first 결과다. 다른 하나가 P eight를 먼저 넣으면 prefill-first 결과다. 다음 plan에 remaining 14 또는 12가 보존되는지 확인한다. 이름이 기대와 같아도 mixed remainder를 어떻게 쓰는지 source로 닫는다.

cache layer에서는 first chunk와 continuation을 구별한다. continuation은 prior positions를 read context로 쓰면서 new positions에 write한다. sliding window가 slot을 재사용할 때 current attention이 필요한 old value를 덮기 전에 읽어야 한다. scheduler priority가 맞아도 continuation이 틀리면 chunked mode에서만 parity가 깨진다.

**실행하지 않는 정적 worksheet**

requested value를 넣을 field, validation predicates, finalized state, scheduler read site와 mutation field를 한 줄로 잇는다. incompatible model path에서 disable되는 분기와 warning도 표시한다.

P fixture의 각 epoch에서 condition을 손으로 평가한다. remaining 20, balance 6, threshold 8이면 grant 6이다. progress 6 뒤 remaining 14, 다음 balance 6이면 `[6,12)`다. 실제 latency를 주장하지 않지만 semantic no-op과 offset 오류는 정적으로 발견할 수 있다.

correctness invariants를 consumer별로 배치한다. scheduler는 intervals 연속성, allocator는 write-slot 비중첩, runner는 partial logits sampling 차단, result processor는 score domain 단일 coverage, parked owner는 suffix 보존을 맡는다. runtime을 실행하지 않는다는 것은 mutation을 얕게 읽는다는 뜻이 아니다.

**세 구현을 같은 P 생애로 다시 왕복하기**

비교의 출발점은 class 이름이 아니라 P의 여섯 사건이다. arrival, first grant, intermediate commit, park, prompt completion, first decode output이다. 각 구현에서 이 사건의 owner와 payload를 찾으면 서로 다른 architecture를 같은 질문으로 읽을 수 있다. `ScheduleBatch`와 `Request`를 억지로 일대일 대응시키지 않는다.

vLLM에서 P arrival 뒤 waiting request가 scheduler candidate가 된다. finalized chunked-prefill state가 true이고 balance가 6이면 schedule output에 P의 new-token count 6이 들어간다. runner는 request state의 computed progress와 grant를 결합해 physical positions와 cache mapping을 만든다. 여기서 scheduler의 logical interval과 input row interval을 구분한다. prefix cache hit나 flattened mixed batch 때문에 physical row index 0이 logical position 0이라는 보장은 없다.

completion은 runner output에서 scheduler 방향으로 올라온다. P는 prompt 전체를 끝내지 않았으므로 intermediate forward를 user token으로 commit하지 않고 progress/cache만 전진한다. next schedule에서 P는 running collection의 partially computed request로 보일 수 있다. 이 위치가 이미 decode 중인 D1/D2와 ordering을 만든다. prompt end에 도달한 epoch에서만 logits selection과 output processing이 generation lifetime으로 넘어가는 경계를 찾는다.

SGLang에서 P는 scheduler가 `PrefillAdder`에 넣는 logical request와 이번 iteration에 materialize된 execution view로 나뉠 수 있다. 첫 grant `[0,8)`을 만든 뒤 remainder를 `chunked_req` owner에 보존한다. current execution result의 cache/logprob interval을 commit한 다음 parked remainder가 next batch construction 앞부분에서 복원되는지 확인한다. execution view가 끝났다는 사실을 logical P terminal로 오해하지 않는 것이 핵심이다.

SGLang의 source 왕복은 result processor까지 내려가야 한다. scheduler가 interval을 올바르게 나눠도 prompt logprob accumulator가 local chunk offsets를 global positions로 변환하지 못하면 API 결과는 틀린다. 마지막 chunk completion time을 first chunk에 기록하면 TTFT decomposition도 틀린다. KV state, score state와 latency state가 같은 `chunked_req` 생애를 서로 다른 좌표로 갱신한다.

Transformers continuous scheduler에서는 P의 remaining prefill tokens representation이 mechanism을 보여 준다. FIFO와 PrefillFirst가 P/D candidates를 어떤 순서로 slice하는지는 policy다. first schedule result에 P 6 또는 P 8이 나타나는지 보고, request object의 remaining suffix가 14 또는 12로 변하는지 잇는다. cache는 continuation에서 prior positions를 읽고 new suffix를 쓴다.

Transformers 왕복의 교육적 가치는 classic generation과 대비할 때 커진다. classic fixed invocation은 외부 D requests가 P chunk 사이에 들어오는 owner가 없다. continuous scheduler는 request별 remaining tokens와 active membership을 유지해 iteration마다 순서를 다시 결정한다. chunk slicing만 복사하고 lifecycle manager가 없다면 true continuous interleave가 되지 않는다.

세 구현의 공통 invariant는 같다. first grant 전 P prompt generation이 고정되고, 각 commit interval이 이전 end에서 시작하며, prompt complete 전 user decode output이 없고, parked 동안 exactly one owner가 suffix를 보존하며, completion 뒤 first decode가 같은 request identity로 돌아간다. 구현 차이는 이 invariant를 어느 object와 callback에 나눠 놓았는가다.

**source link를 주장 단위로 배치하기**

한 broad scheduler link로 “validation부터 latency까지”를 모두 증명하지 않는다. config/argument link는 requested→effective state만 지지한다. scheduler link는 effective state를 읽어 grant를 계산하는 지점을 지지한다. runner/cache link는 physical interval과 persistent state mutation을 지지한다. result processor link는 output/logprob commit을 지지한다.

고정 commit link가 있어도 line 범위가 너무 넓으면 독자가 branch를 찾기 어렵다. 문장에는 symbol과 읽을 predicate를 함께 적는다. 새 version으로 갱신할 때 line number를 기계적으로 옮기지 않고 symbol의 input, mutation field와 call order가 같은지 diff한다. option 이름이 유지돼도 consumer semantics가 바뀔 수 있다.

source에 없는 latency 효과는 inference로 표시한다. running-first loop가 D cadence에 유리할 가능성은 grant ordering에서 추론할 수 있지만 exact milliseconds는 hardware/workload 측정 없이는 주장하지 않는다. 이 장의 12ms/14ms fixture도 계산 예시이지 benchmark가 아니다. source fact, cost-model inference와 운영 관측을 문장 수준에서 분리한다.

### pipeline parallel과 worker boundary에서 생기는 추가 owner

pipeline parallel에서는 scheduler가 선택한 chunk가 microbatch로 다시 나뉠 수 있다. logical P grant 8이 stage pipeline에서 4+4로 흐르더라도 scheduler progress를 첫 microbatch 뒤 8로 올려서는 안 된다. 모든 required stages가 interval을 완료하고 cache/state commit 조건을 만족하는 경계가 필요하다.

SGLang의 PP 관련 scheduler mixin과 latency predictor가 chunk size를 조정하면 single-stage requested size와 actual microbatch가 더 멀어진다. coordinator event에 logical interval, pipeline microbatch IDs와 completion aggregation을 함께 둔다. 한 microbatch retry가 prompt logprob를 중복 append하지 않게 idempotency 범위를 정한다.

tensor parallel workers는 같은 P interval과 cache positions를 공유해야 한다. rank 하나가 suffix generation이나 effective length를 다르게 보면 collective shape 또는 cache content가 갈라진다. plan epoch와 interval hash를 worker submit 전에 비교할 수 있다. service request identity는 coordinator만 알아도 device contract는 모든 ranks가 동의해야 한다.

### 관측 모델을 multimodal과 logprob까지 확장한다

텍스트-only P의 progress는 token interval 하나로 충분해 보인다. multimodal P는 text/token progress, encoder-feature readiness, placeholder/feature consumption과 cross-attention state가 따로 움직일 수 있다. prompt logprob P는 KV-computed progress와 score-committed progress가 다르다. 하나의 `num_computed_tokens`로 모든 준비 상태를 표현하면 cache hit와 partial score에서 모순이 생긴다.

### multimodal commit 좌표 세 개

첫 좌표는 prompt token position이다. image placeholder expansion 뒤 model input positions 5…12가 feature rows와 대응된다고 하자. 둘째는 encoder feature offset이다. positions 5…8이 feature rows 0…3을, positions 9…12가 rows 4…7을 소비할 수 있다. 셋째는 feature-buffer lifetime이다. encoder output ready부터 마지막 prompt/decode consumer까지 유지된다.

backend가 span을 나눌 수 없으면 scheduler는 token chunk를 position 5 앞에서 자른다. requested 8이 effective 5가 되며 unused budget 3이 생긴다. 다음 iteration에는 span 전체를 넣으려면 at least 8 capacity가 필요하다. D1/D2를 먼저 reserve하면 balance 6이라 span이 계속 들어오지 못할 수 있다. 단순 decode-first가 multimodal P를 park시키는 경계다.

span partial 처리가 가능하면 `[5,9)`와 `[9,13)`으로 나눌 수 있지만 feature progress를 4→8로 commit해야 한다. token progress만 9→13으로 옮기고 feature offset이 0에 남으면 second chunk가 first feature rows를 다시 사용한다. shape는 맞고 crash도 없지만 image grounding이 틀리는 silent correctness bug다.

encoder readiness도 priority와 구별한다. P가 queue head여도 feature future가 complete되지 않았으면 grant 0이다. readiness false를 budget/priority skip으로 기록하면 operator가 chunk size를 키워도 TTFT가 안 바뀐다. encoder completion부터 first eligible grant까지가 scheduler delay이며 arrival부터 encoder completion은 preprocessing delay다.

feature buffer release는 prompt completion과 같지 않을 수 있다. decoder-only multimodal model이 image embeddings를 prompt KV로 완전히 흡수했다면 completion 뒤 release할 수 있지만, generation steps가 별도 cross-attention features를 참조하면 terminal까지 필요하다. model architecture contract로 last consumer를 정한다.

### prompt logprob commit 좌표 세 개

첫 좌표는 KV-computed prompt end, 둘째는 logits가 실제 계산된 physical rows, 셋째는 API에 commit한 logical score positions다. last-row-only optimization이면 KV는 8까지 전진해도 all prompt scores가 존재하지 않을 수 있다. prompt-logprob request가 runner path를 all-row logits로 바꾸는지 확인한다.

chunk continuation은 boundary context 때문에 physical interval이 accountable interval보다 넓을 수 있다. result object에 local logits rows만 있으면 global position mapping을 plan에서 받아야 한다. `local row 0`이 new score인지 overlap context인지 source에서 명시되지 않으면 double count 위험이 있다.

score accumulator는 values뿐 아니라 token IDs와 positions를 보존한다. interval count가 맞아도 chunk 2 positions를 one-offset shift해 잘못 붙일 수 있다. expected prompt token at each score position과 returned token ID를 비교한다. prefix cache hit가 있으면 cached prefix scores의 availability를 별도 flag로 둔다.

retry와 pipeline aggregation에서도 exactly-once commit이 필요하다. same `(request_generation, score_interval)`이 이미 accumulator에 있으면 duplicate result를 거부한다. partial failure로 interval 일부만 commit하는 것을 허용한다면 committed bitmap/range set이 필요하지만, 단순성을 위해 chunk interval atomic commit을 택할 수도 있다.

### 두 progress를 분리해야 하는 cache hit 사례

P 20 중 KV prefix hit 12가 있다. generation에 필요한 compute suffix는 8이라 one chunk로 끝날 수 있다. 그런데 prompt scores 1…11이 cache artifact에 없다. `computed=12` 하나만 쓰면 scheduler는 suffix만 실행하고 result processor는 12 scores가 이미 있다고 오해한다.

선택지는 세 가지다. prompt logprob 요청에서는 KV hit를 사용하지 않고 full recompute한다. cache가 KV와 prompt logits/scores를 함께 보존한다. 또는 API가 cached prefix scores 미지원임을 명시하고 요청을 거부한다. 어느 선택이든 hidden omission보다 낫다. source에서 capability validation이 request admission 전에 일어나는지 본다.

### 수치 fixture를 운영 기록 형식으로 닫는다

budget 8 fixture를 실제 traffic record 형식으로 쓴다. epoch e0 before active `[D1,D2]`, waiting `[P]`, grants `D1:[k,k+1), D2:[m,m+1), P:[0,6)`이다. after commit P KV/progress 6, score interval은 contract에 따라 `[1,6)`, suffix `[6,20)`다. graph capacity 8, mixed true와 duration을 붙인다.

e1과 e2도 같은 구조로 P intervals `[6,12)`, `[12,18)`을 commit한다. e3는 P `[18,20)`과 D rows라 logical total 4다. graph capacity가 4면 dummy 0, capacity 8만 있으면 dummy 4다. P prompt complete event와 first-token sampling eligibility를 표시한다.

prefill-first record는 e0 P `[0,8)`, e1 `[8,16)`, e2 `[16,20)`이다. D grant 0인 epochs를 숨기지 않는다. e2 remaining balance 4를 D에 썼는지, phase separation으로 비웠는지 exact plan을 적는다. 그래야 ITL spike가 policy인지 backend mix restriction인지 구분된다.

block size 4를 붙이면 decode-first P allocation delta는 첫 6에서 two blocks, 다음 6에서 one new block, 다음 6에서 two 또는 shared-tail 조건에 따른 delta, 마지막 2에서 zero/one이 될 수 있다. logical interval과 allocator event를 함께 계산한다. physical block rounding이 step budget을 소비하는 것은 아니지만 admission cache predicate를 바꾼다.

multimodal span 5…12 non-splittable을 붙이면 original `[0,6)` plan은 invalid다. P first grant를 5로 줄이면 e0 total rows 7이고 graph 8 dummy 1이다. next balance 6으로 span length 8을 못 넣는다면 P는 park된다. decode-first를 유지하려면 D reserve를 일시 조정하거나 span-aware exception이 필요하며 그 장기 공정성은 30장에 넘긴다.

prompt logprob를 붙이면 each commit record에 score interval을 추가한다. boundary overlap implementation이라 physical intervals와 accountable intervals를 따로 적는다. 최종 union이 API score domain과 일치하지 않으면 latency가 좋아도 배포하지 않는다.

### 운영 변경의 성공 문장

나쁜 기록은 “chunk size를 4로 줄여 latency가 개선됐다”다. 좋은 기록은 다음과 같다. “validated effective cap이 8→4로 바뀌었고 previous actual grants 8이 4로 이동했다. D-ready epochs의 zero-grant ratio가 12%→1%로 줄어 ITL p99가 회복됐다. P TTFT p95는 80ms 늘었고 graph fallback은 변하지 않았다. KV/progress/score interval parity와 multimodal fixture가 통과했다.”

또 다른 좋은 기록은 no-op을 인정한다. “requested cap은 변했지만 multimodal boundary와 workspace가 이전에도 grants를 4 이하로 제한해 interval histogram이 변하지 않았다. TTFT 차이는 traffic noise 범위다.” 이런 결론은 실패가 아니라 잘못된 knob로 더 큰 회귀를 막은 조사 결과다.

rollback 기준은 cross-request text parity만이 아니다. interval gap/overlap, score count/position mismatch, parked owner cardinality, feature generation mismatch, D zero-grant/ITL SLO와 allocator nonreturn을 포함한다. correctness incident는 throughput 이득과 교환하지 않는다.

### 30장으로 넘기는 최소 정보

30장은 어떤 request가 반복적으로 먼저 선택돼도 다른 request가 무한히 기다리지 않게 하는 규칙을 다룬다. 이 장은 그 입력으로 candidate readiness, per-epoch requested/effective grant, zero-grant reason, queue age, last-progress epoch와 chunk atomicity constraint를 넘긴다.

multimodal non-splittable span처럼 minimum atomic grant가 8인 P는 balance 6만 반복 제공하면 영원히 진전하지 못한다. fairness policy는 P age만 올리는 것으로 부족하고 atomic requirement를 만족하는 future capacity를 만들어야 한다. 반면 ordinary text P는 grant 1로도 진전할 수 있다. 이 차이를 30장이 알 수 있게 `minimum_next_chunk`를 넘긴다.

D1/D2에는 last accepted output과 current ITL age를 넘긴다. P에는 arrival/first-grant/prompt progress age를 넘긴다. 어떤 age를 비교하고 reserve할지는 다음 장 몫이다. 이 장에서는 값이 정확한 commit epoch에서 갱신됐는지만 보장한다.

fairness가 P에 우선권을 줘도 current interval commit이 실패하면 credit을 소비했다고 보면 안 된다. planned grant와 committed progress를 분리해 다음 policy state가 실제 progress를 기준으로 작동하게 한다. cancellation과 terminal은 credit/resource 환불 semantics를 함께 넘긴다.

최종적으로 priority는 “decode가 중요하다” 또는 “prefill이 중요하다”라는 구호가 아니다. 같은 budget 안에서 어느 lifetime의 다음 atomic work를 먼저 materialize할지 정하는 순서다. chunking은 긴 work를 interleave 가능하게 만들지만 cache, score와 feature state를 여러 commit으로 쪼개 correctness 표면을 넓힌다. 운영자는 latency 이득만큼 그 commit 증거를 소유해야 한다.

## 29.6 관측은 chunk size보다 interval과 이유를 남긴다

iteration event에는 requested size, effective [start,end), prior committed prefix, remaining suffix, limiting reason, KV allocation delta, graph path와 workspace를 둔다. P chunk completions와 D1/D2 accepted output timestamps를 겹쳐 TTFT/ITL 교환을 본다.

이 record를 iteration ID 없이 서로 다른 metric stream에서 추측해 조인하지 않는다. scheduler가 next plan을 준비하는 동안 previous runner output이 돌아올 수 있다. plan epoch와 commit generation을 공통 key로 두고 requested grant, device completion, cache commit과 accepted output을 한 causal chain으로 묶는다. timestamp가 가까운 사건을 동일 chunk라고 가정하면 pipeline overlap에서 잘못된 first divergence를 고른다.

aggregate metric은 effective chunk histogram, limiting-reason count, mixed-path ratio와 graph fallback을 제한된 labels로 가진다. request ID와 raw prompt는 trace에만 둔다. long-prompt 구간별 P TTFT와 concurrent decode-count별 D ITL을 교차하면 priority가 어느 traffic 조합에서 손해를 만드는지 보인다.

관측 화면에서 requested size와 effective size를 같은 그래프에 겹친다. requested 8이 일정한데 effective distribution이 0,2,6,8로 갈리면 configuration drift가 아니라 runtime predicates의 결과다. 0은 candidate not selected, 2는 cache 또는 remaining balance, 6은 D reserve, 8은 full prefill처럼 reason과 조인한다. reason이 없는 0은 parked-loss 조사 대상으로 올린다.

commit lag도 별도 분포로 본다. device completion부터 cache/progress commit까지, commit부터 next-owner 등록까지 시간을 잰다. priority 자체가 빨라도 commit lag가 길면 P는 다음 candidate가 되지 못하고 D 사이에 끼울 기회를 잃는다. scheduler queue wait와 result-processing delay를 같은 TTFT 원인으로 뭉개지 않는다.

prompt logprob enabled/disabled, multimodal/text-only, prefix-hit/miss를 cohort로 나눈다. logprob 요청에서만 chunk duration이나 memory가 커지면 all-row logits와 accumulator가 원인 후보다. multimodal에서만 effective grant가 작으면 boundary/readiness를 본다. prefix hit에서만 score 누락이 생기면 KV progress와 scoring progress가 합쳐졌는지 본다.

trace sampling은 long P와 D concurrency가 높은 requests를 의도적으로 포함한다. random 1%만 쓰면 rare 20k-token prompt와 graph boundary 사건이 빠질 수 있다. privacy를 위해 raw content 대신 lengths, generation hash와 intervals를 기록해도 priority/commit 조사는 가능하다.

관측값은 설정과 실행 결과를 구분해 이름 붙인다. configured cap, planned prompt rows, committed prompt rows와 graph capacity rows처럼 단위를 붙이면 size 8 네 종류를 혼동하지 않는다. retry에서 counter가 두 번 증가하지 않도록 commit epoch를 기준으로 세고 planned-but-aborted rows는 별도로 남긴다.

alert도 first divergence에 가깝게 둔다. P age가 늘면서 candidate owner가 없으면 ownership alert, candidate는 있으나 grant 0이 반복되면 priority alert, device completion 뒤 progress가 움직이지 않으면 commit alert다. 단일 TTFT alert 뒤에서 세 장애를 한꺼번에 찾는 것보다 빠르다.

## 29.7 네 장애를 first divergence에서 닫는다

parked chunk 유실, mutable stale suffix, prompt logprob boundary double count와 prefill-first ITL spike를 각각 symptom, observation, competing hypotheses, first divergence와 recovery로 닫는다.

### 사건 1: 첫 chunk 뒤 P가 어느 queue에서도 보이지 않는다

증상은 traffic이 조용할 때도 P의 TTFT가 끝없이 증가하는 것이다. D1/D2 streaming은 정상이고 scheduler loop도 계속 돈다. cache used blocks는 P가 들어온 뒤 늘어난 채 내려오지 않는다. waiting request count에는 P가 없고 running decode count에도 포함되지 않는다. 사용자는 timeout을 받지만 GPU error는 없다.

첫 경쟁 가설은 fairness다. D requests가 계속 들어와 P가 우선순위에서 밀린다고 생각할 수 있다. 둘째는 KV shortage, 셋째는 multimodal encoder 미완료, 넷째는 transport cancellation 유실이다. 하지만 P가 candidate collection에 남아 있다면 최소한 매 epoch skip reason이 보여야 한다. 어떤 collection에도 identity가 없다면 priority를 조정해도 돌아오지 않는다.

plan/commit trace를 역추적한다. epoch 10에서 P는 `[0,8)` grant를 받았고 device completion과 cache length 8 commit이 모두 성공했다. materialization 함수는 remainder `[8,20)`를 반환했다. 그런데 scheduler persistent `chunked_req` owner update가 conditional branch에서 생략됐다. epoch 11 candidate snapshot에 P가 처음 사라진다. 이것이 first divergence다.

KV shortage 가설은 epoch 11에 free blocks가 충분했고 P가 allocator predicate까지 도달하지 않았으므로 탈락한다. encoder readiness는 epoch 9에 complete됐다. cancellation record도 없다. fairness는 존재하는 candidate 사이의 ordering을 설명할 뿐 존재하지 않는 request를 복원하지 못한다.

왜 crash가 나지 않았는지도 중요하다. current execution view는 정상적으로 release됐고 cache blocks는 logical request owner가 있다고 믿어 유지됐다. Python/C++ object reference 하나만 유실됐을 뿐 device pointer는 invalid access를 만들지 않았다. resource leak과 infinite TTFT가 결합된 silent lifecycle bug다.

복구는 누락 branch에 assignment 한 줄을 더하는 데서 끝내지 않는다. request가 admitted된 뒤 terminal 전까지 exactly one scheduling owner에 있다는 invariant를 둔다. owner는 current plan, parked chunk field, active decode collection, deferred terminal 중 하나다. commit transition은 current plan에서 parked owner로 원자적 ownership transfer를 한다.

회귀 fixture는 P first chunk completion 직전에 D arrival, cancellation과 cache pressure event를 교차한다. 각 epoch마다 P generation의 owner cardinality가 1인지 검사한다. 정상 completion이면 suffix intervals가 20까지 이어지고, cancellation이면 owner 0과 cache release가 같은 terminal chain에서 일어난다.

### 사건 2: stale suffix가 정상 KV 뒤에 붙는다

증상은 P가 단독 또는 non-chunked 실행에서는 안정적인데 chunked mode에서만 답변 첫 문장이 바뀌는 것이다. cache length, block count와 position IDs는 20으로 정상이다. 같은 random seed와 greedy sampling에서도 차이가 난다. attention kernel nondeterminism이나 graph padding을 먼저 의심하기 쉽다.

input audit에서 epoch 20의 first chunk `[0,8)` tokens는 ingress 당시 prompt와 같다. P가 parked된 사이 upstream middleware가 재사용 buffer에 다음 request tokens를 썼다. epoch 21 suffix view `[8,16)`은 같은 memory reference의 새 내용을 읽었다. logical offsets는 연속이지만 prompt content generation이 갈라졌다. first divergence는 scheduler가 아니라 ingress snapshot 이후 mutable storage를 request가 소유하지 않은 순간이다.

KV corruption 가설은 first eight positions의 cache-derived hidden parity가 맞고 suffix token IDs가 이미 다르므로 탈락한다. graph를 끄고 eager로 실행해도 같은 suffix가 들어간다. RNG state는 logits divergence 뒤에만 영향을 주므로 원인이 아니다. token content를 기록하지 않고 hash/generation만 기록해도 epoch 21 plan의 prompt hash mismatch를 찾을 수 있다.

복구는 ingress에서 immutable token snapshot을 request lifetime에 귀속하거나 buffer owner가 terminal까지 mutation을 금지하는 것이다. 각 chunk plan은 prompt generation/hash와 `[start,end)`를 함께 가진다. cache prefix도 같은 generation tag를 가진다. mismatch라면 old prefix 위에 새 suffix를 이어 쓰지 않고 request를 실패시키거나 old cache를 버린 뒤 처음부터 재시작한다.

adapter나 multimodal feature도 같은 위험이 있다. token suffix는 불변이어도 P가 parked된 사이 adapter mapping이 다른 request 값으로 재사용되거나 image feature scratch가 덮이면 prompt generation이 갈라진다. fixture는 token hash, adapter identity와 feature generation을 각각 바꿔 plan assertion이 device submit 전에 잡는지 본다.

### 사건 3: prompt logprob count가 chunk 수만큼 늘어난다

P prompt length는 20이고 API contract상 scores는 positions 1…19, 총 19개라고 하자. non-chunked 결과는 19인데 chunks `8,8,4`에서는 21개가 반환된다. usage prompt tokens는 20, KV length도 20이며 generated output은 정상이다. correctness test가 text만 비교했다면 놓치는 장애다.

각 chunk의 physical inputs와 committed score positions를 출력한다. 첫 chunk는 physical `[0,8)`, accountable `[1,8)`이다. continuation은 cache API 때문에 previous boundary token을 포함한 physical `[7,16)`이고 accountable은 `[8,16)`이어야 한다. 실제 result processor는 physical start 7을 사용해 `[7,16)`을 append했다. 마지막도 physical `[15,20)`, accountable `[16,20)` 대신 15부터 append했다. positions 7과 15가 중복돼 count가 21이다.

first divergence는 logits computation이 아니다. model은 boundary context를 포함해 올바른 logits를 냈다. result processor가 local output array index를 global accountable position으로 번역할 때 overlap을 빼지 않은 지점이다. tokenizer 가설은 prompt IDs와 contract count가 non-chunked에서 맞으므로 탈락한다. cache corruption도 generated text parity와 score values의 overlapping equality로 반증된다.

단순히 결과 list를 19개로 truncate하면 마지막 valid scores를 버리고 중복은 남을 수 있다. 복구는 plan에 physical input interval과 accountable score interval을 둘 다 넣고 result가 후자만 commit하게 하는 것이다. accumulator는 already committed end와 new start가 같은지 assertion한다. retry duplicate epoch도 idempotency key로 두 번 append되지 않는다.

값 검증은 count보다 강해야 한다. expected score positions set이 `[1,20)`을 정확히 덮고 각 returned token ID가 prompt의 같은 position과 맞는지 본다. chunks `1+19`, `7+7+6`, block boundary를 가로지르는 `5+5+5+5`, prefix hit 조합으로 overlap 길이를 바꾼다.

### 사건 4: prefill-first가 평균을 개선하고 streaming p99를 망친다

변경 전 dashboard는 P 계열 TTFT p95 900ms, D ITL p99 35ms였다. prefill-first와 chunk 8을 적용하자 P TTFT는 620ms, aggregate throughput도 6% 좋아졌다. 하지만 D ITL p99가 180ms로 상승하고 음성/interactive clients가 끊김을 보고한다. 평균 iteration latency만 보면 성공처럼 보인다.

request/iteration trace를 겹치면 spike는 long P continuation epochs와 일치한다. plan에서 P grant 8, D1/D2 grant 0이다. device duration은 12ms로 정상인데 P가 여러 chunks 연속 선택돼 D accepted output 사이에 10개 이상의 iterations가 끼었다. first divergence는 network send가 아니라 scheduler plan에서 ready decode가 처음 0 grant를 받은 epoch다.

GPU clock은 정상이고 kernel duration도 baseline 범위이므로 thermal/compute slowdown 가설은 약하다. output queue backlog도 없으며 sampling processor span은 짧다. cache miss가 D를 막았다는 가설은 D cache blocks가 resident이고 candidate readiness가 true라서 탈락한다. priority ordering 하나가 budget을 모두 소비했다.

복구 후보 A는 D1/D2 each one row를 먼저 reserve하고 P에 six를 주는 decode-first mixed plan이다. P는 `6+6+6+2`로 한 step 늦어지지만 D cadence를 지킨다. 후보 B는 prefill-first를 유지하되 max chunk를 4로 줄여 same iteration 남은 balance를 D에 주는 것이다. 그러나 scheduler가 remaining balance를 decode에 재사용하지 않으면 B는 no-op이거나 빈 capacity만 늘린다.

후보 C는 graph-friendly capacity와 phase path를 고려한 dynamic chunk다. active decode 2면 P 6, decode 6이면 P 2처럼 balance를 채운다. 이는 단기 grant mechanism이며 장기간 누구를 얼마나 기다리게 할지는 다음 장의 fairness 문제다. 여기서는 resulting intervals와 ITL을 측정한다.

검증은 P TTFT만 원복하지 않는다. 동일 arrival trace에서 P prompt-complete/first-output, D per-token ITL, valid/submitted rows, graph replay/fallback, workspace peak와 host gap을 비교한다. text, prompt logprob와 cache progress parity도 통과해야 한다. throughput 6% 이득이 사라져도 interactive SLO를 회복하는 선택이 서비스 목표에 맞을 수 있다.

### incident 네 개가 공유하는 조사 순서

첫째, user symptom을 P TTFT, D ITL, score count와 resource holding처럼 구체화한다. 둘째, request epoch별 plan interval과 commit state를 맞춘다. 셋째, first divergence 이전까지 정상인 owner를 제외해 competing hypotheses를 줄인다. 넷째, source의 mutation branch를 고정 commit에서 찾는다. 다섯째, 복구가 같은 fixture의 invariant를 회복하는지 본다.

parked loss는 scheduling ownership, stale suffix는 prompt generation ownership, double count는 accounting interval ownership, ITL spike는 budget grant ordering의 문제다. 모두 “chunk size가 이상하다”로 보일 수 있지만 고치는 knob가 다르다. size부터 바꾸면 P가 우연히 한 chunk에 끝나 symptom이 숨을 뿐 lifecycle 결함은 남을 수 있다.

### recovery를 배포 가능한 변경으로 만드는 순서

첫 단계는 최소 fixture에서 first divergence를 assertion으로 바꾸는 것이다. parked loss에는 owner cardinality, stale suffix에는 prompt-generation equality, double count에는 accountable interval adjacency, ITL spike에는 ready decode zero-grant event를 둔다. 장애의 최종 symptom이 아니라 처음 어긋난 state에서 실패해야 한다.

둘째 단계는 정상 path와 fault path를 같은 transition 함수로 모은다. remainder 등록, cancellation cleanup과 prompt completion이 각자 suffix/cache owner를 임의로 수정하면 branch 조합이 늘어난다. current-plan owner가 parked, decode-ready 또는 terminal owner로 이동하는 한 transfer API를 두고 generation과 interval을 검증한다.

셋째는 shadow 계산이다. production plan을 바꾸지 않고 alternative priority가 만들 grants와 graph capacities를 계산해 기록할 수 있다. D1/D2/P fixture를 실제 traffic에 일반화해 decode-first와 prefill-first의 counterfactual P completion epoch, D zero-grant count와 dummy rows를 비교한다. model을 두 번 실행하지 않아도 schedule cost 차이는 추정할 수 있다.

넷째는 제한 cohort다. text-only, prompt-logprob off에서 먼저 priority 변경을 검증한 뒤 logprob와 multimodal cohort를 연다. 이 순서는 feature를 영구 제한하자는 뜻이 아니라 서로 다른 commit 좌표의 first divergence를 분리하기 위해서다. 각 cohort가 interval and generation gate를 통과하면 범위를 넓힌다.

다섯째는 rollback 후 state 안전성이다. config를 되돌려도 이미 partially computed P requests가 old policy/effective size 아래 parked돼 있을 수 있다. chunk size가 바뀌는 것은 허용되지만 their committed prefix, score progress와 feature generation은 유지돼야 한다. process restart만 유일한 rollback이면 in-flight behavior와 client impact를 명시한다.

마지막은 post-change source map이다. 변경한 validator, effective config, scheduler consumer, commit owner와 metric link를 함께 갱신한다. option documentation만 고치고 result processor invariant를 남기면 다음 version에서 같은 regression을 반복한다. review에는 D/P timeline과 incident fixtures가 어느 함수 branch를 보호하는지 적는다.

### 무엇을 자동화하고 무엇을 사람이 읽는가

interval adjacency, owner cardinality, generation equality와 score coverage는 자동화하기 좋다. source가 만든 plan object와 result object를 model 실행 없이 구성할 수 있다면 unit fixture로 검사한다. graph/workspace actual performance와 user latency는 hardware/runtime 관측이 필요하므로 이 장의 정적 검토와 구분한다.

사람은 option semantics와 ownership boundary를 읽는다. warning 뒤 fallback이 의도인지, mixed restriction이 backend requirement인지, prompt completion timestamp가 어떤 SLO를 뜻하는지 code만으로도 문맥 판단이 필요하다. 자동 checker가 line link를 resolve한다고 설계 의도까지 증명하지는 않는다.

좋은 review artifact는 source anchor, 작은 numerical timeline, state diagram과 expected observations를 묶는다. P가 e0에서 6을 받고 e3에서 끝난다는 계산이 scheduler predicate와 연결되고, 각 completion의 cache/logprob state가 result processor와 연결되며, D timestamps가 SLO observation으로 연결된다. 독자는 어느 layer를 더 파야 할지 바로 알 수 있다.

## 29.8 priority를 workload 문장으로 설명한다

decode-first는 streaming cadence를 지키지만 P TTFT를 늘릴 수 있다. prefill-first는 P를 빨리 decode-ready로 만들지만 existing streams를 멈출 수 있다. 출구 기준은 prompt intervals가 빈틈과 중복 없이 20을 덮고 cache/progress/logprob generation이 같은 prefix를 가리키는 것이다.

### 한 장의 사건 기록으로 설계를 설명하기

운영 회의에서 chunked prefill을 켤지 물으면 option 장단점부터 열거하지 않는다. 실제 long prompt P 하나와 당시 active였던 D1/D2를 고른다. P arrival, D의 last accepted token, finalized scheduler state와 available budget을 첫 행에 적는다. 이후 epoch별 grants와 commit을 놓으면 추상 논쟁이 물리 사건으로 바뀐다.

e0의 effective state가 chunking true, mixed true, cap 8이라고 하자. D1/D2 each one row가 먼저 선택돼 P는 `[0,6)`을 받았다. selected graph capacity는 8이고 workspace는 grant를 더 줄이지 않았다. completion 뒤 cache/progress는 6, score interval은 API domain에 맞는 `[1,6)`, suffix owner는 parked P generation 7이다. 이 한 행에 validation, consumer, physical path와 mutation이 연결된다.

e1에서 P가 candidate snapshot에 없다면 cap 전에 parked ownership 장애다. candidate에는 있지만 grant 0이고 D가 balance를 다 썼다면 priority 결과다. grant 6이 plan에 있으나 cache/progress가 6에 머물면 commit 장애다. progress는 12인데 score end가 11이면 accounting 좌표 장애다. 같은 긴 TTFT가 네 first divergences로 나뉜다.

e3에서 P `[18,20)`이 끝나면 prompt-complete event, last-position logits eligibility와 first output commit을 잇는다. prompt complete가 cache length 20인지, TTFT timestamp가 token selection인지 transport emission인지 명시한다. D timestamps를 같은 세로선에 놓아 P 마지막 chunk가 cadence를 막았는지 본다.

이 기록은 framework에 묶이지 않는다. vLLM에서는 scheduler/runner state, SGLang에서는 adder/chunked request/result processor, Transformers에서는 policy/request remaining tokens/cache가 열을 채운다. 빈 열은 기능 부재가 아니라 source 왕복이 덜 끝났다는 표시다.

### 왜를 비용과 correctness 양쪽에서 답하기

왜 긴 prompt를 자르는가. 전체 20을 한 iteration에 넣으면 D1/D2 next tokens가 prefill 뒤까지 기다리기 때문이다. 왜 무조건 작게 자르지 않는가. chunks가 늘수록 scheduling, launch, metadata와 commit 횟수가 늘고 graph/workspace path가 불리할 수 있기 때문이다. 왜 decode-first가 항상 답이 아닌가. active decode가 budget을 계속 소모하면 P가 completion에 도달하지 못하기 때문이다.

왜 commit generation이 필요한가. physical row와 buffer가 재사용되는 동안 old completion이나 mutable suffix가 새 prompt에 붙을 수 있기 때문이다. 왜 KV length 하나로 부족한가. score, multimodal feature와 scheduler progress가 서로 다른 computability와 lifetime을 갖기 때문이다. 왜 logprob fixture가 별도인가. text와 KV가 정상이어도 boundary score가 중복될 수 있기 때문이다.

왜 graph enabled만 보지 않는가. 동일 grant라도 mixed eligibility, capacity와 dummy rows가 달라 wall time이 변한다. 왜 requested cap과 actual grant를 나누는가. D reserve, suffix, cache, multimodal atomic span과 workspace가 cap 아래에서 minimum을 만들기 때문이다. 이 이유들은 glossary가 아니라 source predicate와 mutation에서 나온다.

priority를 바꾸면 intervals와 boundaries가 바뀌고, boundaries가 바뀌면 cache/score/feature commit 횟수와 graph shapes가 바뀐다. TTFT option이 새로운 transition을 더 자주 밟게 하므로 silent bug 노출도 바뀐다. 성능 변경에 parity와 interval invariants를 붙이는 이유다.

### 독자가 자기 시스템에서 이어서 파는 순서

request trace 한 건으로 P/D timeline을 복원한다. aggregate dashboard에서 시작해도 request별 plan epoch와 intervals까지 내려간다. user option이 finalized state에 도달했는지 보고, scheduler consumer에서 first limiting predicate와 ordering을 찾는다. grant가 맞으면 runner/cache/result owner의 commit을 역추적한다.

source를 읽으며 prefill 이름의 모든 함수를 모으지 않는다. P generation을 key로 waiting object, schedule result, physical input, cache mapping, result interval과 next owner를 연결한다. D1/D2도 같은 epoch에서 grants와 outputs를 연결한다. 두 lifetimes가 budget 8에서 만나는 함수가 priority의 중심이다.

수치 fixture는 기본 `D1=1,D2=1,P=20`에서 block 4, graph 4/8, prefix hit 12, prompt logprob, multimodal span 5…12를 한 번에 하나씩 붙인다. 매 단계 effective intervals와 commit coordinates를 다시 쓴다. 한꺼번에 넣으면 어느 predicate가 grant를 바꿨는지 잃는다.

cap 8인데 grant 4면 option 무시를 주장하기 전에 balance/workspace/boundary를 본다. grant 8인데 TTFT가 같으면 runner path와 commit lag를 본다. TTFT 개선 뒤 ITL 악화면 D zero-grant epochs를 본다. text는 같은데 usage가 다르면 score intervals를 본다.

이 탐사법은 version이 바뀌어도 남는다. validator, effective field, consumer predicate, mutation과 observation anchor를 다시 연결하면 된다. source를 깊게 읽는 이유는 모든 줄을 암기하기보다 option에서 symptom까지 끊기지 않는 causal chain을 찾기 위해서다.

### 30장 앞에서 멈추는 정확한 지점

이 장은 e0에서 D1/D2가 먼저 두 tokens를 받고 P가 six를 받은 이유와, multimodal atomic span이 balance보다 커 grant 0이 되는 이유를 설명했다. 그러나 P가 몇 epochs 기다리면 D reserve를 줄일지, tenant priority와 deadline/age를 어떻게 비교할지는 결정하지 않았다.

30장에 넘기는 P에는 committed prefix, minimum legal next chunk, last progress epoch, queue age, cache/feature readiness와 zero-grant reason이 있다. D에는 last accepted output, current ITL age와 next legal grant가 있다. 어떤 age를 비교하고 reserve할지는 다음 장 몫이다. 이 장에서는 값이 정확한 commit epoch에서 갱신됐음을 보장한다.

planned grant와 committed progress도 분리한다. P가 우선권을 받았지만 execution이 실패했다면 progress credit으로 세면 안 된다. D가 plan에는 있었지만 cancellation됐다면 cadence state 갱신을 명시한다. fairness 입력이 틀리면 정교한 policy도 잘못된 request를 구제한다.

chunked prefill은 긴 prompt를 작은 tensor로 만드는 option이 아니라 긴 request의 계산과 state commit을 여러 iterations에 걸쳐 재계약하는 mechanism이다. decode priority는 그 계약 사이에 streaming lifetime을 끼워 넣는 순서다. 좋은 구현은 빠른 쪽을 고르는 데서 끝나지 않고 각 interval이 누구의 것인지, 무엇이 commit됐는지, 다음 owner가 누구인지 증명한다.

P가 20 tokens를 정확히 한 번 지나고 D1/D2가 의도한 cadence로 진전하며 KV, score, feature와 suffix generation이 같은 prompt를 가리킬 때 최적화가 닫힌다. 반복되는 우선권이 장기적으로 누구를 굶기는가는 30장의 질문이다.

마지막으로 deployment diff를 읽는 장면을 생각하자. 이전 version은 D-first였고 새 version은 running request ordering과 partially computed prompt placement를 바꿨다. option 이름과 default가 같아도 P grant timeline은 달라질 수 있다. upgrade review는 config diff에서 멈추지 않고 고정 fixture를 새 predicate에 대입해 `6+6+6+2`가 유지되는지, `8+8+4`로 변하는지 계산한다.

변화가 의도됐다면 release note에 사용자 효과와 state effect를 함께 쓴다. P TTFT 방향, D ITL 방향, mixed graph eligibility, parked owner와 score/feature commit coverage를 밝힌다. 변화가 의도되지 않았다면 source first divergence와 fixture assertion으로 regression을 닫는다. benchmark score 하나는 ordering semantic change를 설명하지 못한다.

운영 중 문제가 생겼을 때도 knob를 무작정 되돌리지 않는다. 이미 parked된 P requests의 committed prefix와 generation을 보존한 채 policy만 바뀔 수 있는지 본다. in-flight migration을 지원하지 않으면 drain/restart 경계와 client impact를 명시한다. rollback 자체가 stale suffix나 duplicate score를 만들면 원래 latency incident보다 심각하다.

독자가 기억할 최소 그림은 두 줄이다. 위 줄에는 D1/D2 accepted outputs, 아래 줄에는 P prompt intervals를 놓는다. 각 P interval 끝에 KV/progress/score/feature commit을 표시하고, interval 사이에 parked owner를 적는다. graph capacity와 workspace는 그 아래 physical track으로 둔다. 이 그림에서 빈 구간, 겹친 score, owner 없는 suffix와 D output gap이 즉시 보인다.

그림을 source 함수와 metric에 연결하면 chunk size tuning은 더 이상 경험적 숫자 놀이가 아니다. 어떤 predicate가 effective interval을 만들고, 어떤 owner가 commit하며, 어느 사용자 latency가 그 결과를 받는지 설명 가능한 engineering decision이 된다.

그 결정은 재현 가능한 문장이어야 한다. 동일 request arrivals와 finalized state에서 scheduler가 어떤 intervals를 만들었고, runner가 어떤 physical path를 택했으며, commit owner가 어떤 generation을 전진시켰는지 다른 독자가 다시 따라갈 수 있어야 한다. TTFT와 ITL 변화는 그 사슬의 끝에 놓인다.

이 사슬을 남기면 option을 바꾸지 않았는데 version upgrade로 ordering이 달라진 경우도 찾을 수 있다. 반대로 metric이 흔들렸지만 intervals와 commits가 같다면 batching 밖의 network, kernel 또는 traffic composition을 조사할 근거가 생긴다. 깊은 source 독해는 원인을 무한히 넓히는 일이 아니라 first divergence까지 탐색 범위를 정확히 줄이는 일이다.

## 29.9 option 문자열에서 effective chunk policy까지

`enable_chunked_prefill=true`나 `chunk_size=8192`를 보았다고 실행 의미를 안 것은 아니다. 첫 단계는 CLI/API parser가 값을 어느 config field에 쓰는지 찾는 것이다. boolean, optional integer, auto/default sentinel을 구분한다. 사용자가 생략한 값이 model length, scheduler token budget, GPU memory 또는 feature compatibility로 자동 결정될 수 있다.

둘째 단계는 validation과 normalization이다. requested chunk cap이 maximum batched tokens보다 클 때 clamp하는지 reject하는지, chunking disable과 priority policy 조합을 금지하는지, prefix caching·speculative·multimodal·pipeline parallel capability가 fallback을 만드는지 확인한다. startup log에는 requested와 effective 값을 둘 다 남긴다. parser default와 constructed scheduler field가 다르면 후자를 실행 좌표로 쓴다.

셋째 단계는 constructed component다. vLLM 고정 revision에서 scheduler config가 chunked-prefill enable과 token budget을 받아 scheduler branch와 request progress 계산에 도달하는 경로를 잇는다. partially computed request가 waiting/running 중 어느 collection에 남는지, decode running requests와 new prefill 후보를 어떤 순서로 budget에 넣는지 확인한다. option field를 읽는 `if` 하나보다 실제 scheduled token counts를 만드는 consumer가 중요하다.

SGLang 고정 revision에서는 server arguments가 scheduler configuration으로 전달되고 schedule policy와 new-prefill adder, chunked request 처리 경로가 실제 grant를 정하는 흐름을 잇는다. chunk size가 상한인지 고정 크기인지, available token budget·KV·mixed batch 조건 때문에 더 작은 grant가 가능한지 본다. priority 이름이 같아도 running decode reserve와 waiting prefill ordering을 어디서 적용하는지 분리한다.

넷째 단계는 runner input과 commit이다. scheduler가 P에 512 tokens를 grant했더라도 cache hit, atomic multimodal boundary, backend alignment 또는 runner failure 때문에 실제 committed interval이 달라질 수 있다. planned `[start,end)`, materialized rows와 committed prefix를 같은 plan generation으로 연결한다. scheduled count metric만으로 prompt progress를 단정하지 않는다.

다섯째 단계는 결과와 다음 owner다. chunk 실행 뒤 P가 어느 queue/collection으로 돌아가며 remaining prompt와 age가 보존되는지, final prefill 뒤 decode-ready로 전환되는지 본다. cancellation이나 allocation failure가 suffix generation과 partial KV를 어떻게 닫는지도 연결한다. 이렇게 parser→normalized config→constructed scheduler→grant branch→runner→commit→parked owner의 닫힌 사슬을 만든다.

option card에는 requested value, default provenance, normalized effective value, validator/fallback, constructed field, branch predicate, mutated grant/progress, physical shape, observation anchor와 falsifier를 둔다. “옵션이 활성화됐다”는 startup log만 있고 actual intervals가 없으면 아직 검증되지 않은 것이다.

## 29.10 긴 prompt와 active decode의 수치 timeline

한 iteration token budget을 256이라 하자. active decode D1~D32는 각각 next token 하나가 필요해 decode reserve32 tokens다. 새 prompt P는 2048 tokens이며 prefix hit가 0이다. chunk cap256이고 decode-first 정책이면 epoch마다 D에 32, P에 224를 줄 수 있다. P intervals는 `[0,224)`, `[224,448)`처럼 진행하고 마지막은 남은 32다. 총 10 epochs가 필요하다.

각 epoch wall time을 단순 모형 `L=L0+a·decode_rows+b·prefill_tokens`로 둔다. launch/metadata `L0=0.4ms`, decode row coefficient `a=0.03ms`, prefill coefficient `b=0.006ms`라면 mixed epoch는 `0.4+0.96+1.344=2.704ms`다. D의 이상적 ITL은 약 2.704ms이고 P의 prefill completion은 마지막 작은 chunk를 고려해 약 25ms 규모다. 이 식은 attention 길이와 kernels를 단순화한 비교 자다.

prefill-first로 P에 256을 모두 주고 D grant를 0으로 만들면 P는 8 epochs다. epoch 시간은 `0.4+1.536=1.936ms`, completion은 약 15.5ms다. P TTFT는 약 9.5ms 좋아지지만 D1~D32는 8 epochs 동안 output이 없어 최대 gap이 약 15.5ms 생긴다. 평균 throughput만 보면 prefill-first가 좋아 보여도 streaming ITL SLO가 5ms라면 실패다.

반대로 decode reserve를 너무 크게 잡아 P에 매 epoch32만 주면 D ITL은 짧아질 수 있지만 P는 64 epochs가 필요하다. epoch가 `0.4+0.96+0.192=1.552ms`라면 P completion은 약 99ms다. 새 decode arrivals가 reserve를 계속 채우면 P progress가 32보다 작아지거나 0이 되어 starvation으로 이어진다. cap을 작게 한 것이 아니라 effective residual grant가 작아진 것이다.

chunk overhead도 넣는다. 매 chunk마다 scheduler·metadata·commit 고정 비용 `h=0.15ms`가 추가되면 chunk count8은 1.2ms, count10은 1.5ms, count64는 9.6ms를 낸다. 작은 chunk는 D cadence를 보호하지만 P TTFT와 CPU overhead를 늘린다. graph bucket rounding과 cache metadata copy가 있으면 h는 일정하지 않으므로 bucket별 실측 계수를 쓴다.

prefix hit가 1024라면 remaining P work는 1024다. 동일 cap/priority에서 epochs가 절반으로 줄지만 cache hit 검증과 block table import가 추가된다. scheduler progress가 logical prefix1024에서 시작하는지, prompt logprob가 hit 구간을 계산해야 하는지에 따라 physical work와 visible contract가 달라진다. hit tokens를 scheduled tokens에 다시 넣어 TTFT를 과대예측하지 않는다.

## 29.11 TTFT와 ITL의 손익분기 계산

정책 A가 P에 224, D에 32를 주고 정책 B가 P에 128, D에 32를 주며 남은 96 budget을 일부러 비운다고 하자. B가 graph bucket128을 사용해 epoch 시간이 1.4ms이고 A가 bucket256으로 2.7ms라면 P throughput은 A82.96 tokens/ms, B91.43 tokens/ms로 오히려 B가 높다. grant가 큰 정책이 bucket/효율 때문에 느릴 수 있다.

P remaining2048에서 단순 completion은 A 약 10×2.7=27ms, B 16×1.4=22.4ms다. D ITL도 B가 1.4ms로 낫다. 이 구간에서는 작은 grant가 양쪽을 개선한다. 그러나 per-chunk overhead나 kernel efficiency가 달라 B epoch가 2.0ms라면 completion32ms로 A보다 나빠지고 D ITL만 좋아진다. break-even은 `ceil(P/cA)·LA = ceil(P/cB)·LB`다.

연속 근사로 chunk B가 유리한 TTFT 조건은 `LB/cB < LA/cA`다. D ITL 조건은 `LB≤ITL_target`이며 decode가 매 epoch 실제 grant를 받는다는 전제가 붙는다. 두 조건의 교집합을 찾는다. chunk cap 하나를 전체 traffic에 최적이라고 말하지 않고 prompt length·active decode·bucket cohort별로 계산한다.

사용자 가중 목적함수도 명시할 수 있다. `J=wP·TTFT_P+wD·p99_ITL_D+λ·overhead`다. interactive decode가 중요하면 wD가 크고 offline prompt ingestion이면 wP가 크다. 그러나 correctness와 starvation bound는 가중치로 상쇄하지 않는 hard constraint다. 일부 요청을 무한 대기시켜 평균 J를 낮추는 정책은 허용하지 않는다.

arrival을 포함하면 손익이 바뀐다. P가 여러 chunks로 parked되는 동안 새 decode가 계속 들어오면 active D가 32에서 64로 늘고 residual grant가 224에서 192로 줄 수 있다. static 계산 하나 대신 epoch별 `D_t`, `P_grant_t`, selected bucket과 duration을 trace replay한다. P completion까지 누적 grant가 exactly remaining work를 덮는지 본다.

## 29.12 priority·chunk 조합의 starvation incident

관측은 긴 prompt cohort의 TTFT p99가 30초를 넘지만 GPU utilization과 decode ITL은 좋아진 것이다. P requests는 admission됐고 일부는 첫 chunk까지 실행했으나 `prompt_remaining`이 줄지 않은 채 queue age가 증가한다. OOM, network와 tokenizer가 후보지만 first chunk 뒤 scheduler zero-grant epochs가 반복된다.

branch를 나눈다. option normalization에서 chunking은 true, cap256, priority는 decode-first로 확정됐다. scheduler는 active decode에 one-token grants를 먼저 주고 남은 budget이 minimum legal P chunk보다 작으면 P를 건너뛴다. traffic burst에서 active decode가 240이고 budget256, multimodal/attention boundary 때문에 P minimum chunk32라면 residual16은 사용할 수 없다.

매 epoch decode가 새 requests로 240 근처를 유지하면 P는 residual16을 계속 보고 0 grant를 받는다. cap256은 상한일 뿐 최소 legal32를 만족시키지 못한다. queue age를 priority key에 넣어도 decode reserve를 줄이는 branch가 없다면 P ordering만 앞서고 grant는 0이다. “priority가 적용됐다”와 “progress가 보장됐다”가 다른 이유다.

cause는 decode-first 자체가 아니라 age/zero-grant가 resource reserve를 바꾸지 않는 조합이다. scheduler trace에서 P가 매번 candidate 첫째인데 first false가 `residual < minimum_chunk`로 반복된다. allocator blocks와 runner capacity는 충분하다. P가 queue에서 사라진 것도 아니다. 이 증거로 leak·KV 부족 가설을 반증한다.

수정 후보는 N epochs 또는 age bound 뒤 decode reserve 일부를 P minimum chunk만큼 양보하는 것이다. 예를 들어 3 zero-grant epochs 뒤 32 tokens를 보장하면 D240 중 32 rows가 한 epoch 밀릴 수 있다. 해당 epoch duration과 ITL impact를 계산하고 SLO 안인지 검증한다. 무조건 prefill-first로 뒤집지 않는다.

fixture는 budget256, D counts223/224/225/240, P minimum32와 cap256, continuous decode arrivals를 둔다. P progress bound, D maximum gap, total token budget과 no duplicate commit을 검증한다. candidate ordering만 바꾸는 수정이 240 case에서 여전히 0 grant라면 실패다. grant32가 계획됐지만 execution failure면 progress credit을 주지 않는다.

rollback은 새 policy generation admission을 막고 parked P의 committed prefix/generation을 보존한다. inflight old policy plan을 drain하고 new ordering으로 재평가한다. policy를 되돌릴 때 P를 prompt 처음부터 재실행하거나 prompt logprob/KV를 중복 commit하지 않는다. D streams의 output cursor도 보존한다.

## 29.13 source·관측·회귀를 한 사건에 묶는다

source audit의 첫 행은 option definition이다. vLLM과 SGLang 각각에서 flag/size/priority가 선언된 parser symbol, default와 help만 적지 않고 destination field를 기록한다. 둘째 행은 normalized config constructor와 validation이다. 셋째 행은 scheduler가 그 field를 실제 읽는 branch다. 넷째 행은 scheduled token count와 request ordering mutation, 다섯째는 runner/cache commit과 next owner다.

revision은 tag 이름뿐 아니라 commit과 file/span을 고정한다. source card에서 caller/callee, input units, branch predicate, mutation, rollback과 next consumer를 잇는다. option 문서가 “chunked prefill을 사용한다”고 말하는 것은 사용자 계약 근거이고, 실제 grant 계산은 source 근거다. 둘의 역할을 바꾸지 않는다.

vLLM audit에서는 scheduler가 running requests와 waiting requests를 순회하는 순서, partially computed request의 computed/scheduled token 계산, long prefill token threshold와 budget clamp를 찾는다. scheduler output이 runner input으로 전달된 뒤 computed progress가 어느 acknowledgment에서 갱신되는지 잇는다. preemption/cancel에서 partial KV와 progress가 함께 처리되는지도 본다.

SGLang audit에서는 new prefill adder의 remaining budget, chunk decision, request 상태 mutation과 batch construction을 잇는다. priority policy가 waiting queue 순서만 바꾸는지 running decode budget에도 영향을 주는지 확인한다. normal/overlap loop에서 result completion과 next scheduling이 겹칠 때 old chunk result가 current generation을 갱신하지 않게 fencing되는지 본다.

metric은 requested cap, effective cap, planned/committed interval length, zero-grant reason, parked age, active decode count, decode reserve, selected graph bucket와 iteration duration을 둔다. request ID와 prompt content는 label이 아니라 sampled trace다. counters는 grants/zero-grants, gauges는 current parked/oldest age, histograms는 service interval과 commit lag다.

보존식은 P에 대해 `cache_hit + sum(committed_intervals) = committed_prefix`이며 intervals는 겹치지 않는다. remaining은 total prompt minus committed prefix다. D에 대해 generated, accepted, delivered cursors를 구분한다. iteration budget은 all planned grants 합이 limit 안이고 failure rollback 뒤 committed 합만 progress에 반영돼야 한다.

starvation alarm은 오래 기다린 request 수만 보지 않는다. eligible인데 minimum legal grant보다 residual이 작아 zero-grant된 연속 epochs, oldest eligible age, time since last committed progress를 본다. dependency blocked grammar/media request와 budget-starved request를 분리한다. 전자는 priority를 높여도 실행할 수 없다.

사건 재현은 deterministic scheduler fixture로 시작한다. D240을 유지하는 arrival generator, P2048/minimum32, budget256, decode-first와 age threshold를 둔다. model kernel 없이 grant sequence를 검증한 뒤 synthetic execution acknowledgment를 주입해 commit state를 확인한다. 그다음 실제 runner 통합 fixture에서 graph/workspace와 latency를 측정한다.

failure injection은 P chunk plan 뒤 KV allocation failure, execution failure, acknowledgment delay, cancel-before/after-commit을 둔다. P age는 plan만으로 reset되지 않고 committed progress에서만 갱신돼야 한다. 실패한 plan이 fairness credit을 먹으면 P는 실행하지 않았는데 다시 후순위로 밀린다. D output은 P rollback 때문에 중복되지 않아야 한다.

정상 regression은 chunking off와 full prefill, chunk cap 경계, no active decode, low/high decode, prefix hit, multimodal atomic span, prompt logprob와 overlap을 포함한다. 성능 수정이 text parity만 통과하고 logprob interval을 중복하면 실패다. multimodal feature range가 chunk boundary를 넘을 때 partial commit 계약도 유지한다.

성능 판정은 P prompt length cohort와 D active cohort의 2차원 표로 한다. 각 셀에 P TTFT, D p50/p99 ITL, useful prefill/decode tokens, zero-grant epochs, graph fallback과 CPU scheduling overhead를 둔다. 전체 평균은 긴 P starvation을 숨기므로 최대 progress gap과 tail을 hard gate로 둔다.

## 29.14 배포·rollback terminal과 다음 장 handoff

canary 전에 config diff를 작성한다. requested chunk/priority, normalized effective cap, minimum legal chunk, decode reserve rule, age threshold, graph variants와 model/backend capability를 비교한다. default가 바뀌면 사용자가 flag를 쓰지 않은 deployment도 변경 대상이다. startup effective state와 first schedule trace를 artifact에 남긴다.

canary workload는 P lengths256/2048/16384, D counts0/32/224/240, cache hit0/50%, chunk boundary와 continuous arrivals를 포함한다. 224와 225처럼 minimum32 residual 경계를 밟는다. 각 case에서 planned intervals, commit intervals, zero-grant reasons와 output cadence를 baseline/candidate로 비교한다.

승격 조건은 P가 bounded epochs 안에 progress하고 prompt tokens/KV/logprob/feature가 exactly once commit되며 D ITL hard limit을 지키는 것이다. target workload 가중 TTFT와 goodput도 개선돼야 한다. graph fallback, allocator residue와 scheduler CPU가 허용 범위에 있어야 한다. 평균 TTFT 하나로 승격하지 않는다.

rollback 조건은 P progress bound 위반, D p99 ITL 초과, interval overlap/gap, stale suffix, duplicate logprob, unreleased partial KV 또는 effective branch mismatch다. rollback을 결정하면 new policy admission을 fence하고 current plan generations를 drain한다. parked requests는 committed prefix와 next legal boundary를 snapshot한다.

old policy가 parked object representation을 다르게 기대하면 live migration을 강제하지 않는다. bounded drain 또는 explicit request terminal을 택하고 client impact를 기록한다. queue object를 그대로 옮길 수 있어도 age/priority credit과 failure count semantics가 호환되는지 확인한다. config rollback 성공과 request/resource reconciliation 성공을 분리한다.

readiness는 HTTP health만 보지 않는다. scheduler policy generation, runner/model ready, graph variants, KV allocator, output loop와 parked queue progress probe를 포함한다. restart 뒤 old generation completion이 new P progress를 전진시키지 않게 한다. stale completion counter가 0으로 수렴하고 oldest parked가 감소한 뒤 admission을 연다.

운영 dossier는 revision/config, option chain, P/D numerical timeline, source cards, decision traces, metric queries, failure matrix와 rollback result를 포함한다. “chunk를 128로 낮춰 해결”이라고 쓰지 않는다. 어떤 workload에서 어떤 bucket/interval이 바뀌어 P TTFT와 D ITL이 어떻게 움직였고 starvation invariant가 어떻게 닫혔는지 쓴다.

옵션 질문도 인과적으로 답한다. cap을 늘리면 무조건 P가 빨라지는가? selected bucket과 per-token 효율, D reserve와 workspace가 결정한다. decode priority를 높이면 D가 항상 빨라지는가? epoch duration과 graph path가 길어지면 아닐 수 있다. age를 넣으면 P가 반드시 진전하는가? residual이 minimum chunk를 못 채우고 reserve를 바꾸지 않으면 아니다.

독자가 만드는 마지막 그림은 세 track이다. logical track에는 P committed intervals와 D output cursors, policy track에는 candidate order·reserve·zero-grant reason, physical track에는 scheduled rows·graph bucket·duration을 둔다. plan generation과 commit acknowledgment로 세 track을 잇는다. 증상이 어느 track에서 처음 갈리는지 표시한다.

30장에는 priority의 장기 fairness 설계를 넘긴다. 이 장에서 넘기는 입력은 정확한 last progress epoch, eligible/blocked reason, minimum legal grant, committed prefix, D ITL age와 resource feasibility다. age나 progress가 planned state에서 잘못 갱신되면 어떤 fairness 알고리즘도 구제 대상을 잘못 고른다.

29장의 완료 조건은 옵션 목록을 외우는 것이 아니다. 임의의 긴 prompt와 active decode incident에서 requested option, effective field, scheduler branch, planned/committed intervals, physical epoch와 사용자 TTFT/ITL을 한 줄로 연결할 수 있어야 한다. starvation의 first divergence와 rollback terminal까지 연결되면 chunked prefill은 검증 가능한 scheduling mechanism이 된다.

**두 구현의 option trace를 나란히 적는 워크시트.**

첫 열은 사용자 입력이다. command line, Python config 또는 server argument에서 chunked prefill enable, maximum batched tokens, long-prefill threshold, scheduling policy와 priority 관련 값을 적는다. 생략한 필드도 default provenance를 기록한다. wrapper recipe가 값을 덮는다면 최종 process argv/config dump까지 따라간다.

둘째 열은 finalized config다. type conversion, `None/auto`, model maximum length, device/backend capability와 다른 option 조합 검사를 지난 effective 값을 쓴다. warning 뒤 fallback하는 경우 enabled request와 effective disabled를 모두 남긴다. validation error로 startup이 멈추는 경우는 runtime fallback과 구분한다.

셋째 열은 scheduler object field와 branch다. vLLM에서는 constructed scheduler가 token budget과 chunk-related threshold를 어디에 보관하고 partially computed request의 grant를 어느 loop에서 clamp하는지 적는다. SGLang에서는 scheduler args가 new-prefill selection과 chunking predicate, running batch priority에 전달되는 경로를 적는다.

넷째 열은 한 fixture의 숫자다. budget256, active decode240, P remaining2048, minimum legal32를 넣는다. candidate order가 P first여도 residual16이면 planned grant0인지, policy가 decode reserve를 224로 제한해 P32를 보장하는지 기록한다. branch를 읽었다는 주장을 실제 반환 count로 검산한다.

다섯째 열은 physical consumer다. grant32가 runner token rows32, positions interval과 block allocation으로 materialize되는지 본다. graph selector가 sequences/tokens 어느 bucket을 고르는지, mixed prefill/decode backend가 eager fallback하는지 적는다. scheduler 숫자와 kernel 시간 사이의 중간 경계를 생략하지 않는다.

여섯째 열은 commit과 park다. execution success 뒤 P committed prefix가 32 증가하고 remaining이 2016이 되는지, prompt logprob와 feature interval이 같은 경계를 쓰는지 확인한다. P object가 waiting/running/parked 어디에 놓이고 age와 last-progress epoch를 누가 갱신하는지 적는다.

일곱째 열은 실패다. allocation false, runner exception, cancel과 restart에서 planned grant가 progress로 잘못 세지 않는지 본다. partial blocks와 suffix buffer가 release되고 P 또는 client가 명시적 next owner/terminal을 갖는지 확인한다. 이 열이 없으면 정상 benchmark는 통과해도 운영 recovery를 설명하지 못한다.

두 구현의 행을 같은 이름으로 억지로 맞추지 않는다. vLLM의 request progress와 SGLang의 chunked request fields가 다른 구조여도 input unit, predicate, mutation, acknowledgment, next owner 좌표로 비교한다. 차이는 표의 중요한 결과다.

**TTFT/ITL break-even을 traffic 분포에 적용한다.**

앞의 단일 P 계산을 세 cohort로 늘린다. short P256은 cap224면 2 epochs, medium P2048은 10 epochs, long P16384는 74 epochs다. epoch2.704ms와 chunk overhead0.15ms를 합치면 대략 5.7ms, 28.5ms, 211ms다. 실제 마지막 chunk와 length-dependent attention을 생략한 근사임을 표시한다.

cap128 path가 epoch2.0ms라면 epochs는 2,16,128이고 시간은 4.3ms,34.4ms,275ms다. short에서는 작은 bucket이 이기고 medium/long에서는 chunk 수가 늘어 손해다. 하나의 cap이 모든 prompt에서 같은 방향을 만들지 않는다. prompt length histogram으로 가중하되 long p99 hard bound도 따로 둔다.

traffic이 short70%, medium25%, long5%라면 평균 근사는 cap224가 `0.7×5.7+0.25×28.5+0.05×211≈21.7ms`, cap128이 `0.7×4.3+0.25×34.4+0.05×275≈25.4ms`다. 평균은 224를 고르지만 short TTFT는 128이 낫다. routing/cohort-specific policy가 가능한지, 복잡도와 graph variants 비용을 검토한다.

D ITL은 epoch duration뿐 아니라 zero-grant epoch를 본다. cap224 mixed path에서 D가 매 epoch1 token이면 ITL 약 2.85ms다. prefill-only epoch를 주기적으로 넣으면 그 epoch 동안 D gap이 추가된다. fairness fix가 3 epochs마다 P32를 보장하면서 일부 D를 미룬다면 최대 D gap을 timeline에서 계산한다.

SLO가 P short10ms, medium50ms, long500ms이고 D p99 ITL8ms라면 두 cap 모두 단순 예제에서 가능하지만 starvation policy가 prefill-only20ms epoch를 만들면 D constraint를 깬다. average objective 전에 hard constraint로 제거한다. 남은 후보에서 goodput과 overhead를 비교한다.

실측 coefficient는 trace에서 구한다. 동일 bucket/phase cohort에서 planned tokens와 epoch duration을 regression하되 queue와 synchronization outlier를 분리한다. attention은 linear coefficient 하나로 정확하지 않으므로 context length, decode rows와 backend mode를 features로 둔다. 계산은 설명 가능한 첫 모델이고 최종 선택은 controlled replay로 검증한다.

**starvation 수정이 만든 새 tail 사고.**

age threshold 뒤 P에 minimum chunk를 보장하는 수정이 P progress를 회복했다. 그러나 D p99 ITL이 특정 epoch마다 20ms로 튄다. 관측상 P grant는 32뿐인데 epoch가 예상보다 길다. 단순 token 수로는 설명되지 않는다.

branch trace를 보면 P32와 D224의 combined sequences/shape가 graph capture 대상이 아니어서 eager mixed backend로 fallback했다. 정상 decode-only epoch는 captured bucket256으로 1.8ms였지만 fairness epoch는 eager20ms였다. first divergence는 priority가 아니라 grant 조합 뒤 selector branch다.

원인은 fairness 알고리즘이 minimum token count만 보장하고 physical path cost를 feasibility/priority cost에 넣지 않은 것이다. P32를 별 prefill-only로 실행해도 D gap이 생기지만, supported mixed bucket64나 chunk boundary를 선택하면 더 짧을 수 있다. 가능한 grant candidates의 predicted epoch duration과 D deadline을 함께 본다.

수정 후보는 P grant를 graph-eligible boundary로 맞추거나 해당 common mixed variant를 capture하거나 fairness epoch를 D deadline slack에 배치하는 것이다. graph variant 추가는 persistent memory를 늘리므로 27장의 combined budget 식을 다시 통과한다. latency 문제를 memory OOM으로 바꾸지 않는다.

verification은 active D223/224/225/240, P minimum/eligible boundaries, graph on/off와 fallback을 교차한다. P maximum progress gap과 D maximum ITL을 동시에 assert한다. global graph hit ratio가 높아도 fairness cohort fallback이 0인지 별도로 본다. P text/KV/logprob commit parity도 유지한다.

rollback은 fairness flag만 끄지 않는다. inflight mixed eager plan을 drain하고 parked P age/credit을 이전 policy semantics로 변환한다. 새 capture variant를 제거한다면 last replay consumer와 pool release를 기다린다. D/P output·progress generations가 reconciliation된 뒤 readiness를 연다.

**코드 리뷰 질문을 옵션별로 닫는다.**

chunk cap은 상한인가 목표인가 고정값인가. minimum legal chunk는 어디서 오며 cache block, multimodal span, graph alignment가 바꾸는가. actual grant0의 reason은 관측 가능한가. partially computed request는 다음 epoch에 어떤 queue와 priority key로 돌아가는가.

priority는 candidate order만 바꾸는가, decode reserve/token balance도 바꾸는가. age는 arrival, eligibility, last planned 또는 last committed progress 중 무엇에서 계산되는가. execution failure가 age를 reset하는가. tenant/deadline과 prefill/decode phase가 같은 comparator에 들어가는가.

token budget은 scheduled query tokens인지 total context인지, cache hit를 포함하는지 묻는다. mixed batch에서 decode one-token rows와 prefill rows를 같은 budget coefficient로 세는가. physical cost가 다르면 policy가 별 reserve나 weights를 사용하는지 본다.

runner acknowledgment 전에 progress가 mutation되는가. failure 때 old value로 안전하게 돌아가는가. chunk interval과 KV block table, prompt score와 multimodal feature cursor가 동일 generation을 쓰는가. cancel callback이 late completion을 commit하지 않게 하는가.

metrics가 requested/effective/planned/committed 네 값을 구분하는가. zero-grant reason, oldest eligible age, last progress와 selected physical path를 연결할 수 있는가. high-cardinality request ID는 trace에 두고 dashboard에는 bounded cohort를 쓰는가.

업그레이드에서 option 이름이 같아도 default, normalization, branch ordering과 grant unit이 바뀌었는지 diff하는가. pinned fixture의 interval sequence와 selected path를 전후 비교하는가. release note가 user latency와 state migration effect를 함께 설명하는가.

이 질문에 답하면 잘못된 처방을 거를 수 있다. P가 candidate first인데 0 grant라면 priority order를 더 올릴 문제가 아니다. grant는 있는데 commit이 없으면 scheduler가 아니라 execution/rollback을 본다. intervals는 같고 latency만 다르면 physical selector/kernel/network를 본다. first divergence가 조사 owner를 정한다.

**독자가 따라 하는 60분 조사 실습.**

첫 10분에는 production symptom을 한 cohort로 좁힌다. prompt length, active decode count, cache-hit 여부, model/config generation과 시간 창을 고정한다. 평균 dashboard에서 바로 option을 바꾸지 않는다. 느린 P 하나와 같은 시각 정상 P 하나, 영향을 받은 D 두 개를 고른다.

다음 10분에는 logical timeline을 그린다. epoch별 P planned `[start,end)`, committed prefix, remaining, queue/parked owner와 D generated/delivered cursor를 적는다. P가 candidate였지만 0 grant인지, grant됐지만 commit 실패인지 분리한다. D gap이 scheduler zero-grant인지 긴 physical epoch인지 구분한다.

세 번째 10분에는 config provenance를 걷는다. raw option, normalized value, scheduler field와 effective policy generation을 기록한다. startup log와 trace field가 다르면 constructed object를 source에서 확인한다. rolling deployment면 old/new config 요청을 섞지 않는다.

네 번째 10분에는 scheduler source로 내려간다. ordering comparator, token balance, minimum chunk predicate와 request progress mutation의 exact span을 연결한다. 수치 fixture를 현재 lhs/rhs에 대입한다. branch 조건을 만족한다고 상상하지 말고 trace의 active counts와 residual로 실제 결과를 계산한다.

다섯 번째 10분에는 physical path를 본다. scheduled rows, graph bucket/eligibility, workspace/backend, KV allocation과 epoch duration을 연결한다. scheduler grant가 동일한 passing neighbor와 path가 다른지 비교한다. fallback reason과 commit lag가 latency step과 같은 epoch인지 확인한다.

마지막 10분에는 cause·falsifier·수정·rollback을 한 카드에 쓴다. cause가 residual/minimum이면 priority order만 변경하는 수정은 falsified다. cause가 graph fallback이면 cap만 줄이는 임시 대응의 적용 cohort와 부작용을 적는다. 수정 fixture와 terminal ledger를 먼저 정한 뒤 canary를 설계한다.

실습 결과는 다섯 문장으로 요약한다. observation은 어떤 P/D cohort와 latency인가. branch는 어느 effective policy와 grant/selector path인가. first divergence는 planned, materialized, committed 중 어디인가. 수정은 어떤 invariant와 physical cost를 바꾸는가. rollback은 inflight P/D와 resources를 어떻게 닫는가.

**commit 좌표가 틀렸을 때의 별도 사고.**

P2048이 224-token chunks로 정상 progress하고 TTFT도 목표 안인데 prompt logprob count가 2272로 보고됐다고 하자. text와 KV length는 2048이라 단순 duplicate execution처럼 보이지 않는다. 각 chunk 경계의 overlap token을 score 계산에 포함하고 usage accumulator가 양쪽 interval을 모두 inclusive로 합친 것이 후보가 된다.

planned token intervals는 `[0,224)`, `[224,448)`로 겹치지 않지만 score interval이 첫 chunk `[0,224)`, 둘째 `[223,448)`처럼 boundary context token을 다시 포함할 수 있다. model 계산에 boundary token이 필요해도 API-visible score ownership은 정확히 한 interval이어야 한다. compute range와 commit range를 분리한다.

first divergence는 runner score tensor가 아니라 result processor가 compute range 전체를 visible accumulator에 append한 지점이다. KV와 scheduler progress가 정상인 이유도 설명된다. chunk size를 바꾸면 중복 횟수가 바뀌므로 latency option이 usage correctness를 드러냈다.

수정은 chunk result에 compute interval과 visible commit interval을 명시하고 generation별 score cursor로 중복을 거절한다. prefix hit, first/middle/last chunk, cancel/retry와 overlap late result를 검증한다. 합격 조건은 `sum(committed_score_intervals)=requested_visible_scores`이고 text/KV parity도 유지하는 것이다.

rollback에서는 이미 잘못 보낸 streaming logprob를 회수할 수 없는 API 계약을 고려한다. 새 traffic을 fence하고 inflight generation을 drain하거나 명시적 error terminal을 보낸다. server 내부 accumulator만 되돌려 client가 받은 값을 숨기지 않는다. billing/usage가 연동됐다면 정산 reconciliation도 필요하다.

이 사고는 priority 성능과 별개로 보이지만 같은 interval ledger가 해결한다. planned work, physical compute, semantic commit와 external delivery를 분리하면 chunk boundary optimization이 어느 계약을 바꿨는지 알 수 있다. P/D latency와 correctness를 같은 generation 표에서 봐야 하는 이유다.

**최종 승인 문장과 남은 미검증 조건.**

승인 문장은 “chunk256이 빠르다”가 아니다. “고정 revision/config에서 D0/32/224/240과 P256/2048/16384 fixture의 effective intervals를 확인했고, P progress bound와 D p99 ITL을 동시에 만족했으며, cache/score/feature commit과 rollback resource gap이 0이었다”라고 쓴다.

정적 source만 읽었다면 표현을 제한한다. parser→consumer mutation chain과 예상 branch는 확정할 수 있지만 실제 epoch coefficient, traffic histogram, graph fallback 빈도와 CUDA allocator peak는 실행 검증 항목이다. 가설을 완료형으로 쓰지 않는다. 각 미검증 항목에 fixture, metric과 중단 조건을 붙인다.

운영 trace만 있다면 반대 한계가 있다. 관측된 branch 한 개가 exception/restart/cancel 모든 source 경로의 안전성을 증명하지 않는다. pinned failure handlers와 rollback owner를 읽고 controlled injection으로 확인한다. source와 trace는 서로 대체하지 않고 같은 generation을 양쪽에서 지지한다.

최종 artifact에는 option provenance, source spans, numerical break-even sheet, starvation/tail/logprob incidents, regression matrix와 rollback record가 있다. 다음 담당자는 이 자료로 새 version의 predicate diff를 다시 계산하고, 증상이 같아도 first divergence가 달라졌는지 판정할 수 있다.

이 수준까지 닫혀야 chunked prefill은 “긴 입력을 나눈다”는 설명을 넘어선다. 독자는 chunk가 왜 필요한지, 언제 너무 작거나 큰지, decode cadence와 어떤 값을 교환하는지, priority가 왜 progress를 보장하지 못할 수 있는지, 그리고 실패를 어느 source owner에서 고쳐야 하는지 설명할 수 있다.

**마지막 배포 대조표.**

baseline과 candidate의 첫 행은 동일 workload fingerprint다. arrival, prompt/output length, prefix hit, multimodal/logprob 요구와 stop 조건을 고정한다. 둘째 행은 effective config generation, 셋째는 epoch별 P/D grants, 넷째는 selected physical path, 다섯째는 commit·delivery·release다. workload가 다르면 option 효과로 비교하지 않는다.

P TTFT는 arrival→first visible token을 쓰되 tokenization, queue, partial prefill epochs, final prefill-to-decode 전환과 transport를 나눈다. D ITL은 client delivery timestamps와 scheduler output timestamps를 함께 둔다. scheduler cadence는 정상인데 transport가 느리면 priority를 조정하지 않는다.

goodput 분자에는 정확히 commit·delivery된 유효 tokens를 넣는다. duplicate score, stale suffix, cancelled-after-generation tokens와 dummy graph rows는 제외한다. GPU utilization이 높아도 long P가 굶거나 D tail이 깨지면 성공이 아니다. P progress bound와 D deadline miss를 hard counters로 둔다.

config rollout은 cohort를 generation으로 나눈다. old parked P를 new policy가 이어받으면 committed prefix, age semantics, minimum chunk와 score cursor compatibility를 검사한다. 호환되지 않으면 drain한다. request ID가 같다는 이유로 state representation 호환을 가정하지 않는다.

rollback rehearsal은 실제 장애 전에 한다. admission fence, inflight plan drain, parked snapshot, policy swap, graph/pool handling, request resume와 readiness 순서를 문서화한다. 각 단계의 timeout과 실패 terminal을 둔다. rollback이 오래 걸릴수록 D/P SLO와 client retry 영향도 계산한다.

사후 metric은 rollout 뒤에도 유지한다. effective cap 분포, zero-grant streak, P progress gap, D ITL, graph fallback, interval commit mismatch와 resource residue를 config generation별로 비교한다. 회귀가 없는 기간에도 threshold fixture를 정기 실행해 default나 backend upgrade가 branch를 바꾸지 않았는지 확인한다.

최종 판정에는 반증된 가설도 남긴다. KV headroom이 충분해 allocation starvation이 아니었고, P가 candidate 첫째여서 ordering 누락도 아니었으며, residual16이 minimum32보다 작아 grant0이었던 사실처럼 조사 폭을 줄인 증거를 기록한다. 다음 사건에서 같은 dead end를 반복하지 않는다.

이 대조표를 채운 뒤에만 option 값을 recipe로 제시한다. recipe에는 적용 workload, source revision, effective normalization, expected grant sequence, TTFT/ITL 범위, correctness gates와 rollback 조건이 붙는다. 숫자 하나만 복사해 다른 model·GPU·traffic에 적용하지 않는다.

마지막으로 독자는 수치 하나를 스스로 바꿔 본다. budget256에서 active decode가 240이 아니라 223이면 residual33으로 minimum32를 만족한다. P는 매 epoch32를 받아 progress하지만 decode 한 row와 P32의 mixed shape가 어떤 bucket을 고르는지 다시 계산해야 한다. active224에서는 residual32로 경계에 정확히 걸리고,225에서는 31이라 0 grant다. 1 request 차이가 starvation branch를 바꾼다.

그래서 회귀 fixture가 223/224/225를 모두 포함한다. comparator나 `<=` 변경, reserved special token, alignment 하나가 경계를 이동시킬 수 있다. expected planned grant뿐 아니라 committed interval, selected bucket와 epoch duration을 함께 assert한다. 경계에서만 느려지는 경우 평균 부하 시험은 놓치기 쉽다.

옵션을 올리거나 내린 뒤 first false도 다시 기록한다. token residual을 해결하면 KV, workspace, graph eligibility 또는 multimodal atomic boundary가 새 false가 될 수 있다. 튜닝은 병목을 없애기보다 이동시키는 경우가 많다. 새 owner와 rollback cost까지 확인해야 변경이 끝난다.

29장의 결론은 균형 잡힌 한 숫자가 아니다. 동일 generation에서 option provenance, legal grant, physical epoch, semantic commit와 사용자 latency를 보존하는 절차다. 이 절차가 있으면 framework default가 바뀌거나 workload가 이동해도 독자는 다시 계산하고 반증하며 안전하게 되돌릴 수 있다.

최종 기록에는 담당 owner, 재현 fixture, 판정 metric, 허용 경계와 rollback trigger를 명시한다. 미검증 조건은 완료된 사실과 명확히 구분해 다음 실험의 안전한 시작점으로 남긴다.
