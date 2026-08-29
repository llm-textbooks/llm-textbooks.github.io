# 32장. 네 scheduler를 같은 사건으로 비교하는 법

기능표에는 네 칸 모두 “continuous batching”, “KV cache”, “cancel”이라고 적을 수 있다. 그러나 같은
요청을 넣었을 때 누가 queue를 소유하고, 어떤 predicate에서 막히며, computed progress와 late output을
어떻게 정리하는지는 다르다. 이름을 맞추는 비교는 바로 이 차이를 지운다.

이 장은 동일 workload I/D/P/C를 네 구현에 통과시킨다. I는 prompt 32, decode 16의 interactive
요청이다. D는 prompt 128 뒤 최대 2,048 decode를 생성 중이다. P는 prompt 16,384, decode 32의 새
long-prefill이다. C는 I를 취소한 뒤 같은 request ID로 retry한 요청이다. step query 상한은 512,
active request 상한은 4, free KV는 P 전체를 한 번에 담지 못한다.

각 구현에서 admission→budget→batch membership→KV transaction→model/output commit→finish/abort를
세로로 완주한다. vLLM의 running과 llama.cpp slot을 같은 가상 enum으로 만들지 않는다. SGLang
retraction과 Transformers offload를 같은 preemption box로 합치지 않는다. option 이름보다 owner,
predicate, mutation과 resource effect를 비교한다.

## 32.1 표를 펴기 전에 I/D/P/C를 실제로 도착시킨다

시계가 0을 가리킬 때 D만 실행 중이고 KV 128개와 지금까지 생성한 prefix를 소유한다. 곧바로 I가
도착한다. 다음 차례에는 512-token 예산 안에서 D의 decode 한 토큰과 I의 32-token prompt를 함께 넣을
수 있다. 이어 P가 도착하지만 16,384-token prompt 전체는 free KV에 들어가지 않는다. scheduler는 P를
거절하는 대신 512-token 이하의 chunk로 잘라 앞으로 보낼 수 있는지 판단해야 한다.

step 1 도중 I의 client가 연결을 끊는다. cancel 정리가 끝나기 전에 같은 request ID를 쓴 C가 다시
도착한다. 바로 이 순간이 네 구현의 차이를 드러낸다. 이전 I의 device output과 cache free가 늦게
도착해도 C의 output이나 새 KV를 건드리면 안 된다. 아래 네 절에서는 이 도착 순서와 숫자를 바꾸지
않고 vLLM, SGLang, Transformers, llama.cpp에 차례로 통과시킨다. 요약 표는 네 이야기가 끝난 뒤에만
펼친다.

### 열두 질문은 기능이 아니라 lifetime을 묻는다

1. admission 호출 owner는 누구인가.
2. waiting container와 ordering key는 무엇인가.
3. query-token과 request-count 상한은 어디서 교차하는가.
4. P의 미계산 suffix owner는 누구인가.
5. running membership과 device row identity는 같은가.
6. preemption/retraction에서 progress·KV를 보존하는가.
7. finish·abort·late output을 어느 generation/step identity로 가르는가.
8. 이전 output commit 전에 다음 step을 만들 수 있는가.
9. cache 실패가 queue/future reserve를 어떻게 바꾸는가.
10. graph shape 때문에 padding/bucket을 누가 추가하는가.
11. output 소비/backpressure가 admission/compute로 돌아오는가.
12. 어떤 metric/trace로 위 상태를 복원하는가.

질문에 모든 stack이 같은 field로 답할 필요는 없다. “없음”도 중요한 답이다. explicit per-request
priority가 없는 구현에 가상 priority를 만들지 않는다. graph padding이 runner owner라면 scheduler
field처럼 그리지 않는다.

### 첫 false predicate를 공통 좌표로만 표시한다

공통 좌표는 Q step work, S active ownership, K persistent allocation, W transient execution shape다.
native predicate의 의미를 잃지 않는 보조 축이다.

```text
candidate P
Q: remaining prompt와 step balance
S: 새 request/slot owner 확보
K: current chunk + reserve/block rounding
W: runner shape, logits/mask/workspace/graph bucket
```

P 전체 16,384가 K에 안 들어가도 chunk 512가 들어가면 admission될 수 있다. framework가 whole-request
reserve를 요구한다면 first false가 K가 될 수 있다. 같은 capacity 숫자라도 reserve policy가 달라
결과가 다르다.

### 네 이야기를 놓치지 않기 위한 공통 관찰점

```text
request logical ID + generation/epoch
arrival/admission owner
native queue/container + ordering key
candidate need/grant/skip reason
batch membership + device row mapping
KV old/new owner + allocation/free generation
model step ID + output row
CPU/request-state commit
terminal output/abort + resource free
```

request ID만으로 C를 I와 구분할 수 없다면 retry contract가 취약하다. generation/epoch 또는 old state
완전 제거 fence가 필요하다. source에 explicit epoch가 없으면 어떤 owner/order가 same-ID collision을
막는지 확인한다.

SGLang의 `max_running_requests`도 비슷한 이름이지만 worker profile/request pool과 PrefillAdder consumer를
통과한다. parked chunk/PP exception과 running batch size가 native semantics에 포함된다. 같은 4→8을
주고 active count가 달라졌다고 두 framework가 같은 상태 기계를 쓴다고 결론내리지 않는다. 효과만
I/P admission timing과 KV pressure라는 공통 workload 좌표에서 비교한다.

step 3에서 C first output과 old I late output이 동시에 host에 도착했다고 하자. output row order가
device completion 순서와 다를 수 있다. scheduled snapshot/step ID가 C row를 지정해야 한다. old I는
terminal cancel generation에 연결되어 drop되거나 old consumer에 전달돼야 한다. cache block free도
C가 이미 재할당받은 generation을 건드리면 안 된다.

ABA 검사는 block ID와 request ID 둘 다 필요하다. allocator가 block 7을 I에서 free하고 C에 재사용한
뒤 old free(block7)가 늦게 오면 duplicate free다. block generation/refcount/fence 또는 serialized
cleanup을 source에서 확인한다. pointer가 같다는 이유로 same ownership이라고 쓰지 않는다.

C ABA identity와 KV free는 high-risk, graph padding reason은 performance risk처럼 분류할 수 있다.
팀 workload와 규제에 따라 weight가 다르다. 이 기록은 만능 점수표가 아니라 왜 그 구현을 선택했는지
다음 사람도 재현하게 만드는 근거다.

fixture는 한 번에 모든 변수를 섞되 first divergence를 위한 단순 variants도 둔다. I/D만으로 ordering,
D/P로 chunk/KV victim, I/C로 ABA, P 단독으로 suffix/graph padding을 검증한다. full I/D/P/C는 interaction
regression이다. 단순 fixture가 없으면 full failure 원인을 분리하기 어렵다.

P prefix locality가 높은 workload에서는 SGLang/llama의 locality path와 vLLM/Transformers prefix cache
path가 서로 다른 owner에서 work를 줄일 수 있다. hit lookup, cache key, allocation과 scheduler order를
분리한다. cache hit가 order를 바꾸는 stack과 값만 줄이는 stack을 같은 “prefix scheduling”으로 묶지
않는다.

multi-tenant isolation에서는 cache sharing key와 adapter/model identity가 중요하다. I/P가 같은 token
prefix여도 tenant salt/adapter가 다르면 공유하면 안 된다. 비교 기록 K transaction에 cache
key identity를 추가한다. hit rate만 높이는 선택이 correctness/security를 해치지 않는지 본다.

마지막 보존 법칙은 네 stack 모두에 유용하지만 native 식으로 표현한다. admission된 request는 terminal,
cancelled 또는 여전히 owner collection에 있어야 한다. committed scheduled work는 output/progress update와
맞아야 한다. allocated KV는 live/refcount/free 합과 맞고, output은 정확한 task/generation consumer에
한 번 연결되어야 한다. 이 법칙을 만족하는 field를 각 stack에서 찾는다.

KV도 같고 I가 batch membership에 있는데 first output이 늦다면 5/7번으로 내려간다. device row와 output
commit snapshot, graph padding과 old C row identity를 본다. candidate order를 더 분석할 이유가 없다.
이렇게 old/new를 boundary별로 같음/다름 처리하면 source churn 속에서도 범위를 줄일 수 있다.

C row는 ID collision을 중심으로 쓴다. old I native object/task, cancel flag/status, outstanding batch IDs,
KV owner, result consumer와 new C object를 나란히 둔다. C admission 시점에 old owner가 남아도 safe할
수 있지만 result/free가 generation-aware해야 한다. 단순 `request_id unique` assertion이 어느 lifetime까지
적용되는지 본다.

graph 검토에서는 logical I/D/P membership과 physical execution rows를 분리한다. P 479 logical tokens가
512 bucket으로 padded될 수 있고 I cancel row가 invalid padding으로 남을 수 있다. output update는 logical
future state count만 소비해야 한다. pointer stability와 owner identity를 모두 확인한다.

fatal error도 C retry와 관련된다. old worker가 죽고 retry가 새 worker에 들어가면 old partial output이
gateway에서 늦게 도착할 수 있다. framework internal generation뿐 아니라 service-layer attempt ID가
필요하다. book scope 안에서는 internal output contract와 gap을 명시하고 외부 coordination 필요성을
선택 기준에 넣는다.

model coverage도 gate다. scheduler가 좋아도 필요한 architecture/quantization/adapter/backend가 지원되지
않으면 후보가 아니다. fallback path가 classic generation이라면 이 장의 continuous lifetime comparison이
적용되지 않는다. startup selected implementation을 증명한다.

P final output에서도 16,384 prompt가 어떤 chunks와 KV generations로 누적됐는지 합을 검산한다. D
preemption/offload/retraction이 있었다면 unique required work와 actual compute/transfer를 비교한다. I
cancel은 terminal/cancel output 수와 free가 정확히 하나인지 본다.

마지막으로 한 장짜리 incident 인계를 작성해 보자. 제목은 “P admission 뒤 C output 오염 의심”이다.
workload digest, 네 stack 중 실제 stack/revision, effective options, I cancel timestamp와 C admission,
step/batch IDs, block owner generations와 first divergent output을 첫 화면에 둔다. 기능 목록은 넣지 않는다.

first divergence가 output routing이면 KV/cache tuning을 하지 않는다. block generation부터 다르면 queue
priority를 의심하지 않는다. admission duplicate check에서 실패면 graph profiler가 필요 없다. vertical
trace는 investigation budget도 절약한다.

반대로 모든 identity/KV/output 경계가 맞고 C text만 다르면 model sampling/tokenizer/request inputs를
본다. scheduler 비교 장의 범위를 넘어간 원인을 인정한다. 가능한 가설을 계속 scheduler에 붙이지
않는 것이 정확한 디버깅이다.

업그레이드 승인 조건은 must-pass correctness, evidence coverage와 workload SLO다. C ABA/duplicate free,
lost terminal output와 cross-tenant cache key는 zero tolerance gate가 될 수 있다. I/P/D latency는 weighted
trade-off, raw throughput은 보조다. 이 우선순위를 selection matrix에 반영한다.

runtime validation을 나중에 수행할 때 source inspection에서 만든 expected event를 먼저 확정해 test가
결과에 맞춰 변하지 않게 한다. pass/fail threshold와 allowed nondeterminism을 기록한다. sampling text
동일성보다 greedy logits/output identity, owner counters와 lifetime invariant를 우선한다.

네 scheduler를 선택한 뒤에도 이 사건 기록은 다른 stack을 이해하는 공통 언어로 남는다. 장애 시 community
issue를 검색할 때 native symbol과 mutation을 정확히 말할 수 있다. “batch가 꼬였다”보다 “FutureRequestState
row와 cancel generation 인계에서 first divergence”가 훨씬 재현 가능하다.

hand calculation에는 가정과 단위를 붙인다. KV token-slot, physical byte, query token과 request count를
섞지 않는다. page rounding, TP-local heads, dtype와 prefix hit를 적는다. D victim cost를 token으로
비교하다가 Transformers swap byte와 직접 더하지 않는다. latency 또는 weighted debt 같은 공통 단위로
환산할 때 bandwidth/compute 가정을 표시한다.

마지막 확인은 C cleanup 이후 allocator와 output route다. old I/C native owners의 합이 0 또는 현재 C
하나로 수렴하고, old terminal/result가 정확한 consumer에 한 번 전달되며, freed physical state가 새
generation에서만 재사용되어야 한다. 이 invariant를 표현할 field가 없다면 selection document에
가장 높은 observability gap으로 남긴다.

이 비교 방식은 benchmark 결과가 예상과 다를 때도 유용하다. 선택한 stack이 평균 TTFT는 좋지만
C retry에서 오류가 난다면 평균 점수로 correctness gate를 덮지 않는다. P throughput은 낮지만 I/D
SLO와 cleanup이 안정적이면 workload objective에 따라 더 적합할 수 있다. 모든 결과를 하나의 순위로
압축하지 않는다.

결국 좋은 종합 비교는 정보를 줄이는 표가 아니라 차이를 잃지 않는 압축이다. I/D/P/C라는 동일 사건,
열두 질문과 vertical ownership 사슬이 그 압축의 기준이다. 독자는 제품 이름을 외우는 대신 새로운
revision에서도 같은 증거를 다시 만들 수 있다.
그 재현성이 있어야 선택 근거가 담당자의 기억이나 일회성 benchmark가 아니라 지속 가능한 기술
계약으로 남고, 다음 장애와 업그레이드에서도 다시 검증될 수 있다.
이것이 이 종합장이 남겨야 할 핵심 실무 능력이다.
함수 이름이 이동해도 request identity, budget, owner와 completion evidence를 같은 순서로 다시 연결해야 한다.
그때 비교는 기능표가 아니라 반증 가능한 lifecycle 모델로 남는다.

## 32.2 vLLM에서 I/D/P/C를 세로로 완주한다

API/engine ingress가 request object를 core scheduler에 보내면 scheduler의 waiting 또는 blocked
상태 queue가 lifetime을 소유한다. policy에 따라 FCFS deque 또는 priority queue semantics가 적용된다.
running D는 `Scheduler.running`, new I/P는 waiting에서 시작한다.

[`Scheduler.schedule`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L448-L638)은
running을 먼저 훑고 request별 need와 grant를 계산한다. D는 decode need, P는 computed frontier와
desired frontier 차이로 chunk가 정해진다. waiting admission은 running limit, blocked status, LoRA,
prefix/connector와 KV allocation을 더 본다.

### step 0: D 다음 I가 admission된다

Q=512에서 D에 1을 주고 511이 남는다. active limit 4라 I slot은 가능하다. I prompt 32와 필요한 KV
allocation이 성공하면 scheduled-new map과 running membership이 commit된다. I의 prompt가 짧아 이번
step에 전부 계산될 수 있다. P도 후보지만 remaining Q, active/KV에 따라 same step 또는 다음에 들어간다.

priority I가 vLLM native priority field로 정규화되었다면 queue comparison에 영향을 준다. 그러나
running D를 무조건 밀어내는 interrupt가 아니다. current step running-first와 non-preemptible previous
GPU work가 있다. priority effect는 waiting order와 allocation failure victim branch에서 source로 확인한다.

**P suffix owner와 chunk progress**

P가 admitted되면 request의 `num_computed_tokens`와 total tokens/spec placeholders 차이가 remaining work를
표현한다. 별도 “PREFILL_SUFFIX” 가상 object를 만들 필요가 없다. long-prefill threshold, Q balance와
model length가 grant를 자른다. 512 grant 뒤 computed frontier가 output reconciliation에서 전진하고
KV blocks는 cache manager가 request에 붙인다.

running membership과 device row는 같지 않다. running request가 PP cadence, blocked readiness 또는
token balance 때문에 이번 scheduled map에 없을 수 있다. GPU runner는 scheduler output을 cached
request state와 persistent batch row로 materialize한다. row mapping은 step output과 함께 추적한다.

**KV 실패와 priority preemption**

P current chunk allocation이 실패하면 scheduler는 victim을 골라 blocks를 free할 수 있다. FCFS 계열과
priority policy victim semantics가 다르다. victim이 이번 step provisional scheduled였다면 token/new
block/spec/encoder state를 rollback하고 budget을 환불한다.

[`_preempt_request`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L1274-L1310)는
victim KV와 encoder cache를 free하고 status PREEMPTED, computed frontier 0으로 만든다. pause가 아니라
recompute semantics다. prefix/remote cache로 회복되는 부분은 resume 시점에 달라질 수 있다.

**I cancel과 same-ID C retry**

cancel/abort가 scheduler에 들어오면 I의 request state, KV blocks와 runner/output ownership을 정리해야
한다. C가 같은 ID로 너무 일찍 들어오면 old late output이 C state를 mutation할 위험이 있다.
vLLM은 async/in-flight output에서 stale output counters/drop mode와 finished IDs/step reconciliation을
사용하는 경로가 있다. 상세 overlap은 앞 장을 반복하지 않고 ID+generation/step join이 안전한지 본다.

first divergence는 abort accepted→I removed/marked→KV free safe point→old output reconciliation→C admission
순서다. C가 duplicate로 reject되면 admission guard다. C가 들어갔지만 old I output이 C counters에
적용되면 identity bug다. request ID string 하나만 로그하지 않는다.

**output commit과 finish**

worker output은 scheduler의 scheduled request map/step과 맞춰 computed/output tokens를 갱신한다. EOS,
max tokens, abort에 따라 finish되고 blocks가 free/deferred된다. output processor/engine은 client-facing
result를 전달한다. scheduler running removal과 client delivery는 같은 사건이 아니다.

vLLM 원장에는 waiting/skipped/running membership, request status/frontier, scheduled token map, block
IDs/generation, preemption reset, runner row, stale/in-flight count, output step와 final free를 둔다.

Transformers model coverage와 custom model code, vLLM/SGLang optimized support, llama.cpp artifact/backend
생태계는 scheduler만으로 비교할 수 없다. model architecture가 supported attention/cache path를 가져야
비교 기록의 K/W semantics가 성립한다.

vLLM 검토에서는 `Scheduler.schedule`의 running loop, waiting admission과 `_preempt_request`, output update
caller를 하나의 call graph로 보존한다. queue implementation만 diff하면 computed reset/free 변화가
빠진다. gpu model runner의 cached request/batch row update가 scheduler output field를 어떻게 소비하는지도
링크한다.

## 32.3 SGLang에서 같은 사건은 Req·ScheduleBatch·chunk owner로 흐른다

SGLang scheduler manager는 ingress에서 `Req`를 받아 waiting queue에 넣고 `running_batch`를 오래
보존한다. [`SchedulePolicy`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_policy.py#L216-L330)가
waiting order를 계산하고 PrefillAdder가 prefix, token/KV reserve와 running count를 교차한다. D는
running batch, I/P는 waiting Req로 시작한다.

**I와 P의 waiting order**

SGLang native priority scheduling이 enabled되고 numeric direction이 resolve됐을 때 I priority가 order에
영향을 줄 수 있다. cache-aware LPM/DFS, FCFS/LOF/random/routing-key는 서로 다른 key다. tree cache가
disabled이거나 LPM queue가 큰 경우 active policy가 FCFS로 조정될 수 있다. requested policy string만
비교하지 않는다.

I가 first여도 PrefillAdder의 request/KV/token predicate를 통과해야 한다. prompt 32는 P보다 fit하기
쉽다. I를 add한 뒤 P는 chunked prefill limit과 remaining batch budget에서 일부 suffix를 얻는다.

**P suffix는 Req와 parked chunk state에 걸친다**

P original input, matched prefix indices와 extend/chunk progress는 Req와 scheduler의 `chunked_req` 같은
state에 반영된다. P 전체를 한 ScheduleBatch에 넣지 않고 chunk를 만들 수 있다. PP dynamic chunking이
있으면 requested chunk size와 iteration actual size가 다르다.

parked chunk는 일반 waiting request와 같은 ordering만으로 설명되지 않는다. 이미 시작된 chunk가
microbatch 경계를 넘어갈 때 running-request 상한을 기계적으로 적용하면 ownership을 잃을 수 있다는
source 주석이 있다. native field를 보존한다.

[`PrefillAdder` 생성](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L3222-L3268)은
max prefill, chunk size, running size, max running requests, prefix/backend tile과 `new_token_ratio`를 받는다.
batch membership은 policy order만 아니라 이 predicate 결과다.

**KV 부족은 decode retraction과 reserve mutation을 만든다**

running decode D와 I의 next token memory가 부족하면 batch `check_decode_mem`이 tree-cache eviction 후
allocator capacity를 본다. [`retract_decode`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_batch.py#L2816-L2895)는
retraction order에서 least-preferred victim을 빼고 request/token resources를 release한다. 최소 하나도
fit하지 않으면 마지막 request를 abort하는 경로가 있다.

retracted Req가 computed progress/KV를 어느 정도 보존하는지는 release/tree-cache/offload path를
호출 사슬로 확인한다. vLLM의 unconditional frontier=0 코드와 같은 box로 그리지 않는다. old/new
allocator available과 Req prefix/output state를 비교한다.

retraction 뒤 `new_token_ratio`를 남은 requests에서 다시 추정해 scheduler tracker가 갱신한다. 이는
future PrefillAdder reserve와 admission을 바꾼다. vLLM token-budget refund와 달리 future demand estimate
mutation이 명시적 비교점이다.

**I cancel과 C retry**

abort request는 waiting, running batch, chunked request와 output processing domain 중 어디에 있는지에
따라 cleanup이 다르다. scheduler normal/overlap loop는 abort를 batch state에 반영하고 cache owner를
release해야 한다. C same ID admission 전에 old Req mapping/session/tree cache와 pending result가
정리됐는지 본다.

overlap mode에서는 previous batch result와 next scheduling이 겹칠 수 있지만 상세 protocol은 반복하지
않는다. 비교 원장에는 batch ID, Req copy/original identity, abort epoch와 result application owner를 둔다.
old I output이 C에 적용되지 않는 first divergence를 확인한다.

**output commit**

model worker result는 last/current batch identity와 Req에 연결되어 output IDs, finish reason과 stream
response를 갱신한다. running batch filter/merge가 다음 membership을 만든다. finish/abort는 request-to-token
pool과 tree cache refcount/lock을 올바르게 풀어야 한다.

SGLang 원장에는 waiting index와 active policy, matched prefix, chunk owner/remaining, ScheduleBatch
membership, request/token allocator indices, retracted list/ratio, result batch ID, Req output/finish와 free를
둔다.

query budget option도 같은 원칙이다. vLLM max scheduled/batched tokens는 schedule balance와 grant,
SGLang max prefill/chunk는 PrefillAdder와 mixed batch, Transformers max batch tokens는 memory polynomial과
IO capacity, llama.cpp batch capacity는 ggml batch construction을 바꾼다. I/D/P에서 actual logical Q,
padded Q, step duration과 P suffix progress를 비교한다.

cache capacity option 역시 native allocator byte/layout을 따른다. vLLM blocks, SGLang token-to-KV pool,
Transformers num blocks/pages, llama context/KV cells를 logical token 수로만 맞추면 block rounding,
layer/head/dtype와 sharing이 빠진다. 같은 model/topology에서 rank-local bytes와 reserve를 계산하고
physical owner를 기록한다.

I/D/P/C의 step-by-step expected를 더 구체화한다. step 0 Q=512에서 D=1, I=32를 먼저 주면 P에 최대
479가 남는다. vLLM actual grant는 queue/policy와 current frontier, SGLang은 policy/PrefillAdder,
Transformers candidate status, llama batch/slot iteration에 따라 다를 수 있다. 합이 512 이하여도 order와
membership은 다르다.

vLLM에서는 request status, stale output/drop counters와 finished set/engine output identity를 source로
찾는다. SGLang은 Req/abort result와 batch copy/original을 찾는다. Transformers는 cancel queue,
FutureRequestState와 OutputRouter mapping을 찾는다. llama는 task/slot ID와 result route/final event를
찾는다. 네 칸에 “generation counter”를 가상으로 쓰지 않는다.

결과적으로 네 stack 모두 P가 실행됐다고 해도 future cost가 다르다. D recompute token, SGLang ratio와
retracted Req resume, Transformers H2D restore, llama evicted prompt future miss를 기록한다. 이 cost가
다음 step queue/budget과 latency에 어떻게 돌아오는지 비교 기록 9번에 쓴다.

old/new ingress와 native priority field가 같다면 1번 질문은 통과한다. waiting key가 old에는 I first,
new에는 cache-local P first라면 2번에서 divergence다. SGLang active policy fallback/priority composition
같은 source diff를 찾는다. queue order가 같다면 Q/S budget과 candidate predicate로 내려간다.

SGLang 검토에서는 scheduler normal/overlap entry, `SchedulePolicy.calc_priority`, `PrefillAdder`,
`ScheduleBatch.retract_decode`와 result processing을 잇는다. 이름이 같은 `Req`가 copy/original batch에서
어떤 owner로 쓰이는지 확인한다. ratio tracker update가 moved되면 future admission behavior도 diff한다.

비교 기록의 owner column은 최소 하나의 concrete object를 가져야 한다. “scheduler” 대신
`Scheduler`, `SchedulePolicy/PrefillAdder`, `ContinuousBatchingManager/RequestState`, `server_context/slot`
처럼 쓴다. 여러 owner가 인계하면 arrow와 commit condition을 쓴다.

vLLM incident라면 C request object가 scheduler waiting에 들어간 시점, I finished/preempted/aborted state,
old scheduled map과 stale output count, block free/reallocation을 붙인다. SGLang이라면 I Req와 batch copy,
abort processing, C Req mapping, request-to-token indices와 last/current batch result를 붙인다.

## 32.4 Transformers는 manager session에서 candidate와 FutureRequestState를 잇는다

Transformers continuous API의 admission owner는 `ContinuousBatchingManager`다. `add_request`가
`RequestState`를 만들고 bounded input queue에 넣는다. background thread가 processor를 만들고 scheduler
waiting map으로 넘긴다. public call 반환은 GPU admission 완료가 아니다.

[`ContinuousBatchingManager`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/continuous_api.py#L553-L1081)와
processor의 prepare→compute→update가 owner chain이다.

**FIFO와 PrefillFirst candidate에서 I/D/P 위치**

[`FIFOScheduler`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/scheduler.py#L331-L378)는
active DECODING을 first, active PREFILLING과 waiting을 second에 둔다. 이름이 global strict arrival
FIFO라는 뜻이 아니다. D가 first, I/P waiting은 second 후보가 된다.

PrefillFirst는 active split prefill을 first, decode를 second로 둔다. P가 이미 PREFILLING이면 D보다
앞설 수 있지만 새 waiting P를 무조건 first로 두는 것과 다르다. per-request interactive numeric
priority가 이 두 class source에 같은 방식으로 존재한다고 만들지 않는다.

**P suffix와 batch construction**

RequestState는 initial/remaining prefill, position offset와 generated length를 구분한다. scheduler가
token/cache/request budgets로 candidate를 처리하고 `FutureRequestState`를 만든다. IO packer는 selected
requests를 flat query tensor, cumulative lengths, cache read/write indices와 output row로 번역한다.

P query가 max batch tokens 512보다 길면 여러 step으로 나뉜다. remaining suffix는 RequestState에
남고 physical KV blocks는 cache allocator가 가진다. active membership과 device row는 같지 않다.
scheduled future list에 없는 active request가 있을 수 있다.

**KV allocation 실패와 offload retry**

candidate processing에서 allocation failure가 있고 아무 request도 scheduled되지 않으면 scheduler는
`None` sentinel을 반환한다. processor는 offloading manager가 victim을 offload하고 schedule을 재시도할
수 있다. vLLM frontier reset이나 SGLang retract와 동일하지 않다. CPU KV payload와 logical rollback을
어떻게 보존하는지 offloading source에서 확인한다.

memory handler는 startup에 max batch tokens M과 KV blocks N의 polynomial footprint를 함께 푼다.
runtime allocation failure는 current block occupancy다. graph/static IO W와 persistent K가 startup에서
교차한 뒤 scheduler의 current predicate로 내려온다.

**I cancel과 C retry**

cancel은 public queue에 ID를 넣고 event를 깨운다. 실제 scheduler cleanup은 safe loop 경계에서 waiting
또는 active map을 제거하고 GPU/CPU cache를 free한다. cancel 반환은 완료 fence가 아니다.

C가 같은 ID를 즉시 쓰면 OutputRouter callback/result queue, scheduler maps와 cache key collision을
살핀다. explicit generation epoch가 없다면 duplicate ID validation이나 cleanup-before-reuse contract가
필요하다. CLI non-streaming 경로의 completion callback registration race와도 구분한다.

**async IO와 output commit**

FutureRequestState가 model start 당시 row identity를 고정하고, device output D2H 뒤 update가 sampled
token을 RequestState에 commit한다. finish면 scheduler removal/cache free와 OutputRouter delivery가
일어난다. non-streaming은 중간 output을 전달하지 않아도 internal state는 전진한다.

Transformers 원장에는 manager input queue, scheduler waiting/active, RequestStatus/lengths, FutureRequestState,
IO pair/step row, block IDs/refcount, output commit, router terminal과 cache free를 둔다.

vLLM preempt reset path, SGLang release/tree/offload mode, Transformers CPU offload와 llama slot/context
operation은 서로 다른 cost다. 실제 configured path와 host/remote capacity를 확인한다. “supports
preemption” 기능표로 선택하지 않는다.

Transformers 검토에서는 manager public API, background loop, processor prepare/update, concrete scheduler,
cache allocator와 IO pair를 잇는다. scheduler class diff만으로 output/cancel safety를 알 수 없다. persistent
processor reuse에서 config snapshot이 old/new request에 어떻게 적용되는지도 기록한다.

D progress preservation이 weight 5이고 vLLM selected path가 recompute, Transformers가 CPU offload라면
workload prefix length와 PCIe bandwidth로 cost를 계산한다. 기능명으로 offload가 항상 낫다고 하지
않는다. host capacity/restore tail과 recompute GPU headroom을 비교한다.

Transformers라면 cancel queue enqueue/clear, scheduler active/waiting removal, FutureRequestState row, IO pair
epoch와 OutputRouter handler를 붙인다. llama라면 cancel task, old slot/task ID, final/partial result,
selected slot for C와 cache keep/discard를 붙인다. 같은 column names를 강제하지 않는다.

## 32.5 llama.cpp는 task queue와 server slot lifecycle로 읽는다

llama.cpp server의 admission owner는 task/control queue와 server context다. 요청은 slot을 얻어야 model
decode lifecycle을 시작한다. vLLM/SGLang처럼 동일한 paged request scheduler object를 가정하지 않는다.

[`server-context.cpp` slot selection](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1490-L1590)은
explicit slot ID, prompt LCP similarity와 LRU fallback을 사용한다. 이는 request priority fairness가
아니라 available execution slot과 cached prompt state의 placement다.

**I와 P가 available slot을 찾는다**

idle slot이 있으면 I/P task가 slot lifecycle을 시작한다. explicit slot ID 요청은 특정 slot을 고를
수 있고, 아니면 current prompt cache와 LCP similarity가 threshold를 넘는 slot을 찾는다. suitable
LCP가 없으면 least-recently-used available slot을 선택할 수 있다.

I와 같은 prefix를 가진 cached slot이 있으면 prefill 32 일부/전체를 재사용할 수 있다. P는 prompt가
길어 LCP absolute tokens가 커도 task-length ratio threshold에 따라 선택이 달라진다. selection reason,
LCP length와 kept tokens를 원장에 둔다.

LCP는 waiting service priority가 아니다. task queue에서 어느 task가 먼저 slot selection을 호출하는지
별도다. LRU timestamp는 request waiting age가 아니라 cached slot last use다. 네 scheduler 공통
fairness 표에 LCP/LRU check box를 넣지 않는다.

**P suffix와 batch token membership**

slot은 prompt processing progress, generated tokens, context/KV state를 소유한다. server `update_slots`
loop는 active slots에서 decode batch를 구성하고 prompt chunks를 처리할 수 있다. ggml batch의 token과
sequence IDs가 device graph input membership을 표현한다.

P 전체가 context/batch에 한 번에 들어가지 않으면 slot의 prompt progress가 suffix를 보존한다. 이
owner는 SGLang `chunked_req`나 Transformers FutureRequestState와 동일 객체가 아니다. slot state→batch
token/seq mapping→decode result를 세로로 추적한다.

### KV transaction과 slot replacement

slot이 cached prompt를 keep하면 KV prefix를 재사용하고 mismatch suffix를 제거/재계산할 수 있다.
LRU slot replacement는 old cached context를 희생해 새 task를 수용한다. running request를 priority
preempt해 waiting queue로 되돌리는 vLLM semantics와 다르다. available/idle slot selection 범위인지
active slot 취소 경로인지 source branch를 분리한다.

P가 context capacity를 넘으면 truncation/context-shift 또는 error policy가 관여할 수 있다. page allocator
retraction과 등치하지 않는다. selected context settings와 slot KV operations을 native source로 본다.

**I cancel과 C retry**

cancel/control task가 slot에 전달되면 slot generation을 중단하고 partial/final response, KV/cache 유지
또는 release를 처리한다. C same external ID가 들어왔을 때 old slot/task result가 new task routing과
섞이지 않아야 한다. server task IDs, slot IDs와 result queue identity를 구분한다.

같은 request ID가 application-level로 허용되는지, internal task ID가 새로 부여되는지 확인한다. old
partial result가 늦게 오면 response routing이 task generation을 구분해야 한다. explicit scheduler
generation field가 없으면 task object ownership/queue removal ordering을 본다.

**output commit**

decode result가 slot state/token sampler에 반영되고 partial streaming response가 output queue로 간다.
EOS/limit/cancel에서 final event를 보내고 slot은 idle/cache-reusable state로 전환된다. output final
전달과 slot reuse가 같은 시점인지 source ordering을 확인한다.

llama.cpp 원장에는 server task ID/external ID, queue event, slot selection reason/ID, prompt progress,
ggml batch seq row, KV kept/discarded range, decode iteration, partial/final response와 slot idle timestamp를
둔다.

llama.cpp server slot model은 단순 local deployment에 적합한 선택지가 될 수 있지만 vLLM/SGLang의
distributed scheduler/KV connector semantics와 동일 목표로 강제 비교하지 않는다. Transformers CB도
manager session과 TP 지원 경계가 있다.

Transformers `max_requests_per_batch`는 scheduler candidate와 static output/logits bookkeeping capacity를
제한한다. sequence row 증가가 FP32 logits/transient footprint를 키울 수 있다. llama.cpp slot count는
available execution contexts와 KV/context allocation을 늘리지만 core scheduler running request limit과
동일 option이 아니다. 더 많은 slot은 model/context memory와 task placement를 바꾼다.

output backpressure 비교도 세로 trace가 필요하다. I streaming consumer가 느리다. vLLM output processor/API
queue가 쌓이는지, SGLang detokenizer/IPC가 scheduler에 signal하는지, Transformers callback/result queue가
background loop를 block하는지, llama result queue가 task/slot generation을 pause하는지 확인한다.

fairness 비교도 native feature 범위를 지킨다. vLLM/SGLang priority가 있더라도 tenant weighted fairness/
aging guarantee와 같은지는 별도다. Transformers status group은 request tier가 아니다. llama LCP/LRU는
placement다. 공통 workload에서 I service time만 비교하고 기능 명칭을 통일하지 않는다.

반대로 prototype에서 model compatibility와 Python-level manager simplicity가 우선이면 Transformers
CB가 후보일 수 있고, local quantized deployment와 slot cache reuse가 목표면 llama.cpp가 후보일 수
있다. production suitability는 실제 model/support/status와 runtime 검증을 더 요구한다.

llama 검토에서는 server queue callback, slot selection, update_slots/decode, result send와 cancel control을
잇는다. LCP/LRU 함수만 읽으면 task ordering과 final slot cleanup이 빠진다. internal task ID가 외부
request ID와 어떻게 대응되는지 server route까지 본다.

D가 output 2,048을 생성하며 long-running일 때 fairness와 KV residence가 드러난다. vLLM/SGLang victim,
Transformers decode-first/prefill-first/offload, llama active slot occupancy가 I/P에 미치는 영향을 본다.
하지만 30장의 fairness 정의를 반복하지 않고 D service gaps와 victim debt라는 input metric만 쓴다.

예를 들어 I native tier가 weight 5인데 Transformers cell이 “status policy only”라면 낮은 점수보다
missing requirement가 핵심이다. upstream tier scheduler를 추가할 authority/cost를 평가하거나 다른
후보를 고른다. llama LCP에 priority라고 잘못 5점을 주지 않는다.

P throughput weight가 높으면 prefix cache hit와 suffix chunk뿐 아니라 policy CPU overhead, graph shape와
decode interference를 본다. SGLang locality active policy가 queue>128에서 fallback할 수 있고 llama slot
LCP는 available slot placement에 국한된다. workload queue/slot regime가 source branch를 자극하는지
runtime-required로 남긴다.

새 contributor가 option 비교표를 추가하려 하면 먼저 native consumer link를 요구한다. `max_batch_size`
이름이 비슷하다는 이유만으로 같은 row에 넣지 않는다. 하나는 request slots, 다른 하나는 query tokens,
또 다른 것은 slot count일 수 있다. owner와 unit이 같지 않으면 별 row로 나누고 workload effect에서만
교차한다.

## 32.6 동일 사건의 first divergence를 나란히 놓는다

네 vertical trace를 완성한 뒤에야 비교한다. 공통 virtual state machine을 만들지 않고 사건 경계별
owner와 native mutation을 나란히 둔다.

### I admission의 첫 차이

vLLM은 core scheduler waiting queue와 priority/FCFS key, SGLang은 Req waiting list와 active schedule
policy, Transformers는 manager input queue 뒤 scheduler candidates, llama.cpp는 task queue 뒤 available
slot selection이다. API call이 성공했다는 의미도 다르다.

I priority는 vLLM/SGLang native field/option 조건에서 order를 바꿀 수 있지만 Transformers FIFO/
PrefillFirst 두 class나 llama.cpp LCP/LRU에 같은 numeric tier가 있다고 가정하지 않는다. high I가
늦은 이유의 first divergence가 admission normalization인지, candidate order인지, slot availability인지
stack별로 찾는다.

### D running membership의 차이

vLLM `running`, SGLang `running_batch`, Transformers active RequestState와 llama.cpp active slot은 모두
오래 사는 계산 owner처럼 보이지만 device membership과 관계가 다르다. 각 step scheduled map,
ScheduleBatch/ForwardBatch, FutureRequestState/flat IO, ggml batch seq ID가 실제 device row를 정한다.

`running_count=1`을 네 metric에서 같다고 해석하지 않는다. streaming-wait/blocked, parked chunk,
offloaded state와 idle cached slot 포함 여부가 다르다. native definition을 비교 기록에 적는다.

### P suffix의 차이

vLLM은 computed frontier gap, SGLang은 Req prefix/extend와 chunk owner, Transformers는 remaining prefill
RequestState/Future snapshot, llama.cpp는 slot prompt progress를 쓴다. 같은 `remaining_tokens` field를
요구하지 않는다.

P가 512씩 진행한다는 output은 같아도 commit 지점이 다르다. worker output reconciliation, SGLang
batch result processing, Transformers host update, llama.cpp slot update 뒤에 progress가 durable한지
본다. schedule grant만 보고 suffix가 줄었다고 쓰지 않는다.

### KV failure의 차이

vLLM은 allocation failure에서 running victim preempt/recompute와 provisional refund를 할 수 있다.
SGLang decode retraction은 batch victim release와 future new-token ratio mutation을 만든다. Transformers
scheduler `None`은 processor offload/retry를 촉발할 수 있다. llama.cpp slot/context replacement는
available cached slot reuse/eviction owner다.

공통 “preemption supported” row는 무의미하다. 다음 표를 쓴다.

| stack | victim owner | computed progress | KV payload | future policy mutation |
|---|---|---|---|---|
| vLLM | running request | frontier 0 reset path | blocks free | queue/preempt counters |
| SGLang | decode batch Req | release path별 확인 | allocator/tree state | new-token ratio |
| Transformers | scheduler/offload victim | logical rollback 포함 가능 | CPU copy/restore | retry/current occupancy |
| llama.cpp | cached/selected slot | prompt keep/discard | context KV operation | slot LRU/cache state |

### I cancel→C retry의 차이

공통 위험은 ABA다. ID I의 old object가 free된 뒤 같은 문자열 C가 new object가 되고 old output/free가
늦게 적용된다. 해결은 explicit generation, unique internal task ID, deferred free fence 또는 duplicate
reuse 금지일 수 있다.

각 stack에서 cancel acceptance, native owner removal, device/in-flight result reconciliation, KV reuse와
C admission 순서를 적는다. first divergence가 old output identity부터인지 cache block generation부터인지
가른다. 동일 ID test가 허용되지 않으면 rejection 자체가 contract다.

### graph shape regularity의 차이

vLLM runner는 scheduler output과 persistent batch/graph dispatch shape를 맞춘다. SGLang runner/overlap
경로는 batch shape와 graph config를 소비한다. Transformers IO는 static M/N tensor와 graph key, async
pair를 명시적으로 소유한다. llama.cpp ggml graph는 current batch/slot shape로 graph를 만든다.

padding/bucket을 scheduler fairness 기능으로 쓰지 않는다. owner가 runner/IO일 수 있고 logical
membership과 physical padded row를 분리한다. graph hit 때문에 candidate가 skip되는지, selected batch가
padding되는지 source branch를 구분한다.

### backpressure의 차이

Transformers bounded input queue put과 OutputRouter/CLI callback, vLLM engine/API queues, SGLang IPC/output
channels, llama.cpp task/result queue는 각기 다른 boundary다. 느린 client가 compute admission을 실제로
pause/abort하는 연결이 있는지 source를 잇지 않고 “backpressure 지원”이라고 쓰지 않는다.

output queue가 bounded여도 upstream scheduler까지 await가 전파되지 않으면 GPU는 계속 생성할 수 있다.
request pause가 KV를 유지하면 capacity effect가 있다. owner/queue capacity/overflow action과 cancel
propagation을 비교 기록에 둔다.

P가 max model/context length를 넘는 경우도 비교 contract에 넣는다. admission reject, max-new clipping,
context shift 또는 chunk progress 후 eventual failure가 가능하다. chunking은 per-step work를 줄일 뿐
final context feasibility를 해결하지 않는다. 각 stack validation owner와 error delivery를 적는다.

capacity incident에서는 free KV 숫자를 네 stack에서 그대로 비교하지 않는다. allocator denominator,
page/block/context rounding, shared prefix, reserved/lookahead, offloaded payload와 session static memory를
정의한다. 동일 rank-local byte로 환산할 때도 native allocation failure reason을 보존한다.

이 수준으로 비교하면 이름이 다른 구현을 억지로 통일하지 않으면서도 독자가 새 scheduler를 빠르게
해체할 수 있다. 새 stack에서도 admission owner부터 terminal free까지 세로 trace를 만들고 열두 질문의
빈칸을 채우면 된다. 빈칸 자체가 위험과 추가 조사량을 보여 준다.

## 32.7 새 버전에서도 같은 사건을 다시 추적하는 법

### 1. admission owner

기입 항목은 public entry, internal request constructor, unique ID/generation 생성, queue put owner,
duplicate handling과 stop/backpressure predicate다. “API server”처럼 넓게 쓰지 않는다.

vLLM은 API/engine→core request/scheduler 전달 좌표, SGLang은 tokenizer/manager IPC→scheduler Req,
Transformers는 `ContinuousBatchingManager.add_request`→input queue, llama.cpp는 server task queue→context
dispatch를 고정한다. C same-ID가 reject/replace/new internal ID 중 무엇인지 적는다.

관측은 admission timestamp, native internal ID, queue/container와 rejection reason이다. source만으로
실제 external layer 설정을 확정하지 않고 callable path를 남긴다.

### 2. waiting container와 ordering

container type, key, blocked/skipped domain, tie-break와 dynamic fallback을 적는다. vLLM FCFS/priority
queue와 waiting/skipped merge, SGLang list+active policy/prefix/priority, Transformers scheduler waiting
order와 status candidates, llama.cpp task queue ordering을 분리한다.

### 3. Q/S budget intersection

config maximum, runtime effective balance, candidate demand, cap/skip reason과 committed batch 합을 적는다.
request-count가 scheduler active, runner slot 또는 available server slot 중 무엇인지 정의한다.

P need 16,384, step balance 511이라 grant 511인지, per-request chunk/alignment가 더 자르는지 계산한다.
D/I와 mixed될 때 sum과 active owner count를 함께 쓴다. option 이름 `max_batch`만 옮기지 않는다.

### 4. P suffix owner

native state와 durable commit을 적는다. vLLM computed frontier, SGLang Req/chunk prefix, Transformers
remaining prefill/Future state, llama slot prompt progress다. schedule grant, device execution과 host commit을
세 열로 나눈다.

cancel/preempt 때 suffix와 computed prefix가 어떻게 변하는지 적는다. frontier 0 reset, offloaded
payload, tree cache 보존과 slot cache keep를 같은 “resume=true”로 쓰지 않는다.

### 5. logical membership→device row

long-lived owner collection, current scheduled collection, runner batch cache, tensor row key와 padding row를
적는다. request ID가 device row index와 같다고 가정하지 않는다.

step 7 D row0, P rows1:512처럼 logical mapping과 graph padded capacity를 기록한다. 다음 step row가
바뀌어도 output commit은 scheduled step snapshot에 붙어야 한다. async 여부는 상세 protocol 대신
이 identity requirement로 감사한다.

### 6. KV transaction

allocator 호출, demand rounding/reserve, success mutation, failure sentinel, victim selection, payload
보존/삭제와 free fence를 기록한다. P current chunk block과 whole-request future reserve를 구분한다.

vLLM allocation failure→victim/refund/preempt reset, SGLang tree eviction/check/retract/ratio, Transformers
allocate failure→None/offload/retry, llama slot/context keep/discard를 native sequence로 쓴다. free block
count만 보지 않고 owner와 generation을 쓴다.

### 7. finish·abort·late output identity

step/batch/generation/internal task ID, terminal state mutation, external delivery와 resource free 순서를 적는다.
C retry에서 old I output이 어느 check로 버려지거나 routing되는지 source를 찾는다.

explicit generation counter가 없다면 unique internal object/task ID, duplicate admission guard, serialized
cleanup이나 deferred-free fence를 적는다. “safe”라고만 쓰지 않는다. 증명할 check가 없으면 gap이다.

### 8. 다음 step overlap 경계

이전 output commit 전에 다음 schedule이 가능한 option/path, future-state copy, stale result handling과
buffer owner를 적는다. 31장 정의를 반복하지 않고 comparison에 필요한 yes/no+owner만 둔다.

sync path라면 없음이 답이다. overlap path가 있어도 default/effective condition과 graph mode를 분리한다.
source inspection으로 deployment가 켰다고 주장하지 않는다.

### 9. cache 실패의 future effect

현재 request skip/preempt뿐 아니라 queue 위치, computed progress, future reserve estimator와 offload
mapping이 변하는지 적는다. SGLang ratio mutation은 중요한 예이고 vLLM token refund와 다른 axis다.

같은 KV pressure 뒤 다음 step에서 P grant/admission이 왜 달라졌는지를 native state로 계산한다.
failure count만 비교하지 않는다.

### 10. graph/static shape owner

graph key/bucket, padding decision owner, static buffer capacity, capture fallback과 logical used extent를
적는다. scheduler가 shape를 직접 제한하는지 runner/IO가 materialize하는지 구분한다.

I cancel 뒤 padding row old value가 C output으로 commit되지 않도록 valid rows와 snapshot identity를
확인한다. graph hit rate는 correctness 증거가 아니다.

### 11. backpressure chain

ingress queue capacity/put behavior, IPC/send queue, scheduler waiting cap, output callback/result queue,
slow-consumer action과 cancel propagation을 화살표로 쓴다. 하나가 bounded라고 end-to-end bounded라고
쓰지 않는다.

I client가 읽지 않을 때 compute pause, request abort, output drop 또는 memory growth 중 무엇이 되는지
source handler를 찾는다. pause가 KV를 유지하면 K cost를 적는다.

### 12. evidence surface

각 state를 복원할 existing metric/log/trace와 missing gap을 적는다. source field가 있어도 export되지
않을 수 있다. instrumentation 제안은 current capability와 분리한다.

최소 공통 event는 admission, queue enter/leave, candidate skip, grant, KV alloc/free/victim, runner
batch/row, output commit, terminal delivery와 cancel cleanup이다. content 대신 safe ID/length/shape를 쓴다.

### 한 행을 실제로 채우는 예

같은 template에서 Transformers mutation은 offload/retry가 될 수 있고 llama는 available slot selection이
될 수 있다. 빈 칸을 가상 공통 상태로 채우지 않는다.

### upgrade diff 절차

queue class가 같아도 victim key, reserve, output reconciliation이 바뀌면 semantic change다. release
note만으로 닫지 않는다. 반대로 파일 이동만 있고 owner/predicate/mutation이 같으면 behavior change로
과장하지 않는다.

### evidence confidence

각 비교 기록 cell에 source-confirmed, runtime-required, inference를 표시한다. function branch와 mutation은
source-confirmed다. 현재 deployment option/backend와 latency 효과는 runtime-required다. 여러 source를
합쳐 가능성을 추론하면 inference라고 밝힌다.

default option resolution, queue fallback threshold, blocked status merge, reserve ratio, chunk ownership,
victim sort direction, output snapshot copy와 free fence는 public API가 그대로여도 바뀔 수 있다. pinned
source의 function body와 tests를 본다.

metric rename만 따라가면 semantic change를 놓친다. 비교 기록 native state→evidence mapping을 업데이트한다.
새 field가 생겼다면 어떤 predicate/mutation을 운반하는지 적고 가상 공통 이름으로 숨기지 않는다.

cache failure가 아무 victim 없이 P를 waiting에 남기는 경로도 있다. work-conserving scheduler가 I/D를
계속 진행하면 P starvation 가능성이 있고, head-of-line break면 뒤 request도 막힐 수 있다. native loop가
continue/break/return None 중 무엇인지 읽는다. 기능표의 “preemption=false”만으로 queue effect를 알 수
없다.

slow consumer 때문에 request compute를 pause하면 I KV는 남아 S capacity를 쓴다. abort하면 blocks를
free하지만 partial text가 terminal semantics를 가져야 한다. output을 drop하면서 compute를 계속하면
GPU goodput이 낭비된다. 어느 policy가 실제인지 handler call chain 없이 일반화하지 않는다.

source link 품질도 감사한다. repository root가 아니라 pinned blob과 symbol line range를 사용한다.
line range가 함수 전체를 포함하지 않으면 caller/callee 링크를 추가한다. source comment와 code mutation이
충돌하면 code를 우선하고 comment drift를 gap으로 기록한다. tests가 semantic contract를 고정하는지도
본다.

새 version에서 source file이 이동하면 commit search로 symbol을 찾고 old 비교 기록 state transition을
대입한다. class 이름이 바뀌어도 admission owner와 mutation이 같을 수 있다. 반대로 같은 class 이름에서
queue order/victim/free가 바뀔 수 있다. textual rename과 semantic diff를 분리한다.

TTFT/ITL/throughput만 보고 owner semantics를 추론하지 않는다. 동일 metric 결과도 one stack은 recompute,
다른 stack은 offload를 거쳐 나올 수 있다. raw GPU work, transfer byte, cache hit와 useful output을 함께
본다. selected kernel/graph는 runtime-required evidence다.

분산 topology에서는 scheduler decision이 rank/replica에 전파되는 owner가 추가된다. vLLM engine/worker,
SGLang TP/PP manager, Transformers TP driver와 llama local server는 범위가 다르다. 같은 request set,
block mapping과 output identity를 rank가 합의하는지 source를 확장한다. 단일 process trace만으로 distributed
correctness를 증명하지 않는다.

비교 기록를 책의 appendix처럼 한 번 채우고 버리지 않는다. model revision, scheduler options, GPU/
topology와 deployment version마다 digest를 붙여 incident와 연결한다. C ABA나 P victim behavior가
바뀌면 어떤 source diff가 원인인지 곧바로 돌아갈 수 있다.

합이 맞지 않으면 성능 비교를 중단하고 lifetime bug를 조사한다. lost request, duplicate free, stale
output과 ghost slot은 tokens/s보다 우선이다. 비교 기록는 benchmark 표이기 전에 correctness
검토 도구다.

new P grant가 511인 이유가 step balance 증가인지 per-request chunk 변화인지 구분한다. config maximum,
effective initial Q, D grant와 P cap을 손계산한다. I가 first candidate였지만 KV allocation 32가 실패하고
P prefix hit 때문에 511이 fit했다면 divergence는 6번 KV transaction이다. “scheduler가 P를 우선했다”는
문장은 잘못이다.

pinned line anchor는 검토 시작점이지 영구 symbol locator가 아니다. commit이 바뀌면 old anchor를 그대로
링크해 new behavior를 설명하지 않는다. new commit blob link와 line을 생성하고 비교 기록 revision을
올린다. source excerpt는 핵심 branch 일부만 인용하고 전체 copyrighted file을 복제하지 않는다.

predicate column은 boolean expression이나 branch reason으로 쓴다. “capacity okay” 대신 token balance,
active count, KV allocate return과 readiness를 적는다. mutation column은 container membership, frontier,
block/refcount와 output state before/after를 적는다. effect column은 bytes/work/debt와 latency hypothesis다.

예를 들어 P row는 `Q_left=479`, `K_allocate(chunk=479)=success`, `running_count+1<=4`처럼 쓴다. mutation은
`waiting→running`, blocks assigned, scheduled q=479이다. effect는 new KV 479+rounding, prefill compute와
I/D service gap이다. “chunked prefill supported”보다 훨씬 검증 가능하다.

same-ID가 forbidden이면 retry library가 새 ID를 생성하고 idempotency key로 user operation을 묶어야
한다. framework를 바꾸면서 이 contract가 달라지면 application migration 항목이다. server 내부 semantics를
API gateway가 숨기지 않도록 error/retry documentation을 갱신한다.

backpressure 검토에서는 queue별 max/timeout/overflow를 표로 적는다. ingress bounded queue, scheduler
waiting unbounded, output bounded이면 slow consumer가 scheduler cancel로 역전파되는지 arrow가 필요하다.
없으면 middle unbounded domain에서 memory가 자랄 수 있다. queue 하나의 bounded flag로 service를
평가하지 않는다.

metric coverage gap을 patch할 때 observation이 scheduling을 바꾸지 않게 한다. full queue sort keys를
매 step log하면 CPU overhead와 privacy 문제가 있다. reason counters, sampled top candidates, safe ID
digest와 lengths를 쓴다. CUDA synchronize가 필요한 metric은 별 diagnostic run으로 분리한다.

네 stack metric 이름을 common exporter label로 normalize할 수 있지만 native definition을 metadata에
남긴다. `running_requests`가 runner-resident인지 scheduler-owned인지 label에 owner를 포함한다. common
dashboard는 convenience layer이지 semantics source가 아니다.

selection matrix의 score는 confidence로 할인할 수 있다. source-confirmed support 1.0, runtime-validated
0.9, inference 0.5, unknown 0처럼 임시 rule을 둘 수 있지만 숫자 자체가 객관적 진리는 아니다. 중요한
것은 evidence가 약한 high-weight criterion을 드러내는 것이다.

팀은 비교 기록 한 장에서 unresolved question owner와 deadline을 지정한다. source gap은 code reviewer,
runtime gap은 benchmark/observability owner, service retry gap은 API owner가 맡는다. unknown을 빈 칸으로
방치하지 않되 추측으로 채우지 않는다.

최종 review에서 열두 질문을 reverse로도 걷는다. terminal output C에서 어떤 model step/row, KV owner,
batch membership, grant, queue와 admission object로 왔는지 거슬러 간다. forward와 reverse mapping이
같아야 trace join이 완전하다. reverse에서 끊기는 지점이 observability gap이다.

D ITL 악화에서는 service opportunity 간격과 selected step duration을 곱해 본다. D가 매 step batch에
있는데 ITL이 느리면 model/output, 여러 step에 빠지면 policy/budget/eligibility다. `running=true`만으로
service를 받았다고 판단하지 않는다.

I TTFT 악화에서 waiting age가 늘면 admission/candidate, wait는 같고 first model start가 늦으면 batch/
non-preemptible work, model end는 같고 first output만 늦으면 commit/backpressure다. 같은 end metric을
vertical lifetime으로 분해한다.

검토 결과를 팀 사이에 전달할 때 source fact, hand calculation과 deployment observation을 세 색으로
나눈다. 예를 들어 “vLLM preempt path가 frontier를 0으로 만든다”는 source fact다. “D가 12,000 token을
다시 계산할 수 있다”는 prefix recovery를 모르는 상한 계산이다. “production에서 8,192 token을 실제
recompute했다”는 runtime observation이다. 세 문장을 합쳐 확정 사실처럼 쓰지 않는다.

SGLang ratio update도 마찬가지다. retraction 뒤 tracker가 mutation된다는 것은 source fact이고, 그
결과 P가 다음 step admission되지 않을 수 있다는 것은 predicate에 따른 inference다. 실제 P wait
증가는 trace가 필요하다. Transformers offload는 payload 보존 path가 있어도 deployment가 CPU pool을
사용했고 transfer가 성공했다는 관측이 필요하다. llama LCP source가 존재해도 selected slot reason은
runtime evidence다.

비교 기록 자체의 회귀 test도 가능하다. H2/H3나 링크 형식 같은 문서 gate뿐 아니라 열두 질문이 모두
적어도 one owner와 evidence/gap을 갖는지 검사한다. four stack section에 admission, budget, membership,
KV, output/cancel keyword가 모두 있는지 기계적으로 확인할 수 있다. 그러나 keyword 존재가 semantic
정확성을 증명하지 않으므로 code review를 대체하지 않는다.

장애가 재현되지 않을 때는 evidence gap을 숨기지 않는다. source상 race 가능 경계와 필요한 event,
현재 관측에서 확인된 마지막 동일 지점, 다시 발생하면 캡처할 safe fields를 기록한다. “문제 없음”과
“관측할 수 없음”은 다른 결론이다.

## 32.8 선택 기준은 workload의 owner와 실패 비용에서 나온다

이 비교의 source는 vLLM v0.27.1 `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang
v0.5.18 `71de97b264b04dcd514cf904003028aefe9775c8`, Transformers v5.15.1
`550d7b3834670483a4df436541272c055dc364bf`, llama.cpp v0.2.0 계열
`bb4caa7540188872173c44d161602d9271386413`에 고정했다. 이 장은 런타임을 실행하지 않았으므로
source가 증명하는 가능한 경로와 실제 배포에서 확인해야 할 관측을 구분한다.


비교를 실제 option 변경에 적용하는 예를 더 보자. vLLM에서 `max_num_seqs`를 4에서 8로 올렸다고
하자. parser/config validation 뒤 scheduler `max_num_running_reqs`가 8이 되고 waiting admission의 S
predicate가 달라져야 한다. I/P가 active로 이동하고 scheduled map/runner row가 늘어나는 것이 mutation
증거다. KV allocation이 먼저 false면 option은 effective지만 workload 결과는 그대로일 수 있다.

성능 비교는 동일 artifact/topology와 workload arrival를 요구한다. 네 framework가 지원하는 kernel,
quantization이나 model implementation이 다르면 scheduler 외 차이가 섞인다. scheduler comparison에서
forward kernel time을 별도로 정규화/분해하거나 같은 kernel이 불가능하다는 limitation을 밝힌다.

비교 기록를 실제 upgrade 검토에 적용하는 가상 예를 보자. old version에서 I는 step 1에 admission되고
P는 479-token chunk를 받았다. new version에서 I는 step 3까지 waiting이고 P는 511을 받는다. 최종
throughput은 늘었지만 I TTFT가 악화됐다. 첫 비교는 kernel이 아니라 admission 원장이다.

이 검산이 끝나면 네 vertical trace는 서로 다른 객체 이름을 유지하면서도 같은 workload 질문에
답한다. 비교는 abstraction을 만드는 일이 아니라 evidence를 정렬하는 일임이 드러난다.

비교 기록는 기능 유무에 체크하지 않는다. 같은 source revision에서 하나의 request generation을
세로로 연결하고, 각 화살표의 함수·field·predicate·mutation을 기록한다. version upgrade에서는 old/new
열을 나란히 채운다.

I가 P보다 앞이라는 expected comparator를 native key로 손계산한다. H3 같은 common priority field를
네 stack에 강제하지 않는다. llama LCP는 slot 선택 단계이지 task queue key가 아닐 수 있음을 명시한다.

```text
stack=vLLM rev=6e448...
event=P step=9
owner=Scheduler.schedule
queue=waiting policy=priority key=(p,arrival)
predicate=Q true,S true,K false
mutation=preempt D; refund q=1; D frontier->0; free blocks
batch=P q=511 row_range=[0,511)
output_commit=update step9
evidence=scheduled_map,block_event,preemption_counter,frontier
gap=actual recompute tokens not exported
```

첫째 pinned old symbol/lines를 새 revision symbol로 찾는다. 둘째 field rename보다 predicate/order와
mutation diff를 읽는다. 셋째 I/D/P/C expected trace를 새 source에 손으로 대입한다. 넷째 evidence
surface가 여전히 state를 복원하는지 확인한다.

이 분류가 없으면 source path를 읽었다는 사실이 실제 production 선택을 증명하는 것처럼 변한다.

version diff에서 자주 놓치는 변화도 기록한다.

option→validation→consumer→mutation→resource effect를 닫지 않으면 “512로 통일”은 무의미하다.
Transformers solver가 VRAM 때문에 384로 줄였거나 SGLang dynamic chunk가 256을 골랐거나 vLLM long
threshold가 P를 128로 자를 수 있다. startup effective state와 iteration actual grant가 필요하다.

evidence coverage를 수치화할 때 열두 질문 중 source-confirmed cell, runtime-exported cell과 missing cell을
센다. 예를 들어 12개 중 10개 source-confirmed라도 actual step ID/row mapping metric이 없으면 incident
reconstruction이 어렵다. 단순 10/12 점수보다 missing cell의 risk weight를 둔다.

실행 검증 단계가 허용될 때 사용할 fixture도 비교 기록에서 미리 설계한다. I/D/P/C lengths와 controlled
priority, tiny KV shortfall, same-ID cancel/retry를 만든다. greedy/fixed model input으로 output variability를
줄이고 request content 대신 sentinel IDs/counters를 관측한다. 이 장은 실행하지 않고 expected event만
쓴다.

failure recovery도 선택 기준이다. background/core worker fatal error에서 queued/active request가 terminal
error를 받는가, blocks가 회수되는가, manager/process restart가 필요한가를 본다. C retry가 자동으로
다른 replica에 갈 때 idempotency와 partial output semantics를 service layer까지 잇는다.

운영팀이 source를 patch할 수 있는지도 현실적 기준이다. 필요한 missing metric을 어느 owner에 넣을지,
upgrade conflict와 test maintenance가 얼마나 드는지 평가한다. benchmark 5% 차이보다 incident visibility가
중요한 조직도 있다. 선택 matrix weight에 반영한다.

최종 선택 문장은 조건부여야 한다. “I deadline과 native numeric priority, recompute 허용, distributed
topology가 중요하고 해당 pinned support가 검증되어 X를 후보로 선택한다”처럼 쓴다. “X scheduler가
가장 빠르다”는 source inspection만으로 말할 수 없다.

새 version이 dynamic graph bucket을 바꿔 P logical grant는 같지만 W/step duration이 달라질 수 있다.
scheduler semantics는 같고 performance divergence는 runner shape owner에 있다. 비교 기록가
owner를 분리했기 때문에 scheduler regression으로 오판하지 않는다.

C correctness는 점수 trade-off가 아니라 must-pass gate로 둘 수 있다. same-ID retry isolation을 증명할
수 없으면 performance score와 무관하게 배포 보류다. must-have와 weighted preference를 selection matrix에서
분리한다.

성능 incident 인계도 같은 구조다. P TTFT가 new version에서 2배가 됐다면 admission wait, actual chunk
grants/service gaps, prefix hits, KV failures/victims, runner step duration과 output commit overhead를 분해한다.
첫 변화가 runner duration이면 scheduling order가 같아도 graph/kernel regression일 수 있다.

비교 기록 review 질문은 “어느 stack이 이 feature를 가지는가”가 아니라 “우리 workload의 이 사건을
어느 owner가 어떤 mutation으로 닫는가”다. feature가 있어도 evidence가 없거나 model path가 소비하지
않으면 선택 근거가 약하다. feature가 없어도 upstream service가 requirement를 충족할 수 있지만 owner와
failure semantics를 추가로 감사한다.

문서가 오래되었는지 확인하는 신호도 둔다. pinned revision과 deployed binary digest 불일치, source
symbol 미존재, option effective log 불일치, metric definition change와 expected I/D/P/C trace mismatch다.
하나라도 있으면 비교 기록 status를 stale로 표시한다.

관측 hook이 없는 cell은 test-only assertion 또는 sampled debug build로 검증할 수 있다. production
always-on logging을 무리하게 늘리지 않는다. hook 추가 patch도 pinned source owner 근처에 두고 upgrade
diff에 포함한다.

성능 추천도 조건부 문장 template을 쓴다. “이 source revision에서 X path는 P suffix를 chunk로 보존하고
Y mutation으로 KV pressure를 처리한다. 우리 workload에서는 Z evidence를 수집한 뒤 후보로 평가한다.”
source inspection만으로 “X가 더 빠르다”를 쓰지 않는다.

또한 framework upgrade와 scheduler option 변경을 별개의 change set으로 검증한다. upgrade에서 owner나
mutation이 바뀐 상태로 policy까지 동시에 바꾸면 old/new divergence를 귀속하기 어렵다. 먼저 동일
effective option으로 비교 기록 parity를 확인하고, 그다음 policy/capacity를 바꾼다. model artifact와
CUDA/backend 변경도 가능하면 분리한다.

네 구현을 한 줄 순위로 정할 수 없다. 필요한 API/model coverage, deployment topology, request lifetime,
cache policy, observability와 운영팀의 source ownership이 다르다. I/D/P/C에서 가장 비싼 실패가 무엇인지
먼저 정한다.

interactive I의 deadline이 최우선인 경우부터 본다.

per-request priority가 ingress부터 scheduler comparator와 victim까지 실제로 연결되는지 본다. vLLM/SGLang은
native priority option/path가 있지만 numeric sign, active policy, eligibility와 victim scope를 확인해야
한다. Transformers FIFO/PrefillFirst나 llama slot LCP/LRU를 같은 tier 기능으로 계산하지 않는다.

priority만으로 current non-preemptible step을 줄일 수 없고 KV/adapter가 없으면 I가 들어가지 못한다.
service layer admission/reservation, chunk quantum과 capacity를 함께 설계한다. 비교 기록의
I trace에서 queue order뿐 아니라 first model start와 output commit을 본다.

긴 D의 ITL과 progress 보존이 중요한 경우는 기준이 달라진다.

memory pressure에서 victim computed progress를 reset하는지, prefix cache로 회복하는지, swap/offload로
payload를 보존하는지 비교한다. D 12k/100k context의 recompute debt가 workload goodput을 지배할 수 있다.

긴 P의 TTFT와 throughput이 중요한 경우도 별도다.

suffix owner, chunk policy와 prefix locality를 본다. SGLang cache-aware policy/dynamic chunk, vLLM computed
frontier scheduling, Transformers active prefill status policy, llama slot prompt reuse가 P progress에
미치는 경로를 native trace로 비교한다.

P 전체 fit을 요구하는지 current chunk만 allocation하는지, decode와 mixed될 수 있는지, prefix hit가
Q/K를 얼마나 줄이는지 계산한다. throughput을 위해 P를 앞세웠을 때 I/D SLO cost를 guardrail로 둔다.

C cancel/retry correctness가 중요한 경우에는 identity가 우선이다.

unique internal ID/generation, duplicate ID validation, stale output reconciliation와 block generation을
가장 먼저 감사한다. high QPS API에서는 timeout retry가 흔해 ABA 위험이 우선이다.

framework default가 same ID reuse를 거부한다면 application은 새 ID/idempotency mapping을 써야 한다.
허용한다면 old cleanup fence를 증명한다. metrics가 old/new를 구분하지 못하면 운영 위험으로 기록한다.

model/API compatibility가 우선인 경우 scheduler 비교는 그다음이다.

기능 호환성을 확인한 뒤 scheduler comparison을 적용한다. unsupported feature를 fallback classic
generate/other runner로 보내면 lifetime이 완전히 달라질 수 있다. selected path를 startup/trace로
증명한다.

단일-node 단순 운영과 대규모 분산 운영도 같은 기준으로 점수화하지 않는다.

분산에서는 rank/replica control-plane agreement, PP/async identity와 remote KV failure가 비교 기록에
추가된다. 단일-node에서는 그 복잡성이 없는 대신 slot/context capacity와 model format 제약을 본다.

observability와 source 유지 비용도 선택 기준이다.

필요한 state가 metric으로 노출되는지, trace hook을 안전하게 추가할 수 있는지 평가한다. high performance
구현도 queue/victim/output generation을 복원하지 못하면 incident MTTR가 커질 수 있다. source churn과
upgrade 비교 기록 비용을 운영 cost로 본다.

metric 이름 수가 아니라 열두 질문을 답할 evidence coverage를 score한다. content privacy와 overhead도
포함한다.

선택 matrix는 가중 evidence로 만든다.

```text
criterion                 weight   vLLM  SGLang  Transformers  llama.cpp
I native tier path          ...    evidence 링크/공백
D progress preservation     ...
P suffix/locality           ...
C generation isolation      ...
model/topology support       ...
evidence coverage            ...
team maintenance cost        ...
```

점수만 남기지 않고 각 cell에 pinned source와 runtime gap을 붙인다. weight는 workload/SLO에 따라
결정하며 universal winner를 만들지 않는다.

같은 사건의 종합 timeline을 마지막으로 겹친다.

step 0 D running에서 I/P admission을 시도한다. vLLM은 running D grant 후 waiting I/P를 token/slot/KV
교집합으로 올린다. SGLang은 running batch와 PrefillAdder policy에서 I/P를 고른다. Transformers FIFO면
D first candidate 뒤 waiting I/P, PrefillFirst는 P가 active continuation이 된 이후 status order가 바뀐다.
llama는 available slots와 task queue에서 I/P가 각 slot을 얻는다.

step 1 I cancel 후 C retry가 온다. 네 stack 모두 external ID string만으로 old/new를 합치면 안 된다.
vLLM output reconciliation, SGLang batch result/Req identity, Transformers FutureRequestState/OutputRouter,
llama task/slot result routing에서 late output boundary를 찾는다.

step 2 P chunk allocation이 pressure를 만든다. vLLM은 victim preempt/reset 가능, SGLang은 decode retract와
ratio update, Transformers는 allocation None→offload/retry, llama는 cached slot/context 선택/eviction
경로를 가진다. “P가 실행됨” 이후의 debt가 다르다.

step 3 output commit에서 scheduled membership과 host/request state를 맞춘다. graph padded row와 logical
row, aborted I와 new C, victim D resume frontier를 검증한다. 최종 text가 같다는 것으로 cache lifetime
correctness를 증명하지 않는다.

이 vertical timeline이 기능표보다 가치 있는 이유는 설정 변경의 최초 divergence를 찾을 수 있어서다.
old/new version에서 I가 늦어졌다면 admission부터 한 경계씩 비교한다. queue order가 같으면 comparator
가설을 버리고 KV/row/output으로 내려간다.

이 비교의 회고는 다음과 같다.

네 scheduler 비교에서 가장 중요한 답은 “같다”보다 “비교 불가한 owner가 어디인가”다. vLLM request
priority와 llama LRU slot은 같은 차원이 아니다. SGLang new-token ratio와 Transformers safety margin도
같은 fairness knob가 아니다. 서로 다른 cost를 공통 workload 결과에서만 비교한다.

독자는 이제 새 stack을 보아도 열두 질문으로 vertical trace를 만들 수 있다. public option 목록보다
admission owner, native container, budget intersection, suffix owner, device row, KV transaction, output
generation과 evidence surface를 찾는다.

다음 Part로 넘기는 정확한 상태도 명시한다.

I/D/P/C는 네 구현에서 각각 terminal 또는 계속-running 상태에 있고, native queue/batch/cache/output
owner가 명시됐다. 누락된 evidence gap도 기록됐다. 이후 장에서는 특정 stack의 kernel/분산/운영을
더 깊게 볼 때 이 비교 기록를 regression contract로 재사용한다.

종합장은 winner를 선언하지 않는다. workload와 실패 비용을 명시하고, source-confirmed semantics와
runtime-required evidence를 분리한 선택 문서를 남기는 것으로 닫는다.

step 0 output commit 뒤 D frontier는 +1, I prompt는 decode-ready 또는 first output state, P frontier는
+479가 된다. 이 durable mutation 시점을 각 vertical trace에서 찾는다. schedule output 생성 직후가
아니라 worker/host update 뒤일 수 있다. graph replay가 끝났어도 CPU state가 아직 old일 수 있다.

step 1 I cancel이 model launch 전이면 scheduled membership에서 제거 가능한지, launch 후이면 output을
discard/reconcile하는지 구분한다. C retry가 cancel call 반환 직후 들어오면 duplicate check와 old owner
cleanup이 경쟁한다. same ID가 허용되지 않으면 application은 C에 새 internal ID를 써야 한다.

step 2 P current chunk가 KV shortfall 128 slot을 만든다고 하자. vLLM이 D를 victim으로 129+ blocks를
풀면 computed reset/recompute debt가 생긴다. SGLang이 tree cache 64를 evict하고 I/D 중 하나를 retract해
나머지를 얻을 수 있다. Transformers는 CPU offload victim으로 payload를 보존한 뒤 schedule retry할
수 있다. llama는 idle cached slot의 old context를 버려 P task를 배치할 수 있다.

## 32.9 같은 arrival·token·KV 예산으로 네 구현을 여섯 step 실행한다

비교의 입력을 다시 고정한다. iteration token budget은 8, active sequence capacity는 3, KV capacity는 24 token-equivalent다. D는 시각 0에 prompt 8/output 6인 decode-heavy 요청, I는 시각 1에 prompt 4/output 2인 interactive 요청, P는 시각 1에 prompt 12/output 2인 long-prefill 요청이다. C는 시각 3에 I와 같은 외부 요청의 retry지만 새 internal incarnation을 가진다.

공통 사실은 arrival과 shape뿐이다. 네 구현의 priority, chunked prefill, retraction/preemption, CPU offload, prompt cache와 slot selection capability가 다르므로 결과를 강제로 같게 만들지 않는다. 각 표에는 native branch를 적고 unsupported는 disabled 또는 비교 불가로 표시한다.

**step 0: D prefill**

vLLM은 waiting D를 token/KV budget 안에서 schedule하고 blocks를 할당한다. SGLang은 waiting queue policy와 prefix/cache availability를 계산한 뒤 prefill batch로 보낸다. Transformers continuous scheduler는 candidate D의 required blocks를 확보해 batch를 만든다. llama.cpp는 task D를 available slot에 launch하고 shared server batch에 prompt chunk를 넣는다.

네 구현 모두 이 단순 step에서는 prompt 8을 처리할 수 있지만 owner가 다르다. vLLM scheduler request와 KV manager, SGLang Req/ScheduleBatch와 radix/tree cache, Transformers RequestState/scheduler/block manager, llama task/slot/llama context다. output은 아직 사용자 token이 아니라 decode-ready cache frontier다.

**step 1 — I와 P가 함께 도착한다.**

budget 8에서 D decode 1, I prompt 4를 넣으면 5를 쓰고 남은 3은 P의 4-token chunk 최소보다 작다고 fixture에서 정의한다. vLLM priority가 I를 앞세우면 D+I를 선택하고 P는 waiting/skipped다. SGLang priority sign과 policy가 같게 설정돼도 new-token ratio와 prefix cache accounting 때문에 P eligibility가 다를 수 있다.

Transformers FIFO가 active decode D를 먼저 두고 I/P arrival order가 I,P라면 D+I가 가능하다. PrefillFirst라면 ongoing/new prefill ordering이 달라 P chunk가 I보다 앞설 수 있다. llama.cpp는 request priority queue가 아니라 available slot과 update traversal/slot state에 따라 I/P task를 slot에 배치한다. LCP/LRU slot placement를 I priority와 등치하지 않는다.

**step 2 — KV capacity가 1 모자란다.**

D cache 9, I prompt cache 4, P chunk를 추가하면 단순 합 17이라 아직 여유가 있다. 비교를 자극하려 D가 이전 fixture의 long resident prefix 16을 가진 상태라고 바꾼다. D decode 뒤 17, I 4로 21, P chunk 4는 25가 되어 한 칸 부족하다. 이 state override를 네 구현에 동일하게 준다.

vLLM은 allocation failure에서 running victim을 preempt해 blocks와 provisional budget을 되돌릴 수 있다. priority policy라면 lowest-priority running victim key를 사용한다. [vLLM victim selection과 refund](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L571-L624)

SGLang은 tree cache eviction으로 inactive reusable entries를 먼저 줄이고 decode memory check에서 running requests를 retract할 수 있다. exact victim과 new-token ratio update는 native source를 따른다. priority preemption threshold가 켜졌다면 priority difference branch도 개입한다. [SGLang priority preemption branch](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_policy.py#L1432-L1475)

Transformers continuous는 scheduler candidate processing과 block allocation capability, configured CPU offload가 있는지에 따라 candidate를 미루거나 active state를 offload할 수 있다. vLLM recompute와 같은 victim semantics라고 쓰지 않는다. [Transformers candidate scheduling 경계](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/scheduler.py#L22-L113)

llama.cpp는 active slot D를 priority victim으로 preempt하는 동일 기능이 없을 수 있다. idle cached slot context를 버려 P placement capacity를 얻는 것과 active decode를 중단하는 것은 다르다. 모든 slot이 active이고 context capacity가 부족하면 task를 deferred하거나 context shift/실패 path를 따른다. “preemption 0회라 더 안정적”이라는 비교는 capability 차이를 누락한다.

**step 3 — I cancel과 C retry가 겹친다.**

I가 model launch 전에 cancel되면 scheduler membership에서 제거하고 blocks를 free할 기회가 있다. launch 뒤라면 output discard와 device completion fencing이 필요하다. C는 외부 retry지만 새 internal ID로 들어간다. same external ID dedup/idempotency는 gateway와 manager 계약을 구분한다.

vLLM은 preempted/finished/cancel output reconciliation과 request ID lifecycle을 본다. SGLang은 tokenizer manager→scheduler abort control과 cache cleanup을 본다. Transformers는 cancel queue를 driver가 drain하고 scheduler가 cancelled states와 offloaded cache를 clear한다. llama.cpp는 task queue/deferred/slot/in-flight 위치별 cancel과 response reader를 닫는다.

비교 표의 결과 열은 cancel API latency가 아니라 `last committed token, output discard generation, KV reclaim-ready, C admitted`다. 한 구현이 cancel을 빨리 return해도 in-flight block 재사용이 늦으면 capacity recovery는 다르다.

**step 4와 5 — P가 재개되고 D가 완료된다.**

P가 preempt/retract 없이 waiting이었다면 unseen suffix부터 진행한다. victim이었다면 recompute, restored offload, cache sequence state에 따라 recovery cost가 다르다. D가 completion token을 생성한 iteration에는 terminal update 뒤 KV release 시점을 launch 전 capacity로 소급하지 않는다.

여섯 step 최종 표에는 구현별 D/I/P/C selected set, P computed frontier, victim/action, freed KV, recovery work, committed output과 queue state를 둔다. throughput만 세기 전에 네 구현이 실제로 동일 logical token work와 output을 완료했는지 확인한다.

## 32.10 capability matrix를 branch predicate와 비용으로 번역한다

기능표의 체크 표시는 부족하다. priority가 있다고 해도 waiting ordering만 바꾸는지 running victim preemption까지 하는지 다르다. chunked prefill도 suffix owner와 minimum chunk, decode protection이 다르다. cache offload와 recompute는 recovery resource가 다르다.

**priority와 victim capability.**

vLLM priority queue는 작은 priority와 이른 arrival ordering을 갖고 priority mode에서 running victim key를 사용할 수 있다. SGLang은 low-value-first sign option과 preemption threshold를 가진다. Transformers FIFO/PrefillFirst는 phase/candidate ordering이며 arbitrary tenant priority와 같지 않다. llama LCP/LRU는 slot/cache placement다.

공통 workload에서 “interactive I가 P보다 먼저 first token을 받는가”는 비교할 수 있다. 하지만 그 결과가 request priority, phase ordering, prefix locality, arrival slot availability 중 어느 branch 때문인지 native trace로 설명해야 한다.

**memory recovery capability.**

vLLM preemption recompute는 computed frontier와 stale output generation을 reset해 waiting으로 돌릴 수 있다. SGLang retraction은 active batch에서 요청을 빼고 cache/tree state와 new-token ratio를 갱신한다. Transformers offload는 configured capability에서 CPU payload와 future state를 보존할 수 있다. llama context/slot cache eviction은 idle locality를 버리는 것이며 active victim swap과 동일하지 않다.

비용 열에는 freed GPU byte, lost compute token, D2H/H2D byte, host resident, future prefix miss를 넣는다. capability가 없는 구현은 비용 0이 아니라 해당 overload scenario에서 wait/reject/context shift라는 다른 결과를 갖는다.

**graph와 batching capability.**

vLLM CUDA graph/cudagraph mode, SGLang overlap/capture, Transformers continuous static IO graph, llama ggml/CUDA graph는 capture owner와 supported shape가 다르다. 동일 active sequence 수라도 packed token, padding, graph bucket과 fallback이 다르다. graph hit rate만으로 scheduler 우열을 판단하지 않는다.

한 구현이 inactive padding row를 계산해 tokens/s 분모에 포함하고 다른 구현이 useful token만 세면 throughput 수치가 왜곡된다. metric definition을 committed useful model token 또는 request output token으로 정규화하고 prefill/recompute/padding을 별도 원장으로 둔다.

**output와 backpressure capability.**

vLLM/SGLang/Transformers/llama.cpp는 output routing과 streaming owner가 다르다. 느린 consumer가 core loop를 block하는지, per-request queue가 bounded인지, disconnect가 cancel로 전파되는지를 비교한다. GPU tokens/s가 같아도 delivered tokens/s와 orphan compute가 다를 수 있다.

capability matrix의 한 행은 `capability, option/default, source predicate, state mutation, cost, unsupported/fallback, observation`이다. 체크 표시는 이 일곱 칸을 모두 채운 뒤 요약으로만 쓴다.

## 32.11 겉보기 throughput 1등이 useful goodput 꼴찌였던 사건

benchmark가 vLLM 1,200 tok/s, SGLang 1,150, Transformers 900, llama.cpp 780을 보고했다고 하자. 결론은 vLLM 승리였다. 그러나 분모와 workload를 열어 보니 vLLM 수치는 recomputed prefill과 speculative/padding scheduled token을 포함했고, llama 수치는 delivered output token만 셌다. 또한 llama task 두 개는 context capacity 때문에 deferred되어 measurement window 안에 완료되지 않았다.

**관측.**

vLLM GPU utilization과 scheduled token counter는 높았지만 completed request와 delivered output은 낮았다. preemption storm에서 P prompt 12가 세 번 재계산되어 24 lost token이 추가됐다. Transformers는 CPU offload traffic 때문에 raw throughput은 낮지만 P progress를 보존해 완료율이 높았다. SGLang은 tree prefix hit로 prompt work 자체가 적었다.

**경쟁 가설.**

첫 가설은 kernel이 빠르다는 것이다. 둘째는 scheduler가 더 많은 useful work를 배치했다는 것이다. 셋째는 metric이 recompute/padding을 useful로 셌다는 것이다. 넷째는 admission/deferred 차이로 completed cohort가 다르다는 것이다. 다섯째는 cache hit가 input work를 바꿨다는 것이다.

**원인.**

source와 trace를 맞추면 vLLM preempt reset과 reschedule로 scheduled token이 늘었고 benchmark collector가 이를 분자에 넣었다. llama deferred requests는 denominator/request latency report에서 빠졌다. SGLang prefix reuse token은 애초 compute되지 않았다. 네 엔진이 동일 logical workload를 완료하지 않은 상태에서 raw engine counters를 나란히 놓은 것이 원인이다.

**검증.**

arrival를 닫고 모든 request가 terminal될 때까지 측정한다. logical requested prompt/output, actually evaluated prompt, recomputed, padding/speculative, committed output, delivered output과 wall/GPU time을 분리한다. useful goodput은 제품 목적에 맞게 committed 또는 delivered output과 required prompt progress로 정의한다.

동일 cache warm/cold 조건, same tokenizer/template/model artifact와 output contract를 고정한다. timeout/deferred/reject를 결과에서 제외하지 않고 completion rate와 tail로 보고한다. graph/eager와 offload/recompute branch를 trace에 붙인다.

**선택과 rollback.**

정책 선택은 하나의 winner가 아니라 workload별이다. Gold interactive tail이 우선이면 priority/preemption capability와 amplification guardrail을 본다. long prefill completion이 중요하면 chunk fairness와 progress preservation을 본다. prefix-heavy면 locality와 cache identity를 본다. edge device면 llama.cpp의 footprint와 slot model이 맞을 수 있다.

rollback은 benchmark flag를 되돌리는 일이 아니다. 선택한 scheduler option을 canary에서 끄고 in-flight preempt/retract/offload/slot state를 drain하며 baseline useful-goodput와 SLO curve가 복원되는지 본다. metric definition version도 함께 rollback/고정한다.

**선택 결정 트리.** workload가 긴 prefill 때문에 decode ITL을 깨면 chunk budget과 decode 우선순위를 먼저 비교한다. KV 압력이 preemption을 만들면 단순 throughput이 아니라 recompute·swap 비용과 완료 goodput으로 판정한다. priority 요청이 일반 요청을 굶기면 aging과 fairness bound를 본다. async scheduler가 빠르지만 stale future가 cancel 뒤 commit되면 성능 후보에서 제외한다. 네 분기는 같은 arrival trace와 SLO 위반 비용으로 검증한다.

최종 진단은 평균 처리량 1등을 고르는 일이 아니다. workload별 SLO 위반 요청, 완료 token, 재계산 byte와 취소 뒤 잔여 state를 같은 표에 놓고, 실패 비용을 감당할 owner가 없는 후보는 제외한다.

## 32.12 관측→반증→선택 문서를 네 vertical trace로 닫는다

금요일 선택 회의에서 독자가 완성된 dossier를 열었다고 하자. 첫 화면에서 동일 arrival와 resource budget, artifact/config를 확인하고, 특정 요청 D를 눌러 여섯-step decision table로 내려간다. 선택이 갈린 step에서는 capability branch matrix를 옆에 놓고, 마지막에는 work conservation과 latency/completion 결과에서 비용을 확인한다. 배포 후보를 고른 뒤에도 first divergence와 rollback 기록까지 같은 request trace로 되돌아갈 수 있어야 한다.

이 사용 장면에서 문서의 순서는 제작 순서가 아니라 질문 순서다. “입력이 정말 같았나”, “어느 step에서 선택이 갈렸나”, “native capability 때문인가”, “유용한 일이 늘었나”, “되돌리면 같은 경로가 복원되나”를 차례로 답한다. 아래 항목은 별도 제출물이 아니라 그 질문에 답하는 dossier의 서로 연결된 보기다.

### 32.12.1 사건 하나를 관측하고 첫 분기를 반증한다

**관측 최소 집합.**

arrival/wait/eligible/selected/launch/commit/terminal 시각, prompt/decode/recompute token, free/allocated/released KV, victim/recovery mode, prefix reused/evaluated, graph path와 output routed/delivered를 둔다. native request ID는 trace에, policy/capability cohort는 bounded metric에 둔다.

**반증 순서.**

선택이 다르면 priority/key보다 먼저 native capability와 effective option을 확인한다. key가 같으면 eligibility/budget, selected set, allocation/victim, execution, update/commit으로 내려간다. throughput이 다르면 logical work identity, cache warmness, recompute/padding, completion cohort와 output delivery부터 맞춘다.

**source 범위.**

vLLM scheduler/queue/preemption, SGLang schedule policy/retraction, Transformers scheduler/cache manager/continuous processor, llama server queue/slot/update/context를 pinned revision에서 가리킨다. actual GPU path와 runtime option은 별도 evidence다. 실행하지 않은 branch를 성능 사실로 쓰지 않는다.

관측과 source를 함께 두는 이유는 source가 가능한 branch를, runtime decision row가 실제로 소비한 branch를 각각 증명하기 때문이다. 둘이 만날 때까지 옵션 존재만으로 정책 효과를 단정하지 않는다.

### 32.12.2 선택과 rollback을 같은 request trace에서 검산한다

**선택 terminal.**

correctness는 동일 request output과 cache/position generation을 보존한다. SLO는 workload tier의 TTFT/ITL/completion/recovery를 만족한다. efficiency는 useful goodput, recompute/swap/eviction/padding debt를 포함한다. lifecycle은 cancel/finish/late output과 resource release가 정확히 한 번이다.

**rollback terminal.**

새 admission을 멈추고 provisional/in-flight state를 drain한다. policy/capability generation을 이전 값으로 바꾸고 queue/debt/cache metadata를 호환되게 rebuild한다. baseline arrival trace의 first selections, completion curve와 output semantics가 돌아오는지 확인한다. rank/worker 혼합 generation에서는 admission을 열지 않는다.

선택 terminal과 rollback terminal을 나란히 읽는 이유는 ‘좋아진 후보’가 아니라 ‘실패해도 이전 의미로 돌아갈 수 있는 후보’를 고르기 위해서다. 다음의 구현별 row와 수치 계산은 이 두 terminal의 빈칸을 채우는 근거다.

32장의 최종 invariant는 다음과 같다. **동일 logical arrival와 자원 예산에서 각 구현의 native capability와 branch가 선택·victim·resume·output을 설명하고, 비교 metric은 실제 evaluated/recomputed/committed/delivered work와 completion cohort를 보존하며, 선택과 rollback이 같은 vertical trace로 재현되어야 한다.**

이 문장을 여섯-step table, capability predicate, false throughput incident와 rollout record로 설명할 수 있으면 네 scheduler 비교가 기능표나 홍보 benchmark를 넘어 실제 설계 선택 도구가 된다.

### 32.12.3 완성 dossier로 구현별 decision row를 다시 읽는다

**네 구현의 decision row를 같은 열로 다시 쓴다.**

vLLM row는 waiting/running queue owner, priority/arrival key, token budget, KV block allocation, preempt victim과 reset computed frontier, scheduled output generation을 가진다. D/I/P/C 각각에 `num_computed_tokens`, scheduled tokens, new blocks와 preemption count를 적는다. waiting에서 skipped로 갔다가 merge되는지까지 본다.

SGLang row는 waiting Req와 ScheduleBatch membership, schedule policy key, prefix matching과 available size, new-token ratio, retracted request와 tree cache eviction을 적는다. priority sign과 threshold가 effective인지 확인한다. retraction 뒤 request의 prefix/re-prefill progress와 output generation을 기록한다.

Transformers row는 RequestState status와 scheduler candidate class, required/allocated blocks, batch tensor row와 Future/offload state를 적는다. FIFO/PrefillFirst가 어느 candidate phase를 먼저 처리했는지와 safety margin/CPU offload가 capacity decision을 어떻게 바꿨는지 본다. cancel queue와 output router terminal을 연결한다.

llama.cpp row는 task queue/deferred, selected server slot과 generation, cached prefix similarity/LRU, shared batch token/sequence/position, context capacity와 result queue를 적는다. active victim preemption 칸은 unsupported/해당 없음으로 쓰고 대신 deferred/context shift/cache replacement 결과를 적는다.

같은 열을 쓰되 빈 칸의 의미를 보존한다. llama에 active-victim preemption counter가 없다고 0으로 채우면 “지원하지만 발생하지 않음”과 혼동된다. `unsupported`, `disabled`, `eligible but not chosen`, `not observed`를 구분한다.

**여섯-step decision table의 숫자를 끝까지 채운다.**

step 0 뒤 D computed/resident는 8이다. step 1에서 D decode와 I prompt를 처리하면 D=9, I=4, token budget used=5다. P는 chunk minimum 4 때문에 remaining 3에서 기다린다. 이 결과가 다른 구현에서는 policy/capability 때문에 달라지면 실제 selected set을 별도 열에 적는다.

step 2 override에서 D resident=17, I=4이고 P chunk 4를 더하면 25>24다. vLLM priority가 P보다 D를 보호하면 P를 skip할 수 있고, 높은 P가 들어오면 running victim을 고른다. SGLang은 inactive prefix eviction x와 retraction y를 조합할 수 있다. Transformers는 candidate wait/offload 여부, llama는 slot/context availability 결과를 적는다.

step 3 I cancel은 logical capacity 4를 eventually 돌려준다. 하지만 in-flight면 step 3 launch 전 즉시 free로 세지 않는다. 각 구현 표에 cancel accepted, device/resource completion, cache reclaim-ready 시각을 둔다. C admission은 reclaim-ready와 active sequence capacity 둘을 만족해야 한다.

step 4 P가 4-token chunk를 실행하고 D가 decode한다면 used token 5다. preempt된 D가 recompute해야 하면 D의 17-token recovery가 token budget을 차지해 P와 C latency를 바꾼다. offloaded state restore는 transfer time과 blocks를 먼저 요구한다. deferred였던 llama task는 slot이 비어야 launch된다.

step 5 D 또는 C가 terminal되며 resource를 release한다. completed output token과 evaluated/recomputed prompt token을 합산한다. 같은 wall window가 아니라 모든 D/I/P/C incarnation이 terminal된 closed workload에서 completion time과 useful work를 계산한다.

**source 함수에서 table cell을 찾는 법.**

vLLM의 selected/victim cell은 scheduler schedule loop와 request queue, `_preempt_request`에서 찾는다. SGLang selected/retracted cell은 schedule policy priority calculation과 batch memory/retraction path에서 찾는다. Transformers selected/blocks cell은 `schedule_batch`, candidate processing과 cache manager allocation/refcount에서 찾는다. llama selected/slot cell은 `process_single_task`, `get_available_slot`, `launch_slot_with_task`, `update_slots`에서 찾는다.

각 cell에는 producer와 consumer를 둘 다 붙인다. priority field가 request에 있다는 것만으로 comparator 적용을 증명하지 않는다. block available metric이 있다는 것만으로 해당 allocation branch를 증명하지 않는다. option→effective state→predicate→mutation→next consumer를 연결한다.

source 고정 링크는 가능한 semantics를 보여 준다. runtime decision table은 actual branch를 증명한다. 둘이 다르면 custom fork, default resolution, feature capability fallback과 stale docs를 조사한다. source를 runtime log처럼 쓰지 않는다.

**동일 자원 예산의 함정.**

KV 24 token-equivalent는 비교 설명을 위한 logical capacity다. 실제 구현은 block size, padding, per-layer cache type, prefix sharing, metadata와 reserved margin이 다르다. 동일 GPU byte를 주고 logical token capacity가 달라질 수 있다. 비교에는 logical 요구와 actual allocated byte 둘을 둔다.

token budget 8도 실제 compute 동일성을 보장하지 않는다. prefill token과 decode token attention cost가 다르고 recompute, speculative verify, padding이 추가된다. scheduler가 세는 scheduled token과 backend가 실행한 row, committed useful token을 나눈다.

active sequence capacity 3은 slot/sequence admission을 맞추기 위한 제약이다. llama parallel slot과 vLLM max sequences, Transformers static capacity, SGLang batch requests가 같은 memory overhead를 갖지는 않는다. 이 cap이 어느 native option/structure로 구현됐는지 기록한다.

동일 budget을 강제해 한 구현의 native 최적화를 비활성화할 수도 있다. 공정 비교는 두 단계다. 첫째 semantic controlled fixture로 decision을 비교한다. 둘째 각 구현의 권장 안전 tuning에서 product SLO/total cost를 비교한다. 두 결과를 섞지 않는다.

**false throughput incident를 숫자로 재계산한다.**

measurement window 1초 동안 vLLM scheduled 1,200 token 중 useful evaluated prompt 600, committed output 200, recompute 300, speculative/padding 100이었다고 하자. scheduler counter 1,200을 throughput이라 부르면 1,200 tok/s다. useful model progress를 prompt+output으로 정의하면 800/s이고 delivered output만 보면 200/s다.

SGLang scheduled 1,150 중 prefix reuse로 evaluated prompt 500, output 250, retraction recompute 150, padding/spec 250라고 하자. raw 1,150, useful 750, output 250이다. Transformers 900 중 evaluated 600, output 260, restore overhead token-equivalent를 token counter에 넣지 않았다면 useful 860이다. llama 780 중 evaluated 520, output 260이면 useful 780이지만 deferred P가 window 안 terminal되지 않았다.

closed workload wall time을 vLLM 1.4s, SGLang 1.3s, Transformers 1.35s, llama 1.8s라고 가정하자. required useful work가 동일하게 prompt evaluation 정책을 cold로 고정하면 cache reuse를 끄거나 saved prompt를 accounted work로 별도 표기한다. output completion과 SLO를 같이 보지 않으면 winner가 계속 바뀐다.

completed request가 vLLM 4/4, SGLang 4/4, Transformers 4/4, llama 3/4 at 1s라면 open-window output rate는 survivor bias를 가진다. timeout/deferred P를 denominator에서 빼지 않는다. closed workload와 steady-state offered load 두 실험을 분리한다.

**cache warmness를 공정하게 다루는 두 방법.**

첫 방법은 cold cache다. 모든 구현에서 reusable prefix를 비우고 requested prompt를 실제 eval한다. scheduler/kernel base behavior를 비교하기 쉽지만 production locality 이득을 무시한다. 둘째는 동일 prefix arrival history를 먼저 replay해 각 native cache를 warm하게 한다. hit semantics와 capacity가 다르므로 reused token과 lookup/eviction cost를 보고한다.

cache hit를 work cheating으로 보지 않는다. 동일 output semantics를 유지하며 compute를 제거한 것은 유효 최적화다. 다만 raw evaluated tok/s와 end-to-end useful request goodput을 구분한다. hit가 많아 evaluated token이 적은 구현이 tok/s는 낮아도 TTFT와 cost는 좋을 수 있다.

cache identity를 맞춘다. tokenizer/template/model/adapter와 position representation이 같아야 한다. incompatible hit나 numerical tolerance 차이를 성능 이득으로 세지 않는다. cache off differential로 correctness를 본다.

**priority와 fairness capability를 선택 문서에 쓰는 법.**

workload가 authenticated tenant priority와 strict Gold TTFT를 요구하면 arbitrary priority comparator와 running victim/recovery가 중요한 capability다. vLLM/SGLang의 native branches를 검토하고 amplification/minimum service guardrail을 둔다. Transformers phase policy나 llama slot locality만으로 같은 계약을 제공한다고 쓰지 않는다.

workload가 단일 tenant batch throughput이고 long prefill이 많다면 PrefillFirst/chunk policy와 cache preservation이 더 중요할 수 있다. active victim preemption이 오히려 recompute storm을 만든다. priority 기능 수가 많은 구현이 자동 승자가 아니다.

edge/local 서버에서 memory footprint와 단순 slot isolation, CPU/GPU offload flexibility가 중요하면 llama.cpp가 적합할 수 있다. 그러나 high concurrency dynamic priority와 active victim semantics가 필요하면 capability gap을 application queue로 보완할 비용을 적는다.

선택 문서에는 필수 hard capability, tunable policy, unsupported workaround, failure/rollback cost를 나눈다. benchmark 순위는 마지막 참고 열이다.

**cancellation과 retry parity.**

I cancel과 C retry에서 네 구현이 동일 visible semantics를 제공하는지 본다. I가 이미 세 token을 delivered했다면 C가 새 request로 시작할 때 partial output을 이어받는지 별도 API contract다. engine internal ID를 재사용하지 않는다.

cancel latency는 enqueue/mark, output stop, resource reclaim 세 값이다. 하나의 숫자로 비교하지 않는다. vLLM/SGLang async output reconciliation, Transformers cancel queue, llama response reader/slot release가 각 frontier를 가진다. resource reclaim이 늦어도 output은 즉시 멈출 수 있다.

retry admission은 old generation late output과 cache state가 격리돼야 한다. same prompt cache reuse가 허용돼도 request output/sampler history는 새 incarnation이다. failure injection은 old device completion을 지연한 뒤 C를 넣어 stale commit을 검사한다.

**graph hit rate 비교의 반례.**

구현 A graph hit 95%, B 70%라 A가 빠를 것으로 예상했지만 A는 큰 padded bucket을 replay해 useful rows 40%였고 B는 eager/작은 graph로 useful rows 85%였다. graph hit는 실행 방식의 비율이지 saved work의 비율이 아니다.

metric에 graph bucket capacity, active rows/tokens, padding work, replay/launch time, eager fallback reason을 둔다. capture overhead amortization과 static memory도 본다. scheduler가 batch shape를 regularize하려고 request를 기다리게 하면 queue latency와 함께 본다.

동일 arrival trace에서 graph optimization을 끈 reference로 selected/committed semantics가 같은지 확인한다. graph를 켠 결과 batch membership 자체를 바꾸는 정책이 있다면 latency/throughput과 fairness 차이를 기록한다.

**최종 verification lab.**

lab 1은 cold cache six-step table이다. lab 2는 900-token shared prefix warm trace다. lab 3은 KV one-token shortfall과 victim/recovery다. lab 4는 I in-flight cancel/C retry다. lab 5는 slow output consumer다. lab 6은 graph bucket boundary다.

각 lab은 expected native branch와 unsupported를 먼저 적고 actual trace를 채운다. source expectation이 틀리면 구현을 억지로 fixture에 맞추지 않고 capability matrix를 수정한다. output correctness, terminal/refcount, useful work와 SLO를 모두 본다.

rollout은 workload cohort 하나에서 시작하지만 rare victim/cancel/overflow branch가 실제 실행되게 한다. shadow scheduler는 decision만 비교하고 allocation side effect를 증명하지 못한다는 한계를 적는다. failure injection canary로 lifecycle을 보완한다.

rollback은 engine process option만 내리지 않는다. preempted/retracted/offloaded/deferred request를 drain하고 cache/queue generation을 이전 semantics로 복귀한다. metric definition과 dashboard version도 고정해 rollback 뒤 수치가 같은 의미인지 확인한다.

**한 사건을 관측에서 rollback까지 실제로 닫는 예.**

운영자는 월요일 10시 배포 뒤 Gold 요청의 p99 TTFT가 420 ms에서 1.8 s로 뛰었지만 GPU busy는 72%에서 94%로 올랐고 raw scheduled throughput도 18% 좋아졌다는 경보를 받았다. 이 세 수치만 보면 GPU가 더 열심히 일하므로 부하 증가나 kernel 회귀가 먼저 떠오른다. 그러나 같은 시간 completed-request rate는 11% 떨어졌고 KV preemption counter는 분당 4회에서 310회로 늘었다. 여기서 첫 관측 문장은 “새 scheduler가 느리다”가 아니다. “동일 Gold arrival cohort에서 TTFT와 completion은 악화했고 scheduled work와 preemption은 증가했으며 device busy는 높아졌다”다. 관측과 해석을 분리해야 다음 반증이 가능하다.

먼저 offered load 증가 가설을 반증한다. gateway arrival log에서 model, adapter, prompt-length bucket, requested output bucket과 tenant tier를 묶어 배포 전후 같은 15분 cohort를 replay한다. 새 cohort의 prompt가 길어졌다면 scheduler 탓으로 단정하지 않는다.

다음은 kernel 회귀 가설이다. selected batch의 실제 shape, graph/eager path와 kernel duration을 같은 shape끼리 비교한다. kernel duration이 그대로인데 launch 수와 recompute row가 늘었다면 kernel은 높은 busy의 전달자이지 원인이 아니다. 세 번째는 output backpressure다. committed-to-delivered lag와 bounded output queue depth가 정상이라면 느린 client branch를 닫는다. 네 번째에서야 scheduler decision trace를 연다.

배포 전에는 D가 decode protection을 받고 P가 네 토큰 chunk로 전진했다. 배포 뒤 effective priority threshold가 바뀌어 새 Gold I가 들어올 때마다 낮은 priority D가 victim이 됐다. D의 17-token frontier가 reset되고 다음 admission에서 recompute됐다.

I의 TTFT 한 건은 짧아졌지만, 동시에 여러 D가 victim이 되어 KV가 잠깐 비었다가 recompute로 다시 찼고 이후 Gold I까지 waiting이 길어졌다. scheduled token counter는 이 재계산을 모두 유용한 전진처럼 셌다. 따라서 device busy와 raw throughput이 좋아진 채 completion이 나빠지는 모순이 풀린다. 원인은 priority 자체가 아니라 `threshold → victim predicate → frontier reset → recompute debt → 다음 capacity 부족`의 연쇄다.

이 진단을 네 구현에 기계적으로 복사하면 안 된다. vLLM에서는 running victim 선택과 computed-token reset, waiting 재삽입을 확인한다. SGLang에서는 retraction 대상, tree-cache eviction 이후 usable memory와 new-token ratio 조정을 확인한다. Transformers에서는 해당 배포가 실제로 active state offload를 지원하고 켰는지, 아니면 candidate가 단지 기다렸는지 확인한다. llama.cpp에서는 idle cached slot replacement를 active decode victim으로 오해하지 않고 deferred task와 context shift를 본다. 같은 현상명 “메모리 압박” 아래 mutation과 미래 비용이 전혀 다르다.

검증 canary는 arrival 순서와 random seed만 고정하지 않는다. prompt cache는 cold 한 번과 production history warm 한 번을 따로 실행하고, output consumer는 정상 속도와 초당 한 토큰의 느린 속도로 나눈다. 각 run에서 `selected request`, `reason`, `victim`, `freed block`, `lost frontier`, `restore byte`, `evaluated prompt`, `committed output`, `delivered output`, `terminal reason`을 동일 trace schema로 저장한다.

vLLM canary에서 threshold를 원래 값으로 돌렸을 때 preemption과 recompute가 내려가고 completion curve가 복원되며 kernel duration은 그대로라면 원인 가설이 지지된다. 단순 상관이 아니라 mutation을 제거했을 때 downstream 결과가 사라지는 반증이다.

선택은 “priority를 끈다”로 끝나지 않는다. Gold TTFT를 지키면서 victim amplification을 제한하는 guardrail을 둔다. 예를 들어 한 request의 victim 횟수, cohort recompute/useful 비율, minimum decode service와 KV headroom을 admission 조건으로 둔다. 어떤 구현에 그 native predicate가 없으면 gateway queue나 workload partition으로 보완하되, 그 우회가 engine 내부 active victim과 같은 원자성을 주지 않는다고 명시한다. 선택 기록에는 기대 이득, 손실 가능한 workload, 관측 metric, 자동 중단선, drain 방식과 이전 generation의 호환성을 쓴다.

rollback은 경보가 울린 순간 새 admission을 baseline cohort로 돌리고, 이미 새 policy generation에서 preempted된 요청을 그대로 구 generation queue에 섞지 않는다. 그 요청은 완료시키거나 명시적으로 재시도시킨 뒤 KV와 output generation이 terminal인지 확인한다. cache metadata가 policy-independent라면 유지할 수 있지만, scheduler가 보존 frontier의 의미를 바꿨다면 rebuild한다. dashboard도 raw scheduled tok/s가 아니라 useful/recompute/committed/delivered를 나눈 버전으로 되돌린다. 마지막 확인은 GPU busy 하락이 아니라 동일 closed arrival trace의 output equality, completion 4/4, Gold p99 복원과 resource refcount 0이다.

**네 scheduler의 여섯-step 원장을 완성한 예시.**

아래 행은 성능 예측값이 아니라 decision table을 어떻게 닫는지 보여 주는 감사 예다. 설정으로 명시되지 않은 native option은 추측하지 않고 `확인 필요`로 남긴다.

| step | 공통 snapshot | vLLM에서 확인할 결정 | SGLang에서 확인할 결정 | Transformers continuous에서 확인할 결정 | llama.cpp에서 확인할 결정 |
|---|---|---|---|---|---|
| 0 | D prompt 8, KV 24 free | D selected, block allocation, computed=8 | D prefill batch, prefix match와 tree entry | D candidate, required block와 batch row | D task→slot, prompt batch row |
| 1 | D decode 1, I prompt 4, P chunk min 4, budget 8 | D+I selected, P skipped reason | policy key 뒤 D+I 또는 P first의 실제 branch | FIFO와 PrefillFirst 각각 candidate 순서 | I/P slot availability, deferred 여부 |
| 2 | D resident 17, I 4, P 4이면 25>24 | allocation fail, victim 또는 skip, refund | inactive eviction, retraction, usable memory | candidate wait 또는 configured offload | idle cache replacement 또는 defer/context path |
| 3 | I cancel, C는 새 incarnation | abort frontier, KV reclaim-ready, C admission | abort control, cache cleanup, C batch | cancel queue drain, offload cleanup | queue/slot 위치별 cancel, result generation |
| 4 | P chunk 4와 D/C decode 후보 | recompute가 budget을 먹는지 | re-prefill/retraction debt와 ratio | restore가 allocation보다 앞서는지 | 빈 slot 뒤 deferred launch 여부 |
| 5 | D/C terminal 후보 | output commit 뒤 block free | finish 뒤 Req/tree ownership | terminal state와 refcount release | slot release와 result queue close |

이 표의 핵심은 네 칸을 같은 말로 채우는 데 있지 않다. step 2에서 vLLM의 victim, SGLang의 retracted request, Transformers의 offloaded state, llama.cpp의 replaced idle context는 모두 “메모리 회복” 범주에 들어가지만 동일 사건은 아니다. table cell에는 native 명칭과 mutation을 남기고, 한 단계 위 comparison 열에서만 `freed device byte`, `lost compute`, `transfer`, `future miss`, `wait`로 정규화한다. 그래야 구현 세부를 지우지 않으면서 비용을 비교할 수 있다.

또한 각 행은 다음 행의 입력을 생산해야 한다. step 2에서 freed KV만 적고 lost frontier를 누락하면 step 4의 recompute가 갑자기 생긴다. step 3에서 cancel accepted만 적고 reclaim-ready를 누락하면 C가 왜 기다렸는지 설명할 수 없다. step 5에서 terminal status만 적고 output delivery와 refcount를 누락하면 완료된 요청이 메모리를 계속 잡거나 늦은 token이 새 incarnation으로 섞이는 결함을 놓친다. decision table은 로그 요약이 아니라 state-transition 원장이다.

마지막으로 source anchor의 역할도 제한한다. pinned 함수는 해당 branch가 구현돼 있고 어떤 state를 바꾸는지 증명한다. 실제 canary에서 그 branch가 실행됐다는 증거는 effective config와 decision trace다. 문서 default는 배포의 effective default를 증명하지 않으며, metric 이름은 mutation의 발생을 단독으로 증명하지 않는다. 그래서 모든 중요한 행에는 `source capability`, `effective option`, `runtime predicate`, `state before/after`, `next consumer` 다섯 칸이 필요하다. 이 다섯 칸이 연결될 때 독자는 코드에서 시작해 장애와 선택, rollback까지 같은 인과 사슬을 따라갈 수 있다.

**독자의 완주 조건.**

독자는 D/I/P/C의 같은 snapshot을 받으면 각 구현 source에서 next candidate와 capacity action을 예측한다. 결과가 다르면 native capability 또는 first divergent predicate를 말한다. unsupported를 성능 0이나 성공 0회로 오해하지 않는다.

raw tokens/s를 받으면 분자에 evaluated, recomputed, speculative, padding, committed와 delivered 중 무엇이 있는지 묻는다. completion cohort, cache warmness와 timeout/deferred를 확인한 뒤 useful goodput과 SLO로 다시 계산한다.

마지막으로 선택한 구현이 필수 capability와 failure cost를 만족하는 이유, 선택하지 않은 대안의 정확한 gap과 workaround 비용, rollback state transition을 한 페이지에 쓴다. 이 세 답이 있을 때 네 scheduler 비교는 독자가 실제 설계 결정을 내릴 수 있는 자료가 된다.

독자가 마지막으로 스스로 답할 질문은 단순하다. “다음 요청 하나가 도착하면 누가 그것을 보고, 어떤 predicate가 참이 되어, 어느 state가 바뀌며, 그 비용은 어느 다음 step에서 드러나는가?” 네 구현마다 이 문장을 source와 trace로 완성하고, 지원하지 않는 branch에는 정직하게 unsupported를 적으며, 같은 output contract까지 확인해야 비교가 끝난다. 숫자가 좋아졌다는 사실만으로는 이 질문을 대신할 수 없다.

그 비용을 실제 수용량으로 바꾸려면 scheduler 이름만으로는 부족하다. 33장은 여기서 고른 token budget과 candidate state를 KV block의 byte, free capacity와 eviction 조건에 대입해, admission predicate가 메모리 원장과 같은 답을 내는지 확인한다.
