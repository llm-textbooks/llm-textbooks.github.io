# 30장. 선점과 공정성: 누구를 먼저 계산하고 누구의 과거를 지울 것인가

동시에 네 요청이 들어온다. I는 40-token prompt 뒤 20 token을 원하는 interactive 요청이다. D는
이미 12,000 token을 생성 중인 긴 decode다. P는 16,000-token 긴 prefill이고, H는 높은 priority를
가진 200-token prompt다. GPU는 한 step에 512 query token, active request 네 개와 제한된 KV만
수용한다. 누구를 먼저 실행해야 하는가.

FIFO라면 arrival order가 직관적이지만 P가 앞에 있을 때 I/H가 기다릴 수 있다. decode 우선은 D의
ITL을 보호하지만 prefill이 계속 오면 P 또는 새 요청이 굶을 수 있다. priority는 H를 앞세우지만
낮은 priority D의 12,000-token KV를 지우면 recompute debt가 크다. prefix locality를 우선하면
throughput은 좋아질 수 있지만 cache miss 요청이 굶을 수 있다.

이 장에서는 먼저 **누가 다음 차례를 얻는지**를 결정하는 선택 키를 읽는다. 그다음 메모리가 모자랄 때
**누구의 계산을 되돌릴지** victim 비용을 센다. 마지막에는 밀려난 요청이 다시 서비스를 얻는 경로가
있는지 확인한다. 이 세 질문이 각각 선택, 희생, 굶주림 회복이다. 셋 중 하나라도 빠지면 “우선순위를
켰다”는 설명은 정책의 절반만 말한다.

“가장 좋은 scheduler” 하나를 고르지는 않는다. non-preemptible quantum, victim 비용,
starvation 조건과 goodput 목적을 같은 요청 장면에서 계산한다. 29장의 chunk 크기 자체와 31장의 async
pipeline은 반복하지 않고, 이미 정해진 service quantum을 누구에게 주며 memory pressure에서 누구를
되돌리고 언제 다시 불러오는지에 집중한다.

고정 소스는 vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang
v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, Transformers v5.15.1 commit
`550d7b3834670483a4df436541272c055dc364bf`, llama.cpp v0.2.0 계열 commit
`bb4caa7540188872173c44d161602d9271386413`다. runtime 수치를 만들지 않고 source semantics와
관측·반증 설계를 분리한다.

## 30.1 네 요청의 충돌을 숫자로 펼친다

요청 도착은 D=0ms, P=10ms, I=20ms, H=30ms라고 하자. D는 이미 KV 12,000 slot과 1-token decode
work를 갖고, P는 16,000 prefill 중 4,000을 계산했다. I/H는 아직 waiting이다. step quantum은
512 query token이다.

FIFO가 strictly arrival만 보면 D와 P가 앞이다. 그러나 continuous scheduler가 running을 waiting보다
먼저 보면 D 1과 P 511을 주고 I/H는 active slot이나 KV가 날 때까지 기다릴 수 있다. priority가 H를
가장 높게 놓으면 H 200을 admit하고 남은 312를 P 또는 D에 준다. KV가 부족하면 victim까지 골라야
한다.

### completion과 fairness 목적을 분리한다

I/H의 TTFT, D의 ITL, P의 completion time은 서로 다른 목적이다. total tokens/s가 최대여도 I의
deadline을 놓치면 interactive goodput은 낮다. 모든 request에 동일 token 수를 주면 prompt/decode의
GPU cost가 달라 시간 fairness가 아닐 수 있다.

tenant u가 받은 service를 `x_u`라 할 때 Jain index는
`J=(Σx_u)²/(nΣx_u²)`로 쓸 수 있다. x를 scheduled token으로 잡을지 GPU microseconds로 잡을지에
따라 의미가 다르다. priority tier가 있으면 weight `w_u`에 대한 normalized service `x_u/w_u`를
보거나 deadline-met output token을 goodput으로 센다.

### starvation의 충분조건을 본다

낮은 priority L보다 높은 요청이 service capacity 이상으로 계속 도착하고 strict priority가 aging이나
quota 없이 항상 높은 요청을 고르면 L의 waiting time에는 유한 상한이 없다. 이것이 starvation이다.
한 번 오래 기다렸다는 관측만으로 증명하지 않고, 정책에서 L의 effective rank가 시간에 따라 개선되는
경로가 없는지와 high-priority arrival rate가 service를 포화하는지를 함께 본다.

FIFO도 비용 기반 admission과 결합하면 starvation이 생길 수 있다. queue head P가 현재 KV/block
조건을 만족하지 못하고 구현이 head에서 break하면 뒤 I도 막힌다. 반대로 ineligible P를 skip하면
I는 실행되지만 P가 계속 skip될 수 있다. strict FIFO, work-conserving skip과 fairness는 같은 말이
아니다.

### preemption 비용의 세 형태

recompute는 victim의 유효 KV를 버리고 prompt/출력을 다시 forward한다. prefix `S` token, token당
compute time 평균 `c`라면 단순 debt는 `S×c`지만 chunk shape, prefix hit와 queue delay가 추가된다.
amplification은 `A=실제 계산 token/고유 필요 token`이다. 반복 선점이면 A가 1보다 커진다.

swap/offload는 KV byte를 CPU/remote tier로 옮긴다.

```text
Bytes_KV ≈ 2 × layers × kv_heads_local × head_dim × tokens × bytes(dtype)
T_roundtrip ≈ 2×Bytes_KV/BW_effective + setup + synchronization
```

retraction은 running batch에서 victim을 빼고 allocator state를 풀며 미래 reserve 추정을 바꿀 수 있다.
구현에 따라 prefix 일부를 cache에 남기거나 즉시 release한다. 세 용어를 “preemption” 하나로 등치하지
않고 어떤 state와 byte가 보존되는지 확인한다.

### 네 요청 timeline의 최종 손계산

네 요청의 숫자는 뒤 절 전체의 기준 fixture다. 정책을 비교할 때 이름만 바꾸지 않고 같은 도착 시각,
prompt progress, KV 소유량, priority와 512-token quantum을 넣는다. 각 정책에서 `eligible set`, 정렬 key,
실제 grant, victim, 남은 waiting 위치와 다음 step debt를 한 행에 기록한다. 그래야 H가 빨라졌다는 결과가
정렬 때문인지, P가 memory predicate에서 빠진 덕인지, D를 되감아 얻은 일시적 공간 때문인지 분리된다.

첫 snapshot을 더 엄격하게 적어 보자. D는 logical progress 12,000, 이번 query 1, deadline까지 40ms,
priority 5다. P는 progress 4,000/16,000, 다음 chunk 최대 512, deadline 2초, priority 5다. I는 prompt40과
output20, deadline 100ms, priority1이며 H는 prompt200, deadline60ms, priority0이다. free KV는 H와 I를
동시에 넣기에 160 token 부족하다고 둔다. 이 부족분 때문에 실제 victim branch가 실행된다.

step 0의 baseline FCFS가 running-first라면 D 1과 P 511이 Q budget을 모두 쓴다. I/H는 eligible이어도
waiting이다. strict priority admission이 running grant를 재검토하는 구현이라면 H 200, I 40을 넣고 D 1을
유지한 뒤 P grant 일부 또는 전체를 취소할 수 있다. 어떤 결과든 `Σgrant≤512`, runner rows와 grant map의
key set 일치, 각 row의 KV write capacity 확보를 만족해야 한다.

KV가 160 부족해 victim 하나를 고를 때 D를 지우면 12K debt, P를 지우면 4K debt다. 그러나 P의 prompt
completion과 D의 ITL deadline, prefix cache hit 가능성이 다르므로 작은 debt만 고르는 것도 자동 정답이
아니다. candidate decision에는 victim priority, progress, reusable prefix, freed capacity와 deadline impact를
적는다. native policy가 이 중 일부만 본다면 보지 않는 축을 capability gap으로 남긴다.

step 1에는 새 arrival이 없다고 하자. H/I가 step 0에 admit됐다면 각각 prefill progress와 KV owner를 얻고,
victim은 waiting/resume 상태다. H가 200 prompt를 끝내 첫 token을 만들었는지, I가 40을 끝냈는지에 따라
다음 Q와 active sequence budget이 달라진다. step 0의 선택 비용을 step 0 throughput만으로 평가하지 않고
step 1~completion의 debt와 freed service를 누적한다.

step 2에는 high-priority H2가 새로 도착하도록 변형한다. strict policy가 같은 victim을 다시 고르면 storm
fixture가 되고, aging/quota가 victim에게 service를 주면 starvation escape fixture가 된다. H2를 제거하면
rollback/reference timeline이다. 한 canonical fixture에서 arrival 하나만 바꿔 incident C와 D를 재현하므로
model/backend 차이가 원인으로 섞이지 않는다.

queue 보존식은 각 logical request가 정확히 하나의 scheduler owner에 있다는 것이다. `waiting + running +
provisional + terminal` membership 합은 1이어야 하며, in-flight device row는 별 generation으로 추적한다.
preempt 순간 logical owner가 waiting으로 바뀌어도 old row가 completion 전 존재할 수 있다. old row는 current
KV frontier나 output을 무조건 갱신할 권한이 없다.

resource 보존식은 final scheduled grant, physical allocation과 runner manifest가 같은 generation에서 닫히는
것이다. victim의 provisional grant를 환불하고 block을 free했으면 runner row도 제거한다. admission failure로
grant를 취소했지만 cache/lock을 획득했다면 rollback한다. exception 뒤 balance 숫자만 원상 복구돼도 block
owner나 queue entry가 남으면 transaction은 실패다.

service 보존식은 unique useful work와 debt를 분리한다. D가 preempt 전에 계산한 12K, resume 때 cache로
재획득한 10K, 실제 recompute 2K라면 debt는 12K라고도 2K라고도 단독 표현하지 않는다. lost frontier,
reacquired work, recompute/transfer와 wait를 각각 기록한다. 그래야 네 구현의 reset, radix reuse, CPU offload와
slot cache discard를 같은 원장으로 비교하면서도 서로 같은 동작이라고 오해하지 않는다.

마지막으로 output 보존식을 둔다. H/I/D/P가 선택·preempt·resume 순서를 달리해도 각 request의 accepted token
sequence와 terminal reason은 reference contract에 맞아야 한다. stale output drop으로 latency가 늘 수 있고
duplicate commit은 correctness failure다. fairness 정책이 결과 의미를 바꾸면 성능 비교 전에 반려한다.

이 세 step과 네 보존식이 뒤 절의 denominator다. 구현별 source를 읽을 때 fixture의 입력을 임의로 단순화하거나
없는 priority를 추가하지 않는다. native capability가 표현하지 못하는 축은 빈칸으로 남기고, 실제 선택과
비용을 동일한 request timeline에 다시 투영한다.

## 30.2 vLLM의 priority queue와 preemption을 transaction으로 읽는다

vLLM의 priority는 실행 중인 GPU kernel을 즉시 끊는 인터럽트가 아니다. scheduler가 다음 step을 만들 때
waiting 후보의 순서를 정하고, allocation failure에서 어느 running request를 희생할지 결정하는 정책
입력이다. 이미 제출된 kernel과 collective는 non-preemptible quantum을 끝낸다. H가 30ms에
도착했어도 current step이 6ms 남았다면 첫 scheduling 기회는 그 뒤다. priority latency의 하한은 queue
정렬만이 아니라 step quantum과 output reconciliation으로 정해진다.

고정 좌표는 vLLM v0.27.1의 [`Scheduler.schedule`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L280-L705)과
[`_preempt_request`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L1274-L1310)다.

첫 링크에서는 running loop, waiting admission, token budget 차감, allocation failure와 provisional refund를
한 transaction으로 읽는다. 둘째 링크에서는 victim의 computed frontier, block ownership과 queue 위치가
어떻게 바뀌는지 확인한다. 함수 이름이 `preempt`라는 사실만으로 swap이나 KV 보존을 추측하지 않는다.

### priority victim은 `(priority, arrival_time)` key로 고른다

priority policy를 사용할 때 먼저 숫자의 방향을 고정한다. 작은 값이 더 높은지 큰 값이 더 높은지 source의
comparator와 fixture로 확인한다. H의 priority가 0, I가 1, D/P가 5이고 작은 값 우선이라고 하자. waiting
key는 H, I, P 순서가 될 수 있지만 P가 이미 running이라는 사실과 D의 running membership은 별 축이다.
정렬된 waiting 목록을 보고 H가 즉시 D를 밀어냈다고 쓰면 안 된다.

tie-break도 계약이다. 같은 priority에서 arrival time을 쓰면 오래 기다린 요청이 먼저다. request ID나
heap insertion generation이 숨어 있으면 재시작·batch insert에서 순서가 달라질 수 있다. 비교 기록에는
`effective_priority`, `arrival_seq`, `eligible`, `container`를 함께 둔다. public priority field가 route에서
정규화된 뒤 scheduler object까지 같은 값으로 도착했는지도 확인한다.

KV가 부족할 때 victim 선택은 waiting comparator의 반대 끝을 사용할 수 있다. 가장 낮은 priority, 더 늦게
도착한 request가 먼저 희생되는지 작은 D/P fixture로 계산한다. 그러나 victim이 이번 step에 provisional
token grant를 이미 받았다면 그 grant와 encoder/input budget, speculative token state를 함께 환불해야 한다.
KV block만 free하고 token budget을 돌려주지 않으면 scheduler는 실제보다 가난해지고 뒤 I를 불필요하게
미룬다. 반대로 budget만 환불하고 runner membership에서 victim을 제거하지 않으면 free된 block을 읽는다.

### `_preempt_request`가 지우는 과거

recompute 방식의 preemption에서는 victim의 computed progress가 reset될 수 있다. D가 12,000-token KV를
가졌다면 victim 한 번으로 12,000 token의 과거가 논리적으로 사라질 수 있다. prefix cache에서 일부를 다시
얻더라도 그것은 preemption이 보존했다는 뜻이 아니라 resume 때 새 ownership으로 reacquire한 것이다.
원장에는 preempt 직전 computed tokens, free blocks, cacheable prefix, reset frontier와 resume 시 재계산
tokens를 따로 쓴다.

이 debt는 priority 숫자만으로 보이지 않는다. H가 200-token prompt라서 빨리 끝나더라도 D의 12,000-token
재계산이 뒤 step을 막으면 전체 strict goodput이 떨어질 수 있다. 단순 상한은 `debt_tokens × observed
prefill_cost_per_token`이지만 실제 비용은 chunk shape와 cache hit, 다시 기다린 시간까지 포함한다. 따라서
`preemption_count`보다 `unique_required_tokens`, `recomputed_tokens`, `resume_delay`가 더 직접적인 증거다.

preempt와 abort도 구분한다. preempted D는 logical request가 살아 있고 waiting으로 되돌아가야 한다. abort는
terminal이며 output collector와 KV를 닫는다. stale in-flight output이 도착하면 어느 scheduler generation의
결과인지 확인해 현재 frontier를 전진시키지 않게 한다. preemption 뒤 같은 request가 두 번 waiting에
삽입되거나 old output이 새 generation usage를 더하면 공정성 문제가 correctness 문제로 바뀐다.

### waiting과 skipped queue의 priority merge

candidate가 priority상 앞이어도 현재 memory, multimodal encoder budget, structured-output 준비 또는
connector 상태 때문에 실행 불가능할 수 있다. 구현이 ineligible H에서 break하면 뒤 I/P도 막힌다. skip해
별 목록에 두면 work-conserving하지만 H가 계속 불가능한 동안 순서 복원 규칙이 필요하다. waiting과 skipped를
합칠 때 원래 key가 보존되는지, 새 arrival 뒤로 밀리는지, 매 step predicate를 다시 평가하는지 본다.

이 fixture에서 H는 priority가 가장 높지만 600 token의 encoder budget이 필요하고 현재 잔액이 0이라고 하자.
H skip 뒤 I가 admit되면 priority inversion처럼 보일 수 있으나 comparator는 맞다. first divergence는
eligibility predicate다. 반대로 H와 I가 모두 eligible인데 I가 먼저라면 queue key 또는 field propagation을
본다. 증상이 같아도 고칠 owner가 다르다.

옵션 승인은 세 단계로 닫는다. 첫째 설정이 실제 queue policy를 바꿨는지 effective log와 comparator trace로
확인한다. 둘째 H/I의 waiting p99와 deadline goodput이 좋아졌는지 본다. 셋째 D/P의 starvation, recompute
amplification과 cache churn이 guardrail 안인지 확인한다. priority 요청만 뽑아 latency를 비교하면 피해자가
분모에서 사라진다.

### 옵션에서 goodput까지

priority scheduling을 켜는 것은 “중요 요청이 빨라진다”는 결과를 직접 보장하지 않는다. 바뀌는 상태는
waiting key와 특정 victim order다. 기대 효과는 high tier queue delay 감소다. 비용은 low tier wait tail,
preemption debt와 queue maintenance다. 관측값은 tier별 offered/admitted/completed/deadline pass,
preempted progress histogram, recompute tokens와 effective batch composition이어야 한다.

H를 위해 D를 희생한 candidate를 baseline과 비교할 때 arrival trace를 고정한다. baseline에서 H가 40ms 늦고
D가 계속 진행됐으며 candidate에서 H가 20ms 빨라졌지만 D가 12,000 token을 재계산했다면, 어느 정책이
좋은지는 tier weight와 SLO 계약에 달렸다. aggregate tokens/s 하나로 승인하지 않는다. H의 deadline pass가
제품 가치를 만들더라도 D의 deadline miss와 추가 GPU 비용을 decision record에 함께 남긴다.

D/P/I/H를 vLLM 한 iteration에 대입한다. snapshot 시작에서 running은 D와 P, waiting은 I와 H다. token
budget 512에서 D에 decode 1을 provisional grant하고 P에 511을 줄 수 있다. 그 뒤 H의 priority가 높아도
current running grant를 즉시 빼앗는지는 allocation branch와 policy에 달렸다. H admission에 필요한 KV가
부족해 P를 victim으로 고르면 P의 511 grant, encoder budget과 allocated slots를 모두 refund한 뒤 H 200,
I 40을 넣을 수 있다. 최종 scheduled 합과 balance가 맞는지 손으로 더한다.

runner manifest에는 D/H/I만 있어야 하며 P row와 block table pointer가 남으면 안 된다. P는 waiting에 정확히
한 번 있고 computed frontier가 native preempt semantics에 맞아야 한다. GPU completion 뒤 output update가
D/H/I generation을 갱신하며 P의 old in-flight 결과가 오면 current progress에 합치지 않는다. 다음 step에서
P가 resume할 때 prefix cache hit와 fresh allocation을 별 event로 기록한다.

이 수직 trace의 checkpoint는 queue key, provisional grant, allocation owner, victim release, runner row,
output generation이다. 어느 하나라도 없으면 “priority가 P를 선점했다”는 문장이 지나치게 압축됐다. 특히
free-block 증가와 runner row 제거 사이에 exception이 나면 released KV를 실행할 수 있으므로 failure injection을
두 경계 사이에 둔다.

## 30.3 SGLang의 priority·retraction·new-token ratio를 분리한다

SGLang에서는 waiting policy와 running decode memory recovery가 서로 다른 층이다. waiting request는 active
policy가 만든 순서로 prefill admission을 시도하고, decode batch가 다음 token을 위한 공간을 확보하지 못하면
running request 일부를 retract한다. 두 경로 모두 결과적으로 어떤 request를 늦추지만 하나는 선택 순서,
다른 하나는 allocator pressure에 대한 복구다.

고정 좌표는 [`SchedulePolicy.calc_priority`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_policy.py#L92-L180),
[`ScheduleBatch.retract_decode`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_batch.py#L2816-L2895)다.

정책 계산은 priority와 prefix 조건을 포함한 waiting order를 만들고, retraction은 batch victim, request/token
pool release와 이후 reserve 추정을 바꾼다. pinned source가 보여 주지 않는 aging 또는 tenant quota를 있다고
가정하지 않는다.

### policy validation은 cache availability에 따라 의미를 바꾼다

prefix-aware policy는 longest prefix match를 이용해 이미 계산된 KV를 재사용할 후보를 앞세울 수 있다.
하지만 radix cache가 비활성화됐거나 match가 실제 usable block으로 이어지지 않으면 같은 policy 이름의
효과가 달라진다. option parser에서 accepted됐다는 사실, active policy object, request별 matched prefix와
실제 cached tokens를 차례로 확인한다.

D/P/I/H에서 P가 4,000-token cached prefix를 갖고 H는 cache miss라고 하자. locality 우선은 P를 H보다
앞세워 saved prefill work를 늘릴 수 있다. 이는 numeric priority를 무시한 bug일 수도 있고, 설정된 policy
composition일 수도 있다. expected ordering을 `priority tier → locality`인지 `locality → priority`인지
source comparator로 쓰고 같은 key를 손계산한다. 정책 이름 두 개를 보았다고 composition 순서를 추측하지
않는다.

### priority와 FCFS의 합성에는 자동 aging이 없다

FCFS tie-break가 있다고 해서 낮은 priority request가 언젠가 높은 tier를 이기는 것은 아니다. FCFS는 같은
tier 안의 순서를 정할 뿐 tier 간 aging을 제공하지 않을 수 있다. high-priority arrival이 capacity를 계속
채우면 D/P의 wait bound는 여전히 무한하다. 실제 aging이 있다면 wait time이 effective score를 어느 속도로
바꾸고 최대 몇 step 뒤 경계를 넘는지 식으로 보여야 한다.

priority field가 없거나 policy가 비활성인 request를 default tier에 넣는 방식도 감사한다. tenant가 값을
생략해 의도치 않게 최고 tier가 되거나, 음수/큰 값이 validation 없이 comparator를 뒤집으면 admission
control을 우회할 수 있다. API schema, normalization, internal Req와 policy key를 하나의 trace로 잇는다.

### decode memory check와 retraction loop

decode batch가 다음 step에 필요한 KV/token space를 확보하지 못하면 retraction loop가 victim을 골라 release
한다. victim ordering은 priority, output length 또는 구현의 sort key를 읽어 확인한다. 최소 한 request를
남겨 progress를 보장하는지, 모두 retract했을 때 어떤 error/recovery가 있는지도 본다. `available_tokens`
증가만 확인하지 말고 victim의 batch row가 실행 전에 제거됐는지 검증한다.

retracted request의 request-pool index, token mapping, radix lock과 offload/cache state가 어느 경로로
정리되는지 기록한다. 일부 prefix가 cache에 남는다면 resume cost가 vLLM의 full reset과 다르다. tree cache가
없거나 partial prefix가 unfinished라면 더 많은 work를 잃을 수 있다. 공통 표에는 “preemption 지원” 대신
`victim state`, `preserved progress`, `released owner`, `resume source`를 쓴다.

P가 긴 prefill을 마치고 decode에 들어왔지만 KV reserve가 부족해 retract됐다고 하자. release로 8,000 token
capacity를 얻어 I/H가 진행할 수 있어도 P가 다시 들어올 때 같은 압력이 반복되면 storm이다. step마다
retracted count, freed tokens, preserved prefix, recompute/load tokens와 next admission을 연결한다. count만
높고 debt가 작다면 빠른 cache resume일 수 있고 count가 낮아도 16K victim 하나면 비용이 클 수 있다.

### new-token ratio는 fairness score가 아니다

retraction 뒤 scheduler가 `new_token_ratio`를 갱신하면 이는 남은 decode request의 미래 token reserve를
보수적으로 추정하기 위한 pressure state다. 값이 커졌다고 low-priority request가 더 공정하게 service를
받는다는 뜻이 아니다. ratio의 producer, old/new value, available token과 다음 step admission 결과를
같이 기록한다.

ratio가 과도하게 높으면 allocator safety는 좋아져도 새 prefill admission이 줄 수 있다. 너무 낮으면
utilization은 높지만 retraction이 반복될 수 있다. option이나 heuristic을 바꿀 때 ratio 자체를 목표로
최적화하지 않고 `retraction debt + idle reserve + tier별 goodput`을 본다. workload output-length
distribution이 달라지면 같은 값의 의미도 달라진다.

### eligibility inversion을 comparator inversion과 구분한다

H가 높은 priority인데도 I 뒤에 실행된 사건을 재현할 때 두 request의 policy key만 비교하면 부족하다.
H가 grammar 준비, multimodal input, prefix lock 또는 memory predicate에서 제외됐는지 확인한다. H가 candidate
set에 없었다면 comparator는 호출되지 않았다. first divergence는 eligibility다. 둘 다 후보였고 key가
뒤집혔다면 comparator 또는 priority normalization이다.

negative evidence도 남긴다. H의 key가 정확하고 eligible이었지만 current non-preemptible decode step 때문에
한 quantum 늦은 것은 queue bug가 아니다. H가 admit됐지만 output commit이 늦은 것은 detokenizer/stream
owner다. end-to-end priority latency를 scheduler 하나에 모두 귀속시키지 않는다.

### 옵션에서 효과까지 닫는다

SGLang policy 변경 승인표에는 active policy, tree cache status, effective priority direction, matched/usable
prefix, retraction mode와 ratio generation을 넣는다. 결과 열에는 tier별 wait, prefill/decode goodput,
retracted progress, recompute 또는 transfer bytes, allocator reserve를 둔다. 동일한 이름의 option이라도
cache/offload 조합이 달라지면 별 candidate다.

복구 종료 조건은 단순히 OOM이 사라지는 것이 아니다. D/P/I/H fixture에서 expected order와 eligibility가
맞고, victim resource가 한 번만 release되며, retracted request가 중복 없이 waiting으로 돌아가고, resume
뒤 output과 progress가 reference에 맞아야 한다. ratio와 free-token state도 workload 종료 뒤 baseline으로
수렴해야 한다.

SGLang 수직 trace에서는 waiting P/I/H에 policy를 적용해 ordered list와 prefix match를 만든다. admission이
H와 I를 넣고 P를 chunk하거나 skip한 뒤 running decode D와 merge한다. model execution 전에 decode memory
check가 future KV reserve를 만족하지 못하면 retraction이 시작된다. waiting-order priority와 decode-victim
order를 별 표로 써야 같은 “우선순위”라는 말이 두 selection을 섞지 않는다.

D를 retract하면 request/token pool과 cache/offload owner가 release되고 batch filter가 D row와 row-indexed
sampling/cache tensor를 함께 제거해야 한다. 남은 H/I/P의 available token으로 new-token ratio가 갱신된다.
ratio는 다음 reserve 예측의 generation이며 D의 fairness debt가 아니다. D는 native resume path에 한 번만
귀속되고 preserved prefix/load/recompute 중 실제 선택된 경로를 기록한다.

release 뒤 filter 전에 fault를 주입해 free된 D row가 kernel에 들어가지 않는지 확인한다. filter 뒤 waiting
enqueue 전에 fault가 나면 D가 owner 없는 orphan이 되지 않아야 한다. retry가 같은 D를 두 번 enqueue하지
않고 abort가 동시에 오면 terminal owner가 승리해야 한다. 이 세 boundary가 정상 OOM test보다 retraction
contract를 강하게 검증한다.

## 30.4 Transformers의 FIFO와 PrefillFirst는 상태 class를 정렬한다

Transformers continuous batching의 scheduler는 vLLM priority queue의 축소판이 아니다. 요청 후보를
session 상태와 cache 준비 조건으로 분류한 뒤 FIFO 또는 PrefillFirst가 batch에 넣을 순서를 정한다.
public numeric priority, running victim recompute와 같은 기능을 있다고 만들지 않는다. 비교할 것은 같은
D/P/I/H가 어느 candidate class에 들어가고 어떤 budget predicate에서 선택되는가다.

고정 좌표는 Transformers v5.15.1의
[`FIFOScheduler.schedule_batch`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/scheduler.py#L256-L329),
[`PrefillFirstScheduler.schedule_batch`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/scheduler.py#L332-L421),

공통 [`Scheduler._process_candidates`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/scheduler.py#L146-L254)다.

첫 두 함수는 후보 순서, 마지막 함수는 실제 cache/sequence 예산과 state transition을 보여 준다.

### FIFO는 decode active를 먼저 둔다

FIFO라는 이름을 “모든 request를 단일 arrival deque에서 strict order로 실행한다”로 확대하지 않는다.
continuous generation에는 이미 active decode인 session과 새 prefill 후보가 있고, scheduler는 이 상태를
구분한다. D가 active decode라면 먼저 service를 이어 받고 I/H/P가 waiting 순서로 admission될 수 있다.
exact order는 current source의 candidate assembly를 읽어 fixture에 적는다.

도착 순서가 D, P, I, H여도 P가 16K prefill이라 현재 cache allocation을 만족하지 못하면 뒤 I가 선택될
수 있는지 확인한다. head에서 멈추면 head-of-line blocking, skip하면 P starvation 후보다. FIFO는
comparator 설명일 뿐 eligibility와 break/continue semantics를 대신하지 않는다.

FIFO가 decode를 먼저 두는 이유는 기존 session의 ITL과 cache locality를 보호할 수 있기 때문이다. 그러나
decode arrivals/output lengths가 capacity를 계속 채우면 새 prefill의 TTFT bound가 사라질 수 있다. 관측은
active decode count, candidate waiting age, rejected reason과 batch class별 granted tokens를 사용한다.

### PrefillFirst는 진행 중 prefill을 먼저 둔다

PrefillFirst는 prefill 상태 후보를 decode보다 앞에 놓아 prompt ingestion을 진행시킨다. P가 이미 split
prefill 중이라면 다음 chunk가 D보다 먼저 선택될 수 있다. 이는 high user priority H를 앞세운다는 뜻이
아니다. 상태 class ordering과 tenant/request priority를 같은 열에 놓지 않는다.

P의 chunk가 512 token이고 step duration이 12ms라면 D의 ITL에는 적어도 그 non-preemptible quantum이
추가될 수 있다. prefill이 연속으로 선택되면 계단형 ITL tail이 나타난다. 반면 P의 TTFT와 cache allocation
완료는 빨라질 수 있다. phase별 goodput과 service gap을 동시에 본다.

I/H가 짧은 새 prefill이라고 해도 P와 같은 class에서 arrival order나 candidate assembly에 따라 뒤에 설
수 있다. PrefillFirst 하나로 short-job-first 효과를 주장하지 않는다. prompt length를 key로 쓰는지,
진행 중 prefill을 새 prefill보다 앞세우는지 source에서 각각 확인한다.

### 공통 candidate processor가 실제 admission을 결정한다

정렬된 후보는 `_process_candidates`가 cache availability, maximum batch/sequence 조건과 state를 심사한
뒤에야 실행 batch가 된다. candidate order와 admitted order가 다르면 first false predicate를 찾는다.
cache offload가 필요하면 H2D transfer 준비와 session state도 scheduling 결과의 일부다.

후보 처리 중 provisional allocation이 실패했을 때 이미 바꾼 session state를 rollback하는지 본다. 다음
iteration에서 같은 request가 중복 candidate가 되거나 cache owner 없이 RUNNING으로 보이면 ordering
문제가 아니라 transaction 결함이다. request ID, old/new state, candidate index, allocation/IO handle과
final batch row를 한 event로 둔다.

Transformers의 CPU offload/restore는 vLLM frontier reset과 다르다. KV payload가 host에 보존되고 다음
admission에서 H2D restore될 수 있다면 debt는 recompute token이 아니라 transfer byte·latency다. 실제
구현 branch가 offload를 선택했는지 확인하지 않고 “preemption이 cache를 보존한다”고 일반화하지 않는다.

### 두 scheduler를 vLLM priority와 등치하지 않는다

D/P/I/H 비교표에서 vLLM은 numeric priority와 allocation victim을, Transformers는 FIFO/PrefillFirst
candidate class와 common admission을 기록한다. H priority=0을 Transformers fixture에 억지로 넣지 않는다.
대신 H가 short interactive라는 product tier가 native scheduler에서 표현되지 않는다는 capability gap을
명시한다. gateway 분리나 별 replica가 필요할 수 있지만 source에 없는 scheduler guarantee를 만들지 않는다.

승인 실험은 FIFO와 PrefillFirst에 같은 arrival trace를 넣고 prefill/decode service share, waiting age,
TTFT·ITL, cache IO와 terminal correctness를 비교한다. PrefillFirst가 P TTFT를 줄이는 동시에 D ITL을
망가뜨리면 workload SLO로 판단한다. 이름만 보고 어느 쪽이 더 공정하다고 결론내리지 않는다.

D/P/I/H를 Transformers manager 한 cycle에 넣으면 먼저 session state를 고정해야 한다. D는 decoding,
P는 split prefill 진행 중, I/H는 새 waiting request다. FIFO와 PrefillFirst 각각에서 candidate list를 만들고
그 list가 `_process_candidates`에 들어가는 순간을 기록한다. 여기까지는 정렬 결과일 뿐 device batch가 아니다.
각 candidate의 cache slots, input length, remaining budget과 IO readiness가 admission을 다시 가른다.

PrefillFirst에서 P가 첫 candidate여도 cache block을 즉시 얻지 못하고 H2D restore를 기다리면 I/H가 뒤에서
진행할 수 있는지 source의 loop semantics를 본다. P에서 break하면 head-of-line, skip하면 work-conserving
대신 P debt가 생긴다. `candidate_rank`, `eligible/reason`, `IO handle`, `selected row`를 한 event로 두면 class
order와 resource 결과를 분리할 수 있다.

batch가 확정되면 session state가 PREFILLING/DECODING 가운데 무엇으로 바뀌고 model input row와 cache mapping이
어떻게 생기는지 잇는다. output retrieval 뒤 generated token, finished predicate와 cache owner가 갱신된다.
allocation 뒤 model submit 전에 exception을 넣어 provisional session state가 waiting으로 복귀하는지,
cache block과 IO future가 유일 owner에게 남는지 확인한다. 단순 schedule 함수 반환값만 시험해서는 transaction을
닫지 못한다.

offload fixture에서는 D cache를 CPU로 옮긴 뒤 P/I/H를 실행하고 D를 restore한다. 보존된 logical progress,
offloaded bytes, D2H/H2D completion, resume generation과 first output을 기록한다. restore가 실패하면 D를
terminal error로 닫을지 cold recompute할지 native contract를 따른다. vLLM reset과 수치상 같은 wait가 나와도
원인은 transfer debt이므로 recompute counter로 설명하지 않는다.

FIFO 회귀는 D가 끝날 때까지 P/I/H가 무조건 기다리는지만 보지 않는다. batch capacity가 남을 때 새 prefill이
함께 들어오는지, active decode가 늘 때 admission이 어느 predicate에서 멈추는지 확인한다. PrefillFirst 회귀는
P chunk boundary 1/2/마지막과 D context bucket을 교차한다. class order가 같아도 chunk completion과 cache
availability가 달라 service gap이 달라질 수 있다.

최종 trace는 `manager request→Future/session state→scheduler candidate→common processor→cache/IO owner→model
row→output update→finish/free`를 한 줄로 잇는다. 이 경로에서 native numeric priority가 없다는 사실도 중요한
판정이다. 제품이 H tier 보장을 요구하면 외부 admission/replica 분리 또는 구현 확장이 필요하며 FIFO와
PrefillFirst 이름만으로 요구사항을 충족했다고 승인하지 않는다.

## 30.5 llama.cpp의 LCP·LRU는 queue priority가 아니라 slot placement다

llama.cpp server는 task가 실행할 slot을 고를 때 explicit slot, prompt와 cached context의 longest common
prefix, idle slot의 recency를 사용할 수 있다. 이는 waiting request의 numeric priority나 running request
preemption이 아니다. 먼저 task queue에서 어떤 task가 slot selection에 도달했는지, 그다음 available
slot 가운데 어느 것을 배치했는지 두 단계를 나눈다.

고정 좌표는 llama.cpp commit `bb4caa7540188872173c44d161602d9271386413`의
[`server_context::launch_slot_with_task`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L914-L1035)와
[`server_slot::process_prompt`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1770-L1905)다.
첫 경계는 task/slot placement, 둘째는 선택된 slot의 prompt cache keep/remove와 evaluation을 보여 준다.

### LCP는 prompt reuse를 최대화하는 placement다

idle slot A가 prompt `[s,u,x]`, slot B가 `[s,v]`를 cache했고 새 P가 `[s,u,y]`라면 A의 LCP가 2, B가
1이다. A를 선택하면 두 token을 유지하고 divergence 이후를 제거·평가할 수 있다. 이 선택은 saved prefill
work를 늘리지만 P가 queue에서 I보다 먼저 service를 받는 정책과는 별개다.

LCP threshold가 absolute token이 아니라 prompt length 대비 ratio를 쓰면 긴/짧은 prompt의 선택이 달라진다.
LCP tokens, ratio, slot active/idle, explicit slot constraint와 selected reason을 로그에 둔다. LCP가 가장
큰 slot이 active라 unavailable이면 다음 candidate 또는 LRU로 갈 수 있다. 이를 comparator inversion이라
부르지 않는다.

cache-local request를 계속 같은 slot에 붙이면 throughput은 좋아질 수 있지만 cache-poor tenant의 wait가
늘 수 있다. task queue가 arrival order를 보장해도 slot이 없어 defer된 task와 새 task의 requeue 순서에서
starvation이 생길 수 있다. queue age와 saved prompt tokens를 tenant별로 함께 본다.

### LRU는 idle slot replacement heuristic다

useful LCP가 없으면 least recently used idle slot을 골라 새 task를 배치할 수 있다. LRU timestamp는 request
waiting age가 아니라 slot cache의 최근 사용 시각이다. 오래 기다린 request를 우대하는 aging으로 번역하지
않는다. 또한 running slot을 빼앗는 victim policy가 아니라 available/idle 후보의 cached context를 희생하는
placement일 수 있다.

slot C가 8K cached prompt를 갖고 오래 idle했고 slot D가 200-token cache를 최근 사용했다고 하자. LRU는 C를
선택해 더 큰 cached work를 버릴 수 있다. recency는 future reuse 확률의 heuristic이지 eviction debt를
직접 최소화하지 않는다. removed prompt tokens와 subsequent miss/recompute를 기록해 비용을 판정한다.

### eviction cost를 계산한다

slot 교체 비용은 `discarded_cached_tokens × prefill_cost`의 미래 기대값과 새 request가 얻는 즉시 service
가치의 교환이다. RAM prompt cache로 state를 저장하거나 context shift가 있으면 byte IO와 compatibility
검사를 추가한다. cached token 수와 실제 reusable prefix는 다를 수 있으므로 다음 request LCP distribution을
사용한다.

P를 cache-rich slot에 붙여 4K work를 절약했지만 I가 50ms 더 기다렸다면 aggregate throughput과 interactive
goodput이 반대 방향으로 움직일 수 있다. “LCP policy가 빠르다”가 아니라 saved compute, task wait, slot
occupancy와 tier SLO를 같은 workload에서 비교한다.

### 네 구현의 공통 비교 좌표

vLLM priority는 waiting/victim key, SGLang policy는 waiting order와 decode retraction, Transformers는
state-class candidate order, llama.cpp는 task 이후 slot placement다. 네 구현 모두에 공통 priority 점수를
만들지 않는다. 공통 좌표는 `candidate owner`, `eligibility`, `selection key`, `non-preemptible quantum`,
`displaced state`, `resume cost`, `starvation escape`다.

llama.cpp 사건의 first divergence가 task queue인지 slot selection인지 먼저 정한다. task가 selection에
도달하지 않았다면 LCP/LRU를 고칠 이유가 없다. selection은 맞지만 prompt keep/remove가 틀리면 cache
processing owner다. final output만 보고 scheduler fairness를 탓하지 않는다.

D/P/I/H를 server slot fixture로 바꾸면 D는 active slot에서 decode 중이고 P는 idle slot A와 4K LCP를 가진
task, I는 짧은 cache miss, H는 explicit slot constraint가 없는 새 task다. task queue가 P/I/H를 꺼내는
순서는 slot LCP와 별도로 기록한다. P가 먼저 `launch_slot_with_task`에 도달했다면 A 선택은 placement 결과이지
P를 queue에서 우선한 LCP scheduler의 증거가 아니다.

모든 slot이 active면 task가 deferred되는 위치와 wake-up owner를 찾는다. D가 finish해 slot을 release한 뒤
deferred task가 main queue에 한 번만 돌아오는지, 새 arrival과의 순서는 무엇인지 본다. finish와 client
disconnect가 겹쳐 slot release callback이 두 번 task를 깨우면 같은 task가 두 slot에 들어갈 수 있다.
task id, deferred generation, selected slot id와 launch generation을 함께 assert한다.

P가 slot A를 선택하면 prompt processing은 cached sequence와 new prompt의 LCP를 다시 검산하고 divergence
뒤 KV를 remove한 다음 suffix를 evaluate한다. selection 시 계산한 LCP와 process 시 실제 kept tokens가 다르면
그 차이가 context shift, special token, slot state mutation 때문인지 본다. kept tokens를 selection reason과
동일하다고 가정하지 않는다.

I가 cache miss로 LRU slot C를 골랐을 때 C의 old prompt가 future request에 다시 필요할 가능성은 즉시
관측되지 않는다. discard tokens, old slot age, new prompt cost와 이후 miss를 event로 남긴다. LRU timestamp가
request waiting age와 혼동되지 않게 field 이름과 owner를 분리한다. P의 높은 LCP가 I waiting을 늘렸다면
queue order인지 available-slot scarcity인지 first divergence로 좁힌다.

release trace는 output finish, slot state transition, task result delivery, cached prompt 보존과 next deferred
wake-up을 잇는다. slot이 idle이 되었다고 cache가 free됐다고 쓰지 않고, cache가 남았다고 running owner가
있다고 쓰지 않는다. abort에서는 valid output boundary와 KV keep policy가 달라질 수 있으므로 normal finish와
같은 expected state를 강제하지 않는다.

회귀 matrix는 explicit slot success/failure, LCP ratio threshold 바로 아래·같음·위, no-LCP LRU, all-slots-active
defer, finish/abort wake-up과 context-shift를 포함한다. 각 case는 selected reason, kept/removed tokens, wait,
slot generation과 output parity를 검증한다. tokens/s만 좋아졌다는 결과로 task duplication이나 stale cache를
허용하지 않는다.

이 수직 경로를 닫으면 llama.cpp를 “priority가 없는 구현” 한 줄로 축소하지 않는다. 그것은 queue priority와
다른 placement/locality 문제를 해결한다. 제품 선택에서는 slot 수, prompt repetition, task wait SLO와 cache
discard 비용을 함께 보고, numeric tier 보장이 필요하면 capability gap을 명시한다.

## 30.6 fairness·aging·debt를 같은 단위로 계산한다

공정성은 모든 request에 같은 token 수를 주는 규칙이 아니다. prefill token과 decode token은 비용이
다르고, 긴-context decode는 같은 한 token이라도 더 많은 KV를 읽는다. 먼저 무엇을 공정하게 나눌지
정한다. request service opportunity, query tokens, measured GPU time, KV byte, deadline 안의 useful output은
서로 다른 분배다.

### token fairness와 time fairness

D에 decode 1 token, P에 prefill 511 token을 준 step은 token 수로 극단적으로 불공정해 보인다. 그러나 P의
큰 GEMM과 D의 long-KV attention이 쓰는 시간은 token 비율과 같지 않다. profiler 없이도 phase/shape bucket의
평균 service cost로 first estimate를 만들고, 실제 device interval과 collective wait로 보정한다.

tenant별 service `x_u`를 query token으로 잰 Jain index와 GPU microsecond로 잰 index를 함께 계산하면 정책이
어느 자원을 편향시키는지 보인다. 높은 index가 항상 제품적으로 좋지는 않다. high tier H에 더 큰 weight를
주기로 계약했다면 normalized service `x_u/w_u`와 tier별 deadline goodput을 본다.

### weighted fairness와 tenant 분할 공격

request별 round-robin은 한 tenant가 요청을 잘게 쪼개면 더 많은 차례를 얻는 분할 공격에 취약하다. 공정성
owner를 request ID가 아니라 authenticated tenant/project로 올리고, tenant 안에서 request policy를 둔다.
긴 요청 하나와 짧은 요청 백 개를 같은 request count로 비교할지 token/time debt로 비교할지 명시한다.

priority tier weight도 무한 service entitlement가 아니다. H tier arrival이 전체 capacity를 넘으면 lower tier에
보장량을 남길 quota 또는 admission이 필요하다. 그렇지 않으면 scheduler comparator가 정확해도 starvation은
설계 결과다. offered load와 admitted load를 tier별로 나누지 않으면 낮은 tier가 사라진 사실이 aggregate
goodput에 숨는다.

### aging의 wait bound

base priority `p`, waiting time `w`, aging rate `a`에 대해 작은 score 우선인 예를 `effective=p-a·w`로 둘 수
있다. low tier가 high tier와 score가 같아지는 시간은 `(p_low-p_high)/a`다. 이것은 실제 구현의 식이 아니라
wait bound를 검토하는 worksheet다. source에 aging이 없다면 옵션처럼 제시하지 않는다.

aging이 있어도 candidate가 계속 ineligible이면 service는 보장되지 않는다. score가 가장 높아진 P가 KV
allocation을 만족하지 못하면 skip될 수 있다. wait-bound 주장은 comparator 도달 조건, resource feasibility와
non-preemptible quantum을 포함해야 한다. queue age가 줄지 않는데 effective rank만 좋아지는 dashboard는
완료 증거가 아니다.

### non-preemptible quantum이 priority latency 하한을 만든다

H가 도착한 직후 512-token prefill kernel과 collective가 시작됐다면 scheduler는 보통 이를 중간에 안전하게
절단하지 않는다. H의 추가 wait 하한은 current device work, stream dependency와 scheduler output reconcile
시간이다. chunk를 줄이면 priority 반응은 빨라질 수 있지만 launch overhead와 prefill efficiency가 악화된다.

quantum sweep은 128/256/512에서 H wait, P completion, D ITL, tokens/s와 launch count를 함께 본다. 가장 작은
quantum이 가장 공정하다는 결론도, 가장 큰 quantum이 가장 효율적이라는 결론도 workload 없이 성립하지
않는다. policy와 quantum은 별 option이지만 효과가 곱해진다.

### recompute와 swap의 break-even

victim progress `S`, observed recompute cost `c`, KV payload `B`, effective offload bandwidth `W`, setup/동기화
비용 `t0`라면 대략 `T_recompute=S·c`, `T_swap=2B/W+t0`를 비교한다. swap은 CPU memory pressure와 PCIe
contention을 만들고 recompute는 GPU work와 queue delay를 만든다. 평균값 하나 대신 S와 context bucket별로
경계를 구한다.

prefix cache가 S의 일부를 되살리면 recompute debt는 줄지만 cache hit가 eviction 압력 아래에도 유지되는지
확인한다. offload payload가 준비됐어도 restore가 critical path에서 기다리면 latency debt가 남는다. “state를
보존했다”와 “빠르게 resume했다”를 분리한다.

### preemption amplification과 useful goodput

logical request가 정답을 위해 필요로 한 unique compute token을 U, 실제 실행한 prefill/decode/verification
token을 E라 하면 amplification `A=E/U`다. preemption storm에서는 raw throughput이 높아도 A가 커지고 strict
goodput이 낮아진다. tier별 A와 deadline pass를 함께 본다.

D를 세 번 reset해 12K를 반복 계산했다면 preemption count 3보다 recompute 36K가 핵심이다. 반대로 작은
victim 열 번이 prefix cache로 즉시 복구되면 count는 높지만 debt가 작을 수 있다. counter 이름을 원인으로
쓰지 않고 progress delta를 계산한다.

### starvation을 유한 실험으로 반증하는 방법

무한 실행을 기다리지 않고 정책의 escape path를 시험한다. high tier arrival을 capacity보다 조금 높게 유지한
채 low tier P 하나를 넣고, aging/quota가 있다면 계산된 bound 안에 candidate와 service를 얻는지 본다. 없다면
starvation 가능성을 문서화하고 admission으로 high tier를 제한해야 한다.

유한 run에서 P가 우연히 한 번 실행됐다고 starvation-free가 증명되지는 않는다. 반복 seed/arrival phase,
cache hit·miss와 eligibility 조건을 바꾼다. 최대 waiting age, consecutive skipped steps, effective rank와
실제 granted work를 기록한다. queue에서 제거된 rejection을 service로 세지 않는다.

### priority inversion의 세 종류

첫째 comparator inversion은 priority normalization이나 부호가 틀려 둘 다 eligible인데 I가 H보다 앞서는
경우다. 둘째 eligibility inversion은 H가 resource/grammar/media 조건에서 후보가 아니어서 I가 실행되는
경우다. 셋째 execution inversion은 H가 선택됐지만 이전 low-tier kernel 또는 collective 때문에 시작이
늦는 경우다. 세 증상은 비슷하지만 수정 지점이 다르다.

trace에는 API priority, normalized key, candidate membership/reject reason, selected step, submit/start/end를
둔다. 이 열로 first divergence를 찾기 전에는 scheduler comparator를 고치지 않는다. priority 문제처럼
보이는 output backpressure도 selected/start가 정상이라면 다음 owner로 넘긴다.

### victim debt-aware 정책을 평가하는 계산

debt-aware victim score는 priority뿐 아니라 lost progress, reusable prefix, offload 가능성과 resume deadline을
고려할 수 있다. 하지만 source에 없는 정책을 네 구현에 있다고 쓰지 않는다. 새 설계를 평가할 때만
`service_gain(H)-resume_debt(victim)-fairness_penalty` 형태로 decision record를 만든다.

debt를 너무 크게 벌주면 긴 D가 사실상 선점 불가능해져 H SLO가 무너지고, 너무 작게 보면 storm이 난다.
D progress 1K/12K, H deadline 20/100ms와 cache hit 유무를 교차한 fixture로 경계가 예측대로 움직이는지 본다.
정책 변경 뒤 선택된 victim뿐 아니라 선택되지 않은 후보의 score와 reject 이유도 sampled trace에 남긴다.

## 30.7 여섯 사건에서 최초 불일치를 찾는다

여섯 사건은 서로 다른 policy 이름을 암기하는 목록이 아니다. 모두 D/P/I/H fixture의 동일한 원장을
사용한다. `arrival, normalized key, eligible/reject reason, container, selected quantum, victim progress,
released state, resume work, first/last output, terminal reason`을 기록한다. 증상 뒤 첫 열부터 비교해 최초로
달라진 owner에서 조사를 멈춘다.

### 사고 A: priority 숫자 부호가 뒤집혔다

H=0, I=1, D/P=5이고 작은 값 우선이라는 API 계약인데 trace의 normalized key가 H=5, P=0으로 나타난다.
둘 다 eligible인데 P가 먼저 선택된다. first divergence는 route→internal Req normalization이며 allocator나
kernel을 볼 필요가 없다. source comparator가 큰 값 우선이라면 API adaptation에서 방향을 변환하거나 계약을
정정한다.

수정 후 같은 priority, 음수, 최대값과 batch insert 순서를 시험한다. expected order뿐 아니라 tie-break와
victim 반대 방향을 확인한다. startup log의 policy 이름만으로 통과시키지 않고 selected request sequence를
assert한다. H latency가 좋아졌어도 lower tier starvation과 recompute debt guardrail을 다시 본다.

### 사고 B: priority inversion처럼 보인 eligibility skip

H가 가장 높은 key인데 multimodal preprocessing이 끝나지 않아 candidate set에 없고 I가 실행된다. dashboard는
I selected와 H waiting만 보여 priority inversion처럼 보인다. eligibility trace에 `media_not_ready`가 있으면
comparator는 반증된다. 문제는 preprocessing deadline, admission-before-ready 또는 skip reinsertion owner다.

H가 ready된 뒤에도 계속 skip되면 stale predicate나 notification loss를 본다. ready timestamp, next schedule
step과 first candidate membership을 잇는다. eligibility가 false인 시간을 priority waiting time과 합쳐도 되지만
원인 label은 분리해야 한다.

### 사고 C·D: preemption storm과 low-priority starvation

사고 C에서는 H tier burst가 들어올 때마다 D가 victim이 된다. D는 12K progress를 잃고 resume하지만 다음
H가 오면 다시 reset된다. GPU utilization과 raw token/s는 높고 OOM도 없다. 그러나 actual computed tokens가
unique required tokens의 2.8배이고 D의 deadline goodput은 0에 가깝다. preemption counter만 보면 “정책이
작동한다”고 오독할 수 있다.

first divergence는 H admission 자체가 아니라 같은 D를 반복 victim으로 고른 뒤 debt를 policy state에
반영하지 않는 순간이다. event마다 victim progress, cacheable/reacquired prefix, freed blocks, resume generation과
다음 victim score를 남긴다. D가 12K를 잃었다는 사실이 다음 선택에 아무 영향이 없다면 storm 가설이 강하다.
반증은 실제 prefix cache가 11.5K를 복구해 recompute가 작거나, D가 deadline 밖의 best-effort tier라 product
contract상 허용되는 경우다. 그래도 GPU 비용은 기록한다.

즉시 완화는 high-tier admission cap, victim cooldown, larger non-preemptible region 또는 별 replica일 수 있다.
어느 완화도 보편적 답은 아니다. cooldown은 H tail을 늘리고 큰 quantum은 priority 반응을 늦추며 replica
분리는 capacity fragmentation을 만든다. 수정 후보마다 H deadline pass, D progress/deadline, recompute
amplification, queue recovery와 total useful GPU time을 비교한다.

종료 fixture는 H burst 전·중·후 세 구간을 갖는다. burst가 끝난 뒤 D가 유한 시간 안에 resume하고 queue age,
recompute rate와 cache occupancy가 baseline으로 돌아와야 한다. H를 끈 뒤에도 D가 stale skipped queue에 남거나
ratio/reserve가 과도하게 보수적으로 유지되면 복구가 아니다. rollback으로 original policy를 복원했을 때
selected order와 debt 곡선도 함께 돌아오는지 본다.

사고 D는 crash나 storm 없이 P가 영원히 waiting에 남는 경우다. strict high priority arrivals가 service capacity
이상이고 aging/quota가 없다. 매 step comparator는 정확하며 eligible H가 항상 P보다 앞선다. first divergence를
한 code bug로 찾을 수 없는 정책 사건이다. 이때 “P가 선택되지 않은 첫 줄” 대신 finite service bound가
존재하지 않는다는 contract gap을 판정한다.

관측 window가 짧으면 P가 아직 기다리는 것과 starvation을 구분하기 어렵다. high-tier offered/admitted rate,
service capacity, P effective rank, consecutive skip count와 escape predicate를 수집한다. policy에 aging이
없고 high tier가 포화하며 quota도 없다면 무한 wait 가능성을 source와 load condition으로 닫을 수 있다.
우연히 traffic gap이 생겨 P가 한 번 실행된 것은 guarantee가 아니다.

수정은 낮은 tier 최소 share, maximum consecutive high-tier quanta, aging 또는 admission partition 가운데 제품
계약에 맞는 것을 고른다. 최소 share 5%는 P wait bound를 만들지만 H capacity를 5% 줄일 수 있다. aging은
긴 wait 뒤 tier를 섞고, partition은 burst borrowing과 idle capacity 정책이 필요하다. 선택 문서에는 보장하는
단위가 request, token, GPU time 중 무엇인지 적는다.

회귀는 tenant가 요청을 쪼개 quota를 우회하는 경우도 포함한다. tenant-level owner를 유지하지 않으면 low-tier
minimum share가 attacker의 많은 request에 분산된다. cancellation/retry가 새 arrival age를 얻어 fairness debt를
초기화하는지도 본다. logical request/attempt identity를 분리해 retry가 aging을 악용하거나 영구 후순위가 되지
않게 한다.

### 사고 E·F: locality 편향과 PrefillFirst의 ITL 계단

사고 E에서 SGLang prefix-aware policy 또는 llama.cpp slot locality가 cache-rich tenant A의 request를 계속
유리하게 배치한다. A는 8K common prefix를 갖고 tenant B는 매번 unique prompt다. total prefill tokens/s와
cache hit는 좋아지지만 B의 TTFT p99가 SLO를 넘는다. LCP/radix selection 자체는 기대대로 작동한다.

first divergence는 “잘못된 cache match”가 아니라 locality saving을 waiting debt와 결합하지 않은 정책
경계다. A/B별 offered requests, matched/usable prefix, saved GPU time, waiting age, selection reason과 deadline
pass를 같은 timeline에 놓는다. B가 실제로 resource-ineligible이거나 더 낮은 contracted tier라면 unfairness
주장을 좁힌다. 같은 tier·eligible인데 cache miss 하나 때문에 service bound가 없다면 편향이다.

locality를 완전히 끄면 fairness는 좋아져도 capacity가 줄어 모두의 SLO가 나빠질 수 있다. 후보는 locality
bonus에 상한을 두거나 age debt가 threshold를 넘으면 override하고, tenant별 locality queue를 round-robin하는
방식이다. 비교식은 `saved_compute - added_wait_penalty`이며 두 항의 단위를 GPU time과 SLO miss cost로
명시한다. cache hit율 하나를 목적 함수로 두지 않는다.

사고 F는 Transformers PrefillFirst 또는 큰 prefill quantum 뒤 D의 ITL이 10ms, 10ms, 48ms처럼 계단을
그리는 사건이다. P의 chunk가 들어간 step과 긴 ITL이 정확히 겹치고 decode-only fixture에서는 D kernel
duration이 정상이다. first divergence는 decode attention backend가 아니라 candidate class order와 service
quantum이다.

trace에는 step마다 prefill/decode candidate count, selected class, query tokens, start/end, D last-service age를
둔다. P chunk를 512에서 128로 낮췄을 때 longest D gap이 예측대로 줄고 launch/overall prefill time이 늘면
quantum 가설이 지지된다. backend를 바꿔도 같은 gap이면 kernel 가설이 약해진다. GPU clock, graph miss와
KV pressure는 negative evidence로 함께 남긴다.

수정은 FIFO로 단순 전환하는 것만이 아니다. maximum consecutive prefill chunks, decode guard quantum 또는
SLO-aware class alternation을 평가할 수 있다. source에 없는 기능은 현재 옵션처럼 쓰지 않고 설계 후보로만
구분한다. baseline/candidate에 P completion, I/H TTFT, D ITL p50/p99, tokens/s와 cache/graph state를 모두
넣는다.

여섯 사건의 공통 교훈은 선택 결과만 보면 owner를 틀리기 쉽다는 것이다. selected request 이전의
normalization·eligibility·comparator, 이후의 non-preemptible execution·output commit을 분리한다. preemption은
victim 선택 이후 release·resume·debt까지 닫아야 한다. 공정성은 한 시점의 order가 아니라 시간에 따른
service와 escape condition이다.

## 30.8 관측·승인·다음 async handoff

운영에서 scheduler를 설명하려면 queue depth 하나보다 긴 원장이 필요하다. request별 raw/effective priority,
arrival sequence, tenant/tier, prompt/decode phase, computed progress, cached prefix, KV bytes와 deadline을
기본 identity로 둔다. 각 step에는 eligible 여부와 reject reason, policy key, candidate rank, granted query
tokens, selected batch row, victim과 released owner를 붙인다. resume에서는 preserved/recomputed/loaded state와
generation을 잇는다.

이 원장은 항상 production metric label로 내보내는 표가 아니다. priority, reason, phase와 bounded bucket은
metric에 적합하지만 request ID, raw tenant, exact prompt hash와 queue key는 sampled trace/event store에 둔다.
high-cardinality field를 Prometheus label에 넣어 monitoring 자체가 scheduler를 방해하지 않게 한다. 개인 정보가
있는 prompt나 KV content는 기록하지 않고 safe identity와 길이·세대만 쓴다.

incident timeline의 최소 열은 다음과 같다.

```text
step | rid/gen | tenant/tier | state/container | eligible/reason
     | effective key | Q grant | KV before/after | victim/debt
     | submit/start/end | first/last output | terminal
```

한 행에서 state와 container를 구분한다. request가 RUNNING이어도 current runner batch에 없을 수 있고,
preempted여도 old in-flight output이 도착할 수 있다. KV before/after에는 logical computed count와 physical
owner generation을 나눈다. free blocks가 늘었다는 사실만으로 victim cleanup 완료를 선언하지 않는다.

fairness dashboard는 결과와 원인을 층으로 나눈다. 첫 층은 tier/tenant별 offered, admitted, deadline-goodput,
TTFT·ITL과 completion이다. 둘째는 waiting age distribution, maximum consecutive skips와 service share다.
셋째는 selected policy/reason, preemption/retraction count와 progress debt, cache locality saving이다. 넷째는
GPU time, recompute/load bytes와 useful work다.

aggregate Jain index가 좋아져도 high tier deadline이 무너질 수 있고, high tier goodput이 좋아져도 low tier가
0 service일 수 있다. dashboard 첫 화면에 단일 fairness score를 두지 않는다. product contract의 minimum
tier guardrail과 전체 efficiency를 함께 보인다. tenant 수가 변하면 index가 변하므로 cohort와 observation
window를 고정한다.

queue age metric은 현재 waiting만 보면 이미 rejected/canceled/starved request가 사라진다. terminal reason과
time-in-queue를 completion record에 남긴다. retry attempt가 새 request로 들어오면 logical identity로 합쳐
amplification을 본다. oldest age가 낮아진 이유가 빠른 service인지 빠른 rejection인지 구분한다.

preemption dashboard도 count를 앞세우지 않는다. lost computed tokens, prefix reacquired tokens, swap bytes,
resume delay와 amplification distribution을 보여 준다. same victim 반복 횟수와 victim tier를 둬 storm과
편향을 찾는다. SGLang retraction은 ratio old/new와 allocator reserve, Transformers offload는 transfer
completion, llama slot replacement는 discarded reusable tokens로 native semantics를 보존한다.

배포 승인 전에 D/P/I/H canonical fixture를 세 workload로 확장한다. 첫째 low load에서는 policy overhead와
ordering correctness를 본다. 둘째 near-saturation에서는 tier deadline과 locality 이득을 본다. 셋째 overload
burst에서는 starvation escape, victim debt와 recovery time을 본다. 한 concurrency 점의 평균으로 승인하지
않는다.

baseline과 candidate는 같은 arrival trace, model/tokenizer/backend, cache warm state, GPU clocks와 deadline을
사용한다. policy option 외에 chunk size, KV capacity, prefix cache와 async scheduling을 동시에 바꾸지 않는다.
불가피하면 effective state manifest를 남기고 factorial 또는 단계별 실험으로 first cause를 분리한다.

correctness gate는 output token parity만이 아니다. preempt/retract 뒤 request progress, KV generation,
waiting/running membership, duplicate output와 terminal cleanup을 확인한다. policy가 빠르지만 stale output을
새 resume에 commit하거나 victim block을 두 번 free하면 즉시 reject한다. fairness 개선으로 correctness
failure를 상쇄하지 않는다.

성능 gate에는 tier별 strict goodput, TTFT/ITL p50·p99, useful tokens/GPU-second, queue age와 amplification을
둔다. H가 좋아진 만큼 D/P가 나빠져도 계약한 weighted/minimum 조건을 통과하는지 본다. confidence interval과
반복 run을 사용하고 rare starvation/correctness fixture는 평균에 희석하지 않고 zero-tolerance로 둔다.

recovery gate는 overload를 제거한 뒤 queue age, victim debt, allocator/cache state, ratio/priority generation과
latency가 baseline으로 돌아오는 시간이다. process restart로 그래프가 좋아졌다면 policy 복구 증거가 아니다.
candidate option을 rollback했을 때 comparator, candidate class와 slot selection reason도 원래 상태로 돌아와야
한다.

승인 문장은 구체적으로 쓴다. “priority를 켜서 빨라졌다” 대신 “동일 offered trace에서 H의 normalized key와
waiting order가 의도대로 바뀌어 deadline goodput이 8%p 상승했고, D/P minimum service와 recompute
amplification 1.15 guardrail을 통과했으며, overload 종료 30초 안에 queue/debt가 baseline으로 복귀했다”처럼
원인과 한계를 붙인다. 수치는 실제 실험에서 채우며 이 장에서 만들지 않는다.

반려 문장도 owner를 가리킨다. H가 늦은 이유가 eligibility였다면 comparator option을 반려하고 preprocessing
owner로 돌린다. ITL 계단이 prefill quantum이면 backend 교체를 반려하고 chunk/class policy를 실험한다.
recompute debt가 크면 victim score, cache resume 또는 admission cap을 검토한다. “scheduler가 나쁘다”로
끝내지 않는다.

source upgrade에서는 public option/default, priority normalization, queue container/key, eligibility predicate,
victim/release ordering과 metrics definition을 diff한다. pinned line이 이동하면 symbol과 surrounding mutation을
새 revision에 다시 매핑한다. test가 같은 이름을 유지해도 policy composition 순서나 fallback이 달라질 수
있다.

네 구현 비교는 capability 점수표가 아니라 workload decision이다. numeric tier가 필요한 multi-tenant
service라면 native priority와 starvation guard를 요구한다. prefix 반복이 크고 slot-local 단순성이 중요하면
LCP placement가 적합할 수 있다. long prefill TTFT와 interactive ITL의 비율에 따라 FIFO/PrefillFirst 또는
chunk guard가 달라진다. preemption debt가 비싸면 admission headroom이나 state-preserving path의 가치가 커진다.

option consumer 감사에서는 CLI 이름과 결과 사이를 다섯 세대로 나눈다. requested value, parser가 정규화한
value, scheduler가 구성한 policy object, request별 effective key, 실제 selected/victim event다. 예를 들어
priority scheduling flag가 켜졌어도 request protocol이 priority를 전달하지 않으면 모든 key가 default다.
prefix policy를 골랐어도 cache가 disabled이면 fallback이 active할 수 있다. PrefillFirst class를 만들었어도
manager가 다른 scheduler를 주입하면 실행되지 않는다. llama slot option이 LCP threshold를 바꿔도 task queue
order는 그대로다.

각 option card에는 consumer symbol, mutated state, expected direct observation, downstream benefit, cost와
rollback을 적는다. `max_running_requests`를 늘리면 active candidate와 KV pressure가 늘 수 있지만 priority
comparator 방향은 바뀌지 않는다. chunk를 줄이면 reaction quantum이 줄지만 queue key는 같다. prefix cache를
켜면 locality predicate와 resume debt가 바뀔 수 있다. 서로 다른 state를 바꾸는 옵션을 한 “fairness tuning”
묶음으로 동시에 적용하지 않는다.

D/P/I/H 승인표는 첫 행에 입력을 고정한다. arrival 0/10/20/30ms, D progress 12K, P progress 4K/16K,
I prompt40/output20, H prompt200, priority와 deadline, Q budget512, KV free/reserve를 쓴다. 둘째 행은 각
step snapshot의 eligible set와 key다. 셋째는 grant와 victim, 넷째는 device quantum과 output commit, 다섯째는
resume debt와 terminal outcome이다. 이 다섯 행이 있어야 같은 결과를 새 revision에서 재현한다.

baseline/candidate의 first divergence가 key라면 policy change가 의도대로 작동했다. eligibility에서 갈리면
resource/config side effect다. grant는 같고 device start만 다르면 async/stream owner다. victim은 같은데 debt가
다르면 cache/offload/release semantics가 바뀌었다. final goodput만 달라졌다면 어느 행이 원인인지 아직 모른다.

관측 completeness도 승인 조건이다. selected event가 sampled되어 victim 일부가 보이지 않거나, retry가
logical request로 join되지 않거나, queue age가 terminal request를 잃으면 fairness 결론을 좁힌다. 계측을
추가한 candidate가 더 느려질 수 있으므로 always-on bounded metrics와 short diagnostic trace를 나눈다.
source semantics가 확실해도 production branch selection은 effective manifest와 trace로 확인한다.

failure injection은 comparator unit test를 넘어선다. allocation failure를 정확한 grant 뒤에 주입해 provisional
budget과 victim rollback을 본다. retraction release 중 exception을 주어 batch row가 실행되지 않는지 확인한다.
offload completion을 늦춰 candidate skip/retry order를 본다. llama slot을 모두 active로 만들어 deferred task의
requeue 순서를 본다. 각 실패 뒤 queue membership, KV/ref owner, output generation과 metrics가 기준값으로
돌아와야 한다.

이 fixture들은 정상 순서만 고정하지 않는다. 선택 직전 취소, victim release 직후 abort, resume와 late
output의 교차를 넣어 같은 logical request가 두 queue나 두 runner row를 동시에 소유하지 않는지도 검증한다.

마지막 handoff는 시간축이다. 30장에서는 다음 step의 service owner와 victim을 동기적으로 판정했다. 31장의
async scheduling에서는 CPU가 future batch를 준비하는 동안 current GPU work와 previous output이 겹친다.
priority가 바뀌거나 abort가 와도 이미 준비한 future batch가 stale할 수 있다. 따라서 이 장의 `policy
generation, selected rid/gen, provisional grant, victim debt`가 다음 장의 future-state 입력이다.

async가 선택 의미를 바꾸어서는 안 된다. 같은 snapshot과 policy generation이라면 sync/async scheduler가
같은 eligible set과 order를 만들어야 한다. 달라진다면 future snapshot 시각, output reconciliation 또는
rollback edge를 찾는다. 성능 overlap을 얻으면서 공정성·priority 의미를 보존했는지가 다음 질문이다.

종합하면 scheduler 공정성은 “누가 먼저인가” 한 문장이 아니다. 누가 후보가 되었고 어떤 key로 선택됐으며,
얼마나 긴 quantum을 받았고, 누구의 어떤 과거를 지웠으며, 피해자가 언제 어떤 비용으로 돌아왔는지를
시간축으로 설명하는 계약이다. 이 계약이 닫히면 서로 다른 네 구현을 억지로 같은 기능표에 넣지 않고도
자기 workload에 맞는 정책과 실패 비용을 선택할 수 있다.

최종 회귀 묶음은 여섯 사건을 독립 test로 흩어 놓지 않고 같은 fixture generator에서 만든다. priority
normalization 변형은 H/I key만, eligibility 변형은 H readiness만, storm 변형은 H arrival burst만 바꾼다.
starvation 변형은 high-tier offered rate를 capacity 위로 고정하고, locality 변형은 A/B prefix distribution,
PrefillFirst 변형은 P chunk와 D context만 바꾼다. 나머지 model·cache·backend·deadline은 동일하게 둔다.

각 test의 expected는 selected order 하나가 아니다. key/eligibility 사건은 first candidate와 no unintended
victim, storm은 amplification bound와 post-burst recovery, starvation은 documented service bound, locality는
tenant minimum goodput, PrefillFirst는 maximum decode service gap을 가진다. 모든 test는 request/KV owner와
output parity를 공통 correctness gate로 통과해야 한다.

rollback rehearsal은 candidate policy를 끄는 API 동작만 시험하지 않는다. 새 requests admission을 멈추고
old policy generation으로 준비된 future/provisional batch가 끝나거나 폐기되는지 본다. preempted/retracted
requests의 queue key와 resume owner를 새 generation으로 다시 만들고, old comparator heap entry와 ratio,
debt/cooldown state를 invalidate한다. 두 generation request가 공존할 수 있다면 각 request가 선택된 generation을
끝까지 보존한다.

rollback 뒤 cache와 allocator를 무조건 flush하면 policy 문제는 숨길 수 있다. safe하다면 동일 warm state에서
baseline order와 curve가 복귀하는지 먼저 본다. representation이나 ownership이 incompatible하면 drain/restart를
선택하고 그 비용을 rollback contract에 쓴다. partial rollback으로 rank/worker마다 policy가 달라지면 admission을
열지 않는다.

배포 승인은 source evidence, invariant evidence, workload evidence의 세 묶음을 요구한다. source evidence는
pinned comparator, candidate processor, victim/release와 placement consumer를 가리킨다. invariant evidence는
budget 환불, 단일 queue/row owner, KV generation과 resume/output parity다. workload evidence는 tier goodput,
wait bound, amplification, locality saving과 recovery time이다. 셋 중 하나가 없으면 관측 또는 가설로 표시한다.

canary는 low-risk tenant만 고르는 것으로 끝나지 않는다. D/P/I/H 네 shape가 실제로 나타나는 traffic slice와
overload burst를 포함해야 rare victim path를 실행한다. shadow scheduling이 가능하면 선택 결과만 비교하되
실제 allocation/release side effect가 없다는 한계를 적는다. 작은 canary에서 preemption이 한 번도 없었다면
preemption safety를 승인하지 않는다.

운영 alert는 `priority_mismatch` 같은 추상 label보다 행동 가능한 predicate를 사용한다. eligible high-tier의
wait bound 초과, consecutive skip threshold, recompute amplification, same-victim storm, tier minimum goodput,
post-overload recovery timeout과 orphan/duplicate owner를 둔다. alert가 울리면 해당 incident fixture와 같은
열을 자동 수집해 first divergence를 재현한다.

마지막 decision record에는 선택하지 않은 대안도 적는다. strict priority를 골랐다면 quota/aging 부재와
low-tier admission cap을, locality를 골랐다면 cache-poor minimum service를, PrefillFirst를 골랐다면 decode
gap guardrail을, FIFO를 골랐다면 new-prefill starvation 조건을 쓴다. 선택의 한계를 알아야 workload 변화가
왔을 때 다시 감사할 trigger를 정할 수 있다.

이 장의 완료 문장은 특정 scheduler가 최고라는 선언이 아니다. D/P/I/H의 snapshot을 주면 독자가 native
source에서 candidate와 victim을 예측하고, 실제 trace가 다를 때 normalization·eligibility·selection·execution·
resume 가운데 첫 divergence를 찾으며, 이득과 debt를 같은 SLO 원장으로 승인할 수 있다는 것이다. 이 능력이
31장의 async future state와 32장의 네 scheduler 비교에 그대로 전달된다.

새 버전을 만났을 때도 시작점은 option 목록이 아니다. D/P/I/H fixture를 새 request object와 queue에 넣고
effective key를 손으로 계산한다. 그다음 eligibility와 budget predicate, victim release, resume와 output
generation을 source에서 잇는다. symbol이 이동했더라도 이 state transition이 유지되면 비교 좌표는 살아
있다. 반대로 함수 이름이 같아도 default policy, key 방향, fallback 또는 release ordering이 바뀌었다면
새 의미 계약으로 다시 검증한다.

운영 workload가 바뀌면 승인도 다시 연다. 긴 prompt 비율, tier arrival, prefix locality, output length와
KV pressure가 달라지면 과거의 fairness·debt trade-off는 더 이상 같은 문제를 풀지 않는다. decision
record에 재감사 trigger와 owner를 남겨 자동 튜닝이 조용히 SLO 우선순위를 바꾸지 못하게 한다.

최종적으로 선택, 되감기, 복귀의 세 문장을 모두 말할 수 있어야 한다. 누가 왜 선택됐는가. 누구의 어떤
state가 왜 해제됐는가. 피해자는 어느 generation으로 어떤 비용을 치르고 돌아왔는가. 세 번째 문장이 빠진
scheduler 설명은 빠른 요청만 보고 사라진 work와 굶은 요청을 놓친 미완성 설명이다.

## 30.9 하나의 arrival trace에서 strict priority·aging·fair queue를 비교한다

정책 이름을 따로 설명하면 독자는 결과를 비교하기 어렵다. 같은 arrival trace에 세 정책을 적용해 어느 요청이 언제 실행되고 어떤 work가 사라지는지 계산하자. GPU token budget은 iteration당 8, KV capacity는 24 token-equivalent라고 단순화한다. prefill token 하나는 budget 1, decode token 하나도 budget 1이지만 이미 resident한 KV를 계속 점유한다.

tenant Gold의 H는 시각 0에 도착한다. prompt 8, output 4, priority 0이다. tenant Bronze의 L은 시각 0에 도착하고 prompt 12, output 8, priority 10이다. 숫자가 작을수록 높은 우선순위라고 명시한다. 시각 1에는 Gold의 I가 prompt 4, output 2, priority 0으로 들어온다. 시각 2부터 5까지 Silver의 S0~S3가 매 시각 하나씩 prompt 4, output 2, priority 5로 들어온다.

iteration 0에서 H prompt 8을 prefill하면 budget을 모두 쓰고 KV 8을 점유한다. L은 waiting이다. iteration 1 시작에 I가 도착한다. H decode 1과 I prompt 4를 처리하면 budget 5, KV는 H 9+I 4=13이다. 남은 budget 3으로 L prompt 일부를 chunk할 수 있는지는 scheduler의 chunk/preemption contract에 달렸다. 비교를 단순하게 하려고 이 fixture에서는 prompt가 4-token chunk로만 admission된다고 한다. 3은 쓰지 못한다.

iteration 2에 S0가 도착한다. H decode 1, I decode 1, S0 prompt 4를 처리하면 budget 6이고 KV는 H 10+I 5+S0 4=19다. L은 계속 기다린다. iteration 3에 S1이 도착한다. H 마지막 전 decode와 I 마지막 decode, S1 prompt 4를 넣으려면 KV가 H 11+I 6+S0 4+S1 4=25로 capacity 24를 넘는다. 누군가를 finish/release하거나 admission을 미룬다.

I가 이 iteration token으로 완료되고 update 뒤 KV 6을 release할 수 있지만 allocation check가 launch 전에 이뤄지면 peak 25를 견디지 못한다. “끝날 예정”과 “이미 free”를 같은 것으로 세지 않는다. S1은 다음 iteration으로 미루거나 victim을 preempt해야 한다. 이 작은 차이가 priority inversion처럼 보이는 eligibility skip을 만든다.

**strict priority 결과**

strict priority는 eligible request 중 priority 0 H와 I를 먼저 진행하고, 다음 S tier를 L보다 계속 앞세운다. I가 iteration 3 전에 release된다고 가정하면 S1을 admit할 수 있다. 이후 S2,S3가 이어져 L은 Gold/Silver arrival이 멈출 때까지 기다린다. 이 trace에서는 유한하게 끝나지만 Silver arrival이 매 iteration 계속되면 L은 starvation할 수 있다.

H의 tail은 좋고 Gold SLO는 지켜진다. Bronze minimum service는 없다. priority 10 L이 이미 prompt 8 token을 계산한 뒤 higher-tier burst 때문에 victim이 되면 recompute loss가 생긴다. strict priority가 단순 comparator만의 문제가 아니라 victim selection과 resume 정책까지 포함하는 이유다.

**aging 결과**

effective priority를 `base_priority - floor(wait_iterations/2)`로 두자. 숫자가 작을수록 높으므로 오래 기다릴수록 2 iteration마다 1씩 좋아진다. L base 10이 Silver base 5를 이기려면 wait가 12 iteration 이상 필요하다. bound가 너무 길어 실제 Bronze SLO에는 쓸모없을 수 있다. aging이 있다는 사실보다 rate와 overload arrival이 중요하다.

aging rate를 iteration마다 2로 높이면 L은 wait 3에서 effective 4가 되어 새 Silver보다 앞선다. 그러나 L prompt chunk 4를 admit할 KV가 있어야 한다. score가 높아져도 cache eligibility가 없으면 계속 skipped된다. metric에는 comparator rank와 eligibility reason을 분리한다.

**weighted fair result**

tenant weight를 Gold:Silver:Bronze=4:2:1로 두고 token service debt를 추적하자. 7 token service window에서 목표 share는 4,2,1이다. active tenant만 분모에 넣는지 waiting까지 넣는지 정책을 정한다. iteration 2까지 Bronze가 0 service라면 debt가 쌓여 다음 available 4-token chunk에서 L을 우선할 수 있다.

하지만 prompt chunk가 4라 Bronze share 1 token보다 quantum이 크다. non-preemptible quantum 때문에 순간 share는 튄다. 긴 구간에서 debt를 갚는 방식으로 평가한다. tenant가 request를 여러 개 쪼개도 weight를 request별로 주면 분할 공격이 가능하므로 debt owner를 tenant로 둔다.

**한 표에서 completion·wait·waste를 비교한다**

정책별로 H/I completion iteration, L first service와 completion, S p95 wait, preemption count, recompute token, swap byte, KV occupancy를 적는다. strict priority가 Gold만 빠르고 L wait가 unbounded라면 그 trade-off를 숨기지 않는다. fair 정책이 L을 살리지만 H ITL을 늘리면 tier SLO와 minimum share 중 무엇을 택했는지 decision record에 쓴다.

이 trace는 실제 scheduler 성능 측정값이 아니라 의미 fixture다. 실제 vLLM·SGLang·Transformers·llama.cpp에 동일 native request shape를 만들고 각 정책의 실제 eligibility, chunk, cache accounting을 source와 runtime trace로 다시 확인한다.

## 30.10 recompute·swap·evict를 같은 비용 단위로 계산한다

preemption은 “요청을 잠시 멈춘다”로 끝나지 않는다. victim의 KV와 progress를 어떻게 처리하느냐에 따라 GPU compute, PCIe/NVLink byte, host memory와 resume latency가 달라진다. 세 선택을 saved capacity와 recovery cost로 정규화한다.

앞 장의 KV token당 128 KiB fixture를 사용한다. victim L이 prompt 8 token과 decode 2 token, 총 10 token KV를 가진다면 resident KV는 약 1.25 MiB다. 실제 model의 layer/head/dtype에 따라 값은 달라지므로 formula와 measured cache bytes를 함께 둔다.

**recompute 비용**

KV를 버리고 token history만 유지하면 1.25 MiB GPU capacity가 즉시 풀린다. resume 때 10 token을 다시 forward해야 한다. token당 평균 prefill compute가 0.4 ms라고 단순화하면 4 ms recovery compute다. 실제 chunk/batch에 따라 비선형이고 higher-priority batch와 함께 처리할 수 있지만 lost useful work 10 token은 명확하다.

같은 victim이 세 번 preempt되면 누적 recompute 30 token이다. 최종 output 8 token을 위한 useful model token과 비교해 amplification을 계산한다. prompt가 8인데 매번 거의 끝까지 재계산하면 goodput이 급락한다. preemption count만 보고 storm 강도를 알 수 없는 이유다.

**swap 비용**

KV 1.25 MiB를 host로 내리고 다시 올린다면 최소 logical transfer는 2.5 MiB다. effective transfer bandwidth 20 GiB/s라면 순수 byte 시간 하한은 약 0.122 ms지만 launch, synchronization, pageable/pinned 상태와 contention이 붙는다. host resident memory도 victim 수만큼 필요하다.

swap은 recompute compute를 줄이지만 PCIe와 copy engine을 점유해 active request의 H2D/D2H와 겹칠 수 있다. KV layout이 block/paged이면 valid blocks만 이동하는지 whole allocation을 이동하는지 확인한다. position과 cache metadata도 함께 보존해야 한다.

**evict와 prefix locality 비용**

여기서 evict는 reusable prefix cache entry를 제거해 당장 capacity를 얻는 선택으로 구분하자. active victim을 중단하지 않을 수 있지만 미래 request의 prefix hit를 잃는다. 900-token prefix를 evict하고 다음 10초에 같은 prefix 요청 세 개가 온다면 2,700 prompt token을 추가 계산한다. 미래 도착을 모르면 expected reuse probability로 비용을 추정한다.

entry byte 112.5 MiB, expected hits 0.5, token compute 0.4 ms라면 expected lost compute는 180 ms다. 반면 이 큰 entry를 evict해 여러 active requests를 살릴 수 있다. LRU, size-aware, cost-aware policy가 다른 이유다. 단순 oldest가 recovery cost를 보존하지 않는다.

**break-even 식**

recompute cost를 `R = tokens_lost × cost_per_token`, swap을 `S = 2×kv_bytes/bandwidth + sync_overhead`, eviction을 `E = expected_future_hits×prefix_tokens×cost_per_token + lookup/rebuild`로 둔다. 현재 high-priority wait penalty `W`와 함께 `min(R,S,E)`만 고르는 것이 아니라 capacity freed와 resource contention, correctness capability를 제약으로 둔다.

swap backend가 없거나 host memory가 부족하면 S는 후보가 아니다. context representation이 recompute와 동일 logits를 product tolerance 안에서 보장하지 않으면 correctness gate가 필요하다. prefix entry가 shared refcount를 가지면 active reader가 있는 동안 evict할 수 없다. 비용식 앞에 eligibility가 온다.

**preemption 사건: cheap victim이 가장 비싼 미래를 만들었다**

scheduler가 현재 KV byte가 가장 큰 prefix entry를 eviction victim으로 골라 112.5 MiB를 얻었다. 200ms 뒤 같은 system prompt를 가진 Gold 요청 20개가 burst로 왔다. 모두 prefix miss가 되어 TTFT가 급등했고 prefill이 decode를 밀어 priority inversion처럼 보였다.

observation은 eviction 직후 free KV가 증가했지만 Gold prompt computed token과 TTFT가 동시에 튄 것이다. branch는 active victim recompute가 아니라 prefix eviction이었다. cause는 future locality cost를 victim key에 넣지 않은 것이다. verification은 cache key별 recent reuse, evicted bytes, post-eviction miss compute를 arrival trace로 잇는다.

수정 후보는 protected high-value prefix, reuse-frequency/cost-aware score, tenant reservation이다. cache를 무한 보호하면 active capacity가 부족하므로 cap과 decay를 둔다. rollback은 새 victim policy generation을 끄고 기존 LRU로 돌아가되 in-flight provisional eviction decision을 drain한다.

## 30.11 priority inversion·starvation·tenant fairness를 한 사건으로 진단한다

Gold H가 높은 priority인데 낮은 Bronze L 때문에 늦었다는 보고를 받았다고 하자. 바로 comparator inversion으로 결론 내리지 않는다. L이 non-preemptible decode kernel을 이미 실행 중일 수 있고, H prompt에 필요한 KV block이 없어 eligibility에서 skipped됐을 수 있으며, prefix cache eviction으로 H prefill 자체가 길어졌을 수 있다.

### observation에서 branch를 나눈다

첫 관측은 H arrival, first eligible, selected, first launch, first token 시각이다. arrival→eligible gap은 validation/tokenization/cache allocation이다. eligible→selected gap은 scheduler order/fairness다. selected→launch gap은 batch/async resource다. launch→first token은 model/backend와 prompt work다.

L이 H arrival 뒤에도 selected됐다면 effective keys를 비교한다. H가 eligible=false면 comparator가 아니라 allocation predicate를 본다. H selected=true인데 L kernel 종료를 기다렸다면 non-preemptible quantum이다. H prompt computed token이 예상보다 많다면 locality eviction이다.

### 사건 timeline

t0에 Bronze L decode가 context 8k에서 large batch로 launch됐다. t0+0.2ms Gold H가 도착했다. H priority key는 더 높지만 current CUDA work는 취소할 수 없어 6ms 기다렸다. 이것은 execution inversion이다. t1에 H가 scheduler top이지만 KV capacity가 1 block 부족해 skipped되고 Silver S가 들어갔다. 이것은 eligibility inversion이다.

t2에 allocator가 prefix entry를 evict해 H를 admit했지만 H의 900-token shared prefix가 바로 그 entry였다. H prefill이 full recompute되어 TTFT가 360ms 늘었다. 이것은 locality-cost inversion이다. 사용자 증상 하나에 세 원인이 연속됐다.

### cause와 verification

cause를 “priority가 무시됐다”로 쓰지 않는다. non-preemptible batch quantum 6ms, eligibility block shortfall 1, victim policy가 H prefix의 tenant/reuse cost를 무시한 세 조건을 쓴다. trace에는 effective key, eligibility reason, active batch remaining, allocator/victim decision, prefix reused/evaluated token을 둔다.

verification은 H를 기다리는 동안 L이 선택된 이유가 source branch와 일치하는지 본다. comparator 결과가 맞으면 priority normalization을 배제한다. allocator free 뒤 H가 즉시 selected되는지 본다. prefix cache 보호 fixture에서 H evaluated token이 900 줄고 TTFT가 계산 방향으로 회복되는지 본다.

### 수정과 rollback

non-preemptible quantum은 batch token cap이나 chunking으로 줄일 수 있지만 throughput이 떨어질 수 있다. eligibility는 high-tier reserve block 또는 victim preemption으로 개선할 수 있지만 idle waste/fairness cost가 있다. locality는 tenant-aware protected prefix나 cost-aware victim으로 개선한다. 세 patch를 한 번에 넣지 않고 각각의 first divergence fixture로 검증한다.

rollback은 policy comparator만 원복하지 않는다. reservation, debt/aging, protected cache entries와 pending victims를 policy generation으로 묶는다. 새 admission을 잠시 막고 provisional decisions를 끝낸 뒤 이전 state representation으로 복귀한다. rank/worker마다 다른 generation이면 요청을 받지 않는다.

### starvation을 overload 뒤 recovery로 판정한다

유한 테스트에서 low-tier가 100 iteration 못 돌았다고 무조건 starvation은 아니다. high-tier offered load가 capacity를 계속 초과하면 strict priority는 의도상 low-tier 서비스를 보장하지 않을 수 있다. 제품이 minimum service를 약속하는지 먼저 정한다.

minimum 5% token share를 약속했다면 sliding window에서 Bronze service와 debt를 본다. burst가 끝난 뒤 bounded recovery time 안에 debt가 갚아지고 waiting request가 진행해야 한다. arrival을 멈춰도 L이 queue key/eligibility bug로 영원히 남으면 명백한 starvation defect다.

tenant 분할 공격 fixture도 둔다. Bronze가 request 100개로 쪼갔을 때 request-round-robin이 tenant share를 100배 주지 않아야 한다. tenant ID validation과 nested fair queue owner를 본다. anonymous/default tenant 처리와 cardinality도 정책에 포함한다.

## 30.12 source·metric·rollout을 하나의 scheduling dossier로 묶는다

source walk는 request priority normalization, waiting queue comparator, eligibility/budget predicate, victim selection, state release, resume/recompute/swap, prefix placement/eviction을 잇는다. vLLM은 priority scheduling과 preemption 함수, SGLang은 policy/retraction, Transformers는 FIFO/PrefillFirst candidate processing, llama.cpp는 slot placement/LCP·LRU를 native 의미로 읽는다.

서로 다른 이름을 억지로 같은 policy로 부르지 않는다. queue priority는 어떤 request가 service를 받는지 결정한다. retraction은 memory pressure에서 active victim을 되돌린다. PrefillFirst는 request phase class를 우선한다. LCP/LRU는 idle slot/cache locality placement다. 공통 비교 좌표는 arrival, eligibility, selection, allocation, execution, victim, resume와 completion이다.

### metric 계약

request/tier별 high-cardinality ID를 Prometheus label에 넣지 않는다. priority/tier cohort, waiting age bucket, selected tokens, eligibility skip reason, preemption mode, recompute token, swap byte, evicted prefix byte와 recovered hits를 bounded metric으로 둔다. incident request는 trace exemplar로 연결한다.

fairness는 평균 wait 하나가 아니라 tier goodput, minimum share violation, consecutive skip, overload recovery와 Jain index 같은 보조 지표를 본다. Jain index가 높아도 priority SLO를 위반할 수 있고 priority SLO가 좋아도 low-tier starvation이 생길 수 있다. 목적 함수를 명시한다.

### source incident record

한 행은 `(time,request_incarnation,tenant,tier,phase,prompt_remaining,decode_remaining,kv_bytes,prefix_value,effective_key,eligible,skip_reason,selected,victim,action,cost)`다. iteration snapshot에서 queue와 active, free capacity를 함께 보존한다. 이 표로 arrival trace를 재생해 source 예상과 actual selection을 비교한다.

first divergence가 key면 normalization/comparator, eligible면 allocator/budget, selected면 queue merge, victim이면 cost/debt policy, execution이면 non-preemptible backend, resume면 generation/cache state를 연다. 뒤 증상으로 앞 원인을 단정하지 않는다.

### rollout 승인 조건

correctness terminal은 preempted/resumed request의 output이 no-preemption reference와 허용 의미 안에서 맞고 cache/position generation이 보존된다. fairness terminal은 target overload trace에서 tier SLO와 minimum service/recovery bound를 만족한다. efficiency terminal은 recompute amplification, swap traffic, eviction future miss와 useful token goodput가 예산 안이다.

lifecycle terminal은 cancel/finish와 victim release가 겹쳐도 budget/refcount/queue owner가 정확히 한 번 이동한다. observability terminal은 arrival trace에서 decision을 재생하고 first divergence를 찾을 수 있다. rollback terminal은 policy generation을 drain하고 baseline selection·recovery curve가 복원된다.

### 최종 독자 질문

누가 선택됐는가만 답하면 절반이다. 왜 다른 request는 eligible하지 않았는가, 선택을 위해 누구의 어떤 state를 얼마나 버리거나 옮겼는가, victim이 언제 어떤 비용으로 복귀했는가를 답해야 한다. tenant share와 prefix locality가 그 선택에 어떤 미래 debt를 만들었는지도 계산한다.

30장의 최종 invariant는 다음과 같다. **동일 arrival·capacity snapshot에서 policy가 설명 가능한 key와 eligibility로 service와 victim을 선택하고, recompute·swap·evict의 현재·미래 비용과 tenant debt를 보존하며, preempted request는 generation-safe state로 유한하게 복귀하고 overload 뒤 약속된 fairness bound가 회복되어야 한다.**

이 문장을 하나의 H/L/I/S arrival trace, 세 비용식, 복합 inversion incident와 rollout terminal로 설명할 수 있으면 scheduler 정책을 옵션 취향이 아니라 검증 가능한 운영 계약으로 다룰 수 있다.

**고정 소스에서 arrival trace를 실제 branch에 대입한다**

vLLM의 priority request queue 고정 소스는 작은 priority 값이 먼저이고 동률이면 이른 arrival이 먼저라고 명시한다. queue의 prepend 의미도 priority 모드에서는 `(priority,arrival_time)` ordering으로 다시 해석된다. 따라서 preempted request를 “앞에 넣었다”는 호출만 보고 즉시 재실행된다고 생각하면 안 된다. effective key가 다른 waiting request와 다시 비교된다. [vLLM priority queue 계약](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/request_queue.py#L133-L195)

scheduler는 priority policy에서 running 중 가장 낮은 우선순위 victim을 `(priority,arrival_time)` max로 찾고, 이미 이번 step에 scheduled된 victim이면 token budget과 new block, speculative/encoder budget을 환불한다. 그 뒤 `_preempt_request`로 상태를 옮긴다. preemption은 running list 하나에서 빼는 일이 아니라 provisional transaction을 되감는 일이다. [vLLM priority victim과 budget 환불](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L571-L624)

`_preempt_request` source walk에서는 running 상태 assertion, KV/cache 관련 release, computed progress reset, preemption count와 stale output drop identity를 본다. exact field는 revision에 고정한다. arrival trace의 L이 victim이 됐을 때 `computed_tokens=10 → resume recompute frontier`가 어디서 바뀌고 waiting queue key가 어떻게 재생성되는지 적는다. [vLLM preemption state transition](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L1274-L1320)

SGLang의 schedule policy는 low priority value first 옵션에서 `priority_sign`을 만들고 priority와 received timestamp를 함께 정렬한다. preemption threshold 경로는 running request와 새 request의 priority difference를 계산해 threshold를 넘는지 본다.

같은 숫자 0/10을 vLLM과 그대로 비교하기 전에 sign 옵션과 threshold를 effective config에서 고정한다. 근거는 [SGLang priority·FCFS sorting](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_policy.py#L224-L280)과 [SGLang priority preemption threshold](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_policy.py#L1432-L1475)에서 확인한다.

이 source path를 arrival trace에 대입할 때 표의 각 선택에 file/function을 붙인다. key는 queue/sort, eligible은 token/cache budget, victim은 priority max 또는 retraction selector, release는 preempt/retract, resume는 waiting/retracted queue다. source line이 존재하는 것과 actual option branch가 선택됐다는 사실은 effective config와 runtime trace로 구분한다.

**열두 iteration을 정책별로 replay한다**

iteration 0~3은 앞서 계산한 H/I/S/L trace를 그대로 쓴다. iteration 4에 I가 release되고 free KV 6이 생긴다. strict priority는 S1과 H decode를 진행하고 L은 기다린다. iteration 5에 S2가 들어오며 S0 decode/finish와 S2 prefill이 경쟁한다. finish 예정 cache를 launch 전 capacity로 세지 않는 규칙을 적용한다.

각 iteration snapshot에 `free_before`, `release_after`, `token_budget`, `selected`, `skipped`, `victim`, `free_after_commit`을 둔다. “iteration 끝에 충분했으니 시작에도 eligible”이라는 역산을 금지한다. H high priority가 skip된 사건에서도 allocation check 시점의 free를 본다.

aging replay에서는 L effective key를 매 iteration 갱신하는지 queue insertion 때만 계산하는지 구분한다. heap key가 mutable wait time을 자동 반영하지 않으면 주기적 rebuild 또는 dequeue 계산이 필요하다. 설계 문서에 aging formula가 있어도 consumer가 stale key를 쓰면 실제 aging이 아니다.

fair replay에서는 tenant debt를 service commit 뒤 갱신한다. provisional scheduled token을 미리 service로 세었다가 victim 환불 때 debt를 되돌리지 않으면 preempted tenant가 실제 compute 없이 share를 소비한다. token budget refund와 fairness debt refund를 같은 transaction에서 검증한다.

iteration 8에서 high-tier arrival을 멈춘다. recovery clock을 시작하고 L이 언제 first service와 completion을 얻는지 본다. wait bound가 4 iteration이라면 iteration 12 전에 진행해야 한다. eligible=false가 계속되면 reserve/reclaim, eligible=true인데 selected=false면 key/debt를 본다.

**비용 모델을 victim selector에 넣을 때의 함정**

victim score를 `freed_bytes / recovery_cost`로 만들면 큰 capacity를 싸게 얻는 request를 선호할 수 있다. 그러나 recovery cost 0에 가까운 값과 shared prefix refcount, tenant minimum service를 처리해야 한다. score 하나로 모든 hard constraint를 대체하지 않는다.

예를 들어 A는 100 MiB를 free하고 recompute 1ms, B는 120 MiB를 free하고 swap 0.8ms지만 Gold tenant minimum share를 위반한다고 하자. scalar efficiency는 B를 고르지만 fairness guard가 B를 제외할 수 있다. eligible victim set을 먼저 만들고 비용 순위를 적용한다.

preemption cooldown은 same victim storm을 줄이지만 high-priority admission을 막을 수 있다. victim debt는 이미 잃은 work가 큰 request를 보호하지만 새 long request를 계속 희생시킬 수 있다. trace에서 consecutive victim count, lost token과 time since resume를 함께 본다.

prefix eviction score는 미래 hits 추정이 틀릴 수 있다. tenant burst 주기가 바뀌면 recent LFU가 stale하다. prediction error와 policy regret를 기록하고 보호 cap을 둔다. cached artifact identity가 달라 실제 hit 불가능한 entry를 비싸게 보호하지 않는다.

**fairness 사건을 사용자 영향으로 번역한다**

Gold H의 priority inversion은 TTFT 366ms 증가로 나타났다. Bronze L starvation은 12 iteration 동안 first service 없음으로 나타났다. Silver S는 fair policy에서 ITL이 2ms 늘었다. 운영 decision은 세 숫자를 함께 본다. 하나의 global p99는 tenant와 phase를 섞는다.

요금제 priority를 내부 숫자로 번역할 때 sign과 default tenant를 문서화한다. 요청이 priority를 지정할 수 있다면 허용 범위와 인증된 tier mapping을 검증한다. 사용자가 임의로 최고 priority를 넣어 fairness를 우회하지 않게 한다. priority missing/invalid가 silent 최고값이 되지 않는지 본다.

tenant ID가 없는 batch job을 default tenant 하나로 묶으면 서로 share를 나눈다. request별 fair queue로 처리하면 분할 공격이 가능하다. 조직·API key·project 중 어느 identity가 debt owner인지 privacy와 cardinality를 고려해 정한다.

starvation alert는 low-tier wait만 보지 않고 offered load와 promised minimum을 포함한다. high-tier overload 중 strict policy가 의도대로 low-tier를 막는다면 alert severity와 admission control을 다르게 한다. overload 종료 뒤 recovery bound 위반은 정책 defect로 높인다.

**observation→branch→cause→verification→rollback 한 장 카드**

observation은 “Gold H TTFT p99가 priority enabled 뒤 80ms에서 410ms로 증가했고 Bronze first-service p99도 12s를 넘었다”다. branch 1은 H key가 top인지, branch 2는 eligible인지, branch 3은 selected/launch인지, branch 4는 prefix hit인지다.

trace에서 H key는 top, eligible은 한 block 부족으로 false, capacity 확보 뒤 selected된다. victim selector는 112.5 MiB Gold prefix를 evict했고 H evaluated prompt가 900 늘었다. 동시에 running L의 6ms quantum을 기다렸다. cause는 comparator가 아니라 reserve 부족, locality-unaware eviction과 large non-preemptible batch다.

verification은 reserve 1 block canary에서 eligibility gap을 줄이고, protected prefix canary에서 reused token 900과 TTFT를 회복하며, batch quantum cap에서 execution gap을 줄이는 세 독립 실험이다. output/token correctness, useful goodput와 Silver/Bronze debt를 함께 본다.

rollback은 각 feature generation을 독립적으로 끈다. reserve를 끄면 reserved free blocks를 일반 pool로 반환하되 existing allocation을 깨지 않는다. prefix protection을 끄면 entry metadata를 baseline LRU key로 rebuild한다. batch cap을 끄면 future batches부터 적용하고 in-flight CUDA work는 drain한다.

세 patch를 하나로 배포했다가 하나의 global rollback을 쓰면 어느 효과와 부작용인지 모른다. canary cohort와 metric에 policy generation, reserve/protection/quantum effective value를 둔다. partial rank mismatch에서는 admission을 닫는다.

**선택의 대가를 복원하는 수치 기록**

capacity 열에는 free KV before, requested, reserved, freed by victim, deferred reclaim을 둔다. work 열에는 scheduled, committed useful, recomputed, swapped, evicted future miss token을 둔다. time 열에는 wait-to-eligible, eligible-to-selected, selected-to-launch, execution quantum, resume recovery를 둔다.

fairness 열에는 tenant weight, service, target, debt, consecutive skip과 minimum violation을 둔다. locality 열에는 prefix key class, resident byte, expected/actual reuse와 post-eviction miss를 둔다. lifecycle 열에는 request generation, victim count, stale output drop, cache/refcount release와 resume generation을 둔다.

이 원장에서 conservation이 맞아야 한다. provisional budget은 victim 때 환불되고 committed service만 debt를 줄인다. freed block은 device/resource completion 전 double allocation되지 않는다. preempted output은 stale generation으로 commit되지 않는다. swapped byte는 D2H/H2D 양쪽과 host resident를 설명한다.

**최종 source upgrade audit**

새 vLLM revision에서는 queue key 방향, running victim key, preempt reset field와 skipped queue merge를 diff한다. 새 SGLang에서는 priority sign/default, threshold, retraction stain/debt와 resume queue를 diff한다. Transformers는 FIFO/PrefillFirst candidate order와 allocator eligibility를, llama.cpp는 LCP/LRU slot placement와 cache compatibility를 diff한다.

함수 이름이 유지돼도 default와 branch predicate가 바뀌면 새 계약이다. 테스트 fixture H/L/I/S의 effective key와 first six selections를 golden decision trace로 보존한다. source-only 확인은 actual deployment branch를 대신하지 않으므로 effective config와 selected policy log를 붙인다.

CUDA와 KV backend가 바뀌면 non-preemptible quantum, swap bandwidth, recompute cost와 deferred release가 달라진다. 비용식의 상수를 재측정한다. 과거 break-even을 새 GPU에 그대로 복사하지 않는다. model architecture의 KV token byte도 다시 계산한다.

마지막 release 문장은 결과와 범위를 함께 적는다. “고정 workload와 pinned revision에서 Gold TTFT bound, Bronze minimum 5%와 overload 후 4-iteration recovery를 만족했고 recompute amplification 1.2 이하였으며, swap backend와 PP retraction은 미검증이다.” 이 정도로 써야 다음 운영자가 무엇을 믿고 어디를 다시 파야 하는지 안다.

**30분 policy failure drill**

첫 5분에는 priority sign을 뒤집는다. H=0, L=10에서 L이 먼저 선택되도록 잘못 설정하고 effective key와 selected order alert가 이를 잡는지 본다. request payload priority와 normalized key를 둘 다 기록한다. 숫자 자체만 보면 어느 방향이 높은지 알 수 없다.

다음 5분에는 H가 key top이지만 cache 한 block 부족하도록 만든다. trace는 `eligible=false,skip_reason=kv`를 보여야 한다. 이를 comparator mismatch로 분류하면 실패다. 한 block release 뒤 H가 다음 decision에서 즉시 candidate가 되는지 본다. release가 device completion 때문에 deferred라면 그 generation/event를 기록한다.

세 번째 5분에는 L이 prompt 10 token을 계산한 상태에서 세 번 victim이 되게 한다. recompute token은 30, final useful prompt는 10이므로 prefill amplification만 보면 4배다. cooldown/debt를 켠 뒤 same-victim count와 high-tier wait, alternative victim cost를 함께 본다. storm을 없애려고 H SLO를 무제한 희생하지 않는다.

네 번째 5분에는 112.5 MiB Gold prefix를 evict한 직후 동일 prefix burst를 넣는다. expected future hit와 actual three hits, recomputed 2,700 token을 기록한다. protected prefix 정책에서 free capacity가 부족해 다른 victim이 생기므로 total debt를 비교한다. hit rate 하나만 개선했다고 승인하지 않는다.

다섯 번째 5분에는 Bronze request를 100개로 분할한다. tenant fair queue가 request count가 아니라 tenant weight 1로 service를 제한하는지 본다. request-level round robin에서 share가 튀면 debt owner가 잘못됐다. default/unknown tenant가 우회 경로가 아닌지도 확인한다.

마지막 5분에는 high-tier arrival을 멈추고 recovery clock을 잰다. L이 promised 4 iteration 안에 first service를 얻고 debt가 bounded window에서 줄어야 한다. queue에 남아 key가 aging됐지만 stale heap으로 선택되지 않거나, eligible인데 skipped queue merge에서 유실되는 defect를 분리한다.

**policy 비교 표를 읽는 순서**

첫째 hard safety를 본다. duplicate queue owner, negative/double budget, stale output commit, KV double free가 있으면 fairness 점수와 무관하게 실패다. 둘째 tier SLO와 minimum service를 본다. 셋째 useful goodput와 amplification, swap/eviction debt를 본다. 넷째 locality와 energy/transfer 같은 부가 비용을 본다.

strict priority는 Gold tail을 최소화하는 대신 Bronze bound를 제공하지 않을 수 있다. aging은 유한 wait를 만들 수 있지만 rate가 arrival보다 약하면 실질 starvation이 남는다. weighted fair는 tenant share를 보장하지만 quantum과 long prompt가 순간 latency를 만든다. locality-aware는 saved compute를 늘리지만 cache-poor tenant를 굶길 수 있다. 한 승자를 선언하지 않고 workload 계약에 맞는 선택을 한다.

정책 비교는 같은 admission load와 model/cache/backend에서 한다. policy가 다르다는 이유로 batch token, max sequences, prefix cache 크기를 동시에 바꾸면 원인을 잃는다. 첫 experiment는 decision parity와 cost ledger, 다음 experiment는 tunable sweep으로 나눈다.

**resume correctness를 output까지 닫는다**

preempted request가 waiting queue로 돌아온 사실만으로 resume가 완료되지 않는다. recompute prefix가 원래 token history와 같고 cache position, sampler history, RNG generation과 output cursor가 보존되어야 한다. stale in-flight output은 drop되고 새 generation token만 commit된다.

fixture는 no-preemption reference와 forced-preemption path에 같은 prompt/seed/policy를 준다. deterministic 범위에서 selected IDs와 final text를 비교하고, numerical batch-shape 차이를 허용한다면 raw/processed distribution tolerance와 product contract를 명시한다. text만 우연히 같아도 duplicate/missing output cursor를 별도 검사한다.

swap resume는 D2H/H2D byte와 completion event, restored block/position mapping을 본다. recompute resume는 recomputed token과 first new logit을 본다. prefix eviction은 affected future request의 hit/miss와 first logits를 본다. 세 path를 “preempted” counter 하나에 합치지 않는다.

### 장애를 넘겨받은 사람이 끝까지 답해야 할 질문

독자는 H/L/I/S snapshot에서 native source comparator와 eligibility를 사용해 다음 selected set과 victim을 예측할 수 있어야 한다. prediction이 actual trace와 다르면 key, eligibility, selection, execution, release, resume 중 첫 divergence를 고른다. 각 단계의 owner 함수와 metric을 말할 수 있어야 한다.

또한 victim 비용을 token·byte·time으로 계산한다. recompute lost token, swap 왕복 byte와 host resident, eviction expected miss compute를 같은 표에 놓는다. fairness debt와 tier latency를 더해 왜 cheap-looking victim이 미래에 비싼지 설명한다.

마지막 질문은 “밀려난 요청이 정말 돌아왔는가”다. rollback도 이 질문에 답하는 상태 전이로 쓴다.
admission을 잠시 닫고, provisional batch와 victim을 drain하고, policy generation을 전환한다. 이어 queue
key와 debt를 다시 만들고 cache protection state를 정리한 뒤 baseline curve가 돌아왔는지 확인한다.
config flag 하나를 되돌렸다는 말만으로는 굶주림 회복을 증명할 수 없다.

여기까지 닫히면 scheduler는 throughput 숫자나 priority 옵션 표가 아니다. 도착 순서와 capacity, cache
snapshot을 받아 다음 실행자를 고르고, 공간이 부족하면 희생자를 고르며, 그 빚을 훗날 갚는
transaction이다. 좋은 정책은 중요한 요청을 앞세웠다는 말로 끝나지 않는다. 누구의 시간을 빌렸고,
얼마의 계산을 지웠으며, 그 요청을 언제 다시 움직이게 했는지까지 설명할 수 있어야 한다.

최종 incident report에는 성공하지 않은 수정도 남긴다. aging rate만 높였더니 L key는 올라갔지만 KV eligibility 부족으로 wait가 줄지 않았던 실험, prefix 보호만 켰더니 free capacity가 줄어 active victim recompute가 늘었던 실험, batch quantum만 줄였더니 Gold latency는 좋아졌지만 launch overhead로 total goodput이 떨어진 실험을 기록한다. 실패한 대안은 다음 운영자가 같은 시행착오를 반복하지 않게 한다.

각 실험은 하나의 arrival trace ID와 policy generation, source revision, model/KV byte, GPU/topology를 공유한다. decision log에서 selected/victim과 이유를 재생하고 runtime에서 실제 launch/commit/resume를 확인한다. source 예상과 관측을 같은 열에 덮어쓰지 않는다.

승인 뒤에도 drift alert를 둔다. tenant arrival mix, prompt/output length, prefix reuse, KV pressure와 backend quantum이 기준 범위를 벗어나면 과거 fairness 결론을 자동으로 재사용하지 않는다. 재감사 trigger는 Gold TTFT와 Bronze minimum share뿐 아니라 recompute amplification, swap bandwidth saturation, protected-prefix regret와 overload recovery를 포함한다.

새 정책이 machine-learned predictor를 사용하더라도 이 장의 원장은 사라지지 않는다. predictor score 앞의 eligibility와 뒤의 victim/release/resume safety는 결정론적 계약으로 남긴다. score feature와 model revision을 기록하고 shadow decision을 native baseline과 비교한다. 설명할 수 없는 score가 cache refcount나 request generation safety를 우회해서는 안 된다.

마지막 배포 확인에서는 H/L/I/S fixture를 한 번 더 실행한다. expected key, first six selection, victim, recompute/swap/evict 비용과 recovery bound가 승인 artifact와 일치해야 한다. 다르면 rollout을 멈추고 effective option, queue generation과 backend cost 상수부터 다시 고정한다.

운영 회고는 “priority를 켜서 해결했다”로 끝내지 않는다. 어떤 tier SLO와 minimum service를 선택했고 누구에게 어느 recompute·swap·eviction debt를 전가했는지 쓴다. workload 변화가 그 선택을 무효화하는 조건과 재감사 owner도 적는다.

이 기록이 있으면 다음 scheduler release에서 comparator 구현이 이동하거나 cache backend가 바뀌어도 같은 arrival trace로 의미를 재검증할 수 있다. 정책 이름이 같다는 사실보다 selection·victim·resume curve가 같은지가 중요하다.

그리고 같은 trace로 rollback 뒤 baseline 회복까지 반드시 확인한다.

미검증 backend와 workload 범위도 decision record에 남겨 다음 감사의 출발점으로 삼는다.

그 기록에는 책임 owner, 재감사 trigger, 필요한 fixture와 안전한 fallback을 함께 적는다.

반드시 재현 가능해야 한다.
