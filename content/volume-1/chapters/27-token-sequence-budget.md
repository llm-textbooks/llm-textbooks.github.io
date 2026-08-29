# 27장. 토큰 예산은 하나가 아니다: 네 제약의 교집합에서 admission을 읽기

서버에 여유가 있어 보이는데 새 요청이 들어가지 않는다. GPU utilization은 70%이고 이번 step의
scheduled token도 상한보다 작다. 운영자는 `max_num_batched_tokens`를 올리지만 TTFT는 그대로이고,
조금 뒤 OOM이 난다. 문제는 “batch budget”이라는 한 단어로 서로 다른 단위와 lifetime을 합친 데서
시작한다.

이 장은 예산을 네 축으로 분리한다. 첫째는 이번 forward가 계산할 query token 수다. 둘째는 scheduler,
runner와 output bookkeeping이 동시에 소유할 active sequence/request 수다. 셋째는 이미 계산한 K/V를
여러 step 동안 보존할 persistent capacity다. 넷째는 이번 shape가 잠시 요구하는 activation, FP32
logits, attention mask와 kernel workspace다. 요청은 네 predicate가 모두 참일 때만 admission된다.

```text
admit(R, step) = Q_step(R) ∧ S_active(R) ∧ K_persistent(R) ∧ W_transient(R)
```

이를 극장 비유로만 설명하면 좌석 수, 입장 게이트 처리량, 창고와 무대 공간이 다르다고 말할 수
있다. 그러나 비유는 token lifetime과 byte 계산을 설명하지 못한다. query token은 한 step 뒤
사라질 수 있지만 그 token이 만든 K/V는 request가 끝날 때까지 남는다. sequence slot은 token 수가
아니며 logits는 vocabulary에 비례한다. 이제 단위를 식으로 고정해야 한다.

고정 소스는 vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang
v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, Transformers v5.15.1 commit
`550d7b3834670483a4df436541272c055dc364bf`다. 실행 수치는 만들지 않는다. source가 증명하는 validation,
consumer와 state mutation을 설명하고 실제 선택·성능은 관측 설계로 남긴다.

## 27.1 한 요청이 네 문을 통과하는 장면

동일 GPU에 세 요청이 있다고 하자. A는 prompt 1,024 token을 모두 계산했고 decode 한 token을 기다린다.
B는 prompt 3,000 token 중 1,000개를 계산했고 2,000개가 남았다. C는 700-token 새 prompt다. step query
budget `Q_max=512`, active request 상한 `S_max=3`, persistent KV capacity `K_max=4,096 token-slots`라고
하자. transient memory는 query shape 512까지 허용한다고 우선 가정한다.

이번 step에서 A에 1 token, B에 511-token chunk를 배정하면 query budget은 정확히 찬다. 그러나
이미 persistent KV는 A 1,024+B 1,000=2,024 slot이고 B chunk가 commit되면 2,535다. C까지 active로
넣으면 request slot은 3으로 맞지만 C prompt chunk와 향후 decode reserve를 위한 KV가 남는지 따로
봐야 한다. query budget이 0이라 C가 못 들어오는 것과 KV가 부족해 못 들어오는 것은 결과만 같고
다음 조치가 다르다.

다음 step에서 B에 512를 더 주면 persistent 사용은 A의 decode 결과까지 포함해 약 3,048이 된다.
C 512를 같이 계산하려면 query 합이 1,025라 불가능하다. C를 256으로 줄이고 B를 255, A를 1로
배정하면 query 512다. persistent 증가는 512라 3,560이 된다. 아직 KV 536 slot이 남지만 B의 남은
prompt와 세 요청의 future decode를 모두 보장하지는 못한다. scheduler가 지금 가능한 한 step과
요청 전체 완료 가능성을 어떤 reserve 정책으로 연결하는지가 중요하다.

### 네 예산의 단위와 owner

| 예산 | 단위 | 대표 lifetime | 대표 owner/consumer |
|---|---|---|---|
| step query | tokens per forward | schedule transaction | scheduler→runner input |
| active sequence | requests/sequences | admission~finish | scheduler·runner slots |
| persistent KV | token slots 또는 bytes | computed token~free | cache allocator |
| transient | bytes by shape | kernel/step 또는 graph session | runner·memory handler |

query token 512와 active request 3을 더할 수 없다. KV token-slot 4,096도 model architecture, dtype,
layer 수와 TP partition을 모르면 byte가 아니다. transient 2 GiB는 logits인지 workspace인지에 따라
어느 옵션이 소비하는지 다르다. dashboard가 네 값을 모두 `batch_size`로 export하면 첫 false
predicate를 찾을 수 없다.

### 동일 timeline에서 첫 false predicate를 적는다

step마다 후보 request에 대해 네 행을 계산한다. A는 query 1, slot 추가 0, KV 새 slot 1,
transient marginal byte가 decode bucket에 포함된다. B는 query chunk `min(remaining, Q_left,
chunk_limit)`, slot 추가 0, KV 새 slot이 chunk에 비례한다. C는 query chunk뿐 아니라 slot 1개와
새 block rounding을 요구한다.

```text
step 17: A q=1, B q=255, C q=256
Q: 1+255+256 <= 512                         true
S: active 2 + C 1 <= 3                      true
K: used 3048 + rounded_new_slots 512 <=4096 true
W: footprint(shape q=512, seq=3, kvread=3560)<=limit ?
```

마지막 W가 false라면 token과 KV 숫자가 남아도 실행해서는 안 된다. 반대로 W가 true지만 K가 false면
query budget을 올려도 해결되지 않는다. 각 step decision에 predicate와 lhs/rhs를 남기는 것이 이
장의 핵심 관측이다.

## 27.2 step query-token budget은 schedule transaction 안에서 변한다

vLLM scheduler는 초기화에서 active request 상한과 step token 상한을 별도 field로 둔다.
[`Scheduler.__init__`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L109-L119)는
`max_num_running_reqs=max_num_seqs`를 저장하고, `max_num_scheduled_tokens`가 있으면 그것을,
아니면 `max_num_batched_tokens`를 step 상한으로 쓴다. 이름이 둘인 이유와 fallback을 startup
effective state에서 확인한다.

[`schedule`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L448-L638)은
매 step `token_budget=self.max_num_scheduled_tokens`로 시작하지만 pause state에서는 0으로 바꾼다.
config가 512라는 사실과 이번 step spendable budget이 512라는 사실은 다르다. structured
output 준비, encoder budget, PP cadence와 model-specific alignment도 request별 배정량을 0으로
만들 수 있다.

### 필요한 token과 실제 배정 token

vLLM의 request는 prompt/prefill/decode phase를 별 enum 하나로 고정하기보다 `num_computed_tokens`가
`num_tokens_with_spec`을 따라잡도록 새 token 수를 계산한다. 필요한 양은 개념상 다음이다.

```text
need = tokens_with_spec + output_placeholders - computed_tokens
grant = min(need, step_budget_left, long_prefill_limit, model_len_room, other_constraints)
```

이 식은 긴 prompt chunk, prefix hit와 speculative token을 같은 차이 회계로 표현한다. A decode는
need가 보통 작고 B prefill은 크다. `long_prefill_token_threshold`가 있으면 B grant가 step budget보다
먼저 잘린다. model max length는 sampled-token 여유를 뺀 room으로 또 자른다. encoder input이 있으면
별도 encoder compute/cache budget과 교차한다.

grant가 0이라고 `token_budget==0`인 것은 아니다. PP가 prompt 계산을 끝냈지만 결과 pipeline이 아직
돌아오지 않았거나, async max-token 경계, encoder budget 고갈, hybrid Mamba block alignment 때문에
현재 request가 0일 수 있다. loop가 다음 request를 계속 보는 이유다. “앞 요청이 막혔으니 전체
token budget이 찼다”는 결론은 source branch와 맞지 않는다.

### 차감과 환불이 schedule commit을 만든다

KV slot allocation이 성공한 뒤에야 request별 `num_scheduled_tokens`를 기록하고 budget에서 grant를
뺀다. allocation이 실패해 낮은 priority request를 preempt할 때, 그 request가 이번 step에 이미
배정돼 있었다면 map에서 제거하고 token budget을 되돌린다. encoder compute 배정도 함께 복구한다.
budget은 immutable limit가 아니라 schedule transaction의 잔액이다.

예를 들어 budget 512에서 A 1, B 400을 배정해 잔액 111이 되었다. C의 KV allocation을 위해 B가
preempt되고 B의 400이 환불되면 잔액은 511이다. C에 256을 주면 최종 scheduled 합은 A 1+C 256=257,
잔액 255다. 중간 profiler에서 401을 보고 최종 runner token이 257인 것을 accounting bug라 하면
안 된다. committed schedule output과 transaction 중 provisional allocation을 구분한다.

함수 끝의 assertion은 `sum(num_scheduled_tokens)<=max_num_scheduled_tokens`, `token_budget>=0`과
running 상한을 확인한다. 이 경계 뒤 runner가 소비하는 map이 확정된다. 관측은 config max, initial
effective budget, request별 need/cap/grant, refund reason과 committed sum을 분리한다.

### 옵션에서 지연 효과까지 닫기

`max_num_batched_tokens` 또는 scheduled-token 상한을 올리면 validation을 통과한 뒤 scheduler의
initial step balance가 커진다. consumer는 request별 grant 계산이다. mutation은 더 큰
`num_scheduled_tokens` map과 runner input token shape다. 물리 효과 후보는 prefill chunk 확대,
GEMM 효율 상승과 step 수 감소지만 activation/logits byte와 decode 대기 시간이 늘 수 있다.

반증은 간단하다. effective field가 그대로면 option parsing/override 문제다. field는 바뀌었지만
grant가 long-prefill threshold나 KV predicate에서 잘리면 token 상한이 bottleneck이 아니다. grant와
runner shape가 커졌는데 latency가 그대로면 kernel critical path, memory bandwidth, PP/TP communication과
output 단계가 후보다. 사용자 지표로 바로 점프하지 않는다.

## 27.3 active sequence budget은 token과 무관한 slot을 센다

query budget이 512이고 A decode가 1 token만 쓴다면 511이 남는다. 그렇다고 511개의 decode request를
추가할 수 있는 것은 아니다. scheduler와 runner는 request object, output row, block table row,
sampling state와 adapter metadata를 동시에 보존할 slot이 필요하다. active request 상한은 이
bookkeeping과 실행 shape를 제한하는 별도 predicate다.

vLLM waiting admission loop는 `len(running)`에 streaming input을 기다리지만 runner slot을 유지하는
수를 더해 `max_num_running_reqs`와 비교한다. token을 지금 계산하지 않는 request도 slot을 점유할
수 있다는 뜻이다. paused streaming session이나 pipeline in-flight request를 단순 `scheduled_tokens>0`
count로 세면 active capacity를 과대평가한다.

### request, sequence와 row는 항상 같은 수가 아니다

greedy text generation에서는 외부 request 하나가 sequence 하나로 보이기 쉽다. 그러나 beam/fork,
parallel samples, speculative placeholders와 multimodal encoder item이 들어오면 외부 request 수,
model runner sequence 수와 이번 logits row 수가 달라질 수 있다. 각 framework가 상한에서 무엇을
세는지 field consumer로 확인해야 한다.

이 장에서는 외부 request `R`, logical sequence `S`, 이번 query row `q`를 분리해 쓴다.

```text
R_active = scheduler가 lifetime을 소유한 외부 request 수
S_active = 독립 KV/history frontier 수
Q_step   = 이번 forward의 query token/row 수
L_rows   = sampling을 위해 남기는 logits row 수
```

단순 greedy decode에서는 네 수가 같아질 수 있지만 prefill에서 Q는 prompt chunk만큼 커지고 L은
sequence당 하나일 수 있다. prefix sharing은 K byte를 줄여도 S slot을 줄이지 않는다. chunked
prefill은 같은 R/S가 여러 step에 걸쳐 Q를 반복 소비한다.

### SGLang의 running 상한과 chunked request 예외

SGLang scheduler 초기화는 worker에서
[`max_total_num_tokens`, `max_prefill_tokens`, `max_running_requests`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L1031-L1124)를
별도 값으로 받는다. `max_running_requests`는 KV token capacity도 chunk size도 아니다. request-to-token
pool, batch metadata와 runner가 동시에 표현할 request 수의 상한이다.

prefill admission에서 allocatable request 수가 0이면 batch를 full로 표시할 수 있다. 그러나 이미
진행 중인 chunked request는 예외가 된다. PP에서는 한 chunked request가 microbatch 경계를 넘어
시작·종료할 수 있어 microbatch마다 기계적으로 request count를 적용하면 그 request의 lifetime을
닫지 못하고 memory leak을 만들 수 있다는 source 주석이 있다. 이는 “상한을 무시한다”는 일반 정책이
아니라 이미 소유권을 가진 in-progress request가 완료 경계에 도달하도록 하는 예외다.

[`PrefillAdder` 구성](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L3222-L3268)은
dynamic chunk size, current running batch size, `max_prefill_tokens`, `max_running_requests`와
`new_token_ratio`를 함께 전달한다. 같은 객체가 값을 받는다는 이유로 같은 예산이 되지는 않는다.
adder는 후보가 네 제약 교집합을 만족하는지 판단하는 consumer다.

### sequence 상한을 올렸을 때 실제로 바뀌어야 하는 것

`max_num_seqs`, `max_running_requests`, Transformers의 `max_requests_per_batch` 같은 option을 비교할
때 이름을 통일하지 않는다. 먼저 parser/config validation이 허용 범위를 정하고, scheduler/IO
constructor가 effective 상한을 저장하며, admission predicate가 running count와 비교한다. mutation은
active map과 batch row 수가 늘어나는 것이다.

물리 효과는 decode request가 많을 때 query batch가 커져 weight 재사용과 GPU occupancy가 좋아질
수 있다는 것이다. 하지만 request당 block table row, output bookkeeping과 sampling state가 늘고,
FP32 logits가 sequence별로 만들어지면 transient byte가 커진다. KV capacity가 고정이면 많은 긴
request가 cache를 더 빨리 고갈시킨다. sequence 상한만 올려도 실제 active 수가 늘지 않는다면
waiting candidate, KV allocation, token budget 또는 workload arrival이 bottleneck이다.

작은 계산을 해 보자. query 상한 512, request 상한 64에서 decode request 64개는 Q=64만 쓴다.
token budget utilization은 12.5%지만 request predicate는 full이다. 상한을 128로 올려 decode 128개가
되면 Q=128로 25%다. 아직 token 여유가 있어도 KV가 request당 평균 4,000 slot이면 persistent demand가
256k에서 512k slot으로 뛴다. capacity가 300k라면 실제 admission은 75개 근처에서 KV에 막힐 수 있다.
“max sequences를 두 배로 했는데 75개만 돈다”는 bug가 아니라 K predicate의 결과일 수 있다.

### active 관측과 반증

대시보드에는 waiting request, scheduler-owned running, 이번 scheduled request, runner slot resident,
in-flight/stream-wait request를 구분한다. `running=64` 한 gauge로는 실제 query를 받은 수와 slot만
유지한 수를 가를 수 없다. step trace에 각 count와 active-limit lhs/rhs, 새 admission이 거부된
predicate를 둔다.

request 상한이 false인데 query budget이 남는 장면을 재현하려면 GPU 실행이 필요하지 않다. source
상태 표에서 64개의 running entry와 각 need=1을 놓고 waiting C의 predicate를 계산할 수 있다.
실제 배포에서 확인할 때는 같은 prompt length distribution에서 상한 전후 running count, Q_step,
KV used, logits bytes와 ITL을 함께 본다. Q가 늘지 않으면 option 효과가 scheduler mutation까지
도달하지 않은 것이다.

## 27.4 persistent KV capacity는 과거를 보존하는 장기 예산이다

query token은 이번 forward의 work지만, 그 결과 K/V는 future decode가 읽도록 남는다. layer 수
`L`, local KV head 수 `N_kv`, head dimension `D`, element byte `b`인 단순 cache에서 token 하나의
rank-local byte는 다음과 같다.

```text
B_kv/token/rank = 2 × L × N_kv_local × D × b
```

2는 K와 V다. TP replication, MLA latent, hybrid layer, cache quantization과 alignment가 있으면 식을
architecture별로 바꾼다. logical token-slot과 physical cluster byte를 구분한다. GQA가 logical KV
head를 줄여도 TP가 작은 KV head를 여러 rank에 replicate하면 cluster 합계가 단순 `1/TP`가 아니다.

### block rounding과 future reserve

page/block size가 `P`이면 request 길이 `l`의 최소 physical slot은 흔히 `ceil(l/P)×P`다. A 1,024,
B 1,000, C 700에 P=16이면 각각 1,024, 1,008, 704 slot이다. 논리 합 2,724와 physical 합 2,736이
다르다. request 수가 많고 tail이 짧을수록 internal fragmentation이 커진다.

현재 prompt만 들어간다고 admission을 결정하면 decode 첫 token에서 block을 못 얻을 수 있다.
framework는 최소 lookahead, requested max new tokens 일부 또는 경험적 decode reserve를 고려할 수
있다. 보수적으로 전체 future를 예약하면 admission과 utilization이 낮아지고, 낙관적으로 현재
필요분만 잡으면 retraction/preemption 위험이 높아진다. 이것은 token step 상한과 다른 시간축의
trade-off다.

SGLang은 긴 request를 받을 때 paged input length와 max new tokens가 total token capacity 안에
영원히 들어올 수 있도록 clip한다. source의 조건은 input을 page 단위로 올리고 추가 page 여유와
model/request length를 고려한다. waiting queue에는 들어갔지만 어떤 schedule에서도 완료할 수 없는
request가 head를 막는 상황을 예방하려는 validation이다. `max_new_tokens` mutation은 사용자 요청을
그대로 실행하는 것이 아니라 capacity에 맞춘 effective contract가 된다.

### prefix hit는 K 예산을 절약하지만 Q·S를 없애지 않는다

C의 700-token prompt 중 640 token이 완전 block prefix hit라면 새 KV demand는 tail 60을 page로
올린 64 slot일 수 있다. 그러나 C request slot은 여전히 하나 필요하고, hit 검증·tail prefill query와
decode query는 step budget을 소비한다. cached token을 compute count에 반영해 Q need를 줄이지만
공유 block refcount와 block table row는 남는다.

prefix block은 C가 끝나도 다른 owner가 있으면 free되지 않는다. “request finish당 freed tokens”를
논리 sequence length로 계산하면 cache recovery를 과대평가한다. unique physical block, refcount와
complete 상태를 본다. hash hit 수가 높아도 tail fragmentation과 긴 decode가 capacity를 채울 수 있다.

### preemption과 retraction은 예산 환불의 대가를 가진다

vLLM에서 `allocate_slots`가 실패하면 scheduler는 request를 preempt하고 이번 step provisional token
grant를 환불할 수 있다. KV block이 free되어 새 request가 들어가지만 preempted request의 computed
work를 다시 해야 한다면 미래 query budget과 latency로 비용이 이동한다. free KV byte는 공짜가
아니다.

SGLang decode batch의 memory check가 실패하면 `retract_decode`가 request를 빼고 available token을
회복한다. 그 뒤 `new_token_ratio` tracker가 새 값으로 갱신된다. future decode reserve가 고정 상수가
아니라 recent pressure를 반영하는 상태라는 뜻이다. retraction 전후 old/new available tokens와
ratio mutation을 함께 기록해야 다음 admission이 왜 보수적으로 변했는지 설명된다.

### KV option의 소비 사슬

cache dtype을 BF16에서 8-bit 계열로 바꾸면 config validation이 backend/model 지원을 확인하고 cache
constructor가 element byte와 scale state를 정한다. allocator capacity가 늘 수 있지만 writer/reader의
quantize/dequantize, scale byte와 kernel capability가 소비자다. “byte가 절반”만 쓰면 정확도, traffic와
fallback을 놓친다.

block size는 allocator rounding과 block table 크기, kernel page layout을 바꾼다. 큰 block은 metadata를
줄일 수 있지만 tail waste를 키운다. 작은 block은 sharing granularity를 높이지만 table/index와
allocation work가 늘 수 있다. option→effective block size→allocated block count→page stride/kernel
argument→fragmentation·latency를 닫는다.

KV capacity option을 올렸는데 실제 num blocks가 그대로면 memory profiler 또는 memory handler가
available VRAM에서 잘랐을 수 있다. blocks가 늘었지만 active context 합이 늘지 않으면 request/token
budget이나 workload가 제한한다. cache utilization이 낮은데 allocation failure가 나면 group별
fragmentation, hybrid cache와 contiguous requirement를 본다. 총 free token 하나로 충분하지 않다.

## 27.5 transient capacity는 같은 token 수에서도 vocabulary와 mask에 따라 달라진다

persistent KV가 60%밖에 차지하지 않았는데 prefill OOM이 날 수 있다. 이번 forward의 activation,
LM-head logits, explicit attention mask와 workspace가 cache pool 밖 또는 같은 available memory를
잠시 요구하기 때문이다. query token 상한을 compute budget으로만 보면 이 peak를 놓친다.

Transformers continuous batching의
[`PagedAttentionMemoryHandler`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache.py#L556-L736)는
`M=max_batch_tokens`, `N=num_pages`를 같은 memory 식에서 푼다.

```text
memory(M,N) = c_m M + c_n N + c_mn MN + c_mm M²
```

`c_m M`에는 query 수에 비례하는 hidden, Q/K/V, bulk IO와 logits가 들어간다. `c_n N`에는 persistent
K/V와 cache-read state가 들어간다. explicit mask가 필요하면 current query와 cache length의 곱
`MN`, query끼리의 causal 영역 `M²`이 생길 수 있다. Flash 계열처럼 explicit mask를 만들지 않는
backend에서는 해당 coefficient가 0일 수 있다. 같은 M/N도 backend에 따라 footprint가 다르다.

### FP32 logits가 vocabulary를 예산으로 끌어온다

memory handler의 LM-head peak는 hidden `[M,H]`와 logits `[M,V]`를 센다. logits는 source에서 FP32로
계산하므로 대략 `4MV` byte 항이 있다. `M=512,V=150,000`이면 logits만 약 307,200,000 byte,
즉 decimal 307 MB다. vocabulary 32,000 model이면 약 65.5 MB다. 같은 token budget 512라도 model
vocabulary가 다르면 transient peak가 수배 차이 난다.

실제 runner가 sequence 마지막 row만 남기는 `logits_to_keep`을 지원하면 모든 prefill row의 logits를
materialize하지 않을 수 있다. 하지만 memory capacity solver가 어떤 worst-case를 가정하는지,
model signature와 selected path가 최적화를 쓰는지를 분리한다. profiler 계산이 보수적이라고 임의로
memory option을 넘기면 fallback path에서 OOM이 날 수 있다.

active sequence 상한도 logits와 만난다. decode request 256개는 Q=256이고 sequence마다 logits row가
필요해 `[256,V]`가 된다. 한 긴 prefill 256 query는 마지막 row 하나만 필요할 수 있어 같은 Q라도
LM-head rows가 다를 수 있다. framework의 static output buffer가 worst-case Q rows를 잡는지 sequence
rows를 잡는지 source tensor shape로 확인한다.

### attention peak와 persistent read 범위

attention peak의 M 항은 hidden, Q projection과 새 K/V를 포함한다. N 항은 기존 cache K/V read의
worst case를 반영한다. full attention에서는 긴 context가 N demand를 키우고, sliding attention은
logical total length가 커도 read window를 제한할 수 있다. 그러나 physical cache allocation과
attention read working set이 같은지는 allocator/backend에 달려 있다.

explicit mask `[1,1,M,N+M]`는 작은 dtype이어도 M과 N의 곱으로 커진다. M=512,N=131,072라면
원소 수가 약 67 million이다. bool 1 byte 가정만 해도 67 MB이고, 더 큰 dtype이면 늘어난다.
M을 1,024로 두 배 올리면 MN 항은 두 배, M² 항은 네 배가 된다. token budget option의 memory
효과가 선형이라고 단정할 수 없는 이유다.

kernel workspace도 static 식 밖에서 선택 backend, head dimension, split 수와 graph capture에 따라
필요할 수 있다. memory handler가 세는 항과 backend가 별도로 allocate하는 workspace를 inventory로
대조한다. CUDA allocator reserved/allocated 차이, graph memory pool과 library workspace가 available
memory 계산 시점에 이미 반영되는지도 본다.

### 두 미지수를 같이 푸는 이유

Transformers handler는 M과 num blocks를 둘 다 주면 footprint를 검증한다. 한쪽만 주면 polynomial을
다른 변수에 대해 풀고, 둘 다 없으면 batch가 cache의 일정 비율을 채운다는 관계로 VRAM upper
bound를 구한 뒤 기본 M 8192와 최소 256 경계를 적용하고 N을 다시 푼다. 문서 default 8192는 실제
effective M 보장이 아니다.

예를 들어 available memory가 고정일 때 M을 크게 지정하면 `c_mM+c_mmM²`가 먼저 공간을 먹고 N
solution이 줄어든다. 더 큰 prefill chunk를 얻는 대신 persistent context capacity가 줄 수 있다.
반대로 num blocks를 크게 고정하면 N과 MN 항이 공간을 차지해 허용 M이 줄어든다. 네 예산 가운데
Q와 K가 물리 VRAM W를 통해 결합되는 구체적 사례다.

TP에서는 각 rank가 계산한 M/blocks 중 최소를 all-reduce로 맞춘다.
[`PagedAttentionCache 초기화`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache.py#L200-L240)는
KV head가 TP partition되는 조건을 반영하고 두 capacity를 rank 최소로 정렬한다. 한 rank에 다른
reserved memory가 있으면 cluster effective capacity가 그 rank에 묶일 수 있다. rank 0 memory만
보고 option을 계산하지 않는다.

### async와 graph는 transient를 session lifetime으로 끌어올린다

async batching을 쓰면 memory handler의 IO multiplier가 2가 된다. batch N compute/D2H와 batch N+1
H2D/packing을 겹치기 위해 host/device static IO pair를 두 벌 갖기 때문이다. 요청 한 개가 끝나도
이 buffer는 manager session 동안 남는다. 엄밀히 transient activation과 session-persistent static
workspace를 관측에서 나눠야 하지만 둘 다 request KV가 아닌 W budget을 소비한다.

CUDA Graph는 captured shape의 주소와 memory pool을 보존한다. 여러 bucket을 warmup하면 graph key별
reserved bytes가 늘 수 있다. runtime logical M이 작아도 큰 captured buffer가 살아 있으면 available
memory가 줄어든다. “KV utilization 50%인데 block을 더 못 만든다”는 장면에서 graph/static pool을
빼면 계산이 맞지 않는다.

### transient option의 closed loop

`max_batch_tokens`를 512에서 1,024로 바꾼다고 하자. validation은 memory footprint가 available
memory 이하인지 확인한다. consumer는 static IO allocation, scheduler max query와 graph bucket이다.
mutation은 M capacity, tensor shapes와 가능한 num blocks solution이다. 물리 효과는 큰 prefill
forward, logits/mask/activation byte와 graph capture다. 지연은 prefill step 수 감소 가능성과 decode
interference·OOM 위험의 교환이다.

관측에서 effective M이 1,024지만 runner Q가 계속 256이면 scheduler chunk limit나 workload가
bottleneck이다. runner Q는 1,024인데 blocks가 줄었다면 solver trade-off다. 둘 다 커졌는데 startup
OOM이면 handler 밖 weight/graph/workspace 또는 memory snapshot timing을 본다. mask-free backend로
바꿨는데 `c_mn,c_mm`과 allocated mask가 그대로면 backend selection/fallback을 확인한다.

## 27.6 같은 요청 timeline을 세 구현의 predicate로 번역한다

세 framework의 option 이름을 한 표에 놓는 것만으로 비교가 되지 않는다. 같은 A/B/C request와
동일한 논리 capacity를 주고 각 scheduler가 계산하는 state를 native 이름으로 남긴다. 공통 좌표는
이번 query work, active ownership, persistent KV demand와 transient execution shape다.

### step 0: 빈 서버에 A의 긴 prompt가 들어온다

A prompt는 1,024 token, max new tokens 128, page size 16이라고 하자. query 상한은 512이므로 prompt는
최소 두 forward로 나뉜다. active 상한은 3, KV capacity는 4,096 slot이다. transient handler는
M=512 shape를 허용한다.

vLLM은 A의 `num_computed_tokens=0`, `num_tokens_with_spec≈1024`에서 need를 구하고 grant를 512로
자른다. prefix hit가 없고 KV allocation이 성공하면 `num_scheduled_tokens[A]=512`, balance 0이 된다.
이번 step 뒤 computed frontier와 cache slot이 512로 전진한다.

SGLang은 waiting A를 prefill candidate로 보고 `chunked_prefill_size`와 max prefill tokens에 따라 chunk를
정한다. chunk size가 512라면 같은 물리 work지만 state 이름은 prefix indices, extend length와
PrefillAdder budget이다. `new_token_ratio`가 future decode reserve를 반영해 admission 가능성을 바꿀
수 있다. vLLM의 exact field와 같은 것이라고 부르지 않는다.

Transformers CB는 `max_batch_tokens=512`에서 A의 remaining prefill을 batch tensor에 최대 512 query로
pack한다. scheduler가 blocks를 할당하고 memory handler가 이미 M/N 조합을 startup에서 검증했다.
runtime allocation failure는 static capacity validation과 별도로 block occupancy·sharing에 의해 날
수 있다.

세 경로 모두 “512 token prefill”이지만 첫 false predicate가 다를 수 있다. vLLM은 encoder/structured
state가 0 grant를 만들 수 있고, SGLang은 request reserve ratio 또는 chunk policy가 줄이며,
Transformers는 concrete cache pages와 static IO capacity가 교차한다. 비교 결과에는 native reason을
보존한다.

### step 1: A decode와 B의 긴 prefill이 섞인다

A prompt가 모두 계산되어 decode 1 token을 기다리고, B prompt 2,000 token이 들어왔다고 하자.
query 512를 A=1,B=511로 쓸 수 있다. 이 배정은 arithmetic상 Q predicate를 만족한다. 하지만
persistent current는 A 1,024이고 B 신규 511을 page allocation해야 한다. A의 output token을 위한
lookahead도 필요할 수 있다.

vLLM running-first loop가 A와 진행 중 B를 어떤 order로 훑는지는 policy/state에 달렸지만 각 grant의
합은 commit assertion 아래 512다. B가 waiting new request라면 running A를 먼저 schedule하고 남은
budget에서 prefix lookup, KV allocation과 active slot을 검사한다. A와 B가 모두 running이면 둘 다
same need-minus-computed 회계다.

SGLang mixed chunk가 켜지면 decode running batch 크기를 PrefillAdder에 전달하고 B prefill chunk와
A decode를 한 batch에 섞을 수 있다. 꺼져 있으면 prefill/decode 구성 방식이 다를 수 있다. option이
같은 512라도 mixed-mode predicate와 dynamic chunk predictor가 실제 query shape를 정한다.

Transformers docs의 continuous architecture는 budget보다 긴 prompt가 여러 step에 나뉘고 decode와
교차할 수 있음을 설명한다. IO packer는 A query 1과 B query 511을 flat input으로 만들며 cumulative
length와 logits indices를 보존한다. static M=512는 둘을 수용하지만 active request count 2와 KV N
사용이 별도로 맞아야 한다.

### step 2: C는 token이 남는데도 들어가지 못한다

A/B가 active인 상태에서 C prompt 700을 넣는다. query 잔액이 200인 step이 있어도 active 상한 2라면
C는 waiting이다. 이때 first false는 S다. token limit을 512→1024로 올려도 C는 들어가지 않는다.
active 상한을 3으로 올리면 다음 predicate K를 검사한다.

현재 A KV 1,025, B KV 2,000이라 합 3,025라고 하자. C 첫 512 chunk를 page rounding과 함께 넣으면
3,537이며 K=4,096 아래다. 하지만 세 request의 다음 decode reserve를 20%로 잡아 600 slot을 더
보수적으로 요구하면 4,137이라 false다. SGLang의 `new_token_ratio`, framework별 lookahead/reserve가
이런 차이를 구현한다. reserve를 빼면 지금 step은 가능하지만 다음 decode에 retract할 수 있다.

K도 true라고 가정해도 W가 남는다. C admission으로 active sequence가 3이 되어 logits row/table가
늘고 query M이 512 bucket을 채운다. explicit mask backend에서 현재 N≈3,537이면 MN 항이 커진다.
memory footprint가 limit을 넘으면 C는 W 때문에 막히거나 M을 줄여야 한다. free KV slot만 보고
admit하면 forward OOM이 난다.

### step 3: prefix hit가 predicate 일부만 바꾼다

C prompt 700 중 A와 같은 640-token prefix가 hit했다고 하자. C의 new computed need가 줄고 공유
blocks로 persistent 추가 demand가 크게 줄어든다. Q와 K가 개선된다. 그러나 C active slot은 그대로
하나이고 output/logits state도 필요하다. S가 full이면 hit가 있어도 admission되지 않는다.

W도 자동으로 사라지지 않는다. attention은 C tail query가 shared prefix KV를 읽고 logits를 만든다.
explicit mask/read metadata는 cached context를 포함할 수 있다. prefix hit rate가 높아 TTFT가 줄었지만
active limit이나 logits memory가 새 bottleneck이 되는 이유다.

### step 4: 한 request를 retract하면 어느 예산이 돌아오는가

B를 retract/preempt하면 B의 active slot과 private KV block이 풀릴 수 있다. 이번 step에 provisional
query grant가 있었다면 Q balance도 환불된다. 그러나 B의 이미 수행한 compute는 미래 재계산 work로
돌아오며 prefix-shared block은 refcount 때문에 안 풀릴 수 있다. graph/static workspace W는 request
finish와 무관하게 session에 남는다.

“preemption freed 2,000 tokens”라는 metric 하나는 부정확하다. 반환된 scheduler slot,
physical unique KV slots, current-step query refund와 future recompute tokens를 따로 센다. 사용자는
현재 admission을 얻는 대신 이후 TTFT/ITL debt를 진다.

### 동일 비교표를 채우는 방법

| 사건 | vLLM native state | SGLang native state | Transformers native state |
|---|---|---|---|
| Q 상한 | max scheduled/batched tokens, balance | max prefill/chunk, adder budget | max batch tokens, prepared Q |
| S 상한 | max running reqs | max running requests | max requests per batch |
| K capacity | KV manager blocks/slots | token-to-KV allocator,total tokens | num blocks/pages, allocators |
| W capacity | runner/profile/workspace shape | runner/backend workspace, batch tensors | M/N polynomial, static IO/graphs |

표는 이름 대응표가 아니라 source 탐색 시작점이다. 각 cell에서 config field, validation, consumer,
mutation과 observation을 잇는다. 동일 요청에서 어느 predicate가 처음 false였는지 비교하면 구현
정책 차이를 설명할 수 있지만 field를 서로 alias로 만들지는 않는다.

### llama.cpp를 볼 때 빠진 control plane을 억지로 만들지 않는다

llama.cpp server/slot 경로를 비교한다면 sequence slot, context/KV allocation과 batch token capacity를
찾을 수 있다. 그러나 이 세 framework와 같은 scheduler object나 paged-block policy가 반드시 있는
것은 아니다. ggml batch의 token/sequence fields, server slot state, context memory와 graph workspace가
네 공통 물리 질문에 어떻게 답하는지만 본다.

예를 들어 fixed context 안에서 sequence별 KV cell을 관리한다면 K capacity는 존재하지만 vLLM의
block allocator refcount와 동일하지 않다. batch `n_tokens` 상한은 Q와 가까워도 decoding slot 수와
같지 않다. option 이름을 번역하기보다 “이번 decode token을 batch에 넣는 predicate, sequence owner,
KV cell 확보와 graph allocation”을 source에서 찾는다. 필요한 경우에만 보조 비교로 쓰는 이유다.

## 27.7 옵션 하나를 바꿀 때 네 예산의 연쇄 변화를 닫는다

튜닝이 실패하는 가장 흔한 이유는 option 이름에서 사용자 latency로 바로 점프하기 때문이다.
이 장에서는 모든 option을 다섯 경계로 읽는다.

```text
user option
→ validation/default resolution
→ effective state
→ scheduler/allocator/runner consumer
→ mutation된 Q·S·K·W와 latency/resource 결과
```

어느 화살표가 관측되지 않으면 뒤 효과를 주장하지 않는다. 같은 이름도 version과 framework마다
다른 consumer를 가질 수 있으므로 pinned source의 함수와 field를 함께 기록한다.

**query-token 상한을 올리는 경우**

문제 장면은 긴 prefill이 256-token chunk로 16번 실행되고 TTFT가 길다는 것이다. `Q_max`를 512로
올리고 싶다. 먼저 model max length, long-prefill limit, static IO capacity와 memory validation이
512를 허용하는지 본다. vLLM에서는 resolved scheduled-token field, Transformers에서는 M/N solver,
SGLang에서는 chunked-prefill와 max-prefill state가 consumer로 이어져야 한다.

mutation 기대값은 request별 grant와 total query shape 증가다. B remaining 2,000에서 기존 grant 256,
새 grant 512가 되어야 한다. prefix hit로 remaining이 100이면 실제 grant 100이므로 option 효과가
없는 것이 정상이다. mixed decode 64개가 먼저 64를 쓰면 prefill은 최대 448이다. effective 상한과
single-request chunk는 같은 값이 아니다.

resource 효과는 activation과 new KV write가 대체로 Q에 비례하고 mask의 M² 항은 더 빠르게 늘 수
있다는 것이다. prefill step 수는 줄지만 한 step duration이 늘고 decode request가 같은 batch 또는
queue에서 기다리는 시간이 길어질 수 있다. TTFT 개선과 ITL tail 악화를 함께 본다. persistent KV
최종량은 같은 prompt라면 유사하지만 더 빠른 시간에 block을 소비해 burst pressure가 달라진다.

반증은 runner input의 logical Q가 바뀌지 않는 것이다. config log만 512면 parser→consumer가 끊겼다.
grant가 256이면 long-prefill/chunk cap이다. grant 512인데 padded execution 512가 이미 기존에도 같았다면
logical work만 변한 것인지 graph bucket 비용이 원래 지불되었는지 본다. kernel time은 늘고 end-to-end
TTFT가 같으면 tokenizer/queue/output 또는 PP bubble이 지배할 수 있다.

**active request 상한을 올리는 경우**

문제 장면은 decode token budget이 많이 남지만 waiting이 늘어나는 것이다. 기존 S=32에서 active
32개가 각각 Q=1을 쓰고 Q_max=256이면 224 token이 비어 있다. S를 64로 올리면 이론상 decode Q가
64가 되어 GEMM batch가 커질 수 있다.

validation은 runner/request pool과 graph/static row capacity가 64를 지원하는지 확인해야 한다.
consumer는 waiting admission의 running-count predicate다. mutation은 active map, block table row,
sampling/output state와 scheduled decode 수다. K demand는 새 32개 context 길이 합만큼 늘고 W는
logits `[64,V]`, metadata와 graph bucket이 커진다.

평균 context 8,000, token당 rank-local KV 128 KiB라면 새 32 request의 cache만 약 32 GiB에 해당한다.
실제 GPU에서는 불가능할 수 있어 K가 먼저 false가 된다. 숫자는 model별 계산으로 바꾸지만 active
상한이 memory와 독립인 knob가 아니라는 직관은 유지된다. prefix sharing이 크면 unique K byte는
줄어도 block table/output slot은 그대로다.

상한 뒤 active가 40에서 멈추면 K 또는 W lhs/rhs를 본다. KV free가 충분해도 logits/graph bucket이
40에서 제한될 수 있다. active가 64로 늘었지만 ITL이 나빠지면 request당 service interval과 step
duration을 분해한다. total throughput 증가가 모든 request의 latency 개선을 뜻하지 않는다.

**KV block 수 또는 memory fraction을 올리는 경우**

문제 장면은 query와 active slot이 남지만 `allocate_slots`/cache check가 반복 실패하는 것이다. KV
capacity option을 올리기 전에 available memory가 weight, allocator reserve, graph와 transient peak를
뺀 값인지 본다. Transformers처럼 M/N을 함께 푸는 구현에서는 N을 올리면 M solution이 줄 수 있다.
vLLM/SGLang도 startup memory profile이 다른 pool/workspace와 capacity를 나누는 source를 확인한다.

mutation 기대값은 physical block/token allocator total과 free count 증가다. 단순 config value가
아니다. TP에서는 rank별 최소와 KV head partition/replication이 effective cluster capacity를 정한다.
block size가 같아도 cache dtype과 layer grouping이 바뀌면 block byte가 달라진다.

resource 결과는 더 긴/많은 context를 resident로 두고 preemption을 줄일 가능성이다. 대가는 transient
headroom 감소, startup reserve 증가와 allocator fragmentation이다. CUDA OOM이 cache allocation이
아니라 logits/attention workspace에서 나면 KV fraction을 늘리는 것이 악화시킬 수 있다.

blocks가 늘고 preemption이 줄었는지 확인한다. preemption이 그대로인데 free blocks가 많다면 policy,
group-specific scarcity나 lookahead contiguous constraint가 후보이다. TTFT가 그대로면 waiting이 K가
아니라 S/Q에 막혔을 수 있다. ITL만 좋아지면 recompute/retraction 감소 효과일 수 있다.

**block/page size를 바꾸는 경우**

P=16에서 평균 tail waste는 균등 길이를 단순 가정하면 약 7.5 slot/request다. P=64면 약 31.5다.
active 1,000 request라면 차이는 24,000 slot이다. 하지만 P를 줄이면 block table entry, hash/refcount,
allocation operation과 kernel indexing이 늘 수 있다. 평균 식은 workload length distribution과 prefix
boundary를 반영하지 않으므로 실제 trace histogram으로 다시 계산한다.

validation은 backend가 page size를 지원하는지, quant pack/alignment와 max blocks per request가 맞는지
본다. consumer는 cache tensor shape, allocator rounding, block table과 attention kernel이다. mutation은
num blocks, per-request block list와 page strides다. P만 바꾸고 physical total byte를 고정하면 block
count가 역비례해 metadata가 변한다.

효과를 “작을수록 sharing이 좋다”로 닫지 않는다. prefix가 128-token 경계에 몰리면 P=16과 32가
같은 hit를 줄 수 있다. decode tail이 긴 request에서는 마지막 block waste 비중이 작다. 짧은 요청이
많으면 달라진다. cache hit, unique block, tail waste, table byte와 attention latency를 함께 비교한다.

**cache dtype을 바꾸는 경우**

BF16에서 FP8 계열 cache로 바꾸면 K byte가 줄 가능성이 있지만 Q step token과 S slot이 자동 증가하지
않는다. capacity profiler가 freed byte로 blocks를 더 만들 때만 K rhs가 늘어난다. fixed blocks라면
unused headroom만 늘 수 있다.

validation은 device architecture, backend kernel, scale mode와 head dimension 지원을 확인한다.
consumer는 K/V writer, persistent tensor와 reader/dequant kernel이다. mutation은 dtype, scale state,
block byte와 selected attention backend다. fallback이 BF16 cache를 만들면 option 문자열만 바뀐 것이다.

성능은 cache traffic 감소와 quant/dequant 비용의 교환이다. 긴-context decode에서 bandwidth 이점이
있을 수 있지만 짧은 context나 compute-bound model은 작다. correctness는 head별 cache readback,
logits parity와 quality metric으로 검증한다. capacity 증가를 품질 허용 없이 성과로 보고하지 않는다.

**chunked prefill과 mixed mode를 바꾸는 경우**

chunk size는 Q 상한의 별명 아니다. Q_max=2,048이어도 per-request chunk=512면 긴 B는 한 step에 512만
받고 나머지는 decode/다른 prefill에 쓸 수 있다. chunk를 1,024로 올리면 B TTFT step 수가 줄지만
decode interference와 transient peak가 커진다.

SGLang `init_chunked_prefill`은 nonpositive 값을 disabled로 정규화하고 특정 multimodal Transformers
backend 조합에서 partial mismatch를 피하려 비활성화한다. effective state를 확인하지 않고 CLI 값만
보면 실험이 다른 mode를 비교한다. PP dynamic chunking은 predictor가 runtime chunk를 바꿀 수 있으므로
requested size와 actual batch grant를 분리한다.

mixed mode를 켜면 running decode size가 prefill adder 예산에 들어가고 한 batch에 둘을 조합할 수 있다.
mutation은 batch composition, cumulative lengths와 attention path다. 지연 효과는 decode ITL 보호 또는
prefill progress 변화이며 backend tile shape도 달라질 수 있다. prefill/decode count와 Q token만이
아니라 kernel selected path와 padded work를 기록한다.

**max model/request length를 바꾸는 경우**

length 상한은 현재 step budget이 아니라 request feasibility validation이다. model context를 넘는
position, cache capacity로 절대 완료할 수 없는 max-new-token 조합을 admission 전에 거부하거나 clip한다.
이를 Q knob처럼 올리면 artifact position semantics와 KV K demand가 함께 변한다.

SGLang source처럼 paged input, max request length, total token capacity와 extra page 여유를 교차하면
effective max new tokens가 사용자 값보다 줄 수 있다. mutation된 generation contract를 response/metric에
보존해야 사용자가 조기 종료를 scheduler bug로 오해하지 않는다. min-new-token과 clip 결과의 invariant도
다시 맞춘다.

길이를 늘려 validation만 통과했지만 KV pool이 작으면 request는 waiting/retract를 반복할 수 있다.
“accepted”와 “완료 가능 reserve”가 같은지 본다. 한 요청이 전체 K capacity보다 큰 경우는 어떤
schedule policy도 해결하지 못하므로 빠른 reject가 필요하다.

## 27.8 첫 false predicate를 관측하고 반증하는 법

좋은 예산 대시보드는 utilization 네 개를 나열하는 데 그치지 않는다. waiting request C가 이번
step에 들어가지 못한 이유를 decision record로 남긴다. config maximum, step 시작 balance,
후보의 marginal demand, allocation 결과와 commit 뒤 사용량이 연결되어야 한다.

### decision record의 최소 형태

```text
step=218 candidate=C state=WAITING
Q lhs=used_257 + demand_255 <= limit_512              true
S lhs=active_3 + demand_1 <= limit_3                  false
K lhs=used_3536 + rounded_256 + reserve_32 <=4096     true
W lhs=estimated_peak_6.8GiB <= headroom_7.1GiB        true
decision=defer first_false=S
```

모든 후보마다 거대한 tensor dump를 남길 필요는 없다. lhs/rhs, reason enum, request length bucket과
step epoch면 bottleneck 분포를 만들 수 있다. tenant prompt나 token content는 필요하지 않다. K에는
logical token뿐 아니라 rounded physical demand와 reserve를, W에는 estimator identity와 selected
backend/shape를 포함한다.

predicate 평가 순서 때문에 첫 false가 실제 유일한 false는 아닐 수 있다. 구현이 S에서 바로 break하면
K/W를 계산하지 않았을 수 있다. metric은 `not_evaluated`와 true를 구분해야 한다. 진단용 shadow
calculation을 추가한다면 production decision을 바꾸지 않고 비용이 안전한 범위에서만 수행한다.

### config, balance와 commit을 세 시계열로 나눈다

Q에는 configured maximum, step initial effective balance와 committed scheduled sum이 있다. pause,
PP cadence나 dynamic policy는 initial balance를 줄일 수 있고, provisional grant/preemption은 중간
balance를 흔든다. 최종 runner Q와 assertion이 commit을 증명한다.

S에는 configured max, scheduler-owned active, runner-resident/in-flight와 이번 scheduled request 수가
있다. active인데 이번 Q=0인 request를 누락하지 않는다. K에는 total physical capacity, allocated unique,
shared refcount, free, reserved/lookahead와 fragmentation이 있다. W에는 available headroom snapshot,
static/session reserve, estimated peak와 actual allocator high-water가 있다.

세 시계열의 timestamp domain을 맞춘다. scheduler decision은 CPU monotonic time, CUDA memory peak는
async kernel completion 이후일 수 있다. step epoch와 event completion을 join key로 두지 않으면
다음 batch의 memory를 이전 decision에 붙일 수 있다.

### 증상 1: token budget utilization이 낮고 waiting이 많다

Q utilization 20%, waiting 100이면 Q를 올리고 싶어진다. 먼저 S active/limit을 본다. full이면 decode
request가 각각 한 token만 써 Q가 낮은 전형적 장면이다. S도 여유면 K allocation failures와 reserve,
W graph bucket/backend gate, structured output/encoder readiness를 본다.

반증 순서는 state mutation을 따른다. waiting 후보의 need가 0인지, active predicate가 false인지,
cache allocator가 None인지, transient shape selector가 fallback/deny했는지 확인한다. Q config를 바꾸는
실험은 first false가 Q인 decision 비율이 높을 때만 타당하다.

낮은 Q utilization이 항상 나쁜 것도 아니다. decode-only 32 request에서 Q=32이고 GPU가 model weight
streaming 때문에 충분히 바쁠 수 있다. token count는 FLOP/byte의 직접 대용이 아니다. 목표는 Q gauge를
100%로 만드는 것이 아니라 goodput과 latency SLO 아래에서 bottleneck을 이해하는 것이다.

### 증상 2: KV utilization은 낮은데 allocation이 실패한다

total free slot gauge가 30%인데 특정 request가 block을 얻지 못할 수 있다. hybrid cache group별 free,
block alignment, max blocks per request, contiguous/table limit, prefix block refcount와 page generation을
본다. logical free token 합과 allocator가 요구하는 compatible physical block이 다를 수 있다.

TP rank imbalance도 후보이다. cluster 평균은 낮지만 한 rank가 KV replication, 다른 graph reserve나
fragmentation 때문에 minimum capacity를 정할 수 있다. rank별 used/free와 allocation reason을 본다.
Transformers가 startup에서 M/N 최소를 맞춰도 runtime request distribution과 other allocations가
완전히 같다는 뜻은 아니다.

cache utilization 산식의 denominator도 확인한다. total pool에 padding block, reserved lookahead와
unallocatable safety zone가 포함되는지에 따라 70% 의미가 달라진다. metric definition을 source
allocator state와 맞춘다.

### 증상 3: active와 KV는 여유인데 prefill OOM이 난다

first suspect는 W다. 이번 M, vocabulary, explicit mask의 N, logits_to_keep 지원, backend workspace와
graph capture를 계산한다. OOM stack이 cache block constructor가 아니라 LM head/attention mask/capture에
있는지 확인한다. `allocated`와 `reserved`를 peak 직전·직후 비교하되 async completion을 맞춘다.

M을 절반으로 줄였을 때 선형 항만 있으면 peak 감소가 대략 절반이지만 M² mask가 지배하면 더 크게
줄 수 있다. backend를 mask-free path로 바꾸면 cmn/cmm 항과 실제 mask allocation이 사라지는지 본다.
option 문자열이 아니라 selected attention implementation과 tensor inventory로 반증한다.

vocabulary가 큰 model에서 logits peak가 지배하면 KV blocks를 줄이는 것보다 logits rows를 필요한
position으로 제한하는 path가 효과적일 수 있다. 하지만 model이 `logits_to_keep`을 소비하는지와
continuous processor가 output mapping을 유지하는지 correctness parity가 먼저다.

### 증상 4: 상한을 올렸더니 throughput은 늘고 tail latency가 악화된다

Q 또는 S 상한 확대는 한 step work를 키우고 batch efficiency를 높일 수 있다. 동시에 긴 prefill
chunk가 decode를 기다리게 하거나 active request service interval이 늘 수 있다. 평균 tokens/s와
p50만 보면 성공처럼 보이지만 p99 TTFT/ITL과 deadline goodput이 떨어질 수 있다.

step timeline에서 batch composition, logical/padded Q, duration, 각 request의 previous-service time을
본다. W peak가 커져 graph fallback/capture가 늘었는지, K pressure로 retraction이 늘었는지도 확인한다.
상한 변경은 한 predicate rhs만 바꾸지만 교차 효과로 다른 predicate false 비율을 높일 수 있다.

원인 판정은 “큰 batch라 느리다”가 아니다. query grant 증가→runner shape 증가→kernel duration 증가→
decode service gap 증가라는 연결을 보이거나, active 증가→KV pressure→preemption/recompute→tail 증가를
보여야 한다. 연결이 없으면 arrival burst나 network/output을 본다.

### 네 예산을 byte와 시간으로 환산하는 손계산

model `L=32,N_kv_local=8,D=128,b=2`라면 rank-local KV/token은
`2×32×8×128×2=131,072 byte`, 즉 128 KiB다. 100,000 physical token slot은 약 12.2 GiB다.
TP replication 여부와 allocator overhead를 빼지 않은 단순 baseline이다.

Q=512에서 hidden H=4096 BF16 activation 하나는 약 4 MiB지만 중간 QKV/MLP와 layer workspace는
더 크고 lifetime overlap에 따라 peak가 달라진다. vocabulary 100,000의 FP32 logits 512 rows는
약 195 MiB다. explicit bool mask M=512,N=100,000은 약 49 MiB이며 FP32라면 약 195 MiB다. 여러
attention group과 async pair가 있으면 multiplier가 붙는다.

S=256은 그 자체로 byte가 아니지만 block table이 request당 max 2,048 int32 entry면 table만 약
2 MiB다. output history/static tensor, processor state와 Python object가 추가된다. 더 중요한 것은
256 request의 평균 context가 1,000이면 KV가 32 GiB baseline이라는 점이다. S option의 간접 K 효과가
metadata보다 훨씬 클 수 있다.

이 계산은 정확한 profiler를 대체하지 않는다. 어느 항이 order-of-magnitude를 지배하는지 찾고 source
memory handler/allocator의 coefficient와 대조하는 용도다. model-specific GQA/MLA, TP, quant cache,
sliding window와 graph pool을 적용해 다시 계산한다.

### static source 감사에서 남길 증거

런타임을 실행하지 않는 이번 집필에서도 다음 연결은 source로 확정할 수 있다. option이 어느 config
field로 resolve되는지, validation이 어떤 조합을 거부/clip하는지, scheduler가 어느 predicate에서
field를 읽는지, allocation 성공 후 어떤 map/counter를 mutation하는지, assertion/runner input이
commit을 어디서 닫는지다.

반면 실제 GPU에서 selected backend, available VRAM, graph capture 성공, workload request length와
latency 효과는 source만으로 확정할 수 없다. 이 항목은 관측 계획으로 남긴다. “가능하다”와 “현재
배포에서 일어났다”를 문장 수준에서 분리한다.

고정 링크는 broad repository root보다 핵심 함수 범위를 가리킨다. vLLM scheduler의 init/schedule,
SGLang scheduler init/chunk/admission/retract, Transformers memory handler/cache/IO allocation을 서로
다른 증거로 둔다. 새 version에서는 line이 아니라 symbol과 state transition을 diff한다.

### 세 구현의 budget transaction을 함수 순서로 다시 걷는다

vLLM에서는 scheduler construction에서 Q와 S maximum을 snapshot한다. `schedule()` 진입은 step epoch와
Q balance를 만들고 KV manager에 새 step 시작을 알린다. running request마다 need를 계산해 여러
cap을 적용하고, encoder/alignment가 0을 만들면 다음 후보로 간다. grant가 양수면 KV manager의
`allocate_slots`를 호출한다. 성공 후 request→new blocks map과 request→scheduled tokens map을 쓰고
Q balance를 차감한다. 실패하면 preemption을 수행하며 provisional map과 balance를 rollback할 수 있다.

waiting loop는 단지 남은 Q가 양수인지 보지 않는다. running count, blocked/remote KV state, stale
async output, LoRA concurrency, prefix-cache lookup과 KV allocation이 교차한다. 새 request를 running에
옮기고 map을 쓰는 시점이 S/K ownership commit이다. 함수 끝의 token/running assertions 뒤 결과가
worker로 전달된다. 이 순서를 알면 metrics hook을 config parser나 runner 끝에만 두지 않고 decision
경계에 넣을 수 있다.

vLLM의 source comment는 scheduler가 엄격한 “prefill phase/decode phase” 두 queue만으로 동작하지
않고 computed frontier가 desired frontier를 따라잡는 token 회계를 사용한다고 설명한다. 그러므로
Q budget 관측도 request status 문자열보다 `num_computed_tokens`, `num_tokens_with_spec`, placeholders와
actual grant를 보는 편이 정확하다. speculative draft가 있으면 request의 logical token과 이번 verified
work가 다를 수 있다.

SGLang에서는 worker info가 K total, prefill Q limit, S running limit과 request length limit을 scheduler에
전달한다. `init_chunked_prefill`이 user chunk 값을 disabled/mixed/dynamic state로 정규화한다. runtime
prefill path는 allocatable request predicate를 보고, policy로 waiting priority를 계산한 뒤 current
running size와 dynamic chunk, reserve ratio, prefill limit을 `PrefillAdder`에 준다. adder가 request를
선택하면 batch와 cache allocator ownership이 mutation된다.

decode memory check는 별도 후속 방어선이다. optimistic admission 뒤 actual batch가 cache에 맞지 않으면
retract하고 available tokens를 회복하며 ratio tracker를 바꾼다. 이 mutation은 다음 step K reserve
추정에 영향을 준다. startup max와 현재 ratio만 기록하면 왜 ratio가 변했는지 잃으므로 old/new free,
retracted request 길이와 gained token을 같은 event로 둔다.

Transformers에서는 manager 생성 시 concrete paged cache가 memory handler를 호출한다. handler는 model
head/layer/dtype와 selected attention mask 필요, async multiplier, logprob output row를 coefficient로
만든다. available memory에서 M/N을 풀고 footprint를 검증한다. cache constructor가 TP 최소를 맞춘 뒤
max batch tokens와 pages를 runtime effective state로 보존한다.

runtime processor는 scheduler의 selected request를 static IO에 pack한다. scheduler가 Q/S/K를 만족해도
packer가 capacity를 넘지 않아야 하고 model runner가 selected graph/eager path로 W shape를 소비한다.
output commit 뒤 request finish가 blocks를 반환한다. startup polynomial은 worst-case feasibility이고
runtime free block predicate는 current occupancy다. 둘을 같은 “memory check”로 합치지 않는다.

세 transaction의 공통점은 limit→candidate demand→provisional ownership→commit/free다. 차이는 reserve
추정, phase representation, memory capacity 해결 시점과 rollback 경계다. 이 공통 골격만 사용하면
source symbol을 잃지 않으면서 비교할 수 있다.

### 사고 보고서 예제: 상한 두 개를 올린 뒤 OOM이 난 이유

초기 설정은 Q=256, S=32, K=80,000 physical slots다. 평균 decode context 1,500인 active 32개가
48,000 slot을 쓰고, waiting에는 4,000-token prompt 네 개가 있다. W는 M=256/N=80,000 조합에 맞게
잡혀 있다. 운영자는 낮은 Q utilization을 보고 Q=1,024, S=64로 동시에 올린다.

변경 전 decode 32개가 Q 32를 쓰고 prefill chunk 224가 섞이면 Q는 256이다. persistent K는 step마다
256 안팎 증가하며 headroom 32,000이 완만히 줄어든다. logits는 M=256 bucket, explicit mask가 있다면
MN 약 20.5 million 원소다.

변경 후 admission이 active를 48까지 늘렸다고 하자. 새 16 request의 평균 context가 prefix hit 없이
1,500이면 K demand 24,000이 즉시 추가돼 used가 72,000이다. 남은 8,000 slot은 긴 prefill 두 chunk와
future decode reserve에 부족할 수 있다. 동시에 Q=1,024 batch는 decode 48과 prefill 976을 허용한다.
mask MN은 약 81.9 million 원소로 네 배, M²은 16배가 된다. logits rows가 M에 비례하는 path라면
그 항도 네 배다.

OOM은 cache used 90%에서 attention mask 또는 logits allocation으로 나타날 수 있다. incident를 “KV가
가득 차 OOM”이라고만 쓰면 Q/W 확대를 놓친다. 첫 false predicate 기록은 startup validation이 새
M/N 조합을 실제로 검증했는지, runtime effective M/N이 무엇인지부터 본다. 사용자가 두 옵션을 동시에
강제해 footprint check를 통과하지 못했어야 한다면 validation bypass/available-memory snapshot도
조사한다.

사고를 재구성하는 표는 다음과 같다.

| 항목 | 변경 전 | 변경 후 | 확인할 source state |
|---|---:|---:|---|
| Q limit / actual | 256 / 256 | 1024 / 1024 | committed scheduled tokens |
| S limit / actual | 32 / 32 | 64 / 48 | running owner count |
| K used / total | 48k / 80k | 72k / 80k | unique physical slots |
| mask MN elements | 20.5m | 81.9m | selected backend/mask tensor |
| logits capacity | 256 rows | 1024 또는 sequence rows | runner/model support |

복구를 위해 두 option을 모두 원복하는 것은 안전할 수 있지만 학습은 적다. 먼저 S를 32로 유지하고
Q만 512로 올려 W peak와 TTFT/ITL을 본다. 다음 Q를 유지하고 S만 올려 K pressure와 decode throughput을
본다. production 실행 지시가 아니라 어떤 단일 state mutation이 어떤 predicate를 바꾸는지 분리한
검증 설계다.

Q만 512에서 OOM이 없고 prefill step이 줄면 W headroom 안의 이득이다. S만 48에서 retraction이 늘면
K reserve가 bottleneck이다. 둘을 합쳤을 때만 OOM이면 M/N 교차 항 또는 arrival synchronization이
후보다. polynomial의 MN 항처럼 interaction은 단일 option 실험의 합과 다를 수 있다.

사고 후 guardrail은 단순 “Q 최대 512”가 아니다. startup effective capacity와 runtime decision에
Q/S/K/W lhs/rhs를 export하고, 새 config 조합을 polynomial/profile validation이 거부하도록 한다.
workload context/vocabulary/backend가 바뀌면 안전 조합도 달라지므로 model revision과 함께 capacity
profile을 버전 관리한다.

### 긴 prompt 하나가 네 budget에 만드는 debt

prompt 16,384, output 128인 D가 들어왔다고 하자. Q=512라면 prefix hit 없이 prefill에 최소 32 step이
필요하다. S는 그 기간 내내 한 slot을 점유한다. K는 매 step 약 512 logical token씩 늘어 마지막에
prompt+output 길이만큼 남는다. W는 각 prefill step M=512와 계속 커지는 N의 attention read/mask peak를
겪는다. 하나의 request가 네 시간축에 서로 다른 debt를 만든다.

chunk를 1,024로 올리면 Q debt step은 약 절반이지만 한 step W가 커지고 K를 더 빠르게 소비한다.
S 점유 wall time은 TTFT가 줄면 짧아질 수 있으나 다른 decode 간섭으로 전체 service time은 workload에
따라 달라진다. prefix hit 12,288이 있으면 Q와 new K debt는 tail 4,096 중심으로 줄지만 shared block
refcount와 S는 남고 W attention read N은 긴 context를 반영할 수 있다.

max-new-token 128 전체를 admission reserve하면 K feasibility는 보수적이지만 다른 request를 늦춘다.
현재 필요한 block만 잡으면 utilization은 높지만 16,384 이후 decode에서 free slot이 없어 retract될
수 있다. SGLang ratio tracker 같은 추정은 이 사이를 workload에 맞춰 조절한다. 추정값의 변화는
정책 magic이 아니라 future K debt 예측의 변경이다.

D가 model max length나 total K capacity를 초과하면 chunking은 해결책이 아니다. chunk는 per-step Q/W를
제어하지만 최종 persistent requirement를 줄이지 않는다. sliding-window model이 실제로 old KV를
재사용/해제할 수 있는 architecture라면 K 식이 달라지지만 config와 backend cache semantics로 확인한다.

운영자가 “긴 prompt를 작은 chunk로 쪼갰는데도 admission이 안 된다”고 묻는다면 S/K feasibility를
본다. “admission은 되지만 TTFT가 너무 길다”면 Q grant와 service interval을 본다. “중간에 OOM”이면
증가한 N과 W peak, reserve misprediction을 본다. 같은 D request도 첫 false가 시간에 따라 이동한다.

**좋은 budget tuning이 증명해야 하는 것**

첫째, 어떤 option을 바꿨는지가 아니라 effective rhs가 바뀌었음을 증명한다. 둘째, 후보 request의
first false predicate가 그 rhs였음을 보인다. 셋째, consumer mutation과 physical shape/capacity가
따라 바뀌었음을 보인다. 넷째, correctness와 SLO 아래 goodput 개선을 측정한다.

네 단계 중 하나가 빠지면 결과 해석이 약하다. config만 바뀌면 no-op일 수 있다. first false가 다른
축이면 잘못된 knob다. mutation 없이 metric만 움직이면 workload noise일 수 있다. throughput만 늘고
tail/quality가 나빠지면 serving objective를 달성하지 못했다.

예산을 교집합으로 보면 “GPU가 남는데 왜 못 넣는가”에 답할 수 있다. compute token이 남아도 sequence
slot이 찰 수 있고, slot이 남아도 KV가 없을 수 있으며, KV가 남아도 logits/mask peak가 forward를
막을 수 있다. 반대로 transient headroom이 커도 current request가 model length validation을 통과하지
못하면 admission할 수 없다.

이 장은 batching 정책의 공정성이나 request 상태 enum을 다시 설명하지 않았다. 26장의 batch 구성
개론을 입력으로 네 resource predicate만 분리했고, 다음 28장이 다룰 상태 전이 앞에서 각 transition을
허용하는 비용 조건을 고정했다. scheduler를 읽을 때 queue 이름보다 먼저 budget lhs/rhs와 mutation을
찾는 습관이 남아야 한다.

최종적으로 동일 요청 A/B/C timeline을 다시 본다. A decode, B chunk, C admission에서 Q/S/K/W를
각각 계산하고, 첫 false를 기록하며, option을 바꾼 뒤 같은 row가 true로 바뀌었는지 본다. 그다음
새 first false가 어디로 이동했는지 확인한다. 튜닝은 bottleneck을 없애는 일이 아니라 종종 다른
예산으로 옮기는 일이다. 그 이동을 예측하고 관측할 수 있을 때만 시스템을 이해했다고 말할 수 있다.

이 관점은 capacity planning에도 적용된다. 하루 평균 prompt 길이만으로 GPU 수를 정하지 않는다.
arrival burst에서 동시에 active가 되는 request 분포 S, step마다 섞이는 prefill/decode Q, resident
context의 unique physical K와 selected backend의 W peak를 함께 모델링한다. 평균 길이가 같아도 모든
요청이 동시에 긴 prefix를 채우는 workload와 짧은 요청이 계속 교체되는 workload는 fragmentation,
logits rows와 service interval이 다르다.

capacity report에는 적어도 model artifact의 layer/head/dtype/vocabulary, TP/PP topology, cache page와
dtype, effective Q/S/K/W limit, prefix sharing 가정, prompt/output 분포와 graph/backend mode가 필요하다.
“GPU 80GB이므로 context n개”라는 결론은 weight, graph/static reserve와 transient peak를 뺀 뒤의
rank-local byte 식이 없으면 재현할 수 없다. cluster 합계보다 가장 작은 rank headroom이 실제 rhs가
될 수 있다.

부하가 변할 때 bottleneck 이동도 예측한다. 긴 prompt 비율이 늘면 Q prefill debt와 K growth, attention
W의 N 항이 커진다. 짧은 decode 동시성이 늘면 S와 logits rows가 먼저 찰 수 있다. vocabulary가 큰
model로 교체하면 같은 Q/S/K에서도 LM-head W가 커진다. KV quantization은 K를 줄여도 Q와 S를 바꾸지
않으며 dequant workspace/backend가 W를 바꿀 수 있다. 이런 문장으로 workload 변화와 option 효과를
연결해야 capacity plan이 단순 표가 아닌 인과 모델이 된다.

SLO별로 허용하는 reserve도 다르다. offline throughput은 K를 낙관적으로 채우고 큰 Q를 사용해도
괜찮을 수 있다. interactive serving은 decode ITL을 보호하도록 prefill chunk와 active admission에
headroom을 남길 수 있다. strict no-preemption workload는 future K를 보수적으로 예약하고, elastic
workload는 retraction debt를 허용할 수 있다. 어느 선택도 보편적 최적값이 아니다. 목표 latency와
실패 비용이 reserve policy를 정한다.

마지막 검산은 보존 법칙이다. committed Q 합은 step limit을 넘지 않고 provisional refund 뒤 runner
input과 같아야 한다. active owner 수는 limit과 맞고 finish/cancel에서 유한하게 줄어야 한다. allocated
unique K, shared refcount, free와 reserved 합은 allocator total과 맞아야 한다. W estimator의 static
inventory와 runtime owner는 누락된 graph/workspace를 설명해야 한다. 합이 맞지 않으면 성능 튜닝 전에
accounting 또는 lifetime bug부터 조사한다.

이 네 보존식은 dashboard 장식이 아니다. duplicate admission은 S 합을, leaked block은 K 합을,
stale async row는 Q commit을, graph/session leak은 W inventory를 깨뜨린다. budget 분석은 성능과
correctness를 같은 owner ledger에서 만나는 방법이기도 하다.

검토 문서를 유지할 때는 한계값만 갱신하지 않는다. source revision이 바뀌면 default resolution,
validation predicate, reserve heuristic과 memory coefficient가 변했는지 diff한다. 예전 Q/S/K/W 숫자를
새 binary에 그대로 옮기면 option 이름은 같아도 consumer가 달라졌을 수 있다. startup effective
state와 첫 schedule decision을 새 artifact 기준으로 다시 계산한다.

모델 교체도 같은 절차를 요구한다. hidden size와 layer 수는 activation·KV를, KV head와 cache dtype은
K byte를, vocabulary는 logits W를, maximum position과 sliding semantics는 feasibility를 바꾼다.
serving config만 version control하고 model config digest를 빼면 capacity 변화의 원인을 잃는다.

관측 비용 자체도 W와 latency를 소비한다. 모든 후보의 tensor나 allocator table을 매 step dump하면
CPU serialization, device synchronization과 storage가 decision을 교란한다. 기본 metric은 counter와
reason enum으로 두고, sampled trace에서 lhs/rhs와 request length bucket을 확장한다. CUDA memory peak를
읽기 위해 강제 synchronize한다면 그 실험은 production latency와 분리한다.

보안 측면에서도 token content는 예산 판정에 필요하지 않다. request ID의 안전한 digest, 길이,
block count, dtype, shape와 reason이면 대부분의 budget incident를 재구성할 수 있다. prefix hash와
adapter/model identity처럼 sharing correctness에 필요한 값도 원문 prompt 대신 versioned identifier로
남긴다. 좋은 관측은 원인을 충분히 가르면서 사용자 데이터를 최소화한다.

결국 예산 원장의 품질은 숫자의 개수보다 소유권과 단위의 정확성으로 결정된다. 각 값이 언제 생기고,
누가 차감하며, 어느 commit 뒤 환불되는지를 설명할 수 있어야 다음 장애에서도 같은 분석을 재사용한다.
그 설명이 곧 튜닝 전후를 비교하는 재현 가능한 계약이 된다.
그리고 분명하게 남는다.

**27.8의 종합 회고: 네 budget의 보존식을 다음 상태 기계로 넘긴다.**

다음 장에 넘기는 request에는 admission 전후의 상태만 있는 것이 아니다. 이번 step의 committed query
grant, active ownership slot, allocated persistent block과 transient execution shape가 붙어 있다.
state transition은 이 resource commitment가 성공했을 때만 일어나며 cancellation/preemption은 서로
다른 예산을 서로 다른 시점에 환불한다.

28장에서는 waiting, running, preempted, finished 같은 상태와 event를 본다. 그때 “왜 이 transition이
가능했는가”는 이 장의 predicate record로 답한다. Q balance 환불과 KV block free, active slot 제거,
async workspace reuse fence를 하나의 free라고 부르지 않는다. 네 lifetime을 유지한 채 상태 기계로
넘어간다.

이 handoff 자체가 마지막 검산이다.

## 27.9 다섯 예산을 하나의 admission 식으로 결합한다

후보 요청 `r`을 이번 iteration에 넣을 수 있는지는 독립 boolean 다섯 개가 아니라 같은 실행 계획을 가리키는 결합 predicate다. 기호를 고정한다. `T(plan)`은 scheduled query tokens, `S(plan)`은 active sequences, `B(plan)`은 필요한 unique KV blocks, `W(plan)`은 rank-local transient workspace peak, `G(plan)`은 선택한 graph bucket이 예약하는 bytes다. weight·runtime static reserve를 `M_static`, 안전 여유를 `M_guard`, device capacity를 `M_total`이라 둔다.

기본 조건은 `T≤T_max`, `S≤S_max`, `B≤B_free+B_reclaimable`이며 메모리는 `M_static + bytes(B_committed+B_new) + W(plan) + G(plan) + M_guard ≤ M_total`이다. 여기서 각 항을 독립적으로 현재값과 비교하면 안 된다. graph bucket 선택은 token/sequence shape에 의존하고 workspace도 같은 shape와 backend에 의존한다. 후보를 더한 plan을 먼저 만들고 그 plan에서 `W`와 `G`를 다시 평가한다.

구체적으로 현재 decode sequences 7개가 각 1 token이고 후보 prefill chunk가 120 tokens라고 하자. `T=127`, `S=8`이다. token limit128과 sequence limit8을 각각 통과한다. 후보의 context가 기존 100 tokens이고 page size16이면 append120 뒤 총 220이므로 필요한 전체 pages는 14다. 이미 7 pages를 가졌다면 새 blocks7개가 필요하다. prefix sharing과 partial tail을 고려한 allocator의 실제 delta를 써야지 `ceil(120/16)=8`을 그대로 쓰지 않는다.

graph bucket이 token capacity128에서 256으로 반올림되고 그 variant가 추가 80MiB static/persistent buffer를 요구하며 attention workspace가 current96MiB에서 candidate224MiB로 증가한다고 하자. KV block 하나의 모든 layer rank-local bytes가 2MiB라면 새 KV14MiB다. 후보 이전 free headroom이 250MiB여도 증가분은 `14+80+(224-96)=222MiB`라 28MiB만 남는다. guard64MiB를 요구하면 false다. T/S/K를 각각 통과해도 결합 memory predicate가 거절한다.

반대로 graph reserve80MiB가 이미 startup에 `M_static`으로 잡힌 variant라면 후보마다 다시 더하면 double count다. graph pool이 max bucket 전체를 선예약하는지, bucket 전환 때 추가 allocation하는지 source와 allocator snapshot에서 확인한다. workspace가 graph private pool에 포함되는지도 본다. 식의 항은 개념 이름이 아니라 실제 owner가 중복 없이 한 번씩 나타나야 한다.

결합 함수는 `(admit, first_false, estimate, reservations)`를 반환하도록 읽는다. `first_false`는 evaluation order가 아니라 교정 가능한 원인을 의미해야 한다. 여러 predicate가 false면 full vector를 trace에 두고 API reason은 bounded priority로 고를 수 있다. 그렇지 않으면 token limit만 낮춰 보이는 요청이 실제로는 memory도 초과해 반복 튜닝을 만든다.

## 27.10 provisional reservation에서 commit까지

admission check와 mutation 사이에 다른 request가 자원을 가져갈 수 있다. scheduler가 A와 B를 순차 평가하면서 둘 다 같은 free blocks20을 보고 각 12 blocks가 가능하다고 판단한 뒤 함께 commit하면 24를 요구한다. check 결과를 advisory로만 쓰고 commit에서 원자 reservation을 하지 않은 전형적 TOCTOU다.

plan builder는 후보를 넣을 때 token grant, sequence slot, KV blocks, workspace/graph capacity를 provisional ledger에 차감한다. 다음 후보는 이미 예약된 값을 본다. 최종 plan이 runner에 publish될 때 committed generation으로 전환한다. materialization 실패나 후보 제거에서는 해당 reservation만 환불한다. global counters를 이전 snapshot으로 되돌리면 그 사이 다른 owner의 commit을 지울 수 있다.

workspace는 allocator object처럼 명시적 block handle이 없을 수 있다. 그래도 peak envelope reservation이 필요하다. candidate plan의 selected backend/shape가 peak를 결정하고, graph pool 또는 eager allocator가 실제로 그 envelope 안에서 실행한다는 계약을 둔다. estimator가 underestimate하면 runtime OOM, overestimate하면 false reject다. observed peak와 estimate error distribution을 shape bucket별로 남긴다.

graph bucket도 capacity slot으로 다룬다. capture가 이미 존재한다고 memory가 0은 아니다. replay에 필요한 persistent input/output buffers와 private pool의 lifetime을 inventory한다. 여러 concurrent execution streams가 같은 graph buffers를 공유할 수 있는지, serialize되는지, generation별 replica가 필요한지 확인한다. concurrency를 늘리면 graph reserve가 복제될 수 있다.

commit boundary는 단순 queue append가 아니다. KV allocation, request state running 전환, runner batch update와 graph/workspace selection이 모두 성공한 뒤 plan generation을 publish한다. 구현이 일부를 lazy하게 runner에서 수행하면 scheduler commit은 provisional admission이고 runner acknowledgment 뒤 final commit일 수 있다. 28장의 상태 기계에는 이 중간 상태를 숨기지 않는다.

rollback fixture는 세 지점에 실패를 넣는다. blocks reserve 뒤 graph bucket unavailable, graph selection 뒤 workspace allocation failure, runner materialization 뒤 publish cancellation이다. 각 경우 token grant, sequence slot, blocks/refcounts, graph execution right와 workspace buffer가 exactly once 반환되는지 본다. 후보의 client terminal과 resource terminal을 별도로 닫는다.

## 27.11 oversubscription 사고: 각각은 맞고 합계는 틀리다

GPU capacity80GiB에서 weights와 runtime static이 60GiB, guard2GiB라 가용 execution envelope는 18GiB라고 하자. 현재 KV가 12GiB, graph pools2GiB, workspace current1GiB를 쓴다. nominal free는 3GiB다. scheduler 두 replica 또는 두 async planners가 후보 P와 Q를 거의 동시에 평가한다.

P는 KV1.2GiB와 workspace delta0.8GiB, Q는 KV1.0GiB와 workspace delta1.0GiB를 요구한다. 각각 현재 free3GiB에 대해 P2.0GiB, Q2.0GiB라 통과한다. 둘을 합치면 4.0GiB로 1GiB 초과다. token limits도 P/Q를 별 plan으로 계산해 각각 128 이하이고 sequence도 limit 이하라고 기록됐다. 둘이 같은 allocator generation을 예약하지 않고 commit해 runner에서 OOM이 발생한다.

더 교묘한 변형은 graph bucket 전환이다. P를 넣으면 capacity128 variant, Q를 넣어도 개별 snapshot에서는 128이다. 둘을 합친 active shape는 129가 되어 capacity256을 선택하고 새 pool1.5GiB와 workspace delta가 추가된다. 개별 비용의 단순 합보다 combined plan peak가 크다. admission cost는 request별 scalar가 아니라 set-dependent function이다.

timeline은 `t0 free snapshot`, `t1 P estimate`, `t2 Q estimate`, `t3 P block reserve`, `t4 Q block reserve`, `t5 combined runner shape`, `t6 graph bucket escalation`, `t7 allocation failure`다. first divergence는 t1/t2가 같은 unreserved capacity generation을 소비한 지점이다. OOM은 결과다. allocator fragmentation이나 model leak을 먼저 고치지 않는다.

증거에는 planner generation, candidate set, all predicate lhs/rhs, provisional/committed block handles, selected graph bucket 전후, workspace estimator와 actual allocator peak를 둔다. CUDA OOM 문자열만으로는 weight, KV, graph, workspace 중 owner를 가를 수 없다. memory snapshot을 강제 동기화하면 timing을 바꾸므로 sampled controlled reproduction과 production counters를 구분한다.

수정은 하나의 plan transaction 안에서 모든 후보를 누적 평가하거나 shared allocator에 compare-and-reserve generation을 둔다. set-dependent graph/workspace는 최종 candidate set 변화마다 재평가한다. 마지막 후보 때문에 bucket이 커지면 그 후보를 제거한 plan과 비교해 marginal cost를 구할 수 있다. P/Q 중 어느 것을 거절할지는 fairness와 priority 정책이며 이 장은 feasibility를 제공한다.

회귀는 P-only, Q-only, P+Q order swap, threshold128/129, concurrent planner barrier, cancellation-before/after-reserve를 포함한다. 합격 조건은 no OOM뿐 아니라 accepted set이 식을 만족하고 rejected reservation이 모두 환불되며 next iteration에서 capacity가 회복되는 것이다. 지나치게 보수적으로 둘 다 거절하는 수정은 correctness는 지켜도 goodput 회귀다.

## 27.12 allocator·selector·runner source walk

고정 source에서 option parser부터 시작하지 않고 schedule decision의 consumer를 먼저 찾는다. vLLM scheduler가 token budget을 차감하고 KV manager allocation transaction을 호출하는 span, runner new-state가 physical shape를 만드는 span, graph/backend selector가 capacity를 고르는 span을 연결한다. running allocation failure가 어떤 rollback과 preemption으로 귀결되는지 본다.

SGLang에서는 admission transaction, paged allocation과 decode OOM/retraction 경로를 잇는다. new prefill batch를 고르는 policy가 token/sequence를 계산한 뒤 schedule batch/allocator가 실제 pages를 얻는 순서를 기록한다. overlap mode에서 workspace/result buffers가 몇 generations 겹치는지 확인한다. scheduler의 `available_size`와 CUDA allocator free bytes를 같은 숫자로 보지 않는다.

Transformers continuous scheduler의 candidate processing과 paged cache allocate/decrease-ref-count, batch tensor preparation과 model runner compute를 잇는다. cache blocks가 통과해도 logits/mask tensor shape가 vocabulary와 rows 때문에 커질 수 있다. exception handler가 candidate state와 cache reference를 함께 되돌리는지 본다.

llama.cpp에서는 slot admission, `llama_batch` construction, KV cell allocation과 backend graph/work buffer planning을 같은 request timeline에 놓는다. stable server slot count가 token rows나 KV feasibility를 보장하지 않는다. CUDA graph 사용 시 captured capacity와 backend buffer ownership을 찾는다.

source card에는 revision, symbol/span, input unit, lhs/rhs calculation, mutation, acknowledgment, failure rollback과 next consumer를 둔다. estimator와 allocator가 다른 단위를 쓰면 변환식을 기록한다. blocks를 bytes로 바꿀 때 layer, KV heads, head size, dtype, TP sharding과 page tokens를 포함한다.

## 27.13 관측·반증·용량 계획

decision event는 plan generation, candidate cohort, T/S/B/W/G lhs와 rhs, selected bucket/backend, first false와 all false mask를 갖는다. request ID는 trace에만 두고 metric labels는 model/config generation, reason, phase, bucket처럼 bounded하게 둔다. accepted/rejected counters와 current gauges를 섞지 않는다.

KV dashboard는 allocator total, free, reserved provisional, committed unique, shared references와 reclaimable을 보존식으로 맞춘다. CUDA memory는 weights/static, graph pool, workspace current/peak, cache pools, miscellaneous/fragmentation으로 나눈다. framework reserved와 allocated 차이를 곧 usable free로 해석하지 않는다.

estimator error는 `(actual_peak-estimated_peak)`을 shape/backend bucket별 histogram으로 둔다. positive tail은 OOM risk, 큰 negative는 false rejection risk다. actual 측정이 asynchronous면 interval과 stream completion을 맞춘다. global peak 하나를 request cost로 배분하지 못하면 combined plan 수준에서만 판정한다.

capacity planning은 prompt/output 분포를 candidate set distribution으로 바꾼다. 평균 token length에 평균 bytes를 곱하지 않는다. graph threshold 근처의 동시 shape, long-context KV tail과 vocabulary-dependent logits peak가 결합되는 확률을 본다. p99 workload가 p99 항들의 단순 합과 같지 않으므로 trace replay 또는 conservative scenario를 사용한다.

falsifier를 명시한다. token false 가설은 T headroom을 늘려도 동일 request가 memory predicate에서 거절되면 불충분하다. KV false 가설은 blocks가 충분한데 workspace/graph peak 직전 OOM이면 탈락한다. leak 가설은 workload drain 뒤 owner inventory가 기준선으로 돌아오면 약해진다. fragmentation 가설은 largest alloc failure와 total free의 차이, pool behavior로 검증한다.

## 27.14 옵션 변경과 rollback terminal

`max_num_batched_tokens`, `max_num_seqs`, cache utilization/page 설정, graph capture sizes, workspace/backend 옵션을 parser→normalized config→constructed scheduler/allocator/runner→predicate coefficient→physical effect로 걷는다. option 이름만 표로 만들지 않는다. default, auto resolution, validation clamp와 capability fallback 뒤 effective value를 기록한다.

token limit을 올리면 T rhs만 바뀐다고 끝내지 않는다. 후보 set이 커져 S, new KV blocks, graph bucket과 workspace peak가 함께 변한다. sequence limit도 decode rows와 logits shape를 바꾼다. cache 비율은 KV pool을 늘리지만 graph/workspace guard를 잠식할 수 있다. graph variant 추가는 padding을 줄여도 persistent reserve를 늘릴 수 있다.

canary는 threshold 양쪽 fixture를 포함한다. T `limit-1,limit,limit+1`, S 경계, page tail `15/16/17`, graph bucket `127/128/129`, workspace vocabulary/rows 경계를 교차한다. 각 fixture에서 accepted plan의 combined 식과 all resource terminals를 검증한다. 평균 traffic만 흘리면 discontinuity를 놓친다.

rollback은 option 파일을 되돌리는 것 이상이다. provisional reservations를 막고 inflight committed generations를 drain하며 graph/workspace pool과 KV blocks를 reconcile한다. old/new config generation이 같은 allocator를 공유하면 accounting을 분리한다. rollback 완료는 queue 재개보다 owner ledger 기준선과 no stale generation으로 판정한다.

이 장의 최종 산출물은 실제 source span으로 뒷받침된 combined admission equation, P/Q oversubscription timeline, estimator error dashboard와 rollback matrix다. 독자는 “메모리가 부족했다” 대신 어느 candidate set에서 어떤 항이 처음 false였고 어떤 reservation 경쟁이 식을 깨뜨렸는지 설명할 수 있어야 한다. 그 predicate record를 28장의 request state transition에 넘긴다.

**결합식을 직접 채우는 워크시트.**

모델과 rank-local 좌표부터 적는다. GPU usable capacity는 79GiB, weights·runtime static은 58GiB, graph pool3GiB, guard2GiB라고 하자. KV와 transient workspace가 함께 쓸 envelope는 16GiB다. 현재 unique KV가 11GiB, current workspace peak가 1.5GiB이므로 단순 headroom은 3.5GiB다. framework가 reserved했지만 미사용인 bytes를 자동으로 더하지 않는다. 그 pool의 owner와 largest allocation 가능성을 확인한다.

현재 active sequences가 30, scheduled decode tokens30이고 limits가 `S_max=32`, `T_max=256`이다. 후보 A는 prefill chunk96 tokens와 새 KV blocks24개, 후보 B는 decode1 token과 새 block1개를 요구한다. block rank-local bytes가 32MiB라면 A KV delta768MiB, B32MiB다. A만 넣으면 S31/T126, B만 넣어도 S31/T31이라 두 limit은 넉넉하다.

A의 shape는 graph bucket128과 workspace2.2GiB를, B는 bucket32와 workspace1.6GiB를 고른다고 하자. 현재 대비 A workspace delta0.7GiB와 KV0.75GiB의 합 1.45GiB는 headroom3.5GiB 안이다. B delta 약 0.13GiB도 통과한다. 그러나 A+B combined tokens127, sequences32의 graph selector가 sequence bucket32/token bucket256 조합을 골라 graph scratch1GiB를 추가하고 workspace3.4GiB를 요구할 수 있다.

combined 증가분은 A/B KV 약 0.781GiB, workspace current 대비 1.9GiB, graph scratch1GiB로 약 3.681GiB다. headroom3.5GiB를 181MiB 초과한다. T와 S는 정확히 경계 안이고 KV allocator도 blocks25개를 독립적으로 줄 수 있지만 전체 plan은 false다. “각 request가 들어간다”에서 “둘을 같은 iteration plan에 넣을 수 있다”로 질문을 바꿔야 한다.

후보 B를 다음 iteration으로 미루면 A plan은 통과하고, A 실행 뒤 workspace가 해제된 다음 B를 넣을 수 있다. 이는 B를 영구 reject하는 것과 다르다. feasibility reason과 scheduling defer reason을 분리한다. 반대로 A가 long prefill이라 decode B의 ITL을 보호하려고 A chunk를 64로 줄이면 graph bucket과 workspace가 낮아져 둘을 함께 넣을 수 있을 수 있다. chunk policy는 29장이 다루지만 식은 그 선택의 물리 효과를 보여 준다.

워크시트는 각 숫자 옆에 source owner를 쓴다. token grant는 scheduler, active slot은 request table, block delta는 KV allocator, bucket은 graph selector, workspace peak는 runner/backend estimator가 만든다. 한 함수가 모든 값을 소유한다고 가정하지 않는다. snapshot generation이 서로 다르면 결합식 자체가 의미가 없으므로 각 row에 config/plan/allocator generation을 둔다.

**false-admit 조사 순서.**

첫째, OOM 직전 accepted plan을 복원한다. error를 낸 request 하나가 아니라 같은 runner invocation에 포함된 전체 candidate set을 본다. T/S/B/W/G estimate와 actual selected shape를 나란히 둔다. scheduler trace에 graph bucket이 없다면 runner acknowledgment에서 다시 연결한다.

둘째, estimate와 mutation을 구분한다. blocks `can_allocate`가 true였는지, 실제 handles reserve가 성공했는지, workspace는 estimate만 있었는지 실제 allocation이 있었는지 적는다. “admission passed”는 어느 단계까지 통과했는지에 따라 의미가 다르다. scheduler queue 전환만 성공하고 runner materialization이 실패했으면 상태 rollback도 필요하다.

셋째, double count와 missing count를 동시에 찾는다. graph pool이 static inventory와 candidate delta에 두 번 들어가면 false reject다. 반대로 graph replay private workspace가 둘 다 빠지면 false admit다. KV shared prefix는 logical blocks 합보다 unique physical이 작지만 copy-on-write tail과 reference generation을 누락할 수 있다.

넷째, concurrency를 재현한다. 두 planner가 동일 allocator version을 읽는 barrier fixture를 만들고, 둘 다 estimate한 뒤 commit 순서를 바꾼다. compare-and-reserve가 있다면 한쪽이 stale generation으로 실패해야 한다. 둘 다 성공한다면 total reserved 보존식이 capacity를 넘지 않는지 확인한다.

다섯째, 실패 후 terminal을 본다. OOM request에 error를 보낸 것만으로 복구가 아니다. provisional token/sequence grant, allocated blocks, graph execution lease와 partial workspace가 반환돼야 한다. 같은 batch의 이웃 requests가 retry에서 output cursor를 중복 commit하지 않는지도 본다.

**oversubscription과 fragmentation을 가른다.**

total free가 2GiB인데 1GiB contiguous allocation이 실패하면 fragmentation 또는 pool 제약이 후보가 된다. 그러나 caching allocator의 virtual/segmented behavior와 requested alignment를 알아야 한다. KV는 fixed pages가 충분해도 workspace가 큰 contiguous buffer를 요구할 수 있다. blocks count만 정상이라 memory feasibility가 정상인 것은 아니다.

oversubscription은 owner 합이 envelope를 넘는다. fragmentation은 합계상 공간이 있어도 요청 shape를 만족하는 allocation이 없다. 둘은 함께 일어날 수 있다. owner ledger 합, largest free segment 또는 allocator retry/split events, failed allocation size를 함께 둔다. `reserved≫allocated` 한 지표만 보고 cache를 비우는 대응을 반복하지 않는다.

graph private pool이 주소 안정성을 위해 일반 allocator에 반환되지 않는다면 nominal free와 reusable free가 다르다. capture variants를 많이 만들고 traffic이 사라져도 pool이 남을 수 있다. 이것은 반드시 leak은 아니지만 capacity equation의 static inventory에 들어가야 한다. variant eviction이 있다면 last replay consumer와 capture generation 뒤에만 반환한다.

수정 검증은 원인별로 다르다. oversubscription 수정은 atomic plan reservation과 combined shape 재평가를 검증한다. fragmentation 대응은 allocation shape, pool policy와 warm-up을 검증한다. estimator underestimate는 safety margin과 coefficient를 고치되 지나친 false reject를 측정한다. 원인이 다른데 모두 token limit을 낮추면 우연히 OOM 빈도만 줄고 capacity를 낭비한다.

**사건을 observation에서 rollback까지 닫는다.**

관측은 `accepted` 직후 OOM과 이웃 요청의 preemption이다. 먼저 accepted generation의 결합 ledger를 복원한다. token127/256, sequence32/32, KV25 blocks available40만 보면 통과다. 그러나 runner가 token bucket256을 선택하면서 workspace가 3.4GiB로 뛰고 graph scratch1GiB가 붙어 execution envelope를 181MiB 넘었다. 첫 false는 KV가 아니라 combined memory다.

branch는 scheduler의 후보 누적 loop, allocator reserve 결과, runner의 graph selector와 backend workspace 선택으로 나눈다. 후보별 estimate branch는 bucket128을 보았고 최종 runner branch는 bucket256을 보았다. 같은 `batch_size` 로그가 서로 다른 token/sequence 단위를 써 차이를 숨겼다. source card에 input shape와 반환 capacity 단위를 함께 적는다.

원인은 최종 candidate set에서 selector를 다시 호출하지 않은 것이다. scheduler는 후보별 marginal cost를 더했지만 graph/workspace가 set-dependent한 계단 함수라는 사실을 잃었다. allocator는 요청받은 blocks를 올바르게 제공했다. CUDA allocator도 capacity 이상 요청을 올바르게 거절했다. 따라서 cache 비율이나 allocator 구현을 먼저 바꾸지 않는다.

수정은 후보 append마다 combined shape를 재평가하고 provisional ledger에 graph/workspace envelope를 예약한다. 마지막 append가 bucket 경계를 넘으면 새 total을 기준으로 후보를 defer한다. commit은 allocator generation compare와 runner plan acknowledgment 뒤에만 running state를 publish한다. estimate와 actual selector 결과가 다르면 plan을 실행하지 않고 reservation을 환불한다.

검증은 127/128/129 token과 31/32/33 sequence 경계를 교차한다. P/Q 평가 순서를 바꾸고 두 planner를 barrier로 동시에 실행한다. accepted plan마다 식이 true이고 actual peak가 estimate+guard 안이며, rejected 후보의 blocks와 slots가 0으로 돌아오는지 본다. 이웃 요청 output cursor와 KV generation도 보존한다.

rollback은 새 admission generation을 차단하고 committed plans를 bounded drain한다. provisional reservations는 owner별 취소하고, partially materialized runner batch는 publish하지 않는다. graph pool은 last replay event 뒤 정리하며 KV handles는 request terminal과 allocator acknowledgment를 모두 남긴다. 이전 estimator를 되살리되 임시 conservative cap으로 재발을 막고, reconciliation gap이 0일 때 admission을 다시 연다.

사후 문장은 이렇게 쓴다. “G214의 후보 P/Q는 개별 T/S/K 검사를 통과했지만 combined token bucket이 128에서 256으로 전환되며 graph·workspace 증가분이 execution envelope를 181MiB 초과했다. 최종 set 재평가와 generation reservation을 추가했고 경계·동시 commit·cancel matrix에서 false admit와 잔류 자원이 0이었다.” 증상, 첫 branch, 수치 원인, 수정과 terminal이 한 문장에 연결된다.

**실제 source card를 완성하는 방법.**

첫 카드는 scheduler candidate loop다. 입력 collection, iteration token budget, sequence limit, 후보 정렬과 tentative append 위치를 적는다. 함수가 요청별 scheduled token을 잘라내는지, full request를 넣고 뒤에서 clamp하는지 확인한다. 반환값의 `num_tokens`가 prompt debt, 이번 grant, 누적 context 중 무엇인지 symbol consumer까지 따라간다. 같은 이름을 다른 단위로 해석하면 KV delta 계산도 틀어진다.

둘째 카드는 KV allocator다. `can_allocate`와 `allocate`가 같은 snapshot을 보장하는지, block 수가 logical인지 unique physical인지, shared prefix와 partial tail을 어떻게 세는지 적는다. allocation handle에 owner request와 generation이 있는지, failure가 exception인지 false인지, caller가 이미 차감한 token/sequence reservation을 어떻게 환불하는지 연결한다.

셋째 카드는 graph selector다. key에 token rows, sequences, decode/prefill phase, dtype/backend와 adapter가 들어가는지 확인한다. requested shape를 어느 bucket으로 올리는지, capture가 없을 때 eager fallback인지 on-demand capture인지 본다. selector 반환이 capacity만 주고 memory inventory는 다른 registry에 있다면 두 source를 같은 card에 연결한다.

넷째 카드는 workspace estimator와 runner materialization이다. logits rows×vocabulary, attention temporary, mask, collective staging과 multimodal buffers 중 어떤 peak를 포함하는지 본다. 동시에 살아 있지 않은 tensors를 단순 합하면 과대평가하고, 다른 streams/generations가 겹치는데 max 하나만 쓰면 과소평가한다. allocation lifetime을 timeline으로 그려 peak 식을 검산한다.

다섯째 카드는 acknowledgment와 rollback이다. runner가 plan을 받았다는 응답과 buffers materialized, graph selected, execution submitted가 각각 다른 상태인지 본다. scheduler가 어느 acknowledgment에서 provisional을 committed로 바꾸는지 적는다. exception handler가 KV만 free하고 sequence slot이나 output mailbox를 남기지 않는지 확인한다.

네 구현을 비교할 때 class 이름을 맞추려 하지 않는다. vLLM의 scheduler/KV manager/model runner, SGLang의 scheduling policy/batch allocator/overlap runner, Transformers continuous scheduler/cache/runner, llama.cpp의 server slot/KV cells/backend graph가 같은 질문에 답하는 좌표다. process boundary와 ownership이 다르면 acknowledgment 방식도 다르다. 공통 표는 차이를 지우는 요약이 아니라 차이를 정확히 놓는 틀이다.

각 card의 끝에는 반증을 둔다. scheduler가 final combined shape를 실제로 selector에 전달한다면 “후보별 estimate만 쓴다”는 가설은 탈락한다. allocator handle이 atomic reservation을 제공한다면 TOCTOU 가설은 약해진다. observed peak가 estimator 안에 있는데 OOM이면 fragmentation이나 untracked owner를 본다. source 존재만으로 production branch가 실행됐다고 단정하지 않고 effective trace로 확인한다.

**운영자가 사용하는 decision ledger.**

한 행은 plan generation 하나다. columns는 candidate count, T lhs/rhs, S lhs/rhs, B requested/free/reserved, KV bytes, selected graph key/capacity/bytes, W estimate/observed, static/guard/total, decision과 reason이다. provisional reservation IDs와 commit/rollback timestamps를 별 child table에 둔다. 한 행에 request 원문을 넣지 않는다.

accepted 행은 `memory_lhs≤memory_rhs`를 만족해야 하고 runner actual shape가 estimated selector와 같아야 한다. rejected 행은 mutation이 없어야 한다. deferred 행은 다음 iteration에서 새로운 generation으로 다시 계산한다. 이전 snapshot의 free 값을 재사용하지 않는다. cancelled 행은 어느 commit 전후인지 표시하고 owner별 release timestamp를 갖는다.

시간축 metric은 scheduler decision latency, block reserve latency, runner materialization, graph selection/fallback, execution과 rollback cleanup을 나눈다. OOM counter 하나로 합치면 어느 단계에서 false admit가 나타났는지 모른다. reason labels는 token, sequence, KV, combined-memory, graph-unavailable, workspace-estimate-mismatch처럼 bounded하게 둔다.

보존식도 plan별로 맞춘다. `free+provisional+committed+shared-owned=total blocks`는 allocator semantics에 맞춰 중복 없는 physical units로 쓴다. sequence slots는 waiting이 아니라 running ownership만 셀 수 있다. token grants는 iteration 종료 뒤 소모되고 다음 plan에 그대로 carry하지 않는다. workspace/graph leases는 last stream consumer 뒤 반환된다.

alarm은 결과와 선행 지표를 함께 둔다. runtime OOM은 늦은 결과다. estimator positive error tail, repeated bucket escalation, provisional age, rollback gap과 largest allocation headroom은 선행 지표다. false rejection은 rejection reason이 많은데 actual headroom이 지속적으로 남고 estimator negative error가 큰 cohort에서 찾는다.

**옵션 변경의 인과 카드를 작성한다.**

예를 들어 token limit을 128에서 256으로 바꾼다. parser가 integer를 받고 model/config constraints로 normalize하는지, scheduler token budget object가 새 rhs를 받는지 확인한다. 그 결과 candidate chunk와 combined shape가 어떻게 달라지고 selector가 bucket256을 택하는지 적는다. 최종 효과는 단순 token 두 배가 아니라 workspace, graph padding, KV growth와 decode service interval 변화다.

sequence limit을 32에서 64로 늘리면 decode 동시성은 늘 수 있지만 각 row logits와 sampling state, block table, output queue가 증가한다. graph sequence bucket이 64로 올라가 persistent reserve가 붙을 수도 있다. `max sequences`가 validation에서 memory에 맞춰 clamp되거나 executor topology별 rank-local 값으로 변환되는지도 본다.

cache utilization을 높이면 KV blocks rhs가 늘지만 총 device memory가 늘지는 않는다. graph/workspace/guard 영역을 줄여 같은 combined 식의 다른 항을 압박할 수 있다. startup pool partition이 고정이면 runtime free metric과 option 효과가 다르다. cache dtype을 줄이면 block bytes는 줄지만 backend dequant workspace나 지원 graph가 달라질 수 있다.

graph capture sizes를 촘촘하게 추가하면 padding과 eager fallback은 줄지만 capture pool 총량과 warm-up 시간이 늘어난다. 어떤 variants가 동시에 resident인지 확인한다. unused variant eviction이 없다면 rarely used bucket도 static inventory다. option의 이득은 bucket histogram과 pool cost를 함께 계산한다.

각 카드에는 성공·실패·rollback 기준이 있다. 성공은 target cohort goodput/TTFT/ITL 개선과 estimator error/terminal gap 허용 범위다. 실패는 false admit, cross-request preemption, allocator 기준선 미복귀 또는 tail 악화다. rollback은 new config admission fence, inflight drain, provisional cancel, pool/cache reconciliation과 old generation readiness 순서다.

**최종 회귀 매트릭스와 독자 과제.**

축은 token boundary, sequence boundary, page tail, graph bucket, workspace backend, concurrency와 cancel timing이다. 모든 조합을 무작정 곱하지 않고 pairwise 기본 세트와 사고 조건의 full interaction을 둔다. 핵심은 token127/128/129, sequences31/32/33, page15/16/17, combined P/Q, concurrent snapshot과 cancellation-after-reserve다.

fixture마다 expected decision, first/all false mask, selected bucket, block delta, workspace envelope와 terminal resources를 적는다. P-only와 Q-only는 accept, P+Q는 guard 때문에 one defer가 기대다. candidate order를 바꿔도 feasible accepted count와 안전성은 같아야 하지만 priority 정책에 따라 선택 request는 달라질 수 있다.

failure injection은 allocator reserve 실패, selector missing variant, workspace allocation OOM, runner acknowledgment timeout과 client cancel을 각 commit 경계에 둔다. 이웃 request output/KV는 변하지 않아야 한다. retry가 있다면 old generation reservation과 output cursor를 재사용하지 않게 idempotency identity를 확인한다.

독자 과제는 자신의 model config에서 KV block bytes를 계산하고 실제 allocator 단위와 맞추는 것이다. 그다음 traffic의 한 plan을 골라 T/S/B/W/G ledger를 채운다. 모르는 coefficient는 임의 숫자로 숨기지 않고 미검증으로 표시하고 필요한 source span 또는 controlled measurement를 적는다.

완료 판정은 “OOM이 사라졌다”보다 강하다. 모든 accepted plan이 combined equation을 만족하고, estimate와 selected actual shape가 연결되며, reject/defer/cancel의 reservations가 유한 시간 안에 0이 된다. threshold 양쪽에서 goodput과 SLO가 목표를 만족하고 rollback 뒤 old/new owner ledger가 분리돼야 한다.

이제 28장으로 넘길 record가 선명하다. request는 waiting 또는 running이라는 이름만 갖지 않는다. 어느 plan generation에서 어떤 T/S/B/W/G reservation을 획득했고, 어떤 acknowledgment에서 commit됐으며, transition 실패 시 무엇을 환불해야 하는지가 붙는다. 이 정보가 있어야 상태 기계의 화살표가 자원 현실과 맞는다.

**배포 당일 30분 점검표.**

첫 5분에는 binary/source revision, model digest, topology와 effective limits를 snapshot한다. CLI에 적은 값과 runtime normalized 값이 같은지 확인한다. auto cache sizing과 graph warm-up 뒤 실제 pool inventory가 startup 계산과 맞는지 본다. 이 좌표가 없으면 canary와 baseline의 memory 차이를 코드 변화로 귀속할 수 없다.

다음 5분에는 경계 fixture 하나를 canary에 보낸다. token128, sequence32, page tail과 graph bucket 경계를 의도적으로 밟게 한다. decision event의 T/S/B/W/G와 runner selected shape를 한 trace에서 연결한다. request payload는 저장하지 않고 lengths, shape, safe digest와 generation만 사용한다.

세 번째 5분에는 normal traffic conservation을 본다. accepted plans의 combined lhs 최대, provisional oldest age, blocks 보존 gap, estimator error와 rollback residue를 확인한다. current CUDA free 하나가 안정적이어도 owner ledger가 새고 있으면 중단한다. 반대로 caching allocator reserved가 크다는 이유만으로 leak이라고 판단하지 않는다.

네 번째 5분에는 latency를 본다. admission decision, queue, materialization, graph fallback과 execution을 분리한다. 새 conservative guard가 OOM을 없앴지만 false reject/defer를 늘려 TTFT를 악화할 수 있다. target cohort와 이웃 decode cohort의 TTFT/ITL을 함께 본다. throughput 평균만으로 승격하지 않는다.

다섯 번째 5분에는 cancel과 실패를 주입한다. reserve 직후 cancel, runner ack timeout과 selector fallback을 소량 실행한다. client terminal뿐 아니라 sequence slot, block handle, workspace/graph lease가 닫히는지 확인한다. stale config generation이 새 plan에 commit되지 않아야 한다.

마지막 5분에는 승격 또는 rollback을 판정한다. 승격 문서에는 최대 lhs/rhs margin, observed estimator error, threshold fixtures와 terminal gap을 남긴다. rollback이면 admission fence부터 세우고 inflight generation을 drain한다. restart로 metric을 0으로 만드는 대신 allocator와 request owners를 reconciliation한다.

**잘못된 최적화 제안을 거르는 질문.**

“KV cache를 늘리자”는 제안에는 현재 first false가 정말 B인지 묻는다. workspace/graph가 false면 cache pool 증가는 오히려 headroom을 줄인다. “token limit을 낮추자”에는 OOM plan의 combined shape가 어떤 threshold를 넘었는지 묻는다. 운 좋게 bucket을 피하는 값은 workload가 바뀌면 다시 실패한다.

“메모리가 남으니 sequence를 늘리자”에는 logits·sampling·block-table과 concurrent stream workspace를 포함했는지 묻는다. total free가 아니라 largest required allocation과 guard 뒤 envelope를 본다. “graph를 끄자”에는 eager workspace와 latency, pool 반환 semantics를 비교한다. graph reserve를 없애도 eager peak가 더 클 수 있다.

“OOM이면 retry하면 된다”에는 partial commit을 묻는다. blocks와 sequence state가 남고 output cursor가 일부 진행됐다면 blind retry는 duplicate work와 leak을 만든다. retry identity, old generation terminal과 new reservation을 분리한다. 같은 batch 이웃의 progress도 보존해야 한다.

“estimate를 크게 잡자”에는 false reject 비용을 묻는다. safety guard는 estimator error의 positive tail과 실패 비용으로 정한다. 최대 관측값 하나를 영구 margin으로 두면 rare outlier가 전체 capacity를 잠근다. backend/shape별 error distribution과 confidence를 사용한다.

이 질문을 통과한 변경만 canary에 간다. 좋은 capacity tuning은 knob의 방향이 아니라 predicate의 어떤 항, owner와 lifetime을 바꾸는지 설명한다. 수치 fixture가 있고 source mutation chain이 있으며 실패·cancel·rollback terminal까지 닫혀야 한다. 그때 admission은 추측이 아니라 검증 가능한 자원 transaction이 된다.

**마지막 수치 검산.**

최종 candidate set의 token127, sequence32, KV 증가 0.781GiB, workspace 증가 1.9GiB, graph 증가 1GiB를 다시 더하면 3.681GiB다. headroom3.5GiB와 비교해 181MiB 초과다. 후보 하나를 defer해 workspace와 bucket이 낮아지면 식이 true가 되는지 재계산한다. 단순히 request count를 하나 줄였다고 통과로 표시하지 않는다.

그다음 실제 runner acknowledgment에서 selected bucket, allocated block handles와 observed peak를 대조한다. estimate가 true였는데 actual이 false면 selector/estimator parity 문제다. estimate부터 false인데 commit됐다면 admission control 문제다. estimate와 actual은 true인데 allocator가 실패하면 fragmentation, untracked owner 또는 capacity snapshot을 조사한다. 이 세 branch를 섞지 않는다.

terminal 검산은 accepted, deferred, rejected, cancelled 모두에 적용한다. accepted는 committed handles가 실행 뒤 request lifetime으로 이전된다. deferred는 provisional이 0이고 다음 generation에서 새로 계산된다. rejected는 state mutation이 없다. cancelled는 commit boundary에 따라 runner abort와 resource release를 완료한다.

독자가 이 네 행을 자신의 trace로 채우고 source span을 붙일 수 있다면 장의 목적은 달성됐다. 어떤 option을 움직여야 할지뿐 아니라 왜 그 option이 해당 predicate와 physical allocation을 바꾸는지, 변경 실패 시 무엇을 되돌려야 하는지 설명할 수 있다. 다음 상태 기계는 이 검산된 transaction을 화살표의 전제 조건으로 사용한다.

운영 문서에는 확정된 source 사실, workload에 의존하는 조건부 결론, 실행으로 확인하지 않은 가설을 구분한다. static inventory와 selector branch는 고정 revision에서 확인할 수 있지만 실제 peak 분포와 fragmentation은 trace가 필요하다. 이 구분을 지켜야 다음 담당자가 이미 닫힌 코드를 반복해서 읽지 않고 미검증 위험부터 측정한다.

모든 측정은 동일 model·config·plan generation에 귀속하고, 시간 창이 다른 gauge와 event total을 직접 빼지 않는다. 최종 dossier에는 재현 조건과 중단 조건도 남긴다.
