# 45장. 전체 score matrix를 쓰지 않고도 정확히 같은 attention을 계산하는 법

표준 attention 식은 짧다. `S=QKᵀ·scale+mask`, `P=softmax(S)`, `O=PV`다. 이 식을 그대로 종이에 펼치면 query length와 KV length의 곱만큼 score가 생긴다. 프롬프트 128 tokens라면 한 head에서 128×128 scores이고, 긴 decode context에서는 query 한 row가 8,192개의 scores를 만든다. 모든 scores를 HBM에 썼다가 softmax를 위해 다시 읽고 probabilities를 다시 썼다가 V 곱을 위해 읽는 구현은 arithmetic보다 memory traffic에 큰 비용을 낼 수 있다.

FlashAttention의 핵심은 dense attention의 답을 근사하거나 일부 tokens를 버리는 것이 아니다. Q와 K/V를 tiles로 가져와 score tile을 빠른 on-chip storage에서 만들고, running softmax state를 갱신하며, probability tile을 V와 곱한다. 전체 score/probability matrix를 HBM에 materialize하지 않아도 최종 `softmax(S)V`와 같은 값을 얻을 수 있다. 순서를 바꾸는 대신 수학적 결합법칙과 numerically stable max subtraction를 사용한다.

그러나 “O(N²) 메모리를 O(N)으로 줄인다”는 구호만으로 kernel을 읽을 수 없다. 어떤 intermediate가 HBM에 쓰이지 않는지, running max `m`, exponential sum `l`, weighted accumulator `o`가 tile이 바뀔 때 어떻게 rescale되는지 알아야 한다. split-KV에서는 여러 CTAs가 서로 다른 KV 구간의 partial O와 log-sum-exp를 만들고 별 merge가 정확한 global softmax를 복원한다. empty split의 identity도 정확해야 한다.

동일한 작은 tile fixture가 이 수학과 구현 경계를 끝까지 연결한다.

이 장은 score `[1,2]`와 `[3,0]`을 두 tiles로 나누고 scalar V를 붙여 full softmax와 online merge를 손으로 비교한다. 그 수학을 vLLM v0.27.1이 고정한 FlashAttention fork commit `28e862d21806bc3580207aa0ad4e2759151e9827`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, FlashInfer v0.6.17 snapshot commit `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`의 실제 함수 경로와 연결한다. 실행하거나 성능 수치를 만들지 않는다.

13장은 causal·prefix·window가 **어떤 score 좌표를 허용하는지**와 stable softmax의 정답을 소유한다. 이 장은 그 정답을 oracle로 받아, tile을 순회하는 동안 running state와 HBM traffic, split merge 및 kernel metadata가 의미를 보존하는지를 소유한다. 같은 softmax 식이 다시 나오는 이유는 수학을 반복하기 위해서가 아니라 kernel representation을 검산하기 위해서다.

## 45.1 전체 score matrix가 없어도 답이 같다는 문제

### 45.1.1 표준식에서 실제로 필요한 것은 무엇인가

한 query row `q`와 keys `k_i`, values `v_i`가 있다고 하자. score는 `s_i=q·k_i·scale+mask_i`이고 output은 `Σ exp(s_i)v_i / Σ exp(s_i)`다. 최종 output에 필요한 것은 모든 `s_i` 배열 자체가 아니라 안정화된 exponential의 합과 value-weighted 합이다. max를 알고 있다면 두 합으로 정규화 결과를 만들 수 있다.

full implementation는 scores를 먼저 모두 만들고 row max를 구하고 exponential와 sum을 계산한 뒤 P와 V를 곱을 수 있다. 이 흐름은 이해하기 쉽지만 score와 P intermediate가 HBM을 왕복할 수 있다. tiled implementation는 K/V 구간을 순회하며 필요한 state만 유지한다. score tile은 registers/shared memory에서 소비되고 다음 tile로 넘어가기 전에 running state에 흡수된다.

이 변화가 FLOPs를 자동으로 줄인다고 쓰지 않는다. QK와 PV의 핵심 matmul은 여전히 필요하고 backward에서는 recomputation를 선택할 수 있다. 이 장의 serving forward 관점에서 중요한 것은 intermediate materialization와 HBM traffic, tile work partition다.

### 45.1.2 FlashAttention은 sparse attention이 아니다

causal mask나 sliding window가 허용한 범위 안에서 FlashAttention은 dense attention의 exact algorithm이다. 수치 reduction order와 exponential implementation 때문에 floating-point 결과가 bitwise 같지 않을 수 있지만 algorithm가 일부 scores를 근사해 생기는 차이는 아니다. sparse attention은 애초에 계산할 edges를 줄이는 별 방법이다.

mask는 score를 softmax domain에서 제외한다. causal row가 미래 columns를 보지 못하게 하고 window는 범위 밖 positions를 제외한다. tile가 masked 영역과 겹치면 per-element predicate나 tile-level skip가 필요하다. 완전히 masked row/split의 identity를 잘못 잡으면 NaN 또는 다른 request state가 섞일 수 있다.

softcap가 있으면 score transformation 순서도 중요하다. scale, optional softcap, mask가 어떤 순서와 수치 base로 적용되는지 source를 읽는다. API flag가 같다고 library별 내부 상수 folding와 exp2 사용이 같다고 가정하지 않는다.

### 45.1.3 IO-aware라는 말을 byte 생애로 번역한다

Q tile은 HBM에서 on-chip storage로 들어오고 여러 K tiles와 재사용될 수 있다. K/V tiles도 HBM 또는 paged KV cache에서 읽혀 shared memory/register pipeline을 거친다. QK score tile은 MMA accumulator에 만들어지고 mask/softmax가 적용된다. P tile은 V와 곱해 output accumulator를 갱신한다.

전체 S나 P를 HBM에 쓰지 않는 것이 핵심이다. 대신 final O, optional LSE와 split가 있으면 partial O/LSE workspace를 쓸 수 있다. “intermediate를 쓰지 않는다”가 아니라 “전체 quadratic score/probability matrix를 materialize하지 않는다”고 정확히 말한다.

tile가 작으면 on-chip capacity에 맞지만 더 많은 loop와 boundary 처리가 생길 수 있다. tile가 크면 reuse가 좋아질 수 있으나 registers/shared memory와 residency를 제한한다. selected BlockM/BlockN은 dtype, head dimension, architecture, causal와 sequence shape에 종속된다.

## 45.2 online softmax를 손으로 계산해 충분한 상태를 얻는다

### 45.2.1 full softmax 기준값

scores를 `[1,2,3,0]`, scalar values를 `[10,20,30,40]`이라고 하자. 안정화를 위해 global max 3을 뺀다. weights는 `[e^-2,e^-1,1,e^-3]`다. 분모는 `L=e^-2+e^-1+1+e^-3`, 분자는 `O_num=10e^-2+20e^-1+30+40e^-3`다. 최종 output은 `O_num/L`, LSE는 `3+log(L)`이다.

근삿값으로 `e^-2≈0.135335`, `e^-1≈0.367879`, `e^-3≈0.049787`다. `L≈1.553001`이고 분자는 `1.35335+7.35758+30+1.99148≈40.70241`다. output은 약 `26.2089`, LSE는 `3+log(1.553001)≈3.44019`다. 이 값은 online 계산의 reference다.

### 45.2.2 첫 tile [1,2]

첫 tile local max `m1=2`다. 안정화 weights는 `[e^-1,1]`이고 local sum `l1=e^-1+1≈1.367879`다. weighted value accumulator는 정규화 전 `o1=10e^-1+20≈23.67879`다. local normalized output만 보면 `o1/l1≈17.3106`이지만 다음 tile과 합치려면 `m1,l1,o1` 또는 equivalent LSE/output state를 보존한다.

왜 이미 정규화한 output 하나만으로 부족한가. 두 번째 tile의 scores가 더 크면 첫 tile weights 전체를 새 global max 기준으로 줄여야 한다. local output와 local LSE를 함께 가지면 rescale할 수 있지만 output만 가지면 상대 mass를 잃는다.

### 45.2.3 둘째 tile [3,0]과 online update

둘째 tile local max `m2=3`, weights는 `[1,e^-3]`, `l2=1+e^-3≈1.049787`, `o2=30+40e^-3≈31.99148`다. global max는 `m=max(2,3)=3`이다.

첫 tile state는 새 max 기준으로 `exp(m1-m)=e^-1`을 곱해 rescale한다. 둘째는 `exp(m2-m)=1`이다. 따라서 `l=e^-1·l1+l2`. 수치로 `0.367879×1.367879+1.049787≈1.553001`이다. `o=e^-1·o1+o2≈0.367879×23.67879+31.99148≈40.70241`다.

최종 `o/l≈26.2089`, `m+log(l)≈3.44019`로 full 기준값과 같다. 전체 score vector를 동시에 저장하지 않고 두 tile state만 결합했다. tile 순서를 바꿔도 floating-point rounding 차이는 있을 수 있지만 같은 결합식을 적용한다.

### 45.2.4 streaming recurrence로 쓴다

running state `(m_old,l_old,o_old)`와 새 score tile의 local `(m_t,l_t,o_t)`가 있을 때 `m_new=max(m_old,m_t)`다. `alpha=exp(m_old-m_new)`, `beta=exp(m_t-m_new)`라 두면 `l_new=alpha·l_old+beta·l_t`, `o_new=alpha·o_old+beta·o_t`다.

kernel은 local tile을 별로 완성한 뒤 merge할 수도 있고 score fragments를 순차로 읽으며 running max/sum를 갱신할 수도 있다. 중요한 invariant는 max가 커질 때 이전 sum와 output accumulator가 같은 alpha로 rescale된다는 것이다. l만 rescale하고 o를 그대로 두면 normalization 후 value weights가 틀어진다.

### 45.2.5 exp2와 scale folding

GPU kernel은 자연 exponential 대신 base-2 exponential를 사용할 수 있다. `exp(x)=2^(x·log2(e))`이므로 score scale에 `log2(e)`를 fold하고 `exp2`로 같은 수학을 구현할 수 있다. LSE를 natural-log API로 반환한다면 내부 base-2 state를 적절히 변환해야 한다.

장애 조사에서 local LSE base를 혼동하면 split merge weighting가 틀린다. 한 partial은 natural LSE이고 combine은 base-2 차이로 해석하면 `exp(local-global)` mass가 잘못된다. source에서 scale-softcap-mask-exp와 final LSE conversion 순서를 읽는다.

### 45.2.6 빈 tile과 완전히 masked row

모든 elements가 mask된 tile은 valid softmax mass가 0이다. local max를 임의의 finite 0으로 두고 l=0을 다루는 구현도 가능하지만 merge representation가 명확해야 한다. split state로 LSE를 쓴다면 empty identity는 `LSE=-∞`, normalized/partial output는 zero다.

global merge에서 weight는 `exp(LSE_j-LSE_global)`다. empty split의 `LSE_j=-∞`이면 weight 0이어서 output에 기여하지 않는다. 이를 0으로 두면 global scores가 음수인 경우 empty split이 가짜 mass `exp(0)`를 만들 수 있다. identity는 단순 초기값이 아니라 correctness 계약이다.

## 45.3 paged metadata가 tile math 앞의 주소를 정한다

### 45.3.1 QK→mask→softmax→PV의 순서

한 query tile과 key tile을 MMA해 score fragments를 만든다. scale와 optional softcap을 적용하고 causal/window/page tail mask를 반영한다. 그 뒤 row max와 exponential sum를 갱신하고 probabilities 형태를 V tile과 곱해 output accumulator에 더한다. 다음 K/V tile로 넘어갈 때 running max가 바뀌면 이전 accumulator를 rescale한다.

mask를 exponential 뒤 잘못 적용하면 denominator나 NaN behavior가 달라질 수 있다. softcap을 scale 전/후 어느 위치에 적용하는지도 API 정의와 source를 따른다. QK와 PV가 tensor-core fragments를 쓰더라도 reductions와 conversions가 어떤 dtype인지 확인한다.

### 45.3.2 registers·shared memory·HBM의 역할

HBM에는 Q/K/V 원본 또는 KV cache pages와 final O가 있다. shared memory는 global loads를 tiles로 stage하고 thread/warp groups 사이 data exchange를 돕는다. registers는 MMA accumulators, row max/sum와 output fragments를 가진다. architecture-specific TMA나 async pipeline은 data movement와 compute를 overlap할 수 있다.

이 설명은 모든 kernel의 exact placement를 단정하지 않는다. selected traits와 architecture source를 읽는다. FlashAttention-3 Hopper path의 warp specialization와 WGMMA/TMA는 FA2 generic path와 다르며 FlashInfer Blackwell/prefill/decode paths도 다르다.

### 45.3.3 paged KV는 tile math 앞의 주소 번역을 추가한다

serving KV cache는 logical position가 contiguous K/V array가 아니라 block table와 page offset을 통해 physical blocks로 갈 수 있다. tile loader는 logical KV range를 page 단위 physical address로 번역하고 page boundary/tail을 처리한다. math가 맞아도 block table나 last-page length가 틀리면 wrong K/V를 읽는다.

page boundary 오답은 online softmax bug처럼 보일 수 있다. first bad position가 page size 경계인지, score/LSE가 physical load부터 달라지는지 본다. contiguous reference path와 paged path를 같은 Q/K/V logical contents로 비교하면 address와 math를 나눌 수 있다.

### 45.3.4 prefill Q=128과 decode Q=1은 병렬 축이 다르다

prefill `Q=128,KV=128`에서 BlockM=64를 교육용으로 가정하면 head당 query tiles가 2다. batch와 heads가 더해지면 많은 independent query tiles가 생긴다. causal mask는 각 query row가 보는 KV 범위를 다르게 하지만 Q axis의 parallel supply가 있다.

이 fixture를 tile 좌표로 펼치자. query tile 0은 rows 0–63, tile 1은 rows 64–127을 후보로 가진다. KV tile 0은 columns 0–63, tile 1은 64–127이다. non-causal이면 네 Q×KV tile combinations가 모두 유효하다. causal이면 query tile 0과 KV tile 1의 상당 부분 또는 전체가 future여서 skip/mask될 수 있고 diagonal tiles는 row별 predicate가 필요하다.

query CTA 하나가 KV tiles를 내부 loop로 순회하는 설계라면 launched CTA count와 conceptual Q×KV tile count는 같지 않다. CTA는 query tile/head를 소유하고 두 K/V tiles를 차례로 흡수할 수 있다. split-KV에서는 KV tile intervals가 여러 CTAs로 나뉘어 launched work가 늘어난다. source launcher와 device scheduler를 읽는다.

decode `Q=1,KV=8192`에서는 query tile 후보 64 rows 중 한 row만 valid일 수 있다. 그러나 KV tiles가 128개라면 그 valid row를 위해 긴 loop가 있다. row capacity가 비었다는 사실과 memory/compute work가 작다는 결론은 다르다. 한 row의 D-dimensional Q가 모든 valid K/V positions와 상호작용한다.

batch가 32이고 heads가 32라면 Q=1이어도 batch×heads 조합이 independent work를 제공할 수 있다. GQA에서 query heads 32가 KV heads 8이라면 KV loads를 query head groups가 공유하거나 mapping할 수 있다. grid가 query heads인지 KV heads인지, CTA가 몇 query heads를 묶는지는 specialization를 따른다.

splits=4는 각 split가 KV 2,048 positions 정도를 담당하는 교육 partition를 만들 수 있다. page/tile alignment 때문에 exact intervals는 다를 수 있다. 네 partial CTAs가 각각 local O/LSE를 쓰고 combine가 네 states를 읽는다. 긴 loop는 줄지만 partial workspace와 merge traffic가 생긴다.

splits를 128로 해 KV tile 하나당 CTA를 만든다고 무조건 빠르지 않다. CTAs가 너무 작은 work를 갖고 launch/scheduling overhead가 커지며 combine가 128 states를 읽는다. planner는 available parallel supply와 interval cost, workspace를 함께 고려해야 한다. exact heuristic는 source fact로만 설명한다.

persistent scheduling는 static split와 또 다르다. fixed worker CTAs가 plan work items를 반복해서 가져가면 grid가 logical query×head×split count와 같지 않다. load imbalance를 줄일 수 있지만 work queue와 terminal protocol가 필요하다. 40장에서 배운 grid 해석 규칙을 적용한다.

prefill 128/128과 decode 1/8192의 raw score element 수는 둘 다 각각 16,384와 8,192로 같은 order처럼 보일 수 있지만 work shape가 다르다. prefill에는 많은 query rows와 causal structure, decode에는 한 row의 긴 reduction가 있다. matrix tile efficiency, parallel axes와 memory reuse가 달라진다.

KV cache가 paged이면 decode 8,192 positions가 여러 physical blocks에 흩어질 수 있다. block table loads와 page boundaries가 tile loop에 추가된다. prefill newly written K/V가 contiguous staging 또는 pages로 들어가는 방식도 backend에 따라 다르다. Q/KV lengths만 같아도 layout가 performance/correctness를 바꾼다.

window size가 4,096이면 decode logical context 8,192 중 attention-valid range가 절반일 수 있다. planner와 kernel가 effective KV range를 줄이는지 mask만 하는지 확인한다. declared KV length와 actual attended length를 나눈다. causal decode는 미래가 없지만 left window boundary가 있다.

softcap와 position encoding feature도 selected kernel/key를 바꿀 수 있다. RoPE는 보통 Q/K에 앞서 적용되지만 attention backend가 positional feature를 직접 처리하는 경로가 있을 수 있다. JIT key와 params에서 확인한다. 이 장은 position 수학을 다시 설명하지 않고 score input identity만 지킨다.

prefill/decode 비교의 결론은 하나의 FLOP 숫자가 아니다. independent CTA supply, CTA 내부 KV loop length, on-chip reuse, page access, split workspace/merge와 launch count를 별 원장에 쓴다. 실제 병목은 profiler/measurement로 판정해야 한다.

decode `Q=1,KV=8192`에서는 같은 BlockM이면 query tile 하나가 대부분 비어 있다. 그러나 한 row가 긴 KV를 읽으므로 work가 없는 것은 아니다. batch×heads를 늘리거나 KV를 splits로 나누어 여러 CTAs가 partial states를 만들고 merge할 수 있다. persistent scheduling도 work balance에 쓰일 수 있다.

CTA 손계산은 selected axes를 명시한다. static unsplit라면 `ceil(Q/BlockM)×batch×heads` 같은 형태가 출발점이다. split-KV면 여기에 split count가 곱해지고 merge work가 추가된다. GQA에서는 KV heads와 query heads mapping를 확인한다. exact grid는 launcher source가 정한다.

## 45.4 같은 수학을 backend별 tile 상태로 비교한다

### 45.4.1 Python varlen call까지

vLLM backend는 query `[num_tokens,num_heads,head_size]`, KV cache shape와 attention metadata를 받아 native varlen function으로 보낸다. cumulative query/KV lengths, maximum lengths, block table, causal/window와 softcap/scale가 kernel dispatch state에 들어간다. `num_tokens`만으로 selected kernel을 결정하지 않는다.

prefill과 decode가 동일 Python entry를 공유해도 query length distribution, KV pages와 split predicate가 다를 수 있다. backend version와 device capability, dtype/head size가 current FlashAttention fork의 지원 path를 고른다.

### 45.4.2 QK MMA에서 online rescale까지

current vLLM FlashAttention pin의 forward kernels는 tile traits에 따라 QK MMA를 수행하고 score fragments에 scale/softcap/mask를 적용한다. softmax object/state가 row max와 sum를 갱신하며 이전 output accumulator를 rescale할 factor를 만든다. P fragments로 변환된 weights가 V와 MMA되어 output accumulator가 갱신된다.

읽을 때 함수 이름을 나열하지 않고 state를 따라간다. Q/K tile pointers와 coordinates, score accumulator, softmax running state, rescale factor, output accumulator, epilogue normalization/LSE store의 owner를 찾는다. row tail와 causal mask predicate가 state update 전에 invalid scores를 제외하는지도 본다.

### 45.4.3 finalization는 나눗셈과 LSE를 닫는다

K/V tiles 순회가 끝나면 output accumulator를 running sum로 normalize한다. optional LSE는 max와 log(sum)를 API가 요구하는 base/scale로 저장한다. 완전히 masked row의 output/LSE sentinel behavior도 확인한다.

final normalization가 두 번 적용되거나 l이 zero인데 나누면 NaN이 난다. accumulator가 이미 normalized인지 unnormalized인지 kernel 단계마다 구분한다. split workspace에는 partial normalized O와 LSE를 저장하는지 unnormalized o/l/m을 저장하는지 source contract를 따른다.

### 45.4.4 split workspace와 empty identity

`num_splits>1`이면 각 split CTA가 서로 다른 KV interval를 처리해 partial output와 LSE를 workspace에 쓸 수 있다. workspace layout는 batch/query/head/split와 padded head dimension strides를 갖는다. combine kernel이 같은 logical row/head의 partials를 정확히 모아야 한다.

KV가 짧거나 split range가 실제 sequence 밖이면 empty split가 생길 수 있다. current pin의 empty split identity와 combine source를 확인한다. zero output와 `-∞` LSE가 아니거나 stride가 틀리면 특정 split 수와 tail에서만 오답이 난다.

이 source walk를 실제 한 row의 시간 순서로 다시 읽자. Python metadata가 row R의 query interval와 KV length를 정의한다. dispatcher가 current device와 dtype/head dimension에 맞는 forward specialization를 고른다. launcher가 row/head/split axes와 workspace를 정한다. kernel은 block table를 통해 첫 K/V tile을 읽고 QK scores를 만든다. softmax state가 첫 `(m,l,o)`를 만들고 다음 tiles에서 rescale한다. epilogue가 unsplit이면 final O/LSE를, split이면 partial O/LSE를 쓴다. combine가 있으면 마지막에 global O/LSE를 만든다.

이 순서를 쓰면 함수 사이 빈 경계가 드러난다. Python call에 `num_splits`가 명시되는지 native heuristic가 정하는지, workspace allocation를 framework가 하는지 extension가 하는지, combine launch가 forward launcher 안인지 별 binding인지 확인한다. 이름이 `flash_attn_varlen_func` 하나라고 내부 launch도 하나라고 가정하지 않는다.

current fork commit를 고정하는 이유도 여기에 있다. kernel templates와 line layout, split behavior는 fork revision에 따라 바뀔 수 있다. 과거 commit의 함수 이름이 같아도 empty identity나 combine implementation를 현재 코드의 사실로 승계하지 않는다. 독자는 소스 노트의 40자리 revision와 line range에서 현재 경로를 다시 읽는다.

QK/PV order를 볼 때 accumulator dtype도 함께 적는다. input가 fp16/bf16이어도 row max와 sum 또는 MMA accumulator가 더 넓은 dtype을 쓸 수 있다. P conversion가 낮은 precision으로 내려가는 지점과 output accumulator/final cast를 구분한다. numerical 차이 조사에서 “모두 bf16 kernel”처럼 하나로 뭉개면 first rounding 위치를 찾지 못한다.

softmax rescale factor의 적용 대상도 확인한다. running max가 바뀌면 old exponential sum뿐 아니라 already accumulated PV output가 old normalization 기준을 갖고 있으므로 같은 alpha로 rescale돼야 한다. source에서 factor가 accumulator fragments에 곱해지는 위치를 찾는다. l update만 맞고 output rescale가 빠진 bug는 score/LSE는 맞는데 O만 틀리는 특징을 가질 수 있다.

final LSE store는 output가 필요 없는 최적화에서도 별 optional path일 수 있다. sampling forward가 LSE를 요청하지 않으면 store를 생략할 수 있고 backward/training용 path는 요구가 다를 수 있다. 이 권에서는 serving forward를 범위로 하되 split merge가 LSE를 내부 state로 필요로 하는 경우와 user-visible return-LSE를 구분한다.

causal tile skip도 identity와 연결된다. future-only K tile을 완전히 건너뛰면 running state가 변하지 않아야 한다. 부분 overlap tile은 invalid elements를 mask하고 valid elements만 mass에 넣는다. row마다 valid frontier가 다르므로 tile-level predicate와 element predicate가 함께 있을 수 있다.

sliding window에서는 left/right bounds가 logical absolute position와 맞아야 한다. paged physical order와 logical causal/window position를 혼동하면 주소는 valid하지만 wrong tokens가 softmax mass에 들어간다. page loader, mask coordinate와 score checkpoint를 이어서 본다.

## 45.5 split-LSE merge도 같은 softmax다

### 45.5.1 partial normalized outputs를 합치는 식

split j가 자신의 KV interval에서 normalized output `O_j`와 `LSE_j=log Σ_i∈j exp(s_i)`를 만들었다고 하자. global LSE는 `LSE=log Σ_j exp(LSE_j)`다. 안정적으로 `M=max_j LSE_j`, `Z=Σ_j exp(LSE_j-M)`, `LSE=M+log Z`로 계산한다.

global output은 `Σ_j w_j O_j`, `w_j=exp(LSE_j-LSE)`다. 각 partial output가 local probability로 normalize돼 있으므로 local mass가 global mass에서 차지하는 비율로 다시 가중한다. 단순 평균하거나 split token count로 가중하면 scores 분포를 잃는다.

### 45.5.2 두 score tiles로 split merge를 손검산한다

앞 fixture의 split 1 `[1,2]`에서 `l1≈1.367879`, `m1=2`이므로 `LSE1=2+log(1.367879)≈2.313262`다. local output `O1≈17.310586`이다. split 2 `[3,0]`은 `LSE2=3+log(1.049787)≈3.048587`, `O2≈30.474259`다.

global max of LSE는 `M=3.048587`이다. weights의 unnormalized form은 `a1=exp(2.313262-3.048587)≈0.479350`, `a2=1`이다. `Z≈1.479350`, normalized weights는 약 `w1≈0.323993`, `w2≈0.676007`이다.

merge output는 `0.323993×17.310586+0.676007×30.474259≈26.2089`다. global LSE는 `3.048587+log(1.479350)≈3.44019`다. full softmax와 online unnormalized accumulator 계산의 결과와 일치한다. rounding 때문에 마지막 자리는 달라질 수 있다.

### 45.5.3 empty split identity를 숫자로 확인한다

셋째 split가 완전히 비어 있다고 하자. `O3=0`, `LSE3=-∞`이면 `exp(LSE3-M)=0`이어서 Z와 output에 아무 영향이 없다. merge 결과는 두 split 때와 같다.

잘못해 `LSE3=0`을 저장하면 unnormalized mass `exp(0-3.048587)≈0.04745`가 생긴다. Z가 커지고 기존 partial weights가 줄어 output가 변한다. O3가 zero여도 denominator mass가 추가돼 output가 작아진다. empty output zero만으로 identity가 완성되지 않는다.

### 45.5.4 partial layout는 수학만큼 중요하다

수식이 맞아도 workspace stride가 틀리면 split A의 LSE와 split B의 output를 섞는다. layout가 `[batch,head,split,query,padded_D]`인지 다른 순서인지 producer store와 combine load가 합의해야 한다. padded head dimension를 D로 착각하면 다음 row/head의 partial을 읽을 수 있다.

variable number of splits는 indptr로 각 row의 partial interval를 표현할 수 있다. empty interval, last indptr와 total partial count를 확인한다. fixed maximum workspace 안 actual splits만 valid인지 sentinel/LSE identity로 padding하는지 source를 따른다.

### 45.5.5 split-KV와 tensor parallel를 구분한다

split-KV는 한 device/kernel invocation 영역에서 한 attention row의 KV sequence를 여러 work items로 나누고 partial states를 merge하는 방법이다. tensor parallel은 model heads/weights/tensors를 devices/ranks에 분할하고 collective를 요구할 수 있다. 둘 다 “split”이라 불러도 ownership와 merge primitive가 다르다.

split-KV count를 늘려 CTA supply를 만들 수 있지만 partial workspace, merge launch와 reads/writes가 늘어난다. decode Q=1, long KV에서 유용할 가능성이 있지만 batch×heads가 이미 충분하거나 KV가 짧으면 overhead가 클 수 있다. heuristic와 actual selected count를 본다.

## 45.6 SGLang에서 FlashInfer plan·run·merge까지

### 45.6.1 SGLang adapter는 request metadata를 wrapper 언어로 바꾼다

SGLang attention backend는 extend/prefill과 decode path에서 query, KV cache, block/page indices, sequence lengths와 head/dtype 정보를 FlashInfer wrapper에 전달한다. wrapper choice가 phase와 backend conditions에 따라 다르므로 prefill과 decode를 동일 kernel로 뭉개지 않는다.

MLA backend의 plan 경계는 batch indptr, KV indptr/indices, lengths, heads와 dimensions를 준비한다. 이 state가 host planning의 입력이다. logical request metadata가 잘못되면 CUDA tile math 전에 page/split work가 틀어진다.

#### 45.6.1.1 plan은 kernel 실행이 아니다

FlashInfer Python `_core.py`의 plan은 dtype/head constraints를 검증하고 host indptr를 다루며 native planning module을 호출한다. plan은 padded work items, CTA policy, workspace offsets와 dispatch에 필요한 vector를 준비할 수 있지만 attention output를 계산하는 kernel run 자체가 아니다.

plan/run 분리는 같은 request shape에서 plan를 reuse할 수 있는 가능성과 metadata lifetime 책임을 만든다. plan key가 head dimension, dtype, mask/position feature와 맞아야 한다. stale plan vector를 다른 KV layout에 쓰면 wrong work items를 launch할 수 있다.

host indptr copy/synchronization가 있다면 그것이 어느 metadata readiness를 보장하는지 본다. 계획 완료와 device run completion는 다른 사건이다. workspace는 plan이 계산한 offsets와 run이 사용하는 allocation capacity가 맞아야 한다.

#### 45.6.1.2 native run은 plan를 pointer와 launch state로 복원한다

run은 Python tensors와 plan state를 native module에 전달한다. native dispatch는 dtype, head dimension, mask mode, positional encoding와 architecture에 맞는 specialization를 고른다. JIT key에 이 features가 들어가는 것은 generated kernel의 type/layout/control path가 달라질 수 있기 때문이다.

same Q/KV lengths라도 dtype 또는 position mode가 바뀌면 다른 module/key를 쓸 수 있다. cache key가 feature 하나를 누락하면 incompatible kernel를 재사용할 위험이 있고 지나치게 세분하면 compile/cache churn이 늘 수 있다. source key fields와 selected module을 관측한다.

#### 45.6.1.3 paged prefill launcher에서 grid를 읽는다

FlashInfer `prefill.cuh`는 thread/block indices, query/KV chunks와 head axes를 소비하고 paged KV parameters로 kernel을 launch한다. selected CTA policy와 tile traits가 grid/block/shared memory를 정한다. plan work item count와 launch grid를 연결하되 persistent scheduler면 단순 일대일 매핑을 가정하지 않는다.

`tmp_v != nullptr`인 path는 partial output workspace가 존재하는 split-KV 실행임을 나타낼 수 있다. attention kernel 뒤 `VariableLengthMergeStates`가 호출돼 partial V/output states와 LSE를 variable intervals에 따라 merge한다. tmp pointer 유무, split indptr와 merge output/LSE pointers를 기록한다.

#### 45.6.1.4 decode tensor-core path가 batch prefill module을 재사용할 수 있다

decode라고 항상 별 decode kernel만 쓰는 것은 아니다. head dimension, batch/heads, KV layout와 CTA policy 조건에 따라 tensor-core decode가 batch prefill module을 재사용할 수 있다. phase 이름에서 kernel family를 추측하지 않고 selected module과 plan/run path를 본다.

이 재사용은 Q=1을 prefill처럼 가장한다는 의미가 아니다. batch와 heads를 묶어 tensor-core-friendly work를 만들거나 split 정책을 사용할 수 있다. effective query lengths와 work items가 module contract에 맞게 구성된다.

#### 45.6.1.5 FlashInfer의 source walk를 상태 원장으로 닫는다

첫 행은 SGLang request metadata: active requests, Q/KV indptr, page table, heads/D/dtype/mask/position다. 둘째는 FlashInfer plan key와 work items: CTA policy, split intervals, workspace offsets다. 셋째는 native dispatch module과 launcher geometry다. 넷째는 attention kernel partial O/LSE writes다. 다섯째는 optional merge indptr와 final O/LSE다.

first divergence가 plan work item이면 host metadata/planner, launcher geometry면 dispatch/traits, partial state면 page load/QK/softmax/PV, merge 후만 틀리면 layout/indptr/LSE base를 본다. final token에서 거꾸로 추측하지 않는다.

#### 45.6.1.6 plan reuse가 안전하려면 무엇이 같아야 하는가

plan를 reuse한다는 것은 Python 객체를 재사용한다는 뜻보다 강한 계약이다. work item가 참조하는 batch indptr 구조, head mapping, page layout와 maximum capacities가 current run에 맞아야 한다. actual KV lengths처럼 run마다 변하는 값이 plan vector에 baked되는지 별 runtime buffer에서 읽는지도 확인한다.

같은 batch size라도 per-request KV distribution가 크게 다르면 load-balancing plan가 달라질 수 있다. plan key가 batch size만 쓰지 않을 수 있고, plan를 매번 만들거나 shape bucket별로 저장할 수 있다. source가 보장하지 않는 reuse를 performance 최적화라며 추가하면 wrong split intervals가 생긴다.

workspace offsets는 plan와 allocation가 함께 소유한다. plan A가 partial count 40을 예상했는데 run B가 allocation 32만 제공하면 out-of-bounds다. 반대로 항상 maximum workspace를 확보하면 safe할 수 있지만 memory pressure가 늘어난다. plan output의 required bytes와 actual buffer capacity를 기록한다.

host planning가 끝났다고 CUDA writes가 끝난 것도 아니다. plan vector/indptr가 device로 복사되는 path의 stream ordering를 43장 시간 계약으로 확인한다. run kernel가 same stream인지 event wait가 있는지 본다. stale plan는 math bug와 같은 final symptom을 낼 수 있다.

#### 45.6.1.7 planner가 만드는 parallel work와 kernel math를 분리한다

planner는 어떤 query/head/KV chunk를 어느 work item로 만들지 정한다. kernel은 주어진 interval에서 QK, online softmax와 PV를 수행한다. decode 저활용을 고치려 split count를 바꾸는 것은 planner/work partition 변화이고 online recurrence 식 자체를 바꾸는 것이 아니다.

load balancing가 잘못되면 어떤 CTA는 너무 긴 KV interval, 다른 CTA는 empty interval을 받을 수 있다. correctness는 empty identity와 merge가 지킬 수 있어도 performance imbalance가 남는다. 반대로 모든 intervals가 균등해 보여도 page locality나 head mapping가 달라 실제 cost가 같지 않을 수 있다.

padded work items도 실제 item와 구분한다. static launch capacity를 맞추려고 dummy work를 넣으면 kernel이 valid flag로 빠져나가거나 identity partial을 써야 한다. dummy가 uninitialized pointer를 읽거나 merge indptr에 valid로 포함되면 오답이 난다.

planner의 CTA policy는 device SM 수와 selected kernel resource에 영향을 받을 수 있다. 같은 logical shape가 다른 architecture에서 다른 split/persistent policy를 얻는 것은 가능하다. CUDA version 하나를 원인으로 쓰지 않고 actual planner inputs와 selected specialization를 비교한다.

#### 45.6.1.8 SGLang extend와 decode를 같은 request로 연결한다

request A가 128-token prefill을 마치고 첫 decode token으로 넘어간다고 하자. extend path는 Q=128과 KV append locations를 plan하고 paged cache에 K/V를 쓴다. decode path는 Q=1이지만 prefill에서 만든 physical pages와 new position를 읽는다. 두 phases가 같은 logical sequence와 block table generation를 공유해야 한다.

prefill output가 맞고 첫 decode만 틀리면 decode plan/head mapping, newly appended KV visibility나 position/window를 본다. prefill 마지막 rows부터 틀리면 extend page tail와 causal mask를 본다. 두 phase가 다른 wrappers/kernels를 쓰므로 “FlashInfer 결과” 하나로 합치지 않는다.

decode tensor-core가 batch prefill module을 재사용해도 metadata adapter는 decode semantics를 전달해야 한다. query length, causal position와 KV length가 올바른 module params로 변환되는지 확인한다. module 이름이 prefill이라고 query를 128로 pad했다고 단정하지 않는다.

SGLang의 wrapper selection log와 FlashInfer module/JIT key, plan work count를 correlation한다. Python backend만 보면 selected CUDA specialization를 놓치고 CUDA symbol만 보면 어느 request metadata가 들어왔는지 모른다. 두 trace를 step ID로 잇는다.

#### 45.6.1.9 JIT key 누락과 과분할의 양쪽 위험

dtype가 key에서 빠지면 bf16 inputs에 fp16 assumptions의 generated kernel를 재사용할 수 있다. head dimension가 빠지면 strides/tile traits가 다를 수 있다. mask/position feature가 빠지면 control flow와 coordinate transformation가 달라진다. key는 binary behavior를 결정하는 fields를 포함해야 한다.

반대로 actual KV length처럼 kernel이 runtime param으로 안전하게 처리하는 모든 값을 key에 넣으면 key cardinality와 compile/cache churn가 커질 수 있다. 어떤 값이 compile-time specialization이고 어떤 값이 runtime parameter인지 source template/binding에서 확인한다.

backend별 작은 차이에서 JIT key collision를 의심하려면 selected module identity가 서로 다른 configs에서 동일하게 나타나는지 본다. collision이 아니라 reduction order 차이라면 module key는 정상이고 partial states가 작은 tolerance 안에서 다를 수 있다.

cache eviction로 module가 재compile돼도 algorithm result는 tolerance 안에서 같아야 한다. compile latency와 correctness를 구분한다. stale binary/artifact mismatch는 44장의 compatibility 조사와 연결한다.

## 45.7 prefill·decode에서 병렬성과 비용을 읽는 법

### 45.7.1 prefill Q=128, KV=128

교육용 BlockM=64, BlockN=64, batch 1, heads H라고 하자. query tiles는 2, KV tiles는 각 query tile에서 2다. unsplit CTA supply가 head별 query tile이라면 2H CTAs가 출발점이다. 각 CTA 안에서 두 KV tiles를 online recurrence로 흡수한다.

causal attention에서는 query tile/row에 따라 유효 KV 범위가 다르다. upper triangle scores는 masked다. tile-level skip와 element mask가 work를 줄일 수 있으나 exact behavior는 kernel source에 달렸다. Q/KV가 모두 128이라고 full 128² 유효 scores라고 세지 않는다.

### 45.7.2 decode Q=1, KV=8192

같은 BlockM이면 query tiles는 1이다. BlockN=64라면 conceptual KV tiles는 128이다. 한 CTA가 모두 순회하면 Q axis parallel supply는 작지만 CTA 내부 work는 길다. batch×heads가 독립 CTAs를 제공하고 split-KV가 KV tiles를 여러 CTAs로 나눌 수 있다.

예를 들어 H=8, batch=1, splits=1이면 query/head 축으로 8 CTAs일 수 있다. splits=4라면 partial attention CTAs가 32이고 merge work가 추가된다. 실제 FlashInfer/FA launcher axes와 GQA mapping를 확인한다. 이 수치는 교육용 식이지 current kernel의 측정값이 아니다.

이 계산을 request distribution에 놓자. decode requests 여덟 개가 각각 Q=1이라면 batch 8, heads 8, splits 1의 단순식은 64 head-row work items를 줄 수 있다. 한 request만 남으면 8로 줄어든다. continuous batching가 kernel parallel supply를 만드는 이유가 shape 수준에서 보인다. scheduler가 request를 기다리게 하는 latency cost와 trade-off다.

KV lengths가 `[128,256,512,1024,2048,4096,8192,8192]`처럼 ragged하면 각 work item cost가 다르다. rectangular grid에서 longest length에 맞추면 짧은 sequences가 tail를 갖고, planner가 variable splits를 만들면 merge indptr가 ragged해진다. batch count만으로 balanced work를 가정하지 않는다.

긴 두 requests에 four splits, 짧은 여섯 requests에 one split을 준다면 total partial work는 `2×8 heads×4 + 6×8 heads×1=112` items라는 교육 계산이 된다. fixed split 4를 모두에 적용하면 256 items와 더 많은 empty/short partials가 생긴다. variable planning의 목적과 merge complexity가 함께 보인다.

query heads 32, KV heads 8인 GQA에서 head work를 query heads로 세면 batch 8에서 256 query-head rows다. KV loads를 groups가 공유하는지 CTA mapping가 KV head 중심인지에 따라 actual CTAs와 reuse가 다르다. config의 `num_heads` 하나로 grid를 계산하지 않는다.

decode tensor-core path가 batch prefill module을 재사용하면 work items가 여러 requests/heads를 tile에 묶을 수 있다. Q=1 rows가 batch 축에서 모여 matrix tile을 채울 수 있다. 이 경우 per-request BlockM underfill를 단순 합하지 않고 packed work layout를 본다.

split selection가 CUDA Graph capture key나 static workspace capacity와 연결될 수 있다. maximum splits에 맞춰 capture한 graph가 active splits를 predicate로 처리할 수 있고 shape가 capacity를 넘으면 fallback가 필요할 수 있다. graph temporal contract와 attention math를 별로 검증한다.

decode 저활용 조사에서 GPU utilization aggregate가 낮아도 CPU scheduler gap나 graph miss가 원인일 수 있다. attention kernel 사이 빈 시간이 많은지 kernel 내부 stalls인지 분리한다. 이 장은 kernel work axes를 주지만 전체 service timeline은 43장 원장을 함께 쓴다.

prefill이 decode보다 빠른 token/s처럼 보여도 denominator가 다르다. prefill은 한 launch에서 많은 query tokens를 처리하고 decode는 사용자 step latency가 중요하다. throughput와 ITL을 같은 숫자로 비교하지 않는다. kernel efficiency를 논하려면 processed QK pairs/bytes와 selected shapes를 기록한다.

KV length 8192가 긴 reduction를 만든다고 arithmetic intensity가 자동으로 높아지는 것도 아니다. Q 한 row가 K/V를 스트리밍하면서 data reuse가 제한될 수 있어 memory-bound가 될 수 있다. heads/groups와 cache layout가 reuse를 만든다. roofline와 byte 계산은 41장의 방법을 적용한다.

split가 memory-bound work를 여러 CTAs로 나눠도 total bytes가 줄지 않거나 partial writes 때문에 늘 수 있다. parallelism이 latency를 줄일 가능성과 bandwidth contention를 함께 본다. 논문이나 benchmark의 최적 split를 현재 workload에 그대로 적용하지 않는다.

prefill Q tiles가 많아도 causal upper triangle skip, variable lengths와 last tile tail가 있다. Q=128가 BlockM=128을 정확히 채운다는 사실만으로 모든 warps/tensor cores가 완전 활용됐다고 하지 않는다. D/BlockN와 mask axes를 함께 본다.

decode Q=1의 output는 한 row×D지만 LSE는 head/row당 scalar다. split workspace에서는 partial O가 D에 비례하고 LSE는 scalar라 O bytes가 지배할 수 있다. padded D가 actual D보다 크면 workspace가 더 커진다. OOM 산술에 padded layout를 쓴다.

planner가 `num_splits=1`을 선택했는데 tmp workspace가 non-null일 수 있는지, merge predicate가 pointer 또는 split count를 보는지 source를 읽는다. pointer presence만으로 semantic split count를 추정하는 데 예외가 있을 수 있다. source path와 record를 함께 본다.

merge가 별 kernel이면 launch ordering와 stream도 필요하다. attention partial stores 뒤 merge가 같은 stream이면 ordering가 닫히고 다른 stream이면 event가 필요하다. final consumer가 merge completion를 기다려야 한다. math가 맞아도 시간 edge가 빠지면 stale partial을 읽는다.

multi-GPU tensor parallel가 함께 있으면 local attention split merge와 rank collective를 순서대로 구분한다. local final O가 준비된 뒤 collective/next layer consumer가 읽는다. 이 장은 local split-KV math를 소유하고 collective ordering는 네트워크 편으로 넘긴다.

운영 원장의 decode 한 행은 이렇게 쓴다. R7, Q=1, KV=8192, batch=8, qheads=32, kvheads=8, D=128, bf16, causal, pages=512, selected module X, work items N, splits distribution, partial workspace bytes, merge launch Y. 이 정도가 있어야 “decode kernel가 느리다”를 조사할 수 있다.

source walk에서 이 fields를 누가 만든지도 적는다. scheduler/runner가 Q/KV lengths와 block table를 만들고 SGLang adapter가 wrapper inputs를 만든다. FlashInfer plan가 work items/workspace를 만들고 native dispatch가 module/launcher를 고른다. kernel와 merge가 state를 소비한다.

first divergence가 performance only이면 correctness states는 reference와 맞고 work distribution/bytes/timeline에서 차이가 난다. correctness도 깨지면 optimization heuristic보다 address/mask/state merge를 먼저 보호한다. suspect split path를 eager/unsplit로 안전 fallback할 수 있지만 root fix와 구분한다.

decode와 prefill을 잇는 마지막 질문은 KV producer-consumer다. prefill이 쓴 K/V page가 decode에서 정확한 position와 generation로 읽히고 write completion edge가 닫혀야 한다. online softmax source만 읽어 이 lifetime 문제를 놓치지 않는다.

### 45.7.3 decode 저활용이라는 말을 분해한다

Q tile underfill, insufficient total CTAs, long serial KV loop, memory bandwidth, split merge overhead와 launch overhead를 분리한다. Q=1이므로 GPU가 한 thread만 쓴다는 말은 틀리다. 여러 warps가 한 query row의 head/KV tiles를 협력 처리할 수 있다.

batch를 늘리면 CTA supply가 늘 수 있지만 queue latency와 KV length distribution가 바뀐다. splits를 늘리면 parallelism과 workspace/merge가 함께 늘어난다. 작은 BlockM은 row fill을 높일 수 있지만 selected tensor-core tile/reuse를 바꾼다. 하나의 utilization percentage로 결론 내리지 않는다.

### 45.7.4 prefill과 decode를 공정하게 비교한다

Q/KV lengths, batch, heads/D/dtype, causal/window, page layout, selected kernel와 split count를 기록한다. prefill latency와 decode ITL은 다른 user-visible metric이며 work denominator도 다르다. token당 시간만으로 kernel efficiency를 단정하지 않는다.

실제 profiling를 한다면 launched kernels, grid/block, HBM traffic, occupancy/stalls와 partial/merge count를 보겠지만 이 장에서는 실행하지 않는다. 필요한 관측 관계만 정의한다.

## 45.8 여섯 장애를 first divergence로 좁힌다

### 45.8.1 NaN 또는 Inf

scale 적용 뒤 score, softcap 뒤 score, mask 뒤 valid scores, row max, exp sum l, output accumulator와 final LSE를 순서대로 본다. first non-finite state가 어디인지 찾는다. input Q/K/V가 이미 non-finite인지도 먼저 확인한다.

모든 masked row에서 max=-∞와 subtract가 `-∞-(-∞)`가 되면 NaN handling가 필요하다. empty identity/finalization source를 본다. l=0 division를 guard하는지, output/LSE sentinel가 API contract와 맞는지 확인한다.

base-2 exponential와 scale folding를 비교할 때 natural scale를 두 번 적용하거나 log2(e) conversion를 빠뜨리지 않았는지 본다. softcap의 tanh/scale 순서도 backend별 작은 차이 또는 큰 오류를 만들 수 있다.

### 45.8.2 page boundary에서만 오답

logical KV position를 block table index와 page offset으로 계산해 physical K/V pointer까지 추적한다. page size `P`의 positions `P-1,P,P+1`과 last-page tail을 fixture로 둔다. K/V load checkpoint부터 contiguous reference와 비교한다.

block table generation, valid KV length와 page stride가 맞는데 score부터 다르면 loader/layout를 본다. score는 맞고 softmax 이후 틀리면 mask/online state를 본다. final만 틀리면 PV/store 또는 merge다.

### 45.8.3 split 수가 2 이상일 때만 오답

split 1 reference와 split 2/3을 비교한다. 각 partial O/LSE를 손계산 또는 unsplit interval reference와 맞춘다. partial이 맞고 final이 틀리면 combine LSE base, indptr, layout/stride와 empty identity가 후보다.

KV length가 split partition보다 짧아 empty split가 생기는 fixture를 둔다. LSE가 `-∞`인지, zero output와 함께 merge weight 0이 되는지 확인한다. split ordering를 바꿔 result tolerance가 유지되는지도 본다.

### 45.8.4 decode 저활용

Q tiles, batch×query/KV heads, selected split count와 persistent/CTA policy를 기록한다. grid가 작으면 insufficient work supply 가설, grid는 충분하지만 stalls가 크면 memory/dependency 가설이다. Q underfill를 idle lanes로 환산하지 않는다.

split count를 늘리기 전 workspace와 merge cost를 계산한다. batch/heads가 이미 많은 workload에서는 split가 불필요할 수 있다. selected heuristic가 expected shape bucket에서 무엇을 반환하는지 source와 trace로 확인한다.

### 45.8.5 workspace OOM

partial output workspace는 대략 `num_partials×query_rows×heads×padded_D×element_bytes`와 partial LSE `num_partials×query_rows×heads×4 bytes`, indptr/metadata를 포함한다. exact layout와 allocation dtype를 source에서 확인한다. FP32 accumulator workspace면 output dtype보다 bytes가 클 수 있다.

예를 들어 batch rows 8, heads 32, splits 4, padded D 128, partial O FP32라면 단순 O 항만 `8×32×4×128×4=524,288 bytes`다. 이는 교육용 산술이며 allocator overhead와 other buffers는 별도다. 실제 OOM claim에는 exact implementation shape가 필요하다.

OOM를 KV cache capacity와 구분한다. split heuristic가 갑자기 커졌거나 padded D/kernel variant가 바뀌면 workspace peak가 바뀔 수 있다. allocation failure 전 selected plan와 offsets를 본다.

### 45.8.6 backend별 작은 수치 차이

accumulation dtype, exp/exp2 approximation, reduction tree/order, scale folding와 final conversion가 bitwise 차이를 만들 수 있다. deterministic input에서 absolute/relative tolerance, output/logit impact와 first state difference를 본다. final text equality만 쓰지 않는다.

차이가 tolerance 안이고 algorithm invariants가 맞으면 backend numerical variation일 수 있다. page/split boundary에서 갑자기 큰 차이가 나거나 NaN, wrong mask mass가 생기면 implementation bug를 의심한다. “FlashAttention은 exact”가 bitwise identity를 뜻하지 않음을 기억한다.

### 45.8.7 한 request를 full reference에서 CUDA partial까지 추적한다

작은 deterministic request R을 만든다고 하자. query rows는 2, KV length는 5, heads는 1, D는 source가 지원하는 작은 fixture를 택한다. Q/K/V contents와 scale, causal/window를 고정하고 full CPU 또는 명료한 unfused reference에서 scores, masked scores, row max, exponential sum, output와 LSE를 보존한다. 실제 수행은 독자 환경에서 하며 여기서는 절차만 설계한다.

첫 비교점은 backend input다. Python query/KV shape, cumulative lengths, block table와 scale/mask flags가 reference와 같은 logical problem을 표현하는지 확인한다. 여기서 다르면 kernel를 조사하지 않는다. token flattening나 page table가 다른 문제다.

둘째 비교점은 page loader 뒤 QK score다. 전체 score tile을 production에서 dump할 필요는 없지만 작은 fixture에서 selected rows/columns 또는 checksum를 비교할 수 있다. page boundary 전후와 invalid tail를 포함한다. score가 다르면 address/layout/QK/scale, score가 같으면 softmax 이후로 이동한다.

셋째는 tile별 `(m,l)` 또는 LSE다. 첫 tile local state와 second tile merge 뒤 state를 손계산과 비교한다. l은 exp base/scale convention를 맞춘다. m/l가 맞고 O accumulator만 다르면 output rescale/PV를 본다. m부터 틀리면 mask/softcap/score가 후보다.

넷째는 partial split workspace다. split별 KV interval가 reference partition와 맞는지, local O/LSE와 layout offset가 맞는지 본다. empty split에는 O zero/LSE `-∞`가 있어야 한다. partials가 맞으면 combine input indptr와 global LSE/weight를 본다.

다섯째는 final output cast/store다. FP32 accumulator reference와 output dtype cast 차이를 나누고 allowed tolerance를 정한다. output row stride와 active query tail가 맞는지 본다. final logits 차이는 attention output first divergence 뒤의 결과일 수 있다.

이 ladder의 장점은 “FlashAttention이 틀린다”는 넓은 사건을 loader, score transformation, online recurrence, PV, split store, merge와 cast로 좁힌다는 것이다. 각 단계는 source symbol와 state owner가 있다. 가능한 가장 이른 checkpoint를 잡는다.

production에서는 detailed tensors를 항상 기록하지 않는다. aggregate anomaly에서 sampled request와 deterministic replay로 이동한다. 사용자 prompt나 KV contents를 그대로 저장하지 않고 pseudonymous row/page coordinates와 bounded digests를 쓴다. instrumentation가 kernel selection나 timing를 바꿀 수 있음도 기록한다.

### 45.8.8 수치 차이와 correctness failure의 경계

floating-point addition은 결합법칙이 정확히 성립하지 않는다. full reference가 scores를 순서대로 reduce하고 tiled kernel가 warp tree와 split merge 순서로 reduce하면 마지막 bits가 달라질 수 있다. FP16/BF16 input, FP32 accumulation와 output cast의 조합도 차이를 만든다.

허용 tolerance는 임의의 “조금”이 아니다. output dtype, magnitude와 downstream sensitivity를 고려해 absolute/relative 기준을 정하고 first divergence 크기를 기록한다. 모든 rows의 분포와 boundary-specific spikes를 본다. 평균 오차가 작아도 특정 masked row가 크게 틀릴 수 있다.

LSE 차이는 output와 함께 본다. LSE가 크게 다르고 O도 다르면 softmax mass 문제 가능성이 있다. LSE는 맞지만 O가 다르면 PV/accumulator rescale/layout를 본다. O/LSE는 맞는데 final token만 다르면 later norm/logits/sampling threshold를 본다.

backend A와 B가 다른 exp approximation를 써 작은 차이를 보인다고 algorithm 불일치라 하지 않는다. 반대로 split count가 바뀔 때만 discontinuous jump가 나면 rounding보다 merge identity/layout를 먼저 본다. page boundary에 맞춰 jump가 나면 address/mask다.

softcap는 특히 scale-dependent하다. 매우 큰 raw scores를 softcap으로 제한하는 경우 transformation 순서 차이가 결과를 크게 만들 수 있다. API definition와 source order를 맞춘 reference를 사용한다. softcap 없는 reference와 비교해 kernel가 틀렸다고 하지 않는다.

base-2 kernel를 자연 exponential reference와 비교할 때 내부 `log2(e)` folding와 returned LSE conversion를 반영한다. conversion 뒤 차이를 봐야 한다. internal l 값을 그대로 natural-domain l과 비교하면 의미가 다를 수 있다.

stochastic sampling은 attention correctness reference로 적합하지 않다. 같은 logits의 작은 차이가 threshold 근처에서 다른 token을 뽑을 수 있다. attention O/LSE 또는 deterministic logits checkpoint를 먼저 비교하고 sampling parameters/seed를 고정한다.

### 45.8.9 복구와 최적화가 끝났다는 증거

NaN 수정은 non-finite final output가 사라진 것만으로 끝나지 않는다. fully masked row, very large scores, softcap on/off와 multiple dtypes에서 first non-finite checkpoint가 없어야 한다. empty identity와 l=0 finalization가 source contract대로 작동해야 한다.

page boundary 수정은 positions `P-1,P,P+1`, multiple pages와 partial last page에서 contiguous reference와 맞아야 한다. different block table generations와 sequence lengths를 시험한다. 한 page size에서 맞았다고 모든 layouts를 증명하지 않는다.

split merge 수정은 splits 1,2,3과 empty partial, variable indptr를 비교한다. partial O/LSE와 final O/LSE가 reference tolerance 안에 있어야 한다. workspace stride가 padded D와 heads/batch/query axes에서 맞는지 matrix를 만든다.

decode utilization 최적화는 selected split/CTA policy, Q/KV distribution와 workspace를 기록한다. throughput나 ITL만 좋아졌다고 끝내지 않고 output/LSE correctness를 유지하고 memory peak가 허용 범위인지 본다. split 증가가 merge overhead를 만든다는 비용도 포함한다.

workspace OOM 수정은 단순히 maximum splits를 낮춰 숨기지 않는다. plan required bytes와 allocated capacity가 맞고 fallback/alternative policy가 correctness를 유지해야 한다. concurrent requests와 graph/capture buffers가 겹칠 때 peak owner도 본다.

numerical tolerance 변경은 큰 오류를 허용하는 방식으로 incident를 닫지 않는다. expected rounding envelope를 independent reference와 dtype 조건으로 정하고 boundary-specific outliers를 남긴다. algorithm/path가 달라졌다면 새 baseline의 근거를 기록한다.

성능 주장에는 workload, hardware, dtype, Q/KV distribution, batch/heads, selected kernel, splits, warmup와 metric가 필요하다. 논문의 benchmark를 current vLLM fork나 FlashInfer pin의 성능으로 옮기지 않는다. 이 장의 CTA/workspace 숫자는 손계산 fixture이지 실측 결과가 아니다.

장애 보고서의 마지막 문장은 수정한 option보다 invariant를 말한다. “각 valid score는 정확히 한 partial mass에 포함되고 invalid/empty scores는 mass 0이며, partial normalized outputs는 exp(LSE 차이) weights로 한 번 merge된다.” 이 문장이 page, split와 backend variants에서 검증돼야 한다.

prefill/decode 모두 확인한다. prefill은 Q tile/causal boundary, decode는 long KV/split/empty tail가 서로 다른 corner를 만든다. 한 phase의 green result를 다른 phase로 자동 승계하지 않는다. SGLang이 서로 다른 wrappers를, vLLM fork가 서로 다른 specializations를 선택할 수 있기 때문이다.

이 복구 기준은 최적화의 방향도 지킨다. full score materialization를 임시 reference로 사용해도 production fix로 되돌리지 않고 tiled IO-aware path의 invariant를 고친다. split를 완전히 끄면 merge bug는 사라지지만 long decode parallelism opportunity도 사라진다. 안전 fallback와 root fix를 구분한다.

prefill row에는 Q=128, KV=128, batch/heads/D/dtype, causal flag, pages, selected BlockM/N, query tile count, splits와 workspace를 쓴다. decode row에는 Q=1, KV=8192와 같은 fields를 쓴다. 둘의 kernel family/JIT key가 같은지 다른지 명시한다.

prefill first divergence가 causal diagonal tile에서 생기면 mask coordinate를 본다. decode first divergence가 split tail에서 생기면 KV partition/empty identity를 본다. 둘 다 page boundary에서 생기면 common paged loader/block table를 본다. 공통과 phase-specific owners를 나눈다.

성능 원장에는 logical Q/KV tiles와 launched partial/merge work를 구분한다. prefill 2H query CTAs와 decode H×splits CTAs 같은 교육 식을 actual launcher axes로 교체한다. row-slot fill와 measured warp/SM utilization를 같은 열에 쓰지 않는다.

memory 원장에는 KV reads와 final O 외 full S/P가 materialized되는지, partial workspace O/LSE bytes와 plan buffers를 쓴다. split가 없으면 partial workspace가 0일 수 있고 return-LSE만 별 output가 있을 수 있다. allocator reserved와 live workspace를 구분한다.

수치 원장에는 scale/softcap/mask order, exp base, accumulation/output dtype와 tolerance를 쓴다. backend switch 뒤 차이가 생기면 이 fields를 비교한다. “둘 다 FlashAttention 계열”이라는 label는 수치 contract를 충분히 설명하지 않는다.

source 원장에는 Python adapter, native dispatch, launcher, kernel softmax/PV, partial store와 merge symbols를 순서대로 둔다. 한 단계가 prebuilt binary로 보이지 않으면 gap을 명시한다. 비슷한 함수 이름으로 연결을 추측하지 않는다.

두 rows를 나란히 두면 scheduler와 kernel의 관계도 보인다. scheduler가 chunked prefill와 decode batch를 어떻게 섞는지가 Q distribution를 만들고 planner/dispatcher가 CTA/splits를 고른다. kernel만 독립적으로 최적화한다고 serving workload가 자동으로 좋아지지 않는다.

관측 비용을 통제한다. 평상시 shape/key/split/workspace와 anomaly counters를 수집하고 상세 partial O/LSE는 작은 재현에서만 본다. request ID를 metric label로 넣지 않고 sampled trace로 연결한다. contents 대신 ranges, generations와 digests를 쓴다.

마지막으로 comparison denominator를 붙인다. prefill은 processed query/KV tokens와 TTFT context, decode는 generated step와 ITL context를 갖는다. kernel duration 하나가 사용자 latency 전체가 아니며 graph/launch/queue와 memory transfer가 더해질 수 있다. 이 장은 attention tile path의 원장을 제공한다.

논문을 코드에 연결할 때도 같은 엄격함을 유지한다. FlashAttention v2 논문이 IO-aware exact attention와 SRAM tiling의 원리를 설명한다는 사실은 current vLLM fork의 모든 specialization가 논문의 한 pseudocode와 줄 단위로 같다는 뜻이 아니다. current commit에서 실제 QK/softmax/PV와 launcher를 확인한다.

FlashAttention-2 v1은 sequence-length parallelism와 warp work partitioning 개선을 설명한다. 이 아이디어는 prefill/decode 병렬 축을 이해하는 배경이지만 현재 selected SM90 kernel의 warp group 수와 역할을 논문 그림에서 바로 가져오지 않는다. code traits와 thread-role source가 최종 근거다.

FlashAttention-3 v2는 Hopper의 TMA, WGMMA, warp specialization와 overlap을 설명한다. Hopper path를 이해하는 강한 배경이지만 FlashInfer의 generic/paged/decode 또는 Blackwell path를 FA3 구현으로 부르지 않는다. architecture와 library owner를 분리한다.

논문 benchmark도 current serving 성능으로 승계하지 않는다. model shape, sequence lengths, dtype, GPU, baseline와 metric가 다르다. 논문은 algorithm/mechanism claim에 사용하고 vLLM/SGLang/FlashInfer의 현재 path는 pinned source로 설명한다. 실제 성능은 독자 환경에서 별 측정한다.

정상 요청의 원장을 수식과 code 양쪽으로 한 번 더 완성하자. request A의 query row r, head h와 KV logical interval `[0,8192)`를 적는다. block table가 이 interval를 physical pages로 번역한다. plan 또는 launcher가 split intervals와 work items를 만든다. 각 kernel CTA가 자신의 interval에서 local scores, m/l/o 또는 O/LSE를 만든다. merge가 있으면 global state를 만들고 final output row에 쓴다.

수학 원장은 valid score마다 정확히 한 local interval에 속하는지 확인한다. overlapping splits면 mass를 두 번 세고 gap이면 token mass를 빠뜨린다. causal/window mask가 valid set를 더 제한한다. empty interval는 mass 0이다. indptr start/end가 이 partition invariant를 구현한다.

주소 원장은 각 logical position가 current block table generation의 K/V pointer를 읽는지 확인한다. page boundary와 last-page tail에 valid-length predicate가 있다. wrong page가 memory-safe하게 valid data를 가리킬 수 있으므로 bounds check만으로 correctness가 보장되지 않는다.

수치 원장은 scale와 optional softcap 뒤 masked score, local max/sum, output accumulator, local/global LSE와 final cast를 가진다. internal exp2 domain이면 conversion label를 붙인다. normalized O와 unnormalized o를 같은 이름으로 쓰지 않는다.

실행 원장은 selected backend/module, tile traits, grid/block, splits, workspace, stream와 optional merge launch를 가진다. partial kernel→merge→consumer ordering를 닫는다. graph replay라면 static capacity/active metadata generation도 추가한다.

소유권 원장은 plan state, block table/KV pages, partial workspace와 output buffer가 어느 completion까지 살아 있는지 적는다. request cancellation나 allocator reuse가 kernel/merge보다 앞서지 않는다. correctness math가 맞아도 lifetime가 틀리면 동일 symptom이 난다.

관측 원장은 first divergence와 expected relation를 가진다. loader checkpoint가 reference와 다르면 address, score가 다르면 QK/scale/mask, m/l가 다르면 online reduction, O만 다르면 rescale/PV, partial은 맞고 final만 다르면 merge, attention은 맞고 token만 다르면 downstream을 본다.

이 원장을 prefill과 decode에 각각 채우면 같은 algorithm와 다른 work partition를 동시에 설명할 수 있다. prefill은 많은 query tiles와 causal boundary가 중심이고 decode는 long KV, ragged pages, split/merge와 batch/head packing이 중심이다. 공통 online recurrence를 반복 설명하지 않고 phase-specific mapping만 바꾼다.

그럴듯하지만 틀린 문장도 원장으로 반증한다. “FlashAttention은 scores를 근사한다”는 full/online 손계산과 exact algorithm 정의로 반증한다. “score matrix를 안 쓰니 FLOPs가 줄었다”는 QK/PV work와 IO separation으로 반증한다. “Q=1이면 thread 하나만 일한다”는 CTA/warp cooperative mapping로 반증한다.

“split outputs를 평균하면 된다”는 LSE-weighted 손계산으로 반증한다. local scores mass가 다르므로 weights는 token count나 동일 1/S가 아니다. “empty output가 zero면 안전하다”는 wrong LSE=0이 denominator mass를 만드는 계산으로 반증한다.

“plan가 끝났으니 attention도 끝났다”는 host plan/native run/launch/merge lifecycle로 반증한다. “prefill module을 decode가 쓰면 prefill semantics다”는 adapter params와 effective Q/KV work로 반증한다. “둘 다 exact이므로 bitwise 같다”는 floating reduction/exp/dtype 차이로 반증한다.

이 반증들은 독자가 다음 소스를 고르게 한다. approximation 의문은 논문 algorithm, FLOP/IO는 tile data lifecycle, split average는 cascade merge, plan confusion는 Python/native boundary, bitwise 차이는 accumulator/exp/final cast를 읽는다. 넓은 label 대신 질문에 맞는 owner로 이동한다.

실전 복구에서 reference implementation를 켜는 것은 유용하지만 production answer가 아닐 수 있다. 느린 unfused/unsplit path로 correctness를 보호하고 incident 증거를 얻은 뒤 paged/tiled/split path의 first divergence를 고친다. fallback 상태와 root fix 완료를 release note에서 구분한다.

workspace를 줄이려고 partial O를 더 낮은 dtype으로 바꾸면 numerical contract도 달라진다. split 수를 줄이면 planner와 performance shape가 달라진다. tile를 바꾸면 resources와 reduction order가 달라진다. optimization이 바꾸는 state를 원장에 명시하고 correctness/tolerance를 재검증한다.

page layout를 고치며 contiguous benchmark만 돌리지 않는다. 실제 block table, partial last page, prefix-shared blocks와 sliding window boundary를 포함한다. split merge를 고치며 equal-length splits만 돌리지 않고 empty/uneven intervals를 포함한다. 장애가 드러난 축을 유지한 fixture가 필요하다.

NaN incident에서는 non-finite를 zero로 치환해 숨기지 않는다. fully masked identity처럼 정의된 exceptional row는 contract대로 처리하고 unexpected overflow/invalid load의 first cause를 찾는다. final sanitizer는 안전 장치일 수 있지만 root scale/mask/state bug를 고치지 않는다.

performance incident에서는 correctness checkpoint를 생략하지 않는다. 더 높은 throughput가 wrong mask, missing KV tail 또는 reduced precision error에서 나온 것일 수 있다. expected valid work와 output tolerance를 먼저 고정한 뒤 work partition/IO를 최적화한다.

독자가 이 장을 덮을 때 한 문장으로 current request를 설명할 수 있어야 한다. “SGLang adapter가 만든 paged KV metadata를 FlashInfer plan가 variable split work로 바꾸고 native module이 partial O/LSE를 쓴 뒤 `VariableLengthMergeStates`가 exp(LSE 차이)로 final O를 만든다”처럼 쓴다. vLLM path도 Python varlen call에서 current fork의 QK→softmax rescale→PV→final/merge로 이어 쓴다.

설명이 막히는 지점은 다음 조사 TODO다. split count owner가 불명확하면 planner heuristic를 찾고, LSE base가 불명확하면 store/combine conversion를 찾고, workspace layout가 불명확하면 producer/consumer pointer arithmetic를 함께 읽는다. 빈칸을 “FlashAttention 최적화”라는 말로 덮지 않는다.

인계 문서에는 정상 경로와 실패 경로를 같은 좌표로 쓴다. 정상 request R0의 Q/KV lengths, page generation, selected module, splits, partial/global LSE와 output digest를 남긴다. 실패 R1에서 처음 다른 field를 표시한다. final token 문자열만 붙이면 다음 조사자는 다시 처음부터 source를 읽어야 한다.

split-only 사건이라면 “splits=1은 정상, splits=2에서 partial 0/1 O와 LSE는 reference와 맞지만 merge global LSE가 처음 다르다”처럼 쓴다. 이 문장은 planner/loader/online kernel를 잠시 제외하고 combine base/indptr/reduction를 가리킨다. partial부터 다르면 merge만 고쳐서는 안 된다.

page 사건이라면 “position P-1은 정상, P의 physical block generation가 expected와 다르고 QK score부터 divergence한다”처럼 쓴다. softmax 수식보다 block table/page offset를 먼저 본다. physical load가 맞은 뒤에도 score가 다르면 layout/scale로 이동한다.

NaN 사건이라면 “mask 적용 뒤 row가 fully invalid이고 local max가 -∞, sum update에서 처음 NaN”처럼 쓴다. input non-finite와 overflow를 제외했는지 붙인다. empty row contract를 복구한 뒤 ordinary rows의 numerical tolerance도 다시 확인한다.

저활용 사건이라면 “decode Q=1, KV=8192, batch×heads work N, selected splits S, partial workspace W, merge launch M”을 적는다. logical underfill와 actual stall reason를 구분한다. split를 바꾸었다면 CTA supply뿐 아니라 bytes와 correctness result를 함께 비교한다.

OOM 사건이라면 allocation stack만 보고 끝내지 않는다. plan가 요구한 partial count, padded D, dtype와 concurrency owner를 계산한다. graph/capture pools와 other workspaces가 동시에 live인지 본다. fallback split policy가 capacity 안에서 valid partition를 유지하는지도 확인한다.

작은 수치 차이는 first divergent state와 magnitude distribution를 남긴다. output 평균 오차 하나 대신 boundary rows, maximum absolute/relative error와 non-finite count를 본다. accepted tolerance의 dtype/shape 근거를 기록한다. backend upgrade 때 같은 기준으로 regression를 판정할 수 있다.

이 보고 방식은 긴 책의 세부를 운영 판단으로 압축한다. request→metadata→plan/dispatch→page load→QK/mask→online state→PV→partial store→LSE merge→final output이라는 순서는 유지하되, first divergence 앞의 정상 stages와 뒤의 파생 증상을 구분한다.

궁극적으로 최적화의 가치는 score matrix를 쓰지 않았다는 사실 하나가 아니다. 정확한 dense attention를 유지하면서 memory hierarchy에 맞는 tile, serving shape에 맞는 work partition, paged KV에 맞는 address와 안전한 split merge를 함께 만족해야 한다. 어느 한 계약이 깨지면 빠른 wrong answer거나 느린 correct fallback가 된다.

임시 대응과 완료도 분리한다. split를 1로 고정하거나 paged path를 끄고 contiguous reference로 보내면 사용자 영향을 줄일 수 있다. softcap를 끄거나 backend를 바꾸는 조치도 특정 증상을 피할 수 있다. 그러나 어떤 state가 처음 틀렸는지 고치지 않았다면 이는 안전 fallback다. 원래 최적화 path를 다시 켠 상태에서 page·mask·empty split·ragged merge와 numerical tolerance가 통과해야 완료다.

재발 방지는 평균 benchmark가 아니라 경계 fixture를 남기는 일이다. Q tile 앞뒤, page 앞뒤, split empty/uneven, fully masked row, very large score와 dtype variants를 작은 deterministic tests로 둔다. current commit와 expected source owner를 함께 기록한다. 다음 library update에서 line가 바뀌더라도 같은 invariant를 새 함수에 다시 매핑할 수 있다.

독자는 결국 “어느 kernel이 빠른가”보다 강한 질문을 갖게 된다. “이 request의 valid score set를 누가 정의하고, 어떤 tiles가 그것을 한 번씩 덮으며, m/l/o 또는 O/LSE가 어느 dtype/base/layout로 이동하고, 어떤 merge가 global normalization를 닫는가?” 이 질문에 답하면 correctness와 performance 조사가 같은 실행 경로 위에서 만난다.

## 45.9 종합 판정: tile 사이에 보존할 것은 확률표가 아니라 충분한 상태다

attention output는 모든 scores를 HBM에 저장해야만 계산되는 값이 아니다. 한 row에서 stable softmax에 필요한 running max `m`, exponential mass `l`, weighted value accumulator `o`를 tile마다 갱신하면 전체 dense attention와 같은 결과를 복원할 수 있다. max가 바뀔 때 l과 o를 같은 factor로 rescale하는 것이 핵심 invariant다.

scores `[1,2]`와 `[3,0]`, values `[10,20,30,40]`의 full output와 online output는 약 26.2089, LSE는 약 3.44019로 맞았다. split가 local normalized O/LSE를 저장해도 global LSE 차이로 weights를 만들어 같은 결과를 merge할 수 있다. empty split은 O=0, LSE=-∞여야 mass 0 identity가 된다.

FlashAttention은 이 수학을 QK tile, mask/softmax, P conversion와 PV MMA의 pipeline에 넣어 전체 S/P matrix HBM materialization를 피한다. approximation나 sparse attention가 아니며 FLOPs가 자동으로 줄었다고 주장하지 않는다. tile size, data movement와 warp 역할은 selected architecture specialization에 달렸다.

vLLM은 Python varlen metadata에서 current FlashAttention pin의 dispatch/launcher/kernel로 내려간다. running softmax와 output rescale, final normalization/LSE, split workspace와 combine을 state 순서로 읽는다. 오래된 fork line 좌표를 current pin에 승계하지 않는다.

SGLang은 request/KV metadata를 FlashInfer wrapper plan/run contract로 바꾼다. plan은 padded work items, CTA policy와 workspace를 준비하는 host 단계이고 run/native dispatch가 kernel을 launch한다. partial tmp states가 있으면 variable-length merge가 final O/LSE를 만든다. decode가 조건에 따라 batch prefill module을 재사용할 수 있으므로 phase 이름에서 kernel을 추측하지 않는다.

prefill Q=128은 query tile 병렬성이 있지만 decode Q=1, KV=8192는 batch×heads, KV splits와 persistent work 같은 다른 축이 중요하다. split는 parallel supply와 workspace/merge를 함께 늘린다. logical row underfill를 idle lane이나 measured utilization로 바꾸지 않는다.

장애는 first divergence로 나눈다. non-finite는 scale/mask/max/sum/LSE, page boundary는 logical→physical load, split-only는 partial/layout/LSE merge, decode 저활용은 work axes와 CTA policy, OOM은 workspace 산술, 작은 차이는 dtype/exp/reduction tolerance를 본다. final token 하나로 online softmax를 탓하지 않는다.

다음 장은 같은 tile math를 저비트 GEMM와 MoE kernels로 확장한다. quantized weights의 packing, scales와 accumulators가 matmul tile에 어떻게 들어가고 expert routing가 work distribution를 어떻게 바꾸는지 살펴본다.

## 45.10 같은 네 score로 두 backend 계약을 손검산한다

장애 장면부터 시작한다. 팀은 설정의 backend 문자열을 FlashAttention에서 FlashInfer로 바꾸고 나머지 tensor,
workspace와 metadata 계약은 같다고 가정했다. 짧은 contiguous prefill은 통과했지만 paged decode에서 page
boundary를 넘는 요청 일부가 다른 token을 냈다. 성능 canary도 plan 시간을 제외한 kernel만 비교해 새 backend가
빠르다고 보고했다. 이름은 attention이지만 selector 입력, page metadata, workspace와 merge contract가 달랐다.

작은 fixture는 query 한 행, scores `[1,2,3,0]`, scalar values `[10,20,30,40]`이다. stable full softmax는
max 3을 빼 `e^-2,e^-1,1,e^-3`을 얻는다. mass는 약 `0.135335+0.367879+1+0.049787=1.553001`이고 weighted
sum은 약 `1.35335+7.35758+30+1.99148=40.70241`이다. output은 약 26.2089, LSE는
`3+log(1.553001)≈3.44019`다. backend 비교의 첫 기준은 이 값과 valid score set이다.

첫 tile `[1,2]`은 local max 2, mass `e^-1+1≈1.367879`, accumulator
`10e^-1+20≈23.67879`다. 둘째 `[3,0]`을 볼 때 global max는 3으로 오른다. old mass와 accumulator를
`e^(2-3)=e^-1`로 줄여 각각 약 0.503215와 8.710, 새 tile mass `1+e^-3≈1.049787`, accumulator
`30+40e^-3≈31.99148`를 더한다. global mass 1.553002, accumulator 40.7015로 rounding 범위에서 full
결과와 맞는다. max가 바뀔 때 accumulator만 또는 mass만 rescale하면 오답이다.

split merge도 같은 fixture를 사용한다. split0 normalized output은 `23.67879/1.367879≈17.3106`, local
LSE는 `2+log(1.367879)≈2.31326`이다. split1 output은 `31.99148/1.049787≈30.4743`, LSE는
`3+log(1.049787)≈3.04859`다. global LSE를 두 local LSE의 log-sum-exp로 구하고 weight
`exp(LSE_s-global_LSE)`를 곱하면 약 26.2089다. 두 output의 단순 평균 23.89245는 틀린다.

완전히 masked split은 normalized O=0만으로 부족하다. LSE가 0이면 `exp(0-global)`의 가짜 mass가 생긴다.
identity는 LSE=-∞여서 weight 0이 되는 것이다. empty row의 final output/LSE contract는 API마다 확인하되,
merge recurrence의 mass identity는 보존해야 한다. NaN을 마지막에 0으로 치환하면 잘못된 empty identity를
숨길 수 있다.

수치 안정성은 exact real-number equality와 bitwise equality를 구분한다. 두 backend가 같은 valid scores를
덮고 stable online recurrence를 써도 tile 순서, exp와 exp2, scale folding, accumulator dtype, partial O의
normalization 시점과 final cast가 다르면 마지막 bit가 달라질 수 있다. fp16/bf16 output의 tolerance는 shape와
reference dtype에 근거해 정한다. page boundary에서 큰 오차나 token 변화가 반복되면 일반 rounding으로
덮지 않는다.

exp2 경로에서는 natural exponent score `s`를 `s×log2(e)`로 바꿔 `2^x`를 쓸 수 있다. scale이 QK 이전,
score 이후 또는 exp2 conversion에 fold되는지 source를 따라간다. scale을 두 번 적용하면 softmax가 지나치게
sharp해지고 누락하면 flat해진다. softcap이 있으면 적용 순서와 reference에도 같은 transformation을 넣는다.

같은 수학 뒤 실행 계약은 갈라진다. FlashAttention path는 vLLM Python backend가 varlen lengths, causal/window,
block table와 scale을 native call로 넘기고 current pinned fork의 launcher가 architecture/shape specialization을
고른다. kernel은 QK tile, mask, online state, PV와 finalization을 수행하며 split path면 partial workspace와
combine이 추가된다. [vLLM FlashAttention adapter](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/flash_attn.py#L743-L1065)

역할별 고정 좌표와 판정 범위는 다음과 같다.

- FlashInfer path는 SGLang adapter가 qo/kv indptr, page indices, last-page length, heads, dtype와 workspace를 wrapper plan에 넣는다.
- plan은 attention output이 아니라 work assignment와 auxiliary metadata를 준비한다.
- run이 native module과 kernel을 launch하고 variable split이면 partial O/LSE를 cascade merge가 소비한다.
- [SGLang FlashInfer plan metadata](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_mla_backend.py#L830-L879) [FlashInfer plan contract](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/attention/_core.py#L94-L213)

plan/run을 compile/run처럼 막연히 부르지 않는다. plan input generation이 바뀌면 같은 wrapper object를 재사용할
수 있는지, workspace capacity와 active count가 어떻게 구분되는지, captured graph에서 pointer와 metadata가
static인지 확인한다. plan 완료는 GPU output readiness가 아니다. run/merge가 같은 stream dependency와 workspace
generation을 소비해야 한다.

paged KV fixture는 page size2, logical KV positions0~3, physical pages `[7,3]`, last page valid2로 둔다.
scores `[1,2]`는 page7, `[3,0]`은 page3에서 와야 한다. backend A block table이 physical page IDs를 받고
backend B가 byte offsets 또는 다른 layout을 기대한다면 같은 tensor pointer를 넘겨도 틀린 row를 읽는다.
shape와 bounds는 유효해 crash 없이 wrong output가 난다.

last page valid1로 바꾸면 score0은 masked돼 valid set `[1,2,3]`이 된다. reference output도 다시 계산해야
한다. planner가 last-page length를 element count, token count, zero/one-based convention 중 무엇으로 읽는지
고정 source를 본다. contiguous fixture는 이 metadata를 사용하지 않아 통과할 수 있다. boundary fixture가
필수인 이유다.

prefill/decode selector도 이름으로 추측하지 않는다. prefill Q=128은 query tile 축이 풍부하고 causal diagonal
tail이 있다. decode Q=1, KV=8192는 batch×heads, KV splits와 persistent scheduling이 parallel supply를 만든다.
SGLang adapter가 decode에서 batch prefill module을 재사용할 수 있다면 module class 이름이 phase semantics를
결정하지 않는다. effective Q/KV lengths, wrapper, plan policy와 launched symbol을 기록한다.

selector 원장은 `request phase/shape→backend capability→wrapper/module→plan policy→native symbol→merge`다.
fallback reason도 넣는다. dtype, head dimension, page size, window/causal, graph mode, workspace capacity와
architecture가 predicate다. backend flag가 같아도 one request는 FlashInfer, 다른 shape는 Triton 또는
FlashAttention fallback일 수 있다. metric에 configured와 selected backend를 분리한다.

workspace는 byte와 lifetime으로 계산한다. partial state가 split S, rows R, padded value dimension Dp,
accumulator bytes a, LSE bytes l이면 설명용 하한은 `S×R×(Dp×a+l)`에 metadata/alignment를 더한다. plan이
capacity Smax를 예약하고 run은 active S만 쓰는지 구분한다. concurrency가 늘 때 request별 workspace인지
shared arena인지, graph capture pool과 eager pool이 동시에 live인지 본다.

workspace generation도 correctness다. request A plan이 indptr와 partial offsets를 workspace generation7에
쓰고 취소된 뒤 B가 generation8로 재사용했다고 하자. A의 늦은 run/merge가 generation7 pointer를 소비하면
B metadata 또는 partial을 덮는다. output tensor bounds가 유효해도 wrong answer가 난다. plan token, request
incarnation, workspace generation과 stream completion을 연결한다.

## 45.11 backend 교체 오진 사건을 call graph에서 복구한다

사건의 배포 전 경로는 `vLLM selector→FlashAttention varlen adapter→paged launcher→kernel→optional combine`이었다.
배포 후 팀은 gateway flag만 FlashInfer로 바꾸고 기존 block table, workspace pool과 output tolerance를 그대로
썼다. 실제 경로는 `SGLang attention backend→FlashInfer wrapper plan→native paged run→partial states→cascade
merge`였다. 두 경로는 exact attention이라는 제품 목적은 같지만 metadata와 lifetime 계약이 같지 않았다.

증상은 세 가지였다. page boundary를 넘지 않는 prompt는 정상, last page가 partial이면 output divergence,
split=1은 정상이고 split≥2에서 차이가 커졌다. benchmark는 plan을 warmup에서 한 번만 재고 run kernel만
분모에 넣어 18% 향상을 보고했다. production에서는 ragged batch마다 plan/metadata update와 merge가 들어가
ITL p99가 11% 악화됐다.

첫 가설은 FlashInfer online softmax가 근사라는 것이었다. 네-score contiguous fixture에서 partial m/l/o,
O/LSE와 final output이 fp32 reference tolerance 안에 맞아 기각됐다. 둘째는 exp2 rounding이다. 작은 차이는
설명했지만 page/split boundary에 고정된 큰 divergence는 설명하지 못했다. 셋째 page metadata, 넷째 merge
layout, 다섯째 stale workspace generation을 경쟁시켰다.

first divergence checkpoint는 loader였다. Q와 logical token0~1 scores는 reference와 맞았지만 logical token2의
K fingerprint가 달랐다. 새 wrapper의 page indptr/indices를 만들 때 old block-table row를 byte offset처럼
재해석한 adapter bug였다. split≥2에서는 잘못된 second interval이 별 partial로 격리돼 merge 차이가 더 크게
보였다. online recurrence와 merge는 주어진 잘못된 scores를 정확히 계산했다.

수정은 backend 이름 mapping이 아니라 DTO 변환이었다. logical page index, physical page ID, page size,
last-page length, KV layout, dtype/scale와 generation을 FlashInfer wrapper가 기대하는 metadata로 명시적으로
만들었다. source pointer/stride와 index unit을 assertion했다. 기존 FlashAttention block table tensor를 type이
맞는다는 이유만으로 전달하지 않았다.

이 사고를 각 역할의 좌표로 되짚으면 다음과 같다.

- 그 뒤 split-only 작은 차이가 남았다.
- partial O는 normalized였고 LSE는 natural-log domain인데 custom glue가 unnormalized accumulator와 log2 LSE로 가정했다.
- native `VariableLengthMergeStates`를 사용하도록 되돌리고 producer/consumer layout, dtype와 base를 같은 source walk로 고정했다.
- [FlashInfer split/merge launch](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/attention/prefill.cuh#L4210-L4260) [FlashInfer variable merge](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/attention/cascade.cuh#L650-L720)

performance 오진은 measurement contract를 고쳤다. cold first plan, steady same-shape plan reuse, ragged replan,
run, merge, graph/eager와 scheduler gap을 각각 잰다. prefill은 processed query/KV tokens와 TTFT, decode는
generated steps와 ITL로 정규화한다. kernel-only 18%는 유효한 내부 관측이지만 end-to-end winner 주장이
아니다.

수치 검증 matrix는 score magnitude `[0,1,20,80]`, negative large, fully masked, causal boundary와 softcap을
포함한다. fp32 unfused reference에서 score/mask/output/LSE를 만들고 backend별 dtype tolerance를 둔다.
non-finite count는 hard gate다. exact text equality만 보면 sampling이 작은 logits 차이를 증폭하거나 숨길 수
있으므로 deterministic logits/output를 먼저 본다.

주소 matrix는 contiguous, page boundary 전후, page permutation, last-page 0/1/full, shared prefix generation과
stale block을 포함한다. 각 logical token이 정확히 한 physical K/V row를 읽는지 sampled fingerprint로 본다.
raw customer KV를 로그에 남기지 않는다. wrong page가 valid allocation이면 memory checker만으로 잡히지 않는다.

split matrix는 S=1/2/4, equal/uneven/empty intervals를 포함한다. local valid score partition은 overlap과 gap 없이
전체 valid set을 한 번 덮어야 한다. partial O/LSE가 reference와 맞고 global만 틀리면 merge를, partial부터
틀리면 loader/QK/mask/online state를 본다. empty LSE=-∞ identity를 확인한다.

workspace matrix는 exact capacity, 한 byte 부족, concurrency churn, cancel-after-plan, cancel-after-run-before-merge와
graph replay generation을 넣는다. allocation failure가 safe fallback 또는 reject로 가는지, partial offsets가
겹치지 않는지, 늦은 completion이 새 owner를 건드리지 않는지 확인한다. output correctness와 final live refs0를
동시에 본다.

selector matrix는 prefill/decode, Q=1/2/128, KV short/long, supported/unsupported head dimension, dtype,
page size, window와 graph mode를 조합한다. expected selected module과 fallback reason을 source predicate로 먼저
쓴다. 모든 shape를 한 backend로 강제하는 것이 목표가 아니다. 각 branch가 자기 계약을 지키고 관측 가능한
것이 목표다.

rollout은 backend 이름이 아니라 contract generation으로 나눈다. canary request trace에 selected backend,
metadata schema generation, plan/workspace generation, native symbol, splits, merge와 output digest를 둔다.
configured backend label만으로 cohort를 만들지 않는다. fallback과 mixed workers를 분리한다.

rollback은 새 selector admission을 중단하고 plan/run/merge in-flight를 drain한다. FlashInfer layout으로 만든
workspace와 graph를 FlashAttention request에 재사용하지 않는다. paged metadata와 KV payload 자체가 공통일
수 있어도 adapter view/schema generation이 다르면 rebuild한다. old/new output buffers의 late completion을
request incarnation으로 fence한다.

완료 조건은 reference parity, boundary/split/workspace fault matrix, plan 포함 end-to-end SLO와 resource cleanup이다.
fallback으로 split1/contiguous만 써 사용자 영향을 막은 상태는 containment다. paged ragged split path를 다시
켠 canary가 통과해야 root fix 완료다. release note에는 임시 flag와 최종 DTO/merge 수정의 차이를 남긴다.

## 45.12 plan·run·paged KV의 byte와 lifetime을 숫자로 닫는다

decode fixture를 Q rows 4, KV length `[8192,4097,2048,1]`, heads 8, value dimension 128, fp16으로 둔다.
planner가 각 row를 최대 4 splits로 나눈다면 active partial 수는 shape에 따라 4+3+2+1=10이라고 하자.
partial O를 fp32로 저장하면 `10×8×128×4=40,960 byte`, LSE는 `10×8×4=320 byte`다. alignment와 metadata를
제외한 하한 41,280 byte다. capacity Smax16으로 예약하면 active보다 큰 66,048 byte가 필요하다. 요청 수,
heads, padded dimension과 accumulator dtype이 workspace를 곱한다.

plan metadata에는 qo indptr, paged KV indptr/indices, last-page length, work indptr와 split mapping이 있다.
각 배열의 element width와 capacity를 source에서 확인한다. host plan object size만 세고 device auxiliary buffer를
빼지 않는다. graph capture가 maximum capacity buffer를 따로 잡으면 eager workspace와 동시 live인지 본다.

page size16, KV length4097이면 full pages256개와 last page valid1이다. off-by-one으로 last length0을 넘기면
마지막 token mass가 빠지고, 16으로 넘기면 15 stale/padding rows가 valid set에 들어갈 수 있다. bounds 안의
valid allocation이면 crash가 없다. logical score partition과 last-page predicate가 correctness 증거다.

prefill fixture Q=128, KV=128에서는 query tiles가 충분해 split workspace 없이도 grid가 클 수 있다. decode
Q=1, KV=8192에서는 query 축이 하나라 KV split이 CTA supply를 늘린다. split4가 kernel 80µs→35µs로 줄여도
plan 8µs, partial store 6µs, merge 12µs와 scheduler gap이 더해지면 attention stage는 61µs다. split1의 plan
reuse2µs, kernel80µs, no merge면 82µs라 아직 split4가 낫지만 kernel 숫자만의 2.29배가 아니라 1.34배다.

batch가 작고 ragged replan이 30µs라면 split4 total83µs로 이득이 사라진다. graph에서 plan을 안전하게 reuse할
수 있으면 다시 달라진다. “FlashInfer가 빠르다” 대신 shape/cohort별 plan amortization, run, merge와 launch를
쓴다. FlashAttention도 split/combine과 architecture launcher 비용을 같은 범위로 잰다.

online state byte는 full score matrix와 비교한다. Q128×KV8192×fp16 score matrix는 약 2MiB/head지만 tiled
algorithm은 이를 HBM에 전부 materialize하지 않는다. 그렇다고 workspace가 0은 아니다. split partial O/LSE,
planner metadata와 output이 존재한다. IO-aware 이득과 auxiliary capacity를 동시에 기록한다.

plan/run race는 시간축으로 본다. t0 A plan generation11이 workspace offsets를 쓴다. t1 A가 cancel되고 allocator가
free로 표시한다. t2 B plan generation12가 같은 bytes를 덮는다. t3 A run이 늦게 launch되면 B metadata를 읽는다.
host wrapper object가 살아 있다는 사실은 device workspace generation을 보호하지 않는다. submit/completion과
request incarnation fence가 필요하다.

run 뒤 merge race도 있다. partial kernel end event 전 merge가 다른 stream에서 읽으면 incomplete O/LSE를
소비한다. 같은 stream 순서 또는 explicit event dependency를 확인한다. CPU가 launch를 반환받았다는 사실은
partial completion이 아니다. graph capture에서는 dependency edge가 capture에 들어갔는지 본다.

workspace OOM fallback은 split 수를 줄일 수 있지만 planner partition과 selector를 함께 갱신해야 한다. partial
buffer만 작게 주고 kernel이 Smax offsets를 쓰면 OOB다. split1 fallback이면 merge를 생략하는 branch와 output
layout이 맞는지 확인한다. OOM 뒤 stale plan을 재사용하지 않는다.

observability funnel은 planned rows/splits/workspace, run launched/completed, partial ready, merge completed와
consumer ready다. configured backend, selected module와 native symbol을 분리한다. raw pointers와 request ID는
metric label이 아니라 sampled trace에 둔다. anomaly에는 page boundary, empty split, generation mismatch와
non-finite reason을 bounded label로 둔다.

## 45.13 두 backend를 선택하는 독자용 검증·rollback 절차

### Correctness — 같은 logical input이 같은 output을 만드는가

성능보다 먼저 이 국면을 닫는 이유는 잘못된 valid set이나 page를 덜 읽은 kernel도 더 빠르게 보일 수 있기 때문이다. Falsifier는 tolerance를 벗어난 output/LSE, first-divergence 수학 state 또는 logical position과 다른 physical row다.

첫 단계는 output contract다. model, tokenizer/template, Q/K/V tensors, scale, causal/window/softcap, KV valid
set과 output/LSE dtype/layout을 고정한다. backend마다 요구하는 metadata DTO는 다르게 만들되 logical input은
같게 한다. random sampled text보다 deterministic tiny tensors와 fp32 reference를 먼저 쓴다.

둘째는 수학 checkpoint다. QK score와 mask, tile local max/mass/accumulator, partial normalized O/LSE, global
merge와 final cast 중 available 지점을 비교한다. 모든 내부 state를 production에서 dump하지 않는다. 작은
격리 fixture와 debug build에서 first divergence만 수집한다. ordinary, large magnitude, empty와 uneven split을
포함한다.

셋째는 address checkpoint다. logical token→page index→physical page generation→K/V row를 잇는다. contiguous,
boundary-1/boundary/boundary+1, permuted pages, partial last page와 shared prefix를 시험한다. loader부터 다르면
softmax를 고치지 않는다. page는 맞고 QK부터 다르면 layout/scale/dtype을 본다.

### Selection — 요청한 backend가 실제 call graph가 되었는가

Correctness fixture가 같아도 selector와 native symbol이 다르면 비교 대상 자체가 달라지므로 이 국면을 따로 둔다. Falsifier는 requested option과 다른 effective wrapper·symbol, 설명되지 않은 fallback 또는 plan/merge branch다.

넷째는 selector/call graph다. configured option에서 parser/effective state, capability predicate, chosen wrapper,
plan policy, native symbol, splits와 merge까지 기록한다. unsupported fallback을 failure0으로 세지 않는다.
prefill/decode라는 API phase와 실제 module 이름을 같다고 가정하지 않는다.

### Performance — 전체 serving cost가 실제로 줄었는가

Kernel duration 하나가 아니라 plan·workspace·merge·scheduler gap까지 보는 이유는 backend가 비용을 다른 stage로 옮길 수 있기 때문이다. Falsifier는 output parity를 통과하고도 TTFT·ITL·workspace peak 또는 fallback cohort가 guardrail을 넘는 경우다.

다섯째는 cost다. plan cold/warm/replan, workspace reserve/active, run, merge, graph/launch와 scheduler gap을 같은
timeline에 둔다. prefill은 TTFT와 processed tokens, decode는 ITL과 generated step으로 본다. kernel duration만
winner를 결정하지 않는다. wrong valid set으로 work를 덜 한 빠른 결과는 탈락이다.

### Rollback — 이전 generation으로 섞임 없이 돌아갈 수 있는가

선택을 되돌리는 것만으로 끝내지 않는 이유는 이미 plan된 request, captured graph와 workspace가 새 schema를 계속 참조할 수 있기 때문이다. Falsifier는 late completion, live ref·workspace residue, mixed selected symbol 또는 baseline과 다른 output digest다.

여섯째는 lifetime이다. plan/workspace/KV/output generation, stream events, cancel/finish와 allocator reuse를
연결한다. fault injection으로 plan 직후 cancel, run 중 cancel, partial 후 merge 전 cancel, workspace churn과
late completion을 넣는다. output publish가 없더라도 refs/workspace가 남으면 장기 OOM이므로 실패다.

선택 문서는 workload별이다. Hopper prefill 큰 Q에서 current FlashAttention specialization과 graph가 유리할
수 있고, ragged paged decode에서 FlashInfer planner/split가 유리할 수 있다. 이는 보편 순위가 아니라 pinned
version, architecture, shape와 end-to-end 측정의 결론이다. 필수 capability와 fallback 비용을 적는다.

rollout은 tiny parity, offline shape matrix, shadow selector, one-cohort canary 순서다. shadow는 decision/plan만
비교하고 실제 workspace mutation과 kernel correctness를 증명하지 못한다. canary에서 rare page/split/empty와
cancel branch가 실행되게 한다. 평균 traffic만으로 경계를 기다리지 않는다.

자동 중단선은 non-finite, tolerance 초과, wrong page/generation, workspace overlap/ref leak, unexpected fallback,
TTFT/ITL와 OOM이다. 작은 bitwise 차이는 tolerance distribution으로 판정하지만 systematic boundary divergence는
즉시 중단한다. performance improvement가 correctness gate를 완화하지 않는다.

rollback은 selector generation을 이전 backend로 돌리고 새 admission을 막는다. 이미 plan된 requests는 해당
backend/workspace schema에서 drain하거나 명시적으로 discard/recompute한다. in-flight native kernels와 merge가
끝나기 전에 workspace와 output을 old backend에 넘기지 않는다. captured graph와 module cache도 generation별로
격리한다.

external KV payload가 공통 tensor layout이라도 metadata adapter는 rebuild한다. backend별 block-table/indptr,
last-page convention과 scale view가 다를 수 있다. old FlashInfer plan을 FlashAttention call에 재사용하지 않고
반대도 마찬가지다. cache invalidation 범위는 payload identity와 view schema를 구분해 정한다.

rollback 검증은 tiny four-score output/LSE, paged boundary, split uneven/empty, cancellation churn과 production
cohort SLO를 반복한다. baseline selected symbol, workspace live bytes와 output digest가 돌아오는지 본다.
backend flag만 원래 문자열이면 충분하지 않다.

call graph를 더 구체적으로 걸어 보자. vLLM의 FlashAttention backend에서는 scheduler가 만든 query와 KV cache,
sequence metadata가 Python implementation의 forward 경계로 들어간다. 여기서 cumulative sequence lengths,
maximum query/KV lengths, causal/window, scale, block table과 scheduler metadata가 native varlen call의 인자가
된다. Python 함수 이름에 `flash_attn`이 있다고 실제 CUDA symbol이 하나로 고정되지는 않는다. architecture,
dtype, head shape, causal/window, paged 여부와 split 조건이 launcher/dispatch를 거쳐 specialization을 고른다.

specialization을 고르는 launcher와 kernel의 좌표는 다음과 같다.

- 현재 vLLM pin의 Hopper launcher는 traits와 runtime params에서 grid, scheduler와 kernel template를 정한다.
- [FlashAttention Hopper launch template](https://github.com/vllm-project/flash-attention/blob/28e862d21806bc3580207aa0ad4e2759151e9827/hopper/flash_fwd_launch_template.h#L160-L230) SM90 kernel은 Q/K/V tile movement, score, mask/softmax와 output pipeline을 수행한다.
- [FlashAttention SM90 forward](https://github.com/vllm-project/flash-attention/blob/28e862d21806bc3580207aa0ad4e2759151e9827/hopper/flash_fwd_kernel_sm90.h#L320-L470) 링크는 capability와 state producer를 증명한다.
- 실행 request가 어떤 template를 탔는지는 binary symbol/dispatch trace가 증명한다.
- source link를 runtime log처럼 쓰지 않는다.

- FlashInfer 쪽 call graph는 plan을 별도 소유자로 둔다.
- SGLang backend가 forward mode와 batch metadata를 보고 extend 또는 decode wrapper를 고른다.
- [SGLang FlashInfer extend/decode calls](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_backend.py#L1244-L1465) wrapper plan은 qo/kv indptr, page indices, last-page lengths, heads, dimensions, dtype와 workspace를 native plan 계약으로 바꾼다.
- run은 plan state와 Q/KV/output tensors를 소비한다.
- variable splits라면 partial producer 뒤 cascade merge consumer가 있다.

두 call graph를 같은 열로 쓰면 차이가 선명하다. `logical input`은 공통이지만 `metadata adapter`, `plan or
launch parameters`, `selected symbol`, `partial state layout`, `merge`, `output/LSE contract`는 native다. 공통
interface를 설계하려면 각 칸을 변환해야지 한 backend의 내부 DTO를 다른 backend에 전달해서는 안 된다.

small fixture에 paged 주소를 실제로 붙인다. K/V row digests를 logical position별로 `k0,v0`부터 `k3,v3`이라
하고 physical page7에는 positions0,1, page3에는 2,3을 저장한다. query와 QK scale을 정해 scores가 `[1,2,3,0]`
이 되게 한다. loader checkpoint는 read physical sequence `[7:0,7:1,3:0,3:1]`이어야 한다. page indices를
byte offset으로 오해해 `[7:0,7:1,7:2,7:3]`을 읽으면 memory-safe할 수 있지만 score2부터 갈라진다.

last-page valid1이면 position3은 valid set에서 제외된다. full reference mass는 `e^-2+e^-1+1≈1.503214`,
weighted sum은 `10e^-2+20e^-1+30≈38.71093`, output 약 25.7521, LSE 약 3.40761이다. backend output이
26.2089라면 full four-position set을 사용한 것이고, 약 17.31이라면 first page만 사용한 가능성이 있다.
이 fingerprint는 root cause를 확정하지 않지만 어느 valid set을 의심할지 빠르게 좁힌다.

causal mask fixture도 둔다. query position2라면 positions0~2만 valid이고 위 three-score reference와 같다.
window size2라면 positions1~2만 valid해 mass `e^-1+1`, output `(20e^-1+30)/(e^-1+1)≈27.3106`이다. causal과
window convention이 다르면 plausible하지만 다른 output이 나온다. online softmax 문제처럼 보이지만 valid
score set owner는 metadata/mask다.

scale fixture는 scores before scale `[2,4,6,0]`, scale0.5로 원래 `[1,2,3,0]`을 만든다. adapter와 kernel이
모두 scale을 적용하면 `[0.5,1,1.5,0]`이 되어 output이 더 평평해진다. 둘 다 적용하지 않으면 `[2,4,6,0]`
으로 더 sharp해진다. source에서 raw QK, params scale과 exp2 conversion을 따라 한 번만 적용되는지 본다.

softcap은 선형 scale과 다르다. transformation이 score magnitude를 비선형으로 제한하므로 reference에서 같은
순서로 적용한다. 한 backend가 capability가 없어 fallback하고 다른 backend가 native softcap path를 쓰면
selected backend가 이미 달라질 수 있다. configured label로 두 결과를 같은 cohort에 넣지 않는다.

수치 안정성 stress는 `[80,79,0,-80]`처럼 naive exp가 overflow하기 쉬운 scores를 쓴다. stable max80을 빼면
`[0,-1,-80,-160]`이라 mass는 약 1.367879다. tile0 `[80,79]` 뒤 tile1의 max가 낮으면 old state rescale factor는
1이어야 한다. running max를 tile-local max로 덮으면 old mass를 잘못 줄인다. tile 순서를 뒤집어도 tolerance
안에서 같은 결과가 나와야 한다.

반대로 첫 tile max0, 둘째 max80이면 old mass/accumulator를 `exp(-80)`로 거의 지운다. accumulator dtype이
너무 낮거나 rescale 순서가 틀리면 underflow 자체보다 잘못된 old contribution이 남을 수 있다. full reference,
tile order A/B와 split merge 세 결과를 비교한다. bitwise equality가 아니라 bounded numerical contract다.

fully masked row에서는 score load 자체가 garbage여도 mask 이후 valid mass0을 만들어야 한다. 하지만 OOB
load를 mask 산술로 정당화하지 않는다. address safety와 mathematical mask identity는 별 gate다. kernel이
safe clamp로 valid row를 읽고 predicate로 contribution을 없애는지, 아니면 truly predicated load인지 source와
artifact를 본다.

plan amortization을 workload로 계산한다. shape cohort가 100 decode steps 동안 동일하고 cold plan30µs,
run40µs, merge8µs라면 step 평균 plan은 0.3µs여서 total48.3µs다. ragged arrival로 매 step replan하면 78µs다.
FlashAttention path가 launcher+kernel60µs라면 stable cohort에서는 FlashInfer가 빠르고 ragged cohort에서는
느리다. plan cache hit율 하나가 아니라 safe reuse predicate와 actual amortized time을 보고한다.

plan reuse의 key에는 active shape만이 아니라 capacity/layout에 영향을 주는 fields가 필요하다. qo/kv lengths,
heads, dimensions, page size, dtype, causal/window, split/workspace policy와 graph mode 중 어떤 것이 native plan을
바꾸는지 source를 본다. 불필요한 field까지 넣으면 replan이 늘고, 필요한 field를 빼면 stale work mapping을
재사용한다. performance와 correctness가 같은 key schema에 걸린다.

workspace sizing incident도 수치로 확장한다. 앞의 active partial10 fixture에 concurrent requests64가 각각
private 41,280 byte를 쓰면 약 2.52MiB로 작다. 그러나 rows가 batch256, splits8, heads32, Dp256이면 partial O만
`256×8×32×256×4=64MiB`, LSE 약 256KiB다. graph capacity 두 세대와 double buffer가 겹치면 수백 MiB가 된다.
작은 fixture 결과를 large production shape에 선형으로 오해하지 않는다.

shared workspace면 request별 곱 대신 maximum concurrent active capacity와 serialization contract를 본다.
두 streams가 같은 offsets를 동시에 쓰지 않는지, plan allocator가 slice를 분리하는지 확인한다. arena size가
충분해도 overlapping lifetime이면 wrong output다. OOM이 없다는 것은 isolation 증거가 아니다.

performance counter는 expected work로 정규화한다. QK valid pairs, value dimension, bytes/pages, split partial
bytes와 merge FLOPs를 기록한다. wrong last-page length로 valid pairs를 15개 덜 계산한 kernel이 빠른 것은
optimization이 아니다. output parity 뒤에만 bandwidth, occupancy, tensor-core utilization과 duration을 비교한다.

prefill/decode 비교도 phase denominator를 지킨다. prefill 128 queries×128 keys는 16,384 score pairs/head,
decode 1×8192는 8,192다. pair 수만 보면 prefill이 두 배지만 query tile parallelism, causal valid half, KV
reuse와 launch/grid가 다르다. kernel µs를 pair 수로 나눈 값과 TTFT/ITL을 함께 본다.

selector가 split을 늘릴 때 expected mutation은 CTA supply 증가, partial workspace/merge 증가와 local interval
shortening이다. observed splits가 늘었는데 CTA/grid가 같으면 native policy나 metric 해석을 다시 본다.
workspace와 merge가 늘지 않았다면 다른 kernel 내부 split일 수 있다. source terminology를 공통 model에
억지로 맞추지 않는다.

incident containment은 세 안전 경로를 선택할 수 있다. affected page shapes만 contiguous recompute fallback,
split을 1로 제한, 또는 이전 FlashAttention backend로 cohort를 돌린다. 어떤 경로도 root fix 완료는 아니다.
fallback이 output correctness와 SLO를 지키는지 확인하고 affected capability gap을 명시한다.

원인 수정 뒤 vertical trace는 다음 순서로 닫힌다. request logical lengths와 valid set, native metadata digest,
plan/selector generation, physical page sequence, local scores/mask, partial O/LSE, merge output/LSE, final output
digest와 consumer token이다. first divergence 앞 stages가 정상임을 보여야 downstream 수정의 근거가 된다.

release diff에서는 pinned commits와 build artifact를 보존한다. FlashInfer Python package와 compiled kernels,
SGLang adapter, vLLM/FlashAttention fork가 서로 다른 revisions일 수 있다. Python source만 최신이고 `.so`가
옛 ABI면 plan layout mismatch가 난다. module version, binary digest와 symbol을 trace cohort에 둔다.

rollback terminal에는 mixed generation이 없어야 한다. 새 adapter metadata로 plan된 request가 old native
module에 들어가지 않고, old graph가 new workspace layout을 replay하지 않는다. in-flight count0, partial
workspace refs0, output generation terminal과 cache/page refs baseline을 확인한 뒤 admission을 연다.

최종 선택 기록은 winner 한 줄이 아니다. “Hopper, fp16, Q128 prefill cohort는 pinned FlashAttention path가
plan overhead 없이 SLO를 만족한다. ragged Q1/KV8K decode cohort는 FlashInfer split4가 plan reuse≥95%에서
유리하다. softcap unsupported shape는 verified fallback을 쓴다”처럼 capability와 workload 조건을 쓴다.

독자가 다른 library를 만나도 같은 절차를 쓸 수 있다. exact attention 수학, metadata/page valid set,
plan/dispatch, partial state/merge, workspace/lifetime과 selector를 찾는다. 함수 이름이 달라도 네-score와 page2,
uneven/empty split fixture를 대입해 first divergence를 좁힌다. 이것이 제품 이름보다 오래가는 독법이다.

한 운영 trace를 마지막으로 완주하자. 요청 R은 decode Q=1, KV=4097, page size16, physical pages257개,
last-page valid1, heads8, dimension128, fp16이다. selector는 FlashInfer paged decode wrapper를 골랐고 plan은
splits3, partial rows3×8을 만들었다. workspace generation21, KV block-table generation44, output generation9다.
expected valid score count는 정확히 4097이다.

관측된 output은 reference와 크게 달랐고 non-finite는 없었다. configured backend, plan과 run은 모두 성공했다.
page boundary checkpoint에서 logical position4096이 expected physical page last slot0이 아니라 이전 page slot0을
읽었다. last-page length 자체는 1로 맞았다. page indices array의 마지막 entry가 adapter 변환에서 누락돼 native
loader가 previous page ID를 재사용한 것이다. 모든 pointer는 valid여서 sanitizer와 OOM metric은 조용했다.

이때 softmax local max와 mass는 잘못된 score set에 대해 안정적이었다. split2 partial O/LSE가 fp32 reference와
다르지만 그 split 안에서 recurrence invariant는 맞았다. merge도 partials를 정확히 합쳤다. final output만 보고
online softmax를 고치려 했다면 정상 수학을 망가뜨렸을 것이다. first divergent page row가 조사 owner를
metadata adapter로 되돌렸다.

adapter 수정 뒤 output parity는 돌아왔지만 ITL p99는 여전히 baseline보다 높았다. timeline에서 plan 28µs,
run39µs, merge11µs였고 ragged batch 때문에 plan cache가 매 step miss했다. 이전 benchmark는 same-shape plan을
warmup에서 제외해 run39µs만 보고했다. FlashAttention baseline attention stage62µs와 비교하면 실제 78µs가
느렸다. correctness와 performance는 서로 다른 first divergence였고 한 수정으로 동시에 닫히지 않았다.

plan key를 고칠 때 active request IDs를 넣어 cache hit를 포기하지 않는다. native work mapping을 바꾸는 bounded
shape/layout fields만 key로 삼는다. capacity 안에서 active indptr payload를 update할 수 있는지 source contract를
확인한다. plan reuse가 95%로 올라 평균 plan contribution이 1.4µs가 되자 total51.4µs로 baseline보다 좋아졌다.
이는 fixture 수치이며 다른 shape의 보편 우위를 뜻하지 않는다.

동시에 workspace arena를 관측했다. plan cache entry가 old generation auxiliary buffers를 pin해 live workspace가
예상보다 두 배였다. eviction 때 device completion을 기다린 뒤 refs를 내려야 했다. cache hit를 높이면서 stale
plan lifetime을 늘리면 OOM 위험이 생길 수 있다. plan cache capacity, live bytes와 generation-safe eviction을
같이 tuning한다.

slow consumer나 request cancel도 attention lifetime과 만난다. run이 끝나 partials가 ready지만 merge 전에
request가 취소되면 output은 publish하지 않아도 merge job을 취소할지 drain할지 결정해야 한다. 이미 launch된
GPU work가 workspace를 참조하면 즉시 free하지 않는다. completion 뒤 generation을 확인하고 refs를 회수한다.
다음 request가 같은 workspace slice를 쓸 때 old merge가 덮지 않아야 한다.

graph capture cohort에서는 plan output pointer와 maximum shapes가 capture contract에 들어갈 수 있다. active
length만 바뀐 replay가 valid한지, page indices content update가 capture 전에 완료되는지 본다. selector가 eager
fallback을 택하면 attention kernel duration 외 launch와 scheduling이 달라진다. graph hit율과 backend hit율을
같은 metric으로 합치지 않는다.

수치 tolerance를 release gate로 만들 때 dtype별 threshold 하나만 두지 않는다. ordinary rows, large-range,
causal/window boundary, split/empty와 long-KV rows의 maximum absolute/relative error distribution을 둔다. LSE도
검증한다. output cancellation으로 우연히 맞는 경우 LSE divergence가 state bug를 드러낼 수 있다. downstream
logits tolerance와 greedy token parity를 추가하되 내부 attention 기준을 대체하지 않는다.

performance gate는 cold/warm plan, contiguous/paged, split1/multi, prefill/decode와 graph/eager를 구분한다.
selected symbol과 valid pair count, workspace bytes, run/merge launches를 붙인다. throughput 평균 하나로 rare
page wrong-output와 decode ITL tail을 가리지 않는다. production arrival distribution으로 weighted result를
계산하고 각 hard correctness fixture는 weight와 무관하게 통과해야 한다.

source anchor가 가리키는 범위도 기록한다. Python adapter link는 argument construction과 capability branch를,
launcher link는 template dispatch를, kernel link는 tile pipeline을, cascade link는 merge contract를 증명한다.
논문은 algorithm intuition과 recurrence를 지지한다. runtime selected path, exact counters와 performance는 trace와
artifact evidence다. 서로의 역할을 바꾸지 않는다.

업그레이드 때는 old/new call graph를 같은 열로 diff한다. option default, wrapper class, plan input schema,
workspace sizing, last-page convention, native symbol, partial O/LSE layout와 merge base가 바뀌었는지 본다.
소스 행 번호 이동은 semantic 변화가 아니고 같은 함수명이 유지돼도 DTO field unit이 바뀌면 큰 변화다.

최종 incident terminal은 여섯 증거다. 모든 valid logical positions가 정확히 한 physical row를 읽는다. online
및 split recurrence가 fp32 reference tolerance를 지킨다. workspace producer/consumer generation과 stream
ordering이 맞는다. selector가 expected capability branch와 symbol을 고른다. plan 포함 stage latency와 serving
SLO가 baseline/guardrail을 만족한다. cancel·rollback 뒤 refs와 live buffers가 기준값으로 돌아온다.

이 여섯 가지 중 output parity만 통과하면 성능과 lifetime이 미완료다. kernel speed만 통과하면 wrong work일
수 있다. resource cleanup만 통과해도 metadata와 numerical contract는 남는다. 한 vertical trace에서 모두
설명해야 backend 교체를 완료로 선언할 수 있다.

독자가 직접 source breakpoint를 잡는 순서도 남긴다. 첫째 adapter 직후 logical Q/KV lengths, block/page
metadata와 scale/mask를 본다. 둘째 plan 또는 launcher가 만든 splits, work ranges, workspace offsets와 selected
traits를 본다. 셋째 loader가 읽은 first/last/page-boundary row generation을 본다. 넷째 score/mask와 local
online state를 본다. 다섯째 partial store와 merge input/output을 본다. 여섯째 final consumer와 cleanup을 본다.

page boundary에서 처음 다르면 plan 이후 softmax breakpoint는 잠시 건너뛴다. scores까지 맞고 local mass가
다르면 online reduction/scale/exp를 본다. partial O/LSE까지 맞고 global만 다르면 merge indptr, layout와
log base를 본다. final attention까지 맞고 logits가 다르면 attention backend 밖의 residual/model path로
이동한다. 이 순서가 조사 범위를 빠르게 줄인다.

성능 breakpoint는 별도다. selector 결정 시각, plan queue/start/end, native launch, kernel start/end, partial
completion, merge와 output-ready를 잇는다. `plan`이라는 CPU 범위가 native 내부 device planning을 포함하는지
측정 scope를 확인한다. CUDA event와 CPU monotonic timestamps를 직접 빼지 않고 각 duration과 causal edge를
보존한다.

작은 four-score fixture는 성능 benchmark가 아니다. 수학과 state layout을 사람이 검산하는 correctness 도구다.
성능은 실제 head/dtype/page/context/batch와 selected artifact에서 측정한다. 반대로 production benchmark만으로
수학을 검증하지 않는다. 작은 deterministic test와 큰 workload measurement가 서로 다른 질문을 맡는다.

wrong-output 사고의 영향 범위는 backend flag가 켜진 모든 request가 아닐 수 있다. paged last-page partial,
split≥2, 특정 adapter schema generation과 selected symbol의 교집합을 trace에서 찾는다. contiguous/fallback
cohort를 불필요하게 오염으로 선언하지 않되 evidence retention이 부족하면 보수 범위를 택한다. customer-facing
impact 판단과 source 가능성은 구분한다.

임시 fallback 중에도 관측을 유지한다. split1로 제한하면 split merge 결함은 보이지 않지만 page loader는
계속 검증할 수 있다. contiguous recompute면 output은 보호되지만 paged contract 증거를 얻지 못한다. 이전
backend rollback이면 새 adapter/plan path가 실행되지 않는다. 어떤 failure surface를 가렸는지 incident 문서에
남긴다.

root fix canary는 가렸던 모든 축을 다시 연다. partial page, page permutation, uneven/empty split, large scores,
ragged replanning, graph/eager, cancellation과 workspace churn이 포함된다. 평균 traffic에서 드물다는 이유로
생략하지 않는다. fault injection과 정상 workload가 같은 code generation을 쓰는지 확인한다.

마지막으로 선택을 재평가할 trigger를 둔다. library/driver/toolkit upgrade, model head/layout/dtype 변경,
page size나 scheduler split policy, graph capacity와 plan cache key 변화가 trigger다. 새 backend 이름이 추가돼도
이 장의 동일 fixture와 call-graph 열을 적용한다. 제품명 암기 대신 계약 diff가 재감사의 기준이다.

최종 승인 회의에서는 네 질문에 답한다. 같은 logical valid score set을 두 backend가 정확히 한 번씩 덮었는가.
online 또는 split state의 dtype·normalization·log base와 merge identity가 producer/consumer 사이에서 일치하는가.
plan/workspace/page/output generation이 run과 merge completion까지 보호되는가. plan과 fallback을 포함한 실제
serving cohort에서 SLO와 capacity가 좋아졌는가. 하나라도 빠지면 교체는 실험이지 완료가 아니다.

답은 표의 체크 표시가 아니라 R request trace로 제시한다. score set, page sequence, selected symbol, splits,
workspace bytes, partial/global LSE, output digest와 terminal refs를 시간순으로 잇는다. 기각한 가설과 fallback이
가린 surface도 남긴다. 이 기록이 있으면 다음 upgrade에서 final token이 달라진 뒤 모든 kernel을 다시 읽지
않고 처음 갈라진 계약으로 바로 돌아갈 수 있다.

특히 “backend 교체”라는 변경 이름을 금지할 필요는 없지만 그것만으로 리뷰를 끝내지 않는다. adapter DTO,
page convention, plan key, workspace generation, native artifact와 merge state가 각각 유지·변경·미지원 중
무엇인지 적는다. 변경되지 않은 칸도 source와 trace로 확인한다. 이 diff가 rollback 대상과 cache/graph
무효화 범위를 결정한다.

근거 없는 동일성은 허용하지 않는다. 계약별로 검증한다.

최종 회고는 한 문장으로 압축된다. FlashAttention과 FlashInfer는 stable exact attention이라는 수학적 목표를
공유할 수 있지만 plan/dispatch, paged metadata, split workspace, merge, selector와 lifetime 계약은 서로
대체 가능한 이름이 아니다. 동일 fixture를 logical input부터 final consumer까지 각 native call graph에 맞게
통과시켜야 비교와 교체가 안전하다.

## 45.14 Reference — API·설치·고정 source note

아래 근거는 attention 수학, Python plan 계약, native launch와 merge 구현을 서로 다른 층으로 나누어 읽기 위한 고정 좌표다.

### 출력이 reference와 다른가

같은 Q/K/V와 mask에서 output 또는 LSE가 처음 달라질 때는 여기서 시작한다. Exact-attention 수학과 work partitioning의 보존식을 먼저 고정한다.

- [FlashAttention 논문 v2 — Tri Dao 외, IO-Awareness 기반 exact attention](https://arxiv.org/pdf/2205.14135v2)
- [FlashAttention-2 논문 v1 — work partitioning 개선](https://arxiv.org/pdf/2307.08691v1)
- [FlashAttention-3 논문 v2 — Hopper 비동기성과 low-precision](https://arxiv.org/pdf/2407.08608v2)

### 선택된 kernel과 launch가 예상과 다른가

Python backend는 맞아 보이지만 effective symbol, grid 또는 warp 역할이 다를 때는 여기서 시작한다. Wrapper에서 launch template와 SM90 kernel까지 선택 사슬을 잇는다.

- [vLLM v0.27.1 — FlashAttention Python backend varlen path](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/flash_attn.py#L743-L1065)
- [vLLM FlashAttention current pin — forward launch template](https://github.com/vllm-project/flash-attention/blob/28e862d21806bc3580207aa0ad4e2759151e9827/hopper/flash_fwd_launch_template.h#L160-L230)
- [vLLM FlashAttention current pin — SM90 forward kernel](https://github.com/vllm-project/flash-attention/blob/28e862d21806bc3580207aa0ad4e2759151e9827/hopper/flash_fwd_kernel_sm90.h#L320-L470)

### Paged/split plan과 merge가 어긋났는가

Partial page, uneven split 또는 replanning 뒤에만 오답이나 workspace 문제가 날 때는 여기서 시작한다. Plan metadata와 실행 call, partial-state merge를 같은 generation으로 읽는다.

- [SGLang v0.5.18 — FlashInfer MLA plan metadata](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_mla_backend.py#L830-L879)
- [SGLang v0.5.18 — FlashInfer extend/decode calls](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_backend.py#L1244-L1465)
- [FlashInfer v0.6.17 — Python plan contract](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/attention/_core.py#L94-L213)
- [FlashInfer v0.6.17 — paged prefill split and merge launches](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/attention/prefill.cuh#L4210-L4260)
- [FlashInfer v0.6.17 — variable-length merge states](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/attention/cascade.cuh#L650-L720)
