# 12장. 행렬 optimizer와 최신 계보

11장의 좌표별 AdamW state를 행렬 구조와 비교하고, 13장의 token clock·learning-rate schedule이 preconditioner 재사용·warmup에 미치는 영향을 받는다. 14장은 Newton–Schulz·orthogonalization·moment update가 저정밀·fused kernel에서 안정하게 누적되는지 검증한다.

AdamW는 각 좌표에 쌓인 1차·2차 moment로 그 좌표의 보폭을 정한다. 반면 linear layer의 weight는 단순히 긴 벡터로 펴도 되는 저장 공간인 동시에, 입력 특징 공간을 출력 특징 공간으로 보내는 선형사상이다. 행렬 optimizer는 이 두 번째 사실을 이용한다. 행과 열 사이의 관계, singular direction, 좌우 공간의 곡률을 update에 반영한다. 그 대가로 “2차원 tensor면 적용한다”보다 훨씬 엄격한 정의역, 더 복잡한 수치 경계, state 소유권과 collective 계약을 요구한다.

이 장을 읽는 중심선은 다음 한 줄이다.

> `gradient matrix → momentum/statistic → matrix transform → shape scale → 분산 소유·통신 → parameter update → state commit`

어느 구현을 보더라도 먼저 이 일곱 칸을 채운다. 그래야 `ns_steps=5` 같은 옵션을 성능 knob로만 보지 않고, 어떤 tensor가 몇 번의 matrix multiplication을 거쳐 어떤 update로 바뀌는지 설명할 수 있다. checkpoint 역시 optimizer 이름 하나를 저장하는 파일이 아니다. transform 이전의 momentum, transform을 정의하는 coefficient와 dtype, parameter stack과 owner map, committed step을 함께 복원해야 다음 update가 같아진다.

### 이 장의 독자 지도: 한 matrix가 실제로 이동하는 경로

`W∈R^{m×n}`가 hidden-to-hidden projection이고 backward가 logical gradient `G_t`를 만들었다고 하자. Muon 계열의 전형적인 경로는 다음과 같이 펼쳐진다.

1. **정의역을 판정한다.** `W.ndim==2`만 확인하지 않는다. embedding, tied LM head, router처럼 모양은 행렬이지만 lookup·selection 의미를 가진 tensor를 분리한다. tensor parallel shard라면 현재 rank의 `[m,n/p]` 조각을 변환할지, logical full matrix를 모아 변환할지도 이때 결정한다.
2. **gradient의 의미를 닫는다.** loss reduction, accumulation, AMP unscale, clipping, data-parallel reduction 중 어디까지 끝난 `G_t`인지 고정한다. 평균 gradient와 합 gradient는 polar 방향만 놓고 보면 scale이 사라질 수 있지만, epsilon·momentum·decay·clip이 붙으면 더는 같은 경로가 아니다.
3. **시간 상태를 갱신한다.** 구현이 `M_t=μM_{t-1}+(1-μ)G_t`를 쓰는지, 계수 `(1-μ)`를 생략하는지, Nesterov 조합 `G_t+μM_t`를 transform 입력으로 쓰는지 함수에서 확인한다. 이 차이는 learning-rate 단위와 checkpoint state를 바꾼다.
4. **행렬 변환을 수행한다.** 입력 norm을 제한한 뒤 SVD의 polar factor `UVᵀ`를 Newton–Schulz 다항식으로 근사하거나, Shampoo처럼 좌우 Gram statistic의 inverse root를 적용한다. 여기에서 transpose, coefficient, 반복 횟수, accumulator dtype이 실제 알고리즘을 결정한다.
5. **크기를 다시 부여한다.** polar factor는 singular value를 거의 1로 보내므로 raw gradient의 크기를 그대로 보존하지 않는다. 구현은 aspect ratio나 width에 따른 scale을 붙이고 group learning rate를 곱한다. 이 scale을 빼고 AdamW와 숫자 하나로 learning rate를 비교하면 공정한 비교가 아니다.
6. **소유권과 통신을 해결한다.** 각 rank가 local shard를 독립 변환하는지, owner rank가 full matrix 또는 stack을 변환한 뒤 all-gather하는지 구분한다. 같은 `W`라도 전자는 global polar factor와 일반적으로 같지 않다. owner map과 stack offset은 성능 metadata가 아니라 checkpoint 의미다.
7. **parameter와 state를 원자적으로 commit한다.** weight decay와 transformed update의 적용 순서, overflow 시 skip 범위, scheduler clock을 고정한다. parameter만 새 step이고 momentum은 이전 step인 checkpoint는 load에는 성공해도 다음 gradient에서 갈라진다.

이 흐름에서 관측해야 할 최소값도 정해진다. 입력 `G_t`와 transform 입력 `M_t`의 norm·shape·digest, singular-value 표본, Newton–Schulz residual, 최종 update RMS, update/weight ratio, owner별 계산 시간과 collective bytes, `OptimizerStepID`를 같은 사건에 묶는다. loss 하나만 보면 transform 오류, scale 오류, state mapping 오류가 모두 “학습이 불안정하다”로 뭉개진다.

### AdamW와 비교할 때 먼저 같게 만들어야 할 것

AdamW와 Muon을 같은 learning-rate 숫자 하나로 겨루게 하면 비교가 시작되기도 전에 단위가 어긋난다. AdamW는 `m̂/(√v̂+ε)`라는 좌표별 scale을, Muon은 momentum matrix의 singular direction을 평탄화한 뒤 shape scale을 쓴다. 공정 비교의 공통 입력은 **같은 parameter snapshot, 같은 logical gradient sequence, 같은 token clock, 같은 clipping·decay 정책**이다. 그 위에서 optimizer마다 사전에 정한 같은 탐색 예산으로 안정 learning-rate 범위를 찾는다.

비교 결과도 네 칸으로 나눈다.

| 판정 층 | 반드시 고정할 것 | 읽어야 할 결과 | 흔한 오판 |
|---|---|---|---|
| 수치 정확성 | parameter·state·gradient replay | delta, 다음 state, residual, finite | 한 step loss가 같으니 state도 같다고 판단 |
| 표본 효율 | data order·valid token budget·평가 cadence | token-to-target, seed 분산 | 가장 좋은 seed만 비교 |
| 시스템 효율 | topology·batch·precision·backend | step time, peak bytes, collective tail | batch 확대 이익을 optimizer 수학의 이익으로 계산 |
| 복구 가능성 | checkpoint 경계·owner map·scheduler clock | resume 뒤 다음 여러 delta | load 성공을 재현 성공으로 간주 |

AdamW는 낡은 대조군이고 Muon은 최신 후보라는 서사를 이 장은 사용하지 않는다. AdamW의 대각 preconditioner는 상태와 kernel 생태계가 잘 정립되어 있고 모든 parameter 역할에 적용하기 쉽다. Muon은 특정 dense matrix에서 행렬 기하를 활용할 수 있지만, embedding·router 같은 정의역과 distributed shard에 대한 결정을 독자에게 되돌려 준다. 어느 쪽이 유리한지는 모델 역할, gradient spectrum, 허용 state·통신 비용, 복구 계약을 동시에 만족하는지로 판정한다.

## 12.1 행렬·곡률·sign optimizer의 상태를 분류한다

optimizer 이름보다 gradient에서 어떤 통계를 만들고 어느 좌표계에서 update를 변환하는지로 family를 분류한다.

### preconditioner가 보는 구조

Shampoo는 행렬 gradient `G`에 대해 좌우 통계 `L←βL+(1−β)GGᵀ`, `R←βR+(1−β)GᵀG`를 쌓고 역행렬근으로 양쪽을 precondition한다. 큰 차원에서는 root 계산과 상태가 병목이므로 block size, grafting, precondition frequency가 실제 알고리즘을 바꾼다. 빈도를 늘리면 curvature 적응은 늦어지고 계산비는 줄어든다.

Sophia는 Hessian 대각 추정과 clipping으로 큰 곡률 방향의 step을 제한한다. Hessian 갱신 주기와 추정 batch가 일반 gradient batch와 다르면 별도의 RNG·cursor가 checkpoint 대상이다. Lion은 moment의 부호로 update하므로 state는 하나지만 update magnitude가 gradient 크기를 직접 반영하지 않는다. 따라서 decay와 lr 튜닝을 AdamW에서 그대로 옮길 수 없다.

## 12.2 Muon의 polar geometry와 직교화를 해부한다

Muon이 행렬 gradient의 방향과 singular spectrum을 어떻게 바꾸는지 polar factor와 Newton–Schulz 반복으로 읽는다.

### 왜 직교화하는가

Muon 계열은 보통 raw gradient가 아니라 momentum matrix `M`의 singular value를 비슷한 크기로 보내는 polar 방향을 update에 사용한다. `M=UΣVᵀ`라면 기준이 되는 polar factor는 `UVᵀ`다. `σ₁≫σ₂`인 경우 raw update는 첫 singular direction에 대부분의 보폭을 쓰지만, polar 방향은 nonzero singular direction마다 단위 크기를 부여한다. “큰 방향을 자르고 작은 방향을 살린다”는 직관은 여기까지는 맞다. 다만 작은 singular value가 noise라면 그 방향까지 상대적으로 키울 수 있으므로, 직교화 자체가 유용성을 보장하지는 않는다.

왜 이런 방향이 수학적으로 자연스러운가. 작은 update `Δ`에 대해 `L(W+Δ)≈L(W)+⟨G,Δ⟩`로 선형화하고 update의 크기를 어떤 norm으로 제한하면, 가장 가파른 방향은 그 norm의 dual norm으로 정해진다. Frobenius norm ball에서는 정규화된 raw gradient가 나오지만, matrix의 spectral norm과 nuclear norm을 사용하면 singular vector가 중심이 된다. polar 방향은 행렬을 가장 가까운 orthogonal 또는 semi-orthogonal map으로 보내는 문제와도 연결된다. 따라서 이것은 “Adam보다 더 정확한 gradient”가 아니라 **update budget을 좌표가 아니라 singular direction에 배분하는 다른 기하**다.

rectangular matrix에서는 말을 더 조심해야 한다. `m≥n`이고 full column rank라면 `Q=UVᵀ`에 대해 `QᵀQ=I_n`이지만 `QQᵀ=I_m`일 수는 없다. wide matrix는 반대다. 그래서 residual은 무조건 `||XᵀX-I||`를 쓰지 않고 작은 차원의 orthogonality를 검사해야 한다. rank-deficient matrix의 null space에서는 exact polar factor 선택이 유일하지 않을 수 있다. production의 finite polynomial output을 SVD oracle과 원소별 exact equality로 판정하면 올바른 구현도 실패시킬 수 있다.

Newton–Schulz 반복은 명시적 SVD 대신 accelerator가 잘 수행하는 matrix multiplication으로 polar 방향을 근사한다. 이 장에서 고정한 계열은 `X_{k+1}=aX_k+bX_kX_kᵀX_k+cX_kX_kᵀX_kX_kᵀX_k` 꼴의 다항식을 쓴다. scalar singular value 관점에서는 각 `σ`에 같은 다항 map을 반복 적용해 목표 구간으로 보낸다고 읽을 수 있다. 입력을 먼저 norm으로 안정 영역에 넣지 않으면 큰 singular value가 다항식의 발산 구간으로 나갈 수 있다. 반대로 norm이 지나치게 작은 입력과 BF16 반올림에서는 약한 singular direction이 0으로 사라질 수 있다. 반복 횟수는 정확도 knob인 동시에 matmul 수, rounding 누적, workspace lifetime을 바꾸는 실행 옵션이다.

따라서 `ns_steps`를 늘려 residual 하나가 작아졌다는 이유만으로 개선이라 하지 않는다. 각 반복에서 finite, 작은 차원의 orthogonality residual, output norm, matmul latency를 함께 본다. zero·rank-one·ill-conditioned·tall·wide matrix를 포함하고, FP64 SVD는 작은 fixture의 oracle로만 쓴다. 실제 모델의 장기 수렴은 별도의 behavioral evidence다.

### 함수 경로와 옵션

nanochat은 embedding·head·scalar parameter와 hidden matrix를 구분해 각각 AdamW와 Muon 계열에 맡긴다. Muon의 matrix 변환은 2차원 hidden weight를 전제로 하므로, 이 구분은 단순한 성능 취향이 아니라 optimizer 함수의 정의역이다. 이름 regex나 `ndim`만으로 group을 만들지 말고 최종 resolved parameter ID 목록을 artifact로 남긴다. tied embedding과 LM head가 서로 다른 group에 두 번 들어가면 같은 storage에 두 update가 적용될 수 있다.

고정 소스를 읽을 때 함수 호출을 다음 표처럼 해체한다. 뒤의 12.13과 12.23은 KellerJordan, nanochat, NVIDIA Emerging Optimizers의 고정 commit과 line span을 제공한다. 여기서는 좌표가 무엇을 증명해야 하는지만 먼저 고정한다.

| 함수·옵션 | 실제로 바꾸는 상태/연산 | 직접 관측할 것 | 잘못된 일반화 |
|---|---|---|---|
| momentum coefficient | `G_t`와 이전 buffer가 섞이는 비율 | NS 입력 digest, buffer before/after | 모든 구현이 같은 EMA convention을 쓴다 |
| Nesterov | transform에 들어가는 현재 gradient와 momentum 조합 | transform 직전 tensor | 이름이 같으면 PyTorch SGD와 식도 같다 |
| `ns_steps` | 다항식 반복과 matrix multiplication 횟수 | 반복별 residual·finite·latency | 많을수록 항상 정확하고 안정적이다 |
| coefficient/backend | singular value에 적용되는 scalar map과 dtype 경로 | coefficient digest, selected backend, fallback | fallback도 같은 수치 결과를 낸다 |
| transpose | tall/wide 중 계산 방향과 반환 shape | input/output logical shape, transpose flag | local shard에서도 global shape와 같다 |
| post scale | polar output이 실제 delta 단위로 변환되는 비율 | transform output RMS와 final delta RMS | AdamW와 같은 lr 숫자가 같은 보폭이다 |
| weight decay | parameter에 직접 적용되는 별도 이동 | zero-gradient delta | matrix transform에 포함된 regularization이다 |
| owner/stack | 어느 rank가 어떤 matrix state와 계산을 맡는가 | stable ID, offset, owner, collective sequence | DDP가 알아서 동일 의미를 보장한다 |

`ns_steps`를 바꾸면 matmul 횟수와 직교화 오차가 달라진다. momentum과 Nesterov 선택은 Newton–Schulz에 들어가는 matrix 자체를 바꾼다. update scale은 폭과 aspect ratio에 따른 실제 step 크기를 바꾼다. 이 옵션들을 한꺼번에 변경한 benchmark는 어느 변화가 원인인지 식별할 수 없다. option 하나를 바꿀 때 source digest, resolved config, one-step replay와 peak memory를 함께 새 revision으로 남긴다.

분산 경로에서는 세 가지 질문이 먼저다. 첫째, `G_t`는 이미 data-parallel 평균이 끝난 logical gradient인가. 둘째, transform 대상이 full logical matrix인가 local TP/FSDP shard인가. 셋째, transformed update를 누가 다시 parameter shard에 전달하는가. optimizer 내부에서 reduce-scatter나 all-gather를 수행하면서 DDP reducer도 같은 tensor를 다루면 중복 reduction 또는 collective-order 교착이 생긴다. 반대로 각 shard를 독립 직교화하면 통신은 줄지만 `polar([G_1,G_2])`와 `[polar(G_1),polar(G_2)]`가 일반적으로 다르다. 이것은 최적화가 아니라 알고리즘 선택이므로 recipe에 명시한다.

마지막 함수는 `step()`의 반환이 아니라 commit이다. overflow로 update를 건너뛸 때 momentum도 건너뛰는지, scheduler가 전진하는지, owner 일부가 state를 쓴 뒤 실패하면 전체 step을 되돌릴 수 있는지 확인한다. checkpoint round trip은 parameter equality에서 끝내지 않고 같은 다음 gradient를 두세 step replay해 delta와 state를 비교한다. 이 검사가 통과해야 “동일 optimizer를 재개했다”고 말할 수 있다.

## 12.3 trust ratio·factored state·schedule-free의 대안을 비교한다

LARS·LAMB·Adafactor와 schedule-free를 scale, state byte와 time ownership이라는 공통 질문으로 비교한다.

### 큰 batch와 layerwise trust

LARS/LAMB는 `||θ||/||u||` 비율로 layer별 update를 제한한다. norm이 거의 0인 bias·scale에서 예외 처리가 필요하다. Adafactor는 2차 moment를 row/column factor로 근사해 상태를 줄이지만 모든 tensor에서 정확한 Adam state와 동등하지 않다. relative-step 옵션은 외부 scheduler의 lr 소유권과 충돌할 수 있다.

Schedule-Free 계열은 평균화된 평가 지점과 빠르게 움직이는 학습 지점을 분리한다. `train()`/`eval()` 전환이 단순 dropout 토글을 넘어 어느 parameter view를 노출하는지 바꿀 수 있으므로 checkpoint에는 두 상태와 mode가 함께 들어가야 한다.

## 12.4 tensor 역할과 optimizer state ownership을 고정한다

모든 parameter를 행렬 optimizer에 넣을 수는 없다. dense matrix, vector, embedding과 sparse gradient의 지원 경계를 inventory로 만든다.

### 적용 범위 경계

embedding lookup의 gradient는 방문 row에만 생길 수 있다. 이를 dense matrix처럼 직교화하면 주소 희소성과 token frequency 의미를 지운다. expert parameter도 active token이 없는 rank에서 gradient가 비어 있을 수 있다. parameter group은 `ndim==2` 같은 한 줄 조건이 아니라 역할·희소성·sharding·tied 여부를 함께 판정해야 한다.

### 선택 실험과 handoff

동일 `GoldenBatchID`에서 AdamW와 Muon을 비교할 때 loss만 보지 말고 singular spectrum, update RMS, state bytes, optimizer collective bytes를 기록한다. 1-step parity를 기대해서는 안 된다. 대신 zero gradient, rank-deficient matrix, 매우 작은 norm에서 finite와 scale invariant를 검증한다.

### geometry를 수치로 구분한다

행렬 `G=diag(100,1)`을 생각하자. SGD 방향은 첫 축이 두 번째보다 100배 크다. ideal polar factor는 두 singular direction을 같은 크기로 보내 identity에 가까워진다. AdamW는 각 좌표의 second moment가 충분히 쌓였다면 elementwise scale을 맞출 수 있지만, 좌표계를 회전하면 elementwise 통계의 의미가 달라진다. Shampoo는 `GGᵀ`와 `GᵀG`를 통해 좌우 공간의 상관을 보며, Muon은 momentum matrix의 polar 방향을 근사한다. 이 예는 세 방법이 모두 “정규화”라는 한 단어로 합쳐지지 않는 이유를 보여준다.

Shampoo의 matrix case에서 left state는 `[m,m]`, right state는 `[n,n]`이다. 원 weight가 `[m,n]`일 때 상태 원소 수가 `m²+n²`이므로 큰 square matrix에서 비싸다. block size를 줄이면 root 계산은 싸지지만 block 경계를 넘는 상관을 잃는다. precondition frequency를 낮추면 통계는 매 step 쌓아도 inverse root는 오래된 값을 쓴다. grafting은 Shampoo 방향에 SGD/Adam류 magnitude를 부여해 초반 scale을 안정시키며 어떤 graft를 택했는지 state manifest에 넣는다.

Sophia의 diagonal curvature estimate는 gradient moment와 별도의 cadence로 갱신된다. curvature update batch와 RNG, estimator kind, update interval이 checkpoint state다. coordinate clipping threshold `ρ`는 global norm clip과 다르다. 각 좌표의 preconditioned update를 제한하므로 두 clipping을 함께 쓰면 순서까지 objective dynamics를 바꾼다.

Lion은 하나의 momentum state와 sign update를 사용해 Adam보다 state가 작다. sign은 작은 noise에도 방향이 바뀔 수 있어 weight decay와 lr의 단위가 AdamW와 다르다. Adafactor row/column factorization은 second moment를 exact하게 저장하지 않으며 1D parameter에는 별도 경로가 필요하다. relative-step과 parameter-scale 옵션이 내부 lr을 소유하면 외부 scheduler와 이중 적용하지 않는다.

## 12.5 고정 소스와 test로 구현 주장의 범위를 한정한다

논문의 식, library의 실제 함수와 upstream test가 각각 무엇을 보증하는지 분리해 기록한다.

Muon 기준 좌표는 11장의 Keller commit `f98f1…806`과 nanochat `92d63…bcd`를 그대로 받는다. NVIDIA Emerging-Optimizers commit `83537ba67cb4c998251567f78a534776fecb1965`의 `emerging_optimizers/orthogonalized_optimizers/muon.py:38–144`는 backend 선택과 fallback을, `tests/test_muon_utils.py:231–348`은 Newton–Schulz coefficient·입력 오류 검사를 보여준다. upstream test가 대규모 convergence나 checkpoint reshard를 증명한다고 확대하지 않는다.

optimizer별 test matrix는 zero, rank-one, tall, wide, sparse gradient를 공통 입력으로 쓴다. Shampoo는 root residual과 stale-preconditioner cadence, Sophia는 curvature update 누락·clip, Lion은 zero momentum sign, Adafactor는 factored/unfactored shape를 검사한다. 동일 gradient replay 뒤 update finite, expected state shape, save/load 다음 step을 비교한다.

분산에서는 preconditioner owner가 핵심이다. full matrix state를 모든 rank에 복제할지, block owner가 계산해 gather할지, parameter shard와 같은 mesh에 둘지에 따라 collective bytes가 달라진다. optimizer library가 collective를 소유하면 DDP reducer와 중복되지 않도록 logical gradient ready 시점을 기록한다. owner rank kill을 주입해 partial state update가 commit되지 않는지 본다.

### 선택 결정 트리

메모리가 우선이면 parameter당 state bytes와 temporary root workspace를 계산한다. matrix geometry가 이득일 가능성을 보려면 weight role, aspect ratio, gradient singular spectrum을 본다. lookup·router·norm은 2D 외형만으로 matrix optimizer에 넣지 않는다. 매우 elongated projection은 polar scale이 기대와 다를 수 있어 별도 group ablation을 둔다.

loss가 불안정하면 lr을 먼저 낮추지 않고 update RMS와 orthogonalization/root residual을 확인한다. residual이 나쁘면 norm·dtype·iteration·coefficient를, residual은 좋은데 router만 튀면 적용 범위를, single-rank만 정상이라면 gradient reduction과 state owner를 본다. resume만 다르면 group order, block partition, curvature/root cadence를 확인한다.

공정 비교는 각 optimizer의 안정 lr sweep, 같은 token budget, 같은 parameter grouping 기준으로 수행한다. state 절감으로 batch를 늘린 결과는 fixed-batch 실험과 분리한다. validation loss, wall time, peak memory, collective bytes, update spectrum을 함께 내야 “더 빠른 수렴”이 token 효율인지 system 효율인지 해석할 수 있다.

### optimizer는 어떤 norm에서 가장 가파른가

steepest descent는 좌표계와 무관한 절대 개념이 아니다. 작은 step `Δ`에 대해 1차 근사 `L(W+Δ)≈L(W)+⟨G,Δ⟩`를 쓰고 `||Δ||≤η`라는 제약을 정하면, loss를 가장 많이 줄이는 방향은 선택한 norm의 dual norm으로 결정된다. Euclidean/Frobenius norm 제약에서는 `−G/||G||_F`가 나오고, matrix spectral norm을 제약하면 nuclear norm과의 dual 관계가 등장한다. Muon의 polar 방향을 “gradient 정규화”라고만 부르면 이 norm 선택을 놓친다.

SVD `G=UΣVᵀ`에서 polar factor `UVᵀ`는 nonzero singular direction의 크기를 1로 만든다. full-rank rectangular matrix의 update RMS는 shape에 따라 달라지므로 practical Muon은 폭·aspect ratio scale을 추가한다. rank-deficient matrix에서는 polar factor가 유일하지 않을 수 있고 finite Newton–Schulz polynomial은 exact SVD 결과가 아니다. 이 수학적 경계를 test tolerance에 반영한다.

Shampoo가 accumulation 없이 한 matrix gradient의 좌우 Gram inverse root를 즉시 쓴 이상화는 spectral-norm steepest descent와 연결될 수 있다. 그러나 practical Shampoo는 과거 gradient를 누적한 `L,R`, damping, block partition, root 갱신 주기를 사용한다. 따라서 이론적 한 step 등가성을 practical Shampoo와 Muon의 동일성으로 확대하지 않는다.

## 12.6 inverse root와 trust ratio를 수치 경계에서 검산한다

행렬 분해와 norm ratio가 epsilon, clipping, zero norm과 저정밀에서 어떻게 분기하는지 작은 fixture로 닫는다.

`L_t=βL_{t-1}+(1−β)G_tG_tᵀ+εI`, `R_t=βR_{t-1}+(1−β)G_tᵀG_t+εI`를 유지한다고 하자. preconditioned update는 matrix 차원과 논문 convention에 따라 `L^{-α}GR^{-α}` 꼴이다. inverse root는 eigendecomposition, Newton iteration, coupled iteration 등으로 계산할 수 있고 각 방법은 dtype·수렴 criterion·fallback이 다르다.

damping `εI`는 Adam의 denominator eps처럼 단순 영 나눗셈 방지 이상이다. 작은 eigen direction의 증폭 상한을 정한다. root를 FP64/FP32 host에서 계산하고 accelerator로 보내는 구현과 GPU에서 계산하는 구현은 latency·동기화·checkpoint state가 다르다. root 자체를 저장할지 Gram만 저장하고 재계산할지 resume 비용도 다르다.

block size가 128이면 `[4096,4096]` weight를 수많은 block으로 나눠 state와 root를 만든다. edge block padding과 logical slice mapping을 manifest에 둔다. block order가 checkpoint key 순서에 암묵적으로 의존하지 않게 한다. root 갱신 직전에 process가 죽었을 때 Gram step과 cached root step이 다를 수 있으므로 두 counter를 저장한다.

### SOAP·Adafactor를 eigenbasis에서 읽는다

SOAP는 Shampoo preconditioner의 eigenbasis로 gradient를 회전한 뒤 그 좌표에서 Adam류 통계를 적용하고 다시 되돌리는 관점을 제공한다. eigenbasis 갱신 주기와 Adam moment 주기가 다르므로 state는 basis, eigenvalue/Gram, rotated moment, 각 counter를 포함한다. basis가 바뀔 때 old moment를 새 basis로 어떻게 옮기는지가 구현 의미다.

Adafactor의 row/column factor는 full second-moment matrix를 outer-product 구조로 근사한다. matrix `V`를 row mean과 column mean으로 재구성하므로 arbitrary coordinate correlation을 exact하게 보존하지 않는다. relative step, warmup init, parameter scaling을 켜면 외부 lr가 `None`일 수 있고 optimizer가 parameter RMS에서 effective step을 만든다. Transformers scheduler를 동시에 연결하기 전에 lr owner를 확인한다.

### Sophia의 curvature cadence

Sophia update는 momentum `m`과 diagonal curvature estimate `h`를 사용해 대략 `m/max(h,ε)`를 coordinate-wise clip한다. curvature를 매 step 갱신하지 않고 `k` step마다 별도 estimator batch로 계산하면 forward/backward 비용이 주기적으로 늘어난다. `k`, estimator RNG, 마지막 curvature step은 checkpoint state다.

curvature batch가 training data cursor를 소비하는지 별도 iterator를 쓰는지에 따라 sample lineage가 달라진다. dropout을 켠 estimator와 끈 estimator도 다르다. curvature가 stale한 동안 loss landscape가 급변하면 clip이 잘못된 scale을 쓸 수 있다. cadence sweep은 평균 throughput뿐 아니라 curvature-update step의 tail latency와 loss spike를 본다.

반례 fixture는 `h`가 매우 작은 좌표와 큰 좌표, negative/noisy estimate, curvature update 누락을 포함한다. denominator floor와 clipping 뒤 update가 finite인지, checkpoint resume에서 다음 curvature update가 같은 step인지 확인한다.

### LARS·LAMB trust ratio의 예외

LARS는 layer별 `trust=||W||/(||g||+λ||W||+ε)`를 사용해 update scale을 조정한다. LAMB는 Adam-preconditioned update와 weight norm의 비율을 사용한다. weight norm이 0인 초기 bias, update norm이 0인 frozen/unused parameter에서 trust ratio fallback이 필요하다. ratio clipping 여부도 구현 옵션이다.

큰 batch에서 layer별 update/weight ratio를 안정시키려는 목적이 있지만 모든 layer에 같은 효과가 유리하다는 정리는 아니다. embedding rare row, norm, bias를 exclude하는 규칙을 manifest로 펼친다. DP rank마다 local norm을 쓰면 trust ratio가 달라지므로 logical full parameter/update norm의 owner를 확인한다.

**schedule-free의 두 parameter view**

schedule-free optimizer는 빠르게 움직이는 point와 평균화된 evaluation point를 유지할 수 있다. library가 `optimizer.train()`과 `optimizer.eval()`에서 model parameter view를 전환한다면 module의 dropout mode와 별개다. evaluation 전에 optimizer mode를 바꾸지 않으면 잘못된 point를 평가할 수 있다.

checkpoint에는 두 point, averaging weight, step, 현재 mode를 저장한다. save 직전 eval view로 바꾸고 load 뒤 train view 복원이 빠지면 다음 update가 다른 parameter에서 시작한다. unit test는 train→eval→train roundtrip, save/load 각 mode, evaluation logits를 비교한다.

**공통 state byte 표**

parameter `N`개를 FP32 state로 둔다면 Lion momentum은 대략 `4N`, AdamW 두 moment는 `8N`, master weight까지 포함하면 더 늘어난다. Adafactor matrix state는 row+column 크기로 줄 수 있다. Shampoo는 block Gram과 cached root가 추가되고 shape에 좌우된다. Sophia에는 momentum과 curvature가, Muon에는 momentum과 구현별 second-moment/equilibration state가 필요하다.

이 표는 steady state만이 아니라 root workspace, foreach tensor list, all-gather buffer를 peak에 더한다. sharding factor는 모든 state에 똑같이 적용되지 않는다. replicated scalar/counter와 owner-only block metadata를 분리한다. memory 절감으로 activation batch가 커질 수 있으므로 system experiment에서는 새 batch를 명시한다.

## 12.7 checkpoint migration과 parameter grouping 실패를 주입한다

optimizer 변경은 parameter group, state schema와 owner mapping을 바꾼다. resume에서만 나타나는 오류를 의도적으로 심는다.

optimizer family를 run 중 바꾸려면 어떤 state를 이전할지 선언한다. AdamW moment를 Muon momentum으로 이름만 바꿀 수 없다. 보통 parameter만 이어 받고 optimizer state를 reset한 branch를 만들며 warmup을 재설계한다. state conversion을 제안한다면 수식과 one-step fixture로 검증한다.

Shampoo root object 하나 누락, Sophia curvature counter rollback, Adafactor row/column swap, schedule-free eval point 누락, Muon stack order 변경을 각각 주입한다. loader는 shape만 맞는 silent load를 거부해야 한다. load가 성공했다면 다음 gradient snapshot으로 expected delta를 비교한다.

### 출판용 비교표

비교표의 열은 optimizer family, 적용 parameter 역할, lr ownership, state dtype/bytes, matrix operation, collective owner, checkpoint schema, locally executed test다. convergence 표는 동일 token budget과 별도로 동일 wall-clock을 둔다. hardware·kernel revision이 다른 결과를 한 수치 열에 합치지 않는다.

논문 수치는 조건과 함께 인용하고 이 책의 작은 fixture 결과와 구분한다. 특정 model에서 Muon이 AdamW보다 좋았다는 보고는 모든 router·embedding에 적용하라는 규칙이 아니다. 독자가 자신의 gradient spectrum과 state budget을 측정해 선택할 수 있게 하는 것이 이 장의 결론이다.

### 하나의 4×2 matrix 실습

`W[4,2]`와 gradient `G`를 직접 정해 SGD, AdamW, polar update를 FP64로 계산한다. `G`의 두 column이 거의 평행한 경우와 직교한 경우를 만든다. Frobenius norm은 같게 맞추되 singular spectrum을 다르게 하면 elementwise 통계와 matrix geometry가 무엇을 구분하는지 보인다. SVD reference의 `UVᵀ`, Newton–Schulz 반복별 결과, `||XᵀX−I||`를 표로 쓴다.

같은 fixture를 BF16 Newton–Schulz에 넣어 rounding error를 측정한다. matrix를 1000배 scale해도 입력 normalization 뒤 방향이 얼마나 유지되는지, zero matrix에서 eps가 어떤 결과를 만드는지 본다. tall matrix transpose 경로가 원 shape로 정확히 돌아오는지 검사한다.

Shampoo에는 `GGᵀ`, `GᵀG`를 출력하고 damping별 eigenvalue와 update를 비교한다. block size를 2로 두면 full matrix 결과와 어떻게 달라지는지 본다. Sophia에는 임의 curvature diagonal을 주고 coordinate clip 전후를, Lion에는 momentum 부호가 0 근처 noise에서 바뀌는 반례를 넣는다.

### 실제 model grouping 감사

QKV fused weight, output projection, gate/up fused MLP, router, embedding, LM head, norm을 표본으로 뽑는다. 각 parameter에 “matrix인가”와 “hidden transform인가” 두 boolean을 둔다. 전자만 true인 embedding/table은 Muon 제외, router는 stability ablation 대상, fused submatrix는 split 가능성을 검토한다.

group별 matrix aspect ratio와 numel을 histogram으로 보고 NS/root 비용을 예측한다. 매우 작은 matrix는 launch overhead가 이득을 삼킬 수 있고 매우 큰 square matrix는 Shampoo state가 폭발할 수 있다. optimizer 선택은 논문 headline이 아니라 역할·shape·state budget·kernel 경로의 교집합이다.

**분산 commit protocol**

optimizer 내부 reduce-scatter/update/all-gather가 있다면 gradient-ready event, owner computation, gathered parameter-ready event를 step ledger에 넣는다. all-gather 일부가 실패했는데 다른 group AdamW가 commit되면 parameter groups가 다른 step에 놓인다. 전체 optimizer step을 commit unit으로 볼지 group별 rollback이 가능한지 정한다.

failure fixture는 owner process를 update 직전, state write 뒤, all-gather 중에 죽인다. durable checkpoint는 partial step을 선택하지 않아야 한다. 복구 뒤 같은 gradient snapshot을 재생해 logical full parameter와 state를 reference와 비교한다.

**최종 선택 질문**

선택 전 네 질문에 답한다. 이 parameter가 어떤 map인가, optimizer state와 root workspace가 얼마인가, collective owner가 누구인가, resume에서 mapping을 복원할 수 있는가. 답이 없는 최신 optimizer는 production default가 아니라 격리된 experiment group으로 시작한다.

결과가 좋더라도 적용 범위를 바꾸면 새 optimizer recipe다. hidden matrix 일부를 AdamW로 되돌리거나 router를 제외하면 manifest digest와 benchmark를 새로 만든다. 이 엄격함이 optimizer 이름보다 재현성을 지킨다.

## 12.8 2×2 기하와 회전 반례로 optimizer 차이를 본다

같은 손실 지형을 회전시켜 elementwise preconditioner와 matrix transform의 좌표 의존성을 손으로 비교한다.

같은 quadratic을 좌표축에 정렬한 경우와 45도 회전한 경우를 만든다. AdamW의 diagonal second moment는 선택한 coordinate에서만 대각이므로 두 표현의 trajectory가 달라질 수 있다. full-matrix preconditioner는 회전을 추적할 수 있지만 상태와 inverse 계산 비용을 낸다. Muon의 left/right orthogonal 변환에 대한 성질도 작은 matrix로 확인한다. “모든 reparameterization에 불변”이라고 확대하지 않는다.

gradient matrix의 한 row만 반복적으로 활성화되는 sparse-addressed 반례에서는 polar update가 방문하지 않은 row까지 dense 방향을 만들 수 있다. embedding/codebook을 hidden transform과 분리해야 하는 이유다. expert router처럼 gradient distribution이 highly skewed한 matrix도 dense layer 결과를 그대로 일반화하지 않는다.

### root와 NS 수치 실패를 찾는다

Shampoo Gram의 condition number를 `10²,10⁶,10¹²`로 키우고 damping과 root dtype을 바꾼다. reconstructed identity residual, update norm, finite 여부를 기록한다. eigensolver가 실패하거나 residual threshold를 넘으면 stale root 또는 graft fallback을 사용할지, step을 중단할지 정책을 정한다.

Newton–Schulz 입력 normalization을 제거한 negative test는 반복이 안정 영역을 벗어나는지 보여준다. iteration 수만 늘리는 것이 해결책이 아님을 확인한다. coefficient set, transpose, scale, output renormalization을 artifact config로 고정한다.

### optimizer 교체 branch의 수치 예

AdamW step 10,000 checkpoint에서 Muon branch를 만들 때 parameter는 같지만 Adam moment를 버린다. 첫 Muon momentum은 zero인지 current gradient인지, warmup을 다시 시작하는지 정한다. control branch는 AdamW state를 유지한다. 두 branch의 첫 gradient는 같아야 하고 delta 차이는 새 optimizer 식에서만 와야 한다.

반대로 Muon에서 AdamW로 전환할 때 matrix momentum을 Adam first moment로 복사하는 임의 변환을 기본으로 쓰지 않는다. reset branch와 명시적 conversion branch를 나눠 one-step reference를 제시한다. optimizer 변경 시점을 scheduler clock과 checkpoint parent에 남긴다.

### source–test–주장 표

Keller `muon.py:5–31`은 NS 함수, `34–41`은 update, `44–96`은 분산 class를 직접 뒷받침한다. nanochat `optim.py:274–318,362–417`은 진화형 sharded 경로다. Emerging Optimizers `test_muon_utils.py:231–348`은 coefficient와 오류 test다. 각 좌표가 증명하지 않는 convergence·대규모 fault recovery를 별도 열에 둔다.

Shampoo·Sophia·SOAP·Adafactor도 논문 식, 공식 implementation, unit test, 이 책의 proposed failure test를 네 열로 분리한다. paper ablation을 현재 library default로, unit test를 production convergence로 바꾸어 말하지 않는다.

**최종 복구 리허설**

matrix group 세 개 가운데 두 번째 owner가 state write 뒤 죽는 상황을 만든다. commit marker가 없으면 step 전체를 이전 checkpoint로 rollback한다. partial parameter all-gather가 request에 노출되지 않았는지 hash root를 확인한다. resume 뒤 동일 gradient를 재생해 모든 group delta를 uninterrupted control과 비교한다.

world size 변경은 optimizer별 state planner가 있을 때만 허용한다. planner가 Gram block, Muon stack, parameter logical ID를 새 owner에 매핑하는 table을 출력한다. table이 없거나 중복/누락이면 load 전에 실패한다.

**독자의 선택 보고서**

보고서는 “Muon 사용”이 아니라 적용 matrix 목록과 제외 이유, NS variant, hybrid AdamW group, state bytes, collective owner를 쓴다. Shampoo는 block/root cadence, Sophia는 curvature cadence, Adafactor는 relative-step owner를 쓴다. 선택하지 않은 후보도 memory·stability 이유를 남긴다.

한 page의 decision summary 뒤에는 gradient replay 결과, save/load 결과, distributed result가 있어야 한다. 이 세 검사가 없으면 장기 validation 곡선은 optimizer 기하보다 plumbing 차이를 포함할 수 있다.

**검증 가능한 종료 조건**

이 장의 종료 조건은 optimizer를 선택했다는 문장이 아니다. 모든 trainable parameter가 하나의 semantic group에 속하고, 각 group의 update 식·state schema·owner가 명시되어야 한다. 저장 직전과 load 직후 state mapping이 같고 동일 gradient의 다음 delta가 허용오차를 만족해야 한다.

행렬 optimizer group은 zero/rank-one/tall/wide fixture와 실제 layer 표본을 모두 통과한다. NS/root residual, finite, update RMS를 기록하고 fallback 발생을 숨기지 않는다. embedding·router·norm 제외 규칙은 이름 regex가 아니라 resolved parameter 목록으로 보존한다.

분산 run은 single-rank logical gradient와 update, collective sequence, owner failure recovery를 검증한다. state bytes는 steady tensor뿐 아니라 temporary workspace와 gather buffer를 포함한다. 비교 실험은 fixed token과 fixed wall-clock을 나누고 lr sweep 범위를 공개한다.

**마지막 반례: 같은 loss, 다른 state**

두 optimizer가 한 step 뒤 우연히 같은 parameter를 만들더라도 state가 다르면 다음 step은 갈라질 수 있다. 첫 gradient만 비교하는 test에 서로 다른 second moment·curvature를 심고 두 번째 gradient에서 divergence가 검출되는지 본다. checkpoint parity는 parameter parity보다 강해야 한다.

반대로 state hash가 달라도 parameter order나 serialization layout 차이일 수 있다. logical parameter/state ID별 tensor를 비교해 의미 차이와 byte layout 차이를 구분한다. 이 기준이 optimizer migration과 reshard 판정의 기초다.

**실험 카드**

실험 카드에는 hypothesis, 적용 group diff, gradient snapshot, lr search, state/collective budget, success/failure threshold, 실행 등급을 쓴다. 결과 뒤 threshold를 바꾸지 않는다. upstream test와 local run을 구분한다.

이 카드와 ParameterGroupManifest가 함께 있어야 독자는 논문 optimizer를 자신의 모델 역할과 cluster 제약에 이식할 수 있다.

**구현 고정점 검증 절차**

소스 좌표는 파일이 존재한다는 증명이 아니라 특정 revision의 동작을 다시 찾는 열쇠다. KellerJordan/Muon `f98f1cacc0263b04290753e32be8d498c1efc806`, `muon.py`, `zeropower_via_newtonschulz5`, 5–31행을 열어 transpose·BF16·normalization·다섯 반복을 확인한다. 같은 파일 `Muon`, 44–96행에서는 parameter owner와 all-gather를 분리해 읽는다.

nanochat `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`, `nanochat/optim.py`, `_reduce_muon`, 274–318행과 `_compute_muon`, 362–417행은 reduce-scatter, sharded state, gather가 기준 구현과 다른 지점이다. 독자는 함수명 검색 결과가 아니라 commit의 blob과 line digest가 맞는 checkout에서 읽어야 한다.

이 좌표를 upgrade할 때는 old/new function의 parameter scope, state key, collective 순서를 diff한다. 이름이 같아도 semantics가 바뀌면 새 recipe와 checkpoint schema를 만든다.

**마지막 invariant**

동일 gradient replay에서 공통 AdamW control group은 reference delta와 같고 matrix group은 선택한 optimizer reference와 맞아야 한다. save/load 다음 step도 같은 판정을 반복한다. state byte, collective byte, temporary workspace의 측정 합이 allocator peak를 설명하지 못하면 숨은 buffer를 trace한다.

parameter role이 바뀌거나 module fusion이 달라지면 grouping manifest digest가 바뀌어야 한다. digest가 같은데 resolved parameter 목록이 다르면 canonicalization bug다. 이 invariant를 model upgrade gate에 둔다.

upstream 소스 revision, local fixture 결과, 실행하지 않은 대규모 실험을 표에서 분리한다. 이 구분이 지켜질 때 optimizer 선택은 재현 가능한 engineering decision이 된다.

최종 검토자는 manifest, source 좌표, unit test, distributed replay, recovery report가 같은 optimizer recipe digest를 가리키는지 확인한다. 하나라도 다른 revision이면 비교를 다시 분리한다.

**사례 연구: matrix optimizer 후보를 기하와 시스템 양쪽에서 검증한다**

사례의 후보는 hidden transform matrix에 Muon 계열을 적용하고 embedding, normalization, bias, router에는 AdamW를 유지하는 hybrid recipe다. 질문은 “새 optimizer가 최신인가”가 아니라 동일한 gradient에서 어떤 singular direction을 어떻게 바꾸며, 그 state와 collective 비용이 장기 run에서 얻는 이익보다 작은가다. 수학, 구현, cluster의 세 판정을 독립 endpoint로 둔다.

첫 artifact는 parameter role table이다. attention projection, MLP gate/up/down, fused QKV, MoE expert를 matrix 후보로 열거한다. embedding과 tied LM head는 2차원이어도 lookup/output role이므로 control AdamW에 둔다. router는 작은 matrix지만 gating distribution을 직접 바꾸므로 별 ablation 없이는 포함하지 않는다. shape-only 위험 대조군을 만들어 semantic grouping의 효과를 검증한다.

**2×2 기하를 손으로 계산한다**

gradient `G=diag(4,1)`의 SVD polar factor는 identity다. matrix orthogonalization은 큰 singular value 4와 작은 값 1을 모두 1에 가까운 방향 크기로 만든다. elementwise Adam의 첫 step도 eps가 작으면 좌표별 크기를 비슷하게 만들 수 있지만 회전된 matrix와 history가 들어오면 두 동역학은 달라진다.

`G=R diag(4,1) Rᵀ`에서 45도 회전 `R`을 쓰면 원소는 대각/비대각이 섞인다. Muon의 polar factor는 singular vector를 따라 변환하지만 elementwise second moment는 현재 좌표 원소를 따라 scale한다. 독자는 FP64 SVD reference와 AdamW one-step을 계산해 Frobenius norm, singular value, update/gradient cosine을 비교한다.

rank-one `[[1,2],[2,4]]`, zero, tall `[4,2]`, wide `[2,4]`도 fixture에 넣는다. zero는 normalization 분모, rank-one은 영 singular direction, tall/wide는 transpose와 반환 shape를 시험한다. exact polar와 BF16 Newton–Schulz output의 차이는 허용하되 finite와 residual, shape를 사전 threshold로 판정한다.

**Newton–Schulz 반복의 source와 test 경계**

KellerJordan/Muon 고정 commit의 `zeropower_via_newtonschulz5`에서 BF16 cast, tall matrix transpose, norm scaling, polynomial coefficient와 반복 수를 읽는다. 이 함수가 exact SVD를 호출하지 않는 이유는 accelerator matmul로 근사를 계산하기 위해서다. 그러나 함수 존재는 어떤 model에서 수렴이 더 좋다는 증거가 아니다.

source table에는 commit, file/function/span, claim과 선택 조건을 기록한다. upstream test가 shape와 finite만 assert하는지, exact reference proximity까지 보는지 확인한다. local FP64 toy는 수치 식을, BF16 GPU fixture는 implementation path를, 장기 model run은 optimization behavior를 담당한다. 세 evidence를 합치지 않는다.

반복마다 `X`, `A=XXᵀ`, residual, norm과 finite를 hook한다. coefficient나 NS step을 바꾼 negative control은 residual과 matmul 수가 예상대로 변하는지 본다. iteration을 늘리면 항상 좋은 것이 아니며 low precision rounding과 추가 critical path를 함께 측정한다.

**momentum과 scale이 붙는 순서**

production Muon은 polar transformation 하나가 전부가 아니다. momentum buffer가 raw gradient를 누적한 뒤 orthogonalization하는지, transformed update에 momentum을 적용하는지 source call path로 확인한다. Nesterov가 어떤 tensor 조합을 읽는지도 고정한다. 순서가 다르면 같은 beta와 NS 함수라도 다른 optimizer다.

implementation은 output을 dimension이나 norm에 따라 rescale할 수 있다. 논문식 `UVᵀ`와 최종 parameter delta 사이의 scale, learning rate, weight decay를 분리한다. gradient를 10배 한 fixture에서 normalization 전후와 final delta가 어떻게 변하는지 기록한다. scale invariance가 어느 구간까지만 성립하는지 보인다.

checkpoint state에는 momentum, internal step, group scale과 stack/owner mapping이 들어간다. function argument default만 recipe에 기록하지 않고 resolved 값과 state schema를 manifest에 둔다. resume 뒤 첫 NS 입력과 final delta가 uninterrupted run과 같아야 한다.

**optimizer family별 negative control**

첫 control은 all-AdamW다. hybrid와 같은 initialization, batch, gradient, scheduler에서 auxiliary group delta는 정확히 같아야 한다. 둘째는 모든 2D tensor에 Muon을 적용하는 위험군이다. embedding/router의 update ratio와 validation instability가 semantic allowlist와 어떻게 다른지 본다.

셋째는 exact SVD polar를 작은 model subset에서 사용하는 느린 oracle이다. NS output이 허용오차 안에 있는지 검사하되 exact SVD를 production 성능 후보로 간주하지 않는다. 넷째는 NS 반복 0 또는 coefficient 교란으로 test가 실제 오류에 민감한지 확인한다.

다섯째는 state reset이다. checkpoint에서 momentum 하나를 0으로 만들고 다음 delta mismatch가 검출되는지 본다. parameter만 비교하는 resume test가 이 장애를 놓치는 이유를 보여준다. 여섯째는 group swap으로 embedding 하나를 matrix group에 옮겨 manifest digest와 control assertion이 실패하는지 본다.

**gradient replay pipeline**

GoldenBatch backward 뒤 unscaled·unclipped logical gradient를 immutable artifact로 저장한다. loss sum/count, parameter parent hash, dtype/shape, gradient hash와 norm을 붙인다. replay loader는 fresh state와 checkpoint state를 구분한다. 다른 model이나 reordered stack에 우연히 shape만 맞춰 연결하지 않는다.

같은 artifact를 AdamW, hybrid Muon, 다른 matrix optimizer 후보에 넣는다. one-step report에는 parameter/state before와 after, delta spectrum, temporary bytes를 기록한다. common group parity가 깨지면 장기 run을 시작하지 않는다. 후보 matrix delta는 independent toy/reference와 맞아야 한다.

gradient replay는 algorithmic diff를 data/dropout과 분리하지만 장기 수렴을 대신하지 않는다. short smoke와 full run은 별 단계다. replay에서 correctness를 닫고 smoke에서 finite/update ratio를 닫은 뒤 token/time endpoint를 비교한다.

**state와 byte 원장**

matrix optimizer마다 state tensor 이름, dtype, shape와 lifetime을 표로 만든다. momentum 하나만 두는지, second moment나 factor를 두는지, parameter stack 단위 state인지 적는다. AdamW auxiliary group의 두 FP32 moment와 master/gradient도 포함한다.

temporary는 NS matmul의 `A`, polynomial 중간, stacked update와 collective buffer를 계산한다. allocator peak와 expected live bytes가 다르면 trace로 숨은 buffer를 찾는다. 평균 state bytes만으로 OOM을 예측하지 않는다. first step lazy state와 checkpoint staging이 겹친 peak도 분리한다.

distributed owner에서는 state가 owner rank에만 있지만 update all-gather가 transient copy를 만든다. parameter 수를 world size로 나눈 단순 계산을 쓰지 않고 replicated/sharded/transient 항을 나눈다. topology별 rank max가 실제 capacity를 결정한다.

## 12.9 distributed owner·collective·RCA를 하나의 원장에 둔다

matrix state가 어느 rank에 있고 어떤 collective가 update 의미를 복원하는지 byte와 순서로 기록한다.

two-rank fixture는 single-rank와 동일 global batch를 분할한다. global loss denominator, reduced logical gradient, final delta를 비교한다. gradient가 같고 delta가 다르면 owner mapping, scale, state shard를 본다. gradient부터 다르면 optimizer가 아니라 data/reduction 문제다.

round-robin owner는 matrix count는 맞춰도 numel과 aspect ratio가 불균형할 수 있다. numel-balanced와 measured-cost-balanced mapping을 비교한다. NS matmul cost와 collective bytes를 rank별 p50/max로 본다. owner map 변경은 checkpoint schema와 recipe digest를 바꾼다.

한 rank만 gradient 없는 matrix를 건너뛰면 collective order가 깨질 수 있다. missing gradient, unused expert, empty microbatch를 주입해 모든 rank가 동일 protocol을 실행하는지 본다. timeout report는 마지막 collective와 tensor logical ID를 포함한다.

### scheduler와 geometry를 함께 비교한다

matrix optimizer의 lr 숫자는 AdamW lr와 같은 단위라고 가정하지 않는다. 각 family의 안정 범위를 같은 search budget과 selection rule로 찾는다. scheduler shape와 token clock, warmup budget은 고정하되 base lr는 family별 허용한다. 같은 숫자를 강제한 비교와 무제한 tuning 비교를 모두 피한다.

update/weight ratio를 role별로 기록해 effective scale을 해석한다. scheduler lr는 내려가는데 Muon output scale이나 momentum 때문에 특정 matrix ratio가 오르면 incident로 본다. overflow skip과 committed scheduler clock도 family 간 동일 contract를 쓴다.

full run의 x축은 valid token과 wall time을 둘 다 제공한다. same-workload와 memory saving을 활용한 capacity run을 분리한다. batch를 늘린 결과를 optimizer geometry 효과로만 쓰지 않는다.

### incident/RCA: loss가 정상인데 validation이 갈린다

one-step parity와 train loss는 맞지만 validation이 seed별로 크게 갈리면 data order, lr search selection, matrix grouping과 update spectrum을 본다. 최초 divergence token checkpoint에서 group별 delta cosine과 representation metric을 비교한다. 마지막 checkpoint만 비교하지 않는다.

hybrid run에서 embedding control도 갈리면 scheduler, gradient denominator 또는 data consumption이 다른 것이다. matrix group만 갈리면 의도한 알고리즘 차이인지 unstable role이 포함됐는지 layer/role ablation을 한다. router/expert가 문제라면 semantic scope를 줄인다.

NS residual은 정상인데 wall time이 나쁘면 critical path와 overlap을 본다. owner straggler, temporary memory로 인한 microbatch 감소, graph break가 원인일 수 있다. “Muon kernel이 느리다”로 뭉뚱그리지 않는다.

### incident/RCA: resume 뒤만 달라진다

resume 첫 GoldenBatch와 gradient가 같으면 state mapping, internal step, momentum, scheduler를 비교한다. parameter stack 순서가 바뀌었는데 state tensor shape가 같아 load된 silent corruption을 logical ID hash가 잡아야 한다. world-size change라면 reshard conversion report를 본다.

gradient부터 다르면 sampler/RNG, loss denominator, mixed precision state를 본다. optimizer 이름을 먼저 의심하지 않는다. uninterrupted와 resume의 최초 다른 artifact를 단계별로 찾는다. root metric이 비슷해도 next-step mismatch는 numerical resume 실패다.

partial checkpoint를 load하도록 강제한 negative test가 실제로 거부되는지 확인한다. state shard 하나와 root marker를 제거하고 error code를 고정한다. fallback fresh optimizer를 조용히 만드는 loader는 production에서 금지한다.

**publication-grade experiment card**

card 첫 줄은 hypothesis와 primary endpoint다. 다음은 model/data/tokenizer, GoldenBatch, parameter manifest, source/test revision, hardware/topology다. optimizer recipe에는 family, group scope, lr/momentum/scale/NS와 state dtype을 모두 기록한다.

결과는 correctness, numerical stability, token efficiency, wall efficiency, memory/network로 나눈다. seed와 failed run, search 범위, exclusion reason을 공개한다. 실행하지 않은 multi-node fault는 proposed로 표시한다. source를 읽은 사실을 throughput observation으로 바꾸지 않는다.

artifact에는 replay gradient, one-step report, state manifest, profiler, checkpoint/fault event, raw metric이 있다. 본문의 표에서 RunID를 따라 이 artifact로 내려갈 수 있어야 한다. code snippet은 핵심 update를 이해시키되 전체 동작 주장은 고정 source와 test에 연결한다.

**독자 인수 시험**

독자는 2×2 exact polar와 NS output을 비교하고 residual을 계산한다. 같은 gradient를 AdamW와 matrix optimizer에 replay해 control/matrix delta를 분리한다. semantic grouping과 all-2D 위험군 manifest diff를 만든다.

two-rank global denominator와 owner update를 single reference와 비교한다. rank kill과 missing-gradient collective를 주입한다. checkpoint momentum reset과 stack reorder가 다음-step gate를 실패시키는지 본다. profiler에서 NS matmul, temporary와 all-gather critical path를 찾는다.

최종 선택은 “어느 optimizer가 최고인가”가 아니라 어느 parameter role과 workload에서 어떤 evidence로 승인했는가다. correctness invariant가 하나라도 실패하면 성능 곡선을 채택하지 않는다. 품질 우위가 있어도 memory/복구가 목표를 못 맞추면 조건부 또는 기각이다.

**현장 결정표와 최소 제출 파일**

optimizer 후보가 one-step 수치 gate에서 실패하면 lr tuning으로 넘어가지 않는다. common AdamW group만 실패하면 recipe construction, scheduler 또는 replay 입력 문제다. matrix group만 reference와 다르면 transpose, scale, momentum 순서와 NS source를 본다. single rank는 맞고 distributed만 틀리면 owner/reduction/collective를 본다.

수치 gate는 통과하지만 100-step smoke에서 non-finite가 생기면 최초 matrix의 role, aspect ratio, singular spectrum, residual과 update ratio를 보존한다. all-2D 위험군만 실패하면 semantic grouping evidence다. allowlist도 실패하면 lr/scale/NS 범위와 precision을 좁힌다. 전체 optimizer를 모호하게 기각하지 않는다.

품질은 좋아지지만 wall time이 나쁘면 NS compute, owner straggler, collective와 temporary peak를 나눈다. state memory 절감으로 더 큰 batch를 허용하는 이익은 same-workload 결과와 별 표에 둔다. token efficiency, time efficiency와 capacity effect가 서로 다른 결론을 낼 수 있다.

최소 제출 디렉터리에는 `parameter-role-manifest`, `source-test-map`, `gradient-replay`, `toy-geometry`, `one-step-parity`, `state-byte-ledger`, `distributed-events`, `checkpoint-resume`, `run-card`를 둔다. 모든 파일은 model/recipe digest와 source revision을 공유한다. result table에서 RunID를 따라 raw artifact로 내려갈 수 있어야 한다.

`toy-geometry`에는 exact SVD polar, 반복별 NS output과 residual을 기록한다. `one-step-parity`에는 common/matrix group의 gradient, state, delta와 tolerance를 기록한다. `distributed-events`에는 global denominator, owner map, collective bytes, injected rank failure를 기록한다. `checkpoint-resume`는 next GoldenBatch, NS input, momentum과 delta를 비교한다.

**모델 upgrade 재검증**

모델이 fused QKV, MoE 또는 새 embedding layout으로 바뀌면 기존 parameter role manifest를 재사용하지 않는다. old/new resolved module과 tensor를 diff하고 새 semantic role을 승인한다. shape가 같다는 이유로 optimizer state를 다른 role에 연결하지 않는다. tied storage와 flatten offset도 다시 계산한다.

optimizer source upgrade는 function name이 같아도 coefficient, scale, state key, owner protocol을 diff한다. toy/replay/checkpoint test를 재실행한다. upstream benchmark 개선을 local workload의 승인으로 대신하지 않는다. binary와 profiler evidence가 바뀌면 새 execution RunID다.

world size나 topology upgrade도 recipe migration이다. owner map과 stack order, collective byte, state reshard를 검증한다. same-world-size checkpoint만 지원한다면 loader가 다른 topology를 명시적으로 거부해야 한다. permissive fresh-state fallback은 재현성을 깨뜨린다.

**마지막 구두 검산**

인수자는 임의 matrix 하나를 골라 semantic role, gradient SVD 직관, momentum과 NS 변환 순서, scale/lr/decay, state owner와 collective를 설명한다. 이어 checkpoint에서 그 momentum과 internal step을 찾아 다음 delta를 replay한다. 이 경로가 source와 artifact로 연결돼야 한다.

두 번째 질문은 왜 embedding을 제외했는가다. 단순 shape가 아니라 lookup frequency, sparse/row semantics와 control evidence로 답한다. 포함했다면 별 ablation과 stability evidence가 필요하다. “논문 implementation이 그랬다”만으로 자신의 model grouping을 정당화하지 않는다.

세 번째 질문은 무엇이 아직 미검증인가다. source-verified, upstream-tested, locally-executed와 proposed multi-node/full-run을 구분한다. 이 경계가 명확해야 독자가 자신의 hardware와 model에서 다음 시험을 선택할 수 있다. 완성된 장은 모든 optimizer의 우승자를 선언하는 대신 검증 가능한 선택 절차를 넘긴다.

**최종 회귀 표본**

release 뒤에도 작은 gradient replay와 two-rank fixture를 CI에 남긴다. optimizer source, model module tree, wrapper 또는 checkpoint schema가 바뀌면 자동 실행한다. resolved parameter set과 source function digest가 기존과 같더라도 compiler/backend가 달라지면 numerical report를 새로 만든다.

회귀 표본은 정상 matrix만 포함하지 않는다. zero, rank-one, tall/wide, embedding 오분류, missing gradient와 stale momentum을 유지한다. negative case가 더는 실패하지 않으면 test가 약해졌거나 semantics가 변한 것이다. expectation을 바꾸려면 change record와 새 수학적 근거가 필요하다.

장기 metric regression은 작은 correctness test와 별 job이다. token-to-target, wall time, state peak와 collective tail을 기준 release와 비교한다. hardware 차이로 절대 시간이 바뀌면 normalized workload와 profiler evidence를 사용한다. correctness PASS가 품질/성능 PASS를 의미하지 않는다.

최종 archive에는 승인 recipe와 기각 후보를 함께 둔다. 어느 group, scale, lr 또는 owner map에서 실패했는지 남기면 다음 model에서 같은 탐색을 반복하지 않는다. 기각 결과도 조건이 바뀌면 재검토 가능한 evidence다.

운영 dashboard는 optimizer family별 group update ratio, NS residual 표본, owner straggler와 collective timeout을 같은 committed step에 맞춘다. 전체 loss와 전체 grad norm만으로는 matrix 하나의 불안정과 common control drift를 구분할 수 없다. incident를 클릭하면 logical parameter, source recipe, gradient replay와 checkpoint state로 내려가게 한다.

release 이후 새 matrix role이 추가되면 allowlist에 자동 포함하지 않는다. zero-gradient smoke, toy/replay, memory·owner mapping과 short behavioral branch를 통과한 뒤 recipe revision을 올린다. 기존 checkpoint가 새 group schema와 호환되지 않으면 migration tool 또는 명시적 fresh-state branch를 사용한다.

최종 검토자는 성능표의 모든 행이 동일한 GoldenBatch/data budget과 정확한 optimizer recipe를 가리키는지 확인한다. control group parity가 깨진 run, incomplete checkpoint에서 재개한 run, source revision이 다른 run은 별 실험으로 분리한다. 숫자가 좋아도 비교 계약이 다르면 같은 순위표에 넣지 않는다. 이것이 최종 승인 기준이다.

**이 장이 넘기는 것.** group별 optimizer family, matrix-shape contract, state/collective byte estimate, update checksum.

**다음 장에서 깨질 수 있는 것.** scheduler가 optimizer별 내부 step 의미와 외부 token clock을 혼동하면 같은 곡선이 다른 시점에 적용된다.

**검증 체크포인트.** group 누락·중복 없음, tied parameter 단일 소유, resume 뒤 Newton–Schulz 입력과 step counter 동일.

## 12.10 도입 실험과 선택 회의를 재현 가능한 절차로 만든다

baseline, grouping, memory, 수치 parity, scale-out와 rollback을 차례로 통과해야 새로운 optimizer를 승인한다.

첫 단계는 후보를 고르는 일이 아니라 logical parameter inventory를 만드는 일이다. 각 행에 stable parameter ID, module role, global/local shape, dtype, sparsity, tied alias, sharding axis, gradient owner를 기록한다. `ndim==2`만으로 Muon 또는 Shampoo group을 만들면 embedding table, expert router, vocab head까지 숨어 들어간다. 이 tensor들은 외형은 행렬이어도 dense hidden transform과 gradient 통계가 다르다. 왜 role을 먼저 고정하는가. optimizer를 바꾼 뒤 생긴 개선과 grouping 변경 효과를 분리해야 하기 때문이다.

둘째 단계는 gradient replay corpus다. 초기·warmup 종료·중간·decay 구간에서 logical gradient를 수집하고 zero·rank-one·ill-conditioned·tall·wide synthetic matrix를 더한다. replay는 model forward를 다시 돌리지 않고 optimizer 수치 경로만 비교하므로 optimizer 차이를 data order와 dropout noise에서 격리한다. 실험 단위는 loss curve가 아니라 입력 parameter/state/gradient에서 출력 delta/state로 가는 순수 전이다.

셋째 단계는 수학 oracle이다. 작은 matrix는 FP64 SVD/eigendecomposition으로 polar factor와 inverse root를 계산한다. Muon에는 `||X^TX-I||`, singular-value spread와 scale equivariance를, Shampoo에는 root residual과 symmetry를, Lion에는 sign transition을, Sophia에는 clipping 전후 좌표를 기록한다. oracle이 production kernel과 같은 algorithm이어서는 안 된다. 같은 버그를 공유할 수 있기 때문이다.

넷째는 state와 peak byte 회계다. persistent state, cached preconditioner, root workspace, foreach list, collective staging buffer를 구분한다. `[m,n]` weight의 full Shampoo Gram은 대략 `m²+n²` 원소지만 block partition, padding, root 복제 정책에 따라 실제 byte가 달라진다. Muon momentum은 parameter 크기와 비례하지만 Newton–Schulz intermediate가 peak를 만든다. 평균 memory만 보면 순간 root workspace OOM을 놓친다.

다섯째는 분산 owner 계약이다. logical matrix가 FSDP axis로 잘렸을 때 local shard만 직교화하면 full matrix polar update와 다른 algorithm이다. all-gather 후 계산할지, 2D mesh에서 distributed matmul을 할지, block을 rank에 배정할지 명시한다. optimizer collective와 reducer가 gradient를 중복 reduce하지 않는지도 확인한다. collective sequence ID와 owner group을 trace에 넣으면 hang과 수치 불일치를 구분할 수 있다.

여섯째는 checkpoint commit이다. parameter group 규칙, transpose 여부, block partition, optimizer family/version, backend, iteration coefficient, state dtype, root cadence와 counter를 저장한다. key는 Python iteration order가 아니라 stable ID로 만든다. 저장 도중 rank가 죽으면 모든 shard가 같은 OptimizerStepID를 가리키는 checkpoint만 publish한다. resume 뒤 첫 replay가 uninterrupted branch와 같은 logical delta를 만드는 것이 종료 조건이다.

### Muon 고정 소스의 함수 경계를 추적한다

정의와 실험 조건은 `https://arxiv.org/abs/2502.16982`, PyTorch 공식 recipe는 `https://docs.pytorch.org/tutorials/recipes/recipes/muon_optimizer.html`에서 확인한다. NVIDIA 고정 구현 `https://github.com/NVIDIA/NeMo-Run/tree/83537ba67cb4c998251567f78a534776fecb1965`와 `emerging_optimizers/orthogonalized_optimizers/muon.py:38`은 coefficient, backend, fallback을 읽는 출발점이다. 논문의 수식과 library의 scaling convention이 같다고 가정하지 말고 input normalization, transpose, iteration count, post-scale을 순서대로 펼친다.

Newton–Schulz 한 반복은 matmul을 여러 번 수행한다. 입력 norm이 안정 영역 밖이면 반복 횟수를 늘릴수록 좋아진다는 직관이 깨진다. 실험은 `ns_steps=0..k`, FP32/BF16, aspect ratio와 condition number를 교차한다. 관측 열은 residual, delta RMS, maximum absolute value, kernel time, temporary bytes다. 결과가 나쁘면 lr 전에 normalization과 coefficient family가 실제 선택되었는지 확인한다.

wide matrix를 transpose해 tall 경로로 계산하는 구현은 반환 transpose와 scale convention을 함께 검증한다. `[2,8]`와 그 transpose fixture가 transpose 관계의 delta를 내는지 본다. rank-deficient matrix에서는 polar factor의 null-space 선택이 유일하지 않을 수 있어 원소 equality 대신 row/column space action과 residual을 비교한다. zero matrix는 NaN 없이 정의된 fallback을 내야 한다.

분산 Muon이 parameter를 stack해 collective로 묶는다면 stack order가 checkpoint state다. rank마다 dict order가 다르면 같은 byte 수로 다른 tensor를 reduce하는 silent corruption도 가능하다. stable ID sort, shape vector, concatenation offset digest를 collective 전에 all-rank 비교한다. 디버깅은 collective 이전 gradient digest, 이후 full update digest, unstack 뒤 delta digest 순서로 최초 불일치를 찾는다.

## 12.11 optimizer family를 state machine과 결정 트리로 비교한다

AdamW를 기준 좌표로 두고 Muon, Shampoo, SOAP, Sophia와 Lion의 state 생성·갱신·checkpoint를 비교한다.

Shampoo step은 gradient 수신, Gram accumulation, precondition 여부 판정, inverse-root 계산 또는 cached root 재사용, graft magnitude 계산, update 적용, counter commit 순서다. `precondition_frequency=10`에서 counter를 save 직전 올리는지 뒤에 올리는지에 따라 resume 첫 root 갱신 시점이 달라진다. root를 저장하지 않으면 load 뒤 Gram에서 재계산한 root가 허용오차 안에 드는지, 계산 시간은 restart SLO에 드는지 검증한다.

원 논문은 `https://proceedings.mlr.press/v80/gupta18a.html`, scalable distributed 비교점은 `https://arxiv.org/abs/2002.09018`에 둔다. 이 링크가 production 세부를 모두 증명하지는 않는다. block partition, graft, protected eigendecomposition과 owner는 채택 library commit에서 다시 고정한다. theory coordinate, code coordinate, local fixture를 서로 다른 열에 두는 이유다.

SOAP에는 eigenbasis 갱신 clock과 그 basis의 Adam moment 갱신 clock이 따로 있다. basis가 바뀌는 step에서 old moment를 회전하는지 reset하는지 확인한다. repeated eigenvalue에서는 basis가 회전해도 같은 subspace일 수 있으므로 원소별 basis equality는 부적절하다. subspace projector, transformed update와 one-step을 oracle로 쓴다.

Sophia 논문 `https://arxiv.org/abs/2305.14342`의 curvature estimator와 채택 구현의 estimator를 구분한다. curvature batch가 training sampler cursor를 소비하면 token accounting과 curriculum이 바뀐다. 별도 iterator라면 seed와 cursor가 checkpoint 대상이다. curvature update 직후 kill, 직전 kill, stale curvature를 반복하는 고장 실험으로 cadence counter를 검증한다.

Lion 논문 `https://arxiv.org/abs/2302.06675`의 sign update는 state byte를 줄이지만 lr과 decay를 AdamW에서 복사할 근거를 주지 않는다. sign 직전 momentum 조합과 state 갱신 조합이 다를 수 있어 두 beta 위치를 코드와 test에서 대조한다. gradient가 `+ε,-ε`로 진동하는 fixture, zero fixture, decay-only fixture가 branch를 드러낸다.

### 공정 비교와 디버깅 결정 트리

비교는 세 층이다. 수치 층은 같은 replay와 fixed group에서 delta/state를 본다. 학습 층은 같은 token budget, data order, model seed에서 validation trajectory를 본다. 시스템 층은 같은 topology에서 critical path, peak byte, collective byte를 본다. state 절감으로 batch를 키운 결과는 fixed-batch 결과와 별도 표에 둔다. 그렇지 않으면 algorithm 개선과 memory 재투자를 혼동한다.

첫 step부터 NaN이면 normalization, denominator floor, inverse-root residual과 dtype을 본다. root 갱신 뒤만 튀면 cadence와 stale state, basis rotation을 본다. single rank는 정상이고 분산만 다르면 full logical matrix, block owner, stack order를 본다. uninterrupted는 정상이고 resume만 다르면 counter, group ordering, cached root와 mode를 본다. loss는 같은데 validation만 갈리면 parameter role grouping과 tied alias를 대조한다.

최소 test suite는 zero/constant/scale-transformed/rotated gradient, rank-one/tall/wide matrix, checkpoint roundtrip, owner-rank kill을 포함한다. scale fixture는 normalization, rotation은 coordinate dependence, rank는 numerical boundary, roundtrip은 state completeness, kill은 atomic commit을 반증한다. 모든 실험은 입력 digest, source commit, backend와 tolerance를 남긴다.

최종 `OptimizerGeometryCard`에는 source commit, inclusion/exclusion, equation convention, state schema, backend/fallback, dtype, distributed owner, checkpoint migration, 실행한 fixture와 미실행 범위를 담는다. 이 카드는 13장에 전달된다. optimizer를 바꾸면서 lr curve도 동시에 바꾸지 않는 이유는 변화의 원인을 식별하기 위해서다.

### AdamW를 기준 좌표계로 다시 세운다

### 익숙한 이름이 숨기는 여섯 개의 상태 전이를 펼친다

AdamW 한 step을 `g 수신 → moment 갱신 → bias correction → denominator 구성 → parameter update → decoupled decay`로 펼친다. 흔히 쓰는 식은 `m_t=β1m_{t-1}+(1-β1)g_t`, `v_t=β2v_{t-1}+(1-β2)g_t²`, `θ_t=(1-lr·wd)θ_{t-1}-lr·m̂_t/(sqrt(v̂_t)+eps)`다. 그러나 epsilon이 제곱근 안인지 밖인지, decay가 update 전인지 후인지, step counter가 group인지 parameter인지에 따라 실제 함수는 달라진다.

bias correction은 초기 moment가 0에서 시작한 편향을 보정한다. step counter가 하나 뒤로 밀리면 첫 수십 step의 effective update가 달라진다. gradient가 constant인 fixture에서는 corrected first moment가 곧 gradient가 되고 corrected second moment가 gradient 제곱이 되는 성질을 쓸 수 있다. resume 직후만 차이가 나면 counter restore와 increment 위치를 먼저 본다.

decoupled weight decay는 loss gradient에 `λθ`를 더하는 L2 regularization과 adaptive denominator 아래에서 같지 않다. L2 항은 moment와 coordinate별 scaling을 통과하지만 AdamW decay는 parameter에 직접 적용된다. gradient가 정확히 0인 fixture에서 parameter가 `(1-lr·wd)`만큼 줄어드는지 보면 경로를 분리할 수 있다. embedding, norm, bias를 decay group에서 제외하는 규칙도 logical role과 stable parameter ID로 고정한다.

epsilon은 division-by-zero guard 이상이다. `v`가 작은 좌표에서는 update scale을 지배한다. BF16 state, FP32 state, fused implementation이 epsilon을 어느 dtype에서 더하는지에 따라 underflow 경계가 다르다. `g=[0, 2^-k]` sweep으로 denominator와 delta를 기록한다. 큰 정상 gradient 하나만 쓰는 optimizer test는 이 차이를 숨긴다.

AMSGrad, capturable, differentiable, foreach, fused 같은 옵션은 단순 성능 플래그가 아니다. AMSGrad는 `max(v)` state를 추가하고 함수가 바뀐다. capturable은 step과 lr를 device tensor로 관리할 수 있다. differentiable은 optimizer step 자체를 autograd graph에 포함한다. foreach/fused는 수학적 의도는 같아도 grouping, peak memory, dtype cast와 reduction order를 바꾼다. 옵션별 state schema와 fallback을 검사한다.

PyTorch 고정 revision을 채택할 때는 `torch/optim/adamw.py`의 public `step`, functional `adamw`, single/foreach/fused dispatcher를 따라간다. parameter list와 state initialization이 어디서 만들어지고 backend 선택 조건이 무엇인지 기록한다. 문서의 수식과 실제 foreach/fused branch가 동일한 edge case를 처리하는지는 repository test로 대조한다.

**Adam, Lion, Sophia를 update 방향의 기하로 비교한다**

Adam은 각 좌표의 최근 제곱 gradient로 대각 metric을 만든다. 같은 Euclidean gradient라도 자주 큰 gradient가 나온 좌표의 update를 줄인다. 좌표 회전에 불변하지 않다는 뜻이다. 2차원 quadratic을 회전시킨 fixture에서 원래 좌표와 회전 좌표의 trajectory를 다시 되돌려 비교하면 이 성질을 직관적으로 볼 수 있다.

Lion은 moment 조합의 sign을 update 방향으로 사용한다. 크기를 버려 L∞ 또는 sign geometry와 닮은 행동을 하지만 논문의 정확한 algorithm을 단순 signSGD로 축약해서는 안 된다. update에 쓰는 moment 조합과 다음 state에 저장하는 조합의 beta가 다를 수 있다. `+ε,-ε` 교대 gradient와 큰 outlier 하나를 넣어 state와 sign 전환 시점을 본다.

Sophia는 curvature proxy로 좌표별 update를 제한한다. Hessian diagonal 추정과 clipping이 있어서 Adam의 second moment와 같은 상태라고 부를 수 없다. curvature 갱신 cadence, estimator batch, stochastic estimator seed가 추가 clock을 만든다. curvature를 10 step마다 갱신한다면 step 9/10 경계 checkpoint가 완전해야 한다.

동일한 loss 감소를 비교할 때 learning rate 숫자를 같게 두는 것은 공정하지 않을 수 있다. optimizer마다 update norm convention이 다르다. 먼저 gradient replay에서 update RMS, update-to-weight ratio, cosine with raw gradient를 맞추거나 관측하고, 그 다음 token-budget 학습에서 validation을 비교한다. hyperparameter search budget과 범위를 함께 보고한다.

**Muon과 행렬 직교화를 함수·수치·배치 세 층으로 해부한다**

**polar 방향은 singular value를 평탄화하지만 정보를 모두 보존하지 않는다**

gradient matrix를 `G=UΣVᵀ`라 하면 polar factor는 `UVᵀ`다. nonzero singular direction의 크기를 1로 평탄화한다. 큰 singular mode가 update를 독점하는 것을 막는 직관을 주지만, 이것이 곧 모든 layer에서 최적이라는 증명은 아니다. rank deficiency에서는 null space의 선택과 수치 regularization이 중요하다.

Muon은 보통 momentum matrix를 만든 뒤 Newton–Schulz류 반복으로 직교화 근사를 계산한다. 반복은 matrix inverse를 직접 구하지 않고 matmul 다항식으로 singular value를 목표 구간에 보낸다. convergence를 위해 input norm을 안정 영역에 넣는 normalization이 필요하다. coefficient, normalization norm, iteration count가 algorithm identity다.

로컬 고정 소스 `research/sources/muon-code-audit/nanochat/nanochat/optim.py`의 `Muon` 계보와 `tests/test_optim.py`, `sources/training-muon-keller/muon.py`, `research/sources/muon-code-audit/emerging/emerging_optimizers/orthogonalized_optimizers/muon.py`를 서로 독립 구현으로 읽는다. 동일 이름을 합의 증거로 쓰지 않고 momentum convention, Newton–Schulz coefficient, post scale, parameter selection을 표로 대조한다.

nanochat 구현에서 optimizer가 parameter를 여러 rank에 배정하거나 update를 모으는 경로가 있다면 logical matrix owner와 collective를 추적한다. test가 어떤 shape와 dtype만 덮는지도 본다. test 부재는 실패 증거는 아니지만 미검증 범위다. tall, wide, zero, rank-one, ill-conditioned, odd dimension fixture를 추가 설계한다.

matrix shape scaling은 핵심이다. polar factor는 entry scale이 original gradient와 다르며 width/height에 따라 update RMS가 달라질 수 있다. 구현이 `sqrt(max(1,m/n))` 같은 후처리나 parameter RMS 기반 scale을 쓰는지 확인한다. 논문의 learning rate를 다른 scaling convention 구현에 그대로 옮기지 않는다.

모든 2D tensor를 Muon으로 보내면 안 된다. token embedding은 sparse row access와 frequency imbalance를 가지며, LM head는 vocabulary 축이 거대하고 tying될 수 있다. router matrix는 expert selection을 직접 바꾼다. norm/bias는 1D다. module role과 논문 recipe를 근거로 Muon group과 AdamW group을 분리하고 alias는 한 group에만 둔다.

backward gradient가 TP shard라면 local matrix polar factor는 global matrix의 polar factor와 일반적으로 다르다. column shard 각각을 직교화하면 shard 내부 singular spectrum만 평탄화한다. 이것을 distributed approximation으로 채택할 수는 있지만 full Muon과 동일하다고 쓰면 안 된다. all-gather reference와 local-shard 결과의 angle, norm, one-step loss를 비교한다.

**Newton–Schulz의 실패를 NaN 이전에 발견한다**

각 반복에서 Frobenius norm, spectral norm estimate, `||XXᵀ-I||` 또는 `||XᵀX-I||`, max absolute value를 남긴다. rectangular matrix는 작은 차원 쪽 identity와 비교한다. residual이 몇 회 감소하다 증가하면 coefficient 영역, normalization, precision을 본다. 최종 NaN만 기다리지 않는다.

TF32 matmul은 속도를 높이지만 eigen/polar 근사의 작은 singular mode에 오차를 줄 수 있다. FP32 accumulator라고 입력 mantissa가 완전히 보존되는 것은 아니다. TF32 on/off, BF16, FP32, FP64 oracle을 작은 fixture에서 비교한다. production 선택은 residual과 학습 영향, kernel time을 함께 보고 정한다.

iteration을 늘리면 matmul 수와 temporary lifetime이 늘어난다. compile/fusion이 intermediate를 재사용하는지에 따라 peak memory가 다르다. optimizer step wall time만 보지 않고 peak allocated/reserved, synchronization, collective exposed tail을 기록한다. root/orthogonalization cadence가 있는 변형은 별도 state clock을 가진다.

gradient clipping 순서도 결과를 바꾼다. raw gradient를 global norm clip한 뒤 momentum과 orthogonalization을 하는지, orthogonal update를 만든 뒤 update norm을 제한하는지 다르다. mixed optimizer에서 Adam group과 Muon group을 하나의 global norm으로 묶는지 확인한다. 동일 `clip_grad_norm` 옵션이 어느 tensor 집합과 어느 시점에 적용되는지 source call graph로 고정한다.

**Shampoo와 SOAP의 preconditioner를 분산 상태로 읽는다**

**Kronecker factor는 행렬 구조를 쓰는 대신 제곱 상태를 만든다**

gradient `G∈R^{m×n}`에 대해 Shampoo는 좌우 statistic `L←L+GGᵀ`, `R←R+GᵀG`를 축적하고 inverse root를 이용해 양쪽에서 precondition한다. full factor 상태는 `m²+n²`이므로 큰 축에서 부담이 크다. block Shampoo는 축을 block으로 잘라 상태를 줄이지만 block boundary라는 새로운 algorithm choice를 만든다.

factor에 damping을 더하는 위치와 exponent는 수치 안정성을 결정한다. eigendecomposition, coupled Newton iteration, QR 기반 root 등 backend도 다르다. inverse-root residual `||P^{-r}-A||`에 맞는 검사를 정하고 symmetry와 positive semidefinite invariant를 본다. eigenvalue floor가 큰 경우 update가 사실상 diagonal 또는 identity에 가까워질 수 있다.

grafting은 Shampoo 방향에 SGD 또는 AdaGrad 계열의 update norm을 입힐 수 있다. 방향과 크기를 분리한다는 기하적 해석이 가능하다. 그러나 graft state와 preconditioner state가 모두 checkpoint 대상이다. graft를 끄고 켠 replay에서 cosine은 비슷하고 norm만 바뀌는지 확인하면 구현 위치를 검증할 수 있다.

SOAP은 preconditioner eigenbasis에서 Adam류 moment를 갱신한다. basis가 주기적으로 변하면 moment 좌표계도 다뤄야 한다. basis를 바꾸고 old moment를 그대로 좌표값으로 쓰면 다른 벡터를 의미한다. 구현이 basis 사이 rotation을 적용하는지, 새 basis에서 통계를 reset 또는 그대로 근사하는지 source와 test로 확인한다.

repeated eigenvalue에서는 eigenvector 부호와 basis가 비결정적으로 바뀔 수 있다. basis tensor의 원소 checksum이 다르다고 곧 함수가 다른 것은 아니다. projector `QQᵀ`, transformed update, parameter delta를 비교한다. 반대로 checkpoint가 basis를 누락해도 우연히 같은 projector를 재구성할 가능성과 다음 moment 의미를 구분한다.

**owner-rank 방식은 계산 절약과 복구 책임을 함께 만든다**

분산 Shampoo는 block별 preconditioner 계산을 owner rank에 배정하고 결과를 broadcast할 수 있다. owner map은 parameter stable ID와 block index의 함수여야 한다. Python 순회 순서에 의존하면 rank마다 다른 block을 계산한다. 모든 rank가 `(parameter ID, block interval, owner)` digest를 합의한 뒤 collective를 시작한다.

factor statistic이 replicated인지 sharded인지, inverse root만 broadcast하는지, gradient가 이미 DP reduce된 것인지 확인한다. 같은 `GGᵀ`를 DP rank마다 더한 뒤 다시 sum하면 world-size 배가 된다. 반대로 local microbatch factor를 합치지 않으면 rank마다 다른 preconditioner를 만든다. statistic의 sample denominator와 collective 위치를 수식으로 적는다.

root 계산이 오래 걸려 다른 optimizer collective와 겹치면 모든 rank의 collective order가 같아야 한다. root cadence branch를 owner만 타고 비-owner가 다음 collective로 넘어가면 hang할 수 있다. zero-size block과 skipped parameter도 collective sequence ledger에 포함한다.

owner 장애 뒤 재시작할 때 factor와 root, cadence counter가 마지막 commit에 있어야 한다. root는 factor에서 재계산할 수 있지만 복구 시간과 수치 비결정성을 계약에 적는다. world-size 변경으로 owner가 바뀌면 stable block ID로 state를 이동한다. local rank key는 portable checkpoint key가 아니다.

**optimizer 비교를 재현 가능한 실험으로 닫는다**

**gradient replay가 algorithm 차이와 모델 noise를 분리한다**

동일한 parameter snapshot과 gradient sequence를 여러 optimizer에 공급한다. sequence에는 초기 warmup, steady state, outlier, zero, sign reversal, low-rank 구간을 넣는다. 매 step parameter delta, moment, preconditioner residual, state bytes를 기록한다. 모델 forward가 없으므로 optimizer 자체의 차이를 정확히 설명할 수 있다.

replay만으로 품질 우열을 결론 내리지는 않는다. 다음 단계는 같은 data order, token budget, initialization, evaluation cadence의 작은 학습이다. optimizer마다 안정적인 lr 범위를 공정한 search budget으로 찾는다. best run만 제시하지 않고 divergence와 variance를 포함한다.

시스템 비교는 tokens/s와 model FLOP utilization뿐 아니라 optimizer critical path, peak memory, collective bytes, checkpoint size와 save/load 시간을 본다. state 절감으로 늘어난 batch의 이익은 fixed-batch와 reinvested-memory 두 표로 나눈다. 알고리즘과 자원 재배치 효과를 구분하기 위해서다.

고장 실험은 optimizer step 중 rank kill, root 직전/직후 kill, overflow skip, state shard 누락, group ordering 변경을 포함한다. published checkpoint는 parameter와 모든 optimizer family state가 같은 OptimizerStepID를 가리켜야 한다. partial file이 있어도 manifest commit 전이면 무시한다.

resume 검증은 load 성공이 아니라 다음 세 delta를 uninterrupted branch와 비교하는 것이다. stochastic curvature estimator가 있으면 RNG와 iterator cursor도 복구한다. exact equality가 불가능한 backend라면 tolerance와 허용 원인을 미리 정한다. 단지 loss가 비슷하다는 판정은 state 완전성을 증명하지 못한다.

최종 선택 문서는 “Muon이 빠르다”처럼 끝나지 않는다. 대상 parameter role, update geometry, state/peak byte, distributed approximation, hyperparameter search, 품질 confidence interval, failure recovery를 적는다. 13장은 여기서 확정한 optimizer별 update clock과 overflow commit을 받아 scheduler가 언제 전진해야 하는지 결정한다.

**코드와 checkpoint를 잇는 최종 감사표**

**parameter group 생성 함수부터 읽는다.**

optimizer constructor보다 먼저 어떤 parameter가 어느 group에 들어가는지 본다. module name substring, ndim, weight decay exclusion, tied alias, expert role 규칙이 실제 algorithm 배정을 결정한다. model을 순회하는 Python order가 달라져도 stable group ID가 유지되어야 한다.

Muon과 AdamW 혼합 recipe에서는 한 logical parameter가 두 optimizer에 들어가거나 어디에도 들어가지 않는 오류가 가능하다. 모든 trainable parameter ID의 partition 보존식을 검사한다. alias는 storage 기준으로 한 번만 세고 frozen parameter는 명시적으로 제외한다.

optimizer `step` wrapper가 gradient clipping, scaler unscale, distributed reduction 뒤에 호출되는지 확인한다. Muon 내부 momentum과 orthogonalization 전에 global clipping이 일어나는지 source call graph를 그린다. Shampoo root cadence와 Sophia curvature cadence counter의 increment 위치도 같은 표에 둔다.

state dict는 Python 객체 모양이 아니라 semantic schema로 읽는다. Adam의 first/second moment와 step, Muon momentum과 backend counter, Shampoo factor/root/block metadata, SOAP basis/moment, Sophia curvature/estimator cursor를 stable parameter ID에 매핑한다. 누락 state의 default initialization이 resume를 조용히 바꾸는지 test한다.

**선택 기준은 품질 하나가 아니라 실패 비용까지 포함한다.**

작은 model에서 validation이 좋아도 target cluster에서 root computation이 critical path를 지배할 수 있다. 반대로 state가 작은 optimizer는 더 긴 sequence나 batch에 memory를 재투자할 수 있다. fixed-resource와 reinvested-resource 결과를 분리한다.

checkpoint 크기와 load 시간은 대규모 장애 복구의 일부다. Shampoo factor와 root를 모두 저장할지 재계산할지, Muon stacked momentum을 어떻게 shard할지에 따라 RTO가 달라진다. 저장 byte만 줄이고 재계산이 수십 분 걸리면 운영상 이득이 아닐 수 있다.

backend maturity도 평가한다. 지원 dtype/shape, distributed topology, compile/capture, test coverage, fallback, version migration을 표로 둔다. 실행하지 않은 조합은 “지원” 문서가 있어도 검증됨으로 표시하지 않는다.

최종 one-step certificate에는 parameter/gradient/state 입력 digest, source revision, backend, applied lr/decay, 출력 delta/state digest, collective ledger가 들어간다. uninterrupted, checkpoint resume, world-size reshard 세 경로가 허용오차 안에서 같은 certificate를 내야 한다.

이 certificate가 있으면 13장의 scheduler 오류와 optimizer 오류를 구분할 수 있다. applied lr는 같은데 delta가 다르면 optimizer state 또는 backend 문제다. delta는 비례하지만 lr가 다르면 clock과 schedule 문제다. 두 장의 경계가 수치적으로 닫힌다.

**독자를 위한 optimizer 선택 결정 트리**

먼저 목표가 연구 비교인지 production 도입인지 정한다. 연구 비교라면 gradient replay와 controlled small training으로 geometry를 분리한다. production 도입이라면 checkpoint, distributed owner, backend support와 장애 복구가 동등한 우선순위다.

memory가 병목이면 persistent state와 peak workspace를 따로 측정한다. Lion의 state 감소, Muon momentum, Shampoo factor/root가 실제 target shape에서 몇 byte인지 계산한다. 절약 memory를 batch에 재투자할지 그대로 여유로 둘지도 실험 조건이다.

행렬 구조를 활용하려면 parameter role inventory가 먼저다. dense hidden projection과 embedding/router/head를 분리한다. Muon 또는 Shampoo를 일부 tensor에만 적용하고 나머지는 AdamW로 유지하는 mixed recipe의 group ratio와 clock을 기록한다.

분산 환경에서는 full logical update가 필요한지 local approximation을 허용할지 결정한다. full gather 비용, block owner 방식, local shard geometry를 비교한다. algorithm 이름이 같아도 이 선택이 다르면 별 recipe다.

수치 안정성이 문제면 zero, rank-one, ill-conditioned fixture와 dtype/TF32 sweep을 먼저 한다. learning rate를 낮춰 NaN을 숨기기 전에 inverse-root 또는 Newton–Schulz residual을 확인한다.

품질 평가는 같은 token budget과 공정한 hyperparameter search로 한다. update norm, validation, seed variance를 함께 본다. optimizer 하나의 최상 run과 다른 optimizer의 default를 비교하지 않는다.

resume과 장애 복구가 통과하지 않으면 장기 run에 투입하지 않는다. cadence counter와 stochastic estimator cursor까지 저장하고, next-three-delta certificate를 확인한다. checkpoint 크기와 복구 시간도 결과표에 넣는다.

최종 선택은 단일 순위가 아니라 조건부 결론이다. 어떤 parameter·scale·topology·budget에서 어떤 optimizer가 유리했고, 무엇은 미검증인지 쓴다. 이 정직한 경계가 새로운 optimizer 유행을 재현 가능한 공학 판단으로 바꾼다.

결정 기록에는 반드시 반증 실험을 붙인다. AdamW에는 zero-gradient decay와 first-step bias correction, Lion에는 sign reversal, Sophia에는 curvature cadence, Muon에는 tall·wide·rank-deficient polar residual, Shampoo에는 inverse-root와 owner-rank failure, SOAP에는 basis rotation을 둔다. 각 실험에는 입력 gradient와 state, 예상 delta와 counter, 허용오차를 명시한다. 구현 revision이 바뀌면 이 작은 suite를 먼저 실행해 semantic drift를 찾는다. 전체 training loss가 비슷하다는 이유로 통과시키지 않는다.

분산 test에서는 global logical matrix와 local shard를 모두 보존하고, 어떤 approximation을 채택했는지 이름을 붙인다. checkpoint test는 load 성공이 아니라 다음 delta와 state transition을 비교한다. 운영 metric은 update norm, state byte, root 또는 orthogonalization residual, cadence miss, collective tail을 optimizer family별로 보여 준다. 이 증거가 모이면 optimizer 선택은 유행이나 단일 benchmark가 아니라 수학 함수, parameter 역할, cluster 비용, 복구 가능성을 함께 고려한 판단이 된다. 13장에 넘기는 applied-update certificate도 이 suite의 한 결과다.

**Muon의 직교화는 무엇을 보존하고 무엇을 버리는가**

Muon을 이해하는 가장 빠른 길은 행렬 gradient의 특이값 분해에서 출발하는 것이다. momentum을 반영한 update 후보를 \(G=U\Sigma V^\top\)라고 하자. polar factor \(UV^\top\)는 singular vector가 나타내는 입력·출력 방향을 보존하면서 singular value의 크기 차이를 평탄화한다. 큰 singular direction 하나가 update budget을 독점하고 작은 방향이 사라지는 현상을 줄이는 직관을 준다. 그러나 gradient의 크기 정보를 전부 보존하는 변환은 아니다. 그래서 전체 scale, learning rate, parameter shape에 따른 보정과 weight decay가 별도의 recipe 선택이 된다.

“orthogonal”이라는 말도 shape에 따라 정확히 읽는다. 정사각 행렬이면 \(UV^\top\)의 행과 열이 모두 직교하지만, tall matrix와 wide matrix에서는 각각 가능한 한쪽의 semi-orthogonality가 성립한다. flatten 규칙이 바뀌면 어느 축을 입력·출력 공간으로 보았는지도 바뀐다. convolution kernel이나 expert weight를 2차원으로 바꾸는 축 순서를 manifest에 넣는 이유다. embedding table, scalar, norm vector와 bias에는 같은 행렬 기하를 무비판적으로 적용하지 않고 별 optimizer group을 둔다.

실제 구현은 매 step SVD를 정확히 계산하기보다 Newton–Schulz 계열 반복으로 polar factor를 근사할 수 있다. 먼저 matrix norm으로 update를 정규화하고, 정해진 coefficient와 반복 횟수로 polynomial iteration을 적용한 뒤 scale을 복원한다. 여기서 coefficient, iteration count, transpose 여부, 계산 dtype은 모두 결과와 비용을 바꾼다. 반복 횟수를 늘리면 항상 좋은 것도 아니다. 근사 오차는 줄 수 있지만 kernel·collective 비용이 늘고 저정밀 rounding이 누적된다. target shape와 dtype에서 residual을 직접 측정한다.

residual 하나로 모든 것을 판단하지 않는다. tall matrix에는 \(XX^\top-I\)와 \(X^\top X-I\) 중 가능한 차원의 의미가 다르고, rank-deficient input에서는 정확한 identity를 기대할 수 없다. zero matrix는 normalization에서 division을 일으키지 않아야 한다. singular value가 여러 자릿수로 벌어진 fixture, rank-one fixture, nearly-zero fixture, tall·wide fixture를 둔다. 출력 norm, finite 여부, 방향 cosine, 적절한 orthogonality residual과 최종 parameter delta를 함께 비교한다.

momentum은 orthogonalization 앞에 적용되는지 뒤에 적용되는지가 알고리즘 의미를 바꾼다. 전자는 시간적으로 평활한 행렬을 polar 변환하고, 후자는 매 step 변환된 방향을 누적한다. Nesterov 형태라면 현재 gradient와 momentum이 어느 계수로 섞이는지도 고정한다. source의 `step` wrapper, momentum buffer update, backend 호출 순서를 한 call graph로 만들고, checkpoint에는 momentum과 step뿐 아니라 backend·coefficient recipe를 묶는다. 같은 momentum tensor를 복원해도 polynomial recipe가 바뀌면 다음 delta는 같지 않다.

**분산 Muon은 행렬의 소유권 선택이다**

logical weight가 tensor parallel이나 FSDP로 shard되면 각 rank가 보는 local matrix의 polar factor는 full matrix의 polar factor 조각과 일반적으로 같지 않다. 비선형 변환이므로 `polar(shard(G))`를 이어 붙인 결과와 `shard(polar(G))`가 교환되지 않는다. 이 한 문장이 분산 Muon 설계의 핵심이다. 구현은 full logical matrix를 모아 변환하거나, 특정 rank가 matrix를 소유하고 결과를 나누거나, local approximation을 명시적으로 채택해야 한다.

full gather 방식은 의미가 명확하지만 communication과 peak memory가 크다. owner-rank 방식은 matrix별 계산을 rank에 배분하고 update를 전달해 연산 균형을 맞출 수 있으나, owner assignment와 failure recovery가 state schema가 된다. local 방식은 저렴하지만 다른 optimizer geometry다. 논문이나 benchmark의 결과를 이식할 때 어떤 방식을 썼는지 모르면 “Muon”이라는 이름만 같은 별 실험을 비교하게 된다.

byte ledger는 matrix마다 작성한다. 입력 momentum gather byte, orthogonalized update scatter 또는 reduce byte, workspace, dtype conversion을 세고 collective가 critical path에 놓이는지 본다. 작은 matrix를 많이 묶으면 launch를 줄이지만 padding과 stack shape 제한이 생긴다. 서로 다른 shape를 bucket으로 묶는 기준과 tail 처리, bucket owner를 manifest에 둔다. rank별 matrix 수뿐 아니라 \(mn\)과 반복 비용으로 load balance를 평가한다.

checkpoint에서는 logical matrix ID, flatten rule, shard placement, momentum owner, backend counter를 저장한다. world size가 바뀌면 momentum을 logical tensor로 재조립해 새 mesh에 reshard한다. old local approximation state를 full-global recipe로 자동 변환하지 않는다. owner rank가 checkpoint commit 도중 죽는 시험에서는 다른 rank가 가진 parameter와 owner가 가진 momentum이 같은 step의 consistent cut인지 확인한다. 재개 후 첫 세 delta를 uninterrupted control과 비교한다.

**Shampoo와 SOAP: 축별 통계를 어떻게 읽는가**

Shampoo는 행렬 gradient의 좌우 축에 대한 통계를 별도로 누적한다. 단순화하면 \(G G^\top\)와 \(G^\top G\) 형태의 factor를 만들고 inverse root를 통해 update를 precondition한다. Adam의 좌표별 second moment가 basis 축에 묶여 있는 것과 달리, 축 내부의 correlation을 본다는 것이 핵심이다. 그 대가로 factor state와 inverse-root 계산이 커진다. 크기 \(m\times n\) weight에 대해 factor는 \(m^2+n^2\) 규모가 될 수 있으므로 큰 축을 block으로 나누는 선택이 필요하다.

block size는 성능 knob이면서 근사 기하다. 작은 block은 state와 root 계산을 제어하지만 block 사이 correlation을 버린다. 큰 block은 더 넓은 구조를 보지만 cubic한 matrix function 비용과 workspace가 커진다. block partition은 checkpoint에서 재현되어야 한다. shape가 같아도 block order가 바뀌면 state를 그대로 매핑할 수 없다. diagonal fallback, grafting optimizer, root update cadence도 recipe에 포함한다.

inverse root는 매 step 갱신하지 않을 수 있다. 통계 factor는 매 update 누적하면서 expensive root는 \(k\) step마다 계산하고 사이에는 stale preconditioner를 재사용한다. 따라서 checkpoint에는 factor뿐 아니라 마지막 root와 cadence counter가 필요하다. 저장 용량을 줄이려고 root를 버리고 load 때 다시 계산할 수 있지만, 재계산 dtype·iteration·ridge가 같아야 하며 복구 시간이 늘어난다. “derivable state”라는 이유만으로 RTO 비용을 0으로 세지 않는다.

SOAP 계열은 basis를 찾는 matrix preconditioning의 발상과 Adam류 moment를 결합해 읽을 수 있다. 중요한 질문은 basis가 언제 갱신되고, moment가 어느 basis에서 정의되며, basis가 회전할 때 old state를 어떻게 해석하는가다. basis tensor만 복원하고 basis-associated moment의 좌표계를 누락하면 load는 성공해도 다른 update가 된다. controlled 2D quadratic에서 basis rotation 전후의 state와 delta를 그려 보면 이 문제를 직관적으로 볼 수 있다.

분산 Shampoo/SOAP은 factor owner와 root owner를 명시한다. gradient shard로부터 factor contribution을 reduce할지, block owner가 통계를 모을지, root를 broadcast할지에 따라 collective가 다르다. root 계산이 느린 rank가 전체 step tail을 지배할 수 있다. dashboard에는 평균 root 시간보다 block별 p95/p99, owner imbalance, cadence miss를 둔다. owner failure 뒤 다른 rank가 state를 이어받는 시험도 필요하다.

**Sophia와 Lion을 이름이 아니라 state 식으로 비교한다**

Sophia류 optimizer의 구별점은 curvature 추정치를 denominator 또는 clipping에 활용한다는 데 있다. curvature를 매 step 정확히 계산하지 않고 일정 cadence로 추정할 수 있으므로, optimizer는 gradient moment 외에 curvature state와 갱신 시계를 소유한다. 추정 방식이 stochastic하다면 RNG/cursor도 재현 상태다. curvature update step에서 사용하는 batch와 일반 gradient batch의 관계, estimator 비용이 token budget에 포함되는지를 명시한다.

curvature clipping의 직관은 국소 곡률이 큰 방향에서 같은 gradient가 더 큰 함수값 변화를 일으킬 수 있으므로 update를 제한하는 것이다. 하지만 noisy curvature가 지나치게 작거나 음수·nonfinite라면 denominator가 위험해진다. epsilon, clipping threshold, estimator dtype과 reduction을 시험한다. cadence 직전 checkpoint와 직후 checkpoint를 각각 저장해 재개 후 curvature 갱신 시점이 보존되는지 본다. loss trajectory만 보면 한두 step의 cadence shift가 늦게 드러날 수 있다.

Lion은 second-moment tensor를 두지 않고 momentum 조합의 sign을 update 방향에 사용해 state memory를 줄이는 계보로 읽을 수 있다. sign은 magnitude 정보를 양자적으로 버리므로 learning rate와 decay의 scale이 AdamW에서 그대로 이식된다고 가정하지 않는다. gradient 부호가 교대하거나 0 주변에서 흔들리는 fixture는 momentum이 sign을 어떻게 안정화하는지 보여 준다. 정확히 0일 때 sign·decay·state가 어떻게 움직이는지도 구현별로 확인한다.

state byte가 줄었다는 장점은 fixed batch 비교와 memory 재투자 비교를 나눠야 공정하다. 같은 batch에서 optimizer만 바꾼 결과는 algorithm 차이를 보여 주고, 절약 memory로 sequence나 batch를 키운 결과는 system-level 이득을 보여 준다. 두 결과를 섞으면 품질 향상이 geometry 때문인지 더 많은 token/큰 batch 때문인지 분리할 수 없다. checkpoint byte와 restore time까지 포함하면 장기 run의 운영 이득도 계산할 수 있다.

**optimizer 최신성보다 증거의 폐쇄성을 평가한다**

새 optimizer를 검토할 때 논문 표, reference code, framework port, production fork를 같은 것으로 취급하지 않는다. 논문은 수식과 실험 조건을 정의하고, reference code는 저자가 실제 사용한 선택을 드러내며, framework port는 API·dtype·distributed 제약을 추가한다. production fork는 fusion이나 sharding으로 실행 계획을 다시 바꿀 수 있다. 각 층의 commit과 함수, test assertion을 연결하되 한 층의 성공을 다음 층의 성공으로 확대하지 않는다.

비교 표에는 최소한 parameter 적용 범위, persistent state byte, transient workspace, matrix function cadence, collective byte, update geometry, hyperparameter search budget, checkpoint portability를 둔다. “memory efficient”는 persistent state가 작은지 peak가 작은지 구분한다. “distributed”는 어느 topology와 approximation인지 적는다. “scalable”은 model size만 아니라 matrix shape 분포, world size, network와 recovery 결과를 요구한다.

공정한 optimizer 비교는 같은 초기 parameter, data order, token budget과 evaluation protocol을 공유한다. 동시에 각 optimizer에 합리적인 learning-rate·decay 탐색 예산을 준다. 한 optimizer의 공개 최적값과 다른 optimizer의 default를 비교하지 않는다. search 비용도 결과에 넣고 여러 seed의 confidence interval을 제시한다. train loss가 빨리 내려가도 validation, downstream, stability와 wall-clock/token 비용이 함께 좋아야 선택 근거가 된다.

마지막으로 negative evidence를 보존한다. 어떤 dtype에서 residual이 깨졌는지, 어떤 topology에서 owner bottleneck이 생겼는지, 어떤 checkpoint 변환이 미지원인지 기록한다. 실패를 지우면 다음 팀은 같은 조합을 다시 시험한다. 반대로 작은 fixture에서 실패한 backend를 전체 학습으로 밀어붙이지 않는다. 이 장의 산출물은 optimizer 순위표가 아니라 조건별로 재현 가능한 선택 함수다.

## 12.12 MaxText·OLMo Core 구현에서 분산 Muon을 감사한다

실제 구현의 axis 선택, Newton–Schulz 반복, scale과 parameter partition을 고정 함수와 test로 추적한다.

이 절은 로컬 MaxText revision `aae0386e8932e83cdb4311b101797023121eb38b`의 [`src/maxtext/optimizers/muon.py`](https://github.com/AI-Hypercomputer/maxtext/blob/aae0386e8932e83cdb4311b101797023121eb38b/src/maxtext/optimizers/muon.py)와 [`muon_test.py`](https://github.com/AI-Hypercomputer/maxtext/blob/aae0386e8932e83cdb4311b101797023121eb38b/src/maxtext/optimizers/muon_test.py)를 고정 좌표로 삼는다.

구현 파일에는 `_get_xxt_out_sharding`, `xxt`, `b_times_x`, 세 종류의 first/base Newton–Schulz iteration과 iterator, `MuonDimensionNumbers`, `MuonState`, `scale_by_muon`, reshape·orthogonalize·shape scaling 함수와 최종 `muon` builder가 분리되어 있다. 이 분해는 “Newton–Schulz를 몇 번 돈다”보다 훨씬 많은 설계 상태를 보여 준다.

`MuonDimensionNumbers`는 matrix optimizer를 2차원 tensor에만 묶지 않고 원래 tensor의 reduction axis와 output axis 의미를 표현한다. convolution이나 여러 expert 축이 섞인 tensor를 무조건 `reshape(first_dim, -1)`로 펴면 어느 축 correlation을 보존하는지 달라진다. dimension numbers를 model parameter role과 함께 저장하고, axis normalize 뒤 입력 축이 누락·중복되지 않는지 검사한다. negative axis와 tuple axis가 canonical form으로 바뀌어도 logical 의미가 같아야 한다.

`get_reshape_fns`와 `_normalize_axes`를 함께 읽으면 orthogonalization 직전 matrix shape와 적용 후 원래 shape 복원의 계약을 찾을 수 있다. reshape는 element permutation을 바꾸지 않아야 하고 inverse가 정확해야 한다. test는 다양한 rank와 axis tuple에서 `restore(reshape(x)) == x`를 확인한다. 단순 shape equality만으로는 transpose 오류를 잡지 못하므로 고유한 index pattern tensor를 사용한다.

`xxt`는 \(XX^\top\) 또는 선택된 축의 Gram 계산을 담당하며 sharding과 preferred accumulation dtype을 함께 볼 좌표다. 입력 dtype이 BF16이어도 Gram 누산을 더 높은 정밀도로 수행할 수 있다. 어떤 mesh axis가 contraction이고 output이 어디에 shard되는지 `_get_xxt_out_sharding`과 연결한다. local shard의 Gram과 global logical Gram이 필요한 경우 collective semantics가 다르다.

`b_times_x`는 iteration에서 polynomial matrix와 current X를 곱하는 경계를 드러낸다. 연산 순서를 대수식으로만 합치면 compiler가 만드는 temporary와 sharding constraint를 놓친다. trace에서 Gram, polynomial, matrix multiply의 HBM byte와 collective를 분리한다. preferred dtype, output sharding과 transpose 선택이 target accelerator에서 실제 kernel을 어떻게 바꾸는지 확인한다.

`_aol_first_newton_schulz_iteration`, `_schatten_first_newton_schulz_iteration`, `_base_newton_schulz_iteration`이 따로 있다는 사실은 초기 반복과 이후 반복이 동일하지 않을 수 있음을 뜻한다. coefficient family와 polynomial order가 수렴 영역, 계산량, 저정밀 오차를 바꾼다. config 이름만 저장하지 않고 선택된 iterator, coefficient와 반복 횟수를 recipe digest에 넣는다. source revision이 coefficient default를 바꾸면 checkpoint state가 같아도 next delta는 달라진다.

`orthogonalize`는 normalization, iterator 선택, transpose/reshape와 결과 복원을 잇는 중심 함수다. zero 또는 nearly-zero input에서 norm division을 안전하게 처리하는지 본다. tall/wide 판단으로 transpose optimization을 한다면 최종 semi-orthogonality가 어느 축에 성립해야 하는지 테스트 oracle도 함께 전환한다. complex input에서는 transpose가 conjugate transpose 의미를 갖는지 test를 기준으로 읽는다.

`_scale_update_for_width_transfer`, `_scale_update_for_consistent_rms`와 `scale_by_shape`는 polar factor 뒤의 크기 정보를 다시 넣는 설계다. singular value를 평탄화한 update의 Frobenius/RMS가 matrix shape에 따라 달라질 수 있으므로 폭이 다른 layer 사이 lr 의미를 맞추려는 전략을 구분한다. scaling strategy가 없거나 다른 전략인 결과를 모두 “Muon lr” 하나로 비교하지 않는다.

`scale_by_muon`은 optimizer transform state를 만들고 gradient update를 처리하는 Optax 계열 경계다. momentum accumulator가 언제 갱신되고 Nesterov 선택이 current gradient를 어떻게 섞는지, orthogonalization 전후 순서를 본다. parameter tree에서 dimension number가 없는 leaf는 identity, skip 또는 별 optimizer transform 가운데 무엇을 타는지 확인한다. masked transform과 chain 순서가 weight decay의 위치를 바꿀 수 있다.

최종 `muon` builder는 momentum transform, orthogonalization, shape scaling, learning rate와 decay를 어떤 순서로 합성하는지 보여 준다. Optax chain은 왼쪽에서 오른쪽으로 update transform을 적용하므로 함수 이름 목록이 곧 algebra order다. schedule callable의 step owner, decoupled decay가 parameter를 읽는 시점과 sign convention을 손으로 복원한다. builder default를 논문 수식과 동일하다고 추측하지 않는다.

### MaxText test를 구현 주장의 최소 반증으로 사용한다

같은 revision의 `muon_test.py`는 `MuonDimensionNumbers`의 여러 축 조합, reshape, optimizer state, orthogonalization과 mixed tensor target을 다룰 좌표를 제공한다. test의 존재를 지원 범위로 확대하지 않고 실제 parameterization을 읽는다. reduction axis와 output axis가 scalar, tuple, negative index일 때 어떤 shapes를 포함하는지 표로 만든다. target model의 expert·convolution shape가 표에 없으면 local fixture를 보완한다.

Newton–Schulz orthogonalization test는 real matrix의 Gram이 identity에 가까운지 검사하고 complex matrix에는 unitary 의미를 적용한다. 여기서 tolerance, input conditioning, matrix shape와 iteration setting을 기록한다. random well-conditioned matrix만 통과하면 rank-deficient, nearly-zero와 extreme aspect ratio는 미검증이다. local suite는 singular values를 통제한 matrix를 추가한다.

mixed tensor optimization target test는 dimension numbers를 가진 leaf와 일반 optimizer leaf가 한 parameter tree에서 함께 움직이는지를 보여 줄 수 있다. 모든 trainable leaf가 정확히 한 transform을 받고, state tree가 parameter tree와 구조적으로 맞는지 본다. shared/tied leaf가 JAX tree에서 어떻게 표현되는지는 별 model schema 문제다.

test가 update 결과를 기대값과 비교할 때 reference가 무엇인지 확인한다. 동일 구현을 다른 wrapper로 호출한 결과라면 독립 algebra oracle이 아니다. 작은 matrix에는 SVD로 구한 polar factor를 고정밀 reference로 두고 Newton–Schulz residual과 방향을 비교한다. optimizer 전체에는 손 계산 가능한 quadratic과 state transition을 추가한다.

sharding test가 있다면 단일 mesh 모양만으로 모든 topology를 지원한다고 말하지 않는다. reduction/output axis가 mesh placement와 맞는지, collective 후 global result를 재조립했을 때 unsharded reference와 같은지를 본다. compile 성공이나 shape 일치만으로 numerical collective correctness를 증명하지 않는다.

negative test는 잘못된 axis overlap, 누락 axis, scalar leaf에 Muon 강제, zero norm, unsupported complex/dtype와 incompatible sharding을 포함해야 한다. 기대 terminal이 명확한 오류인지 silent fallback인지 기록한다. silent skip을 허용하면 parameter inventory가 어떤 leaf를 AdamW로 보냈는지 반드시 출력한다.

### OLMo Core Muon의 parameter partition을 감사한다

로컬 OLMo Core revision `b7e9671d7ea48af94838c4f124703c3ae36f0c70`의 [`src/olmo_core/optim/muon.py`](https://github.com/allenai/OLMo-core/blob/b7e9671d7ea48af94838c4f124703c3ae36f0c70/src/olmo_core/optim/muon.py)는 `MuonAdjustLRStrategy`, `MuonConfig`, `default_group_overrides`, `build_groups`, `build_parallelism_config`, `create_optimizer`를 통해 production integration 경계를 보여 준다. actual update engine을 외부 Dion package에서 import한다는 사실도 dependency identity의 일부다.

OLMo Core commit만 고정하고 Dion version을 빠뜨리면 실행 수식을 고정한 것이 아니다.

`MuonConfig` 문서는 hidden weight layer에 Muon을 쓰고 input embedding, output layer와 2D가 아닌 parameter에 AdamW를 사용하는 의도를 명시한다. 이것은 group builder가 model semantics를 알아야 한다는 구체적 사례다. `default_group_overrides`와 `build_groups`에서 ndim, module role, model annotation 가운데 무엇이 실제 분류를 결정하는지 읽는다. 모든 trainable parameter의 partition 보존식을 검사한다.

matrix group과 Adam group은 서로 다른 lr·betas·decay와 state를 가질 수 있다. 같은 global scheduler가 base lr에 multiplier를 적용하는지 optimizer별 schedule을 갖는지 확인한다. checkpoint에는 group family와 stable parameter ID, Muon adjust-lr strategy를 저장한다. model architecture upgrade로 parameter ndim이나 이름이 바뀌면 자동 재분류가 old state mapping을 깨뜨릴 수 있다.

`build_parallelism_config`는 DP replicated, FSDP/HSDP shard와 optimizer가 요구하는 1D mesh 관계를 만든다. source 주석과 guard는 TP가 직접 지원되지 않는 조건을 보여 줄 수 있다. “framework가 TP를 쓴다”와 “Muon optimizer가 TP-sharded logical matrix를 올바르게 orthogonalize한다”를 구분한다. unsupported topology는 조용한 local approximation보다 명시적 오류가 낫다.

flattened mesh dimension 이름이 들어오는 경우 logical DP/EP 축이 optimizer가 보는 outer/inner shard로 어떻게 합쳐지는지 확인한다. process group membership과 matrix owner를 resolved config로 저장한다. mesh 이름 문자열이 맞아도 실제 rank coordinate와 tensor placement가 다를 수 있으므로 runtime manifest와 비교한다.

`create_optimizer`는 group partition, external optimizer class와 parallel config를 실제 객체로 묶는다. strict mode가 누락 parameter나 unsupported override를 어떻게 다루는지 test한다. compile recompile limit 같은 global 설정을 조절한다면 optimizer 생성의 side effect로 기록하고 다른 compiler workload에 미치는 영향을 평가한다.

`NorMuonConfig`는 neuron-wise adaptive learning-rate 변형을 추가한다. 이름이 비슷해도 beta2와 추가 state, scaling geometry가 다르므로 Muon checkpoint를 그대로 로드하지 않는다. parameter group inventory와 state byte, update oracle을 별 family로 둔다. benchmark에서 Muon과 NorMuon을 한 열에 합치지 않는다.

### 분산 matrix optimizer의 세 설계를 정량 비교한다

첫 설계는 full logical matrix를 모든 관련 rank에 all-gather한 뒤 각 rank가 같은 orthogonalization을 계산하고 결과 shard만 소비하는 방식이다. 구현이 단순하고 각 rank 결과가 동일하다는 장점이 있지만 gather peak와 중복 compute가 크다. BF16 momentum matrix \(m\times n\) 하나의 gather payload, workspace와 iteration GEMM FLOP을 matrix inventory 전체에 합산한다.

둘째는 matrix owner rank를 정해 gather·orthogonalization을 한 번 수행하고 update shard를 scatter하는 방식이다. 중복 compute를 줄이지만 owner load balance와 communication serialization이 문제다. owner assignment는 matrix 수가 아니라 shape별 polynomial compute와 byte를 가중해 만든다. checkpoint에서 owner가 바뀌어도 logical momentum을 복원할 수 있어야 한다.

셋째는 각 rank가 local shard만 orthogonalize하는 근사다. communication이 적지만 global polar factor와 같지 않다. TP shard axis, FSDP shard와 matrix reduction/output axis의 관계에 따라 근사 성질이 달라진다. recipe 이름에 local geometry와 shard axis를 명시하고 full reference와 cosine·residual·training effect를 측정한다.

네 번째 혼합 설계로 block owner 또는 inner/outer sharding을 둘 수 있다. matrix를 block으로 나누어 각 block의 polar factor를 구하면 Shampoo의 block approximation처럼 block 간 correlation을 버린다. block size와 partition axis가 geometry와 비용을 동시에 바꾼다. full, block, local 결과를 같은 이름으로 부르지 않는다.

communication 회계에는 gather/scatter byte뿐 아니라 padding, stack·unstack, transpose와 dtype cast를 넣는다. 작은 matrix를 동일 shape bucket으로 stack하면 collective와 kernel 수를 줄이지만 tail padding이 생긴다. bucket formation이 deterministic하고 checkpoint parameter ID와 같은 order인지 검사한다.

overlap은 optimizer compute가 backward 뒤 critical path에 놓이기 때문에 제한될 수 있다. layer별 gradient가 준비되는 즉시 momentum/orthogonalization을 시작하려면 gradient lifetime, accumulation과 clipping 순서가 맞아야 한다. global clipping이 모든 gradient를 기다린다면 조기 update와 충돌한다. overlap 최적화가 수학적 step order를 바꾸지 않는지 event graph로 증명한다.

fault injection은 owner rank kill, collective participant 누락, 잘못된 matrix shape bucket, duplicate parameter와 stale momentum shard를 포함한다. 최초 오류와 peer timeout을 구분한다. partial matrix update가 parameter에 적용되기 전에 transaction을 abort한다. 다음 run은 마지막 global committed optimizer state에서 시작한다.

## 12.13 수렴·basis 회전·curvature cadence를 수학과 코드로 잇는다

수렴 정리의 조건과 유한 정밀 구현을 구분하고 SOAP basis, Sophia cadence와 Lion sign update를 실제 state 전이로 연결한다.

polar factor를 구하는 반복은 입력 singular value가 특정 수렴 영역에 있도록 먼저 normalize한다. norm estimate가 너무 작으면 polynomial이 발산할 수 있고 너무 크면 초기 singular value가 0 근처에 몰려 정해진 반복 횟수 안에 충분히 평탄화되지 않는다. normalization norm의 종류와 계산 dtype, epsilon을 recipe에 넣는다.

polynomial iteration은 singular vector를 보존하고 singular value에 scalar polynomial을 반복 적용하는 관점으로 볼 수 있다. 따라서 2차원 matrix를 무작정 시각화하기보다 singular value map \(\sigma\mapsto f(\sigma)\)를 그리면 coefficient가 어떤 범위를 1 쪽으로 보내는지 직관적이다. low precision에서는 matrix multiply rounding이 이 이상적인 scalar map에 오차를 추가한다.

반복 횟수는 residual과 비용의 trade-off다. 1회부터 설정 상한까지 residual, output norm, candidate update 방향과 kernel 시간을 측정한다. residual 개선이 멈추거나 rounding으로 악화되는 지점을 찾는다. 논문 기본 횟수를 다른 dtype·shape에 그대로 복사하지 않는다.

rank-deficient matrix의 zero singular direction에는 고유한 직교 polar factor가 없거나 partial isometry로 이해해야 한다. identity residual을 무조건 기대하면 올바른 결과도 실패시킬 수 있다. input row/column space에서의 projection과 SVD reference를 사용한다. nearly rank-deficient case에서는 작은 singular direction의 민감도를 error budget에 반영한다.

tall matrix와 wide matrix를 transpose해 더 작은 Gram을 계산하는 최적화가 가능하다. transpose 전후 결과를 원래 orientation으로 되돌렸을 때 appropriate Gram이 identity 또는 projection에 가까운지 확인한다. shape가 경계에서 바뀌면 dispatcher가 다른 branch를 택할 수 있으므로 square, \(m=n+1\), extreme aspect ratio를 fixture에 넣는다.

BF16/FP16 input, FP32 accumulation과 output cast의 조합을 sweep한다. Gram이나 polynomial coefficient calculation은 높은 정밀도가 필요할 수 있다. 최종 update를 낮은 dtype으로 cast할 때 residual이 늘어도 parameter update error가 허용 범위인지 별도로 본다. residual 하나가 model quality를 완전히 대변하지 않는다.

### Shampoo factor와 inverse root의 상태 기계

행렬 gradient \(G_t\)에 대해 좌측 factor \(L_t\)와 우측 factor \(R_t\)를 누적한다고 하자. exponential 또는 sum accumulator인지, diagonal damping이 언제 더해지는지, bias correction이 있는지를 source 수식과 맞춘다. factor dtype과 symmetric 유지 방식도 기록한다. rounding으로 비대칭이 생기면 symmetrize하는지 test한다.

inverse p-th root 계산은 eigen decomposition, iterative method 또는 library primitive를 사용할 수 있다. eigenvalue clipping, ridge epsilon과 exponent가 결과를 바꾼다. ill-conditioned SPD fixture와 repeated eigenvalue, nearly singular factor를 사용한다. output이 finite라는 사실보다 \(P^p A\) 또는 적절한 residual이 identity에 가까운지를 본다.

root update cadence와 grafting에는 별 counter를 둔다. 첫 root가 준비되기 전 fallback update, cadence 사이 stale root와 새 factor의 조합을 정의한다. SGD/Adam grafting은 방향은 Shampoo에서, norm은 base optimizer에서 가져오는 식일 수 있으므로 어느 norm과 어느 시점인지 확인한다.

block partition은 parameter shape에서 deterministic하게 생성되어야 한다. 마지막 작은 block, bias/vector fallback, dimension merge 규칙을 manifest에 둔다. model revision으로 shape가 바뀌면 old factor block을 어떤 방식으로 변환할지 명시한다. 일반적으로 arbitrary shape change는 exact migration이 아니다.

분산에서는 factor contribution reduction, root owner, preconditioned update broadcast의 group을 각각 기록한다. factor는 gradient outer product이므로 local batch gradient를 먼저 평균했는지 sample별 통계를 모았는지 objective가 달라질 수 있다. framework 구현의 실제 순서를 논문 수식과 맞춘다.

checkpoint 선택지는 factor와 root 모두 저장, factor만 저장하고 root 재계산, block별 lazy restore다. 첫 방식은 storage가 크고 둘째는 recovery compute가 크며 셋째는 첫 사용 latency가 분산된다. RTO와 next-update parity로 선택한다. cadence counter와 damping recipe가 없으면 재계산 결과를 고정할 수 없다.

### SOAP의 basis 회전을 작은 예로 해부한다

2차원 quadratic의 gradient covariance가 좌표축과 45도 회전되어 있다고 하자. coordinate-wise Adam은 원래 축의 diagonal 통계만 보고 correlation을 놓친다. SOAP류 접근은 covariance의 eigenbasis 또는 근사를 찾아 그 basis에서 adaptive moment를 적용한 뒤 원래 좌표로 되돌리는 직관을 준다. basis가 정확하고 안정적이면 rotated anisotropy를 더 잘 다룰 수 있다.

그러나 basis는 시간에 따라 변한다. step \(t\)의 moment가 old basis 좌표로 저장되어 있는데 step \(t+1\)에 basis를 갱신하면 state를 새 basis로 회전해야 하는지, 새 gradient만 새 basis에 넣을지 구현 계약이 필요하다. 이 선택이 없으면 “basis와 Adam 결합”이라는 설명은 불완전하다.

작은 fixture는 known rotation matrix와 diagonal curvature를 만든다. basis update 전후 gradient, rotated gradient, moment, preconditioned update와 inverse rotation을 모두 저장한다. basis column의 sign은 eigenvector 비유일 수 있으므로 sign flip에 불변인 비교와 state alignment를 구분한다. repeated eigenvalue에서는 basis 자체가 불안정해도 subspace는 같을 수 있다.

basis update cadence는 compute와 staleness를 교환한다. 빠르게 변하는 training phase에서 오래된 basis가 유효한지 residual 또는 covariance alignment를 관측한다. cadence step을 checkpoint에 저장하고 resume 뒤 같은 batch에서 basis update가 일어나는지 본다. basis tensor를 저장하지 않고 재계산하면 stochastic data와 estimator cursor가 필요할 수 있다.

분산 basis 계산에는 covariance reduction과 eigen/root owner를 지정한다. rank별 local covariance basis를 독립적으로 쓰면 parameter replica가 다른 update를 받을 수 있다. global reduction, owner broadcast 또는 명시적 local approximation을 선택한다. eigenvector ordering과 sign/phase normalization이 rank마다 결정적으로 맞는지도 시험한다.

### Sophia의 curvature cadence를 trainer 사건과 연결한다

Sophia류에서 curvature estimate를 얻는 step은 일반 gradient step과 다른 forward/backward 또는 stochastic estimator를 요구할 수 있다. trainer가 어떤 cadence로 estimator batch를 선택하고 추가 compute를 token/FLOP 회계에 넣는지 본다. curvature update가 gradient accumulation boundary와 어긋나지 않게 한다.

curvature state가 diagonal이면 parameter와 같은 shape를 가지지만 값의 의미는 squared gradient와 다르다. Adam second moment checkpoint slot에 모양이 같다는 이유로 재사용하지 않는다. estimator type, beta, clipping bound와 counter를 schema에 둔다. negative curvature나 numerical noise를 어떻게 처리하는지 source guard를 읽는다.

curvature 기반 clipping은 좌표별 update에 upper bound를 줄 수 있다. threshold가 거의 모든 좌표를 clip하면 optimizer가 sign-like update가 되고, 거의 활성화되지 않으면 curvature state 비용만 낸다. clip fraction, curvature quantile과 update norm을 group별로 관측한다. threshold tuning은 validation과 stability를 함께 본다.

estimator step에서 AMP scale, dropout RNG와 distributed reduction이 일반 step과 다를 수 있다. same model state에서 single-rank high-precision reference와 비교한다. rank별 estimator batch가 다르면 어떤 평균이 global curvature를 정의하는지 목적 함수부터 쓴다.

resume fixture는 curvature 갱신 직전, 직후와 skipped optimizer step을 포함한다. attempted step이 아니라 committed update 또는 별 curvature clock 중 무엇이 cadence를 전진시키는지 확인한다. checkpoint 누락으로 cadence가 한 step 이동하면 짧은 delta tape가 이를 검출해야 한다.

**Lion의 sign update를 memory 절약 이상으로 이해한다**

Lion 계열은 first-moment 성격의 state를 유지하면서 gradient와 momentum 조합의 sign으로 parameter 방향을 결정한다. 좌표별 magnitude를 버리므로 dense update의 절대 크기는 learning rate와 parameter count에 강하게 연결된다. AdamW lr를 그대로 복사하지 않고 update RMS와 weight ratio를 맞추는 controlled sweep을 한다.

sign 함수는 0 근처에서 불연속적이다. low-precision gradient가 rounding으로 부호를 바꾸거나 0이 되면 update가 크게 달라질 수 있다. 작은 양·음 값, exact zero와 alternating sign fixture를 사용한다. momentum이 noise를 얼마나 평활하고 beta 선택이 sign switch 지연을 만드는지 본다.

decoupled weight decay가 sign update 전후 어디에 합성되는지 확인한다. zero gradient에서 momentum이 남아 있으면 sign update가 계속될 수 있으므로 “zero gradient decay fixture” 하나로 decay만 분리되지 않는다. fresh zero state와 nonzero momentum state를 모두 시험한다.

state memory가 AdamW보다 작아도 foreach/fused workspace와 master weight는 남을 수 있다. 실제 parameter당 byte를 source와 snapshot으로 계산한다. 절약 memory를 larger batch나 sequence에 재투자한 결과는 fixed-resource 결과와 분리한다. optimizer 자체 품질과 system 재투자 효과를 각각 보고한다.

Lion checkpoint에서 AdamW로 바꾸면 second moment가 없고 state 의미가 다르다. cold AdamW 전환, 별 transition과 old AdamW baseline checkpoint 가운데 하나를 선택한다. load helper가 state key를 무시하고 성공하더라도 trajectory migration은 검증되지 않았다.

**optimizer family의 분산 state byte를 실제 shape로 센다**

AdamW는 대체로 parameter당 first·second moment를 갖고 Muon은 적용 matrix에 momentum과 직교화 workspace를 갖는다. Shampoo는 block factor와 root, SOAP은 basis와 basis-associated moment, Sophia는 momentum·curvature, Lion은 momentum 중심 state를 가진다. 이 문장은 big-O 출발점일 뿐 target model inventory가 실제 byte를 결정한다.

model checkpoint에서 parameter name, role, global shape, dtype와 trainability를 추출한다. 각 optimizer의 group builder를 실행하지 않더라도 static rule로 후보 group을 만들고 state shape 식을 적용한다. tied storage는 한 번 세고 expert total/active parameter를 구분한다. block optimizer는 block partition까지 전개한다.

persistent byte, transient workspace, collective staging과 checkpoint serialization peak를 별 열로 둔다. Muon Newton–Schulz의 Gram과 temporary polynomial, Shampoo root decomposition workspace, foreach list가 순간 peak를 만들 수 있다. steady memory만으로 batch feasibility를 결정하지 않는다.

DP replication, FSDP shard, TP local parameter와 optimizer-specific owner placement를 mesh 축별로 적용한다. 어떤 state는 sharded parameter를 따라가고 어떤 factor는 다른 축에서 replicated될 수 있다. ideal world-size division과 실제 padding·replication 차이를 보고한다.

communication byte는 state update에 필요한 gather/reduce/broadcast를 센다. Muon full matrix gather, Shampoo factor reduction, SOAP basis broadcast, Sophia estimator reduction이 대상이다. frequency가 매 step인지 cadence인지 곱해 평균 byte와 burst critical path를 분리한다.

checkpoint byte와 load compute도 센다. derivable root/basis를 저장하지 않는 경우 storage 절약과 재계산 FLOP·RTO를 함께 적는다. world-size migration에서 full logical materialization이 필요한 peak도 포함한다. 이 회계가 있어야 memory efficient라는 표현이 운영 비용까지 포함한다.

**공정한 optimizer 연구 설계를 사전에 등록한다**

비교 가설은 “optimizer A가 B보다 좋다”가 아니라 model, data, token budget, topology와 metric을 포함한다. primary metric과 stability·memory·wall-clock secondary metric을 정한다. early termination과 실패 처리, seed 수와 confidence interval을 사전에 정해 cherry-pick을 막는다.

각 optimizer에 동일한 sweep 횟수만 주는 것이 항상 공정하지 않을 수 있지만 탐색 budget을 공개해야 한다. lr, decay, momentum/beta, warmup과 optimizer-specific cadence의 search space를 적는다. 한쪽은 논문 최적값, 다른 쪽은 default 한 점인 비교를 피한다.

fixed-token, fixed-FLOP, fixed-wall-clock과 fixed-memory 실험은 다른 질문이다. 최소한 fixed-token 품질과 target hardware의 time-to-quality를 분리한다. state 절약을 batch에 재투자하면 추가 token/update와 schedule을 기록한다.

data order와 evaluation contamination을 통제한다. same seed만으로 distributed sampler와 packing order가 같다고 가정하지 않고 consumed sample/token ID를 기록한다. optimizer가 throughput을 바꾸어 wall-clock evaluation 시점의 token 수가 달라지는 오류를 피한다.

failure run도 결과다. NaN, OOM, timeout, unsupported backend와 checkpoint failure를 별 terminal로 분류한다. hyperparameter를 사후 낮춰 성공시킨 run은 새 trial이다. 실패 비용과 recovery time을 production decision에 포함한다.

결론에는 평균 validation 하나가 아니라 effect size와 불확실성, cost와 미검증 범위를 함께 담는다. 작은 proxy에서의 이득을 target scale에 외삽할 때 matrix shape 분포, parallel approximation과 kernel maturity가 같은지 확인한다.

**matrix optimizer 장애 런북**

NaN이 orthogonalization 안에서 처음 생기면 input momentum norm, normalization, Gram, 각 iteration residual을 저장한다. zero/near-zero, extreme singular value, low-precision overflow 가운데 분기한다. learning rate를 낮추기 전에 device 함수의 finite invariant를 고친다.

residual은 나쁜데 training loss가 당장 정상이어도 backend를 승인하지 않는다. approximation이 의도한 geometry를 수행하지 않는 증거이기 때문이다. 반대로 residual이 좋고 loss가 나쁘면 scale adjustment, parameter group과 lr를 본다. geometry correctness와 optimization suitability를 분리한다.

분산에서 rank별 parameter가 갈라지면 orthogonalization 입력의 global/local 의미, owner assignment와 broadcast를 확인한다. full logical momentum checksum을 재조립하고 local update를 연결한다. TP local approximation이 의도였다면 rank별 차이가 global recipe와 일치하는지 oracle을 바꾼다.

step latency tail은 matrix shape bucket, owner load, root/orthogonalization cadence와 collective를 본다. 평균 kernel 시간보다 slowest rank의 block/matrix inventory를 비교한다. owner rebalance는 checkpoint mapping과 deterministic assignment를 함께 바꾼다.

resume 뒤만 차이가 나면 momentum뿐 아니라 iteration recipe, scale strategy, block/basis metadata와 cadence counter를 비교한다. derivable state를 load 때 재계산했다면 dtype·backend와 residual을 기록한다. next-three-delta가 uninterrupted control과 맞기 전 장기 run을 재개하지 않는다.

OOM은 persistent state, first-use lazy allocation, matrix workspace, gather staging과 checkpoint materialization으로 분해한다. block size나 bucket을 바꿀 때 geometry recipe도 바뀌는지 표시한다. offload는 traffic과 synchronization을 새 비용으로 추가한다.

backend upgrade regression은 source commit, external package, compiler와 kernel cache를 고정한다. reference path와 optimized path를 같은 matrix fixture에서 비교한다. silent fallback은 성능 결과를 별 backend로 분류한다.

**장의 최종 증거 행렬**

행은 Muon, NorMuon, Shampoo, SOAP, Sophia, Lion, LAMB·LARS·Adafactor와 AdamW 기준선이다. 열은 수식 state, 적용 parameter role, 고정 function, upstream test, local oracle, persistent/peak byte, collective, checkpoint, failure terminal과 품질 실험이다. 빈 셀은 미검증으로 표시한다.

Muon 셀에는 dimension numbers, reshape inverse, Newton–Schulz iterator·residual, shape scaling과 distributed owner가 들어간다. Shampoo에는 factor/block/root/cadence, SOAP에는 basis와 state rotation, Sophia에는 curvature estimator clock, Lion에는 sign transition과 state byte가 들어간다.

각 algorithm의 one-step certificate는 같은 logical parameter와 gradient digest를 받는다. algorithm이 달라 delta가 같은 것을 기대하지 않고 자체 고정밀 oracle과 state 식을 검증한다. 공정 비교 certificate는 별 층에서 token, budget와 search protocol을 맞춘다.

checkpoint certificate는 fresh, uninterrupted/resume, topology migration과 corrupted state negative case를 포함한다. load 성공이 아니라 다음 delta와 counter를 본다. external optimizer dependency와 framework wrapper version을 모두 identity에 넣는다.

분산 certificate는 full logical matrix와 local shard, process group, collective byte와 approximation 이름을 보존한다. single-rank PASS를 distributed support로 확대하지 않는다. owner failure와 sequence mismatch가 expected terminal에서 잡히는지 본다.

운영 certificate는 peak memory, steady throughput, tail, checkpoint RTO와 observability probe effect를 담는다. numerical gate를 통과하지 않은 결과는 speed 순위에서 제외한다. target shape·dtype·topology 밖의 결과에는 범위 표시를 붙인다.

독자는 최종 표에서 임의 optimizer를 골라 gradient가 어떤 state에 들어가고, 어떤 기하 변환과 scale을 거쳐, 어느 rank가 계산하며, 어떤 byte가 이동하고, 어떻게 저장·복구되는지를 말할 수 있어야 한다. 이 연결이 완성될 때 최신 optimizer는 유행 목록이 아니라 검증 가능한 학습 시스템 선택지가 된다.

**Adafactor의 factored state를 PyTorch 함수로 추적한다**

Adafactor는 행렬 parameter의 second-moment 추정치를 full tensor로 저장하는 대신 행과 열 방향 통계로 factor하여 memory를 줄이는 계보다. 이것은 Shampoo처럼 matrix inverse root로 좌우 preconditioning을 하는 것과 같지 않다. coordinate-wise squared-gradient estimate를 low-rank 구조로 근사해 복원하는 관점으로 읽는다. vector나 작은 tensor에는 unfactored state가 필요할 수 있어 parameter ndim별 branch가 생긴다.

PyTorch 고정 revision `3691693263d2b66a68867e39b7449876844e06cf`의 [`torch/optim/_adafactor.py`](https://github.com/pytorch/pytorch/blob/3691693263d2b66a68867e39b7449876844e06cf/torch/optim/_adafactor.py)는 `Adafactor`, `_init_group`, `step`, `_single_tensor_adafactor`, `_group_tensors_by_device_dtype_and_is_multidim`, `_multi_tensor_adafactor`, functional `adafactor`를 제공한다. Adam과 같은 읽기 순서를 적용하되 multidim/unfactored group 분기가 추가된다.

`_init_group`에서 row·column variance state가 어떤 shape로 생성되는지 실제 parameter shape와 맞춘다. step tensor placement, state dtype와 first-moment 사용 여부도 확인한다. first use에서 lazy allocation되는 byte는 full second moment와 비교한다. \(m\times n\) matrix에서 row/column state가 대략 \(m+n\) 규모라는 이득은 vector fallback과 작은 tensor overhead를 포함한 model inventory로 다시 센다.

`_group_tensors_by_device_dtype_and_is_multidim`은 foreach 실행 전에 tensor를 device·dtype·차원으로 분류한다. parameter와 gradient, row/column 또는 full state list가 regrouping 뒤 같은 index를 유지해야 한다. mixed model group에서 matrix와 vector가 섞일 때 각기 올바른 primitive를 타는지 test한다. empty subgroup과 sparse gradient terminal도 본다.

relative step size, parameter RMS scaling, clipping threshold와 decay schedule이 구현 option에 따라 결합될 수 있다. 다른 library의 Adafactor는 learning-rate 기본 의미와 epsilon tuple, warmup initialization이 다를 수 있으므로 이름으로 checkpoint나 recipe를 이식하지 않는다. 손 계산 fixture는 2×3 matrix와 vector를 함께 두어 factored/unfactored state를 한 step에서 비교한다.

factor reconstruction은 row mean과 column statistic의 normalization convention을 확인한다. constant gradient matrix, 한 행만 큰 matrix와 rank-one gradient를 쓰면 근사가 어떤 패턴을 보존하는지 드러난다. full squared-gradient EMA를 고정밀 oracle로 두되 factored 결과가 같아야 한다고 요구하지 않고 source 식과 맞는지 검사한다.

distributed shard가 row 또는 column을 자르면 local factor 통계가 global factor와 다를 수 있다. 어떤 axis 통계가 collective를 요구하는지 placement 대수로 적는다. FSDP local parameter shard를 임의 1D flatten으로 보면 원래 행·열 의미가 사라질 수 있다. optimizer가 full logical shape와 shard metadata를 받는지 확인한다.

checkpoint에는 row/column state, step, parameter shape와 factor axis convention을 둔다. world size 변경 뒤 새 shard에 factor를 어떻게 배치하는지 시험한다. full state를 factor state로 변환하거나 반대 방향은 exact trajectory 보존이 아닐 수 있으므로 명시적 migration이다.

**LARS와 LAMB의 trust ratio를 scale symmetry로 읽는다**

LARS와 LAMB는 큰 batch 학습에서 layer 또는 tensor별 parameter norm과 update norm의 비율을 이용해 local scaling을 조절하는 계보다. 직관은 서로 크기가 다른 layer에 하나의 global lr를 적용했을 때 상대 update 크기가 지나치게 달라지는 문제를 줄이는 것이다. 그러나 norm이 0에 가깝거나 bias·norm vector처럼 작은 parameter에는 trust ratio가 불안정할 수 있어 exclusion과 epsilon 정책이 필요하다.

LAMB는 Adam류 preconditioned update를 만든 뒤 weight decay를 결합하고 parameter norm/update norm으로 trust ratio를 정할 수 있다. 정확한 순서는 구현마다 확인한다. ratio clipping 유무, zero norm fallback, global gradient clipping과 trust ratio의 순서를 fixture에 둔다. layerwise라는 이름이라도 parameter tensor 단위인지 module 단위인지 source group을 본다.

로컬 Apex revision `9e3568a6f90fbc1996a06f8f9e99310bdaf2253a`의 [`apex/optimizers/fused_lamb.py`](https://github.com/NVIDIA/apex/blob/9e3568a6f90fbc1996a06f8f9e99310bdaf2253a/apex/optimizers/fused_lamb.py)는 `FusedLAMB.__init__`, `zero_grad`, `step`을 통해 fused 실행 경계를 보여 준다. Python `step`에서 gradient list, norm, AMP scale과 multi-tensor extension 호출을 잇는다. extension source와 binary build revision까지 닫지 않으면 실제 device 수식을 고정한 것이 아니다.

fused LAMB의 global norm과 per-parameter norm을 구분한다. gradient clipping을 위한 global norm, trust ratio를 위한 weight/update norm이 서로 다른 reduction이다. FP16/BF16 gradient와 FP32 master/state가 섞이면 norm accumulation dtype도 확인한다. distributed rank가 local shard norm만 사용하면 full logical tensor trust ratio와 달라질 수 있다.

반증 fixture는 parameter norm이 0, update norm이 0, 둘 다 nonzero인 세 case다. decay가 있는 zero gradient도 넣는다. ratio cap을 넘는 case와 작은 bias vector exclusion을 시험한다. reference는 fused kernel과 독립된 Python 고정밀 식으로 만든다. 최종 parameter뿐 아니라 moment와 computed trust ratio를 저장한다.

큰 batch 이득을 검증할 때 batch·lr·warmup과 trust ratio를 동시에 바꾼 bundle을 AdamW 한 점과 비교하지 않는다. fixed global batch에서 optimizer만 비교하고, batch scaling을 별 축으로 둔다. layer별 update-to-weight distribution과 time-to-quality를 관측한다.

**OLMo Core Lion 구현으로 sign state를 닫는다**

OLMo Core 고정 revision `b7e9671d7ea48af94838c4f124703c3ae36f0c70`의 [`src/olmo_core/optim/lion.py`](https://github.com/allenai/OLMo-core/blob/b7e9671d7ea48af94838c4f124703c3ae36f0c70/src/olmo_core/optim/lion.py)는 `lion_step`, `Lion.__init__`, `Lion.step`, `SkipStepLion.step_skipped`, `SkipStepLion.step`과 config class를 제공한다. 독립 update 함수와 wrapper를 함께 읽어 sign 수식, state initialization과 production skip 계약을 분리한다.

`lion_step`에는 parameter, gradient, momentum, learning rate, beta와 decay가 어떤 순서로 들어가는지 손으로 표를 만든다. current gradient와 old momentum을 한 beta로 섞어 sign update를 만들고, momentum state 자체는 다른 beta 조합으로 갱신할 수 있다. 두 beta의 역할을 하나의 “momentum”으로 뭉개지 않는다. sign을 취하기 전 조합과 저장할 조합을 각각 fixture로 확인한다.

`Lion.step`은 gradient가 있는 parameter를 순회하고 momentum을 lazy 초기화할 수 있다. `None` gradient와 zero gradient에서 step·decay·momentum이 어떻게 다른지 본다. complex, sparse 또는 unsupported dtype terminal도 constructor와 step test로 고정한다. optimizer state byte는 momentum dtype과 master parameter를 포함해 계산한다.

`SkipStepLion`은 11장의 SkipStepAdamW와 같은 failure framework를 공유할 수 있지만 algorithm state가 다르다. spike/Inf 판정이 모든 rank에 합의되고 sign update와 momentum이 함께 abort되는지 시험한다. scheduler와 consumed-token clock의 관계도 동일한 transaction ledger로 기록한다.

AdamW와 Lion 비교에서 lr scale과 decay를 공정하게 탐색한다. Lion의 dense sign update는 coordinate magnitude가 일정하므로 parameter 수와 group 구성에 따라 update RMS가 달라진다. 동일 lr 숫자가 동일 update budget을 뜻하지 않는다. 초기 update-to-weight ratio를 맞춘 probe와 각 optimizer 최적 sweep을 둘 다 보고한다.

**optimizer state를 기하적 좌표계로 분류한다**

AdamW의 moment는 원래 parameter coordinate마다 정의된다. Adafactor는 행·열 축을 사용해 squared-gradient 구조를 근사한다. Muon은 matrix의 singular vector 방향을 보존하며 singular value를 변환한다. Shampoo는 축별 covariance factor와 inverse root를 사용한다. SOAP은 이동하는 basis에서 adaptive state를 해석한다. 이 분류는 state tensor shape가 왜 다른지 설명한다.

좌표계를 쓰는 optimizer는 parameter reshape와 reparameterization에 민감하다. 같은 linear function도 weight를 transpose해 저장하거나 fused QKV로 합치면 matrix axes와 block partition이 달라진다. checkpoint converter가 값만 transpose하고 optimizer factor·basis를 변환하지 않으면 trajectory가 깨진다. model surgery와 optimizer migration을 하나의 schema 변화로 다룬다.

tied parameter는 두 module coordinate를 공유한다. optimizer는 storage를 한 번 갱신하지만 semantic role별 scaling이 충돌할 수 있다. embedding/head tie, expert sharing과 low-rank factor sharing을 inventory에서 표시한다. 서로 다른 optimizer group이 같은 storage를 소유하지 않게 한다.

tensor parallel은 coordinate를 물리적으로 나눈다. 비선형 matrix transform이 shard와 교환되는지 알고리즘마다 분석한다. Adam의 coordinate-wise update는 gradient normalization만 맞으면 shard-local 적용이 자연스럽지만 Muon polar, global trust ratio, Shampoo factor는 collective가 필요할 수 있다. “optimizer state sharding 지원”과 “global geometry 보존”을 구분한다.

quantization은 coordinate 값의 표현을 바꾼다. optimizer가 FP32 master에서 기하를 계산하고 quantized view만 model에 제공하는지, 낮은 dtype state에서 직접 계산하는지 확인한다. scale group이 matrix axis와 다르면 norm·Gram 의미도 바뀔 수 있다. quantized optimizer state는 별 error ledger와 migration이 필요하다.

**matrix optimizer와 weight decay의 합성 순서**

decoupled decay는 optimizer의 adaptive 또는 matrix preconditioned gradient와 별 parameter shrink 경로를 만든다. 그러나 shape-adjusted learning rate와 group multiplier가 decay에도 적용되는지 구현별로 다를 수 있다. Muon의 matrix update lr와 AdamW fallback group lr가 다르면 같은 decay 숫자가 step당 다른 shrink를 낼 수 있다.

Muon에서 polar transform 전에 gradient에 L2 term을 더하면 decay 신호도 singular-value 평탄화의 대상이 되어 decoupled 의미가 아니다. transform 뒤 parameter shrink를 적용하면 gradient geometry와 분리된다. zero-gradient·nonzero-parameter fixture가 두 경로를 구분한다. momentum에 decay가 들어가는지도 확인한다.

Shampoo/SOAP에서 coupled regularization은 factor와 basis 통계에 parameter 방향을 섞는다. decoupled decay는 preconditioner state에서 분리한다. 논문 실험의 regularization 방식과 framework default가 같은지 확인한다. 이름이 `weight_decay`여도 수식 위치를 source에서 본다.

Lion sign update와 decay는 서로 scale이 크게 다를 수 있다. sign component는 coordinate당 일정 magnitude이고 decay는 parameter magnitude에 비례한다. group별 parameter norm 분포가 regularization balance를 결정한다. update와 decay norm을 분리해 관측한다.

schedule이 lr를 낮추면 decoupled decay도 lr에 곱해지는 구현에서는 함께 약해진다. fixed decay per token을 원하면 schedule 면적과 update 수를 고려한다. optimizer 비교에서 decay 숫자만 같게 두는 것과 누적 shrink를 같게 두는 실험을 구분한다.

**최신 optimizer 논문을 코드로 옮길 때 생기는 번역 손실**

논문의 pseudocode는 tensor layout, dtype, distributed collective, fallback과 checkpoint를 생략한다. reference repository는 특정 model과 hardware에 맞는 선택을 암묵적으로 넣는다. framework port는 general API를 위해 branch를 추가하고 fused backend는 다시 지원 조합을 제한한다. 이 네 층의 차이를 source map에 표시한다.

수식 기호 하나가 구현 tensor 여러 개가 될 수 있다. Shampoo factor의 block list, Muon momentum bucket, SOAP basis와 rotated moment가 그 예다. 반대로 여러 논문 변수가 fusion kernel의 opaque buffer 하나에 합쳐질 수 있다. checkpoint schema와 debug dump가 semantic view를 복원해야 한다.

논문에서 full matrix를 가정해도 production은 local shard 또는 block approximation을 쓸 수 있다. 이 변경은 engineering detail만이 아니라 algorithm이다. benchmark 결과를 인용할 때 approximation, matrix selection, scaling strategy와 hyperparameter search를 같이 가져온다.

test도 번역 과정에서 줄어들 수 있다. reference의 convergence test가 framework port에서는 shape smoke test만 남을 수 있다. assertion strength를 비교하고 SVD/high-precision oracle, distributed reconstruction과 resume를 local suite에 보완한다.

성능 주장에는 kernel library와 accelerator가 조건으로 붙는다. Newton–Schulz GEMM이 큰 dense matrix에서 효율적이어도 작은 ragged matrix bucket에서는 overhead가 클 수 있다. root decomposition이나 all-gather가 다른 topology에서 tail을 만들 수 있다. target shape histogram과 cluster에서 재측정한다.

**최종 선택 회의의 실제 질문**

첫 질문은 어떤 parameter가 candidate optimizer를 받는가다. model inventory에서 정확한 logical ID와 총 byte를 보여 준다. “hidden weights”라는 말만으로 fused QKV, expert, embedding과 adapter의 경계를 넘기지 않는다.

둘째는 한 update가 어떤 기하 함수인가다. momentum, factor/basis/curvature, normalization, matrix transform, scale, decay 순서를 적는다. 고정 source 함수와 독립 oracle이 이를 뒷받침해야 한다.

셋째는 분산에서 global 의미를 보존하는가다. full gather, owner, block/local approximation과 collective byte를 밝힌다. single-device 논문 결과를 sharded implementation에 그대로 붙이지 않는다.

넷째는 memory를 어디서 절약하고 어디에 쓰는가다. persistent, workspace, communication staging과 checkpoint peak를 나눈다. 절약분을 batch에 재투자한 결과와 fixed batch 결과를 분리한다.

다섯째는 장애 뒤 같은 곳으로 돌아오는가다. state schema, cadence counter, dependency revision, topology reshard와 next-three-delta certificate를 본다. root/basis 재계산 비용을 RTO에 넣는다.

여섯째는 품질 비교가 공정한가다. token/FLOP/wall-clock budget, search space, seed와 failure를 공개한다. 한 optimizer의 성공 run만 골라 비교하지 않는다.

일곱째는 어떤 조건을 검증하지 않았는가다. dtype, shape, GPU, topology, compiler와 model scale의 범위를 명시한다. 미검증을 지원됨으로 바꾸지 않는다.

이 질문에 대한 답이 모두 같은 recipe와 artifact lineage를 가리킬 때만 optimizer 교체를 승인한다. 그렇지 않으면 새 알고리즘의 흥미로운 수학과 production readiness를 분리해 다음 실험 항목으로 남긴다.

**하나의 4×3 행렬로 optimizer 기하를 나란히 본다**

동일한 parameter \(W\in\mathbb{R}^{4\times3}\)와 gradient \(G\)를 고정한다. G의 singular value를 일부러 크게 벌리고 한 행에 noise를 더한다. AdamW는 각 좌표의 first·second moment와 decoupled decay를 계산한다. Muon은 momentum matrix의 singular direction을 유지하며 polar 근사를 만든다. Adafactor는 행·열 squared-gradient 통계로 full second moment를 근사한다. Shampoo는 좌우 factor와 inverse root를 사용한다. 이 하나의 fixture에서 state shape와 update geometry 차이가 보인다.

첫 artifact는 입력이다. W, G, logical input/output axis, dtype와 group option을 저장한다. 둘째는 algorithm별 intermediate다. Adam denominator, Muon normalized matrix와 iteration residual, Adafactor row/column state, Shampoo factor/root를 기록한다. 셋째는 decay 전후 update와 parameter delta다. 최종 delta만 보면 원인이 다른 scaling을 구분하기 어렵다.

G를 회전한 두 번째 fixture를 만든다. coordinate optimizer는 basis 변화에 따라 trajectory가 달라지고 matrix-aware optimizer도 axis·rotation invariance가 각기 다르다. “회전에 강하다” 같은 일반 문장 대신 원래/회전 fixture의 update를 다시 좌표계로 되돌려 비교한다. shape scaling과 decay가 invariance를 깨뜨릴 수 있음을 포함한다.

rank-one, zero, ill-conditioned와 extreme aspect ratio fixture를 추가한다. Muon에는 partial-isometry oracle, Shampoo에는 factor conditioning, Adafactor에는 factor approximation, Lion에는 sign switch가 필요하다. 모든 optimizer에 같은 성공 조건을 강요하지 않고 각 수식의 invariant를 정의한다. 공통 조건은 finite, state lifecycle과 deterministic recipe identity다.

분산 열에서는 W를 row shard, column shard와 flattened shard로 각각 나눈다. local update를 재조립해 full reference와 비교한다. coordinate-wise AdamW는 gradient normalization이 맞으면 자연스러운 local 결과를 기대할 수 있지만, Muon·trust ratio·factor optimizer는 collective 또는 근사 차이가 나타난다. 차이를 오류인지 의도된 approximation인지 recipe가 결정한다.

checkpoint 열에서는 첫 gradient 적용 뒤 저장하고 두 번째 gradient를 적용한다. uninterrupted와 resume의 intermediate·delta를 비교한다. matrix axis metadata, block partition, basis, cadence와 external backend revision을 하나씩 누락한 negative checkpoint를 넣는다. validator가 어느 누락을 잡는지 기록한다.

precision 열에서는 FP32 reference, BF16 input/FP32 accumulation, 낮은 state precision을 나눈다. orthogonalization residual, inverse-root residual과 parameter delta error가 서로 같은 추세인지 본다. residual이 조금 커도 final delta가 안정적일 수 있고 그 반대도 가능하다. tolerance를 metric별로 둔다.

performance 열은 state byte, first-use peak, warm kernel, collective와 checkpoint RTO를 기록한다. 작은 4×3 fixture의 시간은 의미가 없으므로 동일 검증을 target shape histogram의 synthetic matrix 묶음으로 확장한다. 작은 fixture는 의미를, 큰 synthetic fixture는 실행 비용을 검증한다.

**최신 optimizer를 장기 운영에 넣기 전의 단계적 승격**

첫 단계는 pure function이다. fixed matrix와 state를 받아 expected update를 내는 high-precision oracle과 source implementation을 비교한다. random test만 쓰지 않고 pathological fixture를 넣는다. 이 단계에서 state 식이 틀리면 model 학습으로 넘어가지 않는다.

둘째는 optimizer object lifecycle이다. parameter group, lazy state, multiple steps, save/load와 dtype/device 이동을 검사한다. dependency와 config default를 고정한다. wrapper가 algorithm 일부를 바꾸는지 call graph로 확인한다.

셋째는 작은 model이다. 동일 checkpoint와 data에서 gradient snapshot replay, 짧은 training과 validation을 수행한다. parameter role partition과 fallback optimizer를 감사한다. 이 단계의 성공은 target scale 품질의 증거가 아니라 integration correctness 증거다.

넷째는 distributed synthetic이다. 실제 mesh와 비슷한 shard, owner, collective, checkpoint reshard와 rank failure를 작은 tensor로 재현한다. full logical oracle와 approximation contract를 비교한다. hang과 partial update terminal을 검증한다.

다섯째는 proxy scale이다. target matrix shape 분포, dtype와 compiler/backend를 사용해 memory, throughput, tail과 short quality를 측정한다. algorithm-specific hyperparameter search를 공정하게 수행한다. unsupported shape fallback 비율을 보고한다.

여섯째는 제한된 production canary다. checkpoint cadence, monitoring, rollback lineage와 failure budget을 갖춘다. 첫 run에서 미검증 topology나 optimizer migration을 동시에 도입하지 않는다. anomaly가 생기면 fixed gradient replay와 intermediate residual로 최초 divergence를 찾는다.

각 승격에는 명시적 입구·출구 gate가 있다. 이전 단계 artifact가 없으면 다음 단계의 매끈한 loss curve로 대체하지 않는다. 실패하면 해당 단계로 돌아가 source, 수식 또는 system 가설을 한 축씩 고친다. 이 절차가 새 optimizer를 무조건 보수적으로 막는 것이 아니라 실패 원인을 작고 저렴한 단계에서 발견하게 한다.

최종 production 표시는 특정 revision, parameter inventory, matrix geometry, topology와 hardware 범위에만 유효하다. source, model shape, CUDA/compiler, world size 또는 checkpoint schema가 바뀌면 영향받는 단계부터 재승격한다. 검증 이력을 그대로 승계하지 않는다.

**다음 장으로 넘기는 optimizer clock 계약**

각 optimizer에는 `attempted update`, `committed update`, algorithm-specific cadence를 구분해 둔다. AdamW·Lion의 moment는 committed update마다 움직이고, Muon momentum과 orthogonalization도 같은 commit에 묶인다. Shampoo root, SOAP basis와 Sophia curvature는 별 cadence counter를 가질 수 있다. scheduler가 어떤 counter를 읽는지 명시한다.

overflow나 spike skip에서는 mathematical optimizer state가 멈추어야 하지만 curvature probe나 basis estimator를 이미 실행했을 수 있다. 이 auxiliary state도 abort할지 독립 commit할지 정한다. 부분 commit을 허용하면 checkpoint와 replay가 그 사건 순서를 보존해야 한다.

gradient accumulation과 matrix cadence의 단위도 고정한다. microbatch마다 factor를 누적하는지 최종 averaged gradient로 한 번 갱신하는지는 다른 통계다. root every 10이라는 설정이 microbatch, attempted 또는 committed update 10회인지 config card에 쓴다.

분산 owner failure로 update가 abort되면 모든 rank의 optimizer clock이 멈춘다. 재시도에서 같은 gradient와 RNG를 복원할 수 없다면 consumed-token clock만 앞설 수 있다. 13장은 이 여러 clock을 schedule과 token budget에 연결한다.

13장으로 넘길 때는 optimizer family와 group별 base lr·multiplier만 적어서는 부족하다. 마지막으로 commit된 step, cadence counter, 건너뛴 event, 누적 global valid token, checkpoint source와 다음에 실행할 update까지 함께 넘긴다. 그래야 scheduler가 복구된 optimizer state와 다른 시계의 lr를 적용하는 일을 막을 수 있다.

마지막으로 handoff의 수치를 고정 gradient 한 번으로 검산한다. 저장 직전 state를 load해 다음 update를 계산하고, 기록된 learning rate와 cadence branch가 실제 함수에서 선택되는지 확인한다. Muon의 iteration recipe, Shampoo의 root age, SOAP의 basis version, Sophia의 curvature age와 Lion의 momentum step이 모두 표시되어야 한다. 단순한 `global_step` 하나로 이 상태를 압축하지 않는다. scheduler는 optimizer가 성공적으로 commit했다는 사건을 받은 뒤에만 전진하며, 별 cadence는 자기 owner가 직렬화한다. 이 계약까지 닫혀야 12장의 비교 결과가 13장의 시간축에서도 같은 optimizer를 뜻한다.

미검증 optimizer·dtype·topology는 handoff 표에서 명시적으로 제외한다. 이후 실험이 범위를 넓히면 동일한 수식, lifecycle, distributed, resume와 failure gate를 다시 수행한다. 이름이 같다는 이유로 검증 범위를 자동 승계하지 않는다.

**AdamW와의 공정 비교를 추정 문제로 다시 쓴다**

행렬 optimizer 비교의 첫 질문은 어느 방법의 최저 validation loss가 아니라 무엇을 같게 두고 어떤 효과를 추정하는가다. 고정 token 효과는 동일한 학습 token과 평가 token에서 optimizer family만 바꾼 결과다. 고정 compute 효과는 optimizer의 행렬곱, root 계산과 통신까지 포함한 달성 FLOP 또는 accelerator 시간을 맞춘 결과다. 고정 wall-clock 효과는 compiler warmup, checkpoint와 장애 복구까지 포함한다. 세 추정값은 서로 바꿔 쓸 수 없다. Muon이 추가 GEMM을 쓰면서 update 수를 줄였다면 token 효율과 시스템 효율이 반대 방향일 수 있다.

AdamW 기준선은 약한 한 점이 아니라 동일한 탐색 예산을 받은 경쟁자다. 각 family에 안정 learning-rate 구간, weight decay, warmup과 algorithm 고유 옵션을 사전에 배정한다. Muon에는 momentum, Nesterov, 반복 수와 shape scale, Shampoo에는 block와 root cadence, SOAP에는 basis cadence, Sophia에는 curvature cadence와 clipping, Lion에는 두 beta와 decay가 있다. 후보 수가 다른 경우 총 trial compute와 조기 중단 규칙을 맞춘다. 실패한 trial도 분모에 남겨 선택 편향을 보인다.

첫 비교는 model, data order, global token batch, sequence policy, precision, clipping과 scheduler multiplier를 고정한다. optimizer별 base lr 숫자를 억지로 같게 하지 않는다. 대신 각 방법의 안정 범위를 같은 pilot budget으로 찾고, 선택된 점과 주변 민감도를 함께 보고한다. 두 번째 비교는 state 절약분을 activation이나 batch에 재투자한다. 이 결과는 optimizer 자체 효과가 아니라 optimizer와 시스템 재배치의 결합 효과다. 두 표를 합치면 memory 절약을 품질 향상으로 오인한다.

seed는 data order, initialization과 dropout을 가능한 한 짝지어 분산을 줄인다. run 하나가 수치 실패하면 성공 seed만 평균하지 않는다. failure rate, time-to-failure와 실패 전 소비 자원을 별 결과로 둔다. validation checkpoint를 같은 token 위치에 놓고, asynchronous evaluator가 읽은 정확한 CheckpointID를 기록한다. best checkpoint 선택 규칙은 결과를 보기 전에 정한다.

효과 표에는 평균 차이만 아니라 seed별 paired difference, 불확실성 구간, 최악 seed, peak memory, optimizer collective byte, update RMS와 실제 token/s를 둔다. 작은 seed 수에서 정규 근사를 과신하지 않고 bootstrap이나 개별 점을 같이 보인다. 여러 benchmark와 여러 checkpoint에서 최선 결과를 고르면 multiplicity가 커지므로 primary metric과 terminal token을 먼저 고정한다.

공정 비교의 negative control은 공통 AdamW parameter group이다. hybrid Muon run에서도 embedding, norm, bias와 제외된 head의 첫 여러 delta가 all-AdamW control과 recipe상 기대한 관계를 가져야 한다. 이 공통군이 달라지면 optimizer family 효과가 아니라 scheduler, grouping, reduction 또는 data drift가 섞인 것이다. candidate matrix만 바뀌고 공통군은 같다는 증거를 gradient replay로 닫는다.

**저정밀 행렬 iteration의 오차 예산**

저정밀 안정성은 최종 loss가 NaN이 아닌지만으로 판정하지 않는다. 입력 cast, norm reduction, polynomial matmul, accumulator, coefficient 저장, output cast와 parameter 적용을 각각 분리한다. BF16 입력과 FP32 accumulation이라는 문구도 fused kernel 내부 reduction이 실제 FP32인지 보장하지 않는다. profiler trace와 작은 값 probe로 실행 dtype을 확인한다.

Newton–Schulz 입력 (X_0=M/(\|M\|_F+\epsilon))에서 norm이 underflow하면 zero가 아닌 matrix가 zero 경로로 들어간다. 반대로 몇 개 outlier가 norm을 독점하면 대부분 singular direction이 BF16 해상도 아래로 내려간다. 테스트는 전체 scale만 바꾸는 경우, 한 singular value만 작게 하는 경우와 원소 상쇄가 큰 경우를 나눈다. scale equivariance, finite, residual과 update cosine을 각각 측정한다.

반복 다항식은 중간 (XX^T)와 고차 항을 만든다. intermediate가 overflow한 뒤 마지막 계수 조합에서 우연히 finite가 되는 경우도 있으므로 반복별 maximum, minimum nonzero, Frobenius norm과 residual을 남긴다. residual 하나가 단조 감소해야 한다고 가정하지 않는다. 채택 coefficient는 유한 반복에서 원하는 근사를 만들 수 있고 고전적 수렴식과 다른 transient를 보일 수 있다. 허용 반복 범위와 입력 norm 범위를 함께 고정한다.

tall matrix를 transpose해 wide 경로로 처리하면 어떤 축의 Gram을 만드는지가 바뀐다. transpose 전후 scale factor와 output inverse transpose를 assertion한다. shape가 `[1,n]`이거나 `[m,1]`이면 polar 방향이 사실상 정규화된 vector가 되며 일반 matrix residual이 부적절할 수 있다. 매우 작은 matrix는 AdamW fallback을 쓰는 정책도 가능하지만 silent fallback 비율을 parameter inventory에 기록한다.

Shampoo inverse root는 eigenvalue floor, damping과 decomposition dtype의 결합을 본다. nearly singular factor에서 작은 eigenvalue를 과도하게 뒤집으면 preconditioned update가 폭발한다. damping을 키우면 finite는 쉬워지지만 약한 방향 적응을 잃는다. factor symmetry 오차, 최소 eigenvalue, root residual, update amplification을 함께 sweep한다. decomposition 실패 시 identity, diagonal 또는 이전 root를 쓰는 fallback은 서로 다른 algorithm이며 event에 이유와 root age를 남긴다.

SOAP은 basis의 부호 뒤집힘보다 subspace와 transformed update를 본다. repeated eigenvalue에서는 유효 eigenvector가 임의 회전할 수 있다. basis 원소의 bitwise 비교는 false alarm을 만든다. projector 거리, rotated moment를 원좌표로 돌린 결과와 다음 delta를 비교한다. basis를 낮은 precision으로 저장했다가 재개할 때 orthogonality가 무너지면 재직교화가 state migration인지 deterministic reconstruction인지 선언한다.

Sophia curvature가 음수이거나 매우 작게 추정될 때 denominator floor와 clipping 순서를 확인한다. FP16 curvature state가 작은 양수를 zero로 만들 수 있다. Lion에서는 sign 직전 합이 zero 주변일 때 cast 한 비트가 방향을 뒤집는다. 두 family 모두 FP32 state control, 저정밀 state와 stochastic rounding 후보를 고정 gradient tape에서 비교한다. memory 절약과 방향 오류를 같은 표에서 교환한다.

정밀도 승격 gate는 well-conditioned 평균 사례가 아니라 위험 분위수를 기준으로 한다. target model에서 shape, condition proxy와 update ratio를 표본 수집하고, synthetic fixture가 그 꼬리를 덮는지 확인한다. 새 accelerator나 compiler에서 fusion이 달라지면 dtype 이름이 같아도 gate를 다시 연다. tolerance는 절대 오차 하나가 아니라 residual, cosine, RMS와 다음 loss 변화의 공동 조건이다.

**함수 단위 source pin과 실행 가능한 시험 지도**

고정 commit은 repository 이름을 적는 것으로 끝나지 않는다. commit hash, dependency lock, 파일 blob digest, 함수 qualified name, 호출 wrapper, 선택 backend와 test node ID를 한 행에 둔다. nanochat의 `_compute_muon`과 `_reduce_muon`, MaxText의 Newton–Schulz iterator와 `scale_by_muon`, OLMo Core의 group builder와 외부 Dion dependency처럼 실제 update가 여러 저장소에 걸치면 모든 좌표를 고정한다.

PyTorch port에서는 `Optimizer.step`, foreach/fused dispatch, parameter-group construction, state serialization과 scheduler 호출 위치를 따로 추적한다. 공개 구현의 순수 함수만 맞아도 wrapper가 `grad is None`, sparse gradient, capturable step tensor, AMP found-inf와 differentiable 옵션을 다르게 처리할 수 있다. local test 이름은 source 함수의 어느 branch를 실행했는지 coverage와 함께 남긴다.

함수 시험의 첫 층은 입력 계약이다. rank 0·1 tensor, tall/wide matrix, non-contiguous view, transposed stride, empty dimension, sparse gradient, complex와 unsupported dtype을 넣는다. 기대 결과가 오류, AdamW fallback 또는 matrix transform인지 명시한다. 단순히 예외가 났다는 사실보다 예외가 parameter를 조금도 수정하지 않았고 모든 rank가 같은 terminal에 도달했는지 본다.

둘째 층은 algebra다. FP64 SVD polar, eigendecomposition inverse root와 손 계산 sign/curvature update를 독립 oracle로 둔다. production 함수와 oracle이 같은 helper를 공유하지 않게 한다. coefficient, epsilon, transpose, scale과 decay 순서를 하나씩 바꾼 mutant가 test에서 반드시 실패하는지 mutation check를 한다. mutant가 통과하면 assertion이 알고리즘을 고정하지 못한 것이다.

셋째 층은 lifecycle이다. lazy state 생성 전 checkpoint, 첫 step 뒤, cadence 직전과 직후, gradient가 `None`인 step, overflow skip과 parameter freeze/unfreeze를 지난 state를 저장한다. load 뒤 다음 세 delta와 state counter를 uninterrupted control에 비교한다. pickle 또는 state dict가 성공했다는 사실만으로 semantic resume를 선언하지 않는다.

넷째 층은 분산 실행이다. logical full gradient를 정하고 각 sharding 경로에 투입한 뒤 재조립 delta를 full reference와 비교한다. approximation을 의도했다면 equality 대신 선언된 block/local 식을 oracle로 쓴다. collective 호출 횟수, payload, group과 sequence number를 trace해 DDP 중복 reduction을 찾는다. single-rank test를 distributed correctness로 확대하지 않는다.

다섯째 층은 최적화 backend다. eager reference, compiled, foreach, fused와 vendor kernel을 같은 fixture로 비교한다. fallback 발생 수와 이유를 결과에 포함한다. 빠른 경로가 unsupported shape를 조용히 느린 경로로 보내면 품질은 맞아도 throughput 주장이 틀릴 수 있다. 반대로 빠른 경로만 수치가 다르면 장기 loss 이전에 intermediate tape로 최초 divergence를 찾는다.

source upgrade 시에는 diff에서 수식과 default뿐 아니라 group rule, state key, dtype cast, collective와 test tolerance 변화를 분류한다. 기존 artifact를 새 commit에 그대로 연결하지 않는다. 영향받은 층부터 test를 재실행하고, 함수가 삭제되거나 이름이 바뀌어도 blob과 call graph로 의미의 이동을 확인한다.

**분산 상태와 장애 주입의 완결된 행렬**

분산 설계는 replicated, owner, sharded와 block-local 네 경우를 구분한다. replicated state는 단순하지만 모든 rank가 같은 gradient와 연산 순서를 봐야 한다. owner 방식은 state memory와 root 계산을 집중시키며 broadcast 또는 gather가 필요하다. sharded 방식은 memory를 나누지만 polar나 factor의 global 축을 복원하는 collective가 생긴다. block-local 방식은 통신을 줄이는 대신 full-matrix 기하를 근사한다.

상태 원장은 parameter shard, momentum, Gram/factor, cached root 또는 basis, cadence counter, workspace와 communication staging을 별 항목으로 센다. steady-state snapshot과 first-use peak, root-update peak, checkpoint serialization peak가 다르다. allocator reserved byte만 보고 algorithm state라고 부르지 않는다. logical tensor numel, physical padding, replication factor와 dtype에서 예상 byte를 계산하고 실제 snapshot과 차이를 설명한다.

통신 원장에는 collective 종류, group size, 논리 payload, wire byte, frequency와 critical-path 시간을 기록한다. Muon의 gather/orthogonalize/scatter, Shampoo factor reduce와 root broadcast, SOAP basis broadcast, Sophia estimator aggregation을 구분한다. cadence 평균으로 나눈 byte와 root step의 burst를 모두 보여 준다. 평균 bandwidth가 좋아도 p99 owner tail이 step 시간을 지배할 수 있다.

장애 주입은 gradient reduce 전, owner state 계산 중, state write 뒤 parameter apply 전, 일부 parameter apply 뒤, all-gather 중과 checkpoint publish 중에 process를 죽인다. 각 지점에서 허용 terminal은 이전 complete commit 또는 새 complete commit뿐이다. group A는 새 step이고 group B는 이전 step인 상태를 정상 resume로 인정하지 않는다. in-memory parameter가 부분 변경되었으면 process를 계속 쓰지 않고 마지막 durable cut에서 다시 시작한다.

packet delay와 collective participant 누락은 단순 rank kill과 다르다. timeout rank마다 최초 원인이 다르게 보일 수 있으므로 coordinator는 first failure event와 peer cancellation을 연결한다. 재시도는 같은 collective sequence와 stale communicator를 재사용하지 않는다. split-brain owner 두 개가 같은 block generation을 쓰는 상황도 lease 또는 generation assertion으로 차단한다.

checkpoint corruption에는 momentum shard 교환, block order permutation, root만 한 cadence 오래됨, SOAP basis와 moment version 불일치, Sophia curvature counter rollback, Lion momentum dtype 변경을 넣는다. shape가 같아 loader가 받아들이기 쉬운 오류를 우선한다. stable logical ID, generation, checksum과 dependency recipe가 이를 거부해야 한다.

복구 certificate에는 마지막 complete OptimizerStepID, 각 state generation, source recipe, topology mapping과 다음 gradient digest를 기록한다. 재개 후 다음 세 parameter delta, root/basis cadence event와 collective trace를 control에 비교한다. topology가 달라 bitwise equality를 요구할 수 없으면 허용 metric과 tolerance를 사전에 선언하고 statistical resume라고 표시한다.

**최종 독자 경로와 종료 판정**

독자는 먼저 model parameter inventory를 만든다. 각 행에는 logical role, shape, tied 관계, sparsity, shard axis, decay 여부와 현재 AdamW group을 넣는다. 그 뒤 matrix optimizer 후보를 semantic allowlist로 표시한다. 이 단계에서 embedding과 router를 자동 제외하거나 포함하는 것이 아니라 역할별 위험과 별 ablation을 적는다.

다음으로 후보 하나의 update 식을 state transition으로 쓴다. 입력 gradient부터 momentum, matrix transform, scale, clipping, decay와 parameter commit까지 순서를 정한다. 각 화살표를 고정 함수와 독립 fixture에 연결한다. 설명할 수 없는 wrapper나 external backend가 있으면 미검증 칸으로 남긴다.

세 번째로 target shape 분포를 반영한 수치 suite를 실행한다. zero, rank-one, ill-conditioned, tall, wide, rotated와 scale-transformed matrix를 포함한다. FP64 oracle, 실제 training precision과 optimized backend 결과를 나란히 둔다. residual만 맞고 update scale이 틀리거나, delta는 맞지만 state가 다음 step에 갈라지는 경우를 모두 실패로 잡는다.

네 번째로 state와 system budget을 계산한다. persistent와 peak byte, optimizer FLOP, collective byte와 cadence tail, checkpoint 크기와 복구 시간을 잰다. 같은 hardware에서 AdamW control을 측정한다. 분석식과 profiler 차이가 큰 buffer는 이름과 owner를 찾기 전까지 이득으로 계산하지 않는다.

다섯 번째로 gradient replay와 작은 model pilot을 수행한다. 공통 parameter group, 후보 matrix group과 제외 group을 분리해 delta를 비교한다. optimizer별 공정한 lr search와 paired seed를 사용하고 fixed-token, fixed-compute와 fixed-wall-clock 추정값을 구분한다. 장기 품질은 이 단계 뒤에야 논한다.

마지막으로 장애와 resume를 통과시킨다. owner kill, overflow, checkpoint cadence 경계, topology 변경과 corrupted state를 주입한다. 다음 세 update certificate가 control과 맞고 partial commit이 없으며 source와 recipe가 immutable ID로 연결되어야 한다.

종료 판정은 “loss가 내려갔다”가 아니다. 적용 parameter 집합이 완전하고 배타적이며, 수식과 함수가 연결되고, 저정밀 오차가 예산 안이며, state·통신 비용이 측정되고, AdamW 비교가 공정하며, 장애 뒤 의미가 보존되어야 한다. 하나라도 비면 candidate는 흥미로운 실험 상태이지 production 승인 상태가 아니다.

**실험실에서 바로 실행하는 단계별 protocol**

첫날에는 모델을 학습하지 않는다. parameter inventory를 내보내고 tied storage, logical role, shape, stride, shard와 optimizer group을 대조한다. 합집합이 전체 trainable parameter와 같고 교집합이 비어 있는지 계산한다. 후보 행렬의 aspect ratio, numel과 예상 state byte를 정렬하면 작은 행렬 launch와 큰 Shampoo factor 위험을 미리 볼 수 있다. 이 산출물이 이후 모든 실험의 모집단이다.

둘째 날에는 gradient tape를 만든다. 실제 model의 초기, warmup 종료와 decay 진입 checkpoint에서 동일한 몇 batch를 replay해 role별 gradient를 저장한다. 개인정보나 원문을 복제하지 않고 digest와 통계만 장기 보존할 수 있다. gradient의 Frobenius norm, stable rank, 상위 singular value 비중, row와 column covariance를 계산한다. 후보 optimizer가 이용한다는 구조가 실제로 존재하는지 확인한다.

셋째 날에는 작은 oracle을 돌린다. 저장 gradient의 축소 행렬과 합성 pathological matrix를 FP64 SVD, eigendecomposition과 손 계산 update에 넣는다. production dtype 결과와 반복별 intermediate를 비교한다. normalization 제거, transpose 반전, coefficient 변경, stale root와 잘못된 basis를 주입해 assertion이 실패하는지 본다. 좋은 입력 통과보다 나쁜 구현 거부가 test 강도를 보여 준다.

넷째 날에는 optimizer object를 시험한다. gradient 없음, 정확한 zero, sparse, frozen parameter와 overflow에서 decay, state와 counter가 어떻게 움직이는지 기록한다. 첫 state 생성 전후, cadence 경계와 save/load 뒤 delta를 비교한다. parameter group 순서를 뒤섞고 stable logical ID가 같은 state를 찾는지 본다. Python 객체 순서에 의존한 checkpoint는 여기서 드러난다.

다섯째 날에는 분산 synthetic를 실행한다. full matrix reference와 row, column, flattened shard를 같은 gradient로 구동한다. collective payload와 owner generation을 기록하고 rank kill, delay와 stale shard를 넣는다. 성공은 job이 다시 뜨는 것이 아니라 partial parameter가 노출되지 않고 마지막 complete state에서 다음 delta가 맞는 것이다.

여섯째 날에는 target shape benchmark를 한다. 실제 shape histogram에서 크기와 aspect ratio를 층화 표본으로 뽑고 eager, compiled와 fused backend를 비교한다. warmup compile time, steady kernel, temporary peak, collective tail과 fallback 비율을 낸다. 평균 하나 대신 작은 행렬과 큰 행렬의 병목을 분리한다.

일곱째 날에는 AdamW paired pilot을 시작한다. 같은 initialization과 batch tape에서 all-AdamW, semantic hybrid와 all-2D 위험군을 비교한다. family별 동일 탐색 예산으로 lr를 찾고 fixed token에서 validation과 system cost를 본다. 공통 group delta가 어긋나면 pilot을 중단한다. 품질 차이를 optimizer 효과로 해석할 전제가 깨졌기 때문이다.

마지막 날에는 checkpoint를 임의 선택해 역방향과 정방향으로 감사한다. 어떤 source 함수와 recipe가 마지막 update를 만들었는지 찾고, state generation과 gradient digest를 복원한다. 이어 다음 세 update, cadence와 collective를 예측한다. 이 양방향 추적이 성공하고 미검증 범위가 명시되면 제한된 canary로 승격한다.

**Muon 갱신을 사건 단위로 재현한다**

Muon을 이식할 때 첫 옵션 묶음은 momentum 계수, Nesterov 사용 여부, Newton–Schulz 반복 횟수와 계수, epsilon, 행렬별 scale, weight decay 순서다. 이 값들은 독립 손잡이가 아니다. momentum 뒤에 orthogonalization을 적용하는 구현과 raw gradient를 직교화한 뒤 momentum을 누적하는 구현은 같은 이름을 써도 다른 상태 전이를 만든다. 따라서 설정 표에는 각 값뿐 아니라 이 장의 update pipeline과 아래의 고정 함수 좌표를 한 recipe digest로 묶는다.

상태에는 parameter별 momentum, step, 선택된 transpose 방향, fallback 이유와 분산 owner generation이 포함된다. 선택 방향을 shape에서 매번 다시 계산할 수 있어도 checkpoint에는 당시 결정을 기록한다. padding이나 tensor parallel 재배치가 shape 표현을 바꾸면 재개 직후 다른 Gram 축을 택할 수 있기 때문이다. 옵션 변경은 기존 momentum을 그대로 해석할 수 있는지 판정하고, 불가능하면 명시적인 migration generation을 만든다. load가 성공했다는 사실은 상태 의미가 유지되었다는 증거가 아니다.

효과는 최종 loss보다 먼저 delta의 norm, raw momentum과 delta의 cosine, singular value 평탄화, parameter RMS 대비 update RMS로 본다. scale 옵션을 바꾸면 방향 residual은 그대로인데 update RMS만 변할 수 있다. 이 경우 orthogonalization 품질 문제로 오진하지 않는다. 반대로 반복 횟수를 낮춰 residual이 나빠졌는데 outer learning rate가 우연히 보상하면 한 step loss는 같을 수 있다. 관측 항목을 단계별로 분리해야 원인을 닫을 수 있다.

고정 source는 commit, `optim.py` 같은 파일 경로, 호출 함수, 계수 literal 주변 blob digest와 wrapper dispatch까지 포함한다. 고정 test는 2×2 diagonal, rank-one, zero, 매우 큰 norm, tall, wide, transposed non-contiguous 행렬을 FP64 SVD polar oracle과 비교한다. 그 다음 같은 gradient tape로 eager와 compiled 경로의 intermediate를 반복별 비교한다. transpose branch를 뒤집거나 scale factor를 제거한 mutant가 반드시 실패해야 한다.

실패 판정은 non-finite만이 아니다. 입력 norm 범위 안에서 residual 예산 초과, delta cosine 하한 위반, 기대 scale 대비 RMS 오차, fallback 비율 초과, 저장과 재개 뒤 세 step 중 최초 불일치도 실패다. 분산 시험에서는 collective sequence와 재조립 delta를 full-matrix oracle에 대조한다. block-local 근사를 썼다면 full equality 대신 고정된 block oracle을 사용하고 근사임을 결과에 남긴다.

장애는 momentum write 전후, 직교화 workspace 생성 뒤, parameter commit 도중에 주입한다. 재시작 결과는 이전 complete step 또는 새 complete step이어야 하며 혼합 세대는 금지한다. 이 규칙은 7장의 checkpoint commit과 11장의 분산 collective ordering을 따른다. source 좌표, 입력 digest, 옵션 digest, 상태 generation과 출력 delta digest가 한 사건으로 이어지지 않으면 Muon 결과를 재현 가능하다고 부르지 않는다.

**Shampoo root cadence를 검증 가능한 상태 기계로 만든다**

Shampoo의 핵심 옵션은 block 크기, factor accumulation dtype, damping, inverse-root exponent, root 갱신 주기, grafting 방식과 precondition 시작 step이다. block 크기는 메모리 옵션인 동시에 알고리즘 옵션이다. 같은 tensor를 다른 block으로 자르면 factor가 포착하는 상관 구조가 달라진다. root 주기를 늘리면 계산량은 줄지만 stale preconditioner가 길게 유지된다. 옵션 표에는 예상 persistent byte, root-step peak와 최대 root age를 함께 적는다.

각 block 상태에는 row와 column factor, cached root, factor generation, root generation, 마지막 성공 step과 decomposition status가 포함된다. gradient를 factor에 반영한 뒤 같은 step에 새 root를 쓰는지, 이전 root로 parameter를 갱신한 뒤 다음 step부터 쓰는지를 명시한다. off-by-one은 평균 loss에서 감춰지므로 cadence 경계 전후의 상태 전이를 직접 고정한다. gradient가 없거나 overflow로 step을 건너뛸 때 factor와 cadence counter가 움직이는지도 별 옵션으로 선언한다.

관측 효과는 factor trace, symmetry error, 최소 eigenvalue, root residual, root age와 preconditioned update amplification으로 나눈다. damping 증가가 residual을 개선하면서 약한 방향의 amplification을 줄이는 현상은 기대 가능한 교환이다. 반면 factor는 새 세대인데 root generation이 뒤처지고 age 표기가 0이면 상태 원장이 거짓이다. root step의 p50과 p99 시간, 임시 workspace peak도 일반 step과 분리한다.

고정 source에는 factor update 함수, block partitioner, inverse-root backend, fallback branch와 distributed reducer가 모두 필요하다. 고정 test는 알려진 SPD 대각 행렬, 중복 eigenvalue, nearly singular factor와 비대칭 오염 factor를 독립 eigendecomposition oracle에 넣는다. damping 적용 전후와 exponent를 손 계산하고, cached root를 한 주기 오래되게 만든 fixture도 둔다. backend가 실패했을 때 identity, diagonal, 이전 root 중 정확히 어떤 결과가 선택되는지 assertion한다.

실패 판정은 root residual 상한, symmetry 오차, amplification 상한, generation 일치, fallback 이유 완전성, memory budget과 cadence tail을 함께 사용한다. decomposition 실패 후 조용히 이전 root를 사용하고 event를 남기지 않으면 수치가 finite여도 실패다. block order를 바꾼 checkpoint가 shape만 맞아 load되는 경우도 stable logical block ID가 거부해야 한다.

분산 장애 시험은 factor reduce 중 rank kill, root owner 계산 중 kill, broadcast 일부 완료 뒤 kill을 포함한다. owner가 바뀌면 lease generation도 바뀌어야 하고 stale root가 새 factor에 붙지 않아야 한다. 복구 뒤 다음 cadence 사건과 세 번의 delta를 uninterrupted control에 비교한다. 9장의 메모리 원장과 10장의 profiler 절차를 사용하면 계산 절감이 통신 burst나 serialization peak로 이동했는지도 확인할 수 있다.

## 12.14 분산 state·비가환성·API 경계를 운영 증거로 만든다

basis checkpoint, collective byte, polar geometry, decay 합성, 관측성과 grad=None·sparse 분기를 한 변경 승인 패키지로 묶는다.

SOAP 옵션은 basis 갱신 주기, eigendecomposition dtype, 정렬 규칙, 모멘트 dtype, epsilon, precondition 방향과 basis 초기화 방식이다. 동일 고윳값이 있는 공간에서는 eigenvector의 부호와 순서가 유일하지 않다. 그러므로 원소별 basis equality를 옵션 계약으로 삼지 않는다. 계약은 subspace projector, 직교성, transformed moment와 원좌표 delta의 동치다.

상태에는 원좌표 또는 회전좌표 momentum, second moment, basis, eigenvalue, basis generation과 moment generation이 포함된다. 새 basis를 계산한 step에 기존 moment를 회전할지 초기화할지, 다음 step부터 적용할지 정한다. 이 선택은 성능 최적화가 아니라 학습 궤적을 바꾸는 상태 전이다. checkpoint loader는 basis와 moment generation이 다르면 재직교화만 하고 진행하지 말고, 정의된 migration이나 명시적 실패를 선택해야 한다.

효과 관측은 projector 거리, orthogonality error, round-trip moment error, 원좌표 delta cosine과 basis-update step의 latency spike를 포함한다. basis 부호가 전부 바뀌어도 projector와 delta가 같으면 성공이다. 반대로 basis 원소가 가까워도 moment를 옛 basis 좌표로 해석해 delta가 달라지면 실패다. 낮은 precision 저장은 checkpoint 크기를 줄이지만 round-trip 오차와 재개 직후 update 변화를 함께 보고한다.

고정 source는 covariance accumulation, eigensolver 호출, eigenvector 정렬, moment rotation, serialization과 load migration 함수까지 연결한다. 고정 test는 distinct eigenvalue, repeated eigenvalue, 거의 겹친 eigenvalue, rank-deficient covariance와 의도적 부호 반전을 포함한다. 랜덤 orthogonal 회전을 적용한 동치 fixture에서 projector와 원좌표 delta가 유지되는지 확인한다. eigenvector 원소 equality를 쓰는 test는 제거 대상이다.

실패 판정은 orthogonality와 round-trip 예산, generation 불일치, basis cadence drift, 재개 뒤 delta 차이, 임시 메모리 peak와 fallback 누락이다. eigensolver가 수렴하지 않아 이전 basis를 쓰면 age와 이유를 남긴다. 이전 basis를 쓰면서 새 eigenvalue만 저장하는 혼합 상태는 금지한다. corruption fixture는 basis block permutation, moment 축 교환과 dtype metadata 위조를 넣는다.

장애 주입은 covariance commit, basis 계산, moment rotation, parameter apply 사이 모든 경계에서 수행한다. partial rotation된 moment를 정상 state로 publish하지 않는다. 6장의 재현성 계약처럼 bitwise 조건과 metric 조건을 구분하고, topology 변화가 있는 경우 projector·delta tolerance를 사전에 고정한다. 이 증거가 없으면 재개 성공은 단지 loader가 예외를 내지 않았다는 뜻에 그친다.

### Sophia와 Lion의 작은 분모·부호 경계를 시험한다

Sophia 옵션은 curvature 추정 주기, estimator 종류, denominator floor, rho clipping, curvature dtype와 weight decay 순서다. Lion 옵션은 두 momentum 계수, sign 적용 위치, zero 처리, state dtype와 decay 순서다. 둘 다 elementwise라 행렬 root가 없지만 수치 경계가 단순하지 않다. 옵션→상태→효과를 한 줄로 연결하면 curvature floor는 denominator와 clipped delta를 바꾸고, sign 직전 cast는 방향과 다음 momentum을 바꾼다.

Sophia 상태에는 momentum, curvature estimate, estimator counter, 마지막 curvature batch lineage와 scaler generation이 필요하다. curvature를 갱신하지 않은 step에서도 counter가 실제 optimizer commit과 일치해야 한다. Lion 상태에는 momentum과 step 외에 계산 dtype과 zero-sign 정책을 recipe에 넣는다. checkpoint에서 FP32 momentum을 저정밀로 내릴 때는 silent cast가 아니라 새 state schema generation으로 기록한다.

효과는 Sophia의 curvature 분위수, floor 적용 비율, clipping 비율, update amplification과 Lion의 sign flip 비율, zero 주변 margin, FP32 control과의 delta cosine으로 본다. 평균 curvature나 평균 cosine만 보면 드문 큰 방향 오류를 놓친다. parameter role과 gradient magnitude bucket별 꼬리 분위수를 보고한다. state memory 절감은 이 오류 표와 같은 행에 둔다.

고정 source는 curvature estimator를 호출하는 training 경로, loss normalization, scaler unscale, optimizer update와 scheduler 적용 순서를 포함한다. 고정 test는 음수·zero·subnormal·매우 큰 curvature, clip 경계 바로 안팎, sign 합이 ±한 ULP인 gradient를 사용한다. FP64 손 계산 oracle과 FP32 control을 두고 floor와 clip 순서를 바꾼 mutant, sign 전후 cast를 바꾼 mutant가 실패하는지 확인한다.

실패 판정은 non-finite, amplification 상한, floor 또는 clip 비율의 허용 범위, sign disagreement 꼬리, counter와 batch lineage 불일치, resume delta 차이를 포함한다. overflow skip에서 curvature만 전진하거나 decay만 적용되면 원자성 실패다. gradient가 없는 parameter에 decay를 적용하는 정책은 허용할 수 있지만 명시적 옵션과 fixture가 있어야 한다.

분산에서는 curvature estimator denominator가 rank마다 다른지 검산하고 합산 순서와 dtype을 고정한다. 일부 rank가 빈 valid label을 가진 batch도 포함한다. 이 검사는 13장의 token denominator와 직접 연결된다. scheduler clock이 commit되지 않은 batch를 세지 않는데 curvature counter만 센다면 두 상태 기계가 갈라진다. 공통 OptimizerStepID로 둘을 결합해 failure event에서 함께 검증한다.

### parameter 분류와 혼합 optimizer 경계를 봉인한다

혼합 optimizer의 옵션은 이름 정규식이 아니라 semantic role, tensor rank, 최소 차원, sparsity, tied storage, shard layout과 fallback family다. rank 2라는 이유만으로 embedding, output head와 router를 같은 행렬 optimizer에 넣지 않는다. 옵션이 parameter 집합을 선택하고, 그 선택이 state allocation과 update rule을 바꾸며, 결과로 memory·통신·품질이 변한다. 이 연결을 inventory 한 행에서 읽을 수 있어야 한다.

상태 원장에는 stable logical parameter ID, 모든 alias, 선택 family, group recipe digest, state owner와 generation을 기록한다. tied weight가 두 이름으로 발견되어 서로 다른 optimizer에 들어가면 교집합 검사가 실패해야 한다. flattening wrapper가 여러 logical tensor를 한 physical tensor로 합쳐도 원래 역할 경계를 보존한다. checkpoint 매핑은 iteration order나 object identity가 아니라 stable ID와 shape recipe를 사용한다.

효과 관측은 family별 parameter 수와 byte, update RMS, decay 적용 수, fallback 수, 작은 matrix kernel launch와 collective payload다. 전체 평균 update ratio는 큰 embedding이 지배할 수 있으므로 role별 분포를 낸다. optimizer를 바꾸지 않은 공통 group은 paired run에서 delta가 같아야 한다. 다르면 data, scheduler, scaler 또는 parameter 분류가 함께 변한 것이다.

고정 source는 model parameter declaration, tying 함수, sharding transformation, group builder와 optimizer constructor를 하나의 call graph로 둔다. 고정 test는 alias, frozen parameter, late unfreeze, sparse gradient, rank 변화 없는 reshape와 flatten/unflatten을 포함한다. inventory 합집합이 전체 trainable storage와 같고 family 교집합이 비어 있으며 제외 이유가 모든 행에 존재하는지 assertion한다.

실패 판정은 누락·중복 storage, recipe 없는 fallback, load 뒤 family 이동, decay 중복, 공통 group delta 차이와 예상 byte 오차다. 새 layer 이름이 정규식에 우연히 걸리는 경우를 막기 위해 unknown role은 자동 포함하지 않고 배포 전 실패로 처리한다. canary에서 fallback 비율이 기준을 넘으면 속도나 품질과 관계없이 분류 계약을 다시 연다.

이 경계는 5장의 model graph, 7장의 checkpoint ID, 9장의 memory accounting과 연결된다. model revision마다 inventory snapshot과 diff를 생성하고, 추가·삭제·retie된 storage를 검토한다. 한 번 성공한 allowlist를 다음 model에 복사하지 않는다. parameter 집합의 의미가 닫혀야 optimizer 비교도 닫힌다.

### backend·정밀도·분산의 삼중 교차 검증

backend 옵션은 eager, compiled, foreach, fused와 vendor kernel이며 정밀도 옵션은 parameter, gradient, state, decomposition과 accumulation dtype이다. 분산 옵션은 replicated, owner, sharded와 block-local이다. 세 축을 따로 시험하면 조합에서만 발생하는 오류를 놓친다. 최소 조합 행렬은 production 후보 전부와 FP64 단일 장치 oracle, FP32 eager control을 포함한다.

상태에는 backend dispatch 결과, compile graph digest, fallback reason, autocast 상태, collective group과 algorithm generation을 기록한다. 사용자가 fused를 요청했어도 unsupported stride 때문에 eager로 갔다면 요청값과 적용값을 모두 남긴다. compiler cache가 다른 shape graph를 재사용하면 dispatch generation이 달라져야 한다. dtype 이름만 같고 accumulator가 바뀐 vendor revision도 source recipe가 구분한다.

효과는 최종 delta뿐 아니라 momentum, Gram 또는 covariance, 반복 intermediate, scale factor와 collective 재조립 결과의 최초 divergence로 찾는다. 성능은 compile warmup, steady latency, root/basis cadence tail, temporary peak와 wire byte를 나눈다. 빠른 경로의 fallback이 많으면 평균 latency가 우연히 좋아도 해당 backend의 주장으로 계산하지 않는다.

고정 source는 framework commit, compiler flags, kernel artifact hash, device capability, collective library와 dependency lock을 포함한다. 고정 test는 contiguous와 transposed stride, odd dimension, empty local shard, padding, extreme aspect ratio와 cadence 경계를 조합한다. 동일 입력 tape를 모든 경로에 replay하고 tolerance는 residual, cosine, RMS와 next-loss change의 공동 조건으로 둔다.

실패 판정은 tolerance 초과, 미기록 fallback, graph 재컴파일 폭증, collective 순서 차이, peak budget 초과와 hang이다. hang 시험에는 timeout만 두지 않고 마지막 collective sequence, 참가 rank와 first failure를 보존한다. 일부 조합이 지원되지 않으면 silent skip이 아니라 명시적 unsupported 결과가 필요하다.

새 accelerator나 compiler upgrade는 source pin이 바뀌므로 조합 행렬을 다시 연다. 10장의 성능 측정과 11장의 분산 진단을 이용해 품질과 시스템 증거를 같은 run ID에 묶는다. kernel microbenchmark만 통과하거나 작은 model loss만 맞는 것으로 production 조합을 승인하지 않는다.

**최종 failure dossier와 제한적 승인**

최종 승인의 옵션 문서에는 적용 parameter 집합, update 순서, dtype, fallback, cadence, sharding, overflow와 checkpoint 정책을 모두 기록한다. 각 옵션은 생성하거나 읽는 상태 key와 관측 효과를 가리킨다. 적용되지 않은 요청값이나 기본값은 제거하지 말고 왜 무효였는지 남긴다. 그래야 같은 명령줄이 다른 backend에서 다른 결과를 내는 것을 탐지한다.

상태 certificate는 OptimizerStepID, parameter inventory digest, source recipe, gradient tape digest, state generation, collective trace와 delta digest를 연결한다. 임의 checkpoint에서 이전 사건을 역추적하고 다음 세 사건을 예측할 수 있어야 한다. root나 basis처럼 비동기 cadence를 가진 상태는 dependency generation도 포함한다. 단순 state dict checksum은 의미적 관계를 증명하지 못한다.

효과 보고서는 AdamW control과 동일 data, token clock, scheduler, scaler와 evaluation budget을 사용한다. fixed-token, fixed-compute, fixed-wall-clock 결과를 섞지 않는다. validation뿐 아니라 persistent·peak memory, optimizer FLOP, wire byte, cadence p99, checkpoint 크기와 recovery time을 제시한다. family별 tuning 예산도 같게 두거나 차이를 공개한다.

고정 시험 묶음은 순수 algebra oracle, optimizer lifecycle, distributed reconstruction, backend 교차, corruption과 kill injection, paired pilot이다. 각 test ID는 source 함수와 failure invariant를 가리킨다. 테스트가 성공 경로만 실행하면 충분하지 않다. coefficient, 순서, dtype, generation, block mapping을 바꾼 mutant를 거부하는지 확인해야 한다.

failure dossier에는 최초 위반 invariant, 마지막 complete step, 영향 parameter와 rank, 적용 옵션, 관련 source blob, state 세대와 rollback 지점을 넣는다. loss spike처럼 늦은 증상만 기록하지 않는다. 자동 복구가 가능해도 partial commit, unknown fallback, state dependency 불일치와 parameter 중복은 즉시 중단 조건이다.

제한적 승인은 검증한 model revision, shape 분포, hardware, topology, precision과 source commit에만 유효하다. 범위 밖 변화는 새 증거를 요구한다. 6장의 재현성, 7장의 복구, 9장의 자원 원장, 13장의 scheduler clock을 모두 같은 사건 ID로 결합했을 때만 optimizer 효과를 독립 원인으로 해석할 수 있다. 빈 증거 칸이 있으면 승인이 아니라 명시된 실험 상태로 남긴다.

**한 step의 인과 사슬을 끝까지 검산한다**

실제 장애 조사는 임의의 parameter 하나와 OptimizerStepID 하나를 고르는 데서 시작한다. 적용 옵션은 group 선택 규칙, momentum 계수, 변환 반복, scale, decay, dtype, backend와 sharding이다. 요청 옵션과 실제 dispatch 옵션을 나란히 읽는다. 이 선택이 parameter의 이전 momentum, factor 또는 basis, cadence counter와 owner generation을 어떤 순서로 읽었는지 event trace에서 복원한다.

입력 gradient는 data batch lineage, loss denominator, scaler와 clipping 결과에 연결한다. optimizer 내부에서 처음 관측한 tensor digest와 backward 직후 digest가 다르면 unscale, reduction 또는 clipping 경계를 추적한다. zero와 `None`은 구분한다. zero gradient는 상태를 갱신할 수 있지만 `None`은 정책에 따라 state와 decay를 건너뛸 수 있다. 이 차이는 고정 fixture와 source branch로 증명한다.

상태 전이는 읽기 집합과 쓰기 집합으로 나눈다. momentum을 먼저 commit하고 matrix transform이 실패한 경우 rollback이 둘 모두를 되돌리는지 확인한다. root 또는 basis가 별 cadence에서 생성되면 사용한 dependency generation을 남긴다. parameter만 새 값이고 state는 이전 값이거나 그 반대인 혼합 commit은 finite loss와 무관하게 실패다. 7장의 원자적 checkpoint 규칙을 optimizer step 내부에도 적용한다.

효과 검산은 raw gradient, transformed direction, decay term과 최종 delta를 분리한다. 각 항의 norm, cosine, dtype과 digest를 기록하면 learning rate 오류와 transform 오류를 구별할 수 있다. clipping 전후 순서를 바꾼 mutant, decay를 direction에 섞은 mutant, transpose branch를 뒤집은 mutant가 서로 다른 assertion에서 실패해야 한다. 하나의 최종 delta assertion만으로는 최초 원인을 특정하기 어렵다.

고정 source 묶음에는 group builder, optimizer wrapper, 순수 transform, backend kernel, serializer와 collective adapter의 immutable 좌표를 기록한다. 고정 test는 같은 gradient tape를 FP64 oracle, FP32 eager, production backend와 재개 경로에 투입한다. source upgrade 시 blob 하나라도 바뀌면 영향 test를 다시 실행한다. 이름이 같은 외부 dependency의 이동을 lockfile과 artifact hash로 막는다.

실패 주입은 unscale 뒤, momentum write 뒤, collective 전후, transform fallback, decay 적용 뒤와 parameter publish 직전에 둔다. 허용 결과는 이전 complete 사건 또는 새 complete 사건뿐이다. timeout이면 마지막 collective와 first failure rank를 남기고 communicator를 새 generation으로 만든다. 같은 sequence를 맹목적으로 재시도해 이중 적용하지 않는다.

마지막 검산은 checkpoint에서 다음 세 step을 예측하는 것이다. 적용 family, cadence event, fallback, collective payload와 delta tolerance가 control과 맞아야 한다. bitwise equality가 불가능한 topology라면 residual, cosine과 RMS 예산을 사전에 정한다. 사후에 tolerance를 넓히지 않는다. 이 절차가 닫히면 옵션→상태→효과와 source→test→failure의 두 사슬이 같은 사건에서 만난다.

**canary에서 승격을 멈추는 조기 신호**

canary 옵션은 traffic 또는 training-token 비율, 관측 기간, paired control, 자동 중단 임계값과 rollback checkpoint다. 상태에는 canary generation, 누적 token, optimizer별 parameter inventory digest, fallback count와 마지막 complete OptimizerStepID가 포함된다. 비율을 늘리는 결정 자체도 사건으로 남겨 어떤 증거가 승격을 촉발했는지 복원한다.

효과는 validation 하나가 아니라 role별 update RMS와 cosine, non-finite, root·basis age, fallback, memory peak, collective p99와 checkpoint 시간을 함께 본다. 평균이 안정적이어도 특정 aspect-ratio bucket의 amplification이나 owner rank tail이 커지면 확대를 멈춘다. AdamW control과 공통 group delta가 달라지는 경우는 optimizer 우열을 해석하기 전에 실험 격리를 실패로 판정한다.

고정 source는 배포 recipe, group builder, backend artifact, monitor query와 중단 controller를 포함한다. 고정 test는 정상 event replay와 함께 amplification 초과, stale generation, unknown fallback, rank hang, partial checkpoint를 주입해 controller가 정확한 rollback cut을 선택하는지 확인한다. 경보만 울리고 training을 계속하는 test는 중단 계약을 충족하지 않는다.

failure 후에는 마지막 complete state에서 source와 gradient digest를 고정해 세 step을 재현한다. 원인이 수치, 분류, backend, collective 또는 checkpoint인지 최초 divergence로 분류한다. 임계값을 사후 완화해 같은 run을 성공으로 바꾸지 않는다. 새 기준은 새 canary generation과 paired control을 요구한다. 이 절차가 앞서 정한 지원 한계와 위험 예산을 운영 중에도 보존한다.

승격 표본에는 정상 step만 아니라 fallback과 cadence 경계가 반드시 들어간다. source 함수별 branch coverage, 고정 fixture ID, 주입한 failure와 기대 terminal을 한 행에 놓는다. 관측되지 않은 branch는 성공으로 채우지 않고 미검증으로 남긴다. canary 종료 checkpoint에서 state generation, parameter inventory와 다음 gradient digest를 보존해 rollback 뒤 같은 세 delta를 다시 계산한다. 수치 tolerance, memory 상한과 collective timeout은 실행 전에 고정한다. 재실행 뒤 값을 보고 기준을 바꾸면 독립 검증이 아니다. 이 표가 완전해야 확대 결정이 loss 곡선이 아니라 검산 가능한 상태 전이에 근거한다.

마지막으로 source upgrade 후보를 기존 certificate에 대입해 영향 branch를 계산한다. 함수 본문, default, state schema, kernel과 collective 중 하나라도 바뀌면 관련 fixture와 failure injection을 다시 연다. 영향 분석이 비어 있으면 이전 canary 승인을 재사용하지 않는다. 새 artifact는 새 generation으로 제한 배포한다.

**Muon의 momentum과 Newton–Schulz를 한 update 식으로 닫는다**

matrix parameter `W∈R^{m×n}`의 gradient `G`에 momentum `M_t=βM_{t-1}+(1-β)G_t` 또는 실제 구현의 convention을 적용한 뒤, matrix transform이 update direction을 만든다. 구현마다 Nesterov form과 coefficient placement가 다를 수 있으므로 이름으로 식을 정하지 않는다. fixed source의 functional step, momentum update와 orthogonalization call을 잇는다.

Newton–Schulz iteration은 적절히 scale된 matrix에서 inverse square root/polar factor에 가까운 변환을 반복한다. coefficient polynomial, iteration count, initial normalization과 transpose handling이 state/effect를 바꾼다. tall/wide matrix에서 어느 orientation을 처리하고 결과를 되돌리는지 shape fixture로 확인한다.

**수치 oracle**

작은 FP64 matrix에서 SVD polar factor와 Newton–Schulz output의 spectral/Frobenius residual을 비교한다. rank-deficient, zero, ill-conditioned, repeated singular values와 extreme scale을 넣는다. zero norm에서 division을 막고 NaN direction을 parameter에 쓰지 않아야 한다.

iteration별 `X^TX` 또는 `XX^T`의 identity residual, norm과 finite를 기록한다. coefficient/iteration을 바꾼 option은 compute와 approximation error를 바꾼다. 결과를 본 뒤 tolerance를 넓히지 않는다.

**spectral·dual norm과 scale invariance 주장을 반례로 제한한다**

orthogonalized direction은 singular values를 평탄화하는 기하를 가질 수 있지만 실제 update에는 momentum, scaling, learning rate와 weight decay가 합성된다. “scale invariant”는 gradient를 양수 scalar로 곱했을 때 direction 또는 normalized update가 어느 범위에서 보존되는지 정확히 정의한다.

spectral norm, Frobenius norm과 parameter/update ratio를 함께 측정한다. rectangular matrix의 polar factor Frobenius norm은 dimensions에 의존한다. implementation의 dimension/shape scaling이 learning-rate semantics를 바꾼다. batch size/world size gradient scale과도 연결한다.

**Scale fixture**

같은 gradient에 `10^-3,1,10^3`을 곱하고 momentum 초기/steady, epsilon/zero와 BF16 paths를 비교한다. direction cosine, spectral norm과 final parameter delta를 본다. decoupled decay는 gradient scale과 무관하지만 total delta의 invariance를 깨뜨릴 수 있다.

**matrix parameter selection을 module·shape·role의 교집합으로 만든다**

`ndim==2`만으로 Muon 대상을 고르면 embedding/head, small projection, tied weight와 expert matrices가 섞인다. selection manifest에는 ParameterID, module role, shape, tie/shard, trainable, optimizer owner와 exclusion reason을 기록한다. exact allow/exclude rules와 counts/bytes를 startup에서 검증한다.

embedding은 sparse/row lookup과 vocabulary semantics, LM head는 tied alias와 vocab logits를 가진다. norm/bias/vector는 matrix transform 대상이 아니다. expert matrices는 EP shard와 zero-token state를 고려한다. convolution을 flatten할지는 별 recipe다. shape만 같다고 같은 geometry를 적용하지 않는다.

**Selection failure**

tied embedding/head가 두 optimizers에 들어가거나 어느 곳에도 없고, adapter matrix가 accidental match, frozen parameter state 생성과 new expert omission을 주입한다. 모든 trainable parameter가 정확히 한 owner에 있고 alias는 하나의 state를 가져야 한다.

AdamW fallback group의 lr/decay/betas와 Muon group의 momentum/scale/decay를 별 manifest로 둔다. scheduler가 groups를 같은 clock으로 전진시키는지 overflow/absent gradient에서 확인한다.

**Shampoo·SOAP·K-FAC와 state geometry를 공통 열로 비교한다**

Shampoo류는 gradient의 left/right covariance statistics와 matrix inverse roots/preconditioner, update cadence를 가진다. SOAP류는 basis/eigen state에서 Adam-like moments를 관리하거나 회전한다. K-FAC류는 layer activations/gradient factors, damping와 inverse cadence를 사용한다. 세 방법의 “second order” 이름을 동일 state로 취급하지 않는다.

공통 열은 statistics source, state shape/byte, decay/update cadence, matrix decomposition, damping/epsilon, grafting/fallback, distributed owner와 checkpoint portability다. Muon은 momentum과 iterative polar-like transform을 같은 표에 넣되 covariance root state가 없음을 `해당 없음`으로 표시한다.

**Paired micro-fixture**

동일 `4×3` gradient sequence를 AdamW, Muon, Shampoo/SOAP의 available reference에 넣고 state/update를 기록한다. equality를 요구하지 않고 각 method-defined recurrence와 invariants를 검증한다. K-FAC는 activation/error batch를 함께 제공한다. update norm을 같은 lr 숫자로 공정하다고 쓰지 않는다.

**bf16/fp32 경계와 AMP overflow clock**

parameter/gradient가 BF16이어도 momentum, covariance/root, Newton–Schulz accumulator와 final cast dtype이 다를 수 있다. dtype ledger에 storage, transform compute, statistics와 update accumulator를 적는다. orthogonalization을 BF16로 직접 반복한 경로와 FP32 oracle을 ill-conditioned fixture에서 비교한다.

AMP overflow면 모든 optimizer groups의 parameter, momentum, roots/basis/cadence와 step clock이 전진하지 않아야 한다. statistics만 갱신하고 parameter를 skip하는 partial policy가 있다면 명시하고 distributed ranks가 합의해야 한다. scheduler clock도 committed update에 맞춘다.

**Low-precision failure**

very small/large gradient, zero matrix, BF16 cast underflow, root cadence 직전 overflow와 resume를 넣는다. next finite update가 uninterrupted reference와 맞는지 본다. FP32 master/statistics checkpoint dtype을 보존한다.

**distributed state와 collective byte를 logical matrix에서 계산한다**

FSDP/ZeRO/TP/EP에서 matrix parameter와 momentum/preconditioner가 shard/replicated되는 축을 기록한다. full matrix가 필요한 transform이면 all-gather byte와 temporary peak, shard-local approximation이면 global method와의 의미 차이를 적는다. group size가 같은 wrong-axis collective를 막는다.

Shampoo/K-FAC statistics reduction, Muon global norm/orthogonalization과 SOAP basis owner를 implementation source에서 확인한다. analytical numel×dtype byte와 trace를 맞춘다. asynchronous collective completion 전에 update가 state를 읽지 않게 event DAG를 검사한다.

**Distributed failure**

one rank stale momentum/root, wrong TP group, uneven shard, expert empty와 collective 도중 rank death를 주입한다. global update reconstruction과 AdamW/single-process method oracle을 비교한다. checkpoint reshard 뒤 first update를 검증한다.

**PyTorch·공식 구현의 함수/옵션을 source card로 고정한다**

PyTorch optimizer/group builder와 사용하는 Muon/Shampoo/SOAP/K-FAC 공식 또는 기준 구현의 exact revision, functional step, state init, backend dispatch, serialization과 distributed wrapper를 기록한다. package 이름이나 논문 pseudo-code로 active branch를 증명하지 않는다.

option card에는 momentum/Nesterov, Newton–Schulz steps/coefficient/backend, shape scaling, weight decay, root/basis cadence, damping, grafting, foreach/fused/compile과 state dtype을 기록한다. config→state keys/buffers→selected function/kernel→effect를 잇는다.

**Source upgrade property**

default/state schema/function이 바뀌면 zero/constant/scale/ill-conditioned gradients, cadence/overflow와 checkpoint fixtures를 다시 연다. unknown fallback은 PASS가 아니다. actual selected branch를 trace한다.

**AdamW paired baseline과 최종 인수**

same ParameterIDs, initialization, data/gradient sequence, committed updates, decay semantics와 scheduler budget을 고정한다. AdamW와 matrix optimizer의 lr/update norm을 sweep 또는 predeclared calibration으로 비교한다. target metric만 아니라 stability, state memory, step time와 recovery를 보고한다.

negative suite는 parameter misclassification, transpose/orientation, zero norm, ill-conditioned iteration, stale cadence, low-precision non-finite, wrong group와 mixed checkpoint를 독립 실행한다. expected first gate와 no partial commit을 확인한다.

최종 dossier에는 selection inventory, method equations/source, dtype/state/clock, per-parameter update geometry, distributed byte/owner, checkpoint/rollback와 paired AdamW report를 담는다. 같은 RunID/UpdateID를 가리킨다.

독립 검토자는 matrix 하나를 gradient→momentum→transform/preconditioner→scale/decay→parameter/optimizer state→checkpoint까지 재생한다. option 하나를 바꿔 changed state와 effect를 예측한다. source, FP64 oracle, runtime trace와 resume가 맞을 때만 matrix optimizer를 AdamW 기준선 위에 승인한다.

**Newton–Schulz backend를 matmul graph와 state lifetime으로 검증한다**

iteration 하나는 matrix products, polynomial coefficients, scaling과 optional transpose로 구성된다. eager PyTorch, compiled/fused 또는 custom backend가 같은 logical sequence를 구현하는지 generated graph와 profiler에서 확인한다. `steps`를 늘리면 kernel count/compute와 convergence residual이 함께 바뀐다.

temporary matrices의 shape/dtype, allocation과 stream lifetime을 기록한다. rectangular matrix를 smaller Gram orientation으로 바꾸는 최적화는 matmul shapes와 final transpose를 바꾼다. source guard가 m/n boundary에서 어느 path를 선택하는지 `m<n`, `m=n`, `m>n` fixtures로 본다.

**Backend failure**

noncontiguous parameter/gradient, odd dimensions, very small matrix, graph capture, compiler cache mismatch와 unsupported device를 넣는다. explicit fallback/error와 selected branch를 검사한다. silent CPU 또는 different iteration fallback은 performance/state report에 나타나야 한다.

CUDA Graph에서는 lazy momentum/temp allocation과 dynamic shape를 warmup에서 materialize한다. graph replay의 step/scalars, overflow branch와 stable addresses를 확인한다. eager/graph output, momentum와 next checkpoint를 paired로 비교한다.

**decay·momentum·orthogonalization의 합성 순서를 손계산한다**

같은 gradient라도 momentum 전에 scaling하는지, momentum 뒤 matrix transform을 하는지, Nesterov를 어느 표현에 적용하는지에 따라 direction이 달라진다. decoupled decay가 transformed delta 전/후 parameter에 적용되는지도 source에서 고정한다. 이름이 Muon이라고 하나의 공통 순서를 가정하지 않는다.

2×2 parameter와 두-step gradient sequence에서 intermediate momentum, normalized/orthogonal direction, shape scale, adaptive/update delta, decay delta와 final parameter를 FP64로 저장한다. first step만 보면 momentum convention 차이를 놓칠 수 있다. zero gradient second step은 momentum/decay 분리를 선명하게 한다.

**Composition property**

decay 0, momentum 0, steps 0/reference branch와 maximize-like sign control을 하나씩 비교한다. implementation에 없는 option을 억지로 일반화하지 않는다. parameter dtype cast 전후 delta도 기록한다. selected formula와 source local variables를 양방향으로 연결한다.

**state memory와 cadence를 time-to-recover까지 확장한다**

Muon momentum은 parameter와 같은 numel state를 가질 수 있고 Shampoo/SOAP/K-FAC는 factor/root/basis와 statistics를 추가한다. logical bytes, alignment, sharding, temporary decomposition과 checkpoint write/read를 분리한다. total parameter 비율만으로 peak를 쓰지 않는다.

root/basis/statistics cadence는 average step time을 줄이지만 cadence boundary latency와 stale preconditioner를 만든다. interval counter가 checkpoint state다. overflow/skip, accumulation와 resume에서 counter가 무엇을 세는지 확인한다. loop iteration과 committed update를 섞지 않는다.

**Recovery rehearsal**

cadence 직전/직후와 decomposition 중 process failure를 주입한다. partial state root를 publish하지 않고 last complete generation에서 같은 next update를 만든다. topology 변경에서 factor owner와 parameter offset을 재배치한다. unsupported reshard는 model-only reset을 명시한다.

steady throughput, checkpoint pause, state reload, root/basis rebuild와 cold compiler cost를 합쳐 useful updates/hour를 비교한다. AdamW보다 빠른/느린 한 step만으로 장기 운영 결론을 내리지 않는다.

**matrix optimizer support matrix의 마지막 봉인**

행은 Muon, Shampoo, SOAP, K-FAC와 AdamW baseline, 열은 eligible parameter roles, update equation, state/cadence, dtype/backend, distributed owner, checkpoint migration, validated shapes/devices와 failure status다. method별 해당 없는 cell을 0으로 채우지 않는다.

검증한 official/reference implementation revision과 local wrapper를 같이 적는다. wrapper가 parameter selection, scaling, decay와 dispatch를 바꾸면 upstream method test만으로 충분하지 않다. full recipe Golden matrix를 실행한다.

**Blind selection test**

reviewer에게 module inventory와 config만 주고 each ParameterID의 optimizer owner, expected state bytes와 first update path를 재구성하게 한다. runtime manifest와 비교한다. embedding/head/tied/expert/adapter 경계가 틀리면 admission을 실패시킨다.

두 번째 reviewer는 source를 보지 않고 state_dict와 trace에서 method/cadence/backend를 추론한 뒤 source card와 맞춘다. unknown buffer, unexpected fallback 또는 stale counter는 새 support cell을 요구한다.

마지막으로 same gradients에서 AdamW와 candidate의 update norm/spectral geometry, state byte, latency와 resume를 재계산한다. 결과가 predeclared budget 안이고 failure injection이 expected gate에서 차단될 때만 canary를 승격한다. 이 봉인이 matrix optimizer 선택을 유행이 아니라 재현 가능한 engineering decision으로 만든다.

**matrix shape scaling을 모델 규모와 layer 역할에서 검증한다**

shape scaling은 `m,n`, aspect ratio 또는 fan-in/out에서 learning-rate multiplier를 만들 수 있다. 실제 구현의 식과 적용 위치를 source에서 고정한다. base learning rate에 곱하는지 transformed direction에 곱하는지에 따라 scheduler/decay semantics가 다르다.

square attention projection, wide/tall MLP와 small adapter matrices에서 multiplier, pre/post update norm과 parameter ratio를 표로 만든다. matrix transpose representation을 바꿨을 때 logical layer update가 의도대로 변환되는지 본다. 저장 layout 우연이 hyperparameter를 바꾸지 않아야 한다.

**Shape failure**

1×N/N×1, extreme aspect, empty/degenerate, transposed tied view와 sharded local shape를 넣는다. scaling은 global logical shape를 써야 하는지 implementation contract를 확인한다. TP local shape를 global로 오인하면 rank/world-size에 따라 effective lr가 달라질 수 있다.

**mixed optimizer scheduler와 checkpoint를 하나의 clock으로 닫는다**

Muon matrix group과 AdamW vector/embedding group은 서로 다른 state/option을 가지지만 같은 training commit에 참여한다. scheduler가 group별 lr multiplier와 base schedule을 어떻게 적용하는지 기록한다. absent gradient가 한 group에만 있을 때 group step/cadence와 global committed clock 정책을 정한다.

AMP overflow 또는 distributed non-finite면 두 optimizers 모두 update를 skip해야 replica/model consistency를 유지할 수 있다. one optimizer만 전진하는 partial policy는 명시적 multi-optimizer transaction 없이는 금지한다. moments, roots/basis, step/cadence와 scheduler를 비교한다.

**Mixed checkpoint failure**

Muon state newest/AdamW state stale, scheduler one step ahead, parameter inventory changed와 group order permutation을 넣는다. root manifest가 component generation과 stable ParameterID mapping을 검증해야 한다. load 뒤 next update를 uninterrupted mixed reference와 비교한다.

optimizer implementation을 바꿀 때 portable logical state와 method-specific state를 구분한다. reset/warm start면 affected groups, lr warmup와 lost trajectory를 선언한다. file deserialize 성공을 compatibility로 쓰지 않는다.

**production option 변경의 최소 paired rehearsal**

candidate는 Newton–Schulz steps/coefficient/backend, momentum, scale, parameter selection, state dtype, decay 또는 cadence 가운데 한 축만 baseline에서 바꾼다. 동일 checkpoint, GoldenBatch와 gradient sequence에서 changed source branch, buffers, update geometry, memory와 latency를 기록한다.

first step, momentum steady step, zero/small/ill-conditioned gradient, overflow, cadence boundary와 checkpoint resume를 실행한다. AdamW fallback groups도 같은 UpdateID에서 비교한다. production-selected matrix 하나의 state를 full precision oracle로 재생한다.

**Decision record**

승인에는 target metric 이전에 selection coverage, finite/numerical budget, no partial commit, checkpoint/rollback와 distributed support가 필요하다. unexpected fallback, unknown state와 unvalidated GPU/shape는 release blocker 또는 explicit unsupported다.

운영 중 update norm/spectral residual, non-finite, iteration residual, fallback/cadence latency와 state generation을 표본 관측한다. threshold breach면 optimizer 전체를 재설정하기 전에 affected ParameterID와 first divergence를 찾는다. last complete parent로 rollback한다.

독립 인수자는 option diff 하나를 보고 state schema, mathematical direction, dispatch/collective와 recovery effect를 예측한다. actual manifest와 trace가 맞고 failure fixtures가 민감하면 child recipe를 봉인한다. 이 최종 rehearsal이 Muon과 다른 matrix optimizers를 AdamW와 공정하고 운영 가능한 기준으로 비교하게 한다.

**마지막 state round trip**

검토자는 checkpoint에서 representative attention/MLP matrix, excluded embedding/head와 vector parameter를 하나씩 고른다. optimizer owner, global shape, state tensors, dtype, step/cadence와 shard offsets를 재구성한다. selection manifest와 실제 state_dict가 일치해야 한다.

matrix 표본에는 고정 gradient를 넣어 momentum, Newton–Schulz 또는 chosen preconditioner, shape scale, decay와 final delta를 계산한다. excluded 표본은 AdamW recurrence를 따른다. mixed groups가 같은 overflow/commit clock을 공유하는지 본다.

이어 checkpoint를 새 process와 supported target topology에서 load한다. lazy/new state, compiled backend와 collectives를 warmup하고 next update를 uninterrupted reference와 비교한다. old compiler cache와 stale rank-local preconditioner는 폐기하거나 generation 검사로 막는다.

마지막 failure는 parameter role 하나를 잘못 분류하고 cadence counter 하나를 rollback하는 것이다. admission/property test가 optimizer step 전에 실패해야 한다. 이를 잡지 못하면 평균 loss나 update norm이 정상이어도 recipe를 승인하지 않는다.

모든 결과는 source revision, ParameterID, UpdateID, RecipeID와 checkpoint root로 연결한다. 이후 framework, backend, world size, dtype 또는 option이 바뀌면 영향받은 state round trip을 다시 실행한다. 이 왕복이 matrix optimizer의 수학·코드·운영 의미를 최종적으로 보존한다.

승인 뒤 production 표본에서도 동일 matrix의 iteration residual, update norm, state generation과 dispatch를 주기적으로 확인한다. canary 범위를 벗어난 shape나 backend가 나타나면 우연한 실행 성공을 support로 인정하지 않는다. 새 FP64 oracle, failure boundary와 resume evidence가 생길 때까지 해당 cell을 미검증으로 유지한다.

독립 reviewer가 같은 RecipeID와 artifact만으로 이 판정을 다시 반복해 동일한 승인 또는 중단 결론을 얻을 수 있어야 최종 운영 봉인이 닫힌다.

**Muon을 하나의 공식이 아니라 합성된 optimizer로 읽는다**

**한 update의 연산 순서를 명시한다.**

Muon이라는 이름 아래에는 gradient 수신, momentum 갱신, 선택적 Nesterov 조합, 행렬 정규화, Newton–Schulz 직교화, shape scale, learning rate, decoupled weight decay와 parameter commit이 있다. 구현마다 이 순서와 계수가 다를 수 있다. 논문 또는 블로그에서 본 수식을 다른 repository의 `step`에 그대로 대입하지 않는다.

대표적인 상태 전이는 `M_t=βM_{t-1}+(1-β)G_t` 같은 momentum에서 시작한다. 어떤 구현은 `(1-β)`를 생략해 scale을 learning rate에 흡수한다. Nesterov candidate도 `βM_t+(1-β)G_t`, `βM_t+G_t` 등 convention을 source로 확인해야 한다. candidate matrix를 `O(·)`로 직교화한 뒤 shape-dependent scale과 lr를 적용한다.

decoupled decay가 있다면 `W←(1-ηλ)W-ηsO(M)`처럼 gradient transform과 분리될 수 있다. lr multiplier가 decay에도 적용되는지 group code를 읽는다. zero-gradient/nonzero-weight fixture는 momentum tail과 decay를 분리한다.

**이름이 같은 구현을 option digest로 구분한다.**

RecipeID에는 momentum convention, Nesterov, NS coefficient/steps/epsilon, input norm, transpose 조건, output scale, state dtype, parameter allowlist, decay 순서와 backend가 있다. 하나라도 달라지면 같은 Muon이라는 label 아래의 다른 algorithm instance다.

KellerJordan 기준 구현, nanochat sharded 구현, PyTorch recipe와 NVIDIA Emerging Optimizers는 서로 다른 목적과 wrapper를 가진다. 공통인 수학 kernel과 달라지는 parameter grouping/dispatch/state를 두 열로 비교한다. source body가 확인되지 않은 최신 port는 이름만으로 동등하다고 쓰지 않는다.

**polar factor를 가장 가까운 직교 행렬이라는 기하로 이해한다**

**SVD에서 update 방향이 어떻게 바뀌는지 본다.**

행렬 `G∈R^{m×n}`의 thin SVD를 `G=UΣVᵀ`라 하자. full-rank에 가까운 경우 polar factor `P=UVᵀ`는 singular vector를 보존하면서 nonzero singular value를 1로 바꾼다. `m≥n`이면 `PᵀP=I_n`, `m≤n`이면 `PPᵀ=I_m`인 partial isometry다.

Frobenius 거리에서 `G`에 가까운 semi-orthogonal matrix 문제와 연결할 수 있지만, practical Muon update에는 momentum, approximate iteration과 output scaling이 붙는다. “weight를 직교화한다”는 표현은 틀리다. 직교화 대상은 일반적으로 parameter가 아니라 update candidate다.

singular value가 큰 방향의 독점을 줄이는 대신 원래 singular magnitude 정보는 버린다. 아주 작은 singular direction도 들어 올릴 수 있어 noise amplification 가능성이 있다. rank deficiency와 low precision에서는 exact polar factor의 비유가 제한된다.

**dual norm 직관의 적용 범위를 고정한다.**

선형화한 loss에서 update norm 제약을 선택하면 steepest direction은 dual norm에 의해 정해진다. matrix spectral norm과 nuclear norm의 dual 관계는 polar 방향을 이해하는 한 관점이다. 하지만 practical scaling, finite NS, momentum과 parameter role selection을 포함한 전체 optimizer가 모든 reparameterization에 불변이라는 결론은 나오지 않는다.

left/right orthogonal coordinate 변환에는 어떤 covariance가 있는지 작은 fixture로 확인한다. arbitrary scaling, transpose storage, TP local shard에는 별 분석이 필요하다. 이론의 범위를 property test 항목으로 바꾼다.

**rectangular matrix의 polar factor와 shape scale을 분리한다**

**tall과 wide orientation을 같은 식으로 오해하지 않는다.**

`m>n`인 tall matrix는 column-orthonormal 방향을, `m<n`인 wide matrix는 row-orthonormal 방향을 목표로 한다. practical implementation은 계산량을 줄이거나 polynomial shape를 맞추려고 wide/tall 중 한 경우 transpose한 뒤 결과를 되돌릴 수 있다. transpose 조건과 final shape는 source branch로 고정한다.

polar output의 Frobenius norm은 대략 `√min(m,n)`이므로 raw output RMS는 shape에 의존한다. parameter width에 따라 multiplier를 곱하는 구현은 layer 사이 effective update scale을 조절한다. 이 scale은 polar 수학 자체와 별 recipe다.

global logical shape와 TP local shard shape 중 어느 것을 scale 식에 쓰는지도 중요하다. local shape를 쓰면 world size 변경만으로 effective lr가 달라질 수 있다. backend가 local approximation을 의도한다면 그 차이를 recipe에 기록한다.

**aspect ratio fixture로 저장 layout의 영향을 제거한다.**

`[2,8]`, `[8,2]`, `[4,4]`, `[1,N]`과 transposed non-contiguous view를 만든다. logical transpose 쌍의 update를 다시 transpose했을 때 예상 관계를 확인한다. stored `[out,in]`와 logical linear map 축을 혼동하지 않는다.

각 shape에서 exact SVD polar, NS output, pre/post scale RMS, spectral norm과 update/weight ratio를 기록한다. 하나의 scalar loss로 shape scale 오류를 찾으려 하지 않는다.

**Newton–Schulz가 inverse square root를 피하는 방식을 유도한다**

**polar decomposition의 계산 경로를 식으로 연결한다.**

full-column-rank tall matrix의 polar factor는 `G(GᵀG)^{-1/2}`로 쓸 수 있고 wide matrix는 `(GGᵀ)^{-1/2}G` 형태를 쓸 수 있다. 명시적 eigendecomposition/SVD는 정확한 oracle로 유용하지만 accelerator에서 매 step 수행하기 비쌀 수 있다. Newton–Schulz 계열은 matrix multiplication polynomial로 polar 방향을 근사한다.

고전적 반복과 practical quintic polynomial은 같지 않을 수 있다. KellerJordan 함수 이름의 `newtonschulz5`에서 5는 다항식 차수 또는 고정 variant와 관계하며, iteration 회수와 혼동하지 않는다. 실제 coefficient literal과 update graph를 fixed source에서 읽는다.

입력 `X_0`를 Frobenius 또는 spectral upper bound로 scale해 singular value를 반복의 안정 영역에 놓는다. `X_{k+1}=aX_k+bA_kX_k+cB_kX_k`처럼 구현되면 `A_k=X_kX_kᵀ`, `B_k=A_kA_k` 등의 exact multiplication order를 기록한다. transpose orientation에 따라 Gram 쪽이 달라진다.

**수렴이라는 말을 residual 세 가지로 나눈다.**

첫째 semi-orthogonality residual `||XᵀX-I||` 또는 `||XXᵀ-I||`, 둘째 exact polar와의 Frobenius/cosine 차이, 셋째 downstream update/loss 차이다. rank-deficient matrix에서는 identity residual이 full-rank처럼 0이 될 수 없으므로 support projector를 기준으로 본다.

iteration별 singular values를 기록하면 큰/작은 방향이 1로 이동하는 모양을 볼 수 있다. residual이 줄어도 output scaling/decay 오류가 있으면 optimizer delta는 틀릴 수 있다. kernel 수렴과 optimizer 의미를 분리한다.

**Newton–Schulz coefficient와 반복 수를 hyperparameter state로 취급한다**

**coefficient는 구현 상수가 아니라 함수 정의의 일부다.**

coefficient triple이나 quintuple은 singular-value transform polynomial을 결정한다. 다른 coefficient set은 수렴 영역, 속도와 overshoot가 다르다. reference 구현의 literal을 documentation default와 대조하고 compiled kernel에 bake된 값도 artifact hash로 고정한다.

반복 수를 늘리면 matmul 수와 temporary lifetime이 늘고 exact polar에 가까워질 수 있지만, 저정밀 rounding 또는 불안정 영역에서는 항상 개선되지 않는다. `steps=0`의 의미가 identity/raw normalized candidate인지 구현에 따라 확인한다.

epsilon은 zero norm division을 막지만 nonzero tiny matrix의 방향을 왜곡할 수 있다. norm에 더하는지 clamp-min인지 branch zero인지 식이 다르다. scale sweep으로 threshold 주변 불연속을 본다.

**hyperparameter sweep을 kernel benchmark와 training trial로 나눈다.**

matrix replay에서는 steps/coefficient/dtype별 residual, cosine, RMS, matmul count, HBM/workspace와 latency를 잰다. model trial에서는 동일 token/compute budget에서 loss와 안정성을 본다. kernel residual 최저가 반드시 최적 validation을 뜻하지 않는다.

coefficient/steps 변경은 checkpoint momentum을 그대로 읽을 수 있어도 다음 update 함수가 달라진다. child RecipeID와 migration event를 만든다. compiled graph/cache도 새 key를 요구한다.

**BF16 Newton–Schulz의 오차를 singular spectrum에서 읽는다**

**cast 위치가 작은 singular direction을 지우는 순간을 찾는다.**

일부 reference 함수는 input candidate를 BF16으로 cast한 뒤 행렬곱을 수행한다. accumulation은 hardware/kernel에 따라 더 높은 정밀도일 수 있지만 operand quantization은 남는다. 큰 singular value가 norm을 지배하면 작은 방향이 BF16 해상도 아래로 내려갈 수 있다.

FP64 SVD oracle, FP32 polynomial, BF16 polynomial을 iteration별 비교한다. condition number, global scale, one-outlier, nearly rank-one과 cancellation fixture를 쓴다. 단순 random Gaussian 평균은 경계 실패를 잘 드러내지 않는다.

Gram matmul의 accumulate dtype, intermediate storage dtype와 final parameter cast를 별로 기록한다. autocast가 함수 내부 explicit cast를 덮는지, compiled/fused path가 다른 precision을 쓰는지 확인한다.

**오차 예산을 update와 next loss까지 전달한다.**

NS output residual, exact polar cosine, update RMS와 one-step quadratic loss 차이에 threshold를 둔다. scale invariance도 `O(cG)`와 `O(G)`의 방향/크기로 확인한다. epsilon dominance 구간은 별 기대값을 쓴다.

non-finite가 없어도 singular 방향이 무너질 수 있다. dashboard에 sampled residual과 input condition proxy, output spectral spread를 둔다. 비싼 SVD는 offline sample에서 하고 production은 cheaper residual을 사용한다.

**KellerJordan 기준 구현을 순수 함수와 optimizer class로 분리한다**

**`zeropower_via_newtonschulz5`의 증명 범위를 한정한다.**

고정 commit `f98f1cacc0263b04290753e32be8d498c1efc806`의 순수 함수는 입력 차원 검사, BF16 cast, tall/wide transpose, norm scaling, coefficient와 반복 matmul, final transpose를 읽는 기준점이다. 이 함수는 polar 근사 kernel의 source evidence다.

그러나 이 함수만으로 parameter group, momentum/Nesterov, weight decay, learning-rate scale, distributed owner와 checkpoint를 증명할 수 없다. 같은 파일의 optimizer class와 호출자를 별 card로 만든다. line span과 blob digest를 고정한다.

mutation test는 transpose 조건 반전, coefficient 한 자리 변경, normalization 제거, iteration 감소와 final scale 누락을 포함한다. FP64 oracle/property test가 예상 항목에서 실패해야 한다.

**분산 class에서 update ownership을 읽는다.**

parameter가 rank 사이 round-robin 또는 bucket으로 배정되고 owner가 momentum/orthogonalization을 계산한 뒤 all-gather하는지 actual class를 따른다. optimizer class가 DDP-reduced gradient를 기대하는지, 자체 reduction을 하는지 구분한다.

parameter iteration order가 모든 rank에서 같다는 가정, padding/flatten, empty owner와 collective sequence를 확인한다. Python list 순서가 distributed protocol이면 stable ParameterID sort와 digest validator를 추가한다.

checkpoint state key가 local parameter object index에 묶이면 world-size migration이 어렵다. global ParameterID mapping과 owner plan이 없으면 fixed topology 지원으로 제한한다.

**nanochat의 Muon 경로를 sharded training 사건으로 읽는다**

**`_reduce_muon`과 `_compute_muon`의 책임을 분리한다.**

고정 nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`에서 reduce/scatter와 local compute가 어떤 input buffer, shape metadata와 process group을 소비하는지 읽는다. reference class와 이름이 같아도 sharded gradient/state/update의 owner가 다를 수 있다.

gradient accumulation 종료, scaling/unscale, non-finite check, distributed reduce와 Muon compute의 사건 순서를 trainer call graph로 잇는다. helper 함수만 읽고 AMP/scheduler/commit contract를 추정하지 않는다.

local matrix를 orthogonalize한 뒤 global update를 재구성하는지, full logical matrix를 owner에 모으는지에 따라 geometry가 달라진다. TP shard에 local polar를 적용하면 full polar와 일반적으로 같지 않다. actual shape/collective를 equation에 반영한다.

**nanochat recipe를 모든 모델의 보편 기본값으로 쓰지 않는다.**

parameter grouping은 nanochat의 module/architecture 이름과 training scale에 맞춰져 있다. 다른 모델의 embedding, expert, adapter, tied head에 그대로 적용하지 않는다. model inventory에서 semantic allowlist를 다시 만든다.

upstream training loss 개선은 해당 data/model/hardware/recipe의 통합 결과다. Muon 단독 효과를 주장하려면 paired ablation과 trial budget이 필요하다. source evidence와 empirical claim을 분리한다.

**PyTorch Muon recipe를 framework integration 관점에서 감사한다**

**tutorial과 stable API를 같은 것으로 보지 않는다.**

PyTorch 공식 recipe는 optimizer 작성과 사용법, compile 또는 distributed 예시를 제공할 수 있지만 tutorial code가 모든 release의 stable `torch.optim` API라는 뜻은 아니다. page revision, linked source commit과 실행 환경을 고정한다.

Optimizer subclass의 `step`, parameter group default, state initialization, `grad is None`, sparse gradient, closure와 serialization을 읽는다. foreach/fused/capturable/differentiable support가 실제로 있는지 source branch로 확인한다. generic AdamW option을 Muon port가 자동 지원한다고 가정하지 않는다.

AMP integration은 unscale/non-finite detection 뒤 Muon step이 호출되는지, found-inf에서 momentum/NS/counter가 멈추는지 본다. optimizer hook과 scheduler가 successful commit을 받는지도 확인한다.

**PyTorch dispatcher의 effective backend를 기록한다.**

device/dtype/shape에 따라 eager matmul, compiled graph, Triton/custom op가 선택될 수 있다. requested flag와 actual profiler trace를 함께 둔다. unsupported path가 silent eager fallback하면 correctness는 유지될 수 있지만 performance cell은 별 backend다.

state dict round-trip은 parameter group 순서와 stable ID mapping 문제를 검사한다. 새 process, group reorder와 excluded parameter 추가 mutation을 넣는다. load 성공 뒤 next delta를 reference와 비교한다.

**NVIDIA Emerging Optimizers를 backend와 test evidence로 사용한다**

**고정 source에서 orthogonalized optimizer dispatch를 찾는다.**

commit `83537ba67cb4c998251567f78a534776fecb1965`의 Muon module은 backend 선택, coefficient/iteration, validation과 fallback을 읽는 좌표다. package wrapper가 parameter selection, optimizer group과 state를 어떻게 구성하는지 순수 NS utility와 분리한다.

test 파일의 coefficient, invalid input, shape/dtype/backend case가 무엇을 assert하는지 표로 만든다. test가 통과해도 대형 model convergence, multi-node fault recovery, all topology checkpoint를 증명하지 않는다. `proves`와 `does_not_prove`를 함께 적는다.

dependency의 외부 kernel 또는 compiler revision도 source lock에 넣는다. NVIDIA repository commit만 고정하고 pip dependency가 움직이면 실행 body가 달라질 수 있다.

**optimized backend와 reference path를 같은 tape로 비교한다.**

contiguous/noncontiguous, square/tall/wide, odd dimensions, zero/rank-deficient/ill-conditioned matrix를 사용한다. iteration intermediate를 노출할 수 없으면 input/output residual과 generated graph를 비교한다.

fallback이 발생하면 reason, actual backend와 latency를 기록한다. fallback result를 optimized backend 성능으로 집계하지 않는다. unsupported hardware는 `NotRun` 또는 explicit reject다.

**parameter role selection을 optimizer 수학의 정의역으로 만든다**

**`ndim==2` 규칙이 만드는 오분류를 찾는다.**

embedding table, LM head, router, relative-position table와 일부 adapter도 2차원이다. 그러나 token-frequency sparse row update, vocabulary classifier, routing probability와 low-rank factor는 dense hidden transform과 gradient 기하가 다르다. 단순 차원 규칙은 optimizer 효과와 grouping 효과를 섞는다.

inventory에는 ParameterID, module class/path, semantic role, global/local shape, tied alias, sparsity, sharding, gradient frequency와 intended optimizer를 기록한다. allowlist/denylist와 reason을 artifact로 저장한다. regex만 사용하면 renamed/fused module을 놓칠 수 있다.

attention q/k/v/o와 MLP gate/up/down은 후보가 될 수 있지만 fused QKV/gate-up은 논리 submatrix를 함께 직교화할지 분리할지 결정해야 한다. 전체 fused matrix polar는 sub-projection별 polar와 같지 않다. checkpoint layout 편의로 수학 grouping을 결정하지 않는다.

**expert와 adapter matrix는 visitation과 rank를 고려한다.**

MoE expert는 token visitation이 불균등하고 zero-gradient step이 있다. router는 작은 margin과 probability geometry 때문에 기본적으로 AdamW 유지 또는 별 ablation이 필요하다. expert별 Muon state age와 momentum decay convention을 확인한다.

LoRA A/B는 매우 rectangular하고 low-rank factorization에 reparameterization ambiguity가 있다. 각 factor에 Muon을 적용하면 merged delta에 어떤 기하를 만드는지 별 연구가 필요하다. 검증 없이는 AdamW fallback으로 둔다.

**hybrid Muon·AdamW group을 하나의 원자적 step으로 설계한다**

**두 optimizer의 state machine을 하나의 commit에 묶는다.**

matrix group은 Muon momentum/NS를, embedding/norm/bias/head 등은 AdamW first/second moment를 가진다. scheduler base lr와 group multiplier, decay 정책이 다를 수 있지만 model UpdateID는 두 group 모두 적용되었을 때만 전진해야 한다.

AMP overflow 또는 rank failure에서 한 group만 update되면 model이 혼합 세대가 된다. 모든 group의 gradient validation, candidate delta 계산과 commit을 단계로 나눈다. in-place parameter 변경 전에 abort 가능 지점을 마련하거나 실패 시 process를 폐기하고 durable parent로 rollback한다.

common AdamW group은 all-AdamW paired run과 gradient tape에서 동일 option/clock을 가져야 한다. hybrid run의 공통군 delta가 다르면 scheduler/grouping/reduction drift가 섞였다.

**optimizer group 순서를 stable semantic ID로 보존한다.**

Python parameter list order가 checkpoint mapping이나 collective 순서를 결정하지 않게 한다. group ID와 ParameterID sorted list, option digest와 state coverage를 manifest에 둔다. tied parameter가 두 group에 들어가면 construction에서 거부한다.

resume 뒤 Muon/AdamW step, scheduler, scaler와 data cursor가 같은 committed UpdateID를 가리키는지 검사한다. group 하나를 reset하면 exact resume가 아니라 optimizer migration branch다.

**momentum convention을 두 step과 zero-gradient로 식별한다**

**첫 step만으로 구분되지 않는 구현 차이를 드러낸다.**

momentum buffer가 `M←βM+(1-β)G`인지 `M←βM+G`인지 learning-rate scale과 함께 보면 첫 step 결과가 비슷하게 조정될 수 있다. 두 개의 비평행 gradient와 세 번째 zero gradient를 replay하면 state와 tail 차이가 드러난다.

Nesterov candidate가 current gradient를 어느 coefficient로 더하는지도 중간 tensor를 기록한다. orthogonalization 전 candidate가 다른데 final polar가 우연히 같을 수 있으므로 momentum 자체를 checkpoint/oracle에 포함한다.

zero gradient에서 NS를 호출하는지, momentum tail만 update하는지, decay만 적용하는지 확인한다. `grad=None`과 zero tensor는 다른 경로일 수 있다. unused expert와 frozen parameter에서 중요하다.

**momentum dtype과 rescale을 checkpoint schema에 넣는다.**

parameter BF16, state FP32 또는 state BF16 조합이 있다. NS input cast와 momentum storage cast를 구분한다. state dtype migration에는 silent load cast가 아니라 child schema와 one-step tolerance가 필요하다.

state initialization이 lazy라면 처음 gradient를 받은 UpdateID와 owner를 기록한다. topology 변경 또는 parameter unfreeze 뒤 lazy state가 replica마다 다른 step에 생기지 않게 한다.

**weight decay와 matrix transform의 비가환성을 손계산한다**

**coupled L2와 decoupled decay를 구분한다.**

gradient에 `λW`를 더한 뒤 polar transform하면 regularization 방향도 singular-value 평탄화에 섞인다. decoupled decay는 transformed gradient와 별로 `-ηλW`를 적용한다. 두 방식은 같은 weight decay 숫자를 써도 다른 함수다.

2×2 W와 zero/data gradient를 두고 coupled/decoupled 결과를 계산한다. zero data gradient에서 coupled polar가 parameter 방향을 단순 shrink와 다르게 바꾸는 것을 볼 수 있다. implementation option 이름보다 actual order를 따른다.

shape scale와 group lr가 decay coefficient에 적용되는지도 확인한다. matrix group과 AdamW group의 동일 `weight_decay` 값이 step당 같은 shrink ratio를 의미하지 않을 수 있다.

**decay 제외 role을 semantic inventory로 검증한다.**

norm/bias는 흔히 decay 제외지만 model-specific QK norm, scale와 mHC parameter가 regex에서 누락될 수 있다. tied embedding/head가 서로 다른 정책에 매칭되면 conflict다. actual group report와 zero-gradient delta로 확인한다.

decay option 변경은 momentum state를 유지할 수 있어도 objective trajectory가 달라진다. child RecipeID, scheduler와 paired trial을 요구한다.

**Newton–Schulz CUDA graph의 memory lifetime을 계산한다**

**한 iteration의 temporary를 시간축에 놓는다.**

input candidate X, Gram A, higher polynomial temporary, output와 momentum이 어느 시점에 동시에 살아 있는지 기록한다. eager allocator는 iteration마다 buffer를 만들 수 있고 compiled/fused graph는 재사용할 수 있다. theoretical `O(mn)`만으로 peak를 추정하지 않는다.

tall/wide transpose가 view인지 contiguous copy인지, noncontiguous input을 kernel이 지원하는지 본다. smaller Gram orientation은 compute를 줄일 수 있지만 copy/workspace가 추가된다. odd shape padding과 tile waste를 executed FLOPs에 포함한다.

CUDA Graph capture에서는 lazy state/workspace를 warm-up에서 materialize하고 pointer가 안정해야 한다. overflow/skip branch나 dynamic shape가 graph 밖으로 나가는지 확인한다. replay가 old coefficient/lr scalar를 capture하지 않게 한다.

**kernel trace를 logical iteration과 연결한다.**

각 matmul kernel에 iteration index와 tensor role NVTX/event를 연결한다. compiler가 fusion/reorder해도 generated graph에서 polynomial dependency를 확인한다. input/output checksum과 residual은 reference path와 비교한다.

성능은 optimizer-only time, whole step critical path, memory 때문에 줄어든 microbatch와 checkpoint overhead까지 본다. 빠른 NS kernel이 training throughput 개선을 보장하지 않는다.

**distributed Muon의 세 가지 소유권 설계를 비교한다**

**replicated full geometry의 비용을 계산한다.**

모든 rank가 같은 full gradient와 momentum을 가지고 NS를 수행하면 통신 후 계산이 복제된다. 구현은 단순하지만 state/compute가 world size만큼 중복된다. gradient reduction이 이미 DDP에서 일어났는지 확인한다.

**owner-compute와 gather의 critical path를 계산한다.**

각 matrix를 owner rank에 배정해 owner가 NS를 계산하고 delta/parameter를 all-gather할 수 있다. compute는 분산되지만 큰 matrix 배치 불균형과 owner straggler가 생긴다. assignment를 numel뿐 아니라 aspect/NS cost로 balance한다.

**shard-local approximation의 기하 차이를 명시한다.**

TP/FSDP local shard마다 polar를 적용하면 full matrix polar와 일반적으로 같지 않다. 통신은 줄지만 algorithm이 달라진다. local-vs-full FP64 fixture에서 delta cosine/spectrum을 측정하고 의도된 approximation으로 RecipeID에 넣는다.

세 설계는 logical bytes, wire bytes, replicated compute, peak memory, tail, checkpoint/reshard와 fault blast radius로 비교한다. “분산 지원” 하나의 체크박스로 합치지 않는다.

**tensor parallel shard와 polar transform의 비가환성을 증명한다**

**row/column shard 예제로 반례를 만든다.**

행렬 G를 row blocks `G_1,G_2`로 나누어 각각 polar한 뒤 concatenate한 결과는 일반적으로 full `polar(G)`와 다르다. column shard도 같다. 4×4 non-block-diagonal gradient로 손계산/SVD reference를 만든다.

full geometry를 보존하려면 full matrix 또는 충분한 Gram/cross-block 정보를 교환해야 한다. communication pattern은 shard axis와 chosen formula에 따라 달라진다. `GᵀG`는 row shards에서 local Gram 합 reduction으로 만들 수 있지만 최종 inverse-root 적용과 state owner를 설계해야 한다.

parameter stored transpose와 TP shard 축을 logical map으로 복원한다. local shape scale가 world size에 의존하지 않는지 본다. TP degree 1/2/4에서 same global gradient delta를 비교한다.

**expert parallel과 pipeline stage ownership을 추가한다.**

MoE global expert는 한 EP owner에 완전히 있을 수도 있고 expert TP로 나뉠 수 있다. pipeline stage는 parameter 집합을 분리한다. optimizer collective group은 model gradient group과 정확히 맞아야 한다.

empty gradient expert와 stage bubble에서도 collective ordinal을 유지한다. owner kill/zero parameter bucket을 failure fixture에 넣는다. global ParameterID가 checkpoint reshard의 join key다.

**distributed optimizer communication을 wire byte와 burst로 센다**

**collective payload를 logical state별로 분해한다.**

gradient reduce/reduce-scatter, owner input gather, transformed delta/parameter all-gather, state reshard와 checkpoint staging을 구분한다. payload element count×dtype bytes 외에 padding, protocol와 repeated chunks를 trace에서 측정한다.

매 step byte와 cadence burst를 분리한다. Muon은 매 step NS/delta exchange가 있을 수 있고 Shampoo/SOAP는 root/basis cadence에서 큰 burst가 생긴다. 평균 bandwidth가 정상이어도 cadence p99가 training tail을 지배할 수 있다.

overlap은 API async 여부가 아니라 timeline의 실제 겹침으로 검증한다. same SM/HBM/network 경쟁, wait event와 buffer lifetime을 본다. optimizer collective가 backward critical path에 들어가는 지점을 표시한다.

**communication correctness를 count보다 identity로 검증한다.**

같은 shape matrix가 여러 개면 order swap이 byte/count 검사를 통과한다. payload에 ParameterID/offset digest를 붙이거나 role-coded gradient fixture를 사용한다. all-rank order digest가 collective 전에 일치해야 한다.

timeout은 peer failure, order mismatch, conditional skip와 network tail을 구분한다. rank별 collective ordinal과 state generation을 incident artifact에 둔다.

**optimizer state sharding과 ZeRO/FSDP의 경계를 분리한다**

**저장 state 분할과 global 기하 보존을 다른 질문으로 묻는다.**

AdamW moment는 coordinate-wise라 parameter shard와 함께 자연스럽게 나눌 수 있다. Muon momentum도 저장은 shard할 수 있지만 polar transform이 full logical matrix를 요구한다면 compute 때 gather/Gram reduction이 필요하다. “state sharded”가 “communication-free”를 뜻하지 않는다.

FSDP flat parameter는 여러 semantic matrices를 한 storage에 합칠 수 있다. Muon grouping은 flatten 이전 module role과 logical boundaries를 보존해야 한다. flat slice 일부가 두 optimizer family에 걸치면 state/step wrapper가 semantic view를 복원한다.

state dict에는 flat offset, original ParameterID/shape, optimizer owner와 NS recipe를 둔다. load가 parameter order에 의존하지 않게 한다. world-size reshard가 momentum slice와 cadence를 함께 옮긴다.

**offload를 performance와 failure state로 포함한다.**

CPU/NVMe optimizer state offload는 GPU memory를 줄이지만 momentum transfer와 NS compute placement가 달라진다. CPU에서 NS를 수행하는 silent fallback을 GPU benchmark로 집계하지 않는다. pinned buffer, async copy와 failure lifetime을 추적한다.

offload checkpoint/restart는 in-flight copy와 durable commit을 구분한다. stale host state가 새 GPU parameter에 적용되지 않게 generation을 검사한다.

**matrix optimizer checkpoint migration을 field 단위로 판정한다**

**portable state와 method-specific state를 나눈다.**

parameter weight와 global ParameterID는 공통이지만 AdamW first/second moment, Muon momentum, Shampoo factor/root, SOAP basis/rotated moment는 서로 직접 호환되지 않는다. 같은 shape라고 이름을 바꿔 load하지 않는다.

Muon 구현 A→B도 momentum convention, scale, transpose/grouping과 state dtype이 다르면 exact migration이 아닐 수 있다. field별 `exact`, `converted`, `reset`, `missing`, `unsupported`를 보고한다. reset group에는 warmup과 risk를 선언한다.

checkpoint manifest에는 RecipeID, source/backend, parameter group selection, state tensor role/dtype/shape, step/cadence와 owner topology를 기록한다. optimizer class pickle에 의미를 맡기지 않는다.

**migration은 다음 gradient tape로 증명한다.**

old checkpoint를 old code에서 load한 control next delta와 migration→new code next delta를 비교한다. exact migration에는 사전 tolerance를, reset에는 expected divergence와 warm-start baseline을 둔다. load 성공은 증거가 아니다.

group reorder, parameter rename, tied alias, TP/EP degree 변경과 missing state를 negative fixture로 넣는다. unknown field를 drop하지 않고 거부하거나 explicit conversion한다.

**checkpoint cadence 경계에서 원자성을 시험한다**

**momentum write와 parameter apply 사이 실패를 분리한다.**

optimizer step은 gradient validate, state candidate, transformed delta, parameter candidate와 commit으로 나눌 수 있다. in-place momentum을 먼저 바꾼 뒤 NS가 실패하면 old parameter/new state 혼합이 생긴다. process를 계속 쓰지 않거나 transactional buffer가 필요하다.

hybrid AdamW 그룹과 matrix 그룹, scheduler/scaler가 모두 같은 UpdateID에 commit해야 한다. Shampoo root/SOAP basis처럼 별 cadence state가 있다면 dependency generation을 기록한다. auxiliary state만 전진하는 정책은 명시적이어야 한다.

failure injection은 state 계산 전/후, collective 중, 일부 parameter apply, checkpoint shard write와 root publish에 둔다. 허용 durable terminal은 old complete 또는 new complete generation뿐이다.

**resume 첫 세 update를 uninterrupted branch와 비교한다.**

parameter, momentum/moments, cadence, lr/scaler, RNG/data cursor와 grouping을 확인한다. 첫 loss만 맞아도 stale momentum이 둘째 step부터 드러날 수 있다. selected matrix의 intermediate NS input/output도 비교한다.

partial checkpoint는 commit marker, component UpdateID와 Merkle/checksum coverage에서 load 전에 거부한다. previous complete parent를 보존한다.

**hyperparameter를 수학·시스템·관측 효과표로 만든다**

**momentum과 learning rate를 update geometry에서 읽는다.**

momentum β는 gradient smoothing과 state memory time constant를 바꾼다. NS 뒤 방향 크기가 평탄화되므로 lr와 shape scale의 의미가 AdamW와 다르다. 같은 숫자 lr를 공정 baseline이라고 할 수 없다.

Nesterov는 current gradient 반응과 momentum candidate를 바꾼다. steps/coefficient/epsilon은 residual·matmul·precision을, state dtype은 memory와 누적 오차를 바꾼다. weight decay는 shrink를, parameter selection은 objective 자체의 optimizer 배치를 바꾼다.

각 option 행에는 config field/default, source consumer, changed tensor/state, compute/memory/collective, expected metric, failure boundary와 checkpoint compatibility를 둔다. 값이 parser에만 존재하고 소비되지 않으면 dead option이다.

**paired rehearsal에서 한 축씩 바꾼다.**

same checkpoint/gradient tape로 intermediate momentum, NS residual, delta spectrum/update ratio, bytes와 latency를 비교한다. long run은 동일 token/compute/trial budget을 사용한다. 여러 option을 한 번에 바꾸지 않는다.

production 변경에는 child RecipeID와 rollback parent를 지정한다. observed effect가 예상 first difference와 다르면 승인하지 않는다.

**Muon learning-rate scale을 AdamW와 같은 숫자로 비교하지 않는다**

**update RMS를 parameter role과 shape별로 측정한다.**

AdamW는 coordinate second moment와 epsilon을 통해 update 크기를 조절하고 Muon은 matrix direction transform과 shape scale을 사용한다. base lr가 같아도 per-parameter delta RMS, spectral norm과 update/weight ratio는 다르다.

attention square matrix, wide MLP up/gate, tall down과 small adapter에서 candidate scale을 측정한다. global average가 아니라 role별 p50/p99와 layer depth를 본다. tied/embedding AdamW fallback도 공통군으로 확인한다.

learning-rate sweep은 각 optimizer에 같은 trial compute와 사전 탐색 범위를 준다. AdamW 한 기본값과 Muon 다수 튜닝 run을 비교하지 않는다. 실패/NaN/OOM trial도 결과에 포함한다.

**scheduler warmup과 optimizer scale의 곱을 기록한다.**

effective matrix step은 base schedule×group multiplier×shape scale×transformed direction이다. dashboard에 schedule lr만 표시하면 실제 update spike를 놓친다. update/weight ratio와 NS output RMS를 함께 본다.

optimizer migration에서 warmup을 재시작하는지, token/global step clock을 이어 가는지 명시한다. 13장의 scheduler contract와 같은 UpdateID를 쓴다.

**Muon 적용 matrix의 gradient spectrum을 데이터와 연결한다**

**optimizer 선택 전에 실제 gradient가 행렬 구조를 갖는지 본다.**

대표 layer/role에서 gradient singular spectrum, stable rank, condition proxy, temporal cosine와 update/weight ratio를 sample한다. full SVD는 비싸므로 offline gradient snapshot에 적용하고 production은 randomized/low-cost summary를 사용한다.

데이터 mixture, sequence length와 training phase가 spectrum을 바꿀 수 있다. 초기 수백 step의 분포만으로 전체 run을 결정하지 않는다. curriculum knot와 SFT/RL objective 전환에서 다시 측정한다.

Muon이 singular magnitude를 평탄화하는 것이 유용한지 task loss만으로 사후 설명하지 않는다. predeclared hypothesis와 mediator—spectrum, update isotropy, layer balance—를 기록한다. mediator가 변하지 않았는데 품질만 바뀌면 다른 grouping/schedule 효과를 조사한다.

**spectrum 관측 자체의 수치 오류를 통제한다.**

BF16 gradient를 FP32/FP64 분석으로 올리고 shard를 global logical matrix로 복원한다. TP local spectrum을 global이라고 쓰지 않는다. sparse/zero expert gradient와 tied alias를 별 처리한다.

snapshot은 data/BatchID, loss scale/unscale, reduction 전후와 gradient accumulation stage를 명시한다. 서로 다른 gradient 정의를 비교하지 않는다.

**failure fixture를 수학·dispatch·state·commit 네 층으로 구성한다**

**수학 층은 경계 matrix를 사용한다.**

zero, tiny norm, rank-one, repeated singular value, ill-conditioned, NaN/Inf, tall/wide/1×N과 extreme scale을 넣는다. exact polar 또는 property oracle, finite와 predeclared residual을 검사한다.

**dispatch 층은 layout와 backend를 깨뜨린다.**

noncontiguous transpose, odd dimension, unsupported dtype/device, compiler cache mismatch, silent fallback과 wrong coefficient kernel을 넣는다. effective backend와 first detector를 확인한다.

**state 층은 identity와 clock을 깨뜨린다.**

momentum parameter swap, stale state dtype, missing counter, group reorder, tied duplicate와 local/global shape scale 오류를 넣는다. state load 전에 semantic mapping validator가 잡아야 한다.

**commit 층은 분산 실패를 주입한다.**

owner kill, collective order/count mismatch, partial group apply, checkpoint shard 누락과 scheduler one-step-ahead를 넣는다. old/new complete generation 외 terminal을 거부한다. recovery 뒤 next delta를 uninterrupted reference와 비교한다.

**optimizer observability를 update 원인 그래프로 설계한다**

**ParameterID별 지표를 적절한 cardinality로 요약한다.**

role/layer별 gradient norm, momentum norm, NS input norm/residual, delta RMS/spectral norm, update/weight ratio, state dtype와 non-finite를 수집한다. 모든 ParameterID를 Prometheus label로 넣지 않고 sampled artifact/trace로 상세를 보존한다.

system 지표에는 optimizer kernel time, matmul shapes, workspace/peak memory, owner queue, collective bytes/tail, fallback/compile과 checkpoint pause를 기록한다. same UpdateID로 loss/scheduler와 join한다.

causal chain 예시는 gradient condition 악화→BF16 small direction 손실→NS residual 증가→delta spectral change→layer update spike다. 또는 parameter regroup→owner imbalance→collective tail→step throughput 하락이다.

**경보를 action 가능한 분기로 만든다.**

NS residual 경보는 input norm/condition, dtype, coefficient/backend와 transpose를 확인한다. update spike는 lr×shape scale, momentum, decay와 group role을 본다. throughput은 shape histogram, fallback, owner/collective와 workspace를 본다.

metric 정상인데 validation이 갈리면 data/gradient tape, common AdamW group parity와 long-horizon state를 조사한다. loss 하나로 optimizer source를 단정하지 않는다.

**benchmark fairness를 네 개의 예산으로 분리한다**

**token·FLOP·wall-clock·운영 예산은 다른 질문이다.**

동일 token budget은 sample efficiency를, 동일 model+optimizer FLOPs는 compute efficiency를, 동일 wall-clock은 system efficiency를 본다. 동일 운영 예산은 checkpoint, 장애/복구와 tuning trial까지 포함한다. 한 결과를 다른 질문의 답으로 사용하지 않는다.

Muon의 NS GEMM, Shampoo root와 communication을 optimizer cost에 넣는다. batch size가 memory 때문에 달라지면 effective batch/schedule을 통제하거나 차이를 결과로 보고한다. compiler warmup과 cadence burst도 wall-clock에 포함한다.

AdamW baseline은 동일한 tuning compute, seed와 engineering quality를 받는다. strong baseline의 fused/foreach backend를 끄고 후보만 최적화하지 않는다. candidate-specific 안정 범위와 baseline-specific 범위를 사전 등록한다.

**결과를 success-only 평균으로 만들지 않는다.**

NaN, OOM, hang과 recovery 실패 run을 분모에 남긴다. multiple seed confidence interval, early-stop rule과 selection criterion을 predeclare한다. final best checkpoint만 아니라 trajectory와 resource를 공개한다.

quality는 overall뿐 아니라 data/domain/task slice와 safety를 본다. optimizer 개선이 특정 rare slice를 희생할 수 있다. checkpoint/resume parity와 serving export compatibility도 admission 표에 넣는다.

**gradient replay로 optimizer 효과를 model noise에서 분리한다**

**동일 gradient tape에서 update function만 바꾼다.**

selected ParameterID의 pre-weight와 여러 step gradient를 FP32 artifact로 저장한다. AdamW, Muon variants, Shampoo/SOAP 등이 동일 tape를 소비해 state/delta trajectory를 만든다. data/model forward stochasticity가 제거되어 algorithm 차이를 좁힐 수 있다.

tape는 reduction/unscale/clipping 전후 중 어느 gradient인지 명시한다. parameter group, shape/layout와 UpdateID를 포함한다. tied alias와 sparse/None gradient를 보존한다. privacy/size 때문에 full tape를 저장하지 못하면 deterministic generator와 digest를 사용한다.

intermediate momentum, NS residual, factor/root와 final delta를 비교한다. common AdamW group은 hybrid/all-AdamW에서 exact 또는 expected tolerance로 같아야 한다.

**replay의 증명 범위를 확대하지 않는다.**

고정 gradient는 optimizer가 바꿀 미래 gradient distribution과 closed-loop training을 반영하지 않는다. 수학/implementation parity를 증명하고 장기 model quality는 paired training이 담당한다. replay가 빠르다는 이유로 convergence 실험을 생략하지 않는다.

분산 replay는 full logical tape를 shard해 TP/owner design의 delta를 global reference와 비교한다. communication failure fixture도 같은 identity를 사용한다.

**closed-loop training에서 optimizer와 gradient 분포의 상호작용을 본다**

**optimizer가 다음 gradient를 바꾼다는 사실을 측정한다.**

한 step 뒤 parameter가 달라지면 다음 activation/loss/gradient spectrum이 달라진다. gradient replay의 open-loop 결과와 실제 training trajectory를 구분한다. 동일 initialization/data order에서 selected layer의 spectrum/update를 여러 step 추적한다.

초기 loss 개선이 안정 구간에서도 유지되는지, update isotropy가 representation이나 downstream quality와 어떤 관계인지 본다. causal mediation을 확정하기 어렵다면 상관과 가설로 제한한다.

hybrid parameter selection도 closed-loop에 영향을 준다. embedding/head AdamW와 hidden Muon의 상호작용, expert visitation과 adapter를 별 slice로 본다. optimizer family만 바꿨다는 주장에는 grouping/scheduler가 같다는 증거가 필요하다.

**checkpoint branch로 trajectory 비교를 재현한다.**

공통 parent checkpoint에서 AdamW와 candidate child를 만들고 같은 next data/RNG를 사용한다. parent optimizer state를 candidate로 어떻게 reset/migrate했는지 명시한다. warmup 차이를 control한다.

branch마다 source/RecipeID, data cursor와 resource budget을 보존한다. cherry-pick된 best run이 아니라 모든 branch 결과를 기록한다.

**Muon과 Shampoo·SOAP을 같은 ‘행렬 optimizer’로 뭉개지 않는다**

**현재 gradient 방향과 누적 통계의 차이를 본다.**

Muon은 momentum candidate의 polar-like 방향을 매 update 계산하는 계열이다. Shampoo는 gradient outer-product factor를 시간 누적하고 inverse root를 cadence마다 갱신한다. SOAP은 basis를 추정/회전하며 그 좌표에서 adaptive moment를 운용할 수 있다. state memory와 시간 의존성이 다르다.

한 gradient에서 idealized relation이 있어도 practical damping, block, graft, EMA와 cadence 때문에 update가 같지 않다. exact paper equation, reference implementation과 production wrapper를 분리한다.

Shampoo/SOAP은 factor/root/basis age와 conditioning, Muon은 NS input/output residual을 관측한다. failure fixture도 root corruption/basis permutation과 coefficient/transpose 오류로 다르다.

**parameter role과 분산 비용을 같은 표에서 비교한다.**

large rectangular MLP, square attention, embedding, expert와 adapter에서 eligible role, persistent/peak bytes, collective와 cadence를 계산한다. block partition은 geometry approximation을 바꾼다. Muon shard-local polar도 별 approximation이다.

method 선택은 validation loss 표 한 줄이 아니라 target shape/hardware/topology와 recovery SLO의 함수다. unsupported cell은 미검증으로 둔다.

**최신 optimizer 주장을 paper·code·run evidence로 분리한다**

**논문 결과는 exact 실험 조건과 함께 읽는다.**

모델 크기, data, token budget, baseline tuning, hardware, batch/schedule와 evaluation을 기록한다. 보고된 speedup이 steps, tokens, FLOPs 또는 wall-clock 중 무엇인지 확인한다. 다른 예산으로 재표현하지 않는다.

official repository가 있어도 paper training commit과 공개 head가 같다는 보장이 없다. resolved commit, config와 checkpoint를 고정한다. third-party port는 별 implementation evidence다.

upstream unit test는 수치 branch를, integration test는 framework call을, 공개 training log는 해당 run을 증명한다. 이들을 결합해 검증하지 않은 multi-cluster recovery나 모든 model convergence를 주장하지 않는다.

**미검증 칸을 독자의 다음 조사 계획으로 바꾼다.**

source 미공개, hardware 부재, long-run 미실행, topology 미지원과 checkpoint migration 미검증을 구분한다. 각 칸에 필요한 artifact, function/fixture와 종료 조건을 적는다.

최신성은 evidence 폐쇄성을 대신하지 않는다. 재현 가능한 구버전 구현이 floating 최신 코드보다 수학/운영 분석에 더 나은 기준점일 수 있다. 새 revision은 semantic diff 뒤 child card로 추가한다.

**Muon 도입을 위한 실무형 parameter 감사표**

**모델 tree에서 후보를 추출한다.**

module traversal로 각 parameter의 stable path/class, shape, tied alias, sharding과 trainable 여부를 수집한다. semantic mapper가 attention/MLP/expert/embedding/head/norm/router/adapter 역할을 지정한다. unknown은 자동 Muon이 아니라 review queue로 보낸다.

각 후보에 matrix transform 의미, aspect ratio, gradient density/visit, expected state/workspace와 distributed plan을 적는다. fused QKV/gate-up은 split/whole 선택과 근거를 갖는다. TP local geometry와 full geometry policy를 명시한다.

**runtime state와 selection manifest를 양방향 정산한다.**

expected Muon/AdamW ParameterID 집합과 optimizer actual groups/state keys를 비교한다. 누락, 중복, tied conflict와 unexpected fallback을 step 전에 거부한다. lazy state는 첫 gradient 후 다시 정산한다.

selected/excluded 표본에 고정 gradient를 넣어 owner, momentum/moments, delta와 decay를 확인한다. model config/revision이 바뀌면 selection manifest를 재생성하고 parent diff를 검토한다.

**한 matrix update를 FP64로 완전히 계산하는 독자 실습**

**비대칭 3×2 gradient로 방향을 비교한다.**

W와 두 step의 G를 작은 유리수로 정하고 FP64 SVD로 `UVᵀ`를 계산한다. Keller coefficient의 NS를 iteration별 재생해 input norm, Gram, polynomial output, residual과 exact polar 차이를 표에 쓴다. tall orientation과 transpose branch를 확인한다.

momentum/Nesterov convention 두 가지, shape scale, lr와 decay를 적용해 final delta를 계산한다. AdamW baseline의 m/v/delta도 같은 W,G에서 계산한다. 같은 lr 숫자가 update RMS를 맞추지 않는 것을 확인한다.

둘째 gradient는 첫째와 비평행, 셋째는 zero로 하여 state memory를 드러낸다. BF16 cast variant와 coefficient mutation이 expected residual을 깨뜨리는지 본다.

**분산 shard로 나눠 full과 local geometry를 비교한다.**

행 shard 두 개를 각각 polar한 결과와 full polar를 비교한다. owner-gather 설계는 full reference와 맞고 local approximation은 다름을 수치로 보인다. communication payload와 state owner를 표에 추가한다.

checkpoint after step 2를 저장하고 새 process에서 step 3을 재생한다. momentum, RecipeID와 group mapping이 빠지면 delta가 달라져야 한다. 이 실습이 장 전체의 semantic checksum이다.

**optimizer source upgrade를 의미 단위로 diff한다**

**변경을 식·state·dispatch·test로 분류한다.**

coefficient, normalization, transpose, steps/default, momentum/Nesterov, scale/decay 순서는 수식 변화다. state key/dtype/lazy init/counter는 checkpoint 변화다. backend selection, foreach/compile/collective는 dispatch 변화다. tolerance/fixture 추가·삭제는 보증 범위 변화다.

함수 이름과 line만 바뀐 경우 body digest/call graph로 의미 이동을 찾는다. common utility나 external dependency 변경도 포함한다. release note만 읽고 behavior 동일성을 선언하지 않는다.

각 semantic change에는 affected property/gradient tape, distributed/checkpoint와 long-run cell을 연결한다. 변경되지 않은 evidence는 invalidation key가 같을 때만 재사용한다.

**upgrade child recipe를 paired rehearsal로 승인한다.**

old/new code에 same W/G/state tape를 넣어 intermediate와 delta를 비교한다. 의도된 change와 unexpected first difference를 분리한다. state schema가 바뀌면 migration/reset branch를 시험한다.

performance는 same backend/hardware/shape histogram에서 다시 잰다. compiler cache를 분리한다. new version이 최신이라는 이유만으로 old production parent를 덮지 않는다.

**production canary에서 optimizer 이상을 조기에 찾는다**

**첫 수십 update를 높은 해상도로 관측한다.**

group/role별 gradient, momentum, NS input norm/residual, delta RMS/spectral, update/weight ratio와 non-finite를 매 update 표본화한다. common AdamW group과 baseline의 expected 범위를 비교한다. scheduler/scaler와 data batch를 같은 사건에 둔다.

owner imbalance, NS kernel time/workspace, collective bytes/tail, fallback와 graph break를 관측한다. 평균 step time 전에 p99와 memory high-water가 악화될 수 있다. checkpoint write/read와 recovery rehearsal도 canary 범위에 넣는다.

stop condition은 non-finite, residual/update budget 초과, unexpected backend, group coverage mismatch, partial state generation과 common-group drift다. validation score가 아직 정상이어도 중단한다.

**승격은 범위를 명시적으로 넓히는 과정이다.**

검증한 model layers/shapes, GPU, dtype, topology와 data phase에서 시작한다. 새로운 extreme aspect, expert/adapter 또는 world size가 나타나면 support cell을 추가 검증한다. 우연한 실행 성공을 자동 승계하지 않는다.

rollback은 model parameter와 optimizer/scheduler/scaler/data cursor를 같은 parent UpdateID로 되돌린다. candidate state만 제거하고 새 weight를 유지하지 않는다.

**12장의 종합 인수 기준**

**수학 계약이 닫혀야 한다.**

Muon의 momentum/Nesterov, normalization, Newton–Schulz coefficient/iteration, rectangular orientation, shape scale와 decay 순서가 고정 source와 FP64 oracle로 재현되는가. polar/dual-norm 직관의 적용 범위와 rank-deficient/저정밀 반례가 명시됐는가.

**parameter와 코드 계약이 닫혀야 한다.**

semantic role allowlist와 AdamW fallback group이 actual optimizer state와 일치하는가. KellerJordan/nanochat/PyTorch/NVIDIA source의 함수·wrapper·test 증명 범위를 구분했는가. requested/effective backend와 fallback이 관측되는가.

**분산·정밀도·복구 계약이 닫혀야 한다.**

full/owner/local geometry, TP/EP/FSDP owner와 communication bytes가 RecipeID에 있는가. CUDA/BF16 residual과 memory lifetime, checkpoint migration/reshard와 atomic next-update resume가 검증됐는가.

**비교와 운영 계약이 닫혀야 한다.**

AdamW baseline이 같은 tuning/resource 예산을 받았고 token/FLOPs/wall-clock/운영 추정값을 구분했는가. failure fixture, observability와 canary/rollback이 actual ParameterID와 UpdateID로 연결되는가.

이 네 계약이 직접 evidence를 가질 때만 특정 model·hardware·topology의 Muon recipe를 승인한다. 빈 칸은 명시된 미검증 범위다. 이 판정은 Muon이 보편적으로 우월하다는 선언이 아니라, 선택한 조건에서 수학·코드·시스템·복구 의미를 독립적으로 재현할 수 있다는 증명이다.

**NorMuon·equilibration 계열을 Muon의 별명으로 처리하지 않는다**

**추가 state와 변환을 식에서 찾는다.**

Muon 변형은 polar-like update 뒤 행/열 또는 parameter coordinate의 scale을 추가로 보정할 수 있다. 이름에 normalization이나 equilibration이 붙었다고 표준 Muon과 같은 state라고 가정하지 않는다. 논문과 fixed source에서 어떤 moving statistic, normalization 축, epsilon과 적용 순서를 쓰는지 확인한다.

행별 RMS, 열별 RMS 또는 adaptive second moment가 있으면 persistent state shape와 initialization, update cadence를 기록한다. polar transform 전인지 후인지에 따라 singular vector/scale 효과가 달라진다. AdamW graft와도 구분한다.

공개 구현이 없거나 fixed code에서 해당 변형을 찾지 못하면 paper-defined mechanism으로만 둔다. third-party reproduction에는 별 source card와 property test를 붙인다. 이름이 비슷한 config flag로 구현됐다고 추정하지 않는다.

**equilibration property를 scale 반례로 검증한다.**

G의 한 행 또는 열만 c배하고 update가 어떻게 변하는지 exact equation으로 예측한다. global scale invariance와 row/column rescaling invariance는 다른 성질이다. zero row/column, tiny statistic과 epsilon dominance를 포함한다.

stateful 변형은 첫 step만으로 판단하지 않고 여러 gradient tape에서 statistic과 delta를 추적한다. checkpoint 누락 또는 row/column swap mutation이 next delta에서 잡혀야 한다. 추가 state byte와 collective도 baseline Muon에 포함시키지 않는다.

**optimizer API의 `grad=None`·sparse·maximize 경계를 명시한다**

**수학 식에 없는 framework 상태를 step contract에 넣는다.**

parameter의 gradient가 None이면 state 생성/update와 decay를 건너뛰는지, zero tensor이면 momentum tail과 decay가 움직이는지 확인한다. frozen/unvisited expert와 conditional modality에서 두 상태가 자주 갈린다. optimizer group별 step counter가 전진하는지도 본다.

sparse gradient embedding은 dense matrix처럼 보이지만 coalescing, row visitation과 memory contract가 다르다. Muon이 sparse gradient를 지원하지 않으면 construction/step에서 명시적으로 거부하고 AdamW/Adagrad 계열 fallback에 배정한다. 암묵적 densify는 OOM과 함수 변화를 만든다.

maximize option이 있다면 gradient sign을 momentum 전에 뒤집는지 확인한다. closure/differentiable optimizer step, capturable tensor step와 foreach/fused option도 실제 implementation이 지원하는 범위만 기록한다.

**API mutation을 upstream/default 변화와 연결한다.**

`set_to_none=True` zeroing, accumulation과 unused parameter detection이 None/zero 비율을 바꾼다. trainer upgrade에서 optimizer 수학은 같아도 state trajectory가 달라질 수 있다. gradient state histogram을 UpdateID별로 기록한다.

fixture는 selected matrix가 None→nonzero→None, zero→nonzero 순서를 경험하게 한다. lazy momentum 생성, decay, counter와 checkpoint를 비교한다. rank마다 None 상태가 다르면 distributed collective 순서를 보존하거나 전역 participation mask를 합의한다.

**gradient clipping과 Muon의 합성 순서를 분리한다**

**raw gradient clip과 transformed update clip은 다른 algorithm이다.**

global norm clipping을 Muon 전에 적용하면 momentum에 들어가는 gradient와 polar 방향이 바뀐다. Muon 뒤 update를 clip하면 orthogonalized direction의 scale만 제한한다. per-parameter, per-group와 adaptive clipping도 서로 다르다.

trainer가 optimizer 밖에서 unscale→clip→step을 수행하는지, optimizer 내부 update clipping이 있는지 call graph로 잇는다. matrix group과 AdamW fallback이 같은 global clip denominator를 공유하는가도 확인한다.

outlier matrix 하나가 global norm을 지배하면 모든 group gradient가 줄지만 polar normalization이 다시 matrix 방향 크기를 평탄화할 수 있다. clip 효과가 직관과 다르게 상쇄될 수 있다. clip 전/후 gradient, NS input과 final delta를 기록한다.

**clip threshold sweep을 overflow와 분리한다.**

AMP unscale 전 clipping은 잘못된 threshold를 적용한다. non-finite detection과 clip의 순서를 확인한다. overflow로 step을 skip할 때 clipping statistic이나 momentum이 commit되지 않아야 한다.

fixture는 한 large singular outlier, 전체 scale, 한 parameter outlier와 non-finite를 사용한다. expected clip scale와 group delta를 FP64로 계산한다. 옵션 변경은 RecipeID와 scheduler/trial 비교를 요구한다.

**Newton–Schulz 반복의 국소 수렴 영역을 scalar map으로 본다**

**행렬 문제를 singular value별 scalar 변환으로 해석한다.**

orthogonally invariant polynomial iteration은 ideal arithmetic에서 각 singular value에 같은 scalar polynomial을 적용하는 관점으로 읽을 수 있다. initial scaling이 singular values를 어느 구간에 넣고 polynomial이 1 쪽으로 보내는지 plot/table로 본다.

fixed point 1 부근의 derivative, zero 부근 성장과 overshoot 구간을 coefficient별로 계산한다. 이것이 normalization 제거 시 divergence, 작은 singular direction의 느린 회복과 반복 수 효과를 직관적으로 설명한다. finite precision과 rank deficiency는 ideal scalar 분석에서 벗어날 수 있다.

coefficient를 인터넷 요약에서 복사하지 않고 fixed function literal로 계산한다. implementation multiplication order와 cast가 polynomial rounding을 바꾸므로 scalar high precision, matrix FP32/BF16을 단계로 나눈다.

**수렴 plot을 optimizer 품질 주장으로 확대하지 않는다.**

singular value가 1에 가까워지는 것은 polar approximation property다. 그 방향이 주어진 task에서 더 좋은 loss를 만든다는 결론은 별 closed-loop experiment가 필요하다. residual과 validation을 같은 축으로 그려도 인과를 단정하지 않는다.

matrix fixture는 prescribed singular values를 가진 UΣVᵀ를 만들어 scalar prediction과 iteration SVD를 비교한다. 반복별 rounding budget과 transpose path를 검증한다.

**matrix scale equivariance를 epsilon·momentum·decay까지 확장한다**

**순수 polar kernel과 전체 optimizer의 성질을 구분한다.**

exact polar는 양의 scalar c에 대해 `polar(cG)=polar(G)`이지만 normalization epsilon, finite iteration과 BF16은 작은 c에서 이를 깨뜨린다. momentum history가 있으면 current gradient만 scale해도 candidate 방향이 단순히 같지 않다. decay는 W에 의존한다.

kernel test는 G 전체를 `10^k` 배해 output direction/RMS를 본다. optimizer test는 gradient history 전체를 scale한 경우, 마지막 gradient만 scale한 경우와 W/decay를 포함한 경우를 나눈다. 어떤 invariance를 기대하는지 명시한다.

learning rate와 shape multiplier가 output scale을 다시 부여한다. “gradient 크기에 무관”이라는 설명은 update norm이 완전히 고정된다는 뜻이 아니며 rank/aspect/scale recipe가 남는다.

**scale failure를 loss scaler와 연결한다.**

AMP loss scale은 unscale 뒤 optimizer에 동일 logical gradient를 줘야 한다. unscale 누락 또는 double unscale은 polar kernel에서 방향이 우연히 비슷해 감춰질 수 있지만 epsilon/momentum/decay와 non-finite clock에서 드러난다. optimizer input digest와 scaler generation을 기록한다.

extreme scale fixture에서 direction만 비교하지 않고 momentum/state와 next checkpoint를 본다. found-inf skip이 모든 hybrid group을 멈추는지도 확인한다.

**optimizer collective와 backward reducer의 중복을 탐지한다**

**gradient가 어디에서 global 의미를 얻는지 한 번만 정의한다.**

DDP가 all-reduce한 gradient를 optimizer가 다시 reduce하면 scale 또는 bandwidth가 중복된다. 반대로 optimizer가 reduce-scatter를 소유하는데 DDP hook을 끄면 모든 parameter가 그 경로에 포함되는지 확인해야 한다. hybrid AdamW/Muon group이 서로 다른 reducer를 쓸 수 있다.

gradient lifecycle에는 local accumulation, unscale, clip, reduce/reduce-scatter, owner compute와 parameter-ready event가 포함된다. 각 tensor가 replicated, summed/averaged 또는 sharded인지 표기한다. framework의 averaging convention을 식 denominator에 반영한다.

communication trace에서 같은 ParameterID payload가 두 collective에 나타나는지, expected wire byte와 scale fixture로 검증한다. all-one rank-specific gradient는 sum/mean/double reduction을 선명하게 구분한다.

**overlap hook과 optimizer state의 수명을 맞춘다.**

backward bucket ready 즉시 optimizer computation을 시작하면 gradient accumulation/no_sync 경계와 충돌할 수 있다. bucket view가 다음 backward에서 재사용되기 전에 NS가 끝나야 한다. CUDA event/stream ownership을 기록한다.

rank failure 또는 overflow가 늦게 발견되면 이미 일부 owner momentum이 바뀌었을 수 있다. global validation 후 commit하거나 candidate state를 폐기할 수 있어야 한다.

**CUDA matmul 효율을 matrix shape histogram으로 예측한다**

**NS의 GEMM이 항상 큰 정방 GEMM이라는 가정을 버린다.**

attention/MLP weight는 크지만 TP shard, expert, adapter와 block partition은 작은 또는 extreme rectangular matrix를 만든다. NS Gram/polynomial matmul shape는 orientation에 따라 달라진다. target model ParameterID별 M/N/K histogram을 생성한다.

tile alignment, tensor-core dtype, leading dimension, noncontiguous copy와 launch overhead를 계산한다. small matrices를 batch/group하는 backend가 있는지, grouping이 coefficient/iteration state와 identity를 보존하는지 본다.

benchmark는 warm-up/compile, allocator, stream과 actual shape frequency를 고정한다. per-matrix microbenchmark를 전체 optimizer critical path로 확대하지 않는다. useful FLOPs와 padded/executed FLOPs를 분리한다.

**architecture 세대별 지원을 명시한다.**

CUDA toolkit/compiler와 GPU capability에 따라 BF16/FP8 matmul, kernel selection과 performance가 달라진다. 확인한 hardware만 support matrix에 넣는다. driver/library upgrade는 optimized performance/parity evidence를 stale로 만든다.

CPU 또는 eager fallback이 correctness를 유지해도 training SLO를 깨뜨릴 수 있다. effective backend와 fallback count를 metric으로 둔다.

**root·orthogonalization workspace OOM을 사전 계산한다**

**persistent state와 transient peak를 분리한다.**

Muon momentum은 matrix numel에 비례하지만 NS intermediate와 batched descriptors가 peak를 만든다. Shampoo는 factor/root와 decomposition workspace, SOAP은 basis rotation temporary가 추가된다. optimizer step이 backward activation과 겹치면 둘이 동시에 살아 있다.

ParameterID별 persistent bytes, candidate/temp, collective staging, compiler workspace와 allocator fragmentation을 시간축으로 합산한다. sequential owner compute가 temp를 재사용하는지 parallel compute가 여러 matrix를 동시에 살리는지 확인한다.

OOM 해결로 NS batch size/overlap/offload를 바꾸면 performance와 collective order가 달라진다. microbatch 감소는 training batch/scheduler에도 영향을 준다. 단순히 “optimizer memory” 숫자 하나를 쓰지 않는다.

**OOM failure에서 state 원자성을 검증한다.**

workspace allocation 전후, 일부 matrix state candidate 생성 뒤 OOM을 주입한다. parameter/momentum/cadence가 partial commit되지 않아야 한다. process를 재사용한다면 candidate buffer를 완전히 폐기하고 scaler/scheduler도 전진하지 않는다.

recovery는 same UpdateID를 재시도하고 uninterrupted delta와 비교한다. 자동 fallback은 새 backend RecipeID와 explicit degraded flag가 필요하다.

**checkpoint I/O와 restart 시간을 optimizer 선택에 포함한다**

**state byte가 저장 pause와 복구 SLO로 바뀌는 경로를 센다.**

Muon momentum, AdamW moments, Shampoo factor/root와 SOAP basis는 serialized bytes와 compression 가능성이 다르다. shard 수, filesystem/object-store bandwidth, checksum과 commit metadata를 포함해 실제 write/read 시간을 측정한다.

cached root/basis를 저장하지 않고 재계산하면 파일은 작아지지만 cold restart 계산과 first-step tail이 늘어난다. 재계산이 deterministic/tolerance 안인지, 어떤 Gram/state generation을 사용하는지 확인한다.

async checkpoint는 snapshot copy memory와 optimizer update concurrency를 관리해야 한다. state가 write 중 변하지 않게 generation/frozen view를 사용한다. file upload 완료 전에 root commit을 publish하지 않는다.

**장기 useful-updates/hour로 비교한다.**

steady step throughput에 checkpoint frequency×pause, expected fault rate×reload/rebuild와 compiler warmup을 더한다. 빠른 optimizer step이 큰 checkpoint로 운영 효율에서 질 수 있다. 실패 run과 recovery 실패도 포함한다.

restart rehearsal은 cadence boundary, world-size change와 corrupted shard에서 수행한다. exact/warm reset을 구분하고 rollback parent를 보존한다.

**hyperparameter 탐색의 선택 편향을 줄인다**

**탐색 공간과 예산을 optimizer family별로 사전 등록한다.**

AdamW에는 lr, β, epsilon, decay와 warmup, Muon에는 lr/momentum/Nesterov/NS steps/scale/decay/grouping, Shampoo/SOAP에는 damping/block/cadence/graft가 있다. 차원 수가 다르므로 동일 trial 수만으로 공정하다고 단정하기 어렵다.

총 accelerator hour, 실패 처리, early-stop와 seed allocation을 정한다. baseline에 충분한 tuning을 제공한다. candidate 개발 중 본 validation set을 최종 선택에 반복 사용하지 않는다.

multi-fidelity 탐색은 짧은 run에서 수치/안정성 gate를 거르고 장기 quality를 별 단계로 본다. 짧은 초기 loss가 장기 순위를 보존한다는 가정도 검증한다.

**winner 보고보다 response surface를 보존한다.**

안정 lr 구간, update ratio, residual, memory/throughput와 quality를 함께 기록한다. best 한 점만 공개하면 optimizer가 얼마나 민감한지 알 수 없다. 실패/중단 trial도 artifact로 남긴다.

selection 후 untouched evaluation과 canary에서 재확인한다. 결과는 target model/data/hardware scope로 제한한다.

**optimizer 효과를 loss 외의 표현 변화로 진단한다**

**layer별 update가 representation에 미치는 중간 효과를 본다.**

activation norm/covariance, attention entropy, MLP gate saturation, embedding drift와 logit scale을 optimizer branch 사이 비교한다. Muon의 matrix update geometry가 어느 layer에서 가장 큰 변화를 만드는지 update/activation을 연결한다.

이 지표는 품질의 대리변수이지 목적 자체가 아니다. 변화가 없다고 optimizer 효과가 없거나 변화가 크다고 개선이라고 단정하지 않는다. predeclared hypothesis와 downstream slice를 사용한다.

common AdamW group의 representation contribution과 candidate matrix group을 ablation으로 나눈다. parameter selection을 바꾼 run과 optimizer 식만 바꾼 run을 구분한다.

**기하 측정을 coordinate artifact와 연결한다.**

singular spectrum/cosine은 parameter representation과 sharding에 의존한다. logical matrix로 복원하고 fixed ParameterID를 사용한다. fused projection의 whole/submatrix 분석을 둘 다 보고 어떤 grouping이 실제 optimizer에 쓰였는지 명시한다.

checkpoint branch마다 same probe batch와 model mode를 사용한다. stochastic dropout과 data drift를 통제한다.

**장애 증상에서 optimizer 함수로 돌아가는 결정 트리**

**loss spike를 first bad delta로 좁힌다.**

input/data/loss denominator가 같으면 gradient finite/unscale/clip을 확인한다. matrix group의 momentum, NS input norm, iteration residual, scale/decay와 final delta에서 최초 차이를 찾는다. common AdamW group도 비교해 global clock/reduction 문제를 배제한다.

**throughput 저하를 shape·backend·owner로 좁힌다.**

ParameterID shape histogram과 effective backend/fallback, NS matmul trace, workspace allocation, owner queue와 collective tail을 본다. cadence/checkpoint event와 겹쳤는지 확인한다. 평균 kernel time 하나로 원인을 확정하지 않는다.

**resume drift를 state generation으로 좁힌다.**

RecipeID/group mapping, momentum/moments, step/cadence, scheduler/scaler/data cursor와 compiled cache를 비교한다. 첫 resumed gradient tape로 delta를 재생한다. state reset이면 expected warm branch와 비교한다.

**분산 hang을 ordinal과 identity로 좁힌다.**

rank별 collective sequence, group, payload ParameterID/order/count와 zero-gradient participation을 비교한다. timeout 뒤 in-memory partial state를 계속 사용하지 않는다. last durable commit에서 복구한다.

**독립 검토자가 재계산하는 optimizer certificate**

**첫 검토자는 source와 recipe만 받는다.**

한 selected matrix의 gradient→momentum/Nesterov→NS iteration→shape scale→decay→delta를 재구성한다. coefficient, cast, transpose와 backend를 source card에서 확인한다. expected state/temporary/collective를 계산한다.

**둘째 검토자는 checkpoint와 trace만 받는다.**

ParameterID별 optimizer owner, state shapes/dtypes, step/cadence, backend/collective와 group selection을 추론한다. 그 뒤 source/recipe와 맞춘다. unknown state나 unexpected fallback은 support gap이다.

**두 검토자는 같은 gradient tape로 만난다.**

FP64 oracle, actual eager/optimized, distributed reconstructed delta와 resume next delta를 비교한다. tolerance와 미검증 범위는 사전에 있다. final loss만 맞는 certificate는 불충분하다.

certificate에는 ModelRevision, RecipeID, ParameterID, UpdateID, topology/hardware, source/dependency digest, option, evidence와 failure terminal을 기록한다. 독립 재실행이 같은 승인/거부 결론을 내야 한다.

## 12.15 GaLore와 APOLLO를 저랭크 상태 기계로 비교한다

AdamW의 두 moment를 모두 원래 행렬 크기로 두기 어렵다고 해서 곧바로 LoRA처럼 weight update를 저랭크로 제한해야 하는 것은 아니다. GaLore는 **weight는 dense하게 학습하되 optimizer가 보는 gradient 좌표만 저랭크로 옮긴다**. `G_t∈R^{m×n}`에서 rank `r`의 직교 기저를 `P_t`라 하면 tall matrix의 한 경로는 `G'_t=G_tP_t^T`로 `[m,r]` gradient를 만들고, Adam류 moment를 이 작은 좌표에 쌓은 뒤 `U_t=U'_tP_t`로 되돌린다. LoRA의 `ΔW=BA`는 학습 가능한 weight 변화 자체의 rank를 제한하지만, GaLore는 매 step dense `W`를 갱신한다. 두 방법을 “저랭크 파인튜닝” 한 칸에 넣으면 checkpoint와 merge 의미부터 틀린다.

고정한 공식 GaLore commit `2cc66f8…4e9`의 `GaLoreProjector.project()`는 행렬의 tall/wide 여부와 `proj_type`에 따라 왼쪽 또는 오른쪽 singular vector를 선택한다. `update_proj_gap`의 배수에서만 `torch.linalg.svd`로 기저를 다시 만들고, `project_back()`이 `scale`을 곱해 원 좌표로 복귀시킨다. 따라서 옵션 하나씩의 소비자와 결과는 다음처럼 닫힌다.

| 옵션 | 소비 함수와 바뀌는 상태 | 직접 효과 | 첫 실패 검출기 |
|---|---|---|---|
| `rank` | `get_orthogonal_matrix()`가 보존할 singular vector 수 | moment 원소 수와 투영 오차를 함께 바꾼다 | `||G-project_back(project(G))/scale||/||G||`, state shape |
| `update_proj_gap` | `project()`의 `iter % gap` 분기와 `ortho_matrix` 수명 | SVD 빈도와 오래된 부분공간 오차를 교환한다 | refresh latency, basis age, refresh 전후 reconstruction |
| `proj_type` | tall/wide 분기와 좌·우·양쪽 투영 | 작은 state의 shape와 matmul 순서를 바꾼다 | rectangular fixture의 shape·inverse path |
| `scale` | `project_back()`의 최종 곱 | 실제 update RMS를 바꾸며 rank와 무관하다 | 동일 projected update의 delta ratio |
| layerwise mode | parameter별 post-accumulate hook과 optimizer/scheduler 사전 | gradient 수명을 줄이지만 accumulation·분산 경계를 바꾼다 | hook 호출 수, parameter별 step clock, resume replay |

여기서 가장 위험한 오해는 `rank`만 보고 peak memory를 예측하는 것이다. refresh step에는 full gradient, SVD workspace와 새 basis가 겹친다. steady-state moment bytes는 줄어도 refresh peak가 OOM을 낼 수 있다. 더구나 basis가 checkpoint에 없거나 복원 시점의 step clock이 달라지면 load는 성공해도 다음 refresh 위치부터 경로가 갈라진다. 작은 deterministic fixture는 tall `[8,3]`, wide `[3,8]`, rank-one, repeated singular value를 포함하고 `project→project_back`, refresh 경계 전후, save/load 뒤 다음 두 update를 비교해야 한다.

APOLLO는 “GaLore에서 SVD만 더 싸게 만든 것”으로 요약하면 부족하다. 핵심 질문은 full-rank update의 크기를 어떤 통계로 추정하고 저랭크 좌표에서 얻은 update를 어떤 scale로 되돌리느냐다. LLaMA-Factory commit `a18110d…b4d`에서 `create_custom_optimizer()`는 `use_galore`와 `use_apollo`를 서로 다른 생성기로 보낸다. parser는 두 옵션을 동시에 허용하지 않고, layerwise 모드는 distributed training에서, 두 optimizer는 DeepSpeed와 함께 쓸 때 거부한다. mixed precision에서 메모리가 커질 수 있다는 경고도 있다. 즉 CLI flag는 optimizer 이름만 바꾸는 토글이 아니라 **parameter group·hook 소유권·분산 가능 조합·dtype state**를 바꾸는 schema migration이다.

실무 승인표에는 resolved parameter ID, 각 tensor의 logical/local shape, rank와 scale, basis 또는 projection state digest, parameter별 scheduler clock을 넣는다. 오류는 세 층으로 분리한다. projection residual이 먼저 갈리면 기저·shape 문제, projected moment까지 같고 reconstructed update가 갈리면 scale 문제, single GPU는 같고 distributed에서만 갈리면 hook과 gradient reduction 순서 문제다. 이 순서는 18장의 adapter parameter 선택과 30장의 recipe compiler 검증으로 이어진다.

### BitFit과 저랭크 optimizer를 같은 절감법으로 묶지 않는다

BitFit은 bias만 `requires_grad=True`로 두는 **parameter 선택 정책**이다. GaLore/APOLLO는 선택된 dense parameter의 optimizer 좌표와 state를 바꾸는 **update 변환 정책**이다. BitFit의 상태 절감은 trainable set이 작아져 생기며, 저랭크 projector state는 존재하지 않는다. 따라서 BitFit fixture는 이름 문자열이 아니라 parameter object ID와 storage alias를 기준으로 trainable manifest를 만들고, backward 뒤 허용 bias에는 finite nonzero gradient가, 나머지 parameter에는 gradient와 delta가 없음을 검사한다. LayerNorm bias, attention projection bias가 없는 Llama 계열처럼 architecture가 바뀌면 같은 `bias` 규칙의 용량도 달라진다.

두 정책을 결합할 수 있다는 사실도 유용성을 뜻하지 않는다. bias vector는 행렬 projection의 정의역이 아니므로 GaLore group에 들어가지 않고 일반 optimizer 경로를 탄다. “BitFit+GaLore”라는 recipe가 실제로는 BitFit parameter만 남겨 projector를 한 번도 호출하지 않을 수 있다. 이를 찾는 가장 싼 검사는 optimizer별 resolved group cardinality와 projector 호출 수다. 10장의 model autopsy가 bias inventory를 공급하고, 18장의 PEFT 비교가 같은 token budget에서 학습 가능한 자유도와 artifact 형식을 분리한다.

## 12.16 scheduler로 넘길 optimizer 시간축과 인수 조건을 닫는다

optimizer step, skipped update, state refresh와 schedule clock을 구분해 다음 장의 학습 시간축으로 넘긴다.

**하나의 global step으로 모든 clock을 압축하지 않는다.**

microbatch accumulation, attempted optimizer step, AMP-skipped step, committed UpdateID, Muon momentum update, Shampoo root/SOAP basis cadence, scheduler와 checkpoint generation이 있다. 각 owner와 전진 조건을 적는다.

Muon momentum/NS와 AdamW moments는 같은 model commit에 참여한다. root/basis/curvature가 별 cadence면 사용한 dependency generation을 delta에 연결한다. failed attempt가 cadence만 전진하지 않게 한다.

13장의 scheduler는 committed update 또는 global token 중 명시된 clock을 읽는다. optimizer group multiplier/shape scale와 effective update ratio를 함께 넘긴다. resume에서 scheduler 한 step ahead를 load 전에 거부한다.

**handoff artifact를 고정 gradient로 검산한다.**

checkpoint를 load하고 next lr/cadence branch, momentum/input transform과 delta를 재계산한다. actual source function이 manifest clock을 소비하는지 확인한다. field가 저장됐지만 무시되면 실패다.

이 시간축이 닫혀야 optimizer 비교가 scheduler와 장기 training에서도 같은 recipe를 뜻한다. 이후 dtype, topology, backend 또는 parameter selection이 달라지면 child RecipeID와 새 handoff를 만든다.

### Muon과 quantized training state의 경계를 정한다

weight가 FP8·INT8·INT4 형식으로 저장되거나 QAT fake quantization을 거쳐도 optimizer가 읽고 쓰는 logical parameter와 master weight를 구분해야 한다. low-bit packed bytes에 직접 Muon update를 적용한다고 가정하지 않는다. 실제 training backend가 FP16/BF16/FP32 master를 갱신한 뒤 requantize하는지 source에서 확인한다.

quantization scale·zero point와 amax history가 trainable 또는 mutable state라면 optimizer/checkpoint owner를 기록한다. matrix gradient가 dequantized logical weight 기준인지, scale parameter가 AdamW fallback인지 분리한다. straight-through estimator가 있으면 backward gradient 정의를 명시한다.

fixture는 동일 master W/G에서 full-precision Muon delta, quantize-before/after path와 next forward를 비교한다. saturation, scale group boundary와 transposed packed layout을 포함한다. polar residual이 정상이어도 requantization이 delta를 지울 수 있으므로 effective applied delta를 측정한다.

저정밀 checkpoint migration은 master weight, momentum, quantization metadata와 RecipeID를 함께 옮긴다. inference-only quantized artifact에서 training을 재개하는 경우 missing master/optimizer state를 exact resume로 부르지 않는다. 14장의 저정밀 계약과 30장의 export lineage에 연결한다.

### data-parallel gradient denominator가 matrix geometry에 미치는 영향을 확인한다

rank별 valid token 수가 다를 때 local mean gradient를 단순 평균하면 global token mean과 다르다. Muon polar가 global scalar scale에 둔감할 수 있어 denominator 오류가 delta 방향에서 일부 감춰질 수 있지만, momentum history·epsilon·clipping과 AdamW fallback group에서는 그대로 드러난다.

정확한 global numerator/denominator를 먼저 만들고 optimizer에 logical gradient를 넘긴다. packed variable length, zero-valid-token rank와 gradient accumulation을 포함한다. reducer가 sum/mean 중 무엇을 내고 trainer가 world size/token count를 어떻게 적용하는지 기록한다.

fixture는 rank A 1 token, rank B 3 token과 서로 다른 matrix gradient를 사용한다. global concatenation reference, rank-local mean 평균과 올바른 weighted reduction의 direction/spectrum을 비교한다. scalar만 다른 경우와 방향까지 다른 경우를 모두 만든다.

observability는 global valid tokens, gradient reduction convention과 pre-Muon digest를 같은 UpdateID에 둔다. optimizer 후보의 scale invariance를 잘못된 objective denominator를 용인하는 근거로 사용하지 않는다. 6장의 packing과 13장의 token clock을 잇는다.

### model architecture 변화가 optimizer allowlist를 깨뜨리는 순간을 찾는다

모델 upgrade에서 dense MLP가 MoE로, separate QKV가 fused projection으로, 일반 residual이 mHC로 바뀌면 기존 parameter regex가 여전히 실행되더라도 semantic role이 달라진다. optimizer config checksum만 같다고 selection을 승계하지 않는다.

old/new module tree와 semantic ParameterID 집합을 diff한다. added/removed/renamed/fused/split, tied alias와 global shape/sharding 변화마다 expected owner를 검토한다. unknown parameter는 기본 Muon이 아니라 admission failure 또는 explicit AdamW quarantine으로 둔다.

Qwen/DeepSeek/Gemma/GLM 같은 family label이 grouping rule이 아니다. 10장의 model dossier에서 actual projection/expert/router/modality/mixing role을 받는다. adapter injection 뒤에도 allowlist를 다시 생성한다.

selection diff는 state migration과 lr/decay policy를 동반한다. old Muon state를 새 semantic matrix에 shape만 맞춰 복사하지 않는다. role-coded gradient fixture와 checkpoint next step으로 mapping을 검증한다. model architecture와 optimizer recipe에는 공동 child generation을 부여한다.

**12장을 실제 선택 능력으로 닫는 질문**

독자는 먼저 “이 parameter가 2차원인가”가 아니라 “어떤 logical map이며 gradient가 어느 좌표·빈도로 만들어지고 어떤 shard가 소유하는가”를 묻는다. 그 답이 matrix optimizer의 정의역, global/local geometry와 state 비용을 정한다.

그다음 “Muon이 더 좋은가”가 아니라 “고정 gradient에서 momentum·Newton–Schulz·scale·decay가 어떤 delta를 만들고 AdamW 기준과 무엇이 다른가”를 묻는다. FP64 oracle, fixed source와 runtime trace가 같은 답을 내야 한다.

마지막으로 “한 step이 빨랐는가”가 아니라 “동일 token·compute·wall-clock·운영 예산에서 품질, 실패율, checkpoint·복구를 포함해 어떤 trade-off가 있는가”를 묻는다. 지원하지 않은 topology와 dtype은 성공으로 세지 않는다.

세 질문을 ParameterID·RecipeID·UpdateID로 연결하면 최신 optimizer를 유행어가 아닌 검증 가능한 시스템 선택으로 바꿀 수 있다. 이후 새로운 직교화, preconditioner 또는 optimizer가 등장해도 같은 수학 oracle, state transition, 분산 ownership, failure와 공정 비교 틀로 해부한다.

**최종 회귀에서 의도적 오류를 다시 통과시킨다**

설명과 구현이 함께 변하면 정상 fixture만 통과해도 detector가 무뎌졌을 수 있다. 배포 직전에는 normalization 제거, transpose 반전, NS coefficient 변경, shape scale의 local/global 혼동, momentum parameter swap과 hybrid group 중복을 다시 주입한다. 각 오류가 예정된 수학·grouping·state gate에서 실패하는지 확인한다.

분산 반례는 all-rank ParameterID 순서 하나를 바꾸고 owner rank를 cadence 경계에서 종료하며 AdamW group만 한 step 앞선 checkpoint를 만든다. collective hang, identity digest와 commit closure 가운데 기대한 detector가 작동해야 한다. 단순 timeout만 발생하면 원인 분류가 충분하지 않다.

저정밀 반례는 BF16 cast를 normalization 전후로 이동하고 tiny singular direction, extreme scale와 noncontiguous rectangular matrix를 사용한다. reference residual, update cosine과 next quadratic loss가 사전 threshold를 넘는지 본다. optimized backend가 silent eager fallback하면 성능 gate가 별도로 실패해야 한다.

공정 비교 반례에서는 AdamW baseline의 trial budget 또는 data order를 일부러 다르게 만들어 experiment validator가 거부하는지 확인한다. 실패 run을 결과 집계에서 제거하는 mutation도 잡는다. benchmark 표의 숫자가 계산 가능해도 estimand 조건이 다르면 승인하지 않는다.

마지막 certificate는 정상 시험뿐 아니라 이 negative control의 mutation ID, expected first detector, observed terminal과 복구 parent를 담는다. reviewer가 한 mutation을 골라 source 줄, affected state, trace와 checkpoint까지 왕복할 수 있어야 한다. detector가 잡지 못한 오류는 지원 범위의 미검증 cell로 되돌리고, 문구로 안전하다고 봉합하지 않는다.

이 회귀의 핵심은 특정 구현을 영구히 옳다고 선언하는 것이 아니다. source revision, compiler, GPU, topology, model parameter inventory 또는 training recipe가 바뀌면 어떤 증거가 무효가 되는지 즉시 알 수 있게 하는 것이다. 변경된 cell만 다시 검사하되, 함수 정의와 state schema가 달라졌다면 downstream 성능과 복구 증거까지 함께 갱신한다.

따라서 최종 승인표에는 성공한 경로뿐 아니라 지원하지 않는 sparse gradient, 미검증 quantized state, 시험하지 않은 world size와 backend도 남는다. 독자는 이 표에서 자신의 조건과 겹치는 범위를 확인하고, 빈 영역에는 같은 FP64 oracle·gradient tape·failure injection을 적용한다. 이 경계가 명확해야 Muon의 수학적 매력과 실제 장기 훈련의 운영 위험을 동시에 정직하게 판단할 수 있다.
## 12.17 Muon 한 스텝을 숫자와 소스 사이에서 끝까지 걷는다

앞 절까지는 선택지를 넓게 비교했다. 이제 Keller Muon의 고정 revision `f98f1cacc0263b04290753e32be8d498c1efc806`에서 `muon_update`와 `zeropower_via_newtonschulz5`를 한 줄씩 따라간다. 목적은 특정 optimizer를 추천하는 것이 아니다. 손으로 재현 가능한 한 스텝을 만들어 **어느 tensor에서 처음 갈라졌는지** 말할 수 있게 하는 것이다. 이 경로의 입력은 hidden weight와 같은 2차원 parameter의 gradient `G[m,n]`, 영속 state인 momentum buffer `M[m,n]`, 그리고 `beta`, `ns_steps`다. AdamW의 `exp_avg[m,n]`, `exp_avg_sq[m,n]`, scalar step과 달리 이 원형 Muon에는 행렬과 같은 shape의 momentum 하나만 있다.

### 함수 다섯 줄이 정하는 실제 알고리즘

핵심 부분은 짧지만 순서는 바꿀 수 없다. 아래는 원본의 34–40행을 이해에 필요한 범위만 인용한 것이다.

```python
momentum.lerp_(grad, 1 - beta)
update = grad.lerp_(momentum, beta) if nesterov else momentum
if update.ndim == 4:
    update = update.view(len(update), -1)
update = zeropower_via_newtonschulz5(update, steps=ns_steps)
update *= max(1, update.size(-2) / update.size(-1))**0.5
```

첫 줄은 (M_t=\beta M_{t-1}+(1-\beta)G_t)다. 둘째 줄의 PyTorch `grad.lerp_(momentum, beta)`는 (U_t=(1-\beta)G_t+\beta M_t)를 `grad` storage에 제자리 기록한다. 따라서 “Nesterov를 켰다”는 말에는 raw gradient tensor가 이 시점부터 덮어써진다는 aliasing 계약도 들어 있다. gradient를 뒤에서 logging하거나 다른 optimizer가 공유한다면 함수 진입 전 snapshot이 필요하다. `nesterov=False`에서는 `U_t=M_t`이므로 직교화 입력부터 달라진다.

4차원 convolution weight `[out,in,kh,kw]`는 `[out,in·kh·kw]`로 펴지만 3차원 expert stack은 이 함수에서 같은 방식으로 펴지지 않는다. 그 다음 `zeropower_via_newtonschulz5`는 tall matrix이면 transpose하고, BF16으로 cast한 뒤 Frobenius norm과 `1e-7`로 정규화한다. 반복 안에서는 (A=XX^\top), (B=bA+cA^2), (X\leftarrow aX+BX)를 다섯 번 수행한다. 즉

\[
X_{k+1}=aX_k+bX_kX_k^\top X_k+c(X_kX_k^\top)^2X_k,
\qquad(a,b,c)=(3.4445,-4.7750,2.0315).
\]

SVD가 (U\Sigma V^\top)라면 이상적인 polar oracle은 (UV^\top)지만, 원본 주석은 이 aggressive quintic이 모든 singular value를 정확히 1로 수렴시키는 반복은 아니라고 선을 긋는다. 따라서 `ns_steps=5` 결과를 `UVᵀ`와 원소별 동일하다고 검사하면 구현 의도를 잘못 시험한다. 마지막 aspect-ratio factor (sqrt{\max(1,m/n)})도 polar 수학 자체가 아니라 layer shape 사이의 update scale을 보정하는 별 단계다.

### FP64 2×2 fixture로 최초 분기를 고정한다

실제 함수는 BF16을 사용하지만, 먼저 rounding을 제거한 FP64 대수 oracle을 만든다. 초기 momentum은 0, (G=\operatorname{diag}(3,1)), `beta=0.95`, `nesterov=True`, `ns_steps=5`, learning rate는 0.02로 둔다. 아래 코드는 구현의 행렬식과 같은 계산을 독립적인 scalar polynomial로 쓴 fixture다. 학습을 실행하는 코드가 아니라 한 스텝 산술 검산이다.

```python
g = [3.0, 1.0]                       # diagonal entries of G[2,2]
m = [0.05 * z for z in g]            # M1 = .95 M0 + .05 G
u = [0.05*g[i] + 0.95*m[i] for i in range(2)]
x = [z / (sqrt(sum(q*q for q in u)) + 1e-7) for z in u]
for _ in range(5):
    x = [3.4445*z - 4.7750*z**3 + 2.0315*z**5 for z in x]
delta = [-0.02*z for z in x]
```

상태 열은 `M₁=(0.15,0.05)`, Nesterov 입력 열은 `U₁=(0.2925,0.0975)`, 정규화 열은 `X₀=(0.9486830,0.3162277)`이다. 다섯 번 뒤 diagonal은 `(0.7530334,1.1337062)`, 작은 차원 Gram residual (\lVert X^\top X-I\rVert_F)은 약 `0.5184861`, parameter delta는 `(-0.0150607,-0.0226741)`다. 첫 축이 더 큰 raw gradient였는데 최종 보폭은 둘째 축이 더 크다. 이것이 singular direction의 크기를 다시 배분한다는 말의 구체적인 뜻이며, residual이 0이 아니라는 사실도 aggressive coefficient의 의도와 맞는다.

같은 (G)에 초기 상태 AdamW, `beta1=0.9`, `beta2=0.95`, `eps=1e-8`, `lr=0.02`를 쓰면 bias correction 뒤 두 diagonal delta는 거의 `(-0.02,-0.02)`다. 두 결과가 다르다는 사실보다 중요한 것은 **왜** 다른가다. AdamW는 각 좌표의 (g/\sqrt{g^2})를 만들고, Muon은 행렬 singular basis에서 polynomial을 적용한다. 행렬을 회전시키면 AdamW의 elementwise 좌표와 Muon의 singular direction 관계가 달라진다. 따라서 learning rate 숫자를 같게 한 이 fixture는 동등 성능 비교가 아니라 식별용 대조군이다.

변형은 한 축씩만 바꾼다. `nesterov=False`면 최초 불일치는 `U₁`이다. `ns_steps`나 coefficient가 다르면 `X₁`이 처음 갈라진다. tall matrix의 transpose guard가 뒤집히면 정규화 전 shape와 최종 복원 shape에서 잡힌다. aspect scale을 빼면 직교화 출력은 같고 `delta`에서 처음 갈라진다. AdamW와 비교할 때는 state manifest가 이미 `exp_avg_sq` 존재 여부에서 갈라지므로, 결과 delta만 보고 “커널 오차”라고 부르지 않는다.

### epsilon·dtype·직교화 실패를 순서대로 진단한다

같은 diagonal fixture를 (10^{-8})배하고 Keller의 `1e-7`을 유지한 FP64 모사에서는 다섯 번 뒤 `(0.7401648,1.0603685)`가 된다. scale 1의 `(0.7530334,1.1337062)`와 방향이 달라진다. `eps`는 단순한 zero-division 보험이 아니라 `norm ≲ eps` 영역에서 effective input scale을 바꾸는 알고리즘 상태다. 반대로 NVIDIA Emerging-Optimizers의 고정 revision은 `newton_schulz` 입력을 FP32 2D/3D로 제한하고 기본 `eps=1e-15`를 쓰며, `normalize_in_double=True`에서는 exact zero를 일부러 guard하지 않는다. 두 구현을 같은 “Muon”으로 묶으면 zero·tiny-norm terminal부터 다르다.

진단은 다음 열을 순서대로 비교한다. `(1)` 함수 진입 `G`의 logical parameter ID·shape·dtype·digest, `(2)` 갱신 뒤 `M`, `(3)` Nesterov 뒤 `U`, `(4)` transpose flag와 정규화 norm·eps, `(5)` 매 반복 `X_k`의 finite·Gram residual·digest, `(6)` shape scale 뒤 update RMS, `(7)` decay와 learning rate를 적용한 parameter delta다.

FP64 oracle과 FP32 구현이 `X₀`부터 다르면 norm·epsilon·transpose 문제다. FP32끼리는 같고 BF16만 `X₁`부터 다르면 cast와 matmul precision 경로를 본다. 모든 `X_k`가 같은데 parameter만 다르면 shape scale, decay 순서, learning-rate owner 또는 stale parameter snapshot 문제다.

NVIDIA의 upstream test `test_muon_utils.py:212–262`는 condition number (10^{12})인 SVD 구성에서 Polar Express와 quintic을 비교하고, 16-step 결과의 작은 차원 Gram이 identity에 가까운지 확인한다. `:322–348`은 1D·4D 입력, FP64 입력, custom coefficient 누락과 잘못된 coefficient 이름이 오류가 되는지 검사한다. 이 근거는 coefficient dispatch·shape·dtype guard를 지지하지만 Keller의 BF16 다섯-step 장기 수렴, `muon_update`의 momentum 순서, distributed all-gather, optimizer checkpoint resume를 증명하지 않는다. 그 네 항목은 별 fixture와 통합 시험이 필요하다.

이 walkthrough는 11장의 AdamW state 전이, 13장의 learning-rate clock, 14장의 BF16 matmul·정밀도 정책, 16장의 optimizer-state checkpoint, 24장의 collective 관측성과 이어진다. 30books의 실제 `paper:cs231n`과 `term:cs231n--gradient-descent`는 gradient descent의 좌표를 제공하지만 Muon이나 Newton–Schulz를 다루지 않는다. 그러므로 그 연결은 기초 개념으로 내려가는 다리이지, 현대 matrix optimizer 구현의 근거를 대신하지 않는다.

이 장의 결론은 Muon이라는 이름이 아니라 **변환 입력, 행렬 반복, shape scale, state owner가 만드는 한 update를 재생할 수 있는가**에 달려 있다. 손계산 oracle과 실제 branch가 처음 갈라지는 열을 찾았다면 수치 오류와 분산·checkpoint 오류를 구분할 수 있다. 다음 13장에서는 이 update의 보폭을 움직이는 learning-rate schedule을 `step`이라는 모호한 정수가 아니라 token·attempt·commit 시계로 다시 정의한다.

## Optimizer를 state tensor와 update 식으로 비교한다

Muon의 PyTorch 구현은 2D gradient에 momentum buffer를 유지하고 선택적으로 Nesterov 조합을 만든다. update를 BF16으로 복사해 Frobenius norm의 eps-clamped 값으로 정규화한 뒤 Newton–Schulz 다항 반복을 적용한다. 이는 정확한 SVD의 `UVᵀ`를 항상 계산하는 구현이 아니다. 마지막에는 parameter에 decoupled weight decay를 곱하고 shape에 따라 조정된 learning rate로 orthogonalized update를 더한다. zero matrix에서는 eps가 0 나눗셈을 막지만 정보 없는 방향을 만들어 주지는 않는다.

Lion은 momentum state 하나와 sign update를, Adafactor는 matrix의 second moment를 행·열 factor로 줄이는 경로를 가질 수 있다. LAMB와 LARS는 parameter norm과 update/gradient norm의 비율을 layerwise scale에 넣지만 zero 또는 near-zero norm fallback이 구현마다 다르다. Sophia는 gradient EMA와 Hessian-diagonal 추정의 clipped ratio를, Shampoo는 축별 Kronecker statistic과 inverse root를 사용한다. 이름 대신 epsilon 위치, bias correction, decay 결합 방식, state dtype·shape와 parameter group rule을 한 행씩 적는다.

matrix fixture는 rank-deficient와 condition number가 큰 gradient covariance를 만들고 inverse-root residual을 FP64 eigendecomposition/SVD reference와 비교한다. BF16 statistic, 작은 eigenvalue와 damping을 바꿔 NaN·방향 폭주를 검사한다. grafting은 preconditioned direction과 SGD/Adam 계열 update의 어떤 norm을 결합하는지 분리한다. 방향과 크기를 모두 원본 optimizer에서 가져왔다고 쓰면 grafting의 의미를 잃는다.

distributed Shampoo는 factor·inverse-root owner와 parameter global range, 통신 뒤 replica equality를 checkpoint manifest에 기록한다. accumulation partition과 world size를 바꾼 restore에서 첫 update를 비교한다. flatten order나 parameter group이 달라졌는데 canonical parameter ID와 명시적 adapter가 없으면 migration을 거부한다. 논문의 scale benchmark와 hardware throughput은 별 RuntimeUnverified 실험이며 작은 수치 fixture의 통과로 대신하지 않는다.
