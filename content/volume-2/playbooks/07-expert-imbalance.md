# Playbook 07. expert imbalance

## 실행 순서

### routing 관측
1. layer/expert별 routed token, probability mass, capacity overflow/drop을 수집한다.
2. auxiliary loss와 실제 token histogram을 함께 본다.
3. rank별 all-to-all bytes와 expert compute time을 비교한다.
4. 고정 batch에서 router logits, top-k index, dispatch permutation을 저장한다.

## 분기

### 판정
- probability mass는 고른데 token만 쏠리면 top-k, tie-breaking, capacity를 조사한다. token과 compute가 함께 쏠리면 router를, token은 고른데 실행 시간만 다르면 expert shape, kernel, placement를 조사한다.

### 균형의 단위를 혼동하지 않는다

expert별 token count, router probability mass, gradient mass, FLOP, kernel time, all-to-all byte는 다른 양이다. top-k 선택 전 probability mass가 고르더라도 tie-breaking·capacity·group restriction 때문에 선택 token은 쏠릴 수 있다. token 수가 같아도 sequence position, hidden sparsity, expert implementation, GPU clock·topology가 다르면 실행 시간은 다르다. 따라서 “expert가 균형”이라는 표현 대신 어느 층·창·rank 범위의 무엇이 고른지 적는다.

각 MoE layer에서 token histogram의 mean, max, p95, coefficient of variation, Gini, zero-token expert 수, dropped/rerouted token을 집계한다. auxiliary loss는 scalar만 남기지 말고 그 loss가 사용한 정확한 probability·assignment 통계와 denominator를 남긴다. layer 평균은 특정 layer의 collapse를 숨기므로 layer×expert×domain×position slice를 유지한다.

### 비교 가능한 관측 창을 먼저 만든다

microbatch 하나의 histogram은 표본 잡음이 크다. 유효 routed token 수가 같은 창, 같은 layer schedule과 동일한 top-k·capacity 정책에서 비교한다. padding과 masked token이 router 통계에 들어가는지 확인하고, gradient accumulation을 가로지르는 집계가 중복되지 않는지 본다. world size가 달라졌다면 local count 평균이 아니라 global numerator와 count로 통계를 다시 만든다.

평균 load와 최대 load를 함께 본다. expert 수가 많으면 일부 zero-token expert는 작은 batch에서 자연스럽다. 반면 장시간 창에서도 같은 expert가 0이면 router score, group restriction, expert mask와 checkpoint column mapping을 의심한다. max/mean만으로 분포를 다 알 수 없으므로 entropy·Gini와 상위 expert 누적 비율을 함께 둔다.

조건부 histogram은 개인정보나 원문을 metric label로 넣지 않는다. 언어·도메인·길이·데이터 소스 같은 고카디널리티 slice는 안전한 offline artifact로 계산하고 production metric에는 coarse bucket과 revision만 둔다. 특정 specialization이 예측 품질을 높이는 정상 현상인지, capacity overflow와 undertraining을 만드는 실패인지 validation slice로 판정한다.

### routing 결정을 한 token씩 재구성한다

고정 GoldenBatch에서 router input, logits, softmax/top-k 전 score, selected expert ID, routing weight, capacity position, drop/reroute flag, dispatch permutation, combine index를 저장한다. global expert ID와 rank-local expert ID 대응을 별도로 고정한다. 특히 expert parallel rank 수가 바뀔 때 modulo 변환이 다른 expert weight를 가리키지 않는지 본다.

값이 완전히 같은 logits로 tie를 만들어 안정적이고 결정적인 tie-breaking인지 검사한다. batch 순서를 permutation했을 때 token의 expert 선택이 불필요하게 바뀌면 sorting stability·capacity order가 sample order에 종속된다. identity-scaled expert fixture—expert `e`가 입력을 `e+1`배하는 작은 구조—로 dispatch·combine permutation을 역산하면 shape-compatible 오류를 잘 잡는다.

### score·selection·controller를 분리한다

router score와 selection bias가 분리된 모델은 두 값을 따로 저장한다. bias가 후보 선택에만 쓰이고 mixture weight에는 쓰이지 않는지 source에서 확인한다. auxiliary-loss-free balancing처럼 gradient 바깥의 controller가 bias를 갱신한다면 update rule, count reduction group, step counter와 checkpoint state를 포함한다. resume 후 bias가 초기화되면 model weight가 같아도 route가 달라진다.

top-k 경계의 margin도 기록한다. top-k 안의 마지막 score와 첫 탈락 score가 거의 같으면 BF16/FP32 cast, tie policy나 작은 parameter 변화로 assignment가 크게 바뀔 수 있다. entropy가 정상이어도 margin이 계속 작으면 route instability가 높다. 같은 input에서 dtype과 backend를 바꾼 counterfactual로 selected set과 output 민감도를 본다.

main loss에서 router로 흐르는 gradient와 auxiliary·z-loss에서 흐르는 gradient를 분리한다. selected weight가 detach되거나 combine backward가 weight gradient를 잃으면 main task는 expert만 학습시키고 router는 보조 목적만 따를 수 있다. 고정 top-k 영역에서 finite difference를 하고 selection boundary의 비미분성은 별도 decision test로 다룬다.

### router 학습과 system imbalance를 나눈다

같은 batch·checkpoint에서 router logits이 이전 정상 run과 다르면 router parameter, normalization, precision, data mixture, auxiliary/z-loss 계수를 본다. logits·assignment는 같은데 all-to-all byte·time이 다르면 dispatch layout, padding/alignment, token packing, process group, topology와 network을 본다. communication은 같지만 expert GEMM time이 다르면 token count당 kernel occupancy, grouped-GEMM descriptor, zero-token expert 처리, clock·thermal state를 본다.

단일 host에서는 정상인데 멀티노드에서만 쏠리면 expert placement가 NVLink/NVSwitch 안과 NIC 밖의 다른 경로를 타는지 본다. token 균형 loss를 높여도 케이블·NIC·process-group 구성은 고쳐지지 않는다. 반대로 placement를 바꾴도 router collapse는 남는다.

### 통신 대기와 expert 계산을 시간축에서 가른다

rank별 시간을 dispatch wait, network transfer, local reorder, expert GEMM, combine wait로 쪼갠다. 느린 rank가 compute 때문에 늦어 다른 rank의 collective wait를 늘리는지, network 자체가 먼저 느린지 최초 차이를 찾는다. collective 총 시간만 보면 원인 rank와 피해 rank가 뒤바뀔 수 있다.

send/receive matrix에서 remote fraction과 최대 rank-pair byte를 본다. 평균 byte가 같아도 한 NIC rail이나 특정 pair에 몰리면 tail이 커진다. expert placement 변경 전후에는 logical route가 같은지 확인한다. route까지 달라졌다면 system-only 실험이 아니라 model function도 바뀐 것이다.

**Megatron routing test를 imbalance 원인 분리기로 읽는다**

Megatron-LM 고정 checkout의 `tests/unit_tests/inference/test_moe_permute.py:105-151`은 `compute_local_tokens_per_expert` 결과를 PyTorch reference와 비교한다. local expert에 들어온 routing pair 총수와 count 합이 같은지도 검사한다.

`tests/unit_tests/transformer/moe/test_token_dispatcher.py:390-416`은 permutation 뒤 `sum(tokens_per_expert)==permuted_input.shape[0]`을 확인하며 alignment를 켰을 때 expert count가 16의 배수인지도 고정한다. 이 oracle은 count·permutation·padding 계약을 좁게 증명한다. router가 좋은 specialization을 배웠는지, all-to-all이 빠른지, capacity drop이 공정한지는 증명하지 않는다.

competing hypothesis를 router·capacity·dispatcher·expert·fabric의 다섯 상태로 나눈다.

1. router 가설: 같은 hidden에서 logits·top-k ID부터 달라진다. 고정 checkpoint·batch의 router forward가 최소 판별식이다.
2. capacity 가설: candidate ID는 같고 accepted/drop·capacity position에서 처음 갈린다. capacity를 무한대로 둔 대조군과 비교한다.
3. dispatcher 가설: accepted assignment는 같은데 permutation, peer split 또는 unpermute 뒤 TokenID가 달라진다. identity-scaled expert가 최소 oracle이다.
4. expert/kernel 가설: assignment와 bytes는 같은데 expert별 output·gradient 또는 GEMM time이 갈린다. 동일 M·N·K의 local expert 대조군을 쓴다.
5. fabric 가설: logical send matrix는 같은데 특정 rank pair의 enqueue→complete만 늦다. intra-node와 inter-node를 같은 payload로 비교한다.

고정 fixture는 token 여덟 개, expert 네 개, top-2로 만든다. 각 token에는 고유 basis 값을 주고 expert `e`는 입력을 `e+1`배한다. 정상군에서 candidate routing pair는 16개다. capacity 변형은 expert 0에 동점 score를 몰아 candidate 16개를 유지하면서 accepted/drop만 바꾼다. permutation 변형은 count를 그대로 두고 두 TokenID의 reverse index를 바꾼다. kernel 변형은 expert 2만 같은 shape의 느린 대조 함수로 바꾼다. 이때 첫 차이는 각각 `accepted map`, `unpermuted output`, `expert completion time`이어야 한다. 모두를 “skew가 증가했다”로만 기록하면 실패다.

pass/fail은 평균 balance loss로 정하지 않는다. count oracle은 candidate·accepted·padded slot의 합과 범위를, function oracle은 원 token 순서의 output·gradient를, system oracle은 peer byte matrix와 expert별 노출 시간을 따로 통과해야 한다. auxiliary loss가 내려가도 drop된 domain token이 늘면 model gate는 실패다. token count가 균등해도 한 expert kernel만 늦으면 router coefficient를 바꾸지 않는다.

복구할 때 capacity factor나 router bias를 즉석에서 바꾸면 objective와 accepted population이 달라진다. 부분 gradient와 controller count를 버리고 새 config digest의 child run으로 시작한다. checkpoint에는 global ExpertID↔rank mapping, router/controller state, capacity policy와 optimizer slot을 함께 복원한다. 첫 update에서 selected IDs, accepted/drop, dispatch permutation, expert별 gradient와 parameter delta를 정상 대조군과 비교한 뒤에만 장기 run을 재개한다. 공개 unit test를 대규모 EP all-to-all·dropless routing·실제 fabric 성능의 증명으로 과장하지 않는 것이 이 플레이북의 음성 근거다.

## 통제 실험과 복구

### 가설 하나만 바꾼다

router auxiliary coefficient, capacity factor, top-k, expert group, dispatch backend, expert placement, microbatch composition을 한꺼번에 바꾸지 않는다. 각 실험은 token histogram, dropped mass, router entropy·gradient, all-to-all byte/time, expert GEMM time, step tail latency, validation loss·domain metric을 동시에 본다. 코드가 옵션을 소비하는 함수와 실제 변경된 state를 로그에 남긴다.

capacity를 늘려 drop이 사라져도 all-to-all·activation 메모리와 tail latency가 커질 수 있다. auxiliary coefficient를 높여 histogram이 고르게 되어도 specialization·품질을 훼손할 수 있다. top-k를 줄이면 compute는 줄지만 redundancy와 gradient 경로가 바뀐다. 따라서 수정은 load 숫자 하나가 아니라 품질–메모리–통신–tail 비용의 사전 budget을 만족해야 한다.

### 변경마다 보존해야 할 상태를 적는다

capacity 실험에서는 offered, accepted, dropped assignment 보존식을 검산한다. drop된 token이 residual, shared expert, zero output 중 어느 경로를 타는지 확인한다. capacity가 커져 accepted set이 바뀌면 output과 gradient가 달라지므로 throughput 비교 전에 새 함수 revision으로 평가한다.

placement 실험에서는 global expert ID와 parameter·optimizer moment를 그대로 유지하고 owner rank만 바꾼다. 같은 GoldenBatch의 logits, selected IDs와 logical output은 같아야 한다. 달라지면 reshard나 router-column mapping 오류다. logical parity를 통과한 뒤 all-to-all matrix와 tail을 비교한다.

dispatch backend를 바꾸면 eager oracle과 token별 output·input gradient·expert weight gradient를 비교한다. padding slot, zero-token expert, skewed assignment와 non-contiguous input을 포함한다. 빠른 backend가 capacity를 암묵적으로 적용하거나 weight dtype을 다르게 cast하지 않는지 본다.

### 지역 조치와 rollback

극단적 쏠림이 OOM·hang을 만들면 문제 batch·checkpoint를 보존하고 자동 resume를 멈춘다. 안전한 capacity/reroute fallback으로 사고를 격리할 수 있지만, dropped token 정책·loss 분모·gradient가 바뀌므로 새 recipe generation으로 기록한다. router state가 이미 collapse된 checkpoint를 계속 학습하는 것보다 마지막 정상 CheckpointID로 rollback해 통제 실험을 적용한다.

영구 수정은 layer×expert×rank metric, route atlas fixture, capacity/drop invariant, placement manifest와 failure injection을 regression suite에 넣는다. domain mixture 변경이 정상적 specialization을 만든 것을 사고로 잘못 판정하지 않도록 domain-conditioned baseline과 장기 분포를 보존한다.

### 종료 전에 복구 동일성을 확인한다

마지막 정상 checkpoint로 rollback했다면 router·expert weight뿐 아니라 optimizer moments, balance bias/controller, auxiliary schedule, RNG, data cursor와 expert placement generation을 복원한다. weight checksum만 같고 balance state가 다르면 첫 batch부터 route가 갈린다. 고정 probe의 logits·top-k·accepted IDs와 첫 optimizer delta를 비교한다.

world size나 expert-parallel 크기를 바꿔 복구했다면 동일 layout의 exact resume로 볼 수 없다. global expert ID별 parameter와 moment, router column을 새 owner에 reshard하고 logical routing fixture를 통과시킨다. 새 placement의 performance와 장기 route 분포를 별도로 승인한다.

failure injection에는 한 expert 지연, 한 rank의 zero-token 입력, correction bias 초기화, expert ID swap, dispatch permutation, capacity overflow와 network rail 저하를 각각 넣는다. 기대한 최초 detector가 어느 것인지 미리 쓴다. 모든 실패가 auxiliary loss 상승으로 나타날 것이라 가정하지 않는다.

## 종료 조건

### 통과
품질 회귀 없이 p95 expert load와 rank step-time spread가 사전 budget 안에 들어온다.

종료 증거는 문제 batch의 token별 route, 계층별 균형 통계, rank별 communication/compute 시간축, 지지·기각된 가설, 수정 구성·commit과 CheckpointID를 포함한다. 같은 GoldenBatch의 logits→top-k→dispatch→expert output→combine가 허용 오차 안에서 재현되고, 반례 fixture가 잘못된 expert ID·permutation·capacity를 예상한 gate에서 검출해야 한다. 장기 run에서는 품질과 route 분포가 모두 안정되고, incident 시에 다음 교대가 IncidentID만으로 첫 불균형 layer·expert·rank와 rollback 조건을 찾을 수 있어야 한다.
