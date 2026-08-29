# 28장 단일 GPU golden run: 복잡성을 늘리기 전 기준선을 만든다

멀티노드에서 처음 드러난 NaN도 원인은 token mask나 한 projection처럼 단일 batch에서 이미 재현되는 오류일 때가 많다. 곧바로 분산 계층부터 조사하면 통신 로그가 원래의 첫 차이를 가린다. 단일 GPU golden run은 작은 점수를 얻기 위한 축소 학습이 아니라, 입력에서 optimizer delta까지 정상 상태를 고정해 이후 TP·DP·kernel 확장에서 최초로 달라진 지점을 찾는 기준선이다.

이 기준선은 사건의 순서를 따라 세운다. 먼저 source·환경·데이터 bytes를 artifact로 고정하고, 같은 row가 같은 token·mask·label을 만드는지 확인한다. 그 batch가 embedding과 각 block을 지나 logits와 loss numerator·denominator를 만들면, backward에서 gradient ownership과 finite 상태를 검산하고 optimizer가 moment와 parameter를 정확히 한 번 갱신했는지 확인한다.

이어 그 상태를 원자적 checkpoint로 저장해 fresh process에서 다음 batch까지 복원한다. 이 eager 기준선이 닫힌 뒤에야 compile·fusion·저정밀 같은 optimized candidate를 비교한다. 차이가 생기면 `artifact→data→forward→loss→backward→optimizer→checkpoint/resume`를 역으로 추측하지 않고 앞에서부터 최초로 갈라진 경계를 찾는다.

## 28.1 fixture와 실행 환경을 동결한다

### 28.1.1 RunID를 만드는 입력

code commit, container digest, CUDA/driver, GPU UUID와 compute capability, model/config/tokenizer digest, dataset shard, seed를 manifest에 쓴다. golden batch는 `input_ids`, shifted labels, attention mask, 유효 label 수, 각 tensor dtype/shape/stride/checksum을 기록한다.

### 28.1.2 실행 전 fail-fast

token ID 범위, special token, embedding/head vocabulary, mask-label 정렬, sequence max를 CPU에서 검사한다. 한 batch의 decode round-trip과 chat template bytes를 저장한다. 이 단계가 실패하면 GPU를 쓰지 않는다.

### 28.1.3 seed의 범위

Python, NumPy, framework CPU/GPU generator, dataloader worker, augmentation의 seed를 구분한다. seed 숫자가 같아도 호출 순서와 kernel determinism이 다르면 결과가 다르다. RNG state checksum과 첫 draw fixture를 checkpoint에 둔다.

golden batch를 고르는 법. 가장 쉬운 batch 하나보다 padding, EOS, 긴/짧은 response, special token을 포함하되 사람이 검산 가능한 작은 batch를 고른다. 모델 성능을 대표하는 표본이 아니라 contract branch를 덮는 fixture다. 민감 데이터는 synthetic으로 대체한다.

## 28.2 forward에서 optimizer delta까지 수치 oracle을 세운다

### 28.2.1 관찰 지점을 정한다

embedding 후, 첫/중간/마지막 block, logits, unreduced token loss의 checksum과 finite 비율을 기록한다. backward에서는 parameter group별 gradient norm, zero/nonfinite count를 기록한다. AMP 순서는 backward→unscale→nonfinite check→clip→step→scheduler다.

### 28.2.2 optimizer invariant를 세운다

한 step 전후 parameter delta를 계산한다. frozen parameter delta는 0, trainable parameter는 예상 group LR과 weight decay를 따라야 한다. accumulation `K`회와 큰 batch 1회의 gradient를 허용 오차 내 비교해 loss denominator 오류를 찾는다.

### 28.2.3 first-divergence atlas를 만든다

각 관찰 지점에는 module path, call index, tensor role, shape/dtype/stride, finite count, norm, checksum을 기록한다. 전체 activation을 항상 저장하지 않고 문제 batch에서만 제한적으로 보존한다. 두 run을 비교할 때 첫 checksum divergence 이전의 원인을 찾는다.

finite-difference로 spot check한다. 작은 parameter 몇 개에 대해 `(L(θ+h)-L(θ-h))/(2h)`와 autograd를 비교한다. 저정밀/비결정적 kernel을 끄고 FP64/FP32 toy path에서 수행한다. 전체 모델 gradient 증명은 아니지만 label shift와 detach 오류를 잡는 교육 fixture다.

## 28.3 checkpoint를 fresh-process resume로 증명한다

### 28.3.1 저장할 상태를 식별한다

model, optimizer, scaler, scheduler, RNG, sampler cursor, consumed sample/token, config와 parent checkpoint를 저장한다. completion marker는 모든 shard/file hash가 확정된 뒤 쓴다.

### 28.3.2 중단 없는 실행과 재개 실행을 비교한다

N step uninterrupted run과 K step 저장+resume run을 비교한다. resume 직후 batch ID, LR, RNG draw, loss, parameter checksum을 본다. bitwise가 요구되지 않으면 허용 오차와 비교 대상을 미리 정한다.

### 28.3.3 checkpoint commit을 원자화한다

temporary generation에 files를 쓰고 size/hash manifest를 만든 뒤 completion marker를 publish한다. reader는 complete generation만 선택한다. single-file save도 write 도중 process가 죽을 수 있다. latest filename 덮어쓰기보다 generation과 parent를 사용한다.

resume 결정 트리를 실행한다. first batch가 다르면 sampler cursor, LR이 다르면 scheduler clock, loss만 다르면 RNG/dropout/kernel, parameter는 같은데 optimizer state가 다르면 load coverage를 본다. final metric이 비슷하다는 이유로 drift를 성공으로 판정하지 않는다.

## 28.4 평가·manifest·출시 관문를 연결한다

### 28.4.1 작은 private fixture를 분리한다

훈련 row와 겹치지 않는 작은 eval을 고정한다. raw response, normalized answer, contribution index와 denominator를 남긴다. 점수 하나보다 golden row별 diff가 회귀를 빨리 찾는다.

### 28.4.2 실행 상태를 정직하게 표기한다

이 장의 명령과 판정은 동반 lab에 있다. 공개 upstream test로 확인한 것, 로컬에서 실행한 것, 실행 예정인 것을 구분한다. 장비 실행 로그가 없으면 성능 결과를 쓰지 않는다.

### 28.4.3 source와 upstream test를 고정한다

Transformers trainer, PyTorch optimizer/AMP, OLMo-core·TorchTitan checkpoint 경로의 고정 revision을 읽는다. upstream test가 state-dict roundtrip을 확인해도 우리의 tokenizer·sampler·custom callback을 포함하지 않을 수 있다. golden lab은 이 gap을 manifest로 닫는다.

단일 GPU 출시 관문를 판정한다. preflight, forward finite, gradient/parameter invariant, checkpoint resume, small eval, 산출물 hash가 모두 통과해야 멀티노드로 간다. 성능이 느리다는 이유로 correctness gate를 생략하지 않는다. failure bundle과 rollback checkpoint도 산출물이다.

이 장이 넘기는 것. `GoldenBatchID`, baseline activation/gradient/parameter checksum, 단일 GPU `CheckpointID`, `EvalID`와 signed manifest를 29장에 넘긴다.

Golden invariant 목록. golden run은 점수 하나가 아니라 단계별 invariant의 묶음이다. 입력에는 row ID, rendered bytes, token IDs, label과 loss mask checksum이 있다. forward에는 주요 layer activation의 shape·dtype·finite 비율·norm과 selected element가 있다. backward에는 parameter별 gradient presence, norm, clipping 전후와 global norm이 있다. update에는 optimizer step, scheduler LR, moment와 parameter delta checksum이 있다.

모든 tensor를 원문 그대로 저장하지 않는다. 작은 공개 fixture는 exact tensor를, 실제 batch는 통계와 keyed digest를 남긴다. 수치 비교에는 exact, absolute/relative tolerance, distributional invariant를 구분한다. dtype과 kernel이 바뀌는데 bitwise equality를 강제하거나, exact가 가능한 token IDs를 느슨한 tolerance로 비교하지 않는다.

invariant에는 예상 실패 지점과 owner가 있다. label mask 합이 0이면 training을 즉시 중단하고, activation norm drift는 진단 alert를 낼 수 있다. parameter delta가 0인 trainable adapter는 hard failure지만 frozen base의 delta 0은 정상이다. ownership 표가 해석을 결정한다.

환경을 bytes로 고정한다. RunID에는 source commit, uncommitted patch digest, config canonical hash, container와 wheel digest, driver/CUDA runtime, GPU UUID/SKU, loaded native library와 environment allowlist가 들어간다. package 이름과 version만으로 locally rebuilt extension을 구분할 수 없다. startup log에서 실제 shared object path와 digest를 수집한다.

deterministic flag, TF32, matmul precision, autocast dtype, fused/compiled mode와 CUDA architecture를 기록한다. 환경 변수 전체를 dump하면 secret이 새므로 결과에 영향을 주는 allowlist만 보존한다. clock이나 thermal state는 exact 수치보다 성능 baseline 해석을 위한 관측으로 둔다.

실행 전 schema validation으로 vocab/tokenizer, model config와 checkpoint tensor shape를 확인한다. GPU capability가 wheel의 kernel target과 맞는지, free disk와 checkpoint atomic rename 조건, golden fixture digest가 맞는지 검사한다. fail-fast가 training 첫 step 이후의 모호한 오류보다 싸다.

Golden batch 설계. batch는 가장 쉬운 평균 사례만 고르지 않는다. 최소 길이와 긴 context, padding 없는 row와 많은 padding, special token, multi-turn template, response mask 경계, 드문 vocabulary와 empty-like edge를 작은 묶음에 넣는다. 각 row의 목적을 기록해 실패 시 어떤 계약이 깨졌는지 알 수 있게 한다.

data collator 전 row와 collated tensor를 모두 fixture로 둔다. padding side, pad-to-multiple, truncation, position IDs와 attention mask를 손으로 검산한다. packed sequence라면 document boundary와 loss isolation을 확인한다. 같은 token count라도 packing이 다르면 attention과 gradient가 다르다.

golden batch는 training data에서 무작위로 하나 뽑은 비밀 표본이 아니다. 재배포 가능한 synthetic 또는 허가된 fixture를 우선한다. 실제 분포의 길이·언어·modality bucket parity는 별도 sampled validation으로 보완한다.

Forward hook의 안전한 사용. 모든 layer output을 장기 보존하면 메모리와 실행 순서를 바꿀 수 있다. 대표 embedding, attention projection, residual, MLP와 logits에 짧은 hook을 두고 detach한 통계만 수집한다. hook이 graph reference를 붙잡지 않는지 두 step memory invariant로 확인한다.

첫 divergence 탐색은 coarse checkpoint에서 시작한다. logits가 다르면 마지막 정상 layer와 첫 비정상 layer 사이를 binary search한다. shape/dtype, input digest, parameter digest와 output statistic을 한 행에 둔다. stochastic layer는 eval/train mode와 RNG counter를 확인한다.

compile이나 fused kernel에서 hook이 graph break를 만들 수 있으므로 eager diagnostic run과 optimized run을 구분한다. eager에서 수학 invariant를 확정한 뒤 optimized output tolerance와 performance를 비교한다. diagnostic 결과를 production throughput처럼 보고하지 않는다.

Loss를 손계산한다. 두세 token의 synthetic logits와 label로 log-softmax와 negative log likelihood를 손계산한다. ignore index와 response mask를 적용한 numerator와 contributing token count를 명시한다. framework가 mean을 내는 축과 label shift 위치를 source 및 fixture test에서 확인한다.

distributed 이전 단일 GPU에서 per-row loss, batch token-sum과 mean 관계를 고정한다. label smoothing, class weight, auxiliary router loss가 있으면 component별 numerator와 weight를 분리한다. total scalar가 같아도 component가 상쇄될 수 있다.

loss가 맞으면 logits의 selected vocabulary slice와 hidden-to-head projection을 비교한다. tied embedding 여부, dtype cast와 vocabulary resize가 결과에 미치는 영향을 검사한다. tokenizer special token 추가 뒤 head shape만 맞고 mapping이 틀리는 오류를 golden token으로 잡는다.

Backward와 optimizer invariant. trainable parameter 목록과 expected gradient 목록을 시작 전에 비교한다. LoRA라면 adapter 행렬에 gradient가 있고 frozen base에는 없어야 한다. checkpointing으로 일부 activation이 재계산돼도 최종 gradient는 tolerance 안에서 baseline과 맞아야 한다. unused parameter는 이유 없는 silent 상태로 두지 않는다.

optimizer 전에는 unscaled gradient와 global norm, clipping coefficient를 저장한다. AMP overflow가 있으면 parameter·moment·scheduler가 모두 어떤 상태로 남는지 contract를 시험한다. zero gradient를 `None`과 tensor 0으로 처리하는 차이도 optimizer state 생성에 영향을 줄 수 있다.

update 뒤 `delta = parameter_after-parameter_before`를 계산해 학습률 방향, weight decay와 moment 수식을 작은 scalar fixture에서 검산한다. scheduler step이 update 전인지 후인지 첫 두 step LR로 확인한다. gradient accumulation에서는 microstep 사이 parameter가 변하지 않는다는 invariant를 둔다.

Finite difference의 범위. finite difference는 모든 거대 parameter를 검사하지 않고 작은 float64 toy module과 selected scalar에 적용한다. `f(x+ε)-f(x-ε)`를 `2ε`로 나눈 값과 autograd를 비교하며 ε를 여러 크기로 바꿔 truncation과 round-off 영역을 본다. non-smooth activation 경계와 stochastic operation은 피하거나 조건을 고정한다.

mixed precision 모델의 gradient가 finite difference와 다르다고 즉시 버그라 하지 않는다. reference는 단순화한 고정 dtype 경로에서 수식과 mask를 검증하고, production dtype은 tolerance와 loss-scaling invariant로 본다. custom autograd나 fused loss가 들어간 지점을 우선 spot check한다.

검사 실패 시 input/parameter를 최소 fixture로 줄여 upstream issue나 regression test에 넣는다. 전체 training script만 제공하면 최초 오류를 재현하기 어렵다. 손계산·autograd·finite difference 세 증거가 맞물리면 수식 경계의 신뢰도가 높아진다.

Checkpoint에 포함할 state. model과 optimizer tensor 외에 scheduler, gradient scaler, global/micro step, tokens seen, sampler cursor, data epoch, Python/NumPy/framework/CUDA RNG, accumulation 위치와 callback state를 저장한다. iterable dataset의 cursor를 복원할 수 없다면 sample-exact resume를 주장하지 않는다. compile cache는 재생성 가능 artifact인지 state인지 구분한다.

checkpoint는 temporary generation에 모든 shard와 manifest를 쓴 뒤 hash를 검증하고 completion marker를 원자적으로 publish한다. 단일 GPU도 disk full, process kill과 partial write를 주입한다. loader는 marker 없는 최신 directory를 건너뛰고 마지막 완전 generation을 선택해야 한다.

optimizer를 빼고 weights-only resume하면 새 실험 branch다. 이전 RunID를 이어 쓰지 않고 parent checkpoint와 변경 사유를 남긴다. scheduler만 복원하지 않은 경우도 학습률 궤적이 달라지므로 동일 resume가 아니다.

중단·재개의 등가성 실험. 연속 N step run과 K step 저장 후 재개해 N까지 간 run을 같은 golden batch sequence로 비교한다. step마다 sample ID, LR, loss numerator/denominator, parameter와 optimizer checksum을 맞춘다. 최초 divergence가 K+1 입력이면 sampler/RNG, gradient면 scaler/accumulation, update면 optimizer/scheduler state를 본다.

exact kernel을 요구할 수 없는 환경에서는 tolerance를 사전에 정하고 여러 seed에서 최종 metric만이 아니라 step trace를 비교한다. 재개 직후 한 step의 작은 차이가 뒤에 증폭될 수 있다. “최종 loss가 비슷함”만으로 state-restorable을 주장하지 않는다.

checkpoint 직전과 data fetch 뒤, backward 중간, optimizer update 뒤 강제 종료를 각각 시험한다. 어느 상태에서 재시작 가능한지 명시하고 불완전 microstep은 처음부터 재실행한다. side-effect log와 W&B step 중복도 확인한다.

성능 golden baseline. 정확성 invariant가 통과한 뒤 warm-up을 제외한 steady window에서 tokens/s, step time, data wait, kernel time과 peak memory를 잰다. batch token 분포와 clock/power state를 기록한다. 첫 compile과 cache warm-up을 평균에 섞지 않는다.

회귀 threshold는 단일 실행 변동보다 커야 하며 반복 run의 median과 dispersion으로 정한다. profiler를 켠 수치와 끈 수치를 섞지 않는다. 성능이 좋아져도 activation·gradient·parameter invariant가 깨지면 승인하지 않는다.

baseline은 hardware, dtype, sequence bucket과 feature flag별로 버전 관리한다. 새로운 compiler나 CUDA library로 바꿀 때 정확성 parity와 성능을 함께 측정한다. 빠른 이유를 kernel fusion, data overlap, memory 감소 같은 trace 증거로 연결한다.

소스/시험 좌표. 사용 중인 Transformers Trainer와 모델 `forward`, data collator, loss shift·mask 경로를 고정 commit의 `path:symbol`로 기록한다. PyTorch optimizer·GradScaler·checkpoint와 RNG API, PEFT adapter injection/merge의 source와 upstream fixture를 연결한다. 문서의 개념 설명과 현재 설치 revision의 실제 default가 다를 수 있다.

소스 기록에는 호출자→callee, config field가 소비되는 branch, 반환 tensor와 side effect를 적는다. upstream test가 보장하는 dtype·shape·resume 사례와 우리 golden fixture가 추가로 보장하는 내용을 분리한다. line number만 두면 revision 변경에 약하므로 commit, path, symbol과 짧은 semantic anchor를 함께 쓴다.

실행 로그에는 이 소스 좌표 목록의 digest를 넣는다. dependency upgrade 뒤 symbol diff를 먼저 보고 영향 invariant를 선택한다. 검토하지 않은 custom model/optimizer 경로는 golden 승인을 상속하지 않는다.

단일 GPU 결정 트리. startup 실패면 artifact/schema/device capability를 먼저 본다. token fixture가 다르면 tokenizer/template/collator에서 중단한다. forward 최초 divergence면 input·weight digest 후 layer binary search, backward만 다르면 mask·autograd·scaler, update만 다르면 clipping·optimizer·scheduler 순으로 좁힌다.

연속 run은 맞고 resume만 다르면 checkpoint state와 data/RNG cursor를 본다. 정확성은 맞고 느리면 warm-up, data wait, CPU launch, kernel과 memory pressure 순으로 trace한다. OOM이면 static footprint와 stepwise retention을 분리한다.

모든 branch는 “설정 변경 후 통과”가 아니라 원 fault를 다시 넣었을 때 regression test가 실패하는 것으로 닫는다. 결정 기록에는 배제한 가설, 최초 divergence, fix commit, 새 golden digest와 잔여 tolerance를 남긴다.

Golden run 완료 조건. 완료하려면 깨끗한 환경에서 manifest만으로 실행을 시작하고 golden batch의 token·mask·loss를 손검산할 수 있어야 한다. forward, gradient, optimizer delta가 선언한 tolerance 안이며 trainable ownership과 finite 상태 assertion을 통과해야 한다. 연속/재개 run이 요구 등급으로 같아야 한다.

checkpoint partial-write와 process-kill 주입에서 마지막 완전 generation으로 복구되고, 평가 fixture와 성능 baseline이 재현돼야 한다. 소스 좌표와 실행 증거가 연결되고 미실행 변형은 명시돼야 한다. 이 기준선이 있어야 29장의 분산 차이를 framework 복잡성 탓으로 숨기지 않을 수 있다.

## 28.5 RNG·모델·데이터 경계를 같은 기준선에 묶는다

RNG를 계층별로 다룬다. seed 하나는 Python, NumPy, framework CPU/GPU generator, data worker, sampler, dropout과 sampling RNG의 전체 state가 아니다. generator별 state와 소비 순서를 기록한다. worker 수나 prefetch가 바뀌면 같은 seed라도 augmentation 순서가 달라질 수 있다.

golden test는 동일 process 재실행, checkpoint resume와 fresh process load를 분리한다. dropout mask digest와 batch order를 몇 step 기록해 최초 RNG divergence를 찾는다. deterministic mode가 지원하지 않는 operation은 목록과 수치 등급을 명시한다.

RNG state를 저장한 뒤 debug logging이 random 값을 소비하지 않는지도 시험한다. 평가 sampling과 training generator를 분리해 평가 횟수가 학습 궤적을 바꾸지 않게 한다. 새 callback 추가 뒤 golden sequence를 재검증한다.

모델 아키텍처 invariant. embedding에서는 token ID 범위, padding row 처리, tied output head와 vocabulary resize를 확인한다. attention에서는 q/k/v shape, head 수와 grouped-query mapping, mask의 허용/차단 위치, position encoding과 KV dtype을 작은 sequence로 검사한다. MLP에서는 gate/up/down shape와 activation 순서를 소스 심볼에 연결한다.

residual과 normalization은 pre/post 위치, epsilon, accumulation dtype을 고정한다. auxiliary expert/router가 있으면 token assignment count, capacity drop, load-balance loss와 total loss weight를 별도 invariant로 둔다. multimodal model은 projector output length와 modality placeholder mapping을 추가한다.

전체 logits가 다를 때 이 경계 통계를 순서대로 비교한다. architecture별 optional branch를 일반 golden에서 생략하지 않고 해당 config가 실제 켜졌는지 assertion한다. 이름이 같은 model family라도 revision이 다르면 invariant set을 상속하기 전에 source diff를 본다.

Tokenizer와 template fixture. 유니코드 정규화, 앞뒤 공백, newline, code block, emoji, 한국어 조사, special-token 문자열과 multi-turn tool message를 포함한 golden strings를 만든다. raw bytes, normalized text, token IDs, decode round trip과 offset mapping을 저장한다. decode가 원문과 완전히 같아야 하는 경우와 normalization으로 달라도 되는 경우를 구분한다.

chat template에서는 system/user/assistant/tool role span, generation prompt와 BOS/EOS 개수를 확인한다. response-only mask boundary를 token offset과 눈으로 대조한다. template가 tokenizer config와 model card 중 어디서 load됐는지 digest로 남긴다.

tokenizer upgrade 뒤 vocabulary size만 같아도 ID mapping이 달라질 수 있다. 모든 golden token ID를 exact 비교하고 embedding/head artifact compatibility를 검사한다. mismatch를 자동 remap하지 않는다.

데이터 순서 invariant. map-style dataset은 revision, split, sampler permutation과 batch index를 기록한다. distributed 전에도 worker scheduling 때문에 transform RNG가 달라질 수 있다. `num_workers=0` golden을 기준으로 worker mode의 output digest를 비교한다.

streaming dataset은 source shard order, byte offset, shuffle buffer state와 retry를 남긴다. remote read 실패로 row를 건너뛰면 skip reason과 denominator가 필요하다. exact resume를 지원하지 않으면 그 한계를 명시하고 작은 local fixture에서만 sample-exact를 요구한다.

curriculum이나 weighted sampling은 기대 비율이 아니라 실제 drawn family ledger를 본다. 첫 N step의 sequence를 golden으로 고정하고 장기 분포에는 interval을 사용한다. data pipeline 최적화가 row order와 packing을 바꾸면 새 baseline이다.

### 28.5.1 accumulation과 scheduler 시계를 맞춘다

global tokens/update를 같게 둔 큰 batch 한 step과 작은 microbatch accumulation을 비교한다. loss reduction이 token mean인지 microbatch mean인지에 따라 gradient가 달라질 수 있다. 각 microbatch numerator/denominator를 합친 뒤 한 번 나누는 reference를 손계산한다.

no-sync 경계, gradient scaling과 clipping 시점을 확인한다. clipping을 microstep마다 하는 것과 accumulation 끝에 하는 것은 다른 알고리즘이다. scheduler는 optimizer가 실제 commit된 횟수 또는 tokens seen 중 선언한 축으로만 진행한다.

마지막 불완전 accumulation window를 drop, scale 보정 또는 commit할지 정한다. AMP overflow로 update가 skip됐을 때 scheduler도 멈추는지 fixture로 검사한다. log step과 optimizer step의 혼동을 막는다.

### 28.5.2 compile·fusion parity를 분리한다

eager reference를 통과한 뒤 compile을 켜고 graph break, recompilation count와 generated kernel cache key를 기록한다. dynamic shape bucket별 첫 compile 비용과 steady 성능을 분리한다. fallback eager가 섞였는데 compiled라고 보고하지 않는다.

fused attention, fused optimizer와 custom loss는 각각 하나씩 켜며 output·gradient·delta parity를 확인한다. 허용 오차는 dtype과 reduction order를 반영하되 NaN·mask leakage·parameter ownership은 exact invariant다. kernel 선택 log와 loaded extension digest를 남긴다.

shape 하나만 검증하지 않는다. 최소/대표/경계 sequence와 odd batch를 포함한다. 빠른 path가 alignment 조건에서만 켜진다면 branch 양쪽의 fixture를 둔다. upgrade 뒤 source와 kernel cache를 무효화하고 재검증한다.

### 28.5.3 optimizer별 fixture를 둔다

Adam 계열은 작은 parameter와 두 gradient step으로 first/second moment, bias correction, epsilon, decoupled weight decay를 손계산한다. fused/foreach 구현이 같은 hyperparameter group과 step counter를 쓰는지 비교한다. capturable 옵션이나 tensor LR이 branch를 바꾸면 별도 fixture다.

Muon 등 matrix-aware optimizer는 어느 ndim/parameter group에 적용되는지, orthogonalization 반복과 fallback을 source에서 확인한다. embedding·norm·bias가 Adam group으로 가는지 ownership 표를 만든다. optimizer 이름 하나로 모든 parameter가 같은 update를 받는다고 쓰지 않는다.

8-bit optimizer는 quantized state block, scale와 outlier 처리, minimum-size fallback을 기록한다. full-precision toy reference와 production tolerance를 구분한다. checkpoint save/load 뒤 state dtype과 step이 보존되는지 확인한다.

golden artifact layout을 고정한다. 한 directory에는 canonical config, environment/소스 명세서, fixture input, expected token/mask, activation-gradient-update invariant, checkpoint manifest, evaluation ledger와 performance summary를 둔다. 큰 tensor는 content-addressed store를 가리키고 index에 digest와 schema를 둔다.

expected 값을 update하는 command는 변화 diff와 승인자를 요구한다. test가 실패할 때 자동으로 새 baseline을 덮어쓰지 않는다. architecture·dtype·feature 조합별 baseline을 분리하되 공통 fixture lineage를 연결한다.

artifact에는 재생 command와 성공/실패 판정 code revision을 넣는다. 사람이 chart를 보고 통과시키는 항목을 최소화한다. private fixture는 접근 제한 store에 두고 공개 synthetic package로 code path를 재현한다.

실행 playbook을 사건 순서로 쓴다. 첫 단계는 artifact·environment fail-fast와 tokenizer/data fixture다. 둘째는 forward-only exact/tolerance 비교, 셋째는 backward와 한 optimizer update, 넷째는 짧은 연속 run과 checkpoint resume다. 다섯째에 평가와 profiler 없는 성능 baseline을 잰다.

실패하면 다음 단계로 가지 않는다. 각 단계에는 예상 output digest, metric denominator와 로그 위치를 기록한다. rerun 전에 old artifact를 보존해 first divergence를 비교한다. 임의 seed 변경으로 통과를 시도하지 않는다.

최종 decision record에는 통과 조합, 미실행 optional path, known tolerance와 소스 좌표를 쓴다. 이 playbook을 깨끗한 새 환경에서 실행할 수 있을 때만 멀티노드 확장을 승인한다.

baseline 갱신을 review한다. dependency나 모델 code를 바꿔 expected tensor가 달라졌다고 baseline을 바로 재생성하지 않는다. old/new source diff에서 영향을 받는 invariant를 예측하고 실제 first divergence가 그 경계와 일치하는지 확인한다. 예상 밖 layer가 먼저 달라지면 변경을 승인하지 않는다.

수치가 달라져도 downstream metric이 좋아졌다는 이유만으로 통과시키지 않는다. mask·ownership·finite·checkpoint completeness 같은 구조 invariant는 그대로 유지돼야 한다. 허용 tolerance 변경에는 반복 run의 분포, dtype/reduction 근거와 reviewer가 필요하다.

새 baseline은 parent digest, change request, old/new 비교표, 소스 좌표와 승인자를 가진 immutable generation이다. old generation을 지우지 않아 회귀가 언제 들어왔는지 bisect할 수 있게 한다. architecture와 GPU 조합의 지원 종료도 폐기 기록으로 남긴다.

주기적으로 새 node와 깨끗한 cache에서 baseline을 실행한다. 개발자 workstation의 warm cache나 남은 compiled artifact에 의존하면 재현 패키지가 아니다. 독립 실행이 실패하면 release pipeline을 멈추고 누락 material부터 복원한다.

baseline의 수명에는 재검토 날짜와 지원 환경이 있다. 새 GPU, CUDA, compiler 또는 model revision을 기존 결과에 끼워 넣지 않고 child baseline을 만든다. 공통 synthetic fixture는 유지해 세대 간 변화의 방향을 볼 수 있게 한다. 폐기된 baseline도 과거 artifact의 재현과 incident bisect에 필요하므로 read-only로 보존한다. 저장 비용 때문에 tensor를 줄일 때도 checksum·통계·소스 명세서와 결정 기록은 남긴다.

새 baseline을 승인하기 전에는 이전 baseline에서도 새 test가 의도대로 통과하거나 실패하는지 확인한다. test 자체의 변경과 model의 변화를 한 번에 받아들이면 회귀를 숨길 수 있다. reviewer는 expected 값의 근거를 손계산 fixture와 source diff 양쪽에서 확인한다.

## 28.6 실행 옵션을 상태 전이 계약으로 번역한다

### 28.6.1 CLI 한 줄을 canonical config로 펼친다

`python train.py --gradient_accumulation_steps 8 --max_grad_norm 1.0 --bf16` 같은 명령은 사람이 읽기에는 그럴듯하지만 재현 계약으로는 부족하다. 첫째, 같은 이름의 옵션을 launcher, argument parser, Trainer와 사용자 callback이 차례로 덮어쓸 수 있다. 둘째, `bf16`은 parameter 저장 dtype, autocast 계산 dtype, optimizer state dtype을 한꺼번에 뜻하지 않는다. 셋째, accumulation 8은 microbatch 여덟 개의 산술평균인지, 유효 token numerator를 모두 더한 뒤 전체 denominator로 한 번 나누는지 말하지 않는다. 그러므로 golden run은 명령을 저장하는 데서 멈추지 않고 파싱 뒤의 canonical config와 실제 실행 상태를 나란히 저장한다.

각 옵션에는 네 칸짜리 원장을 붙인다. `입력 값`에는 사용자가 전달한 문자열과 default 출처를 쓴다. `소비 함수`에는 그 값을 처음 읽고 branch를 바꾸는 symbol을 쓴다. `상태 변화`에는 tensor, counter, process 또는 artifact가 어떻게 바뀌는지 쓴다. `관측 가능한 효과`에는 metric과 assertion을 쓴다. 예컨대 accumulation 값 8은 optimizer commit 사이에 microstep counter가 0에서 7까지 진행하고 앞의 일곱 microstep에서는 parameter checksum이 불변이며, 여덟 번째에만 optimizer step과 scheduler clock이 함께 하나 증가해야 한다. 이 네 칸 중 하나라도 비면 옵션 설명은 사용법일 뿐 메커니즘 설명이 아니다.

canonical config는 key 정렬, 단위 정규화와 명시적 default를 적용한 뒤 digest를 낸다. `1e-4`와 `0.0001`, `8k`와 `8192`처럼 의미가 같은 표기를 하나로 만든다. 반대로 생략된 default와 사용자가 명시한 값은 provenance에서는 구분한다. dependency 업그레이드가 default를 바꿀 수 있기 때문이다. parser 직후 config와 training loop 진입 직전 config가 다르면 diff를 hard failure로 만들거나 승인된 override 원장을 요구한다.

### 28.6.2 batch 크기 옵션을 수식으로 닫는다

단일 GPU에서도 `per_device_train_batch_size`, accumulation, sequence packing과 dynamic padding은 서로 독립이 아니다. sample 기준 nominal batch는 `B_s = B_micro K`지만 token 기준 update 크기는 `B_t = Σ_i Σ_j m_ij`다. 여기서 `m_ij`는 microbatch `i`의 token `j`가 loss에 기여하면 1이다. response-only SFT에서는 prompt와 padding token이 빠지므로 같은 sample 수라도 `B_t`가 크게 달라진다. golden manifest에는 nominal sample 수와 실제 contributing-token 수를 둘 다 둔다.

reference loss는 `L = (Σ_i N_i)/(Σ_i D_i)`로 계산한다. `N_i`는 microbatch의 token loss 합, `D_i`는 유효 label 수다. 각 microbatch의 평균 `N_i/D_i`를 다시 `K`로 나눈 값은 `D_i`가 다르면 같은 식이 아니다. 이 차이는 길이 분포가 일정한 toy batch에서는 숨어 있다가 packing 또는 response mask를 켰을 때 나타난다. 따라서 golden fixture는 유효 token 수가 의도적으로 다른 두 microbatch를 포함하고, reference numerator와 denominator를 CPU에서 손계산한다.

검증 assertion은 세 가지다. accumulation 중간에는 trainable parameter와 optimizer step counter가 변하지 않는다. commit 직전의 unscaled gradient는 모든 microbatch numerator를 같은 global denominator로 정규화한 reference와 맞는다. commit 뒤 scheduler가 참조하는 update counter는 정확히 하나 증가한다. 마지막 불완전 window를 버릴지, 실제 denominator로 보정할지 또한 config에 명시한다. 단순히 loss 곡선이 부드럽다는 사실은 이 셋을 증명하지 못한다.

### 28.6.3 AMP를 READY→UNSCALED→STEPPED로 읽는다

AMP의 핵심은 낮은 dtype이 아니라 optimizer update를 커밋할 수 있는지를 판정하는 상태 기계다. 이 책이 고정한 PyTorch revision `3691693263d2b66a68867e39b7449876844e06cf`에서 `torch/amp/grad_scaler.py:350-361`은 scale의 역수를 만들고 gradient를 unscale한 뒤 optimizer별 stage를 `UNSCALED`로 바꾼다. 같은 파일 `363-373`의 `_maybe_opt_step`은 device별 `found_inf` 합이 0일 때만 `optimizer.step`을 부른다. `375-385`의 공개 `step` 문서와 구현도 nonfinite가 있으면 parameter 손상을 막기 위해 update를 건너뛴다는 계약을 드러낸다.

그러므로 `bf16`이나 `fp16`을 켰다는 로그보다 `scale_before`, `found_inf_per_device`, `optimizer_stage`, `step_committed`, `scale_after`가 중요하다. overflow 주입은 loss에 무작정 `inf`를 넣지 않고 특정 trainable scalar의 gradient를 nonfinite로 만드는 작은 hook으로 수행한다. 기대 결과는 parameter checksum, Adam moment, optimizer step과 scheduler clock이 모두 불변이고 scaler만 선언된 backoff 규칙에 따라 변하는 것이다. parameter만 멈추고 scheduler가 진행하면 다음 정상 update에 잘못된 학습률이 적용된다.

fused optimizer는 다른 경로를 탈 수 있다. 같은 source `418-469`는 `_step_supports_amp_scaling`을 검사하고 `grad_scale`과 `found_inf` tensor를 optimizer에 전달하는 branch를 가진다. 따라서 일반 optimizer에서 golden을 통과했다고 fused 구현까지 승인하지 않는다. 두 경로에 동일한 finite/nonfinite fixture를 넣고 commit 여부와 state delta를 별도로 비교한다. 이 검사는 “AMP가 켜졌다”가 아니라 “overflow 때 부분 commit이 없었다”를 증명한다.

clipping의 관측 시점을 고정한다. `max_grad_norm=1.0`은 gradient가 언제 1.0으로 잘리는지까지 말해야 한다. 고정 PyTorch source `torch/nn/utils/clip_grad.py:185-232`에서 `clip_grad_norm_`은 개별 gradient norm들을 하나의 벡터처럼 보아 total norm을 구하고 gradient를 제자리에서 바꾼다. `error_if_nonfinite`의 default는 이 revision에서 `False`이고, `foreach=None`은 지원 device에서 빠른 구현을 선택할 수 있다. 따라서 CLI 이름만 기록하면 nonfinite 처리와 구현 branch가 빠진다.

golden trace는 `scaled_norm`, `unscaled_norm`, `clip_coefficient`, `post_clip_norm`을 구분한다. 올바른 순서는 accumulation 완료, AMP unscale, nonfinite 판정, global norm 계산, clipping, optimizer commit이다. scale된 gradient를 먼저 자르면 scale 값에 따라 실질 threshold가 달라진다. microstep마다 자르면 합산 gradient 방향도 바뀐다. fixture는 서로 다른 방향의 두 microgradient를 사용해 마지막에 한 번 자른 reference와 비교한다.

failure injection은 gradient 한 원소를 NaN으로 바꾸고 `error_if_nonfinite=True`인 진단 run에서 즉시 실패하는지 확인한다. production policy가 scaler skip을 사용한다면 예외 대신 `step_committed=0`이 돼야 한다. 어느 정책이든 NaN norm이 metric에만 찍힌 채 parameter가 갱신되는 결과는 금지한다. foreach와 scalar fallback의 결과 허용 오차도 별도 baseline으로 둔다.

## 28.7 한 update를 함수 호출 그래프로 해부한다

### 28.7.1 data fetch에서 logits까지 책임을 나눈다

한 step의 시작을 `model.forward` 호출로 잡으면 이미 늦다. dataset index 또는 stream cursor가 row를 고르고, formatter가 role과 field를 text로 만들며, tokenizer와 chat template가 token을 만들고, collator가 padding·labels·mask를 만든 뒤 device transfer가 일어난다. golden trace는 이 경계를 `RowID→RenderedBytes→TokenFixtureID→BatchID→ForwardID`로 연결한다. 어느 경계에서 bytes 또는 checksum이 처음 달라졌는지가 최초 divergence다.

각 경계는 입력과 출력을 동시에 남긴다. collator 뒤에는 `input_ids`, `labels`, attention mask와 position IDs의 shape, dtype, stride, min/max, digest를 쓴다. asynchronous device copy를 쓰면 copy 완료 event 이전 tensor를 읽지 않는다. pinned memory와 nonblocking 옵션은 성능 branch이지 semantic branch가 아니어야 하므로, 동기 copy reference와 token digest가 exact하게 같아야 한다.

forward에서는 전체 tensor dump 대신 architecture boundary를 고른다. embedding 출력, 첫·중간·마지막 residual, attention과 MLP 출력, norm 뒤 hidden state, logits의 selected slice를 기록한다. hook에는 동일 module이 여러 번 호출될 수 있으므로 module path뿐 아니라 call index를 붙인다. gradient checkpointing의 재계산 forward를 최초 forward로 오인하지 않도록 grad-enabled 상태와 phase도 기록한다.

### 28.7.2 logits와 loss를 수치 예제로 고정한다

어휘가 세 개인 한 token의 logits를 `[2, 0, -1]`이라 하자. 안정적인 log-softmax는 최댓값 2를 빼 `[0,-2,-3]`을 만들고, `logsumexp = 2 + log(1+e^-2+e^-3)`를 쓴다. 정답 ID가 0이면 negative log likelihood는 `log(1+e^-2+e^-3)`다. golden test는 framework scalar만 저장하지 않고 이 중간값과 정답 gather index를 저장한다. vocabulary 축을 잘못 선택하거나 label shift가 한 칸 어긋나면 즉시 드러난다.

causal LM에서는 위치 `t`의 logits가 보통 다음 label `t+1`에 대응한다. 그러나 모델 구현이 이미 shift한 loss를 반환하는지, 외부 trainer가 shift하는지는 소스 심볼로 확인해야 한다. 둘 다 shift하면 첫 유효 token이 사라지고, 둘 다 하지 않으면 현재 token 복사를 학습한다. fixture는 서로 다른 token sequence와 마지막 ignore label을 사용해 어느 위치가 numerator에 기여하는지 표로 보존한다.

response-only mask는 label을 ignore index로 바꾸는 시점과 attention mask를 혼동하지 않는다. attention mask는 누가 누구를 볼 수 있는지, loss mask는 어느 예측이 목적함수에 기여하는지를 정한다. prompt token이 attention에는 보이지만 loss에는 기여하지 않는 정상 사례를 fixture에 넣는다. `loss_contributing_tokens=0`은 0 loss로 지나가지 않고 data-contract failure여야 한다.

### 28.7.3 backward의 gradient ownership을 증명한다

backward 전에 모든 parameter를 `frozen`, `trainable`, `conditionally_used`로 분류한다. 각 행에는 fully qualified name, shape, dtype, parameter group, `requires_grad`, expected-gradient 이유를 둔다. LoRA라면 base weight는 frozen이고 A/B adapter가 trainable하다. tied embedding은 두 이름이 같은 storage를 가리킬 수 있으므로 object identity와 storage alias를 함께 기록한다.

backward 뒤에는 단순히 `grad is not None`만 보지 않는다. expected trainable parameter의 gradient finite 비율, norm과 selected checksum을 본다. `None`은 graph에 연결되지 않았다는 뜻일 수 있고, 0 tensor는 연결됐지만 현재 batch의 도함수가 0이라는 뜻일 수 있다. optimizer가 이 둘에 대해 state를 생성하는 방식도 다를 수 있다. 조건부 expert는 route count가 0이면 gradient가 없는 것이 정상일 수 있으므로 route ledger와 함께 판정한다.

고장 주입은 adapter output을 의도적으로 `detach`한 negative control이 유용하다. 정상 fixture에서는 adapter gradient assertion이 통과하고, detach 변형에서는 정확히 그 assertion이 실패해야 한다. test가 양쪽에서 통과하면 gradient ownership gate가 실제 결함을 잡지 못한다. 이처럼 golden test 자체도 결함에 민감하다는 것을 negative control로 증명한다.

optimizer delta를 원인별로 분해한다. update 뒤 parameter delta 하나만 보면 gradient, momentum, weight decay와 learning rate가 합쳐져 있다. 작은 scalar 또는 2×2 matrix fixture에서는 optimizer 수식을 항별로 계산한다. AdamW라면 gradient에서 갱신한 first/second moment, bias correction 뒤의 adaptive term과 decoupled decay term을 나눠 예상 delta를 만든다. 실제 tensor와 비교할 때 parameter group의 step counter, epsilon 위치와 dtype도 기록한다.

frozen parameter는 delta exact zero여야 한다. trainable parameter가 zero인 경우에는 learning rate 0, gradient 0, overflow skip, 잘못된 group 배정 중 어느 것인지 상태 원장으로 분리한다. `optimizer.step()`이 호출됐다는 로그는 parameter가 올바르게 갱신됐다는 증거가 아니다. 반대로 overflow 때문에 step을 건너뛴 것이 정상이라면 zero delta가 성공이다. 맥락 없는 delta threshold는 이 둘을 뒤집는다.

optimizer state checksum은 parameter checksum과 별도로 둔다. moment가 잘못 복원돼도 첫 resume 직후 parameter는 같을 수 있다. 두세 step 뒤에만 차이가 커진다. golden run은 최소 두 update의 moment와 delta를 추적하며, state save/load 뒤에도 같은 수식을 만족하는지 본다.

## 28.8 재현성을 등급과 오차 예산으로 관리한다

### 28.8.1 exact·numerical·statistical 등급

모든 출력을 bitwise로 강제하면 다른 합법적 reduction order를 결함으로 오인하고, 모든 것을 최종 metric tolerance로 풀면 token 순서 오류를 숨긴다. 따라서 invariant마다 등급을 정한다. token IDs, mask, parameter ownership, checkpoint file hash와 commit marker는 exact다. FP32 reference와 BF16 optimized activation·gradient는 absolute/relative tolerance를 쓴다. 장기 학습 metric은 반복 seed의 분포와 confidence interval로 본다.

수치 비교는 `|x-y| ≤ atol + rtol|y|`만으로 끝내지 않는다. 0 주변에서는 absolute error, 큰 norm에서는 relative error, vector 방향에는 cosine, 분포에는 quantile과 finite ratio를 함께 본다. 한 원소의 catastrophic outlier가 평균 오차에 묻히지 않도록 max error와 위치도 기록한다. tolerance는 실행 뒤 결과에 맞춰 늘리지 않고 dtype, reduction depth와 반복 baseline에서 사전에 정한다.

오차 예산은 경계마다 배분한다. eager FP32 reference에서 fused BF16 forward로 가는 오차, backward reduction 오차, optimizer state cast 오차를 따로 측정한다. 종단 parameter 차이만 보면 어느 경계가 예산을 소진했는지 알 수 없다. 첫 divergence atlas가 이 예산 원장과 연결될 때 optimized path를 설명할 수 있다.

### 28.8.2 결정성 flag의 실제 효력을 시험한다

결정성 설정은 선언이 아니라 관측 대상이다. Python·NumPy·CPU·각 CUDA generator state를 저장하고 같은 process 두 번, fresh process 두 번, checkpoint resume 한 번을 비교한다. 같은 seed에서도 worker scheduling, unordered container, autotuning과 비결정적 kernel이 결과를 바꿀 수 있다. 첫 batch 순서와 dropout mask digest가 먼저 같은지 본다.

`deterministic=true`가 unsupported operation에서 오류를 내는지, warning만 내는지 현재 framework source와 작은 실행에서 확인한다. 이 장에서는 모델을 크게 돌리지 않아도 synthetic module로 branch를 검증할 수 있다. 다만 실제 hardware 결과가 없으면 “지원됨”이라고 쓰지 않고 “고정 revision의 source 계약을 확인함, 장비 실행 필요”로 상태를 구분한다.

결정성을 위해 성능을 희생했다면 두 baseline을 섞지 않는다. correctness diagnostic은 deterministic eager, production candidate는 허용된 optimized path로 둘 수 있다. 둘 사이의 수치 parity와 성능 차이를 별도 표에 쓴다. production이 deterministic하지 않다는 이유로 검증을 포기하지 않고 반복 분포와 첫 divergence 관측을 강화한다.

### 28.8.3 compile과 shape specialization을 분리한다

compile 옵션은 단순 on/off가 아니다. 입력 shape와 dtype, control-flow guard가 cache key를 만들고 조건이 달라지면 재compile 또는 eager fallback이 생긴다. golden run은 최소·대표·경계 sequence bucket과 홀수 batch를 차례로 넣어 compile count, graph break와 selected kernel을 기록한다. warm-up step은 throughput 분모에서 뺀다.

eager reference와 compiled candidate의 token·mask는 exact, logits·gradient·delta는 선언된 tolerance로 비교한다. graph break를 유발하는 diagnostic hook을 켠 run은 성능 baseline에서 제외한다. hook 없는 optimized run에도 경계 checksum을 얻을 수 있도록 제한된 observer 또는 별도 replay를 설계한다.

고장 주입은 shape guard를 벗어나는 한 batch를 넣는 것이다. 기대 결과는 승인된 재compile 또는 명시적 fail-fast다. 조용한 eager fallback으로 correctness는 맞지만 throughput이 무너지는 경우를 metric으로 잡아야 한다. `compiled_graph_count`, `graph_break_count`, `fallback_step_count`와 step latency를 함께 본다.

## 28.9 checkpoint를 commit transaction으로 시험한다

### 28.9.1 저장 대상은 tensor 목록보다 넓다

중단·재개 등가성을 위해 model, optimizer, scheduler와 scaler만 저장해서는 부족하다. dataloader cursor, sampler permutation, current accumulation index, tokens seen, callback state, 각 RNG와 config/source digest가 필요하다. iterable source가 byte offset을 제공하지 않으면 sample-exact resume가 불가능할 수 있다. 이때 한계를 숨기지 않고 epoch 또는 distributional resume 등급으로 낮춘다.

TorchTitan 고정 revision `b482babc5f1d5d718e1719a735f9a2d86d1b9aff`의 `torchtitan/components/checkpointer/dcp.py:89-133`은 pipeline rank마다 optimizer index가 충돌할 수 있어 fully qualified name 기반 flattening이 필요하다고 설명한다. 단일 GPU에서는 충돌이 드러나지 않지만 29장으로 확장할 manifest에는 이름 기반 ownership을 지금부터 넣어야 한다. 같은 source `151-180`은 model, optimizer, dataloader와 LR scheduler를 manager state에 결합한다. weights 파일 하나를 checkpoint 전체라고 부를 수 없는 이유가 코드 구조에 드러난다.

같은 파일 `135-149`의 `async_mode`는 `disabled`, `async`, `async_with_pinned_mem` 세 상태를 검증한다. `201-215`에서는 비동기 mode에 별도 gloo process group, stager와 future가 생긴다. 옵션 하나가 저장 latency만 바꾸는 것이 아니라 새 process group과 미완료 future라는 failure surface를 추가한다. 단일 GPU golden에서도 synchronous를 reference로 통과시킨 뒤 async path를 별도 child baseline으로 다룬다.

### 28.9.2 kill point마다 원자성을 판정한다

checkpoint를 `temporary generation→파일 쓰기→size/hash manifest→completion marker→latest pointer` 순으로 publish한다. kill point를 file open 직후, tensor write 중간, manifest 직전, marker 직후에 둔다. loader는 marker가 없거나 hash가 맞지 않는 generation을 절대 선택하지 않고 마지막 완전 세대로 돌아가야 한다. latest pointer가 새 generation을 가리켜도 validation에 실패하면 parent를 찾는다.

복구 assertion은 “load 성공”보다 강하다. 선택한 generation ID와 parent가 예상과 같고, model·optimizer·scheduler·scaler·RNG·data cursor coverage가 100%인지 확인한다. unknown key와 missing key는 allowlist 없이는 실패다. 첫 resume batch ID, LR, dropout draw와 parameter/moment checksum이 연속 run의 같은 지점과 맞아야 한다.

disk full 주입은 실제 volume을 채우지 않고 quota가 작은 임시 directory나 faulting filesystem shim을 사용한다. 기존 완전 checkpoint를 손상시키지 않는 안전 경계를 먼저 확인한다. 저장 실패 뒤 completion marker가 생기지 않고 training이 계속되는지 중단되는지는 policy로 명시한다. 비동기 저장이 실패했는데 main loop가 성공으로 끝나면 출시 관문는 실패해야 한다.

### 28.9.3 다음 update의 동등성으로 resume를 판정한다

연속 N-step run을 A, K-step 저장 후 fresh process에서 N까지 재개한 run을 B라 하자. `K+1`의 RowID가 다르면 data cursor, RowID는 같고 dropout digest가 다르면 RNG, forward는 같고 gradient가 다르면 accumulation/scaler, update만 다르면 optimizer/scheduler state를 본다. 최종 loss 비교부터 시작하면 원인이 증폭된 뒤라 늦다.

resume 실험은 K를 optimizer commit 직후에만 두지 않는다. data fetch 뒤, accumulation 중간, overflow skip 뒤와 checkpoint callback 직전에도 강제 종료한다. 어느 경계가 supported recovery point인지 선언하고, 지원하지 않는 microstep은 마지막 commit부터 재실행한다. 재실행 때문에 중복된 metric이나 evaluation side effect가 idempotent한지도 본다.

weights-only load, optimizer reset 또는 새 dataset cursor는 resume가 아니라 branch다. 새 RunID를 만들고 parent artifact와 변화 이유를 연결한다. 사용자가 `resume_from_checkpoint` 문자열을 넣었다는 사실보다 실제 state coverage와 first-step equality가 resume 의미를 결정한다.

## 28.10 first divergence를 관측 가능한 경계에서 찾는다

### 28.10.1 metric의 denominator와 clock을 고정한다

`loss=1.2`, `tokens_per_second=5000`만 기록하면 비교가 어렵다. loss에는 numerator, contributing-token denominator와 reduction scope를 붙인다. 처리량에는 input token인지 loss token인지, warm-up 포함 여부와 측정 wall-clock 구간을 쓴다. step은 dataloader microstep, backward microstep, optimizer commit과 evaluation step을 분리한다.

최소 metric 묶음은 data wait, host-to-device, forward, backward, unscale/clip, optimizer, checkpoint와 evaluation latency다. memory는 allocated, reserved와 peak를 구분하고 sampling 시점을 쓴다. activation/gradient에는 finite ratio, norm과 selected quantile을 두되 parameter 이름 label을 그대로 Prometheus에 넣어 cardinality를 폭발시키지 않는다. 상세 per-parameter 자료는 artifact에, 운영 metric은 group 단위로 둔다.

모든 metric 행은 RunID, BatchID, optimizer step, 소스/config digest와 연결된다. logger buffer 때문에 process kill 직전 값이 사라질 수 있으므로 critical commit event는 내구성 있는 event ledger에도 남긴다. 관측 code가 CUDA sync를 추가해 성능을 왜곡하는지 observer-off baseline과 비교한다.

### 28.10.2 divergence bundle을 자동으로 만든다

assertion이 실패하면 마지막 수천 줄 로그를 던지는 대신 작은 bundle을 만든다. bundle에는 기대/실제 invariant, 최초 다른 boundary, 관련 input·parameter digest, 직전 optimizer/scaler/RNG state, 소스 좌표와 재현 명령이 들어간다. 민감 row 원문 대신 synthetic fixture ID 또는 접근 제한 reference를 사용한다.

activation mismatch면 마지막 정상 boundary와 첫 비정상 boundary 사이 module을 자동 확장해 다음 run의 observer 범위를 좁힌다. nonfinite면 최초 발생 tensor와 producer symbol, 직전 finite input을 보존한다. resume mismatch면 checkpoint key coverage와 data/RNG diff를 우선한다. 원인 유형마다 bundle schema가 달라야 조사자가 처음부터 같은 정보를 다시 모으지 않는다.

bundle 생성 자체도 golden test로 검증한다. 의도적으로 label 한 칸 shift, adapter detach, overflow와 partial checkpoint를 넣었을 때 예상 category와 source boundary가 선택돼야 한다. 단순 실패 탐지를 넘어 진단 경로가 작동하는지 시험한다.

### 28.10.3 성능 회귀와 정확성 실패를 분리한다

정확성 gate를 모두 통과한 run만 성능 비교에 들어간다. warm-up과 compile을 제외한 steady interval을 여러 번 측정하고 median, dispersion과 thermal/power 상태를 남긴다. profiler를 켠 run과 끈 run, cache가 찬 run과 깨끗한 run을 섞지 않는다. sequence와 contributing-token 분포가 다르면 tokens/s 비교도 무효다.

느려짐이 발생하면 data wait, CPU launch gap, kernel duration, memory pressure와 checkpoint overlap 순으로 분해한다. peak memory가 늘고 allocator retry가 증가했다면 kernel 자체보다 retention 또는 fragmentation을 의심한다. 처리량이 늘었지만 optimizer commit당 유효 token이 줄었다면 진짜 개선이 아니다.

성능 threshold는 반복 baseline의 자연 변동보다 커야 한다. 단 한 번 3% 느렸다는 이유로 회귀라 하지 않으며, 20% 빨라졌다는 이유로 수치 parity 실패를 승인하지도 않는다. 정확성 및 성능 판정은 같은 RunID 아래 별도 gate로 유지한다.

## 28.11 장애를 주입하고 복구·인계까지 닫는다

### 28.11.1 최소 장애를 정상 control과 짝지어 주입한다

첫 실험은 label shift다. collator 뒤 label을 한 칸 밀어 golden token-loss 표가 정확히 어느 위치에서 실패하는지 본다. 둘째는 adapter detach로 gradient ownership gate를 검증한다. 셋째는 한 gradient에 nonfinite를 넣어 AMP가 parameter·moment·scheduler를 함께 멈추는지 본다. 넷째는 clipping을 accumulation microstep마다 잘못 적용해 final gradient 방향 assertion이 실패하는지 본다.

다섯째는 checkpoint manifest 직전 process kill이다. incomplete generation이 선택되지 않고 parent로 복구돼야 한다. 여섯째는 resume 전에 dataloader cursor 하나를 전진시켜 `K+1` BatchID assertion이 가장 먼저 실패하는지 확인한다. 각 fault는 한 번에 하나만 켜며 정상 control과 negative control을 같은 fixture lineage에서 실행한다.

장애 주입의 성공 조건은 training이 실패하는 것이 아니다. 의도한 gate가 최초로 실패하고, 다른 upstream gate는 통과하며, bundle이 결함 위치를 지목하고, fault를 제거한 뒤 원래 baseline이 복원되는 것이다. 엉뚱한 곳에서 먼저 죽으면 관측 순서 또는 fixture가 불완전하다.

### 28.11.2 복구 assertion을 실행 가능한 문장으로 쓴다

“정상적으로 복구된다”를 금지하고 구체적 boolean으로 쓴다. `selected_checkpoint_generation == last_complete_generation`, `resume_batch_id == uninterrupted_batch_id`, `optimizer_step == K`, `scheduler_clock == committed_updates`, `parameter_digest == expected_digest`처럼 비교 대상을 명시한다. numerical invariant는 tolerance와 dtype을 함께 쓴다.

assertion은 실패 시 actual, expected와 provenance를 출력한다. checkpoint key가 빠지면 key 이름과 writer/reader symbol, token이 다르면 row/template/tokenizer digest, gradient가 다르면 first-divergence boundary를 보여준다. 단순 `False`는 자동화됐어도 친절한 진단이 아니다.

recovery 뒤에는 같은 fault를 다시 주입하는 regression test가 남아야 한다. 수동으로 config를 바꿔 통과시킨 기록만 있으면 다음 upgrade에서 재발한다. test는 fault가 있을 때 실패하고 없을 때 통과한다는 두 방향을 증명한다.

### 28.11.3 29장으로 넘길 인수 package를 만든다

멀티노드로 넘기는 것은 model weight 하나가 아니다. canonical config와 source/environment manifest, golden row와 batch, boundary별 activation·gradient·delta invariant, 연속/재개 checkpoint lineage, 작은 evaluation ledger, observer schema와 failure bundle 예제를 넘긴다. 각 artifact에는 digest와 생성 command를 기록한다.

29장은 world size만 늘리고 같은 global contributing tokens, optimizer algorithm, update clock과 fixture 순서를 유지한 reference부터 시작한다. distributed sampler와 reduction 때문에 바뀌어야 하는 항목과 바뀌면 안 되는 항목을 미리 표기한다. single-GPU 결과가 없으면 분산 결과의 차이를 통신·sharding 탓인지 기존 data/loss 오류인지 나눌 수 없다.

handoff gate는 세 문장으로 요약된다. exact invariant는 모두 통과한다. numerical invariant는 사전 오차 예산 안이며 최초 divergence가 설명된다. partial checkpoint, overflow와 data-cursor fault의 복구 assertion이 negative control을 포함해 작동한다. 셋 중 하나라도 빠지면 GPU 수를 늘리지 않는다.

소스 좌표와 판정표를 연결한다. 고정 revision 소스 원장를 연결한다. 이 장의 PyTorch 근거는 revision `3691693263d2b66a68867e39b7449876844e06cf`에 고정한다. AMP unscale 및 stage 전이는 `torch/amp/grad_scaler.py:350-361`, nonfinite 때 update skip은 `363-385`, fused optimizer와 `found_inf` 전달 branch는 `418-469`다. gradient clipping의 함수 서명, global norm 의미, nonfinite 및 foreach 옵션은 `torch/nn/utils/clip_grad.py:185-232`다. 행 번호만 믿지 않도록 path, symbol과 semantic anchor를 모두 manifest에 기록한다.

checkpoint 근거는 TorchTitan revision `b482babc5f1d5d718e1719a735f9a2d86d1b9aff`다. pipeline optimizer state의 이름 충돌과 FQN flattening 설명은 `torchtitan/components/checkpointer/dcp.py:89-118`, manager가 model·optimizer·dataloader·scheduler를 state에 넣는 코드는 `151-180`, async mode와 별도 process group/future 생성은 `201-215`다. 이 좌표들은 단일 GPU에서 보이지 않는 분산 failure surface까지 29장과 연결한다.

source를 읽었다는 사실과 runtime을 실행했다는 사실은 상태가 다르다. ledger에는 `source-confirmed`, `upstream-test-confirmed`, `local-synthetic-executed`, `hardware-pending`을 분리한다. 이 작업에서는 대규모 모델을 실행하지 않으므로 장비 성능값을 만들어내지 않는다. 독자는 자신의 revision에서 semantic anchor가 유지되는지 diff한 뒤 golden fixture를 실행한다.

통과·실패·미검증을 구분한다. 최종 판정표의 행은 environment, tokenizer/data, forward/loss, backward, optimizer, checkpoint/resume, evaluation와 performance다. 열에는 invariant 등급, 기대값, 실제값, 소스 좌표, 실행 artifact, 상태와 owner를 둔다. 상태는 `PASS`, `FAIL`, `NOT-RUN`, `NOT-APPLICABLE`만 사용하고 빈칸을 허용하지 않는다.

`NOT-RUN`은 실패보다 덜 나쁜 PASS가 아니다. fused optimizer나 compile을 실행하지 않았다면 해당 조합은 지원 범위에 들어가지 않는다. `NOT-APPLICABLE`에는 왜 branch가 존재하지 않는지 config 증거를 붙인다. optional feature를 껐다는 말만으로 실제 branch가 비활성인지 알 수 없으므로 runtime counter 또는 source-consumed config를 확인한다.

판정표는 요약 score로 압축하지 않는다. tokenizer exact 실패와 throughput 개선을 평균내면 안 된다. hard correctness gate 하나가 실패하면 멀티노드 승인은 거부된다. 성능 regression은 correctness를 통과한 뒤 별도 owner가 판단한다.

단일 GPU 기준선의 한계를 명시한다. golden run은 “작은 모델이 한 번 학습됐다”는 데모가 아니다. 데이터 bytes가 loss numerator로 변하고 gradient와 optimizer state를 거쳐 durable checkpoint가 되는 모든 경계를 설명 가능한 상태로 만드는 실험이다. 옵션은 함수 branch와 state delta로 번역되고, 수식은 손계산 fixture와 autograd로 검산되며, 장애는 최초 divergence와 복구 assertion으로 닫힌다.

이 기준선이 충분히 작은 이유는 사람이 검산할 수 있기 위해서다. 동시에 충분히 깊은 이유는 실제 training loop와 같은 tokenizer, collator, model boundary, AMP, optimizer와 checkpoint state를 지나기 때문이다. toy 수식과 production code 사이를 끊지 않는 것이 핵심이다.

명령행 옵션 인수표. 독자가 실제 설정을 검토할 때는 옵션 이름을 기능별로 묶지 말고 commit 경계별로 배열하는 편이 낫다. data 경계에는 dataset revision, split, shuffle seed, worker 수, prefetch, packing과 max length가 놓인다. 이 값들은 BatchID와 contributing-token denominator를 바꾼다. model 경계에는 checkpoint revision, attention implementation, gradient checkpointing, adapter target과 trainable ownership이 놓인다.

이 값들은 forward branch와 autograd graph를 바꾼다. update 경계에는 microbatch, accumulation, loss reduction, autocast dtype, scaler, clipping, optimizer와 scheduler가 놓인다. 이 값들은 gradient가 parameter delta로 commit되는 방법을 바꾼다. durability 경계에는 save interval, async mode, retention과 resume policy가 놓인다.

각 옵션을 바꾸기 전에는 예상 first divergence를 적는다. `max_length`를 바꾸면 가장 먼저 TokenFixtureID 또는 BatchID가 달라져야 한다. attention kernel만 바꾸면 token과 weight는 같고 첫 attention output에서 numerical divergence가 시작돼야 한다. optimizer epsilon만 바꾸면 forward와 gradient는 같고 첫 parameter delta에서 차이가 나야 한다. save interval만 바꾸면 training tensor trace는 같고 checkpoint event 시간만 달라져야 한다. 실제 최초 차이가 예상보다 앞서면 숨은 coupling이 있다는 뜻이다. 예를 들어 save callback을 추가했더니 dropout mask가 달라진다면 callback이 공용 RNG를 소비했을 수 있다.

옵션의 효과와 실패 양상도 한 쌍으로 쓴다. worker 수를 늘리는 목적은 data wait를 줄이는 것이지만 transform RNG와 row order가 흔들릴 수 있다. gradient checkpointing은 activation memory를 줄이지만 recomputation이 RNG와 hook 호출 횟수를 바꿀 수 있다. fused optimizer는 launch와 memory traffic을 줄일 수 있지만 AMP의 `found_inf` 전달 branch와 state dtype이 달라진다. async checkpoint는 step stall을 줄일 수 있지만 staging buffer, future와 background failure가 추가된다. 이처럼 이득만 설명하고 새 failure surface를 쓰지 않는 옵션 표는 golden run에 사용할 수 없다.

한 step 수치 원장. microbatch 두 개의 loss numerator를 각각 18과 12, contributing-token 수를 6과 2라 하자. 올바른 update loss는 `(18+12)/(6+2)=3.75`다. microbatch 평균을 다시 평균내면 `(3+6)/2=4.5`가 돼 긴 response보다 짧은 response가 과대 가중된다. golden artifact에는 `[N_1,D_1,N_2,D_2]`를 저장하고 backward에 사용한 effective scalar가 3.75와 일치하는지 본다. distributed로 넘어가면 rank별 numerator와 denominator를 all-reduce한 뒤 같은 식을 사용해야 하므로 이 작은 예가 29장의 reduction contract가 된다.

이 scalar에서 scale을 1024로 두고 backward했다고 하자. 관측해야 할 gradient는 scaled 상태와 unscaled 상태가 다르다. clipping threshold 1.0은 unscaled global norm에 적용한다. unscaled norm이 2.5면 coefficient는 안정화 항을 제외해 대략 0.4이고, post-clip norm은 1.0 근처여야 한다. 어느 gradient 하나가 `inf`라면 coefficient를 계산해 덮는 대신 scaler의 nonfinite policy가 update 전체를 막아야 한다. 이때 optimizer step과 scheduler가 모두 그대로라는 assertion이 partial commit을 막는다.

AdamW scalar fixture에서는 `θ=2`, gradient `g=0.5`, learning rate `η=0.1`, decay `λ=0.01`처럼 사람이 계산 가능한 값을 사용한다. adaptive term과 decay `-ηλθ`를 분리해 기록한다. 실제 delta가 총합과 맞아도 두 항이 우연히 상쇄할 수 있으므로 moment와 decay contribution을 별도로 검산한다. 두 번째 step에서는 bias correction과 saved moment가 들어오므로 checkpoint round-trip 결함을 더 잘 드러낸다. production parameter 전체에 손계산을 확대하는 대신 이 scalar reference와 group-level checksum을 연결한다.

실패 분기표를 읽는 순서. startup 전에 config digest가 다르면 실행하지 않는다. startup은 같지만 첫 RowID가 다르면 sampler와 cursor를 본다. RowID는 같지만 rendered bytes가 다르면 formatter/template, bytes는 같지만 token이 다르면 tokenizer artifact, token은 같지만 labels나 mask가 다르면 collator를 본다. 이 순서는 GPU 계산에 들어가기 전에 data-plane 결함을 닫는다.

forward에서 logits가 다르면 마지막 정상 architecture boundary를 찾는다. embedding부터 다르면 weight/token/position, attention부터 다르면 mask·position encoding·kernel, MLP부터 다르면 gate와 dtype, 마지막 head에서만 다르면 tied weight와 vocabulary mapping을 본다. forward는 같은데 loss만 다르면 shift, ignore index, response mask와 denominator다. loss는 같은데 gradient가 다르면 detach, checkpoint recomputation, scaler와 backward reduction을 본다.

gradient까지 같고 parameter delta가 다르면 clipping 시점, parameter group, optimizer state와 scheduler LR을 본다. 연속 run은 같지만 resume만 다르면 data/RNG cursor와 state coverage다. 모든 수치가 맞고 성능만 다르면 compile cache, data wait, synchronization과 allocator를 본다. 이 순서의 장점은 강한 upstream invariant를 통과한 사실로 가설 공간을 지운다는 데 있다. 로그에서 눈에 띄는 마지막 오류부터 고치는 것과 반대다.

관측 metric의 실제 판정식. `step_committed`를 중심 event로 삼는다. optimizer commit `u`에 대해 `tokens_seen_u`, `loss_numerator_u`, `loss_denominator_u`, `lr_u`, `grad_norm_before_u`, `clip_coefficient_u`, `parameter_delta_norm_u`가 한 행에 있어야 한다. overflow로 commit이 없으면 이 행을 성공 update처럼 쓰지 않고 `attempted_step` event에 `found_inf=1`을 남긴다. scheduler와 evaluation cadence가 attempted step을 쓸지 committed step을 쓸지도 선언한다.

data efficiency는 `data_wait/(step_wall)`로, compute 구간은 forward·backward·update를 겹침을 고려한 trace로 본다. 단순 합이 wall time보다 클 수 있으므로 span과 counter의 의미를 섞지 않는다. memory leak 의심은 첫 step과 마지막 step peak만 비교하지 않고 동일 phase의 allocated/reserved baseline을 여러 step 추적한다. gradient hook이나 retained loss tensor가 있으면 allocated가 commit 뒤 기준선으로 돌아오지 않는다.

alert는 원인과 직접 연결되는 invariant를 우선한다. `loss_nonfinite`와 `gradient_nonfinite`는 서로 다른 producer를 가리킨다. `trainable_zero_delta`는 overflow skip이 아닌 committed step에서만 발화해야 한다. `resume_batch_mismatch`는 checkpoint load 직후 첫 fetch에서 발화해야 한다. throughput alert는 correctness gate가 PASS이고 같은 workload bucket일 때만 비교한다. 그렇지 않으면 data 분포 변화가 성능 회귀로 오인된다.

장애 주입 안전 계약. golden lab이라도 fault를 무제한으로 넣지 않는다. synthetic fixture, 임시 checkpoint root와 단일 process 범위에서 시작한다. disk-full은 실제 공유 volume을 채우지 않고 quota 또는 실패를 반환하는 wrapper를 쓴다. process kill은 child PID와 generation directory를 확인한 뒤 실행한다. nonfinite hook은 선택한 parameter와 단일 step에서만 활성화되고 `finally` 또는 process 종료로 제거된다.

각 fault에는 blast radius, trigger, expected first failure, cleanup과 recovery artifact가 있다. label shift fault는 fixture copy에만 적용하고 원본 dataset을 수정하지 않는다. tokenizer mismatch는 별도 child config를 사용하고 cache를 덮어쓰지 않는다. partial checkpoint는 기존 완전 generation을 read-only로 둔 채 새 temporary generation에서만 일으킨다. 장애 실험이 다음 정상 control을 오염시키면 실험 자체가 실패다.

fault injection 전후 산출물 digest를 비교해 의도한 대상만 달라졌는지 확인한다. fault flag를 끈 뒤 같은 clean baseline이 다시 통과해야 한다. 한 번 실패를 관찰하고 끝내면 cleanup 결함이나 persistent cache 오염을 놓친다. `normal→fault→recovery→normal` 네 단계가 한 세트다.

checkpoint 세대 판정 예. generation 40이 complete이고 41을 쓰다 죽었다고 하자. 41에 model shard가 모두 있어도 optimizer shard 하나와 marker가 없다면 loader는 40을 선택한다. 파일 수정 시간이 최신이라는 이유로 41을 고르면 안 된다. generation 42가 complete지만 manifest parent가 예상 lineage와 다르면 자동 선택하지 않고 branch conflict를 보고한다.

40에서 resume한 첫 batch는 연속 run의 해당 BatchID와 같아야 한다. 41에서 이미 fetch했던 row가 외부 metric store에 남았더라도 training state에는 commit되지 않았을 수 있다. event에는 generation과 commit ID를 붙여 중복을 제거한다. evaluation이나 artifact upload도 idempotency key를 사용해 재시작 때 같은 결과를 새 결과처럼 publish하지 않는다.

retention은 새 generation의 completeness를 검증하고 latest pointer를 publish한 뒤에만 옛 세대를 지운다. `keep_latest_k=2`라고 save 시작과 동시에 오래된 checkpoint를 삭제하면 새 save 실패 때 rollback 폭이 사라진다. 삭제 policy도 checkpoint transaction의 일부다. TorchTitan source에서 retention field가 있다는 사실만으로 이 ordering이 자동 보장된다고 가정하지 않고 해당 writer와 purge path를 revision별로 읽고 fault test로 검증한다.

source upgrade 절차. PyTorch나 TorchTitan revision을 올릴 때 모든 golden expected 값을 한꺼번에 재생성하지 않는다. 먼저 고정 좌표의 semantic anchor를 새 source에서 찾는다. GradScaler stage 전이, nonfinite skip과 fused optimizer branch가 이동하거나 계약이 바뀌었는지 diff한다. clipping의 nonfinite default와 foreach 선택, checkpoint state coverage와 async mode도 비교한다.

그다음 영향을 받을 fixture를 예측한다. AMP 구현 변화는 overflow와 fused optimizer fixture, clipping 변화는 norm·nonfinite fixture, checkpoint 변화는 partial-write와 resume fixture를 우선한다. old/new revision을 같은 synthetic input으로 실행해 최초 divergence가 예상 경계에 있는지 본다. 예상하지 못한 tokenizer나 forward 차이가 먼저 나타나면 dependency closure 또는 environment가 함께 바뀐 것이다.

새 baseline 승인은 source diff, 실행 diff와 negative control 세 증거를 요구한다. expected 값 변경만 담은 commit은 거부한다. old baseline은 과거 artifact를 재현하고 incident를 bisect하는 증거이므로 삭제하지 않는다. 이렇게 해야 golden test가 최신 코드에 맞춰 늘 통과하는 장식이 아니라 변화의 의미를 설명하는 계측기가 된다.

장별 연결 지도. 5장의 tokenizer/template 계약은 이 장의 RenderedBytes와 TokenFixtureID로 들어온다. 6장의 packing과 mixture ledger는 BatchID와 denominator를 결정한다. 7~10장의 architecture 분석은 forward boundary observer를 정한다. 11~14장의 optimizer·scheduler·저정밀 지식은 commit state와 오차 예산을 만든다. 17장의 checkpoint 원칙은 kill-point 실험으로 실행된다. 18~20장의 SFT·preference·online RL은 loss component와 trainable ownership을 바꾼다.

반대 방향 연결도 중요하다. 이 장에서 loss denominator가 불분명하면 19장의 preference pair weighting과 20장의 policy update를 신뢰할 수 없다. optimizer step clock이 불분명하면 scheduler scaling 논의를 실험할 수 없다. checkpoint가 data와 RNG를 복원하지 못하면 장기 evaluation 차이를 algorithm 차이라고 해석할 수 없다. golden run은 앞 장의 개념을 단일 state machine으로 조립하고 뒤 장의 분산·release 실험에 검산 가능한 입력을 제공한다.

독자는 특정 framework API를 외우는 대신 이 연결을 자신의 stack에 옮긴다. 함수 이름이 달라도 row가 token으로, loss가 gradient로, gradient가 durable update로 가는 경계는 남는다. 각 경계의 owner, invariant와 failure injection을 찾으면 새로운 trainer나 custom model도 같은 방식으로 해부할 수 있다.

리뷰어가 묻는 열두 질문. 첫째, 실행한 code와 읽은 소스 리비전이 같은가. 둘째, CLI default와 runtime canonical config가 일치하는가. 셋째, golden row의 provenance와 재배포 조건이 명시됐는가. 넷째, rendered bytes·token·label·mask를 사람이 검산할 수 있는가. 다섯째, loss numerator와 denominator가 microbatch 및 token 수준으로 분리됐는가. 여섯째, trainable·frozen·conditional parameter ownership이 실제 gradient와 맞는가. 일곱째, AMP overflow에서 parameter·moment·scheduler가 원자적으로 멈추는가. 여덟째, clipping이 unscale과 accumulation 뒤 정확히 한 번 일어나는가.

아홉째, optimizer delta의 adaptive term과 decay를 작은 수치로 검산했는가. 열째, checkpoint가 data·RNG·optimizer clock을 포함하고 partial generation을 거부하는가. 열한째, 연속 run과 resume run의 최초 차이를 찾을 수 있는가. 열두째, 모든 고장 주입이 정상 control을 다시 복원하는가.

이 질문에 “프레임워크가 알아서 한다”는 답은 인정하지 않는다. framework의 source와 upstream test가 보장하는 범위를 먼저 적고, custom tokenizer, collator, callback, adapter와 저장소가 추가한 gap을 local fixture로 닫는다. 반대로 모든 내부를 다시 구현할 필요도 없다. 책임 경계와 state delta를 관측하고 negative control이 assertion을 발화시키면 된다. source 독해와 실행 증거가 만나는 지점을 좁고 강하게 만드는 것이 효율적인 golden run이다.

지원 범위 표의 예. `eager+FP32+AdamW+synchronous checkpoint`를 reference로 승인했다면, `compiled+BF16+fused AdamW+async checkpoint`는 하나의 옵션 조합이 아니라 네 개의 새로운 branch다. 먼저 BF16만 켜 forward·gradient·delta 오차 예산을 확정한다. 다음 compile만 켜 graph specialization과 fallback을 확인한다. fused optimizer를 켜 AMP `found_inf` 전달과 state round-trip을 검증한다. 마지막으로 async save를 켜 future failure와 completion ordering을 주입한다. 한꺼번에 켜고 최종 loss만 비교하면 결함이 상쇄되거나 최초 divergence를 찾을 수 없다.

조합 수가 폭발할 때는 pairwise나 위험 기반 표본화를 쓰되 reference로 이어지는 경로를 유지한다. tokenizer·loss mask·parameter ownership·checkpoint completeness처럼 치명적인 exact invariant는 모든 지원 조합에서 검사한다. kernel dtype처럼 numerical한 항목은 architecture와 shape bucket별 대표 조합을 택한다. 실행하지 않은 조합은 `NOT-RUN`으로 남기고 지원된다는 인상을 주지 않는다. 비용 때문에 줄이는 것과 증거 없이 일반화하는 것은 전혀 다르다.

완료 기록의 최소 문장. 좋은 기록은 다음처럼 읽힌다. “고정 source와 config digest에서 synthetic GoldenBatchID G를 사용했고, loss numerator/denominator의 손계산과 eager FP32가 일치했다. BF16 candidate의 boundary별 오차는 사전 예산 안이었다. trainable adapter gradient와 AdamW 두 step delta가 reference에 맞았다. overflow에서는 optimizer·scheduler가 모두 정지했다. checkpoint manifest 직전 kill 뒤 마지막 complete generation을 선택했고, resume 첫 batch·RNG·LR·parameter/moment가 연속 run과 일치했다. compile 경로는 실행하지 않았으므로 지원 범위에서 제외한다.”

이 문장은 “학습이 잘 됐다”보다 길지만 모호하지 않다. 실행한 것, 검증한 메커니즘, fault recovery와 미검증 경계를 한 번에 보여준다. 독자는 artifact ID와 소스 좌표를 따라가 같은 판정을 재현할 수 있고, dependency가 바뀌면 어느 fixture부터 다시 실행할지 알 수 있다. golden run의 최종 산출물은 특정 loss 값이 아니라 이처럼 반증 가능한 주장들의 묶음이다.

마지막으로 baseline을 만든 사람과 승인한 사람을 분리한다. 작성자는 예상값과 실행 artifact를 제출하고, 리뷰어는 source 좌표·손계산·negative control을 독립적으로 대조한다. 같은 사람이 둘을 맡아야 한다면 최소한 깨끗한 환경에서 재생한 두 번째 기록을 남긴다. 승인 시각보다 중요한 것은 어떤 증거 세대를 승인했는지 식별하는 immutable digest다. 이후 수정은 기존 기록을 덮지 않고 parent를 가진 새 세대로 남긴다.

이제 29장에서 rank, process group, collective, sharding과 remote storage가 추가된다. 새 복잡성을 받아들이기 전에 28장의 `GoldenBatchID`, contributing-token denominator, update clock, parameter/optimizer checksum과 checkpoint generation을 그대로 들고 간다. 분산 실행이 다를 때 “원래 수치가 흔들린다”가 아니라 어느 새 경계가 최초 차이를 만들었는지 말할 수 있어야 한다. 그것이 단일 GPU golden run이 제공하는 가장 값비싼 산출물이다.

## 28.12 손으로 닫히는 canonical fixture를 구현한다

앞의 계약은 여기서 처음으로 실행 가능한 객체가 된다. canonical config를 만들고, 사람이 재계산할 수 있는 dataset과 model을 고른 뒤, 한 update와 checkpoint commit을 같은 fixture 안에서 닫는다. 이 절의 출력은 설명 문장이 아니라 다음 절의 runner가 그대로 소비할 tensor·state·expected value다.

CLI와 환경을 실행 가능한 한 문서로 합친다. 사용자가 입력한 command line은 default resolution 전 상태다. parser default, config file, environment, auto device·dtype 선택과 model config가 합쳐진 뒤 실제 runtime config가 만들어진다. golden runner는 학습을 시작하기 전에 이 canonical config를 직렬화하고 digest를 계산한다. 알 수 없는 field와 의미 없는 default도 숨기지 않는다.

config에는 source commit, entrypoint, model·tokenizer·dataset 산출물 digest, output root, seed 집합, deterministic policy, max length, packing, batch와 accumulation, precision, optimizer·scheduler, clipping, checkpoint와 evaluation을 넣는다. device name, compute capability, driver·CUDA·framework와 loaded native inventory는 environment manifest로 분리하되 config가 그 digest를 참조한다.

secret, 사용자 home path와 임시 credential은 canonical config에서 제외한다. 의미 있는 path는 content digest와 logical role로 바꾼다. 실행마다 달라지는 timestamp와 output directory가 config digest를 흔들지 않도록 invocation metadata와 semantic config를 분리한다. 반대로 실제 의미를 바꾸는 environment variable을 편의상 빼서는 안 된다.

canonicalization은 key sort만 뜻하지 않는다. 정수와 문자열 `"8"`, bytes와 MiB, dtype alias, 상대 path와 resolved artifact를 같은 의미로 정규화한다. unknown option은 무시하지 않고 실패한다. framework upgrade로 default가 바뀌면 canonical diff에서 드러나야 한다.

config diff가 예상 최초 divergence를 낸다. 각 field에는 영향 경계를 붙인다. tokenizer revision과 template는 rendered bytes 또는 token ID에서, max length·packing은 BatchID와 mask에서, attention implementation과 dtype은 forward activation에서, optimizer option은 parameter delta에서, checkpoint interval은 durability event에서 처음 차이를 만들어야 한다.

두 run의 config diff와 실제 first-divergence boundary를 비교한다. save interval만 바꿨는데 token order가 달라지면 callback이 RNG나 data iterator를 건드렸을 수 있다. logging 빈도만 바꿨는데 forward가 달라지면 동기화·hook 또는 공용 RNG 소비를 조사한다. 예상보다 이른 divergence는 숨은 coupling의 증거다.

### 28.12.1 golden dataset을 손으로 검산할 크기로 만든다

row provenance와 rendered bytes를 함께 고정한다. fixture는 몇 개의 짧은 row로 구성하되 실제 pipeline과 같은 loader, template, tokenizer와 collator를 지나야 한다. assistant-only loss, multi-turn, empty response, truncation 경계, Unicode normalization, special token과 padding을 드러내는 row를 포함한다. 원문 사용 조건과 digest를 기록한다.

각 row에서 raw field, normalized field, rendered UTF-8 bytes, token IDs, attention mask, labels와 contributing-token count를 보존한다. secret이나 실제 사용자 대화를 쓰지 않는다. synthetic text라도 어느 version에서 생성됐는지 manifest에 넣는다. expected artifact는 사람이 읽는 table과 machine fixture를 함께 둔다.

chat template는 문자열 모양이 아니라 loss mask를 바꾼다. BOS/EOS 중복, assistant marker, tool call serialization과 generation prompt가 기대한 token 위치에 있는지 확인한다. label shift 뒤 logit 위치와 target token을 작은 vocab 예제로 손계산한다. 모든 label이 ignore index인 batch는 명시적으로 거부하거나 zero-loss 정책을 test한다.

shuffle과 sampler는 row ID sequence를 출력한다. dataloader worker 수, prefetch와 persistent worker를 바꿔도 주장한 deterministic 등급 안에서 같은 sample stream이 나오는지 본다. resume fixture에는 다음 row ID와 within-shard cursor가 포함된다. batch tensor만 같고 provenance가 다르면 중복·누락을 찾기 어렵다.

### 28.12.2 memory를 생존 tensor와 순간 peak로 나눈다

단일 GPU에서 OOM을 피하는 첫 방법은 무작정 batch를 줄이는 것이 아니라 메모리 소유자를 계산하는 것이다. parameter, gradient, optimizer state, activation, temporary workspace, allocator fragmentation, compilation과 checkpoint staging을 구분한다. 각 항의 dtype과 생존 구간을 적는다.

Adam 계열은 trainable parameter마다 parameter뿐 아니라 gradient와 두 moment를 대개 유지한다. master weight를 별도로 유지하는 precision path라면 항이 추가된다. adapter 학습은 frozen base의 gradient·optimizer state를 줄이지만 forward activation과 base weight 자체는 남는다. optimizer state offload가 없는 단일 GPU에서는 이 차이를 수치로 예상한다.

activation memory는 batch, sequence, hidden, layer와 attention 구현에 의존하며 단순 parameter 비율로 예측되지 않는다. gradient checkpointing은 선택 구간 activation을 버리고 backward에서 재계산해 memory를 줄이지만 compute와 RNG 경계를 바꾼다. flash 계열 attention은 materialized score matrix를 줄일 수 있지만 shape·dtype·kernel 지원에 따라 fallback한다.

peak는 forward 끝, backward 중, optimizer step과 checkpoint serialization 가운데 발생할 수 있다. phase marker와 allocated·reserved peak를 함께 기록한다. `empty_cache`로 숫자를 낮추는 것을 leak 수정으로 부르지 않는다. golden baseline은 동일 shape 순서의 peak와 retry를 비교하고 snapshot은 짧은 이상 window에서만 쓴다.

### 28.12.3 한 optimizer update를 정확한 순서로 실행한다

gradient 초기화의 의미부터 고정한다. `zero_grad(set_to_none=True)`는 gradient tensor를 0으로 채우는 대신 `.grad`를 `None`으로 만든다. 다음 backward에서 새 gradient가 할당되고, gradient가 전혀 도달하지 않은 parameter와 계산 결과가 정확히 0인 parameter를 구분할 수 있다. 일부 optimizer는 `grad is None`인 parameter를 건너뛰고 zero gradient가 있는 parameter에는 weight decay나 state update를 적용할 수 있으므로 옵션이 parameter delta를 바꿀 수 있다.

golden fixture는 trainable parameter 가운데 의도적으로 사용되지 않는 하나와 사용되지만 gradient가 0인 하나를 둔다. step 전후 parameter와 optimizer state를 비교해 구현 계약을 확인한다. adapter target을 잘못 지정해 모든 gradient가 `None`인 경우 loss가 정상이어도 update가 전혀 없을 수 있다. `trainable_count`, `grad_present_count`, `nonzero_grad_count`를 분리한다.

gradient accumulation에서는 update 시작 때 한 번만 초기화한다. microbatch마다 zeroing하면 accumulation이 사라지고 마지막 microbatch만 반영된다. 반대로 update 뒤 zeroing이 누락되면 이전 update gradient가 섞인다. observer는 accumulation window ID와 microstep index를 기록하고 손계산한 gradient sum 또는 mean과 비교한다.

forward와 loss reduction을 microbatch 계약으로 묶는다. global batch를 \(K\)개 microbatch로 나눌 때 각 microbatch loss를 무조건 \(1/K\)로 나누는 것은 valid token 수가 같을 때만 전역 token mean과 일치한다. 길이와 mask가 다르면 각 loss numerator를 합하고 전체 valid token denominator로 나눠야 한다. framework가 반환하는 scalar mean에 단순 scaling을 적용하는 경로를 fixture로 검산한다.

메모리 때문에 microbatch를 바꾸더라도 effective batch의 sample stream과 objective가 유지돼야 optimizer 비교가 가능하다. dropout mask와 batch-dependent layer가 있으면 microbatch 분할이 bitwise trajectory를 바꿀 수 있다. Transformer의 일반적인 layer normalization은 sample 독립적이지만 이것만으로 모든 custom module의 등가성을 보장하지 않는다. claimed reproducibility grade에 반영한다.

autocast context는 일부 operation의 입력·accumulation dtype을 선택한다. model parameter dtype과 autocast dtype, loss calculation dtype을 같은 것으로 부르지 않는다. logits-to-loss의 안정화와 reduction이 어느 precision에서 수행되는지 source와 observer로 확인한다. reference는 eager FP32로 두고 BF16/FP16 candidate의 boundary error를 비교한다.

scaled backward에서 commit 이전 상태를 구분한다. FP16 GradScaler 경로에서는 loss에 scale을 곱해 backward하고 gradient를 unscale한 뒤 nonfinite를 검사한다. clipping은 unscale 이후에 해야 원래 gradient norm을 기준으로 동작한다. unscale을 두 번 호출하거나 optimizer step 뒤 clipping하는 순서 오류를 negative fixture로 둔다.

overflow가 발견되면 optimizer step이 건너뛰어져야 한다. 그때 parameter, optimizer moment와 scheduler clock이 함께 멈추는지 확인한다. logging loop counter만 증가할 수 있으므로 `attempted_update`와 `committed_update`를 분리한다. scaler update는 scale을 낮출 수 있으므로 전체 상태가 완전히 동일한 것은 아니다. 어느 component가 변해야 하는지 expected state table을 쓴다.

fused optimizer는 scaler가 `found_inf`와 scale 정보를 optimizer 경로에 전달할 수 있다. 일반 optimizer wrapper와 같은 callback 순서를 가정하지 않는다. revision별 branch를 읽고 overflow fixture에서 parameter·moment가 불변인지 exact 검사한다. candidate가 이를 못 보이면 fused mode를 지원 범위에서 제외한다.

clipping과 optimizer를 하나의 commit으로 본다. global norm clipping은 모든 trainable gradient의 p-norm을 결합한다. parameter group별 clipping과 전체 model clipping은 다른 update를 만든다. sparse gradient, nonfinite와 foreach path의 동작을 확인한다. 반환된 norm이 clip 전인지 후인지 metric schema에 적는다.

AdamW golden scalar는 첫 step과 두 번째 step을 손으로 계산한다. 첫 step은 moment 초기화와 decoupled decay, 두 번째 step은 saved moment와 bias correction을 검증한다. epsilon이 분모 안팎 어느 식에 놓이는지 구현과 reference를 맞춘다. weight decay가 loss gradient에 더해지는 L2와 decoupled update를 혼동하지 않는다.

optimizer가 성공한 뒤 scheduler가 한 번 전진한다. scheduler를 microstep마다 호출하거나 overflow skip에도 호출하면 LR clock이 실제 update와 어긋난다. warmup step 수가 optimizer update인지 batch iteration인지 config에서 명시한다. resume 후 scheduler `last_epoch` 같은 내부 이름을 의미 그대로 epoch로 해석하지 말고 실제 함수의 increment 계약을 확인한다.

옵션을 상태 변화와 효과로 번역한다. batch와 accumulation 옵션. `per_device_train_batch_size`는 한 forward에 들어가는 sample 수를 바꾸고 activation memory와 shape를 바꾼다. `gradient_accumulation_steps`는 optimizer commit 사이 microbatch 수를 바꾼다. 둘의 곱은 sample 기준 effective batch지만 variable length에서는 effective valid token 수가 흔들린다. token histogram과 update별 denominator를 기록한다.

accumulation을 늘리면 peak activation을 줄인 채 큰 batch를 근사할 수 있지만 update frequency와 scheduler 해석, logging·evaluation·save cadence를 바꿀 수 있다. framework가 `logging_steps`를 optimizer step으로 세는지 dataloader iteration으로 세는지 source에서 확인한다. 옵션 설명은 memory 이득과 clock failure surface를 함께 적는다.

precision 옵션. `fp16`은 표현 범위 때문에 loss scaling이 주로 필요하고, `bf16`은 exponent 범위가 넓지만 mantissa precision이 낮다. 둘 다 “half precision”으로 묶어 동일 tolerance를 쓰지 않는다. hardware가 dtype을 지원하는지, unsupported operation이 fp32로 올라가는지, optimizer state dtype이 무엇인지 runtime manifest에 넣는다.

TF32 허용은 FP32 matrix multiply의 내부 정밀도와 algorithm path를 바꿀 수 있다. 단일 boolean의 성능 이득만 보지 않고 forward·gradient first divergence와 tolerance를 확정한다. deterministic policy와 compile·fusion option이 동시에 kernel selection에 영향을 줄 수 있으므로 하나씩 추가한다.

gradient checkpointing 옵션. checkpointing은 forward activation 일부를 저장하지 않고 backward에서 forward 구간을 재실행한다. peak memory를 낮추지만 compute가 늘며 dropout 같은 RNG 상태를 보존해야 등가성이 유지된다. reentrant/non-reentrant 구현, `use_cache`와의 충돌, hook 호출 횟수와 detached tensor를 test한다.

memory 절감률은 모델·segment·sequence에 따라 다르다. 옵션을 켰다는 사실만 기록하지 않고 peak phase, step time과 forward/gradient numerical parity를 측정한다. recompute 때문에 observer가 같은 activation을 두 번 기록할 수 있으므로 phase와 invocation ID를 둔다.

compile 옵션. compile은 Python graph capture, graph break, specialization과 backend code generation 상태를 추가한다. 첫 step compile time과 steady-state를 분리한다. dynamic shape bucket에서 recompilation count와 reason을 기록한다. graph break가 있어도 correctness는 맞을 수 있지만 성능 주장 범위가 달라진다.

eager reference와 compiled candidate는 같은 GoldenBatchID와 initial state에서 boundary output을 비교한다. compile cache를 warm한 실행과 clean-cache 실행을 구분한다. compiler cache artifact에는 source, flags, environment·architecture와 output digest를 기록한다. 다른 environment에서 가져온 cache를 무검증 재사용하지 않는다.

optimizer 구현 옵션. foreach, fused와 scalar loop는 같은 수학식을 목표로 해도 operation ordering과 rounding, memory peak가 다르다. 지원 dtype·device, AMP integration과 state serialization을 각각 시험한다. 속도 비교는 같은 workload와 correctness gate 통과 후 수행한다. fallback이 일어나면 선택한 옵션 이름보다 실제 path를 기록한다.

checkpoint를 commit protocol로 구현한다. 세대별 immutable state를 쓴다. checkpoint root 아래 `generation-N.tmp-attempt-X` 같은 staging에 model, optimizer, scheduler, scaler, RNG, sampler와 config reference를 쓴다. 각 child의 digest와 size를 manifest에 기록한다. 모든 write와 필요한 durability barrier가 성공하면 complete manifest를 만들고 catalog의 latest generation을 원자적으로 갱신한다.

파일 시스템 rename이 원자적이라는 가정은 backend 범위 안에서만 유효하다. object store에서는 immutable key와 마지막 commit object를 사용한다. multipart upload가 완료됐다는 provider 응답과 모든 child digest가 검증됐다는 application 상태를 분리한다. latest pointer가 먼저 보이면 loader가 incomplete generation을 선택할 수 있다.

single GPU라도 async save는 training state와 staging copy의 ownership 문제가 있다. writer가 parameter를 복사하는 동안 optimizer가 다음 update를 시작하면 서로 다른 시점의 component가 섞일 수 있다. snapshot boundary와 copy completion, buffer lifetime을 정의한다. async future 성공 전에 generation을 complete로 승격하지 않는다.

kill point를 단계마다 둔다. model write 중, optimizer write 뒤, manifest 전, manifest 뒤 latest 전, latest 뒤 retention 전에 process를 종료한다. loader는 마지막 complete generation만 선택하고 partial을 격리해야 한다. complete manifest가 있지만 child digest가 틀린 fixture도 거부한다. mtime이나 가장 큰 generation 번호만으로 선택하지 않는다.

disk-full은 공유 disk를 실제로 채우지 않고 writer wrapper나 quota가 특정 write에서 실패하도록 한다. permission error, short write, fsync failure와 background exception도 주입한다. failure 뒤 training을 계속할지 중단할지는 RPO와 state 정책에 따라 결정하고 checkpoint 실패를 성공 metric으로 기록하지 않는다.

retention은 새 generation 검증과 publish 뒤 실행한다. keep-last 정책이 parent lineage, best checkpoint와 incident hold를 보호하는지 본다. deletion이 실패해도 새 checkpoint correctness를 훼손하지 않지만 storage pressure alert와 재시도가 필요하다. 삭제된 generation의 catalog entry는 tombstone과 reason을 남긴다.

resume은 clean process에서 시험한다. 같은 process에서 save 직후 load하면 초기화·import·cache 누락을 숨긴다. 새 process와 clean output root에서 manifest를 검증하고 state를 restore한다. 다음 BatchID, RNG draw, loss numerator/denominator, gradient, parameter delta, optimizer moment, LR와 committed update를 uninterrupted control과 비교한다.

resume 직후 evaluation이나 logger가 RNG를 소비하면 다음 training batch가 달라질 수 있다. subsystem별 RNG를 분리하거나 호출 순서를 계약으로 고정한다. dataloader worker state를 완전히 복원할 수 없다면 deterministic replay boundary와 중복/누락 정책을 명시한다.

adapter의 parameter ownership을 증명한다. 이름 필터가 아니라 실제 graph를 확인한다. LoRA target module 문자열이 예상 module과 매치됐다는 로그만으로 충분하지 않다. injection 전후 named module과 parameter를 비교하고 trainable flag, shape, dtype, storage와 parent module을 manifest로 남긴다. target이 0개이거나 너무 많은 module에 적용되면 시작 전에 실패한다.

trainable parameter에는 backward 뒤 gradient가 존재하고 committed step 뒤 delta가 있어야 한다. frozen base는 gradient와 delta가 없어야 한다. 하지만 weight tying, shared storage, buffer와 optimizer decay 때문에 단순 이름 비교가 틀릴 수 있다. storage identity와 optimizer parameter group을 교차한다.

LoRA의 초기화에서 한 factor를 0으로 두면 첫 step에 다른 factor의 gradient가 0일 수 있다. 이를 trainable failure로 오판하지 않도록 작은 수식 fixture로 예상 gradient를 계산한다. 두 step을 실행해 두 factor가 언제 움직이는지 확인한다. scale \(\alpha/r\), dropout과 merge flag가 forward path를 어떻게 바꾸는지 기록한다.

adapter save는 base weight를 포함하지 않을 수 있으므로 base 산출물 digest와 adapter config가 필수 parent다. load fixture는 올바른 base, 잘못된 base, target module 누락과 dtype mismatch를 시험한다. `strict=False` 경고로 base mismatch를 숨기지 않는다. merge export는 별도 파생 artifact와 evaluation을 만든다.

QLoRA 계열에서는 quantized base storage, dequantization compute dtype, trainable adapter와 optimizer state dtype을 분리한다. “4-bit 학습”은 모든 계산과 state가 4-bit라는 뜻이 아니다. quantization option이 memory, forward numerical error와 kernel path를 바꾸는 상태 사슬을 측정한다.

forward·loss·backward oracle을 손으로 닫는다. 작은 vocabulary에서 shift를 눈으로 확인한다. 두세 token의 sequence와 작은 vocabulary logits를 사용해 next-token target을 고정한다. position \(t\)의 logit이 token \(t+1\)을 예측한다는 shift, ignored prompt와 padding 위치를 table로 표시한다. stable log-softmax를 손으로 계산해 token별 negative log likelihood와 numerator 합, denominator를 얻는다.

예를 들어 valid target 두 개의 loss가 \(\ell_1, \ell_2\)이면 batch loss는 \((\ell_1+\ell_2)/2\)다. sequence별 mean의 평균은 각 sequence valid token 수가 다르면 다른 값이다. collator와 model wrapper가 어느 reduction을 반환하는지 tiny fixture로 확인한다. label smoothing, auxiliary router loss나 preference component가 있으면 별도 numerator와 weight를 기록한다.

embedding lookup은 token ID와 row mapping을 검증한다. tied output head에서는 embedding storage와 output projection relation을 확인한다. position ID와 attention mask를 명시해 padding·left/right padding 차이를 드러낸다. attention implementation을 바꿀 때 이 boundary observer가 최초 numerical divergence를 찾는다.

모든 activation을 저장하면 golden artifact가 커지고 implementation detail에 과도하게 결합된다. embedding output, 첫 attention output, 첫 MLP output, 마지막 hidden, logits, loss처럼 architecture boundary를 선택한다. tensor shape, dtype, finite, norm과 small-fixture exact value를 기록한다.

backward oracle을 방향과 크기로 분해한다. autograd가 gradient를 만들었다는 사실만 보지 않는다. scalar loss를 특정 parameter에 대해 finite difference로 근사해 gradient 방향과 크기를 교차 검증할 수 있다. 아주 작은 FP64 fixture에서 \([f(\theta+h)-f(\theta-h)]/(2h)\)를 계산하고 autograd와 비교한다. \(h\)가 너무 작으면 cancellation, 너무 크면 truncation error가 생기므로 여러 scale을 본다.

production dtype과 모델 전체에 gradcheck를 강제하지 않는다. custom loss, adapter injection, mask와 새 kernel wrapper처럼 위험한 작은 함수에 사용한다. stochastic layer를 끄거나 RNG를 동일하게 고정한다. non-smooth clipping·quantization boundary에서는 finite difference 해석의 한계를 명시한다.

gradient accumulation oracle은 각 microbatch의 numerator gradient를 따로 계산해 합한다. token-mean objective면 최종 denominator로 나눈 reference와 실제 accumulated gradient를 비교한다. 각 microbatch mean을 같은 가중치로 더한 잘못된 경로가 length-imbalanced fixture에서 실패해야 한다.

backward hook은 관측을 위해 graph lifetime과 ordering을 바꿀 수 있다. hook on/off의 gradient와 memory를 비교한다. checkpoint recomputation에서는 hook 호출 횟수가 늘 수 있으므로 invocation phase를 구분한다. gradient observer가 학습 code를 오염시키면 oracle 자격이 없다.

AdamW와 Muon을 작은 행렬에서 검산한다. AdamW의 상태를 항별로 기록한다. 첫 step에서 gradient \(g\), first moment \(m\), second moment \(v\), bias correction, adaptive direction과 decoupled decay를 따로 계산한다. optimizer state dict의 step counter와 tensor dtype을 확인한다. parameter group별 learning rate, beta, epsilon과 decay exclusion이 canonical config와 일치해야 한다.

두 번째 step은 checkpoint round-trip에 중요하다. model weight가 같아도 moment나 step counter가 누락되면 update가 달라진다. save 전후 state tensor digest와 next delta를 비교한다. scheduler LR과 optimizer group LR이 restore 순간 언제 동기화되는지도 본다.

Muon은 2차원 hidden weight의 방향 변환을 검증한다. Muon 계열은 momentum update에 행렬 직교화 또는 극분해를 근사하는 변환을 적용할 수 있다. 구현별 Newton–Schulz step, normalization, transpose convention, learning-rate scaling과 fallback을 소스 리비전에 맞춘다. 모든 parameter에 같은 방식으로 적용하지 않고 embedding·norm·bias나 1차원 parameter는 AdamW 같은 별도 group을 사용할 수 있다.

작은 2×2 또는 2×3 gradient matrix에서 momentum과 변환 결과를 reference precision으로 계산한다. transpose가 필요한 tall/wide matrix, zero matrix, rank-deficient와 큰 norm을 fixture에 넣는다. iteration 수를 바꾸면 compute와 orthogonality error가 어떻게 변하는지 측정한다. 최종 loss 하나로 optimizer 구현을 승인하지 않는다.

mixed optimizer checkpoint는 group type, parameter FQN mapping과 각기 다른 state schema를 보존해야 한다. parameter ordering이 바뀌면 state가 다른 weight에 붙을 수 있다. distinctive tensor fixture로 mapping을 검사한다. resume 뒤 Adam group과 Muon group의 next delta를 각각 비교한다.

evaluation을 training 상태에서 격리한다. evaluation 전후 model train/eval mode를 복원한다. dropout과 normalization behavior, gradient enabled 상태를 확인한다. 평가 callback이 training dataloader cursor나 RNG를 소비하지 않게 generator를 분리한다. evaluation 실패가 checkpoint complete marker를 잘못 publish하지 않도록 transaction을 분리한다.

작은 golden evaluation은 fixture별 expected logit/loss 또는 deterministic decoding을 사용한다. generation에서는 tokenizer/template, max tokens, temperature, top-p, stop와 seed를 명시한다. greedy라 해도 backend kernel 차이로 logit tie가 갈릴 수 있으므로 tie fixture와 tolerance를 고려한다.

품질 관문와 training correctness gate를 구분한다. next-step oracle이 맞아도 모델 품질이 낮을 수 있고, 작은 quality score가 우연히 같아도 optimizer state가 틀릴 수 있다. 두 gate는 같은 checkpoint subject를 참조하지만 독립 결과를 낸다.

evaluation artifact에는 harness source, dataset digest, prompt bytes, decoding config, raw prediction, normalized prediction, scorer와 summary denominator를 넣는다. 공개 benchmark의 test answer를 training pipeline에 유출하지 않는다. golden fixture와 실제 benchmark를 같은 이름으로 부르지 않는다.

단일 GPU 성능 기준선을 정직하게 측정한다. 성능은 correctness gate를 통과한 조합에서만 비교한다. warm-up, compile, steady-state와 checkpoint·evaluation 구간을 나눈다. 유효 token/s와 계산 token/s, update/s, step time p50·p95, peak allocated/reserved와 profiler overhead를 기록한다.

한두 step 평균은 clock·cache와 background noise에 민감하다. 고정 shape sequence를 여러 번 반복하고 시작·종료 synchronization의 의미를 확인한다. CUDA event와 wall-clock이 측정하는 범위를 구분한다. `torch.cuda.synchronize()`를 hot loop에 넣어 실제 overlap을 파괴하지 않는다.

GPU model, power·clock state, driver, CUDA, framework, kernel backend와 compile cache 상태를 manifest에 넣는다. 다른 GPU의 결과를 같은 baseline으로 쓰지 않는다. utilization이 높다는 이유로 효율이 좋다고 판단하지 않고 kernel timeline과 HBM·host gap을 본다.

memory 최적화가 처리량과 수치에 주는 효과를 함께 표로 만든다. gradient checkpointing, attention backend, precision, optimizer 구현과 batch/accumulation을 하나씩 바꾼다. OOM이 사라진 대신 objective denominator나 effective batch가 달라지면 동일 비교가 아니다.

first-divergence debugger를 단계적으로 실행한다. 두 run이 다르면 먼저 input identity를 비교한다. canonical config, source/environment, GoldenBatchID, rendered bytes, token·label·mask가 같지 않으면 forward를 조사하지 않는다. 같다면 initial parameter와 buffer digest를 비교한다.

forward는 embedding, attention, MLP, final hidden, logits와 loss boundary 순으로 찾는다. 최초 차이 tensor에서 shape, dtype, finite, max absolute·relative error와 norm을 출력한다. 다음으로 해당 module의 input과 option, kernel path를 좁힌다. 전체 tensor dump는 작은 fixture에서만 사용한다.

forward와 loss가 같다면 gradient boundary를 비교한다. trainable ownership, scaled/unscaled, accumulation microstep과 clipping 전후를 구분한다. gradient가 같고 delta가 다르면 optimizer group, LR, moment·step과 fused path를 본다. delta가 같고 다음 forward가 다르면 parameter application, buffer·RNG와 data cursor를 조사한다.

resume 차이는 checkpoint component digest부터 본다. 저장 state가 같지만 load 뒤 다르면 reader mapping과 dtype/device conversion 문제다. load state도 같은데 다음 batch가 다르면 sampler·worker와 callback RNG를 본다. error를 “nondeterministic CUDA”로 넘기기 전에 이 결정적 경계를 모두 제거한다.

debug bundle은 두 run의 identity, first different boundary, actual/expected summary, 관련 소스 좌표와 최소 reproduction command를 담는다. PASS boundary도 기록해 조사 범위를 증명한다. 수정 뒤 같은 negative fixture가 차이를 검출하고 정상 fixture가 통과하는지 확인한다.

golden run을 여덟 단계로 실행한다. 1단계: 입력을 resolve한다. 27장의 승인된 evidence bundle에서 model, tokenizer, dataset fixture, source와 environment digest를 읽는다. floating revision이나 최신 alias를 학습 함수 안에서 다시 resolve하지 않는다. cache hit에도 digest와 revocation policy를 확인한다. 실제 local path와 immutable ArtifactID의 mapping을 invocation manifest에 쓴다.

2단계: dry initialization을 수행한다. GPU allocation 전에 config를 canonicalize하고 unknown option, incompatible dtype·device, 누락 file과 license/policy failure를 검출한다. model을 만들되 optimizer 전 trainable parameter inventory를 기록한다. adapter injection 뒤 inventory diff와 optimizer group mapping을 검증한다.

3단계: GoldenBatchID를 만든다. dataset row를 normalize·render·tokenize·collate하고 각 경계 artifact를 검사한다. token·label·mask shape, valid denominator와 truncation을 expected fixture와 exact 비교한다. dataloader iterator를 여러 번 새로 만들어 첫 row sequence가 같은지 확인한다.

4단계: reference forward/backward를 실행한다. eager FP32 또는 정한 reference path에서 architecture boundary, loss numerator/denominator와 gradient를 기록한다. 작은 scalar·matrix fixture는 손계산과 finite difference로 검산한다. parameter가 아직 변하지 않았다는 digest를 확인한다.

5단계: candidate update를 commit한다. 선택한 autocast·scaler·clip·optimizer·scheduler 순서를 실행한다. attempted와 committed update를 분리하고 overflow negative control을 포함한다. parameter delta, moment, LR와 scaler state를 reference와 비교한다. observer on/off parity도 검사한다.

6단계: checkpoint를 장애와 함께 저장한다. complete generation 하나를 만든 뒤 다음 generation의 여러 kill point를 시험한다. loader가 마지막 complete를 고르는지, digest mismatch를 거부하는지 본다. clean process에서 resume해 next batch와 next update를 uninterrupted branch와 비교한다.

7단계: evaluation과 성능을 측정한다. 같은 checkpoint subject에서 golden quality fixture를 실행하고 train mode·RNG를 복원한다. correctness 통과 뒤 steady-state throughput과 memory를 측정한다. 실행하지 않은 compile·dtype·optimizer 조합은 `NOT-RUN`으로 남긴다.

8단계: evidence package의 digest와 지원 범위를 확정한다. canonical config, environment, data boundary, numerical oracle, checkpoint lineage, evaluation, performance, fault result와 소스 원장의 digest를 index로 묶는다. status와 exception을 비워 두지 않는다. 29장은 이 index를 input으로 받아 분산 경계만 추가한다.

장애 주입을 정상 control과 쌍으로 만든다. **tokenizer drift.** tokenizer revision 또는 special-token map을 바꾼 child config를 사용한다. 최초 차이는 TokenFixtureID에서 발생해야 하고 training 시작 전 gate가 막아야 한다. 원 cache를 덮어쓰지 않는다. 정상 digest로 돌아왔을 때 control이 다시 통과해야 한다.

label shift. fixture copy에서 label을 한 위치 어긋나게 한다. rendered bytes와 token은 같고 label/mask에서 처음 달라져야 한다. loss가 달라진다는 결과만 보지 않고 boundary detector가 정확한 위치를 보고하는지 확인한다.

overflow. 선택 parameter 또는 loss에 단일 step nonfinite를 주입한다. unscale·found_inf가 검출하고 parameter·moment·scheduler commit을 막아야 한다. scaler의 expected state 변화와 attempted counter는 허용한다. hook은 실험 뒤 반드시 제거한다.

partial checkpoint. 새 staging generation의 optimizer write에서 실패시킨다. 기존 complete generation은 read-only로 유지한다. restart는 partial을 거부하고 이전 generation을 선택한다. cleanup은 incident artifact를 보존한 뒤 수행한다.

sampler cursor 누락. checkpoint에서 data state만 제거한다. model load는 성공할 수 있으나 resume 첫 BatchID assertion이 실패해야 한다. loss curve가 자연스럽다는 이유로 허용하지 않는다. 수정 뒤 next row sequence까지 control과 비교한다.

optimizer moment 교체. 같은 shape의 잘못된 moment 또는 parameter mapping을 넣는다. load schema가 잡지 못하면 next delta oracle이 실패해야 한다. distinctive tiny parameters로 어떤 mapping이 바뀌었는지 진단한다.

disk와 logger failure. checkpoint writer short write와 tracker backend timeout을 별도로 주입한다. checkpoint 실패가 run logging 성공으로 덮이지 않고, logger 실패가 optimizer commit을 오염시키지 않아야 한다. 정책상 학습 지속·중단 결과를 exact assertion으로 쓴다.

compile cache 오염. 다른 config·architecture에서 만든 cache entry를 제공한다. cache key·guard가 재사용을 거부하거나 candidate output이 reference parity에서 실패해야 한다. 실제 compiler cache를 공유 환경에서 파괴하지 않고 별도 temporary root를 쓴다.

각 fault에는 trigger, expected first divergence, detector, safe action, cleanup, recovery oracle와 artifact가 있다. fault가 검출됐다는 사실만 아니라 올바른 계층에서 검출되고 정상 control을 다시 복원했다는 사실까지 성공 조건이다.

logging과 callback이 상태를 바꾸지 않게 한다. training callback은 log, evaluation, checkpoint, early stopping과 profiler를 연결하며 순서와 RNG에 영향을 줄 수 있다. callback list와 호출 event를 canonical config에 넣는다. 같은 priority나 registration order가 바뀌면 checkpoint와 scheduler timing이 달라질 수 있다.

tensor `.item()`을 매 microstep 호출하면 device synchronization을 만들 수 있다. loss logging은 필요한 reduction과 update boundary에서 수행하고 비동기 queue를 사용하되 event order와 durability를 보존한다. queue full 정책이 학습 thread를 block할지 debug event를 drop할지 정한다.

tracker resume는 checkpoint resume와 분리한다. run identity, attempt와 checkpoint generation을 연결한다. 같은 run ID로 오래된 checkpoint를 load하면 출시 관문가 실패해야 한다. 새 run ID가 생겨도 올바른 checkpoint parent relation을 기록할 수 있다.

callback이 evaluation에서 model mode를 복원하지 않거나 공용 RNG를 소비하면 다음 training state가 달라진다. callback 전후 train/eval, grad enabled, RNG summary와 data cursor를 selected fixture에서 비교한다. hook reference가 activation graph를 붙잡아 memory가 증가하지 않는지 본다.

early stopping은 metric event와 optimizer state의 어느 시점을 checkpoint하는지 명시한다. best metric이 generation N에서 나왔는데 N+1 weight를 best alias로 연결하지 않는다. asynchronous evaluation이면 evaluated subject와 current training generation이 다를 수 있으므로 digest로 구분한다.

오류 메시지를 다음 행동으로 연결한다. `input mismatch`는 config, row, rendered bytes, tokenizer와 collator digest를 출력한다. 어느 row·position에서 token/label이 처음 달랐는지 보여 준다. 원문 전체를 무조건 log하지 않는다. 민감 fixture는 stable ID와 제한 artifact를 사용한다.

`forward mismatch`는 마지막 정상 boundary와 첫 비정상 boundary, shape·dtype·error summary를 출력한다. kernel 이름만 원인으로 쓰지 않고 input option과 actual backend를 기록한다. eager fallback으로 차이가 사라지는지 다음 최소 실험을 제안한다.

`gradient mismatch`는 parameter ownership, accumulation microstep, scaler stage, clip 전 norm과 observer 상태를 보여 준다. 전체 parameter dump 대신 첫 mismatch와 group summary를 낸다. zero, None과 nonfinite를 구분한다.

`optimizer mismatch`는 parameter group, LR, step, moment와 decay contribution을 비교한다. fused/foreach/scalar actual path와 AMP skip을 표시한다. checkpoint에서 복원한 state라면 source generation과 mapping을 포함한다.

`resume mismatch`는 selected/expected generation, component digest와 첫 다른 BatchID·RNG·state를 보고한다. partial generation과 lineage conflict를 구분한다. 자동으로 최신 directory를 강제 load하지 않는다.

`performance regression`은 correctness가 통과했고 workload bucket이 같은지 먼저 표시한다. warm-up, compile, profiler, checkpoint 구간과 steady state를 나눈다. peak memory와 throughput만 보고 수치 divergence를 숨기지 않는다.

진단 메시지는 fixture의 expected error로 test한다. 문구 전체에 결합하기보다 stable reason code와 핵심 field를 검사한다. 사람에게는 소스 좌표와 playbook link를 제공하고 machine에는 structured payload를 준다.

source upgrade를 golden diff로 승인한다. PyTorch, Transformers, PEFT, tokenizer나 checkpoint library revision을 올릴 때 expected artifact를 먼저 덮어쓰지 않는다. old/new source 좌표의 semantic anchor를 비교하고 어떤 boundary가 달라질 수 있는지 change hypothesis를 적는다.

parser default나 option rename은 canonical config diff에서 잡는다. tokenizer 변경은 rendered/token fixture, attention·loss 변경은 forward boundary, optimizer 변경은 delta·state, serializer 변경은 bundle schema와 resume oracle을 우선 실행한다. dependency closure 변화는 environment·SBOM에서 확인한다.

old/new를 같은 immutable inputs와 clean cache에서 실행한다. 최초 divergence가 hypothesis와 맞는지 본다. 더 이른 boundary에서 차이가 나면 upgrade 범위를 다시 조사한다. 차이가 없더라도 실제 새 path가 실행됐는지 branch marker와 소스 리비전을 확인한다.

expected baseline 변경에는 old/new numerical diff, 성능·memory, fault fixture와 reviewer 승인이 필요하다. “새 버전이므로 값 변경”은 이유가 아니다. tolerance 확대는 수치 분석과 downstream evaluation 근거를 요구한다. 지원하지 않는 option 조합은 명시적으로 제거한다.

upgrade 뒤 기존 checkpoint load와 새 checkpoint의 old reader 호환성을 정책에 따라 시험한다. 양방향 호환을 약속하지 않는다면 migration tool, rollback 제한과 last compatible generation을 기록한다. production rollout 전 단일 GPU golden이 이 경계를 닫아야 한다.

## 28.13 최소 runner에서 상태 소유권을 증명한다

수치 oracle이 준비됐다고 재현성이 자동으로 생기지는 않는다. runner가 RNG, dataloader cursor, gradient, optimizer moment와 checkpoint generation 가운데 무엇을 소유하는지 드러내야 한다. 이 절은 utility 함수를 나열하지 않고 write/read 대칭과 다음 update의 동등성을 기준으로 구현을 읽는다.

아래 의사 코드는 framework별 장식을 걷어 낸 update 순서다. 실제 구현에서는 각 호출의 고정 revision과 state observer를 연결한다.

```python
optimizer.zero_grad(set_to_none=True)
window = begin_accumulation(batch_ids, committed_step)

for microbatch in window:
    with autocast(device_type="cuda", dtype=amp_dtype):
        outputs = model(**microbatch.inputs)
        numerator, denominator = loss_parts(outputs, microbatch.labels)
        scaled_objective = numerator / window.total_valid_tokens
    scaler.scale(scaled_objective).backward()

scaler.unscale_(optimizer)
grad_norm = clip_grad_norm_(trainable, max_norm, error_if_nonfinite=True)
before = snapshot_commit_state(model, optimizer, scheduler)
scaler.step(optimizer)
scaler.update()
committed = detect_parameter_commit(before, model, optimizer)
if committed:
    scheduler.step()
record_update(window, committed, grad_norm)
```

이 코드는 그대로 복사할 recipe가 아니라 검토할 책임 목록이다. `window.total_valid_tokens`를 microbatch loop 전에 알 수 없는 streaming path라면 numerator gradient를 어떻게 scaling할지 별도 설계가 필요하다. 모든 microbatch의 valid count를 먼저 계산하거나 gradient를 누적한 뒤 global denominator에 맞게 scale하는 방법을 검증한다.

`detect_parameter_commit`을 parameter 전체 비교로 매번 구현하면 비싸다. golden fixture에서는 exact digest와 state를 비교하고 production observer에서는 scaler의 found-inf·optimizer step event와 제한 checksum을 결합한다. 단순히 `scaler.step`이 return했다는 사실을 commit으로 해석하지 않는다. optimizer의 반환값은 commit boolean 계약이 아닐 수 있다.

clipping에서 `error_if_nonfinite=True`를 선택하면 nonfinite를 즉시 exception으로 승격할 수 있다. scaler가 overflow를 정상 skip하도록 설계한 FP16 path와 충돌할 수 있으므로 어느 계층이 nonfinite를 소유하는지 정한다. 옵션 하나를 무조건 권장하지 않고 fixture의 expected transition에 맞춘다.

scheduler는 committed update에서만 전진한다는 정책을 명시했다. 사용하는 framework wrapper가 scheduler를 loop iteration에서 자동 호출한다면 중복 호출을 피한다. canonical config는 scheduler owner를 기록한다. callback과 training loop 두 곳에서 같은 responsibility를 소유하지 않는다.

### 28.13.1 checkpoint writer와 reader의 대칭을 검사한다

writer가 저장한 state key와 reader가 요구하는 key를 machine schema로 비교한다. model, optimizer, scheduler, scaler, RNG, sampler 각각에 schema version, required/optional, serializer와 owner를 둔다. optional field가 없을 때 default가 trajectory를 바꾸면 resume artifact에서는 required로 승격한다.

```text
GenerationManifest
  generation: 12
  parent: 11
  committed_update: 400
  canonical_config_digest: ...
  components:
    model:      {digest, schema, size}
    optimizer:  {digest, schema, size, parameter_map_digest}
    scheduler:  {digest, schema, size}
    scaler:     {digest, schema, size}
    rng:        {digest, schema, size}
    sampler:    {digest, schema, size, next_batch_id}
  status: complete
```

reader는 `status`만 믿지 않고 child digest와 exact set을 확인한다. `parent`가 expected lineage와 다르면 branch conflict다. committed update와 scheduler·optimizer internal step이 맞는지 검사한다. canonical config가 다른 경우 resume, warm-start, transfer 가운데 어떤 mode인지 명시적으로 선택한다.

warm-start는 model weight만 가져오고 optimizer·data를 새로 시작할 수 있다. 이것을 resume 성공으로 기록하지 않는다. transfer는 architecture 일부가 달라 missing/unexpected key를 의도적으로 처리할 수 있지만 mapping recipe와 새 ArtifactID가 필요하다. 같은 loader API가 세 mode를 지원해도 상태 의미는 다르다.

reader가 dtype이나 device를 변환하면 load 후 state digest가 file digest와 달라질 수 있다. file-level integrity와 runtime tensor identity를 분리해 기록한다. optimizer moment의 dtype cast가 다음 delta에 미치는 오차를 golden fixture에서 측정한다.

### 28.13.2 RNG generator와 소비 순서를 해부한다

하나의 `seed=42`는 설명이 아니다. Python random, NumPy, CPU generator, 각 CUDA device generator, dataloader worker, sampler, augmentation와 generation sampling이 별도 상태를 가질 수 있다. 어떤 subsystem이 global generator를 공유하는지 source에서 확인한다.

RNG state는 값뿐 아니라 소비 순서에 의존한다. logging용 sample generation, evaluation, callback, checkpoint serialization과 profiler가 공용 RNG를 한 번 소비하면 이후 dropout mask가 모두 이동할 수 있다. subsystem별 generator를 주입하고 golden trace에는 중요 draw sequence 또는 state digest를 boundary마다 남긴다.

dataloader worker seed는 base seed, worker ID와 iterator generation에서 파생될 수 있다. persistent worker와 iterator 재생성, resume에서 동작을 fixture로 확인한다. dataset transform이 Python·NumPy·framework RNG를 섞어 쓰면 모두 제어하거나 결정적 transform으로 바꾼다.

CUDA kernel의 atomic reduction이나 algorithm selection은 seed가 같아도 bitwise 차이를 만들 수 있다. deterministic mode에서 unsupported operation이 발생하면 조용한 fallback보다 명시적 failure를 선호한다. numerical reproducibility 등급에서는 허용 오차와 first-divergence boundary를 기록한다.

### 28.13.3 단일 GPU 안의 동시성을 드러낸다

GPU가 하나여도 CPU data workers, prefetch thread, async device copy, CUDA streams, compilation worker, checkpoint writer와 logger가 동시에 움직인다. “단일”은 rank와 device가 하나라는 뜻이지 순차 실행이라는 뜻이 아니다.

prefetch는 어느 batch까지 읽었는지와 checkpoint sampler cursor의 차이를 만든다. durable cursor는 fetched, yielded, committed 가운데 어느 시점을 뜻하는지 정한다. 장애 뒤 prefetch된 미소비 batch를 재생할지 버릴지 정책이 필요하다. BatchID ledger로 중복·누락을 측정한다.

non-blocking H2D copy는 pinned host buffer lifetime과 stream dependency를 요구한다. loader가 buffer를 재사용하기 전에 copy가 완료되어야 한다. golden fixture는 content checksum과 stream event를 사용해 오염을 찾는다. 성능 옵션이 correctness 경계를 추가한다는 예다.

async checkpoint가 parameter snapshot을 만들고 있는 동안 다음 update가 진행되면 buffer ownership이 명확해야 한다. copy-on-write, staging copy 또는 optimizer pause 가운데 실제 방식을 기록한다. future failure가 늦게 도착해 이미 여러 step이 전진했을 때 RPO와 중단 정책을 적용한다.

logger queue는 out-of-order event와 shutdown loss를 만들 수 있다. event-time step과 ingestion sequence를 분리한다. process 종료 시 drain이 실패하면 telemetry incomplete를 기록하지만 checkpoint complete와 혼동하지 않는다. observer thread가 GPU tensor reference를 오래 보존하지 않는지 memory fixture로 본다.

OOM을 phase별로 진단한다. OOM event에는 phase, allocation request size, current/peak allocated·reserved, inactive split, batch shape, option과 observer 상태를 남긴다. 예외 문자열만 저장하지 않는다. 동일 checkpoint와 GoldenBatchID에서 재현해 data variability를 제거한다.

forward OOM이면 activation, attention workspace와 input shape를 본다. backward OOM이면 saved activation, recompute, gradient와 temporary buffer를 본다. optimizer OOM이면 moment initialization과 foreach/fused workspace를 본다. checkpoint OOM이면 state consolidation·staging copy를 본다. 같은 peak 숫자라도 owner가 다르다.

해결 후보는 microbatch, accumulation, sequence/packing, checkpointing, attention backend, optimizer implementation, precision과 offload다. 하나씩 바꾸고 objective denominator, effective batch, numerical output, step time과 peak를 비교한다. batch를 줄여 OOM만 없앤 뒤 처리량·optimization 변경을 보고하지 않는 것은 회귀 수정이 아니다.

allocator fragmentation 가설은 reserved-minus-allocated 하나로 확정하지 않는다. memory snapshot에서 segment/block과 allocation stack, 동일 shape 순서의 retry를 본다. 살아 있는 tensor reference가 증가한다면 hook, list, loss accumulation과 retained graph를 찾는다. `empty_cache`가 일시적으로 공간을 반환해도 생존 객체 원인은 남는다.

OOM regression test는 문제 shape, 바로 아래 shape와 정상 혼합 shape를 포함한다. 수정 전 failure와 수정 후 pass, golden numerical parity와 overhead budget을 모두 요구한다. 관측 도구를 켜야만 OOM이 나거나 사라지는 observer effect도 검사한다.

결과 표를 의사결정 가능한 형태로 만든다. 표의 각 행은 config combination이 아니라 하나의 가설과 evidence다. reference eager FP32, BF16, gradient checkpointing, fused optimizer, compiled path와 async checkpoint를 reference에서 한 edge씩 추가한다. edge마다 예상 state change, first divergence, numerical tolerance, memory·throughput effect와 새 fault fixture를 쓴다.

| 변경 | 예상 최초 차이 | correctness oracle | 운영 효과 | 신규 실패면 |
|---|---|---|---|---|
| BF16 autocast | 첫 dtype 전환 연산 | boundary 오차 예산 | memory·tensor-core 경로 | overflow·fallback |
| accumulation 증가 | update clock·microbatch RNG | global-token gradient | peak 감소 가능 | denominator·clock |
| checkpointing | recompute boundary | gradient parity | activation↓ compute↑ | RNG·hook 반복 |
| fused AdamW | parameter delta rounding | two-step state oracle | launch·workspace 변화 | AMP 전달·serialize |
| compile | 첫 compiled region | eager boundary parity | compile/steady 분리 | graph break·recompile |
| async save | durability event | resume next-step | stall 감소 가능 | torn snapshot·future |

최종 열에는 `PASS`, `FAIL`, `NOT-RUN`, `NOT-APPLICABLE`과 artifact link를 둔다. 실행하지 않은 조합을 reference 결과에서 일반화하지 않는다. 여러 옵션을 동시에 켠 production candidate는 reference에서 이어지는 모든 edge가 검증된 뒤 조합 interaction fixture를 추가한다.

결과 설명은 가장 빠른 조합을 선언하는 데서 끝나지 않는다. memory headroom, numerical grade, checkpoint RPO, profiler overhead와 recovery 성공을 함께 본다. golden run의 목적은 benchmark 우승이 아니라 이후 scale-out에서 바뀐 경계를 정확히 고립할 수 있는 기준점을 만드는 것이다.

실행 결과를 읽는 세 가지 종단 사례. 사례 A: loss가 같지만 parameter가 움직이지 않는다. 첫 batch forward와 loss는 reference와 같고 backward도 예외 없이 끝났다. 그러나 trainable adapter checksum이 두 step 뒤에도 같았다. `grad_present_count`를 보니 injection된 parameter 일부의 gradient가 `None`이었다. optimizer group manifest에는 다른 이름의 parameter만 들어 있었다.

원인은 target module rename 뒤 오래된 name filter가 optimizer group을 구성한 것이었다. model에 adapter는 존재해 forward에 영향을 주었지만 optimizer가 소유하지 않았다. 수정은 이름 하나를 바꾸는 데서 끝나지 않았다. injection inventory, optimizer parameter-map digest와 trainable gradient/delta assertion을 regression fixture에 추가했다.

이 사례는 loss 감소만으로 update correctness를 판단할 수 없음을 보여 준다. base model과 dropout 때문에 짧은 구간 loss가 움직일 수 있고, adapter 초기화 구조상 한 factor gradient가 첫 step에 0일 수도 있다. 두 step의 factor별 expected gradient와 delta가 결정적 증거였다.

사례 B: resume 직후만 loss가 다르다. checkpoint component digest는 모두 일치했고 첫 BatchID도 같았다. forward의 embedding까지 같았지만 첫 dropout 뒤 activation이 달랐다. RNG state file은 존재했으나 checkpoint load 뒤 evaluation callback이 sample generation을 실행하며 global CUDA generator를 소비했다.

해결은 tolerance를 넓히는 것이 아니라 evaluation과 training generator를 분리하고 resume validation 순서를 바꾸는 것이었다. negative fixture는 callback을 삽입했을 때 first divergence detector가 dropout boundary를 가리키는지 확인한다. 수정 후에는 callback 유무와 관계없이 training RNG trace와 next delta가 같았다.

사례 C: 메모리는 줄었는데 step이 더 불안정해졌다. gradient checkpointing을 켠 뒤 peak allocated는 감소했지만 p95 step time과 run 간 variation이 커졌다. trace는 일부 shape에서 graph recompilation과 recompute 구간이 겹쳤고 callback hook이 두 번 artifact를 생성하는 것을 보였다. 평균 throughput 하나만 보면 작은 회귀처럼 보였지만 max latency가 checkpoint cadence와 만나 RPO를 악화시켰다.

hook을 recompute-aware하게 만들고 shape bucket을 고정한 뒤 numerical parity, peak와 p95를 다시 측정했다. memory 절감, compute 증가, compile cache와 observer invocation이라는 네 상태를 함께 봐야 원인을 설명할 수 있었다.

독자가 작성할 실행 기록의 형태. 실행 기록 첫 줄에는 `RunID`, `AttemptID`, `BundleID`, canonical config와 environment digest, hardware identity를 둔다. 이어 claimed reproducibility grade와 reference run을 적는다. 명령 문자열은 보조 정보이며 실제 resolved config가 authoritative하다.

data 표에는 raw fixture, rendered bytes, TokenFixtureID, BatchID, valid/padded token, truncation과 sampler next cursor를 둔다. model 표에는 initial state, trainable/frozen inventory, adapter mapping과 architecture boundary schema를 둔다.

update 표에는 attempted/committed step, microsteps, numerator/denominator, autocast·scaler stage, clip norm, optimizer group·moment, delta와 scheduler LR를 둔다. overflow fixture는 어떤 값이 멈추고 scaler 중 무엇이 변했는지 별도 행으로 쓴다.

checkpoint 표에는 generation, parent, component set·digest, writer attempt, complete/publish, kill point, loader selection과 resume oracle을 둔다. evaluation 표에는 evaluated subject, harness/data/decoding, raw result와 denominator를 둔다. performance 표에는 workload bucket, warm-up·steady, throughput, latency와 memory를 둔다.

마지막에는 failure injection별 expected/actual first divergence, detector, recovery와 clean control을 쓴다. 소스 원장는 사용한 exact revision과 semantic anchor, local execution 상태를 연결한다. 실행하지 않은 hardware path나 option은 빈칸 대신 `NOT-RUN`으로 표시한다.

29장 진입을 허용하는 gate. **입력 exact gate.** canonical config와 environment가 고정됐고 raw row에서 rendered bytes, tokens, labels, mask와 BatchID까지 exact fixture가 있다. resume 전후 sample stream의 다음 ID가 검증됐다.

수치 gate. eager reference의 forward, loss numerator/denominator, gradient와 AdamW 또는 선택 optimizer의 두-step delta가 손계산·reference와 맞는다. candidate dtype·compile·kernel은 선언한 numerical tolerance와 first-divergence 설명을 제시한다.

소유권 gate. trainable, frozen, shared와 conditional parameter가 optimizer group, gradient와 delta에서 일치한다. adapter base relation과 save/load mapping이 검증됐다. unused와 exact-zero gradient를 구분한다.

commit gate. accumulation, unscale, nonfinite, clipping, optimizer와 scheduler 순서가 fixture로 고정됐다. overflow에서 parameter·moment·scheduler가 원자적으로 멈추고 scaler의 예상 상태만 변한다. attempted와 committed clock을 구분한다.

durability gate. checkpoint가 model 외 모든 required training state를 포함하고 partial generation을 거부한다. clean process resume의 다음 batch·RNG·loss·delta와 LR이 uninterrupted control과 일치한다. retention과 async failure도 시험됐다.

관측 gate. observer on/off가 수치·memory·performance 예산 안에서 동일하고 first-divergence bundle이 입력, forward, gradient, optimizer, resume 차이를 올바르게 분류한다. tracker와 checkpoint identity가 분리된다.

실험 안전 gate. 모든 fault가 temporary root와 synthetic fixture 안에서 실행되고 cleanup 뒤 clean control이 통과한다. 실제 공유 disk를 채우거나 model cache를 덮어쓰지 않는다. 실행하지 않은 조합과 대규모 hardware 결과를 만들어내지 않는다.

일곱 gate가 모두 통과하면 29장은 world size, rank membership, process group, collective, shard와 remote storage라는 새 변수만 추가한다. 단일 GPU에서 이미 불명확한 denominator, optimizer clock이나 checkpoint를 다중 노드에서 디버깅하려 하면 차이를 통신 탓으로 돌리게 된다.

단일 GPU golden run의 최종 산출물. 완성된 golden package에는 재실행 가능한 작은 입력과 기대 상태가 있다. 독자는 token 하나가 embedding과 attention·MLP를 지나 logit과 loss numerator가 되고, backward gradient와 optimizer delta를 거쳐 checkpoint generation으로 commit되는 과정을 경계별로 확인할 수 있다.

각 옵션은 효과와 위험이 연결된다. batch·accumulation은 memory와 denominator·clock을, precision은 kernel과 scaler·오차를, checkpointing은 activation과 recompute·RNG를, compile은 steady performance와 specialization을, fused optimizer는 launch와 AMP·state schema를, async save는 stall과 snapshot 원자성을 함께 바꾼다.

이 package는 거대한 모델을 실제로 장시간 학습했다는 보고가 아니다. 고정 revision의 source와 test, synthetic fixture와 독자가 실행할 hardware gate를 구분한다. 측정하지 않은 GPU 수치와 지원 조합을 상상으로 채우지 않는다. 대신 실제 실행할 때 무엇을 기록하고 어떤 boolean으로 성공을 판정할지가 완전하다.

가장 중요한 산출물은 baseline 숫자 하나가 아니라 first-divergence map이다. 입력, forward, loss, gradient, optimizer, checkpoint와 resume 가운데 어디까지 같았는지를 말할 수 있다. 다음 장에서 분산 execution이 이 기준에서 갈라질 때 collective인지 shard인지 rank-local data인지 정확히 좁힐 수 있다.

assertion을 모호하지 않은 문장으로 쓴다. `loss가 정상이다` 대신 `abs(actual_loss - reference_loss) <= atol + rtol * abs(reference_loss)`를 쓰고 dtype, reduction과 numerator/denominator를 함께 출력한다. exact fixture에는 tolerance를 사용하지 않는다. TokenID, mask, BatchID, parameter ownership과 checkpoint digest는 byte 또는 정수 exact 비교다.

`gradient가 비슷하다` 대신 parameter group별 max absolute·relative error, cosine, norm과 finite count를 기록한다. reference가 0에 가까울 때 relative error가 폭발하므로 absolute 기준을 함께 쓴다. aggregate norm 통과가 개별 tensor swap을 숨기지 않도록 distinctive fixture와 selected tensor exact 값을 둔다.

`optimizer가 한 번 실행됐다` 대신 committed update, parameter delta, moment, optimizer internal step과 scheduler LR을 비교한다. overflow fixture에는 `parameter_unchanged`, `moment_unchanged`, `scheduler_unchanged`, `scale_reduced_or_policy_expected`를 각각 assertion으로 둔다. 하나의 boolean에 묶어 어느 상태가 틀렸는지 숨기지 않는다.

`resume이 됐다` 대신 selected generation과 parent, component completeness, next BatchID, RNG state, next loss·delta와 LR을 비교한다. uninterrupted branch와 resume branch는 checkpoint 직전까지 같은 parent에서 fork한다. 같은 process cache를 공유하지 않는다. comparison artifact에는 두 branch의 RunID와 generation을 넣는다.

`성능이 유지됐다`는 같은 correctness, workload, warm-up과 environment에서 valid token/s와 p95 step, peak memory가 사전 budget 안이라는 뜻이다. 평균 throughput 하나로 판정하지 않는다. profiler capture가 켜진 sample과 꺼진 baseline을 분리한다.

assertion 실패는 actual·expected, identity와 last passing boundary를 출력한다. fixture가 기대한 negative failure라면 test PASS이고 정상 control에서 같은 failure가 나면 전체 FAIL이다. `xfail`이나 skip에는 issue, owner와 expiry가 필요하다.

작은 실행의 비용과 시간을 계획한다. golden suite는 빠르게 반복할 수 있어야 한다. exact token·loss·optimizer unit fixture, 수 step integration, checkpoint kill/resume, selected compile·precision과 short performance를 tier로 나눈다. source 변경마다 싼 tier를 실행하고 위험 경계 변경 때 관련 tier를 추가한다.

hardware 없는 정적 검토와 CPU synthetic test, 단일 CUDA device test를 상태로 분리한다. CUDA path를 CPU 결과로 승인하지 않는다. 반대로 대규모 모델이나 장시간 run이 없어도 parser, state ordering, checkpoint transaction과 tiny numerical oracle은 깊게 검증할 수 있다.

GPU run에는 최대 model/sequence/batch, wall-time, storage와 fault blast radius 예산을 둔다. OOM·disk fault는 격리된 temporary resource에서 수행한다. watchdog가 runaway compile, deadlock이나 storage retry를 종료하고 partial artifact를 보존한다.

CI는 모든 GPU 조합을 매 commit 돌리지 못할 수 있다. support matrix와 위험 기반 sampling을 사용하고 마지막 실행 revision·age를 표시한다. 오랫동안 실행되지 않은 hardware gate는 과거 PASS가 아니라 stale evidence다. release 전 required matrix를 갱신한다.

baseline update는 suite 실패를 없애기 위한 일괄 재생성이 아니다. semantic change, expected first divergence, old/new diff, reviewer와 migration을 요구한다. performance baseline은 장비 상태와 noise 분포를 포함해 rolling median 하나에 자동 적응하지 않는다.

실습 제출물과 재생 조건. 독자가 제출할 첫 파일은 resolved config와 environment manifest다. 둘째는 네 개 안팎의 golden row와 rendered/token/label fixture다. 셋째는 reference forward·loss·backward와 optimizer two-step oracle이다. 넷째는 checkpoint generation 두 개와 partial generation 하나, uninterrupted/resume comparison이다.

다섯째는 fault matrix다. tokenizer drift, label shift, overflow, sampler omission, optimizer state mismatch, partial write와 logger failure 중 최소 핵심 fixture가 있다. 각 행에 expected first divergence, reason code, recovery와 clean control artifact를 둔다.

여섯째는 option edge 표다. reference에서 precision, checkpointing, compile, optimizer와 async save를 하나씩 추가하고 `PASS/FAIL/NOT-RUN`을 표시한다. speed나 memory 숫자에는 exact environment와 measurement method가 붙는다.

일곱째는 소스 원장다. tokenizer/collator, model forward/loss, scaler, clipping, optimizer, scheduler, serializer·reader의 path, symbol, revision과 semantic anchor를 기록한다. 문서 설명과 실제 실행 source가 같은 revision인지 확인한다.

마지막 evidence index는 모든 child digest와 생성 command, test status를 묶는다. 다른 사람이 clean workspace에서 index 하나로 fixture를 복원하고 같은 assertion을 실행할 수 있어야 한다. 이 제출물이 갖춰지면 golden run은 개인 노트북 성공담이 아니라 다중 노드 확장과 release가 의존할 수 있는 기술적 기준선이 된다.

단일 GPU 기준선의 독립 검토. 인수자는 먼저 evidence index에서 임의의 child를 골라 digest와 생성 command를 다시 확인한다. canonical config에서 parser default, environment와 artifact resolution까지 역추적한다. 명령행에 없던 default가 runtime 의미를 바꾸지 않았는지, cache가 다른 revision을 반환하지 않았는지 본다.

GoldenBatchID 하나를 raw row까지 따라가 rendered bytes, token, label, mask와 denominator를 손으로 검산한다. loss scalar만 비교하지 않고 logit-target shift와 ignored 위치를 확인한다. 같은 batch를 reference와 candidate가 실제로 소비했는지 sample-stream ledger를 본다.

trainable parameter 하나와 frozen parameter 하나를 골라 forward ownership, gradient, optimizer group과 delta를 추적한다. AdamW라면 moment·bias correction·decay, Muon이라면 matrix transform과 별도 parameter group을 reference 계산과 비교한다. adapter save가 base digest와 mapping을 보존하는지 확인한다.

overflow event 하나에서는 attempted와 committed clock, scaler, parameter, moment, scheduler를 순서대로 본다. 정상 update 하나에서는 clipping 전 norm, clip coefficient, delta와 다음 LR을 확인한다. logging과 evaluation callback을 켜고 끈 control이 같은 학습 state를 내는지 비교한다.

checkpoint generation 하나를 staging write부터 complete manifest와 latest publish까지 따라간다. 중간 kill fixture에서 loader가 이전 generation을 선택하는지 본다. clean process resume의 next BatchID, RNG, loss와 delta가 uninterrupted branch와 일치해야 한다. tracker run resume는 별도 identity로 확인한다.

성능 표는 correctness가 통과한 동일 workload만 비교한다. warm-up과 compile, checkpoint·profiler window를 제외하거나 별도 보고했는지 본다. valid·compute token/s, p95 step, peak memory와 environment가 함께 있어야 한다. 실행하지 않은 option 조합은 지원 범위에서 제외한다.

마지막으로 일부러 tokenizer digest, optimizer moment와 checkpoint manifest를 각각 한 번씩 변조한다. 입력, optimizer, durability gate가 서로 다른 reason code와 first divergence를 내야 한다. 변조를 제거한 clean control이 다시 통과하고 temporary artifact와 hook이 남지 않았는지 확인한다.

이 검토가 중요한 이유는 단일 GPU가 단순해서가 아니다. 분산 시스템이 추가되기 전에 학습의 의미와 durable state를 한 사람이 끝까지 추적할 수 있는 마지막 경계이기 때문이다. 여기서 불명확한 상태는 GPU가 늘면 rank, collective와 shard 뒤에 숨는다. 여기서 exact와 numerical 경계를 분명히 하면 29장의 장애는 새로 추가된 분산 변수로 한정할 수 있다.

인수 기록은 판정표의 모든 `PASS`를 실행 artifact와 연결한다. source만 읽은 항목은 `source-confirmed`, upstream test만 확인한 항목은 `upstream-test-confirmed`, local synthetic fixture를 실행한 항목은 `local-synthetic-executed`, 실제 CUDA 장비 검증이 남은 항목은 `hardware-pending`으로 구분한다. 이 책의 정적 분석 결과를 실행 측정값처럼 표현하지 않는다.

`hardware-pending`은 실패를 숨기는 빈칸이 아니라 독자가 자기 환경에서 수행할 command, 예상 invariant, 허용 오차와 artifact 위치를 명시한다. 실행 뒤에는 GPU·driver·CUDA와 loaded kernel identity를 추가하고 결과를 새 evidence generation으로 봉인한다. 기존 정적 evidence를 덮어쓰지 않는다.

지원 범위는 가장 약한 gate를 따른다. eager FP32와 BF16만 실행했다면 compile·FP16·fused optimizer까지 지원한다고 쓰지 않는다. synchronous checkpoint만 fault test했다면 async durability는 미검증이다. 명시적인 경계는 책을 덜 완성돼 보이게 하는 것이 아니라 독자가 무엇을 믿고 무엇을 직접 검증해야 하는지 알려 주는 신뢰 장치다.

이제 golden package는 input identity, 수치 oracle, update commit, durability, 관측과 성능의 여섯 축으로 구성된다. 다음 장은 각각에 rank membership, collective sequence, shard ownership, network와 remote storage를 추가한다. 새 축에서 first divergence가 생겼을 때 단일 GPU package를 대조군으로 사용한다.

최종 서명 전에는 다른 검토자가 fixture 하나를 처음부터 재실행한다. 작성자의 shell history나 기존 cache 없이 evidence index만 사용한다. 재실행 결과가 같아도 생성 과정에서 선언되지 않은 network·local file 접근이 없었는지 audit한다. 차이가 나면 숫자를 평균내지 말고 최초 다른 artifact와 boundary를 찾는다. 검토자 identity와 새 결과 digest를 package에 추가한다. 이 독립 재실행까지 있어야 golden이라는 이름이 단순히 작성자 환경에서 한 번 성공했다는 뜻을 벗어난다.

남은 예외에는 책임자, 기술적 이유, 영향받는 option과 만료 시점을 적는다. 만료 뒤 자동으로 gate를 다시 실패시키고 재검증 없이는 지원 범위를 유지하지 않는다. 이 원칙까지 인수자가 확인한다.

검증 기록은 불변 artifact로 보존한다.

실행 가능한 계약은 명령행보다 먼저 입력을 고정한다. golden run의 시작점은 `python train.py`가 아니라 실행 입력의 폐쇄다. runner가 읽는 canonical config, model·tokenizer bundle root, dataset generation, GoldenBatch fixture, environment manifest, 소스 리비전과 expected oracle을 하나의 RunSpec으로 묶는다. 명령행은 RunSpec을 가리키고 허용된 override만 받는다. parser default가 바뀌어 같은 명령이 다른 의미를 갖는 일을 막기 위해 parse 뒤 resolved config 전체를 canonical serialization해 digest를 계산한다.

RunSpec의 필수 항목에는 model construction, tokenizer·template, sequence와 padding, batch·accumulation, optimizer group, precision, RNG, checkpoint, evaluation, logging과 profiler가 있다. `null`, 누락, 자동 추론을 구분한다. 예를 들어 `dtype=null`이 “model config에서 선택”인지 “framework auto”인지 명시한다. `device=cuda`도 실제 device index, compute capability, driver와 loaded kernel identity로 resolve한다.

입력 계약은 network와 cache 정책도 포함한다. golden run은 immutable material을 미리 검증한 offline mode를 기본으로 삼고, network가 필요하면 요청 URI와 기대 digest를 기록한다. cache hit가 다른 revision을 숨기지 않도록 실제 열린 path와 digest를 evidence에 넣는다. 27장에서 승인한 subject와 runner가 소비한 subject가 다르면 forward를 시작하기 전에 실패한다.

실행 전 validator는 상호 모순을 잡는다. effective batch는 `micro_batch × accumulation`이고 단일 GPU이므로 world size는 1이다. `micro_batch=2`, `accumulation=8`이면 update당 sequence 16개다. 각 sequence가 padding 전 1,024 token이라도 valid token이 각각 700이라면 objective denominator는 16,384가 아니라 valid label 합이다. RunSpec은 sample count, padded token, valid token과 supervised token의 네 clock을 따로 정의한다.

종료 조건도 미리 쓴다. 정확히 committed optimizer update 4회, attempted micro-step 34회, evaluation 1회, complete checkpoint generation 2개처럼 state transition으로 표현한다. wall-clock timeout은 안전 장치이지 학습 진행량의 정의가 아니다. overflow 때문에 update가 skip되면 attempted와 committed가 달라질 수 있다. scheduler가 어느 clock을 쓰는지 assertion으로 둔다.

runner 반환 코드는 성공 하나로 끝나지 않는다. `INPUT_IDENTITY_MISMATCH`, `FIXTURE_MISMATCH`, `NONFINITE_FORWARD`, `GRADIENT_ORACLE_FAILURE`, `UPDATE_COMMIT_FAILURE`, `CHECKPOINT_INCOMPLETE`, `RESUME_DIVERGENCE`, `PERFORMANCE_BUDGET_FAILURE`처럼 최초 실패 경계를 나타낸다. 후속 예외가 원인을 덮지 않도록 failure bundle을 먼저 원자 기록하고 cleanup한다.

GoldenBatch를 byte에서 label denominator까지 검산한다. 작은 batch는 자연스러움보다 구별 가능성이 중요하다. 서로 다른 길이, special token, Unicode, 빈 assistant turn과 padding 경계를 포함하되 라이선스나 개인정보가 없는 합성 row를 쓴다. 각 row에는 stable RowID, raw UTF-8 byte digest, rendered byte, TokenID, attention mask, position, label과 loss mask가 있다. tokenizer decode만으로 원문 동일성을 판단하지 않는다. normalization과 whitespace가 비가역일 수 있다.

causal LM의 shift를 손으로 계산하자. token이 `[11,12,13,14,0,0]`, pad id가 0이고 assistant supervision이 token 13과 14의 예측에만 적용된다고 하자. model input 위치 `t`의 logit은 다음 token `t+1`을 예측한다. label 배열을 `[ignore,13,14,ignore,ignore,ignore]`처럼 구성했다면 supervised denominator는 2다. label을 한 칸 잘못 밀어 `[ignore,ignore,13,14,ignore,ignore]`로 만들면 scalar loss는 우연히 비슷해도 다른 조건부 확률을 학습한다.

두 sample의 supervised token 수가 2와 6이고 loss numerator가 각각 3.0과 4.2라면 올바른 token-mean loss는 `(3.0+4.2)/(2+6)=0.9`다. sample loss를 먼저 평균하면 `(3.0/2 + 4.2/6)/2=1.1`이다. gradient accumulation에서 micro-batch별 mean을 그대로 더하고 accumulation 수로 나누면 token 수가 다른 micro-batch에 같은 가중치를 준다. golden oracle은 numerator와 denominator를 따로 누적한다.

packing fixture는 document boundary와 position policy를 드러낸다. 두 row를 한 sequence에 붙일 때 EOS를 넣는지, attention이 경계를 넘는지, position을 reset하는지와 loss mask를 기록한다. 같은 TokenID라도 block-diagonal attention과 full causal attention은 다른 model이다. collator output digest는 tensor byte와 shape·dtype·stride를 포함한다.

padding side도 반례를 만든다. right padding과 left padding은 mask가 맞으면 token 의미가 같아 보이지만 position id 생성, flash attention varlen metadata와 last-token selection이 달라질 수 있다. 두 fixture를 모두 지원한다고 주장하려면 각각 oracle을 실행한다. 하나만 지원하면 validator가 다른 padding side를 거부한다.

GoldenBatch 생성기와 소비자를 분리한다. 생성기는 고정 tokenizer와 template에서 fixture를 만들고 review 후 봉인한다. runner는 기대 fixture를 다시 계산해 exact 비교한 뒤 사용한다. source upgrade가 의도적으로 token을 바꾸면 기존 기대값을 자동 덮어쓰지 않고 old/new diff와 semantic 승인을 거쳐 새 fixture generation을 만든다.

forward oracle을 작은 행렬로 직접 푼다. 전체 Transformer를 손으로 계산하기 전에 한 token, 한 head, 작은 hidden dimension으로 경계를 만든다. hidden `x=[1,2]`, weight `W=[[1,0],[0,2]]`이고 row-vector convention `y=xW`라면 `y=[1,4]`다. 구현이 column-vector convention이나 transpose된 storage를 쓰면 같은 shape에서도 `[1,2]W^T`가 우연히 같을 수 있으므로 비대칭 weight `[[1,3],[2,4]]`와 `x=[1,2]`를 써 `y=[5,11]`처럼 구별한다.

두 token의 scalar attention을 보자. query `q=[1,0]`, key `k1=[1,0]`, `k2=[0,1]`, value `v1=[2,0]`, `v2=[0,4]`, head dimension 2라면 score는 `[1/√2,0]`이다. softmax 확률은 대략 `[0.670,0.330]`, output은 `[1.340,1.320]`이다. causal mask로 두 번째 key를 가리면 첫 token output은 정확히 `[2,0]`이어야 한다. mask를 softmax 뒤 곱하면 합이 1로 재정규화되지 않아 다른 값이 된다.

residual과 normalization 위치도 checkpoint compatibility를 결정한다. pre-norm block은 `x + F(norm(x))`, post-norm은 `norm(x+F(x))`다. config 이름만 같다고 보지 말고 fixed revision의 model forward에서 실제 호출 순서와 tensor shape를 소스 원장에 적는다. hook은 norm 입력, attention output, residual 합과 MLP output 같은 semantic boundary에 설치한다. 모든 module에 hook을 달아 trace noise와 memory를 폭발시키지 않는다.

최종 logit에서 cross entropy를 검산한다. target class가 1이고 logit `[1,2,0]`이면 log-sum-exp는 `2 + log(exp(-1)+1+exp(-2)) ≈ 2.4076`, loss는 `0.4076`이다. 안정 구현은 max를 빼 overflow를 막는다. target shift, ignore mask와 reduction 전 per-token loss를 저장하면 scalar 차이를 어느 token까지 좁힐 수 있다.

bf16 candidate는 FP32 reference와 exact equality를 요구하지 않는다. 각 경계에서 dtype, accumulator dtype과 tolerance를 정한다. first layer activation은 통과했는데 final logit이 범위를 넘는다면 누적 오차인지 특정 kernel인지 layer별 binary search한다. tolerance를 layer가 깊을수록 무조건 넓히지 않고 clean variants의 empirical envelope와 task 민감도를 근거로 정한다.

backward와 parameter 소유권을 같은 표에서 검사한다. backward oracle은 gradient norm 하나가 아니다. parameter마다 requires-grad, optimizer group, expected gradient 상태, selected element와 update 여부를 기록한다. 상태는 `absent`, `present-zero`, `present-nonzero`, `nonfinite`를 구분한다. frozen parameter의 gradient가 absent여야 하는데 zero tensor가 생기면 불필요한 graph·memory가 유지됐을 수 있다. conditional branch가 사용되지 않은 absent는 정상일 수 있으므로 fixture의 실행 경로와 함께 판단한다.

scalar 함수 `L=(wx-y)^2/2`에서 `w=2`, `x=3`, `y=5`라면 prediction 6, residual 1, loss 0.5, gradient `dL/dw=(6-5)×3=3`이다. finite difference `ε=10^-3`에서 `[L(w+ε)-L(w-ε)]/(2ε)`도 약 3이어야 한다. autograd와 finite difference가 다르면 backward graph나 reduction을 의심한다. bf16 finite difference는 quantization 때문에 작은 ε가 사라질 수 있어 FP32 reference parameter로 계산한다.

gradient accumulation 두 micro-step의 gradient가 3과 5이고 동일 denominator라면 mean objective gradient는 4다. loss를 accumulation 수로 나눠 각 backward를 하면 `.grad`에 `1.5+2.5=4`가 쌓인다. denominator가 2와 6 token으로 다르면 올바른 token mean은 numerator gradient 합을 총 token 8로 나눠야 한다. micro mean `(3/2 + 5/6)/2`는 다른 값이다. runner는 accumulation 전 numerator scale을 명시한다.

weight tying에서는 같은 storage를 두 parameter 이름이 참조할 수 있다. optimizer에 두 번 넣으면 decay와 update가 중복될 위험이 있다. object identity, storage pointer와 logical alias를 기록하고 unique trainable storage가 optimizer group에 정확히 한 번 등장하는지 검사한다. save/load 뒤 tying이 깨졌다면 값이 처음 같아도 다음 update부터 갈라진다.

LoRA fixture는 base weight가 frozen이고 A·B만 trainable인지 확인한다. 일반적인 초기화에서 B가 0이면 첫 step에 A gradient가 exact zero이고 B gradient는 nonzero일 수 있다. 이를 “A가 학습되지 않는다”는 버그로 오해하지 않는다. 두 번째 step에 B가 변한 뒤 A gradient가 생기는 oracle을 둔다. target module 하나를 일부러 누락한 negative fixture는 trainable parameter 수와 mapping gate에서 실패해야 한다.

gradient clipping은 unscale 뒤 수행한다. 두 parameter gradient가 `[3,4]`이고 max norm 2라면 total norm 5, clip coefficient는 `2/5=0.4`, clipped gradient는 `[1.2,1.6]`이다. FP16 scaled gradient에 먼저 clipping하면 scale에 따라 coefficient가 달라져 틀린 update가 된다. runner는 pre-unscale scaled norm, unscaled norm, coefficient와 post-clip norm을 구분해 기록한다.

AdamW 두 step을 state transition으로 검산한다. AdamW의 correctness는 parameter delta만 봐선 부족하다. moment, bias correction, decoupled decay, optimizer internal step과 scheduler LR을 함께 본다. scalar `θ0=1`, gradient `g1=0.2`, `β1=0.9`, `β2=0.99`, `ε=10^-8`, learning rate `η=0.1`, weight decay `λ=0.01`을 쓰자. `m1=0.02`, `v1=0.0004`, bias-corrected 값은 `m̂1=0.2`, `v̂1=0.04`다.

decoupled update는 대략 `θ1 = θ0 - η m̂1/(sqrt(v̂1)+ε) - ηλθ0 = 1-0.1-0.001=0.899`다. 구현이 decay를 gradient에 섞는 L2 regularization을 쓰면 moment에도 decay 성분이 들어가 다른 state가 된다. parameter 한 step이 비슷해 보여도 두 step에서 차이가 커지므로 moment까지 assertion한다.

두 번째 gradient를 `g2=-0.1`이라 하면 `m2=0.9×0.02+0.1×(-0.1)=0.008`, `v2=0.99×0.0004+0.01×0.01=0.000496`이다. bias correction denominator는 `1-0.9²=0.19`, `1-0.99²=0.0199`이므로 `m̂2≈0.042105`, `v̂2≈0.024925`다. adaptive term은 약 `0.2667`, decay는 현재 parameter 0.899에 적용되어 `θ2≈0.899-0.02667-0.000899=0.871431`이다. 실제 oracle은 충분한 precision으로 계산해 반올림 오차를 고정한다.

parameter group 두 개면 step과 scheduler 관계를 따로 확인한다. bias·normalization에는 decay 0, matrix weight에는 decay 0.01처럼 group rule을 canonical order로 기록한다. Python set iteration이나 이름 substring으로 group이 흔들리지 않게 explicit parameter ID 목록을 artifact로 만든다. 모든 trainable parameter가 정확히 한 group에 있고 frozen parameter는 어떤 group에도 없다는 집합 등식을 검사한다.

overflow fixture에서는 scaler가 optimizer step을 skip한다. parameter, moment, optimizer step과 scheduler가 그대로이고 scale만 정책대로 줄어야 한다. callback이 attempted step을 보고 scheduler를 먼저 움직이면 LR clock이 갈라진다. overflow 뒤 정상 batch에서 uninterrupted reference와 합류하는지도 본다. nonfinite gradient를 0으로 바꾸고 update하는 복구는 명시적 정책 없이는 허용하지 않는다.

fused optimizer와 foreach 구현을 비교할 때 같은 hyperparameter 이름만으로 충분하지 않다. capturable, differentiable, tensor LR, master weight, stochastic rounding과 state dtype이 branch를 바꾼다. reference eager 구현의 selected tensor state를 candidate와 비교하고 unsupported option 조합은 validator에서 거부한다. speed가 빠르다는 이유로 수치 semantics가 다른 candidate를 같은 baseline으로 기록하지 않는다.

checkpoint kill matrix로 durability를 증명한다. checkpoint write를 `snapshot`, `serialize`, `flush`, `manifest`, `publish` 다섯 단계로 나누고 각 경계에서 process를 죽인다. snapshot 전 kill은 새 generation이 없어야 한다. shard 일부 serialize 뒤 kill은 staging만 남고 reader가 선택하면 안 된다. 모든 file을 썼지만 complete manifest 전 kill도 불완전하다. manifest 뒤 alias publish 전 kill은 generation을 직접 복구할 수 있지만 latest는 이전 generation을 가리킨다. publish 뒤 kill은 새 generation이 완전히 읽혀야 한다.

component가 model 4 shard, optimizer 2 shard, metadata 1개라고 하자. complete manifest의 required count는 7이고 각 digest가 맞아야 한다. 6개만 존재하는데 file count가 7인 경우도 만든다. 공격자가 빈 파일이나 중복 shard를 넣을 수 있기 때문이다. path uniqueness, logical component ID와 key coverage를 검사한다. 총 byte size만 맞는 것도 충분하지 않다.

async writer는 더 어려운 snapshot 문제를 만든다. training thread가 parameter를 계속 바꾸는 동안 writer가 live tensor를 읽으면 서로 다른 step의 shard가 섞일 수 있다. immutable copy, copy-on-write 또는 synchronization으로 snapshot epoch를 고정한다. manifest에 parameter version을 넣고 shard마다 같은 version인지 확인한다. GPU-to-CPU copy 완료 event와 filesystem durability를 분리한다.

retention은 새 generation 성공 뒤에만 오래된 generation을 지운다. 최소 두 complete generation을 유지하고 delete queue가 current·predecessor를 잘못 지우지 않는 fixture를 둔다. storage pressure에서 partial을 먼저 청소하되 forensic policy를 따른다. disk-full 주입은 shared disk가 아닌 제한된 temporary filesystem에서 수행하고 cleanup 뒤 free space와 mount 상태를 확인한다.

resume comparison은 branch를 checkpoint 직전 같은 parent에서 나눈다. branch U는 중단 없이 다음 두 update를 실행하고 branch R은 process를 완전히 종료한 뒤 checkpoint를 load해 같은 두 update를 실행한다. next BatchID, dropout-sensitive activation, loss numerator·denominator, gradient, delta, moment, LR과 새 checkpoint digest를 비교한다. tracker run ID나 wall-clock은 달라도 된다.

reader도 fault 대상이다. latest pointer가 존재하지 않음, 잘못된 generation을 가리킴, manifest JSON truncation, unknown schema, 한 shard digest 불일치, optimizer state dtype mismatch와 base adapter identity mismatch를 넣는다. reader가 조용히 model-only warm start로 강등하지 않아야 한다. warm start는 별도 action과 새 lineage로만 허용한다.

profiler를 관측 창과 비용 계약으로 사용한다. profiler의 목적은 예쁜 timeline이 아니라 step 시간을 CPU 준비, host-to-device, forward, backward, optimizer, checkpoint와 idle로 분해하고 예상 operator·kernel이 실행됐는지 확인하는 것이다. capture 자체가 synchronization, memory와 file I/O를 추가하므로 observer-off baseline, lightweight counter, detailed trace를 별도 run으로 둔다. trace가 켜진 수치를 baseline throughput으로 보고하지 않는다.

일정은 wait, warmup, active, repeat로 정의한다. compile과 allocator warmup이 끝나기 전에 active window를 열면 steady state를 측정하지 못한다. 반대로 active window를 너무 길게 잡으면 trace가 disk와 memory를 압도한다. 예를 들어 wait 2, warmup 2, active 3 step을 2회 반복하면 detailed sample은 6 step이고 전체 schedule은 14 step이다. 어느 step이 capture됐는지 committed·attempted clock으로 기록한다.

성능 분모를 명확히 한다. micro-step 120 ms, accumulation 4라면 순수 계산상 update 하나는 최소 480 ms지만 optimizer 20 ms와 data stall 40 ms가 update마다 추가되면 540 ms다. 각 micro-batch의 padded token 2,048, valid token 1,500, supervised token 1,000이면 padded throughput, valid throughput과 objective throughput이 다르다. 모델 비교에는 같은 분모를 사용한다.

GPU utilization 하나로 병목을 판정하지 않는다. CPU dataloader queue depth, H2D bandwidth, kernel gap, occupancy, memory allocation, synchronization과 checkpoint stall을 함께 본다. utilization이 높아도 inefficient kernel이 오래 돌 수 있고 낮아도 작은 golden workload의 의도된 결과일 수 있다. golden run은 절대 최대 성능보다 option change의 first effect를 찾는다.

trace에는 민감한 input shape, source path나 annotation이 들어갈 수 있다. export 전 redaction하고 추적 산출물를 접근 제어한다. profiler callback 실패가 training을 중단할지 degraded observation으로 계속할지 정책을 둔다. golden correctness run에서는 profiler failure를 숨기지 않고 관측 gate를 실패시키되 학습 state가 변하지 않았는지 clean control로 확인한다.

option 비교는 한 축씩 한다. eager FP32 reference, BF16, activation checkpointing, compile, fused optimizer를 순차 edge로 만들고 각 edge에 correctness, peak memory, valid token/s, p95 step과 compile cost를 둔다. BF16+compile처럼 interaction이 큰 조합은 별도 edge다. 모든 조합의 수치를 추정으로 채우지 않고 `NOT-RUN`으로 남긴다.

first divergence를 자동 양분한다. 두 run이 다르면 먼저 입력 자격을 비교한다. source, RunSpec, bundle, GoldenBatch와 environment가 다르면 수치 비교 전에 `INPUT_MISMATCH`다. 같다면 token·label·mask를 exact 비교한다. 그다음 selected forward boundary, per-token loss, scaled·unscaled gradient, clipped gradient, optimizer state, parameter delta, checkpoint와 resume 순으로 간다.

layer가 32개이고 final logit만 다르면 midpoint layer 16의 activation digest와 numerical summary를 비교한다. 16까지 같으면 17~32, 다르면 1~16을 반복해 최대 5회 비교로 최초 layer 후보를 찾는다. digest는 exact candidate에만 쓰고 numerical candidate에는 max error, norm, cosine과 nonfinite를 쓴다. 같은 summary가 tensor permutation을 숨길 수 있으므로 distinctive selected elements와 shape도 본다.

forward가 같고 backward가 다르면 loss scalar만 다시 보지 않는다. gradient가 처음 생기는 output부터 reverse order로 hook한다. activation checkpointing이면 recomputed activation과 RNG mask를 비교한다. autocast와 scaler 상태, custom backward kernel을 확인한다. anomaly detection은 디버깅 variant이며 baseline 성능과 분리한다.

gradient까지 같고 delta가 다르면 group membership, unscale·clip order, optimizer hyperparameter, moment, internal step과 scheduler clock을 비교한다. parameter delta가 같고 다음 forward가 다르면 update 후 buffer, RNG, mode, mutable cache를 의심한다. checkpoint 전까지 같고 resume 뒤 다르면 component completeness와 restore order를 본다.

failure bundle은 마지막 통과 boundary, 첫 실패 boundary, expected·actual, tolerance, 소스 심볼, option diff와 reproducer command를 갖는다. 전체 tensor dump는 크기와 민감도를 고려해 선택적으로 격리하고 기본 bundle에는 digest와 summary를 둔다. 문제를 고친 뒤에는 원 fixture와 adjacent clean fixture를 회귀 suite에 넣는다.

반례는 tolerance를 넓혀 통과시키는 것이다. final logit 오차가 크다는 이유로 rtol을 10배 올리면 backward와 update divergence를 정상화할 수 있다. 먼저 최초 layer와 operator를 찾고 kernel·dtype의 기대 오차인지 검증한다. tolerance 변경 자체를 baseline generation 변화로 review한다.

여섯 가지 실패를 복구하는 종단 실습. 첫째 tokenizer drift에서는 raw row는 같지만 TokenID가 달라진다. runner가 fixture exact gate에서 멈추고 forward를 실행하지 않아야 한다. tokenizer bundle을 승인된 digest로 복원하거나 의도된 migration이면 GoldenBatch와 downstream baseline을 새 generation으로 만든다. 기대 token만 고쳐 기존 checkpoint와 같은 lineage로 두지 않는다.

둘째 label shift에서는 TokenID가 같고 loss mask가 한 칸 다르다. per-token target 표가 최초 차이를 보여야 한다. scalar loss가 tolerance 안이어도 exact label gate가 실패한다. collator source와 chat-template supervision boundary를 수정하고 adversarial length fixture를 추가한다.

셋째 FP16 overflow에서는 forward loss가 finite여도 scaled gradient가 inf가 된다. scaler가 update를 skip하고 parameter·moment·scheduler가 그대로인지 확인한다. scale 감소 뒤 같은 batch를 무한 반복하면 sample semantics가 달라질 수 있으므로 retry policy를 명시한다. 정상 BF16 또는 낮은 scale control과 비교한다.

넷째 trainable mapping 오류에서는 adapter tensor 하나가 optimizer group에서 빠진다. parameter coverage 집합 등식이 backward 전에 잡아야 한다. gradient가 생겼지만 delta가 없는 상태를 silent freeze로 분류한다. group을 고친 뒤 two-step fixture에서 B 다음 A gradient가 생기는지 확인한다.

async checkpoint tear가 발생하면 파일 이름은 모두 있어도 shard version이 섞일 수 있다. 이때 generation manifest의 snapshot epoch와 digest가 reader의 접근을 막아야 한다. 이전 complete generation으로 복구한 뒤 writer synchronization을 수정하되, torn generation을 수동으로 complete 표시하지 않는다.

profiler callback이 실패하면 training state와 checkpoint가 정상이어도 trace는 남지 않는다. 따라서 관측 gate는 실패로 판정하되 학습 결과는 correctness evidence로 보존할 수 있다. callback을 제거한 clean control과 state를 비교하고, release에 profiler evidence가 필수라면 재실행한다. logging exception을 무시해 PASS로 만들지는 않는다.

여섯 실습은 각각 input, objective, precision commit, ownership, durability, observability 경계를 겨냥한다. 하나의 거대한 chaos run에 모두 섞지 않는다. 각 fault 뒤 fresh process와 clean fixture로 정상 경로가 돌아오는지 확인한다. failure reason과 artifact cleanup까지 assertion해야 실습이 시스템을 오염시키지 않는다.

다음 장에 evidence index를 넘긴다. 최종 index의 root는 RunSpec, environment, source, input bundle과 실행 결과 child digest를 묶는다. 입력 child에는 config, model·tokenizer, GoldenBatch와 expected oracle이 있다. 실행 child에는 resolved config, runtime inventory, sample stream, forward·backward summary, optimizer transition, checkpoint generations, resume comparison과 profiler result가 있다.

각 gate는 `PASS`, `FAIL`, `NOT-RUN`, `STALE` 중 하나이고 근거 artifact를 가리킨다. source를 읽은 사실은 local execution PASS가 아니다. upstream test는 fixed revision의 기대 행동을 알려 주지만 현재 environment의 kernel 실행을 증명하지 않는다. hardware-pending에는 실행 command, invariant, tolerance와 예상 artifact를 둔다.

29장은 이 index를 대조군으로 받아 rank별 RunSpec, world topology, collective sequence, shard ownership과 remote storage만 추가한다. global loss가 다르면 먼저 rank-local GoldenBatch와 denominator를 합쳐 단일 GPU reference와 비교한다. optimizer shard를 모은 logical state가 28장의 parameter·moment oracle과 같은지 본다. 분산 차이를 기존 단일 GPU 모호성으로 덮지 않는다.

27장으로 되돌리는 edge도 있다. golden checkpoint는 parent bundle, dataset generation, source·environment와 sample stream을 provenance로 남긴다. runtime loaded library와 kernel inventory는 SBOM delta가 된다. 30장은 이 golden evidence를 release candidate의 최소 correctness 기준으로 사용한다. chapter 간 연결이 문장뿐 아니라 동일 ArtifactID로 이어져야 한다.

독립 인수자는 clean workspace에서 index root 하나로 material을 resolve하고 가장 싼 exact tier부터 실행한다. 기존 cache, shell 환경과 tracker secret에 의존하면 실패다. 실행 중 선언하지 않은 file·network read가 있으면 hidden material로 기록하고 RunSpec을 수정한 뒤 새 generation을 만든다.

최종 지원 문장은 실행 범위를 정확히 제한한다. 어떤 GPU, driver, CUDA, dtype, model shape, sequence, optimizer, compile·checkpoint option을 실행했는지 쓴다. 작은 golden fixture의 correctness가 대형 run의 성능과 안정성을 보장한다고 과장하지 않는다. 대신 대형 run이 갈라질 때 비교할 입력, 수치, commit과 durability 기준선을 제공한다고 말한다.

메모리 장부를 생존 tensor와 순간 workspace로 나눈다. OOM을 해결하려면 “GPU 메모리가 부족하다”를 allocation lifetime으로 해체한다. persistent 항목은 parameter, gradient, optimizer state와 장수명 buffer다. step-local 항목은 activation, attention workspace, temporary cast, fused optimizer scratch와 communication 없는 단일 GPU staging이다. checkpoint와 profiler는 CPU pinned memory와 filesystem queue도 사용한다. allocator reserved와 tensor allocated를 구분한다.

간단한 dense model에서 parameter가 10억 개라고 하자. BF16 parameter는 약 2 GB, BF16 gradient는 2 GB다. Adam moment 두 개를 FP32로 두면 8 GB, FP32 master weight가 있으면 4 GB가 추가되어 persistent만 약 16 GB다. 단위는 decimal GB와 GiB를 구분하고 실제 framework storage, padding과 fragmentation을 더한다. “모델이 2 GB이므로 8 GB GPU에서 학습된다”는 계산이 왜 틀린지 드러난다.

activation은 batch, sequence, hidden, layer와 저장 정책에 따라 변한다. 정확한 closed form 하나로 과장하지 말고 profiler snapshot에서 allocation stack과 peak 시각을 본다. batch를 절반으로 줄였는데 peak가 거의 같다면 parameter·optimizer가 지배적이거나 큰 workspace가 고정됐을 수 있다. activation checkpointing으로 peak가 줄지 않으면 checkpoint boundary, long-lived loss tensor와 logging reference를 의심한다.

메모리 누수 fixture는 같은 GoldenBatch를 20 step 반복해 allocated·reserved와 live tensor 수를 step별 기록한다. 그래프를 가진 loss를 Python list에 넣거나 hook output을 detach 없이 보존하면 allocated가 계단처럼 오른다. caching allocator의 reserved 증가를 곧 누수라고 단정하지 않고 live allocation과 재사용 안정화를 함께 본다. cleanup 뒤 새 run control도 실행한다.

OOM first divergence는 phase를 갖는다. model construction, optimizer creation, first forward, backward, unscale·clip, optimizer step, evaluation, checkpoint snapshot 중 어디서 peak가 났는지 기록한다. 같은 “CUDA out of memory”라도 해결이 다르다. optimizer creation OOM에 sequence를 줄이는 것은 효과가 없고, attention workspace OOM에 optimizer offload만 적용하면 핵심 원인을 놓친다.

recovery edge는 한 번에 하나다. micro-batch 감소, accumulation 증가, checkpointing, lower precision, optimizer state 변경, sequence·packing 변경을 각각 correctness·throughput edge로 시험한다. effective batch와 objective denominator, scheduler clock을 유지하는지 확인한다. 메모리를 줄인 대신 다른 학습 문제를 만든 옵션을 성공으로 기록하지 않는다.

compile과 graph capture의 상태 공간을 명시한다. compile option은 단순 속도 스위치가 아니다. graph break, shape specialization, guard, recompilation, kernel selection, RNG와 mutation semantics를 바꾼다. reference는 eager mode이고 candidate compile은 동일 GoldenBatch와 update oracle을 먼저 통과해야 한다. compile 시간과 steady step을 섞지 않으며 cache가 warm인지 cold인지 기록한다.

가변 sequence를 128, 256, 257로 넣어 specialization 경계를 드러낸다. 매 길이마다 새 graph가 생기는지, dynamic shape가 한 graph를 쓰는지와 guard failure reason을 수집한다. production length 분포와 동떨어진 하나의 고정 shape만 빠르게 만든 뒤 일반 성능이라 주장하지 않는다. compile cache key에 source, option, architecture와 relevant environment가 들어가는지 확인한다.

graph break가 있어도 결과가 맞을 수 있지만 성능과 관측성이 달라진다. break 전후 tensor mutation과 RNG 소비가 eager와 같은지 본다. activation checkpointing과 compile을 함께 쓰면 recompute graph, autocast와 side-effect callback이 상호작용할 수 있다. 각 option 단독 edge가 통과한 뒤 pair edge를 별도로 실행한다.

CUDA graph capture를 쓴다면 address 안정성, static shape, warm-up과 replay 가능한 side effect를 계약에 넣는다. input buffer를 in-place 갱신하는 과정과 sample identity가 trace에 남아야 한다. replay가 이전 batch를 재사용하는 negative fixture를 TokenID와 loss에서 잡는다. checkpoint나 logging callback을 capture 안에 잘못 넣지 않는다.

compile failure 복구는 cache 삭제부터 시작하지 않는다. source·config·shape와 guard log를 보존하고 최소 reproducer를 만든다. corrupted cache가 확인됐을 때 해당 key만 quarantine한다. 전체 cache 삭제로 우연히 고친 뒤 원인을 잃으면 회귀 시험을 만들 수 없다. eager fallback을 허용할지, fallback 시 지원 성능을 어떻게 표시할지도 policy로 둔다.

데이터 준비의 동시성이 단일 GPU 의미를 바꾸지 않게 한다. GPU가 하나여도 dataloader worker, prefetch, pinned-memory copy, async checkpoint와 logger가 동시에 움직인다. worker 수를 바꾸면 row completion order, augmentation RNG와 exception 전달 시점이 바뀔 수 있다. sampler가 index 순서를 결정하고 worker가 결과를 reorder하지 않는다는 불변량을 GoldenBatchID stream으로 확인한다.

worker seed는 base seed 하나로 충분하지 않다. worker ID, epoch, sample ID와 augmentation stream의 조합을 명시한다. worker 수가 2에서 4로 바뀌어도 sample별 augmentation을 같게 할 것인지, worker-local stream을 허용할지 선택한다. 전자를 원하면 random draw를 sample identity에 묶는다. 후자는 worker 수를 RunSpec material로 고정하고 재현 범위를 제한한다.

prefetch queue는 checkpoint cursor보다 앞서 row를 읽을 수 있다. 소비 완료 cursor와 fetch cursor를 구분하고 resume에서는 committed batch 다음부터 시작한다. queue에 있던 row를 sampler가 소비한 것으로 착각하면 sample이 건너뛰고, 다시 넣으면 중복된다. kill/resume fixture는 prefetch가 차 있는 시점에 죽여 next BatchID를 비교한다.

pinned-memory와 nonblocking copy는 host buffer lifetime을 요구한다. copy 완료 전에 worker가 buffer를 재사용하면 드문 input corruption이 생길 수 있다. event synchronization 또는 ownership transfer를 source와 profiler trace에서 확인한다. TokenID fixture digest를 device 도착 뒤 selected step에서 검사하면 silent corruption을 input boundary에서 잡을 수 있다.

worker exception을 main process가 늦게 받는 동안 GPU가 이전 batch를 더 실행할 수도 있다. failure contract는 어느 committed update까지 durable한지 기록하고 새 checkpoint publish를 막는다. error가 난 row를 조용히 skip하면 mixture와 denominator가 변하므로 explicit rejection policy 없이는 실패한다. 정상 control에서 같은 stream prefix가 유지되는지 본다.

callback·평가·추적기를 비권위적 관찰자로 제한한다. tracker는 metric을 저장하지만 training clock과 산출물 identity의 권위자가 아니다. network retry, duplicate event와 resume가 run ID를 바꿀 수 있다. authoritative step은 checkpoint와 runner state의 committed update이고 tracker point는 RunID, attempted·committed clock, GoldenBatchID와 checkpoint generation을 함께 가진다.

callback 순서를 고정한다. backward 후 gradient logging이 graph를 보존하지 않는지, optimizer 전 parameter histogram이 synchronization을 만드는지, evaluation이 train/eval mode와 RNG를 원복하는지 확인한다. callback on/off branch의 next batch, loss, gradient와 delta가 tolerance 안에서 같아야 한다. observer effect가 budget을 넘으면 sampling 빈도나 별도 run을 선택한다.

평가 중 dropout을 끄고 끝난 뒤 `train()`을 복원하지 않으면 이후 loss curve가 조용히 달라진다. 평가가 generation을 수행해 RNG를 소비하거나 dataloader worker를 공유할 수도 있다. training과 evaluation generator, sampler와 model mode를 분리하고 before/after state digest를 비교한다. evaluation failure가 optimizer commit을 소급해 취소하는지 여부도 명시한다.

metric reduction은 분모를 기록한다. loss, accuracy와 token count가 서로 다른 ignore mask를 쓰지 않는지 본다. NaN sample을 drop해 평균만 정상으로 만드는 코드를 금지하고 nonfinite count를 별도 metric으로 둔다. p50만으로 tail failure를 숨기지 않고 length·language·task slice를 GoldenBatch와 연결한다.

tracker가 꺼져도 checkpoint와 local evidence가 완전해야 한다. 반대로 tracker 성공이 checkpoint durability를 증명하지 않는다. network 단절 fixture에서 training state가 정책대로 계속되거나 멈추고, buffered event가 중복 없이 처리되는지 본다. secret과 token이 config·trace·checkpoint에 들어가지 않는지 scan한다.

source 함수 좌표를 실행 상태 변화와 연결한다. 고정 Transformers revision `550d7b3834670483a4df436541272c055dc364bf`에서 model 저장 경계는 `src/transformers/modeling_utils.py:3278`의 `save_pretrained`, 로드 경계는 같은 파일 `:3859`의 `from_pretrained`다. golden checkpoint가 이 함수를 사용하면 전달한 safe serialization, shard size, state dict와 config option을 ledger에 남기고 생성 file을 다시 검증한다. 호출 성공을 resume 성공으로 바꾸지 않는다.

Hub resolution은 `src/transformers/utils/hub.py:166`과 `:453` 부근의 snapshot 경계를 따라 실제 immutable revision과 cache path를 기록한다. 모델 생성 전 27장의 bundle identity와 열린 file digest를 비교한다. 소스 원장는 path와 symbol뿐 아니라 어떤 option이 어떤 branch를 선택했는지 RunSpec field와 연결한다.

collator와 processor는 chapter 5의 고정 좌표를 재사용한다. `src/transformers/data/data_collator.py:619`의 `DataCollatorForLanguageModeling`과 `src/transformers/processing_utils.py:1976`의 `ProcessorMixin.apply_chat_template` 경계를 GoldenBatch 생성에 연결한다. multimodal fixture라면 placeholder 수와 media output 수, label mask를 processor output에서 검사한다. trainer 안에서 다시 rendering해 이중 special token을 만들지 않는다.

source test를 읽을 때 positive path만 보지 않는다. shape mismatch, missing key, wrong template, resume partial과 overflow 같은 negative fixture가 어느 layer에서 거부되는지 기록한다. upstream test의 assertion을 local golden contract로 복사할 때 현재 RunSpec과 산출물 identity를 추가한다. 내부 class 이름에 과도하게 결합하지 않고 semantic boundary를 유지한다.

upgrade에서 line이 이동하면 fixed revision link는 과거 evidence로 남기고 새 symbol을 다시 찾는다. old/new branch, default와 test diff를 만든 뒤 GoldenBatch부터 resume까지 위험 기반 tier를 실행한다. baseline expected file을 일괄 재생성하지 않는다. 최초 의도된 차이와 영향받는 downstream oracle을 review한다.

독자가 수행할 세 번의 독립 재실행. 먼저 exact reference를 재실행한다. clean workspace, empty application cache와 verified offline bundle에서 FP32 eager, tiny model, GoldenBatch와 두 AdamW update를 실행한다. token·label부터 forward selected value, loss numerator·denominator, gradient, moment, delta, checkpoint component까지 기대값과 비교하며, mismatch가 하나라도 있으면 다음 variant로 넘어가지 않는다.

exact reference가 통과하면 하나의 수치 option만 바꿔 재실행한다. 같은 parent에서 BF16 또는 선택 precision만 변경하고 declared tolerance로 layer boundary, logit, loss, gradient와 delta를 비교한다. profiler로 실제 dtype·kernel 경로를 확인하며, speed 숫자는 correctness gate를 통과한 sample로만 계산한다. 이때 overflow fixture와 clean control도 함께 실행한다.

마지막 재실행에서는 durability를 검증한다. complete checkpoint 경계와 staging 중간에서 각각 process를 죽인 다음, 이전 generation 선택과 새 generation completeness, clean-process next-step을 uninterrupted branch와 비교한다. prefetch queue와 tracker를 켠 variant에서도 next BatchID가 맞는지 보고, temporary storage와 callback hook의 cleanup 여부까지 확인한다.

세 재실행의 source·RunSpec·environment가 같아도 각 RunID와 evidence generation은 다르다. 결과를 덮어쓰지 않고 parent relation으로 묶는다. 독립 검토자는 작성자 shell history, compiler cache나 tracker session 없이 root index에서 material을 resolve한다. 숨은 입력 접근이 관측되면 실패로 기록한다.

결과표는 exact boundary, numerical boundary, commit, durability, observation과 performance를 별도 열로 둔다. 하나의 “성공” 열로 합치지 않는다. 실행하지 않은 compile·optimizer·shape는 `NOT-RUN`, 오래된 hardware 결과는 `STALE`이다. exception에는 owner와 expiry를 둔다.

이 세 번의 실행이 제공하는 것은 모든 규모의 학습 성공 보장이 아니다. 입력에서 update와 durable state까지 의미가 추적되는 대조군이다. 29장에서 collective, rank와 shard를 추가했을 때 동일 local computation이 유지되는지 비교하고, 30장에서 release candidate가 같은 evidence 계약을 소비하는지 판정한다.

성능 회귀를 통계와 원인 경계로 판정한다. golden run의 성능 gate는 한 번의 wall-clock을 이전 숫자와 비교하지 않는다. 동일 workload를 warm-up 뒤 여러 번 반복하고 median, p95, dispersion과 환경 noise를 기록한다. GPU clock, thermal·power 상태, 다른 process, CPU affinity와 storage cache가 달라지면 비교 자격을 먼저 실패시킨다. correctness가 다른 run은 아무리 빨라도 성능 후보가 아니다.

예를 들어 reference step 시간이 `[100,101,99,100,100]` ms이고 candidate가 `[96,97,140,96,97]` ms라면 candidate median은 더 빠르지만 tail 140 ms가 있다. 평균 하나는 개선을 작게 보이게 하거나 tail을 숨긴다. p50, p95와 stall reason을 함께 보고 140 ms step이 checkpoint, compile, dataloader 또는 allocator인지 trace에서 찾는다. profiler capture step을 일반 sample에 섞지 않는다.

throughput도 workload identity를 검산한다. reference가 초당 padded token 20,000, candidate가 22,000이어도 candidate의 valid token 비율이 낮거나 supervised denominator가 다르면 학습량 비교가 아니다. padded, valid, supervised token/s와 sample/s를 함께 계산한다. packing change는 성능과 objective weighting을 동시에 바꾸므로 새 GoldenBatch 의미 gate를 먼저 통과한다.

회귀 budget은 절대·상대 조건을 조합한다. 짧은 tiny step에서 timer resolution과 launch noise 때문에 2% 조건이 불안정할 수 있다. 충분한 반복과 confidence interval을 쓰되 결과를 본 뒤 threshold를 바꾸지 않는다. 장비 간 비교는 같은 GPU 이름만으로 하지 않고 architecture, memory, driver, power limit와 clock 정책을 고정한다.

원인 양분은 phase별 time과 count를 사용한다. operator 수가 늘었으면 graph break나 recompute, kernel time이 늘었으면 selection·shape·dtype, gap이 늘었으면 CPU·synchronization, H2D가 늘었으면 input pipeline을 의심한다. optimizer가 느려졌다면 parameter group 수, foreach/fused branch와 state dtype을 확인한다. peak memory 증가와 allocator retry가 tail을 만들 수도 있다.

개선도 비용을 함께 기록한다. compile로 steady step이 10 ms 줄지만 첫 compile이 60초라면 총 6,000 step 이후에야 단순 손익분기다. 실제로는 cache 재사용, shape recompilation과 운영 restart 빈도를 고려한다. activation checkpointing이 memory를 줄여 batch를 키울 수 있어도 recompute로 step이 느려지고 objective denominator가 달라질 수 있다.

baseline은 immutable generation이다. hardware maintenance나 framework upgrade 뒤 새 baseline을 만들 때 old/new를 같은 session에서 교차 실행하고 수치 oracle을 먼저 통과한다. rolling baseline이 느린 회귀를 조금씩 흡수하지 않게 release 기준점과 recent distribution을 모두 보존한다. regression waiver에는 영향, owner와 expiry가 있다.

지원 조합을 pairwise 위험 그래프로 관리한다. precision, compile, checkpointing, optimizer, batch shape와 attention kernel의 전체 Cartesian product는 빠르게 폭발한다. 그러나 각 option을 단독으로 한 번 시험한 것으로 조합을 승인할 수도 없다. option을 node, 상호작용을 edge로 보고 의미를 크게 바꾸는 pair와 production 경로를 우선 실행한다.

BF16과 checkpointing은 recompute autocast, FP16과 scaler는 overflow commit, compile과 dynamic shape는 specialization, compile과 checkpointing은 graph break, fused optimizer와 capturable은 step state, async checkpoint와 accumulation은 snapshot 경계를 공유한다. 이 edge들은 단독 test가 통과해도 별도 fixture가 필요하다. 세 option interaction이 알려졌다면 hyperedge로 등록한다.

support matrix 각 cell에는 단순 체크가 아니라 last execution source, environment, fixture generation, result와 age가 있다. 같은 option 이름이어도 model architecture, parameter shape나 GPU architecture가 kernel branch를 바꾸면 별도 범위다. 새로운 GPU에서 기존 결과를 복사하지 않는다. `NOT-RUN`을 실패처럼 숨기지 않되 production promotion은 required cell이 모두 fresh해야 한다.

위험 기반 우선순위는 사용 빈도, 변경 범위, failure severity와 detector strength를 곱해 정한다. 자주 쓰고 silent numerical drift가 가능한 edge를 매 change 실행한다. 드물고 명시적 unsupported error가 나는 조합은 release 전 또는 주기적으로 돌릴 수 있다. 다만 test 비용 때문에 unsupported라고 주장하려면 validator가 실제로 거부해야 한다.

matrix 축소에는 equivalence 근거가 필요하다. sequence 127과 128이 같은 kernel path라는 것을 profiler와 guard로 확인했다면 대표값을 둘 수 있다. 128과 129에서 tile 또는 compile graph가 달라지면 경계값을 모두 시험한다. 평균 길이 하나로 ragged workload를 대표하지 않는다. batch 1과 batch 2에서 optimizer 의미는 같아도 memory·kernel은 다르다.

source upgrade가 특정 branch만 바꿨다면 option graph에서 reachable cell을 계산한다. collator 변경은 모든 input fixture, scaler 변경은 FP16 edge, serializer 변경은 checkpoint 조합을 재실행한다. 공통 core 변경은 전체 required matrix를 갱신한다. 영향 분석 결과와 실제 선택 cell을 evidence에 남긴다.

실패 후 복구가 기준선을 되돌리는지 확인한다. fault test의 성공은 오류를 탐지한 순간이 아니라 정상 상태로 복귀한 뒤 clean control이 같은 oracle을 만족할 때다. fault hook, environment variable, monkey patch, corrupted cache와 temporary file이 남으면 다음 test를 오염시킨다. 각 fixture는 setup, trigger, expected failure, recovery, cleanup과 post-control의 여섯 상태를 가진다.

OOM 뒤에는 optimizer·scaler·gradient와 allocator 상태를 본다. backward 중 OOM이 났는데 일부 `.grad`가 남은 채 작은 batch를 재시도하면 이전 partial gradient가 섞일 수 있다. 정책에 따라 process를 재시작하거나 gradient와 transient state를 명시적으로 초기화한다. 같은 process 복구를 주장하려면 untouched reference와 다음 delta를 비교한다.

nonfinite 뒤에는 skip된 sample을 재사용할지 버릴지 sample-stream ledger로 표현한다. scheduler, scaler와 committed clock이 맞아도 batch 정책이 다르면 학습 trajectory가 달라진다. 반복적으로 실패하는 batch를 무한 retry하지 않도록 횟수와 quarantine을 둔다. raw row를 조용히 삭제하지 않고 IncidentID와 연결한다.

checkpoint corruption 뒤에는 이전 complete generation을 선택하고 corrupted generation을 quarantine한다. reader가 부분 load한 model state를 계속 사용하지 않게 새 process에서 복구한다. alias repair와 storage cleanup은 transaction으로 수행하고 27장의 artifact state를 갱신한다. 복구 checkpoint에서 next-step oracle을 실행한다.

logger·profiler 실패 뒤에는 등록한 hook, background thread와 open file이 종료됐는지 확인한다. observer를 disable한 상태가 다음 run에 전역으로 남지 않아야 한다. trace queue가 disk를 계속 쓰거나 CUDA event를 보존하면 성능 control이 오염된다. thread count, file descriptor와 memory baseline을 post-control에서 검사한다.

복구 시간도 metric이지만 correctness보다 앞서지 않는다. detection latency, containment, last-good selection, state verification과 service-ready 시각을 나눈다. 빠르게 재시작했어도 잘못된 batch나 checkpoint에서 시작하면 실패다. recovery artifact는 어떤 state를 버렸고 어느 committed update부터 재개했는지 쓴다.

golden runner의 최소 책임을 경계별로 검토한다. runner는 모든 framework 기능을 재구현하지 않는다. 대신 resolution, validation, execution, observation, commit과 evidence publish 경계를 소유한다. resolution은 floating 입력을 immutable identity로 바꾸고, validation은 RunSpec·fixture·environment 모순을 forward 전에 거부한다. execution은 정한 state transition 순서를 지키며 observation은 state를 바꾸지 않아야 한다.

commit 경계에서는 overflow, accumulation과 exception을 고려해 attempted micro-step과 committed optimizer update를 분리한다. checkpoint는 committed state만 complete generation으로 publish한다. evidence publish가 실패하면 학습 성공과 검증 package 완성을 별도로 표시한다. 증거 없는 run을 release PASS로 자동 승격하지 않는다.

runner가 framework callback에 맡기는 기능도 adapter contract를 가진다. collator output, model loss reduction, optimizer step return, scaler skip signal, scheduler order와 serializer component 목록을 structured 값으로 받는다. log 문자열을 파싱해 correctness를 추론하지 않는다. framework가 필요한 signal을 주지 않으면 instrumented wrapper나 작은 reference path를 둔다.

예외 처리 순서가 중요하다. 첫 실패 artifact를 쓰고 training state를 더 commit하지 않으며 async task를 취소한다. cleanup 실패는 원래 failure를 덮지 않고 secondary reason으로 기록한다. partial evidence도 digest와 incomplete 상태를 갖는다. keyboard interrupt와 watchdog timeout도 동일 state machine을 따른다.

runner test는 mock만으로 끝내지 않는다. tiny real tensor의 two-step update, 실제 serialize/read, process kill과 clean resume를 포함한다. mock은 disk full, tracker timeout처럼 제어하기 어려운 branch에 사용하되 정상 real control과 쌍을 이룬다. CUDA 지원을 주장하는 path는 실제 CUDA gate가 필요하다.

코드 리뷰에서는 각 config option이 어떤 runtime branch, mutable state, metric과 artifact field를 바꾸는지 묻는다. branch를 바꾸지만 RunSpec digest에 없는 option, checkpoint에 저장되지 않는 mutable state, metric 분모가 없는 counter와 evidence가 없는 success path를 찾는다. 이 네 종류가 golden runner의 핵심 결함이다.

이 책임 경계가 선명하면 구현이 짧아도 강하다. 반대로 거대한 trainer를 감싸고 exit code와 loss screenshot만 남기면 어디서 의미가 달라졌는지 알 수 없다. golden runner의 가치는 기능 수가 아니라 입력부터 durable update까지 모든 상태 전이를 질문 가능한 artifact로 만드는 데 있다.

golden fixture의 크기를 작게 하되 의미는 줄이지 않는다. tiny model은 성능 대표가 아니라 상태 oracle이다. vocabulary, hidden size, layer, head와 sequence를 손으로 검산 가능한 수준으로 줄이되 tied embedding, normalization, causal mask, padding, multiple parameter group과 dropout처럼 실제 모델의 중요한 branch는 남긴다. parameter 수를 줄인다는 이유로 loss mask나 optimizer state를 생략하면 검증 대상도 함께 사라진다.

dataset은 정상 대화, 길이 경계, 빈 assistant target, special token 충돌과 두 document packing을 포함한다. 각 row에는 expected rendered text, IDs, mask, target count와 family ID가 있다. random text를 매 실행 생성하지 않고 immutable fixture를 사용한다. 개인정보나 공개 benchmark 원문을 넣지 않는다.

첫 batch의 forward logits 전체를 영구 baseline으로 크게 저장하기보다 representative positions와 digest, loss numerator·denominator를 둔다. first divergence가 생기면 bounded full tensor를 forensic artifact로 저장한다. fixture가 너무 작아 실제 fused kernel shape를 타지 못하는 경우 별도의 representative shape fixture를 추가하고 두 검증 범위를 구분한다.

bytes에서 loss까지 네 개의 oracle을 둔다. 첫 oracle은 source bytes와 decoder·normalizer 출력이다. 둘째는 tokenizer/template가 만든 IDs와 special token 위치다. 셋째는 collator의 input, attention·position·label mask와 padding이다. 넷째는 model logits에서 target token만 뽑아 계산한 cross-entropy numerator와 denominator다. 최종 loss 하나만 비교하면 앞 경계의 두 오류가 우연히 상쇄될 수 있다.

oracle 구현은 trainer와 같은 helper를 그대로 호출하지 않는다. 동일 결함을 공유하면 비교가 무의미하다. 작은 명시적 reference calculation을 만들고 소스 리비전을 고정한다. label shift, ignored index, sequence packing과 vocabulary resize를 손으로 확인한다. float tolerance는 dtype과 연산 순서에 근거해 정한다.

negative fixture는 assistant mask를 한 token 이동하고, tokenizer revision을 바꾸며, padding side와 BOS를 바꾼다. verifier가 최초 경계에서 구체적으로 실패하는지 본다. loss가 비슷하다는 이유로 IDs·mask mismatch를 허용하지 않는다.

backward와 optimizer commit을 두 step으로 증명한다. 첫 backward에서는 trainable parameter set, gradient 존재·shape·dtype·finite와 representative norm을 확인한다. frozen base와 adapter target, tied parameter의 ownership을 manifest와 비교한다. gradient가 없는 것이 의도인지 graph 단절인지 parameter disposition으로 구분한다. backward hook은 진단용이며 실행 순서를 바꾸지 않는지 확인한다.

첫 optimizer step은 moments와 step clock을 만들고 두 번째 step은 이를 소비한다. 고정 gradient 또는 고정 batch sequence로 parameter, first·second moment, LR, scaler와 scheduler를 reference와 비교한다. accumulation에서는 마지막 microbatch에만 commit되고 valid target denominator가 window 전체와 맞는지 본다.

overflow를 주입하면 parameter·moments·optimizer clock이 보존되고 scaler만 정책대로 바뀌는지 확인한다. clipping은 unscale 뒤에 적용되고 reported norm의 의미가 명확해야 한다. `zero_grad(set_to_none=True)` 뒤 stale gradient가 다음 window에 남지 않는지 시험한다.

checkpoint round-trip은 다음 update의 동등성으로 판정한다. save 직전에는 model, optimizer, scheduler, scaler, RNG, data cursor와 canonical config의 generation을 하나의 manifest로 묶는다. 각 tensor shard·file의 digest와 component disposition을 저장한다. write가 끝난 뒤 validation과 atomic commit이 완료되어야 complete checkpoint다. 파일이 존재하는 시점과 복구 가능한 시점을 구분한다.

load 검증은 key·shape가 맞는지에 그치지 않는다. 연속 run과 resume run이 같은 다음 batch를 읽고 forward·gradient·next update를 허용오차 안에서 재현해야 한다. dropout·sampler·worker RNG, scheduler와 optimizer clock이 빠지면 이 시험에서 차이가 난다. data iterator를 복원하지 못하면 sample replay 의미를 보고한다.

partial write, missing optimizer state, wrong tokenizer·config와 parameter reorder를 negative fixture로 둔다. loader가 0으로 채우거나 warning만 내고 계속하면 golden gate는 실패시킨다. 의도적인 warm start는 resume와 다른 RunID·state policy를 가진다.

CUDA 실행 경로는 dispatch와 synchronization을 함께 기록한다. CUDA를 사용했다는 사실만으로 원하는 kernel을 실행했다고 할 수 없다. autocast dtype, attention backend, fused optimizer, compile·graph capture와 fallback의 requested·effective 값을 저장한다. representative operation의 profiler trace나 dispatch log로 실제 path를 확인한다. source에서 지원된다는 문장과 이 GPU에서 실행됐다는 증거를 구분한다.

비동기 오류는 뒤의 synchronization에서 표면화될 수 있다. golden debug mode에서는 주요 경계에 bounded sync를 넣어 first divergence를 좁히되 성능 측정에서는 제거한다. debug와 benchmark 결과를 같은 series로 섞지 않는다. deterministic option도 지원 범위와 비용을 기록한다.

CUDA·driver·library·GPU architecture가 바뀌면 CPU reference, 이전 CUDA path와 새 path를 같은 fixture로 비교한다. bitwise, numerical, behavioral 등급을 나누고 미실행 dtype·shape를 지원됨으로 표시하지 않는다.

메모리 peak를 phase와 tensor lifetime으로 검산한다. baseline, after load, forward peak, backward peak, optimizer step, checkpoint staging과 cleanup 뒤 memory를 구분한다. allocated·reserved·inactive split과 largest allocation을 기록한다. static parameter·optimizer bytes 계산과 allocator 관측을 대조해 설명되지 않는 차이를 찾는다.

sequence length, microbatch, precision, activation checkpoint와 attention backend를 한 축씩 바꾼다. OOM이 사라졌어도 sample drop·target 감소나 CPU offload stall로 의미가 바뀌지 않았는지 본다. instrumentation이 peak를 추가할 수 있으므로 memory snapshot on/off paired run을 둔다.

run 종료 뒤 memory가 baseline으로 돌아오지 않으면 tensor reference, graph retention, async task와 cache를 조사한다. 단 한 번의 run에 문제가 없어도 반복 golden run에서 증가하는 leak을 잡는다. cleanup failure를 무시하고 다음 fixture를 실행하지 않는다.

성능 baseline은 valid target token과 phase로 나눈다. tokens/s의 tokens가 input인지 valid target인지 명시한다. padding·packing·sequence 변화에서 raw token throughput은 학습 유효 작업을 과장할 수 있다. data wait, forward, backward, optimizer, collective가 없는 단일 GPU 경계, checkpoint와 logging 시간을 phase별로 측정한다.

compile·autotune warmup step을 제외할 때 범위와 이유를 기록하고 correctness artifact는 보존한다. steady-state 표본 수, median과 tail, clock source와 synchronization 방식을 명시한다. thermal·power state, competing process와 CPU input 조건도 environment에 넣는다.

성능 회귀는 old·new를 교차 순서로 반복해 drift를 줄이고 effect와 variance를 보고한다. 빨라졌지만 numerical fixture가 실패하면 승인하지 않는다. 느려졌지만 source·kernel 변화가 없는 경우 관측 overhead와 data cache부터 확인한다.

framework option을 실제 state change로 번역한다. gradient accumulation, checkpointing, mixed precision, compile, packing, `remove_unused_columns`, best-model loading과 adapter target option마다 parser source, effective value, runtime branch, mutable state, checkpoint field와 metric을 표로 만든다. option 이름의 설명만 복사하지 않는다.

boolean 하나가 여러 상태를 바꿀 수 있다. compile은 graph partition, kernel, cache와 error surface를, mixed precision은 autocast, scaler·master state와 collective dtype을, packing은 attention boundary와 denominator를 바꾼다. 영향 edge마다 golden fixture를 선택한다.

unknown·deprecated option과 auto default를 fail 또는 명시적 disposition으로 처리한다. requested value와 runtime clamp가 다르면 manifest에 둘 다 남긴다. upgrade 뒤 동일 option이 다른 branch를 타면 expected snapshot을 먼저 갱신하지 않고 first divergence를 검토한다.

실패 양분은 변경 diff와 상태 경계를 함께 사용한다. golden run이 실패하면 source commit만 bisect하지 않는다. data snapshot, tokenizer, config, dependency, CUDA kernel과 hardware도 change axes다. 먼저 artifact diff로 바뀐 축을 열거하고 bytes→token→forward→backward→update→checkpoint 경계의 최초 차이를 찾는다. 그 뒤 최소 axis를 old/new로 교차한다.

failure artifact에는 마지막 정상 boundary, 최초 비정상 value·tolerance, RunSpec과 재현 명령을 넣는다. 사람의 화면 캡처만 남기지 않는다. nondeterministic failure는 seed 반복, event ordering과 frequency를 기록하고 “재현 안 됨”으로 닫지 않는다.

수정 뒤 원 fixture, 인접 negative fixture와 전체 golden suite를 실행한다. expected 값을 수정 코드로 다시 생성해 같은 검토에서 승인하지 않는다. 독립 reference 또는 이전 signed baseline과 비교한다.

29장으로 넘기는 rank-local reference package. package에는 source·environment·data·tokenizer·config digest, parameter ownership, four-oracle 결과, two-step update, overflow·clipping, checkpoint round-trip, memory·performance와 failure fixtures가 들어간다. 각 artifact는 RunID와 boundary ID로 연결되고 pass·fail·not-executed를 구분한다.

29장의 분산 실험은 이 결과를 rank-local reference로 사용한다. single-GPU loss·gradient·next update를 모르면 all-reduce 뒤 값이 올바른지 판단할 기준이 없다. 분산에서 새로 생기는 group·shard·collective·failure semantics만 추가하고 data·objective 의미를 다시 추정하지 않는다.

독립 검토자는 canonical row에서 checkpoint 다음 update까지 정방향으로, checkpoint에서 원천 행까지 역방향으로 걷는다. option 하나를 골라 실제 branch·state·artifact까지 측방향으로 확인한다. 세 경로가 모두 evidence로 이어지고 미실행 CUDA·shape·framework cell이 명시될 때 single-GPU golden run을 인수한다.

adapter와 quantized base의 단일 GPU 경계를 따로 시험한다. LoRA fixture는 target module match 수, trainable parameter names, rank·alpha·scaling과 initialization을 manifest로 만든다. 첫 forward에서 base와 adapter delta를 분리하고 backward에서 adapter에만 gradient가 흐르는지 확인한다. target 문자열이 0개 또는 의도보다 많은 module을 선택하는 negative fixture를 둔다. tied·shared parameter와 modules-to-save도 state dict disposition을 가진다.

QLoRA에서는 저장 dtype, compute dtype, dequantization kernel, quantization group metadata와 master·optimizer state를 구분한다. low-bit base file이 작다는 이유로 activation·workspace와 adapter optimizer memory를 빼지 않는다. CPU reference가 동일 low-bit kernel을 재현하지 못하면 dequantized weight·layer output·final logits의 단계별 tolerance를 둔다.

adapter save/load·merge 뒤에는 adapter-on reference와 logits·weight delta를 비교한다. merge dtype과 순서가 결과를 바꿀 수 있다. merge된 artifact를 다시 training checkpoint처럼 resume하지 않고 목적별 bundle을 분리한다. unsupported layer·shape의 silent fallback을 dispatch trace에서 확인한다.

preference objective의 canonical pair를 손으로 검산한다. chosen·rejected pair는 동일 prompt와 template를 공유하는지, completion mask와 truncation이 어느 token을 보존하는지 확인한다. policy와 reference의 completion log-prob sum·mean denominator를 명시하고 작은 logits로 log-ratio와 DPO류 loss를 손으로 계산한다. prompt token이 loss에 섞이거나 두 completion 길이 차이가 숨은 가중치가 되지 않게 한다.

reference-free, label smoothing, beta와 loss variant option은 실제 equation·branch와 metric을 연결한다. chosen·rejected가 뒤집힌 fixture, 동일 completion, 모두 truncation된 pair와 reference mismatch를 넣는다. scalar loss만 아니라 pair margin, token count와 gradient direction을 본다.

checkpoint는 policy adapter뿐 아니라 reference identity, preprocessing revision과 objective config를 보존한다. resume 뒤 같은 pair의 다음 update를 재현한다. SFT checkpoint에서 preference run으로 넘어가는 것은 동일 run resume가 아니라 parent가 있는 새 training generation이다.

online RL을 시작하기 전 단일 GPU에서 고정 trajectory를 재생한다. 실시간 rollout을 생성하지 않아도 고정 prompt·response·old log-prob·reward·value 또는 group statistics로 learner update를 검증할 수 있다. trajectory에는 policy revision, tokenizer/template, sampling config, reward·verifier revision과 terminal reason이 있다. learner가 기대하는 tensor shape, mask·advantage normalization과 KL reduction을 손으로 맞춘다.

stale policy, truncated response, reward failure, duplicate trajectory와 모두 같은 reward인 group을 negative fixture로 둔다. discard와 zero-weight를 구분하고 denominator에서 어떻게 처리하는지 확인한다. policy update 뒤 version이 언제 commit되고 다음 rollout이 어느 version을 사용해야 하는지 clock을 남긴다.

이 fixture는 online system의 concurrency를 증명하지 않지만 objective와 state mapping의 작은 oracle을 제공한다. 20장의 rollout·learner 상태와 29장의 분산 failure injection은 이 reference에서 확장한다. 실행하지 않은 environment·tool behavior를 단일 GPU 결과로 일반화하지 않는다.

golden package를 깨끗한 process에서 독립 재생한다. 새 작업자는 저장소 설명을 읽기 전에 immutable manifest와 실행 명령만으로 fixture를 재생한다. environment resolution, data verification, forward·backward·two-step, checkpoint kill/resume와 negative fixture가 예상 boundary에서 통과·실패해야 한다. baseline 파일을 다시 생성하는 option은 별도 권한과 review를 요구한다.

재생 결과가 다르면 final loss부터 비교하지 않고 source bytes, IDs·mask, parameter initialization, first logits, gradient, update와 restored state 순서로 first divergence를 찾는다. dependency·GPU가 달라 허용 가능한 수치 차이면 범위와 tolerance 근거를 기록한다. 설명되지 않은 차이를 “환경 차이”로 닫지 않는다.

최종 report는 실행 시간과 pass 수보다 검증한 의미를 요약한다. 어떤 objective·dtype·kernel·adapter·checkpoint·failure cell을 실제로 실행했고 무엇은 source 확인 또는 미검증인지 표시한다. 이 정직한 경계와 작은 재현 artifact가 있어야 다음 규모 확장에서 발생한 차이를 병렬화 때문인지 기존 의미 오류인지 분리할 수 있다.

데이터 순서와 RNG 소비 위치를 함께 확인한다. 재현 가능한 weight 초기값만으로 학습 순서가 고정되지는 않는다. dataset shuffle, sampler, augmentation, dropout, stochastic rounding, fused kernel과 dataloader worker가 서로 다른 RNG state를 가진다. seed 값 하나가 아니라 generator별 state, 소비 순서와 checkpoint field를 기록한다. worker 수·prefetch·persistent worker 변경이 batch completion order나 transform randomness를 바꾸는지 golden SampleID sequence로 확인한다.

같은 sample order라도 random augmentation이 다르면 input tensor가 달라진다. text masking, image crop, audio segment와 modality dropout 결과를 sample·epoch·transform revision에 연결한다. stateless seed derivation을 사용한다면 formula와 collision test를 둔다. global RNG에 우연히 의존하는 helper를 negative fixture로 찾는다.

resume 직전과 직후의 다음 두 batches, augmentation parameters, dropout mask의 representative digest와 optimizer update를 연속 run과 비교한다. exact 재현을 지원하지 않는 kernel은 numerical·behavioral 등급과 범위를 명시한다. data replay가 발생하면 RNG도 함께 되돌아가는지, 아니면 새 generation으로 처리하는지 정책을 둔다.

마지막으로 worker crash, skipped corrupt sample과 early dataloader exhaustion을 주입한다. trainer가 sample을 조용히 대체하거나 epoch denominator를 바꾸면 ledger가 이를 드러내야 한다. consumed input rows, valid targets와 successful updates의 세 clock을 맞춘다. 이 순서·RNG 계약이 닫혀야 checkpoint round-trip과 loss parity가 우연이 아니라 재현 가능한 상태 복원임을 주장할 수 있다.

인수자는 manifest에 선언되지 않은 random call을 하나 의도적으로 삽입해 first divergence detector가 어느 경계에서 이를 잡는지 확인한다. 또 worker 수를 바꾸되 stateless transform을 유지하는 fixture와 global RNG에 의존하는 fixture를 비교한다. 두 경우를 같은 재현 등급으로 판정하면 안 된다. 최종 evidence에는 generator 이름, state digest, SampleID sequence, augmentation disposition과 다음 update의 비교 결과를 함께 둔다. 신규 data transform이나 fused stochastic kernel이 들어올 때 이 표를 갱신하고 실행하지 않은 branch를 명시한다. 불일치가 발견되면 기대값을 자동 갱신하지 않고 최초 random consumer와 state owner를 찾아 수정한다.

수정 뒤 연속 run과 resume run을 함께 재실행해 순서와 update가 모두 회복됐는지 확인한다.

## 28.14 수치·CUDA·runtime 진단을 같은 oracle에 결박한다

이제 같은 fixture를 dtype·kernel·stream·compile 경로에 통과시킨다. 허용 오차를 넓혀 차이를 덮는 대신 bytes, token, logits, loss, gradient, optimizer delta 순서로 최초 불일치를 찾고, 그 뒤에만 성능과 memory를 비교한다. 따라서 CUDA 진단도 별도의 튜닝 부록이 아니라 앞에서 만든 수치 증명의 연장선이다.

### 28.14.1 작은 fixture의 의미 밀도를 지킨다

golden fixture는 한 GPU에서 빠르게 반복할 수 있을 만큼 작아야 한다. 그렇다고 padding 없는 동일 길이 text 두 줄만 쓰면 packing, ignore mask, accumulation과 checkpoint의 실제 경계를 못 본다. 짧은 표본, 긴 표본, empty/invalid 후보와 special token을 포함해 각 분기가 최소 한 번 실행되게 한다.

모델도 전체 학습 모델과 같은 module family와 objective를 사용하되 layer·hidden·sequence를 줄인다. 완전히 다른 toy MLP는 autograd 자체를 확인할 수 있어도 attention mask, tied embedding, adapter target과 fused loss를 증명하지 못한다. 축소가 바꾼 support 범위를 manifest에 적는다.

golden의 목표는 최고 throughput이나 benchmark score가 아니다. bytes→tokens→logits→loss→gradients→updated weights→checkpoint의 각 경계에 독립 oracle을 두는 것이다. 실패가 나면 마지막 scalar loss가 아니라 최초 달라진 객체를 찾을 수 있어야 한다.

실행 시간이 짧아야 source·dependency·CUDA option을 바꿀 때마다 돌릴 수 있다. 그러나 너무 짧아 scheduler·optimizer second moment·accumulation boundary가 한 번도 진행되지 않으면 부족하다. 최소 두 optimizer updates와 checkpoint 전후 다음 update를 포함한다.

RunSpec을 실행 전에 동결한다. RunSpec은 source commit, dependency lock, container, driver·CUDA, GPU identity, model config, data manifest, tokenizer/template, dtype, optimizer, scheduler, seeds와 output 경로를 가진다. “최신 Transformers”나 branch 이름처럼 mutable reference를 허용하지 않는다.

requested config와 effective config를 분리한다. auto batch size, device capability dispatch, mixed-precision clamp와 environment variable이 값을 바꿀 수 있다. parser 직후와 runtime initialization 뒤 snapshot을 남긴다. unknown option은 조용히 무시하지 않는다.

실행 전에 input artifact의 content hash와 expected schema를 검증한다. model config만 맞고 tokenizer vocabulary·special IDs가 다르면 forward가 성공해도 다른 학습이다. chat template의 rendered bytes와 token IDs를 fixture에 포함한다.

RunID는 config 파일 이름이 아니라 immutable spec digest에서 만든다. 같은 spec 재실행은 AttemptID를 달리하고, artifact를 덮어쓰지 않는다. baseline 승격은 별도 review와 reason을 요구한다.

### 28.14.2 첫 batch를 종이에 펼친다

원문 bytes, decoded text, normalized text, token IDs, role spans, labels, attention mask와 position IDs를 한 표에 놓는다. UTF-8 byte offset과 token span이 일치하지 않을 수 있음을 표시한다. tokenizer round trip이 원문을 그대로 복원하지 않는 경우도 정상일 수 있지만 이유를 안다.

causal LM이면 logits 위치 `t`가 label `t+1`을 예측하도록 shift한다. assistant-only SFT라면 prompt labels가 ignore index인지 확인한다. 유효 target 수 `M`과 loss numerator를 손으로 계산한다. batch·sequence mean의 차이를 적는다.

packing에서는 document boundaries와 causal visibility를 작은 dense matrix로 그린다. cumulative lengths나 block mask가 reference predicate와 같은지 본다. 두 documents 사이에 sentinel을 넣어 leakage가 있으면 loss가 크게 달라지게 만든다.

truncation은 끝에서 잘랐다는 설명으로 충분하지 않다. chat role marker와 assistant answer가 남았는지, 모두 ignore가 된 row의 disposition을 확인한다. dropped tokens·rows와 realized denominator를 ledger에 남긴다.

embedding부터 logits까지 shape 사다리. input IDs `[B,L]`가 embedding lookup 뒤 `[B,L,D]`가 되고, 각 transformer block을 지나 같은 residual shape를 유지한다. attention은 Q/K/V head shapes, MLP는 intermediate width, MoE면 expert/token routing을 별도 기록한다. 최종 norm과 LM head가 `[B,L,V]` logits를 만든다.

tied embedding이면 input embedding과 output head가 같은 storage를 공유하는지 parameter identity로 확인한다. state dict에 두 keys가 있어도 memory alias일 수 있다. optimizer가 같은 parameter를 두 번 update하지 않는지 본다.

한 token·layer·channel을 골라 activation norm과 finite status를 기록한다. 모든 activations를 dump해 I/O로 run을 바꾸지 않는다. selected trace와 anomaly-triggered capture를 사용한다. hook가 graph를 retain해 memory를 늘리지 않게 한다.

model output object에서 logits와 auxiliary/router losses, hidden states가 어디에 들어가는지 실제 caller를 따른다. `return_dict`, cache와 output flags가 graph·memory를 바꾸는지 RunSpec에 둔다.

attention을 작은 행렬로 재계산한다. 아주 작은 head에서 `QK^T/√d`, mask, softmax와 `PV`를 높은 precision reference로 계산한다. causal forbidden 위치의 probability가 0인지, padding row와 all-invalid policy를 확인한다. fused kernel 결과와 허용오차를 비교한다.

RoPE가 있으면 선택 position의 even/odd channel rotation을 손으로 계산한다. position IDs가 padding·packing에서 어떻게 재시작되는지 model contract를 따른다. sequence를 이어붙였다고 position이 자동 document-local이라고 가정하지 않는다.

GQA/MQA에서는 query heads가 key/value heads를 어떤 group으로 공유하는지 index mapping을 확인한다. repeat/materialize와 grouped kernel은 shape가 달라도 논리 결과가 같아야 한다. backward gradient가 shared KV heads에 합쳐지는지 본다.

FlashAttention 같은 fused path는 dense probability matrix를 반환하지 않을 수 있다. 작은 shape의 unfused reference와 output·gradient를 비교하고 selected dispatch를 기록한다. 실행하지 않은 head dimension·mask·GPU를 지원한다고 확대하지 않는다.

### 28.14.3 cross-entropy를 scalar까지 맞춘다

선택한 세 target positions에서 log-sum-exp와 target logit으로 negative log-likelihood를 직접 계산한다. label smoothing, vocabulary masking과 z-loss가 있으면 별도 항으로 더한다. library 반환 total과 components를 구분한다.

FP16/BF16 logits를 FP32 reference와 비교할 때 softmax probability 자체보다 log-sum-exp와 loss tolerance를 본다. 매우 큰 logits fixture로 안정화용 max subtraction이 작동하는지 확인한다. all ignored labels가 NaN·0·error 중 어떤 policy인지 정한다.

distributed 이전 single GPU에서도 gradient accumulation 때문에 denominator가 달라질 수 있다. microbatch means를 단순 평균하는지 global sum/count인지 확인한다. 길이가 다른 두 microbatches로 반례를 만든다.

loss logging용 detached scalar와 backward에 사용한 tensor가 같은 equation인지 본다. callback이 재계산한 loss를 권위값으로 쓰지 않는다. LossID에 numerator, denominator와 objective revision을 연결한다.

backward oracle은 모든 parameter를 보는 것이 아니다. embedding row, 첫 attention projection, 중간 MLP, final norm과 head의 selected elements에 대해 autograd gradient와 finite-difference 또는 작은 high-precision reference를 비교한다. 모든 대형 parameter의 finite difference는 필요하지 않다. 경로별 대표와 negative controls를 고른다.

frozen parameter, unused branch와 ignored target에서는 expected zero/None gradient가 나온다. 0과 None을 구분한다. adapter-only run에서는 base가 frozen이고 adapter·modules-to-save만 gradient를 받아야 한다.

activation checkpointing은 backward 중 forward를 재실행한다. dropout RNG가 보존되는지, hooks가 두 번 관측되는지 확인한다. recompute output과 original forward의 selected digest를 비교한다. compile wrapper가 module name을 바꿔 hook selection을 깨지 않는지 본다.

gradient norm 하나만 보고 correctness를 판정하지 않는다. shape, owner, accumulation, clipping 전후와 optimizer가 실제 읽은 value를 연결한다. nonfinite detector가 어느 phase에서 update를 skip했는지 기록한다.

gradient accumulation의 진짜 등가 조건. microbatch 두 개를 accumulation해 한 update하는 경로와 두 sample을 한 batch로 처리하는 reference를 비교한다. 동일 objective denominator, dropout/RNG와 batch-dependent operations일 때만 gradient equivalence를 기대할 수 있다. BatchNorm, sequence packing과 contrastive negatives는 equivalence를 깨뜨릴 수 있다.

각 microbatch loss를 `1/K`로 나누는 방식은 microbatch의 유효 target 수가 같을 때 token mean과 맞을 수 있다. 길이가 다르면 global numerator/count가 필요하다. framework accelerator가 자동 scaling하는지 source에서 확인한다.

`no_sync`나 equivalent context는 중간 microbatches의 distributed communication을 미루지만 single GPU에서도 optimizer step·zero_grad boundary를 이해하는 fixture가 된다. last microbatch와 incomplete accumulation at epoch end의 policy를 기록한다.

checkpoint가 accumulation 중간에 저장될 수 있다면 partial gradients, cursor와 microstep을 복구할지 안전 boundary까지 버릴지 정한다. 조용히 gradients만 잃고 sample cursor를 진행하면 data가 빠진다.

clipping과 nonfinite skip을 update 식에 넣는다. global gradient norm clipping은 모든 trainable parameter의 norm을 모아 scale factor를 계산한다. mixed precision에서는 unscale 뒤 clip하는지 확인한다. clipping 전·후 norm, threshold와 factor를 손으로 맞춘다.

parameter별 clipping, value clipping과 adaptive clipping은 다른 함수다. option 이름만 보고 global norm이라고 추정하지 않는다. sparse gradients·sharded parameters의 처리도 확인한다.

GradScaler가 overflow를 감지하면 optimizer step을 skip하고 scale을 낮출 수 있다. scheduler와 global update counter도 함께 skip되는지 실제 call order를 본다. data consumed와 update succeeded clocks가 갈린다.

Inf를 한 selected gradient에 주입해 weight, optimizer moments, scheduler와 checkpoint counters가 예상대로 유지되는지 본다. 다음 finite step에서 recovery가 가능한지 확인한다. skip을 0 loss update로 세지 않는다.

AdamW 첫 두 step을 손으로 계산한다. selected scalar parameter에 대해 gradient, first moment, second moment, bias correction, epsilon, learning rate와 decoupled weight decay를 계산한다. implementation이 foreach·fused·single dispatch 중 무엇을 선택했는지 기록한다. 수학 식이 같아도 dtype·rounding과 step representation이 다를 수 있다.

weight decay가 gradient에 더해지는 L2와 decoupled AdamW를 구분한다. bias·norm·embedding 등 no-decay parameter groups를 actual parameter identity로 확인한다. 누락 parameter와 duplicate group을 검사한다.

두 step이 필요한 이유는 first moment와 second moment가 history를 갖기 때문이다. checkpoint round trip 뒤 next update가 맞으려면 weight뿐 아니라 moments, step과 scheduler가 복구돼야 한다. one-step loss parity로는 부족하다.

Muon 같은 matrix optimizer fixture는 12장의 수치 oracle을 가져와 selected 2D parameter의 orthogonalization과 alternate optimizer group을 검증한다. 1D bias/norm이 AdamW로 가는지 parameter routing을 확인한다.

실행 경로와 장애 원인을 단계별로 좁힌다. scheduler를 update clock에 연결한다. warmup과 decay가 microstep, batch, valid tokens 또는 successful optimizer update 중 무엇을 입력으로 쓰는지 확인한다. gradient accumulation과 overflow skip이 있으면 clocks가 다르다. logged `global_step` 이름만 믿지 않는다.

첫 몇 steps의 expected LR을 손으로 계산하고 optimizer param groups의 actual LR과 비교한다. resume 경계에서 warmup이 반복되거나 한 step 건너뛰지 않는지 본다. newly added adapter group의 scheduler age도 정책을 가진다.

token-based scheduler이면 consumed valid target count의 source와 distributed reduction을 확인한다. corrupt-row skip과 packing 변화가 LR 위치를 바꿀 수 있다. data policy가 scheduler state의 일부가 된다.

metric plot의 x축을 명시한다. wall-clock, microstep, update와 tokens를 혼용하면 plateau·throughput을 잘못 해석한다. RunID에서 네 clocks를 변환할 ledger를 둔다.

CUDA dtype과 operator 경로를 기록한다. “BF16 학습”이라는 한 문장 대신 embedding, GEMM input·weight·accumulator·output, norm, softmax, loss, gradients, optimizer master state의 dtype을 적는다. autocast가 operator마다 dtype을 선택하고 일부 reduction은 FP32일 수 있다.

TF32는 FP32 GEMM의 내부 정밀도와 성능을 바꿀 수 있다. flag와 device capability, selected kernel을 기록한다. TF32 on/off 결과를 exact equality가 아니라 정한 numerical oracle로 비교한다.

FP8은 tensor별 scale, amax history, delayed/current scaling과 overflow state를 추가한다. 파일 weight dtype만 보고 계산을 설명하지 않는다. scale state가 checkpoint에 포함되는지 본다.

dtype cast 경계는 memory와 bandwidth도 만든다. profiler에서 explicit/implicit cast kernel과 bytes를 본다. 성능을 실제로 측정하지 않은 조합은 수치 주장 없이 예상 관측점만 남긴다.

CUDA stream과 동기화를 timeline으로 본다. H2D copy, default/compute stream, communication이나 auxiliary stream이 있다면 event dependency를 기록한다. single GPU에서도 asynchronous copy와 compute overlap, metric `.item()`이나 profiler가 동기화를 만들 수 있다.

CPU wall time으로 kernel 하나의 속도를 재려면 적절한 synchronization이 필요하다. 매 step synchronize하면 실제 pipeline 성능을 훼손한다. correctness capture와 steady performance run을 분리한다.

CUDA error는 asynchronous라 최초 잘못된 kernel보다 뒤 API에서 보일 수 있다. 작은 debug run에서는 launch blocking 또는 targeted event로 경계를 좁히되 baseline 성능으로 쓰지 않는다. error timestamp와 last successful boundary를 보존한다.

allocator reserved·allocated, active blocks와 peak reset 시점을 phase별로 기록한다. cache를 비웠다고 live tensor가 사라지는 것은 아니다. OOM 직전 requested bytes와 fragmentation signal을 함께 본다.

compile은 다른 프로그램을 만든다. `torch.compile`류를 켜면 graph capture, guards, graph breaks, compiler lowering과 generated kernels이 추가된다. eager와 compile을 동일 source code의 단순 속도 option으로만 보지 않는다. compiled graph identity와 cache key를 artifact로 다룬다.

dynamic shapes, Python side effects, hooks와 data-dependent branches가 graph break를 만든다. break count와 reason, fallback path를 기록한다. compile success가 모든 steps가 같은 graph를 썼다는 뜻은 아니다.

eager reference와 forward, loss, selected gradients·two-step update를 비교한다. graph recompilation fixture와 stale cache invalidation을 둔다. compiler upgrade는 same model revision이어도 새 execution generation이다.

성능은 compile warmup과 steady steps를 분리한다. 작은 golden run은 compilation cost가 지배할 수 있다. 실행하지 않은 GPU·shape의 speedup을 일반화하지 않는다.

checkpoint 파일이 보이는 순간과 commit 순간. temporary shard가 생성됐다고 complete checkpoint가 아니다. expected files, sizes·digests, model/optimizer/RNG/data cursor와 metadata가 모두 durable한 뒤 root manifest와 commit marker를 원자적으로 공개한다. loader는 incomplete root를 보지 않아야 한다.

kill matrix는 weight 쓰기 중, optimizer 쓰기 뒤, manifest 전·후와 latest alias 전환 중 process를 종료한다. 재시작이 last complete parent를 고르고 partial child를 quarantine하는지 본다. same CheckpointID 재시도는 idempotent해야 한다.

local filesystem rename과 object store semantics가 다를 수 있다. backend별 visibility와 consistency를 명시한다. fsync·multipart completion을 실행하지 않았다면 durability를 과장하지 않는다.

round-trip은 restored loss만 아니라 다음 data IDs, dropout, gradient와 optimizer update를 비교한다. export-only weights와 resumable checkpoint를 구분한다. adapter/quantized artifact를 full trainer state로 오해하지 않는다.

profiler는 질문이 있을 때만 켠다. PyTorch profiler·Nsight Systems·Nsight Compute는 서로 다른 관측을 제공한다. framework operator/shape, CPU-GPU timeline, kernel instruction·memory 분석의 질문을 분리한다. 항상 모든 trace를 켜면 overhead와 disk가 run을 바꾼다.

schedule은 warmup, active와 repeat steps를 명시한다. 어떤 update와 SampleIDs가 capture됐는지 RunID에 연결한다. first step compile·cache warmup을 steady baseline과 섞지 않는다.

trace에는 prompt text나 raw sensitive data를 넣지 않는다. operator name·shape와 restricted exemplar를 사용한다. profiler artifact retention과 access를 정한다.

관측 뒤에는 가설, signal, expected counterfactual과 결정이 있어야 한다. timeline이 길다는 이유로 특정 kernel을 원인이라 하지 않는다. option 하나를 바꾼 paired run과 numerical oracle을 함께 본다.

OOM·NaN·plateau의 최초 원인을 좁힌다. OOM이 나면 phase와 requested allocation을 먼저 찾는다. data H2D, forward activation, backward saved tensor, optimizer step workspace와 checkpoint staging은 원인이 다르다. peak allocated와 reserved, live tensor inventory를 capture한다.

batch size를 줄이면 activation은 줄지만 optimizer state와 model weight는 그대로다. sequence length, visual tokens, expert routing과 padding이 실제 cost를 바꾼다. batch member별 estimated/realized tokens와 peak를 연결한다.

activation checkpointing, gradient accumulation, offload와 lower precision은 memory를 줄이는 대신 recompute, communication와 numerical state를 바꾼다. option 적용 뒤 numerical oracle과 throughput을 다시 측정한다. 단순 “OOM 해결” 표로 끝내지 않는다.

fragmentation이면 큰 contiguous allocation이 실패할 수 있다. allocator config와 shape churn, graph capture를 본다. cache empty를 매 step 호출해 우연히 통과한 run을 정상 baseline으로 승인하지 않는다.

NaN은 최초 nonfinite producer를 찾는다. loss가 NaN인 시점에는 원인이 이미 earlier logits, norm, attention score, gradient나 optimizer state에 있을 수 있다. phase별 finite checks를 coarse-to-fine으로 켜 최초 layer·operator를 양분한다. 모든 tensor를 항상 검사해 run을 바꾸지 않는다.

input에는 invalid IDs, all-masked rows, zero denominator와 extreme lengths를 검사한다. forward는 norm denominator, softmax all-invalid, exp/log와 low-precision overflow를 본다. backward는 gradient scaling, clipping 순서와 accumulation을 본다.

optimizer에서 second moment, epsilon·bias correction, weight decay와 fused kernel을 확인한다. nonfinite update 뒤 checkpoint가 저장됐는지 차단한다. last-good parent와 offending SampleIDs를 보존한다.

수정은 단순 clamp보다 수식의 domain과 data invariant를 고친다. clamp가 필요하면 threshold와 bias를 평가한다. negative fixture로 원 결함이 expected boundary에서 막히는지 확인한다.

loss plateau를 데이터·수학·optimizer로 양분한다. 작은 batch 하나에 반복 overfit이 가능한지 먼저 본다. loss가 내려가지 않으면 대규모 data quality보다 labels·mask, detach, trainable parameters와 optimizer step을 의심한다. 한 batch overfit 성공은 일반화 증명이 아니라 학습 경로 생존 검사다.

초기 loss를 vocabulary 크기의 uniform baseline과 비교하되 tokenizer·label smoothing을 반영한다. target entropy가 낮거나 pretrained model이면 단순 `log V`가 정확한 기대값은 아니다. golden parent에서 reference loss를 사용한다.

gradient norm이 있는데 weight delta가 없으면 LR, scaler skip, optimizer group과 step boundary를 본다. weight delta는 있는데 logits가 안 바뀌면 selected parameter가 unused branch이거나 변화가 너무 작을 수 있다. activation-weighted delta를 본다.

data가 반복·all-ignore거나 sample cursor가 고정된 경우도 plateau를 만든다. SampleID sequence, valid targets와 augmentation을 확인한다. scheduler LR과 clipping fraction을 같은 update clock에 놓는다.

데이터 worker가 GPU를 굶기는지 확인한다. step latency를 queue wait, decode/tokenize/collate, H2D, GPU compute와 optimizer로 나눈다. GPU utilization이 낮다는 사실만으로 kernel을 최적화하지 않는다. prefetch queue depth와 worker CPU·I/O를 본다.

worker 수를 늘리면 throughput이 좋아질 수 있지만 RNG order, memory와 contention이 바뀐다. golden SampleID·transform sequence를 유지하는지 확인한다. persistent workers와 fork/spawn mode도 환경 state다.

tokenization cache는 CPU 비용을 줄이지만 tokenizer/template generation과 raw checksum을 key에 넣는다. stale token cache가 올바른 shape로 잘못된 IDs를 반환할 수 있다. cache-off fixture를 둔다.

pinned memory와 non-blocking H2D가 실제 overlap하는지 timeline에서 본다. `.item()`, logging과 callback이 synchronization을 만드는지 ablation한다. 측정하지 않은 overlap을 주장하지 않는다.

관찰 도구와 변경 인수를 기준선에 묶는다. evaluation callback이 training state를 건드리지 않게 한다. 평가 진입 전 model train/eval mode, dropout, RNG와 data cursor를 snapshot한다. callback 뒤 원 상태가 복구되는지 본다. evaluation이 global RNG를 소비해 다음 training dropout을 바꿀 수 있다.

best-model loading은 현재 training weights를 과거 checkpoint로 바꾼다. optimizer·scheduler를 계속 사용할지 run 종료 export만 할지 명확히 한다. “best” metric, direction과 tie policy를 고정한다.

evaluation OOM이나 timeout을 training success로 숨기지 않는다. retry와 skipped eval을 disposition으로 기록한다. callback failure가 optimizer commit 전/후 어느 지점에서 발생했는지 확인한다.

metric logging은 detached copies를 사용하고 graph references를 보존하지 않는다. generated examples와 raw prompts의 privacy·retention을 관리한다. tracker outage가 core checkpoint commit을 막는지 정책을 둔다.

W&B와 TensorBoard는 관찰 기록이지 권위 state가 아니다. experiment tracker에는 scalar, config, artifacts와 system metrics가 들어갈 수 있지만 trainer의 canonical checkpoint·data ledger를 대신하지 않는다. network outage, retry와 step mismatch가 있어도 update identity를 재구성할 local evidence가 있어야 한다.

logged step가 microstep인지 optimizer update인지 정의한다. accumulation과 overflow skip에서 중복·gap이 생길 수 있다. RunID·UpdateID를 axis와 함께 기록한다. wall-clock reorder를 step reorder로 오해하지 않는다.

config upload 전에 secrets, paths와 private data를 redaction한다. artifact upload 권한·retention과 offline sync를 정한다. 외부 tracker의 latest artifact 이름을 production resolver로 사용하지 않는다.

tracker callback을 끈 paired run에서 logits·updates가 같은지 확인한다. instrumentation overhead는 별도로 측정한다. 차이가 있으면 RNG consumption, synchronization와 callback mutation을 찾는다.

golden performance 숫자의 분모. samples/s는 sample 길이·modality가 다르면 비교가 어렵다. input tokens, valid target tokens, FLOPs estimate와 optimizer updates/s를 함께 둔다. padding·packing과 dropped rows를 포함한 realized counts를 사용한다.

step time은 data wait, forward, backward, optimizer, eval·checkpoint로 분해한다. asynchronous GPU 작업을 적절히 동기화하고 warmup·steady를 분리한다. 평균, median과 tail, 반복 수를 보고한다.

peak memory는 측정 window와 reset 시점이 있어야 한다. reserved·allocated와 host pinned memory를 나눈다. profiler나 anomaly detector가 켜진 run을 steady baseline과 혼합하지 않는다.

old/new를 ABBA 같은 교차 순서로 반복해 thermal·cache drift를 줄인다. numerical·behavioral gates를 먼저 통과한 결과만 성능 비교한다. 실행하지 않은 GPU에 speedup을 일반화하지 않는다.

source upgrade를 golden run으로 인수한다. Transformers, PyTorch, PEFT, TRL 또는 CUDA dependency를 바꾸기 전에 old lock에서 package를 재생한다. 새 lock에서 같은 RunSpec의 소스 좌표s, effective options, dispatched functions와 artifacts를 diff한다.

API가 같아도 default padding, loss reduction, fused optimizer와 kernel dispatch가 바뀔 수 있다. bytes/token/mask→logits→gradient→update 경계에서 최초 차이를 찾는다. release note만으로 semantic parity를 가정하지 않는다.

upstream tests는 library 범위를 증명하며 우리 template·data·hardware 조합을 대신하지 않는다. upstream regression과 local golden fixture를 함께 실행한다. test skip과 optional dependency absence를 PASS로 세지 않는다.

expected baseline 변경은 원인과 독립 oracle 검토 뒤 승인한다. 새 결과로 snapshot을 자동 덮어쓰지 않는다. old baseline과 migration note를 보존해 historical RunID를 해석할 수 있게 한다.

CUDA·driver upgrade의 인수 범위. driver, CUDA runtime, cuBLAS/cuDNN, NCCL과 compiled extension ABI를 inventory한다. framework wheel이 포함한 runtime과 host toolkit을 혼동하지 않는다. 실제 loaded libraries와 device capability를 기록한다.

kernel 선택, numerical result와 compile cache가 바뀔 수 있다. eager/fused attention, GEMM, norm와 optimizer의 selected paths를 diff한다. PTX JIT 또는 cubin fallback이 발생하는지 본다.

CUDA error나 illegal instruction 없이 실행됐다는 것은 수치 parity가 아니다. forward·backward·two-step, checkpoint와 performance cells를 재실행한다. tolerance 변경은 error distribution과 downstream effect로 정당화한다.

single GPU 결과는 NCCL multi-rank compatibility를 증명하지 않는다. 29장에 required versions, rank-local reference와 미검증 topology를 넘긴다. CUDA upgrade 승인도 single/multi-node cells를 분리한다.

adapter target discovery를 fail-closed로 만든다. LoRA target regex가 실제 modules를 enumerate하고 expected allowlist와 비교한다. 0 match, unexpected vision/tool modules와 duplicate shared parameters를 error로 만든다. model family upgrade로 이름이 바뀌면 자동 broad match하지 않는다.

trainable count뿐 아니라 qualified names, shapes, base/adapter dtype와 modules-to-save를 manifest에 넣는다. optimizer parameter groups와 state allocation이 inventory와 맞는지 본다.

initialization이 zero effective delta인지 method에 따라 확인한다. PiSSA·LoftQ처럼 base activation/quantization을 사용하는 initialization은 standard zero-B LoRA와 다르다. 소스 분기와 initial logits를 기록한다.

save/load 뒤 adapter-only file과 base digest를 검증한다. missing modules-to-save, tied embeddings와 vocabulary resize를 negative fixture로 둔다. merge는 별도 export generation이다.

quantization metadata를 weight와 동등하게 보존한다. 4-bit weight bytes만으로 dequantization을 재현할 수 없다. group size, scales, zero points, layout, quantization type와 original shape가 필요하다. NF4, integer와 FP8을 같은 “4/8-bit”로 합치지 않는다.

bitsandbytes·custom kernels의 compute dtype, double quant와 paged optimizer branch를 actual dispatch에서 확인한다. CPU reference가 동일 packed layout을 읽을 수 없다면 dequantized slice와 layer output oracle을 둔다.

quantized base는 frozen이어도 forward activation과 dequant workspace를 가진다. memory 장부에 adapter weights, gradients, optimizer moments와 master state를 더한다. file size로 training peak를 추정하지 않는다.

checkpoint는 quantization config, library/kernel revision과 base digest를 보존한다. 다른 group metadata와 같은 packed bytes를 결합하지 않는다. merge-requantize는 원 runtime과 별도 numerical validation을 한다.

preference와 SFT golden을 같은 runner에서 분리한다. SFT row와 preference pair는 data schema, loss와 reference state가 다르다. 하나의 generic scalar-loss assertion으로 합치지 않는다. shared tokenizer/template와 model forward만 공통 경계로 사용한다.

SFT는 assistant valid targets와 next-token shift, preference는 chosen/rejected completion log-probs, reference와 beta를 검산한다. identical completions, swapped labels와 all-truncated pair를 negative fixtures로 둔다.

runner는 ObjectiveGeneration에 따라 required fields와 checkpoint dependencies를 검증한다. DPO에서 reference digest가 없거나 SFT에서 rejected가 남아 있는 schema mismatch를 차단한다.

SFT child를 preference parent로 쓰는 handoff는 immutable lineage다. optimizer를 이어 쓸지 새로 만들지는 recipe 정책이며 일반 resume로 위장하지 않는다. 두 단계의 UpdateID clocks를 분리한다.

fixed trajectory로 RL learner만 검증한다. rollout server 없이 canonical trajectories를 로드해 log-prob, reward, advantage와 loss를 계산한다. policy/reference/reward revisions, sampling processors와 terminal state를 schema에 둔다. trajectory token IDs가 learner tokenizer와 맞는지 확인한다.

PPO면 ratio·clip·value·entropy·KL, GRPO면 group baseline과 valid group count를 손으로 계산한다. group reward가 모두 같은 경우, length가 다른 responses와 stale policy를 넣는다. discard가 denominator에서 빠지는지 본다.

old log-prob를 learner current log-prob로 다시 계산해 덮지 않는다. trajectory provenance를 보존한다. frozen trajectory에서 한 update 뒤 selected policy delta를 checkpoint round trip한다.

이 run은 actor concurrency, queue와 environment safety를 증명하지 않는다. learner objective oracle만 20·29장에 넘긴다. 미검증 state를 명시하는 것이 golden 범위를 강하게 만든다.

실패 artifact를 사람이 읽을 수 있게 만든다. failure report 첫 줄에는 expected boundary와 first observed divergence를 쓴다. 소스/config/data/environment diff, tensor shape·dtype, tolerance와 SampleID를 포함한다. 최종 stack trace만 남기지 않는다.

small tensors는 값과 checksum, 큰 tensors는 selected slices·statistics와 digest를 저장한다. NaN 위치, max error와 reference scale을 기록한다. 민감 data는 raw text 대신 restricted locator를 둔다.

재현 명령은 immutable RunSpec을 가리키고 baseline regeneration을 기본으로 하지 않는다. failure frequency, seeds와 event ordering을 남긴다. flaky를 사라졌다는 이유로 닫지 않는다.

수정 뒤 원 negative fixture가 expected invariant에서 막히고 neighboring fixtures가 유지되는지 확인한다. postmortem에 root cause, fix source, new guard와 regression ID를 연결한다.

## 28.15 evidence package와 release를 독립 검토한다

마지막 절은 새 기준값을 더 만들지 않는다. 앞 절의 config, oracle, fault result와 resume trace를 제3자가 깨끗한 process에서 재생할 수 있는 package로 묶는다. 성공 로그가 아니라 negative fixture까지 같은 경계에서 실패하는지가 release 판정의 핵심이다.

### 28.15.1 단일 GPU release certificate를 만든다

certificate는 실행한 objective, model/adapter, data/tokenizer, dependency·CUDA/GPU, dtype·kernel과 option 범위를 적는다. bytes/token/mask, forward/loss, backward, two-step optimizer와 checkpoint 결과를 표로 둔다.

performance에는 valid target denominator, warmup/steady, repeats와 profiler state를 적는다. numerical failure를 speedup으로 상쇄하지 않는다. unsupported·not-executed cells를 명시한다.

failure suite에는 corrupt row, all-ignore, overflow, OOM, checkpoint kill, stale cache와 source upgrade가 있다. 모두 성공할 필요가 아니라 기대 boundary에서 올바르게 실패·복구해야 한다.

독립 검토자가 manifest만으로 재생하고 다음 update까지 맞추면 certificate를 승인한다. 29장은 이 rank-local reference 위에 collective·sharding·fault를 추가한다. single GPU 결과를 cluster correctness로 확대하지 않는다.

옵션 상호작용을 pairwise graph로 시험한다. mixed precision, activation checkpointing, compile, LoRA, quantized base와 packing은 각각 독립 option처럼 보여도 조합에서 새 branch를 만든다. compile+checkpointing은 recompute graph를 바꾸고, QLoRA+compile은 dequantization kernel과 graph break를 만들 수 있다. 모든 조합을 전수할 수 없다면 위험 edge를 우선한다.

노드는 option과 effective state, edge는 함께 켰을 때 공유하는 module·kernel·RNG·checkpoint state다. 각 high-risk edge에 최소 fixture를 배정한다. pairwise PASS를 세 개 이상의 조합에 자동 일반화하지 않는다.

조합별 requested/effective config, 소스 분기와 dispatched kernels를 비교한다. unsupported 조합이 silent fallback인지 hard error인지 확인한다. fallback이 수치적으로 맞아도 성능·memory support는 별도다.

실패하면 option 하나씩 끄는 ablation으로 최초 상호작용을 좁힌다. baseline snapshot을 조합별로 무분별하게 늘리지 않고 common numerical oracle을 재사용한다. support matrix에 exact versions와 hardware를 둔다.

### 28.15.2 fixture schema와 artifact를 version한다

fixture는 raw rows, expected token/mask, tensor oracle, negative injection과 tolerances를 가진다. schema version이 없으면 새 field의 부재와 old fixture corruption을 구분하지 못한다. loader가 version별 required fields와 migration을 검증한다.

expected outputs는 source data·model과 별도 content-addressed artifact다. 같은 코드가 계산한 값으로 즉시 expected를 덮어쓰면 공통 오류를 잡지 못한다. 독립 계산, old signed baseline과 reviewer 승인을 요구한다.

tolerance는 tensor/metric별 absolute·relative, dtype와 shape 범위를 가진다. 하나의 넓은 epsilon으로 모든 값을 통과시키지 않는다. NaN/Inf는 tolerance 비교 전에 실패한다. discrete IDs·masks와 parameter ownership은 exact다.

fixture migration은 old/new를 같은 runner에서 실행해 의미 차이를 설명한다. old fixture를 삭제해 historical regression을 잃지 않는다. deprecated support는 reason과 last-valid environment를 보존한다.

### 28.15.3 negative fixture가 예상 경계에서 실패하게 한다

wrong tokenizer, swapped labels, all-ignore row, stale cache, duplicate parameter, Inf gradient, corrupt checkpoint와 adapter-base mismatch를 의도적으로 만든다. 단순히 run이 실패하는 것으로는 부족하다. 각각 admission, collator, loss, optimizer 또는 loader의 기대 boundary에서 고유 reason으로 실패해야 한다.

너무 늦은 실패는 bad state가 이미 durable해졌을 수 있다. 예를 들어 wrong tokenizer가 evaluation에서만 발견되면 여러 updates가 오염됐다. invariant를 가능한 최초 owner에 둔다. validation cost가 큰 경우 preflight sample과 periodic full audit를 조합한다.

negative injection은 production code의 hidden backdoor가 되지 않게 test harness에서만 활성화한다. injection flag, target와 expected error를 artifact에 남긴다. 성공 경로에 같은 flag가 들어가지 않았는지 확인한다.

fix 뒤 negative가 통과해버리면 guard regression이다. error message string exact match보다 structured error code와 boundary를 검증한다. 인접 valid fixture가 계속 통과하는지 본다.

CI에서 golden run을 어떻게 나눈다. 매 commit에는 CPU/static schema·token/mask·small math tests를 실행할 수 있다. GPU smoke에는 forward/backward·one update와 critical negative fixtures를 둔다. nightly에는 two-step, checkpoint kill, compile·dtype와 performance repeats를 실행한다.

hardware 비용 때문에 모든 GPU architecture를 매 commit 테스트할 수 없다. change impact graph로 CUDA/kernel·dtype changes를 relevant cells에 라우팅한다. scheduled matrix는 오래된 지원 조합과 최신 조합을 모두 포함한다.

CI retry가 flaky failure를 숨기지 않게 첫 실패 artifact를 보존한다. retry count와 success pattern을 metric으로 둔다. infrastructure error와 product regression을 disposition하되 unknown을 PASS로 바꾸지 않는다.

baseline regeneration job은 일반 CI와 권한을 분리한다. PR 코드가 expected를 동시에 바꾸면 독립 review가 필요하다. performance threshold는 noise와 runner health를 고려하되 numerical gates는 완화하지 않는다.

GPU 환경 fingerprint를 정확히 남긴다. GPU name만으로 동일 환경이 아니다. UUID 또는 stable device identity, compute capability, memory size, power/clock policy, MIG mode, driver와 firmware, PCIe link와 thermals를 기록한다. 공유 환경에서는 다른 process와 utilization도 본다.

loaded CUDA runtime, cuBLAS/cuDNN/NCCL, extension build flags와 PTX/cubin target을 남긴다. `nvcc --version` 하나가 Python process가 실제 load한 library를 증명하지 않는다. container와 host driver 경계를 구분한다.

performance run은 power limit, clocks와 temperature를 관측한다. correctness run은 하드웨어 오류·ECC event가 없는지 확인한다. single GPU에서 NVLink/NCCL topology를 추론하지 않는다.

환경 fingerprint가 다르면 exact performance baseline을 바로 비교하지 않는다. numerical/behavioral portability와 performance support를 분리한다. 새 fingerprint는 child evidence cell이다.

memory snapshot을 tensor lifetime과 연결한다. allocated blocks 목록만으로 어떤 tensor가 살아 있는지 알기 어렵다. phase markers와 saved-tensor hooks, module boundaries를 제한적으로 사용해 large allocations의 owner를 찾는다. hook overhead를 별도 run으로 둔다.

forward peak에는 attention/MLP activations, backward에는 saved tensors와 gradients, optimizer에는 moments·workspace가 있다. checkpointing은 saved activation을 줄이지만 recompute temporary를 만든다. compile은 memory planning을 바꾼다.

Python reference, callback와 retained outputs가 GPU tensor를 붙들 수 있다. loss history에 tensor 자체를 append하지 않는지 확인한다. `detach().cpu()`도 H2D/D2H와 host memory 비용을 가진다.

leak fixture는 동일 step을 반복해 allocated/reserved trend와 live object를 본다. allocator cache 성장은 leak와 다르다. `empty_cache` 뒤 숫자만으로 판정하지 않고 active blocks와 reachability를 본다.

small overfit을 올바르게 해석한다. 8~32개 examples를 반복해 near-zero training loss에 접근하는 시험은 labels, gradient와 optimizer가 연결됐는지 보는 강한 smoke다. pretrained stochasticity, label smoothing와 noisy/multiple targets가 있으면 zero가 기대값이 아닐 수 있다. objective별 예상 floor를 정한다.

overfit이 안 되면 learning rate, trainable inventory, loss mask, dropout·augmentation와 data duplication을 본다. eval mode로 우연히 dropout을 끈 결과와 training path를 구분한다. capacity가 너무 작은 fixture인지도 확인한다.

overfit 성공은 generalization, data quality와 distributed correctness를 보장하지 않는다. 오히려 leakage·label shortcut으로 너무 빨리 맞을 수 있다. held-out counterexample와 shuffled-label control을 둔다.

curve는 successful update와 valid targets x축으로 본다. skipped overflow와 corrupt rows가 있으면 wall-clock step과 다르다. selected sample predictions와 margins를 추적한다.

deterministic 모드의 의미와 비용. framework deterministic algorithms option은 알려진 nondeterministic operations를 error 또는 deterministic implementation으로 바꿀 수 있다. 모든 hardware·library·data order의 bitwise 재현을 보장하지 않는다. exact supported scope를 기록한다.

deterministic mode가 fallback kernel과 성능 저하를 만들 수 있다. correctness debug run과 production-like performance run을 분리한다. 두 경로의 numerical difference와 dispatch를 비교한다.

atomic reduction, stochastic rounding, asynchronous worker와 uninitialized memory가 nondeterminism을 만들 수 있다. seed만 반복하기보다 first-divergence distribution을 본다. tolerance 내 차이도 final behavior에 amplification될 수 있다.

reproducibility 등급을 bitwise, numerical, behavioral와 statistical로 나눈다. 어떤 artifact와 환경에서 어느 등급을 달성했는지 certificate에 쓴다. 요구보다 낮은 등급이면 reason과 compensating tests를 둔다.

resume와 dataloader prefetch의 틈을 검사한다. checkpoint cursor가 consumed batch를 가리키는지 yielded·prefetched batch를 가리키는지 확인한다. worker queue에는 아직 trainer가 소비하지 않은 rows가 있을 수 있다. save 뒤 process가 죽으면 재시작에서 duplicate 또는 skipped samples가 생긴다.

strict replay가 필요하면 batch plan과 consumed acknowledgements를 ledger에 둔다. safe update boundary에서 cursor를 commit하고, prefetch는 재생 가능하게 만든다. 성능과 exactness의 tradeoff를 명시한다.

augmentation RNG는 sample identity와 epoch/occurrence에서 stateless하게 만들면 worker count 변화에 강할 수 있다. global RNG 방식이면 worker state와 queue를 복구해야 한다. 실제 transform source를 확인한다.

duplicate replay가 허용되는 at-least-once policy라면 optimizer update와 data exposure accounting에 반영한다. exactly-once라고 과장하지 않는다. deletion/tombstone rows는 resume admission에서 다시 검사한다.

산출물·구조 변경·진단 절차를 독립 검토한다. checkpoint와 export를 다른 산출물로 관리한다. training checkpoint는 model, optimizer, scheduler, scaler, RNG와 data cursor를 포함한다. inference export는 merged weights, quantization과 serving config를 가질 수 있지만 resume state는 없다. 두 artifact를 같은 `latest` alias 아래 섞지 않는다.

adapter-only, full merged, quantized와 sharded exports는 parent checkpoint와 transformation edge를 가진다. tool revision, dtype/layout와 digest를 기록한다. export 성공 뒤 golden inference fixture를 실행한다.

training resume loader가 export-only artifact를 받으면 required keys 부족으로 fail-closed해야 한다. serving loader도 incomplete optimizer shards를 무시하고 model weights로 추측하지 않는다. artifact kind와 schema를 검증한다.

rollback은 목적별 alias를 parent 종류에 맞게 되돌린다. training continuation과 serving release가 같은 시점일 필요는 없다. lineage가 둘의 관계를 설명한다.

tokenizer·template upgrade는 model upgrade다. weights가 같아도 tokenizer vocabulary, normalization, special IDs와 chat template가 바뀌면 input IDs와 labels가 달라진다. upgrade를 단순 preprocessing patch로 처리하지 않는다. fixed raw conversations의 rendered bytes·IDs·masks를 old/new로 diff한다.

vocabulary resize가 있으면 embedding/head rows, tied state와 optimizer moments를 migrate해야 한다. 새 token initialization과 no-decay group을 기록한다. old checkpoint loader가 shape mismatch를 어떻게 처리하는지 본다.

template role marker 변화는 assistant loss span과 generation stop을 바꾼다. SFT, DPO와 serving parity를 함께 평가한다. response-only collator가 새 marker를 찾지 못해 all-ignore가 되는 negative fixture를 둔다.

cache와 tokenized shards는 tokenizer/template generation으로 무효화한다. raw path와 model name만 key로 쓰지 않는다. upgrade child는 새 training generation이다.

model architecture upgrade의 state mapping. layer 수, hidden width, attention heads, MLP와 norm이 바뀌면 old weights를 어떻게 이관할지 명시한다. `strict=False` load가 missing/unexpected keys를 숨길 수 있다. 모든 disposition을 allowlist한다.

head count만 바뀌어도 QKV layout과 RoPE mapping이 달라질 수 있다. reshape가 성공한다고 의미가 맞지 않는다. small tensor pattern을 넣어 conversion 전후 index mapping을 검증한다.

MoE expert 추가·제거는 router·experts와 optimizer state를 바꾼다. random init, clone 또는 merge 정책을 기록하고 load balance fixture를 둔다. dense→MoE를 동일 resume로 부르지 않는다.

architecture conversion 뒤 forward, selected activations, loss와 one/two updates를 새 baseline으로 검증한다. old result와 exact equality가 목표가 아닐 수 있지만 변환 의도와 차이를 설명한다.

source function을 실제 call trace와 맞춘다. 문서에 `Trainer.training_step`이나 `AdamW.step`을 적었다고 우리 run이 그 symbol을 호출한 것은 아니다. subclass override, Accelerate wrapper, compiled function와 fused optimizer가 다른 경로를 탈 수 있다. loaded class와 lightweight call trace를 확인한다.

소스 기록에는 caller→callee, guard condition, input/output state와 revision을 둔다. line number는 commit과 함께 사용한다. dynamic code 또는 remote code는 별도 trust boundary다.

trace instrumentation이 dispatch를 바꾸지 않는지 paired run으로 본다. Python profiler가 fused path를 fallback시키는 경우가 있다. source-static evidence와 runtime-selected evidence를 구분한다.

upgrade 뒤 symbol 이름이 같아도 body·guard가 바뀔 수 있다. content hash와 golden behavior를 함께 갱신한다. comment나 docstring만으로 state transition을 주장하지 않는다.

실용 디깅 순서로 독립 재생한다. 디버깅은 artifact와 effective config의 diff에서 시작해 raw bytes→IDs·labels·mask 비교로 이어 간다. 여기까지 같다면 parameter init와 first logits, loss numerator/count, gradients·clipping·skip을 차례로 확인한다. 마지막으로 optimizer delta와 restored next update를 비교한다.

이 순서는 최종 metric에서 거꾸로 추측하는 일을 막는다. source가 바뀌었더라도 first divergence가 data라면 kernel을 조사하지 않는다. logits까지 같고 gradient가 다르면 backward·loss scaling을 본다.

각 boundary에 pass/fail/not-captured와 artifact locator를 둔다. capture가 없으면 “같다”고 쓰지 않는다. 조사 중 새 관측을 추가하되 baseline run을 오염시키지 않는다.

수정 뒤 같은 순서로 root cause가 사라졌고 downstream update가 회복됐는지 확인한다. expected snapshot을 먼저 바꾸지 않는다. 인접 negative fixtures와 performance를 재실행한다.

단일 GPU에서 끝내지 말아야 할 것. collective order, rank-local shard, all-reduce denominator, pipeline bubbles, network·NCCL failure와 distributed checkpoint는 한 GPU에서 증명할 수 없다. single GPU에서 mock한 collective는 수학 oracle일 뿐 실제 transport evidence가 아니다.

반대로 data/token/loss/optimizer 의미를 single GPU에서 못 맞춘 채 cluster로 가면 divergence 원인을 병렬화 탓으로 오인한다. rank-local oracle과 expected global reduction을 먼저 고정한다.

29장에 넘길 항목은 canonical batches, per-sample loss sums/counts, selected gradients·weights, optimizer next state, RNG plan와 checkpoint schema다. target topology와 collectives는 미실행으로 명시한다.

cluster result는 이 package를 확장하며 덮어쓰지 않는다. single GPU baseline과 distributed child의 first divergence를 비교한다. 이 경계가 규모 확장의 출발점이다.

데이터 오류를 optimizer 전에 차단하는 표. 입력 admission은 raw checksum, schema, encoding, length와 rights/tombstone을 확인한다. tokenizer 단계는 special IDs, template와 round trip 범위를 본다. collator는 labels, mask, position, packing boundary와 valid target count를 확인한다. 각 단계는 고유 disposition을 낸다.

corrupt row를 skip할지 run을 중단할지는 정책이다. skip하면 SampleID, reason과 lost target mass를 기록한다. replacement sample을 뽑으면 data order·RNG가 바뀐다. 조용한 replacement를 금지한다.

outlier length, duplicate SampleID와 all-ignore row를 fixture에 넣는다. 정확히 어느 boundary에서 차단되는지 확인한다. invalid row가 model forward까지 갔다면 admission owner를 앞당긴다.

권리 tombstone은 cache·prefetch 뒤에도 재검사할 수 있어야 한다. golden run에서도 삭제된 row가 resume queue에서 살아나는 negative fixture를 둔다. 데이터 correctness와 governance를 별개라 하지 않는다.

model initialization을 content hash로만 보지 않는다. random initialization은 architecture, initializer 함수, seed, parameter creation order와 dtype의 결과다. seed가 같아도 module 등록 순서나 fused initialization이 바뀌면 weights가 달라질 수 있다. selected parameter statistics와 digest를 저장한다.

pretrained load에서는 expected, missing, unexpected, resized와 transformed keys를 disposition한다. `strict=False`가 성공했다고 올바른 load가 아니다. tied aliases와 device/dtype cast를 확인한다.

adapter initialization은 base activation이나 quantization에 의존할 수 있다. LoftQ·PiSSA류는 data/sample, SVD와 precision을 state로 갖는다. standard LoRA zero-effective delta와 같은 expected를 쓰지 않는다.

initial checkpoint를 저장해 repeated attempts가 동일 start를 사용할 수 있게 한다. initialization code 변경을 training instability로 오인하지 않는다. first forward 전에 parameter inventory를 승인한다.

한 layer씩 freeze·unfreeze를 검증한다. freeze policy는 config pattern보다 actual parameters의 `requires_grad`, optimizer inclusion과 train/eval mode다. module별 trainable count와 qualified names를 출력한다. frozen dropout/stochastic depth가 동작하는지도 별도다.

unfreeze 시 새 optimizer group을 추가하거나 optimizer를 재생성한다. moments, step age, LR·weight decay를 어떻게 시작할지 정한다. old scheduler가 group 수와 ordering을 지원하는지 확인한다.

fixture는 projector, 마지막 transformer block, norm·head를 단계적으로 연다. expected gradients와 first update를 본다. accidentally trainable base나 omitted adapter를 negative fixture로 둔다.

checkpoint 전후 trainable inventory와 optimizer state가 같다. stage transition은 일반 resume와 다른 child generation이다. 이전 stage의 cache·feature가 유효한지도 검토한다.

gradient hooks를 진단용으로 안전하게 쓴다. hook는 gradient norm, None·nonfinite와 selected tensor를 볼 수 있지만 반환값으로 gradient를 바꾸거나 reference를 보존할 수 있다. read-only hook와 transform hook를 구분한다. golden baseline에는 필요한 최소 hook만 둔다.

shared/tied parameter에는 hook 호출 수와 accumulation 의미가 다를 수 있다. module backward hook보다 parameter hook가 원하는 객체를 보는지 확인한다. activation checkpointing 재실행도 고려한다.

hook output은 detached aggregate와 limited slice로 저장한다. full gradient dump는 memory·I/O와 민감 정보 위험이 있다. anomaly trigger 때만 restricted artifact를 만든다.

hook off paired run에서 weights·updates가 같은지 본다. instrumentation이 graph capture나 compile을 깨면 별도 debug generation으로 처리한다. 관측 도구가 권위 state를 바꾸지 않게 한다.

optimizer zero_grad의 set-to-none 의미. `zero_grad(set_to_none=True)`는 gradient tensor를 0으로 채우는 대신 None으로 만든다. optimizer가 None gradient parameter를 skip하는지 zero gradient와 동일하게 weight decay·moment를 적용하는지 source에서 확인한다. 둘은 항상 같지 않다.

unused branch, frozen parameter와 실제 zero derivative를 구분하려면 None/zero가 중요하다. accumulation 시작과 optimizer step 뒤 expected state를 검사한다. stale gradient가 다음 update에 남는 negative fixture를 둔다.

gradient clipping과 norm calculation이 None parameters를 어떻게 제외하는지 본다. adapter target이 한 batch에서 unused일 때 optimizer state가 어떻게 진행하는지도 확인한다.

checkpoint가 gradient를 저장하지 않는다면 accumulation 중간 resume 정책과 연결한다. zero_grad timing을 step event ledger에 넣는다. callback이 너무 일찍 gradients를 지우지 않게 한다.

weight decay parameter group을 검사한다. bias와 normalization weight를 no-decay로 두는 관습은 module/parameter 이름 규칙으로 구현될 수 있다. 이름 substring만 사용하면 custom modules를 잘못 분류할 수 있다. actual module type, parameter identity와 group을 inventory한다.

tied embedding/head가 서로 다른 group에 중복 들어가면 오류다. adapter matrices, embeddings, sparse memory와 MoE experts의 policy를 명시한다. Muon과 AdamW split도 group ownership이다.

selected scalar에서 decay-only update를 손으로 계산한다. gradient 0과 None의 동작을 구분한다. fused optimizer가 group option을 실제 적용하는지 next weight에서 확인한다.

architecture·adapter upgrade 뒤 parameter names가 바뀌면 allowlist diff를 review한다. 새 parameter를 default group에 조용히 넣지 않는다. checkpoint optimizer group mapping도 qualified identity로 검증한다.

loss component가 optimizer에 실제로 연결되는지 본다. LM loss, auxiliary router/load-balance, contrastive, KL와 safety reward가 total에 더해질 수 있다. logging scalar만 존재하고 total graph에 연결되지 않은 component를 찾는다. 각 component의 gradient를 selected parameter에서 따로 계산할 수 있다.

weight가 0인 component는 expected zero gradient이고, detached metric은 관측용이다. weight schedule과 runtime effective value를 기록한다. NaN component를 `nan_to_num`으로 숨기는 정책은 disposition을 남긴다.

component별 numerator/count와 global total equation을 손으로 맞춘다. 서로 다른 단위를 평균 하나로 합치기 전에 normalization을 검토한다. local batch에서 없는 component의 zero-count policy를 정한다.

한 component를 disable한 paired run에서 예상 parameter delta가 사라지는지 본다. 이것이 objective wiring의 강한 fixture다. performance effect는 별도로 측정한다.

tied weights와 optimizer serialization. input embedding과 LM head가 tied라면 checkpoint save/load 뒤에도 alias가 유지되는지 확인한다. 두 별도 tensors로 로드돼 값만 같은 상태는 다음 update에서 갈라질 수 있다. storage identity와 gradient accumulation을 본다.

safetensors 같은 format은 tensor alias 표현에 제한이 있을 수 있어 framework save/load 정책이 canonical key와 re-tying을 담당한다. missing duplicate key를 corruption으로 오인하지 않고 model contract를 따른다.

optimizer state는 logical parameter 하나에 하나여야 한다. duplicate parameter registration이나 load mapping이 moments를 두 벌 만들지 않는지 확인한다. vocabulary resize 뒤 tied state migration을 시험한다.

fixture는 save 전 한 row update, load 뒤 alias, 다음 update와 output head를 비교한다. merge·quantization export에서도 tied semantics가 어떻게 변하는지 명시한다.

tokenizer vocabulary resize의 골든 절차. 새 special token을 추가하면 tokenizer length, model embedding와 LM head rows를 resize한다. new row initialization, padding multiple와 tied relation을 기록한다. special ID가 template와 config에 일치하는지 본다.

optimizer가 이미 존재하면 resized parameter identity와 state migration 문제가 생긴다. resize를 optimizer 생성 전에 할지, moments를 확장할지 정책을 둔다. old parameter state를 새 object에 잃지 않았는지 확인한다.

old tokens의 embedding rows가 bitwise 또는 tolerance 내 보존되는지, new token만 초기화됐는지 검사한다. shifted IDs가 발생하는 tokenizer migration은 단순 row append와 다르다.

checkpoint·adapter는 vocabulary generation을 참조한다. old adapter의 modules-to-save head shape가 새 base와 맞는지 본다. serving tokenizer와 training export parity를 재검증한다.

exact·numerical·behavioral oracle의 우선순위. IDs, labels, masks, SampleID order, parameter ownership와 checkpoint inventory는 exact여야 한다. float tensors는 dtype·kernel에 따라 numerical tolerance를 사용한다. generation text와 metric은 behavioral/statistical oracle이 될 수 있다.

아래 단계의 느슨한 oracle로 위 단계의 exact mismatch를 가리지 않는다. token IDs가 다른데 final loss가 비슷하다고 PASS하지 않는다. checkpoint keys가 빠졌는데 next loss 한 번이 맞아도 resume complete가 아니다.

tolerance는 baseline variance와 수치 분석에서 정한다. 결과를 본 뒤 넓히지 않는다. max·mean error, relative scale와 downstream delta를 보고한다. distributional metric은 seeds와 interval을 가진다.

certificate는 각 boundary의 oracle 등급을 표시한다. hardware upgrade에서 bitwise가 불가능해도 numerical·behavioral을 증명할 수 있지만 exact fields는 유지한다. 지원 요구와 실제 달성 등급을 비교한다.

baseline 승격의 독립 검토. 새 source·dependency가 의도적으로 numerical result를 바꿀 수 있다. 먼저 old baseline 실패와 first divergence를 설명한다. 논문·source change가 새로운 수식을 요구하는지, bug fix인지, kernel rounding인지 구분한다.

새 expected는 candidate code와 독립 reference 또는 hand calculation으로 검증한다. 같은 함수가 output과 snapshot을 둘 다 만들지 않는다. reviewer는 old/new tensors, tolerance와 downstream effect를 본다.

승격 commit은 reason, affected fixtures, environment와 approver를 가진다. old artifact를 보존한다. performance 개선 때문에 numerical mismatch를 승인하지 않는다.

baseline change 뒤 negative fixtures가 여전히 실패하고 checkpoint next update가 일관되는지 전수 실행한다. 29장의 distributed reference가 stale가 되므로 dependency graph를 따라 재검증한다.

flaky failure를 통계로 숨기지 않는다. 100번 중 한 번 hang·NaN이 나면 평균 성공률 99%로 승인할 수 없다. failure class, seed, input, thermal·memory와 event order를 보존한다. reproduce budget과 confidence를 사전 계획한다.

flaky가 hardware·driver·race인지 source logic인지 양분한다. deterministic mode, stream sync와 reduced concurrency는 진단용 counterfactual이다. debug option에서 사라졌다는 사실만으로 root cause를 확정하지 않는다.

retry-to-green CI는 첫 실패 artifact를 유지하고 flaky ledger를 증가시킨다. recurrence threshold와 owner, quarantine policy를 둔다. critical checkpoint corruption은 낮은 빈도라도 hard fail이다.

fix 뒤 failure injection과 long-enough repeat를 수행한다. 0 observed가 probability 0은 아니므로 실행 범위와 upper bound를 적는다. production monitoring으로 남은 risk를 이어받는다.

data checksum과 semantic checksum. raw file hash는 bytes가 같은지 보장하지만 sample selection·decode·normalization 의미는 보장하지 않는다. dataset generation은 row IDs, schema, transformation code와 counts를 가진다. sample-level processed digest를 golden fixture에 둔다.

Parquet/JSON ordering, locale·Unicode normalization과 library decoder가 결과를 바꿀 수 있다. logical rows를 canonical serialization로 hash할지 정의한다. float/media transforms는 numerical digest 범위를 사용할 수 있다.

semantic checksum은 label distribution, token lengths, source mixture와 valid target totals 같은 summary다. exact checksum을 대신하지 않고 large dataset drift를 조기에 찾는다. matched counts가 same data를 증명하지 않는다.

golden RunSpec은 immutable small subset의 exact processed artifacts를 가진다. full corpus는 manifest와 sampling proof를 연결한다. data upgrade가 model source upgrade와 함께 일어나면 axes를 분리한다.

evaluation fixture를 training fixture와 분리한다. training golden batch는 gradient와 update를 검증하고 evaluation fixture는 generation·metric·normalization을 검증한다. 같은 examples를 써도 역할과 leakage를 명시한다. private benchmark를 training overfit fixture로 사용하지 않는다.

generation은 prompt bytes·IDs, decode, stop와 max tokens를 고정한다. metric은 raw output, parser, normalization과 denominator를 가진다. callback의 model mode와 RNG 복구도 본다.

evaluation failure가 checkpoint save·best model selection에 어떤 영향을 주는지 시험한다. missing score를 0이나 이전 value로 위장하지 않는다. best alias는 metric generation을 참조한다.

24장의 evaluator/judge revision을 가져오되 single GPU golden은 작은 harness cell만 검증한다. 전체 benchmark 성능을 주장하지 않는다. objective와 evaluation code upgrade를 독립 axes로 둔다.

한 번의 성공을 reproducible package로 바꾼다. package에는 immutable manifest, source/dependency lock, raw+processed fixture, command, environment fingerprint와 expected artifacts가 있다. output은 events, selected tensors, checkpoint와 report를 content-addressed 경로에 쓴다.

새 작업자는 숨은 shell history나 mutable cache 없이 실행할 수 있어야 한다. 필요한 credentials·external services는 없거나 명시적 stub을 쓴다. network access가 필요하면 source pin과 offline mirror 정책을 둔다.

runner는 preflight, run, validate와 package 단계를 분리한다. validation 실패 시 artifact를 보존하고 baseline을 갱신하지 않는다. cleanup이 evidence를 지우지 않게 한다.

README는 단계 설명보다 실패 시 first-divergence 경로와 support 범위를 쓴다. machine-readable ledger와 사람이 읽는 summary가 같은 RunID를 사용한다. 재생 결과를 parent에 덮지 않는다.

골든 런에서 하지 말아야 할 성능 주장. 작은 model·sequence의 tokens/s를 대규모 training throughput으로 외삽하지 않는다. kernel occupancy, memory bandwidth, communication과 data pipeline regime이 다르다. golden은 회귀와 경로 확인에 적합하다.

한 GPU 결과로 multi-node scaling, NCCL overlap이나 checkpoint filesystem 성능을 말하지 않는다. compile warmup이 지배하는 short run의 평균을 steady speedup으로 쓰지 않는다. profiler-on 결과를 production baseline으로 쓰지 않는다.

hardware·dtype·shape를 실제 실행하지 않았다면 source support와 runtime result를 구분한다. “지원 가능”과 “검증됨”을 별도 status로 둔다. vendor peak 수치를 measured throughput처럼 쓰지 않는다.

성능 결론에는 denominator, repetitions, variance, environment와 numerical gates가 있다. 최적 option 하나만 보고하지 않고 baseline과 cost·memory·fallback을 함께 둔다.

실패 복구 실습의 완결 조건. corrupt data는 admission에서, wrong label은 oracle에서, Inf gradient는 optimizer commit 전에, OOM은 phase trace에서, checkpoint kill은 parent fallback에서, stale dependency는 preflight에서 검출한다. 각 실패의 EvidenceID를 연결한다.

복구는 단순 rerun이 아니다. bad artifacts를 quarantine하고 last complete parent, data cursor와 RNG를 선택한다. duplicate/skip policy를 적용한다. root cause guard를 추가한다.

수정 뒤 original failing case, neighboring valid와 negative suite, checkpoint next update와 performance를 재검증한다. incident report에는 symptom, first divergence, cause, fix와 regression fixture가 있다.

새 작업자가 같은 report로 failure를 재주입하고 expected recovery를 얻으면 runbook이 실행 가능하다. 사람의 기억에 의존하면 미완성이다. recovery path도 package 일부다.

29장에 넘길 수학 oracle. canonical global batch의 per-sample loss sums/counts와 concatenated reference를 저장한다. selected parameter의 per-sample 또는 microbatch gradients, accumulated global gradient와 expected AdamW/Muon update를 둔다.

분산 DP에서는 rank-local sums/counts를 어떤 collective로 합칠지 식을 제공한다. TP/PP/CP에서는 global logical tensor와 expected shard layout을 제공한다. single GPU가 actual collectives를 증명하지는 않는다.

RNG는 global sample plan과 rank derivation policy를 제안하되 target world size에서 검증 전 `NOT_RUN`이다. checkpoint schema는 logical keys와 shard-independent digest를 제공한다.

29장은 collective ordinal, process groups, communication bytes와 failure를 추가한다. global reconstructed loss·gradient·next update가 oracle와 맞아야 한다. 차이가 있으면 first divergent boundary를 찾는다.

30장에 넘길 recipe option evidence. 각 option은 requested/effective value, parser/source guard, affected state, checkpoint field, observability와 golden fixture를 가진다. batch, accumulation, packing, precision, compile, adapter, checkpoint와 evaluation options를 포함한다.

option 조합의 known support와 conflict, fallback을 표로 둔다. default는 dependency/hardware에 따라 바뀔 수 있으므로 resolved value를 기록한다. unknown·deprecated는 disposition한다.

30장의 end-to-end recipe는 이 option evidence를 data/model/objective 선택과 연결한다. 숫자를 복사하지 않고 어떤 상황에서 어떤 효과·위험이 있는지 설명한다. actual scale run은 별도 evidence다.

recipe upgrade가 option semantics를 바꾸면 golden fixture와 source anchor를 stale로 돌린다. reader가 옵션을 바꿨을 때 어느 tensor·state·metric이 변할지 예측할 수 있어야 한다.

승인자가 답해야 할 반증 질의. 인수자는 RunID 하나에서 raw SampleID, tokens·mask, first logits, loss numerator/count, selected gradient, optimizer delta와 checkpoint next update를 차례로 조회한다. 각 값은 source·artifact와 oracle을 가진다.

그다음 option 하나를 바꾸어 effective branch와 invalidated evidence를 예측한다. compile은 graph/kernel, mixed precision은 casts/scaler, packing은 mask/denominator, adapter는 parameter ownership을 바꾼다. 실제 child result와 맞는지 본다.

마지막으로 corrupt row와 checkpoint kill을 주입한다. expected boundary에서 실패하고 last complete parent로 복구하며 SampleID/RNG/update clock이 정책과 맞아야 한다. stale artifact가 latest로 보이면 실패다.

이 세 질의가 재생되고 실행하지 않은 hardware·topology cells가 명시되면 단일 GPU golden package를 승인한다. 승인은 training 전체가 옳다는 선언이 아니라 다음 규모와 변경을 비교할 강한 좌표계의 완성이다.

오류 메시지보다 상태 전이를 테스트한다. 예외 문구는 dependency release마다 바뀔 수 있다. golden negative test는 error code, 실패 boundary, durable artifact 부재와 parent 유지 같은 상태를 본다. 문자열 한 줄 exact match로 취약하게 만들지 않는다.

예외가 잡힌 뒤 trainer가 계속 실행하는지, retry하는지, run을 중단하는지도 검증한다. fatal corruption을 skip으로 바꾸거나 transient I/O를 영구 failure로 처리하면 운영 의미가 달라진다. disposition policy를 source와 manifest에 둔다.

부분 state가 memory·disk에 남으면 다음 attempt가 읽지 않게 quarantine한다. callback이 failure를 PASS metric으로 덮지 않는지 확인한다. 실패 event와 cleanup event를 같은 AttemptID에 연결한다.

golden artifact의 최소 보존과 privacy. 재현을 위해 raw prompts와 full gradients를 무제한 보존할 필요는 없다. public/synthetic fixture를 우선하고 민감 row는 restricted locator, checksum과 redacted view를 사용한다. artifact별 access와 retention을 둔다.

token IDs도 rare sequence에서 정보를 노출할 수 있고 gradients는 data leakage 위험이 있다. selected aggregate·slice만 보존하며 원본이 필요한 경우 권한을 제한한다. tracker·profiler upload 전에 redaction한다.

삭제 요청은 fixture, caches, checkpoints와 reports의 lineage를 따라 처리한다. baseline 재현성 때문에 rights floor를 무시하지 않는다. 허용된 대체 fixture로 child baseline을 만든다.

증거 접근이 제한되면 authorized verifier가 digest·counts와 pass attestation을 제공할 수 있다. 독자는 그 제한을 알아야 한다. inaccessible artifact를 자동 success로 세지 않는다.

notebook보다 runner를 권위로 둔다. notebook은 탐색과 설명에 유용하지만 cell 실행 순서, hidden state와 수동 수정이 재현성을 해친다. canonical golden은 non-interactive runner와 immutable config로 실행한다. notebook은 runner artifact를 읽어 시각화한다.

노트북에서 발견한 계산은 script/unit fixture로 옮긴다. output screenshot이 아니라 machine-readable values와 oracle을 저장한다. kernel restart 뒤 처음부터 실행 가능한지 본다.

환경 설치와 data download도 runner preflight 또는 documented immutable step으로 둔다. 개인 cache에 우연히 있던 file을 사용하지 않는다. network-disabled replay 가능성을 높인다.

설명용 notebook이 baseline을 갱신하거나 production alias를 바꿀 권한을 갖지 않는다. 탐색과 승인 경계를 분리한다.

한 장짜리 golden 결과표. 행은 input, token/mask, forward, loss, backward, optimizer, checkpoint, memory, performance와 failures다. 열은 expected, observed, oracle grade, tolerance, source, artifact와 status다. 빈 셀은 `NOT_CAPTURED` 또는 `NOT_RUN`이다.

옵션 표는 effective value, selected branch, state changed, metric와 invalidated fixtures를 보여준다. environment 표는 source/dependency/CUDA/GPU fingerprint를 가진다. 세 표의 RunID가 같아야 한다.

summary는 pass 개수보다 first failure와 support 범위를 앞에 둔다. 성능 개선은 numerical gates 뒤에 쓴다. retry와 flaky disposition을 숨기지 않는다.

independent reviewer는 표의 임의 셀에서 raw artifact와 source로 이동한다. 링크가 끊기면 수치를 추정하지 않는다. 이 표가 29·30장의 인수 입력이다.

독자가 직접 고칠 첫 세 버그. assistant loss mask one-off는 token 표와 valid count에서 찾아 collator boundary를 수정한다. forward가 실행됐다는 사실만으로 문제를 넘기지 말고, 수정 뒤 원 fixture와 인접 fixture를 재실행한다.

optimizer group에서 adapter module 하나가 누락된 문제는 trainable inventory와, gradient는 있지만 weight delta가 없는 trace로 찾는다. 누락된 parameter identity를 group에 추가한 뒤 state initialization과 checkpoint를 검증한다.

checkpoint가 RNG와 sampler cursor를 저장하지 않으면 restored loss가 같아도 다음 batch, dropout과 update가 달라진다. 이 차이를 확인한 뒤 state schema를 확장하고 kill/resume로 수정 결과를 증명한다.

세 버그는 data, optimization와 durability라는 서로 다른 경계를 가르친다. 최종 loss 하나만 보면 모두 늦게 발견된다. first-divergence 방식이 실용적인 이유다.

golden에서 production으로 확대할 때 다시 물을 것. model·sequence·batch가 커지면 kernel, memory, compile와 numerical regime가 바뀐다. single GPU fixture가 같은 code path를 계속 타는지 확인한다. 새 shape class는 child evidence다.

world size가 늘면 data denominator, RNG, parameter sharding, collectives와 checkpoint가 추가된다. single GPU PASS를 복사하지 않는다. global logical oracle와 rank-local trace를 비교한다.

data corpus가 커지면 duplicate, rights, tail lengths와 worker failures가 등장한다. immutable golden subset은 그대로 유지하면서 full pipeline sampling audit를 추가한다. small fixture만으로 data quality를 주장하지 않는다.

운영 tracker, remote store와 scheduler가 붙으면 외부 failure와 authority가 생긴다. core RunSpec과 evidence identity를 보존하고 non-authoritative observers를 분리한다.

분산 기준선으로 승격할 조건. 완료선은 loss가 내려가고 checkpoint 파일이 생겼다는 것이 아니다. raw row 하나를 다음 update까지 계산하고, option·failure 하나가 바꾼 최초 state를 찾으며, resume 뒤 동일 의미를 재생하는 능력이다.

코드 근거는 실제 loaded functions와 guards, 실행 근거는 artifacts와 events, 수학 근거는 독립 oracle에 있다. 어느 하나가 나머지를 대신하지 않는다. 미실행 hardware·kernel·objective는 명시한다.

이 package가 작고 빠르며 독립 재생 가능하면 이후 규모의 복잡성을 양분할 수 있다. cluster divergence가 나타났을 때 data·loss의 원래 오류인지 collective·sharding의 새 오류인지 판단한다.

독자는 golden run을 한 번 만드는 데서 끝내지 않는다. source, CUDA, data, tokenizer와 recipe가 바뀔 때 child evidence를 만들고 baseline 승격을 검토한다. 이 반복이 학습 시스템의 가장 작은 신뢰 단위다.

15분 cold-review 점검. **하나의 닫힌 수치 fixture로 전 경계를 잇는다.** 여러 개의 성공 로그를 나열하는 것보다, 사람이 처음부터 끝까지 다시 계산할 수 있는 한 실행을 갖는 편이 강하다. 예를 들어 vocabulary 7, hidden size 4, sequence length 4인 작은 causal LM과 두 row를 만든다. 첫 row의 labels는 `[-100, 2, 3, 4]`, 둘째 row는 `[-100, -100, 5, 6]`으로 두어 유효 token 수가 각각 3과 2가 되게 한다.

padding·prompt mask·마지막 EOS가 서로 다른 위치에 있으므로 label shift와 denominator 오류를 동시에 드러낸다. embedding과 projection에는 정수에서 만든 고정 FP64 값을 넣고 dropout은 끈다. 이 fixture는 성능을 흉내 내기 위한 축소 모델이 아니라, `bytes→IDs→logits→N,S→gradient→delta→durable state`의 모든 화살표를 독립 계산 가능하게 만드는 수학적 자다.

loss ledger에는 scalar `loss` 하나 대신 token별 negative log likelihood `\ell_{b,t}`, numerator `S=Σm_{b,t}\ell_{b,t}`, denominator `N=Σm_{b,t}`를 둔다. 위 예의 `N`은 5여야 한다. 두 microbatch의 `N_1=3`, `N_2=2`가 다르기 때문에 `(S_1/N_1+S_2/N_2)/2`를 잘못 사용하면 즉시 reference와 갈라진다. backward oracle은 `∇(S_1+S_2)/(N_1+N_2)`이며, 각 microbatch 평균 gradient를 단순 평균한 값이 아니다. 장에서 앞서 사용한 `N`과 `S` 표기가 구현에 따라 뒤집히지 않도록 산출물 schema에는 `loss_sum`과 `valid_token_count`라는 의미 이름을 사용한다.

optimizer fixture는 projection의 2×2 slice만 trainable로 두고 AdamW 두 step을 계산한다. 첫 step 직전에는 `parameter_before`, unscaled gradient, clipping coefficient, first·second moment와 group step이 있고, 직후에는 adaptive contribution, decoupled decay contribution과 `parameter_after`가 있다.

두 번째 step 직전에 checkpoint를 저장한 뒤 fresh process에서 복원한다. 연속 실행과 복원 실행의 다음 `BatchID`, RNG draw, loss sum/count, gradient, moment, LR과 delta가 차례로 같아야 한다. parameter만 같고 moment가 다르면 load 직후에는 숨지만 두 번째 delta에서 드러난다. 그래서 한 step save/load는 optimizer 복원을 충분히 시험하지 못한다.

fixture의 정상 경로는 다음 상태 전이로 고정한다.

| 경계 | 입력 상태 | 커밋 뒤 권위 상태 | exact하게 같아야 하는 것 | 수치 오차를 허용하는 것 |
|---|---|---|---|---|
| admission | environment·source·config·fixture digest | `RunID` | 모든 digest와 resolved option | 없음 |
| collate | row bytes·tokenizer·template | `BatchID` | IDs·labels·mask·position·`valid_token_count` | 없음 |
| forward | `BatchID`·weight digest·RNG state | `ForwardID` | shape·dtype·finite·branch | logits·activation 값 |
| objective | logits·labels·loss mask | `LossID` | 기여 좌표와 denominator | token loss·loss sum |
| backward | `LossID`·scaler state | `GradientID` | owner·presence·commit 가능 여부 | gradient 값·norm |
| update | gradient·optimizer·scheduler state | `UpdateID` | step clock·group membership·skip 여부 | moment·parameter delta |
| checkpoint | 모든 durable state·parent | `CheckpointID` | 파일 hash·completeness·lineage | 없음 |
| replay/eval | checkpoint·eval fixture | `ReplayID`·`EvalID` | row·normalization·denominator·artifact hash | logits·metric 값 |

이 표의 핵심은 observer가 상태를 만들지 않는다는 점이다. 로그, profiler와 W&B callback은 권위 상태를 읽어도 공용 RNG를 소비하거나 optimizer clock을 진행시켜서는 안 된다. observer를 제거한 control과 결과가 달라지면 계측 자체가 hidden input이다. 그 경우 callback source와 호출 순서를 manifest에 넣고, 학습 RNG와 평가·시각화 RNG를 분리한 뒤 새 baseline을 만든다.

작은 fixture가 증명하는 범위를 과장하지 않는다. 이 실험으로는 tokenizer/template bytes와 label mask, loss reduction, autograd 연결, parameter-group ownership, AMP skip의 원자성, optimizer 수식, checkpoint state coverage와 재개 시 다음 update의 의미를 강하게 시험할 수 있다. synthetic OOM과 partial-write를 넣으면 예외 경계와 rollback 선택도 검증할 수 있다. 이 모두는 거대 모델을 학습하지 않고도 실패시킬 수 있는 계약이다.

반면 이 fixture는 HBM 용량 한계, 긴 sequence에서 선택되는 attention kernel, 실제 CUDA reduction의 장기 오차, 대규모 데이터의 품질, 모델 수렴성, throughput, 멀티 GPU collective와 노드 장애를 증명하지 않는다. 작은 shape가 fused path의 alignment 조건을 만족하지 않아 eager fallback을 탈 수도 있다. 따라서 결과표에는 `small-synthetic-confirmed`와 `production-shape-not-run`을 별도 열로 둔다. 작은 실행을 통과한 뒤 실제 shape를 실행하지 않았다면 “CUDA 경로 검증 완료”가 아니라 “수식·상태 계약 검증 완료”라고 써야 한다.

최초 불일치는 원인보다 앞선 증상일 수 있다. NaN이 loss에서 처음 관측됐어도 최초 원인은 tokenizer가 만든 빈 loss mask, 입력의 비정상 값, 이전 update에서 오염된 weight일 수 있다. 그러므로 두 실행을 비교할 때는 마지막 정상 durable state에서 시작해 producer 순서로 걷는다. 다음 표는 눈에 띄는 오류가 아니라 가장 먼저 비교할 권위 상태를 정한다.

| 증상 | 가장 먼저 비교할 상태 | 다음 분리 실험 | 통과 판정 |
|---|---|---|---|
| 첫 batch부터 loss가 다름 | RowID→rendered bytes→IDs→labels/mask→`N` | collator를 CPU reference로 재생 | 최초 다른 경계가 하나로 좁혀짐 |
| loss가 NaN | 입력·weight의 마지막 finite boundary, token loss 전 logits | FP32 eager, component loss 분리, 문제 row 이분 탐색 | 최초 nonfinite producer와 소비 함수가 특정됨 |
| gradient만 NaN | loss scale, backward hook 순서, custom autograd | scaler 없이 FP32, 선택 parameter finite difference | forward 정상과 최초 비정상 gradient edge가 함께 보임 |
| trainable delta가 0 | `step_committed`, LR, grad presence, group identity | overflow 없는 scalar optimizer fixture | freeze·zero grad·skip·group 누락 중 하나로 구분됨 |
| OOM | phase별 allocated/reserved와 tensor lifetime | batch/sequence 이분 탐색, hook·cache·checkpointing A/B | 용량 부족과 step별 retention 증가가 구분됨 |
| 같은 seed인데 batch가 다름 | sampler/worker RNG와 cursor, prefetch queue | `num_workers=0`, transform RNG 고립 | 첫 RNG consumer 또는 cursor drift가 특정됨 |
| resume 직후만 다름 | parent·state coverage·첫 BatchID·LR·RNG draw | K 직전/직후 kill-point replay | K+1의 최초 다른 state가 특정됨 |
| 새 환경에서만 다름 | canonical config와 loaded `.so` digest, kernel dispatch | cache를 비운 eager reference | config·artifact·dispatch drift가 구분됨 |

cache는 환경과 데이터 양쪽에 숨어 있다. tokenizer·dataset cache는 오래된 processed artifact를 재사용할 수 있고, compile·kernel autotune cache는 다른 code path와 성능을 만들 수 있다. clean-room replay에서는 cache root를 새 임시 경로로 두고, reuse run에서는 cache key와 producer digest를 검사한다. cache를 지우니 문제가 사라졌다는 결론으로 닫지 않는다. 어떤 stale key가 어떤 consumer에 들어갔는지 찾아 invalidation rule과 regression fixture를 남긴다.

config drift도 문자열 diff로 끝내지 않는다. requested config, parser 뒤 canonical config, component가 실제 소비한 effective config를 세 층으로 비교한다. 예컨대 CLI에는 `gradient_checkpointing=true`인데 모델의 해당 module branch가 켜지지 않았거나, optimizer 이름은 같지만 fused fallback이 선택될 수 있다. expected branch counter가 0이면 설정 파일이 같아도 실행 의미는 다르다. 반대로 logging interval처럼 수학 상태를 바꾸지 않아야 하는 옵션이 tensor trace를 바꾸면 observer coupling을 의심한다.

인수는 재생과 반증으로 끝낸다. 검토자는 제공자가 고른 성공 로그를 읽는 대신 깨끗한 process에서 fixture를 재생하고, loss mask 한 칸 이동·adapter detach·optimizer group 누락·checkpoint marker 제거라는 네 negative control 중 적어도 둘을 고른다. 각 변형이 예상 경계에서 실패하고, fault flag를 끈 다음 원래 digest로 돌아와야 한다. assertion이 결함을 놓치거나 더 앞선 엉뚱한 경계에서 실패하면 baseline 자체를 승인하지 않는다.

최종 인수 묶음에는 immutable environment/data/tokenizer/model/config manifest, golden batch와 독립 oracle, forward·backward·update trace, checkpoint generation과 resume replay, 평가 원장, fault bundle, source 좌표와 판정 코드가 들어간다. 모든 파일은 하나의 `RunID` 및 parent lineage로 연결한다. `PASS`는 실행 artifact가 있을 때만 허용하고, source를 읽었지만 실행하지 않은 경우는 `SOURCE_CONFIRMED`, small fixture만 통과한 경우는 `SMALL_FIXTURE_PASS`, 실제 shape가 남은 경우는 `PRODUCTION_SHAPE_NOT_RUN`으로 적는다. 이 어휘를 지키면 작은 실험의 강한 결론과 아직 하지 않은 큰 실험을 동시에 정직하게 보존할 수 있다.

시간이 부족하면 먼저 manifest의 source·data·tokenizer·environment digest를 확인한다. 이어 첫 batch의 IDs, labels, mask와 valid count를 본다. first logits와 loss numerator/count가 oracle과 맞는지 확인한다. 이 세 단계가 다르면 optimizer나 CUDA 성능을 조사하지 않는다.

다음으로 selected gradients의 owner·finite status, clipping과 optimizer delta를 본다. global update counter와 scheduler LR이 successful step을 가리키는지 확인한다. checkpoint에서 같은 states와 RNG·cursor를 복원해 next update를 비교한다.

마지막에는 negative fixture 하나와 kill point 하나를 실행한다. expected boundary에서 실패하고 partial artifact가 latest로 보이지 않아야 한다. 성능 표에는 denominator, steady window와 profiler state가 있어야 한다. 빠졌으면 `NOT_CAPTURED`다.

이 15분 점검은 전체 suite를 대신하지 않는다. 다만 가장 중요한 의미 경계를 빠르게 훑어 잘못된 run을 조기에 멈춘다. 이상이 보이면 28.15.5항의 first-divergence 순서로 깊게 들어간다.

독립 검토자의 서명. 검토자는 author가 고른 성공 표본만 보지 않는다. 임의 SampleID, trainable parameter, checkpoint와 failure case를 선택한다. 정방향과 역방향으로 lineage를 걷고 machine report의 수치를 손계산·source와 대조한다.

baseline 승격이 포함됐다면 old failure, 변경 이유, 독립 oracle과 negative regression을 확인한다. performance improvement가 expected 변경의 근거가 되지 않았는지 본다. 실행하지 않은 조합과 known flakiness를 report가 숨기지 않아야 한다.

서명은 RunSpec과 artifact root digest, oracle grades, support 범위와 review identity를 가진다. 이후 파일이 바뀌면 서명이 무효가 된다. 최신 디렉터리 이름에 서명하지 않는다.

승인된 package는 29장의 분산 child와 30장의 recipe가 참조한다. downstream 결과가 이 baseline을 덮지 않는다. source나 fixture가 바뀌면 dependency graph가 관련 인수를 다시 열어야 한다.

최종 서명 전에 reviewer는 baseline 생성 권한과 실행 권한이 분리됐는지도 확인한다. candidate code가 자기 결과를 expected로 자동 승격할 수 있다면 검증이 순환한다. expected artifact 변경에는 이유, old/new diff, 독립 oracle과 별도 승인이 필요하다. CI retry가 첫 실패를 지우거나 tracker의 latest alias가 canonical artifact를 바꾸어서는 안 된다.

또 하나의 RunSpec을 다른 clean environment에서 재생한다. package cache를 비우고 immutable sources만 사용해 dependency resolution, processed fixture, selected tensors와 checkpoint next update를 비교한다. bitwise 재현을 약속하지 않은 CUDA 경로는 선언한 numerical·behavioral 범위로 판정한다. 설명되지 않은 차이는 환경 탓으로 닫지 않고 실제 loaded library, dispatch와 RNG consumer를 찾는다.

이 마지막 분리와 clean-room 재생이 통과하면 golden package는 개인 작업 디렉터리의 우연한 성공을 넘어선다. 다른 사람이 같은 증거로 같은 제한된 결론을 얻을 수 있고, 이후 분산 규모에서 새로 생긴 차이를 정확히 격리할 수 있다.

그때 독자는 단일 GPU 결과를 성능 자랑이 아니라 학습 의미의 기준 좌표로 사용한다. data, 수식, code, CUDA state와 durable artifact가 한 RunID에서 만나는 이 좌표가 다음 모든 확장과 회귀 분석의 출발점이다.

그 좌표는 다른 검토자가 독립적으로 다시 계산할 수 있어야 한다. 정상 fixture가 맞는지만 확인하지 않고 mask 반전, overflow 또는 partial checkpoint 가운데 하나를 주입했을 때 예상 경계에서 실패하는지도 확인한다. 재계산과 반증이 모두 통과해야 이후 멀티노드 run에서 생긴 차이를 단일 GPU 기준선의 결함과 구분할 수 있다.
