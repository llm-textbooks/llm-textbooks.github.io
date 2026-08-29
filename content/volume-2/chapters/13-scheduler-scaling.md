# 13장. scheduler·batch·scaling law

학습률 곡선은 시간 함수처럼 보인다. 그러나 분산 학습에서 “시간”은 저절로 주어지지 않는다. 데이터 로더는 표본을 소비하고, collator는 그중 loss를 갖는 token을 정하며, 여러 microbatch가 gradient 하나로 합쳐진 뒤에야 optimizer가 parameter를 바꾼다. overflow가 검출되면 앞의 일은 모두 일어났어도 update는 commit되지 않는다. 이때 `step`이라는 이름 하나로 모든 사건을 세면, 그래프는 매끈해도 재개한 학습은 이미 다른 실험이다.

이 장의 핵심은 schedule 종류를 외우는 데 있지 않다. 다음 다섯 질문에 수치로 답하는 데 있다.

1. 이번 update의 gradient는 어떤 표본과 몇 개의 **유효 label token**에서 왔는가?
2. loss의 numerator와 denominator는 rank와 accumulation window를 가로질러 어떻게 합쳐졌는가?
3. optimizer가 실제로 parameter와 moment를 바꿨는가, 아니면 attempt만 있었는가?
4. scheduler는 어떤 counter를 읽어 어느 parameter group의 다음 learning rate를 만들었는가?
5. checkpoint는 이 사건 사슬의 어느 원자적 경계에서 저장되었는가?

이를 한 줄로 쓰면 다음과 같다.

`DrawID → MicrobatchID → (loss_sum, valid_labels) → AccumulationWindowID → global finite 합의 → OptimizerCommitID → SchedulerEventID → CheckpointID`

왼쪽 사건이 오른쪽 사건의 원인이다. 따라서 resume drift를 조사할 때도 오른쪽의 loss부터 거슬러 올라가지 않고, 두 run에서 처음 달라진 사건을 왼쪽부터 찾는다. 이 원인 사슬은 10장의 loss denominator, 12장의 optimizer state, 14장의 loss scaler, 16장의 collective, 17장의 checkpoint와 이어진다.

| 관찰 단위 | 증가시키는 사건 | overflow/빈 label batch에서의 동작 | 소유해야 할 상태 |
|---|---|---|---|
| `samples_consumed` | sampler가 draw를 확정 | 대개 증가 | global draw cursor, sampler generation |
| `input_tokens` | collator가 입력을 확정 | 증가 | padding 포함 여부, modality별 단위 |
| `loss_bearing_tokens` | label mask가 확정 | 0일 수 있음 | global numerator·denominator |
| `attempted_update` | accumulation window가 backward를 마침 | 증가 | window ID, finite 검사 결과 |
| `committed_update` | 모든 rank가 commit에 합의하고 optimizer가 state를 변경 | 증가하지 않음 | parameter·moment generation |
| `scheduler_clock` | 선언한 clock의 authoritative event | 정책에 따라 다름 | curve generation, boundary, next LR |

이 표의 마지막 행만 보고 schedule을 설계하면 안 된다. `scheduler_clock=token`이라 해도 input token인지, loss-bearing token인지, replay된 token을 다시 셀 것인지가 남는다. `scheduler_clock=update`라 해도 attempt인지 commit인지가 남는다. **clock의 이름이 아니라 증가 사건과 소유 state가 계약**이다.

## 13.1 schedule family와 scaling 가설을 세운다

학습률 곡선은 먼저 함수 모양으로 비교하되, 곧바로 그 곡선을 움직이는 표본 단위와 batch의 통계적 의미를 붙여야 한다. warmup·linear·cosine·WSD를 선택하는 일과 critical batch를 추정하는 일은 결국 같은 질문, 즉 한 update가 어느 정도의 신호와 잡음을 담는가에 답한다.

### 13.1.1 warmup·linear·cosine·WSD

**초반을 천천히 여는 이유**

초기 moment와 activation 통계가 안정되지 않은 때 큰 lr을 쓰면 update/weight ratio가 튄다. linear warmup은 `η_t=η_max t/T_w`; cosine decay는 warmup 뒤 `η_min+0.5(η_max−η_min)(1+cos πp)`를 쓴다. WSD는 warmup 뒤 안정 구간을 길게 두고 마지막에 decay한다. 이는 중간 checkpoint를 재사용해 budget별 decay branch를 만들기 좋지만, branch의 parent와 소비 token을 기록해야 한다.

**호출 순서가 만드는 off-by-one**

`optimizer.step()` 전후 어디서 `scheduler.step()`을 부르는지에 따라 첫 lr이 달라진다. overflow로 optimizer step이 skip됐는데 scheduler만 전진하면 parameter update 없는 시간이 흐른다. scheduler state에는 `last_epoch`라는 이름이 있어도 실제 단위가 epoch라고 가정하면 안 된다.

### 13.1.2 step·token·sample 시계

**nominal batch와 realized denominator**

고정 길이 dense batch에서 한 성공 update의 nominal input token 수는

\[
N_{\mathrm{input}}=B_{\mathrm{micro}}\,L\,K_{\mathrm{accum}}\,W_{\mathrm{DP}}
\]

이다. 그러나 이 곱은 memory·throughput 계획에는 쓸 수 있어도 objective의 분모를 증명하지 못한다. rank `r`, accumulation slot `k`의 loss 합을 `S_{r,k}`, 유효 label 수를 `n_{r,k}`라 하면 token-mean objective는

\[
\mathcal L=\frac{\sum_{r=1}^{W_{\mathrm{DP}}}\sum_{k=1}^{K}S_{r,k}}
{\sum_{r=1}^{W_{\mathrm{DP}}}\sum_{k=1}^{K}n_{r,k}}.
\]

각 microbatch의 mean loss를 먼저 구해 똑같이 평균하면 일반적으로 이 식과 다르다. response-only SFT, packing, 가변 길이에서는 특히 그렇다. 예컨대 `(loss_sum, valid_labels)`가 `(100,100)`과 `(20,10)`이면 microbatch mean의 평균은 `1.5`지만 global token mean은 `120/110≈1.091`이다. gradient accumulation을 늘린 뒤 loss scale이 달라졌다면 scheduler보다 먼저 이 분모를 의심해야 하는 이유다.

scaling law의 학습 token, scheduler의 진행 token, loss 평균의 유효 token은 목적이 다르다. 하나의 `tokens_seen` 필드에 덮어쓰지 않고 최소한 `input_tokens`, `loss_bearing_tokens`, `unique_source_tokens`, `replayed_tokens`를 구분한다.

**accumulation과 world size를 바꿀 때 보존되는 것**

DP world size를 두 배로 늘리고 accumulation을 절반으로 줄이면 위의 **nominal** token/update는 같게 만들 수 있다. 그렇다고 학습 상태가 같아지는 것은 아니다. rank partition과 prefetch queue, RNG 소비, collective reduction tree가 달라지고, 마지막 가변 길이 batch의 realized denominator도 달라질 수 있다. 보존됐다고 주장할 수 있는 것은 직접 대조한 불변식뿐이다.

| 보존 후보 | 비교할 증거 | 같지 않을 때 먼저 분리할 원인 |
|---|---|---|
| global input token/update | collator 이후 count의 global sum | padding·drop-last·길이 bucket |
| global valid-label denominator | loss mask 이후 count의 global sum | template·truncation·packing |
| 표본 multiset | 첫 window의 `DrawID`/`DocumentID` digest | sampler cursor·prefetch |
| applied LR | group stable ID별 optimizer scalar | scheduler offset·group reorder |
| optimizer state | parameter ID별 moment step/digest | reshard·partial load·skip |
| 수치 궤적 | 고정 batch의 gradient·delta tolerance | reduction order·dtype·kernel |

step 기반 schedule은 commit 번호가 같으면 곡선 위치를 유지한다. 하지만 같은 lr가 같은 데이터를 만났다는 뜻은 아니다. token 기반 schedule도 자동으로 안전하지 않다. 실제 ledger의 **global 증가량**을 읽지 않고 rank-local token 수를 읽으면 world size에 따라 시간이 느려진다.

### 13.1.3 critical batch와 gradient noise

**선형 scaling의 경계**

독립 sample gradient의 분산은 batch 증가로 줄지만 데이터 상관, packing, curriculum이 있으면 단순 `1/B`가 아니다. batch를 키우고 lr을 선형 증가시키는 규칙은 작은 batch 영역의 근사다. gradient noise scale, update norm, validation per token을 함께 보며 임계 영역을 찾는다.

**configured batch와 realized batch**

sequence 수가 같아도 길이와 mask가 다르면 유효 label 수가 변한다. RL에서는 sequence mean과 token mean이 서로 다른 objective를 만든다. scheduler manifest에는 `clock_kind`, 누적 count, 이번 update의 denominator, overflow skip 여부를 기록한다.

## 13.2 clock·state·resume의 원자적 경계

같은 곡선도 `step`이 attempt인지 commit인지에 따라 전혀 다른 학습률 열을 만든다. world size 변경, token clock 변환, framework factory, overflow를 하나의 사건 사슬로 읽어 checkpoint가 다음 learning rate까지 복원하는지 검증한다.

### 13.2.1 world-size 변경과 resume drift

**어떤 동일성을 요구할 것인가**

world size 변경 후 bitwise 동일은 collective 순서와 sample partition이 바뀌어 대개 비현실적이다. sample-exact, token-count-exact, numerical-equivalent를 구분한다. scheduler state만 맞고 sampler cursor가 다르면 같은 lr로 다른 데이터를 본다.

**디깅과 handoff**

resume 직후 loss가 튀면 checkpoint 전후의 `optimizer_step`, `tokens_seen`, lr, scaler, sampler cursor, global batch를 한 줄로 대조한다. 첫 재개 step의 lr을 저장 시점의 “다음 lr”과 비교한다.

**scheduler를 순수 함수로 검산한다**

scheduler는 `lr=f(clock,config)`인 순수 reference를 먼저 만든다. warmup 4 step, total 12 step, cosine min ratio 0.1이면 clock 0–12의 예상 lr을 표로 고정한다. framework scheduler가 optimizer initialization 때 한 번 호출되는지, 첫 `step()` 뒤 clock이 0인지 1인지 비교한다. off-by-one은 긴 run에서 작아 보여도 resume boundary에서 branch를 만든다.

WSD에서는 stable 구간과 decay 시작 token을 명시한다. 중간 checkpoint에서 여러 decay horizon을 branch하면 parent checkpoint의 optimizer/scheduler state를 복사하되 새 schedule config와 RunID를 만든다. 이미 decay가 시작된 scheduler state에 새 WSD를 덧씌우지 않는다. linear/cosine schedule의 `num_training_steps`를 dataset epoch 추정에서 만들었다면 filtering·packing 변화로 실제 token horizon이 달라질 수 있다.

다음 의사 코드는 특정 framework의 구현을 베낀 것이 아니라, 실제 호출자를 읽을 때 대조할 **상태 전이 기준선**이다.

```python
for slot, batch in accumulation_window:
    loss_sum, valid = forward_loss_sum(batch)
    window_loss_sum += loss_sum
    window_valid += valid
    backward(loss_sum)                 # 아직 optimizer commit이 아니다

attempted_update += 1
global_valid = all_reduce_sum(window_valid)
all_finite = all_reduce_and(grads_are_finite())

if global_valid == 0 or not all_finite:
    record_skip(reason, attempted_update, input_tokens)
    clear_partial_window()
else:
    normalize_gradients(global_valid)  # objective의 실제 분모
    applied_lr = lr_reference(clock_snapshot, schedule_config)
    optimizer_step(applied_lr)
    committed_update += 1
    advance_scheduler(authoritative_increment())
    atomic_commit(optimizer_state, scheduler_state, data_cursor)
```

여기서 검토할 핵심은 함수 이름이 아니라 순서다. finite 합의 전에 일부 rank가 update하면 replica가 갈라진다. `global_valid`가 0인데 나누면 NaN을 만든다. optimizer가 skip됐는데 `committed_update`나 update-clock scheduler가 전진하면 lr와 Adam bias-correction step이 어긋난다. `atomic_commit`이 optimizer와 scheduler 사이에서 잘리면 저장된 current lr가 같아도 **다음 lr**은 달라질 수 있다.

실제 DDP 구현에서는 gradient collective가 합인지 평균인지, loss를 backward 전에 이미 나눴는지에 따라 `normalize_gradients`의 위치와 계수가 달라진다. 위 코드를 그대로 구현하라는 뜻이 아니다. 최종 gradient가 위의 global token-mean objective의 미분과 일치하는지를 작은 tensor의 단일-process reference로 검산하라는 뜻이다. world size와 accumulation을 바꿔도 같은 numerator·denominator에서 같은 gradient가 나오는지가 가장 직접적인 판정이다.

재개 직후 loss가 튈 때는 한꺼번에 “checkpoint가 깨졌다”고 결론 내리지 않는다.

| 최초 불일치 | 고정할 것 | 먼저 조사할 state·caller | 반증 실험 |
|---|---|---|---|
| 다음 `DrawID`가 다름 | model·optimizer·lr | sampler global cursor, prefetch, epoch seed | 다음 한 window의 ID multiset만 dry-run |
| valid denominator만 다름 | DrawID·token IDs | chat template, truncation, packing mask | logits를 고정하고 label mask/count 비교 |
| applied LR만 다름 | batch·gradient·optimizer state | scheduler call order, `last_epoch`, horizon closure | constant gradient scalar의 첫 세 delta 비교 |
| LR는 같고 moment step이 다름 | batch·gradient | optimizer state load, group mapping, skip history | group stable ID별 step/moment digest 비교 |
| 한 rank만 state가 다름 | logical event ledger | finite collective, reshard, partial commit | rank별 checksum과 found-inf 주입 |
| 처음에는 같고 경계에서 갈림 | parent state | warmup/decay endpoint 포함 규칙 | 경계 `−2…+2`의 FP64 oracle 비교 |

이 표의 순서가 중요한 이유는 loss가 마지막 관찰값이기 때문이다. data가 먼저 달랐는데 lr를 조절하면 원인을 가리고, scheduler offset이 다른데 warmup을 늘리면 실패를 늦출 뿐이다.

### 13.2.2 세 개의 clock ledger

`attempted_update`는 backward까지 온 횟수, `committed_update`는 optimizer가 실제 parameter를 바꾼 횟수, `tokens_consumed`는 dataloader가 넘긴 input token 수다. AMP overflow, NaN skip, empty valid-label batch에서 셋은 갈라진다. sample count와 valid-label count도 따로 둔다. scheduler가 어느 counter를 읽는지 config가 아니라 매 step event로 기록한다.

token-based schedule은 `tokens_consumed`와 `loss_bearing_tokens` 가운데 무엇을 쓸지 정한다. padding·prompt mask가 많은 SFT에서 둘은 크게 다르다. scaling-law compute budget에는 input token이 적합할 수 있고 optimization clock에는 loss-bearing token이 더 직접적일 수 있다. 하나를 보편 clock으로 강요하지 않고 이름을 붙인다.

**batch scaling 실험**

global batch를 2배로 만들 때 세 실험을 분리한다. A는 lr 고정, B는 lr 선형 증가, C는 warmup과 decay horizon도 token 기준으로 재조정한다. 같은 optimizer update 수가 아니라 같은 소비 token checkpoint에서 validation을 비교한다. gradient noise는 microbatch gradient 여러 개의 mean과 variance로 추정하되 packed sample 상관을 기록한다.

world size를 늘리고 accumulation을 줄여 global batch를 유지하는 실험은 optimizer noise를 대략 유지하지만 collective와 sample partition을 바꾼다. 첫 global batch의 DocumentID multiset과 loss denominator를 reference와 비교한다. data order가 다르면 scheduler parity와 sample parity를 구분한다.

**장애와 결정 트리**

overflow를 의도적으로 만들어 optimizer가 skip될 때 lr이 전진하는지 본다. checkpoint를 scheduler step 직전과 직후에 저장해 resume 첫 lr을 검증한다. total steps config를 바꾸고 옛 state를 load했을 때 framework가 새 horizon을 쓰는지 저장된 lambda를 쓰는지 확인한다. world-size 변경 뒤 token counter가 rank별 local 값으로 축소되지 않는지 검사한다.

resume loss spike에서는 첫째 다음 GoldenBatchID, 둘째 committed step, 셋째 token clock, 넷째 lr, 다섯째 optimizer bias-correction step을 본다. lr만 다르면 scheduler ordering, lr은 같은데 update가 다르면 optimizer state, batch가 다르면 sampler를 조사한다. plateau에서 warmup이 끝나지 않았다면 total horizon 단위와 counter를 확인한다.

### 13.2.3 clock 사이 변환은 언제 가능한가

sequence length와 global batch가 고정되고 모든 token이 loss를 가진다면 `tokens_per_update=B×T`로 step과 token clock을 변환할 수 있다. packing, variable length, response-only mask, dropped MoE token이 있으면 이 상수는 깨진다. 변환은 step별 realized count ledger를 적분해야 한다. 평균 token/update로 나누는 근사는 resume exactness에 쓰지 않는다.

sample clock도 DocumentID와 packed sample이 일대일이 아니다. 한 document가 여러 sequence로 잘리거나 여러 document가 한 pack에 들어간다. `samples_seen`, `documents_touched`, `input_tokens`, `valid_labels`를 분리한다. curriculum weight 변경 시점은 어느 clock에 매달렸는지 기록한다.

**warmup의 scale 진단**

warmup은 초반 update/weight ratio와 optimizer state 초기화를 안정시키려는 수단이다. 기간을 관행적 1%로 정하기 전에 group별 update RMS, gradient noise, loss scaler skip을 본다. Adam bias correction이 이미 초반 moment 축소를 보정해도 큰 lr에서 activation/gradient가 불안정할 수 있다.

linear warmup과 sqrt warmup은 같은 끝점이라도 초반 면적이 다르다. Muon/AdamW hybrid에서 group별 base lr가 다르면 공통 multiplier가 실제 update ratio를 같은 정도로 맞추지 않는다. group별 update/weight panel로 warmup 종료를 확인한다.

**cosine·linear·WSD 면적**

schedule 비교에서는 peak lr만 아니라 학습 전 기간의 lr 적분과 decay tail을 본다. cosine은 중간부터 계속 줄고 WSD는 stable 구간을 유지한다. 같은 total token과 peak라도 parameter가 경험한 누적 step 크기가 다르다. decay 실험 branch는 parent checkpoint 이후 lr sequence를 표로 제시한다.

minimum lr ratio가 0인지 양수인지 마지막 구간의 계속 학습을 바꾼다. weight decay의 실제 shrink `1−lr_t·wd`도 schedule을 따라 변하므로 누적 decay product를 계산한다. decoupled decay가 “wd 고정”이어도 시간 효과는 고정이 아니다.

### 13.2.4 Transformers류 factory와 state

scheduler factory는 name과 warmup/total steps에서 LambdaLR류 객체를 만들 수 있다. trainer가 optimizer 생성 전후 언제 total steps를 계산하는지, epoch 기반 dataset length가 iterable dataset에서 알려지는지 확인한다. ratio로 입력한 warmup이 absolute step으로 언제 변환되는지도 manifest에 남긴다.

state dict에는 `last_epoch`, step count, base lr 등이 들어간다. config의 total steps 자체가 closure에만 들어가 직렬화되지 않는 구현이면 load 시 같은 factory config가 필요하다. state dict만 옮겨도 schedule이 재현된다고 가정하지 않는다. scheduler class·constructor config·state를 함께 저장한다.

**elastic world-size 시나리오**

world size 8, microbatch 2, accumulation 4, length 2048이면 nominal input token/update는 `8×2×4×2048`이다. world size 16으로 바꾸고 accumulation 2로 줄이면 nominal count는 같다. 그러나 rank partition, collective reduction order, prefetch queue가 바뀐다. first update의 GoldenBatchID multiset과 valid denominator를 비교한다.

global batch를 유지하지 않고 world size만 늘리면 step clock schedule은 같은 update 번호에서 더 많은 token을 본다. token horizon이 절반 step에 끝날 수 있다. 재개 정책은 schedule을 token 기준으로 재계산할지 원 step curve를 유지할지 선택한다. 둘은 다른 branch이며 artifact ID가 달라야 한다.

**resume oracle**

checkpoint `K`는 `attempted`, `committed`, input/valid token, sample/document counter, current lr, next lr, optimizer step, scaler growth tracker를 저장한다. oracle은 uninterrupted run의 event `K−2…K+3`과 resume event를 나란히 비교한다. 저장 시 lr은 같지만 next lr이 다르면 호출 ordering 또는 config drift다.

partial accumulation checkpoint라면 accumulated gradient와 microbatch count가 없을 때 window 전체를 재실행한다. 이미 dataloader가 소비한 sample을 rollback할지 ledger에 기록한다. optimizer commit 뒤 scheduler 전진 전 crash처럼 중간 state를 허용하지 않도록 transaction boundary를 정한다.

### 13.2.5 overflow·empty batch 실패 실험

FP16 loss에 큰 scale을 주어 overflow를 만들고 parameter, optimizer moment, committed clock, scheduler lr 중 무엇이 전진하는지 본다. 정책은 일관돼야 한다. empty valid-label batch에서는 denominator 0을 조용히 NaN으로 만들지 않고 skip event를 남긴다. data token은 소비했지만 update는 없다.

gradient clipping으로 update가 작아진 step도 committed update다. 반면 NaN guard가 optimizer를 skip하면 committed가 아니다. skip reason enum과 count를 둔다. 장기 run에서 skip이 특정 data domain이나 sequence length에 몰리는지 확인한다.

**critical batch 실험 설계**

batch 후보마다 independent seed 여러 개의 microbatch gradient를 저장해 mean gradient norm과 sample variance를 추정한다. 단일 batch loss variance를 gradient noise로 대신하지 않는다. packing/document correlation을 줄이기 위해 sampling unit과 cluster를 기록한다.

각 batch에서 lr sweep을 수행하고 같은 token budget의 validation, wall time, update count를 비교한다. batch가 커져 optimizer step 수가 줄면 scheduler decay도 token clock으로 맞춘다. throughput 상승과 sample efficiency 하락의 교차점을 찾아 hardware budget에 맞는 batch를 고른다.

**디버깅 query**

로그에서 `(RunID,committed_step,tokens_seen,lr,GoldenBatchID)`를 key로 join할 수 있어야 한다. 두 run의 loss가 갈라진 최초 row에서 batch hash가 다르면 data, lr만 다르면 schedule, 둘 다 같고 delta가 다르면 optimizer/numeric을 본다. timestamp join만으로 rank skew가 있는 분산 event를 맞추지 않는다.

plateau에서는 lr이 0 또는 min에 일찍 도달했는지, total-step 계산이 accumulation을 microstep으로 셌는지 본다. oscillation에서는 warmup 종료와 batch/world-size change를 찾는다. resume spike에서는 next lr와 bias correction step을 동시에 본다.

## 13.3 branch 실험을 handoff manifest로 닫는다

schedule branch는 config 한 줄을 바꾼 복사본이 아니다. parent checkpoint, clock generation, horizon, 데이터 cursor와 예상 learning-rate 열을 manifest로 고정한 뒤, 작은 branch 실험이 원래 가설만 바꾸었는지 판정한다.

### 13.3.1 branch 계보를 고정하는 handoff manifest

manifest에는 clock kind 하나만 쓰지 않고 모든 counter와 authoritative field를 표시한다. schedule segment별 start/end count, formula, base/min lr, warmup, branch parent를 넣는다. 한 update event는 consumed token/DocumentID, overflow/skip, applied lr, resulting CheckpointID와 연결된다.

15장 분산화 이후에도 이 logical clock은 rank-local counter가 아니다. coordinator 또는 deterministic global reduction이 소유한다. 17장 checkpoint는 이 manifest와 다음 event를 함께 저장해야 resume drift를 판정할 수 있다.

**event-sourced scheduler**

counter 값을 checkpoint에 한 번 쓰는 대신 update event를 append한다. event에는 attempted/committed step, input/valid token 증가량, sample ID 범위, applied lr, skip reason, resulting optimizer commit을 기록한다. checkpoint는 event offset과 counter snapshot을 함께 저장한다. 복구기는 snapshot 이후 event를 재생해 counter를 검증한다.

중복 event는 commit ID로 제거하고 gap이 있으면 재개를 막는다. dataloader가 token을 소비했지만 optimizer가 skip된 event도 남긴다. 이 ledger는 schedule 계산뿐 아니라 “lr가 적용된 데이터”를 역추적하게 한다.

**curriculum과 clock 충돌**

domain mixture가 token 10B에서 바뀌는데 scheduler는 optimizer step 기반이면 variable batch에서 두 사건의 상대 위치가 run마다 달라질 수 있다. curriculum, sequence-length warmup, optimizer lr schedule의 authoritative clock을 각각 선언하고 실제 event에서 교차점을 기록한다.

sequence length를 2배로 늘리며 global sequence batch를 유지하면 token/update가 2배다. 메모리 때문에 accumulation을 바꾸면 또 달라진다. length curriculum 전후의 lr, valid token, gradient norm을 비교한다. loss spike를 길이 자체와 lr clock jump로 분리한다.

### 13.3.2 scheduler branch 실험

동일 parent checkpoint에서 cosine decay, linear decay, WSD 세 branch를 만든다. optimizer state와 다음 batch는 같고 미래 lr sequence만 다르다. 첫 step gradient가 같고 delta가 lr 비율대로 달라지는지 검산한다. branch마다 parent, formula digest, terminal token을 기록한다.

중간 결과가 나쁜 branch를 조기 중단해도 탐색 ledger에 남긴다. best schedule만 남기면 selection bias를 감춘다. validation interval이 step 기반이면 branch별 token 위치가 같도록 조정한다.

**elastic failure rehearsal**

world size 8에서 checkpoint를 저장하고 4로 재개한다. global token/update를 유지하는 config와 유지하지 않는 config를 각각 실행한다. first batch multiset, scheduler next lr, optimizer step, sample cursor를 비교한다. sample-exact가 불가능하면 token-clock continuity만 통과했다고 표시한다.

rank 하나가 overflow를 보고 다른 rank는 finite인 상황에서 found-inf를 global하게 합쳐 모든 rank가 같은 commit/skip 결정을 하는지 본다. 일부 rank만 optimizer와 scheduler를 전진하면 즉시 divergence와 collective mismatch가 생긴다.

**운영 경보**

`lr_expected != lr_applied`, committed step gap, token/update 급변, skip burst, schedule segment 경계 지연을 경보로 둔다. data mixture/length change와 같은 timestamp에 lr jump가 있으면 incident context에 연결한다. 단순 loss threshold보다 원인을 빨리 좁힌다.

**열두 event 수치 timeline**

warmup 4 committed update를 가진 run을 만든다. attempt 1·2는 정상, attempt 3은 overflow, attempt 4·5가 정상이라면 네 번째 committed update는 attempt 5다. scheduler가 committed clock을 읽으면 peak lr도 attempt 5에 도달한다. attempt clock을 읽으면 attempt 4에서 도달한다. 두 lr sequence를 표로 비교한다.

각 attempt가 input token 1,024를 소비하지만 valid label은 `[800,700,900,0,850]`이라고 하자. overflow attempt도 data는 소비했고 empty-label attempt는 optimizer를 건너뛴다. input-token clock, valid-token clock, committed clock은 모두 다른 값이다. 어떤 clock도 숨겨진 “진짜 step”으로 합치지 않는다.

**accumulation denominator 예제**

microbatch A의 loss sum 100, valid 100과 B의 sum 20, valid 10이 있다. microbatch mean 평균은 `(1+2)/2=1.5`, global token mean은 `120/110≈1.091`이다. accumulation code가 각 mean을 같은 비중으로 backward하면 긴/짧은 batch weighting이 objective를 바꾼다. scheduler token count만 고쳐서는 gradient scale이 복원되지 않는다.

DDP rank별 차이까지 더해 local numerator/denominator를 event에 남긴다. global denominator를 만든 뒤 accumulation scale을 적용하는 reference와 한-step gradient를 비교한다.

**epoch라는 이름의 함정**

PyTorch scheduler의 `last_epoch` 같은 필드가 실제 dataset epoch를 뜻한다고 가정하지 않는다. batch 단위 호출이면 update index에 가깝다. iterable dataset에는 안정적인 epoch length가 없을 수 있다. trainer가 epoch progress를 float로 저장하더라도 sampler cursor와 같지 않다.

dataset size 변경 후 checkpoint를 load하면 epoch-derived total steps와 old scheduler horizon이 충돌한다. old horizon 유지 branch와 새 horizon 재계산 branch를 나누고 next lr을 명시한다. silent recompute를 허용하지 않는다.

**scheduler와 weight decay 결합 수치**

gradient가 0인 scalar `θ=1`, decay `0.1`에서 lr sequence가 `[0.01,0.01,0.001]`이면 최종 값은 세 shrink factor의 곱이다. constant wd라도 schedule에 따라 누적 regularization이 달라진다. schedule branch 비교에서 lr만 보고 decay budget을 같다고 말하지 않는다.

LAMB/LARS trust ratio, schedule-free relative step, Adafactor internal lr처럼 optimizer가 scale을 다시 바꾸면 `lr_applied`뿐 아니라 최종 update RMS를 기록한다. scheduler output이 optimizer effective step과 같지 않다.

**checkpoint 전후 호출 순서 test**

`optimizer.step→scheduler.step→save`와 `optimizer.step→save→scheduler.step`은 같은 파일 이름이어도 next lr가 다르다. save hook이 어느 위치에 있는지 고정 source와 trace로 확인한다. scheduler step 직전 process kill, 직후 kill을 각각 주입한다.

commit record에 optimizer commit과 scheduler event를 함께 묶으면 반쪽 전진을 찾을 수 있다. 복구기는 둘의 parent step이 맞지 않으면 이전 complete cut을 고른다.

**token horizon migration**

step schedule checkpoint를 token schedule로 바꿀 때 과거 event ledger에서 consumed token을 재구성한다. `old_step×nominal_tokens`로 근사한 branch는 exact migration과 구분한다. packing/length curriculum이 있었다면 nominal 근사가 크게 틀릴 수 있다.

새 token schedule의 current position, remaining horizon, next lr를 migration report에 쓴다. optimizer state reset 여부와 별개로 schedule migration 자체가 새 RunID를 만든다.

**multi-node clock owner**

global clock을 rank 0 Python 변수 하나로만 두면 failover와 split-brain 위험이 있다. optimizer commit ID에서 deterministic하게 파생하거나 checkpoint coordinator가 durable event를 소유한다. 모든 rank는 broadcast된 applied lr와 commit counter를 assertion한다.

rank 하나만 overflow를 봤을 때 found-inf all-reduce가 끝난 뒤 commit decision을 내린다. decision 전에 scheduler를 호출하지 않는다. rank별 lr checksum이 다르면 다음 collective 전에 fail-fast한다.

**모니터링과 복구 판정**

대시보드는 attempted/committed/token clock을 같은 x축에 억지로 겹치지 않고 변환 가능한 event join으로 보여준다. skip reason, token/update, lr expected/applied, update RMS를 함께 본다. clock gap이 급증하면 data empty batch, overflow, optimizer failure를 분기한다.

resume 성공 판정은 첫 세 event의 GoldenBatchID, 모든 counter 증가량, applied/next lr, optimizer step을 control과 비교한다. bitwise가 불가능한 topology 변경에서는 적어도 declared clock continuity와 sample 등급을 명시한다.

**장간 handoff**

15장은 scheduler를 rank-local loop 변수로 다시 만들지 않고 이 장의 global commit event를 받는다. 17장은 counter snapshot만이 아니라 event offset과 next lr를 저장한다. 18·20장의 mask/rollout denominator도 valid-token clock과 같은 이름을 쓰되 objective별 의미를 기록한다.

이 handoff가 있으면 world size, accumulation, response length가 바뀌어도 “step 10,000”을 모호하게 비교하지 않는다. 모든 곡선의 x축이 어떤 ledger field인지 책 전체에서 추적된다.

**전체 run 예시**

Run A는 world size 8, accumulation 4, variable sequence를 사용한다. event 500에서 4.2M input token, 3.1M valid label, committed update 497을 기록한다. 세 번의 overflow가 attempted와 committed 차이를 만든다. lr formula는 committed update warmup이지만 curriculum은 input token을 읽는다. 두 clock의 교차점을 manifest에 남긴다.

checkpoint 뒤 world size 16, accumulation 2로 재개한다. nominal token/update는 유지되지만 첫 packed batch가 달라져 sample-exact는 실패한다. token-clock continuity와 optimizer state는 통과했다. 이 run을 단순 “완전 재현”이라고 쓰지 않고 topology-portable/numerical-equivalent 조건으로 평가한다.

**schedule 오류를 재현하는 세 줄**

optimizer step이 skip됐는데 scheduler만 호출하는 fixture, scheduler를 optimizer보다 먼저 호출하는 fixture, load 뒤 total horizon을 바꾸는 fixture를 둔다. 세 경우 모두 loss를 오래 돌리기 전에 expected lr table에서 실패해야 한다. 작은 oracle이 장기 divergence보다 싸다.

**평가 시점 정렬**

두 run의 evaluation을 같은 step 번호로 맞추면 token 수가 다를 수 있다. EvalID는 committed step, input/valid token, sample ledger offset을 함께 가진다. scaling-law 그래프는 token x축, 운영 throughput 그래프는 wall-clock, optimizer 안정성은 committed update x축을 사용한다.

**recovery checklist**

복구 직후 authoritative clock, current/next lr, optimizer step, scaler, 첫 GoldenBatchID를 출력한다. 세 event 동안 증가량을 control과 비교한다. 차이가 나면 학습을 계속해 평균내지 않고 data·clock·optimizer 분기로 즉시 좁힌다.

**clock schema의 버전**

새 counter를 추가하거나 의미를 바꾸면 schema version을 올린다. 옛 checkpoint의 `tokens_seen`이 input인지 valid인지 모호하면 자동 변환하지 않는다. migration은 원 field, 가정, 계산 결과를 report에 남긴다.

이 명시성이 17장의 checkpoint loader가 schedule state를 조용히 오해하는 일을 막는다.

**clock·분모·commit의 교차 검산표**

독자는 열 개 event를 손으로 만든다. 정상 update, overflow, empty-label, gradient accumulation, checkpoint, resume을 포함한다. 각 행에는 input token 증가, valid label 증가, attempted/committed 증가, applied/next lr, optimizer step을 쓴다. scheduler 구현의 출력과 이 표가 다른 최초 행을 찾는다.

resume fixture는 checkpoint 직전 두 event와 직후 세 event를 보존한다. load가 성공했다는 로그 대신 next lr, first GoldenBatchID, optimizer bias-correction step을 비교한다. world size가 바뀌면 DocumentID multiset과 token count를 별도 판정한다.

**고정 구현 좌표를 만드는 절차**

Transformers checkout `550d7b3834670483a4df436541272c055dc364bf`에서 `src/transformers/optimization.py`의 scheduler factory와 각 lambda, `src/transformers/trainer.py`의 optimizer/scheduler 호출 및 checkpoint load symbol을 함께 고정한다. factory만 읽으면 trainer 호출 순서와 skip behavior를 알 수 없고 trainer만 읽으면 formula closure를 알 수 없다.

PyTorch optimizer/scheduler integration은 사용 중인 PyTorch commit에서 `LRScheduler.step`, state-dict test, AMP GradScaler step test를 고정한다. 줄 번호는 source note의 floating main이 아니라 실제 book build manifest의 commit/span으로 기록한다. upstream test가 overflow와 elastic world-size까지 검증하지 않으면 이 장의 proposed fixture를 별도 실행한다.

**run 종료 판정**

run 마지막 event와 checkpoint manifest의 committed counter, token counter, scheduler state가 같아야 한다. evaluation x축은 해당 EvalID가 실제 소비한 counter를 쓴다. mismatch가 있으면 curve를 출판하지 않고 event ledger를 복구한다.

이 판정은 17장의 consistent cut에 scheduler를 포함하게 한다. model·optimizer는 step 100인데 scheduler만 101인 checkpoint는 complete가 아니다.

**release 전 clock audit**

모든 checkpoint를 parent 순서로 읽어 committed step, input/valid token, current/next lr가 역행하지 않는지 확인한다. 의도적인 branch는 새 RunID와 parent event를 갖는다. 같은 RunID에서 counter가 줄거나 lr segment가 설명 없이 바뀌면 corruption으로 판정한다.

evaluation table의 각 row를 scheduler event와 join한다. “step 1000”만 있고 token과 denominator를 복원하지 못하는 metric은 다른 batch/world-size run과 직접 비교하지 않는다. 학습률 graph에도 authoritative x축 label을 표시한다.

resume rehearsal은 overflow 직전, warmup 종료, WSD decay 시작처럼 경계가 민감한 checkpoint를 고른다. 평범한 중간 step 하나만 통과해서는 off-by-one을 충분히 검출하지 못한다. 세 경계 모두에서 다음 lr와 delta를 확인한다.

이 audit 결과를 CheckpointID metadata에 넣으면 loader는 schedule state가 불완전한 artifact를 미리 거부할 수 있다.

운영 dashboard도 동일 event schema를 읽어 offline report와 online alert의 counter 해석이 갈라지지 않게 한다. schema migration 전후의 field 의미가 다르면 두 series를 자동 연결하지 않는다. evaluator와 checkpoint loader가 같은 authoritative clock 이름을 사용해야 한다.

최종 counter audit도 통과한다.

**사례 연구: 같은 cosine 이름이 다른 학습을 만드는 과정**

두 run은 warmup 1,000, cosine decay 100,000 step이라는 같은 문장을 쓴다. 그러나 A는 microbatch마다 scheduler를 전진하고 B는 committed optimizer update마다 전진한다. gradient accumulation 8이면 A의 schedule은 여덟 배 빨리 흐른다. overflow skip과 variable packing까지 들어오면 step이라는 말만으로 두 곡선을 비교할 수 없다.

사례의 authoritative clock은 `CommittedOptimizerStep`과 `ValidTrainingToken` 두 개다. microstep, attempted update, consumed input token은 보조 counter다. schedule function이 어느 counter를 읽는지 manifest에 넣는다. evaluation과 checkpoint도 같은 event ledger에서 counter를 얻는다.

**열두 microbatch 수치 timeline**

accumulation 4인 열두 microbatch를 만든다. 첫 window는 성공, 둘째는 세 번째 microbatch에서 overflow, 셋째는 성공이다. attempted update는 3회지만 committed update는 2회다. overflow window의 gradient를 버리는 정책이라면 consumed input token은 증가해도 valid committed token은 update에 기여하지 않는다.

warmup 4 committed update에서 lr factor가 `0.25,0.5,0.75,1.0`이라면 두 성공 update는 `0.25η,0.5η`를 쓴다. attempted clock은 세 번째 attempt에 `0.75η`를 줄 수 있다. microstep clock은 첫 window 안에서 lr이 바뀌는 더 큰 오류를 만든다. 독자는 event table에서 optimizer가 실제 읽은 lr을 계산한다.

GradScaler가 optimizer step을 skip했는지 확인하는 방식은 framework revision에 따라 고정한다. return value 추측이나 scaler 값 감소 하나에 의존하지 않고 optimizer committed event와 parameter/state delta를 확인한다. scheduler 호출 순서를 source와 local overflow fixture로 검증한다.

**token clock의 세 denominator**

input token은 padding과 prompt를 포함할 수 있고 valid loss token은 labels mask만 센다. packed SFT는 같은 input token에서도 assistant valid 비율이 다르다. pretraining causal LM과 assistant-only SFT를 token budget 하나로 비교하려면 어느 token이 schedule에 기여하는지 선언한다.

global valid count는 rank별 count sum이다. local mean을 평균하고 nominal batch×length를 token으로 쓰면 variable length와 last batch에서 틀린다. DP rank마다 `[120,80,100,60]` valid token이면 clock은 360이다. configured `4×128=512`를 더하지 않는다.

gradient accumulation 중 각 microbatch count를 ledger에 더하되 overflow로 commit되지 않은 window를 schedule token에 포함할지 정책을 정한다. compute-consumed token과 update-contributing token을 모두 저장하면 두 해석을 사후 구분할 수 있다. 결과 표의 primary x축을 명시한다.

**schedule 식을 손으로 검산한다**

linear warmup은 `t/W` 또는 `(t+1)/W`처럼 off-by-one 정의가 다를 수 있다. 첫 optimizer update가 0 lr인지 `η/W`인지 source lambda와 호출 순서를 함께 봐야 한다. step 0에서 scheduler를 먼저 호출하는지 optimizer 뒤 호출하는지도 실제 lr을 바꾼다.

cosine decay를 `η_min +(η_max−η_min)(1+cos(πp))/2`로 두고 progress `p=(t−W)/(T−W)`를 계산한다. `t=W`에서 max, `t=T`에서 min인지 경계 fixture로 확인한다. clamp가 없으면 T 이후 cosine이 다시 오를 수 있다. implementation의 clamp/terminal behavior를 source에서 읽는다.

WSD는 warmup, stable, decay 세 구간의 boundary와 continuity를 검사한다. decay 시작 checkpoint에서 next lr가 stable 마지막과 이어지는지 본다. resume off-by-one은 경계에서 가장 잘 드러난다. 각 구간 첫/마지막 세 step을 golden table로 둔다.

**Transformers source와 test 지도**

고정 Transformers commit에서 scheduler factory는 name을 lambda closure로 해석한다. 각 scheduler 함수의 arguments, warmup/training step과 cycles/min ratio를 기록한다. Trainer에서는 optimizer/scheduler 생성, optimizer step success 뒤 scheduler 호출, checkpoint state save/load를 잇는다.

factory source는 식을 보여주지만 GradScaler skip과 Trainer event order 전체를 증명하지 않는다. Trainer source는 호출 순서를 보여주지만 custom trainer override나 distributed wrapper가 같은 branch를 탔다는 증거가 아니다. run trace에서 selected scheduler class와 step event를 확인한다.

upstream test 표에는 경계 lr sequence, state dict resume, invalid option assertion을 적는다. overflow skip, elastic world size, token clock은 upstream이 다루지 않을 수 있어 local/proposed fixture로 남긴다. 함수가 존재하는 것과 test된 behavior를 분리한다.

**scaling rule은 schedule과 별 실험이다**

global batch를 k배 늘릴 때 linear lr scaling은 특정 optimizer와 noise regime의 heuristic이지 법칙이 아니다. warmup step을 그대로 두면 warmup token은 k배가 된다. token budget을 유지하려면 warmup step을 나누거나 token clock을 사용해야 한다. 두 선택을 명시한다.

sqrt scaling과 linear scaling, fixed lr를 같은 search budget에서 비교한다. primary endpoint는 token-to-validation과 stability다. large batch가 update 수를 줄여 decay 횟수와 data order를 바꾸므로 optimizer update budget과 token budget을 모두 보고한다.

gradient accumulation으로 batch를 키우는 경우와 DP world size로 키우는 경우는 system behavior가 다르다. 전자는 더 긴 accumulation window, 후자는 collective와 rank별 denominator를 바꾼다. mathematical effective batch가 같아도 throughput과 failure probability가 다르다.

**negative control 여섯 가지**

첫 control은 scheduler를 완전히 끄고 fixed lr로 one-step/replay를 검증한다. 둘째는 scheduler를 optimizer보다 먼저 호출해 off-by-one test가 실패하는지 본다. 셋째는 overflow인데 scheduler만 전진시킨다. 넷째는 rank 1 valid count를 global clock에서 누락한다.

checkpoint scheduler state를 한 step 앞서 저장하면 parameter/optimizer가 같아도 resume next lr는 달라져야 한다. 이어 world size를 바꾸고 nominal step clock만 유지해 token progress drift를 만들고, metric이 이 오류를 감지하는지 확인한다.

negative control이 실패를 만들지 못하면 정상 test도 민감하지 않다. 각 control은 expected lr sequence와 failure code를 가진다. loss가 finite하다는 이유로 통과하지 않는다.

**checkpoint event schema**

한 scheduler event는 RunID, attempted/committed step, input/valid/contributing token, current lr와 next lr, scaler skip, optimizer commit ID를 가진다. parameter group별 lr가 다르면 배열과 group digest를 저장한다. scheduler state dict hash만으로 의미를 복원하지 않는다.

checkpoint root는 model/optimizer/scheduler/scaler와 data cursor가 같은 committed event를 가리킨다. save 중 한 component만 미래 step이면 incomplete cut이다. root manifest를 마지막에 publish한다. loader는 current/next lr를 재계산해 event와 비교한다.

resume 첫 세 update를 uninterrupted control과 비교한다. warmup 종료, WSD decay 시작, overflow 직전처럼 경계 checkpoint를 선택한다. 평범한 중간 지점만 통과하면 boundary off-by-one을 놓친다.

**incident/RCA: warmup이 너무 빨리 끝났다**

먼저 authoritative clock과 schedule input을 비교해 microstep 또는 attempted step을 읽었는지 확인한다. 그다음 world size/effective batch 변경 전후의 warmup token을 계산하고, resume에서 scheduler가 이중 step됐는지 event ledger로 검사한다.

lr graph의 x축 label이 step뿐이면 valid token으로 다시 그린다. configured warmup 숫자와 realized token, committed update를 표에 둔다. 수정은 새 branch에서 boundary fixture와 next delta를 검증한다. 기존 curve의 x축을 조용히 재해석하지 않는다.

## 13.4 incident에서 scheduler 원인을 분리한다

loss spike와 learning rate 변화가 같은 시점에 나타났다는 사실은 인과의 증거가 아니다. event ledger에서 먼저 달라진 DrawID·분모·optimizer commit·scheduler event를 찾아, elastic migration과 resume가 만든 변화인지 schedule 자체의 변화인지 분리한다.

### 13.4.1 incident/RCA: loss spike와 lr은 동시에 변했다

loss spike 시각의 batch, gradient norm, scaler, clip, optimizer delta와 lr event를 join한다. scheduler jump가 원인인지 data outlier와 같은 시각에 우연히 겹쳤는지 최초 divergence를 찾는다. lr가 정상인데 optimizer internal step이 reset됐을 수도 있다.

여러 group 중 하나만 lr가 잘못됐다면 parameter group order와 scheduler base lr mapping을 본다. adapter를 중간에 추가한 경우 새 group scheduler state가 어떻게 초기화됐는지 확인한다. 전체 lr scalar 하나가 group-local 오류를 숨긴다.

원인 수정 뒤 저장 gradient replay에서 old/new lr delta를 비교하고 short confirmation을 한다. 장기 run에서만 확인하면 또 다른 data order가 변수를 섞는다.

**elastic world-size 변경**

world size가 8에서 16으로 바뀌면 per-rank batch가 같을 때 global batch와 token/step이 두 배가 된다. committed step clock을 그대로 이어도 token schedule은 갑자기 빨라진다. global batch를 유지하도록 per-rank batch를 줄이는 정책과 token clock을 유지하는 정책을 구분한다.

checkpoint에는 이전 world size와 realized global count를 저장한다. resume 첫 batch에서 새 global valid count를 검증하고 schedule progress를 recompute할지 기존 state를 유지할지 migration record를 만든다. 자동 heuristic으로 숨기지 않는다.

data sampler도 world size와 함께 바뀌어 sample sequence가 달라질 수 있다. scheduler-equivalent와 sample-exact를 다른 resume 등급으로 보고한다. lr가 같다는 사실로 data continuity를 주장하지 않는다.

**observability dashboard**

첫 panel은 attempted/committed step, valid/contributing token과 overflow skip이다. 둘째는 group별 current/next lr와 schedule segment다. 셋째는 gradient/update norm, loss와 validation이다. 넷째는 world size, effective batch, throughput이다.

metric은 authoritative event에서 생성하고 logger가 자체 local counter를 만들지 않는다. high-cardinality batch ID는 trace에 두고 dashboard는 RunID/segment를 쓴다. checkpoint marker를 graph에 표시해 resume 전후 discontinuity를 찾는다.

alert는 scheduler advanced without optimizer commit, counter regression, unexpected segment transition, group lr mismatch, token/step ratio drift를 잡는다. loss alert와 독립적으로 correctness invariant를 감시한다.

**독자 실습과 인수 기준**

독자는 12 microbatch overflow timeline을 계산하고 세 clock의 lr를 비교한다. warmup/cosine/WSD 경계의 expected sequence를 source implementation과 맞춘다. rank별 valid count가 다른 distributed fixture에서 global token clock을 만든다.

checkpoint scheduler state를 한 step 변조하고 resume next lr/delta gate가 실패하는지 본다. world-size 2→4 migration에서 step-clock과 token-clock 정책을 각각 재생한다. dashboard가 scheduler-only advance를 경보하는지 negative event를 넣는다.

최종 report는 schedule formula, authoritative clock, scaling rule, event/checkpoint schema, source/test evidence와 incident 결과를 가진다. 동일 lr 이름이나 예쁜 곡선이 아니라 어느 update와 token에 어떤 lr가 적용됐는지 재구성할 수 있어야 통과다.

**현장 결정표와 최소 제출 파일**

loss curve가 예상보다 빠르게 꺾이면 먼저 schedule segment와 authoritative clock을 본다. lr가 expected table보다 앞섰으면 microstep/attempted counter, resume double-step과 world-size token drift를 조사한다. lr sequence는 맞는데 loss만 다르면 data order, optimizer state와 gradient를 본다. scheduler에 원인을 강제로 귀속하지 않는다.

같은 step의 두 run이 다른 lr이면 group base lr, warmup/training total, cycles/min ratio와 resume state를 diff한다. 같은 lr인데 다른 token progress라면 effective batch와 valid denominator가 다르다. 곡선을 step으로 정렬할지 token으로 정렬할지는 연구 질문에 따라 결정하되 두 counter를 보존한다.

overflow가 잦은 run은 attempted step schedule이 빨리 진행할 뿐 아니라 contributing token mixture도 달라질 수 있다. skip된 window의 domain/length를 기록한다. numeric instability가 data selection을 만들지 확인한다. scaler fix 뒤 schedule만 맞추고 mixture shift를 무시하지 않는다.

최소 제출 파일은 `schedule-formula`, `clock-schema`, `boundary-golden`, `overflow-timeline`, `distributed-token-ledger`, `checkpoint-resume`, `source-test-map`, `dashboard-queries`다. 모든 파일은 RunID, optimizer commit과 scheduler recipe digest를 공유한다. current와 next lr, group별 lr을 포함한다.

`boundary-golden`은 warmup/cosine/WSD 구간의 처음과 끝을 수치로 가진다. `overflow-timeline`은 micro/attempted/committed/token clock을 나란히 둔다. `checkpoint-resume`은 경계 세 곳의 다음 lr와 parameter delta를 uninterrupted control과 비교한다. `source-test-map`은 formula factory, Trainer 호출과 upstream assertion을 분리한다.

**scaling 실험의 publication card**

card는 baseline global batch, accumulation, world size, valid token/update와 target total token을 쓴다. 후보마다 lr scaling rule, warmup 단위, decay end와 optimizer family를 적는다. configured batch뿐 아니라 realized valid token distribution을 보고한다.

결과는 validation 대 token, validation 대 wall time, stability/overflow, throughput과 memory를 가진다. large batch가 같은 token에서 update 횟수를 줄인 효과를 설명한다. search range와 failed run을 공개하고 best checkpoint selection rule을 사전에 둔다.

linear, sqrt와 fixed scaling을 비교했어도 보편 법칙을 선언하지 않는다. model, optimizer, data noise와 target regime에 조건부다. 다음 model size로 옮길 때 기존 rule은 prior이며 새 short range test와 clock audit가 필요하다.

**원인을 설명하는 구두 검산**

인수자는 checkpoint 하나를 골라 그때까지 committed update, input/valid/contributing token, current/next group lr와 segment를 event ledger에서 복원한다. 이어 overflow 한 번을 주입했을 때 어느 counter와 lr가 움직이고 무엇이 멈추는지 설명한다. parameter/optimizer와 scheduler가 같은 cut인지 확인한다.

다음 질문은 world size를 두 배로 했을 때 무엇이 달라지는가다. global batch, valid token/step, warmup token, decay progress, sampler와 collective를 구분하며, “linear scaling rule을 쓴다”라는 한 문장으로 schedule migration을 대신하지 않는다.

마지막으로 source에서 읽은 동작과 실제 실행 증거가 맞물리는지 확인한다. scheduler lambda가 계산한 값은 수식 구현을 보여 주고, Trainer의 호출 순서는 그 값이 어느 update에 적용됐는지를 보여 준다. local overflow·resume fixture는 skip과 복구 때 시계가 보존되는지만 증명하므로, 실행하지 않은 elastic fault까지 검증됐다고 확대해서는 안 된다. 이 세 근거가 서로 맞아야 curve의 x축과 lr를 설명할 수 있다.

**경계 회귀 표본**

CI의 빠른 fixture는 첫 warmup 네 update, cosine 시작·끝, WSD 세 boundary, overflow skip과 resume를 포함한다. expected group lr와 current/next counter를 exact 또는 엄격 tolerance로 비교한다. formula source나 Trainer 호출 순서가 바뀌면 golden table을 근거 없이 갱신하지 않는다.

distributed fixture는 rank마다 다른 valid token과 마지막 empty microbatch를 넣는다. global token clock과 gradient denominator가 같은 count source를 쓰는지 확인한다. world size를 바꾼 resume에서는 migration policy가 없으면 명시적으로 실패한다. 자동 nominal counter 이어붙이기를 허용하지 않는다.

장기 회귀는 validation 대 valid token과 wall time, overflow rate, realized lr integral을 비교한다. scheduler curve가 같아도 data mixture나 committed update가 다르면 직접 겹치지 않는다. RunID별 event ledger로 x축을 재구성한다.

**scheduler 면적과 누적 update.**

두 schedule의 peak lr와 마지막 lr가 같아도 중간 면적은 다를 수 있다. committed update별 lr 합과 token-weighted lr integral을 보조 통계로 낸다. 이것이 parameter 이동량을 직접 결정하는 것은 아니지만 schedule exposure 차이를 보여준다. gradient와 optimizer preconditioner를 무시한 “총 학습량”으로 과장하지 않는다.

overflow가 특정 구간에 몰리면 attempted schedule의 면적과 committed schedule의 면적이 갈린다. stable 구간에서 skip이 많았는지 decay 말기에 많았는지 validation 영향도 다를 수 있다. event ledger에서 segment별 skip과 contributing token을 계산한다.

large-batch scaling은 lr만 아니라 update 횟수를 줄인다. 같은 total token에서 `Σlr`와 token-weighted exposure를 함께 보고 optimizer state time constant가 step 단위인지 token 단위인지 논의한다. beta를 그대로 둔 채 batch만 키우면 moment가 보는 token horizon도 달라진다.

**인수 후 변경 절차**

max length, packing, assistant mask가 바뀌면 valid token/step이 달라져 token schedule을 재검증한다. optimizer family나 gradient accumulation 변경도 committed clock과 stable lr 범위를 바꾼다. 단순 data pipeline 변경으로 분류하지 않는다.

change RFC에는 old/new counter ratio, boundary table, expected validation checkpoint와 rollback을 쓴다. shadow event replay로 새 scheduler가 과거 batch ledger에서 어떤 lr를 냈을지 계산할 수 있다. 실제 parameter update와 장기 품질은 별 confirmation이다.

운영자는 변경 뒤 첫 boundary와 checkpoint를 집중 관찰한다. counter regression, double step, group mismatch가 없으면 정상 구간으로 확장한다. release가 실패하면 model만 rollback하지 않고 scheduler/scaler/optimizer와 cursor가 같은 parent cut으로 돌아간다.

최종 archive는 raw metric보다 event ledger를 우선 보존한다. metric backend의 downsampling 뒤에도 어느 optimizer commit에 어떤 lr가 적용됐는지 재계산할 수 있어야 한다. schema migration은 old/new counter 의미와 변환 가능 범위를 기록한다. 의미가 다른 series를 하나의 곡선으로 자동 연결하지 않는다.

인수자가 새 evaluator를 붙이면 EvalID가 참조한 committed/token counter를 함께 저장한다. evaluation 완료 시각을 training x축으로 쓰지 않는다. 비동기 평가가 늦게 돌아와도 실제 checkpoint의 schedule 위치에 결과를 놓는다.

이 원칙이 지켜져야 서로 다른 run의 곡선을 정직하게 비교할 수 있다.

**이 장이 넘기는 것.** `SchedulerClock={kind,count}`, global/effective token denominator, next lr, overflow-skip ledger.

**다음 장에서 깨질 수 있는 것.** 저정밀 overflow가 update를 건너뛸 때 scheduler clock이 함께 멈추지 않으면 곡선이 앞서간다.

**검증 체크포인트.** uninterrupted와 resume의 다음 세 lr, 소비 token ID, 성공 update 수를 비교한다.

**scheduler를 세 시계의 계약으로 읽는다**

Transformers의 공개 factory는 [`optimization.py`](https://github.com/huggingface/transformers/blob/v4.56.0/src/transformers/optimization.py)와 [`trainer.py`](https://github.com/huggingface/transformers/blob/v4.56.0/src/transformers/trainer.py)를 함께 읽는다. PyTorch 기반 계약은 [`lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py)와 [PyTorch optimizer 문서](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)에 고정한다. scaling-law 배경은 [Chinchilla 논문](https://arxiv.org/abs/2203.15556), gradient-noise 관점은 [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)에 둔다.

논문·API·채택 code commit은 서로 다른 증거다.

factory 이름이 같아도 `num_training_steps`의 단위가 optimizer update인지 microbatch인지 확인한다. gradient accumulation 뒤 optimizer가 실제 갱신된 사건에 scheduler를 연결해야 한다. AMP overflow로 optimizer step이 skip됐는데 scheduler만 전진하면 parameter는 그대로인데 lr clock만 소모된다. 왜 loss만으로 찾기 어려운가. overflow가 드물면 수백 step 뒤 decay 위치가 조금씩 어긋나기 때문이다.

세 시계는 `MicrobatchSeen`, `OptimizerUpdate`, `ValidTokenSeen`이다. sample cursor와 wall time은 관측용으로 더할 수 있지만 학습률을 어떤 시계의 함수로 정의했는지 하나를 정한다. variable-length packing에서는 같은 update가 처리한 valid token 수가 다르다. world size나 accumulation을 바꾸면 update clock과 token clock의 변환비가 바뀐다. `global_batch_size` 하나는 schedule의 충분한 상태가 아니다.

**한 update를 여덟 사건으로 펼친다.**

microbatch fetch, valid-token count, forward/backward, gradient accumulation, global denominator reduction, unscale/nonfinite 판정, optimizer commit, scheduler advance 순서다. 각 단계에 `BatchID`, local/global valid count, scaler decision, OptimizerStepID, SchedulerStepID와 applied lr을 남긴다.

마지막 microbatch가 empty라서 drop되면 일곱 microbatch로 update할지 전체 update를 폐기할지는 objective 계약이다. configured batch로 무조건 나누면 gradient scale이 작아진다. rank별 local count로 나누면 각 rank gradient가 다른 scale로 collective에 들어간다. global valid count의 owner와 reduction 시점을 고정한다.

overflow 실험은 특정 step gradient에 Inf를 주입해 optimizer와 scheduler가 함께 멈추는지 본다. scaler state만 변하고 parameter/moment/update clock은 유지되는 convention이라면 다음 finite step lr을 oracle과 비교한다. 중요한 것은 counter들이 같은 정수라는 사실이 아니라 동일 사건을 세는가이다.

**warmup·decay 식을 경계에서 검산한다**

linear warmup은 `t<W`에서 `lr(t)=lr_peak·t/W`처럼 쓰지만 `t=0` 첫 값과 `t=W` 포함 여부가 구현마다 다르다. 작은 `W=4` fixture로 생성 직후 lr, 첫 `scheduler.step()` 전후, optimizer 적용 lr을 표로 쓴다. off-by-one을 자연어로 논쟁하지 않고 실제 네 값을 비교한다.

cosine decay는 endpoint와 minimum ratio가 중요하다. progress를 어떤 horizon으로 나누는지 확인한다. horizon을 100에서 200으로 늘린 resume migration은 현재 lr만 맞춰서는 부족하다. 이후 기울기와 누적 면적이 달라진다. old schedule을 계속할지 remaining horizon을 재계산할지 새 RunID로 남긴다.

WSD는 warmup, stable, decay 경계를 가진다. stable 구간은 lr이 같아도 data와 optimizer state가 변한다. decay 시작을 token budget으로 정의했다면 elastic world size 변경에도 token clock에서 같은 지점을 유지한다. update count로 정의했다면 처리 token이 달라지는 것을 받아들인다. 두 의미를 동시에 만족한다고 쓰지 않는다.

schedule 면적 `Σlr_t`는 update를 완전히 결정하지 않지만 controlled replay에서 곡선 차이를 요약한다. endpoint, peak, warmup/stable/decay area를 계산한다. 두 schedule이 peak와 final lr만 같아도 면적이 다르다. 이 검산은 이름이 같은 cosine 구현이 다른 학습을 만드는 이유를 보여 준다.

**scaling law를 recipe로 오역하지 않는다**

compute-optimal scaling law는 특정 data/model/optimizer 조건에서 model size와 token budget 관계를 추정한다. “parameter 수의 몇 배 token이면 모든 fine-tuning이 최적”이라는 규칙이 아니다. downstream objective, repeated data, contamination, multimodal token cost가 다르면 범위가 달라진다. 논문 fit의 관측 범위와 extrapolation을 구분한다.

critical batch도 training stage, mixture, length와 group에 따라 달라진다. batch sweep은 lr sweep과 얽힌다. fixed lr, scaling-rule 후보, token horizon을 나눠야 batch 효과와 lr 효과를 식별한다. linear lr scaling은 heuristic이며 Adam denominator, warmup, clipping과 loss normalization 아래서 보장되지 않는다. square-root scaling과 fixed lr를 대조군으로 둔다.

관측값은 train loss뿐 아니라 gradient norm/noise scale, update-to-weight ratio, validation, samples-to-target, wall time이다. configured batch와 realized valid-token batch를 함께 기록한다. state 절감으로 batch를 늘린 optimizer 결과는 fixed-batch 결과와 별 실험이다.

**resume와 elastic 변경의 결정 트리**

checkpoint는 scheduler name, constructor arguments, state dict, base lr per group, last applied lr, update/token/sample clock, planned horizon, warmup boundary, optimizer/scaler commit ID를 담는다. config만 저장하면 default 또는 library upgrade로 branch가 달라질 수 있다. 새 process에서 다음 세 lr을 미리 생성해 uninterrupted oracle과 비교한다.

resume 직후 lr이 다르면 load/call 순서를 본다. optimizer state 전에 scheduler를 만들었는지, `last_epoch` convention이 무엇인지, state가 base lr를 덮었는지 확인한다. lr은 같은데 parameter가 다르면 optimizer/scaler/denominator로 이동한다. 몇 step 뒤 경계에서만 갈리면 horizon과 branch를 본다.

world size가 64에서 56으로 줄면 microbatch와 accumulation을 바꿔 global batch를 유지하거나, batch와 lr rule을 새로 정하거나, valid-token clock에서 같은 boundary를 유지할 수 있다. 어느 것도 자동 동등하지 않다. 변경 전후 global batch, token/update ratio, effective lr, remaining budget을 migration card에 쓴다.

고장 주입은 scheduler state 한 step rollback, optimizer만 advance, overflow에서 scheduler advance, partial accumulation checkpoint, token count 중복을 포함한다. verifier는 다음 lr 또는 token checksum 불일치로 실패해야 한다. 디버깅은 `last common event`를 찾고 microbatch→optimizer commit→scheduler advance 순서로 ledger를 대조한다.

**dashboard와 인수 실험**

dashboard는 configured lr가 아니라 group별 applied lr을 낸다. update rate, valid tokens/update, accumulation count, skipped updates, scaler scale, schedule phase와 boundary를 같은 timeline에 둔다. rank마다 lr가 다르면 중단한다. group lr ratio가 recipe와 다르면 group order 또는 state load를 확인한다.

실험 1은 12-step uninterrupted와 step 5 resume를 비교한다. 실험 2는 step 4 overflow를 넣는다. 실험 3은 variable-token microbatch로 token clock을 검증한다. 실험 4는 world-size migration을 dry-run한다. 실험 5는 warmup/cosine/WSD 경계 전후를 snapshot한다. 각 실험은 expected failure를 가진다.

최종 `ScheduleClockCard`에는 clock owner, increment event, denominator, accumulation/overflow policy, curve 식과 boundary, horizon unit, group lr, resume call order, elastic migration policy를 기록한다. Checksum과 다음 세 expected lr를 14장에 인계한다. 저정밀 overflow가 update를 skip할 때 scheduler 계약이 깨지지 않는지 scaler state machine이 이어서 검증한다.

마지막 검토는 네 번의 “왜”로 끝낸다. 왜 이 schedule의 clock이 update인가, 왜 warmup boundary가 이 값인가, 왜 batch 변경 때 lr rule을 유지했는가, 왜 resume 뒤 다음 lr가 동일하다고 말할 수 있는가를 각각 ledger와 수식으로 답한다. 답이 config 이름만 가리키면 실험 근거가 부족하다. 검토자는 한 counter를 의도적으로 뒤로 돌린 뒤 디버깅 경보가 최초 불일치 사건을 정확히 지목하는지도 확인한다.

**하나의 학습 run에는 서로 다른 시계가 여럿 있다**

**microbatch, token, update, sample 시계를 분리한다.**

데이터 로더는 sample cursor를 전진시키고 collator는 실제 valid token을 결정한다. gradient accumulation은 여러 microbatch를 한 optimizer update로 묶는다. overflow가 나면 forward/backward와 token 소비는 있었지만 parameter update는 없을 수 있다. 따라서 `step`이라는 이름 하나로 이 사건들을 세면 resume와 schedule이 모호해진다.

최소 ledger는 `DataBatchID`, `MicrobatchID`, `ConsumedSampleCount`, `ConsumedValidTokenCount`, `AccumulationSlot`, `AttemptedUpdateID`, `CommittedOptimizerStepID`, `SchedulerStepID`를 가진다. 각 counter의 owner와 increment 사건을 적는다. attempted update와 committed update를 나누면 overflow skip을 표현할 수 있다.

gradient accumulation `K`에서 microbatch loss가 mean이면 단순 합산 뒤 `K`로 나누는 관례가 있다. variable length에서는 각 microbatch token 수가 다르므로 이것이 global-token mean과 같지 않다. 정확한 token mean은 numerator를 합하고 global valid-token denominator로 한 번 나눈다. scheduler가 token clock을 쓴다면 바로 그 denominator ledger와 연결한다.

pipeline parallel에서는 microbatch가 여러 stage에 동시에 존재한다. 마지막 stage가 loss를 계산하고 첫 stage backward가 끝난 뒤에야 update가 commit된다. 한 stage의 local microbatch 완료를 global update로 세면 clock이 앞선다. pipeline flush와 virtual stage schedule을 포함한 global commit barrier를 정의한다.

gradient scaler가 overflow를 발견하는 시점도 중요하다. rank 하나만 nonfinite를 발견해도 모든 replica가 update를 skip해야 한다. nonfinite flag를 DP group에서 reduce하고 optimizer/moment/decay/scheduler를 모두 멈춘다. 일부 rank가 update한 뒤 flag를 알면 parameter가 갈라진다. flag collective는 optimizer commit의 선행 조건이다.

`zero_grad`는 clock을 전진시키지 않지만 partial accumulation state를 파괴할 수 있다. checkpoint가 microbatch 경계에 저장된다면 accumulated gradient, accumulation slot, loss numerator/denominator를 저장하거나 마지막 committed update로 rollback해야 한다. “언제든 저장 가능”이라는 설명은 이 선택 없이 성립하지 않는다.

**scheduler API의 호출 순서를 applied learning rate로 검증한다.**

일부 API는 optimizer를 먼저 step하고 scheduler를 뒤에 step하도록 설계된다. 생성 직후 base lr가 첫 update에 적용되고, scheduler call이 다음 lr를 준비한다. 반대 순서면 첫 schedule 값을 건너뛸 수 있다. `last_epoch`라는 이름도 실제 epoch가 아니라 call count일 수 있다.

검산표는 constructor 직후 `optimizer.param_groups[i]['lr']`, update 0에 실제 적용된 lr, scheduler call 뒤 stored lr를 나란히 둔다. parameter 하나와 constant gradient를 써 delta에서 applied lr를 역산한다. 로그에 표시된 lr가 실제 update와 같은지 확인하는 가장 강한 방법이다.

PyTorch scheduler를 고정할 때 `torch/optim/lr_scheduler.py`의 base class 초기화, `step`, `get_lr`, `get_last_lr`, 각 subclass의 boundary 식을 따라간다. trainer wrapper가 epoch argument를 전달하는지, optimizer step을 skip해도 scheduler를 호출하는지 호출자를 함께 읽는다. library 함수만 봐서는 통합 순서를 알 수 없다.

Transformers Trainer 계열은 total training steps를 dataloader length, accumulation, epoch, max steps로 계산할 수 있다. iterable dataset처럼 길이를 모르면 explicit max step이 필요할 수 있다. distributed sampler와 drop_last가 realized updates를 바꾼다. scheduler constructor에 들어간 horizon이 실제 committed update horizon과 맞는지 startup audit에서 계산한다.

DeepSpeed, Megatron, Accelerate 같은 runtime은 optimizer와 scheduler step을 wrapper 안에서 결합할 수 있다. 사용자가 바깥에서 다시 scheduler를 호출하면 두 번 전진한다. 반대로 wrapper가 overflow에서 scheduler를 멈추는지 확인하지 않고 수동 보정하면 갈릴 수 있다. ownership은 정확히 하나여야 한다.

**warmup과 decay를 수학적 경계 조건으로 설계한다**

**warmup은 초기 불확실성을 다루지만 만능 안정화 장치가 아니다.**

초기에는 optimizer moment가 비어 있고 activation/gradient scale이 빠르게 변할 수 있다. 큰 peak lr를 즉시 적용하면 불안정할 수 있어 warmup으로 update scale을 서서히 올린다. 그러나 divergence 원인이 잘못된 loss denominator, initialization, overflow라면 warmup을 늘려 증상을 늦출 뿐이다. warmup 변경 전에 first-step tensor와 update ratio를 검사한다.

linear warmup에서 first applied lr를 0으로 둘지 `peak/W`로 둘지 정해야 한다. `W=4`이면 가능한 수열은 `[0,.25,.5,.75,1]` 또는 `[.25,.5,.75,1]`처럼 다르다. boundary 포함 여부와 총 warmup update 수를 수열로 명시한다. 이름만 `linear`라고 쓰지 않는다.

cosine decay를 `min+(peak-min)·(1+cos(πp))/2`라 할 때 progress `p`의 denominator가 핵심이다. decay 구간에 endpoint를 포함하려면 첫/마지막 index를 식에 넣어 검산한다. integer division, clamp 위치, horizon보다 긴 resume에서의 동작도 test한다.

WSD의 stable phase는 실험적으로 decay 시작점을 뒤로 미루거나 여러 checkpoint에서 decay branch를 시도하는 데 유용할 수 있다. 하지만 stable lr가 학습 상태를 정지시키는 것은 아니다. data curriculum, optimizer moment, weight norm은 계속 변한다. stable checkpoint에서 여러 decay 길이를 fork할 때 data cursor와 optimizer state를 동일하게 복제한다.

inverse-square-root schedule은 warmup 뒤 `1/sqrt(t)` 형태지만 normalization constant와 splice 연속성을 확인한다. polynomial decay는 power와 endpoint, constant schedule with warmup은 warmup 뒤 평탄하다. schedule family 비교는 peak/final뿐 아니라 area와 boundary derivative를 본다.

parameter group마다 lr multiplier가 있으면 scheduler가 absolute lr를 덮는지 base lr 비율을 유지하는지 확인한다. embedding/head, dense/Muon group이 다른 multiplier를 가질 수 있다. resume에서 param group 순서가 바뀌면 같은 index state가 다른 logical group에 붙는다. group stable ID와 base lr를 checkpoint에 둔다.

**token 기반 schedule은 데이터 의미와 연결된다.**

update clock은 batch가 고정일 때 편리하지만 sequence length, packing efficiency, world size가 바뀌면 update당 token이 달라진다. token clock은 실제 유효 token 처리량에 맞춰 warmup과 decay를 유지할 수 있다. 대신 global count reduction, padding 제외, repeated/replayed batch 처리 정책이 필요하다.

token schedule에서 현재 microbatch가 boundary를 넘으면 어느 lr를 적용할지 정한다. update 시작 전 누적 token을 기준으로 할지, 이번 update의 token을 포함한 뒤 기준으로 할지 다르다. 큰 variable batch가 boundary를 건너뛰는 경우 interpolation 또는 stepwise 적용을 명시한다.

sample clock도 token clock과 다르다. 이미지-텍스트, 음성, 가변 길이 문서에서 sample 하나의 compute와 token 의미가 크게 다르다. multimodal mixture는 modality별 cost-normalized clock을 고려할 수 있지만 objective weight와 혼동하지 않는다. 21장의 modality tokenization 계약과 연결한다.

data curriculum이 source mixture를 시간에 따라 바꾸면 같은 token 수라도 gradient distribution이 다르다. schedule phase와 curriculum phase를 동시에 바꾸면 원인을 분리하기 어렵다. change-point를 어긋나게 한 대조군 또는 factorial design을 사용한다. ledger에는 `MixtureRevision`과 `SchedulePhase`를 함께 기록한다.

**scaling law를 예산 결정과 온라인 제어로 나눈다**

**fitted power law는 관측 범위와 잔차를 가진다.**

scaling law는 loss와 model size, data, compute 사이 경험적 관계를 power law로 근사한다. 계수는 tokenizer, data quality, architecture, optimizer와 평가 분포에 의존한다. 논문에서 얻은 exponent를 새 domain에 그대로 적용하기 전에 소규모 sweep에서 residual을 본다.

Chinchilla류 compute-optimal 결과는 주어진 compute에서 parameter와 token 배분을 논한다. 이미 사전학습된 모델의 SFT에 “parameter 수의 일정 배 token”을 직접 적용하는 법칙이 아니다. fine-tuning은 시작점, data diversity, forgetting, downstream metric이 다르다. pretraining budget과 adaptation budget을 분리한다.

loss floor가 있는 power law를 log-log 직선으로 맞출 때 floor 추정 오류가 exponent를 왜곡한다. 작은 run의 optimization transient와 data exhaustion도 fit을 흐린다. confidence interval, held-out scale 예측, 다른 seed를 포함한다. point estimate 하나로 cluster 구매를 확정하지 않는다.

online training에서 scaling law는 stopping forecast에 쓸 수 있지만 schedule 변경과 data mixture 변경으로 stationary assumption이 깨진다. 최근 loss slope만 extrapolate하지 않고 validation, irreducible floor, compute remaining을 함께 본다. forecast model revision을 run ledger에 저장한다.

**batch scaling은 gradient noise와 시스템 효율의 절충이다.**

global batch를 키우면 update당 gradient variance는 줄지만 update 횟수도 줄 수 있다. critical batch 이후에는 추가 병렬성이 sample efficiency를 크게 해칠 수 있다. critical batch는 고정 상수가 아니라 loss level과 data/model에 따라 변할 수 있다.

gradient noise scale을 추정하려면 서로 다른 microbatch의 gradient 통계가 필요하다. 전체 gradient tensor 저장은 비싸므로 layer sample, norm, inner product estimator를 사용할 수 있다. estimator variance와 communication 비용을 보고한다. 관측값을 자동 lr 변경에 쓰기 전 offline 상관을 검증한다.

linear lr scaling은 SGD 직관에서 자주 쓰이지만 AdamW, clipping, normalization에서 보장되지 않는다. batch multiplier `k`에 대해 fixed lr, `sqrt(k)`, `k` 후보를 sweep하고 warmup도 독립 축으로 둔다. 같은 token budget에서 validation과 update norm을 비교한다.

시스템상 batch를 키우면 GEMM 효율과 통신 amortization이 좋아질 수 있으나 activation memory가 늘고 optimizer update 빈도가 줄어든다. gradient accumulation으로 키운 batch와 data parallel world size로 키운 batch는 communication과 batchnorm류 상태가 다르다. LLM에서 norm이 token-local이어도 RNG, sampler, all-reduce order가 달라진다.

### 13.4.2 elastic world-size 변경을 recipe migration으로 취급한다

**보존할 불변식을 먼저 선택한다.**

world size가 바뀔 때 global batch, microbatch per device, accumulation, token/update, lr, remaining horizon을 모두 동시에 보존할 수 없는 경우가 많다. 무엇을 보존할지 우선순위를 정한다. 품질 trajectory를 우선하면 global valid-token batch와 token clock을 유지하려 할 수 있다. throughput을 우선하면 batch가 바뀌고 새 lr 검증이 필요하다.

예를 들어 DP 64에서 device당 microbatch 2, accumulation 4라면 configured global sample batch는 512다. DP 56에서 같은 값을 유지하면 448이 된다. accumulation을 32/28 같은 비정수로 만들 수 없으므로 microbatch나 일부 update의 accumulation pattern을 바꿔야 한다. variable accumulation은 denominator와 scheduler boundary를 복잡하게 한다.

sequence parallel 또는 context parallel 크기가 바뀌면 sample batch는 같아도 local sequence shard와 communication이 바뀐다. TP/PP 변경은 parameter/optimizer reshard와 pipeline schedule을 동반한다. `world_size` 하나가 아니라 DP/TP/PP/CP/EP mesh 각 축을 migration card에 쓴다.

sampler는 global consumed sample cursor에서 새 rank partition을 재생성해야 한다. rank-local cursor를 그대로 load하면 sample 중복과 누락이 생긴다. exact data order를 보존할지 multiset만 보존할지 정한다. curriculum이 token clock을 사용한다면 duplicated token이 clock을 두 번 전진하지 않게 commit ledger를 쓴다.

optimizer state reshard가 끝나기 전에 scheduler를 advance하지 않는다. 모든 parameter shard, moment, scaler, sampler, schedule state가 동일 checkpoint commit을 가리켜야 run을 publish한다. dry-run은 다음 data IDs, 다음 세 lr, parameter group digest, one-step delta를 실제 학습 재개 전에 계산한다.

**overflow와 elasticity가 겹치는 경계를 시험한다.**

마지막 old-world update가 overflow로 skip되고 곧바로 resize될 수 있다. consumed token은 늘었지만 committed optimizer step은 그대로다. schedule이 update clock이면 lr는 멈추고, token clock이면 정책에 따라 전진할 수 있다. 이 선택을 명시하지 않으면 두 구현이 모두 그럴듯해진다.

partial accumulation 중 resize는 accumulated gradient를 global logical sum으로 변환해 새 mesh에 옮기거나 폐기해야 한다. rank-local buffer를 새 rank에 단순 배분하면 denominator가 깨진다. 현실적으로 마지막 committed update로 rollback하는 정책이 단순할 수 있지만 소비 데이터 재생과 RNG rollback을 함께 처리한다.

failure injection matrix는 overflow 직전/후, warmup 마지막 step, cosine 첫/마지막 step, WSD phase boundary, checkpoint save 도중, resize 직전/후를 교차한다. 모든 조합을 대규모로 실행할 필요는 없지만 작은 deterministic parameter와 synthetic gradient로 state machine을 검증할 수 있다.

관측 dashboard는 run별 하나의 `step` 그래프 대신 여러 clock을 겹쳐 보여 준다. lr은 SchedulerStepID와 CommittedOptimizerStepID에 붙이고, throughput은 wall time/token clock에 붙인다. skipped update에는 원인 code를 표시한다. resize event 전후에 batch, topology, lr rule revision을 annotation한다.

최종 승인 조건은 다음과 같다. source 고정 revision과 호출 순서가 문서화되어 있고, 경계 수열 oracle이 있으며, overflow에서 모든 rank가 원자적으로 멈추고, resume 다음 세 lr와 delta가 맞고, elastic migration의 보존 불변식이 명시되어야 한다. 하나라도 빠지면 schedule 이름이 같다는 사실만 남는다.

**운영 관측과 실험 판정을 하나로 닫는다**

**dashboard의 x축을 먼저 고른다.**

loss와 lr를 wall time에만 그리면 장애 정지와 느린 step이 보이지만 학습 진척을 비교하기 어렵다. committed update 축, consumed valid-token 축, wall-time 축을 전환할 수 있게 한다. overflow 구간에서는 token은 늘고 update는 멈추는 모양이 보여야 한다.

필수 metric은 group별 applied lr, valid tokens/update, attempted/committed update, skipped reason, scaler scale, gradient norm, update-to-weight ratio, schedule phase, data mixture revision이다. rank별 lr는 low-cardinality consistency check로 수집하고 불일치 시 즉시 중단한다.

warmup divergence에서는 첫 nonfinite tensor와 applied update ratio를 본다. decay 전환에서 loss가 튀면 boundary off-by-one, data curriculum 동시 전환, optimizer cadence를 분리한다. resume만 갈리면 다음 세 lr oracle과 group digest를 먼저 본다.

alert는 값 하나보다 invariant에 건다. scheduler step이 optimizer commit보다 앞섰거나, skipped update에서 moment가 변했거나, rank별 lr가 다르거나, token counter가 감소하면 실패다. 정상적으로 lr가 낮아지는 것을 threshold alert로 오인하지 않는다.

**최소 인수 시험은 경계와 고장을 겨냥한다.**

첫 시험은 12 update uninterrupted와 step 5 checkpoint/resume다. 둘째는 warmup 마지막과 decay 첫 step 전후다. 셋째는 한 rank에 Inf를 주입한다. 넷째는 가변 token microbatch다. 다섯째는 world-size migration dry-run이다.

각 시험은 expected lr 수열, parameter delta, counter ledger, 다음 data ID를 가진다. 성공 여부를 최종 loss 하나로 판단하지 않는다. 최초 불일치 사건과 관련 source symbol을 report에 남긴다.

scaling 실험은 seed 반복과 confidence interval을 포함한다. lr/batch/warmup을 동시에 무제한 sweep하지 않고 가설별 축을 분리한다. token budget, data order, evaluation cadence가 같은지 확인한다.

최종 `ScheduleClockCard`와 12장의 `OptimizerGeometryCard`를 조인하면 어느 lr가 어느 algorithm state에 적용되었는지 재현할 수 있다. 14장은 여기에 scaler dtype, nonfinite collective, autocast 경계를 추가한다.

이 장의 완료는 cosine 수식을 아는 데 있지 않다. 데이터가 소비되고 gradient가 만들어지고 update가 commit되며 장애 뒤 다시 시작될 때, 각 시계가 왜 그 값인지 증명할 수 있어야 한다. 그 증명이 있어야 scaling law와 cluster elasticity가 실제 recipe가 된다.

**recipe 변경 기록의 표준 형식**

변경 전후를 한 표에 둔다. clock type, current counter, warmup/decay boundary, horizon, group base lr와 multiplier, global valid-token batch, accumulation, topology, optimizer/scaler commit을 기록한다. “lr를 조금 낮춤” 같은 자연어만 남기지 않는다.

변경 이유는 관측 증거와 연결한다. gradient/update ratio, overflow rate, validation slope, throughput 또는 장애로 어떤 가설을 세웠는지 쓴다. 결과가 나빠도 기록을 삭제하지 않는다. 다음 실험이 같은 실패를 반복하지 않게 한다.

schedule을 중간 변경하면 기존 RunID의 동일 recipe 연속이 아니라 migration event다. 현재 lr의 연속성, 이후 곡선 기울기, remaining area 중 무엇을 보존했는지 명시한다. 세 조건을 모두 만족하지 못할 수 있다.

elastic 변경도 같은 형식을 쓴다. old/new mesh, sample/token cursor, reshard manifest, next data IDs, next three lr를 붙인다. dry-run certificate가 통과한 뒤 새 commit을 publish한다.

overflow 대응으로 scaler만 바꿨는지 lr와 batch도 바꿨는지 분리한다. 여러 축을 동시에 바꾸면 원인 식별력이 떨어진다. 긴급 복구라면 운영 변경과 후속 controlled experiment를 별도로 기록한다.

논문 또는 외부 recipe를 옮길 때 schedule 이름보다 실제 수열을 복원한다. framework version, call order, horizon 계산, first/last applied lr를 고정한다. 같은 cosine이라는 명칭은 재현성을 보장하지 않는다.

최종 handoff bundle은 `ScheduleClockCard`, boundary oracle, resume/overflow/elastic 시험 결과, dashboard snapshot, source/test coordinate를 담는다. 14장은 이 bundle에서 unscale과 nonfinite 판정이 optimizer/scheduler commit을 어떻게 gate하는지 이어받는다.

이 표준을 따르면 학습 곡선이 달라졌을 때 막연히 seed 탓을 하지 않는다. 데이터, clock, optimizer, scaler, topology 가운데 최초로 달라진 사건을 찾는다. scheduler는 작은 보조 객체가 아니라 전체 훈련 사건을 시간축에 배치하는 제어기다.

최종 검토자는 세 종류의 증거를 교차한다. 정적 증거는 고정 revision의 scheduler 식과 trainer 호출 순서다. 수치 증거는 작은 horizon에서 손으로 계산한 lr 수열과 parameter delta다. 운영 증거는 attempted/committed update, valid token, overflow, resize event가 담긴 ledger다. 세 증거 중 하나만 있으면 부족하다. source 식이 맞아도 wrapper가 두 번 호출할 수 있고, 수열이 맞아도 resume state가 누락될 수 있으며, dashboard가 매끈해도 잘못된 counter를 표시할 수 있다. warmup 마지막, decay 시작, overflow skip, checkpoint resume, world-size 변경을 경계 fixture로 고정한다. 각 fixture에서 다음 data ID, 다음

세 applied lr, optimizer delta, 모든 counter를 비교한다. scaling law를 적용한 budget 변경은 fit 범위와 confidence interval을 함께 남긴다. batch scaling은 realized valid-token batch와 gradient noise 관측을 포함한다. 이 교차 검증이 통과해야 recipe가 재현 가능하다고 말한다. 그렇지 않으면 schedule 이름과 최종 loss만 남아 원인을 설명할 수 없다.

**scheduler의 시간은 하나가 아니다**

훈련 loop에는 적어도 microbatch, backward, attempted update, committed optimizer update, consumed sample, valid token, wall-clock 시간이 공존한다. `global_step`이라는 변수 하나에 이 의미를 모두 맡기면 accumulation, overflow, elastic resize에서 모순이 생긴다. 예를 들어 microbatch 네 개를 모아 한 번 update하는 동안 data cursor는 네 번 움직이지만 optimizer bias correction과 learning-rate schedule은 한 번 움직인다. 마지막 microbatch에서 Inf가 발견되어 update가 취소되면 attempted counter만 증가하고 committed counter는 멈출 수 있다.

token clock은 길이가 가변적인 언어 학습에서 특히 중요하다. sequence 수가 같아도 padding과 packing에 따라 valid prediction token 수가 다르다. schedule horizon을 sample로 정의한 recipe와 token으로 정의한 recipe는 같은 dataloader에서도 다른 곡선을 만든다. token count에는 prompt masking, dropped tail, multimodal placeholder, loss mask를 반영한 “실제로 denominator에 들어간 token”과 단순 input token을 구분한다. throughput 보고용 token과 optimization clock용 token이 다르면 둘 다 이름을 붙인다.

wall-clock schedule은 spot interruption이나 fixed-time budget에서 유용할 수 있지만 재현성이 hardware와 장애에 묶인다. compile warmup, evaluation, checkpoint pause를 elapsed time에 포함하는지에 따라 lr가 달라진다. 동일 checkpoint를 더 빠른 cluster에서 재개하면 남은 curve가 달라질 수도 있다. 그러므로 wall clock을 택하면 monotonic source, pause 정책, resume 기준과 cluster migration 정책을 state schema에 둔다.

각 clock은 단조성, 증가 조건, owner, serialization, reconciliation 규칙을 가진다. distributed training에서 rank별 valid token이 다르면 어느 collective로 global count를 만들지 정한다. rank 하나가 update commit 전에 죽었을 때 data cursor만 앞선 상태를 publish하지 않는다. checkpoint는 여러 clock의 consistent cut이어야 한다. load 후 첫 event에서 counter를 추측하거나 dataloader 길이로 재계산하지 않는다.

**warmup은 숫자보다 초기 상태의 전환이다**

warmup은 learning rate를 작은 값에서 키우는 모양만 뜻하지 않는다. optimizer moment가 비어 있고 activation·gradient scale이 아직 안정되지 않았으며 distributed pipeline과 compiler가 warmup 중인 초기 구간을 다룬다. linear warmup은 구현이 단순하지만 첫 applied lr가 0인지 base/warmup_steps인지, 경계 step이 중복되는지가 framework마다 다를 수 있다. 작은 horizon 표를 손으로 만들어 boundary를 고정해야 하는 이유다.

warmup 길이를 “전체 step의 3%”라고 적으면 token budget이나 batch가 바뀔 때 의미도 바뀐다. 안정화에 필요한 update 수를 보존할지, 소비 token을 보존할지, 전체 budget 비율을 보존할지를 결정한다. batch를 키우면서 warmup step을 그대로 두면 warmup 동안 훨씬 많은 token을 소비한다. 반대로 token warmup을 고정하면 update 수가 줄어 optimizer moment의 시간 상수가 달라진다. 어느 불변량이 학습 가설인지 명시한다.

optimizer의 beta와 warmup은 독립 knob이 아니다. first·second moment의 유효 기억 길이와 bias correction이 초기 update를 이미 조절한다. Muon이나 Shampoo처럼 별 cadence와 matrix state를 가진 optimizer는 warmup 동안 state가 몇 번 갱신되는지도 본다. FP8 amax history와 dynamic loss scale도 초기 상태를 가진다. 모든 subsystem을 같은 warmup step으로 묶기보다 각 상태가 안정화되는 관측량을 기록한다.

warmup 장애 fixture는 경계 직전 저장, 경계에서 overflow, 경계 직후 재개를 포함한다. scheduler가 `last_epoch`를 어떤 의미로 저장하는지, optimizer step보다 앞서 호출되었는지 확인한다. resume run의 다음 세 learning rate와 delta가 uninterrupted run과 맞아야 한다. 로그의 lr가 “다음 step에 쓸 값”인지 “방금 적용한 값”인지도 구분한다.

**critical batch와 gradient noise를 비용 함수로 읽는다**

batch를 늘리면 독립 sample gradient의 잡음이 평균화되어 update 방향의 분산이 줄어든다. 작은 batch 영역에서는 더 큰 batch가 같은 update 횟수로 더 정확한 방향을 주고 parallel hardware를 활용할 수 있다. 그러나 어느 지점 이후에는 추가 sample이 방향 정보를 충분히 늘리지 못하고, token당 progress가 줄어드는 영역이 나타날 수 있다. 이 전환을 무조건적인 상수로 부르지 않고 model, data, training phase와 objective에 조건부인 관측값으로 취급한다.

linear learning-rate scaling이나 square-root scaling은 법칙이라기보다 특정 noise·optimizer 조건의 출발 가설이다. global batch를 \(k\)배 키웠다고 lr를 자동으로 \(k\)배 올리지 않는다. update norm, gradient norm·variance, loss spike, validation progress를 통해 검증한다. Adam류의 adaptive denominator, clipping, weight decay와 warmup이 scaling을 바꾼다. local microbatch와 accumulation으로 같은 global batch를 만들더라도 normalization, BatchNorm류 state, dropout RNG와 communication timing이 같다는 보장은 없다.

critical batch 실험은 fixed-token 관점과 fixed-step 관점을 분리한다. fixed token에서는 큰 batch가 update 수를 줄이므로 sample efficiency를 본다. fixed step에서는 더 많은 token과 compute를 쓰므로 wall-clock throughput과 품질이 섞인다. hardware utilization이 나쁜 작은 batch와 잘 찬 큰 batch를 비교할 때는 system efficiency와 statistical efficiency를 각각 보고한다. 최종 선택은 둘의 곱인 time-to-quality로 판단할 수 있다.

관측치는 단일 gradient norm으로 충분하지 않다. 동일 checkpoint에서 여러 독립 microbatch gradient를 저장해 평균과 변동을 추정하고, layer/group별 signal-to-noise와 update-to-weight ratio를 본다. exact per-example gradient가 비싸다면 근사를 쓰되 estimator와 confidence interval을 기록한다. training 중 online proxy가 drift하면 주기적으로 고정 probe batch에서 재측정한다.

**scaling law를 외삽 공식이 아니라 의사결정 도구로 쓴다**

scaling law fit은 관측한 model·data·compute 범위에서 loss의 추세를 압축한다. 그 식을 새 architecture, 새 tokenizer, 다른 quality mixture와 post-training objective로 곧바로 외삽하지 않는다. 각 점의 token definition, parameter count convention, evaluation loss와 training compute 회계를 통일한다. embedding을 parameter에 포함했는지, sparse expert의 total과 active parameter 중 무엇을 썼는지도 표시한다.

fit에는 불확실성이 있다. seed variance, 짧은 run의 transient, data contamination, optimizer 미튜닝과 measurement noise를 포함한다. 점 추정 exponent만 보고 수십 배 budget을 결정하지 않고 confidence interval과 alternative fit을 둔다. holdout scale 하나를 남겨 외삽 오차를 검사하고 residual이 architecture나 data regime에 따라 구조적으로 치우치는지 본다. power law가 잘 맞지 않는 구간을 억지로 한 직선으로 만들지 않는다.

compute-optimal 결론은 고정 compute에서 model과 token 배분을 정하는 질문이지, 제품 목표 전체를 대신하지 않는다. memory, latency, inference cost, data availability, training deadline과 downstream 성능이 제약으로 들어간다. 부족한 고품질 data를 여러 epoch 재사용하면 독립 token 가정이 깨지고 memorization·overfitting이 달라진다. mixture와 curriculum이 바뀌면 effective data quality도 변한다.

작은 proxy run에서 대형 run recipe를 얻으려면 무차원 또는 비교 가능한 관측량을 찾는다. token per parameter, update-to-weight ratio, warmup token, gradient noise, validation slope를 사용하되 동일하지 않을 수 있음을 검증한다. proxy와 target의 kernel·parallel topology 차이로 effective batch나 numerical path가 달라지면 system discrepancy도 fit 오차에 포함한다.

**elastic resume는 schedule migration이다**

world size가 바뀐 뒤에도 local batch를 그대로 두면 global batch와 update당 token 수가 함께 달라진다. 이를 피하려면 accumulation을 조절해 global batch를 보존해야 한다. 반대로 global batch 변화를 받아들인다면 learning rate와 schedule horizon도 새 기준에 맞춰 옮겨야 한다. 어느 선택도 process만 다시 띄우는 단순 재시작은 아니다. optimizer state, scheduler clock, data cursor와 parallel shard를 한 번에 일관되게 전환해야 한다.

global batch를 보존해도 per-rank sample assignment와 RNG가 달라질 수 있다. exact trajectory 재현과 statistical continuation을 구분한다. exact를 요구하면 global sample order와 packing 결과를 logical batch 단위로 기록하고 새 rank에 재분배한다. statistical continuation이면 중복·누락 허용 범위와 distribution check를 둔다. 둘을 모두 “resume 성공”으로 부르지 않는다.

horizon이 update step으로 정의되었는데 batch가 바뀌면 남은 token budget도 달라진다. 남은 update를 보존할지, 남은 token을 보존할지에 따라 decay 곡선을 다시 계산한다. 현재 lr 연속성을 지키더라도 derivative나 endpoint가 달라질 수 있다. migration record에는 보존한 조건과 포기한 조건을 적고, old curve와 new curve의 첫 다섯 값을 붙인다.

elastic dry-run은 작은 synthetic horizon으로 수행한다. resize 직전 checkpoint, 새 mesh의 state reshard, next data IDs, next learning rates, next parameter deltas를 확인한다. old rank 일부가 checkpoint를 완전히 쓰지 못한 상황도 주입한다. manifest commit이 없는 partial shard를 사용하지 않는다. scaler overflow와 resize가 같은 boundary에 겹치는 경우 event ordering을 고정한다.

운영 dashboard에는 current world size와 local/global batch, accumulation, valid token/update, optimizer committed step, schedule clock, remaining token budget을 함께 둔다. world size만 바뀌고 global batch label이 이전 값을 유지하는 관측 오류를 경계한다. resize event를 RunID의 lineage에 연결해야 전후 quality와 throughput을 해석할 수 있다.

**하나의 update를 여섯 개 시계로 재생한다**

구체적인 사건을 놓고 보자. 네 rank가 각각 길이 2,048인 sequence 두 개를 받고, gradient accumulation을 네 번 수행한다. 겉으로 보이는 configured global batch는 (4\times2\times4=32) sequences다. 그러나 각 sequence에서 prompt와 padding을 제외한 loss token이 서로 다르면 optimization이 실제 평균한 분모는 32가 아니다. 첫 accumulation window의 rank별 valid token이 `(3100, 2980, 3250, 2670)`이라면 global valid-token batch는 12,000이다. 다음 window가 10,500이면 sequence batch는 같아도 gradient estimator의 통계적 무게가 달라진다.

이때 사건 원장을 여섯 열로 나눈다. `microbatch_id`는 forward를 호출할 때 증가한다. `backward_id`는 실제 backward가 끝났을 때 증가한다. `attempted_update`는 accumulation boundary에서 증가한다. `committed_update`는 nonfinite 검사, unscale, clipping과 optimizer step이 모두 성공했을 때만 증가한다. `consumed_token`은 data cursor가 영구 소비한 token을 센다. `scheduled_token`은 lr 곡선이 참조하는 token을 센다. 마지막 둘을 하나로 합칠 수 있는지는 overflow 때 data를 재사용하는 정책에 달렸다.

예를 들어 네 번째 microbatch에서 Inf가 발견되었다고 하자. 데이터를 다시 재생하지 않는 정책이면 consumed token은 12,000만큼 전진하지만 committed update와 optimizer moment는 멈춘다. scheduler가 committed update를 기준으로 하면 lr도 멈춘다. 반대로 attempted update를 기준으로 하면 lr는 줄어들지만 parameter는 움직이지 않는다. 두 정책 모두 구현할 수 있으나 학습 의미는 다르다. 보고서에 “step 812 overflow”라고만 적어서는 어느 시계가 812인지 알 수 없다.

재현 시험은 이 사건을 작은 tensor로 만든다. 정상 window, overflow window, 정상 window 세 개를 실행하고 각 경계에서 parameter checksum, optimizer `step`, scheduler counter, scaler 값, data cursor를 저장한다. uninterrupted run과 overflow 직전 checkpoint에서 재개한 run이 동일한 사건 순서를 내야 한다. 데이터 재사용 정책이라면 다음 sample ID가 같아야 하고, 폐기 정책이라면 건너뛴 ID 집합이 명시적으로 같아야 한다.

Transformers 계열 구현을 읽을 때도 변수 이름보다 증가 지점을 추적한다. 고정 checkout의 `src/transformers/trainer.py`에서 accumulation boundary, optimizer 호출, lr scheduler 호출, global step 증가, checkpoint 저장 순서를 하나의 call card로 만든다. `src/transformers/optimization.py`에서는 scheduler factory가 어떤 LambdaLR 함수를 만들고 `num_warmup_steps`, `num_training_steps`, `last_epoch`를 어떻게 넘기는지 본다. 고정 revision과 함수 이름, test 이름을 함께 적어야 이후 upstream 변경과 책의 설명을 구분할 수 있다.

**합의된 시계 스키마**

시계 스키마에는 이름, 단위, owner, 증가 전제, rollback 가능성, serialization 위치가 필요하다. `committed_update`의 owner가 모든 rank라면 collective 뒤 동일 값임을 assert한다. rank 0만 owner라면 broadcast 시점을 기록한다. token counter는 local 합계를 all-reduce한 뒤 commit해야 한다. 한 rank가 padding-only batch를 받았을 때 local 0도 collective에 참여해야 한다.

checkpoint에는 counter 값만 넣지 않는다. 마지막 적용 lr, 다음 적용 예정 lr, scheduler class와 인자, horizon의 단위, data cursor commit ID를 넣는다. load 직후 scheduler 객체가 내부적으로 한 번 advance되는 framework라면 saved counter를 그대로 대입해서는 off-by-one이 생길 수 있다. 작은 boundary oracle이 이 차이를 잡는다.

시간 변환식도 versioned artifact다. `tokens_per_update = sum(valid_loss_tokens)`는 매 update마다 달라질 수 있다. 고정 평균을 곱해 update clock을 token clock으로 바꾸면 curriculum, packing, sequence-length warmup에서 drift한다. 이미 관측한 구간은 실제 prefix sum을 쓰고 미래 horizon만 추정치와 신뢰 구간을 쓴다.

**수치 fixture: 12개 사건**

fixture의 첫 세 microbatch는 각각 2,900, 3,100, 2,700 valid token을 낸다. 네 번째는 3,300을 내고 update가 commit된다. 첫 lr가 base의 1/4이면 적용 lr, gradient norm, clipped norm, parameter delta를 저장한다. 다음 네 microbatch에서는 세 번째 backward가 overflow한다. 구현이 accumulation buffer를 즉시 clear하는지, 남은 microbatch를 실행하는지, collective nonfinite 판정을 언제 하는지 확인한다.

세 번째 window는 checkpoint 재개 뒤 실행한다. loader가 첫 두 microbatch를 중복 공급하는 오류를 의도적으로 주입하고 sample-ID ledger가 이를 검출하는지 본다. 최종 loss가 비슷하다는 이유로 통과시키지 않는다. 사건 수열과 다음 열두 sample ID, 세 counter, lr, optimizer moment checksum이 기대값과 일치해야 한다.

이 fixture는 scheduler unit test이면서 trainer integration test다. Lambda 식만 시험하면 호출 순서 오류를 못 잡고, 전체 모델 loss만 보면 최초 divergence를 못 찾는다. optimizer를 scalar parameter 하나로 줄이면 기대 delta를 손으로 계산할 수 있다. 그 위에 실제 mixed precision과 distributed collective를 한 층씩 추가한다.

**warmup·decay를 제어 시스템으로 설계한다**

학습률 곡선은 시간에 따른 scalar 장식이 아니다. gradient라는 noisy observation을 parameter 변화로 바꾸는 폐루프의 gain이다. optimizer moment, clipping, weight decay, loss scaling이 모두 effective gain을 바꾼다. 따라서 nominal lr만 비교하면 같은 제어 입력이라고 착각한다. layer별 update-to-weight ratio와 실제 parameter delta를 함께 보아야 한다.

linear warmup의 단순식은 (\eta_t=\eta_{max}t/W)다. 하지만 (t=0)에서 첫 optimizer update가 0인지, scheduler를 먼저 advance해 (\eta_{max}/W)인지가 핵심이다. warmup 끝의 포함 관계도 구현마다 다르다. `t<W`와 `t\le W`는 한 step 차이지만 작은 run과 resume boundary에서는 분명한 차이를 만든다. 기대 수열을 `[0.0, .25, .5, .75, 1.0]`처럼 적고 “로그된 값”과 “적용된 값”을 구분한다.

cosine decay는 흔히 (\eta(t)=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos(\pi p)))로 쓴다. 여기서 (p)의 분모가 전체 training step인지 warmup을 뺀 decay step인지 확인한다. endpoint를 포함하는지도 본다. restart가 있다면 cycle index와 현재 cycle 길이도 state다. WSD는 warmup-stable-decay 세 구간의 길이뿐 아니라 stable 구간에서 어떤 clock을 소비하는지가 중요하다.

곡선의 면적은 누적 nominal lr의 근사다. horizon을 바꿀 때 현재 lr의 연속성만 보존하면 남은 면적이 크게 달라질 수 있다. 반대로 남은 면적을 보존하면 경계에서 lr jump가 생길 수 있다. migration에서는 연속 값, 연속 기울기, endpoint, 남은 면적 가운데 무엇을 우선했는지 적는다. 이것이 schedule 변경을 단순 config 수정이 아니라 수학적 계약 변경으로 보는 이유다.

**multi-clock warmup의 선택**

update warmup은 optimizer state가 몇 번 갱신되었는지를 보존한다. token warmup은 데이터 노출량을 보존한다. sample warmup은 예제 수를 보존하지만 길이가 다른 데이터에서는 token 노출을 보존하지 못한다. wall-clock warmup은 시간 예산을 지키지만 compiler warmup이나 cluster 혼잡에 영향을 받는다. elastic 환경에서 world size가 바뀔 가능성이 크면 token clock이 더 안정적일 수 있으나, token/update가 급변하면 optimizer moment의 성숙도가 달라진다.

한 가지 해법은 두 조건을 함께 기록하는 것이다. lr는 committed update로 증가시키되, warmup 종료는 최소 update와 최소 token을 모두 만족할 때로 정의할 수 있다. 그러나 이 정책은 standard scheduler와 다르므로 명시적인 state machine과 test가 필요하다. `WARMING`, `STABLE`, `DECAYING`, `DONE` 상태와 전이 guard를 적고 checkpoint에 상태와 counter를 함께 저장한다.

curriculum이 sequence length를 2K에서 8K로 올리는 경계에서는 token/update가 네 배 가까이 변할 수 있다. update 기반 cosine은 같은 속도로 흐르지만 token 관점 decay는 갑자기 빨라진다. curriculum 변경과 schedule 변경을 같은 event log에 두고 경계 전후의 valid token, gradient noise, update norm을 비교한다. 필요하다면 새 segment로 migration하되 retrospective하게 과거 counter를 다시 해석하지 않는다.

**optimizer와 schedule의 결합**

Adam의 bias correction은 초기 step에서 moment의 편향을 보정한다. warmup이 그와 겹치면 effective update는 nominal lr 곡선만으로 설명되지 않는다. beta를 바꾸면서 warmup을 유지하는 실험은 두 축을 동시에 바꾼다. Muon의 orthogonalization이나 matrix update가 일정 cadence로 일어나면 scheduler step과 algorithm-state step의 대응도 기록한다.

decoupled weight decay의 한 step은 대략 (w\leftarrow(1-\eta_t\lambda)w-\eta_tu_t)다. lr schedule은 data gradient뿐 아니라 decay의 누적 세기도 바꾼다. 서로 다른 cosine horizon을 비교하면서 같은 weight decay coefficient를 사용해도 누적 shrinkage는 같지 않다. ​`sum(lr)`와 parameter-group별 `lambda`를 함께 원장에 넣고, no-decay group의 membership 변화도 checkpoint manifest에 기록한다.

clipping은 nonlinear gate다. batch를 키워 gradient distribution이 달라지면 clipping frequency가 변하고, nominal lr scaling이 실제 update scaling으로 이어지지 않을 수 있다. 각 실험에서 unclipped norm, clip coefficient, clipped update norm을 본다. clipping 임곗값을 batch와 함께 바꾸면 별 ablation이 필요하다.

## 13.5 critical batch와 scaling theory를 측정한다

scaling rule은 하드웨어 크기나 관행적 배수에서 나오지 않는다. microbatch gradient의 신호·분산, 유효 label token, update norm과 validation 효율을 같은 예산 축에서 측정하고, 독자가 직접 반례를 만들 수 있는 수치 실험으로 임계 영역을 좁힌다.

### 13.5.1 critical batch를 측정하는 실험의 해부

critical batch는 config 표에 적힌 영구 상수가 아니다. 특정 checkpoint, data distribution, objective, optimizer에서 병렬 sample을 더 추가할 때 얻는 marginal progress가 줄어드는 영역을 요약한 값이다. pretraining 초기와 후반, 쉬운 corpus와 어려운 corpus, dense model과 MoE에서 달라질 수 있다. 한 번 측정한 숫자를 전체 run에 적용하지 않는다.

가장 먼저 비용 축을 고정한다. fixed-token 실험에서는 모든 후보가 같은 valid prediction token을 소비하고 validation loss 개선을 비교한다. 큰 batch는 update 수가 적다. fixed-update 실험에서는 update 수를 같게 하므로 큰 batch가 더 많은 compute를 쓴다. fixed-wall-time 실험에서는 hardware utilization과 통신이 결과에 들어온다. 이 세 실험은 서로 다른 질문에 답한다.

동일 checkpoint에서 (M)개 microbatch gradient (g_i)를 얻으면 평균 gradient와 sample 간 변동을 추정할 수 있다. exact per-example gradient가 너무 비싸면 여러 독립 microbatch를 사용한다. layer별 (\|E[g]\|^2)와 (E[\|g-E[g]\|^2])를 구하고 estimator의 sample 수와 confidence interval을 남긴다. gradient clipping 전 값을 사용해야 clipping에 의해 noise가 숨지 않는다.

후보 batch는 local microbatch, accumulation, data-parallel degree의 조합으로 만든다. global batch가 같더라도 통신 빈도와 activation memory, dropout RNG, normalization이 달라질 수 있으므로 같은 통계 실험이라고 자동 간주하지 않는다. packing seed와 loss denominator를 고정하고 realized valid-token batch 분포를 보고한다.

**scaling rule의 반증 조건**

linear lr scaling 가설은 batch를 (k)배 했을 때 lr도 (k)배 한다. 반증 조건은 초기 loss spike, clip frequency 급증, update-to-weight ratio의 비례 이탈, 같은 token budget에서 validation 열화다. square-root scaling도 같은 방식으로 시험한다. 둘 중 어느 것도 이론적 보증으로 취급하지 않는다.

작은 batch가 hardware를 충분히 채우지 못하면 throughput 차이가 통계적 효율 차이처럼 보인다. samples/sec, tokens/sec, model FLOP utilization, collective wait를 함께 기록한다. time-to-quality는 시스템 효율과 sample efficiency를 결합하지만, 원인을 분리한 표도 유지한다.

critical batch가 training phase에 따라 변한다면 고정 global batch 대신 batch ramp를 고려할 수 있다. 이때 batch 변화는 scheduler clock과 warmup, optimizer state에 영향을 주므로 migration event다. batch를 늘려 update당 token이 늘면 token horizon을 보존하도록 남은 update 수를 줄일지 결정한다. lr와 batch를 동시에 바꾸면 최소한 작은 factorial experiment로 상호작용을 확인한다.

**scaling-law fit과 연결**

scaling law의 loss fit은 장기 예산을 선택하고, noise-scale 실험은 한 시점의 update efficiency를 본다. 둘을 같은 법칙으로 혼동하지 않는다. model size와 data token을 바꾼 여러 run의 envelope에서 compute-optimal 지점을 찾는 일과, 주어진 run의 global batch를 정하는 일은 다른 층이다.

fit dataset에는 각 run의 code revision, tokenizer, unique/seen token, repeat count, architecture, active/total parameter, optimizer recipe, precision, achieved compute를 넣는다. 실패하거나 조기 종료한 run을 제거하면 survivorship bias가 생긴다. divergence도 feasibility boundary 정보로 보존한다.

외삽 전에는 leave-one-scale-out 검증을 한다. 가장 큰 관측 scale을 숨기고 작은 점으로 예측한 뒤 오차를 본다. residual이 model family나 data mixture에 따라 한쪽으로 몰리면 단일 exponent가 부적절하다. confidence interval이 budget 선택을 바꿀 정도로 넓으면 추가 probe run이 비용 효율적일 수 있다.

**elastic resize의 원자적 절차**

resize는 먼저 old world에서 quiesce barrier를 만든다. 새 microbatch를 발급하지 않고 진행 중 backward와 collective를 끝낸다. optimizer update가 commit되었는지 확인하고 counter, optimizer/scaler/scheduler state, global sample cursor를 하나의 checkpoint generation에 쓴다. 모든 shard의 checksum과 expected rank set이 맞아야 manifest를 publish한다.

new world는 manifest를 읽어 model과 optimizer state를 reshard한다. data loader는 logical global sample stream을 새 rank에 배정한다. local batch와 accumulation을 계산하고 새 realized global batch를 검산한다. scheduler migration 함수가 old state, new batch, remaining token budget을 입력받아 new horizon과 다음 lr 수열을 낸다. 이 결과를 resize certificate에 저장한다.

dry run에서는 optimizer update를 commit하지 않고 다음 logical batch의 ID, tensor shape, collective group, 예상 lr를 산출한다. 모든 rank 결과를 모아 중복·누락과 group membership을 검사한다. 성공 뒤에만 new generation을 RUNNING으로 바꾼다. 실패하면 old manifest는 그대로 유효하고 partial new shard는 publish되지 않는다.

**exact와 statistical resume**

exact resume는 같은 sample packing, augmentation RNG, dropout RNG, reduction order까지 요구할 수 있다. world-size 변경에서는 reduction tree와 floating-point 순서가 달라져 bitwise 동일성이 불가능할 수 있다. 요구 수준을 “data exact”, “state exact”, “bitwise exact”, “statistically continuous”로 분리한다.

data exact는 global sample ID와 loss mask를 보존한다. state exact는 logical full optimizer/model state를 보존하되 shard layout 변화를 허용한다. statistical continuation은 validation probe, gradient distribution과 update norm이 허용 범위에 있음을 요구한다. 단순히 process가 다시 실행된 것을 resume 성공이라고 하지 않는다.

RNG도 owner가 있다. data shuffle RNG는 global stream owner, dropout RNG는 parallel coordinate owner, augmentation RNG는 sample owner로 설계할 수 있다. mesh 좌표가 바뀔 때 seed 파생식을 versioning한다. checkpoint에는 base seed 하나만이 아니라 generator state와 소비 offset 또는 재구성 규칙을 둔다.

**장애 주입표**

checkpoint shard 한 개 누락, manifest write 직전 rank 사망, optimizer commit 뒤 data cursor commit 전 사망, overflow와 resize 동시 발생, scheduler state만 이전 generation인 경우를 주입한다. 각 경우 load가 fail closed하는지 확인한다. “가장 최신 파일”을 추측해 섞어 읽으면 안 된다.

network partition으로 일부 rank만 quiesce barrier를 통과한 경우 timeout 뒤 job generation을 폐기한다. 새 rendezvous가 이전 rank의 늦은 collective와 섞이지 않도록 run epoch 또는 rendezvous generation을 communicator 이름에 넣는다. observability에는 old/new world와 generation을 모두 표시한다.

복구 뒤 첫 다섯 update는 고정 probe로 비교한다. next data ID, valid token, applied lr, gradient norm, clip coefficient, update norm, parameter checksum을 uninterrupted reference와 맞춘다. bitwise 요구가 아니라면 tolerance와 통계 검정을 사전에 정한다.

**scheduler observability는 원인 그래프여야 한다**

한 줄의 lr plot만으로는 부족하다. 같은 timestamp에 attempted/committed update, consumed/scheduled valid token, local/global batch, accumulation, world size, scaler overflow, gradient norm, clip coefficient, optimizer update norm, data-mixture ID를 조인할 수 있어야 한다. dashboard는 이 필드의 label cardinality를 제어하되 원본 event log를 RunID로 조회할 수 있게 한다.

첫 경보는 불변식 위반이다. committed update가 늘지 않았는데 schedule counter가 증가하면 즉시 경보한다. global valid token이 rank 합과 다르거나, world size 변경 뒤 configured global batch label이 갱신되지 않아도 경보한다. lr가 기대 oracle과 tolerance 밖이면 최종 loss를 기다리지 않는다.

두 번째 경보는 추세다. overflow rate, clipping frequency, gradient noise proxy, update-to-weight ratio, validation slope를 구간별로 본다. lr 변화와 같은 시간축에 놓되 인과라고 단정하지 않는다. data mixture, sequence length, precision recipe와 topology change event를 annotation한다.

**incident를 최초 불일치로 줄인다**

loss spike가 보이면 spike 시점부터 거꾸로 보지 말고 정상 reference와 event stream을 diff한다. 최초로 달라진 sample IDs, token count, applied lr, scaler decision, optimizer checksum을 찾는다. lr log가 같아도 parameter-group multiplier나 optimizer step skip이 다를 수 있다.

두 run의 설정 파일 diff만으로 끝내지 않는다. resolved config, runtime-derived horizon, scheduler state dict, 실제 적용 수열을 비교한다. CLI 기본값과 environment override, auto batch 계산이 설정 파일 밖에서 값을 바꿀 수 있다. 값의 provenance를 `user`, `default`, `derived`, `restored`, `migrated`로 표시한다.

RCA에는 탐지 신호, 최초 불일치 event, causal chain, escape reason, corrective test를 넣는다. “resume bug 수정”이 아니라 경계 fixture와 expected oracle을 추가한다. 같은 유형의 오류가 framework upgrade에서 재발하지 않게 source revision과 test coordinate를 handoff한다.

**고정 소스에서 옵션 하나를 끝까지 따라간다**

`warmup_ratio`를 예로 들면 CLI 입력에서 끝내지 않는다. argument parser가 값을 읽고, total training steps가 계산되며, ratio가 integer warmup steps로 반올림되고, scheduler factory에 전달되고, Lambda 함수가 lr multiplier를 반환하고, optimizer param group의 lr가 갱신되는 경로를 잇는다. 중간마다 자료형, rounding, boundary를 기록한다.

고정 checkout `sources/transformers-v5.15.1`을 사용할 때 revision을 먼저 기록하고 `src/transformers/trainer.py`, `src/transformers/training_args.py`, `src/transformers/optimization.py`의 실제 symbol을 `rg`로 확인한다. 문서의 설명이 checkout의 함수명과 다르면 추측해 쓰지 않는다. 관련 `tests/trainer`와 scheduler test에서 first/last step 기대값, resume test가 무엇을 검증하는지 읽는다.

source card는 `revision:path:symbol`과 시작/끝 줄, 입력, 출력, mutable state, caller, test를 가진다. line number만 두면 upstream rebasing에 약하고 symbol만 두면 동일 이름이 여러 class에 있을 수 있다. content hash 또는 짧은 semantic excerpt도 붙인다. 책에는 이해에 필요한 짧은 부분만 인용하고, 원장에는 고정 checkout 좌표를 둔다.

**구현에서 확인할 질문**

total steps는 dataloader 길이, max steps, epochs 중 무엇이 우선하는가. iterable dataset에서는 길이를 어떻게 얻는가. gradient accumulation을 나눌 때 ceil과 floor 중 무엇을 쓰는가. distributed world size가 계산에 들어가는 시점은 언제인가. resume에서 scheduler state를 load한 뒤 다시 초기화하는 경로가 있는가.

optimizer step이 skipped되면 scheduler도 skip되는가. mixed-precision scaler의 결과를 trainer가 어떤 flag로 받는가. 여러 optimizer 또는 parameter group이 있을 때 scheduler 객체가 하나인지 여러 개인지 본다. evaluation-only step이나 gradient accumulation partial tail이 counter에 포함되는지도 확인한다.

이 질문을 code review checklist로 바꾸고 작은 fake optimizer로 test한다. framework 이름을 신뢰하지 않고 실제 상태 변화를 관찰한다. 버전을 올릴 때 같은 fixture를 실행하면 의미 변화가 드러난다.

**통합 인수 시험의 기준선을 세운다**

인수 시험은 24 update의 작은 run이다. 처음 여섯 update는 warmup, 다음 열둘은 stable 또는 cosine, 마지막 여섯은 decay다. 세 번째 update에서 overflow를 주입하고, warmup 끝에서 checkpoint/restart하며, 열두 번째 update에서 world size와 accumulation을 바꾼다. sequence length curriculum도 같은 경계에서 바꾸지 않고 별 event로 둔다.

reference는 모든 event를 단일 process scalar model로 계산한다. distributed candidate는 같은 logical batch와 policy를 사용한다. 비교 항목은 sample/token ledger, attempted/committed counter, applied lr 수열, optimizer state, parameter delta다. scaling rule 실험은 이 정확성 시험을 통과한 뒤 별도로 수행한다.

성공 조건은 모호한 “curve가 비슷함”이 아니다. clock invariant 위반 0, 중복·누락 sample 0, boundary lr oracle 일치, checkpoint generation atomicity, resize certificate 통과, 예상한 tolerance의 parameter trajectory다. 실패하면 최초 불일치 event와 source call card를 report에 자동 첨부한다.

운영 handoff에는 resolved recipe, clock schema, schedule 식과 boundary 표, scaling-law fit 범위, critical-batch 실험, elastic policy, dashboard query, failure-injection 결과가 들어간다. 다음 담당자는 이름이 같은 cosine을 재구성하는 것이 아니라 이 bundle로 정확히 같은 사건 수열을 재생할 수 있어야 한다.

이 장의 결론은 단순하다. schedule은 lr 숫자의 목록이 아니라 데이터, optimizer, precision, topology를 시간 위에서 합의시키는 분산 상태 기계다. scaling law는 그 상태 기계의 예산을 제안하지만 실행 의미를 대신하지 않는다. critical batch는 병렬 효율의 조건부 관측값이며, elastic resume는 여러 시계의 원자적 migration이다. 이 네 층을 한 원장으로 닫을 때 비로소 “왜 이 learning rate인가”에 기술적으로 답할 수 있다.

**scaling experiment를 실제 의사결정으로 바꾼다**

예산 회의에서 필요한 것은 exponent 하나가 아니다. 후보 model size, unique data, 반복 횟수, 예상 training FLOP, cluster throughput, 실패 여유, validation 목표를 같은 표에 놓는다. 각 값에는 point estimate와 범위를 둔다. theoretical FLOP를 GPU peak로 나눠 wall time을 내면 통신, checkpoint, data stall, 재시작을 무시한다. 과거 run에서 얻은 achieved tokens/sec와 availability를 사용하고, 아직 관측하지 않은 scale에는 보수적 범위를 둔다.

세 후보를 생각하자. A는 작은 모델에 많은 token, B는 중간 모델과 균형 token, C는 큰 모델에 적은 token이다. scaling fit이 B의 validation loss를 가장 좋게 예측해도 제품 latency나 inference memory가 C를 탈락시킬 수 있고, 고품질 unique token 부족이 A를 반복 학습으로 바꿀 수 있다. compute-optimal이라는 형용사는 제약 조건과 목적 함수를 적지 않으면 의미가 없다.

각 후보에는 pilot ladder를 둔다. 전체 budget의 작은 비율로 여러 scale을 실행하고, tokenizer·data mixture·optimizer recipe를 가능한 한 고정한다. 단, 작은 모델에서 최적인 lr와 batch를 큰 모델에 그대로 쓰지 않는다. 각 scale에서 최소한 안정성 tuning을 하고 그 비용도 fit에 포함한다. 실패한 점을 숨기면 대형 run의 안정성 경계를 과대평가한다.

pilot 종료 뒤 posterior decision을 갱신한다. 예측 loss뿐 아니라 residual, seed variance, throughput, memory headroom, overflow와 retry rate를 본다. fit이 예상보다 불안정하면 한 점을 더 찍는 비용과 바로 대형 run으로 가는 위험을 비교한다. 이 과정은 논문 그래프를 흉내 내는 일이 아니라 제한된 compute에서 정보 가치를 최대화하는 실험 설계다.

**데이터 반복을 별 축으로 둔다**

seen token과 unique token은 다르다. 1조 token을 학습했다고 해도 고품질 2천억 token을 다섯 번 반복했다면 독립 데이터 1조 token과 같은 축에 놓을 수 없다. sample별 exposure count, epoch distribution, dedup cluster의 반복을 기록한다. curriculum이나 temperature sampling 때문에 corpus별 반복 횟수도 다르다.

반복이 유효한 정도는 모델 크기와 regularization, data quality에 따라 달라질 수 있다. training loss가 계속 줄어도 held-out contamination-free loss와 memorization probe가 악화될 수 있다. scaling-law dataset에는 repeat regime를 feature로 넣거나 별 fit을 둔다. unique token 부족을 exponent로 감추지 않는다.

데이터 품질이 바뀌는 경우 raw token 수만으로 compute allocation을 정하지 않는다. quality filter가 aggressive하면 token은 줄지만 유효 정보 밀도가 늘 수 있다. 그러나 이 효과를 임의의 “effective token multiplier”로 선언하지 말고 matched-budget experiment로 측정한다. tokenizer 변경도 token count 단위를 바꾸므로 byte 또는 character 기반 보조 축을 둔다.

**모델 크기 회계**

dense 모델에서는 embedding, attention, MLP, norm과 output head를 포함한 trainable parameter를 구분한다. tied embedding이면 물리적 parameter를 두 번 세지 않는다. MoE에서는 total parameter와 token당 active parameter를 모두 보고한다. training memory와 checkpoint 크기는 total에 가깝고, forward FLOP는 active에 가깝지만 routing·communication 비용이 추가된다.

adapter나 frozen block이 있으면 total, trainable, updated parameter를 분리한다. optimizer state 비용은 updated parameter와 dtype에 묶인다. scaling fit의 x축이 논문의 어떤 convention인지 확인하지 않고 서로 다른 연구의 숫자를 한 그래프에 합치지 않는다.

FLOP 회계도 forward-only 추정과 forward+backward 실제를 나눈다. activation checkpointing은 연산을 재실행하고, MoE imbalance와 padding은 유효 token당 FLOP를 바꾼다. profiler 측정과 analytical estimate가 다르면 둘 다 남기고 차이를 설명한다.

**scheduler 구현을 property로 시험한다**

example 몇 개만 맞추는 test는 horizon이나 resume 조합을 다 덮지 못한다. schedule 함수에는 property를 정의할 수 있다. warmup 구간은 허용한 정책에 따라 단조 증가한다. cosine decay는 경계 안에서 최소·최대 범위를 넘지 않는다. WSD segment는 정의한 접합 조건을 만족한다. committed clock이 멈추면 lr도 멈춘다.

horizon 1, warmup 0, warmup과 horizon이 같은 경우, warmup이 horizon보다 큰 invalid 입력, 음수 last step, resume가 endpoint인 경우를 생성한다. rounding 때문에 ratio가 0 step이 되는 작은 horizon도 본다. invalid 조합은 조용히 이상한 곡선을 만들지 말고 명시적으로 거부하거나 문서화된 clamp를 적용한다.

stateful scheduler에는 serialization property가 필요하다. 임의의 step (k)에서 state를 저장하고 새 객체에 load한 뒤 남은 수열이 uninterrupted 수열과 같아야 한다. optimizer param-group이 여러 개면 base lr와 multiplier가 보존되어야 한다. group 순서가 바뀌는 checkpoint migration은 이름 또는 stable ID로 매핑하고 모호하면 실패한다.

**metamorphic test**

update 기반 schedule에서 accumulation만 두 배 하고 global batch를 유지하면, 같은 committed update index의 lr는 같아야 한다. token 기반 schedule에서 packing이 바뀌어 valid token prefix sum이 달라지면 lr도 정의한 식대로 달라져야 한다. 이 관계를 metamorphic test로 만든다.

base lr를 상수 (c)배 하면 모든 group의 nominal lr가 (c)배 되는지 검사한다. 단, min lr가 absolute value인지 ratio인지에 따라 property가 달라진다. 구현의 옵션 의미를 test가 명시하게 한다. horizon과 warmup을 같은 배수로 늘리고 정규화 progress가 같은 점을 비교하는 property도 유용하다.

elastic migration은 old/new curve의 보존 조건을 property로 둔다. current value 연속성을 선택했다면 경계 양쪽 차이가 tolerance 이하여야 한다. remaining token과 endpoint 보존을 선택했다면 새 curve가 정확히 남은 budget에서 끝나야 한다. 여러 조건이 충돌하면 우선순위를 test 이름에 드러낸다.

**호출 순서 test**

fake optimizer는 `step_called`, `skipped`, `param_delta`를 기록한다. scheduler는 호출될 때 optimizer의 commit generation을 읽는다. 잘못된 선행 scheduler 호출, overflow skip 뒤 호출, 두 번 호출을 각각 주입한다. expected exception 또는 invariant violation을 확인한다.

gradient accumulation tail이 완전한 window보다 작을 때 update를 수행하는지 폐기하는지 정책을 test한다. 수행한다면 denominator와 effective global batch가 달라진다. 마지막 update의 lr와 data cursor가 horizon 안에 포함되는지도 검산한다.

evaluation과 checkpoint callback이 scheduler를 건드리지 않는지 확인한다. 일부 custom callback은 metric 기반 scheduler를 advance할 수 있다. step 기반 scheduler와 plateau scheduler의 owner가 겹치면 이중 갱신이 생긴다. 하나의 parameter group lr에는 하나의 최종 writer만 두거나 명시적인 합성 규칙을 둔다.

**metric 기반 schedule과 training clock**

지금까지의 cosine과 WSD는 미리 정한 progress를 입력으로 받는다. plateau scheduler는 validation metric이라는 지연된 noisy signal을 입력으로 받는다. metric이 어느 checkpoint의 parameter에서 계산되었는지, distributed evaluation이 완료된 시점, smoothing과 patience state가 모두 scheduler state가 된다.

비동기 evaluation에서는 training이 여러 update 앞서갈 수 있다. 오래된 metric이 도착했을 때 현재 lr를 줄이면 원인과 반응 사이에 지연이 생긴다. metric event에 model checkpoint ID를 붙이고 최대 허용 staleness를 정한다. 너무 오래된 결과는 기록만 하고 control input으로 쓰지 않는다.

metric direction, threshold mode, cooldown, patience, best value와 bad-count를 checkpoint한다. resume 뒤 best가 초기화되면 lr 감소가 늦어진다. 여러 validation suite 중 어떤 aggregate가 control owner인지 고정한다. benchmark noise 때문에 임계값 주변에서 oscillation하지 않도록 최소 효과 크기와 반복 확인을 둔다.

**online control의 안전장치**

자동 schedule 변경에는 rate limit과 bounds가 필요하다. 한 metric anomaly가 lr를 연속으로 낮추지 않게 event generation을 deduplicate한다. 최소 lr, 최대 변경 비율, cooldown을 설정하고 사람이 override한 경우 provenance를 기록한다. override 뒤 scheduler internal state와 실제 lr가 불일치하지 않게 reconciliation한다.

control dashboard는 metric observation time, checkpoint step, action time, 적용 update를 잇는다. 단순히 validation loss와 lr 두 선을 같은 그래프에 놓는 것으로는 지연을 알 수 없다. action이 품질을 개선했다는 판단에는 counterfactual 또는 적어도 matched historical control이 필요하다.

대형 pretraining에서는 예산 안정성이 중요하므로 복잡한 online controller보다 사전 정의 schedule이 선호될 수 있다. 반대로 fine-tuning이나 불안정한 preference optimization에서는 adaptive policy가 유용할 수 있다. 어느 경우든 schedule 선택 이유를 workload의 signal-to-noise와 intervention cost로 설명한다.

**분산 token 회계의 함정**

각 rank가 local valid token을 세어 all-reduce sum하면 간단해 보인다. 그러나 pipeline parallel에서는 같은 logical token이 여러 stage를 통과하므로 모든 rank를 합하면 중복된다. tensor parallel도 같은 batch를 복제한다. token counter collective는 data-parallel replica group의 대표 rank에서만 값을 모으거나 mesh 좌표 조건으로 owner를 정한다.

expert parallel에서는 token이 expert 사이로 이동하지만 새 training token이 생긴 것이 아니다. dropped token이나 capacity overflow가 loss에 기여하지 않는다면 optimization denominator와 input throughput counter가 달라진다. context parallel에서 sequence shard별 mask를 합쳐야 하며 causal padding과 packed segment boundary를 반영한다.

멀티모달 batch는 text token, image patch token, audio frame token의 compute cost가 다르다. 하나의 token clock으로 schedule을 움직이려면 어떤 단위를 쓰는지 정의한다. loss-bearing target token을 optimization clock으로 쓰고 modality별 compute unit을 throughput 회계로 별도 둘 수 있다. 단순 total token은 curriculum 변화에서 비용과 학습 신호를 모두 왜곡할 수 있다.

**denominator와 counter를 같은 코드에서 얻는다**

가장 안전한 설계는 loss reduction에 실제 사용한 numerator·denominator를 counter 입력으로 재사용하는 것이다. data loader가 추정한 sequence length를 별도로 세면 label masking과 truncation이 누락된다. backward 전에 local loss sum과 valid count를 만들고, 올바른 replica group에서 global count를 합친다.

gradient accumulation 동안 각 microbatch loss를 local 평균한 뒤 다시 평균하면 token 수가 다른 microbatch에 같은 무게를 준다. 대신 loss sum을 누적하고 전체 valid count로 나누거나, backward scale을 global denominator에 맞게 조절한다. scheduler token clock도 같은 denominator prefix sum을 사용해야 한다.

zero-valid-token microbatch는 division-by-zero와 collective mismatch를 일으킬 수 있다. 모든 rank가 동일 control flow로 collective에 참여하고 global count가 0이면 update를 명시적으로 skip한다. attempted·committed counter와 data policy를 기록한다. 조용히 NaN을 scaler overflow로 처리하면 데이터 파이프라인 문제를 precision 문제로 오진한다.

**운영 변경을 검증 가능한 RFC로 만든다**

RFC 첫 문장은 바꿀 knob가 아니라 해결할 증상을 쓴다. 예컨대 “8K curriculum 전환 뒤 update-to-weight ratio가 두 배가 되고 validation spike가 반복된다.” 다음에는 clock ledger와 원인 가설을 붙인다. token/update 증가로 token-clock decay가 빨라졌는지, gradient variance가 달라졌는지, loss normalization 오류인지 분리한다.

변경안은 old/new schedule 수열, batch와 denominator, optimizer/scaler 상태, 예상 parameter update를 표로 낸다. warmup을 늘린다는 자연어가 아니라 어느 clock에서 몇 단위, 현재 curve와 어떻게 접합하는지 적는다. rollback은 이전 checkpoint와 recipe generation, data replay 정책을 포함한다.

canary는 전체 cluster의 작은 rank subset이 아니라 독립된 짧은 run일 수 있다. distributed group 일부만 다른 lr를 쓰면 collective gradient의 의미가 깨진다. 같은 checkpoint와 고정 probe data에서 candidate recipe를 별 run으로 비교한다. pass criteria는 안정성, 품질 slope, throughput과 reproducibility를 포함한다.

**change window와 lineage**

승인된 변경은 새 RecipeID를 만든다. 동일 RunID 안에서 migration했다면 parent recipe, event step, old/new state hash를 연결한다. 실험 추적 시스템에서 config 값만 덮어쓰지 않는다. checkpoint도 어느 recipe generation에서 만들어졌는지 포함한다.

rollback 후에는 old curve를 단순 재개할 수 있는지 계산한다. candidate 동안 소비한 data와 변경된 optimizer state를 버리고 이전 checkpoint로 돌아가면 compute와 data lineage가 갈라진다. candidate state에서 lr만 되돌리는 것은 동일 rollback이 아니다. 선택한 의미를 기록한다.

변경 종료 뒤에는 예상과 실제를 비교한다. lr 수열과 counters가 설계대로였는지 먼저 보고, 그 다음 quality 효과를 본다. 구현 실패와 가설 실패를 분리한다. 가설이 틀렸더라도 정확히 실행된 negative result는 scaling 지식에 추가한다.

### 13.5.2 작은 수치로 재현하는 scaling 해부

첫 과제는 현재 사용하는 framework에서 scheduler가 실제로 호출되는 한 경로를 고르는 것이다. parser의 옵션에서 trainer loop, optimizer step, state serialization, test까지 호출 그래프를 그린다. 고정 revision과 좌표를 적고 첫 열두 applied lr를 출력이 아니라 독립 oracle로 계산한다.

variable-length synthetic data 과제에서는 sequence batch와 valid-token batch가 다른 run을 만든다. accumulation을 바꾸면서 global valid-token distribution을 맞춘 경우와 맞추지 않은 경우를 비교하고, loss denominator, gradient norm, parameter delta와 clock을 기록한다.

다음 과제에서는 overflow, checkpoint와 resize를 순서대로 주입한다. 각 사건 뒤의 next sample IDs와 next lr를 예측해 실제와 비교한다. partial manifest와 stale scheduler state를 load하려 할 때는 fail closed해야 하며, 성공한 실행뿐 아니라 의도한 실패가 정확히 탐지되는지도 증명한다.

마지막 과제에서는 batch 후보 세 개를 fixed-token으로 비교한다. seed 반복, confidence interval, system/statistical efficiency 분리, clip frequency와 noise proxy를 포함하되, linear scaling과 square-root scaling 중 하나를 “정답”으로 고르지 않고 각 가설의 반증 증거를 보고한다.

마지막 과제는 scaling-law pilot table을 만든다. parameter convention, unique/seen token, compute, recipe revision, achieved throughput과 실패 run을 포함한다. 한 scale을 holdout해 예측 오차를 측정한다. 이 표에서 대형 run 후보를 고르고 불확실성이 결정을 어떻게 바꾸는지 설명한다.

제출물은 코드 한 조각이 아니다. source call card, clock schema, boundary oracle, event ledger, scaling experiment card, resize certificate, incident decision tree와 handoff manifest다. 다른 사람이 동일 checkout에서 자료를 읽고 같은 결론을 재현할 수 있어야 한다. 이 수준에 이르러야 scheduler와 scaling을 “설정값 설명”이 아니라 실제 학습 시스템의 시간·예산·복구 설계로 이해했다고 말할 수 있다.

**사례 연구: 2K에서 8K로 길이를 올리는 날**

운영팀이 2K sequence curriculum을 끝내고 8K로 전환한다고 하자. model, optimizer와 nominal global sequence batch를 그대로 두면 update당 최대 token은 네 배가 된다. 실제 valid token은 packing과 문서 길이 때문에 정확히 네 배가 아니지만 큰 점프가 생긴다. update 기반 schedule은 이 변화를 보지 못하고 같은 속도로 decay한다. token 기반 schedule은 한 update에서 네 배 가까이 전진한다. 어느 쪽이 맞는지는 보존하려는 학습 가설에 달렸다.

전환 전 마지막 백 update의 realized valid-token batch 분포, gradient norm과 variance proxy, clip coefficient, update-to-weight ratio를 baseline으로 저장한다. 8K candidate는 같은 checkpoint에서 시작해 local microbatch를 memory에 맞게 줄이고 accumulation으로 목표 global token을 구성한다. sequence count를 맞추는 실험과 token count를 맞추는 실험을 분리한다. 둘을 한 번에 바꾸면 길이 효과와 batch 효과를 식별하기 어렵다.

attention compute가 길이에 대해 빠르게 늘어 throughput은 떨어지고, activation checkpointing이나 parallel topology가 바뀔 수 있다. wall-clock schedule을 썼다면 동일 progress 동안 소비 token이 크게 달라진다. compile graph가 새 shape로 재컴파일되는 시간도 elapsed clock에 들어갈 수 있다. 따라서 change event에는 sequence policy, token denominator, kernel/compile warmup과 topology를 함께 기록한다.

첫 candidate가 loss spike를 보였다고 즉시 warmup을 다시 시작하지 않는다. loss normalization이 token mean인지 sequence mean인지, long document에서 label mask가 맞는지, position encoding과 attention mask 경계가 맞는지 먼저 본다. gradient norm이 길이에 비례해 커진다면 denominator 오류일 수 있다. scaler overflow와 clipping이 늘면 precision 또는 update magnitude 문제일 수 있다.

schedule migration 후보는 세 가지다. 첫째, committed update curve를 그대로 유지한다. optimizer state의 시간 연속성을 우선한다. 둘째, 남은 valid-token budget을 보존해 horizon을 다시 계산한다. data exposure를 우선한다. 셋째, 짧은 transition segment를 만들어 현재 lr와 목표 curve를 연결한다. 각 후보에서 현재 값, 기울기, 남은 면적, endpoint가 어떻게 달라지는지 표로 낸다.

전환 checkpoint는 데이터 경계와 optimizer commit 뒤에 만든다. partial accumulation gradient를 저장하지 않는다면 accumulation 중간 checkpoint를 허용하지 않는다. 새 loader의 첫 global sample IDs와 packing 결과를 dry-run하고, 2K loader가 이미 소비한 문서 offset과 겹치지 않는지 확인한다. 같은 원문을 다른 chunking으로 다시 노출하는 것이 의도라면 repeat lineage에 표시한다.

승인 gate는 첫 백 update 동안 NaN 0, clock invariant 0, sample 중복·누락 허용 범위, expected lr oracle 일치, update norm과 validation probe의 사전 범위다. throughput 하락은 예상 analytical model과 profiler breakdown을 비교한다. 이 gate를 통과한 뒤에만 긴 run을 이어간다.

이 사례는 curriculum이 data-layer knob가 아님을 보여 준다. 길이는 token clock, batch statistics, kernel shape, memory와 topology를 동시에 바꾼다. scheduler 장에서 이를 다루는 이유는 lr 곡선이 이 변화의 시간 좌표를 정의하기 때문이다.

**수치 검산**

전환 전 global valid token이 update당 평균 48,000이고 100억 token이 남았다면 약 208,334 update가 필요하다. 전환 뒤 평균이 160,000이면 약 62,500 update다. 기존 update horizon을 유지하면 약 333억 token을 더 보게 된다. 반대로 token horizon을 보존하면 optimizer moment와 decay가 3분의 1 이하 update 안에 끝난다.

현재 lr가 (2.0\times10^{-4}), 최소 lr가 (2.0\times10^{-5})이고 cosine progress가 0.4라고 하자. 새 token horizon에서 progress를 단순히 consumed/total로 다시 계산하면 현재 lr가 jump할 수 있다. 경계 lr를 고정한 새 cosine segment를 풀거나 phase offset을 구해야 한다. 값만 맞추면 derivative가 달라질 수 있으므로 첫 열 개 lr와 누적 면적을 비교한다.

global token을 유지하려고 accumulation을 줄이는 경우 committed update당 token은 비슷하지만 microbatch 수가 달라진다. optimizer/scheduler clock은 보존되기 쉽지만 gradient가 모이는 방식, pipeline bubble, communication frequency가 달라진다. dropout RNG와 sample grouping도 다르다. exact trajectory가 아니라 statistical continuation을 목표로 하고 고정 probe에서 tolerance를 정한다.

**사례 연구: preemption이 잦은 cluster**

spot 또는 불안정한 대형 cluster에서는 checkpoint 주기와 schedule semantics가 연결된다. 매 500 update 저장하던 run이 평균 300 update마다 중단된다면 대부분의 작업을 잃는다. 더 자주 저장하면 I/O와 pause가 늘고 wall-clock schedule이 흔들릴 수 있다. asynchronous checkpoint는 pause를 줄이지만 state snapshot의 consistent cut을 어렵게 한다.

기대 낭비 compute는 장애율과 checkpoint interval의 함수다. 그러나 model/optimizer shard를 쓰는 시간, storage bandwidth, restart rendezvous, data loader 복구 시간도 포함해야 한다. update clock은 저장 중에도 학습을 계속할 수 있지만 checkpoint가 어느 commit generation을 나타내는지 명확해야 한다. copy-on-write나 staging buffer가 없다면 optimizer가 state를 수정하는 동안 비동기 writer가 읽어 torn snapshot을 만들 수 있다.

schedule state는 작아서 먼저 쓰기 쉽지만 model shard보다 앞선 generation을 publish하면 위험하다. 모든 component를 generation ID로 묶고 마지막에 manifest를 원자적으로 commit한다. scheduler JSON 하나의 mtime이 최신이라는 이유로 선택하지 않는다. checkpoint reader는 expected component set과 checksum을 검증한다.

재시작 때 lost work를 replay하면 data cursor와 RNG도 checkpoint 시점으로 돌아가야 한다. 외부 streaming source가 이미 acknowledgment된 sample을 재생할 수 없다면 exact resume가 불가능하다. at-least-once와 at-most-once 소비 정책을 명시하고 sample lineage로 중복·누락을 측정한다. schedule clock이 어느 소비 의미를 따르는지도 정한다.

wall-clock decay를 사용하면 downtime 동안 clock을 멈출지 계속 흐르게 할지가 문제다. 예산 마감까지 남은 실제 시간을 기준으로 한다면 downtime도 포함할 수 있지만, 재시작 후 lr가 갑자기 낮아진다. productive training time을 기준으로 하면 downtime을 제외한다. monotonic timer와 accumulated active interval을 checkpoint하고 정책을 문서화한다.

**checkpoint cadence의 실험**

cadence 후보별로 평균 pause, background I/O가 training throughput에 주는 영향, 저장 실패율, recovery point objective를 측정한다. synthetic state가 아니라 실제 shard 크기와 storage path를 쓰되 대규모 모델 실행 없이 파일 layout과 static timing을 소규모 fixture로 검증할 수 있다. fault injection으로 writer 중단과 checksum mismatch를 만든다.

checkpoint interval을 token 기준으로 잡으면 batch와 world size가 바뀌어도 데이터 노출량 간격을 유지한다. update 기준은 optimizer state 변화 횟수를 유지한다. wall time 기준은 운영 RPO를 유지한다. 세 조건을 함께 만족할 수 없으므로 우선순위를 정한다. 대형 run에서는 token 또는 update trigger와 최대 wall-time 상한을 결합할 수 있다.

preemption 직전 알림 시간이 짧다면 emergency checkpoint는 전체 optimizer state를 못 쓸 수 있다. model-only checkpoint를 남기고 optimizer를 재초기화하는 것은 resume가 아니라 새 training segment다. lr warmup과 optimizer moment 재구성 정책, 품질 위험을 별 RecipeID로 남긴다. 불완전 state를 full resume처럼 사용하지 않는다.

**schedule과 평가 cadence를 같은 축에 놓는다**

validation을 매 1,000 update 실행하면 batch가 바뀔 때 평가당 token 간격이 달라진다. scaling 실험에서 후보 batch마다 같은 update cadence를 쓰면 큰 batch가 더 많은 데이터를 본 뒤 평가된다. fixed-token 비교에는 평가 trigger도 consumed valid token에 맞춰야 한다.

평가가 synchronous이면 wall-clock throughput에 포함되고, asynchronous이면 checkpoint staleness가 생긴다. metric point에는 evaluated checkpoint ID, train counter, consumed token과 recipe generation을 붙인다. 현재 dashboard step에 과거 checkpoint의 metric을 그려 넣지 않는다.

evaluation dataset의 sample count가 작으면 noise가 schedule 차이보다 클 수 있다. seed 또는 deterministic generation, confidence interval, multiple metrics를 사용한다. scaling fit에는 training loss와 held-out loss의 정의가 같아야 한다. token-average와 sequence-average를 섞지 않는다.

평가 시점이 warmup 끝, curriculum boundary, resize 직후와 우연히 겹치면 transient를 장기 추세로 오해할 수 있다. event annotation을 넣고 필요하면 경계 전후 고정 probe를 추가한다. 그렇다고 나쁜 결과를 “transient”라며 임의로 제외하지 않는다. 제외 규칙을 사전에 정한다.

**가설별 결정표**

질문이 “warmup을 몇 step으로 할까”라면 먼저 안정화할 상태가 optimizer update인지 token exposure인지 묻는다. batch나 sequence length가 바뀔 예정이면 두 시계의 대응을 표로 낸다. 고정 recipe를 재현하려면 upstream의 실제 호출 순서와 boundary를 oracle로 고정한다.

질문이 “batch를 얼마나 키울까”라면 hardware utilization과 gradient noise를 분리한다. fixed-token pilot에서 품질 효율을, profiler에서 시스템 효율을 측정한다. linear scaling은 출발 후보이지 기본 법칙이 아니다. clipping과 adaptive optimizer가 effective update를 바꾸는지 본다.

질문이 “모델과 데이터를 얼마나 쓸까”라면 scaling fit의 관측 범위와 parameter/token convention을 검증한다. unique/seen token, dense/MoE active parameter, achieved compute를 분리한다. holdout scale 예측과 불확실성을 의사결정에 포함한다.

질문이 “world size를 바꿔도 되나”라면 보존할 동일성을 먼저 정한다. global batch, remaining token, current lr, optimizer update 수를 모두 자동 보존할 수는 없다. old/new state를 migration function으로 연결하고 dry-run certificate와 atomic manifest를 요구한다.

질문이 “resume가 성공했나”라면 process 생존을 보지 않는다. next data IDs, counters, applied lr, optimizer/scaler state와 parameter delta를 uninterrupted reference에 비교한다. 요구 수준이 bitwise인지 statistical인지 선언한다.

질문이 “loss spike가 schedule 때문인가”라면 최초 불일치 사건을 찾는다. data mixture와 denominator, precision skip, clipping, topology event를 같은 시간축에서 비교한다. nominal lr 선 하나의 상관관계를 인과로 부르지 않는다.

모든 판단에는 취소 조건이 있어야 한다. 어떤 관측이 현재 가설을 반박하는지, 어느 checkpoint로 돌아갈지, data를 replay할지, 새 RecipeID를 만들지를 미리 적는다. schedule 운영의 성숙도는 가장 아름다운 cosine 곡선이 아니라 실패를 빠르게 식별하고 의미를 보존한 채 복구하는 능력으로 드러난다.

이 결정표를 실제 배포 전 회의에서 소리 내어 검산한다. 데이터 담당자는 valid-token 정의와 다음 sample ID를 설명하고, optimizer 담당자는 skipped update와 state commit 조건을 설명한다. 분산 담당자는 resize 전후의 mesh와 counter owner를, 관측 담당자는 expected 수열과 최초 불일치 경보를 설명한다. 한 사람이 모든 값을 외우는 대신 같은 event schema로 서로의 가정을 교차 확인한다.

검토 중 “framework가 알아서 한다”는 답이 나오면 고정 revision의 함수와 test로 내려간다. “대략 같은 곡선”이라는 답이 나오면 첫 열두 수열과 경계 값을 계산한다. “resume는 잘 됐다”는 답이 나오면 next data ID, optimizer moment, scheduler state와 parameter delta를 요구한다. 설명할 수 없는 자동성은 운영 위험이다.

최종 승인 기록은 성공 결과만 담지 않는다. 기각한 scaling 후보, 실패한 batch, 맞지 않았던 fit과 rollback 사건도 남긴다. 실패 원인의 코드 좌표와 반증 관측이 다음 run의 prior가 된다. 이렇게 축적된 원장은 특정 framework의 옵션 목록보다 오래 살아남는다. 구현 이름은 바뀌어도 시계의 owner, update의 commit, token denominator, migration의 보존 조건이라는 질문은 그대로이기 때문이다.

이 장을 읽은 뒤 독자는 learning rate 값을 추천하는 데서 멈추지 않아야 한다. 그 값이 어느 사건에 적용되고, 어떤 데이터와 상태를 움직였으며, 장애 뒤 무엇을 보존하는지 증명해야 한다. 그 증명이 schedule 설계의 최종 산출물이다.

마지막 인수자는 checkpoint 하나를 임의로 골라 역방향으로 추적한다. 저장된 scheduler counter에서 마지막 committed update를 찾고, 그 update의 global valid-token batch와 sample ID, scaler 판정, optimizer delta를 복원한다. 이어서 정방향으로 다음 세 lr와 data ID를 예측한다. 실제 dry-run 결과가 예측과 맞지 않으면 설정 설명이 아무리 그럴듯해도 승인하지 않는다. 이 양방향 추적은 stale state, off-by-one, 잘못된 denominator를 한꺼번에 드러낸다.

또한 scaling 결론에는 적용 범위를 붙인다. 관측한 model size, data mixture, training phase, precision과 topology 밖에서는 새 가설로 취급한다. 다른 팀의 수치를 옮길 때도 같은 원칙을 쓴다. 숫자를 복사하는 것이 아니라 그 숫자가 성립한 상태 기계와 비용 조건을 함께 옮긴다. 그래야 대형 학습의 schedule이 전승되는 비법이 아니라 검증 가능한 공학 지식이 된다.

최종 원장에는 작성자와 검토자, 생성 시각, 코드 revision, 데이터 manifest, recipe hash도 남긴다. 동일한 이름의 run이 재사용되어 증거가 섞이지 않게 immutable identifier를 쓴다. 검토자는 표의 숫자를 dashboard 화면이 아니라 원본 event와 checkpoint manifest에서 표본 대조한다. 이 마지막 확인이 끝나야 schedule 변경을 production training에 적용한다.

**option에서 state와 효과까지 추적하는 법**

scheduler 옵션은 곡선 이름만으로 의미가 정해지지 않는다. `warmup_steps`, `warmup_ratio`, `num_training_steps`, `min_lr_ratio`, `last_epoch`와 호출 위치가 함께 state machine을 만든다. 각 옵션은 먼저 어느 시계 단위인지, 누가 절대값으로 변환하는지, 언제 직렬화되는지와 어떤 parameter group에 적용되는지를 가진다. 그 뒤 내부 state가 바뀌고, 최종적으로 optimizer가 읽는 group별 lr와 decay 효과가 바뀐다.

`warmup_ratio=0.03`은 Trainer가 total update 수를 계산한 뒤 integer warmup으로 바꿀 수 있다. dataset length, epoch 수, accumulation과 world size가 total update 계산에 들어가면 data filtering이나 elastic resize가 같은 ratio를 다른 절대 구간으로 만든다. rounding이 floor인지 ceil인지에 따라 짧은 run에서는 첫 경계가 한 update 움직인다. manifest에는 ratio뿐 아니라 해석된 warmup update와 계산 입력을 저장한다.

`num_training_steps`는 optimizer commit 수인지 dataloader iteration 수인지 호출자가 결정한다. gradient accumulation 중 scheduler를 호출하면 microstep schedule이 되고, AMP overflow에도 호출하면 attempted schedule이 된다. factory 이름이 같아도 Trainer loop의 호출 조건이 다르면 다른 곡선이다. trace에서 attempted, committed와 scheduler call event를 join해 실제 owner를 확인한다.

`last_epoch`라는 필드는 dataset epoch가 아니라 scheduler가 몇 번 전진했는지를 표현하는 경우가 많다. load 직후 optimizer group lr와 scheduler의 base lr, last state, next lambda를 모두 출력한다. state dict에 closure가 포착한 total horizon이나 custom lambda 코드가 들어가지 않는다면 constructor config와 code revision이 checkpoint 일부다.

`min_lr_ratio`는 terminal lr만 바꾸지 않는다. cosine 또는 linear decay 전 구간의 면적, decoupled weight decay 누적 product와 late-stage update noise를 바꾼다. 효과 열에는 terminal 값, 누적 lr, 누적 decay와 마지막 구간의 validation 변화를 둔다. option에서 state로, state에서 실제 update로 이어지는 연결을 수치로 닫는다.

parameter group multiplier도 놓치기 쉽다. scheduler가 모든 group의 현재 lr에 multiplier를 곱하는지 base lr에서 재계산하는지, resume 후 수동으로 바꾼 group lr를 덮어쓰는지 본다. Muon/AdamW hybrid처럼 group 의미가 다르면 nominal scheduler multiplier와 effective update RMS가 같은 비율로 변하지 않는다. group별 expected/applied lr와 update-to-weight ratio를 기록한다.

**Transformers Trainer 재개를 사건 순서로 검증한다**

Trainer 기반 run의 재개 상태는 model과 scheduler 파일 둘로 끝나지 않는다. optimizer state, scheduler state, scaler, Trainer global step, gradient accumulation 위치, sampler 또는 iterable cursor, RNG, callback state와 해석된 training arguments가 같은 checkpoint generation을 가리켜야 한다. 저장 파일이 존재하는지보다 각 component가 어느 optimizer commit 뒤에 만들어졌는지 확인한다.

고정 revision에서 scheduler factory, `Trainer.create_optimizer_and_scheduler`, training loop의 optimizer step 조건, scheduler step 호출, checkpoint save와 load 함수를 따라간다. 함수명은 release 사이에 이동할 수 있으므로 commit과 blob을 고정한다. unit test는 단순 state load인지 uninterrupted/resume lr 수열 비교인지 assertion strength를 구분한다. local fixture는 실제 채택 옵션과 callback 조합을 사용한다.

첫 fixture는 accumulation 3, warmup 4, total 12인 작은 optimizer다. 각 microbatch, optimizer commit과 scheduler call에서 group lr를 기록한다. 한 번의 overflow와 한 번의 empty-label skip을 넣는다. expected table은 scheduler가 committed update를 읽는 정책에 따라 손으로 계산한다. framework trace가 table과 다르면 장기 model을 돌리기 전에 호출 ordering을 고친다.

두 번째 fixture는 commit 직전, optimizer step 직후 scheduler 전, scheduler 직후와 checkpoint publish 중에 process를 중단한다. reader는 complete generation만 선택해야 한다. optimizer가 새 state인데 scheduler가 이전 state인 조합을 조용히 읽지 않는다. save hook과 callback이 어느 위치에서 실행되는지 trace event로 고정한다.

세 번째 fixture는 checkpoint load 뒤 같은 config로 세 update를 진행한다. next batch ID, loss scale, optimizer bias-correction step, current/next lr와 parameter delta를 uninterrupted control과 비교한다. 첫 lr만 같고 둘째부터 갈라지면 horizon 또는 lambda constructor가 다를 수 있다. lr은 같은데 delta가 다르면 optimizer/scaler/data state를 본다.

네 번째 fixture는 total horizon, warmup ratio 또는 dataset length를 일부러 바꾼다. loader가 old state와 new config를 어떻게 합성하는지 관찰하고 silent 허용을 금지한다. 옛 curve 유지, 남은 token horizon 재계산 또는 현재 lr에서 새 segment 연결 가운데 정책을 명시한다. 어떤 경우든 새 RecipeID와 migration report가 필요하다.

다섯 번째 fixture는 world size와 accumulation을 함께 바꾼다. nominal global token batch가 같아도 sampler partition과 packed sample 구성이 달라질 수 있다. Trainer `global_step` continuity, global committed clock, consumed/valid token과 first sample multiset을 각각 판정한다. exact sample resume가 실패해도 token continuity가 성공할 수 있으며 두 등급을 분리한다.

callback은 숨은 state owner가 될 수 있다. early stopping patience, plateau metric history, save/eval cadence와 custom scheduler controller가 checkpoint에 포함되는지 본다. callback order가 lr step보다 앞인지 뒤인지도 효과를 바꾼다. callback state가 누락되면 단순한 warning이 아니라 재개 semantics가 달라진 branch다.

**sweep의 추정대상과 통계 설계**

batch, lr와 schedule sweep은 먼저 추정대상을 한 문장으로 쓴다. 예를 들어 “고정된 100억 valid token에서 batch를 두 배로 했을 때 held-out token loss의 평균 변화”는 데이터 노출 효과를 묻는다. “고정된 72 accelerator-hour에서 WSD가 cosine보다 도달하는 최저 loss의 변화”는 시스템 비용을 포함한다. “각 방법을 동일 trial compute로 튜닝했을 때 선택 정책의 기대 성능”은 개별 hyperparameter가 아니라 탐색 절차를 비교한다.

실험 단위는 독립 run이다. 한 run의 여러 checkpoint를 독립 표본처럼 세면 표준오차가 과소평가된다. paired seed를 쓰면 같은 initialization과 data lineage에서 후보 차이를 계산한다. cluster 장애로 한 쌍만 탈락한 경우 complete case만 남기는 규칙이 편향을 만드는지 검토하고 failure 자체를 결과로 보고한다.

primary endpoint, terminal token과 평가 dataset을 사전에 고정한다. best-of-many checkpoint를 쓰려면 selection procedure가 estimand 일부다. validation으로 고르고 같은 validation에서 효과를 보고하면 낙관 편향이 생긴다. selection과 final evaluation split을 분리하거나 nested 절차를 쓴다. 여러 task를 보고할 때 aggregate 정의와 결측 처리도 먼저 정한다.

lr 후보는 로그 간격이 보통 유용하지만 안정 경계 근처를 충분히 덮어야 한다. warmup, batch와 optimizer를 동시에 전격자로 곱하면 비용이 폭발한다. screening 단계와 confirmatory 단계를 나누되 screening 결과만 publication 결론으로 쓰지 않는다. adaptive early stopping은 유망하지 않은 run을 줄일 수 있지만 중단 규칙과 중단 loss를 모두 보존한다.

seed 수는 관행 숫자가 아니라 관심 효과와 run 간 분산으로 정한다. pilot에서 paired difference 분산을 추정하고 최소 의미 효과를 검출할 수 있는 범위를 계산한다. 대형 run이라 표본이 작으면 p-value 하나보다 모든 점, effect interval과 방향 일관성을 보인다. scaling exponent fit도 point estimate와 covariance, holdout 예측 오차를 함께 낸다.

heteroscedasticity가 흔하다. 큰 batch, 불안정 lr 또는 특정 model scale에서 variance와 failure rate가 달라질 수 있다. 동일 분산 t-test만 자동 사용하지 않는다. robust interval, bootstrap 또는 계층 모형을 고려하되 seed와 scale 수가 적을 때 복잡한 모형의 prior 민감도를 공개한다. 로그 loss와 원 loss에서 효과 해석도 다르다.

compute와 data scaling fit에서는 관측 범위를 넘어선 외삽 거리를 표시한다. FLOP는 이론식과 achieved profiler 값을 구분하고, token은 unique와 seen, input과 loss-bearing을 구분한다. dense parameter 수와 MoE active parameter를 섞지 않는다. 후보 frontier에서 같은 validation을 만족하는 compute 최소점과 같은 compute에서 loss 최소점을 별 estimand로 낸다.

결측 run은 무작위가 아닐 가능성이 크다. OOM, NaN, timeout과 preemption을 원인별로 기록한다. 불안정 recipe의 실패를 제외하고 성공 loss만 평균하면 그 recipe를 과대평가한다. 성공 조건부 품질과 전체 trial 성공 확률을 함께 보고, 운영 의사결정에는 예상 재시도 compute까지 포함한다.

sweep 결과표는 recipe, seed, model/data scale, schedule, clock, endpoint, outcome, failure와 resource를 long format으로 보존한다. 그래프용 요약은 이 원장에서 파생한다. dashboard에서 손으로 고른 점을 최종 데이터로 쓰지 않는다. 모든 제외와 변환은 재실행 가능한 query로 남긴다.

**compute·data·model 공동 설계**

scaling law는 model 크기 하나를 키우라는 명령이 아니다. 주어진 예산에서 parameter, 학습 token, sequence length, batch, optimizer와 parallel topology를 공동 선택하는 근사 모델이다. parameter를 늘리면 forward/backward FLOP와 optimizer state가 늘고, token을 늘리면 data 반복과 wall time이 늘며, 길이를 늘리면 attention 비용과 activation memory가 비선형으로 바뀐다.

첫 원장은 model의 dense parameter, embedding, expert total과 token당 active parameter를 분리한다. 둘째 원장은 input, valid, unique와 repeated token을 가진다. 셋째는 achieved accelerator FLOP, communication, input pipeline, checkpoint와 evaluation 시간을 가진다. theoretical compute만 맞추면 통신이 많은 큰 model과 data-heavy 작은 model의 wall-clock을 잘못 비교할 수 있다.

fit은 관측한 model와 token 격자 안에서 먼저 검증한다. 일부 scale을 holdout하고 loss 예측과 불확실성을 확인한다. 작은 proxy에서 얻은 exponent를 훨씬 큰 topology로 옮길 때 kernel efficiency와 batch criticality가 바뀐다. 외삽 결과는 점 하나가 아니라 plausible frontier와 위험 범위로 낸다.

data quality와 반복은 token 수의 숨은 축이다. 높은 품질 subset을 더 반복하는 것과 넓은 corpus를 한 번 보는 것은 같은 seen token이어도 다른 효과다. mixture weight, dedup, filtering revision과 epoch/repeat lineage를 fit feature 또는 별 실험 축으로 둔다. curriculum이 token 위치에 따라 바뀌면 단일 평균 mixture로 압축하지 않는다.

model 크기가 커지면 optimal batch와 안정 lr도 바뀔 수 있다. 각 scale에서 batch·lr를 합리적으로 retune하지 않고 같은 recipe를 복사하면 scaling curve가 optimizer mismatch를 측정한다. 반대로 각 scale에 무제한 tuning을 주면 탐색 compute가 다르다. 공통 tuning protocol과 budget을 정하고 sensitivity를 보고한다.

sequence length 선택은 data와 model 양쪽을 바꾼다. 같은 token 수에서도 긴 sequence는 update당 sample 수, packing waste, attention FLOP와 valid-label 비율을 바꾼다. context capability가 목적이면 평가가 길이별 성능을 측정해야 한다. 짧은 평가 loss만으로 긴 context compute를 낭비라고 결론내리지 않는다.

hardware co-design에서는 tensor/pipeline/data/expert parallel 후보마다 memory feasibility와 communication model을 계산한다. optimizer state sharding으로 model을 키울 수 있어도 collective tail이 throughput을 낮출 수 있다. compiler graph와 kernel이 target shape를 지원하는지 proxy run에서 확인한다. nominal FLOP utilization 하나로 input stall과 checkpoint 비용을 숨기지 않는다.

최종 선택은 예상 loss, interval, 성공 확률, wall-clock, peak memory와 운영 복구 비용의 다목적 문제다. 한 개 scalar score를 쓰면 가중치를 공개한다. deadline 제약, accelerator quota와 최소 품질을 먼저 적용하고 남은 frontier를 비교한다. scaling fit이 틀렸을 때 되돌릴 checkpoint와 다음 측정 scale도 결정에 포함한다.

**token clock을 분산 원장으로 구현한다**

token counter는 각 rank의 tensor numel을 더한 값이 아니다. padding, causal shift, prompt masking, dropped token, repeated pack boundary와 auxiliary objective를 반영한 denominator 정의가 필요하다. input token, attention-valid token, primary loss label, auxiliary label과 accepted rollout token을 별 counter로 둔다. scheduler가 읽는 counter는 그중 하나를 명시한다.

각 microbatch는 local count와 batch lineage를 만든다. accumulation commit 시 모든 rank의 count를 합치되 gradient가 실제 사용한 global denominator와 같은 source에서 계산한다. counter 코드와 loss normalization 코드가 따로 mask를 만들면 둘이 drift할 수 있다. 하나의 mask summary를 양쪽에 공급하고 checksum을 event에 남긴다.

overflow나 NaN skip에서는 consumed token과 committed update가 갈라진다. token-based schedule이 consumed token을 읽으면 lr는 데이터가 지나간 만큼 전진할 수 있고 optimizer state는 멈춘다. valid-token-on-commit을 읽으면 skip token을 schedule에서 제외한다. 어느 정책도 자동 정답이 아니며 목적과 replay 가능성에 맞춰 선택한다. event는 둘 다 복원할 수 있어야 한다.

분산 합의는 parameter apply 전에 commit decision을 만든다. 한 rank가 empty label이거나 found-inf면 policy에 따라 global skip을 합의한다. scheduler owner가 rank-local 결과만 보고 먼저 전진하지 않는다. applied lr를 broadcast하고 모든 rank가 checksum을 assertion한다. token delta, commit ID와 scheduler event를 하나의 transaction generation으로 묶는다.

streaming data에서는 소비 acknowledgment와 optimizer commit의 원자성이 어렵다. checkpoint rollback 뒤 sample을 다시 받을 수 있으면 at-least-once, 받을 수 없으면 at-most-once gap이 생긴다. token ledger에 source offset, sample ID와 acknowledgment policy를 남긴다. schedule continuity와 exact data replay를 별 등급으로 판정한다.

counter overflow, unit 변경과 schema migration도 시험한다. 정수 token을 float progress로 너무 일찍 바꾸면 큰 값에서 작은 증가를 잃을 수 있다. authoritative counter는 충분한 폭의 integer로 저장하고 formula 경계에서만 정규화한다. input에서 valid token으로 clock을 바꾸는 migration은 과거 event를 재생해 새 위치를 구하며 nominal 비율로 덮지 않는다.

대시보드는 각 commit에 attempted, committed, input, valid, sample, wall active time과 applied lr를 보여 준다. token/update가 급변하면 sequence curriculum, packing, empty mask 또는 world-size change event를 겹쳐 본다. global 합과 rank 분포를 함께 저장해 한 rank의 data starvation을 평균이 숨기지 않게 한다.

### 13.5.3 failure diagnosis에서 인수 판정까지

loss spike가 보이면 가장 먼저 spike 직전과 직후의 data ID, denominator, applied lr, update norm과 precision decision을 한 사건 표에 둔다. lr가 바뀌었다는 상관만으로 scheduler를 원인으로 결론내리지 않는다. batch hash와 denominator가 먼저 달라졌다면 data 경로를, found-inf와 clipping이 바뀌었다면 numeric 경로를 우선한다.

warmup이 너무 빨리 끝난 것처럼 보이면 expected lr 순수 함수와 실제 수열을 비교한다. warmup unit, rounding, accumulation, overflow skip과 scheduler 호출 횟수를 차례로 본다. `last_epoch`만 수정해 증상을 가리지 않는다. 최초로 expected와 applied가 갈린 event에서 call stack과 state를 확보한다.

resume 직후만 갈라지면 next sample, current/next lr, optimizer step, scaler와 callback state를 control checkpoint와 대조한다. 첫 lr은 같지만 parameter delta가 다르면 optimizer 또는 data 문제다. 첫 delta는 같고 둘째 lr부터 다르면 scheduler horizon, constructor config 또는 호출 ordering 문제다. 이 분해가 긴 재학습보다 싸다.

world-size 변경 뒤 progress가 빨라졌다면 global token/update와 authoritative clock을 본다. step schedule을 유지하면서 token batch가 커졌을 수 있다. 반대로 token schedule을 유지해 terminal step 수가 줄었을 수 있다. 어느 보존 조건을 승인했는지 migration report와 비교한다. 예상된 변화라면 오류 경보가 아니라 recipe branch로 표시한다.

plateau에서는 terminal lr 조기 도달, min ratio, decay 면적과 evaluation staleness를 확인한다. 동시에 data repeat, mixture와 model capacity limit을 본다. schedule sweep은 원인 후보 하나일 뿐이다. oscillation에서는 warmup boundary, large-batch noise, clipping fraction과 optimizer update ratio를 함께 본다.

failure rehearsal은 overflow, empty labels, scheduler-only advance, optimizer-only commit, corrupted scheduler state, old horizon load, token counter rollback, rank-local split와 checkpoint torn write를 포함한다. 각 주입에는 예상 terminal, 탐지 event, rollback point와 복구 뒤 다음 세 수열이 있다. hang을 timeout으로 끝내는 것만 성공이 아니라 최초 오류가 보존되고 partial state가 publish되지 않아야 한다.

최종 인수는 작은 deterministic fixture, distributed failure fixture, paired scaling pilot와 실제 checkpoint dry-run 네 단계다. 첫 단계는 수식 경계와 호출 순서를, 둘째는 global clock과 atomicity를, 셋째는 estimand와 품질 효과를, 넷째는 운영 resume를 검증한다. 앞 단계 증거 없이 큰 run의 매끈한 curve로 대체하지 않는다.

인수 문서에는 schedule formula, authoritative clock, option 해석값, Trainer source 좌표, state schema, resume 정책, scaling estimand, seed와 실패 처리, compute/data/model 범위, 경보와 rollback을 담는다. 또한 실행하지 않은 dtype, topology, dataset과 model scale을 명시한다. 검증 범위 밖의 재사용은 새 가설이다.

종료 판정은 다음 질문으로 닫는다. 임의 checkpoint에서 이전 commit의 데이터와 lr를 복원할 수 있는가. 다음 세 lr, token delta와 parameter update를 예측할 수 있는가. world size나 horizon 변경이 어느 보존 조건을 선택했는가. sweep 결론이 명시한 estimand와 통계 절차에서 나왔는가. 실패 시 complete cut으로 돌아갈 수 있는가. 다섯 답이 artifact로 확인될 때 schedule과 scaling recipe를 승인한다.

**네 가지 scaling 결정을 수치 사건으로 푼다**

첫 사례는 global batch를 두 배로 키우는 결정이다. baseline의 update당 valid token, gradient noise proxy, throughput, clipping fraction과 validation을 측정한다. 후보 A는 lr와 token schedule을 유지하고, 후보 B는 lr만 키우며, 후보 C는 warmup과 decay 위치도 token 기준으로 맞춘다. 같은 committed step 비교는 큰 batch에 더 많은 데이터를 주므로 primary endpoint로 쓰지 않는다. 같은 valid token과 같은 평가 checkpoint lineage에서 paired difference를 계산한다.

후보 B가 더 빨리 loss를 내리지만 실패 seed가 늘었다면 성공 run 평균만으로 승인하지 않는다. 전체 trial 성공 확률, 재시도 compute와 최악 update ratio를 포함한다. 후보 C가 품질은 같고 wall time을 줄여도 checkpoint와 evaluation overhead가 달라졌다면 productive training time과 총 elapsed를 나눠 보고한다. batch 결정은 lr scaling 법칙 하나가 아니라 품질과 시스템의 공동 결과다.

둘째 사례는 model을 두 배로 하고 token을 줄이는 예산 이동이다. dense와 active parameter, achieved FLOP와 optimizer state를 다시 센다. 작은 model의 lr와 batch를 그대로 복사하지 않고 동일 protocol로 retune한다. data mixture와 unique token은 유지하되 반복 횟수가 달라졌다면 seen token만 같은 비교로 부르지 않는다. holdout scale에서 scaling fit의 예측 오차를 확인한 뒤 큰 run을 승인한다.

셋째 사례는 cosine 중간 checkpoint에서 WSD branch를 만드는 결정이다. parent optimizer, sampler와 다음 batch를 복사하고 미래 lr sequence만 바꾼다. 현재 lr의 값 연속성, 첫 차분, 남은 lr 면적과 누적 decay product를 계산한다. 이미 지난 warmup을 다시 적용하지 않는다. 각 branch의 terminal token과 평가 trigger를 맞추고, 중단된 branch도 sweep 원장에 남긴다.

넷째 사례는 preemption 뒤 다른 world size로 재개하는 결정이다. old/new global sequence batch, nominal과 realized valid token, accumulation, sampler partition과 topology를 표로 만든다. global token batch를 보존할지 남은 token horizon을 보존할지 먼저 선택한다. 두 조건을 동시에 만족하지 못하면 한쪽을 silent 변경하지 않는다. first batch multiset과 next lr dry-run을 checkpoint publish 전에 실행한다.

resize 뒤 첫 lr가 control과 같아도 안심하지 않는다. 다음 세 scheduler call, optimizer bias correction, scaler, token delta와 parameter update를 본다. reduction order가 달라 bitwise trajectory가 불가능하면 고정 probe의 수치 tolerance와 짧은 validation interval을 선언한다. sample-exact, token-continuous와 statistical continuation을 각기 다른 등급으로 기록한다.

**schedule 변경 RFC의 검산 순서**

RFC 첫 문장은 변경 옵션이 아니라 기대 효과다. 예를 들어 “동일한 50억 valid token에서 warmup overflow를 줄이되 terminal loss를 악화시키지 않는다”처럼 estimand와 비열등성 범위를 쓴다. 그 아래 현재 clock owner, formula, option 해석값과 관측된 failure를 둔다. 해결책보다 현재 상태를 먼저 고정해야 사후에 원인을 판정할 수 있다.

둘째 부분은 old/new 수열이다. warmup 시작과 끝, decay 시작, terminal, resume checkpoint 주변의 최소 열두 lr를 순수 함수로 계산한다. group별 multiplier와 weight decay product도 넣는다. 그래프만 제시하면 off-by-one과 짧은 discontinuity가 숨는다. 표의 값은 framework 실행 trace와 자동 비교한다.

셋째 부분은 state migration이다. 그대로 보존할 counter, 재계산할 horizon, 새로 시작할 segment와 폐기할 callback state를 열거한다. migration 전후 current lr, next lr와 progress가 일치하는 조건을 쓴다. old checkpoint를 직접 덮지 않고 parent를 가진 새 immutable generation을 만든다.

넷째 부분은 반증 조건이다. expected/applied lr 불일치, overflow 증가, update ratio 상한, validation 비열등성 실패, token/update drift와 resume mismatch를 정한다. 각 조건에는 자동 중단 시점과 rollback checkpoint가 있다. loss가 좋아 보인다는 이유로 clock invariant 실패를 무시하지 않는다.

다섯째 부분은 관측 범위다. model, data revision, sequence, precision, optimizer, topology와 Trainer commit을 명시한다. 범위 밖 확장은 별 RFC다. 성공한 small model schedule을 큰 MoE나 긴 context에 자동 승계하지 않는다. compute와 communication regime가 달라지기 때문이다.

마지막 검토자는 dashboard가 아니라 event 원장에서 표본을 뽑는다. checkpoint 하나의 이전 update에서 data, denominator, scaler, applied lr와 parameter commit을 역추적하고 다음 세 사건을 예측한다. source trace, 순수 oracle과 dry-run이 같은 수열을 가리키면 변경을 승인한다. 그렇지 않으면 옵션 이름이 익숙해도 state 의미가 닫히지 않은 것이다.

**warmup 경계를 commit 사건으로 고정한다**

warmup 옵션은 시작 lr, peak lr, 기간 단위, 곡선과 첫 update의 index 규칙이다. 기간이 update인지 valid token인지에 따라 같은 숫자가 다른 상태를 만든다. scheduler 상태는 committed update, committed token, last applied lr와 recipe generation을 가진다. gradient accumulation 중간이나 overflow로 취소된 microbatch는 commit clock을 전진시키지 않는다. 그 결과 peak 도달 시점과 parameter delta가 결정된다.

고정 source는 scheduler 함수뿐 아니라 loss denominator, token 집계, scaler skip 판정과 optimizer 호출 순서를 포함한다. 고정 test는 첫 사건, warmup 직전·정확한 경계·직후, 빈 valid-label batch, accumulation 크기 변경과 overflow를 손 계산 oracle에 대조한다. `<`를 `<=`로 바꾼 mutant가 경계 fixture에서 실패해야 한다. failure는 expected/applied lr 불일치, clock drift, 취소된 step의 state 변화와 resume 뒤 다음 세 lr 차이다.

분산에서는 rank별 valid token 합이 optimizer가 사용한 global denominator와 같은 source에서 나왔는지 검사한다. 일부 rank가 빈 batch여도 collective 참가 순서는 같아야 한다. 11장의 collective trace와 12장의 OptimizerStepID를 결합하면 scheduler만 전진하거나 optimizer만 commit된 혼합 사건을 거부할 수 있다.

**cosine decay와 최소 lr의 순서를 검산한다**

cosine 옵션은 peak, minimum, decay 시작과 종료 clock, cycle 여부와 경계 밖 clamp다. 먼저 cosine을 계산한 뒤 minimum 비율을 섞는 식과 절대 minimum으로 clamp하는 식은 다르다. 상태는 cycle index, phase, last clock과 applied lr를 저장한다. 옵션 digest가 달라지면 동일 step counter를 그대로 재해석하지 않고 migration 여부를 판정한다.

효과는 lr 수열, parameter group별 multiplier, update RMS와 종료 뒤 tail로 본다. base lr가 같아도 group multiplier가 두 번 적용되면 일부 layer만 어긋난다. 고정 source는 schedule helper, group construction과 optimizer에 전달되는 실제 lr write를 잇는다. 고정 test는 시작, 중간, 종료, 종료 초과와 매우 큰 token counter를 FP64 식에 비교하고 clamp 순서 mutant를 거부한다.

failure 판정에는 음수 lr, 비의도적 재상승, 경계 불연속, group ratio 변화, float counter 정밀도 손실과 checkpoint 재개 차이가 포함된다. wall-clock dashboard의 반올림값이 아니라 update 직전의 적용값을 감사한다. 6장의 재현성 계약에 따라 source commit과 수열 digest를 함께 보존한다.

**batch 확대에서 lr scaling과 clock을 분리한다**

batch 옵션은 microbatch, accumulation, data-parallel world size, valid-token 목표와 lr scaling rule이다. global sample 수와 global valid token은 packing과 masking 때문에 같지 않다. batch를 두 배로 만들 때 lr를 선형 또는 제곱근으로 바꾸는 선택과 scheduler 기간을 token 기준으로 유지하는 선택을 분리한다. 하나의 `scale_factor`가 둘을 동시에 바꾸게 두지 않는다.

상태는 accumulation progress, committed denominator, optimizer step, token clock과 lr recipe generation을 가진다. elastic world-size 변화가 발생하면 미완성 accumulation을 폐기하거나 정해진 denominator로 commit하는 정책이 필요하다. 효과는 noise scale proxy, update RMS, tokens per update, throughput과 validation을 함께 본다. 빠른 처리량이 update 수 감소를 숨기지 않게 fixed-token 비교를 둔다.

고정 test는 동일 example stream을 서로 다른 microbatch와 accumulation으로 묶어 global gradient와 lr 수열을 비교한다. source pin은 sampler, packer, loss reduction, distributed averaging, scheduler와 optimizer를 포함한다. failure는 denominator 불일치, token clock drift, lr 이중 scaling, partial accumulation 재사용과 공통 checkpoint에서 delta 비열등성 위반이다. 8장의 data lineage와 11장의 topology 원장을 상호참조한다.

**sequence length 전환을 별 schedule로 다룬다**

길이 옵션은 최대 sequence, packing, attention mask, loss mask, RoPE 또는 position scaling과 전환 clock이다. 길이를 늘리면 token당 attention FLOP, activation memory, update당 sequence 수와 valid-label 비율이 함께 바뀐다. lr schedule 옵션 하나로 압축하지 않는다. 상태는 length phase, data cursor, token clock, position recipe와 compiled graph generation을 가진다.

효과는 input·valid token, padding waste, tokens per second, FLOP per token, peak memory, update RMS와 길이별 evaluation으로 본다. 짧은 validation만 좋아도 긴 context 목적을 달성했다고 판정하지 않는다. 고정 source는 packer, mask builder, position transform, model config, compiler specialization과 scheduler phase selector를 연결한다.

고정 test는 전환 직전과 직후 같은 문서를 pack하고 label denominator와 position index를 검산한다. checkpoint를 경계 양쪽에서 재개해 다음 세 batch와 lr를 비교한다. failure는 data 중복·누락, position recipe 불일치, clock 이중 계산, graph fallback 폭증, memory budget 초과와 긴 길이 평가 하락이다. 9장의 memory peak와 10장의 profiler 증거를 사용한다.

**mixture curriculum과 token 가치를 추적한다**

mixture 옵션은 corpus별 weight, temperature, curriculum phase, sampling with replacement, dedup revision과 품질 filter다. seen token 수가 같아도 unique token과 반복 횟수는 다르다. scheduler clock은 물리 token을 읽을 수 있지만 scaling fit에는 corpus lineage와 반복을 별 feature로 둔다. 상태는 sampler RNG, corpus cursor, phase, accepted/rejected count와 mixture recipe generation을 저장한다.

옵션 변경은 다음 sample 분포와 gradient를 바꾸고, 결과로 loss와 optimal schedule 추정이 바뀐다. 효과는 corpus별 input·valid·unique token, repeat histogram, rejection, gradient norm과 evaluation slice로 본다. 전체 loss 하나가 작은 corpus의 과반복을 숨기지 않게 한다. 고정 source는 manifest, filter, dedup artifact, sampler와 checkpoint serializer까지 pin한다.

고정 test는 작은 유한 corpus에서 기대 sampling sequence와 checkpoint resume sequence를 exact 비교한다. weight 정규화, zero-weight corpus, shard 수 변화와 exhausted cursor를 시험한다. failure는 lineage 없는 token, weight drift, resume 중복·누락, RNG rollback과 허용 반복 상한 초과다. 8장의 data provenance가 source certificate의 일부다.

**scaling fit을 외삽 전에 반증한다**

fit 옵션은 loss 식, parameter와 token 정의, irreducible term, weighting, holdout 방식과 uncertainty 모델이다. dense parameter, expert total과 token당 active parameter를 섞지 않는다. 상태는 input table digest, fitted coefficient, covariance, optimizer result와 code generation을 가진다. 데이터 한 행이 수정되면 기존 fit을 같은 ID로 덮어쓰지 않는다.

효과는 training point residual, held-out scale error, interval coverage와 선택 frontier 변화로 본다. 평균 오차가 작아도 가장 큰 holdout을 계속 낙관하면 배포 의사결정에는 실패다. 고정 source는 측정 query, unit conversion, fit code, solver version과 plot data를 포함한다. 고정 test는 합성 coefficient 회수, 단위 변경, 누락 행과 outlier sensitivity다.

failure 판정은 holdout 오차, interval coverage 부족, coefficient 비식별성, 관측 범위 밖 과도한 외삽과 hardware regime 변화다. 새 topology에서 utilization이 달라지면 theoretical compute만으로 기존 curve를 승계하지 않는다. 10장의 achieved FLOP와 communication 측정을 feature 또는 별 제약으로 둔다.

**compute frontier를 wall-clock 제약으로 변환한다**

계획 옵션은 accelerator 수와 종류, deadline, model 후보, token 후보, sequence, parallel topology, checkpoint와 evaluation cadence다. theoretical FLOP를 wall-clock으로 바꿀 때 achieved kernel rate, communication tail, input stall, compile, checkpoint와 장애 복구 시간을 모두 더한다. 상태는 예약 quota, run progress, remaining token, measured throughput과 estimate generation을 가진다.

효과는 예상 loss interval, 성공 확률, peak memory, completion time과 비용 frontier다. 평균 throughput만 쓰면 root step, evaluation과 checkpoint burst를 놓친다. 고정 source는 profiler trace, job scheduler allocation, topology config, cost table와 estimator code를 immutable run ID로 묶는다. 고정 test는 작은 측정 run을 숨기고 완료 시간을 예측한 뒤 실제와 비교한다.

failure는 memory infeasible 후보 선택, deadline 상한 위반, p90 completion 과소추정, quota와 topology 불일치, 평가 시간을 training compute로 누락한 경우다. 장애율 민감도도 계산해 recovery cadence를 선택한다. 7장의 checkpoint 비용과 11장의 communication model이 frontier 입력으로 직접 연결된다.

## 13.6 elastic resume·overflow·종료 경계를 보존한다

elasticity가 바꾸는 것은 rank 수만이 아니다. schedule generation, attempted/committed clock, overflow skip과 endpoint 의미가 함께 이동하므로 각각을 독립 상태로 보존하고, 학습 종료 조건을 schedule 곡선의 끝과 분리한다.

### 13.6.1 elastic resume에서 schedule 세대를 보존한다

elastic 옵션은 허용 world size, batch 보정, token clock authority, partial accumulation 정책과 topology별 lr rule이다. rank 수가 바뀌었다고 step counter를 token counter로 환산해 덮어쓰지 않는다. 상태는 마지막 complete OptimizerStepID, global committed token, data cursor, scaler, scheduler generation과 topology generation을 가진다.

효과는 재개 후 sample lineage, denominator, applied lr, update RMS와 throughput이다. bitwise 동일성이 불가능해도 data와 clock의 의미는 동일해야 한다. 고정 source는 rendezvous, sampler reshard, counter reduction, checkpoint loader, scheduler와 optimizer commit protocol이다. 고정 test는 accumulation 중 rank kill, checkpoint 직후 resize, 빈 shard와 반복 resize를 주입한다.

failure는 token 중복·누락, partial accumulation 혼합, scheduler만 전진, stale topology generation, 다음 세 lr 또는 delta의 허용치 초과다. coordinator는 first failure와 peer cancellation을 연결한다. 7장의 durable cut과 12장의 optimizer certificate를 같이 검증해야 재개가 단순 재기동을 넘어 의미 보존이 된다.

**schedule 승인 dossier를 닫는다**

승인 문서는 clock 정의, warmup·decay 식, group multiplier, batch·length·mixture phase, overflow와 resume 정책을 가진다. 각 옵션은 읽고 쓰는 상태와 관측 효과를 가리킨다. source certificate는 trainer, data, loss normalization, scaler, scheduler, optimizer, checkpoint와 evaluation의 commit과 함수 좌표를 묶는다.

고정 시험은 순수 lr oracle, accumulation과 overflow, distributed token ledger, phase 경계, checkpoint resume, topology resize, scaling holdout과 wall-clock backtest다. 경계 비교, clamp 순서, denominator, generation을 바꾼 mutant가 실패해야 한다. 정확한 수열뿐 아니라 잘못된 구현을 거부하는 능력이 시험 강도를 결정한다.

failure dossier는 expected/applied lr, 마지막 complete clock, data lineage, scaler, optimizer generation, source blob과 rollback checkpoint를 기록한다. 자동 중단 조건은 clock drift, unknown fallback, partial commit, resume mismatch, update ratio 상한과 validation 비열등성 실패다. 좋은 loss가 invariant 위반을 면제하지 않는다.

승인은 검증한 model, corpus revision, sequence, precision, optimizer, hardware와 topology 범위에만 유효하다. 임의 checkpoint에서 이전 사건을 역추적하고 다음 세 data·lr·delta 사건을 예측할 수 있어야 한다. 6장의 재현성, 7장의 복구, 8장의 data lineage, 10장의 성능, 12장의 optimizer state가 같은 사건 ID로 연결될 때 schedule 효과를 독립적으로 해석할 수 있다.

**plateau와 restart를 평가 사건에 결합한다**

plateau scheduler 옵션은 관측 metric, mode, patience, threshold, smoothing, cooldown, 감소 계수와 minimum lr다. restart 옵션은 cycle 길이, 배수, peak 복원값과 clock 단위다. metric이 늦게 도착하는 비동기 evaluation에서는 어느 checkpoint의 결과가 어느 scheduler 사건을 바꾸는지 명확해야 한다. 최근에 도착한 값을 최신 model 값으로 오인하지 않는다.

상태는 evaluation request ID, 대상 checkpoint, metric generation, best value, bad-count, cooldown, cycle과 last applied lr를 가진다. 옵션은 이 상태를 갱신하고 다음 lr를 바꾸며, 효과로 update RMS와 향후 validation이 변한다. evaluation 실패나 timeout 때 bad-count를 증가시키는지 보류하는지 선언한다. 같은 결과를 재전송했을 때 idempotent하게 한 번만 반영해야 한다.

고정 source는 evaluation enqueue, result join, metric reduction, comparator, scheduler transition과 lr write 함수다. 고정 test는 개선, threshold 안의 정체, 정확한 patience 경계, NaN metric, out-of-order result, duplicate result, cooldown 종료와 restart 충돌을 손 상태 기계와 비교한다. 비교 부등호나 bad-count 증가 위치를 바꾼 mutant가 실패해야 한다.

failure는 checkpoint lineage가 없는 metric, 중복 반영, 과거 결과의 lr 변경, NaN을 개선으로 처리, minimum 위반과 resume 뒤 best/cooldown 차이다. 7장의 checkpoint ID와 evaluation artifact를 결합한다. dashboard에 표시된 반올림 metric이 아니라 scheduler가 실제 읽은 raw 값과 comparator 결과를 보존한다.

**여러 parameter group의 schedule 합성을 검산한다**

group 옵션은 base lr, multiplier, 별 warmup, freeze와 unfreeze clock, decay 제외와 layer-wise scaling이다. global scheduler 출력에 multiplier를 곱하는 위치와 optimizer group에 이미 들어간 lr을 다시 scale하는 위치를 하나로 정한다. 두 wrapper가 같은 multiplier를 적용하면 특정 group만 제곱 비율로 움직일 수 있다.

상태는 stable group ID, member parameter inventory digest, base recipe, phase, last lr와 unfreeze generation을 가진다. model wrapper가 parameter 순서를 바꿔도 group ID가 유지되어야 한다. unfreeze 시 global schedule의 현재 lr을 적용할지 별 warmup을 시작할지 명시한다. 옵션→상태→효과 연결은 group별 applied lr, update RMS, decay와 validation slice로 관측한다.

고정 source는 parameter classifier, group constructor, scheduler composition, freeze controller와 optimizer serializer다. 고정 test는 group 순서 permutation, tied parameter, 빈 group, late unfreeze, checkpoint resume와 multiplier 1 control을 포함한다. 모든 trainable storage의 합집합과 교집합을 검사하고 applied lr ratio를 oracle에 비교한다.

failure는 parameter 누락·중복, group ID 이동, multiplier 이중 적용, frozen state 전진, unfreeze 첫 lr 불일치와 공통 group delta 차이다. 12장의 parameter inventory와 같은 stable logical ID를 사용한다. optimizer 분류와 scheduler group이 서로 다른 snapshot을 읽으면 시작 전에 거부한다.

**overflow와 gradient clipping을 clock 계약에 포함한다**

옵션은 dynamic loss scale, growth interval, overflow 시 scheduler skip, gradient clipping norm, unscale 순서와 zero-gradient 정책이다. scheduler가 optimizer 호출 횟수를 세면 overflow로 parameter가 바뀌지 않은 사건에서도 decay가 진행할 수 있다. commit clock을 쓰면 실제 parameter commit과 함께만 전진한다. 어느 의미를 택했는지 recipe에 고정한다.

상태는 scaler value와 growth tracker, overflow flag, unclipped·clipped norm, scheduler clock, optimizer step과 commit generation을 가진다. 효과는 applied lr, skip 비율, update RMS와 warmup 실효 token 수다. overflow가 scale별로 다르면 같은 nominal schedule도 다른 parameter 궤적이 된다. scaling 비교에서 이 차이를 별 원인으로 보고한다.

고정 source는 autocast, loss scale, unscale, distributed overflow reduce, clipping, scheduler와 optimizer commit 순서다. 고정 test는 한 rank overflow, accumulation 마지막 microbatch overflow, 정확한 clip 경계, zero norm, scaler 성장 경계와 resume를 포함한다. overflow flag를 제거하거나 scheduler를 먼저 호출한 mutant가 실패해야 한다.

failure는 skip인데 parameter·state·clock 중 일부만 전진, rank별 overflow 불일치, clip 전 unscale 누락, scaler generation rollback과 다음 세 lr 차이다. 12장의 optimizer 원자성과 같은 OptimizerStepID로 묶어 혼합 commit을 차단한다.

**schedule 관측 자체의 정확도를 감사한다**

관측 옵션은 event sampling 비율, histogram bucket, metric aggregation window, rank selection과 retention이다. 로깅 비용을 줄이는 선택이 상태를 바꾸지는 않아야 하지만, 잘못된 callback 위치는 scheduler를 한 step 앞이나 뒤로 보이게 한다. requested lr, expected lr, optimizer에 write된 lr와 실제 delta에 사용된 lr를 별 필드로 둔다.

상태는 event sequence, clock snapshot, recipe generation, rank와 parameter group ID를 가진다. 효과는 missing-event 비율, join 성공률, telemetry overhead와 감지 지연이다. sampling된 event도 source trace와 checkpoint를 역추적할 수 있어야 한다. rank 0만 기록할 때 group별 값이 rank 간 같다는 별 assertion이 필요하다.

고정 source는 callback registration, scheduler call site, optimizer step, event serializer, aggregation query와 dashboard transform이다. 고정 test는 알려진 lr 수열을 실행해 raw event와 query 결과가 같은지 비교하고, duplicate·out-of-order·dropped event와 timezone 경계를 넣는다. 시각화의 smoothing 값으로 원본 applied lr를 대체하지 않는다.

failure는 sequence gap, generation join 실패, expected/applied 불일치, aggregation의 단위 혼합과 overhead budget 초과다. 경보 query도 version과 digest를 고정한다. query 변경으로 과거 run의 판정이 달라지면 새 audit generation을 만들고 원본 사건은 보존한다.

**제한된 pilot에서 scaling 결정을 검증한다**

pilot 옵션은 model scale, token budget, seed, data slice, tuning 예산, schedule 후보와 중단 규칙이다. 후보마다 다른 횟수로 lr를 탐색하면 schedule이 아니라 탐색 compute를 비교하게 된다. paired initialization과 batch lineage를 사용하고 fixed-token, fixed-compute와 fixed-wall-clock 결과를 분리한다.

상태는 trial recipe, source certificate, data cursor, scheduler와 optimizer generation, consumed compute와 best-checkpoint lineage를 가진다. 효과는 validation interval, update 안정성, achieved throughput, peak memory와 예상 frontier의 posterior 변화다. 작은 pilot의 결과를 관측 범위 밖 model에 점 추정으로 승계하지 않고 uncertainty를 갱신한다.

고정 source는 experiment sampler, training entrypoint, metric reducer, cost accounting, fit updater와 selection rule이다. 고정 test는 동일 trial을 resume해 다음 세 data·lr·delta를 비교하고, 숨긴 pilot의 loss와 wall-clock을 예측한다. trial 실패를 결과 표에서 삭제하지 않고 성공 확률 추정에 포함한다.

failure는 tuning budget 차이, data lineage 차이, 공통 control delta 불일치, 중단 규칙 위반, compute 누락과 holdout 오차다. 승인 후보는 앞의 ‘schedule 승인 dossier를 닫는다’ 항을 채우고 ‘compute frontier를 wall-clock 제약으로 변환한다’ 항에서 정한 deadline 안에 있어야 한다. 품질 점 하나만 좋은 후보는 승격되지 않는다.

pilot 종료 시 임의 checkpoint를 골라 source 함수, 적용 옵션, clock 상태, data lineage와 실제 lr를 역추적한다. 이어 같은 입력으로 다음 세 update를 dry-run하고 scheduler oracle, optimizer delta와 evaluation enqueue 사건을 비교한다. 고정 test ID와 failure invariant가 없는 관측치는 승인 증거로 세지 않는다. 성공하지 못한 seed, overflow가 잦은 scale, deadline을 넘긴 topology도 결과 집합에 남긴다. 이 폐쇄 절차는 좋은 trial만 선택하는 편향을 막고 scaling fit의 성공 확률과 uncertainty가 실제 운영 실패를 포함하게 한다. 범위 밖 model이나 corpus에는 새 generation의 pilot을 요구한다.

마지막 보고에는 요청 옵션과 적용 옵션의 diff, state schema와 source digest를 함께 넣는다. retry나 resume가 있었으면 중복 token, evaluation과 compute를 제거한 방식도 고정한다. 누락된 비용을 사후 추정할 때는 측정값과 모델값을 구분하고 interval을 제시한다. 의사결정자는 예상 loss뿐 아니라 실패 확률, rollback 시간과 다음 정보가치가 큰 측정 scale을 함께 선택한다.

최종 승인자는 clock certificate의 source 좌표와 failure replay 결과를 직접 대조한다.

**schedule family를 progress 함수와 clock mapping으로 분리한다**

learning rate는 base lr에 multiplier `f(p)`를 곱한 값이고 progress `p`는 clock state에서 계산된다. update clock이면 committed update/total updates, token clock이면 consumed valid tokens/target tokens다. 같은 cosine 식도 clock mapping이 다르면 다른 schedule이다. loop iteration을 암묵적 progress로 쓰지 않는다.

warmup은 `p<w`에서 0 또는 initial ratio에서 target까지 linear/other transition한다. constant는 warmup 뒤 1, linear decay는 남은 progress에 비례해 minimum ratio로 간다. cosine은 `min + (1-min)(1+cos(πq))/2`, inverse-sqrt는 warmup 뒤 기준 clock의 제곱근 역비례, WSD는 warmup-stable-decay 세 구간과 각 경계/shape를 가진다. 실제 구현의 endpoint·off-by-one과 minimum 처리 순서를 source에서 확인한다.

**수식 fixture**

clock `0,1,w-1,w,w+1,total-1,total,total+1`에서 multiplier를 FP64 독립 함수로 계산한다. warmup 0/1, total==warmup, minimum ratio와 horizon 밖 clamp/continue/error를 test한다. result를 본 뒤 endpoint 정의를 바꾸지 않는다.

**update·token·sample·compute clock의 보존식을 만든다**

update clock은 optimizer commit 수, token clock은 objective에 기여한 valid tokens, sample clock은 logical examples, compute clock은 FLOPs 또는 measured budget을 센다. padding, prompt mask, modality/expert drop, accumulation와 overflow 때문에 서로 단순 비례하지 않는다.

global token delta는 DP ranks/microbatches의 valid denominator 합이다. local tokens×world size 추정은 uneven packing에서 틀린다. committed update가 skip되면 token을 소비했더라도 update clock은 멈출 수 있다. replay 정책이면 token clock도 어떻게 처리할지 정한다.

**Clock property**

clock은 commit policy에 따라 monotonic하고 checkpoint round trip 뒤 next delta가 같아야 한다. duplicate/replayed batches를 billed compute와 objective tokens에서 구분한다. ledger의 UpdateID, DrawIDs, numerator/denominator와 scheduler event를 잇는다.

**Transformers·PyTorch scheduler 함수와 state_dict를 고정한다**

사용 checkout의 Transformers scheduler factory/enum, warmup/decay helper와 Trainer가 optimizer/scheduler step을 호출·load하는 source 좌표를 기록한다. PyTorch의 `LambdaLR`, cosine/linear/step/restart 등 실제 선택 class의 constructor, `step`, `get_lr`/closed form와 state serialization을 잇는다. library 이름만으로 semantics를 정하지 않는다.

state card는 last epoch/step, base lrs, current group lrs, total/warmup, cycles/minimum, internal counters와 call order를 가진다. optimizer state load와 scheduler construction/load 순서가 base lr를 덮는지 fixed fixture로 본다. param-group 추가/순서 변화도 검증한다.

**State failure**

scheduler state 누락, one-step rollback/ahead, total horizon/config mismatch, group permutation, constructor defaults changed와 optimizer lr stale를 넣는다. load 성공 뒤 current lr뿐 아니라 next three lrs와 parameter deltas를 uninterrupted reference와 비교한다.

### 13.6.2 overflow·skip·accumulation에서 두 clock을 분리한다

AMP overflow면 optimizer parameter/moments와 update clock을 보통 전진시키지 않는다. scheduler가 loop iteration마다 step하면 warmup/decay가 소모된다. GradScaler found-inf 합의와 scheduler call guard를 source/trace에서 확인한다. DP ranks가 같은 commit을 결정해야 한다.

gradient accumulation은 microbatch마다 token clock을 관측할 수 있지만 update clock은 window commit에서 한 번 증가한다. scheduler를 microbatch마다 부르면 horizon이 accumulation factor만큼 빨라진다. loss denominator와 clip/unscale 순서를 14·15장과 맞춘다.

**Skip fixture**

warmup 마지막 직전 overflow, all-ignored batch, gradient accumulation 중 마지막 microbatch failure와 optimizer exception을 넣는다. current/next lr, clock, optimizer state와 data cursor를 비교한다. replay/consume data policy를 manifest에 둔다.

**WSD·restart·resume를 schedule generation으로 관리한다**

WSD는 warmup, stable와 decay durations를 update/token 단위로 고정한다. decay 시작을 evaluation 결과로 동적으로 정하면 controller state와 decision evidence가 추가된다. stable 구간 연장은 same schedule이 아니라 horizon/config child generation일 수 있다.

restart/cycle scheduler는 cycle index, within-cycle progress, period multiplier와 peak/min lr를 state로 가진다. training process restart와 schedule restart를 혼동하지 않는다. checkpoint resume는 같은 cycle을 이어야 하고 intentional restart는 new RecipeID와 optimizer policy를 가진다.

**Boundary rehearsal**

WSD warmup/stable/decay 경계와 cosine restart 직전에 checkpoint하고 uninterrupted/resumed next values를 비교한다. horizon extension, early stop 후 resume와 total token target 변경을 각각 test한다. old state+new constructor를 조용히 합치지 않는다.

**batch-size/LR scaling과 gradient noise scale을 paired 실험으로 제한한다**

linear/sqrt lr scaling은 batch 변화에 대한 후보 규칙이지 보편 법칙이 아니다. global effective batch를 valid tokens/update, sequence/task mixture와 accumulation으로 정의한다. optimizer/momentum, normalization와 schedule clock을 고정하고 lr rule만 paired로 바꾼다.

gradient noise scale 추정은 microbatch gradients의 mean/variance와 sampling assumptions를 가진다. estimator source, sample count, parameter/tensor subset, dtype와 distributed reduction을 기록한다. nonstationary curriculum과 heavy-tailed gradients에서 uncertainty를 보고한다.

**Scaling failure**

batch는 같은데 valid token distribution/mixture가 다른 case, world size만 증가해 local batch/BatchNorm-like state가 변한 case, clipping/overflow가 달라진 case를 넣는다. throughput·loss만 보고 lr law를 승인하지 않는다. update norm, gradient norm/noise와 fixed-token evaluation을 본다.

**elastic world-size와 mixture/curriculum 변경을 clock migration으로 푼다**

world size 변화는 tokens/update, accumulation, collective/gradient scale와 data ordering을 바꾼다. target global tokens/update를 유지할지 update clock을 유지할지 선택한다. scheduler state만 그대로 load해도 future token-lr trajectory가 달라질 수 있다.

mixture/curriculum 변화는 token의 semantic weight와 sequence/valid ratio를 바꾼다. raw token clock을 유지하더라도 objective distribution이 달라진다. schedule change와 data change를 separate axes로 report하고 source-wise committed token mass를 기록한다.

**Elastic fixture**

DP 4→3에서 old checkpoint의 committed updates/tokens, next DrawIDs, accumulation와 target horizon을 migration한다. first three lrs, global gradients와 parameter deltas를 candidate policy oracle에 맞춘다. exact/statistical data resume를 구분한다.

**checkpoint compatibility와 schedule RFC의 필수 표**

checkpoint root는 optimizer/scheduler/scaler, update/token/compute clocks, schedule config/generation, data cursor와 world-size/mixture generation을 묶는다. scheduler file 하나가 deserialize된다는 사실은 objective trajectory compatibility가 아니다.

RFC 표는 current/candidate progress equation, clock, warmup/segments/horizon, param groups, overflow semantics, elastic policy, checkpoint migration, rollback와 paired experiment를 가진다. option→state→effect를 each row에 둔다.

**Compatibility matrix**

same horizon resume, horizon extend/shorten, update↔token clock, new group, WSD segment/restart, library class/version와 topology/data change를 exact/derived/reset/unsupported로 분류한다. derived mapping은 next lr sequence와 first update를 test한다.

rollback은 compatible parent checkpoint와 data/clock generation을 선택한다. candidate schedule로 진행한 optimizer trajectory를 old schedule에 계속 붙이는 것은 exact rollback이 아니다. warm continuation으로 기록한다.

**paired ablation과 운영 승인**

baseline/candidate는 initialization/checkpoint, data order/mixture, optimizer, global objective tokens, eval pipeline와 compute budget을 맞춘다. schedule family, clock, batch/lr rule 중 한 축만 바꾼다. multiple axes면 factorial/combined comparison이라고 명명한다.

failure suite는 warmup off-by-one, skip drift, token miscount, state one-step mismatch, WSD/restart boundary, param-group order, elastic world-size와 stale data generation을 독립 실행한다. expected first gate와 no partial commit을 확인한다.

최종 dossier는 source/functions, equations, clock ledger, lr trace, optimizer deltas, checkpoint migration, scaling/noise estimates, paired metrics/compute와 rollback을 가진다. 같은 RunID/UpdateID/RecipeID를 가리킨다.

독립 검토자는 checkpoint clock에서 next three lrs와 parameter deltas를 재생한다. 이어 world size 또는 mixture option 하나를 바꿔 state migration과 effect를 예측한다. source, FP64 schedule oracle, runtime events와 resume가 맞을 때만 scaling recipe를 승인한다.

**inverse-sqrt와 token clock의 dimension을 검산한다**

inverse-sqrt schedule은 clock 단위가 바뀌면 reference scale도 바뀐다. `lr(c)=A/sqrt(max(c,c0))`에서 `A`, warmup/reference `c0`와 clock unit을 함께 저장한다. updates를 tokens로 바꾸면서 같은 숫자 `c0`를 쓰면 전혀 다른 curve가 된다.

warmup continuity를 원하면 warmup endpoint와 inverse-sqrt branch의 값이 맞는지 FP64로 확인한다. 구현이 warmup step의 제곱근과 current step의 min 조합을 쓰는지 source에서 읽는다. `c=0`, warmup boundary와 huge token counts에서 finite/precision을 test한다.

**Unit migration failure**

updates→tokens mapping에 old average tokens/update를 사용하되 curriculum/packing이 변하는 case를 넣는다. exact future curve가 불가능하면 current lr continuity, remaining integral/budget 등 선택한 mapping objective를 명시한다. “그대로 resume”라고 쓰지 않는다.

**learning-rate 적분과 decay exposure를 장기 budget으로 본다**

parameter가 경험하는 update 크기는 lr 순간값뿐 아니라 gradient/preconditioner와 lr sequence의 누적에 달린다. schedule 비교에서 `Σlr`, `Σlr²`, warmup/stable/decay별 committed updates/tokens를 보조 장부로 둔다. 이를 convergence proof로 과장하지 않고 exposure 차이를 설명하는 진단값으로 쓴다.

decoupled weight decay의 누적 multiplier는 step별 `(1-lr_t λ)` 곱에 의존한다. schedule/horizon이 바뀌면 same decay coefficient도 total shrinkage가 달라진다. representative parameter의 adaptive delta와 decay delta를 11·12장 원장에서 비교한다.

**Exposure fixture**

constant/linear/cosine/WSD가 same peak/min/horizon일 때 sums와 decay-only scalar trajectory를 손계산한다. overflow/skip은 committed terms만 포함한다. scheduler bug로 lr은 dashboard에서 비슷하지만 one extra boundary step이 들어가는 것을 잡는다.

**parameter-group schedule 합성을 explicit multiplier로 표현한다**

optimizer group `g`의 lr를 `base_g × global_schedule(c) × group_schedule_g(c)`로 표현한다. layer-wise decay, warmup exemption, adapter/projector/encoder stage와 newly unfrozen group은 별 multiplier/state를 가질 수 있다. 실제 optimizer group order보다 stable GroupID를 쓴다.

Transformers Trainer/custom callback과 PyTorch scheduler가 둘 다 lr를 쓰면 double scheduling이 생길 수 있다. owner를 하나로 정하고 source call order를 확인한다. group 추가 후 scheduler base_lrs/state가 갱신되지 않는 failure를 넣는다.

**Group property**

두 groups의 known base lr와 multiplier에서 each step expected lr를 계산한다. state_dict group permutation, zero lr frozen group, unfreeze knot와 resume를 test한다. one group overflow/absent gradient여도 global commit/scheduler clock 정책을 적용한다.

**production monitoring과 drift response**

runtime은 requested/applied lr, update/token clocks, schedule segment/cycle, group multipliers, commit/skip reason와 data/world-size generation을 UpdateID마다 또는 sampled aggregation으로 기록한다. dashboard lr만 있고 source state가 없으면 resume drift를 조사할 수 없다.

expected oracle와 actual lr의 first mismatch, tokens/update drift, overflow cadence, mixture/length knot와 evaluation event를 같은 timeline에 둔다. float display rounding과 actual optimizer scalar를 구분한다. device/capturable lr tensor도 읽는다.

**Drift rehearsal**

scheduler call 하나 누락/중복, token counter rollback, stale group, world-size change와 restart counter를 주입한다. monitor가 threshold 뒤가 아니라 first event에서 alert하고 next checkpoint root를 publish하지 않게 한다. last complete parent로 dry-run한다.

수정 뒤 same clock ledger와 next lr/update oracle을 실행한다. threshold를 사후 완화해 old run을 PASS로 바꾸지 않는다. new mapping이면 child RecipeID와 paired ablation을 요구한다.

최종 운영자는 임의 UpdateID에서 raw valid tokens, global commit, scheduler state와 each group lr를 재구성할 수 있어야 한다. checkpoint에서 next events와 rollback parent까지 답할 수 있으면 schedule의 수학·코드·운영 clock이 닫힌다.

### 13.6.3 schedule endpoint를 training termination과 분리한다

total horizon에 도달했다고 process를 종료할지, minimum lr로 계속할지, error를 낼지는 scheduler와 trainer의 별 정책이다. scheduler가 horizon 밖 multiplier를 반환하는 방식과 training loop stop condition을 각각 source card에 둔다. one off-by-one이 extra optimizer commit을 만들 수 있다.

token target을 batch 중간에 넘을 때 full update를 commit할지 exact token budget을 위해 sample weights/stop을 조정할지 정한다. distributed ranks가 같은 boundary를 결정해야 한다. final checkpoint의 update/token clock과 data cursor를 기록한다.

**Endpoint failure**

horizon-1, exact horizon, one-over, overflow at final step와 resume-after-complete를 넣는다. scheduler lr, optimizer commit, evaluation/export/checkpoint events의 ordinal을 확인한다. complete run을 resume해 accidental extra update하지 않아야 한다.

**evaluation·checkpoint cadence와 schedule clock을 교차 검산한다**

evaluation/checkpoint를 updates, tokens, wall time 중 어느 clock으로 enqueue하는지 명시한다. overflow/elastic pause에서 evaluation cadence와 lr segment가 어떻게 관계하는지 event ledger로 본다. wall-time trigger가 same UpdateID를 중복 평가/저장하지 않게 idempotent key를 사용한다.

WSD decay 시작/끝, restart peak와 curriculum knot에 required evaluation/checkpoint를 둘 수 있다. trigger state도 checkpoint한다. resume 뒤 이미 완료한 EvalID를 반복할지 reuse할지 policy를 정한다. evaluation 결과로 controller가 schedule을 바꾸면 new decision generation이다.

**Cadence fixture**

checkpoint/eval boundary와 overflow, process death를 겹친다. one incomplete evaluation은 schedule state를 임의로 바꾸지 않아야 한다. root checkpoint가 어떤 evaluation/decision evidence를 포함하는지 manifest로 확인한다.

**blind clock audit**

한 reviewer는 config와 lr trace만 받아 schedule family, warmup/segments, clock와 group multipliers를 재구성한다. 다른 reviewer는 checkpoint state, UpdateID/token ledger와 source에서 next three lrs를 계산한다. 두 결과는 request/applied option diff와 같아야 한다.

**Blind negative copy**

test copy에서 warmup endpoint, token count, last-step, cycle index, group order와 overflow commit flag를 하나씩 바꾼다. reader/validator가 expected boundary에서 찾는지 본다. production artifact를 변경하지 않는다.

최종 dossier는 equation/source, clock certificate, optimizer/group state, data/world-size generation, evaluation/checkpoint events, paired ablation, failure와 rollback을 한 root 아래 둔다. `NOT_RUN` topology/mixture와 uncertainty를 남긴다.

새 library, optimizer, batch/sequence, mixture, world size와 horizon이 들어오면 previous PASS를 이름으로 상속하지 않는다. affected clock mapping과 checkpoint migration을 다시 실행한다. 같은 scheduler 이름보다 next UpdateID의 실제 lr와 durable state가 기준이다.

독립 reviewer가 parent checkpoint에서 target topology/data policy로 resume해 same declared trajectory를 재생하거나 allowed mapping 차이를 설명하면 scaling recipe를 봉인한다. 이 blind audit가 schedule tuning을 재현 가능한 training control로 만든다.

**production canary의 clock 왕복**

canary는 baseline checkpoint, fixed DrawIDs와 same optimizer를 유지한 채 candidate schedule만 바꾼다. requested/applied config, initial scheduler state와 first lr를 저장하고, warmup/segment boundary, normal commit, overflow skip와 checkpoint resume를 짧은 run에 포함하도록 축소 horizon fixture를 사용한다.

각 사건에서는 loop iteration, committed update, valid tokens, billed compute, group lrs, optimizer delta와 Eval/Checkpoint IDs를 기록한다. analytical oracle와 actual events가 일치해야 하며, 작은 fixture로 production horizon의 절대 성능을 주장하지 않고 clock semantics만 검증한다.

**Canary rollback**

lr mismatch, token mass drift, repeated evaluation, group divergence, non-finite 증가와 deadline regression을 trigger로 둔다. trigger가 발생하면 controller는 last complete parent를 선택해 scheduler/optimizer/data cursor generation을 함께 복원한다. model weights만 rollback하고 clock을 candidate에 남겨서는 안 된다.

rollback 뒤 same next three DrawIDs와 declared policy의 lrs/updates를 dry-run한다. exact data resume가 불가능하면 statistical continuation과 token-clock mapping을 새 child로 기록한다. candidate가 소비한 tokens/compute와 duplicated work를 report에 남긴다.

최종 승인자는 canary event 하나를 source scheduler function, clock ledger, optimizer scalar, parameter delta와 checkpoint root까지 추적한다. 이어 world-size 또는 mixture change를 가정해 migration 표를 적용한다. 빈 state나 unknown fallback이 없고 failure trigger가 민감할 때 production schedule을 승격한다.

승격 뒤 actual tokens/update와 skip cadence가 pilot bound를 벗어나면 같은 schedule name이라도 new operating regime다. affected scaling/noise estimate와 horizon exposure를 다시 계산한다. 이 유지 규칙이 lr curve를 dashboard 그림이 아니라 durable update control로 보존한다.

모든 재계산 결과는 RecipeID, ClockGeneration, UpdateID와 parent checkpoint에 연결한다. source default, total horizon, warmup, cycle 또는 group multiplier가 하나라도 달라지면 기존 trace를 덮지 않고 새 child evidence를 만든다. 독립 검토자가 같은 artifact만으로 동일한 next lr와 rollback 결론을 얻어야 최종 인수 기록이 닫힌다.

이 증거는 다음 optimizer·분산·종단 학습 장에서 schedule clock을 추측하지 않고, 직접 검증 가능한 입력 상태로 안전하게 재사용할 수 있게 한다.

## 13.7 schedule family와 추정기를 조건부로 선택한다

여기서는 앞의 실험을 recipe 선택 규칙으로 압축한다. 계산 예산, 데이터 반복, optimizer와 architecture 조건을 고정한 뒤 estimator와 schedule family를 나란히 비교해 어느 가정 아래 선택이 유효한지 밝힌다.

### 13.7.1 scaling 예산과 기본 schedule family를 조건별로 검산한다

**scaling law는 예산 배분의 관측 모델이지 학습률 공식이 아니다**

모델 크기, 학습 token과 compute에 따른 loss의 경험적 power law는 주어진 family·data·training regime에서 관측된 관계다. 이를 새로운 architecture·tokenizer·quality mixture에 그대로 적용하면 안 된다. fitted 범위, irreducible term, parameter 정의와 compute 산식을 보존한다.

compute-optimal 논의는 고정 compute에서 model parameters와 training tokens를 어떻게 나눌지 묻는다. learning-rate schedule은 선택된 run의 update 크기를 시간에 따라 제어한다. 둘은 예산과 dynamics에서 연결되지만 scaling exponent가 곧 LR scaling rule은 아니다.

training FLOPs를 단순 `6ND` 근사로 계산할 때 dense decoder의 조건과 forward/backward·embedding·attention, sequence, MoE active parameter 차이를 밝힌다. 실제 profiler compute, billed GPU time와 system inefficiency를 별로 둔다. active parameter와 total stored parameter가 다른 MoE를 dense 식에 그대로 넣지 않는다.

**fit 외삽 gate**

원 논문의 data point와 fit 범위를 digitized table 또는 공개 수치로 재구성하고 target run이 어느 축에서 밖에 있는지 표시한다. 작은 pilot의 loss·compute를 fit prediction과 비교하되 차이를 즉시 exponent 재추정으로 덮지 않는다. data quality와 optimizer·schedule 차이를 먼저 본다.

**Chinchilla식 직관을 현재 recipe에 옮길 때 조건을 다시 쓴다**

compute-optimal model/token 비율에 관한 대표 결과는 당시 model family, tokenizer, corpus와 training 설정에 조건부다. “parameter당 token 몇 개”라는 단일 숫자로 법칙을 축약하지 않는다. tokenization이 바뀌면 같은 원문도 token 수가 달라지고, high-quality 반복·curriculum은 token의 정보량을 바꾼다.

fine-tuning에서는 pretraining scaling law와 목표가 다르다. 이미 학습된 representation을 특정 task·behavior에 맞추며 overfitting·forgetting과 data quality가 지배할 수 있다. 2,000페이지 책의 실험 설계에서도 pretraining budget, continued pretraining, SFT와 RL 단계를 별 curve로 다룬다.

모델 size를 늘리면 batch·parallel topology, optimizer state, communication과 failure rate가 변한다. 이론적 compute allocation이 cluster에서 동일 wall-clock·cost optimum을 뜻하지 않는다. model FLOPs utilization, checkpoint overhead와 recovery loss를 포함한 effective compute를 본다.

**조건부 결론**

예산 후보마다 model, unique·repeated tokens, sequence distribution, data mixture, optimizer·schedule, hardware와 expected loss 범위를 표로 만든다. fit에서 추정한 값과 engineering constraint를 분리한다. 16장의 cluster capacity와 24장의 evaluation uncertainty를 연결한다.

**warmup은 초기 곡률·scale·동기화 불확실성을 흡수하는 구간이다**

초기에는 optimizer moments가 형성 중이고 activation·gradient scale이 빠르게 변하며 distributed kernel·loss scaler도 안정화될 수 있다. 작은 LR에서 시작해 올리는 warmup은 큰 초기 update를 제한한다. 그러나 warmup이 길수록 유효 고LR training budget을 소비하므로 무조건 길게 잡지 않는다.

linear warmup, constant plateau, exponential 또는 다른 형태는 endpoint와 derivative가 다르다. `c=0`에서 LR이 정확히 0인지 첫 update가 `peak/W`인지 implementation별 off-by-one을 확인한다. `last_epoch` 같은 API state를 실제 committed update와 연결한다.

warmup 길이를 update 수로 고정하면 global batch 변화 때 token 기준 warmup이 변한다. tokens, samples 또는 compute clock을 쓸지 recipe가 결정한다. variable packing에서는 actual valid target token counter가 필요하다.

**warmup failure probe**

첫 열 update에서 gradient norm, moments, update ratio, overflow와 loss scale을 저장한다. warmup 0, 짧은·긴 후보를 같은 data draws로 비교한다. 안정적이라는 인상보다 최대 update ratio, non-finite와 이후 validation을 판정한다.

**cosine decay의 경계를 식과 코드에서 맞춘다**

대표적 cosine multiplier는 progress `p∈[0,1]`에서 `r+(1-r)(1+cos(πp))/2`다. 여기서 `r`은 minimum/peak ratio다. warmup 뒤 남은 horizon을 분모로 쓰는지 total horizon을 쓰는지, endpoint를 포함하는지 source에서 확인한다.

한 step 차이는 긴 run에서 작아 보여도 checkpoint resume와 exact comparison을 깨뜨린다. warmup end, decay start, penultimate와 final update의 FP64 expected LR을 table로 둔다. horizon 밖 clamp·continue·error도 시험한다.

minimum LR이 0인지 양수인지에 따라 late training의 update와 decay exposure가 달라진다. final loss만 비교하지 않고 late-window progress, update ratio와 validation 변화 대비 compute를 본다.

**cosine state migration**

total horizon을 연장할 때 기존 curve를 재계산하면 현재 LR이 discontinuous할 수 있다. current LR continuity, original absolute curve 유지 또는 새 child schedule 중 목적을 명시한다. checkpoint의 clock·base LR와 new horizon으로 next three LR를 검증한다.

**WSD는 stable phase와 decay 시점을 예산 결정으로 만든다**

warmup-stable-decay schedule은 peak 근처의 stable LR을 오래 유지하고 마지막 구간에서 낮춘다. 장점 주장은 같은 token budget과 평가로 확인해야 한다. decay를 언제 시작할지는 termination budget과 연결되며, 갑작스러운 예산 연장에 대한 정책이 필요하다.

stable phase가 끝나기 전 checkpoint를 여러 개 두면 다른 decay horizon을 branch해 비교할 수 있다. parent weight·optimizer moments와 data cursor가 같아야 paired experiment가 된다. branch가 다른 data를 보면 schedule 효과와 sample variance가 섞인다.

decay shape는 linear, cosine 또는 다른 함수일 수 있다. WSD라는 이름만으로 exact LR를 알 수 없다. warmup·stable·decay length, endpoint, floor와 clock unit을 schema로 저장한다.

**budget extension**

decay 중 추가 compute가 생겼을 때 decay를 되감아 stable로 돌아갈지, 낮은 LR로 계속할지, parent stable checkpoint에서 새 child를 만들지 선택한다. weight만 되돌리고 moments·clock을 유지하는 혼합 상태를 금지한다.

**restart schedule은 cycle index를 checkpoint state로 가진다**

cosine restart나 주기 schedule은 progress뿐 아니라 cycle number, cycle length와 peak·minimum 변화를 가진다. restart에서 LR가 상승하면 optimizer moments와 data phase가 이전 cycle의 state를 그대로 가진다. 새 optimizer 시작과 동일하지 않다.

cycle boundary off-by-one, geometric length 증가와 maximum LR 감소 식을 손계산한다. scheduler state dict가 cycle index를 직접 저장하는지 global step에서 재구성하는지 확인한다. horizon·config 변경 뒤 mapping을 검증한다.

restart가 local minimum 탈출을 돕는다는 설명은 가설이다. 동일 data·compute와 seed 반복에서 non-restart baseline과 quality·stability를 비교한다. peak 재상승 때 update ratio, clipping·overflow를 관측한다.

**restart resume fixture**

boundary 직전, 정확한 boundary와 직후에 checkpoint를 저장한다. next three LR와 optimizer delta가 uninterrupted reference와 맞아야 한다. completed cycle event가 resume 뒤 중복 발생하지 않도록 idempotent key를 둔다.

**batch-size scaling을 gradient noise와 hardware efficiency 사이에서 결정한다**

larger batch는 update당 더 많은 sample을 평균해 noise를 줄이고 GPU utilization을 높일 수 있지만 update 수와 data freshness를 줄인다. critical batch size를 넘어가면 추가 batch가 optimization progress보다 parallel efficiency만 높이거나 generalization을 해칠 수 있다. task·training phase별로 측정한다.

linear LR scaling은 batch가 k배일 때 LR도 k배라는 가설이고 square-root scaling은 noise 관점의 다른 가설이다. AdamW와 momentum·normalization, clipping이 있는 실제 model에서 자동 성립하지 않는다. warmup과 beta의 token-time도 함께 조절하거나 통제한다.

variable sequence training에서는 examples/batch보다 valid tokens/update와 sequence-length distribution이 중요하다. long sequence 하나와 short sequence 여러 개는 attention compute·gradient correlation이 다르다. FLOPs/update와 tokens/update를 둘 다 기록한다.

**scaling ladder**

microbatch, accumulation, DP degree를 단계적으로 바꾸며 same global token batch와 changed batch를 분리한다. throughput, MFU, gradient noise proxy, update ratio, time-to-quality와 final quality를 본다. OOM·communication·straggler도 포함한다.

### 13.7.2 추정기와 schedule family를 같은 학습 조건에서 비교한다

**gradient noise scale은 추정기와 표본 비용을 가진다**

gradient noise scale을 추정하려면 서로 다른 sample gradient의 분산과 평균 signal이 필요하다. full per-sample gradient는 비싸므로 small/large batch gradient 차이, microbatch 통계나 근사 추정기를 사용할 수 있다. exact estimator, bias와 sampling cadence를 기록한다.

layer·parameter별 noise가 다르며 global norm 하나가 모든 group의 critical batch를 대표하지 않는다. embedding, norm, attention, MLP와 expert 표본을 본다. data mixture·curriculum과 loss denominator 변화도 추정치를 바꾼다.

추정값이 요동할 때 scheduler나 batch controller를 즉시 반응시키면 feedback instability가 생길 수 있다. smoothing, confidence interval, minimum dwell time과 fallback을 둔다. controller state는 checkpoint한다.

**추정기 검증**

작은 모델·batch에서 가능한 exact microbatch gradient statistics와 근사를 비교한다. DP reduction·accumulation, clipping 전후와 AMP unscale 위치를 확인한다. 추정 오차가 batch 결정에 미치는 민감도를 보고한다.

**token-based schedule은 data pipeline과 분산 합의가 필요하다**

유효 token clock은 padding·loss-masked token을 제외할지, prompt token과 target token을 어떻게 셀지 정의해야 한다. pretraining next-token, SFT response-only와 multimodal token에서 분모가 다르다. 6장·18장·21장의 loss contract를 재사용한다.

각 rank의 valid token 수가 다르면 optimizer commit 전에 global sum을 합의한다. accumulation 중 microbatch token을 누적하고 overflow skip이 clock을 전진시키는지 정책을 정한다. data consumed clock과 parameter-updated token clock을 별로 둘 수 있다.

elastic world-size에서 token throughput은 바뀌어도 target curve가 token progress 함수라면 LR 의미를 유지할 수 있다. 하지만 batch·noise와 wall-clock이 변하므로 dynamics가 완전히 같다는 뜻은 아니다. world-size generation을 기록한다.

**counter integrity**

padding mask 오류, duplicated batch, rank drop, overflow와 resume cursor rollback을 주입한다. global token ledger와 data shard digest가 first mismatch를 잡아야 한다. counter를 수동 보정하면 child generation과 이유를 남긴다.

**schedule 비교는 같은 면적보다 같은 학습 조건을 요구한다**

두 LR curve의 `Σlr`가 같아도 update 순서와 model state가 달라 결과는 같지 않다. 같은 peak·integral·horizon 중 무엇을 맞출지는 연구 질문에 따라 정한다. 하나를 맞췄다고 공정성이 자동 확보되지 않는다.

paired comparison은 initial checkpoint, optimizer state, data DrawIDs, batch·loss, clipping과 evaluation을 맞춘다. schedule만 다르게 한다. stochastic kernel과 data order의 seed 반복을 포함한다. time-to-quality와 fixed-token quality를 모두 본다.

early stopping이 있는 trial은 서로 다른 token을 소비하므로 survivor bias를 기록한다. 실패 run을 제외하지 않는다. hyperparameter tuning budget도 후보별로 비슷하게 둔다.

**결론 형식**

“cosine이 낫다” 대신 어떤 model·data·optimizer, token horizon, warmup·floor와 evaluation에서 어느 metric·cost가 얼마나 달랐는지 쓴다. 범위 밖 extrapolation과 미검증 topology를 명시한다.

**linear·polynomial decay는 단순하지만 progress 정의가 전부다**

linear decay는 warmup 뒤 peak에서 floor까지 일정 기울기로 내려간다. polynomial은 `(1-p)^q` 같은 power로 곡률을 바꾼다. `p`가 warmup 제외 decay progress인지 total progress인지, floor를 더하는지 source에서 확인한다. power가 1이면 linear reference로 돌아가야 한다.

endpoint 포함 여부와 integer clock 때문에 마지막 두 LR가 예상과 다를 수 있다. `warmup-1`, `warmup`, `horizon-1`, `horizon`을 손계산한다. horizon이 warmup보다 짧거나 0인 invalid config를 explicit error로 처리한다.

linear는 late phase에서도 일정 절대 감소를 유지하고 cosine은 끝에서 기울기가 완만해진다. 이 차이가 training quality에 미치는 영향은 model·budget에 조건부다. 같은 peak·floor·token horizon과 paired data로 비교한다.

**power sweep**

`q<1`, `q=1`, `q>1` 후보의 early·late LR exposure와 decay-only AdamW trajectory를 계산한다. power 하나만 바꾸고 warmup·floor를 고정한다. scheduler serialization과 resume next-LR parity를 포함한다.

**constant LR도 warmup·termination·decay와 결합된 recipe다**

constant schedule은 LR가 변하지 않는다는 뜻이지만 warmup 뒤 constant인지 처음부터 constant인지 구분한다. training horizon을 늘리기 쉽지만 decoupled weight decay와 optimizer noise가 같은 강도로 계속 누적된다. late-stage convergence와 overfitting을 평가한다.

continued pretraining이나 짧은 SFT에서는 작은 constant LR가 합리적 기준점이 될 수 있다. 그러나 기존 checkpoint moments, data shift와 target token budget이 다르면 scratch training의 값과 비교하지 않는다.

constant baseline은 복잡한 schedule의 이득을 검증하는 데 중요하다. 같은 peak가 아니라 합리적으로 tuned constant와 비교한다. time-to-quality, fixed-token quality와 sensitivity를 본다.

**연장 실험**

원래 horizon 뒤 동일 LR로 계속할 때 update ratio, validation과 decay exposure가 어떻게 변하는지 기록한다. 종료 checkpoint에서 resume한 branch와 uninterrupted branch를 맞춘다. “schedule state가 없으니 resume가 쉽다”는 주장도 optimizer·data clocks가 맞아야 한다.

**one-cycle 계열은 상승·하강과 momentum schedule의 결합이다**

one-cycle 방법은 LR를 낮은 값에서 peak로 올린 뒤 크게 내리고 momentum을 반대 방향으로 조정하는 형태를 포함할 수 있다. exact phase 비율, annealing 함수, initial·final division과 momentum cycle을 구현에서 확인한다. LR curve만 복사하면 같은 방법이 아니다.

Adam beta 또는 SGD momentum을 scheduler가 바꾸면 optimizer state transition이 추가된다. checkpoint에는 current phase와 beta·momentum 값이 들어가야 한다. parameter group이 서로 다른 beta를 쓰면 scheduler mapping을 검증한다.

큰 peak LR가 regularization처럼 작동한다는 설명은 target LLM training에서 실험으로 확인한다. overflow, clipping과 loss spike를 숨기지 않는다. short budget에서 유리한 결과를 장기 pretraining으로 일반화하지 않는다.

**cycle property**

phase boundary, peak와 final LR·momentum을 FP64로 계산한다. resume와 group permutation, overflow skip에서 committed clock 정책을 시험한다. optimizer moments의 수치 parity도 확인한다.

**metric 기반 schedule은 evaluation 지연과 noise를 제어해야 한다**

plateau에서 LR를 낮추는 scheduler는 validation metric, mode, patience, threshold, cooldown과 minimum LR를 상태로 가진다. metric을 언제 어떤 checkpoint로 계산했는지 EvalID와 ModelGeneration을 연결한다. asynchronous evaluation의 늦은 결과가 최신 model schedule을 바꾸지 않게 한다.

metric noise가 threshold 근처에서 LR를 반복 변경할 수 있다. smoothing, confidence interval 또는 최소 평가 수를 고려하되 exact controller를 기록한다. 여러 benchmark 중 어떤 metric·aggregation이 owner인지 정한다.

distributed training에서 evaluator failure나 missing slice가 있으면 controller가 임의 기본값으로 step하지 않는다. stale·partial evaluation을 거절하고 last valid decision을 유지하거나 명시적으로 중단한다.

**plateau failure fixture**

개선·동률·악화와 NaN metric sequence를 입력해 patience·cooldown·best state와 LR를 손으로 계산한다. checkpoint resume, duplicated EvalID와 out-of-order result를 주입한다. decision log가 재현되어야 한다.

## 13.8 source state에서 architecture별 scaling 규칙까지

문서의 옵션 이름만으로는 실제 state ownership을 알 수 없다. factory와 caller의 source 좌표에서 clock을 읽고 쓰는 함수를 찾은 뒤, dense·MoE·hybrid optimizer와 학습 단계마다 scaling 규칙이 왜 달라지는지 연결한다.

### 13.8.1 숨은 state와 clock을 source·elastic 조건까지 추적한다

**schedule-free라는 이름도 숨은 averaging·state를 확인한다**

일부 schedule-free optimizer 또는 training 방법은 전통적인 외부 LR decay를 줄이지만 내부 averaging, interpolation이나 train/eval state 전환을 가질 수 있다. “scheduler가 없다”를 clock·state가 없다는 뜻으로 해석하지 않는다. exact paper와 implementation을 분리해 읽는다.

base LR, warmup 또는 weight averaging coefficient가 남아 있을 수 있다. evaluation 전에 parameter view를 전환해야 하는 구현이라면 checkpoint와 inference export가 어느 view를 저장하는지 확인한다. training view로 평가해 품질을 잘못 보고하지 않는다.

AdamW+schedule baseline과 비교할 때 optimizer update 자체가 달라지는지, external curve만 다른지 구분한다. parameter state memory, kernel과 checkpoint 비용을 포함한다.

**view-state fixture**

작은 parameter sequence에서 train step, eval 전환, 다시 train과 save/load를 수행한다. 각 view의 parameter와 persistent state를 reference와 맞춘다. omitted transition이 metric을 바꾸는 negative test를 둔다.

**curriculum knot는 scheduler 독립 event이면서 공동 실험 변수다**

sequence length, domain mixture, task difficulty 또는 modality 비율이 특정 clock에서 바뀌면 gradient scale·noise와 compute/update가 달라진다. LR schedule이 같은 시점에 변하면 효과를 분리하기 어렵다. 가능하면 knot를 엇갈리거나 factorial ablation을 설계한다.

실제 production에서 동시 전환이 필요하면 joint RecipeID와 예상 state change를 명시한다. data generation, packing·loss denominator, LR segment, batch와 optimizer moments를 한 event ledger에 둔다.

curriculum event 뒤 tokens/update가 바뀌면 update-clock schedule의 token exposure도 달라진다. token clock을 쓰더라도 gradient noise와 wall time이 변한다. scaling 가정을 다시 계산한다.

**knot window**

전환 전후 고정 token window에서 data composition, gradient·moment, update ratio, throughput·memory와 validation slice를 비교한다. checkpoint branch로 LR-only와 curriculum-only 후보를 작은 규모에서 실행한다.

**compute 기반 clock은 GPU 종류와 효율을 분리해야 한다**

FLOPs 또는 billed compute를 schedule progress로 쓰려면 model theoretical FLOPs, actual executed FLOPs와 hardware-time cost 중 무엇인지 정의한다. padding, activation recompute, MoE routing, fallback kernel과 communication은 서로 다른 값에 영향을 준다.

theoretical token FLOPs는 topology가 바뀌어도 model work를 나타내지만 system inefficiency를 반영하지 않는다. GPU-hours는 장애·idle과 다른 GPU 세대의 성능을 섞는다. 비용 기반 controller는 가격·quota 변화를 추가 state로 가진다.

heterogeneous cluster에서 같은 GPU-hour가 같은 compute가 아니다. device type별 capacity와 profiler utilization을 기록한다. schedule이 hardware 교체 때문에 갑자기 전진하지 않도록 logical compute와 billed resource를 별 clock으로 둔다.

**compute ledger**

UpdateID마다 valid tokens, theoretical FLOPs, measured kernel FLOPs proxy, GPU-seconds와 billed cost를 연결한다. 누락·중복과 failure retry를 표시한다. 어떤 clock이 LR를 움직이는지 하나로 확정한다.

**elastic world size에서 global batch를 유지할지 바꿀지 결정한다**

rank가 줄거나 늘 때 microbatch·accumulation을 조정해 global tokens/update를 유지할 수 있다. 유지하면 LR clock은 가까울 수 있지만 accumulation latency·noise와 per-rank memory가 변한다. 유지하지 않으면 batch scaling 가설과 beta token-time을 재평가한다.

in-flight accumulation 중 membership이 바뀌면 partial gradient를 버리고 같은 data를 재시도할지 commit할지 정한다. 모든 rank가 동일 UpdateID와 token ledger를 가져야 한다. communicator generation과 scheduler generation을 연결한다.

elastic restart에서 scheduler state는 last complete checkpoint에서 복원한다. wall-clock 기반 progress를 그대로 쓰면 장애 시간이 LR를 전진시킬 수 있다. 의도 여부를 명시한다.

**membership fixture**

accumulation 중간 rank failure, boundary scale-out과 repeated batch를 주입한다. parameter update, token counter, next LR와 data cursor를 uninterrupted policy reference와 비교한다. 16·17·29장의 failure protocol을 재사용한다.

**schedule hyperparameter search의 누수를 막는다**

peak LR, warmup, floor, decay length와 family를 validation 결과로 반복 선택하면 evaluation set에 과적합할 수 있다. tuning validation과 final held-out benchmark를 구분하고 trial 수·결정 기록을 남긴다. 여러 seed와 task slice를 사용한다.

short-run proxy로 long schedule을 고를 때 ranking 안정성을 검증한다. warmup만 끝난 시점의 loss가 late decay 품질을 예측하지 못할 수 있다. multi-fidelity pruning이 특정 family를 체계적으로 불리하게 만들지 본다.

search budget은 후보마다 comparable compute를 주고 failed run을 포함한다. scheduler bug·NaN과 resource preemption을 동일한 low score로 뭉개지 않는다. invalid trial은 별 분류한다.

**selection report**

parent recipe, search space, priors, evaluation cadence, stop rule, seeds, consumed tokens·compute와 chosen rationale를 저장한다. final candidate는 새 data draw와 target topology에서 confirm한다.

**scheduler source를 wrapper에서 optimizer scalar까지 추적한다**

Transformers나 custom trainer의 scheduler factory가 어떤 PyTorch scheduler 또는 lambda를 만드는지 확인한다. config name→constructor args→lambda/function→`scheduler.step` caller→optimizer group LR mutation을 잇는다. 호출 시점이 optimizer step 전인지 후인지가 첫 LR와 off-by-one을 결정한다.

state dict에는 last epoch·step, base LRs와 method-specific counters가 들어갈 수 있다. wrapper가 별 global step으로 재구성하거나 load 뒤 즉시 step하면 값이 달라질 수 있다. fixed checkout에서 save/load path를 읽는다.

capturable optimizer의 LR tensor, compiled loop 또는 device-side schedule은 Python object만 보면 실제 scalar를 놓친다. profiler·runtime sample로 parameter update에 소비된 LR를 확인한다.

**source upgrade**

old/new revision에서 default args, caller order, state schema와 boundary LR를 semantic diff한다. symbol 이름이 같아도 endpoint와 warmup 식이 바뀔 수 있다. next-three-LR fixture와 checkpoint migration이 통과해야 한다.

**모델 폭·깊이 scaling은 같은 parameter 수라도 dynamics가 다르다**

hidden width, layer depth, head 수와 MLP ratio를 바꾸면 parameter 수뿐 아니라 activation scale, gradient path 길이, kernel shape와 communication이 달라진다. 같은 total parameters와 tokens로 맞춰도 동일 optimization 문제는 아니다. initialization·normalization과 LR parameterization을 함께 본다.

width scaling에서 특정 parameterization은 model 크기에 따라 LR·initialization을 조정해 feature learning dynamics를 보존하려 한다. exact theory의 가정, base model과 width multiplier를 확인한다. 모든 architecture·optimizer에 단순 `1/sqrt(width)` 규칙을 적용하지 않는다.

depth 증가에서는 residual accumulation, norm 위치와 gradient scale이 변할 수 있다. layer별 update ratio·activation RMS와 early instability를 관측한다. pipeline parallel stage와 checkpointing 때문에 system throughput도 바뀐다.

**shape ladder**

작은 base에서 width-only, depth-only와 balanced scale 후보를 만든다. parameter count, FLOPs/token, optimizer state, batch·LR·warmup과 tokens를 표로 둔다. same function parameterization을 주장하면 selected activation·gradient·update 통계를 scale별로 비교한다.

### 13.8.2 architecture·optimizer·학습 단계별 scaling 규칙을 분리한다

**μP류 parameterization은 이름보다 tensor별 scaling rule을 읽는다**

maximal update parameterization 계열은 width 변화에서 initialization, learning rate와 multiplier를 parameter 역할별로 조정해 hyperparameter transfer를 목표로 한다. embedding, hidden weight, bias와 output readout이 같은 규칙을 쓰지 않을 수 있다. base·delta width와 implementation coordinate를 확인한다.

standard parameterization checkpoint를 μP recipe로 단순 resume할 수 있는지 별 문제다. weight 값이 같아도 forward multiplier와 optimizer LR가 달라질 수 있다. config, module construction과 checkpoint schema를 함께 본다.

proxy model에서 tuned LR가 target width로 전이된다는 주장은 exact architecture family·depth·data와 optimizer에 조건부다. width 외 축을 동시에 바꾸면 transfer 근거가 약해진다. paired controls를 둔다.

**tensor role audit**

모든 trainable parameter를 μP role과 shape로 분류하고 expected initialization·LR multiplier를 계산한다. unknown·ambiguous role을 거절한다. actual group LR와 forward scale을 hook·source에서 확인한다.

**sequence length scaling은 compute·batch·position을 동시에 바꾼다**

sequence length를 늘리면 dense attention 계산과 activation이 크게 늘고 같은 HBM에서 microbatch가 줄 수 있다. global tokens/update를 유지해도 sequences/update가 줄며 document mixture와 gradient correlation이 달라진다. position distribution과 long-range supervision도 변한다.

length curriculum에서 short→long 전환 시 LR를 유지·조정할 근거를 실험으로 만든다. tokens/update, FLOPs/update, gradient norm·noise, update ratio와 validation을 본다. RoPE scaling이나 attention kernel 변경을 동시에 하면 별 ablation이 필요하다.

padding·packing 효율이 length별로 달라 actual valid token clock과 billed compute가 갈린다. schedule progress가 update 기준이면 long phase가 token·compute budget에서 과소·과대 노출될 수 있다.

**length knot fixture**

같은 token content를 다른 packing·sequence arrangement로 구성해 loss denominator와 gradient를 비교한다. position semantics가 달라지는 부분은 동일성을 주장하지 않는다. transition checkpoint와 next LR·optimizer state를 기록한다.

**optimizer 종류와 schedule을 독립 축으로 비교한다**

AdamW, Muon·matrix optimizer와 다른 방법은 합리적인 LR scale·warmup과 decay가 다를 수 있다. AdamW에 tuned schedule을 그대로 복사해 새 optimizer를 기각하지 않고, 각 방법을 적절히 tuning한 비교와 동일 schedule의 mechanistic ablation을 둘 다 제공한다.

hybrid optimizer는 matrix parameter와 scalar·embedding·norm group에 다른 update rule과 LR를 쓴다. global schedule multiplier 하나가 group별 base LR·normalization과 합성된다. 12장의 ParameterRoleID와 GroupID를 scheduler schema에 넣는다.

optimizer state warmup 또는 Newton–Schulz iteration cadence가 training LR phase와 상호작용할 수 있다. matrix update norm과 AdamW update ratio를 같은 layer·clock에서 관측한다.

**factorial 비교**

optimizer A/B와 schedule X/Y의 2×2 작은 실험으로 주효과와 상호작용을 본다. full target-scale factorial이 비싸면 proxy와 선택된 confirmation을 구분한다. token·compute와 tuning budget을 보고한다.

**weight decay schedule을 LR schedule과 명시적으로 합성한다**

AdamW에서 decay delta는 LR와 decay coefficient의 곱을 포함한다. coefficient를 constant로 두어도 effective decay는 LR schedule을 따른다. 별 decay schedule을 쓰면 `λ(c)`와 `η(c)`의 곱이 실제 수축을 정한다.

late phase에 decay를 유지·줄이거나 늘리는 선택은 generalization 가설이다. parameter group별 exclusion과 base coefficient를 고정하고 controlled experiment로 본다. adapter·norm·embedding과 expert가 다른 정책을 가질 수 있다.

누적 decay factor와 adaptive update 대비 decay 비율을 layer·group별로 기록한다. horizon 연장과 resume에서 `λ` state·clock을 보존한다. scheduler object가 LR만 저장하고 custom decay controller를 놓치지 않게 한다.

**decay oracle**

zero gradient scalar에서 expected `Π(1-η_t λ_t)`를 FP64로 계산한다. warmup, stable·decay, overflow skip와 group multiplier를 포함한다. actual parameter trajectory가 맞아야 한다.

**SFT·LoRA에서는 짧은 horizon과 작은 trainable 집합을 다시 해석한다**

fine-tuning은 pretraining보다 update·token horizon이 짧고 data 반복 가능성이 크다. warmup 비율을 관행대로 복사하면 유효 peak 구간이 지나치게 짧거나 길 수 있다. unique examples, epochs와 target token을 함께 기록한다.

LoRA factor의 초기 gradient dynamics, adapter rank·scaling과 base frozen 상태 때문에 적절한 LR가 full fine-tuning과 다를 수 있다. adapter, modules-to-save와 embedding group을 별 관측한다. schedule이 small tensor foreach/fused 성능에 미치는 영향도 step 수를 통해 비용화한다.

epoch-based schedule은 dataset size·filtering과 distributed sampler에 의존한다. streaming·mixture data에서는 epoch 의미가 약하다. update·token clock으로 canonicalize하고 UI의 epoch를 파생값으로 둔다.

**반복 경계**

data epoch restart와 schedule decay knot가 겹치는지 확인한다. sample order·augmentation, train loss와 held-out quality를 반복 횟수별로 본다. early overfit을 LR decay가 숨기는지 판단한다.

**preference optimization과 online RL에는 여러 schedule clock이 공존한다**

policy optimizer LR, reward/value optimizer, KL coefficient, rollout temperature와 sampling budget이 각자 schedule 또는 controller를 가질 수 있다. 하나의 global step 이름으로 합치지 않는다. PolicyVersion, RolloutBatchID와 각 committed optimizer update를 연결한다.

rollout 생성 속도가 learner보다 느리거나 빠르면 data freshness와 effective batch가 변한다. LR를 learner update로 움직여도 policy가 본 environment token clock은 다르다. stale rollout fraction과 policy lag를 관측한다.

adaptive KL controller는 observed KL에 따라 coefficient를 바꾸므로 metric 기반 schedule과 유사한 지연·noise 문제가 있다. reference·policy version과 evaluation window를 state로 저장한다. LR controller와 동시에 반응하면 feedback interaction을 본다.

**RL clock fixture**

rollout 지연, policy overflow skip, value-only update와 checkpoint resume를 주입한다. 각 schedule·controller의 next state와 PolicyVersion을 hand state machine과 비교한다. 19·20장의 logprob parity를 유지한다.

**multimodal schedule은 tower별 unfreeze와 mixture clock을 가진다**

vision·audio encoder, projector와 language decoder를 서로 다른 phase·LR로 학습할 수 있다. tower freeze·unfreeze, modality mixture와 resolution·frame length 변화가 gradient·compute를 바꾼다. group schedule과 data event를 joint schema로 둔다.

새로 unfreeze된 group의 base LR, warmup, optimizer state initialization과 global schedule multiplier를 정의한다. 이미 decay 후반인 global curve에 새 group을 넣으면 거의 학습되지 않을 수 있다. group-local progress가 필요한지 실험한다.

image patch·audio token 수가 달라 valid token clock의 modality weighting 문제가 생긴다. text token과 visual token을 1:1로 세는 것이 compute·learning progress에 적절한지 명시한다. sample·token·FLOPs clock을 병기한다.

**tower transition**

unfreeze 경계에서 ParameterID manifest, LR·moments, gradient norm, throughput과 modality slice quality를 확인한다. checkpoint resume 뒤 phase event가 중복되지 않게 한다.

## 13.9 관측·checkpoint·데이터 불확실성을 승인 계약으로 묶는다

승인자는 매끈한 loss curve보다 horizon 변경과 데이터 반복이 어떤 불확실성을 만들었는지 알아야 한다. metric, checkpoint closure, counterfactual failure 실험을 한 계약으로 묶어 같은 recipe라는 주장의 범위를 제한한다.

### 13.9.1 관측·checkpoint·horizon 변경을 승인 계약으로 닫는다

**schedule 관측 대시보드는 요청값보다 적용값을 보여 준다**

config의 peak LR와 scheduler가 계산한 requested LR, optimizer group에 기록된 LR와 실제 kernel이 소비한 device scalar를 구분한다. compiled/captured path에서 stale tensor가 있으면 Python dashboard만 정상일 수 있다.

UpdateID별 global·group LR, segment·cycle, valid tokens, overflow·skip, update ratio와 evaluation event를 연결한다. downsampling은 warmup·knot·restart boundary를 보존한다. 여러 group의 min/max만 보여 주지 않고 stable GroupID를 선택할 수 있게 한다.

expected analytical curve와 actual trace의 residual을 자동 계산한다. floating display 차이는 tolerance로 처리하되 one-step shift, group drift와 clock rollback을 탐지한다.

**대시보드 반례**

scheduler step 누락·중복, group reorder, stale device LR와 token counter duplicate를 주입한다. first wrong UpdateID와 source event로 drill-down되어야 한다. 경보를 끄지 않고 fixture를 회귀 suite에 남긴다.

**schedule RFC에 필요한 최소 표**

첫 표에는 schedule exact equation, segments, boundary와 clock unit을 담는다. 이어 optimizer groups, base LR·multipliers와 owner를 정리하고, data batch·token·compute mapping과 elastic policy를 연결한다. checkpoint state·resume·horizon extension도 별 표로 둔다.

후속 표에는 paired baselines, token·compute budget, metrics와 uncertainty를 담는다. overflow·failure·rollback과 monitoring을 함께 정리하고, source revision, runtime dispatch와 `NOT_RUN` 영역까지 명시한다.

모든 표는 RecipeID, ClockGeneration, UpdateID와 parent checkpoint를 공유한다. 서로 다른 run의 best result를 한 표에 섞지 않는다. option 변경이 어느 state와 effect를 바꾸는지 열로 표시한다.

**독립 승인**

reviewer가 equation과 checkpoint만으로 next three group LR를 계산하고 runtime trace와 맞춘다. data/world-size event 뒤 clock mapping과 rollback을 재구성한다. 결과가 맞을 때 schedule은 그림이 아니라 실행 가능한 제어 계약이 된다.

**scheduler 수치 정밀도는 긴 horizon과 device scalar에서 드러난다**

수백만 update·수조 token clock에서는 integer counter와 floating progress 변환의 정밀도를 확인한다. FP32로 거대한 token count를 표현하면 인접 값이 같은 수로 반올림될 수 있다. canonical counter는 충분한 폭의 integer로 유지하고 schedule 계산에서 FP64 또는 검증된 primitive를 사용한다.

cosine·power·inverse-sqrt의 transcendental 연산은 Python, CPU library와 device kernel에서 조금 다를 수 있다. LR 자체의 tolerance와 parameter update에 미치는 effect를 구분한다. capturable·compiled path가 FP32 tensor LR만 허용하면 reference와 장기 drift를 측정한다.

token counter overflow, negative rollback과 NaN multiplier는 optimizer 전에 거절한다. LR가 finite라도 base×global×group 곱에서 underflow·overflow할 수 있다. group별 applied scalar를 검사한다.

**long-horizon oracle**

초기, 경계, 중간과 매우 큰 counter를 FP64로 계산하고 target path와 비교한다. 연속 counter가 monotonic curve에서 역전·정지하지 않는지 property test한다. exact periodic restart는 cycle arithmetic overflow도 포함한다.

**모든 rank는 같은 schedule generation을 소비해야 한다**

일반 data-parallel training에서 replicated parameter는 같은 LR·optimizer step을 사용해야 한다. scheduler를 rank마다 독립 Python state로 두면 resume·exception이나 token count 차이로 갈라질 수 있다. authoritative clock과 config digest를 합의한다.

token-based clock은 local valid token을 all-reduce한 global sum으로 전진한다. rank 하나가 batch를 건너뛰거나 padding 계산이 다르면 mismatch를 optimizer commit 전에 감지한다. pipeline·expert rank가 서로 다른 parameter groups를 소유할 때도 global schedule generation과 local group mapping을 연결한다.

LR를 매 step broadcast할지 deterministic state로 재계산할지는 성능·복구 trade-off다. 어느 방식이든 sampled cross-rank digest와 boundary 강검사를 둔다.

**rank divergence fixture**

한 rank의 scheduler state, group order 또는 token counter를 1만큼 바꾼다. collective validator가 parameter update 전에 실패해야 한다. 잘못된 update 뒤 parameter all-reduce가 평균내어 증상을 숨기게 두지 않는다.

**scheduler checkpoint schema를 method-independent core와 확장으로 나눈다**

공통 core는 schedule type/version, base group LRs, current committed clock, horizon·warmup, floor와 RecipeID다. cosine·polynomial은 progress args, restart는 cycle state, plateau는 best·patience·cooldown, controller는 metric window와 decision generation을 추가한다.

optimizer state dict 안의 group LR와 scheduler state가 서로 다른 값을 가질 수 있다. load 순서와 authoritative owner를 명시한다. validator는 next expected LR를 계산해 두 상태를 교차 검산한다.

serialized Python object에만 의존하지 않고 사람이 읽을 수 있는 canonical schema와 migration version을 둔다. library class 이름 변경, field 추가·제거와 group reorder를 명시 변환한다.

**schema round-trip**

save→load→save canonical digest와 next-three-LR equality를 본다. missing optional·required field, unknown version, reordered group과 corrupt counter를 negative fixture로 둔다. pickle load 성공을 semantic 성공으로 간주하지 않는다.

**horizon 변경은 미래 곡선의 재설계다**

training 중 total token budget을 늘리거나 줄이면 progress denominator와 decay 시작이 달라질 수 있다. 과거 LR를 바꿀 수 없으므로 새 horizon을 처음부터 적용한 ideal curve와 exact 일치는 불가능하다. 현재 LR continuity, derivative, remaining integral 또는 original schedule 유지 중 목적을 고른다.

이미 decay 중인 run을 연장해 LR를 다시 올리면 optimizer moments·model state가 peak 재진입을 경험한다. stable checkpoint에서 새 branch를 만드는 대안과 비교한다. budget 단순 증가라고 부르지 않고 schedule fork로 기록한다.

축소에서는 abrupt termination, compressed decay 또는 parent checkpoint 재시작이 가능하다. 남은 compute와 quality·stability를 비교한다. final evaluation·checkpoint event도 재배치한다.

**horizon mapping table**

old/new clock, current LR·slope, remaining tokens, target floor와 mapping formula를 적는다. boundary·resume fixture와 paired pilot을 통과한 child RecipeID를 만든다. 이전 state를 덮어쓰지 않는다.

**early stopping은 schedule 비교의 censoring을 만든다**

validation 악화나 plateau로 run을 멈추면 후보마다 소비한 token·compute가 다르다. 최종 score만 비교하면 오래 학습한 후보와 일찍 중단된 후보의 비용·잠재력을 잘못 해석할 수 있다. stop rule, patience와 best-checkpoint selection을 함께 보고한다.

asynchronous evaluation 지연 때문에 stop 결정 뒤 추가 updates가 commit될 수 있다. 결정이 적용된 ModelGeneration과 wasted tokens를 기록한다. rollback to best는 optimizer·scheduler continuation과 inference export 목적을 구분한다.

early stopping metric을 hyperparameter 선택과 final 평가에 재사용하면 누수가 생긴다. tuning·selection·held-out set을 분리한다. multiple comparisons와 seed variance를 반영한다.

**stop/resume fixture**

stop trigger 경계에서 checkpoint failure, late metric와 resume를 주입한다. completed run이 accidental extra update를 하지 않고, continuation을 선택하면 새 child schedule·data cursor가 생성되어야 한다.

### 13.9.2 데이터 반복과 불확실성을 실패 실험으로 반증한다

**data 반복과 LR 감소의 상관을 해석한다**

finite SFT dataset을 여러 epoch 반복하면 later epoch의 loss 감소가 새로운 정보 학습인지 memorization인지 구분해야 한다. LR decay가 반복 경계와 맞물리면 overfitting을 완화하거나 단지 update를 작게 만들 수 있다. unique-example exposure와 held-out behavior를 본다.

streaming pretraining도 deduplication 불완전, mixture upsampling과 curriculum 때문에 effective repeat가 있다. raw token count만 아니라 document·cluster repeat와 source quality를 기록한다. 4·6장의 data ledger와 schedule clock을 조인한다.

이 조인은 `optimizer_step` 하나로 끝나지 않는다. checkpoint \(k\)마다 source·skill별 누적 valid-token 벡터 \(\mathbf n_k\), mixture version, selector apply 시점과 global consumed token을 schedule state에 붙인다. 두 run의 step과 LR가 같아도 \(\mathbf n_k\)가 다르면 scheduler 비교가 아니라 data intervention까지 섞인 비교다. resume 뒤 LR는 이어졌지만 selector loss window나 domain cursor가 초기화된 경우도 같은 이유로 동일 run이라 부르지 않는다.

같은 total tokens에서 더 많은 unique data와 반복 data의 schedule optimum이 다를 수 있다. schedule 비교가 data repeat 차이를 숨기지 않도록 DrawID와 mixture를 고정한다.

**repeat-aware plot**

sample visit count bucket별 loss·gradient norm과 update 시점 LR를 표시한다. rare·frequent source의 quality slice를 함께 본다. 개인정보·원문 대신 stable digest와 category를 사용한다.

**scaling 결과의 불확실성을 숫자로 남긴다**

scaling-law fit과 schedule experiment 모두 measurement noise, seed·data order, evaluation sampling과 system variance를 가진다. point estimate만 보고하지 않고 confidence interval, residual과 outlier 처리 규칙을 제공한다. 적은 scale point에서 exponent를 과도하게 정밀하게 쓰지 않는다.

여러 후보·metric을 탐색하면 우연한 best가 생긴다. trial count, selection procedure와 confirmatory run을 분리한다. 실패·preempted run의 제외 기준을 사전에 정한다.

system throughput variance는 training loss와 별이지만 fixed wall-clock 비교에 영향을 준다. hardware generation, straggler·recovery와 utilization을 기록한다. billed cost uncertainty도 별로 둔다.

**결정 여유**

두 후보의 차이가 uncertainty보다 작으면 동률 또는 추가 증거 필요로 판정한다. 복잡한 schedule의 운영 비용과 robustness를 tie-breaker로 포함할 수 있다. 작은 수치 차이를 보편적 우위로 포장하지 않는다.

**schedule failure를 장애 계층별로 분류한다**

수학 오류는 equation·boundary·clock mapping이 잘못된 경우다. 구현 오류는 call order, group mapping·state load와 device scalar가 잘못된 경우다. 분산 오류는 rank counter·commit이 갈린 경우다. 운영 오류는 checkpoint·evaluation·rollback event가 중복·누락된 경우다.

학습 실패는 schedule이 정확히 실행됐지만 target data·model에 부적절한 경우다. 이 다섯 종류를 구분해야 코드 버그를 hyperparameter sweep으로 덮거나 나쁜 recipe를 framework 탓으로 돌리지 않는다.

incident에는 first mismatch UpdateID, expected·actual LR, source node, affected groups, parameter delta와 data event를 넣는다. 최소 clock fixture로 축소한다.

**failure injection matrix**

warmup off-by-one, stale group LR, rank token mismatch, corrupt state, delayed plateau metric와 wrong horizon mapping을 독립 주입한다. detector·rollback과 회귀 test를 확인한다. 29장의 cluster failure와 결합한 복합 시험은 독립 시험 뒤 수행한다.

**독자가 직접 만드는 30-step schedule 실험**

30 committed updates의 축소 horizon을 정하고 warmup 5, stable 15, decay 10 같은 piecewise schedule을 손으로 계산한다. 두 parameter groups에 다른 base LR·multiplier를 둔다. update 8의 overflow skip과 update 20 직전 checkpoint를 포함한다.

training loop의 attempt, microbatch, valid tokens, successful UpdateID, each group LR와 parameter delta를 출력한다. uninterrupted와 resume를 비교한다. scheduler step을 optimizer 앞·뒤에 잘못 호출하는 negative variant를 만든다.

그다음 global tokens/update를 중간에 두 배로 바꾼다. update clock과 token clock schedule이 어떻게 달라지는지 그린다. 어느 mapping을 채택할지 목적과 trade-off를 쓴다.

**실습 합격선**

FP64 oracle, runtime applied LR, checkpoint next-three-LR와 decay-only scalar trajectory가 맞아야 한다. 실패 variant가 expected first event에서 탐지되어야 한다. 이 작은 실험이 대규모 run의 clock semantics를 검증한다.

**스케줄러를 모델 품질과 시스템 비용의 공동 제어기로 읽는다**

LR curve는 optimizer update 크기를 바꾸고, 그 결과 loss·quality와 non-finite·clipping에 영향을 준다. clock 선택은 data·batch·world size를 연결하고, schedule boundary는 evaluation·checkpoint와 cluster event를 조직한다. 따라서 단순 시각화가 아니라 학습 시스템의 제어면이다.

좋은 schedule은 평균 quality만 높이는 것이 아니라 config·state·resume가 재현되고 장애·budget 변경에서 행동이 명확해야 한다. 복잡한 controller의 작은 이득은 추가 state·failure mode와 비교한다.

**13장의 인계**

14장은 LR가 만드는 gradient·update가 저정밀 CUDA kernel에서 어떻게 표현되는지 다룬다. 15–17장은 clock과 state가 분산 owner·checkpoint에서 보존되는지를 다룬다. 18–25장은 fine-tuning·RL·평가의 서로 다른 budget과 controller를 연결한다.

독자가 임의 checkpoint에서 다음 LR, 다음 optimizer commit과 rollback을 계산할 수 있고 scaling 결론의 조건·불확실성을 설명할 수 있을 때 이 장의 목적이 달성된다.

**LR range test는 짧은 탐색이며 장기 schedule의 증명은 아니다**

LR를 짧은 run에서 지수적으로 올리며 loss가 개선·발산하는 구간을 보는 방법은 후보 범위를 찾는 도구다. 시작 checkpoint, optimizer moments, batch·data와 increase rate가 결과를 바꾼다. scratch와 pretrained fine-tuning을 분리한다.

loss smoothing이 lag를 만들고 큰 LR의 위험을 늦게 보여 줄 수 있다. raw loss, gradient·update ratio, clipping과 overflow를 함께 본다. divergence 뒤 weight가 손상된 run을 원래 checkpoint로 되돌리지 않고 계속 사용하지 않는다.

선택한 peak LR는 별 warmup·full-horizon pilot로 검증한다. range test의 한 batch distribution과 target mixture·length가 다르면 재평가한다.

**range fixture**

고정 data draws와 exponential LR 식을 저장하고 attempt·committed clock을 구분한다. stop threshold와 chosen region을 사전에 정한다. seed 반복과 invalid run을 보고한다.

**warm-start checkpoint는 schedule만 이어 받는 문제가 아니다**

pretraining checkpoint에서 continued pretraining·SFT를 시작할 때 optimizer moments·scheduler를 유지할지 reset할지 결정한다. data objective와 batch가 크게 달라지면 old moments가 새 gradient scale을 억제할 수 있다. 반대로 reset은 큰 initial update를 만들 수 있다.

weight-only warm start, full-state continuation과 selective moment reset을 별 recipe로 둔다. 각 경우 warmup 필요성과 LR를 controlled pilot로 비교한다. “resume”와 “새 phase 시작”을 용어부터 구분한다.

checkpoint의 consumed token을 새 schedule clock 0으로 매핑할지 global lifetime token으로 유지할지도 명시한다. evaluation·data generation과 parent lineage를 보존한다.

**first-update probe**

old/new data의 gradient, old moments, reset moments와 parameter delta를 손계산·실행으로 비교한다. 첫 수십 update의 update ratio·loss와 forgetting slice를 본다.

**annealing의 직관을 확률적 optimization과 과장 없이 연결한다**

학습률이 크면 stochastic update가 넓은 영역을 탐색하고 작아지면 국소적으로 안정된다는 비유가 쓰인다. 그러나 gradient noise는 batch·data·optimizer와 model state에 따라 변하며 물리적 온도와 동일하지 않다. 비유를 exact theorem으로 제시하지 않는다.

SGD의 단순 가정에서 나온 diffusion 직관과 AdamW의 adaptive preconditioning·momentum을 구분한다. LR, batch와 gradient covariance가 함께 effective noise를 정한다. quality 결과는 실험으로 확인한다.

late decay에서 training loss가 천천히 줄고 validation이 좋아지는 현상을 sharpness 하나로 단정하지 않는다. update norm, gradient noise proxy, Hessian·curvature 근사와 representation metric은 보조 증거다.

**직관의 사용법**

2D stochastic quadratic에서 LR·batch를 바꿔 trajectory 분포를 그린다. toy 결과가 실제 transformer에서 어떤 관측을 예측하는지 명시하고 layer metric으로 반증한다.

## 13.10 시스템 horizon에서 recipe 승인과 독립 검토까지

마지막 대절은 시스템 pause와 처리량이 effective horizon을 어떻게 바꾸는지 계산한 뒤 실제 recipe 선택으로 내려간다. 회귀 묶음과 독립 검토는 앞선 카드를 반복하는 대신 source·수치 oracle·resume 증거가 하나의 결론을 지지하는지 재생한다.

### 13.10.1 시스템 실행 조건이 effective horizon을 바꾸는 경계를 계산한다

**안정성 경계는 peak LR 하나보다 update와 곡률의 관계다**

단순 quadratic에서 gradient descent 안정성은 LR와 최대 곡률 eigenvalue의 관계로 설명할 수 있다. deep network·AdamW에서는 curvature, preconditioner와 stochasticity가 시간에 따라 변한다. 이론적 경계를 진단 직관으로 쓰되 exact guarantee로 과장하지 않는다.

loss spike 전 update-to-weight ratio, gradient·moment, local curvature proxy와 activation norm이 어떻게 변하는지 본다. 특정 layer가 먼저 경계를 넘을 수 있다. global LR를 내리기 전에 group·normalization·data anomaly를 확인한다.

sharpness-aware 방법이나 curvature 추정기를 쓰면 추가 forward/backward와 noise가 있다. scheduler controller와 결합하기 전 estimator 신뢰도와 delay를 검증한다.

**peak pilot**

peak LR 후보를 짧은 ramp로 시험하고 first non-finite·loss jump가 아니라 최대 stable update·recovery를 본다. rollback checkpoint와 stop guard를 둔다. target batch·dtype·topology에서 실행한다.

**gradient accumulation schedule은 microbatch가 아니라 commit에서 전진한다**

N개 microbatch의 gradient를 모아 한 optimizer update를 한다면 일반 LR schedule은 successful commit마다 한 번 전진한다. microbatch loop에 scheduler call을 넣으면 N배 빠르게 curve를 소비한다. code review와 event ledger로 확인한다.

variable accumulation은 OOM recovery·sequence length에 따라 N이 바뀔 수 있다. update clock은 유지되지만 tokens/update가 달라진다. token schedule은 실제 valid token 합으로 전진한다.

accumulation 중 process failure 뒤 partial gradient와 data를 버릴지 복원할지 정책을 둔다. scheduler는 incomplete commit에서 전진하지 않는다. data consumed와 applied token clock을 구분한다.

**accumulation property**

N=1·2·가변에서 동일 global batch를 재생해 LR와 gradient·parameter delta를 비교한다. floating reduction 차이는 budget으로 관리한다. overflow가 마지막 microbatch에 발생하는 case를 넣는다.

**pipeline parallel은 microbatch schedule과 LR schedule을 혼동하기 쉽다**

pipeline의 1F1B, interleaving 같은 schedule은 forward/backward microbatch 실행 순서를 뜻하고 learning-rate schedule과 다른 객체다. 한 global optimizer commit에 여러 pipeline microbatch가 기여한다. 용어와 clock을 분리한다.

pipeline flush, virtual stages와 gradient accumulation이 update latency와 tokens/commit을 바꾼다. stage failure에서 일부 backward가 완료되어도 atomic global commit 전에는 scheduler가 전진하지 않는다.

stage별 parameter group이 있어도 replicated global LR generation을 공유하거나 명시적 group policy를 쓴다. stage-local optimizer가 서로 다른 counter를 가지지 않도록 digest한다.

**pipeline boundary**

마지막 microbatch 직전 failure, overflow와 checkpoint를 겹친다. 모든 stage의 UpdateID, scheduler state와 data cursor가 last complete generation으로 복원되어야 한다. 15–17장의 event DAG를 사용한다.

**mixture weight 변경을 LR 효과와 분리한다**

domain mixture weight가 바뀌면 gradient 방향·scale과 evaluation composition이 달라진다. 같은 시점의 LR decay가 개선을 만들었는지 mixture가 만들었는지 paired ablation 없이 알기 어렵다. 두 option의 parent·child RecipeID를 분리한다.

mixture별 valid token, loss·gradient contribution과 group update를 기록한다. upsampled small corpus의 repeat와 contamination을 포함한다. schedule tuning validation도 fixed mixture를 사용한다.

mixture controller가 online metric에 반응하면 LR controller와 coupled feedback system이 된다. decision cadence, delay·bounds와 rollback을 설계한다.

**2×2 knot**

old/new mixture와 old/new LR segment를 작은 2×2 branch로 비교한다. full scale이 어렵다면 mechanistic gradient·update와 short quality, 선택 candidate의 long confirmation을 구분한다.

**MoE scaling에서는 active parameter와 expert load를 함께 본다**

MoE는 total parameter가 커도 token당 일부 expert만 활성화된다. compute scaling과 optimizer state memory, communication이 dense model과 다르다. active FLOPs, stored parameters와 routed token distribution을 별로 기록한다.

global batch 증가가 expert당 token 수를 늘려 grouped GEMM 효율과 routing noise를 바꿀 수 있다. LR scaling 결과가 optimizer dynamics뿐 아니라 expert utilization 개선에서 나올 수 있다. router·expert group metric을 분리한다.

expert load imbalance로 어떤 expert는 작은 effective batch를 본다. 하나의 global LR·beta가 적절한지 관측한다. auxiliary balancing controller와 schedule knot를 동시에 바꾸지 않는다.

**MoE scale card**

tokens/update, experts, top-k, capacity, expert token 분위수, all-to-all byte, router/expert LR와 update ratio를 둔다. dense-equivalent FLOPs 주장의 산식을 명시한다.

**저정밀 전환은 기존 LR schedule의 수치 조건을 바꾼다**

BF16, FP16, FP8·quantized training으로 바꾸면 activation·gradient 표현, loss scaling, optimizer state와 kernel이 달라진다. 같은 LR curve가 같은 mathematical update를 의도해도 rounding·overflow가 바뀐다. dtype 전환을 schedule-only 변경과 분리한다.

precision curriculum을 쓰면 전환 knot에서 scaler, amax/scale state와 compiled graph generation을 checkpoint한다. LR를 동시에 바꾸면 joint experiment다. 전환 전후 gradient·moment·update parity와 quality를 본다.

낮은 precision에서 late tiny LR update가 weight cast에서 사라질 수 있다. master weight·stochastic rounding과 update loss fraction을 측정한다. floor LR 선택에 영향을 줄 수 있다.

**precision knot fixture**

serialized gradient sequence를 old/new precision에 적용해 expected delta와 lost-update fraction을 비교한다. scale state resume와 overflow skip clock을 포함한다. 14장의 numerical ladder를 재사용한다.

**장애 비용을 포함한 effective training horizon**

계획 token budget과 실제 unique committed token, retry·duplicated token과 lost compute를 구분한다. 장애 복구가 잦으면 같은 GPU-hours에서 학습 progress가 줄고 wall-time schedule은 다른 curve를 소비할 수 있다. logical progress clock을 우선한다.

checkpoint cadence는 lost work와 I/O overhead의 trade-off다. schedule boundary·expensive data phase 근처에서 cadence를 조정할 수 있지만 controller state를 기록한다. failure rate와 save/load time으로 expected lost compute를 추정한다.

elastic scale-down이 길어지면 token throughput·batch가 바뀐다. deadline 때문에 horizon을 줄이는 결정은 새 schedule fork다. remaining quality uncertainty와 비용을 보고한다.

**failure-adjusted report**

committed, retried, duplicated, discarded tokens·FLOPs와 downtime을 event ledger로 계산한다. model loss curve는 committed clock, cost curve는 billed clock에 그린다. 둘을 하나로 섞지 않는다.

### 13.10.2 실전 recipe 선택의 순서

먼저 objective, data·model, global batch, token·compute budget과 target topology를 고정한다. AdamW 또는 target optimizer의 안정 LR 범위를 small golden run에서 찾는다. warmup과 simple constant·cosine·WSD baseline을 비교한다.

clock은 committed update보다 valid token이 더 안정적인지 판단한다. variable length·elastic batch가 크면 token clock이 유용하지만 exact counter 비용을 감수한다. group schedule, decay와 phase event를 명시한다.

후보는 boundary oracle, checkpoint resume, overflow·rank failure와 target throughput을 통과한다. quality는 fixed-token·time-to-quality와 held-out slice로 본다. 복잡한 controller는 simple baseline을 유의하게 넘어야 한다.

**선택 문장**

선택한 family, exact equation·clock, peak/warmup/floor/horizon, groups, batch·optimizer, data와 evidence 범위를 한 문장에 담는다. “cosine 사용” 같은 축약은 운영 config가 아니다.

**13장의 완료 판정**

독자는 schedule 이름 없이 임의 clock에서 모든 group의 LR를 계산할 수 있어야 한다. warmup·boundary·endpoint, overflow skip와 resume에서 next state를 재생해야 한다. update, token, sample, compute와 wall clocks의 mapping을 설명해야 한다.

scaling-law 결과의 fit 범위, model/token/compute 정의와 불확실성을 말할 수 있어야 한다. batch·width·depth·length·world size 변화가 optimizer 기억, gradient noise와 system efficiency를 어떻게 바꾸는지 구분해야 한다.

config 변경은 source caller, scheduler state, optimizer scalar, parameter delta, checkpoint와 monitoring effect로 이어져야 한다. failure injection과 rollback이 준비되어야 한다. 미실행 영역은 `NOT_RUN`이다.

**독립 event 재생**

parent checkpoint와 event ledger만으로 next three LR, update·token progress, evaluation·checkpoint trigger와 rollback parent를 독립 reviewer가 계산한다. runtime 결과와 맞으면 schedule control이 닫힌다.

**schedule 함수를 pure reference로 따로 구현한다**

training framework와 독립된 작은 FP64 함수가 config와 canonical clock을 받아 group LR를 반환하게 한다. production scheduler의 복사본이 아니라 문서화된 exact equation을 직접 표현한다. boundary table과 property test의 oracle로 사용한다.

reference는 warmup·segments, floor, cycles와 invalid domain을 명시한다. plateau 같은 metric controller는 입력 event sequence를 받는 state machine으로 만든다. mutable global 변수와 wall clock을 숨기지 않는다.

production trace를 reference에 넣어 residual을 계산하고 첫 mismatch를 찾는다. reference와 source가 함께 잘못될 가능성을 줄이기 위해 손계산 boundary와 독립 review를 둔다.

**oracle version**

equation·schema 변경마다 version을 올리고 old checkpoint를 old oracle로 재생한다. 출판 예제의 expected table도 같은 oracle digest를 가리킨다. 모델 학습을 실행하지 않고도 clock semantics를 검증할 수 있다.

**config validator는 실행 전에 모순을 거절한다**

negative warmup, horizon보다 긴 warmup, peak보다 높은 floor, invalid power·cycle, unknown clock과 group multiplier 누락을 검사한다. token schedule이면 counter source와 distributed reduction owner가 필요하다. plateau면 metric direction·patience와 EvalID owner가 필요하다.

resume config와 checkpoint schema를 diff해 allowed override를 분류한다. horizon 연장, group 추가와 family 변경을 silent overwrite하지 않는다. parent·child migration plan을 요구한다.

resolved config에는 defaults와 unit을 모두 출력한다. `warmup=1000`만 쓰지 않고 1000 committed updates인지 million tokens인지 표시한다.

**negative corpus**

서로 모순된 config 표본과 legacy alias를 유지한다. validator error가 actionable field·expected range와 migration hint를 제공하는지 확인한다. auto-correction은 child config에 명시적으로 기록한다.

**callback과 hook이 scheduler 순서를 바꾸는지 감사한다**

Trainer callback, gradient scaler, optimizer wrapper와 custom logging hook이 `scheduler.step`을 호출하거나 group LR를 직접 수정할 수 있다. main loop만 읽고 owner를 확정하지 않는다. repository 전체 caller와 mutation을 검색한다.

evaluation-based callback, early stopping과 unfreeze hook이 boundary에서 scheduler state를 바꿀 수 있다. event ordinal을 trace한다. duplicate callback registration이나 distributed rank별 실행도 확인한다.

logging hook이 `get_last_lr`를 읽는 것과 actual optimizer scalar를 바꾸는 것을 구분한다. dashboard 값이 한 step 늦는 문제도 caller order에서 생긴다.

**mutation watch**

optimizer group LR setter와 scheduler state mutation에 debug hook을 걸어 caller·UpdateID를 기록한다. 허용 owner 밖의 쓰기를 실패시킨다. production에서는 낮은 비용의 generation digest로 축소한다.

**LR가 0이어도 모든 training state가 멈추는 것은 아니다**

LR 0이면 parameter adaptive·decay delta가 0일 수 있지만 optimizer moments와 step이 갱신될 수 있다. scheduler clock, loss scaler, BatchNorm 같은 state와 data cursor도 전진할 수 있다. “동결”을 원하면 어떤 state를 멈출지 정의한다.

parameter group LR 0으로 freeze한 경우 weight decay가 LR에 곱해져 0인지 implementation을 확인한다. gradient 계산·communication과 optimizer state memory는 여전히 발생할 수 있다. `requires_grad=False`와 비용·state가 다르다.

warmup 시작의 LR 0 first update가 moments를 형성하는지 이후 dynamics에 영향을 준다. off-by-one fixture에 포함한다.

**zero-LR fixture**

gradient가 있는 parameter에 LR 0을 한 step 적용해 weight, moments, step·scaler와 scheduler를 기록한다. None gradient와 requires-grad false 대조군을 둔다. checkpoint resume parity를 본다.

**per-token·per-layer 적응형 LR는 control surface를 크게 늘린다**

일부 방법은 layer norm·update ratio, token difficulty나 curvature proxy에 따라 LR를 조정하려 할 수 있다. 이는 static group schedule보다 측정 noise, controller state와 feedback failure를 추가한다. exact formula와 bounds·cadence를 명시한다.

token별 LR라는 표현이 loss weight·gradient scaling인지 실제 optimizer scalar인지 구분한다. microbatch reduction 뒤 효과가 어떻게 합성되는지 손계산한다. distributed ranks가 같은 controller 결정을 내리는지 본다.

복잡한 controller는 simple tuned baseline, disabled controller와 stale/noisy metric failure를 포함해 평가한다. 작은 평균 이득보다 tail instability와 recovery를 본다.

**admission**

metric provenance, delay, finite·range, state checkpoint와 fallback이 없으면 adaptive controller를 거절한다. decision generation을 UpdateID와 연결한다.

**multi-cluster training의 schedule clock은 WAN 중단을 견뎌야 한다**

여러 클러스터가 data·pipeline 또는 비동기 역할을 나눌 때 authoritative optimizer commit이 어디서 발생하는지 명시한다. WAN partition 동안 각 cluster가 독립 LR clock으로 전진하면 model generation이 fork될 수 있다. 합의된 commit log가 필요하다.

active-passive recovery는 last durable UpdateID·scheduler state를 복제한다. active-active 방식은 gradient staleness·merge semantics가 별 연구 문제다. 단순 데이터 병렬처럼 설명하지 않는다.

지역별 hardware·throughput이 달라도 logical token·update clock은 canonical해야 한다. billed cost와 wall-clock은 locality별로 기록한다. data sovereignty로 batch mixture가 다르면 scaling dynamics도 달라진다.

**WAN partition rehearsal**

checkpoint metadata 전달 지연, duplicated commit과 stale scheduler state를 주입한다. split brain을 update 전에 막고 last common parent로 복구한다. 16·17·29장의 multi-cluster event를 연결한다.

**비용 최적화 schedule과 품질 최적화 schedule을 분리한다**

spot/preemptible 가격, energy와 deadline을 반영해 batch·world size나 horizon을 바꾸는 controller는 품질 dynamics와 비용을 동시에 제어한다. 싼 시간대에 더 많은 GPU를 쓴다고 LR를 자동 변경할 근거는 없다. logical batch·clock 정책을 먼저 정한다.

cost per token, committed compute와 time-to-quality를 보고한다. 재시작·checkpoint overhead와 failed work를 포함한다. nominal GPU-hour 가격만 비교하지 않는다.

deadline 때문에 decay를 앞당기면 quality trade-off가 생긴다. 원래 schedule과 compressed child를 paired pilot하고 uncertainty를 보고한다. business constraint와 optimization law를 혼동하지 않는다.

**decision frontier**

quality, wall time, cost, energy와 recovery risk의 Pareto 후보를 제시한다. 하나의 가중 점수 뒤에 판단을 숨기지 않는다. 운영자가 목표에 맞는 RecipeID를 선택하게 한다.

**schedule 문서의 코드 인용은 경계 의미를 보여 주는 데 쓴다**

전체 scheduler 파일을 옮기지 않고 exact multiplier, `step` call order와 state serialization을 보여 주는 짧은 부분을 인용한다. 고정 revision/path/symbol과 caller를 붙인다. line 번호는 보조 좌표다.

코드 뒤에는 입력 clock, branch, output LR와 mutated state를 한국어 실행표로 풀어 쓴다. framework wrapper가 추가하는 default·offset을 함께 설명한다. snippet 밖의 fallback을 범위로 밝힌다.

논문의 schedule 식은 기호 정의·적용 budget과 함께 요약한다. 공개 구현이 식과 다르면 확인된 차이와 issue·commit을 기록한다. 의도 추론은 사실과 분리한다.

**인용의 판정**

독자가 snippet만 보고 target config의 next LR를 계산할 수 없다면 필요한 caller·state 설명이 빠진 것이다. 반대로 단순 boilerplate는 제거한다.

**release canary에서 schedule 효과만 격리한다**

baseline과 candidate는 같은 checkpoint, optimizer state, fixed DrawIDs, model·precision·topology를 사용한다. schedule config와 필요한 migration만 다르게 한다. short horizon에 warmup·knot·overflow·resume를 축소해 포함한다.

analytical LR, actual group scalar, parameter delta, update ratio, loss·non-finite와 step time을 비교한다. 작은 canary는 장기 품질을 증명하지 않지만 semantic·운영 오류를 잡는다.

quality canary는 별 충분한 token budget과 held-out slice를 쓴다. production promotion은 clock canary와 quality evidence를 모두 요구한다. rollback parent와 trigger를 미리 둔다.

**promotion record**

source·binary, RecipeID, checkpoint, event ledger, boundary table, metrics와 `NOT_RUN`을 묶는다. config 한 줄만 배포 티켓에 남기지 않는다.

### 13.10.3 schedule 지식을 유지하는 회귀 묶음

unit suite는 equation, boundaries, invalid config와 state round-trip을 포함한다. integration suite는 optimizer call order, accumulation·overflow, group mapping과 device scalar를 본다. distributed suite는 rank clock, elastic membership과 atomic commit을 본다.

recovery suite는 horizon migration, evaluation controller, restart와 multi-cluster partition을 다룬다. quality suite는 simple baselines, scaling pilots와 uncertainty를 보존한다. performance suite는 scheduler overhead·sync와 compile/capture를 본다.

framework, optimizer, data counter, topology나 schedule config가 바뀌면 affected suite를 실행한다. 이전 PASS를 schedule 이름으로 상속하지 않는다.

**살아 있는 장**

incident fixture와 새 method를 같은 schema에 추가한다. 독자가 수학식에서 production event까지 다시 걸어갈 수 있어야 설명이 유지된다. 이 회귀 묶음이 2,000페이지 분량을 낡은 API 목록이 아니라 재검증 가능한 지식으로 만든다.

**스케줄 변경 전 다섯 숫자를 먼저 적는다**

현재 committed UpdateID, consumed valid tokens, 각 group의 applied LR, optimizer moment의 대표 RMS와 남은 token·compute budget이다. 이 숫자 없이 peak·warmup·decay를 바꾸면 현재 위치와 기대 효과를 설명할 수 없다.

그다음 old/new curve의 next three LR, 남은 `Σlr`, decay exposure와 endpoint를 계산한다. horizon 연장이라면 continuity 목표를 명시한다. group 추가·unfreeze라면 local progress와 state initialization을 추가한다.

**변경 전 반증**

LR 문제가 아니라 data mixture, gradient scale, clipping·overflow나 group mapping 문제일 수 있다. 같은 checkpoint의 고정 batch에서 gradient와 update decomposition을 확인한다. 잘못된 원인을 schedule로 덮지 않는다.

**schedule 식의 차원을 검산한다**

LR 자체는 optimizer update를 scaling하는 계수이며 clock `c`의 단위는 update·token·sample 또는 FLOPs다. `c/W` 같은 progress는 분자·분모 단위가 같아야 한다. inverse-sqrt의 amplitude는 선택한 clock 단위에 의존한다.

config 숫자에는 unit suffix 또는 schema field를 둔다. updates에서 million tokens로 migration하며 숫자만 복사하지 않는다. base LR×global multiplier×group multiplier의 각 항도 무차원·scale 역할을 구분한다.

**dimension test**

동일 물리 progress를 두 clock 표현으로 바꾸어 같은 LR가 나오는 mapping을 FP64로 확인한다. variable tokens/update에서는 단순 상수 변환의 한계를 표시한다.

**모델 카드의 training schedule 표를 검증 자료로 읽는다**

공개 model card는 peak LR, warmup, batch·tokens와 optimizer를 제공할 수 있지만 exact curve·clock·floor·group 규칙이 빠질 수 있다. 적힌 사실과 추정해야 하는 항목을 분리한다. repository config·training script와 고정 revision으로 교차 확인한다.

다른 모델의 recipe를 출발점으로 쓸 때 tokenizer·data, model scale·parameterization, batch와 hardware 차이를 표로 만든다. 숫자를 권위 있는 기본값으로 복사하지 않는다. 공개되지 않은 부분은 미검증이다.

**재현성 등급**

equation·config·source·trace·checkpoint state가 모두 있으면 강한 근거다. card의 단일 숫자만 있으면 후보 범위 근거로 제한한다. 이 등급을 본문 결론에 반영한다.

**scaling plot을 그릴 때 축과 분모를 숨기지 않는다**

loss 대 parameters, tokens, FLOPs, GPU-hours와 cost는 서로 다른 질문이다. log-log fit이면 base, confidence와 excluded points를 표시한다. compute 추정식과 active·total parameter 정의를 캡션에 둔다.

schedule plot에는 requested curve와 actual applied LR를 겹치고 warmup·curriculum·world-size·resume event를 표시한다. 여러 group이면 대표 하나만 골라 전체처럼 보이지 않게 한다.

**시각화 판정**

그림의 각 점이 RecipeID와 raw ledger로 역추적되어야 한다. smoothing·normalization과 missing run을 밝힌다. 아름다운 곡선이 source·state 검증을 대신하지 않는다.

**13장의 승인 증거 한 페이지**

schedule은 `LR=base_group×progress_function(canonical_clock)×group_modifier`로 시작한다. 그러나 canonical clock은 optimizer commit, valid token, data·world-size generation과 checkpoint에 묶여 있다. progress function은 warmup·stable·decay·cycle·metric controller의 exact state를 가진다.

scaling은 model·data·batch·compute를 바꿀 때 이 제어가 어떤 regime로 이동하는지 묻는다. scaling law, gradient noise와 hardware efficiency는 서로 다른 관측이며 조건과 불확실성을 보존한다.

**독립 검토 질문**

다음 LR를 계산할 수 있는가. 그 값이 실제 optimizer kernel에 소비됐음을 증명할 수 있는가. 장애·resume·horizon 변경 뒤 어떤 state가 이어지는가. 품질 차이가 schedule 때문임을 paired evidence로 보였는가.

네 질문에 source, oracle, runtime·checkpoint와 evaluation이 같은 답을 낼 때 schedule과 scaling 설명이 완성된다.

**최소 source 탐색 순서**

user config와 CLI parser에서 시작해 resolved training arguments, scheduler factory와 constructor를 찾는다. 이어 training loop의 optimizer·scheduler 호출 순서, overflow skip branch, state save/load와 metric controller caller를 읽는다. 마지막에는 optimizer group의 실제 LR scalar와 profiler event를 확인한다.

검색 결과의 symbol 이름만 나열하지 않는다. caller predicate, clock 입력, mutated field와 checkpoint key를 표로 만든다. wrapper가 framework default를 덮는지 확인한다.

**고정 좌표**

revision/path/symbol/semantic span과 작은 boundary fixture를 함께 보존한다. upgrade 때 줄 번호가 아니라 의미와 next LR를 diff한다.

**schedule 성능 overhead도 측정한다**

일반 Python scheduler는 비용이 작을 수 있지만 token all-reduce, metric controller, device scalar copy와 frequent logging이 host sync를 만들 수 있다. microstep이 짧은 작은 모델이나 대규모 rank에서는 상대 비용이 커진다.

CPU call time, synchronization, collective byte, compile graph break와 logging cadence를 profiler에서 분리한다. schedule 계산을 줄이기 위해 stale LR를 쓰지 않는다. 필요하면 device-side 계산이나 낮은 cadence를 exact semantics와 함께 설계한다.

**성능 gate**

기능을 끈 baseline과 비교하되 LR curve는 같게 유지한다. overhead 감소가 boundary·resume 검증을 깨지 않아야 한다.

**schedule과 evaluation checkpoint를 함께 보존한다**

best model을 선택할 때 그 checkpoint의 scheduler·optimizer state가 필요할지 inference weight만 필요한지 구분한다. training continuation용 root와 deployment export를 별 artifact로 둔다. best weight에 final scheduler state를 잘못 붙이지 않는다.

evaluation result는 정확한 ModelGeneration, UpdateID, token clock과 data·template version을 가리킨다. delayed evaluation을 최신 schedule decision에 오접속하지 않는다.

**선택 재생**

selection rule을 raw EvalIDs에 다시 적용해 같은 checkpoint가 선택되는지 확인한다. tie·missing slice와 metric version 변경을 처리한다.

**다음 장으로 넘기는 수치 계약**

13장이 14장에 넘기는 것은 LR 숫자만이 아니다. optimizer가 적용할 gradient·update 규모, precision phase, overflow·loss-scale history와 expected boundary가 함께 간다. 저정밀 kernel은 이 값을 표현·누적하고 commit해야 한다.

LR가 작아지는 late phase에서는 update가 weight dtype에서 사라지는지, FP8 scale·FP16 underflow와 stochastic rounding이 필요한지 본다. peak phase에서는 overflow·saturation과 loss scaling을 본다.

**handoff fixture**

warmup peak, stable 중앙과 late floor의 대표 gradient·LR를 고정 tensor로 저장한다. 14장은 dtype·kernel별 parameter delta와 overflow decision을 FP64 reference에 맞춘다. schedule과 precision을 분리하면서도 같은 UpdateID로 연결한다.

**변경 후 첫 세 update를 반드시 본다**

schedule migration, resume나 group 추가 직후에는 첫 세 committed update의 requested·applied LR, moments, adaptive·decay delta와 parameter checksum을 uninterrupted 또는 declared child reference와 비교한다. 첫 forward만으로는 scheduler·optimizer state 오류를 볼 수 없다.

overflow가 끼면 attempt와 commit을 나누고 scheduler 정책을 확인한다. distributed rank별 group scalar digest도 맞춰야 한다. 이 짧은 검사가 장기 run을 잘못된 clock으로 소비하는 일을 막는다.

**실패 조건**

한 값이라도 설명되지 않으면 training을 계속해 평균 loss로 덮지 않는다. parent checkpoint와 migration 표로 돌아간다.

### 13.10.4 독립 검토자가 남기는 결론

결론에는 선택 schedule의 exact equation·clock·groups, model·data·optimizer·batch, token·compute horizon과 source revision이 들어간다. boundary·resume·failure evidence와 quality·cost uncertainty를 함께 적는다.

지원하지 않은 horizon 변경, topology, dtype나 data mixture는 미검증으로 남긴다. 다른 run의 유사한 곡선을 근거로 보간하지 않는다.

**봉인**

같은 artifact를 받은 다른 검토자가 다음 LR, endpoint, rollback과 적용 범위를 동일하게 판단하면 13장의 지식은 재현 가능하다. 그 전에는 완성 주장을 하지 않는다.

최종 ledger는 수식, source, resolved config, optimizer groups, clock events, checkpoint와 evaluation을 하나의 RecipeID로 묶는다. 그림과 본문에 나온 숫자도 이 ledger로 역추적한다. 새 framework·CUDA·optimizer나 data policy가 들어오면 영향받은 boundary와 migration을 재실행한다.

이 원칙을 지키면 schedule tuning은 감각적인 곡선 선택이 아니라 예산, 수치 dynamics, 분산 상태와 품질 증거를 함께 다루는 반복 가능한 공학이 된다.

모든 결정은 다음 update를 예측하고 재생하며 실패했을 때 안전하게 되돌릴 수 있어야 비로소 실제 대규모 학습의 제어 계약이 된다. 14장에서는 이 시계가 overflow 때문에 commit되지 않은 update를 어떻게 다뤄야 하는지, BF16·FP8 표현과 loss scaler의 실제 상태 전이에서 검증한다.

[설정한 mixture가 실제 손실 질량이 되기까지](../labs/06-mixture-realized-mass-lab.md)는 curriculum phase와 optimizer commit clock이 같은 draw를 어떻게 다르게 해석하는지 보여 준다. warmup에서 읽은 SFT 행도 해당 update가 실패하면 커밋 손실 질량은 0이며, 재개 시에는 다음 단계 ID·draw cursor·source별 exhaustion을 함께 복원해야 schedule과 데이터 시계가 다시 맞는다.
