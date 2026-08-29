# Playbook 02. loss plateau

## 실행 순서

### 관측
1. 유효 label/token 분모, LR, skipped step, gradient norm, parameter delta를 확인한다.
2. 고정한 32개 표본에 overfit시켜 model이 작은 데이터조차 학습하지 못하는지 확인한다.
3. task·modality·domain별로 loss와 realized mixture를 나누어 본다.
4. optimizer group별 trainable count와 frozen delta를 검사한다.

## 분기

### 판정
- gradient=0이면 mask/detach/frozen group, gradient>0·delta=0이면 optimizer/AMP skip, delta>0·loss 고정이면 LR/data noise/objective를 본다.
- train loss만 내려가고 eval이 고정되면 contamination 없는 eval renderer와 overfit을 의심한다.

### plateau를 수치로 정의한다

그래프가 평평해 보인다는 인상만으로 incident를 열지 않는다. 고정된 token 창에서 loss slope와 표준 오차, 시작·끝 loss의 차이, learning-rate 변화, valid target 수를 함께 기록한다. sample step이 아니라 실제 loss 분모에 들어간 token을 학습 진행의 시계로 삼는다. packing·padding·assistant mask가 달라지면 step 수가 같아도 실제 학습량은 달라지기 때문이다.

기준 구간은 가능하면 같은 recipe의 정상 run으로 잡는다. 없다면 초기 구간의 expected random-loss, 작은 표본 overfit, 변경 전 checkpoint 재생을 삼각 대조군으로 쓴다. smoothing window를 늘려 noise를 지운 그래프와 raw microbatch loss를 둘 다 보존한다. smoothing된 선만 남기면 주기적 data mixture 문제와 rank 한 곳의 spike를 놓친다.

### 상태 경계를 순서대로 확인한다

**입력과 분모.** `input_ids`, labels, attention mask, sample ID의 체크섬을 기준 run과 비교한다. assistant target이 전부 `-100`이거나 EOS 하나만 살아 있지 않은지 본다. data mixture는 설정 가중치가 아니라 실제 소비 token과 loss-bearing token으로 집계한다. 특정 domain이 반복되거나 corrupt sample이 조용히 제거되면 전체 loss는 평평해질 수 있다.

**gradient.** 각 optimizer group과 대표 layer의 gradient finite ratio, L2 norm, zero fraction, cosine-to-previous-step을 본다. activation checkpoint·adapter·frozen tower 경계에서 `requires_grad`, `grad_fn`, optimizer membership을 함께 검사한다. global norm 하나는 작은 adapter gradient가 거대한 다른 group에 묻히거나, 특정 layer의 0 gradient를 숨긴다.

**update.** parameter checksum만으로는 작은 update를 읽지 못한다. group별 `||Δθ||/||θ||`, update-to-gradient ratio, weight decay 항, Adam moment norm과 step counter를 남긴다. AMP overflow로 `optimizer.step()`이 skip됐는지, accumulation이 끝나기 전에 gradient를 zero했는지, scheduler가 optimizer보다 먼저 진행했는지를 확인한다. 코드에서 step 함수가 호출된 사실과 parameter가 변한 사실을 구분한다.

### 32-sample overfit을 진단 장치로 쓴다

32개를 임의로 고르지 않는다. 짧은 표본과 긴 표본, special token이 있는 표본, 주요 domain과 문제 domain을 고르게 포함하고 ID를 고정한다. augmentation·dropout을 끄고 배치 순서를 고정하며, 필요하면 base full fine-tune과 adapter recipe를 교차한다. 목표는 특정 loss 숫자를 만드는 데 있지 않다. 어느 target token의 NLL이 줄지 않는지 찾아내는 것이 목적이다.

tiny set도 학습하지 못하면 data 다양성이나 일반화 문제가 아니다. mask, detach, optimizer, LR, 정밀도, loss 구현을 먼저 본다. tiny set은 빠르게 암기하지만 전체 run이 고정되면 data noise·mixture, capacity, curriculum, regularization을 본다. train loss는 내려가는데 정확히 같은 rendered prompt의 eval이 고정되면 inference template, generation option, adapter load, metric renderer의 parity를 점검한다.

## 복구 결정과 증거

### 최소 변경 실험

LR을 늘리는 것은 진단이 아니다. 기준 checkpoint에서 가설 하나만 바꾸고 같은 sample stream과 token clock을 재생한다. 예상한 중간 metric—예를 들어 adapter gradient norm, update ratio, 특정 token NLL—이 먼저 바뀌고, 그 뒤에 loss slope가 바뀐 경우에만 가설을 지지한다. sample filter, LR, warmup, clipping을 모두 바꾸어 loss가 내려갔다면 어느 변경이 필요했는지 알 수 없다.

임시 복구 기록에는 rollback checkpoint, 구성 diff, data cursor, 기대 slope, 자동 중단 조건과 담당자를 명시한다. 예상한 방향이 나오지 않으면 실험 기간부터 늘리지 말고 가설을 기각한 뒤 기준 상태로 rollback한다. 이렇게 해야 근거 없는 장기 run이 비용을 소모하고 다른 문제를 만드는 일을 막을 수 있다.

## loss 분모가 달라 만든 가짜 plateau를 먼저 제거한다

### 구현 좌표에서 읽는 기술적 이유

TorchTitan `b482babc5f1d5d718e1719a735f9a2d86d1b9aff`의 `torchtitan/trainer.py:855-874`는 각 microbatch의 `(labels != IGNORE_INDEX).sum()`을 누적하고 data-parallel mesh 전체에서 합쳐 `global_valid_tokens`를 만든다. `:901`은 이 값을 forward/backward 경로로 전달한다. 같은 커밋의 `tests/unit_tests/cpu/test_loss.py:191-239`는 유효 token이 8개, 4개, 2개인 세 microbatch의 loss 합을 총 14개 token으로 나눈 값이 token 수로 가중한 평균과 같다고 단언한다.

```python
total_loss = loss1 + loss2 + loss3
total_tokens = tokens1 + tokens2 + tokens3
global_avg_loss = total_loss / total_tokens
```

이 계산이 필요한 까닭은 microbatch 평균의 단순 평균이 각 microbatch에 같은 무게를 주기 때문이다. padding과 assistant mask 때문에 유효 target 수가 다르면 짧은 microbatch의 token 하나가 긴 microbatch의 token 하나보다 더 큰 영향력을 갖는다. 그 결과 실제 token NLL은 개선되는데 표시 loss의 기울기가 평평해지거나, 반대로 특정 mixture의 mask 비율 변화가 개선처럼 보일 수 있다.

### 최소 분리 실험과 판정선

문제 window를 sample 순서와 checkpoint까지 고정해 두 경로로 재생한다. A는 현행 logger의 loss, B는 rank별 `loss_sum`과 `valid_token_count`를 먼저 합친 뒤 한 번만 나눈 loss다. 둘의 차이는 부동소수점 허용 오차 안이어야 한다. 임계는 임의의 0.01이 아니라 고정 fixture의 FP32 reference와 사용 dtype에서 사전 측정한 `atol`·`rtol`로 version 관리한다. 유효 token이 0인 microbatch, rank마다 유효 token 수가 다른 batch, accumulation 마지막의 짧은 batch를 negative fixture로 둔다.

A만 plateau이고 B가 내려가면 optimizer를 바꾸지 않는다. logging·정규화 경로를 고치고 동일 token clock에서 다시 검증한다. A와 B가 모두 plateau이면 이 원인을 기각하고 gradient→update→objective 분기로 내려간다. 수정 후에는 `loss_sum`, local/global valid token, accumulation index, parameter delta가 한 `UpdateID`에 묶여야 하며, tiny overfit과 전체 mixture window에서 같은 판정이 재현되어야 한다.

## 종료 조건

### 통과
원인 가설이 32-sample overfit과 전체 run 모두에서 예측한 방향으로 metric을 바꾸어야 한다.

발행 전에는 plateau 구간과 정상 구간의 `GoldenBatchID`, valid-token clock, group별 gradient·update 집계, 32-sample token별 NLL, 통제 실험 diff를 IncidentID에 묶는다. 수정 recipe는 같은 window에서 loss slope가 사전 정한 한계를 넘고, eval 훼손과 nonfinite·OOM·throughput 회귀가 없어야 통과한다. 학습률을 높여 잠시 loss만 줄였거나 특정 domain을 제거해 쉽게 만든 경우는 복구가 아니다.
