# Playbook 01. loss NaN

## 실행 순서

### 격리
1. 마지막 finite step과 최초 nonfinite step을 특정한다. 자동 resume을 멈추고 해당 batch와 checkpoint를 보존한다.
2. loss보다 앞선 logits, attention output, normalization, embedding의 finite ratio를 layer 단위로 이분 탐색한다.
3. AMP scale, overflow, skipped step, unscale 전후 gradient norm을 확인한다.
4. 같은 batch를 FP32·dropout off·한 rank에서 재생한다.

## metric과 분기

### 판정
- input/label이 범위를 벗어나면 data/tokenizer 분기다.
- forward 최초 NaN이면 activation·mask·kernel, backward에서만이면 loss/gradient, step 뒤면 optimizer state와 epsilon/decay를 본다.
- FP32에서는 정상이고 저정밀만 실패하면 scale과 reduced-precision accumulation을 본다.

### 다음 step으로 넘어가기 전에 보존할 증거

최초 nonfinite를 발견한 worker가 즉시 다음 step으로 넘어가면 원인을 가리키는 상태가 사라질 수 있다. `RunID`, `UpdateID`, `CheckpointID`, global rank, data-parallel rank, microbatch index, sample ID, token 수, valid-label 분모, optimizer step과 accumulation index를 한 묶음으로 고정한다. 모델·optimizer·scheduler·scaler state의 checksum과 RNG state, autocast dtype, TF32 허용 여부, 활성화된 kernel backend도 남긴다. 민감한 원문은 공용 log에 복사하지 말고, 권한이 있는 sample resolver로 조회한다.

rank별로 `loss`, `logits_absmax`, activation finite ratio, gradient finite ratio, pre/post-unscale norm, loss scale, skipped-step flag를 수집한다. 평균만 보지 말고 min/max와 최초 실패 rank를 남긴다. 한 rank의 NaN이 all-reduce 후 전 rank로 퍼지면 집계 후 log만으로는 발원지를 찾을 수 없다. collective 직전에 지역 finite flag를 남겨야 한다.

### 최초 불일치를 찾는 이분 탐색

같은 checkpoint와 batch를 쓰고 dropout을 끄며 optimizer step을 막은 재생 경로를 만든다. embedding 출력, 각 block의 normalization 후, attention score·output, MLP gate·output, final logits, 개별 loss를 hook로 수집한다. 모든 tensor를 저장하지 말고 shape, dtype, finite count, absmax, 상위 몇 개의 index를 남긴다. 먼저 block 범위를 이분 탐색하고, 처음 깨지는 block 안에서 operator 단위로 내려간다.

attention score만 깨지면 Q/K norm, RoPE position, mask의 모든-key-차단 row, softmax 축과 accumulation dtype을 본다. MLP에서 처음 깨지면 gate 입력 범위와 activation, quantized/dequantized weight scale, fused GEMM accumulation을 본다. loss에서만 깨지면 target ID 범위, all-ignore row, label smoothing, vocabulary-parallel global max·sum-exp와 분모를 본다. backward에서 처음 깨지면 상위 gradient, loss scaling, gradient checkpoint recompute의 dtype·RNG parity를 추적한다.

### 한 번에 하나만 바꾸는 반증 표

| 통제 변경 | 결과가 finite로 바뀐 경우 지지되는 가설 | 아직 증명하지 못하는 것 |
|---|---|---|
| FP32 forward·backward | 저정밀 범위·accumulation·scale 경로 | 어느 operator가 원인인지 |
| fused kernel 비활성화 | fused/reference 경로 차이 | kernel 내 dtype·layout·mask 중 어느 계약인지 |
| 문제 sample 제거 | sample-dependent trigger | sample이 잘못됐는지, model이 정상 sample에 취약한지 |
| loss scale 감소 | overflow margin 부족 | scale이 원인인지, 이미 손상된 activation을 숨겼는지 |
| optimizer state 초기화 | moment·preconditioner 손상 | 저장 손상인지 정상 update가 만든 폭주인지 |

여러 옵션을 한꺼번에 바꾸지 않는다. FP32, dropout off, single rank, reference kernel을 동시에 적용해 NaN이 사라지면 네 가설 가운데 어느 것이 맞는지 알 수 없다. 기준 run에서 통제를 하나씩 교차해 NaN을 없애는 최소 차이를 찾는다.

## 복구와 재발 방지

### 변경·rollback 계약

원인이 data라면 sample부터 삭제하지 말고 schema/tokenizer gate가 왜 걸러내지 못했는지 고친다. kernel이 원인이면 문제 shape·dtype·mask를 최소 fixture로 남기고 안전한 backend로 fallback한다. optimizer state가 손상됐다면 최근 finite checkpoint로 rollback하되, 문제 update를 다시 적용하지 않도록 data cursor와 scheduler clock도 함께 복원한다. 변경 기록에는 구성 diff, 기대 metric, rollback trigger, 담당자와 유효 기간을 명시한다.

## AMP의 `found_inf`를 loss scale 숫자와 분리해 읽는다

### 고정 소스와 테스트가 보장하는 경계

PyTorch `3691693263d2b66a68867e39b7449876844e06cf`의 `torch/amp/grad_scaler.py:246-300`에서 `_unscale_grads_`는 gradient를 device와 dtype별로 모은 뒤 `_amp_foreach_non_finite_check_and_unscale_`에 넘긴다. 이어 `:363-373`의 `_maybe_opt_step`은 device별 `found_inf`의 합이 0일 때만 `optimizer.step()`을 호출한다.

```python
torch._amp_foreach_non_finite_check_and_unscale_(
    grads, per_device_found_inf.get(device), per_device_inv_scale.get(device)
)

if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
    retval = optimizer.step(*args, **kwargs)
```

따라서 loss scale이 줄었다는 관측만으로 “optimizer가 작은 update를 했다”고 해석하면 안 된다. nonfinite가 검출된 step은 update 자체가 생략될 수 있다. 즉시 보존해야 할 최소 상태는 `scale`, device별 `found_inf`, skipped-step flag, unscale 전후 gradient norm, optimizer step counter, scheduler clock이다. 이 가운데 optimizer counter는 그대로인데 scheduler만 전진했다면 NaN의 원인과 별개로 학습 시계가 어긋난 2차 사고가 생긴다.

같은 커밋의 `test/test_torch.py:5756-5793`은 finite, `inf`, sparse FP16 gradient에서 `_unscale_grads_`가 반환하는 flag를 각각 0 또는 1로 단언한다. 이 테스트가 닫는 범위는 nonfinite 검출 계약까지다. 실제 model의 어느 operator가 NaN을 만들었는지, 분산 collective 전에 어느 rank가 먼저 깨졌는지는 보장하지 않는다. 현장 회귀 test에는 이 assertion을 그대로 베끼는 대신 문제 dtype·shape·sparse 여부와 최초 실패 rank를 보존한 fixture를 추가한다.

### 판정 임계와 안전한 폐루프

진단 window에서는 `found_inf > 0` 또는 finite ratio가 1보다 작으면 즉시 실패로 판정한다. gradient norm의 경고선은 모든 model에 통하는 상수가 아니므로 정상 run의 같은 layer·token 구간에서 미리 정한 분위수로 둔다. 복구는 마지막 finite checkpoint, 동일 data cursor, 동일 RNG에서 시작하고 원인 후보 하나만 바꾼다. 문제 batch의 최초 nonfinite 위치가 사라지고 `found_inf == 0`, optimizer·scheduler counter의 동행, parameter delta 발생이 모두 확인되어야 다음 20-step 재검증으로 넘어간다.

## 종료 조건

### 통과
원인 옵션 하나를 바꾼 통제 run에서 같은 batch가 finite이고 다음 20 step도 finite여야 한다. batch를 버려 숨긴 경우 해결로 판정하지 않는다.

원래 checkpoint와 수정 checkpoint에서 문제 batch 직전까지의 logits·loss도 허용 오차 안에 있어야 한다. 문제 batch를 포함한 재현 test, 정상 batch 대조군, 저정밀·FP32 교차 test와 분산 rank test를 regression suite에 넣는다. IncidentID에는 최초 비정상 tensor, 지지·기각된 가설, 수정 commit, checkpoint lineage, 20-step 결과와 장기 monitor threshold를 연결해 다음 교대가 같은 증상을 처음부터 다시 풀지 않게 한다.
