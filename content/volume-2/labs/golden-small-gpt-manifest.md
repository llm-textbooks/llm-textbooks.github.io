# Golden lab A: 작은 GPT artifact manifest

이 lab은 실행 결과가 아니라 3장과 10장이 공유하는 실행 계약이다.

## 고정 대상

- repository: `karpathy/nanoGPT`
- commit: `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`
- 교육 config: `B=2,T=8,V=256,C=32,H=4,L=2,dropout=0,bias=false`
- source window: 각 행 9 tokens, `x=[:,0:8]`, `y=[:,1:9]`

## ID 사슬

`CorpusRevision → DocumentID → normalized-byte-offset → TokenizerRevision → token-offset → GoldenBatchID → RunID → CheckpointID`

## 확정된 입력 artifact

- `x`: `[[11,7,91,44,5,5,19,2],[3,31,8,8,4,77,9,2]]`, little-endian int64 SHA-256 `970b4f800242a5d576ad6ac6ab698cabd4814044635b075c26d6c1e1b4259c6c`
- `y`: `[[7,91,44,5,5,19,2,-1],[31,8,8,4,77,9,2,-1]]`, little-endian int64 SHA-256 `52e1bf56996f8e6b0ba80bdf768eddfd871b2ab3b1dac002af5cf75b72b532e5`
- config canonical UTF-8 SHA-256: `53bac8a03c9096d0e7791aa4b5da551d8892e19a311710dbea9d526e171d1ac8`
- valid labels: 14

Activation·gradient checksum은 `golden_tensor_probe.py` 실행 결과로 채운다. 현재 기본 Python 환경에는 PyTorch가 없어 `NotExecuted: missing torch` 상태다.

## 실행 등급과 환경 manifest

결과 보고서는 `Proposed`, `LocallyExecuted`, `ExternallyReproduced` 중 하나의 등급을 갖는다. 코드와 예상 불변식만 있으면 `Proposed`다. 고정한 환경에서 실제 stdout·exit code·tensor digest를 보존해야 `LocallyExecuted`로 올린다. 독립한 사람이 다른 환경에서 같은 artifact로 재실행하고 허용 오차 계약을 만족해야 `ExternallyReproduced`다. 미실행 상태에 임의의 checksum·loss·속도를 채우지 않는다.

환경 manifest에는 OS·architecture, Python, PyTorch, CUDA runtime/driver, cuDNN, GPU/CPU, deterministic option, TF32, matmul/attention backend, thread 수와 환경 변수를 적는다. CPU로 실행하면 CUDA 항목은 `NotApplicable`로 표시한다. package 이름만 적지 말고 lock file·wheel digest·source commit 중 재생에 필요한 좌표를 남긴다.

## 실행 절차

### 1. 입력과 코드를 검증한다

`golden_tensor_probe.py` 자체의 SHA-256, 이 manifest의 SHA-256, `x`·`y`·config digest를 실행 직전에 다시 계산한다. 하나라도 다르면 실행을 중단하고 어느 artifact가 바뀌었는지 확인한다. 같은 seed라도 코드·backend·dtype가 다르면 같은 RunID를 쓰지 않는다.

### 2. forward atlas를 만든다

probe는 `input_ids`→token/position embedding→정규화→Q/K/V→causal attention→residual→MLP→final norm→logits 순서로 tensor를 남긴다. 각 경계에서 shape·stride·dtype·finite ratio·digest를 본다. `B=2,T=8,C=32,H=4`이므로 Q/K/V는 reshape 전 `[2,8,32]`, head 분할 후 `[2,4,8,8]`이어야 한다. attention output을 다시 합친 후에는 `[2,8,32]`로 돌아와야 한다.

### 3. loss와 backward를 재구성한다

`-1`인 두 label은 loss 분자·분모에서 제외되어 valid label이 14개여야 한다. library mean만 믿지 말고 위치별 negative log-likelihood를 합산한 뒤 14로 나눠 비교한다. backward 후에는 각 parameter의 gradient shape·finite ratio·norm·digest를 남긴다. weight tying 때문에 embedding row의 gradient에는 input lookup 경로와 output classifier 경로가 합쳐진다.

### 4. 한 step과 roundtrip을 검사한다

선택한 optimizer·learning rate·weight decay·gradient clipping 설정을 manifest에 추가한 뒤 한 step만 수행한다. step 전후 parameter·optimizer state digest와 `||Δθ||/||θ||`를 남긴다. checkpoint를 저장하고 새 process에서 load한 뒤 같은 batch의 logits·loss·다음 update를 비교한다. 저장 직전·후의 파일 digest와 CheckpointID를 보고서에 넣는다.

## 필수 tensor 기록

각 tensor에 name, shape, stride, dtype, device, finite ratio, min, max, mean, RMS, checksum을 기록한다. 대상은 token/position embedding, 각 block의 norm/Q/K/V/attention output/MLP output/residual, final norm, logits, per-token loss, parameter gradient다.

## 의미 불변식

- `y[:,:-1] == x[:,1:]`
- token ID는 `[0,V)` 범위다.
- 미래 token 변경은 이전 위치 logits를 바꾸지 않는다.
- manual attention probability의 마지막 축 합은 1이다.
- embedding과 LM head는 같은 parameter storage를 쓴다.
- loss sum / valid-label count가 reported mean과 같다.
- resume run은 주장하는 resume 등급에 맞는 batch/state/parameter 비교를 통과한다.

## Negative control과 장애 주입

### causal mask와 label shift

마지막 input token만 바꾸어 이전 위치 logits이 변하지 않는지 본다. 이전 위치가 바뀌면 causal mask 또는 attention 축이 틀렸다. `y`를 한 칸 밀어 잘못된 label shift를 만들면 입력–target 불변식이 optimizer step 전에 실패해야 한다. 실패한 run의 loss가 계산됐다고 정상으로 승격시키지 않는다.

### denominator·tie·state 누락

무시 label 하나를 유효 label로 바꾸어 valid count가 15로 바뀌고 sum/mean 관계가 따라 바뀌는지 본다. LM head를 embedding의 copy로 바꾸어 storage tie를 끊으면 data pointer 불변식이 잡아야 한다. checkpoint에 optimizer, RNG 또는 sampler state를 하나씩 빼 roundtrip parity가 각각 어느 경계에서 실패하는지 확인한다.

### dtype·backend 교차

CPU FP32를 의미 oracle로 삼아 CUDA FP32, BF16/FP16, SDPA backend와 비교한다. bitwise equality를 무조건 요구하지 않고 tensor별 max/mean absolute error, relative error, logits KL, argmax agreement 허용치를 실행 전에 정한다. backend을 바꾼 뒤 tolerance를 늘려 결과를 맞추지 않는다.

## 결과 artifact와 판정

최소 결과 디렉터리는 `environment.json`, `input-manifest.json`, `tensor-atlas.json`, `gradient-atlas.json`, `checkpoint-manifest.json`, raw stdout/stderr, `verdict.md`를 갖는다. 각 JSON은 schema version과 자체 checksum을 갖고, top manifest가 그 checksum을 참조한다. 실행에 실패하면 나온 부분까지의 artifact를 보존하되 실패 이후 항목을 성공처럼 채우지 않는다.

통과는 입력 digest, shape·mask·denominator, finite forward/backward, weight tie, one-step update, checkpoint roundtrip, 선택한 backend tolerance, 모든 negative control이 함께 만족될 때만 가능하다. 실패 시 `verdict.md`에 최초 불일치 tensor/state, 기대값과 실제값, 재현 명령, 지지·기각된 가설, 다음 조치를 남긴다. loss 하나가 finite이거나 stdout이 생성됐다는 사실은 통과 증거가 아니다.

독립 재현자에게는 기존 tensor 숫자를 미리 보여 주지 않고 입력·코드·환경·허용 오차 계약만 전달한다. 재현 후에 두 결과 manifest를 tensor 이름과 state 경계별로 diff한다. 다른 hardware·backend의 차이를 단순 실패로 취급하지 않고, 어느 경계에서 첫 편차가 나타나 뒤쪽으로 증폭되는지를 관측한다. 예외 처리·tolerance 조정·실행 제약은 모두 최종 판정과 함께 보존한다.
