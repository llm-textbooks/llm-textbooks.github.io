# Lab 28. 단일 GPU golden run

이 lab은 실행 명세다. 저장소에 실제 로그가 추가되기 전 모든 결과 칸은 `실행 예정`이다.

## 목표와 입력

### 고정 입력

- `RunID`: config와 모든 입력 digest의 SHA-256
- `GoldenBatchID`: input/label/mask bytes의 SHA-256
- model/tokenizer/chat-template revision
- GPU UUID, driver, CUDA runtime, framework/container digest

## 실행 절차

### 1. 환경 snapshot

package freeze, loaded CUDA/NCCL library, GPU topology를 저장한다. 성공 조건은 mutable tag가 남지 않는 것이다.

### 2. CPU preflight

token ID 범위, special token, template round-trip, shifted label, ignore mask, 유효 label 수를 assert한다. 실패하면 GPU 단계로 가지 않는다.

### 3. forward probe

embedding, 첫/중간/마지막 block, logits와 unreduced loss에 hook을 걸어 shape/dtype/finite/min/max/norm/checksum을 JSONL로 쓴다. hook이 graph를 바꾸지 않도록 detach한 축약 통계만 복사한다.

### 4. backward와 step

AMP라면 scale과 overflow를 저장한다. unscale 뒤 group별 gradient norm, clip 전후, skipped step을 기록한다. step 전후 parameter checksum과 delta norm을 비교한다.

### 5. accumulation 등가성

같은 sample 순서로 microbatch `K`회와 큰 batch 한 번을 실행한다. dropout을 끄고 loss denominator를 통제한다. gradient max error가 사전 tolerance를 넘으면 reduction과 scaling을 조사한다.

### 6. checkpoint interruption

5 step 기준 run A와 3 step 저장 후 2 step 재개 run B를 만든다. 첫 resume batch ID, LR, scaler, RNG draw, loss와 final parameter를 비교한다.

## 판정표

| 관찰 | 통과 | 실패 시 첫 분기 |
|---|---|---|
| 유효 label 수 | preflight와 loss denominator 동일 | template/mask |
| finite ratio | 모든 주요 지점 1.0 | 최초 nonfinite layer |
| frozen delta | 정확히 0 | optimizer group |
| resume batch | 기준 run과 동일 | sampler cursor |
| parameter parity | 정한 tolerance 이내 | RNG/optimizer/checkpoint |

## 출력

`manifest.json`, `golden-batch.safetensors`, `tensor-atlas.jsonl`, `step-delta.json`, `checkpoint-manifest.json`, `resume-parity.json`, 작은 `EvalID` 결과를 보존한다.

## 증거 등급과 범위

### 실행 상태를 먼저 쓴다

이 문서의 명령과 판정 규칙은 기본적으로 `Proposed`다. 실제 GPU에서 실행하고 원본 logs·reports와 artifact hashes를 보존했을 때만 해당 cell을 `LocallyExecuted`로 바꾼다. 다른 환경의 독립 실행은 `ExternallyReproduced`로 별 기록한다.

모델 전체나 대규모 학습을 실행할 필요는 없다. 독자가 접근 가능한 작은 causal LM 또는 tiny config와 한두 GoldenBatch면 된다. 실행하지 않은 CUDA kernel, dtype와 checkpoint 경로를 성공처럼 서술하지 않는다.

## 실험 manifest

### 고정해야 할 입력과 환경

`manifest.json`에는 다음 필드를 둔다.

```json
{
  "evidence_level": "Proposed",
  "model_revision": "<immutable revision>",
  "weight_index_sha256": "<digest>",
  "tokenizer_revision": "<immutable revision>",
  "template_sha256": "<digest>",
  "config_sha256": "<digest>",
  "source_revisions": {},
  "environment": {
    "container_digest": "<digest>",
    "gpu_uuid": "<uuid>",
    "driver": "<version>",
    "cuda_runtime": "<version>",
    "framework": "<wheel-or-commit>"
  },
  "precision": {},
  "optimizer": {},
  "scheduler": {},
  "seeds": {}
}
```

mutable `latest`, package 이름만 있는 version, floating remote code가 남으면 preflight를 통과시키지 않는다. GPU clock/power 같은 성능 조건은 correctness manifest와 별 field로 둔다.

## 단계 0: 디렉터리와 명령 기록

### 재현 root

```text
lab28/
  manifest.json
  commands.jsonl
  input/golden-row.json
  input/golden-batch.safetensors
  reports/preflight.json
  reports/forward.jsonl
  reports/backward.jsonl
  reports/step.json
  reports/accumulation-parity.json
  reports/resume-parity.json
  failures/
  checkpoints/
```

모든 command record에는 working directory, argv, environment allowlist, input digests, exit code와 stdout/stderr locator가 있다. secret tokens와 원문은 command/log에 넣지 않는다.

## 단계 1: CPU preflight 상세

### tokenizer·labels·shape 검산

작은 대화 또는 문서 fixture를 template로 render하고 token IDs, offsets, roles, labels와 attention/position IDs를 저장한다. causal shift의 owner를 하나로 둔다. valid-label count를 독립 계산한다.

```python
assert input_ids.dtype == torch.long
assert input_ids.min() >= 0
assert input_ids.max() < logical_vocab_size
assert labels.shape == input_ids.shape
valid = labels.ne(-100)
assert valid.any()
assert valid.sum().item() == manifest_valid_count
```

판정은 특정 token ID가 아니라 원문 span→token→target 위치와 denominator의 일치다. padding, BOS/EOS, packed boundary가 있으면 별 행으로 검사한다. CPU preflight가 실패하면 GPU memory를 할당하지 않는다.

## 단계 2: parameter와 optimizer inventory

### update 전에 owner를 정한다

모든 parameters를 `(name,shape,dtype,requires_grad,storage_id,group_id)`로 기록한다. tied aliases는 canonical storage 아래 묶는다. trainable parameter ID 집합과 optimizer groups가 정확히 일치하는지 본다.

frozen parameters, buffers, quantized metadata와 optimizer moments를 구분한다. parameter group마다 LR, decay, optimizer options를 저장한다. shape만 같아 다른 parameter state가 매핑되는 일을 막기 위해 canonical names와 checksums를 쓴다.

## 단계 3: forward tensor atlas

### 관측 위치와 비용

embedding, first/middle/last block의 norm·attention·MLP outputs, final norm, logits와 unreduced loss를 관측한다. 모든 tensors를 CPU로 복사하지 않고 다음 축약 상태를 JSONL에 쓴다.

- logical path와 call index
- shape, stride, layout와 dtype/device
- finite/NaN/Inf counts
- min/max/mean/RMS
- deterministic selected projection/checksum
- producer source symbol과 UpdateID

hook에서 `.item()`을 남발하면 synchronization이 생긴다. correctness run에서만 상세 probe를 켜고 성능 run에서는 제거한다. hook이 output/gradient를 수정하지 않는지 negative control을 둔다.

판정은 “finite ratio 1”만이 아니다. 같은 GoldenBatch reference와 first-difference를 비교한다. input/config option을 바꿨을 때 예상보다 앞선 tensor가 달라지면 실험 통제가 실패한 것이다.

## 단계 4: backward·AMP·optimizer step

### 순서와 상태

```text
zero_grad
forward + loss numerator/count
scale(loss) if AMP
backward
DDP 없음 확인(single GPU)
unscale
global/group norm
clip
optimizer step or overflow skip
scheduler/scaler update
parameter delta 기록
```

raw/scaled/unscaled gradient를 구분한다. selected parameters에서 norm, finite count와 projection을 기록한다. tied parameter는 norm에 한 번만 포함한다. step 전후 `delta=after-before`를 직접 계산한다.

판정 규칙은 frozen delta 0, intended trainable delta 또는 expected initialization behavior, unscale 후 clip threshold, skip 시 parameter/moments 불변이다. scheduler가 overflow skip에서 전진하는지는 manifest policy와 맞아야 한다.

## 단계 5: accumulation 등가성

### 두 경로를 같은 objective로 만든다

Run A는 concatenated 큰 batch 한 번, Run B는 동일 samples를 K microbatches로 나눈다. dropout을 끄고 sample order, initial model/optimizer state를 같게 한다. microbatch valid counts가 일부러 다르도록 만든다.

각 microbatch numerator `S_k`, count `N_k`에서 global objective는 `ΣS_k/ΣN_k`다. microbatch means의 단순 평균을 wrong reference로 함께 실행한다. correct accumulation의 gradient와 first parameter delta가 concatenated reference tolerance 안에 있어야 한다.

다음 failure를 주입한다.

- loss를 K로 두 번 나눈다.
- local/microbatch mean을 평균한다.
- 중간 microstep에서 `zero_grad`한다.
- 마지막 remainder를 잘못 K로 normalize한다.

각 failure는 numerator/count reconciliation, gradient projection 또는 delta에서 잡혀야 한다.

## 단계 6: checkpoint interruption

### 5-step A와 3+2-step B

Run A와 B는 동일 initial root에서 시작한다. B는 committed step 3 직후 checkpoint하고 process를 종료한 뒤 새 process에서 load한다. checkpoint에는 다음을 포함한다.

- model parameters/buffers와 aliases
- optimizer moments/step counters
- scheduler와 AMP scaler
- CPU/CUDA RNG states
- sampler/data cursor와 next GoldenBatchID
- current/next UpdateID, config/source revisions

resume 후 first batch IDs, loss sum/count, gradient, LR/scale, first two deltas와 final parameter/optimizer state를 비교한다. exact parity를 지원하지 않는 backend면 허용 tolerance와 expected drift를 실행 전에 정한다.

weights만 같고 sampler/scaler가 다르면 resume 성공이 아니다. checkpoint 항목 하나씩 누락하는 failure child를 만든다. validator가 load를 거부하거나 evidence level을 낮춰야 한다.

## 단계 7: 작은 평가

### training plumbing과 품질을 분리한다

EvalID에는 고정된 아주 작은 validation fixture, tokenizer/template revision, metric denominator를 묶는다. 이 평가는 모델 품질 benchmark가 아니라 checkpoint/load, eval mode, loss calculation이 올바르게 이어지는지 확인하는 배관 검사다.

dropout이 꺼졌는지, gradients가 생성되지 않는지, train/eval loss numerator/count가 재현되는지 본다. generation을 한다면 prompt IDs와 decoding config/seed를 고정하고 token agreement를 artifact로 남긴다. 결과가 좋다는 주장을 하지 않는다.

## fault injection matrix

### 최초 detector를 검증한다

| 주입 | 가장 먼저 실패해야 할 지점 |
|---|---|
| tokenizer/model vocab mismatch | CPU preflight |
| labels double shift | target-position table |
| all-ignore labels | valid-count/skip state |
| norm epsilon 변경 | 해당 norm output |
| attention mask 방향 반전 | selected attention row |
| AMP clip-before-unscale | grad/clip report |
| optimizer group에 frozen weight | inventory assertion |
| zero_grad 누락 | 다음 microstep gradient |
| sampler state 누락 | resume next batch |
| scaler state 누락 | resume scale/update |
| tied alias 해제 | load inventory/첫 delta |
| checkpoint shard truncate | checksum/load gate |

failure는 정상 artifact의 child copy에 주입한다. 최종 loss가 달라질 때까지 기다리지 말고 예상한 최초 경계에서 차단한다. 의도와 다른 이유로 실패하면 test를 승인하지 않는다.

## 실패 분기

GoldenBatch가 다르면 tokenizer/template/sampler를 조사한다. embedding부터 다르면 checkpoint/IDs, 특정 layer부터 다르면 그 layer source/config/kernel을 본다. logits까지 같고 loss만 다르면 shift/mask/denominator다.

loss가 같고 gradient가 다르면 saved state, precision/custom backward를 본다. gradient는 같고 delta가 다르면 unscale/clip/optimizer/scheduler다. resume 뒤 input부터 다르면 sampler/RNG이고 first step은 같은데 둘째가 다르면 hidden optimizer/scaler/RNG state를 본다.

NaN은 input에서 forward 순서로 최초 non-finite를 찾는다. `nan_to_num`으로 통과시키지 않는다. OOM은 activation, gradient/optimizer, workspace와 instrumentation을 분리한다. batch를 줄이면 새 RunRevision이다.

## artifact 보존과 종료 조건

모든 report에는 manifest root, producer script digest, raw-log locator를 기록한다. 새 framework 결과로 golden을 자동 덮어쓰지 않는다. diff와 변경 의도, tolerance 조정을 검토해 child revision을 만든다.

종료 조건은 다음과 같다.

1. CPU preflight와 GoldenBatch lineage가 완결되었다.
2. forward atlas에 reference와 first-difference 규칙이 명시되었다.
3. backward/AMP/clip/step을 selected delta까지 재구성했다.
4. unequal-count accumulation이 concatenated reference와 맞는다.
5. 3+2 resume가 5-step reference와 선언한 parity를 만족한다.
6. fault injections가 예상 최초 detector에서 실패한다.
7. 산출물과 evidence level, `NOT_RUN` 범위가 명시된다.

이 일곱 조건 중 하나라도 raw evidence가 없으면 해당 cell은 `Proposed`다. 실습은 실행 계획을 실제 결과로 위장하지 않으며, 독자가 자신의 환경에서 같은 판정을 독립 재생하도록 설계된다.

## 독립 재현 확인

다른 검토자는 manifest와 artifacts만 받아 Run A/B의 selected tensors와 parameter delta를 다시 계산한다. 작성자의 in-memory state나 구두 설명이 필요하면 bundle이 불완전하다. command record에서 exact environment와 input digest를 확인한다.

`LocallyExecuted` 승격은 모든 필수 files의 hashes, exit codes와 raw reports가 있을 때만 허용한다. GPU가 다른 외부 재현은 별 environment root와 tolerance report를 갖는다. 과거 GPU의 bitwise 결과를 새 architecture의 기준으로 강제하지 않는다.

framework·CUDA·model revision이 바뀌면 영향받는 preflight, forward/backward, checkpoint와 performance cells를 stale로 표시한다. 정상 결과를 새 golden으로 자동 갱신하지 않는다. source/config diff와 예상 first difference를 먼저 승인한다.

민감한 원문은 일반 tensor atlas나 stdout에 넣지 않는다. opaque SampleID와 접근 통제 input artifact를 사용한다. 재현 가능성과 정보 노출을 서로 바꾸지 않는다.

마지막 보고서에는 성공 결과뿐 아니라 failure injection이 차단된 exact boundary와 미실행 항목을 같은 비중으로 둔다. 이 기록이 있어야 단일 GPU golden run이 다음 분산·fine-tuning 실험의 신뢰 가능한 기준점이 된다.

## 단계 8: 한 step의 순서를 고정 소스와 대조한다

dry-run fixture는 `input_ids [1,4]`, `labels [1,4]`에서 유효 target 세 개, vocabulary 8인 작은 모델이다. data seed, dropout seed와 parameter initialization seed를 분리하고 dtype, accumulation 1, max norm 1.0, optimizer와 scheduler config digest를 고정한다. 예상 atlas는 logits `[1,4,8]`, loss numerator scalar와 valid count 3, parameter별 gradient, clip 전후 global norm, optimizer state, scheduler step, checkpoint generation이다.

TorchTitan commit `b482babc…`의 [`Trainer.train_step:842-947`](https://github.com/pytorch/torchtitan/blob/b482babc5f1d5d718e1719a735f9a2d86d1b9aff/torchtitan/trainer.py#L842-L947)은 이 순서를 직접 보여 준다. batch를 읽으며 valid token을 세고, forward/backward 뒤 [`clip_grad_norm_`](https://github.com/pytorch/torchtitan/blob/b482babc5f1d5d718e1719a735f9a2d86d1b9aff/torchtitan/trainer.py#L919-L947)으로 norm과 finite 상태를 확인한다.

그다음 staging save를 기다리고 optimizer와 scheduler를 차례로 step한다. 이 lab의 사건열도 `fetch → valid count → forward/loss → backward → grad norm/clip → finite gate → optimizer → scheduler → checkpoint`로 고정한다.

정적 수치 oracle에서 clip 전 norm을 2.0으로 두면 계수는 `min(1,1.0/(2.0+ε))`이고 모든 gradient에 같은 비율이 적용돼야 한다. optimizer delta는 clip 뒤 gradient와 이전 moment로 계산하고, scheduler는 그 optimizer commit 뒤 정확히 한 번만 진행한다. checkpoint fixture는 model·optimizer·scheduler·RNG·sampler와 다음 BatchID를 저장한다. 값 복원의 최소 upstream 경계는 [`test_checkpoint.py:245-272`](https://github.com/pytorch/torchtitan/blob/b482babc5f1d5d718e1719a735f9a2d86d1b9aff/tests/unit_tests/cpu/test_checkpoint.py#L245-L272)다.

negative control은 labels 한 칸을 ignore로 바꾸기, clip을 optimizer 뒤로 옮기기, scheduler를 두 번 step하기, optimizer moment를 checkpoint에서 빼기 네 가지다. 각각 최초 차이는 valid count/loss denominator, parameter delta, next LR, resume 뒤 첫 delta에서 나야 한다. 최종 loss만 비교해서는 네 원인을 분리할 수 없다.

pass는 각 중간 state가 수기 oracle과 맞고 interruption 없는 A 경로와 checkpoint-resume B 경로의 다음 BatchID·LR·parameter/optimizer state가 선언한 tolerance에서 같은 것이다. input부터 다르면 sampler, logits에서면 model/RNG, loss에서만이면 shift·mask·denominator, gradient에서면 backward·precision, delta에서면 clip·optimizer, 다음 LR에서면 scheduler, resume에서만이면 checkpoint inventory를 조사한다. 실제 GPU 학습은 실행하지 않았으므로 kernel·성능·fresh-process 결과는 `NotExecuted`다.
