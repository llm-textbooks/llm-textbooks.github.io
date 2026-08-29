# Golden Lab 18. SFT→LoRA/QLoRA→merge→serving parity

## L18.1 목적과 고정 입력

### L18.1.1 실행 등급

이 랩은 명령을 제시하는 것과 실행 결과를 구분한다. 독자는 `reports/lab-18-run.json`에 `Proposed`, `LocallyExecuted`, `ExternallyReproduced` 중 하나를 기록한다. GPU 종류, driver, PyTorch·Transformers·TRL·PEFT·bitsandbytes revision을 manifest에 고정한다.

입력 row는 system/user/assistant가 하나씩 있는 짧은 대화다. 다음을 먼저 저장한다.

- 원 JSON canonical SHA-256
- rendered chat text SHA-256
- token IDs, role span, shifted labels와 `-100` mask
- `GoldenBatchID=sha256(input_ids||labels||attention_mask)`
- valid-label denominator

## L18.2 관찰 코드

### L18.2.1 trainable state 검사

모델 wrapping 전후 `(name,shape,dtype,requires_grad,data_ptr)`를 덤프한다. LoRA에서는 target module의 base weight가 frozen이고 A/B만 trainable인지, QLoRA에서는 base storage가 4-bit container이고 compute dtype이 manifest와 같은지 본다. optimizer group의 parameter ID 집합이 trainable ID 집합과 정확히 같아야 한다.

```python
trainable = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
grouped = {id(p) for g in optimizer.param_groups for p in g["params"]}
assert grouped == set(trainable)
```

첫 forward에서 LoRA `B=0` 초기화라면 dropout을 끈 base와 wrapped logits가 tolerance 안에서 같아야 한다. 이 검사가 실패하면 학습을 시작하지 않는다.

## L18.3 한 step과 checkpoint

### L18.3.1 순서

`zero_grad→forward→loss denominator 확인→backward→unscale→clip→optimizer.step→scheduler.step`을 한 번 수행한다. base와 adapter tensor checksum을 전후 비교한다. base가 바뀌거나 adapter가 전혀 바뀌지 않으면 실패다.

adapter checkpoint에는 base revision, tokenizer/template, PEFT config, tensor hashes, optimizer/scheduler/scaler state를 넣는다. 저장 후 새 process에서 base를 다시 읽고 adapter를 load해 logits를 비교한다.

## L18.4 merge·quantize·serve

### L18.4.1 DAG와 허용오차

`BaseID→AdapterID→MergedBF16ID→QuantizedID`를 immutable artifact로 만든다. runtime adapter와 BF16 merge는 max/mean logit error, KL, greedy token agreement를 기록한다. quantized 결과는 별도 넓은 threshold를 쓰되 threshold를 결과를 본 뒤 정하지 않는다.

### L18.4.2 실패 주입

다음 네 번을 일부러 실패시킨다.

1. 다른 base revision에 adapter load: loader가 거부하거나 parity가 실패해야 한다.
2. template의 공백 하나 변경: `GoldenBatchID`가 바뀌어야 한다.
3. merge dtype을 FP16으로 변경: artifact ID와 tolerance report가 분리돼야 한다.
4. tokenizer 없이 serving export: release gate가 닫혀야 한다.

최종 보고서는 어느 layer에서 최초 divergence가 났는지, 그것이 adapter delta·merge rounding·quantization·template 중 어디에 속하는지 적는다.

## L18.5 사전 조건과 실험 디렉터리

### L18.5.1 실행 전에 결정할 것

이 실습은 특정 대형 모델을 요구하지 않는다. 독자가 합법적으로 접근할 수 있고 단일 장비 메모리에 들어가는 작은 causal LM 또는 tiny configuration을 사용한다. 실제 실행하지 않았다면 model 이름이나 수치를 예시 결과처럼 적지 않는다.

실험 root는 다음 immutable inputs를 가리킨다.

- `BaseModelRevision`, weight-index와 shard SHA-256
- `TokenizerRevision`, vocabulary와 added-token digest
- `ChatTemplateRevision`, rendered fixture digest
- Transformers·PEFT·quantization library commit 또는 package/wheel digest
- adapter target modules, rank, alpha, dropout, bias와 task type
- precision, optimizer, scheduler, gradient accumulation과 seed

`manifest.json`의 `evidence_level`은 처음에 `Proposed`다. 명령을 실제 실행하고 원본 로그와 산출물을 보존했을 때만 `LocallyExecuted`로 바꾼다. 다른 환경에서 독립 재현되어 artifact hashes와 판정이 들어왔을 때만 `ExternallyReproduced`를 추가한다.

디렉터리 예시는 다음과 같다.

```text
lab18/
  manifest.json
  input/chat-row.json
  input/golden-batch.safetensors
  reports/preflight.json
  reports/trainable-inventory.json
  reports/step-0001.json
  reports/reload-parity.json
  reports/merge-parity.json
  reports/quantized-parity.json
  failures/
  artifacts/
```

경로 존재를 성공으로 세지 않는다. 각 report에는 producer command/script digest, input IDs, 시작·종료 시각, exit status, raw-log locator와 schema version이 있어야 한다.

## L18.6 GoldenBatch를 만드는 절차

### L18.6.1 template에서 labels까지

한 개의 정상 대화와 두 boundary 대화를 준비한다. boundary에는 빈 assistant 답변, 긴 답변 truncation, tool/special marker 중 현재 model이 지원하는 사례를 넣는다. 지원하지 않는 role은 억지로 추가하지 않는다.

의사코드는 다음 순서를 따른다.

```python
row = load_json("input/chat-row.json")
rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False)
encoded = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False)
labels = build_labels(encoded, role_spans=row["role_spans"], ignore=-100)
assert labels.shape == encoded["input_ids"].shape
valid = labels.ne(-100)
assert valid.sum() > 0
save_golden_batch(encoded, labels, valid)
```

실제 tokenizer가 offsets를 제공하지 않으면 byte/span mapping을 별도 tokenizer trace로 만든다. assistant-only SFT라면 user/system labels가 `-100`이고 assistant target만 유효한지 token 표로 검토한다. causal loss 내부 shift owner와 collator shift를 동시에 적용하지 않는다.

판정은 특정 token ID 숫자가 아니다. 다음 불변식을 사용한다.

- 모든 input ID가 logical vocabulary 범위에 있다.
- role별 target bitmap이 template 정책과 일치한다.
- shift 뒤 각 target이 의도한 다음 token/source span을 가리킨다.
- framework와 별개로 numerator와 valid-target denominator를 계산해 대조할 수 있다.
- template/normalization 변경 시 `GoldenBatchID`가 반드시 바뀐다.

## L18.7 adapter 주입을 함수 그래프에서 검증한다

### L18.7.1 주입 전후 inventory

adapter 주입 전 모든 modules와 parameters를 `(semantic_role,name,shape,dtype,storage_id)`로 기록한다. 주입 뒤에는 base와 A/B matrices, scaling, dropout modules를 다시 기록한다. regex target은 문자열 설정만 보지 말고 실제 matched module names와 개수를 보고한다.

주입 결과가 0개이거나 예상 model roles와 다른 modules를 잡으면 즉시 중단한다. 예를 들어 다른 model family의 `q_proj` 예제를 그대로 사용해 실제 fused QKV module을 놓치는 경우를 차단한다.

LoRA layer의 reference는 개념적으로 다음과 같다.

```python
base = linear(x, W)
delta = scale * linear(linear(dropout(x), A), B)
wrapped = base + delta
```

초기화가 B=0이라는 전제는 실제 PEFT config/source에서 확인한다. 초기 delta가 0인 경우 dropout을 꺼 동일 input에서 base/wrapped output과 selected logits가 사전 tolerance 안에서 맞아야 한다. 초기화가 다른 adapter는 그 식에 맞는 별 oracle을 만든다.

QLoRA에서는 다음 state를 분리한다.

- quantized base container와 scale/metadata
- matmul에 사용되는 compute dtype
- adapter parameter/storage dtype
- optimizer moments dtype과 offload owner
- dequantization/fused kernel과 fallback

base weight를 dense BF16처럼 checksum하지 못하면 quantized payload와 metadata digest를 별도로 저장한다.

## L18.8 한 update를 재구성한다

### L18.8.1 단계별 관측

각 단계에서 아래 항목을 `step-0001.json`에 쓴다.

| 단계 | 관측 상태 | 판정 규칙 |
|---|---|---|
| forward | selected hidden/logits, loss sum/count | base 입력과 labels가 manifest와 동일 |
| backward | adapter A/B와 base gradient | trainable policy와 일치 |
| unscale | scale, nonfinite count | clip 전에 true-unit gradient |
| clip | raw norm, coefficient | configured group/threshold와 일치 |
| step | adapter delta, base delta | adapter 변화, frozen base 불변 |
| scheduler | applied LR/counter | committed update와 함께 전진 |
| zero | gradient state | 다음 window 계약과 일치 |

예상 loss나 gradient 숫자를 책에서 정하지 않는다. 동일 GoldenBatch의 FP32 또는 허용한 reference path와 비교하고 tolerance를 manifest에서 실행 전에 정한다. norm이 같다는 이유로 gradient 방향을 승인하지 않고 selected projection과 cosine도 본다.

LoRA의 한 factor가 zero initialization 때문에 첫 step gradient가 0일 수 있다. “모든 adapter tensor가 반드시 첫 step에 변한다”를 판정 규칙으로 쓰지 않는다. 실제 initialization 식에서 어느 factor가 언제 gradient를 받는지 예상표를 만든다.

## L18.9 저장·재개·merge DAG

### L18.9.1 adapter checkpoint round trip

save 전에는 base, adapter, optimizer/scheduler/scaler, RNG, sampler cursor와 GoldenBatchID를 묶는다. adapter-only 배포 artifact와 exact training-resume checkpoint를 구분한다. 전자는 optimizer/RNG를 생략할 수 있지만 후자는 생략할 수 없다.

새 process에서 다음을 수행한다.

```text
1. exact BaseModelRevision 로드
2. exact PEFT config로 module graph 구성
3. adapter tensors와 training state 로드
4. alias/trainable inventory 재검증
5. 같은 GoldenBatch forward/backward
6. uninterrupted reference의 다음 parameter delta와 비교
```

runtime adapter `Base+Δ`, merged BF16 `W'=W+Δ`, quantized merged artifact를 DAG의 서로 다른 nodes로 둔다. merge 후 base artifact를 덮어쓰지 않는다. merge 전후에는 selected weights와 logits를 비교하고, quantized node는 별 tolerance와 downstream small eval을 사용한다.

## L18.10 negative·fault injection matrix

### L18.10.1 의도적으로 깨뜨릴 것

| 주입 | 가장 먼저 실패해야 할 경계 | 조용히 통과하면 의심할 것 |
|---|---|---|
| Base revision 교체 | artifact compatibility/load | base ID 검증 부재 |
| tokenizer added token 교체 | GoldenBatch/vocab preflight | tokenizer 미봉인 |
| template 공백 변경 | rendered/GoldenBatch digest | template hash 누락 |
| target module 오타 | injection inventory | zero-match 허용 |
| adapter rank 불일치 | state schema/load | shape 기반 부분 load |
| tied head alias 해제 | alias inventory/첫 update | 값만 비교한 검사 |
| frozen base를 optimizer에 추가 | group-set assertion | requires-grad만 검사 |
| scheduler state 누락 | resume 첫/둘째 delta | weight-only resume |
| FP16 merge | artifact ID/parity report | merge dtype 미기록 |
| tokenizer 없는 export | serving release gate | 불완전 bundle 허용 |

실패는 정상 artifact를 파괴하지 않는 child directory에 주입한다. failure가 예상보다 뒤에서 발견되면 detector를 앞당긴다. 예외가 발생했다는 사실만이 아니라 의도한 reason/schema assertion에서 실패했는지 본다.

## L18.11 실패 분기와 종료 조건

### L18.11.1 최초 차이에 따른 분기

GoldenBatch가 다르면 model/adapter를 조사하지 않고 tokenizer·template·mask로 돌아간다. adapter 주입 직후 base parity가 다르면 merge나 optimizer 이전에 injection/initialization을 본다. forward는 같고 gradient가 다르면 detach, dropout RNG, custom backward를 본다.

gradient는 같고 delta가 다르면 unscale·clip·optimizer group/state를 본다. runtime adapter와 merge가 다르면 weight equation, dtype와 alias를 본다. merge는 맞고 quantized만 다르면 quantizer config/scale/kernel을 본다. serving만 다르면 tokenizer/template와 backend dispatch를 본다.

실습 종료 조건은 다음 모두다.

1. manifest와 GoldenBatch lineage가 완결되었다.
2. adapter injection과 trainable/optimizer 집합이 검증되었다.
3. 한 update를 loss에서 adapter delta까지 재구성했다.
4. adapter checkpoint reload와 next-step parity가 통과했다.
5. runtime·merged·quantized artifact가 각각 별도의 node와 report로 남는다.
6. 모든 failure injection이 예상 최초 경계에서 차단되었다.
7. 실행하지 않은 GPU/serving cells는 `Proposed` 또는 `NOT_RUN`으로 남았다.

마지막 `reports/lab-18-run.json`에는 각 조건의 evidence locator와 판정을 기록한다. prose 요약만 있고 raw report가 없으면 완료로 세지 않는다.

## L18.12 artifact 보존 규칙

정상 기준과 failure child artifacts는 별 directory에 둔다. 새 library 결과로 golden 파일을 자동 덮어쓰지 않는다. source/config diff와 tolerance 변경 이유를 사람이 검토한 뒤 새 revision을 만든다.

민감한 대화 원문은 일반 로그에 복사하지 않고 접근 통제 input artifact와 salted digest를 사용한다. 공개 report에는 token 역할·shape·통계와 opaque SampleID만 둔다. 재현자는 권한 범위 안에서 checksum을 확인한다.

최종 bundle에는 실행 environment snapshot과 exact commands, stdout/stderr locator, exit code가 있다. `LocallyExecuted`는 이 원본 증거가 존재할 때만 허용한다. 일부 단계만 실행했으면 단계별 evidence level을 따로 쓴다.

향후 PEFT·Transformers·quantizer revision이 바뀌면 injection, one-step, reload, merge, export cells를 영향 범위에 따라 다시 수행한다. 과거 PASS를 다른 base model이나 GPU/backend에 자동 승계하지 않는다.

이 규칙까지 확인한 뒤 실습을 봉인한다.

## L18.13 resize에서 merge까지 하나의 정적 tensor oracle로 닫는다

fixture는 vocabulary 8, hidden size 4인 tied embedding/LM head와 `q_proj [4,4]` 하나로 만든다. 새 token 두 개를 append하고 LoRA rank 2, `alpha=4`, dropout 0, seed를 tokenizer·adapter initialization·batch 순서별로 따로 고정한다. 예상 shape는 resize 뒤 embedding과 head가 모두 `[10,4]`, LoRA `A [2,4]`, `B [4,2]`다. optimizer parameter ID 집합은 A와 B만 포함하며 base embedding·head·q_proj는 포함하지 않는다.

Transformers commit `550d7b38…`의 [`resize_token_embeddings:2710`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_utils.py#L2710-L2755)은 resize 뒤 tie를 다시 적용하는 계약을 드러낸다. [`tie_weights:2607-2688`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_utils.py#L2607-L2688)은 target attribute를 source parameter로 다시 지정한다.

dry-run에서는 shape만 보지 않고 embedding weight와 head weight의 parameter identity·storage pointer가 같은지 기록한다. upstream resize fixture는 [`test_modeling_common.py:2284-2316`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/tests/test_modeling_common.py#L2284-L2316)에서 크기 증가·축소 뒤 shape와 보존 행을 확인할 수 있다.

PEFT commit `1feedf1a…`의 [`inject_adapter:787`](https://github.com/huggingface/peft/blob/1feedf1a4b96c86e2efcdd28b84ce9b949e3732c/src/peft/tuners/tuners_utils.py#L787-L850)은 target module을 adapter layer로 교체한다. target 이름을 `q_proj`로 고정하고 주입 전후 module path, base parameter identity, A/B parameter ID와 `requires_grad`를 비교한다. target을 `k_proj`로 잘못 지정한 negative control은 injection inventory에서 먼저 실패해야 한다. optimizer 생성 뒤에 adapter를 주입하는 변형은 A/B가 optimizer group에서 빠지는 순간 실패하며, loss가 내려가지 않을 때까지 기다리지 않는다.

한 update 뒤 runtime adapter의 기대 weight는 `W + (alpha/r)·B·A`다. [`merge_and_unload:730-769`](https://github.com/huggingface/peft/blob/1feedf1a4b96c86e2efcdd28b84ce9b949e3732c/src/peft/tuners/tuners_utils.py#L730-L769)이 새 반환 model을 사용해야 한다는 점도 manifest에 반영한다. pass oracle은 target weight의 수기 합성과 merged weight, runtime-adapter logits와 merged logits가 사전 tolerance 안에서 맞고, resize한 tied storage가 중복 merge되지 않는 것이다. `safe_merge=False`에서 B에 NaN을 넣는 negative control은 별 선행 finite gate가 막아야 한다.

최초 불일치는 `tokenizer length → embedding/head shape → tied identity → target module set → optimizer parameter IDs → A/B delta → merge equation → logits` 순서로 찾는다. 앞 단계가 같고 optimizer IDs만 다르면 group construction을, merge weight부터 다르면 scale·dtype·alias를, weight는 같고 logits만 다르면 eval mode·dropout·forward dispatch를 판다. 이 상태열이 닫혀야 export·serving parity로 넘어간다.
