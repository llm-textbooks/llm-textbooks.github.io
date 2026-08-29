# Lab 30. SFT→RL→배포 종합

## artifact 계보

### 입력

base checkpoint, tokenizer/template, SFT dataset revision, preference dataset, reward/judge revision, quantizer와 serving runtime을 digest로 고정한다.

### DAG

`base → adapter-SFT → merged-SFT → preference/RL policy → merged-policy → quantized → serving package`의 모든 edge에 변환 command와 parent digest를 둔다.

### 실행 등급과 출판 규칙

각 edge는 `Proposed`, `LocallyExecuted`, `ExternallyReproduced`의 등급을 개별로 갖는다. SFT를 실행했다고 quantization·serving parity도 실행한 것으로 승격되지 않는다. 명령, 환경 manifest, raw log, exit code, output digest가 있고 각 판정 규칙을 통과해야 해당 edge를 `LocallyExecuted`로 표시한다. 실행하지 않은 구간은 추정 점수·예상 속도·임의의 checksum을 채우지 않는다.

이 lab은 파이프라인 전체를 한 번에 무조건 실행하라는 처방이 아니다. 작은 model·synthetic data로 state 계약을 먼저 검증하고, 실제 자원은 별도 승인·비용·안전 범위 안에서 사용한다. 파이프라인을 연속 명령 하나로 묶기 전에 각 edge의 입력·출력·rollback을 독립적으로 검증한다.

## 사전 manifest와 GoldenSet

### 데이터·prompt·policy를 고정한다

SFT row, preference pair, RL prompt를 최소 몇 개씩 선택해 `GoldenSetID`를 만든다. raw JSON, canonicalized bytes, rendered chat bytes, token IDs, role span, label/response mask, reference logprob, reward component의 checksum을 남긴다. 같은 문자열이어도 template·normalization·special token이 다르면 다른 GoldenSetID다.

정상 fixture에는 일반 대화, 긴 대화, tool call, safety boundary, 다국어·Unicode·code, 짧고 긴 response를 포함한다. negative fixture로 empty assistant mask, wrong base adapter, stale rollout, swapped preference, tokenizer mismatch, corrupt checkpoint, quantization scale 누락, serving stop 변경을 둔다. 모든 negative fixture는 예상한 첫 gate와 오류 메시지를 갖는다.

### 환경과 상태 소유권

repository/commit, Python, framework·trainer·PEFT/RL/quantization/runtime revision, CUDA/driver/GPU, dtype/backend, seed·determinism 설정을 적는다. parameter manifest에 name, shape, dtype, storage alias, `requires_grad`, optimizer group, shard owner를 저장한다. SFT·RL 경계에서 optimizer·scheduler·scaler·RNG·sampler, PolicyVersion의 owner가 바뀌는 지점을 표시한다.

## 단계별 실행

### SFT

assistant-only label mask를 시각화하고 유효 target 수를 assert한다. 한 optimizer step에서 base frozen delta=0, adapter delta>0을 확인한다. checkpoint resume 뒤 첫 sample과 optimizer state를 비교한다.

GoldenBatch에서 loss 분자·분모, token별 NLL, layer/adapter별 gradient norm, pre/post-clip norm, parameter update ratio를 저장한다. LoRA라면 A/B 배열 shape·scale·dropout·target module을 확인하고, QLoRA라면 base storage format, dequant compute dtype, master/optimizer state owner를 나눈다. optimizer parameter ID 집합이 `requires_grad=True` 집합과 일치하는지 assert한다.

checkpoint 저장 후 새 process에서 base·adapter·tokenizer를 불러 같은 batch의 logits·loss·다음 update를 비교한다. base revision을 의도적으로 바꾸거나 adapter key 하나를 빼 loader/parity gate가 실패하는지 본다.

### preference와 RL

chosen/rejected token length, reference logprob, denominator를 저장한다. rollout에는 `PromptID`, behavior `PolicyVersion`, token-span version, reward component, old logprob를 붙인다. stale threshold 밖 trajectory가 optimizer manifest에 들어가지 않는지 검사한다.

preference pair의 response common prefix, valid response token, length 차이, chosen/rejected 순서를 손으로 검산한다. reference/policy logprob이 같은 token span·mask·temperature에서 계산됐는지 확인한다. pair를 뒤집으면 loss·margin이 예측한 방향으로 바뀌어야 한다. chosen의 길이만 늘린 counterfactual은 길이 shortcut을 검사한다.

RL에서는 prompt dispatch, behavior sampling, environment/tool, reward, advantage, learner update, policy publication의 시간축을 남긴다. reward 성분의 raw/normalized/clipped value, KL·ratio·clip fraction, valid-token denominator, group statistics, discarded trajectory를 기록한다. old logprob을 current policy로 재계산하거나 mixed-version prefix/suffix를 허용하면 fail closed한다.

### merge와 quantize

adapter-enabled와 merged BF16의 first-step logits max error를 측정한다. quantized artifact에는 별도 tolerance를 적용하고 task eval과 memory를 측정한다. dtype·scale·group size가 manifest에 없다면 실패다.

merge equation `W' = W + scale·BA`를 target module 하나에서 수치로 재구성하고 tied embedding/head·shared weight가 중복 merge되지 않았는지 본다. runtime adapter, merged BF16의 layer별 hidden·logits max/mean error, KL, greedy token agreement을 비교한다. tolerance는 결과를 본 뒤 늘리지 않는다.

quantization은 calibration dataset·observer, weight/activation 방식, bit width, group, axis, zero-point, scale dtype, excluded module, kernel/backend을 고정한다. random prompt만 쓰지 말고 GoldenSet의 길이·domain·outlier를 포함한다. artifact size·peak memory·latency를 실제 실행했을 때만 기록하고, 파일 크기를 런타임 memory로 오인하지 않는다.

### serving parity

offline과 API renderer에 같은 UTF-8 prompt를 넣는다. rendered bytes→token IDs→first logits→greedy tokens 순서로 최초 divergence를 찾는다. sampling parity는 RNG backend가 같을 때만 추가한다.

API request body와 server의 parsed request, template option, truncation/padding, stop handling, adapter/model mapping, quantization/kernel option을 로그에 고정한다. streaming은 partial UTF-8, stop token과 stop string, usage count, finish reason을 검사한다. batch 순서와 동시성을 바꾸어 다른 request의 token/KV가 섞이지 않는 permutation test를 한다.

offline·API 모두 greedy인데 token이 다르면 최종 text만 비교하지 않는다. bytes, IDs, attention/position, first differing layer/logit, backend 순서로 내려간다. quantized runtime의 차이는 BF16 merge parity와 분리한 tolerance report로 다룬다.

### release

task eval, private red-team, over-refusal, contamination, runtime smoke를 실행한다. 실행하지 않은 성능 칸은 비워 두고 `미실행`으로 쓴다. rollback artifact의 실제 load smoke를 통과해야 승인한다.

evaluation은 baseline·candidate에 같은 renderer·harness·item population을 쓰고 row-level output과 uncertainty를 보존한다. 안전 평가는 harmful compliance만이 아니라 benign refusal·helpfulness·tool side effect를 함께 본다. contamination scan은 eval row와 SFT/preference/RL data의 exact/n-gram/semantic 후보를 추적하고 영향 항목을 별도 보고한다.

canary는 적은 traffic에서 error, latency, GPU memory, output length, refusal·safety 신호를 기준과 비교한다. rollback threshold는 배포 전에 고정하고, threshold를 넘으면 새 request를 이전 artifact로 전환한 뒤 in-flight 처리와 cache 상태를 정책대로 정리한다.

## 판정 분기

| 불일치 | 먼저 확인 | 다음 분기 |
|---|---|---|
| token ID | tokenizer/template digest | normalization/special token |
| BF16 logits | merge equation/target modules | dtype/tied head |
| quant logits | scale/group/calibration | kernel/backend |
| RL 품질 | policy version/reward/denominator | stale queue/judge bias |
| API output | renderer/stop/greedy | runtime model mapping |

## 최종 산출물

release manifest에는 모든 parent digest, parity report, EvalID, red-team 결과, known limitation, runtime-unverified 목록, rollback command와 만료 시점을 담는다.

artifact 디렉터리는 top DAG manifest, 각 edge의 command·environment·raw log, GoldenSet, parameter/optimizer manifest, checkpoint roundtrip, preference/RL trajectory sample, merge·quantization·serving parity, evaluation·red-team row, canary·rollback report를 포함한다. 각 파일에 checksum과 evidence grade를 붙이고 top manifest에서 부모→자식 경로를 재구성할 수 있게 한다.

## 통제 실험·실패 주입과 최종 판정

wrong base revision, template의 공백 하나 변경, zero-valid-label batch, frozen base의 변형, stale/mixed PolicyVersion, adapter 중복 merge, scale 누락 quantization, server의 old tokenizer, rollback artifact 누락을 하나씩 주입한다. 각 실패는 예상한 첫 경계에서 mutation effect가 다음 artifact로 전파되기 전에 실패해야 한다. 오류를 잡았지만 이미 merged/release artifact를 publish했다면 gate가 너무 늦은 것이다.

최종 통과는 DAG의 모든 필수 edge가 실행·검증되고, parent digest와 state 소유권이 끊기지 않으며, GoldenSet의 bytes→IDs→mask→logits→update→merge→quantize→API 경계가 사전 tolerance를 만족할 때만 가능하다. 품질·안전·오염·운영 gate와 rollback load smoke가 모두 통과해야 한다. 하나라도 `NotExecuted`이거나 근거가 없으면 release 판정은 미완료다.

실패 시 `verdict.md`에 최초 불일치 경계, 기대·실제 state, 영향받은 후속 artifact, 지지·기각된 가설, 격리·재실행·rollback 조치, owner와 재검증 조건을 남긴다. 다음 교대자는 이 문서와 top manifest만 보고도 어느 edge에서 멈췄으며 어느 artifact를 다시 만들어야 하는지 판단해야 한다.

## 고정 함수와 테스트에서 종단 dry-run을 만든다

SFT·preference 경계의 원본 좌표는 TRL commit `a7be897f…`로 고정한다. [`dpo_trainer.py:1755-1780`](https://github.com/huggingface/trl/blob/a7be897f5c8d7b52161f9f8a47d8e6242456b898/trl/trainer/dpo_trainer.py#L1755-L1780)의 `compute_loss`는 일반 `_compute_loss`와 Liger·ZeRO-3/FSDP redirection을 분기한다.

따라서 같은 batch라도 selected path, sharded `lm_head` gather와 loss denominator가 artifact에 들어가야 한다. `tests/test_dpo_trainer.py:266-282`는 DPO trainer fixture의 입력·구성 경계를, `tests/test_sft_trainer.py:1284-1334`는 dataset preparation과 completion label 생성을 조사할 다음 좌표다.

정적 fixture는 두 대화 row와 한 preference pair로 충분하다. rendered IDs `[B,T]`, labels `[B,T]`, chosen/rejected log-prob sum `[B,2]`, policy version과 adapter parameter inventory를 고정한다. seed는 template/data order, dropout, rollout sampling을 분리해 적는다. 예상 상태는 assistant 유효 token 수, SFT loss numerator/denominator, chosen-rejected margin, merge 전후 한 target weight와 API first-logit checksum이다.

assistant mask 한 칸을 user token 쪽으로 옮기는 변형에서는 labels에서 가장 먼저 달라지고, DPO pair를 바꾸면 margin 부호에서, wrong base를 merge하면 parent digest에서, old tokenizer를 server에 두면 rendered IDs에서 갈라져야 한다. pass는 최종 응답이 비슷하다는 뜻이 아니라 각 변형이 예정한 첫 gate에서 후손 artifact 생성 전에 멈추는 것이다. IDs까지 같고 SFT loss만 다르면 collator·denominator, margin만 다르면 reference/policy log-prob span, merge 뒤만 다르면 target module·dtype, API에서만 다르면 renderer·runtime mapping으로 내려간다.

이 좌표들은 정적 oracle의 출발점이다. 실제 CUDA kernel, 대규모 rollout과 canary 배포를 실행하지 않았으므로 해당 셀은 `NotExecuted`다. 독자는 원본 함수에서 selected branch를 표시하고 test fixture의 tensor 열을 이 lab manifest에 옮긴 뒤에만 실행 승인을 요청한다.
