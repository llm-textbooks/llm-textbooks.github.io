# 장애 대응 플레이북 색인

증상 이름만으로 원인을 단정할 수는 없다. 먼저 문제를 재현하는 입력과 최초 불일치 경계를 고정한 뒤, 가설을 하나씩 반증한다. 각 플레이북을 마칠 때는 `IncidentID`, 문제를 일으킨 fixture, 최초 불일치, 수정 revision, 회귀 digest를 관련 본문 장에 연결한다.

## GR-001에서 어느 화살표가 먼저 끊겼는가

플레이북은 별도의 운영 부록이 아니라 본문의 수직 trace를 역방향으로 읽는 도구다. 최종 증상에서 시작하되, `GR-001`의 마지막 정상 artifact까지 거슬러 올라간다. 한 번에 설정 여러 개를 바꾸지 않고 정상 fixture와 mutation의 최초 divergence를 비교한다.

| 증상 | 먼저 고정할 artifact·관측값 | 첫 플레이북 | 본문·실습으로 돌아갈 곳 |
|---|---|---|---|
| loss가 NaN/Inf | BatchID, loss sum/count, scaler finite flag, 최초 nonfinite grad | [NaN과 Inf](01-nan.md) | 2·11·14장, [단일 GPU lab](../labs/28-single-gpu-golden-lab.md) |
| loss가 멈춤 | valid target 수, LR, grad/update delta, sample contribution | [학습 정체](02-plateau.md) | 6·11·13장 |
| sample이 반복됨 | SourceRow/PackID, sampler cursor, next batch | [sample 반복](03-sample-repeat.md) | 4·6·17장, [source→commit lab](../labs/06-source-to-commit-golden-lab.md) |
| token·응답 경계가 다름 | raw bytes, template/tokenizer revision, role span, shifted labels | [tokenizer 불일치](04-tokenizer-mismatch.md) | 5·18장, [SFT lab](../labs/18-sft-adapter-golden-lab.md) |
| OOM | 사건별 allocated/reserved, activation·workspace owner | [GPU OOM](05-oom.md) | 14–16장 |
| rank가 멈춤 | process-group membership, collective sequence, 마지막 합의 UpdateID | [rank hang](06-rank-hang.md) | 15–17·29장, [멀티노드 lab](../labs/29-multinode-failure-lab.md) |
| expert가 쏠림 | router logits, token/expert count, capacity/drop·aux 분모 | [expert imbalance](07-expert-imbalance.md) | 9·15·26장 |
| RL이 갑자기 불안정 | behavior/current/reward revision, action mask, queue age | [stale rollout](08-stale-rollout.md) | 19·20장, [policy-version lab](../labs/20-online-rl-policy-version-lab.md) |
| 재개 후만 달라짐 | checkpoint generation, model/optimizer/RNG/sampler/queue cut | [partial checkpoint](09-partial-checkpoint.md) | 17·20장 |
| 평가가 비정상적으로 좋음 | train/eval row lineage, exposure, metric denominator | [contamination](10-contamination.md) | 4·24·27장, [평가 lab](../labs/24-eval-contamination-uncertainty-lab.md) |

### 한 incident를 닫는 순서

1. 증상이 나타난 `RunID·BatchID·UpdateID·PolicyVersion·EvalID`를 보존한다.
2. 표에서 가장 가까운 플레이북을 열고 정상 대조군과 단 하나의 mutation을 고른다.
3. 본문의 고정 source link에서 실제로 상태를 소비하는 함수와 caller를 확인한다.
4. 관련 lab의 작은 fixture로 expected first divergence를 계산한다.
5. 수정 뒤 같은 fixture와 원래 incident를 재생한다.
6. 증상 소실이 아니라 부모·자식 artifact와 회귀 digest가 맞을 때 종료한다.

## 수치·최적화

- [NaN과 Inf](01-nan.md)
- [학습 정체와 plateau](02-plateau.md)
- [GPU OOM](05-oom.md)

## 데이터·표현

- [sample 반복](03-sample-repeat.md)
- [tokenizer 불일치](04-tokenizer-mismatch.md)
- [contamination](10-contamination.md)

## 분산·상태 수명

- [rank hang](06-rank-hang.md)
- [expert imbalance](07-expert-imbalance.md)
- [stale rollout](08-stale-rollout.md)
- [partial checkpoint](09-partial-checkpoint.md)

## 공통 종료 조건

timeout을 늘리거나 batch를 줄여 증상만 감춘 조치는 해결로 기록하지 않는다. 정상 대조군과 부정 대조군을 함께 실행하고, 수정 범위 밖의 loss 분모·sample 순서·optimizer step·artifact lineage가 그대로인지 확인한다.
