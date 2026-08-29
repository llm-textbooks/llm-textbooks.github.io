# 장애 대응 플레이북 색인

증상 이름만으로 원인을 단정할 수는 없다. 먼저 문제를 재현하는 입력과 최초 불일치 경계를 고정한 뒤, 가설을 하나씩 반증한다. 각 플레이북을 마칠 때는 `IncidentID`, 문제를 일으킨 fixture, 최초 불일치, 수정 revision, 회귀 digest를 관련 본문 장에 연결한다.

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
