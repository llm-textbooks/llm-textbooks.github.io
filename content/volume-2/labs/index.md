# 실습 색인

이 부록은 본문에서 설명한 계약을 작은 고정 입력으로 다시 확인하는 출발점이다. 실제 대규모 학습을 재현했다고 주장하지 않는다. 각 실습은 실행 전에 입력·revision·seed·예상 불변조건을 기록하고, 실행 뒤에는 최초 불일치와 산출물 digest를 남긴다.

## 데이터에서 update commit까지

- [SourceRow에서 committed UpdateID까지](06-source-to-commit-golden-lab.md): 정제·토큰화·패킹·분모·commit·checkpoint·fresh resume를 모델 학습 없이 한 deterministic ledger로 검산한다.
- [설정한 mixture가 실제 손실 질량이 되기까지](06-mixture-realized-mass-lab.md): 설정 확률과 실현 문서·입력 토큰·유효 타깃·커밋 손실 질량을 분리하고, 소진 뒤 재정규화·curriculum 경계·resume cursor·중복/평가 누수 격리를 순수 산술로 검산한다.

## 파인튜닝과 정렬

- [SFT·adapter golden lab](18-sft-adapter-golden-lab.md)
- [Reward calibration·tie·disagreement lab](19-reward-calibration-disagreement-lab.md): score 중심화와 확률 보정, tie·사람 불일치, 길이 shortcut, Brier/ECE, 전역 분모와 proxy hacking을 고정 표로 검산한다.
- [온라인 RL policy version lab](20-online-rl-policy-version-lab.md)
- [SFT·RL·배포 종단 lab](30-sft-rl-deploy-golden-lab.md)

## 단일 GPU와 멀티노드

- [단일 GPU golden lab](28-single-gpu-golden-lab.md)
- [멀티노드 failure lab](29-multinode-failure-lab.md)

## 판정 계약

- [평가·오염·불확실성 결정적 실습](24-eval-contamination-uncertainty-lab.md): 고정 열두 행으로 exact/near contamination, 가중 분모, paired·층화 bootstrap, 다중 비교, 기권·judge confusion과 최초 불일치를 검산한다.

성공 로그만 남기지 않는다. 실패 fixture가 예상 경계에서 실패했는지, 수정 뒤 같은 fixture가 회귀 테스트로 닫혔는지, checkpoint·평가·배포가 동일한 입력과 상태 ID를 가리키는지 확인한다.
