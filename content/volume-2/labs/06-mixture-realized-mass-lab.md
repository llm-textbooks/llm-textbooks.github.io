# 실습: 설정한 mixture가 실제 손실 질량이 되기까지

이 실습은 모델을 학습하지 않는다. 세 데이터 원천, 여덟 개의 짧은 메타데이터 행, 여섯 번의 추출과 세 번의 가상 update만으로 혼합 확률이 실제 학습 기여로 바뀌는 과정을 계산한다. [SourceRow에서 committed UpdateID까지](06-source-to-commit-golden-lab.md)가 한 batch의 계보와 commit 원자성을 세로로 추적했다면, 여기서는 여러 원천의 질량이 선택·마스킹·실패 update를 지나며 가로로 어떻게 재배분되는지 본다. 두 실습은 분모가 만나는 지점만 공유하며 같은 문제를 되풀이하지 않는다.

## 목표

완료 뒤에는 `configured probability`를 “학습 데이터 비율”이라고 부르면 왜 불충분한지 숫자로 설명해야 한다. 원천별로 설정 확률 질량, 실현 문서 수, 입력 토큰, shift 뒤 유효 타깃, 성공한 optimizer update에 들어간 가중 손실 질량을 분리한다. 짧은 원천이 소진됐을 때 확률을 어떻게 재정규화했는지, curriculum 단계가 어느 draw에서 바뀌었는지, 중단 뒤 어느 cursor에서 이어야 같은 suffix가 나오는지도 판정한다.

특히 문서 수가 세 원천에서 모두 2개라는 결과에 안심하지 않는다. 길이와 label mask가 다르면 토큰 질량이 달라지고, loss weight와 update 성공 여부가 개입하면 최종 손실 질량은 더 크게 갈라진다. 이 차이가 sampler 버그인지 의도한 변환인지는 중간 분모를 남겼을 때만 구별된다.

## 준비

Python 표준 라이브러리만 필요하다. CUDA, PyTorch, Transformers, tokenizer와 네트워크는 사용하지 않는다. 저장된 원장을 검증하려면 저장소 루트에서 다음을 실행한다.

```bash
python scripts/verify_training_mixture_mass_golden.py
python scripts/test_verify_training_mixture_mass_golden.py
```

권위 fixture는 `research/training/mixture-realized-mass-golden-2026-08-29.json`이다. `scripts/build_training_mixture_mass_golden.py`가 고정 입력으로 원장을 만들고 verifier가 이를 독립 재계산한다. mutation test는 누수 제외, 재개 cursor, 소진 뒤 재정규화, commit 질량을 각각 훼손해 fail-closed인지 확인한다. 실제 언어 모델 gradient를 모사하는 실험이 아니라 데이터 회계의 정적 oracle임을 결과에 명시한다.

## 고정 fixture

`web`, `code`, `sft` 원천에는 각각 다음 행이 있다. 숫자는 tokenizer 실행 결과가 아니라 이미 고정한 교육용 메타데이터다.

| 원천 | 행 | 입력 토큰 | 유효 타깃 | loss weight | 선택 전 판정 |
|---|---|---:|---:|---:|---|
| web | web-1 | 8 | 8 | 1.0 | 허용 |
| web | web-2 | 4 | 4 | 1.0 | 허용 |
| web | web-3 | 8 | 8 | 1.0 | `duplicate_of:web-1`로 제외 |
| code | code-1 | 12 | 6 | 1.0 | 허용 |
| code | code-2 | 9 | 9 | 1.0 | `benchmark_leak:eval-17`로 제외 |
| code | code-3 | 6 | 6 | 1.0 | 허용 |
| sft | sft-1 | 10 | 3 | 2.0 | 허용 |
| sft | sft-2 | 4 | 0 | 2.0 | 허용, zero-target |

중복과 누수는 sampler가 뽑은 뒤 버리는 retry가 아니라 허용 universe를 만들 때 제거한다. 그래야 탈락 행이 draw 확률이나 source cursor를 소비하지 않는다. `web-3`과 `code-2`는 서로 다른 실패 이유를 갖는다. 하나는 학습 모집단 내부 중복이고, 다른 하나는 평가 split과의 계보 누수다. 둘을 `filtered=true` 하나로 합치면 나중에 dedup 정책 변경과 benchmark 격리를 따로 감사할 수 없다.

curriculum은 두 단계다. warmup의 네 draw는 `(web=.5, code=.3, sft=.2)`, instruction의 두 draw는 `(web=.2, code=.3, sft=.5)`를 설정한다. 고정 uniform tape는 `[.10, .65, .95, .40, .10, .90]`이다. 각 원천 안에서는 허용 행을 순서대로 소비하며 replacement는 없다. 원천이 소진되면 남은 원천의 현재 단계 확률을 합이 1이 되도록 재정규화한다.

따라서 draw 순서는 다음과 같다.

| draw | 단계 | u | 선택 행 | update | commit | 가중 손실 질량 |
|---:|---|---:|---|---|---|---:|
| 0 | warmup | .10 | web-1 | update-0 | 성공 | 8 |
| 1 | warmup | .65 | code-1 | update-0 | 성공 | 6 |
| 2 | warmup | .95 | sft-1 | update-1 | 실패 | 0 |
| 3 | warmup | .40 | web-2 | update-1 | 실패 | 0 |
| 4 | instruction | .10 | code-3 | update-2 | 성공 | 6 |
| 5 | instruction | .90 | sft-2 | update-2 | 성공 | 0 |

draw 3이 끝나면 web cursor는 2여서 web이 소진된다. instruction 단계의 원래 확률 `.2/.3/.5`에서 web을 빼면 code와 sft의 유효 확률은 `.3/(.3+.5)=.375`, `.5/(.3+.5)=.625`다. 이때 `.10`은 code를 고른다. 소진된 web을 여전히 CDF에 넣었다가 retry하는 구현은 uniform tape와 cursor를 더 소비하므로 최종 문서 수가 우연히 같아도 sample-exact 실행은 아니다.

## 예상 표를 독립 계산한다

여섯 draw에 적용된 설정 확률을 평균하면 web `.4`, code `.3`, sft `.3`이다. 이것은 curriculum이 의도한 확률 질량일 뿐 결과 비율이 아니다. 원천별 결과는 다음과 같아야 한다.

| 원천 | 설정 확률 질량 | 문서 수·비율 | 입력 토큰·비율 | 유효 타깃·비율 | 커밋 손실 질량·비율 |
|---|---:|---:|---:|---:|---:|
| web | .4000 | 2 / .3333 | 12 / .2727 | 12 / .4444 | 8 / .4000 |
| code | .3000 | 2 / .3333 | 18 / .4091 | 12 / .4444 | 12 / .6000 |
| sft | .3000 | 2 / .3333 | 14 / .3182 | 3 / .1111 | 0 / .0000 |
| 합계 | 1.0000 | 6 / 1.0000 | 44 / 1.0000 | 27 / 1.0000 | 20 / 1.0000 |

`sft-1`의 원래 가중 손실 질량은 `3×2=6`이지만 update-1이 실패했으므로 committed mass에는 들어가지 않는다. `sft-2`는 update-2가 성공해도 valid target이 0이라 질량이 0이다. “읽었다”, “batch에 들어갔다”, “loss 분모에 들어갔다”, “파라미터 변경에 커밋됐다”는 서로 다른 사건이다. 6장의 realized mixture와 13장의 committed update clock을 연결해야 하는 이유가 이 행 하나에 드러난다.

## 재개 cursor와 suffix를 검산한다

draw 3 직후 checkpoint는 최소한 `draw_cursor=4`, `uniform_cursor=4`, `next_phase=instruction`, source cursor `{web:2, code:1, sft:1}`, `exhausted_sources=[web]`를 가져야 한다. 새 process는 이 상태에서 유효 확률 `.375/.625`를 다시 만들고 `code-3`, `sft-2`를 차례로 내야 한다.

확률 벡터와 seed만 저장해서는 부족하다. curriculum boundary, child iterator cursor와 exhaustion 집합이 빠지면 같은 난수 `.10`의 의미가 달라진다. 반대로 cursor를 draw 전에 저장했는지 뒤에 저장했는지도 명시해야 한다. 이 fixture의 cursor는 “draw 0~3이 이미 소비됐고 다음 draw는 4”라는 half-open 경계를 뜻한다. 기준 실행의 suffix 행 ID뿐 아니라 단계, 유효 확률, 누적 네 분모까지 같을 때 exact resume로 판정한다.

## 실패를 한 축씩 주입한다

첫째, `web-3`을 허용 목록에 되돌린다. 최초 불일치는 selection universe digest와 duplicate exclusion set이다. 최종 토큰 비율이 달라질 때까지 기다리면 검출 지점이 너무 늦다.

둘째, `code-2`의 `benchmark_leak` 판정을 삭제한다. 최초 불일치는 split isolation gate다. 이 행이 draw된 뒤 평가 점수가 오르는 것은 이미 두 번째 결과이며, 첫 증거는 평가 ID와 학습 허용 universe가 교차했다는 사실이다.

셋째, draw 4에서 web의 `.2`를 분모에 남긴다. code 유효 확률이 `.375`가 아니라 `.3`으로 기록되므로 exhaustion/renormalization assertion이 먼저 실패해야 한다. retry 횟수나 최종 source count만 검사해서는 이 오류를 조기에 찾지 못한다.

넷째, checkpoint의 `draw_cursor`만 3으로 되감는다. 최초 불일치는 resume state schema다. 실제 재개까지 진행하면 web-2 중복 소비나 phase boundary 재적용으로 나타날 수 있지만 verifier는 실행 전에 거부해야 한다.

다섯째, update-1을 성공으로 바꾸고 metrics만 그대로 둔다. commit ledger와 source별 committed-loss mass가 모순된다. 반대로 update 상태를 그대로 두고 sft 질량을 6으로 바꿔도 같은 경계에서 실패한다. `valid_targets`와 `committed_loss_mass`를 같은 카운터로 구현하면 이 반사실을 구분하지 못한다.

## 최초 불일치 기록법

보고서는 최종 비율 하나가 아니라 `허용 universe → curriculum phase → active source set → effective probability → selected row → input/valid mass → UpdateID commit → 누적 committed mass → resume suffix` 순서로 비교한다. 각 단계에는 기대 digest 또는 정확한 숫자, 관측값, 최초로 달라진 필드와 생산자를 적는다.

문서 순서가 처음부터 다르면 optimizer를 조사하지 않는다. 문서와 입력 토큰이 같고 valid target만 다르면 template, truncation과 label mask 경계를 본다. valid 질량까지 같지만 committed 질량만 다르면 overflow skip, accumulation window, update commit 원장을 본다. uninterrupted와 resumed 실행의 누적 합계는 같지만 다음 행이 다르면 sampler cursor·exhaustion·prefetch 상태가 빠진 것이다. 이런 분류가 “mixture가 이상하다”는 막연한 장애를 소유 가능한 상태 오류로 바꾼다.

## 완료 체크리스트

- [ ] 중복 `web-3`과 평가 누수 `code-2`가 서로 다른 이유로 허용 universe에서 제외된다.
- [ ] warmup과 instruction의 설정 확률, 적용 draw 범위와 경계가 원장에 남는다.
- [ ] web 소진 뒤 code/sft 확률 `.375/.625`를 계산하고 active source set과 함께 기록한다.
- [ ] draw 순서가 `web-1, code-1, sft-1, web-2, code-3, sft-2`와 같다.
- [ ] 문서 6, 입력 토큰 44, 유효 타깃 27, committed weighted-loss mass 20을 독립 계산한다.
- [ ] source별 네 분모와 비율이 예상 표와 같으며 문서 비율을 토큰 비율로 오인하지 않는다.
- [ ] 실패 update-1과 zero-target `sft-2`가 서로 다른 이유로 committed mass 0이 됨을 설명한다.
- [ ] draw 3 뒤 resume cursor와 소진 집합을 복원해 다음 행 `code-3`과 suffix parity를 확인한다.
- [ ] 네 mutation이 기대한 최초 경계에서 거부되고 verifier가 최종 합계만 검사하지 않는다.
- [ ] 결과의 한계를 적는다. 이 실습은 실제 tokenizer, packing, autograd, 분산 collective나 optimizer kernel을 실행했다는 증거가 아니다.

본문으로 돌아갈 때는 [4장](../chapters/04-web-corpus.md)의 중복·오염·split 허용 universe, [6장](../chapters/06-packing-mixture.md)의 realized document/input/valid/committed 분모와 streaming resume, [13장](../chapters/13-scheduler-scaling.md)의 curriculum·update clock을 같은 RecipeID 아래 연결한다. 세 장 중 하나라도 빠지면 데이터 선택, 손실 기여, 학습 시계 중 한 축을 추정으로 메우게 된다.
