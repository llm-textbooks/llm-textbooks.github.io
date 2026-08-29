# Golden Lab 19. Reward calibration·tie·disagreement 검산

## L19.1 이 실습이 묻는 질문

### L19.1.1 높은 pair accuracy는 믿을 만한 확률인가

reward model의 차이 `Δr=r_chosen-r_rejected`가 양수면 chosen을 고른다는 사실과 `σ(Δr)=0.8`이 사람 열 명 가운데 여덟 명의 선택을 뜻한다는 주장은 다르다. 전자는 순위 계약이고 후자는 calibration 계약이다. 이 실습은 모델을 실행하지 않는다. 작은 고정 표를 손계산하거나 짧은 배열 연산으로 다시 계산해 pair 방향, margin, tie, 사람 간 불일치, 길이 shortcut, 보정 확률, 분산 분모와 reward hacking 징후를 분리한다.

manifest에는 `FixtureRevision=RCAL-1`, 행 순서, vote count, score와 길이, bin 경계, calibration temperature, tie 정책과 reducer 식을 적는다. 결과를 본 뒤 bin이나 tie 정책을 바꾸면 새 revision이다.

## L19.2 고정 입력과 예상 표

### L19.2.1 여섯 pair를 그대로 복사한다

`human_frequency`는 chosen을 선택한 유효 표의 비율이다. binary 정답으로 다시 만들지 않는다.

| PairID | votes chosen:rejected | `r+` | `r-` | `Δr` | `p=σ(Δr)` | 길이 차 `L+-L-` | 사람 빈도 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 8:2 | 1.386 | 0.000 | 1.386 | 0.800 | 0 | 0.800 |
| B | 6:4 | 0.847 | 0.000 | 0.847 | 0.700 | 20 | 0.600 |
| C | 2:2 | 0.000 | 0.000 | 0.000 | 0.500 | 0 | 0.500 |
| D | 2:8 | -1.386 | 0.000 | -1.386 | 0.200 | -20 | 0.200 |
| E | 7:3 | 1.386 | 0.000 | 1.386 | 0.800 | 80 | 0.700 |
| F | 4:6 | 0.405 | 0.000 | 0.405 | 0.600 | 10 | 0.400 |

부호 검사는 가장 먼저 한다. A의 chosen/rejected 열만 바꾸면 `Δr=-1.386`, `p=0.2`가 되어야 한다. display order만 바꾸고 canonical candidate identity까지 바꾸지 않으면 label leakage다. margin `m`을 사용하는 loss는 `-log σ(Δr-m)`이다. A에 `m=0.2`를 주면 loss logit은 `1.186`이지 `1.586`이 아니다. margin 배열을 pair 순서와 따로 shuffle하면 scalar는 계속 나오지만 PairID 소유권이 깨진다.

C는 두 사건을 겹쳐 놓은 행이다. reward score equality라는 model tie와 사람 표가 2:2로 갈린 annotator disagreement가 동시에 있다. model tie를 accuracy 분모에서 제외하는 구현이 있어도 사람 disagreement를 삭제해도 된다는 뜻은 아니다. `wins/losses/model_ties/human_ties/abstains/invalid`를 별도 열로 보존한다. C를 임의 chosen label로 바꾸면 관측되지 않은 확신을 만든다.

## L19.3 score 중심화와 확률 보정을 분리한다

### L19.3.1 같은 margin에는 무한히 많은 절대 score가 있다

`(r+,r-)=(5,4)`와 `(0.5,-0.5)`는 모두 `Δr=1`이고 preference probability도 `σ(1)`로 같다. pairwise likelihood만으로 공통 offset은 식별되지 않는다. center regularizer가 `(r++r-)²`를 줄이는 것은 score 원점을 제한하는 학습 선택이지, `σ(Δr)`를 사람 빈도에 맞추는 calibration이 아니다. centered score가 0 근처라는 이유로 확률이 정확하다고 판정하지 않는다.

temperature scaling을 `p_T=σ(Δr/T)`로 정의하고 예시로 `T=2`를 적용한다. 모든 margin의 부호와 순위는 유지되지만 1.386의 확률은 0.8에서 약 0.667로, 0.847은 0.7에서 약 0.604로 누그러진다. `T`는 이 표를 보고 임의로 고르는 값이 아니라 별도 calibration split에서 fit한 상태다. score center coefficient, temperature, fit split과 RewardRevision을 따로 기록한다.

## L19.4 Brier와 reliability bin을 손으로 닫는다

### L19.4.1 soft vote frequency를 target으로 쓴다

고정 표의 Brier score는 `mean((p_i-y_i)²)`다. 행별 제곱 오차는 `[0,.01,0,0,.01,.04]`이고 평균은 **0.0100**이다. pair accuracy는 F의 과신과 B·E의 작은 miscalibration 크기를 보여 주지 못하지만 Brier는 보여 준다.

bin을 `[0,.33)`, `[.33,.67)`, `[.67,1]`로 사전 고정한다.

| bin | PairID | 평균 예측 | 평균 사람 빈도 | 절대 차 | 전체 가중 기여 |
|---|---|---:|---:|---:|---:|
| low | D | 0.200 | 0.200 | 0.000 | 0.0000 |
| middle | C,F | 0.550 | 0.450 | 0.100 | 0.0333 |
| high | A,B,E | 0.767 | 0.700 | 0.067 | 0.0333 |

따라서 이 bin 정의의 ECE는 약 **0.0667**이다. 행이 여섯 개뿐이라 통계적 결론을 내릴 수는 없다. 이 표는 구현 oracle이다. 실제 보고에서는 prompt 또는 candidate graph component 단위 bootstrap과 bin별 표본 수를 함께 낸다. 결과를 유리하게 만들려고 bin 경계를 옮기지 않는다.

## L19.5 길이만 보는 baseline과 reward hacking proxy

### L19.5.1 shortcut이 본 모델과 얼마나 겹치는가

길이 전용 baseline을 `p_len=σ(0.01(L+-L-))`로 고정한다. A~F의 예상값은 약 `[0.500,0.550,0.500,0.450,0.690,0.525]`다. 이 baseline이 높은 accuracy를 내면 reward가 의미를 읽었다는 증거가 아니라 데이터의 chosen 길이 편향을 측정했을 가능성이 커진다. 원 응답에서 의미를 유지한 채 장황한 문구만 붙인 counterfactual과 길이를 맞춘 pair를 추가한다.

별도 hacking probe H는 길이 차 `+200`, reward probability `0.95`, 사람 빈도 `0.30`으로 둔다. policy가 장황한 boilerplate를 붙여 reward만 올렸다는 모의 사례다. proxy reward가 올랐어도 사람 utility, 독립 verifier, 길이와 phrase frequency가 악화되면 승격을 거부한다. 같은 reward model로 H를 만들고 같은 모델로 성공을 판정하지 않는다.

## L19.6 분산 reducer의 전역 분모

### L19.6.1 rank 평균의 평균은 정답이 아니다

rank 0이 loss `[0.2,0.8]` 두 개, rank 1이 `[0.1]` 한 개를 소유한다고 하자. 전역 sum/count는 `(1.0+0.1)/(2+1)=0.366666…`이다. rank local mean을 동일 가중 평균하면 `(0.5+0.1)/2=0.3`으로 틀린다. all-reduce할 것은 `(loss_sum, valid_count)`이며 count가 0인 rank도 collective에 참여해야 한다. margin·tie 제외·invalid filter가 적용된 뒤의 유효 count를 쓴다.

negative fixture는 rank 배치를 `[A,B]/[C]`에서 `[A]/[B,C]`로 바꾼다. global row 집합이 같으면 global objective와 gradient scale은 같아야 한다. local mean 평균이 달라지면 reducer가 shard composition을 학습 목적에 섞은 것이다. 이 실습은 collective를 실행하지 않고 expected sum/count만 설계한다.

## L19.7 canonical test와 designed fixture의 경계

### L19.7.1 무엇을 직접 확인했고 무엇을 설계했는가

고정 TRL canonical test는 margin column을 collator에서 trainer까지 운반해 parameter가 변하는 경로와 reward centering coefficient를 켠 학습 경로를 확인한다. 공개 `compute_accuracy` test는 score equality를 tie로 보고 제외하는 동작을 확인한다. 이 upstream test가 사람 vote calibration, annotator cohort 대표성, ECE, 길이 counterfactual, 분산 global denominator나 reward hacking을 증명하지는 않는다.

이 문서의 A~H, Brier/ECE 표와 rank fixture는 `DesignedNotExecuted`다. 계산식과 기대값은 고정됐지만 모델·GPU·분산 runtime을 실행하지 않았다. 실행자가 배열 oracle을 구현하면 source revision, 코드 digest, 실제 출력과 expected diff를 붙여 `LocallyExecuted`로 올린다. 작은 fixture 통과를 production reward의 행동 타당성으로 확대하지 않는다.

## L19.8 최종 판정표

### L19.8.1 한 행이라도 빠지면 calibration 승인을 보류한다

| 검산 축 | 통과 조건 | 최초 실패 경계 |
|---|---|---|
| pair ordering | swap 시 margin·확률 부호가 함께 반전 | row ownership/chunk |
| margin | `Δr-m` 사용, PairID 순서 보존 | collator→loss |
| tie/disagreement | model tie와 사람 tie·기권 분리 | metric denominator |
| center/calibration | offset 규제와 확률 mapping revision 분리 | reward postprocess |
| Brier/ECE | expected 0.0100/0.0667, bin 고정 | aggregation |
| length baseline | 본 모델과 counterfactual을 함께 보고 | dataset shortcut |
| distributed reducer | global sum/count 0.366666… | all-reduce denominator |
| hacking proxy | reward와 독립 utility의 반대 움직임 검출 | release decision |

결과 묶음에는 PairID별 raw score·margin·vote, tie disposition, length, calibrated probability, bin, loss sum/count, RewardRevision과 fixture 상태를 넣는다. [19장의 reward 학습](../chapters/19-preference-reward.md), [20장의 online consumer](../chapters/20-online-rl.md), [24장의 신뢰할 수 있는 평가](../chapters/24-trustworthy-evaluation.md), [25장의 red-team 환류](../chapters/25-redteam-safety-training.md)를 같은 표에서 역추적해야 한다.
