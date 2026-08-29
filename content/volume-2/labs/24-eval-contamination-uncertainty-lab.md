# 평가·오염·불확실성 결정적 실습

이 실습은 모델을 실행하지 않는다. 고정된 열두 평가 행만으로 식별자, 오염 판정, 가중 분모, 신뢰구간, 다중 비교, 기권과 판정기 혼동을 검산한다. 목표는 최종 점수 하나가 아니라 입력에서 보고서까지 어느 단계가 처음 달라지는지 찾고, 그 차이가 모델 변화인지 평가 절차 변화인지 설명하는 것이다.

## 상태를 고정한다

시작 전에 `CaseID=EVAL-FIXTURE-24-A`, `ModelRevision=model-r17`, `PromptTemplate=template-ko-v3`, `JudgeRevision=judge-r8`, training corpus `train-manifest-r12`를 적는다. exact duplicate는 정규화한 UTF-8 bytes의 SHA-256 일치, near duplicate는 고정 3-gram Jaccard `0.80` 이상으로 정의한다. judge는 harmful 확률 `0.60` 이상이면 양성, `0.40` 이하면 음성, 그 사이는 기권한다. 하나라도 바뀌면 같은 EvalID를 재사용하지 않는다.

열두 행은 `safety` 여섯 개, `helpfulness` 네 개, `long-tail-ko` 두 개다. 각 weight는 2, 1, 3이다. model-r17의 success를 `1,1,0,1,0,1 / 1,0,1,1 / 0,1`로 둔다. judge-r8 verdict는 `P,P,N,A,P,N / N,N,P,A / P,N`이다. `A`는 abstention이다. 행 2는 training row와 exact duplicate, 행 5와 9는 near duplicate다. 행 7은 유사해 보여도 Jaccard `0.71`이므로 현재 규칙에서는 clean이다.

exact 집합 `{2}`, near 집합 `{5,9}`, clean 집합 아홉 행을 손으로 적는다. 전체 success는 `8/12`, clean success는 `6/9`다. 두 수의 차이를 곧바로 오염의 인과 효과라고 부르지 않는다. 오염 행과 clean 행의 난도가 다르고 표본도 작다. 여기서 증명한 것은 오염 판정이 분모와 평가 집합을 바꾼다는 사실뿐이다.

## 가중 분모와 paired difference를 검산한다

가중 정확도는 `Σ(weight_i×success_i)/Σweight_i`다. 분모는 `6×2+4×1+2×3=22`다. 분자는 safety 성공 네 개로 8, helpfulness 성공 세 개로 3, long-tail-ko 성공 한 개로 3이므로 14다. 결과는 `14/22`다. 단순 정확도와 다른 까닭은 long-tail-ko 행의 무게가 더 크기 때문이다.

failure D1에서는 rank 0에 safety 여섯 행, rank 1에 나머지 여섯 행을 둔 뒤 local weighted mean을 평균낸다. 두 local denominator는 12와 10이다. local mean 평균은 global numerator/global denominator와 일반적으로 다르다. 최초 divergence는 formatter가 아니라 rank별 numerator와 denominator를 합치는 reduction 단계여야 한다. 수정 뒤에는 두 값을 각각 합산하고 마지막에 한 번만 나눈다.

비교 모델 model-r16의 success를 `1,0,0,1,0,1 / 0,0,1,1 / 0,0`으로 둔다. 같은 CaseID에서 `r17-r16`을 구하면 개선은 2, 7, 12번이고 악화는 없다. paired mean difference는 `3/12`다. 독립된 두 비율로 계산하면 prompt 난도라는 공유 변동을 버리므로 paired 계산을 유지한다.

## 층화 bootstrap과 다중 검정을 고정한다

bootstrap index는 실행 때 무작위로 만들지 않는다. safety에서 여섯 개, helpfulness에서 네 개, long-tail-ko에서 두 개를 복원 추출하는 다섯 묶음을 미리 기록한다. 첫 묶음은 `1,2,2,4,5,6 / 7,8,9,10 / 11,12`, 둘째는 `1,3,3,4,5,6 / 7,7,9,10 / 11,11`이다. 나머지도 결과를 보기 전에 고정한다. 층을 유지하면 작은 long-tail-ko가 재표본에서 통째로 사라지지 않는다. 같은 conversation이나 duplicate family가 여러 행이면 row가 아니라 group을 재표본 단위로 쓴다.

다섯 값의 최소와 최대를 정식 95% 신뢰구간이라고 부르지 않는다. 이 fixture는 계산 경로 검산용이다. 실제 평가는 충분한 resample 수와 사전에 정한 percentile 또는 BCa 방식을 사용한다. primary hypothesis는 clean 전체 paired difference 하나다. 세 slice는 secondary family다. Bonferroni 기준은 각 `0.05/3`이다. 결과를 본 뒤 보정법이나 primary metric을 바꾸지 않는다. 탐색 결과는 다음 고정 평가의 가설로 넘긴다.

## 기권과 judge confusion을 분리한다

judge confusion은 `P/N/A`와 사람 label의 교차표다. coverage는 non-abstain 수를 전체 수로 나눈 값이다. conditional accuracy는 non-abstain에 대해서만 계산하되 coverage와 함께 제시한다. failure J1에서 `A`를 모두 정답으로 바꾸면 최초 차이는 최종 accuracy가 아니라 confusion matrix의 abstain 열과 coverage denominator에서 나야 한다.

J2는 JudgeRevision만 `judge-r9`로 바꾸고 같은 cache를 읽는다. cache key에 revision이 없으면 judge drift가 model change처럼 보인다. J3는 threshold를 0.60에서 0.55로 바꾸면서 revision을 유지한다. 이 입력은 configuration admission에서 거부한다. J4는 장문 응답만 양성으로 치우친 judge를 가정한다. 전체 agreement 대신 length slice confusion과 abstention을 본다. response bytes가 같은데 verdict부터 갈리면 최초 owner는 trainer가 아니라 judge pipeline이다.

## leakage와 분모 실패를 주입한다

L1은 exact 행 2를 clean으로 바꾼다. clean membership digest가 summary보다 먼저 달라져야 한다. L2는 near threshold를 0.70으로 낮춰 행 7까지 오염으로 만든다. policy revision이 그대로면 admission failure다. L3는 train manifest를 r13으로 바꾸고 r12의 near index를 재사용한다. index parent digest 불일치가 검색 결과 전에 실패해야 한다.

D2는 numerator 14를 유지하면서 denominator를 row count 12로 바꾼다. metric record에는 항상 numerator, denominator, weighting policy와 제외된 CaseID를 함께 저장한다. 그러면 최종 값이 그럴듯해도 첫 불일치를 되짚을 수 있다. exact, near, judge와 weighting 변화는 한 번에 하나씩 주입한다. 여러 축을 동시에 바꾸면 최초 원인을 식별할 수 없다.

## 종료 조건

통과하려면 exact·near 집합, clean membership digest, 전체·clean 정확도, weighted numerator/denominator, paired difference, 고정 stratified bootstrap, hypothesis family, judge confusion·coverage·conditional accuracy가 같은 입력 revision을 가리켜야 한다. 각 failure는 예상한 최초 단계에서 멈추고 downstream report를 publish하지 않아야 한다.

제출물에는 EvalID, 네 핵심 revision, 열두 CaseID와 group ID, contamination policy, raw success와 verdict, numerator/count, bootstrap index, multiple-testing family와 failure ledger를 담는다. 모델을 실행하지 않았음도 적는다. 이 실습은 모델 품질을 증명하지 않는다. 평가 파이프라인이 같은 고정 입력에서 같은 계산을 하고 오염·judge·분모 변화의 책임 경계를 보존한다는 제한된 증거다.

배경은 [4장 데이터 오염](../chapters/04-web-corpus.md), [24장 신뢰할 수 있는 평가](../chapters/24-trustworthy-evaluation.md), [25장 red-team 안전 학습](../chapters/25-redteam-safety-training.md)에서 확인한다.
