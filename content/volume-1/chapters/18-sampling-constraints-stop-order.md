# 18장. sampling·constraint·stop 조건의 정확한 순서

16장이 vocabulary score의 producer를, 17장이 normalization과 logprob의 의미를 닫았다면 이 장은 그 score를 실제 선택과 visible output으로 바꾸는 정책을 소유한다. 앞 장의 작은 penalty·temperature 계산은 이 장의 입력 preview이며, 여기서는 processor 순서, constraint support, RNG 주소, reject와 stop commit까지 상태 수명을 완성한다.

모델이 마지막 hidden state에서 어휘 로짓 한 줄을 만들었다고 하자. 아직 사용자에게 보낼 token은 없다. repetition penalty가 history를 읽고 점수를 바꾸며, temperature와 후보 절단이 분포를 바꾸고, grammar가 허용 불가능한 token을 막고, 난수가 하나를 고른다. speculative decoding이라면 고른 후보도 아직 확정이 아니고, stop 문자열이라면 token을 decode한 뒤에도 일부 text를 보류해야 한다. 이 단계를 전부 “sampling”이라고 부르면 옵션의 실제 효과와 rollback 책임을 찾을 수 없다.

이 장은 17장이 정의한 raw logits를 입력으로 받는다. 질문은 하나다. **어느 상태가 어느 순서로 점수와 후보 집합을 바꾸고, 선택된 token이 언제 되돌릴 수 없는 출력이 되는가?** scheduler가 어느 request를 실행할지는 32장에 맡기고, KV page의 구체 rollback은 뒤 cache 장으로 넘긴다. 여기서는 generation 의미론의 accept/commit 경계만 고정한다.

## 18.1 score transforms가 후보 점수를 순서대로 바꾼다

어휘 세 token의 raw logits가 `[2,1,0]`이라고 하자. softmax 확률은 대략 `[0.665,0.245,0.090]`이다. temperature `T=0.5`를 적용하면 logits를 0.5로 나눈 `[4,2,0]`이 되고 확률은 약 `[0.867,0.117,0.016]`으로 날카로워진다. `T=2`이면 `[1,0.5,0]`에서 `[0.506,0.307,0.186]`으로 평평해진다.

temperature 0은 실제 나눗셈 값이 아니다. 구현이 greedy branch로 전환하거나 validation error를 내야 한다. `logits/0`은 infinity와 NaN을 만들 수 있다. 옵션 설명은 `temperature=0이면 결정적`에서 끝나지 않는다. field가 sampling 여부 분기를 바꾸고 RNG 소비를 제거하며 argmax 선택으로 전환하는지 source에서 확인해야 한다.

**top-k·top-p·min-p는 서로 다른 후보 집합을 만든다**

top-k=2는 점수 상위 두 token만 남긴다. 위 분포라면 `{0,1}`이다. top-p=0.8은 확률 내림차순 누적합이 threshold를 넘을 때까지 남긴다. 첫 token 0.665만으로 부족하고 두 번째까지 0.910이므로 역시 `{0,1}`이다. 결과가 같아도 정의는 다르다. 다른 분포에서는 후보 수가 달라진다.

min-p를 최고 확률의 일정 비율 아래 후보를 제거하는 방식으로 정의한 구현이라면 `min_p=0.2`에서 threshold는 `0.665×0.2=0.133`이고 token 2만 제거된다. 절대 probability 0.2 cutoff와 혼동하면 token 1까지 제거한다. 같은 option 이름도 엔진의 정확한 수식과 적용 시점을 확인해야 한다.

typical sampling은 각 token의 surprisal이 분포 entropy에서 얼마나 벗어나는지를 사용해 전형적인 후보를 남기는 계열이다. 단순히 확률 상위부터 자르는 top-p와 축이 다르다. “낮은 확률 제거”로만 요약하면 왜 최고 확률 token이 설정에 따라 먼저 제외될 수도 있는지 설명하지 못한다.

Transformers의 temperature, top-k, top-p, min-p, typical processor 구현은 [Transformers v5.15.1 `logits_process.py:280-590`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/logits_process.py#L280-L590)에서 각각의 mask와 threshold를 확인할 수 있다. class 이름보다 입력 score가 raw logits인지 log-softmax인지, 최소 보존 token 수와 filter value가 무엇인지 읽는다.

### 순서가 비가환인 것을 수치로 본다

분포 `[0.6,0.25,0.15]`에서 top-p=0.7을 먼저 적용하면 `{0,1}`이 남는다. 그 뒤 temperature를 적용해도 token 2는 돌아오지 않는다. 반대로 높은 temperature로 먼저 평탄화해 `[0.45,0.32,0.23]`이 되었다고 하면 top-p=0.7은 세부 inclusive 정책에 따라 `{0,1}` 또는 threshold를 넘기는 최소 집합을 남긴다. 정확한 수치는 logits로 계산해야 하지만 후보 절단과 rescale이 교환되지 않는다는 사실은 분명하다.

repetition penalty와 top-k도 비가환이다. top-k가 먼저 token X를 버리면 뒤 penalty가 다른 token을 낮추어도 X는 복귀하지 않는다. penalty가 먼저면 X가 top-k에 들어올 수 있다. processor list의 순서가 generation semantics의 일부인 이유다.

Transformers가 generation config에서 processor list를 구성하는 경계는 [Transformers v5.15.1 `utils.py:1260-1515`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L1260-L1515)다. option이 존재한다는 사실보다 어느 processor가 어느 순서로 append되고 custom processor와 충돌을 어떻게 다루는지 본다.

**softmax를 하지 않고도 순위를 바꿀 수 있다**

processor 대부분은 확률이 아니라 logits를 직접 다룬다. logits에 같은 상수를 더하면 softmax 확률은 변하지 않지만 특정 token만 바꾸면 상대 확률이 달라진다. `[3,2,1]`에 token 0만 `-2` bias를 주면 `[1,2,1]`이 되고 top-1이 token 1로 바뀐다. 확률을 먼저 materialize하지 않아도 순위와 support를 바꿀 수 있다.

softmax는 수치 안정성을 위해 최대 logit을 빼고 exponentiation한다. `[1001,1000,999]`는 `[1,0,-1]`로 옮겨도 같은 확률이다. processor trace에서 raw와 shifted logits를 비교할 때 이 공통 상수 차이를 semantic divergence로 오인하지 않는다. 후보 순위와 normalized probability를 함께 본다.

filter value가 `-inf`이면 뒤 softmax 확률이 정확히 0이 된다. 매우 큰 음수 finite 값은 dtype과 kernel에 따라 underflow로 0처럼 보일 수 있지만 계약은 다르다. 모든 원소가 `-inf`이면 max subtraction에서 `-inf-(-inf)`가 NaN이 될 수 있다. finite count와 nonmasked count를 processor마다 기록하는 이유다.

### temperature와 additive bias의 비가환성

raw `[2,1]`, token 1 bias `+1`을 생각하자. bias 후 temperature 0.5면 `[4,4]`가 되어 동률이다. temperature 후 bias면 `[4,2]+[0,1]=[4,3]`으로 token 0이 우세하다. temperature는 기존 logits뿐 아니라 그 전에 더한 bias까지 scale하지만 뒤 bias는 scale하지 않는다.

API에서 logit bias, penalty, temperature가 모두 제공될 때 순서는 public semantics다. “결국 다 점수를 바꾼다”는 설명으로는 엔진 parity를 판단할 수 없다. 각 transform 입력과 출력을 bounded top-k slice로 남기고, 첫 순서 차이를 찾는다.

**top-k tie는 후보 수보다 많은 것을 묻는다**

logits `[5,4,4,1]`에서 top-k=2라면 token 0과 동률인 1·2 가운데 어느 하나를 남겨야 한다. stable sort가 ID 순서를 보존하는지, GPU select가 임의 tie ordering을 쓰는지에 따라 support가 달라진다. seed가 같아도 support가 다르면 출력이 다르다.

`min_tokens_to_keep`가 있으면 k나 p가 극단적이어도 최소 후보 수를 남길 수 있다. top-p=0이 정확히 한 token을 남기는지 validation error인지, 최소 보존 설정이 EOS에 별도 적용되는지 구현을 본다. option 숫자만 비교하지 않는다.

반증 fixture는 의도적인 동률 logits와 서로 다른 token IDs를 쓴다. processed support, cutoff score, tie token ordering을 비교한다. 실제 model logits에서 우연한 exact tie를 기다릴 필요가 없다. 이 장의 손계산 oracle은 production 성능을 재는 것이 아니라 semantics를 고정한다.

### top-p 경계의 inclusive 정책

확률 `[0.5,0.3,0.2]`, p=0.8에서 누적합은 `[0.5,0.8,1.0]`이다. threshold와 정확히 같은 두 번째 token을 남기는지, 초과하는 첫 token을 보장하기 위해 shift mask를 하는지 구현 세부가 결과를 바꾼다. 일반적으로 최소 한 token은 남겨야 한다.

정렬 뒤 mask를 원래 vocabulary order로 scatter하는 단계도 있다. 정렬 index를 잘못 역매핑하면 후보 확률은 맞아 보여도 다른 ID가 살아남는다. 관측에는 sorted score, cumulative probability, sorted removal mask, original-order mask를 둔다.

top-p 비용을 확률 계산 탓으로만 돌리지 않는다. vocab 정렬 또는 selection, batch별 다른 p, mask scatter, synchronization이 후보 원인이다. source로 연산 경로를 찾고 실제 latency 비율은 profiler 없이는 단정하지 않는다.

**min-p와 top-p를 순서대로 계산한다**

확률 `[0.55,0.25,0.12,0.08]`, min-p ratio 0.2라면 최고 확률 기준 threshold는 0.11이어서 앞 세 token이 남는다. 그 결과를 재정규화하면 대략 `[0.598,0.272,0.130]`이다. 이어 top-p=0.8이면 앞 두 token 누적이 0.870이므로 `{0,1}`이 남는다.

top-p를 원 분포에 먼저 적용하면 p=0.8에서 `{0,1}`이 남고 min-p는 둘 다 유지한다. 이 예에서는 최종 support가 같지만 중간 probability가 다르다. 경계값을 바꾸면 최종도 달라진다. 같은 결과 예제 하나로 교환법칙을 주장하지 않는다.

min-p가 logits에서 max-logit-relative threshold로 구현되는지 softmax probability를 materialize하는지에 따라 numeric 경로가 다를 수 있다. 수학적으로 동등한 변환도 finite precision과 dtype에서 경계 token이 갈릴 수 있다. cutoff 주변 fixture와 tolerance를 둔다.

### typical sampling의 손계산

확률 `[0.5,0.25,0.125,0.125]`의 entropy를 natural log로 계산하면 약 `1.213`이다. 각 surprisal `-log p`는 `[0.693,1.386,2.079,2.079]`이고 entropy와 거리는 `[0.520,0.173,0.866,0.866]`이다. typical ranking은 token 1을 token 0보다 먼저 둘 수 있다. probability ranking과 다르다.

typical mass threshold까지 이 순서로 확률을 누적해 후보를 고른다. “가장 높은 확률을 보존한다”는 top-p 직관이 그대로 적용되지 않는다. 구현이 최소 token 보존과 mask shift를 어떻게 하는지 본다.

entropy 계산에 이미 normalized log-probability가 필요하므로 temperature와 penalty가 앞에서 분포를 바꾸면 typical distance도 모두 바뀐다. typical을 processor chain 어디에 놓는지가 단순 성능 순서가 아니라 의미론이다.

이 절의 종료 조건은 최종 sampled ID가 같다는 것이 아니다. 각 transform 뒤 support와 top scores가 reference 손계산과 맞고, 설정을 하나 바꾸었을 때 예상한 branch와 state만 달라져야 한다. 첫 transform부터 다르면 RNG와 stop으로 내려가지 않는다.

**penalty history는 score transform의 request state다.**

repetition, presence, frequency penalty는 이전에 나온 token을 알아야 한다. 이 history가 prompt를 포함하는지 generated tokens만 포함하는지, accepted token만 포함하는지가 결과를 바꾼다. request slot이 재배치될 때 penalty state가 다른 사용자 row와 섞이면 같은 raw logits와 seed에서도 결과가 달라진다.

repetition penalty의 흔한 구현은 양수 logit은 penalty로 나누고 음수 logit은 penalty를 곱한다. `[2,-1]`에 penalty 2를 적용하면 `[1,-2]`다. 모든 점수를 무조건 나누면 음수는 `-0.5`로 올라가 반복 token을 오히려 유리하게 만든다. sign-dependent transform을 작은 수치 fixture로 고정한다.

presence penalty는 등장 여부, frequency penalty는 등장 횟수를 사용할 수 있다. prompt에 token A가 세 번, output에 한 번 있었다면 history policy에 따라 count가 4, 1, 또는 speculative accepted prefix 기준의 다른 값이 된다. API 호환이라는 이름만으로 동일 결과를 기대하지 않는다.

Transformers repetition processor의 sign branch와 gather/scatter는 [Transformers v5.15.1 `logits_process.py:900-980`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/logits_process.py#L900-L980)에서 확인한다. vLLM sampler의 penalties 경로는 [vLLM v0.27.1 `sampler.py:180-310`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L180-L310)에서 request별 tensors와 적용 순서를 따라간다.

옵션을 인과로 쓰면 이렇다. `frequency_penalty` field가 penalty-enabled branch를 켜고 request token counts tensor를 소비해 score row를 바꾼다. 기대 효과는 반복 억제지만 prompt 포함 정책과 sign 수식에 따라 강도가 달라진다. 반증 관측은 raw score, history token/count, transformed score다. 최종 text만 보면 tokenizer와 RNG 차이를 분리할 수 없다.

**history의 네 길이를 분리한다**

request에는 prompt IDs, accepted output IDs, speculative draft IDs, visible decoded IDs가 있다. repetition processor가 어느 열을 읽는지 명시해야 한다. rejected draft는 history에서 제거되어야 하고, stop marker를 output에서 숨겨도 accepted token history에는 존재할 수 있다. visible text와 penalty history는 같은 단위가 아니다.

`penalty_last_n=64`도 어느 열의 마지막 64개인지 묻는다. prompt와 output을 이어 붙인 열인지, output만인지, context 전체보다 짧을 때 어떻게 처리하는지 본다. `-1` 같은 sentinel이 context 전체를 뜻하는 구현도 있을 수 있다. field validation과 derived window를 함께 기록한다.

batch row compaction에서 history tensor와 sampling params가 같은 permutation으로 이동해야 한다. token IDs만 새 row로 옮기고 frequency count가 옛 row에 남으면 다른 사용자의 반복을 벌한다. mixed batch에서만 품질이 이상한 장애의 강한 후보다.

**presence와 frequency를 수치로 분리한다**

token A count가 3이고 raw logit이 2라고 하자. presence penalty 0.5만 적용하면 등장했다는 이유로 1.5가 된다. frequency penalty 0.2라면 count에 비례해 1.4가 된다. 둘 다 적용하면 정의에 따라 0.5와 0.6을 빼 0.9가 될 수 있다. token B count 1에는 frequency penalty가 0.2뿐이다.

positive penalty만 생각해서도 안 된다. negative presence/frequency 값은 이전 token을 장려할 수 있다. validation range와 sign semantics를 확인한다. “반복 억제 옵션”이라는 이름이 음수 설정의 실제 효과를 설명하지 못한다.

count를 prompt에 포함하면 system marker와 whitespace token도 벌점 대상일 수 있다. chat protocol token이 output 후보에 거의 나오지 않아 영향이 작을 수 있지만 일반화하지 않는다. prompt 포함 toggle 또는 engine default를 source에서 확인한다.

**repetition penalty의 sign branch 손계산**

history에 IDs 0과 2가 있고 logits `[3,-2,-0.5,1]`, penalty 1.5라 하자. ID 0의 양수는 `3/1.5=2`, ID 2의 음수는 `-0.5×1.5=-0.75`가 된다. IDs 1과 3은 그대로다. 결과 `[2,-2,-0.75,1]`이다.

ID 0이 history에 다섯 번 있어도 repetition penalty가 set membership만 본다면 한 번만 변환한다. frequency penalty와 차이다. 구현이 gather한 history에 duplicate가 있을 때 scatter가 중복 적용되는지 unique semantics인지 확인한다. 중복으로 다섯 번 나누면 전혀 다른 processor가 된다.

penalty 1.0은 identity여야 하고 processor를 아예 만들지 않는 branch일 수 있다. 값은 같지만 processor list 길이와 overhead가 다를 수 있다. config에 1.0이 있다는 이유로 kernel이 실행되었다고 단정하지 않는다.

**history mutation의 commit 시점**

sampled token을 즉시 count에 넣고 뒤 grammar 또는 speculative acceptance가 거부하면 state를 되돌려야 한다. 더 안전한 구조는 accept 뒤 commit하는 것이다. 구현이 provisional update를 쓰면 undo log나 accepted length crop이 필요하다.

stop token도 accepted token이다. EOS를 감지해 loop를 끝내더라도 penalty/grammar history에 commit한 뒤 request를 종료할 수 있다. visible output에서 EOS를 숨긴다는 사실과 state transition을 혼동하지 않는다.

장애 fixture는 draft `[a,b,c]`, accepted `[a,b]` 뒤 다음 step raw logits를 고정한다. c가 history에 남았을 때만 달라지는 penalty token score를 본다. first divergence가 history count라면 sampler 수학이나 RNG를 조사할 필요가 없다.

**장애 사건: request slot 교환 뒤 다른 사용자의 min-p가 적용된다**

증상은 단독 요청에서는 재현되지 않고 한 request가 끝나 batch compaction이 일어난 직후 남은 request 출력만 달라지는 것이다. raw logits는 reference와 같다. processor 뒤 support가 다르며 남은 request가 종료된 request의 min-p 값을 받은다.

최초 divergence는 batch row→sampling param mapping이다. 함수 분기는 finished row 제거와 sampling batch info update다. input IDs와 history뿐 아니라 temperature, top-k, top-p, min-p, generator, grammar state를 같은 index_select/permutation으로 이동했는지 본다.

반증은 A와 B에 극단적으로 다른 설정을 주고 `[A,B]`, `[B,A]`, A 조기 종료 세 fixture를 비교하는 것이다. request ID별 processed support가 batch lifecycle과 무관하게 같아야 한다. 최종 text만 비교하면 RNG 때문에 불필요한 noise가 생기므로 sampling 직전 mask를 본다.

**같은 확률 벡터로 transform 순서를 검산한다.**

후보 제한 옵션을 “낮을수록 보수적”이라는 한 줄로 설명하면 서로 다른 알고리즘의 의미가 사라진다. 여섯 token의 normalized probability를 `[0.40, 0.25, 0.15, 0.10, 0.06, 0.04]`라 하자. token ID는 0~5이고 이미 큰 값부터 정렬돼 있다. 이 작은 배열로 각 selector가 무엇을 보고 무엇을 버리는지 계산한다.

`top_k=3`은 확률 크기 상위 세 token 0,1,2만 남긴다. 남은 질량은 0.80이므로 재정규화하면 `[0.50,0.3125,0.1875]`다. k는 확률 임계값이 아니라 후보 개수다. 분포가 뾰족해도 세 개, 평평해도 세 개를 남긴다. tie가 cutoff에 걸리면 어느 global ID가 남는지 deterministic policy가 필요하다.

`top_p=0.70`은 큰 확률부터 누적해 threshold에 도달하는 최소 prefix를 남긴다고 가정하자. token 0까지 0.40, token 1까지 0.65, token 2까지 0.80이므로 0,1,2가 남는다. cutoff를 넘긴 token을 포함해야 누적 질량이 최소 p가 된다. 구현이 cutoff token을 shift해서 포함하는지 source와 fixture로 확인한다. 같은 벡터에서 p=0.65라면 floating rounding과 inclusive 규칙이 후보 수를 바꿀 수 있다.

`min_p=0.20`을 최고 확률에 대한 상대 threshold로 정의하면 기준은 `0.40×0.20=0.08`이다. token 0~3은 남고 0.06과 0.04는 사라진다. top-p가 누적 질량을 보는 반면 min-p는 최고 후보 대비 각 후보의 상대 크기를 본다. 분포가 뾰족하면 threshold 자체가 커져 tail을 더 강하게 자르고, 최고 확률이 낮으면 더 많은 후보를 허용한다.

typical sampling은 각 token의 surprisal `-log p_i`가 entropy `H=-sum p log p`에서 얼마나 떨어졌는지를 본다. 확률 순위와 typicality 순위가 같지 않을 수 있다. 가장 높은 token이 평균 정보량에서 멀면 typical prefix에서 뒤로 갈 수 있다. 따라서 “top-p의 다른 구현”이라고 설명하면 안 된다. entropy 계산과 deviation sort, cumulative mass라는 추가 상태가 있다.

grammar가 허용 집합 `{1,3,4}`를 만든다고 하자. grammar를 먼저 적용하고 재정규화하면 원래 질량 0.41에서 `[0.6098,0.2439,0.1463]`가 된다. 그 뒤 top-p=0.70을 적용하면 token 1과 3이 남는다. 반대로 원래 분포에서 top-p=0.70을 먼저 적용하면 `{0,1,2}`이고 grammar와 교집합은 token 1 하나뿐이다. 두 순서는 모두 문법적으로 유효할 수 있지만 support와 확률이 전혀 다르다.

더 위험한 반례는 top-k=2다. top-k 먼저면 `{0,1}`, grammar 교집합은 `{1}`이다. grammar 먼저면 허용 score 순위가 1,3,4이므로 top-k 결과는 `{1,3}`이다. structured generation이 “문법상 허용된 후보 중 상위 k”를 약속한다면 grammar mask가 selector 전에 들어가야 한다. 엔진이 다른 계약을 갖는다면 그것을 명시해야 한다. 옵션 이름만으로 순서를 추측하지 않는다.

**배열 상태를 support와 mass로 나눠 기록한다**

각 단계는 score, support, normalization mass를 바꿀 수 있다. grammar mask는 invalid token을 `-inf`로 만들어 support를 바꾼다. top-k와 top-p도 support를 줄인다. temperature는 support를 보통 유지하지만 mass를 바꾼다. penalty와 bias는 score를 바꾸고 간접적으로 support selector 결과를 바꾼다. debug trace에 후보 수만 남기면 왜 후보가 사라졌는지 알 수 없다.

작은 fixture에는 `(stage, allowed_ids, pre_renorm_mass, normalized_probs, cutoff)`를 둔다. grammar 뒤 allowed count가 3이고 top-p 뒤 2라는 사실, grammar 전 mass가 0.41이었다는 사실이 보인다. empty support에서는 softmax를 부르기 전에 명시적 error 또는 fallback을 선택한다. all `-inf`를 softmax하면 NaN이 되어 최초 원인이 numerical failure처럼 보인다.

vLLM 고정 sampler는 temperature 뒤 argmax-invariant processor를 적용하고 top-k/top-p sampler를 호출한다. top-k/top-p 연산은 logprob 반환 mode에 따라 processed log-softmax도 만들 수 있다.

실제 grammar processor가 어느 processor bucket에 들어오는지 metadata 구성까지 읽어야 한다. [vLLM sample ordering](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L258-L306), [vLLM top-k/top-p sampler](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/ops/topk_topp_sampler.py#L119-L207)

SGLang speculative utility는 top-k와 top-p probability renormalization helper를 구분하고, fallback top-k에서 `torch.topk` 뒤 rank mask와 합 normalization을 수행한다. 같은 파일이 grammar mask가 있으면 speculative 경로의 조건을 바꾸는 것도 보여 준다.

이는 일반 sampler의 모든 순서를 증명하는 링크가 아니라, speculative fast path가 selector와 grammar capability를 별도로 판정한다는 증거다. [SGLang speculative top-k·top-p renormalization](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/speculative/dflash_utils.py#L107-L136), [SGLang grammar mask fast-path 조건](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/speculative/dflash_utils.py#L235-L266)

**padded vocabulary와 bitmask word 경계를 시험한다**

grammar bitmask가 32-bit word로 vocabulary를 표현한다고 하자. 실제 V=65이면 세 word가 필요하고 마지막 word의 bit 1~31은 padding이다. 마지막 유효 token 64만 허용했을 때 mask는 word 2의 bit 0만 1이어야 한다. padding bit가 1로 남고 그 score가 크면 invalid global ID가 선택될 수 있다. logits trim과 grammar mask padding이 같은 original vocabulary 계약을 써야 한다.

TP shard가 bitmask를 local slice로 받으면 global token 64의 owner와 local bit를 계산한다. word boundary와 shard boundary가 일치하지 않을 수 있다. global bitmask를 단순 word 단위로 나누면 한 word가 두 shard에 걸치는 경우가 생긴다. 구현이 global mask를 적용하는지 local mask를 재구성하는지 source를 읽고, V=31,32,33,63,64,65 fixture를 둔다.

adapter token이 추가되면 grammar compiler가 그 token의 byte piece를 아는지도 확인한다. tokenizer revision과 grammar vocabulary mapping이 다르면 schema는 compile되지만 added token이 영원히 금지되거나 잘못 허용될 수 있다. grammar cache key에는 schema digest뿐 아니라 tokenizer/vocab identity와 backend option이 필요하다.

이 절의 완료 조건은 모든 selector를 켠 결과 한 문장이 그럴듯한 것이 아니다. 각 selector가 약속한 입력 stage를 받고, support와 mass가 손계산과 같고, grammar·padding 뒤 empty support가 명시적으로 처리되며, optimized kernel과 reference가 cutoff tie까지 같은 결과를 만드는 것이다.

## 18.2 constraint가 허용 support를 확정하고 empty set을 처리한다

structured output은 생성 뒤 JSON parser로 검사하는 것과 생성 전에 grammar로 후보를 제한하는 것이 다르다. grammar state가 현재 prefix에서 허용 가능한 token set을 계산하고 나머지 score를 `-inf`로 mask하면 sampling support가 바뀐다. 선택 token을 accept한 뒤 grammar automaton state도 전진해야 한다.

문자 grammar와 token vocabulary 사이에는 경계가 있다. token 하나가 여러 문자, partial UTF-8, grammar delimiter 일부를 포함할 수 있다. 단순히 token string 첫 문자만 검사하면 안 된다. llama.cpp grammar는 candidate token piece를 decode하고 partial UTF-8 state와 grammar stack으로 rejection을 계산한다.

적용은 [llama.cpp v0.2.0 `llama-grammar.cpp:1353-1394`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-grammar.cpp#L1353-L1394), accept state 전진은 [같은 파일 `:1396-1453`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-grammar.cpp#L1396-L1453)에 있다.

모든 후보가 `-inf`이면 softmax denominator가 0이거나 NaN이 될 수 있다. 이 empty support를 임의 token fallback으로 숨기면 invalid structured output을 만든다. grammar/config contradiction, EOS policy, numeric corruption을 구별해 명시적 error 또는 정의된 recovery를 해야 한다.

grammar mask와 top-k 순서도 중요하다. top-k가 먼저 grammar-valid token을 모두 제거하고 grammar가 뒤에서 invalid만 제거하면 empty support다. grammar를 먼저 적용하고 valid set 안에서 top-k를 고르면 후보가 남을 수 있다. engine의 ordered pipeline을 source로 확인하고 동일 API 옵션 이름만 비교하지 않는다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- SGLang constrained decoding backend와 mask update 경계는 [SGLang v0.5.18 `sampling_batch_info.py:220-390`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/sampling/sampling_batch_info.py#L220-L390)에서 grammar object와 vocab mask lifecycle을 따라간다.
- vLLM structured output state는 [vLLM v0.27.1 `structured_output/`](https://github.com/vllm-project/vllm/tree/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/structured_output)에서 request grammar compile, bitmask, batch update owner를 나눠 읽는다.

**JSON prefix를 token support로 바꾸는 손실 없는 질문**

현재 visible prefix가 `{"x":`라고 하자. schema가 x를 integer로 요구하면 다음 문자는 whitespace, minus sign, digit 등이 가능하고 quote는 불가능할 수 있다. 그러나 tokenizer token은 `" 12"`, `"-3"`, `"null"`처럼 여러 문자를 담는다. grammar는 token piece 전체가 어떤 automaton transition을 만드는지 보아야 한다.

token piece가 valid prefix로 시작해 뒤에서 invalid해지면 전체 token을 reject해야 한다. 반대로 UTF-8 code point 일부만 담은 token은 다음 token과 결합해 valid 문자가 될 수 있어 partial byte state가 필요하다. 문자 하나와 token 하나를 대응시키는 설명의 한계다.

EOS는 grammar가 완결 상태일 때만 허용되어야 할 수 있다. JSON object가 닫히지 않았는데 EOS를 허용하면 syntactically incomplete output이다. grammar stack이 accepting state인지와 tokenizer의 여러 EOG IDs를 함께 본다.

### bitmask의 shape와 vocabulary padding

GPU structured-output backend는 vocab allowed mask를 bitset으로 표현할 수 있다. logical vocab V를 32-bit word로 pack하면 대략 `ceil(V/32)` words per row다. TP shard 또는 padded vocab에서는 logical ID와 bit position mapping을 맞춰야 한다.

physical padding IDs가 mask에서 우연히 allowed여도 sampling 대상으로 들어가면 안 된다. grammar compiler가 logical vocab만 알고 sampler가 padded logits를 다룬다면 tail bits를 명시적으로 차단한다. empty support count도 physical padding을 제외한 logical 후보로 계산한다.

batch reorder 때 grammar state object와 compiled mask row가 함께 이동해야 한다. 서로 다른 JSON schema request가 mask를 바꾸어 가지면 valid JSON이지만 다른 schema를 만족하는 silent failure가 될 수 있다. request ID와 grammar identity digest를 batch row에 묶는다.

**processor와 grammar의 순서를 수치로 반증한다**

logits `[5,4,3]`, grammar valid set `{2}`, top-k=2를 생각하자. top-k 먼저면 token 2가 제거되고 grammar 뒤 support는 empty다. grammar 먼저면 `[−inf,−inf,3]`이고 top-k의 최소 보존 정책이 token 2를 남긴다. 구조화 출력에서는 두 번째가 의도일 가능성이 높지만 실제 엔진 계약을 source로 확인한다.

반대로 repetition penalty가 grammar mask 뒤에 적용되어 `-inf`를 산술 변환해도 보통 `-inf`는 유지되지만, custom processor가 finite sentinel로 바꾸면 invalid token이 부활할 수 있다. allowed mask가 마지막에 재적용되는지 또는 processor들이 mask invariant를 보존하는지 본다.

logit bias가 grammar-invalid token에 `+inf`를 주는 극단 fixture도 유용하다. 최종 support에서 반드시 제거되어야 한다. invalid token이 살아나면 order 또는 numeric handling이 잘못되었다.

### empty support를 세 원인으로 나눈다

첫째, grammar와 prefix가 논리적으로 모순이다. 이미 invalid prefix를 accept했거나 schema가 불가능하다. 둘째, grammar-valid token은 있었지만 top-k/ban/penalty 같은 다른 constraint가 모두 제거했다. 셋째, numeric corruption으로 valid logits가 NaN 또는 `-inf`가 되었다.

오류 처리에서 이 원인을 구분한다. grammar compile error, runtime no-valid-token, nonfinite-score는 사용자 수정과 운영 대응이 다르다. 임의 EOS fallback은 grammar 완결 상태가 아니면 잘못된 output을 성공처럼 반환한다.

metric에는 grammar backend, compile failure, empty-support count와 step, allowed count before/after other processors를 bounded label/histogram으로 둔다. schema 원문을 label로 넣으면 cardinality와 정보 유출 문제가 있다. 승인된 digest와 fixture ID를 사용한다.

**grammar state는 accepted token에만 전진한다**

candidate를 검사하기 위해 tentative transition을 만들 수 있지만, sampled 뒤 speculative target에서 reject되면 canonical grammar state는 전진하면 안 된다. clone/rollback 또는 accept-only update가 필요하다.

llama.cpp의 apply와 accept가 별도 함수인 점은 이 의미를 잘 드러낸다. apply는 현재 state에서 candidates를 거르고 accept는 확정 token piece로 stacks와 partial UTF-8을 전진시킨다. 두 호출 사이 token identity가 바뀌면 안 된다.

장애 fixture는 grammar상 `a` 뒤 `b`만 가능한 상태에서 draft `a,c`, accepted `a`를 만든다. 다음 allowed set이 b를 포함해야 한다. rejected c가 grammar state에 남으면 empty support 또는 엉뚱한 branch가 나온다.

### 장애 사건: schema 요청에서만 NaN

증상은 unconstrained sampling은 정상이고 특정 prefix에서 grammar를 켜면 multinomial이 invalid probability error를 낸다. 최초 divergence는 grammar mask 뒤 finite/allowed count 0이다. source 분기는 mask application order와 automaton accepting state다.

반증은 같은 raw logits에 grammar mask만 on/off하고 processor별 count를 기록한다. grammar 전 valid token이 하나라도 있는지, top-k가 앞서 제거했는지, EOS가 완결 상태에서 허용되는지 본다. 수정 뒤에는 contradiction에 명시적 error가 나고 valid 최소 prefix에는 한 후보 이상 남아야 한다.

## 18.3 RNG는 seed와 request별 draw 주소를 함께 소유한다

같은 seed는 같은 난수 stream의 시작을 뜻할 수 있지만, 어느 request가 몇 개를 언제 소비하는지까지 같다는 뜻은 아니다. batch row 순서, finished row 제거, top-k tie, speculative draft 수, distributed rank가 RNG consumption order를 바꾼다.

두 request A와 B가 같은 generator를 공유해 난수 `u1,u2,u3...`를 소비한다고 하자. batch `[A,B]`에서는 A가 `u1`, B가 `u2`를 쓸 수 있다. row reorder 뒤 `[B,A]`인데 request generator가 아니라 batch generator라면 B가 `u1`을 받는다. request별 generator와 stable request identity가 필요한 이유다.

분산 vocab에서 각 rank가 local top-1 또는 local sample을 고르고 나중 합치면 global distribution의 표본이 아니다. logits shard를 global selection 의미에 맞게 gather/reduce하거나 distributed sampling algorithm을 써야 한다. collective의 floating reduction과 tie ordering도 bitwise determinism에 영향을 줄 수 있다.

vLLM sampler가 generators와 sampled output을 다루는 주 경계는 [vLLM v0.27.1 `sampler.py:90-180`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L90-L180)다. SGLang sampling batch state는 [SGLang v0.5.18 `sampling_batch_info.py:60-220`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/sampling/sampling_batch_info.py#L60-L220)에서 request-row parameter tensor와 update를 읽는다.

determinism을 세 등급으로 나눈다. 동일 backend·batch shape에서 재현되는가, batch reorder에도 request별로 재현되는가, 다른 backend/TP size에서도 같은 token인가. 마지막은 floating tie와 algorithm 차이 때문에 더 강한 계약이다. seed가 같다는 한 문장으로 어느 등급도 자동 보장하지 않는다.

**categorical sample을 구간으로 손계산한다**

확률 `[0.5,0.3,0.2]`의 누적 구간을 `[0,0.5)`, `[0.5,0.8)`, `[0.8,1)`로 놓자. uniform draw `u=0.72`이면 token 1, `u=0.91`이면 token 2다. 같은 u라도 processed probability가 `[0.7,0.2,0.1]`로 바뀌면 0.72는 token 1이지만 경계가 다르다. RNG divergence와 distribution divergence를 분리해야 한다.

floating cumulative sum이 정확히 1이 아닐 수 있고 `u`가 boundary와 같을 때 inclusive 규칙이 필요하다. production categorical kernel이 이 단순 inverse-CDF를 그대로 쓰지 않을 수 있지만, 작은 oracle은 입력 probability와 난수에서 expected token을 설명하는 기준이다.

Gumbel-max나 exponential race 같은 다른 표본화 구현은 token마다 난수를 소비할 수 있다. 단일 uniform 하나를 소비하는 categorical과 RNG stream advance가 다르다. 같은 seed·확률이어도 algorithm이 다르면 같은 token sequence를 약속하지 않을 수 있다.

### greedy도 tie 정책을 가진다

temperature 0에서 argmax를 쓰더라도 logits `[2,2,1]`의 tie를 어느 ID가 이기는지 정해야 한다. 흔히 첫 index지만 parallel reduction의 pair ordering이 다르면 달라질 수 있다. greedy를 “완전 결정적”이라고 부르기 전에 동일 reduction/backend 범위를 명시한다.

NaN이 하나 섞인 argmax의 동작도 backend별로 믿지 않는다. sampler 입구에서 finite invariant를 검사하거나 앞 processor에서 empty support를 잡는다. NaN tie를 deterministic하게 고르는 것은 correctness가 아니다.

**request별 generator와 batch generator**

request별 generator는 batch reorder에서도 각 request stream을 유지하기 쉽다. 그러나 request가 speculative 후보 수에 따라 다른 수의 draw를 소비하면 이후 sequence는 달라진다. generator state에는 seed뿐 아니라 offset/counter가 포함된다.

batch generator는 vectorized sampling에 편하지만 row removal과 padding row가 draw를 소비하는지 계약해야 한다. finished row에도 난수를 뽑고 결과를 버리면 살아 있는 row의 다음 draw offset이 batch shape에 의존한다. fixed-size random tensor를 매 step 만드는 구현에서 생길 수 있다.

관측에는 generator object/request key, seed, step 전후 offset 또는 재현 가능한 state digest, draw shape, active row mask를 둔다. 난수 값 전체를 production log에 남길 필요는 없지만 격리된 fixture에서는 bounded draw를 비교할 수 있다.

### tensor parallel determinism의 세 분기

첫 방법은 global logits를 한 rank에 모아 sample하고 selected ID를 broadcast하는 것이다. 의미는 단순하지만 vocab-sized communication과 root work가 있다. 둘째는 distributed 확률 알고리즘으로 shard local reduction과 global normalization/selection을 수행한다. 셋째로 각 rank local sample 후 임의 결합을 하는 것은 일반적으로 global categorical과 같지 않다.

global max와 sum-exp reduction 순서가 TP size에 따라 바뀌면 last-bit probability가 달라질 수 있다. cutoff나 tie 근처에서는 후보 support 또는 선택이 바뀐다. 이를 무조건 bug라 하지 않고 promised tolerance/determinism level을 확인한다.

rank마다 같은 seed를 준다고 해결되지 않는다. local vocab 크기와 mask가 달라 각 rank 난수 소비와 후보가 다르다. 어느 rank가 RNG owner인지, selected global ID를 어떻게 공유하는지 source로 닫는다.

### 장애 사건: 같은 seed인데 batch 순서만 바꾸면 출력이 달라진다

raw와 processed logits를 request identity로 정렬했을 때 같다. 최초 divergence는 uniform draw다. source 분기는 sampler가 request-specific generator dict를 찾는지, global generator로 batch-shaped random tensor를 만드는지다.

반증 fixture는 A와 B에 같은 길이와 다른 길이를 각각 주고 순서를 바꾼다. 첫 step, B가 먼저 종료한 다음 step, speculative 후보가 있는 step을 나눈다. request별 generator state 전후와 sampled token을 비교한다.

수정은 무조건 request generator로 바꾸는 것이 아니다. API가 batch-shape-dependent reproducibility만 약속할 수도 있고 vectorized path 성능 tradeoff가 있다. 원하는 계약을 먼저 정하고 state ownership을 맞춘다. 문서와 metric에는 reproducibility scope를 명시한다.

**batch reorder에서도 request별 draw 주소를 보존한다.**

seed 42는 난수열을 시작하는 재료일 뿐, 어떤 request가 어느 난수를 소비하는지를 정의하지 않는다. 하나의 batch-global generator를 사용하면 batch에 포함된 row와 kernel의 draw 순서가 request 결과를 결정한다. A와 B가 각 한 번씩 draw할 때 순서가 A,B면 A가 `u0`, B가 `u1`을 받는다. B가 먼저 끝나거나 batch가 B,A로 compact되면 A가 다른 값을 받을 수 있다.

request별 generator는 이 결합을 줄인다. state key를 `(seed, request_incarnation)`으로 두고 request가 commit할 sample position마다 counter를 진행한다. 하지만 speculative decoding에서는 한 logical output token에 proposal, acceptance coin, recovered token sampling 등 여러 draw branch가 있을 수 있다. 단순 `generated_length`만 counter로 쓰면 reject pattern이 달라질 때 주소가 충돌한다.

명시적인 주소는 `(seed, request_incarnation, logical_step, branch, lane)`처럼 생각할 수 있다. branch는 ordinary categorical, speculative accept, residual resample을 구분한다. lane은 여러 draft token 또는 후보 draw를 구분한다. 실제 구현이 counter-based RNG를 쓰지 않더라도 이 좌표로 trace를 설계하면 어느 소비가 sequence를 밀었는지 설명할 수 있다.

**세 request의 batch reorder 손계산**

uniform stream이 `[0.12,0.83,0.41,0.67,...]`이고 A,B,C가 같은 분포 `[0.5,0.3,0.2]`에서 하나씩 sample한다고 하자. batch-global 순서 A,B,C이면 token은 A=0, B=2, C=0이다. B가 취소되어 A,C만 실행되면 A=0, C=2가 된다. C의 모델·분포·seed는 같지만 결과가 바뀐다.

request별 stream이라면 A는 A의 첫 draw, C는 C의 첫 draw를 받는다. batch order가 바뀌어도 같다. 그러나 generator object 배열이 physical batch row에 묶여 있고 compaction 때 permutation하지 않으면 A가 C의 generator를 받을 수 있다. metadata tensor와 generator list가 동일 mapping을 따르는지 검사한다.

TP에서는 어느 rank가 draw를 소유하는지 정한다. 모든 rank가 같은 global distribution과 generator state로 sample할 수도 있고, 한 rank가 sample해 global token을 broadcast할 수도 있다. local shard가 각자 draw해 후보를 내는 방식은 exact global categorical distribution을 자동으로 만들지 않는다. 분포 합성과 selection 알고리즘을 별도로 증명해야 한다.

floating difference도 재현성 범위를 제한한다. 같은 uniform 0.5000이 두 후보 경계 가까이에 있고 backend별 normalization 차이가 1e-6이면 token이 달라질 수 있다. RNG state가 같은지와 probability가 같은지를 함께 비교한다. “seed bug”라는 말로 numerical divergence를 숨기지 않는다.

**greedy request가 섞인 batch에서 draw를 세지 않는다**

temperature가 epsilon 아래인 request는 greedy 결과를 사용한다. 구현 최적화 때문에 random sampler kernel이 mixed batch 전체를 처리하더라도 greedy request가 소비한 draw가 다른 request generator를 밀어서는 안 된다. request별 generator면 영향이 격리되지만 global generator라면 dummy draw 정책까지 계약이 된다.

fixture는 A=random, B=greedy로 시작해 B를 제거하거나 위치를 바꾼다. A의 processed distribution과 selected token, generator counter가 같은지 본다. B가 finished row로 남아 padding sample을 수행하는 경우도 넣는다. 최종 text만 같으면 우연일 수 있으므로 uniform 또는 counter digest를 비교한다.

vLLM sampler metadata는 request별 generator collection을 top-k/top-p sampler에 넘긴다. `all_greedy`, `all_random`, mixed 조건에 따라 greedy 결과와 random 결과를 계산하고 request temperature로 최종 선택한다. source로 generator가 전달되는 사실을 확인하되 실제 reproducibility 범위는 backend sampler와 metadata lifecycle까지 읽어야 한다. [vLLM request generators와 mixed sampling](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L258-L304)

**RNG observability와 보안 경계**

production log에 seed와 full RNG state를 무조건 남기면 사용자 요청 재현이나 공격 표면이 될 수 있다. 일반 metric에는 generator ownership mode, draw count mismatch, deterministic cohort pass율을 둔다. 승인된 trace에는 seed의 keyed digest, request incarnation, logical step, branch, counter를 제한적으로 남긴다. raw random bytes와 사용자 prompt를 metric label에 넣지 않는다.

성능 metric은 generator 생성·lookup, sampling kernel, broadcast를 분리한다. request별 generator가 correctness를 높여도 객체 관리와 CPU synchronization이 병목이 될 수 있다. counter-based batched 구현이나 device-resident state로 개선할 수 있지만, compaction과 retry에서 identity를 잃지 않는 것이 먼저다.

RNG 수정의 terminal은 같은 request가 batch reorder, unrelated cancellation, TP topology에서 제품이 약속한 범위의 sequence를 유지하는 것이다. 제품이 cross-topology bitwise reproducibility를 약속하지 않는다면 그 한계를 문서화하고, 최소한 같은 effective backend·artifact·request schedule의 replay 범위를 정의한다.

## 18.4 sample 뒤 speculative accept와 commit을 분리한다

ordinary decoding에서는 sampled token이 곧 accepted token처럼 보인다. 그래도 processor history, grammar state, KV update, visible stream commit은 순서가 있다. speculative decoding에서는 draft가 여러 후보를 제안하고 target이 일부만 accept하므로 구분이 필수다.

draft `[a,b,c]`에서 target이 `[a,b]`만 accept했다고 하자. sampler가 c까지 임시 선택했어도 committed token history에는 a,b만 들어가야 한다. repetition count, grammar automaton, stop matcher, output text에 c를 남기면 rollback 뒤 상태가 오염된다. KV에도 rejected suffix가 있다면 뒤 cache owner가 crop/remove해야 한다.

acceptance RNG가 sampling RNG와 같은 stream을 쓰는지 별도 generator인지도 determinism에 영향을 준다. draft length가 바뀌면 소비 난수 수가 달라져 이후 ordinary sample까지 달라질 수 있다. “같은 seed인데 speculative on/off 결과가 다르다”는 현상이 반드시 bug는 아니지만, API가 어떤 재현성을 약속하는지 명시해야 한다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- Transformers assisted generation candidate/accept loop는 [Transformers v5.15.1 `candidate_generator.py:1-220`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/candidate_generator.py#L1-L220)와 generation loop [Transformers v5.15.1 `utils.py:3000-3350`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L3000-L3350)에서 candidate length, accepted count, model kwargs update를 잇는다.

vLLM speculative rejection sampler는 [vLLM v0.27.1 `rejection_sampler.py:1-220`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/rejection_sampler.py#L1-L220)에서 acceptance와 recovered token 생성 의미를 읽는다. acceptance 결과가 scheduler request state와 KV commit으로 넘어가는 경계는 32장 및 cache 장에 연결한다.

**ordinary decoding에서도 네 commit을 구별한다**

token X가 categorical sampler에서 선택되었다. 첫 commit은 request token history에 X를 append하는 것이다. 둘째는 penalty와 grammar 같은 generation state가 X를 accept하는 것이다. 셋째는 model/cache state가 다음 forward에서 X를 문맥으로 사용할 수 있게 되는 것이다. 넷째는 decoded text가 client에게 전송되는 것이다.

구현은 일부를 묶을 수 있지만 장애 분석에서는 분리한다. client cancellation이 selection 직후 오면 history와 KV를 유지할지 폐기할지 lifecycle 정책이 필요하다. stream에 이미 보낸 text는 되돌릴 수 없으므로 visible commit이 가장 강한 경계다.

stop token X는 model state에 accepted되더라도 visible text에서 숨길 수 있다. exclude 정책이라면 fourth commit만 하지 않는다. history와 finish reason에는 X가 반영될 수 있다. “출력에 없으니 생성되지 않았다”는 결론이 틀린 이유다.

### speculative acceptance를 확률로 손계산한다

draft가 token a를 확률 `q(a)=0.4`로 제안하고 target이 `p(a)=0.2`라면 rejection sampling의 한 형태에서 acceptance probability는 `min(1,p/q)=0.5`다. uniform `u=0.3`이면 accept, `u=0.8`이면 reject다. `p(a)=0.6`, `q(a)=0.4`라면 비율이 1보다 커 accept probability는 1이다.

reject 뒤에는 residual distribution에서 recovered token을 뽑을 수 있다. 단순히 target argmax를 고르는 것과 다르다. 정확한 알고리즘은 engine source를 따른다. draft와 target probability가 normalized인지, zero q를 어떻게 처리하는지, bonus token 조건이 무엇인지 본다.

여러 draft tokens의 acceptance는 앞에서 reject되면 뒤 후보가 canonical sequence에 들어가지 않는 prefix 과정이다. `[a,b,c]`에서 b가 reject되면 c를 독립적으로 accept할 수 있다고 가정하지 않는다. accepted length가 state crop의 기준이다.

**rollback 대상 장부**

rejected suffix가 건드릴 수 있는 state는 output IDs, penalty counts, grammar stacks, stop matcher buffer, detokenizer partial bytes, RNG offset, KV/cache length, usage counter다. 구현에 따라 draft state와 target state가 분리되어 일부는 canonical state를 건드리지 않을 수 있다. 각 state owner를 적는다.

RNG는 되돌리는지 소비를 유지하는지 계약이 필요하다. acceptance 판정에 이미 쓴 random draw를 rollback해 재사용하면 bias가 생길 수 있다. token history는 되돌리되 RNG counter는 전진하는 설계가 가능하다. “모든 state를 accepted length로 crop”이라는 표현은 지나치게 넓다.

usage accounting도 speculative compute tokens와 사용자 output tokens를 구별한다. rejected draft가 GPU work를 사용했지만 completion token usage에는 포함되지 않을 수 있다. metric 이름에 proposed, accepted, emitted 단위를 붙인다.

### 장애 사건: rejected token이 repetition history에 남는다

증상은 speculative mode에서만 특정 단어가 다음 step에서 과도하게 억제된다. raw target logits는 ordinary mode와 같지만 penalty 뒤 해당 token score가 낮다. 최초 divergence는 accepted output에는 없는 rejected draft ID의 count다.

source 분기는 acceptance result를 history update에 넘기는 slice다. proposed length를 쓰는지 accepted length를 쓰는지, bonus token을 포함하는지 본다. KV crop 상세가 아니라 sampler history owner에서 먼저 증명한다.

반증 fixture는 rejected token R의 다음 raw logit을 높게 고정하고 repetition penalty를 켠다. speculative off, all-accepted, R-rejected 세 경우의 history count와 processed R score를 비교한다. R-rejected는 off와 같아야 한다.

### 장애 사건: grammar state만 rollback되지 않는다

accepted text는 valid prefix인데 다음 step allowed set이 empty다. penalty history와 IDs는 올바르게 crop되었다. grammar stack digest가 proposed suffix 뒤 state를 가리키는 것이 최초 divergence다.

apply와 accept를 분리하고 canonical accept는 accepted prefix token에만 호출해야 한다. tentative grammar clone을 썼다면 reject 시 버린다. 검증은 accepted prefix를 처음부터 ordinary path로 feed한 grammar state와 speculative rollback 뒤 state가 같은지 비교한다.

### scheduler와 cache 장으로 넘길 정확한 계약

이 장이 32장에 넘기는 값은 request별 proposed length, accepted length, committed output IDs, finished/stop reason이다. scheduler는 이를 이용해 다음 실행과 slot lifecycle을 정한다. cache 장에는 accepted prefix에 대응하는 logical cache length와 rejected suffix 제거 요구를 넘긴다.

어느 block을 free하고 page table을 어떻게 바꾸는지는 여기서 반복하지 않는다. 다만 sampler가 accepted length를 잘못 보고하면 cache owner가 올바르게 구현되어도 틀린 suffix를 유지한다. 두 장 사이 invariant는 “다음 forward가 보는 token history와 cache logical length가 같은 accepted prefix를 표현한다”이다.

**speculative acceptance는 proposal이 아니라 commit 수를 결정한다.**

draft model이 token x를 확률 q(x)로 제안하고 target model이 p(x)를 준다고 하자. rejection sampling의 대표 규칙은 `min(1,p(x)/q(x))` 확률로 draft를 accept한다. q=0.5, p=0.3이면 acceptance probability는 0.6이다. uniform이 0.4면 accept, 0.8이면 reject한다. reject 뒤에는 target과 draft 차이의 residual distribution에서 recovered token을 뽑는 알고리즘이 이어질 수 있다.

draft token이 target argmax와 같다는 사실만으로 stochastic acceptance가 완료되지 않는다. 반대로 greedy verification에서는 연속 일치 길이로 accept할 수 있다. 옵션이 rejection sampling을 켜는지 greedy verify를 쓰는지에 따라 필요한 draft probability, temperature, RNG와 state가 다르다.

SGLang 고정 multi-layer EAGLE worker의 주석은 rejection mode에서 draft q를 verify로 운반하고 `coin*q < p` 조건과 residual resample을 사용한다고 명시한다. [SGLang speculative rejection 계약](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py#L130-L142)

**세 draft token의 acceptance 원장**

draft가 `[A,B,C]`를 제안하고 각 acceptance probability가 `[0.9,0.5,0.8]`, uniform이 `[0.2,0.7,...]`이라고 하자. A는 accept되고 B는 reject된다. C는 검증 tensor에 값이 있어도 logical prefix에는 도달하지 않는다. accepted draft 수는 1이다. reject 위치에서 recovered token R을 하나 sample하면 이번 verify가 commit하는 token은 `[A,R]` 두 개일 수 있다.

여기서 `accept_len`이라는 필드가 accepted draft만 세는지 bonus/recovered token까지 포함하는지 반드시 확인한다. SGLang EAGLE worker 고정 소스는 `accept_lens`가 bonus token을 포함하고 correct drafts는 `accept_lens-1`이라고 주석으로 구분한다. metric 이름만 보고 acceptance rate 분모를 만들면 off-by-one이 생긴다. [SGLang accepted drafts와 bonus 구분](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/speculative/eagle_worker_v2.py#L855-L880)

history, grammar, stop, KV는 `[A,R]`만 commit해야 한다. proposal `[A,B,C]` 전체를 repetition count에 넣었다가 B,C를 제거하지 않으면 다음 distribution이 오염된다. grammar FSM을 A,B,C까지 advance한 뒤 accept prefix 길이만큼 되돌리는 구현은 rollback snapshot이 정확해야 한다. 더 안전한 의미 모델은 candidate state와 committed state를 분리하고 accepted prefix를 publish하는 것이다.

**reproducibility는 acceptance coin과 recovered draw를 분리한다**

ordinary decoding step 하나가 draw 하나를 쓰던 기준선과 speculative step은 draw 수가 다르다. acceptance coin을 categorical generator와 같은 sequential stream에서 소비하면 draft 길이나 reject 위치가 바뀔 때 이후 output draw가 밀린다. 알고리즘 자체가 같은 output distribution을 보장해도 bitwise sequence는 다를 수 있다. 제품이 어떤 reproducibility를 약속하는지 명시한다.

branch-addressed counter를 쓰면 accept coin `(step,draft_lane,accept)`과 recovered sample `(step,reject_lane,residual)`을 분리할 수 있다. 하지만 reference 구현과 같은 random consumption을 약속한다면 오히려 순차 소비를 맞춰야 할 수 있다. 목표는 무조건 같은 방식이 아니라 명시된 계약과 trace 가능한 ownership이다.

TP/DP에서는 draft q와 target p가 같은 global token 좌표인지 확인한다. draft tokenizer나 vocab revision이 다르면 token ID만으로 ratio를 계산할 수 없다. padding token과 grammar-masked support도 두 distribution에서 일관되게 정규화되어야 한다. q(x)=0인데 draft가 x를 제안하는 모순, p support가 empty인 경우를 명시적으로 실패시킨다.

**stop과 constraint가 acceptance 경계에서 충돌한 사건**

JSON grammar가 `}` 뒤 EOS만 허용하고 stop string도 `}`라고 하자. draft가 `}`, EOS, 추가 공백을 제안했고 target verify가 첫 두 token을 accept했다. engine이 proposal 단계에서 stop을 감지해 request를 finished로 표시했지만 acceptance 결과는 나중에 첫 token만 commit하도록 바뀌었다. grammar state는 EOS까지 advance했고 visible buffer는 `}`를 stop marker로 숨겼다. 다음 cleanup에서 committed token, grammar state, visible text, finish reason이 서로 다른 frontier를 가졌다.

증상은 JSON body가 빈 문자열로 끝나거나 다음 요청이 같은 slot에서 empty-support를 만나는 것이다. first divergence는 raw logits가 아니라 proposal-side stop mutation이다. stop은 accepted token commit 뒤 visible text policy로 평가해야 하고, speculative candidate가 final state를 직접 mutate한다면 generation snapshot으로 rollback돼야 한다.

debug record에는 proposed IDs, per-lane p/q, uniform, accepted draft count, recovered/bonus ID, committed IDs, grammar before/after, stop withheld bytes를 둔다. `accept_len`의 정의를 명시한다. 이 record가 있으면 “speculative가 가끔 깨짐”을 어느 frontier가 앞서 갔는지로 바꿀 수 있다.

수정 뒤 ordinary decoding과 speculative decoding에 같은 target distribution과 fixed RNG coordinate를 주고 committed prefix parity를 본다. performance는 proposed/verified/accepted/committed token과 rollback copy byte로 나눈다. acceptance rate만 좋아지고 grammar snapshot cost가 폭증하면 latency 이득이 없을 수 있다.

## 18.5 stop holdback 뒤에만 visible byte를 commit한다

EOS token은 model vocabulary의 ID다. stop token list는 EOS 외 IDs를 포함할 수 있다. stop string은 decoded 문자 sequence이며 token 경계를 가로지를 수 있다. max length는 token 회계 guard다. 네 조건은 서로 대체되지 않는다.

stop string `ABC`가 token pieces `A`, `B`, `C`로 올 수도 있고 `AB`, `C`로 올 수도 있다. matcher는 누적 decoded suffix를 보아야 한다. 현재 `AB`를 client에 보냈다가 다음 `C`에서 exclude-stop 정책을 알게 되면 되돌릴 수 없다. 가능한 stop prefix를 보류하고 확정된 text만 commit해야 한다.

Transformers `StopStringCriteria`의 token-overlap preparation과 tensor match는 [Transformers v5.15.1 `stopping_criteria.py:110-320`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/stopping_criteria.py#L110-L320), max length와 EOS 계열 기준은 [같은 파일 `:40-110`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/stopping_criteria.py#L40-L110)에서 구분한다.

UTF-8 한 문자가 여러 token byte 조각에 걸치면 stop matcher와 streamer는 incomplete suffix를 보존해야 한다. replacement character를 먼저 commit하면 나중 byte가 와도 수정할 수 없다. finish reason, 마지막 content chunk, usage event의 순서도 API 계약이다. terminal event 뒤 content를 보내거나 cancellation 뒤 token을 commit해서는 안 된다.

max length는 prompt 포함인지 new tokens만인지 분리한다. `max_new_tokens`가 effective total length로 변환되는 owner와 model context limit, scheduler admission limit은 서로 다를 수 있다. length stop이 발생해도 EOS를 실제 sequence에 append했는지, finish reason만 length인지 구분한다.

**EOS와 stop token의 집합 관계**

model config에는 EOS ID 하나 또는 여러 EOG IDs가 있을 수 있다. request stop token list가 이를 포함하거나 별도 application delimiter ID를 추가할 수 있다. sampled ID가 집합에 들어갔을 때 finish reason이 `stop`인지 model EOS인지 API마다 표현이 다를 수 있다.

`ignore_eos` field는 EOS score를 mask하는지, EOS가 선택되어도 종료 판단만 무시하는지 구분한다. 전자는 후보 분포를 바꾸고 후자는 token history에 EOS가 들어간 뒤 generation을 계속할 수 있다. field→branch→state→effect를 source에서 닫아야 한다.

`min_new_tokens`가 있으면 최소 길이 전 EOS score를 suppress하는 processor가 들어갈 수 있다. 이는 stopping criteria가 EOS를 보고도 무시하는 방식과 순서가 다르다. processed logits의 EOS score를 관측하면 어느 구현인지 알 수 있다.

### stop string automaton을 손으로 걷는다

stop strings가 `END`, `END!` 두 개라고 하자. 현재 decoded suffix가 `EN`이면 둘 모두의 prefix라 보류한다. 다음 `D`가 오면 짧은 stop은 완성되지만 긴 stop도 계속 가능하다. 정책이 earliest match인지 longest match인지, stop 목록 순서가 우선인지 정의해야 한다.

include-stop=false이면 `END` bytes를 visible output에 내지 않는다. 이미 그 앞 text 중 stop prefix 가능성이 없는 부분만 commit한다. include=true라도 partial UTF-8이 완성된 뒤 text event를 만들어야 한다. stop matcher와 detokenizer의 buffer ownership을 연결한다.

token pieces가 `"E"`, `"ND"`이든 `"EN"`, `"D"`이든 같은 stop을 찾아야 한다. token-level suffix list만 비교하면 segmentation에 의존한다. 반대로 stop token ID는 decode 문자열이 같아도 다른 ID에는 반응하지 않는다.

**max length의 세 숫자**

prompt length 100, `max_new_tokens=20`이면 effective total cap은 120일 수 있다. `max_length=110`도 함께 주어지면 어느 field가 우선하는지 config normalization이 정한다. model context cap 128과 scheduler admission cap 116이 있으면 실제로는 더 작은 guard가 먼저 작동할 수 있다.

finish reason이 length라고 해서 model의 position table 끝에 도달했다는 뜻은 아니다. generation budget이 먼저 끝났을 수 있다. scheduler reject와 generation loop stop도 다르다. ledger에 prompt tokens, accepted new tokens, effective generation cap, model context cap, engine cap, 최초 firing guard를 둔다.

padding된 batch tensor width를 row별 generated length로 오인하지 않는다. finished rows가 있어도 common width는 늘 수 있다. request state의 semantic length가 stopping owner다.

### incremental UTF-8 commit 손계산

한글 `가`의 UTF-8 bytes `EA B0 80`이 세 token piece에 나뉘었다고 하자. 첫 byte `EA`는 3-byte sequence 시작이며 두 continuation이 필요하다. `EA B0`도 아직 incomplete다. `EA B0 80`에서 비로소 `가`를 commit한다.

첫 step에서 replacement character를 client에 보내면 뒤 bytes가 와도 append-only stream에서 고칠 수 없다. incremental decoder는 incomplete와 invalid를 구분한다. sequence 종료 시 incomplete suffix를 error, replacement, raw escape 중 어떻게 처리할지도 API 정책이다.

stop string이 `가`라면 complete code point가 되기 전에는 match할 수 없다. bytes가 완성된 순간 stop으로 판정해 exclude 정책이면 `가`를 보내지 않는다. detokenizer가 먼저 visible commit하고 stop matcher가 뒤에서 검사하면 누출된다.

### terminal event의 정확한 순서

마지막 content suffix, finish reason, usage, end-of-stream marker가 있다. protocol이 정한 순서를 지켜야 한다. usage가 accepted token count를 반영하기 전에 terminal을 보내거나 terminal 뒤 content를 보내면 client state machine이 깨진다.

client disconnect/cancellation과 model stop이 경쟁할 수도 있다. 어떤 reason이 승리하는지, 이미 queued content를 flush하는지 정책이 필요하다. scheduler lifecycle은 32장이 소유하지만 visible event commit은 이 장의 경계다.

stream metric은 sampled, accepted, decoded-complete, emitted token/byte를 분리한다. token 하나가 문자 하나가 아니므로 emitted token 수라는 표현도 정확한 정의가 필요하다. usage는 일반적으로 token 회계이고 network bytes와 다르다.

### 장애 사건: stop prefix가 한 chunk 먼저 노출된다

증상은 최종 assembled response에서는 stop이 제거되지만 streaming client 화면에 `</to`가 잠시 보인다. 최초 divergence는 stop matcher가 보류해야 할 suffix를 emitter가 먼저 commit한 것이다. offline decode와 final postprocessing은 정상이다.

source 분기는 detokenize suffix→stop match→emit 호출 순서다. 가능한 stop prefix 길이를 계산하고 withheld buffer가 emitter 앞에 있는지 본다. 반증은 stop을 여러 token segmentation으로 만드는 fixture와 chunk boundary를 바꾸는 fixture다.

수정 뒤 각 event concatenation에 stop 문자열이 없어야 하고 non-stop common prefix는 불필요하게 오래 지연되지 않아야 한다. correctness와 latency를 함께 관측하되 stop 최대 길이만큼의 보류 가능성을 문서화한다.

## 18.6 하나의 token commit 기록으로 장애·검산·배포를 닫는다

**세 실패를 같은 token commit 행에서 찾는다.**

첫 사례는 temperature 0에서 NaN이다. raw logits가 finite인지 확인하고 temperature branch 전후를 본다. 실제 division이 실행되었다면 field validation/greedy 분기가 owner다. sampler 이후 NaN만 보지 말고 processor별 finite count를 남긴다.

둘째는 grammar 사용 시 간헐적 empty support다. grammar mask 전 valid count, top-k/top-p 뒤 count, EOS 허용, automaton state를 step별로 기록한다. mask 전부터 후보가 없으면 이전 processor numeric 문제고, mask에서 0이 되면 grammar/prefix contradiction이며, top-k 뒤 0이면 순서 문제다.

셋째는 batch reorder에서 seed 재현이 깨진다. request별 generator identity, batch row, consumed draw count, sampled token을 비교한다. raw/processed logits가 같고 uniform draw가 다르면 RNG ownership이다. draw까지 같고 token이 다르면 probability normalization, tie 또는 distributed selection을 본다.

넷째는 speculative rejection 뒤 금지 문자열이 다시 나타난다. accepted length, rejected suffix, penalty count, grammar stack, stop matcher buffer, visible committed text를 전후로 비교한다. rejected token 흔적이 어느 state에 남았는지 최초 divergence를 찾는다. KV rollback 구현은 뒤 장으로 넘기되 accepted prefix 길이라는 계약을 제공한다.

다섯째는 stop 문자열 일부가 client에 노출된다. per-step token IDs, decoded complete bytes, withheld stop-prefix buffer, emitted suffix, terminal event를 기록한다. offline full decode가 맞아도 incremental commit이 틀릴 수 있다. include/exclude stop option이 buffer branch와 emitted text를 어떻게 바꾸는지 본다.

**workbook 공통 장부**

각 step에 request ID, raw logits top slice와 finite count, processor 이름/순서, transform 뒤 support count, grammar state digest, generator state digest, sampled ID, accepted length, committed IDs, decoded pending bytes, withheld stop suffix, emitted text, finish reason을 둔다. 전체 vocab tensor를 production에 남기라는 뜻은 아니다. 격리 fixture에서 bounded slice와 support digest를 쓴다.

첫 divergence 앞 단계가 같다는 negative evidence가 중요하다. processed logits가 같으면 penalty와 grammar 이전을 기각한다. sampled ID가 같으면 RNG를 기각한다. accepted IDs가 같으면 speculative acceptance를 기각한다. decoded bytes가 같고 emitted text만 다르면 streamer다.

**실습 A — processor order를 뒤집는다.**

raw logits `[5,4,3]`, grammar valid `{2}`, top-k=2를 oracle로 쓴다. pipeline A는 grammar→top-k, B는 top-k→grammar다. 각 단계 score와 support를 손으로 적고 empty support 발생 위치를 확인한다.

경쟁 가설은 top-k inclusive bug, grammar mask ID mapping, processor order다. grammar 직후 mask가 맞고 top-k 단독 결과가 맞는데 결합에서만 empty면 order가 증거다. 수정 뒤 valid token 2가 support에 남고 invalid tokens는 어떤 bias로도 부활하지 않아야 한다.

**실습 B — repetition sign을 깨뜨린다.**

history IDs `{0,2}`, raw `[3,-2,-0.5,1]`, penalty 1.5를 사용한다. expected `[2,-2,-0.75,1]`을 reference로 둔다. buggy all-divide는 `[2,-2,-0.333,1]`이 된다.

최종 sampled token은 두 결과에서 우연히 같을 수 있으므로 processed score를 직접 assert한다. duplicate history를 추가해 set semantics도 검사한다. penalty 1.0에서는 identity 및 processor bypass를 확인한다.

**실습 C — RNG row ownership을 흔든다.**

request A/B에 서로 다른 seed와 분포를 주고 batch 순서를 바꾼다. B를 첫 step 뒤 종료시켜 compaction도 만든다. request generator state와 draw가 identity를 따라가는지 본다.

processed probability가 같고 draw만 다르면 RNG mapping이다. draw가 같고 token만 다르면 categorical boundary/tie다. TP에서만 다르면 rank owner와 broadcast를 본다. 이 세 분기를 한 “seed bug”로 묶지 않는다.

**실습 D — grammar empty support를 원인별로 만든다.**

첫 fixture는 invalid prefix, 둘째는 valid grammar token을 top-k가 제거, 셋째는 모든 raw logits NaN이다. 세 경우 모두 최종 no-token처럼 보이지만 최초 divergence가 다르다.

error type과 metric도 다르게 기대한다. invalid prefix는 constraint error, order conflict는 pipeline bug 또는 명시 정책, NaN은 numeric error다. EOS fallback 하나로 셋을 덮으면 회귀 검사가 실패해야 한다.

**실습 E — speculative rollback 장부.**

draft `[a,b,c]`, accepted length 2, c가 repetition과 grammar에 영향을 주도록 설계한다. rollback 뒤 canonical IDs, counts, grammar stack, stop buffer를 ordinary `[a,b]` path와 비교한다.

RNG state는 동일해야 한다고 임의로 assert하지 않는다. 알고리즘 계약상 소비된 acceptance draw를 유지할 수 있다. 대신 다음 sample을 재현하려면 expected counter advance를 문서화한다. KV logical length는 accepted prefix와 같아야 하며 구체 page 검증은 cache 장으로 넘긴다.

**실습 F — stop과 stream terminal.**

stop strings `END`, `END!`, UTF-8 stop 하나를 준비하고 token segmentation을 여러 방식으로 나눈다. step마다 pending bytes, decoded suffix, withheld prefix, emitted chunk를 적는다. include/exclude 정책을 각각 검증한다.

마지막에는 content concatenation, finish reason, usage, terminal ordering을 확인한다. stop token과 stop string이 동시에 같은 step에 발생하면 precedence와 reason을 source 계약에 맞춘다. max length도 동시에 도달하는 tie fixture를 둔다.

이 workbook의 종료 조건은 문제 요청이 한 번 성공하는 것이 아니다. 최초 divergence를 만든 함수 분기, 그 분기가 읽은 field/state, 사용자 증상으로 이어진 consumer, 같은 축의 regression fixture가 연결되어야 한다. 연결이 없으면 아직 추측이다.

**통합 incident: JSON 응답이 가끔 깨지고 stop marker가 한 번 보인다**

증상은 복합적이다. 동일 schema와 seed를 쓰는데 동시 요청이 많을 때만 JSON 마지막 괄호가 빠지고, streaming client에는 드물게 `</tool>` 일부가 보인다. 팀은 처음에 “GPU sampler가 비결정적”이라고 추측한다. 그러나 이 문장은 서로 다른 두 failure를 한 원인으로 묶었을 수 있다.

첫 단계는 request 두 개 A와 B를 고정한다. A는 schema가 있고 top-p 0.9, B는 grammar 없이 top-p 0.7이다. B가 먼저 종료되어 batch compaction이 일어나는 fixture와 단독 A fixture의 step ledger를 비교한다. model revision, exact input IDs, raw logits bounded slice가 같은지 확인한다.

raw logits는 같다. 따라서 17장과 model forward 가설은 잠정 기각한다. temperature와 penalty 뒤 score도 같다. grammar mask 뒤 A의 allowed count가 단독에서는 14, compaction 뒤에는 31이고 grammar identity digest가 B의 default state와 같다. 최초 divergence는 RNG가 아니라 request-row grammar mapping이다.

source는 batch filter/reorder 분기로 좁혀진다. sampling parameter tensors는 살아 있는 row index로 재배치되지만 grammar state list 또는 bitmask row가 같은 permutation을 쓰는지 본다. A가 B의 unconstrained mask를 받으면 invalid closing token이 support에 들어온다. seed가 같아도 distribution이 다르므로 다른 token을 고르는 것은 결과이지 원인이 아니다.

수정 뒤에는 `[A,B]`, `[B,A]`, B 조기 종료, A 조기 종료 네 경우에서 request ID별 grammar digest와 allowed support가 같아야 한다. processed probabilities가 같아진 뒤에야 RNG state와 sampled token을 비교한다. 이 순서를 거꾸로 하면 seed 조작으로 증상을 잠시 숨길 수 있다.

그런데 grammar row 수정 뒤 JSON은 맞아졌지만 `</tool>` 일부 노출은 남는다. 이는 첫 원인으로 모든 증상을 설명하려던 가설이 틀렸다는 negative evidence다. sampled/accepted token IDs와 offline final text는 정상이고, emitted chunk concatenation에만 stop prefix가 있다. 두 번째 최초 divergence는 visible commit이다.

step trace를 보면 decoded suffix가 `</to`일 때 matcher는 possible stop prefix로 표시했지만 emitter가 withheld buffer를 제외하지 않고 chunk를 보냈다. 다음 token `ol>`에서 stop 완성을 판정하고 final assembled response에서는 후처리로 제거한다. 이미 network에 보낸 prefix는 되돌릴 수 없다.

source 분기는 detokenizer output, stop matcher result, streamer enqueue의 호출 순서다. include-stop=false일 때 possible prefix를 pending buffer에 남기고 확정되지 않은 suffix를 emitter에 넘기지 않아야 한다. 수정은 final string replace를 강화하는 것이 아니라 commit 경계를 앞으로 옮기는 것이다.

반증 fixture는 `</tool>`을 여러 token segmentation으로 만들고 `</tools>`처럼 긴 공통 prefix지만 stop이 아닌 text도 넣는다. stop은 어떤 segmentation에서도 노출되지 않아야 하고 non-stop text는 mismatch가 확정된 즉시 방출되어야 한다. 무조건 stop 최대 길이 전체를 기다려 correctness는 얻되 latency를 불필요하게 늘리는 수정도 피한다.

이 incident에서 두 fix의 회귀 단위는 다르다. grammar fix는 request reorder 후 allowed support identity이고, stream fix는 pending/emitted byte boundary다. 최종 JSON 하나만 검사하면 둘 중 하나가 다시 깨져도 원인을 찾기 어렵다. 증상→최초 divergence→owner 분기→직접 state 검증의 사슬을 각각 보존한다.

**통합 incident의 speculative 변형**

같은 schema request에서 speculative decoding을 켰을 때만 다음 step grammar가 empty가 된다고 하자. batch row mapping은 올바르고 draft `[",", "x"]` 가운데 첫 token만 accepted되었다. canonical output IDs는 accepted prefix와 맞지만 grammar stack은 두 token을 모두 accept한 상태다.

최초 divergence는 grammar state commit length다. draft candidates를 검사하기 위한 tentative state를 canonical state와 공유했거나, proposed length로 accept loop를 돌렸을 수 있다. target acceptance algorithm의 수학과 grammar mask 자체는 정상이다.

ordinary path에 accepted token 하나만 feed한 grammar stack을 reference로 삼는다. speculative rollback 뒤 stack digest, partial UTF-8 buffer, next allowed set이 reference와 같아야 한다. penalty history와 stop buffer도 함께 비교하지만 첫 divergence가 grammar라면 각각 독립 assertion으로 둔다.

KV cache가 proposed suffix를 제거했는지는 뒤 cache 장이 검증한다. 이 장은 accepted length와 committed IDs를 정확히 넘겼는지 확인한다. scheduler는 finished state와 next token budget을 소비한다. handoff invariant가 명확하면 어느 장의 owner가 잘못했는지 분리할 수 있다.

**장말 독자 경로: 관측값 하나로 다음 파일을 고른다**

raw logits가 reference와 다르면 processor 파일을 열지 않는다. logits 선택 위치와 model forward를 17장으로 되돌린다. raw는 같고 penalty 직후 다르면 history owner와 sign transform을 본다. penalty는 같고 candidate support가 다르면 ordered warper 또는 grammar bitmask다.

support와 normalized probability가 같고 draw가 다르면 generator mapping과 counter를 본다. draw까지 같고 selected ID가 다르면 categorical boundary, tie, distributed ID mapping이다. selected ID는 같고 accepted prefix가 다르면 speculative acceptance다. accepted prefix는 같고 다음 processor state가 다르면 rollback/commit owner다.

accepted IDs와 decoded complete bytes가 같고 pending suffix만 다르면 incremental decoder 또는 stop matcher다. pending도 같고 emitted event만 다르면 streamer ordering이다. content events가 같고 finish reason·usage·terminal만 다르면 API output state machine으로 간다.

이 경로는 긴 checklist가 아니라 이분 탐색 좌표다. 모든 state를 항상 production log에 남길 필요는 없다. bounded metric으로 이상 구간을 찾고 승인된 최소 fixture에서 해당 경계의 상세 state를 수집한다. schema, prompt, generated text 같은 민감 정보는 redaction과 synthetic reproduction을 사용한다.

## 18.7 고정 source·배포 검산을 같은 token commit 기록에 붙인다

소스를 열기 전에 token 하나의 순서를 다시 고정한다. `raw logits → ordered transforms → allowed support → RNG selection → acceptance → state commit → stop 판단 → visible commit → terminal event`다. 모든 항목을 매번 검사하지 않고, 첫으로 다른 checkpoint에서만 아래 분기를 연다.

processed logits까지 같으면 penalty·warper 가설을 버리고 RNG로 간다. sampled ID가 같은데 visible output이 다르면 acceptance·stop·stream owner를 본다. terminal event까지 같으면 generation 의미론은 닫히며, API 전달이나 scheduler resource lifetime을 다음 owner로 넘긴다. 이 stop rule 때문에 이어지는 명령 목록은 전수 검사가 아니라 incident branch가 된다.

### 18.7.1 회수할 순서와 반증 기준

조사의 주어는 설정 이름이 아니라 현재 checkpoint의 값이다. temperature, top-p, grammar, stop string은 각각 field·branch·state mutation·effect·falsifier로 남긴다. branch와 state가 관측되지 않으면 option이 실제 경로를 바꿘다고 쓰지 않는다.

### 18.7.2 고정 source에서 각 branch의 owner를 찾는다

이 장의 구현 관찰점은 Transformers v5.15.1 commit
`550d7b3834670483a4df436541272c055dc364bf`, vLLM v0.27.1 commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang v0.5.18 commit
`71de97b264b04dcd514cf904003028aefe9775c8`, llama.cpp v0.2.0 commit
`bb4caa7540188872173c44d161602d9271386413`에 고정했다. 다음 좌표와 검산은 source를 정적으로
읽은 결과이며 모델이나 서버를 실행한 관측은 아니다.

Transformers에서는 processor 구성, processor call, multinomial/argmax, stopping criteria, streamer 순으로 읽는다. vLLM과 SGLang에서는 request별 sampling tensor와 batch reorder, GPU sampler, structured-output mask, engine output commit을 잇는다. llama.cpp에서는 sampler chain과 grammar apply/accept, token piece decode를 분리한다.

## 18.8 네 구현에서 같은 token step의 owner를 찾는다

### 18.8.1 Transformers 호출 경로를 한 step으로 잇는다

generation config는 processor와 stopping criteria를 만드는 입력이다. config validation 뒤 processor list를 구성하고 매 step model score에 ordered call을 적용한다. `LogitsProcessorList.__call__`이 custom processor signature와 순서를 보존하는 경계는 [Transformers v5.15.1 `logits_process.py:70-110`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/logits_process.py#L70-L110)에서 읽는다.

sampling loop에서는 processed scores를 softmax/multinomial 또는 argmax에 보내고 unfinished sequence mask와 next token을 갱신한다. 정확한 branch는 [Transformers v5.15.1 `utils.py:2800-3000`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2800-L3000)에서 해당 revision의 loop를 확인한다. method 이름이 바뀌어도 score→selection→append→criteria 순서를 찾는다.

streamer가 있으면 token IDs를 전달하는 시점과 stopping criteria 판정 시점을 비교한다. stop string criterion이 true여도 이미 streamer에 token이 전달되었다면 streamer 자체 withholding 또는 higher-layer filtering이 필요할 수 있다. high-level API만 보고 stop이 visible text에서 자동 제외된다고 가정하지 않는다.

### 18.8.2 vLLM의 request row와 sampler를 잇는다

vLLM에서는 scheduler/runner가 여러 request sampling metadata를 batch tensor로 만든다. sampler는 logits와 temperature/top-k/top-p/min-p, penalties, generators를 소비한다. batch reorder나 finished request 제거가 있으면 모든 per-request tensor가 같은 identity mapping을 유지해야 한다.

GPU sampler output이 곧 client event는 아니다. engine output processor가 request token state를 갱신하고 stop 조건, detokenization, logprob, finish reason을 처리한다. Sampled ID parity와 visible response parity를 별도 checkpoint로 둔다.

structured-output bitmask가 sampler 전에 logits에 적용되는 실제 caller를 따라간다. tree 디렉터리만 링크하는 것으로 끝내지 않고 compile state, request grammar, batch bitmask, GPU application, accepted-token update owner를 잇는다. backend가 비동기로 grammar mask를 준비할 때 readiness와 fallback 정책도 state다.

### 18.8.3 SGLang의 sampling batch mutation을 잇는다

SGLang `SamplingBatchInfo`는 request별 옵션을 tensor화하고 merge/filter/reorder lifecycle을 가질 수 있다. class constructor만 읽지 않고 batch가 축소·확장될 때 temperature/top-p/min-p와 grammar state가 어떤 index로 이동하는지 읽는다.

sampler layer는 logits normalization과 selection을 맡지만 penalty가 다른 module에서 앞서 적용되거나 custom kernel에 fused될 수 있다. option field가 어느 object에서 tensor가 되고 어떤 kernel/function 인자로 들어가는지 역추적한다. Python source에 수식이 없다고 기능이 없는 것이 아니다.

output side에서는 token ID를 request state에 append하고 stop token/string과 detokenizer를 처리하는 owner를 찾는다. grammar accept가 sampling mask 생성과 같은 object인지 별도 manager인지도 확인한다. request cancellation 뒤 queued output이 commit되지 않는 lifecycle은 scheduler 장과 연결한다.

### 18.8.4 llama.cpp sampler chain의 apply와 accept를 잇는다

llama.cpp는 sampler chain 순서를 명시적으로 구성할 수 있다. 각 sampler `apply`가 candidate array를 변형하고 dist sampler가 ID를 고른 뒤 `accept`가 stateful sampler history를 갱신한다. chain order가 metadata/default/user config 중 어디서 정해지는지 본다.

candidate array에는 ID, logit, probability와 sorted flag 같은 state가 있을 수 있다. 한 sampler가 logits를 바꾼 뒤 sorted flag를 무효화하지 않으면 다음 top-p가 stale ordering을 사용할 위험이 있다. API struct와 primitive가 이를 어떻게 관리하는지 source로 확인한다.

grammar는 generic sampler chain과 결합될 수 있지만 apply와 accept의 의미가 같다. apply는 현재 stacks에서 invalid candidate를 차단하고, accept는 선택 token piece로 stacks를 전진한다. stop string은 grammar 완결과 별도의 higher-level text state다.

### 18.8.5 옵션을 읽는 다섯 칸

temperature를 예로 들면 field는 request config의 숫자다. branch는 greedy 전환 또는 temperature sampler 추가다. state mutation은 logits scale 또는 sampling mode/RNG 사용 여부다. 효과 후보는 entropy와 후보 상대 확률 변화다. 반증 관측은 processor list, 전후 logits, RNG draw count다.

top-p field는 cumulative cutoff branch를 켜고 sorted support mask를 바꾸며 후보 수와 selection work에 영향을 준다. 반증은 sorted probability와 cutoff index다. repetition penalty는 history-enabled branch와 token score mutation을 만들며 반증은 history set/count와 sign별 전후 score다.

grammar field는 compile backend와 per-step allowed mask state를 만들고 structured validity와 compile/mask 비용에 영향을 준다. 반증은 grammar identity, automaton state, allowed count, masked score다. stop string field는 matcher와 withheld suffix buffer를 만들고 visible latency/finish에 영향을 주며 반증은 decoded suffix, pending prefix, emitted chunk다.

이 다섯 칸 가운데 branch와 state가 확인되지 않으면 option 효과를 실행 사실처럼 쓰지 않는다. config가 파싱되었지만 backend가 지원하지 않아 무시되거나 fallback할 수 있다. warning, selected implementation, actual state를 함께 확인한다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- llama.cpp sampler chain의 apply/accept lifecycle은 [llama.cpp v0.2.0 `llama-sampler.cpp:665-920`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-sampler.cpp#L665-L920), top-k/top-p/min-p primitives는 [같은 파일 `:1440-1815`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-sampler.cpp#L1440-L1815), penalties는 [같은 파일 `:2856-3240`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-sampler.cpp#L2856-L3240)에서 고정한다.

SGLang sampler의 top-k/top-p/min-p entry는 [SGLang v0.5.18 `sampler.py:45-190`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/sampler.py#L45-L190), result와 logprob 처리 경계는 [같은 파일 `:190-360`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/sampler.py#L190-L360)에서 읽는다. 실제 path가 revision에서 다른 wrapper를 선택하는지는 caller를 따라 확인한다.

이 장의 순서는 `raw logits → ordered transforms → allowed support → RNG selection → acceptance → state commit → stop 판단 → visible commit → terminal event`다. 어느 option도 이 전체를 혼자 소유하지 않는다. processed logits까지 같으면 penalty/warper 가설을 버리고 RNG로 간다. sampled token이 같고 output이 다르면 accept/stop/stream으로 간다. terminal event까지 같으면 generation 의미론은 닫힌다.

### 18.8.6 고정 소스 좌표

- [Transformers processor implementations](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/logits_process.py#L280-L980)
- [Transformers processor construction](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L1260-L1515)
- [Transformers stopping criteria](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/stopping_criteria.py#L40-L320)
- [Transformers candidate generator](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/candidate_generator.py#L1-L220)
- [vLLM sampler](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L90-L310)
- [vLLM rejection sampler](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/rejection_sampler.py#L1-L220)
- [SGLang sampling batch state](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/sampling/sampling_batch_info.py#L60-L390)
- [SGLang sampler](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/sampler.py#L45-L360)
- [llama.cpp sampling primitives](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-sampler.cpp#L1440-L1815)
- [llama.cpp grammar apply와 accept](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-grammar.cpp#L1353-L1453)

### 18.8.7 소스 좌표를 검산하는 방법

링크가 존재한다는 사실만으로 설명이 맞지 않는다. 각 좌표에 대해 symbol, input state, mutation, return/consumer를 네 칸으로 적는다. Transformers temperature processor라면 score tensor를 입력받아 temperature로 나눈 score를 반환하고 뒤 processor 또는 selection이 소비한다. stopping criterion은 score를 바꾸지 않고 row별 stop boolean을 만든다. 두 class가 generation 아래 있다는 이유로 같은 역할이라고 쓰지 않는다.

processor constructor validation과 `__call__` runtime을 분리한다. temperature 음수나 0을 constructor가 거부하는지, greedy 전환이 processor 생성 이전 higher-level branch에서 일어나는지 정확히 찾는다. line range가 class 전체를 포함해도 실제 조건문 위치를 확인한다. 이 구분이 field→branch 설명의 근거다.

vLLM sampler에서는 forward entry의 인자에서 logits shape와 sampling metadata를 적고, penalties, temperature, top-k/top-p, random sampling, logprob 반환 순서를 call graph로 만든다. helper 이름만 나란히 쓰지 않고 실제 caller 순서를 확인한다. fused/custom op가 선택되면 Python reference와 actual branch를 구분한다.

structured-output 디렉터리는 compiler, request state, GPU bitmask가 나뉠 수 있다. schema compile 성공이 per-step mask 적용을 증명하지 않는다. grammar backend identity, bitmask update, sampler consumer, accepted token state update까지 네 개 owner를 연결한다. 하나가 비동기라면 readiness와 error propagation도 본다.

SGLang에서는 `SamplingBatchInfo` field가 생성되는 constructor와 request filter/merge 함수를 함께 읽는다. static config 설명만으로 mixed-batch correctness를 알 수 없다. request row가 변할 때 temperature tensor, penalty state, generator, grammar가 모두 같은 mapping을 쓰는지 각 mutation을 확인한다.

llama.cpp에서는 sampler chain의 순서와 primitive 구현을 나눈다. top-k 함수가 정확해도 chain construction에서 top-p보다 앞인지 뒤인지에 따라 semantics가 달라진다. `apply` 뒤 candidate array의 sorted flag와 probability materialization state가 다음 sampler 기대와 맞는지 본다. stateful sampler의 `accept`가 언제 호출되는지도 caller까지 따라간다.

stop 처리 좌표는 token sampling source와 text streaming source가 다를 수 있다. sampler에서 EOG를 감지하는 코드만 읽고 arbitrary stop string이 구현되었다고 결론 내리지 않는다. token ID stop, decoded-string matcher, incremental UTF-8, network event emitter의 파일과 owner를 각각 찾는다.

speculative 좌표는 candidate proposal, target verification, acceptance/recovered token, request state commit, cache rollback으로 나눈다. 이 장은 앞 네 개 중 generation state 의미를 설명하고 cache data structure는 뒤 장에 넘긴다. source range 하나가 전 과정을 소유한다고 가정하지 않는다.

## 18.9 옵션을 설정 카드와 token commit 기록으로 검산한다

### 18.9.1 설정 카드 네 개를 끝까지 검산한다

`top_p=0.9` field는 top-p enabled validation과 sampler/processor 추가 branch를 탄다. state는 sorted cumulative probability mask이고 효과는 request별 support 크기와 selection work의 변화다. 반증은 raw probability, cutoff index, allowed ID digest다. 응답 다양성 하나만 보면 temperature나 RNG 변화와 분리되지 않는다.

`seed=42` field는 generator 생성 또는 request generator lookup branch를 탄다. state는 generator key와 counter이며 효과는 약속된 범위의 재현성이다. 반증은 batch reorder 전후 request별 state digest와 draw다. 최종 string만 같아도 중간 draw가 우연히 다른 token 확률 구간에 함께 들어갔을 수 있다.

`response_format=json_schema` field는 schema compile/backend selection branch를 타고 grammar automaton과 allowed bitmask state를 만든다. 효과는 support 제한과 compile/mask 비용이다. 반증은 grammar identity, prefix state, allowed count, invalid token score다. parse 성공률만으로 generation-time constraint 적용을 증명하지 않는다.

`stop=["END"]` field는 decoded suffix matcher와 include/exclude/finish branch를 만든다. state는 pending UTF-8와 possible-prefix buffer이며 효과는 visible output과 tail latency다. 반증은 token segmentation별 decoded bytes, withheld suffix, emitted chunks, terminal reason이다.

### 18.9.2 이 장을 읽은 뒤 작성할 한 문장

좋은 장애 요약은 “top-p가 이상하다”가 아니다. “동일 raw logits와 penalty 결과에서 batch compaction 뒤 grammar bitmask row가 다른 request state를 가리켜 allowed support가 최초로 달랐고, 그 분포에서 sampled invalid token이 JSON 종료를 깨뜨렸다”라고 쓴다. state와 consumer가 연결되어 있다.

RNG 문제라면 “processed probability는 같았으나 request generator 대신 batch generator가 finished row draw까지 소비해 compaction 다음 counter가 최초로 달랐다”라고 쓴다. stop 문제라면 “accepted IDs와 offline decode는 같았으나 possible stop prefix가 emitter 전에 보류되지 않아 visible commit에서 최초로 달랐다”라고 쓴다.

이 문장은 수정 위치를 자동으로 정하지는 않지만 잘못된 층을 제외한다. 첫 사례는 grammar-row lifecycle, 둘째는 RNG ownership, 셋째는 stream commit owner다. 각각의 회귀 fixture도 같은 state를 직접 검증할 수 있다.

### 18.9.3 다음 심층 fixture 전에 확인하는 token 경계

raw logits는 아직 분포도 선택도 아니다. ordered transforms는 score와 support를 만들지만 난수를 선택하지 않는다. RNG sample은 후보를 제안하지만 speculative target이 reject할 수 있다. accepted token은 generation state에 commit되지만 stop 정책 때문에 사용자에게 보이지 않을 수 있다. decoded text도 possible stop prefix이거나 incomplete UTF-8이면 아직 emit할 수 없다.

terminal event가 나간 뒤에야 그 request의 visible generation이 끝난다. KV와 scheduler resource가 언제 해제되는지는 별도 lifecycle이므로 terminal과 동일 시각이라고 가정하지 않는다. 이 장이 제공하는 것은 scheduler와 cache 장이 소비할 accepted prefix, finished reason, visible commit의 정확한 의미다.

source를 읽을 때도 이 순서를 거꾸로 섞지 않는다. output bug에서 시작해 first divergence를 위로 올리고, raw logits까지 같음을 확인한 다음 다시 ordered state transition을 내려온다. 이 왕복이 완료되어야 원인 설명과 사용자 증상이 연결된다.

운영 metric도 같은 경계를 따른다. processor별 전체 vocab tensor를 저장하지 않더라도 nonfinite count, support size, grammar allowed count, speculative proposed/accepted count, stop-withheld byte와 finish reason을 bounded metric으로 둘 수 있다. request trace가 필요할 때는 sampling configuration digest와 step index로 승인된 sample을 연결한다. prompt나 schema 원문을 고-cardinality label로 쓰지 않는다.

성능을 해석할 때는 의미를 보존한 기준선과 비교한다. top-p를 끄고 빨라졌다는 사실은 sort/select 비용 가능성을 보이지만 출력 support도 바뀌었으므로 동등 workload가 아니다. reference implementation과 optimized kernel에 동일 processed logits, mask, RNG input을 주어 selected ID와 state transition을 differential하는 방식이 correctness 검증에 적합하다. end-to-end latency는 별도 workload 계약에서 측정한다.

grammar backend를 바꿀 때도 compile 성공률만 보지 않는다. 동일 prefix corpus에서 allowed ID set을 비교하고, partial UTF-8과 token piece 경계, EOG acceptance, empty support error를 포함한다. accepted token을 feed한 다음 state까지 같아야 한 step parity가 아니라 state-machine parity가 된다.

speculative mode의 성능 이득도 acceptance rate 하나로 설명하지 않는다. proposed tokens, target verification work, accepted tokens, recovered sample, rollback state cost, output commit을 나눈다. acceptance rate가 높아도 grammar나 stop state를 매 draft마다 복사하는 비용이 클 수 있다. 실제 비율은 측정 사항이며 소스 감사에서는 state copy와 synchronization 가능 경로까지만 확정한다.

마지막 release gate는 option 조합을 포함한다. temperature와 top-p, penalty와 prompt inclusion, grammar와 top-k, speculative와 grammar, stop string과 streaming을 각각 교차한다. 모든 조합을 무한히 시험할 수는 없으므로 이 장의 비가환성·state ownership 경계를 자극하는 최소 fixture를 선택한다. 새 processor가 추가되면 list의 어느 위치에 들어가는지와 mask/finite invariant를 다시 검토한다.

이렇게 하면 option 문서는 기본값 표에서 끝나지 않는다. field가 branch를 선택하고 state를 바꾸며 다음 consumer와 비용을 변화시키고, 어떤 관측이 그 설명을 반박할 수 있는지까지 제공한다. 독자는 결과가 이상할 때 옵션을 무작위로 조절하지 않고 최초로 달라진 상태의 owner 함수를 연다.

수정 승인도 동일하다. processed support를 고친 patch는 sampled text 한 건이 아니라 cutoff·tie·empty-support fixture를 통과해야 한다. RNG patch는 request reorder와 finished-row compaction에서 generator identity를 보존해야 한다. rollback patch는 ordinary accepted-prefix state와 동치여야 한다. streamer patch는 여러 token segmentation에서 stop을 숨기면서 non-stop suffix를 즉시 방출해야 한다. 증상과 같은 상태 축을 검증해야 우연한 성공을 배제할 수 있다.

어느 fixture도 모든 실제 workload를 증명하지는 않는다. 그래서 고정 source revision, configuration digest, negative evidence, 알려진 미검증 backend를 함께 기록한다. 정확성은 자신감 있는 문장이 아니라 반증 가능한 경계들의 연결이다.

그 연결이 유지되면 엔진 구현이 바뀌어도 같은 의미 좌표로 회귀를 찾을 수 있다. 함수 이름보다 입력, mutation, acceptance, visible commit의 계약이 오래 남는다.

다음 조사 위치는 first divergence가 정한다. raw logits가 다르면 17장 또는 앞 forward로 돌아간다. transform 뒤가 다르면 ordered processor와 history다. allowed set부터 다르면 grammar다. uniform draw부터 다르면 RNG ownership이다. accepted prefix부터 다르면 speculative state이고, visible text만 다르면 stop/decode/stream commit이다. 이 분리를 지키면 “seed가 안 맞는다”나 “stop이 가끔 샌다”를 재현 가능한 상태 전이 문제로 바꿀 수 있다.

그리고 같은 canonical token 장부로 수정 뒤의 의미 보존까지 다시 검증한다.

### 18.9.4 배포와 독자 검토도 같은 token commit 기록을 재사용한다

배포 전 첫 표는 ordered stages다. raw score, bias/penalty, temperature, grammar allowed set, min-p/top-k/top-p/typical selector, renormalization, RNG draw, speculative acceptance, committed IDs, stop/visible commit 순서를 적는다. engine의 실제 순서가 다르면 문서가 아니라 source와 API 계약에 맞춰 표를 고친다. 중요한 것은 하나의 확정된 순서와 반례 fixture다.

둘째 표는 mutable state owner다. penalty count, grammar FSM, RNG counter, speculative candidate generation, accepted prefix length, stop matcher, UTF-8 pending buffer가 어느 request incarnation과 generation에 속하는지 적는다. batch physical row와 state identity를 동일시하지 않는다. filter, merge, compaction, cancel, slot reuse 함수에서 모든 state가 같은 permutation과 lifetime을 따르는지 본다.

셋째 표는 비용이다. vocabulary V와 batch B에 대해 dense mask byte, selector temporary와 sort/select work, grammar compile/cache, RNG state, speculative q tensor, snapshot/rollback byte, stop withheld buffer를 계산한다. option coefficient 자체보다 켜지는 representation과 kernel이 비용을 만든다. grammar schema 크기는 compile 비용과 반드시 선형이 아니며 실제 backend 측정으로 보완한다.

넷째 표는 incident signature다. support가 처음 다르면 processor/grammar, distribution은 같고 uniform이 다르면 RNG, proposal은 같고 accepted prefix가 다르면 acceptance, committed IDs는 같고 visible text가 다르면 stop/stream이다. 각 row에 첫 source function과 반증 관측을 붙인다.

다섯째 표는 terminal이다. boundary probability와 tie fixture, bitmask word·vocab padding fixture, request reorder RNG fixture, speculative reject/bonus off-by-one fixture, stop string token segmentation과 UTF-8 fixture가 모두 통과해야 한다. mixed batch에서 서로 다른 option을 가진 request가 state를 교환하지 않아야 한다.

운영 metric은 support size, grammar allowed count, empty-support error, RNG ownership mismatch, proposed/accepted/committed token, stop-withheld byte와 finish reason을 둔다. schema 원문, seed, 후보 token 전체를 high-cardinality label로 두지 않는다. sampled trace에 configuration digest와 request-step identity를 연결한다.

rollback은 selector fast path, grammar backend, speculative mode, streamer 변경을 각각 독립적으로 끌 수 있어야 한다. 하나의 global kill switch만 있으면 원인을 좁히기 어렵지만 fallback 조합이 검증되지 않으면 더 위험하다. rollout 전에 fallback path가 같은 stage 의미와 state migration을 보존하는지 시험한다.

이 장의 최종 invariant는 다음과 같다. **동일한 processed score와 request state에서 정의된 ordered constraint가 유효한 global support를 만들고, request 소유 RNG와 acceptance 규칙이 하나의 accepted prefix를 결정하며, 그 prefix만 history·grammar·cache에 commit되고 stop 정책을 지난 bytes만 사용자에게 보인다.**

이 문장을 작은 probability 반례, request reorder, speculative reject, stop segmentation 네 fixture로 증명할 수 있으면 sampling 옵션은 더 이상 마법의 knob가 아니다. 독자는 어떤 값과 state가 바뀌는지, 왜 결과가 바뀌는지, 어느 함수에서 첫 divergence를 찾을지 설명할 수 있다.

## 18.10 45분 동안 옵션 필드에서 visible byte까지 추적한다

첫 5분에는 API field의 validation과 effective default를 찾는다. `top_k=-1`이나 0이 disabled를 뜻하는지, `top_p=1`과 `min_p=0`이 identity인지, temperature 0이 greedy로 정규화되는지 적는다. grammar option은 schema 종류와 backend 선택, stop은 include/exclude 정책을 적는다. payload에 없던 default가 processor를 켜면 이후 비교가 모두 어긋난다.

다음 5분에는 batch metadata producer를 읽는다. scalar option이 `[B]` tensor로 바뀌는 곳, logit bias가 `[B,V]` 또는 sparse representation이 되는 곳, grammar bitmask row와 generator가 request에 붙는 곳을 찾는다. filter와 merge 함수도 같이 읽는다. constructor만 보면 최초 batch는 맞아도 continuous batch mutation을 놓친다.

세 번째 5분에는 processor ordering을 찾는다. 등록 list의 순서와 sampler forward의 호출 순서를 구분한다. processor가 score를 mutate하는지 support mask만 만드는지, argmax-invariant로 분류되는지, top-k/p와 어느 쪽에 있는지 적는다. fused kernel이면 Python reference와 runtime selected backend를 나눈다.

네 번째 5분에는 grammar state를 찾는다. schema compile result, tokenizer vocabulary mapping, per-request FSM, step별 bitmask producer, sampler consumer, accepted token feed를 연결한다. compile future가 준비되지 않았을 때 scheduler가 기다리는지 fallback하는지 error를 내는지 본다. speculative overlap이면 candidate와 committed grammar frontier를 어떻게 나누는지 찾는다.

다섯 번째 5분에는 RNG를 찾는다. generator가 request 생성 때 만들어지는지 batch마다 lookup되는지, seed와 counter가 어디 저장되는지, sampler backend에 어떻게 전달되는지 적는다. greedy row, finished row, padded row가 draw를 소비하는지 확인한다. TP rank 중 누가 sample하고 token을 어떻게 broadcast하는지 찾는다.

여섯 번째 5분에는 speculative verify를 읽는다. draft q가 보존되는지, target p가 어떤 processed stage인지, accept length가 draft만인지 bonus를 포함하는지, recovered token이 어디서 sample되는지 적는다. output history와 grammar가 업데이트되는 시점이 verify 결과 publish 뒤인지 확인한다.

일곱 번째 5분에는 stop과 stream을 읽는다. EOS token, max token, arbitrary stop string, parser completion이 어느 owner에서 finish reason을 만든는지 적는다. decoded suffix matcher가 possible prefix를 얼마나 보류하는지, incomplete UTF-8을 어떻게 다루는지, include_stop 옵션이 visible buffer를 어떻게 바꾸는지 본다.

마지막 10분에는 한 request-step을 세로로 그린다. effective option, raw score digest, after-processor support, uniform coordinate, proposed ID, accepted prefix, committed grammar/history, decoded pending bytes, emitted delta, terminal event를 한 줄씩 연결한다. 각 edge에 source anchor와 runtime 반증 관측을 하나 둔다. 이 산출물이 있으면 새 backend의 함수 이름이 달라도 같은 의미 흐름으로 비교할 수 있다.

## 18.11 경계 fixture로 정확성과 비용을 함께 반증한다

### 18.11.1 열두 가지 failure injection으로 설명의 빈틈을 찾는다

첫째, top-p cutoff 누적값을 정확히 0.9로 만든다. cutoff token 포함 여부와 FP32 rounding을 본다. 둘째, top-k 경계 두 token score를 같게 만든다. global ID tie-break와 TP topology parity를 본다. 셋째, grammar가 top-k 상위 후보를 모두 금지하도록 만든다. grammar-first 계약과 empty intersection 처리를 본다.

넷째, 실제 vocabulary가 bitmask word보다 하나 크게 되게 한다. 마지막 valid token과 padding bit를 본다. 다섯째, added vocabulary token만 grammar가 허용하게 한다. tokenizer/compiler/head identity를 본다. 여섯째, grammar allowed set을 완전히 비운다. NaN이 아니라 명시적 terminal 또는 정의된 fallback이 나오는지 본다.

일곱째, random A 사이에 greedy B를 넣었다 뺀다. A의 request generator counter가 같은지 본다. 여덟째, A와 C batch row를 교환한다. generator와 grammar state가 request identity를 따라가는지 본다. 아홉째, 한 TP rank의 probability에 cutoff 근처 작은 perturbation을 넣는다. numerical parity 약속과 selected owner를 본다.

열째, speculative chain의 두 번째 draft를 reject한다. 세 번째 draft가 history와 grammar에 들어가지 않는지 본다. 열한째, accept length가 bonus를 포함하는 path에서 metric과 commit slice의 off-by-one을 본다. 열두째, stop string을 서로 다른 token segmentation과 UTF-8 byte split으로 보낸다. marker가 노출되지 않고 non-stop suffix는 불필요하게 지연되지 않는지 본다.

각 injection은 기대하는 최초 divergence를 먼저 적는다. top-p fixture는 cutoff/support, generator fixture는 uniform coordinate, speculative fixture는 accepted prefix, stop fixture는 visible commit이다. 예상보다 앞 단계가 갈리면 실험 setup이나 competing bug가 있다. 예상보다 뒤에서만 증상이 보이면 관측 checkpoint가 부족하다.

### 18.11.2 옵션 조합을 pairwise가 아니라 비가환 경계로 고른다

모든 option Cartesian product는 감당할 수 없다. 조합을 수학적 비가환성과 state 공유로 선택한다. bias×repetition은 sign branch를 자극한다. temperature×top-p는 mass와 cutoff를 자극한다. grammar×top-k는 support order를 자극한다. penalty×speculative는 candidate/commit history를 자극한다. grammar×speculative는 FSM rollback을 자극한다. stop×streaming은 visible frontier를 자극한다.

top-p×min-p는 둘 다 support를 줄이지만 기준이 다르므로 순서를 확인한다. 먼저 min-p로 tail을 없애고 재정규화한 뒤 top-p를 적용하는 것과, top-p prefix 뒤 min-p를 적용하는 것은 후보가 다를 수 있다. typical×temperature는 entropy와 surprisal 자체가 바뀐다. “둘 다 다양성 옵션”이라는 분류는 test selection에 충분하지 않다.

RNG×batching은 state identity 조합이고 RNG×speculative는 draw branch 조합이다. seed 값 여러 개를 무작위로 늘리는 것보다 한 seed에서 reorder, cancel, reject 위치를 바꾸는 fixture가 ownership bug를 잘 드러낸다. numerical backend 차이는 cutoff와 cumulative boundary에 가까운 분포를 별도 corpus로 만든다.

### 18.11.3 계산량과 saved work를 실제 shape로 연결한다

dense grammar bitmask가 token마다 V bits라면 V=152,064에서 row당 19,008 byte, B=64에서 약 1.16 MiB다. int32 word tensor로 보아도 같은 bit payload에 alignment와 metadata가 더해진다. 매 step CPU에서 새 mask를 만들고 H2D copy한다면 compile time과 별도로 transfer와 synchronization이 critical path가 된다. GPU FSM이나 overlap은 이 비용을 숨기거나 줄일 수 있지만 state readiness 계약이 생긴다.

full sort는 V log V 성격이지만 top-k selection과 fused top-p는 다른 work shape를 가질 수 있다. 실제 kernel complexity와 temporary는 backend를 읽고 측정한다. k=1 greedy는 max reduction으로 충분하고, k=50과 flat top-p는 더 많은 후보 work를 요구한다. option 값이 낮다고 항상 빠른 것은 아니다. top-p가 매우 낮아도 cutoff를 알기 위해 score ordering이 필요할 수 있다.

speculative q tensor는 `B×draft_steps×V`가 될 수 있다. B=32, draft steps=5, V=152,064, FP32면 약 92.8 MiB다. rejection sampling을 위해 full q를 보존하는지, top-k 제한 representation을 쓰는지에 따라 memory가 크게 달라진다. accepted token 수만 보고 이 persistent/temporary byte를 놓치면 OOM 원인을 설명하지 못한다.

grammar snapshot을 draft step마다 복사한다면 state 크기와 steps에 비례한다. immutable state node나 replay로 줄일 수 있지만 rollback compute가 늘 수 있다. stop matcher도 모든 draft text를 decode했다가 버리는지 accepted prefix만 처리하는지에 따라 CPU work가 다르다. 의미 경계를 먼저 정하고 saved work를 계산한다.

## 18.12 incident review와 배포 판정을 같은 인과선에 놓는다

### 18.12.1 incident review를 사람이 읽을 수 있는 서사로 쓴다

좋은 review는 request가 무엇을 원했는지에서 시작한다. “temperature 0.7, top-p 0.9, JSON grammar, stop `END`, speculative rejection을 사용한 request A가 mixed batch compaction 뒤 invalid JSON을 반환했다”고 쓴다. 그다음 기대 stage 순서와 실제 first divergence를 적는다.

수치 표에는 grammar 전 상위 후보, allowed set, cutoff, uniform, proposed chain, p/q와 accept coin, committed prefix를 넣는다. 로그 수십 줄을 그대로 붙이지 않는다. 핵심 source 인용은 state producer와 잘못된 consumer edge 주변만 쓴다. 나머지는 고정 링크로 제공한다.

원인 문장은 “race condition”으로 끝내지 않는다. “generation 7의 grammar mask H2D가 완료되기 전에 재사용 slot의 generation 8 sampler가 같은 physical row를 읽어, request B allowed set이 A에 적용됐다”처럼 세대, resource, happens-before를 적는다. 수정은 slot generation 검증과 event wait인지, buffer 분리인지 명확히 한다.

검증 문장은 증상과 같은 state 축을 닫는다. mixed batch order 100개를 무작위로 돌렸다는 말보다, 늦은 generation mask가 있어도 consumer가 mismatch를 거부하고 올바른 mask readiness 뒤 sampling하며 accepted token만 FSM을 advance한다고 쓴다. soak와 fuzz는 이 invariant를 다양한 schedule에서 자극한다.

### 18.12.2 최종 독자 체크리스트

독자는 먼저 “지금 배열은 score인가 probability인가, global인가 local인가”를 묻는다. 다음으로 “어떤 processor가 이미 적용됐고 support는 무엇인가”를 묻는다. 세 번째로 “RNG draw는 어느 request-step-branch 소유인가”를 묻는다. 네 번째로 “proposal 중 몇 token이 accepted와 committed가 됐는가”를 묻는다. 마지막으로 “committed token 중 어느 bytes가 stop matcher를 지나 visible해졌는가”를 묻는다.

이 다섯 질문에 답하지 못하면 option 값을 바꾸기 전에 관측을 보강한다. 답할 수 있으면 첫 divergence stage의 producer와 consumer를 연다. raw score가 맞으면 model layer를 다시 보지 않고, support가 맞으면 grammar를 다시 compile하지 않고, accepted prefix가 맞으면 speculative verifier를 다시 의심하지 않는다. negative evidence가 조사 범위를 줄인다.

배포 승인 문서는 지원 backend와 미검증 조합도 적는다. reference와 fast path가 parity를 보인 GPU·dtype·vocab 범위, grammar backend, speculative mode를 명시한다. 새 CUDA/library release에서 tie와 cutoff가 달라질 수 있으므로 boundary fixture를 재실행한다. 실행하지 않은 조합을 “지원”으로 확대하지 않는다.

최종 terminal에서는 correctness, lifecycle, performance, observability가 동시에 닫힌다. correctness는 support·draw·accept·visible bytes가 reference와 맞다. lifecycle은 cancel·compact·reject·reuse에서 generation state가 섞이지 않는다. performance는 target workload에서 mask, selection, q, snapshot 비용이 예산 안이다. observability는 전체 민감 데이터를 저장하지 않고도 첫 divergence를 복원할 수 있다.

이 네 terminal을 지나야 빠른 sampler나 speculative path를 기본값으로 승격한다. 실패하면 검증된 ordinary/reference path로 되돌리고 option 의미와 finish reason을 보존한다. rollback 자체도 request in flight에서 state를 섞지 않도록 drain 또는 generation fencing 전략을 가진다.

이제 한 token의 여정이 닫힌다. logits는 ordered policy를 지나 support와 mass가 되고, request-owned draw가 proposal을 만들며, verifier가 accepted prefix를 정하고, 오직 그 prefix만 mutable generation state에 반영된다. decoder와 stop matcher가 visible bytes를 확정한 뒤 terminal event가 나간다. 이 순서를 설명하고 손으로 계산하고 source와 trace로 반증할 수 있다면 독자는 실제 장애에서 다음에 어디를 파야 할지 안다.

## 18.13 fallback 전환과 종합 문제로 장을 마무리한다

### 18.13.1 마지막 모의 장애: 빠른 path는 맞았지만 fallback 전환이 틀렸다

grammar fast path가 top-p와 speculative rejection 조합을 지원하지 않아 reference fallback으로 전환된다고 하자. scheduler는 sampler backend만 바꿨지만 fast path에서 이미 grammar FSM을 draft 두 칸 advance했고 RNG acceptance coin 하나를 소비했다. fallback은 같은 request의 committed frontier가 아니라 변경된 candidate frontier에서 시작했다. 선택 token은 우연히 문법에 맞지만 seed replay와 stop 위치가 달라졌다.

이 사건은 각 backend 내부 parity 테스트만으로 잡히지 않는다. 전환 edge가 state를 넘기는 계약이 없기 때문이다. debug record에는 `from_backend`, `to_backend`, transition reason, committed generation, candidate generation, grammar state digest, RNG counter, accepted length를 둔다. fallback은 committed snapshot에서 재시작하거나 candidate state를 완전하게 전달해야 한다. 일부 field만 복사하는 것은 안전한 migration이 아니다.

수치 fixture는 fast path가 A를 propose하고 accept coin 0.7을 소비한 뒤 unsupported grammar를 발견하도록 만든다. reference-only 실행과 fallback 실행에서 processor support, 첫 uniform address, committed token을 비교한다. candidate A가 reject되었다면 penalty count, grammar, stop matcher 어디에도 A가 남지 않아야 한다. fallback latency는 correctness와 별도로 기록한다.

배포 중 backend를 feature flag로 바꿀 때도 동일하다. in-flight request는 기존 backend로 drain하거나 generation-fenced snapshot을 새 backend가 읽어야 한다. 새 요청만 전환하는 정책이 단순하지만 shared grammar cache와 generator pool revision이 섞이지 않는지 본다. rollback 버튼의 존재가 rollback 의미의 정확성을 보장하지 않는다.

### 18.13.2 책을 덮기 전 독자가 스스로 답할 문제

확률 `[0.5,0.2,0.15,0.1,0.05]`, grammar allowed `{1,2,3,4}`, top-k=2, top-p=0.8을 준다. grammar를 먼저 적용해 재정규화하면 `[0.4,0.3,0.2,0.1]`이고 top-k는 token 1,2를 남긴다. 그 둘의 재정규화 확률은 약 `[0.5714,0.4286]`이며 top-p cutoff 정책에 따라 둘째 token이 포함된다. top-k를 grammar 전에 적용하면 원래 top 2는 token 0,1이고 교집합은 token 1 하나다.

uniform 0.6이면 첫 계약에서는 token 2, 두 번째 계약에서는 token 1이다. seed가 같아도 processor order가 달라 결과가 달라진다. 여기에 draft가 token 2를 q=0.6으로 제안하고 target processed p=0.4286이라면 acceptance probability는 약 0.7143이다. accept coin 0.8이면 reject되고 residual sample이 필요하다. uniform draw와 acceptance coin을 같은 숫자로 재사용하면 안 된다.

stop string이 token 2의 decoded piece 끝에 걸리면 accepted 뒤 visible commit 정책이 적용된다. reject된 proposal의 bytes는 stop matcher에 최종 commit돼서는 안 된다. 이 한 문제는 support order, RNG branch, acceptance, visible frontier를 모두 연결한다. 답이 달라질 때 어느 단계의 계약을 바꿨는지 말할 수 있어야 한다.

운영자는 이 fixture를 source revision마다 자동화하고 결과와 backend identity를 보존한다. 저자는 본문 설명을 그 fixture의 변수와 연결한다. 독자는 표를 손으로 다시 계산한 뒤 실제 trace에서 같은 필드를 찾는다. 이 세 역할이 같은 의미 좌표를 공유할 때 문서는 단순 소개가 아니라 유지 가능한 디버깅 도구가 된다.

최종 승인 회의에서는 평균 품질 인상 대신 네 개의 반증 질문을 던진다. cutoff tie에서 support가 안정적인가. unrelated request의 종료가 draw 주소를 바꾸는가. reject된 draft가 어떤 mutable state에도 남는가. stop prefix가 visible commit 전에 노출되는가. 각 질문에는 실패 fixture, source owner, metric, rollback이 하나씩 대응해야 한다.

답을 모르면 배포를 미루고 관측을 추가한다. 답이 “아니오”라면 그 증거의 revision과 configuration 범위를 적는다. backend나 CUDA library가 바뀌면 boundary fixture를 재실행한다. 이 습관이 옵션 표를 살아 있는 실행 계약으로 만든다.

책의 다음 판에서도 이 표는 그대로 복사하지 않는다. 새 selector와 grammar backend, sampler kernel이 어느 stage를 대체했는지 diff하고 같은 반례를 다시 계산한다. 값이 달라졌다면 의도된 계약 변경인지 regression인지 release note와 source, fixture로 판정한다. 설명의 최신성은 버전 숫자가 아니라 이 재검증에서 나온다.

그 결과와 미검증 범위까지 독자가 다시 사용할 수 있는 형태로 남긴다.

그리고 동일 fixture로 rollback 경로까지 다시 검증한다.

이제 내부의 visible commit 기록을 외부 응답 계약으로 넘긴다. 어느 token이 확정됐고 stop 경계에서 무엇이 숨겨졌는지를 보존한 채, 19장은 그것이 non-stream 응답과 stream delta, finish reason, disconnect 처리에서 정확히 한 번 전달되는지 묻는다.
