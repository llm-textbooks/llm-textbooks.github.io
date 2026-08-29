# 1장. 다음 토큰을 맞힌다는 것

다음 토큰 학습을 안다고 말하려면 `softmax` 식 하나를 외우는 것으로는 부족하다. 화면의 문자열이 어느 token ID가 되었는지, 어느 위치의 logit row가 그 ID를 정답으로 삼는지, mask를 통과한 손실이 어떤 분모로 평균되는지까지 한 사건으로 이어져야 한다. 이 사슬이 끊기면 loss는 정상 범위에 있어도 엉뚱한 위치나 표본을 학습할 수 있다.

이 장에서는 원문 byte에서 token·logit·loss·gradient 직전까지를 한 줄로 추적한다. 여기서 고정한 token·logit·loss 분모는 2장의 gradient 소유권으로 넘어간다. 5장은 token ID를 byte·template까지 역추적하고, 24장은 같은 per-token loss가 실제 일반화를 측정하는지 다시 묻는다.

## 1.1 byte를 token ID와 embedding 좌표로 바꾼다

학습의 첫 입력은 화면에 보이는 문장이 아니라 revision이 고정된 byte와 변환 pipeline이다. normalization·template·tokenizer가 만든 ID를 embedding row까지 추적해야 같은 문자열처럼 보이는 두 run이 실제로 같은 함수를 계산하는지 판정할 수 있다.

### raw byte에서 token ID와 offset을 함께 보존한다

모델은 문자를 읽지 않는다. `DocumentID`가 가리키는 UTF-8 바이트열을 고정된 `TokenizerRevision`으로 분절한 뒤, 각 조각을 정수 ID로 바꾸어 읽는다. ID `[B,T]`는 embedding 표 `E∈R^{V×C}`의 행 주소다. lookup 결과 `X=E[id]`의 shape는 `[B,T,C]`이며, 같은 ID가 배치 안에 세 번 나오면 그 행의 gradient에는 세 경로의 기여가 더해진다. 이 때문에 토크나이저 변경은 단순 전처리 변경이 아니다. 좌표축 `V`와 특정 행이 뜻하는 조각을 함께 바꾼다.

golden 사례는 `GoldenBatchID`가 입력 ID, 다음-token label, loss mask의 checksum을 묶는다. 문자열만 저장하면 normalization이나 template 변경을 잡지 못하고, ID만 저장하면 어느 byte 구간에서 왔는지 되짚지 못한다. 둘을 offset map으로 함께 보존한다.

**왜 다음 토큰인가.** 길이가 다른 모든 문장을 고정된 class label로 만들기는 어렵다. autoregressive factorization은 문장 확률을 `p(x_1,…,x_T)=Π_t p(x_t|x_<t)`로 쪼갠다. 하나의 sequence에서 T개의 감독 신호를 얻고, 생성할 때도 같은 조건부 분포를 한 step씩 사용한다. 이 분해는 언어가 반드시 왼쪽에서 오른쪽으로 이해된다는 철학적 주장이 아니라 joint distribution을 계산 가능한 조건부 곱으로 표현하는 선택이다.

teacher forcing에서는 학습 중 이전 위치에 model sample이 아니라 정답 token을 넣는다. 병렬로 모든 위치의 logits를 계산할 수 있지만 생성 때에는 자기 출력이 다음 입력이 된다. 이 train/generation 차이 때문에 낮은 token loss가 장기 생성의 모든 오류를 설명하지는 않는다.

| handoff | tensor | dtype | 의미 | 첫 불변식 |
|---|---|---|---|---|
| tokenizer output | IDs `[B,T]` | integer | embedding 행 주소 | `0≤id<V` |
| token embedding | `[B,T,C]` | float | 학습된 좌표 | row lookup 일치 |
| hidden state | `[B,T,C]` | float | context 표현 | causal prefix 불변 |
| LM logits | `[B,T,V]` | float | 후보 상대 점수 | vocab 축 일치 |
| labels | `[B,T]` | integer | 다음 token/ignore | shift 계약 |

이 표를 읽는 가장 안전한 방법은 shape를 외우는 것이 아니라 **한 target의 계보를 끝까지 끊지 않는 것**이다. raw byte 구간 `r[12:15]`가 `TokenizerRevision=tok-r7`에서 ID 91이 되었다고 하자. 이 ID는 `input_ids[0,2]`에 놓여 `E[91]`을 읽고, 여러 block을 지난 `hidden[0,2,:]`는 LM head의 모든 vocabulary 행과 내적되어 `logits[0,2,:]`를 만든다.

그러나 이 logit row의 정답은 ID 91 자체가 아니다. causal shift 계약에 따라 원 sequence의 다음 ID인 `labels[0,3]` 또는 미리 이동된 `shift_labels[0,2]`다. 그 위치가 loss mask에서 유효할 때에만 NLL이 numerator에 더해지고 valid-target count가 denominator를 1 늘린다.

backward에서는 먼저 vocabulary 전 좌표의 `p-one_hot(y)`가 생기고, LM head를 거쳐 hidden과 output rows로, 다시 transformer와 input embedding row로 흐른다. 같은 raw span을 찾을 수 없는 gradient는 설명할 수 없는 gradient다.

### tokenizer에서 embedding lookup까지 owner를 고정한다

구현마다 symbol 이름은 달라도 상태 전이는 다음 순서를 보존해야 한다. `tokenizer(...)`는 text를 `input_ids`와 가능하면 offset map으로 바꾼다. model `forward`는 embedding lookup과 blocks를 거쳐 hidden을 만들고 LM head projection으로 logits를 만든다. causal-loss wrapper 또는 collator 중 정확히 하나가 shift를 소유한다. cross-entropy 진입점은 log-sum-exp와 정답 logit의 차이를 만들고 reduction layer는 numerator와 denominator를 결합한다. `backward()`는 이미 정규화된 scalar의 vector-Jacobian product를 parameter graph에 흘린다.

| 경계 | 대표 symbol | 입력 shape | 출력·상태 변화 | 최초 비교값 |
|---|---|---|---|---|
| tokenization | tokenizer call | raw bytes/text | IDs `[B,T]`, offsets | IDs·offset checksum |
| lookup | embedding `forward` | `[B,T]` | `[B,T,H]` | 선택 row와 alias |
| contextualization | model blocks | `[B,T,H]` | `[B,T,H]` | block-boundary RMS |
| projection | LM head | `[B,T,H]` | logits `[B,T,V]` | 선택 row·logsumexp |
| target alignment | collator 또는 causal loss | labels `[B,T]` | shifted labels·valid bitmap | position→target 표 |
| unreduced CE | cross entropy | `[B,T,V]`, `[B,T]` | NLL `[B,T]` | FP64 hand fixture |
| reduction | loss/trainer | NLL·bitmap | `S/N` | numerator `S`, count `N` |
| reverse pass | autograd | scalar | parameter `.grad` | `dlogits`, head·embedding grad |

여기서 `softmax` tensor가 실제로 materialize된다고 단정하지 않는다. fused cross-entropy는 확률 배열을 저장하지 않고도 같은 log-sum-exp와 gradient를 계산할 수 있다. 표의 “대표 symbol”은 의미 경계이고, 실제 kernel 경계는 backend·dtype·compile mode에 따라 달라진다. 그러므로 Python 함수 일치, tensor 의미 일치, kernel bitwise 일치를 서로 다른 주장으로 기록한다.

nanoGPT 고정 commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`의 `model.py:170-183`은 `idx [b,t]`와 position `[t]`를 token/position embedding으로 바꾸고 block을 통과시킨다. `184-193`은 target이 있을 때 모든 위치의 logits와 CE를 계산하고, 없을 때 마지막 위치만 LM head에 넣는다. training과 inference branch의 output shape가 다른 점은 호출자가 이해해야 할 API 계약이다.

**Embedding gradient의 두 얼굴.** lookup만 보면 등장한 ID 행에 gradient가 scatter-add된다. tied LM head에서는 같은 표가 모든 vocabulary logit을 만드는 dense projection에도 쓰여 등장하지 않은 행도 경쟁 확률을 통해 gradient를 받을 수 있다. “embedding gradient는 sparse하다”는 말은 모듈 단독과 전체 tied model을 구분해야 한다.

**반례 1—문자열이 같은데 IDs가 다르다.** Unicode normalization, leading space, BOS 삽입, template가 달라질 수 있다. model 입력의 실체는 화면 문자열이 아니라 tokenizer revision과 ID sequence다.

**반례 2—IDs가 같은데 함수가 다르다.** embedding row와 LM-head checkpoint가 다른 revision이면 같은 주소가 다른 vector를 읽는다. IDs checksum만으로 model parity를 보장하지 않는다.

## 1.2 hidden을 logits·확률·causal target으로 바꾼다

embedding과 transformer block이 만든 hidden은 vocabulary 방향의 상대 점수로 투영된다. softmax의 수치 안정성과 causal shift·mask를 한 흐름에서 읽어야 ‘정답 확률’이 어느 원문 위치의 어떤 token을 뜻하는지 확정할 수 있다.

### hidden projection과 stable softmax를 계산한다

마지막 hidden state `h∈R^C`와 출력 행렬 `W∈R^{V×C}`의 곱 `z=Wh`가 logit이다. logit은 확률이 아니다. 모든 후보에 같은 상수를 더해도 선택은 달라지지 않는 상대 점수다. softmax는 `p_i=exp(z_i-m)/Σ_j exp(z_j-m)`로 이를 확률 simplex에 놓는다. `m=max(z)`를 빼는 까닭은 수학적 값이 같으면서 지수 overflow를 막기 위해서다.

정답 ID가 `y`일 때 한 위치의 손실은 `-log p_y`이고 logit gradient는 `p-one_hot(y)`다. 정답 확률을 올리는 동시에 다른 후보를 현재 확률만큼 내린다. 평균 손실 하나만 보면 어느 위치의 어느 오답이 원인이었는지 사라지므로, 디버깅 때는 위치별 loss와 `logits.max()`를 함께 남긴다.

**Log-sum-exp로 직접 계산한다.** `log p_y=z_y-logΣ_j exp(z_j)`이므로 CE는 `logsumexp(z)-z_y`다. 안정 구현은 `m=max_j z_j`를 빼서 `m+logΣ_j exp(z_j-m)-z_y`를 쓴다. `m`은 gradient에서 상수처럼 보일 수 있지만 max 선택의 미분을 따로 넣지 않아도 전체 logsumexp derivative는 softmax가 된다.

두 logits `[2,0]`이면 정답 0의 확률은 `e²/(e²+1)≈0.881`, loss는 약 `0.127`이다. temperature `τ`를 적용한 `[2/τ,0]`에서 `τ<1`은 분포를 날카롭게 하고 gradient를 softmax 변화와 `1/τ` chain factor로 바꾼다. training objective의 temperature와 generation sampling temperature를 같은 옵션으로 취급하지 않는다.

**확률·entropy·temperature를 같은 toy row에서 읽는다.** logits `[2,0]`의 확률은 약 `[0.881,0.119]`, entropy는 약 `0.365 nat`다. `τ=2`이면 loss 함수에 들어가는 logits는 `[1,0]`, 확률은 약 `[0.731,0.269]`, entropy는 약 `0.582 nat`가 된다. 분포는 완만해졌지만 원 logits `z`에 대한 gradient는 `(p-one_hot)/τ`, 즉 정답 좌표가 약 `-0.134`, 오답 좌표가 `+0.134`다. `τ=1`의 약 `±0.119`보다 작은 것이 아니라 오히려 이 예에서는 조금 크다. `1/τ`만 보고 gradient가 언제나 줄어든다고 결론 내릴 수 없는 이유는 `p` 자체도 동시에 변하기 때문이다. 반면 decoding temperature는 backward가 없으므로 이 chain factor가 없다.

entropy는 분포 전체의 불확실성을 묻고, 정답 NLL은 지정된 한 좌표에 준 질량을 묻는다. 세 class에서 `[0.49,0.49,0.02]`와 `[0.49,0.255,0.255]`는 정답이 첫 class라면 NLL이 같지만 entropy는 다르다. top-1 confidence와 NLL이 같아도 tail geometry가 다를 수 있다는 뜻이다. calibration까지 말하려면 이 confidence가 반복 표본의 실제 적중률과 맞는지 별도 집계해야 한다.

**최대우도와 KL의 관계.** 데이터의 empirical distribution을 `q`, model을 `pθ`라 하면 cross entropy `H(q,pθ)=H(q)+KL(q||pθ)`다. 데이터가 고정이면 `H(q)`는 θ와 무관하므로 CE 최소화는 forward KL 최소화와 같다. 그러나 finite corpus의 empirical q가 현실 언어 분포 자체는 아니다. 중복·mixture·curriculum이 q를 바꾸고 곧 학습 목적을 바꾼다.

**Entropy와 confidence.** 낮은 CE는 정답에 높은 확률을 주었다는 뜻이지 전체 분포가 잘 calibration됐다는 뜻은 아니다. entropy `-Σp logp`, top-1 margin, expected calibration error는 다른 질문이다. label smoothing을 쓰면 target one-hot 대신 `(1-ε)`와 나머지 질량을 섞어 logit gradient를 바꾼다. smoothing 계수는 데이터 노이즈 대응과 과신 억제의 trade-off지만 희귀 정답의 최대 confidence도 제한한다.

| 수량 | 식 | 분모 | 디버깅 용도 |
|---|---|---|---|
| token loss | `-log p_y` | 없음 | 어려운 위치 찾기 |
| loss sum | `Σ_i l_i` | 없음 | 분산 합산 numerator |
| token mean | `Σ_i l_i/N_valid` | valid label | optimizer objective |
| perplexity | `exp(token mean)` | tokenizer token | 같은 좌표계에서만 비교 |
| bits/byte | `loss sum/(bytes·ln2)` | 원문 byte | tokenizer 간 보조 비교 |

**반례 3—perplexity가 낮다고 tokenizer가 우월하지 않다.** token 단위가 다르면 분모가 다르다. 같은 문자열의 총 NLL이나 bits-per-byte를 함께 본다.

**반례 4—logit max가 크다고 overflow인 것은 아니다.** stable logsumexp는 큰 공통 offset을 제거한다. 반면 `inf`가 이미 projection에서 생겼다면 max subtraction으로 고칠 수 없다. 최초 non-finite tensor를 찾는다.

**실험 1-A—shift invariance.** 모든 logits에 같은 scalar를 더해 softmax와 CE가 같은지 본다. naive exp 구현은 큰 scalar에서 overflow하고 stable 구현은 유지돼야 한다.

**실험 1-B—finite difference.** vocab 3의 logits에 대해 각 좌표를 `±ε` 이동한 중앙차분과 `p-one_hot`을 비교한다. ε가 너무 작으면 floating rounding, 너무 크면 truncation error가 커진다. 여러 ε에서 안정 구간을 찾는다.

### causal shift와 세 종류 mask를 분리한다

**label shift·mask·reduction denominator**

입력 `[a,b,c]`의 예측 대상은 `[b,c,d]`다. 학습 텐서가 `[B,T]`라면 위치 `t`의 logit은 label `t`를 맞히는 것이 아니라 원본 시퀀스의 `t+1`을 맞힌다. prompt-only 토큰이나 padding에는 `-100` 같은 ignore 표지를 두되 attention mask와 혼동하지 않는다. 전자는 loss 기여 여부이고 후자는 어느 key를 볼 수 있는지다.

유효 위치 집합 `M`에 대한 손실은 `L=-Σ_(b,t∈M)log p(y_bt)/|M|`이다. 길이별 예제 평균을 다시 평균하면 짧은 예제의 토큰이 더 큰 가중치를 얻는다. 분산 학습에서도 rank별 평균을 평균하면 유효 토큰 수가 다른 rank에서 같은 왜곡이 생긴다. 먼저 loss sum과 valid count를 각각 합쳐야 한다.

**Shift를 index로 쓴다.** 원 token sequence가 `s_0,…,s_T`라면 input `x_t=s_t`, target `y_t=s_{t+1}`다. model 내부에서 labels를 shift하는 stack도 있고 collator가 이미 shifted target을 만드는 stack도 있다. 둘 다 적용하면 두 칸 앞을 예측한다. nanoGPT `train.py:123-125`는 memmap slice를 한 칸 어긋나게 만들어 model에 넘기므로 `model.py:184-187`은 추가 shift 없이 flatten CE를 계산한다.

golden batch 첫 행 `x=[11,7,91,44,5,5,19,2]`, `y=[7,91,44,5,5,19,2,-1]`에서 `y[0:7]=x[1:8]`이다. 마지막 `-1`은 이 교육 model에서 ignore다. Transformers collator의 흔한 `-100`을 그대로 넣으면 계약이 다르다.

**Attention mask와 label mask의 교차표.** prompt token은 attention key로 볼 수 있으나 label loss에서는 제외될 수 있다. padding은 attention과 label 모두에서 제외한다. packed document 경계는 label만 끊을지 attention도 끊을지 선택한다. 네 경우를 한 boolean로 표현하려 하면 오류가 생긴다.

| token 종류 | query로 계산 | key로 보임 | target loss | 이유 |
|---|---:|---:|---:|---|
| system/user prompt | 예 | 예 | recipe별 제외 | 답변 context |
| assistant answer | 예 | 예 | 예 | supervised target |
| padding | 보통 아니오 | 아니오 | 아니오 | shape 채움 |
| 이전 packed 문서 | 현재 token과 별도 policy | block 여부 | 경계 target 제외 | leakage 제어 |

**분산 분모 유도.** rank r의 loss sum `S_r`, valid count `N_r`에서 전역 loss는 `ΣS_r/ΣN_r`다. rank mean `S_r/N_r`의 단순 평균은 `N_r`가 모두 같을 때만 같다. DDP가 gradient를 rank 수로 평균하는 구현이라면 local loss scaling을 전역 count와 collective semantics에 맞춘다. metric용 all-reduce와 gradient용 all-reduce를 섞지 않는다.

**Gradient accumulation 분모.** microbatch별 valid count가 다르면 각 mean을 K로 나누는 것은 token mean이 아니라 microbatch mean의 평균이다. loss sum을 누적하고 전체 count로 scale하거나 framework가 제공하는 batch-wide item count를 사용한다. assistant-only SFT에서 답변 길이가 달라질 때 특히 중요하다.

**반례 5—padding loss를 가렸는데 결과가 달라진다.** attention mask가 padding key를 막지 않으면 valid token hidden state가 padding embedding을 볼 수 있다. label ignore만으로 충분하지 않다.

**반례 6—rank별 loss가 같아도 전역 gradient가 틀릴 수 있다.** 서로 다른 token에서 우연히 scalar mean이 같을 수 있다. numerator/count와 parameter gradient를 비교한다.

**실패 주입 1-C—double shift.** label을 이미 이동한 뒤 model loss에서 다시 slice한다. loss는 finite하고 내려갈 수 있다. golden `x,y` relation과 특정 position의 expected target을 assertion으로 둔다.

**실패 주입 1-D—ignore ID 혼동.** `-1`과 `-100`을 바꾸어 framework가 fail-fast하는지 본다. 조용히 vocabulary negative index로 해석하는 custom kernel이라면 심각한 corruption이다.

### 작은 fixture로 numerator·denominator를 손계산한다

vocabulary 3, `B=1`, loss-bearing logit row 2개인 최소 fixture를 만든다. raw token window가 `s=[2,1,0]`이라면 model input은 `x=[2,1]`, 정렬된 target은 `y=[1,0]`이다. 첫 row logits를 `[0,ln2,0]`, 둘째를 `[ln3,0,0]`으로 고정한다. 첫 row 확률은 `[1/4,1/2,1/4]`이고 NLL은 `ln2`다. 둘째 확률은 `[3/5,1/5,1/5]`이고 NLL은 `ln(5/3)`이다. 두 위치가 모두 유효하면

`S=ln2+ln(5/3)=ln(10/3)`, `N=2`, `L=S/N`이다.

mean reduction 뒤 첫 row의 `dlogits`는 `[1/8,-1/4,1/8]`, 둘째는 `[-1/5,1/10,1/10]`이다. 각 row 성분 합이 0이고 두 row 모두 unreduced gradient의 정확히 절반이어야 한다. 둘째 target을 ignore로 바꾸면 `S=ln2`, `N=1`, 둘째 row의 direct loss gradient는 0이며 첫 row gradient는 `[1/4,-1/2,1/4]`로 두 배가 된다. “한 위치를 mask했으니 남은 위치 gradient는 그대로다”라는 직관이 틀리는 이유는 mean의 denominator도 바뀌기 때문이다. sum reduction이라면 남은 위치 gradient는 그대로다.

이 fixture는 네 종류의 오류를 분리한다. 정답이 `[0,?]`로 보이면 shift 또는 position mapping이 틀렸다. 확률은 맞는데 첫 NLL이 `ln4`라면 정답 좌표를 잘못 읽었다. 둘째를 ignore했는데 `N=2`면 mask/count 경계가 틀렸다. scalar는 맞는데 첫 gradient가 unreduced 값이면 reduction이 forward에만 적용되고 backward scale이 어긋난 custom kernel이다. 실제 integration test에서는 이 숫자를 LM head 앞의 고정 hidden과 weight로 재구성해 `dW=dzhᵀ`, `dh=Wᵀdz`까지 비교한다.

분산 fixture로 확장할 때는 첫 row를 rank 0, 둘째 row를 rank 1에 놓는다. local mean을 단순 rank 평균하면 우연히 이 경우 전역 token mean과 같지만, 둘째 rank에 같은 target을 두 번 복제하는 순간 count가 1:2가 되어 달라진다. 따라서 대칭 fixture 하나의 성공은 global-denominator 구현의 증거가 아니다. 반드시 불균등 count 반례를 함께 둔다.

## 1.3 logsumexp와 CE의 기하를 두 logit에서 본다

두 class toy model은 공통 logit offset이 사라지는 이유, stable logsumexp와 gradient·곡률을 동시에 보여 준다. 이 성질을 먼저 손으로 확인한 뒤 큰 vocabulary의 수치 결과를 해석한다.

### logit 차이만 남는 loss surface를 그린다

정답이 첫 클래스이고 logits가 `[u,v]`라면 `L=log(exp(u)+exp(v))-u=softplus(v-u)`다. 손실은 두 좌표의 절대값이 아니라 차이 `v-u`만 본다. `[1000,999]`를 그대로 지수화하면 overflow할 수 있지만 차이는 안전하다. gradient는 `[-p_2,p_2]`이므로 이미 정답이 압도적이면 update가 작아진다.

검산은 세 단계로 한다. 중앙차분 `(L(u+ε)-L(u-ε))/(2ε)`, autograd의 `u.grad`, 닫힌식 `p_1-1`이 허용 오차 안에서 같아야 한다. 다르면 먼저 dtype과 reduction을 확인한다.

### Hessian의 양의 준정부호성과 보이지 않는 방향을 해석한다

softmax CE의 logit Hessian은 `diag(p)-ppᵀ`다. 임의 vector `v`에 대해 `vᵀHv=Σp_i v_i²-(Σp_i v_i)²`, 즉 p 아래의 variance이므로 음수가 아니다. 모든 logit에 같은 상수를 더하는 방향에서는 variance가 0이다. loss가 공통 offset에 불변인 것과 같은 사실이다.

두 class에서는 확률 `p,1-p`일 때 차이 방향 curvature가 `p(1-p)`에 비례한다. 확률이 0.5 근처면 curvature가 크고 포화되면 작다. 정답을 매우 확신해 틀린 경우 gradient는 크지만 수치적 softmax가 underflow하지 않게 stable 구현이 필요하다.

**Embedding까지 chain rule.** logit `z=Wh`, hidden `h=E[id]+…`라면 `dW=dzhᵀ`, `dh=Wᵀdz`다. embedding row에는 `dh`가 더해진다. 같은 ID가 여러 위치에 나오면 합산한다. tied W=E이면 앞서 설명한 output gradient `dW`도 같은 parameter에 더해진다.

**Toy tensor ledger.** vocab 2, hidden 2, batch 1, sequence 1로 두고 `h=[1,-1]`, `W=[[1,0],[0,1]]`, target 0이라 하자. logits `[1,-1]`, `p_0≈0.881`, `p_1≈0.119`, loss 약 0.127이다. `dz≈[-0.119,0.119]`, `dh=Wᵀdz`도 같다. 이 수치를 손계산·script·framework로 비교한다.

**옵션이 실제로 바꾸는 것.** label smoothing은 target distribution, temperature는 logit scale과 gradient, ignore index는 유효 집합, reduction은 분모, weight tying은 parameter identity와 gradient 합산을 바꾼다. 옵션 표를 외우지 말고 tensor/state diff로 남긴다.

**Upstream test의 범위.** nanoGPT snapshot에는 이 loss 경로를 검증하는 독립 unit test가 없다. `model.py`와 실행 recipe가 고정 source다. PyTorch cross-entropy test가 stable reduction을 검증하더라도 nanoGPT의 pre-shifted batch와 tied model을 대신 검사하지 않는다. 이 장의 golden fixture는 독자용 test이며 upstream 보장으로 표현하지 않는다.

**조사 체크리스트—loss가 이상할 때.** tokenizer와 model vocab을 대조한다. 한 행의 x/y shift를 출력한다. ignore ID와 valid count를 센다. 위치별 loss를 원 token span에 역투영한다. logits finite/max와 stable CE를 본다. reduction이 sum/mean 중 무엇인지 확인한다. accumulation/rank별 numerator와 count를 합친다. tied storage와 gradient를 확인한다.

**디버깅 결정 트리.** ID가 범위를 벗어나면 tokenizer/resize 문제다. ID는 맞고 embedding부터 다르면 checkpoint/tie 문제다. logits가 non-finite면 첫 비정상 layer로 올라간다. logits는 같고 loss가 다르면 target/ignore/reduction이다. loss는 같고 gradient가 다르면 smoothing/scale/tie/accumulation이다. scalar metric만 다르면 logging denominator를 본다.

**재현 절차.** golden IDs와 labels를 little-endian int64 checksum으로 확인한다. FP64 작은 logits에서 closed-form, finite difference, framework gradient를 비교한다. FP32·BF16로 dtype을 낮추며 loss와 gradient error를 기록한다. label mask와 batch 구성을 바꿔 numerator/count 합산을 검사한다. 실행하지 않은 결과에는 예상 shape와 invariant만 남긴다.

**1장의 실제 인계.** 2장에는 scalar loss, logit gradient, valid count를 넘긴다. 5장에는 문자열→ID 좌표의 요구사항을, 7장에는 embedding row와 tied identity를 넘긴다. 10장은 같은 `GoldenBatchID`로 logits와 backward atlas를 만든다. 24장은 token loss를 evaluation denominator와 연결한다.

**Forward KL과 reverse KL을 섞지 않는다.** 최대우도는 데이터에서 표본을 뽑아 `-log pθ(x)`를 평균하므로 `KL(q_data||pθ)` 방향이다. model이 질량을 주지 않은 데이터 영역은 큰 벌을 받지만, 데이터에 없는 model mode는 직접 표본화되지 않는다. 반대로 `KL(pθ||q)`는 q가 작은 영역에 model이 질량을 두는 것을 강하게 꺼리고 mode-seeking 성질을 보일 수 있다. DPO·RL의 reference KL은 어느 분포에서 expectation을 취하고 어느 방향인지 식으로 확인한다. “KL penalty”라는 이름만으로 효과를 추정하지 않는다.

**Log base와 단위.** 자연로그 CE의 단위는 nat이고 base-2 log면 bit다. perplexity는 자연로그 mean loss에 exp를 적용하거나 bit loss에 2의 거듭제곱을 적용해야 한다. byte-normalized 값은 `NLL_nats/(byte_count·ln2)`다. tokenizer가 normalization으로 byte를 바꾸었다면 원 byte와 normalized byte 중 어느 것을 분모로 썼는지 적는다.

**Loss 평균의 세 가지 서로 다른 질문.** token mean은 corpus의 각 supervised token에 같은 무게를 준다. sequence mean은 각 sequence에 같은 무게를 준다. domain mean은 각 domain metric에 같은 무게를 줄 수 있다. 어느 것이 “공정한가”는 목적에 달렸지만 optimizer objective와 dashboard metric이 다르면 명칭을 구분한다. 길이 2와 8인 sequence에서 각각 loss sum 2와 16이면 token mean은 `18/10=1.8`, sequence mean은 `(1+2)/2=1.5`다.

**Class weight와 vocabulary masking.** 특정 token에 class weight를 주거나 허용 vocabulary 밖 logits를 `-∞`로 막으면 gradient target과 partition function이 달라진다. padding target을 ignore하는 것과 padding logit을 후보에서 제거하는 것은 별개다. vocabulary mask가 position마다 다르면 softmax 분모도 position마다 달라진다. constrained generation의 mask를 training CE에 무심코 적용하지 않는다.

**Tied weight의 parameter ledger.** input embedding 이름과 LM head 이름, storage pointer, shape, optimizer group, gradient checksum을 한 행의 alias group으로 묶는다. state dict가 같은 storage를 두 key로 직렬화할 수도 있고 한 key로 다룰 수도 있다. 값이 같은 두 복사본과 실제 alias를 구분한다. optimizer는 alias parameter를 두 번 step해서는 안 된다.

**Label shift test를 property로 만든다.** 임의 token window `s`에서 loader가 만든 `x,y`에 대해 모든 유효 t에서 `y_t=x_{t+1}`을 검사한다. 마지막 target은 다음 원 token 또는 ignore 정책과 일치해야 한다. model-internal shift stack에서는 raw labels와 shifted view를 별도로 출력한다. 한 example만 고정한 test보다 random length와 boundary를 생성하는 property test가 packing off-by-one을 더 잘 잡는다.

**Mask test는 두 perturbation을 쓴다.** ignored target의 label 값을 다른 유효 ID로 바꿔도 loss가 변하지 않아야 한다. 그러나 ignored 위치의 input token을 바꾸면 뒤 valid position의 context가 달라져 loss가 변할 수 있다. 두 test를 함께 두면 label ignore와 attention/context 효과를 구분한다.

**Tied-gradient test.** 같은 초기값에서 tied model과 untied model을 만든다. untied embedding gradient와 head gradient를 같은 좌표로 합친 값이 tied gradient와 맞아야 한다. repeated ID 행의 lookup 기여는 occurrence별 hidden gradient 합과 맞아야 한다. optimizer step 전 test해야 moment와 decay가 섞이지 않는다.

**분모 test matrix.** batch마다 valid count가 같은 경우와 다른 경우, rank마다 같은 경우와 다른 경우를 만든다. single concatenated reference의 loss sum/count/gradient를 microbatch·DDP simulation 결과와 비교한다. 단순 scalar mean 비교뿐 아니라 parameter gradient를 본다. rank average semantics를 mock collective로 명시한다.

**수치 실패 주입.** logits에 `10^4` 공통 offset을 더해 naive exp와 stable logsumexp를 비교한다. 한 logit을 `inf`, 하나를 `nan`으로 만들어 fail-fast policy를 확인한다. BF16에서 매우 작은 확률이 0으로 underflow해도 log-softmax가 finite한지 본다. loss가 finite라도 gradient가 non-finite일 수 있으므로 둘을 별도 검사한다.

**옵션 상태 변화 조사표.** `ignore_index`는 M 집합, `reduction`은 numerator/denominator 반환 형태, smoothing은 target q, temperature는 logits와 chain scale, class weight는 token별 계수, tying은 parameter graph, vocabulary mask는 partition function을 바꾼다. config field→loss 함수 인자→실제 tensor diff→gradient diff를 추적한다.

**현장 조사—loss가 다른 두 stack.** 먼저 동일 logits tensor를 파일로 고정한다. 두 stack의 labels·mask·valid count를 비교한다. 같은 logits/labels로 unreduced per-token loss를 비교한다. 그다음 reduction과 dtype을 맞춘다. 여기까지 같으면 model forward 문제로 올라간다. model부터 비교하면 tokenizer·분모 차이가 모든 layer mismatch처럼 보인다.

**현장 조사—perplexity regression.** tokenizer revision, evaluated bytes, truncation, overlap stride, BOS/EOS, loss-bearing target 수를 먼저 diff한다. model weight가 같아도 evaluation window가 달라지면 context와 target set이 바뀐다. aggregate perplexity 뒤에 document별 NLL/byte와 high-loss span을 둔다.

**상류 소스와 로컬 fixture의 경계.** nanoGPT 소스는 embedding→LM head→CE 호출을 보여준다. PyTorch CE는 stable log-softmax/reduction 구현을 제공한다. 그러나 golden loader shift, tokenizer offsets, DDP global denominator, tied optimizer alias까지 하나의 상류 테스트가 모두 보장하지는 않는다. 로컬 fixture는 이 stack 조합을 검사하고 소스와 test script의 commit을 함께 기록한다.

**종료 판정.** 독자는 raw string에서 한 target token의 byte span, ID, embedding row, logit, probability, NLL, valid-count 기여, logit gradient와 tied parameter gradient까지 역추적할 수 있어야 한다. scalar loss 하나만 재현한 상태는 종료가 아니다. 이 chain이 고정돼야 2장의 backward 차이를 올바르게 해석할 수 있다.

**조사 체크리스트—데이터에서 logit까지.** `CorpusRevision`과 `DocumentID`를 적는다. normalization 전후 bytes와 offset을 확인한다. tokenizer·template checksum과 ID 범위를 본다. `GoldenBatchID`의 x/y relation, mask, valid count를 검산한다. embedding/head shape와 tying을 확인한다. 한 위치 logits에서 stable logsumexp를 재계산한다. per-token NLL을 원 byte span에 되돌린다. 이 순서의 첫 mismatch를 원인 후보로 삼는다.

**조사 체크리스트—분산 metric.** rank별 loss sum, valid count, sequence count를 별도 수집한다. 전역 token mean을 재계산하고 dashboard 값과 비교한다. accumulation window와 optimizer step ID를 붙인다. padding·assistant mask 비율을 rank별로 본다. rank mean의 평균만 제공하는 logger라면 numerator/count metric을 추가한다.

**실험 1-E—짧은 예제 편향.** valid 길이 2와 8인 두 sequence를 만들고 token mean과 sequence mean gradient를 비교한다. 두 objective가 어느 row를 더 크게 업데이트하는지 기록한다. recipe가 sequence mean을 의도했다면 오류가 아니지만 명칭과 분모를 명시한다.

**실험 1-F—tying과 decay.** tied·untied model을 같은 초기값으로 만들고 CE gradient를 비교한다. 이어 weight decay만 적용한 step에서 tied parameter가 한 번만 decay되는지 본다. parameter alias가 optimizer group에 중복되면 gradient가 맞아도 update가 틀릴 수 있다.

**실험 1-G—tokenizer 단위.** 같은 raw bytes를 두 tokenizer로 나누고 token mean CE를 직접 비교하지 않는다. 각 model의 total NLL, supervised bytes, bits/byte, truncation된 byte 수를 함께 report한다. 모델 weight와 학습량이 다르면 tokenizer quality의 인과 결론을 보류한다.

**Logit clipping의 함정.** overflow를 막겠다며 logits를 임의 범위로 clip하면 포화 영역의 gradient가 0이 될 수 있다. stable logsumexp와 compute dtype 승격이 먼저다. clipping이 의도된 objective라면 threshold와 clipped 비율을 metric으로 둔다.

**Weight tying의 trade-off.** parameter와 lexical 좌표를 공유해 통계 효율과 메모리를 얻지만 input representation과 output classifier가 같은 geometry를 써야 한다. multimodal placeholder나 control token처럼 입력과 출력 역할이 비대칭인 token에서는 gradient 경로를 살핀다. tying을 무조건 우월한 default로 설명하지 않는다.

**복구 handoff.** loss 자체에는 durable state가 없지만 어떤 token이 numerator와 denominator에 기여했는지는 consumption ledger에 남는다. resume에서 같은 batch를 재처리하면 metric과 gradient가 중복된다. `CheckpointID`가 마지막 committed optimizer step과 다음 `GoldenBatchID`를 함께 가리켜야 한다.

**최종 의미 불변식.** 공통 logit offset은 확률을 바꾸지 않는다. softmax row 합은 1이다. 유효하지 않은 target의 direct loss gradient는 0이다. 전역 loss는 loss sum/valid count다. tied parameter gradient는 input/output 경로의 합이다. 이 다섯 assertion이 1장의 최소 실행 계약이다.

**코드 검토 질문.** loss 함수가 logits와 labels 중 무엇을 shift하는가. flatten 전에 contiguous가 필요한가. ignore index가 collator와 같은가. reduction 전에 class/token weight가 적용되는가. denominator가 valid token인지 batch size인지. distributed wrapper가 numerator/count 또는 이미 평균된 gradient 중 무엇을 줄이는가. tied weight가 optimizer에 중복 등록되는가. evaluation도 같은 contract를 쓰는가. 이 질문에 소스 코드 줄을 근거로 답하지 못하면 metric 이름만 믿지 않는다.

**실패 보고 예시.** “loss가 0.3 다르다” 대신 `Run A/B에서 logits checksum 동일, labels 동일, valid count A=14/B=16, 두 padding label이 B에서 loss에 포함됨`이라고 쓴다. 원인과 영향 범위가 즉시 드러난다. logits부터 다르면 첫 다른 hidden node를 10장 방식으로 연결한다.

**장간 산출물 표.** 5장에서 `TokenizerRevision`과 offsets를 읽고, 1장에서 positions별 `target_id,NLL,dlogit`을 쓴다. 2장은 그 gradient를 읽는다. 3장은 UpdateID에 묶는다. 24장은 EvalID의 metric contribution으로 다시 사용한다. 동일 token contribution ID가 학습과 평가에서 충돌하지 않도록 namespace와 split을 둔다.

**독자 확인 문제.** valid token이 각각 1개와 9개인 두 rank가 mean loss 1과 3을 냈다면 rank mean은 2지만 token mean은 2.8이다. 어느 gradient가 원하는 objective인지 설명하고 local loss scaling 식을 써 본다. tied embedding에서 input에 없던 row도 gradient를 받을 수 있는 경로를 그린다. 이 두 문제를 풀 수 있어야 다음 장으로 간다.

답은 숫자만 맞히지 말고 numerator와 denominator의 소유자를 표시해야 한다. 같은 식을 microbatch 두 개, DDP rank 두 개, 평가 batch 두 개에 적용해 어떤 collective와 logging 단계가 필요한지 적는다. 마지막으로 해당 token contribution을 원 byte offset까지 되짚어 본다.

이 역추적 결과와 계산 script revision을 함께 저장해야 다른 독자가 동일한 분모와 target을 독립적으로 검산할 수 있다.

검산이 끝난 artifact만 다음 장으로 전달한다.

**이 장이 넘기는 것.** `GoldenBatchID`, token IDs, shifted labels, loss mask, valid-label count, embedding 입력 계약을 2장과 7장에 넘긴다.

**다음 장에서 깨질 수 있는 것.** loss는 맞아도 gradient accumulation 분모나 mixed-precision scaling 순서가 틀릴 수 있다.

**검증 체크포인트.** 모든 ID는 `[0,V)`에 있고, 위치별 softmax 합은 1이며, loss sum을 valid count로 나눈 값이 framework loss와 일치해야 한다.

## 1.4 구현에서 shift·reduction·kernel의 소유자를 찾는다

수학식이 같아도 collator, model forward와 loss helper 가운데 누가 shift와 reduction을 수행하는지는 stack마다 다르다. public API부터 selected backend까지 따라가되 source 좌표와 upstream test가 증명하는 범위를 분리한다.

### `ForCausalLMLoss`의 pad·shift·flatten 경계를 추적한다

이제 식을 실제 라이브러리 경로에 붙여 보자. 기준 Transformers checkout은 commit `36deb0b53ed0863f4b4dfdea23dcaec7f3df3701`이다. `src/transformers/loss/loss_utils.py:49-70`의 `ForCausalLMLoss`는 logits를 float로 올리고, 호출자가 `shift_labels`를 주지 않았을 때 labels 오른쪽에 ignore 값을 붙인 다음 `labels[...,1:]`를 연속 텐서로 만든다. 이어 logits를 `[-1,V]`, labels를 `[-1]`로 펴서 `fixed_cross_entropy`에 넘긴다. 함수 하나가 dtype 승격, causal shift, flatten, device 이동, reduction이라는 다섯 계약을 소유한다.

collator가 이미 `x=s[:-1], y=s[1:]`를 만들었는데 raw `labels=y`를 이 함수에 넘기면 target이 두 칸 이동한다. 반대로 raw sequence를 inputs와 labels에 그대로 복사하는 언어 모델 collator에는 함수 안 이동이 필요하다. 어느 관례가 옳은지는 이름으로 판단하지 않는다. 첫 batch 여섯 ID와 각 logit 위치의 원문 target을 표로 만들면 소유권이 드러난다. loss가 잘 내려가는지는 증거가 아니다. 두 칸 뒤 token에도 통계적 규칙이 있으므로 잘못된 목표도 학습된다.

같은 파일 `loss_utils.py:31-45`의 `fixed_cross_entropy`는 `num_items_in_batch`가 없으면 `mean`, 있으면 `sum`을 고른 뒤 합을 전달받은 항목 수로 나눈다. 길이가 다른 microbatch를 accumulation할 때 microbatch mean의 평균이 아니라 전체 유효 target 평균을 만들기 위한 분기다. 따라서 이 값은 padding 포함 element 수도 input token 수도 아니고 실제 loss-bearing target 수여야 한다.

Trainer의 같은 checkout에서 `trainer.py:518-527`은 causal loss에서 각 sequence 첫 token이 prediction target이 아니라는 사실을 count에 반영한다. `trainer.py:1741-1745`는 한 optimizer update에 들어갈 microbatch와 분모를 함께 준비한다. `trainer.py:1985-2061`의 `compute_loss` 경로는 사용자 정의 loss가 그 값을 받거나 무시하는 경계를 보여 준다. loss 분모는 loader, collator, model config, loss mapping의 공동 산물이다.

| 연산 | 소유자 | 입력→출력 | 반드시 남길 증거 |
|---|---|---|---|
| normalization | tokenizer | bytes→text | revision·offset diff |
| special token | template | roles→IDs | role별 ID 열 |
| causal shift | collator 또는 loss | sequence→targets | 위치 대응표 |
| label ignore | collator | targets→valid set | valid count |
| CE numerator | loss kernel | logits,target→sum | FP64 fixture |
| normalization | trainer/loss | sum,count→mean | concat reference |

shift 소유자는 정확히 하나여야 한다. kernel이 이미 mean을 냈는데 Trainer가 count로 다시 나누면 이중 평균이다. rank별 local mean을 DDP가 평균하면 rank valid count가 다를 때 token mean이 아니다. 옵션을 이해했다는 말은 이 표에서 어느 상태가 바뀌는지 설명할 수 있다는 뜻이다.

### source assertion과 upstream fixture의 증명 범위를 제한한다

함수 test가 단일 batch shift와 reduction을 확인해도 자기 collator와 accumulation까지 증명하지 않는다. 자기 golden test가 통과해도 라이브러리의 모든 dtype/device 경로를 보장하지 않는다. 증거를 함수, integration, run 세 층으로 나눈다. 함수 test는 local invariant, integration은 collator→model→loss, run test는 accumulation·DDP·resume를 검사한다. 어느 test 파일의 어떤 assertion인지 좌표가 없는 “upstream 검증 완료”는 근거로 쓰지 않는다.

독자용 통제 실험은 작은 tensor면 충분하다. raw labels 경로와 명시적 `shift_labels` 경로에 같은 target을 구성해 numerator와 gradient를 비교한다. ignore 위치의 label을 바꾸어도 loss가 유지되는지 본다. 그러나 ignore 위치 input ID를 바꾸면 뒤 context가 바뀔 수 있으므로 별도 실험으로 둔다. 길이 1과 9의 microbatch를 연결한 reference와 `sum/count` accumulation의 parameter gradient도 비교한다.

## 1.5 probability geometry와 tensor atlas를 함께 읽는다

loss scalar만 보면 공통 offset, temperature와 smoothing이 만든 서로 다른 변화를 구분할 수 없다. 위치별 logits·probability·hidden을 atlas로 남겨 수학적 불변식과 실제 tensor의 최초 차이를 연결한다.

### 공통 offset 불변성과 simplex tangent를 읽는다

softmax는 V차원 logit을 V-1차원 simplex로 보낸다. 모든 좌표에 같은 상수 `c`를 더하는 방향은 확률을 바꾸지 않는다. logit Hessian에는 그 방향의 영 고윳값이 있다. 절대 logit 크기만으로 confidence를 해석할 수 없는 이유다. 그러나 절대 크기는 dtype overflow와 양자화 범위에는 영향을 준다. 목적함수의 불변성과 수치 표현의 안전성은 다른 질문이다.

simplex 경계에서 일부 확률은 0에 가까워진다. softmax 확률을 만든 뒤 log를 취하는 순진한 구현은 작은 값이 먼저 0으로 underflow할 수 있다. `log_softmax`와 log-sum-exp를 결합하는 까닭은 중간의 취약한 확률을 materialize하지 않기 위해서다. projection에서 이미 `inf`가 생겼다면 stable CE가 되살릴 수 없으므로 최초 non-finite tensor를 위로 추적한다.

categorical score covariance는 `diag(p)-ppᵀ`이고 logit CE Hessian과 같은 꼴이다. 그렇다고 깊은 network parameter 공간의 Hessian이 항상 양의 준정부호라는 뜻은 아니다. parameter Jacobian과 network의 2차 미분이 더 들어간다. “CE는 convex”라는 문장은 logits를 직접 변수로 볼 때만 유효하다.

### smoothing과 temperature의 target·gradient 변화를 분리한다

label smoothing은 one-hot target을 simplex 내부로 옮긴다. 구현에 따라 남은 질량을 모든 V class에 `ε/V`로 나누거나 정답을 제외한 V-1개에 나눈다. 작은 vocabulary fixture로 target 합, 정답 질량, ignore 처리까지 확인한다. noisy label에서는 과신을 줄일 수 있지만 희귀 정답 신호를 약화할 수도 있다. clean NLL, rare-token recall, calibration을 같은 지표로 뭉개지 않는다.

학습 logits를 `z/τ`로 나누면 분포뿐 아니라 chain factor `1/τ`가 gradient에 들어간다. decoding temperature는 checkpoint를 바꾸지 않고 sampling entropy만 바꾼다. distillation temperature는 softened teacher target과 흔히 `τ²` 보정에 연결된다. 이름이 같아도 바꾸는 state와 derivative가 다르다. training, distillation, decoding 세 실험을 한 그래프의 “temperature 효과”로 합치지 않는다.

**텐서 atlas로 최초 오류를 찾는다**

### forward와 backward의 최소 tensor probe를 정한다

모든 activation을 저장하지 않는다. token IDs min/max/checksum, embedding과 block boundary의 finite 비율·RMS·max, final norm, logits margin, loss numerator와 valid count만 먼저 남긴다. golden batch에서는 선택 위치의 slice checksum을 더한다. checksum이 처음 달라지는 경계가 원인 후보를 나눈다. logits까지 같고 loss만 다르면 target·mask·reduction이고, embedding부터 다르면 revision·weight·position·dtype을 본다.

backward에서는 loss scale 전후, LM head·final norm·첫 block·embedding gradient의 finite 비율과 RMS를 본다. tied embedding/head는 같은 storage인지 확인해 두 번 세지 않는다. accumulation 중 `.grad`에는 앞 microbatch 기여가 있으므로 microstep과 zero-grad 경계를 붙인다. 모든 gradient가 같은 배율로 다르면 reduction·accumulation divisor·AMP scale·DDP semantics가 우선 후보다.

probe가 graph를 붙잡아 memory leak를 만들지 않도록 detach 경계를 명시한다. device 값을 매 step host로 가져오면 synchronization이 생길 수 있으므로 정상 운전은 낮은 빈도의 통계, 재현 run은 상세 slice로 나눈다. 이 책에서는 CUDA나 모델을 실행하지 않고 관측 계약과 예상 invariant만 정의한다.

**실패 주입 네 가지**

vocabulary 밖 ID를 넣으면 embedding 경계에서 즉시 실패해야 한다. 한 target을 ignore로 바꾸면 numerator와 count만 예상대로 변해야 한다. logits 전부에 큰 공통 offset을 더하면 stable CE와 gradient가 유지돼야 한다. microbatch valid count를 1:9로 만들면 concat reference와 token-sum accumulation은 같고 mean-of-means는 달라야 한다.

실패가 발생했다는 사실보다 예상한 계층에서 감지됐는지가 중요하다. out-of-range ID가 loss kernel에서야 발견되면 앞 경계 validation이 약하다. zero-valid target이 NaN인 채 dashboard까지 흐르면 collator와 Trainer 사이 fail-fast가 없다. 주입점, 기대 증상, 최초 관측점, 통과 판정, 복구 뒤 재검증을 한 행으로 기록한다.

## 1.6 backward와 분산 reduction에서 전역 목적함수를 보존한다

한 rank의 올바른 gradient를 여러 rank로 확장할 때 loss mean과 DDP gradient average가 겹친다. numerator와 valid count를 먼저 전역화하고, backward scale과 zero-valid 처리까지 single-rank oracle과 맞춘다.

### DDP 평균 convention과 token mean을 맞춘다

rank r의 loss 합을 `S_r`, valid target 수를 `N_r`라 하자. 원하는 전역 목적은 `ΣS_r/ΣN_r`다. 각 rank가 `S_r/N_r`를 backward하고 DDP가 rank 평균을 내면 `(1/R)Σ∇S_r/N_r`이 되어 `N_r`가 다를 때 원하는 gradient와 다르다. local sum, global count, collective의 sum/average 의미를 함께 유도해야 한다. tensor-parallel group이 별도 division을 한다면 process group 범위도 붙인다.

assistant-only mask와 dynamic packing은 같은 `[B,T]`에서도 rank별 `N_r`를 크게 바꾼다. input tokens, attention-compute tokens, valid targets, unique source tokens를 따로 센다. throughput 분모와 optimizer objective 분모에 같은 `tokens` 이름을 쓰지 않는다.

dataset 길이가 accumulation steps로 나누어떨어지지 않으면 마지막 update의 microbatch 수가 적다. 고정 K로 나누면 마지막 gradient가 작아진다. sample을 반복해 K를 채우면 노출이 달라진다. `trainer.py:1729-1745`가 remainder와 현재 accumulation 수를 계산하는 이유다. resume에는 다음 sample IDs뿐 아니라 accumulation phase와 valid count가 필요하다.

### zero-valid local shard와 global batch를 구분한다

assistant-only SFT에서는 모든 labels가 ignore인 microbatch가 생길 수 있다. mean CE는 분모 0이고, 한 rank만 backward를 건너뛰면 collective hang이 날 수 있다. collator가 최소 유효 target을 보장하거나 모든 rank가 공동 skip 결정을 내려 optimizer·scheduler·UpdateID를 일관되게 처리해야 한다. graph에 연결된 zero를 만드는 것과 그 update를 유효 학습 step으로 세는 것은 별도 정책이다.

dashboard에는 skipped microbatch, skipped optimizer update, zero-valid sample 수를 나눈다. NaN 필터가 scalar를 이전 평균으로 바꾸어 표시해도 실제 gradient 문제를 고친 것은 아니다. 로깅의 완화와 학습 state 전이는 분리한다.

**loss 차이를 증거로 축소하는 실습**

Run A와 B가 다르면 config 전체를 훑기 전에 `CorpusRevision`, `TokenizerRevision`, `ModelRevision`, `GoldenBatchID`를 맞춘다. IDs·labels·attention mask·position IDs의 shape, dtype, checksum을 비교한다. 입력이 같으면 parameter와 alias checksum, logits probe, 위치별 numerator, valid count, scalar reduction, gradient probe 순으로 내려간다.

logits가 같고 scalar만 다르면 optimizer나 CUDA kernel을 의심할 이유가 없다. per-position loss까지 같고 mean만 다르면 분모다. 첫 block 전부터 다르면 loss 옵션을 파는 것은 시간 낭비다. 어느 경계까지 동일한지를 증명하는 일이 원인 이름을 추측하는 일보다 먼저다.

높은 loss 위치는 tokenizer offset으로 원 byte span까지 되돌린다. special token이면 template role과 삽입 규칙을 찾고, packed boundary면 이전 document attention과 target 연결을 본다. hotspot 비율에는 전체 domain별 supervised token 분모를 함께 둔다. 어떤 domain이 hotspot의 30%라는 말은 전체 target의 5%인지 60%인지에 따라 전혀 다르다.

최소 재현 묶음에는 tokenizer/model/config revision, ID·label·mask fixture, 필요한 원문 span locator, expected logits slice, numerator/count, dtype 정책, source/test anchor가 들어간다. framework bug라면 호출을 줄이되 collator bug를 조사하면서 collator를 없애지 않는다. 최소 재현은 줄 수가 가장 적은 코드가 아니라 문제를 일으키는 마지막 계약을 보존한 가장 작은 system이다.

**중간 계산을 검산하는 열두 질문**

원 문자열은 어느 revision에서 어떤 IDs가 되었는가. causal shift 소유자는 누구인가. attention mask와 label mask는 각각 무엇을 막는가. embedding과 head는 실제 alias인가. CE는 어느 dtype에서 계산되는가. numerator는 어느 targets의 합인가. denominator는 token·sequence·rank 중 무엇인가. 마지막 accumulation 묶음은 어떻게 scale되는가. collective는 sum인가 average인가. zero-valid batch 정책은 무엇인가. 높은 loss를 원문 span으로 되돌릴 수 있는가. resume 뒤 같은 sample과 accumulation phase를 증명할 수 있는가.

이 질문에 답하면 “다음 token을 맞힌다”가 tensor와 state machine으로 바뀐다. scalar가 내려간다는 사실은 shift, mask, 분모가 맞다는 증거가 아니다. 잘못된 목표도 일관되게 학습될 수 있다.

2장에는 `LossNumerator`, `ValidTargetCount`, normalized loss, loss-scale 적용 전후 값, microstep, accumulation target 수, alias ledger를 넘긴다. `ForCausalLMLoss` shift 소유권과 `num_items_in_batch` 산출식도 manifest에 고정한다. 2장의 첫 질문은 Adam 옵션이 아니라 어느 scalar가 어떤 scale로 backward에 들어왔는가이다.

### softmax CE gradient와 Hessian을 유도한다

### one-hot target이 모든 vocabulary row에 주는 gradient를 읽는다

한 위치의 logit을 $z$, softmax 확률을 $p_i=\exp(z_i)/\sum_j\exp(z_j)$, 정답 ID를 $y$라 두면 손실은 $\ell=-z_y+\log\sum_j\exp z_j$다. 이를 $z_k$로 미분하면 $\partial\ell/\partial z_k=p_k-\mathbf1[k=y]$다. 정답 좌표에는 $p_y-1$이, 나머지 모든 좌표에는 $p_k$가 흐른다. 그러므로 “정답 token 하나만 학습한다”는 설명은 틀렸다. 한 target이 vocabulary 전체의 상대 점수를 함께 움직인다. fused kernel은 dense Jacobian을 만들 필요 없이 softmax 결과의 정답 좌표에서 1을 빼 같은 vector-Jacobian product를 만든다.

gradient 성분의 합은 0이다. 모든 logit에 같은 상수를 더하는 방향에서는 확률과 손실이 변하지 않는다. 두 구현의 logits checksum이 달라도 각 위치에서 공통 상수 차이뿐이라면 목적함수는 같다. centered logits, log-probability, 정답 margin을 비교해야 하는 이유다. 반대로 scalar loss가 같아도 tail probability가 다른 분포일 수 있으므로 top-k만으로 동등성을 선언하지 않는다.

Hessian은 $H=\operatorname{diag}(p)-pp^\top$이다. $v^\top Hv$는 확률 $p$ 아래에서 $v$의 분산이므로 음수가 아니다. 한 위치의 CE는 logit에 대해 볼록하지만 공통 상수 방향에는 곡률이 없다. 이 사실이 신경망 parameter 전체의 목적함수를 볼록하게 만들지는 않는다. hidden state에서 logit으로 가는 mapping과 그 앞의 attention·MLP가 비선형이기 때문이다. logit 수준의 성질을 optimizer 전체의 수렴 보장으로 확장하면 계층을 건너뛴다.

네 class짜리 FP64 fixture로 식을 검증한다. 각 logit에 작은 $\epsilon$을 더하고 빼 finite difference를 구해 autograd 및 $p-onehot(y)$와 맞춘다. 모든 logit에 1000을 더해 naive exponential은 overflow시키고 stable log-sum-exp는 같은 값을 내는지 확인한다. 이 CPU 실험은 CUDA fused CE를 증명하지 않는다. 다만 수학 reference, framework 연산, backend kernel 중 최초 불일치를 구분하는 기준을 준다.

**smoothing·temperature·weight는 서로 다른 변경이다**

label smoothing은 one-hot target을 완만한 분포로 바꿔 gradient를 $p-q'$로 만든다. 균일 질량을 정답에도 배분하는지 정답 외 class에만 배분하는지는 구현 계약이다. ignore target은 위치를 numerator와 denominator에서 제외하는 정책이고 smoothing은 남은 위치의 class 분포를 바꾸는 정책이다. padding 위치가 smoothing 분포에 섞이면 class 축과 sequence 축을 혼동한 버그다.

temperature로 $z/T$를 쓰면 분포뿐 아니라 원래 logit에 대한 gradient에 $1/T$가 붙는다. distillation에서 $T^2$ 보정을 쓰는 이유와 추론 temperature를 구분한다. class weight는 위치별 손실 질량을 바꾸며 mean 분모가 valid 개수인지 weight 합인지 확인해야 한다. 옵션 이름 대신 unreduced loss, weight 적용 전후 numerator, 실제 denominator를 기록한다.

### Transformers forward에서 PyTorch backend까지 호출을 고정한다

## 1.7 data weighting과 accumulation이 만드는 학습 분포를 계산한다

목적함수는 CE 이름만으로 정해지지 않는다. sequence 길이, assistant mask, microbatch와 rank별 valid count가 각 token의 실질 weight를 만들므로 configured mixture와 realized objective measure를 함께 기록한다.

Transformers에서는 모델 `forward`가 hidden state로 logits를 만든 뒤 구성된 loss function을 호출한다. 검토 anchor는 revision과 함께 `src/transformers/loss/loss_utils.py`의 `ForCausalLMLoss`, `fixed_cross_entropy`, 각 `modeling_*.py`의 `forward`다. 파일명만 적지 않고 commit, symbol, signature, caller, line span을 묶는다. 모델별로 logits를 float로 올리는 시점과 `num_items_in_batch` 전달 여부가 달라질 수 있기 때문이다.

`ForCausalLMLoss`가 labels 오른쪽에 ignore 값을 pad하고 한 칸 slice하는 revision이라면 collator가 미리 shift해서는 안 된다. `[a,b,c,d]`에서 위치 `[a,b,c]`가 `[b,c,d]`를 맞히는 fixture를 만든다. BOS·EOS·assistant 경계와 packed document 경계에서도 반복한다. double shift는 shape가 맞고 loss도 내려갈 수 있어 smoke test로 잡히지 않는다.

`fixed_cross_entropy`가 `num_items_in_batch`를 받으면 sum을 그 값으로 나누는지 읽는다. 그 이름이 sample 수가 아니라 accumulation 묶음의 valid targets일 수 있다. Trainer 산출 경로, model이 loss kwargs를 수용하는지, custom model이 kwargs를 버리지 않는지 잇는다. custom loss가 local mean을 반환하고 Trainer가 다시 accumulation 수로 나누면 global token mean과 달라질 수 있다.

진단 probe는 model 반환 loss와 독립 reference를 맞춘다. logits의 마지막 위치를 제외하고 labels 첫 위치를 제외해 정렬한 뒤 `cross_entropy(reduction="none", ignore_index=-100)`를 계산한다. valid bitmap으로 numerator와 count를 따로 만든다. 차이가 나면 shift, mask, logits dtype, reduction 순서로 본다. probe는 `no_grad` 또는 detach로 분리해 학습 graph와 memory를 바꾸지 않는다.

**PyTorch 표면과 backend를 구분한다**

`torch.nn.functional.cross_entropy`는 의미의 진입점이지 항상 물질화되는 연산 목록은 아니다. 실제 dispatch는 dtype, device, layout, compile 상태에 따라 달라진다. source anchor는 `torch/nn/functional.py`의 함수, native 구현, CUDA dispatch로 층을 나눈다. Python 표현만 보고 중간 softmax tensor가 반드시 저장된다고 단정하지 않는다.

test matrix에는 class 수, contiguous/strided layout, FP32/BF16, ignore 일부/전부, none/sum/mean, smoothing, 큰 logit 차이를 둔다. CPU FP64 hand reference, CPU framework, CUDA eager, compiled/fused를 허용 오차와 비교한다. 실행하지 않은 backend 결과는 성공으로 적지 않는다. 전부 ignore인 mean이 NaN인지 graph-connected zero인지도 revision 계약으로 고정한다.

### token mean과 sequence mean의 sample weight를 비교한다

### 길이와 truncation이 objective measure를 바꾸는 경로를 잰다

sequence $s$의 target 수를 $n_s$, loss 합을 $S_s$라 하면 token mean은 $\sum_sS_s/\sum_sn_s$, sequence mean은 $(1/B)\sum_sS_s/n_s$다. 전자는 긴 sequence에 큰 질량을 주고 후자는 각 sequence에 같은 질량을 준다. packing row 평균은 document 평균도 아니다. 무엇이 옳은지는 목적에 달렸지만 서로 같은 옵션은 아니다.

assistant-only SFT에서는 긴 prompt와 짧은 answer가 흔하다. attention compute tokens와 supervised targets가 크게 다르다. tokens/sec가 늘면서 targets/sec가 줄 수도 있다. `InputTokenCount`, `AttentionTokenCount`, `ValidTargetCount`, `UniqueSourceTokenCount`를 분리한다. effective batch tokens라는 표현에는 어느 count인지 붙인다.

길이 bucket별 loss 합과 target 수, gradient norm contribution을 함께 본다. 긴 sample loss가 크다는 이유로 제거하면 정상적인 합산을 품질 문제로 오인한다. 같은 content의 길이만 바꾼 실험, token mean과 sequence mean의 비교, packing on/off 통제 실험에서 model·order·총 valid target을 고정한다.

### accumulation window의 loss sum과 count를 전역 합산한다

rank $r$, microstep $m$의 합을 $S_{rm}$, count를 $N_{rm}$라 하면 원하는 gradient는 $\nabla\sum S_{rm}/\sum N_{rm}$다. rank별 local mean을 backward한 뒤 평균내면 $N_r$가 다를 때 이 값이 아니다. numerator gradient를 합하고 global count로 scale하되 DDP reducer가 sum인지 average인지 대입해 world-size 보정을 결정한다.

로깅용 local mean과 backward용 global-normalized 값을 분리한다. count는 미분 대상이 아니지만 update 정의에 속한다. 한 rank가 zero-valid여도 동일한 collective 순서에 참여해야 한다. 전역 count가 0이면 optimizer, scheduler, scaler, UpdateID를 모든 rank에서 같은 정책으로 전이한다. 한 rank만 `continue`하면 다음 collective가 어긋나며, 이 hang의 원인은 NCCL보다 loss control flow에 있다.

두 rank fixture에서 rank 0에 target 하나, rank 1에 세 개를 둔다. 네 target hand gradient와 parameter delta를 비교한다. local-mean 평균이 실패하는 반례를 expected-failure로 남긴다. 덜 찬 마지막 accumulation, zero-valid rank, 전역 zero, `no_sync` 경계도 조합한다. rank별 bitmap과 collective sequence를 GoldenBatchID에 붙인다.

## 1.8 embedding·LM head·mask에서 최초 gradient 차이를 찾는다

logit gradient는 hidden과 projection weight를 거쳐 embedding으로 되돌아간다. tied weight에서는 lookup과 output projection 경로가 같은 parameter에 합쳐지므로 mask나 shift 오류를 sparse gradient 직관만으로 판단하지 않는다.

### tied embedding의 lookup·projection gradient를 합산한다

입력 embedding $E$와 출력 head $W$가 같은 storage를 공유하면 한 parameter는 lookup 경로와 projection 경로의 gradient를 함께 받는다. 출력 경로에서는 모든 vocabulary 행이 확률 오차의 영향을 받고, 입력 경로에서는 실제 lookup된 행에 hidden gradient가 scatter-add된다. 작은 model에서 두 경로를 각각 끊어 계산한 합과 실제 `.grad`를 비교한다.

`state_dict`에 key 두 개가 있다고 alias인 것은 아니다. data pointer, storage offset, shape, stride와 tie 함수 호출 시점을 본다. embedding resize, adapter 삽입, checkpoint load, compile 뒤 alias가 깨질 수 있다. optimizer group에 같은 storage가 중복되면 두 번 update될 수 있다. alias ledger는 parameter 이름보다 storage identity를 추적한다.

untied와 tied 비교에서는 같은 값으로 clone한 뒤 alias만 바꾼다. 한 update 뒤 입력 행, 출력 행, 전체 loss를 비교한다. 이는 tying의 우월성을 증명하는 실험이 아니라 gradient coupling의 구현을 밝히는 실험이다.

### ID→embedding→hidden→logits→gradient 순서로 이분 탐색한다

embedding output, block residual 경계, final norm, 선택 위치 hidden, vocabulary logits, centered logits, per-token loss 순으로 비교한다. 전체 tensor dump 대신 deterministic projection, norm, min/max, nonfinite count, 선택 slice를 쓴다. 첫 차이가 embedding이면 IDs·position·resize를, block 뒤면 mask·dropout·kernel을, logits에서면 head alias·dtype을, loss에서면 shift·ignore·reduction을 본다.

확률 artifact에는 정답 log-probability, top-k IDs/scores, log-sum-exp, entropy, margin, tail mass를 둔다. top-k가 같아도 tail 차이로 CE가 달라지고 scalar CE가 같아도 분포가 다를 수 있다. 어느 경계까지 같았는지가 원인 이름보다 강한 증거다.

**여덟 고장 주입과 2장 인계서**

**detector가 올바른 경계에서 울리는가**

labels를 미리 shift해 double shift를 만들면 causal fixture가 실패해야 한다. padding 하나를 ignore하지 않으면 valid count와 원문 span detector가 잡아야 한다. assistant mask를 뒤집으면 supervised-role histogram이 막아야 한다. 마지막 accumulation을 고정 K로 나누면 hand delta test가 실패해야 한다.

rank target 비율을 1:3으로 하고 local mean을 평균내면 global-denominator test가 실패해야 한다. all-ignore batch에서는 공동 skip protocol이 작동해야 한다. load 뒤 head tie를 끊으면 alias ledger가 첫 forward 전에 막아야 한다. 큰 공통 logit 상수를 더하면 stable CE invariant가 유지되어야 한다. 각 기록에는 injection diff, 기대 invariant, 최초 detector, 실제 증상, 복구 revision, 재검증 ID를 둔다.

**loss manifest를 상태기계에 넘긴다**

인계서는 먼저 어느 입력과 구현을 재현해야 하는지 고정한다. 이를 위해 corpus·tokenizer·template·model revision과 `GoldenBatchID`를 기록한다. 입력 경계에는 IDs·labels·mask·position의 shape, dtype, checksum을, 목적함수 경계에는 shift owner, ignore ID, smoothing·weight·temperature 정책을 적는다. 출력 경계에는 logits dtype, numerator, valid count와 global normalization 공식을 남긴다. accumulation phase와 process-group semantics까지 연결해야 다음 사람이 loss 차이를 입력, 목적함수, 분산 reduction 가운데 한 경계로 좁힐 수 있다.

embedding/head alias ledger, per-position fixture, source/test anchor, zero-valid 정책, skipped-update 의미도 기록한다. 완료 증거는 hand CE 일치, special boundary shift, uneven-rank global delta, all-ignore 공동 전이, tied gradient 합, 원문 byte span 역추적이다. 이 여섯 증거가 없으면 loss graph가 내려간다는 사실만으로 optimizer 단계에 넘기지 않는다.

**logit이 만들어지는 마지막 선형층을 수치로 읽는다**

**행렬 곱의 각 축에는 구체적인 의미가 있다**

마지막 hidden tensor가 `[B,T,H]`, vocabulary weight가 `[V,H]`이면 logits는 대체로 `[B,T,H] @ [H,V]`다. 위치 $(b,t)$와 token $v$의 점수는 hidden vector와 vocabulary row의 내적이며 bias가 있으면 이를 더한다. 이 점수는 확률이 아니라 방향 정렬과 크기를 함께 담은 비정규화 좌표다. hidden norm이나 weight norm이 커지면 cosine 방향이 같아도 margin이 커지고 entropy가 줄 수 있다. 따라서 “embedding과 가까운 단어”라는 직관에는 norm과 normalization 여부를 붙여야 한다.

한 logit의 parameter 미분은 단순하다. 출력 row $w_v$에는 $(p_v-1[v=y])h$가 흐르고 hidden에는 $W^\top(p-onehot(y))$가 흐른다. 이 식으로 작은 행렬의 한 update를 손으로 계산한다. framework 결과와 row별 gradient를 맞춘 뒤 tied input 경로를 더한다. vocab row norm, hidden norm, cosine, logit, probability를 나란히 기록하면 어느 요소가 확신을 만들었는지 분해할 수 있다.

tensor parallel vocabulary sharding에서는 각 rank가 logits 일부만 갖는다. stable softmax에는 전역 maximum과 전역 exponential sum이 필요하고 정답 logit의 소유 rank도 찾아야 한다. local maximum만 쓰면 rank partition에 따라 확률이 달라진다. 구현은 collective로 global max와 sum을 만들거나 특화된 parallel CE를 쓴다. 검증 fixture는 같은 logits를 shard 수 1·2·4로 나누어 loss와 gradient가 같은지 본다. shard 경계에 정답 ID를 놓고 out-of-range local indexing을 주입한다.

vocabulary 확장 뒤 head row 수와 tokenizer vocabulary 크기가 다르면 어떤 ID는 projection에 없거나 사용되지 않는 row가 된다. resize가 initialization과 tie를 어떻게 처리했는지 확인한다. special token 추가 전후 `len(tokenizer)`, config vocab size, embedding rows, head rows, checkpoint tensor shape를 manifest에 둔다. 평균 loss가 정상이어도 새 special token이 한 번도 target이 아니면 그 row는 의도대로 학습되지 않는다.

**확률과 calibration을 loss와 혼동하지 않는다**

낮은 NLL은 정답에 높은 확률을 주었다는 뜻이지만 생성 품질 전부를 뜻하지 않는다. calibration은 예측 확률과 실제 적중 빈도의 대응을 묻는다. teacher forcing loss는 실제 이전 정답 token 조건에서 측정되고 generation은 모델이 낸 token을 다시 조건으로 넣는다. exposure 경로가 다르므로 validation CE 개선이 장기 생성 오류를 항상 줄이지 않는다.

calibration 진단은 confidence bin의 accuracy만으로 끝내지 않는다. language/domain/position/sequence length/special boundary별로 나누고 각 bin의 count를 둔다. vocabulary가 큰 causal model의 top-1 accuracy는 제한적이므로 정답 probability, rank, entropy, margin, Brier-like statistic을 목적에 맞게 본다. temperature scaling은 validation 분포에서 calibration을 바꿀 뿐 base model 지식을 추가하지 않는다.

### attention·causal·label mask의 서로 다른 그래프를 시험한다

**attention mask, causal mask, label mask**

attention mask는 어떤 key/value 위치를 읽을 수 있는지, causal mask는 미래 위치를 막는지, label mask는 어느 위치가 손실에 기여하는지를 정한다. label을 `-100`으로 바꿔도 해당 prompt token은 attention context로 사용될 수 있다. 반대로 attention에서 padding을 막았다고 그 위치가 자동으로 loss에서 제외되는 것은 아니다. 세 mask가 같은 shape를 가질 수 있어도 역할은 다르다.

packed samples에서는 document boundary attention 정책이 추가된다. 단순 causal mask만 쓰면 뒤 문서가 앞 문서를 읽는다. 어떤 pretraining recipe는 이를 허용하고 어떤 recipe는 block-diagonal mask를 쓴다. boundary 다음 첫 token의 target이 무엇인지도 정해야 한다. EOS를 예측하는지, 다음 BOS를 예측하는지, 경계 loss를 지우는지에 따라 numerator가 달라진다. 정책을 “packing=true” 한 비트로 표현할 수 없다.

left padding과 right padding은 position ID와 last-token selection을 바꿀 수 있다. training에서 right padding, generation에서 left padding을 쓴다고 해도 shared collator가 어느 mode인지 확인한다. padding token과 EOS가 같은 ID인 model에서는 ID만 보고 실제 EOS까지 mask하면 문장 종료 학습이 사라진다. attention mask나 sequence length를 이용해 실제 padding 위치를 구분한다.

mask fixture는 색칠된 `[B,T]` 표를 artifact로 둔다. 각 cell에 input ID, role, document, attention availability, target ID, valid 여부, loss를 붙인다. 합계 valid count와 scalar가 표에서 직접 계산되어야 한다. template 변경 뒤 이 표의 diff가 예상한 role boundary에만 나타나는지 snapshot test한다.

**perplexity를 비교 가능한 단위로 되돌린다**

**지수 하나가 숨기는 분모와 tokenizer**

perplexity는 평균 NLL의 지수이므로 평균의 분모와 로그 밑을 상속한다. tokenization이 다르면 한 token의 정보량이 달라져 숫자를 직접 비교하기 어렵다. 같은 문자열도 tokenizer A는 두 target, B는 다섯 target을 만들 수 있다. token PPL만 보고 tokenizer A가 우월하다고 하면 단위가 바뀐 측정기를 비교한 셈이다.

가능하면 원 byte나 character 기준 bits-per-byte, bits-per-character를 함께 보고한다. 그러나 normalization으로 원문 byte가 바뀌면 interval map이 필요하다. multilingual corpus에서는 script별 fertility와 byte coverage를 나눈다. code·수식·한글에서 tokenizer가 만드는 길이 차이가 attention compute와 loss denominator를 동시에 바꾼다.

evaluation overlap과 document stride도 PPL에 영향을 준다. 긴 문서를 context window로 자를 때 각 window의 처음 token은 짧은 context를 갖는다. sliding window에서 겹친 target을 중복 계산하는지, context로만 사용하고 target에서 제외하는지 고정한다. BOS 삽입과 문서 연결 정책도 training과 같아야 한다. 수치에는 corpus revision, tokenizer, normalization, context, stride, mask, denominator를 붙인다.

**rare token과 높은 loss를 조사하는 법**

**빈도만으로 난도를 설명하지 않는다**

rare token은 관측이 적어 어려울 수 있지만 context가 결정적이면 loss가 낮을 수도 있다. 흔한 token도 여러 의미와 문법 역할을 가지면 entropy가 높다. token frequency, conditional entropy, position, domain, preceding context length를 분리한다. 평균 loss 상위 token 목록만 만들면 긴 subword나 손상된 text가 과대표집될 수 있다.

high-loss dossier는 token ID를 decoded 문자열로만 보여주지 않는다. tokenizer byte span, 원 normalization 이전 span, 주변 tokens, role와 document boundary, model top alternatives, entropy, margin, corpus frequency를 함께 둔다. replacement character, mojibake, HTML residue, code indentation, math control sequence가 모이면 데이터 제조 단계로 되돌린다. 정상적인 모호성이면 필터로 제거할 이유가 없다.

gradient 영향은 loss 순위와 다르다. 이미 확신한 오답은 큰 gradient를 만들 수 있지만 hidden norm과 parameter Jacobian이 최종 norm을 바꾼다. per-example gradient를 전부 저장하기 어렵다면 selected layer projection이나 influence proxy를 사용하되 근사임을 표시한다. 높은 loss를 곧바로 “나쁜 데이터”라 부르지 않는 이유다.

**custom loss를 넣기 전 지켜야 할 경계**

**focal·unlikelihood·auxiliary loss의 소유권**

focal 계열은 쉬운 예제의 질량을 줄이고 어려운 예제를 강조하지만 language modeling의 확률 추정과 calibration을 바꾼다. unlikelihood는 금지 token이나 반복에 음의 목적을 더한다. router balance, contrastive, multimodal alignment 같은 auxiliary loss도 서로 다른 denominator와 scale을 갖는다. 합산 scalar 하나만 기록하면 어느 항이 update를 지배했는지 알 수 없다.

각 loss 항마다 numerator, denominator, coefficient, valid count, gradient norm을 기록한다. coefficient가 같아도 raw scale이 다르면 영향이 다르다. gradient cosine을 selected parameters에서 측정하면 항들이 협력하는지 충돌하는지 볼 수 있다. auxiliary coefficient schedule이 있다면 optimizer scheduler와 별도 state로 checkpoint한다.

custom autograd function은 forward 값만 맞아서는 부족하다. gradcheck, gradgradcheck가 필요한지, noncontiguous input, mixed dtype, empty selection, distributed reduction을 test한다. saved tensor memory와 in-place mutation도 본다. compile path에서 graph break 또는 다른 kernel이 선택되는지 확인한다. reference implementation은 느려도 명료하게 유지해 fused 구현의 oracle로 쓴다.

**실험 설계는 손실 변경과 데이터 변경을 격리한다**

loss A/B에서는 batch IDs, order, tokenizer, model initialization, total valid targets, optimizer update count를 맞춘다. weighted loss가 effective mass를 바꾸면 단순 learning-rate 동일 비교가 공정한지 논의한다. evaluation은 base CE뿐 아니라 변경 목적의 metric과 부작용 metric을 함께 둔다. 예컨대 반복 억제 손실은 repetition 외 calibration과 rare-token recall도 본다.

작은 proxy에서 성공해도 큰 model로 바로 일반화하지 않는다. scale에 따라 gradient noise, capacity, data repetition이 달라진다. pilot은 구현 오류와 큰 역효과를 제거하는 단계이지 최종 효과 증명이 아니다. 각 결론에는 model size, tokens, seed, confidence interval, stopped runs를 붙인다.

## 1.9 reproducibility와 observability를 loss contract로 만든다

같은 checkpoint에서 이어졌다는 주장은 첫 batch의 IDs·mask·logits·numerator·denominator가 맞아야 성립한다. 학습 중에는 평균 loss 하나 대신 목적함수를 구성한 상태와 throughput의 원인을 분리해 관측한다.

### resume 직후 첫 batch를 pre-step oracle로 사용한다

중단 없는 Run A와 checkpoint에서 재개한 Run B의 첫 다음 batch를 비교한다. sample IDs, token tensors, mask, model mode, RNG, logits probe, numerator, count가 같아야 한다. 이 지점에서 loss가 다르면 optimizer 이후를 볼 필요가 없다. input부터 다르면 sampler cursor나 async preprocessing, logits부터 다르면 parameter·RNG·kernel, scalar만 다르면 reduction state다.

dropout RNG는 global seed 하나로 충분하지 않을 수 있다. CPU/CUDA generator와 data worker, model-parallel RNG tracker를 기록한다. activation checkpointing은 recomputation에서 같은 random mask를 재현해야 한다. deterministic flag는 가능한 연산을 제한할 뿐 데이터 cursor와 callback side effect를 복원하지 않는다.

bitwise 동일성이 필요하지 않은 backend에서는 허용 오차와 비교 지점을 사전에 정한다. scalar가 가까운 것만 보지 않고 첫 두 update의 parameter projection과 optimizer state까지 이어 본다. 작은 차이가 expected numerical drift인지 control-flow divergence인지 시간에 따른 성장률로 분류한다.

**독자가 직접 수행하는 계층별 실습**

**실습 A: 손으로 계산한 네 token CE**

고정 logits와 labels를 JSON fixture로 둔다. stable log-sum-exp, per-token loss, numerator, count, gradient, Hessian-vector product를 FP64로 계산한다. ignore 위치 하나를 추가해 numerator가 그대로이고 count만 예상대로 바뀌는지 본다. smoothing과 class weight는 별도 fixture로 만들어 의미를 섞지 않는다.

**실습 B: causal shift를 원문까지 추적한다**

짧은 한국어 문장과 role boundary를 tokenize하고 각 input 위치가 예측하는 다음 ID를 표로 만든다. tokenizer offset으로 target을 원 byte span에 연결한다. BOS·EOS·padding·packed boundary를 하나씩 넣는다. collator shift와 model shift 중 하나만 남기고 double-shift 고장을 주입한다.

**실습 C: 불균등 rank의 global mean**

동일한 작은 linear head를 두 logical rank로 나누고 valid targets를 1개와 3개로 배치한다. single-process 네-target reference, local-mean average, corrected global normalization의 gradient를 비교한다. 마지막에는 rank 하나를 all-ignore로 바꿔 collective 순서와 skip state를 점검한다.

**실습 D: tensor atlas로 최초 차이 찾기**

Run B에서 embedding row 하나만 바꾸고 atlas가 embedding 경계에서 처음 실패하는지 본다. 다음에는 loss denominator만 바꿔 logits까지 같고 scalar에서 갈라지는지 본다. detector가 예상 경계보다 늦으면 계측 자체가 결함이다. 결과는 `GoldenBatchID`, fixture checksum, 최초 mismatch path, source revision과 함께 인계한다.

이 실습들의 목적은 특정 숫자를 외우는 것이 아니다. 문자열에서 target, logit, 확률, numerator, denominator, gradient, distributed update로 이어지는 한 줄의 계보를 직접 복원하는 데 있다. 이 계보가 닫히면 loss spike나 재개 divergence를 추측이 아니라 최초로 깨진 계약에서 조사할 수 있다.

**운영에서 loss를 읽는 관측 계약**

### loss sum·valid count·domain·position을 함께 관측한다

train loss에는 aggregation window와 지연 시간이 있다. microbatch scalar를 단순 평균하면 valid target 수가 다른 batch가 같은 무게를 갖는다. 관측 시스템에는 window 안의 numerator 합과 denominator 합을 보내 비율을 계산한다. rank별 local loss를 모두 같은 시계열에 넣어 다시 평균하는 실수도 막는다. metric 이름에 `token_mean`, `sequence_mean`, `local`, `global`을 구분하고 unit을 명시한다.

평균과 함께 p50·p95 per-token loss, nonfinite count, zero-valid count, supervised target ratio, entropy, 정답 rank를 제한된 label로 집계한다. domain·language·role·length bucket은 cardinality를 통제한다. document ID나 URL을 metric label로 넣지 않고 높은 loss 표본은 접근 통제된 artifact에 correlation ID로 연결한다. privacy를 지키면서 aggregate에서 원인 표본으로 내려가는 경로를 만드는 방식이다.

loss spike alert는 절대 threshold 하나보다 baseline 대비 변화, 지속 시간, denominator를 함께 본다. valid count가 급락해 noisy mean이 된 상황과 실제 numerator가 증가한 상황은 대응이 다르다. data mixture 전환, context length 전환, learning-rate warmup 끝처럼 예정된 state change를 annotation으로 남긴다. 예정되었다는 이유로 alert를 지우지 않고 기대 범위와 비교한다.

### throughput 변화와 objective drift를 별도 gate로 판정한다

tokens/sec가 좋아졌는데 loss 곡선의 update당 진전이 느려졌다면 먼저 분모를 확인한다. 더 많은 padding을 세었거나 supervised ratio가 낮아졌을 수 있다. seconds, optimizer updates, input tokens, valid targets, estimated FLOPs를 각각 x축으로 그린다. 어느 자원을 고정했을 때 품질이 달라지는지에 따라 loader, packing, optimizer 문제를 나눈다.

kernel 변경의 A/B에서는 logits와 loss 허용 오차뿐 아니라 같은 valid targets까지 걸린 시간, peak memory, compilation overhead를 본다. 첫 몇 step의 compile 비용을 steady-state 처리량에 섞지 않는다. numerical drift가 허용 범위여도 특정 token이나 길이에서만 커지는지 stratified fixture로 검사한다. 빠르다는 이유로 목적함수 계약의 변화를 받아들이지 않는다.

**흔한 설명의 함정을 반례로 고친다**

**“loss 0이면 완벽하다”는 문장**

유한 logit에서 정답 확률이 정확히 1이 되지 않으므로 CE 0은 극한적 표현이다. 표시 자릿수 반올림, underflow, ignore-only batch가 0처럼 보일 수 있다. training data memorization으로 매우 낮아져도 unseen distribution의 품질을 보장하지 않는다. 0 근처 값을 보면 numerator·count·dtype·표시 precision과 split integrity를 확인한다.

**“batch를 키우면 같은 gradient다”는 문장**

같은 samples의 loss sum을 같은 전역 count로 정규화하고 model stochasticity와 optimizer state 전이를 맞춘 경우에만 gradient가 대응한다. dropout mask, batch-dependent layer, clipping 시점, accumulation rounding, DDP reduction, 마지막 remainder가 다르면 달라진다. batch equivalence test는 sample multiset뿐 아니라 순서와 RNG, valid target 수, update boundary를 고정한다.

**“perplexity가 낮으면 tokenizer가 좋다”는 문장**

token 단위가 바뀌었으므로 tokenizer 사이 raw PPL 비교는 대개 단위가 다르다. bytes-per-token이 큰 tokenizer는 token 수를 줄여 PPL 숫자를 유리하게 보이게 할 수 있다. bits-per-byte와 downstream compute, fertility, rare script coverage를 함께 본다. normalization 손실이나 unknown-byte fallback도 확인한다.

**“DDP가 평균을 알아서 맞춘다”는 문장**

DDP가 평균내는 것은 rank별 gradient이지 서로 다른 local denominator를 되돌리는 일이 아니다. objective normalization은 model/trainer의 책임이고 collective semantics는 그 식에 들어가는 요소다. uneven targets 반례를 손으로 계산하면 이 차이가 즉시 드러난다. framework 버전과 custom communication hook이 reduction 의미를 바꿀 수 있어 source와 test를 고정한다.

**장 종료 검토표**

**수학·코드·데이터·분산을 한 행으로 닫는다**

목적함수 식에는 target 분포, ignore 정책, numerator, denominator가 적혀 있어야 한다. 코드에는 shift owner부터 model forward, loss utility, framework CE, backend까지 revision anchor가 있어야 한다. 데이터에는 원문 byte에서 normalized span, token ID, role, packed 위치, target 위치로 이어지는 좌표가 있어야 한다. 분산에는 rank와 accumulation을 가로지르는 count, reducer 의미, zero-valid 전이가 있어야 한다.

재현 묶음에는 GoldenBatchID, tensor checksums, expected per-token loss, global count, gradient projection, alias ledger가 있다. 디버깅 묶음에는 tensor atlas와 최초 불일치 규칙이 있다. 관측 묶음에는 numerator/count 기반 aggregate와 sample dossier 연결이 있다. 인계 묶음에는 2장이 loss scale 이전 scalar부터 optimizer update까지 이어받을 상태가 있다.

최종 질문은 “loss가 얼마인가”가 아니다. 그 숫자를 구성한 각 target을 원문까지 되돌릴 수 있는가, 같은 target 집합이 rank 배치와 accumulation 경계가 달라도 같은 gradient를 만드는가, source revision을 바꿨을 때 최초 변화 지점을 test가 잡는가를 묻는다. 셋 모두 증거로 답할 수 있을 때 다음 token 예측은 구호가 아니라 검증 가능한 시스템 계약이 된다.

**마지막 반증 실험**

완료를 선언하기 전에 의도적으로 정상처럼 보이는 잘못된 run을 만든다. labels를 두 칸 이동하되 vocabulary와 shape는 유지하고, padding이 많은 rank와 target이 많은 rank를 섞으며, dashboard에는 local mean만 보낸다. 표면적으로 loss가 매끄럽게 감소해도 causal fixture, global denominator reference, numerator/count metric이 각각 다른 층에서 실패해야 한다. 세 detector 중 하나라도 침묵하면 본문의 설명은 운영 가능한 계약으로 아직 번역되지 않은 것이다.

복구 뒤에는 설정만 되돌리고 끝내지 않는다. 같은 GoldenBatchID로 원문 span, IDs, target bitmap, logits probe, per-token CE, global gradient projection을 재계산한다. 실패 run과 복구 run의 artifact를 나란히 두고 최초 불일치가 사라졌는지 확인한다. 수정 이후 다른 boundary에서 새로운 차이가 생기지 않았는지도 본다.

이 장의 산출물은 지식을 요약한 문장이 아니라 재현 가능한 loss dossier다. 독자는 그 dossier를 2장의 update state machine에 넣어 scale, unscale, clipping, optimizer moments가 동일한 목적함수를 실제 parameter 변화로 옮기는지 이어서 검증한다.

검토자는 dossier에서 임의의 target 하나를 골라 역방향으로도 걸어 본다. scalar에서 해당 위치의 가중치와 denominator를 찾고, label ID와 shift 전 위치, tokenizer interval, normalized span, raw checksum까지 도달해야 한다. 다시 정방향으로 걸었을 때 같은 logit slice와 loss contribution이 나와야 한다. 이 왕복 검사가 끊기면 계보의 어느 edge가 추정에 불과한지 표시한다.

마지막으로 revision을 하나 바꾼 child run을 만든다. tokenizer만 바꿨다면 IDs 이후가, reduction만 바꿨다면 scalar 이후가 달라져야 한다. 예상보다 앞선 경계가 바뀌면 실험 통제가 실패한 것이고 예상보다 뒤에서도 차이가 없으면 옵션이 실제 호출 경로에 연결되지 않았을 가능성이 있다. 이렇게 변화의 영향 반경까지 증명해야 옵션 설명이 실제 코드의 상태 변화와 맞물린다.

이 증거 묶음에는 작성 시각보다 source revision과 fixture checksum을 우선 식별자로 쓴다. 시간이 같아도 입력이 다르면 같은 실험이 아니며, 시간이 달라도 revision과 입력, 상태가 같으면 비교 가능한 재현이다. 독자는 이 원칙을 이후 모든 학습 단계의 인계 규칙으로 재사용한다.

## 1.10 next-token objective를 확률모형 전체에서 다시 유도한다

앞에서 본 개별 tensor를 joint likelihood의 factorization으로 다시 묶는다. teacher forcing이 제공하는 조건부 표본, empirical data measure와 softmax CE가 어떻게 하나의 scalar objective가 되는지 유도한다.

**문장 확률에서 한 위치의 손실까지 내려간다**

길이 `T`인 토큰열 `x_0,…,x_{T-1}`의 결합확률은 조건부확률의 연쇄법칙으로 `p(x_0:T)=∏_t p(x_t|x_<t)`라고 쓸 수 있다. 이 식은 Transformer만의 성질이 아니다. 어떤 결합분포도 조건부 분포의 곱으로 분해할 수 있다. causal language model이 선택한 것은 각 조건부 분포를 동일한 parameter 집합으로 근사하고, 미래 token을 보지 못하도록 계산 그래프를 제한하는 방식이다. 따라서 “다음 토큰 예측”은 문장 전체를 이해하지 않는다는 뜻이 아니라, 문장 전체의 likelihood를 계산 가능한 지역 항의 합으로 바꾸는 factorization이다.

곱은 긴 sequence에서 수치적으로 불안정하고 미분하기도 불편하다. 로그를 취하면 `log p(x_0:T)=Σ_t log p(x_t|x_<t)`가 되고, 최대화 대신 음수를 최소화하면 token별 negative log-likelihood 합이 된다. 학습 sample에 문서 경계, padding, instruction role mask가 들어오면 모든 위치가 목적함수에 참여하지 않는다. 유효 집합을 `M`이라 할 때 실제 scalar는 흔히 `L=(Σ_{i∈M} w_i[-log p(y_i|c_i)])/(Σ_{i∈M} w_i)`다. 여기서 `i`는 batch와 position을 함께 편 인덱스이고, `c_i`는 causal context, `y_i`는 target, `w_i`는 선택적 가중치다. 코드에서 `reduction="mean"` 한 단어가 이 분모를 완전히 설명하지 못하는 이유가 여기에 있다.

독자는 이 식의 각 기호를 tensor와 데이터 계보에 붙여야 한다. `c_i`는 단순한 prefix 문자열이 아니라 normalization·tokenization·template·packing·causal mask를 거친 입력 좌표다. `y_i`는 label shift 뒤의 ID이고 `M`은 ignore index와 packed boundary 정책이 만든 bitmap이다. `w_i`는 class weight, sample weight, curriculum weight, importance weight 가운데 무엇인지 구분해야 한다. 분자는 loss kernel이 계산해도 분모의 의미는 collator와 trainer가 함께 결정한다.

sequence likelihood를 비교할 때도 길이 정규화를 명시한다. 합 NLL은 긴 sequence를 더 나쁘게 보이게 하고 token mean은 각 token을 같은 단위로 본다. sequence mean은 각 sequence를 같은 무게로 보므로 짧은 예제의 token이 상대적으로 무거워진다. instruction tuning에서 긴 assistant 답변과 짧은 답변이 섞이면 이 선택은 곧 학습 분포 선택이다. “평균 loss”라는 이름으로 세 목적함수를 교환해서는 안 된다.

**최대우도와 교차엔트로피가 만나는 지점을 구분한다**

경험 데이터 분포를 `q`, 모델 분포를 `p_θ`라 하면 cross entropy는 `H(q,p_θ)=-E_q log p_θ`다. `H(q,p_θ)=H(q)+KL(q||p_θ)`이고 데이터가 고정되면 `H(q)`는 θ와 무관하므로 cross entropy 최소화는 forward KL 최소화와 같다. 그러나 finite corpus에서 우리가 가진 것은 q 자체가 아니라 sample 평균이다. 중복 제거, mixture weight, curriculum, filtering이 바뀌면 empirical q가 바뀐다. 같은 코드와 optimizer여도 다른 목적함수를 푸는 셈이다.

one-hot target에서는 cross entropy가 정답 log-probability의 음수와 같지만 label smoothing, distillation, soft target에서는 target 분포 전체가 들어간다. binary cross entropy와 categorical cross entropy도 교환할 수 없다. vocabulary에서 단 하나의 class가 정답인 causal LM은 categorical distribution을 쓴다. 각 vocabulary 항목을 독립 Bernoulli로 취급하는 BCE는 확률 합이 1이라는 경쟁 구조를 잃는다.

forward KL이라는 표현도 과장하지 않는다. corpus에 없는 context에서 q는 관측되지 않는다. maximum likelihood는 관측 support 위에서 model probability를 높인다. hallucination, calibration, preference alignment가 자동으로 해결되는 것은 아니다. 이 장의 식이 보장하는 것은 지정된 empirical objective에 대한 gradient이며, 세계 지식의 참이나 인간 선호의 총체가 아니다.

### softmax와 logsumexp를 안정된 수치 알고리즘으로 쓴다

**log-sum-exp는 단순한 공식 변형이 아니다**

한 위치의 logit vector를 `z∈R^V`라 하자. 정답 `y`의 NLL은 `ℓ=-z_y+log Σ_j exp(z_j)`다. `z_j`가 1000이면 FP32에서도 `exp(z_j)`는 overflow한다. `m=max_j z_j`를 빼면 `ℓ=-z_y+m+log Σ_j exp(z_j-m)`이 되고 모든 지수의 입력은 0 이하가 된다. softmax가 공통 상수 이동에 불변이므로 함수값은 같다. 이 변형은 대수 장식이 아니라 overflow를 막는 구현 계약이다.

vocabulary parallelism에서는 `m`과 지수합이 rank에 나뉜다. 각 rank가 local maximum `m_r`을 구한 뒤 global maximum `m=max_r m_r`을 collective로 얻는다. local sum `s_r=Σ_{j∈r} exp(z_j-m)`을 계산하고 global sum `s=Σ_r s_r`을 다시 reduce한다. 정답 logit도 target ID를 소유한 rank에서 골라 합친다. local softmax를 먼저 만들고 확률을 평균하면 전역 vocabulary softmax가 아니다. 두 collective의 순서와 dtype, target owner가 목적함수의 일부가 된다.

online log-sum-exp는 vocabulary tile을 순회하며 `(m,s)` 상태를 갱신한다. 새 tile maximum `m'`이 더 크면 이전 합을 `exp(m-m')`만큼 재조정한다. 이 결합 연산은 tile 순서가 달라도 실수 산술에서는 같지만 부동소수점에서는 reduction 순서에 따른 작은 차이가 생긴다. fused linear-cross-entropy가 전체 `[N,V]` logits를 materialize하지 않아도 되는 근거다. 다만 target logit, global maximum, global exponential sum, backward에 필요한 확률 또는 재계산 정보는 보존해야 한다.

수치 검증은 scalar tolerance 하나로 끝내지 않는다. logit offset을 `+10^3`, `-10^3`로 바꾸어 loss와 gradient가 유지되는지 본다. vocabulary를 여러 tile로 쪼개도 FP64 reference와 맞는지 본다. 정답이 첫 tile·마지막 tile·rank 경계에 있을 때를 각각 시험한다. 모든 logits가 같은 경우 loss는 `log V`, gradient는 정답에서 `1/V-1`, 나머지에서 `1/V`여야 한다. 이 대칭 fixture는 off-by-one과 target owner 오류를 강하게 드러낸다.

**dtype 전이는 어느 시점에 일어나는가**

projection은 BF16이나 FP16으로 수행하고 CE는 logits를 FP32로 올릴 수 있다. 그렇다고 저정밀 projection에서 이미 생긴 overflow나 rounding이 사라지는 것은 아니다. hidden과 weight의 matmul accumulator dtype, output dtype, logits cast, logsumexp accumulator를 각각 기록한다. “loss는 FP32”라는 문장 하나로 전체 경로의 안정성을 보증할 수 없다.

BF16은 FP16보다 exponent 범위가 넓지만 mantissa가 짧다. FP16은 작은 gradient underflow와 큰 activation overflow에 더 취약하다. TF32는 Ampere 이후 일부 FP32 matmul의 내부 precision 선택이며 CE의 scalar dtype과 같은 개념이 아니다. AMP autocast 정책이 projection과 loss에 어떤 dtype을 선택했는지는 실행 환경과 framework revision에 따라 확인해야 한다.

non-finite가 나오면 loss 함수부터 의심하지 않는다. hidden state, LM head weight, matmul output, cast 뒤 logits, row maximum, exponential sum, target logit 순으로 최초 비정상 tensor를 찾는다. loss kernel이 stable해도 입력 logits가 이미 `inf`와 `-inf`를 함께 포함하면 뺄셈에서 NaN이 생길 수 있다. detector는 원인을 만든 연산의 바로 다음 경계에 있어야 한다.

### CE gradient와 curvature를 simplex geometry로 읽는다

**gradient는 확률 질량을 옮기는 방향이다**

one-hot target `e_y`에 대해 `∂ℓ/∂z=p-e_y`다. 정답 좌표에는 `p_y-1≤0`, 오답 좌표에는 `p_j≥0`이 들어간다. gradient descent는 정답 logit을 올리고 오답 logits를 내린다. 모든 성분의 합은 0이므로 공통 상수 방향으로는 움직이지 않는다. 이는 softmax의 gauge freedom과 정확히 대응한다.

gradient의 크기는 단순히 “틀린 정도”가 아니다. 정답 확률이 거의 0이면 정답 성분은 약 -1이고 오답 질량이 분산된다. 이미 확신하는 정답이면 전체 gradient가 작다. label smoothing target `q`에서는 `p-q`가 되어 정답에서도 무한히 큰 margin을 요구하지 않는다. class weight나 token weight는 이 vector 전체를 배율한다. denominator 변경도 모든 기여의 배율을 바꾸지만 batch 구성에 따라 상대 weight까지 바꿀 수 있다.

hidden state `h∈R^C`, LM head `W∈R^{V×C}`에서 `z=Wh+b`이면 `∂ℓ/∂h=Wᵀ(p-e_y)`, `∂ℓ/∂W=(p-e_y)hᵀ`다. 한 token의 gradient는 vocab 방향 오차와 hidden 방향의 외적이다. tied embedding이면 이 LM-head 경로가 input lookup에서 온 sparse row gradient와 같은 storage에 합쳐진다. optimizer가 둘을 별도 parameter로 등록하면 중복 step, weight decay 중복, checkpoint alias 파괴가 생길 수 있다.

Hessian은 `H_z=diag(p)-ppᵀ`다. 이는 categorical one-hot vector의 covariance와 같다. 임의 vector `v`에 대해 `vᵀH_zv=Var_{j∼p}(v_j)≥0`이므로 logit 공간에서 positive semidefinite다. 모든 원소가 같은 `v`에서는 분산이 0이고 공통 이동 방향이 nullspace다. 확률이 한 class에 몰리면 covariance와 곡률도 작아져 saturated 영역이 된다. 하지만 network parameter θ에 대한 Hessian은 `JᵀH_zJ` 외에 logits의 θ에 대한 2차 미분 항이 있어 전역 convex가 아니다.

Fisher information과 Gauss–Newton 근사도 이 구조를 이용한다. 모델 자체에서 label을 sample한다고 볼 때 score covariance가 softmax covariance를 만든다. 다만 empirical Fisher, true Fisher, generalized Gauss–Newton을 같은 이름으로 부르면 optimizer 설명이 흐려진다. 어떤 target과 expectation, 어떤 Jacobian을 사용했는지 적어야 한다. 11장의 optimizer는 이 곡률 정보를 직접 또는 간접적으로 근사하지만, 여기서는 logit 한 행의 정확한 구조를 기준점으로 남긴다.

**margin 직관이 유효한 범위를 정한다**

두 class에서는 logit 차이 `d=z_y-z_o`만 확률을 결정하고 loss는 `log(1+exp(-d))`다. 큰 양의 margin에서는 loss와 gradient가 지수적으로 작아지고, margin 0에서는 loss가 `log 2`다. 다중 class에서는 정답과 최대 오답 하나의 차이만으로 충분하지 않다. 많은 중간 오답의 exp 질량이 분모에 합쳐진다. vocabulary가 커질수록 tail 질량을 무시한 top-2 설명이 틀릴 수 있다.

정답 logit이 그대로여도 모든 오답을 조금씩 내리면 loss가 감소한다. 반대로 top-1 예측이 맞아도 두 번째 이하의 총 질량이 크면 NLL은 높다. accuracy와 NLL이 서로 다른 정보를 주는 이유다. calibration은 예측 confidence와 실제 빈도의 대응이고 ranking accuracy나 likelihood와도 구분한다. 학습 loss, validation NLL, ECE, task accuracy를 하나의 “좋아짐”으로 축약하지 않는다.

### denominator를 유효 학습률과 data weighting으로 해석한다

**microbatch·rank·sequence를 가로지르는 보존식을 세운다**

rank `r`, microstep `k`의 numerator를 `S_rk`, 유효 target 수를 `N_rk`라 하자. 원하는 global token mean은 `L=(Σ_rk S_rk)/(Σ_rk N_rk)`다. 각 microbatch에서 `S_rk/N_rk`를 만든 뒤 K와 world size로 평균하면 `N_rk`가 모두 같을 때만 같은 값이다. 길이와 supervised ratio가 다른 instruction data에서는 거의 성립하지 않는다.

gradient도 같은 식을 따라야 한다. DDP가 rank gradient를 평균하는 경우 local loss를 `S_r/N_global`에 world size를 곱해 backward하거나, reduction semantics에 맞는 등가식을 사용한다. 정확한 배율은 framework가 sum인지 mean인지에 달렸으므로 추측해서는 안 된다. 최종 parameter delta를 single-process concatenated-batch reference와 비교해 증명한다. scalar loss가 같더라도 gradient accumulation wrapper가 추가 division을 하면 update는 달라진다.

`N=0`인 microbatch는 특별한 상태다. 단순히 loss 0을 반환하고 그 rank만 backward를 건너뛰면 다른 rank는 collective에 들어가 hang할 수 있다. graph-connected zero를 만들어 동일한 collective 순서를 유지할지, batch 전체를 coordinated skip할지 정한다. optimizer step, scheduler step, scaler growth tracker, data cursor가 함께 움직이거나 함께 멈추어야 한다. zero-valid event를 정상 loss 0과 구분해 계측한다.

packed sequence에서는 document boundary 뒤 첫 token을 앞 문서가 예측하지 않게 할지 정책을 명시한다. attention을 block diagonal로 막아도 labels가 경계를 가로질러 shift되면 cross-document target이 남을 수 있다. 반대로 EOS를 학습하려고 경계 target을 일부 유지할 수 있다. 어느 쪽이든 numerator와 denominator의 집합이 달라진다. attention mask만 보고 label mask가 맞다고 가정하지 않는다.

**분모 변경을 learning-rate 변경과 구분한다**

모든 gradient에 일정한 상수만 곱하면 SGD 첫 step에서는 learning rate 역배율과 비슷해 보일 수 있다. Adam은 first·second moments, epsilon, weight decay, clipping 때문에 완전히 같지 않다. 분모 오류가 batch마다 달라지면 방향도 바뀐다. 특히 domain별 평균을 다시 평균하면 domain weight가 token count가 아니라 batch 등장 횟수에 의해 결정된다.

gradient clipping이 있으면 scale 오류가 threshold를 넘는지에 따라 비선형 효과를 만든다. AMP scaler는 loss scale을 나중에 되돌리는 수치 장치이지 objective denominator를 교정하지 않는다. scheduler는 optimizer update 수를 기준으로 움직일 수 있으므로 valid token 수가 달라져도 같은 LR이 적용된다. 그래서 로그에는 `input_tokens`, `valid_targets`, `sequences`, `microsteps`, `optimizer_updates`를 따로 남긴다.

**고정 revision의 함수와 테스트를 증거 사슬로 묶는다**

**구현 좌표는 현재 snapshot과 역사적 설명을 분리한다**

이 책의 로컬 검증 snapshot `sources/transformers-v5.15.1`은 commit `550d7b3834670483a4df436541272c055dc364bf`다. 이 snapshot에서 `src/transformers/loss/loss_utils.py:32`의 `fixed_cross_entropy`와 `:49`의 `ForCausalLMLoss`가 현재 직접 확인 가능한 핵심 좌표다. 앞 절에서 사용한 commit `36deb0b53ed0863f4b4dfdea23dcaec7f3df3701`은 별도의 역사적 기준점이다. 두 revision의 line number와 함수 본문이 우연히 비슷하더라도 동일 source라고 합치지 않는다.

현재 snapshot의 `tests/trainer/test_trainer.py:275` 부근은 `num_items_in_batch`가 있을 때 accumulation과 큰 batch의 gradient norm을 촘촘한 tolerance로 비교한다. `:337` 부근은 causal LM count가 shift 뒤 `labels[...,1:]`를 기준으로 해야 함을 검사한다. `tests/test_modeling_common.py:1745` 부근은 이미 shift한 labels가 다시 causal loss 경로에서 이동되는 double-shift 위험을 명시한다. 이 세 test는 각각 accumulation, count, shift 경계를 지지한다. 어느 하나도 tokenizer offset, packed boundary, 실제 분산 collective, checkpoint resume 전체를 증명하지는 않는다.

source dossier에는 `repository, commit, path, symbol, start_line, end_line, content_hash`를 기록한다. line은 탐색 편의를 위한 좌표이고 content hash가 이동을 감지한다. 호출자 `modeling_*.py`가 `loss_function`에 넘기는 kwargs, `LOSS_MAPPING`이 선택하는 함수, Trainer가 `num_items_in_batch`를 산출하는 경로를 별도 edge로 연결한다. 함수가 존재한다는 사실과 현재 모델 호출에서 실제 실행된다는 사실을 구분한다.

다음은 독자가 작성할 수 있는 작은 reference의 논리다. 실제 production 함수의 복제품이 아니라 oracle이므로 FP64와 명시적 sum/count를 사용한다.

```python
def reference_causal_nll(logits64, labels, ignore=-100):
    targets = labels[..., 1:].contiguous().view(-1)
    scores = logits64[..., :-1, :].contiguous().view(-1, logits64.size(-1))
    keep = targets.ne(ignore)
    chosen = scores[keep].gather(1, targets[keep, None]).squeeze(1)
    lse = torch.logsumexp(scores[keep], dim=-1)
    return (lse - chosen).sum(), keep.sum()
```

이 oracle의 shift 방식은 호출 계약에 맞게 조정해야 한다. 현재 `ForCausalLMLoss`의 pad-and-slice 경로와 비교할 때 logits 전체 길이와 labels 전체 길이를 동일하게 놓는 fixture가 필요하다. 핵심은 numerator와 count를 따로 반환해 accumulation과 DDP 단계가 목적함수를 재조립하게 하는 것이다.

**테스트를 함수·통합·분산·재개의 네 층으로 만든다**

함수 층에서는 all-equal logits, 극단 offset, ignore-only, target edge ID, non-contiguous input을 시험한다. 통합 층에서는 tokenizer와 collator에서 model loss까지 원문 span을 추적하고 double shift와 packed boundary를 주입한다. 분산 층에서는 rank별 valid count를 1과 7처럼 의도적으로 다르게 해 concatenated reference와 비교한다. 재개 층에서는 checkpoint 직후 같은 batch의 logits·numerator·count·gradient projection이 uninterrupted run과 맞는지 본다.

각 층은 예상 실패도 가져야 한다. target out-of-range는 조용히 ignore하지 말고 명시적 오류여야 한다. all-ignore는 NaN이나 hang이 아니라 정의된 skip 전이를 보여야 한다. wrong denominator fixture는 scalar와 parameter delta 모두에서 실패해야 한다. double-shift fixture는 loss가 감소할 수 있어도 원문-target alignment assertion에서 실패해야 한다.

**한 token에서 전체 학습 시스템으로 왕복한다**

**정방향 추적은 주소와 상태를 잃지 않는다**

원문 byte span 하나를 고른다. normalization 뒤 span, tokenizer piece와 ID, packed sample의 sequence·position, embedding row, block별 hidden checksum, LM head의 target logit과 row maximum, log-sum-exp, NLL contribution, mask weight, global denominator를 차례로 기록한다. 이때 전체 activation을 보존할 필요는 없다. shape·dtype·stride·checksum과 좁은 projection probe로 최초 차이를 찾을 수 있다.

역방향에서는 scalar에서 해당 contribution의 weight를 확인하고 logit gradient, hidden gradient projection, LM head row·column projection, tied embedding alias contribution까지 걷는다. 그 다음 2장의 unscale·clip·optimizer step으로 넘어간다. 한 token의 영향이 parameter 전체에 퍼지므로 모든 원소를 저장하는 대신 고정 random projection과 선택 row를 사용한다. projection seed도 재현 상태다.

이 왕복은 설명을 멋있게 만드는 장치가 아니라 장애 격리 도구다. 재개 뒤 loss가 달라졌다면 raw document부터 denominator까지 어느 경계가 처음 달라졌는지 찾는다. loss는 같은데 parameter가 달라졌다면 gradient 이후를 조사한다. validation PPL만 달라졌다면 tokenizer 단위와 eval denominator부터 맞춘다. 현상을 담당 팀 이름으로 넘기기 전에 최초 불일치 tensor와 그 owner를 제시한다.

**운영자가 사용할 최종 판정표**

먼저 objective를 식으로 쓰고 target distribution, mask, weight, numerator와 denominator를 명시했는지 확인한다. 식의 각 항은 revision이 고정된 함수와 tensor에 연결하고, 원문에서 target까지와 scalar에서 원문까지 양방향 lineage를 갖춰야 한다. 이어 single process, accumulation과 uneven DDP가 같은 global objective를 만드는지, zero-valid와 non-finite가 collective·scheduler·cursor까지 원자적으로 처리되는지 검증한다.

그다음 fused path를 FP64 reference와 비교하되 forward scalar뿐 아니라 per-token contribution과 gradient까지 본다. tied parameter alias가 checkpoint와 optimizer에서 보존되는지, dashboard가 numerator와 denominator를 따로 보존하는지도 확인한다. 마지막으로 source test가 실제로 증명하는 범위를 넘겨 말하지 않았는지 살피고, 2장에 넘기는 manifest로 첫 update를 재현한다.

이 열 항목 중 하나라도 빠지면 “다음 토큰 loss를 이해했다”는 결론은 보류한다. 모델의 철학을 설명하는 것과 실제 학습 목적함수를 복원하는 것은 다르다. 이 장의 완성 조건은 독자가 임의의 loss spike를 만났을 때 tokenizer, shift, mask, denominator, logits, distributed reduction 가운데 어디를 먼저 파야 하는지 증거로 결정할 수 있는 상태다.

## 1.11 vocabulary projection과 target data boundary를 검산한다

vocabulary 폭은 softmax 비용뿐 아니라 tokenizer 좌표와 target construction을 함께 결정한다. full-logit oracle을 기준으로 sharding·근사 loss와 data boundary가 확률 질량과 label 의미를 보존하는지 본다.

**LM head는 마지막에 붙은 사소한 층이 아니다**

hidden width가 `C`, vocabulary가 `V`, 유효 위치가 `N`이면 dense LM head는 대략 `2NCV` FLOPs의 행렬곱을 수행하고 `[N,V]` logits를 만든다. 긴 context와 큰 vocabulary에서는 이 activation의 메모리가 상당하다. 예를 들어 logits를 FP32로 보존하면 원소마다 4바이트가 필요하다. sequence parallel이나 tensor parallel로 hidden이 나뉘어 있어도 vocabulary projection과 loss가 어떤 축을 소유하는지에 따라 통신과 메모리 경로가 달라진다.

그래서 구현은 logits를 chunk로 계산하거나 linear와 cross entropy를 fuse하고, vocabulary를 rank에 shard한다. 최적화의 질문은 단순히 “더 빠른가”가 아니다. target logit을 정확한 shard에서 읽는가, global maximum과 exponential sum을 올바르게 reduce하는가, loss mask와 denominator가 chunk 경계에서 유지되는가, backward가 hidden과 weight gradient를 reference와 같은 방식으로 합치는가를 확인해야 한다.

chunked loss에서 각 chunk mean을 다시 평균하면 마지막 chunk가 짧을 때 가중치가 틀어진다. 각 chunk는 numerator와 valid count를 내고 마지막에 합쳐야 한다. vocabulary chunk와 token chunk도 구분한다. vocabulary chunk는 한 token의 분모를 분할하므로 log-sum-exp 결합이 필요하고, token chunk는 독립적인 loss 항을 분할하므로 numerator/count 합산이 필요하다. 두 축을 같은 `chunk_size` 옵션으로 뭉개면 버그를 설명하기 어렵다.

sampled softmax나 negative sampling은 full vocabulary denominator를 근사하거나 다른 objective를 사용한다. 그것을 dense CE의 단순 성능 최적화라고 부르면 안 된다. sampling distribution과 importance correction이 gradient의 bias·variance를 결정한다. 대규모 causal LM에서 full softmax를 쓰는지, adaptive 또는 sampled variant를 쓰는지는 model card와 training code로 확인한다. 추정만으로 objective를 바꾸지 않는다.

**weight tying은 함수와 저장장치의 두 계약이다**

input embedding `E`와 output weight `W`를 tie해 `W=E`로 두면 같은 parameter가 lookup과 projection 두 경로에서 쓰인다. 수학적으로 gradient는 두 경로의 합이다. 구현에서는 두 Python attribute가 같은 `Parameter` 객체인지, 다른 view가 같은 storage를 가리키는지, load 뒤 다시 tie하는지에 따라 optimizer와 serialization 동작이 달라진다.

parameter enumeration이 alias를 한 번만 반환하는지 확인한다. weight decay가 두 번 적용되거나 optimizer state가 두 벌 생기면 tying의 의미가 깨진다. FSDP flattening과 tensor parallel shard는 alias를 변환할 수 있다. checkpoint state dict는 같은 tensor를 두 key에 저장하거나 한 key와 tie metadata를 저장할 수 있다. save/load round trip 뒤 object identity, storage pointer, 값, gradient 합산, optimizer state owner를 각각 검증한다.

vocabulary resize는 더 위험하다. 새 token rows를 추가하면 embedding과 LM head 양쪽 shape, tokenizer vocab size, config vocab size, optimizer moments, distributed shard metadata가 함께 바뀌어야 한다. 새 row 초기화를 평균 embedding으로 할지 random으로 할지는 학습 초기 logits와 gradient에 영향을 준다. 기존 optimizer checkpoint에 새 rows의 moment가 없을 때 0으로 시작하는 정책도 기록한다. resize 뒤 old token logits가 동일한지, 새 IDs가 범위 안인지, tie가 유지되는지 fixture로 확인한다.

### template·packing·truncation에서 target 위치를 재구성한다

**chat template은 문자열 장식이 아니라 supervision compiler다**

대화 `system/user/assistant`를 template에 넣으면 role marker, 줄바꿈, BOS·EOS가 추가된다. tokenizer가 이를 ID로 바꾸고 collator는 어느 위치를 학습할지 label mask를 만든다. assistant-only loss에서는 user와 system token의 labels를 ignore해도 그 token들은 context로 남아 assistant 예측에 영향을 준다. attention에서 제거하는 것과 loss에서 제거하는 것은 전혀 다르다.

template가 assistant 시작 marker를 여러 token으로 분절할 때 character substring 검색으로 mask 경계를 잡으면 normalization과 special-token 처리에서 어긋날 수 있다. tokenizer가 반환한 offsets는 special token에 빈 span이나 sentinel을 둘 수 있다. 가장 안전한 경계는 template가 생성 과정에서 role별 token span을 함께 내거나, generation mask를 명시적으로 제공하고 round-trip fixture로 검증하는 것이다.

assistant 답변 끝의 EOS를 학습할지 결정한다. EOS를 ignore하면 모델이 답을 멈추는 신호를 덜 배우고, 포함하면 truncated example에서 잘못된 종료를 가르칠 수 있다. 마지막 assistant turn만 학습할지 모든 assistant turn을 학습할지도 denominator와 turn weight를 바꾼다. tool call JSON과 tool result는 어떤 role에 속하고 어느 token이 target인지 별도로 정의한다.

다음과 같은 최소 표를 실제 batch마다 만들 수 있어야 한다.

| 위치 | 원문/생성 span | token ID | role | input 사용 | target 사용 | 다음 target |
|---:|---|---:|---|---|---|---:|
| 0 | BOS | 1 | special | 예 | 아니오 | 42 |
| 1 | system marker | 42 | system | 예 | 아니오 | 913 |
| 2 | system text | 913 | system | 예 | 아니오 | 17 |
| 3 | assistant marker | 17 | assistant boundary | 예 | 정책에 따름 | 804 |
| 4 | answer piece | 804 | assistant | 예 | 예 | 2 |
| 5 | EOS | 2 | special | 예 | 예/정책 | 없음 |

표의 숫자는 fixture용 예시일 뿐 특정 tokenizer의 실제 ID가 아니다. 중요한 것은 각 위치의 input 역할과 그 위치 logit이 맞힐 target을 분리하는 것이다. label tensor의 같은 index가 현재 input token인지 다음 target인지 API 관례에 따라 헷갈릴 수 있으므로 실제 `ForCausalLMLoss` shift 뒤 대응표를 함께 둔다.

**packing은 빈 공간 제거 이상의 변환이다**

여러 문서를 한 fixed-length sequence에 붙이면 padding을 줄일 수 있다. 그러나 문서 경계를 넘어 attention을 허용할지, EOS로만 구분할지, block-diagonal mask를 쓸지에 따라 모델이 보는 context가 달라진다. position IDs를 각 문서에서 재시작할지 계속 증가시킬지도 RoPE phase와 kernel 입력을 바꾼다.

경계에서 마지막 token이 다음 문서 첫 token을 예측하는 target을 제거하려면 label bitmap에 반영한다. block-diagonal attention만으로 자동 해결되지 않는다. sequence 끝에서 다음 target이 없는 마지막 위치도 제외해야 한다. packing 후 `sum(valid_count)`가 개별 examples의 예상 target 수와 맞는지 보존식을 둔다. EOS 포함 정책에 따라 정확한 식을 적는다.

varlen attention은 cumulative sequence lengths를 사용하고 loss flatten은 별도의 order를 사용할 수 있다. 둘의 token order가 다르면 attention output과 labels가 엇갈린다. packed sample manifest에 original example ID, local position, packed flat index, cu-seqlens segment, label-valid bit를 저장한다. 전체 corpus에는 무겁지만 golden batch에는 필수다.

truncation은 뒤를 자르는 단순 연산처럼 보여도 supervision을 바꾼다. assistant 답변이 잘려 EOS가 사라지거나 질문만 남아 valid target이 0이 될 수 있다. left truncation은 system instruction을 제거할 수 있고 right truncation은 정답 후반을 제거한다. length bucket별 supervised ratio와 truncation reason을 계측한다. “max length를 늘렸다”는 옵션은 memory뿐 아니라 데이터 분포와 denominator를 바꾼다.

### focal·unlikelihood·auxiliary objective의 변경점을 격리한다

**label smoothing·class weight·focal loss를 한 축에 놓지 않는다**

label smoothing은 target distribution을 바꾼다. class weight는 관측 target class에 따른 contribution scale을 바꾼다. focal loss는 현재 예측 난도에 따라 동적으로 weight를 바꾼다. 셋 모두 scalar가 달라지지만 gradient field가 바뀌는 방식은 다르다.

categorical label smoothing에서 `q=(1-ε)e_y+εu`라 하면 gradient는 `p-q`다. `u`가 uniform over V인지 non-target V-1인지 구현을 확인한다. ignore 위치에 smoothing mass를 만들면 안 된다. vocabulary가 매우 크면 각 오답에 가는 질량은 작지만 합은 ε다. rare token 개선이라는 주장은 frequency bucket 평가로 확인해야 한다.

class weight `a_y`가 있으면 token loss가 `a_yℓ`이 되고 mean reduction의 분모가 weight 합인지 valid item 수인지 framework 정의를 확인한다. weight sum으로 나누면 batch class composition에 따라 effective scale이 달라진다. item count로 나누면 평균 weight가 곧 gradient scale이 된다. source와 작은 fixture로 구분한다.

focal 형태 `(1-p_y)^γℓ`은 쉬운 예제의 기여를 줄이지만 weight 자체가 p에 의존해 derivative에 추가 항이 생긴다. 단순히 CE gradient에 weight만 곱한 것으로 설명하면 틀린다. causal vocabulary에서 focal loss가 calibration과 rare-token 학습, noisy label에 미치는 영향은 별도 실험이 필요하다. 기존 vision classification의 직관을 그대로 옮기지 않는다.

unlikelihood loss는 금지 token 집합의 probability를 낮추는 항을 추가할 수 있다. positive CE와 negative constraint의 mask, coefficient, 중복 token 처리, numerical clamp를 기록한다. 금지 항이 모든 vocabulary를 밀어낼 때 positive target과 충돌하는지 본다. auxiliary router loss, z-loss, contrastive loss가 더해지면 각 numerator와 denominator, coefficient schedule을 따로 로그한다. 총 loss 하나만으로 어느 항이 update를 지배했는지 알 수 없다.

**z-loss와 logit regularization의 의도를 밝힌다**

log-partition `log Z=logΣexp z_j`에 벌점을 주는 z-loss는 softmax 확률의 공통 이동 불변성을 깨고 logit scale을 제어한다. CE 확률은 공통 상수에 불변이지만 수치 표현과 distributed reduction은 절대 logit 크기의 영향을 받는다. z-loss는 이 보이지 않는 방향을 regularize할 수 있다. coefficient가 너무 크면 예측 목적보다 scale 제어가 앞설 수 있다.

logit clipping, temperature, norm penalty도 겉으로는 confidence를 낮출 수 있으나 함수가 다르다. clipping은 특정 범위 밖 derivative를 바꾸고 temperature는 모든 logit 차이의 scale을 바꾸며 norm penalty는 절대 좌표에 벌점을 준다. 옵션명보다 정확한 식과 적용 위치, backward를 확인한다.

복수 loss를 accumulation할 때 각 항의 denominator가 다를 수 있다. token CE는 valid token 수, sequence reward는 sequence 수, router auxiliary는 token-expert assignment 수를 쓸 수 있다. 먼저 각 항을 자기 단위로 normalize한 뒤 coefficient를 적용하는지, numerator를 모두 합친 뒤 한 분모를 쓰는지에 따라 의미가 달라진다. loss ledger에는 `name,numerator,denominator,coefficient,scaled_value`를 남긴다.

## 1.12 성능 최적화와 실제 incident를 가설 경쟁으로 진단한다

fused kernel이나 sharded vocabulary는 빠르다는 이유로 수학적 동치를 가정할 수 없다. 작은 oracle과 negative control을 두고 수치 오차, mask·denominator corruption과 data drift 가설을 경쟁시킨다.

### fused kernel을 full-logit oracle과 단계별로 비교한다

reference와 fused loss의 scalar만 비교하면 상쇄 오류를 놓친다. 위치별 NLL 또는 고정 subset, total numerator, count, hidden gradient, weight gradient를 비교한다. dtype별 tolerance는 절대·상대 오차와 작은 값 영역을 나눠 정한다. 동일 input을 여러 layout과 stride, contiguous/non-contiguous, vocabulary tail size로 시험한다.

deterministic reference는 FP64 CPU 또는 명확한 고정 reduction을 사용한다. GPU kernel 간 bitwise 일치를 요구할 필요는 없지만 오차가 sequence length, vocabulary size, logit range와 함께 체계적으로 커지는지 본다. random test만으로 드문 경계를 놓치므로 all-equal, one-dominant, two-tied-max, extreme-negative, ignore-interleaved fixture를 둔다.

kernel은 성능 metric도 분해한다. forward·backward latency, peak allocated/reserved memory, HBM bytes, kernel launch 수, compilation time, communication overlap을 측정한다. 첫 iteration compilation을 steady state에 섞지 않는다. 하지만 빠른 결과가 parity failure를 정당화하지 않는다. correctness gate를 먼저 통과한 configuration만 성능 비교에 넣는다.

CUDA graph capture나 `torch.compile`이 loss 경로를 고정하면 dynamic valid count와 zero-valid branch가 graph break 또는 stale scalar를 만들 수 있다. count tensor가 host `.item()`으로 빠지는지, shape specialization이 sequence length마다 재compile되는지, collective가 capture-safe한지 본다. eager와 compiled path의 GoldenBatch parity를 별도 artifact로 남긴다.

**통신 overlap이 count ordering을 깨뜨리지 않는지 본다**

vocabulary-parallel CE의 maximum·sum collective와 DDP gradient collective는 서로 다른 단계다. stream과 process group을 겹칠 때 event dependency가 빠지면 이전 batch count나 미완료 logsumexp를 읽을 수 있다. 오류가 비결정적으로 나타나므로 stress test에서 stream synchronization을 강제한 reference와 비교한다.

rank가 target을 소유하지 않더라도 collective 호출 순서는 같아야 한다. conditional branch로 일부 rank만 all-reduce에 들어가면 hang한다. empty token shard나 zero-valid rank도 neutral element를 들고 collective에 참여한다. timeout은 원인이 아니라 호출 순서 불일치의 증상일 수 있다. NCCL 로그와 loss state ledger를 UpdateID로 결합한다.

### loss 급변을 data·model·optimizer·runtime 가설로 나눈다

**사례 A: GPU 수를 늘리자 loss가 달라졌다**

먼저 global sample multiset과 순서를 맞춘다. world size 변경으로 sampler partition과 마지막 remainder가 달라졌는지 확인한다. 다음으로 rank별 valid target count와 global count를 비교한다. local mean 평균을 쓰면 긴 sequence가 몰린 rank의 token weight가 줄어든다. dropout RNG와 reduction order는 그 뒤에 조사한다.

single-process concatenated GoldenBatch를 oracle로 둔다. 각 rank numerator와 count를 수집해 scalar를 재조립하고, gradient projection을 비교한다. scalar부터 다르면 objective aggregation 문제다. scalar는 같고 gradient가 다르면 DDP hook, accumulation scale, alias, nondeterministic kernel을 본다. gradient도 같고 parameter delta가 다르면 clipping과 optimizer state를 본다.

수정은 “GPU 수별 learning rate 튜닝”이 아니라 global objective를 동일하게 만드는 것이다. 의도적으로 global batch를 바꾸는 실험이라면 그 사실과 LR scaling rule을 별도 변수로 둔다. 인프라 변화와 최적화 실험을 한 변경으로 섞지 않는다.

**사례 B: tokenizer 교체 뒤 perplexity가 크게 좋아졌다**

raw token PPL 숫자를 바로 성능 향상으로 해석하지 않는다. 동일 byte corpus에서 token count와 bits-per-byte를 계산한다. normalization과 byte fallback, special token 삽입, unknown 처리, document truncation이 달라졌는지 본다. 새 tokenizer가 한 token에 더 많은 bytes를 담으면 token당 NLL 단위가 바뀐다.

old/new tokenizer의 alignment table을 만들고 language·script·code별 fertility를 본다. context window가 token 단위로 고정이면 새 tokenizer는 모델이 보는 byte 범위도 바꾼다. compute budget을 update나 token으로 고정했는지 FLOPs로 고정했는지도 중요하다. tokenizer 품질, 문맥 범위, 학습량 효과를 분리한 뒤 결론을 낸다.

**사례 C: resume 직후 loss가 한 번 튄다**

첫 resume batch의 `DocumentID`, packed layout, labels, valid count를 uninterrupted run과 비교한다. data cursor가 같은 batch를 반복하거나 하나 건너뛰었을 수 있다. model parameters와 optimizer moments뿐 아니라 AMP scaler, scheduler, RNG, sampler epoch, gradient accumulation partial state를 확인한다.

logits까지 같고 loss만 다르면 reduction configuration이나 dynamic count를 본다. loss와 gradient는 같은데 update 뒤 갈라지면 optimizer/scheduler/scaler다. 첫 batch만 다르고 이후 수렴한다는 이유로 무시하지 않는다. 정확한 재현이 필요한 연구 결과에서는 한 update의 차이가 이후 모든 state를 바꾼다.

**사례 D: loss는 내려가지만 답변 학습이 안 된다**

assistant-only mask가 실제 assistant token을 포함하는지 role별 valid ratio를 본다. user prompt나 반복 template marker가 대부분의 denominator를 차지하면 쉬운 형식 token loss가 내려가면서 답변 내용 개선을 가릴 수 있다. token category별 numerator/count와 teacher-forced accuracy를 분리한다.

label shift가 두 칸이어도 언어 통계 때문에 loss는 내려갈 수 있다. 작은 example을 memorization시키고 각 position target을 원문과 대조한다. generation failure가 training objective 문제인지 decoding template 문제인지도 나눈다. train template와 inference prompt가 동일한 role marker·generation prompt 계약을 쓰는지 확인한다.

**2장으로 넘기는 실행 가능한 loss manifest**

**manifest는 설명문이 아니라 재계산 입력이다**

최종 manifest에는 `RunID`, `UpdateID`, `GoldenBatchID`, corpus·tokenizer·template·model revision을 둔다. tensor 항목에는 input IDs, labels, attention mask, position IDs의 shape·dtype·checksum과 target bitmap이 있다. loss 항목에는 각 component의 numerator, denominator, coefficient, reduction owner, shift owner가 있다. 분산 항목에는 world size, rank별 count, collective semantics, accumulation window가 있다.

수치 항목에는 selected logits, row max, logsumexp, per-token NLL subset, scalar loss, gradient projection을 둔다. source 항목에는 repository commit과 함수·test 좌표가 있다. alias 항목에는 embedding/head storage identity와 optimizer parameter owner가 있다. privacy 때문에 원문을 그대로 보존할 수 없다면 접근 통제 artifact ID와 salted checksum, 허가된 redacted span을 사용한다.

2장은 이 manifest를 읽어 loss scale 전 scalar가 맞는지 확인한다. backward 뒤 unscaled gradient projection이 reference와 같은지, clipping이 어떤 norm을 보았는지, optimizer delta가 어느 UpdateID에 commit됐는지 잇는다. 1장과 2장의 경계에서 동일 이름을 재사용하지 않고 loss numerator와 scaled loss, raw gradient와 scaled gradient를 구분한다.

**마지막 독자 실습은 일부러 틀린 시스템을 만든다**

고장 주입은 unequal microbatch에서 local mean을 평균하는 실험으로 시작한다. 이어 collator와 loss에서 shift를 두 번 적용하고, packed boundary label만 남기며, tied weight를 load 뒤 복제해 alias를 끊는다. 마지막으로 zero-valid rank만 backward를 skip한다. 이 고장들은 모두 shape와 초기 loss가 정상처럼 보일 수 있다.

독자는 각 실패를 어느 detector가 가장 먼저 잡아야 하는지 예측한다. 원문-target 표, numerator/count reconciliation, alias ledger, collective trace, parameter delta 가운데 최초 detector가 설계와 다르면 계측 경계를 고친다. 고장을 제거한 뒤 동일 GoldenBatch로 처음부터 왕복해 다른 차이가 남지 않았음을 확인한다.

이제 다음 토큰 예측은 “앞의 단어로 뒤의 단어를 맞힌다”는 입문 문장을 넘어선다. 데이터가 만든 empirical distribution, tokenizer가 만든 좌표, LM head가 만든 score, softmax가 만든 simplex, CE가 만든 gradient, 분모가 만든 sample weight, collective가 만든 전역 목적함수가 한 계약으로 이어진다. 이 계약이 닫혀야 2장의 autograd와 optimizer가 무엇을 전달받아야 하는지 정확히 말할 수 있다.

## 1.13 GoldenTokenRun으로 byte부터 backward까지 왕복한다

한 개의 작고 닫힌 fixture를 byte, IDs, embedding, logits, mask, CE와 gradient까지 양방향으로 계산한다. 큰 모델의 결과는 이 oracle과 같은 불변식을 만족할 때만 다음 단계의 근거가 된다.

**입력과 target을 만든다.** vocabulary를 다섯 개 ID `{0:BOS, 1:가, 2:나, 3:EOS, 4:PAD}`로 두고 한 sample을 `[0,1,2,3]`이라 하자. raw labels도 같은 배열로 model에 넘기고 loss 함수가 causal shift를 소유한다면 위치 0,1,2의 logits는 각각 target 1,2,3을 맞힌다. 위치 3 뒤에는 target이 없으므로 pad한 ignore label을 받는다. 유효 target 수는 3이다. padding을 붙여 `[0,1,2,3,4,4]`로 만들더라도 label mask가 올바르면 count는 여전히 3이다.

첫 위치 logits를 `[0,2,1,-1,0]`이라 하자. 최대값 2를 빼면 `[-2,0,-1,-3,-2]`이고 지수합은 `e^-2+1+e^-1+e^-3+e^-2`다. 정답은 ID 1이므로 target logit은 2다. NLL은 `logsumexp(z)-2`다. 손계산에서는 소수점 반올림을 늦추고 FP64 script가 정확한 reference를 만든다. 같은 logits에 1000을 더해도 NLL이 변하지 않아야 한다.

둘째·셋째 위치도 같은 방식으로 계산해 `S=ℓ_0+ℓ_1+ℓ_2`, `N=3`, `L=S/3`을 얻는다. framework가 반환한 scalar만 저장하지 않고 세 위치 NLL과 S,N을 저장한다. ID 4 위치의 logit은 분모의 class로는 참여하지만 target 위치 자체는 ignore된다. “PAD class를 vocabulary에서 제거한다”와 “PAD 위치의 loss를 제거한다”는 다른 연산이다.

첫 위치 gradient는 `p-e_1`이다. 다섯 성분을 합하면 0이고 정답 성분은 음수다. 전체 mean loss에서는 1/3이 곱해진다. 나머지 위치 gradient와 합쳐 LM head weight gradient를 만든다. hidden vector를 `h_0`라 하면 첫 위치의 weight gradient 기여는 `(p-e_1)h_0ᵀ/3`이다. embedding과 head가 tied이면 입력 ID 0,1,2,3의 lookup 경로 gradient도 같은 storage에 더해진다.

**두 microbatch로 나눈다.** 두 번째 sample이 target 하나만 가진다고 하자. 첫 microbatch는 `(S_1,N_1=3)`, 둘째는 `(S_2,N_2=1)`이다. 올바른 전체 mean은 `(S_1+S_2)/4`다. 두 mean의 평균 `(S_1/3+S_2)/2`은 둘째 sample의 한 target을 첫 sample의 target 하나보다 세 배 무겁게 만든다. scalar 예제로 보인 차이는 gradient에도 그대로 들어간다.

accumulation 구현은 각 microbatch numerator를 전역 denominator 4로 나누어 backward하거나 합법적인 등가 방식을 쓴다. DDP가 gradient를 world-size 평균하면 그 배율까지 반영한다. 이때 `num_items_in_batch=4`가 model loss에 전달되었는지, training loop가 추가로 accumulation steps로 나누는지 source를 확인한다. 한 곳에서만 normalize해야 한다는 말은 API별 reduction semantics를 확인한 뒤의 결론이다.

**고장을 하나씩 주입한다.** labels를 미리 `[1,2,3,-100]`으로 이동한 뒤 `ForCausalLMLoss`가 다시 이동하게 하면 target은 `[2,3,-100,…]`이 된다. count가 3에서 2로 바뀌거나 잘못 구성하면 shape만 유지된 채 엉뚱한 목표가 된다. PAD label을 4로 남기면 count가 늘고 모델은 padding을 예측하도록 학습한다. 두 microbatch mean을 평균하면 count는 로그상 맞아도 effective weight가 달라진다.

각 고장에 detector를 배치한다. double shift는 position-target table, PAD 누락은 valid bitmap과 count, 평균 오류는 concatenated numerator/count reference가 잡는다. detector의 출력에는 관찰값뿐 아니라 기대값을 계산한 fixture revision을 넣는다. “loss가 예상과 다름” 대신 “packed index 5의 target이 raw byte span 18–21이 아니라 span 21–24를 가리킴”처럼 보고한다.

**분산으로 확장한다.** rank 0에 target 세 개, rank 1에 target 한 개를 놓는다. 각 rank의 S와 N을 all-reduce해 global scalar를 만들거나 global N에 맞게 local backward scale을 정한다. rank 1이 all-ignore가 되어도 collective 순서는 유지한다. rank별 scalar 평균이 아니라 global S/global N을 dashboard에 보낸다. 이 작은 fixture가 통과한 뒤 실제 vocabulary parallel과 accumulation으로 규모를 키운다.

### tokenizer revision별 token·byte denominator를 비교한다

**같은 문장이 다른 학습 문제로 바뀐다.** 문자열 “학습한다”가 tokenizer A에서는 두 token, B에서는 네 token이면 같은 byte sequence가 서로 다른 수의 조건부 예측 항으로 분해된다. 각 token의 context 경계와 정답 단위가 달라진다. 총 log-likelihood를 token mean으로 나누면 분모까지 달라진다. 모델 크기와 compute가 같아도 update당 예측 사건 수와 sequence당 byte 범위가 바뀐다.

tokenizer 비교표에는 vocabulary size, normalization, pre-tokenization, model algorithm, byte fallback, special tokens, fertility, bytes/token, unknown rate를 넣는다. 언어·script·code·수학·공백 패턴별로 분리한다. 평균 fertility 하나는 소수 언어에서 발생하는 극단적 분절을 숨긴다. 긴 token이 항상 좋은 것도 아니다. 희귀 문자열을 큰 단위로 외우면 조합 일반화와 vocabulary 효율이 나빠질 수 있다.

normalization이 compatibility character나 공백을 합치면 원문으로의 역추적이 어려워진다. offset mapping이 normalized string 기준인지 raw bytes 기준인지 확인한다. Unicode code point, grapheme cluster, UTF-8 byte offset을 섞지 않는다. 삭제 요청과 contamination 조사에서는 raw byte 좌표가 필요하고 UI 표시는 grapheme 좌표가 유용할 수 있다. 둘 사이 변환표를 보존한다.

special token은 일반 문자열 matching보다 tokenizer vocabulary의 atomic ID로 처리되어야 한다. user text가 marker 문자열을 포함했을 때 role boundary로 오인하지 않는지 시험한다. added token의 whitespace stripping 옵션이 주변 tokenization을 바꿀 수 있다. chat template revision과 tokenizer revision을 별도 식별하되 호환성 pair를 manifest로 고정한다.

**vocabulary 확장의 학습 영향을 추적한다.** 새 domain token을 추가하면 기존 문자열의 segmentation이 달라질 수 있다. tokenizer algorithm이 greedy longest match라면 새 token이 기존 여러 token을 대체한다. pretrained checkpoint의 새 embedding row에는 학습된 의미가 없고 optimizer state도 없다. 새 token 출현 빈도, initialization, learning-rate group, freezing 정책을 기록한다.

새 token의 input embedding은 문맥을 만드는 경로, output row는 정답 probability를 만드는 경로다. tying되어 있으면 하나의 row가 둘을 함께 담당한다. 초기 output logit이 지나치게 크거나 작으면 새 token probability와 기존 denominator 전체를 교란한다. resize 직후 old corpus의 logit distribution, new token rank, loss를 비교한다.

tokenizer 교체가 필요하다면 old checkpoint를 그대로 이어 학습하는 것과 embedding을 remap하는 것은 별도 방법이다. 동일 문자열 조각, decomposition average, learned mapping 등 remap 정책마다 보존하는 의미가 다르다. 정확한 equivalence를 주장하지 않고 downstream recovery curve와 old-language regression을 측정한다.

### loss curve를 data·representation·optimization 원인으로 분해한다

**시간축부터 바로 잡는다.** x축을 wall-clock, optimizer update, input token, valid target, training FLOPs로 각각 본다. throughput 최적화는 wall-clock 곡선을 개선하지만 update당 objective가 같을 수 있다. packing 변경은 input token당 valid ratio를 바꾼다. gradient accumulation이나 world size 변경은 update당 token 수를 바꾼다. 축 하나만 보면 인과를 잘못 읽는다.

loss를 domain, language, role, length, tokenizer fertility, duplication bucket으로 분해하되 metric cardinality를 제한한다. 각 bucket도 numerator와 denominator를 보존한다. 전체 평균 개선이 특정 대량 domain의 개선으로 생겼는지, 희귀 domain regression을 숨기는지 본다. curriculum stage가 바뀌면 mixture weights를 annotation으로 기록한다.

train loss와 validation loss의 간격은 과적합의 한 신호지만 pipeline 차이도 반영한다. validation tokenizer/template가 같은지, packing과 mask가 같은지, contamination이 없는지 먼저 확인한다. dropout과 eval mode, model loss branch도 맞춘다. validation PPL 비교에는 tokenizer 단위를 고정한다.

loss spike를 조사할 때 직전 batch 하나만 보지 않는다. optimizer moment와 scheduler state 때문에 이전 여러 updates의 영향이 늦게 나타날 수 있다. spike update의 input dossier, gradient norm, clip coefficient, scaler skip, parameter delta를 보고 이전 checkpoint에서 같은 batch를 재생한다. batch를 바꿔도 재현되면 state 문제, 같은 batch에서만 재현되면 data/content 또는 length 문제 가능성이 커진다.

plateau에서는 label correctness, valid ratio, trainable parameter count, learning rate, gradient flow, delta/weight ratio 순으로 본다. 작은 clean batch를 반복해 overfit 가능한지 시험한다. 이 실험은 성능 benchmark가 아니라 plumbing 검사다. overfit도 못하면 capacity 논쟁 전에 objective와 update 경로를 고친다.

**loss가 좋아도 모델이 나빠지는 경우를 기록한다.** 데이터 중복이 늘면 train loss는 쉽게 내려가지만 generalization은 나빠질 수 있다. 답변 template marker가 denominator를 많이 차지하면 형식 예측 개선이 의미 예측을 가린다. 긴 chain-of-thought를 token mean으로 학습하면 짧은 정답 sample의 weight가 줄어들 수 있다. harmful content를 잘 예측하는 것도 likelihood 관점에서는 개선이다.

따라서 loss는 목적함수 최적화의 계기판이지 최종 가치 함수가 아니다. 24장의 benchmark, 25장의 red teaming, 26장의 multimodal 평가와 연결한다. 하지만 loss를 무시해서도 안 된다. downstream 이상이 objective 입력, target, gradient 중 어디서 시작됐는지 찾는 가장 세밀한 신호이기 때문이다.

### loss code 변경을 option→state→effect 순서로 검토한다

**diff를 옵션 이름이 아니라 상태 변화로 읽는다.** loss 관련 pull request에서 signature, default, dtype cast, shift, flatten, ignore, reduction, count 전달, distributed branch를 확인한다. 호출자가 새 인자를 실제로 넘기는지 repository 전체에서 symbol을 검색한다. model별 override와 custom loss hook이 기본 함수를 우회하는지 본다.

테스트 diff에서는 어떤 failure를 재현하는 fixture인지 읽는다. assertion이 scalar만 보는지 gradient와 count도 보는지, CPU만인지 CUDA·dtype 경로를 포함하는지, single process인지 distributed인지 기록한다. regression test가 추가되지 않았다면 작은 fixture를 로컬 evidence로 만든다. test 이름만 인용하지 않고 assertion의 범위를 설명한다.

line number는 revision마다 변한다. 그래서 commit과 symbol, signature, surrounding content hash를 함께 저장한다. release tag만으로는 vendored patch나 downstream fork를 식별하지 못할 수 있다. 설치된 package의 version, wheel hash, source commit mapping을 기록한다. runtime 실행은 이 책의 조사 범위에서 대규모 모델을 요구하지 않으며 정적 source와 작은 tensor fixture로 대부분의 계약을 검토할 수 있다.

**변경 영향 반경을 예측한다.** shift가 바뀌면 position-target alignment와 valid count 이후가 달라져야 한다. dtype cast가 바뀌면 logits 이전은 같고 loss/gradient tolerance가 달라질 수 있다. denominator가 바뀌면 per-token NLL은 같고 scalar와 gradient scale 이후가 달라진다. 예상보다 앞선 tensor가 바뀌면 실험 입력이 통제되지 않은 것이다.

영향 반경 표에는 `changed_symbol`, `first_expected_difference`, `states_that_must_not_change`, `tests`, `rollback`을 둔다. 배포 전 canary에서 같은 GoldenBatch를 old/new path에 넣고 비교한다. 성능과 수치 차이를 별도 gate로 둔다. rollback 뒤 optimizer state가 이미 다른 update를 먹었다면 단순 binary rollback으로 trajectory가 복원되지 않으므로 checkpoint 경계도 계획한다.

**장 전체를 관통하는 조사 질문**

**목적함수 질문.** 어떤 joint distribution factorization을 쓰는가. 어느 positions가 target이고 어떤 weight를 갖는가. numerator와 denominator는 무엇이며 누가 계산하는가. label smoothing이나 auxiliary loss가 target 또는 함수에 무엇을 더하는가. sequence·domain·rank 사이의 평균 방식은 무엇인가.

**데이터 질문.** 한 target을 raw byte까지 되돌릴 수 있는가. normalization, tokenizer, template, packing, truncation revision은 무엇인가. attention mask와 label mask가 각각 무엇을 막는가. EOS와 boundary를 학습하는 정책은 무엇인가. zero-valid example을 어느 단계에서 처리하는가.

**수치 질문.** projection과 loss의 dtype은 무엇인가. stable log-sum-exp와 vocabulary shard combine은 어떻게 구현되는가. extreme logits, all-equal logits, tail vocabulary에서 reference와 맞는가. non-finite 최초 경계와 허용 tolerance는 무엇인가.

**미분 질문.** `p-q` gradient가 hidden, LM head, tied embedding으로 어떻게 흐르는가. denominator와 coefficient가 어느 시점에 곱해지는가. fused backward가 reference와 맞는가. accumulation과 DDP 뒤 global objective의 gradient인가.

**상태 질문.** GoldenBatch, UpdateID, source revision, alias, RNG, accumulation window가 연결되는가. zero-valid와 overflow에서 collective·optimizer·scheduler·cursor가 함께 전이하는가. resume 첫 batch가 uninterrupted run과 동일한가.

**운영 질문.** dashboard가 numerator/count를 보존하는가. loss spike를 sample dossier와 연결할 수 있는가. privacy와 접근 제어를 지키는가. 성능 최적화의 first-difference를 예측하고 rollback할 수 있는가. upstream test가 증명하지 않는 범위를 명시했는가.

이 질문들은 장 끝의 암기 목록이 아니다. 코드 리뷰, 학습 시작 전 readiness review, 장애 triage, checkpoint 재개, framework upgrade마다 반복하는 순서다. 답이 prose에만 있고 artifact로 재계산되지 않으면 미완성으로 본다. 답이 모두 연결되면 독자는 loss 숫자를 바라보는 데서 멈추지 않고 그 숫자가 태어난 전 과정을 조사할 수 있다.

**한 장을 실제 조사에 사용하는 순서**

**첫 30분에는 현상을 고정한다.** “loss가 이상하다”를 관측 가능한 문장으로 바꾼다. 어느 RunID와 UpdateID에서, train과 validation 중 어디서, 모든 rank인지 일부 rank인지, scalar jump인지 NaN인지 plateau인지 적는다. 직전 정상 checkpoint와 최초 이상 checkpoint를 고르고 config diff, source revision, data cursor를 보존한다. 새로운 run을 여러 옵션으로 무작정 시작하기 전에 재현 가능한 한 batch를 확보한다.

GoldenBatch를 만들 때 민감한 원문을 일반 로그에 복사하지 않는다. 접근 통제 artifact에 raw checksum과 필요한 span을 두고 공개 ledger에는 opaque ID를 남긴다. tokenizer output, template result, packing map, labels, valid bitmap을 함께 저장한다. batch를 저장했는데 tokenizer revision이 없으면 재현 묶음이 아니다.

**다음 30분에는 forward를 이분한다.** input IDs와 masks가 같으면 embedding output projection을 비교한다. block midpoint hidden checksum으로 상·하반을 나누고 최초 다른 layer를 찾는다. 모든 hidden이 같으면 LM head target logits, row maximum, logsumexp, per-token NLL을 본다. per-token NLL까지 같으면 numerator와 denominator aggregation만 남는다.

이때 checksum 하나만 믿지 않는다. shape, dtype, stride, finite count, RMS, 선택 원소와 함께 본다. checksum collision 가능성보다 더 흔한 문제는 서로 다른 layout을 같은 logical order로 hash하지 않거나 nondeterministic reduction의 작은 차이를 완전 불일치로 해석하는 것이다. 비교 목적에 맞는 canonicalization과 tolerance를 정한다.

**그 다음에는 backward 경계를 확인한다.** scalar loss와 scale 전 numerator/count가 같다면 selected logit gradient와 hidden gradient projection을 비교한다. tied head/embedding alias와 parameter registration을 확인한다. accumulation microstep별 local gradient contribution, 마지막 sync, unscale 전후를 2장으로 넘긴다. forward가 같은데 backward가 갈라지면 데이터 pipeline을 계속 뒤지지 않는다.

**분모 오류는 별도 실험으로 확인한다.** 실제 batch에서 길이가 비슷해 우연히 local mean과 global mean이 가까울 수 있다. rank와 microbatch마다 valid count가 극단적으로 다른 synthetic fixture를 만든다. 모든 target NLL을 상수로 두면 기대 scalar를 손으로 계산하기 쉽다. count가 1인 group과 9인 group을 평균해 잘못된 weighting이 명확히 드러나게 한다.

**수정 뒤에는 반증한다.** 고친 configuration이 원래 failure를 없앤 것만으로 부족하다. 정상 fixture, boundary fixture, zero-valid fixture, distributed uneven fixture를 모두 돌린다. source diff에서 예상한 첫 변화 지점과 실제 첫 변화 지점이 맞는지 확인한다. 성능 regression과 memory 변화도 기록하지만 correctness와 별도 판정한다.

**조사 보고서는 원인과 증거를 분리한다.** 원인은 “Trainer 문제”처럼 구성요소 이름으로 쓰지 않는다. 예컨대 “commit X의 함수 Y가 `num_items_in_batch` 없이 local mean을 반환하고 wrapper Z가 rank mean을 수행해 valid target당 weight가 rank별로 달라졌다”라고 쓴다. 증거에는 concatenated reference, rank별 S/N, gradient projection, 수정 test 좌표를 넣는다.

영향 범위에는 어떤 runs와 checkpoints가 해당 configuration을 사용했는지 쓴다. scalar logging만 잘못된 것인지 실제 gradient도 달랐는지 구분한다. 전자라면 model을 폐기할 필요가 없고 후자라면 trajectory가 달라졌다. rollback 가능한 checkpoint와 data cursor, 재학습 범위를 제시한다.

**설명을 직관과 증명으로 동시에 유지한다**

**확률 질량의 직관.** softmax는 vocabulary 후보에 질량 1을 나누어 준다. 정답 loss는 정답에 배정되지 않은 질량을 벌한다고 볼 수 있다. gradient `p-e_y`는 오답에 놓인 질량을 빼고 정답으로 옮기는 방향이다. 이 직관은 한 logit 행을 이해하는 데 유용하다.

그러나 모델이 실제로 probability mass를 물리적으로 이동시키는 것은 아니다. optimizer는 모든 examples의 parameter gradient를 합쳐 weight를 바꾸고 다음 forward에서 분포가 달라진다. 한 token의 개선이 다른 contexts의 분포를 악화할 수 있다. 직관 뒤에 Jacobian과 shared parameters를 붙여야 과장이 되지 않는다.

**지형의 직관.** logit 공간에서 공통 이동 방향은 평평하고 class 차이 방향에 곡률이 있다. 확률이 한 class에 몰리면 일부 방향의 곡률이 작아진다. 이 그림은 label smoothing과 z-loss가 무엇을 바꾸는지 설명한다. 하지만 deep network parameter 공간은 비선형 composition과 symmetry가 있어 이 단순 bowl이 아니다.

**분모의 직관.** denominator는 각 token contribution을 전체 update에서 얼마만큼의 표로 셀지 정하는 투표 규칙이다. token mean은 각 유효 token에 한 표, sequence mean은 각 sequence에 한 표를 준다. domain-balanced mean은 domain마다 표 총량을 맞출 수 있다. 직관은 정책 선택을 드러내지만 실제 구현은 sum과 collective 식으로 증명해야 한다.

**좌표계의 직관.** tokenizer는 텍스트를 모델이 예측하는 사건으로 나눈다. 지도에서 격자를 바꾸면 같은 땅도 셀 개수와 경계가 달라지는 것과 비슷하다. PPL 단위가 tokenizer에 의존하는 이유를 설명한다. 그러나 token은 의미 원자의 보장이 없고 byte와 grapheme 경계를 깨뜨릴 수 있으므로 원문 offset 검증이 필요하다.

좋은 설명은 직관에서 멈추지 않는다. 직관마다 유효 범위, 반례, 수식, source 좌표, fixture를 붙인다. 독자가 기억하기 쉬운 그림과 장애 때 사용할 수 있는 증거가 함께 있어야 한다. 이 원칙은 이후 embedding, attention, optimizer, 분산 학습 장에서도 반복된다.

**softmax 곡률을 update 안정성과 연결한다**

**Hessian은 확률 simplex의 공분산이다.**

한 token의 logit \(z\)와 target \(y\)에 대해 gradient는 \(p-e_y\), Hessian은 \(\mathrm{diag}(p)-pp^\top\)다. 이 행렬은 categorical one-hot 변수의 공분산과 같다. 모든 logit에 같은 상수를 더하는 방향에서는 고윳값이 0이다. softmax가 score의 절대 위치가 아니라 차이만 본다는 사실이 곡률에 나타난다.

두 class만 보면 nonzero curvature는 확률이 비슷할 때 크고 한 class가 포화될수록 작아진다. 확신이 낮은 token은 logit 차이를 조금 바꿔도 probability가 크게 움직이고, 이미 극도로 확신한 token은 같은 logit 변화가 확률에 덜 반영된다. 그러나 잘못된 class에 포화된 경우 gradient \(p-e_y\)는 여전히 큰 방향을 가질 수 있다. “곡률이 작다”와 “학습 신호가 작다”를 혼동하지 않는다.

logit Hessian은 positive semidefinite지만 deep network parameter Hessian 전체가 convex라는 뜻은 아니다. \(z(\theta)\)의 Jacobian과 2차 항이 합쳐지고 parameter symmetry와 nonlinear composition이 들어온다. 이 절의 기하학은 output layer 주변의 local sensitivity를 설명하는 도구다.

optimizer 관점에서는 같은 scalar loss라도 logit margin과 probability 분포가 gradient 방향·curvature를 바꾼다. label smoothing, temperature, z-loss와 confidence penalty는 이 geometry를 의도적으로 수정한다. 11·12장에서 AdamW와 Muon이 parameter-space update를 다룰 때 output-space `p-q`가 Jacobian을 통해 어떻게 전달되는지 다시 연결한다.

**관측은 loss뿐 아니라 margin과 entropy를 포함한다.**

loss가 같은 두 token도 분포가 다를 수 있다. 한쪽은 정답과 한 오답이 경쟁하고 다른 쪽은 많은 오답에 질량이 퍼져 있을 수 있다. target probability, top-1/top-2 margin, entropy와 logit norm을 bounded slice로 기록한다. vocabulary 전체 histogram을 매 step 전송하지 않는다.

logit norm이 계속 커지는데 accuracy와 NLL 개선이 작다면 scale 방향이 calibration과 numerical range를 악화시킬 수 있다. z-loss 같은 regularizer를 고려하기 전에 tokenizer·mask, label noise와 evaluation calibration을 확인한다. regularizer는 증상을 가리는 새 objective가 될 수 있다.

**작은 곡률 fixture.** vocabulary 3개에서 logits를 `[0,0,0]`, `[8,0,0]`, `[0,8,0]`로 두고 target 0에 대한 probability, gradient, Hessian eigenvalue를 FP64로 계산한다. 공통 상수 100을 더해 결과가 같은지 본다. 불안정한 naive exponent 구현은 이 invariant를 깨거나 overflow한다.

### vocabulary-parallel CE를 full-logit oracle과 비교한다

vocabulary가 tensor-parallel rank에 나뉘면 한 rank는 전체 logits를 갖지 않는다. 그렇다고 rank-local softmax를 계산해 loss를 평균낼 수 없다. 각 rank의 local maximum에서 전역 maximum을 all-reduce하고, 안정화한 exponential sum을 전역으로 합쳐 같은 log-sum-exp를 만들어야 한다.

\[ m=\max_r\max_{j\in V_r}z_j, \qquad \log Z=m+\log\sum_r\sum_{j\in V_r}e^{z_j-m} \]

target logit은 target ID를 소유한 rank만 local 값을 선택하고 다른 rank는 중립 값을 내어 전역 reduce한다. vocabulary padding이나 uneven shard가 있으면 padded class가 분모에 들어가지 않게 mask한다. target ID→owner mapping과 shard offset을 exact fixture로 검증한다.

gradient도 global \(p-e_y\)와 같아야 한다. 각 shard는 자기 class probability를 계산하고 target owner만 1을 뺀다. 최종 hidden-state gradient는 sharded output projection의 contributions를 합쳐야 한다. scalar loss parity만 보면 wrong-owner와 compensating reduction을 놓칠 수 있으므로 selected logit gradient와 hidden gradient를 비교한다.

**실패 주입.** target이 shard 경계의 첫·마지막 ID인 fixture, vocabulary size가 rank 수로 나누어지지 않는 fixture, 극단적으로 큰 logit이 target이 아닌 다른 rank에 있는 fixture를 둔다. local max만 쓰면 overflow·잘못된 normalization이 나타나고 owner offset이 틀리면 target NLL이 달라진다.

통신 최적화가 max, sum과 target gather를 fuse하더라도 수학적 collective contract는 남는다. 15장의 tensor-parallel ownership, 29장의 collective sequence ledger와 연결한다. 한 rank가 zero-valid batch라고 loss collective를 건너뛰면 다른 rank가 hang할 수 있으므로 coordinated graph-connected zero 또는 global skip을 사용한다.

**target distribution을 바꾸는 목적함수를 비교한다**

**label smoothing은 one-hot target을 혼합한다.**

일반적인 smoothing은 \(q=(1-\varepsilon)e_y+\varepsilon u\) 형태다. \(u\)가 전체 vocabulary 균등인지 정답을 제외한 오답 균등인지 구현에 따라 다르다. gradient는 \(p-q\)가 되어 정답 logit을 무한히 벌리는 압력을 줄이고 오답 class에도 작은 target mass를 준다.

ignore position, class weight와 smoothing이 결합될 때 denominator와 적용 순서를 source에서 확인한다. padding class를 smoothing support에 넣는지, vocabulary padding을 제외하는지에 따라 objective가 달라진다. `label_smoothing_factor`라는 이름만으로 공식을 추측하지 않는다.

calibration이 좋아질 수 있다는 일반적 기대를 release 결과로 간주하지 않는다. NLL, expected calibration error, accuracy, rare-token과 structured-output slice를 평가한다. smoothing이 exact token prediction이나 logit margin을 약화할 수 있다. epsilon sweep은 data·optimizer를 고정한다.

**unlikelihood는 금지 target에 별도 압력을 준다.**

negative candidate \(c\)에 \(-\log(1-p_c)\)를 더하면 반복이나 금지 token probability를 낮추는 방향을 만든다. \(p_c\to1\)에서 gradient가 커질 수 있으므로 numerical stability, candidate generation과 coefficient를 확인한다. positive CE와 같은 token을 동시에 positive·negative로 표시하지 않는 data invariant가 필요하다.

sequence-level repetition candidate를 어떻게 만드는지가 objective만큼 중요하다. history window, n-gram, special token과 boundary가 바뀌면 negative set이 달라진다. generated candidate를 사용하면 policy/model generation도 lineage에 들어간다. negative denominator와 empty-candidate 정책을 기록한다.

**focal 계열은 쉬운 예의 가중치를 줄인다.**

정답 probability에 \((1-p_y)^\gamma\)를 곱하는 형태는 class imbalance나 hard example 집중에 쓰일 수 있지만 gradient에는 weight derivative도 들어간다. 단순히 CE gradient에 상수 weight를 곱한 것과 다르다. 구현이 weight를 detach하는지까지 source에서 본다.

hard example을 강화하면 label noise와 outlier도 강화할 수 있다. data quality slice, gradient norm과 learning dynamics를 함께 본다. 기본 CE에서 발생한 mask·denominator 오류를 custom loss로 덮지 않는다. custom objective는 fixed reference, finite difference와 regression suite를 요구한다.

## 1.14 causal loss API와 failure contract를 종단 검증한다

Golden fixture를 실제 stack에 주입해 caller, shift owner, flatten, reduction과 CUDA backend를 통과시킨다. 정상 output뿐 아니라 double shift·wrong ignore ID·empty denominator가 어느 경계에서 실패해야 하는지 계약한다.

현재 snapshot에서 `ForCausalLMLoss`가 존재한다는 사실만으로 모든 causal model이 같은 경로를 실행한다고 말할 수 없다. 모델의 `forward`가 `self.loss_function`을 언제 호출하고 `labels`, `vocab_size`, `num_items_in_batch`와 `shift_labels`를 어떻게 넘기는지 확인한다. custom model과 remote code는 kwargs를 버릴 수 있다.

Trainer는 batch grouping과 gradient accumulation에서 loss kwargs를 준비한다. `num_items_in_batch`가 실제 valid causal target count인지, sequence count인지, collator의 ignore mask를 반영하는지 test 좌표와 runtime probe로 검증한다. 이름보다 producer와 consumer의 계약이 중요하다.

`ForCausalLMLoss`는 logits dtype을 올리고 shift를 소유할 수 있다. caller가 이미 shifted labels를 넘기면서 `shift_labels`를 명시하지 않으면 double shift가 생긴다. 반대로 pre-shifted custom loss에서 자동 shift를 끄지 않으면 원문 span과 target이 어긋난다. test의 expected target indices를 사람이 읽을 수 있는 row로 연결한다.

`fixed_cross_entropy`가 sum/count 경로를 선택하면 denominator 0과 device·dtype를 확인한다. accumulation wrapper가 반환 loss를 다시 accumulation steps로 나누는지, model의 `accepts_loss_kwargs` 같은 capability가 실제 호출을 바꾸는지 source diff로 본다. scalar logging 보정과 backward scale 보정을 분리한다.

runtime probe는 logits와 labels를 detach해 독립 FP64 oracle의 numerator/count를 계산한다. model loss, local numerator/count, accumulated global value와 selected gradient projection을 비교한다. probe가 `.item()` synchronization이나 graph retention으로 성능·memory를 바꾸지 않게 bounded golden window에서만 실행한다.

revision upgrade에서는 함수 line만 다시 찾지 않는다. signature, shift owner, dtype cast, reduction, caller kwargs와 Trainer test를 semantic diff한다. expected artifact를 먼저 갱신하지 않고 old/new GoldenBatchID에서 first divergence를 확인한다.

### artifact·source·fixture·trace가 합의할 때 incident를 닫는다

증거 사슬은 원문→rendered bytes→tokens→labels의 exact alignment에서 시작한다. 이어 logits에서 독립 계산한 token별 NLL, numerator와 denominator를 두고, selected logit·parameter gradient와 committed update까지 연결한다. 마지막에는 source revision, 호출 branch와 regression fixture를 붙인다.

loss curve만 다른 사건은 첫째·둘째를 비교해 data/objective를 가른다. scalar는 같은데 trajectory가 다르면 셋째에서 gradient scaling·alias·optimizer로 넘어간다. checkpoint 이후만 다르면 same GoldenBatchID와 RNG를 고정한다. 분산에서만 다르면 global numerator/count와 collective sequence를 본다.

수정은 expected first divergence를 없애고 정상·negative fixture를 모두 통과해야 한다. tolerance를 넓히거나 offending row를 조용히 삭제해 닫지 않는다. data row가 잘못됐다면 lineage와 affected checkpoints를 query한다. logging만 틀렸다면 model trajectory 영향이 없음을 gradient·state로 증명한다.

보고서에는 원인, 지지·반박 evidence, 영향 run/artifact, 안전한 완화, 근본 수정과 남은 risk를 구분한다. “loss bug”라는 이름 대신 어떤 producer가 어떤 denominator·shift·mask state를 잘못 만들었는지 쓴다.

이 네 증거는 2장의 backward/update, 5장의 tokenizer/template, 6장의 packing, 15장의 sharded vocabulary, 26장의 observability, 28장의 golden run으로 양방향 연결된다. 독자는 어느 장에서 시작해도 같은 TokenFixtureID와 LossManifest를 따라갈 수 있다.

### teacher forcing과 autoregressive generation의 조건 분포를 구분한다

훈련의 다음 토큰 loss는 실제 prefix를 조건으로 정답 token을 예측한다. 생성에서는 모델이 뽑은 token이 다음 prefix가 된다. 앞선 오류가 context를 바꾸어 이후 분포까지 이동하는 현상을 흔히 exposure 차이로 설명한다. 그러나 이것이 causal loss 구현 오류를 뜻하지는 않는다.

teacher-forced NLL과 free-running generation은 서로 다른 평가다. 같은 checkpoint에서 고정 prompt의 token별 teacher-forced NLL, greedy rollout과 sampled rollout을 분리한다. 생성이 실패했는데 NLL이 낮다면 error compounding, decoding, stop/template와 task metric을 본다. NLL부터 틀리면 generation 전략을 조정하기 전에 data·model을 조사한다.

scheduled sampling 같은 방법은 일부 prefix를 model token으로 바꿔 training distribution과 objective를 수정한다. 어느 token을 언제 바꾸는지, gradient가 sampled decision을 통과하는지, RNG와 curriculum을 manifest에 넣는다. 기본 causal LM과 같은 recipe로 취급하지 않는다.

sequence-level objective나 RL은 생성 trajectory와 reward를 직접 사용하지만 token-level likelihood, KL와 importance ratio를 여전히 계산할 수 있다. 19·20장의 preference/RL로 넘어갈 때 SFT의 TokenFixtureID와 policy generation을 유지한다. “generation gap”을 이유로 objective 변경을 서두르기 전에 작은 rollout ledger를 만든다.

**진단 fixture.** 정답 prefix에서는 다음 token 확률이 높지만 첫 token을 틀리면 다른 attractor로 가는 tiny transition model을 만든다. teacher-forced NLL과 greedy sequence score가 왜 다른지 손으로 계산한다. temperature·top-p를 바꿔도 underlying logits가 같음을 확인한다.

**sampled vocabulary와 근사 loss의 계약을 점검한다**

큰 vocabulary의 full softmax는 모든 class logit과 normalization을 계산한다. sampled softmax, noise-contrastive 계열이나 adaptive output은 계산을 줄이지만 objective와 estimator를 바꿀 수 있다. full CE와 이름만 다른 동일 연산이라고 설명하지 않는다.

negative sampling distribution, sample 수, correction term, target 포함과 duplicate 처리를 기록한다. proposal probability를 보정하지 않으면 frequent/rare class의 effective weight가 달라진다. RNG와 sampler state가 checkpoint·resume에 포함된다.

small vocabulary fixture에서는 full normalization을 정확히 계산하고 sampled estimator의 expectation·variance를 반복 측정한다. 한 sample result가 full CE와 같아야 한다고 요구하지 않는다. estimator가 주장한 unbiased/bias contract와 gradient 방향을 본다.

distributed sampled vocabulary에서는 rank별 negative set, dedup와 ownership을 합의해야 한다. 동일 negative를 여러 rank가 중복 count하거나 target owner를 놓치지 않는다. communication 절감과 estimator variance를 함께 보고한다.

production LLM이 full vocabulary projection을 쓰는 path라면 sampled loss 논의를 실제 사용 경로처럼 표현하지 않는다. model config와 forward branch에서 actual output head를 확인한다. 이 절은 대안 objective를 검토하는 방법과 source audit 항목을 제공한다.

**perplexity와 calibration을 서로 다른 축으로 측정한다**

perplexity는 token-average NLL의 지수다. 같은 tokenizer·normalization·mask·context와 denominator에서만 직접 비교한다. 문서 stride와 overlapping window에서 token을 중복 count하는지, 첫 context token과 EOS를 어떻게 처리하는지 평가 code에서 확인한다.

calibration은 예측 confidence와 실제 correctness의 관계다. next-token accuracy·confidence bin, NLL와 Brier-like score가 서로 다른 면을 본다. vocabulary가 매우 크고 sequence 의존적이므로 binning·slice와 sample size를 기록한다.

temperature scaling은 logits를 \(T\)로 나눠 calibration을 조정하지만 ranking이 유지될 수 있다. 학습 objective를 바꾸는 것이 아니라 evaluation/serving transform일 수 있다. fitted temperature의 calibration set과 deployment config를 artifact로 둔다. generation temperature와 calibration temperature를 이름만 같다고 혼동하지 않는다.

rare token, multilingual, structured syntax와 long context에서 calibration을 slice한다. 전체 ECE가 좋아져도 safety-critical token margin이 나빠질 수 있다. calibration 향상을 accuracy·safety 개선으로 확대 해석하지 않는다.

**next-token loss가 지식을 어디에 저장하는지 과장하지 않는다**

한 사실 문장이 낮은 loss가 됐다고 특정 parameter나 neuron에 그 지식이 독립 저장됐다는 뜻은 아니다. gradient는 shared embedding, attention, MLP와 output head의 Jacobian을 따라 분산된다. 같은 parameter update가 많은 contexts의 logits를 바꾼다.

한 example의 influence를 보려면 update 전후 해당 prompt와 paraphrase, neighborhood, unrelated control의 logit·loss를 비교한다. parameter delta norm과 output-space change를 연결한다. training row exact recall만으로 일반화된 지식 주입을 주장하지 않는다.

continual editing과 unlearning은 23장에서 다루지만 출발점은 같은 LossManifest다. edit target, locality와 retention set이 어느 token events를 평가하는지 고정한다. n-gram/engram 형태의 memory도 tokenizer·context와 objective 경계 없이 설명할 수 없다.

기하학적으로 한 update는 여러 examples의 Jacobian row가 만드는 parameter-space 방향을 합친다. gradient가 비슷한 examples는 협력하고 반대 방향이면 간섭한다. data curriculum과 optimizer가 이 합을 시간에 따라 바꾼다. 단일 token의 `p-e_y` 직관을 전체 knowledge representation으로 곧바로 확대하지 않는다.

**실험.** 작은 model에서 한 fact row를 한 step 업데이트하고 target, paraphrase, same-subject conflict와 unrelated prompts의 logits를 저장한다. update를 되돌리고 control row를 학습해 차이를 비교한다. seed·optimizer·denominator를 고정한다. 이 실험은 메커니즘 probe이지 대형 model 지식 저장의 일반 법칙이 아니다.

**데이터 가중치를 objective measure로 해석한다**

token 평균은 관측한 token empirical distribution에 대한 기대값이다. domain-balanced loss, per-example weight와 curriculum은 이 measure를 바꾼다. 단순 training trick이 아니라 어느 예측 사건에 얼마의 질량을 줄지 정하는 objective 정책이다.

sample weight가 있으면 token NLL에 적용하는지 sequence mean에 적용하는지 명시한다. 길이가 긴 sample에 sequence weight를 그대로 각 token에 곱하면 총 영향이 길이에 비례할 수 있다. 먼저 sample 내부 reduction을 하고 weight를 적용하는 방식과 결과가 다르다.

domain별 loss를 같은 비중으로 평균하면 작은 domain token 하나가 큰 domain token보다 큰 weight를 가질 수 있다. 이것이 의도한 균형인지, batch sampler가 이미 oversampling해 이중 가중되는지 확인한다. actual draw와 loss weight를 함께 기록한다.

curriculum schedule은 optimizer step 또는 consumed token에 따라 measure를 바꾼다. resume에 schedule state와 mixture cursor가 필요하다. scheduler clock과 curriculum clock을 혼동하지 않는다. 6장의 mixture ledger와 13장의 learning-rate schedule을 같은 committed update에 연결한다.

**검산 fixture.** 길이 1과 9인 두 sequence, domain weight 1과 3을 두고 token mean, sequence mean, domain-balanced mean을 손으로 계산한다. collator·Trainer와 distributed reduction이 선택한 수식과 exact 일치해야 한다. scalar뿐 아니라 gradient projection을 비교한다.

### shape·dtype·ignore·denominator failure를 fail-fast한다

정상 input만 테스트하지 않는다. logits vocabulary와 config vocab이 다름, label out of range, all-ignore, empty sequence, non-contiguous view, dtype 극단, shifted label 길이 mismatch와 zero denominator를 넣는다. 각 경우 명시적 error 또는 정의된 skip 상태를 가진다.

silent clamp나 out-of-range ignore는 데이터·tokenizer 결함을 숨긴다. error message는 batch/row identity, shape, valid min/max와 source revision을 제공하되 원문·secret을 노출하지 않는다. stable reason code로 playbook과 연결한다.

compiled/fused path에서도 같은 negative fixture가 같은 semantic gate를 통과해야 한다. eager가 거부한 malformed target을 fused kernel이 메모리 오류로 바꾸지 않는다. backend별 exact error 문구보다 reason과 no-commit invariant를 본다.

failure 뒤 parameter·optimizer·scheduler와 data cursor가 전진했는지 확인한다. forward 중 error와 backward/collective 중 error는 cleanup이 다르다. 분산에서는 모든 rank가 coordinated abort 또는 skip으로 이동한다. 한 rank만 exception을 삼키지 않는다.

**loss 계산의 메모리와 bandwidth를 설명한다**

logits tensor 크기는 batch token 수 \(T\), vocabulary \(V\), element bytes \(b\)에 대해 대략 \(TVb\)다. 큰 vocabulary에서는 LM head output과 loss materialization이 memory·bandwidth의 큰 부분이 된다. sequence 길이만 보고 activation memory를 설명하면 이 항을 놓친다.

chunked projection/loss, fused linear-cross-entropy와 vocabulary parallel은 full logits 생존을 줄일 수 있다. 그러나 global log-sum-exp, target logit, gradient와 reduction contract를 보존해야 한다. metric·generation이 logits를 필요로 하면 별도 path와 비용을 명시한다.

최적화 비교는 같은 TokenFixtureID와 numerator/count, selected gradient와 parameter delta를 먼저 통과한다. 그 뒤 peak allocated, HBM traffic, kernel launch와 valid token/s를 비교한다. logits를 저장하지 않아 observer가 약해지면 bounded debug mode와 source-level proof를 마련한다.

**다음 장으로 넘기는 LossManifest schema**

manifest에는 tokenizer/template/data revision, BatchID, logits shape·dtype·sharding, shift owner, target/mask/weight policy, loss components, numerator·denominator와 reduction scope가 있다. model caller, loss function과 Trainer producer의 fixed source coordinate를 연결한다.

runtime state에는 microbatch·accumulation window, local/global valid count, scaler state, selected logit gradient와 parameter alias를 둔다. exact/numerical invariant, tolerance와 observer overhead를 기록한다. zero-valid와 failure reason도 schema 일부다.

2장은 이 manifest를 받아 backward graph, gradient accumulation, AMP unscale·clip과 optimizer commit을 검증한다. 5·6장은 tokenizer/template와 packing이 target set을 어떻게 만들었는지 역추적한다. 15·29장은 sharding과 collective가 global objective를 보존하는지 확인한다.

### off-by-one을 token ID가 아닌 원문 offset까지 역추적한다

causal shift는 tensor slice 두 줄처럼 보이지만 tokenizer가 만든 special token과 template boundary를 알지 못하면 검증할 수 없다. fixture table에 원문 span, rendered byte offset, token index, logit position, target token과 loss mask를 한 행씩 둔다.

BOS가 position 0에 추가되면 첫 예측 target과 원문 첫 token의 관계가 바뀐다. EOS를 response 끝에 넣으면 마지막 response token의 logit이 EOS를 예측할 수 있다. generation prompt를 붙이면 training-only template와 target set이 달라질 수 있다. special token 자동 추가와 template 내 수동 삽입을 중복하지 않는다.

left padding은 position IDs와 logits slice를, right padding은 마지막 valid token과 generation start를 다르게 다룰 수 있다. attention mask가 padding을 숨겨도 label이 ignore가 아니면 loss에는 참여할 수 있다. 반대로 attention 가능한 prompt token을 label ignore하는 assistant-only objective는 정상이다.

packed documents에서는 physical adjacency와 semantic continuation을 구분한다. EOS target을 유지할지, 새 문서 첫 token을 앞 문서에서 예측하지 않게 mask할지 policy를 table로 드러낸다. block-diagonal attention과 label boundary를 각각 검사한다.

오류를 찾을 때 decoded string만 보지 않는다. byte fallback, normalization과 whitespace token 때문에 decode round-trip이 원문 경계를 흐릴 수 있다. tokenizer offset mapping과 raw bytes, token ID를 함께 보존한다. 5장의 tokenizer fixture를 재사용한다.

**15분 triage에서 최초 divergence를 확보한다**

첫 3분에는 RunID, model/tokenizer/data/config digest, last normal step과 affected range를 고정한다. loss가 NaN인지 상승인지 plateau인지, valid denominator와 committed update가 전진하는지 본다. telemetry stale을 loss 0으로 읽지 않는다.

다음 5분에는 한 affected BatchID를 확보한다. token·label·mask, length/packing/domain과 loss components를 정상 batch와 비교한다. raw 민감 text는 접근 제한 artifact로 두고 summary·digest를 사용한다. all-ignore, out-of-range와 double shift를 exact assertion한다.

남은 시간에는 model loss와 독립 FP64 numerator/count를 비교한다. 같으면 forward logits와 data distribution, 다르면 loss wrapper·reduction을 조사한다. selected gradient와 committed update를 확인해 logging-only인지 training trajectory 영향인지 가른다.

분산 run이면 rank별 S/N과 collective sequence를 추가한다. count가 불균등한지, zero-valid rank가 branch를 건너뛰었는지 본다. concatenated single-process reference로 global scalar와 gradient를 검산한다.

즉시 완화는 evidence를 보존한 뒤 한다. bad shard 격리는 lineage와 skipped count를 남기고, checkpoint rollback은 data cursor와 generation을 확인한다. learning rate를 임의로 낮춰 NaN을 숨기거나 threshold를 올려 경보만 끄지 않는다.

**attention mask와 loss mask의 독립성을 실험한다**

attention mask는 어떤 key/value 위치를 context로 읽을 수 있는지 정하고, loss mask는 어떤 target 위치가 objective에 기여하는지 정한다. 둘은 shape가 비슷해도 같은 tensor가 아니다. prompt token은 attention에는 보이지만 assistant-only loss에서는 ignore일 수 있다.

네 칸의 진리표를 만든다. context에도 보이고 loss에도 참여하는 response 내부, context에는 보이지만 loss에서는 제외되는 prompt, context와 loss 모두 제외되는 padding, 정책 오류로 context에서는 막혔지만 loss target이 남은 위치를 각각 fixture로 둔다. 마지막 상태는 model이 필요한 prefix를 못 보면서 target을 맞히도록 요구하는 모순이다.

causal mask, padding mask, document block mask와 label ignore를 단계별로 시각화한다. additive mask의 0/negative-infinity convention과 boolean polarity를 source에서 확인한다. SDPA·Flash attention backend가 mask shape를 변환하거나 causal flag와 결합하는 경로를 8장과 연결한다.

loss만 바뀌고 logits가 같은 실험은 label mask 문제를 고립한다. attention mask만 바꾸면 첫 attention output부터 달라져야 한다. 둘을 동시에 바꾸고 final loss만 보면 원인을 잃는다. expected first divergence를 먼저 적는다.

packed boundary에서 block attention과 cross-document labels를 독립적으로 toggle해 네 조합을 실행한다. 의도한 EOS 학습 정책과 invalid continuation을 구분한다. valid denominator와 per-boundary NLL을 기록한다.

runtime probe는 mask tensor 전체를 매 step 저장하지 않고 GoldenBatchID에서 exact bitmap과 summary count를 남긴다. production에는 padding ratio, valid targets와 boundary count를 bounded metric으로 둔다. sample 원문은 metric label에 넣지 않는다.

mask dtype과 값 범위도 numerical contract다. 낮은 정밀도에서 충분히 큰 음수를 사용하더라도 backend가 softmax 전에 어떤 dtype으로 더하는지 확인한다. 모든 key가 masked된 query row는 softmax 분모와 NaN 처리에 특별한 경계다. padding query output이 뒤의 residual이나 loss에 남는지 본다.

Transformers model별 mask preparation helper, attention module의 causal option과 loss 함수의 ignore 처리는 서로 다른 source 좌표다. 공통 이름만 보고 동일 구현이라고 가정하지 않는다. selected architecture에서 caller→helper→kernel dispatch를 따라 actual mask representation을 기록한다.

실패 fixture는 boolean polarity 반전, 한 칸 off-by-one causal diagonal, padding side 변경과 all-masked row를 포함한다. eager·SDPA·Flash candidate가 선언한 exact/numerical invariant를 만족하는지 비교한다. unsupported mask가 fallback되면 actual backend와 성능을 기록한다.

수정 뒤에는 output loss만 아니라 attention probability 또는 selected output boundary, token NLL, valid count와 gradient를 비교한다. mask 오류가 학습되지 않은 token을 분모에 넣었으면 영향 RunID와 checkpoints를 data lineage에서 찾는다. 단순 metric display 오류인지 trajectory 변경인지 committed update evidence로 판정한다.

인수자는 임의의 유효 target 하나와 ignore target 하나를 골라 원문 span부터 logit 위치, NLL, numerator와 gradient까지 역추적한다. 이어 임의의 loss spike에서 BatchID, mask·count, source branch와 optimizer update를 찾는다. 두 방향의 추적이 같은 LossManifest에서 만나야 한다.

이 시험이 통과하면 다음 장은 scalar loss를 다시 해석하지 않고 autograd가 만든 gradient 경로와 update 원자성에 집중할 수 있다. 실패하면 backward 설정을 바꾸기 전에 tokenizer·mask·shift와 reduction 계약부터 고친다. 목적함수의 좌표가 틀린 상태에서 optimizer를 조정하는 것은 잘못된 지도를 더 빠르게 따라가는 일이다.

마지막 기록에는 검토한 Transformers와 모델 구현의 commit, loss caller와 test 좌표, GoldenBatchID, exact·numerical 판정과 미실행 backend를 적는다. 새 revision이나 tokenizer를 도입하면 같은 fixture를 다시 실행하고, 과거 PASS를 자동 상속하지 않는다. 변경된 첫 경계와 영향 artifact를 새 evidence generation으로 보존한다.

검토자와 실제 실행 시각, environment digest와 남은 예외의 책임자·만료 조건도 반드시 함께 기록해 다음 재검증의 출발점을 고정한다.

**1장의 검증 결과를 다음 단계로 인계한다**

**봉인 전 확인한다.** 원문에서 labels까지의 변환이 결정적이고 revision이 고정되었는가. shift owner가 하나인가. attention mask와 label mask를 구분했는가. numerator와 denominator를 재계산할 수 있는가. local·accumulation·global reduction이 하나의 식과 일치하는가. zero-valid 상태가 정의되었는가.

logits 경로에서는 LM head shape와 tying, dtype 전이, stable log-sum-exp, vocabulary shard collective를 확인한다. loss 경로에서는 target distribution, ignore, weights, auxiliary components를 확인한다. backward 경로에서는 selected gradient와 alias 합산을 확인한다. source 경로에서는 commit, symbol, test assertion의 범위를 확인한다.

artifact에는 최소 하나의 정상 fixture와 다섯 failure injections가 있다. 정상 fixture는 FP64 reference와 맞는다. double shift, wrong mask, wrong denominator, broken alias, zero-valid rank가 각자 예상 detector에서 실패한다. 수정 뒤 모든 fixture가 통과하고 first-difference map이 예상과 맞는다.

이 봉인은 “코드가 영원히 옳다”는 선언이 아니다. 어떤 revision과 입력, 상태에서 무엇을 검증했는지 경계를 닫는 일이다. 다음 revision에서는 같은 fixture와 영향 반경 표를 다시 적용한다. 독자는 이 반복 가능한 절차 덕분에 framework가 바뀌어도 함수 이름을 외우는 대신 목적함수 계약을 복원한다.

2장에 넘기는 마지막 값은 scalar loss 하나가 아니다. loss numerator와 valid count, normalized scalar, shift와 reduction owner, selected logit gradient, parameter alias, accumulation window, source evidence가 한 묶음이다. 2장은 이 묶음을 받아 autograd가 모든 경로를 합치는지, AMP가 수치 범위를 지키는지, DDP가 전역 gradient를 만드는지, optimizer가 정확히 한 번 update를 commit하는지 검사한다.

**독자가 기억할 단 하나의 흐름**

문자열은 tokenizer를 거쳐 ID가 되고, ID는 embedding과 transformer를 지나 hidden state가 되며, LM head는 vocabulary마다 조건부 score를 만든다. softmax는 score 차이를 확률 simplex의 한 점으로 옮기고, 교차엔트로피는 target distribution과의 차이를 scalar와 `p-q` gradient로 바꾼다. mask와 weight는 어느 예측 사건이 얼마나 참여하는지 결정하고, denominator는 그 사건들의 상대적 표를 결정한다.

이 흐름에서 어느 단계도 혼자 완결되지 않는다. tokenizer가 경계를 바꾸면 target과 count가 바뀐다. packing이 위치를 바꾸면 context와 mask가 바뀐다. LM head tying이 깨지면 같은 loss에서도 update 경로가 바뀐다. rank별 count가 다르면 DDP의 gradient 평균만으로 global token mean이 되지 않는다. fused kernel이 logits를 생략하면 memory는 줄지만 global log-sum-exp와 target gradient의 보존 책임은 남는다.

따라서 문제가 생겼다고 모델이 “언어를 이해하지 못했다”고 바로 결론내리지 않는다. 먼저 raw span과 target을 맞춘 뒤 logits와 위치별 NLL, numerator와 count, selected gradient를 순서대로 확인한다. 최초로 달라진 경계가 tokenizer면 데이터를, logit이면 model forward를, scalar면 reduction을, gradient 이후면 2장의 update state machine을 판다.

반대로 이 순서가 모두 맞았다고 downstream 품질이 자동으로 보장되는 것도 아니다. 그것은 지정한 목적함수가 코드대로 학습되었다는 증거다. 데이터 선택이 적절한지, 선호와 안전을 반영하는지, benchmark가 실제 사용을 대표하는지는 이후 장의 별도 질문이다. 정확한 plumbing과 좋은 목적은 둘 다 필요하며 서로를 대신하지 않는다.

이 장을 다시 펼칠 시점은 loss가 NaN일 때만이 아니다. tokenizer 교체, chat template 수정, context 확장, packing 변경, gradient accumulation 변경, GPU 수 변경, loss fusion, vocabulary parallelism, checkpoint 재개마다 돌아와야 한다. 모든 변경이 `target→logit→numerator/count→gradient`의 어느 상태를 바꾸는지 표시하면 옵션 목록은 원인과 효과가 있는 설계도가 된다.

최종 산출물은 한 문장으로 요약된다. 임의의 유효 target 하나를 raw byte에서 global gradient까지 정방향으로 추적하고, scalar에서 raw byte까지 역방향으로 되짚으며, 그 사이 모든 함수와 상태를 고정 revision의 근거로 설명할 수 있어야 한다. 이 왕복이 가능하면 다음 토큰 학습은 더 이상 검은 상자가 아니다.

검토자는 마지막으로 다른 사람이 같은 dossier만 받아 독립적으로 loss를 재계산하게 한다. 구두 설명이나 작성자의 기억이 필요하면 증거 묶음이 아직 닫히지 않았다. 재계산자는 source commit과 fixture checksum을 확인하고 원문 좌표, target bitmap, numerator, denominator, gradient projection을 순서대로 복원한다. 결과가 맞으면 artifact를 봉인하고, 다르면 최초로 모호한 필드를 schema에 추가한다. 이 독립 재현 절차가 책의 설명을 실제 운영 지식으로 바꾸는 마지막 단계다.
**next-token 목적함수를 확률공간의 투영으로 본다**

문맥 `c`에서 vocabulary 크기가 `V`라 하자. 모델은 logit vector `z∈R^V`를 만들고 `p_i=exp(z_i−m)/Σ_j exp(z_j−m)`로 simplex 내부의 확률을 만든다. `m=max_j z_j`를 빼는 것은 확률을 바꾸지 않으면서 overflow를 막는다. 모든 logit에 같은 상수 `a`를 더해도 `p`가 같은 이유는 softmax가 `z`의 공통 이동 방향을 식별하지 않기 때문이다.

정답 분포를 `q`라 하면 한 위치의 cross entropy는 `L=−Σ_i q_i log p_i`이고 logit gradient는 `∂L/∂z=p−q`다. one-hot target이면 정답 좌표에는 `p_y−1`, 나머지에는 `p_i`가 흐른다. gradient 합은 0이므로 공통 이동 방향으로는 update 신호가 없다. 이 성질은 logits에 상수를 더한 fixture와 gradient sum assertion으로 검사한다.

Hessian은 `H=diag(p)−ppᵀ`다. 임의 vector `v`에 대해 `vᵀHv=Σ_i p_i v_i²−(Σ_i p_i v_i)²`이므로 음이 아닌 가중 분산이다. 공통 상수 vector는 영공간에 있다. 확률이 한 class에 거의 집중되면 많은 방향의 곡률이 작아지고, 불확실한 분포에서는 class 간 이동 방향의 곡률이 커진다.

binary vocabulary에서는 margin `d=z_1−z_0` 하나로 충분하다. target이 class 1이면 `L=softplus(−d)`, `dL/dd=σ(d)−1`, `d²L/dd²=σ(d)(1−σ(d))`다. `d=0`에서 곡률은 `1/4`, 큰 양·음 margin에서 0에 가까워진다. 이 한 차원 그림을 다중 class의 pairwise logit 차이로 확장한다.

temperature `τ`를 쓰면 `p=softmax(z/τ)`이고 logit gradient에는 `1/τ`가 붙는다. 높은 temperature는 분포를 평평하게 만들지만 gradient scale과 curvature도 바꾼다. distillation에서 teacher와 student temperature를 바꾸고 `τ²` 보정을 사용하는 이유를 단순한 “부드러운 label”로만 설명하지 않는다.

label smoothing `ε`는 target을 `(1−ε)onehot(y)+εu`로 바꾼다. uniform `u`인지 vocabulary 일부의 prior인지 구현을 확인한다. ignore target과 padding 위치에는 smoothing 분포를 만들지 않아야 한다. smoothing을 켜면 최저 가능한 loss와 perplexity 해석이 달라진다.

z-loss는 log partition `logΣexp(z)`의 크기를 제어할 수 있다. softmax 확률은 공통 이동에 불변이지만 유한 정밀도와 downstream kernel은 logit scale에 영향을 받는다. z-loss coefficient, numerator와 denominator를 CE와 분리한다.

confidence penalty나 entropy regularization은 target CE와 다른 기하를 더한다. total scalar가 내려가도 CE component가 오를 수 있다. component별 loss sum, count, gradient norm을 기록한다. option 이름만으로 regularization 효과를 추정하지 않는다.

class weight는 empirical token measure를 바꾼다. rare token에 weight 10을 주면 그 위치가 valid target 열 개처럼 기여할 수 있다. weighted numerator와 weight sum을 denominator로 쓸지 target count를 쓸지 구현에 따라 objective가 다르다.

sample weight, sequence weight, token weight도 구분한다. 긴 sequence가 더 많은 target으로 기여하는 token mean과 각 sequence를 같은 비중으로 두는 sequence mean은 다른 population risk다. 학습 data mixture와 evaluation perplexity가 같은 measure를 쓰는지 확인한다.

geometric fixture는 `V=3`, logits `(−1,0,2)`, target 1로 시작한다. stable probability, loss, gradient, Hessian을 FP64로 계산한다. finite difference gradient와 Hessian-vector product를 autograd와 비교한다. logits에 `+10000`을 더하고 같은 결과를 기대한다.

다음 fixture에서는 logits를 permute하고 target ID도 같은 permutation으로 옮긴다. 이때 loss는 같아야 하지만 target만 옮기면 달라져야 한다. 이 시험은 tokenizer vocabulary remap에서 embedding과 LM head row를 함께 permute해야 하는 이유를 보여준다.

이어 target probability가 0.9인 easy token과 0.1인 hard token을 같은 batch에 두고 mean loss와 gradient norm 기여를 비교한다. hard token이 scalar와 update를 크게 지배할 수 있지만, 그렇다고 반드시 data 오류인 것은 아니다.

마지막 fixture에서는 같은 semantic text를 두 tokenizer가 각각 2 token과 5 token으로 나누게 한다. token mean은 더 잘게 쪼갠 표현에 다른 가중을 주며, byte-normalized 또는 word-level 평가는 별도 metric이다. 이는 tokenizer 교체를 model architecture 변경과 분리할 수 없는 이유다.

**원문에서 loss까지 tensor state를 한 줄씩 고정한다**

`RawDocumentID`에는 UTF-8 bytes와 normalization 전 checksum이 연결된다. decoder가 invalid bytes를 replacement character로 바꾸는지, newline과 Unicode normalization을 어떻게 처리하는지 기록한다. 문자열이 눈으로 같아도 bytes가 다를 수 있다.

pretokenizer는 whitespace, punctuation, byte fallback 경계를 만든다. BPE나 unigram tokenizer는 subword ID를 선택한다. token ID뿐 아니라 byte offset, special-token flag, normalized offset을 ledger에 둔다. offset을 제공하지 않는 added token은 별도 span 규칙을 갖는다.

chat template는 role marker, BOS, EOS, generation prompt를 삽입한다. next-token objective에서 이 marker도 일반 target이 될 수 있다. 어느 marker를 학습하고 어느 위치를 `ignore_index`로 가리는지 `LabelPolicyID`로 고정한다.

collator는 길이 제한, truncation side, padding side, sequence packing을 적용한다. 결과 tensor는 `input_ids[B,T]`, `attention_mask[B,T]`, `labels[B,T]`, 선택적으로 `position_ids`와 document boundary다. 각 차원과 dtype, device 이동 전후를 기록한다.

embedding lookup은 `E[input_ids]`로 `[B,T,D]`를 만든다. vocab resize가 빠지면 out-of-range error가 나거나 새 row가 존재하지 않는다. tied LM head라면 embedding storage와 output weight가 같은지 확인한다.

position encoding은 learned embedding, rotary transform, relative bias 등 architecture에 따라 state가 다르다. padding-free packing에서 position reset이 어떻게 표현되는지 본다. position 오류는 token ID가 같아도 첫 block부터 hidden을 바꾼다.

transformer block은 residual stream을 갱신한다. layer별 input/output shape, selected slice checksum, norm을 저장하면 최초 차이 layer를 찾을 수 있다. 모든 activation을 보존하지 않아도 `sum,sumsq,max,hash` probe로 이분 탐색할 수 있다.

final normalization을 거친 hidden `h[B,T,D]`가 LM head `W[V,D]`와 곱해져 logits `[B,T,V]`가 된다. tensor parallel이면 vocabulary axis가 shard될 수 있다. gathered full logits와 distributed log-sum-exp가 같은 확률을 만드는지 작은 fixture로 확인한다.

loss owner가 shift한다면 labels 오른쪽을 ignore로 pad한 뒤 `labels[...,1:]`를 사용하고 logits 위치 `t`가 원 token `t+1`을 맞힌다. collator가 pre-shift한다면 model loss는 추가 shift하지 않는다. owner는 정확히 하나여야 한다.

flatten은 `[B,T,V]→[B·T,V]`, labels `[B,T]→[B·T]`로 바꾼다. noncontiguous slice에는 `.contiguous()` 또는 reshape semantics가 필요할 수 있다. view 실패를 고치려고 잘못된 copy나 transpose를 넣지 않는다.

CE kernel은 log-sum-exp와 target gather를 계산한다. ignore 위치는 numerator와 denominator에서 빠져야 한다. `num_items_in_batch`가 있으면 local valid count가 아니라 accumulation/global contract의 count일 수 있다.

scalar loss가 반환되기 전에 numerator, valid count, reduction, output dtype을 기록한다. logging용 detached scalar와 backward에 들어가는 scalar를 구분한다. 2장은 latter의 exact tensor와 scale을 받는다.

**Transformers causal loss의 caller를 끝까지 추적한다**

기준 snapshot `550d7b3834670483a4df436541272c055dc364bf`에서 `src/transformers/loss/loss_utils.py:32`의 `fixed_cross_entropy`와 `:49`의 `ForCausalLMLoss`를 anchor로 둔다. line number, symbol, signature, function hash를 함께 저장한다.

그러나 실제 진입은 model-specific `forward`다. `labels is not None` branch가 `self.loss_function`을 호출하는지, `loss_type` 또는 config mapping이 어떤 callable을 고르는지 확인한다. remote-code model이 독자 CE를 쓰면 공통 helper 설명을 적용하지 않는다.

caller가 `logits`, `labels`, `vocab_size`, `num_items_in_batch`, `shift_labels`를 어떻게 넘기는지 trace한다. kwargs를 받지만 버리는 wrapper, `accepts_loss_kwargs` capability가 false인 custom model, Trainer가 count를 산출하지 않는 경로를 구분한다.

`ForCausalLMLoss`의 logits float 승격은 softmax/CE 계산 dtype을 바꾼다. model hidden과 LM head가 BF16이어도 loss 내부는 FP32일 수 있다. fused kernel이 이를 대체하면 actual accumulator dtype을 profiler와 source에서 확인한다.

labels pad-and-slice는 마지막 target을 ignore로 만든다. labels device 이동과 contiguous 변환이 어디서 일어나는지 본다. CPU labels와 accelerator logits 사이 암묵적 copy가 performance와 correctness에 미치는 영향을 분리한다.

`fixed_cross_entropy`가 count가 없을 때 mean, 있을 때 sum/count를 선택한다면 count dtype과 device, zero 처리, distributed wrapper의 scale을 확인한다. count가 Python int인지 device tensor인지 compile/capture 경로에도 영향을 준다.

model output의 `loss`는 scalar지만 `logits` 반환 shape와 cache branch가 training/inference에서 다를 수 있다. labels가 있을 때 모든 positions, 없을 때 마지막 position만 계산하는 구현과 Transformers 일반 경로를 혼동하지 않는다.

Trainer의 gradient accumulation은 model loss가 global token count로 이미 보정됐는지에 따라 추가 scale을 달리해야 한다. source의 model accept-loss-kwargs detection, accumulation count 수집, backward 호출을 같은 revision에서 읽는다.

local test는 mock logits를 직접 `ForCausalLMLoss`에 넣는 단계에서 시작한다. 이어 tiny model `forward`로 같은 logits를 만들고 Trainer first batch까지 통과시킨다. 이렇게 해야 어느 층에서 최초 차이가 생기는지 분리할 수 있다.

signature upgrade test는 loss function의 parameters, defaults, reduction branch를 이전 snapshot과 diff한다. unit fixture expected indices와 denominator를 그대로 재실행한다. 이름이 같아도 semantics가 바뀔 수 있다.

**nanoGPT의 pre-shift 경로를 tensor 표로 재구성한다**

고정 commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`의 `train.py:123–125`는 같은 memmap에서 `x`와 한 칸 뒤 `y` slice를 만든다. 이 좌표에서 shift owner는 loader다. `x=[s_0…s_{T−1}]`, `y=[s_1…s_T]`다.

`model.py:170`의 `GPT.forward(idx,targets)`는 `idx[B,T]`에서 token과 position embedding을 만든다. block과 final norm을 거쳐 hidden을 만든다. `targets is not None` branch는 LM head로 모든 position logits를 계산한다.

`model.py:187`의 `F.cross_entropy(logits.view(-1,V),targets.view(-1),ignore_index=-1)`는 추가 causal shift를 하지 않는다. Transformers의 pad-and-slice helper와 같은 labels tensor를 넣으면 objective가 다르다.

training branch의 logits는 `[B,T,V]`다. inference branch가 마지막 position `[:,[-1],:]`만 계산하면 `[B,1,V]`다. caller가 두 shape를 같은 것으로 가정하면 generation cache나 evaluation code가 틀릴 수 있다.

fixture sequence `0,1,2,3,4`에서 block size 4라면 `x=0,1,2,3`, `y=1,2,3,4`다. identity-like tiny model logits를 직접 구성해 네 target contribution을 출력한다. target 4를 놓치거나 target 0을 포함하면 shift가 틀렸다.

memmap start index sampling은 dataset measure의 일부다. document boundary를 모르는 연속 token stream이면 한 문서 끝에서 다음 문서 시작을 예측할 수 있다. EOS 삽입 여부와 sampling window crossing을 기록한다.

nanoGPT tokenizer/data preparation은 사용 dataset script에 따라 다를 수 있다. GPT-2 BPE ID를 쓰는 recipe와 char-level Shakespeare를 같은 token objective로 비교하되 token count와 perplexity 단위가 다름을 명시한다.

weight tying은 `transformer.wte.weight`와 `lm_head.weight` alias로 나타날 수 있다. state dict에 두 key가 있어도 optimizer logical parameter는 하나여야 한다. vocabulary row permutation은 두 logical views에 일관되게 적용한다.

compile path가 forward를 감싸면 source Python line이 실제 kernel launch 하나와 대응하지 않을 수 있다. graph trace와 output parity를 본다. reference eager path를 oracle로 보존한다.

nanoGPT source는 작은 명시적 경로를 제공하지만 padding, response mask, variable valid count의 production contract를 모두 구현하지 않는다. 교육 reference가 지원하지 않는 기능을 silent extension하지 않고 local wrapper의 책임으로 둔다.

**tokenizer부터 logits까지 최초 차이를 찾는 이분 탐색**

두 run A와 B를 비교할 때 raw bytes hash가 다르면 model을 보지 않는다. normalization, decode, data revision부터 고친다. bytes가 같고 rendered text가 다르면 template와 special token state를 본다.

rendered text가 같고 token IDs가 다르면 tokenizer files, normalizer, pretokenizer, added token, version을 본다. token ID가 같고 attention/labels가 다르면 collator, truncation, padding, mask policy를 본다.

input tensor가 같고 embedding output이 다르면 base tensor, vocab row, dtype, device kernel을 본다. embedding이 같고 첫 block부터 다르면 position state, dropout mode, attention mask, kernel을 본다.

중간 layer `k`까지 같고 `k+1`에서 다르면 그 block의 weights, normalization epsilon, attention/MLP dispatch, cache를 본다. layer probe를 모두 저장하기 어렵다면 절반 layer의 checksum을 비교해 범위를 이분한다.

final hidden이 같고 logits가 다르면 LM head weight, tie, vocabulary shard gather, bias를 본다. logits가 같고 loss가 다르면 labels shift, ignore, smoothing, weight, reduction, dtype을 본다.

loss scalar가 같아도 per-token contribution이 다를 수 있다. numerator vector 또는 selected positions를 비교한다. 상쇄된 평균이 두 오류를 숨길 수 있다.

fixed-token fixture가 같고 raw-text fixture만 다르면 tokenizer/template edge다. eager가 같고 compiled만 다르면 graph/kernel edge다. single GPU가 같고 distributed만 다르면 shard/reduction edge다.

훈련 첫 step은 같고 뒤에 갈라지면 RNG, dropout, sampler cursor, optimizer state를 본다. 이 장은 first forward/loss를 닫고 2장은 backward/update divergence를 이어서 찾는다.

probe 자체가 dtype cast나 synchronization으로 실행을 바꿀 수 있다. 관측용 hook의 side effect를 최소화하고 probe on/off 결과를 비교한다. checksum collision 가능성을 낮추기 위해 sum 하나가 아니라 multiple statistics와 selected values를 쓴다.

첫 차이 보고에는 마지막 동일 state, 최초 다른 state, tensor shape/dtype/device, max/mean error, parent hashes, selected source branch를 넣는다. “loss가 다르다”는 끝점만 보고하지 않는다.

**causal masking과 loss masking의 직교 실험**

causal attention mask는 position `t`가 미래 key를 보지 못하게 한다. loss mask는 어느 target이 objective에 기여하는지 정한다. 둘은 shape가 비슷해도 역할이 다르다.

prompt token을 loss에서 가려도 completion token은 prompt를 attend해야 한다. prompt attention까지 가리면 조건부 분포가 바뀐다. response-only SFT에서 자주 생기는 혼동이다.

padding은 key attention과 loss 양쪽에서 보통 제외되지만 query output 처리와 kernel semantics는 구현마다 다를 수 있다. left/right padding과 position IDs를 함께 시험한다.

packed document는 loss mask가 모두 유효해도 cross-document attention을 막아야 할 수 있다. block-diagonal mask와 단순 concatenation은 다른 model input이다. throughput 옵션이 objective context를 바꾸는지 명시한다.

fixture는 같은 token/labels에 attention mask만 바꾸는 실험, loss mask만 바꾸는 실험, 둘을 바꾸는 실험을 만든다. 첫 실험은 logits와 loss를, 둘째는 logits 동일·loss 변화, 셋째는 둘 다 변화를 기대한다.

flash attention backend가 arbitrary document mask를 지원하지 않으면 fallback 또는 다른 packing semantics가 발생할 수 있다. selected kernel과 actual mask representation을 기록한다.

**vocabulary parallel CE를 full-logit oracle과 비교한다**

vocabulary를 ranks에 shard하면 각 rank는 local logits만 가진다. global stable softmax에는 global max와 global exp sum이 필요하다. target logit은 target ID를 소유한 rank에서 선택해 collective로 합친다.

먼저 local max를 all-reduce max해 `m`을 얻는다. 각 rank가 `Σ_local exp(z_i−m)`을 계산하고 all-reduce sum해 denominator를 얻는다. global target logit과 `log denominator`로 NLL을 만든다.

target ID의 shard range와 local index mapping이 틀리면 shape와 collective는 성공해도 다른 class를 gather한다. shard boundary target를 fixture에 넣는다.

ignore target은 local gather 전에 안전한 sentinel로 바꾸되 최종 numerator와 count에서 제외한다. `-100`을 vocabulary index로 직접 쓰지 않는다.

label smoothing은 non-target class 전체에 분포하므로 shard별 sum과 global reduction이 더 필요하다. full-logit reference와 loss·gradient를 비교한다.

unequal vocabulary shard나 padded vocabulary row가 있으면 padded class가 softmax denominator에 들어가지 않도록 mask한다. vocab size와 padded size를 분리한다.

distributed CE가 logits 전체 gather를 피하더라도 backward는 local `p−q` shard를 만들어야 한다. selected rows의 full reference gradient slice와 비교한다.

collective 순서가 rank마다 다르면 hang한다. zero-valid rank도 동일 collective sequence에 참여해야 한다. global zero count는 명시적 failure다.

### shift·mask·denominator·kernel 장애 주입을 반증표로 묶는다

UTF-8 normalization을 NFC에서 NFKC로 바꾼다. raw bytes gate 또는 token checksum이 잡아야 한다. BOS를 두 번 넣는다. rendered/token fixture가 잡아야 한다.

EOS label 하나를 ignore로 바꾼다. valid count와 per-token ledger가 잡아야 한다. PAD를 일반 target ID로 둔다. ignore count와 loss slice가 잡아야 한다.

labels를 collator와 loss에서 두 번 shift한다. expected target index table이 잡아야 한다. truncation side를 바꾼다. raw span ledger가 잡아야 한다.

tokenizer vocabulary 두 row를 교환하고 embedding만 교환한다. LM head parity가 실패해야 한다. tied storage에 row permutation을 두 번 적용한다. storage identity guard가 잡아야 한다.

attention mask와 loss mask를 교환한다. logits와 target set invariant가 함께 실패해야 한다. packed boundary를 제거한다. cross-document attention probe가 잡아야 한다.

logits에서 max를 빼지 않는 naive softmax를 큰 값에 실행한다. stable FP64 oracle이 nonfinite를 잡아야 한다. logits cast를 BF16에 유지한다. near-tie fixture가 tolerance 밖이면 실패한다.

`num_items_in_batch`에 sample 수를 넣는다. manual numerator/count가 잡아야 한다. 두 microbatch local mean을 평균한다. unequal-length fixture가 parameter weighting 차이를 보여야 한다.

vocabulary shard target offset을 한 칸 틀린다. shard-boundary fixture가 잡아야 한다. padded vocabulary row를 denominator에 넣는다. full-logit oracle이 잡아야 한다.

custom model이 loss kwargs를 버리게 한다. selected callable trace와 count-dependent fixture가 잡아야 한다. compiled fused loss만 다른 shift를 쓰게 한다. eager/compiled parity가 잡아야 한다.

각 장애마다 expected first divergence를 정한다. 더 뒤 gate에서 우연히 잡혔다면 earlier observability를 보강한다. 여러 장애를 동시에 넣지 않는다.

### LossEnvelope에 forward·backward·artifact evidence를 담는다

`LossEnvelope`에는 RawDocumentID, RenderedTextID, TokenizerID, BatchID, ModelID, source revisions를 넣는다. input, labels, masks, shift owner, selected loss callable을 넣는다.

수치에는 per-token NLL 또는 검증 slice, numerator, valid count, normalized scalar, dtype, scale 이전 loss를 넣는다. distributed이면 rank-local sums/counts와 intended global objective를 넣는다.

기하에는 selected logits, probabilities, `p−q`, Hessian-vector probe를 넣는다. 모든 vocabulary tensor를 저장할 수 없으면 deterministic indices와 aggregate를 둔다.

state에는 embedding/head alias, vocabulary shard, attention boundary, RNG mode, compiled/eager branch를 넣는다. source evidence와 runtime evidence를 분리한다.

2장은 이 envelope의 scalar에 backward seed가 정확히 들어가는지 확인한다. selected logit gradient가 autograd를 거쳐 parameter gradient에 기여하는지, AMP scale과 collective가 global objective를 보존하는지 검사한다.

loss envelope가 불완전하면 optimizer 비교를 시작하지 않는다. 잘못된 denominator에서 나온 gradient를 정확하게 update해도 원하는 목적함수는 학습되지 않는다.

최종 인수는 raw bytes부터 per-token loss까지 정방향 재현, loss에서 target span까지 역추적, fixed-source와 runtime branch 대조, 장애 주입 검출을 요구한다. 이 네 경로가 닫혀야 1장이 끝난다.

**GoldenTokenRun 완전 기록**

원문은 짧은 한글, ASCII, newline, emoji, combining character를 포함한다. UTF-8 bytes, normalization 전후, byte offsets를 저장한다. tokenizer가 만든 tokens, IDs, offsets, special flags를 표로 만든다.

template 전후 bytes를 비교한다. BOS, role marker, EOS 위치를 표시한다. truncation 전후 길이와 제거 span을 기록한다. padding 전후 attention과 label mask를 적는다.

각 position에 input token, expected next token, loss 포함 여부를 적는다. automatic shift 뒤 실제 flattened target index를 적는다. 사람이 읽은 표와 tensor assertion이 일치해야 한다.

embedding row checksum과 position state를 기록한다. block 0, middle, final hidden의 selected slice와 norm을 기록한다. final norm과 LM head 뒤 selected logits를 기록한다.

stable softmax의 max, exp sum, target probability를 FP64로 계산한다. framework probability, fused CE loss와 비교한다. target logit gradient `p−onehot`을 비교한다.

batch에는 짧은 row, 긴 row, zero-label 후보, packed boundary row를 넣는다. row별 numerator/count와 batch 합을 계산한다. count 0 row 처리와 global zero failure를 확인한다.

두 microbatch와 두 rank에 row를 불균등 배치한다. global numerator/count가 concatenated reference와 같아야 한다. local mean 평균은 expected failure로 남긴다.

Transformers direct loss, model forward, Trainer first batch의 세 경로를 실행한다. selected callable, shift labels, reduction이 같음을 확인한다. nanoGPT pre-shift 경로는 별도 expected tensor로 비교한다.

eager와 compiled, full vocabulary와 sharded CE를 비교한다. logits 또는 gradient tolerance를 사전 고정한다. 차이가 나면 최초 tensor edge를 찾는다.

Run 결과는 통과 숫자만 아니라 raw artifacts와 assertion log를 가진다. source commit, function hash, build, dtype, backend를 함께 저장한다.

**failure 위치별 질문 백 개를 압축한다**

bytes가 같은가. normalization이 같은가. decoder가 같은가. template가 같은가. special token이 같은가. tokenizer files가 같은가. vocabulary order가 같은가. offsets가 같은가.

BOS가 한 개인가. EOS가 있는가. PAD ID가 맞는가. added token이 atomic한가. truncation side가 맞는가. maximum length가 같은가. packing boundary가 있는가.

input shape가 맞는가. input dtype이 integer인가. attention shape가 맞는가. labels shape가 맞는가. ignore ID가 loss와 같은가. valid count가 양수인가.

shift owner가 하나인가. target index가 한 칸 뒤인가. EOS가 target인가. prompt marker가 target인가. padding이 빠졌는가. packed tail이 보존됐는가.

embedding vocab가 충분한가. 새 row가 저장됐는가. head와 tied인가. position IDs가 맞는가. padding position이 맞는가. document reset이 의도와 맞는가.

base revision이 같은가. remote code가 같은가. dropout mode가 같은가. RNG가 같은가. attention backend가 같은가. causal mask가 같은가.

첫 block input이 같은가. normalization epsilon이 같은가. rotary state가 같은가. cache가 꺼졌는가. final hidden이 같은가. final norm이 같은가.

LM head weight가 같은가. bias가 같은가. vocab shard mapping이 같은가. padded vocab가 masked인가. logits dtype이 같은가. logits shape가 같은가.

stable max를 빼는가. accumulator dtype이 충분한가. target gather가 맞는가. ignore가 gather 전에 안전한가. smoothing target가 맞는가. class weight가 맞는가.

numerator가 같은가. denominator가 같은가. reduction이 sum인가 mean인가. count가 sample인가 token인가. zero count가 실패하는가. loss dtype이 같은가.

model이 kwargs를 받는가. wrapper가 count를 전달하는가. selected loss가 기대 함수인가. custom loss가 shift하는가. fused loss가 같은 식인가. compile이 branch를 바꾸는가.

rank local count가 기록되는가. global count가 reduce되는가. vocab max가 global인가. exp sum이 global인가. target shard offset이 맞는가. zero-valid rank가 참여하는가.

per-token loss가 재구성되는가. scalar가 합과 맞는가. gradient 합이 0인가. finite difference와 맞는가. Hessian-vector가 맞는가. constant shift에 불변인가.

token permutation과 row permutation이 일관적인가. tokenizer 교체의 평가 단위가 같은가. byte-normalized metric이 필요한가. perplexity 비교가 같은 vocabulary인가.

fixed-token 시험이 통과하는가. raw-text 시험이 통과하는가. eager 시험이 통과하는가. distributed 시험이 통과하는가. failure injection이 기대 gate에서 멈추는가.

마지막 동일 state가 기록됐는가. 최초 다른 state가 기록됐는가. parent hashes가 있는가. source branch가 있는가. tolerance가 사전 선언됐는가. 복구 뒤 재시험했는가.

**loss 변경을 승인하는 변경 관리**

tokenizer 변경은 새 objective coordinate다. old/new token count, offsets, fertility, byte coverage, downstream metric을 비교한다. 단순한 vocabulary 파일 교체로 취급하지 않는다.

template 변경은 marker와 target span을 바꾼다. GoldenTokenRun을 재생성한다. serving prompt도 같은 revision으로 묶는다.

max length 변경은 truncation과 data weight를 바꾼다. length slice별 retained targets와 quality를 보고한다. throughput 개선과 objective 변경을 함께 쓴다.

packing 변경은 context graph를 바꿀 수 있다. boundary attention과 valid count를 검증한다. cache를 무효화한다.

loss implementation 변경은 source diff, scalar fixture, gradient fixture, performance를 순서대로 승인한다. 빠른 kernel이라는 이유로 수치 gate를 생략하지 않는다.

reduction 변경은 learning rate와 gradient scale에 영향을 준다. old checkpoint와 curve를 직접 비교하기 전에 effective objective를 맞춘다.

label smoothing, z-loss, weighting 변경은 새 component manifest를 만든다. component별 numerator/count와 held-out calibration을 본다.

distributed topology 변경은 global objective equivalence를 다시 시험한다. world size가 바뀌어도 sample weighting이 같아야 한다.

변경 승인에는 이전 good RunID, 후보 RunID, semantic diff, numerical diff, behavioral diff, rollback 조건을 넣는다. metric 향상만으로 원인을 설명하지 않는다.

최종 서명은 데이터, tokenizer, model, loss, distributed 담당자가 공동으로 한다. 다음 장은 서명된 LossEnvelope만 받아 update를 시작한다.

**상태별 최소 관측값 사전**

원문 상태: bytes hash, encoding, normalization, document ID, span offsets.

template 상태: renderer hash, roles, BOS, EOS, markers, rendered bytes.

tokenizer 상태: vocab hash, merges, normalizer, pretokenizer, added tokens, token IDs.

collator 상태: max length, truncation side, padding side, packing, row mapping.

입력 상태: shape, dtype, device, stride, input checksum, attention checksum.

label 상태: shift owner, ignore ID, valid positions, EOS policy, count.

position 상태: position IDs, reset points, rotary offsets, document boundaries.

embedding 상태: weight revision, selected rows, tie identity, output checksum.

block 상태: layer ID, input norm, output norm, finite count, slice hash.

head 상태: weight hash, bias hash, vocab layout, shard range, logits dtype.

softmax 상태: global max, exp sum, target logit, target probability, accumulator dtype.

loss 상태: numerator, denominator, reduction, smoothing, weighting, scalar dtype.

분산 상태: local count, global count, vocab shard, collective order, reduction factor.

compile 상태: graph ID, guards, breaks, kernel ID, cache revision, fallback.

검증 상태: oracle version, tolerance, expected invariant, observed error, verdict.

계보 상태: parent IDs, source commits, function hashes, artifact hashes, EvalID.

이 사전의 각 행은 실제 값이 있어야 한다. “기본값”은 값이 아니다. resolved 값을 저장한다.

shape만 같아도 checksum이 다를 수 있다. checksum만 같아도 semantics가 다를 수 있다. parent와 role을 함께 기록한다.

scalar만 같아도 token 기여가 다를 수 있다. numerator vector 표본과 valid bitmap을 함께 본다.

확률만 같아도 logits common shift가 다를 수 있다. numerical range와 z-loss가 중요하면 logits 통계를 보존한다.

token IDs만 같아도 position과 attention이 다를 수 있다. model input 전체를 하나의 BatchID로 묶는다.

loss가 같아도 target permutation이 상쇄될 수 있다. per-position target와 contribution을 대조한다.

eager 결과만 맞아도 fused branch가 틀릴 수 있다. actual dispatch별 fixture를 실행한다.

single rank만 맞아도 shard offset이 틀릴 수 있다. boundary target와 zero-valid rank를 포함한다.

train branch만 맞아도 generation branch shape가 다를 수 있다. labels on/off output contract를 시험한다.

**사례별 복구 처방**

한글만 token이 달라지면 Unicode normalization과 byte fallback을 조사한다. ASCII 기준선만 통과했다고 tokenizer를 승인하지 않는다.

emoji만 offset이 어긋나면 byte, code point, grapheme 좌표를 구분한다. character index를 byte index로 오해하지 않는다.

role marker만 분해되면 added-token registration과 template whitespace를 본다. 문자열 검색 mask를 쓰지 않는다.

긴 row만 count가 줄면 truncation과 EOS loss를 본다. max length를 늘리기 전 data distribution을 측정한다.

packed row만 loss가 다르면 boundary, position reset, tail discard를 본다. packing을 끈 결과로 원인을 격리한다.

새 vocab row만 logits가 이상하면 embedding resize, head resize, initialization, tied storage를 본다.

첫 layer부터 다르면 input, position, mask, base weight, dropout을 본다. loss implementation을 먼저 고치지 않는다.

마지막 head에서만 다르면 tied weight, vocab permutation, shard gather, bias를 본다.

target probability만 다르면 stable softmax, accumulator dtype, padded vocabulary를 본다.

loss만 다르면 shift, ignore, smoothing, class weight, denominator를 본다.

microbatch 크기에 따라 다르면 reduction과 count 전달을 본다. learning rate로 보상하지 않는다.

world size에 따라 다르면 local mean, global count, collective factor를 본다.

compile에서만 다르면 graph guard, fused loss, noncontiguous layout, fallback을 본다.

checkpoint load 뒤만 다르면 tokenizer/base revision과 tensor alias를 본다. 파일명 일치로 승인하지 않는다.

serving에서만 다르면 template, BOS, generation prompt, last-position branch를 본다.

복구 뒤에는 원 고장 fixture, neighboring fixtures, full GoldenTokenRun을 차례로 실행한다.

**장간 연결**

2장은 valid target count를 gradient denominator로 사용한다. 5장은 template와 assistant span을 확장한다. 10장은 head와 tied weight 구조를 사용한다.

11장은 `p−q`가 Jacobian을 거쳐 만든 parameter gradient에 optimizer geometry를 적용한다. 15장은 vocab shard CE collective를 확장한다.

18장은 같은 shift와 mask 계약으로 SFT를 수행한다. 19장은 response sequence log-prob의 token 합을 사용한다. 24장은 perplexity와 calibration을 평가한다.

어느 장도 1장의 TokenizerID, LabelPolicyID, LossEnvelope를 이름만으로 재구성하지 않는다. immutable parent를 직접 참조한다.

이 연결은 next-token 수학을 추상식에 머물지 않게 한다. raw byte 하나의 변화가 gradient, optimizer, preference score, serving token까지 이어지는 경로를 보존한다.

최종 독자는 임의 row를 선택해 bytes에서 loss까지 계산하고, 임의 loss contribution을 원 span까지 되짚을 수 있어야 한다.

**독립 재현 시험**

독립 검토자는 작성자가 고른 row가 아니라 sealed set에서 row를 뽑는다. tokenizer API 출력만 믿지 않고 bytes와 offsets를 직접 대조한다. special token과 일반 token의 경계를 확인한다.

검토자는 model forward를 호출하기 전에 expected input, next target, ignore bitmap을 작성한다. 실제 collator 결과를 그 뒤 공개한다. 결과를 본 뒤 기대표를 고치지 않는다.

tiny deterministic model에서는 embedding, 한 block, final norm, head를 차례로 실행한다. framework 전체 forward와 단계별 실행이 같은 logits를 내야 한다. layer probe가 결과를 바꾸지 않는지도 확인한다.

loss는 stable FP64 식, framework helper, selected production kernel의 세 경로로 계산한다. scalar, numerator, denominator, selected logit gradient를 비교한다. kernel tolerance는 실행 전에 정한다.

문서가 긴 row, EOS가 없는 row, 모든 label이 masked된 row, vocabulary shard 경계 target를 반드시 포함한다. 정상 row만으로 error contract를 승인하지 않는다.

한 option을 바꿀 때 expected state diff를 먼저 쓴다. 실제 diff가 더 넓으면 숨은 default나 cache를 조사한다. 실제 diff가 없으면 option 전달 경로를 조사한다.

재현자는 source commit을 checkout하고 symbol hash를 확인한다. line number가 이동했으면 현재 function을 새 좌표로 기록한다. 오래된 좌표를 현재 실행의 증거로 인용하지 않는다.

최종 보고는 “loss 일치”가 아니라 raw→token, token→hidden, hidden→logits, logits→loss 네 edge의 결과를 각각 담는다. 실패 edge는 parent와 child tensor를 함께 보존한다.

이 시험이 다른 machine과 backend에서도 tolerance 안에서 재현되면 LossEnvelope를 봉인한다. backend별 차이가 있으면 지원 matrix와 별도 tolerance를 기록한다. 미실행 backend는 합격으로 표시하지 않는다.

봉인 뒤 data, tokenizer, template, model, loss, reduction 중 하나라도 바뀌면 새 envelope를 만든다. hash가 같은 artifact만 기존 검증을 상속한다.

검토자는 마지막으로 두 counterfactual을 실행한다. 첫째, 원문 의미를 유지한 채 whitespace와 Unicode 표기를 바꾸어 tokenizer 민감도를 측정한다. 둘째, token IDs를 고정한 채 attention과 loss mask만 각각 바꾸어 model context와 objective weight를 분리한다.

**logit을 좌표가 아니라 방향 차이로 이해한다**

vocabulary가 세 개인 한 위치를 생각하자. logits `z=(z_0,z_1,z_2)`는 3차원 점처럼 보이지만 softmax가 구분하는 자유도는 두 개뿐이다. `(1,2,3)`과 `(101,102,103)`은 같은 확률을 만든다. 모든 성분에 같은 값을 더하는 방향 `(1,1,1)`은 확률 simplex에서 움직임을 만들지 않는다. 따라서 logit 공간은 공통 이동 방향을 quotient한 상대 점수 공간으로 읽는 편이 정확하다.

두 후보만 있을 때는 이 사실이 더 선명하다. `p(y=1)=σ(z_1-z_0)`이므로 필요한 값은 margin 하나다. margin이 0이면 두 후보가 같은 확률이고, 양수로 커질수록 후보 1이 우세하다. 세 후보 이상에서는 정답 logit과 각 경쟁 logit의 margin이 함께 작용한다. top-2 margin만으로 전체 확률을 복원할 수 없는 까닭은 나머지 후보도 partition function에 질량을 보태기 때문이다.

softmax는 logit 차이를 확률 simplex 내부로 보내는 매끄러운 사상이다. simplex는 `p_i≥0`, `Σp_i=1`인 점들의 집합이다. vocab 3이면 삼각형 내부다. 한 후보의 logit을 크게 올리면 해당 꼭짓점으로 다가가지만 finite logit에서는 경계에 정확히 닿지 않는다. 낮은 precision에서 작은 확률이 0으로 underflow하면 수치 표현은 경계에 닿을 수 있으나 수학적 softmax와 구분한다.

cross entropy의 gradient `p-q`는 simplex 위 두 점의 좌표 차이다. one-hot target이면 정답 꼭짓점 `q`에서 현재 예측 `p`까지의 차이가 된다. gradient 성분 합은 항상 0이다. 공통 logit 이동 방향으로는 손실이 변하지 않는다는 기하와 같은 주장이다. golden fixture는 `abs(dz.sum())`이 tolerance 안인지 검사한다. 이 합이 크면 smoothing·mask·custom weighting 또는 구현 오류를 조사한다.

hidden state 관점에서는 LM head의 각 row `w_i`가 후보 token의 방향을 정의하고 `z_i=w_i·h+b_i`가 정렬 정도를 점수화한다. 그러나 단순 cosine similarity라고 부르면 안 된다. row와 hidden norm, bias가 모두 점수에 영향을 준다. layer normalization이 hidden norm을 제한하더라도 row norm은 후보별 logit scale을 바꿀 수 있다. tied weight에서는 입력 embedding row와 출력 classifier row가 같은 parameter지만 역할은 서로 다른 두 계산 그래프에서 나타난다.

작은 도식은 `h`를 움직였을 때 decision boundary가 어떻게 바뀌는지 보여 준다. 후보 i와 j의 경계는 `(w_i-w_j)·h+(b_i-b_j)=0`인 초평면이다. next-token 학습은 정답 후보가 모든 경쟁 후보보다 높은 쪽으로 hidden과 weight를 함께 움직인다. 다만 CE는 hard margin만 보지 않고 모든 후보의 exp-weighted 경쟁을 본다. 이미 매우 낮은 logit인 후보는 gradient 기여가 작고, 가까운 경쟁자는 크게 기여한다.

이 직관은 “모델이 token을 embedding에서 찾는다”는 거친 표현을 교정한다. 모델은 마지막 hidden과 전체 output vocabulary row의 상대 점수를 계산한다. approximate nearest-neighbor search를 하는 것이 아니라 dense 또는 sharded matrix multiplication과 stable normalization을 수행한다. 양자화나 fused head가 들어가도 보존해야 할 의미는 같은 logits 또는 허용 오차 안의 같은 확률 분포다.

## 1.15 numerical parity에서 release와 2장 handoff까지 닫는다

마지막 절은 반복적인 종료표를 하나의 release 증명으로 합친다. stable reduction, CUDA tolerance, first-divergence fixture와 LossManifest가 서로 맞을 때만 2장의 autograd graph와 optimizer step으로 넘긴다.

수학식 `logΣexp(z_i)`는 간단하지만 실제 계산은 dtype과 reduction 순서에 민감하다. FP16의 최대 finite 값은 exp 입력 범위보다 훨씬 작고, BF16은 exponent 범위가 넓지만 mantissa가 짧다. max subtraction은 큰 양수 exp overflow를 막지만, LM head가 이미 `inf`를 만들었거나 `inf-inf`가 발생하면 복구하지 못한다. 따라서 projection output의 finite 검사가 loss helper보다 앞선다.

안정 알고리즘은 row max `m`을 구하고, `u_i=z_i-m`, `s=Σexp(u_i)`, `lse=m+log(s)`를 계산한다. target NLL은 `lse-z_y`다. `u_i≤0`이므로 exp overflow는 사라진다. 매우 작은 `u_i`의 exp가 0으로 underflow해도 그 후보의 확률 질량이 표현 precision 아래였다는 뜻일 수 있다. 그러나 target이 그 후보라면 `softmax`를 먼저 materialize해 `log(0)`을 취하는 구현은 무한 손실을 만든다. `log_softmax` 또는 fused CE는 target NLL을 log domain에서 계산해야 한다.

row max와 sum을 어느 dtype에서 누적하는지 기록한다. logits가 BF16이어도 PyTorch loss 경로가 FP32로 올릴 수 있고, custom fused kernel은 tile별 FP32 accumulator를 쓸 수 있다. “입력이 BF16이므로 loss도 BF16” 또는 “출력이 FP32이므로 모든 중간 합이 FP32”라고 추정하지 않는다. source의 dispatch와 profiler의 kernel, numerical fixture를 함께 본다.

vocab이 매우 크면 sum reduction의 결합 순서가 결과를 바꾼다. `(a+b)+c`와 `a+(b+c)`는 floating arithmetic에서 같지 않다. eager, compiled, vocabulary parallel 경로가 서로 다른 tree reduction을 쓰면 bitwise equality를 요구하기 어렵다. 먼저 FP64 oracle과 각 경로의 error bound를 측정하고 tolerance를 사전 고정한다. tolerance를 regression 결과를 본 뒤 넓히지 않는다.

near-tie fixture는 logit 차이가 작은 후보를 여러 개 둔다. 공통 offset을 `0,10^2,10^4`로 바꾸고 stable NLL이 유지되는지 본다. long-tail fixture는 한 후보 0, 수천 후보 -20처럼 작은 질량이 많이 합쳐지는 경우다. 개별 exp는 작아도 합은 무시할 수 없다. two-class toy만 통과한 kernel이 큰 vocabulary에서 partition function을 틀릴 수 있다.

non-finite policy도 정한다. 입력 row에 NaN이 하나 있으면 일반적으로 row loss를 NaN으로 전파하고 즉시 중단하는 편이 안전하다. `nan_to_num`으로 조용히 바꾸면 최초 corruption을 숨긴다. positive infinity 하나와 finite 값들이 있는 row의 수학적 limit는 해당 후보 확률 1처럼 보일 수 있지만 실제 `inf-inf` 연산은 NaN을 만들 수 있다. overflow를 정상 confidence로 해석하지 않는다.

backward는 저장한 softmax 또는 log-softmax 통계를 재사용할 수 있다. fused CE는 full probability tensor를 메모리에 쓰지 않고 target loss와 gradient를 tile 단위로 만들 수 있다. 메모리 절약은 의미 변경을 허용하지 않는다. selected row에 대해 forward NLL, `dz`, row-sum zero와 finite 상태를 full FP64 reference와 대조한다.

### CE API의 shape·dtype·reduction과 CUDA branch를 고정한다

PyTorch의 `torch.nn.functional.cross_entropy`는 언어 모델에서 흔히 input `[N,C]`, target `[N]`으로 호출된다. 여기서 `C=V`가 class 축이다. logits `[B,T,V]`를 `[B·T,V]`로 flatten하고 labels `[B,T]`를 `[B·T]`로 맞춘다. `[B,V,T]` 형태로도 API를 쓸 수 있지만 class 축 의미가 달라진다. transpose를 빼먹어 shape가 우연히 맞는 경우가 가장 위험하다.

flatten 전 stride를 본다. `logits[..., :-1, :]` 같은 slice는 contiguous하지 않을 수 있다. `.view`는 contiguous layout을 요구하고 `.reshape`는 필요하면 copy한다. copy는 correctness에는 맞아도 메모리 peak와 kernel fusion을 바꾼다. `.contiguous()`를 넣었는지뿐 아니라 왜 필요한지, 실제 storage와 stride가 무엇인지 기록한다.

target dtype은 class index 경로에서는 보통 integer long이다. float target을 받는 probability-target 경로는 shape와 의미가 다르다. labels를 실수로 one-hot 또는 cast한 경우 API가 다른 branch를 선택하거나 shape error를 낼 수 있다. label smoothing이 내부에서 target distribution을 만드는 것과 caller가 soft target을 직접 넘기는 것도 구분한다.

`ignore_index`는 class vocabulary 밖 sentinel이어도 괜찮지만 ignored 위치에만 나타나야 한다. target gather 전에 sentinel을 안전한 index로 바꾸고 최종 contribution을 0으로 만드는 구현이 있을 수 있다. custom CUDA kernel이 bounds check 없이 negative index를 사용하면 illegal access 또는 잘못된 row를 읽을 수 있다. `-100`, `-1`, PAD ID가 stack마다 어떻게 resolved되는지 한 표에 둔다.

`reduction='none'`의 output shape는 target 위치 shape다. 이 경로로 per-token NLL을 얻어 mask와 직접 곱할 때 이미 ignore가 0인지 확인한다. `sum`은 numerator를 주고 `mean`은 weight가 있는 경우 단순 valid count가 아닌 유효 weight 합을 분모로 쓸 수 있다. class weight, soft target, label smoothing을 쓰면 denominator 계약을 공식 문서와 source test에서 다시 읽는다.

모든 target이 ignored인 row 또는 batch는 중요한 error fixture다. 구현과 버전에 따라 mean이 NaN이 될 수 있고 sum은 0일 수 있다. 이를 정상적인 zero loss로 backward하면 그 microbatch의 data cursor는 진행했지만 gradient는 없어진다. training policy는 empty supervised batch를 collator에서 거부할지, accumulation window의 다른 batch와 합칠지 명시한다. global valid count가 0이면 division 전에 fail-fast한다.

autocast 아래에서 loss op가 어떤 compute dtype으로 dispatch되는지도 확인한다. API 반환 scalar dtype만 보지 말고 logits cast, internal accumulator와 backward gradient dtype을 본다. compiled graph나 vendor fused op가 eager의 autocast policy를 그대로 따르는지 parity fixture로 확인한다.

### causal shift를 네 stack의 실제 owner에서 비교한다

next-token 학습에는 항상 한 칸 관계가 있지만 shift를 수행하는 주체는 stack마다 다르다. 첫 유형은 dataset가 연속 token stream에서 `x=s[i:i+T]`, `y=s[i+1:i+T+1]`를 만드는 pre-shift 방식이다. nanoGPT가 대표적인 읽기 좋은 예다. model loss는 같은 위치의 logits와 이미 이동한 target을 바로 flatten한다.

다른 유형에서는 collator가 `labels=input_ids.clone()`을 만들고 model loss helper가 logits의 앞 `T-1` 위치와 labels의 뒤 `T-1` 위치를 맞춘다. Transformers causal LM에서 흔한 계약이다. 이때 raw labels가 input과 같아 보여도 자기 자신을 예측하는 것은 아니므로, 실제 selected loss 함수의 pad/slice 또는 shift_labels 경로를 확인해야 한다.

또 다른 유형에서는 model-specific forward가 직접 `shift_logits=logits[..., :-1, :]`, `shift_labels=labels[..., 1:]`를 만든다. 공통 loss helper로 migration할 때 old shift와 new helper shift가 겹칠 수 있다. caller가 이미 `shift_labels`를 명시해 helper의 기본 shift를 우회하는 경로도 있으므로, 이름이 비슷한 인자가 실제로 어느 tensor를 기대하는지 revision별 signature를 고정한다.

fixture는 숫자가 증가하는 token sequence를 사용한다. `[10,11,12,13,14]`에서 position 0의 target은 11, position 3의 target은 14다. BOS와 EOS를 넣은 chat sequence에서는 BOS가 input context가 되고 첫 실제 token이 target이며, EOS를 학습할지 policy에 따라 마지막 answer token 다음 target이 달라진다. 표에 raw position, input ID, label tensor 값, loss가 실제로 선택한 target을 모두 적는다.

double shift는 loss가 finite하고 학습도 진행돼 발견이 늦다. 모델은 두 칸 뒤 token을 맞히며 쉬운 데이터에서는 curve가 내려갈 수도 있다. no shift는 현재 input을 복원하는 shortcut을 만든다. language structure가 아니라 embedding-to-head identity를 학습해 매우 낮은 loss가 나올 수 있다. 따라서 “loss가 잘 내려간다”는 shift 검증이 아니다.

sequence length 1도 시험한다. causal shift 뒤 유효 target이 0개다. helper가 empty tensor CE를 호출하는지, collator가 미리 제거하는지 확인한다. packed sequence에서는 document 마지막 token 다음에 다른 document 첫 token을 target으로 삼을지 EOS를 삽입할지 결정한다. attention boundary와 target boundary를 독립적으로 기록한다.

generation은 shift tensor를 만들지 않는다. 현재 prefix의 마지막 hidden에서 다음 token logits를 읽고 선택한 token을 prefix에 붙인다. training branch의 `[B,T,V]`와 cached generation branch의 `[B,1,V]` 또는 `[B,V]` shape를 구분한다. training shift 코드를 generation output에 재사용하면 마지막 token을 버리거나 잘못된 position을 읽을 수 있다.

**teacher forcing의 장점과 노출 차이를 과장 없이 설명한다**

teacher forcing은 위치 t의 조건으로 정답 prefix `x_{<t}`를 준다. 그러면 causal mask 아래 모든 위치를 한 번의 transformer forward로 병렬 계산할 수 있다. 각 위치의 hidden은 미래 token을 볼 수 없으므로 수학적으로는 서로 다른 prefix 조건부 확률을 동시에 평가한 것이다. “정답을 미리 보여 주므로 cheating”이 아니다. 정답 현재 token은 해당 위치 hidden의 입력이지만 target은 다음 위치 token이다.

생성에서는 이전 step의 model sample이 다음 prefix에 들어간다. 한 번 잘못 고른 token은 training corpus에서 드문 prefix를 만들고 이후 분포도 달라진다. 이것을 exposure bias라고 부를 수 있지만, 모든 장기 생성 오류를 하나의 원인으로 환원하지 않는다. decoding strategy, calibration, context length, data support와 reward tuning도 영향을 준다.

teacher-forced token NLL은 고정된 정답 prefix에서 local conditional quality를 측정한다. free-running generation quality는 model이 만든 prefix 분포에서 평가한다. 두 expectation의 measure가 다르다. 낮은 held-out NLL이 유용하지만 대화의 일관성, 사실성, 반복 억제와 안전성을 전부 보장하지 않는 이유다. 24장에서는 teacher-forced metric과 generation-based evaluation을 분리한다.

scheduled sampling처럼 학습 중 일부 정답 prefix를 model sample로 바꾸는 기법은 objective와 gradient estimator를 바꾼다. discrete sample을 통한 gradient, biased training distribution과 병렬성 손실을 검토해야 한다. next-token 최대우도의 단순한 “보완 옵션”으로 넣지 않는다. 사용한다면 sample policy, RNG, stop-gradient 경계와 target alignment를 별도 manifest로 둔다.

golden experiment는 같은 문장에 세 prefix를 만든다. 정답 prefix, 한 token을 의도적으로 바꾼 prefix, model이 실제 생성한 prefix다. 다음-token 분포의 KL, entropy, 정답 rank와 hidden 차이를 비교한다. 이 실험은 exposure에 대한 직관을 주지만, 한 예제 결과를 corpus 전체의 인과 주장으로 확대하지 않는다.

causal transformer가 training에서 병렬인 까닭과 generation에서 순차인 까닭도 구분한다. training에는 모든 정답 input token이 이미 있어 각 query의 attention을 한 번에 계산한다. generation에서는 다음 input token 자체가 아직 없으므로 sampling 결과를 기다려야 한다. KV cache는 과거 key/value 재계산을 줄이지만 token 간 의존성을 제거하지 않는다.

**tokenizer와 chat template가 supervision measure를 만든다**

raw 대화 하나가 곧 하나의 training example은 아니다. template는 system, user, assistant role을 control token과 구분자로 직렬화하고, tokenizer는 그 byte stream을 ID sequence로 바꾼다. label policy는 그 가운데 어떤 위치가 loss를 갖는지 정한다. 세 요소가 합쳐져 empirical objective의 sample space와 weight를 만든다.

assistant-only loss에서는 user와 system token이 context에는 남지만 direct target contribution에서는 빠진다. 그렇다고 prompt가 gradient에 전혀 영향을 주지 않는 것은 아니다. assistant token의 hidden이 prompt를 attend하므로 prompt embedding과 이를 처리한 layer에도 indirect gradient가 흐른다. “mask된 token은 학습되지 않는다”보다 “그 위치를 target으로 하는 direct CE 항이 없다”가 정확하다.

template marker를 target에 포함하면 모델이 role delimiter와 EOS를 생성하는 법을 배운다. 전부 가리면 serving에서 대화 turn을 닫는 능력이 약해질 수 있다. 반대로 system/user marker까지 loss에 포함하면 긴 prompt가 objective를 지배하거나 사용자의 문장을 그대로 모델링하는 비중이 커진다. 어느 정책이 맞는지는 목적에 달렸지만, token category별 numerator/count를 측정해야 한다.

문자열 검색으로 assistant span을 찾는 방식은 취약하다. 같은 delimiter가 user 내용에 나타날 수 있고, normalization과 tokenizer split이 offset을 바꾼다. template renderer가 제공하는 role boundary 또는 token-level assistant mask를 사용하고, raw byte span과 대응을 검증한다. truncation이 marker 중간을 자르거나 answer를 전부 제거한 example도 fail-fast fixture에 넣는다.

left padding과 right padding은 labels뿐 아니라 position IDs와 generation last-token 선택에 영향을 준다. training CE에서 padding labels를 ignore해도 attention과 positions가 다르면 valid logits가 달라질 수 있다. 같은 semantic examples를 padding side만 바꿔 unpadded valid positions의 logits가 architecture가 약속한 범위에서 일치하는지 본다.

vocabulary에 새 special token을 추가하면 tokenizer length, embedding rows와 LM head rows를 함께 resize해야 한다. 새 row initialization, tied identity와 checkpoint load warning을 기록한다. PAD를 EOS와 같은 ID로 공유하는 recipe는 attention mask와 generation stopping에서 의미가 충돌할 수 있다. ID가 같다는 사실과 위치별 role을 mask로 구분한다.

tokenizer 비교에서는 fertility, byte coverage와 sequence length뿐 아니라 supervision weight를 본다. 같은 answer가 tokenizer A에서 20 token, B에서 35 token이면 token-mean objective에서 그 example과 내부 span의 가중이 달라진다. bits-per-byte는 평가 보조축이지만 학습 gradient가 자동으로 byte-normalized되는 것은 아니다.

**embedding과 tied LM head에서 하나의 token이 두 역할을 한다**

입력 embedding은 token ID를 residual stream의 시작 vector로 바꾼다. 출력 LM head는 residual hidden을 모든 token 후보의 logit으로 바꾼다. untied model에서는 `E_in[V,D]`와 `W_out[V,D]`가 별도 parameter다. tied model에서는 같은 storage를 두 역할이 공유한다. shape가 같고 값이 우연히 같은 것과 실제 alias는 다르다.

한 위치의 output gradient는 `dW_out=(p-q)h^T`다. 모든 vocabulary row가 확률에 비례해 gradient를 받을 수 있다. 입력 lookup gradient는 해당 ID가 나타난 위치의 `dh`만 `E_in` row에 scatter-add한다. tied라면 두 기여가 같은 parameter gradient buffer에 합쳐진다. 따라서 batch input에 한 번도 없던 token row도 output 경쟁 경로로 업데이트될 수 있다.

반대로 등장 빈도가 높은 token은 lookup 경로의 기여가 여러 위치에서 합쳐진다. output 경로에서는 자주 정답인 것뿐 아니라 자주 높은 오답 확률을 받은 정도가 영향을 준다. row gradient를 frequency만으로 설명할 수 없다. golden fixture는 작은 vocab에서 input occurrence contribution과 output classifier contribution을 따로 계산한 뒤 합을 tied autograd 결과와 비교한다.

weight tying은 parameter 수와 memory traffic을 줄이고 lexical input/output geometry를 연결한다. 그러나 control token, multimodal placeholder, byte fallback처럼 입력과 출력 역할이 비대칭인 row도 같은 공간을 공유한다. 특정 architecture가 별도 output bias, scaling 또는 untied head를 택한 이유를 config와 source에서 확인한다. tying을 일반 법칙으로 가정하지 않는다.

Transformers model에서 `tie_word_embeddings`, `tie_weights`, `_tied_weights_keys`와 resize 경로가 어떻게 연결되는지 revision별로 본다. 일부 model은 initialization 뒤 tie하고, 일부는 torchscript나 quantization 조건에서 clone할 수 있다. state dict key 두 개가 같은 storage인지 `data_ptr`, parameter object identity와 optimizer parameter list로 확인한다.

vocab resize에서 입력만 늘리고 head를 빼먹으면 shape mismatch 또는 새 token 출력 불능이 생긴다. tied model은 resize helper가 alias를 다시 만드는지 본다. checkpoint save/load 뒤 alias가 보존되는지, adapter나 quantization wrapper가 base parameter를 복제하지 않는지 시험한다. optimizer group에 같은 parameter가 두 번 들어가면 gradient가 맞아도 decay와 step이 두 번 적용될 수 있다.

**sampling은 training softmax를 어떻게 다시 사용하는가**

training CE는 전체 vocabulary 분포에서 정답 log probability를 읽는다. generation은 같은 logits를 decoding policy로 변환해 실제 token 하나를 고른다. greedy는 argmax, temperature sampling은 logits를 `τ`로 나누고 categorical sample을 뽑는다. top-k와 top-p는 후보 집합을 잘라 남은 질량을 재정규화한다. 이 옵션들은 model weight가 아니라 inference distribution을 바꾼다.

temperature `τ<1`은 margin을 확대해 분포를 날카롭게 하고 `τ>1`은 평평하게 한다. `τ→0`을 일반 division으로 구현하면 overflow 또는 NaN이 생길 수 있으므로 greedy branch와 구분한다. top-k는 k개 후보를 유지하지만 누적 질량은 prompt마다 다르고, top-p는 필요한 후보 수가 prompt마다 달라진다. 적용 순서가 다르면 최종 분포도 다르다.

sampling logits에는 repetition penalty, bad-word mask, minimum length EOS mask와 grammar constraint가 추가될 수 있다. 이 처리 뒤의 확률을 model의 raw calibration과 혼동하지 않는다. raw logits, processor 적용 logits, truncated-renormalized distribution과 selected token을 단계별로 기록한다. log probability를 preference training이나 evaluation에 쓸 때 어느 분포의 값을 썼는지 명시한다.

categorical sample은 RNG state와 device implementation에 의존한다. 동일 seed만으로 batch shape, sampling 순서와 backend가 다른 실행을 완전히 재현한다고 가정하지 않는다. `RunID`, sequence별 RNG 또는 counter, logits checksum과 selected ID를 남긴다. deterministic golden fixture에서는 고정 uniform variate를 inverse-CDF에 넣어 policy를 검산할 수 있다.

calibration은 confidence와 empirical correctness의 관계다. token top-1 confidence를 구간화한 ECE는 선택한 target set과 binning에 민감하다. autoregressive sequence에서 token들이 독립이 아니고 쉬운 punctuation이 metric을 지배할 수 있다. token category, position, domain과 frequency slice를 함께 본다. temperature scaling을 validation set에 맞추면 그 분포 밖에서 보장되지 않는다.

NLL은 proper scoring rule이라 전체 분포의 확률 품질을 평가하지만, 낮은 평균 NLL이 모든 slice의 calibration을 보장하지 않는다. top-k accuracy, Brier score, entropy, selective risk와 generation behavior를 목적에 맞게 조합한다. sampling 품질 문제를 training temperature나 label smoothing 하나로 곧바로 수정하지 않는다.

**loss 호출에서 실제 CUDA kernel까지 내려간다**

Python에서 `F.cross_entropy` 한 줄을 보았다고 실행 구현을 다 읽은 것은 아니다. 일반적으로 cross entropy는 log-softmax와 negative log-likelihood의 조합 의미를 가지며, dispatcher는 device, dtype, layout, autocast, compile 여부에 따라 CPU 또는 CUDA implementation을 고른다. PyTorch snapshot을 인용할 때 Python wrapper, ATen operator schema, native implementation과 CUDA dispatch registration을 한 계보로 묶는다.

첫 질문은 실제 op graph다. eager profiler에서 `cross_entropy_loss`, `log_softmax`, `nll_loss` 계열 op가 따로 보이는지, compiler가 fusion했는지 확인한다. 이름만으로 kernel 하나라고 가정하지 않는다. 작은 row와 큰 vocabulary, reduction none과 mean, label smoothing 여부에 따라 다른 kernel 또는 launch 구성이 선택될 수 있다.

LM head의 matrix multiplication이 loss보다 훨씬 큰 memory를 만들 수 있다. `[B,T,D]×[V,D]^T`가 `[B,T,V]` logits를 쓰고, loss가 이를 다시 읽는다. fused linear-cross-entropy는 full logits materialization을 줄일 수 있지만 target gather와 logsumexp, backward weight gradient까지 같은 objective를 보존해야 한다. 메모리 절약 수치와 numerical parity를 별도 gate로 둔다.

CUDA kernel을 읽을 때 thread/block mapping, 한 row가 여러 block으로 나뉘는지, max와 sum reduction이 warp/block/global 어느 단계에서 일어나는지 본다. target index bounds, ignore branch, accumulator type와 reduction output의 atomic 사용도 확인한다. 소스의 template dtype과 runtime instantiation을 연결하지 않으면 FP32 accumulator라고 잘못 단정할 수 있다.

메모리 layout은 coalescing과 직접 연결된다. `[N,V]`의 마지막 vocabulary 축이 contiguous면 한 row를 연속 읽기 쉽다. transpose된 logits를 암묵적으로 contiguous copy하면 kernel 시간은 빨라 보여도 앞선 copy 비용과 peak memory가 숨는다. profiler에서 allocation, memcpy와 kernel을 같은 step 범위로 본다.

reduction none은 row별 loss를 쓰고 mean은 numerator와 count reduction을 추가한다. ignore 비율이 높으면 branch divergence와 effective rows가 달라질 수 있다. 성능 비교는 nominal `B·T`뿐 아니라 valid token 수, vocabulary, dtype와 logits bytes를 기록한다. “tokens/s”가 context transformer를 포함한 것인지 loss kernel만의 것인지 구분한다.

CUDA graph capture나 `torch.compile` 아래에서는 Python scalar count, dynamic shape, data-dependent empty batch가 graph break를 만들 수 있다. count를 device tensor로 유지하는 이유와 division 위치를 source에서 확인한다. graph가 성공했다는 사실보다 eager와 compiled의 per-token NLL·gradient parity가 먼저다.

kernel failure fixture는 out-of-range target, all-ignore, noncontiguous input, very large vocabulary, odd row count, shard boundary와 NaN/Inf를 포함한다. illegal memory access는 뒤 kernel synchronization에서 보고될 수 있으므로 launch site와 report site를 구분한다. debug mode의 synchronization이 timing을 바꾸는 것도 기록한다.

**vocabulary parallel loss의 통신과 미분을 손으로 유도한다**

vocabulary를 P개 rank에 나누고 rank r이 class 집합 `C_r`의 logits만 가진다고 하자. 각 rank의 local max `m_r=max_{i∈C_r}z_i`에서 `m=max_r m_r`를 all-reduce max한다. 그 뒤 `s_r=Σ_{i∈C_r}exp(z_i-m)`을 계산하고 `s=Σ_r s_r`를 all-reduce sum한다. NLL은 `m+log s-z_y`다.

target logit은 소유 rank만 값을 내고 나머지는 0을 내 sum reduce할 수 있다. 그러나 target이 ignore인지, padded vocabulary인지, shard range가 uneven한지 먼저 판정한다. ownership predicate `start_r≤y<end_r`와 local index `y-start_r`를 boundary ID마다 시험한다. 한 칸 offset 오류는 collective를 정상 완료하면서 완전히 다른 token을 학습시킨다.

backward의 local gradient는 각 local class에 `p_i`를 만들고 target 소유 rank의 정답 좌표에서 1을 뺀다. 모든 shard gradient를 논리적으로 합치면 full softmax의 `p-onehot`과 같다. full logits oracle에서 local interval을 slice해 비교한다. global row gradient 합도 0이어야 한다.

global max의 derivative가 rank 경계를 지나도 최종 logsumexp derivative는 softmax다. max op를 별도 loss 항처럼 미분해 winner rank에 추가 gradient를 주면 안 된다. stable 계산의 중간 `m`은 수치적 재표현이고 전체 식의 autodiff 또는 닫힌 backward가 올바른 취소를 구현해야 한다.

sequence parallel 또는 context parallel과 겹치면 어떤 축이 local인지 명확히 한다. 한 rank가 token 위치 일부와 vocabulary 일부만 가질 수 있다. vocab partition을 위한 max/sum collective와 token denominator를 위한 loss sum/count collective는 group과 목적이 다르다. process group을 혼동하면 deadlock 또는 잘못된 scaling이 생긴다.

tensor-parallel library가 backward gradient를 어느 시점에 reduce하는지도 본다. local LM head weight shard는 local `dz`와 hidden으로 gradient를 만들지만 hidden gradient는 vocabulary shard 기여를 합쳐야 한다. loss scalar가 각 rank에 동일하게 복제됐다고 parameter gradient가 자동으로 올바른 것은 아니다. autograd graph의 collective edge를 추적한다.

통신 최적화는 max와 sum collective를 pipeline하거나 fused op 안에 숨길 수 있다. 그래도 collective sequence, group, dtype와 element count를 기록한다. 한 rank가 valid token 0이라고 op를 건너뛰면 다른 rank가 hang한다. empty local row도 global collective에 참여하고, global count가 0일 때만 명시적으로 실패한다.

**데이터 병렬 분모와 DDP gradient 평균을 정확히 맞춘다**

rank r의 유효 token 손실 합을 `S_r(θ)`, count를 `N_r`라 하면 원하는 전역 token mean은 `L=Σ_r S_r/Σ_r N_r`다. gradient는 `Σ_r ∇S_r/N`, `N=ΣN_r`다. DDP가 각 rank parameter gradient를 world size P로 나눈다면 각 rank가 backward할 local scalar를 단순 `S_r/N`으로 두면 결과가 `1/P` 작아질 수 있다.

한 가지 구성은 local mean이 아니라 `L_r=P·S_r/N`을 backward하는 것이다. DDP 평균 뒤 `1/P Σ∇L_r=Σ∇S_r/N`이 된다. 여기서 N은 detached global count다. framework가 DDP sum을 쓰거나 custom post-scale을 하면 식이 달라진다. “loss에 world size를 곱한다”를 암기하지 말고 collective가 sum인지 average인지부터 적는다.

두 rank 예제로 검산한다. rank 0은 token 1개 loss `a`, rank 1은 token 3개 loss `b+c+d`를 가진다. 원하는 gradient는 네 token gradient 합의 1/4이다. rank별 mean을 DDP 평균하면 첫 token은 1/2, 나머지는 각각 1/6의 weight를 받아 짧은 rank가 과대 가중된다. scalar loss가 비슷해 보여도 parameter gradient에서 차이가 명확하다.

gradient accumulation K개 microbatch에서도 전체 accumulation window의 count를 써야 한다. 미래 microbatch count를 미리 모를 때는 각 loss sum을 그대로 backward해 gradient sum을 쌓고 optimizer step 직전 global count로 gradient를 나누거나, window count를 collator가 미리 산출해 각 microbatch를 scale할 수 있다. AMP scaling, DDP no-sync와 clipping 순서에 맞게 구현한다.

Trainer 또는 framework가 `num_items_in_batch`를 전달하는 목적은 이 unequal-token 문제를 해결하기 위해서일 수 있다. 하지만 값이 sequence 수인지 non-ignore token 수인지 caller source에서 확인한다. custom model이 kwargs를 받지 않거나 loss wrapper가 버리면 이전 microbatch-mean semantics로 돌아갈 수 있다. warning만 남고 training은 계속되는 경로가 특히 위험하다.

metric 계산은 backward scaling과 분리한다. 각 rank가 detached numerator와 count를 all-reduce해 dashboard global mean을 만들 수 있다. 이 metric이 맞다고 gradient도 같은 분모라고 단정하지 않는다. 반대로 backward가 맞아도 logger가 rank mean 평균을 보여 줄 수 있다. 두 경로에 같은 UpdateID를 붙이고 각각 검산한다.

fixture는 rank partition을 바꾸어도 concatenated single-device reference와 gradient가 같아야 한다. token을 `1+7`, `3+5`, `4+4`로 나누고 zero-valid rank도 넣는다. world size와 accumulation 수를 바꾸어 parameter delta까지 비교한다. dropout은 끄거나 sample별 RNG를 고정해 denominator 차이만 격리한다.

**masking은 boolean 하나가 아니라 세 개의 그래프다**

attention mask는 정보 흐름 그래프를 정의한다. loss mask는 objective에 포함되는 target vertex를 고른다. padding/compute mask는 불필요한 query 계산을 생략하거나 packed representation에서 token을 제거한다. 구현에서는 같은 `[B,T]` 모양을 공유할 수 있지만 의미와 소비 함수가 다르다.

position t의 label을 ignore하면 그 위치의 CE 항 `l_t`가 사라진다. 그러나 그 token은 뒤 위치의 key/value로 쓰이고 뒤 loss를 통해 gradient를 받을 수 있다. attention에서 막으면 뒤 hidden 자체가 바뀐다. compute에서 제거하면 position mapping과 kernel layout도 달라진다. “masked token gradient는 0”이라는 문장은 어느 mask인지 지정하지 않으면 틀리다.

causal mask는 미래 key를 막는다. padding mask는 PAD key를 막는다. packed document mask는 다른 document key를 막는다. prefix-LM이나 multimodal model은 일부 구간에 bidirectional 또는 block pattern을 쓸 수 있다. `is_causal=True` 하나가 실제 intended mask를 표현하는지 architecture별로 확인한다.

assistant-only SFT fixture에서는 user token input을 바꿨을 때 assistant logits가 변해야 한다. user label 값을 바꿔도 user 위치가 ignored라면 direct per-token loss는 변하지 않아야 한다. user attention을 제거하면 assistant logits가 달라지는 별도 결과를 기대한다. 이 세 perturbation이 mask 역할을 분리한다.

all-ignore sequence도 batch 안에서 context로 사용될 이유가 없다면 collator에서 제거할 수 있다. 하지만 packing 과정에서 다른 sequence와 같은 row에 섞였으면 row count만 보고 버리면 유효 answer도 사라질 수 있다. document/segment별 valid count를 유지한다.

loss denominator는 boolean mask 합으로 단순해 보이지만 token weight, class weight, soft label과 sequence weighting이 들어가면 effective measure가 달라진다. `Σw_i l_i/Σw_i`인지 `Σw_i l_i/N_valid`인지 식을 고정한다. dashboard에는 raw valid count와 weight sum을 모두 둔다.

**실제 모델의 forward에서 loss owner를 찾는다**

모델 카드를 읽을 때 “CausalLM”이라는 architecture label만으로 loss 경로를 확정하지 않는다. Transformers의 model-specific `forward`에서 labels branch, LM head 호출, logits cast와 `self.loss_function` 진입을 찾는다. config의 `vocab_size`, `tie_word_embeddings`, `pad_token_id`, `bos_token_id`, `eos_token_id`와 loss 관련 field의 resolved 값을 함께 기록한다.

GPT-2 계열은 token embedding과 position embedding, transformer blocks, final norm과 LM head라는 읽기 쉬운 기준선을 준다. Llama 계열은 rotary position, RMSNorm과 bias 없는 head 등 내부 구조가 다르지만 final hidden `[B,T,D]`에서 logits `[B,T,V]`와 causal loss로 이어지는 외부 계약은 비교할 수 있다. architecture 차이와 objective 공통점을 나눈다.

Qwen, Gemma 같은 model family는 vocabulary 크기, tied 여부, soft-capping 또는 model-specific logits 처리 가능성을 source와 config에서 확인한다. model card의 서술을 실제 selected class와 revision에 연결한다. remote code 또는 새 architecture가 공통 `ForCausalLMLoss`를 쓰지 않으면 공통 helper line을 그 실행의 증거로 인용하지 않는다.

multimodal causal model은 image/audio placeholder token과 projected modality embeddings를 input sequence에 삽입할 수 있다. labels에는 text target만 남기거나 modality position을 ignore한다. input_ids의 placeholder와 실제 `inputs_embeds` replacement, attention/position mapping을 추적한다. vocabulary token이 아닌 modality patch를 LM target으로 잘못 포함하지 않는지 본다.

Mixture-of-Experts model도 최종 next-token CE 자체는 같은 형태일 수 있지만 router auxiliary loss가 함께 반환될 수 있다. 출력의 `loss`가 CE만인지 CE와 auxiliary weighted sum인지 확인한다. logging에서 total loss를 perplexity로 exponentiate하면 의미가 틀린다. component별 numerator, denominator와 coefficient를 분리한다.

model example은 동일한 tiny batch로 비교한다. rendered bytes와 tokenizer가 family마다 다르므로 raw text 비교와 fixed hidden/logits loss 비교를 나눈다. fixed hidden을 각 head에 넣어 projection 계약을 보고, 실제 tokenizer path에서는 family별 special token과 template를 기록한다. 서로 다른 vocabulary의 token mean perplexity를 순위로 단정하지 않는다.

**loss가 낮아지는 이유를 네 축으로 분해한다**

training loss 하락은 곧 “지식이 늘었다”는 단일 사건이 아니다. 첫 축은 prediction improvement다. 같은 held-out conditional distribution에서 정답 probability가 올라간다. 둘째는 data measure 변화다. curriculum, truncation, masking과 mixture가 쉬운 token의 weight를 늘릴 수 있다. 셋째는 denominator 또는 logging 변화다. valid count 계산이 달라 scalar가 내려갈 수 있다. 넷째는 leakage나 shortcut이다.

curve를 해석할 때 동일 GoldenBatch의 per-token NLL을 주기적으로 재생한다. training aggregate가 내려가는데 fixed batch가 그대로면 data mixture나 logger를 의심한다. fixed batch는 내려가지만 held-out bytes가 악화되면 overfitting 또는 distribution shift를 본다. train과 eval 모두 갑자기 뛰면 code/config transition을 찾는다.

token category slice를 둔다. whitespace/punctuation, frequent lexical token, rare/byte fallback, code, 수식, role marker와 EOS의 NLL을 분리한다. 평균 개선이 쉬운 delimiter 예측에서만 왔는지, 내용 token에도 퍼졌는지 본다. category 정의는 tokenizer revision에 묶는다.

position slice도 중요하다. 짧은 prefix, 긴 context 후반, document boundary, answer 시작과 종료를 나눈다. long-context packing 변경은 전체 mean이 좋아도 뒤 위치를 덜 포함해 얻은 착시일 수 있다. retained target position histogram을 함께 본다.

memorization과 generalization을 구분하려면 exact/near duplicate, contamination과 entity slice를 평가한다. next-token CE 하나만으로 사실 지식의 저장 위치나 추론 능력을 판정하지 않는다. 24장의 contamination audit와 behavioral benchmark로 연결한다.

loss spike도 같은 네 축으로 본다. 특정 batch의 rare token, model non-finite, learning-rate update, global denominator 감소 또는 distributed rank 누락을 경쟁 가설로 둔다. raw batch, logits finite, numerator/count, gradient norm과 system event를 같은 UpdateID로 묶는다.

**디버깅용 GoldenTokenFixture를 실행 가능한 계약으로 만든다**

fixture A는 vocab 4, hidden 3의 손계산 가능한 logits와 hard target이다. stable FP64 NLL, probability, `p-onehot`, Hessian-vector를 저장한다. 공통 offset, candidate permutation과 target permutation을 적용해 invariance와 equivariance를 확인한다.

fixture B는 raw UTF-8에서 시작한다. 한글, combining mark, emoji, leading space, newline, BOS/EOS와 role marker를 포함한다. byte offset, normalized offset, token text와 ID를 고정한다. tokenizer upgrade는 expected diff 승인 없이 fixture를 갱신하지 않는다.

fixture C는 padding과 response mask다. 길이가 다른 두 대화, all-ignore prompt-only example과 answer가 truncation 경계에 놓인 example을 넣는다. attention perturbation과 label perturbation을 독립 실행한다. valid bitmap과 effective weight sum을 검산한다.

fixture D는 distributed denominator다. 같은 eight-token objective를 rank와 microbatch에 불균등하게 배치한다. concatenated reference의 loss/gradient/update와 DDP simulation을 비교한다. zero-valid rank도 collective에 참여해야 한다.

fixture E는 vocabulary parallel이다. target을 각 shard 첫/마지막 ID와 padded row 근처에 둔다. global max가 다른 rank에 있고 target logit은 또 다른 rank에 있는 row를 만든다. full logits와 distributed NLL·gradient가 맞아야 한다.

fixture F는 execution backend다. eager, autocast, compiled, fused loss와 production sharded path를 같은 input으로 실행한다. 실행하지 못한 backend는 NotExecuted다. kernel 이름, dtype, shape, tolerance와 최대 오차를 artifact에 남긴다.

fixture G는 tied weight다. input에 반복된 ID와 한 번도 등장하지 않은 vocabulary row를 넣는다. untied input/head gradient 합과 tied gradient를 비교하고 optimizer parameter 중복도 검사한다. save/load와 vocab resize 뒤 alias를 다시 본다.

모든 fixture는 expected first detector를 가진다. double shift는 target table, ignore mismatch는 valid bitmap, unstable exp는 FP64 oracle, rank mean 오류는 gradient parity가 먼저 잡아야 한다. 더 뒤의 end-to-end loss만 실패한다면 앞 gate의 관측성이 부족하다.

**현장 문제를 최초 차이 하나로 축소한다**

“새 버전에서 perplexity가 0.2 나빠졌다”는 시작점이지 진단이 아니다. 먼저 evaluation document revision, raw byte count, tokenizer/template, truncation과 target bitmap을 비교한다. target set이 다르면 model logits 비교를 멈추고 objective coordinate 변경부터 설명한다.

target set이 같으면 fixed batch의 input IDs, attention, position과 labels checksum을 맞춘다. 그다음 selected embedding rows, block probes, final hidden, logits와 per-token NLL 순서로 이동한다. 마지막 동일 edge와 최초 다른 edge만 찾으면 조사 범위가 급격히 줄어든다.

logits가 다르면 loss helper 옵션을 만지지 않는다. embedding부터 다르면 checkpoint/token row, 첫 block부터면 position·mask·kernel, head에서만이면 tie·quantization·vocab layout을 본다. logits가 같은데 NLL만 다르면 dtype, shift, smoothing, class weight와 reduction으로 한정한다.

single GPU는 같고 multi GPU만 다르면 full-logit oracle과 rank-local trace를 비교한다. vocabulary max/sum, target ownership, loss sum/count와 DDP scaling을 각각 본다. hang이 아니라 finite scalar mismatch여도 collective group 또는 scale 오류일 수 있다.

forward loss는 같고 parameter update가 다르면 2장으로 넘긴다. backward seed, AMP scale, accumulation, gradient collective, clipping과 optimizer state를 본다. 1장 단계에서 learning rate를 조절해 차이를 덮지 않는다. LossEnvelope가 같다는 증거가 있어야 downstream 원인을 좁힐 수 있다.

보고서에는 가설을 지운 증거도 남긴다. “tokenizer 동일”이 아니라 raw/token fixture hash와 diff 0, “CUDA 문제 아님”이 아니라 eager/fused parity와 kernel revision을 적는다. 미실행 경로는 배제한 가설로 쓰지 않는다.

**label smoothing과 z-loss가 바꾸는 목적함수를 분리한다**

hard target CE는 `q_y=1`인 one-hot target을 쓴다. label smoothing은 일부 질량을 다른 class로 나누어 `q`를 바꾼다. 구현에 따라 전체 V개 class에 `ε/V`를 더하거나 non-target `V-1`개에만 나눌 수 있다. 두 식은 finite vocabulary에서 다르므로 계수 이름만 보지 말고 source를 확인한다.

gradient는 여전히 `p-q` 형태지만 정답 좌표의 목표가 1보다 낮고 다른 좌표에도 음의 target 질량이 생긴다. 정답 확률을 무한히 1로 밀어붙이는 힘을 줄여 과신을 완화할 수 있다. 그러나 희귀하고 확실한 token의 최대 confidence도 제한하고, vocabulary가 크면 smoothing 질량의 해석이 달라진다. calibration 개선을 자동 보장하지 않는다.

ignore 위치와 smoothing의 순서를 시험한다. ignored target은 uniform component까지 포함해 전체 contribution이 0이어야 한다. padded vocabulary row나 tensor-parallel padding class에는 smoothing 질량을 줄 것인지 명시한다. 실제 vocabulary V와 padded size V'를 혼동하면 partition function과 target distribution이 바뀐다.

z-loss는 흔히 `λ(logsumexp(z))²` 같은 항으로 log partition의 크기를 제한한다. softmax CE가 공통 logit offset에 불변인 것과 달리 z-loss는 그 offset을 본다. 따라서 CE 확률이 완전히 같은 두 logits도 z-loss가 다를 수 있다. 공통 offset invariance fixture는 total loss가 아니라 CE component에만 적용해야 한다.

router auxiliary, load balancing, modality loss와 z-loss가 total scalar에 합쳐지면 component를 분리한다. perplexity는 pure token NLL에만 적용한다. total loss를 exp하면 확률적 의미가 없다. 각 component의 coefficient, numerator, denominator와 gradient norm을 LossEnvelope에 둔다.

**calibration을 token frequency와 context 조건으로 쪼갠다**

calibration은 confidence 0.8인 예측 집합에서 실제 정답 비율도 약 0.8인지 묻는다. 그러나 next-token 예측에서는 punctuation, whitespace와 빈번한 function word가 표본 대부분을 차지할 수 있다. 전체 ECE 하나는 rare factual token의 과신을 숨긴다. frequency decile, token category, domain과 context position별 reliability curve를 만든다.

top-1 calibration과 full-distribution NLL은 다른 정보를 준다. top-1이 틀렸더라도 정답에 두 번째로 높은 확률을 준 경우와 거의 0을 준 경우의 NLL은 크게 다르다. 반대로 top-1 accuracy가 같아도 confidence가 0.51인지 0.99인지 calibration risk가 다르다. Brier score와 entropy를 보조로 쓸 때도 vocabulary 크기와 target measure를 기록한다.

temperature scaling은 validation logits를 `z/τ`로 바꾸어 NLL을 최적화할 수 있다. ranking은 유지하지만 confidence가 바뀐다. domain, length와 decoding processor가 바뀌면 fitted τ의 보장이 약해진다. raw model calibration과 top-p로 잘린 sampling distribution calibration을 섞지 않는다.

sequence probability는 token conditional probability의 곱이고 log probability는 합이다. 긴 sequence는 대체로 더 작은 joint probability를 가지므로 길이가 다른 답변을 raw sum으로 비교하면 length effect가 생긴다. length normalization은 평가 규칙을 바꾸며 training token mean과 동일하지 않다. 19장의 preference log-prob에서도 이 차이를 다시 사용한다.

calibration fixture는 정답 여부뿐 아니라 predicted confidence, target rank, entropy와 raw logits checksum을 보존한다. tokenizer나 mask가 바뀌면 평가 event 자체가 달라지므로 old calibration curve와 직접 이어 붙이지 않는다.

**한 개의 실제 대화를 byte에서 loss scalar까지 따라간다**

예제로 system 한 줄, user 질문 한 줄, assistant 답 한 줄을 택한다. 먼저 원 UTF-8 bytes와 role별 byte span을 고정한다. template를 적용한 뒤 BOS, role marker, newline, EOS가 삽입된 위치를 표시한다. 화면 문자열만 복사하지 않고 renderer revision과 rendered-byte digest를 남긴다.

tokenizer 출력 표에는 position, token ID, 표시용 token, raw/normalized byte offset, special flag와 role을 둔다. byte fallback 조각은 사람이 읽기 어렵더라도 합쳐 원 bytes를 복원할 수 있어야 한다. added control token은 일반 문자열 검색 결과와 구분한다.

collator는 `input_ids[1,T]`, `attention_mask[1,T]`, `labels[1,T]`를 만든다. assistant-only policy라면 system/user target은 `-100`, assistant와 EOS target은 실제 ID다. model-internal shift를 쓸 때 position t logits가 labels t+1을 읽는지 표에 한 칸 화살표로 표시한다.

한 assistant 위치 k를 골라 hidden `h[0,k,D]`, head rows와 selected logits를 기록한다. full V개를 책에 나열하지 않고 target, top competitors, max와 logsumexp 통계를 보여 준다. FP64로 `lse-z_y`를 계산하고 framework unreduced NLL과 비교한다. probability는 `exp(z_y-lse)`로 얻는다.

batch scalar는 모든 valid position NLL 합과 valid count로 재계산한다. token category별 subtotal도 둔다. backward 전 선택 위치의 `dz_target=p_y-1`, competitor `dz_i=p_i`를 확인한다. tied head이면 output row gradient와 input lookup 경로가 같은 parameter에 합쳐짐을 표시한다.

이 예제의 숫자는 실행 artifact가 있을 때만 실제 값으로 싣는다. 실행하지 않았다면 shape, 식과 기대 invariant만 적고 그럴듯한 loss를 만들지 않는다. 독자가 같은 snapshot으로 재생할 script와 fixture ID를 제공하는 것이 가상의 결과표보다 낫다.

**real model 비교는 공통 계약과 고유 branch를 함께 본다**

두 causal LM을 비교할 때 공통 경로는 token IDs, hidden, LM head logits와 next-token CE다. 고유 경로는 tokenizer/template, position encoding, logits scaling, tied 여부, auxiliary loss와 custom code다. 공통점만 쓰면 실제 debugging에 부족하고 차이만 나열하면 왜 같은 objective로 학습 가능한지 보이지 않는다.

각 model에 `ModelLossCard`를 만든다. exact repository revision, class, config digest, tokenizer revision, template, labels owner, selected loss callable, logits dtype, vocab size, head shape, tie identity, ignore ID, reduction과 auxiliary component를 한 표에 넣는다. model card 설명은 근거 하나이며 runtime resolved state가 최종 증거다.

같은 raw prompt를 넣은 비교는 user experience를 보여 주지만 tokenizer와 target 좌표가 다르다. 같은 synthetic hidden과 head weight로 loss helper만 비교하는 시험은 objective implementation을 격리한다. 같은 token IDs를 억지로 서로 다른 vocabulary model에 넣는 시험은 의미 비교가 아니다. 비교 질문에 맞는 고정점을 선택한다.

chat model의 serving template와 training template가 다르면 모델 architecture가 같아도 prefix distribution이 달라진다. BOS 중복, generation prompt 누락과 EOS 정책을 먼저 본다. “base model은 되는데 instruct model은 안 된다”를 weight 차이로만 설명하지 않는다.

model upgrade에서는 config default와 loss helper signature가 함께 바뀔 수 있다. old/new source diff, resolved ModelLossCard diff와 GoldenTokenFixture 결과를 묶는다. upstream release note만으로 semantic parity를 승인하지 않는다.

**성능 최적화의 정확성 gate를 단계별로 둔다**

첫 gate는 pure math다. synthetic FP64 logits에서 NLL과 `dz`를 reference 식과 비교한다. 둘째 gate는 dtype이다. FP32, BF16, FP16에서 error 분포와 non-finite를 본다. 셋째는 layout과 shape다. contiguous/noncontiguous, odd T, large V와 all-ignore를 넣는다.

integration 단계에서는 actual model head 출력과 labels를 production loss에 넣고 eager reference와 비교한다. 이어 distributed 단계에서 vocabulary shard와 data parallel denominator를 동시에 시험한다. 마지막 update parity 단계는 같은 optimizer가 첫 parameter delta를 허용 오차 안에서 만드는지 2장 fixture와 연결해 확인한다.

각 gate가 통과한 뒤에만 throughput과 memory를 측정한다. warm-up, compile time, synchronization, batch/token shape, valid ratio, dtype, GPU와 software revision을 기록한다. loss kernel microbenchmark와 whole-step throughput을 구분한다. 실행하지 않은 shape에 성능 결론을 확대하지 않는다.

fused implementation이 logits를 반환하지 않으면 observability mode를 제공한다. selected row logits 또는 partial statistics를 reference path와 비교할 수 있어야 한다. 빠른 path에 debug edge가 전혀 없으면 regression이 발생했을 때 원인을 찾기 어렵다. debug mode 자체의 overhead는 production 수치와 분리한다.

성능 회귀와 정확성 회귀가 동시에 생기면 한 원인으로 묶지 않는다. fallback kernel, implicit contiguous copy, accumulator cast와 changed reduction을 각각 확인한다. profiler trace와 numerical diff를 같은 RunID로 연결한다.

**loss와 gradient의 관계를 Jacobian으로 확장한다**

logit gradient `g_z=p-q`는 출발점이다. hidden gradient는 LM head Jacobian을 거쳐 `g_h=W^Tg_z`가 된다. head weight gradient는 각 position에서 outer product `g_z h^T`를 합한 것이다. batch와 sequence reduction scale이 이 두 gradient에 동일하게 곱해진다.

transformer 이전으로는 `g_h`가 final norm, residual branches, attention과 MLP Jacobian을 거꾸로 지난다. 한 token의 loss가 causal attention을 통해 이전 prefix token의 state와 embedding에도 영향을 줄 수 있다. 미래 token에는 causal graph상 경로가 없다. 2장은 이 graph와 실제 autograd saved tensor를 자세히 추적한다.

Jacobian-vector product를 쓰면 전체 Jacobian을 materialize하지 않고 특정 방향의 민감도를 검사할 수 있다. 작은 model에서는 finite difference와 autograd JVP/VJP를 비교한다. logit common-offset 방향, selected hidden direction과 tied row direction을 fixture로 둔다.

scalar loss가 같아도 gradient가 다를 수 있다. label smoothing, reduction과 custom backward가 forward scalar를 우연히 맞출 수 있다. 따라서 implementation parity에는 selected `dz`, hidden gradient와 parameter gradient가 필요하다. forward-only evaluation 통과를 training kernel 승인으로 사용하지 않는다.

gradient norm이 큰 token을 찾을 때 NLL만 보지 않는다. `g_h=W^T(p-q)`는 head geometry에 따라 달라진다. 높은 NLL token이 항상 모든 parameter에 가장 큰 gradient를 주는 것은 아니다. per-token attribution은 reduction과 cross-token attention까지 고려해야 한다.

**오류 메시지가 없는 objective corruption을 찾는다**

가장 위험한 오류는 crash가 아니라 다른 목적함수를 안정적으로 최적화하는 경우다. double shift, rank mean 평균, prompt target 포함, packed boundary leakage와 tokenizer row permutation은 loss를 finite하게 만들고 curve도 내려갈 수 있다. health check는 NaN 탐지만으로 충분하지 않다.

semantic assertion을 실행 중 표본화한다. target position과 원 next token 관계, valid count, token category ratio, shift owner, embedding/head alias와 global numerator/count를 주기적으로 기록한다. 모든 step의 원문을 보존할 필요는 없지만 deterministic sampled BatchID와 secure artifact를 둔다.

control chart는 loss뿐 아니라 assistant-token 비율, EOS target 비율, mean sequence length, ignored ratio와 bytes/token을 감시한다. 갑작스러운 변화는 data/template/tokenizer pipeline drift를 가리킬 수 있다. threshold는 정상 변동 분포와 release change window를 근거로 정한다.

canary batch는 training stream과 별개로 고정된 GoldenTokenFixture를 새 binary에서 실행한다. source/config 변경이 없는 배포라도 compiler cache, native library와 GPU architecture가 달라질 수 있다. canary 결과와 production first batch의 schema check를 모두 통과해야 긴 run을 시작한다.

corruption을 발견하면 이미 소비한 UpdateID, affected checkpoint와 dataset cursor 범위를 계산한다. logger만 고쳐도 되는지 parameter history가 오염됐는지 구분한다. objective가 틀렸다면 metric 재계산만으로 checkpoint를 정상화할 수 없다.

**초보자가 직접 수행할 다섯 단계 실험**

첫 실험은 logits 두 개다. 종이에 `[2,0]`, target 0의 softmax, NLL과 gradient를 계산한다. script에서 같은 값을 구하고 logits에 10000을 더한다. naive exp는 실패하고 stable CE는 같은 값을 내야 한다. 이 실험으로 상대 점수와 logsumexp를 동시에 익힌다.

sequence shift fixture에서는 `[BOS,나,는,EOS]`의 각 position에 해당하는 다음 target을 표로 쓴다. collator labels와 model helper가 실제로 선택한 label을 출력하고 한 칸, 두 칸, no-shift fixture의 loss를 비교하되, 낮은 값만으로 정답을 고르지 않는다.

mask fixture에서는 user와 assistant가 있는 짧은 대화에서 user label만 바꾸기, user input 바꾸기, user attention 막기를 차례로 수행한다. 무엇이 per-token loss와 assistant logits을 바꿀지 먼저 예측한 뒤 실행하며, 결과가 예측과 다르면 mask consumer를 source에서 찾는다.

denominator fixture에서는 길이 1과 3인 두 sample에 서로 다른 logits를 두고 token mean과 sequence mean을 손으로 계산한다. 두 rank로 나눈 DDP simulation의 gradient는 concatenated reference와 비교하고, scalar metric과 backward scale은 따로 출력한다.

tied weight fixture에서는 vocab과 hidden이 작은 model을 만들고 같은 parameter를 embedding과 head에 사용한다. input에 없는 row의 gradient도 확인하며, untied 두 gradient의 합을 tied gradient 및 optimizer parameter identity와 비교한다.

각 실험은 실행 환경, source revision, input tensor, expected invariant와 actual result를 노트에 남긴다. 성공 화면만 캡처하지 않는다. 한 option을 바꾸기 전에 어느 state가 변해야 하는지 먼저 쓴다. 이 습관이 큰 model과 cluster에서도 같은 debugging 방법으로 확장된다.

**2장 이후로 넘길 경계를 명확히 한다**

1장은 raw bytes가 어떤 target event가 되고, model이 그 event에 어떤 probability를 주며, scalar objective와 logit gradient가 어떻게 생기는지 닫는다. 2장은 그 gradient가 transformer graph를 지나 parameter gradient와 update가 되는 과정을 맡는다. 경계를 분명히 해야 loss 오류와 optimizer 오류를 섞지 않는다.

5장은 tokenizer normalizer, pretokenizer, BPE/unigram, chat template와 offset을 더 깊게 파고 1장의 TokenizerRevision을 생산한다. 7장은 embedding, position과 norm의 실제 코드 경로를 확장한다. 10장은 real model family의 forward와 tied head를 해부한다.

14장은 low-precision matmul과 loss kernel의 accumulator, fusion과 수치 gate를 확장한다. 15장은 DP/TP/PP/CP/EP ownership과 vocabulary parallel collective를 시스템 수준으로 이어 간다. 26장은 여기서 정의한 numerator/count, entropy, non-finite와 kernel trace를 monitoring contract로 만든다.

18장의 SFT는 response mask, adapter와 template를 이 objective 위에 얹는다. 19장과 20장은 token log probability의 합을 preference와 online RL objective로 바꾸므로 raw model distribution과 sampling policy 구분을 다시 쓴다. 24장은 perplexity, calibration과 contamination을 평가 설계로 확장한다.

인계 artifact는 이름만 맞추지 않는다. 동일한 immutable parent ID와 digest로 연결한다. 이후 장에서 tokenizer나 denominator를 바꾸면 새 LossEnvelope를 만들고 변경 전 결과를 자동 상속하지 않는다.

**인수는 한 token을 양방향으로 추적하는 일이다**

정방향으로는 raw byte span에서 template role, token ID, input position, embedding row, hidden, LM-head logit, stable probability, NLL과 numerator/count 기여까지 간다. 역방향으로는 scalar loss의 한 contribution에서 target ID, rendered span과 원 DocumentID까지 돌아간다. 두 경로의 모든 edge가 revision과 tensor state를 가져야 한다.

수학 인수는 common-offset invariance, probability sum, `p-q`, row-gradient sum 0, stable logsumexp와 finite-difference를 요구한다. API 인수는 shape, dtype, stride, shift owner, ignore, reduction과 empty-batch contract를 요구한다. system 인수는 CUDA dispatch, vocabulary shard, DDP denominator와 backend parity를 요구한다.

의미 인수는 assistant/prompt/padding/document boundary가 의도한 context와 target graph를 만드는지 묻는다. 운영 인수는 tokenizer/template/model/loss revision, GoldenTokenFixture, monitoring과 rollback을 묻는다. 어느 한 축이 비어 있으면 평균 loss 숫자만 맞는 것이다.

독자는 이제 “모델이 다음 token을 맞힌다”를 한 문장으로 넘기지 않아야 한다. 어떤 byte가 어떤 확률 사건이 되었는지, 왜 그 분모로 평균했는지, 어느 kernel과 collective가 같은 수학을 보존했는지 물어야 한다. 그 질문에 코드, tensor와 fixture로 답할 수 있을 때만 다음 장의 backward와 optimizer update를 믿을 수 있다.

**perplexity 숫자를 재현 가능한 평가 단위로 만든다**

perplexity는 `exp(total NLL/valid token count)`다. 이 정의는 간단하지만 corpus bytes, tokenizer, BOS/EOS, sliding window, overlap과 loss mask가 같을 때만 숫자를 직접 비교할 수 있다. 모델 A가 한 문자열을 10 token, 모델 B가 16 token으로 나누면 token당 평균의 좌표가 다르다. total NLL과 bits-per-byte를 보조로 함께 제시한다.

긴 문서를 context window로 자를 때 각 window의 첫 token들을 어떻게 평가하는지 결정한다. disjoint chunk는 chunk 시작마다 짧은 context를 주고, strided window는 이전 context를 제공하되 overlap target을 한 번만 세야 한다. input으로 재사용된 token과 loss target으로 센 token을 구분한다. denominator에는 실제 평가한 target만 들어간다.

문서마다 perplexity를 구해 평균하는 값과 corpus 전체 NLL/count에서 구한 값은 다르다. 전자는 짧은 문서와 긴 문서에 같은 무게를 주고 후자는 token에 같은 무게를 준다. 둘 다 유용할 수 있으나 이름을 명시한다. distributed evaluator도 rank별 perplexity 또는 mean loss를 평균하지 않고 numerator와 count를 합친다.

overflow를 피하려고 document probability를 직접 곱하지 않는다. log probability를 합하고 마지막에 필요한 경우만 exp한다. 평균 NLL이 매우 크면 perplexity exp도 overflow할 수 있으므로 NLL 자체를 기본 metric으로 보존한다. dashboard formatting 실패를 model non-finite와 혼동하지 않는다.

평가 artifact에는 dataset revision, evaluated raw bytes, normalized bytes, token count, valid target count, truncated count, overlap 정책, tokenizer/template와 context length를 넣는다. 숫자 하나가 아니라 어떤 확률 사건 집합의 평균인지 재현 가능해야 한다.

**CUDA 수치 차이를 ULP와 의미 오차로 함께 판정한다**

absolute error 하나는 값의 크기에 따라 과하거나 느슨하다. FP32 reference와 BF16/fused 결과를 비교할 때 absolute, relative error와 ULP 성격을 함께 본다. 다만 ULP가 작다고 training 의미가 자동으로 같은 것은 아니다. near-tie logits에서 작은 오차가 top-1 순위를 바꿀 수 있고, 낮은 확률 tail에서는 큰 relative error가 total NLL에 거의 영향이 없을 수도 있다.

그래서 세 층으로 판정한다. 원시 수치 층은 selected logits, lse, NLL과 gradient error다. 확률 의미 층은 target rank, top-k set, entropy와 normalized row sum이다. training 층은 parameter gradient와 첫 update delta다. kernel 승인은 세 층의 사전 tolerance를 모두 통과해야 한다.

비결정적 reduction은 반복 분포로 본다. 같은 input을 여러 번 실행해 reference와 error 분포, run-to-run 변동을 기록한다. 최대 오차가 특정 shape나 GPU architecture에 몰리는지 본다. 평균 오차만 보고 희귀 catastrophic row를 숨기지 않는다.

fast-math, TF32와 compiler option은 LM head matmul과 후속 logits에 영향을 줄 수 있다. CE kernel만 격리한 fixture와 head까지 포함한 fixture를 나눈다. 어느 option이 어떤 op에 적용됐는지 resolved runtime state로 남긴다. 환경 변수 이름만으로 적용을 단정하지 않는다.

**label 정책을 데이터 품질과 연결한다**

loss mask는 기술적 padding 처리만이 아니라 어떤 발화를 모방할지 정하는 편집 정책이다. assistant answer 전부를 학습하는지, reasoning 구간과 final 구간을 다르게 다루는지, tool call과 tool result 중 무엇을 target으로 하는지에 따라 모델 행동이 달라진다. token category별 count와 weight를 data manifest에 둔다.

잘못된 answer나 unsafe span을 단순 ignore하면 context에는 남아 뒤 target에 영향을 줄 수 있다. example 전체 제거, span masking, corrected target 교체가 서로 다른 objective를 만든다. redaction marker를 삽입하면 그것도 새로운 token event다. 데이터 정제 결정을 label graph까지 추적한다.

multi-turn 대화에서 모든 assistant turn을 학습할지 마지막 turn만 학습할지도 가중을 바꾼다. 앞 turn을 context-only로 두면 그 답변의 direct CE는 없지만 뒤 turn 조건으로 사용된다. conversation 길이에 따라 long dialogue가 더 많은 target을 제공하므로 token mean에서 더 큰 weight를 갖는다.

quality score로 example 또는 token weight를 주면 effective denominator를 `Σw`로 할지 raw count로 할지 결정한다. weight 0은 ignore와 비슷해 보여도 구현 branch와 metric count가 다를 수 있다. negative weight는 일반 CE measure를 깨므로 특별한 목적과 안정성 분석 없이 허용하지 않는다.

**source line을 실행 branch 증거로 바꾸는 법**

고정 commit의 함수 line을 인용하는 것은 시작이다. runtime model class가 그 함수를 호출했는지 확인해야 한다. forward hook, profiler op, selected callable 이름과 config mapping을 통해 caller→loss helper→dispatcher를 연결한다. dead code의 정확한 line은 실행 증거가 아니다.

source anchor에는 repository, commit, path, symbol, line span, signature와 content hash를 둔다. line은 revision이 바뀌면 이동하므로 symbol/hash가 재탐색을 돕는다. local patch, installed wheel과 checkout source가 같은 build인지 package metadata와 binary provenance를 확인한다.

Python source 아래 native binary는 별 revision일 수 있다. PyTorch, CUDA runtime, compiler와 fused extension의 build 정보를 함께 둔다. custom op가 `F.cross_entropy`를 우회하면 upstream native CE 설명을 actual path로 쓰지 않는다. reference oracle 경로와 production path를 구분한다.

test source도 읽는다. upstream test가 어떤 dtype, shape, ignore, smoothing과 reduction을 검증하는지 표로 만들고 빈 cell은 local fixture로 채운다. test 이름이 존재한다고 production model의 shift와 distributed denominator까지 검증된 것은 아니다.

**운영 중 loss 이상을 판정하는 10분 체크**

첫 2분에는 UpdateID와 이전 정상 step을 고정하고 optimizer 진행을 멈출지 판단한다. numerator, valid count, scalar loss, learning rate와 finite 상태를 본다. scalar만 보지 않는다. count가 급변했다면 data/mask branch를 우선한다.

다음 2분에는 rank별 count와 loss sum을 본다. 특정 rank가 0이거나 분포가 달라졌는지 확인한다. global metric 재계산과 logged 값이 맞는지 본다. collective timeout 같은 별도 장애가 있으면 29장 runbook으로 넘긴다.

다음 3분에는 sampled BatchID의 tokenizer/template, x/y shift와 ignored bitmap을 이전 정상 run과 비교한다. target set이 같으면 selected logits와 per-token NLL을 비교한다. first non-finite 또는 first differing tensor를 찾는다.

마지막 3분에는 최근 config/source/container 변경, loss callable, compile/fused dispatch와 model tie를 확인한다. 원인 미확정 상태에서 learning rate, clipping이나 epsilon을 임의 조절하지 않는다. incident artifact와 재현 fixture를 봉인한다.

10분 안에 해결한다는 뜻이 아니라 가설 공간을 올바르게 자른다는 뜻이다. evidence가 부족하면 Unknown으로 남기고 안전한 checkpoint와 rollback을 선택한다. 추정으로 training을 재개해 더 많은 UpdateID를 오염시키지 않는다.

**마지막으로 기억할 것은 확률 한 개가 만들어지는 전체 계보다**

다음 token 확률은 LM head 끝에서 갑자기 생기지 않는다. 원 bytes가 template와 tokenizer를 지나 좌표가 되고, causal context 안에서 hidden을 만들며, 전체 vocabulary와 경쟁한 logits가 stable softmax의 한 점이 된다. label policy는 그 점을 학습 사건으로 셀지 결정하고 reduction은 사건들의 가중을 정한다.

코드는 이 수학을 여러 함수와 kernel로 나눈다. collator 또는 model이 shift를 소유하고, head matmul이 logits를 만들며, loss dispatch가 logsumexp와 target gather를 수행한다. data parallel은 사건의 분모를 합치고 vocabulary parallel은 partition function을 합친다. 어느 경계에서든 shape와 scale이 맞아 보여도 의미는 틀릴 수 있다.

따라서 한 token을 설명할 때 ID나 loss 숫자 하나로 멈추지 않는다. 원 span, context, target, logit margin, probability, denominator, gradient와 source/runtime branch를 함께 말한다. golden fixture는 정상값뿐 아니라 double shift, mask 혼동, unstable exp, shard offset과 rank weighting 실패를 의도적으로 잡아야 한다.

이 계보가 닫히면 2장의 질문이 선명해진다. 이제 확인할 것은 올바르게 정의된 `p-q`가 autograd, mixed precision, accumulation과 optimizer를 지나 어떤 parameter update가 되는가이다. 1장의 LossEnvelope가 흔들리면 그 뒤의 정교한 최적화 분석도 출발점부터 틀린다.

독립 검토자는 인수 표의 각 행에 주장, 필요한 증거, 실제 artifact, 허용 오차와 판정을 적는다. “shift가 맞다”에는 selected raw sequence, input/labels tensor와 실제 loss index가 필요하다. “분모가 맞다”에는 rank·microbatch별 numerator/count와 concatenated reference gradient가 필요하다. “CUDA path가 맞다”에는 selected dispatch, kernel/build revision과 numerical fixture가 필요하다. 설명문만으로 PASS를 주지 않는다.

반대로 모든 원본 tensor를 무제한 보존할 필요도 없다. 재현에 필요한 작은 golden batch, selected positions, stable digest, aggregate statistics와 immutable parent를 고른다. 민감한 원문은 access-controlled artifact로 두고 본문에는 비식별 span ID를 쓴다. 정보량과 개인정보 보호를 함께 설계한다.

첫 번째 counterfactual은 공통 logit offset이다. 확률, pure CE와 `p-q`는 유지돼야 하지만 z-loss가 있다면 total loss는 달라질 수 있다. 두 번째는 ignored label 교체다. direct contribution은 유지돼야 하지만 input token까지 바꾸면 뒤 context가 달라질 수 있다. 세 번째는 rank repartition이다. global objective와 gradient는 유지돼야 하지만 reduction 순서에 따른 허용 수치 차이는 생길 수 있다. 기대 범위를 먼저 적는다.

네 번째는 vocabulary permutation이다. tokenizer ID, embedding row, LM-head row와 target ID를 같은 permutation으로 옮기면 논리적 분포는 보존될 수 있다. 일부만 옮기면 shape와 loss는 finite하지만 의미가 깨진다. 다섯 번째는 template marker 추가다. marker가 context와 target에 들어가는 위치, valid count와 serving prefix가 모두 바뀐다. 이를 단순한 문자열 스타일 변경으로 승인하지 않는다.

한 fixture가 통과했다고 전체 corpus를 보장하지 않는다. property test는 random length, Unicode, packing boundary, empty answer와 shard boundary를 넓히고, production monitoring은 실제 분포의 count·category drift를 본다. golden fixture, property test와 runtime observation은 서로 다른 실패를 잡는 세 층이다.

지원 matrix에는 device, dtype, eager/compiled, loss implementation, vocabulary sharding, label policy와 distributed topology를 적는다. 실제 실행한 cell만 결과와 tolerance를 갖는다. 다른 GPU 또는 새 compiler에서 이름이 같은 op가 보인다는 이유로 결과를 상속하지 않는다. 새 cell은 작은 수학 oracle부터 다시 시작한다.

오류를 고친 뒤에는 원 fixture만 재실행하지 않는다. 인접한 shift·mask·denominator fixture, full GoldenTokenRun과 첫 backward fixture까지 차례로 확인한다. double shift 수정이 EOS target을 빼거나, denominator 수정이 AMP scale을 바꾸는 식의 회귀가 있을 수 있다. 수정 diff와 새 evidence를 같은 incident ID에 묶는다.

결국 인수의 기준은 독자가 특정 token 하나를 골라 “왜 이 확률이고 왜 이만큼 학습되는가”를 독립적으로 재계산할 수 있느냐이다. 답은 model의 직관적 서술과 수식, source 함수, tensor shape, kernel, collective, 데이터 provenance 가운데 어느 하나에만 있지 않다. 이들이 같은 사건을 가리킬 때 next-token objective는 비로소 검증 가능한 시스템이 된다.

책을 읽은 뒤 실제 저장소를 파고들 때도 같은 순서를 유지한다. 먼저 tokenizer와 collator의 출력 tensor를 고정하고, model forward의 labels branch와 selected loss callable을 찾는다. 그다음 unreduced NLL을 손계산과 비교하고 profiler로 actual dispatch를 확인한다. 마지막에 distributed scale과 parameter gradient를 본다. 처음부터 거대한 training loop의 평균 loss만 비교하면 서로 다른 원인이 한 숫자에 겹친다.

검토 결과에는 확정, 반증, 미실행을 나눈다. source로 확인한 동작과 runtime으로 관측한 동작도 구분한다. 정확성 주장은 fixture가 뒷받침하고 성능 주장은 실제 benchmark가 뒷받침해야 한다. 이 구분을 지키면 새로운 model family와 loss kernel이 등장해도 추측 대신 같은 검증 절차로 지식을 갱신할 수 있다.

이 장의 기록은 일회성 교육 예제가 아니다. tokenizer, template, model 또는 runtime을 업그레이드할 때 old/new LossEnvelope를 나란히 놓고 예상한 state diff만 발생했는지 확인한다. 예상 밖 차이가 하나라도 있으면 다음 optimizer step으로 넘어가기 전에 그 edge를 해명한다. 작은 token 하나를 끝까지 추적하는 규율이 긴 학습 실행의 비용과 오류를 줄인다.

두 실험의 결과는 동일하거나 달라야 한다는 단일 기대가 없다. 중요한 것은 어느 state가 변했고 어느 edge에서 logits 또는 loss가 처음 달라졌는지 설명하는 것이다. 예상하지 않은 state가 변하면 cache key와 implicit default를 조사한다.

마지막 artifact에는 성공 fixture뿐 아니라 의도적으로 실패한 double-shift, PAD leakage, vocab permutation, shard offset 사례를 포함한다. 새 구현이 실패 사례를 통과시켜 버리면 validation이 약해진 것이다.

최종 재현자는 다른 seed의 row도 표본 검사한다. 같은 invariant가 특정 예제의 우연한 token 배열에만 의존하지 않아야 한다. 언어, 길이, special token, packed boundary가 다른 표본에서 target index, denominator, gradient 부호를 다시 확인한다. 결과와 source revision을 봉인 기록에 추가한다.

봉인 뒤 checksum이 달라지면 이전 판정을 재사용하지 않는다. 원문 span에서 target index, denominator와 gradient 부호까지 다시 따라가 최초 차이를 찾는다. 이 왕복이 가능할 때 next-token objective는 문장 설명, 수학식, source 좌표, runtime tensor, 반례와 복구가 같은 좌표계에서 닫히며, 다음 장들이 의존할 검증 가능한 출발점이 된다.
