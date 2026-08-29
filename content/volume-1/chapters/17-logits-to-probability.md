# 17장. logits는 어디서 오고 확률은 언제 필요한가

사용자가 보는 다음 token은 정수 ID 하나다. 그 직전 model은 vocabulary의 각 ID에 실수 점수를 붙인다. 이 점수가 logits다. logits를 “softmax 전의 확률”이라고만 부르면 중요한 설계가 사라진다. logits는 마지막 hidden state와 LM head가 만든 좌표이며, tensor parallel에서는 여러 GPU에 나뉜 채 존재할 수 있고, greedy 선택에는 전체 확률 tensor가 필요하지 않다.

9장은 token과 position이 hidden state가 되는 입구를 다뤘다. 이 장은 transformer 마지막 block과 final normalization을 지난 hidden row가 vocabulary 점수로 바뀌는 경계부터 시작한다. 18장이 소유할 penalty, top-k, top-p, grammar, RNG와 stop 순서는 여기서 펼치지 않는다. 대신 그 정책이 받을 raw logits의 의미와 확률 계산의 최소 조건을 고정한다.

## 17.1 마지막 hidden state에서 vocabulary 축이 열린다

decoder-only model의 마지막 hidden tensor를 `H ∈ R[B,S,D]`라 하자. `B`는 batch, `S`는 계산한 sequence row, `D`는 hidden size다. vocabulary 크기가 `V`이고 LM head weight가 `W ∈ R[V,D]`이면 bias가 없는 기본 projection은 `Z = H W^T`다. 결과 `Z ∈ R[B,S,V]`의 마지막 축 각 칸이 한 token ID의 logit이다.

작은 예에서 마지막 hidden row가 `h=[2,-1]`이고 세 vocabulary row가 `w0=[1,0]`, `w1=[0,2]`, `w2=[1,1]`이면 logits는 `[2,-2,1]`이다. 0번 token이 가장 크다. 이 계산은 아직 확률이 아니다. 합이 1일 필요도 없고 음수여도 되며, 공통 상수 100을 모두 더해도 순위와 softmax 확률은 같다.

### 어느 sequence row를 projection하는가

prefill에서 모든 `S` 위치의 next-token distribution이 항상 필요한 것은 아니다. completion 첫 token에는 각 request의 마지막 유효 prompt row만 필요하다. decode에서는 새 token row 하나가 다음 분포를 만든다. prompt logprobs를 요청하면 prompt 내부 여러 위치가 필요하다. 출력할 row 선택은 LM head 이전 hidden gather 또는 projection 이후 logits gather로 구현할 수 있다.

row 선택은 작은 최적화가 아니다. `M`개 row를 projection하면 출력은 `M×V`다. `V=128,000`, FP32 logits라면 row 하나가 약 512 KB이고 2,048 row면 약 1 GiB다. hidden을 먼저 필요한 row로 줄이면 GEMM과 출력 memory를 함께 줄인다. 반면 prompt logprobs가 필요하면 더 많은 row를 보존해야 하므로 TTFT와 memory가 늘 수 있다.

8장에서 본 right padding 오류도 이 경계에서 드러난다. `Z[:, -1, :]`가 마지막 유효 token row가 아니라 PAD row라면 LM head는 주어진 hidden을 정확히 projection하면서 틀린 질문에 답한다. logits divergence를 보았다고 head weight부터 의심하지 말고 selected hidden row identity를 먼저 확인한다.

row ledger에는 네 index를 따로 둔다. request 안의 logical token index, packed hidden tensor의 row, LM head에 넘긴 compacted row, 반환 logits의 row다. dense eager model에서는 네 값이 우연히 같아 보일 수 있다. continuous batching이 request를 섞고 필요한 output만 compact하면 서로 달라진다. 모두 integer라 type checker가 의미 혼동을 잡지 못한다.

prefill request A의 마지막 token이 packed row 117이고 output gather 뒤 row 3이라고 하자. sampler는 request A를 output row 3과 연결해야 한다. cancellation로 앞 request가 빠지거나 CUDA Graph용 dummy row가 끼어도 이 mapping은 보존되어야 한다. 잘못되면 계산은 finite하고 shape도 맞지만 A가 B의 distribution을 뽑는다.

`logits_to_keep` 류 field는 model forward의 반환 크기만 바꾸는 옵션으로 설명하면 부족하다. field가 head 전 slicing branch를 선택하고 hidden `[B,S,D]`를 selected `[M,D]`로 만든다. LM head GEMM의 M축과 logits allocation `[M,V]`가 바뀌며 prompt logprob 가능 범위도 달라진다. 반증은 head hook의 input shape, selected indices, 반환 row-to-request map이다.

loss를 함께 요청하는 training/evaluation forward는 또 다르다. label alignment 때문에 여러 sequence row가 필요하고, model implementation이 full logits를 materialize할 수 있다. 이 책은 serving을 다루지만 offline perplexity 코드를 production generation 비용의 근거로 쓰지 말아야 하는 이유는 분명하다. 같은 `forward` 이름 아래 retained rows와 consumer가 다르다.

### weight tying은 입력과 출력 vocabulary를 같은 행렬로 묶는다

많은 causal LM은 input embedding matrix와 output LM head weight를 공유한다. 입력에서 ID `i`가 고른 row와 출력에서 후보 `i`를 점수화하는 row가 같은 parameter storage다. parameter 수를 줄이고 입력·출력 token space를 연결하는 inductive bias를 준다. 그러나 “embedding과 logits는 역연산”이라는 뜻은 아니다. 여러 layer를 지난 hidden state와 모든 row의 내적을 계산할 뿐이다.

Transformers의 [`tie_weights`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_utils.py#L2530-L2568)는 config와 encoder-decoder 조건에 따라 input/output embedding을 연결한다. [`resize_token_embeddings`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_utils.py#L2710-L2768)는 vocabulary row를 바꾼 뒤 tie 관계를 다시 다룬다. tokenizer만 늘렸을 때 input lookup과 output projection 모두가 위험한 이유다.

tied 여부는 field→branch→state→effect로 추적한다. `tie_word_embeddings`가 model construction 분기를 고르고 output head parameter가 input embedding과 같은 storage를 가리키거나 복제된다. checkpoint load와 resize 뒤에도 alias가 유지되어야 한다. 효과는 parameter memory와 update/quantization 계약 변화다. 반증은 두 parameter의 storage identity, shape, reload 전후 logits다.

untied model도 정상이다. architecture가 별도 output head를 학습했거나 multimodal vocabulary extension, adaptive head를 쓸 수 있다. 이름이 `lm_head`라고 무조건 embedding과 같다고 가정하지 않는다. config, construction, checkpoint keys, 실제 pointer를 함께 본다.

tied storage는 dtype 변환과 device placement에서도 주의한다. input embedding을 한 device에, LM head를 다른 device에 놓는 model parallel policy는 실제 alias를 유지하기 어렵다. framework가 view, 복제, hook으로 논리적 tying을 흉내 낼 수 있다. memory report에서 parameter 이름 두 개를 단순 합하거나 하나로 간주하기 전에 data pointer와 lifecycle을 본다.

quantization conversion이 tied parameter를 두 번 방문하면 서로 다른 packed representation을 만들 수 있다. 이것이 의도적이면 input lookup용 format과 GEMM용 format이 달라 memory 절감이 예상보다 작다. 의도하지 않았다면 reload 뒤 logits가 달라질 수 있다. conversion log에 original parameter identity와 produced packed tensors를 연결한다.

vocabulary 확장도 양쪽 의미가 다르다. input row는 새 ID가 prompt에 들어올 때 읽히고 output row는 새 ID가 생성 후보가 될 때 점수화된다. 새 PAD를 mask해 입력에만 사용한다면 output 후보에서 금지해야 할 수 있다. 새 domain token을 생성시키려면 row 초기화만으로 충분하지 않고 학습된 head geometry가 필요하다.

## 17.2 LM head는 큰 GEMM이며 분산 경계이기도 하다

이 절은 분산 LM head를 다시 소유하지 않는다. Vocab shard의 물리 배치, gather·candidate collective와 selected token의 broadcast·commit은 16장의 canonical ledger를 참조한다. 여기서는 local score들이 하나의 global normalization domain을 이루려면 어떤 max·sum·후보 정보가 필요한지만 다룬다. 즉 통신의 구현 소유자는 16장이고, 통신 전후에도 보존돼야 하는 확률 수학의 소유자는 이 장이다.

단일 GPU에서는 선택된 hidden `H_sel ∈ R[M,D]`와 `W^T ∈ R[D,V]`의 matrix multiplication이다. prefill prompt logprobs처럼 `M`이 크면 GEMM다운 모양이고, decode에서 request 수가 작으면 skinny matrix다. vocabulary가 크므로 weight read와 output write가 비용을 지배할 수 있다. 모든 logits를 FP32로 materialize하는지도 peak memory에 영향을 준다.

### tensor parallel은 vocabulary row를 나눈다

vocabulary-parallel head는 `W`의 row, 즉 vocabulary 축을 TP rank에 나눈다. rank `r`가 `V_r`개 row를 가지면 local logits는 `[M,V_r]`다. 이 상태의 local argmax는 세계 전체 argmax가 아니다. 각 rank의 local top value와 global token ID를 비교·reduce해야 한다. 전체 logprobs가 필요하면 global normalization에 필요한 통계와 요청한 token의 logit을 통신해야 한다.

전체 logits를 모든 rank에 all-gather한 뒤 일반 sampler를 실행하는 방법은 단순하지만 `M×V` 통신과 memory를 만든다. 다른 설계는 local top-k 후보만 모으거나 distributed max/sum reduction으로 softmax denominator를 계산한다. 어느 것이 맞는지는 requested output과 sampler 기능에 달려 있다. vocabulary 전체 logprobs 반환과 greedy token 하나 선택은 필요한 정보량이 다르다.

TP=4, `V=128,000`, `M=256`을 생각하자. 각 rank local logits는 8,192,000 scalar다. FP32라면 약 31.25 MiB다. 모든 rank가 full logits를 가지도록 gather하면 rank당 약 125 MiB가 되고 step마다 collective traffic이 생긴다. decode M이 작아도 매 token 반복되며, prompt logprob에서는 M이 커진다. 최적화가 어떤 축을 줄이는지 숫자로 써야 한다.

hidden replication 여부도 비용식에 포함한다. column-parallel transformer의 마지막 hidden이 rank마다 완전한지, reduce-scatter 상태인지에 따라 vocab-parallel head 입력 통신이 달라진다. head 소스 한 함수만 보고 통신이 없다고 단정하지 않는다. 직전 normalization과 residual parallel layout에서 tensor가 어느 rank에 어떤 shape로 존재하는지 이어 본다.

bias가 있는 head는 vocabulary row와 같은 방식으로 shard할 수 있지만 model 대부분은 bias 없는 head를 쓴다. softcap이나 final logit scaling을 적용하는 architecture도 있다. 이 변환이 head 내부 raw score인지 sampling processor인지 구분한다. model-defined scale은 temperature와 이름이 비슷해도 model function의 일부이며 request별 sampling temperature와 책임이 다르다.

vLLM의 [`LogitsProcessor`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/logits_processor.py#L29-L88)는 hidden state와 LM head, parallel logits gathering의 경계를 한 객체에 묶는다. [`_get_logits`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/logits_processor.py#L99-L133)는 head projection 뒤 tensor-parallel gather 여부를 결정한다. option 이름보다 local tensor shape와 gather 이후 shape를 기록해야 한다.

SGLang의 [`LogitsProcessor` forward`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/logits_processor.py#L79-L150)는 forward mode와 metadata에 따라 필요한 hidden rows를 고르고 LM head를 호출한다. 같은 model forward라도 decode, extend, target verification, prompt logprob 요청이 보존할 row와 후속 reduction을 바꾼다.

### distributed top-k는 gather 생략의 조건을 드러낸다

greedy라면 각 rank의 최대 `(value, global_id)`만 비교하면 된다. top-k라면 각 rank에서 k개보다 많은 후보가 global top-k에 들어갈 수 없으므로 local top-k를 모아 다시 top-k할 수 있다. 하지만 top-p는 누적 확률 질량을 알아야 하므로 필요한 후보 수가 데이터에 따라 달라진다. logit bias나 grammar가 global vocabulary 어느 ID든 바꿀 수 있다면 processor 적용 위치도 제약한다.

따라서 “distributed sampling을 켜면 gather가 사라진다”는 설명은 불완전하다. field가 distributed branch를 고르고, local logits와 processor state가 rank별로 유지되며, candidate/reduction tensor만 통신한다. 효과는 통신량과 memory 감소일 수 있다. 반증 관측은 collective 종류와 bytes, local/global candidate identity, reference gathered logits의 selected token이다.

vocabulary padding도 ledger에 넣는다. TP divisibility를 위해 실제 tokenizer vocabulary보다 head row를 더 크게 padding할 수 있다. padded rows가 sampling 후보가 되지 않도록 mask하거나 gather 뒤 잘라야 한다. tokenizer의 `V_tokenizer`, model config의 `V_model`, physical shard 합 `V_physical`, 반환 logits의 `V_visible`을 구별한다.

예를 들어 tokenizer V가 32,003이고 TP=8 정렬 때문에 physical V가 32,064라면 마지막 61개 row는 사용자 token이 아니다. quantized packing tile 때문에 더 큰 정렬이 붙을 수도 있다. 이 row의 random logit이 커도 global argmax 후보에서 제외되어야 한다. output을 `V_tokenizer`로 trim하는 위치와 allowed ID validation의 상한이 일치하는지 본다.

LoRA가 LM head를 수정하는 경우 base projection 뒤 adapter delta가 더해질 수 있다. adapter마다 vocabulary가 같아도 logit 결과가 다르며, mixed-adapter batch는 row별 adapter identity를 유지해야 한다. base head hash만 같다고 logits identity를 기대하지 않는다. selected hidden row와 함께 adapter ID, scaling, extra vocabulary mapping을 trace에 남긴다.

분리 serving에서 logits가 어느 rank/process까지 이동하는지도 정한다. worker GPU에서 sampling까지 끝내 token ID만 coordinator로 보내면 network payload가 작다. coordinator가 arbitrary processor를 적용하려고 full logits를 요구하면 vocabulary-sized transfer가 생긴다. extension flexibility와 data movement의 tradeoff이며 API plugin 설계가 GPU collective 비용을 바꿀 수 있다.

## 17.3 raw logits의 절대값보다 차이가 중요하다

softmax는 `p_i = exp(z_i)/Σ_j exp(z_j)`다. 모든 logit에 같은 상수 `c`를 더하면 분자와 분모에 `exp(c)`가 곱해져 사라진다. 그래서 `[2,-2,1]`과 `[102,98,101]`은 같은 분포다. logits를 서로 다른 요청 사이에서 절대 크기만 비교해 “더 확신한다”고 말하면 위험하다.

두 후보의 확률 비는 `p_i/p_j = exp(z_i-z_j)`다. logit 차이 1은 odds 약 2.718배, 차이 2는 약 7.39배다. top-1 margin은 greedy 안정성의 유용한 단서다. 1위와 2위 차이가 `1e-5`인데 quantization/backend 오차가 그보다 크면 현재 token은 같아도 작은 변화로 뒤집힐 수 있다.

### temperature는 energy 차이의 단위를 바꾼다

temperature `T>0`은 `p_i(T)=softmax(z_i/T)`다. `T<1`이면 차이가 확대되어 분포가 날카로워지고, `T>1`이면 차이가 줄어 평평해진다. logits에 `T`를 곱하는 것이 아니라 나눈다. `[2,-2,1]`에 `T=2`를 적용하면 `[1,-1,0.5]`, `T=0.5`면 `[4,-4,2]`다.

temperature field→branch→tensor→effect는 구현마다 다를 수 있다. `temperature=0`을 수학식에 넣으면 division by zero다. runtime은 보통 greedy branch를 선택하거나 sampling 없이 argmax한다. `T>0` branch만 logits scaling tensor를 만든다. 반증은 processed logits, softmax 호출 유무, RNG 소비, selected ID다.

`T→0+`에서 분포가 argmax에 집중한다는 수학적 극한과 `temperature=0` 구현 branch는 구별한다. tie가 있으면 극한 분포가 최대 후보들에 나뉠 수 있지만 구현 argmax는 보통 가장 작은 index 같은 deterministic tie-break를 쓴다. 재현에는 tie-break도 포함한다.

temperature는 logit 차이에 작용하므로 공통 offset을 제거해도 결과가 같다. 하지만 token별 bias, repetition penalty와는 일반적으로 교환법칙이 성립하지 않는다. additive bias `b`를 temperature 전 적용하면 `(z+b)/T`, temperature 뒤 더하면 `z/T+b`다. `T≠1`이면 bias 효과가 다르다. 설정 목록이 같아도 processor order가 다르면 분포가 달라지는 수학적 이유다.

API default도 effective state로 확인한다. request에서 temperature를 생략했을 때 server default, model generation config, endpoint compatibility default 중 무엇이 채우는지 본다. 값 `None`, 0, 1은 서로 다르다. 1은 scaling identity이고, 0은 greedy branch일 수 있으며, None은 상위 default를 의미할 수 있다. serialized JSON만 비교하지 말고 merge 뒤 값을 기록한다.

temperature를 높이면 항상 다양성이 증가한다고 단정하지 않는다. 뒤에서 top-k=1이나 grammar가 후보 하나만 남기면 distribution은 여전히 degenerate하다. 이 장은 temperature가 raw logits에 만드는 국소 효과를 설명하고, 조합 결과는 18장에서 ordered chain으로 검증한다.

Transformers의 [`TemperatureLogitsWarper`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/logits_process.py#L239-L290)는 positive temperature를 검증하고 scores를 나눈다. temperature 0이면 이 warper를 억지로 호출하는 것이 아니라 generation mode가 greedy인지 확인해야 한다. 18장에서 processor 조립 순서를 이어 본다.

### argmax에는 softmax가 필요 없다

exponential은 strictly increasing 함수이고 공통 denominator는 모든 후보에 같다. 따라서 `argmax_i z_i = argmax_i softmax(z)_i`다. greedy token 하나만 필요하면 전체 `V` exponential과 sum, probability tensor를 만들 이유가 없다. LM head logits에서 직접 maximum을 찾을 수 있다.

temperature가 양수인 경우도 `argmax(z/T)=argmax(z)`라 greedy 결과를 바꾸지 않는다. greedy mode에서 temperature를 조정했는데 token이 바뀌길 기대하면 mode 의미를 혼동한 것이다. penalty나 grammar가 logits를 먼저 바꾸면 그 processed scores의 argmax를 취해야 하지만, 여전히 softmax 자체는 불필요하다.

확률이 필요한 경우는 sampling을 위한 categorical distribution, logprob 반환, entropy/확률 질량 기반 policy, cross entropy 평가다. top-p는 누적 probability를 쓰므로 normalization이 필요하다. 다만 구현은 전체 확률 tensor를 항상 materialize하지 않고 fused 또는 candidate-limited algorithm을 쓸 수 있다. API가 확률을 요구하는 것과 intermediate storage shape는 구별한다.

entropy `-Σ p_i log p_i`는 전체 또는 충분한 분포가 필요하다. top-N logprobs만으로 계산하면 누락 질량 때문에 하한/왜곡된 값이 된다. uncertainty monitoring을 위해 entropy를 켰다면 full-vocabulary normalization 비용과 privacy/cardinality를 예산화한다. 매 request 매 step 전체 분포를 CPU로 복사하는 방식은 피한다.

beam search는 누적 log probability로 후보를 비교하므로 log-softmax가 필요할 수 있다. raw logits의 합은 step별 arbitrary offset 때문에 sequence score가 아니다. 각 step LSE를 빼야 조건부 logprob가 된다. beam의 전체 계약은 18장에 두되, raw score와 normalized transition score를 구별하는 이유는 여기서 고정한다.

perplexity나 prompt scoring도 정답 token의 logprob가 필요하지만 full probability vector를 반환할 필요는 없다. fused cross entropy는 logsumexp와 target logit gather를 결합해 `[M,V]` probability를 저장하지 않을 수 있다. “확률을 계산한다”는 수학적 필요와 “확률 tensor를 materialize한다”는 구현 선택은 같은 문장이 아니다.

## 17.4 logsumexp가 overflow 없이 확률을 만든다

### 17.4.1 max shift·sum-exp·normalization을 다른 상태로 기록한다

naive softmax는 큰 logit에서 overflow한다. FP32에서 `exp(1000)`은 표현할 수 없다. `m=max(z)`를 빼면 `softmax(z_i)=exp(z_i-m)/Σ exp(z_j-m)`이고 가장 큰 exponent가 1이 된다. 공통 상수를 빼도 분포가 같다는 성질을 수치 안정성에 사용한 것이다.

log normalization constant는 `LSE(z)=m+log Σ_j exp(z_j-m)`이다. token `i`의 log probability는 `log p_i=z_i-LSE(z)`다. 확률을 먼저 계산한 뒤 log를 취하는 것보다 underflow에 강하다. 아주 작은 확률이 0으로 round되어 `log(0)`가 되는 일을 피한다.

작은 손계산을 하자. logits `[1000,999,997]`에서 `m=1000`, shifted logits는 `[0,-1,-3]`이다. exponent는 대략 `[1,0.3679,0.0498]`, 합은 `1.4177`이다. 확률은 `[0.7054,0.2595,0.0351]`, LSE는 `1000+log(1.4177)≈1000.349`다. 원래 값에 직접 exp하지 않아도 정확한 비율을 얻는다.

### 분산 softmax도 global max와 global sum이 필요하다

vocabulary shard마다 local max만 빼고 local sum으로 normalize하면 rank별 확률 합이 각각 1이 되어 틀린 global 분포다. 먼저 local max들을 all-reduce MAX하여 global `m`을 얻는다. 각 rank가 `exp(z_local-m)`의 local sum을 계산하고 all-reduce SUM하여 global denominator를 얻는다. 요청 token logit이 어느 shard에 있는지도 찾아야 한다.

통신을 줄이는 distributed logprob 구현은 이 두 scalar-like reduction과 selected token gather를 이용할 수 있다. 전체 `[M,V]`를 gather하지 않아도 `[M]` global max와 sum, 관심 ID logit으로 selected logprob를 계산한다. 반면 top-N logprobs를 반환하려면 global top candidates merge가 추가된다.

두 shard 손계산으로 확인하자. rank 0 logits `[4,1]`, rank 1 logits `[3,2]`이면 global max는 4다. rank 0 local shifted sum은 `1+exp(-3)≈1.0498`, rank 1은 `exp(-1)+exp(-2)≈0.5032`다. global denominator는 1.5530이고 ID 0 확률은 약 0.6439다. 각 rank가 local softmax하면 rank 0 ID 0을 0.9526으로 잘못 보고한다.

global token ID `y`의 owner rank를 계산할 때 uneven shard와 padded rows를 고려한다. 단순 `y // (V/TP)`가 실제 partition mapping과 다를 수 있다. owner가 target logit을 내고 다른 rank는 `-inf` 또는 0 contribution을 낸 뒤 collective로 합친다. mapping 오류는 finite하지만 다른 token의 logprob를 반환한다.

collective dtype도 결과에 영향을 준다. local logits이 FP16/BF16이어도 max와 sum을 FP32로 승격하는 구현이 많다. bandwidth를 줄이려고 낮은 dtype으로 sum하면 긴 vocabulary의 작은 항 누적 오차가 커질 수 있다. reference와 비교할 때 head projection 오차와 normalization reduction 오차를 별도로 측정한다.

online logsumexp는 vocabulary tile을 한 번에 보지 않아도 running max `m`과 scaled sum `l`을 갱신한다. 새 tile max가 더 크면 이전 sum을 `exp(m_old-m_new)`로 rescale한 뒤 새 항을 더한다. attention의 online softmax와 같은 수학 구조지만 축과 consumer가 다르다. 여기서는 vocabulary 후보 정규화이고 attention에서는 key position 가중치다.

NaN과 positive infinity의 policy도 확인한다. 하나의 NaN이 max/reduction을 오염시킬 수 있다. 모든 logits가 `-inf`가 되면 denominator가 0인 invalid distribution이다. grammar나 allowed-token mask가 후보를 전부 제거할 때 생길 수 있다. sampler가 오류를 내는지 fallback token을 택하는지 18장에서 보되, 이 장에서는 raw/processed 어느 단계에서 non-finite가 시작됐는지 분리한다.

positive infinity가 하나면 수학적 극한은 그 후보 확률 1이지만 naive `inf-inf`는 NaN이다. 여러 `+inf` 후보면 질량 분배 policy가 필요하다. runtime이 sanitize하는지 오류를 내는지 source와 fixture로 고정한다. 조용히 0으로 바꾸면 upstream overflow 증거를 숨길 수 있다.

logsumexp fixture는 dtype별 tolerance를 가진다. FP64 CPU 기준값과 FP32 implementation, model dtype projection을 분리 비교한다. 최종 probability 오차만 보면 projection과 reduction 원인이 섞인다. 동일 raw FP32 logits를 normalization 함수에 넣는 test가 reduction을 격리한다.

## 17.5 logprobs는 점수표가 아니라 조건부 증거다

next-token logprob `log P(x_t | x_<t)`는 해당 prefix 아래 실제 token의 조건부 점수다. sequence log likelihood는 token logprobs의 합이다. 길이가 길수록 음수 항이 더해지므로 서로 다른 길이의 문장을 단순 합으로 비교하면 짧은 문장을 선호한다. 평균, length normalization, task-specific calibration의 목적을 명시해야 한다.

prompt logprobs는 prompt token `x_t`를 예측한 직전 row `t-1`의 logits에서 읽는다. 첫 token은 BOS나 외부 context가 없으면 정의 방식이 model/API에 따라 다르다. output logprobs는 생성 step에서 선택된 token과 후보들의 점수다. 둘을 같은 배열로 보여도 row alignment와 소유 시점이 다르다.

### 한 칸 shift가 cross entropy의 핵심이다

causal LM에서 position `t`의 hidden은 다음 token `x_{t+1}`을 예측한다. logits `[S,V]`와 labels `[S]`를 같은 index로 직접 비교하는 것이 아니라 logits의 마지막 row를 제외하고 labels의 첫 token을 제외해 맞춘다. padding과 ignored labels는 loss에 포함하지 않는다.

정답 token `y`의 cross entropy는 `-log p_y = LSE(z)-z_y`다. 정답 logit을 높이거나 경쟁 후보를 낮추면 loss가 줄어든다. perplexity는 평균 token NLL의 exponential이다. tokenizer가 다르면 token 분해와 token 수가 달라 perplexity를 직접 비교하기 어렵다.

손계산 logits `[2,-2,1]`에서 정답이 ID 2라면 앞서 probability는 대략 `[0.721,0.013,0.265]`이고 NLL은 `-log(0.265)≈1.33`이다. 정답이 ID 0이면 약 `0.327`이다. argmax accuracy는 둘 중 하나를 0/1로만 보지만 NLL은 margin과 전체 경쟁 질량을 반영한다.

token sequence `[BOS,A,B]`가 있으면 BOS row의 logits에서 A logprob를, A row에서 B logprob를 읽는다. B row는 그 뒤 token이 prompt에 없으므로 prompt logprob 대상이 아니다. API가 첫 prompt token에 null을 반환하는 것은 버그가 아니라 preceding context가 없는 정렬 표현일 수 있다. template가 BOS를 삽입하면 첫 visible user token에도 preceding hidden이 존재할 수 있다.

padding batch에서는 labels가 PAD인 위치를 ignore index로 바꾼다. attention mask가 PAD를 attend하지 못하게 하는 것과 loss에서 PAD target을 제외하는 것은 별도 계약이다. prompt logprob serving에서도 반환 span이 padding row를 포함하지 않아야 한다. mask sum과 returned prompt-logprob count가 어떻게 연결되는지 request별로 검사한다.

truncation이 prompt 앞을 제거하면 첫 남은 token의 logprob 정의가 달라진다. 잘리기 전 prefix가 KV cache에 남아 있는가, 완전히 제거되었는가에 따라 조건부 context가 다르다. response에 original character offset만 붙이면 오해할 수 있다. final token span과 effective prefix length를 함께 반환하거나 trace한다.

packed sequence는 서로 다른 request 사이에 label shift가 넘어가지 않도록 boundary를 가진다. request A 마지막 logits를 request B 첫 token label과 비교하면 loss는 finite하지만 무의미하다. cumulative sequence length나 row-to-request mapping이 shift/gather 단계에서도 사용되는지 본다. 이 버그는 전체 평균 loss가 조금 나빠지는 형태로 숨어 있을 수 있다.

cross entropy를 디버깅할 때 정답 token logit, LSE, NLL 세 값을 분리한다. NLL이 달라졌는데 target logit은 같다면 경쟁 후보나 denominator가 달라졌다. LSE는 같은데 target logit만 다르면 해당 vocabulary row projection을 본다. 둘 다 다르면 hidden/head 전체 변화 가능성이 크다.

vLLM의 logits processor는 prompt logprob mode에서 필요한 prompt rows와 sampled rows를 구분한다. [`forward metadata와 logprob 계산 경로`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/logits_processor.py#L134-L211)는 hidden selection, logits, logprob result가 request metadata에 의존함을 보여 준다. `prompt_logprobs` field는 단지 response decoration이 아니라 projection row와 reduction work를 늘릴 수 있다.

SGLang도 [`LogitsMetadata`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/logits_processor.py#L30-L77)에 extend length와 logprob 관련 상태를 모은다. field가 metadata branch를 바꾸고 selected hidden row와 normalized prompt logprob tensor를 만든다. 효과는 관측 가능성 증가와 TTFT/memory 비용이다. 반증은 retained row 수, LM-head input shape, collective bytes, 반환 token alignment다.

### logprob API를 확률 보정으로 오해하지 않는다

높은 model probability가 factual correctness를 보장하지 않는다. logprob는 model distribution 안에서의 상대적 선호다. prompt wording, template, temperature 전 raw/processed 기준, quantization에 따라 달라진다. 서로 다른 model의 raw logprob를 confidence로 직접 비교하려면 calibration이 필요하다.

API가 반환하는 top logprobs가 vocabulary 전체를 뜻하지도 않는다. 상위 N 후보와 선택 token만 반환할 수 있고, 합이 1보다 작다. 반환 후보를 다시 normalize하면 원래 분포가 아니라 truncated conditional distribution이 된다. audit에는 requested N, raw/processed 기준, omitted mass 가능성을 기록한다.

raw logprobs와 processed logprobs를 구별한다. repetition penalty나 grammar 전 model distribution을 보고 싶은 분석과 실제 sampler가 선택한 policy distribution을 재현하려는 분석은 다른 값을 요구한다. API 이름이 `logprobs` 하나뿐이면 source에서 processor 적용 전후 어느 tensor에 log-softmax하는지 확인한다. 문서 표현만으로 추측하지 않는다.

temperature도 반환 의미를 바꾼다. raw logits에 대한 logprob와 temperature-scaled distribution의 logprob는 다르다. greedy branch가 softmax를 생략해도 API가 logprob를 요청하면 별도로 normalization해야 할 수 있다. 그래서 temperature 0이면서 logprobs on인 request는 token selection과 score reporting이 서로 다른 computation을 요구할 수 있다.

top-N의 tie ordering도 재현에 중요하다. 같은 logit 후보가 여러 개면 token ID ascending, stable sort, backend-specific top-k가 순서를 정할 수 있다. 확률값은 같아도 JSON 후보 순서가 달라 snapshot test가 실패할 수 있다. semantic equality와 deterministic serialization 요구를 분리한다.

prompt logprobs는 민감 정보를 드러낼 수 있다. 특정 secret token이 context에서 얼마나 예상되는지 반복 질의하는 side channel을 고려한다. 최대 N, prompt 길이, tenant authorization과 rate limit을 둔다. 성능 option인 동시에 정보 노출 surface다.

운영 metric에 raw token 문자열을 label로 넣지 않는다. logprob histogram, top-1 margin bucket, non-finite count처럼 bounded aggregate를 쓰고 상세 candidate는 접근 통제된 trace에 sampling한다. vocabulary ID도 model revision 없이는 의미가 고정되지 않으므로 fingerprint와 묶는다.

## 17.6 logit divergence는 더 위의 오류를 비추는 계기판이다

두 backend의 답이 달라졌을 때 first-token raw logits는 강력한 이분점이다. final input IDs, positions, mask, selected hidden row가 같고 raw logits가 다르면 model computation·weight·quantization을 본다. raw logits는 같은데 processed scores가 다르면 sampling policy를 본다. selected IDs도 같은데 text가 다르면 detokenization을 본다.

### padding과 position 오류는 head에서 확대되어 보인다

LM head는 hidden error를 만들지 않고 vocabulary 좌표로 projection한다. hidden 차이 `δh`가 있으면 logit 차이는 `δz=W δh`다. 큰 row norm이나 특정 방향 정렬 때문에 작은 hidden 차이가 일부 token margin을 크게 바꿀 수 있다. logits divergence가 발견된 층과 원인이 발생한 층을 구별한다.

single과 mixed-length batch의 selected hidden row를 저장하고 norm과 checksum을 비교한다. hidden부터 다르면 padding mask, logical position, KV ownership으로 위를 추적한다. hidden은 같은데 logits가 다르면 LM head weight shard, dtype, quantization, gather 순서를 본다. text만 비교하는 것보다 훨씬 좁은 분기다.

### quantization은 오차와 decision margin의 관계로 판정한다

quantized LM head는 weight row를 낮은 bit로 저장하고 scale/zero point로 dequantize하거나 fused GEMM에서 누산한다. 옵션 field가 quantized linear class를 선택하고 packed weight와 scale tensor를 만든다. kernel이 accumulator dtype과 dequant order를 결정한다. 효과는 memory bandwidth와 capacity 이득, logit 오차다.

top-1 token이 reference와 같다고 충분하지 않다. reference margin이 0.0001인데 maximum logit error가 0.01이면 다음 prompt에서 쉽게 뒤집힐 수 있다. top-N overlap, selected-token logprob error, KL 같은 통계와 margin-conditioned flip rate를 본다. 반대로 큰 공통 offset은 softmax에 영향이 없으므로 raw absolute error만으로 과대평가할 수 있다.

weight tying model에서 input embedding과 LM head의 quantization policy가 다를 수 있는지도 본다. framework가 alias를 유지하지 않고 서로 다른 module wrapper로 바꾸면 tied semantics와 memory 예상이 달라진다. config 문자열보다 실제 module class, parameter storage, packed shape와 kernel dispatch를 기록한다.

### non-finite의 최초 발생 위치를 찾는다

raw logits에 NaN이 있으면 softmax 안정화만 고쳐서는 안 된다. final hidden이 이미 NaN인지, LM head accumulation에서 overflow했는지 확인한다. hidden이 finite이고 특정 shard logits만 NaN이면 해당 weight shard와 quant kernel을 좁힌다. raw는 finite인데 processor 후 모두 `-inf`라면 allowed set이나 grammar가 원인이다.

vLLM의 sampler는 [`logits non-finite 처리와 sampling 입구`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L49-L120)에서 logits state를 받아 후속 policy로 보낸다. 여기서 이 장은 raw와 normalized 필요성까지만 보고 top-k/p와 multinomial 순서는 18장으로 넘긴다.

장애 A는 backend 교체 뒤 greedy 첫 token이 0.2% 요청에서 달라진 경우다. final IDs와 positions를 고정하고 final hidden을 비교한다. hidden maximum error가 이미 크면 LM head를 건너뛰고 최초 layer divergence를 찾는다. hidden은 허용 오차 안인데 logit flip이 나면 top-2 margin과 head projection error를 함께 그린다.

장애 B는 prompt logprobs를 켰을 때만 OOM이 나는 경우다. request당 prompt 길이와 retained LM-head rows를 곱해 `[M,V]` allocation을 계산한다. TP gather가 rank마다 full tensor를 복제하는지, FP32 upcast가 있는지, response top-N과 무관하게 full logits가 잠시 존재하는지 본다. top-N을 줄였는데 OOM이 그대로인 이유가 intermediate에 있을 수 있다.

장애 C는 TP 크기를 바꾸자 logprob만 조금 달라지고 greedy token은 같은 경우다. local GEMM partition, reduction tree와 dtype이 달라 수치 합 순서가 변할 수 있다. raw gathered logits 차이와 LSE reduction 차이를 분리하고 SLA tolerance를 정한다. bitwise equality를 무리하게 요구하거나 모든 차이를 정상으로 치부하지 않는다.

장애 D는 tokenizer vocabulary 끝 근처의 존재하지 않는 ID가 가끔 선택되는 경우다. physical padded vocabulary row가 mask되지 않았을 가능성을 본다. selected global ID가 `V_tokenizer` 이상인지 즉시 검증하고 head physical shape와 trim 위치를 추적한다. detokenizer에서 UNK로 보이기 전에 sampler boundary에서 막아야 한다.

장애 E는 adapter를 바꿨는데 logits가 전혀 변하지 않는 경우다. adapter가 transformer layer에만 적용되고 LM head에는 원래 적용되지 않는 구성일 수도 있다. 반대로 row-to-adapter mapping이 compacted hidden과 어긋나 base head만 실행됐을 수 있다. expected adapter target modules와 actual module hooks, delta norm을 비교한다.

장애 F는 raw logits가 모두 finite인데 softmax 뒤 NaN이다. 모든 후보가 mask되어 `-inf`가 되었는지, temperature가 0인데 division branch를 탔는지, low-precision exp/sum이 overflow했는지 본다. raw는 “model head 직후”, processed는 각 transform 뒤로 checkpoint해 최초 non-finite stage를 고정한다.

## 17.7 네 구현의 함수 경로를 같은 질문으로 읽는다

### Transformers: model head와 generation loop 사이

model별 causal LM class는 base model output hidden을 `lm_head`에 넣는다. 예를 들어 [`LlamaForCausalLM.forward`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/modeling_llama.py#L421-L490)는 필요한 sequence slice와 LM head projection, loss 경계를 보여 준다. version에 따라 `logits_to_keep` 같은 인자가 head 전 row selection을 가능하게 한다.

generation loop의 [`_sample` 핵심`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2826-L2935)은 model output의 마지막 logits row를 가져와 processor에 전달하고 greedy 또는 sampling branch로 token을 고른다. `output_scores`와 `output_logits`는 반환 저장량을 바꾼다. field→return-state branch→tuple accumulation→host/device memory와 응답 관측 효과로 이어진다.

[`compute_transition_scores`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L1383-L1465)는 저장된 generation scores와 selected sequences를 맞춰 transition score를 복원한다. beam index와 normalization 여부 때문에 단순히 매 step top probability를 더하는 것과 다를 수 있다. 이 장에서는 row/token alignment의 사례로 읽는다.

Transformers 경로를 실제로 읽을 때 `forward` signature에서 `logits_to_keep` 지원 여부를 먼저 본다. generation utility가 field를 전달해도 model이 받지 않으면 최적화가 적용되지 않는다. 지원 predicate, kwargs construction, model slice, head input shape를 연결한다. option을 켰다는 로그보다 profiler의 GEMM M축과 output allocation이 반증이다.

`output_logits=True`와 `output_scores=True`도 같은 tensor를 뜻한다고 가정하지 않는다. raw model logits와 processor 이후 scores를 각각 저장할 수 있고 generation 길이만큼 tuple이 자란다. debugging에는 유용하지만 긴 generation과 큰 batch에서는 device memory를 붙잡을 수 있다. 필요한 step만 sampling trace로 남기는 운영 경로와 연구용 full return을 나눈다.

loss가 요청되면 model forward가 labels를 loss function에 전달하며 shift가 model/loss abstraction 어디에서 일어나는지 version별로 확인한다. custom head나 model fork가 shift를 두 번 하거나 하지 않는 사고를 막으려면 toy sequence의 expected target IDs를 hook으로 확인한다.

### vLLM과 SGLang: selected hidden과 distributed normalization

vLLM은 model runner가 필요한 hidden을 logits processor에 넘기고, processor가 vocabulary projection과 logprob 결과를 만든다. TP gather를 생략할 수 있는 경로에서는 local vocabulary range와 global token ID mapping이 핵심이다. 요청한 prompt/output logprobs가 projection rows와 collective를 어떻게 늘리는지 profiler보다 shape ledger를 먼저 본다.

SGLang의 logits processor는 forward mode별 hidden selection을 명시한다. extend request가 여러 prompt logprobs를 요구할 때 last row 하나만 남기는 decode fast path와 같을 수 없다. `return_logprob`, top-logprobs 수, start length가 metadata를 바꾸고 selected indices와 normalized tensor를 바꾼다. 반증은 같은 request에서 flag on/off의 LM-head rows와 TTFT다.

vLLM에서 `logprobs=N` request field는 scheduler token budget을 직접 바꾸지 않을 수 있지만 compute/output memory와 serialization을 바꾼다. model runner가 hidden을 보존하는 범위, logits processor가 gather하는 범위, sampler가 top candidates를 CPU-visible result로 만드는 범위를 따라간다. 어느 worker가 결과를 소유하는지도 distributed deployment에서 중요하다.

prompt logprobs는 chunked prefill과 만날 때 여러 chunk의 결과를 원래 prompt 순서로 조립해야 한다. chunk 첫 row의 label은 이전 chunk 마지막 hidden이 예측한다. chunk boundary를 request boundary처럼 취급해 한 token을 빠뜨리거나 중복하지 않는지 본다. cached prefix에 대한 logprob를 재계산할지 반환 불가로 둘지도 명시한다.

SGLang의 extend metadata에서 `logprob_start_len`은 전체 prompt 시작이 아니라 사용자가 점수를 원하는 suffix 경계일 수 있다. cache hit와 prefix length를 뺀 local row index로 변환되는 지점을 찾는다. field→index branch→selected hidden rows→returned token positions→projection/latency 효과를 하나의 예로 trace한다.

두 engine 모두 custom logits processor가 있으면 distributed optimization을 제한할 수 있다. processor가 full vocabulary tensor와 Python callback을 요구하면 GPU-resident distributed top-k fast path를 쓰기 어렵다. extension API의 표현력이 collective와 D2H 비용으로 환산되는 지점을 문서화한다.

### llama.cpp: graph output row와 CPU sampler의 경계

llama.cpp는 logical batch의 어느 token에서 logits가 필요한지 output mask로 지정할 수 있다. [`llama_batch`와 logits flag 계약](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/include/llama.h#L191-L235)은 token별 logits 요청이 batch metadata임을 보여 준다. 모든 prompt row를 출력하지 않으면 head와 readback 비용을 줄일 수 있다.

decode 뒤 [`llama_get_logits`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-context.cpp#L736-L755)는 계산된 output logits buffer를 application에 노출한다. 반환 pointer의 row ordering은 batch의 requested-output ordering과 맞춰야 한다. request token index와 compacted output row index를 혼동하면 다른 sequence의 분포를 sampling한다.

llama.cpp sampler chain의 softmax와 temperature는 [`llama-sampling.cpp`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-sampler.cpp#L270-L319)에서 candidate array를 정렬하고 확률을 계산하는 경계로 나타난다. model graph가 언제나 probability를 출력하는 것이 아니라 application-side candidate processing이 필요할 때 softmax를 수행한다는 증거다.

`llama_batch.logits[i]`가 false인 row는 output buffer에서 자리를 차지하지 않을 수 있다. application이 원래 token index `i`로 `llama_get_logits_ith`를 조회할 수 있다고 가정하면 compacted ordering과 충돌한다. batch construction 때 output index mapping을 저장하고 decode 후 같은 lifetime에 읽는다.

GPU offload 구성에서는 LM head가 GPU에 있을 수도 CPU에 있을 수도 있고 logits buffer가 host-visible하기 위해 copy될 수 있다. `n_gpu_layers` 같은 높은 수준 field가 실제 graph placement branch를 거쳐 output tensor backend를 바꾸는지 본다. 효과는 head GEMM 위치와 D2H bytes, sampler latency다. 반증은 graph node backend, transfer trace, logits checksum이다.

quantized GGUF head tensor type은 model-wide 이름만으로 확정하지 않는다. output weight가 별도 type을 가지거나 tensor override가 적용될 수 있다. loaded tensor metadata, selected matmul kernel, accumulator type을 기록하고 FP32/F16 reference projection과 top margin별로 비교한다.

## 17.8 손계산과 shape ledger로 score path를 검증한다

request 두 개가 있고 마지막 hidden만 필요하다고 하자. `M=2`, `D=4`, `V=6`이면 selected hidden은 `[2,4]`, head weight는 `[6,4]`, logits는 `[2,6]`다. TP=2 vocabulary sharding이면 각 rank weight `[3,4]`, local logits `[2,3]`다. 전체 gather는 `[2,6]`, distributed greedy라면 각 rank의 `[2]` local max와 ID 후보를 global 비교한다.

prompt 길이가 각각 3과 5이고 모든 prompt token logprob를 요청하면 예측 가능한 label row는 BOS 정책을 제외해 대략 `(3-1)+(5-1)=6`개다. head input이 `[6,4]`, local logits `[6,3]`로 늘어난다. `prompt_logprobs`를 켰을 때 TTFT가 늘어난 이유를 response JSON serialization만으로 설명하면 이 projection 증가를 놓친다.

### 최소 수치 fixture

fixture A는 logits `[0,0]`이다. softmax는 `[0.5,0.5]`이고 argmax tie-break는 구현 규칙을 따른다. fixture B는 `[1000,999]`이며 stable softmax로 계산해야 한다. fixture C는 `[0,-inf,-inf]`로 allowed token 하나만 남은 경우다. fixture D는 모두 `-inf`여서 invalid distribution 처리 policy를 확인한다.

fixture E는 두 TP shard `[2,0]`, `[1,3]`이다. global argmax는 두 번째 shard의 logit 3이다. shard별 softmax를 합치면 틀리고, global max 3을 기준으로 `[exp(-1),exp(-3),exp(-2),1]`을 normalize해야 한다. selected token logprob만 필요해도 global denominator는 필요하다.

fixture F는 reference logits `[5,4.999,0]`와 quantized `[4.998,5.001,0.002]`다. maximum absolute error는 작아 보여도 top-1이 뒤집힌다. 반대로 모든 logits에 `+10` 오차가 더해진 경우 absolute error는 크지만 distribution은 같다. validation metric이 invariance와 margin을 반영해야 한다.

fixture G는 temperature와 bias 순서를 확인한다. raw `[2,1]`, bias `[0,1]`, `T=2`라 하자. bias 후 scale은 `[1,1]`이지만 scale 후 bias는 `[1,1.5]`다. option 두 개의 값은 같아도 결과가 다르다. runtime의 processor list에서 실제 순서를 읽고 예상 logits를 손으로 계산한다.

fixture H는 prompt shift다. IDs `[10,20,30]`에 대해 row 0은 label 20, row 1은 label 30을 점수화하고 row 2에는 prompt label이 없다. 반환 배열이 token별 세 칸이면 첫 칸 null 또는 BOS-conditioned score의 의미를 문서화한다. padding을 앞에 두어도 유효 token alignment가 유지되는지 확인한다.

fixture I는 vocabulary padding이다. visible V=5, physical V=8이고 logits `[0,1,2,3,4,100,100,100]`을 만든다. 올바른 selected ID는 4다. physical padding row가 argmax에 들어가면 즉시 실패한다. TP shard가 padding 구간을 소유하는 구성에서 mask/trim이 collective 전후 어디에 있는지 검증한다.

fixture J는 all-masked row다. grammar나 allowed-token set을 빈 집합으로 만들어 processed scores가 모두 `-inf`가 되는 조건을 직접 구성한다. 기대가 명시적 오류인지 EOS fallback인지 정하고, NaN multinomial까지 내려가지 않도록 assert한다. 정책을 정하지 않은 silent fallback은 재현성과 안전성을 모두 해친다.

fixture K는 공통 offset invariance다. random vector z와 `z+10,000`을 stable log-softmax에 넣어 결과 차이를 본다. input dtype이 FP16이면 큰 offset을 더하는 과정에서 작은 차이가 소실될 수 있으므로 FP32 변환 시점도 확인한다. 수학적 invariance가 유한 정밀도 representation에서는 입력 quantization 때문에 깨질 수 있다.

fixture L은 shard reduction이다. 같은 global logits를 TP=1,2,4 partition으로 나누고 global LSE와 selected logprob를 비교한다. partition마다 reduction 순서가 달라 tolerance 내 차이는 가능하지만 owner mapping과 padded row 때문에 큰 차이가 나서는 안 된다. collective 직전 local max/sum을 보존하면 원인을 재구성할 수 있다.

### shape ledger는 성능 주장을 산술로 바꾼다

ledger 첫 줄은 model 상수다. hidden size D, visible vocabulary V, physical padded vocabulary Vp, model dtype과 logits dtype, TP size P를 적는다. 둘째 줄은 step 상태다. active requests B, model이 계산한 token rows T, head가 보존한 rows M, prompt-logprob rows L을 적는다. `T`와 `M`이 같다고 가정하지 않는다.

단일 head projection의 대략적 multiply-add work는 `M×D×Vp`에 비례한다. weight bytes는 cache reuse와 quantization에 좌우되고 output bytes는 `M×Vp×logit_bytes`다. local vocab sharding이면 rank당 Vp/P지만 gather 뒤 full output이 생길 수 있다. 이 식은 kernel latency를 정확히 예측하지 않지만 어떤 option이 어느 항을 키우는지 드러낸다.

예를 들어 D=8192, Vp=131072, M=1이면 matvec 성격의 약 10억 곱셈 항을 가진다. M=256이면 큰 GEMM으로 arithmetic intensity와 utilization이 달라진다. decode batching이 head 효율을 높이는 이유와 prompt-logprob가 work를 늘리면서도 GEMM 효율은 높일 수 있는 이유를 동시에 설명한다. tokens/s 숫자 하나로는 이 차이를 볼 수 없다.

output FP32 정책은 model dtype과 별도다. hidden과 weight가 BF16이어도 logits를 FP32로 변환하면 output bytes는 4다. sampler 안정성에는 좋지만 V가 크면 memory가 두드러진다. code에서 cast가 GEMM 전 accumulator, GEMM output, processor 입구 중 어디에 있는지 확인한다. `dtype=model.dtype`이라는 설정만으로 logits dtype을 추측하지 않는다.

gather 통신의 단순 payload 하한은 full logits라면 `M×Vp×bytes` 규모다. 실제 all-gather algorithm은 rank 수와 topology에 따라 wire bytes와 latency가 다르다. distributed greedy는 rank당 M개의 value/ID pair reduction으로 훨씬 작을 수 있다. logprob는 M개의 max/sum reductions와 selected logit owner communication이 필요하다. 요구 output별 lower bound를 비교한다.

response serialization은 top-N에 비례한다. full logits intermediate 비용은 N과 무관할 수 있지만 GPU→CPU candidate copy와 JSON 크기는 `M×N`에 가깝다. N을 5에서 20으로 늘렸을 때 GPU head 시간은 그대로이고 CPU latency만 늘 수도 있다. stage별 timer와 bytes로 field effect를 분리한다.

CUDA Graph는 최대 M 또는 bucket shape에 맞춘 static buffer를 가질 수 있다. 실제 M이 작아도 allocated logits workspace는 capture bucket 크기일 수 있다. dummy rows가 sampling 결과에 섞이지 않도록 valid row count와 mask가 필요하다. profiler memory만 보고 매 step 실제 full work가 수행된다고 단정하지 않고 kernel grid와 valid count를 본다.

speculative verification은 한 request에서 여러 proposed token row의 logits가 필요하다. 일반 decode M=B와 달리 M이 proposed length 합까지 늘어난다. target model이 각 proposal의 conditional distribution을 검증하기 때문이다. 이 장은 acceptance 수식을 다루지 않지만 head row count와 logits memory가 decode-one-token 가정에서 벗어나는 이유를 ledger에 넣는다.

### logit differential은 layer bisect의 출발점이다

reference와 candidate가 다를 때 먼저 exact same artifact/input 조건을 확보한다. tokenizer, template, model weights, adapter, quantization이 다르면 backend comparison이 아니다. deterministic greedy와 cache state를 고정하고 같은 selected row의 raw logits를 캡처한다. sampling 결과 text는 증거로는 너무 멀리 있다.

logits 차이를 네 요약으로 본다. maximum absolute/relative error, cosine 또는 centered correlation, top-k overlap, top-1 margin 대비 error다. 공통 offset을 제거한 centered logits도 비교한다. softmax-invariant offset과 실제 ranking distortion을 구별하기 위해서다. 단, temperature나 absolute logsumexp가 필요한 scoring에서는 offset 처리 전 원본도 보존한다.

head input hidden이 다르면 final normalization output부터 역으로 layer boundary checksum을 비교한다. 모든 tensor를 저장하지 않고 binary search처럼 중간 layer를 선택한다. 최초 divergent layer를 찾으면 attention output, MLP output, residual과 cache read를 분리한다. CUDA kernel 상세는 뒤 장으로 넘기되 이 장의 logits가 bisect trigger다.

hidden은 같은데 head output만 다르면 head weight logical matrix를 비교한다. quantized packed bytes 자체보다 dequantized sampled rows와 scale metadata를 비교한다. TP shard vocabulary range가 같은지, transpose/layout이 맞는지, bias/softcap이 있는지 본다. 한 token row만 크게 다르면 해당 vocabulary row의 corruption 가능성이 있다.

local logits는 같고 global 결과만 다르면 collective 단계다. rank order, shard offsets, uneven/padded trim, stream synchronization, output buffer reuse를 본다. async collective가 끝나기 전 sampler가 읽는 race는 간헐적이며 batch 크기에 따라 달라질 수 있다. event dependency와 buffer lifetime을 trace한다.

raw global logits도 같으면 processor 전후 checkpoint를 둔다. temperature, bias, penalty, allowed mask 각각 뒤의 top candidates와 non-finite count를 비교한다. 최초 다른 transform의 field와 state를 찾는다. 18장에서 전체 순서를 다루므로 여기서는 raw boundary를 확실히 넘겨주는 데 집중한다.

processed scores까지 같고 selected token만 다르면 RNG state나 categorical sampler로 넘어간다. selected ID도 같은데 finish/text가 다르면 stop과 detokenization이다. 이렇게 최초 divergence를 경계별로 찾으면 “GPU가 비결정적” 같은 큰 결론을 증거 없이 내리지 않게 된다.

### 확률을 직관으로 읽되 뜻을 과장하지 않는다

logit을 산의 높이, softmax를 높이에 따른 인구 분포로 비유할 수 있다. temperature가 낮으면 높은 봉우리에 인구가 몰리고 높으면 골짜기에도 퍼진다. 그러나 비유에서 절대 해발은 중요해 보이지만 softmax에서는 모든 봉우리에 같은 높이를 더해도 분포가 같다. 실제 의미는 상대 높이와 후보 집합이다.

후보 집합도 분포 해석에 포함된다. 같은 token logit 5라도 경쟁 vocabulary logits가 모두 0이면 확률이 높고, 수천 후보가 4.9라면 낮을 수 있다. selected token logit 하나만 로그해 confidence를 계산할 수 없는 이유다. 최소한 LSE 또는 충분한 경쟁 분포가 필요하다.

모델 확률은 “세계에서 이 문장이 참일 확률”이 아니다. training distribution과 주어진 prefix 아래 다음 token을 model이 얼마나 선호하는지다. 허구 문체를 강하게 요구한 prompt에서는 사실이 아닌 token에 높은 확률을 주는 것이 objective에 맞을 수 있다. factual confidence dashboard가 token logprob를 그대로 사용하면 의미 층을 혼동한다.

tokenization도 직관을 흔든다. 한 단어가 token 하나인 경우와 세 token인 경우 sequence logprob는 항의 수가 다르다. 문자열 후보를 비교하려면 동일 prefix에서 각 candidate의 전체 token conditional logprobs를 합하고 length policy를 명시한다. 첫 token logprob만 비교하면 공통 prefix나 후속 spelling 제약을 놓친다.

공백과 대소문자도 별도 tokenization을 만든다. API가 top logprobs에 사람이 같은 단어로 보는 pieces를 나눠 보여 줄 수 있다. UI에서 token pieces를 단어 확률처럼 합치려면 가능한 segmentations와 conditional path를 고려해야 한다. 단순히 같은 표시 문자열 후보의 확률을 더하는 것이 항상 올바르지 않다.

calibration은 별도 평가 문제다. 예측 probability 0.8인 사건이 장기적으로 약 80% 맞는지 task label로 측정한다. temperature scaling 같은 post-hoc calibration의 temperature는 generation diversity option과 수학 형태가 비슷해도 목적과 fit 과정이 다르다. serving request temperature를 바꿔 factual calibration을 자동 해결했다고 말하지 않는다.

entropy가 높으면 model이 후보 사이에서 퍼져 있다는 뜻이지만 원인을 말하지 않는다. 진짜 모호한 질문, 낯선 domain, 여러 자연스러운 표현, 잘못된 template 모두 entropy를 높일 수 있다. entropy가 낮아도 자신 있게 틀릴 수 있다. 관측 신호를 원인 판정으로 승격하려면 controlled fixture와 task outcome이 필요하다.

### 한 요청의 숫자를 끝까지 복원한다

vocabulary 네 개, hidden size 세 개의 toy model을 만들자. selected hidden `h=[1,2,-1]`, head rows `w0=[1,0,0]`, `w1=[0,1,0]`, `w2=[0,0,1]`, `w3=[1,1,0]`이면 raw logits는 `[1,2,-1,3]`이다. shape ledger는 `[1,3]×[3,4]→[1,4]`다.

temperature 2만 적용하면 processed scores `[0.5,1,-0.5,1.5]`다. max 1.5를 빼 exponent를 계산하면 약 `[0.368,0.607,0.135,1]`, 합 2.110이다. probability는 대략 `[0.174,0.288,0.064,0.474]`다. greedy라면 raw에서도 processed에서도 ID 3이며 softmax를 건너뛸 수 있다.

정답 token이 ID 1이면 logprob는 `1- LSE(processed)`다. shifted sum을 썼으므로 `LSE=1.5+log(2.110)≈2.247`, logprob는 약 `-1.247`이다. raw distribution의 ID 1 logprob와는 다르다. API가 어느 값을 반환하는지 명시하지 않으면 사용자가 재현할 수 없다.

이 head가 TP=2로 `[w0,w1]`과 `[w2,w3]`에 나뉘면 local logits `[1,2]`, `[-1,3]`다. distributed greedy는 local 후보 `(2,ID1)`과 `(3,ID3)`를 비교한다. logprob에는 global max와 sum이 필요하다. local rank 0 softmax에서 ID1이 약 0.731이라고 반환하면 global 0.288과 크게 다르다.

physical padding으로 두 번째 shard에 dummy rows 두 개가 더 있고 logit 9가 나왔다면 visible V=4 trim 전 argmax를 하면 dummy가 이긴다. distributed candidate merge 전에 dummy를 mask하거나 global merge 뒤 V로 잘라야 한다. 이 toy request 하나로 projection, temperature, LSE, TP, vocabulary padding 계약을 동시에 점검할 수 있다.

### 실전 조사 기록은 결론보다 반증을 먼저 쓴다

사건 기록 첫 줄에는 현상을 쓴다. “model revision R에서 TP=4와 prompt logprobs on일 때 token position 317 이후 logprob가 TP=1과 달라진다.” 다음 줄에는 같은 상태를 쓴다. final IDs, positions, selected hidden row mapping, weight hash가 동일하다는 증거다. 그다음 최초 다른 상태를 쓴다.

예를 들어 chunked prefill 두 번째 chunk 첫 row에서만 returned token alignment가 한 칸 밀렸다면 head numerical 문제 가설을 버린다. raw logits row는 reference와 같지만 다른 label ID를 gather한 것이다. fix는 GEMM이나 softmax가 아니라 chunk boundary의 preceding-row mapping이다. 회귀 fixture는 chunk 크기 앞뒤에서 tokenwise logprobs를 비교한다.

다른 사건에서 quantized TP rank 2 local logits부터 차이가 시작되고 해당 shard weight scale에 NaN이 있다면 global softmax는 피해자다. startup tensor validation과 loader를 고치고, sampler의 non-finite guard는 방어선으로 남긴다. softmax에서 NaN을 0으로 치환해 응답만 살리면 corruption을 숨긴다.

OOM 사건에서 head input M이 예상 1이 아니라 prompt 전체 8,000이고 output이 FP32 `[8000,128000]`이면 약 3.8 GiB다. `top_logprobs=5`가 output을 5개만 반환해도 full intermediate가 이미 크다. hidden preselection, chunked logprob computation, distributed selected logprob 같은 설계 대안을 평가한다.

latency 사건에서 greedy인데 profiler에 vocabulary softmax와 full D2H logits copy가 보인다면 consumer를 찾는다. logprobs/entropy observability가 암묵적으로 켜졌거나 custom processor가 CPU에서 full scores를 요구할 수 있다. 해당 field를 끈 A/B에서 selected token equivalence와 transfer bytes를 확인한다. “softmax kernel이 느리다”보다 왜 호출됐는지가 먼저다.

품질 사건에서 raw logits top-1은 같지만 temperature 이후 token이 다르다는 표현은 정확하지 않을 수 있다. positive temperature는 순위를 보존한다. token이 달라졌다면 stochastic draw, 다른 processor와의 순서, RNG state가 개입했다. 수학적 invariant로 불가능한 가설을 먼저 제거하면 조사 공간이 줄어든다.

## 17.9 probability 경계를 검증하고 18장으로 넘긴다

첫 불변식은 vocabulary identity다. head row `i`, tokenizer ID `i`, sampler global ID `i`, detokenizer ID `i`가 같은 token을 뜻해야 한다. TP shard offset과 padded rows는 이 identity를 바꾸지 않는다. adapter extra vocabulary가 있으면 mapping table을 명시한다.

둘째는 row identity다. request의 어떤 prefix를 조건으로 한 hidden인지와 logits output row가 일치해야 한다. prefill, decode, chunked prefill, speculative verification, cancellation과 batch compaction에서 mapping을 보존한다. shape equality는 row identity의 증명이 아니다.

셋째는 normalization identity다. probability나 logprob를 주장할 때 denominator가 같은 후보 집합 전체를 포함해야 한다. TP local sum, top-N subset sum, mask 전후 sum을 global processed distribution과 혼동하지 않는다. greedy token만 주장할 때는 불필요한 normalization을 요구하지 않는다.

넷째는 stage identity다. raw model logits, model-defined scaling 후 logits, request processor 후 scores, normalized logprobs를 이름과 trace field로 구분한다. 모두 `[M,V]` shape일 수 있어 type만으로 구별되지 않는다. stage가 없는 “logits dump”는 비교 증거로 약하다.

다섯째는 numerical identity의 범위다. backend와 TP reduction이 달라 bitwise equality가 필요하지 않을 수 있지만 tolerance는 top margin과 task effect를 고려해 사전에 정한다. token이 같다는 이유로 큰 distribution drift를 허용하지 않고, 작은 공통 offset 때문에 실패시키지도 않는다.

여섯째는 비용 identity다. API가 반환한 N개 후보 수와 내부 projection M×V, collective, D2H bytes는 다르다. option의 비용을 response 크기만으로 설명하지 않는다. shape ledger와 profiler가 같은 stage를 가리키는지 맞춘다.

### 17.9.1 최종 acceptance lab

lab 1은 head algebra다. 작은 D와 V의 fixed hidden/weight를 두고 matrix multiplication, max-shift softmax, LSE, selected logprob를 손계산한다. implementation 결과가 각 중간값과 맞아야 한다. temperature 1과 2, 공통 offset을 교차해 순위 보존과 probability 변화를 확인한다.

lab 2는 row selection이다. 길이가 다른 두 prompt를 단독과 batch로 구성하고 각 request의 마지막 유효 hidden row index를 표시한다. full projection 뒤 gather와 hidden gather 뒤 projection이 같은 raw logits를 내야 한다. right PAD, left PAD, packed representation에서 구현별 mapping을 검증하되 동일한 dense layout을 강요하지 않는다.

lab 3은 prompt alignment다. BOS 포함 여부가 알려진 짧은 ID sequence의 tokenwise expected label을 작성한다. prompt logprob on/off, 시작 offset, chunk boundary, cached prefix를 교차한다. 반환 token ID와 logprob row, original prompt position을 triple로 비교한다. 평균값만 비교하지 않는다.

lab 4는 vocabulary parallel이다. TP=1 reference weight를 정해 TP=2/4 shard로 잘라 local logits를 재구성한다. global argmax, top candidates, LSE와 selected logprob가 tolerance 안에서 같아야 한다. visible vocabulary 밖 physical padding row에 큰 값을 주어 exclusion을 확인한다.

lab 5는 dtype와 quantization이다. 같은 hidden으로 FP32 reference, model dtype, quantized head 결과를 얻었다고 가정하고 source-level expected error budget을 정한다. centered logit error, top-k overlap, margin-conditioned flip을 계산한다. 실제 model을 실행하지 않는 검토 단계에서는 kernel formula와 stored scale로 synthetic row를 계산해 contract를 확인한다.

lab 6은 non-finite다. raw hidden NaN, 한 head row NaN, processor가 만든 all-`-inf`, positive infinity tie를 각각 별도 fixture로 둔다. 어느 guard가 어느 오류를 반환하는지 정한다. 모든 경우를 “invalid probability” 하나로 합치면 최초 원인을 잃는다.

lab 7은 비용이다. M, D, visible/physical V, dtype bytes, TP를 입력한 표에서 local logits bytes와 gathered bytes를 계산한다. prompt logprobs와 output logprobs N을 바꿔 projection, collective, serialization 중 무엇이 변하는지 예상한다. profiler 없이도 불가능한 성능 주장을 걸러낸다.

lab 8은 source trace다. request field 하나를 골라 validation, effective default merge, branch predicate, selected indices, tensor shape, collective, response field까지 file#line으로 잇는다. field가 존재하지만 실제 branch가 실행되지 않는 경우를 반증할 관측도 적는다. 이 과정을 temperature, prompt logprobs, TP, quantization에 반복한다.

acceptance는 text가 그럴듯한지로 끝내지 않는다. fixed input의 raw score path가 재현되고, probability를 요구한 consumer만 stable normalization 비용을 지불하며, distributed result가 global vocabulary 의미를 보존해야 한다. failure fixture는 명시적으로 거부되고, option별 비용 변화가 shape ledger와 일치해야 한다.

독자가 새로운 engine을 만났을 때도 같은 순서를 적용할 수 있다. final hidden owner를 찾고, output row selection과 LM head partition을 찾고, raw score stage를 고정한다. 그다음 consumer가 argmax, categorical sampling, logprob, loss 중 무엇을 요구하는지 보고 필요한 reduction만 추적한다. framework 이름이 달라도 수학적 정보 요구량은 변하지 않는다.

장애 회고에는 놓친 invariant를 기록한다. row mapping을 shape test로만 검증했는지, local logprob를 global로 오해했는지, output 관측 option이 full logits memory를 만들었는지, quant error를 margin 없이 평가했는지 적는다. 다음 배포 gate와 metric을 그 invariant에 연결해야 회고가 재발 방지가 된다.

이 lab을 통과하면 logits를 신비한 “model confidence”로 보지 않게 된다. hidden row, head weight, vocabulary partition이 만든 구체적 tensor이고, softmax는 consumer가 normalized mass를 요구할 때만 수행하는 reduction이다. 그 경계를 이해해야 18장의 sampling policy가 무엇을 바꾸었는지 정확히 말할 수 있다.

acceptance lab을 통과한 뒤에는 결과를 다시 세 질문으로 압축한다. 이 질문은 새 체크리스트를
추가하는 것이 아니라, 앞의 fixture가 stage·consumer·first divergence를 실제로 닫았는지
확인하는 회고다.

첫째, 지금 보고 있는 숫자의 stage를 한 문장으로 말할 수 있는가. “Llama block의 final norm을 지난 request R의 logical position 42 hidden을 tied vocabulary head로 projection한 TP rank 1의 raw local logits”처럼 owner와 좌표를 포함해야 한다. 단지 “scores tensor”라고 하면 비교가 시작되지 않는다.

둘째, 이 consumer가 왜 확률을 요구하는가. greedy argmax라면 순위만 필요하다. selected token logprob라면 global LSE와 그 token logit이 필요하다. top-p라면 누적 질량이 필요하고, full entropy라면 전체 분포가 필요하다. 요구 정보보다 큰 gather와 materialization을 발견하면 correctness를 보존하는 축소 경로를 검토한다.

셋째, 차이가 보였을 때 최초 divergence를 어디까지 좁혔는가. final input, hidden, local head, global collective, normalization, processor, selection을 순서대로 비교한다. 뒤 stage의 현상으로 앞 stage의 원인을 단정하지 않는다. 각 경계에 하나의 반증 관측이 있어야 한다.

이 세 질문은 코드 리뷰에도 적용된다. 새로운 option이 추가되면 field default와 validation만 보지 않고 branch, state/tensor, effect, test fixture를 요구한다. distributed path라면 local/global 의미와 collective를, quantized path라면 reference와 margin을, logprob path라면 row/label alignment를 요구한다. 문서와 구현의 책임 경계가 일치해야 한다.

운영자는 전체 logits를 늘 저장하지 않아도 이 질문에 답할 수 있도록 trace를 설계한다. artifact fingerprint, row mapping, selected top candidates와 LSE, stage별 finite/checksum, collective shape를 sampling한다. 사건 때 synthetic fixture로 상세 tensor를 복원한다. 관측 가능성과 사용자 데이터 최소 수집을 동시에 지킬 수 있다.

여기까지 닫히면 9장에서 넘어온 hidden state는 설명 가능한 vocabulary score가 된다. 다음 장은 raw score를 받아 ordered constraints와 stochastic choice를 적용한다. 그때 temperature, penalty, top-k/p가 섞여 보여도 이 장에서 고정한 raw boundary로 돌아오면 model 계산과 선택 정책을 분리할 수 있다.

결국 핵심은 숫자를 많이 저장하는 일이 아니라, 각 숫자가 어느 prefix와 vocabulary, 어느 분산 상태를 대표하는지 끝까지 잃지 않는 일이다.

세 질문에 답했다면 request option을 그 score path 위에 놓는다. 옵션 이름과 기본값을 다시
나열하는 대신 validation이 어느 branch를 선택하고 어떤 tensor와 비용을 바꾸는지 다섯 칸으로
연결한다.

`prompt_logprobs=N`을 예로 들자. field는 API validation에서 범위가 정해진다. branch는 prompt row 보존과 logprob computation을 켠다. state/tensor는 logprob start index, selected hidden `[L,D]`, local/global logits 또는 LSE와 top-N result다. 효과는 TTFT, projection/collective, response 크기 증가다. 반증은 flag off/on의 L, GEMM shape, collective bytes와 반환 alignment다.

`temperature=T`는 positive sampling branch에서 score division을 켠다. state는 request별 scalar 또는 batched temperature tensor다. 효과는 logit differences와 distribution entropy 변화이며 model/KV tensor를 바꾸지 않는다. 반증은 raw logits 동일, scaled logits의 ratio, softmax와 RNG 소비다. T=0은 별도 greedy branch인지 확인한다.

`output_logits`는 generation state에 raw step logits를 보존하는 branch를 켤 수 있다. tensor는 step당 `[B,V]` 또는 선택 row이며 sequence length만큼 reference가 쌓인다. 효과는 debugging 관측과 memory 증가다. 반증은 allocated/retained bytes, 반환 object 종류, processor 전후 checksum이다.

`logits_to_keep=K`는 model 지원 branch에서 hidden sequence slice를 줄인다. state는 selected indices나 tail slice, tensor는 `[M,D]`다. 효과는 LM-head work와 logits memory 감소지만 prompt loss/logprob 범위를 제한한다. 반증은 forward signature 지원, head input M, output equivalence와 요청 기능 성공 여부다.

`tensor_parallel_size=P`는 model weight partition과 collective group을 만든다. state는 rank vocabulary range와 padded size이며 tensor는 local `[M,Vp/P]`다. 효과는 head weight capacity 분산과 collective 추가다. 반증은 shard mapping, local logits reconstruction, TP=1 reference의 global logits/logprob다. P만 보고 gather인지 distributed selection인지 단정하지 않는다.

quantization field는 loader/linear-method branch를 선택하고 packed weight, scales, zero points와 workspace를 만든다. 효과는 weight bytes와 kernel, accumulator/rounding, logits 오차다. 반증은 actual module/kernel, tensor metadata, reference margin-conditioned differential이다. config에 이름이 찍힌 것과 LM head에 실제 적용된 것은 다를 수 있다.

### 17.9.2 운영 검증은 raw logits 전체를 수집하지 않는다

vocabulary 전체 logits는 크고 민감하다. 평상시 metric에는 finite 여부, centered norm, top-1/top-2 margin bucket, entropy 또는 selected logprob의 bounded histogram, selected ID 유효 범위, prompt-logprob row count를 남긴다. model/template revision을 bounded label로 사용하고 request별 후보는 label에 넣지 않는다.

canary cohort에서는 first-step top-N ID/logit과 centered digest를 secure trace로 남길 수 있다. 전체 V를 저장하지 않아도 backend drift와 selected boundary를 잡는다. top-N 밖 변화가 중요한 logsumexp에는 LSE scalar도 남긴다. `(top-N, LSE, selected ID, raw/processed stage)`가 compact한 증거 묶음이다.

경보는 증상 조합을 사용한다. non-finite raw logits는 즉시 높은 심각도다. invalid selected ID, TP rank별 finite 불일치, empty allowed distribution도 hard failure다. top-1 margin 분포의 완만한 변화는 model update의 정상 변화일 수 있어 reference canary와 결합한다. prompt-logprob on cohort의 TTFT/oom 상승은 row count로 normalize한다.

사용자별 prompt나 token 후보를 그대로 저장하면 개인정보와 model extraction 위험이 있다. ID trace는 revision-specific mapping과 함께 암호화·접근 통제하고 짧게 보존한다. synthetic fixtures는 상세 logits를 장기간 보존해 회귀 기준으로 사용한다. production observability와 laboratory evidence의 보존 정책을 분리한다.

배포 gate는 세 층이다. artifact gate는 head/embedding shape와 shard mapping을 검사한다. numerical gate는 fixed fixtures의 raw logits, LSE, selected token과 margin-conditioned tolerance를 검사한다. operational gate는 target batch/TP/logprob 설정의 memory, collective bytes, TTFT를 검사한다. 한 층의 성공이 다른 층을 대신하지 않는다.

### 17.9.3 장애 워크북

첫 단계는 final input identity다. token IDs, mask, logical positions, selected sequence row가 같지 않으면 logits 비교를 중단하고 8·9장으로 돌아간다. 둘째는 final hidden checksum과 dtype이다. 여기서 갈리면 attention/MLP/cache/backend를 본다. 셋째는 LM head module, weight hash, tied storage, shard range와 quant metadata다.

넷째는 raw local logits를 비교한다. 특정 TP rank만 다르면 weight shard나 collective 이전 kernel로 좁힌다. local은 같은데 gathered global이 다르면 rank ordering, padded vocabulary trim, collective buffer를 본다. raw global은 같은데 logprobs가 다르면 temperature, stable normalization, processor order, dtype를 본다.

다섯째는 필요한 확률의 범위를 묻는다. greedy token만 필요하면 softmax 호출 자체가 불필요한지 확인한다. selected logprob만 필요하면 global LSE와 selected logit으로 충분한지 본다. top-N 반환이면 candidate merge가 정확한지, full vocabulary audit이면 memory 예산을 명시한다. 요구사항보다 큰 tensor를 만드는 것이 병목일 수 있다.

여섯째는 반증 가능한 결론을 쓴다. “양자화라 조금 다름” 대신 “final hidden은 동일하고 rank 1의 quantized LM-head projection에서 최대 0.012 logit 오차가 시작되며 top-2 margin 0.004인 3.1% row에서 argmax가 뒤집힌다”고 쓴다. “logprobs가 느림” 대신 “prompt logprobs가 retained head rows를 64에서 1,920으로 늘리고 TP gather bytes가 30배가 된다”고 쓴다.

### 17.9.4 소스 노트

Transformers 기준 revision은 `550d7b3834670483a4df436541272c055dc364bf`다. model weight tying과 resize, Llama causal head, generation loop, temperature warper와 transition scores를 같은 revision에서 연결했다. model class의 세부 line은 architecture와 release에 따라 이동할 수 있으므로 pinned link의 호출 흐름을 기준으로 읽는다.

vLLM 기준 revision은 `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`다. logits processor의 hidden selection, LM head, tensor-parallel gather, prompt/output logprob 경계를 사용했다. SGLang 기준 revision은 `71de97b264b04dcd514cf904003028aefe9775c8`이며 logits metadata와 forward-mode별 row selection을 근거로 삼았다.

llama.cpp 기준 revision은 `bb4caa7540188872173c44d161602d9271386413`다. batch의 token별 output flag, logits buffer accessor, sampler-side softmax/temperature 경계를 연결했다. attention 내부 softmax와 vocabulary softmax는 목적과 축이 다르므로 이 장에서 혼동하지 않았다.

이 장의 산출물은 probability tensor가 아니라 설명 가능한 score path다. 어떤 hidden row가 어떤 head shard를 지나 raw logits가 되었고, 어떤 요청 때문에 global normalization 또는 logprob가 필요했는지 말할 수 있어야 한다. 18장은 이 점수에 ordered policy를 적용하고 후보와 RNG로 실제 token을 고르는 과정을 이어 간다.

## 17.10 미리 보는 sampling 입력: 다섯 logits의 변환을 손으로 계산한다

이 절은 sampling 알고리즘을 소유하지 않는다. 17장의 확률 경계를 이해하려고 logits 변환이 normalization 입력을 어떻게 바꾸는지만 한 번 손으로 펼친다. Processor 순서, constraint, 후보 집합, RNG와 visible commit의 구현 판정은 18장에서 같은 원장을 이어받아 닫는다.

옵션 설명이 불친절해지는 가장 흔한 이유는 “반복을 줄인다”, “다양성을 높인다” 같은 효과만 말하고 실제 배열이 어떻게 바뀌는지 보여 주지 않기 때문이다. 여기서는 vocabulary가 다섯 개인 한 row를 끝까지 계산한다. token은 편의상 A=0, B=1, C=2, D=3, E=4라 하자. LM head가 만든 raw logits는 `[2.0, 1.5, 0.4, -0.5, -1.0]`이다. prompt에는 A가 두 번, B가 한 번 있었고, 현재 생성 output에는 A가 한 번, C가 두 번 있었다고 하자.

먼저 중요한 질문은 penalty가 prompt와 output 중 무엇을 세는가다. API와 엔진에 따라 repetition penalty가 prompt와 generated history를 모두 보거나 특정 집합을 보고, frequency/presence가 output token count를 중심으로 적용될 수 있다. 수식 이름만 같아도 count source가 다르면 값이 다르다. fixture에는 `prompt_counts=[2,1,0,0,0]`, `output_counts=[1,0,2,0,0]`, 그리고 각 processor가 읽는 합성 규칙을 명시한다. 여기서는 예시를 위해 합친 count `[3,1,2,0,0]`을 세 penalty 모두에 쓴다. 실제 구현 검증에서는 source contract에 맞춰 분리한다.

request의 logit bias가 C에 +1.2, D에 -2.0이라고 하자. bias 뒤 배열은 `[2.0, 1.5, 1.6, -2.5, -1.0]`이다. bias는 모델 지식을 다시 계산하지 않는다. global token ID가 가리키는 특정 score에 요청별 상수를 더한다. C는 raw 순위 3위였지만 bias 뒤에는 B를 넘어 2위가 된다. TP local tensor에서 적용한다면 global C를 owner rank의 local index로 번역해야 한다.

SGLang의 `SamplingBatchInfo`는 request별 bias를 `[batch,vocab]` tensor에 만들고 logits에 in-place add한다. [SGLang logit bias tensor 생성과 적용](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/sampling/sampling_batch_info.py#L84-L140)

repetition penalty를 1.25로 두자. 흔히 쓰이는 sign-aware scaling은 이미 본 token의 logit이 양수이면 1.25로 나누고 음수이면 1.25를 곱한다. 그러면 A는 `2.0/1.25=1.6`, B는 `1.5/1.25=1.2`, C는 `1.6/1.25=1.28`이 된다. D와 E는 보지 않은 token이므로 그대로다. 음수 값을 단순히 나누면 -2.5가 -2.0이 되어 오히려 확률이 커진다. sign-aware 규칙이 필요한 이유는 “반복 token을 불리하게 한다”는 방향을 양수와 음수 양쪽에서 보존하기 위해서다.

이 penalty는 횟수 3과 1을 구분하지 않는다. 한 번이라도 등장했다는 집합 membership을 기준으로 같은 비율을 적용하는 형태다. 따라서 “repetition penalty를 높이면 많이 반복한 token을 더 크게 누른다”는 설명은 frequency penalty와 혼동한 것이다. 값이 1이면 identity이고, 1보다 크면 반복 후보를 불리하게 하는 일반적 설정이다. 1보다 작은 값을 허용하는 API에서는 반대로 반복을 장려할 수 있다. validation 범위와 구현 수식을 함께 읽어야 한다.

frequency penalty를 0.3으로 두고 count에 비례해 빼자. 현재 배열 `[1.6,1.2,1.28,-2.5,-1.0]`에서 `[0.9,0.3,0.6,0,0]`을 빼면 `[0.7,0.9,0.68,-2.5,-1.0]`이다. A는 세 번 등장했기 때문에 가장 많이 내려간다. C는 두 번, B는 한 번 내려간다. coefficient가 음수라면 count가 많은 token을 오히려 올리는 효과가 된다. SGLang의 sampling parameter validation은 frequency와 presence를 각각 `[-2,2]`, repetition을 `(0,2]`로 제한하고 기본값을 0, 0, 1로 정규화한다.

옵션 이름만 적지 말고 identity 값과 허용된 반대 방향까지 설명해야 한다. [SGLang penalty 기본값과 validation](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/sampling/sampling_params.py#L107-L176)

presence penalty를 0.4로 두자. count가 0보다 큰 A, B, C에서 각각 0.4를 한 번만 뺀다. 배열은 `[0.3,0.5,0.28,-2.5,-1.0]`이 된다. 이 단계에서는 B가 최고다. presence penalty는 세 번 나온 A와 한 번 나온 B를 같은 양만큼 내린다. “등장 여부”와 “등장 횟수”가 분리되는 지점이다. 둘을 같이 켜면 본 token은 최소 presence만큼 내려가고, 횟수마다 frequency만큼 더 내려간다.

마지막으로 temperature 0.5를 적용하면 logits를 0.5로 나누어 `[0.6,1.0,0.56,-5.0,-2.0]`으로 만든다. 순위는 바뀌지 않지만 차이는 두 배가 된다. max 1.0을 빼고 exp를 계산하면 대략 `[0.6703,1.0,0.6440,0.00248,0.0498]`, 합은 약 2.3666이다. 확률은 `[0.2833,0.4226,0.2721,0.0010,0.0210]` 정도다. temperature 2.0이었다면 차이가 절반으로 줄어 더 평평해진다. temperature는 “무작위 양”을 직접 더하는 값이 아니라 score 차이의 단위를 바꾸고, 그 뒤 categorical draw가 달라질 확률 분포를 만든다.

이 예제의 단계 원장은 다음처럼 남긴다.

| 단계 | A | B | C | D | E | winner |
|---|---:|---:|---:|---:|---:|---|
| raw | 2.00 | 1.50 | 0.40 | -0.50 | -1.00 | A |
| bias | 2.00 | 1.50 | 1.60 | -2.50 | -1.00 | A |
| repetition 1.25 | 1.60 | 1.20 | 1.28 | -2.50 | -1.00 | A |
| frequency 0.3 | 0.70 | 0.90 | 0.68 | -2.50 | -1.00 | B |
| presence 0.4 | 0.30 | 0.50 | 0.28 | -2.50 | -1.00 | B |
| temperature 0.5 | 0.60 | 1.00 | 0.56 | -5.00 | -2.00 | B |

이 표에서 raw와 processed를 한 필드 `logits`로 덮어쓰면 first divergence를 잃는다. 구현은 성능 때문에 in-place update를 해도 관측 언어는 `raw_head`, `after_bias`, `after_repetition`, `after_frequency`, `after_presence`, `temperature_scaled`, `normalized`를 구분해야 한다. 모든 stage의 full tensor를 production에서 저장하라는 뜻은 아니다. 작은 fixture에서는 전체 값을 보존하고, 실제 요청에서는 고정 token slice와 top candidate, checksum을 stage별로 남긴다.

### 순서를 바꾸면 왜 다른 답이 되는가

additive penalty끼리는 같은 count를 사용한다면 더하는 순서를 바꿔도 수학적으로 같을 수 있다. 하지만 repetition은 sign-aware multiplication/division이고 bias는 부호를 바꿀 수 있으므로 교환 법칙이 성립하지 않는다. raw C=0.4에 bias -1.0을 먼저 적용하면 -0.6이 되고 repetition 1.25는 -0.75를 만든다. repetition을 먼저 적용하면 0.32이고 그 뒤 bias를 더해 -0.68이다. 0.07 차이가 생긴다. decision margin이 작으면 winner가 바뀐다.

temperature도 additive penalty와 교환되지 않는다. penalty를 뺀 뒤 전체를 T로 나누면 penalty 자체도 `1/T`만큼 확대된다. temperature를 먼저 적용하고 같은 penalty를 빼면 penalty의 상대 강도가 달라진다. 예를 들어 z=1.0, frequency deduction=0.4, T=0.5라면 `(1.0-0.4)/0.5=1.2`지만 `1.0/0.5-0.4=1.6`이다. API가 약속하는 순서를 알아야 option의 의미가 고정된다.

vLLM 고정 sampler의 docstring은 min-token과 logit bias processor 뒤 penalties를 적용하고, sampling 안에서 temperature, argmax-invariant processor, top-k/p 순으로 진행한다고 적는다. forward는 logprob mode에 따라 processor 전 raw logprobs를 먼저 계산할 수 있고, 이후 logits processor와 penalty를 적용해 sampling으로 넘긴다.

`apply_temperature`는 in-place division이며 greedy request가 섞였을 때 0에 가까운 temperature를 1로 대체한 뒤, 마지막 token 선택에서 greedy 결과를 고르는 구조다. [vLLM sampler 단계와 raw logprob 경계](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L35-L115), [vLLM temperature와 sample 분기](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L228-L306)

여기서 중요한 친절한 설명은 temperature 0이 실제로 0으로 나누어지는 것이 아니라는 점이다. mixed batch에서는 greedy 후보를 scaling 전 logits에서 구하고, random path 계산을 위해 작은 temperature 자리에 안전한 값이 들어갈 수 있다. 마지막 `where`가 request별로 greedy 또는 random 결과를 선택한다. 따라서 profiler에서 greedy request가 포함된 batch에 temperature kernel이 보인다고 “greedy가 temperature sampling됐다”고 단정하지 않는다. 값의 소비 지점을 끝까지 본다.

### count state는 request lifecycle의 일부다

penalty 계산은 logits tensor만의 순수 함수가 아니다. prompt token과 이미 commit된 output history라는 mutable state를 읽는다. speculative draft token을 frequency count에 넣었다가 reject 후 빼지 않으면 존재하지 않는 역사가 다음 distribution을 바꾼다. 취소된 request의 physical slot을 새 request가 재사용할 때 count buffer가 초기화되지 않아도 정상 범위의 이상한 답이 나온다. KV cache뿐 아니라 penalty counter도 request incarnation에 묶여야 한다.

batch compaction에서는 row permutation과 count row permutation이 같아야 한다. request A와 B의 logits는 올바르게 compact됐는데 frequency count tensor만 이전 순서라면 A가 B의 반복 이력을 받는다. 같은 prompt를 단독으로 실행하면 문제가 사라지고 mixed concurrency에서만 나타난다. fixture에는 서로 구분되는 history를 넣는다. A는 token 0을 열 번, B는 token 4를 열 번 본 것으로 만들고 batch 순서를 바꿔 processed score가 request identity를 따라가는지 본다.

문자열 repetition과 token repetition도 구분한다. 사용자가 같은 단어가 반복된다고 느껴도 tokenizer가 앞 공백, 대소문자, byte fallback으로 다른 token ID를 만들면 token-level penalty는 각각 다른 count를 본다. 반대로 같은 token이 다른 문맥에서 나타나도 동일 ID면 count된다. 옵션 효과를 설명할 때 자연어 의미의 중복 제거처럼 과장하지 않는다.

### 옵션 카드가 답해야 하는 여섯 질문

각 옵션 문서는 먼저 identity 값을 말한다. logit bias는 0, presence와 frequency는 0, repetition은 1, temperature는 보통 1이 변환을 하지 않는 값이다. 둘째 validation 범위와 T=0 같은 special branch를 말한다. 셋째 count source 또는 global token 좌표를 말한다. 넷째 processor order와 in-place 여부를 말한다. 다섯째 tensor·state·비용 변화를 말한다. 여섯째 반증 fixture를 말한다.

예를 들어 presence penalty를 0에서 0.5로 바꾸면 model forward와 KV는 그대로다. request history에서 등장 token의 boolean mask가 만들어지고 해당 score에서 0.5가 빠진다. candidate 순위와 entropy가 변할 수 있으며 count/mask update 비용이 생긴다. raw logits가 달라졌다면 이 옵션의 직접 효과가 아니다. processed logits는 달라져야 하고, 보지 않은 token은 동일해야 한다.

frequency penalty를 높이면 count가 큰 token이 선형으로 더 내려간다. state는 per-request token frequency다. prompt 포함 여부와 committed output 반영 시점을 확인한다. 매우 긴 history에서 dense `[B,V]` count를 유지하는지 sparse update를 쓰는지에 따라 memory와 kernel이 다르다. 성능 설명은 coefficient 숫자가 아니라 실제 representation을 근거로 한다.

repetition penalty를 바꾸면 seen mask와 score 부호에 따라 multiply 또는 divide가 적용된다. 값 1에서 fast-path가 생기는지, prompt와 output을 함께 보는지, generated-only인지 확인한다. quantized model이라도 penalty는 보통 logits가 FP32로 승격된 sampler 경계에서 적용될 수 있으므로 model weight dtype과 penalty arithmetic dtype을 혼동하지 않는다.

temperature를 낮추면 distribution이 날카로워지지만 greedy와 동일하다고 단정할 수 없다. 양수인 한 tail token의 확률은 남고 RNG가 소비된다. T가 epsilon 아래일 때만 별도 greedy path가 선택될 수 있으며 threshold는 구현 계약이다. 같은 seed 비교에서는 threshold 양쪽, mixed greedy/random batch, top-p와 결합을 검사한다.

이 절을 읽은 뒤 독자는 “penalty가 반복을 줄입니다”에서 멈추지 않는다. 어떤 token count가 어느 score에서 얼마나 빠지고, bias와 sign-aware scaling과 temperature 사이의 순서가 무엇이며, 그 state가 scheduler compaction과 speculative rollback에서 누구에게 속하는지를 질문할 수 있다.

## 17.11 exact logprob의 분산 원장과 값은 맞지만 사용자에게 틀리게 돌아간 사건

logprob는 단순히 softmax에 log를 취한 장식이 아니다. 특정 token score와 같은 row 전체의 normalization을 연결한 값이다. TP에서 vocabulary가 나뉘면 선택 token은 한 rank가 소유하지만 denominator는 모든 rank가 소유한다. 따라서 local `log_softmax` 결과를 global logprob로 반환하면 값이 체계적으로 너무 커진다. local shard 안에서만 확률 질량을 1로 만들었기 때문이다.

두 rank의 작은 예를 계산하자. rank 0은 global token 0,1을 소유하고 logits `[4,3]`, rank 1은 token 2,3을 소유하고 `[2,1]`을 가진다. rank 0 local logsumexp는 `4+log(1+e^-1)≈4.3133`이다. token 0 local logprob는 약 -0.3133이다. 하지만 global logsumexp는 `4+log(1+e^-1+e^-2+e^-3)≈4.4402`이고 exact global logprob는 약 -0.4402다. local 값은 token 2와 3의 질량을 무시해 0.1269만큼 과대평가한다.

분산 stable normalization은 두 단계로 쓸 수 있다. 각 rank local maximum `m_r`을 구하고 all-reduce max로 `m=max_r m_r`을 얻는다. 각 rank는 `s_r=sum_j exp(z_rj-m)`을 계산하고 all-reduce sum으로 `s=sum_r s_r`을 얻는다. global LSE는 `m+log(s)`다. 선택 token t의 owner가 `z_t`를 제공하면 `log p_t=z_t-LSE`다. top-k token 값도 같은 LSE를 빼면 된다.

이때 FP32 accumulator가 중요한 이유를 수치로 본다. logits가 `[1000,999]`이면 직접 exp는 overflow할 수 있지만 max shift 뒤 `[0,-1]`은 안전하다. 여러 rank에서 local max만 빼고 local sum을 합치면 기준이 달라져 틀린다. 모든 rank가 같은 global max를 기준으로 exp sum을 계산하거나, local `(max,sum)` pair를 결합하는 수학적으로 올바른 reduction이 필요하다.

SGLang의 고정 logprob processor는 per-row `(max, logsumexp-max)`를 FP32로 계산하는 helper를 두고, 일반 fallback에서는 `torch.logsumexp(x-row_max)`를 사용한다. top logprob용 chunk 경로는 normalizer와 top-k를 함께 다루며, 다른 경로에서는 raw logits를 해제한 뒤 out-of-place `log_softmax`를 수행해 peak memory를 관리한다.

이는 logprob 옵션이 단순 response formatting이 아니라 normalization kernel, chunking, tensor lifetime을 바꾼다는 소스 증거다. [SGLang FP32 row normalizer와 top-k](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/logprob_processor.py#L69-L128), [SGLang chunked logprob tensor lifetime](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/logprob_processor.py#L529-L632)

### raw logprob와 processed logprob는 서로 다른 질문에 답한다

raw logprob는 model head score를 normalization한 값이다. processed logprob는 bias, penalty, temperature, truncation 같은 sampling policy가 적용된 distribution의 값일 수 있다. 둘 다 유용하지만 의미가 다르다. model scoring이나 perplexity에 sampling temperature를 섞으면 안 되고, 사용자가 실제로 어떤 policy로 token이 선택됐는지를 설명하려면 processed distribution이 필요할 수 있다.

vLLM 고정 sampler는 `logprobs_mode`에 따라 processor 전 raw logits에서 `raw_logprobs`를 계산하거나 raw logits 자체를 복사할 수 있다. sampling 뒤 processed logprobs가 있으면 반환 후보를 그것으로 바꾸는 경로도 있다. 코드 주석은 V1 raw top-k logprobs가 penalty와 temperature 이전이라는 점과 이전 sampler의 차이를 명시한다. release를 올렸을 때 숫자가 달라졌다면 model regression 전에 mode와 stage 계약이 바뀌었는지 본다. [vLLM raw·processed logprob 선택](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L70-L105)

temperature를 적용한 logprob도 분리한다. raw z의 logprob와 z/T의 logprob는 T가 1이 아니면 다르다. SGLang speculative logprob 경로에는 gathered logits를 그대로 log-softmax하거나 request별 temperature로 나눈 뒤 log-softmax하는 분기가 있다. speculative acceptance가 요구하는 target/draft 확률과 API가 반환하는 raw scoring 값이 같은 stage인지 확인해야 한다. [SGLang gathered logits와 temperature log-softmax](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/logprob_processor.py#L376-L408)

### 통신과 memory 비용을 세 반환 계약으로 나눈다

V=152,064, row M=64, FP32를 다시 사용하자. full global logits는 약 37.1 MiB다. 모든 rank에 all-gather해 log-softmax하면 구현은 단순하지만 TP=4 각 rank가 full tensor와 logprob output, temporary를 가질 수 있다. out-of-place log-softmax가 raw와 output을 동시에 보유하면 tensor 두 벌만으로 약 74.2 MiB다. allocator workspace와 top-k output은 별도다.

선택 token logprob 하나만 반환한다면 full tensor는 필요 없다. global max와 exp sum reduction, selected token score 전달이면 row당 몇 개 scalar로 줄일 수 있다. M=64, max와 sum을 FP32 두 scalar로 표현하면 논리 payload는 512 byte 수준이다. 실제 collective는 alignment와 algorithm overhead가 있으며 head local logits 계산은 여전히 필요하다. “통신 512 byte”를 end-to-end byte라고 쓰지 않고 reduction operand의 정보량이라고 쓴다.

top-N logprobs N=5라면 rank별 local top-5 score와 global ID를 후보로 모으고 global LSE를 적용할 수 있다. TP=4에서 row당 최대 20 pair다. score FP32 4 byte, ID int32 4 byte로 단순 계산하면 160 byte/row, 64 row에서 10 KiB 후보다. reduction overhead와 result broadcast를 제외한 비교치지만 37.1 MiB full tensor와 차이가 크다. 다만 arbitrary logit processor가 global array를 요구하면 이 경로를 그대로 쓸 수 없다.

prompt logprobs M=8192에서는 row 수가 비용을 지배한다. selected token logprob만 필요하면 각 row의 target ID score와 LSE만 보존할 수 있다. top-N까지 반환하면 `M×N` 후보와 serialization이 생긴다. API의 `top_logprobs=5`가 GPU tensor뿐 아니라 CPU copy, token text decode, JSON/SSE byte를 늘린다. 성능 회귀를 sampler kernel 하나로 좁히지 않는다.

### 사건: 선택 token은 맞지만 반환 logprob가 다른 request의 값이다

continuous batch에 A와 B 두 요청이 있다. A는 top logprobs 5를 요구하고 B는 logprobs를 요구하지 않는다. 다음 step에서 B가 먼저 완료되어 batch가 compact되고, A의 row가 1에서 0으로 이동한다. GPU sampler는 새 row 0에서 올바른 token 42와 logprob -0.7을 계산했다. 그러나 output metadata의 inverse mapping은 이전 row 1을 사용해 다른 buffer의 -3.2를 A에게 붙였다. token stream은 정확하고 logprob만 틀렸다.

이 사건은 model quality 테스트로 잡기 어렵다. text는 reference와 같고 top token ID도 같다. logprob 범위도 정상이라 validation을 통과한다. streaming에서는 token delta와 logprob metadata가 서로 다른 비동기 path에서 합쳐져 특정 concurrency에서만 보일 수 있다. 사용자는 confidence threshold나 downstream reranking에 잘못된 값을 사용한다.

first divergence를 찾으려면 tuple을 보존한다. `(request_incarnation, generation_step, logical_row, sampled_global_id, selected_logit, global_lse, returned_logprob)`다. GPU 경계에서 `selected_logit-LSE=-0.7`인데 formatter 직전 request A record가 -3.2라면 normalization 수학이나 model을 조사하지 않는다. device-to-host row mapping, batch compaction inverse permutation, streaming cursor를 본다.

비슷한 증상의 competing hypothesis도 둔다. local LSE를 global로 오해하면 TP 크기에 따라 모든 logprob가 체계적으로 이동한다. temperature stage를 잘못 반환하면 T가 1이 아닌 cohort에서만 다르다. token/logprob shift가 한 칸이면 prompt boundary나 generation step 정렬 문제다. row permutation이면 mixed batch에서 값이 서로 교환된다. 네 fixture의 signature가 다르므로 한꺼번에 “logprob bug”로 부르지 않는다.

수정은 logprob record에 request incarnation과 generation step을 붙여 token commit record와 함께 이동시키는 것이다. physical batch row는 임시 실행 좌표이지 response identity가 아니다. cancellation과 compaction 뒤에는 inverse mapping을 generation별로 검증한다. 늦게 도착한 이전 generation의 D2H copy가 재사용된 slot에 쓰이지 않도록 completion과 lifetime을 닫는다.

회귀 테스트는 A와 B에 구분하기 쉬운 synthetic logits를 준다. A의 selected logprob는 -0.1 부근, B는 -5 부근이 되게 하고 logprob on/off, top-N 크기, batch order, completion order를 바꾼다. streaming chunk마다 token ID와 logprob가 같은 `(request,step)`을 공유하는지 검사한다. TP=1과 TP=4, raw와 processed mode, temperature 1과 0.5를 조합하되 모든 Cartesian product를 무작정 늘리지 않고 각 가설을 분리하는 최소 pair를 만든다.

### 반환 정확성의 conservation check

한 row의 normalized probability 합은 허용 오차 안에서 1이어야 하지만 top-N 반환값의 exp 합은 1보다 작아도 정상이다. top-N은 tail mass를 생략했기 때문이다. 대신 `LSE`, top 후보 raw score, 반환 logprob 사이에 `logprob=score-LSE`가 성립하는지 검사한다. processed mode라면 score도 같은 processed stage여야 한다.

sampled token이 top-N 밖일 때 API가 sampled token을 별도 항목으로 추가하는지 확인한다. vLLM sampler docstring은 sampled token이 top max logprobs 안에 있으면 output 처리에서 병합되어 최종 개수가 N 또는 N+1일 수 있다고 설명한다. 소비자가 항상 정확히 N개라고 가정하면 정상 응답을 오류로 볼 수 있다. 값뿐 아니라 list cardinality와 dedup 계약을 문서화한다. [vLLM top logprobs와 sampled token 병합 계약](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/sample/sampler.py#L35-L66)

prompt logprob는 causal shift를 conservation check에 넣는다. input token j의 logprob는 보통 이전 position hidden이 예측한 score에서 온다. BOS, multimodal placeholder, chunk boundary, cached prefix 시작점에는 반환 불가능하거나 별도 sentinel이 있을 수 있다. row 개수와 token 개수만 같다고 alignment가 맞는 것이 아니다. `(predicted_token_id, predicting_position, logical_prompt_offset)`을 fixture에 둔다.

### 모니터링과 rollout terminal

상시 metric에는 logprob request 비율, prompt/output row 수, top-N, raw/processed mode, normalization kernel/collective 구간, returned item byte를 둔다. logprob 값을 request별 label로 내보내지 않는다. bounded histogram은 selected logprob와 top-1 margin 정도로 제한하고 model/template revision cohort와 연결한다. non-finite, selected ID 불일치, row count mismatch는 counter와 sampled trace를 남긴다.

distributed normalization에는 local max finite 여부, global max, sum-exp finite 여부를 debug cohort에서 기록한다. rank 하나가 padding만 보거나 mask 결과가 all `-inf`이면 global reduction 전후 어느 단계에서 empty distribution이 생겼는지 구분한다. collective latency는 TP group size와 M으로 normalize하고 prompt-logprob large-row cohort를 별도로 본다.

rollout은 세 terminal을 모두 요구한다. correctness terminal은 small-vector processor order, TP global LSE, request-row 반환 mapping과 causal shift fixture가 통과한다. performance terminal은 target workload의 projection rows, collective operand, peak temporary, D2H/serialization byte가 원장과 맞고 SLO를 충족한다. lifecycle terminal은 cancellation, speculative reject, batch compaction, slot reuse에서 count와 logprob record가 request incarnation을 잃지 않는다.

하나라도 실패하면 raw full-gather reference나 기존 formatter 경로로 rollback한다. 최적화 path의 tensor만 되돌리고 API stage 의미를 조용히 바꾸지 않는다. 실패 trace는 다음 수정의 regression fixture로 보존한다. 사용자에게 반환된 text가 같다는 사실은 logprob correctness terminal을 대신하지 않는다.

이 절의 최종 invariant는 다음과 같다. 동일한 raw global logits와 동일한 request history에서 ordered processor가 정의된 score를 만들고, 요청이 지정한 raw 또는 processed stage의 global normalization이 정확한 `(request incarnation, generation step, token ID)`에 붙어 반환되어야 한다. 이 문장을 배열 변화, 분산 reduction, output mapping의 세 증거로 설명할 수 있어야 18장의 실제 token draw로 넘어갈 준비가 끝난다.

## 17.12 증상에서 첫 함수까지 가는 실전 디깅 지도

독자가 현장에서 받는 보고는 대개 “temperature가 이상하다”, “반복이 안 줄었다”, “logprobs가 다른 서비스와 다르다”처럼 넓다. 곧바로 sampler 파일 전체를 읽으면 빠른 path, fallback, API formatter가 뒤섞여 방향을 잃는다. 먼저 같은 입력과 artifact에서 raw head score가 같은지 확인한다. raw가 다르면 이 장의 processor보다 앞선 hidden, head weight, shard reconstruction 문제다. raw가 같을 때만 history와 ordered transformation으로 내려간다.

첫 fixture는 processor를 전부 identity로 둔다. bias 0, presence 0, frequency 0, repetition 1, temperature 1, top-k/p disabled 상태에서 processed score와 raw score가 같아야 한다. 이 테스트가 실패하면 숨은 default, forced token, min-token EOS mask, grammar 같은 다른 processor가 있다. request payload만 보지 말고 validation과 default merge 뒤 effective sampling state를 덤프한다.

둘째 fixture는 옵션을 하나씩 켠다. bias는 한 global token만 +1로 바꾸고 나머지가 bitwise 또는 허용 오차 안에서 같은지 본다. presence는 등장 횟수가 다른 두 token을 같은 양만큼 내린다. frequency는 1회와 3회 token의 감소량 비가 1:3인지 본다. repetition은 양수와 음수 token을 각각 넣어 sign-aware branch를 본다. temperature는 모든 pairwise difference가 `1/T`배인지 본다. 이렇게 해야 이름이 비슷한 penalty의 wiring swap을 잡는다.

셋째 fixture는 조합 순서를 겨냥한다. bias가 score 부호를 넘도록 값을 정하고 repetition과 함께 켠다. 앞의 C 예처럼 순서에 따라 -0.75와 -0.68이 갈리는 입력을 사용한다. frequency와 temperature도 함께 켜 `(z-fc)/T`인지 `z/T-fc`인지 확인한다. winner만 검사하지 않고 고정 token score를 비교한다. 두 결과의 winner가 우연히 같아도 수식은 틀릴 수 있다.

넷째 fixture는 lifecycle이다. request A와 B에 서로 반대인 token history를 주고 batch order를 바꾼다. 한 request를 취소하고 같은 physical slot에 C를 넣는다. speculative draft를 만들었다가 전부 reject한다. 세 경우에서 count state가 request incarnation과 committed output만 따라가는지 확인한다. processor 값 문제가 concurrency에 의존하면 수식보다 state owner부터 의심한다.

다섯째 fixture는 TP다. 같은 global raw logits를 TP=1 reference와 TP=2 shard로 구성한다. 각 rank local log-softmax, 올바른 global LSE, full gather log-softmax를 모두 계산한다. 반환 값이 local 결과와 일치하면 normalization scope bug다. global 값은 맞지만 token ID가 다르면 shard offset이다. GPU record는 맞지만 API만 다르면 row/step mapping이다.

### symptom signature로 competing hypothesis를 정렬한다

모든 request에서 logprob가 비슷한 양만큼 덜 음수이고 TP를 키울수록 차이가 커지면 local normalization이 강한 가설이다. temperature가 1일 때는 같고 0.5에서만 다르면 raw/processed stage 또는 temperature order를 본다. token은 같지만 logprob가 다른 request 값과 정확히 교환되면 row permutation을 본다. 첫 generated token부터 한 칸씩 밀리면 causal alignment와 streaming cursor를 본다.

반복 penalty가 단독 요청에서는 맞고 continuous batch에서만 틀리면 count row compaction을 본다. 오래 실행한 뒤 새 요청만 이상하면 slot initialization과 late write를 본다. speculative mode에서만 반복이 과도하면 rejected draft가 count에 들어갔는지 본다. 특정 tokenizer에서만 사용자가 반복을 느끼면 token ID count와 문자열 단위 기대의 차이를 본다.

temperature를 낮췄는데 더 다양해 보인다는 보고는 바로 수식 bug로 결론 내리지 않는다. seed와 request batching이 달라졌거나 top-p candidate cutoff와 결합되었을 수 있다. raw/processed logits, eligible set, uniform draw를 같은 request-step에서 고정한다. temperature scaling 뒤 entropy가 예상 방향인지와 실제 sample sequence를 분리한다. 작은 표본의 문체 인상은 distribution 증거가 아니다.

logit bias가 무시됐다는 보고는 tokenization부터 확인한다. 사용자가 금지하려는 문자열이 한 token인지 여러 token인지, 앞 공백 variant가 다른 ID인지 본다. 지정 global ID의 after-bias score가 변했다면 구현은 요청을 적용했다. 문자열을 완전히 금지하려면 grammar나 token sequence constraint가 필요할 수 있다. score transform의 능력을 정책 언어보다 과장하지 않는다.

### stage-aware trace의 최소 스키마

trace 한 행에는 request incarnation, generation step, physical batch row, selected hidden row, global token ID와 owner rank가 들어간다. score 쪽에는 raw 고정 slice, after-processor slice, top candidates, normalization mode와 LSE를 둔다. history 쪽에는 prompt/output count digest, committed length, speculative generation을 둔다. 반환 쪽에는 sampled ID, returned token logprob, top-N count, stream cursor를 둔다.

모든 값을 항상 남기지 않는다. 정상 traffic은 shape, timing, finite flag, bounded digest만 기록한다. canary와 incident fixture에서만 고정 token slice와 candidate를 안전하게 보존한다. prompt text와 전체 vocabulary 배열을 metric label에 넣지 않는다. trace schema가 stage와 좌표를 보존하면 상세 값이 적어도 어느 함수 사이가 갈렸는지 알 수 있다.

processor별 latency metric은 켜진 request 수와 touched token 수로 해석한다. dense bias tensor `[B,V]`를 만드는 구현은 bias entry가 한 개여도 vocabulary 크기 비용이 생길 수 있다. sparse kernel은 entry 수에 비례할 수 있다. penalty는 history representation과 unique token 수에 따라 다르다. coefficient 크기가 커진다고 계산량이 선형 증가하는 것은 아니다. 값의 효과와 실행 비용을 구분한다.

normalization metric은 row 수, vocabulary 크기, dtype, collective scope를 같이 본다. `logsoftmax_ms` 상승이 top-N 증가 때문인지 prompt row 증가 때문인지 분리한다. top-N은 selection과 output byte를 늘리고, row 증가는 projection과 normalization 전체를 늘린다. TP 변경은 local width를 줄이지만 reduction을 추가한다. 하나의 평균 latency에 세 축을 섞지 않는다.

### 코드 리뷰에서 요구할 변경 영향표

새 sampling option이나 fast path가 들어오면 리뷰 설명에 `input field → validation/default → state producer → tensor mutation → consumer → observable effect → fallback`을 요구한다. processor order를 바꾸는 변경은 기존 API 의미 변경인지, 특정 mode만의 수정인지 명시한다. in-place update를 추가하면 raw logprob가 mutation 전에 복사되는지와 temporary memory가 실제로 줄었는지 본다.

distributed logprob 최적화는 exactness 범위를 명시한다. selected token만 exact한지, top-N 값과 rank도 exact한지, arbitrary bias·grammar·top-p를 지원하는지 적는다. 지원하지 않는 조합은 validation에서 거부하거나 검증된 gather fallback으로 보내야 한다. 조용히 local normalization이나 고정 후보 근사로 바꾸면 API 숫자의 의미가 달라진다.

batching 변경은 request-row metadata를 tensor와 같은 permutation으로 이동시키는 테스트가 필요하다. token ID, logprob, top-N, finish reason이 하나의 commit record로 묶이는지 본다. output field마다 별도 cursor를 두면 partial stream과 parser split에서 정렬이 깨질 수 있다. 이미 19장에서 다룰 transport frame 이전에 engine output tuple의 identity를 닫는다.

### 완료 판정과 독자용 한 페이지 dossier

완료 판정표의 첫 줄은 artifact와 source revision이다. 둘째는 raw score identity다. 셋째는 effective processor 목록과 순서, identity가 아닌 option 값이다. 넷째는 history/count owner와 commit frontier다. 다섯째는 global normalization method와 TP group이다. 여섯째는 selected token, returned stage, row/step mapping이다.

수치 증거에는 작은 다섯-token 표, TP 두-rank LSE 계산, 실제 workload byte 원장을 함께 넣는다. 작은 표는 배열 의미를 증명하고 TP 계산은 global 정보 요구량을 증명하며 실제 원장은 최적화 가치를 설명한다. source evidence에는 validation, penalty application, temperature branch, logprob normalization, output mapping 함수가 포함된다. runtime evidence에는 effective config, stage digest, collective trace, response fixture가 포함된다.

rollback 조건은 막연한 품질 저하가 아니다. identity fixture에서 raw와 processed가 다름, processor order fixture 불일치, TP=1과 global LSE 불일치, request-step mapping 불일치, non-finite 또는 empty distribution, 목표 SLO/peak memory 초과를 각각 명시한다. rollback 뒤 기존 path와 동일 API stage 의미가 복원됐는지 확인한다.

독자가 이 dossier를 만들 수 있다면 다음 질문에 답할 수 있다. “반복이 왜 줄었는가”에는 count와 score 감소량으로 답한다. “temperature가 왜 문장을 바꾸는가”에는 difference scaling과 probability, RNG로 답한다. “logprob가 왜 비싼가”에는 selected rows, global normalization, top-N과 serialization byte로 답한다. “어디를 더 파야 하는가”에는 최초로 갈린 stage의 producer와 consumer 함수로 답한다.

이것이 이 장에서 요구한 친절함이다. 용어를 쉽게 바꾸는 데 그치지 않고, 독자가 숫자를 직접 계산하고 source에서 같은 변환을 찾고 장애에서 틀린 경계를 배제할 수 있게 한다. 다음 장에서는 이 processed distribution이 top-k·top-p·grammar와 RNG를 거쳐 한 token으로 선택되는 순간을 같은 방식으로 확장한다.

마지막 rehearsal에서는 일부러 세 실패를 주입한다. 첫 실행은 rank 1의 local LSE를 global 값처럼 반환한다. 두 번째는 batch compaction 뒤 A와 B의 logprob row를 교환한다. 세 번째는 rejected speculative token을 frequency count에 남긴다. 운영자가 각각 TP normalization, output inverse mapping, lifecycle state로 15분 안에 분류할 수 있는지 본다. 같은 “logprob가 다르다”라는 표면 증상에서 서로 다른 첫 divergence를 찾아야 한다.

성공 trace도 보존한다. raw score digest, processor order와 effective values, global max·sum 또는 LSE, selected score, 반환 logprob의 산술 관계를 한 request-step에 묶는다. top-N 밖 tail을 저장하지 않아도 이 관계는 검사할 수 있다. stream formatter가 token text를 여러 delta로 나누더라도 logprob record는 정확히 한 logical token commit에 붙어야 한다.

배포 후에는 canary에서 TP 크기, temperature cohort, logprob mode, prompt row bucket별로 reference 차이를 감시한다. 값 분포가 달라지면 model revision 변화인지 sampler stage 변화인지 artifact fingerprint로 나눈다. latency가 달라지면 row 수와 collective operand, serialization byte로 나눈다. correctness와 cost가 같은 dashboard에 보이되 서로 다른 판정 기준을 가진다.

최종적으로 이 장은 probability를 모델의 막연한 확신으로 설명하지 않는다. 그것은 특정 hidden row와 vocabulary artifact가 만든 score에 명시된 request policy와 전역 normalization을 적용한 결과다. 어느 stage의 값인지, 어느 history를 읽었는지, 어느 rank의 질량을 포함했는지, 어느 request-step에 반환됐는지를 잃지 않을 때만 그 숫자는 API 계약이 된다.

완료 보고에는 “테스트 통과” 대신 fixture 이름과 첫 divergence checkpoint, exactness 범위, rollback 경로를 적는다. 다음 담당자는 같은 증거를 재현하고 새 backend에서도 동일한 의미 계약을 검증할 수 있어야 한다. 설명이 운영 가능한 지식이 되는 마지막 조건이다.

그 증거는 release가 바뀌어도 비교 가능한 의미 좌표를 유지해야 한다.
