# 13장. causal attention과 mask의 좌표계

12장은 projection에서 나온 Q·K·V를 head별 tensor로 복원했다. 이제 각 query row가 어떤 key row를 읽을 수 있는지 정해야 한다. 이 규칙을 틀리면 연산은 유한한 숫자를 내고 shape도 맞지만, 요청의 미래 token이나 다른 문서의 padding을 읽는다. 그래서 mask 오류는 단순 품질 결함을 넘어 격리 결함이 될 수 있다.

이 장은 `QK^T/√D→mask→stable softmax→PV`를 작은 숫자로 끝까지 계산한 뒤, prefill·chunked prefill·decode·paged KV에서 같은 의미가 어떻게 다른 metadata가 되는지 내려간다. 먼저 수학적 좌표를 잡고, 그 좌표가 실제 함수와 launcher에서 어느 인자로 표현되는지는 장말 소스 노트에서 고정한다.

## 13.1 작은 QK-softmax-PV를 끝까지 계산한다

attention을 “중요한 token을 찾는다”라고만 설명하면 구현을 검사할 수 없다. 한 head의 query를 `Q[Sq,D]`, key를 `K[Sk,D]`, value를 `V[Sk,Dv]`라고 하자. raw score는 `R=QK^T`, scaled score는 `S=R/√D`, 허용하지 않는 좌표에 additive mask `M`을 더한 값은 `A=S+M`이다. 각 query row에서 `P=softmax(A)`를 계산하고 output은 `O=PV`다.

작은 예제로 모든 단계를 밟자. head dimension `D=2`, query와 key 길이는 3이다.

```text
Q = [[1,0],
     [0,1],
     [1,1]]

K = [[1,0],
     [0,1],
     [1,1]]

V = [[10,0],
     [0,20],
     [30,30]]
```

raw score `QK^T`는 다음과 같다.

```text
R = [[1,0,1],
     [0,1,1],
     [1,1,2]]
```

첫 query `[1,0]`은 첫 key와 내적 1, 둘째와 0, 셋째와 1이다. 둘째 query는 `[0,1,1]`, 셋째는 `[1,1,2]`다. transpose는 key의 token 축을 score의 column으로 옮긴다. score row는 query 위치, column은 key 위치다. 이 두 축을 뒤집으면 causal triangle도 뒤집힌다.

`√D=√2≈1.4142`이므로 scale은 약 `0.7071`이다. scaled score는 대략 다음과 같다.

```text
S = [[0.7071,0,     0.7071],
     [0,     0.7071,0.7071],
     [0.7071,0.7071,1.4142]]
```

왜 `1/√D`인가. Q와 K의 각 성분이 비슷한 분산을 갖고 독립에 가깝다고 보면 D개 곱의 합인 내적 분산은 D에 비례한다. D가 커질수록 raw score 폭이 커지고 softmax가 과도하게 날카로워지는 것을 완화하려는 scale이다. 학습된 모델은 이 규약을 전제로 한다. head dimension이 아니라 hidden size의 제곱근으로 나누거나 backend와 caller가 두 번 scale하면 shape는 맞지만 확률이 달라진다.

세 token의 causal self-attention에서 query 절대 위치 `q`는 key 절대 위치 `k≤q`만 볼 수 있다. keep predicate는 `k≤q`다. additive mask는 허용 좌표에 0, 금지 좌표에 `-∞`를 둔다.

```text
M = [[0, -inf, -inf],
     [0, 0,    -inf],
     [0, 0,     0   ]]
```

첫 row는 `[0.7071,-∞,-∞]`이므로 softmax가 `[1,0,0]`이다. output은 첫 value `[10,0]`이다. 둘째 row는 `[0,0.7071,-∞]`이다. stable softmax를 계산하면 최대 `m=0.7071`을 뺀 값이 `[-0.7071,0]`, 지수는 약 `[0.4931,1]`, 합은 `1.4931`이다. 확률은 `[0.3302,0.6698,0]`이고 output은 `0.3302×[10,0]+0.6698×[0,20]≈[3.302,13.396]`이다.

셋째 row는 모두 허용된다. 최대 1.4142를 뺀 값은 `[-0.7071,-0.7071,0]`, 지수는 `[0.4931,0.4931,1]`, 합은 `1.9862`다. 확률은 대략 `[0.2483,0.2483,0.5035]`다. output 첫 성분은 `0.2483×10+0×0.2483+0.5035×30≈17.588`, 둘째는 `0+0.2483×20+0.5035×30≈20.071`이다. 이 값이 O projection에 넘어간다.

이 손계산에는 네 개의 독립 계약이 있다. Q/K 내적 orientation, scale 값, mask predicate, row-wise softmax axis다. output이 다를 때 이 네 계약을 한꺼번에 “attention 문제”라고 부르지 않는다. raw score, scaled score, masked score, probability, PV output을 각각 checkpoint로 둔다.

stable softmax는 `softmax(x)=exp(x-m)/Σexp(x-m)`에서 row 최대 `m`을 뺀다. 모든 원소에 같은 상수를 빼도 확률은 같다. 큰 양수의 exp가 overflow하는 것을 막고 가장 큰 원소의 exp를 정확히 1 근처로 둔다. softmax accumulation을 fp32로 올리는 reference가 많은 이유도 작은 확률의 합과 normalization을 안정화하기 위해서다.

scale과 softmax dtype을 구분한다. QK matmul accumulator dtype, scale multiplication dtype, mask dtype, row max·exp·sum dtype, output cast dtype이 각각 다를 수 있다. “attention은 bf16”이라는 한 문장으로는 numeric contract를 설명하지 못한다. eager reference는 [Transformers Qwen3.5 `eager_attention_forward`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L604-L627)에서 matmul, scale, mask slicing, fp32 softmax, PV와 transpose 순서를 보여 준다.

GQA에서는 query head마다 대응하는 KV head를 정해야 한다. query heads 8, KV heads 2라면 보통 네 query head가 한 KV head를 공유한다. eager 구현은 K/V를 query head 수까지 논리적으로 repeat할 수 있고, fused backend는 mapping을 kernel 안에서 처리할 수 있다. 두 방식은 같은 output을 내야 하지만 물리 K/V read와 temporary는 다르다. 12장에서 만든 global/local head mapping을 이 장의 score head 축에 그대로 가져온다.

query heads 0~3은 KV head 0, query heads 4~7은 KV head 1을 공유한다고 하자. query head `h_q`가 읽는 KV head는 `floor(h_q/4)`다. TP=2에서 rank 0이 query heads 0~3, rank 1이 4~7을 갖는다면 각 rank는 KV head 하나를 갖는다. TP=4에서 rank마다 query heads 두 개라면 KV head 0은 rank 0·1, KV head 1은 rank 2·3에 복제되어야 한다.

eager `repeat_kv`는 K/V의 group 축을 expand한 뒤 query head 축으로 reshape할 수 있다. expand view는 storage를 복제하지 않을 수 있지만 reshape나 contiguous 요구가 copy를 만들 수 있다. fused kernel은 query head index에서 KV head index를 계산해 원본 K/V를 직접 읽을 수 있다. 결과 불변조건은 head별 raw score와 PV output이지, K tensor의 물리 head 수가 아니다.

output만 보고 score를 역추측하지 않는다. 서로 다른 probability와 V가 같은 PV를 우연히 만들 수 있다. 작은 fixture에서는 V rows를 basis vector로 둔다. `V0=[1,0,0]`, `V1=[0,1,0]`, `V2=[0,0,1]`이면 output이 probability row 그 자체다. value dimension이 작으면 `[1,10,100]`처럼 자릿수가 다른 scalar를 쓴다.

attention dropout은 추론에서 보통 0이어야 한다. module이 training mode이거나 dropout 인자가 잘못 전달되면 같은 input도 달라진다. differential 전에 eval mode와 dropout을 고정한다. probability 합이 1이라는 검사만으로는 부족하다. 잘못된 causal row도 합은 1이므로 금지 좌표의 probability가 0인지 support를 먼저 본다.

head mapping 오류는 mask 오류와 비슷한 finite 오답을 만든다. raw score부터 reference와 다르면 먼저 Q/K와 head mapping을 보고, raw score는 같은데 masked score부터 다르면 mask를 본다. probability까지 같고 PV만 다르면 V head mapping이나 V layout을 본다. first divergence로 소유권을 닫는다.

## 13.2 같은 predicate를 prefill·chunk·decode에 적용한다

causal mask를 왼쪽 아래 삼각형 그림으로만 기억하면 query 길이와 key 길이가 달라지는 순간 실패한다. 먼저 predicate를 쓴다. query의 절대 position을 `p_q`, key의 절대 position을 `p_k`라 하면 causal 허용 조건은 `p_k≤p_q`다. sliding window `W`를 더하면 보통 `p_q-W+1≤p_k≤p_q`다. padding은 해당 key가 실제 token인지, packed document mask는 query와 key가 같은 문서인지 추가로 묻는다.

최종 keep predicate 예시는 다음처럼 합성된다.

```text
keep(q,k) = same_request(q,k)
         and key_is_valid(k)
         and p_k <= p_q
         and (global_layer or p_k >= p_q-W+1)
```

prefix-LM 같은 규칙은 더 복잡하다. prefix 구간의 query와 key는 서로 양방향으로 볼 수 있고, 생성 구간 query는 prefix 전체와 자기 이전 생성 token을 볼 수 있지만 생성 구간의 미래는 못 본다. 단순 causal boolean 하나로 표현되지 않는다. model과 request contract가 어떤 attention topology를 요구하는지 먼저 확인한다.

padding mask와 causal mask는 서로 다른 질문이다. causal은 시간 순서를, padding은 batch rectangularization 때문에 생긴 가짜 slot을 막는다. `[B,S]` attention mask의 1과 0이 score의 `[B,H,Sq,Sk]` additive bias로 확장될 수 있지만, 모든 backend가 4D dense tensor를 요구하는 것은 아니다. varlen kernel은 padding을 제거하고 cumulative lengths로 request boundary를 복원할 수 있다.

boolean convention은 특히 위험하다. 어떤 API는 `True=keep`, 다른 API는 `True=mask out`을 쓴다. additive convention은 keep에 0, block에 `-∞` 또는 매우 작은 finite 값을 둔다. bool tensor를 float로 cast해 그대로 score에 더하면 block 좌표에 1을 더하는 완전히 다른 연산이 된다. backend adapter가 기대하는 dtype과 truth meaning을 source에서 읽어야 한다.

2×3 keep predicate가 `[[True,False,False],[True,True,False]]`라고 하자. `True=keep` API에는 그대로 넘긴다. `True=block` API에는 `[[False,True,True],[False,False,True]]`가 필요하다. additive bias에는 `[[0,-∞,-∞],[0,0,-∞]]`가 필요하다. 세 tensor는 같은 의미를 표현하지만 raw 값과 dtype은 다르다. backend 비교는 representation parity가 아니라 각 절대 좌표의 keep 의미를 본다.

mask broadcast 축도 적는다. key padding `[B,Sk]`는 `[B,1,1,Sk]`, causal mask는 `[1,1,Sq,Sk]`, head-specific bias는 `[B,H,Sq,Sk]`일 수 있다. B와 H 숫자가 우연히 같으면 잘못 끼운 축도 broadcast에 성공한다. 숫자 shape만 쓰지 말고 축 이름을 붙인다.

fp32 additive mask를 bf16 score에 더하면 result dtype과 backend eligibility가 바뀔 수 있다. mask를 bf16로 내리면 finite sentinel 표현이 바뀐다. adapter가 score dtype 최솟값을 만드는지, fp32 bias를 유지하는지, boolean predicate를 kernel에 직접 넘기는지 확인한다.

finite sentinel을 `-65504`처럼 fp16 최솟값으로 두면 exp가 사실상 0이 될 수 있지만, 이후 scale/add 순서나 dtype 승격에 따라 의미가 달라질 수 있다. `-1e9`는 fp16로 cast될 때 `-∞`가 될 수 있다. `-∞`는 명확해 보이지만 all-masked row에서 `max=-∞`, `x-max=-∞-(-∞)=NaN`이 된다. sentinel 선택만으로 all-masked 문제를 해결했다고 볼 수 없다.

all-masked row는 왜 생기는가. padding query 자체를 score에 남겼거나, chunk offset을 잘못 적용해 첫 query보다 작은 key가 하나도 없거나, window와 cache offset이 어긋났거나, empty/static cache의 미사용 slot만 보게 만들었을 수 있다. 구현은 invalid query output을 0으로 두거나 row를 계산에서 제외하는 계약을 가질 수 있다. reference와 backend가 같은 계약인지 확인한다.

`x=[-∞,-∞]`의 row max도 `-∞`다. shift는 `-∞-(-∞)`라 NaN이고 exp와 sum도 NaN이다. finite sentinel `[-65504,-65504]`라면 shift `[0,0]`, softmax `[0.5,0.5]`다. NaN은 사라졌지만 invalid keys에 균등 확률을 주었다. sentinel 변경은 좌표 결함의 해결책이 아니다.

invalid query를 packed input에서 제거하거나 softmax 뒤 row를 zero로 만들거나 kernel이 valid-row predicate로 건너뛸 수 있다. 어느 방식이든 downstream residual 계약을 확인한다. padding query가 최종 token selection에 쓰이지 않는지도 별도 invariant다.

packed request 안의 여러 document를 document ID로 격리하는 경우도 있다. cu_seqlens가 request boundary만 표현하면 document boundary에는 별도 predicate가 필요하다. padding이 없다는 사실은 모든 key가 허용된다는 뜻이 아니다.

sliding window `W=4`, query position 5가 최근 네 key를 포함한다면 positions 2·3·4·5다. 식은 `p_k≥p_q-W+1`이다. `p_k≥p_q-W`라고 쓰면 다섯 token을 허용한다. 문서의 “window 4”를 source의 inequality로 내려가야 한다.

prefix cache hit에서도 마지막 partial block capacity를 logical length로 오해하면 unused slots가 열린다. prefix identity가 맞는 것과 mask extent가 맞는 것은 별개다. sink token처럼 window 밖에서도 허용되는 예외가 있다면 predicate는 단순 interval이 아니며 backend 표현 능력을 확인한다.

두 요청을 길이 3과 1로 padding해 `[2,3]`으로 만들었다고 하자. 둘째 요청의 실제 token은 위치 0 하나이고 위치 1·2는 padding이다. causal triangle만 적용하면 query 2가 padding key 1·2 일부를 볼 수 있다. key padding mask가 필요하다. padding query row도 최종 output이나 loss에서 제거되어야 한다. embedding의 padding index와 attention mask는 별개라는 사실을 기억한다.

packed prefill에서는 두 요청 token을 `[A0,A1,A2,B0]`처럼 한 축에 붙일 수 있다. causal predicate를 flattened index에 적용하면 B0가 A0~A2를 과거로 보고 읽는다. request boundary metadata가 필요하다. cumulative sequence lengths `cu_seqlens=[0,3,4]`라면 A는 `[0,3)`, B는 `[3,4)`다. 각 segment 안에서 local causal 관계를 적용하되 절대 model position은 요청별 position 규약을 따른다.

Transformers의 causal mask construction은 [고정 revision `masking_utils.py:720-820`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/masking_utils.py#L720-L820)에서 공통 mask argument와 callable을 따라간다.

SDPA mask 생략 gate는 [같은 파일 `:235-278`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/masking_utils.py#L235-L278)에서 query/KV 길이, offset, padding, tracing, local window 조건을 읽는다. mask를 만들지 않는 경로는 의미를 버리는 것이 아니라 `is_causal` 같은 backend flag가 동일 predicate를 표현할 수 있다고 판정한 경우여야 한다.

mask 생략은 correctness와 dispatch를 동시에 바꾼다. 4D custom mask가 없으면 PyTorch SDPA가 fused backend를 선택할 여지가 커질 수 있다. 그러나 static cache의 미래 capacity slot, continuation의 unequal Q/K lengths, local window, packed documents가 있으면 단순 `is_causal`로 부족할 수 있다. config에서 SDPA를 골랐다는 사실만으로 실제 mask representation과 CUDA kernel을 단정하지 않는다.

Qwen3.5 text model은 layer 종류에 따라 mask mapping을 만든다. [Transformers `modeling_qwen3_5.py:1147-1215`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1147-L1215)에서 full attention mask와 recurrent 계열 mask의 생성·선택을 본다. architecture가 hybrid라면 “모든 layer가 같은 causal mask를 받는다”는 가정부터 버린다.

### 13.2.1 prefill·chunk·decode의 서로 다른 직사각형

빈 cache에서 길이 4 prompt를 prefill하면 `Sq=Sk=4`이고 causal keep 행렬은 정사각 lower triangle이다. query 0은 key 0, query 1은 key 0·1, query 3은 key 0~3을 본다. 이 경우 local row index와 absolute position이 모두 0~3이라 그림이 단순하다.

두 번째 chunk로 absolute positions 4·5를 처리한다고 하자. cache에는 positions 0~3의 K/V가 있고 current chunk가 positions 4·5의 K/V를 제공한다. `Sq=2`, `Sk=6`이다. keep matrix는 다음과 같다.

```text
absolute query 4: key 0 1 2 3 4 허용, key 5 금지
absolute query 5: key 0 1 2 3 4 5 허용

[[1,1,1,1,1,0],
 [1,1,1,1,1,1]]
```

chunk local query index 0을 absolute position 0처럼 취급하면 첫 row가 key 0만 보게 된다. 반대로 모든 current chunk key를 과거로 취급하면 query 4가 미래인 key 5를 본다. causal triangle의 diagonal은 matrix의 왼쪽 위가 아니라 query offset과 key offset으로 정렬해야 한다.

일반식으로 query local row `i`의 절대 위치가 `q_start+i`, key local column `j`의 절대 위치가 `k_start+j`라면 `k_start+j≤q_start+i`를 검사한다. contiguous cache에서 key start가 0일 수 있지만 sliding window에서는 왼쪽 과거가 제거되어 `k_start>0`일 수 있다. physical column 0을 position 0으로 보면 안 된다.

decode에서 신규 query가 한 token이면 보통 `Sq=1`, `Sk=context_length`다. context positions 0~5 뒤 position 6을 decode한다면 한 row의 모든 key 0~6이 과거 또는 현재다. 미래 slot이 operand에 없으면 별도 causal triangle이 필요 없을 수 있다. 하지만 static cache가 capacity 128 전체를 K/V shape로 노출하면 positions 7~127의 미사용 slot을 막아야 한다.

그래서 decode에서 `is_causal=False`를 보았다고 즉시 future leak이라 결론 내리지 않는다. operand가 현재까지 K/V만 포함하면 미래 column이 없다. 반대로 static capacity나 speculative candidates가 포함되면 false가 위험하다. flag는 operand extent와 함께 판정한다.

정사각 prefill에서는 top-left와 bottom-right triangle이 같아 alignment bug가 숨는다. `Sq=2`, `Sk=5` bottom-right query positions가 3·4라면 keep rows는 keys 0..3과 0..4다. top-left로 해석하면 0과 0..1만 본다. unequal fixture가 필수다.

`Sq=5`, `Sk=2`는 앞 query rows가 all-masked가 될 수 있다. API가 이 topology를 허용하는지, 허용한다면 output을 어떻게 정의하는지 확인한다. 일반 decode에서 드물어도 negative test는 alignment convention을 선명하게 드러낸다.

FlashAttention 계열 API는 unequal sequence lengths에서 causal mask 정렬 규약을 명시한다. vLLM의 vendored interface는 [고정 revision `flash_attn_interface.py:187-283`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/vllm_flash_attn/flash_attn_interface.py#L187-L283)에서 bottom-right alignment 예와 `block_table`, `seqused_k`, softmax scale을 설명한다. `Sq=2`, `Sk=5`에서 단순 upper-left triangle과 bottom-right causal alignment는 다르다.

bottom-right 정렬을 절대 좌표로 해석하면 query rows가 key sequence의 마지막 두 positions에 대응한다. query 0은 key 0~3, query 1은 key 0~4를 볼 수 있다. cached continuation에 자연스럽다. API version이 top-left 정렬을 쓴다고 가정하면 cached decode·chunk에서만 오답이 생기고 정사각 prefill test는 통과할 수 있다.

ragged batch에서는 요청마다 `Sq`, `Sk`, offsets가 다르다. query 누적 시작 `qo_indptr=[0,2,3]`이면 첫 요청 query 두 개, 둘째 요청 query 한 개다. KV 누적 시작 `kv_indptr=[0,6,10]`이면 KV 길이는 각각 6과 4다. query request 1이 KV request 0 segment를 읽지 않도록 segment identity가 kernel plan에 들어간다.

indptr 첫 값은 0, 값은 단조 비감소, 마지막 값은 flattened row 수와 같아야 한다. difference `[2,1]`과 `[6,4]`가 요청별 길이다. query와 KV totals가 우연히 같은 fixture에서는 두 indptr를 바꿔도 shape가 맞을 수 있어 segment identity test가 필요하다.

zero-length segment를 API가 허용하는지도 확인한다. cancelled request를 metadata에서 제거하지 못하면 adjacent equal indptr가 생긴다. plan이 이를 지원하지 않으면 host compaction이 필요하다. empty KV를 가진 query는 all-masked 계약과 연결된다.

current chunk K/V를 cache에 먼저 commit하고 전체 `Sk`를 읽는 구현과 prefix cache·current K/V를 별도 operand로 넘기는 구현이 모두 가능하다. 자기 K는 포함하고 뒤 current K는 막아야 한다. cache commit owner와 mask offset owner를 잇는다.

speculative decoding에서는 continuation인데도 `Sq>1`일 수 있다. accepted prefix length가 query start를 정한다. rollback 뒤 cache logical length는 줄었는데 mask offset이 이전 candidate 끝에 남으면 stale position을 읽는다.

chunked prefill과 decode를 같은 flattened tensor에 섞을 수도 있다. 요청 A는 query positions 4·5, KV length 6이고 요청 B는 decode position 9 하나, KV length 10일 수 있다. global `Sq=3`만 보고 하나의 triangle을 만들 수 없다. request별 `q_start`, logical positions, KV length가 필요하다. continuous batching metadata가 attention correctness의 일부인 이유다.

sliding window `W=4`에서 absolute query 5는 causal 과거 0~5 중 positions 2~5만 본다. current physical cache가 `[2,3,4,5]`라면 column 0의 absolute key position은 2다. query 5의 local keep row는 모두 true다. query 4가 같은 cache extent를 본다면 position 5는 미래여서 막아야 한다. 물리 cache window와 query time을 함께 본다.

global/local hybrid attention에서는 layer별 predicate가 다르다. global layer는 전체 retained context를, local layer는 window 안을 본다. local layer output이 오래된 token 변화에 간접 영향을 받을 수는 있다. 이전 global layer가 그 정보를 residual에 이미 섞었기 때문이다. mask correctness test는 synthetic layer input을 직접 넣어 해당 layer의 direct attention만 격리한다.

prefill과 decode 동등성은 강한 invariant다. 같은 prompt를 한 번에 prefill한 마지막 token attention output과, 앞 token을 cache에 넣고 마지막 token을 `Sq=1` decode한 output이 tolerance 안에서 같아야 한다. chunk 크기 1·2·전체도 같은 final hidden을 만들어야 한다. 차이가 생기면 position, cache write, mask alignment, backend numeric order를 순서대로 나눈다.

## 13.3 paged metadata를 내려 한 wrong-row 사건으로 닫는다

이제부터 page table, online softmax, launcher, coordinate ledger, ragged indptr와 배포 회귀표를 별개 workbook으로 만들지 않는다. 모두 13.1의 기대 확률을 보존하면서 `절대 query/key 위치 → request-local 길이 → page·slot → kernel row → PV 결과`를 한 장부에 기록한다. 이 장의 대표 사건은 **두 번째 chunk의 첫 query row가 이전 page의 마지막 유효 key 대신 다음 물리 slot을 읽는 wrong-row** 하나다. 뒤의 모든 수치 fixture와 source 좌표는 이 사건의 최초 divergence를 좁히거나 재발을 막는 자료로 합친다.

교과서 행렬의 key column `j`는 연속 K tensor의 `j`번째 row처럼 보인다. paged KV에서는 요청의 논리 token `j`가 `block_table`을 통해 physical block과 slot으로 번역된다. block size가 4이고 요청의 block table이 `[7,2,9]`라면 logical positions 0~3은 physical block 7, 4~7은 block 2, 8~11은 block 9에 있다.

논리 position `j`의 logical block은 `j//4`, offset은 `j%4`다. physical block은 `block_table[j//4]`다. attention score의 column 순서는 여전히 logical 0,1,2...여야 한다. physical block 번호 2,7,9 순으로 읽는 것이 아니다. kernel은 page를 gather하면서 논리 순서를 보존한다.

context length가 6이면 마지막 logical block에서 slots 4·5만 유효하다. physical block의 나머지 두 slot은 다른 과거 데이터나 미초기화 값일 수 있으므로 읽으면 안 된다. `seq_lens` 또는 `kv_len`이 마지막 page의 유효 extent를 제한한다. block table shape만 맞고 length가 하나 크면 stale slot leak이 생길 수 있다.

두 요청이 prefix block을 공유할 수 있다. block table이 A `[7,2]`, B `[7,5]`라면 block 7은 같은 prefix다. request boundary는 block table row와 logical length가 보존한다. physical page가 공유된다는 사실이 서로의 private suffix를 볼 권한을 뜻하지 않는다. cache allocator의 refcount와 attention metadata를 별도 owner로 둔다.

paged decode에서 query 한 row는 block table을 따라 여러 K/V tile을 읽고 online softmax state를 갱신할 수 있다. 긴 context를 partition으로 나누면 각 partition이 local max, local exp sum, weighted V accumulator를 만들고 두 번째 reduction이 global row를 합친다. partition max만 평균하거나 local outputs를 단순 평균하면 틀린다. max 차이를 반영해 rescale해야 한다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- FlashInfer wrapper는 paged/ragged metadata와 workspace를 plan/run 수명으로 다룬다.
- 고정 source의 batch prefill wrapper는 [FlashInfer v0.6.17 `prefill.py:650-850`](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/prefill.py#L650-L850), paged decode wrapper는 [`decode.py:530-760`](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/decode.py#L530-L760)에서 `indptr`, indices, last-page length, head counts, page layout, workspace와 plan/run 인자를 읽는다.
- 고정 commit의 파일 배치를 확인하며 함수 이름보다 metadata 의미를 기록한다.

- SGLang의 FlashInfer backend는 [고정 revision `flashinfer_backend.py:1-220`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_backend.py#L1-L220)에서 wrapper와 workspace, metadata initialization을 찾고, [같은 파일 `:220-520`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_backend.py#L220-L520)에서 forward/plan 분기를 잇는다.
- scheduler가 만든 request/token metadata가 backend의 `qo_indptr`, page indices, last-page lengths가 되는 생산 경로도 runner에서 거슬러 올라간다.

- vLLM의 attention layer는 backend implementation을 위임한다.
- [vLLM v0.27.1 `attention/layer.py:223-550`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/attention/attention.py#L223-L550)에서 model이 넘기는 Q/K/V, scale, cache와 implementation 경계를 읽고, FlashAttention backend는 [`attention/backends/flash_attn.py:1-260`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/flash_attn.py#L1-L260)에서 metadata와 forward를 읽는다.
- revision에서 파일이 이동했다면 고정 tree의 실제 symbol로 좌표를 갱신해야 한다.

paged metadata는 작은 int tensor라서 부차적이라고 보기 쉽다. 그러나 block index 하나가 잘못되면 다른 request의 page를 읽을 수 있고, context length 하나가 틀리면 stale slot을 확률 분모에 넣는다. value가 유한하면 NaN 없이 정보가 섞인다. page table·sequence length·slot mapping은 correctness와 격리의 핵심 입력이다.

request가 preempt되어 blocks를 반납하고 재개되면 새 physical pages를 받을 수 있다. logical positions는 같아도 block table identity는 바뀐다. backend plan이나 graph가 옛 indices를 재사용하면 다른 request가 재할당받은 page를 읽을 수 있다. batch shape만 같은 것으로 plan metadata 재사용을 승인하지 않는다.

prefix 공유의 copy-on-write도 중요하다. A와 B가 마지막 partial prefix block을 공유하다 B가 빈 slot에 suffix를 쓰면 A cache가 오염될 수 있다. allocator가 partial block을 공유하지 않거나 write 전에 copy해야 한다. attention kernel은 table대로 읽으므로 cache ownership 결함을 mask로 고칠 수 없다.

block table dtype과 device는 ABI다. int32/int64 차이, host plan과 device run 사이 복사, compiled binding signature를 연결한다. page layout도 `[blocks,slots,heads,D]`와 `[blocks,heads,slots,D]`가 다르다. block size 4, heads 2, D 3처럼 서로 다른 작은 숫자로 stride 오류를 드러낸다.

KV quantization이면 payload와 scale page가 같은 logical block을 가리켜야 한다. scale이 token별·head별·block별인지에 따라 indexing이 다르다. attention output부터 틀려도 먼저 dequantized K/V slice를 비교해 page mapping과 quantization을 나눈다.

block table 장애 fixture는 block마다 식별 가능한 K/V를 둔다. physical block 7은 700대, block 2는 200대 값으로 채운다고 생각한다. logical order `[7,2]`를 따라 output이 700대 page 뒤 200대 page를 보는지 source offset을 전개한다. physical sort `[2,7]`를 하면 즉시 표식 순서가 바뀐다.

last-page fixture는 context length를 block boundary `B-1`, `B`, `B+1`로 둔다. block size 4라면 3·4·5다. 길이 4에서 둘째 page를 읽지 않아야 하고 길이 5에서 둘째 page slot 0만 읽어야 한다. off-by-one은 경계 fixture가 가장 잘 잡는다.

격리 사고를 끝까지 분기해 보자. 요청 B의 첫 decode output이 동시 요청 A의 prompt를 바꿀 때 달라진다. sampling을 고정하고 B의 Q와 absolute position이 불변인지 확인한다. 다음으로 B block table, context length, refcount를 비교한다. raw K/V를 logical order로 gather한 작은 slice에 A 표식이 보이는지 본다.

gathered K/V부터 A 값이 보이면 allocator, table, cache commit owner다. gathered K/V는 맞는데 score support가 A segment를 포함하면 ragged indptr나 request boundary owner다. support는 맞고 output만 다르면 head layout이나 softmax numeric을 본다. “GPU nondeterminism”이라는 넓은 설명으로 보안성 문제를 덮지 않는다.

경쟁 가설은 A page가 B table에 들어간 재사용 오류, B valid length가 capacity로 부푼 off-by-one, flattened segment indptr가 한 칸 밀린 오류다. block 표식, boundary length, indptr difference가 각각 다른 가설을 기각한다.

복구 검증은 동시 request 내용과 batch order를 바꾸어 B output 불변을 확인한다. block boundary 3·4·5, partial prefix 공유, preemption/reallocation fixture를 포함한다. table identity와 valid length를 독립 축으로 검사한다. source 감사는 update 순서를 증명하고 실제 격리 결과는 승인된 후속 환경에서 수집한다.

## 13.4 online softmax가 수학을 보존하는 조건

naive eager attention은 score `[Sq,Sk]`, masked score, probability를 HBM에 materialize할 수 있다. 긴 prefill에서는 이 행렬 크기가 sequence 길이의 제곱으로 증가한다. FlashAttention은 Q/K/V tile을 on-chip memory로 가져와 score tile을 계산하고 row별 running statistics를 갱신해 전체 score와 probability 행렬을 HBM에 쓰지 않는다.

online softmax 병합을 작은 예제로 보자. 한 query row의 scores가 두 tile `a=[1,2]`, `b=[3,0]`으로 나뉜다. 첫 tile max `m1=2`, exp sum `l1=e^-1+1≈1.3679`다. value weighted accumulator도 같은 shifted exponent로 만든다. 둘째 tile까지 global max는 `m2=3`이다. 이전 state는 `exp(m1-m2)=e^-1`을 곱해 새 기준으로 rescale한다. 새 sum은 `l=l1×e^-1 + e^(3-3)+e^(0-3)≈0.5032+1+0.0498=1.5530`이다.

value를 scalar `[10,20,30,40]`이라 하자. 첫 tile accumulator는 `e^(1-2)×10+e^(2-2)×20≈23.679`다. global max가 3으로 올라가면 이전 accumulator를 `e^(2-3)`로 rescale해 약 8.710으로 만든다. 둘째 tile 기여는 `30+e^-3×40≈31.991`이다. 최종 accumulator 40.701을 sum 1.5530으로 나누면 output은 약 26.208이다.

전체 scores를 한 번에 계산해도 max 3, exponent `[e^-2,e^-1,1,e^-3]`, denominator 1.5530, weighted sum 약 40.702로 같은 결과다. tile별 softmax output을 평균하는 알고리즘과 다르다. global max가 바뀔 때 이전 sum과 accumulator를 rescale하는 것이 핵심이다.

tile 전체가 masked여도 이전 tile에 valid key가 있으면 row 전체가 invalid인 것은 아니다. causal/window bounds로 완전히 미래인 tile을 건너뛰고 완전히 과거인 tile은 element predicate 없이 처리할 수 있다. diagonal tile만 세밀한 mask가 필요하다. chunk offset 오류는 tile classification 자체를 뒤집는다.

online state dtype도 기록한다. QK와 output이 bf16이어도 max, sum, accumulator를 fp32로 둘 수 있다. 실제 specialization의 template와 kernel body를 확인한다. source에 여러 specialization이 있다는 사실만으로 선택된 dtype 경로를 단정하지 않는다.

이 rescale이 없으면 tile마다 따로 softmax한 뒤 평균하는 잘못된 결과가 된다. online algorithm은 global softmax와 수학적으로 같은 분모를 만든다. floating 연산 순서와 tile 크기가 달라 bitwise result는 다를 수 있지만 표준 FlashAttention은 근사 mask나 top-k attention을 의미하지 않는다.

value accumulator도 이전 max 기준에서 새 max 기준으로 rescale한다. 마지막에 accumulator를 global exp sum으로 나눠 output row를 얻는다. partitioned paged decode reduction도 같은 원리를 더 큰 단위에서 쓴다. `(max, sum, weighted value)` state가 충분하고, probability matrix 전체는 필요 없다.

HBM IO가 줄어드는 것이 핵심이다. FLOP 수가 극적으로 사라지는 것이 아니다. causal tile을 건너뛰어 일부 계산을 줄일 수 있지만, 주된 설계는 score/probability intermediate의 read/write를 줄이는 데 있다. 따라서 “FlashAttention은 softmax를 안 한다”는 설명은 틀리다. softmax를 online 형태로 수행한다.

FlashAttention 고정 source의 Python interface와 launcher는 [flash-attention `flash_attn_interface.py:80-220`](https://github.com/Dao-AILab/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/flash_attn/flash_attn_interface.py#L80-L220), varlen callable은 [같은 파일 `:560-750`](https://github.com/Dao-AILab/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/flash_attn/flash_attn_interface.py#L560-L750)에서 Q/K/V layout, cumulative lengths, scale, causal, window, return LSE를 읽는다.

CUDA entry와 kernel specialization은 고정 commit의 [`csrc/flash_attn/flash_api.cpp:400-620`](https://github.com/Dao-AILab/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/csrc/flash_attn/flash_api.cpp#L400-L620)과 [`csrc/flash_attn/src/flash_fwd_launch_template.h:1-180`](https://github.com/Dao-AILab/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/csrc/flash_attn/src/flash_fwd_launch_template.h#L1-L180)을 잇는다.

Python에서 `causal=True`를 넘겼다는 사실은 launcher argument를 증명할 수 있지만 특정 kernel specialization이 실제 GPU에서 선택됐다는 사실까지 증명하지 않는다. dtype, head dimension, architecture, dropout, window, build flags가 dispatch에 관여한다. source 경로 존재, eligibility, selected launcher, profiler symbol을 네 단계로 분리한다.

return softmax LSE는 row별 `log(sum(exp(scores)))`와 max 안정화를 결합한 통계를 제공할 수 있다. 전체 probability matrix는 아니다. debugging에서 LSE가 맞다는 사실은 normalization 분모가 맞다는 강한 신호지만 V mapping까지 증명하지 않는다. score/mask가 같아도 V가 틀리면 output이 틀린다.

fused kernel의 한계도 분명히 한다. 중간 score를 직접 보기 어렵고, 지원하지 않는 arbitrary mask는 fallback이나 별도 representation이 필요하며, non-contiguous layout이 copy를 만들 수 있다. 짧은 sequence에서는 planning·launch overhead가 이득보다 클 수 있다. “항상 빠르다”가 아니라 shape·dtype·mask·hardware 조건에서 IO 절감이 비용을 상쇄하는지 측정한다.

논문의 IO-aware 이득과 서비스 latency를 구분한다. 실제 요청에는 QKV projection, cache gather, layout copy, launcher와 scheduler gap이 있다. attention kernel이 빨라져도 TTFT가 거의 같을 수 있다. prefill은 많은 query-key 조합과 intermediate IO, decode `Sq=1`은 매 step의 긴 K/V read와 page gather가 중요하다.

window attention은 읽는 byte를 줄이지만 full-attention model에 임의 window를 넣으면 모델 수학을 바꾼다. model이 local layer를 요구하는 경우와 performance option이 backend만 바꾸는 경우를 분리한다. “최적화”라는 이름이 semantic 변경을 숨기지 못하게 한다.

attention weights 반환 요청은 fused 경로를 닫을 수 있다. probability 전체 materialization은 FlashAttention의 IO 절감과 충돌한다. 일부 interface의 debug output이나 LSE는 전체 probability가 아니다. 관측 옵션이 backend와 memory를 바꾸는 observer effect를 기록한다.

SDPA, FlashAttention, FlashInfer를 같은 이름으로 묶지 않는다. PyTorch SDPA는 high-level operator이며 dispatcher가 math, memory-efficient, flash 계열 backend를 고른다. FlashAttention은 contiguous/varlen fused attention package와 kernel이다. FlashInfer는 서빙에서 paged KV, ragged batch, plan/run, workspace를 포괄하는 library다. 같은 attention 수학을 보존해도 ABI와 cache owner가 다르다.

## 13.5 네 스택에서 좌표가 함수와 launcher가 되는 길

- Transformers walk는 model 의미에서 시작한다.
- [Qwen3.5 attention class `modeling_qwen3_5.py:629-705`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L629-L705)에서 head config, scale, causal flag와 forward의 interface 선택을 읽는다.
- [eager function `:604-627`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L604-L627)은 관찰 가능한 reference다.
- [text model `:1147-1215`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1147-L1215)은 cache position과 layer별 mask를 준비한다.

Transformers attention adapter는 [고정 revision `integrations/sdpa_attention.py:1-120`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/sdpa_attention.py#L1-L120)에서 query/key/value layout, GQA, mask와 `is_causal`이 PyTorch operator로 넘어가는 경계를 보여 준다.

Flash adapter는 [`integrations/flash_attention.py:1-240`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/flash_attention.py#L1-L102)에서 unpadding, varlen, window와 callable selection을 읽는다. model source만 보고 backend layout을 추측하지 않는다.

- vLLM walk는 model layer의 `Attention` construction에서 attention layer로, backend metadata와 implementation으로, vendored FlashAttention interface와 compiled extension으로 내려간다.
- [`attention/layer.py`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/attention/attention.py#L223-L550)는 scale, heads, cache config와 forward boundary를 소유한다.
- [`attention/backends/flash_attn.py`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/flash_attn.py#L1-L300)는 prefill/decode metadata와 callable을 소유한다.
- [`vllm_flash_attn/flash_attn_interface.py:187-417`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/vllm_flash_attn/flash_attn_interface.py#L187-L417)는 varlen/paged arguments를 extension으로 넘긴다.

- SGLang walk는 scheduler/runner가 만든 forward batch에서 attention backend metadata가 되는 길을 먼저 찾는다.
- model의 attention call은 [`srt/layers/attention/base_attn_backend.py:1-180`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/base_attn_backend.py#L1-L180)의 interface를 통과하고, FlashInfer implementation은 [`flashinfer_backend.py:1-520`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_backend.py#L1-L520)에서 plan/run과 cache metadata를 잇는다.
- backend registry는 [`attention_registry.py:1-180`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/attention_registry.py#L1-L180)에서 요청 이름과 concrete class를 연결한다.

- llama.cpp walk는 graph input mask와 KV cache mask writer부터 시작한다.
- [`llama-graph.cpp:400-520`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L400-L520)에서 causal flag와 input mask tensor가 graph에 들어가는 경계를 보고, [`llama-kv-cache.cpp:1537-1758`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L1537-L1758)에서 ubatch position, sequence identity, sliding window와 causal predicate가 mask data가 되는 코드를 읽는다.

- graph의 attention 연산은 [`llama-graph.cpp:2520-2635`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L2520-L2635)에서 KQ, scale, mask, softmax, V multiplication 또는 fused flash op 구성을 찾는다.
- ggml CUDA flash attention launcher는 [`ggml/src/ggml-cuda/fattn.cu:1-220`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/fattn.cu#L1-L220)와 dispatch 범위에서 head dimension·dtype·mask 인자를 확인한다.
- source에 op가 있다는 사실과 current graph가 이를 선택했다는 사실은 다르다.

네 stack을 함수명으로 일렬 정렬하지 않는다. 의미 좌표 표를 쓴다.

| 의미 | Transformers | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| absolute query/key 좌표 | cache position·mask kwargs | runner metadata | forward batch | ubatch positions |
| request/ragged boundary | padding/packed mask | cumulative lengths | indptr | sequence IDs |
| physical KV mapping | cache class | block table | page indices | KV cell/cache mapping |
| attention callable | eager/SDPA/Flash adapter | backend impl | backend class | graph op/backend |
| device launcher | PyTorch/external package | vendored extension | FlashInfer/native | ggml CUDA |

이 표에서 빈 칸이 있으면 source chain이 아직 닫히지 않은 것이다. model의 `is_causal=True`와 kernel의 page table 사이에 누가 offsets와 lengths를 만들었는지 찾아야 한다.

실제 review에서는 model constructor에서 heads, KV heads, D, scale, causal, window를 먼저 기록한다. forward에서 Q/K/V layout과 cache update 순서를 적는다. mask builder에서 query/KV lengths, offsets, padding·document·window 합성을 적는다. adapter에서 representation과 layout copy를 적고 launcher에서 metadata argument order를 적는다.

vLLM과 SGLang에서는 runner metadata producer를 생략하지 않는다. backend가 받은 length가 맞는지 보려면 scheduler output과 input builder를 거슬러 올라간다. cache manager의 block과 runner table row가 같은 request incarnation을 가리키는지 확인한다. cancelled slot reuse는 model attention file만 읽어서는 찾을 수 없다.

llama.cpp에서는 ubatch token의 position과 sequence IDs가 mask writer로 들어간다. nested predicate가 KV cell position과 sequence membership을 어떻게 비교하는지, 만들어진 mask tensor가 graph KQ score에 어느 orientation으로 더해지는지 잇는다. host mask가 맞아도 graph transpose가 다르면 triangle이 뒤집힌다.

FlashInfer plan/run에서는 plan metadata와 run Q/K/V/cache를 구분한다. batch나 indptr 변경 때 plan 재사용 조건, workspace 수명, graph mode auxiliary address 안정성을 본다. plan 성공은 page contents와 request identity의 증거가 아니다.

정적 감사가 증명하지 못하는 것도 적는다. 특정 GPU에서 FA2/FA3 중 무엇이 선택됐는지, SDPA가 math/flash 중 무엇을 골랐는지, layout copy latency와 numeric error 분포는 runtime trace가 필요하다. source inspection은 예상 상태와 관찰 필드를 만든다.

## 13.6 한 coordinate ledger에서 wrong-row 후보를 가른다

coordinate ledger는 shape보다 많은 것을 담는다. 요청 ID, query local index, query absolute position, key logical range, key absolute start, physical block/slot, valid length, window start, document ID, query/KV head mapping을 둔다. tensor에는 shape, stride, dtype, storage alias를 붙인다.

예시 행은 다음과 같다.

| request | q row | `p_q` | allowed `p_k` | physical pages | Q head→KV head | mask repr |
|---|---:|---:|---|---|---|---|
| A | 0 | 4 | 0..4 | `[7,2]`, len 5 | 0..3→0 | varlen causal |
| A | 1 | 5 | 0..5 | `[7,2]`, len 6 | 4..7→1 | varlen causal |
| B | 0 | 9 | 6..9, W=4 | `[11]`, offset 6 | 0..3→0 | paged window |

shape ledger는 Q `[total_q,Hq,D]`, paged K/V `[num_blocks,block_size,Hkv,D]`, `qo_indptr`, page indices, page indptr, last page length을 함께 둔다. stride가 kernel ABI와 맞는지, Q head가 contiguous한지, page layout이 NHD인지 HND인지 확인한다.

첫 장애는 padding leak이다. 길이 3과 1 요청을 padded batch로 만들었는데 둘째 요청 output이 첫째 요청 padding 값 변화에 따라 달라진다. 경쟁 가설은 key padding mask 누락, request boundary 누락, V layout 오류다. raw score의 둘째 request row에서 invalid key columns가 존재하는지 보고, masked score가 sentinel인지, probability가 정확히 0인지, PV가 invalid V 변화에 불변인지 차례로 확인한다.

negative test는 padding V에 매우 큰 식별값을 넣는다. mask가 맞으면 output은 변하지 않는다. output이 변하되 probability probe가 invalid column 0이면 mask owner다. probability는 0인데 output이 변하면 V indexing이나 request boundary owner다. 최종 logits만 비교하는 것보다 빠르다.

둘째 장애는 causal off-by-one이다. query position `p`가 key `p` 자신을 보지 못하거나 key `p+1`을 본다. diagonal fixture에서 각 value를 one-hot identity로 두면 output support가 허용 set을 드러낸다. predicate가 `<`인지 `≤`인지, cache write가 attention 전인지 후인지 함께 본다. decode에서 current K/V를 cache에 먼저 쓴 뒤 `length`를 하나 잘못 늘리면 미래 stale slot을 포함할 수 있다.

셋째 장애는 chunk boundary다. full prefill은 맞지만 chunk size 4에서 두 번째 chunk 첫 token부터 다르다. Q/K/V projection과 RoPE 뒤까지 같다면 mask query offset, key offset, cache commit length를 본다. absolute query 4의 expected allowed keys 0..4를 종이에 적고 backend local row가 어느 columns를 허용하는지 비교한다. local triangle을 그린 흔적은 prefix 0..3이 빠지거나 current key 5가 새는 형태로 나타난다.

넷째 장애는 all-masked NaN이다. 짧은 ragged request나 padding query에서 attention output이 NaN이고 collective 뒤 전체 layer로 퍼진다. 첫 non-finite가 masked score인지 row max 이후인지 확인한다. masked score가 모두 `-∞`면 upstream coordinate가 row에 허용 key를 하나도 주지 않은 이유를 찾는다. sentinel만 finite 값으로 바꿔 NaN을 숨기면 invalid row가 arbitrary uniform probability를 가질 수 있다.

GQA mapping incident도 mask처럼 보일 수 있다. 특정 query heads만 오래된 token을 잘못 참조한다면 query→KV head map과 TP replication을 본다. 모든 heads가 같은 key columns을 허용하더라도 다른 K/V head를 읽으면 attention pattern이 달라진다. coordinate ledger에 head identity를 넣는 이유다.

sliding/global incident에서는 window 경계 `W-1`, `W`, `W+1` 길이를 쓴다. local layer의 query `p`가 `p-W` key를 막고 `p-W+1`을 허용하는지 확인한다. physical circular cache가 slot을 덮어썼다면 slot index보다 absolute position tag를 본다. global layer는 같은 key를 허용해야 한다. layer type mapping이 뒤바뀌면 둘의 증상이 교환된다.

first divergence decision은 다음 순서다. Q/K/V와 absolute positions가 reference와 같은가. raw QK score가 같은가. scale 후가 같은가. allowed predicate가 같은가. backend representation이 같은 의미인가. softmax LSE와 probability가 같은가. PV output이 같은가. page read와 V head가 같은가. 이 순서를 지키면 kernel 전체를 막연히 의심하지 않는다.

shape·stride 장부 예를 만들자. 요청 A query 2개, B query 1개, query heads 8, KV heads 2, D=64라면 packed Q는 `[3,8,64]`, 신규 K/V는 `[3,2,64]`일 수 있다. cache가 blocks 20개, block size 16이면 NHD layout은 `[20,16,2,64]`다. axes가 다른 layout과 숫자로 구분된다.

Q element stride가 `(512,64,1)`인지 head-major transpose view인지 기록한다. cache stride는 block, slot, head, D와 일치하는지 적는다. wrapper가 Q를 contiguous로 만드는지, page layout flag를 넘기는지 본다. implicit copy는 correctness를 보존해도 latency와 graph address를 바꾼다.

physical slot과 absolute position tag를 함께 둔다. circular cache slot 0은 여러 시간에 재사용된다. slot index만으로 어느 token인지 알 수 없다. request incarnation이나 write epoch가 있어야 stale read를 찾는다.

backend마다 dense bool, additive float, indptr, compressed block predicate를 쓰므로 raw mask hash는 공통 비교값이 아니다. 작은 coordinate set에서 predicate oracle을 평가해 bitset을 만들면 semantic mask fingerprint가 된다.

성능 장부에는 score materialization, unpadding/layout copy bytes, page table와 K/V reads, workspace, LSE return, kernel count를 적는다. correctness 장부와 fixture ID로 연결한다. mask 표현 변화가 fused eligibility를 바꾸면 correctness와 latency가 동시에 움직인다.

좋은 장애 문장은 좌표가 있다. “FlashInfer가 틀린다”가 아니라 “B의 query 17에서 expected keys 0..17인데 last-page length가 3 대신 4여서 page 9 slot 3의 stale V가 nonzero probability를 얻고 block boundary 15에서는 oracle과 일치한다”라고 쓴다. owner와 반증이 보인다.

수정 뒤에는 failing coordinate와 양옆 boundary, request permutation을 본다. length 17 결함이면 15·16·17과 다른 block size를 검사한다. physical page permutation과 table을 함께 바꿔도 output이 같아야 한다. 회귀 행렬은 결함의 차원을 반영한다.

## 13.7 단일 differential 장부와 14장 handoff

이 장의 구현 관찰점은 Transformers v5.15.1 commit `550d7b3834670483a4df436541272c055dc364bf`, vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, llama.cpp v0.2.0 commit `bb4caa7540188872173c44d161602d9271386413`에 고정했다. FlashAttention과 FlashInfer도 같은 방식으로 고정 source의 함수 계약을 읽었다. 아래 비교는 모델·서버·CUDA 실행 결과가 아니라 수식, source와 손계산을 연결한 정적 분석이다.

실행이 허용되는 후속 검증을 위해 workbook을 설계하되 이 장에서는 실행하지 않는다. fixture는 square prefill, unequal-length continuation, chunked prefill, single-token decode, padded batch, packed ragged batch, paged block boundary, sliding window boundary, GQA head sharing, all-masked invalid query를 포함한다.

각 fixture는 같은 Q/K/V numeric input을 eager oracle과 대상 backend에 준다. model 전체가 아니라 attention operator boundary를 먼저 비교한다. raw score를 제공하지 않는 fused backend는 output과 softmax LSE, mask predicate oracle을 비교하고, failure가 나면 eager와 작은 shape로 축소한다.

관찰 field는 다음과 같다.

```text
revision/backend/callable/device build identity
Sq, Sk, head counts, head dimension, scale
query absolute starts, key absolute starts, valid lengths
request/document segments, window, causal flag
Q/K/V shape, stride, dtype, page layout
block table or page indices, indptr, last-page length
mask representation dtype and truth convention
row max, LSE, finite ratio, selected probability rows
attention output error and first divergent layer/head/row
```

정확성 판정은 mask support를 먼저 본다. 금지 좌표 probability가 0인지, 허용 row probability 합이 1인지, invalid query row 계약이 맞는지 검사한다. 그다음 numeric tolerance를 본다. fp32 eager와 저정밀 tiled kernel의 bitwise equality를 요구하지 않지만, 잘못된 support를 tolerance로 덮지 않는다.

metamorphic test가 강하다. 금지된 미래 V를 큰 값으로 바꿔도 현재 output은 변하지 않아야 한다. window 밖 V를 바꿔도 local layer output은 변하지 않아야 한다. 다른 request padding을 바꿔도 output은 변하지 않아야 한다. 허용된 현재 V를 바꾸면 output이 변해야 probe가 살아 있음을 확인할 수 있다.

chunk equivalence는 full prefill, chunks `[4,2]`, `[1,1,1,1,1,1]`, cached decode의 마지막 position output을 비교한다. 동일 mask topology와 position/cache state라면 허용 오차 안에서 같아야 한다. 차이가 numeric tiling order인지 coordinate 오류인지 LSE와 support로 나눈다.

paged equivalence는 같은 logical K/V를 contiguous 배열과 서로 다른 physical block permutation에 둔다. block table이 논리 순서를 복원하면 output은 같아야 한다. page permutation을 바꾸고 table도 함께 바꾸었을 때 불변이어야 한다. table만 바꾸면 의도적으로 달라져 probe sensitivity를 확인한다.

backend 변경 기록은 option→requested backend→eligibility→actual callable→mask ABI→launcher→metric 순서다. eager에서 FlashAttention으로 바꿔 빨라졌다는 결과만 쓰지 않는다. unpadding copy, workspace plan, fallback, graph capture, LSE return 여부가 함께 바뀔 수 있다. correctness gate를 먼저 통과한 뒤 latency·HBM traffic을 측정한다.

첫 실험은 3×3 손계산을 fixture로 고정한다. Q/K/V, raw·scaled scores, keep matrix, probabilities, output을 저장한다. fp32에서 먼저 맞추고 저정밀은 별도 tolerance를 쓴다. fused backend가 중간을 주지 않으면 output과 LSE를 비교한다.

둘째는 `(Sq,Sk)=(2,5),(1,5),(5,2)` 직사각 alignment다. query starts와 expected support를 bitset으로 둔다. all-masked row contract를 source로 고정한다. 정사각 5×5는 control일 뿐 alignment 증거가 아니다.

셋째는 ragged isolation이다. 두 request V에 다른 큰 표식을 넣고 batch order를 A/B, B/A로 바꾼다. 각 output은 다른 request 내용과 order에 불변이어야 한다. cancellation로 한 segment를 제거한 metadata도 검사한다.

넷째는 chunk equivalence다. absolute positions와 commit length를 매 chunk 적고 sizes 전체·2·1을 비교한다. 첫 차이가 chunk 시작이면 query offset, chunk 끝이면 current K inclusion과 length increment를 우선한다.

다섯째는 paged permutation이다. 동일 logical KV를 pages `[7,2,9]`와 `[4,8,1]`에 두고 table을 맞추면 output은 같아야 한다. unused last slot에는 극단값을 넣어 valid length를 시험한다.

여섯째는 local/global layer다. window 밖 V를 바꾸어 local operator 불변, global operator 민감성을 본다. residual 간접 효과를 배제하도록 layer input을 고정한다. window boundary 세 길이를 쓴다.

일곱째는 GQA다. KV head마다 orthogonal K와 digit-coded V를 둔다. query group별 expected output과 TP mapping을 비교한다. eager repeat와 native GQA가 같아야 한다. K는 맞고 V만 바꾼 negative fixture도 둔다.

여덟째는 online numeric이다. tile 사이 max가 크게 변하는 scores와 거의 동률인 scores를 모두 쓴다. LSE와 output error를 비교하되 mask support나 page mapping 오류를 tolerance로 통과시키지 않는다.

결과에는 fixture, expected predicate, callable, representation, first divergence, max error, prohibited probability mass, 아직 실행하지 않은 항목을 남긴다. source revision과 extension build identity가 없으면 재현 가능한 보고가 아니다.

### 사건 기록 1: full prefill은 맞고 두 번째 chunk 첫 row만 틀린다

하나의 장애를 시간순으로 끝까지 닫아 보자. 길이 10 prompt를 한 번에 처리하면 reference와 맞지만 chunks `[4,4,2]`로 처리하면 absolute position 4부터 logits가 달라진다. position 0~3까지는 양쪽 경로가 같다. 이 증상은 “FlashAttention 오차”보다 chunk boundary 좌표 가설을 먼저 세우게 한다.

재현 계약은 token IDs, model revision, dtype, attention backend, cache 빈 상태, chunk sizes를 고정한다. sampling 이전의 layer attention output을 비교한다. full과 chunked에서 layer 0 norm, Q/K/V projection, RoPE 적용 뒤 tensor의 position 4 row가 허용 오차 안에 같은지 본다. 여기까지 같다면 13·12장과 RoPE projection은 우선 기각된다.

position 4 query의 기대 절대 predicate는 keys 0,1,2,3,4 허용, 5 이후 금지다. 둘째 chunk의 local query row 0과 key operand를 장부에 적는다. prefix K/V length 4와 current K/V length 4를 함께 읽는다면 `Sq=4`, `Sk=8`, query absolute start 4다. expected support rows는 `0..4`, `0..5`, `0..6`, `0..7`이다.

관찰 결과를 가정하지 않고 세 경쟁 가설을 준비한다. A는 query start가 0으로 들어가 local triangle `0`, `0..1`, `0..2`, `0..3`만 허용한다. B는 current chunk 전체를 과거로 취급해 모든 row가 `0..7`을 허용한다. C는 prefix cache commit length가 3으로 기록되어 logical key 3이 빠지고 current keys offsets도 한 칸 당겨진다.

A는 semantic support bitset이 각 row에서 너무 짧아지는 것으로 드러난다. B는 첫 row에서 keys 5~7 probability가 nonzero인 것으로 드러난다. C는 block/table에서 gathered K rows의 absolute tags와 query offset이 동시에 밀린다. probability를 반환할 수 없는 fused path에서는 V를 basis 또는 digit-coded로 둔 작은 operator fixture와 LSE를 사용한다.

source branch는 metadata producer부터 소비자까지 잇는다. scheduler가 chunk token count와 already-computed count를 만든다. runner가 flattened query rows, cumulative query starts, sequence length와 block table을 만든다. backend plan이 이를 query/key indptr와 lengths로 바꾼다. launcher는 causal alignment와 offsets를 받는다. 어느 함수가 absolute start를 직접 넘기지 않는다면 unequal lengths와 bottom-right convention이 이를 암묵적으로 표현하는지 확인한다.

반증은 한 축씩 바꾼다. chunk sizes `[5,5]`에서 first divergence가 position 5로 이동하면 chunk-local offset 가설이 강해진다. prefix length 0인 첫 chunk가 계속 맞는 것도 같은 방향이다. backend를 eager dense mask로 바꿨을 때 맞는다면 Q/K/V와 cache payload보다 backend representation 또는 metadata adapter로 좁혀진다. 그러나 eager가 cache construction을 다르게 한다면 완전한 단독 반증은 아니므로 payload를 함께 비교한다.

수정은 단순히 position 4 fixture를 통과하는 것으로 닫지 않는다. chunk sizes 1,2,3,전체와 block boundary를 가로지르는 chunks를 본다. prefill final output과 repeated decode output도 비교한다. query start, key logical start, valid length 세 좌표가 각각 expected를 만족해야 한다. 숫자가 우연히 같아지는 정사각 chunk만 검사하지 않는다.

인계 문장은 이렇게 쓴다. “layer 0 position 4에서 Q/K/V와 RoPE는 reference와 일치한다. expected support 0..4와 달리 backend plan은 query local start 0으로 causal row를 만들어 support 0만 남긴다. chunk size를 5로 바꾸면 first divergence도 5로 이동한다. metadata producer의 computed-token count는 4지만 adapter가 query offset에 반영하지 않는다.” 이 문장은 owner와 반증을 담는다.

### 사건 기록 2: NaN은 mask에서 보였지만 원인은 request compaction이다

두 번째 장애는 부하 중 드물게 한 request의 attention output이 NaN이 되고 같은 batch의 다른 requests도 뒤 layer에서 NaN이 되는 상황이다. profiler symbol이 attention softmax라서 kernel numeric 문제로 보이지만, 먼저 first non-finite row와 coordinate ledger를 찾는다.

문제 row의 masked scores가 모두 `-∞`이고 Q/K raw values는 유한하다고 하자. 이는 softmax가 NaN을 표면화했지만 row에 valid key가 하나도 없다는 뜻이다. 왜 empty row가 생겼는지 upstream metadata를 본다. 해당 flattened query가 어느 request ID와 absolute position인지, `qo_indptr`, page `indptr`, context length, last-page length를 기록한다.

요청 하나가 직전에 취소되어 batch compaction이 일어났다고 하자. Q rows는 cancelled request를 제거해 새 order `[A,C]`인데 page-table rows는 한 step 동안 옛 order `[A,B,C]`의 앞 두 rows `[A,B]`를 썼다면 C query가 B의 empty/released cache metadata를 받는다. Q shape와 batch size는 둘 다 2라 assertion이 통과할 수 있다.

경쟁 가설 A는 finite sentinel/softmax 구현 자체, B는 zero-length request를 plan이 지원하지 못한 것, C는 compaction permutation 불일치다. A는 올바른 metadata의 ordinary rows에서도 재현되어야 한다. B는 indptr에 equal adjacent entries가 남아 있어야 한다. C는 query request IDs와 block-table owner IDs를 row별로 비교하면 드러난다.

NaN을 없애려고 all-masked output을 zero로 만드는 patch는 방어층일 수 있지만 C의 cross-request table 오류를 고치지 않는다. 잘못된 page에 valid length가 있었다면 NaN 없이 데이터가 샜을 것이다. 따라서 safe softmax와 metadata identity 검증은 둘 다 필요하며, 전자가 후자를 대체하지 않는다.

정적 source 감사에서는 compaction owner가 Q rows, sequence lengths, block tables, slot mapping을 같은 permutation으로 재배열하는지 본다. asynchronous host-to-device copy가 있다면 metadata buffers가 같은 step generation을 가리키는지 본다. graph replay가 고정 주소 buffer를 쓴다면 contents update와 synchronization owner를 확인한다.

negative fixture는 batch `[A,B,C]`에서 B를 취소하고 C만 zero-length와 다른 page pattern을 갖게 한다. Q 표식, block 표식, request incarnation을 각 row에 부여한다. batch order를 바꿔 오류가 특정 index가 아니라 compaction gap을 따르는지 확인한다. 이 장에서는 실행하지 않고 expected permutation과 assertion을 정의한다.

수정 검증은 NaN 0건만 세지 않는다. 각 Q row와 page-table row owner identity가 같고, prohibited cross-request probability mass가 0이며, cancellation·preemption·prefix-sharing permutations에서 유지되어야 한다. all-masked row가 합법적으로 생기는 padding topology는 별도 expected contract로 검사한다.

### source를 열었을 때 적을 함수 감사 카드

함수 하나를 읽을 때 입력 이름만 나열하지 않는다. `semantic input`, `shape/layout`, `coordinate source`, `state read/write`, `output meaning`, `fallback`, `next owner`의 일곱 칸을 채운다. mask builder라면 coordinate source가 cache position과 query/KV offsets이고, output meaning은 dense bias나 backend predicate다. backend wrapper라면 page table과 indptr가 semantic input이며 launcher가 next owner다.

Transformers eager 함수 카드에는 query `[B,Hq,Sq,D]`, repeated 또는 mapped key `[B,Hq,Sk,D]`, scale, sliced additive mask, fp32 row softmax, PV와 transpose를 적는다. mask construction 카드는 past seen tokens, cache position, target length, padding/document/window pattern과 backend-specific conversion을 적는다.

vLLM attention layer 카드는 model이 넘긴 Q/K/V와 output buffer, KV cache state, metadata context를 적는다. backend implementation 카드는 prefill/decode 구분, cumulative lengths, block tables, sequence lengths와 causal/window flags를 적는다. vendored Flash interface 카드는 Python tensor가 extension argument가 되는 정확한 순서를 적는다.

SGLang FlashInfer backend 카드는 plan에 들어가는 qo/page indptr와 last-page length, heads, page size, workspace를 적고 run에 들어가는 Q와 cache를 분리한다. plan이 host와 device auxiliary buffers 중 무엇을 보존하는지, forward mode마다 어떤 wrapper를 고르는지 기록한다.

llama.cpp mask writer 카드는 ubatch token position, sequence IDs, KV cell position, causal/window predicate, output mask orientation과 dtype을 적는다. graph builder 카드는 mask가 KQ tensor 어느 축에 더해지고 softmax와 V multiply가 어떤 node로 이어지는지 적는다. CUDA fattn 카드는 op tensor extent와 specialization dispatch 조건을 적는다.

FlashAttention interface 카드는 fixed/varlen 함수, Q/K/V layout, cumulative lengths, max lengths, causal alignment, window, scale, dropout, return LSE를 적는다. C++ launcher 카드에는 dtype, head dimension, architecture gate와 kernel template 선택을 적되 실제 선택은 runtime 미검증으로 남긴다.

FlashInfer 카드에는 page layout, index dtype, KV heads, query heads, head dim, page size, window, softmax scale과 custom mask 지원을 적는다. wrapper version에 따라 함수명과 parameter가 달라질 수 있으므로 tag와 symbol을 함께 고정한다. 일반 설명을 현재 revision의 사실처럼 쓰지 않는다.

### 측정 결과를 해석하는 순서

prohibited probability mass가 0이 아니면 numeric tolerance를 논하지 않고 mask correctness를 실패로 판정한다. support가 맞고 LSE가 크게 다르면 scale, score dtype, online reduction을 본다. LSE가 맞고 output만 다르면 V mapping과 accumulator를 본다. output이 맞고 latency만 나쁘면 layout copy, fallback, planning, tile와 IO를 본다.

긴 prompt에서만 latency가 급증한다고 mask 결함으로 단정하지 않는다. dense fallback 때문에 score temporary가 제곱으로 커질 수 있고, paged gather가 fragmented될 수 있고, local window option이 적용되지 않았을 수 있다. actual callable과 kernel, K/V bytes, temporary allocation을 관찰한다. mask tensor가 존재한다는 사실은 원인 후보이지 결론이 아니다.

짧은 decode에서 intermittent spike가 있으면 page plan rebuild, metadata copy synchronization, graph miss, allocator event를 본다. attention kernel time 자체와 host gap을 분리한다. FlashInfer wrapper plan 시간이 kernel range 밖에 있을 수 있다. service timeline과 device timeline을 연결한다.

quality divergence가 특정 sequence length부터 시작하면 block/window/chunk boundary와 대조한다. 정확히 block size+1이면 last page, window+1이면 inequality, chunk start면 offset을 우선한다. 2의 거듭제곱 head dimension boundary면 backend specialization이나 layout padding도 후보다. 증상의 계단 위치가 coordinate 가설을 제공한다.

NaN은 finite ratio를 Q, K, raw score, masked score, row max, LSE, output 순으로 기록한다. all-masked row인지 large finite overflow인지 나눈다. 잘못된 V page는 유한 오답을 내므로 NaN monitoring만으로 격리를 보장하지 않는다. metamorphic isolation test와 metadata identity assertion이 필요하다.

허용 오차는 결과를 본 뒤 넓히지 않는다. eager fp32 oracle, eager activation dtype, fused backend를 계층화해 비교한다. sequence length와 score dynamic range별 error를 기록한다. mask support와 request isolation은 exact semantic invariant이고 numeric tolerance 대상이 아니다.

## 13.8 독자가 스스로 새 backend를 파는 경로

처음 보는 backend에서는 이름보다 signature를 읽는다. Q/K/V axes, cache representation, lengths와 offsets, causal/window/custom mask, GQA mapping, scale, output과 auxiliary statistics를 표로 옮긴다. 그다음 producer를 찾아 각 metadata가 어느 request state에서 왔는지 확인한다.

작은 eager oracle을 만든 다음 backend representation으로 같은 predicate를 encode한다. square prefill 하나로 만족하지 않고 unequal, ragged, page boundary, window boundary를 넣는다. negative fixture가 실제로 output을 바꾸는지 확인해 probe sensitivity를 증명한다.

함수에서 compiled binding으로 내려갈 때 argument order와 dtype assertion을 잇는다. binding에서 launcher, launcher에서 specialization과 kernel로 내려간다. Python config 문자열로 profiler symbol을 추측하지 않는다. source는 가능한 경로를, runtime은 선택된 경로를 증명한다.

성능을 볼 때 logical work와 physical work를 나눈다. allowed QK pairs, 실제 K/V bytes, skipped causal tiles, page fragmentation, layout copies, workspace와 launch 수를 센다. backend가 빠른 이유를 “flash라서”가 아니라 줄어든 HBM intermediate, 더 나은 page gather, planning amortization 같은 관찰 가능한 원인으로 설명한다.

마지막으로 문제를 upstream에 보고할 owner를 고른다. model의 head/scale이면 model implementation, mask builder의 offsets이면 framework integration, scheduler metadata면 serving engine, page plan이면 backend wrapper, online reduction이면 kernel library다. first divergence와 고정 source 좌표가 owner 선택을 뒷받침한다.

- source note의 핵심 좌표를 다시 묶는다.
- Transformers eager는 [`modeling_qwen3_5.py:604-627`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L604-L627), mask preprocessing은 [`masking_utils.py:235-278`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/masking_utils.py#L235-L278), SDPA adapter는 [`sdpa_attention.py:1-120`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/sdpa_attention.py#L1-L120)이다.

- vLLM의 상위 attention 경계는 [`attention/layer.py:223-550`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/attention/attention.py#L223-L550)에서 읽는다.
- backend 구현은 [`flash_attn.py:1-300`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/flash_attn.py#L1-L300)으로 이어진다.
- bundled interface 경계는 [`flash_attn_interface.py:187-417`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/vllm_flash_attn/flash_attn_interface.py#L187-L417)에서 확인한다.

SGLang은 [`base_attn_backend.py`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/base_attn_backend.py#L1-L180)와 [`flashinfer_backend.py`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_backend.py#L1-L520)를 연결한다.

llama.cpp는 [`llama-graph.cpp:400-520`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L400-L520), [`llama-kv-cache.cpp:1537-1758`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L1537-L1758), [`llama-graph.cpp:2520-2635`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L2520-L2635)을 연결한다.

- FlashAttention은 [Python interface](https://github.com/Dao-AILab/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/flash_attn/flash_attn_interface.py#L560-L750)에서 [C++ launcher](https://github.com/Dao-AILab/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/csrc/flash_attn/flash_api.cpp#L400-L620)로, FlashInfer는 [prefill wrapper](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/prefill.py#L650-L850)와 [decode wrapper](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/decode.py#L530-L760)를 읽는다.

이 장의 출구 질문은 여덟 가지다. score row와 column이 각각 무엇인가. scale은 어느 D에서 왔는가. keep predicate는 absolute coordinates로 무엇인가. padding·document·window 규칙은 어디서 합성되는가. chunk query offset과 key offset은 무엇인가. paged logical column이 어느 physical page/slot인가. online softmax state는 tile을 어떻게 합치는가. first divergence가 score·mask·softmax·V 중 어디인가.

12장으로 돌아갈 조건은 raw QK score나 head mapping부터 틀린 경우다. 이 장에 남을 조건은 Q/K가 맞고 allowed coordinates, softmax, page read에서 처음 틀린 경우다. 14장으로 넘길 것은 RoPE가 position을 Q/K에 적용하는 방식, GQA의 cache physical shape, MLA와 cache state다. 이 장은 position의 숫자를 소비했지만 그 position embedding을 만드는 수학은 소유하지 않는다.

attention을 이해했다는 것은 삼각형 그림을 기억하는 것이 아니다. 한 query의 절대 시간, 한 key의 논리 시간, request 경계, physical page, head identity가 하나의 허용 predicate와 kernel operand로 이어지는 길을 설명하는 것이다. 이 길이 닫히면 prefill과 decode가 다른 shape로도 같은 causal 의미를 보존하는지 증명할 수 있다.

## 13.9 dense mask가 보이지 않을 때 predicate를 복원한다

처음 이 코드를 읽는 독자에게 가장 당황스러운 장면은 mask tensor가 보이지 않는 경우다. 교과서에서는 언제나 삼각 행렬을 score에 더했는데, 서빙 backend에는 Q, block table, length와 `causal=True`만 있다. mask가 사라진 것처럼 보인다. 그러나 실제로 사라진 것은 dense representation이다. 허용 predicate는 query/key lengths의 정렬, page의 valid extent, request indptr, causal/window flag에 분산되어 있다.

이 관점을 가지면 source를 읽는 순서도 자연스러워진다. 먼저 dense triangle을 찾는 대신 한 query row의 절대 position을 찾는다. 다음으로 이 query가 속한 request segment와 읽을 KV logical length를 찾는다. 그 뒤 logical key가 physical page 어디에 있는지 찾는다. 마지막으로 launcher가 causal alignment와 last-page extent를 어떻게 받는지 본다. 네 단계를 통과하면 dense mask가 없어도 predicate를 복원할 수 있다.

반대로 4D mask tensor가 있다고 안심할 수도 없다. query offset이 잘못되면 잘못된 triangle을 정교하게 materialize할 뿐이다. bool convention이 뒤집히면 shape는 완벽한 채 허용과 금지가 바뀐다. padding 축 broadcast가 틀리면 다른 request의 padding 규칙을 적용할 수 있다. 보이는 tensor의 존재보다 좌표에서 계산한 expected support가 중요하다.

이 장의 작은 3×3 계산은 그래서 단순 입문 예제가 아니다. raw score와 scale, support, probability, V의 책임을 분리하는 기준선이다. prefill이 paged decode가 되어도 이 의미 경계는 남는다. kernel이 score matrix를 쓰지 않아도 online max와 sum은 같은 softmax를 만든다. K/V가 physical page에 흩어져도 logical column order는 같다. GQA가 K/V head를 공유해도 각 query head가 읽을 identity는 결정되어 있다.

성능 설명도 이 의미 기준선 위에서만 안전하다. causal tile skip은 허용 predicate가 맞아야 유효한 최적화다. page gather는 logical order를 보존해야 한다. online softmax는 rescale을 정확히 해야 한다. mask 생략은 operand에 미래 slot이 없거나 backend flag가 같은 predicate를 표현할 때만 가능하다. 빠른 경로가 이 조건을 어기면 그것은 trade-off가 아니라 correctness 결함이다.

독자는 장애를 만났을 때 최종 문장부터 쓰지 말아야 한다. “FlashAttention이 이상하다”, “paged cache가 샌다”, “bf16 오차다”는 모두 너무 큰 결론이다. 대신 한 row를 고른다. 그 row의 Q/K/V와 absolute position을 고정하고, expected key support를 종이에 쓴다. backend metadata로 support를 복원하고, 첫 차이가 score인지 support인지 LSE인지 V read인지 찾는다. 이 작은 행위가 거대한 serving stack을 조사 가능한 조각으로 바꾼다.

예를 들어 query position 17에서만 문제가 생긴다면 expected keys를 먼저 쓴다. full causal이면 0..17, window 8이면 10..17이다. cache block size 16이면 position 17은 두 번째 page slot 1에 있다. continuation `Sq=1`이면 bottom-right 정렬로 마지막 query에 대응한다. 이 네 문장이 있으면 off-by-one, window inequality, page last length, top-left alignment 가설을 서로 구분할 수 있다.

문제가 고쳐졌다는 설명도 같은 좌표로 돌아온다. 단지 token이 정상처럼 보였다고 닫지 않는다. position 17의 support가 expected와 같고, 금지 V 변화에 output이 불변이며, pages를 permutation해도 logical output이 같고, positions 15·16·17 경계에서도 재발하지 않아야 한다. numeric error가 남으면 support와 LSE가 맞는 상태에서 별도 tolerance로 판정한다.

이런 설명은 다소 느리게 느껴질 수 있다. 하지만 원인을 모른 채 backend를 바꾸고 sentinel을 바꾸고 cache를 끄는 반복보다 훨씬 빠르다. 각 관찰이 경쟁 가설 하나를 기각하고 다음 owner로 넘어가게 하기 때문이다. 독자에게 필요한 것은 옵션 목록이 아니라 어느 증거가 어느 결론을 허용하는지에 대한 감각이다.

13장을 빠져나가면 attention을 하나의 검은 상자로 부르지 않게 된다. 12장의 Q·K·V가 score를 만들고, request와 시간 좌표가 support를 정하고, stable 또는 online softmax가 row를 정규화하고, logical V rows가 output을 만든다. serving engine은 이 의미를 lengths, indptr, page table과 launcher arguments로 운반한다. 14장은 이제 이 좌표 중 position이 RoPE와 cache shape를 어떻게 바꾸는지 이어받는다.

마지막으로 흔한 질문 하나를 정리하자. “eager 결과가 기준이면 fused 결과가 한 자리라도 다를 때 버그인가?” 그렇지 않다. tile 순서와 reduction 순서, accumulator dtype이 달라 마지막 bit는 달라질 수 있다. 하지만 causal support, request 격리, window 경계, page의 logical order는 근사 대상이 아니다. 이 둘을 섞으면 정상적인 수치 차이를 결함으로 신고하거나, 심각한 미래 token 누출을 허용 오차로 덮는다.

판정 순서는 의미가 먼저이고 수치가 나중이다. 금지 coordinate에 probability mass가 없는지, 각 query가 자기 request의 올바른 key 범위를 읽는지, GQA가 기대 KV head를 고르는지, page permutation이 logical output을 보존하는지 확인한다. 이 조건이 맞은 뒤 LSE와 output의 absolute·relative error를 dtype별 tolerance로 본다. 마지막으로 model logits와 token 선택 영향을 본다.

이 순서는 디버깅뿐 아니라 최적화 리뷰에도 유용하다. 새로운 kernel이 intermediate IO를 줄였다고 해도 predicate와 logical page walk가 같다는 증명이 먼저다. 수치 차이가 있다면 어느 reduction 순서에서 생기는지 설명하고 error budget을 제시한다. 그 뒤에야 latency, bandwidth, workspace를 비교한다. correctness와 performance가 같은 실험 기록에 있되 서로 다른 판정 열을 갖는 이유다.

독자가 source를 덮을 때 남겨야 할 것은 backend 이름이 아니라 한 문장짜리 인과다. “query absolute start와 KV logical length가 이 metadata에서 나오고, 이 alignment 규칙이 causal support를 만들며, block table이 support의 logical columns를 physical pages로 번역하고, online softmax가 전체 row를 저장하지 않고 같은 normalization을 만든다.” 이 문장을 고정 source 좌표와 함께 말할 수 있다면 attention 경로를 실제로 이해한 것이다.

## 13.10 ragged batch를 indptr 좌표로 복원한다

서빙 batch는 request마다 query와 KV 길이가 다르다. dense `[B,Sq,Sk]` padding 대신 모든 query row를 이어 붙이고 `qo_indptr`, KV는 page/table 또는 `kv_indptr`로 segment를 표현한다. mask는 사라진 것이 아니라 segment boundary와 absolute position으로 분해된다.

### 세 요청의 cumulative length를 손으로 계산한다

요청 A는 query 3, KV total 5, B는 query 1, KV 9, C는 query 2, KV 2라고 하자. packed query rows는 총 6이고 `qo_indptr=[0,3,4,6]`이다. request r의 packed query 범위는 `[qo_indptr[r],qo_indptr[r+1])`다. KV를 contiguous ragged로 저장한다면 `kv_indptr=[0,5,14,16]`이다.

A의 local query row 0~2가 새 positions 2~4에 대응한다고 하자. causal support는 각각 key positions 0..2,0..3,0..4다. B의 packed row 3은 absolute query position 8이고 key 0..8을 본다. C의 rows 4~5는 positions 0,1이다. packed row index 4를 absolute position 4로 쓰면 C가 존재하지 않는 과거를 읽는다.

변환식은 `request = upper_bound(qo_indptr, packed_row)-1`, `local_q=packed_row-qo_indptr[request]`, `abs_q=query_start[request]+local_q`다. `query_start`는 보통 `kv_len - q_len`인 bottom-right causal alignment에서 유도될 수 있지만 chunk/prefix/window 정책에 따라 명시적 coordinate가 더 안전하다. source의 실제 convention을 확인한다.

#### indptr invariant가 request isolation을 만든다

indptr는 0에서 시작하고 비감소하며 마지막 값이 total rows와 같아야 한다. request reorder/compaction 뒤 lengths와 indptr, block table row가 같은 permutation generation을 가져야 한다. 하나만 old order면 A query가 B KV segment를 읽을 수 있다.

empty query나 zero-length segment를 지원한다면 equal adjacent indptr가 가능하다. 이를 malformed로 거부할지 no-op으로 허용할지 backend contract를 읽는다. index dtype도 total rows/pages 상한을 담을 수 있어야 한다. int32 tensor를 int64로 해석하거나 host/device dtype이 다르면 pointer arithmetic이 틀어진다.

wrong-row fixture는 `[A,B,C]`와 `[C,A,B]` 순서를 바꾸고 request identity로 output을 다시 정렬한다. 각 request의 logical support와 output이 같아야 한다. physical packed row와 page number가 달라지는 것은 정상이다. canonical request-position key로 비교한다.

## 13.11 prefix·sliding window·chunked prefill predicate를 합성한다

full causal predicate 하나만으로 실제 serving visibility를 설명할 수 없다. prefix-LM 또는 shared prefix, local sliding window, chunked prefill, padding/document boundary가 겹친다. 규칙을 dense mask 여러 개로 생각하기보다 absolute `(request,q,k)` predicate의 conjunction/disjunction으로 쓴다.

### full causal과 sliding window의 경계

full causal은 `k <= q`다. 왼쪽 window W가 inclusive length W라면 `q-W+1 <= k <= q`다. q=17,W=8이면 allowed keys 10..17로 8개다. `q-W <= k`를 쓰면 9개가 되어 off-by-one이다. window가 past tokens 수를 뜻하는 API라면 식이 다를 수 있으므로 option 정의와 kernel parameter convention을 확인한다.

초기 position q=3,W=8에서는 lower bound가 음수이므로 0으로 clamp해 keys 0..3을 본다. page/block optimization은 lower bound 이전 tiles를 skip할 수 있지만 first partial page의 정확한 slots를 mask해야 한다. physical page를 통째로 버리거나 살리는 것만으로 경계가 맞지 않을 수 있다.

#### prefix가 bidirectional인지 cached causal prefix인지 구분한다

“prefix”는 두 의미가 있다. prefix-LM에서 prefix token끼리 bidirectional visibility를 허용하고 suffix는 prefix+causal suffix를 본다. serving prefix cache는 이미 계산된 causal prefix KV를 재사용할 뿐 visibility rule 자체는 full causal일 수 있다. 같은 단어를 mask predicate에 섞지 않는다.

prefix-LM 길이 P=4, total 7이면 q<4인 prefix query는 k<4를 모두 볼 수 있고, q≥4 suffix query는 k<4 또는 `4<=k<=q`를 본다. q=1이 k=3을 보는 것은 full causal과 다르다. backend가 custom mask를 지원하지 않고 causal flag만 받는다면 prefix-LM을 같은 fast path에 보낼 수 없다.

prefix cache hit length H=4인 ordinary causal decode q=4는 keys 0..4를 본다. cached 0..3이 bidirectional이 된 것은 아니다. cache reuse identity와 mask semantics를 별 field로 둔다.

#### chunked prefill의 query start를 계산한다

길이 10 prompt를 chunks 4,4,2로 처리한다고 하자. chunk 0 query positions 0..3, KV length after include-current 4다. chunk1은 q positions 4..7이고 keys 0..q를 본다. chunk2는 q 8..9다. 각 chunk의 local row는 다시 0부터 시작하지만 causal diagonal은 absolute query start 0,4,8로 이동한다.

chunk1을 `[Sq=4,Sk=8]` rectangular score로 만들면 local row i의 max key는 `past_len+i = 4+i`다. top-left triangle `k<=i`를 쓰면 row0이 key0만 보고 cached 1..4를 놓친다. bottom-right alignment 또는 explicit offsets가 필요하다.

prefix cache hit H=3 뒤 chunk length 4를 처리하면 abs q 3..6, Sk=7이다. scheduler가 `past_len`, `query_start`, `kv_len` 중 무엇을 metadata에 넣는지 source를 따라간다. 같은 숫자가 pre-write인지 post-write인지도 확인한다.

#### 규칙 합성의 우선순위와 empty row

sliding window와 required prefix를 함께 쓰면 prefix keys를 window 밖에서도 유지할지 정책이 필요하다. predicate를 `(k<P) OR (max(P,q-W+1)<=k<=q)`처럼 정의할 수 있다. 단순 window를 마지막에 AND하면 prefix를 잘라 버릴 수 있다. architecture/document policy가 실제 식을 결정한다.

padding과 request boundary는 항상 hard isolation이다. custom document mask가 있어도 다른 request key를 허용하지 않는다. 합성 순서는 최종 predicate로 검산하되 implementation이 bitmask/additive bias/length clamp 어느 방식인지 구분한다.

모든 key가 금지된 query row가 생기면 softmax가 NaN이 될 수 있다. 이런 row가 합법적인 padding no-op인지 malformed request인지 contract를 정한다. kernel이 protected zero output을 내더라도 semantic 오류를 숨기지 않게 reason을 기록한다.

#### logical predicate를 paged kernel metadata로 낮춘다

logical key position k는 block size P에서 `logical_block=floor(k/P)`, `offset=k mod P`가 된다. request block table이 logical block을 physical page로 매핑한다. physical address는 page layout, KV head, token offset, dimension stride와 dtype에 따라 계산된다.

#### page table 계산과 last-page extent

P=4, KV length 10이면 logical blocks 0,1,2가 필요하고 last page valid length는 2다. block table이 `[7,3,11]`이면 key k=0..3은 physical page7, 4..7은 page3, 8..9는 page11 slots0..1이다. page11 slots2..3은 allocation돼 있어도 logical column이 아니다.

q=9 full causal은 0..9를 본다. kernel이 page11 전체를 읽고 invalid slots를 mask하지 않으면 stale value가 softmax에 들어갈 수 있다. last_page_len=2 또는 kv_len=10이 predicate를 제한해야 한다. physical slot에 zero가 있다고 안전한 것이 아니다. zero K score와 nonzero V가 유효 probability를 받을 수 있다.

window W=5이면 q=9 allowed 5..9다. logical block1 slots1..3과 block2 slots0..1이다. page granularity skip 뒤 first/last partial page masks가 필요하다. lower bound를 `q-W=4`로 잘못 계산하면 key4가 하나 더 들어간다.

#### ragged page metadata의 generation

요청 A/B/C는 각자 block table row, page indptr, last page length, KV length를 가진다. compaction/reorder에서 query indptr만 새 order이고 block table은 old order면 cross-request read가 된다. metadata bundle에 request permutation generation을 붙이고 atomic하게 publish한다.

page allocator는 physical page 번호를 재사용할 수 있다. 같은 page7이라도 incarnation이 다르면 다른 request state다. block table generation과 page generation을 맞춘다. cancellation 직후 same page reuse는 중요한 regression fixture다.

prefix sharing은 여러 request block table이 같은 physical page를 가리킬 수 있다. immutable/shared prefix generation과 private suffix ownership을 구분한다. 한 request의 cache write가 shared page를 mutate하지 않도록 copy-on-write 또는 append boundary를 지킨다.

#### FlashAttention varlen과 paged decode 표현을 비교한다

varlen interface는 cumulative sequence lengths, max lengths, causal/window flags로 ragged dense K/V segments를 표현할 수 있다. paged decode는 block/page tables, sequence lengths, page size와 last extent를 사용한다. 둘 다 같은 logical predicate를 다른 physical representation으로 encode한다.

conversion 비교표에는 request segment, abs query start, kv length, window/prefix, logical-to-physical mapping을 둔다. varlen에 page table은 not-applicable이고 paged에 contiguous kv_indptr는 optional일 수 있다. 빈칸을 같은 값으로 처리하지 않는다.

#### wrong-row incident R13

R13은 chunked prefill 두 번째 chunk 첫 row에서만 품질이 틀렸다. Q/K fingerprints와 positions는 reference와 같았다. dense oracle expected support는 keys0..4였지만 backend metadata로 복원한 support는 0만 허용했다. query local index 0을 absolute position으로 사용한 top-left causal alignment가 first divergence였다.

passing neighbor는 first chunk와 full prefill이다. first chunk는 past_len=0이라 local/absolute가 같고 bug가 숨는다. full square prefill도 top-left triangle이 맞는다. rectangular second chunk에서만 드러난다. 이 negative evidence가 offset 가설을 강하게 만든다.

수정은 query_start/past_len을 metadata에 명시하거나 backend의 bottom-right alignment contract에 맞춘다. chunks 1/4/4/2, cache hit0/3, limit boundary와 T=1 decode를 matrix로 검증한다. support exact parity를 먼저 보고 LSE/output tolerance를 나중에 본다.

#### source walk·관측·성능 판정으로 사건을 닫는다

pinned source walk는 mask builder에서 끝나지 않는다. scheduler/request state가 query lengths, past/cache lengths, window/prefix policy를 만드는 producer를 찾는다. runner가 indptr, starts, block tables, slot mapping으로 바꾸는 지점, backend wrapper plan/forward, compiled binding, launcher와 kernel predicate까지 잇는다.

Transformers masking utilities는 dense/SDPA reference predicate와 chunk/window construction의 semantic oracle다. vLLM attention backend metadata는 scheduler/runner state가 varlen/page operands가 되는 경계를 보여 준다. SGLang FlashInfer wrapper는 qo/page indptr와 last page length가 plan/run으로 들어가는 경로다. llama.cpp mask writer와 KV cell mapping은 positions/sequence IDs에서 graph mask를 만든다.

#### 계산 fixture와 source claim을 결합한다

각 source span에 fixture row를 하나 붙인다. query-start 계산은 second chunk row0 keys0..4로, window lower bound는 q9,W5 keys5..9로, last page는 len10,P4 valid2로, reorder는 `[A,B,C]` permutation으로 반증한다. source가 실제 predicate를 어떻게 encode하는지 expected support와 비교한다.

kernel body가 dense support를 materialize하지 않아도 tile/page iteration과 boundary condition에서 같은 set을 복원할 수 있다. 확인하지 못한 template specialization은 미확인으로 남기고 launcher predicate와 필요한 runtime symbol을 적는다.

#### 모니터링은 logical work와 physical work를 분리한다

metric에는 q rows, kv lengths/window-visible lengths, pages, last-page utilization, chunk count, prefix hit, backend/fallback, plan/rebuild reason을 bounded histogram/enum으로 둔다. request별 indptr/table은 trace artifact에 둔다. raw prompt나 block table 전체를 label로 넣지 않는다.

logical allowed pairs는 full causal chunk에서 row별 support 합으로 계산한다. physical work는 tile skip, page reads, padding slots, layout copy와 workspace를 포함한다. 둘의 차이가 최적화 여지를 보여 주지만 profiler/trace 없이 실제 bytes로 단정하지 않는다.

#### correctness와 성능 rollback을 분리한다

cross-request page read, future token exposure, required prefix loss는 즉시 correctness rollback 문턱이다. fallback 증가나 planning overhead는 성능 terminal로 별도 판정한다. fast path를 끄는 완화가 correctness를 회복할 수 있지만 capacity/latency 비용과 해제 조건을 기록한다.

old metadata/cache generation을 새 kernel이 읽지 않도록 graph key와 cache namespace를 version한다. in-flight chunk가 old query-start convention을 가진 채 new reader로 넘어가지 않게 drain/fence한다. cache flush 하나로 scheduler metadata generation까지 해결됐다고 보지 않는다.

#### 최종 regression matrix

축은 ragged order, q length1/>1, kv length, chunk start, prefix cache hit, prefix-LM 여부, window boundary, page boundary/last length, cancellation/reuse, backend dense/varlen/paged다. 모든 Cartesian cell 대신 first divergence를 가르는 boundary와 metamorphic pair를 고른다.

판정은 expected support exact set, request isolation, logical page ordering, prohibited-V invariance, LSE/output tolerance, selected backend와 latency/workspace다. 의미 invariant와 numerical/performance terminal을 한 boolean으로 합치지 않는다.

13장의 종료 문장은 다음과 같다. “Request별 absolute query start와 KV logical length에서 causal/prefix/window predicate를 만들고, ragged indptr와 block table이 동일 generation으로 이를 physical rows/pages에 내리며, launcher/kernel이 partial page와 chunk alignment를 보존한다.” 이 문장에 수치 fixture와 pinned producer/consumer로 답하면 14장의 position/cache 수학으로 넘어갈 수 있다.

#### 처음 보는 kernel metadata도 같은 사건 장부로 감사한다

새 backend signature에 `qo_indptr`, `paged_kv_indptr`, `paged_kv_indices`, `paged_kv_last_page_len`, `window_left`, `causal`이 보인다고 하자. 이름을 해석하는 데서 멈추지 않고 각 tensor element가 어느 request state에서 왔는지, 단위와 pre/post-write 시점, device와 dtype, lifetime을 적는다.

#### metadata 한 줄의 단위를 고정한다

`qo_indptr` 값은 query token row 누적 수이지 byte offset이나 request ID가 아니다. `paged_kv_indptr`는 request별 page-index list의 누적 길이일 수 있고 token 누적 길이와 다르다. `paged_kv_indices` element는 physical page ID다. `last_page_len`은 마지막 page의 valid token 수이며 KV total length 자체가 아니다.

page size P=16, KV len=33이면 pages=3, last_page_len=1이다. kv_len을 last_page_len 자리에 33으로 전달하면 kernel boundary guard가 의미를 잃는다. 반대로 page count 3을 kv_len으로 쓰면 query가 key0..2만 본다. dtype/shape가 모두 integer vector라 schema assertion만으로 잡히지 않을 수 있다.

metadata card에는 `name`, `semantic unit`, `shape`, `dtype`, `producer`, `consumer`, `generation`, `valid range`를 둔다. 같은 int32라도 page ID와 token count를 섞지 않는다.

#### plan과 run 사이 lifetime을 확인한다

FlashInfer류 wrapper는 plan 단계에서 indptr, lengths, workspace와 kernel plan을 준비하고 run 단계에서 Q와 cache를 소비할 수 있다. plan metadata가 host/device auxiliary buffer로 복사되는지, run까지 어떤 object가 소유하는지 확인한다. batch compaction 뒤 old plan을 new Q와 재사용할 수 있는 compatibility predicate가 필요하다.

graph replay에서는 pointer address가 고정돼도 buffer contents와 generation을 갱신해야 한다. query lengths는 current batch인데 block indices는 previous batch라면 valid address로 wrong request page를 읽는다. plan key가 batch size/max length만 포함하고 request permutation을 누락했는지 본다.

#### prefix sharing과 sliding window의 상호작용

두 요청 A/B가 prefix 0..31을 공유하고 suffix가 다르다고 하자. page size16이면 prefix pages 두 개를 공유할 수 있다. A q=40,W=8은 keys33..40을 보므로 shared prefix를 읽지 않을 수 있다. policy가 system prefix를 window 밖에서도 보존하면 shared pages 일부를 계속 읽어야 한다.

cache sharing 여부와 visibility 여부를 구분한다. physical page가 table에 있다고 모두 visible한 것이 아니고, window 밖 page를 table에서 제거하지 않아도 kernel predicate가 skip할 수 있다. allocation/reclaim 최적화는 별 owner다. mask correctness를 physical residency와 동일시하지 않는다.

prefix-LM bidirectional block을 공유한다면 custom predicate 지원이 필요하다. ordinary causal prefix cache fast path가 같은 bytes를 가졌다는 이유만으로 semantic compatibility를 선언하지 않는다. cache key에 mask/prefix policy generation을 포함한다.

#### chunk scheduler와 kernel이 length 시점을 합의한다

chunk를 cache에 write하기 전 metadata가 past_len만 가리키는지, current chunk를 포함한 kv_len을 가리키는지 convention을 정한다. query row가 current K/V를 볼 수 있어야 하므로 kernel이 separate current K/V와 past cache를 받거나 cache write 후 total length를 받는다. 두 방식을 섞으면 current self token이 누락되거나 중복된다.

past=4,chunk=4라면 pre-write kv_len=4와 post-write total=8을 구별한다. local row0 abs q=4는 keys0..4를 봐야 한다. kernel이 total8과 bottom-right alignment를 쓰면 맞을 수 있고, past4를 total로 해석하면 current keys를 보지 못한다. source wrapper와 cache write 순서를 함께 읽는다.

chunked prefill과 prefix hit가 겹치면 cached=3,new chunk=4,total=7이다. scheduler의 accepted prefix length, actual computed query rows, published cache length가 conservation을 만족해야 한다. hit token을 query로 다시 넣거나 new rows를 length에서 빠뜨리지 않는다.

#### document mask와 request isolation

한 request 안에 여러 document가 있고 cross-document attention을 제한하는 policy가 있을 수 있다. request segment와 document segment는 다른 축이다. document indptr/mask가 틀려도 다른 tenant request까지 넘어가서는 안 된다. request isolation은 가장 바깥 hard boundary다.

fixture는 A request documents a1,a2와 B request b1을 packed한다. A의 policy만 바꿔도 B output은 불변이어야 한다. prohibited V rows에 큰 sentinel을 넣어 output이 바뀌지 않는지 본다. 확률을 직접 얻지 못해도 metamorphic isolation으로 누출을 검출한다.

custom mask bit convention도 확인한다. 1이 keep인지 mask인지, bit packing order, row stride와 padding bits를 작은 2×5 predicate로 검산한다. unused padding bit가 1이라 kernel이 extra physical slot을 읽지 않도록 valid length guard도 필요하다.

## 13.12 online softmax가 partial page를 합치는 계산

allowed score를 page/tile별로 처리할 때 running max m과 sum l을 보존한다. 첫 tile scores `[1,2]`이면 m1=2,l1=`e^-1+1`. 둘째 tile `[3]`을 합치면 new m=3, old l은 `l1*e^(2-3)`, new contribution1을 더한다. V accumulator도 같은 rescale을 해야 한다.

mask로 금지된 stale slots는 max/sum에 들어가면 안 된다. 매우 음수 finite bias가 dtype에서 충분히 작지 않거나 invalid slot load가 NaN이면 online state를 오염시킬 수 있다. logical support exactness를 numeric tolerance보다 먼저 검사한다.

page 순서를 physical ID 순으로 처리해도 online softmax는 순서 독립에 가까운 수학을 보존할 수 있지만 V와 score의 logical pair가 함께 이동해야 한다. block table permutation metamorphic test는 physical pages를 바꾸고 logical contents/mapping을 유지했을 때 output parity를 본다.

### 잘못된 metadata가 latency 문제처럼 보이는 경우

window_left가 적용되지 않으면 correctness가 full causal과 같을 수 있지만 intended local-attention architecture에서는 의미와 성능이 모두 달라진다. read KV가 늘어 ITL이 악화된다. selected backend가 custom window를 지원하지 않아 fallback하거나 flag를 무시했는지 본다.

last_page_len이 항상 page size로 설정되면 stale slots가 zero/masked돼 output이 우연히 맞더라도 불필요 read와 tile work가 늘 수 있다. correctness canary와 logical/physical work metric을 함께 둔다. 최적화가 의미 predicate를 바꾸지 않는지 확인한다.

plan rebuild가 매 step 발생하면 kernel은 빠르지만 host gap이 ITL을 지배할 수 있다. plan compatibility가 너무 좁거나 metadata buffer를 재사용하지 못하는지 본다. 반대로 compatibility를 넓혀 stale plan을 재사용하면 correctness 위험이 생긴다. generation predicate와 amortization을 함께 설계한다.

## 13.13 30분 source audit 순서

첫 5분은 semantic oracle이다. failing request 한 row의 abs q, kv len, prefix/window/document 규칙과 expected key set을 쓴다. 다음 5분은 scheduler/runner producer에서 q len, past/total len, indptr를 계산한다. 다음 5분은 block table과 last page로 logical keys를 physical slots에 매핑한다.

다음 5분은 wrapper argument order, dtype/device와 plan/run generation을 본다. 다음 5분은 launcher가 causal/window/custom mask와 alignment를 specialization에 전달하는지 본다. 마지막 5분은 passing neighbor와 boundary matrix, rollback owner를 적는다. source에서 확인 못한 device predicate는 후속 probe로 남긴다.

### option 설명을 실제 state mutation으로 바꾼다

chunk size option은 query rows, chunk start, cache write/commit 횟수, launcher shape와 TTFT/ITL fairness를 바꾼다. sliding window는 lower-bound predicate와 visible KV work를 바꾸지만 physical cache allocation은 별 정책이다. prefix caching은 accepted prefix와 query start/cache namespace를 바꾸지만 causal semantics를 자동으로 bidirectional로 바꾸지 않는다.

attention backend option은 dense/varlen/paged representation, supported predicate, planning/workspace와 fallback을 바꾼다. page size는 block table length, last-page utilization, address arithmetic과 kernel specialization을 바꾼다. 각 option을 parser→state→metadata→launcher→effect→falsifier로 잇는다.

성능 효과는 workload 없이 일반화하지 않는다. 긴 decode에서는 smaller visible window가 KV traffic을 줄일 수 있고, 짧은 prompt에서는 planning/launch overhead가 더 클 수 있다. prefix hit는 prefill work를 줄이지만 fragmentation과 metadata lookup을 추가할 수 있다. 정확성 terminal을 먼저 통과한다.

#### 최종 incident와 배포 terminal

R13 종료 레코드는 “second chunk row0의 abs q=4가 local q=0으로 전달돼 support 0 대신 0..4가 되어야 했다”처럼 expected/observed set을 쓴다. producer symbol, wrong metadata field, wrapper/launcher consumer, passing first-chunk/full-prefill neighbor를 연결한다.

배포는 mask metadata generation, cache layout, graph/plan key와 backend version을 묶는다. old in-flight chunks를 drain하고 incompatible cached plans를 격리한다. canary는 ragged reorder, chunk offset, window/page boundary, prefix policy와 cancellation reuse를 포함한다.

correctness terminal은 support와 request isolation exact parity다. coordinate terminal은 logical-to-physical mapping과 generation 일치다. numerical terminal은 support가 맞은 뒤 LSE/output tolerance다. performance terminal은 logical work 대비 page/tile reads, plan overhead, TTFT/ITL이다. observability terminal은 selected backend와 fallback reason을 확인할 수 있는 것이다.

이 dossier를 열면 독자는 dense triangle을 찾지 못해도 mask를 복원할 수 있다. query absolute coordinate, segment indptr, KV length, prefix/window predicate, page table와 last extent가 kernel operand가 되는 길을 따라가면 된다. 14장에는 position과 cache representation의 수학만 넘기고, visibility predicate의 ownership은 여기서 닫는다.

## 13.14 배포 전 종이 실습과 최종 handoff

마지막 종이 실습은 네 요청을 사용한다. A는 full prefill 5, B는 prefix hit3 뒤 chunk4, C는 decode1 with KV9/window5, D는 prefix-LM total6/prefix2다. page size는4다. 각 요청의 packed query range, absolute positions, KV total, expected support, logical blocks와 last page valid length를 계산한다.

A의 q positions는0..4, KV len5, pages2,last1이다. row q4는 keys0..4를 본다. B의 new q는3..6, total KV7, pages2,last3이다. first chunk row abs3은 keys0..3을 본다. C q9는 full causal이면0..9지만 window5이면5..9, pages3,last2다. D는 prefix q0,1이 keys0,1을 모두 보고 suffix q2..5는 prefix0,1과 causal suffix2..q를 본다.

packed order `[A,B,C,D]`에서 qo lengths는5,4,1,6, `qo_indptr=[0,5,9,10,16]`이다. order를 `[D,C,A,B]`로 바꾸면 `[0,6,7,12,16]`이지만 request별 expected support는 같다. block table rows와 policy metadata도 같은 permutation을 따라야 한다.

### expected support를 표로 고정한다

전체 matrix를 저장할 필요는 없다. 각 boundary row만 적는다. A q0/q4, B q3/q6, C q9, D prefix q0/q1과 suffix q2/q5다. allowed range가 contiguous하지 않은 prefix policy는 set/ranges 두 구간으로 표현한다. window/prefix union도 같은 방식이다.

`support_count`, min/max key만으로 non-contiguous 오류를 놓칠 수 있다. prefix0,1+window5..9와 잘못된 range0..6은 count가 같을 수도 있다. small fixture에서는 exact bitset을 사용한다. 큰 trace에서는 range list와 digest를 둔다.

#### metadata 변환을 손으로 실행한다

각 request에 query_start, q_len, kv_len, window_left, prefix_len/policy, page list, last_page_len을 채운다. wrapper가 query_start를 직접 받지 않는다면 causal alignment가 `kv_len-q_len`으로 복원되는지 확인한다. B는 `7-4=3`, A는0, C는`10-1=9`다. D prefix-LM은 단순 causal flag로 표현되지 않는다.

physical pages에 sentinel을 둔다. logical block b의 value fingerprint는 100+b, physical page는 임의 permutation을 쓴다. block table을 통해 logical order로 읽었을 때 같은 output이 나와야 한다. page table을 identity로 바꾸고 contents를 함께 permutation하는 metamorphic pair를 만든다.

#### failure injection을 source mutation 없이 설계한다

첫 실패는 B query_start를0으로 두는 것이다. second chunk first row support가0만 남는다. 둘째는 C lower bound를 `q-W`로 두어 key4가 추가된다. 셋째는 A last_page_len을4로 두어 stale slots1..3이 들어간다. 넷째는 reorder 뒤 D block table을 old row에 둬 cross-request page를 읽는다.

각 실패는 expected first divergence가 다르다. query_start와 window는 logical support, last length와 reorder는 logical-to-physical page mapping이다. Q/K values가 같다는 전제를 유지한다. fixture가 실제로 prohibited V sentinel에 민감하도록 값을 고른다.

#### source review에서 stop해야 할 위치

metadata producer 값이 이미 틀리면 kernel body까지 내려가지 않는다. wrapper argument가 뒤바뀌면 scheduler를 고치지 않는다. launcher predicate가 맞고 kernel support만 다르면 kernel owner로 간다. support와 page read가 맞고 LSE만 다르면 online reduction/dtype을 본다.

성능도 같은 stop rule을 쓴다. logical visible pairs가 예상보다 크면 policy/metadata, logical은 맞지만 page read가 크면 tiling/fragmentation, kernel range는 짧지만 host gap이 크면 plan/graph/metadata copy를 본다. end-to-end latency 하나로 kernel을 비난하지 않는다.

#### 운영 회귀 artifact

artifact에는 revision, architecture/backend, q/kv lengths, abs starts, indptr, policy, block table digest/generation, last lengths, expected support digest, selected launcher와 result reason을 둔다. user tokens와 raw cache contents는 synthetic fixture를 우선하고 실제 incident는 접근 통제한다.

metric에는 chunk start bucket, q/kv length histogram, window/prefix mode, pages/last utilization, backend/fallback, plan rebuild, support assertion/rejection을 둔다. high-cardinality tables는 trace로 보낸다. metric은 문제 cohort를 찾고 artifact는 좌표를 증명한다.

#### rollout과 rollback 계산

새 metadata convention을 generation G2로 배포할 때 old G1 planner/graph/cache와 섞지 않는다. router는 compatible worker로 보내고 in-flight G1 chunk를 drain한다. cached KV bytes가 호환돼도 query-start/visibility convention이 달라지면 plan은 호환되지 않을 수 있다.

canary에서 correctness는 exact support/metamorphic isolation, numerical은 LSE/output tolerance, performance는 plan+kernel total과 TTFT/ITL을 본다. old generation read가 관측 가능한 최장 lifetime 뒤 0이 되고 두 window 동안 boundary fixture가 통과하면 migration을 닫는다.

rollback은 fast backend를 끄는 즉시 완화와 metadata producer를 G1으로 되돌리는 근본 조치를 구분한다. dense fallback이 같은 predicate를 표현하는지 먼저 확인하고 capacity 비용을 계산한다. correctness를 위해 fallback했지만 overload를 만들 수 있으므로 admission 제한과 함께 운영한다.

#### 다음 장으로 넘기는 정확한 상태

14장에는 request별 logical positions, position transform config, KV representation/layout와 cache generation을 넘긴다. 13장은 어떤 key가 visible한지를 이미 닫았다. RoPE가 Q/K 값을 틀리게 만들면 14장이고, positions는 맞지만 support가 틀리면 13장이다.

handoff 표에는 `request`, `q absolute positions`, `visible logical keys`, `Q/K head identity`, `block/page mapping`, `cache position/layout generation`, `backend descriptor`가 있다. Q/K value fingerprint와 support/page mapping이 각각 PASS인지 구분한다. 하나가 미확인이면 다음 장에서 가정하지 않는다.

독자는 이제 “causal mask가 있다”가 아니라 q=17,W=8에서 keys10..17이 보이고, chunk start와 ragged segment가 이 절대 좌표를 만들며, block table이 해당 logical keys를 page/slot로 옮기고, kernel metadata가 partial page까지 같은 predicate를 보존한다고 말할 수 있다. 이 문장이 13장의 최종 완료 조건이다.

실무에서는 support 전체를 직접 관측하기 어려울 수 있다. 그때는 세 종류의 probe를 조합한다. metadata reconstruction은 source와 trace에서 expected support를 계산한다. prohibited-V metamorphic probe는 금지 위치 V만 크게 바꿔 output 불변성을 본다. allowed-key deletion probe는 허용 경계 key를 제거하거나 zero로 만들어 output이 민감한지 확인한다. probe sensitivity를 증명해야 false negative를 줄인다.

softmax probability를 반환하는 debug backend는 강한 증거지만 execution path와 memory를 바꿀 수 있다. production fused path와 같은 metadata를 소비하는지 확인하고, debug 결과가 reference 역할인지 실제 lane 관측인지 구분한다. debug option을 켰을 때 fallback했다면 그 사실을 artifact에 남긴다.

수치 tolerance는 row별 dynamic range와 length를 반영한다. 아주 긴 row는 reduction 순서 차이가 커질 수 있지만 prohibited mass와 request isolation은 여전히 exact semantic invariant다. LSE가 close해도 wrong V page가 비슷한 값이면 output 오류가 숨을 수 있으므로 page permutation/sentinel fixture를 유지한다.

batch scheduling 변화도 cohort에 넣는다. 같은 요청이 단독일 때와 mixed ragged batch일 때 logical support가 같아야 한다. scheduler가 chunk를 다르게 나눌 수 있으므로 final output만 비교하지 않고 각 step의 abs q와 support union이 동일 전체 causal computation을 이루는지 본다. chunk partition metamorphic test다.

prefix hit 길이가 block boundary 바로 전후일 때를 선택한다. page size16에서 hit15,16,17은 current query start와 first writable slot, shared/private page 경계를 바꾼다. correctness가 16에서만 계단처럼 달라지면 last/shared page와 copy-on-write를 우선한다. hit율 평균으로는 찾기 어렵다.

window boundary도 W-1,W,W+1 context를 사용한다. allowed count가 min(q+1,W)인지 확인한다. API가 `window_left` past count를 받는다면 current token 포함 개수와 변환식을 적는다. framework option W와 kernel argument가 같은 이름이어도 off-by-one convention이 다를 수 있다.

cancellation fixture는 A가 page를 반납한 직후 B가 같은 physical page ID를 할당받도록 설계한다. A의 old block table/plan이 B run과 섞이지 않아야 한다. page incarnation과 request generation을 assertion한다. stale read가 유한한 B data를 읽으면 NaN 없이 cross-request contamination이 발생할 수 있다.

분산 환경에서는 rank별 block table/cache shard가 같은 request permutation generation을 보는지 확인한다. Q head rank와 KV shard/replica가 다른 metadata order를 쓰면 collective 이전 local attention부터 갈린다. rank별 shape equality는 충분하지 않다. request-position-page key로 canonicalize한다.

observability failure도 별 terminal이다. selected backend나 fallback, metadata generation을 알 수 없으면 correctness가 우연히 통과해도 안전한 migration을 증명하기 어렵다. trace coverage와 dropped span, plan cache visibility를 측정한다. 서비스 회복과 telemetry 회복을 분리해 닫는다.

마지막 review는 “왜 이 최적화가 필요한가”를 되묻는다. ragged packing은 pad score/compute를 줄이고, paging은 contiguous 재배치 없이 cache를 관리하며, chunking은 긴 prefill과 decode의 scheduling을 조절하고, window는 visible KV work를 제한한다. 각 이득은 coordinate contract를 정확히 encode할 때만 성립한다. 빠르지만 다른 support를 계산하는 경로는 최적화가 아니다.

따라서 변경 승인 문장은 효과와 위험을 함께 쓴다. “Dense mask materialization을 indptr·length·page metadata로 대체해 score temporary와 padding work를 줄이되, absolute query start, request permutation, partial-page extent와 window/prefix predicate를 boundary fixture로 보존한다.” 이 수준으로 설명해야 source의 설계 의도와 운영 검증이 맞물린다.

새 CUDA/toolkit이나 backend revision으로 이동할 때도 이 문장을 회귀 계약으로 사용한다. wrapper signature와 launcher specialization이 바뀌면 metadata 단위, alignment convention, index dtype, last-page 처리와 supported predicate를 다시 확인한다. release note의 “성능 개선”만으로 semantic 호환성을 선언하지 않는다.

kernel 선택 조건이 넓어졌다면 이전 fallback shape가 fast path에 새로 들어간다. 그 경계 shape를 canary에 추가한다. 특히 q_len>1 decode, non-default window, prefix/custom mask, large page, index dtype과 GQA/MLA 조합을 본다. 새 path의 support exactness가 먼저다.

문제가 생기면 old backend fallback을 보존해 rollback한다. 단 fallback이 prefix/window 정책을 모두 지원하는지 확인하고, capacity 비용을 admission에 반영한다. correctness lane과 performance lane을 동시에 잃지 않도록 배포 manifest에 known-good 조합을 둔다.

최종 보고서에는 확인한 source revision, 손계산 fixture, expected/observed support, first divergent metadata, selected backend, 수정과 두 terminal window를 남긴다. 함수 목록이나 옵션 스크린샷만으로 끝내지 않는다. 다음 담당자가 바로 producer와 consumer edge를 열 수 있어야 한다.

이제 장의 모든 추가 디테일은 하나의 목적을 가진다. 독자가 ragged·prefix·window·chunk·page를 별 기능 목록으로 외우는 대신, 동일한 logical visibility가 서로 다른 metadata 표현을 통과하는 과정을 이해하고 검증하게 하는 것이다.

미확인 항목에는 필요한 runtime probe를 붙인다. source가 지원 가능성을 보여도 실제 selected specialization, tile skip, physical K/V read byte와 numerical envelope는 관측 전까지 확정하지 않는다. 반면 indptr·length·block table의 expected 좌표와 boundary predicate는 정적 계산으로 검산한다.

이 구분을 지키면 성능 수치를 꾸며내지 않으면서도 구현 검토를 깊게 진행할 수 있다. 후속 실행자는 이미 준비된 fixture와 trace schema를 사용해 expected/observed 차이만 채우면 된다. 검증 결과는 다시 같은 coordinate ledger로 돌아와 source claim과 배포 terminal을 갱신한다.
