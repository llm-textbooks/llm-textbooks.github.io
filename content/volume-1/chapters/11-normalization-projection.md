# 11장. normalization과 projection이 하는 일

10장에서 residual row 하나를 따라왔다. 이제 decoder layer는 이 row를 그대로 Q·K·V나 MLP에 넣지 않는다. norm이 크기 기준을 다시 잡고 linear projection이 다른 좌표계로 보낸다. 수식은 짧지만 epsilon, accumulation dtype, weight orientation, fused residual alias, packed output order, TP collective 중 하나가 어긋나면 shape가 정상인 채 모든 뒤 layer가 틀릴 수 있다.

이 장의 질문은 **같은 residual input이 어느 수치 규약으로 정규화되고, 어느 weight와 shard를 지나 어떤 packed tensor가 되는지 어떻게 증명하는가**다. attention score와 head 의미는 12장 이후, MLP activation과 gating 의미는 15장에 맡긴다. 여기서는 그 계산에 들어갈 projection을 정확히 만든다.

## 11.1 LayerNorm과 RMSNorm은 무엇을 같게 만들고 무엇을 남기는가

LayerNorm은 hidden row `x`의 평균 `μ`와 variance를 feature axis에서 계산한다. `y=(x-μ)/sqrt(var+eps)×γ+β`다. RMSNorm은 평균을 빼지 않고 `y=x/sqrt(mean(x²)+eps)×γ`에 가깝다. RMSNorm은 전체 offset을 제거하지 않는다. 둘을 “값을 -1과 1 사이로 만든다”라고 설명하면 틀리다. 범위 제한 함수가 아니라 row 통계로 scale을 정하는 함수다.

여기서 row라는 말부터 고정하자. 서비스가 여러 요청의 토큰을 납작한 token 축으로 합쳤다면 norm이 보는 한 row는 요청 하나가 아니라 토큰 하나의 hidden vector다. 입력이 `[T,H]`일 때 통계 축은 `H`이고, `T`개의 row는 서로 다른 분모를 가진다. 첫 토큰의 큰 값이 둘째 토큰의 scale을 직접 바꾸지 않는다. 반대로 hidden 축과 token 축을 뒤집은 구현은 shape가 우연히 맞더라도 전혀 다른 연산이다. 장애 조사에서는 “어느 축을 줄였는가”를 수식 옆에 반드시 쓴다.

`x=[1,3]`을 손으로 계산하자. LayerNorm의 평균은 2, centered row는 `[-1,1]`, variance는 1이다. epsilon을 무시하고 γ=1, β=0이면 `[-1,1]`이다. RMS는 `sqrt((1+9)/2)=sqrt(5)`이고 결과는 대략 `[0.447,1.342]`다. 평균을 빼지 않았으므로 두 결과는 구조적으로 다르다.

조금 더 현실적인 네 원소 예제를 보자. `x=[-2,0,2,4]`이면 평균은 1이다. LayerNorm의 centered row는 `[-3,-1,1,3]`, 제곱합은 20, population variance는 5다. `eps=10^-5`, `γ=[1,2,1,0.5]`, `β=[0,0.1,0,-0.2]`라면 정규화 전 centered scale은 대략 `[-1.34164,-0.44721,0.44721,1.34164]`다. affine까지 적용한 결과는 `[-1.34164,-0.79442,0.44721,0.47082]`다. 같은 입력의 RMS 제곱 평균은 `(4+0+4+16)/4=6`, inverse RMS는 약 `0.408248`이다. 표준 RMSNorm의 affine 전 결과는 `[-0.81650,0,0.81650,1.63299]`다. 두 norm이 같은 hidden width와 같은 출력 shape를 내더라도 평균 이동을 보존하는 방식이 다르다는 사실이 숫자로 드러난다.

이 차이는 “어느 쪽이 더 좋다”는 단순 순위가 아니다. 학습된 weight는 선택된 norm 규약과 함께 의미를 얻는다. LayerNorm으로 학습한 checkpoint에 RMSNorm을 끼우거나, RMSNorm weight를 LayerNorm의 `γ`처럼 읽으면서 잘못된 bias를 더하면 모델 전체 좌표계가 달라진다. 서빙 엔진은 임의로 norm 종류를 선택하는 것이 아니라 config와 architecture class가 정한 모듈을 재현해야 한다.

shift 불변성도 좋은 직관 검사다. LayerNorm 입력의 모든 원소에 같은 상수 10을 더하면 centered 값은 그대로이므로, rounding을 무시하면 affine 전 출력이 같다. RMSNorm에는 그렇지 않다. `[1,3]`과 `[11,13]`의 방향과 RMS scale은 다르다. 반면 두 입력을 양의 상수 2로 곱하면 epsilon이 무시될 만큼 클 때 RMSNorm 출력은 거의 같다. 정확히 같지 않은 이유는 분모 안의 epsilon이 입력 scale과 함께 곱해지지 않기 때문이다. 이 작은 단서가 tiny row 검증에서 중요하다.

RMSNorm에서 `x=[0,0]`이면 epsilon이 없을 때 0으로 나눈다. epsilon은 장식이 아니라 zero와 tiny row에서 denominator를 지키는 수치 계약이다. `eps=1e-6`과 `1e-5`는 큰 row에서는 차이가 작아도 작은 row에서는 output scale을 바꾼다. config field가 module constructor에 전달되고 actual kernel epsilon으로 쓰이는지 확인한다.

예를 들어 `x=[0.001,-0.001]`의 mean square는 `10^-6`이다. `eps=10^-6`이면 분모는 `sqrt(2×10^-6)≈0.0014142`여서 결과 절댓값은 약 `0.7071`이다. `eps=10^-5`이면 분모는 `sqrt(11×10^-6)≈0.0033166`이고 결과 절댓값은 약 `0.3015`다. epsilon을 열 배 바꾼 것이 출력 마지막 자리만 건드린 게 아니다. 작은 row에서는 scale을 두 배 이상 바꿨다. 따라서 differential tolerance를 정할 때 “eps 차이는 미미하다”라고 미리 결론 내리면 안 된다.

또 하나 확인할 것은 variance 정의다. LayerNorm은 이 문맥에서 hidden width로 나누는 population variance를 쓴다. 통계 패키지의 unbiased sample variance처럼 `H-1`로 나누지 않는다. `x=[1,3]`에서 전자는 1, 후자는 2다. 작은 fixture에서는 이 차이가 즉시 보이지만 큰 hidden size에서는 상대 차이가 작아져 품질 저하로 숨어들 수 있다. 손계산 oracle은 구현이 실제 사용하는 정의와 일치해야 한다.

Norm weight의 규약도 모델마다 같지 않다. 보통 RMSNorm은 계산한 normalized row에 저장 weight를 그대로 곱한다. Gemma 계열 구현은 저장된 parameter에 1을 더한 scale을 사용할 수 있다. 저장 weight가 모두 0인 fixture를 생각하면 표준 방식은 모든 출력을 0으로 만들지만 `1+weight` 방식은 normalized row를 그대로 통과시킨다. 이 fixture는 checkpoint loader, Python reference, custom kernel 사이의 shift 중복을 찾는 데 아주 강하다.

Transformers Qwen3.5 RMSNorm은 fp32로 variance를 계산하고 원 dtype으로 되돌린 뒤 weight를 적용한다. [Transformers v5.15.1 `modeling_qwen3_5.py:720-738`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L720-L738)에서 cast 순서를 읽는다.

Gemma3의 `1+weight` scale 규약은 [Transformers v5.15.1 `modeling_gemma3.py:136-155`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L136-L155)에 고정한다.

이 소스를 읽을 때 class 이름만 베끼지 않는다. 입력 dtype 저장, fp32 cast, `pow(2)`, 마지막 축 mean, `rsqrt`, 원 dtype 복귀, weight 적용의 순서를 한 줄씩 옮긴다. 특히 weight 곱이 cast 전인지 후인지 기록한다. 두 구현이 수학적으로 같은 식을 표방하더라도 finite precision에서 순서가 다르면 bitwise equality는 깨질 수 있다. correctness 기준을 bitwise, absolute/relative tolerance, selected-logit agreement 중 무엇으로 둘지 먼저 정해야 한다.

독자가 이 절을 빠져나갈 때 답할 수 있어야 하는 질문은 네 가지다. 첫째, norm이 줄이는 축은 무엇인가. 둘째, 평균을 빼는가. 셋째, epsilon은 어디에 더해지는가. 넷째, learned scale이 `w`인가 `1+w`인가. 이 네 답이 없으면 “RMSNorm을 쓴다”는 정보는 장애 분석에 충분하지 않다.

## 11.2 FP32 accumulation과 fused add-norm은 lifetime 계약이다

bf16 residual의 제곱합을 bf16로 누적하면 rounding과 overflow 여유가 fp32보다 나쁘다. reference가 fp32 accumulation을 쓰는 이유다. 그러나 output을 언제 activation dtype으로 cast하고 γ를 어느 dtype에서 곱하는지에 따라 마지막 bit가 달라질 수 있다. “FP32 norm”이라는 말만으로 충분하지 않다.

bf16은 지수 범위는 넓지만 유효 자릿수가 짧다. 큰 값과 작은 값의 제곱을 같은 accumulator에 더할 때 작은 기여가 사라질 수 있다. 예를 들어 제곱값이 `[1, 1/256, 1/256, ...]`처럼 구성되면 낮은 정밀도 누적 순서에 따라 작은 항 일부가 round away될 수 있다. parallel reduction은 합산 트리도 바꾼다. fp32로 승격하는 목적은 수학식을 바꾸려는 것이 아니라 제곱과 reduction의 오차를 줄이려는 것이다.

그렇다고 fp32 승격이 모든 차이를 없애는 것은 아니다. 입력은 이미 bf16로 양자화되어 있을 수 있다. custom kernel은 각 lane에서 fp32 partial을 만들고 warp 또는 block reduction으로 합친다. reduction 순서가 reference의 순차 합과 다를 수 있다. inverse square root 구현도 정확한 `sqrt`와 reciprocal 조합인지 근사 intrinsic인지에 따라 다르다. 마지막으로 normalized fp32를 bf16로 내린 뒤 weight를 곱하는지, fp32 weight multiplication 후 내리는지도 결과를 바꾼다. 따라서 norm differential에는 중간 통계를 함께 저장한다. `sum_sq`, `mean_sq`, `inv_rms`, affine 전 row, affine 후 row가 최소 경계다.

overflow만 보지 말고 underflow와 비정상 입력도 분리한다. 입력에 하나라도 NaN이 있으면 제곱합과 출력 전체 row가 NaN으로 퍼지는 것이 자연스럽다. `+inf`와 `-inf`가 들어오면 `inf × 0` 형태가 생겨 NaN이 될 수 있다. 반면 모든 값이 유한한데 norm 직후 처음 NaN이 나타났다면 이전 residual의 max absolute, accumulation dtype, epsilon 전달, kernel launch shape를 확인한다. norm은 흔히 NaN을 발견하는 첫 장소이지 반드시 만든 장소는 아니다.

pre-norm layer에서 residual add와 norm은 연속한다. fused add-norm은 `residual = residual + branch`, `normalized = norm(residual)`을 한 kernel에서 수행해 memory traffic을 줄일 수 있다. 이때 반환 tuple이 `(normalized,residual)`인지, input buffer를 in-place로 갱신하는지, residual alias가 다음 branch까지 살아 있는지 확인해야 한다.

두 tensor의 의미를 시간순으로 그려 보자. layer 입구 residual을 `r0`, attention branch 출력을 `a`, 합을 `r1=r0+a`, MLP에 들어갈 normalized tensor를 `n1=norm(r1)`이라고 하자. unfused reference는 `r1`과 `n1`을 별도 allocation에 둘 수 있다. fused 구현은 `r0` buffer를 `r1`으로 덮고 별도 `n1`을 돌려줄 수도 있고, caller가 제공한 output buffer와 residual buffer를 함께 쓸 수도 있다. 어느 선택도 수학만 보면 같다. 그러나 `r0`를 나중에 로그하거나 다른 branch가 참조한다면 lifetime은 같지 않다.

alias 표는 포인터 동일성만 기록하지 않는다. 논리 이름, storage identity, write 시점, 마지막 reader를 함께 적는다.

| 단계 | 논리 값 | 허용되는 storage | 쓰기 이후 reader |
|---|---|---|---|
| layer 입구 | `r0` | residual buffer A | attention norm 또는 audit probe |
| branch 완료 | `a` | branch buffer B | add-norm |
| fused add 뒤 | `r1` | A를 덮거나 새 C | MLP norm, 다음 residual add |
| norm 출력 | `n1` | B 재사용 또는 새 D | MLP projection |

여기서 B 재사용은 attention output을 더 이상 읽지 않는다는 liveness 증명이 있을 때만 안전하다. graph capture나 asynchronous 실행에서는 Python 문장의 순서만 보고 lifetime을 판단할 수 없다. 실행 graph의 dependency와 kernel stream ordering이 실제 소유권을 정한다.

fused op의 이득은 대개 arithmetic 감소가 아니라 memory traffic과 launch 감소에서 나온다. unfused 경로는 residual 두 개를 읽고 합을 쓰고, 그 합을 다시 읽어 norm 통계를 만들고, norm 결과를 쓴다. fusion은 합 값을 register에 둔 채 제곱합에 기여시키고 최종 residual과 normalized output을 쓸 수 있다. 하지만 reduction 때문에 모든 값을 한 번에 영원히 register에 보관할 수 있는 것은 아니다. 구체적인 kernel이 몇 번 읽고 쓰는지는 구현을 봐야 한다. “fused”라는 이름만으로 byte 수를 단정하지 않는다.

vLLM RMSNorm custom op의 native와 fused residual 경계는 [vLLM v0.27.1 `layernorm.py:37-130`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/layernorm.py#L37-L130), Gemma scale 차이는 [같은 파일 `:132-170`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/layernorm.py#L132-L170)에서 읽는다. source가 fused path를 제공한다는 사실과 현재 backend가 선택했다는 사실은 다르다.

이 파일에서는 `forward_native`와 custom op 경로의 인자 및 반환을 비교한다. residual이 `None`인 경우와 제공된 경우가 서로 다른 계약을 가질 수 있다. caller도 함께 읽어야 한다. 모델 layer가 첫 norm에서 `(hidden_states, residual)`을 받고, 다음 norm에 둘을 다시 넘기는지 확인한다. API 정의만 보고 tuple 순서를 추측하면 normalized tensor를 residual로 보존하는 치명적인 오독이 생긴다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- llama.cpp CUDA norm 구현은 더 낮은 층의 조건을 보여 준다.
- [llama.cpp v0.2.0 `norm.cu:77-157`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/norm.cu#L77-L157)에서 row reduction과 scale 적용을 보고, [같은 파일 `:478-560`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/norm.cu#L478-L560)에서 graph tensor의 extent·byte stride·epsilon이 launcher 인자로 변환되는 경계를 본다.
- fused add variant는 [같은 파일 `:562-645`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/norm.cu#L562-L645)에 있다.
- 이 장에서는 실행하지 않지만, contiguous first dimension과 source type assertion이 fusion eligibility에 미치는 의미는 source로 증명할 수 있다.

운영에서 자주 만나는 오판은 “fused 옵션을 켰으니 fused kernel이 실행됐다”는 결론이다. 실제 선택에는 dtype, shape, stride, backend 지원, graph pattern, compile flag가 관여한다. 요청된 상태, eligibility 판정, 선택된 op, 실제 symbol을 네 칸으로 분리한다. 이 가운데 앞의 두 칸만 config와 Python source에서 확인했다면 kernel 실행을 주장하지 않는다.

fused와 unfused의 정확성 비교도 최종 logits 하나만 보면 부족하다. 첫 checkpoint는 add 직후 residual, 둘째는 affine 전 normalized row, 셋째는 affine 후 output이다. residual부터 다르면 alias/add ordering 문제, residual은 같고 inverse RMS부터 다르면 reduction·epsilon 문제, affine 전은 같고 후만 다르면 weight convention·dtype 문제다. 이 세 갈래가 조사 범위를 크게 줄인다.

## 11.3 linear projection은 weight orientation과 packed order의 계약이다

input `X`가 `[T,H_in]`, mathematical weight `W`가 `[H_in,H_out]`이면 `Y=XW`는 `[T,H_out]`이다. PyTorch `nn.Linear`는 weight를 `[H_out,H_in]`으로 저장하고 `X W^T+b`를 계산한다. checkpoint shape만 보고 transpose를 추가하면 이미 맞는 orientation을 뒤집을 수 있다.

orientation은 표기 취향이 아니라 loader와 kernel 사이의 ABI다. PyTorch 문서식 저장 배열에서 `weight[o,i]`는 input feature `i`가 output feature `o`에 기여하는 계수다. 반면 ggml tensor의 차원 표시와 multiplication helper는 다른 순서를 노출할 수 있다. 파일에 저장된 shape, graph가 해석한 logical axes, kernel operand layout을 별도로 기록해야 한다. 같은 `[4096,4096]` 정사각형 weight는 transpose 오류가 shape 검사로 절대 드러나지 않는다.

`x=[1,2]`, stored weight `[[1,0],[0,1],[1,1]]`이면 output은 `[1,2,3]`이다. stored rows가 output features다. bias `[0.5,0,-1]`를 더하면 `[1.5,2,2]`다. quantized weight라면 packed integer와 scale을 dequantize한 effective row가 이 식을 만족해야 한다.

정사각형이 아닌 fixture를 쓰는 이유가 여기에 있다. `H_in=2`, `H_out=3`이면 잘못된 transpose는 곱셈 자체가 성립하지 않아 빨리 실패한다. 각 output row에 식별 가능한 패턴을 넣으면 order도 확인할 수 있다. 첫 row `[1,0]`, 둘째 `[0,10]`, 셋째 `[100,100]`을 두고 `x=[2,3]`을 넣으면 `[2,30,500]`이다. 단순 증가 패턴보다 자릿수가 분리된 패턴이 slice 교환을 눈으로 찾기 쉽다.

bias는 “있다/없다”뿐 아니라 shard와 accumulation 시점을 묻는다. column-parallel에서는 각 rank가 자신의 output slice에 해당하는 bias slice를 더할 수 있다. row-parallel에서는 모든 rank가 full output bias를 더한 뒤 all-reduce하면 bias가 TP 크기만큼 중복된다. 따라서 bias는 보통 reduction 뒤 한 번 더하거나 특정 rank의 partial에만 더하는 등의 규약이 필요하다. 실제 class의 `skip_bias_add`, `reduce_results`, caller-side fusion을 읽어야 한다.

activation dtype과 weight storage dtype도 구분한다. checkpoint가 fp16이나 bf16일 수 있고, quantized weight는 int4 payload와 group scale·zero point를 가질 수 있다. GEMM accumulator는 fp32 또는 backend-specific 정밀도를 쓸 수 있으며 output은 다시 activation dtype으로 내려갈 수 있다. “int4 linear”는 input과 output이 int4라는 뜻이 아니다. effective 계산을 다음처럼 장부화한다.

```text
packed_weight bytes
  --unpack/group metadata--> integer codes
  --scale/zero rule--------> effective weight values
  --matmul with activation-> accumulator
  --bias/output cast-------> projected activation
```

group-wise scale shape를 예로 들자. stored logical weight가 `[6,4]`이고 input-axis group size가 2라면 output row마다 두 scale을 둘 수 있어 scale shape가 `[6,2]`가 된다. output-axis grouping이나 block quantization이면 shape는 달라진다. scale 하나를 output row 전체에 broadcast한다고 가정하면 dequantized 값은 shape가 맞아도 틀릴 수 있다. quantization 장이 format을 자세히 다루지만, 이 장에서는 projection 경계에서 effective row와 broadcast axis를 반드시 확인한다.

QKV projection은 Q, K, V matrices를 한 output axis에 pack할 수 있다. gate와 up도 하나로 pack할 수 있다. loader가 checkpoint의 `q_proj`, `k_proj`, `v_proj`를 어느 slice 순서에 넣었는지와 forward split 순서가 같아야 한다. shape가 같아도 Q와 K가 교환되면 silent failure다.

GQA 모델에서는 Q·K·V 폭이 같지 않을 수 있다. `n_q=8`, `n_kv=2`, `d=64`라면 Q width는 512, K와 V width는 각각 128이다. packed output의 총 폭은 768이고 offsets는 `[0,512)`, `[512,640)`, `[640,768)`이다. 이를 단순히 세 등분하면 각 slice가 256이 되어 shape가 후속 reshape에서 깨지거나, 더 위험하게 별도 padding 때문에 shape가 맞은 채 의미만 섞일 수 있다. offsets는 head count와 head dimension에서 유도해야 한다.

TP가 더해지면 global order와 rank-local order가 모두 필요하다. TP=2이고 query heads가 균등 분할되면 rank당 Q width는 256이다. KV heads도 rank당 하나씩이면 K·V는 각각 64다. local packed width는 384다. 그러나 KV head 수가 TP보다 작으면 replication 규칙이 필요하다. global checkpoint의 연속 slice를 단순히 rank로 나누는 것만으로는 충분하지 않다. 12장에서 head mapping을 자세히 다루지만 이 장에서는 loader의 shard identifier가 `q`, `k`, `v` 중 무엇이며 local offset을 어떻게 계산하는지까지 확인한다.

gate-up도 두 등분한다고 무조건 안전하지 않다. architecture가 intermediate width 두 개를 같은 크기로 쓰는지 config에서 확인한다. loader는 `gate_proj`를 packed slot 0, `up_proj`를 slot 1에 넣고 forward는 같은 순서로 split해야 한다. 둘을 바꾸면 SwiGLU의 `silu(gate) * up`이 `silu(up) * gate`가 된다. 곱셈은 교환법칙이 있어 보이지만 SiLU가 한쪽에만 적용되므로 결과는 같지 않다.

vLLM merged column/QKV linear construction과 loading 경계는 [vLLM v0.27.1 `linear.py:661-760`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/linear.py#L661-L760), [같은 파일 `:1022-1115`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/linear.py#L1022-L1115)에서 읽는다.

첫 범위에서는 output partition size 목록과 parameter allocation을 본다. 둘째 범위에서는 QKV의 head 수, replication, shard offset을 본다. weight loader가 checkpoint tensor를 받은 뒤 `shard_id`를 어떤 숫자나 문자열로 해석하는지도 추적한다. model file의 stacked parameter mapping은 이름을 바꾸고 shard identifier를 넘기며, linear class는 실제 destination slice를 고른다. 이 두 단계를 한쪽만 읽으면 pack order를 증명하지 못한다.

SGLang에도 이름이 비슷한 linear class가 있다고 해서 vLLM과 구현이 같다고 가정하지 않는다. [SGLang v0.5.18 `linear.py:590-760`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/linear.py#L590-L760)에서 merged output partition과 loader를 확인하고, [같은 파일 `:1030-1185`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/linear.py#L1030-L1185)에서 QKV-specific partition을 확인한다.

정확한 class 범위가 revision에서 달라질 수 있으므로 링크의 symbol과 실제 선언을 함께 기록한다.

Transformers reference는 packed serving module의 oracle이라기보다 separate projection의 의미 oracle이다. [Transformers v5.15.1 `modeling_qwen3_5.py:806-878`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L806-L878)에서 q/k/v linear와 reshape 경계를 읽는다. serving implementation의 packed slice를 각각 이 separate output과 비교하면 pack과 model semantics를 분리할 수 있다.

## 11.4 tensor parallel은 output과 input 축을 서로 다르게 나눈다

column-parallel linear는 output feature axis를 rank별로 나눈다. 각 rank가 full input과 weight column shard로 local output slice를 만든다. 다음 연산이 그 shard를 소비할 수 있으면 즉시 gather하지 않는다. row-parallel linear는 input feature와 weight input axis를 나누고 local partial output을 만든 뒤 all-reduce로 합칠 수 있다.

이름은 수학적 weight `W[H_in,H_out]`의 column과 row에서 왔다고 생각하면 쉽다. 다만 저장 weight가 `[H_out,H_in]`이면 배열의 눈에 보이는 축은 반대로 느껴질 수 있다. column-parallel은 logical output 축을 나누고, row-parallel은 logical input 축을 나눈다고 쓰는 편이 덜 혼란스럽다.

`H_in=4`, `H_out=6`, TP=2라면 column shard는 rank당 output 3개다. row-parallel은 rank당 input 2개를 받고 같은 output width의 partial을 만든다. shard offset 오류는 collective가 성공해도 숫자를 틀리게 한다. global checkpoint slice, local weight shape, input partition, collective 전후 output을 ledger로 둔다.

완전한 숫자 예제를 만들자. input `x=[1,2,3,4]`이고 logical output 두 개를 만드는 stored weight가 `w0=[1,1,1,1]`, `w1=[1,10,100,1000]`이라 하자. full output은 `[10,4321]`이다. row-parallel TP=2에서 rank 0은 input `[1,2]`와 각 weight 앞 절반으로 partial `[3,21]`을 만든다. rank 1은 `[3,4]`와 뒤 절반으로 `[7,4300]`을 만든다. elementwise sum all-reduce 뒤 `[10,4321]`을 얻는다. rank 1의 input shard가 잘못 `[1,2]`를 받으면 collective는 정상 완료되지만 `[6,2031]`이라는 틀린 결과가 나온다.

column-parallel 예제에서는 stored output rows 여섯 개를 rank 0이 0~2, rank 1이 3~5로 갖는다. 각 rank는 full `x`를 본다. attention의 QKV처럼 다음 연산도 head shard별로 수행된다면 local output을 gather하지 않고 유지할 수 있다. 반대로 모든 output feature를 요구하는 소비자에게 넘어갈 때 gather가 필요하다. `gather_output` 옵션은 단순 성능 토글이 아니라 반환 tensor의 의미와 shape를 바꾸는 상태다.

row-parallel의 `input_is_parallel`도 마찬가지다. true라면 caller가 이미 input feature를 나누었다는 뜻이다. false라면 linear가 local rank slice를 취해야 한다. caller와 callee가 모두 split하면 폭이 한 번 더 줄고, 둘 다 split하지 않으면 각 rank가 잘못된 full input과 local weight를 곱하려 한다. 옵션 이름과 실제 tensor shape를 함께 확인한다.

collective 경계를 residual add와 연결해 보자. attention output projection은 흔히 row-parallel이다. 각 rank가 자신의 head slice를 model width로 투영해 partial을 만들고, 합쳐진 결과가 residual branch가 된다. residual add가 all-reduce 전에 각 rank에서 수행되면 residual이 reduction에 여러 번 포함될 위험이 있다. 구현은 reduction 결과를 받은 뒤 residual을 더하거나, reduction 규약 안에서 residual이 정확히 한 번만 기여하도록 설계해야 한다. “O projection 뒤 residual”이라는 순서를 tensor와 collective 단위로 풀어 써야 한다.

MLP에서도 column-parallel gate-up 뒤 activation과 elementwise product를 rank-local로 수행하고, row-parallel down projection이 partial을 합치는 구성이 자연스럽다. 이렇게 하면 intermediate activation 전체를 gather하지 않는다. TP가 빠르게 만드는 이유를 “GPU를 여러 개 쓴다”로 설명하면 부족하다. 어느 넓은 intermediate tensor를 분산 상태로 유지하고, 어느 좁은 model-width 경계에서 collective하는지가 핵심이다.

vLLM column과 row classes는 [vLLM v0.27.1 `linear.py:419-620`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/linear.py#L419-L620), [같은 파일 `:1613-1735`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/linear.py#L1613-L1735)에 있다.

SGLang 대응 class는 [SGLang v0.5.18 `linear.py:302-507`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/linear.py#L302-L507), [같은 파일 `:1407-1510`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/linear.py#L1407-L1510)에서 비교한다.

소스를 읽는 순서는 constructor→parameter allocation→weight loader→forward→collective helper다. constructor만 보면 shard shape는 알 수 있지만 runtime input이 이미 partitioned인지 알 수 없다. forward만 보면 collective 여부는 보이지만 checkpoint의 어느 slice가 local parameter에 들어왔는지 모른다. 최소 다섯 경계를 이어야 global 수식과 local 실행을 대응시킬 수 있다.

TP 장애의 증상도 분류한다. 단일 rank는 맞고 TP에서만 틀리면 먼저 divisibility, shard offsets, replication, collective ordering을 본다. TP=2는 맞고 TP=4에서만 틀리면 KV head replication이나 pack alignment처럼 특정 분할 수에서 갈라지는 규칙을 의심한다. 값은 맞지만 느리다면 불필요한 gather, collective 중복, 작은 GEMM fragmentation을 본다. OOM이면 rank-local weight만 줄었는지, full output gather와 temporary가 남는지 분리한다.

collective 자체가 성공했다는 로그는 correctness 증거가 아니다. 올바른 process group, 올바른 rank order, 올바른 element 수를 reduce했는지 확인해야 한다. source inspection 단계에서는 group을 만드는 owner와 linear가 호출하는 helper를 연결한다. runtime을 허용하는 후속 워크북에서는 local partial의 작은 slice와 collective 후 slice를 저장한다. 이 장에서는 그 관찰 필드만 정의하고 실행 결과를 만들지 않는다.

## 11.5 Qwen과 Gemma에서 같은 질문을 던진다

이제 공통 수식을 실제 architecture에 꽂아 보자. 목적은 모델 이름을 외우는 것이 아니다. 처음 보는 구현에서도 `config→module construction→forward call→weight loading→backend op`의 다섯 칸을 채우는 습관을 만드는 것이다. Qwen과 Gemma를 나란히 두면 같은 RMSNorm이라는 이름 아래에도 scale 규약과 layer composition이 다를 수 있음을 분명히 볼 수 있다.

Qwen3.5 vLLM layer는 Gemma-style RMSNorm을 input/post-attention에 배치한다. [vLLM v0.27.1 `qwen3_5.py:170-190`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5.py#L170-L190)을 보면 config epsilon이 module로 들어간다. QKV와 gate-up checkpoint mapping은 [같은 파일 `:288-307`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5.py#L288-L307)에 있다.

여기서 “Qwen3.5가 Gemma-style norm을 쓴다”는 문장은 source의 실제 class 선택과 weight 규약을 확인한 뒤에만 쓴다. 이름에 Gemma가 들어 있다고 architecture 전체가 Gemma라는 뜻은 아니다. custom norm class가 `1+weight`를 적용하는지, constructor flag가 이를 바꾸는지, checkpoint loader가 weight를 미리 shift하는지 세 장소를 연결한다. shift는 정확히 한 번이어야 한다.

Qwen 계열 layer의 forward를 읽을 때 residual 변수의 상태 전이를 먼저 적는다. 첫 layer에서는 residual이 아직 없을 수 있고 input norm이 hidden state를 정규화하면서 residual 소유권을 만든다. 이후에는 이전 branch와 residual을 fused add-norm에 넘길 수 있다. attention output 뒤 post-attention norm도 같은 tuple 계약을 반복한다. 함수가 반환하는 첫 항과 둘째 항의 의미를 이름만으로 추측하지 말고 실제 assignment의 좌변을 따른다.

그다음 projection construction을 본다. attention 모듈이 hidden size, total query heads, total KV heads, head dimension, bias flag, quantization config, prefix를 QKV parallel linear에 넘긴다. 이 인자들은 local packed width와 loader shard를 결정한다. MLP의 merged gate-up은 hidden size에서 두 intermediate branches로 나가며, down projection은 다시 model width로 돌아온다. architecture config에서 읽은 전역 숫자가 TP class의 지역 숫자로 바뀌는 지점을 표시한다.

checkpoint mapping은 문자열 치환 이상의 역할을 한다. 별도 `q_proj.weight`, `k_proj.weight`, `v_proj.weight` 이름을 packed destination 이름으로 바꾸고 각각 `q`, `k`, `v` shard identifier를 함께 넘긴다. gate와 up도 같은 방식이다. destination parameter의 loader가 identifier에 따라 local slice를 고른다. 이름 치환만 맞고 identifier가 틀리면 세 weight가 같은 destination slice를 덮어쓸 수 있다.

Gemma의 norm weight가 `(1+w)`인지 `w`인지 차이는 checkpoint value 해석을 바꾼다. weight가 0일 때 standard multiplicative scale은 output 0이지만 Gemma-style은 identity scale 1이다. loader와 custom kernel이 이 shift를 두 번 또는 전혀 적용하지 않는지 raw norm output과 weight 적용 뒤 slice를 비교한다.

Gemma3 reference layer의 norm 배치는 [Transformers v5.15.1 `modeling_gemma3.py:500-575`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L500-L575)에서 확인한다. revision의 실제 class와 줄 범위를 열어 input/post-attention뿐 아니라 architecture가 추가 norm을 두는지 확인한다. 모델마다 “pre-norm decoder”라는 한 문장으로는 충분하지 않다. branch 전·후 어느 위치에서 어떤 parameter를 쓰는지 layer diagram에 표시해야 한다.

Gemma의 projection에도 bias와 attention variant 같은 모델별 규약이 있을 수 있다. [Transformers v5.15.1 `modeling_gemma3.py:250-360`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L250-L360)에서 q/k/v/o 선언과 forward를 함께 읽는다. 이 reference의 separate linear output을 serving stack의 packed QKV oracle로 사용하되, attention backend 내부 layout까지 같다고 가정하지 않는다.

Qwen과 Gemma 비교표는 feature 목록이 아니라 질문 목록이어야 한다.

| 질문 | Qwen 조사 필드 | Gemma 조사 필드 | 틀렸을 때 첫 증상 |
|---|---|---|---|
| norm scale은 `w`인가 `1+w`인가 | selected class와 loader | reference class와 serving class | 첫 norm부터 값 불일치 |
| epsilon source는 어디인가 | config field→constructor | config field→constructor | tiny row에서 큰 차이 |
| QKV 폭은 어떻게 유도되는가 | q/KV heads와 head dim | q/KV heads와 head dim | split·reshape 실패 또는 silent mix |
| bias가 있는가 | projection flag | projection flag | constant offset 또는 TP 중복 |
| gate-up order는 무엇인가 | stacked mapping과 split | model MLP order | MLP에서 첫 불일치 |

이 표의 장점은 새 모델에도 그대로 쓸 수 있다는 점이다. 이름이 달라도 norm 수치 규약, projection logical axes, pack order, shard rule, collective boundary는 반드시 존재한다.

SGLang Qwen3.5 norm과 QKV construction은 [SGLang v0.5.18 `qwen3_5.py:984-1010`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_5.py#L984-L1010), [같은 파일 `:1045-1105`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_5.py#L1045-L1105)에서 actual fused/fallback 분기를 따라간다.

SGLang source walk도 같은 다섯 칸을 쓴다. model constructor가 어떤 norm/linear class를 고르는지, layer forward가 residual을 어떻게 넘기는지, load_weights가 packed parameter를 어떻게 찾는지, linear class가 TP/quant method를 어떻게 위임하는지, backend op가 어떤 dtype·stride 조건을 요구하는지 잇는다. vLLM과 같은 class 이름이 보여도 반환 tuple이나 loader hook까지 같다는 뜻은 아니다.

Transformers walk는 reference semantics를 고정한다. Qwen RMSNorm의 fp32 승격과 separate q/k/v를 먼저 기록한다. vLLM walk는 fused residual, packed parameter, TP collective를 추가한다. SGLang walk는 자신의 fused/fallback과 loader를 추가한다. llama.cpp walk는 graph node와 physical tensor layout을 추가한다. 이렇게 쌓으면 네 구현을 억지로 동일한 함수명 표에 넣지 않고 같은 의미 좌표에서 비교할 수 있다.

llama.cpp에서는 architecture builder가 norm op, norm weight multiply, matrix multiplication을 어떤 graph 순서로 만드는지 본다. [llama.cpp v0.2.0 `llama-graph.cpp:1480-1560`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L1480-L1560)의 helper와 [같은 파일 `:2290-2400`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L2290-L2400)의 builder 호출을 연결한다.

CUDA backend가 graph pattern을 fuse할 수 있다는 사실과 builder가 semantic nodes를 만들었다는 사실은 동시에 참일 수 있다.

네 스택 differential의 기준점은 최종 token이 아니다. 같은 token ID와 같은 checkpoint에서 첫 layer input row, norm affine 전·후, separate 또는 unpacked Q/K/V slice, O projection collective 후, gate/up slice, down projection 후를 비교한다. reference에 packed tensor가 없다면 serving packed tensor를 slice한 뒤 비교한다. 값이 처음 갈라지는 경계가 owner를 정한다.

소스 검토만으로도 찾을 수 있는 결함이 많다. config epsilon이 module에 전달되지 않는 경우, stacked mapping order와 split order가 다른 경우, bias가 TP reduction 전에 모든 rank에 더해지는 경우, local KV head 수의 replication 계산이 어긋나는 경우다. 반면 실제 backend 선택이나 numeric tolerance는 실행 증거가 필요하다. 이 책의 비실행 분석은 전자를 증명하고 후자의 관찰 설계를 준비하는 데 선을 긋는다.

## 11.6 shape·stride ledger로 packed projection을 검증한다

각 checkpoint에 residual shape/stride/dtype, norm epsilon과 accumulation dtype, normalized output alias, weight logical/stored shape, quant scale shape, packed slice order, TP shard range, local/global output을 둔다. contiguous 여부는 shape에 나타나지 않으므로 stride를 별도 기록한다.

shape ledger를 추상적인 체크리스트로 남겨 두면 실제 장애에서 쓰지 못한다. `T=5`, `H=8`, query heads 4, KV heads 2, head dim 2, intermediate width 12, TP=2인 작은 모델을 끝까지 적어 보자. residual은 global·local 모두 `[5,8]`이다. RMSNorm weight는 `[8]`, normalized output도 `[5,8]`이다. global Q width는 8, K와 V는 각각 4이므로 packed QKV는 `[5,16]`이다. TP rank마다 Q width 4, K width 2, V width 2를 가져 local packed output은 `[5,8]`이다.

gate-up의 global output은 gate 12와 up 12를 합쳐 `[5,24]`다. TP rank는 각각 6+6을 가져 `[5,12]`다. activation과 product 뒤 local intermediate는 `[5,6]`이다. row-parallel down projection은 local input `[5,6]`과 logical weight input shard를 곱해 partial `[5,8]`을 만든다. all-reduce 뒤 각 rank가 같은 `[5,8]` branch 결과를 갖는다. 이 한 장부가 pack과 TP를 동시에 보여 준다.

stride는 element 단위와 byte 단위를 구분한다. contiguous fp16 `[5,8]`의 PyTorch element stride는 보통 `(8,1)`, byte stride는 `(16,2)`다. transpose view는 shape `[8,5]`, element stride `(1,8)`일 수 있다. `view`로 다시 `[5,8]`처럼 보이게 만들 수 없는 layout도 있고, reshape가 암묵적 copy를 만들 수 있다. kernel ABI가 contiguous last dimension을 요구한다면 shape만 맞는 transpose view는 fallback 또는 잘못된 address 계산의 원인이 된다.

packed slice도 view인지 copy인지 기록한다. `[T,Q+K+V]`의 연속 output 축을 `split`하면 각 slice의 token stride는 여전히 전체 packed width일 수 있다. Q slice 자체는 last dimension이 contiguous지만 row 사이 간격에는 K·V가 끼어 있다. backend가 Q를 완전 contiguous `[T,Q]`로 요구하면 copy가 생길 수 있다. head reshape·transpose 뒤 stride는 더 복잡해진다. 12장에서 head layout을 다루기 전에 이 장에서 projection output의 base storage와 offsets를 고정한다.

실제 장부는 다음 열을 갖는다.

| checkpoint | logical shape | stored shape | element stride | dtype | storage/alias | global slice | rank-local slice | consumer |
|---|---|---|---|---|---|---|---|---|
| layer input | `[T,H]` | `[T,H]` | 측정값 | activation | residual A | 전체 | replicated | input norm |
| norm output | `[T,H]` | `[T,H]` | 측정값 | activation | output B | 전체 | replicated | QKV·gate-up |
| QKV weight | `[H,Q+K+V]` | backend 규약 | block stride | storage | parameter | q/k/v offsets | loader 계산 | GEMM |
| packed QKV | `[T,Q+K+V]` | local packed | 측정값 | activation | output C | global order | local offsets | split/head reshape |
| quant scale | format 규약 | format 규약 | 측정값 | scale dtype | parameter metadata | group axes | local groups | dequant/GEMM |

“측정값”이라고 쓴 칸은 실행 전에는 source가 보장하는 invariant와 관찰 예정 값을 나눈다. source가 contiguous allocation을 만든다고 증명할 수 있으면 expected에 적고, 실제 pointer/stride는 runtime evidence 칸을 비워 둔다. 실행하지 않은 값을 결과처럼 꾸미지 않는 것이 증거 장부의 기본이다.

packed QKV가 `[T,Q+K+V]`이면 split offsets를 config head counts와 head dim에서 계산한다. gate-up은 보통 동등 width 두 slice지만 model-specific config를 확인한다. loader mapping과 forward split이 같은 offsets를 쓰는지 정적 fixture로 검산한다.

정적 fixture는 거대한 실제 checkpoint가 없어도 된다. 각 source tensor의 logical row에 서로 다른 표식을 부여한다. q는 100대, k는 200대, v는 300대, gate는 400대, up은 500대 숫자라고 생각한다. loader code의 destination offset 식에 이를 대입해 packed 배열을 종이에 쓴다. forward의 split offsets로 다시 꺼냈을 때 원래 그룹이 복원되는지 확인한다. 이 검산은 GPU도 서버도 필요 없다.

quantized packed weight에서는 payload와 metadata가 함께 이동해야 한다. q payload를 Q slice에 넣고 scale을 K slice에 넣는 식의 독립 offset 버그가 가능하다. parameter subclass가 weight loader를 override하는지, quant method가 scale parameter의 output partition을 어떻게 등록하는지 확인한다. checkpoint 이름 mapping만 봐서는 metadata pack을 증명할 수 없다.

padding도 장부에 넣는다. backend alignment 때문에 logical width 130을 physical width 136으로 둘 수 있다. logical split offsets와 physical storage offsets가 달라질 수 있다. padding을 output feature로 잘못 노출하면 head reshape가 깨지고, padding을 무시한 loader는 다음 slice를 6개 앞당긴다. 각 class가 `output_size`, `output_size_per_partition`, padded size를 어떤 이름으로 구분하는지 기록한다.

O projection은 QKV와 반대 방향의 좋은 검증점이다. head output이 local `[T,local_q_heads,d]`에서 flatten된 `[T,local_q_width]`가 되고 row-parallel weight와 곱해 local partial `[T,H]`가 된다. collective 후 global branch `[T,H]`가 residual에 더해진다. flatten이 copy인지 view인지, local head order가 checkpoint input-axis shard order와 같은지 기록한다. QKV projection만 맞아도 O projection input shard가 뒤섞이면 attention branch는 틀린다.

장부를 채우는 실제 순서는 config-derived global shape, TP-derived local shape, checkpoint stored shape, loader destination range, forward output shape, consumer expected shape다. 중간에 “아마”가 나오면 source link를 추가한다. 숫자가 나누어떨어진다는 사실은 correctness 증거가 아니다. 어떤 축을 어떻게 나눴는지가 증거다.

이 장의 최소 static assertion 묶음은 다음과 같다.

```text
sum(global_qkv_widths) == packed_global_width
sum(local_qkv_widths) == packed_local_width
each_loader_destination == matching_forward_split
gate_destination != up_destination
row_parallel_input_width * TP == global_input_width  # replication 예외 별도
collective_output_width == residual_width
norm_weight_width == hidden_width
quant_metadata_partition matches payload partition
```

마지막 assertion은 format마다 식이 달라 별도 함수가 필요하다. 중요한 점은 payload shape만 검사하지 않는다는 것이다.

## 11.7 실패 재현: NaN·precision·packed-order의 최초 차이

zero/tiny/large row fixture에서 reference fp32 RMS와 epsilon을 손계산한다. native와 fused 경로의 norm 직후를 비교한다. norm부터 NaN이면 epsilon/cast/reduction이고 projection 뒤부터면 weight·quant scale·GEMM을 본다.

장애 장면 하나를 따라가 보자. 단일 GPU fp16은 정상인데 bf16 TP=4에서 세 번째 layer부터 logits가 NaN이 된다. “bf16이 불안정하다”는 결론부터 내리면 너무 넓다. 먼저 layer 0~3 residual max, finite ratio, norm mean square, inverse RMS를 수집할 설계를 만든다. layer 2 residual 입력까지 유한하고 norm inverse RMS부터 NaN이면 norm 경계다. norm output은 유한하고 QKV projection 뒤 처음 NaN이면 weight shard·scale·GEMM 경계다.

경쟁 가설을 적는다. 가설 A는 residual에 이미 inf가 있었지만 이전 probe가 max만 보고 놓쳤다는 것이다. 가설 B는 TP local hidden slice만 norm해 global hidden row와 다른 분모를 만들었다는 것이다. 가설 C는 quant scale shard가 잘못 broadcast되어 projection에서 overflow했다는 것이다. 각 가설은 다른 관찰로 반증된다. A는 full finite mask와 first non-finite index, B는 rank별 norm reduction axis와 input replication, C는 affine 후 norm은 정상이고 projection output부터 깨지는 것으로 나눈다.

norm은 보통 TP rank마다 full hidden row를 보므로 rank-local로 독립 계산할 수 있다. hidden axis 자체가 sequence parallel 등으로 나뉜 구성이라면 global sum-square reduction이 필요할 수 있다. architecture의 TP와 sequence parallel을 혼동하면 B를 잘못 기각한다. tensor가 어느 parallel domain에서 replicated인지 장부가 필요한 이유다.

precision incident에서는 tolerance를 계층화한다. affine 전 norm row의 작은 absolute error, projection 뒤 누적된 relative error, selected logits 순위 변화, 최종 token 차이를 구분한다. bitwise 불일치가 곧 semantic failure는 아니지만, 첫 layer부터 예상보다 큰 오차가 자라면 backend 차이로 뭉개지 않는다. input magnitude별 error curve를 만들고 epsilon-dominated 구간과 ordinary 구간을 분리한다.

fixture 세트는 다음처럼 구성한다.

| fixture | 목적 | 예상되는 민감도 |
|---|---|---|
| all zero | epsilon·Gemma scale | denominator와 weight convention |
| tiny alternating | epsilon·underflow | cast와 eps 값 |
| constant nonzero | LayerNorm/RMSNorm 구분 | mean subtraction |
| one large outlier | reduction·overflow | accumulation dtype |
| asymmetric signs | shift와 orientation | RMS vs centered norm |
| non-square weight | transpose | stored/logical axes |
| digit-coded rows | packed order | loader/split offsets |
| boundary feature | TP shard | rank offset과 collective |

all-zero 입력에서 projection bias가 있으면 norm 뒤는 zero여도 projection output은 bias다. 이를 NaN fixture와 섞지 않는다. norm의 oracle과 linear의 oracle을 단계별로 둬야 한다.

packed-order incident는 Q에 식별 가능한 constant row, K에 다른 row, V에 세 번째 row를 넣는 synthetic checkpoint schema로 검증한다. 실행은 하지 않더라도 loader slice와 split offsets를 source에서 대조할 수 있다. shape parity보다 slice identity가 중요하다.

가령 query width 4, key width 2, value width 2인 local pack에 q 표식 `[101,102,103,104]`, k `[201,202]`, v `[301,302]`를 둔다. 예상 packed row는 정확히 이 순서다. loader가 K와 V identifiers를 바꾸면 `[101,102,103,104,301,302,201,202]`가 된다. split shape는 여전히 4·2·2로 완벽하다. attention score는 value를 key로 사용하고, softmax 결과를 key-like 값에 곱한다. crash가 아니라 품질 붕괴가 나타난다.

gate/up order도 같은 방식으로 검사한다. scalar fixture에서 gate=1, up=3이면 `silu(1)×3≈2.193`이다. 뒤집으면 `silu(3)×1≈2.858`이다. shape와 dtype은 같다. final output만 보면 모델 오차로 보이지만 packed slice 직후 표식을 보면 즉시 잡힌다.

packed-order가 의심될 때 실제 모델 weight의 통계만 비교하는 것은 약하다. q/k/v weight 분포가 비슷할 수 있기 때문이다. parameter 이름, destination range, source tensor identity를 loader trace에 넣는 것이 더 강하다. runtime trace를 하지 않는다면 loader source의 branch와 offset 식을 정적 표로 전개한다.

TP에서만 틀리면 shard boundary feature를 포함하고 collective 전 local partial을 본다. local부터 다르면 checkpoint slice, local은 맞고 global만 다르면 collective group/reduction이다. residual과 norm이 같으면 앞 장을 더 의심하지 않는다.

boundary feature란 각 shard의 첫 원소와 마지막 원소에 표식을 둔 fixture다. global input width 8, TP=2라면 index 3과 4가 경계다. 두 값만 nonzero로 두고 row-parallel partial을 손계산한다. off-by-one slice는 다른 모든 zero 때문에 선명하게 드러난다. output shard도 같은 방식으로 rank 0 마지막과 rank 1 첫 output을 검사한다.

collective 전 local partial이 맞고 collective 후만 틀릴 때는 operation 종류도 확인한다. row-parallel 합에는 sum이 필요하다. 실수로 average를 쓰면 TP 크기만큼 작아지고, gather를 쓰면 width가 늘어난다. reduce-scatter를 쓰는 설계라면 후속 residual도 shard 상태여야 한다. helper 이름보다 반환 shape와 semantic을 본다.

NaN이 특정 rank에서 먼저 생기면 all-reduce가 모든 rank로 퍼뜨릴 수 있다. collective 후 모든 rank가 동시에 NaN이라고 해서 원인이 collective라고 단정할 수 없다. collective 전 probe가 필요하다. 반대로 local partial은 모두 유한하지만 sum에서 overflow할 수도 있다. rank별 max와 sum expected bound를 함께 기록한다.

quantization incident는 세 경계로 나눈다. loader가 payload와 scale을 올바른 local slice에 넣었는가, quant method가 올바른 group axis로 dequant/GEMM했는가, output dtype·bias가 올바르게 적용됐는가. fp16 checkpoint는 맞고 quantized checkpoint만 틀리면 norm을 다시 의심하기보다 effective weight row를 작은 fixture로 복원한다.

복구 판정은 “문제가 사라졌다”가 아니다. 수정 전 fixture가 예상한 first divergence를 재현하고, 수정 뒤 그 경계가 reference tolerance 안으로 들어오며, 인접한 pack/TP 조합도 통과해야 한다. 예를 들어 K/V order를 고친 뒤 TP=1만 검사하면 local packing과 global shard의 조합 오류를 놓칠 수 있다. TP=1·2, MQA/GQA처럼 폭이 다른 두 configuration을 최소 회귀 행렬로 둔다.

비실행 코드 감사의 종료 조건도 명시한다. source에서 config field 소비자, norm 수식, loader mapping, offsets, collective 위치가 모두 고정 링크로 연결되고 손계산과 모순이 없어야 한다. 실제 numeric parity와 backend symbol은 미검증으로 남긴다. 증명한 것과 관찰 예정인 것을 섞지 않는다.

## 11.8 RMSNorm reduction과 fused residual을 숫자로 해부한다

RMSNorm을 `y_i = w_i x_i / sqrt(mean_j(x_j²)+eps)`로 시작하되 구현에서는 네 단계로 나눈다. 입력 cast, 제곱합 reduction, reciprocal square root, affine/cast다. fused add-norm은 그 앞에 residual add를 넣고 residual carrier와 normalized branch의 생명주기를 함께 바꾼다. 결과 수식만 같다고 kernel 계약이 같다고 볼 수 없는 이유다.

### 11.8.1 네 원소 row로 reduction을 검산한다

`x=[1,-2,3,-4]`, `eps=1e-6`, `w=[1,1,1,1]`이라고 하자. 제곱은 `[1,4,9,16]`, 합은 30, mean square는 7.5다. inverse RMS는 `1/sqrt(7.500001)`로 약 0.365148이고 출력은 약 `[0.365148,-0.730297,1.095445,-1.460593]`다. 평균을 빼지 않으므로 LayerNorm 결과와 다르다.

FP16/BF16 입력을 그대로 제곱하고 줄이면 작은 값은 underflow하고 큰 값은 overflow할 수 있다. reference가 FP32 accumulation을 사용한다면 custom kernel도 partial sum과 final reduction이 어느 dtype인지 확인한다. 입력/출력 dtype이 BF16이라는 사실은 accumulator dtype을 말하지 않는다. kernel template, shared memory type, warp reduction helper, launcher dispatch를 연결한다.

reduction 축은 hidden dimension D다. packed token tensor `[T,D]`에서 각 row가 독립 RMS를 가져야 한다. stride가 잘못돼 이웃 token 일부를 같은 reduction에 넣으면 전체 shape와 finite ratio는 정상일 수 있다. sentinel row A=`[1,1,1,1]`, B=`[2,2,2,2]`를 두면 올바른 normalized 값은 둘 다 거의 `[1,1,1,1]`이지만 잘못 섞인 통계는 서로 다른 scale을 만든다. row별 expected inverse RMS를 별도로 비교한다.

### 11.8.2 warp와 block reduction에서 동일한 합을 누가 소유하는가

D가 warp보다 크면 각 lane이 여러 element의 partial sum을 만들고 warp/block reduction으로 합친다. D=4096, block 256 thread라면 thread 하나가 대략 16 element를 읽을 수 있다. 정확한 mapping은 kernel source가 결정한다. 중요한 불변식은 valid hidden element가 정확히 한 partial에 포함되고 최종 sum이 row 하나에 귀속되는 것이다.

vectorized load가 8-byte나 16-byte 단위를 요구하면 pointer alignment와 D divisibility가 eligibility를 바꾼다. fast path가 tail을 별도로 처리하지 않는데 D가 vector width의 배수가 아니면 OOB 또는 누락이 생긴다. launcher guard가 contiguity, alignment, width를 검사하는지 읽는다. guard가 실패하면 어떤 fallback을 선택하는지도 적는다.

reduction order 차이는 작은 수치 차이를 만들 수 있지만 row가 섞이거나 epsilon/weight convention이 다른 구조 오류를 정당화하지 않는다. tolerance를 정하기 전에 constant, alternating-sign, tiny, large, one-hot row를 손계산한다. non-finite와 wrong-row는 exact structural failure로 분류한다.

### 11.8.3 fused add-norm의 두 출력 계약

입력 hidden `h`와 residual `r`가 있을 때 fused op는 `s=h+r`을 만들고 `n=RMSNorm(s)`을 반환한다. 다음 residual carrier는 s이고 projection 입력은 n이다. 구현은 `(n,s)`를 반환하거나 s를 residual buffer에 in-place write하고 n만 반환할 수 있다. caller가 tuple 순서를 잘못 해석하면 둘 다 `[T,D]`라 shape assertion이 통과한다.

작은 fixture로 `h=[1,1,1,1]`, `r=[1,2,3,4]`를 두면 `s=[2,3,4,5]`다. s의 RMS는 `sqrt(54/4)=sqrt(13.5)≈3.6742`, n은 약 `[0.5443,0.8165,1.0887,1.3608]`다. projection 입력 첫 값이 2라면 raw residual carrier를 읽은 것이고, 다음 add의 residual 첫 값이 0.5443이면 normalized branch를 잘못 보존한 것이다.

residual 인자가 `None`인 첫 호출에서는 op가 input 자체를 carrier로 삼거나 별 copy를 만들 수 있다. 두 번째 호출과 return contract가 다를 수 있으므로 두 branch를 모두 읽는다. model caller가 첫 layer와 후속 layer에서 다른 unpack을 하는지도 확인한다.

### 11.8.4 bias와 affine 위치를 분리한다

RMSNorm learned weight와 projection bias는 서로 다른 affine다. norm 출력은 elementwise `w_i`를 곱하고, linear output은 output feature별 bias를 더한다. packed QKV에서 Q/K/V bias가 모두 존재하는지, 일부만 존재하는지 model config와 constructor를 읽는다. bias 없는 model에 zero bias buffer를 넣는 최적화와 bias가 semantic parameter로 존재하는 경우를 구분한다.

Gemma 계열 norm은 stored parameter를 `1+w`로 해석할 수 있다. stored w=0 fixture에서 표준 RMSNorm 출력은 zero지만 `1+w` 규약은 normalized x를 유지한다. checkpoint converter가 이미 1을 더했는지 runtime kernel이 더하는지 owner를 하나로 정한다. loader와 forward가 모두 더하면 scale이 두 번 적용된다.

## 11.9 packed QKV stride·bias·dtype divergence 사건

사건 N11은 reference와 serving stack의 norm output이 허용 오차 안에서 같지만 packed QKV 직후 첫 layer부터 답이 달라졌다. TP=1 BF16은 정상이고, TP=2에서 KV head 수가 작은 config와 bias enabled artifact에서만 재현됐다. shape는 모두 기대값이었다. 원인 후보는 packed destination offset, non-contiguous stride, bias shard, output cast 네 가지다.

### 11.9.1 logical QKV와 physical storage를 따로 그린다

H=8, query heads=4, KV heads=2, head dim=2라면 Q width=8, K=4, V=4이고 packed logical width는 16이다. 수학적 projection은 `[T,8] @ [8,16] -> [T,16]`이지만 PyTorch stored weight는 `[16,8]`일 수 있다. canonical output ranges는 Q `[0,8)`, K `[8,12)`, V `[12,16)`다.

TP=2에서 Q는 rank당 4, K/V는 각각 2라 local packed width는 8이다. rank-local canonical ranges는 Q `[0,4)`, K `[4,6)`, V `[6,8)`다. global packed tensor를 단순 contiguous 절반 `[0,8)`과 `[8,16)`로 자르면 rank 0은 Q 전체, rank 1은 K/V 전체를 갖게 되어 head sharding이 아니다. loader는 각 semantic slice를 별도로 shard한 뒤 local destination에 pack해야 한다.

stored buffer가 alignment 때문에 local physical width 12를 갖고 Q `[0,4)`, padding `[4,8)`, K `[8,10)`, V `[10,12)`일 수도 있다. forward가 logical contiguous offset을 가정해 `[4,6)`을 K로 읽으면 padding row를 읽는다. shape를 `[T,8]` view로 강제하는 과정에서 stride나 storage offset이 잘못될 수 있다. logical offset과 physical destination descriptor를 분리한다.

### 11.9.2 stride를 element 단위로 계산한다

T=2, local packed width=8인 row-major BF16 contiguous output의 element stride는 `(8,1)`, byte stride는 `(16,2)`다. Q view `[2,4]`의 row stride는 여전히 8 element이고 storage offset 0이다. K view `[2,2]`는 row stride 8, offset 4, V는 offset 6이다. 각 slice는 shape만 보면 contiguous처럼 보여도 row 사이에 다른 slice가 끼므로 자체 contiguous가 아니다.

kernel이 K pointer와 width 2만 받고 row stride를 2로 추정하면 첫 row는 맞고 둘째 row에서 Q/K/V 경계를 잘못 읽는다. T=1 decode가 통과하고 T>1 prefill에서 실패하는 전형적 패턴이다. launcher가 leading dimension 또는 stride를 전달하는지, split 뒤 `.contiguous()`를 만드는지, custom kernel이 base packed tensor와 offsets를 받는지 확인한다.

sentinel packed output을 row 0 `[100,101,102,103,200,201,300,301]`, row 1 `[110,111,112,113,210,211,310,311]`로 둔다. 잘못된 K stride 2는 row 1 K 대신 row 0 V나 row 1 Q 일부를 읽을 수 있다. 값의 백 단위가 semantic group을 보여 주므로 wrong stride와 wrong order를 구별한다.

### 11.9.3 bias도 같은 semantic slice와 shard를 가져야 한다

global bias가 Q 8, K 4, V 4라면 TP local bias도 Q/K/V source range를 각각 shard해 local destination `[0,4),[4,6),[6,8)`에 넣어야 한다. weight loader는 올바르게 semantic shard를 했는데 bias loader가 global contiguous 절반을 쓰면 TP=1은 맞고 TP=2만 틀린다.

zero input fixture는 bias 오류를 가장 잘 드러낸다. normalized input을 모두 0으로 두면 matmul output은 0이고 packed output은 bias 자체다. Q bias를 100대, K 200대, V 300대로 표시해 rank별 destination을 확인한다. bias와 weight를 동시에 nonzero로 두면 원인을 섞는다.

일부 architecture는 Q/K/V bias 존재 여부가 다를 수 있다. absent bias를 길이 0으로 취급할지 zero-filled segment로 둘지 packed class contract를 읽는다. forward split offset이 bias presence에 따라 바뀌는 설계인지, widths는 weight에 의해 항상 고정되는지 확인한다.

### 11.9.4 dtype divergence의 최초 cast를 찾는다

norm은 FP32 accumulation 뒤 BF16 output을 낼 수 있고 packed GEMM은 BF16 input/weight, FP32 accumulator, BF16 output을 사용할 수 있다. quantized weight는 dequant scale dtype과 output cast가 추가된다. reference가 norm output을 FP32로 projection에 넘기고 serving이 BF16으로 cast한다면 first divergence는 GEMM이 아니라 norm-to-projection boundary cast다.

ledger에는 stored weight dtype, input dtype, accumulator dtype, bias dtype, output dtype를 별 열로 둔다. autocast나 custom op가 effective dtype을 바꾸는지 source predicate를 찾는다. `torch_dtype=bf16` 하나로 다섯 칸을 채우지 않는다.

작은 projection에서 cancellation을 만든다. input `[1,1]`, weight column `[10000,-10000]`에 작은 delta를 넣으면 accumulation/cast order 차이가 크게 드러날 수 있다. 그러나 이런 수치 민감 fixture와 pack sentinel을 분리한다. pack 오류를 넓은 tolerance로 숨기지 않고 dtype 오차를 exact equality로 거부하지 않는다.

### 11.9.5 incident를 first divergence로 분기한다

N11에서 residual과 norm affine 후가 같고, zero-input bias fixture가 TP=2에서 틀리면 bias shard 가설이 강하다. bias fixture가 맞고 T=1은 통과하지만 T=2 packed slice가 틀리면 row stride를 본다. BF16만 실패하고 FP32 oracle과 effective weight가 맞으면 cast/accumulation을 본다. Q는 맞고 K/V만 틀리면 semantic offsets와 KV replication을 본다.

local packed output이 맞는데 attention input view부터 틀리면 split/view consumer 문제다. local output부터 틀리면 loader/GEMM/bias producer 문제다. collective는 QKV column-parallel 뒤 보통 즉시 필요하지 않으므로 무작정 NCCL을 원인에 넣지 않는다. 다음 consumer가 local heads를 기대하는지 source로 확인한다.

복구는 packed layout version을 명시하고 weight·bias·quant metadata loader와 forward splitter가 같은 descriptor를 쓰게 한다. old packed artifact와 graph cache를 새 reader가 소비하지 않게 generation을 분리한다. 수정 뒤 TP1/2, T1/T2, bias off/on, unequal Q/KV width, non-contiguous view, quant lane을 최소 matrix로 검증한다.

## 11.10 norm에서 projection consumer까지 pinned source를 걷는다

소스 워크는 model config에서 시작한다. norm epsilon과 scale convention, hidden size, Q/KV head 수, head dimension, bias flag, TP size를 적는다. constructor에서 norm class와 QKV projection class가 이 값을 어떻게 받는지 확인한다. checkpoint loader가 stored parameter를 runtime parameter로 변환하는 지점, forward가 norm output을 projection에 넘기는 지점, custom kernel launcher가 stride/dtype을 해석하는 지점을 잇는다.

### 11.10.1 Transformers reference를 수학 oracle로 읽는다

Transformers Qwen/Gemma source의 RMSNorm forward는 cast, variance/mean-square, rsqrt, weight 적용 순서를 보여 준다. model별 weight convention과 output cast를 그대로 옮긴다. attention constructor는 separate Q/K/V linear의 in/out feature와 bias를 보여 주고 forward는 reshape/transposition 전에 각 output이 어떤 logical tensor인지 고정한다.

reference가 separate linear라고 해서 serving의 packed linear보다 항상 정확하거나 빠르다는 뜻은 아니다. semantic Q/K/V oracle로 사용한다. 각 reference output을 serving packed range에 대응시킨다. reference에서도 backend dispatch나 compiled kernel이 있을 수 있으므로 source 수식과 실제 runtime bit pattern을 혼합하지 않는다.

### 11.10.2 vLLM의 norm과 parallel linear를 caller까지 잇는다

vLLM `RMSNorm`과 fused add 경로에서 residual이 없는 호출과 있는 호출의 반환 값을 읽는다. custom op 호출 인자의 epsilon, weight, output/residual buffer를 확인한다. model decoder layer가 tuple을 어떻게 unpack하고 어떤 값을 QKV와 다음 residual add에 넘기는지 이어야 fused primitive의 의미가 닫힌다.

`QKVParallelLinear`에서는 total head 수, KV head 수, TP size에서 local Q/K/V widths와 replication을 계산하는 부분을 찾는다. generic `MergedColumnParallelLinear`의 output sizes, weight loader의 shard identifier, parameter metadata, forward 반환 layout을 잇는다. bias가 packed parameter의 동일 shard descriptor를 사용하는지도 본다.

`RowParallelLinear`은 input_is_parallel과 reduce_results 같은 상태가 output 의미를 바꾼다. 이 장에서는 QKV 이후 O projection과 MLP down projection의 local partial/complete 경계를 확인한다. caller가 residual add 전에 complete 결과를 받는지, bias가 reduction 전후 어느 rank에서 몇 번 더해지는지 판정한다.

### 11.10.3 SGLang의 fused/parallel 경계를 같은 표에 놓는다

SGLang model layer에서 norm primitive, QKV class, O/down projection, tuple/residual carrier를 찾는다. vLLM과 이름이 비슷해도 반환 계약과 loader hook이 같다고 가정하지 않는다. model-specific weight mapping이 stacked parameter 이름을 Q/K/V identifier로 어떻게 바꾸는지 확인한다.

server가 custom all-reduce나 fused kernel을 선택할 수 있다면 requested option, eligibility, selected primitive를 분리한다. source에 branch가 있다는 사실은 실제 배포가 그 lane을 탔다는 증거가 아니다. 정적 설명에는 predicate를, 후속 runtime artifact에는 selected symbol과 shape를 요구한다.

### 11.10.4 llama.cpp graph에서 extent와 byte stride를 읽는다

llama.cpp graph의 RMS norm node는 logical tensor extent와 epsilon op parameter를 만든다. CUDA norm launcher는 `ne`와 `nb` 같은 extent/byte-stride를 kernel grid와 pointer 산술로 바꾼다. `nb`가 byte 단위인지 element 단위인지 type definition과 caller를 확인한다. contiguous eligibility가 실패할 때 fallback op가 같은 수학을 보존하는지도 본다.

matrix multiplication graph에서는 weight tensor의 logical axes와 backend representation을 분리한다. GGUF tensor shape와 ggml axis order를 PyTorch stored `[out,in]`에 기계적으로 덮지 않는다. graph consumer가 어느 dimension을 contraction 축으로 해석하는지 작은 H=2 fixture로 검산한다. quantized type은 block size와 row bytes가 element_size×count 식과 다를 수 있다.

### 11.10.5 source claim card를 만든다

각 링크에는 `claim`, `input`, `mutation`, `output`, `next consumer`, `not proven`을 붙인다. 예를 들어 norm kernel body는 reduction과 affine 순서를 증명하지만 특정 GPU가 그 kernel을 선택했거나 latency가 개선됐음을 증명하지 않는다. packed loader는 destination mapping을 증명하지만 forward split이 같은 mapping을 소비하는지는 별 link가 필요하다.

revision이 바뀌면 symbol 이름만 검색해 옮기지 않는다. epsilon/scale convention, tuple order, local width 식, physical descriptor, collective 위치가 유지되는지 비교한다. 하나라도 바뀌면 cache/graph artifact compatibility와 regression fixture를 갱신한다.

## 11.11 정적 incident dossier와 배포 종료 조건

incident dossier의 첫 장은 environment가 아니라 first divergence다. artifact/config revision, T/D, TP, bias/quant state, expected/observed checkpoint를 적는다. norm input과 output, local packed output과 split view, local partial/global complete 중 어디까지 맞는지 표시한다. 실행하지 않은 값은 expected source contract로 명시한다.

### 11.11.1 RMS reduction 오류의 경쟁 가설

row A만 tiny magnitude이고 row B는 정상일 때 A에서만 차이가 크다면 epsilon 위치나 underflow를 본다. 모든 row scale이 일정 배수로 틀리면 weight convention이나 mean/sum division을 본다. T>1에서만 row가 서로 영향을 주면 reduction axis/stride를 본다. fused lane만 틀리면 add order, tuple unpack, alias를 본다.

mean 대신 sum을 쓴 D=4 fixture는 inverse scale이 정확히 1/2가 된다. epsilon을 sqrt 밖에 더하면 tiny row에서 예측 가능한 차이가 난다. Gemma stored zero weight는 `w`와 `1+w`를 가른다. 각 가설에 하나의 sharp fixture를 붙이면 무작정 kernel 전체를 읽지 않는다.

### 11.11.2 packed projection 오류의 경쟁 가설

T=1도 틀리면 semantic order, loader destination, weight orientation, bias를 우선한다. T=1은 맞고 T=2 이상만 틀리면 row stride/view를 우선한다. TP=1은 맞고 TP>1만 틀리면 semantic sharding, KV replication, bias shard를 본다. quant only면 payload/scale/zero descriptor와 group axis를 본다.

Q/K/V 모두 같은 방식으로 틀리면 shared input/cast/GEMM을, K/V만 틀리면 unequal widths와 replication을, 한 slice 첫/마지막 group만 틀리면 offset/alignment boundary를 본다. collective 전 local QKV가 이미 틀리면 NCCL은 뒤로 미룬다.

### 11.11.3 dtype divergence를 tolerance로 숨기지 않는다

dtype comparison에는 FP32 oracle, reference cast order, serving effective dtype를 둔다. norm affine 전, affine 후, GEMM accumulator reference, output cast 후를 나눈다. 구조 sentinel은 exact identity로, 수치값은 미리 선언한 atol/rtol로 판정한다. NaN/inf와 sign, token-row permutation은 별 hard fail이다.

오차가 depth에 따라 매끄럽게 누적되는지 특정 layer/slice에서 계단처럼 뛰는지 본다. 첫 layer packed K에서 갑자기 큰 차이가 나면 전체 모델 quantization tolerance를 넓히지 않는다. selected safe row와 boundary feature를 직접 비교한다.

### 11.11.4 regression matrix를 작게 유지한다

norm 축은 zero/tiny/constant/alternating/large row, standard/Gemma scale, fused/unfused다. projection 축은 T1/T2, Q=KV/unequal, bias off/on, TP1/2, contiguous/strided, BF16/quant 대표 lane이다. 전체 Cartesian product 대신 incident 가설을 가르는 pairwise와 경계 cell을 고른다.

각 cell은 norm inverse RMS, affine output, packed base stride/offset, Q/K/V view identity, bias destination, effective dtype, first consumer를 판정한다. output token은 마지막 보조 열이다. first divergence가 맞지 않는데 token만 같은 cell을 PASS로 처리하지 않는다.

### 11.11.5 배포와 rollback을 generation으로 묶는다

packed layout이나 norm convention이 바뀌면 model weight만 교체하지 않는다. loader, custom kernel, graph capture, adapter/quant metadata, cache namespace가 같은 generation을 사용해야 한다. old writer가 만든 packed artifact를 new reader가 읽지 않게 key version과 compatibility predicate를 둔다.

canary는 ordinary row뿐 아니라 tiny norm, Gemma zero weight, Q/K/V boundary, zero-input bias, T2 stride, TP shard boundary를 포함한다. mixed old/new worker에서 selected generation을 trace하고 old graph/weight entry가 더 이상 소비되지 않을 때 migration을 닫는다.

correctness terminal은 norm과 projection의 의미 checkpoint가 reference contract와 맞는 것이다. layout terminal은 loader와 forward가 같은 Q/K/V descriptor와 stride를 쓰는 것이다. numerical terminal은 선언 dtype envelope 안에서 finite와 tolerance를 통과하는 것이다. lifetime terminal은 fused residual carrier와 packed base가 마지막 consumer까지 유효한 것이다. performance terminal은 correctness 뒤에 latency/memory 효과를 측정한다.

최종 사건 문장은 이렇다. “TP=2, T>1, bias enabled lane에서 norm affine 출력은 reference와 일치했으나 K view의 row stride가 local width 2로 추정돼 packed base stride 8을 잃었다. row 1 K가 row 0 V를 읽은 것이 first divergence였다. base+offset+leading-dimension descriptor로 수정하고 zero-input bias shard와 unequal Q/KV boundary를 함께 검증했으며 old graph generation을 격리했다.”

이 dossier가 있으면 독자는 “fused kernel이라 수치가 조금 다르다”는 설명에 머물지 않는다. reduction, affine, residual carrier, packed storage, bias, dtype 중 최초로 다른 계약을 지목하고 다음 consumer까지 확인한다. 12장에는 검증된 Q/K/V logical slices와 local head identity만 넘긴다.

## 11.12 성능 최적화의 이유를 메모리 traffic과 consumer 계약으로 설명한다

RMSNorm fusion이 필요한 이유를 “kernel 수를 줄여 빠르다”로 끝내지 않는다. unfused residual add는 h와 r을 읽고 s를 쓰며, norm은 s를 다시 읽고 normalized n을 쓴다. BF16 `[T,D]` 기준으로 단순 하한은 add가 두 번 read+한 번 write, norm이 적어도 한 번 read+한 번 write다. fused op는 s의 중간 global-memory round trip 일부와 launch를 줄일 수 있다.

T=4096,D=4096이면 tensor 하나가 32MiB다. 단순 모델에서 add의 h/r read 64MiB+s write 32MiB, norm의 s read 32MiB+n write 32MiB로 최소 160MiB traffic이 보인다. fusion이 s write/read 64MiB를 피할 잠재력이 있다. 실제 kernel은 weight read, reduction partial, cache behavior, alignment과 cast가 있으므로 이 숫자는 상한 설명이지 실측 bandwidth가 아니다.

fusion은 대신 residual carrier s의 ownership을 kernel 안으로 옮긴다. s를 다음 add와 hidden-state observer가 필요로 한다면 완전히 없앨 수 없고 output buffer로 유지해야 한다. normalized n과 s 두 출력을 쓰는 fused op는 launch와 input read를 줄여도 output traffic은 남는다. “중간 tensor 제거”라는 설명을 실제 반환 contract로 교정한다.

packed QKV도 세 GEMM을 하나로 묶어 input activation을 재사용하고 큰 GEMM shape를 만들 수 있다. 그러나 Q/K/V widths가 다르고 TP에서 semantic shard 규칙이 다르므로 pack descriptor가 복잡해진다. 성능 이득은 loader/forward가 같은 layout을 유지할 때만 correctness와 양립한다.

H=4096, Q width=4096, K=V=1024라면 separate output 총 width는 6144다. packed GEMM weight logical element 수는 동일하지만 input `[T,4096]`을 한 번의 GEMM scheduling으로 소비할 수 있다. launch 세 개가 하나가 되고 weight access와 output write는 여전히 필요하다. 작은 T=1 decode와 큰 T prefill에서 이득 방향이 다를 수 있다.

### 11.12.1 kernel eligibility가 효과를 결정한다

fused/packed source가 존재해도 dtype, alignment, contiguous stride, width, bias, quant method, graph capture 조건이 맞아야 선택될 수 있다. requested optimization과 effective backend를 구분한다. fallback이 correctness를 보존하면 정상 기능일 수 있지만 latency 회귀의 원인이 된다. fallback reason을 bounded enum으로 관측한다.

T=1 decode에서 특화 GEMV를 쓰고 T가 크면 GEMM을 쓸 수 있다. 같은 packed parameter라도 consumer kernel과 stride 요구가 다르다. prefill은 맞고 decode만 틀리면 shape-specific split/launcher를, decode는 맞고 prefill만 틀리면 leading dimension과 row stride를 본다.

### 11.12.2 bias와 residual fusion의 중복을 막는다

row-parallel output에서 각 rank partial에 동일 full bias를 더한 뒤 sum하면 bias가 TP배 중복된다. 올바른 설계는 bias를 한 rank/collective 후에 더하거나 partial bias를 적절히 나누는 것이다. fused residual add도 collective 전 각 rank에서 full residual을 더하면 residual이 TP배가 된다.

TP=2, local partial y0=3,y1=5, bias=2,residual=10이면 기대 output은 `3+5+2+10=20`이다. 각 rank가 bias와 residual을 더해 sum하면 `(3+2+10)+(5+2+10)=32`다. shape와 finite ratio는 정상이고 모든 rank 결과도 같다. 수치 fixture로 collective 전후 affine owner를 검산해야 한다.

column-parallel QKV bias는 각 output slice가 한 rank에 소유되므로 local bias 적용이 자연스럽다. KV replication에서는 같은 global K/V head가 여러 rank에 있을 수 있지만 각 rank의 local attention consumer를 위한 replica이므로 bias도 같은 head와 함께 복제돼야 한다. output reduction과 replication 의미를 혼합하지 않는다.

### 11.12.3 packed base와 view의 lifetime을 닫는다

Q/K/V가 packed base의 strided view라면 attention과 cache write가 모두 끝날 때까지 base storage가 살아 있어야 한다. Q consumer가 끝났다고 base를 재사용하면 비동기 K/V cache write가 stale/overwritten 값을 읽을 수 있다. Python view object 생존만으로 custom CUDA consumer completion을 보장한다고 가정하지 않는다.

cache write가 K/V를 별 persistent buffer에 copy한 뒤 attention이 그 cache를 읽는지, current-token K/V view를 직접 읽는지 backend별로 다를 수 있다. source에서 launcher input pointer와 event dependency를 확인한다. 12~14장으로 넘길 lifetime handoff에 base generation과 K/V consumer를 포함한다.

### 11.12.4 모니터링과 디버그 artifact를 분리한다

fleet metric에는 norm non-finite count, fused eligibility/fallback, packed layout version, projection dtype, TP size, first-divergence bucket 같은 bounded state를 둔다. QKV offsets, stride, selected safe slice는 sampled synthetic trace에 둔다. raw activation과 weight는 metric label이나 일반 log에 넣지 않는다.

latency histogram은 norm/add, QKV GEMM, collective를 가능한 범위에서 분리하되 profiler label이 실제 fused kernel과 맞는지 확인한다. fusion 뒤 기존 norm span이 사라졌다고 norm 비용이 0이 된 것은 아니다. fused span에 비용이 합쳐졌다. 전후 비교는 동일 의미 구간의 total duration과 traffic을 본다.

### 11.12.5 변경 review의 인과 문장

좋은 최적화 설명은 다음 형식이다. “Residual add와 RMS reduction이 같은 `[T,D]` input을 연속으로 읽어 중간 s write/read가 발생하므로, fused op가 s를 residual output으로 보존하면서 normalized n을 함께 생성한다. 이로써 launch와 예상 64MiB intermediate traffic을 줄이되 tuple ownership과 hidden-state observer lifetime을 새 contract로 검증한다.”

packed projection은 “세 projection이 같은 normalized input을 읽고 output axis로 concat 가능하므로 하나의 packed GEMM을 사용한다. unequal Q/K/V widths와 TP replication 때문에 semantic source/destination descriptor를 loader와 forward가 공유하며, bias와 quant metadata도 같은 descriptor를 따른다”라고 쓴다.

성능 결과가 기대보다 작으면 병목을 다시 본다. norm이 memory-bound여도 전체 layer는 GEMM/attention/collective가 지배할 수 있다. QKV fusion이 launch를 줄여도 weight bandwidth와 small-batch kernel occupancy가 제한할 수 있다. tokens/s 하나보다 phase, T/D, TP, backend, fallback rate와 correctness terminal을 함께 제시한다.

### 11.12.6 독자가 다음 소스를 여는 stop rule

norm input부터 다르면 10장 residual/row mapping으로 돌아간다. inverse RMS부터 다르면 reduction axis/dtype/epsilon을 본다. inverse RMS는 맞고 affine만 다르면 weight convention/loader를 본다. norm은 맞고 packed base부터 다르면 projection weight/bias/dtype을 본다. base는 맞고 view만 다르면 stride/offset consumer를 본다.

Q/K/V까지 맞으면 이 장을 떠난다. head reshape와 TP/KV identity는 12장, causal visibility와 ragged coordinates는 13장, RoPE/cache position은 14장으로 간다. gate/up이 맞고 elementwise/down에서 갈리면 15장이다. first divergence가 경계를 정하므로 익숙한 kernel만 계속 의심하지 않는다.

11장의 최종 artifact는 normalized row oracle, fused residual tuple contract, packed layout descriptor, bias/quant metadata mapping, dtype ledger, local/global checkpoint다. 이 여섯 항목을 고정하면 fused kernel의 이유와 위험을 같은 문장으로 설명할 수 있고, 12장의 head partition 계산이 잘못된 QKV 위에서 시작되는 일을 막는다.

마지막 검산은 의도적으로 서로 다른 실패를 한 fixture에 섞지 않는다. norm reduction fixture는 identity weight와 bias 없는 projection을 사용한다. bias shard fixture는 zero input을 사용한다. stride fixture는 packed base를 직접 sentinel로 만들고 GEMM 수치를 우회한다. dtype fixture는 layout이 확정된 작은 dense projection을 사용한다. 한 실험이 한 가설을 가를 때 결과가 설명 가능하다.

RMSNorm tiny-row 예로 `x=[1e-4,-1e-4,1e-4,-1e-4]`를 둔다. mean square는 `1e-8`이다. eps=1e-6이면 denominator는 약 `sqrt(1.01e-6)`이고 epsilon이 값의 크기를 지배한다. eps가 sqrt 밖에 더해지거나 입력 dtype에서 제곱이 0으로 underflow하면 예측 가능한 다른 scale이 나온다. ordinary row만으로는 이 차이가 작아 보일 수 있다.

large-row는 overflow를 가른다. FP16 max에 가까운 값을 직접 쓰기보다 제곱 시 FP16 범위를 넘지만 FP32에는 안전한 값을 고른다. 입력 cast 전후와 accumulator를 확인한다. kernel이 안정화를 위해 max scaling을 쓰는지 단순 sum을 쓰는지도 source로 확인한다. 안정화 방식이 다르면 수학적으로 동등해도 rounding envelope가 달라질 수 있다.

packed stride의 passing neighbor는 T=1이다. failing T=2와 같은 weight, bias, TP, backend에서 T만 바꾼다. T=1이 맞다는 사실은 semantic offsets가 대체로 맞고 row advance에서 문제가 생긴다는 가설을 강화한다. 그러나 T=1 전용 kernel을 쓴다면 backend 차이가 새 축이므로 selected symbol을 기록한다.

bias shard의 passing neighbor는 bias disabled artifact다. bias off에서 맞고 on에서만 틀리면 weight pack과 input dtype 가설이 약해진다. zero input에서 observed output을 expected local bias와 직접 비교한다. collective 뒤 full bias가 TP배가 됐는지도 scalar 계산으로 판정한다.

quantized lane에서는 code, scale, zero, group index, output cast 중 어디서 처음 다르는지 본다. dequantized effective row가 reference와 같으면 loader/metadata를 닫고 GEMM consumer로 간다. effective row가 다르면 GPU matmul 성능을 먼저 분석하지 않는다. padding group과 shard boundary가 겹치는 마지막 group을 반드시 포함한다.

fused residual incident에서는 old residual snapshot, new carrier, normalized output의 세 값을 보존한다. 두 값만 있으면 tuple swap과 in-place overwrite를 구별하기 어렵다. producer 직후 snapshot과 next consumer input을 함께 비교해 어느 edge에서 의미가 바뀌었는지 찾는다.

문서의 option 설명도 이 계약을 따른다. dtype option은 weight/activation/accumulator/output와 kernel eligibility를 어떻게 바꾸는지, TP option은 local widths와 collective/bias owner를 어떻게 바꾸는지, quant option은 physical descriptor와 scale axis를 어떻게 바꾸는지 적는다. “메모리 절약”, “빠른 fused path” 같은 효과만 쓰지 않는다.

배포 전 정적 감사에서 확인하지 못한 selected kernel과 실측 stride는 후속 runtime TODO로 남긴다. source predicate와 expected shape, 필요한 trace field를 명시한다. 실행하지 않은 latency나 numerical distribution을 추정치처럼 쓰지 않는다. 정확한 미확인은 부정확한 확정보다 유용하다.

장애 종료 뒤에는 잘못된 가설을 기록한다. 예를 들어 모든 rank가 틀려 collective를 의심했지만 collective 전 rank 1 bias부터 달랐다면, “global symptom은 local partial 오류가 reduction으로 전파된 결과”라는 반례를 남긴다. 다음 분산 사건에서 같은 오류를 반복하지 않게 한다.

최종 review 문장은 짧다. “어느 row의 어떤 reduction이 어떤 dtype으로 계산됐고, residual carrier와 normalized branch는 어떤 storage/generation을 가지며, packed Q/K/V의 logical range가 어떤 physical stride와 bias/metadata descriptor로 소비되는가.” 이 문장에 source와 fixture로 답하면 11장은 닫힌다.

운영 dashboard에서는 fused 사용률만 보지 않는다. eligibility 실패 이유, fallback backend, T/D bucket, dtype, bias/quant mode를 함께 본다. fused 사용률이 떨어져 latency가 늘었는지, workload shape가 바뀌어 원래 선택 대상이 줄었는지 구분한다. config 변경과 traffic 변화가 같은 시각에 일어나면 generation별 cohort를 나눈다.

메모리 metric도 allocated byte 하나로 끝내지 않는다. packed base와 views는 storage를 공유하므로 view 크기를 모두 더하면 중복 계산된다. 반대로 fused residual snapshot이나 async copy staging은 별 storage를 만든다. storage identity와 live interval을 기준으로 peak를 해석한다. allocator reserved와 active, graph pool도 구분한다.

latency 회귀에서 norm kernel 시간이 늘었다면 row 수 T와 hidden D, selected vector width, fallback을 먼저 본다. projection이 늘었다면 M/N/K shape, quant group, packed width, TP local width를 본다. 같은 “layer 0 느림”이라도 norm은 memory/reduction, projection은 GEMM/weight access, collective는 통신이라는 다른 owner를 가진다.

correctness 경보가 없다 해도 canary를 유지한다. physical padding row, wrong bias shard, tuple swap은 finite output을 내고 자연어가 그럴듯할 수 있다. synthetic sentinel과 first-layer checkpoint는 silent error를 조기에 잡는다. 최종 token distribution만 보는 품질 metric은 특정 경계 ID 오류를 희석할 수 있다.

새 model architecture를 추가할 때는 기존 packed descriptor에 억지로 맞추지 않는다. Q/K/V 외에 gate나 latent projection이 섞이거나 bias/scale 규약이 다르면 semantic groups를 명시적으로 확장한다. loader와 forward가 동일 descriptor를 소비하고 unknown group을 거부하도록 한다. 이름 문자열의 접두사 매칭만으로 destination을 선택하면 새로운 parameter가 조용히 잘못 pack될 수 있다.

마지막 handoff에는 Q/K/V slice별 shape, stride, dtype, bias, global head identity, rank owner와 base lifetime을 넣는다. 12장은 이 값을 head 차원으로 reshape하고 KV replication/collective를 검산한다. 이 장에서 미확인인 값을 12장에서 기본값으로 채우지 않는다.

reader-facing 설명에서는 kernel 이름보다 먼저 작은 숫자 예를 둔다. 독자가 `[T,D]` reduction과 packed row stride를 손으로 확인한 뒤 source symbol을 열게 한다. source link는 설명의 대체물이 아니라 계산의 producer와 consumer를 검증하는 좌표다. 이렇게 해야 backend가 바뀌어도 독자가 같은 계약을 새 함수에서 찾을 수 있다.

반대로 source note에만 있는 중요한 조건은 본문으로 올린다. fused eligibility의 contiguity, KV의 unequal width, bias 적용 위치, accumulator dtype처럼 결론을 바꾸는 조건은 링크 목록에 묻히면 안 된다. 부차적인 file inventory는 늘리지 않는다.

장 종료 시 unknown 항목을 세어 0으로 꾸미지 않는다. selected runtime symbol, 실제 numerical envelope처럼 실행 증거가 필요한 칸은 후속 검증으로 표시하고 expected predicate를 남긴다. 정적 확정과 관측 예정이 분리되어야 독자가 뇌피셜과 source fact를 구별한다.

이제 norm output과 packed QKV가 정확히 어떤 의미와 layout을 갖는지 닫혔다. 다음 장에서는 이 slice를 query head와 KV head로 나누고, TP rank별 local head, KV replication과 collective가 attention 의미를 보존하는지 계산한다.

동일 fixture와 descriptor를 다음 revision 감사에서도 재사용해 의미 이동을 즉시 검출한다.
## 11.13 네 스택에서 불변식의 구현 좌표를 찾는다

이 절의 좌표를 열기 전에 tensor 하나를 다시 본다. residual row가 norm을 거쳐 scale 규약을 적용받고, packed projection이 Q·K·V 또는 gate·up slice를 만든다. 오답이 나오면 residual→norm 전→norm 후→packed output→unpacked slice에서 첫으로 다른 값을 찾는다. 이 사건을 주어로 두면 아래 함수 이름이 목록이 아니라 경계의 증거가 된다.

### 11.13.1 소스를 열기 전에 남길 불변식

최초 불일치가 norm 전이면 projection 가설을 버린다. norm 후까지 같고 packed output이 다르면 loader·projection·dtype 경계로 조사를 좁힌다. local partial은 같고 collective 후만 다르면 norm과 loader를 다시 열지 않는다. 아래 좌표는 이 세 반증을 확인할 때만 연다.

### 11.13.2 stack별 구현 좌표

이 장의 구현 관찰점은 Transformers v5.15.1 commit `550d7b3834670483a4df436541272c055dc364bf`, vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, llama.cpp v0.2.0 commit `bb4caa7540188872173c44d161602d9271386413`에 고정했다. 아래 좌표와 손계산은 이 source를 정적으로 읽어 연결한 것이며 모델이나 서버를 실행한 성능 관측은 아니다.

Transformers reference에서 norm 수식과 projection module을 읽고, vLLM/SGLang에서 fused op와 TP linear, packed loader mapping을 잇는다. llama.cpp에서는 graph의 norm과 matrix multiplication node, tensor layout을 같은 의미 좌표로 찾는다. 함수 이름보다 residual→normalized→projected tensor 전이를 맞춘다.

소스 지도는 독자가 실제로 따라갈 순서로 정리한다. 첫째 config에서 hidden size, head 수, KV head 수, intermediate size, epsilon, bias를 찾는다. 둘째 model constructor에서 norm과 projection class를 찾는다. 셋째 layer forward에서 residual과 projection 호출 순서를 찾는다. 넷째 loader에서 checkpoint 이름과 packed destination을 찾는다. 다섯째 linear/norm primitive에서 dtype, shard, collective, fusion을 찾는다.

#### Transformers Qwen 기준점

- Transformers Qwen 경로의 핵심은 [RMSNorm 구현](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L720-L738), [attention projection 선언과 forward](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L806-L878), [decoder layer residual 흐름](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1030-L1105)이다.
- 줄 번호가 revision 실제 symbol과 맞는지 로컬 clone에서도 확인하고, 링크는 고정 commit을 유지한다.

#### Transformers Gemma 비교점

- Transformers Gemma 경로는 [norm scale 규약](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L136-L155), [attention projections](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L250-L360), [decoder layer](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L500-L575)다.
- Qwen과 같은 질문을 던지되 답이 같다고 가정하지 않는다.

#### vLLM primitive·loader 경계

- vLLM primitive 경로는 [`RMSNorm`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/layernorm.py#L37-L130), [`GemmaRMSNorm`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/layernorm.py#L132-L170), [`ColumnParallelLinear`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/linear.py#L419-L620), [`MergedColumnParallelLinear`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/linear.py#L661-L760), [`QKVParallelLinear`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/linear.py#L1022-L1115), [`RowParallelLinear`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/linear.py#L1613-L1735) 순으로 읽는다.
- model class의 construction과 stacked mapping을 이 primitive에 연결해야 한다.

#### SGLang primitive·model 경계

- SGLang primitive 경로는 [`ColumnParallelLinear`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/linear.py#L302-L507), [merged linear 범위](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/linear.py#L590-L760), [QKV 범위](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/linear.py#L1030-L1185), [`RowParallelLinear`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/linear.py#L1407-L1510)와 [Qwen3.5 model construction](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_5.py#L984-L1105)을 잇는다.
- 링크 범위가 넓으므로 실제 symbol 선언을 source note에 함께 적는다.

llama.cpp graph normalization과 multiplication 경계는 [llama.cpp v0.2.0 `llama-graph.cpp:1480-1560`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L1480-L1560), transformer graph의 norm/projection 호출은 [같은 파일 `:2290-2400`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L2290-L2400)에서 architecture-specific builder와 함께 확인한다.

#### llama.cpp graph·CUDA 경계

- llama.cpp CUDA로 한 단계 더 내려가면 [`rms_norm_f32` kernel](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/norm.cu#L77-L157), [launcher의 extent·stride 해석](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/norm.cu#L478-L560), [fused add launcher](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/norm.cu#L562-L645), [fusion eligibility의 contiguous 검사](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/ggml-cuda.cu#L2686-L2715)를 읽는다.
- 이 좌표들은 source가 허용하는 경로를 증명하지만 특정 배포가 실제로 그 kernel을 선택했다는 증거는 아니다.

## 11.14 불변식에서 장애 판정과 다음 장 인계까지 이어지는 실전 워크북

앞 절의 고정 소스 좌표를 실제 조사 절차로 바꾼다. 이 절은 명령 목록이 아니라 세 갈래 독자 경로다. 처음 구현을 읽는 독자는 11.14.1에서 의미 장부를 만들고, 장애를 조사하는 독자는 11.14.2~11.14.4에서 최초 불일치를 좁히며, 변경을 검토하는 독자는 11.14.6~11.14.8에서 승인 조건을 만든다.

### 11.14.1 읽기 워크북: 30분 static differential

첫 5분에는 두 구현의 revision, model config, checkpoint tensor 이름을 고정한다. 다음 5분에는 norm 식과 epsilon·weight 규약을 표로 옮긴다. 다음 10분에는 QKV와 gate-up의 global/local widths와 destination offsets를 계산한다. 마지막 10분에는 residual alias, collective 위치, consumer shape를 연결한다. 빈 칸이 생기면 그 칸의 owner source를 찾는다.

워크북 결과물은 다음 한 장이면 된다.

| 의미 경계 | reference | serving stack | 기대 동일성 | 미검증 항목 |
|---|---|---|---|---|
| norm affine 전 | fp32 식·eps | native/custom op | tolerance 내 값 | 실제 backend symbol |
| norm affine 후 | `w` 또는 `1+w` | loader+kernel | 동일 scale 규약 | reduction order |
| Q/K/V | separate linear | packed slice | slice별 동일 | physical layout |
| O projection | full linear | row-parallel partial+sum | collective 후 동일 | communication timing |
| gate/up | separate outputs | merged slices | order별 동일 | quant kernel |

### 11.14.2 장애 워크북: 증상에서 owner까지

증상이 첫 layer부터 전체 품질 저하라면 norm convention과 packed order를 우선한다. 특정 TP 크기에서만 틀리면 shard/replication/collective를 우선한다. quantized artifact에서만 틀리면 payload-scale partition과 effective weight를 우선한다. 긴 입력에서만 틀리면 이 장의 projection보다 12~14장의 head·attention·position·cache 경계를 우선하되, QKV slice가 길이에 따라 달라지지 않는지 먼저 확인한다.

관측 순서는 residual input→norm affine 전→norm affine 후→packed projection→unpacked slices→collective 전 partial→collective 후 output이다. 각 단계에서 shape, stride, dtype, finite ratio, 작은 value slice, storage identity를 기록한다. 개인정보가 있는 실제 activation 전체를 남기지 말고 synthetic fixture 또는 승인된 hash/statistic을 사용한다.

원인 판정은 경쟁 가설을 기각하는 방식으로 한다. norm 출력부터 다르면 projection을 고쳐서 우연히 logits를 맞추려 하지 않는다. norm은 같고 packed tensor만 다르면 앞 layer를 다시 조사하지 않는다. local partial은 같고 collective 후만 다르면 loader를 다시 고치지 않는다. first divergence가 조사 소유권을 닫는다.

검증은 수정한 경계와 인접 경계를 함께 본다. norm 수정은 zero·tiny·ordinary·large row와 Gemma zero-weight fixture를 통과해야 한다. pack 수정은 Q/K/V 폭이 같은 모델과 GQA처럼 다른 모델을 모두 본다. TP 수정은 1·2·4 rank 논리 fixture와 shard boundary를 본다. quant 수정은 payload뿐 아니라 scale·zero metadata partition을 본다.

### 11.14.3 사례 A: TP=4에서만 조용히 품질이 무너진다

상황을 구체화하자. TP=1과 TP=2에서는 selected logits가 reference와 허용 오차 안에 있는데 TP=4에서만 답변이 반복되고 점수가 크게 갈라진다. 서버는 crash하지 않고 collective error도 없다. config의 query head 수는 32, KV head 수는 2, head dimension은 128이다. 이 숫자만 보면 query head는 rank당 8개지만 KV head는 rank 수보다 적다.

첫 반응으로 NCCL을 의심하기 쉽다. 하지만 먼저 layer 0의 residual과 norm output을 rank별로 본다고 설계한다. 두 tensor가 모든 rank에서 reference와 같다면 norm과 input replication은 일단 기각된다. 다음 checkpoint는 local packed QKV다. query local width는 `8×128=1024`다. KV는 단순 나눗셈 `2/4`가 정수가 아니므로 각 rank에 어느 KV head를 복제할지 규칙이 필요하다.

global KV head 0을 rank 0·1, head 1을 rank 2·3에 복제한다고 가정해 보자. loader destination도 이 규칙을 따라야 하고, forward의 local query head가 참조하는 KV head mapping도 같아야 한다. loader가 modulo로 `rank%2`를 쓴다면 rank 0·2가 head 0, rank 1·3이 head 1을 받는다. 두 규칙은 local shape `[128]`이 같아서 assertion을 모두 통과하지만 query group과 KV identity가 어긋난다.

정적 검산은 rank별 표를 만든다.

| rank | local query global heads | 기대 KV global head | loader가 고른 KV head | 판정 |
|---:|---|---:|---:|---|
| 0 | 0–7 | 0 | source 식 | 비교 |
| 1 | 8–15 | 0 | source 식 | 비교 |
| 2 | 16–23 | 1 | source 식 | 비교 |
| 3 | 24–31 | 1 | source 식 | 비교 |

이 표를 Q와 K, V 각각에 만든다. K만 맞고 V mapping이 다를 수도 있기 때문이다. packed local offsets도 `Q[0:1024]`, `K[1024:1152]`, `V[1152:1280]`처럼 적는다. checkpoint mapping의 `shard_id`, QKV class의 replication 계산, forward split이 세 칸 모두 같은 identity를 만들어야 한다.

만약 local Q/K/V가 모두 맞다면 다음은 O projection 전 attention output과 row-parallel partial이다. local partial까지 reference local decomposition과 맞고 all-reduce 후만 다르면 process group·operation·element count로 분기한다. 반대로 attention output부터 다르면 collective를 건드리지 않는다. 이 순서가 “TP 문제니까 통신 문제”라는 성급한 결론을 막는다.

수정 검증은 TP=4 한 점으로 끝내지 않는다. KV heads 1인 MQA, KV heads 2인 작은 GQA, KV heads가 TP 이상인 GQA를 정적 행렬에 넣는다. rank 수가 KV heads를 나누는 경우와 KV heads가 rank 수를 나누는 경우를 모두 본다. query heads divisibility도 독립 assertion으로 둔다. 이 사례의 핵심은 local tensor width가 맞다는 사실과 local tensor identity가 맞다는 사실이 다르다는 것이다.

### 11.14.4 사례 B: 양자화 모델의 첫 layer norm은 맞지만 projection 뒤 NaN이다

두 번째 상황은 fp16 artifact는 정상이고 특정 group-wise quantized artifact만 첫 layer QKV projection에서 NaN이 되는 경우다. residual input, mean square, inverse RMS, affine 후 norm output이 reference와 모두 맞는다. 이 증거가 있으면 epsilon이나 fused norm을 계속 바꾸는 것은 조사 범위를 거꾸로 넓히는 일이다.

projection을 세 조각으로 나눈다. 첫째 loader가 payload·scale·zero metadata를 올바른 parameter와 shard에 넣었는가. 둘째 quant method가 group axis와 packing bit order를 올바르게 해석했는가. 셋째 GEMM accumulator와 output cast가 유한 범위에 있는가. effective weight 몇 행을 복원하면 첫째와 둘째를 빠르게 나눌 수 있다.

간단한 affine quantization을 예로 들자. integer code가 `[1,2,3,4]`, group size 2, scale이 `[0.5,10]`, zero가 `[1,3]`이면 effective row는 앞 group `(q-1)×0.5=[0,0.5]`, 뒤 group `(q-3)×10=[0,10]`이다. scale 두 개를 output row가 아니라 input group에 broadcast해야 한다. scale order가 뒤집히면 `[0,10,-1,0.5]` 같은 전혀 다른 row가 된다. shape는 둘 다 `[4]`다.

packed QKV에서는 scale metadata도 Q/K/V destination range를 따라야 한다. payload의 output width는 quant block에 맞게 padding될 수 있고 scale width는 group count에 맞게 별도 padding될 수 있다. payload offset을 그대로 scale offset으로 재사용할 수 있다는 보장은 없다. quant parameter class의 loader와 linear class의 weight loader hook을 각각 읽는다.

effective row가 유한하고 reference와 맞는데 GEMM output만 NaN이라면 input magnitude, accumulator dtype, backend kernel eligibility, output scale를 본다. 반대로 effective row에 이미 비정상적으로 큰 값이나 NaN이 있다면 kernel을 프로파일링하기 전에 metadata 해석을 고친다. first divergence 원칙은 성능 kernel 조사에도 그대로 적용된다.

수정 뒤에는 Q·K·V 각 slice의 첫·마지막 group, TP shard 경계 group, padding이 끼는 마지막 group을 검사한다. 평균 오차만 보면 경계 한 블록의 오류가 전체 weight에 희석된다. selected rows를 의도적으로 고르는 이유다. bias가 있다면 dequantized matmul 결과와 bias 적용 후 결과를 별도 checkpoint로 둔다.

### 11.14.5 수치 오차 예산을 결과 뒤에 붙이지 않는 법

허용 오차는 결과가 마음에 들지 않을 때 늘리는 숫자가 아니다. 연산 경계별 dtype과 reduction 길이에서 미리 정한다. norm은 hidden width만큼 제곱합을 줄이고 inverse square root를 수행한다. projection은 hidden width만큼 곱셈·덧셈을 누적한다. projection 출력의 error가 norm 출력 error보다 커질 수 있는 구조적 이유가 있다.

절대 오차만 쓰면 큰 값에 너무 엄격하고 zero 부근에서 상대 오차가 폭발한다. 보통 `|a-b| <= atol + rtol×|b|` 형태를 쓰되, NaN/inf parity는 별도로 검사한다. selected logits 순위와 token agreement는 수치 checkpoint를 대체하지 않는다. 작은 activation 차이가 최종 argmax를 바꾸지 않을 수도 있고, 거의 동률인 logits에서는 아주 작은 차이가 token을 바꿀 수도 있다.

norm affine 전 비교는 fp32 oracle을 둔다. affine 후에는 weight dtype과 cast 순서를 반영한다. linear는 effective dequantized weight oracle과 서빙 quant kernel을 분리한다. TP는 local partial oracle과 global sum oracle을 분리한다. 이렇게 해야 하나의 넓은 tolerance가 pack order나 missing collective 같은 큰 오류를 숨기지 않는다.

reduction 비결정성도 무제한 면죄부가 아니다. 같은 input과 같은 backend에서 반복 변동 범위를 먼저 추정하고, cross-backend 차이를 그다음 본다. 이 장의 비실행 단계에서는 실제 범위를 주장하지 않고, 어떤 checkpoint에서 어떤 tolerance를 측정할지 계약만 만든다. 이후 실험 결과에는 hardware, toolkit, dtype, graph mode, backend identity를 함께 기록한다.

### 11.14.6 코드 리뷰에서 바로 쓰는 질문 순서

PR이나 새 backend를 검토할 때 diff 줄부터 파고들면 전체 계약을 놓치기 쉽다. 먼저 변경 전후 logical tensor 의미를 한 문장으로 적는다. 다음으로 저장 shape와 local shape를 적는다. 세 번째로 dtype·accumulation·epsilon을 적는다. 네 번째로 alias와 last reader를 적는다. 다섯 번째로 loader와 forward가 공유하는 pack order를 적는다. 마지막으로 collective 전후 의미를 적는다.

norm fusion PR이라면 semantic nodes가 add→norm 순서를 유지하는지, residual output과 normalized output이 각각 어디에 쓰이는지 본다. in-place write가 새로 생겼다면 이전 buffer의 모든 reader가 write 전에 끝나는지 본다. fp32 accumulation을 유지하는지, Gemma scale 규약을 보존하는지, epsilon이 kernel op params에 전달되는지 본다.

linear packing PR이라면 destination ranges가 겹치거나 비는지 합을 검사한다. global width와 local width를 혼용하지 않는지 본다. quant metadata loader가 payload 변경을 따라가는지 본다. forward split이 loader와 같은 canonical offsets를 사용하는지 본다. 가능하면 offsets 계산을 한 helper로 공유하되, 공유했다는 사실만으로 config-derived 숫자의 정확성이 증명되지는 않는다.

TP PR이라면 input replicated/partitioned 상태, output local/global 상태를 함수 signature나 타입만큼 명시적으로 다룬다. collective를 제거했다면 다음 consumer가 shard를 받을 수 있음을 증명한다. collective를 늦췄다면 residual이나 bias가 reduction에 중복되지 않음을 증명한다. reduce-scatter로 바꿨다면 후속 norm이 hidden 축 전체 통계를 어떻게 얻는지 확인한다.

리뷰 승인은 단위 test 이름의 개수보다 invariant coverage를 본다. non-square orientation, unequal Q/KV widths, KV replication, gate/up asymmetry, Gemma zero weight, tiny epsilon row, quant group boundary, TP shard boundary를 포함해야 한다. 모두 작은 fixture로 만들 수 있으며 실제 대형 모델 실행이 없어도 많은 구조 오류를 잡는다.

### 11.14.7 비실행 검토와 실행 검증의 경계

이 장은 모델·서버·CUDA를 실행하지 않는다. 그래서 source에서 확정할 수 있는 것과 실제 trace가 필요한 것을 구분한다. 확정 가능한 것은 class construction, config field 전달, 수식 순서, parameter shape 식, loader mapping, static offsets, collective helper 호출, fusion eligibility 조건이다. 실제 실행이 필요한 것은 선택된 backend symbol, 실측 stride가 dynamic path에서 바뀌는지, numeric error 분포, latency와 memory traffic이다.

비실행 검토가 약한 것은 아니다. 오히려 “옵션을 켰다”와 “kernel이 선택됐다”, “source에 fused op가 있다”와 “이 shape가 eligibility를 통과했다”를 분리해 과장된 결론을 막는다. 후속 실행자가 무엇을 기록해야 하는지도 선명해진다. source inspection 결과는 expected state, runtime trace는 observed state로 두고 둘의 차이를 조사한다.

어떤 source link도 영구 진리는 아니다. 이 장은 앞에서 밝힌 고정 revision의 계약을 설명한다. 최신 release로 이동할 때는 같은 의미 좌표를 다시 감사한다. 파일명이 같아도 class가 이동하거나 loader가 통합될 수 있다. 고정 commit 링크는 독자가 이 설명을 재현하게 하고, release note는 이후 변화의 시작점을 알려 준다.

### 11.14.8 종합 종이 실습: 한 layer의 projection 장부를 닫는다

마지막으로 실행 없이 풀 수 있는 종합 문제를 보자. hidden size 12, query heads 6, KV heads 2, head dimension 2, intermediate size 20, TP=2인 decoder layer가 있다. activation은 bf16, norm accumulation은 fp32, epsilon은 `10^-6`, projection bias는 없다고 하자. norm weight는 표준 `w` 규약이고 QKV와 gate-up은 packed parameter다.

첫째 global shape를 계산한다. residual과 normalized tensor는 `[T,12]`다. Q width는 `6×2=12`, K와 V width는 각각 `2×2=4`다. packed QKV global width는 20이므로 output은 `[T,20]`이다. gate와 up은 각각 20이므로 packed gate-up은 `[T,40]`이다. O projection은 query head output width 12를 model width 12로 보내고, down projection은 intermediate width 20을 model width 12로 보낸다.

둘째 rank-local shape를 계산한다. query heads는 rank당 3개여서 local Q width 6이다. KV heads는 rank당 하나여서 K와 V는 각각 2다. local packed QKV width는 10이고 offsets는 Q `[0,6)`, K `[6,8)`, V `[8,10)`이다. gate와 up은 각각 local intermediate width 10을 가져 local packed width 20이다. activation product 뒤 `[T,10]`이 down projection의 local input이다.

셋째 weight의 logical/stored shape를 구분한다. 수학적 packed QKV weight는 `[12,20]`이지만 PyTorch식 stored weight는 `[20,12]`이다. rank-local stored parameter는 output slice를 나누므로 `[10,12]`이다. O projection stored global weight는 `[12,12]`이고 row-parallel local stored weight는 input-axis 절반을 가져 `[12,6]`이다. down projection stored global weight는 `[12,20]`, local은 `[12,10]`이다.

넷째 collective를 표시한다. packed QKV와 gate-up은 column-parallel local output을 다음 head/activation 연산이 그대로 소비하므로 즉시 gather할 필요가 없다. O와 down projection은 row-parallel local partial `[T,12]`을 만들고 sum collective 뒤 global 의미의 `[T,12]` branch를 만든다. residual add는 이 합이 정확히 한 번 만들어진 뒤 수행되어야 한다.

다섯째 norm 손계산 fixture를 넣는다. hidden row의 모든 원소가 2이고 weight가 모두 1이면 mean square는 4, inverse RMS는 `1/sqrt(4+10^-6)`로 0.5에 아주 가깝다. affine 후 각 원소는 약 1이다. LayerNorm이라면 constant row를 center해 거의 zero가 되므로, 이 fixture 하나로 norm 종류를 구분할 수 있다. weight를 모두 0으로 바꾸면 표준 규약 출력은 zero이고 Gemma식 `1+w`라면 약 1이 남는다.

여섯째 pack 표식을 넣는다. local Q 여섯 칸은 100대, K 두 칸은 200대, V 두 칸은 300대로 둔다. gate 열 칸은 400대, up 열 칸은 500대로 둔다. loader destination ranges와 forward split을 종이에 적용한다. 어느 표식도 다른 그룹에 나오지 않아야 한다. rank 0과 rank 1의 global source ranges도 별도로 적는다.

일곱째 stride와 alias를 묻는다. norm input과 residual output이 같은 storage를 공유할 수 있는가, normalized output은 별도 storage인가, packed slice는 base packed output의 view인가를 source에서 확인한다. shape만으로 답하지 않는다. fused residual write 이후 옛 residual을 읽는 consumer가 없는지도 확인한다.

여덟째 failure branch를 만든다. norm constant-row 결과가 zero라면 잘못된 LayerNorm 선택을 의심한다. norm은 맞고 K 표식이 300대라면 packed identifier를 의심한다. local O partial은 맞고 collective 후 값이 절반이면 average reduction을 의심한다. fp16은 맞고 int4만 틀리면 effective weight와 metadata shard를 의심한다. TP=1은 맞고 TP=2만 틀리면 local source ranges와 collective를 의심한다.

이 종이 실습을 새 모델 config에 적용해 모든 칸을 채울 수 있다면 이 장의 핵심을 익힌 것이다. 숫자는 모델마다 달라져도 절차는 바뀌지 않는다. global 의미를 먼저 세우고, stored orientation을 분리하고, local shard를 계산하고, pack identity와 collective를 연결하고, first divergence로 owner를 고른다.

### 11.14.9 출처를 인용할 때 지켜야 할 정밀도

소스 링크는 주장 바로 옆에 둔다. 넓은 파일 링크 하나로 여러 주장을 덮지 않는다. 예를 들어 norm 수식 링크로 TP collective까지 증명할 수 없고, linear constructor 링크로 실제 model loader order까지 증명할 수 없다. 각 주장은 그것을 구현하거나 호출하는 가장 가까운 범위를 가져야 한다.

줄 범위도 역할이 있다. class 선언과 constructor는 설정·parameter shape를 뒷받침한다. forward 범위는 input/output과 collective를 뒷받침한다. weight loader 범위는 checkpoint slice를 뒷받침한다. CUDA launcher는 graph tensor가 kernel 인자로 변환되는 방식을 뒷받침한다. kernel body는 reduction과 dtype 연산을 뒷받침한다. 이 역할을 섞지 않으면 독자가 링크를 열었을 때 바로 근거를 찾는다.

문서와 source가 다르면 고정 revision source의 실제 동작을 우선 기록하되, 문서가 약속한 public contract와의 차이를 별도 표시한다. source에 경로가 존재한다는 것만으로 default 선택이라고 쓰지 않는다. build flag와 runtime eligibility가 남아 있으면 조건부 경로라고 쓴다. 논문이 fused norm의 일반적 이득을 보고했더라도 이 revision·shape의 실측 이득으로 옮겨 쓰지 않는다.

마지막으로 고정 링크가 20개를 넘는다는 사실 자체는 품질이 아니다. 이 링크들이 config에서 tensor와 kernel까지 끊기지 않는 인과 사슬을 만들어야 한다. 독자는 이 사슬을 이용해 새로운 revision에서 같은 symbol을 찾고, 달라진 규약을 비교하고, 자신의 장애에서 첫 차이를 좁힐 수 있어야 한다.

실무 인계 메모에는 결론만 쓰지 않는다. 재현에 쓴 revision과 config, 예상한 global·local shape, 확인한 loader destination, first divergence 직전과 직후 checkpoint, 기각한 경쟁 가설, 아직 관찰하지 못한 runtime 항목을 남긴다. “TP에서 틀린다”보다 “rank-local norm과 Q slice는 일치하지만 rank 2의 K destination identity가 기대 global head와 다르며 collective 전부터 차이가 난다”가 훨씬 강한 인계다. 다음 조사자는 이미 기각된 norm이나 NCCL을 반복하지 않고 정확한 loader branch에서 시작할 수 있다. 수정 검증에도 같은 문장을 뒤집어 쓴다.

해당 destination identity가 기대 head와 일치하고 local K slice가 reference와 맞으며 인접한 TP·KV-head 조합에서도 first divergence가 재발하지 않아야 닫는다. 이런 기록 습관이 수식, source, tensor 관찰을 하나의 검증 가능한 설명으로 묶는다.

### 11.14.10 이 장의 출구 질문

1. norm 통계가 줄이는 정확한 축과 accumulation dtype을 말할 수 있는가?
2. epsilon과 learned weight가 수식의 어느 위치에 들어가는가?
3. residual add 뒤 어느 buffer가 다음 layer까지 살아남는가?
4. weight의 logical axes와 checkpoint stored axes가 어떻게 대응하는가?
5. Q·K·V와 gate·up의 global/local offsets를 config에서 계산할 수 있는가?
6. column-parallel과 row-parallel에서 collective가 각각 어디에 필요한가?
7. quant payload와 scale metadata가 같은 logical shard를 가리키는지 증명할 수 있는가?
8. NaN·precision·pack·TP 장애의 first divergence를 어느 checkpoint에서 나눌 것인가?

여덟 질문에 답하지 못하면 다음 장의 attention score를 디버깅할 준비가 덜 된 것이다. attention은 Q와 K가 의미 있게 만들어졌다는 전제에서 시작한다.

normalized residual과 packed QKV가 맞으면 12장으로 간다. gate-up projection까지 맞으면 activation과 elementwise gating은 15장으로 넘긴다. first divergence가 residual add 이전이면 10장으로 돌아간다. 이 경계가 있어야 norm, GEMM, attention, MLP를 한 “forward 오차”로 뭉개지 않는다.

이 handoff를 tensor로 다시 쓰면 간단하다. 12장은 검증된 Q/K/V slice와 head metadata를 입력으로 받는다. 15장은 검증된 gate/up slice와 down-projection 계약을 입력으로 받는다. CUDA kernel 장은 여기서 확정한 logical shape·stride·dtype를 operand 계약으로 받는다. 분산 장은 collective group과 rank-local/global 의미를 받는다. 각 장이 자기 경계만 책임질 때 깊게 내려가도 길을 잃지 않는다.
