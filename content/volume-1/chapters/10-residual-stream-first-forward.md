# 10장. residual stream을 따라가는 첫 forward

첫 forward를 이해하려면 module 이름보다 한 tensor를 붙잡아야 한다. token embedding에서 시작한 hidden tensor는 각 decoder layer에서 normalization, attention, MLP를 거치면서도 residual stream이라는 주된 운반 경로를 유지한다. 이 경로의 shape는 대체로 `[token_rows, hidden_size]`지만, 값과 dtype, device, request ownership은 연산마다 달라질 수 있다.

9장은 input ID와 position이 첫 layer 입력이 되는 경계를 만들었다. 이 장은 그 tensor를 받아 한 pre-norm decoder layer와 전체 layer list를 통과시킨다. attention score와 mask kernel의 세부는 13장, rotary coordinate는 14장에 맡긴다. 마지막 hidden은 16장의 LM head로 넘긴다. 여기서는 “어느 tensor가 누구의 residual인가”를 잃지 않는 것이 목표다.

## 10.1 token row 하나가 첫 layer에서 두 번 갈라지고 다시 합쳐지는 장면

요청 A와 B가 막 첫 layer에 들어왔다. A의 마지막 prompt token은 physical row 0, B의 마지막 prompt token은
row 1에 pack됐다. 두 row는 embedding을 지나 hidden width 네 칸의 연속값이 됐다. 이제 attention과 MLP가 이 값을
각자 바꾸지만, 둘 다 원본을 통째로 대체하지는 않는다. 원본을 운반하는 residual identity와 정규화된 branch가
갈라졌다가 update가 다시 원본 좌표로 돌아와 합쳐진다.

설명을 위해 embedding 뒤 residual을 다음처럼 두자.

```text
X0 = [[ 1.0,  2.0, -1.0, 0.0],
      [ 0.5, -0.5,  1.5, 2.0]]
```

첫 row는 request A의 logical position 2, 둘째는 B의 position 1이다. 첫 row `[1,2,-1,0]`에 scale weight가
모두 1인 RMSNorm을 적용하고 설명을 위해 epsilon을 생략하면 mean square는 1.5, RMS는 약 1.225다. Attention이
읽는 normalized branch `N0`는 약 `[0.816,1.633,-0.816,0]`이다. 이때 `X0`는 사라지지 않는다. `N0`는 Q·K·V를
만드는 쪽으로 가고, `X0`는 skip path에서 attention update가 돌아오기를 기다린다.

Attention output projection이 첫 row에 `A=[0.1,-0.2,0.3,0.4]`를 돌려줬다고 하자. 첫 합은
`Xa=X0+A=[1.1,1.8,-0.7,0.4]`다. `N0+A`가 아니다. 이제 `Xa`가 다시 normalize돼 MLP branch `Na`가 되고,
MLP down projection이 `M=[-0.05,0.2,0.1,-0.15]`를 반환하면 layer output은
`X1=Xa+M=[1.05,2.0,-0.6,0.25]`다. 한 layer의 핵심은 다음 두 줄이다.

```text
Xa = X0 + Attention(Norm(X0))
X1 = Xa + MLP(Norm(Xa))
```

변수 이름이 모두 `hidden_states`여도 `X0`, `N0`, `A`, `Xa`, `Na`, `M`, `X1`은 서로 다른 역할이다. 이 장은
이 일곱 값을 하나의 **layer transaction**으로 추적한다. Transaction 표는 별도의 workbook이 아니라 앞의 계산을
소스와 실행 상태에 연결하는 유일한 기록이다.

| 단계 | 예제 값 또는 shape | producer→consumer | 소유권·완료 | lifetime·bytes 질문 |
|---|---|---|---|---|
| `X0` | A row `[1,2,-1,0]` | embedding/이전 layer→첫 norm과 skip | request-position row, residual identity | 두 branch의 마지막 read까지 살아 있는가 |
| `N0` | `[0.816,1.633,-0.816,0]` | 첫 norm→QKV projection | branch temporary, residual 아님 | compute dtype과 cast는 무엇인가 |
| `A` | `[0.1,-0.2,0.3,0.4]` | attention output projection→첫 add | TP partial인가 global complete인가 | collective와 add 중 무엇이 먼저인가 |
| `Xa` | `[1.1,1.8,-0.7,0.4]` | 첫 add→둘째 norm과 skip | post-attention residual | `X0` storage와 alias하는가 |
| `Na`,`M` | norm branch, MLP update | 둘째 norm/MLP→둘째 add | expert permutation 뒤 원 row인가 | gate/up workspace와 async consumer는 무엇인가 |
| `X1` | `[1.05,2.0,-0.6,0.25]` | 둘째 add→다음 layer | layer output residual | hook·PP send·다음 layer 중 last consumer는 누구인가 |

두 요청의 유효 token이 3개와 2개이고 padding 없이 pack되면 실제 `X0`는 `[5,D]`다. Dense batch라면
`[2,S,D]`일 수 있다. 수학적 token이 같아도 physical row layout은 다르므로 transaction의 각 행에는
row→request→logical position mapping을 유지한다. 실제 BF16 값과 위 decimal 예도 구분한다. Decimal은 손계산,
dtype byte 또는 bounded digest는 실행 비교에 쓴다.

### 같은 `[T,D]`라도 residual identity와 완료 상태는 다르다

tensor parallel에서 residual은 rank마다 replicated일 수 있고, sequence parallel에서는 token/sequence 축 일부만 소유할 수 있다. row-parallel projection 뒤 all-reduce 또는 reduce-scatter가 끝나기 전 partial output은 아직 global residual에 더할 수 있는 완성값이 아닐 수 있다. `[5,D]`라는 같은 shape가 global complete와 rank partial을 구별하지 않는다.

pipeline parallel에서는 stage 0의 마지막 layer residual이 send buffer가 되어 stage 1로 이동한다. device와 owner가 바뀌고 microbatch identity가 붙는다. stage 1이 embedding을 다시 만들지 않고 intermediate tensor를 입력으로 받아야 한다. field→pipeline partition branch→intermediate tensor state→P2P 통신/latency effect→stage boundary checksum으로 설명한다.

### dtype는 transaction의 저장·계산·cast 세 칸이다

embedding/residual이 BF16이어도 norm statistics와 softmax, GEMM accumulator가 FP32일 수 있다. quantized weight는 INT4 packed storage지만 activation은 BF16이고 projection output은 다시 BF16일 수 있다. “모델 dtype BF16”은 모든 중간 tensor가 BF16이라는 뜻이 아니다.

각 checkpoint에는 storage dtype, compute/accumulator dtype, output cast를 따로 적는다. non-finite가 발생하면 cast 전 accumulator인지 cast 후 residual인지 찾아야 한다. BF16 residual에 큰 sublayer output을 더하면서 overflow하거나 작은 update가 round away될 수 있다.

## 10.2 model.forward는 layer list를 순서대로 호출한다

상위 model forward는 input IDs 또는 precomputed embeddings를 받고 position/mask/cache metadata를 준비한 뒤 decoder layer list를 순회한다. 이 구조는 단순해 보이지만 gradient checkpointing, output hidden states, pipeline stage, layer type 선택이 실제 호출과 activation lifetime을 바꾼다.

Transformers Qwen3.5의 [`model forward`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1502-L1589)는 embedding, cache/position 상태와 layer iteration 경계를 보여 준다. [`Qwen3_5DecoderLayer`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L740-L817)는 residual 저장, norm, attention/MLP, 두 residual add의 순서를 고정한다.

Gemma3의 [`Gemma3Model.forward`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L600-L665)와 [`Gemma3DecoderLayer.forward`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L386-L445)는 layer type별 attention와 normalization 배치를 읽는 기준점이다. Qwen과 이름이 비슷해도 norm 위치와 architecture-specific scaling을 source에서 확인한다.

### hidden-state 출력은 관측 기능이자 lifetime 변경이다

`output_hidden_states=True`가 layer 전후 tensor를 tuple에 보존하면 allocator가 원래 재사용할 수 있던 activation을 오래 붙잡는다. field가 append branch를 켜고 tensor reference state가 전체 forward lifetime까지 연장된다. 효과는 debugging 관측과 peak memory 증가다. 반증은 saved tensor 수, storage pointer와 peak allocation이다.

inference mode는 autograd graph를 만들지 않지만 반환 tuple이 storage lifetime을 연장하는 사실은 남는다. layer differential을 위해 모든 hidden을 production에서 항상 반환하지 말고 synthetic fixture나 sampling canary에서 제한한다. checksum과 bounded slice만 저장하는 hook도 고려한다.

gradient checkpointing은 training 관심사지만 같은 model forward source에 branch가 있다. inference에서는 꺼져 있어야 하며, 이 책의 serving trace에서 activation recomputation을 기본 경로로 설명하지 않는다. config에 존재하는 것과 active branch를 구별한다.

## 10.3 pre-norm layer의 첫 절반은 attention update다

입력 residual `X`에 RMSNorm을 적용한다고 하자. row vector `x∈R^D`는 `rms(x)=sqrt(mean(x²)+ε)`, `n = x/rms(x) ⊙ g`로 계산한다. LayerNorm과 달리 평균을 빼지 않는 RMSNorm이 흔하다. architecture가 bias나 unit offset을 쓰는지는 source를 본다.

첫 예의 row `[1,2,-1,0]`, `g=1`, ε를 생략하면 mean square는 `(1+4+1+0)/4=1.5`, RMS는 약 1.225다. normalized row는 약 `[0.816,1.633,-0.816,0]`이다. 이 값은 residual을 대체하지 않는다. attention branch 입력으로 사용되고 원래 X는 skip path에 남는다.

Transformers Gemma3의 [`RMSNorm`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L124-L151)은 FP32 normalization과 output dtype 복귀를 확인할 수 있다. Qwen3.5의 normalization 구현도 architecture module에서 실제 cast와 weight convention을 읽는다.

### QKV projection은 residual D를 head 좌표로 바꾼다

normalized hidden `[T,D]`에 Q, K, V weight를 곱한다. query heads `Hq`, KV heads `Hkv`, head dimension `Dh`라면 Q는 `[T,Hq,Dh]`, K/V는 `[T,Hkv,Dh]`로 해석된다. GQA에서는 Hkv가 Hq보다 작다. fused QKV weight 하나라도 logical slices의 owner와 offset은 남는다.

Q는 현재 query를, K/V는 cache에 저장하거나 attention consumer로 보낼 state를 만든다. rotary가 Q/K에 position을 적용하는 구체 수학은 14장이다. 여기서는 Q/K의 position metadata와 V, residual row ownership이 같은 request 순서를 유지하는지 확인한다.

Qwen3.5 attention의 [`forward 경계`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L432-L520)는 projection, reshape, rotary/cache, attention interface, output projection을 잇는다.

Gemma3의 [`attention forward`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L341-L385)는 architecture별 query scaling과 window/layer metadata를 비교하는 기준이다.

### attention output은 아직 residual update 후보다

attention backend는 query row별 value의 weighted combination을 만들어 `[T,Hq,Dh]`를 반환한다. head 축을 합쳐 `[T,D_attn]`로 만들고 output projection `Wo`가 residual width D로 되돌린다. residual add를 하려면 shape, dtype, device와 parallel completeness가 X와 호환되어야 한다.

TP column-parallel QKV는 head를 rank에 나누고 local attention을 계산할 수 있다. output projection은 row-parallel이 되어 local partial `[T,D]`를 만든 뒤 all-reduce한다. collective 전 partial을 X에 각 rank에서 더하고 이후 reduce하면 X가 P번 합산될 수 있다. 구현이 residual을 collective 전후 어디에서 fuse하는지 읽는다.

완성 attention update를 `A`라 하면 `Xa=X+A`다. 작은 예에서 `A=[[0.1,-0.2,0.3,0.4],[0,0.5,-0.5,0.25]]`라면 첫 residual checkpoint는 `[[1.1,1.8,-0.7,0.4],[0.5,0,1,2.25]]`다. Transaction의 A 행에는 global complete 여부와 add output이 X storage를 alias하는지 기록한다.

## 10.4 두 번째 절반은 MLP 또는 MoE update다

pre-norm layer는 `Xa`를 다시 normalize한 뒤 feed-forward branch로 보낸다. gated MLP의 전형은 `u=W_up n`, `g=W_gate n`, `m=W_down(activation(g)⊙u)`다. intermediate width `I`가 D보다 크므로 gate/up activation `[T,I]`가 layer peak memory에 중요하다.

activation이 SiLU이면 `silu(x)=x·sigmoid(x)`다. gate와 up의 elementwise product는 두 projection이 서로 다른 역할을 가진다는 뜻이다. fused gate-up GEMM은 output `[T,2I]`를 한 번에 만들고 split한다. 이름이 fused여도 logical tensor와 quant scale 범위를 transaction의 branch 행에 남긴다.

Qwen3.5의 [`decoder layer MLP 경로`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L740-L817)에서 post-attention norm과 MLP residual add를 확인한다. Gemma3의 [`MLP`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L153-L180)는 gate/up/down과 activation 순서를 보여 준다.

### tensor parallel MLP도 partial과 complete를 구별한다

gate/up을 column parallel로 나누면 rank마다 intermediate slice `[T,I/P]`를 계산한다. elementwise activation/product는 local로 가능하다. down projection은 row parallel partial `[T,D]`를 만들고 collective로 합친다. attention과 마찬가지로 residual add fusion이 collective semantics와 맞아야 한다.

field가 reduce-results false branch를 고르면 caller가 partial을 책임질 수 있다. state는 rank-local output과 deferred reduction handle이다. 효과는 communication fusion/overlap 가능성과 alias 위험이다. 반증은 collective trace, rank별 partial sum과 global reference, residual checksum이다.

### MoE는 token ownership을 잠시 바꾼다

MoE layer는 router logits `[T,E]`에서 top experts를 고르고 token rows를 expert owner rank로 dispatch한다. expert MLP 뒤 weighted combine과 inverse permutation으로 원 residual row 순서를 복원한다. residual add 전 `[T,D]` shape가 돌아왔다고 request ordering까지 맞았다는 보장은 없다.

router와 dispatch 상세는 별도 장에서 확대하되 이 장의 invariant는 명확하다. Input row identity, expert assignment,
dispatched row, inverse row를 M 행의 owner 변화로 연결한다. Dropped token, capacity 정책, shared expert가 있으면 update
의미가 달라진다. Expert output이 0이어도 residual skip path는 살아 있어 layer가 finite하게 끝날 수 있다.

두 번째 update `M`을 더해 `X1=Xa+M`을 만든다. 이 X1이 다음 layer의 residual input이다. MLP temporary를 다음 residual로 오인하거나 norm output을 checkpoint하면 layer differential이 잘못된 지점을 비교한다.

## 10.5 한 층을 숫자로 끝까지 걸으면 이름보다 불변식이 보인다

지금까지는 연산을 조각으로 보았다. 이제 아주 작은 dense layer 하나를 끝까지 걸어 보자. 실제 모델보다 터무니없이 작지만, 작은 예의 목적은 성능을 흉내 내는 것이 아니라 어느 값이 보존되고 어느 값이 새로 만들어지는지 손으로 확인하는 데 있다. token row 두 개와 hidden width 네 개를 그대로 사용한다. 첫 residual `X0`의 첫 row는 `[1, 2, -1, 0]`이고, 첫 normalization 결과를 반올림해 `N0=[0.816, 1.633, -0.816, 0]`이라고 두었다. attention 내부 Q·K·V 계산은 12장과 13장의 몫이므로 여기서는 그 결과가 output projection까지 통과한 완성 update `A=[0.1,-0.2,0.3,0.4]`라고 놓는다.

첫 합은 `Xa=X0+A=[1.1,1.8,-0.7,0.4]`다. 중요한 점은 `N0+A`가 아니라 `X0+A`라는 사실이다. norm은 branch가 읽는 값이고 skip path가 운반하는 기준값은 X0다. debugger에서 `hidden_states`라는 변수 이름만 좇다 보면 norm 뒤 변수와 residual 변수의 이름이 재사용되어 이 차이를 놓치기 쉽다. 그래서 checkpoint 이름을 Python 변수 이름이 아니라 의미로 붙인다. `L0.input.residual`, `L0.attn.normalized`, `L0.attn.update.complete`, `L0.post_attn.residual`처럼 붙이면 구현이 local variable을 재할당해도 비교 좌표가 흔들리지 않는다.

두 번째 norm이 `Na≈[0.959,1.569,-0.610,0.349]`를 만들었다고 하자. 설명용 MLP에서 gate projection 결과가 `[1,-1,0.5,2]`, up projection이 `[0.2,0.4,-0.6,0.5]`라고 두면 SiLU gate와 up의 곱은 대략 `[0.146,-0.108,-0.187,0.881]`이다. down projection이 이를 residual width로 보내 `M=[-0.05,0.2,0.1,-0.15]`를 만들었다면 layer output은 `X1=Xa+M=[1.05,2.0,-0.6,0.25]`다. 이 숫자에서 attention과 MLP update가 residual보다 작아야 한다는 보편 법칙을 도출해서는 안 된다. 다만 두 update가 서로 다른 normalized input에서 계산되고, 각각 원래 residual 좌표로 돌아온 뒤 더해진다는 순서를 확인할 수 있다.

두 row를 한꺼번에 계산하면 row identity 불변식도 보인다. dense attention은 서로 다른 request 사이를 보게 해서는 안 된다. packed physical tensor가 `[A0,A1,A2,B0,B1]` 순서라면 mask와 sequence boundary가 A와 B를 분리한다. MLP는 row별 연산이므로 permutation을 하지 않는 한 같은 physical order를 유지한다. MoE는 dispatch 때문에 잠시 순서를 바꾸지만 combine 뒤에는 반드시 이 order를 복원한다. 따라서 layer checkpoint 비교에서 단순한 전체 평균은 약하다. 두 request row가 뒤바뀌어도 평균과 norm이 비슷할 수 있다. request ID, logical position, layer ID를 key로 삼아 row별 digest를 비교해야 한다.

### transaction의 shape·row 열을 실제 크기로 확장한다

hidden width `D=4096`, query head 32개, KV head 8개, head dimension 128, MLP intermediate width 14336인 실제적인 예를 붙여 보자. packed token이 37개라면 residual은 `[37,4096]`, Q는 `[37,32,128]`, K/V는 각각 `[37,8,128]`이다. Q의 element 수는 151,552개이고 K와 V는 각각 37,888개다. BF16 storage만 단순 계산하면 Q 약 296 KiB, K와 V 각각 약 74 KiB다. 이 수치는 weight, allocator alignment, workspace, cache page, accumulator를 제외한 임시 activation 하한일 뿐이다.

head output을 합치면 `[37,4096]`이고 O projection 뒤에도 `[37,4096]`이다. MLP gate와 up은 각각 `[37,14336]`, 즉 530,432 element다. 두 tensor가 별도로 materialize되면 BF16 기준 각각 약 1.01 MiB다. fused implementation은 둘을 packed output 하나로 만들거나 activation과 multiply를 이어 임시 lifetime을 줄일 수 있다. 그렇다고 logical gate와 up이 사라지는 것은 아니다. 정답 차이를 조사할 때 quantization scale, slice offset, activation order를 여전히 따로 확인한다.

shape ledger에는 네 종류의 축을 혼합하지 않는다. `T=37`은 이번 step에서 실제 계산할 token row 수다. `B=요청 수`와 같지 않다. `S=각 request의 logical sequence length`도 아니다. `D`는 residual width이고 `Hq×Dh`와 같을 때가 많지만 architecture가 projection width나 low-rank latent를 다르게 두면 자동으로 동일하다고 가정할 수 없다. `Hkv`는 cache 저장량과 직접 연결되지만 query compute의 head 수는 Hq다. 이 축들을 `batch size`라는 한 단어로 뭉개면 prefill과 decode 비용을 설명할 수 없다.

### transaction의 dtype 열은 저장·계산·cast를 나눈다

각 연산에 storage, compute, accumulation, output dtype을 적는 이유는 오차가 나타난 위치를 좁히기 위해서다. BF16 residual이 RMS 계산을 위해 FP32로 cast되고, normalized output은 다시 input dtype으로 내려올 수 있다. quantized linear의 weight는 packed INT4이지만 scale과 zero point, activation, accumulator, output dtype은 서로 다르다. attention score는 FP32 안정화가 개입할 수 있고 최종 update는 residual dtype으로 되돌아온다.

두 서버의 layer output이 처음 달라졌다고 해서 그 layer의 수학이 틀렸다는 뜻은 아니다. 한쪽 embedding이 BF16, 다른 쪽이 FP16이면 첫 layer 입구부터 미세한 차이가 있고 깊이를 거치며 증폭될 수 있다. 반대로 layer 0~17 checksum은 허용 오차 안인데 layer 18의 post-norm부터 NaN이라면 layer 18 norm 입력의 magnitude, variance reduction dtype, epsilon, fused residual 경계를 우선 조사한다. `allclose` 하나로 끝내지 말고 최대 절대 오차, 상대 오차, cosine, non-finite 위치, top-k logit 순위 보존을 목적에 맞게 선택한다.

### 이 예가 설명하지 못하는 것

작은 dense 예는 cache page slot, causal mask, online softmax, RoPE coordinate, TP collective, expert dispatch를 생략했다. 이 예로 backend의 bitwise equality나 kernel launch 수를 예측해서는 안 된다. decimal 값도 손으로 반올림했으므로 실제 dtype 결과의 oracle가 아니다. 이 예가 증명하는 범위는 pre-norm layer의 소유권과 합 순서, residual checkpoint의 의미, shape ledger 작성법이다. 좋은 비유와 작은 계산은 경계를 명시할 때만 친절하다.

## 10.6 prefill과 decode는 같은 layer를 다른 물리 모양으로 통과한다

“prefill 모델”과 “decode 모델”이 따로 있는 것은 아니다. 같은 parameter와 거의 같은 layer semantics를 사용하지만, 이번 step에 들어오는 query row 수와 이미 존재하는 KV 길이, scheduler가 섞은 request 구성이 다르다. 그래서 Python의 `forward` 이름이 같아도 kernel 선택, GEMM 모양, attention metadata, 병목이 달라진다.

prefill에서 길이 512인 요청 하나가 처음 들어오면 이번 query token 수 `Tq`가 512일 수 있다. residual은 `[512,D]`이고 projection GEMM의 M축도 512다. attention은 각 query position이 causal prefix를 보며 cache에는 512개의 K/V가 새로 기록된다. chunked prefill이 128 token씩 네 step으로 나누면 각 step residual은 `[128,D]`이지만 두 번째 chunk부터는 앞 chunk의 KV를 읽는다. logical forward는 한 prompt를 처리하지만 physical layer invocation은 여러 scheduler step으로 갈라진다.

decode에서는 활성 request 37개가 각각 새 token 하나를 계산하면 flattened residual이 `[37,D]`다. request마다 과거 길이는 20, 400, 8000처럼 다를 수 있다. query projection의 T는 37이지만 attention이 읽는 KV row 총량은 그 과거 길이들의 합에 가깝다. 따라서 residual shape만 보고 attention 비용을 판단하면 안 된다. ledger에 query rows, per-request context lengths, page table 또는 slot mapping, sequence boundary를 같이 남긴다.

mixed batch에서는 prefill chunk 128개와 decode row 30개를 같은 step이 처리할 수 있다. physical token tensor는 158 row일 수 있지만 attention metadata는 prefill과 decode query를 서로 다른 방식으로 해석한다. 어떤 backend는 공통 wrapper 아래에서 분리 kernel을 호출하고, 어떤 backend는 통합 가능한 구간만 합친다. “batch가 158”이라는 설명은 부족하다. 128개는 동일 request의 연속 position이고 30개는 서로 다른 request의 단일 position이라는 구성까지 말해야 한다.

### cache가 residual stream 밖에 있다는 말의 정확한 뜻

residual은 layer에서 다음 layer로 흘러가는 현재 token representation이다. KV cache는 attention layer가 과거 token의 K/V를 다음 step에 다시 읽기 위해 보존하는 길이 의존 state다. cache는 residual에 더해지는 update가 아니며, layer 사이가 아니라 같은 attention layer의 시간축 호출 사이를 연결한다. layer 7의 K/V는 다음 decode step의 layer 7이 읽고, layer 8 residual이 직접 읽는 것이 아니다.

이 구분은 취소와 slot 재사용에서 중요하다. request A가 쓰던 KV page가 B에 재할당되었는데 page table generation이나 logical length가 잘못되면 B의 residual row는 정상이어도 attention update가 오염된다. 반대로 residual buffer alias가 깨졌다면 cache checksum은 정상인데 layer 출력이 틀릴 수 있다. 장애 분기에서 `현재 step activation`과 `step 간 persistent state`를 별도 owner로 둔다.

Qwen3.5처럼 attention 계층과 선형 recurrent 계층을 혼합하는 모델은 이 지도를 확장한다. recurrent/conv state 역시 step 간 보존되지만 KV와 shape 및 update 법칙이 다르다. 이 장에서는 residual 공통 spine만 유지한다. 어떤 layer가 KV를 만들고 어떤 layer가 recurrent state를 만들지는 15장과 모델 수직 사례에서 분리한다. 모든 layer가 KV cache를 가진다는 가정은 hybrid 모델에서 이미 틀렸다.

### TTFT와 ITL을 layer 모양에 연결한다

TTFT가 길 때 “모델 forward가 느리다”는 말만으로는 조치할 수 없다. queue 뒤 실제 prefill layer의 token M축이 큰지, chunk 정책 때문에 몇 번 호출되는지, embedding과 LM head를 포함해 어느 stage가 시간을 쓰는지 확인해야 한다. projection과 MLP가 큰 GEMM으로 weight를 재사용하는 구간, attention이 긴 prompt의 score를 처리하는 구간, pipeline bubble과 collective를 구분한다.

ITL이 길 때는 각 decode step의 active row 수, context length 분포, KV page 접근, 작은 GEMM의 launch/weight-byte 비용, TP collective를 본다. 같은 request 수라도 긴 context가 많으면 attention read가 늘고, 완료 직전 request가 많아 batch가 빠르게 줄면 GEMM 효율이 달라진다. scheduler가 token을 선택한 결과가 바로 layer tensor shape가 되므로 scheduler metric과 runner checkpoint를 연결해야 한다.

### prefill/decode differential에서 비교할 좌표

첫째, 동일 prompt의 Transformers dense batch와 serving engine packed batch를 logical `(request,position,hidden)` 좌표로 다시 맞춘다. 둘째, embedding 직후, 첫 layer의 두 residual add 뒤, 중간 layer, final norm 전후를 비교한다. 셋째, prefill 마지막 position의 hidden과 그 뒤 첫 decode input/output을 연결한다. 넷째, cache length가 sampling으로 선택된 token 수와 맞는지 확인한다. 다섯째, padding row나 speculative token이 비교 집합에 섞이지 않았는지 확인한다.

prefill 전체 tensor를 그대로 덤프하는 것은 메모리와 개인정보 문제를 만든다. synthetic token fixture에서만 작은 slice를 저장하거나, production canary에서는 row별 finite count, norm, checksum, bounded projection을 사용한다. checksum이 같다는 사실은 semantic equality의 강한 신호지만 collision과 dtype metadata 누락 가능성이 있다. checksum이 다르면 곧바로 bug라고 하지 않고 layout permutation, 허용 numeric 차이, padding 포함 여부를 먼저 반증한다.

## 10.7 네 구현에서 같은 forward spine을 찾는 법

framework마다 이름과 abstraction이 달라도 독자가 찾아야 할 의미 좌표는 같다. 입력 token row가 embedding weight에서 residual이 되는 곳, layer container가 반복되는 곳, attention과 MLP update가 residual에 합쳐지는 곳, final normalization과 LM head로 넘어가는 곳이다. source atlas는 파일 목록이 아니라 이 의미 좌표를 구현별 symbol에 연결해야 한다.

**Transformers는 reference semantics를 읽기 좋다**

Transformers에서는 model class의 `forward`가 inputs embeds, cache position, mask를 준비하고 decoder layer를 순회한다. Qwen3.5 model loop는 [고정 revision의 model forward](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1502-L1589)에서

시작하고, 한 층의 두 residual 합은 [decoder layer](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L740-L817)에서 확인한다.

attention projection 경계는 [attention module](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L432-L520), final causal LM 경계는 같은 파일의 [LM wrapper](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1592-L1690)를 따른다.

Gemma 계열을 옆에 놓는 이유는 이름이 같다고 순서까지 같다고 믿지 않기 위해서다. Gemma3의 [model loop](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L600-L665), [decoder layer](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L386-L445),

[RMSNorm](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L124-L151), [MLP](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L153-L180)을 각각 같은 의미 좌표에 놓는다. architecture-specific pre/post feed-forward normalization이나 scaling이 있으면 “전형적인 Llama layer”로 덮지 않는다.

Transformers는 serving oracle로 유용하지만 자동으로 절대 oracle이 되는 것은 아니다. attention implementation 선택, dtype, cache class, model config, tokenizer input이 같아야 비교가 의미 있다. eager와 SDPA가 허용 오차 안에서 다를 수 있고, 서버가 quantized artifact를 쓰면 full precision 기준과의 차이를 계약으로 정의해야 한다. reference는 비교 기준이지 무조건 bitwise 정답이라는 호칭이 아니다.

**vLLM은 residual과 pipeline ownership을 명시한다**

vLLM의 Qwen3 구현에서 [decoder layer 정의](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3.py#L173-L263)는 attention과 MLP 경로를 serving용 parallel layer로 구성한다.

[Qwen3 model wrapper](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3.py#L264-L270)와 [causal LM wrapper](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3.py#L271-L343)는 embedding/model/logits 경계를 잇는다.

실제 공통 layer loop와 PP 처리는 상속한 Qwen2 경로의 [model class](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen2.py#L322-L440)까지 내려가 읽어야 한다.

Qwen3.5는 단순히 Qwen3 파일의 이름만 바꾼 모델이 아니다. vLLM의 [Qwen3.5 decoder layer](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5.py#L118-L214)는 hybrid layer 구성을 상속·변형하고, [model](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5.py#L215-L285)과

[causal LM base](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5.py#L286-L421)가 pipeline과 logits 경계를 제공한다. 여기서 이 장이 가져갈 것은 residual spine이며, recurrent state의 세부는 15장으로 넘긴다.

vLLM decoder layer API가 `hidden_states`와 별도 `residual`을 주고받는다면 이를 단순한 코딩 취향으로 읽지 않는다. fused RMSNorm이 add와 normalization을 묶을 때 skip tensor의 storage를 재사용하고 다음 residual을 별도 반환할 수 있다. 첫 layer에서는 residual이 `None`이고 이후에는 두 tensor 계약이 유지되는 경로도 있다. 변수 두 개가 수학의 residual 두 개를 뜻한다고 단정하지 말고, 각 함수 반환값이 pre-add 값인지 post-add 값인지 source와 작은 static trace로 확인한다.

pipeline stage가 embedding을 소유하지 않으면 intermediate tensor를 입력받고, 마지막 stage가 아니면 logits 대신 intermediate를 반환한다. 이 분기는 모델 수학을 바꾸지 않지만 tensor owner와 lifetime, 통신 경계를 바꾼다. 따라서 PP differential은 global layer 번호를 key로 하고 stage local index만 기록하면 안 된다. send 전 checksum과 receive 후 checksum, dtype, shape, microbatch ID를 묶어야 한다.

**SGLang도 runner와 model의 책임을 분리한다**

SGLang의 serving model은 request batch metadata를 받아 model-specific forward로 보낸다. Qwen3의 [decoder layer](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3.py#L310-L434), [model](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3.py#L435-L451),

[causal LM wrapper](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3.py#L452-L560)를 의미 좌표로 읽는다. 실제 line은 고정 revision에서 다시 검산해야 하며, class 이름만 현재 branch 검색 결과로 대체해서는 안 된다.

hybrid Qwen 경로는 [linear decoder layer](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_next.py#L507-L597), [attention decoder layer](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_next.py#L598-L730), [model

loop](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_next.py#L881-L989), [LM wrapper](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_next.py#L990-L1080)를 함께 본다. layer type branch를 빼고 attention layer만 checkpoint하면 첫 divergence가 다른 family에 있어도 놓친다.

runner가 만든 `forward_batch` 또는 attention metadata는 model input의 일부다. token IDs가 같아도 positions, cache location, extend/decode mode가 다르면 같은 forward 계약이 아니다. SGLang과 vLLM을 비교할 때 내부 object 이름을 억지로 맞추지 말고 logical fields를 맞춘다. `(request incarnation, logical position, layer, token row, cache length, selected backend)`가 비교 key다.

**llama.cpp는 eager Python 호출 대신 계산 그래프를 읽는다**

llama.cpp에서는 `DecoderLayer.forward`라는 Python symbol을 찾는 방식이 통하지 않는다. model architecture별 builder가 ggml tensor node를 만들고 graph가 backend에서 실행된다. 공통 입구인 [llama_model::build_graph](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model.cpp#L2461-L2550)와

호출 측 [context graph build](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-context.cpp#L1358-L1430)를 시작으로, architecture builder에서 embedding node, norm, attention, feed-forward, residual add node를 찾는다. 이 좌표는 vendored v0.2.0 commit에 고정되어 있다.

ggml tensor의 차원 순서는 PyTorch 표기와 바로 같지 않을 수 있다. row와 column의 의미, `ne[]` 차원, transpose/view/contiguous node를 확인한 뒤 logical `[T,D]`로 번역한다. 그래프 node name이 있으면 layer index와 의미 checkpoint를 연결하지만, node name만으로 buffer alias와 backend placement를 단정하지 않는다. graph allocator가 서로 lifetime이 겹치지 않는 tensor에 같은 storage를 재사용할 수 있기 때문이다.

llama.cpp 비교의 장점은 operator graph와 buffer planning을 함께 볼 수 있다는 점이고, 함정은 source-level tensor가 실행 시 backend fusion으로 그대로 남는다고 생각하는 것이다. graph에서 norm, mul_mat, add node가 보이더라도 CUDA backend가 어떤 kernel로 합치는지는 별도 층이다. 이 장의 residual semantic checkpoint와 CUDA kernel launch checkpoint를 1:1로 억지 대응시키지 않는다.

## 10.8 병렬 경계와 activation lifetime이 정답과 성능을 함께 바꾼다

한 GPU의 dense reference에서 맞던 residual 식이 여러 rank에서도 맞으려면 partial tensor를 합치는 순서가 정확해야 한다. tensor parallel은 parameter와 activation 일부를 rank에 나누고 collective로 global 의미를 복원한다. pipeline parallel은 연속 layer 묶음을 stage에 나누고 residual activation 자체를 다음 device로 보낸다. 두 방식 모두 최종 수학을 보존하려 하지만, 소유권과 실패 양상은 다르다.

### column parallel과 row parallel을 식으로 읽는다

linear `Y=XW`에서 W의 output column을 P개 rank로 나누면 `W=[W1 … WP]`이고 각 rank는 `Yi=XWi`를 만든다. 이것이 column parallel이다. QKV나 MLP gate/up처럼 output feature를 head 또는 intermediate slice로 나누기 좋다. rank local tensor는 global Y의 서로 다른 slice이므로 같은 위치끼리 더하는 all-reduce가 필요하지 않다. 다음 연산도 해당 slice에서 local로 진행할 수 있다.

반대로 W의 input row를 나누면 `W=[W1;…;WP]`, X도 feature slice `Xi`로 나뉘고 각 rank는 partial `Pi=XiWi`를 만든다. global output은 `Y=ΣPi`다. 이것이 row parallel의 핵심이다. O projection과 MLP down projection 뒤에 reduction이 필요한 이유다. reduce-scatter를 사용하면 global Y가 token 또는 sequence 축으로 shard된 형태로 돌아올 수 있으므로 residual도 같은 shard 계약이어야 한다.

예를 들어 TP=2에서 rank 0 partial A0와 rank 1 partial A1이 있고 residual X가 두 rank에 replicated되어 있다고 하자. 올바른 결과는 `X+(A0+A1)`이다. 각 rank가 먼저 `X+Ai`를 만든 뒤 all-reduce하면 `2X+A0+A1`이 되어 틀린다. 다만 구현이 reduce 연산을 평균으로 정의하거나 residual을 한 rank에만 넣는 특수 fusion을 쓰면 식이 달라질 수 있다. 그래서 함수 이름이 `all_reduce`인지보다 collective 전 input에 무엇이 들어갔는지를 확인한다.

### sequence parallel에서는 transaction의 row identity가 더 중요하다

sequence parallel은 residual의 token row를 rank에 나눠 norm이나 elementwise 연산 메모리를 줄일 수 있다. QKV projection 전후에 gather 또는 reduce-scatter가 배치될 수 있다. local shape가 `[T/P,D]`라고 해서 request가 균등하게 나뉘는 것은 아니다. packed token의 shard boundary가 한 request 중간을 가를 수 있고, ragged distribution이면 rank별 T도 다를 수 있다.

layer checkpoint를 rank별 배열 순서로 비교하지 말고 global token index로 복원한다. global index는 scheduler의 packed row index만으로 충분하지 않을 수 있다. request가 step마다 다른 row를 차지하므로 request incarnation과 logical position을 포함한다. gather 뒤 checksum이 맞는데 local norm부터 틀리면 shard mapping, norm reduction 범위, padding 처리를 본다. local checkpoint가 맞고 gather 뒤만 틀리면 collective count/order와 stream dependency를 본다.

### pipeline parallel은 residual을 네트워크 메시지로 만든다

PP=2에서 layer 0~15를 stage 0, 16~31을 stage 1이 소유한다고 하자. stage 0은 embedding과 앞 layer를 실행한 뒤 `X16`을 보낸다. stage 1은 이 tensor를 자신의 `inputs_embeds`와 같은 의미로 받아 layer 16을 시작한다. final norm과 LM head는 마지막 stage에만 있을 수 있다. 첫 stage가 아닌 model forward에서 token IDs를 무시하거나 intermediate container를 요구하는 이유다.

메시지에는 적어도 microbatch, scheduler step, virtual engine 또는 pipeline slot, shape, dtype가 합의되어야 한다. 통신은 성공했지만 이전 microbatch tensor를 잘못 소비하면 shape는 맞고 결과만 간헐적으로 틀릴 수 있다. P2P send가 비동기라면 producer buffer를 너무 일찍 재사용하는 alias bug도 가능하다. send enqueue 시점이 아니라 통신 completion 이후에 storage lifetime이 끝나는지 확인한다.

PP bubble은 residual 계산식의 문제가 아니지만 latency를 해석할 때 model layer gap으로 나타난다. stage 1 trace가 비어 있다고 attention kernel을 최적화할 일이 아니다. upstream stage 또는 P2P wait, microbatch schedule을 확인한다. 반대로 stage 경계 checksum이 일치하고 stage 1 첫 norm 직후부터 값이 갈라지면 통신보다 stage 1 weight/dtype/backend를 우선한다.

### in-place는 변수 재할당과 다르다

Python에서 `hidden_states = hidden_states + update`는 새 tensor를 만들 가능성이 높고, `hidden_states += update` 또는 fused op는 storage를 수정할 수 있다. 그러나 compiler와 graph optimizer, custom op가 내부 buffer를 재사용하므로 surface syntax만으로 alias를 확정할 수 없다. storage pointer, view base, stride, version counter 또는 framework의 alias contract를 본다.

in-place residual add가 유리한 이유는 `[T,D]` buffer allocation과 memory traffic을 줄일 수 있기 때문이다. 위험은 skip path가 아직 필요하거나 다른 consumer가 같은 storage를 참조하는 상황이다. output hidden states tuple, hook, pipeline send, speculative branch가 오래된 X를 필요로 하면 덮어쓰기가 semantic bug나 관측 오염을 만든다. graph capture는 고정 address를 선호하지만 request incarnation 사이 값 격리를 더 엄격히 요구한다.

activation lifetime을 interval로 적어 보자. X는 attention norm과 residual add가 끝날 때까지 필요하다. normalized N은 QKV projection이 input을 모두 소비할 때까지, Q는 attention output이 끝날 때까지, K/V 새 row는 cache write completion까지, A partial은 collective completion까지 살아야 한다. Xa는 MLP norm과 두 번째 add까지, gate/up temporary는 down projection input 소비까지, X1은 다음 layer 또는 PP send 완료까지 필요하다. allocator는 interval이 겹치지 않는 storage만 안전하게 재사용할 수 있다.

비동기 CUDA 실행에서는 Python 함수 반환이 lifetime 끝을 뜻하지 않는다. kernel이 stream에서 아직 buffer를 읽고 있는데 CPU가 reference를 버려 allocator가 같은 block을 다른 tensor에 주면 stream ordering이 안전을 보장해야 한다. 서로 다른 stream이나 communication stream을 쓰면 event dependency가 필요하다. 간헐 오답, 특정 batch에서만 NaN, graph replay 때 cross-request contamination은 이 경계를 의심할 신호다. 이 장에서는 kernel을 실행하지 않지만 source의 stream/event와 tensor ownership을 정적으로 추적할 수 있다.

### quantization과 adapter도 residual contract를 통과해야 한다

weight-only quantization은 projection 내부의 weight representation과 kernel을 바꾸지만 output update는 residual add가 요구하는 `[T,D]`와 dtype 계약으로 돌아와야 한다. scale group이나 shard가 틀리면 QKV 또는 MLP update에서 첫 divergence가 난다. residual 자체를 INT4로 저장한다는 뜻은 아니다. KV cache quantization도 residual dtype과 별개다.

LoRA adapter는 base projection에 low-rank update를 더한다. adapter가 QKV, O, gate/up, down 중 어느 module에 붙었는지에 따라 첫 divergence checkpoint가 달라진다. request A와 B가 다른 adapter를 쓰는 mixed batch라면 row별 adapter identity가 projection consumer까지 보존되어야 한다. 전체 layer checksum만 비교하면 서로 다른 adapter row가 섞인 문제를 놓친다. adapter ID와 token row를 ledger key에 추가한다.

## 10.9 layer-checkpoint differential은 최초의 의미 불일치를 찾는다

전체 생성 문자열이 다르다는 사실은 출발점일 뿐이다. sampling은 작은 logit 차이를 다른 token으로 증폭하고, 한 token이 갈라지면 이후 cache와 모든 hidden이 달라진다. 따라서 첫 생성 이후를 비교하는 대신 같은 teacher-forced token IDs를 넣고 embedding부터 final hidden까지 최초로 허용 오차를 넘는 checkpoint를 찾는다. 이 절의 workbook은 실행 명령을 요구하지 않는다. 어떤 증거를 수집하고 어떻게 분기할지 설계한다.

### fixture를 얼린다

비교 전에 model artifact hash, config, tokenizer revision, chat template 결과 token IDs, position IDs, attention mask 의미, dtype, quantization, adapter, attention backend 요청과 실제 선택, TP/PP 크기를 기록한다. prefill과 decode를 별도 fixture로 만든다. prefill fixture는 짧고 사람이 token position을 확인할 수 있는 prompt를 쓰고, decode fixture는 고정된 past tokens와 다음 input token을 사용한다.

랜덤 sampling을 끄는 것만으로 충분하지 않다. 모델 입력이 같아야 한다. 서버가 BOS를 자동 추가했거나 padding side가 다르고 cache position이 어긋나면 model layer를 비교할 이유가 없다. 6~9장의 input contract가 일치한 뒤 이 장으로 들어온다. multimodal이면 placeholder와 embedding splice 위치도 포함하지만, 첫 dense text fixture를 먼저 통과시킨다.

checkpoint schema는 다음처럼 둔다.

```text
run_identity: artifact, revision, config, dtype, backend
row_identity: request_incarnation, logical_position, packed_row
tensor_identity: global_layer, semantic_point, rank, stage
layout: logical_shape, physical_shape, stride, shard_spec
numeric: finite_count, l2_norm, max_abs, mean, digest, bounded_slice
ownership: device, storage_id, alias_of, lifetime_event
```

semantic point는 최소 `embedding.output`, 각 layer의 `input.residual`, `attn.norm.output`, `attn.update.complete`, `post_attn.residual`, `mlp.norm.output`, `mlp.update.complete`, `layer.output.residual`, `final_norm.output`이다. 처음부터 모든 임시 tensor를 저장하지 않는다. coarse checkpoint로 layer를 찾고 그 layer에서만 norm/projection/head로 확대한다. 이것이 관측 비용과 개인정보 노출을 줄인다.

### coarse-to-fine 이분 탐색

32 layer 모델이라면 embedding, layer 7, 15, 23, 31, final hidden부터 비교한다. layer 15는 맞고 23이 다르면 16~23 중간을 비교한다. 최초 layer를 찾으면 그 입력이 같은지 확인하고 두 residual branch로 나눈다. attention update부터 다르면 norm/QKV/position/cache/backend를, attention까지 맞고 MLP update부터 다르면 norm/gate/up/activation/down/adapter를 조사한다.

입력 residual은 같지만 norm output이 다르면 epsilon, weight convention, compute dtype, fused residual ordering을 본다. norm은 같지만 QKV가 다르면 weight load/shard/quant scale/adapter를 본다. QKV는 맞지만 attention update가 다르면 position, mask, cache, backend numeric을 본다. attention update complete는 같은데 post-attention residual이 다르면 partial collective 또는 add/alias를 본다. MLP update complete는 같은데 layer output이 다르면 두 번째 add와 residual source를 본다.

첫 divergence가 곧 root cause인 경우가 많지만 언제나 그런 것은 아니다. checkpoint 자체가 다른 layout을 같은 순서로 hash했거나, observer가 graph path를 바꿨거나, 앞선 미세 오차가 threshold를 늦게 넘었을 수 있다. 직전 checkpoint에서 tighter metric과 bounded exact values를 확인하고, probe를 제거한 재현에서도 사용자 증상이 유지되는지 검증한다.

### rank와 stage를 포함한 판정표

모든 TP rank의 local Q slice가 각 reference slice와 맞고 O projection partial도 맞지만 all-reduce 뒤만 다르면 collective ordering 또는 dtype을 본다. 한 rank만 projection부터 다르면 그 rank의 weight shard, quant scale, adapter load, device error를 본다. 모든 rank가 같은 방식으로 다르면 config 또는 공통 implementation 가능성이 높다.

PP send 전과 receive 후 checksum이 다르면 transport buffer, dtype, microbatch identity, stream completion을 본다. 둘은 같고 receiver 첫 norm부터 다르면 stage weight와 norm implementation을 본다. 특정 microbatch에서만 send/receive가 교차하면 pipeline slot reuse와 request incarnation을 본다. shape mismatch가 없어도 identity mismatch는 존재할 수 있다.

MoE layer에서는 pre-router residual, router logits/top-k, dispatch permutation, expert output, inverse permutation 뒤 update를 추가한다. dense layer와 같은 checkpoint만 두면 routing은 다르지만 우연히 aggregate norm이 비슷한 문제를 놓친다. hybrid recurrent layer에서는 state read digest, state write digest와 request slot generation을 추가한다. 공통 schema를 유지하되 layer family별 persistent state를 확장한다.

### 증상에서 첫 probe로 가는 분기

모든 prompt가 첫 token부터 틀리면 input/embedding/final head를 포함한 coarse differential을 한다. 긴 context에서만 틀리면 position, mask, cache length가 개입하는 attention checkpoint를 먼저 촘촘히 둔다. TP=1은 맞고 TP>1만 틀리면 projection shard와 collective 전후를 본다. 특정 adapter에서만 틀리면 adapter 적용 projection을 찾는다. 특정 batch 조합에서만 틀리면 row identity, padding/packing, adapter mixture, MoE permutation을 본다.

성능만 느리고 값은 맞다면 tensor dump를 늘리기 전에 shape와 lifetime을 본다. hidden-state 보존 옵션이 allocator 재사용을 막았는지, PP stage가 wait하는지, TP collective가 update completion을 지연하는지, prefill/decode 구성 때문에 GEMM M축이 달라졌는지 확인한다. correctness probe가 성능을 바꾸므로 두 조사를 같은 run에서 무리하게 해결하지 않는다.

NaN이 나타나면 final logits에서 뒤로 추측하지 말고 최초 non-finite checkpoint를 찾는다. embedding이 finite, layer 11 input이 finite, attention update가 non-finite라면 그 layer의 norm/QKV/score/backend를 확대한다. MLP gate에서 처음 overflow하면 quant scale, activation magnitude, accumulator와 down projection을 본다. residual add 뒤 처음 non-finite라면 두 operand 각각과 cast 순서를 저장한다.

### 검증은 원인을 고친 뒤 같은 경계를 닫는 일이다

수정 후 최종 문자열 한 번이 맞았다고 끝내지 않는다. 최초 divergence checkpoint가 허용 오차 안으로 돌아왔는지, 경쟁 가설을 반증한 fixture에서도 맞는지, prefill과 decode 양쪽에서 맞는지 확인한다. TP/PP, 짧은/긴 context, adapter 유무 중 원인과 직접 관련된 최소 축을 선택해 회귀 행렬을 만든다.

성능 수정이라면 output parity와 함께 목표 metric, shape, workload를 다시 확인한다. in-place fusion으로 memory가 줄었다면 output hidden state 옵션, PP send, graph replay, cancellation에서도 lifetime invariant가 유지되는지 정적·허용된 테스트로 확인한다. 이 작업의 범위에서는 런타임을 실행하지 않으므로 test source와 invariant, 기존 CI가 무엇을 검증하는지 기록하고 실행 결과를 꾸며내지 않는다.

### layer transaction을 실습에 넘기기 전 확인할 질문

독자는 이제 `hidden_states`라는 이름을 보았을 때 그것이 residual인지 normalized branch인지 묻는다. `[T,D]`를 보았을 때 T가 request 수인지 token row 수인지 묻는다. projection output을 보았을 때 rank partial인지 global complete인지 묻는다. tensor가 함수에서 사라질 때 CUDA/communication consumer가 정말 끝났는지 묻는다. prefill과 decode가 같은 수학을 쓰면서 왜 다른 물리 shape와 병목을 갖는지 설명할 수 있어야 한다.

마지막 layer의 residual은 final normalization을 거쳐 vocab projection의 입력이 된다. 이 hidden row는 아직 token도 확률도 아니다. 16장의 LM head가 vocab 차원의 logits를 만들고, 확률은 17·18장의 logits 해석과 sampling 규칙이 필요할 때 계산된다. 다음 장들은 이 장의 전체 지도에서 일부를 확대한다. 11장은 norm과 projection의 수치·병렬 계약, 12장은 QKV head shape, 13장은 causal attention, 14장은 position과 cache, 15장은 MLP·MoE·recurrent state, 16장은 final hidden에서 생성 경계까지를 맡는다.

### 고정 소스 확인 지도

이 장의 링크는 설명을 대신하는 목록이 아니라 의미 checkpoint의 증거다. Transformers Qwen3.5의 [model loop](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1502-L1589), [decoder

layer](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L740-L817), [attention](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L432-L520)을 먼저 연결한다.

Gemma3의 [layer loop](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L600-L665)와 [layer body](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L386-L445)를 대조해 공통 spine과 변형을 나눈다.

serving 경로에서는 vLLM Qwen3의 [layer](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3.py#L173-L263), Qwen2 공통 [model loop](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen2.py#L322-L440), Qwen3.5의 [hybrid model](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5.py#L215-L285)을 본다.

SGLang에서는 Qwen3 [layer](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3.py#L310-L434), Qwen3Next [attention layer](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_next.py#L598-L730), [model loop](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_next.py#L881-L989)을 대응시킨다.

llama.cpp는 [graph builder](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model.cpp#L2461-L2550)에서 architecture graph로 내려간다.

source revision이 바뀌면 line anchor가 이동할 수 있다. 새 버전에서는 class 이름 검색으로 대충 대체하지 말고 commit, file, symbol, semantic checkpoint를 함께 재고정한다. source에 branch가 있다는 사실과 배포에서 그 branch가 실행되었다는 사실도 구별한다. 후자는 config와 runtime evidence가 필요하다. 이 원칙을 지키면 source map은 장식이 아니라 다음 디깅의 출발점이 된다.

## 10.10 한 요청의 layer transaction을 source·수치·운영 판단으로 완성한다

10.1의 transaction 표를 실제 구현에 대입할 때 목표는 tensor를 많이 저장하는 것이 아니다. `X0→N0/A→Xa→Na/M→X1`
각 행에서 관측 한 건이 어느 가설을 지지하고 기각하는지 밝히는 것이다. 실제 서비스에서 임의 hook이나 전체 activation
dump를 켜라는 지시도 아니다. 먼저 고정 source와 config로 producer·consumer를 찾고, 실행 승인이 있는 별도 환경에서는
synthetic fixture와 최소 probe만 사용한다. 이 책의 집필 검증에서는 모델과 CUDA를 실행하지 않는다.

**1단계: transaction의 producer와 consumer를 source에서 찾는다**

고정 revision 하나를 고르고 causal LM wrapper에서 시작한다. `forward`가 logits까지 직접 만드는지, model body의 hidden을 반환한 뒤 별도 `compute_logits`가 소비하는지 표시한다. model body로 내려가 embedding owner와 layer container를 찾는다. layer list가 Python loop인지 helper 함수인지, PP stage의 start/end layer가 어떻게 결정되는지, final norm이 어느 stage에 있는지 적는다.

각 decoder layer에서 다음 여덟 의미점을 symbol과 source span에 연결한다.

```text
layer input residual
attention pre-norm output
QKV 또는 attention call input
attention output projection 뒤 global-complete update
첫 residual add 뒤 checkpoint
MLP/MoE pre-norm output
down projection 또는 expert combine 뒤 global-complete update
둘째 residual add 뒤 layer output
```

표를 채울 때 `hidden_states`처럼 변수 이름만 복사하지 않는다. producer와 consumer를 한 줄씩 쓴다. update가 TP partial일 가능성이 있으면 collective owner를 찾기 전까지 `complete` 칸을 미확정으로 둔다. fused norm이 `(normalized, residual)` tuple을 반환하면 각 원소가 어느 수학값인지 caller와 callee 양쪽을 읽는다. 함수 한쪽만 보고 residual 의미를 추측하지 않는다.

완료 조건은 source link 수가 아니다. embedding에서 layer 0 입력으로 가는 edge, layer i 출력에서 i+1 입력으로 가는 edge, 마지막 layer에서 final norm과 logits owner로 가는 edge가 모두 닫혀야 한다. PP가 켜진 경우 stage send/receive edge도 하나의 forward graph에 들어가야 한다. 구현에 hybrid layer가 있으면 layer type selector와 두 family가 공통 residual contract로 돌아오는 지점을 표시한다.

**2단계: 101-row mixed batch로 shape·owner·bytes 열을 채운다**

다음 workload를 가정한다. request A는 64-token prefill chunk, B는 32-token prefill chunk, C~G 다섯 request는 decode token 하나씩이다. physical query row는 101개다. hidden width는 4096, query head 32, KV head 8, head dimension 128, TP=2다. A와 B는 서로 다른 request이고 decode 다섯 개의 context length는 20, 100, 1000, 4096, 8192다.

먼저 embedding/residual의 logical shape를 `[101,4096]`로 적는다. TP residual이 replicated라고 가정하면 rank마다 같은 logical shape를 가진다. Q는 global `[101,32,128]`, rank local `[101,16,128]`이다. K/V가 head shard라면 global `[101,8,128]`, local `[101,4,128]`이다. 단, 구현이 KV head를 TP rank에 replicate하는 경우 local shape가 달라질 수 있으므로 source를 확인하지 않은 가정에는 `가설` 표시를 붙인다.

QKV projection 뒤 attention이 읽는 과거 KV 양은 query row 101과 같지 않다. A와 B chunk가 읽는 prefix 길이, decode 다섯 context 길이의 합을 별도 열에 둔다. 정확한 read element는 sliding window, prefix cache, page sharing, layer type에 따라 달라지므로 단순 합은 상한 또는 근사로 이름 붙인다. 이 구분 없이 `batch=101`만 dashboard에 올리면 attention latency의 변화가 설명되지 않는다.

MLP intermediate global shape는 gate/up 각각 `[101,14336]`이다. column parallel이면 local feature 폭은 7168일 수 있다. down projection local partial은 `[101,4096]`이고 reduction 뒤 update도 `[101,4096]`이다. 여기서 residual add가 reduction 전인지 후인지 source span을 적는다. 마지막으로 각 tensor의 예상 lifetime을 `produce event`와 `last consumer event`로 써서 동시에 살아 있는 큰 buffer를 찾는다.

정답표는 특정 byte 숫자 하나가 아니다. 독자가 global과 local, logical과 physical, query rows와 context rows, temporary와 persistent state를 분리했는지가 핵심이다. allocator alignment나 fused kernel을 모르는 상태에서 peak memory를 단정했다면 원장을 다시 쓴다. 모르는 항목은 누락하지 말고 unknown과 그것을 결정할 source owner를 기록한다.

**3단계: first divergence 여섯 개를 transaction의 서로 다른 행에 심는다**

실제로 source를 바꾸지 않고 사고 실험으로 진행한다. 첫 번째 결함은 embedding row permutation이다. A position 3과 B position 1이 뒤바뀌었지만 tensor shape와 전체 평균은 같다. embedding checkpoint의 row-keyed digest부터 다르고 이후 모든 layer가 달라진다. 전체 tensor norm만 비교하면 놓칠 수 있다는 결론을 쓴다.

두 번째 결함은 layer 5 attention norm epsilon 차이다. layer 5 input residual까지는 맞고 attention norm부터 작은 차이가 생긴다. attention update와 이후 residual에서 차이가 커진다. 이 경우 cache나 MLP를 먼저 조사하는 것은 잘못이다. epsilon/config, norm implementation, accumulator dtype을 경쟁 가설로 둔다. layer 4와 5 input이 같은 것이 선행 증거다.

세 번째 결함은 TP rank 1의 O projection shard가 잘못 로드된 경우다. rank 0 QKV와 local attention은 맞고 rank 1의 O partial부터 다르다. all-reduce 뒤 모든 rank의 post-attention residual이 같은 방식으로 틀린다. “모든 rank에서 틀렸으니 공통 config 문제”라는 추론이 왜 성급한지 설명한다. collective가 한 rank의 잘못된 partial을 모두에게 전파하기 때문이다.

네 번째 결함은 PP send buffer의 조기 재사용이다. 정상 run 대부분은 맞지만 특정 microbatch overlap에서 stage 0 send 전 checksum과 stage 1 receive 후 checksum이 다르다. shape와 dtype은 맞는다. P2P 자체의 산술보다 buffer lifetime, stream event, microbatch identity를 우선한다. 동기화를 무조건 추가해 증상을 숨기기 전에 ownership invariant를 찾는다.

다섯 번째 결함은 MoE inverse permutation 오류다. router top-k와 expert output의 aggregate norm은 정상인데 두 token row가 바뀐다. layer output 전체 mean도 비슷할 수 있다. dispatch index와 inverse index를 request-position key로 비교한다. dense layer fixture가 통과하고 MoE layer에서만 실패한다는 negative evidence를 보존한다.

여섯 번째 결함은 cache slot incarnation 오류다. prefill 전체 checkpoint는 맞지만 request 취소 후 slot을 재사용한 decode에서 attention update부터 다르다. 새 decode의 layer input residual과 Q projection은 맞고 K/V read state가 다르다. residual bug가 아니라 persistent state ownership 문제다. slot number가 같더라도 generation/incarnation이 달라야 한다는 invariant를 적는다.

각 사고 실험은 다섯 줄로 마친다. 증상, 최초로 볼 coarse checkpoint, 예상 first divergence, 경쟁 가설 두 개, 수정 후 검증이다. 원인 이름만 쓰지 말고 왜 해당 관측이 다른 가설을 약화하는지 한 문장씩 적는다. 이 형식이 증상→관측→분기→원인→검증의 최소 단위다.

**4단계: 저장·계산·cast 차이와 semantic bug를 분리한다**

두 구현의 layer 0 output 최대 절대 오차가 `2e-3`, layer 31에서 `4e-2`라고 하자. 이것만으로 bug 여부를 정할 수 없다. parameter dtype, norm compute dtype, attention backend, quantization, 허용 계약을 기록한다. 같은 argmax와 top-k 순위가 유지되는지, non-finite가 있는지, 특정 row/head에 오차가 집중되는지 본다. layer 깊이에 따른 오차 증가가 매끄러운지 어느 layer에서 계단처럼 뛰는지도 본다.

비교 A는 BF16 eager reference와 BF16 fused backend다. bitwise equality 대신 operation별 tolerance와 최종 기능 계약을 쓴다. 비교 B는 BF16 reference와 INT4 weight-only serving이다. 차이가 예상되므로 quantization 품질 기준과 selected prompt cohort가 필요하다. 비교 C는 같은 artifact와 backend인데 TP=1과 TP=2다. collective reduction order 때문에 미세 차이는 가능하지만 큰 localized divergence는 shard/load/collective 문제를 의심한다.

RMSNorm output을 비교할 때 input magnitude가 매우 작은 row와 큰 row를 분리한다. epsilon 영향은 작은 mean-square에서 상대적으로 크다. projection을 비교할 때 전체 output norm뿐 아니라 shard boundary 열을 본다. attention output은 긴 context와 짧은 context를 나눈다. MLP는 gate saturation이나 quant group boundary에 오차가 몰리는지 본다. 하나의 global tolerance를 모든 checkpoint에 적용하지 않는다.

semantic bug를 반증하는 강한 방법은 동일 implementation에서 dtype만 바꾼 control과 동일 dtype에서 implementation만 바꾼 control을 두는 것이다. 두 축을 동시에 바꾸면 원인을 알 수 없다. 이 작업에서는 실행하지 않지만 실험 행렬을 설계할 때 `artifact`, `dtype`, `backend`, `parallelism`, `input` 중 한 축만 바꾸는 행을 만든다. 결과가 없는 행에는 예상 결론을 쓰지 않는다.

**5단계: 같은 표의 alias·last-consumer 열을 검증한다**

source의 각 residual add 주변에서 input tensor, norm output, branch update, add output이 storage를 공유할 수 있는지 표시한다. API 문서 또는 custom op contract가 없으면 확정하지 않는다. `inplace=True` flag, underscore op, output parameter, fused residual norm 반환 tuple, allocator wrapper를 검색한다. view와 reshape가 copy인지 alias인지 stride 조건까지 확인한다.

다음 consumer 목록을 만든다. attention norm은 QKV projection이 끝나면 더 필요 없는가? cache write가 K/V view를 비동기로 읽는 동안 base fused-QKV buffer가 살아 있는가? O projection partial은 communication stream reduction이 끝나기 전에 재사용되지 않는가? post-attention residual은 MLP norm뿐 아니라 output hook이나 PP send가 참조하는가? layer output은 next layer가 읽는 동시에 hidden-state tuple이 보존하는가?

정적 source에서 stream/event가 보이면 producer stream, consumer stream, dependency event를 연결한다. event가 없다고 즉시 bug라고 단정하지 않는다. framework allocator와 current stream semantics가 의존성을 관리할 수 있다. 반대로 Python reference가 살아 있다고 GPU consumer completion까지 자동 보장된다고 단정하지 않는다. custom allocator와 external communication library 경계를 확인한다.

세 가지 모드의 lifetime을 별도로 그린다. 일반 eager inference, CUDA graph capture/replay, output-hidden-states 또는 debug hook 활성화다. graph는 address를 고정하거나 pool을 분리할 수 있고, debug는 reference lifetime을 늘린다. debug를 켰을 때 OOM이 생겼다면 모델 capacity가 갑자기 줄었다고 표현하지 않고 관측 옵션이 activation retention을 바꿨는지 확인한다.

검토 결과는 위험도와 검증 방법을 붙인다. 예를 들어 “PP send buffer가 send completion 전에 next microbatch output으로 재사용될 가능성—source event dependency 확인, synthetic overlap test 필요”처럼 쓴다. “메모리 문제 있음”은 소유권도 검증도 없는 메모다.

**6단계: 옵션 하나를 transaction state 변화까지 추적한다**

`tensor_parallel_size=2`를 예로 든다. parser가 값을 읽는 데서 멈추지 않는다. model loader가 QKV/O/gate/up/down weight를 어떤 shard로 나누는지, attention head와 KV head divisibility가 무엇인지, decoder layer가 어떤 parallel linear class를 구성하는지, collective group이 어디서 만들어지는지, residual이 replicated인지 sequence-sharded인지 추적한다. 최종 effect는 “GPU 두 개 사용”이 아니라 local tensor shape, weight ownership, collective 위치, latency/memory/correctness invariant 변화다.

`pipeline_parallel_size=2`는 layer partition, embedding/final head owner, intermediate tensor transport, stage schedule을 바꾼다. layer 수가 균등하게 나뉘지 않거나 architecture가 특정 module을 한 stage에 강제할 수 있다. 옵션을 바꿨는데 실제 model이 PP를 지원하지 않아 reject 또는 fallback한다면 requested state와 effective state가 다르다. 로그나 constructed module tree로 effective partition을 검증할 증거를 설계한다.

`output_hidden_states=True`는 모델 수학의 선택처럼 보이지만 관측 및 lifetime state를 바꾼다. layer 전 checkpoint reference가 반환 객체까지 살아 있고 peak activation memory와 compile/graph compatibility가 달라질 수 있다. 결과 tensor가 생겼다는 것만 확인하지 말고 allocator peak와 selected execution path를 함께 본다. production에서 이 옵션을 무제한 켜지 않는 이유를 설명한다.

`torch_dtype` 또는 서버의 dtype 옵션은 weight loading, activation storage, kernel eligibility, accumulation, output cast에 영향을 줄 수 있다. 옵션 문자열을 적는 대신 constructed parameter dtype, residual checkpoint dtype, norm compute path, selected backend를 검증한다. `auto`는 artifact metadata와 hardware 지원에 따라 effective 값이 달라질 수 있으므로 특히 결과 상태를 기록한다.

각 옵션 기록은 다음 사슬을 완성해야 한다. `사용자 field → validation/branch → constructed module 또는 runtime state → tensor shape/dtype/owner → backend/collective eligibility → 관측 효과 → 반증 probe`. 사슬 중간이 비면 “이 값을 올리면 빨라진다” 같은 레시피를 쓰지 않는다. workload와 병목이 없으면 효과 방향도 보편적이지 않다.

**7단계: prefill의 `X1`에서 첫 decode의 `X0`까지 잇는다**

prompt token이 `[t0,t1,t2,t3]`이고 prefill이 네 row를 한 번에 처리했다고 하자. sampling에 쓰는 hidden은 보통 마지막 유효 position t3의 final hidden이다. LM head와 sampler가 다음 token t4를 선택하면 다음 decode step의 input ID는 t4다. embedding lookup이 t4 row를 만들고 position은 다음 logical 위치를 가리킨다. cache는 t0~t3의 layer별 K/V를 이미 보유해야 한다.

같은 transaction의 handoff 행에는 `prefill.final_hidden(request,pos=3)`, `prefill.logits`, `selected_token=t4`,
`decode.embedding(t4,pos=4)`, `cache.logical_length_before_decode=4`를 둔다. 구현에 따라 cache write와 length commit
시점이 다르고 speculative decoding은 draft token과 accepted token을 구분해야 한다. 여기서는 비추론 기본 경로를 먼저 고정한다.

첫 decode 결과가 reference와 다를 때 prefill final hidden부터 확인한다. prefill이 맞고 selected token도 같은데 decode embedding부터 다르면 token handoff, embedding artifact, adapter를 본다. embedding은 같고 attention update부터 다르면 position/cache metadata를 본다. 모든 layer hidden은 같은데 logits가 다르면 final norm/LM head/tied weight/TP gather를 16장과 17장의 좌표로 넘긴다.

padding이 있는 dense reference에서는 마지막 array index가 마지막 유효 token이 아닐 수 있다. attention mask나 sequence length로 t3 row를 선택해야 한다. packed engine에서는 request별 last row mapping이 필요하다. 잘못된 row에서 logits를 뽑으면 layer 전체가 정확해도 생성이 틀린다. 이 오류를 model layer correctness 문제와 분리한다.

### 8단계: first divergence를 재현 가능한 사건 문장으로 쓴다

나쁜 기록은 “vLLM에서 Qwen이 가끔 틀림”이다. 좋은 기록은 관측 범위를 좁힌다. 예를 들어 “고정 artifact와 token IDs, BF16, TP=2, PP=1에서 길이 4096 이상 decode 첫 step의 layer 18 attention update가 TP=1 reference와 허용 오차를 넘는다. layer 18 input residual과 norm/Q projection은 일치하고, rank 1의 cache-read 뒤 attention local output부터 갈린다. 짧은 context와 eager backend에서는 재현되지 않는다”라고 쓴다.

이 기록은 증상, fixture, first divergence, negative evidence를 포함한다. 아직 root cause를 모른다면 모른다고 쓴다. 경쟁 가설은 cache page mapping, backend long-context specialization, rank 1 KV shard다. 각 가설에 반증 관측을 붙인다. page mapping이 같고 eager에서도 재현되면 첫 가설은 약해진다. TP=1에서도 같은 backend에서 재현되면 shard 가설은 약해진다.

source evidence에는 commit, file, symbol, relevant span과 그 span이 증명하는 명제를 적는다. source가 long-context branch를 가진다는 사실은 실제 run이 그 branch를 탔다는 증거가 아니다. effective backend log 또는 trace가 필요하다. 반대로 runtime metric만으로 어느 source branch가 원인이라고 단정하지 않는다. 5장의 증거 등급을 그대로 적용한다.

수정 검증 항목은 원래 fixture, 경계값 바로 아래/위, 경쟁 가설 control, output parity, latency/memory regression이다. 긴 context fix가 짧은 context를 깨지 않았는지, TP=2 fix가 TP=1 path에 불필요한 sync를 추가하지 않았는지 본다. 문제가 사라졌다는 한 문장보다 first divergence가 닫혔다는 증거가 강하다.

### 9단계: transaction 표가 설명을 대신하지 않았는지 다시 읽는다

이 장을 읽고 “residual stream은 정보를 계속 더하는 고속도로다”만 기억했다면 충분하지 않다. 고속도로 비유는 주 경로가 유지된다는 직관에는 유용하지만, dtype cast, rank partial, buffer alias, cache라는 시간축 state를 설명하지 못한다. 비유 뒤에 정확한 tensor 식과 ownership ledger가 따라왔는지 확인한다.

또한 모든 모델이 정확히 `X+Attention(Norm(X))`, `X+MLP(Norm(X))` 두 줄이라고 일반화하지 않는다. sandwich norm, parallel residual, learned scaling, hybrid recurrent layer, multimodal splice, MoE가 순서와 state를 바꿀 수 있다. 이 장의 식은 흔한 pre-norm decoder의 기준축이다. 실제 architecture는 config와 source에서 변형을 확인한다.

본문에 소스 좌표가 많다고 친절한 설명이 되는 것도 아니다. 각 링크 앞에는 독자가 무엇을 확인해야 하는지 있어야 하고, 링크를 열지 않아도 인과 흐름은 이해할 수 있어야 한다. 반대로 source 없는 구체 구현 주장은 피한다. “대개”, “항상”, “자동” 같은 단어가 source의 조건문과 모순하지 않는지 검토한다.

마지막으로 각 절이 같은 내용을 다른 목록으로 반복하지 않았는지 본다. shape 표는 shape의 의미를 설명한 뒤에만 유용하다. 체크리스트는 원인을 생각하는 법을 배운 뒤에만 유용하다. source note는 본문 논증을 떠받칠 때만 유용하다. 독자가 다음 장으로 갈 때는 residual 전체 지도를 머리에 둔 채 norm, QKV, attention, cache, feed-forward 중 원하는 확대 경로를 선택할 수 있어야 한다.

### 10단계: transaction의 첫 빈칸을 다음 owner의 행동으로 번역한다

마지막 과제는 layer checkpoint를 수집하는 데서 끝나지 않고 어떤 팀이 다음 행동을 소유하는지 결정하는 일이다. 상황은 다음과 같다. 배포 직후 TTFT p99가 늘었지만 ITL 중앙값은 거의 같다. 짧은 prompt는 정상이고 8K token 이상에서만 느리다. 출력 parity는 유지된다. GPU utilization 평균은 높아졌지만 request goodput은 낮아졌다. 이 정보만으로 attention kernel을 원인으로 확정하지 않는다.

먼저 queue와 model 실행 시간을 나눈다. queue가 늘었다면 scheduler/admission owner로 간다. model prefill 시간이 늘었다면 같은 prompt token 수와 chunk 구성을 확인한다. 배포 전후 scheduler가 만드는 layer input T가 달라졌다면 model code가 같아도 GEMM과 attention shape가 다르다. chunk 수가 늘어 같은 prompt가 layer loop를 더 자주 통과했다면 launch와 cache read가 늘 수 있다. 이때 checkpoint의 목적은 값 비교가 아니라 step별 shape와 phase identity를 복원하는 것이다.

shape가 같고 attention 구간만 늘었다면 backend selection, context-length specialization, workspace와 cache layout을 본다. QKV와 MLP 시간이 함께 늘었다면 dtype 또는 graph path, clock/thermal, batch composition을 경쟁 가설에 둔다. PP stage wait가 늘었다면 stage imbalance나 P2P를 본다. TP collective만 늘었다면 rank별 token shape, interconnect, collective algorithm을 분리한다. Layer transaction은 원인 그 자체가 아니라 metric을 source owner로 보내는 주소 체계다.

두 번째 상황은 특정 adapter를 섞은 batch에서만 일부 출력이 틀리는 경우다. 단일 adapter fixture는 모두 통과한다. 이때 전체 layer output을 평균 비교하기보다 row별 adapter identity를 붙인다. embedding과 attention norm은 같고 Q projection부터 특정 row만 달라지면 adapter dispatch 또는 projection 적용 범위를 본다. QKV는 맞고 MLP gate부터 갈리면 adapter target module과 packed row mapping을 본다. 서로 다른 adapter row가 바뀌어도 tensor shape는 정상일 수 있다.

세 번째 상황은 취소가 많은 부하 뒤 메모리는 회복되지만 새 요청 첫 decode가 간헐적으로 틀리는 경우다. allocator byte가 회복되었다는 사실은 state가 논리적으로 초기화되었다는 증거가 아니다. 새 요청의 embedding과 first-layer Q가 reference와 맞고 attention update부터 틀리면 KV slot incarnation을 본다. hybrid layer라면 recurrent state slot도 따로 본다. 모든 persistent state가 정상이고 PP receive부터 다르면 activation buffer incarnation을 본다.

각 상황에서 조치는 네 칸으로 쓴다. `지금 아는 것`, `아직 모르는 것`, `다음 최소 관측`, `관측 결과별 owner`다. 예를 들어 지금 아는 것은 긴 prompt prefill에서만 latency가 증가했다는 사실이다. 아직 모르는 것은 scheduler shape 변화와 backend 변화 중 어느 것인지다. 다음 관측은 step별 prefill rows, context lengths, selected backend다. shape 변화면 scheduler, backend 변화면 selector/kernel owner로 간다. 이렇게 쓰면 회의에서 목소리가 큰 가설이 아니라 증거가 조사 순서를 결정한다.

관측의 비용도 판단한다. 전체 activation dump는 first divergence에는 강하지만 latency와 memory를 크게 교란하고 사용자 입력을 노출할 수 있다. shape와 duration은 저비용이지만 correctness divergence를 직접 증명하지 못한다. bounded synthetic canary의 row digest는 중간 비용으로 의미 checkpoint를 비교할 수 있다. 따라서 항상 가장 자세한 probe를 켜는 것이 아니라 질문을 가르는 최소 probe를 선택한다.

운영 판단의 종료 조건은 “원인을 찾은 것 같다”가 아니다. 선택한 owner의 변경이 first divergence 또는 latency 구간을 닫았고, 경쟁 가설 control에서는 예상대로 변화가 없으며, 관련 없는 request cohort의 goodput과 correctness가 악화되지 않아야 한다. rollback이 더 안전한 조건도 미리 쓴다. non-finite, cross-request state contamination, stage identity mismatch는 성능 저하보다 즉시 rollback 문턱을 낮게 둔다.

이 과제를 마치면 residual stream은 단순한 신경망 그림이 아니다. API와 scheduler가 만든 token row를 model layer의 shape로 번역하고, parallel rank와 cache state를 지나, 관측 metric과 source owner를 잇는 공통 좌표가 된다. 이 좌표가 있어야 “왜 느린가”, “어디서 값이 갈렸는가”, “어떤 옵션이 실제로 무엇을 바꿨는가”를 같은 언어로 논의할 수 있다.

아래 열을 빈칸 없이 채우면 이 장의 작업을 마칠 수 있다.

| 경계 | 반드시 아는 값 | 모르면 찾아갈 owner |
|---|---|---|
| input→embedding | token row, position, request mapping | tokenizer/processor와 embedding module |
| embedding→layer 0 | shape, dtype, device, PP owner | model forward와 stage partition |
| pre-norm→attention | residual source, norm compute dtype | decoder layer와 norm module |
| QKV→attention | head shape, position, cache slot | attention과 cache metadata |
| O projection→add | TP partial/complete, collective | parallel linear과 communication group |
| add→MLP/MoE | alias, lifetime, normalized input | fused norm/residual contract |
| MLP/MoE→add | shard/dispatch/inverse mapping | down projection 또는 expert combine |
| layer i→i+1 | global layer ID, row identity | model loop 또는 PP transport |
| last layer→final norm | final stage owner, hidden contract | model body |
| final hidden→logits | vocab shard/gather, selected rows | LM head와 logits processor |

이 표는 정답을 대신하지 않는다. 실제 source에서 각 owner를 찾아야 한다. 그러나 빈칸이 어디인지 보여 주므로 조사 범위를 정직하게 만든다. shape를 모르면 kernel 성능을 단정하지 않고, owner를 모르면 lifetime을 단정하지 않으며, effective branch를 모르면 옵션 효과를 단정하지 않는다. 그 태도가 한 forward를 읽는 기술의 핵심이다.

독자 경로도 여기서 갈라진다. 수치 오차와 fused residual이 궁금하면 11장으로, head shard와 QKV layout이 궁금하면 12장으로 간다. 긴 context에서만 문제가 생기면 13장의 mask·attention 계산과 14장의 position·cache 원장을 차례로 본다. dense layer는 맞고 expert 또는 recurrent layer에서 갈리면 15장으로 간다. 모든 hidden checkpoint는 맞는데 token만 다르면 16장의 LM head, 17장의 logits·확률, 18장의 sampling·commit 경계를 차례로 본다. 어느 길을 택하든 이 장의 request-position-layer key와 partial-versus-complete 표기를 유지해야 다시 전체 forward로 돌아올 수 있다.

그리고 조사 결과를 되돌아와 같은 transaction에 기록한다. 그래야 국소 최적화의 효과와 부작용을 요청 전체의 인과 사슬에서 다시 검토할 수 있다.
동일 fixture와 ownership 좌표를 유지해야 수정 뒤 두 residual update도 정확히 재검증할 수 있다.

## 10.11 Llama block hook의 좌표와 activation lifetime을 사건으로 닫는다

`LlamaDecoderLayer`에 hook 하나를 붙여 “layer output”을 얻었다는 설명은 충분하지 않다. block 안에는 최소한 입력 residual, attention pre-norm, attention output projection, 첫 residual add, MLP pre-norm, down projection, 둘째 residual add가 있다. 같은 `[T,D]` shape가 여러 번 등장하므로 shape와 Python 변수 이름만으로 어느 값을 관찰했는지 알 수 없다. hook coordinate는 producer symbol, 연산 직전/직후, residual add 이전/이후, TP collective 이전/이후, request-position row mapping을 함께 가져야 한다.

### 10.11.1 reference Llama block의 의미 좌표를 먼저 고정한다

흔한 pre-norm Llama block을 다음 식으로 적자.

```text
x0 = layer input residual
n1 = RMSNorm(x0)
a  = O(Attention(QKV(n1)))
x1 = x0 + a
n2 = RMSNorm(x1)
m  = Down(SiLU(Gate(n2)) * Up(n2))
x2 = x1 + m
```

모듈 전체의 forward hook output은 보통 `x2`다. attention submodule hook은 구현에 따라 O projection 전 local head tensor가 아니라 projection 뒤 `a`를 돌려줄 수 있다. norm hook은 `n1`이나 `n2`를 주지만 fused residual-norm 구현에서는 반환 tuple 중 하나가 normalized branch이고 다른 하나가 residual carrier일 수 있다. 그러므로 hook 이름을 의미 좌표로 착각하지 않는다.

각 좌표에 key를 붙인다. `(request_id, logical_token_position, global_layer_id, checkpoint_kind, generation)`이다. packed batch의 physical row는 실행마다 바뀔 수 있으므로 key의 중심이 아니다. PP stage-local layer 0도 global layer 0과 다를 수 있으므로 global과 local layer ID를 둘 다 저장한다. prefill chunk와 decode step이 같은 logical position을 처리하지 않도록 phase와 step generation도 둔다.

Transformers의 고정 Llama decoder source에서는 block forward의 norm, self-attention, residual add, MLP, 둘째 add 순서를 읽는다. model loop에서는 어느 tensor가 다음 layer로 넘어가고 hidden-state collection이 어느 시점에 참조를 보존하는지 확인한다. vLLM과 SGLang의 Llama 계열 block에서는 fused residual norm과 TP linear가 같은 의미 checkpoint를 다른 tuple/layout으로 표현할 수 있다. llama.cpp graph에서는 Python hook이 없으므로 graph node와 callback이 같은 역할을 한다. 네 구현을 함수 이름이 아니라 위 일곱 좌표로 맞춘다.

### 10.11.2 숫자 여덟 개로 residual alias를 손으로 검산한다

T=2, D=4인 작은 tensor를 둔다.

```text
x0 = [[1, 2, 3, 4],
      [5, 6, 7, 8]]
a  = [[10, 20, 30, 40],
      [50, 60, 70, 80]]
x1 = [[11, 22, 33, 44],
      [55, 66, 77, 88]]
```

attention 이전 hook이 `x0`의 view를 보존했다고 하자. 구현이 `x0.add_(a)`로 in-place update하면 hook이 저장한 참조를 나중에 읽을 때 값은 `x1`이다. hook 실행 순간에 출력한 digest와 요청 종료 뒤 저장 객체의 digest가 달라질 수 있다. Python 변수 `residual = hidden_states`는 copy가 아니라 동일 storage에 대한 두 reference일 수 있다. 재할당 `hidden_states = residual + a`는 새 storage를 만들 가능성이 높지만 allocator, compiler, functionalization이 실제 buffer를 재사용할 수 있으므로 source syntax만으로 storage identity를 확정하지 않는다.

관측 artifact에는 shape/dtype 외에 storage identity를 설명할 수 있는 값이 필요하다. 허용된 test 환경이라면 data pointer, storage offset, stride, version counter, copy 시점, producer/consumer event를 기록한다. 이 책의 정적 검토에서는 custom op contract, schema alias annotation, underscore/in-place op, output parameter, compiler pass와 test assertion을 읽어 alias 가능성을 표시한다. “hook list에 tensor를 append했으니 보존됐다”는 주장을 하지 않는다.

작은 fixture에서 snapshot은 `clone`한 `x0_snapshot`, live reference는 `x0_ref`로 구분한다. residual add 전후 두 digest를 기대한다. `x0_snapshot`은 `[1..8]`을 유지하고 `x0_ref`가 `[11..88]`로 변했다면 수학 오류가 아니라 관측 alias다. 둘 다 예상치 않게 변했다면 snapshot 시점이나 비동기 copy completion을 본다. `x1` 자체가 reference와 다르면 실제 compute 또는 wrong row mapping을 본다.

### 10.11.3 hook 위치가 collective 경계를 가로지르는지 확인한다

TP row-parallel O projection을 생각하자. rank r은 local attention heads로 partial update `a_r[T,D]`를 만든다. global update는 `a = Σ_r a_r`다. residual add가 collective 뒤에 있다면 `x1=x0+a`는 모든 rank에서 complete다. hook이 linear 내부에서 reduction 전에 실행되면 rank-local partial을 보며, block output hook은 complete residual을 볼 수 있다.

TP=2, D=4에서 `a0=[[1,0,1,0]]`, `a1=[[0,2,0,2]]`, `x0=[[10,10,10,10]]`이면 global `x1=[[11,12,11,12]]`다. rank 0 partial hook만 보고 reference global `a=[[1,2,1,2]]`와 비교하면 두 열이 틀렸다고 오판한다. artifact에 `parallel_state=local_partial|global_complete`, rank, group generation, collective kind를 넣어야 한다.

reduce-scatter나 sequence parallel을 쓰면 complete tensor가 모든 rank에 복제되지 않을 수 있다. 각 rank가 token row 일부만 소유하고 feature는 complete할 수 있다. 이 경우 rank별 digest를 단순 equality로 비교하지 않는다. logical row ownership에 따라 canonical order로 gather한 reference와 비교한다. source에서 collective 함수 이름만 보는 것이 아니라 output layout과 next consumer가 기대하는 layout을 읽는다.

PP에서는 block output `x2`가 send buffer가 된다. stage 0의 last local layer hook과 stage 1 receive hook은 같은 logical activation을 다른 storage와 stream에서 본다. dtype cast, serialization/padding, microbatch ordering이 끼어들 수 있다. checkpoint key에 stage, microbatch, send generation을 넣고 send-before와 receive-after를 비교한다. stage-local hook 번호만 쓰면 microbatch가 섞였을 때 잘못된 tensor를 정상으로 매칭한다.

### 10.11.4 activation lifetime을 produce에서 last consumer까지 그린다

activation이 “함수 안에서 생성돼 함수 끝에서 사라진다”는 설명은 GPU 비동기 실행에서 틀릴 수 있다. Python call이 반환돼도 CUDA kernel이나 collective가 buffer를 읽고 있을 수 있다. 반대로 Python reference가 남아 있어도 allocator 관점에서는 다른 view와 storage를 공유해 값이 mutate될 수 있다. lifetime은 producer enqueue, consumer enqueue, completion dependency, allocator reuse 가능 시점으로 적는다.

`n1`은 QKV projection이 입력을 완전히 소비할 때까지 살아야 한다. packed QKV output은 Q/K/V view와 cache write, attention kernel이 마지막으로 읽을 때까지 살아야 한다. attention output partial은 collective completion까지, global `a`는 residual add까지 필요하다. `x1`은 MLP norm뿐 아니라 debug hidden state, PP send, cancellation cleanup이 참조할 수 있다. `n2`는 gate/up projection completion까지, gated intermediate는 down projection completion까지 필요하다.

두 stream 예를 보자. compute stream C가 O projection partial P를 만들고 communication stream M이 all-reduce(P)를 시작한다. allocator가 C의 다음 op에서 P storage를 재사용하면 M이 아직 읽는 중일 수 있다. 올바른 contract는 event 또는 allocator의 stream-recording으로 reuse를 M completion 뒤로 미룬다. `torch.cuda.synchronize()`를 전역으로 넣으면 증상은 사라질 수 있지만 overlap과 latency를 파괴하고 missing dependency의 정확한 owner를 숨긴다.

정적 source walk에서는 stream record, event wait, async collective handle, custom allocator release, graph capture pool을 찾는다. 보이지 않는다고 즉시 race로 단정하지 않는다. framework가 내부적으로 lifetime을 관리할 수 있다. 대신 “P의 last consumer completion을 allocator가 어떻게 아는가”라는 질문과 근거 symbol을 남긴다.

### 10.11.5 output_hidden_states와 debug hook은 모델의 메모리 상태를 바꾼다

L layers, T token rows, D hidden, BF16 2 bytes일 때 layer output 하나는 `T×D×2` byte다. T=4096, D=4096이면 32MiB다. 32개 layer output을 모두 보존하면 tensor payload만 약 1GiB다. allocator alignment, tuple/reference overhead, norm/branch snapshot, device-to-host copy staging은 별도다. “관측만 켰다”가 peak memory와 graph eligibility를 바꿀 수 있다.

hook이 `clone()`을 device에 남기면 수학값은 안정적으로 보존하지만 메모리가 늘어난다. CPU로 즉시 copy하면 device lifetime은 줄일 수 있어도 D2H bandwidth와 synchronization이 execution을 교란한다. 비동기 copy를 쓰면 source buffer가 copy completion까지 살아야 한다. 표본 row, layer 이분 탐색, safe synthetic fixture로 probe를 줄이는 이유다.

CUDA graph는 주소와 실행 shape를 고정한다. 임의 hook, 동적 clone, Python callback은 graph capture를 깨뜨리거나 eager fallback을 유발할 수 있다. 디버그 모드에서 오류가 사라지면 hook이 값을 고쳤다는 뜻이 아니라 effective backend와 scheduling이 달라졌을 가능성이 있다. trace에는 requested hook뿐 아니라 selected graph/eager path를 기록한다.

### 10.11.6 in-place wrong-answer incident를 경쟁 가설로 분해한다

사건 R10은 debug checkpoint를 켰을 때 layer 14 input이 reference와 달랐고, hook을 끄면 최종 output parity가 맞았다. 첫 가설은 실제 layer 13 residual add가 틀렸다는 것이다. 두 번째는 hook이 live alias를 늦게 읽었다는 것이다. 세 번째는 hook 때문에 graph에서 eager로 fallback해 execution path가 바뀌었다는 것이다. 네 번째는 device-to-host copy synchronization이 원래 lifetime race를 가렸다는 것이다.

첫 probe는 hook callback 순간의 bounded row snapshot과 종료 뒤 저장 object를 분리한다. callback snapshot은 reference와 같지만 종료 뒤 object가 바뀌면 alias 가설을 지지한다. 둘 다 다르고 graph/eager selected path도 다르면 동일 path의 safe checkpoint 방식이 필요하다. hook을 켜야만 정상이라면 missing dependency race 가능성이 커지지만, 입력 cohort와 scheduling도 같다는 증거가 있어야 한다.

source에서는 layer forward의 residual assignment, fused norm 반환 contract, in-place add/custom op, hidden-state collection 위치를 읽는다. vLLM/SGLang path에서는 custom RMSNorm이 residual tuple을 어떻게 반환하고 caller가 어느 변수를 다음 block으로 넘기는지 본다. llama.cpp에서는 graph tensor가 view인지 새 node인지, in-place flag와 backend buffer reuse가 어떻게 표현되는지 확인한다.

수정은 hook에서 무조건 모든 tensor를 clone하는 것으로 끝내지 않는다. 관측 목적별 checkpoint contract를 만든다. correctness canary는 선택한 row/layer만 stable snapshot한다. fleet metric은 shape, dtype, norm, non-finite, generation 같은 bounded summary를 producer 시점에 계산한다. full activation은 승인된 격리 환경에서만 수집한다. 각 probe의 alias 안정성, graph 영향, 메모리 상한을 문서화한다.

### 10.11.7 Llama source를 caller와 consumer까지 고정하는 법

Transformers에서는 `LlamaModel.forward`의 layer loop에서 decoder layer 호출 전 hidden과 반환 후 hidden을 찾는다. `LlamaDecoderLayer.forward`에서 input residual 보존, input layernorm, self-attention, residual add, post-attention layernorm, MLP, second add를 연결한다. hidden-state tuple이 add 전인지 후인지 model loop의 append 위치로 확인한다. hook framework의 module input/output semantics도 별도 공식 contract에서 확인해야 한다.

vLLM과 SGLang에서는 model-specific Llama layer와 공통 RMSNorm/parallel linear implementation을 함께 읽는다. fused add-norm이 `(normalized, residual)`을 반환한다면 tuple order를 callee return과 caller unpack 양쪽에서 확인한다. O/down projection이 reduce_results를 언제 수행하고 residual add가 어느 쪽에 있는지 본다. PP intermediate tensor container가 hidden과 residual을 따로 운반하는지도 확인한다.

llama.cpp에서는 architecture graph builder가 attention norm node, attention branch, add node, FFN norm/branch, final add node를 만드는 순서를 찾는다. graph callback 또는 tensor name이 의미 checkpoint를 식별하는지, backend planner가 in-place view를 허용하는지 구분한다. graph 정의는 실행 backend의 allocation/lifetime을 전부 증명하지 않으므로 CUDA backend와 allocator source는 후속 owner로 남긴다.

각 pinned link 옆에는 명제를 한 줄로 쓴다. “이 span은 둘째 residual add 뒤 값이 next layer로 전달됨을 증명한다.” “이 span은 fused norm tuple의 첫 값이 normalized branch임을 증명한다.” “이 span은 row-parallel output reduction 여부를 config flag로 결정함을 증명한다.” source가 증명하지 않는 runtime branch, latency, race 발생 여부는 추론으로 표시한다.

### 10.11.8 승인 matrix와 종료 terminal

regression matrix는 architecture(reference Llama/fused serving), phase(prefill/decode), parallelism(TP1/TP2, PP1/PP2), execution(eager/graph), observation(no hook/summary/stable snapshot), output-hidden-state off/on을 축으로 삼는다. 모든 조합을 폭발적으로 실행하라는 뜻은 아니다. incident 가설을 가르는 pairwise cell과 boundary cell을 선택한다.

각 cell은 layer input, post-attention residual, layer output의 request-position keyed digest, alias/storage state, local/complete 표기, selected path, peak retained bytes를 판정한다. lifetime 사건이면 producer/consumer event generation도 넣는다. final text는 보조 terminal이다. first divergence가 닫히지 않았는데 우연히 같은 token이 나온 것을 성공으로 보지 않는다.

correctness terminal은 동일 artifact와 logical input에서 같은 의미 checkpoint가 허용 오차 안에 있고 row identity가 보존되는 것이다. ownership terminal은 hook artifact가 local/complete, pre/post add, snapshot/live reference를 명시하는 것이다. lifetime terminal은 모든 비동기 consumer completion 전 storage가 재사용되지 않는 것이다. observability terminal은 probe가 유발하는 graph fallback과 peak memory가 문서화되고 bounded한 것이다.

사건 종료 문장은 구체적으로 쓴다. “Layer 14 input divergence는 compute 오류가 아니라 post-hook list가 in-place residual storage를 참조해 종료 시 layer 14 이후 값으로 변한 관측 alias였다. producer 시점 stable row snapshot으로 바꾸고 eager/graph selected path를 보존했으며, TP local partial과 post-reduce global checkpoint를 분리했다. 8K prefill과 first decode, output-hidden-states on/off에서 request-position digest가 reference와 일치하고 retained memory 상한을 통과했다.”

이 절을 통과하면 독자는 hook을 어디에 붙였는지가 아니라 무엇을 관찰했는지 설명할 수 있다. residual 변수 이름, module 경계와 Python reference를 수학적 checkpoint로 오인하지 않는다. 다음 11장에서는 여기서 고정한 pre-norm, projection input/output, fused residual tuple을 RMSNorm reduction과 packed projection stride의 수치 계약으로 더 깊게 확대한다.

## 10.12 최소 probe로 first divergence를 찾는 운영 실습

서비스 장애에서 모든 layer activation을 한 번에 저장하는 것은 거의 항상 과하다. 먼저 final hidden 또는 first logits가 기준과 다른지 확인하고, layer checkpoint를 네 구간으로 나눈다. 32-layer 모델이라면 layer 0 input, 8 output, 16 output, 24 output, final norm을 비교한다. 첫 차이가 8과 16 사이면 그 구간만 이분 탐색한다. 각 probe는 같은 request-position key와 execution generation을 사용해야 한다.

### 10.12.1 coarse checkpoint도 의미 경계에 둔다

layer 8 “output”은 block의 둘째 residual add 뒤여야 한다. 한 구현에서 layer module output, 다른 구현에서 다음 layer pre-norm output을 비교하면 norm 차이가 섞인다. coarse probe라 해도 checkpoint kind를 맞춘다. PP 경계에서는 send 전과 receive 후를 별 checkpoint로 두어 compute와 transport를 가른다.

첫 비교의 수치는 단일 global norm만 쓰지 않는다. request-position별 non-finite count, bounded row digest, L2/max absolute difference, top differing dimensions를 둔다. 두 token row가 뒤바뀌면 global mean과 norm이 같을 수 있다. canonical row key로 정렬하고 shape/stride/dtype을 함께 판정한다.

허용 오차는 checkpoint와 dtype에 따라 다르다. 같은 BF16 artifact의 eager/fused 비교와 INT4/BF16 비교를 같은 tolerance로 처리하지 않는다. reference가 무엇인지, cast 시점과 accumulation dtype, expected numerical envelope를 먼저 적는다. non-finite, row permutation, generation mismatch는 tolerance로 허용하지 않는 구조 오류다.

### 10.12.2 세 종류의 alias를 구분한다

첫째는 동일 storage의 직접 alias다. `residual`과 `hidden_states`가 같은 base와 offset을 가진다. 둘째는 view alias다. reshape/slice가 다른 shape와 stride로 같은 storage를 본다. 셋째는 allocator reuse다. lifetime이 끝난 storage 주소를 다음 tensor가 재사용해 pointer는 같지만 동시에 살아 있는 alias는 아니다. pointer equality 하나로 세 경우를 합치면 안 된다.

직접 alias는 in-place write가 양쪽 이름에 즉시 보인다. view alias는 특정 row/dimension만 겹칠 수 있다. allocator reuse는 producer 시점 snapshot과 consumer completion 사이가 안전하면 정상 최적화다. 사건 ledger에는 base storage generation과 live interval을 붙인다. 같은 address라도 generation이 다르면 다른 allocation incarnation이다.

compiler functionalization은 source의 in-place op를 out-of-place로 바꾸거나 반대로 buffer reuse를 적용할 수 있다. eager source syntax와 compiled effective graph를 구별한다. source는 가능한 semantics를, graph/selected backend evidence는 실제 lane을 설명한다. 정적 검토만으로 effective reuse를 확정할 수 없다면 그 한계를 명시한다.

### 10.12.3 transaction의 lifetime·bytes 열로 probe 비용을 계산한다

T=8192, D=8192, BF16이면 `[T,D]` 하나가 `8192×8192×2 = 134,217,728 byte`, 즉 128MiB다. 40개 layer output을 모두 clone하면 payload만 5GiB다. attention/MLP 중간과 allocator overhead는 빠져 있다. debug OOM이 model weight OOM이 아니라 retained activation 5GiB 때문일 수 있다.

token row 4개만 snapshot하면 `4×8192×2 = 65,536 byte`, 64KiB다. layer 이분 탐색에서 5개 checkpoint를 잡아도 320KiB 수준이다. row selection은 첫/마지막 유효 token, failing request의 경계 position, passing neighbor를 포함한다. 평균적인 row만 뽑아 batch reorder나 padding 경계 오류를 놓치지 않는다.

summary probe는 각 row의 norm과 digest만 계산할 수 있지만 digest compute도 GPU work와 read bandwidth를 쓴다. every layer/every request가 아니라 synthetic canary와 sampled incident에 제한한다. D2H copy를 할 경우 copy stream과 completion event가 원 execution lifetime에 어떤 dependency를 추가하는지 기록한다.

### 10.12.4 cancellation과 buffer incarnation 사건

요청 A의 prefill이 layer 20까지 진행된 뒤 취소됐다고 하자. activation pool slot 7이 반환되고 요청 B가 같은 slot을 쓴다. P2P send나 async reduction이 A의 buffer를 아직 읽고 있다면 B의 first layer output이 A consumer와 충돌할 수 있다. 또는 cleanup이 slot metadata만 초기화하고 recurrent/KV state generation을 남길 수 있다.

재현 축은 cancellation point, same-bucket reuse, overlap on/off, PP/TP 여부다. A와 B의 request-position-layer-generation key를 이벤트에 붙인다. allocator byte가 정상으로 돌아왔다는 사실은 A consumer가 끝났거나 slot incarnation이 바뀌었다는 증거가 아니다. completion handle과 free/reuse generation을 확인한다.

완화로 cancellation 후 전역 synchronize를 넣으면 race를 숨길 수 있지만 tail latency와 goodput을 크게 해친다. 더 좁은 수정은 A의 last consumer event를 slot reuse dependency로 연결하거나 cancel cleanup이 async handles를 drain하도록 하는 것이다. 어떤 owner가 completion을 소유하는지는 runtime/communication/allocator 경계를 source로 걷는다.

### 10.12.5 fused residual norm tuple을 잘못 unpack한 사건

reference norm API는 입력 x에서 normalized n만 반환한다. fused API는 `(n, residual)` 또는 `(residual, n)`을 반환할 수 있고, residual 인자가 `None`인지에 따라 반환 형태가 달라질 수도 있다. caller가 tuple order를 잘못 가정하면 shape와 dtype이 같은 두 `[T,D]`가 뒤바뀌어 QKV가 raw residual을 읽고 다음 add가 normalized branch를 residual로 사용한다.

작은 fixture에서 x=`[1,2,3,4]`, normalized n의 RMS가 약 1이라고 하자. 두 tensor shape는 `[1,4]`로 같다. QKV input norm이 raw x의 RMS 약 2.74인지 normalized 약 1인지 보면 분기가 드러난다. 정확한 epsilon과 weight를 포함한 reference를 만들어 elementwise 비교한다. tuple 변수 이름이 아니라 return statement와 caller unpack, next consumer를 잇는다.

이 사건은 첫 layer부터 크게 틀릴 수 있지만 일부 weight와 residual path에서 유한한 출력이 계속돼 명시적 error가 없을 수 있다. non-finite metric만으로 잡히지 않는다. pre-norm checkpoint와 post-add checkpoint를 둘 다 두는 이유다. 수정 뒤 residual carrier가 다음 add까지 보존되고 normalized branch만 projection으로 들어가는지 assertion한다.

### 10.12.6 hidden-state 관측의 보안과 cardinality

activation은 사용자 입력의 정보를 포함할 수 있다. production에서 raw tensor를 무제한 저장하지 않는다. synthetic fixture를 우선하고, 실제 incident에는 승인된 bounded row와 짧은 보존 기간, 암호화, 접근 감사를 적용한다. digest도 같은 입력의 반복 여부를 드러낼 수 있으므로 keyed digest와 tenant scope를 고려한다.

Prometheus label에는 request ID, layer별 digest, pointer를 넣지 않는다. `checkpoint_stage`, `first_divergence_bucket`, `nonfinite`, `selected_path`, `hook_mode` 같은 bounded enum과 counter/histogram을 사용한다. 상세 artifact는 trace 또는 incident store에 연결한다. metric은 범위를 찾고 artifact는 원인을 증명한다.

### 10.12.7 같은 transaction으로 source review와 배포 판정을 닫는다

10.1 transaction의 열을 그대로 사용한다. `semantic checkpoint`, `producer`, `consumer`, `alias contract`, `parallel state`,
`last async consumer`, `falsifier fixture`를 두 norm, 두 branch, 두 add와 layer output에 채운다. 별도의 review card를
새로 만들지 않는다. Pinned revision과 span, 확인된 사실과 추론을 구분한다.

배포 전에는 hook off/on, output-hidden-states off/on, eager/graph, TP/PP 대표 조합에서 selected path와 correctness checkpoint를 확인한다. debug 기능이 production graph를 자동으로 깨뜨린다면 이를 지원 제한으로 노출하고 안전한 fallback의 비용을 측정한다. 제한을 숨기고 느려진 원인을 model compute로 보고하지 않는다.

배포 후에는 peak retained activation bytes, graph fallback, checkpoint collection failure, non-finite와 sampled parity를 generation별로 본다. 관측 기능 자체가 latency p99를 목표 이상 악화하면 sampling을 줄이거나 격리한다. correctness 사고를 조사하기 위해 켠 probe가 새 capacity incident를 만들지 않게 한다.

최종 terminal은 네 질문에 답한다. hook이 가리키는 수학 좌표는 무엇인가. 저장 artifact는 snapshot인가 live alias인가. TP/PP에서 local partial인가 global/transport complete인가. 마지막 비동기 consumer가 끝날 때까지 storage lifetime은 누가 보장하는가. 이 답과 fixture가 있으면 Llama block의 activation 설명은 건조한 함수 목록이 아니라 실제 장애를 좁히는 지도다.

10장은 이제 한 forward의 큰 지도와 관측의 정확한 좌표를 함께 제공한다. 11장으로 넘어갈 때는 `x0,n1,a,x1,n2,m,x2`의 owner와 lifetime이 닫혀 있다. 다음 장은 이 가운데 RMS reduction, fused add-norm tuple, packed projection storage를 수치와 stride 수준으로 해부한다.

## 10.13 실제 리뷰에서 놓치기 쉬운 경계 사례

첫 번째는 gradient checkpointing 이름에 끌려 training 동작을 그대로 추정하는 경우다. 이 책은 serving forward를 다루지만 model source에 training용 branch와 inference branch가 함께 있다. `self.training`, gradient enabled, cache use, compile 조건에 따라 layer 호출 wrapper가 달라질 수 있다. 현재 serving lane에서 유효한 predicate를 먼저 고정하고, 실행되지 않는 training branch를 activation lifetime 설명에 섞지 않는다.

두 번째는 tuple/dataclass 반환을 tensor 하나로 보는 경우다. attention은 output, weights, cache update를 함께 반환할 수 있고 model layer는 hidden과 residual carrier를 묶을 수 있다. index 0만 다음 hidden이라고 추정하지 말고 type definition, return site, unpacking consumer를 모두 본다. 버전 변경에서 tuple 순서가 바뀌거나 optional field가 추가되면 positional unpacking이 위험해질 수 있다.

세 번째는 `residual`이라는 이름이 항상 layer input을 뜻한다고 믿는 경우다. fused add-norm API에서는 residual carrier가 이전 residual과 branch update를 이미 더한 값일 수 있다. 첫 norm 호출에서 residual이 `None`이고 이후에는 누적 carrier가 전달될 수도 있다. 각 호출의 입력 식을 쓰고, 같은 변수 이름의 layer별 의미 변화를 ledger에 기록한다.

네 번째는 hidden state collection이 layer output을 보존한다고 자동 가정하는 경우다. 어떤 model loop는 layer 호출 전에 현재 hidden을 tuple에 추가하고 마지막 final norm 뒤 출력을 따로 추가한다. 그러면 `hidden_states[i]`와 decoder layer i output의 index 관계가 한 칸 다를 수 있다. 고정 source의 append 위치와 final append를 읽어 checkpoint 번호를 맞춘다.

다섯 번째는 forward hook 순서를 GPU completion 순서로 해석하는 경우다. Python hook callback 순서는 module call 구조를 보여 주지만 device kernel 완료와 collective 완료를 직접 보장하지 않는다. hook에서 즉시 값을 읽는 연산이 implicit synchronization을 만들 수도 있다. timeline 설명에는 host enqueue와 device completion을 분리한다.

여섯 번째는 residual add의 commutativity 때문에 row identity가 중요하지 않다고 생각하는 경우다. 같은 row 안에서 `x+a`는 순서를 바꿔도 같지만 서로 다른 request row의 a가 섞이면 전체 tensor 합계가 우연히 같을 수 있다. request-position key로 row를 비교하고 aggregate checksum만 사용하지 않는다.

일곱 번째는 in-place 최적화가 항상 메모리를 절약한다고 단정하는 경우다. alias 때문에 이전 activation을 보존하려는 consumer가 copy를 만들거나 compiler가 functionalization을 수행하면 peak가 오히려 늘 수 있다. graph pool과 allocator fragmentation, hidden-state retention을 포함한 effective peak를 측정해야 한다. source의 underscore 하나는 실제 memory delta를 증명하지 않는다.

여덟 번째는 activation lifetime을 Python garbage collection으로 설명하는 경우다. device allocator, stream recording, graph pool, communication library가 별 생명주기를 갖는다. reference count가 0이 된 시점은 재사용 가능성의 한 입력일 뿐이다. custom op가 external pointer를 보유한다면 framework가 이를 알 수 있는 contract가 필요하다.

아홉 번째는 PP receive가 source stage output과 동일 dtype/layout이라고 가정하는 경우다. transport 효율을 위한 cast, contiguous pack, sequence partition, padding이 있을 수 있다. send manifest와 receive manifest를 비교하고 next layer가 이를 어떤 logical row로 해석하는지 본다. checksum은 canonical bytes 규칙이 같을 때만 의미가 있다.

열 번째는 adapter가 branch output만 바꾸고 residual contract에는 영향이 없다고 보는 경우다. adapter dispatch가 row별로 다르면 projection update의 request mapping이 달라지고 residual add가 이를 증폭한다. mixed-adapter batch fixture에서 row identity와 adapter generation을 붙인다. 단일 adapter 정상은 dispatch correctness를 증명하지 않는다.

열한 번째는 quantization 차이를 모두 허용 오차로 처리하는 경우다. quantized projection의 예상 수치 차이와 wrong scale group, wrong packed row, stale weight generation은 다르다. first divergence가 group boundary나 특정 rank에 집중되면 구조 가설을 먼저 본다. tolerance는 artifact가 선언한 정상 approximation envelope 안에서만 사용한다.

열두 번째는 cancellation cleanup을 성능 경로로만 보는 경우다. 취소된 request의 activation/KV/recurrent slot이 새 request에 섞이면 cross-request correctness와 격리 문제다. 즉시 rollback 문턱을 낮게 두고 failing/passing request generation을 보존한다. 평균 quality metric이 정상이어도 한 건의 contamination은 중대하다.

이 경계 사례를 한 표로 외울 필요는 없다. 모두 같은 네 질문으로 돌아간다. 지금 tensor의 수학적 의미는 무엇인가. 누가 이 storage와 logical rows를 소유하는가. 마지막 consumer는 언제 끝나는가. 다음 consumer는 local partial과 global complete 중 무엇을 기대하는가. 네 답 중 하나가 비면 source와 artifact를 더 읽는다.

리뷰 결과는 `confirmed`, `inferred`, `unknown`, `not-applicable` 네 상태로 쓴다. 정적 source가 alias 가능성을 보여도 실제 compiled lane의 reuse는 `inferred`일 수 있다. serving에서 training branch는 `not-applicable`이다. probe 없이 runtime race가 발생했다고 확정하지 않는다. 상태가 명확하면 다음 담당자가 추론을 사실로 이어받지 않는다.

마지막으로 변경 diff를 읽을 때 이 좌표를 사용한다. residual add가 fused norm으로 이동했다면 hook coordinate와 tuple ownership, lifetime terminal이 바뀐다. parallel linear의 reduce flag가 바뀌면 partial/complete checkpoint와 residual add 위치가 바뀐다. graph capture 지원이 추가되면 address generation과 hook fallback이 바뀐다. 변경 파일 수보다 어떤 invariant가 이동했는지를 기록한다.

10장의 최종 산출물은 layer별 함수 목록이 아니라 source와 tensor를 잇는 관측 계약이다. Llama block 하나를 열었을 때 pre/post norm과 두 residual add를 정확히 지목하고, 숫자 fixture로 alias를 구별하며, TP/PP와 async lifetime까지 질문할 수 있어야 한다. 그 능력이 다음 장의 fused kernel과 packed projection을 안전하게 읽는 전제다.

실전에서는 첫 divergence가 layer 경계와 정확히 겹치지 않을 수 있다. layer input은 맞고 output이 다르면 attention과 MLP 사이를 다시 나눈다. post-attention residual이 맞으면 attention branch 전체를 닫고 MLP pre-norm부터 본다. post-attention residual이 다르면 pre-norm, QKV, cache/attention, O projection, collective, first add 순서로 좁힌다. 무조건 모든 내부 tensor를 저장하지 않고 앞 단계 parity가 확인될 때 다음 probe를 선택한다.

두 구현의 block 순서가 다르면 억지로 같은 hook index를 맞추지 않는다. parallel residual처럼 attention과 MLP가 같은 normalized input에서 나와 한 번에 더해지는 architecture는 `x1`이라는 중간 상태가 없을 수 있다. sandwich norm이나 residual scaling도 checkpoint 의미를 바꾼다. 공통 비교점은 architecture가 실제로 제공하는 수학 식에서 다시 정한다.

Llama라는 모델 이름도 구현 동일성을 보장하지 않는다. reference library, serving runtime, quantized loader, CUDA graph model runner가 wrapper와 fused op를 추가한다. config와 checkpoint가 같은 것은 입력 artifact가 같다는 뜻이고, activation lifetime과 collective 위치까지 같다는 뜻이 아니다. source walk는 model class에서 custom op와 runtime consumer까지 내려가야 한다.

probe가 가설을 반증하지 못하면 제거한다. 모든 layer의 mean만 수집하면서 row permutation을 찾으려는 관측은 비용만 만들고 답을 주지 않는다. alias 가설에는 producer 시점 snapshot과 live reference 비교, collective 가설에는 local partial과 global complete, lifetime 가설에는 event generation과 reuse incarnation이 필요하다. 질문에 맞는 artifact를 선택한다.

회귀 테스트도 동일하다. 최종 token parity 하나는 수많은 내부 오류를 우연히 통과시킬 수 있고, 모든 activation bitwise equality는 정상적인 수치 backend 차이를 거부할 수 있다. 구조 invariant는 exact하게, dtype 수치는 선언된 tolerance로, semantic output은 별 품질 기준으로 판정한다. 세 terminal을 한 boolean으로 합치지 않는다.

운영 handoff에는 수정 범위와 남은 uncertainty를 쓴다. 관측 alias를 고쳤지만 async lifetime race는 실행 검증하지 못했다면 완료로 포장하지 않는다. 정적 source에서 확인한 dependency와 필요한 synthetic overlap test를 구분한다. 반대로 first divergence가 norm 이후라면 tokenizer와 scheduler를 다시 의심 목록에 넣지 않는다.

이러한 disciplined stop rule이 소스 깊이와 독자 친절성을 동시에 만든다. 독자는 모든 함수 이름을 외우지 않아도 현재 값이 어느 checkpoint인지, 무엇과 비교할지, 다음에 어느 producer/consumer를 열지 알 수 있다. 디테일은 목록의 양이 아니라 조사 결정을 바꾸는 좌표에서 가치가 생긴다.

배포 revision 비교에서는 checkpoint contract의 이동을 별도로 감사한다. 예전에는 decoder layer가 normalized branch만 반환했는데 새 fused layer가 residual tuple을 반환한다면 caller unpack, hidden-state collection, PP transport가 모두 영향을 받을 수 있다. 함수 signature가 호환돼도 tuple의 의미와 storage lifetime이 바뀌면 관측 도구와 cache/graph 경계가 달라진다.

diff review는 변경 전후 source span, 수학 식, 반환 값, alias annotation, collective 위치, last consumer를 한 행에 놓는다. 새 최적화의 의도도 이 행에서 설명한다. 예를 들어 residual을 fused norm에 넘기는 이유가 memory read/write와 kernel launch를 줄이는 것이라면, 절약하는 buffer와 새로 생기는 alias/lifetime 책임을 함께 적는다.

성능 기대는 상한으로 계산한다. `[T,D]` BF16 residual을 별 kernel에서 읽고 쓰던 경로를 fusion으로 없애면 최소한 해당 tensor의 read/write traffic 일부와 launch를 줄일 수 있다. 그러나 norm reduction, projection GEMM, collective가 지배하면 end-to-end 개선은 작을 수 있다. workload shape와 selected backend 없이 보편적인 속도 배수를 쓰지 않는다.

수정이 들어간 뒤에는 old/new lane을 같은 fixture로 비교한다. semantic checkpoint parity, peak retained bytes, graph eligibility, launch/collective 구간을 함께 본다. 빠르지만 output-hidden-state mode에서 alias artifact를 반환하거나 cancellation overlap에서 race가 생기면 승인하지 않는다. 느리지만 정확한 fallback은 rollback lane으로 보존한다.

이제 다음 장의 독자는 RMSNorm과 packed projection을 독립 연산으로만 보지 않는다. 그 입력과 출력이 어느 residual storage를 공유하고, 어느 순간 global complete가 되며, 누가 마지막으로 buffer를 읽는지를 이미 알고 있다. 이 연결이 fused kernel의 “왜”를 실제 serving 안정성과 결합한다.

장 종료 시에는 failing fixture와 passing neighbor의 checkpoint ledger를 함께 보관한다. 값이 일치한 경계도 삭제하지 않는다. 어디까지 정상인지가 다음 조사 범위를 제한하는 강한 증거이기 때문이다. 수정 후에는 바로 그 최초 divergence와 앞선 정상 checkpoint를 다시 비교해 회귀가 이동하지 않았는지 확인한다.

남은 미확인 경계에는 담당 symbol, 필요한 probe, 승인 조건을 붙여 다음 조사에서 추측을 반복하지 않게 한다.
