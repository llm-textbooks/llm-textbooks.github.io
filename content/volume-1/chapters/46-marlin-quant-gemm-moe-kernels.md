# 46장. 4비트 packed weight가 정확한 답이 되기까지: Marlin과 MoE kernel의 데이터 생애

모델 서버가 시작되지 않는다. 로그에는 “Marlin이 이 shape를 지원하지 않는다”는 문장이 나온다. 다른 모델은 시작하지만 첫 token부터 reference와 값이 다르다. 세 번째 모델은 정답은 맞는데 BF16 모델보다 오히려 느리다. 세 모델의 디렉터리에는 모두 “4bit”가 적혀 있다. 운영자는 quantization을 하나의 기능으로 생각했지만, 실제 실패 지점은 서로 다르다.

첫 모델은 tensor-parallel shard 뒤의 K 또는 N이 kernel tile 제약과 맞지 않았다. 둘째 모델은 AWQ zero-point를 GPTQ symmetric weight처럼 해석하거나 scale을 Marlin이 기대하는 순서로 재배치하지 않았다. 셋째 모델은 작은 weight byte에서 얻은 이득보다 activation, dequantization, workspace reduction과 큰 batch의 compute 비용이 더 컸다. “4비트”는 원인도 kernel 이름도 아니다. value encoding, scale granularity, zero-point, group index, packed layout, activation dtype, architecture gate가 함께 있어야 하나의 실행 계약이 된다.

MoE 모델에서는 문제가 한 단계 더 늘어난다. router가 token마다 top-k expert를 고르면 dense batch가 곧바로 expert GEMM이 되는 것이 아니다. `(token, expert)` pair를 세고, expert별로 모으고, block 크기에 맞춰 padding하고, 정렬된 row와 원래 token의 관계를 보존해야 한다. gate/up GEMM과 activation, down GEMM을 지난 결과는 router weight를 정확히 한 번 적용해 원래 token 순서로 합쳐야 한다. packed expert weight가 맞아도 permutation이나 combine이 틀리면 최종 답은 틀린다.

두 canonical fixture가 packing과 expert routing의 좌표 보존을 끝까지 연결한다.

이 장의 중심 질문은 하나다. `Y=XW`라는 수식이 dense BF16 GEMM에서 W4A16 Marlin, FP8,
NVFP4, quantized MoE로 바뀔 때 어떤 byte와 좌표가 새로 생기는가. 먼저 한 logical weight가
code·scale·zero point·packed word·kernel lane으로 바뀌는 생애를 따라가고, 구현 revision과
논문이 각각 어느 주장을 지지하는지는 장말에서 분리해 고정한다.

> **먼저 구분할 네 층.** Logical weight는 수식이 뜻하는 값이다. Checkpoint payload는 그 값을 code, scale, zero point와 packed container로 저장한 표현이다. Loader와 repack은 payload의 물리적 순서를 바꿀 수 있고, kernel consumer는 자신이 약속한 layout과 scale stride로 그것을 읽는다. Shape와 byte 수가 맞아도 이 네 층의 좌표가 어긋나면 값은 틀린다. 포맷별 scale axis, pack order와 runtime ABI는 50장에서 같은 logical weight를 기준으로 다시 맞춘다.

## 46.1 dense values에서 byte 기준선을 세운다

### 46.1.1 수식은 같지만 저장 계약은 달라진다

Dense linear layer를 `X[M,K] · W[K,N] = Y[M,N]`으로 쓰자. BF16이라면 X와 W의 각 원소는 2 byte이고 accumulator는 구현에 따라 FP32일 수 있다. row-major인지 column-major인지, transpose view인지, tensor core tile에 어떻게 공급하는지는 별도지만 logical element `(k,n)`의 값은 `W[k,n]` 하나다.

Quantized weight에서는 logical element와 저장 element가 갈라진다. 4-bit code 두 개가 한 byte에 들어갈 수 있고 여덟 code가 int32 하나에 pack될 수 있다. code `q`만으로 원래 weight를 복원하지 못한다. symmetric group quantization이면 대략 `w=s·q`, asymmetric이면 `w=s·(q-z)`다. 어느 `s`와 `z`를 쓸지는 group coordinate가 정한다.

따라서 최소 네 좌표가 필요하다. 첫째 logical matrix coordinate `(k,n)`다. 둘째 packed-word 안의 nibble 또는 bit-field coordinate다. 셋째 scale/zero-point의 group coordinate `(floor(k/G),n)` 또는 format이 정한 block coordinate다. 넷째 kernel tile에서 lane과 shared/register fragment가 맡는 execution coordinate다. repack은 logical 값은 보존하면서 둘째와 넷째 좌표의 대응을 바꾸는 과정이다.

오답을 찾을 때 이 네 좌표를 한꺼번에 보면 안 된다. checkpoint qweight의 int32 word에서 특정 `(k,n)` code를 먼저 복원한다. 그다음 scale과 zero-point group을 적용해 reference float weight를 얻는다. repacked Marlin tile에서 같은 logical element를 다시 복원한다. 마지막에 kernel의 lane mapping과 MMA output을 본다. 첫 divergence를 앞에서부터 찾으면 packing 오류와 GEMM 오류를 구분할 수 있다.

### 46.1.2 dense와 W4의 byte fixture

`M=8`, `K=4096`, `N=11008`인 linear를 생각하자. BF16 dense weight의 logical payload는 다음과 같다.

`4096 × 11008 × 2 = 90,177,536 byte`

MiB로 나누면 약 86 MiB다. X는 `8×4096×2 = 65,536 byte`, Y는 `8×11008×2 = 176,128 byte`다. 이 작은 M에서는 weight byte가 input과 output보다 훨씬 크다. weight-only quantization이 decode와 작은 batch에서 매력적인 첫 이유다.

ideal INT4 weight code는 원소당 0.5 byte이므로 `4096×11008÷2 = 22,544,384 byte`, 약 21.5 MiB다. dense weight의 4분의 1이다. 그러나 이것은 전체 format 크기가 아니다. group size `G=128`, FP16 scale 하나를 각 group과 output channel에 둔다면 scale 수는 `(4096/128)×11008 = 352,256`, payload는 `704,512 byte`다.

asymmetric 4-bit zero-point를 같은 granularity로 ideal하게 pack하면 `352,256÷2 = 176,128 byte`다. 실제 checkpoint나 Marlin representation은 int32 packing, interleave, padding 때문에 shape와 byte가 달라질 수 있다. 합계의 단순 하한은 `22,544,384 + 704,512 + 176,128 = 23,425,024 byte`다. 여기에 bias, metadata, workspace와 alignment가 더해진다.

“4배 압축”이 아니라 `90,177,536 / 23,425,024 ≈ 3.85`라는 logical format 비율이 된다. symmetric이면 ZP가 없어 조금 달라지고, channelwise scale이면 scale 수가 줄며, smaller group이면 scale/ZP overhead가 늘어난다. group size는 accuracy와 metadata, dequant address 계산을 함께 바꾼다.

이 byte fixture는 bandwidth 측정값이 아니다. cache hit와 transaction, tile reuse, repeated load를 반영하지 않는다. 다만 weight traffic이 전체에서 큰 shape인지, scale overhead가 무시할 수준인지, padding이 몇 퍼센트인지 확인할 기준을 준다. 관측 byte가 이 logical 하한보다 크게 다르면 반복 load, padding, temporary, 다른 tensor를 목록에 추가한다.

### 46.1.3 batch가 커지면 교환비가 달라진다

weight는 batch의 여러 rows가 공유한다. M이 8에서 128로 커지면 X와 Y payload, MMA 수는 16배가 되지만 weight matrix 크기는 그대로다. cache와 tile reuse가 충분하면 weight load 한 번당 더 많은 compute가 생긴다. W4로 줄인 weight byte의 상대적 이득은 남지만 unpack, scale, activation movement와 tensor-core compute가 더 중요한 비중을 가질 수 있다.

MARLIN 논문이 autoregressive parallel inference와 여러 batch 범위를 함께 논하는 이유가 여기에 있다. weight-only kernel은 single-token memory movement만 줄이는 데서 끝나지 않고, batch가 늘어 compute requirement가 커져도 packed-load와 dequant/MMA pipeline이 병목을 지나치게 만들지 않도록 설계해야 한다. 그러나 논문의 speedup을 현재 vLLM의 임의 shape에 옮기지 않는다.

prefill처럼 M이 큰 경우 dense BF16 GEMM도 tensor core를 잘 채우고 높은 arithmetic intensity를 만들 수 있다. W4A16 kernel은 low-bit weight를 BF16/FP16 operand로 dequant하는 추가 instruction과 layout 제한이 있다. kernel availability와 shape가 좋지 않으면 dense backend가 더 유리할 수 있다. quantization은 저장 용량 이득과 모든 shape에서의 속도 이득을 보장하는 단일 스위치가 아니다.

## 46.2 packed weight와 scale·zero-point representation을 만든다

### 46.2.1 shape 검증과 padding

loader가 읽은 logical K/N은 tensor parallel shard 후 per-rank K/N으로 바뀔 수 있다. row-parallel linear는 K가 나뉘고 column-parallel linear는 N이 나뉜다. group size가 shard 경계를 가로지르면 각 rank에 필요한 scale group을 반복하거나 partition 규칙을 바르게 적용해야 한다. act-order가 있으면 permutation도 shard coordinate와 맞아야 한다.

`verify_marlin_supports_shape`는 output partition이 minimum thread N으로, input partition이 minimum thread K로 나뉘는지 본다. group size가 input보다 작으면 partition을 나누는지도 확인한다. 실패 메시지가 generic “unsupported GPU”가 아니라 shape와 TP 설정을 가리킬 수 있는 이유다.

`marlin_padded_nk`는 두 tile family 중 padding 비용이 작은 후보를 선택하되 padded K가 group size로 나뉘도록 한다. 예를 들어 `N=11008`은 64로 나뉘지만 K가 제약에 맞는지 별도다. 어떤 모델의 intermediate size가 11008에서 TP=3으로 나뉘면 정수 shard와 tile alignment부터 문제가 된다. model architecture의 예쁜 dimension도 per-rank kernel dimension이 예쁘다는 보장은 없다.

padding overhead는 `(padded_K×padded_N)/(K×N)-1`로 계산한다. N이 11008에서 11072로 늘고 K가 그대로라면 약 `64/11008≈0.58%` weight code와 scale column overhead가 생긴다. 작은 layer에서는 비율이 더 클 수 있다. padding이 support를 넓히지만 무조건 무료가 아니다.

### 46.2.2 qweight repack

checkpoint의 qweight는 보통 logical K/N을 따라 int32 words에 code를 pack한다. Marlin repack은 kernel의 vectorized global load, shared layout, warp MMA 소비 순서에 맞도록 tile 내부 code 순서를 바꾼다. logical matrix transpose와 bit interleave, tile permutation이 결합될 수 있다.

repack validation fixture는 random float GEMM output만 비교하기 전에 integer code identity를 본다. 각 logical `(k,n)`에 고유한 작은 code pattern을 넣고 checkpoint unpack 결과와 Marlin unpack 결과를 비교한다. code가 반복되면 permutation 오류가 숨으므로 coordinate에서 만든 pattern을 쓴다. 4-bit range를 넘지 않게 `(ak+bn) mod 16`처럼 만든다.

TP shard에서도 같은 검사를 한다. packed dimension의 shard offset은 logical element offset에 packed factor와 Marlin tile size를 적용해야 한다. vLLM parameter loader가 `marlin_tile_size`를 별도로 보존하는 까닭이다. int32 tensor shape만 보고 logical K/N shard를 자르면 tile 중간을 잘라 잘못된 expert나 output channel을 읽을 수 있다.

### 46.2.3 scale permutation과 zero-point packing

현재 vLLM의 [`marlin_utils.py` 460–533행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L460-L533)은 scale permutation을 만든다. groupwise W4와 channelwise 또는 activation-8bit 경로에서 permutation 모양이 다르다. 마지막에는 `(-1,size_n)` contiguous tensor로 만든다.

zero-point는 scale과 비슷한 group order를 따르지만 동일 객체가 아니다. 4-bit이면 `[0,2,4,6,1,3,5,7]`, 8-bit이면 `[0,2,1,3]` 같은 column interleave를 적용하고 int32로 pack한다. kernel의 dequant code가 MMA operand 조각마다 원하는 ZP를 바로 가져오게 하기 위한 배치다.

scale은 float/BF16/FP16 값이고 ZP는 low-bit integer라는 차이도 있다. asymmetric dequant에서 둘의 좌표가 하나라도 어긋나면 `s(q-z)` 전체가 systematic하게 이동한다. output 오차가 random rounding처럼 보이지 않고 특정 output channel이나 K group마다 bias를 갖는다. first-divergence에서 group boundary를 검사하는 이유다.

AWQ zero-point conversion은 checkpoint convention의 offset과 Marlin scalar type convention을 맞춘다. 일부 unsigned 4-bit representation은 logical zero를 특정 bias code로 표현한다. code 0이라는 bit pattern이 float zero라는 가정을 format마다 반복하면 안 된다. `ScalarType`의 bias와 min/max, dequant helper를 함께 본다.

### 46.2.4 act-order와 K permutation

GPTQ act-order에서 `g_idx`는 K column이 어느 scale group에 속하는지 나타낸다. `marlin_sort_g_idx`는 `argsort`를 만들고 sorted group index와 column permutation을 보존한다. weight가 sorted K order로 repack되면 activation X의 K columns도 같은 순서로 읽어야 dot product가 보존된다.

간단히 `X=[x0,x1,x2]`, weight column vector가 `[w0,w1,w2]`, permutation이 `[2,0,1]`이라고 하자. weight를 `[w2,w0,w1]`로 바꾸고 X를 그대로 두면 `x0w2+x1w0+x2w1`이 되어 다른 값이다. X도 `[x2,x0,x1]`로 바꾸면 원래 dot product가 된다. permutation은 weight-only metadata가 아니라 양 operand의 공동 계약이다.

row-parallel shard에서 K 전체가 한 rank에 없으면 act-order permutation과 group scale을 어떻게 나눌지가 복잡해진다. `is_k_full`, scale repeat 정책, empty `g_idx`가 kernel argument에 들어간다. empty tensor가 “group 없음”인지 “feature disabled”인지 API contract를 확인해야 한다. null pointer와 zero-length tensor도 dispatcher 분기에 다른 의미를 가질 수 있다.

## 46.3 packed tile을 load·dequant·MMA한다

### 46.3.1 Python call이 넘기는 완전한 상태

현재 vLLM의 [`apply_gptq_marlin_linear` 698–765행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L698-L765)은 input을 2D로 펴고 repacked weight에서 padded N/K를 복원한다. X의 K도 padded K로 맞춘다. M/N/K와 device/dtype에 따라 atomic add reduction을 선택할 수 있다.

kernel argument에는 X, packed weight, optional bias, weight scale, optional activation scale, optional global scale, weight zero-point, `g_idx`, sort indices, workspace, scalar type가 함께 들어간다. `size_m/n/k`, `is_k_full`, atomic과 FP32 reduction flag도 전달된다. “Marlin weight tensor 하나”가 kernel 계약 전부가 아니다.

activation이 INT8 또는 FP8인 mixed path에서는 X를 per-token quantize하고 activation scale을 만든다. W4A16과 달리 activation code와 scale byte, quantization 산술이 추가된다. supported weight scalar type 조합을 assert한다. W4A16 Marlin 결과를 W4A8 또는 FP8 activation path의 성능과 정확성으로 일반화하지 않는다.

### 46.3.2 workspace는 scratch byte 이상이다

[`marlin_make_workspace_new` 408–433행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L408-L433)은 SM 수와 `max_blocks_per_sm`의 곱만큼 int32 workspace를 만든다. 여러 threadblocks가 output tile을 나누어 계산할 때 lock/counter 또는 reduction coordination에 쓰이는 state다.

weight reload에서 기존 workspace storage를 재사용한다. device, dtype, numel이 기대와 다르면 오류를 내고 같으면 zero한다. graph capture가 workspace 주소를 기억할 수 있기 때문이다. 내용만 같은 새 tensor를 만들면 pointer identity가 바뀌어 captured graph가 오래된 주소를 참조할 수 있다. workspace의 correctness invariant는 크기뿐 아니라 주소 안정성이다.

MoE Marlin은 `max_blocks_per_sm=4` 같은 더 큰 coordination 예산과 FP32 temporary output을 가질 수 있다. workspace라는 이름 아래 int32 semaphore state와 large FP32 reduction buffer를 섞지 않는다. 각 tensor의 shape, dtype, zeroing, lifetime, graph-capture ownership을 따로 기록한다.

### 46.3.3 global packed load와 staging

native source의 [`kernel.h` 1–43행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/csrc/libtorch_stable/quantization/marlin/kernel.h#L1-L43)은 `Marlin` kernel entry의 ABI와 template 축을 선언한다.

실제 async-copy helper와 stage 상수는 포함된 [`marlin.cuh` 23–166행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/csrc/libtorch_stable/quantization/marlin/marlin.cuh#L23-L166)에서 확인한다. generated specialization은 quant type, thread tile, group과 zero-point/act-order/reduction option을 compile-time 또는 dispatch parameter로 고정한다.

packed B는 dense BF16보다 적은 global byte로 load된다. threadblock은 X tile, packed B tile, scale/ZP를 shared/register pipeline에 staging한다. asynchronous copy와 double buffering은 다음 tile movement를 현재 MMA와 겹치려는 의도다. stage가 늘면 shared와 register address state가 늘어난다는 41장의 trade-off가 그대로 적용된다.

packing 순서는 vector load가 필요한 contiguous word를 가져오고 warp가 소비할 code를 낮은 shuffle 비용으로 얻도록 설계된다. checkpoint row-major qweight를 kernel hot loop에서 매번 복잡하게 gather하는 대신 model load 시 한 번 repack 비용을 지불한다. model load latency와 persistent weight layout을 바꾸어 token step의 비용을 줄이는 serving-level 교환이다.

### 46.3.4 unpack과 dequant

[`dequant.h` 1–240행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/csrc/libtorch_stable/quantization/marlin/dequant.h#L1-L240)은 packed code를 MMA operand가 소비할 fragment로 바꾸는 중심이다. bit mask, shift, permutation으로 low-bit values를 분리하고 scalar type의 bias와 ZP를 적용한 뒤 scale을 곱는다.

중요한 점은 dense dequantized W matrix를 global에 쓰지 않는다는 것이다. packed global byte를 load해 register/shared 범위에서 필요한 tile만 복원하고 곧바로 MMA에 넣는다. weight storage 절감이 실행 traffic 절감으로 이어지려면 dequant intermediate의 lifetime을 on-chip tile에 제한해야 한다.

scale granularity는 hot loop의 address pattern을 바꾼다. channelwise scale은 N channel마다 하나를 재사용하고 groupwise scale은 K group boundary마다 바뀐다. group size가 작으면 scale payload와 load 빈도가 늘지만 weight approximation은 더 세밀할 수 있다. act-order에서는 group sequence가 정렬 metadata를 따른다.

zero-point가 없으면 symmetric specialization이 subtraction과 ZP load를 생략할 수 있다. ZP tensor가 empty인데 `has_zp` 분기가 잘못 켜지거나 반대라면 kernel argument stride부터 어긋난다. option은 성능 toggle이기 전에 ABI와 수식의 일부다.

### 46.3.5 MMA와 accumulator

[`marlin_mma.h` 1–220행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/csrc/libtorch_stable/quantization/marlin/marlin_mma.h#L1-L220)은 operand fragment와 MMA wrapper를 정의한다. dequantized B fragment와 X fragment가 MMA instruction이 요구하는 lane layout으로 들어가고 partial output accumulator가 register에 남는다.

W4A16이라는 이름은 MMA가 4-bit integer weight를 그대로 FP16 activation과 곱하는 단일 instruction이라는 뜻이 아니다. Marlin path는 packed weight를 복원해 지원되는 MMA operand representation으로 공급한다. architecture와 scalar type에 따라 instruction path가 다를 수 있다. PTX/SASS instruction을 확인하지 않고 tensor core opcode를 단정하지 않는다.

K를 여러 CTA가 나누면 partial accumulator를 합쳐야 한다. workspace coordination 후 owner가 reduce할 수도 있고 atomic add를 선택할 수도 있다. `use_fp32_reduce`는 reduction temporary와 rounding order를 바꾸며 `use_atomic_add`는 output write ownership과 determinism 특성을 바꾼다. 같은 packed multiplication이어도 epilogue 결과의 작은 차이가 생길 수 있다.

### 46.3.6 epilogue와 unpadding

모든 K tile이 누적되면 accumulator를 output dtype으로 변환하고 optional bias와 activation scale/global scale을 적용한다. 정확한 순서는 native specialization에서 확인한다. bias를 partial CTA마다 더하면 중복되므로 최종 owner 또는 reduction 뒤 한 번 적용해야 한다.

padded N columns는 output allocation에 존재할 수 있지만 caller는 original `output_size_per_partition`만 원한다. `marlin_unpad_output`이 마지막 dimension을 잘라 original shape를 복원한다. TODO가 말하듯 kernel이 처음부터 padded column을 쓰지 않는 것과 padded output을 만든 뒤 slice하는 것은 memory와 view contract가 다르다.

CUDA Graph 안에서 output view와 workspace address, padded shape가 고정돼야 할 수 있다. model reload나 TP reconfiguration으로 shape/storage가 바뀌면 graph key와 buffer ownership을 함께 갱신해야 한다. quantized kernel만 교체하고 capture metadata를 그대로 두면 pointer 문제로 이어진다.

## 46.4 MoE가 token을 expert tile로 다시 배열한다

### 46.4.1 router output은 아직 expert input이 아니다

MoE layer 입력을 `X[M,K]`, expert 수를 `E`, token당 선택 expert 수를 `T`라고 하자. router는 각 token의 expert logits를 만들고 top-k를 선택해 `topk_ids[M,T]`와 `topk_weights[M,T]`를 낸다. logical token-expert pair는 `M×T`개다.

예를 들어 token 0이 expert 2와 0, token 1이 1과 2를 선택했다고 하자. 원래 X rows는 token 순서지만 expert GEMM은 같은 expert weight를 쓰는 rows를 모으는 편이 효율적이다. token 0의 X row가 두 expert에 복제되어야 하고 token 1도 두 번 나타난다. 따라서 expert input batch는 일반적으로 M rows가 아니라 M×T logical rows다.

router weight를 언제 곱할지도 계약이다. GEMM1 input에 미리 곱을 수도 있고, GEMM2 output에 곱거나 final combine에서 곱을 수 있다. gated activation이 nonlinear이므로 임의로 앞뒤를 옮길 수 없다. source의 `apply_router_weight_on_input`과 `mul_topk_weights` 분기를 따라 정확히 한 번 적용되는지 확인한다.

### 46.4.2 count, alignment, padding

각 expert가 받은 pair 수 `count[e]`를 센다. grouped kernel의 row block이 `B_M`이면 expert별 실행 rows를 `ceil(count[e]/B_M)B_M`로 맞출 수 있다. 전체 padded rows는 `P=Σ_e ceil(count[e]/B_M)B_M`다. logical pair 대비 padding amplification은 `P/(M×T)`다.

`M=5`, `T=2`, `E=4`, `B_M=4`, counts가 `[4,3,2,1]`이면 logical pairs는 10이다. padded rows는 `4+4+4+4=16`, amplification은 1.6이다. kernel은 실제 token work보다 60% 많은 row slot을 배치한다. invalid slot은 sentinel token ID와 mask로 output에 기여하지 않아야 한다.

route가 `[10,0,0,0]`처럼 한 expert로 몰리면 padded rows는 `ceil(10/4)4=12`로 amplification이 1.2다. padding 비율은 좋아지지만 expert 병렬성은 하나로 줄어 load balance가 나빠진다. padding efficiency와 parallel expert utilization은 다른 축이다.

expert 수가 많고 M이 작은 decode에서는 빈 expert가 많다. 빈 expert마다 무조건 block을 만들면 낭비가 크므로 alignment kernel은 active expert와 padded block mapping을 효율적으로 만든다. expert-parallel에서는 local rank에 없는 expert ID를 sentinel로 바꾸거나 dispatch 통신으로 다른 rank에 보낸다. local expert map과 global ID를 혼동하면 다른 weight를 고른다.

### 46.4.3 sorted token IDs와 expert IDs

alignment 결과에는 보통 `sorted_token_ids`, block별 `expert_ids`, `num_tokens_post_padded`가 있다. sorted token ID는 원래 token-expert pair의 flattened coordinate를 보존해야 한다. `pair_id = token*T + topk_slot`처럼 역산할 수 있어야 final combine이 원래 token과 router weight를 찾는다.

expert block의 모든 valid row는 같은 expert weight를 소비한다. `expert_ids[block]`가 packed weight의 expert axis를 선택한다. expert parallel mapping이 있으면 global expert에서 local weight index로 변환한다. out-of-rank expert는 연산하지 않거나 transferred input만 받는다.

sentinel padded token은 X를 load하지 않거나 zero를 공급해야 한다. invalid row가 arbitrary memory를 읽고 결과를 쓰면 correctness와 memory safety 문제가 된다. activation과 GEMM2에서도 padding row를 유지할지 compact할지 source lifetime을 따른다. GEMM1만 mask하고 activation이 uninitialized padded row를 읽는 경로가 없는지 본다.

### 46.4.4 Marlin MoE intermediate cache 손계산

current vLLM의 [`marlin_moe.py` 57–126행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L57-L126)은 `M,K,N`과 gated activation shard 수를 얻는다. top-k pair마다 gate/up output과 activated intermediate, down output을 보관할 cache view를 만든다.

`M=5`, `T=2`, `K=4096`, `N=14336`, gated activation이라 gate/up shard가 2라고 하자. BF16 `intermediate_cache1` logical shape는 `(10,28672)`이고 payload는 `10×28672×2 = 573,440 byte`. cache2 `(10,14336)`은 286,720 byte, cache3 `(10,4096)`은 81,920 byte다.

source는 cache1과 cache3에 같은 `intermediate_cache13` storage를 resize view로 사용할 수 있도록 최대 `max(2N,K)`를 잡는다. 두 lifetime이 겹치지 않으면 peak는 합이 아니라 큰 쪽이다. gate/up output을 activation이 모두 소비한 뒤 storage를 down output으로 재사용할 수 있다. 이 alias의 correctness는 GEMM1/activation 완료와 GEMM2 output 시점에 달려 있다.

cache2는 activation output으로 GEMM2 input이다. activation을 INT8/FP8로 online quantize하면 cache2 dtype 또는 별도 quantized tensor와 a_scale가 생긴다. original BF16 cache가 즉시 해제/alias되는지 backend에 따라 다르다. W4 expert weight만 계산해 activation temporary를 빼면 peak memory를 과소평가한다.

### 46.4.5 GEMM1, activation, GEMM2

[`marlin_moe.py` 128–178행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L128-L178)은 optional activation quantization 후 W1 gate/up expert GEMM을 호출한다. argument에는 sorted token IDs, expert IDs, padded token count와 top-k weights가 포함된다. output N은 gated activation이면 `2N`이다.

activation은 SiLU와 multiply 같은 gated function을 적용해 `2N`을 N으로 줄인다. gate half와 up half의 layout이 interleaved인지 concatenated인지 kernel contract에 맞아야 한다. clamp나 model-specific SwiGLU variant가 있으면 generic `silu(gate)*up`으로 바꾸지 않는다.

[`marlin_moe.py` 180–224행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L180-L224)은 activated intermediate를 필요하면 다시 quantize하고 W2 down GEMM을 호출한다. GEMM2의 logical M은 `M×T`, N은 original K, K는 intermediate N이다. output은 각 token-expert pair의 K-dimensional contribution이다.

`apply_router_weight_on_input`가 거짓이면 GEMM2 쪽 `mul_topk_weights`가 참이 될 수 있다. 참이면 앞 단계에 적용하고 GEMM2에서 중복 적용하지 않는다. activation의 nonlinear 위치 때문에 weight 적용 방식을 바꿀 때 algebraic equivalence를 확인해야 한다. source comment와 model architecture contract가 필요하다.

### 46.4.6 combine과 원래 token 순서

pair output shape를 `(M,T,K)`로 보면 final output은 `Y[m,:]=Σ_t α[m,t]·O[m,t,:]`다. router weight가 이미 적용됐다면 combine은 sum만 하고, 아직이면 multiply와 sum을 함께 한다. invalid expert ID나 expert-parallel에서 처리되지 않은 slot은 zero contribution이어야 한다.

combine은 sorted order의 output을 원래 `(token,topk_slot)`으로 scatter/reduce할 수 있고, GEMM kernel이 original pair index로 output을 써 이미 contiguous view를 만들 수도 있다. `TopKWeightAndReduceDelegate`, `NoOP`, `Contiguous` 같은 policy가 이 차이를 표현한다. class 이름만 보고 실제 multiply 위치를 정하지 않고 selected modular kernel을 본다.

top-k weights의 renormalization도 router contract다. top-k만 고른 뒤 합이 1이 되도록 다시 나누는 모델과 원래 softmax probability를 그대로 쓰는 모델이 있다. combine kernel은 전달된 weights를 소비할 뿐 어떤 의미인지 추정하지 않아야 한다. router reference와 첫 `topk_weights` tensor를 고정한다.

## 46.5 dense→packed→scale→tile을 작은 fixture로 검산한다

### 46.5.1 큰 모델 output부터 비교하지 않는다

70B model의 마지막 logits가 다르다는 사실은 출발점일 뿐이다. 수십 개 layer와 nonlinear activation을 지난 오차는 최초 원인을 가리지 못한다. quantized linear 하나를 `M=2,K=128,N=64`, group size 32로 줄인다. K/N은 Marlin minimum tile에 맞고 M은 autoregressive small batch를 흉내 낸다.

X는 random half 대신 해석 가능한 pattern을 쓴다. 첫 row는 `x[k]=(k mod 7)-3`, 둘째 row는 특정 group만 nonzero로 만든다. quantized code는 `(3k+5n) mod 16`처럼 coordinate마다 달라지게 한다. scale은 group과 N을 식별할 수 있게 `s[g,n]=1+0.01g+0.001n`, zero-point도 작은 반복 pattern을 둔다.

이 fixture의 목적은 production accuracy를 예측하는 것이 아니다. transpose, nibble order, group index, scale column, ZP bias, tail padding 오류가 output pattern에 드러나게 한다. random input은 여러 오류가 상쇄되거나 결과만 보고 coordinate를 역추적하기 어렵다.

#### 46.5.1.1 단계 1: checkpoint code를 복원한다

checkpoint int32 word에서 logical `(k,n)`의 4-bit code를 CPU reference 식으로 꺼낸다. packed dimension이 K인지 N인지, 한 word 안의 element order와 signed/bias convention을 config와 loader source에서 확인한다. 몇 개 sample만 보지 않고 모든 K/N을 비교해 mismatch coordinate를 모은다.

첫 mismatch가 8 또는 16 element 주기로 반복되면 nibble interleave를 의심한다. N channel 전체가 서로 바뀌면 transpose나 shard offset을 본다. TP rank 경계부터 틀리면 packed-factor-adjusted shard slicing을 본다. 이 단계에서는 scale을 적용하지 않는다. code identity와 float dequant를 한 번에 보면 원인을 분리하기 어렵다.

#### 46.5.1.2 단계 2: dequantized logical weight

code가 맞으면 group index를 계산하고 `s(q-z)` 또는 scalar type이 정의한 식으로 float reference W를 만든다. act-order가 없으면 group은 보통 K 구간과 연결되지만 channelwise `group_size=-1` 의미를 별도 처리한다. act-order가 있으면 checkpoint `g_idx[k]`를 따른다.

각 `(k,n)`에 대해 code, group ID, scale coordinate/value, zero-point coordinate/value, final float를 tuple로 저장한다. mismatch가 group boundary `k=31/32`, `63/64`에서 시작하면 group size 또는 scale stride를 의심한다. 모든 element가 일정 offset이면 unsigned bias/ZP convention을 본다. 특정 N 묶음만 틀리면 scale permutation 전 checkpoint interpretation을 본다.

이 reference는 BF16 original W와 다를 수 있다. 그것이 정상 quantization error다. 이후 Marlin을 비교할 target은 original W가 아니라 여기서 만든 quantized logical W다. quantizer quality 평가는 별도 모델 품질 단계다.

#### 46.5.1.3 단계 3: repack identity

Marlin repacked qweight를 inverse mapping으로 읽어 같은 logical code를 복원한다. production code에 inverse helper가 없으면 test utility의 Marlin tile permutation과 pack rule로 독립 reference를 만든다. checkpoint code matrix와 element-wise exact equality를 검사한다.

repack은 lossless integer permutation이어야 한다. tolerance가 필요하지 않다. mismatch가 있으면 CUDA GEMM을 볼 이유가 없다. padding region은 별도다. original K/N 안에서는 exact identity, padded K/N에서는 dequant contribution zero를 확인한다.

scale permutation도 inverse permutation으로 original `(group,n)` scale과 exact 또는 dtype-exact 비교한다. zero-point는 deinterleave와 int32 unpack을 거친 code identity를 본다. weight code가 맞아도 scale/ZP tile이 한 MMA fragment만큼 밀리면 output은 틀린다.

#### 46.5.1.4 단계 4: act-order 짝

act-order fixture에서는 sorted `g_idx`와 `sort_indices`가 permutation인지 확인한다. sort indices가 `0..K-1`을 중복·누락 없이 포함해야 한다. sorted group index는 nondecreasing이어야 하고 `sorted_g_idx[i]=g_idx[sort_indices[i]]`여야 한다.

weight K dimension과 activation K columns에 같은 permutation을 적용한 dot product가 original quantized W dot product와 같은지 CPU에서 검산한다. X만 permute하거나 W만 permute한 negative fixture가 실제로 다른 값을 내는지도 확인한다. negative test가 차이를 못 내면 X/code pattern이 permutation bug를 검출하기에 약하다.

row-parallel rank는 local K coordinate와 global `g_idx` slice를 구분한다. scale을 rank마다 반복하는 경로에서는 rank 1의 local column 0이 global K offset의 group을 참조해야 한다. `is_k_full`이 거짓인 path의 group size가 0 sentinel을 쓰는 이유와 native 해석을 연결한다.

#### 46.5.1.5 단계 5: 한 K group의 GEMM

이제 X와 dequantized W의 CPU matmul을 FP32 accumulator로 계산한다. group별 partial `Y_g=X[:,gG:(g+1)G] W_g`도 저장한다. Marlin 또는 native test가 partial을 노출하지 않으면 input에서 한 group만 nonzero로 만들어 어느 group이 첫 divergence인지 찾는다.

output N channel 하나만 선택해 dot product 항을 출력한다. mismatch가 모든 M row에서 같은 N에 있으면 weight/scale column 문제 가능성이 높다. 특정 M row만 틀리면 activation quantization, input stride, per-token scale을 본다. K group 수가 늘 때만 차이가 커지면 reduction order 또는 group transition을 본다.

FP16/BF16 operand와 FP32 accumulate의 reference cast 위치를 kernel에 맞춘다. float W를 full precision으로 곱한 reference와 dequant 후 half cast를 곱한 reference는 다를 수 있다. expected numerical difference와 layout bug를 분리하려면 cast sequence를 명시한다.

#### 46.5.1.6 단계 6: reduction과 epilogue

single-CTA 또는 split 없는 shape에서 맞고 큰 K/N에서만 틀리면 parallel reduction을 본다. workspace counter가 zero-initialized됐는지, output owner가 partial 수를 정확히 기다리는지, atomic path에서 output buffer가 zero인지 확인한다. graph replay 사이에 workspace가 stale하면 반복 횟수에 따라 달라질 수 있다.

FP32 reduce path와 lower-precision reduce path는 rounding이 다르다. small tolerance의 deterministic 차이는 예상 가능하지만 bias를 여러 partial에 중복 적용하거나 global scale을 누락하면 큰 systematic error가 난다. bias-only fixture에서 X=0으로 두면 epilogue 적용 횟수를 쉽게 본다.

padding N path에서는 original N output만 reference와 비교하고 padded columns가 public shape로 새지 않는지 본다. K padding fixture는 padded X가 zero인지와 padded weight dequant contribution이 zero인지 확인한다. asymmetric format에서 qcode 0만 채우면 ZP 때문에 zero weight가 아닐 수 있으므로 scale/ZP와 함께 설계한다.

#### 46.5.1.7 단계 7: format-specific activation scale

W4A16이 맞고 W4A8/FP8 activation path만 틀리면 per-token quantization을 분리한다. 각 X row의 amax, quant scale, quantized code, dequantized X를 저장한다. zero row, maximum representable 값 근처, outlier가 하나인 row를 넣는다.

activation scale과 weight scale을 accumulator/epilogue에서 몇 번 곱하는지 확인한다. input global scale이 이미 a_scale에 fold됐는데 epilogue에서 다시 곱하면 제곱 오차가 난다. scale shape가 `(M,1)`인지 contiguous vector인지 native stride와 비교한다.

FP8 cast의 saturation과 special value 처리는 PyTorch/reference dtype 규칙을 따른다. NVFP4는 block maximum과 FP8 block scale, global scale을 단계별로 비교한다. INT8 reference를 FP8/NVFP4에 재사용하지 않는다.

#### 46.5.1.8 단계 8: MoE permutation

MoE fixture는 `M=5,T=2,E=4`에서 각 token row를 고유한 one-hot 또는 coordinate pattern으로 만든다. `topk_ids`를 직접 지정해 counts `[4,3,2,1]`을 만든다. alignment output의 valid sorted slots를 순회해 각 logical pair가 정확히 한 번 나타나는지 multiset으로 검사한다.

각 sorted block의 expert ID가 그 slot의 topk expert와 같은지 본다. padded slot은 sentinel이고 valid count에 포함되지 않아야 한다. expert map이 있으면 global-to-local mapping 뒤 expected local expert와 비교한다. 이 단계에서 틀리면 expert GEMM 결과를 볼 필요가 없다.

expert weight를 쉽게 식별하도록 expert e의 GEMM이 input에 `(e+1)`을 곱하는 diagonal matrix처럼 만든다. sorted token이 잘못된 expert weight를 쓰면 output factor로 곧 드러난다. W1과 W2를 각각 identity-like로 만들어 first/second GEMM을 분리한다.

#### 46.5.1.9 단계 9: activation과 combine

GEMM1 output을 reference와 비교한 뒤 activation output, GEMM2 pair output, final token output 순으로 비교한다. gated activation이면 gate/up halves의 layout을 표시한다. clamp와 model-specific activation parameter가 있으면 reference에 포함한다.

top-k weights는 서로 다른 값, 예를 들어 token별 `[0.25,0.75]`를 사용한다. pair outputs가 모두 같으면 weight가 중복 또는 누락돼도 숨을 수 있으므로 expert별 다른 output을 만든다. final expected는 token마다 두 expert contribution의 weighted sum이다.

`apply_router_weight_on_input` 두 mode에서 weight가 정확히 한 번 적용되는지 본다. 앞에서 적용한 mode는 final combine이 sum만 해야 하고 뒤에서 적용한 mode는 pair output 또는 combine에서 곱해야 한다. 결과가 weight 제곱에 비례하면 중복 적용, unweighted sum이면 누락이다.

**ABI-46: metadata는 모두 맞았지만 N축 scale 순서가 달랐다**

작은 weight `W[K=16,N=8]`을 4-bit symmetric quantization한다고 하자. group size는 K축 8이므로 scale tensor의 논리 shape는 `[K/group=2,N=8]`이다. checkpoint metadata에는 bits 4, group size 8, symmetric true, shape 16×8이 정확히 기록됐다. loader validation도 모두 통과했다.

문제는 repack cache가 이전 kernel ABI의 scale permutation을 사용한 데 있었다. 논리 scale `S[g,n]`은 맞았지만 새 kernel은 N tile 안에서 columns를 `[0,2,4,6,1,3,5,7]` 순서로 기대했고 cache artifact는 `[0,1,2,3,4,5,6,7]` 순서였다. shape와 dtype, byte count가 같아 일반 metadata 검사는 차이를 잡지 못했다.

입력 activation은 `X[M=2,K=16]`이고 첫 row는 K index를 알아보기 쉽게 `[1,2,...,16]`으로 둔다. column 1의 quantized weights가 모두 1, column 2가 모두 2라고 하자. scale `S[0,1]=0.1`, `S[0,2]=0.5`다. permutation이 뒤바뀌면 column 1 accumulator에 다른 column scale이 적용돼 output이 체계적으로 커진다. NaN이나 crash가 아니라 특정 N columns만 틀린다.

fallback dense dequant 경로는 logical `S[g,n]`를 직접 읽으므로 정답이다. Marlin path만 틀려 “kernel 수치 오차”처럼 보였다. 최초 불일치는 MMA가 아니라 loader가 logical scale tensor를 kernel ABI order로 바꾸는 permutation generation이었다. format metadata truth와 kernel-consumer truth가 갈라진 순간이다.

**nibble 좌표를 packed word까지 손으로 내린다**

교육용 row-major pack은 uint32 word 하나에 4-bit values 여덟 개를 넣는다고 하자. 논리 `(k,n)`을 flat index `i=k×N+n`으로 만들고 `word=i/8`, `nibble=i%8`, bit shift `4×nibble`로 둔다. `(k=3,n=5)`면 `i=29`, word 3, nibble 5, shift 20이다.

quantized value가 signed -3이고 저장 code가 two's-complement lower nibble `0xD`라면 word mask `0xF<<20` 위치에 들어간다. unpack는 `(word>>20)&0xF` 뒤 signed 복원을 수행한다. 이 logical pack 식은 checkpoint 설명용이다. 실제 Marlin repack은 MMA load와 tile traversal에 맞춰 words를 permutation하므로 kernel physical index는 다를 수 있다.

repack을 작은 permutation `P=[0,2,4,6,1,3,5,7]`로 설명하자. tile 내 logical column n의 physical column slot은 `P^{-1}(n)` 또는 구현 contract에 따라 `P(n)`다. 방향을 틀리면 permutation 자체는 bijection이고 byte count도 보존돼 검사가 통과하지만 columns가 바뀐다. forward map과 inverse map을 둘 다 sample로 검산한다.

예를 들어 physical slots가 logical columns `[0,2,4,6,1,3,5,7]`을 담으면 logical n=5는 physical slot 6이다. scale도 같은 logical-to-physical mapping을 써야 한다. qweight만 repack하고 scales는 identity order로 두면 packed values와 dequant multiplier가 다른 columns를 가리킨다.

group index도 함께 계산한다. k=3이면 g=0, k=11이면 g=1이다. `(11,5)` weight가 physical column slot 6으로 갔다면 scale은 physical `(g=1,slot=6)`에 있어야 한다. K permutation이나 act-order가 있으면 original k와 packed k, scale group k가 어느 coordinate를 쓰는지 별도 열로 둔다.

**tile과 lane이 packed 좌표를 소비하는 장면**

교육용 kernel tile을 `Ktile=16`, `Ntile=8`로 두면 fixture 전체가 CTA 하나에 들어간다. threads 128을 네 warps로 나누고 warp마다 N columns 두 개를 담당한다고 가정한다. warp 0은 physical slots 0,1, warp 1은 2,3, warp 2는 4,5, warp 3은 6,7을 읽는다.

logical column 5는 physical slot 6이므로 warp 3이 처리한다. loader가 logical n=5 output을 warp 2가 쓴다고 가정하면 epilogue inverse permutation이 빠진 것이다. kernel은 physical order로 accumulate한 뒤 output columns를 logical order로 store하거나 launcher가 이미 expected output mapping을 준비해야 한다. 어느 층이 inverse map을 소유하는지 source에서 찾는다.

lane은 K packed words와 activation fragments를 협력해 load한다. exact Marlin mapping은 source specialization에 따르지만 독자는 적어도 sample `(k=11,n=5)`가 어느 packed word, physical slot, group scale, accumulator column, final output column으로 이동하는지 한 줄로 연결해야 한다. “repack한다”는 동사만으로 ABI가 맞았다고 결론내리지 않는다.

MMA fragment layout은 일반 row-major tensor 눈금과 다를 수 있다. shared-memory staging과 `ldmatrix`/MMA fragment mapping이 개입하면 lane이 보유한 fragment element를 단순 `(threadIdx.x,n)`로 읽을 수 없다. 이 장의 sample은 host pack/repack contract를 검산하는 기준이고 실제 lane mapping은 pinned kernel indexing과 tile constants로 확인한다.

**workspace와 capability gate가 ABI 선택에 개입한다**

Marlin workspace는 kernel 내부 parallel split이나 reduction, locks/counters에 쓰일 수 있다. shape로 계산한 required entries와 실제 allocated dtype/bytes를 구분한다. stale workspace를 zero/init하지 않거나 이전 shape의 lock state를 재사용하면 scale permutation이 맞아도 hang 또는 partial output이 생길 수 있다.

M=2처럼 작은 batch에서는 N/K tiles를 여러 CTAs가 분담하거나 reduction가 필요할 수 있다. workspace owner는 launch 전 initialization, kernel generation, completion 뒤 reuse를 관리한다. graph capture가 workspace pointer를 고정하면 shape cache key와 allocation generation도 ABI 일부다.

SM capability gate는 “GPU가 4-bit를 지원한다”는 bool이 아니다. kernel이 요구하는 instructions, shared-memory capacity, warp behavior, compiled code object가 target capability에 맞아야 한다. gate 실패 때 generic quant GEMM이나 dequant+dense fallback로 가는 것은 정상 선택일 수 있다.

문제는 fallback 이유를 숨긴 채 `quantization=marlin` 설정만 dashboard에 노출하는 경우다. effective backend, specialization, repack artifact ABI version, workspace plan, SM capability를 함께 기록한다. 요청 option과 실제 kernel은 다를 수 있다.

**vLLM·SGLang·Marlin source를 소비 순서로 걷는다**

vLLM source walk는 model loader의 quant config recognition에서 시작해 parameter object가 qweight, scales, zeros, g_idx를 어떤 shape로 받는지 본다. 이어 Marlin 호환성 판정, repack/permutation utility, linear method apply, native operator argument 순서를 연결한다. 각 경계에서 logical shape와 physical ABI shape를 표에 남긴다.

SGLang도 loader가 동일 checkpoint format을 읽는다는 사실만으로 같은 physical artifact를 쓴다고 가정하지 않는다. quant method 선택, Marlin preparation, fused MoE backend selection, workspace/intermediate cache allocation을 별도로 따라간다. dense linear와 expert weights가 다른 repack path를 가질 수 있다.

Marlin native source에서는 template parameters와 launcher arguments를 먼저 읽는다. bits, group size, K/N divisibility, thread/tile config, workspace pointer, scale/zero layout, act-order flags가 kernel specialization과 일치하는지 확인한다. pack utility가 만드는 order와 kernel loader가 inverse로 소비하는 order를 sample coordinate로 맞춘다.

source pin은 producer와 consumer가 쌍이어야 한다. `marlin_repack` 함수만 링크하면 부족하다. repacked word index를 계산하는 native load와 scale index를 계산하는 load를 함께 pin한다. ABI version이 명시돼 있지 않다면 commit hash와 cache key에 source version을 포함한다.

**ABI-46을 반증하고 rollback한다**

첫 test는 basis weight다. 논리 `(k=11,n=5)` 하나만 nonzero code로 두고 activation도 k=11만 1로 둔다. expected output은 column 5 하나만 nonzero다. 다른 column이 켜지면 qweight permutation 또는 epilogue inverse map이 틀렸다. 값 크기만 틀리면 scale group/column mapping을 본다.

둘째 test는 column별 distinct scales다. scales를 `[0.1,0.2,...,0.8]`처럼 두고 qweight codes를 모두 1로 둔다. output column ratios가 logical scale order를 따라야 한다. permutation error는 안정적인 column signature로 나타난다. random tensor tolerance test보다 first divergence가 선명하다.

셋째 test는 group boundary k=7/8을 사용한다. 두 K positions만 활성화하고 group 0/1 scales를 크게 다르게 둔다. act-order/K permutation 뒤 scale group이 original k인지 permuted k인지 검증한다. off-by-one group index도 잡는다.

넷째 test는 M=1/2/17, N/K tile 경계 전후를 돈다. shape padding과 unpadding, workspace split policy가 달라지는 지점을 표본화한다. dense와 MoE expert path 모두 같은 artifact를 소비하는지 확인한다. expert별 weights에는 expert index 축을 basis에 추가한다.

rollback는 repack cache를 무효화하고 known-good ABI producer로 artifact를 다시 만드는 것부터 시작한다. metadata JSON만 되돌리고 physical packed cache를 남기면 오류가 유지된다. artifact key에 model weight identity, bits/group, act-order, scale layout, kernel ABI/source version, SM target을 넣는다.

긴급 fallback는 reference quant kernel 또는 dequant+dense GEMM을 사용한다. output equality를 회복한 뒤 latency/VRAM 영향을 별도 기록한다. unsupported gate를 강제로 우회해 Marlin을 실행하지 않는다. correctness가 확인되지 않은 빠른 path보다 설명 가능한 fallback가 우선이다.

90분 soak는 multiple layers/experts, M 경계, graph eager, cache cold/warm을 섞는다. basis sentinel mismatch, NaN, workspace lock 잔류, backend fallback 변화가 0인지 본다. performance는 tokens/s와 kernel duration뿐 아니라 repack one-time cost와 cache hit를 분리한다.

terminal의 최초 불일치는 “4-bit metadata가 틀렸다”가 아니다. logical scale `[2,8]`을 ABI-17 identity order로 cache했는데 kernel ABI-18이 even/odd N permutation을 소비한 순간이다. fix 뒤 basis `(11,5)`가 output column 5와 group 1 scale을 정확히 선택하고 fallback과 equality가 닫혀야 한다.

## 46.6 MoE fixture에서 expert ABI와 combine을 검산한다

### 46.6.1 네 token을 expert tile로 다시 배열한다

tokens T0–T3, hidden K=16, experts E0–E2, top-k=2 fixture를 만든다. router 선택은 T0→E0/E2, T1→E1/E2, T2→E0/E1, T3→E2/E1이다. routing pairs는 8개다. expert별 counts는 E0=2, E1=3, E2=3이다.

kernel이 expert batch를 4 rows alignment로 요구하면 padded rows는 E0=4, E1=4, E2=4, 총 12다. sorted token IDs는 예를 들어 `[T0,T2,pad,pad | T1,T2,T3,pad | T0,T1,T3,pad]`다. 이 12-row M축은 original batch 4도 routing pairs 8도 아니다. padding predicate와 expert offsets `[0,4,8,12]`가 GEMM launcher ABI다.

각 sorted row에는 original token ID와 top-k slot 또는 routing weight가 연결된다. GEMM1 output 뒤 activation/GEMM2를 거쳐 combine할 때 `(token,route-slot)`로 되돌린다. sorted row index를 token ID로 오인하면 memory bounds 안에서 다른 token output을 합친다. shape는 정상이고 wrong output만 생긴다.

expert weight는 `[E,N,K]` 또는 backend physical order를 가진다. E1 tile을 처리하는 CTA가 weight base `expert_stride×1`을 더해야 한다. repack artifact가 expert axis를 N/K permutation보다 앞에 두는지 뒤에 두는지 확인한다. logical expert index가 physical shard/local expert index로 변환되는 EP 환경도 별도다.

### 46.6.2 expert별 scale 좌표를 basis fixture로 밝힌다

E0/E1/E2 scales에 서로 다른 signature를 준다. E0는 0.1, E1은 1.0, E2는 10.0 배율을 쓰고 column별로 작은 차이를 둔다. qweight code를 모두 1로 두면 output magnitude만으로 선택된 expert와 column permutation을 식별할 수 있다.

T0은 E0/E2로 가므로 두 route outputs가 약 100배 차이를 보여야 한다. 둘이 비슷하면 expert scale base가 고정됐거나 cache artifact가 expert dimension을 누락했을 수 있다. E1만 특정 columns가 틀리면 expert base보다 N permutation을 본다. failure signature를 coordinate 축에 대응시킨다.

group boundary fixture도 expert별로 반복한다. E2의 `(k=7,n=5)`와 `(k=8,n=5)`만 nonzero로 두고 group scales를 10과 20으로 둔다. sorted token T3/E2 row가 두 contributions를 올바르게 합치는지 본다. expert offset, K group, N permutation 세 축을 동시에 검증한다.

random output cosine similarity 하나로 이 오류를 찾기 어려운 이유는 routing weights와 experts 합이 permutation signature를 희석하기 때문이다. top-k=1 basis, one-hot activation, distinct scale signatures로 분해한 뒤 top-k=2 combine을 검증한다. 작은 deterministic fixture가 large benchmark보다 first divergence에 가깝다.

### 46.6.3 intermediate cache와 workspace를 byte offset으로 계산한다

GEMM1 output dimension N1=32, padded routing rows 12, FP16이라면 intermediate activation은 `12×32×2=768 bytes`다. activation 뒤 gated product가 Nmid=16이면 384 bytes다. GEMM2 output hidden 16이면 또 384 bytes가 필요하다. backend가 buffers를 alias/reuse하는 시점은 kernel completion과 graph capture contract에 달렸다.

expert offsets `[0,4,8,12]`에 N1 stride 32를 적용하면 E2 첫 row scalar offset은 `8×32=256`, byte offset 512다. T0/E2가 sorted row 8이면 그 output은 여기서 시작한다. combine kernel이 original T0 output에 routing weight를 곱해 더한다. row 8을 routing pair index 0으로 읽으면 다른 address다.

workspace에 expert prefix sums, sorted IDs, token counts, locks/reduction partials가 함께 있으면 각 subregion offset과 alignment를 기록한다. total bytes만 맞아도 offset version이 달라지면 kernel이 counts를 IDs로 읽을 수 있다. host plan struct ABI와 device kernel struct view를 쌍으로 pin한다.

graph capture는 maximum padded routing rows의 workspace address를 고정할 수 있다. runtime active pairs가 줄어도 unused rows와 counts를 current generation sentinel로 초기화해야 한다. previous expert IDs가 남으면 stale expert computation이 combine에 들어갈 수 있다. capacity와 active pair count를 분리한다.

vLLM fused MoE 선택에서 Marlin consumer까지 걷는다. 다음 source walk는 별도 단계라기보다 같은 MoE fixture의 producer와 consumer를 잇는 과정이다.

vLLM option의 MoE backend는 요청 의도다. model quant method, expert weight format, GPU capability, shape, distributed mode가 effective implementation을 고른다. router/top-k 결과가 alignment/sort utility로 들어가고, intermediate cache가 준비되며, fused expert kernel 또는 fallback가 호출되는 경로를 잇는다.

source 표에는 `num_tokens`, `num_topk`, `num_valid_pairs`, `num_padded_rows`, `num_experts`, `local_experts`, `K/N`, bits/group, workspace bytes를 적는다. 같은 이름 `M`이 tokens인지 padded pairs인지 native launcher에서 다시 확인한다. dense Marlin M과 MoE Marlin M은 upstream 의미가 다르다.

expert parallel에서는 global expert E7이 rank local expert 1일 수 있다. weights/repacked scales는 local order이고 routing metadata는 global IDs를 가질 수 있다. all-to-all dispatch가 local IDs로 바꾸는 boundary를 찾는다. global ID를 weight base에 직접 쓰면 bounds 또는 wrong expert가 된다.

fallback 이유도 보존한다. unsupported group size, act-order, zero-point mode, tile divisibility, SM capability, workspace 부족이 서로 다른 fallback를 만들 수 있다. “Marlin miss” counter 하나로 합치지 않는다. effective kernel symbol과 reason code를 trace에 붙인다.

SGLang backend와 cache artifact identity도 같은 원장에서 검토한다.

SGLang server option이 MoE A2A backend를 고르면 routing/dispatch topology가 바뀔 수 있다. 그러나 quant weight ABI는 local expert GEMM consumer와 맞아야 한다. communication backend 선택과 Marlin format 선택을 같은 flag로 보지 않는다. dispatch 뒤 local sorted rows가 어느 kernel로 들어가는지 따라간다.

model loading 단계에서 repack 결과를 cache하면 key에 expert sharding과 local order가 포함돼야 한다. TP/EP degree를 바꿨는데 같은 artifact를 재사용하면 tensor byte count 일부가 맞아도 expert bases가 달라진다. deployment topology는 physical ABI identity의 일부다.

cache file header에는 logical model weight hash, source format, bits/group, zero mode, act-order, scale permutation version, Marlin source commit, SM target, TP/EP layout을 둔다. 로드 시 모두 비교한다. filename에 `marlin`이 있다는 사실로 호환성을 가정하지 않는다.

SGLang graph runner가 fused MoE workspace pointer를 capture한다면 cache eviction/reallocation와 graph exec lifetime을 연결한다. artifact compatibility가 맞아도 stale workspace pointer는 별도 use-after-free다. format correctness, launch ABI, object lifetime를 세 층으로 검사한다.

tile coverage와 combine conservation으로 fixture를 검증한다.

expected routing pair set은 `{(T0,E0),(T0,E2),(T1,E1),(T1,E2),(T2,E0),(T2,E1),(T3,E2),(T3,E1)}`다. sorted valid rows가 이 set과 정확히 같아야 한다. padded rows는 별도 sentinel이고 set에 포함하지 않는다. duplicate/missing pair를 잡는다.

각 expert tile의 row interval은 `[0,2)`, `[4,7)`, `[8,11)`이고 padding intervals는 `[2,4)`, `[7,8)`, `[11,12)`다. kernel predicate가 counts를 사용해 padding stores/combines를 막는지 본다. alignment capacity만 보고 12 rows 모두 valid로 처리하지 않는다.

combine conservation은 token마다 두 route weights 합과 output contributions를 연결한다. routing weights가 정규화됐다면 basis output 합이 reference와 일치해야 한다. sorted row permutation을 바꿔도 combine 결과는 동일해야 한다. order-dependent 차이가 크면 wrong index 또는 unsafe reduction을 의심한다.

tile boundary는 experts counts 3/4/5, K/N divisibility 전후, tokens 0/1을 포함한다. empty expert가 weight tile을 launch하는지, offsets가 다음 expert base와 겹치지 않는지 본다. hot expert 하나가 많은 rows를 가질 때 workspace capacity와 split policy도 검증한다.

마지막으로 MoE-46 incident rollback을 terminal로 닫는다.

사건 변형은 TP2/EP2에서 만든 repack cache를 TP1/EP3 deployment가 재사용한 것이다. bits, group size, logical expert count, tensor byte 합계는 맞았지만 local expert order와 scale bases가 달랐다. E0은 정상이고 E1/E2 outputs만 바뀌어 router instability처럼 보였다.

first divergence는 routing이 아니다. global→local expert mapping 뒤 consumer가 local E1 weight를 요구했지만 artifact offset table은 이전 topology의 local E2 scale base를 가리켰다. top-k와 sorted IDs가 모두 맞아도 kernel ABI coordinate가 틀렸다.

긴급 rollback는 artifact cache를 격리하고 current topology에서 repack를 다시 수행한다. known-good generic fused MoE 또는 dense-per-expert fallback로 basis equality를 확인한다. routing temperature나 capacity factor를 바꾸어 증상을 숨기지 않는다.

회귀 fixture는 topology key를 바꾸면 cache miss/rebuild가 일어나는지, local expert basis가 distinct scale signature를 유지하는지, padded rows가 combine되지 않는지 검사한다. graph cold/warm과 workspace reuse를 교차한다. unsupported capability에서는 명시적 fallback reason을 기대한다.

90분 soak는 expert skew, empty experts, hot expert, abort, graph replay, TP/EP topology별 artifact cold/warm을 섞는다. expert basis sentinel, output reference, workspace bounds, fallback stability, cache key mismatch가 모두 닫혀야 한다.

terminal 문장은 “MoE kernel이 틀렸다”보다 구체적이다. “TP2/EP2 repack artifact의 local expert scale offset을 TP1/EP3 consumer가 재사용해 E1 logical coordinate가 E2 physical base를 읽었다.” fix는 topology-bound ABI key와 basis coverage test로 증명한다.

## 46.7 GPTQ, AWQ, Marlin은 서로 다른 질문에 답한다

### 46.7.1 GPTQ가 결정하는 것

GPTQ는 pretrained weight를 low-bit code로 만드는 post-training quantization 방법이다. approximate second-order information을 사용해 한 weight를 quantize할 때 뒤의 weight를 보정하며 layer output error를 줄이려 한다. 결과 checkpoint에는 qweight, scale, zero-point 또는 관련 metadata, group index가 들어갈 수 있다.

GPTQ의 `act_order`는 중요한 channel을 먼저 처리하는 방식과 연결되며 저장된 `g_idx`가 단조로운 K group 순서가 아닐 수 있다. kernel은 activation column을 sort permutation에 맞게 보거나 각 weight column이 참조할 group을 정확히 찾아야 한다. `g_idx`를 무시하고 단순히 `floor(k/G)` scale을 쓰면 packing이 완벽해도 dequant weight가 틀린다.

GPTQ 논문은 quantized weight를 어떻게 고르는지 설명한다. vLLM의 `gptq_marlin_repack`이 어떤 16×64 tile 순서로 int32 word를 재배치하는지는 현재 source의 별도 계약이다. quantizer accuracy와 kernel ABI를 섞지 않는 것이 중요하다. 같은 GPTQ checkpoint를 generic GPTQ kernel과 Marlin kernel이 서로 다른 packed representation으로 소비할 수 있기 때문이다.

### 46.7.2 AWQ가 결정하는 것

AWQ는 activation statistics를 이용해 salient weight channel을 찾고 equivalent scaling으로 중요한 channel의 quantization error를 줄인다. salient weight만 별도 high precision으로 저장하는 단순 mixed-precision scheme과 같지 않다. activation-aware scale transformation 뒤에 uniform low-bit weight가 만들어질 수 있다.

checkpoint가 AWQ INT4라고 해서 Marlin이 곧장 읽을 수 있다는 뜻은 아니다. AWQ qweight의 packing과 zero-point convention을 Marlin이 dequant하는 순서로 바꿔야 한다. vLLM의 `awq_to_marlin_zero_points`와 repack 경계가 존재하는 이유다. zero-point는 scale과 같은 permutation을 일부 공유하면서도 MMA마다 적용되는 interleave와 int32 packing이 필요하다.

AWQ와 GPTQ 둘 다 W4A16으로 실행될 수 있지만 offline weight 생성의 의미, checkpoint field, group index와 zero-point 처리에서 다르다. 실행 뒤 값이 틀릴 때 “Marlin 문제”라고 시작하지 않고 checkpoint를 reference float로 복원하는 단계부터 비교해야 한다.

### 46.7.3 Marlin이 결정하는 것

Marlin은 packed weight-only 또는 mixed input을 효율적으로 소비하는 kernel family와 layout이다. 어떤 logical quantized value를 선택했는지가 아니라, 그 value를 GPU tile과 lane이 어떻게 읽고 dequant하며 MMA에 공급할지를 결정한다. weight와 scale을 소비 순서에 맞게 미리 permute하면 hot loop의 gather와 address overhead를 줄일 수 있다.

Marlin은 shape 제약을 갖는다. 현재 vLLM의 [`marlin_utils.py` 153–230행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L153-L230)은 partitioned N과 K, group size를 검증하고 `(N%64,K%128)` 또는 `(N%128,K%64)` thread-tile family에 맞도록 padding 후보를 만든다. tensor parallelism 뒤의 shard shape가 이 제약을 만족하는지가 중요하다.

padding된 quantized region은 output에 기여하지 않아야 한다. integer format에서는 padded scale/zero-point와 code가 함께 0 contribution을 만들어야 하고 FP4/FP8 format은 zero code의 dequant 의미를 확인해야 한다. output N을 padding했으면 GEMM 뒤 원래 N으로 unpad한다. K padding은 activation에도 같은 zero columns를 붙여 dot product를 보존해야 한다.

### 46.7.4 정확성의 세 층

첫 층은 quantization error다. 올바른 GPTQ/AWQ weight도 BF16 원본과 완전히 같지 않다. calibration과 group size, bit width가 허용 가능한 모델 품질을 결정한다. 둘째 층은 representation correctness다. 같은 quantized logical weight가 repack 전후에 정확히 같아야 한다. 셋째 층은 kernel numerical correctness다. dequantized operand의 MMA와 reduction/epilogue가 reference tolerance 안에 있어야 한다.

세 층을 섞으면 deterministic packing bug를 “4비트라 오차가 크다”고 넘기거나 정상 quantization error를 kernel bug로 오인한다. 작은 fixture에서 quantized logical reference를 먼저 만든다. dense BF16 원본이 아니라 이 reference와 Marlin output을 비교하면 두 번째와 세 번째 층을 분리할 수 있다.

## 46.8 GPTQ·AWQ·FP8·NVFP4 적용 경계

### 46.8.1 W4A16은 format 전체를 말하지 않는다

W4A16이라는 표기는 weight가 4-bit, activation이 16-bit라는 계산 경계만 요약한다. signed인지 unsigned/bias encoding인지, group size가 얼마인지, symmetric인지 asymmetric인지, scale dtype이 무엇인지, act-order가 있는지, packed dimension과 tile layout이 무엇인지는 말하지 않는다. 같은 W4A16 label의 두 checkpoint가 binary-compatible하지 않을 수 있다.

model config의 quantization method는 loader class를 선택한다. loader가 checkpoint field를 parameter object에 매핑하고 `process_weights_after_loading`이 repack과 permutation을 수행한다. linear method의 `apply`가 kernel wrapper를 고르고 native op가 scalar type과 shape specialization을 dispatch한다. CLI option 하나에서 CUDA kernel까지 이 네 경계를 따라야 한다.

fallback도 기록한다. shape 또는 SM이 Marlin을 지원하지 않으면 generic GPTQ/AWQ kernel, Triton, dense dequant path로 갈 수 있다. “model이 GPTQ Marlin config다”와 “이번 layer/shape가 Marlin kernel을 실행한다”는 다른 주장이다. startup log, effective method object, custom-op dispatch를 함께 확인한다.

### 46.8.2 FP8은 값 표현과 scale mode를 붙여 읽는다

FP8은 8-bit floating encoding이다. E4M3과 E5M2처럼 exponent/mantissa 배치가 다르고 framework가 사용하는 PyTorch dtype과 hardware instruction 지원이 달라질 수 있다. weight와 activation이 모두 FP8일 수도 있고 weight만 FP8이며 activation을 online quantize할 수도 있다. accumulator와 output은 더 높은 precision일 수 있다.

scale은 per-tensor, per-channel, per-token, blockwise일 수 있다. `X_q≈X/s_x`, `W_q≈W/s_w` 뒤 accumulator에 `s_x s_w`를 적용하는 위치와 broadcasting shape가 정확성 계약이다. scale이 scalar인지 row vector인지 K/N block matrix인지 모른 채 “FP8 scale”이라고 부르면 stride bug를 찾지 못한다.

current vLLM의 Marlin mixed path는 activation dtype이 INT8 또는 `float8_e4m3fn`일 때 `marlin_quant_input`을 호출하고 per-token scale을 만든다. 모든 FP8 checkpoint가 이 path를 쓰는 것은 아니다. scaled-mm, CUTLASS, DeepGEMM 같은 다른 backend가 shape와 SM에 따라 선택될 수 있다. format support와 Marlin support를 분리한다.

FP8은 code 자체가 exponent를 가지므로 INT4처럼 `q-z`라는 정수 affine 식으로만 설명하지 않는다. finite range, subnormal, saturation, NaN encoding과 scale 적용이 있다. reference fixture는 framework가 정의한 dtype cast와 scale을 사용하고 임의 정수 quantizer로 대체하지 않는다.

### 46.8.3 NVFP4는 INT4보다 scale hierarchy가 깊다

NVFP4는 4-bit floating value와 FP8 block scale, global scale의 결합으로 이해해야 한다. 두 4-bit values가 한 byte에 pack될 수 있지만 payload 계산에는 block scale이 반드시 포함된다. activation도 W4A4 path에서 blockwise FP4로 quantize하면 per-token 또는 block scale storage와 quantization kernel이 추가된다.

예를 들어 weight block size를 16이라 가정하면 `K×N` values에 대략 `KN/2` data bytes와 `KN/16`개의 scale entries가 필요하다. scale entry가 한 byte라면 raw overhead는 value당 `1/16` byte다. global scale과 interleave/padding이 더해진다. 이것은 설명용 식이며 실제 NVFP4 block axes와 pack layout은 선택 backend source를 따른다.

NVFP4 weight scale은 단순 row-major `(K/16,N)`으로 저장되지 않을 수 있다. Tensor Core 또는 grouped GEMM이 기대하는 tile에 맞게 interleave/swizzle한다. FlashInfer의 block-scale interleave, vLLM loader의 preprocess, TRTLLM kernel의 expectation이 모두 같다는 보장은 없다. “NVFP4”라는 dtype 이름만으로 tensor를 backend 사이에 전달하면 layout mismatch가 날 수 있다.

Blackwell 계열에서도 compute capability 10.0과 12.0을 구분한다. backend가 SM100용인지 SM120용인지, compiled arch에 해당 cubin/PTX가 있는지, toolkit과 CUTLASS/CuTeDSL version이 필요한 instruction/type을 지원하는지 확인한다. 최신 toolkit 설치만으로 현재 wheel에 specialization이 생기지 않는다.

### 46.8.4 선택 표를 실행 계약으로 바꾼다

| 질문 | GPTQ/AWQ W4A16 | FP8 | NVFP4 |
|---|---|---|---|
| weight code | packed integer 4-bit 중심 | FP8 encoding | packed FP4 encoding |
| activation | BF16/FP16, 일부 mixed INT8/FP8 | FP8 또는 higher precision | W4A4이면 FP4 block quant, W4A16 path도 가능 |
| scale | group/channel FP16/BF16 등 | tensor/channel/token/block | FP8 block scale + global scale |
| zero-point | symmetric이면 없음, AWQ/asym이면 존재 | affine ZP가 핵심이 아님 | INT4 ZP로 설명하지 않음 |
| ordering metadata | GPTQ `g_idx`, repack permutation | backend별 layout | block-scale interleave와 backend pack |
| 대표 실패 | scale/ZP/group/act-order mismatch | scale broadcast/dtype/saturation | block axis/global scale/SM backend mismatch |

이 표는 backend 선택기가 아니다. 실제 method는 model config, layer type, per-rank shape, device capability, installed optional package, build arch, user override를 평가한다. 한 layer가 Marlin을 쓰고 다른 layer가 dense fallback을 쓸 수도 있다. MoE expert와 lm_head가 서로 다른 backend를 쓸 수도 있다.

“지원된다”는 말도 네 단계로 나눈다. loader가 format을 인식한다. conversion/repack이 해당 tensor shape를 처리한다. binary에 kernel specialization이 있다. dispatcher가 현재 shape에서 그 kernel을 선택한다. 마지막으로 output이 reference tolerance를 만족한다. 앞 단계 통과가 뒤 단계를 보장하지 않는다.

## 46.9 네 코드베이스를 같은 kernel처럼 읽지 않는다

### 46.9.1 vLLM: format loader와 modular MoE

vLLM current pin은 quantization config, layer method, Marlin utilities, custom ops, stable-ABI native source를 연결한다. source walk의 시작은 model config의 method 선택이고, weight load 뒤 repack과 scale/ZP permutation을 거쳐 `apply_gptq_marlin_linear` 또는 Marlin MoE expert runner로 들어간다.

linear와 MoE는 weight tensor axis가 다르다. dense packed weight는 K/N이고 MoE weight는 expert axis E가 앞에 붙는다. `marlin_moe_permute_scales`는 expert마다 scale permutation을 적용한다. expert weight repack peak memory를 줄이려 chunk할 수 있어도 final layout invariant는 같아야 한다.

vLLM의 current code에는 Marlin 외에도 CUTLASS, DeepGEMM, FlashInfer, Triton, TRTLLM family가 있다. `auto` selection 결과를 source로 확인한다. optional import wrapper가 있다는 것과 해당 backend가 device에서 선택됐다는 것을 분리한다.

### 46.9.2 SGLang: JIT specialization과 override

SGLang current pin의 [`moe_wna16_marlin.py` 18–130행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/ops/moe/moe_wna16_marlin.py#L18-L130)은 dtype, expert-parallel 여부, bias 존재를 JIT module key로 만들고 CUDA wrapper를 load한다. Python wrapper는 packed weight, scale/ZP, g_idx/permutation, workspace, sorted token/expert IDs, top-k weight를 받는다.

wrapper는 act-order가 있는지 non-empty tensors로 판단하고 scale shape에서 num groups와 group size를 복원한다. FP32 reduce와 non-atomic path에서는 SM 수, sorted token storage와 block size를 사용해 temporary 크기를 제한한다. vLLM과 이름이 유사해도 allocation 코드와 JIT ABI를 line-by-line 비교해야 한다.

[`overrides.py` 1270–1305행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/arg_groups/overrides.py#L1270-L1305)은 특정 NVFP4/MXFP4 MoE 조건에서 backend를 Marlin으로 조정한다. 다른 SM에서는 TRTLLM-gen/FlashInfer path가 유리하거나 필수일 수 있다. SGLang 정책을 vLLM의 자동 선택 규칙으로 옮기지 않는다.

### 46.9.3 llama.cpp: GGUF block quant와 MMQ

llama.cpp current pin의 GGUF quantized tensor는 Q4_0, Q4_1, Q4_K, Q5, Q8, IQ, MXFP4처럼 block struct가 scale/min/high bits를 서로 다른 방식으로 보유한다. Marlin GPTQ qweight와 binary-compatible하지 않다. 이름에 Q4가 있다는 사실은 value당 평균 bits만 비슷할 뿐이다.

[`dequantize.cuh` 46–123행](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/dequantize.cuh#L46-L123)은 Q4_0/Q4_1/Q5/Q8 block에서 values를 복원하는 device helper를 보여 준다. Q4_0의 block scale과 symmetric nibble bias, Q4_1의 scale과 minimum은 서로 다른 식이다.

[`mmq.cuh` 1594–1597행](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/mmq.cuh#L1594-L1597)은 `ggml_cuda_mul_mat_q`와 MMQ 선택 함수의 공개 선언 경계다. 실제 launch와 template specialization을 읽을 때는 이 선언에서 구현 파일로 더 따라가야 한다. llama.cpp CUDA MMQ를 Marlin의 fallback이나 동일 kernel이라고 부르지 않는다. 비교할 것은 packed bytes를 dense HBM intermediate 없이 tile에서 dequant/MMA한다는 설계 문제이지 ABI가 아니다.

MoE graph에서도 llama.cpp는 selected expert IDs를 `GGML_OP_MUL_MAT_ID` 같은 graph operation과 CUDA/backend implementation으로 표현할 수 있다. vLLM처럼 Python modular MoE runner와 top-k cache를 가진다고 가정하지 않는다. graph scheduler, tensor IDs axis, backend op support를 따라간다.

### 46.9.4 FlashInfer: NVFP4와 Blackwell grouped GEMM

FlashInfer current pin의 [`fp4_quantization.py` 1701–1775행](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/quantization/fp4_quantization.py#L1701-L1775)은 batched NVFP4 quantization API를 제공한다. data output과 scale factor layout, global scale, padding 조건을 함께 읽는다. quantizer output을 임의 FP4 GEMM에 넣을 수 있다고 가정하지 않는다.

[`fused_moe.py` 778–967행](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/fused_moe/cute_dsl/fused_moe.py#L778-L967)은 CuTeDSL NVFP4 fused MoE implementation과 public API를 연결한다. input/weight scale, top-k, expert 수, workspace와 output contract가 Marlin WNA16과 다르다.

[`grouped_gemm_masked_blackwell.py` 3508–3675행](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py#L3508-L3675)은 masked-M grouped GEMM의 shape와 scale/layout 검증, Blackwell path를 드러낸다. expert별 M을 padding한 contiguous representation과 masked grouped representation은 workspace와 waste 모델이 다르다.

vLLM이나 SGLang이 FlashInfer package를 import할 수 있다는 사실만으로 이 kernel이 선택된 것은 아니다. installed version, device SM, dtype와 scale layout, backend option, compile availability가 모두 맞아야 한다. 로그의 backend name과 dispatch call을 고정한다.

### 46.9.5 공통점과 비호환성을 함께 기록한다

네 구현의 공통 문제는 packed weight byte를 줄이고 dequantized full matrix materialization을 피하며 tile consumer에 맞게 배치하는 것이다. 그러나 block format, scale granularity, zero convention, tile size, warp layout, workspace, supported SM, MoE routing representation이 다르다.

benchmark 비교를 하려면 “4-bit” label만 맞추지 않는다. logical quantized weight가 같은지, activation dtype, group/block size, scale dtype, symmetric/asymmetric, M/K/N, expert routing distribution, top-k, padding, output tolerance를 맞춘다. 서로 다른 algorithm accuracy와 kernel efficiency를 한 숫자로 섞지 않는다.

## 46.10 현장에서 보는 자원과 선택 실패

현장 감사는 checkpoint metadata 표에서 시작하지만 거기서 끝나지 않는다. 첫 열은 logical weight identity `(layer, expert, k, n)`이고 둘째는 source-format packed coordinate, 셋째는 Marlin repacked coordinate, 넷째는 scale/zero/g_idx coordinate, 다섯째는 native tile/lane consumer, 여섯째는 output logical coordinate다. sample 하나가 여섯 열을 왕복해야 한다.

dense sample은 layer L7, `(k=11,n=5)`, code 0xD, group 1이다. source row-major pack에서는 flat 93, word 11, nibble 5일 수 있다. Marlin physical slot은 fixture permutation에 따라 N slot 6이다. scale은 `(group=1,physical_slot=6)`이고 accumulator는 physical column 6, epilogue는 logical output column 5로 돌린다.

이 표에서 qweight repack만 맞고 scale slot이 5면 ABI-46이 재현된다. scale과 qweight가 모두 slot 6인데 output store가 column 6이면 inverse epilogue 문제다. output column은 맞지만 magnitude가 group 0이면 K permutation/g_idx 문제다. failure signature가 조사할 coordinate를 좁힌다.

zero-point가 있는 asymmetric format은 zero coordinate도 같은 표에 추가한다. packed zero convention이 `zero`인지 `zero-1`인지, nibble order와 group/N permutation이 weights/scales와 일치하는지 본다. symmetric fixture만 통과했다고 asymmetric artifact를 승인하지 않는다.

act-order에서는 original K index, permuted K index, group lookup index가 갈라진다. activation X는 어떤 order로 gather되는지, qweight K tiles가 어떤 order인지, scale group이 g_idx를 통해 원래 group을 찾는지 source로 확인한다. permutation이 bijection이어도 scale association가 틀릴 수 있다.

padding도 ABI 열이다. logical K=15를 kernel Ktile 16으로 padding하면 activation, qweight, g_idx, scale group의 padded entry가 안전한 값을 가져야 한다. output N padding은 epilogue store predicate와 unpadding를 요구한다. padded nibble에 stale data가 있어도 multiply contribution가 0이 되는 contract를 검증한다.

native call arguments를 positional 순서로만 읽지 않는다. qweight pointer, scales, zeros, g_idx, permutation, workspace, M/N/K, bits/group, flags가 각각 어느 producer artifact와 generation을 가리키는지 이름 붙인다. binding signature와 C++ launcher signature가 업데이트될 때 한 argument가 밀려도 dtype이 같은 pointers는 컴파일을 통과할 수 있다.

workspace sample은 entries와 bytes를 모두 계산한다. locks 16개가 int32면 64 bytes, partial accumulators 4×Ntile32×FP32면 512 bytes다. alignment를 128 bytes로 맞추면 partial region offset은 128, total 최소 640 bytes다. host가 576 bytes만 할당하면 small shape에서는 우연히 넘어가고 split이 커질 때 corruption가 난다.

workspace initialization generation도 남긴다. lock counter는 launch마다 0으로 초기화해야 하는지 kernel protocol을 확인한다. graph replay가 memset node를 capture했는지, eager path가 별도 zero를 하는지 비교한다. pointer가 같다는 이유로 이전 launch의 completion state를 신뢰하지 않는다.

SM capability 감사에서는 compiled artifact target과 runtime device capability를 나란히 둔다. launcher gate가 true여도 필요한 cubin/PTX image가 wheel에 있는지, JIT fallback가 가능한지, shared-memory opt-in이 설정됐는지 본다. capability 숫자 하나만 통과했다고 launch 가능한 것은 아니다.

tile selection은 M/N/K와 SM count, bits/group, act-order, zero mode의 함수일 수 있다. 작은 M은 memory-bound 최적 tile을, 큰 M은 다른 split/parallelism을 고를 수 있다. source가 실제 사용하는 predicate를 기록하고 경험적 추측을 구현 사실로 쓰지 않는다.

effective backend 관측에는 requested quant method, compatible-format conversion, selected linear method, native kernel symbol, fallback reason을 둔다. `quantization=awq` 모델이 AWQ-Marlin 변환 뒤 Marlin kernel을 쓸 수 있고, 요청 Marlin이 shape gate 때문에 generic path로 갈 수도 있다. format 이름과 execution backend를 분리한다.

MoE 감사표는 logical identity에 `(token,route_slot,global_expert,local_expert)`를 추가한다. T0 route1 E2가 sorted row 8, local expert 2, weight `(k=11,n=5)`를 읽고 combine에서 T0 slot1로 돌아오는지 본다. 각 mapping의 producer와 inverse consumer를 쌍으로 찾는다.

expert-parallel dispatch는 rank owner를 추가한다. global expert 7이 rank 2 local expert 1이면 repack artifact key는 rank/local order를 반영한다. rank 0의 local expert 1과 같은 숫자여도 다른 weight다. artifact cache를 shared filesystem에서 쓸 때 rank/topology identity가 빠지지 않아야 한다.

tensor-parallel은 N 또는 K shard coordinate를 바꾼다. global n=5가 local shard 범위에 없을 수 있고 global K group이 shard boundary를 걸칠 수 있다. quantization group과 shard alignment가 compatibility predicate에 미치는 영향을 본다. shard 뒤 repack인지 global repack 뒤 shard인지에 따라 physical order가 달라진다.

loader source walk는 checkpoint tensor name에서 시작한다. layer/expert shard를 선택하고 logical shape를 검증하며 format-specific unpack/repack 준비를 한다. cache hit면 producer를 건너뛰므로 cache header 검증이 source path의 일부다. cold와 warm path를 각각 따라간다.

vLLM consumer walk는 quant config→parameter loader→Marlin conversion/compatibility→linear/MoE method→native op으로 잇는다. option 문서나 registry entry만 pin하지 않는다. actual tensor mutation과 native argument construction의 고정 줄을 evidence로 둔다.

SGLang consumer walk도 동일 질문을 쓰되 code path를 억지로 같게 만들지 않는다. model loader integration, quant method class, fused MoE backend, distributed dispatch, kernel wrapper를 실제로 잇는다. 공유 upstream library를 사용해도 version과 local patch가 다르면 commit을 별도로 고정한다.

Marlin source walk는 host repack utility의 permutation table과 native load의 inverse/consumer index를 대조한다. scale permutation, zero packing, act-order, thread config selection, workspace formula를 각각 sample coordinate로 검산한다. paper의 high-level layout 설명은 의도를 제공하지만 pinned implementation ABI를 대신하지 않는다.

wrong-output investigation는 reference ladder를 쓴다. FP16 dense reference, logical dequant+dense, generic quant kernel, Marlin dense, Marlin MoE 순으로 비교한다. logical dequant부터 틀리면 checkpoint/quant metadata를 보고 generic은 맞고 Marlin만 틀리면 repack/kernel ABI를 본다. dense Marlin은 맞고 MoE만 틀리면 routing/expert/workspace 축으로 좁힌다.

basis test는 output tolerance를 거의 exact signature로 만든다. one-hot X와 one-nonzero qweight는 accumulator association를 드러낸다. distinct powers-of-two scales는 column/group/expert permutation을 binary signature처럼 보이게 한다. random Gaussian만 쓰면 여러 오류가 상쇄될 수 있다.

tile-boundary matrix는 K/N 15/16/17, group 7/8/9이 아니라 실제 지원 group 경계, M 1/2/16/17, expert rows alignment 전후를 고른다. unsupported 조합에는 명시적 rejection/fallback를 기대한다. padding으로 지원하는지 직접 kernel이 지원하는지 구분한다.

cache invalidation test는 header field 하나씩 바꾼다. source weight hash, bits, group, zero convention, act-order, scale permutation version, Marlin commit, SM target, TP/EP topology를 바꿀 때 hit가 나면 실패다. performance 때문에 validation을 생략하지 않는다.

fallback test는 output equality와 reason stability를 함께 본다. unsupported shape가 upgrade 뒤 새 Marlin specialization으로 이동할 수 있으므로 kernel 이름의 영구 golden보다 decision predicate와 output을 검증한다. unexpected fallback는 성능 regression이고 unsafe forced kernel은 correctness 위험이다.

ABI-46 incident timeline은 cold load, repack, cache publish, warm load, native launch, wrong column discovery로 적는다. cold producer가 어떤 version이었고 warm consumer가 어떤 version인지 기록한다. cache file mtime보다 ABI header와 commit identity가 causal evidence다.

MoE-46 timeline은 topology T1에서 artifact publish, topology T2 배포, cache hit, global→local mapping, E1 wrong scale base, combine output corruption 순서다. router logits가 정상임을 basis로 반증하고 first divergence를 artifact offset lookup로 고정한다.

rollback 1단계는 cache namespace를 새 ABI/topology로 분리하고 stale artifacts를 quarantine한다. 삭제만 하면 forensic evidence가 사라질 수 있으므로 header/hash와 failing fixture를 보존한다. 새 artifact는 cold basis tests 통과 뒤 publish한다.

2단계는 known-good generic backend로 서비스 correctness를 회복한다. fallback의 VRAM, TTFT/ITL, throughput 비용을 capacity plan에 반영한다. traffic를 줄이거나 admission을 조정할 수 있지만 unsupported Marlin gate를 우회하지 않는다.

3단계는 corrected producer와 consumer를 같은 release unit으로 배포한다. rolling upgrade에서 old artifacts/new kernels 또는 new artifacts/old kernels 조합이 생기지 않도록 ABI namespace를 version한다. mixed workers가 shared cache를 쓰면 read compatibility matrix를 둔다.

canary는 layer/expert basis를 sampling하고 dense/MoE reference difference, selected backend, cache header, workspace high-water를 기록한다. raw model weights를 로그에 남기지 않고 deterministic fixture나 hashes를 쓴다. 특정 expert만 traffic가 적어 오류가 숨지 않게 synthetic probes를 사용한다.

90분 soak는 cache cold/warm, M/tile boundaries, graph/eager, experts skew, TP/EP deployment variants를 반복한다. wrong column/group/expert sentinel 0, workspace bounds violation 0, unexpected fallback 0을 요구한다. repack latency와 disk/cache overhead도 steady kernel time과 분리한다.

terminal 승인문은 좌표를 포함한다. “L7 E1 `(k11,n5)` code가 packed word/nibble에서 physical N slot 6, group 1 scale slot 6, warp owner, logical output n5로 왕복했다. TP/EP-bound cache key가 다른 topology artifact를 거부했고 basis·boundary·soak가 통과했다.” 이 문장을 source와 fixture로 재현할 수 있어야 한다.

이 감사표의 가치는 특정 Marlin version 암기에 있지 않다. 다음 release에서 permutation이나 tile이 달라져도 logical coordinate, producer artifact, native ABI, consumer coordinate, output inverse의 다섯 관계를 다시 채울 수 있다. 포맷 metadata가 맞다는 안도감 뒤에 숨은 physical ABI 오류를 구조적으로 찾는다.

### 46.10.1 “지원하지 않는 shape”

원본 model K/N만 보지 않고 TP rank별 input/output partition을 기록한다. group size가 local K를 나누는지, Marlin minimum thread K/N과 두 tile family에 맞는지, padding이 허용되는 format인지 확인한다. loader validation 메시지의 actual partition 값을 hand calculation과 맞춘다.

TP size를 줄이라는 메시지는 단순 권고가 아니다. N을 더 많은 rank로 나누면 per-rank N이 minimum tile보다 작거나 divisibility를 잃을 수 있다. K group이 rank 경계를 가로지르면 act-order와 scale repeat 정책도 바뀐다. 메모리 확보를 위해 TP를 늘린 결정이 kernel compatibility를 깨는 serving-level trade-off다.

### 46.10.2 “Marlin을 요청했지만 다른 backend다”

effective quantization config, layer method class, selected kernel/backend startup log를 순서대로 본다. model 전체 label이 아니라 문제 layer마다 본다. lm_head, embedding, expert, dense linear가 서로 다른 method를 가질 수 있다.

device compute capability와 compiled architecture 목록, toolkit/build dependency를 확인한다. source가 SM 지원 분기를 갖더라도 installed binary에 specialization이 없을 수 있다. optional FlashInfer/CUTLASS module import 실패가 fallback을 만들 수 있다. backend override와 auto policy의 최종값을 저장한다.

fallback이 correctness를 보존하면 성능 문제이고, loader representation이 fallback ABI와 맞지 않으면 correctness 문제다. repacked Marlin weight를 generic GPTQ kernel에 그대로 넘기지 않도록 parameter representation과 method ownership을 본다.

### 46.10.3 “4비트인데 느리다”

먼저 logical byte fixture를 실제 shape로 계산한다. weight, scale, ZP, activation, output, padding, intermediate와 workspace를 분리한다. M이 커질수록 weight byte 절감 대비 compute와 activation traffic이 커지는 것을 표시한다. MoE는 routing padding amplification과 expert skew를 추가한다.

selected kernel이 W4A16인지 activation quantized mixed path인지 본다. online activation quantization과 scale generation이 작은 M에서 고정 비용이 될 수 있다. act-order permutation temporary, FP32 reduction buffer, atomic contention, output unpadding copy/view도 후보다.

weight load 시 repack 비용과 steady-state step 비용을 분리한다. model을 한 번 load하고 오래 serving하면 repack은 amortize되지만 serverless cold start나 frequent reload에서는 중요하다. graph capture를 다시 만들거나 workspace address가 바뀌면 준비 비용이 반복될 수 있다.

### 46.10.4 “MoE에서만 workspace가 커진다”

`M×T×max(2N,K)` cache 식과 expert padding rows를 계산한다. top-k가 2에서 8로 늘면 pair intermediate가 네 배가 될 수 있다. expert 수 자체보다 active pair와 padding distribution이 temporary 크기와 work를 정한다.

FP32 reduce temporary는 sorted token capacity, SM×blocks budget, block M/N으로 제한될 수 있다. dense Marlin의 작은 int workspace와 같은 것으로 합치지 않는다. FlashInfer/CUTLASS grouped backend는 plan/workspace 식이 다르므로 backend 교체 전후 peak를 같은 식으로 예측하지 않는다.

expert parallel이면 dispatch/receive buffer와 communication workspace가 추가된다. 이 장의 single-rank GEMM cache 식은 그 byte를 포함하지 않는다. local expert count, capacity factor, all-to-all payload를 분리하고 distributed communication 장과 연결한다.

### 46.10.5 “특정 expert에서만 값이 틀린다”

expert axis의 weight load/repack을 먼저 본다. expert마다 scale/ZP permutation loop가 모두 실행됐는지, shared expert가 별도 mapping을 쓰는지, expert-parallel local index가 맞는지 확인한다. 하나의 expert만 corruption되면 common GEMM code보다 expert slice 가능성이 높다.

router distribution에서 해당 expert가 선택될 때만 재현되는 최소 topk_ids를 만든다. 다른 expert weight를 identity로 두고 문제 expert만 coordinate pattern을 둔다. alignment sorted block의 expert ID, packed weight base pointer, scale base pointer를 함께 기록한다.

GEMM1만 틀리는지 GEMM2만 틀리는지도 나눈다. W1은 `2N×K` gated shape, W2는 `K×N`이라 packing axis와 group divisibility가 다를 수 있다. 한 weight만 shape padding path를 탈 수 있다. “Marlin MoE가 틀린다”보다 `expert e, W2, K group g, N tile n`까지 좁힌다.

### 46.10.6 변경 기록표

| 범주 | 필수 기록 |
|---|---|
| model/format | revision, GPTQ/AWQ/FP8/NVFP4, bits, scalar type, group/block size, symmetric/ZP, act-order |
| shape | original·per-rank·padded M/K/N, TP/EP, expert E, top-k T, counts와 padded counts |
| storage | qweight shape/stride/dtype, scale/ZP shape/stride/dtype, g_idx/permutation, activation scale, global scale |
| backend | framework commit, method class, selected op, SM, toolkit/build arch, optional package commit |
| scratch | workspace shape/dtype/address owner, FP32 temporary, intermediate cache alias와 lifetime |
| correctness | first bad code/group/tile/expert/stage, reference cast order, tolerance, router-weight 적용 위치 |

이 표를 채우면 “AWQ 모델이 느리다”는 보고가 실행 가능한 조사로 바뀐다. 빠진 필드가 있으면 어느 source boundary로 돌아갈지도 알 수 있다. format field가 모호하면 loader, padded shape가 없으면 Marlin utility, expert counts가 없으면 router/alignment, workspace owner가 없으면 runner를 본다.

## 46.11 Release·SM·toolkit gate는 75·76장의 artifact closure로 잇는다

gate 감사 fixture를 하나 더 둔다. 동일 checkpoint와 logical shapes를 GPU A와 B에 배포한다. A는 target specialization을 지원하고 B는 capability 또는 packaged code object 조건 때문에 generic fallback를 쓴다. 두 outputs는 reference tolerance 안에서 같아야 하지만 selected symbol, workspace, throughput은 달라도 된다. “두 GPU가 다른 kernel을 썼다”는 사실 자체는 오류가 아니다.

반대로 두 devices가 같은 Marlin label을 보고해도 실제 specialization이 다를 수 있다. bits/group, tile config, compiled architecture, dynamic shared-memory attribute를 기록한다. kernel symbol이 template parameters를 모두 드러내지 않으면 launcher trace에 decision tuple을 남긴다.

toolkit upgrade는 source-level pack utility를 바꾸지 않아도 compiler register allocation, code object selection, instruction lowering을 바꿀 수 있다. output mismatch가 생기면 ABI artifact를 먼저 검증하고, 같은 artifact/reference가 통과한 뒤 compiled kernel difference를 조사한다. repack와 compiler를 동시에 바꾸면 원인을 분리할 수 없다.

wheel/container에는 지원 architectures의 cubin/PTX inventory와 runtime driver compatibility가 있다. SM gate predicate가 true인데 image가 없어 JIT/fallback가 일어날 수 있다. build manifest, runtime selected image, compilation/JIT log를 source option과 함께 보존한다.

shared-memory opt-in이나 maximum dynamic shared bytes 설정 실패는 launch failure 또는 다른 tile fallback를 만들 수 있다. workspace bytes와 shared bytes는 다른 memory다. dashboard에서 둘을 `scratch` 하나로 합치지 않는다. CTA residence 계산에는 shared memory를, allocation pressure에는 global workspace를 넣는다.

register pressure는 repack correctness와 별개지만 tile selection 효과를 설명한다. 더 큰 tile이 weight reuse를 늘려도 registers/thread가 올라 occupancy와 spill을 바꿀 수 있다. performance regression에서 quant bits만 보지 않고 compiled resources, active warps, memory traffic를 함께 본다.

capability gate test는 supported exact path, supported alternate tile, explicit fallback, explicit rejection 네 결과를 구분한다. silent forced launch는 허용하지 않는다. fallback가 존재하지 않는 incompatible format이면 load 단계에서 이해 가능한 오류를 내고 partial model publish를 막는다.

distributed deployment에서는 모든 ranks가 compatible decision을 했는지 확인한다. 한 rank만 fallback하면 collective timing 차이뿐 아니라 workspace/output dtype contract가 달라질 수 있다. effective backend tuple을 rank startup handshake 또는 diagnostic snapshot로 비교한다.

heterogeneous cluster를 의도적으로 지원한다면 rank group을 capability-compatible workers로 구성하거나 common fallback를 선택한다. fastest rank의 gate를 전체 group에 적용하지 않는다. scheduler가 model replica를 routing할 때 replica effective backend와 capacity를 관측한다.

graph capture cache key에는 selected kernel specialization과 workspace/static pointer contract가 포함돼야 한다. toolkit upgrade나 gate change로 kernel이 바뀌었는데 old graph exec를 재사용하지 않는다. graph recapture와 repack rebuild는 별도 identities지만 release deployment에서 함께 invalidation될 수 있다.

scale format이 architecture ABI라는 말은 scale values의 수학적 의미가 GPU마다 달라진다는 뜻이 아니다. native kernel이 기대하는 packing, permutation, vector load, supported dtype가 specialization contract라는 뜻이다. logical scale reference를 변하지 않는 비교 기준으로 보존한다.

FP8/NVFP4처럼 hardware-assisted formats를 비교할 때도 같은 질문을 쓴다. logical tensor, block/group scale granularity, packed payload, scale encoding, kernel load/MMA contract를 분리한다. Marlin fixture의 exact permutation을 다른 format에 복사하지 않는다.

fallback 성능을 측정할 때 repack startup 비용을 steady inference에 섞지 않는다. cold model load, artifact cache warm load, first kernel, steady tokens/s를 나눈다. Marlin이 memory-bound workload를 개선하려는 의도와 현재 M/K/N workload에서 실제 손익을 구분한다.

incident canary가 generic fallback를 사용하면 service correctness는 회복돼도 capacity가 줄 수 있다. admission limit와 replica 수를 임시 조정하고 SLO 영향을 공개한다. correctness rollback와 performance recovery를 한 단계로 보고 무리하게 optimized kernel을 재활성화하지 않는다.

corrected Marlin 재활성화는 낮은 traffic에서 basis probes와 real-request shadow comparison을 먼저 수행한다. layer/expert별 max error와 signature mismatch를 보고 traffic를 단계적으로 늘린다. fallback reason distribution가 baseline으로 돌아왔는지도 확인한다.

최종 capability 표에는 GPU model nickname보다 compute capability, packaged targets, driver/toolkit, selected specialization, required shared/workspace, supported quant modes를 둔다. 운영자는 새 GPU 이름을 외우지 않고 이 contract로 지원 여부를 판정한다.

실제 승인 예시는 이렇게 읽는다. device capability가 gate를 통과하고 packaged cubin이 선택됐으며 K/N divisibility와 group size가 specialization predicate를 만족한다. qweight artifact ABI version과 native consumer version이 같고 workspace 640 bytes 이상이 allocation됐다. basis `(11,5)`와 MoE T0/E2 signature가 reference와 일치한다. 이때만 “Marlin active”라는 label이 의미를 가진다.

거절 예시도 구체적으로 남긴다. group size가 지원 집합 밖이거나 act-order/zero mode 조합이 specialization에 없으면 generic backend를 선택한다. fallback reason은 `unsupported_group`이나 `unsupported_act_order_zero`처럼 predicate를 드러낸다. 단순히 `kernel unavailable`이라고 하면 model conversion, packaging, capability, shape 가운데 어디를 고칠지 알 수 없다.

artifact 검토자는 header와 payload sample을 함께 본다. header version만 맞아도 corrupted 또는 잘못 permutation된 payload일 수 있다. 몇 개의 known logical coordinates를 unpack/repack round-trip하고 scale/zero associations를 검사한다. 전체 model dequant 없이도 deterministic probes로 artifact truth를 확인할 수 있다.

native consumer 검토자는 kernel launch 전에 sample mapping을 debug build 또는 host reference로 검산한다. production hot path에서 모든 coordinates를 검사하지 않고 model load/canary에서 표본화한다. mismatch가 있으면 optimized kernel을 publish하지 않는다. assertion을 제거해 launch를 강행하는 것은 해결이 아니다.

MoE consumer 검토자는 expert별 probe traffic를 보장한다. real workload에서 거의 선택되지 않는 expert는 wrong scale base가 오래 숨을 수 있다. synthetic one-hot router fixture나 direct expert kernel test로 모든 local experts를 덮는다. EP topology별 global/local mapping도 set equality로 검사한다.

workspace 검토자는 high-water와 bounds뿐 아니라 lifecycle을 본다. graph replay나 async kernel이 in-flight인데 buffer를 resize/reuse하지 않는지 generation을 기록한다. ABI mismatch와 lifetime bug가 동시에 wrong output을 만들 수 있으므로 format fix 뒤에도 completion edge를 검증한다.

성능 검토자는 kernel duration만으로 repack의 가치를 판단하지 않는다. model load/repack cost, artifact disk size, warm cache hit, fallback fraction, end-to-end ITL/throughput을 함께 본다. 작은 batch와 큰 batch의 memory/compute balance가 다르므로 workload strata를 공개한다.

release review는 producer와 consumer 변경을 별도 diff로 표시한다. permutation table, scale layout, cache header가 바뀌면 ABI migration 계획이 필요하다. kernel tile만 바뀌고 artifact contract가 유지된다면 불필요한 cache rebuild를 피할 수 있다. 무엇이 stable이고 무엇이 versioned인지 명시한다.

마지막 regression은 old artifact/new consumer, new artifact/old consumer 조합이 명시적으로 거부되는지 본다. 우연히 shape가 같아 실행되는 것이 가장 위험하다. compatibility matrix가 허용한 조합만 load하고 나머지는 rebuild 또는 fallback한다.

이렇게 gate를 읽으면 capability는 단순 지원 목록이 아니다. logical quant format을 특정 packed artifact와 kernel specialization, workspace, compiled target에 연결하는 배포 계약이다. wrong output incident는 그 계약의 어느 좌표가 갈라졌는지로 설명하고 복구한다.

최종 terminal에는 세 가지 반증이 나란히 있어야 한다. logical dequant reference가 맞아 checkpoint format 오류를 배제하고, generic quant kernel이 맞아 routing과 activation 오류를 좁히며, corrected Marlin artifact가 basis와 workload 모두에서 맞아 ABI fix를 증명한다. MoE에서는 top-k=1 direct expert와 top-k=2 combine을 모두 통과한다.

그 뒤 cold repack와 warm cache가 동일 payload hash와 coordinate probes를 내는지 확인한다. SM/toolkit가 다른 worker는 compatible artifact를 읽거나 명시적으로 rebuild/fallback한다. graph replay와 eager가 같은 outputs를 만들며 workspace generation overlap이 없다. 이 조건이 90분 유지되고 unexpected fallback가 0일 때 optimized path를 전면 재개한다.

사후 문서에는 단순히 “scale permutation 수정”이라고 쓰지 않는다. old producer order, new consumer order, failing `(layer,expert,k,n)` sample, physical slots, cache key 누락 field, rollback backend와 성능 비용을 적는다. 독자는 이 기록으로 다음 ABI 변경에서 같은 검사를 반복할 수 있다.

결국 4-bit라는 작은 저장 단위가 복잡성의 핵심은 아니다. 어려운 부분은 logical 의미가 pack, permutation, sharding, expert sorting, tile/lane loading을 지나도 동일 identity를 유지하는가다. 그 identity를 좌표와 source로 보존하면 silent wrong output을 first divergence에서 멈출 수 있다.

승인자는 마지막으로 cache artifact 하나를 임의 선택해 cold producer와 warm consumer의 commit, topology, capability, permutation tuple을 대조한다. basis 좌표가 header 주장과 실제 payload에서 일치하고 fallback 전환도 같은 reference를 유지해야 한다. 이 표본 감사까지 실제로 통과해야 최종 운영 배포 승인 기록을 닫는다.

**GPU 이름보다 compute capability**

backend 문서가 “Ampere 이상”이라고 요약해도 dispatch는 보통 compute capability의 major/minor와 compiled specialization을 본다. A100의 SM80과 소비자 Ampere의 SM86은 shared capacity와 지원 path가 완전히 같다고 가정할 수 없다. Blackwell도 SM100과 SM120을 하나의 숫자로 합치지 않는다.

Marlin W4A16의 supported quant types와 architecture gate는 `query_marlin_supported_quant_types`, `check_marlin_supported`에서 시작한다. Python이 지원한다고 판단해도 native binary가 해당 arch로 compile됐는지 확인해야 한다. fatbin에 cubin이 없고 PTX JIT fallback도 적절하지 않으면 source의 branch에 도달하지 못한다.

반대로 binary에 kernel이 있어도 shape와 format gate가 막을 수 있다. `N/K` divisibility, group size, zero-point support, activation dtype, bias, reduction mode가 specialization key다. “SM80에서 Marlin 지원”은 모든 GPTQ/AWQ tensor에 대한 명제가 아니다.

**CUDA 12.x와 13.x를 version 숫자로만 비교하지 않는다**

toolkit은 compiler, PTX ISA, headers, libraries와 supported host compiler를 묶는다. CUDA 13.x 설치는 CUDA 12.x에서 build한 wheel의 embedded kernel을 자동 재compile하지 않는다. framework wheel이 어떤 toolkit과 arch list로 build됐는지, JIT extension이면 어느 compiler를 실제 사용했는지 나눈다.

새 toolkit이 새로운 type이나 instruction wrapper를 제공해도 target SM이 hardware support를 가져야 한다. 반대로 hardware가 기능을 가져도 old CUTLASS/CuTeDSL 또는 framework source가 specialization을 구현하지 않았을 수 있다. `hardware capability ∧ toolkit/compiler exposure ∧ library implementation ∧ build inclusion ∧ dispatch gate`가 모두 참이어야 한다.

SGLang의 JIT Marlin wrapper는 dtype, expert parallel, bias를 source key로 삼는다. cache된 extension이 이전 toolkit/build option에서 만들어졌다면 source change와 binary가 어긋날 수 있다. JIT cache key와 build log, loaded module 경로를 기록한다. cache 삭제를 첫 해결책으로 쓰기 전에 mismatch 증거를 확인한다.

FlashInfer의 CuTeDSL NVFP4와 masked grouped GEMM은 Blackwell-specific layout과 instruction을 사용할 수 있다. API가 import된다는 것, trace/reference function이 있다는 것, 현재 GPU에서 compiled kernel이 launch 가능하다는 것은 세 단계다. `has_*` helper와 runner selection, actual compiled module을 연결한다.

**scale format도 architecture ABI다**

Tensor Core가 block-scaled FP4를 소비할 때 data fragment와 scale-factor fragment의 tile 관계가 정해진다. scale tensor가 수학적으로 같은 `(block,n)` 값을 가져도 physical interleave가 instruction/kernel expectation과 다르면 틀린다. architecture별 preprocess가 필요한 이유다.

NVFP4 global scale과 FP8 block scale의 multiply 위치, supported scale dtype, block axis는 backend ABI다. SM100용 packed scale을 SM120 kernel이 그대로 받는다고 추정하지 않는다. current source의 validation과 preprocess function을 따라간다.

INT4 Marlin도 architecture-independent checkpoint qweight를 architecture-friendly Marlin layout으로 repack한다. repacked artifact를 disk cache에 저장한다면 cache key에 framework/kernel revision, scalar type, tile format, TP shard shape를 포함해야 한다. GPU model 이름만 key로 쓰면 source layout change 뒤 stale artifact를 읽을 수 있다.

**gate 실패를 우회하지 않는다**

shape check나 dtype assert를 없애 kernel을 강제로 호출하는 것은 optimization이 아니다. check가 보호하는 native precondition을 먼저 읽는다. minimum tile, vector alignment, group divisibility, workspace capacity가 깨지면 wrong answer나 out-of-bounds가 날 수 있다.

fallback이 있다면 correctness reference로 활용할 수 있다. 같은 quantized logical weight를 generic dequant+dense matmul 또는 alternate backend로 계산해 Marlin과 비교한다. 그러나 fallback이 다른 packed ABI를 요구하면 loader 단계에서 representation을 다시 만들어야 한다. op 이름만 바꾸지 않는다.

새 shape support를 추가하려면 padding proof, repack inverse test, scale/ZP padding, output unpad, TP shard, act-order, MoE expert axis를 포함한 tests가 필요하다. 성능 benchmark 전에 exact integer identity와 small GEMM correctness를 닫는다.

**한 layer를 끝까지 기록하는 예시**

가상의 Mixtral 계열 expert W1을 조사한다고 하자. global shape는 expert 8개, input K 4096, gate/up output 28672이고 TP=2 column partition을 사용한다. per-rank N은 14336이다. checkpoint는 AWQ INT4, group size 128, asymmetric ZP, activation BF16이라고 한다.

첫 기록은 이름이 아니라 tensor다. logical expert weight shape `(8,4096,14336)` per rank, qweight int32 shape와 packed axis, scale `(8,32,14336)`, zero-point packed shape를 적는다. scale group 32는 `4096/128`과 일치해야 한다. bias가 있는지, scalar type의 unsigned bias가 얼마인지도 적는다.

Marlin shape family를 계산한다. K 4096은 128과 64 모두로 나뉘고 N 14336은 64와 128로 나뉜다. 이 fixture에서는 padding이 필요 없을 수 있다. 그러나 loader가 gate/up shard를 interleave하거나 expert-parallel로 다시 shard한다면 실제 op의 N을 다시 확인한다. model config dimension에서 support를 선언하지 않는다.

AWQ qweight를 expert별 Marlin tile로 repack하고 scale과 zero-point를 expert axis를 유지한 채 permute한다. expert 3, K group 7, N channel 129의 code/scale/ZP tuple을 checkpoint와 repacked inverse에서 비교한다. 한 point만 아니라 tile corner와 group boundary를 포함한 deterministic sample, 가능하면 전체 identity test를 둔다.

workspace는 device SM 수에 4 blocks-per-SM을 곱한 int32 state로 만들 수 있다. GEMM1의 FP32 reduce temporary와 intermediate cache를 별도 기록한다. decode step `M=5,T=2`이면 logical pair 10개, block M=4와 counts `[4,3,2,1,0,0,0,0]`일 때 padded rows 16이다. cache1 logical payload는 앞서 계산한 573,440 byte지만 padded native temporary가 16 rows를 쓰면 `16×28672×2=917,504 byte`까지 모델을 넓혀야 한다. wrapper가 cache를 logical pairs로 쓰고 kernel internal만 padded slots를 쓰는지 source에서 구분한다.

router가 token 0을 expert 3과 1로 보냈다면 sorted list에서 두 pair가 각각 한 번 나타나야 한다. expert 3 block은 expert 3의 packed weight base와 scale/ZP base를 가져야 한다. GEMM1 output을 pair ID 기준으로 원래 order에 대응시킨 뒤 gated activation을 확인한다. W2 down GEMM은 per-rank/parallel mode에 따라 output reduction이 추가될 수 있다.

router weights `[0.25,0.75]`가 GEMM1 input에 적용되지 않는 mode라면 GEMM2 또는 combine에서 정확히 한 번 곱한다. token 0 final output은 expert 3 contribution의 0.25와 expert 1 contribution의 0.75 합이다. expert ID를 바꾼 negative fixture와 weight를 swap한 fixture가 다른 결과를 내야 test가 mapping 오류를 검출한다.

이 한 장의 기록으로 support, packing, workspace, routing, combine을 연결할 수 있다. 값이 틀리면 마지막 logits가 아니라 tuple identity, pair multiset, GEMM1 pair row, activation row, GEMM2 pair row, final token 순으로 돌아간다. 느리면 padding amplification, selected backend, activation quantization, workspace reduction, expert skew를 같은 shape 장부에서 본다.

**포맷 변환 cache의 무효화 규칙**

대형 model의 repack은 model load 시간을 늘리므로 변환된 weight를 cache하고 싶어진다. 그러나 cache payload는 원 checkpoint와 동등한 범용 weight가 아니라 특정 kernel ABI일 수 있다. key가 부실하면 조용한 wrong answer를 만든다.

최소 key에는 원 tensor content hash와 quantization config, framework commit, repack implementation revision, scalar type, group size, ZP/act-order, original/padded K/N, TP/EP rank와 shard axis, target architecture/layout version을 넣는다. scale와 ZP artifact도 qweight와 같은 generation ID로 묶는다. qweight만 새 cache이고 scale은 이전 cache인 혼합을 허용하지 않는다.

CUDA toolkit patch version을 항상 key에 넣어야 하는지는 artifact가 host-side layout만 담는지 generated binary까지 담는지에 달렸다. binary/JIT module cache는 compiler, toolkit, arch flags와 dependency version이 필요하다. pure repacked data는 kernel layout revision이 핵심이다. 둘을 같은 cache directory에 두더라도 manifest type을 구분한다.

load 시 shape/dtype/stride와 expected padded dimensions를 검증하고, 작은 sentinel tile의 inverse identity 또는 artifact checksum을 확인한다. expert 수와 expert map도 확인한다. validation 실패 시 원 checkpoint에서 안전하게 다시 만들고 원인을 기록한다. stale cache를 읽은 뒤 kernel error로 보이는 시간을 줄이는 장치다.

model reload에서는 weight artifact뿐 아니라 workspace와 graph-captured pointer ownership을 함께 갱신한다. weight 내용만 바뀌고 shape가 같아도 graph가 새 weight storage를 참조하도록 capture/registration contract를 확인한다. workspace는 기존 주소를 zero/reuse할 수 있지만 서로 다른 model instance가 같은 semaphore state를 동시에 공유하면 안 된다.

**정확성과 성능 변경을 한 PR에서 분리한다**

새 quant format이나 shape를 지원하는 변경은 먼저 representation correctness를 닫는다. checkpoint unpack, repack inverse, scale/ZP permutation, padding zero contribution, TP shard, small GEMM tests를 추가한다. 이 단계의 합격 기준은 exact code identity와 명시한 numerical tolerance다.

그다음 kernel specialization과 scheduling을 바꾼다. tile, stage, reduction, atomic 선택을 바꿀 때 output contract tests를 그대로 통과시킨다. 성능 결과는 M/K/N과 batch, SM, dtype, group, option을 붙인다. correctness와 speed change를 한 숫자로 합치지 않는다.

MoE는 routing/alignment tests를 독립시킨다. 임의 expert counts와 empty/skewed expert, top-k sentinel, expert map, padded block을 property test로 검사한다. expert GEMM은 fixed alignment input에서 따로 검사하고 combine은 synthetic pair output으로 검사한다. end-to-end 하나만 두면 어느 모듈이 깨졌는지 알 수 없다.

reviewer는 새 branch가 fallback과 같은 parameter representation을 소비하는지 확인한다. loader가 Marlin repack을 수행한 뒤 조건에 따라 generic kernel로 fallback한다면 generic kernel이 repacked layout을 이해하는지 증명해야 한다. 그렇지 않으면 fallback 전에 canonical representation을 보존하거나 backend별 parameter를 소유해야 한다.

마지막으로 observability field를 추가한다. effective format, original/padded shape, selected backend/specialization, workspace bytes, MoE padded pair count를 debug log 또는 structured trace에 남길 수 있어야 한다. 값 자체를 상시 높은 cardinality metric으로 노출할지는 운영 설계를 따르지만, 재현 시 꺼낼 좌표가 있어야 한다.

**통제 실험과 선택 결정 트리.** 같은 packed weight와 activation으로 dequantize-then-FP16 GEMM reference, Marlin 경로와 fallback을 차례로 실행한다. 실험은 output 오차뿐 아니라 selected kernel, workspace, pack copy와 bytes/token을 기록한다. 값이 먼저 다르면 scale·zero-point·group mapping을, 값은 맞고 느리면 SM target·alignment·workspace copy를, MoE에서만 느리면 expert sort·token imbalance를 진단한다. requested Marlin이 fallback이면 빠르다는 판정을 보류한다.

## 46.12 종합 회고

구현 좌표는 vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, llama.cpp v0.2.0 commit `bb4caa7540188872173c44d161602d9271386413`, FlashInfer v0.6.17 commit `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`에 고정했다.

논문의 역할은 구현 ABI와 다르다. [MARLIN `2408.11743v1`](https://arxiv.org/abs/2408.11743v1)은 mixed-precision autoregressive linear kernel의 설계 의도를, [GPTQ `2210.17323v2`](https://arxiv.org/abs/2210.17323v2)는 approximate second-order one-shot weight quantization을, [AWQ `2306.00978v4`](https://arxiv.org/abs/2306.00978v4)는 activation-aware salient-channel scaling을 설명한다. 논문의 quantizer, 현재 packed ABI와 source 좌표의 성능 수치는 서로 대신하지 않는다.

처음의 세 장애로 돌아가자. “unsupported shape”는 4-bit 자체의 문제가 아니라 per-rank K/N, group size와 Marlin tile family의 불일치였다. 첫 token부터 틀린 AWQ model은 quantization 오차가 아니라 zero-point convention과 scale permutation의 representation 오류일 수 있었다. 느린 model은 ideal weight byte만 보고 activation, dequant, reduction, padding과 batch compute를 빼먹었다.

이 장의 첫 번째 원칙은 quantization algorithm, checkpoint format, kernel layout을 분리하는 것이다. GPTQ는 approximate second-order 정보를 이용해 weight code를 고른다. AWQ는 activation-aware scaling으로 salient channel의 error를 줄인다. Marlin은 선택된 code를 GPU tile이 효율적으로 소비하도록 repack하고 dequant/MMA pipeline을 만든다. 셋은 연결되지만 같은 이름이 아니다.

두 번째 원칙은 byte 계산에 metadata를 포함하는 것이다. W4 code는 BF16 weight의 4분의 1이지만 group scale, optional zero-point, padding이 있다. W4A8/FP8/NVFP4는 activation code와 scale, quantization work가 더해진다. logical byte 절감이 elapsed-time speedup과 같지 않다. shape와 batch가 weight reuse와 compute 비율을 바꾼다.

세 번째 원칙은 packing을 coordinate transform으로 읽는 것이다. logical `(k,n)`, packed word/bit, scale group, kernel tile/lane 좌표를 구분한다. repack은 logical code를 보존해야 하며 scale/ZP permutation은 같은 logical weight를 복원해야 한다. act-order에서는 weight와 activation K permutation이 한 쌍이다.

네 번째 원칙은 workspace에 수명과 주소가 있다는 것이다. Marlin의 int32 coordination state는 SM과 blocks-per-SM에 연결되고 graph capture 뒤 같은 storage 주소를 재사용해야 할 수 있다. MoE의 FP32 reduction temporary와 gate/up/down intermediate는 훨씬 크며 alias가 lifetime proof에 의존한다. “scratch”라는 이름이 공짜나 무상태를 뜻하지 않는다.

다섯 번째 원칙은 MoE가 GEMM 전에 batch를 다시 만든다는 것이다. router의 `M×T` pairs를 expert별로 count하고 block alignment에 맞춰 padding하며 sorted token ID와 expert ID를 보존한다. W1, activation, W2 뒤에는 router weight를 정확히 한 번 적용해 원래 token으로 combine한다. expert skew는 padding과 parallelism을 서로 다른 방향으로 바꿀 수 있다.

여섯 번째 원칙은 네 codebase의 공통 아이디어와 ABI를 분리하는 것이다. vLLM Marlin, SGLang JIT wrapper, llama.cpp GGUF CUDA MMQ, FlashInfer NVFP4 grouped MoE는 packed byte를 tile에서 복원한다는 문제를 공유한다. 그러나 block encoding, scale, zero, tile, workspace와 dispatch가 다르다. Q4 또는 FP4라는 이름만으로 호환성을 선언하지 않는다.

마지막 원칙은 first-divergence 순서다. checkpoint code, float dequant, repack identity, scale/ZP, act-order, 한 K group GEMM, reduction/epilogue, activation scale, MoE alignment, GEMM1, activation, GEMM2, combine을 앞에서부터 비교한다. 앞 단계가 틀렸는데 뒤 kernel counter를 보는 것은 원인을 멀리 보낸다.

독자가 새 quantized model을 만났을 때 다음 문장을 완성할 수 있어야 한다. “이 layer는 format F의 code와 scale/ZP를 loader L이 logical shape K/N에서 Marlin tile P로 변환하고, per-rank padded shape K′/N′와 workspace W를 op O에 넘긴다. MoE라면 router pairs를 alignment A가 P rows로 만들고 두 expert GEMM과 activation 뒤 policy C가 weighted combine한다.” 빈칸 하나가 증거가 없는 경계다.

이렇게 읽으면 4-bit는 마법의 속도 옵션이 아니라 검증 가능한 데이터 계약이 된다. weight byte를 줄인 이득, dequant/MMA pipeline의 비용, MoE 동적 batch의 padding과 combine, architecture/build gate가 하나의 인과 사슬로 연결된다. 다음 장의 일반 kernel 진단은 바로 이 사슬에서 처음 갈라진 지점을 찾은 뒤 시작한다.

소스가 복잡할수록 이 순서가 더 중요하다. template 이름이나 backend 브랜드를 외우는 대신 한 logical weight와 한 token-expert pair를 끝까지 추적한다. 어느 좌표 변환에서 code가 이동하고, 어느 scale이 붙으며, 누가 temporary를 소유하고, 어느 단계에서 원래 token으로 돌아오는지 설명할 수 있다면 새로운 format과 kernel도 같은 방법으로 검증할 수 있다. 설명할 수 없는 경계는 성급하게 우회할 최적화의 여지가 아니라 아직 더 구체적인 증거가 필요한 지점이다.
