# 50장. 4비트라는 이름 뒤의 계약 — scale axis에서 runtime ABI까지

서버는 정상적으로 시작됐다. weight 파일도 모두 열렸고 missing key와 shape mismatch도 없었다. 새 FP8 checkpoint는 dense BF16보다 메모리를 덜 썼고 backend log에도 기대한 scaled GEMM 이름이 보였다. 그런데 첫 token부터 logits가 달랐다. 오차는 무작위처럼 보이지 않았다. 어떤 output channel은 거의 맞았고, 다른 channel은 두 배나 네 배의 일정한 비율로 어긋났다. 같은 model을 작은 square matrix로 시험하면 통과했지만 실제 non-square projection에서는 즉시 실패했다.

이 장은 이 장면에서 출발한다. 커널의 MMA pipeline을 다시 설명하지 않는다. 46장에서 이미 packed weight가 GPU tile로 들어가 unpack·dequant되는 과정과 resource trade-off를 살펴봤다. 여기서 필요한 질문은 그보다 앞에 있다. checkpoint의 `weight_scale`이 어떤 축을 뜻했는가. loader가 transpose와 TP shard를 적용한 뒤 그 축은 어디로 갔는가. runtime parameter는 direct scale을 들고 있는가 inverse scale을 들고 있는가. 선택된 backend는 scalar, channel, token, block 가운데 어느 broadcast ABI를 기대했는가.

“FP8 모델”이나 “GPTQ 4비트 모델”이라는 이름은 이 질문에 답하지 않는다. GPTQ와 AWQ는 weight code를 고르는 방법을 설명한다. checkpoint schema는 code와 scale을 어떤 tensor 이름·shape·container로 저장하는지 정한다. serving loader는 그 tensor를 shard하고 재배치한다. runtime backend는 다시 자신이 소비할 pack order와 scale stride를 요구한다. 네 층의 이름이 비슷해도 byte 호환성은 자동으로 생기지 않는다.

이 장을 다 읽은 뒤에는 한 weight coordinate를 끝까지 추적할 수 있어야 한다. config의 bits·group size·act-order에서 시작해 source tensor의 word와 nibble, scale/ZP/g_idx 좌표를 복원하고, TP/EP slice와 repack 뒤 runtime parameter를 찾고, backend가 거절됐을 때 fallback이 현재 representation을 안전하게 읽는지 판정한다. 값이 틀리면 config, native unpack, slice, repack, runtime dequant 가운데 최초로 갈라진 경계를 찾는다.

조사는 current source에 고정한다. Transformers v5.15.1 commit `550d7b3834670483a4df436541272c055dc364bf`, vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, llama.cpp v0.2.0 commit `bb4caa7540188872173c44d161602d9271386413`, FlashInfer v0.6.17 commit `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`이다. 논문은 algorithm의 의도를 설명하고, 현재 source는 file-to-runtime ABI를 증명한다.

## 50.1 shape가 맞는데 scale axis가 틀린 사건

이 장의 비교 단위는 서로 다른 임의 예제가 아니라 **같은 4×8 logical weight 32개를 GPTQ·AWQ·FP8·NVFP4 네 표현으로 옮긴 128-value 블록**이다. 네 포맷은 같은 실수 좌표를 출발점으로 삼고, 각 32개 표현값의 pack 순서·scale 소유자·zero point 유무·consumer 해석을 나란히 적는다. 따라서 포맷별 설명이 끝난 뒤에도 비교 분모가 바뀌지 않는다.

### 50.1.1 non-square fixture가 숨은 축을 드러낸다

문제를 가장 작은 행렬로 줄여 보자. logical weight를 `W[K=3,N=4]`라 하고 output-channel scale을 `s=[1,2,4,8]`로 둔다. 저장 code의 각 column은 일부러 모두 같은 pattern을 갖게 한다. activation은 첫 K basis vector를 차례로 넣는다. 올바른 복원은 `W[k,n]≈q[k,n]·s[n]`이다. 따라서 같은 code라도 N=0과 N=3 output은 여덟 배 차이가 난다.

잘못된 loader가 scale을 K축으로 보아 `s[k]`를 적용하면 무엇이 생길까. 첫 activation row에서는 모든 output이 같은 배율을 받고, 둘째 row에서는 모두 두 배가 된다. output channel에 따라 달라져야 할 비율이 input coordinate에 따라 달라진다. 이 pattern은 floating rounding과 다르다. 오차의 방향이 axis를 가리킨다.

왜 square fixture는 이 문제를 숨겼을까. K=N=4이고 scale tensor가 길이 4라면 K축과 N축 모두 shape validation을 통과한다. transpose를 두 번 잘못 적용해도 최종 shape는 같다. 그래서 quantized loader 테스트에 square matrix만 두면 위험하다. K=3,N=4처럼 축 길이를 다르게 하고 각 축의 값 pattern도 다르게 둬야 한다.

첫 divergence ledger는 다음 순서를 가진다.

### 50.1.2 최초로 의미가 바뀐 경계를 고정한다

```text
checkpoint scale name/shape/meaning
→ quant config granularity
→ loader transpose와 TP slice
→ runtime parameter shape/stride
→ backend broadcast index
→ first dequantized weight coordinate
→ matmul output
```

checkpoint tensor까지 맞고 loader parameter에서 처음 transpose됐다면 커널을 profile할 이유가 없다. runtime parameter까지 맞고 backend가 scale stride를 scalar처럼 전달했다면 kernel call ABI 경계다. output만 보고 “FP8 오차가 크다”고 tolerance를 넓히면 representation bug를 quantization error로 숨긴다.

scale 이름만으로 축을 추정해서도 안 된다. `weight_scale`, `weight_scale_2`, `input_scale`, `activation_scale`, `global_scale`, `scale_inv`는 stack마다 다른 lifetime과 방향을 가질 수 있다. direct multiplier인지 reciprocal인지, scalar인지 vector인지, calibrated static tensor인지 request마다 계산한 dynamic tensor인지 consumer 식에서 확인한다.

이 사건의 수정은 단순 transpose 한 줄일 수 있다. 그러나 종료 fixture는 더 넓다. non-square weight, TP column shard와 row shard, singleton scale dimension, contiguous와 transposed view, scalar/per-channel mode를 통과해야 한다. first dequantized coordinate가 reference와 일치하고 최종 logits가 사전 tolerance 안에 있어야 한다. backend가 scale mode를 지원하지 않으면 조용한 broadcast가 아니라 compatible fallback 또는 명시적 거절이 필요하다.

## 50.2 algorithm·checkpoint·runtime ABI는 서로 다른 층이다

### 50.2.1 같은 quant 이름이 byte 호환성을 뜻하지 않는다

[GPTQ `2210.17323v2`](https://arxiv.org/abs/2210.17323v2)는 approximate second-order 정보를 사용해 one-shot weight quantization error를 줄이는 방법을 설명한다. [AWQ `2306.00978v4`](https://arxiv.org/abs/2306.00978v4)는 activation statistics로 salient channel을 찾고 equivalent scaling을 적용하는 이유를 설명한다. 두 논문은 왜 특정 low-bit code와 scale을 선택하는가에 답한다. current vLLM이 qweight를 어느 Marlin tile 순서로 재배치하는지는 논문 claim이 아니다.

checkpoint schema는 선택된 logical code를 tensor로 만든다. 4-bit code 여러 개를 int32 word에 넣을 수 있고, scale은 FP16 tensor, zero-point는 별도 packed tensor, act-order는 `g_idx`로 저장할 수 있다. int32는 compute dtype이 아니라 container다. `qweight.dtype == int32`를 보고 INT32 GEMM이라고 말하면 representation의 층을 잃는다.

loader는 file tensor를 그대로 보관하지 않을 수 있다. TP rank가 logical K/N 일부를 선택하고, pack factor를 반영해 physical slice offset을 계산하며, backend가 원하는 tile order로 qweight를 repack한다. scale과 zero-point도 같은 semantic permutation을 따라야 한다. qweight만 옮기고 scale을 원래 순서에 두면 모든 shape와 byte 수가 맞아도 값은 틀린다.

runtime ABI는 변환 결과의 새 owner다. generic GPTQ kernel이 checkpoint-native layout을 기대하고 Marlin이 repacked layout을 기대한다면 두 pointer는 교환할 수 없다. backend selection이 post-load conversion 뒤 바뀔 수 있는지, fallback이 어느 representation을 소비하는지 lifecycle로 확인해야 한다. “둘 다 GPTQ”는 호환성 증거가 아니다.

이 분리는 debugging 순서도 정한다. 먼저 checkpoint-native unpack으로 logical q/scale/ZP를 복원한다. 다음으로 shard 결과가 같은 logical subset인지 확인한다. 그다음 repack inverse identity를 본다. 마지막으로 runtime dequant와 output을 본다. 가장 앞에서 틀린 층을 고치며 뒤 커널에서 보정하지 않는다.

## 50.3 nibble과 word를 손으로 복원한다

### 50.3.1 logical coordinate와 bit shift를 분리한다

`W[K=4,N=4]`의 unsigned code를 첫 row `0,1,2,3`, 둘째 `4,5,6,7`, 셋째 `8,9,A,B`, 넷째 `C,D,E,F`로 둔다. fixture가 N축 인접 두 code를 uint8에 low nibble first로 묶는다면 첫 row bytes는 `0x10,0x32`다. high nibble first면 `0x01,0x23`이다. 두 결과를 눈으로 구별할 수 있어야 한다.

여기서 little-endian과 low-nibble-first를 같은 말로 쓰면 안 된다. endianness는 multi-byte word가 memory byte로 배열되는 순서를 말한다. nibble order는 word 안에서 logical coordinate가 어느 bit shift를 쓰는지 말한다. `code_i=(word>>(4*i))&0xF`인지, tile interleave 뒤 다른 i를 쓰는지는 source의 shift와 permutation이 정한다.

int32에 여덟 code를 묶으면 세 좌표가 생긴다. logical `(k,n)`, packed tensor element index, word 내부 bit shift다. Marlin repack이 추가되면 tile 내부 좌표도 생긴다. 조사 worksheet는 이 네 단계를 생략하지 않는다. 한 점만 보지 않고 word 첫/끝, group 경계, tile corner와 padded tail을 고른다.

group size 2이고 scale logical shape가 `[K-group=2,N=4]`라면 k=0,1은 group 0, k=2,3은 group 1을 쓴다. asymmetric이면 `w≈s[g,n]·(q-z[g,n])`이다. symmetric이면 ZP가 없거나 backend convention의 implicit bias를 사용할 수 있다. code 0이 float zero라고 가정하지 않는다.

pack→unpack identity와 dequant correctness는 다른 테스트다. nibble을 정확히 복원해도 wrong scale axis면 dequant가 틀린다. dequant가 맞아도 repack permutation에서 scale과 qweight가 갈라질 수 있다. 따라서 native unpack, native dequant, repack inverse, runtime dequant를 네 checkpoint로 둔다.

## 50.4 group size와 act-order는 weight와 activation을 함께 움직인다

### 50.4.1 permutation은 weight·scale·activation의 공동 계약이다

GPTQ checkpoint에 `desc_act` 또는 act-order가 켜지면 `g_idx`를 장식 metadata로 취급할 수 없다. 보통 group size만 본 구현은 k번째 column이 `floor(k/G)` group scale을 쓴다고 생각한다. 하지만 act-order가 저장한 순서에서는 각 K coordinate가 참조하는 group이 단조롭지 않을 수 있다. weight를 sorted K order로 repack하면 activation X의 K column도 같은 permutation으로 읽어야 dot product가 보존된다.

K=8, group size 2인 fixture에서 `g_idx=[1,1,0,0,3,3,2,2]`를 둔다. group별 scale은 `[1,10,100,1000]`처럼 구별한다. `floor(k/2)`와 `g_idx[k]`가 첫 coordinate부터 다른 값을 만들게 해야 한다. 모든 scale을 1로 두면 g_idx 누락이 숨는다. activation도 basis vector와 서로 다른 값을 가진 dense row를 함께 사용한다. basis는 어느 column이 틀렸는지 보여 주고 dense row는 permutation을 두 번 적용하거나 전혀 적용하지 않은 오류를 드러낸다.

조사자는 `desc_act=true`라는 config에서 바로 kernel로 뛰지 않는다. config deserialization 뒤 effective method object가 flag를 보존하는지, parameter 생성에 `g_idx` storage가 있는지, source name이 loaded-name inventory에서 소비되는지, post-load sort가 어떤 permutation과 inverse를 만드는지 따라간다. qweight, scale와 activation reader가 같은 bijection을 공유하는지가 invariant다.

`g_idx` tensor가 load됐어도 runtime에서 무시될 수 있다. backend가 full K만 지원하거나 act-order specialization을 따로 고를 수 있다. 지원하지 않는 backend가 성능을 위해 g_idx를 버리는 것은 fallback이 아니라 wrong answer다. generic GPTQ path로 남거나 명시적으로 거절해야 한다. log에 Marlin이라는 이름이 보이는 것보다 effective method와 argument가 더 강한 증거다.

TP가 끼면 문제가 더 미묘해진다. row-parallel linear는 rank마다 K slice를 가질 수 있다. global g_idx를 local K coordinate로 자를 때 group boundary와 permutation의 domain을 보존해야 한다. column-parallel은 N을 자르므로 scale/ZP의 N axis도 같은 slice를 가져야 한다. packed qweight의 physical dimension을 logical K/N처럼 자르면 pack word 중간을 잘못 공유하거나 버릴 수 있다.

그래서 shard ledger는 global logical range, local logical range, packed storage range를 따로 둔다. 예를 들어 K=128, group size 32, int32당 8 codes일 때 rank boundary가 group과 word에 모두 정렬되는지 계산한다. 정렬되지 않으면 loader가 unpack/repack 또는 padding을 하는지, 해당 TP 조합을 거절하는지 source에서 확인한다. integer division으로 조용히 내림하면 마지막 codes가 사라질 수 있다.

종료 fixture는 act-order on/off, group size 2와 channel-wise convention, TP K/N slice, g_idx가 단조/비단조인 경우를 포함한다. native logical dequant와 shard를 합친 결과가 global reference와 같아야 한다. runtime output만 맞는 우연에 기대지 않는다. permutation을 두 번 적용한 오류가 대칭 input에서 상쇄될 수 있으므로 coordinate 식별 pattern을 유지한다.

## 50.5 FP8에서는 dtype 이름보다 scale broadcast를 먼저 읽는다

### 50.5.1 code format과 broadcast coordinate를 함께 읽는다

[FP8 Formats for Deep Learning `2209.05433v2`](https://arxiv.org/abs/2209.05433v2)는 E4M3과 E5M2가 exponent와 mantissa 범위를 달리해 학습·추론 tensor에 사용할 수 있는 배경을 제공한다. 그러나 논문에서 FP8을 지원한다고 current serving loader의 scale axis가 정해지는 것은 아니다. checkpoint는 E4M3 weight와 static scale을 저장할 수 있고, runtime은 BF16 activation을 token별 FP8로 바꾸며 dynamic scale을 만들 수 있다. 둘은 같은 “FP8 scale”이 아니다.

weight code `Wq`, activation code `Xq`를 생각하면 output은 대략 accumulator에 activation scale과 weight scale을 결합한다. 여기서 식 하나보다 broadcast coordinate가 중요하다. weight scale이 per-tensor면 모든 `(k,n)`이 하나를 쓴다. per-output-channel이면 n에 따라 달라진다. blockwise면 `(floor(k/Bk),floor(n/Bn))`를 쓴다. activation per-token scale은 m에 따라 달라진다. backend ABI는 각 pointer의 shape와 stride를 통해 이 좌표를 구현한다.

storage dtype도 분리한다. E4M3 값이 framework의 `float8_e4m3fn` tensor로 저장될 수 있고, 같은 bits를 `uint8` view로 운반할 수도 있다. 첫 경우 dtype cast가 format semantics를 갖고, 둘째는 consumer가 bits를 E4M3으로 해석해야 한다. `uint8`을 integer affine quantization으로 계산하면 전혀 다른 값이다. accumulator와 output은 FP16/BF16/FP32일 수 있으므로 parameter `.dtype` 하나로 compute precision을 쓰지 않는다.

scale이 `scale_inv`라면 loader가 reciprocal을 언제 취하는지도 본다. zero가 있거나 매우 작은 scale에서 reciprocal 변환은 rounding과 special value를 바꾼다. host scalar로 변환한 뒤 device tensor를 만드는지, 원래 dtype에서 reciprocal하는지 기록한다. direct scale을 기대하는 backend에 inverse scale pointer를 넘기면 output 전체가 일정 배율로 어긋나며 shape 검사는 모두 통과한다.

동적 activation quantization에는 request shape가 들어온다. per-token scale은 M개의 row를 갖지만 graph capture bucket이 M보다 크면 unused row의 scale generation을 초기화해야 한다. 이전 request의 scale이 남아 있고 mask가 잘못되면 stale-but-valid metadata 사건이 된다. 이것은 FP8 format 오차가 아니라 runtime lifetime 오류다. address와 generation을 함께 기록한다.

backend compatibility는 GPU가 FP8 instruction을 갖는다는 한 조건으로 끝나지 않는다. E4M3/E5M2 지원, weight/activation 조합, scale granularity, K/N alignment, bias와 output dtype, toolkit/build와 package가 모두 gate다. unsupported block scale을 scalar로 축약해 실행하는 것은 fallback이 아니다. 동일한 logical weight를 보존하는 explicit conversion 또는 compatible backend가 필요하다.

## 50.6 NVFP4는 data와 두 층의 scale이 하나의 format이다

### 50.6.1 data·block scale·global scale을 한 bundle로 추적한다

NVFP4를 “FP8의 절반 byte”라고 소개하면 가장 중요한 부분이 사라진다. 4-bit floating code 두 개가 한 byte에 들어갈 수 있지만, block별 FP8 scale과 global scale이 함께 있어야 logical value가 복원된다. data byte만 복사해 다른 backend에 넣을 수 없다. block size와 scale axis, scale byte의 encoding, interleave와 global multiplier 방향까지 하나의 ABI다.

작은 fixture는 32개 logical values, 즉 16-value block 두 개를 쓴다. 첫 block code는 반복되는 작은 값, 둘째 block은 다른 pattern으로 두고 block scale도 서로 다르게 둔다. global scale은 1이 아닌 값으로 둔다. 첫 block만 검사하면 stride가 틀려도 통과할 수 있으므로 둘째 block 첫/끝 coordinate와 padding tail을 포함한다.

canonical worksheet에는 `logical i→data byte i/2→low/high nibble→block floor(i/16)→canonical block scale→backend physical scale index→global scale→value`를 쓴다. backend가 block scale을 tensor-core tile에 맞춰 swizzle하면 canonical index와 physical index가 다르다. converter 전후에 inverse identity를 검사하되, inverse helper 자체가 같은 잘못된 permutation을 공유할 위험을 줄이기 위해 몇 좌표는 손으로 계산한다.

FlashInfer current pin의 [`fp4_quantization.py` 1701–1775행](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/quantization/fp4_quantization.py#L1701-L1775)은 batched NVFP4 quantization에서 data, scale factor layout, global scale와 padding 조건을 잇는 고정점이다. API 반환 tensor의 dtype 이름만 기록하지 않고 logical input shape가 각각 어떤 output shape로 바뀌는지 적는다.

fused MoE runtime 경계는 [`fused_moe.py` 778–967행](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/fused_moe/cute_dsl/fused_moe.py#L778-L967)에서 activation/weight data, block scales, expert/top-k, workspace와 output으로 나타난다. quantizer output이 이 runner의 모든 architecture variant와 자동 호환된다고 일반화하지 않는다. SM family와 runner별 scale interleave·alignment validation을 확인한다.

NVFP4 incident에서 첫 block은 맞고 둘째부터 틀리면 block stride와 padding을 의심한다. 모든 값이 같은 배율로 틀리면 global scale 또는 direct/inverse convention을 본다. 인접 expert의 scale이 섞이면 expert axis와 interleave를 본다. 값의 pattern이 원인 후보를 좁히도록 fixture를 설계한다.

## 50.7 loader가 128-value 블록의 소유권을 만든다

vLLM의 post-load repack은 representation이 바뀌는 명확한 경계다. [`marlin_utils.py` 465–549행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L465-L549)은 GPTQ/AWQ qweight repack, scale permutation과 zero-point conversion을 연결한다. qweight만 변환됐다는 식으로 읽지 않고 qweight·scale·ZP·g_idx가 같은 logical coordinate를 유지하는지 본다.

runtime call 경계인 [`marlin_utils.py` 698–765행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L698-L765)에서는 input reshape/pad, optional activation quantization, weight/scales/ZP/g_idx/workspace와 exact M/N/K가 만난다. 이 장은 kernel 내부를 다시 열지 않는다. loader가 만든 parameter shape와 이 call의 argument 의미가 같은지만 잇는다.

SGLang에서는 같은 Marlin 이름이 보여도 vLLM parameter object를 그대로 공유한다고 가정하지 않는다. [`moe_wna16_marlin.py` 42–130행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/ops/moe/moe_wna16_marlin.py#L42-L130)은 packed expert weight, scale/ZP, act-order, workspace와 sorted token/expert metadata를 wrapper ABI에서 잇는다. expert axis가 추가됐으므로 dense linear의 `[K,N]` scale permutation을 앞에 expert dimension만 붙여 재사용할 수 있는지 source shape 식으로 확인한다.

NVFP4/MXFP4 MoE 정책은 [`overrides.py` 1270–1305행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/arg_groups/overrides.py#L1270-L1305)에서 device와 format 조건에 따라 backend를 조정한다. 이 override는 import 성공이 아니라 effective backend state를 바꾸는 정책이다. 그러나 모든 linear와 모든 SM에 적용되는 일반 규칙은 아니다. user option, model format, device capability와 package availability가 어느 순서로 우선하는지 기록한다.

Transformers는 config와 quantizer lifecycle의 시작점이다. `quantization_config.py`에서 GPTQ/AWQ config constructor가 bits, group size, desc_act, sym, version/backend를 deserialize하고 어떤 invalid 조합을 거절하는지 읽는다. 이어 quantizer registry가 method를 고르고, model preprocess가 dense parameter를 quantized placeholder로 교체하며, state-dict loader가 source tensor 이름을 소비하는 순서를 잇는다. config가 serialize되는 이름과 serving engine이 인식하는 alias도 비교한다.

중요한 것은 Transformers가 format을 이해한다는 사실과 vLLM/SGLang에서 해당 backend가 실행 가능하다는 사실의 분리다. Transformers quantizer가 만든 module class나 parameter name을 serving engine이 자체 model loader에서 다시 매핑할 수 있다. 따라서 “Transformers에서 로드됨”을 runtime ABI 검증으로 쓰지 않는다. snapshot config hash, effective quant method class와 최종 parameter inventory를 각각 보존한다.

llama.cpp의 GGUF quant block은 비교 기준을 더 선명하게 만든다. [`ggml-quants.h` 1–240행](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-quants.h#L1-L115)과 type traits에서 Q4_0, Q4_1, Q4_K, IQ와 MXFP4의 block elements, struct bytes, scale/min/high-bit field를 읽는다. 이름에 Q4가 있어도 GPTQ qweight와 word order, block scale와 zero convention이 다르다. 평균 bits가 binary ABI를 만들지 않는다.

GGUF loader는 tensor type을 유지한 채 backend buffer에 둘 수 있고, 지원하지 않는 op/type 조합에서는 다른 CUDA path, dequantized temporary 또는 CPU backend를 선택할 수 있다. 어떤 경로인지는 graph/backend selection으로 확인한다. Q4 tensor가 GPU memory에 있다는 사실만으로 quantized CUDA kernel이 선택됐다고 쓰지 않는다.

FlashInfer는 앞 절의 quantizer와 runner 사이 ownership을 특히 명확히 기록한다. quantizer가 반환한 packed data와 scale tensor를 누가 보관하고, runner 생성 시 어떤 architecture·alignment를 고정하며, call마다 activation data/global scale이 어떻게 전달되는지 따라간다. wrapper의 dtype/shape validation이 통과해도 scale byte의 semantic interleave가 맞다는 증명은 아니다. canonical fixture를 runner 직전 physical layout까지 변환해 비교한다.

## 50.8 conversion이 포맷을 바꾸고 generation을 commit한다

checkpoint-native parameter는 아직 특정 고속 kernel의 소유물이 아니다. loader가 schema와 tensor inventory를 검증하고 TP/EP slice를 만든 뒤 backend compatibility를 판정한다. conversion이 필요하면 qweight만 아니라 scale, ZP와 ordering metadata를 함께 새 representation으로 옮긴다. 이 순간 이후 parameter의 개념적 tag는 `checkpoint_native`에서 `marlin_repacked` 또는 `nvfp4_interleaved`로 바뀐다.

실제 code에 tag field가 없더라도 owning method object, parameter class와 post-load lifecycle이 tag 역할을 한다. 문제는 backend selection이 나중에 바뀌는 경우다. Marlin conversion까지 끝난 pointer를 generic GPTQ fallback이 checkpoint-native로 해석하면 launch와 shape가 성공하면서 wrong answer가 난다. fallback은 representation-compatible consumer를 선택하거나 inverse conversion을 명시적으로 수행해야 한다.

지원 gate를 한 boolean으로 합치지 않는다. bits/group/ZP/act-order, K/N divisibility, activation dtype, scale mode, device SM, toolkit/build, optional package와 workspace가 각각 reason을 가진다. model-level config가 지원돼도 특정 layer의 shard shape가 거절될 수 있다. startup log에는 model quant method와 layer별 effective method·fallback reason을 분리한다.

fallback이 correctness를 보존해도 비용은 달라진다. generic kernel은 repack startup을 피하지만 request마다 dequant 비용이 클 수 있다. dense dequant fallback은 peak memory를 늘리고 CPU fallback은 transfer와 scheduler blocking을 만든다. 이 장은 kernel 성능을 반복하지 않고 선택 reason과 conversion·memory lifetime이 관측되는지만 확인한다.

## 50.9 네 사건에서 first divergence를 찾는다

### 50.9.1 scale axis는 shape 검사를 통과했다

checkpoint weight는 `[N,K]`, model parameter는 `[K,N]`, TP는 N축을 둘로 나눴다. scale은 `[N]`이었다. loader는 weight transpose 뒤 scale을 `[1,N]`이 아니라 `[N,1]`로 reshape했다. square test에서는 broadcast가 가능했지만 실제 K≠N projection에서 channel pattern이 틀렸다.

source coordinate `(n,k)`의 scale `s[n]`은 transpose 뒤 logical `(k,n)`에서도 같은 n을 가리켜야 한다. TP rank는 N range만 잘라 local scale을 얻는다. checkpoint→reshape→shard→runtime stride→first dequant coordinate를 비교한다. 첫 divergence가 reshape이면 backend counter는 필요 없다.

경쟁 가설은 FP8 saturation, reduction order와 stale activation scale이었다. 작은 code로 saturation을 없애고 first dequantized W에서 이미 정수배 차이가 나 reduction을 기각했다. static weight scale에서도 차이가 나 stale 가설도 낮췄다. 수정은 semantic axis를 metadata로 보존하고 adapter가 기대 shape로만 view하는 것이었다.

회귀 fixture는 non-square, TP 1/2/4, transpose view, scalar·channel·block scale을 가진다. unsupported scale mode는 compatible fallback 또는 명시적 error여야 한다. 자동 squeeze로 우연히 broadcast되는 path를 금지한다.

### 50.9.2 nibble 순서 하나가 histogram을 속였다

AWQ qweight와 Marlin repack은 성공했지만 첫 layer output이 틀렸다. converter가 checkpoint word의 high-nibble-first를 low-nibble-first로 풀었다. `0x12`가 logical `[1,2]`가 아니라 `[2,1]`이 됐다. 인접 weights가 교환돼 histogram과 byte count는 보존됐다.

`0..F` fixture로 file bytes, framework int32, native unpack sequence를 기록한다. host byte endian과 word 내부 shift를 분리하고 그다음 repack inverse를 비교한다. native unpack에서 처음 틀렸으므로 tile과 MMA는 범위 밖이다. int32 word가 기대와 같아 file byte swap 가설을 기각했다.

zero-point를 모두 0으로 둬도 code 위치가 교환돼 AWQ bias 가설도 기각했다. 수정은 format version별 shift 식을 명시하고 알 수 없는 version을 추측하지 않게 하는 것이었다. 종료 fixture는 word 첫/끝 nibble, 여러 words, signed/bias code, pack axis, sub-shard offset과 padding을 포함한다.

### 50.9.3 act-order metadata가 조용히 사라졌다

`desc_act=true` GPTQ model이 generic path에서는 맞고 optimized path에서만 틀렸다. qweight와 scales는 있었지만 `g_idx`가 optional parameter로 분류돼 loaded inventory에서 빠졌다. optimized method는 identity group order를 만들었다.

K=8 비단조 g_idx와 서로 다른 group scale로 first dequant coordinate를 찾는다. native reference와 repack inverse는 맞고 runtime method의 g_idx만 identity였다. scale permutation 문제라면 sorted scale 자체가 달라야 하므로 기각했다. basis activation의 이동 방향으로 double permutation도 구별했다.

수정은 desc_act config, g_idx consume와 backend capability를 하나의 validation으로 묶는 것이다. g_idx가 없을 때 identity default는 act-order off가 보장될 때만 허용한다. 지원하지 않는 backend는 generic method로 남는다. cache key에도 quant config와 shard shape가 들어가야 한다.

### 50.9.4 NVFP4 scale가 다른 runner의 언어였다

packed data와 global scale은 맞았지만 expert 1부터 output이 틀렸다. loader는 canonical `[expert,K-block,N]` scale을 backend A용으로 interleave했다. runtime policy가 backend B로 fallback하면서 같은 tensor를 넘겼다. dtype와 elements가 같아 shape validation은 통과했다.

두 expert와 두 block에 다른 pattern을 주고 canonical coordinate, A index와 B index를 손으로 계산한다. expert 0 첫 block은 우연히 같았지만 expert 1/block 1에서 first divergence가 났다. FP4 unpack과 global multiplier는 맞아 두 후보를 기각했다.

수정은 backend 결정 뒤 conversion하거나 representation별 copy를 cache하는 것이다. cache key에는 backend family, SM/layout version, shape와 scale convention을 넣는다. 종료 fixture는 expert/block 경계, padded N/K, global scale 1이 아닌 값, A/B selection과 fallback을 포함한다.

## 50.10 loader mutation을 한 coordinate로 추적한다

실전에서는 tensor 전체를 눈으로 비교하지 않는다. 먼저 source weight에서 식별 가능한 coordinate 세 개를 고른다. pack word의 첫 값, group 경계 직후 값, TP shard와 tile tail이 만나는 값을 택한다. 각 값에 q code, ZP, scale, g_idx와 expected dequant를 붙인다. loader 함수가 끝날 때마다 이 tuple이 어느 physical index로 이동했는지 기록한다.

첫 경계는 parameter 생성이다. config가 bits=4, group=128이라고 해도 destination shape가 dense `[K,N]`인지 packed `[K/8,N]`인지 class마다 다르다. parameter의 output/input dimension attribute와 weight loader callback을 읽는다. model architecture가 QKV나 gate/up을 한 destination에 쌓으면 source name 하나가 destination slice 일부만 소유한다. loaded-name set만으로 coverage를 증명하지 못한다.

둘째는 shard다. global logical range를 먼저 계산하고 pack factor를 적용해 storage range로 바꾼다. 반대로 packed offset을 먼저 정수 나눗셈하면 word 중간 경계에서 code를 잃는다. scale은 qweight와 pack factor가 다르므로 같은 byte offset을 재사용하지 않는다. ZP와 g_idx도 자신의 logical axis에서 slice한다. expert parallel은 expert name을 건너뛰는 것과 expert tensor 내부 axis를 자르는 것을 구별한다.

셋째는 post-load mutation이다. repack 함수가 in-place처럼 보이더라도 새 tensor를 parameter data에 대입하거나 auxiliary permutation을 삭제할 수 있다. mutation 전후 pointer, shape, stride, dtype, owner method와 representation을 기록한다. source checkpoint tensor와 converted tensor가 동시에 살아 conversion peak를 만드는 구간도 표시한다. conversion 실패 뒤 반쯤 바뀐 parameter로 fallback하지 않도록 atomic publish 또는 cleanup을 확인한다.

넷째는 runner binding이다. runtime object가 parameter reference를 언제 잡는지 본다. runner가 conversion 전에 pointer를 cache하면 parameter data 교체 뒤 stale tensor를 볼 수 있다. graph capture는 주소를 더 오래 고정한다. weight는 immutable하더라도 backend 재선택, adapter load와 lazy conversion이 주소 generation을 바꿀 수 있다. capture key에 representation generation이 포함되는지 확인한다.

다섯째는 serialization과 reload다. runtime-repacked tensor를 원래 checkpoint field 이름으로 저장하면 다음 load가 두 번 repack할 수 있다. save path가 canonical checkpoint representation을 보존하는지, converted cache를 제외하는지 본다. `quant_method` version과 tensor names가 서로 다른 schema revision에서 섞이지 않게 snapshot manifest를 둔다.

이 추적은 framework마다 함수 이름이 달라도 유지된다. Transformers에서는 config/quantizer가 placeholder를 만들고 state dict conversion이 채운다. vLLM에서는 quant method parameter와 model-specific loader, post-load processing이 역할을 나눈다. SGLang에서는 loader와 JIT/kernel wrapper 사이에 backend override가 끼어들 수 있다. llama.cpp는 GGUF block-native tensor type과 backend buffer/type traits가 경계를 만든다. FlashInfer는 caller가 만든 packs와 runner-specific validation/interleave가 만난다.

관측표에는 source revision과 함수 line만 적지 않는다. 각 line이 보장하는 것을 문장으로 쓴다. config constructor는 field validation을 보장하지만 file tensor contents를 보장하지 않는다. weight loader는 destination slice copy를 보장하지만 runtime backend 선택을 보장하지 않을 수 있다. repack helper는 layout 변환을 보장해도 다른 runner 호환을 보장하지 않는다. call wrapper는 shape를 검사해도 scale semantic axis를 알지 못할 수 있다.

작은 fixture는 이 모든 경계를 통과해야 한다. 실제 model file을 실행할 필요 없이 정적 source에서 expected shape 식과 index mapping을 손으로 계산할 수 있다. 본문 숫자는 실행 측정값이 아니라 설명용 fixture임을 밝힌다. runtime 성능 수치나 특정 GPU 결과를 만들어 내지 않는다.

### 50.10.1 Transformers에서 시작된 이름이 serving parameter가 되기까지

Transformers config를 읽을 때 가장 먼저 볼 값은 class 이름이 아니라 serialized dictionary다. 같은 GPTQ라는 label 아래에도 bits, group size, desc_act, sym과 backend/version hint가 있다. constructor가 default를 채우면 원본 JSON에 없던 값이 effective state에 생긴다. incident manifest에는 raw config와 normalized config를 둘 다 남겨야 “사용자가 쓰지 않은 옵션”이 어디서 왔는지 설명할 수 있다.

quantizer registry는 이 normalized state로 implementation을 고른다. environment validation은 package와 device를 확인할 수 있지만 checkpoint tensor가 그 schema와 맞는지까지 보장하지 않는다. preprocess가 dense linear를 quantized module로 교체할 때 destination parameter names와 physical shapes가 생긴다. 이때 source file을 아직 읽지 않았다면 placeholder shape는 config와 model dimension만으로 계산된다. 잘못된 group size가 file load 전에 이미 잘못된 scale shape를 만들 수 있다.

state dict load에서는 source names의 consume count를 기록한다. `qweight`, `qzeros`, `scales`, `g_idx`가 각각 정확히 한 semantic destination에 들어가는지, stacked QKV/gate-up에서는 destination slice coverage가 겹치거나 비지 않는지 본다. optional이라는 표시는 “없어도 언제나 안전”이 아니라 sym 또는 act-order 같은 predicate가 참일 때 default가 정의된다는 뜻이다.

Transformers에서 다시 저장할 수 있는 config와 serving runtime의 converted parameter를 구분한다. runtime-specific interleave를 canonical checkpoint로 오인해 save하면 portability가 깨진다. serialization method가 어떤 representation을 내보내는지 모르면 converted cache를 저장 대상에서 제외한다.

### 50.10.2 vLLM의 method 선택과 post-load 소유권

vLLM은 model-level quant string으로 candidate config를 만들지만 layer마다 적용 가능한 method가 달라질 수 있다. embedding, LM head, dense linear와 MoE expert는 parameter class와 supported backend가 다르다. 조사표에는 model quant method 하나만 쓰지 않고 layer qualified name, global/local K/N, parameter method와 effective backend를 쓴다.

parameter loader에는 logical axis와 packed axis에 관한 attribute가 붙을 수 있다. TP shard callback이 이 정보를 사용해 source tensor 일부를 destination에 복사한다. QKV처럼 output dimension이 여러 projection으로 쌓이면 rank slice 전에 component offset이 계산될 수 있다. pack factor를 어느 단계에서 나누는지 source 식을 따라간다. scale/ZP에는 qweight와 다른 packed dimension rule이 있을 수 있다.

post-load repack 뒤에는 원래 parameter와 auxiliary metadata가 교체되거나 재배열된다. 이 시점에서 generic format 검사 결과를 그대로 사용하면 안 된다. converted shape, scalar type, sorted g_idx와 workspace requirement로 validation을 다시 연결한다. method가 own하는 representation과 apply call이 기대하는 representation이 일치해야 한다.

fallback은 conversion 이전에 결정하는 편이 단순하지만 모든 gate를 load 전에 알 수 있는 것은 아니다. per-rank shape, device capability와 package build가 늦게 드러날 수 있다. 늦은 fallback을 허용한다면 canonical copy 유지, reversible conversion 또는 alternative-compatible converted copy 가운데 정책이 필요하다. 메모리 비용 때문에 canonical을 버렸다면 incompatible fallback은 명시적으로 실패해야 한다.

### 50.10.3 SGLang override와 loader 사이의 시간 순서

SGLang의 backend override는 user option을 실제 state로 바꾸는 policy다. 중요한 질문은 override가 parameter 생성 전인지 후인지다. 전이면 loader가 처음부터 target layout을 만들 수 있다. 후이면 이미 만들어진 representation을 target backend에 맞게 변환하는 경계가 필요하다. option 문자열이 바뀐 것만으로 tensor layout은 바뀌지 않는다.

MoE에서는 expert axis와 routing metadata가 추가된다. checkpoint expert id, EP local expert, packed weight의 첫 axis와 runner의 expert index가 같은 mapping을 가져야 한다. scale/ZP도 expert별 slice를 따라야 한다. non-local expert를 file iterator에서 skip했는지, tensor materialize 뒤 버렸는지는 memory/IO 차이를 만들지만 correctness invariant는 local expert coverage다.

JIT wrapper key에 dtype, expert parallel flag와 bias가 들어가도 quant representation version이 포함되는지 확인한다. 같은 shape/dtype의 다른 interleave가 하나의 compiled/cache key를 공유하면 validation을 통과할 수 있다. runner object와 converted parameter cache의 key를 함께 비교한다.

### 50.10.4 llama.cpp와 FlashInfer가 보여 주는 비호환성의 형태

llama.cpp GGUF block은 struct 안 scale/min/high bits의 배치까지 type identity다. loader가 `GGML_TYPE_Q4_0`을 유지한다는 것은 “일반 4-bit”를 유지한다는 뜻이 아니라 Q4_0 block contract를 유지한다는 뜻이다. CUDA backend가 특정 operation에서 그 type을 지원하지 않으면 다른 path를 골라야지 GPTQ kernel에 pointer를 넘길 수 없다.

FlashInfer NVFP4도 마찬가지다. public API의 packed uint8와 scale tensor shape는 입구 계약이고, runner-specific interleave와 architecture alignment가 다음 계약이다. upstream framework가 이미 interleave했다면 FlashInfer helper가 다시 변환하지 않는지, canonical input을 기대한다면 어느 함수가 변환하는지 ownership을 하나로 정한다. “둘 다 NVFP4”라는 이유로 conversion 책임이 사라지지 않는다.

## 50.11 재현 fixture와 증거 장부를 운영 장애에 연결한다

실제 모델은 너무 커서 첫 조사 대상으로 부적합하다. 그렇다고 임의의 작은 random matrix로 바꾸면 group, shard와 pack boundary가 사라진다. 축소 fixture는 오류를 만든 semantic boundary를 보존해야 한다. scale-axis 사건이면 K와 N을 다르게 하고 scale마다 다른 값을 둔다. packing이면 word 전체의 연속 code와 다음 word 첫 code가 필요하다. act-order이면 비단조 g_idx, NVFP4이면 expert·block을 최소 두 개씩 둔다.

fixture header에는 source revision, quant config raw/normalized form, logical layer shape, TP/EP rank, tensor names/shapes/dtypes와 selected backend를 적는다. 첫 표는 format inventory다. qweight의 pack axis/factor, storage word와 bit shift, scale의 granularity/axis/direct-or-inverse, ZP bias, g_idx domain을 기록한다.

둘째 표는 mutation ledger다. 함수/line, 입력 representation, logical/physical range, 출력 representation, 새 owner와 validation을 쓴다. view shape/stride 변경도 mutation이다. qweight가 repack될 때 scale·ZP가 어느 함수에서 함께 변하는지 같은 transaction으로 묶는다. conversion failure 뒤 반쯤 바뀐 parameter가 publish되지 않는지도 본다.

셋째 표는 compatibility다. candidate마다 bits/group/ZP/act-order, activation dtype, scale mode, alignment, SM/build/package gate와 reason을 쓴다. false인데 선택됐다면 policy bug다. fallback이면 현재 representation을 읽을 수 있는지 다음 행에서 검증한다.

넷째 표는 native unpack, native dequant, shard merge, repack inverse, runner 직전 dequant와 output이다. checkpoint-native부터 틀리면 schema/unpack, native는 맞고 shard가 틀리면 packed-coordinate loader, shard는 맞고 inverse가 틀리면 converter, 모두 맞고 runner 해석이 틀리면 ABI adapter다. output에서만 다르면 그제야 46장의 kernel numerical path를 연다.

### 50.11.1 한 변수씩 바꾸는 반증 순서

backend off로 정상화됐다고 kernel이 범인인 것은 아니다. off는 repack, parameter class, padding과 activation quantization을 함께 바꾼다. 같은 source에서 generic native dequant와 optimized inverse-repack dequant를 먼저 비교한다.

TP=1에서 맞고 TP=2에서 틀리면 각 rank logical slice를 합쳐 global reference와 비교한다. qweight뿐 아니라 scale/ZP/g_idx coverage도 합친다. local output에서 이미 틀리면 collective를 기각한다. rank 경계가 group 또는 pack word를 자르는지 손으로 계산한다.

sym=true에서 맞고 false에서 틀리면 ZP 후보가 강하지만 backend가 같이 바뀌었는지 확인한다. q와 scale을 고정하고 ZP만 구별 pattern으로 바꿔 `q-z`, `q+z`, implicit bias와 off-by-one을 구별한다.

act-order off/on 비교에는 identity g_idx와 비단조 g_idx를 모두 둔다. identity에서도 틀리면 flag에 따른 layout/dispatch, 비단조에서만 틀리면 permutation propagation을 본다. global scale은 1이 아닌 값을 둔다. 모든 output이 일정 배율이면 global scale, block마다 다르면 block index, channel마다 다르면 axis를 우선한다.

### 50.11.2 재발 테스트는 semantic contract를 고정한다

함수 호출 횟수보다 source hex→logical codes golden vector를 고정한다. backend converter는 inverse했을 때 같은 logical q/scale/ZP/g_idx tuple을 내야 한다. shard test는 global→local→merge identity를 K/N/expert 축과 group/word/tile 경계에서 본다.

fallback test는 unsupported group, misaligned N, missing package, 다른 SM과 scale mismatch를 하나씩 만든다. 정확한 reason과 representation-compatible path가 필요하다. fallback이 없으면 launch 전에 실패해야 한다. save/reload는 runtime interleave를 canonical checkpoint로 저장하지 않고 두 번 load해도 repack이 누적되지 않음을 확인한다.

## 50.12 같은 128-value 블록을 네 consumer까지 비교한다

가상의 gate projection을 조사해 보자. global logical weight는 K=12, N=16이고 TP=2 column parallel이라 rank마다 N=8을 갖는다. checkpoint는 GPTQ 4-bit, group size 4, asymmetric ZP, act-order on이다. 숫자는 설명용이며 실행 결과가 아니다.

### 50.12.1 config와 source coordinate를 고정한다

raw config와 normalized config를 먼저 나란히 둔다. raw에는 `bits=4`, `group_size=4`, `sym=false`, `desc_act=true`가 있다. normalized method가 backend version default를 추가했다면 그 출처를 기록한다. model hidden/intermediate shape로 source qweight/scales/qzeros/g_idx의 expected logical·physical shape를 계산한다. file header shape와 다르면 load 전에 schema mismatch다.

source coordinate로 `(k=4,n=9)`를 고른다. rank 1의 local n=1이고 K group은 g_idx[4]가 결정한다. q code가 어느 int32 word·shift에 있는지 checkpoint format 식으로 찾는다. ZP와 scale도 global `(group,n=9)`에서 읽는다. native dequant reference를 손으로 계산한다.

loader가 column shard를 만들 때 qweight의 N packed axis 범위를 자르고, scale/ZP도 N=8..15를 자른다. g_idx는 K domain이므로 두 rank 모두 필요한 local copy를 갖는다. scale offset에 qweight pack factor를 적용하면 틀린다. 각 tensor가 독립적인 physical coordinate 식을 가진다.

### 50.12.2 shard와 act-order 뒤 같은 값을 찾는다

act-order sort는 K permutation P를 만든다. qweight K columns가 P 순서로 repack되면 scale group lookup과 activation read가 같은 mapping을 따라야 한다. coordinate k=4가 repack tile의 어디로 갔는지, runtime에서 activation original k=4가 어떻게 그 위치와 곱해지는지 적는다. P와 inverse P의 이름만 보고 방향을 추정하지 않고 source gather/scatter 식을 본다.

Marlin conversion이 qweight tile을 바꾸고 scale/ZP를 permute한 뒤 parameter owner가 optimized method로 바뀐다. 이때 checkpoint-native qweight pointer는 runtime argument가 아니다. apply call에는 converted weight, converted scale/ZP, sorted g_idx와 workspace가 들어간다. M/N/K는 local physical padding과 logical output unpadding을 구별한다.

### 50.12.3 변환 뒤 consumer와 fallback의 표현을 검산한다

지원 gate에서 rank N=8이 tile requirement를 만족하지 못한다고 하자. 중요한 것은 언제 알았는가다. conversion 전에 알았다면 generic native representation을 유지할 수 있다. conversion 뒤 알았다면 canonical copy 또는 inverse가 필요하다. repacked pointer를 generic fallback에 넘기면 모든 dtype/shape가 맞아도 coordinate가 틀린다.

fallback이 generic GPTQ를 골랐다면 같은 `(4,9)` coordinate가 native word/shift와 native scale/ZP/g_idx로 복원되는지 확인한다. optimized와 generic output을 먼저 비교하지 않는다. 두 path의 first dequant weight가 같은 logical quantized value인지 비교한다. 그 뒤에 각 kernel의 허용 numerical tolerance를 적용한다.

이 walkthrough를 FP8으로 바꾸면 qweight nibble/ZP/g_idx 대신 format code와 scale broadcast가 중심이 된다. TP column rank는 per-channel weight scale을 자르지만 per-tensor scale은 공유한다. dynamic per-token activation scale은 request M에 따라 새로 생긴다. graph bucket padding이 있다면 active rows와 generation을 runtime ledger에 추가한다.

NVFP4로 바꾸면 logical coordinate는 data nibble뿐 아니라 K/N block, block-scale canonical index와 backend physical interleave를 가진다. global scale은 rank에 공유되거나 format 정의에 따라 별도 parameter가 된다. backend 변경은 qweight pointer 하나가 아니라 data+block scale+global scale bundle의 representation을 바꾼다.

llama.cpp GGUF Q4로 바꾸면 config의 GPTQ field를 억지로 대응시키지 않는다. GGUF tensor type trait에서 block size와 struct layout을 새로 시작한다. 동일한 `(k,n)` 추적법은 유지하되 word/scale coordinate 식은 Q4_0/Q4_K 등 정확한 type에서 가져온다. 조사 방법은 이식되지만 ABI는 이식되지 않는다.

### 50.12.4 로그와 metric이 보여 줄 수 있는 범위

운영 로그에는 model snapshot, quant method, layer/backend/fallback category, conversion cache hit/miss와 error reason을 bounded cardinality로 남길 수 있다. exact tensor 이름과 shape는 sampled diagnostic log로 연결한다. pointer와 weight 값은 보안상 직접 노출하지 않고 allocation-relative id, representation generation과 digest를 사용한다.

“quantized model loaded”라는 log 하나는 부족하다. schema parse 완료, source tensors consumed, conversion publish와 backend binding은 서로 다른 completion이다. startup crash가 어느 단계인지 알아야 partial parameter cleanup과 retry를 판단한다. conversion cache hit도 cache key와 representation version이 맞다는 validation 뒤에만 성공으로 센다.

성능 metric은 correctness를 증명하지 않는다. repack 시간이 줄고 kernel throughput이 좋아도 scale axis가 틀릴 수 있다. 반대로 fallback 비율 증가는 correctness를 보존한 안전 조치일 수 있다. effective backend와 reason을 latency와 연결하되 wrong-answer fixture를 별도 gate로 둔다.

### 50.12.5 수정 승인에 필요한 네 종류의 증거

첫째는 schema 증거다. raw/normalized config와 tensor inventory가 기대 format version에 맞는다. 둘째는 representation 증거다. golden coordinates가 native unpack, shard와 inverse conversion에서 보존된다. 셋째는 runtime ABI 증거다. selected backend가 scale/ZP/g_idx bundle의 shape·stride·convention을 정확히 받는다. 넷째는 serving 증거다. reload, graph capture, cancellation, adapter/backend 전환과 fallback에서도 generation과 owner가 보존된다.

네 증거가 모두 있어야 “모델이 잘 로드됐다”고 말할 수 있다. file checksum만으로는 schema interpretation을, output tolerance 하나로는 이웃 coordinate corruption을, backend log만으로는 실제 parameter layout을 증명하지 못한다. 증거마다 주장 범위를 넘기지 않는다.

### 50.12.6 packed shape와 scale byte를 손으로 검산하는 두 번째 fixture

첫 fixture보다 조금 현실적인 `K=8,N=6`, group size 4, unsigned INT4 asymmetric weight를 생각하자. qweight가 K축 codes를 int32 word 하나에 여덟 개씩 묶는 convention이라면 logical 48 codes의 ideal payload는 24 bytes이고 physical qweight shape는 `[K/8=1,N=6]` int32, 즉 24 bytes다. 여기서 qweight가 N축을 pack하는 다른 schema라면 shape는 달라진다. bits와 logical shape만으로 pack axis를 추정하지 않고 parameter loader의 packed dimension을 확인한다.

scale과 ZP logical shape는 K group 두 개와 N channel 여섯 개, 즉 `[2,6]`이다. scale이 FP16이면 24 bytes다. ZP도 4-bit로 pack하고 N축 여섯 값을 word alignment 때문에 여덟 slot으로 padding한다면 실제 byte는 ideal `12×0.5=6`보다 클 수 있다. struct/container와 alignment를 source에서 계산한다. “4-bit weight는 BF16의 1/4”는 qweight payload 비율이지 전체 parameter byte가 아니다.

첫 group의 N=0..5 scale을 `[1,2,3,4,5,6]`, 둘째를 `[10,20,30,40,50,60]`으로 둔다. ZP도 channel마다 다르게 둔다. q code는 k와 n을 동시에 식별하도록 `q[k,n]=(3k+n) mod 16` 같은 설명용 식을 쓴다. `(k=3,n=5)`와 `(k=4,n=0)`은 group 경계 양쪽이다. 두 coordinate의 word/shift, group, scale/ZP와 dequant를 계산한다.

TP=2 column shard를 적용하면 rank 0은 N=0..2, rank 1은 N=3..5를 갖는다. N=3은 qweight physical column 3, scale physical column 3과 ZP logical column 3에서 local column 0으로 이동한다. qweight K-pack word는 그대로다. scale을 K group 축으로 잘못 shard하면 shape가 `[1,6]`처럼 그럴듯할 수 있지만 rank가 output channel 전체를 들고 잘못된 group 일부만 갖는다.

N=6이 backend tile을 만족하지 않아 N=8로 padding될 수도 있다. padded q code를 0으로 두는 것만으로 zero contribution이 보장되지 않는다. asymmetric 식에서 `q-z=0`이 되도록 code와 ZP convention이 맞아야 한다. padded scale은 finite해야 하고 output은 original N=6으로 잘라야 한다. padding channels가 다음 stacked projection의 storage와 겹치지 않는지도 destination coverage로 본다.

act-order를 켜고 P가 `[2,3,0,1,6,7,4,5]`라면 original k=0은 sorted position 2로 간다. 여기서 `P[sorted]=original`인지 `P[original]=sorted`인지 이름으로 알 수 없다. gather 식 `W_sorted[:,i]=W[:,P[i]]` 같은 실제 source expression을 적는다. activation gather도 같은 original coordinate가 sorted weight와 만나도록 방향을 확인한다.

repack 뒤 physical tile index는 native word와 달라진다. 그러나 inverse했을 때 `(k=4,n=3)`의 q/scale/ZP/g_idx tuple이 동일해야 한다. tuple이 맞고 raw bytes가 다른 것은 정상이다. raw byte equality를 repack correctness 기준으로 쓰지 않는다. 반대로 output이 비슷해도 tuple이 다르면 symmetric input의 우연일 수 있어 실패다.

FP8 fixture는 동일 K/N을 유지하되 qweight pack을 없애고 per-channel scale `[1,2,3,4,5,6]`과 per-token activation scale을 둔다. TP column shard는 weight scale을 자르지만 activation scale은 M row 기준으로 두 rank가 공유한다. scale 두 종류를 같은 shard callback에 넣으면 activation scale을 N축으로 잘라 버릴 수 있다.

block FP8이라면 block coordinate를 추가한다. K block 4, N block 2라면 weight scale logical shape는 `[ceil(8/4)=2,ceil(6/2)=3]`이다. N padding 8이면 physical scale 열이 4개가 될 수 있다. logical/physical scale shape를 구분하고 padded block이 original N output에 기여하지 않는지 본다.

NVFP4 fixture에서는 48 FP4 codes가 ideal 24 bytes이고 K block 16이라면 K=8 자체가 block padding을 요구한다. 실제 parameter는 K를 16으로 늘려 48 additional padded codes를 가질 수 있다. block scale count도 padded K 기준으로 계산될 수 있다. logical payload 계산과 backend physical allocation을 나란히 두어 memory 차이를 “누수”로 오해하지 않는다.

### 50.12.7 incident 보고서를 실제로 리뷰하는 순서

리뷰어는 “AWQ model output mismatch fixed”라는 제목에서 시작하지 않는다. 첨부된 config가 raw인지 normalized인지, model snapshot과 loader commit이 고정됐는지 확인한다. quant_method version과 tensor inventory가 없으면 같은 증상을 재현해도 같은 format을 시험한 것이 아닐 수 있다.

다음으로 first divergence가 충분히 이른지 본다. 최종 logits만 있으면 embedding부터 LM head까지 모두 후보로 남는다. quant layer input을 고정하고 native logical weight, post-shard weight, inverse-repack weight와 layer output을 비교해야 한다. 첫 divergence가 weight tuple인데 kernel profiler screenshot만 붙어 있다면 조사 경계가 잘못됐다.

경쟁 가설은 최소 두 개 이상 실제 관측으로 기각해야 한다. scale-axis 주장이라면 saturation을 피한 code와 non-square fixture가 있어야 한다. nibble 주장이라면 endian과 ZP convention을 분리한 known hex가 있어야 한다. act-order라면 identity와 non-monotonic g_idx 비교가 있어야 한다. NVFP4 interleave라면 data/global scale가 맞고 block physical index가 처음 다른 좌표가 있어야 한다.

patch review에서는 parameter bundle이 atomic하게 바뀌는지 본다. qweight assignment 뒤 scale permutation에서 exception이 나면 half-converted object가 cache나 runner에 publish되지 않아야 한다. retry가 같은 object를 다시 repack하지 않는지 확인한다. conversion state를 inferred shape만으로 판단하면 native와 converted shape가 우연히 같을 때 double conversion이 가능하다.

fallback review는 happy path보다 까다롭다. 지원 gate 실패 reason이 conversion 전후 어느 시점에 나오는지, 현재 owner가 canonical copy를 갖는지 확인한다. generic fallback이라는 함수 이름보다 그 함수가 기대하는 word order와 scale convention을 본다. 안전한 fallback을 만들 수 없으면 explicit error가 silent reinterpretation보다 낫다.

회귀 범위에는 layer family가 들어간다. dense linear에서 고친 shard callback이 stacked QKV, gate/up와 expert weight에도 쓰이는지 찾는다. 같은 helper라도 pack axis attribute와 expert dimension이 달라질 수 있다. 모든 quantized layer를 하나의 representative tensor로 대신하지 않는다.

마지막으로 serving lifetime을 본다. model startup 한 번의 conversion이라도 lazy compile, adapter attach, backend fallback과 graph capture가 runner binding을 늦게 바꿀 수 있다. converted parameter generation과 graph key가 맞는지, reload 중 old runner가 새 parameter를 보지 않는지 확인한다. immutable weight라는 말은 pointer owner와 graph lifetime까지 자동 보장하지 않는다.

### 50.12.8 source claim과 관측 claim을 섞지 않는 검증

source link는 구현이 존재한다는 증거이고, 이번 배포가 그 path를 실행했다는 증거는 아니다. vLLM repack helper가 current tree에 있어도 model config, layer shape와 backend gate가 그것을 선택하지 않을 수 있다. SGLang override source가 있어도 user option과 device 조건이 branch를 통과했는지 확인해야 한다. FlashInfer package를 import할 수 있어도 runner가 다른 backend일 수 있다.

그래서 claim을 세 종류로 나눈다. source claim은 “이 predicate에서 이 parameter를 변환한다”처럼 commit/line으로 고정한다. artifact claim은 installed wheel/build가 그 source와 target architecture를 포함한다는 manifest로 고정한다. execution claim은 effective method, conversion counter, selected runner와 call trace로 고정한다. source link 하나로 세 주장을 대신하지 않는다.

Transformers config class가 field를 받아들인다는 사실도 format tensor를 모두 검증한다는 뜻이 아니다. constructor는 integer range와 enum을 검사할 수 있지만 qweight shape, scale axis와 name inventory는 file을 읽은 뒤에야 안다. 반대로 file header shape가 맞아도 normalized default가 다른 backend version을 선택할 수 있다. validation의 시간과 소유자를 적는다.

vLLM post-load source에서 `replace_parameter` 또는 `.data` mutation이 보이면 old/new tensor lifetime을 읽는다. conversion temporary와 final parameter가 peak에서 겹치는지, failure가 old parameter를 보존하는지, distributed ranks 가운데 하나만 실패했을 때 collective cleanup이 가능한지 확인한다. 이를 runtime 수치로 추측하지 않고 allocation owner와 exception path를 정적으로 추적한다.

SGLang override는 명시 옵션을 무시하는 것처럼 보일 수 있으므로 reason을 독자에게 설명해야 한다. format/device 조합에서 선택한 runner만 ABI를 지원한다면 correctness gate다. 단지 더 빠르다는 경험값이라면 performance policy다. source condition과 error/fallback path를 읽어 둘을 구분한다. user가 강제한 incompatible backend를 silently reinterpret하지 않는다.

llama.cpp type trait는 GGUF block byte 계산의 source of truth다. `type_size/block_size` 평균만으로 tail tensor byte를 계산하지 않고 block count의 ceil/padding과 tensor row alignment를 본다. backend가 native block을 소비하는지 dequant conversion을 삽입하는지 graph node type에서 확인한다. 파일 format과 compute path를 같은 enum 하나로 설명하지 않는다.

FlashInfer wrapper가 `uint8` data와 scale tensor를 검사할 때 shape check가 canonical인지 interleaved physical인지 읽는다. docstring의 logical shape와 internal view/reshape 뒤 shape를 분리한다. validation 직후 int32 view로 바꾸는 path가 있다면 byte length/alignment와 word order가 새 ABI가 된다. uint8 view와 int32 view는 같은 storage를 가리킬 수 있지만 logical indexing은 달라진다.

관측 claim에는 negative evidence도 필요하다. fallback counter가 0이라고 target backend가 실행됐다고 확정하지 않는다. layer가 dense/CPU path로 처음부터 만들어져 fallback event 자체가 없을 수 있다. repack count가 1이어도 모든 layer가 변환됐다는 뜻이 아니다. expected layer inventory와 method별 coverage를 비교한다.

반대로 로그가 없다고 path가 없었다고 결론내리지 않는다. source에 log가 없다면 외부 profiler 이름을 framework metric이라고 쓰지 않고 관측 공백으로 둔다. 최소 계측은 layer name hash, representation owner, conversion generation과 backend reason 정도로 제한한다. weight values와 raw pointer를 운영 로그에 노출하지 않는다.

### 50.12.9 하나의 4×8 텐서를 네 포맷으로 번역하는 좌표 실험

이제 설명을 한 표면 위에 포개 보자. logical weight는 `W[K=4,N=8]`이고 row-major 표기의 첫 행은 `[-4,-3,-2,-1,0,1,2,3]`이다. 나머지 행은 어느 행인지 눈으로 알아볼 수 있도록 각각 4, 8, 12를 더한다. 이 수들은 실제 모델 분포를 흉내 내기 위한 값이 아니다. 축 전치, group 경계, nibble 순서와 scale broadcast를 서로 다른 무늬로 드러내는 식별자다. 실제 포맷의 양자화 결과를 주장하지 않고, 각 포맷에서 어떤 좌표 장부가 필요한지를 비교한다.

GPTQ fixture는 K축 group size를 2로 둔다. group 0과 1의 scale은 N마다 다른 `s[g,n]`을 가지며, ZP도 `z[g,n]`으로 둔다. code는 `q[k,n]`이고 복원 식은 이 checkpoint가 채택한 convention에 따라 `s[g_idx[k],n]·(q[k,n]-z[g_idx[k],n])`이다. `g_idx=[1,1,0,0]`을 일부러 사용하면 물리적 k 순서와 group 번호가 반대로 보인다. `floor(k/2)`를 몰래 사용한 구현은 첫 좌표부터 실패한다.

AWQ fixture도 4-bit unsigned code와 group scale·zero를 쓸 수 있다. 그러나 “GPTQ와 같은 4-bit”라는 이유로 같은 file layout을 가정하지 않는다. AWQ는 salient activation channel을 고려해 weight scaling을 선택한다는 algorithmic 의도를 갖지만, serving 시점의 핵심은 export schema가 정한 packed dimension과 zero convention이다. 이 실험에서는 같은 logical q를 넣더라도 AWQ loader가 기대하는 word packing과 Marlin 변환 입구를 별도 열로 적는다. 값이 같다는 사실과 bytes가 교환 가능하다는 주장은 다르다.

FP8 fixture는 32개의 code 각각을 한 byte E4M3 payload로 둔다. weight scale은 N축 per-channel `sw[n]`, activation scale은 M축 per-token `sx[m]`이다. zero-point는 없다. conceptual output은 FP32 accumulator에 `sx[m]·sw[n]`을 결합한다. checkpoint가 `weight_scale_inv`를 저장했다면 file value는 `1/sw[n]`일 수 있으므로 consumer 앞에서 direct/inverse 방향을 표시한다. 이름을 보고 추측하지 않고 실제 dequant 또는 scaled GEMM 호출식으로 판정한다.

NVFP4 fixture는 같은 32개 logical weight를 16-value block 두 개로 나눈다. 두 code가 data byte 하나를 공유하고, 각 block에는 E4M3 scale 하나가 있으며 전체 tensor에는 FP32 global scale 하나가 있다. zero-point는 없지만 metadata가 단순한 것도 아니다. logical value는 FP4 code decode, block scale decode, global scale 적용을 모두 통과해야 한다. backend용 scale interleave가 들어가면 canonical block 번호와 physical scale byte offset을 함께 기록한다.

| 질문 | GPTQ fixture | AWQ fixture | FP8 fixture | NVFP4 fixture |
|---|---|---|---|---|
| logical code 좌표 | `q[k,n]` | `q[k,n]` | `e4m3[k,n]` | `e2m1[k,n]` |
| data 저장 단위 | 여러 4-bit code를 담은 word | 여러 4-bit code를 담은 word | code당 1 byte | code 2개당 1 byte |
| local scale 좌표 | `s[g_idx[k],n]` | export schema의 `s[group(k),n]` | `sw[n]` 또는 scalar/block | `sb[block(k,n)]` |
| 두 번째 scale | 없음 | 없음 | activation의 `sx[m]` | tensor global `sg` |
| zero 의미 | explicit/implicit ZP convention | explicit packed ZP convention | 없음 | 없음 |
| ordering metadata | optional `g_idx`, sort permutation | pack/repack permutation | layout와 broadcast stride | nibble·block·scale interleave |
| runtime 핵심 인자 | qweight, scales, qzeros, g_idx | qweight, scales, qzeros | data, input scale, weight scale | data, block scales, global scales |
| 가장 위험한 silent mismatch | g_idx나 ZP bias 누락 | ZP pack와 Marlin permutation 불일치 | N축 scale을 K축으로 broadcast | canonical scale을 interleaved scale로 오인 |

표의 “저장 단위”는 평균 bit 수가 아니다. GPTQ/AWQ의 int32 container에는 padding과 pack alignment가 붙을 수 있고, NVFP4에는 data 외 scale bytes가 필요하다. 따라서 `numel×bits/8`만으로 checkpoint 크기나 device allocation을 계산하지 않는다. data bytes, scale bytes, ZP, ordering metadata, padding, converted copy와 conversion temporary를 따로 합친다.

첫 좌표 `(k=0,n=0)`을 추적할 때 GPTQ는 `packed_word`, `shift`, `q`, `g_idx[0]`, `scale[1,0]`, `zero[1,0]`의 여섯 칸이 필요하다. AWQ는 `packed_word`, `shift`, `q`, `group`, `scale`, `packed_zero_word`, `zero_shift`, `zero_bias`가 필요하다. FP8은 `byte`, `decoded_e4m3`, `weight_scale[0]`, 그리고 output을 볼 때 `input_scale[m]`이 필요하다. NVFP4는 `byte`, `nibble`, `decoded_fp4`, `block_id`, `canonical_scale_byte`, `physical_scale_offset`, `global_scale`가 필요하다.

이 비교의 목적은 포맷을 하나의 보편 식으로 뭉개는 것이 아니다. 오히려 공통 조사 틀과 포맷별 의미를 분리한다. 공통 틀은 logical coordinate→physical address→metadata coordinate→dequant value→runtime consumer다. 포맷별 의미는 shift, group mapping, scale 방향, ZP convention과 interleave다. 공통 틀만 재사용하고 index 식은 source revision마다 다시 고정한다.

### 50.12.10 byte offset을 손으로 계산하는 워크시트

GPTQ/AWQ의 예시 container가 int32이고 한 code가 4 bit라면 word 하나에 여덟 code가 들어간다. 그러나 어느 logical axis의 여덟 값인지는 parameter의 packed dimension이 정한다. N-packed라고 가정한 fixture에서 `(k,n)`의 word column은 `floor(n/8)`, lane은 `n mod 8`, shift는 `4·lane`이다. K-packed라면 같은 식의 n 자리에 k가 들어간다. 실제 loader attribute가 어느 dimension을 packed dimension으로 선언하는지 확인하지 않은 식은 증거가 아니다.

첫 row의 code가 `0,1,2,3,4,5,6,7`이고 low bits first라면 word의 수치 표기는 `0x76543210`이다. little-endian memory에서는 낮은 주소부터 bytes `10 32 54 76`으로 보인다. 여기서 memory byte 순서를 뒤집는 오류와 nibble lane을 뒤집는 오류를 구별할 수 있다. 전자는 네 byte 덩어리의 순서가 달라지고, 후자는 각 byte 안의 두 값이 바뀔 수 있다. debugger 화면 한 줄을 보고 두 용어를 섞지 않는다.

packed zero도 같은 pack factor를 쓴다고 단정하지 않는다. qweight와 qzeros의 logical shapes가 다르고, zero에는 group axis가 들어간다. 어떤 schema는 unsigned code가 표현하는 zero에 bias를 더하거나 뺀 값을 저장할 수 있다. 따라서 `qzeros`의 raw nibble 7을 수학식의 ZP 7로 바로 쓰지 않는다. unpack helper의 bit mask 뒤 보정 연산과 kernel dequant 식을 양쪽에서 확인한다.

AWQ의 SGLang parameter 생성은 고정 소스의 [`awq_linear.py` 18–101행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py#L18-L101)에서 qweight, qzeros, scales가 각각 어떤 parameter class와 packed attributes를 받는지 보여 준다. 이 링크가 특정 checkpoint의 실제 값이나 선택된 kernel을 증명하지는 않는다. parameter의 physical shape와 loader slicing 규칙을 읽는 출발점이다.

Transformers의 고정 소스에서 [`quantization_config.py` 618–879행](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/utils/quantization_config.py#L618-L879)은 GPTQConfig와 이를 잇는 AwqConfig의 bits, group size, damp, desc_act, sym, backend/version 계열 상태를 보여 준다. 이것은 config contract다. qweight byte order는 quantizer integration과 외부 producer schema까지 내려가 확인해야 한다. config class line을 packed ABI의 증거로 확대하지 않는다.

FP8에서 `byte_offset=k·N+n`이라는 식은 contiguous row-major payload에만 해당한다. tensor가 `[N,K]`로 저장되거나 tile swizzle을 거치면 바뀐다. 더욱 중요한 것은 scale offset이다. per-tensor라면 0, per-output-channel이라면 n, blockwise라면 `floor(k/BK)·ceil(N/BN)+floor(n/BN)` 같은 좌표를 가진다. 실제 physical scale이 padded 또는 permuted됐다면 canonical block index에서 변환표를 한 단계 더 거친다.

SGLang 고정 소스의 [`marlin_utils_fp8.py` 145–188행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/quantization/marlin_utils_fp8.py#L145-L188)은 `weight_scale`과 `weight_scale_inv` 이름을 받아 scale tensor를 만들고 Marlin용 permutation을 적용해 parameter를 교체하는 경계다. 독자는 이 구간에서 이름, reciprocal 처리 여부, scale shape 변환과 old attribute 삭제를 순서대로 확인해야 한다. 변수명이 `scales`가 됐다고 direct multiplier 의미가 자동 확정되는 것은 아니다. 이후 call consumer까지 이어 읽는다.

NVFP4의 data offset은 두 logical code가 한 byte를 공유하므로 flat logical index `i=k·N+n`에서 `floor(i/2)`가 첫 후보가 된다. nibble shift는 `4·(i mod 2)`다. 하지만 row padding과 tile packing이 있으면 row stride를 먼저 적용한다. block size 16이면 canonical scale id는 `floor(i/16)`이지만, 2차원 block schema는 K/N block 좌표를 따로 가질 수 있다. `i/16`은 fixture의 flat convention일 뿐 모든 NVFP4 파일의 규격이 아니다.

SGLang의 [`utils.py` 599–770행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/quantization/utils.py#L599-L770)은 NVFP4 scale을 pad하고 blockwise interleave하는 helper와 MoE scale shuffle 흐름을 보여 준다. 특히 FlashInfer helper가 input을 flatten한다는 주석 때문에 caller가 사전에 어떤 axis permutation을 만드는지가 중요하다. flat length가 맞다는 사실은 expert, N-block와 K-block 의미가 맞다는 증거가 아니다.

각 포맷 worksheet의 마지막 칸은 byte가 아니라 consumer argument다. pointer만 기록하지 않고 shape, stride, dtype, direct/inverse, canonical/interleaved, generation과 owner를 적는다. pointer 주소가 같아도 view stride나 의미 version이 달라질 수 있고, 주소가 달라도 값과 layout이 동등한 복사본일 수 있다. ABI 판정은 주소 동일성보다 tuple의 의미 보존을 본다.

## 50.13 shape 검사를 통과한 silent wrong-output을 추적한다

### 50.13.1 scale-axis 사고의 시간선

사고는 새 FP8 checkpoint를 배포한 직후 시작됐다. health probe는 통과했고 model load에서 missing key도 없었다. 모든 parameter shape는 expected shape와 같았다. synthetic test는 `K=N=4096`인 projection과 scale 값이 거의 균일한 layer만 검사했다. backend log에는 accelerated path가 찍혔고 latency도 나빠지지 않았다. 그러나 canary의 첫-token top candidate가 reference와 달랐다.

첫 대응자는 FP8 rounding을 의심해 logit tolerance를 넓혔다. 이것은 잘못된 조치였다. rounding이면 오차가 값의 크기와 accumulator 순서에 따라 부드럽게 변할 가능성이 높지만, 실제 오차는 output channel마다 거의 일정한 배율이었다. channel 0은 맞고 channel 1은 약 두 배, channel 2는 절반에 가까웠다. 배율 무늬는 weight scale을 다른 축에서 읽었다는 가설과 잘 맞았다.

두 번째 대응자는 backend를 끄자 결과가 정상화되는 것을 확인했다. 그래도 kernel bug로 결론내리지 않았다. backend off는 scaled GEMM만 바꾸지 않았다. post-load repack, scale reshape, parameter class와 fallback dequant path도 함께 달라졌다. intervention의 변경 집합이 넓으므로 원인 위치는 아직 loader부터 call wrapper까지 열려 있었다.

팀은 실제 layer를 바로 profile하지 않고 `K=3,N=4` fixture를 만들었다. code matrix의 각 column은 같게 두고 scale은 `[1,2,4,8]`로 두었다. 입력은 K basis vectors 세 개였다. checkpoint-native CPU reference는 output channel n에 따라 scale이 바뀌었다. loader 직후 weight code도 같았다. 그러나 runtime parameter의 scale view는 `[4,1]`이 아니라 consumer가 K축으로 읽을 수 있는 stride를 가졌고, wrapper의 reshape가 singleton dimension을 반대편에 놓았다.

shape assertion은 `numel==4`만 확인했기 때문에 통과했다. square production layer에서는 `[4]`, `[1,4]`, `[4,1]`을 broadcast한 결과 shape가 모두 계산 가능했다. backend wrapper는 pointer와 element 수를 받았고 semantic axis enum은 받지 않았다. kernel은 전달된 stride대로 정확히 읽었다. 첫 divergence는 kernel arithmetic이 아니라 loader와 wrapper 사이의 scale-axis 의미 변경이었다.

이 incident에서 증거 경계는 다음과 같이 닫혔다.

| 경계 | 관측 | 판정 |
|---|---|---|
| raw config | per-output-channel weight scale | 정상 |
| checkpoint tensor | 네 식별 scale과 N축 의미 | 정상 |
| file→parameter copy | 값과 numel 보존 | 정상 |
| post-load reshape | singleton dimension 위치 변경 | 최초 의미 불일치 |
| runtime call | semantic axis 없이 shape/stride 전달 | 오류 전파 |
| kernel output | 잘못된 K축 broadcast를 일관되게 계산 | 결과 불일치 |
| logits | channel별 일정 배율의 오차 | 증상 |

수정은 단순히 `.t()`를 추가하는 것으로 끝내지 않았다. parameter type에 scale granularity와 semantic axis를 보존하고, backend adapter가 지원하는 axis와 일치하는지 검사했다. scalar, N-channel, K-group, 2D-block을 서로 다른 mode로 취급했다. unsupported mode는 reshape로 맞추지 않고 generic dequant fallback으로 보냈다. consumer 직전 assertion은 단순 numel 외에 expected logical axis와 stride/layout version을 검사했다.

회귀 fixture에는 `K≠N`, scale 값이 모두 다름, transpose view와 contiguous copy, TP=1/2, column/row parallel, scalar와 channel scale을 넣었다. code와 scale의 exact coordinate는 tolerance 없이 비교했고, dequantized values와 output은 dtype별 합리적 tolerance를 사용했다. 실제 model에서는 첫-token logits, 여러 prompt와 layer sampling을 비교했다. tolerance를 넓혀 고친 척하지 않았다.

vLLM 검증에서는 layer가 만든 quant method, post-load 변환과 apply wrapper를 하나의 call graph로 연결했다. SGLang 검증에서는 backend override 시점과 `weight_scale`/`weight_scale_inv` mutation을 연결했다. Transformers 검증에서는 raw quantization config와 normalized defaults, checkpoint tensor inventory를 고정했다. 세 framework에서 변수명이 비슷해도 한쪽의 의미를 다른 쪽에 복사하지 않았다.

rollback은 새 binary만 되돌리는 것으로 끝내지 않았다. 잘못된 version이 생성한 converted-weight cache와 captured graph를 함께 무효화했다. canonical checkpoint digest는 유지하고 layout/cache generation만 이전 값으로 되돌렸다. old and new replicas의 canary 결과를 확인한 뒤 traffic을 옮겼다. cache 파일이 남아 있었다면 old binary가 incompatible representation을 다시 읽어 같은 오류를 낼 수 있었다.

여기서 rollback terminal은 “이전 replica가 ready다”가 아니다. 이전 binary가 이전 layout consumer를 사용하고, 이전 또는 폐기된 conversion cache 정책을 따르며, 새 generation graph를 재사용하지 않고, reference와 같은 canonical tuple을 복원하는 상태다. traffic 전환 전에는 식별 fixture와 실제 canary를 모두 통과시킨다. traffic 전환 뒤에는 first-token mismatch, backend selection과 fallback reason이 예상 cohort 안에 머무는지 본다.

사고 중 임시 완화도 correctness를 보존해야 한다. 문제가 있는 backend를 특정 scale mode와 layer family에서만 disable하고, generic path가 native representation을 읽는다는 것을 확인한다. 전체 FP8을 BF16으로 즉석 변환하면 memory capacity와 scheduler admission이 함께 바뀔 수 있으므로 별도 위험을 평가한다. tolerance 확대, scale 평균화나 문제가 있는 channel 무시는 완화가 아니다.

운영 공지는 quantization 자체가 불안정하다고 뭉뚱그리지 않는다. 영향 snapshot, layer family, backend와 scale mode, 잘못된 generation, 탐지 시각과 차단 시각을 명시한다. 응답 content가 영향을 받았으므로 latency incident보다 높은 correctness 기준으로 이미 생성된 결과의 범위도 조사한다. raw prompt나 weight를 노출하지 않으면서 request cohort와 model generation을 연결한다.

사후 검토에서는 왜 사전 검증이 놓쳤는지 네 항목으로 정리했다. 첫째 square fixture가 축 교환을 숨겼다. 둘째 scale들이 비슷해 mismatch가 model-level tolerance 안에 들어왔다. 셋째 shape check가 semantic axis를 표현하지 않았다. 넷째 backend-off 실험을 처음에는 kernel 단독 변경으로 오해했다. 재발 방지는 “테스트 추가” 한 줄이 아니라 이 네 blind spot 각각을 닫았다.

### 50.13.2 zero-point가 한 칸 밀린 AWQ 오답을 분해한다

두 번째 incident는 asymmetric AWQ checkpoint였다. sym checkpoint와 small model은 정상인데 특정 projection만 top-token이 바뀌었다. qweight, qzeros와 scales의 shapes는 모두 맞았고 load도 성공했다. qweight를 native helper로 unpack한 결과도 reference와 같았다. 문제는 Marlin conversion 뒤 qzeros의 bias convention이었다.

fixture는 code `q=[0,1,7,8,14,15]`, raw zero nibble을 서로 다른 값으로 만들었다. scale은 모두 1로 두지 않고 group마다 1, 3, 7을 사용했다. 세 후보 식 `q-z`, `q-(z+1)`, `q-(z-1)`을 나란히 계산했다. runtime 결과가 두 번째 식과 정확히 맞아 bit corruption이나 scale permutation을 기각하고 ZP bias가 한 번 더 적용됐다는 가설을 세웠다.

source walk는 checkpoint-native qzeros unpack, conversion helper의 zero adjustment, optimized kernel의 dequant convention을 순서대로 읽었다. native representation은 encoded zero를 consumer가 보정하도록 설계됐지만 converter가 이미 보정한 값을 새 layout에 넣었다. adapter는 representation version을 바꾸면서 ZP convention field를 갱신하지 않았다. kernel은 old convention이라고 보고 다시 보정했다. shape, dtype와 histogram은 전부 정상인 silent semantic mismatch였다.

고칠 위치는 kernel이 아니었다. converter output contract를 “runtime numerical ZP”와 “encoded checkpoint ZP” 중 하나로 명명하고, parameter metadata도 같은 version으로 atomic 교체했다. generic fallback은 native qzeros만 받고 optimized path는 converted qzeros만 받게 했다. converted qweight에 native qzeros를 섞는 조합은 launch 전에 실패했다.

검증은 code 극값과 ZP 극값, group 첫/끝, word 첫/끝 nibble, N tail, TP shard boundary를 포함했다. symmetric mode에서는 qzeros parameter가 비어 있거나 implicit convention을 사용하므로 asymmetric fix가 그 path를 바꾸지 않는지 확인했다. ZP 없는 FP8/NVFP4 test가 우연히 이 버그를 가릴 수 있으므로 format family별 fixture를 분리했다.

rollback 때는 converter version을 cache key에 넣지 않았던 사실이 드러났다. 이전 binary가 만든 qweight와 새 binary가 만든 qzeros가 같은 directory에 공존할 수 있었다. bundle manifest에 모든 component digest와 representation version, producer commit, completion marker를 넣고 하나의 atomic rename으로 publish하도록 설계했다. component별 cache hit를 조합하지 않았다.

이 사건이 주는 실용적 교훈은 “zero-point를 확인하라”보다 구체적이다. raw nibble, decoded checkpoint zero, converted zero와 kernel-effective zero를 별도 열로 둔다. 어느 열도 그냥 `z`라고 쓰지 않는다. 각 변환의 입력·출력 convention을 명명하면 off-by-one이 정확히 한 경계에서 보인다.

### 50.13.3 NVFP4 scale interleave가 expert를 바꾼 사건

세 번째 incident는 fused MoE의 NVFP4 path에서만 나타났다. dense layer는 맞았고 expert 0도 대체로 맞았지만 expert 1 이후 output이 주기적으로 어긋났다. data payload를 unpack하면 FP4 codes는 정확했다. global scale을 1로 강제해도 pattern이 남았다. block scale을 모두 1로 두면 사라졌다. 따라서 data nibble과 global multiplier를 기각하고 block-scale addressing을 열었다.

canonical scale tensor는 `[expert,N-block,K-block]` 의미를 가졌지만 interleave helper는 flat input을 받았다. caller는 runner가 기대하는 permutation 전에 expert와 N-block axis를 바꿔야 했다. 문제 버전은 contiguous flat만 만들었다. total numel과 dtype은 동일했고 padding도 충분했으므로 validation을 통과했다. expert 0의 첫 tile은 permutation의 고정점이라 맞았고 다음 tile부터 다른 expert scale을 읽었다.

fixture는 expert 2개, N-block 2개, K-block 2개를 두고 scale 값을 `1,2,4,8,16,32,64,128`로 만들었다. data code는 모두 같은 값으로 두었다. output 배율만으로 어느 canonical scale이 어느 physical slot으로 이동했는지 역추적할 수 있었다. expert 하나 또는 모든 scale 1인 fixture는 이 오류를 숨긴다.

SGLang 고정 source의 scale shuffle helper를 읽을 때 `permute`, scratch shape, padding, flatten, FlashInfer interleave와 destination assignment를 각각 mutation 단계로 기록한다. 함수 전체를 “scale conversion” 한 줄로 요약하지 않는다. 각 단계 뒤 golden scale 위치 세 개를 기록하면 first divergence가 caller permutation인지 library interleave인지 구별된다.

수정은 runner별 layout descriptor를 cache key와 parameter owner에 붙였다. canonical scale은 immutable source로 유지하거나, memory 때문에 버릴 경우 late fallback을 금지했다. expert/N/K block counts와 padded counts를 descriptor에 넣었다. 같은 numel의 다른 axis permutation은 cache hit가 될 수 없게 했다.

검증은 expert 0/1 첫·끝 block, padding 바로 전후, EP rank local expert mapping, different runner 선택과 reload를 포함했다. optimized runner를 끈 결과만 보지 않고 canonical dequant, interleave inverse와 call 직전 physical scale 세 checkpoint를 비교했다. first divergence가 interleave 입력에서 사라진 뒤에만 full MoE output을 승인했다.

### 50.13.4 네 포맷을 비교하는 독자용 실전 체크리스트

처음 받은 checkpoint에서 먼저 model config와 quant config 파일을 분리한다. method label, bits, group size, sym, desc_act, activation scheme, scale recipe와 format/version을 raw form 그대로 보존한다. library가 default와 alias를 적용한 normalized form도 별도 저장한다. 두 dictionary의 차이가 실제 method 선택을 바꿀 수 있다.

다음으로 tensor inventory를 만든다. layer 하나를 골라 qweight/weight, qzeros, scales/weight_scale/weight_scale_inv, g_idx, input_scale, weight_scale_2 같은 이름을 모두 적는다. shape와 dtype뿐 아니라 file byte offset, storage order와 source shard를 기록한다. optional tensor가 빠졌다면 어떤 predicate 때문에 안전한지 source에서 확인한다.

세 번째는 canonical logical tuple이다. GPTQ는 `(k,n,q,g,z,s)`, AWQ는 `(k,n,q,group,encoded_z,effective_z,s)`, FP8은 `(k,n,raw8,format,sw)`와 request의 `(m,sx)`, NVFP4는 `(i,raw4,block,sb,sg)`를 만든다. 최소 세 좌표는 helper 없이 hex와 식으로 손계산한다. expected 값을 production converter로 생성하면 같은 bug를 공유한다.

네 번째는 loader range다. global K/N과 local K/N, pack factor, word range, group range, TP/EP slice를 나란히 둔다. logical range가 word나 group 중간을 자르면 loader가 padding, partial unpack 또는 reject 중 무엇을 하는지 확인한다. integer division 결과가 맞아 보인다는 이유로 tail을 버리지 않는다.

다섯 번째는 mutation ledger다. transpose, reshape, contiguous, shard copy, repack, scale permutation, ZP adjustment, g_idx sort, block-scale interleave와 parameter replacement를 시간순으로 적는다. 각 단계의 owner와 representation version을 붙인다. qweight·scale·ZP·ordering metadata가 하나의 transaction으로 publish되는지 본다.

여섯 번째는 backend gate다. GPU architecture만 보지 않는다. format variant, group/scale mode, activation dtype, M/N/K alignment, TP/EP, bias/output dtype, dependency build와 kernel availability를 모두 본다. false reason은 layer별로 남긴다. fallback consumer가 현재 representation을 읽을 수 있는지 별도 판정한다.

일곱 번째는 runtime call이다. argument 위치, pointer, shape, stride, scalar type, scale direction, workspace와 generation을 적는다. wrapper가 padding한 M/N/K와 logical M/N/K를 함께 적는다. output slice가 padding을 제거하는지도 본다. call signature에 semantic axis가 없다면 adapter가 어떤 invariant로 올바른 stride를 만들었는지 확인한다.

여덟 번째는 first divergence다. native unpack, native dequant, local shard, repack inverse, call 직전 dequant, kernel output, logits 순서로 비교한다. 앞 checkpoint가 틀렸는데 뒤 결과를 profile하지 않는다. 값 comparison은 code/index에 exact, float dequant와 output에 근거 있는 tolerance를 적용한다.

아홉 번째는 반증이다. backend off, TP=1, sym=true, scale=1, square shape와 contiguous copy는 하나씩만 바꾼다. 각 intervention이 실제로 바꾸는 state 목록을 쓴다. backend off가 repack과 loader를 함께 바꾸면 kernel 단독 반증으로 쓰지 않는다. scale=1이 정상화되면 scale path 후보가 강해지지만 axis와 direction, interleave는 아직 구별되지 않았다.

열 번째는 복구 terminal이다. corrected golden tuple, non-square/word/group/tile/expert boundary, full logits, fallback, reload와 graph generation을 통과한다. converted cache version과 bundle atomicity를 확인한다. rollback은 binary, cache, graph와 traffic generation을 함께 다룬다. canonical checkpoint를 변형하지 않아야 안전한 폐기가 가능하다.

이 체크리스트를 완료하면 보고서가 “AWQ가 깨졌다”에서 멈추지 않는다. 어느 source revision의 어떤 config가 어떤 tensor tuple을 만들었고, 어느 loader 함수의 어느 변환 뒤 몇 번째 logical coordinate가 처음 달라졌으며, 선택된 backend가 어떤 ABI로 그것을 소비했는지를 말할 수 있다. 수정과 회귀 fixture도 그 좌표에 직접 연결된다.

현장에서는 이 장부를 한 번에 완성하려 하지 않는다. 먼저 최초 오답을 재현하는 최소 projection과 세 식별 좌표를 고정한다. 다음으로 native unpack부터 consumer 직전까지 checkpoint를 하나씩 추가한다. 정상 경계는 닫고 최초 불일치 이후만 확장한다. source link에는 “이 줄이 보장하는 것”을 한 문장으로 붙이고, 실행 여부는 별도 trace로 증명한다. 조사 도중 backend나 tolerance를 바꾸면 새 실험 행을 만들며 이전 결과를 덮어쓰지 않는다. 이렇게 하면 긴 조사도 config, byte, metadata, loader, ABI, output의 인과 사슬을 잃지 않는다.

## 50.14 loader→conversion→consumer→rollback을 한 transaction으로 묶는다

### 50.14.1 conversion 실패와 동시 loading의 원자성

serving loader가 여러 shard 또는 layer를 병렬 변환하면 하나의 exception이 partial model을 남길 수 있다. layer A는 repacked, layer B는 native, method registry는 전체 model이 optimized라고 표시된 상태가 될 수 있다. publish 전 staging model을 사용하거나 layer별 representation state를 확인해 요청 수락 전에 전부 일관적인지 검증한다.

conversion cache 파일도 원자성이 필요하다. writer가 qweight를 쓴 뒤 scale/ZP metadata 전에 중단되면 filename이 존재해도 bundle은 불완전하다. manifest에는 source snapshot/config, converter revision, backend/layout version, tensor hashes/shapes와 completion marker가 필요하다. 임시 artifact를 final name으로 atomic publish하는 lifecycle을 본다.

여러 process가 같은 cache key를 만들면 lock 또는 content-addressed publish가 필요하다. last writer의 bytes가 논리적으로 같다면 안전할 수 있지만 library/build가 다른데 key가 부족하면 interleave가 섞인다. cache hit는 파일 존재가 아니라 manifest exact match와 bundle validation 뒤에 기록한다.

TP ranks도 같은 effective policy에 합의해야 한다. rank 0만 optimized backend를 선택하고 rank 1이 generic fallback이면 local outputs의 numerical/shape contract와 collective timing이 달라질 수 있다. device capability가 heterogeneous하거나 package import가 rank별로 다를 때 fail-fast 또는 globally compatible method를 선택한다. rank-local fallback을 허용하는 설계라면 동등한 logical dequant와 output contract를 명시한다.

model reload는 old/new generation을 분리한다. old requests가 old runner와 parameter를 마지막까지 소유하고, new requests만 new representation을 본다. 주소 allocator가 재사용돼도 generation이 다르다. graph executable이 old pointer를 capture했다면 old graph lifetime이 끝나기 전에 storage를 해제하지 않는다.

conversion cancellation도 다룬다. startup 취소나 health timeout 뒤 background worker가 계속 parameter를 mutate하면 재시도 object와 충돌할 수 있다. cancel signal, worker join, temporary cleanup과 publish 금지를 source lifecycle에서 확인한다. partial cache는 다음 load에서 miss 또는 corrupt로 판정돼야 한다.

### 50.14.2 배포 전 마지막 dry review

첫 질문은 file manifest다. snapshot, config와 all quant tensors가 같은 revision인가. 둘째는 logical reference다. golden coordinate의 q/scale/ZP/g_idx와 FP8/NVFP4 scale hierarchy를 손으로 복원했는가. 셋째는 loader coverage다. 모든 destination slice가 정확히 한 source mapping으로 채워졌는가.

넷째는 representation transition이다. native→shard→repack/interleave 단계마다 owner와 inverse identity가 있는가. 다섯째는 backend gate다. layer별 effective method와 false reason, fallback consumer가 현재 representation과 맞는가. 여섯째는 lifecycle이다. conversion cache, graph, reload와 cancellation generation이 분리되는가.

일곱째는 정확성 fixture다. non-square axis, word/group/tile 경계, TP/EP, act-order, asym ZP, block/global scale와 padding을 포함하는가. 여덟째는 failure fixture다. unsupported 조합, missing metadata, corrupt cache와 partial conversion이 launch 전에 실패하는가. 아홉째는 serving fixture다. fallback, reload와 동시 request가 old/new weight를 섞지 않는가.

마지막은 성능 승인이다. correctness가 확정된 두 representation만 비교한다. conversion startup, peak storage, effective kernel과 end-to-end latency를 각각 귀속한다. faster wrong representation과 slower correct fallback을 비교해 fallback을 regression이라 부르지 않는다. 먼저 correct baseline을 세운다.

dry review 결과는 “GPTQ 지원” 같은 한 줄이 아니다. format version과 config, layer/shard shape, representation owner, backend/fallback, golden fixture와 failure behavior가 한 page에 연결돼야 한다. 새로운 GPU나 backend가 추가되면 support matrix의 체크 표시가 아니라 해당 좌표 변환과 cache key를 다시 검토한다.

### 50.14.3 오답 pattern을 좌표 가설로 번역하는 법

모든 channel이 같은 배율로 틀리면 global direct/inverse scale을 먼저 본다. channel마다 일정한 배율이면 per-channel axis 또는 TP slice다. K group마다 달라지면 group index/ZP, 인접 값이 쌍으로 교환되면 nibble order, tile 주기로 섞이면 repack permutation, expert 경계에서만 바뀌면 expert axis/interleave가 강한 후보다. 이 표지는 결론이 아니라 다음 checkpoint 선택이다.

오류가 첫 token부터 동일하게 재현되면 immutable weight representation 후보가 높다. request마다 달라지면 dynamic activation scale, workspace generation과 backend selection을 본다. graph replay에서만 생기면 static scale/metadata buffer lifetime을 추가한다. 그러나 재현 pattern 하나로 층을 확정하지 않고 first dequant coordinate를 비교한다.

NaN/Inf는 scale이 0 또는 비정상 reciprocal, FP8 special encoding, uninitialized padding에서 올 수 있다. 모든 NaN을 overflow로 부르지 않는다. code, scale raw bits와 dequant 순서 중 처음 special value가 생기는 곳을 찾는다. finite reference fixture로 format decoding을 먼저 검증한다.

오차가 작다는 사실도 representation correctness 증거가 아니다. scale permutation이 비슷한 값끼리 바뀌거나 model weight가 대칭이면 logits 차이가 tolerance 안일 수 있다. 식별 scale을 가진 adversarial fixture에서 exact tuple을 본다. 실제 model quality tolerance는 그다음 별도 승인이다.

### 50.14.4 vLLM·SGLang·Transformers에서 같은 필드를 찾는 법

세 codebase를 비교할 때 파일 이름을 기계적으로 맞추지 않는다. Transformers는 serialized config를 Python object로 정상화하고 quantizer가 module replacement와 load 전후 처리를 조정한다. vLLM은 quant config가 layer별 quant method를 만들고 parameter loader와 post-load processing이 runtime representation을 만든다. SGLang은 이와 비슷한 parameter 계층 위에 backend override와 runner-specific conversion이 더 가까이 붙을 수 있다. 같은 역할을 찾되 class 이름이 같을 것이라 기대하지 않는다.

첫 trace는 config field의 생애다. 예를 들어 `group_size`를 JSON key에서 constructor argument, normalized attribute, method selection, parameter shape 식, group lookup까지 잇는다. 중간에 `-1`이 channel-wise 의미로 바뀌거나 default가 채워지면 그 상태를 기록한다. CLI가 checkpoint config를 override할 수 있다면 precedence를 적는다. 최종 consumer가 field를 직접 받지 않아도 이미 parameter shape나 converted layout에 효과가 굳어 있을 수 있다.

둘째 trace는 tensor name의 생애다. `qweight`라는 source name이 destination parameter 하나로 들어가는지, stacked QKV와 gate-up에서 여러 slice로 나뉘는지 본다. `weight_scale`은 scalar, channel vector, group matrix와 block matrix일 수 있다. name equality를 semantic equality로 쓰지 않는다. load callback의 destination axis attribute와 source shard id를 함께 읽는다.

셋째 trace는 backend 선택의 시간이다. parameter 생성 전 선택이면 target representation을 바로 allocate할 수 있다. load 후 선택이면 native representation을 converted representation으로 바꾸는 경계가 있다. device capability, local shard alignment나 package import처럼 늦게만 알 수 있는 gate가 무엇인지 찾는다. 늦은 gate가 false일 때 canonical copy가 남아 있는지도 확인한다.

넷째 trace는 call argument다. Python method의 `apply`에서 extension operator나 custom op까지 내려가며 qweight, scales, qzeros, g_idx가 어느 순서로 전달되는지 기록한다. wrapper가 reshape, pad, cast와 contiguous copy를 삽입하면 각각 새 physical representation이다. C++/CUDA binding의 argument check가 numel과 dtype만 보는지, layout version도 검증하는지 확인한다.

다섯째 trace는 source가 보장하지 않는 항목이다. config validation은 file contents를 보장하지 않는다. parameter registration은 모든 bytes가 loaded됐음을 보장하지 않는다. repack helper 존재는 해당 layer가 그 helper를 실행했음을 보장하지 않는다. operator symbol 등록은 현재 build가 kernel image를 포함하거나 runtime gate가 선택했음을 보장하지 않는다. 증거 범위를 한 문장씩 제한한다.

Transformers를 source of truth라고 부를 때도 범위를 명시한다. Hugging Face model config와 quant config를 해석하는 기준일 수 있지만, 외부 quantizer가 만든 packed tensors의 모든 ABI를 Transformers 자체가 정의하지 않을 수 있다. producer repository와 checkpoint manifest가 추가 source of truth다. serving engine이 지원을 표방해도 특정 exporter revision의 schema까지 호환되는지 support predicate를 확인한다.

vLLM과 SGLang의 공통 parameter utility를 보면 pack factor, packed dimension, output/input dimension 같은 attributes가 shard logic을 움직인다. model-specific loader가 이 attributes를 덮거나 stacked parameter callback을 쓰는지 확인한다. generic quant file만 읽고 특정 model의 QKV loader를 생략하면 실제 slice 좌표를 놓칠 수 있다.

MoE에서는 trace table에 expert 축을 맨 앞에 추가한다. global expert id, EP local id, checkpoint tensor first dimension, qweight/scales/ZP의 expert slice와 runner routing id를 연결한다. 일부 framework는 non-local expert name을 load 단계에서 skip하고, 다른 path는 tensor 일부를 materialize한 뒤 자를 수 있다. correctness는 local coverage로, startup memory와 IO는 materialization 시점으로 따로 평가한다.

### 50.14.5 ABI mismatch를 launch 전에 거절하는 validation 설계

좋은 validation은 “shape mismatch”라는 한 문장보다 구체적인 expected/actual contract를 보여 준다. 오류에는 layer name, format/version, logical K/N, physical qweight shape, scale granularity/axis, group size, ZP convention, ordering mode, selected backend와 failed predicate를 포함한다. raw tensor values나 pointer는 노출하지 않는다.

validation 순서는 값싼 검사부터 시작한다. config enum과 range, required tensor inventory, dtype와 rank, logical-to-physical shape 식, alignment, representation version, backend capability를 본다. expensive inverse-repack이나 sampled dequant는 canary/offline validation에서 수행할 수 있다. 다만 값싼 검사로 semantic axis를 표현하지 못한다면 axis descriptor를 schema에 추가한다.

scale tensor가 length N이라는 검사만으로 per-channel 의미를 증명할 수 없다. K와 N이 같을 수 있기 때문이다. loader가 source field의 semantic axis를 parameter metadata에 붙이고, transpose/shard가 이를 갱신하며, adapter가 backend expected axis와 비교해야 한다. singleton squeeze로 descriptor를 잃지 않는다. scalar와 length-one channel vector도 의미가 다를 수 있다.

ZP validation은 sym flag와 tensor 존재만 보지 않는다. encoded bias convention과 bit width를 representation version에 넣는다. qzeros pack factor와 logical group/N coverage를 계산한다. optimized converter가 ZP를 numerical value로 바꾼다면 output descriptor도 바꾼다. consumer가 encoded인지 numerical인지 선언하지 않으면 launch를 거절한다.

ordering validation은 `g_idx.numel()==K`를 넘는다. domain이 valid group 범위인지, sort permutation이 bijection인지, inverse가 존재하는지, qweight row permutation과 activation gather가 같은 mapping을 쓰는지 확인한다. TP local slice라면 global/local domain을 구분한다. sorted g_idx를 보고 원래 activation 순서가 자동 복원된다고 가정하지 않는다.

NVFP4 validation은 data와 scales를 bundle로 본다. logical element count에서 expected data bytes와 block count를 계산하고, row/tile padding을 별도 합친다. block-scale dtype/encoding, canonical axes, physical interleave version, global scale shape와 direction을 검사한다. runner가 architecture별 layout을 요구하면 device capability와 layout id를 함께 gate한다.

FP8 validation은 E4M3/E5M2, uint8 transport인지 typed tensor인지, scale direct/inverse, static/dynamic, per-tensor/per-channel/per-token/per-block을 구별한다. activation scale이 dynamic이면 request M과 capture bucket M의 lifetime을 본다. weight scale이 immutable이어도 graph가 pointer를 capture한 뒤 parameter replacement를 허용하면 generation assertion이 필요하다.

오류 정책은 세 가지다. representation 변환 전이라 canonical consumer가 있으면 compatible fallback을 고른다. reversible conversion 또는 canonical copy가 있으면 안전하게 되돌린 뒤 fallback한다. 현재 representation을 읽는 consumer가 없으면 명시적으로 실패한다. tensor를 reshape하거나 scale을 평균내 unsupported mode를 억지로 실행하지 않는다.

### 50.14.6 정확성을 지킨 뒤에만 성능을 비교한다

양자화 성능표는 weight bit 수 하나로 설명되지 않는다. 작은 M decode에서는 launch, dequant와 metadata traffic이 크게 보일 수 있고, 큰 M prefill에서는 GEMM throughput이 지배적일 수 있다. group이 작아지면 scale metadata와 lookup이 늘고 accuracy는 좋아질 수 있다. act-order는 permutation/gather와 specialized support를 요구할 수 있다. NVFP4의 block scale interleave는 tensor-core path를 열지만 conversion과 cache 비용을 만든다.

비교 실험의 첫 열은 correctness status다. golden tuple, output tolerance, backend/fallback과 representation version이 같은 의미를 계산하는지 먼저 표시한다. 이 행을 통과하지 못한 결과는 latency chart에서 제외한다. 빠른 wrong-output을 최적화 성과로 표시하지 않는다.

둘째 열은 저장과 peak memory를 나눈다. canonical checkpoint bytes, loaded native parameters, converted parameters, scales/ZP/ordering metadata, workspace, conversion temporary와 graph-captured buffers를 분리한다. runtime steady state에서 canonical을 버렸다면 fallback 제약을 기록한다. startup peak가 device OOM을 만들 수 있으므로 평균 resident memory만 보지 않는다.

셋째 열은 shape cohort다. prefill M 구간, decode M=1 또는 batched decode, K/N alignment, TP/EP와 layer family를 나눈다. model 평균 latency 하나는 rare tail이 generic fallback으로 가는 것을 숨긴다. backend selection counter와 reason을 bounded category로 묶고 sampled trace에서 exact shape를 확인한다.

넷째 열은 conversion cost다. download/file read, CPU unpack, host/device copy, GPU repack, JIT, graph capture와 cache write를 분리한다. first start와 warm cache를 섞지 않는다. cache hit가 correctness validation을 생략한다면 manifest exact match 비용도 측정에 포함한다. load time 단축이 stale representation 위험을 늘리지 않아야 한다.

다섯째 열은 end-to-end serving 효과다. kernel time이 줄어도 scheduler queue, tokenizer, communication과 sampling이 지배하면 request latency 변화가 작을 수 있다. 반대로 weight memory 절감으로 더 큰 batch나 KV capacity가 가능해 throughput이 늘 수 있다. kernel microbenchmark와 capacity effect를 다른 claim으로 쓴다.

GPTQ와 AWQ를 비교할 때 같은 bits/group/backend인지 확인한다. algorithm quality 비교와 runtime layout/kernel 비교를 섞지 않는다. 한쪽만 Marlin이고 다른 쪽이 generic이면 latency 차이는 quantization algorithm보다 backend path일 수 있다. 동일 logical layer와 effective backend를 맞춘 비교, 실제 checkpoint quality 비교를 별도 실험으로 둔다.

FP8과 NVFP4 비교도 같은 원칙이다. FP8 data byte와 NVFP4 data nibble만 비교하지 않고 scale overhead, padding, supported architecture, activation quantization과 accumulator/output dtype을 포함한다. NVFP4가 더 작아도 unsupported layer가 BF16 fallback이면 model 전체 memory와 latency는 단순 비율이 아니다.

성능 regression이 보이면 먼저 backend coverage가 바뀌었는지 본다. config default, alignment, dependency build나 GPU capability gate 때문에 optimized path 일부가 fallback됐을 수 있다. 다음으로 conversion/cache, metadata traffic와 kernel을 분리한다. 정확성 fix가 descriptor check 하나를 추가했는데 latency가 크게 늘었다면 check가 request hot path에 들어갔는지 확인한다. immutable weight contract는 load/capture 시점에 검증하고 hot call에는 compact generation assertion만 둘 수 있다.

### 50.14.7 패치 리뷰와 롤백 승인표

패치가 scale transpose를 바꿨다면 source semantic axis, TP shard axis, destination shape/stride와 backend broadcast를 한 표에 보여 준다. 단위 테스트가 output만 비교하면 우연한 상쇄를 놓칠 수 있다. 식별 scale을 사용한 exact coordinate assertion을 추가한다. square layer 외 non-square layer를 반드시 포함한다.

qweight repack을 바꿨다면 word/lane/tile 경계와 inverse identity를 본다. scale/ZP/g_idx가 같은 logical permutation을 따르는지 검토한다. repack helper만 수정하고 parameter metadata나 cache version을 그대로 두면 old converted artifact가 새 consumer에 들어갈 수 있다. layout version과 invalidation을 패치 일부로 본다.

ZP 수정을 리뷰할 때 raw encoded, converter output과 kernel effective value 세 숫자를 fixture에 남긴다. symmetric path와 asymmetric path를 분리하고, implicit bias가 있는 format을 별도 처리한다. code와 ZP의 bit width가 같은지 가정하지 않는다. qzeros tail padding이 dequant에서 읽히지 않는지도 확인한다.

NVFP4 scale shuffle 수정은 expert/N/K block permutation을 작은 정수 표로 보여 준다. canonical index→scratch index→interleave output index→runner read index의 합성이 identity인지 확인한다. helper round trip만으로 끝내지 않고 독립적으로 계산한 세 좌표를 둔다. padding slot이 valid scale처럼 읽히지 않게 mask/alignment를 본다.

동시성 리뷰는 staging과 publish 경계를 찾는다. 모든 component 변환이 성공하기 전에 method registry, graph와 cache가 새 representation을 볼 수 없어야 한다. 한 TP rank 실패 시 다른 rank가 요청 수락 상태로 남지 않게 합의 terminal을 둔다. cancellation worker가 join된 뒤 temporary를 지우고, reload의 old generation은 진행 중 request가 끝날 때까지 산다.

관측 리뷰는 high-cardinality를 피한다. layer별 exact details는 sampled trace나 diagnostic dump에 두고, metric은 format, backend, fallback reason, conversion result와 generation mismatch 같은 bounded label을 쓴다. fallback count와 expected layer inventory를 함께 봐야 silent generic construction을 발견할 수 있다. raw weights와 scale values는 metric/log에 넣지 않는다.

롤백 승인표에는 binary image, Python package/build, model snapshot, raw/normalized config, conversion/cache version, graph generation과 traffic cohort가 들어간다. 무엇을 되돌렸는지 하나라도 빠지면 mixed generation이 남을 수 있다. rollback 후 canonical golden fixture와 canary logits를 다시 실행한다. 단지 error rate가 내려갔다는 관측으로 wrong-output 복구를 승인하지 않는다.

마지막 칸은 반증된 가설이다. scale-axis 사건에서 FP8 rounding, kernel reduction과 file corruption을 어떤 checkpoint로 제외했는지 쓴다. AWQ ZP 사건에서 nibble order와 scale permutation을 왜 제외했는지 쓴다. NVFP4 사건에서 data payload와 global scale을 왜 제외했는지 쓴다. 이 기록이 있어야 다음 담당자가 같은 넓은 의심부터 반복하지 않는다.

### 50.14.8 revision 변경을 다음 검증으로 넘긴다

upgrade에서는 quant config aliases/default, parameter packed dimension, shard callback, repack permutation, backend support predicate와 cache key를 우선 diff한다. 함수 이름 이동보다 semantic state transition이 바뀌었는지 본다. 새로운 backend가 default가 되면 기존 converted cache를 읽는지와 layout version을 확인한다.

Transformers upgrade로 serialized config default가 바뀌면 같은 checkpoint hash도 normalized state가 달라질 수 있다. vLLM/SGLang upgrade로 supported method가 늘면 이전 fallback layer가 optimized conversion을 시작한다. FlashInfer upgrade로 scale interleave가 바뀌면 cache invalidation이 필요하다. llama.cpp GGUF type 추가는 기존 Q4와 자동 호환을 의미하지 않는다.

diff review fixture는 이전/current source에 같은 raw config와 golden vector를 입력한다고 가정해 expected normalized state와 coordinate mapping을 정적으로 비교한다. 변경이 의도된 경우 schema/layout version과 migration을 요구한다. 의도되지 않았다면 first different predicate나 index 식에서 멈춘다.

rollback도 forward upgrade만큼 중요하다. 새 version이 만든 converted cache를 old binary가 읽지 않도록 version key를 둔다. canonical checkpoint는 immutable source로 유지하고 runtime cache는 폐기 가능해야 한다. rollback이 inverse conversion에 의존하면 converter bug까지 되돌릴 수 없으므로 위험하다.

이 최소 diff는 모든 quant code를 다시 읽는 대신 ABI가 바뀔 수 있는 좁은 경계를 제공한다. 다만 source predicate가 새 helper로 이동했으면 call graph를 따라 owner를 재확인한다. release note의 “FP8 support improved”를 layout 안정성 보장으로 쓰지 않는다.

특히 함수명만 유지된 채 내부 consumer가 바뀌는 경우를 경계한다. signature diff가 없어도 scale stride, encoded zero bias, supported alignment와 cache version은 달라질 수 있다. 이전 fixture를 그대로 실행하는 이유는 API 모양이 아니라 logical tuple의 보존을 다시 증명하기 위해서다.

upgrade 승인 문서에는 이전→새 normalized config, layer별 이전→새 backend, representation/cache version과 golden coordinate 결과를 나란히 둔다. backend가 그대로여도 compiler나 dependency가 converter output을 바꿀 수 있으므로 converted tensor digest만 비교하지 않고 inverse logical tuple을 비교한다. byte가 달라도 tuple이 같으면 layout 변화일 수 있고, byte가 같아도 consumer convention이 바뀌면 ABI는 깨질 수 있다.

새 format field가 들어왔지만 current consumer가 읽지 않는다면 효과가 있다고 쓰지 않는다. parser가 dictionary에 보존하는 것, method selection이 사용하는 것, loader shape를 바꾸는 것, runner argument가 되는 것을 구분한다. 사용되지 않는 metadata는 forward compatibility를 위한 보존일 수 있지만 serving state는 바꾸지 않는다. 반대로 default로 채워진 field는 file에 없어도 실행 path를 바꿀 수 있다.

지원표를 업데이트할 때는 format 이름, GPU와 체크 표시만 추가하지 않는다. 최소 supported bits/group/scale mode, required conversion, representation owner, fallback과 known unsupported combination을 링크한다. “지원”은 올바른 checkpoint가 특정 조건에서 load되고 logical tuple을 보존하며 실행 가능한 backend 또는 명시적 fallback에 도달한다는 문장이다.

마지막으로 fixture 자체도 revision을 가진다. golden hex, expected codes/scales와 coordinate 식을 사람 읽을 수 있는 문서와 machine-readable vector로 함께 관리한다. converter source와 같은 helper로 expected 값을 생성하면 같은 bug를 공유할 수 있으므로 최소 경계 값은 독립 손계산을 남긴다. 이것이 upgrade 때 first divergence를 빠르게 찾는 기준점이다.

검토자는 golden vector가 특정 backend 출력에 맞춰 사후 수정되지 않았는지도 본다. canonical logical tuple은 checkpoint schema와 독립 계산에서 나오고, backend output은 그 tuple의 소비 결과여야 한다. tolerance 변경도 mismatch를 숨기는 수단이 아니라 accumulator와 reduction 변화에 관한 별도 근거를 가져야 한다. code, ZP, g_idx와 representation index에는 floating tolerance를 적용하지 않는다.

이 기준을 지키면 새 quant backend를 추가할 때 검증 순서가 안정된다. 먼저 canonical tuple을 읽는 adapter를 증명하고, 다음으로 backend layout converter와 inverse를 증명하며, 마지막으로 runtime call과 serving lifetime을 검증한다. 성능 측정은 이 세 correctness boundary를 통과한 뒤에만 의미가 있다.

### 50.14.9 종합 회고: 4-bit가 아니라 좌표 변환을 읽는다

처음의 FP8 사건은 shape가 맞으면 ABI도 맞을 것이라는 믿음에서 시작했다. `[4]` scale은 K=4와 N=4 양쪽에 붙을 수 있었다. square fixture와 자동 broadcast는 오류를 숨겼다. non-square matrix와 channel별 식별 scale을 넣자 first divergence는 kernel output이 아니라 loader reshape였다. 이 장의 가장 중요한 기술은 더 많은 metric이 아니라 가설을 구별하는 fixture였다.

GPTQ/AWQ에서도 같은 교훈이 다른 모습으로 나타났다. qweight byte 수가 맞고 histogram이 같아도 nibble permutation이 틀릴 수 있다. group size가 config와 맞아도 act-order g_idx가 runtime에서 사라질 수 있다. qweight repack이 정확해도 scale/ZP가 같은 permutation을 따르지 않으면 logical tuple은 깨진다. representation correctness는 tensor 하나가 아니라 `(q,scale,ZP,g_idx)` bundle의 좌표 보존이다.

FP8에서는 code format과 scale broadcast가 bundle이다. weight E4M3이라는 dtype 이름만으로 per-tensor인지 per-channel인지 알 수 없다. activation도 static FP8 file tensor일 수 있고 request마다 per-token quantize될 수 있다. direct/inverse scale, block axes와 accumulator/output dtype이 call ABI에서 만난다. storage byte를 설명하는 문장과 compute type을 설명하는 문장을 분리해야 한다.

NVFP4는 bundle이 더 분명하다. data 두 values/byte라는 계산만 하면 block scale와 global scale storage를 빼먹는다. 두 backend가 NVFP4를 지원해도 physical scale interleave와 architecture alignment는 다를 수 있다. dtype와 elements가 같은 tensor를 직접 넘기는 것이 가장 위험한 silent reinterpretation이다. representation owner와 backend-specific cache key가 필요한 이유다.

llama.cpp GGUF는 이름 유사성이 호환성을 만들지 않는 극단적인 사례였다. Q4_0, Q4_K와 GPTQ INT4는 평균 bits가 비슷할 뿐 block struct, scale/min, high bits와 consumer가 다르다. 공통으로 쓸 수 있는 것은 logical coordinate를 physical byte와 dequant value까지 추적하는 방법이다. 실제 index 식은 format type마다 다시 읽는다.

독자가 새 quant format을 만났을 때 첫 질문은 “몇 비트인가”가 아니다. source config에서 어떤 method/version이 effective state가 됐는가, file tensor 각각이 logical value의 어느 요소를 담는가, loader가 어느 축으로 shard하고 어떤 representation으로 mutate하는가, runtime backend가 그 representation을 정확히 소비하는가를 묻는다. 이 네 질문이 byte 절감이나 성능 비교보다 먼저다.

조사 보고서는 다음 장면을 재현할 수 있어야 한다. snapshot과 raw/normalized config를 고정하고, 한 source coordinate의 word/shift, group/ZP/g_idx와 dequant reference를 계산한다. TP/EP slice 뒤 local coordinate를 찾고, repack inverse가 같은 tuple을 내는지 본다. runtime call의 pointer/shape/stride/convention을 확인하고 selected backend gate와 fallback reason을 적는다. first divergence 이전의 층은 잠정 정상, 이후 층만 조사 범위로 남긴다.

반증 기록도 결론만큼 중요하다. scale-axis 사건에서 saturation과 reduction을 왜 버렸는지, nibble 사건에서 byte endian과 ZP bias를 왜 버렸는지, act-order 사건에서 scale permutation과 double permutation을 어떻게 구별했는지, NVFP4 사건에서 data unpack과 global scale을 왜 제외했는지 남긴다. 다음 조사자는 같은 포괄적 의심을 반복하지 않고 첫 미확정 경계에서 시작할 수 있다.

수정은 invariant 문장으로 리뷰한다. scale 수정은 “source output-channel n의 scale이 transpose와 TP 뒤 local n을 가리킨다”를 보장한다. nibble 수정은 “format version의 logical i가 명시된 word shift에서 복원된다”를 보장한다. act-order 수정은 “weight와 activation이 같은 permutation을 공유하고 g_idx group을 읽는다”를 보장한다. NVFP4 수정은 “selected runner가 자신의 interleave version으로 변환된 scale bundle만 받는다”를 보장한다.

성능 최적화는 이 invariant를 약화할 권한이 없다. canonical copy를 버려 memory를 줄일 수 있지만 늦은 incompatible fallback을 금지해야 한다. conversion을 cache해 startup을 줄일 수 있지만 key에 backend/layout/config generation을 넣어야 한다. validation을 줄여 load를 빠르게 할 수 있지만 build-time 또는 offline manifest가 같은 증거를 제공해야 한다. 비용을 다른 시점으로 옮길 수는 있어도 representation 책임을 없앨 수는 없다.

운영자는 model-level “GPTQ” metric보다 layer별 effective backend와 fallback reason을 본다. 그러나 고카디널리티 shape를 무제한 label로 만들지는 않는다. bounded category metric에서 sampled trace로 들어가 정확한 parameter ledger를 연다. wrong-answer는 latency dashboard와 별개로 golden fixture, canary와 generation assertion을 통과해야 한다.

마지막 승인에서는 boundary가 넓어진다. 작은 matrix의 exact logical tuple, 실제 layer shape의 tolerance output, TP/EP coverage, graph/reload generation, fallback과 save/reload를 함께 본다. 하나의 happy-path benchmark만 통과한 quant model은 아직 serving-ready가 아니다. 특히 cancellation이나 adapter/backend 전환이 converted parameter 자체를 바꾸지 않더라도 runner binding과 graph address lifetime을 흔들 수 있다.

이제 첫 질문에 답할 수 있다. 파일과 shape가 정상인데 logits가 틀렸던 이유는 FP8이 본질적으로 부정확해서가 아니었다. checkpoint가 N축 scale을 말했지만 loader와 backend 사이에서 그 의미가 K축 broadcast로 바뀌었기 때문이다. first dequant coordinate에서 이미 차이가 났으므로 kernel은 주어진 ABI를 충실히 계산했을 가능성이 높았다.

“왜 이 옵션이 있는가”에도 같은 방식으로 답한다. bits는 code range와 pack factor를, group size는 scale lookup 좌표를, sym/ZP는 dequant 식을, act-order는 K permutation과 group mapping을, scale mode는 broadcast stride를, backend version은 physical representation consumer를 바꾼다. 옵션은 label이 아니라 state transition이다. 소비 함수와 tensor 변화를 말하지 못하면 옵션 설명은 완성되지 않았다.

이 장의 결론은 양자화가 숫자를 줄이는 기술이라는 상식보다 구체적이다. serving에서 quantized checkpoint를 안전하게 실행한다는 것은 logical weight coordinate를 file byte, loader mutation과 runtime argument 사이에서 보존하는 일이다. code, scale, zero-point와 ordering metadata 가운데 하나라도 다른 좌표계를 쓰면 빠르게 틀린 답을 낸다. 반대로 이 좌표 사슬을 장부로 만들면 처음 보는 format도 같은 질문으로 해부할 수 있다.
