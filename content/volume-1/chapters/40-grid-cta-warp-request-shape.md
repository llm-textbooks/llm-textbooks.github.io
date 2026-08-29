# 40장. 요청의 모양이 grid·CTA·warp가 되기까지

한 사용자가 128개 토큰으로 된 프롬프트를 보냈고, 다른 여덟 사용자는 각자 다음 토큰 하나를 기다린다고 하자. HTTP 계층에서 두 사건은 각각 “요청 한 건”과 “요청 여덟 건”으로 보인다. 모델 실행기가 두 사건을 모으면 첫 사건은 보통 128개의 query row를 가진 prefill이 되고, 둘째 사건은 여덟 개의 query row를 가진 decode batch가 된다. 그런데 GPU는 HTTP 요청도, 문장도, prefill이라는 이름도 실행하지 않는다. GPU가 받는 것은 어느 stream에서 어떤 kernel을 `gridDim`, `blockDim`, dynamic shared-memory 크기로 launch하라는 명령이다. kernel 안의 thread는 `blockIdx`와 `threadIdx`를 이용해 자신이 맡을 tile과 원소를 계산한다.

이 번역을 건너뛰면 흔한 오진이 생긴다. `M=8`이므로 thread 여덟 개만 일한다고 말하거나, CTA가 하나뿐이므로 occupancy가 낮다고 단정하거나, grid의 `y`축을 언제나 head라고 읽는다. 셋 다 그럴듯하지만 kernel 구현을 보기 전에는 참이 아니다. logical row가 kernel thread와 일대일로 대응한다는 보장이 없고, occupancy는 launch된 CTA 수가 아니라 SM에 상주할 수 있는 active warp의 비율이며, grid 축의 의미는 launcher와 tile scheduler가 정한다.

이 장의 목표는 요청 하나를 다음 사슬로 끝까지 추적하는 것이다.

`HTTP request → token rows M → kernel work tiles → grid/CTA → warp와 thread 역할 → tail predicate`

중심 예제는 교육용으로 고정한다. row tile은 `BLOCK_M=64`, CTA당 thread는 128개, 즉 NVIDIA GPU에서 네 warp이며, static launch라면 `grid.x=ceil_div(M,64)`라고 하자. prefill `M=128`과 decode `M=8`을 손으로 계산한 뒤, 이 단순식이 실제 vLLM FlashAttention과 SGLang·FlashInfer에서 어디까지 유지되고 어디서 깨지는지 읽는다. 실제 kernel을 실행하거나 성능을 측정하지 않는다. 여기서 제시하는 수치는 shape 산술이며, 실측 latency·occupancy·throughput이 아니다.

공식 실행 계약은 NVIDIA CUDA C++ Programming Guide 12.9.1의 §5.2와 CUDA Programming Guide 13.3.0의 §1.2.2.1–1.2.2.2에 고정한다. 두 문서 사이에 목차와 설명 방식은 달라졌지만 grid, thread block, thread, warp라는 기본 실행 계층은 이어진다. 구현 산책은 vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, vLLM의 FlashAttention fork commit `caaa4eb59845388a20b1f435ecaafb4bd9517ad8`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, FlashInfer v0.6.17 snapshot commit `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`만 사용한다.

## 40.1 HTTP 요청 수와 GPU의 일감 수는 왜 다른가

### 40.1.1 요청은 제어 단위이고 row는 계산 단위다

HTTP 서버는 request ID, prompt, sampling parameters, deadline, tenant와 streaming connection을 관리한다. tokenizer를 지난 prompt는 token IDs가 되고 scheduler는 이번 step에 각 sequence에서 몇 token을 계산할지 정한다. 이때 request의 개수와 model forward의 첫 차원은 같을 수도 있고 다를 수도 있다. 한 prefill request가 128개 query tokens를 내면 `M=128`일 수 있고, 여덟 decode requests가 각각 한 query token을 내면 합친 tensor도 `M=8`일 수 있다.

`M`은 이 장에서 “이번 forward에 실제 query projection과 attention output을 계산할 token row의 합”을 뜻한다. 전체 context length와 같지 않다. decode `M=8`이어도 각 sequence의 KV length는 수천일 수 있다. 따라서 query row는 작지만 읽어야 할 K/V 영역은 클 수 있다. 반대로 chunked prefill은 전체 prompt가 8,192 tokens여도 scheduler가 이번 step에 512개만 허용해 `M=512`를 만들 수 있다. HTTP payload, logical sequence length, scheduled tokens, flattened query rows를 따로 기록해야 한다.

flattening은 sequence 경계를 없애는 것이 아니라 표현을 바꾼다. query tensor가 `[M,H_q,D]`로 이어져 있어도 cumulative sequence lengths, per-request lengths, slot mapping, block table 같은 metadata가 어느 row가 어느 sequence에 속하는지 보존한다. varlen attention은 이 metadata로 row의 시작과 끝, causal 범위, KV page를 복원한다. `M`만 보고 attention의 총 일을 계산할 수 없는 이유다.

예를 들어 prefill A의 길이가 128이고 decode requests B–I가 각각 한 token을 낸다고 하자. scheduler가 둘을 같은 mixed batch에 넣으면 flattened query rows는 136이다. 그러나 kernel backend가 prefill과 decode를 서로 다른 launcher로 분리하면 실제 launch는 `M=128`과 `M=8` 두 개가 된다. 반대로 unified kernel family가 metadata로 mode를 구분하면 하나의 work queue로 보낼 수도 있다. “batch size 9”라는 API 숫자에서 grid를 직접 계산하면 이 분기부터 놓친다.

### 40.1.2 shape 번역에는 다섯 명의 소유자가 있다

첫째, scheduler가 이번 step의 sequence와 token budget을 소유한다. 둘째, model runner가 선택된 tokens를 contiguous 또는 padded tensor로 포장하고 query/KV metadata를 만든다. 셋째, attention backend가 dtype, head dimension, causal/varlen, architecture와 library availability를 검사해 native entry point를 고른다. 넷째, native dispatcher가 template traits와 kernel specialization을 정하고 grid, block, shared-memory bytes를 계산한다. 다섯째, kernel 또는 device tile scheduler가 `blockIdx`와 `threadIdx`를 실제 work coordinate로 해석한다.

각 경계의 출력은 다음 경계의 입력이지만 의미가 그대로 유지된다고 가정해서는 안 된다. Python의 `num_tokens`가 native `batch_size`라는 이름으로 들어갈 수 있고, native launcher의 `num_m_blocks`는 `ceil_div(M,BLOCK_M)`일 수 있다. persistent kernel은 launch grid를 logical tile count가 아니라 SM 수로 잡고 device 안에서 work tile을 반복해서 가져갈 수 있다. split-KV는 한 query tile을 여러 KV chunks로 복제해 reduction 전 partial outputs를 만들 수 있다.

따라서 소스를 읽을 때 변수 이름 하나를 따라가는 것보다 불변식을 따라가는 편이 안전하다. 입력 row interval은 누락이나 중복 없이 output row에 대응해야 한다. tail 밖의 row는 load와 store가 predicate로 보호돼야 한다. split이 있다면 partial result가 올바른 query/head와 결합돼야 한다. persistent work queue라면 모든 logical tile이 정확히 한 번 claim돼야 한다. 이 불변식이 shape 번역의 correctness 계약이다.

### 40.1.3 배치는 launch amortization과 tile 충전이라는 두 문제를 푼다

여러 요청을 모으면 Python 호출, dispatcher, kernel launch 같은 고정 비용을 더 많은 token에 나눌 수 있다. 동시에 큰 `M`은 row tile의 빈 공간을 줄일 수 있다. 하지만 두 효과는 같은 것이 아니다. CUDA Graph replay로 launch overhead를 줄여도 `M=8`, `BLOCK_M=64`의 logical tail은 남는다. 반대로 `M=128`이 tile을 꽉 채워도 지나치게 많은 작은 kernels와 host synchronization이 있으면 launch overhead가 남는다.

continuous batching이 중요한 까닭도 여기서 구체화된다. 종료한 sequence 자리에 새 request를 넣는 것은 단순히 “GPU를 바쁘게” 하는 행위가 아니다. 다음 forward의 `M`, per-sequence KV length 분포, head grouping과 tile count를 바꾼다. scheduler 정책이 kernel shape distribution을 만든다. 따라서 scheduler 개선을 판단하려면 queue-level goodput와 함께 downstream launch family와 tile fill을 보아야 한다.

batch를 크게 만드는 것도 무조건 답은 아니다. 더 많은 rows는 CTA 수를 늘리지만 KV cache bytes, metadata, latency waiting time도 늘린다. prefill rows가 decode critical path와 같은 resource를 쓰면 ITL이 악화될 수 있다. 이 장은 scheduler 정책 자체보다, 결정된 batch가 GPU execution shape로 변하는 마지막 번역을 소유한다.

## 40.2 grid·block 좌표에서 warp 실행 묶음까지 내려간다

### 40.2.1 execution configuration은 좌표계를 만든다

CUDA kernel launch의 execution configuration은 grid dimension과 thread-block dimension을 지정한다. CUDA 12.9.1 Programming Guide §5.2는 `threadIdx`가 3-component vector이며 1D, 2D, 3D thread block을 표현한다고 설명한다. CUDA 13.3.0 Programming Guide §1.2.2.1도 thread들이 block으로, block들이 grid로 조직되고 grid의 모든 block이 같은 크기와 차원을 가진다고 설명한다. 여기서 CTA(cooperative thread array)는 보통 CUDA thread block과 같은 실행 단위를 가리킨다.

`dim3 grid(gx,gy,gz)`와 `dim3 block(bx,by,bz)`가 주어지면 launch된 CTA 수는 `gx×gy×gz`, CTA당 thread 수는 `bx×by×bz`다. 총 launch thread instance는 두 값을 곱한 수다. 하지만 이것은 동시에 resident하는 thread 수가 아니다. CTA들은 SM 자원과 scheduler에 따라 waves로 실행될 수 있다. grid 크기는 logical work supply이고 residency는 hardware resource 배치다.

kernel 안에서 `blockIdx.{x,y,z}`는 CTA 좌표, `threadIdx.{x,y,z}`는 CTA 내부 thread 좌표다. `blockDim`과 `gridDim`은 launch dimensions를 알려 준다. 1D 배열의 흔한 indexing인 `i=blockIdx.x*blockDim.x+threadIdx.x`는 가능한 매핑 하나일 뿐 보편 법칙이 아니다. attention kernel은 `blockIdx.x`를 query tile, `blockIdx.y`를 split, `blockIdx.z`를 KV head로 쓸 수 있고, thread들이 matrix fragment 하나를 협력 처리할 수 있다.

### 40.2.2 warp는 좌표축이 아니라 실행 묶음이다

thread block의 threads는 선형 thread ID로 정렬된 뒤 warp로 묶인다. 3D block에서 x가 가장 빨리 변하고 그 다음 y, z가 변한다. NVIDIA CUDA에서 warp는 32 threads로 다뤄지므로 `blockDim.x=128`, y=z=1이면 네 warp가 생긴다. 그러나 source가 literal 32를 쓰는지, `warpSize`, architecture traits 또는 library 상수를 쓰는지 구별해야 한다.

lane은 warp 안의 thread 위치다. warp 0의 lanes 0–31, warp 1의 lanes 0–31처럼 생각할 수 있다. SIMT는 같은 warp의 threads가 같은 instruction stream을 함께 진행하도록 만들지만 predicate와 branch에 따라 일부 lanes만 instruction 효과를 낼 수 있다. divergence는 서로 다른 control path가 serialization되는 문제이며, logical tile의 빈 row와 동의어가 아니다.

warp-specialized kernel에서는 warp마다 역할도 다르다. 어떤 warp group은 global memory에서 K/V tile을 옮기는 producer이고 다른 warp group은 tensor-core MMA를 수행하는 consumer일 수 있다. 이때 “producer warp가 output row를 계산하지 않는다”는 사실을 idle이라 부르면 틀린다. 그 warp는 pipeline의 다른 필수 일을 한다. utilization을 말하려면 lane이 어떤 instruction에서 active였는지, 어떤 resource가 병목인지까지 내려가야 한다.

### 40.2.3 occupancy는 CTA 개수의 별명이 아니다

occupancy는 보통 SM에 resident한 active warps와 그 SM이 지원할 수 있는 maximum active warps의 비율로 다룬다. 한 CTA의 thread 수, thread당 registers, CTA당 shared memory, architecture limit가 동시에 resident할 CTA와 warp 수를 제한한다. grid에 CTA가 하나뿐이면 여러 SM에 공급할 일이 부족할 수 있지만, 그 사실만으로 해당 CTA가 놓인 SM의 occupancy를 숫자로 확정할 수 없다.

반대로 occupancy가 높아도 빠르다는 보장은 없다. memory dependency로 warp들이 기다리거나, useful math가 적거나, instruction mix가 병목일 수 있다. attention에서는 query tile이 작아도 긴 KV를 순회하며 많은 일을 할 수 있다. 높은 occupancy를 목표로 tile을 줄이면 data reuse가 나빠지고 launch/partial reduction가 늘 수 있다. occupancy는 원인 후보를 좁히는 resource 지표이지 성능 목적 함수 자체가 아니다.

이 구분은 장애 보고 문장을 바꾼다. “decode는 CTA 하나라 occupancy가 3%다”라고 쓰지 않는다. 대신 “이 launch의 grid는 CTA 하나여서 device 전체에 공급할 independent CTA가 부족할 가능성이 있다. selected kernel의 registers/shared memory와 device limit를 사용해 residency를 별도로 계산하고, actual active warps는 profiler에서 검증해야 한다”고 쓴다.

## 40.3 warp의 tail·idle lane·occupancy를 작은 M으로 검산한다

### 40.3.1 교육용 tile 계약

교육용 kernel은 logical row dimension `M`을 64-row tile로 자른다. static scheduler라면 `grid.x=ceil_div(M,64)`다. CTA `b`의 logical interval은 `[64b,min(64(b+1),M))`이고, tile의 후보 row는 `row=64b+r`로 계산한다. `row<M`인 경우에만 logical data를 load/store한다. CTA당 128 threads, 네 warp를 launch한다고 하자.

prefill `M=128`에서는 `ceil_div(128,64)=2`이므로 CTA 두 개가 생긴다. CTA 0은 rows `[0,64)`, CTA 1은 `[64,128)`을 맡는다. row-slot capacity는 128, valid rows도 128이므로 tail row는 없다. launch instances는 2×128=256 threads, 여덟 warps다. 여기까지는 산술 사실이다. 두 CTA가 동시에 실행되는지, 각 thread가 어느 row/feature를 맡는지는 아직 결정되지 않았다.

decode `M=8`에서는 `ceil_div(8,64)=1`이므로 CTA 하나가 생긴다. tile interval은 `[0,64)`지만 valid logical interval은 `[0,8)`이다. row-slot capacity 64 중 valid row fraction은 8/64, 즉 12.5%이고 invalid tail row slots는 56이다. launch instances는 128 threads, 네 warps다. 여기서 56은 logical row slots의 수이지 idle CUDA lanes의 수가 아니다.

### 40.3.2 idle row를 idle lane으로 바꾸려면 indexing 증거가 필요하다

첫 번째 가상 kernel이 row-per-thread라고 하자. threads 0–63만 row candidates이고 `row=64*blockIdx.x+threadIdx.x`를 사용하며 threads 64–127은 다른 feature 일을 하지 않는다고 가정한다. `M=8`에서는 candidate threads 중 0–7만 valid row를 가진다. 이 가정 아래에서는 row predicate가 많은 lanes의 memory operation을 막는다. 그래도 warp 단위 control flow와 threads 64–127의 실제 코드를 읽어야 exact active instruction 비율을 말할 수 있다.

두 번째 kernel은 128 threads가 여덟 rows의 head dimension과 K/V tiles를 협력 처리한다고 하자. 한 row의 query vector를 여러 lanes가 load하고, 여러 warps가 KV chunks를 순회하고, reduction를 합친다. logical rows 56개가 비어 있어도 네 warp 모두 유효한 여덟 rows를 위해 일할 수 있다. 이때 underfill은 “row 방향으로 tile reuse와 output 수가 적다”는 뜻이지 “87.5% lanes가 잠든다”는 뜻이 아니다.

세 번째 kernel은 fixed tile조차 쓰지 않고 persistent work queue를 돈다. launch CTA 수는 SM 수에 맞추고 각 CTA가 atomic counter 또는 tile scheduler에서 다음 logical tile을 가져온다. `grid.x`는 `ceil_div(M,64)`가 아니며 block 0이 row tile 0이라는 등식도 깨진다. logical tile count와 physical worker CTA count를 별도로 세어야 한다.

### 40.3.3 경계값 63·64·65가 off-by-one을 드러낸다

`M=63`이면 grid는 1이고 tail은 한 row다. `M=64`면 grid는 1이고 tail은 0이다. `M=65`면 grid는 2이고 두 번째 tile의 valid row는 하나, tail은 63이다. 평균적인 큰 M만 시험하면 마지막 CTA의 predicate bug를 놓친다. 63/64/65는 ceil-div, launch count, load bounds와 output store bounds가 같은 logical frontier를 쓰는지 확인하는 최소 fixture다.

tile 크기도 비교할 수 있다. decode `M=8`에서 `BLOCK_M=16,32,64,128`의 valid row fraction은 각각 50%, 25%, 12.5%, 6.25%다. 그렇다고 16이 가장 빠르다고 결론 내릴 수 없다. 작은 tile은 CTA 수와 scheduling flexibility를 늘릴 수 있지만 K/V reuse, tensor-core shape, shared-memory staging, reduction 횟수와 specialization availability를 바꾼다. 손계산은 후보 trade-off를 드러낼 뿐 optimal kernel을 증명하지 않는다.

head axis를 넣으면 static grid 예는 `ceil_div(M,BLOCK_M)×H` CTAs가 된다. `M=8`, `H=8`, `BLOCK_M=64`라면 row tile은 하나라도 head별 CTA를 쓰는 설계에서는 여덟 CTAs가 생긴다. grouped-query attention에서 query heads와 KV heads가 다르면 어느 head count가 grid axis인지도 확인해야 한다. `grid.x`만 보고 device parallelism이 하나라고 단정할 수 없다.

### 40.3.4 63·64·65를 load와 store 양쪽에서 검산한다

손계산을 grid에서 멈추지 말고 각 CTA의 후보 interval과 predicate 결과까지 쓴다. M=63에서 CTA 0의 후보는 0–63이고 valid는 0–62다. candidate row 63은 `63<63`이 거짓이므로 load와 output store가 모두 막혀야 한다. M=64에서는 같은 candidate row 63이 valid다. 경계 비교의 핵심은 소스 한 줄 차이가 아니라 predicate가 받는 actual M과 비교 연산이다.

M=65에서는 CTA 0의 후보 0–63이 전부 valid이고 CTA 1의 후보 64–127 중 row 64만 valid다. 두 번째 CTA의 local row 0이 global row 64로 바뀌는 식은 `64*blockIdx.x+local_row`다. 만약 output pointer offset에서 blockIdx factor를 빠뜨리면 CTA 1이 row 0을 덮어쓸 수 있다. 반대로 factor를 bytes와 elements 단위로 혼동하면 훨씬 먼 주소를 쓴다.

load guard와 store guard가 같은 조건일 필요는 있지만 같은 코드 위치일 필요는 없다. tiled algorithm은 invalid query rows에 zero 또는 neutral values를 채워 mainloop를 진행하고 epilogue에서 store를 막을 수 있다. K/V tail은 softmax mask에 `-∞` 성격의 값을 넣어 확률 기여를 없앨 수 있다. predicate가 있다는 사실만 확인하지 말고 invalid element가 reduction에 미치는 중간 의미까지 확인한다.

M=63에서 row 63을 읽었지만 store만 막으면 out-of-bounds read가 생길 수 있다. allocator padding 때문에 당장 crash하지 않아도 잘못된 값이 reduction나 NaN에 영향을 줄 수 있다. M=65에서 load는 row 64를 읽었지만 store predicate가 `row<64` 같은 tile boundary를 쓰면 마지막 output만 빠진다. input와 output checkpoints를 나누는 이유다.

varlen에서는 global M predicate 외 sequence-local predicate가 있다. flattened row 64가 두 번째 sequence의 첫 row라면 causal position은 global 64가 아니라 해당 sequence의 query offset과 KV length에서 계산된다. total M boundary가 맞아도 indptr lookup가 이전 sequence를 가리키면 memory-safe한 오답이 난다. M=64+1과 single-sequence M=65를 나란히 비교해야 한다.

### 40.3.5 row-slot 표를 사람이 읽을 수 있는 이야기로 바꾼다

64석짜리 셔틀을 비유로 사용할 수 있다. M=128은 두 대가 64명씩 태우고 M=8은 한 대가 여덟 명만 태운다. 이 비유는 ceil-div와 tail capacity를 설명하는 데 유용하다. 그러나 좌석이 CUDA lane이라고 말하는 순간 비유가 깨진다. 실제 kernel의 한 승객, 즉 query row를 여러 workers가 함께 처리할 수 있고 workers는 짐 운반, 계산, 합산처럼 서로 다른 역할을 맡기 때문이다.

또한 셔틀 비유는 split-KV를 설명하지 못한다. 같은 여덟 승객의 짐을 네 구간에서 나눠 처리하면 여러 CTA가 동일 query rows를 위해 일한다. persistent scheduler에서는 고정 노선 셔틀이 아니라 workers가 중앙 작업표에서 다음 tile을 가져간다. cluster에서는 여러 차량이 하나의 협력 단위로 묶일 수 있다. 비유는 static row capacity까지만 쓰고 실제 역할은 source로 돌아온다.

이 한계를 독자에게 먼저 밝히면 직관과 정확성이 충돌하지 않는다. “M=8은 64-row tile을 덜 채운다”는 강한 사실을 유지하면서, “그러므로 lanes 56개가 idle”이라는 근거 없는 결론을 막는다. 좋은 비유는 설명을 끝내는 것이 아니라 어디서 코드를 읽어야 하는지 알려 준다.

### 40.3.6 tile utilization이라는 말을 쓸 때 분모를 붙인다

`8/64=12.5%`를 쓸 때는 “query row-slot utilization”이라고 부른다. arithmetic operations utilization, warp execution efficiency, SM utilization, memory bandwidth utilization와 구분한다. 같은 percentage 단위를 쓴다고 같은 metric이 아니다. 분모가 logical capacity인지 hardware cycles인지 먼저 쓴다.

row-slot utilization는 tile selection의 한 trade-off를 보여 준다. 작은 M 분포가 많으면 더 작은 BLOCK_M specialization가 유리할 가능성이 있지만, tensor-core fragment, K/V reuse와 launch count가 함께 변한다. 따라서 workload histogram에서 M buckets를 보고 supported kernel families의 tile shapes와 연결한 뒤 후보를 세운다. row-slot fraction만으로 kernel을 교체하지 않는다.

prefill에서도 tail이 없다고 완전 활용이라 하지 않는다. M tile은 꽉 찼어도 sequence별 causal triangle 때문에 일부 query-key pairs가 masked되고, head dimension/KV length tail이 있을 수 있다. 40장은 M axis만 계산했음을 명시하고 45장에서 attention tile과 online softmax의 다른 축을 다룬다.

## 40.4 Python tensor에서 native launcher까지 무엇을 기록하는가

### 40.4.1 shape ledger는 이름보다 의미를 보존한다

요청 R을 추적할 때 HTTP request ID에서 바로 CUDA block ID로 점프하지 않는다. scheduler step ID, model runner batch row interval, backend call ID, native launch ID를 연결한다. 각 단계에서 `M`, sequence count, query heads, KV heads, head dimension, per-sequence query/KV lengths, dtype, causal/window, split count를 기록한다. privacy 때문에 token content를 남기지 않아도 shape와 pseudonymous correlation ID는 남길 수 있다.

Python tensor의 shape와 stride도 필요하다. `[M,H,D]`라고 표시돼도 contiguous layout인지, view/reshape가 copy를 만들었는지, packed QKV인지 separate Q/K/V인지가 native dispatcher 조건을 바꿀 수 있다. dtype과 alignment가 specialization 선택에 들어가며 head dimension가 지원 집합 밖이면 fallback 또는 validation error가 날 수 있다. “같은 M인데 launch가 달라졌다”면 먼저 이 dispatch key를 비교한다.

metadata는 tensor보다 작지만 의미를 결정한다. cumulative query lengths는 flattened row를 sequence interval로 나누고, cumulative KV lengths는 attention 범위를 정한다. page/block table은 logical KV position을 physical cache로 번역한다. split-KV metadata는 partial work와 merge workspace를 정의한다. launcher만 보면 grid 계산식은 보여도 각 축이 왜 그 수가 됐는지 알 수 없으므로 Python call arguments까지 거슬러 올라간다.

### 40.4.2 dispatcher는 성능 스위치이기 전에 유효성 경계다

backend selection은 “가장 빠른 kernel을 고른다”로 요약하기 쉽다. 실제로는 먼저 지원 가능성을 판정한다. device capability, dtype, head size, causal/varlen, sliding window, return-LSE, soft cap, quantization와 library version이 native path의 전제다. 전제가 맞지 않으면 다른 version, 다른 backend 또는 오류로 간다.

같은 함수 이름 아래에서도 template specialization가 갈린다. head dimension, element type, tile size, architecture와 flags가 compile-time traits를 정할 수 있다. dynamic values인 M, sequence lengths와 stream은 runtime parameters가 된다. 이 경계를 알아야 option이나 model shape가 어느 상태를 바꾸는지 설명할 수 있다. head size 변경은 단순 tensor column 수 변경이 아니라 지원 specialization와 registers/shared-memory footprint를 함께 바꿀 수 있다.

launcher는 grid와 block만 만들지 않는다. kernel params의 lifetime, workspace, dynamic shared-memory bytes, stream, optional function attribute와 cluster dimensions를 준비할 수 있다. launch return은 asynchronous일 수 있으므로 Python 함수가 돌아왔다는 사실을 kernel 완료로 읽지 않는다. 이 장은 stream ordering의 세부를 43장에 넘기지만, launch record에는 stream과 workspace ownership를 남긴다.

### 40.4.3 같은 M이 같은 launch를 만들지 않는 네 장면

첫 장면은 head 수가 다르다. 두 request batch 모두 M=8이고 D=128이지만 하나는 query heads 32, KV heads 8인 GQA이고 다른 하나는 query/KV heads가 모두 32라고 하자. kernel이 KV head를 grid.z에 두고 한 KV head가 여러 query heads를 처리한다면 grid와 내부 group mapping가 달라질 수 있다. M만 같다고 같은 CTA 수를 기대할 수 없다.

둘째 장면은 KV length가 다르다. 두 decode batch 모두 M=8이지만 첫 batch의 각 context가 128이고 둘째는 32,768이라고 하자. query tile 수는 같아도 K/V loop trip count와 split-KV 선택이 달라질 수 있다. 둘째 batch가 splits를 사용하면 grid의 chunk axis와 merge workspace가 추가된다. “decode batch size 8”은 device work를 특정하기에 불충분하다.

셋째 장면은 dtype과 architecture가 다르다. 동일 tensor dimensions라도 bf16, fp8 또는 quantized cache path가 서로 다른 traits와 kernel family를 선택할 수 있다. SM90과 Blackwell용 specialization는 TMA, warp-group 역할, tile shape와 persistent scheduler availability가 다를 수 있다. CUDA의 `blockIdx` 의미는 그대로지만 library가 정한 축 mapping와 block shape는 달라진다.

넷째 장면은 output 요구가 다르다. attention output만 필요한 호출과 log-sum-exp도 반환하는 호출, causal과 non-causal, sliding window 유무는 dispatcher flags와 workspace를 바꿀 수 있다. 이 옵션들은 HTTP에 직접 드러나지 않아도 sampling/logprob 또는 model architecture에서 파생될 수 있다. 따라서 launch 비교 표에는 effective flags를 넣는다.

이 네 장면의 교훈은 “shape”를 dimensions만으로 정의하지 않는 것이다. 실행 shape는 logical dimensions와 layout, dtype, mode, architecture, output contract를 합친 dispatch state다. incident에서 M만 맞는다고 shape regression를 배제하지 않고, 반대로 grid가 다르다고 M corruption을 단정하지 않는다.

### 40.4.4 padding capacity와 actual rows를 섞으면 생기는 일

CUDA Graph나 compile bucket을 위해 capacity M을 128로 고정하고 actual rows가 8인 tensor를 replay할 수 있다. 이때 Python tensor storage shape는 `[128,H,D]`처럼 보이지만 metadata의 `num_actual_tokens`는 8일 수 있다. kernel이 capacity grid를 launch하고 actual-length predicate로 rows 8–127을 막는 설계도 가능하며, compact input을 별 kernel로 처리하는 설계도 가능하다.

이 장의 M을 actual logical rows로 정의한 이유가 여기 있다. capacity와 actual을 모두 `num_tokens`라고 부르면 `grid=2`인데 M=8이라는 모순처럼 보인다. 실제로는 captured capacity tiles가 두 개이고 valid logical rows가 여덟일 수 있다. launch shape 원장에는 `M_actual`, `M_capacity`, captured/replay flag를 분리한다.

stale actual length는 correctness 사고를 만든다. 이전 replay에서 actual 128이었고 현재 8인데 predicate가 이전 값을 읽으면 invalid rows가 output 또는 KV cache를 오염시킬 수 있다. 반대로 actual을 너무 작게 읽으면 valid rows가 누락된다. graph replay 뒤 M boundary에서만 오답이 난다면 captured pointer뿐 아니라 actual-length buffer와 update ordering를 본다.

capacity padding 때문에 많은 CTA가 launch됐다고 곧 낭비율을 계산하지 않는다. invalid tiles가 device scheduler에서 빠르게 return하는지, kernel 안에서 어느 지점까지 진행하는지에 따라 cost가 다르다. graph가 줄인 host overhead와 padded work cost도 별 항이다. runtime trace 없이 어느 쪽이 우세하다고 쓰지 않는다.

### 40.4.5 3차원 좌표를 종이에 펼치는 연습

교육용으로 `grid=(2,3,4)`를 생각하자. CUDA가 보장하는 것은 x/y/z 좌표 범위뿐이다. 총 CTA는 24다. library가 x=query tile, y=KV split, z=KV head라고 정했다면 query tiles 2개 각각에 splits 3개와 KV heads 4개가 조합된다. CTA `(1,2,3)`은 두 번째 query tile, 세 번째 split, 네 번째 KV head를 처리한다.

다른 specialization가 x=head, y=query tile, z=batch라고 정하면 숫자가 같아도 CTA `(1,2,3)`의 의미는 완전히 다르다. grid tuple만 저장한 trace는 source version와 scheduler symbol 없이 해석할 수 없다. 운영 도구가 축에 고정 라벨을 붙인다면 kernel family별 mapping registry가 필요하다.

block도 같은 원리다. `block=(32,4,1)`이면 128 threads지만 x-fastest linearization에 따라 linear thread ID는 `x+32*y`다. warp 0은 y=0의 x 0–31, warp 1은 y=1의 x 0–31이다. kernel이 y를 warp role로 쓰면 각 warp가 서로 다른 row group 또는 pipeline stage를 맡을 수 있다. `threadIdx.x`만 기록하면 role을 놓친다.

반대로 `block=(128,1,1)`이면 warp ID는 보통 linear ID를 32로 나눈 값으로 구할 수 있다. 그러나 kernel source가 warp group 크기와 named barriers를 어떻게 구성하는지 읽어야 producer/consumer 의미를 안다. 좌표 산술은 역할 발견의 출발점이지 역할 자체가 아니다.

## 40.5 static grid를 깨뜨리는 세 반례

### 40.5.1 split-KV: CTA가 output tile보다 많아진다

decode `M=8`과 row tile 64만 보면 query tile은 하나다. 그러나 KV length가 길고 이를 네 chunks로 나누며 KV heads가 여덟이라고 가정하면 rectangular work supply는 `1 query tile × 4 splits × 8 KV heads=32` partial CTAs가 될 수 있다. 실제 축 배치와 CTA 수는 specialization에 달렸지만, 이 손계산은 query tile 하나가 CTA 하나라는 등식이 깨지는 방식을 보여 준다.

각 split은 동일 query rows를 읽되 서로 다른 KV interval을 처리한다. online softmax의 partial statistics와 output accumulator를 merge해야 최종 attention output이 된다. 그러므로 output store owner가 partial kernel인지 merge kernel인지 확인한다. partial CTA가 최종 output을 직접 덮어쓰면 race이고, workspace index의 split/head 좌표가 틀리면 다른 partial이 섞인다.

장애 fixture는 query boundary와 split boundary를 따로 움직인다. M은 8로 고정하고 KV length를 chunk size의 `k-1,k,k+1`로 바꾼다. 이어 KV length를 고정하고 M을 63/64/65로 바꾼다. 두 경계를 동시에 하나씩만 바꾸면 어느 predicate가 틀렸는지 좁힐 수 있다. final output divergence가 merge 뒤 처음 생기는지 partial workspace부터 생기는지도 나눈다.

split heuristic이 바뀌면 같은 request shape에서도 grid가 달라질 수 있다. toolkit upgrade가 원인처럼 보일 수 있지만 실제로는 library version, workspace capacity, batch shape 또는 SM count heuristic 변화일 수 있다. selected split count와 scheduler symbol을 기록하지 않은 채 CUDA version만 비교하지 않는다.

### 40.5.2 persistent scheduling: CTA가 여러 tile을 순회한다

static scheduler에서는 block index와 work tile 사이에 거의 직접적인 함수가 있다. persistent scheduler에서는 CTA가 loop를 돌며 next tile을 claim한다. 따라서 first tile은 block index와 관계가 있어도 두 번째 이후 work는 queue state에 달릴 수 있다. block 하나에서 여러 output intervals가 나온다.

correctness 불변식은 queue가 모든 valid tiles를 정확히 한 번 내주는 것이다. 누락되면 output row가 미작성 상태로 남고, 중복 claim이면 두 CTAs가 같은 destination를 쓰거나 불필요한 work를 한다. terminal sentinel를 너무 일찍 반환하면 tail tiles가 빠지고 너무 늦으면 out-of-range metadata를 읽는다. static kernel의 `row<M` 하나보다 scheduler state까지 검사 범위가 넓다.

관측도 달라진다. grid dimensions만 기록하는 launch trace는 부족하다. scheduler mode, initial work count, split/head dimensions, persistent worker count를 함께 기록한다. device-side claim 순서를 모두 운영 로그에 남기는 것은 비쌀 수 있으므로 deterministic 작은 fixture에서 sampled tile IDs 또는 output coverage bitmap을 사용할 수 있다. 이 장에서는 실행하지 않지만 어떤 증거가 가설을 가르는지 명시한다.

persistent grid가 `num_sms`에 비례한다고 해서 CTA 하나가 SM 하나에 영구 고정된다고 단정하지 않는다. CUDA scheduler가 CTA를 SM에 배치하며 resource와 cluster constraints가 영향을 준다. 이름이 persistent인 것은 kernel이 여러 logical works를 처리하는 실행 패턴을 가리키지, source 확인 없이 exact residency를 보장하는 표현이 아니다.

### 40.5.3 thread-block cluster: block 위에 선택적 협력 계층이 생긴다

CUDA thread-block cluster는 block들이 더 넓은 협력 범위를 갖도록 하는 선택적 hierarchy다. 기본 grid→block→thread 의미를 없애지 않는다. launcher는 cluster dimensions와 cluster launch API를 사용할 수 있고, kernel은 cluster synchronization나 distributed shared memory 같은 기능을 쓸 수 있다. 지원 compute capability와 portable/architecture-specific limits를 확인해야 한다.

cluster를 쓰는 kernel에서 grid의 CTA 수가 cluster size의 배수인지, tail cluster가 허용되는지, launch attribute가 어떻게 설정되는지 본다. cluster당 blocks가 함께 schedule되어야 한다면 shared-memory/register 자원과 device configuration가 residency에 더 강한 제약을 줄 수 있다. block 하나의 resource 계산만으로 resident clusters를 추정할 수 없다.

MIG 또는 architecture 조건에서 requested cluster size가 지원 범위를 벗어나면 launch failure 또는 다른 path 선택이 생길 수 있다. 이를 request batch OOM으로 분류하지 않는다. selected kernel traits, cluster dims, device capability와 launch error를 함께 본다. cluster path가 비활성화돼 non-cluster fallback으로 갔다면 grid가 달라지는 것은 정상 분기일 수 있다.

CUDA 12.9.1과 13.3.0의 기본 hierarchy 설명을 교차 확인하되, architecture feature availability와 toolkit/driver compatibility를 같은 표로 섞지 않는다. CUDA Compatibility 13.0.2는 44장에서 별도로 다룬다. 40장에서 필요한 결론은 optional hierarchy가 source launcher의 execution configuration를 확장하지만 HTTP row가 CTA가 되는 기본 번역 문제를 대신 해결하지 않는다는 것이다.

## 40.6 launch·resource 사고를 첫 divergence로 조사한다

**사건 A: decode가 예상보다 느리다**

증상은 prefill `M=128`보다 decode `M=8`의 token당 시간이 지나치게 길다는 것이다. 첫 가설은 row tile underfill이다. 경쟁 가설은 긴 KV read, 다른 decode kernel family, split/merge overhead, insufficient independent tiles, launch/host overhead, synchronization와 memory dependency다. “M이 작아서 GPU가 쉰다” 하나로 닫지 않는다.

먼저 Python shape ledger에서 M, sequence count, per-sequence KV lengths, heads와 D를 확인한다. 다음으로 selected backend/version, split count와 persistent flag를 확인한다. launcher record에서 grid, block, shared bytes, cluster dims와 stream을 본다. 마지막으로 kernel traits와 indexing source를 연결한다. logical row fill은 손으로 계산하되 lane activity와 occupancy는 측정 없이 수치화하지 않는다.

분기는 명확하다. M=8이고 같은 static family, `BLOCK_M=64`라면 row underfill가 존재한다. 그러나 split-KV로 CTAs가 늘었다면 device work supply는 query tile 하나보다 크다. persistent scheduler라면 grid에서 tile count를 읽지 않는다. decode 전용 family라면 prefill과 tile size·warp roles를 직접 비교한다. KV length가 크게 다르면 M만 고정한 비교가 아니다.

검증은 한 변수씩 움직인다. 동일 backend 조건에서 M을 8, 16, 32, 64로 바꾸고 launch family와 tile constants가 유지되는 범위만 비교한다. KV length를 별도로 고정하거나 buckets로 나눈다. runtime 실험을 수행할 때는 TTFT/ITL, kernel duration, launch count와 profiler activity를 함께 봐야 하지만, 이 장은 source와 fixture 설계까지만 제공한다.

**사건 B: M=65에서만 오답이 난다**

64까지 맞고 65에서 깨지면 두 번째 CTA와 tail boundary가 처음 등장한다. 가장 먼저 `ceil_div`와 grid count를 확인하고 마지막 CTA의 load/store predicate가 같은 M을 쓰는지 본다. padding capacity를 actual valid length로 오인하거나 output stride를 tile capacity로 계산했을 수 있다.

관측은 first bad row를 찾는다. rows 0–63이 맞고 row 64부터 틀리면 block 1 mapping가 강한 후보다. row 64의 input load는 맞지만 output store address가 틀리면 epilogue 문제다. partial workspace부터 틀리면 mainloop 또는 scheduler coordinate다. final text만 비교하면 sampler까지 긴 경로를 거쳐 원인을 흐린다.

varlen batch에서는 total M=65뿐 아니라 per-sequence boundaries를 바꾼다. 한 sequence 65와 sequences 64+1은 같은 total M이지만 cumulative lengths와 causal ranges가 다르다. 둘 다 실패하면 global tile boundary, 하나만 실패하면 sequence indptr/varlen mapping를 의심한다. heads와 splits를 1로 줄일 수 있는 supported fixture라면 축 혼동을 좁힌다.

predicate를 제거해 crash를 재현하는 방식은 안전하지 않고 필요하지 않다. fixed source에서 predicate의 input과 output address를 추적하고, guard 전후 selected coordinates를 제한적으로 관측한다. out-of-bounds가 allocator corruption로 늦게 드러날 수 있으므로 처음 이상한 row와 launch에 집중한다.

**사건 C: invalid configuration 또는 launch failure**

증상은 kernel launch가 invalid configuration, too many resources requested 같은 오류로 실패하거나 backend가 fallback하는 것이다. KV capacity OOM와 구분한다. threads per block이 device limit를 넘었는지, static+dynamic shared memory가 limit/opt-in 조건을 넘었는지, cluster dimensions가 지원되는지, selected architecture binary가 존재하는지 본다.

관측 record에는 kernel symbol과 specialization, block dimensions의 곱, shared bytes, function attribute 설정 결과, device compute capability와 cluster dims를 둔다. Python model config의 head size/dtype도 남긴다. head size 하나가 traits를 바꾸어 block threads나 shared storage를 키웠을 수 있기 때문이다.

분기는 launch 전 validation failure, CUDA API launch return, asynchronous error report를 나눈다. 이전 kernel 오류가 뒤 API에서 보고될 수 있으므로 correlation와 error-check boundary가 중요하다. whole-device synchronize를 상시 해결책으로 넣지 않는다. 진단에서 exact failing launch를 좁힌 뒤 올바른 stream/error boundary로 확인한다.

검증은 지원 specialization로 돌아갔을 때 단순히 오류가 사라졌는지만 보지 않는다. expected backend가 선택됐는지, fallback이 correctness를 유지하는지, resource record가 limit 안으로 들어왔는지 확인한다. shared memory를 줄이는 변경이 tile algorithm와 output을 바꿀 수 있으므로 source traits와 함께 검토한다.

**사건 D: occupancy는 높은데 throughput이 낮다**

이 증상은 occupancy가 목적 함수가 아니라는 대표 사례다. resident warps가 많아도 global memory dependency, shared-memory bank conflict, divergence, instruction dependency, underfilled logical work 또는 너무 적은 total CTAs가 병목일 수 있다. occupancy metric 하나로 register count를 줄이는 수정부터 하지 않는다.

먼저 metric의 분모와 scope를 확인한다. achieved occupancy인지 theoretical occupancy인지, 특정 kernel인지 device interval aggregate인지 구분한다. kernel별 registers/thread, shared bytes/CTA, block threads와 grid를 함께 본다. 다음 장에서는 bytes movement와 roofline 관점으로 memory-bound 여부를 다룬다.

경쟁 가설은 관측이 다르다. total CTAs 부족이면 grid supply와 SM distribution가 문제다. dependency stall이면 active warps가 있어도 eligible instruction가 부족하다. tail predicate면 특정 boundary shapes에서 useful output per launch가 급락한다. split merge overhead면 partial/merge launch와 workspace traffic이 늘어난다. 같은 “높은 occupancy” 아래 서로 다른 first divergence가 있다.

검증은 optimization이 바꾼 상태를 명시한다. tile 축소가 occupancy를 높였다면 data reuse와 CTA count, shared bytes도 바뀐다. register cap은 spill을 만들 수 있다. batch 확대는 queue latency와 M distribution를 바꾼다. throughput 상승 하나로 causal claim을 만들지 않고 변경된 launch/resource/bytes를 함께 기록한다.

**사건 E: CUDA 또는 library upgrade 뒤 grid가 달라졌다**

toolkit 12.x에서 13.x로 바꾼 뒤 grid가 달라졌다는 관측은 기본 CUDA hierarchy 변화의 증거가 아니다. framework/library commit, compiled architecture targets, CUTLASS/FlashAttention/FlashInfer version, backend availability와 heuristic가 함께 바뀌었을 수 있다. 먼저 software artifact를 고정한다.

동일 vLLM/SGLang commit과 동일 extension binary를 정말 비교하는지 확인한다. binary 안의 target architecture와 JIT 여부는 44장 범위지만 selected kernel symbol은 이 장에서도 필요하다. source dispatcher가 architecture predicate로 SM90/SM100 path를 바꿨다면 grid 변화는 그 분기의 결과다.

logical shape ledger가 같고 selected specialization도 같은데 grid만 다르면 launcher params와 compiled constants를 비교한다. specialization가 다르면 각 `get_grid_shape`를 따로 읽는다. persistent flag나 split count가 다르면 static equation 비교를 중단한다. output correctness와 resource error가 없다면 grid 숫자 변화 자체를 regression이라고 부르지 않는다.

**복구 완료는 원래 숫자로 돌아오는 것이 아니다**

launch 사고를 고친 뒤 grid가 과거 값으로 돌아왔다고 완료하지 않는다. 과거 grid 해석 자체가 틀렸을 수 있고 새 specialization가 정상적으로 다른 geometry를 사용할 수 있다. 완료 조건은 logical coverage, tail safety, resource validity, correct output과 의도한 dispatch가 함께 닫히는 것이다.

M=65 오답 수정이라면 rows 0–64가 reference와 맞고 rows 65–127의 invalid accesses/stores가 없어야 한다. single-sequence와 64+1 varlen fixtures를 모두 본다. heads와 split axes를 다시 켰을 때 coordinate가 섞이지 않는지 확인한다. 우연히 padding zeros 덕분에 답이 맞는 fixture만 쓰지 않는다.

invalid configuration 수정이라면 selected launcher의 block threads, shared bytes와 cluster dims가 device limits에 맞고 fallback가 의도대로인지 확인한다. tile을 줄여 launch가 성공했지만 다른 head dimension에서 unsupported output을 내면 완료가 아니다. validation와 runtime launch contract가 같은 지원 범위를 말해야 한다.

decode underfill 최적화라면 동일 workload 조건에서 selected family와 M/KV distributions를 기록한다. 작은 tile로 바꾸어 row-slot fill은 개선됐지만 split merge와 launch 수가 늘 수 있다. 실제 작업에서는 latency와 throughput을 측정해야 하나 이 장에서는 어떤 분모와 state를 함께 기록해야 causal comparison가 되는지만 정한다.

persistent scheduler 수정이라면 logical tile coverage를 검증한다. worker CTA count가 예상과 같다는 것만으로는 부족하다. 각 valid tile이 한 번 처리되고 terminal 뒤 추가 claim이 없으며 output ownership가 겹치지 않아야 한다. scheduler queue와 epilogue store가 같은 work ID를 공유하는지 본다.

cluster fallback 수정이라면 cluster path와 non-cluster path가 모두 correctness를 유지하는지 확인한다. 지원 장비에서는 cluster launch attributes가 적용되고 미지원 조건에서는 명시적 fallback 또는 선명한 validation error가 나와야 한다. silent wrong geometry는 허용하지 않는다.

**관측을 추가할 때 서비스 자체를 망가뜨리지 않는다**

모든 thread의 coordinate와 predicate를 production log에 쓰면 kernel 동작을 크게 바꾼다. device `printf`는 실행 순서와 성능에 영향을 주고 출력량도 감당하기 어렵다. 평상시에는 host-side shape, dispatch와 launch record를 수집하고, 작은 deterministic fixture에서만 제한된 device coordinates 또는 output coverage를 검사한다.

metric label에 request ID, grid tuple과 kernel symbol을 모두 넣으면 cardinality가 폭증한다. aggregate metric에는 bounded kernel family, dtype, architecture와 M/KV buckets를 쓰고 exact request/launch correlation는 sampled trace로 보낸다. 개인정보인 prompt를 저장하지 않아도 lengths와 anonymous IDs로 shape 문제를 조사할 수 있다.

비동기 오류를 좁히려고 모든 launch 뒤 synchronize를 넣으면 race 위치를 찾는 데 일시적으로 도움이 될 수 있지만 정상 운영 해법은 아니다. overlap을 없애 latency를 바꾸고 hidden ordering bug를 가릴 수 있다. failing launch correlation를 좁힌 뒤 필요한 producer-consumer event와 error boundary를 복구한다.

profiler를 켰을 때 kernel selection나 CUDA Graph replay가 달라질 수 있음도 기록한다. 관측 도구가 compile/cache/graph behavior에 미치는 영향을 비교한다. source fact, uninstrumented symptom, instrumented trace를 구분해야 관측으로 만든 새 경로를 원래 사고로 오인하지 않는다.

**그럴듯하지만 틀린 다섯 문장을 반증한다**

“batch size가 8이므로 threads도 8개다.” query tensor M과 launch block을 비교하면 반증된다. M=8이어도 교육 fixture는 128-thread CTA를 launch하며 실제 attention은 여러 warps가 협력한다. “grid.x가 1이므로 CTA는 하나다.” grid.y/z가 1인지 확인해야 하며 split/head axes가 있으면 총 CTA는 더 많다.

“CTA가 하나라 occupancy가 낮다.” theoretical residency와 device-wide work supply를 구분하면 반증된다. 하나의 CTA가 놓인 SM에서 여러 warps가 resident할 수 있지만 다른 SM에 일이 없을 수 있다. exact occupancy는 resources와 device limits가 필요하다. “row slots 56개가 비었으니 lanes 56개가 idle다.” kernel indexing과 warp roles를 읽으면 이 변환이 성립하지 않을 수 있다.

“CUDA 13으로 바꾸자 grid 의미가 달라졌다.” CUDA 12.9.1과 13.3.0의 기본 hierarchy를 교차 확인하고 selected library kernel/scheduler를 고정하면 기본 의미와 implementation geometry를 분리할 수 있다. “occupancy를 올리면 빨라진다.” memory dependency, spill, reuse와 total work 관측을 함께 보면 높은 occupancy와 낮은 throughput가 공존할 수 있다.

이 다섯 반증은 말싸움이 아니라 조사 순서다. 각각 필요한 관측점이 다르다. tensor/metadata, 전체 grid tuple, resource residency, kernel indexing, artifact/scheduler와 stall/bytes를 차례로 요구한다. 좋은 진단 문장은 틀릴 수 있는 조건과 다음 확인 지점을 포함한다.

**사건 F: 요청 32개를 grid.x 32로 읽어 GPU가 놀고 있다고 결론냈다**

실제 사고를 숫자로 고정한다. scheduler snapshot에는 decode 요청 32개가 있고 각 요청은 이번 step에서 token 하나를 낸다. 모델은 query heads 32개, KV heads 8개, head dimension 128이며 page size는 16이다. 운영자는 profiler에서 kernel 하나의 `grid=(256,1,1)`을 보고 “요청은 32개인데 CTA가 256개나 떠 launch가 비효율적”이라고 결론냈다. 다음 배포에서 grid.x를 request count에 맞춰 32로 제한했고 ITL이 11 ms에서 37 ms로 악화됐다.

오류는 숫자 256이 어디서 왔는지 번역하지 않은 데 있다. 이 kernel의 교육용 계약을 `grid.x = batch × kv_head`라고 두면 `32×8=256`이다. CTA 하나는 request 하나 전체가 아니라 `(request, kv_head)` pair 하나를 담당한다. CTA 내부 128 threads는 네 warps이고, 각 warp는 query heads 가운데 같은 KV head를 공유하는 GQA group 네 개 중 하나를 맡는다. lane은 head dimension 128을 32개씩 나눠 네 elements를 load한다.

좌표를 코드처럼 펼치면 `request = blockIdx.x / 8`, `kv_head = blockIdx.x % 8`, `q_head = kv_head*4 + warp_id`, `d = lane_id + 32*k`다. `blockIdx.x=91`이면 request 11, KV head 3이다. warp 2는 query head 14를 계산하고 lane 7은 dimensions 7,39,71,103을 담당한다. 이 변환을 적으면 256 CTAs가 중복 work가 아니라 필요한 head partition임을 바로 알 수 있다.

실제 kernel이 반드시 이 식을 쓴다는 뜻은 아니다. split-KV는 sequence chunk 축을 추가하고 persistent scheduler는 CTA가 work queue에서 여러 tile을 가져간다. 어떤 구현은 `grid.y`에 head를 두고 `grid.x`에 batch 또는 tile을 둔다. 중요한 것은 request count를 launch dimension과 동일시하지 않고 launcher의 indexing 식으로 좌표를 복원하는 것이다.

잘못된 patch는 grid 32개만 launch하면서 각 CTA가 KV head 0만 처리하거나 loop로 여덟 heads를 직렬 처리하게 만들었다. 첫 변형은 output 일부를 쓰지 않아 correctness defect이고, 둘째는 correctness를 유지하지만 parallel work를 8배 접어 넣는다. register lifetime과 loop-carried state가 늘고, memory latency를 가릴 independent warps가 줄어 ITL tail이 커졌다.

**작은 kernel로 request·head·tile 좌표를 끝까지 검산한다**

교육용 kernel은 query `Q[B,Hq,D]`와 paged KV를 읽어 head별 partial score를 만든다고 하자. `B=3`, `Hq=8`, `Hkv=2`, `D=64`, query heads per KV head는 4다. CTA는 128 threads, 즉 네 warps다. grid는 `B×Hkv=6` CTAs다. warp 하나가 query head 하나를 맡고 lane 하나가 두 dimensions를 처리한다.

`blockIdx.x=5`이면 `request=2`, `kv_head=1`이다. `threadIdx.x=70`이면 warp 2, lane 6이므로 `q_head=1×4+2=6`이고 dimensions 6과 38을 읽는다. load address는 contiguous dimension을 따라 lane이 증가해 warp의 첫 load가 `d=0..31`, 둘째가 `d=32..63`을 덮는다. boundary predicate는 request, head, dimension 축마다 따로 검산한다.

token tile 축을 추가하자. 요청별 context lengths가 `[17,33,65]`, KV tile이 32 tokens라면 필요한 tiles는 `[1,2,3]`이 아니라 ceiling 기준 `[1,2,3]`이 맞다. 총 pair work는 `Hkv×(1+2+3)=12`다. rectangular grid를 `B×Hkv×max_tiles=3×2×3=18`로 띄우면 여섯 CTAs는 predicate로 빠진다. compact plan을 만들면 valid work 12개만 launch할 수 있지만 plan build와 indirection 비용이 생긴다.

request 3개라는 숫자는 어느 경우에도 grid size를 결정하지 않는다. static rectangular launcher의 grid는 18이고 compact work list의 grid는 12이며 persistent kernel은 SM 수에 맞춘 더 작은 CTA 집합일 수 있다. 같은 request batch가 backend와 plan에 따라 세 geometry를 갖는다. 성능 비교에서는 request 수뿐 아니라 valid tiles, padded tiles, split count, persistent CTA count를 함께 기록한다.

tail tile도 계산한다. 길이 65의 마지막 tile에는 valid token 하나만 있다. CTA가 32-token tile 전체를 준비하면 token utilization은 `1/32`지만 head와 dimension lanes는 여전히 유효할 수 있다. “CTA utilization 3.1%”라고 말하려면 분모가 token slots인지 active lanes인지 issue cycles인지 붙인다. 서로 다른 utilization을 한 숫자로 합치지 않는다.

**occupancy 계산은 launch CTA 수가 아니라 resident 한계를 묻는다**

GPU에 CTA가 256개 launch됐다고 256개가 동시에 resident하는 것은 아니다. SM별 resident CTA 수는 threads, registers, shared memory, architecture limit의 최솟값으로 제한된다. 예를 들어 SM당 최대 2,048 threads, 65,536 registers, shared memory 164 KiB, 최대 CTA 16이라고 하자. kernel은 CTA당 128 threads, thread당 96 registers, shared memory 48 KiB를 쓴다.

thread 한계는 `2048/128=16 CTAs`다. register 한계는 CTA당 `128×96=12,288 registers`이므로 floor `65,536/12,288=5 CTAs`다. shared-memory 한계는 floor `164/48=3 CTAs`다. architecture CTA limit는 16이다. 따라서 resident limit는 SM당 3 CTAs이고 resident warps는 `3×4=12`, theoretical warp occupancy는 maximum 64 warps를 기준으로 18.75%다.

여기서 register를 64로 낮춰도 shared memory가 그대로면 resident CTA는 여전히 3이다. register 최적화만 하고 occupancy가 오르지 않는 이유다. shared memory를 32 KiB로 낮추면 shared limit가 5가 되고 register limit도 5라 resident CTAs가 5, warps가 20으로 늘 수 있다. 그러나 더 작은 tile 때문에 global loads가 늘면 전체 kernel 시간은 나빠질 수 있다. occupancy는 목적이 아니라 latency hiding capacity의 한 제약이다.

SM이 120개인 GPU에서 256 CTAs를 launch하고 SM당 최대 3개가 resident 가능하면 첫 wave에 이론상 모두 배치할 여지가 있다. grid를 32로 줄이면 최대 32 SM 정도만 work를 받아 나머지가 idle할 수 있다. 바로 이 때문에 “request 32이므로 CTA 32면 충분”이라는 patch가 병렬성을 죽였다. scheduler batch와 device work decomposition 사이의 head/tile expansion을 지웠기 때문이다.

반대로 grid가 수십만이라고 항상 좋지 않다. CTA당 work가 너무 작으면 scheduling overhead, plan indirection, partial reduction 비용이 커진다. split-KV가 만든 partials는 뒤 reduction을 요구한다. occupancy 계산은 resident 가능성을 말하고, grid size는 waves를 말하며, useful work ratio는 predicate와 tiles를 말한다. 세 숫자를 분리해야 tuning 방향이 나온다.

**CUDA 공식 의미를 구현 추측과 분리한다**

CUDA execution model에서 grid는 thread blocks의 집합이고 block 내부 threads는 shared memory와 block-scoped synchronization으로 협력할 수 있다. 서로 다른 blocks의 일반적인 실행 순서는 보장되지 않는다. 따라서 CTA 0이 request 0을 끝낸 뒤 CTA 1이 request 1을 실행한다고 가정할 수 없다. persistent work queue가 순서를 만들면 그것은 kernel이 구현한 protocol이다.

warp는 현재 NVIDIA execution에서 32 threads의 scheduling 단위지만 correctness를 warp 간 암묵적 lockstep에만 기대지 않는다. divergence 뒤 active mask가 달라질 수 있고 warp-level primitive는 참여 mask contract를 가져야 한다. lane 31이 tail predicate로 빠졌는데 full mask shuffle을 호출하는 식의 코드는 경계값에서만 틀릴 수 있다.

block dimensions와 threads per block에는 device limit가 있고 dynamic shared memory 요청도 launch eligibility를 제한한다. compile된 kernel의 register allocation과 launcher의 dynamic shared bytes를 함께 봐야 한다. `invalid configuration argument`를 grid가 너무 크다는 말로 축약하지 않는다. threads, axes, shared memory, cluster constraints 가운데 어느 contract를 넘었는지 launch record로 확인한다.

occupancy API나 calculator가 주는 값은 주어진 resource 사용량 아래 가능한 residency의 모델이다. memory coalescing, instruction dependency, tensor-core eligibility, cache miss, imbalance를 직접 측정하지 않는다. 높은 occupancy와 낮은 throughput이 동시에 가능하고, register spilling을 감수해 occupancy를 높인 patch가 local memory traffic 때문에 느려질 수 있다.

**vLLM·SGLang·FlashInfer 호출을 geometry 소비자까지 잇는다**

vLLM source walk는 scheduler의 request 수에서 멈추지 않는다. runner가 scheduled tokens와 slot mappings, query shape를 만들고 attention backend를 선택하며 native operator에 넘기는 경로를 따른다. native launcher에서 batch, heads, sequence metadata, tile constants가 grid와 block configuration으로 바뀌는 줄을 찾는다. Python `num_reqs`가 존재한다는 사실만으로 `grid.x=num_reqs`라고 쓰지 않는다.

SGLang도 batch object의 request cardinality와 FlashInfer wrapper의 plan을 분리한다. wrapper가 ragged lengths와 page indices로 workspace plan을 만들면 run 단계는 그 plan의 work descriptors를 소비할 수 있다. static decode wrapper와 extend/prefill wrapper는 같은 batch라도 tiles와 split 정책이 다르다. plan이 cache되면 shape key와 실제 lengths가 일치하는지도 검증한다.

FlashInfer에서는 plan 단계가 work를 확장하거나 CTA policy를 고를 수 있다는 사실이 중요하다. launcher의 `gridDim`만 보고 request mapping을 복원할 수 없으면 plan buffer가 가진 request index, tile index, KV chunk index를 함께 읽는다. persistent CTA가 atomic work counter에서 다음 descriptor를 가져간다면 `blockIdx.x`는 request identity가 아니라 worker identity다.

source pin은 두 종류로 남긴다. 첫째는 Python/runner에서 semantic tensors가 만들어지는 위치다. 둘째는 native launcher 또는 plan에서 execution coordinates가 만들어지는 위치다. 그 사이 dispatcher가 backend, dtype, head dimension, architecture에 따라 다른 kernel을 선택한다면 decision predicate도 기록한다. source walk의 산출물은 파일 목록이 아니라 `semantic shape → dispatch → plan → launch → indexing`의 연속 표다.

**검증과 rollback: grid를 되돌리는 대신 오진을 제거한다**

재현은 request count 32를 고정하고 heads와 lengths를 바꾼다. Hkv를 4,8,16으로 바꿀 때 grid가 128,256,512로 변하는 계약인지 확인한다. context lengths를 `[1]×32`와 긴 ragged 분포로 바꿔 static tiles, compact work items, split count를 기록한다. request count가 같아도 geometry가 변하면 최초 가설은 반증된다.

correctness fixture는 작은 B=3 예제에서 모든 `(request,q_head,d)`가 정확히 한 output owner에게 쓰이는지 본다. grid 축을 flatten/unflatten해 set equality를 검사하고 missing, duplicate coordinates를 찾는다. lengths 31,32,33과 heads group 경계 3,4,5를 포함한다. tail predicate가 load와 store 양쪽에 있는지 확인한다.

performance fixture는 launch 수, CTAs, waves, registers/thread, shared/CTA, achieved active warps, memory throughput, kernel duration을 함께 수집한다. occupancy 하나로 판정하지 않는다. 잘못된 grid cap을 제거했을 때 ITL 37 ms가 원래 11 ms 부근으로 돌아오는지 보되, output equality와 tail latency를 동시에 확인한다.

rollback는 상수 32를 이전 256으로 되돌리는 일이 아니다. backend별 native geometry 선택을 복원하고 임의 cap을 제거한다. 변경 전 source commit과 dispatcher predicate를 고정하며, 새 CUDA/library version에서 launcher contract가 달라졌다면 이전 숫자를 강제하지 않는다. known-good backend fallback는 correctness를 보존하는 동안만 사용한다.

완료 조건은 세 가지다. 첫째, request·token·head·tile이 grid/CTA/warp/lane 또는 persistent work descriptor로 변환되는 식을 한 fixture에서 재현한다. 둘째, resource limits로 resident CTA 상한을 손계산하고 profiler 값과 차이를 설명한다. 셋째, request count만 바꾼 grid patch가 다시 들어오면 regression test가 geometry coverage 또는 성능 threshold에서 실패한다.

이 사건의 최초 불일치는 “grid 256이 request 32보다 크다”가 아니다. launcher가 `(request,kv_head)` 256 work units를 요구하는데 patch가 request 축만 남긴 순간이다. 원인을 이렇게 쓰면 다음 backend가 `grid.x` 대신 plan buffer나 persistent queue를 써도 같은 검토 질문을 적용할 수 있다.

## 40.7 좌표·plan·occupancy를 재현 실험으로 고정한다

### 40.7.1 실제 주소까지 내리면 idle lane과 낭비 CTA를 구별할 수 있다

앞의 B=3 fixture를 memory address까지 내린다. Q layout이 row-major `[B,Hq,D]`, FP16이라고 하자. `(request=2,q_head=6,d=38)`의 element index는 `(2×8+6)×64+38=1,446`, byte offset은 2,892다. warp 2의 lane 6이 두 번째 vector에서 이 주소를 읽는다. lane 7은 offset 2,894를 읽으므로 인접 lanes가 인접 FP16 elements를 요청한다.

하지만 base pointer alignment와 vector load width가 맞는지는 별도다. kernel이 lane마다 8-byte vector를 읽는다면 `d` mapping은 scalar 예제와 달라진다. `D=64`가 vector width로 나누어지고 base가 필요한 alignment를 만족하는지 확인한다. 교육용 scalar 식을 실제 vectorized kernel에 그대로 대입하지 않는다. launcher specialization의 head dimension predicate와 native type을 함께 읽는다.

KV는 paged layout이므로 logical token을 physical address로 바꾸는 단계가 하나 더 있다. request 2의 token 64가 logical block 2, offset 0이고 block table entry가 physical block 19라면 address는 layout stride에 따라 `(physical_block=19, kv_head=1, token_offset=0, d)`로 번역된다. CTA 좌표가 맞아도 block table generation이나 stride를 틀리면 wrong answer다. geometry 검산과 address translation 검산을 분리한다.

길이 65의 마지막 tile에서 CTA는 token 64만 유효하다. token lane을 warp 축으로 펼친 kernel이라면 많은 lanes가 predicate로 빠질 수 있다. 반대로 head dimension을 lane 축으로 둔 예제에서는 token 하나를 처리해도 32 lanes가 dimension loads를 수행한다. “tail tile이므로 warp 31/32가 idle” 같은 주장은 indexing 식 없이는 성립하지 않는다.

split-KV에서는 같은 `(request,head)`를 여러 CTAs가 sequence chunks로 나눈다. 각 CTA는 partial maximum, partial sum, partial output을 쓰고 reduction kernel이 합친다. grid가 request×head×split로 늘지만 output owner는 partial buffer다. 두 CTAs가 final output에 직접 non-atomic write하면 race다. source에서 partial buffer stride와 reduction launch를 찾아야 split factor가 안전한 expansion임을 설명할 수 있다.

GQA의 warp mapping도 specialization에 따라 달라진다. query heads per KV head가 4라서 warp 네 개에 자연스럽게 맞았지만 ratio가 8이면 CTA가 여덟 warps를 쓰거나 한 warp가 두 query heads를 순회할 수 있다. ratio가 1인 MHA에서는 네-warps 계약이 다른 축을 병렬화할 수 있다. 하나의 모델 fixture를 모든 architecture에 일반화하지 않는다.

MoE나 GEMM kernel과 attention kernel도 grid 의미가 다르다. GEMM은 M/N tiles가 CTA coordinates가 되고 M이 flattened tokens일 수 있다. attention decode는 request/head/split가 work units일 수 있다. profiler에서 둘 다 `grid.x=256`이라고 보여도 같은 병렬성을 뜻하지 않는다. kernel symbol, arguments, plan descriptor를 붙여 geometry를 해석한다.

lane-level 검증은 32개 addresses를 표로 뽑는 것이 가장 빠르다. lane 0–31의 first load, second load, predicate, expected tensor coordinate를 생성하고 중복과 누락을 set으로 검사한다. source를 실행하지 않아도 indexing expression을 작은 reference function으로 옮겨 boundary를 검산할 수 있다. production GPU benchmark를 돌리지 않는 상황에서도 정적 계약 test는 충분히 가치가 있다.

### 40.7.2 plan descriptor를 역추적해 persistent kernel의 좌표를 복원한다

persistent kernel에서는 `blockIdx.x`를 request로 나누는 식 자체가 없을 수 있다. 예를 들어 SM 수를 기준으로 120 CTAs만 launch하고 각 CTA가 global work counter에서 index를 가져온다. plan에는 12,000 descriptors가 있고 descriptor 하나가 `(request,kv_head,tile_begin,tile_end,output_slot)`을 가진다. CTA 7은 descriptor 7을 끝낸 뒤 127,247처럼 다음 work를 가져갈 수 있다.

이때 profiler의 grid 120을 request count 32와 비교하는 것도 무의미하다. grid는 workers 수이고 logical work는 descriptor count다. concurrency는 counter contention, descriptor imbalance, CTA residence, per-work duration에 의해 결정된다. 마지막 몇 개 긴 descriptors 때문에 CTAs 대부분이 먼저 끝나는 tail imbalance도 생긴다.

plan 단계 입력을 기록한다. batch size, indptr, page indices, last-page length, heads, head dimension, dtype, causal/window mode, split policy, workspace capacity가 대표적이다. plan output에서는 descriptor count, split count distribution, chosen CTA count, temporary buffer size를 본다. run이 plan을 재사용한다면 shape key가 이 입력들을 충분히 포함하는지 확인한다.

stale plan 사건을 가정하자. batch는 여전히 32지만 context lengths가 짧은 분포에서 긴 분포로 바뀌었다. cache key가 batch size만 포함하면 old descriptor ranges가 새 page table을 덜 읽거나 잘못된 output slot을 가리킬 수 있다. request count와 grid가 동일해 dashboard에는 변화가 없지만 kernel duration과 output은 깨진다. geometry identity에는 ragged metadata와 policy generation이 필요하다.

FlashInfer 호출 경로를 읽을 때 Python wrapper의 `plan()`과 `run()`을 한 쌍으로 고정하는 이유가 여기 있다. plan은 단순 preallocation이 아니라 work decomposition을 결정할 수 있고 run은 opaque buffers를 native launcher에 다시 넣는다. run line만 보면 grid의 의미를 복원할 정보가 사라져 있다. plan arguments와 generated metadata를 함께 pin한다.

SGLang runner는 request batch를 만들지만 backend wrapper가 이를 page/sequence metadata로 바꾸고 plan을 호출한다. scheduler priority나 batch cardinality는 upstream control fact다. 실제 CUDA work shape는 downstream plan fact다. 둘 사이에는 padding, prefix sharing, speculative tokens, split policy가 개입할 수 있다. trace span을 두 경계에 각각 남겨 causal join을 만든다.

vLLM의 attention backend 선택도 같은 역할을 한다. runner input shape가 같아도 selected backend, quantization, cache dtype, sliding window, architecture capability에 따라 native path가 달라질 수 있다. release upgrade 뒤 grid가 바뀌면 regression이라고 단정하기 전에 dispatch predicate와 kernel contract가 바뀌었는지 source diff를 읽는다.

persistent descriptor 검증은 coverage와 uniqueness로 닫는다. expected logical work set을 reference로 만들고 plan descriptors가 이를 정확히 덮는지 검사한다. split work는 `(request,head,token-range)` intervals의 union이 expected range와 같고 overlap policy가 reduction contract와 맞아야 한다. descriptor order는 달라도 coverage가 같을 수 있으므로 순서 equality만 요구하지 않는다.

load balancing은 descriptor count만으로 판단하지 않는다. token ranges 1개와 1,024개는 work cost가 다르다. estimated bytes, sequence length, split reduction cost를 descriptor별 weight로 둔다. persistent counter가 dynamic scheduling을 해도 한 descriptor가 지나치게 크면 마지막 CTA tail이 남는다. tile size를 줄이면 balance는 좋아지지만 descriptor와 reduction overhead가 늘어난다.

### 40.7.3 occupancy 오진을 막는 네 개의 분모와 실험 matrix

첫 분모는 launch coverage다. 필요한 logical work units 가운데 descriptor 또는 CTA가 몇 개를 덮었는지 본다. 이 값이 100%가 아니면 성능 이전에 correctness 문제다. request-grid cap 사건은 head work를 loop로 보상하지 않은 variant에서 coverage가 12.5%였다.

둘째 분모는 tile utilization이다. allocated token slots 가운데 valid tokens 비율이다. lengths `[17,33,65]`, tile 32면 allocated slots는 `32+64+96=192`, valid tokens는 115라 utilization은 약 59.9%다. compact descriptor를 써도 마지막 tile 내부 padding은 남을 수 있다.

셋째 분모는 lane activity다. warp issue에서 predicate true lanes 비율을 sampling counter나 indexing fixture로 본다. token padding이 많아도 dimension-parallel warp는 active lanes가 높을 수 있고, irregular head dimension tail은 lane divergence를 만들 수 있다. tile utilization을 lane activity라고 부르지 않는다.

넷째 분모는 resident occupancy다. architecture 최대 resident warps 대비 실제 또는 theoretical active warps다. 앞 fixture는 shared memory 때문에 theoretical 18.75%였다. grid가 32로 작으면 theoretical resource occupancy와 별개로 전체 GPU에 충분한 CTAs가 없어 achieved occupancy가 더 낮다.

실험 A는 batch만 바꾼다. B=1,8,32,64에서 heads와 lengths를 고정하고 logical work, grid/descriptors, waves, kernel duration을 기록한다. 작은 B에서 persistent kernel과 static kernel의 crossover를 본다. request count 증가가 grid에 선형 반영되는지는 backend contract에 따라 관측한다.

실험 B는 Hkv만 1,2,8로 바꾼다. B는 32로 고정한다. static pair mapping이면 work가 32,64,256으로 변한다. grid cap patch는 Hkv 증가에 반응하지 않아 바로 드러난다. GQA ratio 변화가 CTA threads나 warp loop를 바꾸면 registers와 shared memory도 다시 측정한다.

실험 C는 lengths distribution을 바꾼다. 모두 1, 모두 32, 31/32/33 혼합, long-tail 65/1025를 비교한다. request와 heads는 같다. descriptor count, split count, last tile utilization, reduction launches가 변한다. 이 실험은 request count dashboard가 숨기는 token work를 보여 준다.

실험 D는 head dimension 64,128,256을 바꾼다. vector width, warps per CTA, registers, shared memory specialization이 바뀔 수 있다. unsupported dimension이 generic fallback로 가면 grid가 같아도 kernel time이 급증할 수 있다. dispatcher 선택을 결과에 붙인다.

실험 E는 resource knob만 바꾼다. tile은 그대로 두고 compiler register cap을 강제하는 실험은 spill bytes와 local load/store를 함께 본다. dynamic shared memory를 줄이는 variant는 recomputation/global traffic을 기록한다. occupancy 상승만으로 승자를 고르지 않는다.

실험 F는 split factor를 바꾼다. split 1,2,4,8에서 CTAs, partial buffer bytes, reduction time, end-to-end ITL을 측정한다. sequence가 짧을 때 split은 overhead만 늘 수 있고 길 때 parallelism을 준다. scheduler request 수는 모든 행에서 같게 유지한다.

실험 G는 plan cache hit/miss를 분리한다. plan build CPU time, workspace allocation, run kernel time를 따로 잰다. 캐시 hit만 benchmark하면 dynamic shape 서비스의 plan overhead를 숨긴다. 반대로 매번 plan하면 steady state를 과대평가한다.

실험 H는 warmup과 graph capture를 분리한다. 첫 launch JIT/module load, graph capture, steady replay를 같은 kernel duration으로 섞지 않는다. graph replay가 고정 grid를 갖더라도 valid work metadata가 동적으로 안전하게 갱신되는지 확인한다.

각 실험 행에는 correctness hash 또는 reference tolerance를 붙인다. 빠른데 missing heads가 있는 grid cap을 성능 개선으로 기록하지 않는다. NaN, unwritten sentinel, duplicate output writer, boundary mismatch를 먼저 검사하고 성능 통계는 그 뒤에 본다.

통계는 median 하나로 닫지 않는다. kernel duration p50/p99, ITL distribution, request length strata, GPU clocks와 concurrent work를 기록한다. grid geometry patch는 짧은 requests에는 이득이고 긴 requests에는 손해일 수 있다. workload mix를 공개한다.

### 40.7.4 배포 terminal: 숫자가 아니라 번역 계약을 고정한다

canary에는 launch ledger sampling을 넣는다. model/backend signature별로 batch, query tokens, heads, lengths bucket, plan work count, grid, block threads, registers, shared bytes를 낮은 비율로 수집한다. request text나 KV content는 필요 없다. label cardinality를 통제하고 raw descriptor dump는 anomaly trace에만 붙인다.

alert는 `grid != request_count`에 걸지 않는다. expected geometry contract에서 coverage가 벗어나거나, 동일 signature에서 work-per-request가 설명 없이 급변하거나, unwritten sentinel과 launch failure가 생길 때 건다. persistent kernel은 grid가 고정이어도 descriptor count와 tail imbalance를 본다.

rollback 후보는 세 층이다. dispatcher flag로 known-good backend를 선택하는 것, split/persistent policy를 이전 값으로 되돌리는 것, 전체 build를 이전 commit으로 되돌리는 것이다. 각 후보가 semantic support를 유지하는지 확인한다. head dimension이나 dtype를 지원하지 않는 fallback를 강제하면 가용성 대신 오류가 난다.

배포 전 gate는 B/Hkv/D/length 경계 matrix의 coverage test, source-pinned launcher contract, resource limit sanity, representative performance threshold를 포함한다. CUDA/library upgrade에서는 compiled register count와 shared bytes가 달라질 수 있으므로 golden absolute occupancy 하나만 강제하지 않는다. 변화가 설명되고 end-to-end 목표를 만족하는지 본다.

90분 soak에서는 batch와 length를 주기적으로 흔들고 abort/preemption도 섞는다. launch failure, NaN, missing coordinate, plan generation mismatch가 0이어야 한다. ITL p99가 baseline budget 안에 있고 descriptor tail이 특정 SM에 고착되지 않는지 본다. clock throttling과 다른 workload 간섭도 함께 기록한다.

terminal report의 첫 표는 upstream과 downstream을 나란히 둔다. scheduler requests 32, scheduled query tokens 32, Hkv 8, logical head work 256, plan descriptors 256, grid 256, CTA threads 128, resident limit 3처럼 쓴다. 어느 숫자도 다른 숫자의 별명으로 쓰지 않는다.

둘째 표는 before/bug/fix를 비교한다. before는 native policy grid 256, ITL 11 ms다. bug는 request cap 32, coverage 또는 serial loop 문제, ITL 37 ms다. fix는 native decomposition 복원, coverage 100%, ITL 11–12 ms다. correctness hash와 resource counters를 같이 둔다.

셋째 표는 반증된 가설을 남긴다. “CTA가 request보다 많아 낭비”는 head work로 반증됐다. “occupancy가 낮아 register가 원인”은 shared-memory minimum으로 반증됐다. “grid만 복구하면 완료”는 plan identity와 boundary correctness gate 때문에 기각됐다. 반증 기록은 다음 튜너가 같은 shortcut을 반복하지 않게 한다.

마지막 승인 문장은 구체적이어야 한다. “request 32를 `(request,kv_head)` 256 work units로 번역하고, CTA 내부 네 warps가 GQA query heads를 덮으며, resource 상한은 shared memory로 SM당 3 CTAs다. source-pinned plan/launcher와 coverage fixture가 이를 증명하고 90분 soak에서 오류 0, ITL budget 통과했다.” 이 문장을 못 쓰면 아직 완료가 아니다.

## 40.8 한 요청을 끝까지 읽는 실전 순서

### 40.8.1 정상 경로를 한 줄씩 닫는다

첫 줄에는 API가 아니라 scheduler 결과를 쓴다. “request A의 prefill 128 rows, requests B–I의 decode 각 1 row”처럼 logical source를 남긴다. 둘째 줄에는 runner가 만든 query shape와 cumulative lengths를 쓴다. 셋째 줄에는 backend와 kernel family의 dispatch keys를 쓴다. 넷째 줄에는 native params의 M, heads, lengths와 split를 쓴다. 다섯째 줄에는 grid/block/shared/cluster/stream을 쓴다. 여섯째 줄에는 device scheduler의 axis mapping와 warp roles, tail predicates를 쓴다.

교육용 prefill은 이렇게 닫힌다. A의 scheduled rows 128이 query `[128,H,D]`가 된다. 같은 static `BLOCK_M=64` family를 가정하면 logical tiles는 2다. head/split axes를 1로 고정하면 grid.x 2, CTA당 128 threads, 총 8 warps가 launch된다. CTA 0/1은 각각 64 rows를 담당하고 tail이 없다. 실제 source에서 head/split/persistent axes가 있으면 그 항을 추가해 다시 계산한다.

교육용 decode는 B–I의 rows 합 8이 query `[8,H,D]`가 된다. 같은 static family라면 logical row tile은 1이고 invalid tail row slots는 56이다. CTA당 128 threads와 네 warps는 그대로다. kernel indexing을 읽기 전에는 idle lane 수를 쓰지 않는다. split-KV가 네 개면 partial work axis를 추가하고, persistent면 worker CTA와 claimed logical tile을 분리한다.

### 40.8.2 소스를 읽는 질문은 함수 이름보다 구체적이어야 한다

Python backend에서는 “M은 어느 tensor dimension이고 actual tokens가 따로 있는가?”, “어떤 predicate가 version/backend를 고르는가?”를 묻는다. native binding에서는 “Python objects가 어떤 C++ params로 복사되며 dtype/device/contiguity를 누가 검증하는가?”를 묻는다. launcher에서는 “grid/block/shared bytes와 stream을 계산하는 symbol은 무엇인가?”를 묻는다.

tile scheduler에서는 “각 blockIdx 축이 어떤 work coordinate가 되는가?”, “static인가 persistent인가?”, “invalid work와 terminal은 어떻게 표현되는가?”를 묻는다. kernel에서는 “threadIdx가 row, feature, warp group 중 무엇을 정하는가?”, “producer와 consumer 역할은 무엇인가?”, “tail load/store와 split merge의 predicate는 어디인가?”를 묻는다.

이 질문에 답하지 못한 빈 경계가 있으면 추론으로 메우지 않는다. extension binding가 generated code에 있거나 precompiled binary만 제공될 수 있다. 그 경우 확인된 Python call과 launcher source 사이의 미확인 binding를 명시하고 exported symbol/build manifest를 더 찾는다. 함수 이름이 비슷하다는 이유로 연결하지 않는다.

### 40.8.3 관측값과 source fact를 서로 대신 쓰지 않는다

source는 가능한 실행 경로를 보여 준다. 현재 request가 그 분기를 실행했다는 증거는 selected backend/kernel symbol 또는 trace다. 반대로 profiler의 grid는 실행 사실을 보여 주지만 각 축의 logical 의미는 source scheduler가 설명한다. 두 증거를 결합해야 request shape 번역이 닫힌다.

고정 source에서 `grid=num_sm`이 가능하다는 사실만으로 모든 Blackwell run이 persistent라고 쓰지 않는다. runtime selected scheduler를 확인한다. trace에서 grid.x가 SM 수와 같아 보인다고 persistent라 단정하지도 않는다. 다른 공식이 우연히 같은 수를 낼 수 있다. mode flag와 symbol을 함께 본다.

성능 수치는 더 엄격하다. 이 장의 12.5%는 logical row-slot fraction이지 measured utilization가 아니다. 256 launched thread instances는 두 CTA의 configuration 산술이지 simultaneous active threads가 아니다. 실제 latency와 occupancy를 제시하려면 hardware, workload, warm-up, concurrency, selected kernel와 도구 좌표가 필요하다.

### 40.8.4 독자가 직접 채우는 launch 번역 원장

원장의 첫 칸은 workload identity다. request 수, 각 request의 phase, scheduled query rows와 KV length를 적는다. 둘째 칸은 runner output으로 query shape/stride/dtype, cumulative query와 KV lengths, heads, D와 actual/capacity rows를 적는다. 두 칸의 합계가 맞지 않으면 native kernel로 내려가지 않는다.

셋째 칸은 dispatch다. backend와 version, architecture target, causal/window, varlen, split/persistent/cluster flags와 kernel symbol을 적는다. 넷째 칸은 launcher다. grid 세 축, block 세 축, dynamic shared bytes, workspace와 stream을 적는다. 총 CTA와 CTA당 threads는 곱으로 검산하지만 logical 의미는 아직 붙이지 않는다.

다섯째 칸에서 source scheduler를 읽어 각 grid axis를 query tile, batch, head, split 또는 worker ID에 대응시킨다. 여섯째 칸에서 thread linearization, warp/warp-group roles와 load/store predicates를 적는다. 마지막으로 logical tile coverage와 invalid tail을 계산한다. 이때 row-slot, launched thread, resident warp와 measured active lane을 서로 다른 열에 둔다.

prefill fixture의 완성 행은 actual M 128, capacity 128, query row tiles 2, row tail 0으로 시작한다. decode fixture는 actual M 8, capacity가 compact라면 8, query row tiles 1, row tail 56으로 시작한다. graph capacity 128이면 decode의 capacity tiles와 actual row tiles를 별도로 쓴다. 실제 launcher가 heads/splits를 추가하면 total CTA 식을 확장한다.

원장이 유용한 이유는 답을 미리 정하지 않기 때문이다. grid가 예상보다 크면 head/split/capacity를 확인하고, 작으면 persistent workers나 fused axes를 확인한다. thread 수가 많아도 row 협력 역할을 찾고, tail이 커도 KV work와 split를 본다. 첫 모순이 발견된 경계부터 source와 관측을 좁힐 수 있다.

### 40.8.5 이 장을 읽은 뒤 할 수 있어야 하는 설명

독자는 “decode가 작아서 느리다” 대신 더 강한 문장을 쓸 수 있어야 한다. 예를 들면 “이번 step의 actual query rows는 8이고 selected static kernel의 BLOCK_M은 64라 query row-slot fill은 12.5%다. grid의 head/split axes와 warp roles를 확인하기 전에는 lane utilization를 확정하지 않는다”라고 쓴다.

또 “CUDA grid가 달라졌다” 대신 “logical M과 metadata는 같지만 backend가 SM90 static scheduler에서 Blackwell persistent scheduler로 바뀌어 grid가 logical tile count가 아니라 worker count를 반영한다. output tile coverage와 selected symbol을 비교해야 regression 여부를 판정할 수 있다”라고 쓴다.

오답에서도 “GPU race 같다” 대신 “M=65에서 처음 생기는 CTA 1의 global row 64가 partial workspace부터 틀린다. Python M과 grid count는 맞으므로 tile coordinate, sequence-local indptr와 load predicate를 차례로 확인한다”라고 쓴다. 이 정도로 첫 divergence가 좁혀져야 다음 코드 위치와 검증 fixture가 정해진다.

**번역된 좌표가 GPU work가 되는 경로를 회수한다**

HTTP 요청은 제어 단위이고 token row는 model 계산 단위다. scheduler와 runner가 request들을 flattened M과 sequence metadata로 바꾸고, backend가 dtype·heads·D·capability·mode로 native path를 고른다. launcher는 traits와 params를 grid, block, shared-memory bytes, cluster와 stream으로 바꾼다. device scheduler와 kernel이 비로소 blockIdx/threadIdx를 logical attention work로 해석한다.

prefill `M=128`, `BLOCK_M=64`에서는 static row tiles가 두 개이고 tail이 없다. decode `M=8`에서는 row tile 하나에 valid rows 8, invalid tail row slots 56이 있다. 두 계산은 logical tile fill을 보여 준다. 그러나 row-per-thread 증거가 없으므로 56 idle lanes나 87.5% inactive threads라는 결론은 나오지 않는다. warp-specialized kernel에서는 네 warps가 여덟 rows의 K/V processing를 협력할 수 있다.

grid도 logical tile count와 항상 같지 않다. head axis가 CTA를 늘리고 split-KV가 한 output tile을 여러 partial works로 나누며 persistent scheduler가 SM 수의 worker CTAs로 여러 tiles를 순회할 수 있다. thread-block cluster는 선택적 협력 계층과 launch constraints를 추가한다. 그래서 grid 숫자를 읽기 전에 selected scheduler와 축 mapping를 읽는다.

occupancy, utilization와 throughput도 분리한다. grid supply가 적은 문제, SM residency를 제한하는 register/shared memory, warp instruction activity, logical tile underfill, memory dependency와 launch overhead는 서로 다른 현상이다. 하나의 percentage로 합치면 수정 방향이 틀어진다.

장애는 shape ledger의 first divergence로 좁힌다. Python M이 틀리면 scheduler/packing, metadata가 틀리면 varlen construction, specialization가 다르면 dispatcher, grid가 다르면 launcher/scheduler, boundary output이 틀리면 tile coordinate와 predicate를 본다. invalid configuration은 KV OOM와 분리하고, upgrade 뒤 grid 변화는 artifact와 selected kernel을 고정한 뒤 판정한다.

다음 장은 같은 launch를 byte 흐름으로 다시 본다. CTA와 warp 역할이 맞아도 registers, shared memory, L2와 HBM 사이에서 어느 bytes를 몇 번 옮기는지에 따라 성능이 달라진다. 40장에서 만든 `request shape→tile→CTA→warp role` 원장이 41장의 `tensor interval→memory space→transaction` 원장으로 이어진다.

### 40.8.6 15분 source drill: 숫자 하나를 다섯 좌표로 번역한다

첫 2분에는 scheduler snapshot에서 의미 숫자만 옮긴다. request count, scheduled query tokens, context lengths, query/KV heads, head dimension, cache page size를 적는다. `batch=32` 하나만 복사하지 않는다. decode라 해도 speculative verification이면 request마다 query tokens가 여러 개일 수 있고, chunked prefill이 섞이면 flattened rows가 request 수보다 크다.

다음 2분에는 tensor shape owner를 찾는다. runner가 Q를 `[total_query_tokens,Hq,D]`로 만드는지, padded `[B,max_q,Hq,D]`로 만드는지 확인한다. sequence indptr, page table, slot mapping이 어느 request boundary를 보존하는지도 적는다. 변수 이름 `num_tokens`가 capacity인지 actual rows인지 source mutation으로 판별한다.

다음 2분에는 dispatcher predicate를 고정한다. dtype, head dimension, cache dtype, GPU capability, causal/window mode, quantization, graph mode가 어떤 backend와 specialization을 선택하는지 기록한다. fallback가 같은 signature를 갖더라도 tile과 launch geometry는 다를 수 있다. 선택된 kernel symbol 또는 native entry point를 trace와 연결한다.

다음 3분에는 plan을 읽는다. static launcher면 grid 계산식의 각 factor를 semantic axis에 붙인다. persistent launcher면 descriptors의 fields와 count, worker CTAs, work counter를 적는다. split-KV면 split axis와 reduction output을 추가한다. `grid.x=256`만 남기지 않고 `256=(32 requests×8 KV heads)`처럼 factorization을 보존한다.

다음 3분에는 CTA 내부를 펼친다. threads 128은 warps 4개다. warp가 query head인지 token tile인지 output row인지 indexing expression에서 확인한다. lane 0과 lane 31의 첫 두 addresses를 계산한다. tail predicate가 load, accumulation, store에 각각 적용되는지 본다. warp-level primitive의 participation mask가 predicate와 일치하는지도 확인한다.

마지막 3분에는 resource ceiling과 terminal을 계산한다. registers/thread×threads/CTA, dynamic+static shared bytes/CTA, threads limit, architecture CTA limit의 minimum을 구한다. grid를 resident CTAs로 나눠 waves를 추정한다. correctness coverage, output sentinel, kernel duration, ITL을 확인하고 변경을 유지하거나 rollback할 근거를 한 문장으로 쓴다.

drill 예제를 다시 실행해 보자. snapshot은 requests 32, query tokens 32, Hq 32, Hkv 8, D 128이다. selected decode plan은 `(request,kv_head)` 256 descriptors를 만들고 worker mapping은 CTA당 descriptor 하나다. CTA 91은 request 11/KV head 3이고 warp 2는 query head 14, lane 7은 dimensions 7·39·71·103을 읽는다.

resource record는 threads 128, registers 96, shared 48 KiB다. 가정한 SM limit에서 threads는 16, registers는 5, shared는 3, CTA limit는 16이므로 resident ceiling은 3이다. 이 문장 하나로 register cap만 낮춘 patch가 occupancy를 못 바꾼 이유와 grid 32 cap이 device-wide parallelism을 줄인 이유를 동시에 설명할 수 있다.

drill의 실패 신호도 정한다. semantic shape와 plan work union이 다르면 plan/metadata 경계로 돌아간다. descriptor coverage는 맞지만 lane address가 틀리면 kernel indexing specialization을 본다. 좌표가 맞고 occupancy가 낮으면 limiting resource를 찾는다. occupancy도 설명되지만 시간이 느리면 memory traffic, dependency, reduction, imbalance로 이동한다. 순서를 지키면 모든 느림을 occupancy 탓으로 돌리지 않는다.

vLLM 경로에서는 runner execute 순서와 attention backend handoff를 첫 pin으로 삼는다. scheduled request metadata가 query rows와 KV slot mapping으로 변한 뒤 native attention entry로 넘어가는 구간을 연결한다. FlashAttention 계열 launcher의 tile shape와 warp 역할을 둘째 pin으로 둔다. 두 pin 사이 dispatcher branch를 빼면 model input과 실제 kernel을 잘못 연결할 수 있다.

SGLang 경로에서는 batch 생성, extend/decode wrapper 선택, FlashInfer plan/run boundary를 잇는다. plan이 workspace와 descriptor metadata를 만들고 run이 이를 소비한다면 두 함수를 한 source unit으로 읽는다. request batch만 보고 native grid를 추정하지 않는다. radix/cache locations는 attention work가 읽는 physical identity를 제공하지만 launch coordinate 자체와 동일하지 않다.

FlashInfer 경로에서는 plan work expansion과 CTA policy를 확인한다. work descriptors가 request보다 많아지는 것은 split/head/tile expansion일 수 있고 worker CTAs가 descriptors보다 적은 것은 persistent scheduling일 수 있다. 어느 쪽도 그 자체로 낭비 증거가 아니다. coverage, balance, resource residence, reduction cost가 판정을 완성한다.

공식 CUDA 의미는 이 source walk의 안전 난간이다. block 내부 협력과 block 간 독립성, warp-level participation contract, device resource limit, launch configuration validity를 구현 사실과 구분해 적는다. 특정 library가 persistent queue로 block 간 work distribution을 구현했으면 그것은 library protocol이지 CUDA가 보장한 request ordering이 아니다.

리뷰 결과는 여섯 열 표로 남긴다. semantic owner, shape/value, plan expansion, launch coordinate, lane address, evidence link다. 한 행은 Q row, 한 행은 KV page, 한 행은 output head, 한 행은 partial reduction을 담는다. 빈 evidence cell은 “추정”으로 표시하고 성능 결론의 전제로 사용하지 않는다.

incident review에서는 before/first divergence/after를 같은 표 형식으로 쓴다. before는 256 head work coverage, bug는 grid cap 32에서 missing 또는 serial head work, after는 native geometry 복원과 coverage 100%다. ITL 11→37→11–12 ms, correctness sentinel, register/shared values를 같은 timeline에 둔다.

rollback decision은 grid 숫자를 hard-code하지 않는다. selected backend가 바뀌었으면 그 backend의 native contract로 돌아간다. plan cache identity가 의심되면 cache를 끄고 rebuild해 correctness를 먼저 확인한다. CUDA upgrade로 register allocation가 달라졌으면 build와 resource record를 함께 되돌린다. 서로 다른 층을 한 flag로 뭉개지 않는다.

회귀 test는 세 축을 최소로 가진다. request 1/32, Hkv 1/8, lengths 31/32/33을 조합하고 D 64/128 specialization을 표본화한다. expected logical coordinate set과 actual plan coverage를 비교한다. performance test는 correctness를 통과한 조합만 평가한다. boundary 하나만 정상인 microbenchmark로 배포를 승인하지 않는다.

observability budget도 검토한다. 모든 launch의 descriptor dump는 서비스에 부담을 준다. signature별 counter와 histogram은 상시 수집하고 generation mismatch, coverage assertion, latency regression 때 ring buffer 상세를 보존한다. sampling 자체가 stream synchronize를 일으키지 않는지 확인한다. 관측 때문에 launch ordering을 바꾸면 원래 사건을 가릴 수 있다.

최종 terminal은 독자가 재현 가능한 문장이어야 한다. 어느 request shape가 어느 plan을 만들고, 어느 CTA/warp/lane이 어느 tensor coordinates를 소비하며, 어떤 resource가 residency를 제한하고, 어떤 source와 measurement가 이를 증명하는지 말한다. “GPU utilization이 회복됐다”만으로는 부족하다.

이 drill을 다른 kernel에 적용할 때도 질문 순서는 유지된다. GEMM이면 request/head 대신 M/N/K tiles가 들어가고 MoE면 routed tokens/expert tiles가 들어간다. 좌표 이름은 바뀌지만 semantic work, plan, launch, lane address, resource ceiling의 다섯 번역은 그대로다. 이것이 request 수와 grid 크기를 다시 혼동하지 않게 하는 재사용 가능한 도구다.

현장 체크리스트는 질문과 실패 의미를 함께 둔다. `total_query_tokens`가 request count와 다른가를 묻고, 다르면 batching 번역부터 다시 본다. Hq/Hkv ratio가 CTA 내부 warp 수와 일치하는가를 묻고, 다르면 loop 또는 다른 head axis를 찾는다. sequence tiles 합계가 descriptor count를 설명하는가를 묻고, 다르면 padding·split·persistent policy를 찾는다. 이 세 질문은 host semantic work를 닫는다.

다음으로 `blockIdx` 각 축을 어느 semantic axis가 소비하는지 묻는다. 직접 소비하지 않으면 work counter나 plan lookup을 찾는다. `threadIdx/32`가 어떤 row/head/tile을 고르는지 묻는다. lane이 읽는 첫·마지막 address를 계산한다. boundary predicate가 false인 lane이 warp collective에 참여하는지 묻는다. 이 세 질문은 CTA 내부 correctness를 닫는다.

resource 질문은 네 개를 동시에 적는다. threads ceiling, registers ceiling, shared-memory ceiling, architecture CTA ceiling이다. minimum만 최종 resident bound가 된다. compiler report의 registers/thread와 launcher의 dynamic shared bytes가 같은 specialization인지 확인한다. generic kernel 값을 optimized kernel에 대입하지 않는다. graph replay도 capture 당시 kernel signature와 현재 plan identity를 대조한다.

성능 질문은 waves와 useful work를 분리한다. grid/resident capacity로 대략적인 waves를 계산하고 descriptor cost 분포로 tail imbalance를 예상한다. valid token slots, active lanes, reduction bytes를 따로 센다. achieved occupancy가 낮으면 grid shortage인지 resource ceiling인지 stall 때문인지 분기한다. 높은 occupancy면 좋은 kernel이라는 결론은 내리지 않는다.

정확성 질문은 output coordinate coverage, writer uniqueness, partial reduction completeness다. expected set에 없는 writer가 있으면 out-of-bounds 가능성이고 missing coordinate는 unwritten output다. split ranges overlap이 의도됐다면 reduction owner가 있어야 한다. page generation과 output generation이 request plan과 일치하는지도 확인한다.

배포 질문은 source version, dispatcher decision, plan key, compiled resource record, performance fixture를 한 artifact로 묶었는가다. 숫자 하나만 golden으로 두면 backend upgrade를 정상 변화와 regression으로 구별할 수 없다. 계약과 evidence가 함께 versioned돼야 release diff가 설명 가능하다.

사고 대응 질문은 first divergence가 어느 번역 단계인가다. scheduler shape가 틀리면 upstream budget/packing을 본다. plan coverage가 틀리면 wrapper/cache key를 본다. launch와 plan이 다르면 launcher arguments를 본다. lane address만 틀리면 specialization indexing을 본다. coordinates가 모두 맞으면 resource와 memory behavior로 이동한다.

마지막 질문은 rollback 뒤 무엇이 증명됐는가다. 이전 grid가 돌아왔다는 답은 부족하다. semantic coverage 100%, boundary output equality, resource limit 설명, representative ITL budget, soak 오류 0을 요구한다. 이 조건들이 incident terminal을 재현 가능한 engineering record로 만든다.

한 번 더 수치로 압축하면 리뷰가 쉬워진다. requests 32라는 upstream 값은 query rows 32, `(request,KV-head)` work 256, CTA 256, warps 1,024, logical lanes 32,768로 확장된다. 그러나 이 숫자들은 동시에 resident하지 않는다. SM 120개와 SM당 resident CTA 3이라는 가정에서는 capacity가 360 CTAs라 한 wave 안에 256 CTAs를 수용할 수 있다. grid cap 32는 work를 줄인 것이 아니라 head axis를 삭제하거나 CTA 내부 loop로 직렬화한 것이다.

길이 축이 추가되면 같은 requests 32도 전혀 다른 결과가 된다. 모든 context가 32 이하라 split이 없다면 256 head work가 충분할 수 있다. context가 길어 split factor 4가 선택되면 partial work는 1,024가 되고 reduction work가 뒤따른다. persistent policy가 worker CTAs 240만 띄우더라도 descriptor 1,024개를 순회할 수 있다. profiler의 grid 240을 보고 work가 240이라고 계산하면 다시 같은 오류다.

이 계산을 dashboard 설명에도 그대로 쓴다. request gauge는 admission과 scheduler 상태를 말한다. scheduled tokens는 이번 step의 semantic rows를 말한다. descriptor count는 backend가 만든 work decomposition을 말한다. grid는 workers 또는 static tiles를 말한다. active warps는 resource와 runtime 결과를 말한다. 각 metric의 주어를 보존하면 서로 다른 층의 숫자를 나누어 비교할 수 있다.

따라서 튜닝 제안에는 변경할 주어가 명시돼야 한다. batch policy를 바꾸는지, split threshold를 바꾸는지, CTA tile을 바꾸는지, registers/shared memory를 바꾸는지, persistent worker count를 바꾸는지 적는다. “grid를 줄인다”는 문장만으로는 어느 semantic work를 유지하고 어느 overhead를 줄이는지 검토할 수 없다.

최종 리뷰어는 변경 전후의 동일 request fixture에서 이 다섯 주어 중 정확히 하나만 달라졌는지 확인한다. 여러 층이 함께 바뀌었다면 별도 ablation으로 효과를 분리한다. 그래야 CUDA version이나 backend upgrade가 겹쳐도 성능 회귀의 실제 owner를 다시 찾을 수 있다. 이 확인 뒤에만 새로운 geometry를 운영 기본값으로 최종 승인한다.

## 40.9 장말 reference — 전수 source 좌표와 최종 검산

### 40.9.1 Launch mapping은 어디서 정해지는가

Python의 `M`이나 CUDA version만 보고 grid 축을 추측하게 될 때는 여기서 시작한다. CUDA hierarchy의 공통 계약을 먼저 고정한 뒤 선택된 backend의 launcher와 scheduler가 각 축에 부여한 의미를 대조한다.

- [NVIDIA CUDA C++ Programming Guide 12.9.1 — §5.2 Thread Hierarchy](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#thread-hierarchy)
- [NVIDIA CUDA Programming Guide 13.3.0 — §1.2.2.1 Thread Blocks and Grids, §1.2.2.2 Warps and SIMT](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [NVIDIA CUDA Compatibility 13.0.2 — compatibility 경계 참고](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-compatibility/index.html). 기본 launch hierarchy 근거로 사용하지 않는다.

### 40.9.2 Shape가 처음부터 틀렸는가

Query rows, head/KV shape 또는 cumulative lengths가 native call 전에 이미 어긋날 때는 여기서 시작한다. Backend 생성자의 capability 계약에서 `forward`와 varlen call까지 같은 request의 shape가 보존되는지 읽는다.

- vLLM v0.27.1·`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` — `vllm/v1/attention/backends/flash_attn.py:743-838` — `FlashAttentionImpl.__init__` — https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/flash_attn.py#L743-L838
- vLLM v0.27.1·`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` — `vllm/v1/attention/backends/flash_attn.py:838-900` — `FlashAttentionImpl.forward` — https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/flash_attn.py#L838-L900
- vLLM v0.27.1·`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` — `vllm/v1/attention/backends/flash_attn.py:1000-1065` — varlen native call — https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/flash_attn.py#L1000-L1065
- vLLM v0.27.1·`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` — `vllm/v1/attention/backends/fa_utils.py:24-60,350-370` — CUDA extension import boundary — https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/fa_utils.py#L24-L60 — https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/fa_utils.py#L350-L370

### 40.9.3 Occupancy가 낮은 이유는 무엇인가

Logical rows는 맞는데 resident CTA나 active warp가 기대보다 적을 때는 여기서 시작한다. Launcher의 block/shared-memory 설정과 SM90 producer/consumer warp-group 역할을 함께 읽어 logical fill과 hardware occupancy를 구분한다.

- vLLM FlashAttention·`caaa4eb59845388a20b1f435ecaafb4bd9517ad8` — `hopper/flash_fwd_launch_template.h:165-205` — forward launcher — https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/hopper/flash_fwd_launch_template.h#L165-L205
- vLLM FlashAttention·`caaa4eb59845388a20b1f435ecaafb4bd9517ad8` — `hopper/tile_scheduler.hpp:96-112` — static tile scheduler — https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/hopper/tile_scheduler.hpp#L96-L112
- vLLM FlashAttention·`caaa4eb59845388a20b1f435ecaafb4bd9517ad8` — `hopper/flash_fwd_kernel_sm90.h:330-465` — producer/consumer warp-group kernel — https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/hopper/flash_fwd_kernel_sm90.h#L330-L465

### 40.9.4 Tail CTA는 어디에서 생기고 누가 다시 배분하는가

M 또는 KV length의 경계에서 마지막 CTA만 느리거나 빈 work가 보일 때는 여기서 시작한다. Plan metadata, prefill grid-axis indexing과 persistent scheduler를 이어 읽어 static tail, split/chunk tail과 worker 재배분을 구별한다.

- SGLang v0.5.18·`71de97b264b04dcd514cf904003028aefe9775c8` — `python/sglang/srt/layers/attention/flashinfer_mla_backend.py:830-879` — wrapper plan metadata — https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_mla_backend.py#L830-L879
- SGLang v0.5.18·`71de97b264b04dcd514cf904003028aefe9775c8` — `python/sglang/srt/layers/attention/flashinfer_backend.py:1244-1465` — extend/decode wrapper calls — https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_backend.py#L1244-L1404 — https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/attention/flashinfer_backend.py#L1405-L1465
- FlashInfer v0.6.17 snapshot·`a0a6b019b9b27d49d209f85d028a1ae5a9b347d7` — `flashinfer/attention/_core.py:94-213` — Python plan contract — https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/attention/_core.py#L94-L213
- FlashInfer v0.6.17 snapshot·`a0a6b019b9b27d49d209f85d028a1ae5a9b347d7` — `include/flashinfer/attention/prefill.cuh:2068-2084` — prefill grid-axis indexing — https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/attention/prefill.cuh#L2068-L2084
- FlashInfer v0.6.17 snapshot·`a0a6b019b9b27d49d209f85d028a1ae5a9b347d7` — `include/flashinfer/attention/blackwell/device/fmha.hpp:180-214` — Blackwell launcher — https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/attention/blackwell/device/fmha.hpp#L180-L214
- FlashInfer v0.6.17 snapshot·`a0a6b019b9b27d49d209f85d028a1ae5a9b347d7` — `include/flashinfer/attention/blackwell/kernel/fmha_tile_scheduler.hpp:70-100` — persistent scheduler — https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/attention/blackwell/kernel/fmha_tile_scheduler.hpp#L70-L100
- FlashInfer v0.6.17 snapshot·`a0a6b019b9b27d49d209f85d028a1ae5a9b347d7` — `include/flashinfer/norm.cuh:961-997` — token-grid norm launcher — https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/norm.cuh#L961-L997

### 40.9.5 좌표 해석의 핵심: GPU는 요청이 아니라 번역된 좌표를 실행한다

canonical fixture의 request 수, scheduled rows, capacity와 head/KV shape가 dispatcher와 launcher를 지나 grid·CTA·warp role로 번역되는 사슬이 끊기지 않아야 한다. M=8과 M=128의 row-slot 계산은 이 사슬의 시작이며 measured lane activity나 occupancy의 대체물이 아니다.

첫 divergence가 runner shape면 scheduler/packing, specialization이면 dispatch, grid/shared configuration이면 launcher, boundary output이면 tile coordinate와 predicate를 본다. static, split-KV, persistent와 cluster scheduler는 같은 grid 숫자에 서로 다른 논리 의미를 줄 수 있으므로 selected source와 artifact를 함께 고정한다.

이 원장을 유지하면 다음 장에서 같은 launch의 register, shared, L2와 HBM byte를 이어 볼 수 있다. 수정 뒤에는 동일 request fixture에서 logical coverage, output·memory safety와 end-to-end latency를 함께 재검증한다.

**vLLM과 FlashAttention의 geometry source 전수표.**

### 40.9.6 Python backend는 query shape와 kernel 세대를 함께 고른다

vLLM v0.27.1의 `FlashAttentionImpl.__init__`은 query head 수, KV head 수, head size, scale, dtype, sliding window와 선택된 FlashAttention version을 객체 상태로 만든다. 여기서 중요한 것은 이 생성자가 단순 wrapper가 아니라 이후 모든 forward에 적용할 shape·capability 계약을 고정한다는 점이다. 지원하지 않는 head size나 dtype을 native kernel에 보낸 뒤 실패시키는 것이 아니라 backend 구성 단계에서 가능한 경로를 좁힌다.

`forward`가 받는 query는 `[num_tokens,num_heads,head_size]` 형태다. 이 `num_tokens`가 우리 원장의 `M`에 해당하지만, KV cache에는 이전 context가 이미 있고 attention metadata에는 실제 token 수와 sequence 경계가 따로 있다. query의 첫 차원이 곧 sequence count가 아니며, KV length도 아니다. forward는 cache write와 attention 호출에 필요한 metadata를 결합하고 encoder/decoder 조건에 따라 경로를 나눈다.

vLLM의 varlen 호출로 내려가면 query, key, value 또는 cache view와 cumulative sequence lengths, maximum lengths, block table, causal/window 관련 값이 native wrapper에 전달된다. 이 지점에서 shape ledger의 의미를 대조한다. query rows 합은 M, cumulative query array의 차이는 각 sequence가 이번 step에 낸 rows, KV lengths는 이미 cache된 context를 포함한 attention 범위다. 세 값을 하나의 “batch length”로 뭉개면 decode 비용을 설명할 수 없다.

CUDA backend import 경계도 읽을 가치가 있다. vLLM의 `fa_utils.py`는 CUDA에서 `vllm.vllm_flash_attn.flash_attn_varlen_func`를 사용할 수 있는지 확인하고 symbol을 노출한다. Python `forward`에서 바로 `.cu` kernel로 점프하는 것이 아니라 이 wrapper와 extension binding을 통과한다. 빌드된 extension, selected FlashAttention version와 architecture가 달라지면 동일 Python class 아래 native 경로가 달라질 수 있다.

이 경로에서 관찰되는 효과는 분명하다. scheduler가 만든 M은 query tensor로 전달되지만, launcher를 결정하는 key에는 M 외에도 heads, D, dtype, causal/varlen, window와 capability가 들어간다. 따라서 `M=8`과 `M=128`이 반드시 같은 kernel family를 사용한다고 단정하지 않는다. 교육용 fixture는 같은 family를 가정해 tail 효과를 분리하는 도구이고, 실제 source walk에서는 selected version와 traits를 먼저 확인한다.

### 40.9.7 Hopper launcher는 grid·block·shared memory·stream을 한곳에서 닫는다

vLLM FlashAttention fork의 Hopper `flash_fwd_launch_template.h`는 params와 kernel traits를 바탕으로 kernel을 구성한다. 고정 소스 범위에는 scheduler metadata를 반영한 kernel params, `get_grid_shape`, `get_block_shape`, shared storage 크기, cluster/non-cluster launch와 stream 전달이 함께 보인다. 이곳이 Python 의미가 CUDA execution configuration으로 바뀌는 핵심 launcher 경계다.

읽는 순서는 grid 식부터가 아니다. 먼저 어떤 `Kernel_traits`와 scheduler가 template parameter인지 본다. 그 다음 params 안에서 실제 batch, heads, sequence lengths, split과 pointers가 무엇인지 확인한다. 마지막에 grid와 block shape가 traits·params의 어느 값을 소비하는지 본다. 같은 `get_grid_shape` 이름도 static scheduler와 persistent scheduler에서 반환 의미가 다를 수 있다.

dynamic shared memory는 tile algorithm의 scratch storage다. launcher가 요청한 bytes가 device/kernel의 허용 범위를 넘으면 invalid configuration 또는 launch failure가 날 수 있다. 이 실패는 KV cache OOM과 다르다. KV pool free bytes가 충분해도 CTA당 shared-memory requirement나 opt-in attribute가 맞지 않으면 launch가 거부될 수 있다. 반대로 shared-memory bytes가 크다고 무조건 OOM라고 부르지 않는다. shared memory는 CTA residency를 제한하는 SM 자원이다.

stream 인자는 temporal ownership를 연결한다. query/KV metadata가 준비된 stream과 kernel launch stream 사이의 ordering가 맞아야 하며 output consumer도 완료를 기다려야 한다. 여기서는 grid shape만 추적하더라도 launch record에서 stream을 버리지 않는다. 잘못된 stream ordering는 shape가 모두 맞아도 stale data나 race를 만들 수 있고, 이는 43장의 조사 대상이다.

### 40.9.8 tile scheduler가 blockIdx의 뜻을 결정한다

Hopper `tile_scheduler.hpp`의 static scheduler는 `blockIdx.x`, `.y`, `.z`를 `WorkTileInfo`의 batch, head, split 또는 tile coordinates로 바꾼다. 핵심은 CUDA가 축의 업무 의미를 정하지 않는다는 사실이다. CUDA는 세 정수 좌표를 제공할 뿐이며 library scheduler가 그 좌표를 logical attention work로 해석한다.

따라서 profiler에서 `grid=(g_x,g_y,g_z)`를 봤을 때 `g_x=batch`, `g_y=head`라고 관습으로 라벨링하지 않는다. 해당 specialization의 `get_grid_shape`와 `get_current_work`를 함께 읽는다. grouped heads, varlen batch, split count가 들어가면 같은 숫자도 다른 logical axis를 뜻할 수 있다. source version가 바뀌어 scheduler mapping이 바뀌면 dashboard의 축 해석도 갱신해야 한다.

work tile은 valid flag와 coordinates를 함께 가질 수 있다. grid를 rectangular하게 잡으면 일부 combinations가 실제 variable-length sequence에서 invalid할 수 있다. device scheduler나 kernel mainloop가 bounds를 검사해 건너뛴다. 이것도 logical idle work이지 자동으로 idle warp count가 아니다. invalid tile을 발견한 CTA가 즉시 다음 tile을 가져오는 persistent scheduler인지 그대로 return하는 static scheduler인지에 따라 비용이 달라진다.

### 40.9.9 SM90 kernel의 threadIdx는 producer와 consumer를 가른다

`flash_fwd_kernel_sm90.h`의 forward kernel은 `threadIdx.x`와 warp-group 정보를 이용해 producer/consumer 역할을 나눈다. producer 측은 mainloop에 필요한 data movement와 pipeline coordination를 맡고 consumer 측은 matrix work와 epilogue를 수행한다. 모든 threads가 동일한 `row=blockIdx*BLOCK_M+threadIdx` 식으로 output row 하나씩 담당하는 교육 kernel과 근본적으로 다르다.

이 코드를 읽을 때 “어느 warp가 몇 row를 맡는다”만 찾지 않는다. producer가 어떤 tensors를 stage하고 어느 barrier/pipeline state를 갱신하는지, consumer가 어떤 tile coordinates로 MMA와 softmax/output을 수행하는지, epilogue store predicate가 variable length와 tail을 어떻게 반영하는지 본다. warp specialization는 역할 분업이므로 producer lanes를 non-compute idle로 세지 않는다.

`M=8`에서 query tile의 valid rows가 적어도 K/V mainloop는 각 row의 context를 처리한다. exact instruction activity는 sequence length와 tile traits에 달렸다. 따라서 이 source가 증명하는 것은 thread roles가 row-per-thread가 아니라는 사실과 launch shape가 traits/scheduler로 계산된다는 사실이다. 12.5% logical row fill에서 12.5% GPU utilization이라는 수치는 나오지 않는다.

### 40.9.10 vLLM 경로의 first-divergence 지점

예상 `M=8`인데 Python query가 `[9,H,D]`라면 scheduler/runner packing부터 본다. Python은 `[8,H,D]`인데 cumulative query lengths의 마지막 값이 다르면 metadata construction 문제다. native params까지 8이 유지되는데 예상과 다른 kernel version가 선택되면 capability/dtype/head-size dispatch를 본다. specialization도 예상대로인데 grid가 다르면 `get_grid_shape`와 scheduler type을 본다. grid는 맞지만 output boundary가 깨지면 device work coordinate와 tail store predicate를 본다.

이 순서는 “kernel이 이상하다”는 넓은 가설을 좁힌다. 각 경계에서 같은 request correlation와 shape ledger를 비교하면 최초로 값이나 의미가 달라진 위치가 생긴다. downstream wrong output은 그 뒤의 결과일 수 있다. source coordinate와 runtime 관측을 혼동하지 않되, 실제 운영에서는 이 원장 필드를 계측해 source의 예상과 대조한다.

**SGLang과 FlashInfer의 static·persistent source 전수표.**

### 40.9.11 plan은 실행 전에 shape와 workspace를 계약한다

SGLang v0.5.18의 FlashInfer MLA backend는 batch indptr, lengths, head 수, head dimension, dtype 같은 metadata를 wrapper의 `plan`에 전달한다. plan은 단순 캐시 힌트가 아니다. 이후 run이 사용할 schedule와 workspace를 현재 batch shape에 맞춰 준비하는 경계다. plan에 들어간 indptr와 실제 query tensor rows가 어긋나면 launch arithmetic 이전에 logical mapping이 깨진다.

FlashInfer v0.6.17 snapshot의 Python `_core.py`에 있는 plan 경로는 head/dtype을 검증하고 host indptr를 다루며 native module plan으로 내려간다. host metadata copy와 synchronization가 나타나는 이유는 device kernel이 사용할 schedule를 안전한 시점에 만들기 위해서다. plan/run 분리는 launch 횟수만의 문제가 아니라 metadata lifetime와 workspace ownership 문제다.

SGLang `flashinfer_backend.py`는 extend/prefill wrapper와 decode wrapper를 구분해 호출한다. 그래서 prefill `M=128`과 decode `M=8`이 같은 교육용 kernel family로 간다는 가정은 SGLang 실제 경로의 보편 사실이 아니다. wrapper 선택, plan arguments와 native module이 다를 수 있다. 비교하려면 logical fixture는 같게 유지하되 selected wrapper와 launch family를 별 필드로 기록한다.

extend에서도 모든 request가 같은 query length를 갖지 않는다. indptr difference로 per-sequence rows를 복원하고 total M과 맞춘다. decode wrapper는 보통 sequence마다 적은 query row를 처리하지만 KV lengths와 pages는 서로 다르다. batch size만 비교하면 긴-context decode와 짧은-context decode의 device work 차이를 숨긴다.

### 40.9.12 prefill device function은 grid의 세 축을 직접 소비한다

FlashInfer `prefill.cuh`의 고정 범위는 device function이 `threadIdx`, `blockIdx.x`, chunk를 나타내는 `blockIdx.y`, KV head를 나타내는 `blockIdx.z`, 그리고 `gridDim.y/z`를 소비하는 모습을 보여 준다. 이 한 구간만으로도 `grid.y=head`라는 상투적 해석이 틀릴 수 있음을 확인할 수 있다. 여기서는 y가 chunk/split 역할을 갖고 z가 KV head 역할을 갖는다.

split 또는 chunk axis가 생기면 하나의 query tile이 여러 CTAs에서 부분적으로 처리될 수 있다. partial outputs와 log-sum-exp 같은 merge 정보가 workspace에 저장되고 나중에 결합된다. 이때 launched CTA 수를 unique output row 수로 나누어 “중복 계산”이라고 부르면 안 된다. 긴 KV 범위를 병렬화하기 위해 reduction 가능한 부분 문제로 나눈 것이다.

그러나 split 수가 늘면 공짜 병렬성도 아니다. partial output write, merge kernel 또는 synchronization, workspace bytes가 늘어난다. query rows가 작고 KV가 길 때 independent query tiles가 부족하므로 split-KV가 parallel supply를 늘릴 수 있지만, KV가 짧거나 batch가 충분히 크면 merge overhead가 이득을 상쇄할 수 있다. 어느 split이 선택되는지는 plan과 heuristic source를 읽고 실제 launch record로 검증해야 한다.

tail predicate도 두 축에 있을 수 있다. 마지막 query tile의 invalid rows와 마지막 KV chunk의 invalid columns를 각각 보호한다. query tail만 맞아도 KV chunk boundary가 틀리면 out-of-bounds 또는 잘못된 softmax normalization가 생긴다. `M=63/64/65`와 함께 KV length의 chunk boundary 앞뒤도 fixture로 둔다.

### 40.9.13 Blackwell launcher와 persistent scheduler는 ceil-div 반례다

FlashInfer의 Blackwell `fmha.hpp`는 grid, block, shared-memory 크기를 구성해 device kernel을 launch하는 경계를 보여 준다. 여기서 architecture-specific traits와 scheduler가 선택되면 CUDA의 기본 hierarchy는 같아도 grid 식과 warp roles는 Hopper 또는 generic prefill과 달라질 수 있다. “CUDA 13이라 grid 의미가 바뀌었다”가 아니라 “선택된 Blackwell specialization의 launcher와 scheduler가 다르다”고 써야 한다.

`fmha_tile_scheduler.hpp`의 persistent scheduler는 grid size에 `num_sm`을 사용할 수 있다. logical tiles가 100이어도 100 CTAs를 launch해 blockIdx와 tile을 고정 매핑하지 않고, 예를 들어 device SM 수를 기준으로 worker CTAs를 launch한 뒤 각 CTA가 여러 work tiles를 가져갈 수 있다. 반대로 logical tiles가 SM 수보다 적으면 일부 workers가 얻을 일이 없을 수 있다.

이 반례에서 세 수를 분리한다. logical tile count는 request shape와 tile size에서 나온다. worker CTA count는 persistent launch policy에서 나온다. total claimed work count는 device scheduler의 반복 결과다. profiler의 grid만으로 logical tile count를 역산할 수 없고, `blockIdx.x`만으로 output row interval을 정할 수 없다.

persistent scheduling의 장점은 variable work를 workers 사이에 재분배하고 launch geometry를 hardware supply에 맞출 수 있다는 데 있다. 하지만 queue claim overhead, fairness와 tail workers가 생긴다. 이 장에서는 성능 우열을 수치로 주장하지 않는다. 중요한 독해 규칙은 static mapping equation을 발견하기 전까지 `ceil_div(M,tile)`을 grid 공식으로 가정하지 않는 것이다.

### 40.9.14 norm kernel은 단순 mapping의 좋은 대조군이다

FlashInfer `norm.cuh`의 launcher는 tokens 수를 grid로, hidden dimension을 바탕으로 block threads와 shared memory를 정하는 비교적 단순한 사례다. attention처럼 sequence indptr, KV chunks와 heads가 여러 grid axes에 얽히지 않으므로 Python/native parameter가 execution configuration으로 바뀌는 과정을 연습하기 좋다.

이 경로에서 `grid(tokens)`이면 한 token row와 CTA를 대응시키는 해석이 source로 뒷받침될 수 있다. block은 hidden dimension의 elements를 협력 처리하고 warp multiple과 최대 thread 제한을 고려한다. shared memory는 hidden dimension과 element type에 따라 정해진다. `M=8`이면 grid.x가 8일 수 있지만 CTA당 threads는 hidden dimension에 따라 수백 개일 수 있다. token 한 개가 thread 한 개라는 결론은 여전히 나오지 않는다.

norm과 attention을 나란히 두는 목적은 kernel마다 mapping을 다시 읽어야 함을 보여 주는 것이다. 같은 tensor 첫 차원 M이 norm에서는 CTA count로 직접, attention에서는 tile count·head·split·persistent queue의 입력으로 간접 번역될 수 있다. framework 이름이나 연산 이름이 아니라 launcher와 indexing이 최종 계약을 정한다.
