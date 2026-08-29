# 47장. kernel 병목·race·오답을 first divergence로 진단하기

“GPU utilization이 낮아서 느립니다.” 보고서는 이 문장으로 끝났지만 아무도 다음 행동을 정할 수 없었다. scheduler가 GPU에 일을 늦게 건넨 것인지, kernel grid가 GPU를 채우지 못한 것인지, warp가 memory dependency를 기다린 것인지, graph replay가 옛 metadata를 읽은 것인지가 모두 남아 있었다. utilization은 관측 window 안에서 device가 바빴다는 요약일 수 있지만 원인이 아니다.

이 장의 도구는 first divergence, 즉 정상 fixture와 실패 fixture가 처음 달라지는 경계를 찾는 ledger다. request shape에서 backend key, launch, address와 dependency, 첫 arithmetic checkpoint, output store와 completion까지 사건을 나열한다. 그 위에 profiler와 sanitizer가 실제로 관측할 수 있는 범위만 붙인다. 입문·운영 경로는 47.1~47.3절을, CUDA·source 경로는 47.4~47.7절을 따라가면 된다. 실제 runtime 수치를 만들지 않고 재현·관측 설계만 제시한다.

## 47.1 utilization이 낮다는 보고서가 설명하지 못한 것

긴 context의 ITL p99가 늘었다고 하자. monitoring의 GPU utilization은 45%였고 팀은 batch를 키우기로 했다. 그러나 timeline을 열어 보니 두 가능성이 있었다. 첫 경우에는 host scheduler와 graph selection 사이 gap이 길어 GPU가 실제로 할 일을 받지 못했다. 둘째 경우에는 짧은 kernel이 계속 실행됐지만 sampling window가 activity를 낮게 보였다. 셋째 경우에는 큰 kernel 하나가 memory dependency에 오래 머물렀다. 세 현상에 같은 batch 확대를 적용하면 queue latency, memory pressure 또는 tail underfill을 악화시킬 수 있다.

utilization을 쓰려면 먼저 정의와 window가 필요하다. device active time인지 engine-specific activity인지, 어느 sampling interval인지, 여러 process를 합쳤는지 확인한다. 값이 낮다는 사실은 “왜 issue하지 못했는가”를 말하지 않는다. kernel launch가 없었는지, resident warp가 적었는지, resident warp가 dependency 때문에 eligible하지 않았는지 분기한다.

occupancy도 같은 함정이 있다. occupancy는 architecture resource 한도에 비해 resident active warp를 얼마나 둘 수 있는지 나타내는 capacity ratio다. warp가 useful instruction을 매 cycle issue한다는 뜻이 아니다. register/shared-memory 사용이 낮아 occupancy가 높아도 grid에 CTA가 몇 개뿐이면 GPU 전체에 일이 부족할 수 있다. warp가 long memory dependency나 barrier에 막히면 resident이면서 idle할 수 있다.

반대로 낮은 occupancy가 항상 병목은 아니다. tile당 register와 shared memory를 많이 써 data reuse를 높이고 memory traffic을 줄였다면 적은 warp로도 throughput이 좋을 수 있다. occupancy를 올리려고 tile을 줄이면 transaction·synchronization과 tail이 늘 수 있다. 최적화 전에는 낮은 occupancy가 실제 issue eligibility와 latency를 제한한다는 경쟁 증거가 필요하다.

오답 사건에서도 broad label은 위험하다. `race`라고 부르기 전에 wrong value가 처음 나타난 output element와 그 writer를 찾는다. out-of-bounds write가 다른 request buffer를 덮은 것인지, graph replay가 stale static address를 읽은 것인지, missing barrier가 shared tile을 반쯤 읽게 했는지 구분한다. sanitizer가 race를 보고하지 않았다고 dependency protocol이 모두 올바르다는 뜻도 아니다.

## 47.2 first-divergence ledger와 재현 fixture

모든 조사에서 다음 사건 열을 채운다.

```text
request/step
→ logical shape
→ backend/kernel key
→ grid/block/dynamic smem/stream
→ input address/range/generation
→ dependency edge
→ first arithmetic checkpoint
→ output address/range/generation
→ completion/error observation
```

### 47.2.1 정상과 실패에서 고정할 것

model과 source revision, tokenizer 결과 token id, batch/Q/KV/head/dtype shape를 고정한다. scheduler trace, selected backend, toolkit·driver·GPU, graph key, seed와 clock/power policy도 기록한다. 한 번에 한 변수만 바꾼다. `M=64`는 정상이고 `M=65`가 오답이면 다른 batch·prompt·seed까지 바꾸지 않는다.

fixture의 입력 원문보다 tensor shape와 digest가 중요할 수 있다. privacy를 지키며 token id digest와 sentinel pattern을 남긴다. output은 exact equality가 필요한 integer/index, 허용 tolerance가 있는 floating result와 NaN/Inf를 분리한다. reference implementation이 다른 arithmetic order를 쓰면 tolerance와 first divergence 의미를 명시한다.

### 47.2.2 first divergence가 root cause는 아니다

정상과 실패가 처음 다른 지점이 kernel specialization 선택이라고 하자. 이것은 조사 시작점이지 specialization이 틀렸다는 결론이 아니다. 실패 fixture가 다른 shape라 다른 kernel을 고르는 것은 정상일 수 있다. 그 안의 tail predicate, address와 arithmetic checkpoint를 더 비교해야 한다.

반대로 output에서 처음 차이를 발견해도 writer보다 앞선 input generation이 이미 틀렸을 수 있다. graph replay가 옛 metadata를 읽었다면 arithmetic은 주어진 입력에 대해 정확하다. ledger는 output에서 input·dependency로 되감는 경로를 제공한다.

### 47.2.3 address와 generation을 함께 기록한다

pointer 값이 같다는 것은 같은 payload라는 뜻이 아니다. allocator가 static graph buffer 주소를 재사용해도 generation과 active length가 달라진다. request A와 B가 같은 address를 차례로 사용하면 replay 전에 metadata write가 B generation을 publish했는지 확인한다.

input range는 `base`, byte extent, stride와 logical owner를 가진다. output도 destination capacity와 sentinel canary를 둔다. `M=65` tail CTA가 output 65개만 써야 하는데 vector store가 68개를 쓰면 canary가 first corrupted byte를 보여 준다. crash가 없어도 인접 buffer corruption을 찾을 수 있다.

### 47.2.4 dependency edge는 stream 이름보다 구체적이어야 한다

“같은 stream이라 안전”이라고만 쓰지 않는다. metadata write event, replay stream wait, async copy barrier와 consumer instruction처럼 producer→completion→consumer를 적는다. 다른 stream이면 event/graph dependency가 필요하고, 같은 stream도 external producer가 host에서 값을 늦게 채우면 별도 lifetime이 있다.

synchronization을 추가해 오류 위치를 좁히는 진단은 가능하지만 production 수정과 구분한다. device-wide sync로 오답이 사라지면 ordering 가설이 강해지지만 모든 overlap을 없앤다. 정확한 missing edge를 찾아 최소 wait/barrier로 복구해야 한다.

## 47.3 질문에 따라 Nsight와 Compute Sanitizer를 고른다

도구 이름부터 실행하면 많은 counter와 경고가 쌓이고 원인 선택은 더 어려워진다. 먼저 ledger의 빈 칸을 정하고 그 칸을 관측하는 도구를 고른다.

### 47.3.1 Nsight Systems: 귀속과 timeline

질문이 “GPU가 왜 비었는가”라면 CPU thread, CUDA API call, stream kernel/memcpy와 collective timeline이 필요하다. Nsight Systems는 host launch gap, stream overlap과 copy/collective 위치를 같은 시간축에 놓는 데 적합하다. kernel 내부 lane address나 warp stall의 source root cause를 직접 증명하지 않는다.

timeline에서 host가 launch하지 않은 gap은 scheduler·Python·graph selection 후보로, API는 제출됐지만 kernel start가 늦은 gap은 dependency·queue 후보로 간다. kernel duration이 길면 그다음 Nsight Compute 또는 source worksheet로 들어간다. 모든 gap을 launch overhead라고 부르지 않는다.

### 47.3.2 Nsight Compute: 한 kernel의 launch·warp·memory

질문이 “이 kernel 안에서 어떤 resource·instruction stage가 제한되는가”라면 launch statistics, occupancy/resource, warp state, memory workload와 source correlation을 본다. counter collection은 kernel을 재실행하거나 serialization을 만들 수 있어 원래 timing을 바꾼다. production p99와 profile run duration을 직접 비교하지 않는다.

stall reason은 표본 시점에 warp가 issue하지 못한 이유다. long scoreboard가 높다고 source line 하나가 root cause인 것은 아니다. load latency, address scatter, cache miss 또는 dependent chain이 후보가 된다. selected source stage와 memory counter를 함께 연결한다.

### 47.3.3 Compute Sanitizer: memory와 synchronization 오류의 범위

memcheck는 invalid global/local/shared access와 일부 memory error를 찾는 데 쓴다. racecheck는 shared-memory data hazard를 중심으로 보고한다. initcheck는 uninitialized device global memory access를, synccheck는 synchronization primitive의 잘못된 사용을 찾는 범위가 있다. 한 tool의 무보고를 모든 memory ordering과 semantic correctness의 증명으로 쓰지 않는다.

예를 들어 graph replay가 stale but valid address의 옛 generation을 읽으면 memory range는 유효해 memcheck가 조용할 수 있다. 서로 다른 kernel/stream 사이 missing event가 racecheck의 shared-memory hazard 범위를 벗어날 수 있다. sanitizer는 contract violation 일부를 잡는 강한 도구지만 application identity와 generation을 알지 못한다.

instrumentation은 timing과 scheduling을 바꾸어 race를 숨기거나 드러낼 수 있다. sanitizer에서만 재현되지 않는다고 race를 기각하지 않는다. sentinel, generation trace와 explicit dependency fixture를 함께 둔다.

### 47.3.4 artifact 도구와 compiler report

`ptxas -v` resource report는 registers, spill/local memory와 static resource 후보를 보여 준다. `cuobjdump`와 `nvdisasm`은 binary target과 SASS inspection에 쓴다. source의 vector type이 실제 instruction으로 어떻게 내려갔는지, compiler change가 register를 바꿨는지 확인할 수 있다. instruction 하나를 memory transaction이나 performance 원인과 일대일로 놓지 않는다.

## 47.4 launch·async error와 wrong value를 분리한다

### 47.4.1 launch 전에 실패하는 네 조건

kernel key가 정해져도 launch configuration이 device limit을 넘으면 실행되지 않는다. block thread 수, static+dynamic shared memory, register/resource, grid와 opt-in shared-memory attribute를 확인한다. “shared tile을 키운 뒤 invalid configuration”이면 selected specialization의 compiled static smem과 launcher가 요청한 dynamic smem을 더한다.

artifact 문제도 launch 가능성 축이다. device용 kernel image가 없거나 symbol/module load가 실패하면 arithmetic에 들어가지 않는다. 44장에서 만든 code-object inventory와 driver JIT ledger를 재사용한다. 성능 profiler를 붙이기 전에 kernel이 실행됐는지부터 닫는다.

### 47.4.2 asynchronous error의 보고 위치

CUDA work submit은 비동기일 수 있다. kernel A의 illegal access가 launch call에서 즉시 보고되지 않고 뒤의 synchronization, memcpy 또는 unrelated API check에서 나타날 수 있다. host stack의 error line을 offender launch로 단정하지 않는다. launch sequence와 stream dependency를 기록하고 진단용 synchronization을 좁혀 first failing boundary를 찾는다.

모든 launch 뒤 device synchronize를 넣으면 위치는 좁혀지지만 race timing과 overlap이 바뀐다. binary search처럼 synchronization 지점을 옮겨 offending interval을 줄이고, 최종 수정에서는 올바른 event/barrier와 bounds predicate를 넣는다. 진단 build와 production performance 결과를 섞지 않는다.

### 47.4.3 wrong value의 첫 checkpoint

attention이라면 QK score, running max, sum-exp, normalized output 같은 algorithm checkpoint를 고를 수 있다. quant GEMM이면 loaded packed tile, dequant scale, partial accumulator와 epilogue store가 checkpoint다. 모든 element를 dump하는 대신 작은 sentinel row와 digest·NaN count를 둔다.

checkpoint A까지 정상이고 B에서 갈라지면 A 이전 address/load 가설을 낮추고 A→B arithmetic·dependency를 본다. 그러나 instrumentation이 layout·register를 바꿀 수 있으므로 debug output가 race를 숨길 수 있다. source predicate와 sanitizer, boundary fixture를 함께 사용한다.

### 47.4.4 race와 stale generation의 차이

두 thread가 같은 shared location을 잘못 읽고 쓰는 data hazard는 racecheck 후보다. graph replay가 유효한 static buffer에서 이전 request generation을 읽는 것은 application lifetime 오류다. address는 valid하고 write/read가 stream-ordered일 수도 있지만 active length update가 누락됐다. sanitizer 분류가 다르다.

ledger에 pointer뿐 아니라 generation, logical owner와 active extent를 넣는 이유다. output이 이전 token과 일치한다면 random memory corruption보다 stale metadata 가설이 강해진다. replay 전 metadata write→graph consumer edge를 확인한다.

## 47.5 네 serving stack의 관측 경계를 읽는다

### 47.5.1 vLLM: graph replay와 persistent address

vLLM graph path는 captured graph가 읽는 buffer address identity를 보존하고 replay 전에 새 metadata 값을 채운다. source에서 `graph.replay()` 호출만 읽으면 input mutation과 completion edge를 놓친다. descriptor가 어떤 static tensor를 소유하고 active request 수·token count를 어떻게 갱신하는지 앞 caller까지 따라간다. [vLLM v0.27.1 — `vllm/v1/worker/gpu/cudagraph_utils.py:360-410` — graph descriptor replay](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu/cudagraph_utils.py#L360-L410)

full/piecewise/eager fallback은 runtime key에 따라 달라질 수 있다. graph miss가 곧 오류는 아니지만 host launch와 memory behavior가 달라진다. selected execution mode와 fallback reason을 request shape에 묶는다. graph replay가 빠르다는 평균만으로 stale buffer correctness를 가리지 않는다.

vLLM source의 persistent metadata 주석은 graph replay가 같은 buffer를 읽으므로 producer write order가 중요함을 드러낸다. [vLLM v0.27.1 — `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py:590-615` — replay metadata ordering contract](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py#L590-L615) CUDA 중심 책에서 ROCm backend를 성능 대상으로 다루지 않지만, application-level persistent buffer lifetime을 보여 주는 source contract로 제한해 읽는다.

### 47.5.2 SGLang: kernel API logging과 barrier source

SGLang kernel API logging wrapper는 Python call의 argument/dispatch를 관측하는 경계를 제공한다. 이것은 GPU 내부 transaction이나 completion을 자동 측정하지 않는다. selected wrapper, shape와 option을 source key에 연결하는 데 쓴다. [SGLang v0.5.18 — `python/sglang/kernels/kernel_api_logging.py:1-120` — kernel API debug logging](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/kernel_api_logging.py#L1-L120)

CuTe attention source에는 producer/consumer stage 사이 barrier가 반복된다. `cute.arch.barrier()` 존재만으로 모든 shared data가 안전하다고 결론내리지 않는다. 어떤 warp/CTA가 어떤 tile을 쓰고 누가 기다리는지, predicate로 일부 participant가 barrier를 건너뛰지 않는지 본다. [SGLang v0.5.18 — `python/sglang/kernels/ops/attention/cutedsl_kda.py:130-220` — tile load·barrier 구간](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/ops/attention/cutedsl_kda.py#L130-L220)

FlashInfer backend를 감싸는 plan/run 경로에서는 workspace와 metadata plan이 request shape로부터 만들어지고 run에서 소비된다. plan과 run 사이 input generation이 맞는지, graph capture에서 pointer가 stable한지 확인한다. Python plan 성공은 native launch success와 다르며 wrapper return은 async completion과 다르다.

### 47.5.3 FlashInfer: policy·workspace·native launch

FlashInfer kernel family는 CTA tile, split, workspace를 plan하고 native/JIT module을 dispatch한다. 성능 회귀에서 Python API 이름이 같아도 plan key와 selected specialization이 달라질 수 있다. batch, head dimension, sequence distribution, page size와 SM을 key에 둔다.

workspace가 부족하면 validation error, reallocation 또는 다른 split policy가 생길 수 있다. workspace base·capacity와 required byte를 ledger에 넣는다. allocator growth 때문에 첫 request만 느린 현상과 JIT compile을 구분한다. module load/JIT는 44장 artifact timeline, workspace allocation은 memory timeline, graph capture는 execution timeline에 둔다.

native launcher 뒤 error check가 없다면 asynchronous error가 상위 synchronization에서 보일 수 있다. source에서 실제 error/log 노출만 기록한다. 존재하지 않는 log 문자열을 진단 절차에 만들지 않는다.

### 47.5.4 llama.cpp: graph update와 fallback

llama.cpp CUDA path는 graph capture 뒤 기존 executable을 update하고 update가 실패하면 instance를 파괴·재생성하는 경로를 가진다. graph compatibility가 달라진 request는 update result를 통해 fallback lifecycle을 탄다. [llama.cpp v0.2.0 — `ggml/src/ggml-cuda/ggml-cuda.cu:2610-2652` — graph exec update와 reinstantiate](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/ggml-cuda.cu#L2610-L2652)

실행 경로 후반에는 graph instantiate와 `cudaGraphLaunch`가 stream에 들어간다. [llama.cpp v0.2.0 — `ggml/src/ggml-cuda/ggml-cuda.cu:4180-4222` — graph instantiate/launch](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/ggml-cuda.cu#L4180-L4222) launch call 반환과 graph work completion을 같게 놓지 않는다. error가 뒤 event synchronization에서 보일 수 있다.

event/copy synchronization은 backend scheduler와 tensor lifetime을 잇는다. graph update 실패가 성능상 capture overhead를 만들 수 있지만 correctness fallback일 수 있다. update success/instantiate count, selected graph key와 request shape를 관측하고 무조건 update failure를 bug라고 하지 않는다.

## 47.6 일곱 사건: 증상에서 first divergence까지

### 47.6.1 사건 1: `M=65`에서만 wrong answer가 났다

GEMM epilogue kernel은 M=64까지 reference와 일치했지만 M=65에서 마지막 row와 다음 buffer의 첫 값이 달라졌다. M=66~68도 유사했고 128에서는 다시 정상이었다. 팀은 floating reduction 순서 차이라고 보았지만 tolerance를 크게 해도 인접 buffer sentinel이 바뀌었다.

최소 fixture는 같은 input seed와 K/N, M만 64와 65로 나눴다. 예상 invariant는 output writer가 `[0,M*N)` 범위만 수정한다는 것이다. output 뒤 64-byte canary를 두고 first corrupted byte를 기록했다. divergence는 arithmetic checkpoint가 아니라 마지막 CTA store였다.

source에서 grid는 `ceil_div(M, TILE_M)`였고 마지막 CTA도 full vector store를 실행했다. load에는 `row<M` predicate가 있었지만 epilogue store의 vector 전체에는 predicate가 없었다. M=64에서는 모든 CTA가 full이고 bug가 숨었다. 65에서는 row 65 이후 lane이 canary를 덮었다.

경쟁 가설은 reduction tolerance, misaligned load와 stale input이었다. canary corruption과 first bad address가 output bound 밖이라는 사실로 세 가설을 낮췄다. memcheck가 invalid store를 보고할 수 있는 fixture지만 보고 위치는 뒤 sync일 수 있다. launch 직후 진단 sync로 offender interval을 좁힌다.

수정은 마지막 CTA의 scalar tail로 모두 바꾸는 것뿐 아니라 vector store가 온전히 유효한 구간과 masked tail을 분리하는 것이었다. output allocation padding이 있더라도 logical isolation을 위해 다른 request row를 읽거나 쓰지 않는다.

종료 조건은 M=65 한 값이 아니다. `TILE_M-1`, `TILE_M`, `TILE_M+1`, vector width 경계와 subview offset에서 exact/tolerance output, canary, sanitizer를 통과한다. fast full-tile path가 유지되고 tail overhead가 production shape에서 허용되는지도 확인한다.

### 47.6.2 사건 2: graph replay에 이전 요청 token이 섞였다

eager mode에서는 정상이지만 graph replay에서만 첫 token이 이전 request의 token과 일치했다. batch size가 capture bucket보다 작을 때 빈 slot 주변에서 재현됐다. memory address는 모두 valid해 memcheck는 조용했다.

ledger는 request B token metadata write, static input buffer generation, graph replay와 attention consumer를 적었다. first divergence는 B의 active length였다. token buffer 일부는 갱신됐지만 captured graph가 읽는 active-length tensor의 tail slot은 A 값을 유지했다. graph는 static address를 정확히 읽고 계산도 정확히 했다.

경쟁 가설은 sampling seed, KV cache collision과 race였다. eager 동일 seed가 정상이고 first logits divergence가 padded slot contribution에서 나타나 seed를 기각했다. KV generation은 맞았다. shared race report도 없었다. stale-but-valid metadata와 missing producer→replay edge가 남았다.

수정은 replay 전에 active range 전체를 B generation으로 갱신하고 unused slot을 neutral value로 초기화한 뒤 metadata write stream과 replay stream 사이 dependency를 보존하는 것이었다. host에서 length scalar만 바꾸고 device tensor copy를 빼먹지 않는다.

종료 fixture는 capture bucket보다 1 작은 batch, 빈 batch slot 재사용, cancellation 직후 새 request, 여러 generation 교대다. output exactness와 metadata generation trace, graph/eager 일치를 함께 본다. device-wide sync 없이 올바른 edge로 안정돼야 한다.

### 47.6.3 사건 3: shared tile을 키운 뒤 launch가 실패했다

attention tile을 키워 data reuse를 늘렸더니 특정 GPU에서 `invalid configuration argument` 또는 shared-memory 관련 launch failure가 났다. 작은 tile은 정상이고 compiler build도 성공했다. 첫 가설은 새 GPU image가 없다는 것이었다.

selected symbol은 device에서 load됐고 failure는 launch configuration 단계였다. ledger에 static shared memory, launcher의 dynamic smem, block threads와 opt-in attribute를 넣었다. 합계가 default per-block 한도를 넘었지만 kernel attribute로 larger dynamic shared memory opt-in을 설정하지 않았거나 device 최대보다 컸다.

no-image 가설은 module/symbol load가 성공하고 작은 dynamic smem으로 같은 symbol이 launch되는 것으로 기각됐다. register pressure는 launch failure보다 occupancy 후보였다. first divergence는 tile predicate가 큰 specialization을 고른 뒤 required smem 계산이었다.

수정 후보는 tile 축소, stage 수 감소, opt-in attribute 설정 또는 device별 specialization fallback이다. opt-in은 device가 지원하는 최대 안에서만 가능하다. shared memory를 줄이면 global traffic과 synchronization이 바뀌므로 성능 ledger를 갱신한다.

종료 조건은 supported GPU matrix에서 selected specialization의 static+dynamic smem, attribute와 device limit이 일치하는 것이다. boundary tile, head dimension과 stage 수를 fixture로 둔다. fallback이 선택되면 reason과 SLO도 확인한다.

### 47.6.4 사건 4: occupancy는 높은데 throughput이 낮았다

kernel A는 occupancy가 높았지만 kernel B보다 token throughput이 낮았다. 팀은 occupancy counter가 잘못됐다고 보거나 block size를 더 늘리려 했다. 그러나 occupancy는 resident capacity이고 useful issue·work 분배를 말하지 않았다.

첫 분기는 grid CTA 수였다. 작은 batch에서 CTA가 SM 수보다 적어 일부 SM은 일하지 않았다. active SM의 occupancy는 높을 수 있지만 GPU 전체 work가 부족했다. batch를 키운 fixture에서는 CTA 수가 늘었지만 warp issue eligibility가 낮고 long dependency가 나타났다.

경쟁 가설은 insufficient parallelism, memory latency, barrier serialization과 instruction dependency였다. Systems timeline은 host gap이 없고 kernel duration이 지배함을 보였다. Compute 관측 설계에서는 total CTA, active warps, eligible/issued, memory sectors/hit와 barrier sample을 함께 본다. stall reason 하나를 root cause로 복사하지 않는다.

source에서 한 warp가 K tile load 뒤 즉시 dependent compute를 했고 다른 warp가 숨길 independent work가 부족했다. 더 큰 block은 occupancy 숫자를 유지해도 같은 dependency chain을 늘렸다. tile을 작게 하거나 producer/consumer pipeline stage를 바꾸는 후보를 세웠다.

종료 조건은 occupancy 상승이 아니다. 작은/큰 batch에서 CTA distribution, issue eligibility, memory byte와 kernel time이 예상대로 바뀌고 end-to-end throughput·latency가 개선돼야 한다. resource 변경 후 output과 barrier safety도 재검증한다.

### 47.6.5 사건 5: register가 늘어난 뒤 느려졌다

compiler/toolkit upgrade 후 kernel register report가 thread당 증가했고 duration도 늘었다. 팀은 register 수가 늘어 occupancy가 내려간 것이 원인이라고 결론냈다. 하지만 generated code에는 spill load/store도 달라졌고 selected tile까지 변했다.

ledger는 same source·target·flags인지 먼저 확인했다. toolkit 외 dependency와 dispatch를 고정한 artifact 비교에서 ptxas register, spill/local memory, static shared와 binary symbol을 둔다. runtime에서는 occupancy limit, local-memory traffic, instruction count와 issue를 경쟁 가설로 둔다.

register 증가가 occupancy threshold를 넘지 않았다면 resident CTA 수는 그대로일 수 있다. 반대로 spill이 줄었다면 register 증가가 이득일 수도 있다. duration 증가는 extra instruction 또는 memory schedule에서 왔을 수 있다. “register가 많다=느리다”를 기각할 수 있는 경우다.

수정으로 launch bounds나 max register를 강제하면 compiler가 spill을 늘릴 수 있다. tile/stage를 조정해 live range를 줄이거나 source lifetime을 좁히는 방법과 비교한다. 특정 compiler만 탓하기 전에 undefined behavior와 race가 scheduling 변화로 드러난 것은 아닌지도 본다.

종료 조건은 resource table, spills/local traffic, occupancy·issue와 duration의 인과가 A/B에서 맞고 correctness fixture가 통과하는 것이다. compiler flag 하나의 kernel win이 다른 shape specialization을 해치지 않는지 production key로 확인한다.

### 47.6.6 사건 6: DRAM bytes가 갑자기 늘었다

logical KV payload와 output shape는 같은데 kernel DRAM bytes가 늘었다. “L2 cache가 나빠졌다”는 가설이 먼저 나왔다. 그러나 requested bytes 자체, coalescing, tail, cache reuse와 eviction을 분리하지 않은 설명이었다.

42장의 주소 worksheet를 가져온다. useful byte, lane requested interval과 sectors를 source stride·layout·tail에서 계산한다. requested sectors가 늘면 address/layout 가설을 본다. requested는 같고 L2 hit가 내려 DRAM만 늘면 working set·reuse·eviction 후보로 간다. intermediate write나 spill이 늘었으면 KV load 외 traffic이다.

first divergence는 backend가 NHD fast path에서 per-head/tail path로 바뀐 것이었다. head dimension은 같았지만 per-head scale stride가 0이 아니어서 다른 branch를 탔다. data payload는 같아도 scale load와 warp mapping, tail instruction이 달라졌다.

경쟁 가설은 L2 eviction, page size와 register spill이었다. cold/warm cache 관측 설계, source branch와 ptxas report로 분리한다. cache hit rate가 높아도 requested byte가 늘면 DRAM pressure가 남을 수 있다. DRAM byte만 보고 HBM burst를 추정하지 않는다.

수정은 scale layout을 합치는 것, specialization 추가 또는 policy 유지였다. numeric scale semantics를 바꾸면 correctness에 영향이 있으므로 단순 memory optimization이 아니다. 종료 조건은 useful/requested/DRAM ledger, selected branch, output tolerance와 serving memory/latency를 함께 통과하는 것이다.

### 47.6.7 사건 7: 첫 요청만 느렸다

pod 첫 요청은 수십 초, 이후는 빨랐다. 원인 후보는 CUDA Graph capture, PTX/Triton JIT, cubin download, allocator/workspace growth와 autotune이었다. 모두 warm 후 사라져 평균 latency로는 구분되지 않았다.

timeline을 import, artifact lookup/download, compiler, module load, workspace allocation, autotune, graph capture, first launch로 나눈다. CPU compiler process와 cache file publish가 보이면 JIT 후보, network와 checksum이면 artifact 후보, memory allocation API와 zero/init이면 workspace 후보, 반복 kernel 후보 실행이면 autotune, capture API면 graph 후보다.

first divergence는 replica마다 달랐다. cubin cache가 있는 node는 graph capture만, 빈 node는 download+JIT를 탔다. 하나의 “warm-up” metric으로 합치면 개선 owner를 찾지 못한다. artifact cache와 graph key, workspace capacity를 별도 lifetime으로 둔다.

수정은 build-time artifact, persistent cache, controlled warm-up, workspace preallocation 또는 graph bucket 사전 capture 중 원인에 맞게 고른다. 모든 shape를 미리 compile/capture하면 startup·disk·memory가 폭증할 수 있어 production key coverage로 제한한다.

종료 조건은 warm request 하나가 빠른 것이 아니다. cold deploy, restart, new replica와 cache loss에서 각 unique key의 compile/download/capture 횟수와 first-request budget을 검증한다. stale cache correctness, checksum, graph static buffer generation도 함께 통과한다.

### 47.6.8 일곱 사건의 숫자를 원인으로 오해하지 않는 판독법

커널 조사에서 가장 위험한 문장은 “GPU 사용률이 낮으므로 GPU를 충분히 쓰지 못한다”이다. 문장 앞뒤가 같은 말이고, 다음 행동을 정해 주지 못한다. 사용률을 산출한 도구의 표본 간격이 1초라면 3밀리초 커널과 7밀리초 공백이 여러 번 섞인 결과일 수 있다. 같은 30퍼센트도 하나의 긴 커널이 일부 SM만 쓴 경우, 짧은 커널 사이에 CPU 공백이 긴 경우, 다른 stream의 복사가 겹친 경우가 전혀 다르다. 먼저 시간 구간의 주인을 찾는다. 요청이 scheduler queue에서 기다렸는지, host가 launch를 준비했는지, device가 kernel을 실행했는지, collective를 기다렸는지 분리한 뒤에야 커널 내부 숫자를 읽을 자격이 생긴다.

Systems timeline에서 요청 시작부터 응답까지 수직선을 긋고 구간마다 owner를 붙여 보자. `scheduler`, `Python/C++ host`, `CUDA API`, `kernel`, `memcpy`, `collective`, `idle` 가운데 하나로 귀속하지 못하는 회색 구간은 아직 조사되지 않은 구간이다. CUDA API call이 길다고 device work가 길다는 뜻은 아니다. 동기 API가 앞선 비동기 오류나 work completion을 기다렸을 수 있다. 반대로 launch API가 짧아도 stream queue 뒤에서 kernel이 오래 기다릴 수 있다. 그래서 API 시작·끝, stream enqueue 순서, kernel 실제 시작·끝과 completion event를 서로 다른 열에 적는다.

그다음 total CTA 수를 본다. occupancy는 한 SM에 동시에 상주할 수 있거나 실제 상주한 warp의 비율을 나타내지만, grid가 GPU 전체를 채웠는지 알려 주지 않는다. 120개 SM인 장치에 CTA가 16개뿐이면 각 CTA 내부 occupancy가 100퍼센트여도 대부분의 SM은 할 일이 없다. decode의 작은 batch, 작은 head 수, tensor parallel로 더 잘게 나뉜 shard에서 이 장면이 흔하다. 이때 block의 thread를 늘려 occupancy 표를 예쁘게 만드는 것은 병렬 work item을 늘리지 않는다. 여러 request/head/page를 한 grid로 합치거나 split 정책을 바꾸는 것이 후보가 된다. 다만 합치기는 대기 시간을 늘릴 수 있으므로 scheduler의 batching window와 함께 평가한다.

CTA가 충분한데 느리다면 resident warp와 eligible warp를 구분한다. resident warp는 자리를 차지한다. eligible warp는 다음 cycle에 발행할 준비가 됐다. 많은 warp가 상주해도 모두 같은 long-latency load, barrier 또는 dependency chain을 기다리면 scheduler가 고를 warp가 없다. 이 경우 occupancy를 더 올리는 시도는 대기자만 늘릴 수 있다. source에서 load와 최초 consumer 사이 독립 instruction이 있는지, producer warp와 consumer warp가 어떤 barrier로 만나는지, stage 수가 실제 latency를 가리는지 확인한다. counter의 stall label은 이 source 질문을 고르는 표지판이지 답안지가 아니다.

`long scoreboard` 같은 표본이 높다고 곧바로 HBM이 병목이라고 쓰지 않는다. dependency가 걸린 operand가 global load인지, local spill인지, texture/cache 경로인지 source correlation과 instruction을 통해 확인해야 한다. 주소가 coalesced여도 working set이 커 cache miss가 날 수 있고, cache에 맞아도 같은 warp가 결과를 곧바로 소비하면 latency를 숨길 독립 work가 없을 수 있다. 반대로 memory stall 표본이 커도 kernel 전체 시간이 짧고 end-to-end 비중이 작다면 우선순위가 아니다. 개선 가능 시간의 상한은 해당 구간 시간이다.

barrier stall도 동일하다. barrier가 많다는 사실보다 어느 producer의 어떤 데이터가 ready임을 보장하는지가 먼저다. 불필요한 block-wide barrier를 warp-level synchronization으로 줄일 후보가 있을 수 있지만, 참여 mask와 memory ordering을 설명하지 못하면 최적화가 아니라 race 삽입이다. consumer가 shared tile을 읽기 전에 모든 writer가 완료돼야 하는지, double buffer stage 재사용 전에 이전 reader가 끝나야 하는지 invariant로 적는다. barrier 삭제 전후에는 timing뿐 아니라 racecheck, adversarial scheduling을 대신할 반복 fixture, boundary shape의 exactness를 함께 본다.

roofline은 결론이 아니라 분류 좌표다. operational intensity의 분자는 실제 수행한 산술량이고 분모는 선택한 memory level의 traffic이다. 모델의 논리 FLOP와 kernel이 실제 수행한 instruction은 tail padding, recomputation, fusion 때문에 다를 수 있다. HBM byte와 L2 byte도 다르다. 어느 분모를 사용했는지 밝히지 않은 “memory bound”는 반증할 수 없다. 낮은 batch decode가 roofline상 memory 쪽에 있어도 CTA 부족이나 launch overhead가 먼저 지배할 수 있다. 반대로 compute ceiling 가까이 있어도 tensor core를 효율적으로 쓴다는 뜻은 아니며, 불필요한 padded 연산으로 FLOP가 높아졌을 수 있다.

따라서 roofline을 읽을 때 세 장부를 나란히 둔다. 첫째, useful work 장부에는 유효 token·head·element와 알고리즘상 필요한 byte/FLOP를 적는다. 둘째, executed work 장부에는 padding, tail, 재계산, instruction과 requested transaction을 적는다. 셋째, delivered work 장부에는 latency, throughput, 실제 DRAM traffic을 적는다. useful는 그대로인데 executed가 늘었다면 policy·layout·tail 문제다. executed는 같은데 DRAM만 늘면 cache reuse·eviction 후보가 강해진다. 두 장부가 같은데 latency만 늘면 dependency, contention, clock, launch 간격과 다른 tenant를 본다.

캐시 hit rate도 단독 판정에서 빼야 한다. 95퍼센트 hit라도 요청 byte가 두 배면 miss byte는 늘 수 있다. hit가 낮아져도 전체 requested byte가 크게 줄면 DRAM은 감소할 수 있다. L1, L2, HBM의 분모와 transaction granularity가 다르므로 퍼센트를 서로 직접 비교하지 않는다. KV page traversal에서 같은 page를 여러 head가 재사용하는지, scale/metadata가 별도 stream으로 들어오는지, tail lane이 sector를 추가하는지 source address 식으로 돌아간다. 메트릭은 식을 검증하는 관측이다.

register는 숫자 하나가 아니라 lifetime의 결과다. thread당 register가 늘면 일정 임계에서 resident block이 줄 수 있지만, 임계를 넘지 않으면 occupancy는 그대로다. register 증가로 spill이 사라져 빨라질 수도 있다. compiler report에서 register, spill load/store, local memory, static shared를 같은 표에 놓고, 생성된 artifact의 target과 flags를 고정한다. source의 accumulator, stage buffer, unrolled loop가 live range를 어떻게 겹치게 하는지 찾는다. 강제로 register 상한을 낮추는 실험은 인과를 확인할 수 있지만 production fix로 채택하기 전에 spill traffic과 모든 specialization을 본다.

성능 A/B는 한 번의 평균으로 승인하지 않는다. 정상과 변경 빌드가 같은 model revision, input token, batch sequence, graph warm state, clock/power policy와 동시 tenant 조건을 가져야 한다. cold와 warm을 섞지 않는다. latency distribution은 prefill·decode, batch bucket, sequence length와 selected kernel key별로 나눈다. 작은 개선이 scheduler queue 변화에서 온 것인지 kernel에서 온 것인지 timeline 귀속을 유지한다. profiler를 붙이면 serialization과 replay가 바뀔 수 있으므로 profile run의 절대 시간을 production benchmark로 복사하지 않는다.

정확성 승인도 평균 오차 하나로 끝나지 않는다. reduction 순서가 바뀌는 최적화라면 허용 오차의 근거와 누적 지점을 쓴다. integer index, page table, token id와 generation은 tolerance 대상이 아니다. NaN/Inf, canary, out-of-range access, uninitialized read와 synchronization invariant를 검사한다. cancellation 직후 slot reuse, graph replay, empty/tail batch, 최대 context와 최소 context를 회귀 fixture로 둔다. 빠르지만 다른 request의 데이터를 섞는 kernel은 실패다.

## 47.7 Compute Sanitizer와 profiler가 말하는 것과 말하지 않는 것

`memcheck`는 invalid global/local/shared access와 일부 allocation misuse를 찾는 출발점이다. 사건 1처럼 logical output 뒤가 실제 allocation 밖이면 강한 증거가 된다. 하지만 allocator가 큰 slab을 잡고 여러 request를 subview로 나눴다면 다른 request 영역을 덮어도 물리 allocation 안일 수 있다. memcheck가 조용하다는 사실은 logical bounds가 맞다는 증명이 아니다. 그래서 allocation base/end와 tensor logical base/end를 ledger에 모두 적고 canary 또는 ownership generation을 추가한다.

`racecheck`는 shared-memory hazard를 찾는 데 유용하지만 모든 global-memory lifetime bug를 판정하는 만능 도구가 아니다. graph replay의 stale metadata는 두 access가 시간상 겹치지 않아 race가 아닐 수 있다. 잘못된 generation을 순서대로 읽은 논리 오류다. 반대로 보고된 hazard가 의도된 warp-synchronous protocol이라 주장하려면 참여 thread와 ordering의 공식 보장을 source로 설명해야 한다. “지금 GPU에서는 우연히 맞았다”는 반증이 아니다.

`initcheck`는 초기화되지 않은 device memory read 후보를 좁힌다. padding slot과 workspace를 일부만 쓰는 kernel에서 유용하다. 그러나 초기화된 오래된 값은 initialized이므로 사건 2를 놓칠 수 있다. zero fill은 진단에 도움이 되지만 production correctness를 zero의 우연한 중립성에 기대지 않는다. slot generation과 active mask가 더 강한 invariant다.

`synccheck`는 barrier API의 잘못된 사용과 divergent participation을 찾는 데 도움을 준다. 모든 thread가 도달해야 하는 barrier를 predicate 내부에 넣은 경우를 조사한다. 그러나 알고리즘상 필요한 barrier가 아예 빠져 결과가 scheduling에 의존하는 장면이 언제나 직접 보고된다고 가정하지 않는다. source의 producer/consumer ownership과 first read/write checkpoint가 여전히 필요하다.

Sanitizer 실행은 시간과 scheduling을 크게 바꾼다. 오류가 사라지거나 재현 빈도가 달라질 수 있다. 그렇다고 race 가설을 폐기하지 않는다. 최소 shape, 단일 stream, graph off 같은 축소 fixture로 검출 가능성을 높이고, production에서만 생기는 lifetime edge는 trace와 generation ledger로 보완한다. sanitizer 통과는 필요한 증거 중 하나이지 충분조건이 아니다.

### 47.7.1 Nsight Systems와 Nsight Compute의 역할을 뒤섞지 않는다

Systems가 답하기 좋은 질문은 “누가 언제 기다렸는가”이다. CPU thread가 scheduler lock을 기다렸는지, CUDA API가 언제 enqueue됐는지, 어느 stream에서 kernel과 copy가 겹쳤는지, NCCL 구간이 critical path인지 본다. kernel 이름이 같아도 request와 graph key를 annotation으로 이어야 serving 의미를 되찾는다. timeline만으로 특정 warp가 왜 발행되지 않았는지 단정하지 않는다.

이 구분은 NVIDIA가 고정 공개한 2025.3 문서의 제품 경계와도 맞는다. [Nsight Systems 2025.3 User Guide — system-wide timeline과 trace 범위](https://archive.docs.nvidia.com/nsight-systems/2025.3/UserGuide/index.html)는 process·thread·CUDA 활동의 시간 관계를 읽는 기준으로 사용한다. 이 문서가 kernel 내부의 source-level 원인을 자동 판정해 준다고 확대 해석하지 않는다.

Compute가 답하기 좋은 질문은 선택한 kernel instance가 어떤 launch resource와 instruction/memory 행동을 보였는가이다. grid/block, register/shared, active/eligible warp, memory workload와 source correlation을 묶는다. 모든 kernel을 무차별 profile하면 재생 횟수와 overhead가 커져 workload가 변한다. Systems로 critical kernel과 대표 shape를 고른 다음, deterministic fixture에서 instance를 좁힌다. graph 안 kernel은 replay/capture 조건이 profile에서 달라지지 않았는지 기록한다.

[Nsight Compute 2025.3 User Guide — kernel profiling, metric collection과 source correlation 범위](https://archive.docs.nvidia.com/nsight-compute/2025.3/NsightCompute/index.html)는 단일 kernel 관측의 기준으로 고정한다. 보고서의 stall sample과 section 요약은 가설을 좁히는 관측이며, serving request의 logical owner나 stale generation을 스스로 복원하지 않는다. 도구가 수집을 위해 kernel을 replay하거나 serialization할 수 있는 조건도 실험 기록에 남긴다.

두 도구 사이에는 request identity를 잇는 손잡이가 필요하다. request id 자체를 민감정보로 남기기 어렵다면 익명 trace id, scheduler step, batch bucket, backend/kernel key와 graph generation을 쓴다. Systems의 kernel instance를 ledger의 launch row와 연결하고, Compute report의 instance가 같은 symbol·shape·specialization인지 확인한다. 이름 문자열만 같고 template parameter가 다르면 다른 프로그램일 수 있다.

### 47.7.2 artifact와 source correlation이 필요한 순간

source가 같다고 executable이 같지 않다. toolkit, compiler flags, target architecture, template specialization과 link artifact가 달라질 수 있다. `ptxas` resource report는 register·spill·shared 같은 정적 자원을 알려 주지만 runtime grid, cache state와 dependency는 알려 주지 않는다. `cuobjdump`와 `nvdisasm`은 어떤 architecture image와 instruction이 들어 있는지 확인하는 데 쓰지만 고수준 tensor 의미를 자동 복원하지 않는다.

`no kernel image`나 invalid device function 후보에서는 artifact에 대상 SM image 또는 호환 PTX가 있는지, driver가 load한 module이 예상 파일인지 본다. launch configuration failure라면 symbol 존재와 별개로 block threads, static+dynamic shared, opt-in limit을 본다. wrong value라면 disassembly부터 읽기 전에 first arithmetic/store checkpoint로 범위를 줄인다. 도구가 제공하는 가장 낮은 층으로 무조건 내려가는 것은 깊은 조사가 아니다. 질문과 가까운 경계부터 증거 사슬을 만든 것이 깊은 조사다.

source correlation line은 최적화된 code에서 한 instruction이 여러 source expression에 대응하거나 반대가 될 수 있다. inline, unroll과 scheduling 때문이다. 특정 line에 stall 표본이 많다는 이유만으로 그 줄을 원인이라고 부르지 않는다. 그 instruction의 operand producer, address expression과 consumer까지 작은 slice로 읽는다. source 수정 뒤 line mapping이 이동해도 semantic checkpoint 이름은 유지한다.

### 47.7.3 현장에서 사용하는 증거 장부를 읽는 법

실제 조사는 하나의 거대한 profiler report보다 작은 장부 여러 개에서 빨라진다. 첫 장부는 request identity다. `trace`, scheduler step, batch bucket, sequence lengths, graph key, backend key를 한 행에 둔다. 이 가운데 값이 없으면 “같은 요청을 비교했다”는 말을 검증할 수 없다. text는 같아도 chat template, tokenizer revision 또는 prefix cache hit가 달라 token sequence가 달라질 수 있다. 그러므로 개인정보를 제거한 token id digest와 token count를 함께 기록한다. digest가 같고 count가 다르면 기록 자체가 잘못된 것이다.

둘째 장부는 dispatch다. logical shape와 physical shape를 분리한다. logical shape는 실제 유효 batch, query length, KV length, head와 dimension이다. physical shape에는 padding, page size, tile, split 수와 capture bucket을 쓴다. policy predicate를 사람이 읽을 수 있는 문장으로 복사한다. 예컨대 “Q length가 1이고 head dimension이 128이며 custom mask가 없어서 decode specialization 선택”처럼 쓴다. 단지 enum 값 `2`만 남기면 다음 revision에서 의미가 바뀌었을 때 비교가 깨진다. source revision과 predicate 위치가 있어야 한다.

셋째 장부는 launch다. kernel symbol 또는 template key, grid, block, static shared, dynamic shared, stream과 graph node를 기록한다. launch 반환값과 completion 관측을 별도 칸으로 둔다. launch API가 성공한 뒤 execution 중 invalid access가 생길 수 있다. 반대로 나중 API에서 이전 비동기 오류가 보고될 수 있다. 진단 중 launch 직후 synchronization을 삽입했다면 `diagnostic-only`라고 표시하고 production 결과와 섞지 않는다. synchronization 때문에 race나 overlap이 사라질 수 있기 때문이다.

넷째 장부는 memory ownership이다. 단순 pointer hex 값만으로 부족하다. allocation id, allocation range, logical tensor range, dtype, stride/layout, request owner, generation, producer와 마지막 consumer를 적는다. allocator가 주소를 재사용하면 동일 pointer가 다른 tensor를 의미한다. graph는 주소 안정성을 요구해 buffer를 오래 유지할 수 있지만 그 안의 contents generation은 매 replay 바뀐다. “주소가 같으니 같은 입력”이라는 문장은 이 장부에서 금지된다.

다섯째 장부는 dependency다. `stream 7`처럼 이름만 적지 않고 `metadata_copy(B,g=42) completion event → attention_replay(B,g=42) wait`처럼 producer, payload generation, completion과 consumer를 모두 쓴다. default stream semantics에 기대는 코드와 explicit event를 쓰는 코드를 구분한다. host enqueue 순서는 다른 stream의 device execution 순서를 자동으로 보장하지 않는다. allocator free/reuse도 dependency 일부다. 마지막 consumer가 끝나기 전에 다른 request가 같은 slot generation을 획득하면 주소는 valid해도 의미는 틀린다.

여섯째 장부는 arithmetic checkpoint다. 모든 tensor를 dump할 필요는 없다. 첫 입력 checksum, index/page id, partial max, normalization sum, output 일부와 NaN/Inf count처럼 알고리즘 단계를 가르는 값을 고른다. checksum collision 가능성을 고려해 작은 fixture에서는 exact dump를 보존하고 큰 fixture에서는 여러 digest와 sentinel을 쓴다. floating 값은 비교 dtype, absolute/relative tolerance와 reduction order를 기록한다. index와 generation은 exact 비교한다.

일곱째 장부는 결과와 종료 조건이다. latency, throughput, useful/requested/DRAM byte, sanitizer 결과, exactness/tolerance, cancellation/reuse를 한 행에 둔다. “성능 개선” 행과 “정확성 통과” 행을 분리하면 한쪽만 보고 merge하는 실수를 줄인다. 변경으로 backend 선택이 달라졌다면 같은 kernel A/B가 아니므로 dispatch 장부부터 다시 읽는다. 숫자가 좋아도 다른 fallback으로 이동했을 수 있다.

이 장부들은 로그를 무한히 늘리라는 요구가 아니다. production에는 request identity와 이미 노출된 coarse event만 남기고, 재현 환경에서 선택적 계측을 켠다. pointer와 tensor 값은 보안·프라이버시 경계를 지킨다. 원본 값 대신 allocation-relative offset, generation과 digest로도 많은 lifetime bug를 찾을 수 있다. 계측 자체가 timing을 바꾸는지 logging off/on fixture로 확인한다.

### 47.7.4 일곱 사건을 더 깊이 파는 반증 질문

M=65 오답에서 첫 질문은 “왜 65인가”이다. tile 64의 다음 값이므로 tail CTA를 의심하지만 이것만으로 확정하지 않는다. M=63, 64, 65, 66, 127, 128, 129를 배열하고 failure pattern이 tile 경계인지 vector width인지 본다. output base를 한 element offset해 alignment를 바꾸고 pattern이 이동하는지 본다. K와 N은 고정한다. M과 K를 동시에 바꾸면 load tail과 store tail을 분리할 수 없다.

canary는 output 뒤에만 두지 않는다. output 앞, 각 row padding과 이웃 subview 경계에 서로 다른 pattern을 둔다. 첫 corrupted address가 어느 lane의 vector store와 대응하는지 계산한다. store address가 allocation 안이면 memcheck가 조용할 수 있으므로 logical canary가 필요하다. 반대로 canary는 정상인데 output last row만 틀리면 load predicate, accumulator initialization과 reduction tail로 이동한다. 이처럼 negative invariant가 조사 방향을 가른다.

graph stale token 사건에서는 요청 A와 B가 우연히 같은 token을 가지면 오류가 숨는다. 각 slot과 generation을 구별하는 synthetic token/page pattern을 쓴다. A의 slot마다 `A0`, `A1`, B에는 `B0`, `B1`에 해당하는 안전한 식별 pattern을 배치한다. 어느 값이 섞였는지 보면 stale generation의 출처가 드러난다. 모든 unused slot을 zero로만 채우면 zero가 attention mask에서 중립인지 확인해야 한다. 중립이 아니면 진단이 계산을 바꾼다.

graph key에는 batch bucket만 있는지, sequence bucket, backend, dtype, adapter와 memory layout도 포함되는지 읽는다. key에 없는 상태가 captured node parameter 또는 static buffer contents로 안전하게 갱신되는지 확인한다. graph update 성공은 논리 metadata가 최신이라는 증명이 아니다. CUDA가 graph topology를 받아들였다는 뜻과 serving request generation이 맞다는 뜻을 분리한다. eager 정상은 graph lifetime 후보를 올리지만 eager와 graph가 같은 specialization인지 dispatch 장부로 확인한다.

shared-memory launch 실패에서는 오류가 어느 GPU에서만 나는 이유를 device limit 표로 설명한다. architecture 이름만으로 판단하지 않고 실제 device attribute, compile target과 selected symbol을 고정한다. static shared는 compiler artifact에서, dynamic shared는 launcher 계산에서 온다. 둘을 더하고 alignment 또는 runtime reservation 조건을 확인한다. kernel attribute 설정이 호출됐는지, module 재로딩이나 다른 symbol에 설정한 것은 아닌지 source lifecycle을 따라간다.

tile을 줄이는 수정은 가장 쉬운 복구지만 가장 좋은 설계라는 보장은 없다. stage 수를 줄이면 shared는 감소하나 latency hiding이 나빠질 수 있다. block thread를 바꾸면 register·barrier와 mapping이 달라진다. fallback은 서비스 복구에 유용하지만 특정 device에서 조용히 느린 path로 가는 운영 위험이 있다. 선택 reason과 fallback counter를 노출하고 해당 device의 latency budget을 회귀 조건으로 둔다.

높은 occupancy·낮은 throughput 사건에서는 작은 grid와 issue starvation을 한 문장에 합치지 않는다. 먼저 전체 CTA/SM 배치를 보고 충분한 wave가 있는지 판정한다. 부족하면 active SM이 적은 문제다. 충분하면 active SM 내부에서 eligible warp, instruction mix와 dependency를 본다. 두 현상은 동시에 있을 수 있지만 수정 지렛대가 다르다. grid fusion은 전자를, pipeline scheduling은 후자를 겨냥한다.

batch를 늘려 throughput이 좋아졌다고 kernel fix가 된 것은 아니다. scheduler가 더 오래 기다려 큰 batch를 만들면 per-request latency가 악화될 수 있다. arrival distribution에서 batching wait, prefill interference와 decode deadline을 함께 본다. continuous batching 환경에서는 한 static benchmark의 batch 숫자가 운영 batch lifetime을 대표하지 않는다. step별 active sequence와 token budget을 trace에 남긴다.

register 사건에서는 두 artifact를 정말 같은 조건으로 만들었는지 먼저 반증한다. source commit만 같아도 generated header, architecture flag, fast-math, LTO와 dependency revision이 다를 수 있다. binary hash와 symbol, compiler version을 기록한다. register 수가 달라진 source region을 live variable 관점에서 읽고, disassembly instruction count와 spill을 비교한다. occupancy calculator의 theoretical 값과 실제 launch의 active block을 구분한다.

register 상한 실험에서 시간이 좋아지면 곧바로 상한을 채택하지 않는다. spill이 늘었는데 cache에 우연히 맞은 작은 fixture일 수 있다. 긴 KV, 다른 batch와 concurrent kernel에서는 local traffic이 L2를 밀어낼 수 있다. 반대로 register가 늘고 빨라졌다면 occupancy 감소보다 spill 제거가 컸을 수 있다. 한 resource를 선악으로 분류하지 않고 request shape별 trade-off 표를 만든다.

DRAM byte 사건에서 counter 분모를 확인한다. 어떤 값은 sector, 어떤 값은 byte 추정, 어떤 값은 read/write 합계다. 프로파일러 version과 metric 정의 없이 이전 보고서 숫자를 직접 비교하지 않는다. logical KV payload를 계산하고 metadata, scale, output, workspace, spill을 별도 항목으로 둔다. measured total과 차이가 크면 누락된 traffic 또는 counter scope를 조사한다.

cache 실험은 cold와 warm을 명확히 정의한다. 다른 kernel과 request가 공유 L2를 교란하는 serving 환경에서 microbenchmark warm cache가 재현되지 않을 수 있다. cache flush라는 조작 자체가 운영에는 없는 상태를 만들 수 있다. 그래서 isolated cold/warm은 원인 분리에 쓰고, 최종 승인은 대표 concurrency trace에서 한다. layout 변경은 requested transaction과 reuse distance를 함께 바꾸므로 둘을 각각 예측한다.

첫 요청 지연 사건에서는 process lifetime과 node lifetime을 구분한다. container restart 뒤에도 host artifact cache가 남을 수 있고, 새 node에는 없을 수 있다. module/JIT cache, model weight cache, allocator pool, graph executable과 autotune result는 수명이 다르다. 각 cache의 key, 저장 위치, invalidation과 checksum을 적는다. “warmup 완료” boolean 하나는 어느 cache가 준비됐는지 말하지 않는다.

warm-up request가 실제 production key를 대표하는지도 본다. prompt 길이, decode batch, dtype, quantization, adapter, head dimension과 graph bucket이 달라지면 첫 실요청에서 새 compile/capture가 발생한다. 모든 조합을 선행하면 시작 시간과 memory가 폭발한다. traffic 상위 key와 latency 민감 key를 선정하고, 미포함 key는 bounded fallback을 가져야 한다. cold path도 correctness fixture를 통과해야 한다.

### 47.7.5 vLLM·SGLang·FlashInfer·llama.cpp를 가로지르는 추적 질문

프레임워크 이름으로 원인을 나누면 경계 버그를 놓친다. vLLM scheduler가 만든 batch가 backend selector를 지나 FlashInfer plan/run에 들어가고, CUDA Graph가 static buffer를 재사용한다면 한 요청의 의미는 네 ownership 층을 건넌다. 각 층에서 입력 계약, 출력 계약과 lifetime을 한 줄로 잇는다. scheduler의 `num_tokens`, backend의 `qo_len`, kernel의 grid dimension이 같은 논리량을 가리키는지 단위와 padding을 확인한다.

vLLM source를 읽을 때 graph wrapper의 persistent buffer와 replay lifecycle만 보지 않는다. graph key를 만든 upstream state와 fallback reason, request cancellation이 slot allocator에 미치는 영향까지 거슬러 올라간다. source에 실제로 노출된 log와 metric만 운영 관측으로 기록한다. profiler에서 본 grid/block을 프레임워크가 원래 로그한다고 오해하지 않는다. 관측 공백이 중요하면 최소 계측 후보와 비용을 따로 제안한다.

SGLang의 kernel API logging은 호출 인자 경계를 찾는 손잡이다. logging이 tensor contents, device completion 또는 barrier correctness를 증명하지는 않는다. scheduler/forward stream에서 metadata producer와 consumer를 찾아 event를 잇고, custom kernel의 WAR barrier가 어느 buffer 재사용을 보호하는지 읽는다. barrier 이름만으로 hazard 방향을 가정하지 않고 이전 reader, 다음 writer와 generation을 명시한다.

FlashInfer에서는 plan 단계가 계산한 workspace, split와 launch policy가 run 단계의 실제 shape와 일치하는지 본다. workspace address가 persistent해도 capacity와 generation이 요청별로 올바른지 확인한다. native dispatch가 선택한 specialization의 tile, stage와 shared requirement를 launcher 계산과 잇는다. wrapper의 shape validation이 논리 오류를 모두 막는다고 가정하지 않는다. tail predicate와 merge가 어떤 partial range를 소유하는지 kernel source까지 내려간다.

llama.cpp에서는 graph update 성공·실패와 reinstantiate가 성능·정확성에서 다른 의미를 가진다. update failure 뒤 안전한 재생성은 correctness 경로일 수 있고 첫 요청 지연을 만든다. update success라도 static tensor contents와 generation 갱신은 별도 계약이다. graph launch가 enqueue된 뒤 completion을 어느 event/sync에서 관측하는지 따라가며, 뒤 API의 error를 현재 line의 오류로 단정하지 않는다.

네 stack을 비교할 때 구현 이름보다 공통 질문을 유지한다. 누가 batch를 소유하는가, 누가 backend를 선택하는가, 누가 workspace를 할당하고 언제 재사용하는가, graph key는 무엇을 포함하는가, metadata producer와 kernel consumer 사이 edge는 무엇인가, error와 completion은 어디에서 관측되는가, fallback은 어떤 reason으로 노출되는가. 이 질문표를 유지하면 revision이 바뀌어 함수 이름이 이동해도 조사를 다시 시작하지 않는다.

소스 링크는 증거의 시작이지 장식이 아니다. 고정 commit과 line range가 가리키는 code가 어떤 주장을 뒷받침하는지 문장으로 붙인다. line 전체를 길게 복사하기보다 branch predicate, buffer lifetime 또는 launch call의 핵심 부분만 인용한다. 다음 revision에서 diff를 볼 때 predicate가 바뀌었는지, ownership이 이동했는지 검토한다. 현재 source와 과거 장애 source를 섞지 않는다.

## 47.8 한 번의 종합 리뷰: 독자가 실제 장애를 끝낼 수 있는가

이 장의 마지막 리뷰는 용어 암기 시험이 아니다. 독자가 새 장애를 받았을 때 무엇을 고정하고, 어디서 처음 갈라졌는지 찾고, 경쟁 가설을 버리고, 수정 뒤 무엇까지 통과시킬 수 있는지를 한 사건으로 검증한다. 상황은 이렇다. vLLM 또는 SGLang 기반 서비스에서 새 attention backend를 켠 뒤 긴 prompt의 첫 decode token이 간헐적으로 달라졌다. 오류율은 낮고 GPU utilization dashboard는 이전보다 높다. graph를 끄면 재현이 줄고 batch 8에서만 보인다. 운영자는 backend를 되돌릴지, sanitizer부터 돌릴지, scheduler를 의심할지 결정해야 한다.

첫 행동은 utilization graph를 확대하는 것이 아니다. 실패 request를 보존 가능한 fixture로 바꾼다. model/source revision, tokenizer와 token ids, dtype, batch의 각 sequence length, Q/KV head, page table, selected backend/kernel key, toolkit/driver/GPU, graph key와 generation, seed를 기록한다. 개인정보가 있는 text 대신 token id와 shape를 안전하게 보존한다. 정상 request와 실패 request 사이에서 한 변수만 바꿀 수 있도록 batch 7·8·9, graph on/off, backend old/new를 축으로 만든다.

다음으로 요청 장부를 채운다. scheduler step에서 logical batch가 결정되고, backend policy가 specialization을 고르고, workspace와 static graph input의 주소·range·generation이 정해지고, metadata write가 어느 stream에 enqueue되며, graph replay가 어떤 edge 뒤에 오고, attention의 첫 partial max/sum과 output store가 어디인지 적는다. 모르는 칸은 추측으로 채우지 않는다. source log가 노출하지 않는 필드는 외부 annotation이나 최소 계측 후보로 남긴다.

정상과 실패 logits만 비교하면 divergence가 너무 늦다. embedding input token, Q/K/V projection checksum, page table·active length generation, attention tile의 partial max와 normalization sum, attention output, output projection logits 순으로 checkpoint를 세운다. 모든 값을 production 로그로 영구 노출하라는 뜻은 아니다. 작은 재현 fixture에서 안전하고 결정적인 digest나 sentinel을 사용한다. 첫 divergence가 active length라면 kernel arithmetic을 profile하기 전에 metadata lifetime을 고친다. attention partial에서 처음 갈라지면 선택된 tile과 barrier/address를 조사한다.

경쟁 가설 표에는 최소 네 개를 둔다. 첫째, batch 8에서 다른 specialization의 tail predicate가 틀렸다. 둘째, graph static metadata가 이전 generation을 읽었다. 셋째, shared producer/consumer barrier가 빠졌다. 넷째, floating reduction 순서가 tolerance 안에서 달라졌다. 각 가설은 자신이 맞다면 반드시 달라질 관측을 하나 가진다. canary와 first bad address, metadata generation, sanitizer hazard, partial checkpoint의 오차 크기가 그것이다. 아무 관측도 예측하지 않는 “CUDA가 불안정하다”는 가설은 표에서 제거한다.

graph off에서 재현이 줄었다는 사실은 graph가 범인이라는 결론이 아니다. graph off는 launch timing, address lifetime, backend policy와 batch formation을 동시에 바꿀 수 있다. 같은 address와 shape를 유지하며 explicit dependency만 추가하는 진단, static metadata를 매 replay 전 neutral fill하는 진단, graph key를 강제로 분리하는 진단처럼 한 축씩 움직인다. device-wide synchronization으로 오류가 사라지면 ordering 가설이 강해지지만 그것을 production fix로 두면 concurrency를 잃고 missing edge를 숨긴다.

Sanitizer 선택도 질문을 따른다. output allocation 밖 store가 의심되면 memcheck, shared barrier 참여가 의심되면 racecheck와 synccheck, padding initialization이면 initcheck를 쓴다. stale-but-valid generation은 별도 ledger로 찾는다. Systems는 scheduler부터 graph replay까지 edge와 공백을 귀속하고, Compute는 first divergent kernel의 대표 instance에서 launch resource와 memory/warp 행동을 본다. correctness 원인이 확인되기 전 성능 counter를 고치는 데 시간을 쓰지 않는다.

수정이 metadata write→replay event 추가였다고 하자. 테스트 한 번이 맞았다고 끝내지 않는다. batch bucket 경계, empty slot, cancellation 뒤 reuse, 두 stream의 지연을 뒤집는 fixture, graph/eager, backend old/new를 반복한다. token/page generation이 consumer 완료 전 재사용되지 않는지 확인한다. exact output 또는 사전에 정한 tolerance, canary, sanitizer와 first-divergence ledger가 모두 정상이어야 한다.

그 뒤 성능 승인을 별도로 한다. event가 critical path에 필요한 edge인지, 더 넓은 device sync를 우연히 넣지 않았는지 Systems에서 본다. kernel CTA·eligible warp·memory traffic은 동일할 가능성이 크지만 batch timing이 달라질 수 있다. latency p50만 아니라 p95/p99, throughput과 cancellation workload를 확인한다. 정확성 수정으로 이전의 잘못된 overlap이 사라져 조금 느려졌다면, 그 차이를 regression이라고 되돌리지 않는다. 올바른 baseline과 비교한다.

반대로 원인이 tail store였다면 masked store 수정 뒤 full-tile fast path가 유지되는지 본다. M 또는 sequence가 tile 경계보다 하나 작고 같고 큰 fixture, vector width와 subview alignment를 조합한다. memcheck가 조용해도 slab 내부 이웃 request canary를 본다. output tolerance뿐 아니라 잘못 쓰지 않아야 할 byte가 그대로라는 negative invariant를 승인 조건에 둔다.

shared tile launch 실패라면 error 문자열만 보고 tile을 줄이지 않는다. selected specialization, block threads, static+dynamic shared와 device limit, opt-in attribute를 표로 만든다. 같은 artifact가 지원 GPU마다 어느 fallback을 고르는지 확인한다. tile 축소가 launch를 살려도 global traffic과 barrier 수가 늘 수 있으므로 correctness 승인 뒤 성능 장부를 다시 연다. 지원하지 않는 조합을 명시적으로 거절할지 안전한 kernel로 fallback할지도 API 계약이다.

독자가 리뷰를 통과했는지 확인하는 질문은 단순하다. 첫째, “느리다”를 host gap, insufficient work, dependency, memory traffic과 resource 가운데 어디로 좁혔는가. 둘째, 오류가 보고된 API line과 실제 offender interval을 구분했는가. 셋째, allocation bounds와 logical tensor ownership을 구분했는가. 넷째, address와 함께 generation을 기록했는가. 다섯째, metric 이름이 아니라 source expression과 dependency edge를 지목했는가. 여섯째, 한 가설을 지지한 증거뿐 아니라 경쟁 가설을 기각한 증거가 있는가. 일곱째, 수정 뒤 성능·정확성·memory safety·cancellation/reuse를 함께 승인했는가.

좋은 장애 보고서는 “occupancy가 낮아서 block을 키웠다”로 끝나지 않는다. “batch 4 decode에서 grid가 24 CTA라 장치 전체 work가 부족했고, block 확대는 CTA 수를 늘리지 않았다. request/head fusion으로 grid를 늘렸으며 batching wait가 SLO 안인지 검증했다. kernel time과 end-to-end latency가 개선됐고 tail·cancellation fixture가 통과했다”처럼 관측과 선택을 연결한다. 다른 사람이 같은 ledger로 결론을 재현할 수 있어야 한다.

좋은 오답 보고서도 “race 같아서 sync를 넣었다”로 끝나지 않는다. “request B의 active-length generation이 replay generation보다 하나 뒤처진 것이 첫 divergence였고, producer stream event가 consumer replay에 연결되지 않았다. 정확한 event edge를 추가했으며 device-wide sync 없이 graph/eager와 slot reuse fixture가 통과했다”라고 쓴다. 이 문장은 원인, 수정 범위와 성능 위험을 동시에 드러낸다.

리뷰에는 운영 중 수집할 수 있는 최소 관측과 재현실에서만 켤 상세 관측을 구분하는 항목도 있다. 운영 metric에는 queue wait, batch bucket, backend·fallback reason, graph hit/miss, compile/capture 횟수, kernel·collective 구간과 오류 count가 적합하다. tensor dump, pointer, sanitizer report와 instruction counter는 비용과 민감도가 높아 제한된 fixture로 보낸다. 운영 metric이 직접 root cause를 말한다고 기대하지 않고, 어느 ledger를 열어야 하는지 알려 주는 경보로 사용한다.

예를 들어 p99가 오르고 graph miss가 함께 늘었다면 graph가 느리다는 결론을 내리지 않는다. miss request의 shape/key, fallback backend와 capture/instantiate 시간을 분리한다. graph miss가 새로운 긴 context에서 생겼고 실제 시간은 attention kernel에 있다면 miss는 workload 변화의 동반 신호일 뿐이다. 반대로 capture가 critical path를 차지하고 같은 key가 반복 capture된다면 cache lifetime이나 invalidation을 조사한다. 상관관계가 dependency edge가 되려면 source lifecycle과 시간 순서가 필요하다.

오류 metric도 종류를 합치지 않는다. launch configuration, no-image/module load, asynchronous execution error, sanitizer finding, numeric mismatch와 stale-generation assertion은 owner가 다르다. 하나의 `cuda_error_total`만 있으면 복구 행동을 선택할 수 없다. 다만 고카디널리티 kernel symbol과 request shape를 그대로 metric label로 넣으면 모니터링 시스템이 무너진다. bounded backend/error category를 metric으로 두고 exact symbol, revision, shape와 trace id는 sample log 또는 artifact에 연결한다.

재현이 되지 않는 간헐 오류에는 성공 trace도 필요하다. 실패 하나만 보면 모든 특성이 특별해 보인다. 같은 bucket·backend·graph key에서 직전과 직후 성공 request를 골라 first-divergence ledger를 비교한다. allocator generation, stream edge와 cancellation이 실패에만 다른지 본다. 실패율을 낮추는 변화와 원인을 제거하는 변화를 구분한다. concurrency를 1로 줄여 사라졌다는 사실은 lifetime/order 후보를 올리지만 production을 concurrency 1로 두는 것은 대개 진단 우회다.

재현 fixture를 축소할 때 의미를 지운 채 tensor 크기만 줄이지 않는다. batch slot 재사용이 핵심이면 최소 두 request와 cancellation을 남겨야 한다. tail이 핵심이면 tile 경계의 세 shape를 남긴다. graph lifetime이면 capture와 두 번 이상의 replay를 남긴다. barrier race이면 producer와 consumer의 stage 재사용을 남긴다. 잘 축소된 fixture는 runtime이 짧아서 좋은 것이 아니라 예상 invariant와 divergence를 그대로 보존해서 좋다.

반대로 거대한 production trace를 그대로 반복하는 것도 좋지 않다. 여러 kernel과 stream이 얽히면 어느 변화가 divergence를 옮겼는지 알 수 없다. 먼저 output 또는 metadata checkpoint로 offender interval을 좁히고 그 interval을 보존하는 최소 prefix를 만든다. prefix를 잘라낼 때 allocator와 cache warm state가 바뀌면 fixture header에 명시한다. 원본 trace와 축소 fixture의 first divergence가 같은 semantic checkpoint인지 확인한다.

수정 리뷰에서는 code diff보다 invariant diff를 먼저 읽는다. tail predicate를 추가했다면 “모든 writer address가 logical output range 안이다”가 새로 보장된다. event를 추가했다면 “generation g metadata completion이 generation g replay보다 앞선다”가 보장된다. tile fallback을 추가했다면 “required static+dynamic shared가 selected device limit 안이다”가 보장된다. invariant를 한 문장으로 쓰지 못하면 patch가 어느 사건을 막는지 불명확하다.

그 invariant가 다른 path에도 적용되는지 찾는다. decode tail을 고쳤어도 prefill, speculative decode, adapter path와 quantized specialization에 같은 epilogue가 복제돼 있을 수 있다. 하나의 wrapper가 여러 native kernel을 dispatch한다면 predicate의 source of truth가 어디인지 본다. 회귀 fixture를 발견한 함수 한 곳에만 묶지 않고 semantic key와 boundary shape에 묶는다. revision이 바뀌어 함수가 이동해도 테스트 의도는 남아야 한다.

성능 수정은 비용 이동을 기록한다. fusion으로 launch gap을 줄였지만 register와 compile 시간이 늘 수 있다. split을 늘려 CTA 병렬성을 얻었지만 workspace와 merge traffic이 늘 수 있다. graph capture로 host overhead를 줄였지만 static buffer와 key cardinality가 늘 수 있다. preallocation으로 첫 요청은 빨라졌지만 idle memory가 늘 수 있다. 변경 전후 장부에서 얻은 구간과 지불한 구간을 동시에 표시한다. 한 fixture의 win을 전체 stack의 win으로 확대하지 않는다.

운영 배포에는 중단 조건도 필요하다. numeric mismatch, sanitizer finding, generation assertion은 즉시 실패다. latency는 사전에 정한 shape별 budget과 통계 기준을 쓴다. fallback 비율, graph recapture, compile 횟수와 memory watermark가 예상 범위를 벗어나면 rollback 또는 조사로 전환한다. 평균 throughput이 좋아도 p99와 cancellation correctness가 나빠지면 승인하지 않는다. 모호한 “유의미하게 빨라짐” 대신 측정 설계와 허용 범위를 변경 전에 쓴다.

사후 문서에는 반증된 가설도 남긴다. reduction 오차를 의심했지만 canary가 allocation 밖에서 먼저 바뀌어 기각했다거나, occupancy 감소를 의심했지만 resident CTA 임계를 넘지 않았고 spill traffic 변화가 first divergence였다고 기록한다. 이 기록은 실패한 시도의 일지가 아니라 다음 조사자가 같은 지름길에 빠지지 않게 하는 지식이다. 다만 도구 counter를 맥락 없이 붙이지 않고 어떤 fixture와 source predicate에서 관측했는지 연결한다.

최종 handoff는 여섯 문장으로 요약할 수 있어야 한다. 어떤 request/shape에서 무엇이 틀리거나 느렸는가. 정상과 실패의 첫 divergence는 어디인가. 그 지점의 예상 invariant는 무엇인가. 어떤 관측이 선택한 가설을 지지하고 경쟁 가설을 기각했는가. patch가 invariant를 어떻게 복원했는가. 어떤 correctness·safety·performance·reuse fixture가 재발을 막는가. 여섯 문장 뒤에 source link, trace와 report를 붙이면 다른 사람이 검토할 수 있다.

handoff를 받은 검토자는 동일한 결론에 동의하는지만 보지 않는다. ledger의 빈칸이 결론을 바꿀 수 있는지 공격적으로 묻는다. graph key에는 adapter가 빠졌는가, allocation range만 있고 logical subview가 없는가, 성공 fixture는 실패와 같은 specialization인가, synchronization을 넣은 profile이 production ordering을 바꿨는가를 확인한다. 누락이 결론을 바꿀 수 있으면 승인을 보류하고 필요한 최소 관측을 지정한다. 모든 counter를 다시 수집하라는 모호한 요구는 하지 않는다.

또한 재발 테스트가 구현 세부에 과도하게 붙어 있지 않은지 본다. 특정 함수 호출 횟수만 검사하면 refactor 후 의미가 사라질 수 있다. graph 구현이 바뀌어도 “generation g producer completion이 generation g consumer보다 앞선다”는 계약은 남는다. kernel 이름이 바뀌어도 tile 경계 밖을 쓰지 않는 계약은 남는다. semantic invariant를 검증하고, 함수·metric 검사는 그 invariant의 현재 관측 adapter로 둔다.

마지막으로 장애 해결 속도와 설명의 완전성을 적으로 만들지 않는다. 우선 안전한 backend fallback으로 영향을 줄이는 운영 조치와 root cause 수정은 별도 change로 관리할 수 있다. fallback을 원인 해결이라고 닫지 않고 선택 reason과 성능 비용을 기록한다. root fix가 준비되면 동일 fixture로 fallback 전후를 비교하고, 관측이 사라져서가 아니라 invariant가 복원됐기 때문에 종료한다.

마지막 원칙은 단순하다. 커널을 깊게 본다는 것은 counter를 많이 수집하거나 SASS를 오래 바라보는 일이 아니다. 요청의 의미가 scheduler, address lifetime, launch configuration, instruction과 store를 지나 응답으로 돌아올 때까지 끊기지 않는 설명을 만드는 일이다. first divergence는 그 긴 사슬에서 조사 범위를 가장 작게 만드는 좌표다. 좌표 앞의 invariant가 맞고 좌표 뒤의 결과가 틀리다면, 비로소 수정할 source와 재발 방지 fixture가 같은 문장 안에 들어온다.

## 47.9 네 증상을 첫 30분에 분류한다

wrong answer는 reference와 semantic value가 갈리는 사건이다. race는 timing/interleaving에 따라 ownership/order가 깨지는 원인 후보이며 wrong answer, hang 또는 crash로 나타날 수 있다. bandwidth는 특정 memory-level traffic/throughput이 critical path를 제한하는 성능 가설이다. launch overhead는 host submission·dispatcher·graph/JIT와 짧은 kernels 사이 gap이 지배하는 가설이다.

네 단어는 같은 축이 아니다. wrong answer는 결과 분류, race는 correctness 원인, bandwidth와 launch overhead는 성능 원인이다. “wrong answer 아니므로 race가 아니다” 또는 “GPU utilization 낮으므로 launch overhead”처럼 배타적으로 쓰지 않는다. 하나의 request에 race와 launch gaps가 함께 있을 수 있다.

첫 5분에는 request generation, model/backend/specialization, tensor shapes/strides, graph/eager, streams와 binary/source revision을 고정한다. failing trace 앞뒤의 same-key success를 고른다. aggregate 평균에서 바로 profiler를 켜지 않는다.

다음 5분에는 CPU/reference fixture와 output checkpoints를 비교한다. input/token IDs, pre-kernel tensors, kernel output, layer output와 final logits/token 중 first mismatch를 찾는다. deterministic seed와 same semantic options를 사용한다. final text만 비교하면 sampling이 first divergence를 숨긴다.

세 번째 5분에는 Nsight Systems timeline으로 host→CUDA API→kernel/memcpy/event/collective의 귀속을 본다. long host gaps, many tiny launches, serialization, first JIT/capture와 overlap을 찾는다. Systems는 한 kernel의 exact memory stall 원인을 자동 확정하지 않는다.

네 번째 5분에는 failing kernel의 launch coordinates와 source span을 고정한다. grid/block/dynamic shared, template/tile/stage, pointer/shape/stride, active length와 generation을 기록한다. wrong kernel key를 profile하면 counter가 정확해도 사건과 무관하다.

다섯 번째 5분에는 hypothesis별 도구를 선택한다. bounds/uninitialized/race는 Compute Sanitizer의 relevant tools와 minimal fixture, bandwidth/resource는 Nsight Compute/compile artifact, launch overhead는 Systems/CPU call trace다. 모든 도구를 동시에 켜 timing을 크게 바꾸지 않는다.

마지막 5분에는 perturbation을 기록한다. profiler/sanitizer on에서 failure가 사라지거나 이동하는지, kernel serialization/replay/cache warmness가 바뀌는지 본다. instrumented success는 production race 반증이 아니다. same invariant를 barrier-controlled fixture로 재현한다.

## 47.10 CPU/reference와 kernel source 좌표를 잇는다

작은 fixture는 의미를 보존한다. tail wrong answer라면 tile boundary `N-1,N,N+1`, batch slot race라면 requests A/B와 cancel/reuse, graph race라면 capture+두 replays, bandwidth라면 same FLOPs with controlled reuse, launch overhead라면 repeated short ops를 남긴다.

CPU/reference는 absolute truth라는 이름이 아니라 independent implementation과 semantic contract다. dtype/rounding/softmax order 허용 오차를 정한다. exact token decision을 보려면 logits margin과 sampling seed를 기록한다. tolerance를 넓혀 identity/order bug를 numerical noise로 숨기지 않는다.

checkpoint0은 host/request metadata다. logical lengths, positions, block tables, RNG state와 output cursor를 비교한다. checkpoint1은 device input view/selected kernel arguments다. metadata가 이미 다르면 kernel arithmetic으로 내려가지 않는다.

checkpoint2는 kernel internal boundary에서 가능한 output tile/partial reduction 또는 debug checksum이다. checkpoint3은 final kernel output, checkpoint4는 next layer/consumer다. instrumentation이 너무 비싸면 binary search로 earliest layer/kernel를 찾고 controlled build에서 내부 checkpoint를 추가한다.

source card에는 dispatcher predicate, launcher arguments, kernel address expression, synchronization, compute/reduction와 store predicate를 둔다. actual pinned file/symbol/span과 build specialization을 연결한다. wrapper와 native kernel 사이 argument transforms를 생략하지 않는다.

vLLM path는 scheduler/runner metadata→custom op/backend selector→extension launcher→kernel→output reconcile을 잇는다. graph replay면 captured inputs, persistent buffers와 update/copy generation을 별 row로 둔다. eager success/graph failure는 kernel body보다 replay lifetime 후보를 올린다.

SGLang path는 schedule batch/token mappings→kernel API wrapper/logging→FlashInfer/other extension plan/run→native launcher/barrier→result processing을 잇는다. normal/overlap과 workspace owners를 기록한다. API log가 host call 완료만 뜻하는지 device completion을 뜻하는지 구분한다.

FlashInfer path는 policy/backend choice, workspace query/allocation, plan serialization/rehydration, native op and kernel key를 연결한다. prebuilt cubin/JIT/fallback artifact generation도 붙인다. same Python function이 다른 native specialization을 고를 수 있다.

llama.cpp path는 server slot/batch metadata→ggml graph/backend scheduling→CUDA op dispatch→kernel/graph launch→copy/sync/output을 잇는다. graph update fallback과 slot generation을 본다. CPU/reference backend fixture가 semantic comparison에 유용하지만 different algorithm tolerance를 명시한다.

source fact는 possible dataflow/invariant를, compile artifact는 registers/shared/spill/binary를, Systems는 timeline/귀속을, Compute는 selected kernel counters를, sanitizer는 instrumented execution의 classes를, reference는 semantic divergence를 제공한다. 역할을 바꾸지 않는다.

## 47.11 wrong answer와 race의 evidence ladder

관측은 batch slot reuse 뒤 request B의 token이 가끔 틀리고 concurrency1 또는 profiler on에서 사라지는 것이다. final output만으로 numerical issue와 race를 가르지 않는다. B input/logical metadata, device buffer generation과 first bad layer를 비교한다.

Systems baseline trace는 producer stream metadata copy, consumer graph replay와 event edge를 보여 줄 수 있다. event가 없거나 wrong generation event를 기다리면 ordering 후보다. 하지만 timestamp overlap만으로 data race를 증명하지 않는다. CUDA stream dependency source와 controlled delay가 필요하다.

Compute Sanitizer memcheck는 out-of-bounds/misaligned/uninitialized access 후보를, racecheck는 supported shared-memory hazards 범위 등을 탐지한다. 도구가 report0이라고 inter-stream logical ownership race가 없다고 단정하지 않는다. tool scope와 unsupported synchronization/address spaces를 공식 docs에 맞춘다.

CPU/reference fixture는 A/B distinct sentinel과 slot reuse generation을 사용한다. producer completion을 barrier로 멈추고 B replay를 시도한다. expected는 B가 matching event를 기다리거나 old completion을 stale로 reject하는 것이다. pointer bounds는 모두 유효하게 만들어 valid-but-wrong을 잡는다.

사건에서 graph input buffer pointer는 동일하고 allocation bounds도 맞았다. A generation copy가 늦게 완료돼 B metadata를 덮었고 B replay가 current event가 아니라 reused event slot의 stale ready state를 보았다. first bad checkpoint는 kernel input active length였다. kernel math와 store는 주어진 wrong input에 대해 올바랐다.

profiler on에서 failure가 사라진 이유는 launch serialization/collection overhead가 A copy를 B replay 전에 끝내 race window를 닫았기 때문이다. “프로파일하면 정상”은 hardware issue 반증이 아니라 perturbation evidence다. baseline trace와 controlled barrier fixture가 원인을 고정한다.

수정은 request/slot generation을 buffer and event descriptor에 포함하고 producer completion G가 consumer replay G보다 앞선 exact dependency를 둔다. old completion은 new owner metadata를 publish하지 않는다. device-wide synchronize 대신 minimal edge를 사용한다.

회귀는 eager/graph, normal/overlap, cancel-before/after-copy, A/B reorder, duplicate/late completion, slot reuse와 tail shapes를 교차한다. reference parity, generation assertion, no sanitizer finding와 performance를 본다. profiler off baseline에서 반복한다.

rollback은 new graph/event generation admission을 fence하고 inflight launches/copies를 drain한다. suspect graphs/buffers/event pools를 폐기하고 verified eager 또는 old artifact로 route한다. already delivered wrong outputs scope와 client terminal을 기록한다.

## 47.12 bandwidth와 launch overhead의 evidence ladder

latency 증가가 bandwidth인지 먼저 Systems에서 kernel duration과 host gaps를 나눈다. GPU timeline이 many 5µs kernels 사이 10µs host gaps면 launch/dispatch overhead 후보다. 한 kernel이 긴 duration을 차지하면 Compute/source/roofline으로 내려간다.

launch overhead fixture는 same small op를 N회 eager, fused, graph replay로 비교한다. total valid work와 output parity를 맞춘다. CPU API duration, inter-launch gaps, kernel durations, capture/instantiate와 warm replay를 분리한다. first request JIT/capture를 steady average에 숨기지 않는다.

bandwidth fixture는 FLOPs와 logical bytes, reuse bounds를 계산한다. Compute의 DRAM/L2 absolute bytes/time과 achieved bandwidth, memory stalls를 counter definition/version과 함께 본다. HBM utilization percentage 하나로 판정하지 않는다. L2 hit와 shared/register spill traffic도 본다.

사건 A에서 Systems는 kernels5µs×100, gaps8µs×99로 total 약 1.29ms를 보였다. Compute는 개별 kernel DRAM bytes가 작고 profile replay overhead가 상대적으로 컸다. bandwidth가 아니라 launch sequence가 first dominant interval이다. fusion/graph 후보를 본다.

사건 B에서 one kernel500µs, DRAM bytes600MiB면 약 1.2TB/s다. device sustained bound와 L2 misses, logical lower/upper를 비교한다. source에서 repeated KV passes or uncoalesced tail을 찾는다. launch overhead5µs는 total의 1%이므로 주원인이 아니다.

average counter가 두 shapes를 합치면 숨길 수 있다. common small shape90%는 launch-bound이고 rare long10%는 bandwidth-bound일 수 있다. kernel name average는 어느 것도 정확히 설명하지 않는다. specialization/shape cohort와 per-request trace를 연결한다.

수정은 launch-bound면 fusion, batching, graph or persistent scheduling을 검토하되 registers/shared/workspace와 queue wait를 기록한다. bandwidth-bound면 reuse/layout/coalescing/tile을 검토하되 traffic level과 correctness를 검증한다. 둘을 같은 “GPU 최적화”로 합치지 않는다.

rollback은 fusion/graph artifact generation과 inflight output commit을 관리한다. bandwidth layout 변경은 cache/tensor layout compatibility와 old handles를 확인한다. performance rollback도 service/resource/telemetry terminal을 갖는다.

## 47.13 profiler가 원인을 바꾼 incident

관측은 production에서 1,000회 중 3회 wrong answer가 나지만 Nsight Compute로 kernel을 profile하면 10,000회 모두 정상인 것이다. 팀은 hardware transient 또는 이미 해결된 문제라고 결론 내렸다. 그러나 profiler가 kernel replay/serialization, cache state와 timing을 바꿀 수 있다는 조건을 빠뜨렸다.

첫 분기는 deterministic semantic bug와 timing-sensitive bug다. profiler off에서 same seed/input/binary인데 결과가 드물게 갈리고 concurrency1에서 사라진다. tile boundary와 input values보다 A/B request overlap/cancel에 상관된다. race/lifetime 후보가 올라간다.

Systems의 낮은-overhead baseline collection에서도 instrumentation overhead가 있지만 production ordering에 더 가까운 timeline을 얻는다. 실패 직전 A metadata copy와 B graph replay가 different streams에서 겹치고 expected event edge가 없다. successful neighbors는 copy가 우연히 먼저 끝났다.

Compute kernel replay mode는 selected kernel을 counter collection을 위해 반복하거나 serialized context에서 실행할 수 있다. replay 전에 inputs/cache state를 save/restore할 수도 있다. exact behavior는 tool version/config documentation을 확인한다. 이 변화가 inter-request producer race를 제거할 수 있다.

Sanitizer도 instrumentation으로 execution을 크게 늦추고 ordering을 바꾼다. report가 없고 failure가 사라졌다고 race-free가 아니다. 반대로 report가 있으면 instrumented path에서 실제 invalid access/hazard evidence다. false-positive/unsupported scope를 공식 docs에 맞춰 해석한다.

CPU/reference는 production ordering을 직접 보존하지 않으므로 race를 재현하려 barrier-controlled device fixture를 만든다. A copy를 event 직전 멈추고 B buffer generation을 publish/replay한 뒤 A completion을 풀어 준다. expected generation invariant를 assert한다.

fixture에서 B replay input checksum이 A late completion 뒤 바뀌고 first kernel output부터 reference와 갈렸다. allocation bounds는 맞고 sanitizer memcheck report는 0이다. physical pointer는 valid하지만 logical owner generation이 틀린 valid-but-wrong incident다.

source walk는 slot allocator generation, async H2D/device copy producer stream, event-pool allocation/reuse, graph input update와 replay consumer wait를 잇는다. event object가 reused됐지만 generation G+1 descriptor는 G ready flag를 current로 오인했다.

first divergence는 profiler counter가 아니라 event-generation publication이다. profiler on에서는 extra synchronization/serialization으로 A completion이 B replay 전에 끝나 invariant가 우연히 만족됐다. 따라서 profiled success는 cause를 숨긴 observation이다.

수정은 event/buffer descriptor에 request and reuse generation을 넣고 producer completion G+1만 consumer G+1을 release하게 한다. B publish 이전에 descriptor initialized, copy issued/recorded and exact wait edge가 있어야 한다. stale callback/event는 drop/cleanup한다.

성능을 위해 global sync를 넣지 않는다. exact producer-consumer stream dependency를 둔다. event pool reuse는 prior wait consumers가 terminal된 뒤 이루어진다. graph update/replay와 buffer overwrite의 last-consumer/first-writer fence를 함께 본다.

회귀 matrix는 profiler off를 primary로 하고 Systems minimal, Compute, sanitizer를 secondary evidence로 둔다. eager/graph, normal/overlap, cancel/reuse, A/B order, delay positions와 event-pool wrap을 교차한다. each run은 generation/checksum/parity를 검증한다.

profiler perturbation fixture 자체도 유지한다. instrumentation on/off에서 semantic output parity는 같아야 하지만 timing distribution은 다를 수 있다. failure가 on에서만 또는 off에서만 나타나면 도구 effect를 incident record에 포함한다. tool version/config를 고정한다.

rollback은 new event/graph code generation admission을 fence하고 inflight streams/graphs/copies를 drain한다. verified eager path 또는 old event implementation으로 route한다. buffers/event pool을 process restart 없이 안전히 reconcile할 수 없으면 bounded worker restart를 사용한다.

already delivered wrong output는 internal state reset으로 복구되지 않는다. affected request generation/time/worker scope를 조사하고 client-facing terminal/incident follow-up을 기록한다. correctness incident를 latency-only rollback으로 축소하지 않는다.

incident 문장은 “profiler에서는 재현되지 않았다”가 아니다. “Compute replay가 streams를 serialize해 race window를 닫았고, production에서는 event pool의 old ready generation이 B replay를 조기 release해 A late copy가 B input을 덮었다”라고 쓴다.

## 47.14 네 증상의 회귀·관측·rollback terminal

### 47.14.1 증상별 최소 경로를 먼저 고른다

모든 profiler와 sanitizer를 한꺼번에 켜기 전에 사용자 증상을 네 갈래로 고른다. 값이 틀리면 `CPU/reference checkpoint→첫 mismatch→launcher·kernel·consumer`만 먼저 연다. 실행할 때마다 달라지거나 취소·재사용 직후에만 깨지면 `request generation→producer event→consumer wait→reuse`를 본다. kernel 하나가 길어졌다면 `selected binary·shape→logical bytes/FLOPs→HBM·L2 counter`를, 짧은 kernel과 빈 gap이 늘었다면 `request span→CPU API→launch/capture/replay`를 먼저 잇는다.

이 최소 경로에서 결론을 바꿀 관측 하나를 얻은 뒤에만 다른 도구로 넓힌다. reference mismatch가 kernel input 전에 이미 보이면 Compute counter를 먼저 모을 이유가 없다. valid pointer인데 generation이 다르면 memcheck clean은 반증이 아니다. many-short-gaps 사건에 bandwidth report부터 붙이면 host launch 원인을 가릴 수 있다. 아래 전수 항목은 네 경로를 모두 수행하라는 목록이 아니라 선택한 가설을 수정·배포·rollback까지 닫는 확장 장부다.

### 47.14.2 선택한 경로를 회귀 fixture와 artifact로 확장한다

wrong-answer fixture는 reference checkpoints, tail shapes, graph/eager, identity/generation와 output parity를 갖는다. tolerance와 RNG를 고정한다. first mismatch가 kernel input 전이면 launcher/metadata, kernel 내부면 address/sync/compute, 이후면 consumer/output을 본다.

race fixture는 interleaving을 확률에 맡기지 않는다. barriers/events로 producer/consumer/cancel/reuse 순서를 강제한다. pointer bounds가 유효한 sentinel을 사용해 sanitizer가 못 잡는 ownership race를 검증한다. N 반복 뒤 stale writes0과 resource cleanup을 본다.

bandwidth fixture는 logical FLOPs/bytes와 HBM/L2 upper/lower, same valid work를 갖는다. Compute counter version/definition, binary/specialization와 shape cohort를 고정한다. duration 개선이 traffic 감소/throughput/other stall 중 무엇인지 source mutation과 잇는다.

launch-overhead fixture는 op count, CPU API/gap/kernel timeline, cold JIT/capture와 warm replay를 분리한다. fusion/graph/batching이 queue wait, compile, registers/workspace와 correctness에 지불하는 비용을 기록한다. end-to-end SLO를 승인한다.

Systems artifact는 request→CPU span→CUDA API→stream/kernel/memcpy/event의 attribution을 소유한다. Compute artifact는 selected kernel resource/warp/memory counters를 소유한다. sanitizer artifact는 instrumented memory/sync findings를 소유한다. CPU/reference artifact는 semantic expected values/checkpoints를 소유한다.

한 artifact가 다른 결론을 대신하지 않는다. Systems gap은 kernel bandwidth를 확정하지 않고, Compute average는 request ordering을 보여 주지 않으며, sanitizer clean은 logical generation race 부재를 증명하지 않는다. reference mismatch는 source owner를 자동 지정하지 않는다.

운영 metric은 backend/specialization/fallback, graph hit/capture, kernel interval/error category, queue/host gap proxy와 generation assertions을 bounded하게 둔다. exact symbol/shape/pointer/tensor/counter는 sampled trace/profile artifact다. profiler를 production 상시 gate로 요구하지 않는다.

incident 시작 trigger는 wrong output/generation assertion/sanitizer finding, shape-specific latency, DRAM/local traffic proxy, tiny-kernel/gap increase다. trigger가 어느 evidence ladder를 열지 정한다. generic `gpu_slow` alert 하나로 모든 tool을 켜지 않는다.

source card에는 dispatcher→launcher→kernel address/sync/compute/store→consumer를 잇고 build artifact를 붙인다. fix diff는 invariant로 표현한다. tail bounds, generation happens-before, traffic reuse 또는 launch-count reduction 중 무엇이 새로 보장되는지 쓴다.

반증 table은 가설, expected observation, actual evidence, verdict와 next owner를 갖는다. HBM 가설은 bytes/bandwidth bounds, launch 가설은 Systems gaps, race 가설은 interleaving/generation, numerical 가설은 reference/checkpoint/tolerance로 판정한다.

rollout은 binary/config/graph/event generation cohort로 나눈다. common/rare shapes와 failing interleaving을 canary한다. correctness hard gates를 먼저, then SLO/goodput and resource budgets를 본다. profiler-generated cache/graphs가 production artifact와 섞이지 않게 한다.

rollback trigger는 numeric mismatch, sanitizer finding, generation violation, launch error, memory bound breach, host-gap/kernel-tail SLO와 resource leak다. affected generation admission을 fence한다. inflight requests/kernels/copies/graphs/events/workspaces를 drain/terminal한다.

fallback은 안전성/성능 계약을 명시한다. suspect custom kernel에서 reference/generic backend로 전환할 수 있지만 supported shapes/dtypes, output parity와 latency capacity를 검증한다. fallback success를 root fix로 닫지 않는다.

rollback readiness는 old binary/source/config fingerprint, selected backend/specialization, graph/event self-test와 failing fixture를 포함한다. process alive나 aggregate error0만으로 traffic을 열지 않는다. old generation pending0과 resource baseline을 본다.

service terminal은 request/stream output/error, resource terminal은 device allocations/workspaces/graphs/events/JIT modules, telemetry terminal은 trace/profile generation and counters다. wrong output affected scope와 client impact도 terminal record에 둔다.

reviewer는 profiler가 scheduling/cache state를 바꿨는지, average가 shape cohorts를 섞었는지, successful profile가 failing binary/key와 같은지 공격적으로 묻는다. 결론을 바꿀 빈칸에만 최소 관측을 추가한다.

최종 dossier는 observation, same-key neighbor, CPU/reference checkpoints, Systems timeline, selected Compute report, sanitizer scope/report, pinned source/binary, first divergence, falsifiers, patch invariant, regression and rollback ledger다.

독자는 이 dossier로 네 증상을 분리할 수 있어야 한다. wrong value이면 semantic checkpoint, timing이면 generation/order, long kernel이면 memory/resource/compute, many short gaps이면 launch/host path를 먼저 연다. 이후 evidence가 가설을 바꾸면 owner도 바꾼다.

이 장의 완료 조건은 도구를 모두 실행하는 것이 아니다. 사건을 바꾸지 않는 최소 관측과 controlled fixture로 first divergence를 찾고, 도구 perturbation을 evidence로 기록하며, 수정과 rollback을 같은 invariant에서 검증하는 것이다.

**평균 counter가 first divergence를 숨긴 두 번째 incident.**

관측은 attention kernel 평균 duration과 평균 DRAM throughput이 release 전후 거의 같지만 user p99가 두 배가 된 것이다. 평균 report만 보고 GPU kernel이 원인이 아니라고 결론 냈다. request trace를 shape cohort로 나누자 rare tail `M=65`에서만 duration이 10배였다.

common `M=64`가 traffic95%, tail65가 5%라고 하자. baseline durations100µs/120µs, candidate90µs/1,200µs면 weighted average baseline101µs, candidate145.5µs다. traffic mix/aggregation window가 candidate common 비중을 더 높이면 global average 차이가 작아질 수 있다. p99는 tail을 직접 본다.

Compute report가 kernel name으로 specializations을 합쳤다면 candidate M64 native optimized path와 M65 fallback/generic path의 counters가 섞인다. selected function/binary/template와 launch shape를 separate reports로 고정한다. same display name이 same code object를 뜻하지 않는다.

Systems request correlation은 M65에서 preferred kernel reject→fallback launch sequence와 extra metadata/memcpy를 보였다. M64는 one optimized launch다. long duration이 DRAM bandwidth 때문인지 fallback launch overhead/algorithm 때문인지 더 나눈다.

M65 Compute evidence는 DRAM bytes per valid output가 M64보다 커지고 multiple passes가 있으며 achieved bandwidth는 lower였다. source selector의 tail predicate가 new alignment constraint 때문에 fallback을 골랐다. 평균 DRAM counter는 M64 savings가 M65 explosion을 가렸다.

CPU/reference output은 맞아 correctness 문제는 아니다. sanitizer도 report0이고 bounds가 정상이다. source first divergence는 candidate selector condition `M % tile == 0`에서 M65를 new generic fallback으로 보낸 지점이다. kernel body regression이 아니다.

수정 후보는 tail-capable specialization, padding with valid-output predicate 또는 old fallback 복원이다. padding은 submitted work/bytes와 graph bucket을 늘릴 수 있고 tail kernel은 extra code size/register cost가 있다. production M histogram과 SLO로 선택한다.

regression은 `63,64,65,127,128,129`, batch/head dims와 dtype을 포함한다. selected specialization/fallback reason, valid/submitted work, launches, DRAM/L2 bytes, duration, output parity를 assert한다. kernel-name average를 gate로 쓰지 않는다.

rollback은 selector/config/binary generation을 old mapping으로 되돌리고 captured graphs/specialization caches를 폐기한다. inflight M65 requests를 safe fallback 또는 explicit terminal로 닫는다. global average 복귀보다 tail cohort p99 and selection leaf를 확인한다.

incident 문장은 “평균 counter는 정상”이 아니다. “M64 optimized path savings가 M65 fallback의 multi-pass bytes/launch를 aggregate에서 가렸고 selector alignment predicate가 first divergence였다”라고 쓴다.

**Compute Sanitizer를 incident class에 맞춰 선택한다.**

out-of-bounds, misaligned, invalid address와 some memory errors는 memcheck 후보다. uninitialized memory usage는 initcheck 범위를 확인한다. shared-memory race hazards는 racecheck, barrier/synchronization misuse는 synccheck 범위를 official tool docs/version으로 읽는다. tool 이름만 보고 모든 CUDA race를 포괄한다고 쓰지 않는다.

sanitizer 실행은 minimal fixture와 exact binary/specialization을 사용한다. debug/line info가 source correlation에 필요할 수 있지만 compilation flags가 optimization/resource/timing을 바꾸는지 기록한다. release binary와 instrumented binary 차이를 남긴다.

report의 address/thread/block/instruction/source line을 logical tensor/view와 연결한다. allocation bounds 안이어도 wrong request generation이면 report가 없을 수 있다. allocation violation과 logical ownership violation을 구분한다.

sanitizer overhead로 timeout/watchdog 또는 scheduling change가 생길 수 있다. production SLO를 instrumented duration으로 판단하지 않는다. report evidence와 semantic fixture를 분리한다. race가 사라지는 것도 perturbation이다.

clean report는 selected tool scope의 관측 결과다. wrong output가 남으면 reference checkpoint와 logical generation, numerical/reduction/order로 이동한다. sanitizer가 clean이라고 kernel correctness를 승인하지 않는다.

report가 있으면 first error와 cascaded errors를 구분한다. first invalid write가 뒤의 unrelated read faults를 만들 수 있다. source address predicate와 tail fixture로 first mutation을 고친다. suppress/mute로 incident를 닫지 않는다.

fix 뒤 original report0, reference parity, boundary/reuse fixtures와 release performance를 함께 본다. sanitizer-only build에서만 통과하면 production artifact에서도 invariant assertion/fixture를 확인한다.

**Nsight Systems timeline을 serving latency와 연결한다.**

request arrival, queue/admission, CPU preprocessing, scheduler plan, H2D, kernel/collective, D2H/output and client delivery spans를 연결한다. GPU kernel interval만 줄여도 queue/transport가 지배하면 TTFT/ITL은 안 변할 수 있다.

launch overhead는 CPU API calls와 GPU executions 사이 gap, launch count와 duration 분포로 본다. Python/GIL, dispatcher, serialization, graph capture/JIT와 synchronization API를 별 spans로 둔다. blank GPU timeline을 전부 CPU launch overhead라고 부르지 않는다.

asynchronous API duration과 device completion을 구분한다. host launch가 짧고 later synchronize가 길면 offender는 preceding device work일 수 있다. error가 synchronize에서 보고돼도 actual invalid kernel interval을 trace correlation으로 찾는다.

multi-stream overlap은 visual overlap만으로 dependency correctness를 증명하지 않는다. expected event edges와 source stream assignments를 붙인다. missing/wrong-generation event는 controlled fixture로 검증한다.

collective/communication overlap도 구분한다. kernel wait가 NCCL/transfer dependency라면 local memory bandwidth를 튜닝하지 않는다. request/plan generation과 collective ranks를 연결한다. 이 장에서는 profiler general론보다 selected incident edge만 본다.

Systems collection overhead와 trace buffer/drop도 기록한다. long production trace가 events를 누락하면 absence를 proof로 쓰지 않는다. shortest failing prefix와 sampling/capture window를 설계한다.

**Nsight Compute report를 source mutation에 묶는다.**

report 전에 exact launch instance를 request trace에서 고른다. kernel name, grid/block, dynamic shared, stream, correlation ID, shape/key와 binary hash를 저장한다. wrong instance의 detailed counters는 사건을 해결하지 않는다.

kernel replay/cache control이 inputs/timing에 미치는 영향을 tool config와 docs에서 확인한다. race incident는 replay보다 stable performance fixture에서 사용한다. profile-on/off parity와 duration difference를 artifact에 둔다.

sections/counters는 hypothesis에 필요한 최소로 선택한다. launch stats/resource, memory workload, scheduler/warp stalls와 compute를 필요에 따라 본다. 모든 sections 수집이 replay/overhead를 늘리고 원인을 흐릴 수 있다.

counter는 absolute work와 normalization을 함께 둔다. bytes, sectors, instructions, cycles, active warps를 valid outputs/tokens와 submitted work로 나눈다. percentage alone을 피한다. architecture/toolkit metric definition을 저장한다.

source correlation은 exact SASS/source line이 가능하면 사용하되 generated/template code와 inlining을 고려한다. hotspot line이 causal mutation인지 baseline/candidate diff와 source path로 검증한다. duration이 긴 line은 downstream wait일 수 있다.

fix가 selector/launch count라면 individual kernel Compute report가 같아도 end-to-end가 개선될 수 있다. Systems와 service metric이 owner다. fix가 memory layout이면 Compute bytes/stalls와 source address path가 owner다. 도구 역할을 맞춘다.

**CPU/reference fixture 설계의 함정.**

CPU backend가 GPU와 다른 dtype/accumulation/order를 쓰면 exact equality를 요구하지 않는다. known tolerance와 invariant, logits margin을 사용한다. 그러나 request identity/position/shape mismatch는 tolerance로 허용하지 않는다.

fixture가 너무 작아 다른 specialization을 선택하면 failure를 잃는다. launcher key를 assert하고 source failing path가 선택되는 최소 shape를 유지한다. M65를 M2로 줄여 generic CPU/GPU path로 바꾸지 않는다.

cache/allocator warm state와 graph capture가 원인이면 fixture setup에 포함한다. first invocation only/cold JIT 사건은 warm-only loop로 재현되지 않는다. cancel/reuse race는 two requests와 interleaving을 남긴다.

random sampling을 제거해도 stochastic kernel/reduction behavior를 조사하려 multiple seeds/orders를 보존할 수 있다. deterministic reference와 production semantics를 둘 다 기록한다. correctness contract가 deterministic인지 tolerance-based인지 밝힌다.

reference도 bug가 있을 수 있다. second independent implementation, mathematical invariant와 small hand calculation로 triangulate한다. GPU path 둘이 같은 wrong metadata를 공유하면 서로 일치할 수 있다. input checkpoint를 포함한다.

fixture output에는 final value뿐 아니라 selected path, intermediate checkpoints, generations, events와 resource terminal을 포함한다. refactor 후 function name이 바뀌어도 semantic invariant가 남게 한다.

**수정 승인과 운영 rollback.**

patch review는 invariant diff로 시작한다. bounds predicate, generation happens-before, shape-specific selector, traffic reuse 또는 launch fusion 중 무엇을 보장하는지 한 문장으로 쓴다. unrelated synchronization이나 larger buffers로 symptom만 줄이지 않는다.

correctness gate는 original failure, boundaries/interleavings, cache/graph/eager, cancellation/reuse와 reference parity다. sanitizer supported checks와 generation assertions을 포함한다. profiler off production-like run이 primary다.

performance gate는 shape cohort별 end-to-end TTFT/ITL/goodput와 Systems intervals, 필요한 Compute bounds다. average-only 승인 금지다. profiler overhead/cold-warm and valid/submitted work를 구분한다.

resource gate는 allocations/workspaces, graphs/events, compiler registers/shared/spill과 cleanup baseline이다. fusion/graph/sync 수정이 memory/occupancy/JIT cache를 악화하지 않는지 본다.

rollout은 binary/config/graph generation canary와 bounded failure injection을 사용한다. unexpected fallback, generation assertion, numeric mismatch는 즉시 fence다. latency/traffic counters는 predeclared threshold를 쓴다.

rollback은 new admission 차단→inflight request/kernel/copy/graph drain→artifact/cache 격리→verified fallback/old binary load→self-test→canary→readiness 순서다. affected client outputs와 resource reconciliation을 기록한다.

final terminal은 service, device/runtime resource, telemetry/profile artifact와 incident scope다. process restart로 counters0이 됐다는 사실과 root invariant 복원을 구분한다. original fixture가 old/new path에서 expected 결과를 내야 한다.

최종 승인 문장은 observation, first divergence, falsifier, patch invariant and fixtures를 연결한다. 다른 엔지니어가 source coordinates와 artifacts로 결론을 재현할 수 없으면 아직 완료가 아니다.

**네 사건의 최종 worksheet.**

wrong-answer 행에는 request/key, expected/actual output, earliest mismatching checkpoint, selected kernel/source store와 reference tolerance를 쓴다. allocator bounds, logical subview와 generation을 분리한다. passing neighbor는 same specialization과 shape boundary를 가져야 한다.

race 행에는 producer/consumer owners, streams/events, buffer/event generations, last consumer와 first writer를 쓴다. observed timestamps보다 expected happens-before를 먼저 쓴다. barrier fixture로 each interleaving expected wait/reject/commit을 검증한다.

bandwidth 행에는 FLOPs, logical and observed HBM/L2/shared/local bytes, duration/achieved bandwidth와 upper/lower bounds를 쓴다. source layout/pass/reuse and compiler spill을 연결한다. device peak percentage는 보조다.

launch-overhead 행에는 request CPU spans, CUDA API count/duration, GPU gaps, kernel count/duration, graph/capture/JIT cold/warm과 end-to-end SLO를 쓴다. empty GPU interval의 actual host/dependency owner를 표시한다.

도구 열은 Systems/Compute/Sanitizer/reference를 체크하는 목록이 아니다. 각 도구가 답할 질문과 perturbation을 쓴다. 필요 없는 detailed profiling을 하지 않는다. report가 없는 칸은 inference와 observed fact를 구분한다.

source 열은 dispatcher predicate, launcher args, kernel address/sync/compute/store, consumer를 pin한다. build binary/toolkit/SM도 붙인다. function 이동보다 semantic key/invariant를 regression identity로 둔다.

fix 열은 restored invariant와 cost movement다. exact event edge, tail predicate, selector correction, fusion/graph or layout reuse가 무엇을 얻고 registers/shared/workspace/queue에서 무엇을 지불하는지 적는다.

terminal 열은 output/client, device resources and telemetry다. rollback generation/fixture와 old pending0을 확인한다. 이 worksheet가 한 사건에서 닫히면 증상 label을 넘어 원인과 복구가 연결된다.

**숫자 예제로 네 원인을 구분한다.**

request A는 total1.2ms이며 100개의 5µs kernels와 평균 7µs gaps다. kernel 합 0.5ms, gaps 약 0.693ms다. HBM bytes total50MiB이고 GPU kernel 동안 bandwidth100GB/s 수준이면 launch/host gap이 먼저다. fusion/graph를 검토한다.

request B는 one kernel1ms, DRAM1.2GiB면 약 1.2TB/s다. device sustained range와 L2 miss, logical bytes를 확인한다. host gap20µs를 줄여도 2%다. bandwidth/reuse/layout을 먼저 본다.

request C는 kernel100µs인데 output mismatch1/1,000이고 profiler on0/10,000이다. bytes/duration 평균은 정상이다. A/B generation barrier에서 reproducible mismatch가 나오면 race owner다. bandwidth/launch tuning이 correctness를 고치지 않는다.

request D는 M65만 output mismatch100%, M64 정상이다. sanitizer reports invalid store at tail epilogue and reference first diverges kernel output. deterministic bounds bug다. concurrency를 줄여도 유지되며 race보다 tail predicate가 먼저다.

같은 `cuda_kernel_latency` 평균에 A/B가 섞이고 error rate에 C/D가 섞이면 네 대응이 보이지 않는다. request/shape/key와 semantic evidence로 cohort를 나눈다. bounded metric은 경보, trace/fixture는 판정이다.

**프로파일러 사용 전후 체크리스트.**

before에는 exact failing binary/key, reproduction rate/timing, workload state와 reference checkpoints를 보존한다. profiling이 cache warmness, graph/JIT, stream concurrency와 request schedule을 바꿀 가능성을 쓴다.

collection은 가장 덜 교란하는 tool/query부터 시작한다. Systems attribution, selected kernel Compute sections, relevant sanitizer tool과 controlled reference 순서는 incident에 따라 바뀐다. 한 번에 모두 켜지 않는다.

after에는 selected launch가 failing one인지, failure rate/ordering이 이동했는지, replay/serialization/cache control이 무엇인지 기록한다. profiled success와 baseline failure를 같은 결과로 평균내지 않는다.

counter report는 exact metric definitions, toolkit/driver/SM, duration/work normalization and collection mode를 보존한다. screenshot만 incident artifact로 남기지 않는다. source/binary correlation IDs를 붙인다.

tool-induced divergence 자체가 evidence다. failure가 사라지면 timing/lifetime 후보를 올리되 root cause로 단정하지 않는다. failure가 새로 생기면 instrumentation/debug build differences를 본다. controlled fixture로 invariant를 직접 검증한다.

**다음 release에서도 살아남는 회귀 계약.**

tail contract는 all writer addresses inside logical output and all required rows written exactly once다. generation contract는 producer completion G precedes consumer G and stale G-1 cannot mutate G다. bandwidth contract는 traffic bounds per valid work다. launch contract는 intended op sequence and cold/warm ownership이다.

tests는 function/counter 이름보다 이 contracts를 assert한다. current adapters가 source symbols/metrics를 제공하고 revision diff 때 업데이트된다. semantic fixture는 유지된다.

CUDA/toolkit/compiler가 바뀌면 resource/counter baseline을 다시 만든다. source same이라고 binary scheduling/spill이 같다고 가정하지 않는다. graph/profiler behavior와 sanitizer scope/version도 재고정한다.

backend selector가 바뀌면 fixture가 expected failing/fixed specialization을 실제 선택하는지 먼저 assert한다. fallback 통과가 custom kernel fix를 위장하지 않는다. leaf와 reason을 기록한다.

production에는 lightweight generation/error/specialization/latency proxies를 유지하고 periodic controlled artifacts를 생성한다. low-level all-counter monitoring을 요구하지 않는다. alert는 어느 worksheet를 열지 알려 준다.

최종 handoff는 six sentences와 artifact links다. symptom/cohort, first divergence, expected invariant, supporting/falsifying evidence, fix invariant, regression/rollback terminal을 쓴다. 이 형식이 있으면 profiler tool names가 바뀌어도 조사 논리는 유지된다.

이제 독자는 wrong answer/race/bandwidth/launch overhead를 한 `GPU kernel issue`로 합치지 않는다. 각 사건의 의미·시간·byte와 owner를 분리하고, 도구가 관찰 자체를 바꿀 수 있음을 포함해 가장 작은 source mutation을 찾는다.

**배포 당일 20분 canary 판정.**

첫 5분에는 artifact/config generation과 selected backend/specialization을 확인한다. original failing boundary/interleaving과 common passing fixture를 보낸다. output/reference checkpoint와 generation assertions을 correctness hard gate로 둔다.

다음 5분에는 Systems-level request intervals를 본다. queue/CPU launch gaps, kernels/copies/events와 output delivery가 baseline budget 안인지 확인한다. cold JIT/capture와 warm path를 분리한다. profiler 없이 lightweight trace로 시작한다.

세 번째 5분에는 resource/traffic proxies를 본다. fallback ratio, graph recapture, error categories, memory/workspace watermark and kernel-duration shape cohorts다. expected bounds 밖이면 canary를 멈추고 controlled Compute/sanitizer reproduction으로 이동한다.

마지막 5분에는 rollback readiness를 점검한다. new admissions fence, inflight drain, old artifact/fallback routing, graph/event/workspace cleanup과 self-test가 실행 가능한지 확인한다. traffic 확대 전 rollback을 실제로 rehearsal한다.

**wrong answer가 발견된 뒤의 안전 우선순위.**

numeric mismatch는 평균 성능 개선보다 우선한다. affected specialization/generation을 즉시 격리하고 verified generic/reference backend가 있으면 route한다. unsafe path의 additional profiling을 위해 user traffic을 계속 흘리지 않는다.

snapshot에는 failing input digest/shape, generations, selected binary/key, checkpoints, streams/events와 allocator owners를 보존한다. raw sensitive tensor는 policy에 따라 controlled environment에서만 저장한다. evidence 보존이 확산 차단보다 앞서지 않는다.

fallback은 output parity와 supported feature, capacity/SLO를 확인한다. fallback queue overload가 새 incident를 만들 수 있다. admission limit and routing scope를 조정한다. client retry/partial output idempotency를 본다.

root fix가 준비되면 original race/tail fixture, sanitizer applicable checks와 profiler-off stress를 통과한다. profiler-on success는 secondary다. safe path와 performance path의 artifact identity를 기록한다.

affected wrong outputs의 scope를 request generation/time/worker/key로 추정하고 명확히 보고한다. internal rollback로 이미 전달된 결과가 복구됐다고 쓰지 않는다. service terminal과 follow-up을 남긴다.

**성능 incident의 반대 안전장치.**

latency가 느리다는 이유로 synchronization/bounds/generation checks를 먼저 제거하지 않는다. counters로 overhead가 지배함을 증명하고 semantic invariant를 보존하는 최적화를 찾는다. safety check sampling을 줄이면 detection coverage를 문서화한다.

fusion은 launch count를 줄이지만 register/shared, compile/JIT와 failure blast radius를 키울 수 있다. graph는 host gaps를 줄이지만 persistent inputs/generations와 key cardinality를 만든다. bandwidth layout은 tensor/cache ABI를 바꿀 수 있다. 비용 이동을 review한다.

performance canary는 shape distribution과 p50/p99, valid goodput, correctness/resource gates를 함께 본다. common win이 rare tail을 가리지 않게 한다. average counter-only release gate를 금지한다.

rollback 뒤 latency가 회복돼도 source first divergence가 맞았는지 regression artifacts를 남긴다. feature disable은 containment일 수 있다. root fix ticket과 완료 조건을 별도로 유지한다.

**최종 완료 문장.**

“profiler가 켜지면 정상”이라는 현상을 원인으로 쓰지 않는다. tool replay/serialization이 event race window를 닫았고 controlled interleaving에서 stale generation publish가 first divergence였다는 식으로 쓴다.

“평균 bandwidth는 정상”도 종료 문장이 아니다. M64/M65 specialization을 나누자 fallback multi-pass traffic과 selector predicate가 갈렸다는 식으로 쓴다. aggregate가 숨긴 cohort를 명시한다.

수정 문장은 restored happens-before 또는 selector/bounds/traffic/launch invariant를 쓴다. 회귀는 exact failing key와 passing neighbors, profiler off/on, reference and resources를 포함한다. rollback terminal은 requests/kernels/graphs/events/artifacts/telemetry를 닫는다.

이 문장을 pinned source coordinates와 artifact report로 다른 사람이 재현할 수 있을 때만 장의 조사 계약이 완성된다. 도구 목록이 아니라 first divergence를 줄이는 evidence chain이 결과다.

**마지막 도구 선택 연습.**

증상 1은 deterministic M65 wrong output다. CPU/reference checkpoint와 sanitizer memcheck를 먼저 사용하고 tail launcher/store source를 본다. Systems full trace나 all-counter profiling은 후순위다. expected는 bounds report 또는 kernel-output first mismatch다.

증상 2는 cancel/reuse 때만 intermittent wrong output다. generation trace, Systems stream/event attribution과 barrier-controlled fixture가 우선이다. sanitizer clean은 logical ownership race를 기각하지 않는다. Compute replay가 failure를 숨길 가능성을 기록한다.

증상 3은 one long kernel and high DRAM bytes다. logical roofline bounds, selected Compute memory counters와 source reuse/layout가 우선이다. CPU launch optimization을 먼저 하지 않는다. reference parity는 correctness gate로 유지한다.

증상 4는 hundreds of tiny kernels and GPU gaps다. Systems CPU/API timeline, cold/warm graph/JIT와 op sequence가 우선이다. individual kernel Compute report는 launch gap을 설명하지 못한다. fusion/graph candidate의 resource/parity를 후속 검증한다.

각 선택은 “무슨 도구가 좋은가”가 아니라 어떤 가설을 가장 적게 교란하며 반증하는가에 답한다. 첫 결과가 가설과 다르면 evidence ladder를 바꾼다. 모든 report를 모은 뒤 해석하는 방식보다 빠르고 안전하다.

최종 기록에는 tool version/config, selected binary/key, perturbation, raw units와 source coordinates를 넣는다. screenshot이나 평균 percentage만 남기지 않는다. controlled fixture와 production-like profiler-off 결과를 함께 보존한다.

rollback rehearsal까지 통과하면 incident owner는 종료를 선언한다. 이후 monitoring은 bounded alerts와 periodic fixture로 재발을 감시하고, 새 toolkit/backend/source revision에서 same semantic contracts를 다시 검증한다.

완료 원장에는 passing neighbor, 실패 cohort, profiler-off 재현율과 instrumented perturbation을 함께 남긴다. source mutation과 artifact generation이 일치하고 service·resource·telemetry terminal이 모두 닫혀야 회귀가 사라졌다고 판정한다.

남은 미검증 조건에는 필요한 fixture, 도구 설정, 담당 owner와 중단 기준을 붙인다. 평균 counter나 profiler 성공으로 빈칸을 추정해 닫지 않는다.

이 구분이 다음 release에서도 동일한 조사 계약과 안전한 rollback을 보존한다.

이 장에서 kernel 내부의 first divergence를 찾았다고 model 입력 shape의 생성 이유까지 설명한 것은 아니다. selected binary와 launch shape가 예상과 다르지만 dispatcher·kernel 안에서는 계약대로 움직였다면 48장으로 owner를 넘긴다. 48장은 model config와 architecture class가 tensor shape·head partition·dtype을 어떻게 만들고, loader와 framework가 그 값을 실제 module과 weight에 결합하는지 추적한다. 반대로 48장의 effective config와 runtime tensor shape가 같다면 다시 이 장의 specialization key·launcher·kernel 경계로 돌아온다.
