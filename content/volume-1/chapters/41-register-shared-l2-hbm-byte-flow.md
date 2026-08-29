# 41장. 바이트는 어디에 살고, 누가 다음 장소로 옮기는가

새 모델을 올린 뒤 첫 부하 시험을 시작했다고 하자. GPU 메모리 사용량은 예상 범위이고 오류도 없다. 그런데 attention kernel이 긴 시간을 차지한다. 한 사람은 “HBM이 느려서 그렇다”고 말하고, 다른 사람은 “shared memory로 올리면 된다”고 답한다. 세 번째 사람은 register를 더 쓰도록 tile을 키우자고 한다. 네 번째 사람은 L2 cache hit가 높으니 메모리 문제는 아니라고 결론 내린다. 모두 메모리 계층의 이름을 알고 있지만, 지금 느린 바이트가 어느 주소 공간에 있고 어느 경로를 지나며 무엇과 용량을 다투는지는 아직 아무도 말하지 않았다.

이 상태에서 최적화를 시작하면 원인보다 이름을 고치게 된다. shared tile을 크게 만든 결과 한 CTA가 쓰는 shared memory가 늘어 동시에 상주할 CTA가 줄 수 있다. 중간값을 register에 오래 붙잡아 둔 결과 register pressure가 커지고, 컴파일러가 일부 값을 local memory로 spill할 수도 있다. L2 hit가 높아도 shared staging과 barrier가 critical path일 수 있다. 반대로 profiler에 HBM traffic이 보인다고 해서 모든 global load가 매번 HBM까지 갔다고 말할 수도 없다. global은 CUDA의 주소 공간이고 HBM은 물리 저장 장치이며 L2가 그 사이에 있기 때문이다.

이 장은 메모리 이름 암기표가 아니다. 질문을 바꾼다. 한 byte는 지금 어디에 주소를 갖는가. 그 byte를 어느 thread, block, cluster, device가 볼 수 있는가. 얼마 동안 살아 있어야 하는가. 다음 소비자는 누구인가. 그 소비자에게 전달하려고 어떤 copy, synchronization, layout conversion이 필요한가. 그리고 그 선택이 register 수, shared capacity, L2 residency, device-memory bandwidth에 어떤 압력을 만드는가.

중심 사례는 vLLM v0.27.1의 CMake가 실제로 고정한 FlashAttention 포크의 forward kernel이다. 현재 pin `28e862d21806bc3580207aa0ad4e2759151e9827`의 [`flash_fwd_kernel.h` 163–212행](https://github.com/vllm-project/flash-attention/blob/28e862d21806bc3580207aa0ad4e2759151e9827/csrc/flash_attn/src/flash_fwd_kernel.h#L163-L212)에서 Q, K, V가 global tensor view에서 shared tile로 들어가고, MMA fragment와 accumulator가 되고, online softmax를 거쳐 output으로 저장되는 경로를 한 함수 안에서 추적한다. 코드를 실행하거나 성능을 측정하지 않는다.

코드가 보장하는 구조와 손으로 계산할 수 있는 자원 모델, 나중에 관측해야 할 증거를 구분한다.

이 pin은 이름이 비슷한 오래된 checkout에서 추정한 값이 아니다. vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`의 [`vllm_flash_attn.cmake` 38–44행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/cmake/external_projects/vllm_flash_attn.cmake#L38-L44)이 지정한 `GIT_TAG`를 따른다. 이전 조사 pin과 현재 pin을 Git object 단위로 실제 diff한 결과, 이 파일에는 include와 dropout seed unpack 조건부 처리에서 8행 추가·1행 변경이 있었고, 이로 인해 뒤의 byte-flow 구간 행 번호가 일곱 줄 이동했다.

Q/K/V staging, MMA, online softmax, epilogue의 본체가 diff에서 바뀌지 않았다는 것을 확인한 뒤 아래 좌표를 현재 pin에 다시 맞췄다. “비슷해 보이니 같다”는 가정이 아니라 두 Git object의 파일 diff와 현재 파일의 line view를 근거로 한다.

공식 의미는 NVIDIA CUDA C++ Programming Guide [12.9.1의 Memory Hierarchy](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#memory-hierarchy)와 [13.3.0의 Programming Model](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/01-introduction/programming-model.html)에 고정한다.

세대 차이는 [Ampere Tuning Guide 12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/ampere-tuning-guide/index.html#asynchronous-data-copy-from-global-memory-to-shared-memory), [Hopper Tuning Guide 12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/hopper-tuning-guide/index.html#tensor-memory-accelerator), [Blackwell Tuning Guide 13.3.0](https://docs.nvidia.com/cuda/archive/13.3.0/blackwell-tuning-guide/index.html#thread-block-clusters)을 제품과 compute capability별로 읽는다.

최신 문서의 목차가 달라졌다는 이유만으로 하드웨어 의미가 달라졌다고 추론하지 않는다.

## 41.1 “GPU 메모리”라는 한 단어가 진단을 망치는 순간

### 41.1.1 주소 공간, 할당, 물리 매체는 같은 축이 아니다

개발자는 흔히 `cudaMalloc`이 돌려준 pointer를 “HBM pointer”라고 부른다. 특정 데이터센터 GPU의 일반적인 device allocation을 짧게 말할 때는 통할 수 있다. 그러나 분석 문장으로는 부족하다. CUDA C++에서 global memory는 모든 thread가 접근할 수 있는 global address space와 그 접근 규칙을 가리킨다. device allocation은 CUDA runtime 또는 상위 allocator가 만든 저장 객체와 수명을 가리킨다. HBM은 GPU package에 붙은 물리 메모리 기술을 가리킨다. 세 표현은 서로 관련되지만 같은 분류가 아니다.

왜 엄격히 나눠야 할까. unified memory allocation은 같은 virtual address를 CPU와 GPU가 사용할 수 있고 page migration이나 placement가 개입한다. mapped host memory는 GPU가 global address를 통해 접근해도 byte의 물리적 backing이 host 쪽일 수 있다. peer GPU memory도 한 GPU의 명령이 global address로 접근하지만 물리 byte는 다른 GPU에 있을 수 있다. 일반 device allocation조차 load 시점에 해당 cache line이 L2에 있다면 요청이 곧바로 HBM transaction이 된다고 단정할 수 없다. 주소를 읽는 코드 한 줄과 물리 traffic 한 건은 일대일 대응이 아니다.

따라서 이 책은 세 문장을 분리한다. “kernel이 global address의 Q를 읽는다”는 소스 수준 사실이다. “Q allocation은 device allocator가 요청 lifetime보다 오래 보유한다”는 allocator와 serving runtime 수준 사실이다. “그 load의 일부가 L2에서 충족되지 않아 HBM traffic을 만든다”는 관측 수준 사실이다. 첫 문장만 코드로 확인하고 셋째 문장까지 사실처럼 늘리면 뇌피셜이 된다.

### 41.1.2 local이라는 이름도 함정이다

`local memory`는 CPU 개발자가 떠올리는 작은 on-chip scratchpad가 아니다. CUDA의 local address space는 thread-private naming을 제공하지만, 물리적으로 register file과 동의어가 아니다. 컴파일러가 register에 담지 못한 값, 주소를 취한 자동 배열, 동적 index가 필요한 큰 thread-local 객체가 local memory access가 되면 device memory 계층의 load와 store를 만들 수 있다. 이름의 “local”은 공유 범위를 말하지 속도를 보장하지 않는다.

이 차이는 register pressure를 진단할 때 결정적이다. source에 local array가 보이지 않아도 compiler allocation 결과에서 spill load와 spill store가 생길 수 있다. 반대로 C++ scalar가 많아 보여도 최적화가 값을 제거하거나 live range를 줄일 수 있다. source variable 개수를 세어 register 수를 확정하지 말아야 한다. source는 압력의 후보를 보여 주고, compiler resource report와 disassembly는 실제 배치를 보여 주며, profiler는 실행 중 영향을 보여 준다. 이 장은 첫 단계와 손계산 설계까지만 수행한다.

### 41.1.3 cache와 scratchpad는 관리 주체가 다르다

L1과 L2는 hardware-managed cache다. 프로그램은 load를 발행하고 일부 cache policy나 persistence hint를 지정할 수 있지만, 보통 “이 Q tile의 정확히 이 byte를 다음 barrier까지 L2의 이 위치에 둔다”고 소유하지 않는다. shared memory는 software-managed scratchpad다. kernel이 크기와 layout을 정하고 어느 thread가 언제 채우며 언제 읽어도 되는지 synchronization을 명시한다.

둘을 모두 “빠른 메모리”라고 부르면 중요한 차이가 사라진다. cache는 tag lookup과 replacement 정책으로 투명하게 재사용을 찾는다. shared는 명시적 copy와 address calculation, barrier를 요구하지만, 올바르게 동기화한 block 안에서는 kernel이 tile의 존재와 layout을 알고 재사용할 수 있다. cache hit를 기대하는 것과 shared tile을 소유하는 것은 다른 계약이다.

일부 NVIDIA 세대에서 L1/texture cache와 shared memory가 같은 unified data cache 용량의 carveout을 나눠 쓰는 사실도 혼동을 키운다. 물리 자원의 일부가 결합돼도 프로그래밍 의미가 같아지는 것은 아니다. `cudaFuncSetAttribute`로 preferred shared-memory carveout을 바꾸는 것은 L1 variable을 shared variable로 바꾸는 API가 아니다. 같은 SM 예산을 어떤 방식으로 나눌지 선택하는 것이다.

## 41.2 memory incident에서 byte 생애의 첫 불일치를 찾는다

관측은 attention kernel duration이 baseline 대비 35% 늘고 GPU memory usage가 높다는 것이다. 팀은 HBM 병목이라 판단해 shared tile을 두 배로 늘렸다. candidate는 duration이 다시 20% 악화됐고 process VRAM은 거의 같았다. “GPU memory” 하나로 allocation capacity와 traffic/staging을 섞었다.

첫 분기는 capacity와 data movement다. process allocated/reserved bytes는 model/KV arena 때문에 높지만 kernel step 동안 추가 allocation은 없다. OOM도 없다. duration 회귀는 runtime traffic, resource residency, synchronization 또는 compute다. VRAM 사용량을 HBM bandwidth 증거로 쓰지 않는다.

roofline 손계산은 baseline logical FLOPs와 HBM upper bytes가 같고 candidate도 global input/output bytes를 줄이지 않는다고 예측한다. shared double stage만 16KiB에서 32KiB로 늘었다. 목적은 copy-compute overlap이지 off-chip payload 감소가 아니다. “shared가 커져 HBM load가 절반”이라는 기대가 잘못됐다.

compile evidence에서 baseline dynamic shared32KiB/CTA, registers96/thread, candidate shared64KiB/CTA, registers104/thread라고 하자. occupancy model은 resident CTAs가 baseline2에서 candidate1로 줄 가능성을 보인다. actual limit은 architecture capacity/granularity와 other constraints로 확인한다.

Nsight evidence는 DRAM bytes와 throughput이 baseline/candidate 거의 같고 peak의 45% 수준이다. L2 hit도 비슷하다. candidate barrier stall과 not-eligible cycles가 늘고 achieved resident warps가 줄었다. local traffic은 소폭 증가하지만 dominant하지 않다. HBM saturation 가설은 반증된다.

source branch는 stage-count option이 shared storage union/array와 async copy pipeline을 바꾸고 wait distance를 옮기는 지점이다. candidate는 tail/short sequence에서도 double stage를 할당하지만 loop iterations가 적어 overlap 이득을 얻지 못한다. 늘어난 shared/register capacity만 지불한다.

first divergence는 HBM이 아니라 constructed specialization의 stage count→CTA resource allocation이다. runtime에서 낮은 occupancy/eligible warps와 barrier schedule로 이어진다. process GPU memory usage는 unrelated persistent allocation이고 symptom correlation일 뿐이다.

수정 후보는 stage count를 workload/tile에 맞춰 baseline으로 되돌리거나 long-enough K specialization에만 double stage를 선택하는 것이다. tile 자체를 줄여 shared/register를 낮추는 후보도 있다. 어느 하나도 보편 답이 아니며 source selector와 shape histogram으로 고른다.

verification은 same inputs에서 full/tail, short/long context, batch/decode shapes를 교차한다. compiler resources, selected specialization, occupancy, HBM/L2/shared/local bytes, barrier/eligible warps, duration과 output parity를 비교한다. 평균 throughput만 보지 않는다.

wrong-answer fixture도 붙인다. double-buffer stage overwrite와 wait-group off-by-one은 성능과 별개로 finite wrong output를 만들 수 있다. copy completion을 barrier에서 지연시키고 reader/writer buffer generation을 교차한다. global synchronization으로 숨기지 않고 exact dependency를 검증한다.

rollback은 new specialization selection을 admission/build generation에서 fence하고 inflight kernels를 drain한다. CUDA code/binary를 바꾸는 rollout이면 old/new workers를 cohort로 분리한다. already captured graphs가 old function/resource assumptions을 갖는지 확인하고 필요하면 recapture/restart한다.

service terminal은 request output/error, resource terminal은 kernels/events/graph/shared CTA lifetime, telemetry terminal은 new generation counters와 no stale launches다. kernel shared/register는 launch/CTA lifetime에 자동 회수되지만 higher-level workspace/graph modules는 별 owner가 있을 수 있다.

incident 문장은 “GPU memory가 느렸다”가 아니다. “stage-count 변경이 CTA shared32→64KiB와 registers96→104를 만들어 resident CTA를 2→1로 낮췄고, DRAM bytes는 불변인 반면 barrier/eligible-warp evidence가 처음 갈렸다”라고 쓴다. 수치는 실제 compile/profile artifact로 대체한다.

Prometheus GPU memory gauge는 fleet symptom correlation에 남길 수 있지만 kernel root cause 증거로 승격하지 않는다. Nsight/compile evidence를 production마다 상시 수집하라는 뜻도 아니다. controlled reproduction에서 first divergence를 확정하고 bounded runtime proxies를 운영한다.

runtime proxy는 selected specialization/stage count, shape cohort, kernel duration와 fallback/error counters다. request IDs는 trace에 둔다. occupancy나 low-level counters를 상시 노출할 수 없으면 periodic profiling artifact와 release gate를 유지한다.

rollback 성공은 duration baseline 복귀만 아니다. compiler/resource fingerprint가 expected이고 graphs/specializations가 old generation으로 정렬되며 output parity, no stale event와 request SLO를 통과해야 한다. shared를 줄여 HBM saturation이 새 병목이 되지 않았는지도 본다.

**load/store 수명을 instruction 단계로 더 잘게 나눈다.**

global A tile의 첫 owner는 device allocation과 tensor view다. CTA copy partition이 source addresses를 만든다. async copy가 requests를 발행하고 L2/HBM path에서 data를 가져와 shared stage에 쓴다. wait는 copy completion의 program-visible 전제이고 block/warp synchronization은 intended readers의 visibility/order를 닫는다.

shared A bytes는 MMA loader가 fragment로 읽을 때까지 살아야 한다. fragment가 registers에 들어간 뒤 해당 stage를 모든 reader warps가 다 읽었다면 shared slot을 다음 K tile이 overwrite할 수 있다. 한 warp의 read 완료만으로 CTA-shared stage를 재사용하지 않는다. producer/consumer warp specialization이면 barrier participant set을 명시한다.

register fragment는 MMA instruction consumer까지 짧게 살지만 compiler scheduling으로 live range가 늘 수 있다. C accumulator는 모든 K iterations와 online rescale/epilogue까지 산다. source scope가 끝났다는 사실보다 compiled live range가 resource pressure를 결정한다. SASS/live analysis 없이는 exact register lifetime을 확정하지 않는다.

output accumulator가 shared epilogue에 쓰이면 ownership이 registers→shared로 이동한다. layout conversion reader와 predicated global store가 끝난 뒤 shared epilogue storage를 재사용한다. tail predicate는 invalid output rows/columns의 global store를 막아야 한다. shared에 dummy 값이 있어도 global destination을 덮지 않는다.

global output store는 L2에 도달한 뒤 HBM writeback이 kernel completion과 정확히 같은 시각일 필요는 없다. CUDA kernel completion은 memory model의 다음 operation visibility 계약으로 읽고 physical writeback counter는 cache behavior로 본다. client가 읽는 D2H/next kernel dependency는 stream/event 계약을 따른다.

local spill은 thread-private address지만 L1/L2/device-memory path를 사용한다. spilled accumulator가 loop마다 load/store되면 logical C16KiB를 훨씬 넘는 traffic이 생길 수 있다. compiler spill bytes와 runtime local load/store sectors를 source stage에 대응시킨다.

constant parameters와 descriptors도 byte-flow에 있다. kernel arguments, shape/stride values가 constant/parameter space와 registers로 들어간다. payload에 비해 작아도 wrong stride는 모든 address를 틀리게 한다. performance byte count와 correctness identity를 분리한다.

**register pressure의 손익분기를 숫자로 본다.**

CTA threads256, baseline registers/thread80, candidate112라고 하자. register allocation은 20,480 대 28,672 registers/CTA다. 가상의 SM register capacity65,536에서 register만 보면 baseline3 CTAs, candidate2 CTAs가 상한이다. shared 또는 warp/thread limit이 더 낮을 수 있다.

baseline shared16KiB, candidate32KiB이고 가상의 SM shared capacity100KiB라면 shared 상한은 6 대 3 CTAs다. 결합하면 register 상한 3/2가 지배할 수 있다. architecture granularity와 max blocks/warps를 적용하면 실제 값이 더 낮아질 수 있다. 숫자는 계산 절차 예다.

candidate가 global intermediate CTA당 64KiB를 없애고 registers32/thread를 추가했다고 하자. 1,000 CTAs에서 logical64MiB traffic을 제거한다. 하지만 resident CTAs3→2로 줄어 latency hiding이 나빠지고 spills가 생기면 이득을 상쇄할 수 있다. removed bytes만으로 성공을 선언하지 않는다.

kernel duration baseline12µs, candidate10µs이면 candidate가 이긴다. short shape에서 baseline3µs, candidate4µs라면 selector가 shape별 specialization을 택할 수 있다. register-heavy path가 long work에서 amortize되는 break-even을 traffic histogram으로 찾는다.

compile resource fingerprint는 compiler/toolkit/arch flags와 함께 저장한다. CUDA 12.x→13.x upgrade가 register allocation/scheduling을 바꿀 수 있다. source diff가 없어도 binary resource와 performance가 달라질 수 있으므로 pinned build artifact를 비교한다.

**shared capacity와 barrier 비용을 계산한다.**

A/B stage each8KiB에서 single16KiB, double32KiB, triple48KiB다. epilogue shared8KiB가 lifetime alias되지 않으면 총 24/40/56KiB다. alias된다면 max(stage storage, epilogue)일 수 있지만 lifetime overlap과 barrier를 증명해야 한다.

SM shared capacity가 100KiB라면 raw capacity 상한은 single total24KiB로 4 CTAs, double40KiB로 2, triple56KiB로 1이다. allocation granularity와 per-block max를 공식 device property에 적용한다. stage가 늘어 async latency overlap이 좋아져도 concurrency가 계단식으로 줄어든다.

iteration copy latency가 300ns, compute200ns라면 single stage는 단순 500ns/iteration 상한, double overlap ideal max300ns다. 하지만 pipeline fill/drain, wait/barrier50ns와 low occupancy가 추가된다. K iterations2에서는 saved overlap보다 fill/drain 비용이 클 수 있고 iterations32에서는 이득이 커질 수 있다.

ideal model로 N iterations single `N(Copy+Compute)`, double `Copy+N·max(Copy,Compute)+Compute+(N-1)·BarrierOverhead`를 쓴다. N2면 1000ns 대 300+600+200+50=1150ns로 double이 느리다. N32면 16000ns 대 300+9600+200+1550=11650ns로 이득 후보다.

실제 async engine concurrency, memory dependency와 MMA scheduling은 더 복잡하다. 이 계산은 long/short specialization 가설을 만든다. Nsight에서 copy/barrier/eligible warp와 duration을 N cohort별로 검증한다.

**L2 reuse와 HBM traffic 상하한을 계산한다.**

CTAs1,000이 동일 B tile8KiB를 읽고 각자 다른 A8KiB와 C8KiB를 처리한다고 하자. reuse가 전혀 없으면 HBM logical reads16MiB+writes8MiB=24MiB다. B가 첫 load 뒤 L2에 완전히 유지되면 HBM B는 8KiB뿐이고 total 약 16.008MiB다.

L2-to-SM traffic은 각 CTA가 B를 읽으므로 B8MiB가 남는다. HBM 감소와 SM demand 감소를 혼동하지 않는다. shared staging은 CTA 내부 B reuse를 줄이지만 CTAs 간 reuse는 L2가 맡을 수 있다. cache working set/eviction과 scheduling order가 실제 hit를 정한다.

B tiles100개가 CTAs에 불규칙하게 배정되고 combined working set이 L2를 넘으면 lower bound를 달성하지 못한다. hit rate와 absolute B requests/miss bytes를 본다. persistence hint가 있다면 access policy window, reservation과 다른 workload interference를 source/config에서 확인한다.

decode attention은 여러 query heads가 same KV heads를 공유하는 GQA reuse가 있다. kernel이 head groups를 same CTA/cluster에서 처리하면 shared/register reuse, separate CTAs면 L2 reuse 가능성이 있다. global logical load upper는 query-head별, lower는 KV-head별 한 번이다. source grid/tile mapping으로 범위를 좁힌다.

observed HBM이 lower bound보다 작으면 compression/counter scope/other caching assumptions를 재검토한다. upper보다 크면 repeated passes, spills, scales/metadata, tail, eviction/reloads 또는 unrelated traffic을 본다. bounds가 falsifier 역할을 한다.

**Nsight counter에서 first divergence를 고르는 순서.**

첫째, workload와 kernel specialization을 고정한다. 같은 kernel 이름이어도 template/tile/stage/dtype와 shape가 다를 수 있다. launch grid, dynamic shared, registers와 binary hash를 저장한다. aggregate kernel-name report만 비교하지 않는다.

둘째, duration과 work를 고정한다. CTAs, valid output tokens/elements와 FLOPs를 맞춘다. candidate가 더 많은 work를 했으면 per-work normalization을 한다. tail/dummy rows를 포함한 submitted work와 valid work를 분리한다.

셋째, HBM/L2 absolute bytes와 roofline bounds를 비교한다. expected global payload가 같고 DRAM bytes가 같으면 HBM traffic reduction 가설을 지운다. bytes가 늘면 source에서 repeated passes/spill/loads를 찾는다. percentage utilization만 보지 않는다.

넷째, compile resources/local traffic을 본다. registers/shared 증가와 occupancy 계단, spill local sectors가 first difference인지 확인한다. source variable 수가 아니라 actual build artifact를 사용한다.

다섯째, shared/barrier/eligible warps와 instruction mix를 본다. bytes가 낮아도 issue가 부족하면 bandwidth를 못 채운다. barrier wait가 늘면 stage protocol과 participant/iteration cohort를 source에 맞춘다.

여섯째, output parity와 synchronization assertions를 본다. faster candidate가 occasional wrong answer면 performance 승격에서 제외한다. copy/write generation과 wait/barrier fixture를 통과해야 한다.

마지막에 source branch를 고정한다. stage selector, tile sizes, epilogue fusion 또는 fallback이 어떤 resource/traffic 변화를 만들었는지 한 문장으로 쓴다. counter correlation만 있고 mutation chain이 없으면 원인 확정이 아니다.

**CUDA 12.x와 13.x를 비교할 때 지키는 경계.**

Programming Guide/tuning guide의 version을 build toolkit, driver, target compute capability와 함께 기록한다. 문서 버전이 다르다고 하드웨어 자원 크기가 자동으로 달라지지 않는다. compiler code generation, library specialization과 지원 feature 변화는 별 diff다.

동일 source를 CUDA 12.x/13.x로 빌드하면 registers/thread, spill, shared, SASS instructions와 graph/kernel behavior를 artifact로 비교한다. exact compiler flags와 architecture target을 고정한다. 성능 차이를 guide 문구 변화로 설명하지 않는다.

새 architecture의 TMA/cluster 기능은 capability predicate를 통과한 source branch에서만 적용한다. Ampere async copy, Hopper TMA와 Blackwell cluster scope를 하나의 일반 “async shared copy” 성능 수치로 합치지 않는다. producer/scope/completion contract가 다르다.

mixed fleet에서는 selected binary/specialization을 runtime trace에 남긴다. unsupported path가 fallback했는데 option enabled만 보고 같은 execution이라고 가정하지 않는다. release gate는 architecture cohort별 resource/traffic/parity fixture를 갖는다.

rollback은 source config뿐 아니라 binary/container/toolkit generation을 되돌린다. compiled graph/module cache와 JIT artifacts가 남으면 old/new code를 혼용할 수 있다. workers를 drain하고 artifact fingerprint/self-test 뒤 readiness를 연다.

**attention tile에 같은 계산을 적용한다.**

head dimension128, query rows64, key rows64, FP16 Q/K/V와 FP32 output accumulator를 둔다. Q payload64×128×2=16KiB, K16KiB, V16KiB다. QKᵀ FLOPs는 `2×64×64×128=1,048,576`, PV도 동일해 main MMA 합 약 2.10MFLOPs다. softmax 산술은 별도다.

global logical input은 Q/K/V48KiB, output64×128×2=16KiB로 64KiB다. HBM upper intensity는 약 32 FLOP/B다. score matrix64×64 FP32를 global에 materialize하면 write16KiB+read16KiB가 추가돼 denominator96KiB, intensity 약 21.3으로 떨어진다. online softmax가 피하는 traffic의 첫 상한이다.

online softmax는 공짜가 아니다. row max/sum, rescale과 partial output accumulator가 registers/shared에서 산다. output accumulator64×128×4=32KiB logical per CTA다. threads256 균등이면 thread당 128B, FP32 registers32개가 output만으로 필요하다. QK score fragments와 addresses를 더하면 register pressure가 커진다.

Q가 K/V loop 전체에 재사용되면 shared/register lifetime이 길다. K/V는 stages마다 교체된다. Q와 K shared storage를 alias하는 source 경로라면 Q fragment가 모든 향후 consumer에 안전하게 보존된 뒤 K가 overwrite해야 한다. barrier 위치와 register live range trade-off를 본다.

decode query rows1이면 Q256B, one KV tile K/V32KiB, output256B다. FLOPs는 `2×1×64×128×2≈32,768`, bytes 약 32.5KiB, intensity 약 1 FLOP/B다. decode가 bandwidth-sensitive하기 쉬운 직관이지만 L2 reuse, GQA sharing과 page layout이 실제 HBM bytes를 바꾼다.

query heads32, KV heads8이면 four query heads가 KV를 공유한다. four heads를 함께 처리해 K/V32KiB를 한 번 읽으면 lower traffic이고 separate CTAs가 각각 읽으면 logical upper128KiB다. L2가 잡으면 HBM은 lower에 가까워도 L2-to-SM requests는 upper일 수 있다. grid mapping과 counters를 연결한다.

page table/scale metadata와 non-contiguous KV pages도 추가한다. logical K/V payload가 같아도 address loads, uncoalesced sectors와 page boundary tail이 bytes/latency를 늘린다. 다음 장의 transaction 분석으로 넘기되 이 장에서는 payload upper/lower와 ownership을 기록한다.

**source walk를 reviewer가 재현하는 절차.**

첫 카드는 launcher/selector다. shape, dtype, head dimension, causal/window, architecture와 options가 어느 template/tile/stage specialization을 고르는지 찾는다. fallback branch와 dynamic shared bytes 설정을 기록한다. option requested와 selected kernel을 구분한다.

둘째 카드는 global tensor views와 copy partition이다. Q/K/V base, strides, valid lengths와 thread coordinates가 source addresses를 만드는 span을 고정한다. logical tile bounds와 tail predicates를 적는다. address space fact와 HBM traffic inference를 구분한다.

셋째 카드는 shared layout/allocation이다. static/dynamic arrays, alias union, stage count, padding/skew와 bytes를 계산한다. CTA/cluster scope와 allocation lifetime을 공식 guide 계약에 맞춘다. producer copy와 reader warp set을 적는다.

넷째 카드는 async copy/wait/barrier다. copies issued, commit/wait group, barrier participants와 stage reuse를 잇는다. early return/divergent tail이 protocol을 건너뛰지 않는지 본다. Hopper TMA면 transaction/barrier contract를 해당 capability source로 읽는다.

다섯째 카드는 fragment/MMA/softmax다. shared loads, register fragments와 long-lived accumulators를 표시한다. online state가 loop마다 어떻게 update/rescale되는지, optional dropout/logprob state가 live range를 늘리는지 본다. source로 exact registers를 단정하지 않는다.

여섯째 카드는 epilogue/store다. accumulator conversion/layout, optional shared staging, predicated global stores와 last consumer를 잇는다. output write 뒤 next kernel/transport dependency도 serving trace에 연결한다.

일곱째 카드는 build/compiler/runtime evidence다. pinned commit과 CMake tag, toolkit/arch flags, registers/shared/local report, binary hash, workload shape와 Nsight counter definition을 source card에 붙인다. source line이 같아도 binary artifact가 다르면 별 cohort다.

**실패 주입과 correctness terminal.**

double buffer fixture는 stage0 copy 완료를 지연하고 stage1 producer/reader 순서를 제어한다. reader가 matching wait 없이 stage를 소비하면 sentinel mismatch가 나야 한다. correct kernel은 event/barrier까지 기다린다. timeout 자체보다 wrong stage generation 접근을 검증한다.

tail fixture는 M/N/K tile-1, tile, tile+1을 사용한다. invalid global load와 store가 predicated되고 shared dummy가 deterministic하며 valid output parity가 맞아야 한다. barrier participants가 tail branch 때문에 줄어 deadlock하지 않는지 본다.

spill fixture는 epilogue fusion/tile을 바꿔 compiler resources가 경계를 넘게 한다. local load/store 증가와 output parity, duration을 비교한다. spill은 correctness error가 아닐 수 있지만 stack/local bounds나 uninitialized branch를 함께 검증한다.

graph fixture는 captured dynamic shared/function specialization과 replay inputs가 호환되는지 본다. binary/toolkit rollback 뒤 old graph executable이 stale kernel을 launch하지 않게 generation을 fence하고 recapture한다. launch error와 silent old path를 모두 관측한다.

service fixture는 concurrent requests와 slow consumer/cancel에서 kernel completion, output ownership과 workspace release를 확인한다. CTA register/shared는 hardware lifetime에 닫혀도 persistent workspace/event/graph pool은 runtime owner가 닫아야 한다.

correctness terminal은 layer/kernel output parity, no NaN, stage generation assertions와 tail sentinels다. resource terminal은 no pending launches/events/workspace owners, telemetry terminal은 selected artifact/resource fingerprint와 error counters다. duration 회복만으로 닫지 않는다.

**최종 배포 worksheet.**

행 1은 pinned CUDA official guide/toolkit/driver/compute capability와 source/binary hash다. 행 2는 workload shape와 selected kernel/tile/stage다. 행 3은 logical FLOPs, global input/output, avoided intermediates와 intensity bounds다.

행 4는 compiler registers/thread, local spill, static/dynamic shared와 occupancy limiting resource다. 행 5는 observed DRAM/L2/shared/local bytes, duration, achieved FLOPs/bandwidth와 stalls다. counter names/definitions와 profiling overhead를 붙인다.

행 6은 source mutation chain이다. option/selector→layout/copy stages→resource allocation→wait/MMA/epilogue를 잇는다. 행 7은 parity/tail/race/graph fixtures와 service/resource terminals다. 행 8은 rollout/rollback generations와 readiness다.

baseline/candidate가 same valid work인지 먼저 확인한다. candidate가 dummy/tail work를 늘렸다면 per-valid-output와 per-submitted-work 두 normalized 수치를 둔다. traffic distribution으로 weighted SLO/goodput을 평가한다.

승격 문장은 “GPU memory 최적화”가 아니다. “long-K specialization이 stage2로 copy-compute overlap을 늘렸고 shared/register resource와 resident warps가 허용 범위이며 HBM/L2 bytes, barrier stalls, duration과 parity fixture가 예상 모델에 맞았다”처럼 쓴다.

rollback trigger는 wrong output, stage/tag assertion, spill/local traffic 폭증, occupancy/barrier tail 회귀, HBM/L2 bound 위반, stale binary/graph와 SLO 악화다. trigger 발생 시 new artifact admission을 fence하고 inflight kernels/graphs를 drain한 뒤 verified binary를 로드한다.

운영에서는 모든 Nsight counter를 상시 수집하지 않는다. release/profile artifact와 bounded runtime kernel duration·selected specialization·error proxies를 연결한다. incident 시 same binary/workload reproduction으로 low-level evidence를 다시 얻는다.

독자가 작은 64×64 kernel과 attention tile의 bytes/FLOPs를 다시 계산하고 compiler/counter 표의 각 차이를 source branch에 붙일 수 있다면, “GPU memory”는 더 이상 하나의 원인이 아니다. allocation capacity, HBM/L2 traffic, shared staging, register pressure와 synchronization이 각각 검증 가능한 후보가 된다.

**사건을 반증하는 다섯 질문.**

첫째, HBM 병목이라면 동일 work에서 observed DRAM bandwidth가 sustained roof에 가깝고 duration 변화가 DRAM bytes 또는 bandwidth 변화와 맞는가. bytes 불변, bandwidth 낮음, barrier stalls 증가라면 HBM saturation 설명은 약하다. peak percentage 하나 대신 absolute bytes/time을 쓴다.

둘째, L2 reuse가 원인이라면 request bytes와 hit/miss bytes가 shape/working-set 변화와 맞는가. hit rate가 높아도 total requests가 늘어 miss bytes가 증가할 수 있다. persistence/cache policy option이 실제 source branch와 capability에서 선택됐는지 본다.

셋째, register pressure라면 compile registers/local spill과 runtime local sectors, occupancy/eligible warps가 candidate에서 함께 변하는가. registers만 늘고 다른 limiting resource가 이미 occupancy를 제한했다면 register 증가가 causal하지 않을 수 있다. occupancy calculator와 actual launch를 맞춘다.

넷째, shared/barrier라면 dynamic shared/stage count, resident CTAs와 barrier/wait stalls가 source mutation 이후 갈리는가. shared bytes 증가가 있어도 long-K에서 overlap 이득이 더 크면 duration은 좋아질 수 있다. shape cohort별로 본다.

다섯째, compute dependency라면 memory bytes/stalls가 설명하지 못하고 tensor/core instruction utilization, issue dependency와 critical instruction chain이 갈리는가. fusion이 FLOPs/instructions를 추가했는지도 work model에 넣는다. “memory가 아니다”에서 곧바로 “compute-bound”로 점프하지 않는다.

**숫자 단위와 counter scope를 검산한다.**

FLOP는 FMA를 2로 셌는지, tensor operation의 effective FLOPs를 어떻게 정의했는지 밝힌다. bytes는 B/KiB/MiB 변환, read/write 합과 compression 포함 여부를 밝힌다. duration은 kernel active time인지 application interval인지 구분한다.

DRAM counter가 device 전체 또는 profiled kernel 범위인지, concurrent kernels/copies가 포함되는지 확인한다. isolated profiling과 production concurrency 결과를 같은 분모로 비교하지 않는다. replay/profiling이 cache warmness를 바꿀 수도 있다.

L2 hit percentage의 numerator/denominator, sector byte width와 request 종류를 metric definition에서 확인한다. toolkit version이 바뀌면 metric 이름/availability와 계산식을 다시 고정한다. 예전 대시보드 식을 새 architecture에 그대로 적용하지 않는다.

shared bank conflict/replay counter도 instruction 종류와 architecture 의미를 확인한다. high counter가 critical path인지 duration/warp stalls와 맞춘다. source layout이 bank conflict를 피하려 padding/skew를 추가했다면 allocation bytes 증가와 conflict 감소를 함께 본다.

compiler `local memory` report와 runtime local traffic은 동일하지 않을 수 있다. report는 per-thread static frame/spill 후보, runtime은 executed path와 cache behavior를 반영한다. 둘을 source live range와 함께 읽는다.

**운영 rollback을 실제 순서로 적는다.**

새 kernel artifact가 suspect하면 routing/admission에서 new generation을 차단한다. 이미 submit된 launches와 dependent output copies를 drain한다. request를 곧바로 다른 worker로 retry할 수 있는지는 output commit/idempotency를 확인한다.

captured CUDA Graph가 new kernel function과 dynamic shared configuration을 참조하면 graph executables를 폐기한다. module unload 전에 outstanding work/events가 끝났는지 확인한다. JIT/cache directory의 stale artifact가 다시 선택되지 않게 build fingerprint를 검증한다.

verified baseline binary/container를 로드하고 startup self-test를 실행한다. 64×64/tail, stage race와 attention parity, resource fingerprint를 확인한다. toolkit/driver compatibility와 selected compute capability도 기록한다.

readiness 뒤 작은 canary에서 kernel duration, selected specialization, errors와 request SLO를 확인한다. process GPU memory gauge가 baseline과 다르더라도 owner inventory가 설명하면 무조건 실패로 보지 않는다. 반대로 gauge가 같아도 low-level regression이 남을 수 있다.

old worker/resource terminal은 inflight requests, streams/events, graph/workspaces와 telemetry scrape 종료를 포함한다. process kill로 자원이 사라졌다는 사실과 root cause 수정 증거를 구분한다. incident artifact를 보존한다.

**공식 문서와 실제 구현을 연결하는 최종 원칙.**

Programming Guide의 memory hierarchy와 synchronization은 가능한 주소 공간/scope 계약이다. tuning guide의 async copy/TMA/cluster는 해당 architecture feature 계약이다. 이 문서는 특정 FlashAttention specialization이 그 기능을 선택했다는 증거가 아니다.

구현 source는 actual pointer/layout/copy/wait/MMA/store와 selector predicate를 보여 준다. source만으로 compiler registers, L2 hit와 HBM bandwidth를 확정하지 않는다. build artifact와 runtime counter가 각각 빈칸을 채운다.

손계산은 공식 계약과 source path 사이에서 expected bounds를 만든다. observed 값이 bounds 밖이면 단위를 확인하고 누락된 traffic/path 또는 counter scope를 찾는다. model과 관측을 억지로 맞추는 fudge factor를 넣지 않는다.

최종 incident dossier는 official section/version, source commit/span, binary/toolkit/arch, workload, byte/FLOP worksheet, counters, first divergence, falsifiers, fix fixtures와 rollback terminal을 갖는다. 다른 독자가 같은 artifact로 결론을 반복할 수 있어야 한다.

이 원칙을 지키면 CUDA 12.x/13.x, Ampere/Hopper/Blackwell 또는 다른 kernel에서도 분석법이 유지된다. 수치와 feature branch는 다시 고정하지만 byte producer, storage scope, last consumer, synchronization과 관측 분모라는 질문은 그대로다.

**최종 한 페이지 review.**

object마다 allocation과 lifetime을 쓴다. device Q/K/V allocation은 request/model input lifetime, L2/L1 line은 hardware replacement lifetime, shared stage는 CTA lifetime과 stage reuse barrier, register fragment/accumulator는 compiled live range, output allocation은 downstream consumer lifetime이다. 모두 “GPU memory”라는 한 owner가 아니다.

64×64 GEMM row에는 FLOPs524,288, global logical load16KiB/store8KiB, shared stage16KiB, accumulator16KiB를 쓴다. attention row에는 Q/K/V48KiB, output16KiB, avoided score intermediate32KiB와 accumulator32KiB를 쓴다. decode row에는 GQA KV lower/upper bounds를 쓴다.

compile row에는 registers/thread, spills/local, static/dynamic shared, threads/CTA와 limiting resource를 쓴다. runtime row에는 selected specialization, grid, valid/submitted work, DRAM/L2/shared/local bytes, duration, achieved FLOPs/bandwidth, barrier/eligible warp를 쓴다.

증거 row에는 official guide section/version, pinned source spans, binary/toolkit/compute capability와 profiler metric definitions를 쓴다. source 사실을 runtime 수치로, profiler correlation을 source mutation으로 바꾸어 쓰지 않는다.

incident row에는 observation, candidate branches, first divergence와 falsified hypotheses를 쓴다. 앞 사건은 process VRAM이 아니라 stage-count가 shared/register allocation과 residency를 바꾼 지점이었다. DRAM bytes 불변과 barrier/eligible-warp 변화가 HBM 가설을 반증했다.

fixture row에는 full/tail, short/long K, stage1/2, register/spill boundary, async-delay, graph replay와 output parity를 쓴다. correctness gate를 성능 평균 뒤에 두지 않는다. same valid work와 binary generation을 확인한다.

rollback row에는 admission fence, inflight drain, graph/module/JIT artifact 폐기, baseline load, self-test, canary와 old resource terminal을 쓴다. process restart만으로 원인이 고쳐졌다고 선언하지 않는다.

승격 row에는 workload cohort별 SLO/goodput, resource/counter bounds와 parity를 쓴다. long shape 개선이 short/tail을 악화하면 selector 분리 또는 rollback 기준을 적용한다. 평균 하나로 덮지 않는다.

마지막으로 미검증 조건을 남긴다. 실행하지 않은 source 분석은 byte-flow와 예상 bounds까지만 확정한다. actual compiler allocation, cache reuse와 stall은 pinned build/profile이 필요하다. 각 빈칸에 fixture와 중단 조건을 붙인다.

이 한 페이지를 다른 engineer가 다시 계산할 수 있으면 장의 목적이 달성된다. kernel이 느리다는 신고를 받았을 때 장소 이름부터 고르지 않고 byte의 생산·이동·재사용·마지막 소비와 증거 층을 따라 first divergence를 찾을 수 있다.

배포 후에는 selected specialization과 binary fingerprint, shape별 duration·fallback·error를 bounded telemetry로 유지한다. 정기 controlled profile에서 logical bounds와 DRAM/L2/shared/local counters, compiler resources를 다시 대조한다. toolkit·driver·GPU cohort가 바뀌면 이전 counter 수치를 그대로 이식하지 않는다.

회귀가 감지되면 request 영향과 kernel artifact generation을 먼저 격리한다. 정확성 이상은 즉시 traffic을 차단하고, 성능 회귀는 SLO 기준으로 canary를 중단한다. inflight launches와 graph/workspace를 닫고 검증된 artifact로 돌아간 뒤 동일 worksheet와 fixtures로 복구를 증명한다.

최종 기록에는 raw units, denominator, counter scope와 시간 창을 남긴다. “bandwidth가 높다”, “register가 많다”, “shared가 크다”는 상대 문장만 쓰지 않는다. 어떤 work와 architecture에서 어느 resource가 처음 limit이 됐는지 재현 가능한 수치로 쓴다.

그 수치가 source mutation과 rollback terminal까지 재현가능하게 이어질 때만 incident를 완전히 닫는다.
## 41.3 집과 창고 비유는 어디까지 유효한가

### 41.3.1 유용한 첫 직관

처음에는 register를 작업자의 손, shared memory를 같은 작업반의 작업대, L2를 공장 공용 창고, HBM을 외부 대형 창고라고 생각할 수 있다. 손에 든 부품은 바로 연산할 수 있지만 손의 수가 제한된다. 작업대에 올린 부품은 같은 block의 thread가 함께 쓰지만 작업대 면적이 제한된다. 공용 창고는 여러 작업반의 요청을 흡수하지만 어느 부품이 계속 남을지 작업반이 완전히 결정하지 않는다. 대형 창고는 크지만 왕복해야 할 byte가 많으면 운송 대역폭이 병목이 된다.

이 비유는 scope와 capacity를 묻는 습관을 만든다는 점에서 쓸모가 있다. 누가 볼 수 있는지, 얼마를 둘 수 있는지, 재사용하려면 누가 옮겨야 하는지를 떠올리게 한다. FlashAttention이 Q/K/V를 shared tile로 가져오는 이유도 “HBM보다 빠르다” 한 문장보다, 여러 warp가 같은 tile을 MMA operand로 반복 소비하도록 block이 명시적으로 소유한다는 설명이 더 정확하다.

### 41.3.2 비유가 깨지는 지점

그러나 register는 개별 thread가 주소로 자유롭게 전달하는 작은 RAM이 아니다. compiler allocation과 instruction operand 규칙에 묶이고, MMA fragment는 lane마다 분산된 특별한 layout을 가진다. shared 작업대도 모든 작업자가 같은 순간 아무 위치나 안전하게 읽는 공간이 아니다. bank mapping, barrier, memory ordering, producer-consumer protocol이 필요하다. 이 장에서는 bank와 transaction의 세부를 다음 장으로 미루지만 synchronization의 필요는 지금 포함한다.

L2를 공용 창고라고 부르면 software가 선반을 예약할 수 있다고 오해하기 쉽다. L2 persistence control은 set-aside와 access policy window를 통해 재사용 가능성을 조절하지만 무조건적인 residence guarantee로 쓰면 안 된다. working set이 예산보다 크거나 다른 stream과 tenant가 경쟁하면 eviction이 일어날 수 있다. HBM도 단순히 멀리 떨어진 한 창고가 아니다. 여러 memory partition과 controller, channel로 구성되고 address mapping에 따라 병렬성이 달라진다. 그 상세 역시 이 장의 범위를 넘는다.

가장 큰 한계는 byte가 반드시 `HBM → L2 → shared → register`라는 계단을 한 번씩 오른다고 생각하게 만든다는 점이다. source 수준에서는 global-to-shared async copy가 register staging을 피할 수 있고 L1을 bypass할 수 있다. load가 L2 hit이면 HBM까지 가지 않는다. output은 accumulator register에서 shared로 재배열된 뒤 thread register를 거쳐 global store될 수 있다. 같은 shared buffer가 Q와 output에 시간 분할로 재사용될 수도 있다. 실제 경로는 조건과 lifetime에 따라 갈라진다.

그러므로 비유를 사용한 다음에는 반드시 다섯 질문으로 돌아온다. 주소 공간은 무엇인가. 물리 backing을 무엇까지 알고 있는가. 공유 범위는 어디까지인가. lifetime은 어느 synchronization 지점에서 끝나는가. 이 선택이 소비하는 유한 자원은 무엇인가. 이 질문에 답하지 못하면 비유는 설명이 아니라 장식이다.

## 41.4 register→shared→L2→HBM 네 byte 예산을 순서대로 읽는다

### 41.4.1 register 예산: thread-private라고 공짜가 아니다

register file은 SM의 유한 자원이다. NVIDIA tuning guide가 SM당 64K개의 32-bit register와 thread당 최대 255 register 같은 한도를 제시할 때, 두 숫자는 서로 다른 실패와 성능 경계를 만든다. thread 한 명의 요구량이 architecture limit를 넘을 수 없고, block의 thread 수와 thread당 allocation을 곱한 값은 그 block이 SM에서 차지할 register 예산을 만든다. 동시에 여러 block을 resident하게 하려면 그 합이 SM 예산 안에 들어야 한다.

교육용으로 CTA당 256 thread, compiler allocation이 thread당 128개의 32-bit register라고 가정하자. 단순 곱은 `256 × 128 = 32,768 registers/CTA`다. SM의 register file이 65,536개라면 register만 고려한 이론적 상한은 두 CTA다. thread당 160개라면 `40,960 registers/CTA`이므로 두 CTA는 `81,920`개가 되어 맞지 않고 register 제약만으로 한 CTA가 상한이 된다. 실제 allocation granularity, 다른 자원, architecture scheduling limit 때문에 이 계산이 곧 실제 occupancy는 아니지만, 어느 방향으로 경계가 움직이는지 보여 준다.

attention에서는 score accumulator와 output accumulator가 크다. Q fragment를 loop 내내 register에 유지하면 global/shared 재독을 줄일 수 있지만 live range가 길어진다. unroll을 늘리면 여러 iteration의 operand와 address state가 겹쳐 살아 있을 수 있다. `acc_s`를 FP32로 유지하고 `acc_o`도 FP32로 누적하면 수치 안정성과 tensor-core 사용에 유리한 면이 있지만 register footprint를 소비한다. 그래서 “register를 쓰면 빠르다”가 아니라 “어떤 재사용을 위해 어느 값을 얼마 동안 register에 유지하며, 그 대가로 몇 warp의 residency 가능성을 잃는가”라고 질문해야 한다.

spill은 이 trade-off의 극단이다. compiler가 모든 live value를 register에 둘 수 없으면 일부를 local memory에 저장했다가 다시 읽을 수 있다. 이때 source의 thread-private 값이 device memory traffic 후보로 변한다. 하지만 source만 보고 spill이 발생했다고 쓰면 안 된다. 확인 순서는 compile log의 register와 spill bytes, generated code의 local load/store, profiler의 local-memory counters다. 이 장에는 관측 항목을 제시하되 결과를 만들지 않는다.

### 41.4.2 shared 예산: tile의 크기와 동시성이 맞교환된다

shared memory allocation은 block 단위다. thread당 register 계산과 달리 CTA 하나가 요청한 dynamic/static shared bytes가 block 전체의 resident 비용이 된다. 예를 들어 한 SM에서 사용할 수 있는 shared budget을 교육용으로 164 KiB라고 하고 CTA가 80 KiB를 요청하면 shared 기준으로 두 CTA가 `160 KiB`를 사용해 들어갈 수 있다. 84 KiB로 키우면 두 CTA는 `168 KiB`라 들어가지 않아 한 CTA가 상한이 된다. 단 4 KiB 증가가 resident CTA 상한을 절반으로 바꿀 수 있는 경계다.

그렇다고 80 KiB가 84 KiB보다 무조건 빠른 것은 아니다. 큰 tile이 K/V global load를 더 많이 재사용하고 loop iteration이나 barrier 수를 줄일 수 있다. CTA 하나만 resident해도 tensor-core pipeline을 충분히 채울 수 있는 kernel도 있다. 반대로 tail이 많은 decode shape에서는 큰 tile의 상당 부분이 predicate로 비어 shared와 register를 낭비할 수 있다. 따라서 tile 결정은 reuse gain, tail waste, register footprint, shared footprint, synchronization 횟수를 함께 본다.

shared buffer aliasing은 capacity를 줄이는 강력한 기법이다. FlashAttention 소스는 `Share_Q_K_smem` 조건에서 Q와 K가 같은 shared base를 쓰도록 만들고, epilogue에서는 Q가 있던 영역을 output `sO`로 재사용한다. 이 최적화의 핵심은 pointer arithmetic이 아니라 lifetime proof다. 이전 소비자가 완전히 끝났고 다음 producer가 덮어써도 된다는 사실이 barrier와 control flow로 보장되어야 한다. lifetime이 겹치는데 alias하면 속도가 느려지는 것이 아니라 정답이 틀린다.

### 41.4.3 L2 예산: 소유하지 않는 재사용

L2는 여러 SM이 발생시키는 global traffic과 device memory 사이에서 재사용을 흡수한다. model weight처럼 여러 CTA가 반복 읽는 데이터, KV cache의 반복 영역, metadata, output writeback이 경쟁한다. prefix가 같다고 해서 같은 KV byte가 항상 L2에 남는 것은 아니다. working set, 접근 순서, 다른 kernel과 stream의 traffic, set contention, persistence policy가 모두 영향을 준다.

capacity만 보고 hit를 예측해도 안 된다. 50 MB L2에 40 MB working set이 있으니 모두 hit한다고 결론 내릴 수 없다. 동시에 접근하는 다른 데이터가 있고 mapping과 replacement가 있으며, 시간 간격이 길면 사이에 많은 traffic이 들어온다. 반대로 working set이 L2보다 커도 특정 hot subset은 반복 hit할 수 있다. 이 장에서는 정확한 line과 sector 산정을 하지 않고, reuse distance와 competing bytes를 질문하는 수준으로 제한한다.

L2 persistence control도 용어 그대로 읽어야 한다. access policy window와 set-aside는 특정 access의 persisting 가능성을 높이는 제어다. cache를 software-managed shared처럼 바꾸지 않는다. 여러 stream에 persistence window를 과도하게 주면 합계 working set이 set-aside capacity를 넘어 서로 밀어낼 수 있다. serving에서 여러 model instance나 tenant가 한 GPU를 공유하면 단일 kernel의 의도와 장치 전체 pressure가 달라질 수 있다.

### 41.4.4 device-memory bandwidth 예산: byte 수부터 계산한다

memory-bound 여부를 말하기 전에 algorithmic byte를 계산한다. decode attention 한 query head가 KV length `L`, head dimension `D`, element size `b` byte인 K와 V를 한 번씩 읽는 단순 하한은 `2 × L × D × b` byte다. `L=8192`, `D=128`, FP16/BF16의 `b=2`라면 한 KV head당 `2 × 8192 × 128 × 2 = 4,194,304 byte`, 약 4 MiB다. 이는 metadata, page-table lookup, alignment waste, cache hit, repeated load, output traffic을 제외한 logical tensor byte다.

GQA에서 여러 query heads가 같은 KV head를 공유하면 구현이 K/V를 cache/shared에서 재사용하는 정도에 따라 physical traffic이 달라진다. query head 수를 단순히 곱하면 상한에 가까운 모델이 될 수 있고, KV head 수만 곱하면 완전한 재사용을 가정한 하한이 될 수 있다. 두 경계를 함께 적고 실제 counter가 어느 쪽에 가까운지 나중에 본다. source에서 CTA가 head를 어떻게 묶는지 확인하지 않은 채 한 숫자를 정답으로 쓰지 않는다.

prefill에서는 Q rows가 많아 QKᵀ와 PV의 연산량이 커지고 tile 내 reuse가 풍부하다. decode에서는 Q row가 적고 긴 KV를 읽으므로 arithmetic intensity가 낮아지기 쉽다. 그렇지만 “decode는 언제나 HBM-bound”라고 고정하지 않는다. 짧은 context, 작은 batch, launch overhead, poor occupancy, page indirection, synchronization이 지배할 수 있다. 병목은 workload와 kernel specialization의 결합이다.

## 41.5 Ampere, Hopper, Blackwell을 한 문장에 섞지 않는다

### 41.5.1 Ampere의 async global-to-shared copy

Ampere tuning guide는 global memory에서 shared memory로의 hardware-accelerated asynchronous copy를 명시한다. 이 경로는 계산과 데이터 이동을 겹치도록 pipeline을 구성할 수 있고, 단순 load-to-register 후 store-to-shared 경로에서 복사용 중간 register를 피할 수 있으며, 조건에 따라 L1을 bypass할 수 있다. FlashAttention의 `cp_async_fence`와 wait를 읽을 때 이 배경이 필요하다.

“비동기”는 기다리지 않아도 된다는 뜻이 아니다. producer가 copy를 발행한 뒤 consumer가 해당 shared tile을 읽기 전에 completion을 확인해야 한다. 여러 thread가 tile을 채우고 다른 thread들이 소비한다면 block synchronization도 필요하다. fence는 copy group의 ordering/commit에 관련되고 wait는 완료된 group 수를 관리하며 `__syncthreads()`는 block thread들의 도달과 shared visibility를 맞춘다. 각각을 하나의 “동기화”로 뭉개면 race를 찾기 어렵다.

Ampere라는 이름 안에서도 CC 8.0과 8.6의 shared capacity가 다르다. 가이드 기준 A100/CC 8.0은 164 KB per SM과 163 KB per block을 제시하고, CC 8.6은 100 KB per SM과 99 KB per block을 제시한다. 같은 kernel의 dynamic shared request가 한 제품에서는 두 CTA residency 경계 안이고 다른 제품에서는 launch opt-in이나 한 CTA 경계에 걸릴 수 있다. 따라서 “Ampere에서 된다”는 호환성과 “Ampere에서 같은 occupancy다”는 성능 예측을 구분한다.

### 41.5.2 Hopper의 TMA와 cluster 범위

Hopper는 Ampere의 async copy 개념 위에 Tensor Memory Accelerator를 제공한다. tuning guide에 따르면 TMA는 1D부터 5D tensor를 global과 shared 사이에서 양방향으로 옮기고, 같은 cluster에 속한 SM들의 shared 영역 사이 이동도 지원한다. 이동 자체에 register를 사용하지 않는 특징은 producer warp가 address와 element copy를 반복 수행하는 부담을 줄일 수 있다.

하지만 TMA는 “더 빠른 memcpy”라는 한 문장으로 끝나지 않는다. tensor descriptor가 shape와 stride를 기술하고, producer-consumer barrier가 도착을 알리며, tile pipeline의 stage 수와 shared allocation이 정해져야 한다. transfer setup과 synchronization, shared capacity의 비용이 있다. source walk에서 Ampere 계열 `cp.async` kernel을 본 뒤 Hopper kernel도 똑같이 register fragment를 거친다고 일반화하면 TMA의 의도를 놓친다.

Hopper의 thread-block cluster는 sharing scope를 확장한다. 기본 shared memory는 block 범위지만, cluster 안의 block은 distributed shared memory를 통해 다른 block의 shared에 접근할 수 있다. 이것은 L2가 shared로 바뀐 것이 아니다. cluster가 같은 GPC에 함께 배치되는 실행 제약, cluster synchronization, remote shared access의 의미가 추가된 것이다. cluster kernel은 `cudaOccupancyMaxActiveClusters` 같은 cluster-aware occupancy 계산이 필요하다는 tuning guide의 권고를 따라야 한다.

가이드 기준 H100/CC 9.0은 228 KB shared per SM, 227 KB per block을 제시한다. 큰 capacity는 더 큰 tile이나 pipeline stage를 허용할 수 있지만 자동 이득은 아니다. tile이 커지면서 register와 compute work도 늘고, cluster 사용 시 여러 block의 동시 배치 조건이 추가된다. capacity는 설계 공간을 넓히며 답을 대신하지 않는다.

### 41.5.3 Blackwell은 SKU와 compute capability를 붙여 말한다

Blackwell tuning guide는 CC 10.0과 CC 12.0의 shared capacity를 다르게 적는다. CC 10.0은 228 KB per SM과 227 KB per block, CC 12.0은 128 KB per SM과 99 KB per block이다. “Blackwell은 shared가 228 KB”라고 쓰면 CC 12.0 경로를 틀리게 설명한다. kernel dispatch와 resource validation은 architecture name이 아니라 실제 device capability와 specialization을 확인해야 한다.

가이드는 Hopper에서 도입된 thread-block cluster와 distributed shared memory가 Blackwell에서도 지원된다고 설명한다. 계속 지원되는 기능과 새로 도입된 기능을 구분해야 한다. GB200의 L2 126 MB라는 제품 수치도 모든 Blackwell GPU의 보편 상수로 옮기지 않는다. B200의 combined L1/texture/shared capacity 설명 역시 해당 제품 문맥에 붙인다.

Blackwell의 tensor memory 같은 기능은 흥미롭지만, 이 장의 대표 source가 SM80 계열 FlashAttention이고 선택한 공식 구간만으로 실제 TMEM kernel의 descriptor, scope, lifetime, accumulator transfer를 끝까지 증명하지 못한다. 이름만 추가하면 책은 넓어 보이지만 독자는 쓸 수 없다. SM100 전용 kernel을 다룰 때는 공식 tensor-memory 절과 고정 source의 allocate, MMA, copy-out, deallocation protocol을 별도 경로로 연결해야 한다. 여기서는 의도적으로 그 주장을 보류한다.

## 41.6 고정 소스 산책: Q 한 타일의 생애

### 41.6.1 먼저 함수의 경계를 읽는다

대표 함수는 `compute_attn_1rowblock`이다. 이름이 말하듯 한 query row block의 attention을 계산한다. template parameter에는 dropout, causal/local attention, even shape, softcap, softmax 반환 같은 분기가 들어 있다. `Kernel_traits`는 element type, block shape, warp 수, shared layout, copy atom, tiled MMA를 묶는다. 따라서 이 함수의 한 줄을 읽을 때도 모든 instantiation이 같은 instruction을 낸다고 가정하지 않는다.

함수의 입력 `params`에는 Q, K, V, output pointer와 stride, sequence length, scale, optional mask/alibi 관련 값이 들어간다. `bidb`, `bidh`, `m_block`은 현재 batch, head, query block을 특정한다. 이 세 coordinate가 global tensor의 어느 tile을 읽고 어느 output tile에 쓸지를 정한다. attention의 수식만 보고 kernel을 읽으면 이 indexing과 tail protection을 놓친다.

처음 해야 할 일은 이름 접두사를 번역하는 것이 아니라 tensor engine을 확인하는 것이다. 이 코드베이스는 관습적으로 `g`를 global view, `s`를 shared view, `r`을 register fragment에 많이 쓰지만 접두사는 증거가 아니다. `make_gmem_ptr`, `make_smem_ptr`, `partition_fragment_*`, 실제 `copy`와 `gemm` call을 함께 확인해야 한다. CUTE tensor는 pointer와 layout을 조합한 view이며 view 생성 자체가 byte를 옮기지 않는다는 점도 중요하다.

### 41.6.2 163–184행: 주소를 그렸지만 아직 움직이지 않았다

고정 파일의 163–165행은 optional softmax output `gP`를 `make_gmem_ptr`로 만든다. 167–174행은 `smem_`을 `Element*`로 해석해 `sQ`, `sK`, `sV`, transposed V view를 만든다. `sK`의 시작 주소는 `Share_Q_K_smem`이 참이면 Q와 같은 base이고, 거짓이면 `sQ` 크기 뒤다. `sV`는 K 영역 다음에 놓인다. 이 세 줄만으로 이미 capacity 최적화와 lifetime 위험이 함께 보인다.

같은 base를 쓴다는 것은 Q와 K가 동시에 유효할 수 없다는 뜻이다. kernel은 필요한 Q를 register fragment로 옮기고 모든 thread가 그 이동을 끝냈음을 확인한 다음 K가 같은 shared 영역을 덮어쓰도록 해야 한다. 어느 한 warp라도 이전 Q를 shared에서 더 읽어야 하는데 producer가 K를 쓰면 data race다. shared aliasing의 정확성은 layout 계산보다 phase 전환 protocol에 달려 있다.

176–184행은 global copy의 thread slice와 source/destination partition을 만든다. `tQgQ`는 global Q source partition, `tQsQ`는 shared Q destination partition이다. K와 V도 같은 쌍을 가진다. partition은 “thread `tidx`가 어느 logical coordinate를 담당하는가”를 정한다. 아직 실제 copy instruction은 발행되지 않았다. profiler에서 load를 찾을 때 view/partition 생성 줄에 breakpoint를 걸고 byte가 이동했다고 생각하면 안 된다.

이 구분은 C++ template kernel을 읽는 기본 기술이다. 첫 단계는 storage engine과 layout을 선언한다. 둘째 단계는 tiled copy가 각 thread의 source와 destination slice를 정한다. 셋째 단계의 `copy` 호출이 실제 load/store 또는 async copy를 생성한다. 넷째 단계의 fence/wait/barrier가 producer와 consumer 사이의 readiness를 만든다. 한 단계라도 빼고 “global에서 shared로 복사한다”고 설명하면 디버깅 가능한 지식이 되지 않는다.

### 41.6.3 186–212행: MMA가 원하는 모양으로 소유권을 나눈다

186–194행은 `TiledMma`에서 현재 thread의 MMA slice를 얻는다. 188행의 `tSrQ`, 189행의 `tSrK`, 190행의 `tOrVt`는 shared tensor를 MMA A/B operand fragment 관점으로 partition한다. 187행의 `acc_o`는 output accumulator fragment다. 이 객체들은 수학적으로 Q, K, V, O 일부를 가리키지만, lane별 물리 layout은 일반 row-major 작은 행렬과 다르다.

왜 fragment layout이 필요한가. Tensor Core MMA instruction은 warp의 lane들이 각자 가진 operand 조각을 정해진 방식으로 소비하고 accumulator 조각을 lane에 돌려준다. 한 thread가 완전한 16×16 matrix를 register에 갖는 구조가 아니다. warp 전체에 분산된 fragment가 하나의 tile 연산을 이룬다. 그래서 debugger에서 한 thread의 register만 보고 전체 score matrix를 복원하기 어렵고, CUTE의 partition/layout 변환이 핵심 의미를 가진다.

200–212행은 shared에서 MMA fragment로 옮길 copy atom을 다시 tile한다. Q와 K는 MMA A/B operand에 맞고, V는 transposed shared layout에 맞는 별도 copy atom을 쓴다. 같은 element byte라도 소비 instruction이 원하는 lane mapping이 다르면 staging layout과 register arrangement가 달라진다. “shared를 쓴다”보다 “MMA consumer가 요구하는 배열로 cooperative staging한다”가 정확한 설명이다.

`acc_o`의 lifetime은 특히 길다. K/V block loop가 진행되는 동안 이전 block의 partial output이 계속 살아 있고, online softmax가 새 block의 maximum과 normalization factor에 맞춰 기존 accumulator를 rescale한다. 긴 live range는 재료를 다시 읽지 않는 장점과 register pressure라는 비용을 동시에 만든다. compiler가 실제 몇 register를 할당하는지는 source fragment 크기만으로 확정하지 않지만, 압력 후보는 명확하다.

### 41.6.4 [254–287행](https://github.com/vllm-project/flash-attention/blob/28e862d21806bc3580207aa0ad4e2759151e9827/csrc/flash_attn/src/flash_fwd_kernel.h#L254-L287): 비동기 copy는 완료 계약을 요구한다

257행의 `FLASH_NAMESPACE::copy`가 global Q partition에서 shared Q partition으로 prologue copy를 발행한다. shape가 block에 딱 맞지 않으면 coordinate와 predicate가 유효 row와 head-dimension element만 복사한다. 259행은 Q를 register에 유지하는 trait일 때 `cp_async_fence`를 발행한다. fence는 앞서 발행한 async operation group의 경계를 만든다.

K copy는 276–277행에서 현재 마지막 K block을 global에서 shared로 보낸다. 이어지는 `cute::cp_async_fence()`는 다음 consumer wait가 참조할 group을 commit한다. copy call이 return했다고 shared byte를 즉시 읽을 수 있다고 생각하면 안 된다. asynchronous라는 말은 issuing thread가 data arrival까지 멈춰 있지 않을 수 있다는 뜻이며, 정확성은 이후 wait에 의존한다.

`Share_Q_K_smem` 경로의 265–271행을 자세히 보자. `cp_async_wait<0>()`가 outstanding group의 완료를 기다리고 `__syncthreads()`가 block thread를 모은다. 그 뒤 Q shared tile을 `tSrQ` register fragment view로 copy한다. 다시 `__syncthreads()`를 호출해 모든 thread가 shared Q를 다 읽었음을 확인한 뒤 같은 영역을 K에 양보할 수 있게 한다. 첫 barrier는 producer completion과 collective consumption 시작을 맞추고 둘째 barrier는 shared lifetime 종료를 맞춘다.

Q와 K가 shared를 공유하지 않지만 Q를 register에 둘 때도 282–287행에서 wait와 barrier 후 shared Q를 register fragment로 가져온다. `cp_async_wait<1>()`의 template 숫자는 pipeline에 남겨 둘 group 수와 관련된다. `0`과 `1`을 단순한 boolean으로 읽으면 안 된다. 정확한 group semantics는 helper 구현과 CUDA async-copy 계약을 함께 봐야 한다.

성능 의도도 정확성 protocol 뒤에 놓인다. Q copy와 다른 준비 작업, K copy와 이전 tile의 compute를 겹치면 memory latency를 숨길 수 있다. 하지만 pipeline stage가 늘면 shared에 여러 tile을 동시에 보유해야 하고 address/predicate state가 register를 차지한다. wait를 늦추면 overlap 가능성이 늘지만 consumer 전에 완료되지 않을 위험이 있고, 너무 일찍 기다리면 overlap이 사라진다. 최적 위치는 instruction dependency와 tile latency의 함수다.

### 41.6.5 [310–351행](https://github.com/vllm-project/flash-attention/blob/28e862d21806bc3580207aa0ad4e2759151e9827/csrc/flash_attn/src/flash_fwd_kernel.h#L310-L351): score를 만들고 전체 행렬을 저장하지 않는다

masking loop가 시작되면 310행에서 `acc_s` score accumulator fragment를 만들고 311행에서 clear한다. 312행의 async wait와 313행의 block barrier 뒤에 현재 K tile을 읽어도 된다. 동시에 315–324행은 다음 V tile을 shared로 가져오기 시작한다. compute와 다음 operand 이동이 pipeline으로 엮인다.

326–329행의 `gemm`은 Q와 K를 소비해 `acc_s`에 QKᵀ score tile을 누적한다. source signature에는 `acc_s`, Q/K register/shared partition, tiled MMA, shared copy object가 함께 전달된다. `A_in_regs` trait에 따라 Q가 register fragment에 있을 수도 있고 shared에서 fragment로 공급될 수도 있다. 하나의 함수 호출이므로 모든 operand가 같은 주소 공간에 있다고 생각하면 안 된다.

score를 얻은 뒤 softcap과 mask가 적용된다. causal mask는 현재 query position보다 미래의 K position을 제외하고, local mask는 window 밖을 제외하며, uneven tail은 실제 sequence length 밖의 column을 제외한다. mask는 correctness와 수치 안정성의 일부다. invalid score를 softmax 전에 충분히 작은 값으로 만들지 않으면 probability mass가 padding이나 미래 token으로 샌다.

349–351행은 `softmax_rescale_o`를 호출한다. 첫 K block과 이후 block을 구분한다. 이 함수의 중요한 의도는 전체 `L×L` score matrix를 global memory에 쓰고 다시 읽는 대신, blockwise score를 처리하면서 running maximum과 normalization sum을 갱신하고 기존 output accumulator를 새 scale에 맞추는 것이다.

두 score block의 직관적 예를 보자. 첫 block의 최대가 `m₁`이고 지수합이 `l₁`, weighted value 누산이 `o₁`라고 하자. 둘째 block 최대가 `m₂`, 지수합이 `l₂`, 누산이 `o₂`다. 전체 최대 `m = max(m₁,m₂)`를 잡으면 결합된 합은 `l = exp(m₁-m)l₁ + exp(m₂-m)l₂`이고 결합 누산은 `o = exp(m₁-m)o₁ + exp(m₂-m)o₂`다. 이전 accumulator를 버리지 않고 새 maximum 기준으로 rescale하는 이유가 여기에 있다.

이 기법은 단지 softmax 함수를 kernel 안에 fusion했다는 말보다 깊다. 중간 score matrix의 global materialization을 피하여 device-memory byte를 줄이고, tile-local score fragment와 running statistics의 lifetime을 register/shared 범위로 제한한다. 대신 accumulator가 loop 전체에서 살아야 하고 rescale 연산과 수치 안정성 규칙이 필요하다. 메모리 절감과 register pressure, 추가 산술의 교환이다.

### 41.6.6 353–435행: probability fragment와 V가 output을 만든다

353–354행은 FP32 score accumulator를 `Element` 타입으로 변환해 `rP` fragment를 만든다. FP16 또는 BF16 tensor-core operand로 줄이는 단계는 byte와 throughput에 유리하지만 수치 표현 범위를 바꾼다. softmax의 maximum subtraction과 accumulation을 FP32에 가깝게 유지하고 MMA operand를 낮은 precision으로 변환하는 분업은 속도와 안정성을 맞추려는 설계다.

357–364행은 `Return_softmax`가 켜졌을 때 probability 또는 dropout-encoded 결과를 global output `gP`에 쓰는 선택 경로다. 일반 serving forward에서는 전체 probability를 반환하지 않는 구성이 흔하지만, template option이 켜지면 traffic과 storage가 늘 수 있다. 같은 source file의 kernel 이름만 보고 output byte 수를 고정하지 말고 instantiated option을 확인해야 한다.

370–372행은 `rP`의 layout을 MMA A-register layout으로 재해석한다. 432–435행의 `gemm_rs`는 probability fragment와 shared V tile을 곱해 `acc_o`에 누적한다. 이름의 `rs`는 이 구현에서 register A와 shared B의 공급 형태를 암시하지만, 정확한 instruction sequence는 helper와 instantiated trait를 확인해야 한다. 중요한 사실은 P 전체를 global tensor에 저장하지 않고 register fragment를 바로 V MMA에 연결한다는 점이다.

V tile은 앞선 async copy로 shared에 도착했고, MMA가 요구하는 transposed/no-swizzle view와 copy atom을 가진다. 같은 V byte는 여러 query row의 probability와 곱해질 수 있으므로 block 내 reuse가 생긴다. K/V block loop가 끝날 때까지 `acc_o`는 running normalization에 맞춰 누적된다. 이 구조 덕분에 attention의 logical score 크기가 커져도 중간 global storage가 sequence-length 제곱으로 늘지 않는다.

여기서 “FlashAttention은 memory efficient하다”를 더 정확히 말할 수 있다. 핵심은 모든 데이터를 더 빠른 곳에 한꺼번에 넣는 것이 아니다. Q/K/V를 block tile로 나누고, score와 probability의 lifetime을 짧은 register fragment로 제한하며, online normalization으로 output accumulator만 loop를 가로질러 유지하여 중간 tensor의 global read/write를 제거한다. tile은 유한한 register와 shared 예산 안에서 반복된다.

### 41.6.7 [438–499행](https://github.com/vllm-project/flash-attention/blob/28e862d21806bc3580207aa0ad4e2759151e9827/csrc/flash_attn/src/flash_fwd_kernel.h#L438-L499): 결과도 곧바로 global에 쓰지 않는다

loop가 끝나면 440행은 final log-sum-exp와 normalization을 계산한다. 442–443행은 FP32 `acc_o`를 output element type의 register fragment `rO`로 변환한다. output dtype이 FP16/BF16이면 이 지점에서 rounding이 발생한다. LSE는 backward나 split 결합 등 option에 따라 별도 global output으로 저장될 수 있다.

444행은 이전 `sQ` 영역에 output shared view `sO`를 만든다. Q lifetime이 끝났기 때문에 같은 storage를 epilogue에 재사용한다. 446–454행은 accumulator layout을 shared output layout에 맞춰 partition하고 register `rO`에서 shared `sO`로 cooperative copy한다. 이 단계가 필요한 이유는 MMA accumulator의 lane layout과 global output의 contiguous/coalesced-friendly cooperative layout이 같지 않기 때문이다.

“왜 register에서 곧장 global에 쓰지 않는가”라는 질문은 좋다. 각 lane이 가진 accumulator 조각의 배열이 global row-major destination에 효율적으로 쓰기 어려울 수 있다. shared를 exchange/reordering 공간으로 사용하면 warp/block thread가 output tile을 global copy atom이 원하는 조각으로 다시 읽을 수 있다. shared staging은 입력 재사용뿐 아니라 epilogue layout 변환에도 쓰인다.

456–467행은 global output tensor `mO/gO`와 LSE tile을 만들고 global output copy의 thread slice를 구성한다. 469행의 barrier는 shared output이 모두 준비됐음을 보장한다. 471–472행은 shared `sO`를 thread-local `tOrO` staging fragment로 읽는다. 마지막 488–499행은 identity coordinate와 predicate를 만들어 실제 sequence row와 head dimension 밖의 element를 쓰지 않고 global output에 copy한다.

이 마지막 경로를 `shared → register → global`이라고 요약할 수 있지만 register staging의 실제 instruction과 개수는 compiler 결과로 확인해야 한다. source 수준에서 확정할 수 있는 것은 `tOrO` fragment가 만들어지고 shared source에서 채워진 뒤 predicated global destination copy의 source가 된다는 dataflow다. output store가 L2에 어떻게 allocate되고 언제 HBM에 writeback되는지는 source 한 줄만으로 확정하지 않는다.

### 41.6.8 한 장의 수명표로 다시 읽는다

| 객체 | 만들어지는 위치 | 주요 소비자 | lifetime 종료 조건 | 주된 자원 압력 |
|---|---|---|---|---|
| global Q/K/V view | 함수 prologue | async tiled copy | 해당 kernel invocation의 관련 tile copy 완료 | address state, device traffic 후보 |
| shared Q | prologue copy | Q MMA fragment | Q fragment copy와 block synchronization 완료 | block shared bytes |
| shared K/V | 각 KV iteration | QKᵀ MMA, P×V MMA | 해당 iteration 소비 및 다음 overwrite 전 barrier | pipeline stage shared bytes |
| `tSrQ/tSrK/tOrVt` | MMA partition | tiled MMA | trait와 loop scheduling이 정한 마지막 use | thread register footprint 후보 |
| `acc_s` | KV iteration 시작 | mask와 online softmax | 현재 iteration P fragment 변환 뒤 | FP32 register accumulator |
| `acc_o` | loop 전 | rescale와 P×V, final normalization | epilogue conversion 뒤 | 긴 live-range register accumulator |
| shared `sO` | epilogue | global output copy | predicated store source read 완료 | Q storage와 시간 분할된 shared bytes |
| global O | epilogue | 후속 model layer | allocation/serving runtime가 정한 수명 | L2/device-memory write traffic 후보 |

이 표의 핵심 열은 위치보다 종료 조건이다. buffer reuse와 pipeline overlap은 lifetime 종료를 증명할 때만 안전하다. 성능 버그는 byte를 너무 오래 보유해 capacity를 잠식할 때 생기고, correctness bug는 아직 필요한 byte를 너무 일찍 덮어쓸 때 생긴다. 같은 lifetime 분석이 두 종류의 장애를 연결한다.

## 41.7 실행하지 않고 하는 손계산

### 41.7.1 Q/K/V shared footprint의 첫 모델

실제 trait를 열기 전에 symbolic equation을 만든다. query tile을 `B_M × D`, key/value tile을 각각 `B_N × D`, element 크기를 `b` byte라고 하자. Q, K, V를 동시에 shared에 한 stage씩 둔다면 단순 payload는 `(B_M + 2B_N) × D × b` byte다. padding, swizzle, alignment, multi-stage buffering은 아직 제외한다.

예를 들어 `B_M=64`, `B_N=64`, `D=128`, `b=2`이면 Q는 `64×128×2 = 16,384 byte`, K도 16,384 byte, V도 16,384 byte다. 합은 49,152 byte, 즉 48 KiB다. K/V를 double buffer하여 다음 tile을 미리 가져오면 Q 한 벌과 K/V 두 벌이 필요해 단순 payload는 `16 KiB + 2×(16 KiB+16 KiB) = 80 KiB`가 된다.

이 숫자를 곧 kernel dynamic shared request라고 쓰면 안 된다. CUTE layout의 padding이나 swizzle storage, output epilogue가 peak lifetime에서 겹치는지, Q/K aliasing 여부, alignment, barrier object, trait의 stage 수를 확인해야 한다. 손계산은 source를 어디서 더 볼지 알려 주는 하한 모델이다. 실제 `sizeof` 또는 launcher가 넘기는 shared bytes와 비교해 차이를 설명해야 한다.

`Share_Q_K_smem`이 참이고 Q를 register로 옮긴 뒤 같은 영역을 K에 쓴다면 peak shared payload가 줄 수 있다. 그러나 Q fragment가 register에 상주하여 register pressure는 늘어난다. shared를 절약한 byte가 사라진 것이 아니라 일부 working state의 소유 위치가 바뀐 것이다. optimization을 공간 이동으로 읽으면 한 예산의 절약이 다른 예산의 비용이 되는 모습을 놓치지 않는다.

head dimension이 256으로 두 배가 되면 위 payload는 단순히 두 배가 된다. 같은 tile과 stage를 유지하면 160 KiB에 접근할 수 있어 architecture별 shared limit과 opt-in 경계가 달라진다. 실제 kernel은 head dimension에 따라 `B_M/B_N`, warp 수, stage, split 여부를 바꿀 수 있다. “D가 두 배니 같은 kernel이 shared만 두 배 쓴다”가 아니라 dispatcher가 다른 specialization을 고르는지 먼저 확인한다.

### 41.7.2 score와 output accumulator의 register 압력

register fragment는 lane에 분산되므로 정확한 register count는 compile 결과가 필요하다. 그래도 logical accumulator element로 압력 방향을 계산할 수 있다. score tile이 `B_M × B_N`이고 FP32 accumulator라면 logical payload는 `B_M×B_N×4` byte다. `64×64`면 16 KiB다. 128 thread가 균등하게 나눈다는 단순 모델에서는 thread당 128 byte, 즉 32개의 32-bit register 상당이다.

output accumulator가 `B_M × D` FP32라면 `64×128×4 = 32 KiB`, 128 thread 균등 모델에서 thread당 256 byte, 64 register 상당이다. 두 accumulator만 합쳐도 평균 96 register 상당이다. 여기에 Q/K/V operand fragment, probability fragment, pointers, strides, predicates, loop counter, random/dropout state가 더해진다. 실제 lane layout과 temporary overlap 때문에 단순 평균과 compiler allocation은 다르지만, 왜 attention kernel이 register-heavy해지기 쉬운지는 보인다.

`B_M`을 64에서 128로 늘리면 score와 output accumulator logical payload가 모두 두 배가 된다. global/shared reuse가 좋아질 가능성과 register residency가 악화될 가능성이 동시에 생긴다. `B_N`을 늘리면 score accumulator와 K/V staging이 커지고 KV loop 횟수는 줄 수 있다. `D`를 늘리면 Q/K/V tile과 output accumulator가 함께 커진다. 각 tile 축이 어느 자원을 건드리는지 표로 적으면 autotuning 결과를 해석하기 쉬워진다.

| 변경 | 줄어들 수 있는 것 | 늘어나는 주요 후보 | 확인할 경계 |
|---|---|---|---|
| `B_M` 증가 | query tile 수, Q 재로딩, launch tile overhead | score/output accumulator, Q shared, tail waste | registers/thread, shared/CTA, 실제 M 분포 |
| `B_N` 증가 | KV loop와 barrier 횟수 | score accumulator, K/V shared, tail waste | pipeline stage, long-context와 short-context 분포 |
| stage 증가 | copy latency 노출 | K/V shared, address state, barrier state | resident CTA와 producer-consumer overlap |
| Q register 유지 | shared alias 기회, Q shared 재독 | register footprint와 live range | spill, occupancy limit, Q reuse 횟수 |
| lower element bytes | shared/global payload | conversion 및 수치 오차 위험 | supported MMA path, accumulator precision |

### 41.7.3 online softmax가 없을 때의 중간 byte

naive한 구현이 QKᵀ score 전체를 global memory에 썼다가 softmax와 PV를 위해 다시 읽는다고 가정해 보자. 한 head, sequence length `L=8192`, score element가 FP16 2 byte라면 score matrix 하나는 `8192²×2 = 134,217,728 byte`, 128 MiB다. 쓰기 한 번과 읽기 한 번만 세어도 256 MiB이고, softmax가 별도 output을 쓰고 다시 읽으면 더 커진다.

이 계산은 naive implementation의 교육용 비교이지 실제 framework가 그렇게 실행한다는 주장이 아니다. fused attention이 tile-local score와 online statistics를 유지하는 이유를 보여 준다. sequence 길이가 두 배가 되면 materialized score byte는 네 배가 된다. 반면 tiled online algorithm의 intermediate score storage는 tile 크기에 묶이고 KV loop 횟수가 늘어난다. global intermediate capacity를 bounded on-chip working set과 반복 compute로 바꾸는 것이다.

online softmax에도 비용은 있다. 각 tile마다 maximum과 exponential sum을 갱신하고 이전 output accumulator를 rescale한다. 수학적으로 동일한 결과를 안정적으로 결합해야 하며, floating-point reduction order가 달라질 수 있다. tile 크기가 작으면 loop와 rescale 횟수가 늘어난다. “global byte를 없앴으니 무료”가 아니라, device-memory traffic을 추가 산술과 on-chip state로 교환했다.

### 41.7.4 decode KV byte의 상하한

GQA 모델을 가정해 query heads가 32개, KV heads가 8개, head dimension이 128, context가 16,384, KV element가 2 byte라고 하자. logical K/V storage read의 완전 공유 하한은 KV head 기준 `2 × 8 × 16,384 × 128 × 2 = 67,108,864 byte`, 64 MiB다. 각 query head가 K/V를 별도로 읽는 극단적 상한 모델은 `2 × 32 × 16,384 × 128 × 2 = 268,435,456 byte`, 256 MiB다.

실제 physical traffic은 이 두 숫자 사이에 단순히 놓인다고 보장할 수도 없다. page metadata와 scale/zero point, quantized formats, repeated loads, write traffic을 더하면 상한 모델을 넘을 수 있고, L2 reuse가 있으면 HBM read는 logical read보다 작을 수 있다. 이 계산은 “query head 간 KV reuse가 얼마나 실현되는가”를 source와 counter에서 물을 범위를 만든다.

batch가 `B`로 늘면 서로 다른 sequence의 KV는 보통 공유되지 않으므로 logical bytes가 대체로 합산된다. prefix cache가 logical KV blocks를 공유하더라도 한 decode step에서 kernel grid와 head mapping이 physical cache reuse를 만드는지는 별도 문제다. prefix sharing은 allocation 중복을 줄이는 효과와 실행 중 L2 hit를 높일 가능성을 구분해야 한다.

### 41.7.5 arithmetic intensity는 분모를 명시한다

attention decode의 연산량을 대략 QK dot product와 PV weighted sum으로 잡으면 query head당 약 `4LD` floating-point operations 규모로 볼 수 있다. K와 V logical read는 약 `2LDb` byte다. 완전한 단일-head 단순 모델의 arithmetic intensity는 `4LD/(2LDb)=2/b FLOP/byte`가 된다. `b=2`면 약 1 FLOP/byte다. 이 값은 매우 거친 알고리즘 비율이다.

GQA 공유, cache hit, quantization/dequantization, softmax 산술, tensor-core counting convention, output과 metadata를 포함하면 분자와 분모가 바뀐다. HBM 기준 intensity를 구할 때는 profiler의 device-memory bytes를 분모로 써야 하고, global instruction 기준 intensity와 혼동하지 않는다. L2 기준 intensity는 또 다른 질문이다. “arithmetic intensity가 1”이라는 숫자만 쓰지 말고 어느 계층의 byte인지 적는다.

roofline식 추론도 조건문으로 쓴다. 계산 peak를 `P FLOP/s`, 해당 계층 sustainable bandwidth를 `W byte/s`, workload intensity를 `I FLOP/byte`라고 할 때 가능한 상한은 `min(P, I×W)`다. 그러나 peak 사양은 achieved 값이 아니고 작은 decode grid는 충분한 concurrency를 만들지 못할 수 있다. 이 장의 손계산은 후보 병목을 좁히며 측정을 대신하지 않는다.

## 41.8 무엇을 관측해야 설명이 사실이 되는가

### 41.8.1 source, compile, runtime evidence를 층별로 쌓는다

첫째 층은 source evidence다. global/shared view, copy call, fragment, barrier, store predicate를 고정 commit과 line으로 기록한다. source evidence는 구현 의도와 dataflow를 보여 준다. register 개수나 cache hit, elapsed time을 보여 주지 않는다.

둘째 층은 compile evidence다. 어떤 architecture flag와 template specialization이 선택됐는지, static/dynamic shared bytes, registers per thread, spill load/store bytes, generated instruction을 기록한다. 같은 source라도 `Kernel_traits`, dtype, head dimension, dropout, return-softmax option에 따라 결과가 다르다. binary가 어떤 path를 담았는지 확인하지 않으면 source walk와 배포 artifact가 어긋날 수 있다.

셋째 층은 runtime observation이다. 실제 launch grid/block/shared bytes, active warps의 제한 원인, memory throughput, L1/L2/device-memory bytes, local load/store, barrier stall, tensor-core instruction, kernel duration을 workload shape와 함께 본다. user가 이 장에서 실행을 원하지 않았으므로 여기서는 항목과 판정 순서만 설계한다. 수치와 그래프는 만들지 않는다.

세 층을 섞지 않는 문장 예시는 다음과 같다. “source는 Q를 async copy로 shared에 staging한다.” “SM80 specialization의 compile report는 CTA당 X shared bytes와 thread당 Y registers를 사용한다.” “shape S의 trace에서 kernel은 Z microseconds이고 device-memory read가 W bytes다.” 첫 문장은 이 장에서 확정할 수 있고, 둘째와 셋째는 실제 artifact와 관측값이 있을 때만 채운다.

### 41.8.2 register 문제의 조사 순서

증상은 tile을 키우거나 head dimension을 바꾼 뒤 latency가 계단식으로 악화되는 것이다. 먼저 dispatcher가 같은 kernel family와 specialization을 선택했는지 확인한다. 다른 kernel로 바뀌었다면 register 하나만 원인으로 잡을 수 없다. 다음으로 compile report의 registers/thread, static shared, spill stores/loads를 비교한다.

register allocation 증가가 보이면 CTA의 thread 수와 곱해 SM register budget에 대한 resident CTA 상한을 손으로 계산한다. allocation granularity와 warp/block limit를 반영한 occupancy 도구 결과를 그다음에 본다. active warps가 줄었어도 latency 악화의 충분조건은 아니다. instruction-level parallelism과 tile reuse가 좋아져 전체 시간은 줄 수 있다.

spill이 생겼다면 generated local load/store와 runtime local-memory counter를 연결한다. local traffic이 L2에 hit할 수 있으므로 HBM bytes만 보고 spill이 없다고 결론 내리지 않는다. spill instruction이 hot loop 안에 있는지, prologue/epilogue에만 있는지 위치를 본다. live range를 줄이거나 unroll/stage를 조절한 비교가 필요하다.

### 41.8.3 shared와 barrier 문제의 조사 순서

shared tile 변경 뒤에는 launcher가 넘긴 dynamic shared bytes와 `cudaFuncSetAttribute` opt-in 설정을 확인한다. architecture별 per-block limit를 넘으면 launch failure가 날 수 있고, limit 안이어도 resident block 수가 달라질 수 있다. preferred carveout이 설정됐는지와 실제 device capability도 기록한다.

정답이 간헐적으로 틀리면 성능 counter보다 lifetime protocol을 먼저 본다. async copy가 commit됐는지, consumer 전에 필요한 wait가 있는지, producer와 consumer thread 집합을 barrier가 모두 포함하는지, shared alias overwrite 전에 마지막 reader가 끝났는지 확인한다. tail predicate가 copy source와 destination에 일관되게 적용됐는지도 본다.

성능만 나쁘다면 barrier stall을 바로 “동기화가 많다”고 해석하지 않는다. barrier에서 기다리는 이유는 producer copy가 늦어서일 수 있고, warp별 work imbalance나 tail mask 때문에 일부 warp가 늦게 도달해서일 수 있다. pipeline stage를 늘리면 기다림이 줄 가능성이 있지만 shared pressure가 늘어난다. trace의 issue 시점과 wait interval, active CTA 변화를 함께 본다.

### 41.8.4 L2와 HBM 문제의 조사 순서

먼저 tensor shape로 logical bytes의 하한과 상한을 계산한다. K/V, weight, output, metadata, quantization scale, spill 후보를 목록으로 나눈다. 다음으로 실제 kernel이 어느 global view를 몇 iteration에서 읽는지 source에서 확인한다. 마지막에 profiler의 L2 sector/byte와 device-memory byte를 계층별로 비교한다.

L2 hit ratio 하나만으로 결론 내리지 않는다. hit ratio가 높아도 요청 byte 총량이 매우 크면 miss byte가 HBM bandwidth를 채울 수 있다. hit ratio가 낮아도 작은 traffic이면 critical path가 아닐 수 있다. read/write 방향과 kernel별 attribution, concurrent kernel/communication traffic을 분리한다. NCCL이나 다른 stream의 DMA가 같은 device-memory 자원을 쓰는 시간도 확인 대상이다.

HBM bandwidth가 사양 peak에 못 미친다고 memory-bound가 아니라고 말해서도 안 된다. 작은 grid, dependency chain, irregular access, insufficient outstanding requests, partition imbalance가 sustainable bandwidth를 제한할 수 있다. 반대로 bandwidth가 높다고 그 traffic이 유용하다는 뜻도 아니다. spill, redundant load, materialized intermediate가 bandwidth를 채울 수 있다. logical useful bytes 대비 observed bytes의 증폭을 계산한다.

### 41.8.5 최소 재현 기록

관측을 수행할 미래의 조사 노트에는 model identifier와 revision, dtype/quantization, GPU 정확한 SKU와 compute capability, driver/CUDA/runtime/library version, framework commit, 선택된 attention backend, prompt/context/output lengths, batch/concurrency, head 수와 KV head 수, head dimension을 적는다. shape 없이 kernel metric을 저장하면 재현할 수 없다.

kernel에는 mangled name만이 아니라 dispatch option을 연결한다. causal/local, varlen, split-KV, return-softmax, dropout, Q-in-register, shared alias, block M/N, warp 수, stages가 가능한 범위에서 필요하다. compile artifact의 hash와 launch의 grid/block/dynamic shared도 남긴다. source permalink만 같아도 build flag가 다르면 다른 resource usage가 나올 수 있다.

비교 실험은 한 번에 한 축을 바꾸고 correctness를 먼저 검사한다. output tolerance, NaN/Inf, tail shape, causal boundary, ragged sequence를 확인한 뒤 latency와 throughput을 본다. warmup과 graph capture 여부, clock/power 상태, 다른 tenant 유무를 기록한다. 평균 하나보다 distribution과 반복 간 변동을 남긴다. 다만 이 장 자체는 실행 결과를 제시하지 않는다.

## 41.9 사고 장면으로 되짚는 진단법

### 41.9.1 “shared를 키웠는데 느려졌다”

첫 가설은 shared capacity 경계를 넘었다는 것이다. 이전 CTA footprint와 새 footprint를 architecture per-SM budget으로 나누어 resident CTA 상한이 바뀌는지 계산한다. 동시에 register allocation과 block threads가 같았는지 확인한다. specialization이 바뀌었다면 tile 크기 외의 instruction과 stage도 달라졌을 수 있다.

둘째 가설은 tail waste다. 실제 serving M과 KV length 분포에서 tile utilization을 계산한다. `B_N=128`인데 짧은 KV가 129라면 두 번째 tile의 대부분이 비어도 shared allocation과 일부 pipeline overhead는 지불한다. 긴 benchmark에서는 좋아지고 실제 짧은 요청에서는 나빠질 수 있다.

셋째 가설은 synchronization과 pipeline이다. 큰 tile의 copy completion이 늦고 compute가 이를 숨기지 못할 수 있다. 반대로 stage가 줄어 overlap이 약해졌을 수도 있다. “shared가 빠르다”는 속성은 이 세 비용을 상쇄하는 reuse가 있을 때만 결과로 나타난다.

### 41.9.2 “register에 올렸는데 local traffic이 늘었다”

source option의 의도는 Q를 register에 유지하는 것이지만, Q fragment 추가로 전체 live set이 register budget을 압박하면 다른 temporary가 spill될 수 있다. 한 객체의 위치를 register로 옮긴 결정이 kernel 전체의 register allocation을 바꾼다. compile report에서 option 전후 register와 spill을 비교하고, spill instruction의 loop 위치를 본다.

occupancy가 줄어도 Q 재사용 이득이 더 클 수 있으므로 resident warp만으로 승패를 정하지 않는다. Q가 몇 K tiles에서 재사용되는지 계산한다. KV length가 짧아 reuse 횟수가 적으면 register 상주의 이득은 작고 footprint 비용은 그대로다. workload-dependent specialization이 필요한 이유다.

### 41.9.3 “L2 hit가 높은데 HBM 병목이라고 한다”

두 문장은 동시에 참일 수 있다. 1 TiB의 요청 중 90%가 L2 hit라도 10% miss는 100 GiB의 device-memory read다. 반대로 1 GiB의 요청에서 10% hit라면 miss byte는 훨씬 작다. ratio보다 절대 byte와 시간당 byte를 본다. read와 write, 해당 kernel과 concurrent traffic을 분리한다.

또한 profiler metric의 분모와 scope를 확인한다. 특정 unit의 sector hit ratio가 전체 application L2 byte와 같지 않을 수 있다. metric 이름과 공식 정의, replay/pass 조건을 기록한다. 정확한 cache line과 sector transaction 해석은 다음 장에서 다룬다. 여기서는 계층별 byte accounting이 먼저라는 원칙만 세운다.

### 41.9.4 “HBM 사용률이 낮으니 compute-bound다”

낮은 HBM bandwidth는 compute saturation의 충분한 증거가 아니다. grid가 작아 memory-level parallelism이 부족하거나, address dependency와 page-table lookup, barrier가 load issuance를 막을 수 있다. L2 hit가 높아 HBM을 적게 쓰면서도 L2/shared 경로가 병목일 수 있다. tensor core도 충분히 사용하지 못하면 compute-bound와 memory-bound 사이의 latency-bound 상태가 된다.

필요한 비교는 achieved tensor-core activity, issue stall, active warps, L2/device-memory bytes, barrier stall, kernel grid를 한 shape에서 보는 것이다. roofline의 두 peak만으로 설명되지 않는 launch와 dependency 제한을 인정한다. serving decode의 작은 M은 이런 제약을 자주 드러낼 수 있지만 실제 evidence 없이 보편 명제로 쓰지 않는다.

### 41.9.5 “가끔 wrong answer가 난다”

메모리 최적화 뒤 간헐적 오답은 cache가 오래돼서라는 막연한 설명보다 ownership protocol을 의심한다. shared alias의 이전 consumer가 끝나기 전에 다음 producer가 덮어썼는지, async wait group 숫자가 pipeline 변경 뒤에도 맞는지, barrier가 divergent branch 안에서 일부 thread만 도달하는지 확인한다.

tail shape는 반드시 재현에 포함한다. exact multiple shape에서는 predicate와 clear path가 실행되지 않아 bug가 숨을 수 있다. `seqlen_k = n×B_N ± 1`, `head_dim = supported width ± tail`, ragged batch, causal diagonal 주변을 검사한다. source의 497–499행처럼 global store가 OOB를 쓰지 않도록 `Clear_OOB`와 predicate가 의도대로 결합되는지 본다.

수치 오차와 race도 분리한다. deterministic한 작은 차이가 dtype conversion과 reduction order에서 오면 tolerance와 reference 비교로 분석한다. 반복마다 위치와 크기가 바뀌거나 NaN이 산발적으로 나오면 uninitialized shared, missing synchronization, OOB access 가능성이 높다. sanitizer와 최소 shape는 나중 실행 단계의 도구이며 이 장에서는 조사 설계만 제시한다.

## 41.10 현장 적용: 빠른 장소가 아니라 올바른 생애를 설계한다

이 장의 문제 장면으로 돌아가자. “HBM이 느리다”, “shared로 올리자”, “register를 더 쓰자”, “L2 hit가 높다”는 네 문장은 모두 너무 빨리 결론으로 건너갔다. 올바른 출발점은 byte biography다. 어떤 allocation의 어떤 logical tile이 어느 global address로 보이고, 어느 CTA가 그것을 shared에 staging하며, 어느 lane의 MMA fragment가 소비하고, accumulator가 몇 iteration을 살아남고, 어떤 barrier 뒤에 shared storage가 다음 용도로 재사용되며, output이 어느 predicate를 거쳐 global address에 저장되는지 먼저 그려야 한다.

첫 번째 교훈은 이름의 축을 분리하는 것이다. global memory는 주소 공간과 접근 모델이다. device allocation은 저장 객체와 allocator lifetime이다. HBM은 물리 저장 매체다. L2는 그 사이의 hardware-managed cache다. local memory는 thread-private 주소 공간이지만 on-chip register의 다른 이름이 아니다. shared memory는 block이 관리하는 scratchpad이고 L1과 물리 capacity 일부를 나눌 수 있어도 같은 프로그래밍 객체가 아니다.

이 구분은 문장 정확성만 위한 것이 아니다. 장애의 담당 계층을 찾게 한다. allocation이 잘못됐다면 serving allocator와 KV manager를 본다. global indexing이 잘못됐다면 tensor stride, block table, predicate를 본다. shared lifetime이 잘못됐다면 copy/wait/barrier와 alias protocol을 본다. register pressure가 의심되면 compiler allocation과 spill을 본다. HBM pressure가 의심되면 logical bytes와 L2/device-memory counters를 비교한다.

두 번째 교훈은 빠른 저장소도 유한 예산이라는 것이다. register에 더 오래 두면 재사용이 좋아질 수 있지만 resident warp와 spill 경계를 건드린다. shared tile을 키우면 global load 반복과 barrier 횟수를 줄일 수 있지만 CTA당 capacity와 tail waste가 늘어난다. L2 persistence를 요청하면 hot working set의 재사용 가능성을 높일 수 있지만 다른 stream과 tenant의 working set과 경쟁한다. HBM byte를 줄이려는 모든 선택은 compute, capacity, synchronization 중 다른 비용을 낸다.

세 번째 교훈은 lifetime이 성능과 correctness의 공통 언어라는 것이다. Q shared tile을 너무 오래 보유하면 epilogue가 쓸 공간이 줄고 capacity가 커진다. 너무 일찍 K나 output으로 덮으면 wrong answer가 난다. accumulator를 너무 짧게 유지하면 intermediate를 global에 materialize해야 할 수 있다. 너무 길게 유지하면 register pressure가 커진다. optimization은 byte의 생존 시간을 다음 소비자에게 필요한 최소 범위로 맞추는 일이다.

네 번째 교훈은 view와 movement를 구분하는 것이다. `make_gmem_ptr`와 `make_smem_ptr`는 주소와 layout을 표현한다. `partition_S/D`는 thread별 담당 조각을 정한다. 실제 `copy`가 movement를 발행한다. async copy는 fence와 wait를 통해 completion group을 관리한다. barrier는 여러 thread의 phase를 맞춘다. `gemm`은 fragment를 소비하고 accumulator를 갱신한다. source walk에서 이 단계를 색으로 구분하면 template 코드도 읽을 수 있다.

다섯 번째 교훈은 FlashAttention의 “왜”다. 전체 score matrix를 global에 쓰지 않고 blockwise QKᵀ score를 register fragment에 만들고 online softmax로 running maximum, sum, output accumulator를 갱신한다. probability fragment는 곧바로 shared V와 MMA에 들어간다. intermediate global byte를 추가 산술, shared staging, register lifetime으로 교환한다. 그래서 memory efficient라는 표현은 저장 공간 감소뿐 아니라 dataflow 재설계라는 의미다.

여섯 번째 교훈은 architecture 이름을 기능 목록처럼 섞지 않는 것이다. Ampere의 async global-to-shared copy는 중간 register copy를 피하고 overlap을 구성하는 기반이다. Hopper의 TMA는 다차원 tensor transfer와 cluster 범위를 확장하고, thread-block cluster는 distributed shared memory라는 새로운 sharing scope를 만든다. Blackwell은 이 기능 일부를 이어가지만 CC 10.0과 12.0의 shared capacity가 다르다. 제품명, compute capability, 문서 버전을 수치 옆에 붙여야 한다.

**독자가 코드 앞에서 답해야 할 열두 질문**

1. 이 pointer는 global, shared, local 가운데 어느 주소 공간을 표현하는가.
2. view를 만든 줄과 실제 byte를 옮기는 줄은 각각 어디인가.
3. copy source와 destination을 어느 thread들이 나누어 맡는가.
4. producer가 발행한 async operation은 어느 fence에서 group이 되고 어느 wait에서 완료가 요구되는가.
5. shared tile을 읽는 모든 consumer가 끝났음을 어느 barrier가 보장하는가.
6. 같은 shared storage를 두 tensor가 alias한다면 두 lifetime이 겹치지 않는 근거는 무엇인가.
7. 어느 fragment와 accumulator가 thread register에 오래 살아 있는가.
8. tile 축을 키울 때 register와 shared logical payload는 각각 어떻게 변하는가.
9. global intermediate를 없애는 대신 추가된 rescale, conversion, synchronization은 무엇인가.
10. global load가 곧 HBM load라고 말하지 않기 위해 어떤 cache/placement 증거가 필요한가.
11. tail과 ragged sequence에서 invalid source load와 destination store를 막는 predicate는 어디인가.
12. source 사실, compile artifact, runtime counter 가운데 지금 가진 증거는 어느 층인가.

이 질문에 답하면 “이 kernel은 shared memory를 쓴다”보다 훨씬 유용한 설명을 만들 수 있다. 예를 들어 “Q tile은 async global-to-shared copy 뒤 wait와 block barrier를 거쳐 MMA fragment로 들어가고, Q/K shared alias 경로에서는 두 번째 barrier 뒤에 storage lifetime이 끝난다. 이 선택은 shared capacity를 줄이는 대신 Q fragment의 register live range를 늘린다”라고 말할 수 있다. 원인, 기제, 대가, 확인 지점이 한 문장에 들어 있다.

**손계산과 관측의 연결**

손계산은 답을 꾸미는 도구가 아니라 관측 전에 범위를 정하는 도구다. Q/K/V tile payload를 element bytes로 계산하면 launcher shared bytes와 차이가 얼마나 되는지 물을 수 있다. logical score/output accumulator를 CTA thread로 나누면 register-heavy specialization을 예상하고 compile report를 먼저 볼 수 있다. decode KV logical bytes의 공유 하한과 비공유 상한을 계산하면 observed device-memory bytes가 어떤 reuse 수준을 암시하는지 질문할 수 있다.

관측값이 모델과 다르면 모델을 버리지 말고 빠진 byte와 경로를 찾는다. observed bytes가 크면 spill, optional softmax output, quantization scale, repeated page loads, tail waste를 추가한다. 작으면 L2 reuse, head sharing, zero-work predicate, 다른 specialization을 확인한다. duration이 나쁜데 bytes가 작으면 barrier, insufficient concurrency, instruction dependency, conversion을 본다. 한 counter로 결론 내리지 않는다.

**이 장이 일부러 답하지 않은 것**

이 장은 한 warp의 주소가 몇 개의 memory transaction으로 합쳐지는지 계산하지 않았다. cache line과 sector의 정확한 크기, alignment가 transaction을 어떻게 늘리는지, AoS/SoA와 stride가 coalescing에 어떤 영향을 주는지도 단정하지 않았다. 이 주제는 주소 식, element width, instruction, compute capability의 공식 규칙이 함께 있어야 정확하다. 다음 장에서 별도 증거 계약으로 다룬다.

또한 특정 kernel이 몇 GB/s를 달성하거나 어느 tile이 가장 빠르다고 주장하지 않았다. 실행하지 않았기 때문이다. source는 dataflow와 synchronization을 증명하고, 공식 guide는 architecture contract를 증명하며, compile artifact는 resource allocation을 증명하고, runtime trace와 counter가 workload에서의 결과를 증명한다. 네 종류의 증거를 바꾸어 쓰지 않는 것이 극도로 디테일한 기술 문서의 기본이다.

마지막으로 기억할 문장은 간단하다. **메모리 최적화는 byte를 무조건 가장 빠른 장소에 넣는 일이 아니다. 다음 소비자에게 필요한 순간까지, 공유 범위와 용량 예산 안에서, 증명 가능한 순서로 살아 있게 만드는 일이다.** 이 관점이 있으면 register, shared, L2, HBM이라는 계층표가 정적인 그림을 벗어나 실제 kernel의 실행 이야기로 바뀐다.

실무에서 이 관점을 유지하는 가장 좋은 방법은 변경 전후에 작은 수명 장부를 쓰는 것이다. 행마다 tensor 또는 fragment 이름, producer, 주소 공간, 공유 범위, 첫 write, 마지막 read, overwrite를 허용하는 synchronization, logical bytes, compile resource 후보를 적는다. tile 또는 stage option을 바꾸면 어느 행의 lifetime과 payload가 달라지는지 먼저 표시한다. 이렇게 하면 “성능을 위해 shared를 늘렸다”는 모호한 변경 설명이 “다음 K/V tile을 이전 QKᵀ MMA와 겹치려고 두 번째 shared stage를 추가했으며, CTA당 payload가 한 타일만큼 늘고 wait 위치가 한 iteration 뒤로 이동한다”는 검증 가능한 설명으로 바뀐다.

코드 리뷰에서도 같은 장부가 유용하다. reviewer는 pointer arithmetic만 확인하지 않고 alias된 두 객체의 lifetime interval이 겹치는지, barrier가 producer와 consumer 전체를 포함하는지, early return이나 divergent branch가 protocol을 깨는지, tail predicate가 source와 destination 양쪽에서 동일한 logical bound를 사용하는지 확인할 수 있다. 성능 reviewer는 추가 fragment가 loop를 가로질러 살아 있는지, stage 증가가 shared residency 경계를 넘는지, 없앤 global intermediate의 logical bytes가 추가한 on-chip state와 산술에 비해 충분히 큰지 질문할 수 있다.

문서 작성자도 표현의 강도를 증거에 맞춘다. source만 보았으면 “이 dataflow를 의도한다”고 쓴다. compiler report가 있으면 “이 build에서 register와 shared를 이만큼 할당했다”고 쓴다. counter까지 있으면 “이 workload에서 이 계층의 byte와 stall이 관측됐다”고 쓴다. 여러 GPU에서 재현했을 때만 architecture나 workload 범위를 넓힌다. 이 단계적 서술은 조심스러운 말투를 위한 장치가 아니라 독자가 같은 판단을 반복할 수 있게 하는 재현성 계약이다.

다음 장에서 cache transaction을 공부할 때도 이 장의 장부가 출발점이다. 어느 global access가 어느 thread coordinate에서 만들어지는지 모르면 coalescing을 계산할 주소 식이 없다. 어느 byte가 shared에서 재사용되는지 모르면 global transaction 감소가 유용한 reuse인지 단순한 cache 우연인지 구분하기 어렵다. 먼저 byte의 소유와 생애를 그리고, 그다음 warp의 주소를 transaction으로 묶는다. 순서를 지키면 미시적인 cache 규칙이 serving 문제와 다시 연결된다.

결국 좋은 최적화 설명은 장소의 이름으로 끝나지 않는다. 이동을 시작한 producer, 완료를 증명하는 synchronization, 실제 소비 instruction, storage를 반환하는 마지막 read, 다음 사용자가 모두 끊김 없이 연결되어야 한다. 이 연결이 있으면 독자는 새로운 kernel을 만나도 같은 방법으로 읽고, 느린 이유와 틀린 이유를 서로 다른 증거로 좁힐 수 있다. 연결이 없으면 register와 shared라는 정확한 단어도 막연한 주문에 머문다.

## 41.11 중간 결산: byte lifetime과 증거 층

이 장에서 가장 중요한 변화는 메모리 계층을 빠르기 순서로 외우지 않고 byte의 생애로 읽게 된 것이다. global address는 물리 HBM 위치를 보장하지 않고, device allocation은 주소 공간과 다른 수명 문제이며, L2는 software가 shared처럼 소유하는 공간이 아니다. register와 shared는 가까운 대신 유한하고, 값을 오래 붙잡거나 tile을 크게 만들수록 동시 실행 예산을 소비한다.

현재 vLLM v0.27.1이 고정한 FlashAttention commit `28e862d…`의 한 forward 함수에서 그 원리를 확인했다. global Q/K/V view와 copy partition을 만들고, async copy가 shared tile을 채우며, wait와 barrier가 소비 가능 시점을 증명한다. MMA fragment와 accumulator는 register pressure를 만들고, online softmax는 전체 score matrix의 global materialization을 피한다. output은 accumulator에서 shared epilogue layout으로 재배열되고 predicated global store로 끝난다.

따라서 최적화 질문은 “어느 메모리가 빠른가”가 아니다. 어느 producer가 byte를 만들고, 어느 scope가 공유하며, 마지막 consumer가 언제 읽고, 어느 synchronization 뒤에 storage를 재사용할 수 있는지 묻는다. 그다음 register 수와 shared bytes를 손으로 계산하고 compile artifact로 확인하며, 마지막에 계층별 traffic과 stall을 workload shape와 함께 관측한다. 이 순서를 지키면 성능 추측과 correctness 검증이 같은 수명표 위에서 만난다.

다음 장에서는 이 global access를 warp의 실제 주소 묶음으로 확대한다. cache line, sector, alignment, coalescing을 다룰 때도 출발점은 같다. 어떤 thread가 어떤 element 주소를 만드는지 먼저 증명하고, 그 뒤에 transaction을 계산한다. byte의 소유와 생애를 모른 채 transaction 숫자부터 세지 않는 것이 두 장을 잇는 원칙이다.

## 41.12 64×64 작은 kernel의 byte-flow를 끝까지 센다

행렬 `C=A×B`의 한 CTA가 64×64 output tile을 계산한다고 하자. K dimension도 64, dtype은 FP16 input과 FP32 accumulator다. 단순 비교를 위해 한 CTA가 A `[64,64]`와 B `[64,64]`를 한 번 읽고 C `[64,64]` FP16을 한 번 쓴다고 가정한다. 실제 tensor core instruction과 tile 분할은 specialization에 따라 달라진다.

global logical loads는 A8,192B, B8,192B, 합 16KiB다. output store는 8KiB다. HBM까지 모두 간다는 상한에서 device-memory traffic은 24KiB다. A/B가 L2에 있고 write allocate/eviction 세부를 제외하면 HBM read는 더 작을 수 있다. source의 global load bytes를 HBM bytes라고 바로 부르지 않는다.

shared staging은 A와 B 각 8KiB, single stage16KiB다. double buffering이면 32KiB이며 padding/skew가 있으면 더 늘어난다. 이 bytes는 CTA lifetime 동안 SM shared allocation을 점유한다. iteration별 copy traffic은 global→shared16KiB이고 threads가 shared에서 MMA fragments로 읽는 내부 traffic은 reuse 횟수와 instruction layout에 따라 훨씬 클 수 있다.

output accumulator logical payload는 64×64×4B=16KiB다. CTA threads256개가 균등하게 소유한다고 단순화하면 thread당 64B, FP32 registers16개다. 주소, fragments, loop state, softmax나 epilogue temporaries는 별도여서 실제 register 수는 더 크다. logical accumulator bytes를 compiler allocated registers와 동일시하지 않는다.

CTA당 FLOPs는 GEMM 기준 `2×64×64×64=524,288`다. HBM 상한 traffic24KiB를 분모로 쓰면 arithmetic intensity 약 21.3 FLOP/B다. output store를 제외한 read-only 분모면 32 FLOP/B다. L2-to-SM input16KiB+store8KiB도 같은 24KiB지만 의미가 다르다. shared internal traffic을 분모에 넣으면 더 낮아진다.

K를 16씩 네 stages로 나누면 stage마다 A2KiB+B2KiB=4KiB copy다. single-stage allocation4KiB, double-stage8KiB지만 총 global logical load는 여전히 16KiB다. stage를 작게 했다고 off-chip bytes가 자동으로 줄지 않는다. overlap, occupancy와 tail/layout이 바뀐다.

shared tile의 reuse를 계산한다. A element 하나는 같은 output row의 64 columns에 기여하고 B element 하나는 64 rows에 기여한다. 이상적인 CTA 내부 reuse64회다. naive kernel이 각 output마다 A/B를 global에서 읽으면 input loads는 `2×64×64×64×2B=1MiB`다. staging은 이를 logical16KiB로 줄여 64배 reuse를 만든다.

하지만 L1/L2가 naive loads 일부를 잡을 수 있어 observed HBM ratio가 64배가 아닐 수 있다. shared staging의 이득은 deterministic CTA reuse와 coalesced/vectorized copy, 비용은 shared capacity·barrier와 copy instructions다. profiler에서 HBM만 비교하면 L2-to-SM traffic과 barrier 비용을 놓친다.

register lifetime도 그린다. A/B fragments는 각 MMA instruction 전에 shared에서 load되어 짧게 살고, C accumulators는 K stages 전체를 가로질러 산다. epilogue scale/bias를 fusion하면 C와 additional values의 live ranges가 겹친다. tile을 키우면 global reuse는 좋아져도 registers/thread가 늘어 spill/occupancy 경계를 넘을 수 있다.

spill 예측은 compiler artifact로 확인한다. thread당 accumulator16 registers 외 fragments/addresses 등 80 registers가 있어 total96이라고 가정한다. CTA256이면 24,576 registers다. SM register capacity와 allocation granularity, max CTAs/warps는 architecture-specific 공식/occupancy 계산에 넣는다. source 숫자만으로 resident CTAs를 확정하지 않는다.

## 41.13 register·shared·L1/L2·HBM lifetime worksheet

register 행은 thread-private logical values, live range와 compiler allocation을 기록한다. source variable이 register라고 보장되지 않고 spills는 local address traffic을 만든다. compile resource report의 registers/thread, local memory bytes와 SASS load/store 후보를 붙인다.

shared 행은 CTA/cluster scope, static/dynamic allocation bytes, stage count, producer copy, completion primitive, reader warps와 last barrier를 기록한다. allocation은 CTA residency 동안 유지된다. tile의 일부를 일찍 다 썼어도 CTA shared allocation capacity가 부분 반환된다고 가정하지 않는다.

L1 행은 hardware-managed cache와 local/global traffic의 관측 좌표다. shared와 unified capacity/config 관계는 architecture와 device setting을 공식 문서에서 확인한다. hit가 높다는 사실은 software가 특정 tile lifetime을 소유했다는 뜻이 아니다. sector/request와 bytes를 함께 본다.

L2 행은 device-wide hardware-managed reuse와 persistence policy 가능성을 기록한다. 다른 CTAs/heads/requests가 같은 K/V lines를 재사용할 수 있지만 eviction/working-set competition이 있다. logical global loads, L2 hit bytes와 HBM bytes를 구분한다.

HBM 행은 physical device-memory traffic의 runtime observation이다. `cudaMalloc` bytes나 global instruction bytes와 같지 않다. read/write throughput, duration과 device peak를 roofline에 쓰되 counter definition과 architecture를 고정한다. peer/host/managed placement는 별 경로다.

작은 kernel worksheet에는 object A tile, B tile, C accumulator, C output을 행으로 둔다. A/B는 global allocation→L2/L1 path→shared stage→register fragment→MMA last use를 갖는다. C는 register accumulator→shared/register epilogue→global store→L2/HBM writeback의 경로다.

각 arrow에 logical bytes와 observed bytes를 분리한다. A logical global load8KiB, observed L2 sector bytes가 padding/alignment 때문에 더 클 수 있고 HBM read는 cache hit 때문에 작을 수 있다. shared loads는 MMA instruction layout/reuse로 logical copy bytes와 다르다.

lifetime terminal도 쓴다. async global→shared copy는 wait 뒤 shared consumer가 읽을 수 있다. shared stage reuse는 all prior readers가 끝난 barrier/event 뒤다. accumulator register는 epilogue consumer 뒤 dead다. output global allocation은 request/model output lifetime을 따른다.

double buffering은 stage0 reader와 stage1 producer overlap을 허용하지만 buffer index/generation을 요구한다. wait group이 wrong stage를 기다리거나 early overwrite하면 finite wrong answer가 난다. 성능 worksheet에도 correctness generation을 붙인다.

CUDA 공식 guide는 주소 공간, scope, synchronization와 architecture 기능의 계약을 제공한다. source는 actual copy/barrier/MMA/store 경로를 제공한다. compiler report는 registers/shared/local allocation을, Nsight counter는 workload traffic/stalls를 제공한다. 네 증거 층을 서로 대신하지 않는다.

## 41.14 roofline 예상과 Nsight 검산을 같은 분모에 놓는다

roofline의 첫 입력은 FLOPs와 특정 memory-level bytes다. 앞 CTA FLOPs524,288과 HBM upper bytes24,576을 쓰면 21.33 FLOP/B다. GPU의 sustained HBM bandwidth를 임의 예로 1.5TB/s라 두면 bandwidth roof는 32TFLOP/s다. compute roof가 120TFLOP/s라면 이 분모에서는 bandwidth-bound 가능성을 예상한다.

그러나 실제 HBM bytes가 L2 reuse로 CTA당 12KiB라면 intensity42.67 FLOP/B, bandwidth roof64TFLOP/s로 오른다. L2-to-SM bytes는 24KiB라면 L2-level intensity는 21.33이다. 하나의 arithmetic intensity가 아니라 memory level별 intensity가 있다. 어느 roof를 말하는지 쓴다.

runtime에서 kernel time10µs, CTAs1,000이면 total FLOPs524,288,000, 약 0.524GFLOP이고 achieved52.4TFLOP/s다. observed HBM bytes12MiB면 bandwidth1.2TB/s, L2 bytes24MiB면 2.4TB/s다. 이 수치는 설명용이며 실제 counter와 duration 단위로 다시 계산한다.

achieved52.4TFLOP/s는 HBM roof64보다 낮고 compute roof120보다 낮다. HBM utilization80%와 compute44%만으로 병목을 확정하지 않는다. barrier, instruction dependency, occupancy와 L2/shared path가 roof gap을 만들 수 있다. roofline은 후보를 줄이는 상한 모델이다.

Nsight Compute에서 확인할 counter group은 compile/resource, launch/occupancy, memory workload analysis, instruction/compute와 warp stall이다. metric 이름은 toolkit/architecture에서 바뀔 수 있으므로 보고서에 exact version과 metric definition을 저장한다. 이름을 기억해 임의로 재구성하지 않는다.

HBM 검산은 DRAM read/write bytes와 duration을 사용한다. 예상 A/B reads16MiB, C writes8MiB for1,000 CTAs라면 logical24MiB다. observed DRAM12MiB read+8MiB write라면 B/A 일부 L2 reuse 가능성을 본다. compression, write behavior, tail/CTA overlap과 counter semantics도 고려한다.

L2 검산은 L2 sector/bytes, hit rate와 request path를 본다. hit rate90%인데 요청량이 10배 늘면 miss bytes는 그대로 클 수 있다. percentage만 보지 않고 absolute bytes를 손계산과 비교한다. prefetch가 사용되지 않은 lines를 가져오면 bytes가 logical보다 클 수 있다.

L1/local 검산은 global/local load/store sectors와 hit, spill-related local traffic을 본다. compile report에 spill stores/loads가 있고 runtime local sectors가 candidate build에서 증가하면 register pressure→spill 가설이 강해진다. local이라는 이름 때문에 on-chip이라고 오판하지 않는다.

shared 검산은 shared load/store throughput, bank conflict/replay 관련 지표와 barrier stall을 본다. logical global→shared copy16MiB가 shared reads16MiB라는 뜻은 아니다. fragments가 tile을 여러 번 읽고 layout transforms/epilogue가 추가 traffic을 만든다. source loop와 instruction counts로 예상 범위를 만든다.

register는 runtime byte counter보다 compiler allocation과 occupancy/warp state로 본다. registers/thread 변화, local spill, achieved occupancy와 eligible warps를 연결한다. occupancy가 낮다고 항상 나쁜 것은 아니지만 latency hiding 부족과 stall 패턴이 함께면 causal candidate다.

barrier/copy pipeline은 async copy wait, barrier stalls와 issue gaps를 source stage에 맞춘다. HBM throughput이 낮아도 warps가 wait/barrier에 묶이면 memory instructions를 충분히 발행하지 못한다. “HBM 사용률이 낮으니 compute-bound”라는 결론을 반증한다.

예상표는 baseline/candidate를 같은 workload shape로 비교한다. logical FLOPs/bytes, compiler registers/shared/local, CTAs/SM 예상, observed HBM/L2/shared/local bytes, duration, stalls와 output parity를 둔다. workload가 다르면 counter 차이를 source 변경에 귀속하지 않는다.

tail shapes도 따로 본다. M/N/K가 tile multiple이 아니면 predicated loads/stores, wasted lanes와 bytes/valid-output가 달라진다. average kernel metric은 common full tiles와 rare tail slow path를 합쳐 first divergence를 숨길 수 있다. specialization/shape labels는 bounded cohort로 둔다.

attention kernel에서는 Q/K/V logical bytes와 score materialization avoided bytes를 계산한다. online softmax가 global score matrix를 없애도 rescale/register/shared traffic과 multiple K/V passes가 있다. decode에서 same KV가 query heads 사이 공유되는 정도와 L2 reuse를 upper/lower bounds로 둔다.

source pin의 async global-to-shared copy, wait/barrier, MMA, online softmax와 epilogue를 counter intervals에 대응시킨다. source fact는 counter 값이 아니라 어느 activity가 가능한지 알려 준다. profiler correlation과 controlled option change로 실제 critical path를 판정한다.
