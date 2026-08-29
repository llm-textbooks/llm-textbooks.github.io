# 54장. CPU page에서 PCIe를 지나 GPU HBM까지: 한 byte의 소유권

모델 shard를 CPU에서 읽은 뒤 `cudaMemcpyAsync`를 호출했다고 하자. 이 한 줄 사이에는 host virtual page, NUMA placement, page lock, DMA address, IOMMU, PCIe root와 switch, GPU copy engine, destination allocation과 stream event가 있다. 복사가 끝난 뒤에도 “L2를 지나 HBM에 저장됐다”는 단순한 파이프 그림만으로 kernel의 첫 load를 설명할 수 없다. 이 장은 8 GiB staging buffer 하나의 주소와 소유권을 끝까지 놓치지 않는다.

## 54.1 COPY-54: async copy가 겹치지 않은 사건부터 조사한다

이 장의 등뼈는 API 목록이 아니라 COPY-54 시간선이다. source buffer의 NUMA placement, pageable/pinned/registered 상태, DMA mapping, PCIe 공유 경로, copy engine queue, ready event, destination consumer를 차례로 확인한다. 각 절은 이 시간선에서 최초로 기대와 실제가 갈린 한 칸을 설명한다.

사건은 model reload 시간 증가로 시작한다. worker는 8 GiB safetensors shard를 host buffer에 읽고 256 MiB chunk 32개로 GPU에 보낸다. copy stream과 compute stream을 나눴고 API 이름도 asynchronous다. 그런데 profiler timeline에서 kernel은 copy가 끝난 뒤에만 시작한다. H2D throughput도 장비 기대치보다 낮다. 운영자는 PCIe link 불량, pageable memory, 잘못된 stream dependency를 동시에 의심한다.

API 호출 이름만으로 어느 가설도 선택하지 않는다. 첫 원장은 다음과 같다.

```text
object: shard R / chunk 7 / [1.75 GiB, 2.00 GiB)
host virtual range와 allocator
physical page NUMA node 분포
pageable | cudaHostAlloc | cudaHostRegister
copy source lifetime owner
GPU BDF·PCI NUMA node·upstream path
destination allocation과 byte interval
copy stream / ready event / consumer stream wait
actual H2D start·end / consumer first read
```

정상 사슬은 source range가 copy 완료까지 살아 있고 page-locked 조건을 만족하며, destination의 같은 interval이 ready event 뒤 consumer에게 공개되는 것이다. copy engine과 kernel을 동시에 실행할 장치 능력, 독립적인 buffer와 stream dependency도 필요하다. page lock은 필요한 조건일 수 있지만 충분한 조건은 아니다.

### 첫 divergence는 API가 아니라 timeline에 있다

fixture를 256 MiB 두 buffer의 ping-pong으로 줄인다. A를 copy하는 동안 B에 다음 file range를 읽고, A copy가 끝나면 consumer가 A destination을 읽는다. pageable allocation과 `cudaMallocHost` allocation을 같은 NUMA node, 같은 size, 같은 stream graph에서 비교한다. 호출 반환 시간, actual DMA interval, CPU staging interval과 kernel interval을 따로 기록한다.

pageable case에서 호출이 오래 block되고 copy/kernel interval이 겹치지 않지만 pinned case에서 겹친다면 source registration state가 첫 차이다. 두 경우 모두 copy interval은 같은데 kernel이 늦게 시작하면 explicit device-wide synchronize, default-stream dependency나 destination ready event graph를 본다. pinned case도 throughput이 낮고 source pages가 remote node라면 pinning 가설은 correctness 조건을 닫았을 뿐 locality 가설은 남는다.

NVIDIA CUDA 12.9.1과 13.x 공식 문서는 CPU memory가 포함된 실제 asynchronous transfer에서 pinned·page-locked host buffer 조건을 강조한다. CUDA 13.0.2의 고정 [`Page-Locked Host Memory`](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-c-programming-guide/index.html#page-locked-host-memory) 좌표는 `cudaHostAlloc`, `cudaHostRegister`, mapped와 portable option을 구분한다. 이를 “pinned이면 무조건 최고 bandwidth”로 확대하지 않는다.

### ownership이 없는 double buffer는 corruption을 만든다

slot A를 H2D copy에 제출한 뒤 file reader가 A를 다음 chunk로 덮으면 DMA가 old/new byte 혼합을 읽을 수 있다. API가 비동기로 반환했다는 것은 source를 즉시 재사용해도 된다는 뜻이 아니다. slot state를 `FREE→FILLING→READY_HOST→COPYING→READY_DEVICE→FREE`로 두고 copy 완료 event가 `COPYING` ownership을 해제한다.

destination도 같다. consumer stream은 해당 chunk ready event를 기다려야 한다. 전체 device synchronize는 correctness를 만들지만 overlap을 없앨 수 있다. chunk별 event는 필요한 interval만 순서화한다. event record stream과 wait stream이 뒤바뀌거나 같은 event object를 다음 chunk에 너무 일찍 재사용하면 shape가 맞는 stale read가 생긴다.

### 경쟁 가설을 작은 실험으로 제거한다

copy size를 4 KiB부터 1 GiB까지 바꿔 작은 transfer overhead와 sustained region을 분리한다. 단일 copy와 두 stream copy를 비교해 copy engine concurrency를 본다. kernel은 destination과 독립된 compute-bound fixture와 같은 HBM을 경쟁하는 memory-bound fixture를 나눠 쓴다. 후자에서 overlap이 줄어든다고 engine 부재로 결론내리지 않는다.

PCIe negotiated speed/width를 확인하되 theoretical envelope를 payload throughput expected와 동일시하지 않는다. remote NUMA, switch 공유, protocol overhead와 host DRAM read가 개입한다. 첫 사건의 수정은 “pinned 사용” 한 줄이 아니라 source pool allocation/fault node, bounded in-flight slots, event lifetime과 consumer dependency를 함께 닫는 것이다.

실제 loader에서는 file reader가 `read()`한 destination과 CUDA copy source가 같은 buffer인지 먼저 본다. file bytes를 Python `bytes` 또는 pageable tensor에 받은 뒤 pinned tensor로 다시 복사한다면 storage→pageable→pinned→device라는 두 host owner가 있다. profiler의 H2D만 빨라져도 CPU memcpy와 두 buffer peak가 load wall time에 남는다. direct-read가 pinned pool을 destination으로 사용할 수 있는지, alignment와 file API가 허용하는지 별도로 확인한다.

8 GiB source와 8 GiB destination, 512 MiB pinned staging 두 slot을 생각하자. whole source가 pageable object로 유지되고 두 staging slot과 device final allocation이 겹치면 logical peak만 17 GiB다. conversion destination이 추가되면 더 늘어난다. “chunk copy라 host peak가 512 MiB”라고 쓰려면 whole source가 streaming read로 남지 않는다는 조건이 필요하다. 객체 lifetime timeline으로 검산한다.

copy와 kernel overlap의 목표도 구체적으로 쓴다. weight load 중 conversion kernel이 chunk 6을 처리할 때 copy engine이 chunk 7을 넣고 storage가 chunk 8을 채우는 구조다. conversion kernel이 chunk 7 destination까지 잘못 넓게 읽으면 dependency가 전체 buffer로 확장돼 pipeline이 직렬화된다. tensor view의 storage 범위와 actual kernel read interval을 맞춘다.

pageable fallback을 검출하는 작은 guard도 유용하다. pool creation 뒤 pointer attribute/registration 결과를 확인하고 effective pool kind를 manifest에 남긴다. copy마다 high-cardinality pointer를 metric label로 내보내지 않고 pool ID와 fallback counter를 둔다. trace에는 문제가 된 chunk generation만 샘플링한다. 설정에 pinned가 적혀 있다는 사실보다 effective pointer state가 중요하다.

완료 조건은 timeline 개선만이 아니다. pinned/pageable 양쪽에서 destination checksum과 final parameter output이 같아야 한다. ping-pong depth를 1,2,3으로 바꿔도 chunk ordering과 bytes가 같아야 한다. concurrency를 늘렸을 때만 corruption이 난다면 source/destination slot lease, event generation과 file offset을 먼저 확인한다.

## 54.2 host virtual page는 어느 NUMA node에 놓이는가

`malloc(8 GiB)`이 성공한 시점에는 virtual address range만 예약됐을 수 있다. anonymous page의 실제 물리 backing은 fault가 일어날 때 policy와 실행 CPU의 locality 영향을 받는다. allocate thread, zero-fill/initialization thread, file read thread와 copy submission thread가 서로 다른 socket에서 실행되면 “GPU worker가 local buffer를 만들었다”는 설명이 틀릴 수 있다.

Linux v6.7 [`What is NUMA?`](https://www.kernel.org/doc/html/v6.7/mm/numa.html)는 기본 local allocation과 remote fallback, task migration이 locality를 깨는 조건을 설명한다. [`NUMA Memory Policy`](https://www.kernel.org/doc/html/v6.7/admin-guide/mm/numa_memory_policy.html)는 task/VMA policy와 bind, preferred, interleave, cpuset의 상호작용을 구분한다. 이 문서의 v6.7 coordinate를 기준으로 쓰며 최신 kernel의 동작을 자동 승계하지 않는다.

### first touch는 slogan이 아니라 fault ledger다

dual-socket fixture에서 GPU 0은 PCI device NUMA node 0, GPU 1은 node 1에 가깝다고 하자. main thread가 node 0 CPU에서 buffer를 allocate하고 모든 page를 zero-fill했다. 이후 GPU 1 worker가 그 buffer를 사용한다. source host DRAM read는 node 0에서 시작해 socket interconnect를 건너 GPU 1 쪽 PCIe root로 갈 수 있다.

원장에는 VMA policy, page별 node sample, faulting CPU, current worker affinity를 적는다. `/proc/<pid>/numa_maps`, `move_pages` query, page fault counter와 sysfs PCI NUMA를 조합한다. 한 도구의 요약 숫자를 절대 truth로 보지 않고 sampling 시각과 page migration 가능성을 기록한다.

first touch라는 표현은 anonymous demand allocation fixture에 유용하지만 모든 mapping에 같은 방식으로 적용되지 않는다. file-backed page cache, MAP_SHARED, copy-on-write, huge page, already faulted allocator arena와 automatic NUMA balancing은 별도 조건이다. `cudaHostRegister`는 기존 page의 placement를 원하는 GPU node로 자동 이동시키는 API가 아니다.

### CPU affinity와 memory policy는 다른 knob다

worker CPU를 node 1에 bind해도 node 0에 이미 놓인 page가 자동으로 옮겨진다고 가정하지 않는다. 반대로 `mbind`로 node 1 policy를 설정해도 submit thread가 node 0에서 실행되면 control path와 memory access가 remote일 수 있다. cpuset이 허용한 memory node 집합이 policy를 제한할 수도 있다.

검사 순서는 process/thread CPU affinity, cpuset allowed CPUs/mems, task/VMA memory policy, actual page placement, PCI device node다. `numa_node=-1`은 device가 node 0이라는 뜻이 아니라 firmware가 locality를 제공하지 못했다는 뜻이다. 이 경우 PCI tree와 socket root complex를 vendor 자료/host inventory로 보강하고 불확실성을 남긴다.

### remote NUMA 사건을 throughput으로만 보지 않는다

같은 pinned buffer를 node 0과 node 1에 각각 fault하고 GPU 1로 copy한다. CPU read bandwidth, socket interconnect traffic, H2D interval과 total load time을 비교한다. local case가 빨라져도 NUMA가 유일 원인이라고 확정하기 전에 page distribution과 link sharing을 동일하게 유지했는지 확인한다.

작은 chunk에서는 registration/event overhead가 차이를 가릴 수 있다. 큰 chunk에서는 PCIe와 host memory가 포화될 수 있다. 여러 GPU가 동시에 local node DRAM을 읽으면 local placement가 맞아도 memory controller 경쟁이 생긴다. `interleave`가 aggregate bandwidth를 높일 수 있는 workload도 있지만 각 GPU copy가 remote page를 섞어 읽게 할 수 있다. 정책은 workload와 topology 가설로 검증한다.

### long-lived pinned pool의 배치 순서

안전한 초기화는 worker affinity와 memory policy를 먼저 확정하고 pool을 allocate/fault한 뒤 registration한다. 그러나 이것이 모든 환경의 유일 순서는 아니다. allocator가 page를 미리 fault했는지, registration이 어떤 page를 요구하는지 확인한다. 시작 시 page distribution을 audit하고 기대와 다르면 serving ready를 막거나 degraded mode를 명시한다.

pool을 너무 크게 pin하면 page reclaim과 다른 process에 압력을 준다. pinning은 GPU bandwidth knob이면서 host memory lifetime 계약이다. request마다 pin/unpin하면 registration overhead와 fragmentation이 커질 수 있다. bounded pool은 재사용성과 pressure 사이의 절충이며 slot lease가 copy 완료까지 유지되어야 한다.

NUMA 사건에서는 평균 node 분포보다 hot range가 중요할 수 있다. 8 GiB pool의 앞 4 GiB만 GPU 1 worker가 반복 사용하고 그 범위가 node 0에 몰렸다면 전체 50:50 분포는 locality를 숨긴다. chunk interval별 page node sample과 transfer timeline을 연결한다. allocator metadata page의 node는 payload locality를 대표하지 않는다.

automatic NUMA balancing이나 explicit migration이 page를 옮길 수 있다. migration이 copy와 동시에 일어나는지, pinned registration 뒤 이동이 제한되는지 해당 kernel/driver 조건을 확인한다. “한 번 local이면 영원히 local”이라고 가정하지 않는다. 장시간 serving에서는 periodic audit와 pool generation 교체가 필요할 수 있다.

file-backed loading은 page cache owner를 추가한다. storage read를 담당한 CPU node의 page cache가 remote에 놓이고 이를 pinned pool로 CPU copy한다면 첫 remote access는 GPU DMA가 아니라 host memcpy일 수 있다. direct storage-to-pinned path가 없을 때 page cache→pinned copy thread의 affinity와 memory bandwidth도 측정한다. I/O 완료 시간과 H2D 시간을 합쳐서 PCIe 문제로 부르지 않는다.

dual socket에서 GPU마다 별도 pool을 둔다면 capacity가 두 배 필요하다. shared pool은 memory를 절약하지만 remote access와 lease contention을 만들 수 있다. model cold load가 드물면 shared pool이 합리적일 수 있고 steady KV offload면 local pool이 유리할 수 있다. policy는 frequency, bytes와 SLO로 결정하며 correctness state는 어느 pool에서도 같아야 한다.

process를 fork한 뒤 pinned pool을 공유하거나 worker를 재시작하는 경우도 lifetime을 확인한다. virtual address inheritance와 CUDA registration/context ownership을 같은 것으로 보지 않는다. 지원되지 않는 공유를 pointer가 보인다는 이유로 사용하지 않는다. 각 process가 registration owner와 cleanup을 명확히 갖거나 IPC가 공식 지원하는 handle을 쓴다.

host OOM/pressure 사건에서 무작정 pin limit을 늘리지 않는다. pool logical use, registered but idle, pending copy lease, failed attempt가 남긴 slot과 allocator cache를 나눈다. request cancellation 뒤 event 완료를 기다리는 slot은 잠시 idle처럼 보여도 안전하게 free할 수 없다. timeout 후 device reset/stream cleanup 정책이 필요하다.

## 54.3 pageable·pinned·registered·mapped는 서로 다른 상태다

“CPU memory”라는 하나의 이름 아래 네 상태가 있다. pageable allocation은 일반 host virtual memory다. allocated pinned는 CUDA가 page-locked range를 새로 만든다. registered는 외부 allocator가 만든 기존 range를 CUDA에 등록한다. mapped pinned는 device address space에서 접근할 수 있도록 mapping option을 가진다. 각 상태는 allocation owner, NUMA placement 시점, 해제 API와 device access path가 다르다.

### `cudaHostAlloc`와 `cudaHostRegister`의 lifetime

`cudaHostAlloc` 결과는 `cudaFreeHost`까지 owner가 유지한다. `cudaHostRegister`는 기존 allocation의 lifetime 위에 registration lifetime을 얹는다. unregister 전에 pending copy나 kernel mapped access가 끝나야 하고, underlying allocation을 먼저 free하면 안 된다. error path에서 register 성공 여부와 partial pool state를 추적한다.

Mooncake의 고정 source는 page-aligned host region을 `cudaHostRegister`하고 이미 등록됨과 일반 오류를 구분하는 실제 예를 제공한다. 이 source는 end-to-end zero-copy를 증명하지 않는다. 등록 함수가 성공했다는 사실은 이후 transport와 GPU destination path가 무엇인지 보장하지 않는다.

portable registration은 여러 device context에서 이점을 사용할 조건과 관련된다. 이를 모든 GPU에 같은 NUMA locality가 생긴다고 해석하지 않는다. page는 여전히 어떤 host node에 있다. mapped option도 HBM residency를 뜻하지 않는다.

### mapped host memory는 HBM copy의 대체가 아닐 수 있다

kernel이 mapped host pointer를 읽으면 GPU가 host memory를 interconnect를 통해 접근할 수 있다. 작은 control data나 한 번만 읽는 payload에는 explicit copy를 피할 수 있지만, 반복 재사용하는 weight/KV에는 PCIe latency와 bandwidth가 kernel hot path에 남는다. HBM에 한 번 copy해 여러 번 읽는 경로와 손익이 다르다.

원장에는 host pointer와 device-visible pointer, physical backing, consumer load count를 쓴다. Unified Virtual Addressing으로 pointer value가 같아 보일 수 있어도 물리 location과 page migration/traffic을 확인한다. “zero-copy”는 CPU memcpy가 없는지, host staging이 없는지, GPU HBM copy가 없는지 범위를 붙여 말한다.

### write-combining과 CPU read cost

CUDA 문서는 write-combining pinned memory가 PCIe transfer에 유리할 수 있지만 CPU read에는 불리할 수 있음을 설명한다. CPU가 file bytes를 쓰고 GPU로 보내기만 하는 staging과 CPU가 tokenizer/transform을 위해 다시 읽는 buffer를 구분한다. option을 켜기 전에 producer/consumer 방향을 원장에 적는다.

performance claim은 장비와 transfer pattern에서 측정한다. 문서의 조건부 수치를 모든 platform에 기대값으로 쓰지 않는다. CPU cache snoop, host architecture와 PCIe path가 다를 수 있다.

### registration failure를 fallback과 혼동하지 않는다

pin limit, alignment, overlapping registration, resource pressure로 registration이 실패할 수 있다. loader가 pageable fallback을 허용하면 correctness는 유지돼도 overlap과 latency가 달라진다. fallback을 metric과 log에 명시한다. silent fallback 뒤 “async path enabled” 설정만 보고 성능을 진단하지 않는다.

fail-fast가 필요한 workload도 있다. decode offload restore가 latency SLO를 만족하려면 pageable fallback을 거부할 수 있다. model cold load처럼 느려도 진행하는 것이 나은 경우에는 degraded path를 허용할 수 있다. policy와 mechanism을 분리한다.

등록 비용을 batch한다는 말도 range lifetime과 맞물린다. 작은 request buffer 수천 개를 각각 register하면 page walk와 driver bookkeeping이 커질 수 있다. 큰 pool을 한 번 등록하고 subrange lease를 나누면 control overhead를 줄일 수 있지만 한 subrange overflow가 다른 request를 침범하지 않도록 allocator boundary가 필요하다. unregister는 마지막 subrange lease와 pending DMA 뒤에만 한다.

registered range와 file/direct IO alignment 요구가 다를 수 있다. page alignment를 맞췄다고 storage direct IO의 block alignment와 GPU tensor offset이 모두 맞는 것은 아니다. storage가 padding을 읽은 뒤 useful subrange만 H2D하는지, padding까지 copy하는지 byte ledger에 쓴다. useful payload throughput과 physical read/copy bytes를 나눈다.

mapped host memory를 serving weight에 쓰는 실험은 access count를 손계산한다. 256 MiB table에서 decode마다 1 MiB만 불규칙하게 한 번 읽는다면 whole H2D를 피할 가능성이 있다. 같은 256 MiB weight를 모든 token GEMM이 반복 읽으면 host link가 hot path 병목이 된다. kernel이 host mapping을 지원하는지, access granularity와 cache behavior는 device별로 검증한다.

CPU write-combining buffer에 checkpoint를 decode/validate하려고 다시 읽는 path는 불리할 수 있다. producer가 storage DMA인지 CPU parsing인지도 중요하다. option을 pool 전체에 일괄 적용하기보다 upload-only, bidirectional metadata와 CPU-transform pool을 역할별로 나눌 수 있다. pool 수 증가에 따른 capacity와 관리 복잡성을 함께 계산한다.

portable flag는 multi-GPU pool 관리에 편리할 수 있지만 device별 context와 topology를 감추지 않는다. GPU 0에서 만든 portable pool을 GPU 3도 사용할 수 있다는 API 조건과 GPU 3에 local하다는 물리 조건은 별개다. actual page node와 각 GPU route를 비교한다.

## 54.4 DMA address·IOMMU·BAR를 한 주소로 합치지 않는다

CPU virtual address, CPU physical address와 device가 bus에서 사용하는 DMA address는 같은 개념이 아니다. Linux v6.7 [`Dynamic DMA mapping Guide`](https://www.kernel.org/doc/html/v6.7/core-api/dma-api-howto.html)는 CPU virtual/physical/bus address 구분과 DMA mask, coherent/streaming mapping, map/unmap ownership을 설명한다. CUDA application이 kernel DMA API를 직접 호출한다는 뜻은 아니지만 driver 아래 경계를 이해하는 근거다.

### DMA mapping은 device-visible lifetime을 만든다

page lock은 page가 transfer 중 사라지거나 swap/migrate되는 위험을 제어하는 한 조건이다. driver는 device가 접근할 I/O address mapping을 준비한다. IOMMU가 있으면 I/O virtual address가 page table을 통해 host physical page로 번역될 수 있다. 정확한 NVIDIA driver batching, ATS/PASID policy는 공개 source 없이 단정하지 않는다.

mapping owner는 DMA가 끝나기 전에 unmap하지 않는다. CPU와 device가 같은 streaming buffer를 번갈아 쓰는 low-level API에서는 ownership synchronization이 필요하다. CUDA stream/event가 application에 제공하는 ordering과 kernel DMA API의 sync primitive를 같은 API라고 섞지 않는다.

### IOMMU on/off는 단일 성능 스위치가 아니다

IOMMU는 isolation과 address translation을 제공한다. 켜져 있다는 이유만으로 모든 P2P가 막힌다고 쓰지 않고, 꺼져 있다는 이유만으로 peer route가 된다고도 쓰지 않는다. platform, device capability, driver와 virtualization policy가 함께 결정한다. Linux v6.7 [`IOMMU Userspace API`](https://www.kernel.org/doc/html/v6.7/userspace-api/iommu.html)는 guest SVA/IOVA UAPI의 범위를 보여 주지만 일반 CUDA bare-metal path의 직접 API로 승계하지 않는다.

incident record에는 boot/kernel IOMMU mode, virtualization/container context, effective peer capability와 transport 결과를 남긴다. 성능 비교를 위해 isolation을 무작정 끄는 운영 조치를 권하지 않는다. 보안·device ownership 변경은 별도 승인과 platform 문서가 필요하다.

### BAR와 payload path를 구분한다

PCI BAR는 device가 노출한 register 또는 memory aperture를 system address space에 배치하는 resource다. BAR1 aperture와 GPU HBM capacity는 같은 숫자가 아니다. 모든 H2D payload를 CPU가 BAR에 store해 옮긴다고 설명하지 않는다. DMA copy, mapped host access, peer memory access와 control register access를 구분한다.

large BAR/Resizable BAR 설정이 보인다고 copy engine throughput이 자동 증가한다고 결론내리지 않는다. 어떤 API path가 aperture를 소비하는지, driver/device 조건과 실제 profiler evidence를 확인한다. configuration space 한 field를 end-to-end data path로 과장하지 않는다.

### 주소 ledger로 corruption을 좁힌다

chunk마다 host virtual interval, registration handle/generation, destination device interval과 event generation을 기록한다. physical/DMA address 자체를 일반 application log에 노출할 필요는 없고 보안상 피할 수 있다. 중요한 것은 같은 lease generation이 map→submit→complete→unmap 순서를 지키는지다.

재사용된 host virtual address가 이전 registration record와 우연히 같을 수 있다. pointer value만 cache key로 쓰면 ABA 문제가 생긴다. allocation/registration generation과 length를 함께 사용한다. destination allocator도 같은 address를 재사용할 수 있으므로 event가 old generation을 ready로 표시하지 않게 한다.

DMA mask와 addressability는 driver가 device에 맞춰 확인하는 low-level 계약이다. application은 `cudaHostRegister` 성공을 통해 더 높은 수준의 결과를 받지만 실패 log를 “메모리 부족” 하나로 뭉개지 않는다. invalid range, unsupported mapping, resource exhaustion과 context error를 분리한다. 원인별로 retry가 안전한지도 다르다.

IOMMU translation overhead를 논할 때 mapping setup과 steady DMA를 나눈다. long-lived pinned pool은 setup을 amortize할 수 있다. per-request map/unmap은 control latency가 보일 수 있다. 실제 translation cache나 ATS behavior를 공개 근거 없이 상상하지 않고 registration latency, transfer active interval과 CPU usage를 측정한다.

virtual machine이나 device passthrough에서는 guest-visible topology와 host physical topology가 다를 수 있다. guest `numa_node`와 vCPU pinning, host device root를 모두 확인한다. container는 보통 host kernel을 공유하지만 cgroup/cpuset과 device namespace가 관측을 제한할 수 있다. deployment layer를 ownership 원장에 포함한다.

BAR-related 설정을 바꾸는 것은 안정성·보안과 firmware compatibility를 포함한다. 단순 benchmark 조치로 BIOS 값을 권하지 않는다. 먼저 current configuration, effective API path와 vendor support matrix를 확인한다. 설정 변경 전후에는 correctness, peer path와 all devices enumeration을 함께 검증한다.

주소 오류 fixture는 두 allocation을 인접시켜 offset boundary를 검사한다. 마지막 4 KiB chunk, zero-length tail과 non-page-aligned subrange를 포함한다. registered parent range 안의 subrange가 length를 넘지 않는지, destination offset과 source file offset이 같은 generation에 속하는지 assert한다. large happy-path copy만으로 edge를 증명하지 않는다.

## 54.5 PCIe topology는 link 숫자가 아니라 공유 경로다

GPU의 `current_link_speed`와 `current_link_width`를 읽으면 endpoint link의 negotiated 상태를 알 수 있다. 그러나 CPU DRAM에서 GPU까지의 전체 경로는 endpoint 하나가 아니다. CPU socket의 root complex, root port, 외부 switch, 다른 endpoint와 공유하는 upstream link, retimer와 firmware routing이 있을 수 있다. `lspci -t` tree와 sysfs device ancestry를 함께 읽는다.

### BDF에서 root까지 역으로 올라간다

GPU BDF가 `0000:65:00.0`이라고 하자. sysfs real path를 따라 upstream bridges를 기록하고 각 link의 current/max speed와 width를 확인한다. GPU와 NIC 또는 다른 GPU가 어느 switch 아래 있는지 그린다. PCI device의 `numa_node`와 CPU node distance를 붙인다. topology tool의 `PIX/PXB/PHB/SYS` 같은 축약은 해당 tool version 정의를 확인해 해석하고 보편 용어처럼 쓰지 않는다.

endpoint가 x16으로 연결돼도 switch upstream이 x16 하나이고 GPU 두 개가 동시에 전송하면 aggregate 경로를 공유할 수 있다. 단독 benchmark는 빠르지만 동시 model load에서 절반으로 떨어지는 사건은 endpoint downgrade가 아니라 upstream contention일 수 있다. 각 GPU H2D bytes와 root/switch aggregate를 같은 시간축에서 본다.

반대로 endpoint가 max보다 낮은 width로 negotiate됐다면 payload optimization 전에 hardware/slot/firmware 상태를 조사한다. 그러나 낮은 utilization을 보았다는 이유만으로 width가 낮다고 추측하지 않는다. 작은 copy, pageable staging이나 synchronization도 link를 채우지 못한다.

### theoretical bandwidth와 useful payload를 나눈다

PCIe generation·lane의 raw signaling 수치에서 protocol encoding과 packet overhead를 고려한 envelope를 계산할 수 있다. 실제 `cudaMemcpy` payload는 direction, request size, host memory supply, switch arbitration과 engine에 좌우된다. 책에서는 특정 세대 수치를 암기표로 늘어놓기보다 장비가 보고한 negotiated link와 measured payload의 차이를 계산하는 방법을 제시한다.

예를 들어 8 GiB copy가 0.5초면 application payload는 약 16 GiB/s다. profiler가 실제 DMA interval 0.4초를 보이면 engine-active payload는 20 GiB/s이고 나머지 0.1초는 staging/event/queue일 수 있다. wall time과 active transfer time을 구분한다. 같은 link에서 32개의 256 MiB chunk 사이 gap이 크다면 link capacity보다 pipeline owner를 본다.

duplex도 주의한다. CPU offload는 D2H와 restore H2D가 동시에 있을 수 있다. 양방향 copy engine과 link가 어느 정도 overlap하는지는 장치·topology와 traffic에 달렸다. 방향별 throughput을 따로 측정하고 aggregate 숫자를 단방향 기대치와 비교하지 않는다.

### switch와 root 공유 사건

GPU 0과 GPU 1을 개별로 copy하면 각각 24 GiB/s인데 동시에 하면 합계가 28 GiB/s라고 하자. 두 GPU가 같은 switch upstream을 공유한다면 정상 contention 후보다. GPU 2가 다른 root 아래에서 동시에도 24 GiB/s를 유지하면 근거가 강해진다. NUMA page node와 source DRAM channel을 동일하게 하지 않으면 host memory contention이라는 경쟁 가설이 남는다.

fixture를 네 조합으로 만든다. 같은 root/같은 NUMA DRAM, 같은 root/분리 DRAM, 다른 root/같은 DRAM, 다른 root/분리 DRAM이다. 실제 platform이 모든 조합을 제공하지 않으면 가능한 범위와 한계를 적는다. link counter가 없다면 timeline과 topology의 상관을 inference라고 명시한다.

### topology inventory를 serving placement로 바꾼다

model loading workers를 GPU별 local NUMA CPU와 memory node에 배치한다. 동시에 load하는 GPU가 shared upstream을 과포화하면 wave를 나누거나 chunk scheduling을 조정한다. 이는 correctness 요구가 아니라 startup/goodput 정책이다. 잘못 추정해도 output byte가 바뀌어서는 안 된다.

CPU offload와 KV transfer는 steady-state ITL에 들어오므로 더 엄격하다. GPU·NIC·CPU pool이 가까운 topology를 선택하고 remote route가 불가피하면 SLO/capacity에 반영한다. topology는 device index 순서로 추측하지 않는다. 컨테이너 안의 visible index와 host BDF mapping을 기록한다.

placement planner의 입력을 구체화한다. GPU UUID/BDF, PCI NUMA node, root/switch ancestry, local CPU set과 memory nodes, NIC ancestry, shared upstream group을 inventory로 만든다. output은 worker CPU affinity, pool memory policy, GPU assignment와 concurrent-load wave다. inventory generation을 deployment manifest에 남겨 hotplug나 firmware 변경 뒤 stale placement를 재사용하지 않는다.

device ordinal은 process마다 `CUDA_VISIBLE_DEVICES` 순서로 다시 매겨질 수 있다. rank 0이 항상 BDF가 작은 GPU라는 가정도 안전하지 않다. NCCL rank, serving worker rank, CUDA ordinal, UUID와 BDF를 명시적으로 연결한다. 장애 log가 `GPU 1`만 남기면 어느 physical route였는지 복원할 수 없다.

MIG 또는 virtual function 환경에서는 logical device가 physical GPU resources를 나눌 수 있다. PCI function과 memory/copy resource sharing을 product별 공식 문서에서 확인한다. 하나의 BDF 아래 instance가 있다고 독립 PCIe link를 가진다고 계산하지 않는다. 다른 tenant traffic이 보이지 않는 경우 관측 한계를 적는다.

링크 downgrade 사건은 부팅 시점과 load 시점을 나눈다. max/current speed·width, correctable error와 retraining evidence가 있으면 hardware 운영 절차로 넘긴다. application이 link retrain을 임의로 수행하지 않는다. 단순히 throughput이 낮고 current link가 정상이라면 software pipeline으로 돌아간다.

NUMA node와 root가 가까워도 CPU memory controller channel utilization이 포화될 수 있다. 두 GPU가 같은 node DRAM에서 읽으면 PCIe upstream과 host DRAM 두 공유 자원이 있다. GPU마다 별도 node pool을 둘 수 없는 single-socket host에서는 page interleave나 load staggering을 workload로 검증한다. topology 최적화가 새로운 DRAM bottleneck을 만들 수 있다.

model load와 online offload traffic이 같은 root/link를 공유하면 cold deployment가 live ITL을 흔들 수 있다. scheduler는 load bandwidth budget과 online transfer 우선순위를 분리한다. chunk 사이 yield 또는 rate limit은 startup을 늦추지만 serving tail을 보호할 수 있다. correctness ordering과 QoS policy를 섞지 않는다.

## 54.6 copy engine·stream·event가 ready 시점을 정한다

H2D API가 submission을 반환한 뒤 실제 DMA가 언제 시작하고 끝나는지는 queue와 dependency에 달렸다. source가 pinned여도 앞선 stream operation을 기다리거나 engine이 다른 transfer로 바쁘면 즉시 시작하지 않는다. destination consumer도 같은 stream이면 stream order를 따르지만 다른 stream이면 event/wait가 필요하다.

### 세 시각을 분리한다

`submit_return`, `dma_start/end`, `consumer_first_read`를 기록한다. CPU wall clock으로 API 앞뒤만 재면 async copy의 실제 duration을 알 수 없다. CUDA event는 stream work의 timestamp를 제공하지만 host/file IO와 registration 시간을 포함하지 않는다. profiler timeline과 application ledger를 request/chunk ID로 맞춘다.

submit이 늦으면 pageable staging, allocation, queue backpressure나 host lock을 본다. submit은 빠른데 DMA start가 늦으면 stream dependency와 engine queue를 본다. DMA end는 빠른데 consumer가 늦으면 event wait, graph schedule나 kernel queue를 본다. “H2D latency” 하나를 세 원인에 공통으로 쓰지 않는다.

### default stream과 device synchronize 함정

legacy/default stream semantics와 framework stream context가 어떤 ordering을 만드는지 version/source로 확인한다. 편의를 위해 `cudaDeviceSynchronize`를 넣으면 source 재사용 안전성은 생기지만 모든 engine/compute overlap을 막을 수 있다. chunk event로 필요한 dependency만 표현한다.

framework tensor `.to(device)`나 loader helper가 내부 stream을 선택할 수 있다. outer code가 만든 event를 잘못된 stream에 record하면 copy가 제출되기 전에 ready가 표시될 수 있다. effective current stream, copy stream과 consumer stream을 trace에 남긴다. Python 호출 순서가 device execution 순서와 같다고 가정하지 않는다.

### chunk pipeline의 손계산

256 MiB chunk file read가 12 ms, H2D가 10 ms, conversion kernel이 6 ms라고 하자. 완전 직렬이면 chunk당 28 ms, 32개면 896 ms다. 세 stage가 독립 buffer/engine에서 안정적으로 pipeline되면 초기 fill과 drain을 제외한 steady-state는 가장 느린 12 ms에 가까워질 수 있다. 실제로는 storage와 host DRAM, H2D와 conversion이 memory bandwidth를 공유해 이 하한에 도달하지 않을 수 있다.

buffer 두 개만 있으면 세 stage 가운데 누가 slot을 오래 소유하는지 본다. file read A, H2D B 동안 conversion C까지 겹치려면 세 destination/source lease가 필요할 수 있다. slot 수를 늘리면 overlap 가능성과 pinned/resident peak가 함께 증가한다. `workers×chunk`로 peak를 계산하고 bounded queue를 둔다.

chunk를 4 MiB로 줄이면 2,048 events와 submissions가 생긴다. link utilization이 떨어질 수 있다. 1 GiB로 늘리면 네 chunk라 pipeline 기회와 load balancing이 줄고 pinned pool peak가 커진다. size sweep에서 throughput, gap, event count와 peak를 함께 본다.

### copy engine 수를 과장하지 않는다

device property나 profiler가 여러 copy engine 가능성을 보여도 특정 방향/메모리 path가 모두 독립이라는 뜻은 아니다. H2D 두 개, H2D+D2H, peer와 host transfer를 각각 측정한다. engine count를 stream count와 동일시하지 않는다. stream을 16개 만든다고 link bandwidth가 16배가 되지 않는다.

memory-bound kernel과 H2D가 동시에 HBM destination을 쓰면 device memory subsystem에서 경쟁할 수 있다. overlap percentage가 낮아져도 PCIe copy engine 고장이 아니다. independent compute-bound kernel fixture로 engine concurrency를 가르고 production kernel과의 resource competition을 별도로 측정한다.

### ready event의 세대

event pool을 재사용할 때 chunk generation을 붙인다. slot 0 event가 chunk 0 완료를 나타낸 뒤 chunk 2에 재record되었다면 늦게 도착한 consumer가 어느 generation을 기다리는지 명확해야 한다. high-level future가 event object만 보유하고 generation을 잃으면 ABA 문제가 생긴다.

error와 cancellation에서도 pending event와 slot owner를 정리한다. copy가 진행 중이면 source pool을 바로 free/unregister하지 않는다. stream synchronize 또는 driver가 보장하는 cancellation/cleanup 경계를 기다린다. 부분 destination을 model registry나 KV state에 publish하지 않는다.

timeline 분석에는 CPU thread state도 붙인다. copy submission 사이 5 ms gap이 있을 때 thread가 storage future를 기다렸는지, GIL/lock에 막혔는지, allocator/register를 수행했는지 구분한다. GPU profiler에 빈 구간만 보여도 GPU가 원인이라는 뜻은 아니다. CPU scheduling trace와 application spans를 chunk ID로 맞춘다.

event를 너무 많이 만들고 파괴하면 host-side overhead가 커질 수 있다. event pool을 쓰되 세대 안전성을 지킨다. timing이 필요 없는 production dependency event와 진단 timing event를 구분할 수 있다. 진단을 켰을 때 overhead가 pipeline을 바꾸는지 control run으로 확인한다.

priority stream이 copy engine arbitration을 원하는 방식으로 보장한다고 가정하지 않는다. CUDA 공식 semantics와 device behavior 범위 안에서 사용한다. high-priority compute가 memory system을 점유하면 low-priority H2D가 느려질 수 있고 반대도 가능하다. priority option의 effective stream과 timeline을 직접 본다.

CUDA Graph에 memcpy node가 들어가면 source/destination address와 size update 규칙이 graph contract에 들어간다. pool slot 주소가 바뀌거나 tail chunk size가 다르면 update 가능 여부와 fallback을 확인한다. graph launch가 성공했다고 old pointer가 아닌지 generation ledger로 검증한다. captured graph의 static lifetime 때문에 pool을 해제하지 못하는 hidden owner도 기록한다.

multi-process service에서는 각 process copy stream과 context가 같은 GPU engine을 경쟁한다. process 하나의 stream 수만 조절해 전체 concurrency를 알 수 없다. MPS나 scheduling mode가 있으면 공식 support 범위를 확인한다. worker별 bytes/interval과 device aggregate를 함께 본다.

작은 control copy와 큰 weight copy를 같은 stream FIFO에 놓으면 control이 큰 transfer 뒤에서 기다릴 수 있다. 별도 stream/priority 또는 chunk boundary를 고려하되 destination dependency가 올바른지 확인한다. latency-sensitive KV metadata와 bulk model load를 분류한다. 너무 많은 분리는 engine queue와 event overhead를 늘릴 수 있다.

copy completion callback에서 heavy CPU work를 수행하면 다음 submission이 지연될 수 있다. callback은 slot state 전이와 queue signal만 하고 parsing/logging을 다른 worker에 넘길 수 있다. 하지만 callback과 pool owner 사이 lock-free 설계를 도입하기 전에 ABA generation과 shutdown ordering을 test한다.

failure cleanup fixture는 마지막 chunk에서 error를 내는 경우가 중요하다. 앞 31개 destination interval이 채워졌어도 parameter는 incomplete다. coverage/publish barrier가 전체 32개 ready generation을 요구해야 한다. retry가 같은 destination을 재사용한다면 모든 interval readiness와 conversion state를 reset하거나 새 allocation을 쓴다.

## 54.7 destination pointer가 곧 HBM traffic을 뜻하지는 않는다

H2D destination은 device allocation의 logical byte range다. 데이터센터 GPU에서 그 allocation의 대표 물리 backing은 HBM이지만 CUDA global address space와 HBM이라는 물리 memory 기술은 같은 용어가 아니다. managed/mapped host memory처럼 global address로 접근하지만 residency가 다른 반례가 있다.

### copy completion과 visibility

같은 stream consumer는 앞선 copy 완료 뒤 실행된다. 다른 stream consumer는 event dependency가 필요하다. ready는 destination byte가 consumer에게 유효하다는 software happens-before다. 이 계약이 없으면 kernel이 old 또는 partial data를 읽을 수 있다. cache flush를 임의로 넣기 전에 API memory ordering을 확인한다.

copy engine이 device memory에 write한 뒤 kernel load가 L2 hit인지 HBM access인지는 architecture, cache state와 competing traffic에 달렸다. “PCIe→L2→HBM”을 byte가 반드시 세 창고에 차례로 완전 복제되는 흐름으로 그리지 않는다. L2는 여러 SM/device traffic 앞의 cache이고 HBM controller는 backing memory request를 처리한다.

### L2 hit와 HBM byte를 분리한다

kernel logical load byte와 HBM traffic은 같지 않다. L2 reuse가 있으면 HBM read가 줄 수 있고, poor coalescing·writeback·spill과 cache miss가 있으면 useful tensor byte보다 traffic이 커질 수 있다. 42장의 sector/coalescing metric과 연결하되 PCIe payload와 섞지 않는다.

H2D 8 GiB가 끝난 뒤 conversion kernel이 source를 한 번 읽고 destination을 8 GiB 쓴다면 최소 algorithmic device traffic을 손계산한다. dtype 변환으로 size가 달라질 수 있고 read-modify-write나 temporary가 더해질 수 있다. profiler HBM-related metric과 비교해 first kernel bottleneck을 분석한다. H2D throughput이 높아도 conversion이 HBM-bound면 load wall time은 여전히 길다.

### HBM channel/bank를 source 없이 상상하지 않는다

HBM stack, channel과 controller가 parallel bandwidth를 제공한다는 공개 architecture 설명은 사용할 수 있다. 그러나 physical address의 어느 bit가 어느 channel/bank를 고르는지, exact burst와 scheduler policy는 공식 공개 근거 없이 적지 않는다. pointer alignment 하나로 특정 channel에 몰린다고 단정하지 않는다.

관측자는 achieved device-memory bandwidth, L2 hit/miss/sector, memory stall과 kernel duration을 본다. address distribution 가설이 필요하면 microbenchmark와 vendor profiler 증거를 함께 둔다. 결과를 해당 GPU/driver에 한정한다.

### CUDA 12.x와 13.x를 물리 topology 차이로 쓰지 않는다

같은 host와 GPU에서 toolkit 12.9.1을 13.x로 바꾼다고 PCIe lane, copy engine 개수나 HBM channel이 물리적으로 변하지 않는다. 달라질 수 있는 것은 supported platform/architecture, API·compiler·library와 profiler behavior다. release 문서에서 확인된 차이만 옵션/지원 gate로 쓴다.

CUDA 13.x 문서가 page-locked, mapped와 async 조건을 재구성하거나 새 API 설명을 추가해도 12.x binary의 실제 driver compatibility와 device capability를 별도로 확인한다. toolkit version 문자열만 보고 serving framework의 prebuilt wheel이 어느 CUDA runtime/library를 포함하거나 요구하는지 추측하지 않는다.

### load path를 HBM consumer까지 닫는다

weight loader는 destination tensor에 copy한 뒤 dtype/quant conversion, transpose 또는 packing kernel을 실행할 수 있다. 원장에는 intermediate device allocation과 last consumer를 넣는다. source FP16 8 GiB와 packed 4-bit destination 2 GiB가 동시에 살아 peak 10 GiB 이상이 될 수 있다. workspace와 allocator padding은 별도다.

conversion 완료 뒤 source allocation을 해제하고 model parameter coverage를 publish한다. copy 완료만으로 final weight ready가 아니다. consumer kernel이 packed scale/metadata까지 필요하면 parameter group 전체의 ready event가 있어야 한다.

device allocation의 물리 residency를 관찰할 때 managed memory와 explicit device memory를 구분한다. managed pointer는 page migration과 fault/prefetch가 있을 수 있다. H2D API를 호출했다는 사실만으로 모든 page가 같은 방식으로 resident한다고 가정하지 않는다. allocation kind와 advise/prefetch state를 manifest에 둔다.

CPU offload restore는 destination이 기존 KV block일 수 있다. block table이 아직 다른 request를 가리키거나 copy-on-write tail이 공유 중이면 correct bytes를 wrong owner에게 쓴다. copy 전 allocator lease와 request generation을 확인하고 ready 뒤 block table을 atomic publish한다. address가 valid하다는 사실과 ownership이 valid하다는 사실은 다르다.

L2 persistence/access-policy window 같은 CUDA 기능을 사용한다면 설정한 range와 actual working set, 동시 stream 경쟁을 검증한다. H2D 직후 weight 전체가 L2에 남는다고 기대하지 않는다. L2 capacity보다 큰 8 GiB shard는 특히 그렇다. persistence option이 다른 tenant/kernel working set을 밀어낼 수 있다.

HBM bandwidth metric은 read/write와 compression/partition effect를 profiler 정의에 따라 해석한다. 제품 spec의 peak TB/s와 application useful bytes/time를 직접 비교해 효율을 계산할 때 ECC, clock, partition와 metric domain 차이를 적는다. exact channel utilization을 제공하지 않는 metric에서 channel imbalance를 확정하지 않는다.

kernel이 destination을 처음 읽기 전에 validation checksum kernel을 돌리면 그 kernel이 cache/HBM state를 바꿀 수 있다. 관찰이 production first-touch를 교란한다. 작은 sampled range 또는 별도 diagnostic run을 사용하고 checksum overhead를 timeline에 표시한다. correctness guard와 성능 benchmark mode를 구분한다.

device-to-host result copy도 같은 원칙을 반대로 적용한다. source kernel completion→D2H submission→pinned destination→CPU consumer ordering이 필요하다. CPU가 event 전에 buffer를 읽으면 stale result다. H2D 장의 사슬을 방향만 뒤집되 CPU cache visibility와 consumer synchronization을 공식 API semantics로 확인한다.

HBM OOM과 host pinned OOM은 서로 다른 allocator 사건이다. HBM pressure 때문에 offload를 늘리면 pinned pool과 PCIe traffic이 커져 host pressure/ITL을 악화할 수 있다. option을 한 memory tier의 해결로만 평가하지 않고 end-to-end byte ledger에서 이동된 pressure를 본다.

## 54.8 GPU P2P가 host staging으로 돌아오는 순간

GPU 0의 KV 4 GiB를 GPU 1로 옮긴다고 하자. direct peer path를 기대했지만 ITL이 늘고 host pinned pool 8 GiB가 새로 보인다. topology log에는 P2P enabled가 찍혔다. “NVLink가 고장”이나 “IOMMU 때문” 중 하나를 바로 고르지 않는다.

### direct peer라는 말의 범위를 붙인다

peer access가 가능하면 GPU가 다른 GPU memory를 접근하거나 peer copy를 수행할 수 있다. 실제 physical route는 NVLink/NVSwitch 또는 PCIe topology일 수 있다. platform/virtualization과 driver policy가 지원을 제한할 수 있다. capability query는 가능성을 말하고 profiler/effective transport는 이번 transfer가 택한 경로를 말한다.

source/destination BDF, peer capability, selected API/transport, bytes와 timeline을 기록한다. NVLink counter, PCIe traffic과 host D2H/H2D를 함께 보면 경로 가설을 좁힐 수 있다. counter availability와 정확한 의미는 장비/tool version에 한정한다.

### staging fallback은 두 개의 copy다

direct path가 없으면 GPU 0→host slot D2H와 host slot→GPU 1 H2D가 필요할 수 있다. 한 4 GiB slot을 직렬로 쓰면 두 transfer 합이 latency 하한이다. 여러 chunk와 두세 slot로 pipeline하면 D2H/H2D를 overlap할 가능성이 있지만 host memory와 PCIe root를 공유할 수 있다.

slot state는 `FREE→D2H→HOST_READY→H2D→FREE`다. source GPU event, destination ready event와 host lease가 필요하다. D2H 완료 전에 H2D를 시작하거나 H2D 완료 전에 slot을 source D2H에 재사용하면 corruption이 난다. 두 device context의 event/future를 한 owner가 조정한다.

host slot NUMA placement는 양쪽 GPU 중 어느 쪽에 가까울지 선택 문제를 만든다. GPU들이 다른 socket이면 한 방향은 remote가 될 수 있다. interleave 또는 두 pool을 direction별로 두는 정책을 측정한다. “pinned” 하나로 locality가 해결되지 않는다.

### GPUDirect RDMA와 P2P를 합치지 않는다

NIC가 GPU memory와 직접 DMA하는 GPUDirect RDMA는 NIC/GPU topology, memory registration, transport와 security 조건이 추가된다. GPU 0→GPU 1 local peer copy와 같은 계약이 아니다. NIC staging이 보이지 않는다고 network end-to-end zero-copy가 증명되는 것도 아니다.

Mooncake/LMCache path에서는 connector가 CPU pinned buffer를 등록하는 구간, RDMA transport가 사용하는 region, serving GPU destination copy를 분리한다. `get_into`가 local CPU allocation에 직접 기록해도 GPU HBM까지 copy 없는 것은 아니다. 각 boundary의 owner와 completion을 쓴다.

### fallback 사건의 first divergence

정상 manifest는 peer capability true, selected direct transport, host bytes 0, peer interval과 destination ready를 기대한다. actual manifest에서 transport가 host staging으로 선택됐다면 first divergence는 transport selection이다. capability true인데 direct submit이 실패해 fallback했다면 error reason과 policy를 본다. selected direct인데 host bytes가 증가하면 hidden staging 또는 metric attribution을 조사한다.

P2P를 강제해 실패시키기보다 explicit fallback을 허용할 수 있다. correctness는 유지하되 예상 latency/capacity와 pinned pool을 admission에 반영한다. latency SLO가 허용하지 않으면 request placement나 topology를 바꾸거나 reject한다. silent fallback이 가장 나쁘다.

### 재발 fixture

같은 root/switch, 다른 root, NVLink 연결, peer-disabled, IOMMU/VM 지원 조합을 가능한 test matrix로 만든다. 모든 hardware 조합을 CI에 갖출 수 없으면 capability mock으로 selection logic을 검사하고 lab에서 실제 route/counter를 주기적으로 검증한다. chunk contents는 source GPU/offset sentinel로 채워 destination checksum과 row identity를 확인한다.

failure injection으로 D2H 뒤 H2D 전 abort, destination GPU reset, one-slot timeout을 넣는다. pending source/destination event와 pinned slot이 회수되고 다음 attempt가 clean generation에서 시작하는지 확인한다.

P2P route의 byte 식을 손으로 확인한다. 4 GiB direct peer copy는 application payload 4 GiB다. host staging fallback은 device link 관점에서 D2H 4 GiB와 H2D 4 GiB, 합계 8 GiB의 endpoint traffic과 host memory write/read를 만든다. chunk padding이나 retry가 있으면 physical traffic은 더 크다. destination useful byte만 보고 4 GiB로 기록하면 fallback 비용을 절반으로 숨긴다.

두 GPU가 같은 PCIe switch 아래 있을 때 peer packet이 upstream root까지 올라가는지는 platform routing에 달릴 수 있다. topology label 하나로 확정하지 않고 counter/vendor 자료를 사용한다. ACS 설정과 virtualization isolation이 route에 영향을 줄 수 있지만 보안 설정을 성능을 위해 임의 해제하지 않는다.

NVLink가 있어도 selected library/transport가 이를 사용하지 않거나 payload type/size 때문에 다른 path를 택할 수 있다. 반대로 profiler에 PCIe traffic이 일부 보여도 control/metadata일 수 있다. payload interval과 byte count를 맞춘다. “NVLink 사용률 0” 한 metric만으로 fallback을 선언하지 않는다.

distributed serving에서 rank mapping이 topology와 어긋나면 TP collective와 KV transfer가 느려진다. rank 순서를 바꾸면 collective topology, expert placement와 request routing도 함께 변할 수 있다. P2P 하나만 최적화하고 다른 traffic을 악화시키지 않도록 전체 communication matrix를 본다.

host staging pool이 여러 transfer protocol에 공유되면 lease class와 priority가 필요할 수 있다. model load가 모든 slots를 차지해 decode KV restore가 기다리는 starvation을 막는다. reserved slots 또는 bounded priority queue를 사용할 수 있다. cancellation은 자기 lease만 해제하며 다른 protocol generation을 건드리지 않는다.

RDMA NIC와 GPU가 다른 root/NUMA에 놓이면 GPUDirect 가능 여부와 actual performance가 다를 수 있다. NIC registration 성공, GPU memory registration, transport selected와 network completion을 각각 기록한다. remote store에서 host pinned buffer로 받은 뒤 GPU로 copy하는 path를 GPUDirect라고 부르지 않는다.

P2P checksum은 전송 correctness를 보지만 semantic owner도 함께 검사한다. source `(request,layer,block,generation)` metadata와 destination mapping을 비교한다. 동일 byte block을 잘못된 request slot에 넣으면 checksum은 통과한다. control metadata와 payload publish를 transaction으로 닫는다.

## 54.9 COPY-54 조사표와 northbridge/southbridge 역사 sidebar

현장에서는 “PCIe가 느리다”는 말보다 한 chunk의 ownership ledger가 빠르다. workbook은 정적 topology, allocation/fault, registration, submission/timeline, destination consumer의 다섯 장으로 나눈다. 앞 장이 닫히지 않으면 뒤 장의 tuning을 하지 않는다.

첫 단계에서는 identity와 topology를 고정한다.

host BIOS/firmware, kernel, NVIDIA driver와 CUDA toolkit, GPU UUID/BDF를 기록한다. container visible index를 host BDF에 매핑한다. PCI tree, device NUMA node, negotiated/max link와 peer topology를 snapshot한다. toolkit 12.x/13.x 결과를 비교할 때 hardware/driver가 같은지 먼저 확인한다.

GPU node가 unknown이면 그 사실을 gap으로 남긴다. 장비 매뉴얼, firmware/ACPI와 CPU root 정보를 보강한다. 임의 node bind로 benchmark 하나가 빨라졌다고 topology fact로 만들지 않는다.

그 위에서 host page와 pool을 조사한다.

allocation API, size/alignment, VMA policy, fault/init thread affinity와 actual page node 분포를 기록한다. registration kind와 flags, 성공/실패/fallback, pinned total과 slot 수를 쓴다. long-lived pool의 pressure와 cleanup owner도 확인한다.

pageable/pinned 비교는 같은 page placement와 copy graph에서 한다. NUMA 비교는 같은 registration과 link contention에서 한다. 두 변수를 동시에 바꾸고 원인을 하나로 부르지 않는다.

배치 위치가 확정되면 byte와 timeline을 잇는다.

useful payload, chunk distribution, in-flight upper bound와 theoretical peak memory를 계산한다. submit return, DMA start/end, event ready와 consumer start를 trace에서 잇는다. gaps마다 owner를 붙인다. file reader, registration, engine queue와 consumer wait를 분리한다.

throughput은 useful bytes/interval로 정의하고 wall/active를 구분한다. compression/quantization이 있으면 source/destination bytes를 각각 기록한다. bidirectional/peer는 direction별로 나눈다.

전송이 끝난 지점부터는 destination과 kernel을 분리해 본다.

destination allocation interval, copy ready generation, conversion temporary와 final parameter를 적는다. kernel logical bytes, L2/HBM 관련 traffic과 duration을 H2D와 별도 분석한다. copy가 빠른데 model load가 느리면 conversion, synchronization과 coverage publish를 본다.

HBM bandwidth가 낮다고 PCIe를 tuning하지 않는다. PCIe H2D가 낮다고 kernel coalescing을 tuning하지 않는다. 둘의 경계는 destination ready event와 consumer start다.

마지막으로 option을 이름이 아니라 실제 상태 변화로 검증한다.

`pinned`는 source registration과 async 가능성을, NUMA option은 worker/pool placement를, stream count는 queue와 in-flight lease를, chunk size는 submission 수와 overlap/peak를 바꾼다. P2P option은 capability/transport selection을, CPU offload size는 host pool과 steady-state transfer bytes를 바꾼다.

각 option 옆에 effective state, 관측 metric, 기대 효과 조건과 반증을 쓴다. field가 바뀌었지만 pool은 pageable이면 pinned 효과가 없다. worker node가 바뀌었지만 pages가 이미 remote면 first-touch 효과가 없다. stream은 늘었지만 engine/link가 포화면 throughput은 늘지 않는다.

workbook의 최종 incident 표에는 최소 열 열을 둔다. 증상/SLO, object와 byte range, owner generation, host backing와 NUMA, registration, PCI path, submission/engine interval, destination ready, consumer first read, first divergence다. 수정과 재발 fixture는 별도 열로 둔다. 이 표만으로 다른 엔지니어가 같은 evidence를 재구성할 수 있어야 한다.

model cold load 사례에서는 resolved model revision과 shard hash를 object identity에 포함한다. CPU offload에서는 request/layer/KV block generation을 쓴다. P2P에서는 source/destination rank와 block identity를 쓴다. 같은 path 문제라도 payload owner가 다르면 cleanup과 retry가 다르다.

관측 도구의 clock domain을 맞춘다. CPU monotonic clock, CUDA event와 profiler timestamp가 직접 같은 origin이 아닐 수 있다. trace framework의 correlation 또는 calibration을 사용한다. 100 µs gap을 두 clock의 offset 오류로 잘못 진단하지 않는다. 큰 단계 관계와 stream ordering은 먼저 확정한다.

metric cardinality를 제한한다. GPU UUID/BDF와 pool type, direction, topology group, fallback reason은 bounded label이 될 수 있다. request ID, pointer와 chunk offset은 sampled trace에 둔다. histogram에는 copy size/latency와 queue delay를, gauge에는 pinned/resident bytes와 in-flight slots를 둔다.

alert는 invariant와 SLO를 분리한다. consumer가 ready generation 이전에 시작하거나 pool lease가 pending DMA 중 해제되는 것은 correctness alert다. remote NUMA ratio 증가, pageable fallback, P2P staging과 link throughput 저하는 performance/capacity alert다. 느리지만 올바른 fallback을 corruption과 같은 severity로 다루지 않는다.

변경 전후 비교는 한 변수와 effective path를 확인한다. `numactl` 명령만 바꾸지 않고 page placement가 실제로 달라졌는지 본다. pinned flag만 바꾸지 않고 registration과 timeline이 달라졌는지 본다. P2P option만 바꾸지 않고 transport와 host bytes가 달라졌는지 본다. 변화하지 않은 knob는 결과의 원인이 아니다.

배포 gate는 작은 smoke transfer를 사용할 수 있다. 각 GPU local pool에서 pinned H2D, peer pair와 fallback 정책, event ordering과 checksum을 확인한다. 이는 production bandwidth benchmark를 대체하지 않지만 잘못된 topology inventory, pageable pool과 disabled peer를 ready 전에 발견한다. smoke가 실패하면 degraded mode 또는 reject를 명시한다.

장비 교체나 BIOS/kernel/driver upgrade 뒤에는 inventory와 fixture를 다시 실행한다. application code가 같아도 NUMA firmware 정보, IOMMU policy, link negotiation과 peer capability가 바뀔 수 있다. 이전 benchmark 숫자를 새 host에 승계하지 않는다. manifest generation과 결과를 배포 artifact에 묶는다.

두 시간 조사 순서를 실제 사건에 적용해 보자. 첫 15분에는 software/hardware identity와 payload를 고정한다. model load라면 shard 7의 256 MiB range, CPU offload라면 request R의 layer 14 block 31을 선택한다. 전체 8 GiB 평균으로 시작하지 않는다. 선택한 range의 source checksum, host pointer generation과 destination interval을 적는다.

다음 15분에는 topology를 닫는다. source page nodes, submission CPU, GPU BDF/node와 root/switch를 한 줄로 연결한다. 이 단계에서 remote NUMA가 보이면 원인 후보지만 아직 결론은 아니다. PCI current/max link가 다르면 hardware 운영 후보를 분리한다. topology가 정상이라고 pinned/timeline이 정상인 것은 아니다.

30분부터 registration과 ownership을 본다. effective buffer가 pageable인지, registered pool의 어느 slot인지, lease가 DMA 끝까지 유지되는지 확인한다. pool 설정 파일이 아니라 pointer/registration 결과와 fallback counter를 본다. source checksum은 copy 직전과 slot release 전 같은지 표본 검사한다. 달라지면 PCIe 이후를 조사하지 않는다.

45분에는 stream timeline을 닫는다. submit, actual DMA, ready event, consumer interval을 그린다. submit 전 gap, engine queue gap과 consumer wait를 각각 owner에 배정한다. device synchronize, default stream, graph update와 event generation을 확인한다. copy interval 자체가 정상 throughput인데 wall time만 길다면 link tuning을 중단한다.

60분에는 byte 손계산을 profiler와 맞춘다. useful source/destination, staging 복제, bidirectional/P2P fallback, retry와 padding bytes를 합한다. measured traffic이 식의 두 배라면 hidden staging 또는 retry를 찾는다. measured가 더 작으면 compression, cache/metric domain 또는 관측 누락을 본다. 단위 GB/GiB와 direction을 명시한다.

75분에는 destination 이후를 분리한다. copy ready 뒤 conversion/packing kernel, L2/HBM traffic, final parameter publish까지 본다. H2D가 끝났는데 kernel memory stall이 길다면 host topology는 반증된다. 반대로 consumer가 기다리며 GPU가 idle이면 device HBM tuning은 우선순위가 아니다.

90분에는 한 변수로 재실험한다. pageable→pinned, node 0→local node, stream dependency 제거가 아니라 정확한 event로 교체, concurrent peer traffic off 가운데 evidence가 가리킨 하나만 바꾼다. effective state가 실제로 변했는지 확인하고 checksum을 유지한다. 여러 옵션을 한꺼번에 바꾸면 first divergence 지식을 잃는다.

마지막 30분에는 cleanup과 재발 fixture를 만든다. copy mid-flight abort, tail chunk와 worker cancellation을 넣고 pool lease·destination publish가 안전한지 본다. 정상 경로 성능 숫자만 고치고 error path가 pending DMA 중 unregister하면 다음 장애는 corruption이 된다. 수정의 종료 조건은 속도, bytes와 ownership invariant를 함께 통과하는 것이다.

이 조사표는 monitoring 설계로 이어진다. `h2d_submit_delay`, `h2d_active_duration`, `h2d_bytes`, `pinned_fallback_total`, `pool_inflight_bytes`, `remote_numa_page_ratio`, `peer_staging_bytes`를 bounded dimension으로 둔다. conversion에는 별도 kernel duration과 device-memory traffic을 둔다. 하나의 `model_load_seconds`만으로는 어느 단계가 달라졌는지 알 수 없다.

Prometheus에서 BDF나 topology group을 label로 쓸 때 fleet cardinality를 검토한다. chunk ID와 pointer는 label로 쓰지 않는다. exact incident correlation은 trace ID와 structured log에 둔다. histogram bucket은 실제 chunk/latency distribution에 맞추고 평균만으로 tail gap을 숨기지 않는다. scrape interval보다 짧은 burst는 counter와 trace가 필요하다.

Nsight Systems는 CPU thread와 CUDA stream timeline을 잇는 데 유용하고 Nsight Compute는 consumer kernel의 L2/device-memory behavior를 파는 데 유용하다. 두 도구의 책임을 섞지 않는다. `nvidia-smi topo`, PCI sysfs와 `lspci`는 static/effective topology를 보조한다. tool output은 version과 host snapshot을 함께 저장한다.

권한이 제한된 container에서는 PCI tree나 page node를 전부 못 볼 수 있다. 관측 부재를 node 0 또는 direct path로 채우지 않는다. privileged sidecar/inventory service가 bounded topology manifest를 제공하거나 operator가 host에서 증거를 수집한다. tenant에게 physical address 같은 민감 정보를 노출하지 않고 locality group과 capability만 전달할 수 있다.

자동 튜너를 만든다면 correctness guard 밖에서 작동하게 한다. chunk size, in-flight depth와 load wave를 조정할 수 있지만 registration kind, ownership ordering과 checksum invariant를 깨면 안 된다. 탐색 중 pinned peak와 online SLO budget을 넘지 않게 constraint를 둔다. 최적값은 topology/driver generation과 workload에 cache하고 환경 변경 시 폐기한다.

CPU offload knob의 비용을 예로 계산한다. layer weights 20 GiB를 CPU에 두고 token마다 2 GiB를 H2D해야 한다면 20 GiB resident를 아끼는 대신 decode hot path에 2 GiB transfer가 들어온다. effective 20 GiB/s라 해도 순수 transfer 하한이 약 100 ms다. prefetch/overlap이 이를 숨길 compute slack이 있는지 보지 않고 “HBM 20 GiB 절약”만 말할 수 없다.

KV offload는 token마다 전체 KV를 옮기지 않고 victim/restore block 단위일 수 있다. block 64 MiB 열 개를 restore하면 useful H2D 640 MiB다. remote NUMA와 staging fallback이 있어 active 16 GiB/s라면 약 40 ms 하한이고 queue gap이 더해진다. scheduler가 restore를 compute 전에 끝내거나 overlap해야 한다. restored block ready 이전에 attention block table을 publish하면 correctness가 깨진다.

model loading은 latency tolerant해 더 큰 chunk와 load staggering을 선택할 수 있다. decode offload는 작은 block과 tail latency가 중요하다. 같은 pinned pool implementation을 공유해도 queue policy와 slot reservation은 다를 수 있다. bulk load가 모든 slots를 차지하지 않도록 workload class별 quota를 둔다.

마지막으로 결과를 “PCIe 80% 활용” 한 숫자로 요약하지 않는다. useful payload, wall/active throughput, overlap, pinned/resident peak, remote bytes, fallback과 checksum 결과를 함께 보고한다. 최적화가 active throughput을 높였지만 peak가 두 배가 되고 online ITL이 나빠졌다면 성공이 아니다. serving 목적 함수는 정확도와 SLO, capacity와 startup을 함께 가진다.

세 incident를 한 번 더 나란히 놓으면 조사 경계가 선명해진다. 첫 사건은 pageable source라 실제 asynchronous overlap 조건이 닫히지 않았다. 두 번째는 range가 pinned였지만 physical pages가 destination GPU와 먼 node에 있었다. 세 번째는 source와 destination이 모두 GPU allocation이었지만 peer transport가 host staging을 선택했다. 모두 “전송이 느리다”는 증상이지만 first divergence는 registration, placement, transport로 다르다.

pageable 사건에서 PCI negotiated width가 정상이라는 사실은 link failure를 약하게 만들지만 registration 원인을 증명하지 않는다. pinned control fixture의 submit/timeline이 달라져야 한다. remote NUMA 사건에서 local node fixture가 빨라져야 하며 page distribution도 실제로 달라져야 한다. P2P 사건에서는 host D2H/H2D bytes와 selected transport가 함께 바뀌어야 한다. 각 수정에는 독립된 반증 증거가 필요하다.

corruption 사건이라면 속도 조사보다 owner generation을 먼저 본다. chunk 7 checksum이 source에서는 맞고 destination에서 틀리면 file parsing을 제외할 수 있다. copy 직전 host checksum과 copy 완료 뒤 destination checksum 사이가 첫 경계다. source slot이 DMA 중 덮였는지, length/offset과 event generation을 확인한다. PCIe가 bit를 임의로 바꿨다는 희귀 가설로 먼저 가지 않는다.

destination checksum은 맞지만 model output이 틀리면 conversion과 parameter mapping으로 이동한다. H2D path를 반복 tuning하지 않는다. 반대로 final parameter는 맞지만 first inference만 느리면 lazy page/residency, graph warm-up과 kernel cache를 본다. model load wall time과 first-request latency의 owner가 다를 수 있다.

NUMA 최적화 뒤에도 throughput이 같다면 host DRAM 또는 PCIe가 이미 다른 limit에 도달했을 수 있다. local/remote interconnect traffic이 줄었는지 보고 policy가 적용됐음을 확인한다. 적용됐지만 user metric이 안 변했다면 “NUMA가 중요하지 않다”가 아니라 이 workload에서는 다른 bottleneck이 지배한다고 결론낸다.

stream 수 증가 뒤 active throughput은 같고 wall gap만 줄었다면 submission pipeline이 개선된 것이다. active throughput과 overlap이 모두 같고 pinned peak만 늘면 되돌린다. chunk size 증가 뒤 active throughput이 오르지만 online control copy tail이 악화되면 separate queues 또는 load rate limit을 고려한다. 한 지표의 개선을 전체 성공으로 만들지 않는다.

P2P direct path가 staging보다 빠르더라도 peer traffic이 TP collective와 같은 NVLink를 경쟁할 수 있다. production simultaneous matrix에서 측정한다. KV transfer를 PCIe staging으로 돌리는 편이 collective tail을 보호하는 특수 workload도 있을 수 있지만, 그 선택은 explicit transport policy와 capacity 계산으로 해야 한다. “direct는 항상 최적”도 절대 법칙이 아니다.

회귀 fixture는 정확성, path selection과 성능 guard를 층으로 나눈다. CPU-only/unit tier는 slot state machine, generation과 topology planner를 검사한다. single-GPU tier는 pageable/pinned, event ordering과 checksum을 검사한다. multi-GPU tier는 peer/direct/fallback과 abort를 검사한다. lab performance tier는 NUMA/root contention과 threshold를 검증한다. 하위 tier 통과를 실제 hardware path 증거로 과장하지 않는다.

threshold는 절대 vendor peak 비율보다 known-good host baseline과 허용 변화로 둘 수 있다. driver/toolkit 변경 때 baseline을 무조건 갱신하지 않고 topology, effective path와 profiler 변화 원인을 검토한다. 정확성 invariant는 threshold 완화 대상이 아니다. 느린 fallback은 정책상 허용할 수 있지만 wrong generation은 언제나 실패다.

incident가 끝나면 정적 inventory, effective path manifest, minimal trace, 손계산과 수정 fixture를 함께 보존한다. 거대한 profiler capture만 남기면 다음 사람이 특정 tool version 없이는 재현하기 어렵다. 핵심 interval과 counters를 bounded JSON/표로 요약하되 원본 trace의 hash와 보관 위치를 연결한다.

ownership ledger의 마지막 행은 반드시 해제를 포함한다. model load라면 source file/mapping, pageable object, pinned slot, intermediate device tensor와 final parameter가 각자의 last consumer 뒤 해제되는지 본다. CPU offload라면 request block, host slot과 destination block의 reference count가 cancellation과 completion 양쪽에서 닫혀야 한다. P2P staging은 source event, host lease와 destination event 세 owner가 모두 끝나야 slot을 반환한다.

정상 종료만 보면 leak과 use-after-free의 절반만 검사한다. tail chunk가 zero이거나 copy submit 전 cancel, DMA 중 timeout, destination conversion 실패, worker shutdown을 각각 주입한다. 해제가 너무 늦으면 pool starvation과 memory peak가 생기고 너무 이르면 corruption이 생긴다. 다음 attempt가 같은 pointer를 재사용해도 generation이 달라 old event가 publish하지 못해야 한다.

최종 regression report에는 네 판정을 둔다. byte correctness, ownership cleanup, effective path와 performance envelope다. correctness/cleanup 실패는 배포를 막는다. expected direct path가 staging으로 바뀌면 policy에 따라 막거나 degraded로 표시한다. throughput 변화는 topology와 workload noise 범위에서 판단한다. 이 구분이 있어야 느린 fallback을 안전하게 운영하면서 silent corruption은 단호히 차단할 수 있다.

이렇게 한 chunk를 끝까지 닫으면 “CPU와 GPU 연결이 느리다”는 추상 문장이 실제 조치로 바뀐다. 어느 page가 어디에 있었고, 누가 pin/map했으며, 어느 root와 queue를 지나 어떤 generation의 destination을 공개했는지가 답이다. 측정은 그 사슬을 반증하기 위해 존재하고 옵션은 확인된 첫 divergence를 바꾸기 위해서만 사용한다.

### 현대 topology에서 northbridge를 찾지 않는다

과거 PC 설명에서는 northbridge가 CPU·memory·graphics의 빠른 경로를, southbridge가 주변 I/O를 담당했다. 오늘날 데이터센터 CPU에서는 memory controller와 PCIe root complex가 CPU package에 통합되고 socket 간 coherent interconnect와 외부 PCIe switch가 실제 locality를 결정하는 경우가 일반적이다. 고전 그림은 역사적 직관일 뿐 장비 조사 도구가 아니다.

CUDA 문서의 오래된 front-side-bus 조건 문구를 modern server 전체에 일반화하지 않는다. 현재 host의 CPU block diagram, PCI sysfs와 vendor topology를 사용한다. “southbridge를 거쳐 느리다” 대신 어느 BDF가 어느 root/switch와 NUMA node를 공유하는지 쓴다.

여기까지의 workbook을 바탕으로 마지막 물리 경로 사건을 검산한다.

## 54.10 COPY-54: Async라는 이름은 같았지만 세 물리 경로는 달랐다

fixture는 socket0에 GPU0가 PCIe root를 통해 연결되고 socket1은 CPU interconnect를 건너 GPU0에 도달하는 dual-socket host다. transfer는 64 MiB FP16 KV chunk다. compute kernel C0는 GPU0 stream compute에서 4 ms 실행되고 H2D copy H0는 copy stream에 enqueue된다.

경로 A의 host buffer는 socket0 local NUMA pages에 first-touch됐고 CUDA 등록/pinned pool에 있다. effective H2D payload가 24 GB/s라고 가정하면 이상적 copy 시간은 `64 MiB / 24 GB/s ≈ 2.8 ms`다. copy engine과 compute resources가 overlap 가능한 조건이면 4 ms compute 안에 copy가 숨을 수 있다.

경로 B는 pinned이지만 pages가 socket1에 있다. CPU interconnect와 socket0 root를 지나야 하고 effective path가 12 GB/s라면 약 5.6 ms다. kernel4 ms와 겹쳐도 최소 약1.6 ms가 critical path에 남는다. `pinned=true` metric은 A/B를 구분하지 못한다.

경로 C는 pageable source다. runtime/driver가 staging pin/copy를 수행해야 하는 path라 host-side preparation과 synchronization behavior가 달라질 수 있다. 1.5 ms staging과 2.8 ms DMA가 순차적이라면 4.3 ms이고 host return/overlap 조건도 pinned path와 같다고 가정할 수 없다. exact semantics는 CUDA 공식 문서의 API/host-memory 조건으로 확인한다.

사건의 dashboard는 세 paths 모두 `cudaMemcpyAsync`, bytes64MiB, stream copy로 기록했다. 운영자는 API 이름만 보고 DMA와 kernel이 겹친다고 결론냈다. 실제 trace에서 C path host thread가 1.5 ms 준비에 묶이고, B path DMA가 compute 뒤까지 이어졌다. overlap failure 원인은 하나가 아니었다.

첫 divergence는 copy call이 아니다. A는 allocation/first-touch가 GPU-local node였고 등록이 성공했다. B는 worker CPU만 socket0에 pin했지만 memory pool pages는 startup 때 socket1 thread가 first-touch했다. C는 pinned pool exhaustion 뒤 pageable fallback가 발생했지만 fallback reason가 metrics에 없었다.

물리 path 원장은 `(virtual_range,page_nodes,pin_registration,DMA_mapping,PCIe_route,engine,stream_seq,event_generation,destination_range,consumer)` 열을 가진다. `src_ptr`와 `dst_ptr`만 남기지 않는다. 같은 virtual pool object도 pages placement와 registration generation이 달라질 수 있다.

page placement는 allocation call의 CPU보다 first write/fault context에 의해 정해질 수 있다. pool을 socket0 worker가 나중에 사용해도 initialization thread가 socket1에서 모든 pages를 zero했다면 remote placement가 남는다. CPU affinity와 memory policy/first-touch를 함께 검증한다.

pinning은 physical pages를 DMA 가능한 lifetime에 묶지만 topology를 local로 옮겨 주지 않는다. existing remote pages를 register해도 remote NUMA path다. pin 성공과 NUMA placement는 독립 columns다. long-lived pool을 만들 때 placement 후 registration 순서와 owner를 기록한다.

DMA engine에 submit됐다는 사실과 device destination이 consumer-ready라는 사실도 다르다. copy stream H0 뒤 Ready64 event를 record하고 compute stream의 consumer C1이 wait한다. event는 해당 record generation 앞 copy completion을 표시한다. host API return로 C1을 시작하지 않는다.

double buffer X0/X1을 쓰면 generation을 붙인다. H0가 X0 generation17에 copy하고 C1이 읽는 동안 next host fill가 X0을 덮지 않는다. host buffer reuse도 DMA source read completion 뒤여야 하고 device buffer reuse도 GPU consumer completion 뒤여야 한다. 양쪽 lifetime이 있다.

HBM consumer는 destination pointer를 받은 kernel이다. copy bytes가 GPU memory에 도착해도 kernel이 L2-resident data를 읽거나 cache behavior가 다를 수 있으므로 HBM counter와 H2D bytes를 동일시하지 않는다. 이 사건은 path readiness를 소유하고 memory hierarchy 성능은 consumer counters로 별도 확인한다.

vLLM offload source walk는 offload/pinned host buffer allocation, tensor/page movement scheduling, stream/event handoff, destination consumer를 잇는다. option으로 offload bytes를 설정한 지점에서 멈추지 않는다. actual host allocation가 pinned인지, NUMA placement를 누가 정하는지, async copy completion을 누가 기다리는지 찾는다.

SGLang offload/HiCache 계열 path도 tier policy 설명을 반복하지 않고 물리 move만 따라간다. host pool allocation/registration, device transfer call, stream/event, cache location publish, attention consumer 순서를 pin한다. backend/connector가 내부 stream을 쓰면 caller ready event가 실제 join을 포함하는지 확인한다.

source에 explicit NUMA placement가 없다면 “local allocation을 보장한다”고 쓰지 않는다. deployment affinity/first-touch로 관측된 사실과 library contract를 분리한다. pinned memory API가 NUMA policy까지 설정한다고 추론하지 않는다.

반증 A는 동일 pinned buffer를 socket0/1에서 first-touch해 placement만 바꾼다. bytes/stream/kernel은 고정한다. bandwidth와 overlap tail이 topology 차이를 따라 바뀌면 remote NUMA 가설을 지지한다. pinning까지 동시에 바꾸지 않는다.

반증 B는 placement local을 고정하고 pinned/pageable만 바꾼다. host call duration, staging bytes, DMA interval, compute overlap를 기록한다. pageable path가 async 이름을 가졌다는 사실보다 실제 timeline을 본다. runtime version과 allocation method를 함께 기록한다.

반증 C는 pinned/local을 고정하고 copy/compute streams와 event edge를 바꾼다. same stream serialization, separate streams+correct wait, accidental device synchronize를 비교한다. engine capability가 있어도 dependency/stream 배치가 직렬이면 overlap하지 않는다.

반증 D는 two simultaneous H2D copies를 넣어 engine contention와 shared PCIe route를 본다. advertised engines 수가 독립 full-bandwidth paths를 의미하지 않는다. root/switch/link를 공유하면 aggregate가 ceiling에 걸린다. GPU compute와 copy counters를 함께 본다.

rollback는 먼저 pageable fallback 원인을 제거하거나 명시적 synchronous path로 capacity를 낮춰 correctness를 지킨다. remote pool은 신규 allocations를 local node에서 first-touch/register하고 old pool requests를 drain한다. in-flight DMA source를 즉시 unpin/free하지 않는다.

과도한 device synchronize를 임시 진단으로 넣었다면 final fix에서 좁은 Ready/Done events로 되돌린다. local pinned path A가 compute와 실제 겹치고 output equality가 유지되는지 확인한다. overlap을 위해 dependency를 제거하지 않는다.

90분 soak는 local/remote pressure, pool exhaustion, chunk sizes4/16/64MiB, concurrent copies, graph/eager를 섞는다. page-node distribution, pinned fallback, H2D duration, event wait, buffer generation overlap, output sentinel을 본다. average bandwidth만으로 종료하지 않는다.

terminal 문장은 구체적이다. “copy option은 async였지만 38% transfers가 pageable fallback, pinned pool pages 중 62%가 remote node여서 host staging/NUMA path가 critical 4ms compute를 넘었다.” fix는 local first-touch+registration, bounded pinned pool, event-bound double buffers와 trace overlap로 증명한다.

**왜 같은 memcpy가 다른 비용을 내는가.** source NUMA node가 GPU와 먼 socket이면 CPU interconnect와 PCIe root complex를 더 지나므로 pinned 여부가 같아도 대역폭과 tail latency가 달라진다. page가 pageable이면 staging pin·copy가 추가되고, 왜 async API가 host에서 빨리 돌아와도 device consumer가 즉시 안전하지 않은지는 stream completion이 별도이기 때문이다. 따라서 비용은 API 이름이 아니라 page placement, route, link width·generation, copy engine과 synchronization edge로 설명한다.

## 54.11 물리 byte-path를 20분 source drill로 재현한다

첫 3분에는 host virtual range를 고정한다. buffer X17은 64 MiB, virtual `[V,V+64MiB)`, pool slot 3, generation17이다. allocation API, allocation thread CPU affinity, memory policy, first-write thread를 기록한다. pointer만 보고 physical locality를 결정하지 않는다.

pages가 어느 NUMA node에 있는지는 OS가 제공하는 mapping/page placement 관측으로 확인한다. 전체 range의 node histogram을 표본화하고 node0 38%, node1 62%처럼 적는다. virtual range가 contiguous여도 physical pages는 여러 nodes에 분산될 수 있다. 한 page 표본을 전체로 확대하지 않는다.

두 번째 3분에는 pin/registration owner를 찾는다. `cudaHostAlloc`로 처음부터 pinned allocation인지, 기존 pages를 `cudaHostRegister`했는지, framework의 pinned allocator인지 구분한다. registration success, range/flags, registration generation, unregister/free owner를 기록한다.

registration이 성공해도 pages가 local로 이동했다고 가정하지 않는다. register 전후 node histogram을 비교하되 CUDA API에 없는 migration guarantee를 만들지 않는다. deployment가 bind/membind/first-touch로 locality를 만든다면 그 설정과 실행 순서를 evidence로 둔다.

세 번째 3분에는 GPU BDF에서 root path를 거슬러 간다. GPU0의 PCI domain:bus:device.function, link generation/width, upstream switch, root complex, NUMA node를 inventory한다. host pages node1에서 GPU0 node0 root까지 socket interconnect가 추가되는지 topology graph에 표시한다.

`PCIe Gen5 x16` theoretical number를 transfer expected로 바로 쓰지 않는다. negotiated current width/speed, protocol overhead, shared switch/root contention, direction, payload size, concurrent devices를 고려한다. 64 MiB measured24GB/s와 link theoretical ceiling의 차이를 “PCIe 문제” 한 단어로 끝내지 않는다.

네 번째 3분에는 CUDA source path를 읽는다. host buffer producer가 어느 function에서 slot을 claim하고 fill 완료를 publish하는지, copy call arguments와 stream을 누가 고르는지, destination device buffer generation을 누가 claim하는지 찾는다. `non_blocking=True` 같은 wrapper option에서 멈추지 않는다.

copy call 직전 source range가 pinned인지 runtime check/allocator provenance로 확인한다. pinned pool exhaustion가 pageable allocation로 fallback하는 branch와 metric/reason을 찾는다. fallback를 허용한다면 caller가 같은 async/overlap 기대를 유지하는지 검토한다.

다섯 번째 3분에는 stream timeline을 만든다. host fill DoneHost17, H2D enqueue sequence301, ReadyDevice17 event record302, consumer stream wait501, kernel consume502, DoneConsume17 event503을 적는다. host call return timestamps와 device execution intervals를 다른 columns로 둔다.

copy/compute가 separate streams여도 dependency와 engine/resources가 overlap를 허용해야 한다. same default stream, implicit synchronization, device-wide synchronize, preceding operations가 timeline을 직렬화할 수 있다. profiler bar가 겹치지 않는다는 결과에서 원인을 역추적한다.

마지막 5분에는 byte correctness와 release를 닫는다. X17 host sentinel가 destination D17에서 같고 consumer output reference가 맞는지 본다. ReadyDevice17 전에 consumer가 읽지 않고 DoneConsume17 전에 D17을 재사용하지 않는다. source X17은 DMA read completion 전 fill/reuse/unregister되지 않는다.

COPY-54 trace를 이 drill로 채우면 node histogram62% remote, pin fallback38%, copy stream separate, Ready wait correct, engine interval 5.6ms가 나온다. dependency는 맞았지만 물리 source path가 4ms compute보다 길었다. overlap가 partial인 것이 정상 결과다.

### vLLM 물리 source walk의 종료 조건

vLLM에서 offload option normalization만 찾으면 tutorial의 시작점이다. 실제 CPU storage/pinned buffer를 누가 만들고 model/KV data를 어떤 iterator/worker가 채우며 어느 CUDA stream에서 destination copy를 수행하고 consumer module이 어떤 ready condition 뒤 읽는지 이어야 한다.

weight CPU offload와 KV cache offload는 owners와 frequency가 다를 수 있다. 이 절은 policy 이득을 비교하지 않고 각 path의 allocation provenance, copy range, stream/event, destination consumer만 기록한다. 한 path의 pinned behavior를 다른 path에 일반화하지 않는다.

PyTorch tensor의 `pin_memory`/non-blocking flags가 보이면 underlying storage가 실제 pinned allocator에서 왔는지, view/range가 registration 안에 있는지 확인한다. tensor metadata flag만으로 NUMA locality나 copy-engine concurrency를 증명하지 않는다.

worker/rank별 CPU affinity가 달라지면 pool creation/first-touch thread도 본다. GPU rank0가 node0인데 global loader thread가 node1에서 pools를 초기화할 수 있다. serving worker가 local CPU에 있어도 existing pages는 remote다.

native/custom transfer manager가 events를 내부에서 처리하면 public caller의 synchronize 여부만 보지 않는다. ready future/callback가 어느 stream event generation을 소비하는지, destination block/table publish가 completion 뒤인지 찾는다. 내부 copy 완료와 consumer 완료를 구분한다.

### SGLang 물리 source walk의 종료 조건

SGLang의 host/offload/HiCache 관련 option은 tier capacity/policy의 entry다. physical path는 memory pool allocation, host storage type/registration, transfer backend, source/destination location metadata, stream/event, attention consumer로 이어진다. option name에 `async`나 `pinned`가 있어도 runtime fallback를 확인한다.

connector/NIXL 같은 별도 transport가 개입하면 local CUDA copy와 remote transfer 단계가 나뉠 수 있다. 이 장은 local CPU page→GPU HBM 구간만 소유한다. remote network registration/protocol은 해당 장의 contract를 인용하고 local staging range가 어디서 시작되는지만 표시한다.

host location publish가 copy submission 시점인지 completion 시점인지 본다. cache metadata가 destination-ready로 너무 일찍 바뀌면 attention consumer가 partial bytes를 읽는다. transfer future가 완료돼도 어느 buffer generation을 가리키는지 확인한다.

pool eviction/reuse policy는 38장에 맡기고 여기서는 selected slot의 lifetime만 본다. X17 slot이 DMA source, D17이 kernel consumer인 동안 next transfer가 same generation을 덮지 않는가가 질문이다. capacity 손익 계산을 반복하지 않는다.

### pageable·pinned·NUMA 반증 matrix

matrix 행 A는 local pinned, B remote pinned, C local pageable, D remote pageable다. columns는 host call duration, CPU staging bytes/time, H2D engine interval, root/interconnect traffic, copy-compute overlap, output checksum이다. bytes64MiB와 streams/kernel은 고정한다.

A는 host preparation 최소, DMA2.8ms라는 fixture baseline이다. B는 remote route로5.6ms다. C는 staging1.5+DMA2.8ms 가능성을 관측한다. D는 staging source read와 remote placement가 모두 개입할 수 있다. 숫자는 measurement fixture이지 모든 장비의 고정값이 아니다.

matrix 행 E는 local pinned지만 same stream, F는 local pinned separate streams+event, G는 local pinned separate streams+device sync다. A/B/C/D가 memory path를 분리했다면 E/F/G는 dependency/serialization를 분리한다. 여러 axes를 한 번에 바꾸지 않는다.

행 H/I는 concurrent H2D one/two copies다. aggregate link bytes와 per-copy duration을 본다. copy engines가 둘이어도 root/link/shared memory subsystem에서 contention가 생길 수 있다. engine count를 bandwidth multiplier로 쓰지 않는다.

chunk 4/16/64MiB는 launch/setup amortization와 pipeline depth를 보여 준다. 4MiB는 많은 events/launch overhead, 64MiB는 overlap tail과 buffer pressure가 커질 수 있다. 이 절에서는 cache hit/eviction 이득이 아니라 physical copy timeline만 비교한다.

### HBM consumer까지 주소 identity를 보존한다

host X17 relative offset r가 device D17 offset r에 복사된다고 가정한다. chunk/range coalescing가 있으면 source/destination offsets가 달라질 수 있으므로 copy descriptor를 기록한다. sentinel를 first/middle/last cache lines에 둬 truncation/offset 오류를 잡는다.

consumer kernel argument가 D17 base와 expected stride/layout을 가리키는지 본다. H2D bytes가 정확해도 kernel이 D16 stale pointer 또는 wrong layer/block offset을 읽으면 output이 틀린다. copy path와 consumer binding을 마지막 edge로 연결한다.

copy completion 뒤 data가 device memory에 있지만 HBM DRAM read bytes는 kernel access/cache state에 따라 달라진다. L2 hit data는 HBM counter가 적을 수 있다. H2D link bytes와 kernel HBM bytes를 equality invariant로 두지 않는다. correctness는 sentinel/output로 검증한다.

destination allocation가 unified/mapped/device memory인지에 따라 physical path가 다르다. pointer API/type과 allocation owner를 확인한다. mapped host memory를 device pointer로 접근하는 path를 H2D-to-HBM copy라고 부르지 않는다.

### COPY-54 rollback와 배포 terminal

첫 rollback는 pool provenance를 노출하고 pageable fallback가 발생하면 bounded degraded mode로 전환하는 것이다. synchronous behavior/latency capacity를 반영한다. fallback를 숨긴 채 overlap SLO를 유지하지 않는다.

둘째는 new pinned pools를 GPU-local node에서 allocate/first-touch/register한다. old remote pool은 active DMA/consumers가 끝난 뒤 drain한다. pages migration를 in-flight slot에 강제하지 않는다. worker startup order와 CPU/memory affinity를 deployment manifest에 둔다.

셋째는 copy/consumer Ready/Done events를 좁은 dependencies로 복구한다. diagnostic device synchronize를 제거하고 double buffers의 source/destination generations를 확인한다. overlap를 늘리기 위해 wait를 제거하지 않는다.

canary는 pinned success/fallback, node histogram, effective PCI route, H2D duration, host call time, overlap fraction, sentinel mismatch를 본다. throughput 하나로 승인하지 않는다. remote node fraction이나 fallback가 baseline에서 벗어나면 root cause를 먼저 찾는다.

shutdown/abort fixture는 host fill 전, H2D enqueue 뒤, Ready 전, consumer 뒤에 cancellation를 넣는다. source slot registration와 destination storage가 각 last consumer 뒤 해제되는지 본다. late event가 new generation을 publish하지 못하게 한다.

terminal report에는 A–I matrix, source pins, topology inventory, stream sequence301–503, output fixture를 포함한다. COPY-54 first divergence가 pageable fallback/remote first-touch였고 dependency는 correct였음을 분리한다. performance fix와 correctness edge를 함께 보존한다.

수치표 첫 행은 transfer identity다. request R54, host slot X17, device slot D17, 64 MiB, direction H2D, rank0/GPU0를 적는다. 둘째 행은 source pages node0 38%, node1 62%, registration pinned generation7이다. 셋째는 GPU BDF와 node0 root, link/switch path다. 넷째는 copy stream sequence301과 event302, consumer sequence501–503이다.

이 표에서 `pinned=yes` 하나만 보면 placement62% remote를 놓친다. `stream=copy` 하나만 보면 consumer wait와 preceding work를 놓친다. `Gen5x16` 하나만 보면 negotiated/shared route와 payload를 놓친다. 각 값은 물리 path의 한 edge일 뿐 전체 overlap predicate가 아니다.

64 MiB는 67,108,864 bytes다. decimal 24 GB/s를 쓰면 약2.80 ms, 12 GB/s면5.59 ms다. GiB/s로 표시된 도구와 혼합하면 수% 차이가 생기므로 단위도 기록한다. 이 계산은 expected order를 검산하는 데 쓰고 measured duration을 억지로 맞추지 않는다.

overlap fraction도 정의한다. compute interval `[10,14]ms`, copy A `[10.2,13.0]`이면 copy 대부분이 compute 안이다. B `[10.2,15.8]`이면3.8ms만 겹치고2ms가 tail이다. host API call `[9.0,9.1]`은 이 device overlap 계산에 직접 쓰지 않는다.

pageable C에서 host call이 `[9.0,10.5]`이고 DMA가 `[10.5,13.3]`이라면 staging preparation가 submission critical path에 있다. UI가 하나의 memcpy bar만 보여도 host span과 device span을 분리한다. runtime가 내부 staging을 어떻게 구현하는지는 공식 semantics 이상으로 추측하지 않고 관측 가능한 intervals를 기록한다.

NUMA remote traffic counter는 node histogram의 보조 증거다. CPU interconnect bytes 증가가 H2D와 시간상 맞는지 본다. 다른 workload traffic를 제외하거나 canary에서 격리한다. page placement가 remote라는 사실만으로 measured bandwidth loss의 전부를 설명하지 않는다.

PCIe payload counters가 있으면 requested64MiB와 실제 link traffic를 비교하되 protocol overhead/read/write direction를 고려한다. root/switch counters와 GPU copy bytes가 다른 scope일 수 있다. 동일 metric 이름처럼 보여도 counter domain을 명시한다.

copy engine utilization가 낮아도 host staging/serialization 때문에 work가 공급되지 않았을 수 있다. utilization가 높아도 link/root contention로 bandwidth가 낮을 수 있다. engine 수·utilization·payload·timeline을 한꺼번에 본다. “engine idle”을 kernel bottleneck으로 바로 해석하지 않는다.

stream priority가 copy를 자동으로 preempt/overlap시킨다고 가정하지 않는다. priority는 scheduling hint/contract 범위가 있고 running kernels, engine availability, dependency를 무시하지 않는다. source에서 priority 설정이 실제 어느 stream에 적용되는지 확인한다.

CUDA Graph 안에 memcpy node가 있으면 captured source/destination address와 replay generation를 본다. host slot contents가 current generation으로 채워졌고 previous DMA/host fill가 끝났는지 확인한다. graph launch return는 copy completion이 아니다. replay completion/event를 lifetime ledger에 넣는다.

managed memory/prefetch path가 선택되면 explicit H2D copy와 다른 물리 state machine이다. page migration, residency/advice/fault가 개입할 수 있다. 이 장의 memcpy fixture 결과를 managed path에 적용하지 않고 effective path manifest에서 별도 분기한다.

mapped host memory zero-copy도 destination HBM allocation/copy 없이 GPU가 host memory를 접근할 수 있다. 작은 control data에는 유용할 수 있지만 latency/bandwidth/access pattern가 다르다. pointer가 device-visible하다는 사실을 HBM resident로 쓰지 않는다.

P2P staging fallback는 source GPU→host X17과 host X17→destination GPU 두 transfers를 가진다. X17은 first DMA destination completion 뒤 second DMA source로 handoff된다. 두 events와 host slot generation가 필요하다. direct peer path의 event 하나를 그대로 재사용하지 않는다.

vLLM code에서 CPU offload buffer가 ordinary torch CPU tensor인지 pinned allocator인지 exact constructor를 확인한다. `.to(device, non_blocking=True)` 호출이 있어도 source storage 조건과 stream context를 본다. wrapper가 current stream을 쓰는지 dedicated stream을 쓰는지 pinned source line과 caller로 확인한다.

vLLM parameter loading/offload가 forward 때 weight를 이동한다면 destination tensor가 kernel launch까지 살아 있는지 module/runner owner를 본다. layer마다 temporary destination을 reuse하면 previous kernel completion edge가 필요하다. model load one-time copy와 request-time offload를 섞지 않는다.

SGLang host cache transfer가 transfer manager future를 반환하면 future ready의 의미를 source에서 찾는다. network/local copy submission인지 GPU destination-ready인지 구분한다. cache metadata publish와 attention block-table consumer 사이에 correct wait가 있는지 trace한다.

두 framework source에 NUMA bind가 없다면 deployment 수준에서 service process CPU set, memory policy, pool initializer affinity를 맞춘다. library가 local page placement를 보장한다고 문서화하지 않는다. source fact와 deployment remedy를 다른 columns로 둔다.

pool capacity가 부족해 pageable fallback를 허용할지 admission/wait할지는 정책 문제다. 이 장은 fallback가 선택된 뒤 실제 physical path와 semantics를 기록한다. cache 효율/용량 손익은 38장에 맡긴다. 여기서는 fallback reason와 overlap contract만 판정한다.

pool resize는 in-flight slots를 재할당하지 않는다. 새 generation pool을 local node에서 만들고 registrations를 완료한 뒤 신규 transfers를 보낸다. old generation owners가0이 되면 unregister/free한다. pointer만 같거나 size가 늘었다고 live DMA를 migrate하지 않는다.

registration limit/OS locked-memory limit failure는 explicit error 또는 fallback로 노출한다. startup success 뒤 pressure에서 registration failure가 발생할 수 있는지 본다. metrics는 attempted/succeeded/failed registrations와 pinned bytes high-water를 구분한다.

long-lived pinned memory는 OS 전체에 비용이 있으므로 무한 확대하지 않는다. 이 역시 “pinned가 빠르다”에서 끝나지 않는 운영 조건이다. bounded pool, admission/backpressure, fallback mode를 정하되 correctness lifetime를 우선한다.

page placement drift test는 service startup 뒤 다른 node에서 pool growth를 유도한다. initial buffers local이어도 autoscale/late allocations가 remote일 수 있다. node histogram를 pool generation별로 관측한다. startup benchmark 한 번으로 장기 locality를 보장하지 않는다.

CPU scheduler migration도 host fill bandwidth와 first-touch placement에 영향을 줄 수 있다. CPU affinity를 걸었는지와 memory policy를 별도로 본다. affinity 변경은 already-faulted pages를 자동 이동시키지 않는다. old pool drain/recreate가 필요한 이유다.

PCIe link degradation test는 negotiated width/speed가 expected보다 낮은 상태를 감지한다. topology inventory와 current link status를 startup/incident에 기록한다. software stream tuning으로 physical x8 downtrain을 고치려 하지 않는다.

shared root contention test는 sibling NIC/GPU/storage workload를 하나씩 켠다. H2D payload와 interconnect/root counters가 함께 변하는지 본다. concurrent matrix가 production mix를 반영해야 한다. isolated microbenchmark peak를 SLO capacity로 그대로 쓰지 않는다.

copy chunk pipeline은 2-way/3-way buffers 각각의 ownership를 검증한다. buffers를 늘리면 queue depth와 pinned/HBM bytes가 늘고 tail cancellation cleanup가 복잡해진다. overlap 개선이 실제 critical path를 줄이는지 보고 필요한 최소 depth를 선택한다.

chunk boundaries는 tensor/layout constraints를 보존해야 한다. arbitrary 64MiB slice가 quant block, KV page, layer weight shard 중간을 자를 수 있다. transfer descriptor가 consumer expected unit와 맞는지 본다. byte copy correctness와 semantic chunk completeness를 구분한다.

checksum은 transfer corruption를 잡지만 wrong destination offset/generation도 sample coordinates로 확인한다. 같은 bytes가 D16에 복사되면 source checksum은 맞다. `(destination,offset,generation)` identity를 hash record에 포함한다.

consumer readiness trace는 ReadyDevice17 wait를 실제 kernel stream에서 확인한다. event를 record했지만 wrong stream가 wait하거나 wait가 kernel 뒤에 있으면 무효다. source code 순서와 runtime sequence IDs를 함께 본다.

consumer completion DoneConsume17은 D17 reuse를 보호한다. ReadyDevice17은 source DMA completion/host slot reuse를 보호할 수 있지만 정확한 API/engine semantics와 owner design을 확인한다. 하나의 event로 source와 destination 모든 lifetime을 뭉치지 않는다.

abort at t0 host fill 전이면 slot을 즉시 반환할 수 있다. fill 뒤 copy submit 전이면 descriptor를 취소하고 slot generation을 닫는다. DMA 중이면 Ready completion 뒤 source slot을 반환한다. kernel consumer 중이면 DoneConsume 뒤 D17을 반환한다. request status보다 operation frontier가 cleanup를 정한다.

shutdown은 new transfers admission를 막고 producers/futures를 취소한 뒤 submitted DMA와 consumers를 drain한다. completion를 알 수 없는 mappings/slots를 normal pool로 강제 반환하지 않는다. worker/device allocator epoch를 폐기한다.

wrong-output fixture는 first/middle/last 64-byte lines에 generation-specific pattern을 둔다. chunk copy와 consumer가 모두17 pattern을 읽어야 한다. old event/offset가 D16 또는 partial D17을 publish하면 어느 line에서 first mismatch인지 기록한다.

performance terminal은 local pinned path에서 host call, DMA, compute intervals가 expected graph를 보이고 remote/pageable fractions가 목표 범위 안인 것이다. 100% overlap를 일반 목표로 두지 않는다. copy duration이 compute보다 길면 correct partial overlap가 정상이다.

correctness terminal은 source/destination generation overlap0, sentinel mismatch0, early publish0, unregister/free before completion0이다. performance가 좋아도 하나라도 실패하면 배포를 막는다. correctness와 overlap score를 분리한다.

COPY-54 사후 기록은 “async가 동작하지 않았다”라고 쓰지 않는다. “node1-first-touch pinned path5.6ms와 pageable fallback38%가 compute4ms보다 길었고 stream/event dependency는 정상”이라고 쓴다. 이 문장이 올바른 수정 축을 지킨다.

fix 뒤 기록은 “pool generation18을 node0에서 first-touch/register했고 fallback를 bounded backpressure로 바꿨으며 sequence301–503 dependencies를 유지했다. local DMA2.8ms가 compute4ms와 겹치고 sentinel/ownership soak가 통과했다”라고 쓴다.

이 숫자와 source walk가 있으면 다음 장비에서 bandwidth 값이 달라도 같은 절차를 적용할 수 있다. page placement, pin lifetime, route, engine, stream edge, consumer identity가 물리 path의 변하지 않는 질문이다.

배포 전 reference fixture는 OS/toolkit/framework version과 topology snapshot를 함께 version한다. CUDA 공식 문서는 pageable/pinned host memory와 asynchronous copy/overlap의 조건을 제공하고, Linux topology/NUMA 관측은 현재 host placement를 제공하며, framework source는 buffer와 stream owner를 제공한다. 세 evidence 역할을 섞지 않는다.

공식 문서에서 asynchronous behavior가 조건부로 설명되면 그 조건을 표에 그대로 옮긴다. API suffix `Async`를 unconditional host-return/nonblocking/overlap 보장으로 바꾸지 않는다. memory type, direction, runtime behavior, hardware capability를 current version 문서에서 확인한다.

device property의 async engine 관련 값도 capability evidence 중 하나다. 실제 overlap는 work availability와 dependencies, resource contention에 달렸다. capability true와 measured overlap false가 모순은 아니다. false일 때 stream 수를 늘려 hardware path를 만들 수 있다고 가정하지 않는다.

copy/compute concurrency는 kernel resource 사용과 device architecture에 따라 달라질 수 있다. HBM bandwidth를 크게 쓰는 kernel과 H2D destination traffic가 memory subsystem에서 경쟁할 수 있다. engine interval가 겹쳐도 kernel duration가 늘면 end-to-end 이득이 작다. simultaneous fixture로 본다.

L2/HBM consumer counters는 copy가 준비한 bytes를 kernel이 어떻게 재사용하는지 설명하는 보조다. 하지만 이 장의 terminal은 copy readiness와 lifetime correctness다. kernel tiling/roofline 최적화는 40–47장 범위로 넘긴다. 물리 path 경계를 명확히 유지한다.

CPU cache effects도 과장하지 않는다. host fill가 write-combining/pinned memory에서 CPU read-modify workload에 불리할 수 있고 memcpy staging가 CPU caches를 사용한다. source buffer 생성 workload와 host call time을 측정한다. northbridge 같은 역사적 모델로 modern NUMA/root path를 설명하지 않는다.

IOMMU mode와 DMA mapping 비용은 host/config에 따라 다를 수 있다. 매 transfer map/unmap인지 long-lived registration인지 source owner를 본다. IOMMU off를 성능 만능 knob로 권하지 않고 security/isolation/driver contract를 함께 고려한다. 현재 사건은 measured first divergence만 고친다.

BAR size나 address visibility를 payload H2D bandwidth와 동일시하지 않는다. device가 host mapped memory를 직접 접근하는 path, peer mappings, ordinary DMA copies는 다른 transactions다. topology/source descriptor가 actual path를 말해야 한다.

NUMA node label이 `-1`이거나 ambiguous한 virtualized/container environment에서는 PCI sysfs와 host topology visibility 한계를 기록한다. 보이지 않는 값을 local이라고 가정하지 않는다. controlled placement experiment와 bandwidth/traffic counters로 가설을 좁힌다.

container CPU set과 memory nodes set이 불일치할 수 있다. process가 node0 CPUs만 보지만 allowed memory nodes에 node1도 포함되면 first-touch/allocator behavior를 확인한다. orchestrator placement manifest에 CPU/GPU/memory locality를 함께 둔다.

multi-process workers가 같은 pinned pool을 공유할 수 있는지, each process가 별도 pools를 만드는지 source를 본다. fork 이후 registered memory lifetime/support를 추정하지 않는다. process creation order와 CUDA context initialization contract를 지킨다.

fork/spawn 차이로 pool first-touch thread가 달라질 수 있다. parent가 node1에서 huge pool을 zero한 뒤 node0 child에 넘기면 remote placement가 고정된다. child-local lazy initialization가 개선할 수 있지만 registration/lifetime test가 필요하다.

huge pages를 쓰면 page size와 pin/registration granularity, NUMA placement 단위가 달라질 수 있다. TLB/registration 성능 이득을 가정하지 않고 actual mapping와 support를 검증한다. mixed page sizes도 histogram/ledger에 표시한다.

memory pressure로 NUMA migration/swap/compaction가 개입할 수 있다. pinned pages는 이동/회수 제약을 만들 수 있으므로 pool size와 system headroom을 운영한다. 장기 soak에서 initial placement뿐 아니라 drift와 allocation failure를 본다.

transfer checksum cost가 overlap를 바꾸지 않게 canary sampling을 한다. full checksum은 offline/fault fixture, production은 selected cache-line sentinel/hash를 사용한다. 관측 CPU work가 host staging bottleneck을 만들지 않는지 확인한다.

profiler instrumentation도 implicit synchronization를 넣을 수 있다. production minimal trace와 detailed profiling run을 비교한다. race/overlap가 profiler에서만 사라지면 sequence/event 로그로 원래 behavior를 복원한다. tool output를 절대적 timeline로 보지 않는다.

clock/power state는 measured bandwidth와 kernel duration에 영향을 준다. COPY-54 A/B/C 비교에서 GPU/CPU clocks, power throttling, competing workloads를 기록한다. correctness edges는 이러한 noise와 무관하게 invariant로 검증한다.

statistical summary는 chunk/path strata별 p50/p95/p99를 둔다. remote/pageable minority가 average에 숨을 수 있다. 38% fallback 같은 mixture는 separate distributions로 본다. tail request가 어느 pool/path를 썼는지 trace join한다.

SLO 영향은 copy tail가 TTFT/ITL 어느 stage에 들어가는지 연결한다. prefetch가 충분하면 일부 transfer가 critical path 밖일 수 있고 demand miss는 직접 tail에 들어간다. 이 장은 cache policy를 분석하지 않고 observed consumer wait interval만 stage timeline에 붙인다.

vLLM/SGLang option 변경 검토는 raw→normalized→allocator/transfer consumer로 간다. pinned/offload size/threads 관련 값이 어떤 pool/queue/stream을 실제 변경했는지 source와 effective config로 확인한다. option name만 보고 physical result를 주장하지 않는다.

configuration rollback는 previous raw value보다 known-good effective path를 목표로 한다. allocator default나 toolkit behavior가 release 사이 바뀌었으면 같은 option 값이 다른 physical path를 만들 수 있다. node histogram, pinned status, stream/event trace로 rollback 성공을 판정한다.

release upgrade test는 CUDA 12.x/13.x 숫자만으로 topology가 바뀌었다고 쓰지 않는다. host hardware path는 같을 수 있고 runtime semantics/implementation/compiled code가 변할 수 있다. official release/docs와 measurements를 대조한다.

COPY-54 regression artifact에는 topology graph, pool creation trace, range identity, A–I matrix, sequence graph, source pins, fault results가 있다. raw profiler file만 남기지 않고 bounded human-readable 표와 원본 hash를 함께 둔다.

운영자가 처음 볼 세 질문은 간단하다. source pages는 어디 있는가. source는 정말 current registered generation인가. destination consumer는 어느 completion edge를 기다리는가. 그 다음에야 link, engine, chunk를 조정한다.

이 순서를 지키면 `cudaMemcpyAsync`라는 함수명은 사건 결론이 아니라 한 source call 사실로 제자리로 돌아간다. overlap는 page·pin·route·engine·stream·consumer conditions가 모두 만든 runtime 결과다. 각 condition를 독립적으로 반증하고 필요한 edge만 수정한다.

최종 fault campaign은 pool 생성, registration, copy submit, event record, consumer wait, buffer release 위치에 failure를 넣는다. allocation 뒤 registration 실패면 pageable fallback 또는 admission failure가 명시돼야 한다. 일부 pages만 등록된 range를 full pinned로 publish하지 않는다. registration owner가 rollback에서 정확한 range만 해제한다.

copy submit 전 cancel은 host slot을 반환할 수 있지만 destination reservation도 되돌려야 한다. submit 성공 여부가 불명확하면 event/query 또는 stream drain으로 source DMA read frontier를 확인한다. timeout만으로 X17을 fill pool에 되돌리지 않는다.

Ready event record 실패는 consumer를 launch하지 않고 submitted copy completion를 안전하게 drain한다. wait enqueue 실패도 D17을 ready로 publish하지 않는다. kernel launch 뒤 output drop는 가능하지만 D17 consumer lifetime는 Done event까지 남는다.

pool exhaustion test는 requests를 pinned slot 수보다 하나 더 만든다. extra request가 wait/backpressure, explicit pageable degraded path, fail 중 configured policy를 따른다. 어느 경우든 metrics와 trace reason가 일치한다. silent ordinary allocation를 pinned으로 집계하면 실패다.

NUMA fault test는 initializer affinity를 node1로 강제하고 worker를 node0에 둔다. histogram와 bandwidth가 remote fixture를 재현하고 corrected startup에서 local로 바뀌는지 본다. process CPU affinity만 고치고 old pages를 재사용하면 test가 계속 remote여야 한다.

link contention test는 sibling traffic를 켜 aggregate ceiling를 확인한다. software regression와 external contention를 구분하지만 serving placement가 항상 contention를 만든다면 capacity issue로 기록한다. theoretical link value로 incident를 닫지 않는다.

event generation test는 Ready16 완료 handle을 Ready17에 재사용한다. consumer17이 old completion을 current copy readiness로 읽지 않아야 한다. source X17/D17 tuple과 event generation를 비교한다. late callback가 X18 slot을 publish하거나 release하지 못한다.

double-buffer test는 X0/X1 host fill와 D0/D1 consumer를 교차한다. source DMA completion 전 overwrite, destination consumer completion 전 reuse가 0이어야 한다. addresses가 반복돼도 generation intervals는 겹치지 않는다.

restart test는 pool creation가 매번 correct node/registration policy를 재현하는지 확인한다. warm process만 local이고 cold boot가 remote면 initialization ordering가 아직 불안정하다. autoscale replica와 rolling upgrade에서도 topology manifest를 검증한다.

terminal의 correctness 표는 short/missing bytes, sentinel mismatch, early event, generation overlap, premature unregister/free가 모두0이다. performance 표는 pageable fraction, remote fraction, host preparation, H2D duration, overlap tail, shared-root contention를 path strata별로 보인다.

rollback 후 local pinned A는 2.8ms copy가 4ms compute interval에 들어오고, remote/pageable는 explicit degraded categories로 남거나 admission에서 제한된다. 이 수치는 fixture 결과이며 hardware가 바뀌면 다시 측정한다. invariant와 source drill은 그대로 유지한다.

마지막 승인자는 X17 virtual range에서 page-node histogram, registration7, BDF/root, sequence301–503, D17 consumer와 release를 직접 왕복한다. 한 link라도 추정이면 path를 known이라고 표시하지 않는다. COPY-54가 이 왕복과 fault campaign을 통과해야 물리 overlap 수정이 완료된다.

현장 runbook은 증상별 첫 관측도 정한다. host thread가 copy call에서 오래 머물면 source pageability와 staging를 먼저 본다. host return는 빠르지만 DMA가 느리면 NUMA/root/link contention를 본다. DMA는 빠른데 consumer wait가 길면 stream dependency와 preceding operations를 본다. copy/consumer 모두 빠른데 output이 틀리면 range·generation·destination binding을 본다.

이 분기는 서로 배타적이지 않다. pageable fallback와 remote pages가 함께 있을 수 있고 device synchronize가 추가로 overlap를 없앨 수 있다. 하나를 고친 뒤 같은 matrix를 다시 실행해 남은 first divergence를 찾는다. 첫 성공 지표에서 incident를 닫지 않는다.

source review자는 wrapper option, allocator constructor, transfer submit, event record/wait, consumer kernel, cleanup callback 여섯 위치를 pin한다. 운영 review자는 page histogram, topology, intervals, sentinel, generations를 붙인다. 공식 CUDA reference는 host-memory와 async/stream/event semantics의 범위를 제공한다. 세 묶음이 서로 일치해야 한다.

deployment review자는 CPU set, memory nodes, GPU BDF, pool initialization thread, locked-memory limit, expected fallback policy를 manifest에 둔다. code가 바뀌지 않아도 placement/orchestrator 변경이 path를 바꿀 수 있다. release diff에 infrastructure identity를 포함한다.

capacity review자는 pinned bytes를 무한히 늘리지 않고 workload transfer concurrency에 맞춘 slots와 headroom을 계산한다. 이는 cache hit 손익이 아니라 in-flight physical transfers의 owner 수다. source fill, DMA, consumer stages의 최대 동시 slot을 따로 센다.

terminal 뒤에는 diagnostic device synchronize와 verbose tracing가 제거됐는지 확인한다. 제거 후에도 narrow event edges, sentinel equality, node locality, overlap tail이 유지돼야 한다. instrumentation가 사라지자 race가 돌아오면 수정은 완료가 아니다.

이 마지막 검토까지 끝나면 `cudaMemcpyAsync`는 더 이상 “겹칠 것”이라는 약속이 아니다. current source range와 hardware route, stream graph가 만들어낸 실제 transfer 한 건을 부르는 API다. overlap 여부는 그 transfer의 측정되고 검증된 결과다.

최종 기록에는 known-good trace hash와 재현 명령의 환경 조건도 남긴다. 다음 운영자는 같은 NUMA 배치와 topology에서 A fixture를 재실행하고, 다른 장비에서는 bandwidth 상수만 다시 측정한다. ownership와 dependency 판정은 모든 재현 환경에서 동일하게 적용한다.

CPU page에서 GPU HBM까지는 하나의 bandwidth 숫자가 아니라 연속된 소유권 전이이다. page fault가 물리 node를 정하고 registration이 DMA에 필요한 안정된 host range lifetime을 만들며, driver/IOMMU가 device-visible mapping을 관리하고, PCIe topology와 copy queue가 transfer 시점을 정한다. event가 destination을 consumer에게 넘긴 뒤에야 kernel의 L2/HBM 문제가 시작된다.

pinned memory는 locality가 아니고, mapped pointer는 HBM residency가 아니며, async API는 overlap의 보장이 아니다. BDF link width는 end-to-end payload가 아니고, peer capability는 actual direct route가 아니다. global address space도 HBM과 동의어가 아니다. 이 부정문들은 까다로운 예외가 아니라 오진을 막는 기본 좌표다.

실용적인 최적화는 첫 divergence를 찾은 뒤 이루어진다. pageable staging이면 pool 계약을 고치고, remote pages면 allocation/fault placement를 고치며, stream dependency면 event graph를 고친다. shared upstream이면 placement와 concurrency를 바꾸고, P2P fallback이면 transport와 staging capacity를 명시한다. destination 이후가 느리면 비로소 L2·HBM traffic과 kernel을 본다. 마지막 byte가 소비되고 모든 event·mapping·pool lease가 해제되는 순간까지 ownership ledger를 닫아야 다음 요청도 안전하다. 그것이 이 경로의 종료 조건이다.
