# 42장. cache line·sector·coalescing과 정렬: 주소를 세지 않으면 보이지 않는 낭비

“cache line을 맞추면 빨라집니다.” 성능 회의에서 자주 듣는 말이지만, 이 문장만으로는 무엇을 고쳐야 하는지 알 수 없다. CPU queue object가 같은 cache line을 다투는 문제일 수도 있고, GPU warp의 lane들이 여러 32-byte 구간에 흩어진 문제일 수도 있다. KV cache의 software page가 잘게 나뉜 문제나, quantized weight row가 vector load 경계를 어긴 문제일 수도 있다. 네 현상은 모두 ‘cache’와 ‘정렬’이라는 말을 쓰지만 주소를 소비하는 주체와 수명주기가 다르다.

이 장은 모호한 용어 대신 한 가지 습관을 익힌다. lane마다 실제 byte address를 적고, 참여 lane이 요청한 byte 범위를 합친 뒤, 어떤 transaction 구간을 덮는지 센다. 그다음 source의 alignment predicate, vector width와 tail mask가 이 주소 집합을 어떻게 바꾸는지 읽는다. 입문 독자는 42.1~42.4절에서 손계산을 끝내면 된다. 소스·CUDA 경로 독자는 42.5~42.6절에서 다섯 구현을 따라간다. 운영자는 42.7절의 여섯 사건에서 증상과 강한 증거를 연결한다.

## 42.1 page size를 키웠는데 메모리 traffic이 늘어난 사건

한 팀이 paged KV cache의 page size를 키웠다. block table lookup이 줄고 더 긴 연속 구간을 읽으므로 memory transaction도 줄 것이라고 예상했다. 부하 테스트 결과 scheduler의 metadata overhead는 조금 줄었지만 attention kernel의 DRAM byte와 tail latency는 오히려 늘었다. 회의에서는 “GPU cache line과 KV page 크기가 맞지 않는다”는 설명이 나왔다. 이어서 page size를 32, 64, 128처럼 더 큰 2의 거듭제곱으로 맞춰 보자는 제안이 나왔다.

이 설명에는 세 단위가 섞여 있었다. KV page는 여러 token의 K/V를 한 physical block에 배치하고 block table로 주소를 찾기 위한 software allocation 단위다. warp memory transaction은 한 memory instruction에서 참여 lane들이 요청한 주소를 hardware가 처리하는 단위다. L1/L2 cache는 그 transaction이 이전에 가져온 data를 찾는 저장 계층이다. software page가 16 token이든 64 token이든, 한 warp의 lane이 page 안에서 어떤 head와 dimension을 맡는지가 같다면 한 instruction의 주소 집합은 같을 수 있다. 반대로 큰 page 안에서도 lane들이 서로 다른 head stride를 따르면 transaction이 흩어진다.

실제 원인은 두 가지였다. 첫째, 긴 page의 마지막 부분을 거의 사용하지 못한 짧은 요청이 많아 internal fragmentation이 늘었다. 동일 GPU memory budget에서 resident request 수가 줄고 eviction·reload가 증가했다. 둘째, 특정 head dimension에서 full tile specialization을 벗어나 partial tail path가 선택됐다. 마지막 tile의 참여 lane이 줄고 page boundary clamp가 추가되면서 requested byte 대비 사용한 byte 비율이 낮아졌다. page size는 원인의 한 입력이었지만 hardware cache line과 “맞추는” 문제가 아니었다.

이 사건이 가르치는 첫 원칙은 **software 연속성과 warp 연속성을 구별하라**는 것이다. memory allocation이 연속이어도 lane mapping이 stride를 두고 뛰면 transaction은 흩어진다. physical page가 서로 떨어져 있어도 각 warp가 한 page 안의 연속 dimension을 읽으면 그 instruction은 효율적일 수 있다. block table lookup 횟수, page 내부 단편화, lane address의 transaction 수는 서로 다른 ledger에 적는다.

두 번째 원칙은 **요청 byte와 DRAM byte 사이에 cache가 있다**는 것이다. source 식으로 다섯 32-byte 구간을 예상해도 profiler의 DRAM traffic이 정확히 160 byte라고 보장할 수 없다. 인접 warp가 이미 가져온 sector를 L1/L2에서 재사용할 수 있고, writeback이나 다른 instruction이 traffic에 섞일 수 있다. 주소 계산은 expected requested transaction을 세우는 출발점이며, 실제 DRAM byte는 cache·metric scope까지 포함한 관측 결과다.

세 번째 원칙은 **HBM burst를 source에서 발명하지 말라**는 것이다. CUDA 공식 문서가 device memory transaction과 coalescing을 설명한다고 해서 특정 GPU memory controller가 한 transaction을 어느 channel의 몇 byte burst로 바꾸는지 알게 되는 것은 아니다. 공개된 architecture 문서와 counter 없이 “HBM burst 두 번”이라고 쓰면 설명이 더 구체적으로 보일 뿐 더 정확해지지 않는다. 이 장은 warp address와 CUDA가 공개한 transaction 계약까지만 내려간다.

## 42.2 32개 lane의 address example에서 line·sector transaction을 센다

가장 단순한 fixture부터 시작한다. warp lane `i`가 `float x[i]`를 읽고, `float`는 4 byte라고 하자. base address를 `B`라 하면 lane 주소는 다음과 같다.

```text
addr(i) = B + 4*i,  i = 0..31
```

각 lane은 `[addr(i), addr(i)+3]`의 네 byte를 요청한다. `B mod 32 = 0`이면 전체 요청은 `B`부터 `B+127`까지다. 32-byte 구간은 `[B,B+31]`, `[B+32,B+63]`, `[B+64,B+95]`, `[B+96,B+127]` 네 개다. useful byte도 128이고 덮은 구간 byte도 128이므로 이 단순 모델의 이용률은 100%다.

### 42.2.1 base를 4 byte 밀면 왜 다섯 구간이 되는가

이제 첫 lane이 `B+4`에서 시작한다고 하자. 요청 범위는 `B+4`부터 `B+131`이다. 앞의 4 byte를 쓰지 않지만 첫 32-byte 구간을 건드리고, 마지막 네 byte 때문에 `[B+128,B+159]` 구간도 필요하다. 총 다섯 구간, 160 byte를 덮어 useful 128 byte를 얻는다. 단순 이용률은 80%다.

이 계산에서 중요한 것은 allocation base 자체만이 아니다. `cudaMalloc`이 충분히 큰 정렬을 보장해도 row `r`의 시작은 `base + r*row_stride`다. `row_stride mod 32 != 0`이면 첫 row는 맞지만 다음 row는 밀린다. subview가 base에 element offset을 더해 시작하면 역시 alignment가 바뀐다. “allocator가 256-byte aligned이므로 모든 row가 aligned”라는 결론은 틀리다.

CUDA Best Practices는 runtime allocation이 최소 256-byte aligned라고 설명하지만, program이 만든 offset과 block size가 그 정렬을 보존해야 한다. [CUDA C++ Best Practices Guide 13.3.0 — §10.2.1 Sequential but Misaligned Access](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-c-best-practices-guide/index.html#a-sequential-but-misaligned-access-pattern)

### 42.2.2 stride가 2이면 alignment보다 먼저 주소가 흩어진다

lane `i`가 `x[2*i]`를 읽으면 주소는 `B + 8*i`다. 전체 범위는 약 252 byte지만 그 안의 절반만 쓴다. base를 잘 맞춰도 lane이 두 word 간격으로 뛰므로 더 많은 32-byte 구간을 건드린다. row allocation이 contiguous라는 사실은 coalescing을 보장하지 않는다. lane mapping의 stride가 transaction 수를 결정한다.

KV에서 이 패턴은 HND layout의 head stride, block table로 흩어진 token row, GQA에서 서로 다른 KV head를 맡은 warp에서 생길 수 있다. 반대로 HND라도 warp 하나가 head 하나의 contiguous dimension만 맡으면 head-local load는 효율적일 수 있다. layout 이름만으로 판정하지 않고 kernel의 lane→head/token/d mapping을 대입한다.

### 42.2.3 참여 lane이 줄어드는 tail

head dimension이 warp tile의 배수가 아니면 마지막 instruction에서 일부 lane만 유효하다. 예를 들어 남은 float가 10개면 useful byte는 40 byte다. 주소가 잘 맞아도 두 32-byte 구간을 덮어 64 byte를 요청할 수 있다. mask는 OOB read를 막지만 transaction waste를 자동으로 제거하지 않는다. 여러 작은 tail을 서로 다른 row에서 처리하면 sectors/request가 늘 수 있다.

vector load는 lane 하나가 여러 element를 맡아 instruction 수를 줄일 수 있다. lane마다 16 byte를 읽고 8개 lane만 참여하면 useful 128 byte가 연속일 수 있다. 하지만 16-byte type은 자연 정렬을 요구하고, 각 lane의 vector가 서로 먼 page를 가리키면 warp 전체 주소는 흩어진다. “vectorized”는 lane 내부 폭에 대한 사실이지 warp coalescing 결론이 아니다.

### 42.2.4 requested transaction과 measured DRAM byte

공식 Best Practices의 misaligned copy 예시는 V100에서 인접 warp가 사용하지 않은 cache line을 재사용하여 단순 4/5 예상보다 bandwidth 감소가 완화될 수 있음을 보여 준다. 이 수치를 현재 GPU의 보편 법칙으로 쓰지 않는다. 핵심은 source-level 다섯 구간과 DRAM traffic 사이에 cache reuse가 있다는 반례다.

조사 worksheet에는 `useful bytes`, `covered 32-byte intervals`, `instruction count`, `participating lanes`를 먼저 적는다. profiler 단계에서는 L1/L2 sectors, hit, DRAM bytes를 shape별로 붙인다. predicted covered interval이 늘었지만 DRAM byte가 같다면 cache reuse나 metric 범위를 본다. interval은 같은데 DRAM byte가 늘면 miss, eviction, 다른 instruction 또는 writeback을 찾는다. 주소식과 counter 중 하나만으로 원인을 확정하지 않는다.

### 42.2.5 natural alignment와 warp alignment는 다른 질문이다

16-byte `int4`를 읽는 lane 하나를 보자. pointer 주소가 16의 배수이면 그 lane의 word는 자연 정렬됐다. 그런데 warp lane 주소가 `B + lane*64`라면 32개 lane은 서로 멀리 떨어진 16-byte word를 읽는다. 각 lane은 완벽히 정렬됐지만 warp는 넓은 범위의 많은 구간을 건드린다. natural alignment는 instruction이 word를 올바르고 효율적으로 표현하기 위한 lane-local 조건이고, coalescing은 참여 lane 전체 주소 관계다.

반대 사례도 있다. lane이 4-byte scalar 하나씩 인접하게 읽으면 넓은 vector type이 없어도 warp 주소는 네 32-byte 구간에 조밀하게 모인다. compiler가 여러 scalar instruction을 만들 수 있어 issue 비용은 다르지만 “scalar이므로 uncoalesced”는 아니다. vector width, instruction 수와 transaction 구간을 세 칸에 나눠 적는 이유다.

source에서 `reinterpret_cast<int4*>`를 보면 세 검사를 한다. allocation base가 16-byte aligned인가. row·head·token offset이 16의 배수인가. 남은 element가 16 byte를 온전히 담는가. 첫 조건만 만족해도 subview offset이나 마지막 tail에서 둘째·셋째 조건이 깨질 수 있다. runtime API allocation alignment는 program 내부 분할의 alignment를 자동 보존하지 않는다.

예를 들어 256-byte aligned buffer 안에 header 12 byte 뒤 payload를 놓고 `int4*`로 cast하면 payload base는 16-byte 경계를 어긴다. padding 4 byte를 넣으면 base는 맞지만 row stride가 244 byte라면 다음 row는 다시 4 byte씩 밀린다. row stride를 256으로 padding해야 모든 row가 같은 alignment class를 유지한다. 이 padding의 capacity cost도 ledger에 넣는다.

KV cache에서 token row가 2048 byte이고 head row가 256 byte면 16/32-byte 정렬을 보존하기 쉽다. head dimension 124 fp16이면 head row가 248 byte라 다음 head가 8 byte만큼 alignment class를 바꾼다. kernel이 head마다 `int4`를 기대한다면 첫 head만 맞고 다음 head가 깨질 수 있다. NHD 전체 row copy가 scalar prologue를 처리할 수도 있고, HND warp path가 head별 fallback을 탈 수도 있다. dimension 숫자를 byte stride로 바꾸기 전에는 fast path를 예측할 수 없다.

### 42.2.6 alignment-aware vector helper의 세 구간

alignment-aware helper는 보통 prologue, vector body, tail이라는 세 구간으로 생각할 수 있다. source와 destination의 alignment class가 같다면 앞의 몇 scalar element를 복사해 둘 다 vector boundary에 맞추고, 가운데를 큰 vector로 처리하고, 남은 element를 좁은 load로 끝낸다. 두 pointer의 alignment class가 다르면 같은 prologue로 둘을 동시에 맞출 수 없어 vector body를 포기할 수 있다.

source `S mod 16 = 8`, destination `D mod 16 = 8`인 fp16 row라면 앞의 4 element, 8 byte를 scalar/좁은 vector로 처리한 뒤 둘 다 16-byte aligned가 된다. row 길이가 충분하면 나머지 대부분을 `int4`로 옮길 수 있다. 하지만 `S mod 16 = 8`, `D mod 16 = 0`이면 같은 element offset을 더할 때 두 pointer의 차이 8 byte가 유지된다. 16-byte load/store를 동시에 자연 정렬시키지 못하므로 좁은 vector가 필요하다.

짧은 row에서는 prologue와 tail이 전부가 된다. head dimension이 10 fp16, 즉 20 byte인데 시작이 8 byte 밀려 있다고 하자. alignment까지 8 byte를 처리하고 남은 12 byte는 16-byte body를 만들지 못한다. helper 호출이 `vectorized`라는 이름이어도 실제 실행은 scalar·8/4-byte path뿐일 수 있다. fast path hit 비율을 shape별로 보는 이유다.

tail mask는 memory safety와 useful-byte efficiency를 동시에 결정한다. 남은 6 byte에 8-byte load를 하고 두 byte를 버리는 구현은 allocation padding이 있고 logical boundary를 넘지 않는다는 계약이 필요하다. 다른 row나 request의 data를 읽으면 비록 결과에 mask를 곱해도 isolation·fault 문제가 생길 수 있다. 안전한 helper는 remaining count와 physical allocation boundary를 함께 알아야 한다.

source/destination dtype이 다르면 element count 기반 vector가 같은 byte 폭을 뜻하지 않는다. fp16 source 8 element는 16 byte지만 fp8 destination 8 element는 8 byte다. quantize-and-cache helper는 source와 destination에 다른 vector type 또는 operation을 쓸 수 있다. `VEC_SIZE=8`만 보고 두 주소가 같은 transaction pattern이라고 쓰지 않는다. 각 pointer의 byte interval을 따로 센다.

이 세 구간 모델은 compiler 구현을 단정하는 설명이 아니다. 실제 helper source에서 alignment predicate와 loop를 확인하기 위한 질문 틀이다. prologue가 있는지, unaligned intrinsic을 쓰는지, tail을 mask하는지, 완전히 다른 specialization을 고르는지는 구현마다 다르다. 중요한 것은 평균 row에서 vector body만 보고 boundary shape의 behavior를 추정하지 않는 것이다.

**이 주소 예를 공식 계약·source·counter·serving 결과까지 수직으로 잇는다.**

독자가 새 kernel을 만났을 때 시작점은 profiler가 아니다. 먼저 한 memory instruction의 logical owner를 찾는다.
요청, token, head, dimension 중 warp와 lane이 무엇을 나눠 맡는지 source에서 읽는다. launcher의 tensor shape,
stride, block table과 specialization predicate를 대입해 physical address 식을 만든다. full, tail, misaligned-but-
supported 세 fixture를 고른다.

두 번째로 lane 0~31의 byte interval을 쓴다. lane-local width와 natural alignment를 확인하고 false predicate인
lane을 제거한다. interval 합집합과 교차하는 32-byte sector index를 센다. useful byte, covered sector byte,
instruction 수와 lane participation을 따로 둔다. stride scatter는 span이 아니라 실제 touched sectors를 센다.
여기까지는 CUDA 공식 coalescing 계약을 적용한 예측이다.

세 번째로 compiler artifact를 확인한다. C++ vector cast나 Triton hint가 실제 어느 load width와 predicate로
내려갔는지 PTX/SASS 또는 compiler report에서 본다. compiler가 scalar로 쪼갔다면 source의 16-byte vector
예측을 runtime instruction 사실로 쓰지 않는다. 반대로 scalar source가 합쳐졌다면 instruction count가 줄 수
있다. build commit, CUDA/toolkit, arch flags, kernel binary digest를 고정한다.

네 번째로 Nsight counter를 선택한다. 해당 GPU와 Nsight Compute version의 metric description에서 global-load
request/sector, L1/L2 hit 또는 sector, DRAM byte, long scoreboard, eligible warp와 achieved occupancy의 scope와
denominator를 읽는다. 비슷한 이름의 옛 metric을 새 GPU에 그대로 매핑하지 않는다. 존재하지 않는 metric은
0이나 추정값으로 채우지 않고 requested address와 available cache/DRAM observation으로 결론 강도를 낮춘다.

예측 표의 한 행은 이렇다. aligned scalar contiguous fixture는 useful128, expected sectors4다. +4 byte
misaligned는 useful128, sectors5로 sector/useful가 0.03125→0.03906 sector/byte가 된다. stride2는 sectors8,
0.0625다. 10-lane tail은 useful40, sectors2, 0.05다. 이 값은 observed counter가 아니라 address model이다.
counter가 다르면 instruction aggregation, cache/request definition, predication과 source address를 다시 본다.

vector fixture는 별 행이다. lane당16 byte contiguous는 useful512, sectors16으로 sector/byte는 aligned scalar와
같다. instruction이 줄어 issue overhead가 개선될 수 있지만 registers와 alignment requirement가 달라진다.
lane stride64는 useful512, sectors32로 두 배다. “16 sector라 scalar 4 sector보다 나쁘다”가 아니라 같은 useful
byte로 정규화한다.

cache hierarchy 예측은 조건으로 붙인다. working set이 L1에 맞고 policy가 허용하면 repeated sector의 L1
hit가 높아 DRAM effect는 작다. L1을 우회하거나 scan이 크면 L2와 DRAM이 지배할 수 있다. 인접 warp reuse가
있으면 misaligned fifth sector가 재사용될 수 있다. page permutation이 reuse distance를 늘리면 L2 miss와 DRAM
byte가 증가할 수 있다. source address만으로 hit율 숫자를 발명하지 않는다.

latency hiding 예측도 자원 조건으로 붙인다. sector가 늘어도 eligible warps가 충분하면 long scoreboard와
duration 변화가 작을 수 있다. vectorization/padding으로 register 또는 shared-memory가 늘어 occupancy와 eligible
warps가 줄면 byte 감소가 stall로 이어지지 않을 수 있다. pointer dependency가 serial이면 occupancy보다
memory-level parallelism이 제한된다. counter 하나가 아니라 causal chain을 본다.

vLLM cache-write source 산책에서는 slot→block/page offset, source/destination strides, NHD/HND predicate,
vector helper와 tail을 잇는다. [vLLM cache row와 path](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/csrc/libtorch_stable/cache_kernels.cu#L330-L400)
source row와 destination row의 alignment class가 다르면 helper가 어떤 fallback을 택하는지 확인한다. cache
write 개선이 다음 attention layout에 transpose/copy를 추가하지 않는지도 본다.

SGLang MLA source에서는 `kv_loc`과 stride가 row base를 만들고 dimension offset과 mask가 붙는다.
[SGLang verify MLA addresses](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/ops/attention/verify_mla.py#L116-L205)
innermost dimension이 연속이어도 lane/program mapping이 여러 `kv_loc`을 동시에 읽으면 scatter될 수 있다.
split tail과 program shape를 복원한다.

FlashInfer에서는 `sizeof(Vec)×32 == head bytes` static assertion이 compile-time coverage를 증명한다. [FlashInfer vector trait](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/concat_mla.cuh#L40-L190) runtime token/head stride와 base alignment는 별도다.

FlashAttention에서는 virtual→physical page와 partial clamp가 row를 바꾼다. [FlashAttention paged row](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/csrc/flash_attn/src/utils.h#L302-L356)

Marlin에서는 16-byte async-copy helper와 predicated tail, packed layout을 잇는다.
[Marlin async copy](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/csrc/libtorch_stable/quantization/marlin/marlin.cuh#L54-L166)
transaction 이득과 repack/padding artifact byte, shared stage와 occupancy를 함께 기록한다. `aligned`라는 이름을
global pointer alignment로 번역하지 않는다.

serving 결과는 마지막 소비자다. kernel duration이 줄었어도 padding으로 KV capacity가 줄고 eviction이 늘면
ITL과 goodput은 나빠질 수 있다. page size 변경으로 block-table load가 줄어도 fragmentation과 tail이 늘 수
있다. layout 변경이 graph specialization을 깨뜨려 eager fallback을 만들 수도 있다. address 최적화의 terminal은
원래 arrival, context와 concurrency에서 correctness·TTFT·ITL·deadline goodput이다.

관측→반증→선택 기록은 한 문단으로 쓴다. 관측은 shape, symbol, requested/cache/DRAM counters와 ITL이다.
경쟁 가설은 row misalignment, stride scatter, tail/split, cache residency, insufficient hiding과 scheduler gap이다.
주소 계산으로 불가능한 가설을 제거하고 source predicate와 artifact로 instruction을 고정하며 counter로 cache/stall
경로를 선택한다. 수정은 예상 address/sector/resource/capacity 변화를 먼저 쓰고 canary에서 검증한다.

rollback은 source flag만 되돌리지 않는다. layout, padded model/KV artifact, graph capture와 cache generation이
바뀌었다면 old/new request를 drain 또는 격리한다. binary digest와 metric baseline도 이전 것으로 맞춘다.
tail correctness, OOB mask와 output parity를 확인하고 resource occupancy와 serving SLO가 복원되는지 본다.

한 번 더 실제 32-lane 주소표를 작성해 보자. page base `P=0x100000`, token stride 2,048 byte, head stride
256 byte, lane width 8 byte인 NHD/HND-compatible row를 가정한다. token 3, head 2의 row base는
`P + 3×2048 + 2×256 = 0x101A00`이다. 이 주소는 32-byte 경계에 맞는다. lane 0은 `0x101A00..07`,
lane 1은 `08..0F`, lane 2는 `10..17`, lane 3은 `18..1F`를 읽어 첫 sector를 채운다. lane 4~7이 둘째,
8~11이 셋째로 이어져 lane 28~31이 여덟째 sector를 채운다.

subview header 때문에 base가 8 byte 밀리면 row는 `0x101A08`이다. lane 0~2가 첫 sector의 offset 8~31을,
lane 3~6이 다음 sector를 채운다. lane 31의 끝은 원래 256-byte interval을 넘어 ninth sector에 닿는다.
주소식으로 expected 9 sectors를 얻는다. 하지만 compiler가 lane당 두 개 4-byte load로 쪼갰다면 memory
instruction은 둘이고 각 instruction 주소 집합을 따로 세야 한다. 합계 touched sector가 같아도 request
counter와 instruction count는 달라진다.

이번에는 page table이 lane mapping에 끼는 gather를 보자. lane 0~15는 physical page 7의 같은 dimension,
lane 16~31은 page 93을 가리킨다고 하자. 각 half-warp 안 주소가 연속 4-byte라도 두 page base가 멀리 떨어져
최소 두 주소 군을 만든다. 각 군이 64 byte라 alignment에 따라 두 또는 세 sector, 전체 네~여섯 sector다.
logical token이 연속이라는 사실은 physical page가 연속임을 보장하지 않는다. source block-table lookup을
lane별로 대입해야 한다.

반대로 page ID가 `7,93,2,81`처럼 CTA 전체에서 흩어져도 각 warp가 page 하나의 연속 row만 맡으면 warp별
coalescing은 좋을 수 있다. CTA 총 span이나 page permutation만 보고 한 warp transaction이 나쁘다고 쓰지
않는다. 다만 여러 warp의 temporal locality와 L2 set pressure, TLB/page metadata behavior는 달라질 수 있다.
coalescing과 inter-warp locality를 분리하는 사례다.

store도 같은 방식으로 센다. reshape-and-cache가 source contiguous row를 destination paged row로 쓸 때 source
load와 destination store는 서로 다른 base와 stride를 가진다. source는 8 sectors, destination은 9 sectors일
수 있다. profiler에서 global-load efficiency만 보면 write amplification을 놓친다. quantization 경로라면 scale
load/store와 data conversion도 있다. direction별 requested sectors와 bytes를 나누고 writeback/DRAM 관측의
scope를 확인한다.

cache policy 비교 fixture는 주소를 동일하게 고정한다. policy A/B에서 lane addresses, vector width, predicate와
artifact instruction 수가 같아야 cache 정책 효과를 분리할 수 있다. A에서 L1 hit가 높고 B에서 L2 hit가
높더라도 kernel duration이 같을 수 있다. latency가 다른 independent warps에 가려졌거나 bottleneck이 math,
shared-memory, reduction일 수 있기 때문이다. counter 변화는 원인이 아니라 계층 선택의 증거다.

working set도 수치로 둔다. 한 CTA가 64KiB KV tile을 읽고 같은 SM에서 active CTA가 4개면 순간 footprint는
대략 256KiB이며 다른 data와 경쟁한다. 실제 cache allocation/associativity를 이 숫자만으로 단정하지 않지만,
작은 L1 working set 가설이 가능한지 판단할 수 있다. context 증가로 tile reuse distance가 수 MiB가 되면 L1
reuse 기대를 낮추고 L2/DRAM 관측을 본다. architecture별 capacity와 policy는 공식 tuning guide에 묶는다.

latency hiding 수치 fixture를 만들자. memory dependency latency를 설명용 400 cycle, warp가 dependency 사이에
독립 instruction 40개를 가진다고 하자. resident eligible warp 16개가 고르게 교대하면 상당 부분을 가릴 수
있지만 eligible warp 2개라면 긴 idle/stall이 남을 가능성이 크다. 이 숫자를 hardware 보편값으로 쓰지 않는다.
중요한 것은 sector 증가가 memory latency 또는 queue를 늘려도 scheduler가 공급할 independent work가 있으면
duration 변화가 작을 수 있다는 관계다.

Nsight에서 achieved occupancy가 70%인데 eligible warps per cycle이 낮고 long scoreboard가 높다면 resident는
있지만 준비된 warp가 부족한 상태를 의심한다. register를 줄여 occupancy만 80%로 올렸는데 eligible/stall과
duration이 그대로면 occupancy가 원인이 아니었다. 반대로 vector width 확대 뒤 register가 늘어 occupancy,
eligible이 내려가고 long scoreboard/duration이 오르면 byte 이득과 hiding 손실을 함께 계산한다.

memory throughput이 peak에 가깝고 sectors/useful가 나쁘다면 byte amplification을 줄이는 가치가 크다. throughput이
낮고 long scoreboard가 높으면 dependent latency나 insufficient concurrency일 수 있다. throughput도 낮고
GPU idle/launch gap이 크면 kernel alignment보다 scheduler/graph 경계를 먼저 본다. 동일 “memory-bound” label
아래 필요한 수정이 다르다.

misalignment A/B benchmark를 만들 때 offset을 임의 raw pointer 조작으로 만들지 않는다. 실제 API가 허용하는
tensor subview, head/token stride와 page offset 조합을 사용한다. unsupported misaligned vector pointer를
강제로 넘겨 crash시키는 것은 성능 fixture가 아니라 validation fixture다. supported fallback 경로가 있다면
expected scalar/prologue/vector/tail branch를 source에서 쓴다.

counter 수집 범위도 workload와 맞춘다. kernel instance 하나를 profile한 sector와 1초 serving window의 DRAM
byte를 바로 나누지 않는다. request당 해당 symbol launch 수, split 수와 layer 수를 곱해 total expected
instruction/sector 범위를 만든다. 다른 kernels, allocator memset, offload copy와 NCCL traffic이 window에
포함됐는지 분리한다. counter scope가 다르면 수치가 주소식과 맞지 않는 것이 정상이다.

예를 들어 한 decode token이 attention kernel을 32 layers에서 한 번씩 launch하고 각 launch가 KV row load
1,024회, misalignment로 row당 sector 하나를 추가한다면 최대 additional requested sectors는 설명상
`32×1024=32,768`, 약 1MiB sector byte다. 그러나 row reuse, instruction mapping과 실제 launch count가 다르면
수정한다. 이 upper-bound가 profiler total 증가보다 훨씬 크다면 cache reuse나 mapping 가정을 조사한다.

padding의 손익도 byte로 비교한다. row당 8 byte padding이 token·head·layer·K/V마다 반복되는지, token row
끝에서 한 번만 붙는지 source layout에 따라 비용이 크게 다르다. “6.67%”를 전체 KV에 적용하기 전에 physical
stride와 allocation shape를 확인한다. padding이 kernel fast path를 열어 10% duration을 줄여도 KV capacity
7% 감소가 eviction을 20% 늘리면 serving result는 악화할 수 있다.

vector width 선택은 three-way decision이다. narrow scalar/vector는 alignment/tail에 강하지만 instruction 수가
많다. wide vector는 full aligned row에서 issue가 적지만 natural alignment와 register demand가 커질 수 있다.
alignment-aware helper는 prologue/body/tail로 중간을 제공하지만 짧은 row에서는 prologue/tail overhead가
지배한다. workload head dimensions와 row offsets histogram으로 specialization coverage를 계산한다.

histogram이 중요하다. 평균 head dimension 128이어도 모델별 traffic의 40%가 120, adapter/quant path의 20%가
96이면 128 fast path만 최적화한 benchmark는 production을 대표하지 않는다. context remainder, page boundary,
split count와 alignment class를 bounded bucket으로 수집한다. raw pointer나 request ID는 metric label로 쓰지
않는다.

correctness fixture는 full/tail/OOB를 모두 포함한다. tail mask가 false lane의 address 계산 자체를 allocation
밖으로 만들더라도 load가 정말 predicated되는지 artifact를 확인한다. clamp가 마지막 valid row를 중복 읽고
나중 mask로 제거하는 설계라면 masked value가 softmax/reduction에 들어가지 않는지 본다. misaligned wide
access는 성능만 아니라 undefined/incorrect behavior 가능성이 있으므로 natural alignment contract를 우선한다.

source upgrade에서는 함수명보다 predicate diff를 본다. `head_stride == head_size`, vector trait static assertion,
partial block clamp, cp.async predication과 launcher alignment validation이 바뀌었는지 확인한다. CUDA toolkit
upgrade가 compiler instruction을 바꿨다면 source가 같아도 artifact branch가 달라질 수 있다. 새 counter
baseline은 binary digest와 함께 만든다.

rollout 문서의 자동 중단선은 `wrong output 또는 OOB 1건`, unexpected fallback rate, KV capacity loss,
eviction/offload amplification, long-scoreboard와 ITL p99다. sector counter가 목표보다 조금 높다는 이유만으로
rollback하지 않지만 correctness나 capacity terminal이 깨지면 즉시 중단한다. optimization metric과 product
guardrail의 우선순위를 명확히 한다.

최종 회고 문장은 “alignment를 맞췄다”가 아니라 다음처럼 쓴다. “lane 0~31의 8-byte intervals가 row base
mod32=8에서 sectors 0~8을 덮었고, aligned layout에서 0~7로 줄었다. artifact는 같은 vector load를 유지했고
L2 hit는 같았으며 requested sectors와 long scoreboard, duration이 예상 방향으로 내려갔다. padding byte와
resident capacity 변화가 없고 production ITL/goodput이 회복됐다.” 반대 결과라면 같은 인과 열에서 처음
어긋난 지점을 찾는다.

실전에서 counter 예측을 더 구체화해 보자. aligned float fixture를 한 warp가 1,000회 반복하면 source model의
requested sectors는 4,000, requested sector byte는 128,000이다. +4 byte offset이면 5,000과 160,000이다.
useful byte는 둘 다 128,000이다. 따라서 address-level amplification은 1.0→1.25다. profiler가 L1 request
sectors를 이 scope로 제공한다면 비슷한 방향을 기대하지만, compiler가 loop를 unroll하거나 load를 합치고
predicate가 바뀌면 request count부터 다시 센다.

L2 sector 관측이 4,000→4,200만 늘었다면 나머지 misaligned fifth sectors가 L1 reuse됐을 가능성, metric
scope와 sampling을 조사한다. DRAM sector가 오히려 같다면 인접 warp reuse 또는 L2 hit가 설명 후보다.
그렇다고 misalignment가 공짜라고 결론내리지는 않는다. L1 bandwidth와 request 처리, 다른 working set의
eviction 비용이 남을 수 있다. kernel duration과 stall이 안정적이면 현재 workload critical path에는 노출되지
않았다고 제한적으로 말한다.

반대로 requested sectors가 4→5 예상인데 observed는 8→10처럼 정확히 두 배라면 K와 V 두 load, source와
destination 두 방향, 또는 compiler가 만든 두 instructions가 metric에 포함됐는지 본다. 비율이 맞는다는 이유로
주소식을 즉시 확정하지 않는다. 같은 배율을 만드는 구조가 여럿이다. source instruction owner와 counter
filter를 연결해야 한다.

cache policy A에서 L1 hit 70%, B에서 20%라는 관측이 나왔다고 하자. requested sectors가 같고 L2/DRAM이
B에서 늘면 policy 영향이 plausible하다. 하지만 B가 더 빠를 수도 있다. L1 lookup/competition을 피하고 더
많은 concurrent streaming request를 처리하거나 다른 hot data의 L1 residency를 보호할 수 있기 때문이다.
policy 선택은 해당 load뿐 아니라 kernel 전체와 이웃 loads의 hit/stall을 본다.

latency hiding을 request arrival과도 연결한다. microbenchmark는 grid가 커 항상 많은 CTA를 공급하지만 decode
batch가 작으면 grid가 작아 SM 일부만 사용하고 eligible warp가 부족할 수 있다. 같은 kernel과 같은 row sectors라도
batch 1에서 misalignment latency가 노출되고 batch 64에서는 가려질 수 있다. 따라서 Nsight fixture에 batch,
sequence count, grid/CTA와 waves per SM을 기록한다. production ITL bucket과 다른 대규모 synthetic grid를
근거로 tail을 약속하지 않는다.

prefill은 decode와 다르다. prefill은 query/token parallelism이 풍부하고 compute도 많아 일부 memory latency를
가릴 수 있다. decode long context는 query가 적고 KV scan 비중이 커 bandwidth와 latency가 더 직접 노출될
수 있다. 같은 address improvement가 TTFT와 ITL에 다른 효과를 낸다. 책의 serving 결론은 phase별로 나눈다.

multi-tenant batch에서는 한 요청의 padded row가 다른 요청의 capacity와 schedule을 바꾼다. padding 최적화
canary가 대상 request kernel을 5% 빠르게 했어도 active sequence capacity가 64→60으로 줄면 queue와 TTFT가
나빠질 수 있다. request-local kernel counter와 server-level goodput을 같은 causal record로 잇는다. padding
byte를 “작은 정렬 비용”으로 뭉개지 않는다.

graph capture도 함께 확인한다. 새로운 alignment branch가 runtime pointer mod 값에 따라 서로 다른 workspace,
kernel 또는 grid를 선택하면 captured graph의 static assumptions와 충돌할 수 있다. graph fallback이 생기면
launch overhead와 batch shape가 바뀌어 ITL이 나빠진다. counter 변화가 alignment 때문인지 graph/eager path
변경 때문인지 dispatch trace에서 분리한다.

quantized KV에서는 scale/zero-point row가 data row와 다른 width와 stride를 가진다. data load가 perfectly
coalesced해도 scale을 lane/head마다 scatter하면 dependent latency가 남을 수 있다. fp8 destination은 source
fp16과 byte width가 달라 같은 element vector count가 서로 다른 sectors를 만든다. source와 destination,
data와 scale 각각의 address worksheet를 작성한다.

Marlin weight kernel에서는 global→shared copy 뒤 shared-memory bank와 MMA consumption이 이어진다. global
sector를 줄인 layout이 shared bank conflict나 extra permutation을 만들 수 있다. 42장은 global requested
address를 소유하지만 source path의 다음 consumer side effect는 기록한다. global counter만 좋아지고 stage
wait 또는 compute issue가 나빠지면 end-to-end 이득이 없다.

공식 문서의 숫자는 안전 난간이다. compute capability 6.0+ 32-byte transaction 요약과 allocation alignment
설명은 손계산의 근거다. 특정 V100 benchmark의 bandwidth, 특정 Nsight metric 이름과 특정 architecture cache
behavior는 그 판에 묶는다. CUDA 12.x와 13.x 문서 절 위치가 달라졌다는 이유만으로 hardware coalescing
규칙이 바뀌었다고 쓰지 않는다. toolkit 변화는 compiler artifact와 metric availability에서 별도로 검증한다.

사건 문서에는 기각한 수정도 남긴다. allocator 512-byte alignment 확대는 subview offset을 고치지 않아
기각했다. token row 2,048-byte padding은 row sector 이득이 없고 6.67% capacity를 잃어 rollback했다. device-wide
synchronize는 memory ordering 사건이 아니며 latency hiding을 없애므로 사용하지 않았다. split을 완전히 끄는
수정은 긴 context parallelism을 해쳐 length-aware branch를 남겼다. 실패한 선택의 이유가 다음 회귀의
탐색 공간을 줄인다.

최종 검증은 세 시각을 맞춘다. source revision에서 address/predicate를 고정한 시각, binary가 빌드된 시각,
profile과 serving trace가 수집된 시각이다. rolling deploy에서 old/new binary가 섞였으면 counter와 ITL을
generation별로 분리한다. source link가 최신이어도 running pod가 옛 cubin이면 근거 사슬이 끊긴다.

독자가 counter 이름을 모르는 환경에서도 최소 판정은 가능하다. source로 lane interval과 expected sector를
만들고, kernel duration과 total device-memory byte처럼 제공되는 관측을 붙인다. cache-level counter가 없으면
“L2 miss가 원인”이라고 확정하지 않고 address amplification 가능성과 duration 상관까지만 쓴다. 필요한
추가 증거와 지원 GPU/tool을 명시한다. 빈 칸을 자신감 있는 추측으로 채우지 않는 것이 정확성이다.

반대로 counter가 풍부해도 source 주소를 생략하지 않는다. L1 sectors/request 증가, L2 hit 하락, DRAM byte와
long scoreboard 증가가 함께 보여도 원인은 row misalignment, page scatter, split tail, 새로운 instruction,
working-set expansion 중 하나 이상일 수 있다. first divergence의 dispatch와 lane mapping을 찾아야 수정 대상이
정해진다. 관측량이 많다는 사실은 인과가 자동으로 생겼다는 뜻이 아니다.

작은 자동 worksheet를 만들 수는 있다. 입력은 base modulo 32, participating lanes, lane별 byte width/stride와
row stride다. 출력은 touched sector set, useful/covered byte와 alignment class histogram이다. 다만 script가
kernel의 lane mapping을 발명하게 해서는 안 된다. source에서 복원한 mapping을 사람이 입력하고 결과를
fixture로 검산한다. broadcast, overlapping intervals와 predicated lane도 처리해야 한다.

worksheet의 unit test는 A~H fixture다. contiguous scalar 4 sectors, +4 offset 5, stride2 8, contiguous
16-byte vector 16, stride64 vector 32, aligned 10-lane tail 2, offset28 tail 3, broadcast unique sector 1을
기대한다. fp16 row 128 aligned8/misaligned9, row120 aligned와 offset8 모두8도 넣는다. 이 결과가 틀리면
production 주소 분석 전에 계산 도구부터 고친다.

source review에서 unknown도 정상 값이다. compiler artifact를 확보하지 못했다면 load width는 expected이지
observed가 아니다. cache operator의 architecture effect를 확인하지 못했다면 policy는 semantic hint다.
Nsight counter가 multiplex/sampling되면 exact equality 대신 방향과 오차를 쓴다. 이 구분이 책의 설명을
덜 단정적으로 보이게 할 수 있지만 독자가 어디를 더 파야 하는지를 정확히 알려 준다.

마지막으로 성능 개선을 percentage 하나로 끝내지 않는다. aligned cohort kernel -6%, KV capacity 변화0,
ITL p99 -4%, goodput +2%처럼 각 층을 적는다. tail cohort가 +3% 느려졌다면 traffic weight와 SLO를 보고
length-aware dispatch 여부를 판단한다. 평균 이득이 rare correctness boundary를 덮지 않게 full/tail output
parity와 OOB 검사를 별도 hard gate로 둔다.

새 GPU 세대에서 같은 fixture를 다시 돌릴 때도 주소-level expected sectors는 출발점으로 유지하되 cache와
counter 결과는 새 architecture 문서로 갱신한다. allocation, compiler, kernel symbol과 workload를 고정하지
못하면 세대 비교를 보류한다. “더 최신 GPU이므로 misalignment penalty가 없다”거나 “sector 수가 같으므로
성능도 같다”는 결론을 피한다. hardware가 낭비를 일부 숨겨도 extra requested work와 capacity side effect를
설계에서 제거할 이유는 남는다.

이 장의 수치 fixture는 튜닝 정답표가 아니다. 실제 request의 base, stride, vector, predicate를 넣는 계산
틀이다. 숫자를 바꾸어도 같은 방법으로 sector 집합을 얻고, 그 뒤 cache/hiding/serving 계층을 순서대로
검증할 수 있어야 한다. 이 재계산 가능성이 특정 GPU에서 얻은 단발성 bandwidth 숫자보다 오래가는 지식이다.

마지막 exit condition은 독자가 임의의 warp 주소표를 받았을 때 32 sector를 손으로 셀 수 있는가다. 이어 그
예측이 어느 cache level과 counter에 해당하지 않는지 말할 수 있어야 한다. serving source에서 lane mapping과
vector/tail predicate를 찾고, cache hierarchy와 eligible warps가 sector 낭비를 latency로 노출하는 조건을
설명하며, padding·layout 수정의 capacity side effect와 rollback을 닫을 수 있어야 한다.

## 42.3 같은 ‘line’이라는 말 아래 숨은 네 단위

주소를 세기 전에 용어의 소유자를 분명히 하자. 서로 관련은 있지만 같은 것은 아니다.

### 42.3.1 CPU cache line: locality와 coherence의 단위

CPU cache line은 CPU core가 memory hierarchy에서 data를 가져오고 coherence state를 관리하는 단위다. scheduler queue의 head와 tail counter가 같은 line에 놓여 서로 다른 core가 계속 쓴다면 false sharing이 생길 수 있다. tokenizer table을 순차로 읽으면 spatial locality 덕분에 한 번 가져온 line 안의 여러 byte를 쓸 수 있다. 여기서 중요한 주소는 CPU virtual/physical mapping, core와 NUMA node, coherence participant다.

GPU serving process의 CPU side에도 이 문제가 존재한다. HTTP parser, scheduler state, completion queue, block allocator의 metadata는 CPU에서 돈다. 하지만 CPU line을 64 byte라고 가정해 padding했다고 GPU KV read가 coalesced해지는 것은 아니다. CPU가 관리하는 block-table array의 line locality와 GPU가 device copy를 읽는 transaction은 주소 공간과 실행 주체가 다르다. 둘을 연결하려면 host에서 device로 metadata가 언제 복사되고 GPU lane이 어떤 device address를 읽는지 별도 경로를 추적해야 한다.

### 42.3.2 GPU transaction과 sector: 한 warp instruction의 주소 집합

CUDA Best Practices Guide는 global memory load/store를 warp thread의 접근에서 가능한 적은 transaction으로 합친다고 설명한다. compute capability 6.0 이상에 대한 요약에서는 참여 thread의 주소를 처리하는 데 필요한 32-byte transaction 수로 설명한다. 32개 lane이 인접한 4-byte word를 읽으면 useful byte는 128 byte이고, 32-byte 경계에 맞으면 네 구간이 필요하다. [CUDA C++ Best Practices Guide 12.9.1 — §9.2.1 Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

여기서 “sector”라는 말을 쓸 때도 조심한다. profiler가 L1/L2 sector나 sectors/request를 노출할 수 있지만 정확한 metric 정의와 availability는 architecture·tool version에 맞춰 읽어야 한다. 32-byte 구간을 설명하는 교육 모델을 모든 cache 내부 구현의 완전한 설명으로 확장하지 않는다. source에서 우리가 확정할 수 있는 것은 lane address와 instruction의 requested range다. cache allocation·promotion·replacement는 해당 architecture 공식 자료와 관측이 더 필요하다.

### 42.3.3 software KV block: token state의 allocation과 identity 단위

KV block은 token 범위의 K/V state를 할당하고 block table로 찾고 reference count와 generation을 관리하는 software 단위다. page size 16 token은 hardware가 16 token을 한 transaction으로 읽는다는 뜻이 아니다. 한 token의 KV row만 해도 layer, KV head, head dimension, dtype에 따라 수백~수천 byte다. 여러 warp와 여러 instruction이 나눠 읽는다.

software page 크기는 간접적으로 transaction에 영향을 줄 수 있다. page boundary가 attention tile 가운데 오면 다음 physical block-table entry를 읽고 base를 바꿔야 한다. partial last page에서는 참여 row와 mask가 달라진다. page stride가 vector natural alignment의 배수를 보존하는지도 중요하다. 그러나 이 영향은 `page_size → cache line 일치`가 아니라 `page_size → address equation과 specialization predicate → lane address 집합`의 사슬로 증명한다.

### 42.3.4 HBM과 memory controller: 공개 근거 밖을 넘지 않는다

HBM은 L2 뒤의 device memory를 제공한다. coalescing이 나쁘면 일반적으로 불필요한 data movement와 bandwidth pressure가 늘 수 있다. 하지만 PTX의 `ld.global.v4` 한 줄이나 C++의 `int4` 한 번이 HBM command 하나에 대응한다고 말할 수 없다. cache hit이면 HBM까지 가지 않을 수 있고, memory controller는 여러 request를 다르게 스케줄할 수 있다. ECC, partition, compression 같은 요인도 architecture별이다.

따라서 본문에서는 세 층을 표시한다. source에서 복원한 requested address, CUDA 공식 계약으로 계산한 transaction 구간, profiler에서 관측한 cache·DRAM 결과다. HBM burst 수는 공개 근거가 있을 때만 별도 사실로 둔다. 이것은 디테일을 포기하는 태도가 아니라 관측 가능한 마지막 지점을 정확히 표시하는 태도다.

## 42.4 fp16 KV row를 주소 worksheet로 해부한다

fixture는 page당 16 token, KV head 8개, head dimension 128, fp16 2 byte다. NHD layout에서 한 page의 모양을 `[token, head, d]`로 생각하자. 단순 contiguous layout이라면 head stride는 128 element, token stride는 `8×128=1024` element, page stride는 `16×1024=16384` element다.

```text
addr(page, token, head, d)
 = base + (page*16384 + token*1024 + head*128 + d)*2
```

한 head row는 `128×2=256` byte다. lane마다 fp16 4개, 즉 8 byte vector를 읽고 lane 0~31이 `d=4*i..4*i+3`을 맡으면 256 byte를 연속으로 덮는다. base와 row 시작이 32-byte aligned라면 여덟 32-byte 구간이다. lane마다 vector가 자연 정렬되어야 하고 row stride 256 byte가 다음 head의 정렬을 보존한다.

### 42.4.1 NHD fast path가 보는 연속성

NHD에서 한 token의 모든 head가 이어져 있으면 kernel은 `num_heads*head_size`를 하나의 contiguous row처럼 복사할 수 있다. 이 row는 `8×128×2=2048` byte다. thread block이 vector chunk를 나눠 맡을 때 `key_stride`, destination block/page offset과 base alignment가 모두 vector width를 보존해야 한다. 어느 하나가 어긋나면 alignment-aware helper가 scalar prologue/tail 또는 좁은 vector로 갈 수 있다.

KV cache가 fp8이면 source와 destination element size가 다르고 per-head scale read가 추가될 수 있다. 단순 copy가 아니라 quantize/cast op가 들어가며 scale layout도 lane address를 만든다. data row만 coalesced해도 head별 scale가 scatter되거나 branch를 만들 수 있다. 그래서 kernel의 fast predicate가 layout뿐 아니라 `kv_scale_stride == 0` 같은 조건을 포함한다.

### 42.4.2 HND에서는 head-local row를 warp 하나가 맡는다

HND layout을 `[head, token, d]`로 두면 head stride가 page 안의 `16×128=2048` element, token 안의 dimension은 여전히 contiguous다. 모든 head를 하나의 row로 합치면 head 사이에 token stride 구조가 끼므로 NHD fast path를 그대로 쓸 수 없다. 대신 warp 하나가 특정 head의 128 dimension을 복사한다. warp lane은 head-local 주소에서는 연속이지만 여러 warp가 서로 다른 head base를 본다.

이 구분은 “HND는 coalescing이 나쁘다”는 단순 결론을 막는다. 한 instruction의 warp가 한 head row를 읽으면 효율적일 수 있다. 다만 head 수가 적거나 head dimension tail이 작으면 warp 활용률이 달라지고, loop로 여러 head를 맡는 scheduling 비용이 생긴다. layout 선택은 address뿐 아니라 backend가 기대하는 attention read pattern과 cache capacity까지 포함한다.

### 42.4.3 page boundary와 block table indirection

logical token 15와 16은 연속이지만 page size 16이면 서로 다른 physical block에 있을 수 있다. token 16의 address는 이전 page base에 token stride를 더하는 대신 block table에서 다음 physical page id를 읽고 `page_stride`를 곱한다. 두 physical page가 device allocation에서 이웃일 필요는 없다.

attention tile이 token 8~23을 읽으면 가운데 page boundary가 있다. loader는 첫 page의 row를 읽고 다음 block-table entry로 base를 바꾼다. warp memory instruction 하나가 boundary 양쪽을 무조건 합친다고 가정하지 않는다. kernel은 tile을 page별로 나누거나 gather layout을 만들 수 있다. 이때 page size는 block-table lookup과 boundary branch 수를 바꾸지만, 각 page 내부 row의 정렬은 `page_stride mod vector_alignment`가 결정한다.

### 42.4.4 page size를 키우는 손익은 두 ledger로 센다

page size를 16에서 32 token으로 키우면 page stride는 두 배가 된다. row alignment가 원래 보존됐다면 두 배 stride도 보존되므로 한 row의 transaction 수가 줄지 않을 수 있다. block table entry와 boundary 수는 줄어든다. 반면 평균 요청이 17 token이라면 마지막 page의 unused token slot이 크게 늘 수 있다. resident capacity 저하가 eviction과 DRAM traffic을 간접적으로 늘린다.

첫 ledger에는 kernel address 비용을 적는다. block-table load 수, boundary branch, row의 covered interval, tail mask다. 둘째 ledger에는 allocator 비용을 적는다. allocated bytes, useful token bytes, internal fragmentation, resident request와 eviction이다. 두 ledger를 합친 뒤 end-to-end 결과를 본다. page size 하나를 hardware line 하나에 맞추는 식으로 결론을 내리지 않는다.

## 42.5 KV row가 source에서 실제 주소가 되는 순간

이제 worksheet의 기호를 실제 구현의 변수로 바꾼다. 목표는 함수 이름을 외우는 것이 아니라 `logical slot → physical block/page offset → row base → lane vector → tail`의 경로를 닫는 것이다. vLLM과 SGLang은 같은 KV를 다루지만 layout predicate와 lane mapping이 다르다.

### 42.5.1 vLLM reshape-and-cache: slot에서 NHD/HND 분기로

vLLM cache kernel은 먼저 `slot_idx / block_size`로 physical block index를, `slot_idx % block_size`로 page 안 offset을 구한다. source row는 `key + token_idx*key_stride`, destination row는 `key_cache + block_idx*block_stride + block_offset*page_stride`다. 이 네 stride가 42.4절 주소식의 실제 입력이다. [vLLM v0.27.1 — `csrc/libtorch_stable/cache_kernels.cu:330-345` — slot과 row base](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/csrc/libtorch_stable/cache_kernels.cu#L330-L345)

그다음 `head_stride == head_size`인지 검사한다. 참이면 head들이 연속인 NHD로 보고 `num_heads*head_size` 전체를 alignment-aware vector helper로 복사한다. source dtype이 2 byte면 `VEC_SIZE=8`, 그 밖에는 4를 고른다. 여기서 8은 warp transaction 수가 아니라 lane/helper가 다루는 element vector 폭이다. base와 destination offset이 필요한 natural alignment를 만족하는지 helper 내부 분기가 따로 결정한다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- predicate가 거짓이거나 per-head scale가 있으면 warp별 head 경로로 간다.
- lane은 `threadIdx.x & 31`, warp id는 `threadIdx.x >> 5`다.
- warp마다 head를 선택하고 `head*head_stride`로 destination base를 옮긴 뒤 32개 lane이 head-local `head_size`를 복사한다.
- source에서는 head가 연속이어도 destination layout이 HND일 수 있다.
- per-head scale이면 scale address `head*kv_scale_stride`도 더해진다.
- [vLLM v0.27.1 — `csrc/libtorch_stable/cache_kernels.cu:346-400` — NHD fast path와 HND/per-head path](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/csrc/libtorch_stable/cache_kernels.cu#L346-L400)

독자는 이 분기에서 세 주소 집합을 적는다. NHD data row, HND head-local row, per-head scale다. NHD는 큰 연속 copy에 유리할 수 있지만 thread block의 마지막 vector tail이 생긴다. HND는 warp별 row가 작아 head dimension이 32×vector 폭보다 작으면 inactive lane이 늘 수 있다. scale load는 data byte에 비해 작아도 dependent instruction과 cache miss를 만들 수 있다. “NHD가 무조건 빠르다”가 아니라 선택 backend와 다음 attention read layout까지 보아야 한다.

- Triton 버전은 Python source에서 stride를 더 직접 노출한다.
- kernel 인자에 source key/value stride, block/head/dimension/page stride가 들어가고, block index와 page offset, current head·dimension으로 pointer를 만든다.
- quantized cache 경로에서는 scale cache도 block·slot·head stride를 가진다.
- launcher는 실제 tensor의 `.stride()`를 넘기므로 contiguous라는 추정 대신 runtime tensor layout 계약을 읽을 수 있다.
- [vLLM v0.27.1 — `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:190-265` — quantized KV pointer와 mask](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py#L190-L265)

Triton의 `tl.multiple_of` 같은 표현을 발견해도 그것이 runtime assert인지 compiler hint인지 구분한다. hint가 틀린 input에 대한 validation은 caller가 보장할 수 있다. source note에는 어느 shape·stride predicate가 그 보장을 만드는지 연결해야 한다. hint가 있으므로 모든 pointer가 aligned라고 쓰면 caller validation 변화나 subview offset을 놓친다.

### 42.5.2 SGLang verify MLA: dimension은 연속인데 token row가 흩어질 수 있다

SGLang의 MLA verification attention 경로에서는 `kv_loc`이 logical token을 physical KV buffer row로 바꾼다. K base는 `kv_loc * stride_buf_kbs`, V도 자신의 row stride를 곱한다. 그 뒤 dimension offset을 더해 load한다.

한 token row 안의 dimension은 연속일 수 있지만, warp 또는 program instance가 여러 `kv_loc`을 동시에 다룰 때 row base는 block table 결과에 따라 흩어진다. [SGLang v0.5.18 — `python/sglang/kernels/ops/attention/verify_mla.py:116-205` — KV location과 K/V load](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/ops/attention/verify_mla.py#L116-L205)

이 코드는 “innermost dimension stride가 1이면 coalesced”라는 가설의 반례다. 한 row를 lane들이 나눠 읽는 instruction은 효율적일 수 있지만, program mapping이 lane마다 다른 `kv_loc`과 같은 dimension을 읽게 하면 row stride만큼 scatter된다. 정확한 판정에는 Triton program id와 offset tensor shape를 복원해야 한다. pointer 수식의 마지막 `+ offs_d`만 보고 결론내리지 않는다.

tail은 mask에서 드러난다. sequence split의 마지막 구간과 head dimension 끝에서 일부 offset만 유효하다. mask는 correctness를 지키지만 inactive lane의 빈 vector와 covered interval waste를 남길 수 있다. split 수를 늘렸을 때 compute parallelism은 좋아져도 짧은 tail row가 많아지면 memory efficiency가 떨어질 수 있다. scheduler batch shape와 kernel split policy가 sector 관측에 함께 들어오는 이유다.

HiCache transfer 경로는 다른 종류의 predicate를 보여 준다. one-layer launcher는 destination/source K/V tensor, index와 element dimension을 받아 flatten하고 JIT module을 고른다. all-layer launcher는 pointer array와 source·destination stride bytes를 넘긴다. `element_size % 128 == 0`인지 확인하는 지원 gate가 있지만, 이것을 “128-byte transaction에 맞았다”로 읽으면 안 된다.

- 이는 해당 JIT template이 처리할 element row size 조건이다.
- lane mapping과 pointer base alignment는 kernel 내부에서 따로 확인해야 한다.
- [SGLang v0.5.18 — `python/sglang/kernels/ops/kvcache/hicache.py:68-89` — JIT 지원 predicate](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/ops/kvcache/hicache.py#L68-L89) [SGLang v0.5.18 — `python/sglang/kernels/ops/kvcache/hicache.py:127-198` — one/all-layer launcher](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/ops/kvcache/hicache.py#L127-L198)

HiCache에서 source와 destination stride가 다르면 같은 logical element가 다른 row offset을 가진다. all-layer path의 pointer array는 layer별 base도 바꾼다. 총 transfer byte가 같아도 각 layer row가 vector alignment를 보존하는지, index가 연속 page를 가리키는지에 따라 memory request가 달라진다. CPU↔GPU transfer latency와 GPU kernel global-load efficiency도 구분한다. 이 kernel이 device에서 gather/scatter를 수행하는 부분과 PCIe DMA 자체는 같은 transaction 계약이 아니다.

## 42.6 vector load와 tail을 다섯 구현에서 비교한다

vector type은 유용한 도구지만 coalescing의 결론이 아니다. FlashInfer는 dtype별 vector를 lane에 배분하고, FlashAttention은 page/block tail을 clamp하며, Marlin은 packed weight를 16-byte chunk로 shared memory에 옮긴다. 세 사례는 각각 lane 내부 폭, physical page indirection, tile padding이라는 다른 조건을 보여 준다.

### 42.6.1 FlashInfer MLA concat: `static_assert`가 증명하는 것

FlashInfer의 MLA concat kernel은 dtype에 따라 `NopeVec`와 `RopeVec` type을 고른다. fp16·bf16의 no-PE 부분은 `int2`, RoPE 부분은 `int`처럼 vector width를 정한다. 이어 `sizeof(NopeVec)*32 == QK_NOPE_HEAD_DIM*sizeof(DType)`를 compile-time에 검사한다.

즉 한 warp 32 lane이 vector 하나씩 맡으면 head dimension의 byte 수를 정확히 덮는다는 계약이다. [FlashInfer v0.6.17 — `include/flashinfer/concat_mla.cuh:40-134` — dtype vector traits](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/concat_mla.cuh#L40-L134)

kernel은 token id와 head chunk를 warp id에서 만들고, token stride와 head-row stride로 source base를 계산한 뒤 vector pointer로 reinterpret cast한다. lane id를 vector pointer offset으로 더한다. 이 구조에서 warp 한 개의 lane vector는 연속 head row를 덮을 수 있다.

하지만 token stride나 head row 시작이 vector natural alignment를 보존해야 한다. compile-time head byte equality는 runtime base alignment를 증명하지 않는다. [FlashInfer v0.6.17 — `include/flashinfer/concat_mla.cuh:140-190` — warp mapping과 vector pointer](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/include/flashinfer/concat_mla.cuh#L140-L190)

RoPE vector는 한 번 읽어 여러 local head에 재사용한다. 이것은 global load 수를 줄일 수 있지만 source row가 L2에 hit하는지, register pressure가 occupancy를 바꾸는지는 별도 문제다. prefetch instruction도 future use를 알리는 최적화이지 transaction 수가 줄었다는 보증이 아니다. source에서 관찰되는 의도와 profiler 결과를 분리한다.

### 42.6.2 FlashAttention paged row: full tile과 partial block

이 절의 좌표는 vLLM FlashAttention fork에서 thread id가 block table의 physical row로 내려가는 길을 따른다.

- vLLM FlashAttention fork의 utility는 thread id에서 column offset과 block row offset을 만든다.
- global row를 page size로 나눠 virtual page index와 page offset을 구한 뒤, block table의 physical page id에 `page_stride`를 곱하고 `page_offset*row_stride`, column offset을 더한다.
- worksheet의 page address식이 그대로 보인다.
- [vLLM FlashAttention — `csrc/flash_attn/src/utils.h:302-333` — paged global row offset](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/csrc/flash_attn/src/utils.h#L302-L333)

partial block이면 마지막 유효 row에서 벗어나지 않도록 block row offset을 clamp한다. 여러 thread가 마지막 row를 중복 가리킬 수 있고 나중 mask/layout이 유효 element를 정리한다. 이것은 OOB block-table read를 막는 correctness 경계다. 동시에 full tile과 다른 lane address·participation을 만든다. tail에서 sector 효율이 달라지는 이유를 “분기 overhead” 한 단어로 끝내지 않고 clamp된 row와 column vector를 대입해 센다.

CuTe layout reshape는 paged와 non-paged copy가 같은 logical shape를 갖게 만들지만 stride가 같다고 보장하지 않는다. 주석도 shape equivalence와 stride 차이를 구분한다. 같은 tile abstraction이므로 transaction도 같다고 결론내리면 physical page stride를 놓친다. [vLLM FlashAttention — `csrc/flash_attn/src/utils.h:335-356` — paged tile layout reshape](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/csrc/flash_attn/src/utils.h#L335-L356)

filename이나 specialization 이름의 `blocksize-aligned`도 pointer alignment와 같지 않을 수 있다. sequence length 또는 block size divisibility를 가리키는 predicate인지 launcher에서 확인한다. full block specialization은 tail mask와 split reduction을 줄일 수 있지만 base pointer의 16-byte natural alignment는 별도 전제다. 이름을 주소 계약으로 번역하지 않는다.

### 42.6.3 Marlin: packed tile과 16-byte async copy

Marlin helper는 4, 8, 16-byte global→shared copy와 predicated variant를 제공한다. 지원 architecture에서는 `cp.async`를 사용하고, fallback에서는 대응 integer pointer load/store로 구현한다. 16-byte helper는 `int4` pointer를 사용한다.

따라서 source·shared destination의 natural alignment와 tail predicate가 중요하다. [vLLM v0.27.1 — `csrc/libtorch_stable/quantization/marlin/marlin.cuh:54-166` — async-copy helper와 commit/wait](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/csrc/libtorch_stable/quantization/marlin/marlin.cuh#L54-L166)

- GPTQ repack kernel은 permutation과 packed weight tile을 shared stage로 가져온다.
- `int4` pointer로 permutation/weight chunk를 읽고, 유효 tile인지 predicate를 적용하며, copy group을 commit하고 이후 stage에서 wait한다.
- offline repacking은 quantized bit를 줄이는 것뿐 아니라 runtime lane이 규칙적인 tile을 읽도록 배열을 바꾼다.
- [vLLM v0.27.1 — `csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu:61-115` — packed tile load](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu#L61-L115)

K나 N이 tile multiple이 아니면 padding과 predicate가 필요하다. padding은 runtime address를 규칙적으로 만들지만 model artifact와 workspace byte를 늘린다. 마지막 tile은 일부 quantized value만 useful한데 16-byte vector 또는 shared stage를 예약할 수 있다. transaction 감소만 보면 이득이지만 repack startup, padded resident weight, shared memory와 occupancy를 합치면 end-to-end가 악화될 수 있다.

`mma.sync.aligned`의 `aligned`도 global pointer가 32-byte aligned라는 뜻으로 읽지 않는다. PTX instruction의 operand/thread participation 계약과 global-memory coalescing은 다른 층이다. global→shared loader의 pointer와 vector width, shared layout, MMA fragment mapping을 순서대로 읽는다. 한 식별자의 영어 단어가 여러 계약에서 다른 의미를 갖는다.

### 42.6.4 다섯 source walk를 한 판정식으로 모은다

다섯 구현에 공통으로 적용할 질문은 여섯 개다. 첫째, logical `(request, token, head, d)`를 physical base로 바꾸는 indirection은 무엇인가. 둘째, row·head·page stride는 byte로 얼마인가. 셋째, warp lane은 token/head/d 중 어느 축을 나눠 맡는가. 넷째, lane 하나의 instruction/vector width는 얼마인가. 다섯째, base와 모든 row offset이 natural alignment를 보존하는가. 여섯째, full tile에서 벗어난 tail은 mask, clamp, scalar fallback 중 무엇으로 처리되는가.

답을 얻으면 `vectorized=true` 같은 label 대신 구체적인 예측을 쓸 수 있다. “lane 0~31이 같은 physical page의 연속 8-byte chunk를 읽어 256 byte를 덮고, row base는 32-byte 정렬을 보존한다.” 또는 “lane별 `kv_loc`이 서로 다른 row를 가리켜 각 vector는 정렬됐지만 warp 주소는 scatter된다.” 이 문장이어야 profiler counter가 예상과 다를 때 어느 입력을 다시 볼지 알 수 있다.

## 42.7 여섯 장애에서 주소 가설을 반증하는 법

이제 주소 worksheet를 실제 조사에 쓴다. 여섯 사건은 모두 memory traffic 또는 tail latency로 나타나지만, alignment 하나로 고칠 수 없다. 각 사건은 증상, 첫 가설, 강한 증거, 반증, 수정과 종료 조건을 하나의 서사로 닫는다.

### 42.7.1 사건 1: head dimension을 바꾸자 sectors/request가 늘었다

모델을 바꾼 뒤 KV head dimension이 128에서 120으로 줄었다. KV payload 자체는 작아졌으므로 attention memory traffic도 줄 것이라고 예상했다. 그런데 cache-write kernel과 decode attention의 sectors/request가 늘었고 ITL도 악화됐다. 첫 가설은 새 tensor allocation의 base가 잘못 정렬됐다는 것이었다. 팀은 allocator alignment를 256 byte에서 더 크게 만들려 했다.

worksheet에서 base는 문제가 아니었다. `cudaMalloc` base와 page stride 모두 필요한 경계를 보존했다. 달라진 것은 warp tile tail이었다. 기존 head row는 fp16 256 byte로 lane 32개가 각 8 byte를 읽으면 정확히 닫혔다. 120 dimension은 240 byte다. 같은 mapping에서는 마지막 16 byte가 비거나 일부 lane만 참여한다. 더 중요한 것은 backend가 head dimension 128 전용 specialization에서 일반 masked path로 바뀌었다는 사실이었다. 일반 경로는 head마다 prologue/tail을 처리했고 vector 폭도 달랐다.

base misalignment 가설은 모든 row의 `addr mod 32`가 0이고 offset을 바꾸지 않은 대조에서도 sectors가 늘어난 것으로 기각됐다. strong evidence는 selected kernel symbol, launcher predicate, 마지막 instruction의 lane mask와 covered interval이었다. 120 dimension이 작다는 payload ledger만으로는 설명되지 않던 추가 instruction과 tail sector가 보였다.

수정 선택은 padding 또는 다른 specialization이었다. head row를 128까지 padding하면 규칙적인 load를 회복하지만 layer×token×head 전체 KV capacity가 늘어난다. custom 120 path는 개발·검증 비용이 든다. model architecture가 허용하지 않으므로 dimension 자체를 바꾸는 선택은 없었다. 두 대안을 동일 request 길이에서 비교하여 padding으로 줄인 sector와 늘어난 resident KV bytes, eviction을 함께 셌다.

복구는 sectors/request가 줄었다는 이유만으로 닫지 않았다. boundary head와 마지막 page에서 출력이 일치하고, KV capacity 감소가 admission을 해치지 않으며, ITL p99가 개선돼야 했다. padding이 kernel은 빠르게 했지만 eviction을 늘려 end-to-end를 악화시키면 일반 path를 유지한다. 주소 최적화의 local win이 serving win이라는 보장은 없다.

### 42.7.2 사건 2: page size를 키우자 DRAM byte가 늘었다

42.1절 사건을 진단 절차로 다시 보자. page 16에서 64로 바꾸자 block-table load 수는 줄었지만 DRAM read byte와 GPU memory pressure가 늘었다. 첫 가설은 큰 page의 row stride가 L2 sector와 충돌한다는 것이었다. 하지만 `page_stride mod 32`는 두 설정 모두 0이었고 head-local 주소 집합도 같았다.

요청 길이 histogram을 page occupancy로 바꾸자 짧은 대화가 대부분이었다. page 64에서는 마지막 block의 unused token slot이 크게 늘었다. resident request 수가 줄어 scheduler가 prefix block을 더 자주 evict했고, 다음 turn에서 같은 KV를 다시 읽거나 재계산했다. 한 attention instruction의 transaction은 나빠지지 않았지만 workload 전체에서 cache reuse가 줄어 DRAM byte가 늘었다.

두 번째 요인은 physical block scatter였다. 큰 page가 연속 logical token을 더 많이 담지만 allocator pressure가 높아지면서 다른 요청의 page가 교대로 배치됐다. kernel tile이 page boundary를 넘는 횟수는 줄었지만 working set이 L2에 머무는 방식과 request scheduling이 달라졌다. source 주소식만으로 L2 replacement를 단정하지 않고, 동일 logical request의 page occupancy와 L2 hit 변화로 가설을 좁혔다.

“row alignment 충돌” 가설은 row별 covered interval이 동일하고 cold-cache micro shape가 같은 transaction 예상치를 보인 것으로 기각됐다. 수정은 page를 무조건 원상복구하는 것이 아니라 length bucket별 capacity와 boundary 비용을 비교하는 것이었다. page 16과 32를 후보로 두고 internal fragmentation, block-table byte, L2/DRAM byte, eviction, goodput을 함께 측정하는 설계를 남겼다.

종료 조건은 새 page size에서 단위 request의 주소식뿐 아니라 production length 분포의 allocated/useful KV 비율과 eviction이 안정되는 것이다. 특정 긴 prompt benchmark만 좋아지는 설정을 전체 트래픽에 적용하지 않는다. page는 hardware line과 맞추는 숫자가 아니라 allocator와 kernel 사이의 trade-off다.

### 42.7.3 사건 3: `int4` load를 넣었는데 속도가 그대로였다

한 custom KV transform kernel이 scalar 4-byte load를 `int4` 16-byte load로 바꿨다. source instruction 수가 줄고 정렬 assert도 통과했지만 kernel duration은 거의 변하지 않았다. 구현자는 compiler가 vector instruction을 생성하지 않았거나 GPU가 vector load를 지원하지 않는다고 의심했다.

SASS를 보기 전 주소 worksheet가 더 빠른 답을 주었다. lane마다 16 byte를 읽었지만 lane `i`의 base는 서로 다른 physical page row였다. 각 vector 내부 네 word는 연속이지만 warp 전체는 32개의 먼 구간을 가리켰다. scalar일 때도 같은 scatter였고 vector로 바꿔도 필요한 sector 수가 크게 줄지 않았다. instruction issue는 줄었지만 long memory latency와 transaction waste가 지배했다.

두 번째 가능성은 kernel이 memory-bound가 아니라는 것이었다. vectorization으로 global load instruction은 줄었지만 address calculation, dequantization 또는 reduction이 critical path일 수 있다. register가 늘어 occupancy가 내려가면 instruction 감소를 상쇄한다. 따라서 “속도 불변=compiler가 scalarized”라는 가설은 generated instruction과 warp address 두 증거로 나눠 검증해야 한다.

수정은 vector 폭을 더 키우는 것이 아니었다. warp가 같은 page/head의 연속 dimension을 맡도록 lane mapping 또는 input grouping을 바꾸었다. 이 변경은 output order를 유지하기 위한 permutation cost를 만들 수 있으므로 end-to-end ledger에 넣었다. layout 변경이 불가능하면 vectorization의 이득이 작은 사실을 받아들이고 단순한 scalar path를 유지할 수도 있다.

종료 조건은 source에 `int4`가 남아 있는지가 아니다. lane address가 예상한 구간 수로 줄고, compiler artifact에서 의도한 width가 확인되며, register/occupancy와 output correctness를 포함해 kernel 및 serving latency가 개선되는 것이다. vector type은 목표가 아니라 주소 집합을 만드는 한 수단이다.

### 42.7.4 사건 4: 다섯 구간을 예상했는데 profiler가 네 구간처럼 보였다

misaligned copy fixture에서 base를 4 byte 밀었다. 손계산은 다섯 32-byte 구간이었지만 DRAM byte와 achieved bandwidth는 예상한 20% 악화보다 작았다. 팀은 CUDA 13에서 coalescing 규칙이 바뀌어 misalignment penalty가 사라졌다고 결론내렸다.

먼저 CUDA 12.9.1과 13.3.0 Best Practices를 대조했다. 절 번호는 9.2.1에서 10.2.1로 이동했지만 cc 6.0+의 32-byte transaction 요약과 aligned/misaligned 예시는 두 판에 남아 있었다. 문서 구조 변경은 hardware 의미 변경의 증거가 아니었다. [CUDA C++ Best Practices Guide 13.3.0 — §10.2.1 Coalesced Access](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

공식 예시 자체가 인접 warp의 overfetched segment를 cache에서 재사용할 수 있다고 설명한다. warp 0이 끝에서 가져온 다섯 번째 구간을 warp 1이 시작에서 쓸 수 있다. source-level request는 warp마다 다섯 구간이어도 DRAM은 모든 구간을 매번 다시 가져오지 않을 수 있다. profiler metric이 L1 request, L2 sector, DRAM byte 중 무엇을 세는지도 확인해야 한다.

“CUDA 13 규칙 변경” 가설은 같은 source와 target GPU에서 toolkit만 바꾼 artifact의 address mapping이 같고, 두 문서의 계약도 같은 것으로 기각됐다. cache를 교란하거나 warp 간 reuse를 없애는 stride fixture에서는 penalty가 더 명확해졌다. 그러나 이것을 production 최적화 수치로 쓰지 않는다. 실험은 cache reuse 가설을 반증하기 위한 통제일 뿐 실제 attention pattern과 다르다.

복구 종료 조건은 hand calculation과 metric을 억지로 같은 숫자로 만드는 것이 아니다. requested interval, L1/L2 hit, DRAM byte의 관계를 설명하고 tool·architecture별 metric scope를 명시해야 한다. 예상과 관측 차이가 cache reuse로 닫히면 coalescing 규칙을 새로 발명할 필요가 없다.

### 42.7.5 사건 5: tail shape에서만 오답과 crash가 났다

정규 sequence length와 head dimension에서는 안정적이지만 마지막 partial page가 1~7 token일 때 드물게 illegal memory access 또는 logits divergence가 났다. 성능 최적화 과정에서 scalar tail을 vector load와 clamp로 합친 뒤 시작된 문제였다. 첫 가설은 misaligned vector가 단지 느려진다는 것이었다.

CUDA Programming Guide의 size/alignment 계약은 1, 2, 4, 8, 16-byte word가 single global instruction이 되려면 자연 정렬을 만족해야 한다고 설명한다. 특히 잘못 정렬된 넓은 word를 correctness 관점에서 가볍게 보아서는 안 된다. [CUDA C++ Programming Guide 12.9.1 — Device Memory Accesses, Size and Alignment Requirement](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#device-memory-accesses)

source에서 두 결함이 발견됐다. partial page clamp는 block-table row가 범위를 넘지 않게 했지만 column vector의 마지막 element mask가 vector 폭 전체를 보호하지 않았다. 유효 element가 두 개인데 16-byte load를 실행해 다음 row를 읽었다. 또한 subview의 시작 offset이 8 byte라 `int4` natural alignment를 깨뜨렸다. full row에서는 allocator base와 row stride가 맞아 문제가 숨었다.

“misalignment는 성능만의 문제”라는 가설은 offset별 correctness fixture와 source contract로 기각됐다. 수정은 vector fast path에 `pointer mod 16 == 0`, remaining elements와 row boundary predicate를 넣고, 아니면 좁은 vector/scalar tail로 보냈다. FlashAttention partial block의 row clamp와 element mask를 각각 독립적으로 검사했다.

복구는 가능한 모든 tail length와 base offset class에서 OOB sanitizer 또는 정적 boundary 계산, output reference 일치를 통과해야 한다. fast path 비율과 sector도 확인하되 correctness predicate를 완화해 숫자를 맞추지 않는다. 평균 shape benchmark는 tail bug의 증거가 되지 않는다.

### 42.7.6 사건 6: Marlin padding으로 load는 좋아졌지만 서빙은 나빠졌다

quantized model의 K/N을 Marlin tile에 맞춰 padding하고 offline repack했다. global→shared transaction과 GEMM kernel duration은 개선됐다. 그런데 cold start와 GPU memory peak가 늘고, 작은 batch의 TTFT는 악화됐다. 팀은 kernel이 빨라졌으므로 serving도 결국 좋아질 것이라고 보았다.

padding ledger를 만들자 숨은 byte가 보였다. logical quantized weight, tile multiple까지 추가한 padded weight, scale·zero-point, repack workspace, shared stage가 각각 달랐다. model loader가 artifact를 runtime format으로 repack하면 첫 요청 전 CPU/GPU copy와 kernel이 추가된다. 여러 layer의 padding은 resident model bytes를 늘려 KV cache budget을 줄일 수 있다. KV block이 줄면 preemption이나 offload가 늘어 GEMM 절약을 삼킨다.

kernel 내부에서도 16-byte `cp.async`가 transaction 효율을 높였지만 last tile의 useful packed value가 적었다. predicate가 OOB를 막아도 reserved shared stage와 instruction은 남았다. 큰 batch에서는 compute tile reuse가 이 비용을 상쇄했지만 작은 M에서는 weight traffic과 launch가 지배했다.

“kernel win이면 serving win” 가설은 model-load→repack→resident memory→KV admission→forward 전체 timeline으로 기각됐다. 대안은 serialized packed artifact, shape별 padding 정책, 다른 quant backend 또는 일반 tail path였다. packed artifact는 공급망·version compatibility 비용이 있고, backend 변경은 numeric validation이 필요하다.

종료 조건은 Marlin GEMM duration 하나가 아니다. cold/warm TTFT, ITL, model resident·workspace·KV bytes, batch 분포별 kernel 선택, 출력 오차와 goodput을 함께 비교한다. sector 감소가 capacity 감소를 통해 더 큰 serving 비용을 만들면 최적화를 보류한다.

## 42.8 CUDA 판과 GPU 세대를 안전하게 분리한다

주소 설명은 hardware와 가깝기 때문에 새로운 toolkit이나 GPU 이름을 원인으로 붙이기 쉽다. 그러나 CUDA toolkit 판, compiler artifact, target compute capability, 실제 GPU microarchitecture는 서로 다른 축이다. CUDA 12.9.1 source를 Hopper용으로 빌드할 수도 있고, CUDA 13.3 source를 이전 지원 GPU용으로 빌드할 수도 있다. 같은 source라도 bundled library, compiler flag와 target architecture가 바뀌면 instruction과 register allocation이 달라질 수 있다.

### 42.8.1 문서 절 번호가 바뀐 것은 transaction 변화가 아니다

12.9.1 Best Practices에서 coalescing은 §9.2.1이고 13.3.0에서는 §10.2.1이다. 두 고정 판의 cc 6.0+ 요약은 warp의 concurrent address를 처리하는 데 필요한 32-byte transaction 수로 설명한다. aligned 4-byte word 예제와 misaligned 예제도 이어진다. 따라서 URL·절 구조가 바뀐 사실만으로 CUDA 13의 coalescing 의미가 바뀌었다고 쓰지 않는다.

13.3 Programming Guide는 memory 설명을 재구성된 주제별 경로에서 제공한다. 12.9.1의 `Device Memory Accesses` 절 문장을 같은 anchor에서 찾지 못할 수 있다. migration audit에서는 먼저 동일 semantic topic을 매핑하고, 문구가 빠졌는지 위치만 바뀌었는지 확인한다. 실제 의미 변화 주장은 release note, programming guide의 명시적 변경과 target artifact를 추가로 요구한다.

### 42.8.2 일반 계약과 세대별 cache behavior를 두 상자로 둔다

일반 상자에는 lane address, requested byte, natural alignment, instruction width와 transaction 구간을 둔다. 세대 상자에는 해당 GPU tuning guide가 명시하는 L1/shared configuration, L2 기능, async copy와 profiler metric availability를 둔다. V100 misalignment benchmark의 790GB/s 같은 수치는 Volta 사례이지 Hopper·Blackwell의 기대값이 아니다.

source에서 cache operator나 prefetch가 보이면 해당 instruction의 semantic을 PTX 판에서 읽는다. 이것만으로 L2 hit rate와 DRAM traffic을 예측하지 않는다. working set, competing kernel과 access order가 필요하다. architecture 이름을 붙인 “sector promotion” 같은 설명도 공식 문서나 counter definition이 없으면 관측 가설로 낮춘다.

### 42.8.3 compiler artifact는 source와 profiler 사이의 다리다

C++ `int4` 또는 Triton vector expression이 어느 PTX/SASS instruction으로 내려갔는지는 build artifact로 확인해야 한다. compiler가 alignment를 증명하지 못하면 load를 쪼갤 수 있고, 여러 scalar를 합칠 수도 있다. predicate와 tail은 branch, predicated instruction 또는 별도 kernel specialization으로 나타날 수 있다.

그렇다고 SASS load 한 줄을 HBM transaction 하나로 번역하지 않는다. instruction이 warp의 lane마다 실행되고 주소 집합이 coalescer에 들어가며 cache를 거친다. 수직 경로는 `source pointer equation → compiler load width/predicate → warp lane addresses → L1/L2 request → DRAM observation`이다. 중간 artifact가 없으면 source에서 기대한 width라고 표시하고 관측 사실로 쓰지 않는다.

toolkit A/B에서 성능이 달라졌다면 source commit, compile flags, architecture list, bundled Triton/CUTLASS/FlashAttention, kernel symbol과 binary digest를 먼저 고정한다. 여러 축이 바뀌었으면 “환경 bundle 차이”라고 부른다. compiler version 하나로 원인을 좁히려면 다른 축을 같은 artifact matrix가 필요하다. 이 원칙은 44장의 CUDA 12.x/13.x migration에서 더 깊게 다루지만, coalescing 결론에도 즉시 적용된다.

## 42.9 한 요청의 주소를 증상에서 source까지 추적하는 실습

독자 A가 다음 증상을 받았다고 하자. Qwen 계열 모델의 decode ITL p99가 context 8K를 넘으면 늘고, attention kernel의 sectors/request가 함께 오른다. page size는 16, KV dtype은 fp16, KV head 8, head dimension 128이다. 평균 GPU utilization만으로는 memory access 문제인지 scheduler gap인지 알 수 없다. 여기서는 실행 명령을 만들지 않고 어떤 증거를 수집하고 어떻게 판정할지 source 기반 worksheet를 완성한다.

### 42.9.1 첫 분기: kernel 안인가, kernel 밖인가

ITL timeline에서 scheduler queue, host launch gap과 attention kernel duration을 나눈다. context가 길어질 때 kernel duration이 sectors와 함께 늘고 launch gap은 안정적이라면 memory-address 가설을 계속한다. kernel 사이 GPU idle이 커졌다면 coalescing을 파기 전에 scheduler·graph fallback을 본다. 같은 ITL 증상이라도 조사 owner가 다르다.

kernel symbol과 backend를 request shape와 연결한다. context 8K 아래와 위에서 같은 specialization인지, split-KV 수나 partial-block path가 바뀌는지 확인한다. kernel이 바뀌었다면 counter 차이는 alignment만이 아니라 tile·split·reduction 전체 차이다. A/B worksheet의 첫 행에 symbol, template parameters, page size, layout과 split 수를 적는다.

### 42.9.2 logical shape에서 byte stride를 만든다

fixture NHD의 contiguous stride는 head 128 element, token 1024 element, page 16384 element다. fp16이므로 byte stride는 각각 256, 2048, 32768 byte다. 모두 32-byte 배수다. allocator base가 256-byte aligned이고 cache subview offset도 32-byte 배수라면 모든 head row 시작은 32-byte 정렬을 보존한다.

그러나 실제 tensor `.stride()`가 fixture와 같은지 source/metadata에서 확인한다. quantized scale tensor, padding, hybrid cache group이 stride를 바꿀 수 있다. block table의 physical page id는 page stride에 곱해지므로 page stride가 정렬 배수라면 page id가 흩어져도 각 row base alignment는 보존한다. scatter는 locality와 page 간 lane mapping을 바꾸지만 natural alignment와 같은 문제가 아니다.

### 42.9.3 lane mapping과 full/tail을 적는다

선택 kernel에서 warp가 한 head의 `d`를 나눠 맡는지, 여러 token row를 맡는지 찾는다. lane마다 8-byte vector라면 full head row 256 byte를 정확히 덮는다. covered interval은 여덟 32-byte 구간이다. context length가 늘었다고 head row transaction 자체가 바뀌지 않는다.

이 결과는 중요한 반증이다. 8K boundary에서 head dimension·row alignment가 그대로라면 sectors 증가 원인을 “KV row misalignment”로 닫을 수 없다. 다음으로 split-KV tile이 더 많은 physical page row를 동시에 읽는지, last split의 partial block이 늘어나는지 본다. FlashAttention paged utility의 virtual page index, page offset과 partial clamp에 context/split 값을 대입한다.

8K 위에서 split 수가 1에서 4로 바뀌고 각 split 마지막에 작은 tail이 생겼다고 하자. total useful KV byte는 같아도 네 개 partial tile의 inactive lane과 reduction intermediate traffic이 추가된다. sectors/request metric의 request가 어떤 instruction/request 단위인지 확인한 뒤 total sectors와 useful output을 비교한다. 단순히 context가 길어서 cache가 나빠졌다는 설명보다 구체적이다.

### 42.9.4 세 가지 대립 가설을 세운다

가설 A는 KV row alignment가 깨졌다는 것이다. row base mod 32, actual stride와 selected vector predicate로 반증한다. 가설 B는 page scatter가 L2 locality를 떨어뜨렸다는 것이다. 같은 physical page permutation과 working set에서 L2 hit·DRAM byte 변화가 필요하다. 가설 C는 split/tail path가 extra transaction과 reduction을 만들었다는 것이다. symbol/split, partial lane mask, intermediate byte와 duration이 boundary에서 함께 변해야 한다.

세 가설은 배타적이지 않지만 강한 증거가 다르다. A는 address arithmetic, B는 cache observation과 access order, C는 dispatch와 tail instruction이다. profiler에서 sectors 하나만 높다고 셋 모두를 참으로 두지 않는다. source worksheet로 가능한 가설을 먼저 제거해 관측 범위를 줄인다.

### 42.9.5 수정 후보의 side effect를 먼저 쓴다

split threshold를 늦추면 tail과 reduction은 줄지만 한 CTA가 긴 context를 맡아 occupancy·latency가 나빠질 수 있다. page size를 키우면 boundary는 줄지만 internal fragmentation과 resident capacity가 악화될 수 있다. KV layout을 바꾸면 cache write는 좋아져도 attention backend가 transpose/copy를 요구할 수 있다. padding은 vector path를 회복하지만 KV byte를 늘린다.

각 후보 옆에 `예상 주소 변화`, `추가/절약 bytes`, `resource side effect`, `correctness boundary`를 적는다. 주소 변화가 없는 후보를 coalescing 수정이라고 부르지 않는다. 예를 들어 L2 persisting policy는 hit를 바꿀 수 있지만 warp requested transaction을 바꾸지 않는다. 효과 층을 구분하면 실패한 실험에서도 무엇이 반증됐는지 남는다.

### 42.9.6 복구 종료 조건을 workload로 돌려놓는다

수정 후 full row와 모든 partial length에서 output이 일치해야 한다. expected covered interval과 선택 specialization이 source 식과 맞아야 한다. sectors/request뿐 아니라 total sectors, L2 hit, DRAM bytes, long-scoreboard와 kernel duration이 예상 방향으로 움직여야 한다. 마지막으로 KV allocated/useful bytes, admission, TTFT와 ITL p50/p99를 원래 context distribution에서 비교한다.

context 8K 한 점에서만 좋아지고 4K 짧은 요청이 느려지면 length-aware dispatch를 고려한다. sector는 줄었지만 page padding으로 OOM이 늘면 복구가 아니다. kernel duration은 줄었지만 scheduler가 다른 backend fallback을 택해 graph replay가 깨져도 end-to-end 성공이 아니다. 종료 조건은 주소식에서 시작하지만 serving goodput에서 끝난다.

### 42.9.7 first divergence를 source 분기까지 되감는다

성능 회귀를 더 집요하게 좁히는 방법은 두 실행이 처음 달라지는 지점을 찾는 것이다. 여기서 실행은 GPU runtime을 새로 돌리라는 뜻이 아니라, 이미 수집된 정상·회귀 trace와 고정 source를 같은 사건 열에 놓는다는 뜻이다. 요청 길이 8191과 8192, head dimension 128과 120, page tail 15와 16처럼 predicate 경계 양쪽의 기록을 비교한다.

예를 들어 context 8191에서는 backend symbol `F_full`, split 수 1, partial block 15였고 8192에서는 `F_split`, split 수 4, partial block 0이라고 하자. sector counter가 갈라진 첫 지점은 attention kernel이지만 원인 후보는 kernel 내부 주소만이 아니다. dispatcher가 split path를 선택한 predicate가 첫 software divergence다. `F_split`은 intermediate output과 reduction을 추가하고 각 split의 KV range를 다시 계산한다. 두 symbol의 sectors/request를 직접 비교하면 instruction mix와 metric 분모가 달라질 수 있다.

first divergence table에는 여섯 열을 둔다.

```text
request shape | dispatch predicate | kernel symbol/template
logical range | physical page rows | lane/vector/tail mapping
```

첫 행은 요청과 scheduler가 제공한 shape다. 둘째는 source에서 다른 branch를 택한 정확한 조건이다. 셋째는 실제 artifact symbol 또는 기대 specialization이다. 넷째는 각 CTA·warp가 맡은 token/head/d 범위다. 다섯째는 block table을 대입한 physical row다. 여섯째는 covered interval과 inactive lane이다. 이 표에서 처음 달라진 열이 다음 source walk의 시작점이다.

vLLM cache-write에서 head dimension 변경이 첫 divergence라면 `head_stride == head_size`, `kv_scale_stride == 0`, vector helper의 alignment 조건을 순서대로 본다. 첫 predicate가 유지되고 vector helper tail만 달라지면 backend 전체를 의심할 필요가 없다. 첫 predicate가 깨져 NHD에서 per-head warp path로 바뀌면 instruction과 lane ownership이 함께 달라지므로 단순 tail 비교로는 부족하다.

SGLang MLA에서 sequence 길이만 바뀌었다면 `kv_loc` array의 길이와 split 경계를 대입한다. dimension pointer는 그대로인데 program별 `kv_loc` grouping이 달라졌다면 row scatter와 tail이 첫 divergence다. HiCache transfer가 같은 시점에 나타나도 attention row address와 CPU tier 이동을 혼동하지 않는다. transfer bytes 증가가 eviction 결과인지, attention sectors 증가가 gather 결과인지 timeline owner를 나눈다.

FlashAttention에서는 block table의 physical page와 `partial_block_size`를 utility 식에 넣는다. full path는 `block_row_offset`을 그대로 쓰지만 partial path는 마지막 thread row offset으로 clamp한다. 두 실행의 first divergence가 clamp predicate라면 page allocation base 정렬을 다시 조정하는 실험은 직접적인 반증이 아니다. lane들이 어떤 row를 중복 또는 mask해 읽는지 계산해야 한다.

FlashInfer MLA concat에서 dtype 또는 head shape가 바뀌었다면 vector trait와 compile-time static assertion을 확인한다. 지원된 specialization 자체가 달라지면 `NopeVec/RopeVec` byte 폭과 head chunk가 함께 달라진다. 동일 specialization인데 token stride만 바뀌면 runtime row alignment를 본다. compile-time coverage와 runtime base를 한 증거로 합치지 않는다.

Marlin에서는 quant artifact의 K/N, group size와 packed layout이 dispatcher·repack path를 결정한다. first divergence가 repack padding이라면 runtime GEMM load만 비교하지 않고 artifact byte와 startup timeline을 포함한다. first divergence가 `cp.async` predication인 last tile이라면 full tile counter의 평균이 tail 문제를 숨길 수 있다. tail tile을 요청 전체에서 몇 번 실행하는지 weighted count를 만든다.

이 되감기 방식의 장점은 “memory-bound 같다”는 넓은 가설을 작은 source predicate로 바꾸는 것이다. 하지만 first divergence가 곧 root cause라는 보장은 없다. dispatch 변화가 workload 길이 증가의 정상 결과이고 실제 병목은 L2 capacity일 수 있다. 그래서 각 divergence마다 대립 가설을 둔다. branch를 고정했을 때 주소·resource side effect가 무엇인지 예측하고, 이미 가진 shape/counter로 일치 여부를 확인한다.

correctness incident에서는 first divergence가 더 중요하다. 정상 출력과 오답 출력의 logits가 처음 달라진 layer·token을 찾고, 그 consumer가 읽은 KV physical page와 generation을 연결한다. boundary tail에서만 갈라지면 vector OOB와 mask를, eviction 뒤만 갈라지면 generation과 ownership을 먼저 본다. sector 증가와 logits divergence가 함께 나타났다는 이유로 coalescing 자체가 오답을 만든다고 쓰지 않는다. misaligned wide access, OOB predicate와 stale mapping처럼 정확한 correctness 위반을 source에서 찾아야 한다.

복구 뒤에는 divergence table의 경계 양쪽을 다시 감사한다. predicate를 없애 모든 shape를 fast path로 강제한 수정은 tail correctness를 잃을 수 있다. padding으로 양쪽이 같은 specialization을 택하게 했다면 capacity ledger를 갱신한다. 새로운 scalar fallback을 넣었다면 정상 full shape가 계속 vector path를 타는지 확인한다. 수정은 first divergence를 지우는 것이 아니라 각 branch가 자신의 계약을 지키게 하는 일이다.

최종 보고 문장은 이렇게 구체적이어야 한다. “context 8192에서 split predicate가 1→4로 바뀌며 네 partial range가 생겼고, paged row clamp 뒤 각 tail의 participating lane이 줄어 total covered interval이 증가했다. row base 정렬은 두 실행 모두 유지됐다. threshold 변경은 sectors를 줄였지만 single-split duration을 늘려 length 12K 이상에서는 역전됐으므로 length bucket별 dispatch를 유지한다.” 이 정도 좌표가 있어야 다음 버전에서 같은 문제를 다시 찾을 수 있다.

재발 방지에는 세 종류의 fixture가 필요하다. 첫째는 alignment class fixture다. allocation base를 무작위로 깨뜨리는 것이 아니라 API가 허용하는 subview offset과 row stride 조합을 열거한다. `base mod 16`, `row_stride mod 16`, source-destination alignment 차이를 기록하고 각 조합이 fast, prologue/vector/tail, scalar 중 어느 경로를 택해야 하는지 source predicate로 기대값을 만든다. 허용되지 않는 misaligned wide access는 성능 대조가 아니라 validation 실패로 판정한다.

둘째는 shape boundary fixture다. head dimension, sequence length, page remainder, quant K/N을 specialization threshold 바로 아래·같음·바로 위에 둔다. 예를 들어 dimension 120/128/136, page remainder 0/1/15, split threshold 8191/8192/8193을 짝으로 둔다. 목표는 모든 shape가 같은 빠른 kernel을 타게 하는 것이 아니다. 각 shape가 의도한 branch를 타고 tail mask·clamp가 correctness를 지키며, branch 전환의 byte·latency 비용이 설명 가능한지 확인하는 것이다.

셋째는 serving coupling fixture다. kernel만 고립시키지 않고 같은 KV memory budget에서 padding 전후 resident block, eviction, scheduler batch와 ITL을 비교한다. Marlin weight padding이 KV capacity를 줄이는 경우, page size가 fragmentation을 늘리는 경우, layout 변경이 transpose copy를 추가하는 경우가 여기에 잡힌다. kernel counter가 좋아져도 deadline goodput이 떨어지면 최적화는 실패다.

fixture 결과에는 source commit과 artifact symbol을 함께 묶는다. 새 release에서 파일 행이 이동하거나 helper 이름이 바뀌어도 dispatch predicate, address equation과 tail ownership을 다시 찾을 수 있어야 한다. binary symbol이 바뀌었는데 옛 counter 기준선을 그대로 비교하면 다른 instruction mix를 같은 kernel로 오인한다. 반대로 symbol 이름만 바뀌고 주소·template 계약이 같다면 의미 변경으로 과장하지 않는다.

마지막으로 counter budget을 원인 판정문과 분리한다. “sectors/request 10% 증가”는 관측이다. “head row stride가 16-byte vector alignment를 깨뜨려 scalar tail이 선택됐다”는 source·주소 기반 원인 가설이다. “stride padding 뒤 branch와 covered interval이 예상대로 바뀌고 tail correctness 및 ITL이 회복됐다”가 검증이다. 세 문장을 한 칸에 섞지 않으면 다음 독자는 관측값만 달라졌을 때 원인부터 다시 추측하지 않아도 된다.

이 기록에서 성능 수치는 반드시 workload 좌표를 가진다. model과 dtype, KV layout·page size, request 길이 분포, batch·split, GPU와 kernel symbol이 빠지면 sectors 변화의 분모를 재현할 수 없다. 특히 sectors/request에서 `request`가 source-level load 하나인지 profiler가 정의한 wavefront/request인지 확인한다. 서로 다른 tool 판의 비슷한 metric 이름을 그대로 이어 붙이지 않는다. metric 정의가 바뀌거나 지원되지 않으면 requested byte와 DRAM byte의 보조 증거로 돌아가고, 존재하지 않는 counter를 추정값으로 채우지 않는다.

source coordinate도 단순 참고문헌이 아니라 predicate를 다시 찾는 시작점이다. 링크 범위 안에서 입력 stride, branch, pointer mutation, mask와 launcher 인자를 차례로 읽는다. 줄 번호가 이동한 새 commit에서는 옛 설명을 복사하지 않고 동일 symbol의 caller와 consumer를 다시 연결한다. 이 규율이 있어야 “정렬 최적화”가 특정 commit의 우연한 layout을 영구 법칙으로 만드는 일을 막는다.

재발 방지 fixture는 빠른 shape 하나만 보호하는 장치가 아니다. full·tail·misaligned-but-supported 경계에서 서로 다른 branch가 모두 자신의 memory-safety와 byte-cost 계약을 지키는지 보존하는 장치다. 이 범위가 닫혀야 다음 compiler가 load를 합치거나 새로운 backend가 선택돼도 관측 변화와 correctness 퇴행을 구별할 수 있다. 경계 전체가 회귀 계약이다. 반복 검증한다.

## 42.10 판정: 정렬은 숫자가 아니라 주소 관계다

이 장의 첫 사건에서 page size를 키운 팀은 틀린 숫자를 고른 것이 아니었다. 서로 다른 단위를 같은 것으로 생각했다. CPU cache line, GPU transaction·sector, software KV block, HBM memory command를 모두 “cache line”이라 불렀기 때문에 page 값을 바꾸는 행위가 어느 층을 바꾸는지 알 수 없었다.

가장 오래가는 도구는 간단한 주소 worksheet다. base, row·page·head stride, lane mapping, element/vector width와 참여 mask를 적는다. 32개 lane의 byte interval을 합치고 덮는 구간을 센다. 이 작업은 profiler 없이도 불가능한 가설을 제거한다. allocator base가 맞아도 row stride가 어긋날 수 있고, vector가 정렬돼도 warp row가 scatter될 수 있으며, page가 커져도 한 row transaction은 같을 수 있다.

공식 CUDA 계약은 계산의 경계를 준다. cc 6.0+ 요약에서 참여 warp 주소를 처리하는 32-byte transaction 수를 세되, L1/L2와 DRAM을 같은 것으로 보지 않는다. 12.9.1에서 13.3으로 절 번호가 바뀌어도 의미 변경을 자동 추론하지 않는다. 특정 GPU의 cache behavior와 profiler metric은 architecture 문서와 artifact에 묶는다. HBM burst는 source가 말하지 않은 구체성을 채우지 않는다.

다섯 source walk는 같은 원칙의 다른 모습을 보여 줬다. vLLM은 NHD 연속 row와 HND head-local warp를 predicate로 나눈다. SGLang MLA는 연속 dimension 앞에 흩어진 `kv_loc` row base를 둔다. FlashInfer는 warp 32 lane과 vector byte를 compile-time에 맞추지만 runtime stride alignment가 남는다. FlashAttention은 block table, page·row stride와 partial clamp로 full/tail 주소를 바꾼다. Marlin은 packed tile을 16-byte async copy로 옮기되 padding과 stage 비용을 낸다.

따라서 `vectorized`, `aligned`, `paged`, `coalesced`라는 label은 결론이 아니다. 독자가 써야 할 문장은 어느 lane이 어느 physical row의 몇 byte를 읽고, 어떤 predicate에서 tail로 가며, 그 결과 몇 구간과 어떤 side effect가 생기는지다. 이 문장이 source와 counter를 잇는다.

성능 복구도 같은 순서를 따른다. 증상에서 kernel 안팎을 나누고, 주소식으로 대립 가설을 세우고, compiler·cache·DRAM 관측을 각 층에 붙인다. 수정은 sector 감소뿐 아니라 padding, capacity, occupancy, backend fallback과 correctness를 함께 평가한다. 최종 판정은 원래 request 길이와 동시성에서 TTFT·ITL·goodput이 나아졌는지다.

다음 장에서는 주소만큼 중요한 시간을 다룬다. stream과 event, barrier와 CUDA Graph가 어느 작업이 먼저 끝나야 다른 작업이 memory를 안전하게 읽을 수 있는지 정한다. 42장이 “어디를 읽는가”를 닫았다면 43장은 “언제 읽어도 되는가”를 닫는다. 두 질문을 합쳐야 빠르지만 틀린 kernel과 정확하지만 직렬화된 kernel 사이에서 올바른 dependency를 설계할 수 있다.

## 42.11 warp 32개 주소를 transaction과 sector로 끝까지 센다

여기서는 CUDA Best Practices Guide의 compute capability 6.0 이상 교육 모델, 즉 참여 thread 주소를 처리하는
데 필요한 32-byte 구간 수를 사용한다. 이 계산은 L1/L2 hit와 DRAM command를 예언하지 않는다. 한 warp
memory instruction이 만든 requested address 집합의 하한과 낭비를 구한다.

fixture A는 32 lane이 float 하나씩 읽는다. 기준 주소 `B`는 256-byte aligned이고 lane `i`의 주소는
`B+4i`다. lane 0~7은 sector 0, 8~15는 1, 16~23은 2, 24~31은 3이다. useful 128 byte, 네 sector
128 byte, 이용률 100%다.

fixture B는 subview가 float 하나 밀려 `B+4+4i`다. lane 0~6은 sector 0, 7~14는 1, 15~22는 2,
23~30은 3, lane 31은 sector 4를 건드린다. useful 128 byte지만 다섯 sector 160 byte를 덮어 80%다.
allocation base aligned와 warp row misaligned가 동시에 참이다. allocator alignment를 키워도 offset 4가
남으면 결과는 같다.

fixture C는 base가 `B+28`에서 시작한다. lane 0은 sector 0 끝, lane 1부터 sector 1로 넘어가며 마지막 lane은
sector 4에 있다. 역시 다섯 sector다. 다만 참여 lane이나 vector 폭이 달라지면 암기하지 말고 다시 센다.

fixture D는 stride two다. `addr(i)=B+8i`, 각 lane은 float 4 byte를 읽는다. lane 0~3이 sector 0의 offset
0,8,16,24를 쓰고 lane 4~7은 sector 1로 간다. lane 28~31은 sector 7이다. useful 128 byte지만 여덟 sector
256 byte를 덮어 50%다. 모든 pointer는 natural alignment를 만족한다. 문제는 warp address density다.

fixture E는 lane마다 16-byte vector를 연속으로 읽는다. `addr(i)=B+16i`이면 warp useful 512 byte, sector
0~15 총 16개다. scalar fixture보다 sector 수가 큰 것은 useful byte가 네 배인 결과다. sectors/request를
vector width가 다른 kernel끼리 단독 비교하지 않는다.

fixture F는 `addr(i)=B+64i`다. lane vector는 정렬됐지만 사이에 48-byte gap이 있다. lane 0은 sector 0,
lane 1은 sector 2, lane 31은 sector 62를 건드리며 서로 다른 32 sector가 필요하다. useful 512 byte,
covered sector byte 1,024로 50%다. 빈 gap sector는 건드리지 않으므로 span/32도 정답이 아니다.

fixture G는 lane 0~9만 float를 읽는 tail이다. aligned base에서 40 useful byte가 두 sector 64 byte를 덮어
62.5%다. base가 28 byte 밀리면 세 sector 96 byte, 약 41.7%다. tail과 misalignment가 결합하면 상대
낭비가 커진다.

fixture H는 broadcast다. 32 lane 모두 같은 4-byte 주소를 읽으면 unique address interval은 sector 하나다.
useful을 lane별 128 byte로 셀지 unique payload 4 byte로 셀지는 질문에 따라 다르다. metadata broadcast와
KV dimension read를 같은 효율 식으로 비교하지 않는다.

fp16 KV row에 대입한다. dimension 128, lane당 fp16 네 개 즉 8 byte를 읽으면 warp useful 256 byte다.
`R mod32=0`이면 sector 여덟 개, `R mod32=8`이면 아홉 개로 256/288≈88.9%다. lane vector는 8-byte
natural alignment를 만족해도 warp sector는 하나 늘 수 있다.

dimension 120이면 row 240 byte고 lane 0~29만 참여한다. aligned row는 여덟 sector 256 byte로 93.75%다.
`R mod32=8`도 offset 8부터 247까지라 여덟 sector다. 이 tail에서는 8-byte shift가 sector 수를 늘리지
않는다. “misaligned면 항상 하나 추가” 규칙이 틀리는 예다.

dimension 124는 row 248 byte이고 `248 mod32=24`다. page base가 맞아도 head0 start mod32=0,
head1=24, head2=16, head3=8, head4=0으로 alignment class가 반복된다. 평균 counter는 head별 class의
가중 평균이다. head0 한 주소만으로 전체 alignment를 증명하지 않는다.

cache policy는 이 계산 뒤에 적용한다. 같은 다섯 sector라도 인접 warp가 마지막 sector를 이미 가져왔다면
DRAM byte 증가는 작을 수 있다. 네 requested sector라도 working set이 L2를 밀어내고 다른 loads/writeback이
섞이면 kernel DRAM byte는 커질 수 있다. 주소 예측과 profiler 관측을 합치지 않는다.

## 42.12 cache hierarchy와 latency hiding이 “왜”를 완성한다

coalescing이 중요한 첫 이유는 useful byte보다 더 많은 sector를 움직일 수 있기 때문이다. 그러나 sector가
하나 늘었다고 latency가 정확히 25% 늘지는 않는다. 요청은 L1/L2에 hit할 수 있고 memory subsystem은 여러
request를 병행하며, SM은 한 warp가 기다리는 동안 다른 eligible warp를 실행한다. byte amplification과
exposed stall을 연결해야 “왜 느려졌는가”가 완성된다.

warp W0가 네 sector를 요청하고 L1 miss, L2 hit라고 하자. W1은 같은 마지막 sector를 곧 읽는다. W1 주소식은
다섯 sector를 덮어도 일부가 cache에서 제공될 수 있다. 반대로 working set이 L2보다 크고 page permutation이
reuse distance를 늘리면 같은 four-sector row도 DRAM으로 반복 내려간다. coalescing은 spatial density를,
cache는 reuse와 residency를 다룬다.

latency hiding의 핵심은 resident warp 수보다 eligible warp와 independent work다. memory dependency가 풀릴
때까지 warp는 long scoreboard에 걸릴 수 있다. 다른 warp가 준비됐다면 교대하지만 register/shared memory로
occupancy가 낮거나 모든 warp가 같은 memory phase에 있거나 pointer chasing이면 숨길 work가 없다.

kernel A는 warp당 네 sector지만 resident warp 16개와 independent math가 있다. B는 vectorization으로 세
sector지만 register가 늘어 resident warp가 8개로 줄었다고 하자. B의 DRAM byte가 줄어도 exposed stall과
duration이 늘 수 있다. transaction 감소는 serving latency의 충분조건이 아니다.

occupancy 100%도 성공을 뜻하지 않는다. resident warp 모두 block-table dependent gather를 동시에 수행하면
전부 scoreboard를 기다린다. split-KV를 늘리면 parallelism은 좋아질 수 있지만 split tail과 reduction byte가
늘어난다. 최적 split은 memory-level parallelism과 extra work의 균형이다.

cache policy 예측은 조건부다. L1 사용 경로에서 작은 반복 row가 hot하면 L1 hit와 DRAM byte가 좋아질 수
있다. streaming 또는 L1 bypass 성격의 경로에서는 L2/DRAM이 더 직접 보일 수 있다. 큰 one-pass KV scan이
L1을 오염시키지 않는 정책은 다른 data를 보호하지만 scan byte 자체를 줄이지 않을 수 있다.

alignment 개선이 locality를 악화시키기도 한다. 248-byte payload마다 8-byte padding을 넣으면 sector는
규칙적일 수 있지만 resident working set과 L2 footprint가 커진다. KV capacity가 줄어 eviction/offload가
증가하면 DRAM/PCIe traffic이 더 커질 수 있다. 주소와 capacity ledger를 함께 본다.

Nsight 예측은 방향으로 쓴다. instruction mix가 같은 aligned A→misaligned B에서 requested sectors/request는
4→5로 늘 것으로 예상한다. cache hit가 충분하면 DRAM byte 증가는 작을 수 있다. long scoreboard는 latency가
critical path에 노출될 때만 늘며 다른 warp가 가리면 유지될 수 있다. vector width를 키우면 instruction은
줄지만 request당 sector는 useful byte와 함께 늘 수 있다.

tail lane 32→10에서는 absolute sector가 4→2로 줄어 counter만 좋아 보일 수 있다. useful byte/sector는
128/128에서 40/64로 나빠진다. metric 방향을 efficiency와 동일시하지 않는다.

serving source에서는 vLLM의 `head_stride == head_size`가 row/helper를 바꾸고, FlashInfer trait가 lane 32개와
head byte를 맞춘다. FlashAttention paged utility는 block table/partial clamp로 row와 participation을 바꾼다.
Marlin async copy는 16-byte width와 stage/occupancy를 함께 바꾼다. 각 predicate마다
`addresses→sectors→cache→eligible warps/stall→duration→ITL`을 쓴다.

## 42.13 misalignment를 오진해 padding으로 capacity를 잃은 사건

장애는 Qwen 계열 fp16 KV 모델을 새 backend로 옮긴 직후 시작됐다. 이전 backend에서 decode attention의
global-load sector 관련 counter는 request당 8.2였고 새 backend는 9.1이었다. ITL p99도 31ms에서 39ms로
올랐다. 팀은 head row가 32-byte 경계를 어겼다고 결론내리고 모든 row를 256-byte stride로 padding했다.
counter는 8.6으로 조금 내려갔지만 ITL p99는 44ms로 더 나빠졌고 긴 요청에서 OOM이 늘었다.

첫 관측 문장은 “misalignment가 느리게 했다”가 아니다. “backend 전환과 함께 symbol, split 수, requested-sector
counter, ITL이 변했고 padding 후 sector counter는 일부 회복했지만 capacity와 ITL은 악화했다”다. 원인과
관측을 분리해야 padding 수정이 무엇을 바꿨는지 검증할 수 있다.

fixture의 logical shape는 KV head 8, head dimension 120, fp16이다. head row는 240 byte, token row는
1,920 byte다. allocator base는 256-byte aligned이고 original token stride `1920 mod32=0`이다. 따라서 각
token row start는 32-byte 정렬을 보존한다. head stride는 `240 mod32=16`이므로 같은 token 안에서 head 0은
mod0, head1 mod16, head2 mod0처럼 반복한다. warp가 head-local row를 읽는다면 head 절반이 다른 alignment
class지만, NHD fast path가 token 전체 1,920 byte를 연속 처리한다면 helper 분할은 다르다. 먼저 실제
`head_stride == head_size` predicate와 selected path를 확인해야 한다.

새 backend source trace에서 NHD fast path가 아니라 split paged attention specialization이 선택됐다. context
length 8K에서 split 수가 1→4로 바뀌었고, 각 split이 마지막 partial range를 만들었다. profiler symbol도
달랐다. sectors/request 분모인 memory instruction mix가 이전 kernel과 같지 않았다. head-row alignment는
존재하는 비용 후보였지만 backend 전환의 first divergence는 dispatcher의 split predicate였다.

warp 주소를 손으로 계산한다. head dimension 120에서 lane당 8 byte, 30 lane이 참여한다. head0 mod0은
240 byte가 여덟 sector를 덮는다. head1 mod16은 address offset16부터255까지로 역시 sector 0~7 여덟 개다.
즉 이 vector/tail 조합에서는 head1 shift가 sector를 추가하지 않는다. 팀은 dimension128 full row의
“misalignment면 8→9” 계산을 dimension120 tail에 그대로 적용했다. 실제 row sector 증가 예측은 없었다.

그렇다면 observed 9.1은 어디서 왔는가. 새 split kernel은 block table/partial clamp load와 reduction
intermediate를 추가했고, 각 split tail에서 별도 memory instructions가 실행됐다. sectors/request가 어떤
instruction group을 집계하는지 확인하니 KV row load만이 아니라 관련 global loads가 포함됐다. total useful
KV byte는 비슷해도 instruction 분모와 metadata/intermediate sector가 달라졌다. row base padding으로 이
traffic을 없앨 수 없다.

padding patch는 token row 1,920을 2,048 byte로 늘렸다. token당 128 byte, 약 6.67% 증가다. layer 32,
K/V가 별도이고 context 16,384, batch active 48의 단순 추가 byte는 `128×32×2×16384×48`, 약 12GiB다.
실제 layout에서 padding 적용 범위와 공유를 확인해야 하지만, 이 fixture에서는 resident KV budget을 눈에 띄게
줄였다. physical blocks가 더 빨리 찼고 eviction/offload와 block-table scatter가 늘어 ITL이 44ms가 됐다.

첫 경쟁 가설은 genuine row misalignment다. row base mod32와 lane vector interval이 8 sector로 유지돼
반증됐다. 둘째는 split/tail instruction 증가다. symbol/split boundary, partial lane과 metadata/reduction bytes가
counter 및 duration 변화와 정렬돼 지지됐다. 셋째는 L2 locality 저하다. padding 전후 L2 footprint와 hit
변화는 있었지만 backend 전환 직후 first divergence를 단독 설명하지 못했다. 넷째는 scheduler gap이다.
kernel duration 자체가 늘고 launch gap은 안정적이어서 주원인에서 제외됐다.

Nsight 기대를 수정한다. split을 1→4로 바꾸면 KV logical byte가 같아도 block-table loads, partial instructions,
reduction read/write와 total sectors가 늘 수 있다. sectors/request는 instruction grouping 변화로 직접 비교가
어려울 수 있으므로 useful KV byte, total sectors, intermediate byte, L2 hit, DRAM byte와 kernel duration을
함께 본다. long scoreboard가 늘었다면 extra memory latency가 노출됐는지 eligible warps와 occupancy를 본다.

수정 후보 A는 padding 유지, B는 padding rollback과 split threshold 조정, C는 length-aware specialization이다.
A는 row sectors를 바꾸지 않으면서 capacity만 잃으므로 탈락한다. B는 8K 근처 tail/reduction을 줄이지만 더
긴 context에서 한 CTA work가 커져 parallelism이 부족할 수 있다. C는 8K 이하 full/single split, 긴 context는
multi-split을 유지하며 boundary bucket별 trade-off를 표현한다. 실제 source가 지원하는 dispatch 범위 안에서
선택하고 존재하지 않는 dynamic policy를 가정하지 않는다.

canary는 context 7,999/8,000/8,001과 12K/16K, head dimension 120/128을 짝으로 둔다. 각 run에서 symbol,
split, block table rows, participating lanes, useful/sector/intermediate bytes, registers, occupancy, eligible warps,
long scoreboard와 duration을 기록한다. output parity와 tail OOB mask도 확인한다. one-point 8K 개선만으로
rollout하지 않는다.

rollback은 padding artifact를 되돌리는 것에서 시작한다. 이미 padded layout으로 만들어진 KV blocks와 original
layout blocks를 같은 pool generation에서 섞지 않는다. 새 admission을 original layout generation으로 보내고
padded active requests를 drain한다. external/offloaded payload가 layout schema를 포함하는지 확인하고 incompatible
entry는 invalidate/recompute한다. kernel graph가 captured pointer/stride를 갖고 있다면 original stride로 다시
capture하거나 eager safe path를 쓴다.

rollback 종료 조건은 sector counter 하나가 아니다. token row byte가 2,048→1,920으로 돌아오고 resident
blocks/admission이 복원된다. 120-dimension row는 손계산한 여덟 sector expectation과 actual instruction scope가
맞는다. split/tail total traffic과 duration이 length bucket별 guardrail 안에 들어온다. OOM, eviction/offload,
ITL p50/p99와 deadline goodput이 baseline을 회복하며 logits parity가 통과해야 한다.

사건의 최종 원인은 “32-byte alignment가 중요하지 않다”가 아니다. 정확한 원인은 `다른 vector/tail shape에
full-row 규칙을 오용 → 서로 다른 kernel의 sectors/request를 같은 분모로 비교 → padding으로 stride와 capacity를
변경 → KV footprint와 eviction 증가`다. alignment는 중요한 주소 관계지만 실제 lane interval을 세지 않은
alignment 수정은 새로운 병목을 만들 수 있다.
