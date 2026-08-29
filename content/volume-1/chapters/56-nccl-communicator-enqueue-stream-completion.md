# 56장. `ncclAllReduce`는 언제 끝나는가: communicator에서 stream completion까지

TP rank 네 개가 각자 만든 partial output을 합치는 코드는 한 줄이다. 그러나 host API가 반환한 순간, group end가 성공한 순간과 다른 CUDA stream의 consumer가 결과를 안전하게 읽는 순간은 같지 않다. 이 장은 BF16 16개 원소 all-reduce 하나를 NCCL v2.30.7-1 commit `73cf112295c33aee2b895f329f592f2a9b4b0f97`의 수명으로 끝까지 추적한다.

rank `r`의 입력은 모든 원소가 `r+1`이다. rank 0은 1, rank 1은 2, rank 2는 3, rank 3은 4다. sum 결과는 모든 rank에서 10이어야 한다. logical count는 16, input/output payload는 BF16이므로 rank당 32 bytes다. algorithm chunk, protocol transaction, link bytes와 HBM traffic은 이 32 bytes와 같은 숫자가 아니다.

## 56.1 대표 all-reduce 하나를 host 호출부터 consumer까지 계산한다

대표 fixture는 8개 rank가 각자 32-byte fp32 partial vector를 제출하고 합을 같은 output buffer에서 받는 all-reduce다. 각 rank에 `communicator generation, sequence, count, dtype, input/output pointer, enqueue 시각, device launch, proxy progress, completion event, consumer read`를 기록한다. 이 한 fixture로 communicator 수명, group, algorithm/protocol/channel, stream ordering과 rollback을 설명한다. 나머지 여섯 사건은 새 튜토리얼이 아니라 이 장부의 반례 행으로만 참조한다.

[`ncclCommInitRank와 init job`](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/init.cc#L1734-L1944)과 [`public init entry`](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/init.cc#L2472-L2574)를 읽으면 communicator가 단순한 rank list가 아님을 알 수 있다. unique ID와 membership, rank/device, bootstrap, topology, channels와 peer connection, config, task queues, async error와 finalize/abort state가 한 lifetime에 묶인다.

```text
comm_generation = G7
nranks = 4
rank_order = [0,1,2,3]
rank r -> process / CUDA device UUID / BDF
unique_id_hash = U
blocking mode / split parent / config
init submit / init return / init job completion
first collective sequence = 0
```

모든 process에서 `nranks=4`라는 사실만 같아서는 부족하다. unique ID, rank order와 device mapping이 같은 generation이어야 한다. rank 2가 이전 restart의 unique ID를 사용하면 pointer와 count가 맞아도 bootstrap 집합이 갈린다. rank가 중복되거나 한 process가 다른 GPU context를 current로 둔 사건도 steady collective 이전에 닫아야 한다.

**init API return과 ready를 구분한다**

blocking configuration과 async job state에 따라 init call의 return 의미가 달라질 수 있다. handle을 얻었다는 사실을 모든 peer transport가 warm하고 첫 collective latency가 steady-state라는 뜻으로 확대하지 않는다. first collective는 lazy connection, memory registration, channel setup, graph/kernel warm-up을 포함할 수 있다.

첫 collective만 느리다면 init start/return, async state, first enqueue, proxy connect, device launch를 같은 clock으로 잇는다. init return 전에 긴 구간이 있으면 bootstrap/topology 후보다. enqueue 이후 proxy/transport 준비가 길면 lazy connect 후보다. device kernel 첫 launch만 늦으면 module/context warm-up이나 stream dependency를 본다. “NCCL warm-up”이라는 한 문장으로 닫지 않는다.

**communicator identity는 rank마다 증명한다**

collective signature 원장에는 `comm_generation, sequence, function, count, dtype, op, root, send/recv alias class, stream identity`를 둔다. communicator 생성 성공이 이후 calls의 순서를 자동으로 맞추지 않는다. application scheduler가 rank마다 다른 branch를 타면 동일 communicator에서 다른 collective가 enqueue될 수 있다.

**communicator가 소유하는 resource**

channels와 peer connection, proxy/network state, device-side communication structures와 task queues는 communicator lifetime에 속한다. buffer payload는 application owner지만 NCCL work가 끝날 때까지 send/recv range를 재사용하거나 free할 수 없다. CUDA stream은 application/framework가 소유해도 communicator enqueue가 그 stream ordering에 참가한다.

communicator cache를 model instance 사이에 재사용하려면 membership, device, config와 process group generation이 같아야 한다. 이전 model unload의 in-flight work나 async error를 가진 communicator를 새 model에 넘기지 않는다. integer world size가 같다고 재사용 가능하지 않다.

**bootstrap failure cleanup**

rank 하나가 init 중 실패하면 성공한 ranks의 partial communicator와 jobs를 정리해야 한다. 일부 rank만 serving ready를 publish하면 첫 request가 hang한다. group-wide ready barrier는 모든 rank의 communicator state가 usable한지 확인한 뒤 model group을 공개한다.

retry는 새 generation을 사용한다. failed rank만 old communicator를 이어 쓰고 다른 rank가 새 unique ID로 시작하면 더 어려운 mismatch가 된다. init timeout log에는 최초 local error와 다른 ranks의 derivative wait를 구분한다.

### 사건 5: 첫 collective만 느리다

init return, first enqueue, registration/connect, device launch를 분리한다. subsequent same-size sequence와 비교한다. lazy connection이면 controlled warm-up이나 eager setup을 고려한다. module/context warm-up이면 communication topology tuning을 하지 않는다.

warm-up도 all ranks 같은 sequence로 실행하고 output/buffer lifetime을 닫는다. startup readiness가 warm-up completion까지 요구되는지 SLO 정책을 명시한다.

첫 20분에는 NCCL tag/commit, driver/CUDA, rank-device-BDF, communicator generation과 config를 모은다. 다음 20분에는 rank별 last completed/current sequence와 signature를 비교한다. 여기서 다르면 plan/kernel을 보지 않는다.

communicator readiness를 세 단계로 나누면 첫 collective 사건이 선명해진다. `HANDLE_RETURNED`는 public init entry가 handle을 application에 돌려준 상태다. `COMM_USABLE`은 async init 결과와 error를 확인해 collective enqueue를 받을 수 있는 상태다. `PATH_WARM`은 이번 topology의 connection, registration과 first launch 비용이 이미 지불된 상태다. API가 제공하지 않는 상태 이름을 내부 원장으로 쓰되 공식 보장처럼 말하지 않는다.

unique ID bootstrap에는 deployment generation을 붙인다. D8의 unique ID를 rank 2가 늦게 받았는데 coordinator가 D9를 시작했다면 rank 2는 old job을 폐기해야 한다. message에 communicator generation과 membership hash를 넣는다. timeout 후 old message가 도착해 새 init에 적용되지 않게 한다.

initialization memory peak에는 channels, peer resource와 registration cache를 넣는다. 여러 model communicator를 동시에 init하면 weight load와 connection pressure가 겹친다. 정상 destroy 후 새 communicator를 만들 때 old proxy thread, CUDA graph와 cached buffer가 남았는지 resource counter로 확인한다. pointer address가 재사용돼도 generation은 새롭다.

multiple communicators가 opposite order로 호출될 때 framework locking과 NCCL ordering contract를 확인한다. mutex로 hang이 사라져도 rank call order와 stream dependency를 검증하지 않고 영구 serialization로 끝내지 않는다.

init timeout과 collective timeout을 구분한다. bootstrap이 느린데 collective watchdog을 늘려도 해결되지 않는다. first collective lazy connect가 느릴 때 init timeout만 늘려도 해결되지 않는다. 각 timer가 감싼 interval을 적는다.

communicator readiness도 handle returned, usable, path warm으로 나눌 수 있다. API 보장 이름이 아니라 운영 원장이다. ready barrier가 어디까지 요구하는지 명시하고 warm-up도 모든 ranks 같은 sequence로 실행한다.

unique ID control message에는 deployment generation을 붙인다. timeout 뒤 old unique ID가 새 init에 적용되지 않게 한다. retry는 partial rank가 아니라 all ranks 새 generation으로 간다.

first collective 400 ms, 이후 2 ms 사건에서는 producer/control run을 만든다. same communicator/same buffer second call, new buffer same communicator, new communicator same buffer를 비교한다. new buffer만 느리면 registration/allocation 후보, new communicator만 느리면 connection/init state 후보, 첫 CUDA work 전체가 느리면 context/module warm-up 후보다. 세 변수를 한 warm-up으로 뭉개지 않는다.

graph capture를 도입할 때 eager ordering fixture를 그대로 replay fixture로 확장한다. graph A/B replay rank order, communicator generation, captured buffer와 event를 검사한다. capture warm-up call이 production sequence에 섞이지 않도록 별도 communicator 또는 all-rank canonical phase를 사용한다.

communicator teardown을 model hot-swap과 연결하면 old model requests와 new model init이 겹칠 수 있다. old comm G7은 drain/abort 중이고 new G8은 bootstrap한다. unique ID, streams, graph cache와 buffers를 세대별로 분리한다. G8 ready가 G7 cleanup을 기다려야 하는 resource와 병렬 가능한 resource를 명시한다.

GPU OOM 때문에 new comm init이 실패하면 old model을 이미 내렸는지에 따라 rollback이 다르다. swap transaction은 new resources ready 뒤 routing switch, old drain 순서를 선호할 수 있지만 peak가 커진다. 운영 정책이 무엇이든 partial communicator를 serving registry에 공개하지 않는다.

NCCL environment를 변경하려면 process restart가 필요한지 effective read 시점을 확인한다. runtime에 variable를 바꿔 existing communicator plan이 달라진다고 가정하지 않는다. comm creation 때 snapshot되는 config와 per-call 읽는 state를 구분한다.

## 56.2 `ncclAllReduce`는 `ncclInfo`를 만들어 enqueue한다

public [`ncclAllReduce`](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/collectives.cc#L166-L177)는 send/recv pointer, count, datatype, reduction op, communicator와 CUDA stream을 collective 정보로 묶어 enqueue path에 넘긴다. wrapper가 짧다는 사실은 collective 계산이 host return 전에 끝난다는 뜻이 아니다.

fixture info는 `send=A_r`, `recv=B_r`, `count=16`, `datatype=bf16`, `op=sum`, `comm=G7`, `stream=S_r`다. in-place라면 send/recv alias 규칙을 명시한다. pointer 주소가 유효해도 allocation length가 count×dtype bytes보다 작을 수 있고, 다른 GPU device allocation일 수 있다. validation이 어디까지 보장하고 어디부터 application 계약인지 구분한다.

**count=15인 rank 2 사건**

rank 0,1,3은 count 16인데 rank 2만 15를 넘겼다고 하자. 각 call의 pointer와 local count는 개별적으로 valid할 수 있다. 그러나 collective sequence의 signature가 rank 사이에서 갈린다. outcome은 단순한 Python exception으로 친절하게 끝난다고 보장할 수 없다. hang, protocol mismatch 또는 wrong result 위험으로 다룬다.

signature 비교를 NCCL 내부 자동 협상이라고 가정하지 않는다. application/framework control plane이 rank별 scheduled tensor shape와 collective order를 일치시켜야 한다. debug mode에서는 bounded signature hash를 all ranks에서 비교할 수 있지만 이 검사가 collective 자체보다 큰 overhead를 만들지 않게 한다.

first divergence는 model layer output shape가 rank 2에서 15였는지, wrapper가 count를 잘못 계산했는지, enqueue info가 달라졌는지 순서로 찾는다. info가 이미 15면 NCCL protocol tuning을 하지 않는다. source tensor logical shape는 16인데 info만 15라면 binding/wrapper owner다.

**dtype와 reduction semantics**

BF16 16 elements는 32 bytes지만 operation은 byte-wise sum이 아니다. NCCL datatype과 reduction op가 semantic을 정한다. rank 하나가 FP16 enum을 넘겨도 2-byte width가 같아 pointer bounds는 통과할 수 있고 numerical result는 틀린다. signature에 element count와 dtype/op를 모두 둔다.

pre/post-multiply reduction 또는 user operation 같은 기능은 current API/source 조건에서만 설명한다. 기본 sum fixture와 섞지 않는다. overflow/rounding과 accumulation algorithm의 numerical 차이는 semantic signature가 맞은 뒤 조사한다.

**stream identity는 info의 일부다**

rank별 CUDA stream object가 같은 pointer일 필요는 없지만 각 local producer→collective→consumer ordering을 만족해야 한다. producer kernel이 P_r를 S_prod에 쓰고 all-reduce를 S_comm에 enqueue한다면 event dependency가 필요하다. 같은 rank에서 source ready 전에 NCCL이 읽는 race는 rank signature가 모두 같아도 발생한다.

info ledger에는 producer-ready generation, stream과 dependency를 붙인다. collective output을 다른 stream에서 읽는 사건은 56.7에서 닫는다. 여기서는 API가 stream을 받는다는 사실이 host synchronize를 의미하지 않는다는 점을 고정한다.

**zero count와 pointer edge**

count 0, null pointer와 in-place alias의 exact 허용 조건은 current source/API 문서에서 확인한다. 일반 fixture의 규칙을 edge에 추측 적용하지 않는다. empty tensor가 rank마다 다르게 처리되면 sequence order가 갈릴 수 있다. framework는 모든 rank가 collective call을 생략하거나 동일 empty call을 수행하는 policy를 일치시킨다.

### 사건 1: rank 2 count=15

count 계산에서는 tensor `numel`과 byte length를 분리한다. BF16 `[2,8]`은 16 elements이고 32 bytes다. wrapper가 `nbytes=32`를 count로 넘기면 NCCL은 32 BF16 elements, 즉 64 bytes를 읽을 수 있다. allocation padding 때문에 bounds error 없이 다음 tensor까지 reduction할 수 있다. 모든 rank가 똑같이 틀려도 corruption이다.

local guard는 `count × datatype_size`가 allocation interval을 넘지 않는지와 semantic tensor view의 numel이 맞는지를 검사한다. strided/non-contiguous view를 contiguous payload처럼 넘길 수 있는지 binding contract를 본다. NCCL count는 arbitrary tensor stride를 표현하지 않는다. 필요한 pack/unpack temporary와 lifetime을 적는다.

send/recv alias는 exact in-place와 partial overlap을 구분한다. same pointer가 허용돼도 `recv=send+2 bytes`가 안전하다는 뜻은 아니다. allocator interval과 offset으로 alias class를 만든다. raw pointer 값은 ranks 사이에서 같을 필요가 없으며 ABA 때문에 단독 identity로도 부족하다.

rank별 dtype mismatch fixture는 BF16/FP16처럼 element width가 같은 경우와 BF16/FP32처럼 width가 다른 경우를 나눈다. 전자는 bounds가 같아도 numerical semantics가 다르고 후자는 payload/chunk도 갈린다. op mismatch도 sum/max처럼 output range가 정상일 수 있어 expected value로 잡아야 한다.

signature를 production에서 샘플링할 때 pointer/value는 빼고 function/count/dtype/op/root/comm generation을 canonical encoding해 hash한다. tensor semantic ID는 trace에 둔다. rank 간 signature hash exchange가 collective order 자체를 바꾸지 않는 out-of-band path인지 확인한다.

immediate validation error와 deferred async error를 request future에 다르게 표시한다. immediate error면 work가 enqueue되지 않았는지 source로 확인하고 buffer lease를 회수한다. async failure 가능성이 있으면 completion/error resolution까지 lease를 유지한다. 하나의 exception handler가 두 수명을 자동으로 처리한다고 가정하지 않는다.

NCCL bus bandwidth는 tool이 정의한 collective normalization metric이며 physical line rate와 같지 않을 수 있다. rank payload와 latency에서 application bandwidth를 계산하고 tool-defined factor를 명시한 뒤 physical counters와 별도로 보여 준다.

order mismatch fixture는 tensor A 합이 10, B 합이 100이 되게 한다. 동일 값이면 mismatch가 우연히 보이지 않을 수 있다. count mismatch fixture는 guard/sentinel interval로 out-of-bounds를 잡는다.

count는 bytes가 아니라 elements다. BF16 32 bytes를 count 32로 넘기면 64 bytes를 읽는다. allocation padding 때문에 실패하지 않을 수 있으므로 semantic numel guard가 필요하다.

non-contiguous tensor는 pack owner와 temporary가 필요할 수 있다. NCCL count가 stride를 표현한다고 가정하지 않는다. partial alias와 exact in-place도 구분한다.

signature hash가 네 rank 모두 같고 count/dtype/op가 16/BF16/sum이면 mismatch 가설을 약하게 한다. tensor A/B semantic ID도 같아야 한다. hash가 pointer를 제외하므로 local pointer bounds와 allocation generation은 별도로 검사한다. rank 2 source producer-ready event가 누락됐다면 transport error 이전에 wrong source race가 있을 수 있지만 hang owner와 numerical owner를 구분한다.

all ranks output 일부만 10이고 tail이 이전 값이면 count가 15인 rank가 아니라 모든 ranks count가 15로 잘못 계산됐을 수도 있다. signature equality는 통과한다. local semantic numel과 info count 비교가 필요하다. rank mismatch guard와 local binding guard가 서로 다른 이유다.

metric retention은 sequence 단위 raw series가 아니라 aggregate와 sampled trace를 쓴다. latency histogram은 size bucket/collective kind/comm group, errors는 class/rank role, selected plan은 change counter로 둔다. exact sequence는 incident trace에만 남긴다. cardinality 때문에 monitoring 자체가 proxy CPU를 방해하지 않게 한다.

## 56.3 group은 calls를 모으지만 device completion을 보장하지 않는다

NCCL [`group.cc`](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/group.cc#L27-L150)와 [`communicator group task fields`](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/include/comm.h#L410-L455)는 group nesting, per-communicator task와 async/preconnect 수명을 읽는 좌표다. `ncclGroupStart/End`는 여러 call의 host-side accumulation과 coordinated launch를 만든다. group end success를 output byte completion으로 쓰지 않는다.

fixture에서 layer 0과 layer 1의 all-reduce A/B를 한 group에 넣었다고 하자. 각 rank가 `A→B` 순서로 task를 append해야 한다. rank 0이 A→B인데 rank 1이 B→A라면 각 task count/dtype은 개별적으로 맞아도 collective sequence가 갈린다.

**nesting과 thread-local group state**

group start/end가 어느 thread의 state를 사용하고 nesting depth를 어떻게 다루는지 current source로 확인한다. start 하나를 빠뜨리거나 error path가 end를 건너뛰면 다음 request call이 이전 group에 섞일 수 있다. context manager/finally가 group depth를 닫아야 한다.

여러 communicator를 한 group에 넣는 경우 comm별 task list와 global coordination을 구분한다. group이 모든 comm을 하나의 membership으로 합친다고 생각하지 않는다. 각 communicator identity와 call sequence가 별도다.

**A→B와 B→A 사건**

rank 0/2는 layer A reduce 뒤 B, rank 1/3은 B 뒤 A를 append했다고 하자. pointer sizes와 BF16 sum은 모두 valid다. sequence 12에서 rank별 tensor identity가 다르다. hang이 sequence 13에서 보이더라도 first divergence는 sequence 12 task order다.

signature log에 layer name을 unbounded metric label로 쓰지 않고 trace에 tensor semantic ID를 둔다. sequence, count/dtype/op hash와 communicator generation은 bounded diagnostic record로 비교한다. application scheduler가 conditional expert branch 때문에 collective를 생략했는지 확인한다.

forced synchronous wait를 넣어도 order mismatch는 해결되지 않는다. 모든 rank가 같은 call graph를 실행하도록 scheduler/control flow를 고친다. 재발 fixture는 A/B ready 순서를 rank마다 바꾸되 enqueue sequence는 canonical order로 유지하는 coordinator를 검사한다.

**task accumulation과 memory lifetime**

group 안에 call을 append한 뒤 group end까지 info가 참조하는 buffer와 stream은 살아 있어야 한다. group end가 enqueue한 device work가 끝날 때까지 payload buffer도 살아 있어야 한다. host info object lifetime과 device buffer lifetime을 구분한다.

large group은 host launch overhead를 줄일 수 있지만 task descriptor, in-flight buffer와 delayed error surface를 늘릴 수 있다. group size option을 성능 카드로 외우지 않고 accumulated task count, launch plan, buffer peak와 error cleanup을 본다.

**group error cleanup**

두 번째 task validation이 실패했을 때 첫 task가 이미 어떤 상태인지 source path를 확인한다. group 전체가 launch 전 abort되는지 partial async job이 존재하는지에 따라 buffer cleanup이 달라진다. application은 error return 뒤 output 일부를 publish하지 않는다.

nested group error에서 depth와 task queues가 reset돼 다음 request가 clean generation으로 시작하는지 test한다. timeout 때문에 process를 종료하는 policy와 communicator abort를 호출하는 policy를 구분한다.

### 사건 2: A→B와 B→A

각 call signature는 맞지만 sequence tensor semantic이 다르다. group task trace에서 처음 갈린다. stream synchronize나 protocol 강제는 해결이 아니다. rank별 ready timing과 무관하게 coordinator가 canonical collective order를 정하도록 고친다.

재발 fixture는 A producer를 rank마다 다른 delay로 완료시켜도 enqueue order가 A→B인지 본다. conditional branch에서 empty tensor가 생길 때 call 생략 policy도 맞춘다.

group은 launch coordination과 mathematical fusion을 구분한다. A와 B를 group에 넣어도 두 tensor가 한 all-reduce로 합쳐지는 것은 아니다. 각각 signature와 output을 가진다. internal plan이 launch를 결합할 수 있어도 application semantic order는 유지돼야 한다.

group end latency를 calls append, preconnect, plan과 launch submission으로 가능한 범위에서 분해한다. group end가 오래 걸리면 device collective가 느린 것이라고 결론내리지 않는다. host plan/connection이 원인일 수 있다. group end가 빠르고 stream completion이 늦으면 device/transport progress를 본다.

task A validation 성공 뒤 B failure가 발생한 경우 안전한 application policy는 group output 전체를 publish하지 않는 것이다. buffer가 부분적으로 변했을 수 있으므로 retry는 clean destination 또는 reset을 사용한다. exact partial launch state는 current source로 확인한다.

group nesting leak fixture는 start와 end 사이에 exception을 주입한다. next request의 group depth가 0이고 pending tasks가 없는지 검사한다. binding context manager가 NCCL error뿐 아니라 application exception에서도 cleanup하는지 본다.

task가 많으면 descriptor/host memory와 buffer lease peak가 늘어난다. group을 크게 하면 launch overhead를 줄 수 있지만 error location과 tail latency가 나빠질 수 있다. 어느 layer 범위를 group할지는 dependency graph와 peak를 함께 고려한다.

graph replay sequence가 ranks 사이에서 달라질 수 있다. rank 0이 graph A를 두 번 replay하고 rank 1이 A/B를 replay하면 captured NCCL calls도 order mismatch다. graph launch coordinator가 canonical sequence를 유지해야 한다.

work queue backpressure는 host group end를 늦추고 activation buffer lifetime을 늘릴 수 있다. bounded outstanding collectives와 dependency를 둔다. producer가 무한히 앞서 enqueue한다고 overlap이 계속 좋아지지 않는다.

dashboard는 latency rank max와 p50을 함께 둔다. max가 critical path를 지배하고 차이가 straggler를 보여 준다. size bucket, comm group, transport family를 bounded label로 쓰고 sequence/tensor ID는 trace에 둔다.

group end failure에서는 output 전체를 publish하지 않고 clean retry destination을 쓴다. exception 뒤 group depth와 tasks가 reset됐는지 검사한다. large group의 descriptor/lease peak도 계산한다.

## 56.4 enqueue는 validation·task append·plan 경계를 지난다

[`ncclEnqueueCheck`](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/enqueue.cc#L3124-L3170)는 public info가 내부 enqueue machinery로 들어가는 고정 entry다. 본문에서 함수 이름을 나열하기보다 info가 validation되고 communicator task로 append되며 plan/device/proxy work로 바뀌는 상태 전이를 추적한다.

```text
Info(signature, pointers, stream)
→ local validation / comm state check
→ task append and sequence ownership
→ per-collective scheduling fields
→ plan partition/channel assignment
→ device work descriptor + proxy operations
→ stream launch
```

**validation의 끝을 쓴다**

local validation은 communicator state, count/datatype/op와 pointer/device 조건 일부를 검사할 수 있다. 이것이 all-rank signature equality, remote peer 건강과 future stream completion을 보장하지 않는다. 각 source branch 뒤 “여기까지 보장하고 무엇은 보장하지 않는가”를 적는다.

async error나 abort flag가 이미 설정된 communicator에 새 work를 enqueue하면 어떤 error를 반환하는지 확인한다. framework가 error를 무시하고 future를 만들면 consumer가 영원히 기다릴 수 있다. enqueue result와 request state 전이를 atomic하게 연결한다.

**task는 sequence와 tensor identity를 가진다**

내부 task field가 application layer name을 알지 못해도 diagnostic layer는 sequence를 tensor semantic과 연결해야 한다. task append 시점에 comm generation과 sequence, info signature hash, buffer generation을 trace한다. pointer raw value는 보안/ABA 때문에 단독 identity로 쓰지 않는다.

rank별 sequence가 같은지 online마다 별도 collective로 검사하면 recursion과 overhead가 생길 수 있다. scheduler deterministic invariant, debug control channel이나 sampled signature exchange를 사용한다. 장애 후 log를 합칠 수 있도록 clock/correlation을 둔다.

**plan construction과 cost candidate**

message size, collective kind, topology/capability, config/environment와 registration state는 algorithm/protocol/channel 후보와 비용에 영향을 줄 수 있다. “NCCL이 ring을 쓴다”를 default fact로 쓰지 않는다. actual selected algorithm과 protocol, channel count를 debug/trace/source predicate로 확인한다.

강제 environment는 후보를 제한하는 가설 스위치다. unsupported combination이 reject되거나 fallback되는 exact behavior를 current source에서 확인한다. startup log에 environment value가 찍혔다고 effective plan이 그 값이라고 결론내리지 않는다.

**Plan과 payload byte를 구분한다.**

**Plan lifetime과 graph.**

CUDA Graph capture/replay에서 communicator, buffer address, stream과 plan 관련 state lifetime을 확인한다. capture support predicate를 만족하지 않으면 eager launch 또는 reject로 가야 한다. graph node가 존재한다고 network/proxy completion까지 자동으로 재사용 안전한 것은 아니다.

replay 전에 communicator async error가 생겼거나 buffer generation이 바뀌면 stale graph launch를 막는다. framework graph cache key에 comm/device/topology generation과 relevant shapes를 포함한다.

registration state가 candidate에 영향을 주는 path가 있다면 first-call miss와 subsequent hit를 나눈다. cache key가 address generation, size와 transport를 어떻게 포함하는지 본다. allocator가 pointer를 재사용할 때 stale registration을 invalidate하는 owner가 필요하다.

graph capture plan은 replay마다 network condition을 재선택하지 않을 수 있다. topology/error generation 변화에서 graph를 invalidate한다. eager auto selection과 captured plan을 같은 조건에서 비교한다. capture가 plan overhead 전체를 없앤다고 쓰지 않는다.

continuous batching에서는 rows가 step마다 변해 message bucket과 selected protocol이 흔들릴 수 있다. graph padding으로 shape를 안정화하면 collective bytes가 늘어난다. padded rows의 reduction semantics와 extra HBM/link traffic을 함께 본다.

## 56.5 algorithm·protocol·channel은 세 개의 서로 다른 선택이다

algorithm은 collective dataflow family, protocol은 transport buffer/step 표현, channel은 work partition과 parallel progress 단위다. `Ring/Tree`, `Simple/LL/LL128`, channel count를 한 단어 “NCCL 경로”로 합치지 않는다. physical NVLink/PCIe/NIC edge는 transport/topology의 또 다른 축이다.

**작은 message와 큰 message의 목적.**

cost model이 message size와 topology를 어떻게 사용하고 candidate를 배제하는지 source에서 좁혀 읽는다. selection log가 있다면 comm generation과 collective sequence에 연결한다. init-time tuning table과 per-call dynamic field를 구분한다.

**Channel은 CPU worker thread가 아니다.**

channel은 device work와 transport/proxy progress가 분할되는 NCCL 실행 단위다. channel을 늘리면 parallel chunks와 kernel CTA/work descriptor, proxy operations가 늘 수 있다. 하지만 link, HBM, SM과 launch overhead를 경쟁한다. `NCCL_NCHANNELS`류 설정을 host thread count처럼 설명하지 않는다.

작은 message에 channel이 너무 많으면 padding/launch overhead가 커질 수 있다. 큰 message에 너무 적으면 available paths를 충분히 쓰지 못할 수 있다. effective channels와 per-channel bytes, kernel occupancy/SM contention, link utilization을 함께 본다.

serving compute kernel과 NCCL kernel이 같은 GPU SM/HBM을 경쟁할 수 있다. communication overlap이 늘어도 compute latency가 악화될 수 있다. collective 단독 bandwidth뿐 아니라 overlapped layer critical path와 goodput을 측정한다.

**Protocol forced 사건.**

운영자가 LL128을 강제했지만 current topology/capability에서 unsupported이거나 작은/큰 message에 느리다고 하자. startup environment log만 보고 selected protocol이라 믿지 않는다. collective sequence의 actual plan과 fallback/error를 확인한다.

force 해제, Simple, LL/LL128을 동일 topology와 message에서 비교한다. selection만 바꾸고 call signature/stream을 유지한다. forced path가 output checksum을 유지하지만 느리면 performance hypothesis다. error/hang이 생기면 capability gate와 cleanup을 먼저 본다.

**Algorithm 이름과 physical route.**

Ring algorithm이라고 traffic이 반드시 NVLink physical ring을 그대로 돈다는 뜻은 아니다. logical peers의 transport edge가 NVLink, PCIe shared memory 또는 network일 수 있다. Tree도 switch hardware tree와 동일하다고 단정하지 않는다. 57·58장의 topology evidence와 연결한다.

NVLS, CollNet, GIN 같은 path는 compile symbol 존재보다 current predicates, hardware/runtime와 actual selection을 확인한다. unsupported platform에서 silently selected됐다고 쓰지 않는다. fallback reason을 bounded enum/trace로 남긴다.

**사건 4: forced protocol 오판.**

environment에는 LL128이 적혀 있지만 actual plan은 fallback 또는 reject다. startup log가 아니라 sequence plan과 effective protocol을 본다. correctness가 같고 latency만 나쁘면 performance incident다. unsupported error가 async로 나타나면 cleanup을 확인한다.

force를 제거한 auto, Simple/LL/LL128 후보를 fixed topology/message에서 비교하고 version scope를 남긴다. 영구 override는 workload 변화에서 재검증한다.

communicator init/ready latency, enqueue/group latency, collective sequence latency의 rank max, selected algorithm/protocol/channel histogram, proxy progress stall, async error와 abort duration을 둔다. model/tensor exact name은 trace에 두고 metric label은 comm group, collective kind, size bucket, transport family와 error class로 제한한다.

`algbw`, `busbw`, link counters, kernel HBM bytes와 user critical path를 별도 panel에 둔다. 한 숫자의 하락을 algorithm 문제로 직결하지 않는다. rank skew와 stream queue, compute overlap을 함께 본다.

plan selection trace에는 sequence, message bytes, algorithm/protocol, channels와 transport family를 둔다. cost table 전체를 매 call dump하지 않는다. initialization tuning table hash와 per-call selected result를 나누고 environment/config generation을 붙인다.

candidate가 왜 제외됐는지 bounded reason을 얻을 수 있으면 기록한다. topology unsupported, protocol constraint, registration unavailable, size threshold와 forced config다. source가 reason을 직접 노출하지 않으면 debug log와 predicate를 교차하고 inference라고 표시한다. 선택 결과만 보고 cost를 역산하지 않는다.

32 MiB payload를 8 channels에 나누면 단순 평균은 4 MiB/channel이지만 alignment, rank steps와 protocol chunk 때문에 actual은 달라질 수 있다. 10 channels라고 exact 3.2 MiB descriptor를 단정하지 않는다. actual plan과 trace로 확인한다.

message bucket 경계에서 algorithm/protocol이 바뀌면 latency가 불연속일 수 있다. serving tensor shape가 threshold 주변을 오갈 때 jitter가 생긴다. override 전에 selected plan histogram과 rank critical path를 본다. padding으로 안정화하면 extra tensor/HBM bytes와 graph compatibility를 계산한다.

algorithm/protocol 강제 실험은 isolated process에서 하고 모든 rank의 environment hash를 맞춘다. 한 rank만 override되면 새로운 divergence다. 실험 뒤 deployment config에 override가 남지 않는지 확인한다.

heterogeneous transport에서는 channel별 path/latency가 다를 수 있다. collective는 가장 늦은 channel/rank에 묶인다. rank average보다 skew를 본다. channel ID를 무제한 metric label로 내보내지 않고 histogram과 sampled trace를 쓴다.

prefill TP collective와 decode collective는 size와 frequency가 다르다. prefill은 큰 rows로 link utilization이 좋을 수 있고 decode는 작은 message가 layer마다 반복돼 launch/protocol latency가 지배할 수 있다. large all-reduce 하나로 ITL을 예측하지 않는다.

channel별 proxy operation이 one-thread-per-channel이라고 추측하지 않는다. observed thread/channel count가 같아도 current source mapping을 확인한다. proxy가 busy-spin인지 blocking인지에 따라 CPU utilization 해석도 달라진다.

protocol override의 경쟁 가설은 size, channels, topology, registration miss와 concurrent compute다. actual plan 외 조건을 고정한다. force 해제로 회복됐다는 사실만으로 protocol implementation bug를 선언하지 않는다.

algorithm/protocol/channel은 every-call label 대신 histogram과 change event를 쓴다. config hash 변화 annotation과 fallback/reject reason을 둔다. exact channel은 sampled trace로 제한한다.

plan trace는 sequence, bytes, algorithm/protocol/channels/transport와 exclusion reason을 bounded하게 남긴다. cost table 전체 dump로 timing을 교란하지 않는다. graph plan은 comm/error generation 변화에서 invalidate한다.

channel 최적화는 compute-bound와 memory-bound overlap workload에서 각각 검증한다. collective 단독 최고 bandwidth가 serving 최고 goodput과 같지 않을 수 있다. protocol low-latency 이름도 모든 size의 우위를 뜻하지 않는다.

selected plan이 네 rank에서 compatible하고 algorithm/protocol/channels가 expected라면 force override를 하지 않는다. rank 2의 last proxy post와 first error, peer rank의 last receive를 잇는다. NIC/system log와 topology chapter evidence를 연결한다. 이 장에서는 NCCL progress가 어디서 멈췄는지만 닫고 physical link root cause는 다음 장으로 넘긴다.

forced Simple이 빠르고 auto가 느린 사건에서는 auto selected plan이 실제 무엇인지 확인한다. auto가 이미 Simple이면 environment 변화가 원인이 아니다. protocol은 달랐지만 channels도 동시에 달라졌다면 one-variable experiment가 아니다. actual plan을 고정하거나 source predicate를 통해 축을 나눈다.

channel을 8에서 16으로 늘려 collective 단독 latency가 10% 줄었지만 overlapped GEMM이 20% 느려졌다면 serving critical path를 계산한다. communication과 compute가 serial dependency인지 overlap인지에 따라 goodput 결과가 달라진다. HBM/SM contention과 launch CTA를 profiler로 확인한다. 높은 busbw만 보고 채택하지 않는다.

user guide의 environment 설명은 public contract이고 source predicate는 current implementation evidence다. 둘이 다르게 보이면 version/package identity를 먼저 확인한다. 문서가 algorithm 내부를 보장한다고 확대하지 않고 source가 public support policy를 대체한다고도 하지 않는다.

communicator는 membership과 topology, channels, tasks, error와 cleanup을 가진 generation object다. `ncclAllReduce`는 info를 enqueue할 뿐 host return으로 output을 공개하지 않는다. group은 calls를 모으지만 rank별 sequence를 자동으로 고쳐 주지 않는다. plan은 algorithm, protocol, channel과 transport를 각자 선택하며 이름 하나로 physical route를 증명하지 않는다.


## 56.6 device work와 proxy progress는 함께 끝나야 한다

plan이 정해지면 device-side work descriptor와 transport/proxy operations가 연결된다. GPU kernel만 launch됐다고 network transfer가 끝난 것은 아니고, proxy thread가 진행 중이라고 destination reduction이 완료된 것도 아니다. 한 collective의 last progress를 device와 host/network 양쪽에서 추적한다.

**Channel plan에서 work descriptor까지.**

work FIFO 또는 descriptor에는 collective function, count/chunk, buffer와 peers/protocol을 device kernel이 소비할 수 있는 표현으로 담을 수 있다. descriptor publish와 kernel launch의 ordering이 필요하다. stale communicator generation의 device state를 새 launch가 읽지 않게 teardown/reuse barrier가 있어야 한다.

**Transport edge와 proxy.**

intra-node peer path는 direct GPU access, shared-memory/PCIe 또는 NVLink-related transport를 사용할 수 있고 inter-node는 network proxy/transport가 개입할 수 있다. 정확한 selected transport를 source predicate와 log/trace로 확인한다. algorithm 이름으로 edge를 추정하지 않는다.

proxy는 network send/recv, progress와 completion을 host-side resource와 연결할 수 있다. CPU affinity와 scheduling이 나쁘면 GPU kernel이 network data를 기다릴 수 있다. 반대로 GPU가 앞선 compute 때문에 collective kernel을 시작하지 않으면 proxy가 idle일 수 있다. CPU utilization 한 숫자만으로 proxy bottleneck을 선언하지 않는다.

**Progress 원장.**

sequence 42에 대해 다음 시각을 둔다.

```text
host enqueue return
device kernel launch/start
channel c proxy op post
transport first/last byte progress
device reduction last step
stream completion marker
consumer first read
```

모든 backend가 이 exact timestamp를 제공하지 않을 수 있다. available log/counter와 correlation 한계를 적는다. 마지막 log line을 completion으로 과장하지 않는다. progress counter가 멈춘 지점과 async error 관찰 시점을 구분한다.

**One-rank proxy stall 사건.**

rank 2의 network/proxy progress가 멈췄는데 rank 0은 CUDA stream synchronize에서 기다린다고 하자. rank 0의 wait를 root cause로 부르지 않는다. communicator/signature/plan이 모두 같고 rank 2의 last transport progress에서 처음 갈린다면 NIC/transport/proxy thread와 async error를 본다.

CPU affinity를 바꾸거나 busy thread를 줄여 회복됐다고 proxy가 유일 원인으로 확정하지 않는다. network retry, peer failure와 resource exhaustion log를 함께 본다. rank 2 process의 clock과 rank 0 watchdog clock을 correlation한다.

**Registration과 buffer lifetime.**

network/RDMA path가 GPU 또는 host buffer registration을 요구할 수 있다. registration cache hit/miss와 first collective latency를 분리한다. buffer가 free/reused되기 전에 transport completion이 끝나야 한다. CUDA stream completion만으로 host proxy resource가 언제 safe free인지 current NCCL contract를 확인한다.

first collective가 느리고 subsequent가 빠르면 peer connection, registration, proxy thread start와 device warm-up을 각각 timestamp한다. dummy warm-up으로 숨기기 전에 어떤 state를 pre-create해야 하는지 찾는다. warm-up collective도 communicator sequence에 참가하므로 all ranks에서 같은 순서로 실행한다.

**HBM과 link bytes.**

all-reduce는 input read, partial reduce와 output write로 device memory traffic을 만든다. link payload와 HBM traffic은 다르다. protocol staging/scratch와 channel chunk가 추가 traffic을 만들 수 있다. profiler metric domain과 algorithmic lower bound를 구분한다.

device/proxy progress trace는 clock domain을 맞춰야 한다. CPU monotonic, CUDA events, NIC/proxy log의 origin이 다를 수 있다. correlation marker나 넓은 causal ordering을 사용한다. microsecond gap을 clock offset으로 오진하지 않는다.

rank 2 proxy stall에서 CPU가 sleep인지 runnable-but-not-scheduled인지, socket retry를 기다리는지 구분한다. affinity 변경으로 회복돼도 peer/network error log를 함께 본다. proxy thread CPU를 GPU/NIC NUMA와 가깝게 두는 것은 성능 정책이지 signature correctness가 아니다.

registration cache warm-up은 buffer allocator와 맞물린다. warm-up buffer와 production buffer 주소가 다르면 warm 효과가 없을 수 있다. actual cache key를 확인한다. 큰 persistent tensor와 transient activation의 registration policy가 다를 수 있다.

transport progress가 끝나도 device reduction/write가 남을 수 있고, device kernel이 running이어도 remote data가 오지 않아 기다릴 수 있다. GPU utilization 또는 proxy utilization 하나로 owner를 정하지 않는다. last completed step과 dependency를 같이 본다.

input payload rank당 32 bytes와 output 32 bytes를 적지만 ring의 link bytes를 단순 `32×(ranks-1)`로 확정하지 않는다. chunking, reduce-scatter/all-gather dataflow, protocol와 padding이 transaction을 바꾼다. HBM traffic도 intermediate read/write를 포함할 수 있다. source/trace의 actual step을 사용한다.

MoE EP traffic과 TP all-reduce가 다른 communicator여도 GPU streams, network와 proxy를 경쟁할 수 있다. 각 sequence는 독립이지만 physical resource는 공유한다. TP stall이 EP burst와 겹치는지 comm별 plan과 shared counters로 본다.

error log에는 rank time뿐 아니라 sequence와 last progress를 둔다. log flush 지연과 clock offset이 순서를 바꿀 수 있다. coordinator correlation과 causal relation을 사용한다.

progress stall alert는 `in_flight && no_progress_for` 형태가 유용하지만 progress signal 신뢰도를 확인한다. 작은 collective는 sampling에서 보이지 않을 수 있다. false positive와 detection delay를 측정한다.

device/proxy clocks를 correlation하고 rank 2 last progress를 찾는다. proxy CPU affinity 변경으로 회복돼도 network retry와 peer error를 함께 본다. CPU utilization 하나가 progress 증거가 아니다.

async error observation이 t0 실패보다 12초 늦었다면 watchdog/poll policy가 derivative wait를 길게 만든 것이다. root transport error 수정과 별개로 detection latency를 개선할 수 있다. poll을 빠르게 했을 때 CPU overhead와 false failure가 없는지 측정한다. timeout을 짧게 하는 것이 transport reliability 수정은 아니다.

rank max latency가 5 ms이고 p50이 2 ms라면 가장 느린 rank path가 전체를 묶는다. average 2.75 ms라는 보고는 user latency를 설명하지 못한다. rank 2 proxy/transport, GPU clock/compute contention과 topology를 비교한다. straggler identity가 매 call 바뀌면 shared congestion이나 scheduling 후보가 강하다.

결과가 준비됐다는 마지막 증거는 local CUDA stream ordering과 completion, 그리고 모든 rank의 collective progress다. 실패에서는 최초 transport/device error와 host watchdog 관찰 시각이 다르다. teardown은 in-flight work와 proxy/resource를 drain하거나 abort한 뒤 buffer와 communicator를 해제해야 한다.

## 56.7 host return이 아니라 CUDA stream completion이 결과를 공개한다

collective API의 host return은 work가 enqueue됐거나 error가 즉시 발견됐다는 뜻일 수 있다. output BF16 16 elements가 10으로 준비됐다는 증거는 CUDA stream ordering/completion이다. producer, NCCL, consumer가 같은 stream이면 FIFO ordering을 이용할 수 있다. 다른 stream이면 event/wait를 명시한다.

### producer→collective ordering

rank r의 producer kernel P_r가 input A_r를 S_prod에 쓴다. all-reduce를 S_comm에 enqueue한다. S_comm이 P_r completion event를 기다리지 않으면 NCCL이 old/partial input을 읽을 수 있다. 모든 ranks의 call signature가 같고 communication이 완전히 끝나도 wrong sum이 나온다.

### collective→consumer ordering

all-reduce는 S_comm에 있고 next matmul C_r은 S_compute에 있다. event 없이 C_r을 바로 launch하면 stale B_r를 읽을 수 있다. shape와 pointer는 정상이다. 동일 stream으로 바꾸면 문제가 사라져도 NCCL numerical bug가 아니라 ordering 가설을 증명하려면 event trace와 output checkpoint를 본다.

정상 graph는 `P_done → S_comm wait → allreduce → AR_done → S_compute wait → C`다. event를 host synchronize로 바꾸면 correctness는 생기지만 overlap을 잃는다. 필요한 dependency만 표현한다. event object를 pool에서 재사용하면 sequence/generation을 붙인다.

### group end와 future completion

framework가 group end 뒤 Python future를 complete로 표시하면 consumer가 너무 일찍 진행할 수 있다. future가 의미하는 것이 host enqueue 완료인지 device stream 완료인지 API 이름/문서에 적는다. device completion future라면 event 또는 stream callback과 communicator error를 연결해야 한다.

host-side batching coordinator는 enqueue acceptance와 output readiness 두 상태를 둔다. request scheduler가 KV block이나 activation buffer를 재사용하는 시점은 output readiness 뒤다. enqueue failure와 async failure는 서로 다른 transition으로 처리한다.

### stream synchronize가 hang으로 보이는 사건

rank 0이 `cudaStreamSynchronize(S_comm)`에서 멈췄다고 CUDA stream API가 원인인 것은 아니다. 앞선 NCCL work가 peer/proxy를 기다린다. rank별 sequence, last device/proxy progress와 async error를 모은다. synchronize stack만 보고 deadlock owner를 정하지 않는다.

timeout wrapper가 stream을 abandon하고 buffer/communicator를 free하면 in-flight DMA/kernel use-after-free 위험이 있다. abort/teardown contract를 따른다. timeout은 관찰 정책이지 device work를 자동 취소하는 API가 아니다.

### graph capture/replay

collective를 CUDA Graph에 capture할 때 supported conditions와 communicator/buffer/stream lifetime을 current NCCL/CUDA source로 확인한다. capture 성공은 replay마다 rank call order가 자동 조정된다는 뜻이 아니다. 모든 ranks가 compatible graph replay sequence를 실행해야 한다.

graph cache key에는 communicator generation, buffer addresses/shape, stream/capture context와 relevant NCCL state를 둔다. communicator abort 뒤 old graph를 replay하지 않는다. graph update로 buffer를 바꿀 수 있는 범위도 CUDA contract에 따른다.

### completion metric

`enqueue_latency`, `stream_wait_latency`, `collective_active/progress`, `consumer_wait`를 분리한다. GPU event duration만으로 network proxy queue 시간을 포함하는지 tool definition을 확인한다. rank max critical path를 사용자 latency와 연결하고 rank average로 straggler를 숨기지 않는다.

output validation은 all ranks에서 16개 값이 10인지 본다. 한 rank만 검사하면 asymmetric stale read를 놓친다. 큰 tensor는 sampled checksum과 deterministic reference를 쓰되 numerical tolerance와 sequence identity는 분리한다.

### 사건 3: consumer stream stale read

collective plan과 output event 전 checksum은 정상인데 consumer가 AR_done wait 없이 시작한다. first divergence는 consumer input checkpoint다. 같은 stream에서는 맞는다는 사실과 event graph를 증거로 ordering을 고친다. host synchronize가 아니라 explicit wait를 쓴다.

serving ready barrier가 어느 단계를 요구하는지는 SLO 정책이다. cold first request를 허용하면 usable에서 공개할 수 있다. 모든 request가 steady latency를 요구하면 controlled warm-up까지 수행한다. warm-up 실패를 무시하고 ready를 publish하지 않는다. warm-up tensor와 stream도 정상 ownership과 sequence를 가져야 한다.

producer→collective event와 collective→consumer event를 별개로 검증한다. 첫 dependency가 없으면 wrong input을 정확히 reduce하고, 두 번째가 없으면 correct output을 너무 일찍 읽는다. 최종 wrong logits만 보면 둘을 구분할 수 없다.

event pool에는 `(comm_generation, sequence, direction)` generation을 붙인다. AR_done event가 sequence 42 완료를 알린 뒤 43에 재record됐을 때 늦은 consumer가 42를 기다리는지 명확해야 한다. graph replay에서도 event/reference lifetime을 확인한다.

framework future가 host-enqueued를 뜻한다면 이름과 문서에 드러내고 device-ready future와 구분한다. scheduler는 device-ready 뒤에만 activation/KV buffer를 재사용한다. future cancellation이 CUDA/NCCL work 취소를 의미한다고 가정하지 않는다.

blocking/nonblocking config는 return/error observation과 teardown을 바꾼다. 단순 latency knob가 아니다. framework polling/future state machine이 effective config와 맞아야 하고 rank별 config hash가 같아야 한다.

GPU kernel name이 timeline에 나타나도 모든 ranks가 같은 semantic sequence라는 뜻은 아니다. correlation ID, stream과 comm generation을 연결한다. 이전 collective kernel이 늦게 끝난 것일 수 있다.

communication-compute overlap은 independent work가 있을 때만 가능하다. all-reduce output을 다음 layer가 즉시 필요로 하면 그 branch는 기다린다. stream 분리는 data dependency를 제거하지 않는다.

NCCL async success여도 CUDA stream은 다른 error를 가질 수 있고 반대도 가능하다. NCCL, CUDA launch/runtime와 proxy/network error를 한 field로 합치지 않는다. causal chain을 연결한다.

first collective 사건에는 CUDA context lazy init과 first allocation/fault도 있다. producer-only control, no-op kernel과 second collective를 비교해 communicator/connect와 general CUDA warm-up을 분리한다.

function output state를 다음 consumer의 owner와 잇는다. pointer/count가 plan descriptor가 되고 plan이 device/proxy work가 되며 stream completion이 output owner를 consumer로 넘긴다. 이 연결이 source 목록을 실행 설명으로 바꾼다.

producer-ready와 allreduce-ready event는 별개다. 첫 race는 wrong input을 정확히 reduce하고 둘째 race는 correct output을 일찍 읽는다. 최종 logits만으로 둘을 합치지 않는다.

event pool에는 comm generation과 sequence를 붙인다. future가 host-enqueued인지 device-ready인지 명시한다. cancellation이 device work 취소를 뜻하지 않는다.

framework의 process-group wrapper가 NCCL future를 어떻게 완성하는지 source를 읽는다. high-level `wait()`가 host enqueue, CUDA event, stream block 가운데 무엇을 의미하는지 확인한다. 이름이 같아도 blocking CPU wait와 stream-aware dependency가 다를 수 있다. 이 경계를 모르면 application code의 wait 유무를 평가할 수 없다.

마지막 handoff 표에는 `comm G7 seq42 result-ready event E42`를 쓴다. consumer가 E42를 wait하고 output checksum 10을 확인한 뒤 activation owner를 받는다. request가 끝나 buffer ref가 0이고 G7에 in-flight task가 없을 때만 allocator가 재사용한다. comm shutdown은 모든 sequences와 proxy resources가 닫힌 뒤다.

## 56.8 async error와 finalize·destroy·abort는 같은 동작이 아니다

[`ncclCommGetAsyncError`](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/init.cc#L3448-L3465)는 communicator의 비동기 오류를 host가 관찰하는 좌표다. [`ncclCommAbort`](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/init.cc#L3024-L3055)는 abort path를 읽는 고정점이다. finalize, destroy와 abort를 “comm free” 하나로 합치지 않는다.

### 최초 실패와 관찰 시각

rank 2 transport가 t0에 실패하고 proxy가 t1에 이를 기록하며 watchdog이 t2에 async error를 읽고 rank 0 request가 t3에 timeout될 수 있다. user symptom은 t3지만 first failure는 t0다. rank별 monotonic clock과 correlation을 맞춘다. log arrival order를 event order로 착각하지 않는다.

async poll interval이 길면 error detection latency가 늘어난다. 너무 짧으면 host overhead가 커질 수 있다. framework watchdog policy와 NCCL communicator state를 분리한다. poll이 오류를 만들지는 않지만 cleanup을 언제 시작하는지 결정한다.

### one-rank error가 all-rank hang처럼 보인다

rank 2가 async error를 가졌지만 rank 0/1/3은 다음 collective나 stream wait에서 멈춘다. 모든 ranks의 last completed sequence와 current sequence를 비교한다. rank 2 error가 sequence 42에서 발생했고 others가 42를 기다리면 derivative wait다. 각 rank를 독립 restart해 old communicator를 살리는 방식은 membership generation을 더 깨뜨릴 수 있다.

control plane은 first observed error를 group에 전파하고 new work admission을 막는다. in-flight requests를 fail하고 communicator generation을 abort/cleanup한다. 재시작은 all ranks가 새 unique ID/generation을 합의한 뒤 수행한다.

### graceful finalize와 abort

graceful path는 new work를 막고 in-flight work를 drain/finalize한 뒤 resource를 destroy하는 의미를 가질 수 있다. abort는 error/hang에서 progress를 중단하고 peer/proxy/resource를 revoke하는 path다. exact API semantics와 blocking mode를 current user guide/source로 확인한다.

abort 호출이 모든 CUDA work를 즉시 안전하게 사라지게 한다고 가정하지 않는다. buffer free, stream/context destruction과 process exit ordering을 source/driver contract로 확인한다. framework가 abort 뒤 model parameter를 즉시 다른 communicator에 재사용할 때 old work가 참조하지 않는지 보장해야 한다.

### shutdown race

sequence 99 all-reduce가 in-flight인데 server shutdown이 communicator cleanup을 시작했다. request cancellation, stream work, proxy resource와 communicator object의 네 owner가 있다. 정상 shutdown은 admission stop→requests drain/cancel→device/proxy completion 또는 abort→comm cleanup→buffers/context free 순서를 가진다.

signal handler에서 직접 복잡한 NCCL/CUDA cleanup을 호출하는 것이 안전한지 확인하지 않고 구현하지 않는다. signal은 shutdown coordinator를 깨우고 normal thread가 state machine을 실행하는 방식이 필요할 수 있다. source의 thread-safety/reentrancy 범위를 따른다.

### teardown fixture

collective 시작 전, host enqueue 직후, device/proxy progress 중, stream completion 직후 네 시점에 shutdown을 주입한다. deadlock, leak, double free와 stale output publish가 없는지 본다. abort timeout 뒤 process termination policy도 명시한다.

communicator cache/reference count가 있으면 last user와 in-flight generation을 비교한다. handle pointer가 null이 됐다는 사실만으로 proxy threads/registration이 정리됐다고 보지 않는다. resource metric과 thread join을 확인한다.

### error metric과 bounded labels

comm generation, rank, last completed/current sequence, async error class, cleanup state와 elapsed time을 기록한다. raw unique ID, buffer pointer와 tensor name은 metric label로 쓰지 않는다. trace에 bounded/hash identity를 둔다. watchdog timeout, CUDA error, NCCL async error와 process signal을 별도 source로 구분한다.

### 사건 6: rank 2 async error

rank 2 last progress와 error t0/t1, other ranks wait를 맞춘다. rank 0 timeout을 root cause로 기록하지 않는다. new work를 막고 group error를 전파하며 all ranks communicator를 abort/recreate한다. partial-rank retry를 하지 않는다.

failure injection은 proxy/network error와 rank process exit를 넣고 detection/cleanup latency를 측정한다. request future가 명확한 error로 끝나고 pool/buffer가 안전하게 회수되는지 본다.

### 사건 7: shutdown 중 in-flight collective

admission stop 없이 comm destroy를 호출해 stream/proxy가 old state를 참조한다. first divergence는 cleanup state transition이다. graceful drain과 bounded abort path를 구현한다. process signal, framework watchdog과 NCCL error가 동시에 cleanup을 시작해도 single owner가 실행해야 한다.

stream synchronize timeout은 관찰 정책이다. timeout 후 buffer free나 communicator destroy를 바로 수행하지 않는다. async error와 abort state machine을 시작하고 bounded cleanup 또는 process termination으로 간다.

async error polling은 first failure를 만들지는 않지만 detection latency와 cleanup 시작을 바꾼다. poll interval, watchdog timeout과 request SLO를 분리한다. error observation이 늦어 derivative hangs가 많아질 수 있다.

graceful shutdown은 admission stop, in-flight drain, finalize/destroy, buffer/context free 순서다. bounded drain이 실패하면 abort path로 승격한다. abort와 graceful destroy를 동시에 여러 threads가 호출하지 않게 single cleanup owner와 idempotent state를 둔다.

signal handler는 coordinator를 깨우고 normal thread가 cleanup state machine을 수행하는 식이 필요할 수 있다. NCCL/CUDA API의 signal safety를 확인하지 않고 handler에서 직접 호출하지 않는다. forced process kill policy도 resource manager 재시작과 연결한다.

async error가 난 뒤 prior sequence output을 사용할지는 request transaction 정책이다. sequence 41이 완료돼도 42 실패로 전체 request를 fail할 수 있다. partial layer state를 client에 publish하지 않는다.

watchdog가 process group 전체를 restart하면 다른 model/communicator blast radius를 기록한다. shared CUDA context reset이 unrelated work를 깨뜨릴 수 있다. worker isolation과 cleanup scope를 맞춘다.

abort latency도 metric이다. bounded timeout 뒤 force process termination이 필요할 수 있다. external orchestrator가 GPU/process resource를 clean generation으로 재생성하는지 확인한다.

destroy/finalize의 blocking 의미는 current API/config를 따른다. 이름만으로 drain을 추측하지 않는다. application shutdown state machine에 returned status와 async polling을 반영한다.

communicator metric에는 active generations, init/finalize/abort state, proxy threads, cache/registration bytes와 pending tasks를 가능한 범위에서 둔다. public API에 없는 값은 framework/debug trace로 얻고 overhead를 표시한다.

source claim은 보장하지 않는 것까지 기록한다. wrapper는 info를 만들고 enqueue check는 local queue 경계이며 group은 host accumulation이다. async getter는 관찰 경계이고 abort는 cleanup path다. 어느 하나도 혼자 all-rank completion을 증명하지 않는다.

one-rank error에서는 new admission을 막고 last accepted/completed sequence를 freeze한다. all ranks에 failure generation을 전파해 single cleanup owner가 abort/recreate한다.

## 56.9 공식 API의 반환값을 수명 상태로 번역한다

NCCL API를 읽을 때 가장 위험한 축약은 `ncclSuccess`를 “collective 완료”로 번역하는 것이다. Host call의 반환, group submission, CUDA stream에서의 device work 완료, proxy/network progress 완료와 application consumer가 결과를 읽을 수 있는 시점은 서로 다르다. Blocking/nonblocking communicator 설정과 함수 종류에 따라 host 반환 의미도 달라질 수 있으므로 함수 이름 하나로 통일하지 않는다.

[NCCL communicator API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)는 init, finalize, destroy와 abort를 서로 다른 operation으로 설명한다. [Collective API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html)는 collective call이 CUDA stream과 연관된다는 계약을 제공한다.

[Group calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html)는 여러 operation의 host-side grouping과 nonblocking 진행을 읽는 출발점이다. 공식 문서의 문장을 특정 source revision의 내부 state 이름으로 확대하지 않고, 문서 계약과 고정 source 구현을 나란히 둔다.

### 56.9.1 API별 “성공”이 닫는 경계

| 관측 | 닫힌 경계 | 아직 닫히지 않은 경계 |
|---|---|---|
| init 함수가 handle을 반환 | local call이 handle/result를 돌려줌 | 모든 peer path warm, 첫 collective steady latency |
| `ncclGroupEnd` 성공 | grouped host operations의 submission 상태 | 각 output의 CUDA stream completion |
| collective call 성공 | local validation/enqueue path가 성공 | peer progress, GPU completion, consumer visibility |
| `cudaEventQuery` success | event 앞 stream work 완료 | 다른 stream의 wait가 실제 삽입됐는지 |
| async error가 `ncclSuccess` | 관측 시점에 보고된 async error 없음 | future operation이 영구 성공함 |
| finalize 성공/진행 | communicator finalization protocol 상태 | application buffer와 graph를 즉시 free 가능함 |
| destroy 반환 | documented destroy lifecycle을 수행 | 잘못된 in-flight ownership이 사후 안전해짐 |
| abort 호출 | 실패한 communicator를 중단/회수하는 경로 진입 | 동일 handle 재사용, partial output publish 가능 |

이 표의 핵심은 negative space다. Host enqueue 성공은 remote rank가 같은 signature를 enqueue했다는 증거가 아니다. Async error poll이 아직 success라는 사실은 device stream event가 완료됐다는 증거가 아니다. Destroy가 cleanup API라는 사실은 application이 이미 buffer lease를 조기에 반납한 race를 되돌려 주지 않는다.

### 56.9.2 communicator state와 application state를 분리한다

운영 원장에는 NCCL 내부 state를 추측해 복제하지 않고 application이 책임지는 상태를 둔다. `CREATING`, `USABLE`, `DRAINING`, `FAILED`, `ABORTING`, `RETIRED`는 serving registry의 상태다. 각 state transition에는 communicator handle generation, membership hash, rank-device map, last submitted sequence, last device-completed sequence, async error observation과 buffer/graph lease count가 붙는다.

`USABLE` 전에는 request scheduler가 collective를 제출하지 않는다. `DRAINING`은 new enqueue를 막되 old sequence completion을 기다린다. `FAILED`는 output publication을 막고 all-rank failure coordination을 시작한다. `ABORTING`은 graceful completion을 기다리는 경로가 아니라 failure containment다. `RETIRED`는 handle pointer가 아니라 generation tombstone까지 남겨 late callback이 새 object를 건드리지 못하게 하는 상태다.

NCCL source의 communicator fields와 job/task ownership은 [`comm.h` 410–455행](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/include/comm.h#L410-L455)과 init source에서 확인한다. Application state 이름은 이해를 위한 외부 모델이며 NCCL enum이라고 주장하지 않는다. 대신 각 transition을 실제 API call, return, async poll, CUDA event와 registry mutation에 연결한다.

### 56.9.3 완료 증거는 rank별 세 clock을 가진다

rank마다 최소 세 시간을 기록한다. `t_host_return`은 collective/group call이 host에 반환한 시각이다. `t_stream_done`은 collective 뒤 같은 stream에 기록한 CUDA event가 완료된 시각이다. `t_consumer_ready`는 consumer stream이 그 event dependency를 만족해 output을 읽을 수 있는 시각이다. Proxy/network progress trace가 있으면 `t_proxy_terminal`을 추가하지만 이것을 application completion API처럼 일반화하지 않는다.

네 rank의 host clock이 동기화돼 있지 않으면 절대 timestamp만 비교하지 않는다. Process별 monotonic duration과 coordinator correlation을 함께 둔다. GPU timestamp와 CPU timestamp도 동일 clock이라고 가정하지 않고 NVTX/event correlation을 사용한다. “rank 2가 4 ms 늦었다”는 문장은 clock basis를 명시해야 한다.

예시 정상 timeline은 다음과 같다.

| rank | host enqueue start | host return | stream event done | consumer start | value |
|---:|---:|---:|---:|---:|---:|
| 0 | 100.000 ms | 100.041 ms | 100.612 ms | 100.620 ms | 10 |
| 1 | 100.006 ms | 100.047 ms | 100.609 ms | 100.618 ms | 10 |
| 2 | 100.011 ms | 100.053 ms | 100.615 ms | 100.623 ms | 10 |
| 3 | 100.004 ms | 100.046 ms | 100.611 ms | 100.619 ms | 10 |

수치는 fixture이며 실제 장비 측정값이 아니다. Host return과 event completion 사이 약 0.56 ms가 있다는 사실을 보여 주기 위한 식별 값이다. Consumer가 100.050 ms에 다른 stream에서 시작했다면 stale read 가능성이 있다. Host return 네 개가 모두 success여도 output은 아직 공개할 수 없다.

## 56.10 communicator generation이 섞인 multi-rank 사건

사건은 model hot reload 중 발생했다. Old model의 communicator G41을 drain하면서 new model은 G42를 만들었다. Rank 0, 1, 3은 new membership/unique ID로 G42를 publish했다. Rank 2는 control message retry 때문에 늦게 도착한 G41 ready callback을 새 registry slot에 기록했다. World size, rank index와 CUDA device는 모두 같았고 handle pointer 주소도 allocator 재사용 때문에 이전 값과 우연히 같았다.

첫 new request에서 rank 0, 1, 3은 G42 sequence 0 all-reduce를 enqueue했다. Rank 2는 G41의 last sequence 883 다음인 sequence 884를 enqueue했다. 모든 local call은 count 16, BF16, sum, valid pointer와 stream을 가졌다. Host call도 즉시 실패하지 않았다. 그러나 peers와 bootstrap/transport generation이 갈렸으므로 device completion event는 끝나지 않았다.

### 56.10.1 수치 timeline에서 host 성공과 hang을 분리한다

| 상대 시각 | rank 0 | rank 1 | rank 2 | rank 3 |
|---:|---|---|---|---|
| 0 ms | G42 seq0 enqueue | G42 seq0 enqueue | G41 seq884 enqueue | G42 seq0 enqueue |
| 0.05 ms | host success | host success | host success | host success |
| 0.8 ms | event pending | event pending | event pending | event pending |
| 100 ms | watchdog observes no completion | 동일 | async poll still no decisive local error | 동일 |
| 2,000 ms | group failure coordination | group failure coordination | old generation trace 발견 | group failure coordination |
| 2,050 ms | G42 admission closed | closed | G41 handle quarantined | closed |

이 timeline은 “NCCL call success인데 2초 뒤 timeout”을 network bandwidth 문제로 오해하기 쉽다. 그러나 first divergence는 0 ms 이전 registry publish다. Algorithm/protocol/channel을 강제하거나 timeout을 늘려도 generation mismatch는 해결되지 않는다. 59장의 일반 rank imbalance 진단이나 71장의 cluster hang 분류로 확장하기 전에 이 장에서는 object identity와 lifetime만 닫는다.

### 56.10.2 pointer·world size·rank는 identity가 아니다

Communicator identity tuple은 deployment/model generation, unique-ID digest, membership/rank-device digest와 local handle generation을 포함한다. Raw handle pointer는 trace correlation 보조값일 뿐이다. Allocator ABA 때문에 old object가 해제된 주소에 new object가 생길 수 있다. Pointer equality로 graph와 callback을 승인하지 않는다.

World size 4와 rank 2라는 두 숫자도 부족하다. G41과 G42 모두 같은 size/rank mapping일 수 있다. Device UUID까지 같아도 peer bootstrap session과 communicator-owned resources가 다르다. Registry key가 `(world_size, rank, device)`뿐이면 hot reload에서 세대를 섞을 수 있다.

Callback과 future에는 generation token을 캡처한다. Completion callback이 registry를 갱신할 때 current slot generation과 비교하고 다르면 stale completion으로 폐기한다. 폐기는 NCCL work가 사라졌다는 뜻이 아니므로 old owner가 별도로 drain/abort와 resource cleanup을 끝낸다. New slot을 mutate하지 않는다는 뜻이다.

### 56.10.3 UAF window는 destroy 호출보다 앞에서 열린다

Rank 2의 old G41을 잘못 retired했다고 하자. Host enqueue가 반환했다는 이유로 buffer lease와 communicator reference를 감소시키고 destroy를 호출하면 GPU/proxy가 아직 descriptor와 buffer를 사용할 수 있다. Crash가 즉시 나지 않고 later allocator reuse 뒤 unrelated tensor를 건드릴 수 있다. 그래서 UAF의 first divergence는 destroy API가 아니라 조기 completion 판정이다.

Lease는 `host_submitted`, `device_pending`, `consumer_pending`, `released`로 나눈다. Host return에서 `device_pending`이 된다. Collective stream event 완료와 async error resolution 뒤 consumer dependency를 만족해야 release 가능하다. Abort path에서는 정상 output을 publish하지 않으며 API가 요구하는 cleanup terminal과 application reference quiescence를 모두 기다린다.

CUDA graph는 communicator와 buffers의 lease를 더 길게 만든다. Graph executable이 node에 handle/pointer 관련 state를 포착했다면 마지막 replay completion과 graph destruction/invalidation 전 communicator generation을 retire할 수 없다. 새 communicator가 같은 주소를 얻어도 old graph replay를 허용하지 않는다.

### 56.10.4 반증 순서로 network와 stream 가설을 걷어낸다

첫 반증은 collective signature다. Count, dtype, op와 tensor semantic ID가 네 rank 모두 같았다. 따라서 rank 2 count mismatch와 A/B order mismatch는 약해졌다. 둘째는 producer readiness다. 각 rank source event가 collective stream 전에 완료돼 local source race를 기각했다. 셋째는 transport health다. G42로 새로 만든 독립 control fixture는 같은 links에서 완료돼 broad network outage를 약화했다.

넷째는 stream cycle이다. Dependency graph에는 producer→collective→consumer만 있고 consumer→producer 역방향 wait가 없었다. 다섯째는 communicator manifest였다. 세 rank는 generation 42/sequence 0, rank 2만 generation 41/sequence 884였다. 이 지점이 최초 의미 불일치다. Protocol 강제 실험은 필요하지 않았다.

독립 control communicator가 정상이라는 사실만으로 production communicator bug를 확정하지 않는다. Message size와 route가 다를 수 있다. 여기서는 rank별 manifest mismatch라는 positive evidence와 함께 network-wide outage를 반증하는 보조 증거로 사용한다.

## 56.11 abort·destroy·reload를 generation transaction으로 묶는다

장애 containment에서 rank 2만 communicator를 새로 만드는 것은 안전하지 않다. Collective membership의 generation이므로 네 rank 모두 request admission을 닫고 current in-flight inventory를 snapshot한다. Completed output과 pending/unknown output을 분리하며 pending 결과를 retry response로 publish하지 않는다.

### 56.11.1 정상 drain과 failure abort를 구분한다

정상 rolling reload는 new enqueue를 막고 last submitted sequence까지 stream completion을 기다린다. Consumer leases와 graph replay가 끝나면 finalize/destroy lifecycle로 간다. Failure path는 completion이 영원히 오지 않을 수 있어 async error/timeout policy로 abort를 선택한다. Abort 뒤 communicator를 usable state로 되돌리지 않는다.

`finalize`, `destroy`, `abort`를 동의어 cleanup으로 쓰지 않는다. 현재 official API의 blocking/nonblocking 조건과 반환 상태를 확인한다. Framework wrapper가 background progress를 쓰면 API return 뒤 어떤 poll/join이 필요한지 본다. Exact source는 public entry에서 communicator state mutation, async job과 resource reclamation까지 따라간다.

### 56.11.2 all-rank rollback protocol

Rollback coordinator는 generation G42를 `FAILED`로 표시하고 routing을 차단한다. 각 rank에서 last submitted, last device-completed와 current async error를 모은다. 정상 완료를 증명하지 못한 sequence의 output은 폐기한다. All ranks가 abort/cleanup terminal을 보고하거나 process replacement 정책으로 전환된 뒤에만 새 unique ID G43을 배포한다.

G43 init message에는 generation, membership hash, rank-device mapping과 expiry를 넣는다. Rank별 manifest를 coordinator가 비교하고 전원 usable을 확인한다. Warm-up all-reduce는 sequence와 expected value 10을 가진 canonical fixture로 실행한다. Stream event completion과 output value를 확인한 뒤 serving ready를 publish한다.

Old G41/G42 callback은 generation check에서 새 registry mutation을 거절한다. Old graph cache, event, buffer pool과 registration/cache entries가 어느 owner에게 귀속되는지 cleanup ledger로 확인한다. Process를 재시작했다면 OS/process boundary가 일부 resource를 회수하지만 external control-plane late message와 shared cache key는 여전히 generation을 검사한다.

### 56.11.3 rollback terminal

종료 조건은 `ncclCommInitRank` 성공 네 줄이 아니다. 모든 rank의 generation/membership/device digest 일치, canonical collective signature 일치, host enqueue success, collective stream event completion, consumer dependency completion, expected BF16 sum 10, async error success, old generation lease 0과 stale callback rejection이 필요하다.

성능 terminal은 별도다. First collective와 warm steady collective를 나눠 registration/connect/JIT 비용을 귀속한다. 이전 SLO와 비교할 때 host enqueue latency가 아니라 event-based collective interval과 consumer-visible interval을 사용한다. Correctness terminal을 통과한 뒤에만 channel/protocol이나 warm-up을 조정한다.

## 56.12 고정 source에서 lifetime owner를 따라간다

소스 walk의 목적은 함수 목록을 늘리는 것이 아니다. 각 함수가 어느 object를 만들고, 언제 다음 owner에게 넘기며, 어떤 return이 어느 lifetime 경계를 닫는지 찾는 것이다. Public API wrapper부터 device kernel까지 내려간 뒤 completion과 cleanup 방향으로 다시 올라와야 한다. Launch에서 읽기를 멈추면 가장 중요한 release 시점을 놓친다.

### 56.12.1 init entry에서 publish 전까지

[`init.cc` public init 구간](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/init.cc#L2472-L2574)에서 argument, current device, config와 job creation을 읽는다. 이어 [`init job과 communicator 구성`](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/init.cc#L1734-L1944)에서 bootstrap/topology/transport 관련 단계와 failure cleanup을 추적한다.

이 line들은 application registry의 publish를 보장하지 않는다. Framework가 handle/result를 받은 뒤 어떤 state check와 all-rank ready barrier를 두는지는 별도 owner다.

Init trace에는 `api enter`, `job submit`, `job terminal`, `handle returned`, `registry published`를 나눈다. Nonblocking mode에서 handle이 먼저 보일 수 있다면 registry는 pending state를 표현해야 한다. Poll result가 success terminal인지 in-progress인지 구분하고, failure가 발견되면 partial handle을 new request에 노출하지 않는다.

Unique ID bytes 자체를 운영 log에 그대로 노출하지 않고 digest와 deployment generation을 남긴다. Membership은 ordered rank-device digest로 만든다. Same set이더라도 rank order가 다르면 다른 mapping이다. Device는 local ordinal보다 UUID/BDF 등 안정된 identity를 manifest에 둔다.

Init failure source path에서 어느 allocation/job이 cleanup되는지 확인하되, application control message와 future가 자동 취소된다고 가정하지 않는다. NCCL local cleanup과 distributed orchestration cleanup은 다른 층이다. Late ready/failure callback이 새 generation을 mutate하지 않게 token compare가 필요하다.

### 56.12.2 public collective에서 communicator task까지

[`collectives.cc`의 all-reduce entry](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/collectives.cc#L166-L177)는 send/recv, count, datatype, operation, communicator와 stream을 `ncclInfo`로 묶어 enqueue로 보낸다. Wrapper가 열 줄 남짓이라는 사실은 work가 synchronous라는 뜻이 아니라 ownership이 enqueue machinery로 빨리 넘어간다는 뜻이다.

[`ncclEnqueueCheck`](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/enqueue.cc#L3124-L3170)에서는 communicator state와 info가 내부 task/plan path로 들어가는 경계를 본다. Local return code가 all-rank signature equality를 검사하는지 추측하지 않는다. Count/dtype/op/root/tensor semantic order는 framework control plane이 일치시킬 책임이 남을 수 있다.

Trace hook은 info 생성 전 logical tensor numel과 allocation interval, info의 element count/dtype/op, task append sequence와 communicator generation을 잇는다. BF16 16 elements를 32-byte count로 잘못 전달하면 pointer가 valid해도 bounds/semantic가 깨진다. Allocation padding 때문에 immediate error가 없을 수 있으므로 local semantic guard가 필요하다.

Stream identity에는 device, stream generation과 producer-ready event를 붙인다. Raw `cudaStream_t` 값은 destroy/recreate ABA가 가능하다. NCCL이 application stream을 소유하지 않더라도 work가 그 stream ordering에 의존하므로 application은 event completion까지 stream과 buffers를 유지한다.

### 56.12.3 group task와 error cleanup

[`group.cc` 27–150행](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/group.cc#L27-L150)은 group depth, jobs/tasks와 end processing을 읽는 고정점이다. Start/end가 host-side grouping을 제공하지만 output completion event를 대신하지 않는다. Nested group과 exception path에서 depth와 pending task list가 clean state로 돌아오는지 wrapper test를 둔다.

Group 안 task A가 accepted된 뒤 B validation이 실패하면 A가 어느 state인지 current source에서 확인한다. Application은 partial group output을 publish하지 않는다. Retry는 clean destination과 canonical sequence를 사용한다. Error handler가 end를 건너뛰어 다음 request C를 이전 group에 append하지 않도록 `finally` 또는 RAII owner를 확인한다.

여러 communicator call을 한 group에 넣을 때 각 comm generation과 membership을 별도 기록한다. Group object가 서로 다른 communicator를 같은 generation으로 만드는 것은 아니다. G41 task와 G42 task를 같은 group에 넣었다는 이유로 compatibility가 생기지 않는다.

Group return 뒤 buffer lease는 `device_pending`이다. 각 collective stream에 event를 두거나 framework work object가 제공하는 completion을 사용한다. Group end duration이 길면 host plan/preconnect일 수 있고, 짧아도 device/network completion이 길 수 있다. 두 interval을 분리한다.

### 56.12.4 plan·device work·proxy의 owner 연결

Enqueue 이후 task가 algorithm/protocol/channel plan, device work descriptor와 proxy operation으로 변하는 source를 call graph로 따라간다. 내부 구조 이름은 revision에 따라 바뀔 수 있으므로 `task owner→plan owner→device/proxy owner→completion cleanup` 역할로 기록한다. 하나의 plan이 여러 tasks를 담거나 task가 여러 channel chunk로 나뉠 수 있어 application sequence와 kernel launch 수는 일대일이 아니다.

Device kernel completion만 보고 remote transport/proxy resources를 즉시 재사용해도 되는지는 public completion contract와 source cleanup을 함께 본다. Application이 관찰하는 가장 안전한 output boundary는 collective stream ordering이다. 내부 proxy counter를 비공식 completion API처럼 사용하지 않는다. 다만 hang 분석에서는 channel별 device/proxy last progress를 비교해 어느 owner가 멈췄는지 좁힌다.

Proxy thread 또는 network plugin request는 communicator generation에 귀속한다. Abort가 시작된 뒤 old proxy callback이 new communicator queue를 깨우지 않게 object ownership과 cancellation/join을 본다. Plugin request handle과 registration도 address만이 아니라 generation/size/transport key를 가진다.

Plan selection trace는 actual algorithm, protocol와 channel을 남기되 이것을 correctness signature와 섞지 않는다. Rank signature와 generation이 먼저 맞아야 한다. Hang 때 protocol을 강제해 증상이 달라져도 mismatch가 해결됐다고 결론내리지 않는다. Timing 변화가 race window를 옮겼을 수 있다.

### 56.12.5 stream completion에서 consumer publication까지

Collective와 같은 stream의 later work는 CUDA stream order로 output 뒤에 온다. 다른 consumer stream은 event record/wait 같은 explicit dependency가 필요하다. Host API 반환 직후 CPU가 allocator lease를 회수하거나 다른 stream이 읽는 것은 completion 계약을 위반한다. Default stream 암묵 semantics에 기대지 않고 explicit dependency를 manifest에 남긴다.

Completion event는 collective enqueue 전에 기록하면 안 된다. Grouped calls이면 어느 collective까지 event가 덮는지 sequence를 표시한다. Event object도 generation과 owner를 가진다. Event query failure, not-ready와 success를 구분하고 success일 때만 해당 event 앞 work의 completion을 승인한다.

Consumer가 output을 읽은 뒤 buffer를 release할 수 있다. In-place all-reduce면 input/output lease가 같은 allocation interval에 묶인다. Out-of-place면 send buffer read completion과 recv buffer consumer completion이 다른 terminal을 가질 수 있다. Memory를 아끼려고 send lease를 host return에서 놓지 않는다.

Distributed framework future가 `completed=true`라고 보고하는 시점이 host enqueue인지 stream event인지 확인한다. Python future 완료가 device readiness를 뜻하지 않는 API도 있을 수 있다. Work object의 `wait`, stream synchronization insertion과 CPU blocking 의미를 source/documentation에서 분리한다.

### 56.12.6 async error 관찰에서 teardown까지

[NCCL communicator API 문서](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)는 async error 조회와 finalize/destroy/abort API를 구분하는 기준이다. Async error polling 간격은 발견 latency를 바꾸지만 최초 failure 시각과 같지 않다. Rank 2가 40 ms에 error를 poll하고 rank 0이 2 s watchdog에서 파생 timeout을 본다면 root observation은 rank 2의 local error일 수 있다.

Async error result가 success인 한 번의 sample은 communicator가 영구 healthy라는 증거가 아니다. Poll sample에는 comm generation, last submitted/completed sequence와 timestamp를 붙인다. Error가 발견되면 new enqueue admission을 atomic하게 닫고 peer coordination을 시작한다. Poller와 request thread가 race해 failure 뒤 task를 append하지 않게 한다.

Finalize는 정상 completion을 향한 protocol, abort는 failure containment, destroy는 communicator resource lifecycle의 public operation으로 구분한다. Exact ordering과 nonblocking return 의미는 official docs와 pinned source를 따른다. Wrapper가 `destroy`를 일반 finally에 넣어 in-flight work를 무조건 안전하게 만든다고 가정하지 않는다.

Teardown trace는 admission close, last submitted, last stream-completed, graph invalidated, buffer leases zero, finalize/abort enter/terminal, destroy enter/return, registry tombstone과 late callback count를 가진다. Process exit이 cleanup을 대신하는 정책이라면 request result와 control-plane generation을 어떻게 정리하는지 적는다.

## 56.13 host enqueue 성공 뒤 buffer를 재사용한 사건

두 번째 핵심 사건은 communicator generation이 아니라 completion 의미를 잘못 해석한 경우다. Framework wrapper는 `ncclAllReduce` host return이 success이면 work future를 complete로 표시했다. Activation pool은 recv buffer B를 즉시 다음 layer temporary C로 재할당했다. NCCL device work는 같은 주소 B에 sum 결과를 쓰는 중이었고 consumer stream은 event wait 없이 C를 읽었다.

### 56.13.1 32-byte fixture의 시간선

BF16 16 elements의 payload는 32 bytes다. Rank별 input은 1, 2, 3, 4이고 expected sum은 10이다. Buffer interval `B=[0x1000,0x1020)`라 하자. Allocator generation은 B가 77, 재사용된 C가 78이다. 주소는 같지만 의미와 generation은 다르다.

| 상대 시각 | host/stream 사건 | allocation state |
|---:|---|---|
| 0.000 ms | producer 완료 event | B gen77 input ready |
| 0.010 ms | all-reduce enqueue | B gen77 device_pending |
| 0.045 ms | host API success | 잘못된 wrapper가 completed 표시 |
| 0.052 ms | pool release/reallocate | same address C gen78 |
| 0.070 ms | consumer stream C read | event wait 없음 |
| 0.410 ms | NCCL write/progress | old B semantic으로 same address 접근 |
| 0.620 ms | collective event would complete | 이미 generation 침범 |

결과는 세 형태로 보일 수 있다. Consumer가 먼저 읽으면 이전 또는 C 초기화 값을 본다. NCCL write가 C 사용 중 겹치면 C가 부분적으로 10으로 오염된다. Allocator가 다른 object metadata를 배치하면 UAF/illegal access가 later sync에서 나타날 수 있다. Host call과 immediate CUDA API가 모두 success여도 가능하다.

### 56.13.2 first divergence와 반증

Rank signature와 communicator generation은 모두 같았다. 따라서 count/order/generation mismatch를 기각했다. Network control fixture와 proxy progress도 정상 범위였다. Collective stream에 event를 추가하고 buffer release를 event 뒤로 옮기자 오류가 사라졌다. 단순 `cudaDeviceSynchronize`도 증상을 없앴지만 전역 serialization이므로 원인 국소화용 반증일 뿐 수정이 아니다.

Allocator를 느리게 해도 오류 빈도가 줄었다. 이것은 allocator bug 확정이 아니라 reuse window가 닫혔다는 timing evidence다. 다른 stream consumer에 explicit wait만 추가했는데 allocator release가 여전히 host return이면 NCCL write UAF 위험은 남는다. Producer/consumer ordering과 allocation lease를 각각 고친다.

Compute Sanitizer나 강제 synchronization이 race window를 바꿀 수 있다. Tool에서 재현되지 않는다고 race를 기각하지 않는다. Generation-tagged allocation trace, stream event timeline과 first corrupted interval을 결합한다. Raw pointer 로그만 보면 gen77과 gen78이 같은 object처럼 보이므로 allocation generation이 핵심이다.

### 56.13.3 수정과 terminal

Framework work object는 `SUBMITTED`, `DEVICE_PENDING`, `DEVICE_DONE`, `CONSUMER_VISIBLE`, `RELEASED` 상태를 가진다. NCCL host return은 `SUBMITTED→DEVICE_PENDING`만 허용한다. Collective stream completion evidence가 `DEVICE_DONE`을 만들고, consumer stream wait가 `CONSUMER_VISIBLE`을 만든다. Buffer별 last consumer에 따라 lease를 release한다.

Same-stream consumer는 stream ordering으로 visibility를 얻지만 allocator가 같은 stream completion을 아는지 확인한다. Stream-ordered allocator라면 free/reuse operation의 ordering contract를 사용하고, host pool이면 event callback/poll로 lease를 관리한다. 서로 다른 allocator API의 보장을 섞지 않는다.

회귀 fixture는 consumer delay를 0/중간/완료 뒤로 바꾸고 allocator reuse를 강제한다. In-place/out-of-place, group 두 collectives, graph replay, async error와 cancellation을 포함한다. Output exact value 10, sentinel guard, allocation generation assertion과 no-late-write를 검사한다.

Rollback은 wrapper binary만 되돌리지 않는다. Host-complete로 만들어진 cached graph/work future, buffer pool entries와 in-flight requests를 drain한다. Old generation이 lease 없이 계속 실행할 수 있으면 process replacement가 더 안전할 수 있다. New replicas는 event-based fixture를 통과한 뒤 traffic을 받는다.

**Rank별 완료 원장을 실제 request에 붙인다.** Request 912의 TP group이 G43이라면 rank별 row에 model/request generation, communicator generation, collective sequence, tensor semantic ID, count/dtype/op, producer event, host enqueue result, collective event, consumer wait와 release generation을 둔다. 모든 row가 같은 signature를 갖는지와 각 local dependency가 닫혔는지는 다른 검사다.

예를 들어 네 rank 모두 G43/seq7/count16/BF16/sum이면 distributed signature는 맞다. 그러나 rank 1만 producer event가 없으면 source race가 남고, rank 3만 consumer wait가 없으면 stale read가 남는다. Distributed equality와 local stream correctness를 하나의 green check로 합치지 않는다.

Last submitted sequence와 last completed sequence도 정의를 고정한다. Submitted는 host task append인지 group end return인지 정하고, completed는 CUDA event terminal인지 framework future인지 정한다. 서로 다른 rank가 다른 정의를 metric에 쓰면 gap을 비교할 수 없다. Application 원장에서는 collective 뒤 event 완료를 device-completed 기준으로 사용한다.

Outstanding은 `submitted-device_completed`로 계산하되 sequence가 contiguous하다는 invariant를 확인한다. Error나 cancellation으로 hole이 있으면 단순 subtraction이 틀릴 수 있다. Sequence별 terminal 또는 high-watermark+hole bitmap을 diagnostic snapshot에 둔다. Metric에는 bounded outstanding count만 노출한다.

**Async error timeline은 최초 오류와 파생 대기를 분리한다.** Rank 2가 40 ms에 transport error를 처음 관찰하고 rank 0/1/3이 2 s에 watchdog timeout을 낸 경우 네 timeout을 동등 root cause로 세지 않는다. Error occurrence, NCCL async poll observation, framework propagation, admission close, peer abort와 request failure 시각을 따로 기록한다.

Poll interval이 100 ms이면 error observation은 실제 occurrence보다 최대 그 정도 늦을 수 있다. Poll을 촘촘히 해도 stream completion 증거를 대신하지 않는다. 너무 공격적인 poll/log가 progress thread를 방해하지 않는지 본다. Async error enum과 source rank를 bounded label로, exact sequence와 transport detail을 incident trace로 둔다.

Rank 하나가 error를 봤을 때 peer ranks에 failure를 어떻게 알리는지 framework owner를 찾는다. 같은 failed communicator로 새 all-reduce를 사용해 error를 합의하려 하면 이미 progress가 멈춘 path에 의존한다. Out-of-band coordinator/watchdog 또는 process-group control channel의 failure semantics를 확인한다.

Error propagation이 늦어 rank 0이 seq8을 enqueue하고 rank 2가 seq7에서 abort했다면 new sequence가 추가 hang처럼 보인다. Admission close가 poller와 enqueue thread 사이에 atomic해야 하는 이유다. Registry state를 `FAILED`로 바꾼 뒤 task append가 불가능하도록 generation/state check를 enqueue 직전에도 둔다.

**Destroy와 abort의 double-owner를 막는다.** Timeout handler, model unload thread와 process shutdown hook이 동시에 cleanup을 호출할 수 있다. Communicator lifecycle owner를 하나로 정하고 다른 caller는 idempotent transition 요청만 한다. `USABLE→DRAINING→RETIRED`와 `USABLE/DRAINING→FAILED→ABORTING→RETIRED` 경로를 CAS나 lock으로 보호한다.

Abort와 destroy가 동시 호출되는 exact public API 허용 여부와 source behavior를 확인하지 않은 채 “둘 다 해도 cleanup”이라 생각하지 않는다. Wrapper는 한 terminal path만 소유하고 completion future를 모든 waiter에게 공유한다. Late second caller는 already-retired generation을 발견하고 raw handle을 다시 호출하지 않는다.

Handle을 registry에서 먼저 삭제하면 late callback이 lookup miss로 조용히 사라져 resource cleanup evidence를 잃을 수 있다. Tombstone에는 generation, terminal reason, abort/destroy result, lease count와 expiry를 둔다. New handle이 같은 pointer를 얻어도 tombstone generation으로 stale callback을 식별한다.

**Graph와 communicator 수명을 함께 시험한다.** Graph capture G43-C5가 communicator G43과 buffers generation 91을 포착했다고 하자. Replay R1/R2가 끝나기 전에 model reload가 G44를 publish해도 C5를 G44 request에 사용하지 않는다. Graph key에 communicator/parameter/buffer generation과 collective shapes를 넣는다.

Capture 자체의 host return과 captured graph execution completion도 구분한다. Capture 중 collective node가 등록됐다는 사실은 실제 network work가 완료됐다는 뜻이 아니다. Replay 뒤 completion event가 output readiness를 증명한다. Graph destruction 전 last replay completion을 기다리고 communicator retirement와 ordering을 명시한다.

Graph miss가 eager fallback으로 갈 때 eager path가 같은 communicator generation을 쓰는지 확인한다. Capture G43인데 registry current가 G44라면 request snapshot policy에 따라 old G43을 끝까지 소유하거나 request를 취소한다. 한 request 안에서 graph node는 G43, eager tail은 G44를 쓰지 않는다.

**Buffer lease를 bytes와 구간으로 검증한다.** Send/recv interval, allocation generation, alias class와 last NCCL/consumer sequence를 기록한다. In-place는 exact same interval, out-of-place는 distinct interval, partial overlap은 별도 unsupported/validated 상태다. Pointer equality만 기록하면 offset과 length bug를 놓친다.

Pool capacity 때문에 event pending buffer를 즉시 재사용하고 싶다면 더 많은 buffers, backpressure 또는 stream-ordered allocator를 선택한다. Host-success 조기 release로 capacity를 만든 것은 최적화가 아니라 lifetime violation이다. Queue admission과 outstanding collective cap을 조정해 peak를 제한한다.

Memory regression은 payload 32 bytes만 세지 않는다. Event objects, work descriptors, graph-held allocations, registration와 fallback temporary가 lease를 늘릴 수 있다. 정상 steady state와 failure/abort peak를 나눈다. Abort가 pending buffers를 언제 안전하게 reclaim하는지는 API/source terminal을 따른다.

**반증 matrix는 한 축씩 움직인다.** Same G43에서 event wait만 추가해 stale read가 사라지면 consumer ordering 후보가 강해진다. Allocator reuse만 금지해 corruption이 사라지면 buffer lifetime 후보가 강해진다. Network path를 바꿔도 동일 generation/lease 오류가 남으면 transport hypothesis가 약해진다.

`cudaDeviceSynchronize`로 해결되면 GPU work 미완료 범주를 지지하지만 어느 stream edge나 lease가 문제인지는 확정하지 못한다. 모든 rank를 serialize하면 call order race까지 바뀐다. 정확한 fix는 collective stream event, consumer wait와 release terminal을 최소 범위에 둔다.

Communicator generation incident에서는 같은 network에서 fresh G43 control이 성공하는 것, old G41 manifest가 rank 2에만 남은 것, registry callback generation mismatch를 함께 본다. Pointer 주소를 바꾸기 위해 allocator를 perturb해 hang이 사라져도 ABA symptom일 뿐 identity fix가 아니다.

**재승인 fixture를 startup과 serving으로 나눈다.** Startup fixture는 all-rank unique-ID/membership/device digest, init terminal과 canonical 32-byte all-reduce를 검증한다. 첫 call과 두 번째 call 시간을 나눠 warm cost를 기록한다. Output 10과 collective event completion을 모두 확인한다.

Serving fixture는 prefill/decode step에서 실제 framework work object와 buffer pool을 통과한다. Same/different stream consumer, group A/B, graph hit/miss, cancellation과 hot reload를 포함한다. Rank 하나에 producer delay와 async failure를 주입해 admission close와 peer cleanup을 본다.

Teardown fixture는 outstanding 0의 graceful drain, outstanding work 중 failure abort, init partial failure, double cleanup caller와 stale callback을 포함한다. 각 case가 exactly one terminal owner, no new enqueue, no output publication for unknown work, lease zero와 tombstone을 만족해야 한다.

**Rollback 후 관측 terminal을 확인한다.** Metric에는 active communicator generations, unexpected generation mismatch, outstanding work, async error class, abort/destroy terminal, event-based latency와 buffer lease age를 둔다. Exact request/sequence는 trace exemplar로 연다. Rank/device/model 같은 unbounded 조합을 무제한 label로 만들지 않는다.

Host enqueue latency가 정상인데 event latency가 늘면 device/transport/dependency를 본다. Event latency도 정상인데 consumer-visible latency가 늘면 cross-stream wait와 scheduler를 본다. Host return에서 work success를 세면 이 세 구간이 하나로 뭉개져 incident를 재현할 수 없다.

최종 incident 문장은 다음처럼 닫힌다. “G41 late callback이 rank 2의 G42 registry slot을 덮어 세 rank는 G42 seq0, rank 2는 G41 seq884를 제출했다. 네 host enqueue는 성공했으나 stream events는 완료되지 않았다. Generation-token publish와 all-rank G43 transaction으로 수정했고 stale callback, boundary timeline, output 10과 lease-zero terminal을 통과했다.”

Buffer 사건도 구체적으로 쓴다. “Host return 0.045 ms에 gen77 buffer를 release해 0.052 ms gen78로 재사용했고 NCCL completion은 0.620 ms였다. Event-based work state와 consumer wait/lease를 도입한 뒤 forced-reuse fixture에서 late write가 없고 output 10, graph/reload terminal이 일치했다.” 이 정도의 시간·세대·좌표가 있어야 수정이 재검증 가능하다.

**운영자가 마지막으로 확인할 lifetime checklist.** Init에서는 unique-ID generation, ordered membership, rank-device mapping, blocking mode와 publish barrier를 확인한다. Enqueue에서는 communicator generation, sequence, tensor semantic ID, count/dtype/op, allocation interval과 producer readiness를 확인한다. Group에서는 rank별 canonical order, depth cleanup과 partial failure publication 금지를 확인한다.

Plan/launch에서는 selected algorithm·protocol·channel을 관측하되 signature와 generation 확인보다 앞세우지 않는다. Device/proxy progress를 각각 보며 application completion은 collective stream evidence로 판정한다. Consumer에서는 same-stream ordering 또는 explicit event wait와 allocation lease를 확인한다. Python/C++ future의 `done`이 이 경계 중 어디를 뜻하는지도 고정한다.

Error에서는 최초 local error, poll observation, peer propagation과 derivative timeout을 분리한다. New enqueue admission을 닫은 뒤 pending outputs를 completed/failed/unknown으로 나눈다. Unknown output을 retry 성공처럼 publish하지 않는다. Graceful drain이 가능한 정상 reload와 completion을 기다릴 수 없는 abort path를 구분한다.

Teardown에서는 last submitted/completed, graph last replay, consumer last use, lease zero와 exactly-one cleanup owner를 확인한다. Destroy/abort return을 임의로 device quiescence라고 확대하지 않고 current official contract와 wrapper completion을 따른다. Registry는 pointer를 삭제하는 데서 끝나지 않고 generation tombstone으로 late callback을 거절한다.

Multi-rank terminal은 가장 느린 rank를 포함한다. Rank 0의 event success 하나로 group completion을 승인하지 않는다. 각 rank가 같은 communicator manifest와 expected collective signature를 제출하고 local event/output을 통과해야 한다. Collective API가 자동 global acknowledgement를 제공한다고 추측하지 않고 framework가 필요한 합의를 소유한다.

이 checklist를 runbook에 넣을 때 raw pointer, unique ID와 user tensor value를 노출하지 않는다. Digest, generation, bounded reason과 sampled sequence를 사용한다. Metric cardinality를 제한하면서도 incident trace에서는 init→enqueue→event→consumer→teardown 인과를 복원할 수 있어야 한다.

Upgrade에서는 official API 문서의 반환/cleanup 계약, pinned source의 async job와 state transition, framework work-object semantics를 각각 diff한다. NCCL version만 같아도 framework가 host future 의미를 바꾸면 buffer lifetime이 달라질 수 있다. Framework가 같아도 NCCL nonblocking/finalize behavior를 쓰기 시작하면 teardown terminal을 다시 검증한다.

마지막 원칙은 단순하다. Communicator는 handle 값이 아니라 세대가 있는 distributed resource이고, collective는 host function이 아니라 stream-ordered distributed work다. Generation과 completion evidence를 잃으면 정상 return 뒤에도 hang과 UAF가 생긴다. 두 값을 끝까지 보존하면 timeout 숫자를 늘리기 전에 최초 잘못된 publish나 release를 찾을 수 있다.

따라서 코드 리뷰에서 `ncclSuccess` 다음 줄만 보지 않는다. 그 성공이 init handle, host enqueue, group submission, async poll 또는 cleanup 중 어느 경계를 뜻하는지 이름 붙인다. 다음 owner가 요구하는 event, lease와 generation을 전달하는지 확인한다. 생애주기 표의 빈 칸 하나가 production에서는 수초짜리 hang이나 훨씬 늦게 나타나는 memory corruption이 될 수 있다.

**왜 enqueue 성공이 완료가 아닌가.** host return은 operation이 CUDA stream에 제출됐다는 뜻이며 peer transport와 destination write가 끝났다는 뜻이 아니다. 왜 buffer를 바로 재사용하면 간헐적 오답이 되는지는 producer stream, NCCL stream과 consumer stream 사이 event ordering이 깨지기 때문이다. group call은 launch ordering 비용을 줄여도 communicator별 collective tuple 일치를 대신 보장하지 않는다.

**완료 진단 결정 트리.** 모든 rank가 enqueue 전 멈추면 application branch·communicator tuple을, enqueue 뒤 한 channel만 멈추면 transport/peer progress를, NCCL은 끝났는데 consumer 값이 틀리면 stream event·buffer lifetime을 본다. 정상 통제에서는 event 전 조기 read가 실패하고 event 후 checksum이 reference와 맞아야 한다. timeout 증액은 이 판정의 수정이 아니다.

## 56.14 참고: 대표 all-reduce 장부의 여섯 반례

이제 앞의 lifecycle을 두 시간 조사표로 압축해 현장 순서에 맞춘다.

조사표에서 확인한 경계는 다음 관측 dashboard로 이어진다.

두 시간 workbook의 첫 종료 조건은 모든 ranks communicator identity와 sequence/signature가 같다이다. 둘째는 selected plan과 last progress가 설명된다이다. 셋째는 output readiness event가 consumer ordering을 만든다이다. 넷째는 error/cleanup generation이 buffer와 communicator를 안전하게 해제한다이다.

성능 변경은 이 네 correctness 조건을 유지한 뒤 측정한다. channel/protocol override가 latency를 줄여도 cleanup fixture가 hang하거나 rank output이 다르면 채택하지 않는다. first collective warm-up이 startup memory peak를 넘기면 readiness policy를 다시 설계한다.

incident report에는 symptom rank가 아니라 first divergent rank/sequence를 제목에 쓴다. “rank 0 watchdog timeout”보다 “rank 2 sequence 42 proxy progress 이후 async transport error”가 재사용 가능한 지식이다. A/B order mismatch와 stream stale read도 같은 방식으로 의미 owner를 기록한다.

재발 CI는 CPU/unit tier에서 signature/order/group state machine을, single-node GPU tier에서 stream/event와 local transport를, multi-node tier에서 proxy/network async error를 검사한다. unit test 통과를 실제 network path 증거로 과장하지 않는다.

canonical fixture를 byte 단계로 다시 펼친다. producer가 rank별 BF16 16 elements를 쓰고 ready event를 기록한다. collective stream은 event를 기다린 뒤 info를 sequence 42로 enqueue한다. group을 사용하면 모든 ranks의 sequence 42 task가 plan에 들어간다. device/proxy work와 stream completion 뒤 consumer가 16개 값 10을 읽는다. 어느 단계든 `(G7,42)`가 유지돼야 한다.

실제 장애 교대 기록을 예로 든다. 02:14에 TP group G7의 request latency가 30초 watchdog을 넘었다. rank 0 stack은 stream wait, rank 1은 다음 layer enqueue, rank 2 log에는 transport error, rank 3은 proxy progress 대기다. symptom rank를 0으로 적지 않고 네 rank의 current sequence를 먼저 모은다. 모두 sequence 42라면 call order는 아직 후보지만 43으로 앞선 rank가 있으면 sequence divergence가 먼저다.

postmortem에는 `first_failure=rank2/seq42/transport`, `first_observation_delay`, `derivative_wait_ranks`, `cleanup_generation`을 적는다. startup environment dump 수백 줄보다 이 causal tuple이 중요하다. 원본 logs/trace hash와 manifest를 연결해 source evidence를 재검증할 수 있게 한다.

반대 사건도 본다. rank 2 async error는 없고 all ranks device/proxy progress가 완료됐지만 rank 0 consumer output만 777이다. collective root cause 가설은 약하다. rank 0 AR_done wait가 consumer stream에 없고 rank 1~3에는 있다면 first divergence는 local ordering이다. communicator abort/recreate는 불필요하며 event graph를 고친다.

또 다른 사건은 all ranks output이 9다. signature는 같고 producer checkpoint에서 rank 3 input이 3으로 stale했다면 NCCL은 입력을 정확히 합쳤다. producer→collective event가 누락된 것이다. all-reduce output만 보면 reduction arithmetic bug처럼 보이지만 pre-collective checkpoint가 owner를 가른다.

운영자가 보존할 reduced fixture는 네 rank와 두 sequences면 충분하다. sequence 0은 정상 16-element sum이고 sequence 1에는 한 번에 한 divergence만 넣는다. count 15, A/B order swap, producer event 누락, consumer wait 누락, proxy failure와 teardown injection을 각각 독립 case로 만든다. 여러 오류를 한 fixture에 넣으면 first divergence를 잃는다.

각 case는 communicator manifest, 네 signature rows, selected plan, progress timeline, output/failure와 cleanup state를 같은 schema로 출력한다. 정상 case와 field 단위 diff가 가능해야 한다. source 내부 모든 struct를 dump하지 않아도 owner transition을 재구성할 수 있다. pointer와 unique ID는 generation/hash로 치환한다.

performance fixture는 correctness fixture를 대체하지 않는다. 32 MiB throughput threshold를 통과해도 16-element cross-stream fixture가 stale이면 배포를 막는다. 반대로 작은 fixture latency가 높아도 launch overhead 지배이므로 large bandwidth failure라고 부르지 않는다.

rank 2 failure fixture는 rank 0 timeout으로만 보지 않게 모든 rank의 async observation과 coordinator propagation을 검사한다. new admission이 닫히고 futures가 failure로 끝나며 buffers/communicator가 bounded cleanup 또는 force-termination owner로 넘어가야 한다.

shutdown fixture는 success completion과 abort completion 두 종료를 가진다. 어느 쪽도 없이 registry에서 comm을 지우면 안 된다. proxy thread, registration과 graph cache가 새 generation에 old reference를 남기지 않는지 확인한다.

이 reduced suite를 NCCL·driver·framework upgrade마다 실행하면 selected algorithm이 바뀌어도 semantic contract를 비교할 수 있다. performance 변화는 plan diff와 함께 검토하고 signature·ordering·cleanup invariant는 유지한다. current source 분석을 오래 쓰이는 운영 지식으로 바꾸는 방법이다.

최종 배포 승인표에는 rank별 comm generation과 device UUID, smoke sequence 결과, effective plan hash, cross-stream readiness, async polling과 abort fixture 결과를 남긴다. 한 rank라도 다른 generation/config를 보거나 smoke completion을 증명하지 못하면 group 전체를 ready로 만들지 않는다. 성능 baseline은 같은 message histogram과 overlap workload에서 비교한다.

운영 중 manifest가 바뀌면 communicator hot mutation으로 처리하지 않고 지원되는 lifecycle을 따른다. topology/NIC/environment 변경이 새 communicator를 요구하는지 확인한다. old requests를 drain하고 new generation을 만들며 routing switch 뒤 old resources를 정리한다. 세대가 겹치는 동안 metrics와 traces가 어느 comm을 가리키는지 명확해야 한다.

마지막으로 정상 output 10을 얻었다고 cleanup이 증명된 것은 아니다. buffer reuse, next sequence와 shutdown까지 fixture를 이어 old work가 남지 않았음을 확인한다. 반대로 cleanup이 깔끔해도 signature/result가 틀리면 성공이 아니다. execution과 lifetime의 두 축을 모두 닫아야 collective 한 번이 끝난다.

fixture manifest는 다음처럼 쓴다.

rank별 manifest hash를 out-of-band control plane으로 비교할 수 있다. exact unique ID byte를 log에 노출하지 않고 hash와 generation을 기록한다. CUDA ordinal은 container마다 달라질 수 있으므로 device UUID/BDF를 함께 둔다. communicator rank와 application TP rank가 같은 순서인지도 확인한다.

rank-device mapping은 duplicate GPU도 검사한다. standard TP fixture에서는 device UUID uniqueness를 invariant로 둔다. communicator split로 subgroup을 만들면 parent generation, color/key와 child rank order를 manifest에 추가한다. child communicator가 parent collective sequence를 공유한다고 가정하지 않는다.

증상은 hang 또는 wrong output이다. rank별 manifest는 communicator G7로 같다. sequence 42 signature에서 rank 2만 count 15다. first divergence가 API/info 이전 application shape라면 NCCL transport는 반증된다. scheduler의 tensor partition/count 계산을 고친다.

재발 fixture는 element count 16, zero count와 tail shard를 모든 rank에 canonical signature로 배포한다. rank 하나를 의도적으로 15로 바꾸면 pre-enqueue debug guard 또는 deterministic failure가 발생해야 한다. production overhead 때문에 guard를 끄더라도 scheduler invariant test는 남긴다.

TP rank 하나가 다른 rows를 가지면 count mismatch다. scheduler가 모든 ranks에 동일 active row order와 count를 배포해야 한다. finished request removal 시점이 다르면 NCCL timeout 이전 model runner state가 갈린다.

환경 변수를 영구 recipe로 남기기 전에 왜 auto selection이 잘못됐는지 version/hardware 범위에서 증명한다. framework upgrade나 topology 변경 때 override를 재검증한다. environment가 process/rank마다 다르면 communicator plan divergence 위험이 있으므로 effective config hash를 init manifest에 넣는다.

release upgrade 뒤 selection이 달라지면 cost model/config/default 변화 후보다. old override로 되돌리기 전에 current source와 effective plan을 비교한다. correctness가 같다면 performance regression으로 다룬다.

upgrade gate는 NCCL commit, CUDA/driver와 binding을 묶는다. packaged library가 expected commit인지 binary inventory로 확인한다. environment override와 transport plugin version도 manifest에 둔다.

두 시간 조사의 종료 조건은 identity/signature, selected plan/progress, output event ordering, error/cleanup generation 네 가지다. 이 조건 뒤에만 algorithm override와 topology tuning을 평가한다.

network interface environment는 communicator transport 입력이다. 변수 사전을 나열하지 않고 effective NIC, BDF/NUMA, transport와 proxy thread를 manifest에 둔다. 설정과 actual interface가 다르면 path selection이 first divergence다.

deployment smoke는 rank-device identity, 16-element sum, cross-stream events와 group A/B를 검사한다. multi-node면 NIC/transport와 large bandwidth fixture를 추가한다. smoke 통과를 congestion 부재로 과장하지 않는다.

최종 regression은 signature, result, ordering, error propagation, cleanup과 performance를 분리한다. 앞 다섯 실패는 배포를 막는다. 느리지만 올바른 fallback과 빠르지만 stale output을 같은 성공으로 보지 않는다.

consumer가 같은 current stream에 있어 implicit ordering이 성립하는 path를 다른 stream으로 리팩터링할 때 regression이 생기기 쉽다. stream topology 변경을 code review checklist에 넣는다. producer/NCCL/consumer stream, record/wait events와 buffer record-stream lifetime을 그림으로 첨부한다.

one-rank error 전파에서는 new work admission을 먼저 막는다. 이미 failed communicator에 sequence 43을 append하면 logs와 cleanup이 더 복잡해진다. rank별 last accepted/completed sequence를 freeze하고 group controller가 공통 failure generation을 배포한다.

cleanup에서는 rank 0/1/3이 stream wait 중이고 rank 2가 error state다. coordinator가 new admission을 막고 G7 failure generation을 배포한다. request futures를 fail하고 all ranks single owner가 abort로 들어간다. rank 하나가 graceful destroy를, 다른 rank가 abort를 선택하지 않게 한다. bounded timeout 뒤 process group replacement를 수행한다.

application payload는 rank당 32 bytes다. ring/tree algorithm이 chunk를 나누고 send/recv하며 protocol metadata를 붙이면 link transaction은 다르다. channel padding과 alignment도 있을 수 있다. 작은 32-byte fixture는 overhead 지배라 bandwidth benchmark가 아니라 signature/ordering correctness에 적합하다.

성능 fixture는 같은 semantic을 더 큰 M×N으로 확장하되 element count와 payload를 다시 계산한다. `algbw`, `busbw`, physical link throughput과 critical-path latency 정의를 명시한다. 공식/tool 정의 없이 bus bandwidth 공식을 임의로 만들지 않는다.

32-byte fixture는 launch/protocol overhead가 payload보다 크다. correctness와 task order를 보기에 좋다. 32 MiB fixture는 chunk/channel과 link utilization을 관찰하기 좋다. 한 size에서 선택된 protocol을 모든 serving tensor에 일반화하지 않는다. TP collective sizes는 layer/batch/prefill-decode에 따라 달라진다.

channel 수와 SM contention 실험에는 compute-bound와 memory-bound overlap fixture를 각각 둔다. 전자는 SM 경쟁을, 후자는 HBM 경쟁도 포함한다. collective 단독 최적 channel이 serving critical path 최적과 다를 수 있다.

protocol의 low-latency 이름을 모든 size에서 더 빠르다는 뜻으로 읽지 않는다. protocol별 payload representation, step, buffer와 capability trade-off가 있다. output semantics는 같아야 하므로 performance run에도 checksum fixture를 둔다.

32-byte fixture의 link bytes는 protocol header/steps 때문에 useful 32 bytes보다 훨씬 클 수 있다. 이를 inefficiency benchmark로 쓰지 않는다. 작은 fixture의 목적은 sequence, signature와 ordering이다. throughput은 큰 fixture에서 본다.

성능 fixture로 32 MiB rank payload를 쓰면 BF16 elements는 16,777,216개다. wall latency 4 ms라면 rank application payload/latency는 약 8 GiB/s다. NCCL tool busbw가 다르면 normalization 정의를 확인한다. physical line rate와 직접 같다고 비교하지 않는다.

따라서 NCCL 장애의 첫 질문은 “ring인가 tree인가”가 아니다. 네 rank가 같은 communicator generation에서 같은 sequence와 signature를 enqueue했고, 같은 plan의 work가 어느 device/proxy 단계까지 진행했으며, consumer가 어떤 completion을 기다렸는가이다. 이 원장이 닫힌 뒤에야 algorithm override와 topology tuning이 의미를 가진다. 마지막 buffer lease와 proxy resource가 새 generation에 old reference를 남기지 않아야 collective lifecycle도 비로소 완결된다.

각 channel이 payload의 어느 chunk와 peer edge를 담당하는지 원장에 둔다. exact internal struct field는 v2.30.7-1 source 범위에서만 설명한다. application tensor row가 channel object에 영구 귀속된다고 생각하지 않는다. plan은 이번 call의 work partition이다.

32-byte fixture는 descriptor와 launch overhead가 지배한다. 32 MiB fixture에서 rank당 useful input/output, algorithm/link transaction과 HBM read/write를 별도 열로 둔다. bus bandwidth conversion은 NCCL test/tool의 정의를 그대로 인용하고 application useful bandwidth와 혼동하지 않는다.

새 collective 장애를 만나면 algorithm environment부터 바꾸지 않는다. communicator identity, sequence/signature, task/plan, device/proxy progress, stream completion, async error와 cleanup 순서로 좁힌다. 같은 BF16 sum fixture를 유지하면 경계마다 expected value와 owner가 명확하다.

40~70분에는 group tasks, selected algorithm/protocol/channels와 transport edge를 확인한다. 70~90분에는 device/proxy last progress와 CUDA stream event graph를 잇는다. 마지막 30분에는 async error 관찰 delay, cleanup owner와 retry generation을 닫고 small fixture를 만든다.

teardown fixture는 enqueue 전, group end 직후, proxy progress 중, stream completion 직후를 주입한다. output publish, buffer lease, proxy thread와 communicator reference가 정확히 닫히는지 본다. double abort/destroy와 old graph replay도 검사한다.

sentinel fixture는 producer가 generation `g`마다 `r+1+100g`를 쓴다. collective output이 이전 generation sum이면 communication algorithm보다 local stream dependency가 첫 후보다. producer output checksum을 event 전후에 비교한다.

fixture는 consumer stream priority와 launch timing을 뒤집어 race를 드러낸다. event pool generation 재사용과 graph replay도 포함한다.

stale consumer fixture는 output을 이전 값 777로 채운다. event 없이 consumer가 777을 읽으면 first divergence가 명확하다. cleanup fixture는 freed buffer를 999로 채워 old work use-after-free를 드러낸다. isolated lab에서 수행한다.

fixture는 shutdown 시점을 네 단계에 주입하고 double abort/destroy를 idempotent하게 처리한다. 다음 process generation이 old unique ID/cache를 재사용하지 않는지 확인한다.

shutdown은 admission stop→drain→bounded abort→comm resource→buffer/context 순서다. signal handler에서 안전성이 확인되지 않은 API를 직접 부르지 않는다. cleanup fixture는 네 progress 시점과 double abort를 검사한다.

이 종료 원장은 happy path만을 위한 것이 아니다. async error가 있으면 E42가 success ready가 아니라 failure future로 닫히고 buffer는 publish되지 않는다. abort cleanup completion이 재사용 권한을 allocator/process manager에 넘긴다. “timeout이라 포기했다”는 상태는 ownership 종료가 아니다.

### 회고
