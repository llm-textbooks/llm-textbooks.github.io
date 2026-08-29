# 43장. 함수가 돌아온 뒤에도 끝나지 않은 일

KV cache를 다른 buffer로 복사하는 Python 함수가 정상적으로 반환했다. 바로 다음 줄에서 attention을 호출했다. 대부분의 요청은 맞았지만 부하가 높을 때만 첫 token이 달라졌다. 조사자는 “copy 함수가 먼저 호출됐으니 attention은 새 KV를 읽었을 것”이라고 생각했다. 이 문장은 host 프로그램의 호출 순서는 설명하지만 GPU가 작업을 완료한 순서는 증명하지 않는다.

두 작업이 같은 CUDA stream에 enqueue됐다면 stream 순서가 copy 뒤 attention이라는 의존성을 만든다. copy stream과 compute stream이 다르면 host가 copy를 먼저 enqueue했다는 사실만으로 두 device 작업 사이의 선후관계는 생기지 않는다. producer stream이 copy 뒤 event를 record하고 consumer stream이 그 event를 wait해야 “attention은 copy가 도달한 뒤 시작한다”는 edge가 생긴다. buffer를 누가 언제까지 살려 두는가도 별 계약이다.

두 번째 사건은 CUDA Graph replay에서 시작한다. batch capacity 8로 capture한 graph에 active requests 5개를 넣었다. rows 0–4의 metadata는 새 값으로 갱신했지만 rows 5–7에는 이전 replay의 sequence lengths와 KV locations가 남았다. graph launch 자체는 성공했다. captured node의 주소도 변하지 않았다. 그러나 kernel이 active length를 8로 읽거나 padding predicate가 빠지면 이전 요청의 state를 소비할 수 있다. 주소 안정성은 값의 최신성을 보장하지 않는다.

이 장은 비동기를 “동시에 실행한다”는 분위기 좋은 말로 설명하지 않는다. producer, produced range, consumer, required scope, ordering primitive와 lifetime owner를 한 줄로 연결한다. stream은 순서가 있는 device work queue로 이해하되 CPU thread나 전용 GPU 실행 자원이라고 동일시하지 않는다. event는 host timestamp가 아니라 stream 사이에 happens-before edge를 세우는 device-side ordering point로 읽는다. graph는 kernel fusion이 아니라 이미 구성한 node와 dependency를 낮은 제출 비용으로 replay하는 실행 객체로 읽는다.

공식 계약은 NVIDIA CUDA C++ Programming Guide 12.9.1과 CUDA Programming Guide 13.3.0의 고정 archive를 사용한다. 구현은 vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, llama.cpp v0.2.0 commit `bb4caa7540188872173c44d161602d9271386413`에서 읽는다. 실제 CUDA 작업이나 model을 실행하지 않으며 성능 수치를 만들지 않는다.

## 43.1 host return·enqueue·completion·visibility·lifetime을 나눈다

### 43.1.1 함수 반환은 무엇을 증명하는가

비동기 CUDA API를 호출한 host 함수는 device 작업이 끝나기 전에 돌아올 수 있다. 반환은 대개 launch 또는 copy가 제출됐고 즉시 발견 가능한 오류가 없었다는 정도를 말한다. kernel output이 consumer에게 준비됐다는 뜻도, host가 그 memory를 재사용해도 된다는 뜻도 아니다. asynchronous error가 뒤 API나 명시적 synchronization에서 보고될 수도 있다.

enqueue는 특정 stream의 순서열에 작업을 놓는 사건이다. completion은 device가 그 작업을 끝낸 사건이다. visibility는 consumer가 producer의 writes를 올바른 ordering와 scope 아래 관찰할 수 있다는 조건이다. lifetime은 작업이 참조하는 storage와 metadata가 completion까지 유효하다는 소유권 조건이다. 네 사건을 같은 “끝남”으로 부르면 use-before-ready와 use-after-free를 구분할 수 없다.

예를 들어 host가 pinned memory에서 device metadata로 asynchronous copy를 enqueue하고 Python local object를 놓았다고 하자. copy enqueue가 성공했어도 source pinned storage가 copy completion 전에 재사용되면 값이 바뀔 수 있다. device destination가 준비돼도 compute stream이 wait하지 않으면 attention이 먼저 읽을 수 있다. compute가 끝나기 전에 destination buffer를 pool에 반환하면 다른 request가 덮어쓸 수 있다. ordering와 lifetime는 함께 닫혀야 한다.

### 43.1.2 stream은 줄이지만 전용 worker thread는 아니다

stream을 식당 주문표나 단일 차선에 비유할 수 있다. 한 stream에 들어간 작업은 정의된 stream ordering를 따른다. 앞 작업 뒤에 enqueue된 작업은 앞 dependency가 충족된 뒤 진행한다. 그러나 stream마다 전용 SM이 하나씩 배정되거나 서로 다른 streams가 반드시 동시에 실행된다는 뜻은 아니다. 실제 overlap은 hardware resources, dependencies와 implementation에 달렸다.

default stream semantics도 무심코 가정하지 않는다. legacy default stream과 per-thread default stream, library가 전달받은 explicit stream에 따라 암묵적 ordering 기대가 달라질 수 있다. production code에서는 “기본 stream이 알아서 기다릴 것”보다 producer/consumer가 사용하는 실제 stream handle과 explicit edge를 확인한다.

같은 stream은 강력한 기본 해법이다. copy와 attention을 같은 stream에 놓으면 별 event 없이 순서가 닫힌다. 하지만 모든 일을 한 stream에 직렬화하면 overlap opportunity를 잃을 수 있다. 여러 streams를 쓰려면 빨라질 가능성과 함께 cross-stream dependency를 명시해야 하는 책임이 생긴다.

### 43.1.3 synchronization은 device-wide stop의 별명이 아니다

동기화에는 범위가 있다. host가 device 전체를 기다리는 synchronization, 특정 stream completion을 기다리는 synchronization, consumer stream이 event를 기다리는 device-side dependency, CTA 내부 threads가 barrier에서 만나는 synchronization는 서로 다르다. correctness에 필요한 가장 좁은 scope를 선택해야 unrelated work를 막지 않는다.

whole-device synchronize를 넣어 오답이 사라지면 ordering bug 가설은 강해지지만 그것이 최종 수정은 아니다. 모든 streams를 멈춰 우연히 필요한 edge도 만들기 때문이다. producer buffer range와 consumer를 찾은 뒤 producer stream의 event와 consumer wait 또는 같은-stream ordering로 scope를 좁힌다.

과동기화는 correctness를 보존하면서 성능을 해칠 수 있다. scheduler metadata 준비와 이전 forward read 사이에 필요한 WAR edge 하나 대신 매 step device synchronize를 쓰면 CPU scheduling, copy와 compute overlap이 사라질 수 있다. 반대로 synchronization를 제거해 benchmark가 좋아져도 stale read가 가능해지면 성공이 아니다.

**시간 원장을 만든다.**

시간 원장의 각 행은 operation ID, stream, input ranges와 generations, output ranges, enqueue host timestamp, prerequisite event generations, completion evidence, lifetime owner를 가진다. timestamp만으로 device ordering를 만들지 않고 event/stream dependency를 edge로 기록한다.

KV fixture에서는 producer가 `copy K,V generation 17`이고 destination range는 physical blocks 20–23이라고 하자. consumer는 attention launch A이며 같은 blocks generation 17을 읽어야 한다. required edge는 copy completion point E17에서 A stream wait까지다. owner는 E17과 A completion가 닫힐 때까지 destination blocks를 재사용하지 않는다.

graph fixture에서는 metadata write W42, graph launch G42와 static buffer generation S3를 기록한다. 주소 S3는 capture 때부터 같지만 active rows values generation은 replay마다 42로 바뀐다. G42는 W42 뒤에 있어야 하고 rows 5–7은 invalid generation/sentinel여야 한다. address generation와 value generation를 분리해야 stale metadata를 볼 수 있다.

## 43.2 fixture A: 두 stream에서 KV를 안전하게 넘긴다

### 43.2.1 같은 stream은 가장 짧은 이야기다

`S_compute` 하나에 metadata copy, KV copy와 attention을 차례로 enqueue했다고 하자. stream ordering가 producer→consumer를 닫는다. host가 copy call에서 즉시 반환해도 attention은 같은 stream의 앞선 copy 뒤에 있다. 별 event를 만들 필요는 없다.

이 장면의 직관은 한 줄에 놓인 작업은 앞뒤가 있다는 것이다. 한계는 host return와 device completion를 여전히 구분해야 한다는 점이다. attention 뒤 output을 CPU가 즉시 읽으려면 host-side completion를 기다려야 한다. 같은-stream ordering는 device operations 사이 edge이지 host가 자동으로 기다린다는 뜻이 아니다.

buffer lifetime도 남는다. copy destination KV는 attention이 끝날 때까지 유효해야 하고 source host memory는 copy가 끝날 때까지 재사용하지 않는다. allocator나 block manager가 request cancellation에서 storage를 너무 일찍 반환하면 stream 순서가 맞아도 use-after-free가 난다.

### 43.2.2 다른 stream에는 event edge를 놓는다

copy를 `S_copy`, attention을 `S_compute`에 놓아 overlap을 얻고 싶다고 하자. 올바른 timeline은 producer stream에 copy를 enqueue하고 같은 stream에서 event E를 record한 뒤 consumer stream에 `wait(E)`를 enqueue하고 attention을 놓는다. event는 producer stream이 앞의 copy까지 도달했음을 나타내고 wait는 consumer의 뒤 작업을 그 지점 뒤로 미룬다.

중요한 순서는 `copy(S_copy) → record E(S_copy) → wait E(S_compute) → attention(S_compute)`다. event를 copy 전에 record하면 너무 이른 지점을 가리킨다. wait를 attention 뒤에 enqueue하면 consumer를 보호하지 못한다. event를 잘못된 stream에 record하면 host 호출 순서만 다시 적은 셈이 될 수 있다.

event timing 기능과 dependency 기능도 나눈다. elapsed time을 재려고 timing-enabled events를 둘 수 있지만 dependency만 필요하다면 timing 자체가 목적이 아니다. “event를 기록했다”는 로그에 host timestamp만 남기지 말고 event identity/generation, record stream과 wait stream, 보호하는 buffer range를 남긴다.

**wait가 없으면 왜 간헐적인가.**

host는 copy를 먼저 enqueue하고 attention을 나중에 enqueue했으므로 로그는 항상 정상 순서처럼 보인다. 하지만 streams 사이에 edge가 없으면 GPU scheduler는 compute stream의 attention을 copy completion 전에 진행할 수 있다. copy가 작은 날에는 우연히 먼저 끝나고, 부하와 transfer size가 바뀐 날에만 race가 드러날 수 있다.

stale read는 항상 crash하지 않는다. destination에 이전 request의 valid-looking KV와 lengths가 남아 있으면 attention은 memory-safe하게 틀린 state를 읽는다. 첫 output token이 달라질 수 있고 이후 generated KV도 달라져 divergence가 증폭된다. final text보다 first attention output 또는 first bad row를 찾는다.

partial copy도 가능하다. K range는 준비됐지만 V tail이나 metadata length가 아직이면 일부 heads/positions만 틀릴 수 있다. producer record에는 하나의 “KV ready” 대신 실제 ranges와 copy operations를 쓴다. 여러 copies 뒤 하나의 event를 record한다면 그 event가 모두 같은 producer stream에서 앞선 작업 뒤에 있는지 확인한다.

### 43.2.3 fixture A 손검산

정상 같은-stream 경우의 producer는 copy, consumer는 attention, required scope는 stream order, primitive는 동일 `S_compute`, lifetime owner는 copy/attention completion를 추적하는 request 또는 cache block owner다. first divergence는 attention input checkpoint가 destination generation 17과 다를 때다.

정상 cross-stream 경우에는 producer `S_copy`, event E17, consumer `S_compute`가 있다. E17 record는 copy 뒤, wait는 attention 앞이다. source와 destination pointer가 맞고 block generation도 17이어야 한다. first divergence가 wait 전 attention launch라면 ordering edge가 빠졌다.

누락 경우에는 host enqueue sequence만 있고 device edge가 없다. host timestamps가 copy call→attention call이어도 반증되지 않는다. whole-device synchronize를 copy 뒤 임시로 넣어 증상이 사라질 수 있지만 정확한 검증은 E17 edge를 복구하고 unrelated stream work를 막지 않은 상태에서 correctness를 확인하는 것이다.

### 43.2.4 event 재사용에도 generation가 있다

event object 하나를 여러 steps에서 재사용할 수 있어도 관측에서는 generation를 나눈다. step 42가 E를 record했고 consumer가 기다리는 동안 step 43이 같은 event를 다시 record하는 방식의 의미는 API와 ordering를 정확히 따라야 한다. 단순 event pointer equality로 어느 producer generation을 기다렸는지 판정하지 않는다.

pool에서 events를 빌리고 반환한다면 마지막 wait가 안전하게 enqueue되거나 완료되기 전에 다른 의미로 재사용하지 않는다. event lifetime도 buffer lifetime처럼 owner를 가진다. deadlock 조사에서는 A stream이 E_B를 기다리고 B stream이 E_A를 기다리는 circular wait가 생기지 않았는지 dependency graph를 그린다.

## 43.3 block barrier와 async-copy barrier는 stream event가 아니다

### 43.3.1 `__syncthreads()`의 세계는 CTA 안이다

`__syncthreads()`는 같은 thread block의 participating threads를 barrier에서 만나게 하고 그 scope의 shared/global memory ordering 의미를 제공한다. 다른 CTA, 다른 kernel 또는 다른 stream을 동기화하는 API가 아니다. CTA 0이 shared tile을 채웠다고 CTA 1이 `__syncthreads()`로 그것을 기다릴 수 없다.

barrier가 conditional branch 안에 있고 block의 일부 threads만 도달하면 hang 또는 undefined behavior 위험이 생긴다. predicate가 warp마다 다를 수 있는 tail path에서 모든 필요한 participants가 같은 barrier protocol을 따르는지 본다. “barrier가 있으니 안전”이 아니라 누가 arrive하고 누가 wait하는지 센다.

shared memory producer/consumer에서는 thread들이 load를 마친 뒤 compute 전에 block barrier를 둘 수 있다. 그러나 global async copy engine와 pipeline를 쓰면 단순 barrier만으로 completion 의미가 충분한지 API 계약을 읽어야 한다. arrival count와 transaction completion를 포함하는 `cuda::barrier` 또는 `cuda::pipeline` protocol가 사용될 수 있다.

### 43.3.2 async-copy pipeline은 stage lifetime를 관리한다

global→shared asynchronous copy를 여러 stages로 겹치면 producer가 stage N을 채우고 consumer가 준비된 stage를 읽는다. consumer가 wait 전에 읽으면 use-before-ready이고 producer가 consumer 종료 전에 같은 shared stage를 덮어쓰면 WAR hazard다. stage acquire/commit/wait/release의 pairing이 필요하다.

`cuda::barrier`의 expected arrival count는 participating threads와 transaction protocol에 맞아야 한다. 일부 thread가 early return하면 남은 threads가 영원히 wait할 수 있다. tail tile에서도 barrier participation와 data predicate를 분리할 수 있다. invalid lanes가 data load는 하지 않아도 barrier에는 도달해야 할 수 있다.

이 scope는 fixture A의 stream event보다 작다. async-copy barrier는 한 kernel/CTA 또는 정의된 thread scope 안의 stage readiness를 다룬다. stream event는 서로 다른 kernel/copy operations 사이 dependency를 다룬다. 하나를 다른 것으로 대체하지 않는다.

### 43.3.3 세 primitive를 질문으로 구별한다

“같은 CTA의 threads가 shared tile 작성 뒤 함께 읽는가?”라면 block barrier를 본다. “같은 kernel의 async copy stage가 준비됐는가?”라면 barrier/pipeline completion를 본다. “copy stream의 operation 뒤 compute stream kernel이 시작해야 하는가?”라면 event record/wait를 본다. “host가 결과를 읽기 전에 device가 끝났는가?”라면 host-facing completion primitive를 본다.

scope가 너무 좁으면 correctness가 깨지고 너무 넓으면 불필요한 serialization가 생긴다. 정확성 수정은 필요한 producer/consumer edge를 먼저 증명한 뒤 좁은 primitive를 고른다. performance 최적화는 edge를 삭제하는 것이 아니라 independent work를 edge 밖으로 옮기는 방식으로 한다.

## 43.4 fixture B: bucket 8 graph에 active batch 5를 replay한다

### 43.4.1 capture는 미래 실행의 골격을 고정한다

CUDA Graph는 operations와 dependencies를 graph로 구성하고 executable graph를 instantiate한 뒤 stream에 launch할 수 있게 한다. stream capture는 capture 구간에 enqueue된 work를 graph nodes와 edges로 기록하는 한 방식이다. replay는 이 topology를 다시 제출한다. node kernels가 자동으로 합쳐지거나 계산량이 사라지는 것은 아니다.

capture는 흔히 static addresses와 shape buckets를 요구한다. node parameter가 capture 때의 pointers와 dimensions를 갖기 때문이다. replay마다 새 값은 같은 buffers에 복사할 수 있지만 address와 capacity contract를 유지해야 한다. dynamic control flow가 topology를 바꾸면 다른 graph key, exec update 또는 eager fallback가 필요할 수 있다.

instantiate는 captured graph definition에서 launch 가능한 executable을 만든다. upload는 필요에 따라 준비 비용을 앞당길 수 있고 launch는 executable을 stream에 제출한다. graph exec update가 허용하는 변경 범위 안이면 기존 executable을 갱신할 수 있지만 topology/parameter 변경이 호환되지 않으면 재instantiate해야 한다.

### 43.4.2 capacity 8과 active 5는 서로 다른 state다

capture capacity `B_cap=8`, replay active batch `B_act=5`라고 하자. static input/output/KV metadata buffers는 여덟 rows를 담는다. replay 전에 rows 0–4를 현재 requests의 값으로 갱신한다. rows 5–7에는 invalid sentinel 또는 zero를 채우고 kernel이 active length/predicate로 무효화해야 한다.

padding row fraction은 `(8-5)/8=37.5%`다. 이는 logical capacity의 빈 비율이지 GPU 계산이 정확히 37.5% 낭비됐다는 측정값이 아니다. kernel이 invalid rows를 초기에 return하는지, fixed tile에서 함께 진행하는지, KV lengths와 heads가 어떤지를 봐야 한다. graph가 줄이는 host submission 비용과 padded device work도 서로 다른 항이다.

static address가 유지돼도 values는 replay generation마다 달라진다. metadata writer가 copy stream에 있고 graph launch가 compute stream에 있다면 fixture A와 같은 event edge가 필요하다. graph object가 dependency를 캡처했다고 생각하기 전에 metadata update가 graph 안 node인지 graph 밖 operation인지, capture/replay stream과 어떤 관계인지 확인한다.

### 43.4.3 stale B_act=8은 이전 요청을 되살린다

이전 replay가 active 8이었고 현재 active 5인데 active length scalar가 8로 남았다고 하자. rows 5–7 buffer addresses는 유효하고 값도 이전 세 requests의 valid metadata일 수 있다. kernel은 memory fault 없이 이 rows를 처리한다. output slicing이 첫 다섯 rows만 반환하더라도 shared KV/cache writes나 reductions가 섞이면 correctness와 resource state를 오염시킬 수 있다.

first divergence는 final token이 아니라 metadata generation다. replay G43의 active scalar가 generation 42인지, rows 5–7 sentinel가 generation 43으로 invalidated됐는지 본다. attention metadata가 separate buffers라면 sequence lengths, slot mapping, block table와 request IDs를 함께 검사한다. 하나만 갱신되고 다른 하나가 stale일 수 있다.

padding을 zeros로 채우는 것만으로 충분한지도 source가 정한다. slot mapping에서 `-1`이 invalid sentinel일 수 있고 sequence length에는 backend별 fill value가 필요할 수 있다. zero가 valid physical slot 또는 valid zero-length semantics라면 잘못된 sentinel다. runner와 attention backend가 합의한 predicate를 읽는다.

### 43.4.4 buffer 재할당은 값이 맞아도 위험하다

capture 때 input pointer P0를 node가 참조했는데 replay 전에 tensor resize/reallocation로 current input이 P1이 됐다고 하자. host는 P1에 최신 rows를 썼지만 graph는 여전히 P0를 읽을 수 있다. P0가 해제돼 다른 allocation에 쓰이면 stale read나 corruption가 난다. Python tensor variable 이름이 같다는 사실은 device pointer stability를 증명하지 않는다.

static buffer owner는 capture graph executable 수명 동안 storage를 유지한다. replay input는 current dynamic tensor에서 static buffer slice로 copy하고 output도 static storage에서 active slice만 소비할 수 있다. debug mode에서 input `data_ptr` consistency를 검사하는 구현도 있지만 production correctness를 debug log에만 의존하지 않는다.

graph exec update가 pointer parameter 변경을 지원하는 경로라면 update 결과를 확인한다. 성공했다고 가정하지 않고 result와 error node를 기록한다. update 불가면 old exec를 파기하고 새 graph를 instantiate하는 lifetime transition가 필요하다. old exec가 in flight일 때 storage를 회수하지 않는다.

**batch 9는 bucket 8에 넣을 수 없다.**

`B_act=9`는 capacity 8을 넘으므로 bucket 8 replay가 불가능하다. 더 큰 captured bucket을 선택하거나 eager path로 fallback해야 한다. active length를 8로 clamp하면 한 request를 조용히 누락하고, buffers를 9로 resize하면 captured pointer/capacity contract를 깨뜨린다.

bucket selection predicate는 requested shape를 정규화해 smallest fitting captured size를 찾는다. inventory가 `[1,2,4,8,16]`이면 active 5는 8, active 9는 16을 선택할 수 있다. exact 정책은 implementation source를 따른다. no-padding backend는 exact key만 허용할 수도 있다.

graph hit rate가 낮다면 graph 시스템이 느리다고 결론 내리기 전에 requested shape histogram, normalized key, captured inventory와 rejection reason을 본다. speculative width, LoRA variant, encoder condition나 backend support가 key를 더 세분할 수 있다. batch size 하나만 맞아도 graph가 거부될 수 있다.

**fixture B 손검산 원장.**

정상 replay G43은 key capacity 8, active 5, static buffer allocation generation S3를 사용한다. metadata write W43은 rows 0–4를 current generations로 쓰고 rows 5–7을 invalidates한다. W43→G43 edge가 있고 kernels는 active/padding predicate를 쓴다. output consumer는 active slice 0–4만 받는다.

stale-active 사건은 address S3와 graph key 8이 맞지만 `B_act` value generation가 42다. address 검사만 통과한다. stale-padding 사건은 scalar 5가 맞아도 rows 5–7의 slot mapping가 이전 값이다. predicate가 scalar만 쓰면 안전할 수 있지만 backend가 sentinel를 읽는다면 위험하다. exact consumer를 확인한다.

reallocation 사건은 writer destination P1과 captured node pointer P0가 다르다. event를 아무리 정확히 연결해도 graph는 잘못된 address를 읽는다. ordering와 address identity를 별 검증한다. batch 9 사건은 predicate 단계에서 bucket 8을 거부해야 하며 first divergence가 buffer copy까지 내려가면 늦다.

운영 로그에 “graph 실행” 하나만 있으면 cold capture, instantiate와 warm replay를 구분할 수 없다. capture는 실제 operations를 기록하는 단계라 warmup나 lazy initialization가 선행될 수 있다. instantiate는 graph definition를 executable로 바꾸는 단계이고 실패할 수 있다. replay는 이미 준비된 executable을 stream에 launch하는 단계다.

첫 shape가 느리고 이후 빨라지는 현상은 capture/instantiate 비용일 수 있지만 source와 trace 없이 수치로 단정하지 않는다. 반대로 매번 느리다면 key가 매번 달라 새 capture가 생기거나 update 실패로 reinstantiate할 수 있다. lifecycle generation를 기록하면 cold와 pathological churn을 나눌 수 있다.

capture 자체가 model result를 사용자에게 반환하는 정상 replay인지도 구현마다 다르다. dummy inputs로 warmup/capture를 수행할 수 있고 그 output은 버릴 수 있다. capture forward가 KV cache 같은 external state를 쓰면 dummy locations와 reset가 정확해야 한다. SGLang runner가 index buffers를 reset하는 이유를 이 위험과 연결해 읽는다.

instantiate가 성공해도 첫 replay 전에 buffers가 current request values를 가졌다는 뜻은 아니다. executable readiness와 input readiness를 분리한다. graph upload가 있어도 input metadata write ordering를 대신하지 않는다. 준비 단계를 하나의 “graph ready” boolean으로 합치지 않는다.

graph exec update는 일부 node parameters나 topology 변화를 기존 executable에 반영할 수 있지만 모든 변경을 허용한다는 뜻은 아니다. API가 update result와 error node를 제공한다면 호출자는 결과를 분기해야 한다. 실패했는데 old executable을 그대로 launch하면 stale topology/parameters를 쓸 수 있다.

fixture B에서 active 값만 static scalar buffer에 새로 쓰는 설계라면 topology update 없이 replay할 수 있다. kernel node grid dimension 자체를 5로 바꾸려는 설계라면 parameter update 경로가 필요할 수 있다. buffer pointer P0→P1 변경도 node parameter update 가능 범위와 implementation branch를 확인한다.

update 실패 후 reinstantiate는 executable과 graph definition, memory pool의 소유권 전환이다. old exec가 아직 in flight면 즉시 destroy하면 안 되고, new exec가 준비되기 전 request를 어느 path로 보낼지도 정해야 한다. fallback reason와 generation를 기록한다.

graph churn를 줄이려고 incompatible shapes를 억지로 한 bucket에 넣으면 padding와 stale predicate 위험이 커진다. 반대로 모든 exact shape를 capture하면 inventory와 memory/startup cost가 커진다. bucket 정책은 이 trade-off를 다루지만 correctness baseline은 capacity를 넘지 않고 active/padding values를 정확히 전달하는 것이다.

capture 구간 안에서 동적 allocation가 허용되는지, graph memory pool과 framework allocator가 어떤 계약을 갖는지 확인한다. warmup에서 필요한 workspace를 미리 확보하는 이유는 capture 중 allocation behavior와 pointer stability를 제어하기 위해서일 수 있다. 모든 allocation를 금지한다고 일반화하지 않고 사용 API와 framework graph pool을 본다.

workspace address가 static이어도 여러 captured graphs가 같은 pool range를 공유하면 동시 replay safety를 확인해야 한다. vLLM wrapper 주석이 multiple streams에서 global pool sharing 안전성을 별 조사 대상으로 남기는 것은 stream concurrency와 memory ownership가 연결됨을 보여 준다. graph objects별 pointer가 같다고 무조건 alias bug도 아니고, lifetimes가 겹치는데 같은 range를 쓰면 위험하다.

output tensor도 graph pool이 관리할 수 있다. weak reference나 active slice만 반환하는 path에서는 다음 replay가 같은 output storage를 덮어쓰기 전에 consumer가 끝났는지 본다. Python object가 살아 있다고 device producer-consumer lifetime가 자동으로 안전한 것은 아니다.

capture-safe라는 label를 단순 API allowlist로 쓰지 않는다. operation가 capture에 허용되고 graph가 만들어져도 external state mutation, host callback, random state와 buffer lifetime가 replay semantics에 맞아야 한다. capture 성공은 correctness 검증의 시작이다.

graph replay는 여러 launches와 dependency를 하나의 executable submission으로 묶을 수 있지만 nodes 사이 kernels를 자동으로 한 kernel로 합친다는 뜻은 아니다. profiler에는 여전히 여러 kernel nodes가 보일 수 있고 각 kernel의 HBM traffic와 synchronization는 남는다. fusion은 compiler/kernel transformation의 별도 문제다.

compile과 graph도 구분한다. compile은 Python/model graph를 lower하고 kernels를 생성·선택할 수 있으며 graph capture는 결과 execution를 기록할 수 있다. vLLM과 SGLang에서 compile buckets와 graph buckets가 겹치거나 subset 관계를 가질 수 있지만 같은 state가 아니다. compile success인데 graph fallback일 수 있고 eager compiled function를 실행할 수도 있다.

“graph를 켰더니 kernel 수가 줄었다”면 actual trace와 compiler changes를 분리한다. configuration rollout가 compile mode와 graph mode를 동시에 바꿨을 수 있다. option 연쇄를 각각 따라야 원인을 설명할 수 있다.

## 43.5 vLLM: capture size에서 runtime descriptor까지

### 43.5.1 capture size option은 정렬된 inventory가 된다

vLLM v0.27.1의 `CompilationConfig.post_init_cudagraph_sizes`는 `compile_sizes` 안의 특별 값 `cudagraph_capture_sizes`를 실제 capture size 목록으로 확장하고 capture sizes를 정렬하며 마지막 값이 maximum capture size와 맞는지 검사한다. 사용자 입력 문자열이 그대로 replay predicate에 들어가는 것이 아니라 정규화된 구성 state가 된다.

첫 option 연쇄를 닫아 보자. `cudagraph_capture_sizes` 입력은 compilation config의 list field가 된다. post-init은 이를 정렬하고 maximum invariant를 검사한다. GPU runner의 graph manager는 이 목록으로 capture descriptors와 candidate buckets를 만든다. runtime token/request shape와 mode가 candidate key에 맞으면 graph path가 선택되고 맞지 않으면 다른 descriptor 또는 eager path로 간다. 관측에는 effective sorted sizes, selected batch descriptor, runtime mode와 capture/replay 여부가 나타나야 한다.

이 연쇄에서 값을 `[8]`로 준다고 active batch 5가 자동 지원된다고 단정하지 않는다. descriptor가 token count를 key로 하는지 request count와 uniform width를 함께 갖는지, full과 piecewise mode가 padding를 어떻게 처리하는지 본다. speculative decoding와 LoRA cases는 후보를 더 나눌 수 있다.

잘못된 configuration는 startup 단계에서 드러날 수 있다. maximum과 sorted list invariant가 맞지 않거나 compile sizes special token가 잘못되면 runtime replay 전 validation가 실패한다. graph miss 조사에서 raw CLI만 보지 않고 post-init effective config를 기록하는 이유다.

### 43.5.2 candidate manager는 하나의 batch size보다 풍부한 key를 만든다

`vllm/v1/worker/gpu/cudagraph_utils.py`의 manager는 capture sizes, decode query length, maximum requests, mode와 active LoRA cases를 이용해 `BatchExecutionDescriptor` 후보를 만든다. uniform decode에서는 token count를 query width에 맞춰 round up하고 request count를 계산한다. 지원 maximum을 넘는 descriptor는 건너뛴다.

mixed piecewise와 full mode도 다르다. 소스 주석은 piecewise graph에서 request padding가 필요하지 않을 수 있고 breakable graph의 breakpoint kernels가 real batch를 forward context에서 읽으며 in-graph kernels는 padded slot mapping의 `-1` rows를 처리한다고 설명한다. fixture B의 sentinel는 구현 계약에 맞춰야 한다는 구체적 사례다.

active token count에서 smallest fitting candidate로 mapping하는 table이 만들어질 수 있다. active 5가 token bucket 8에 대응해도 descriptor에는 mode, request count, uniform token count와 LoRA case가 있다. replay key를 `(8)` 하나로 축약하면 다른 variant graph를 잘못 고를 수 있다.

capture는 mode별 descriptors를 순회하며 factory로 inputs와 forward function을 준비한다. warmup은 capture mode NONE으로 먼저 실행되고 그 뒤 graph capture가 일어난다. warmup가 metadata를 lazy initialize하거나 kernel autotune를 수행할 수 있기 때문에 warmup과 capture의 buffer values/lifetime를 구분한다.

### 43.5.3 CUDAGraphWrapper는 주소 복사를 소유하지 않는다

vLLM `CUDAGraphWrapper.__call__`은 forward context에서 batch descriptor와 runtime mode를 읽는다. mode NONE이거나 wrapper mode와 다르면 underlying runnable을 직접 호출한다. mode가 맞으면 descriptor별 entry를 만들고 graph가 없으면 capture하며 있으면 replay한다. 이 predicate가 graph/eager path의 중심이다.

중요한 주석은 wrapper가 persistent buffers를 저장하거나 runtime inputs를 그 buffers로 copy하지 않는다고 밝힌다는 점이다. dynamic shape에 대한 가정을 피하기 위해 그 책임을 wrapper 바깥에 둔다. debug logging에서는 tensor input addresses consistency를 검사할 수 있지만, production source walk는 runner/input preparer까지 올라가 static storage owner와 copy를 찾아야 한다.

capture 전에 offloader copy stream과의 관계도 보인다. wrapper는 pre-capture prefetches가 끝나도록 sync하고 captured forward 뒤 copy stream join을 관리한다. graph pool과 stream도 capture context에 전달한다. graph는 compute nodes만의 상자가 아니라 외부 copy dependencies와 memory pool lifetime를 맞춰야 한다.

이 코드가 보여 주지 않는 의도를 과장하지 않는다. wrapper가 주소 복사를 소유하지 않는다는 설계 경계와 mode/key predicate는 source fact다. 왜 전 시스템이 그 책임 분리를 선택했는지는 주석 범위 안에서만 말한다. 실제 persistent input owner는 호출 경로의 다른 객체에서 확인한다.

### 43.5.4 vLLM의 option→관측 연쇄를 반증한다

`cudagraph_mode=NONE`이면 wrapper predicate가 runnable direct path를 선택한다. capture sizes가 있어도 mode가 NONE이면 graph hit가 생기지 않는다. 관측에서 capture inventory만 보고 graph가 쓰인다고 결론 내리지 않는다. selected runtime mode와 wrapper mode를 함께 본다.

mode가 맞고 descriptor entry가 없으면 첫 호출은 capture path일 수 있다. entry가 있으면 replay다. 그래서 cold shape의 첫 latency와 warm replay를 섞으면 graph 효과를 잘못 읽는다. capture counter, replay counter와 descriptor generation를 나눈다.

input address mismatch가 debug에서 발견된다면 static buffer copy owner가 잘못됐거나 tensor가 재할당됐을 가능성이 있다. debug off에서 증상이 사라진 것이 아니라 검사만 사라질 수 있다. fixture B처럼 captured pointer와 current writer pointer를 별도로 수집한다.

active 9에 bucket 8만 있다면 candidate selection가 지원 descriptor를 반환하지 않아야 한다. eager fallback 또는 명시적 rejection reason이 예상 관측이다. 만약 bucket 8로 clamp돼 output 하나가 누락되면 predicate invariant가 깨진 것이다.

## 43.6 SGLang: graph bucket과 scheduler WAR를 함께 읽는다

### 43.6.1 decode capture sizes는 정렬·alignment·capacity를 통과한다

SGLang `get_batch_sizes_to_capture`는 decode graph config의 batch sizes를 가져와 attention TP/CP alignment와 request-pool capacity를 적용한다. maximum requests가 작으면 그 값을 목록에 보강할 수 있고, alignment와 maximum 조건으로 filter한 뒤 deduplicate/sort한다. compile buckets는 torch compile maximum 안의 subset이 될 수 있다.

두 번째 option 연쇄는 `cuda_graph_config.decode.bs`에서 시작한다. 파서/구성 객체가 batch sizes를 보유하고 `get_batch_sizes_to_capture`가 alignment와 pool capacity로 effective `capture_bs`를 만든다. decode runner는 이를 큰 size부터 capture하고 runtime `can_run_graph`가 actual batch와 width, backend, encoder/TBO/DP 조건을 검사한다. predicate가 참이면 graph key로 replay하고 거짓이면 eager runner를 쓴다. 관측에는 requested/effective sizes, graph key, rejection reason와 graph pass counter가 필요하다.

raw config에 8이 있어도 alignment filter가 제거할 수 있다. 반대로 maximum request size를 보강해 예상하지 않은 bucket이 생길 수 있다. configuration audit는 effective `capture_bs`와 `compile_bs`를 봐야 한다. graph hit rate가 낮을 때 더 많은 sizes를 무작정 추가하면 capture memory/startup cost가 늘 수 있다.

### 43.6.2 `can_run_graph`는 batch 크기 이상의 조건을 검사한다

SGLang decode runner의 `can_run_graph`는 dynamic token embedding override가 있으면 graph를 거부한다. speculative per-request width가 captured width와 다르면 eager로 보낸다. actual batch 또는 DP maximum에서 graph key를 만들고 padding disabled 여부에 따라 backend exact support 또는 `bs<=max_bs`를 판정한다. encoder-decoder mixed batch, TBO와 ngram conditions도 이어진다.

이 predicate chain은 fallback을 장애와 구분하게 한다. active batch 5가 maximum 8 이하더라도 replace embeddings가 있거나 width가 다르면 graph miss가 정상이다. graph pass metric만 보고 capture가 고장났다고 하지 않고 첫 false predicate를 기록한다.

graph key에는 stream index나 LoRA variant label이 포함될 수 있다. 같은 batch 5도 다른 stream group 또는 model variant에 다른 captured artifact가 필요할 수 있다. key cardinality와 capture inventory를 관측하지 않으면 “같은 shape인데 어떤 요청만 eager”라는 현상을 설명할 수 없다.

### 43.6.3 capture는 shared buffers를 capture 값으로 복구한다

decode runner `capture`는 warmup 뒤 capture를 수행한다. shared buffer registry가 sequence lengths를 다른 runner와 공유할 수 있으므로 capture가 필요로 하는 fill value를 다시 채우고 index buffers를 reset한다. capture는 실제 forwards를 실행하므로 이전 live batch values로 KV를 잘못 index/write하지 않게 준비한다는 주석도 있다.

큰 shapes부터 capture해 smaller shapes가 memory pool을 reuse하도록 순서를 잡는다. stream group가 있으면 각 capture stream context와 backend capture session을 사용한다. `capture_one_shape`는 dummy forward batch와 attention backend를 준비하고 graph 안/밖 metadata hooks를 구분한다.

fixture B의 핵심이 여기서 구현된다. static capacity buffer는 capture/replay 동안 유지되지만 seq lengths, indices와 physical locations는 적절한 fill/reset/refresh를 가져야 한다. “pointer가 고정됐다”는 이유로 metadata initialization를 생략하면 이전 request state가 graph에 기록되거나 replay에서 읽힐 수 있다.

### 43.6.4 WAR는 이전 read와 다음 write 사이의 edge다

SGLang scheduler는 forward stream과 별도의 schedule stream을 만들고 CUDA에서 두 handles가 같지 않도록 다시 뽑는다. overlap을 위해 streams를 나눈 순간 shared unified memory pool에 write-after-read 위험이 생긴다. 다음 batch의 scheduler write가 이전 forward의 shared-buffer read보다 먼저 실행되면 이전 forward가 새 metadata를 볼 수 있다.

`_apply_war_barrier`는 이 위험을 막는다. previous forward의 `shared_read_done_event`가 있고 coarse barrier를 강제하지 않으면 schedule stream이 그 event를 wait한다. event가 없거나 coarse mode라면 schedule stream이 forward stream을 기다린다. 그 뒤에야 다음 shared-buffer write가 안전하다.

`SGLANG_ENABLE_WAR_BARRIER` 연쇄를 닫자. 환경 option은 boolean state가 되고 CUDA에서는 barrier가 활성화된다. scheduler가 다음 batch shared buffers를 쓰려는 path에서 `_apply_war_barrier` predicate가 event 유무와 force-coarse state를 본다. fine event path는 필요한 previous-read completion만 기다리고 coarse path는 forward stream 전체 frontier를 기다린다. 관측에는 selected path, event identity와 schedule/forward stream handles, wait로 인한 queue edge가 나타난다.

barrier를 끄면 overlap이 늘 수 있어 보여도 shared read가 끝나기 전 write하면 간헐적 wrong answer가 생긴다. coarse barrier를 항상 쓰면 correctness는 지킬 수 있지만 독립적인 schedule work까지 직렬화할 수 있다. fine event publication가 누락된 phase에서는 coarse fallback가 안전망이고, 이벤트 누락 원인을 고치는 것이 후속 과제다.

### 43.6.5 SGLang fixture B를 실제 predicate로 다시 쓴다

active 5가 captured bucket 8에 들어가려면 width, backend와 other conditions가 통과해야 한다. live layout는 capacity layout에 맞게 pad되고 verify lengths와 query/output indptr 같은 metadata가 static layout로 copy될 수 있다. padding rows에는 backend가 이해하는 sentinel와 fill value가 필요하다.

metadata refresh가 schedule stream에서 일어나고 graph forward가 forward stream에서 읽는다면 현재 batch write→current forward read edge가 필요하다. 동시에 previous forward read→next batch write WAR edge도 필요하다. 두 edges는 방향이 다르다. 하나만 있으면 current data readiness 또는 buffer overwrite 중 하나가 열린다.

오답이 rows 5–7에서만 시작한다면 padding invalidation를 보고, rows 0–4 전체가 이전 generation이면 metadata refresh ordering를 본다. request가 바뀔 때만 깨지고 same batch replay는 맞으면 buffer generation와 WAR가 강한 후보다. graph가 거부돼 eager에서만 맞는다면 graph-specific static metadata path를 비교한다.

## 43.7 llama.cpp: native graph와 multi-stream fork/join

### 43.7.1 capture·end·instantiate·launch가 한 native lifecycle을 이룬다

llama.cpp v0.2.0의 CUDA backend는 `cudaStreamBeginCapture`로 stream capture를 시작하고 work를 enqueue한 뒤 `cudaStreamEndCapture`로 graph를 얻는다. executable이 없으면 `cudaGraphInstantiate`를 수행하고 `cudaGraphLaunch`로 main stream에 제출한다. Python wrapper 없이 native CUDA objects의 lifetime가 직접 드러난다.

capture mode가 relaxed라고 해서 모든 operation이 자동으로 안전한 것은 아니다. capture 구간에 허용되는 operations와 external dependencies를 확인해야 한다. allocator, host synchronization 또는 uncaptured stream interaction가 capture restrictions를 위반할 수 있다. capture failure에서는 begin/end API만 보지 않고 capture 안에 들어간 실제 operation를 찾는다.

graph launch 뒤 host function return는 replay completion가 아니다. graph nodes가 참조하는 tensor buffers와 graph executable은 launch completion까지 살아 있어야 한다. 다음 evaluation가 같은 buffers를 덮어쓸 수 있다면 stream/event ordering가 필요하다.

### 43.7.2 graph exec update는 성공과 실패가 갈린다

llama.cpp CUDA source는 새 graph를 기존 executable에 `cudaGraphExecUpdate`하려 시도하고 result를 검사한다. update가 성공하지 않으면 기존 instance를 파기하거나 새 graph를 instantiate하는 경로가 필요하다. topology와 parameters가 변해도 무조건 기존 exec를 쓸 수 있다는 가정은 틀리다.

관측 record에는 old/new graph generation, update result와 error node, reinstantiate count를 둔다. latency 상승이 graph 때문이라면 replay만 센 것이 아니라 update failure로 매번 instantiate하는지 본다. shape/topology가 안정적일 때와 변할 때를 나눈다.

update 성공도 output correctness 증명은 아니다. node parameters가 의도한 pointers와 dimensions로 바뀌었는지, static buffers가 valid generation인지 확인한다. update API return와 data readiness edge는 서로 다른 계약이다.

### 43.7.3 fork event는 auxiliary streams를 출발시키고 join event는 되돌린다

llama.cpp의 multi-stream 경로는 main stream에서 fork event를 record하고 auxiliary streams가 이를 wait하는 모습을 보여 준다. 이것은 main stream의 선행 준비가 끝난 뒤 parallel branches가 시작하도록 한다. branch work가 끝나면 auxiliary streams에서 join events를 record하고 main stream이 wait해 이후 consumer가 모든 branches 뒤에 놓이게 한다.

fork만 있고 join이 없으면 main stream이 auxiliary output 준비 전에 소비할 수 있다. join만 있고 fork가 없으면 auxiliary branch가 input 준비 전에 시작할 수 있다. fork/join은 parallel region의 양쪽 경계다. event identity와 branch stream index를 함께 기록한다.

이 구조를 fixture A에 대응시키면 main preparation가 producer, auxiliary kernels가 consumers인 첫 edge와 auxiliary outputs가 producers, main continuation가 consumer인 둘째 edge가 있다. 하나의 “동기화됨” flag로는 두 방향을 표현할 수 없다.

### 43.7.4 세 스택을 같은 틀로만 비교한다

vLLM은 compilation/runtime mode와 batch descriptor로 wrapper capture/replay를 dispatch하며 persistent buffer copy 책임을 바깥에 둔다. SGLang runner는 phase별 capture buckets, shared buffers와 attention metadata hooks를 소유하고 scheduler/forward streams 사이 WAR를 명시한다. llama.cpp는 native CUDA graph lifecycle와 fork/join events를 직접 관리한다.

셋을 동일 구현이라 하지 않는다. 비교축은 graph key가 무엇인지, static address owner가 누구인지, active metadata를 누가 갱신하는지, capture/update가 실패할 때 어디로 fallback하는지, multi-stream dependency를 어떤 primitive로 닫는지다. 이 축으로 보면 Python/native 차이보다 시간 계약이 선명해진다.

세 구현을 fixture B에 넣어 보면 차이가 더 구체적이다. vLLM에서는 runtime mode와 `BatchExecutionDescriptor`가 먼저 bucket 8에 대응해야 하고 wrapper 바깥의 input owner가 active 5 values를 persistent buffers에 준비해야 한다. wrapper는 descriptor entry를 찾아 capture 또는 replay하지만 runtime inputs copy를 스스로 소유하지 않는다. 따라서 stale row가 보이면 wrapper의 graph cache만 보지 않고 upstream static-input preparation를 찾는다.

SGLang에서는 decode runner가 effective capture sizes와 width를 알고 `can_run_graph`에서 현재 `ForwardBatch`를 검사한다. runner와 attention backend가 shared buffers, seq lengths, slot mapping와 graph metadata hooks를 나눠 가진다. scheduler stream과 forward stream가 분리돼 있으면 current metadata readiness와 previous read completion 두 방향의 edges도 닫아야 한다.

llama.cpp에서는 native code가 graph definition와 executable instance를 직접 보유하고 update result에 따라 재instantiate할 수 있다. shape/topology 변화가 update 가능한지 여부가 bucket 전환의 핵심이다. auxiliary streams가 capture/replay work에 참여하면 fork/join events가 graph lifecycle와 함께 유효해야 한다.

이 비교는 어느 구현이 더 낫다는 순위를 만들지 않는다. 책임 배치가 다르므로 첫 조사 위치가 다르다는 뜻이다. vLLM wrapper에서 copy를 찾지 못했다고 copy가 없다고 결론 내리지 않고, SGLang runner가 buffers를 소유한다고 모든 backend sentinel가 같다고 가정하지 않으며, llama.cpp update API가 있다고 모든 dynamic shape가 update 가능하다고 쓰지 않는다.

fixture A도 같은 비교가 가능하다. vLLM source의 offloader sync/join는 copy stream와 captured forward 사이 경계를 드러낸다. SGLang WAR barrier는 scheduler write와 forward read의 반대 방향 hazard를 보여 준다. llama.cpp fork/join는 main과 auxiliary streams 사이 fan-out/fan-in을 보여 준다. 세 사례 모두 “event를 쓴다”보다 producer와 consumer를 정확히 말해야 한다.

fan-out에서는 한 producer event를 여러 consumer streams가 wait할 수 있다. 각 consumer가 같은 준비 range를 읽는다면 자연스럽다. fan-in에서는 여러 producers가 각 completion event를 record하고 main consumer가 모두 기다려야 한다. 하나의 branch event만 기다리면 partial readiness다. branch count가 runtime shape에 따라 달라진다면 graph topology/update와 event inventory도 맞아야 한다.

copy engine와 compute overlap을 원할 때도 data partition를 본다. KV blocks 20–23을 한 copy가 모두 준비하는지, heads 또는 layers별 copies가 여러 streams에 나뉘는지에 따라 event granularity가 달라진다. 너무 이른 단일 event는 partial range만 보호하고, range마다 지나치게 많은 events는 overhead와 복잡성을 늘릴 수 있다. ownership boundary와 consumer read set에 맞춘다.

graph capture는 이 event topology를 기록할 수 있지만 graph 바깥 producer와의 edge는 replay마다 다시 연결해야 할 수 있다. 예를 들어 host-to-device metadata copy가 graph 외부 `S_copy`에 있고 graph가 `S_compute`에 launch되면 E43 wait는 graph launch 앞에 있어야 한다. capture 때 한 번 있었던 event가 모든 replay의 새 metadata generation를 자동으로 보호한다고 가정하지 않는다.

반대로 metadata copy node를 graph 안에 capture했다면 source host buffer의 address와 lifetime가 replay-safe한지 확인한다. replay 전에 host buffer values를 바꾸는 ordering와 pinned storage stability가 필요하다. device static buffer 주소만 고정한 경우와 contract가 다르다. “copy가 graph 안/밖”을 launch trace에 포함한다.

output consumer도 잊지 않는다. graph launch 뒤 sampler나 cache update가 같은 stream이면 stream order가 이어진다. 다른 stream이면 graph completion event와 wait가 필요할 수 있다. host가 output token을 읽거나 network response를 만들기 전에는 적절한 completion를 기다려야 한다. input readiness만 고치고 output readiness를 열어 두지 않는다.

graph executable cache의 eviction도 lifetime 사건이다. descriptor entry를 지우거나 clear graphs를 호출해 Python mapping에서 제거했어도 in-flight launch가 instance와 pool storage를 참조할 수 있다. framework가 안전한 destruction를 제공하는지 확인한다. capture inventory 감소와 device memory release 시점은 같지 않을 수 있다.

multi-tenant serving에서는 static buffer row가 어느 request generation를 담는지 민감하다. padding predicate 누락은 단순 zero work 문제가 아니라 다른 tenant state를 읽는 isolation 사건이 될 수 있다. raw contents를 로그에 남기지 않아도 row owner digest와 generation mismatch를 검출할 수 있어야 한다.

speculative decoding는 active batch와 per-request width를 함께 변화시킨다. request 다섯 개라도 각 request가 여러 draft positions를 검증하면 token rows와 captured width가 달라진다. vLLM descriptor와 SGLang captured request width predicate가 이 차이를 반영한다. fixture B의 “5”가 request count인지 token count인지 원장 첫 줄에서 정의한다.

LoRA나 encoder override도 graph key/predicate를 바꿀 수 있다. 같은 capacity 8 buffer에 다른 effective model state를 섞지 않는다. variant label가 key에 들어가면 expected miss이고, key에 없다면 runtime inputs가 graph-safe하게 model variant를 전달하는지 확인한다. graph hit를 올리려고 correctness identity를 key에서 제거하지 않는다.

compile/piecewise graph에서는 graph break 전후 outputs가 다음 segment inputs가 된다. segment 사이 address와 lifetime, current stream ordering를 확인한다. 한 full graph의 nodes처럼 보이지 않아도 wrapper chain가 같은 step의 dependency를 이어야 한다. breakpoint kernel가 real batch를 읽는다면 active scalar generation가 segment마다 일치해야 한다.

이 모든 비교의 최소 산출물은 여섯 문장이다. 어떤 option/config가 effective state를 만들었는가. 어떤 predicate가 graph key와 path를 골랐는가. 누가 static address를 소유하는가. 누가 active values와 padding를 갱신하는가. 어떤 event/barrier가 producer와 consumer를 잇는가. 실패하면 어떤 eager/update/reinstantiate 관측이 나타나는가. 여섯 문장이 닫히지 않으면 “CUDA Graph가 사용됐다”는 설명은 너무 얕다.

공식 문서의 버전 차이도 이 틀로 제한한다. CUDA 13.3.0 문서가 재구성되고 새로운 graph 기능이나 PDL 설명이 추가돼도 stream/event의 기본 edge 의미가 뒤집힌 것으로 쓰지 않는다. CUDA 12.9.1의 baseline semantics와 교차 확인하고 선택 기능은 실제 source가 사용하는 경우에만 implementation path에 연결한다.

toolkit/driver compatibility는 44장의 binary 실행 계약이다. graph capture가 실패했다고 CUDA 13.x semantics 변화라고 결론 내리기 전에 library commit, capture operation, device capability와 error를 고정한다. 동일 source라도 compiled artifact와 backend availability가 달라 selected path가 바뀔 수 있다.

마지막으로 graph는 request scheduling를 대신하지 않는다. scheduler가 batch와 token rows를 결정하고 graph runner가 supported bucket에 매핑한다. queue fairness나 preemption가 만든 shape distribution가 graph hit opportunity를 바꾼다. graph option만 조정하기 전에 upstream distribution와 downstream padding/metadata cost를 함께 본다.

## 43.8 stale state·hang·과동기화를 first divergence로 가른다

### 43.8.1 간헐적 wrong answer

첫 증상은 부하에서만 output이 달라지는 것이다. producer/consumer streams, event record/wait 위치, buffer address/generation와 active length를 확인한다. 같은 deterministic input와 sampling 조건에서 first attention output 또는 metadata checkpoint를 비교한다.

event wait를 임시로 whole-stream/device synchronize로 바꿨을 때 증상이 사라지면 ordering 가설이 강해진다. 하지만 source pointer, generation와 predicate가 틀린 사건도 synchronize로 가려질 수 있다. 정확한 E record/wait를 복구하고 address/value identity를 별 검증한다.

**같은 분기의 두 번째 징후: graph replay 뒤 이전 요청 token이 섞인다.**

fixture B의 row generation table을 만든다. active rows 0–4는 current request IDs와 metadata generation여야 하고 rows 5–7은 invalid다. captured pointer, writer pointer, active scalar와 sentinel를 비교한다. first bad row가 padding boundary인지 current active range 안인지 나눈다.

current rows도 모두 stale면 write→replay edge나 writer destination pointer를 본다. padding만 stale면 invalidation와 predicate를 본다. batch 9에서만 누락되면 bucket selection/fallback을 본다. 같은 “이전 token 혼입”도 first divergence가 다르다.

### 43.8.2 graph 선택과 latency를 같은 비용 원장에서 가른다

requested shape와 normalized graph key, capture inventory, runtime mode와 rejection predicate를 기록한다. vLLM에서는 descriptor가 mode/token/request/width/variant를 가질 수 있고 SGLang에서는 width, backend, encoder/TBO/DP 조건이 있다. batch size histogram 하나로 hit opportunity를 계산하지 않는다.

capture sizes를 늘리는 수정은 startup capture time, graph pool memory와 key cardinality를 늘릴 수 있다. 가장 많은 miss reason가 unsupported dynamic feature라면 buckets 추가로 해결되지 않는다. eager fallback correctness와 graph path의 result를 먼저 맞춘다.

**graph를 켠 뒤 latency가 더 나빠졌다면.**

padding work, graph miss/eager fallback, graph exec update 실패/reinstantiate, serialization events와 CPU submission을 분해한다. `B_cap=8,B_act=5`의 37.5%는 logical padding fraction일 뿐 latency 원인이 확정되지 않는다. selected kernels가 invalid rows를 처리하는 지점을 읽는다.

fine-grained event가 coarse wait로 fallback하면 scheduler overlap가 줄 수 있다. graph가 hit해도 metadata preparation와 copy가 critical path에 남을 수 있다. capture가 반복되거나 update가 계속 실패하면 replay 이점이 상쇄될 수 있다. 실제 측정은 별도 환경에서 해야 하며 이 장은 관측 항목과 분기만 준다.

### 43.8.3 capture failure와 hang을 lifecycle·dependency로 가른다

capture mode, capture stream, forbidden operation, dynamic allocation, external stream dependency와 backend hook을 확인한다. failure API가 end capture에서 보였다고 root operation가 마지막 node라는 뜻은 아니다. capture 시작부터 operation timeline를 좁힌다.

capture failure 후 graph object와 pool/buffers를 어떻게 정리하는지도 본다. partially initialized entry를 cache에 남기면 다음 호출이 replay 가능한 것으로 오인할 수 있다. eager fallback가 있다면 failure reason와 graph entry generation를 기록한다.

**capture는 끝났지만 deadlock 또는 hang이 난다면.**

CTA barrier에서는 expected participants와 divergent branch를 센다. pipeline에서는 acquire/commit/wait/release pairing와 stage reuse를 본다. cross-stream에서는 events를 nodes로, waits를 directed edges로 그려 cycle을 찾는다. event object 재사용 generation도 표시한다.

host thread가 stream synchronize를 기다리고 그 stream은 host가 아직 enqueue하지 않은 event를 기다리는 control dependency도 있을 수 있다. device graph만 보지 않고 host enqueue state를 포함한다. timeout으로 barrier를 건너뛰어 correctness를 희생하지 않는다.

### 43.8.4 복구 증거를 두 fixture의 전이로 닫는다

cross-stream KV 수정은 producer event generation와 consumer wait가 같은 buffer generation를 보호하고 deterministic fixture에서 first divergence가 사라져야 한다. whole-device synchronize 없이 required overlap 범위를 유지해야 한다. cancellation에서도 buffers/events lifetime가 닫혀야 한다.

graph stale-state 수정은 bucket 8/active 5에서 rows 0–4 current, 5–7 invalid이며 active 8→5와 5→8 transitions 모두 맞아야 한다. buffer address를 의도적으로 바꾸는 supported/unsupported path에서 update 또는 rejection가 명확해야 한다. batch 9는 larger bucket/eager로 간다.

WAR 수정은 previous read done event와 next schedule write 사이 edge가 관측되고 fine path 누락 시 coarse fallback가 작동해야 한다. barrier off에서 빠르다는 결과로 되돌리지 않는다. coarse만 계속 선택되면 event publication gap와 과동기화 비용을 별 후속 문제로 남긴다.

**fixture A를 세 번 재현하는 조사 대화.**

첫 실행은 same-stream baseline이다. copy와 attention을 한 stream에 놓고 buffer generation를 고정한다. 이 경로에서도 틀리면 cross-stream event보다 pointer, length, kernel correctness를 먼저 본다. same-stream이 맞고 two-stream에서만 틀리면 missing/incorrect edge 가설이 강해진다.

둘째 실행은 explicit E를 둔다. record가 copy 뒤인지, wait가 attention 앞인지 source와 trace에서 확인한다. event object 이름만 맞는 것이 아니라 generation와 buffer range가 current request와 맞아야 한다. 이 경로가 맞으면 required dependency가 확인된다.

셋째 실행은 의도적으로 wait가 없는 original behavior를 관찰 대상으로 삼되 production data에서 corruption를 유발하지 않는다. 작은 deterministic fixture와 격리 환경에서만 비교해야 한다. 이 집필에서는 실행하지 않으며, 실제 팀은 안전 정책과 승인 아래 수행한다. 재현이 안 돼도 edge 누락이라는 source fact가 사라지지 않는다.

결론 문장은 “event를 넣으니 고쳐졌다”보다 구체적이다. “generation 17 KV copy가 S_copy에서 E17 전에 완료되고 S_compute attention A17이 E17 wait 뒤에 enqueue되도록 edge를 추가했다. same buffer generation를 유지하고 device-wide sync 없이 reference output과 일치했다”라고 쓴다.

**fixture B를 전이 행렬로 검증한다.**

active batch를 8→5→8→1→9로 바꾼다고 하자. 8→5는 padding invalidation, 5→8은 previously invalid rows의 complete refresh, 8→1은 넓은 stale tail 방지, 1→9는 bucket miss/greater bucket 선택을 시험한다. 한 번의 active 5 replay만으로 양방향 전이를 증명하지 않는다.

각 replay에서 static address generation는 유지될 수 있지만 value generation는 증가한다. row table에 request ID digest, seq length, slot mapping sentinel와 writer operation ID를 둔다. output은 active slice만 비교하고 side effects인 KV writes/allocator refs도 padding rows에서 발생하지 않는지 본다.

graph key가 capacity 외 width와 variant를 포함하면 전이 행렬을 key별로 나눈다. active 5 width1과 active 5 speculative width4가 다른 graph를 요구할 수 있다. fallback가 정상인 cell과 replay가 정상인 cell을 미리 정한다.

batch 9가 eager로 갔다가 다음 batch 5가 graph로 돌아올 때 static buffers가 eager values에 오염되지 않았는지도 본다. graph/eager paths가 shared buffers를 쓴다면 handoff ordering와 reset가 필요하다. graph-only 연속 replay만 시험하면 이 경계를 놓친다.

### 43.8.5 cancellation과 불완전한 동기화를 lifetime 원장에 넣는다

request가 취소돼도 이미 enqueue된 copy/kernel가 storage를 참조할 수 있다. host request object를 terminal로 만들었다고 device work가 사라지지 않는다. event/future completion까지 buffer owner를 delayed release하거나 안전한 cancellation protocol를 써야 한다.

graph replay 중 batch의 한 request가 취소되면 captured active rows를 mid-flight에서 바꾸지 않는다. 다음 replay metadata에서 제외하고 current replay output를 discard하는 방식일 수 있다. implementation contract를 읽는다. static buffer row를 즉시 다른 request로 재사용하면 in-flight graph가 새 values를 읽을 위험이 있다.

cancellation leak와 correctness를 함께 본다. 너무 늦게 release하면 capacity leak가 되고 너무 일찍 release하면 use-after-free다. owner state에 terminal request, in-flight event/graph generation와 release condition를 둔다. 37장에서 다룬 owner ledger를 시간 edge와 연결한다.

**event가 있어도 틀릴 수 있는 네 경우.**

첫째, event를 producer operation 전에 record했다. consumer wait는 존재하지만 잘못된 frontier를 기다린다. 둘째, correct producer stream가 아니라 unrelated stream에서 record했다. host log 순서만 반영할 뿐 copy completion를 포함하지 않는다.

셋째, wait가 consumer operation 뒤에 enqueue됐다. 이후 work는 보호하지만 이미 시작 가능한 attention은 보호하지 못한다. 넷째, event와 wait는 맞지만 buffer pointer/generation가 다르다. E17은 P0 copy를 보호하는데 attention은 P1을 읽거나 그 반대일 수 있다.

따라서 event count metric는 correctness metric가 아니다. event edge tuple `(producer op, range generation, record stream, event generation, wait stream, consumer op)`을 검증한다. source에서 record/wait 위치를 보고 trace에서 current tuple을 확인한다.

**barrier가 있어도 hang할 수 있는 네 경우.**

첫째, conditional branch 때문에 일부 participating threads가 barrier에 도달하지 않는다. 둘째, expected arrival count가 actual participants보다 크다. 셋째, pipeline stage release가 빠져 producer가 다음 acquire에서 막힌다. 넷째, cross-stream waits가 cycle을 이룬다.

hang stack가 host synchronize에서 보인다고 host가 root는 아니다. device kernel barrier가 끝나지 않아 stream completion가 없고 host가 그 결과를 기다릴 수 있다. kernel progress, stream dependency graph와 host wait chain를 함께 그린다.

timeout 후 process를 재시작하면 서비스는 회복될 수 있지만 root barrier protocol를 고친 것은 아니다. failing shape, CTA predicate와 event graph를 보존한다. nondeterministic hang일수록 last completed event generation와 oldest wait age가 유용하다.

### 43.8.6 과동기화를 단계적으로 줄인다

correctness incident에서 device synchronize로 먼저 안전을 확보했다면 다음 단계는 producer stream synchronize, event wait, 더 좁은 pipeline/barrier로 scope를 줄이는 것이다. 각 단계에서 보호하는 buffer range와 consumer set가 같아야 한다. 단순히 latency가 좋아지는 방향으로 edge를 삭제하지 않는다.

coarse forward-stream wait가 fine shared-read event로 바뀌면 다음 scheduler writes 중 truly dependent한 것만 뒤로 미룰 수 있다. 그러나 event publish가 모든 read consumers 뒤에 있는지 확인한다. 일부 backend가 event를 일찍 record하면 fine path가 correctness를 깨뜨릴 수 있다.

과동기화 개선의 결과는 graph hit 여부와 별로 본다. graph replay가 빨라도 coarse WAR가 critical path를 직렬화할 수 있고 eager에서도 같은 barrier cost가 있을 수 있다. selected path와 wait duration를 실제 환경에서 측정해야 하지만 여기서는 causal fields만 정한다.

request/step R43은 active 5, capacity 8, graph key K8-v1, graph exec generation G3라고 적는다. static metadata address P0 allocation generation A7, values generation V43을 적는다. rows 0–4 current, 5–7 sentinel이라는 digest를 남긴다.

metadata writer W43은 schedule/copy stream S1에 있고 event E43를 record한다. graph launch L43은 compute stream S2에서 E43 wait 뒤다. 이전 graph read completion R42 event를 S1이 기다려 WAR도 닫는다. 즉 `R42→W43→E43→L43`의 양방향 buffer handoff가 보인다.

selected path가 replay인지 capture/eager인지, rejection reason가 무엇인지 기록한다. update를 시도했다면 result, reinstantiate generation를 넣는다. first bad row/token와 그 값을 마지막으로 쓴 operation를 연결한다. 이 record는 prompt content 없이도 시간 문제를 좁힐 수 있다.

request ID를 metric label로 넣지 않는다. graph mode, bounded bucket, backend, rejection category와 architecture처럼 제한된 labels를 쓰고 exact request trace는 sampling/exemplar로 연결한다. static addresses도 외부 dashboard에 원문으로 노출하지 않고 process-local pseudonymous IDs와 allocation generation를 쓸 수 있다.

metadata values 전체를 dump하면 token/tenant 정보가 노출될 수 있다. row validity, lengths buckets, sentinel 여부와 digest로 충분한 경우가 많다. correctness escalation에서 필요한 최소 범위만 승인된 저장소에 보존한다.

device printf나 모든 event trace는 실행을 바꿀 수 있다. 평상시 host-side dispatch/edge record를 유지하고 작은 fixture에서만 상세 kernel/barrier instrumentation를 쓴다. instrumented와 uninstrumented behavior 차이도 사건 기록에 남긴다.

복구 보고서를 fixture A로 마무리할 때는 호출 순서 그림만 넣지 않는다. copy가 생산한 exact KV block range와 generation, event record stream, consumer wait stream, attention launch, cancellation 때 release condition를 적는다. 같은-stream baseline과 explicit-event path가 모두 reference output과 맞는지 구분한다. 전체 synchronize를 제거한 뒤에도 맞아야 required edge가 복구된 것이다.

fixture B 보고서에는 graph key, capacity와 active를 함께 쓴다. active 5가 bucket 8에 들어갔다는 한 줄 뒤에 static address generation, row별 value generation와 sentinel, metadata writer와 replay edge를 붙인다. 8→5→8과 bucket overflow 9의 path를 각각 기록한다. graph hit 하나만으로 stale-row 방지가 검증되지는 않는다.

graph miss를 고친 변화는 correctness matrix를 다시 돈다. bucket inventory를 넓혀 hit가 늘었더라도 LoRA/width/encoder 같은 variant가 잘못 합쳐지면 다른 state를 replay할 수 있다. expected eager cells가 graph cells로 바뀌었다면 key와 static metadata가 새 identity를 충분히 표현하는지 확인한다.

과동기화를 줄인 변화는 race matrix를 다시 돈다. coarse wait를 fine event로 바꿨다면 모든 backend가 shared-read-done event를 올바른 frontier에서 publish하는지 확인한다. event가 없는 phase에서 fallback가 남는지, cancellation와 error path에서도 event/lifetime owner가 닫히는지 본다.

update/reinstantiate 변화는 graph generation 원장을 다시 돈다. compatible update 성공, incompatible update failure 뒤 새 instantiate, capture failure 뒤 eager fallback를 구분한다. old instance가 in flight인 동안 파기되지 않고 new instance가 올바른 buffers를 참조하는지 확인한다. latency만 좋아졌다고 lifecycle가 안전해진 것은 아니다.

hang 수정은 정상 결과만으로 부족하다. barrier participant count와 stage protocol, cross-stream dependency graph에서 cycle가 없어야 한다. tail shape와 early-exit path에서 모든 required threads가 barrier에 도달하는지 확인한다. timeout 횟수가 0으로 줄었다는 aggregate만으로 rare divergent branch를 배제하지 않는다.

독자가 가져갈 최종 조사 습관은 단순하다. “누가 썼는가, 어디까지 썼는가, 누가 읽는가, 어느 edge를 기다리는가, 언제 재사용 가능한가”를 매 buffer마다 묻는다. graph가 있어도 같은 질문을 replay generation마다 반복한다. 주소가 고정돼도 값과 owner는 시간에 따라 변하기 때문이다.

이 습관은 correctness와 performance를 동시에 보호한다. edge가 없으면 stale read가 나고 edge가 너무 넓으면 overlap가 사라진다. lifetime이 짧으면 use-after-free가 나고 지나치게 길면 memory pressure가 생긴다. 정답은 synchronization의 양을 늘리거나 줄이는 것이 아니라 실제 producer-consumer 범위와 일치시키는 것이다.

운영자가 incident를 인계할 때도 이 형식을 유지한다. “CUDA Graph 문제”라고 쓰는 대신 “active 5 replay G43이 static address P0를 사용했지만 slot metadata generation는 G42였고 writer W43→launch edge가 없었다”처럼 쓴다. 또는 “forward read completion event가 publish되지 않아 scheduler가 coarse wait로 fallback했고 correctness는 유지됐지만 overlap 후보가 직렬화됐다”고 쓴다. 이 문장은 다음 조사자가 source predicate와 trace를 바로 찾게 한다.

근거가 모자라면 불확실성을 숨기지 않는다. event record는 source에 있지만 current request가 wait했는지 trace가 없을 수 있다. graph key는 알지만 row별 generation를 수집하지 않았을 수 있다. 이때 race나 stale state를 확정하지 않고 다음 재현에서 필요한 bounded instrumentation를 추가한다. 추측으로 빈 edge를 채우는 것보다 missing evidence를 명시하는 편이 안전하다.

마지막으로 임시 안전 조치와 완료를 구분한다. graph를 끄고 eager로 보내거나 streams를 하나로 합치고 전체 synchronize를 넣으면 blast radius를 줄일 수 있다. 그러나 이는 root key, metadata refresh, event frontier나 lifetime를 고친 것이 아니다. 안전 상태에서 증거를 보존한 뒤 정확한 edge를 복구하고, 임시 serialization를 제거한 상태로 다시 검증해야 incident가 닫힌다.

### 43.8.7 사건 E43: 완료된 E7을 새 buffer의 완료로 잘못 읽었다

두 streams와 buffers 두 개로 사건을 고정한다. stream `S_copy`는 host staging에서 device KV buffer K17로 copy하고, stream `S_compute`는 K17을 읽어 attention output O17을 만든다. allocation identity는 `(ptr=0xA000,generation=17)`이다. event pool에서 가져온 event object E7은 여러 step에서 재사용된다.

`t0=0 µs`에 host가 async copy를 `S_copy`에 enqueue하고 API가 반환한다. 반환은 명령이 stream에 받아들여졌음을 뜻하지만 DMA 완료를 뜻하지 않는다. `t1=3 µs`에 host가 E7 record를 `S_copy` 뒤에 enqueue한다. `t2=5 µs`에 `S_compute`가 E7 wait를 enqueue하고 그 뒤 attention kernel A17을 enqueue한다. 이 edge는 copy completion이 A17 read보다 앞서게 한다.

`t3=9 µs`에 host는 request abort를 처리한다. request table에서는 K17 logical reference가 0이 된다. 버그 난 cleanup은 `cudaEventQuery(E7)==success`를 보고 K17을 allocator에 반환했다. 그러나 query한 E7은 이번 generation의 copy event가 아니라 event pool에서 이전 step에 record된 완료 상태였다. 새 `cudaEventRecord`가 어느 host ordering에서 실제 E7 state를 갱신하는지와 callback가 어느 generation을 가리키는지 ledger가 없었다.

`t4=12 µs`에 allocator는 같은 pointer 0xA000을 K18로 재사용한다. `S_compute`의 A17은 아직 실행되지 않았다. `t5=17 µs`에 새 request가 K18을 초기화한다. `t6=20 µs`에 과거 A17이 0xA000을 읽어 O17을 쓰거나, ordering에 따라 K18 initialization과 경쟁한다. exception 없이 logits만 흔들리는 use-after-free다.

핵심은 event object가 완료됐다는 bool과 “어느 record generation 앞의 work가 완료됐는가”가 다르다는 데 있다. E7이라는 handle은 identity가 아니라 재사용 가능한 synchronization object다. delayed lifetime record는 `(event_handle, event_generation, buffer_generation, producer_stream, protected_effect)`를 가져야 한다. query consumer는 자신이 기다리는 record generation을 확인한다.

또한 copy event E7은 K17의 last reader completion이 아니다. E7은 `S_copy`의 producer work가 끝났음을 증명해 A17이 안전하게 시작할 수 있게 한다. K17을 free하려면 A17 read가 끝났다는 별도 event, 예를 들어 E8을 `S_compute`에서 A17 뒤에 record해야 한다. producer-ready와 consumer-done을 같은 event로 합치면 조기 반환이 생긴다.

### 43.8.8 여섯 시각을 한 행에 합치지 않는다

E43을 여섯 열로 다시 쓴다. host return은 API call이 CPU에 control을 돌려준 시각이다. enqueue frontier는 stream order에 operation이 들어간 시각이다. execution start와 execution finish는 device가 실제 work를 수행한 구간이다. event completion은 record operation 앞의 stream work가 완료된 상태다. visibility는 consumer가 올바른 dependency 뒤 memory를 읽을 수 있다는 관계다. lifetime end는 어떤 future work도 allocation generation을 읽거나 쓰지 않는다는 증거다.

copy API가 t0에 반환돼도 execution은 나중일 수 있다. E7 record call이 t1에 반환돼도 E7 complete는 copy finish 뒤다. `S_compute` wait call이 t2에 반환돼도 host가 기다린 것은 아니다. A17 enqueue call이 반환돼도 K17 reader lifetime은 끝나지 않는다. 이 네 반환을 “GPU 작업 완료” 로그로 합치면 cleanup가 너무 빨라진다.

memory visibility도 단순 wall-clock 비교가 아니다. copy가 우연히 먼저 끝났어도 `S_compute`가 dependency 없이 K17을 읽으면 program contract가 없다. 같은 stream order 또는 event wait edge를 둔다. 반대로 dependency가 있어도 buffer owner가 edge를 등록하기 전에 반환되면 새 generation이 같은 address를 쓸 수 있다. visibility와 allocation lifetime이 모두 필요하다.

object lifetime에는 device buffer, event object, graph executable, captured argument storage가 각각 있다. graph exec가 kernel node에 pointer를 보존한다면 host request가 끝났다고 pointer allocation를 재사용할 수 없다. graph exec update가 새 pointer를 성공적으로 반영했는지, old replay가 in-flight인지 확인한다. event와 graph handle destroy call의 host return도 device use 종료와 동일시하지 않는다.

### 43.8.9 multi-stream 손검산: edge가 하나 빠질 때 생기는 세 결과

정상 graph는 `copy K17 on S0 → record Ready17 → wait Ready17 on S1 → attention read K17 → record Done17 → allocator reuse`다. happens-before edges는 copy→Ready17 completion, Ready17→attention start, attention finish→Done17 completion, Done17→reuse다. 네 edge 가운데 하나라도 빠지면 서로 다른 사고가 난다.

첫 edge가 없거나 event가 copy 앞에 record되면 Ready17이 너무 일찍 complete한다. attention은 partial KV를 읽는다. 둘째 wait edge가 없으면 streams 실행 순서에 기대게 되어 load에 따라 간헐적으로 틀린다. 셋째 Done17 record가 attention 앞이면 allocator가 reader 종료를 잘못 증명한다. 넷째 completion generation check가 없으면 old Done16이 K17 reuse를 허용한다.

시간표로 검산한다. copy 실행이 10–30 µs, attention이 dependency 없으면 15–25 µs라 overlapping read가 생긴다. Ready17 wait를 넣으면 attention start는 최소 30 µs다. attention duration 10 µs면 Done17은 최소 40 µs 뒤 complete한다. allocator claim K18은 40 µs 뒤여야 한다. host enqueue timestamps 0–5 µs만 보고 9 µs에 free하면 이 관계를 위반한다.

auxiliary streams가 두 개면 join 조건은 둘 모두다. S1 attention reader가 40 µs, S2 transfer reader가 55 µs에 끝나면 K17 reuse frontier는 max 55 µs다. event 하나만 기다리거나 “가장 최근 event” 하나로 덮지 않는다. lifetime ledger는 outstanding consumers set을 가지고 각 completion effect를 한 번씩 제거한다.

**Graph replay는 pointer value보다 generation 계약이 중요하다.**

bucket 8 graph는 input/output buffers의 capacity를 8로 잡고 active rows 5를 metadata로 전달할 수 있다. replay 직전 buffer pointer가 같아도 allocation generation이 바뀌었으면 captured graph의 owner contract가 유효한지 확인해야 한다. caching allocator는 같은 address를 돌려주므로 pointer equality는 lifetime equality가 아니다.

replay R17이 K17을 읽는 동안 host가 next batch R18 metadata를 같은 shared input buffer에 쓰면 write-after-read hazard가 된다. double buffering, per-replay slot, completion event 뒤 overwrite 가운데 하나가 필요하다. SGLang이 shared buffers와 scheduler WAR edge를 관리하는 경로를 읽을 때 capture size만 보지 않는 이유다.

graph launch API return은 replay 완료가 아니다. exec object를 update하거나 destroy하기 전에 in-flight launches와 update contract를 확인한다. update success는 topology/parameters가 허용 범위에서 바뀌었다는 뜻이지 과거 launch가 끝났다는 뜻이 아니다. lifecycle state에는 instantiated, launched generation, completed generation, update generation을 분리한다.

stale active batch 값은 memory safety와 correctness를 동시에 해친다. capacity 8 buffer에 현재 active 5인데 graph node가 old active 8을 읽으면 rows 5–7의 stale pointers나 slot IDs를 소비할 수 있다. padding rows를 안전한 sentinel로 초기화하고 active count가 correct generation에서 update됐는지 검증한다. “capacity 안이므로 out-of-bounds가 아니다”로 끝내지 않는다.

**vLLM·SGLang source walk를 completion consumer까지 잇는다.**

vLLM에서는 capture size inventory와 runtime candidate manager가 graph key를 만드는 경로에서 시작한다. selected graph가 어떤 input buffers와 descriptors를 재사용하는지 보고, wrapper launch 뒤 누가 completion 또는 buffer slot readiness를 관리하는지 호출자로 계속 올라간다. graph wrapper 함수만 읽고 request abort cleanup와 연결하지 않으면 lifetime handoff가 빠진다.

source 기록에는 capture bucket, actual batch, graph mode, input buffer identity, replay generation, output consumer frontier를 남긴다. abort가 replay launch 전이면 queue에서 제거 가능한지, launch 뒤면 terminal output만 drop하고 device lifetime은 drain하는지 구분한다. request status terminal이 graph buffer reusable과 같은 predicate인지 확인한다.

SGLang에서는 decode capture sizes를 만드는 설정 경로, `can_run_graph` predicate, replay buffer update, scheduler WAR synchronization을 잇는다. capture 가능한 batch 크기라는 사실과 현재 metadata가 안전하게 replay 가능한 generation이라는 사실은 다르다. shared buffer overwrite를 막는 event가 어떤 stream에서 record되고 어느 다음 writer가 wait하는지 찾는다.

FlashInfer나 attention backend가 별도 stream/workspace를 쓰면 graph runner의 event만으로 모든 consumer가 닫히는지 확인한다. native call이 caller stream contract를 따르는지 내부 auxiliary stream을 join하는지 source와 공식 contract로 증명한다. 알 수 없으면 “graph launch 완료 event가 backend workspace lifetime까지 보호한다”고 쓰지 않는다.

호출 source pin은 네 지점이다. graph 선택 predicate, input copy/update, graph launch, completion consumer 또는 next overwrite guard다. 그 사이 request abort/free 경로를 다섯 번째로 연결한다. 이 다섯 pin이 있어야 performance option이 실제 object lifetime을 어떻게 바꾸는지 설명할 수 있다.

**반증·rollback·terminal을 generation으로 닫는다.**

첫 regression fixture는 event pool에서 E7을 완료 상태로 만든 뒤 새 generation record를 지연한다. cleanup가 old ready를 current ready로 읽지 않아야 한다. event generation mismatch metric은 증가할 수 있지만 K17은 free queue에 들어가면 안 된다. current Done17 completion 뒤에만 reusable transition가 일어난다.

둘째 fixture는 two readers다. S1과 S2 completion gates를 따로 두고 S1만 연다. refcount가 0이어도 K17 reuse는 금지된다. S2 gate를 연 뒤 delayed consumer set이 empty가 되고 한 번만 allocator return effect가 실행된다. duplicate callback도 current generation을 다시 free하지 못한다.

셋째 fixture는 graph bucket capacity 8, active 5에서 rows 5–7에 poison sentinel을 둔다. output이 active rows만 포함하고 poison reader counter가 0인지 본다. 다음 replay 전에 metadata overwrite를 시도해 WAR guard가 wait 또는 alternate buffer를 선택하는지 확인한다.

넷째 fixture는 abort 시점을 launch 전, launch 직후, Done event 전, Done event 후로 나눈다. launch 전에는 graph slot을 즉시 반환할 수 있다. launch 뒤에는 request output owner를 terminal로 바꿔도 slot과 buffers는 delayed state다. Done 뒤 callback가 allocator return를 한 번 수행한다. 모든 위치에서 pointer-generation partition가 겹치지 않아야 한다.

운영 완화는 graph mode를 끄거나 batch bucket을 줄이는 것으로 race window를 낮출 수 있다. 그러나 일반 async eager kernel도 같은 lifetime 문제를 가질 수 있으므로 root fix로 선언하지 않는다. 불확실한 generation은 worker-local pool과 함께 폐기하고, device-wide synchronize는 긴급 진단 또는 안전한 drain에 제한한다.

정확한 fix는 producer-ready event와 consumer-done event를 구분하고, delayed record에 buffer/event/replay generations를 넣으며, completion effect를 idempotent하게 소비하는 것이다. shared graph inputs에는 overwrite guard를 둔다. event pooling은 handle 재사용 전에 previous record consumer가 닫혔음을 보장한다.

90분 soak는 active batch 1/5/8, graph hit/miss, abort, preemption, two-stream copy를 섞는다. wrong-answer sentinel, stale event generation, duplicate release, buffer overlap, hang가 0이어야 한다. eager fallback와 graph replay의 output equality, ITL/p99, synchronization overhead도 함께 본다.

incident terminal은 최초 불일치를 구체적으로 적는다. “cleanup가 producer Ready event E7의 과거 완료를 consumer Done으로 오인해 A17 reader 전 K17을 반환했다.” 그 다음 “same pointer K18 claim 뒤 stale callback가 current generation을 변경하지 못하도록 했다”고 적는다. host return, event completion, visibility, lifetime을 다시 하나의 `done` bool로 합치지 않는다.

## 43.9 CUDA 비동기 계약을 dependency ledger로 검산한다

### 43.9.1 API 이름 대신 반환 조건과 보호 범위를 적는다

ledger 첫 열은 operation이다. async copy, kernel launch, event record, stream wait, event query, graph launch를 한 행씩 둔다. 둘째 열은 host call return가 보장하는 것, 셋째는 device completion predicate, 넷째는 ordering scope, 다섯째는 protected allocations다. `async`라는 형용사 하나로 행을 합치지 않는다.

kernel launch return는 launch submission 단계의 host 결과다. 이후 asynchronous launch error나 kernel execution failure가 다른 synchronization/query 지점에서 관측될 수 있다. 따라서 launch 직후 request object를 지우는 cleanup와 device work lifetime은 분리된다. 오류 확인 API를 호출했다는 사실도 해당 buffer의 모든 consumer가 끝났다는 증거와 같지 않다.

event record는 그 event가 해당 stream의 그 위치를 대표하도록 operation을 넣는다. record 호출 반환 시 event completion을 가정하지 않는다. event query success는 그 record generation 앞의 work completion을 뜻하도록 사용해야 한다. pooled handle이 다시 record됐다면 어떤 generation을 query consumer가 기대하는지 ledger가 필요하다.

stream wait event는 host를 block하는 장치가 아니라 destination stream의 뒤 work에 dependency를 넣는 장치다. wait API가 반환됐다는 이유로 source event가 이미 complete했다고 쓰지 않는다. destination work의 start frontier가 source completion 뒤로 이동한다. 이 distinction이 host latency와 device correctness를 동시에 설명한다.

stream synchronize와 device synchronize의 scope도 다르다. 하나의 stream을 기다리는 것이 다른 stream의 readers까지 닫지 않을 수 있다. device-wide synchronization는 넓은 진단 fence가 될 수 있지만 정상 hot path의 owner model을 대신하면 concurrency를 잃는다. 필요한 consumer set에 맞춘 event joins를 우선한다.

memory allocation/free API의 semantics는 allocator와 사용 방식에 따라 다르므로 이름만으로 sync를 추론하지 않는다. caching allocator가 pointer를 pool에 돌리는 host mutation과 physical address가 다음 generation에 claim 가능한 시점을 구분한다. stream-ordered allocator를 쓴다면 allocation/free operation의 stream order와 cross-stream dependencies를 함께 기록한다.

### 43.9.2 두 stream·세 buffers를 표로 손검산한다

fixture C는 input I17, workspace W17, output O17을 쓴다. S0은 I17 copy 뒤 ReadyI를 record한다. S1은 ReadyI를 wait하고 kernel K가 I17을 읽고 W17을 읽고 쓰며 O17을 쓴다. S2는 K completion event DoneK를 wait하고 O17을 host-visible staging으로 copy한다. 각 buffer의 last consumer가 다르다.

I17은 K read가 끝나야 reusable이다. W17도 K가 끝나야 reusable이지만 graph exec가 다음 replay에서도 동일 workspace를 요구하면 replay slot owner가 추가된다. O17은 S2 copy가 끝나야 reusable이다. DoneK만 기다려 O17을 free하면 S2가 old output을 읽는 동안 새 generation이 덮을 수 있다.

시간을 넣는다. I copy는 0–20 µs, K는 ReadyI 뒤 20–50 µs, O copy는 DoneK 뒤 50–65 µs다. I/W reuse frontier는 50 µs, O reuse frontier는 65 µs다. 모든 buffers를 request finish 시각 50 µs에 한꺼번에 free하는 cleanup는 O만 조기 반환한다. aggregate request status가 buffer별 lifetime을 가린다.

S2 copy와 별도로 metrics kernel이 O17을 읽어 70 µs에 끝난다면 O frontier는 max(65,70)=70이다. consumer count 또는 explicit join event가 두 readers를 닫아야 한다. “마지막으로 enqueue한 stream” 하나를 찾는 방식은 concurrent branches에서 성립하지 않는다.

event pooling을 포함하면 ReadyI, DoneK, DoneO 각각의 role을 기록한다. ReadyI는 producer readiness, DoneK는 kernel consumer completion, DoneO는 output transfer completion이다. object handle을 재사용해도 role/generation record는 섞이지 않는다. callback는 role에 맞는 owner edge만 해제한다.

### 43.9.3 graph lifecycle을 capture부터 destroy까지 펼친다

graph lifecycle은 `uncaptured → capturing → graph-created → exec-instantiated → replay-enqueued → replay-complete → update-or-destroy`로 읽는다. capture begin 호출 뒤 stream에 허용되지 않는 operation이 들어가면 capture invalidation가 생길 수 있다. capture end가 성공해야 graph object가 생기고 instantiate가 성공해야 executable이 생긴다.

instantiate 성공은 replay가 완료됐다는 뜻이 아니라 아직 replay 전이다. graph launch return는 replay enqueue다. replay-complete evidence가 있어야 captured buffers의 해당 slot을 overwrite하거나 exec lifetime mutation을 안전하게 진행할 수 있다. 여러 in-flight replay를 허용하면 generation별 slots와 completions가 필요하다.

exec update는 topology와 node parameters의 compatibility 판정을 가진다. update 실패 뒤 old exec가 여전히 어떤 parameters를 갖는지 확인하고 fallback path를 선택한다. update return를 무시하고 new pointers가 반영됐다고 가정하면 graph는 old generation addresses를 재사용한다. 성공/실패 result와 current exec generation을 함께 기록한다.

destroy도 host object 정리와 in-flight device use frontier를 혼동하지 않는다. source가 destroy 전에 synchronize하는지, deferred destruction을 쓰는지, higher-level owner가 launches를 drain하는지 확인한다. request abort가 graph exec 전체를 destroy하는 구조인지 shared cache entry ref만 놓는 구조인지 구분한다.

capture buffer는 capacity와 active values를 나눈다. bucket 8 exec가 pointer arrays 8개를 capture하고 active 5를 runtime metadata로 주입하면 rows 5–7의 safe padding contract가 필요하다. 다음 replay active 8이 되면 metadata와 pointers가 모두 new generation으로 publish된 뒤 launch돼야 한다.

**vLLM graph key에서 allocation lifetime까지 다섯 질문을 던진다.**

첫 질문은 graph candidate key가 무엇을 포함하는가다. batch size만인지, uniform decode 여부, attention backend, shape/dtype, graph mode와 padding policy를 포함하는지 본다. key collision은 compatible하지 않은 exec를 재사용하게 만들 수 있다. option의 capture sizes는 inventory일 뿐 runtime compatibility 전체가 아니다.

둘째 질문은 input buffers를 누가 소유하는가다. wrapper 내부 static buffers인지 runner shared buffers인지, replay slot별 buffers인지 확인한다. request tensor의 pointer를 capture exec가 직접 보존한다면 request lifetime보다 graph lifetime이 길 수 있다. copy-in을 한다면 copy completion과 overwrite guard를 찾는다.

셋째는 replay generation publish다. runtime descriptors와 active batch를 어느 stream에서 update하고 graph launch가 그 write를 어떤 edge로 기다리는지 본다. host assignment가 끝났다는 사실만으로 asynchronous device copy/update visibility를 보장하지 않는다.

넷째는 output consumer다. graph completion 뒤 logits/sampling/output copy가 다른 stream에서 진행되는지, output buffer가 언제 next replay에 재사용되는지 본다. request finish notification가 output buffer completion보다 앞설 수 있다면 delayed owner가 필요하다.

다섯째는 abort와 fallback다. candidate predicate가 false여 eager path로 갈 때 graph slot reservation을 되돌리는지, graph launch 뒤 abort가 device work를 drain하는지, capture/update failure가 old exec state를 오염시키지 않는지 본다. 각 branch의 owner inverse를 적는다.

**SGLang WAR를 세 replay timeline으로 비교한다.**

R17은 shared input buffer X를 읽고, R18 host path는 같은 X에 다음 metadata를 쓴다. R17 read 완료 전에 R18 write가 시작되면 write-after-read hazard다. scheduler가 빠르게 다음 batch를 준비할수록 window가 커진다. graph mode가 throughput을 높여도 buffer handoff가 없으면 correctness가 깨진다.

timeline A는 single buffer와 explicit WAR event다. R17 graph 뒤 DoneRead17을 record하고 R18 input update stream이 이를 wait한다. correctness는 단순하지만 update가 serial frontier를 따른다. timeline B는 double buffer X0/X1로 alternating replay를 허용한다. 두 slots가 모두 in-flight면 다시 completion을 기다린다.

timeline C는 capacity bucket별 buffer pool이다. active/capture sizes와 backend signature로 slot을 고르고 generation을 claim한다. 같은 bucket의 concurrent replays가 같은 slot을 받지 않도록 ref/lease가 필요하다. pool hit가 buffer readiness와 동일하지 않다.

SGLang source walk에서는 `can_run_graph`가 true인 순간, capture values 복구, shared buffer write, graph replay, WAR event record/wait를 순서대로 적는다. scheduler request object가 terminal이 되는 시각과 slot release 시각을 따로 둔다. retraction/abort가 output을 버려도 replay reader는 완료돼야 한다.

관측은 synchronization을 새로 만들지 않아야 한다. 이 원칙은 별도 실행 단계가 아니라 앞선 모든 fixture에 공통으로 적용한다.

모든 request에서 event elapsed time을 얻으려고 blocking synchronize를 넣으면 race가 사라져 버리고 latency도 달라진다. anomaly-triggered event/query logging, stream sequence IDs, sampled replay generations를 사용한다. profiler run과 production behavior가 달라질 수 있음을 기록한다.

trace에는 host call begin/end, stream enqueue sequence, event record generation, wait edge, graph replay generation, buffer claim/release를 남긴다. device timestamps가 있으면 execution interval을 보완하지만 host와 device clocks를 단순 비교해 ordering을 만들지 않는다. dependency IDs가 happens-before 증거다.

metrics는 stale-event-generation, buffer-reuse-before-completion, duplicate-completion-consume, graph-update-failure, capture-fallback, WAR-wait-age를 둔다. graph hit rate와 latency만으로 correctness를 판단하지 않는다. correctness counters는 정상적으로 0이어야 하며 평균으로 희석하지 않는다.

buffer dump는 pointer만 남기지 않고 allocator epoch와 generation을 붙인다. 같은 pointer가 반복되는 것은 caching allocator에서 정상이다. 겹친 lifetime intervals가 문제다. event handle도 record generation과 protected buffer set을 붙여야 reuse를 정상과 stale completion으로 구분할 수 있다.

## 43.10 독자 점검표: 비동기를 edge와 lifetime로 읽는다

함수 호출 순서는 GPU completion 순서가 아니다. host return, stream enqueue, device completion, consumer visibility와 storage lifetime를 나누면 “먼저 호출했으니 준비됐다”는 오해가 사라진다. 같은 stream은 기본 ordering를 주지만 다른 streams 사이에는 explicit event edge가 필요하다.

fixture A에서 올바른 cross-stream 계약은 copy `S_copy` 뒤 E record, `S_compute`의 E wait 뒤 attention이다. event는 host timestamp가 아니라 producer stream frontier다. event record/wait 위치, buffer generation와 lifetime owner가 함께 맞아야 한다. wait 누락은 간헐적 stale/partial KV를 만들 수 있다.

barrier도 scope가 다르다. `__syncthreads()`는 CTA 안 threads의 협력을, async-copy barrier/pipeline은 stage readiness와 reuse를, stream event는 operations 사이 dependency를 다룬다. device-wide synchronize는 진단 도구가 될 수 있지만 좁은 edge를 대신하는 최종 설계가 아니다.

fixture B에서 graph capacity 8과 active 5를 분리했다. static address는 graph executable의 pointer contract이고 active metadata values는 replay generation의 계약이다. rows 0–4를 갱신하고 rows 5–7을 backend가 이해하는 sentinel/predicate로 무효화하며 metadata write→graph read ordering를 닫아야 한다. 37.5%는 logical padding fraction이지 measured GPU waste가 아니다.

vLLM은 capture sizes를 정렬된 config로 만들고 runtime mode와 batch descriptor로 wrapper를 dispatch한다. SGLang은 aligned capture buckets, rich `can_run_graph` predicate, shared metadata buffers와 scheduler/forward WAR edge를 관리한다. llama.cpp는 native capture/end/instantiate/launch/update와 multi-stream fork/join를 직접 보여 준다. 구현은 다르지만 key, address owner, value refresh, fallback와 dependency라는 질문은 같다.

option은 이름에서 끝나지 않는다. vLLM capture sizes는 effective config→candidate descriptors→runtime predicate→capture/replay/eager→selected descriptor 관측으로 이어진다. SGLang graph config와 WAR flags는 effective buckets/barrier state→support/event predicate→runner 또는 wait path→graph pass와 stream edge 관측으로 이어진다. 이 사슬이 닫혀야 옵션의 효과를 설명했다고 할 수 있다.

first divergence는 final wrong token이나 긴 latency보다 앞에 있다. stale active scalar, wrong sentinel, captured P0와 writer P1의 불일치, missing E wait, barrier participant 누락, update failure와 repeated instantiate처럼 시간 원장이 처음 모순되는 지점을 찾는다. source는 가능한 edge를 설명하고 trace는 현재 request가 실제로 그 edge를 지났는지 확인한다.

terminal 실습은 빈 표에서 시작한다. 요청 R43, allocation K17, copy stream S0, compute stream S1, output stream S2, graph replay G17을 행으로 둔다. 열은 host return, enqueue sequence, execution frontier, event generation, memory role, release condition다. source의 함수 호출 하나를 발견할 때마다 한 cell만 채운다. 증거 없이 `done`이라고 쓰지 않는다.

첫 행은 K17 allocation이다. pointer 0xA000, allocator epoch 4, generation 17, owner R43을 적는다. 다음 reusable 조건은 logical ref 0만이 아니라 last writer와 모든 readers completion이다. caching allocator가 동일 pointer를 K18에 줄 수 있으므로 pointer 값은 generation을 대신하지 않는다.

둘째 행은 S0 copy다. host call return 2 µs, stream sequence 301, device execution 10–30 µs라고 하자. Ready17 record는 sequence 302다. 공식 semantics에 기대는 범위는 S0에서 record 앞 operations completion이다. Ready17은 K17 producer readiness를 보호하지만 S1 reader completion을 보호하지 않는다.

셋째 행은 S1 wait와 attention이다. wait Ready17은 sequence 501, kernel A17은 502, DoneA17 record는 503이다. wait call의 host return를 completion으로 쓰지 않는다. A17 start frontier가 Ready17 completion 뒤라는 edge와 DoneA17이 A17 finish 뒤라는 edge를 각각 그린다.

넷째 행은 S2 output copy다. S2는 DoneA17을 wait하고 O17을 읽는다. K17을 읽지 않는다면 K17 lifetime에는 포함되지 않지만 O17 lifetime에는 포함된다. buffer별 consumer set을 따로 만드는 이유다. request 하나의 마지막 event로 모든 allocations를 free하지 않는다.

다섯째 행은 abort다. abort host 시각이 9 µs여도 S0/S1 commands는 이미 enqueue됐을 수 있다. output delivery owner는 즉시 terminal로 만들 수 있지만 K17 device owner는 DoneA17까지 delayed다. queue에서 제거 성공 여부와 device launch frontier를 source branch로 확인한다.

여섯째 행은 graph replay다. G17 launch return, exec generation, input slot generation, active rows, completion event를 적는다. graph cache key hit는 slot readiness를 증명하지 않는다. 같은 exec를 재사용해도 replay별 mutable buffers와 completion은 독립 identity를 가져야 한다.

표가 완성되면 forward 검산을 한다. K17 claim→S0 write→Ready17→S1 read→DoneA17→free 순서가 dependency edges로 이어져야 한다. wall-clock 로그가 우연히 이 순서인 것만으로 부족하다. wait/record 또는 same-stream order라는 program edge를 찾는다.

backward 검산은 K18 claim에서 시작한다. allocator가 무엇을 보고 0xA000을 reusable로 판단했는지 거슬러 올라간다. free-list insertion이 DoneA17 generation을 소비했는지, duplicate callback를 막았는지, abort refcount만 본 것은 아닌지 확인한다. backward walk는 조기 reuse를 빠르게 찾는다.

event object E7도 양방향으로 걷는다. record site가 어떤 producer operations 뒤인지 forward로 보고, query/callback가 어떤 protected effect를 release하는지 backward로 본다. 두 경로의 buffer generation이 다르면 stale event다. handle equality만으로 연결하지 않는다.

graph input update는 write→read edge를 검산한다. host가 metadata를 채운 뒤 device copy를 S0에 enqueue하고 graph가 S1에서 읽는다면 ReadyMeta event가 필요할 수 있다. captured graph 내부에 copy node가 포함됐다면 node dependency가 역할을 할 수 있다. 실제 topology를 source에서 확인한다.

다음 replay write는 read→write WAR edge를 검산한다. G17이 shared metadata를 다 읽었다는 completion 뒤 G18 writer가 시작해야 한다. producer-ready edge와 방향이 반대다. ReadyMeta 하나로 다음 overwrite까지 보호한다고 생각하지 않는다. double buffer면 slot lease가 이 edge를 대체한다.

공식 CUDA evidence를 붙일 때는 문서 문장을 구현 주장보다 좁게 사용한다. streams가 ordered operations를 표현하고 events가 dependency를 만들며 graph가 operations/dependencies를 capture한다는 semantics를 근거로 삼는다. vLLM이나 SGLang이 어느 buffer를 어느 event로 보호하는지는 고정 source가 증명해야 한다.

CUDA 12.9.1과 13.3 문서 표현이 달라도 이 장의 핵심 판정은 versioned source와 contract로 남긴다. upgrade audit에서는 event/graph API behavior, capture restrictions, memory allocation semantics, driver compatibility를 release notes와 함께 확인한다. 한 버전의 예제를 영구 보편 법칙으로 만들지 않는다.

vLLM fixture는 capture size 8, active 5, graph candidate C17로 기록한다. runtime descriptor가 rows 0–4를 current requests로 채우고 5–7을 안전하게 padding하는지 본다. wrapper call 뒤 output consumer가 끝나기 전에 shared output storage가 다음 candidate에 넘어가지 않는지 확인한다.

SGLang fixture는 capture bucket 8, scheduler batch 5, shared slot X0 generation 17로 쓴다. `can_run_graph` true는 compatibility evidence다. WAR event 또는 slot alternation은 lifetime evidence다. 둘 중 하나가 빠지면 graph 선택은 맞아도 shared value가 stale할 수 있다.

llama.cpp fixture는 auxiliary streams fork/join를 따른다. main stream event로 auxiliary streams를 출발시키고 각 auxiliary completion을 main이 다시 기다리는지 본다. 하나의 join이 누락되면 main stream 이후 consumer가 partial result를 볼 수 있다. graph instantiate/update/launch lifecycle도 같은 ledger에 넣는다.

반증 실험 1은 Ready event를 제거한다. copy를 artificial gate로 늦추고 attention을 먼저 진행시킨다. reference output과 달라져야 dependency가 실제로 필요함을 보여 준다. gate 없이도 항상 정상이라면 workload timing이 race를 가렸을 수 있으므로 contract가 생기는 것은 아니다.

반증 실험 2는 Done event를 Ready event로 바꾼다. producer copy는 완료됐지만 consumer attention은 막아 둔다. allocator가 K17을 반환하면 test가 실패한다. 이 fixture가 producer-ready와 consumer-done 혼동을 직접 잡는다.

반증 실험 3은 E7 handle을 generation 16 완료 상태에서 재사용한다. generation 17 record completion 전 query callback를 호출한다. K17 state가 변하지 않아야 하고 stale-generation counter만 증가해야 한다. current record 완료 뒤 effect는 정확히 한 번 실행된다.

반증 실험 4는 graph active rows를 8에서 5로 줄인다. rows 5–7에 poison pointers와 sentinel values를 두고 어떤 node도 이를 dereference하거나 output으로 publish하지 않는지 본다. capacity 안의 stale read도 correctness defect임을 확인한다.

반증 실험 5는 G17 read가 끝나기 전에 G18 metadata write를 시도한다. single buffer면 writer가 wait하고 double buffer면 X1을 선택해야 한다. X0을 즉시 덮으면 WAR invariant failure다. profiler로 인해 implicit sync가 생기지 않는 fake gates를 사용한다.

반증 실험 6은 graph update failure를 강제한다. old exec generation과 new arguments를 섞어 launch하지 않아야 한다. fallback가 eager를 선택하거나 safe reinstantiate를 수행하고, reserved slots와 temporary graph objects를 cleanup해야 한다. update result를 무시하는 branch를 test가 잡는다.

rollback ladder 첫 단계는 해당 graph signature를 eager로 보내는 것이다. correctness를 빠르게 회복하지만 eager path 자체의 multi-stream lifetime도 동일 fixture로 확인한다. 둘째는 shared buffer concurrency를 1로 제한한다. 셋째는 worker epoch를 drain/restart해 불확실 generations를 폐기한다.

device-wide synchronize는 긴급 diagnosis에서 race가 사라지는지 보는 반증 도구가 될 수 있다. 증상이 사라지면 missing edge 가설을 강화하지만 어느 edge인지 알려 주지는 않는다. production fix는 Ready/Done/WAR 각각의 좁은 dependency와 generation-bound release로 대체한다.

fix 배포 canary는 stale event, early reuse, duplicate release, graph update failure, fallback count를 본다. correctness counters는 0이어야 한다. graph hit rate, TTFT, ITL, CPU launch overhead를 함께 보아 과도한 synchronization 회귀를 찾는다. 성능이 좋아도 sentinel mismatch 하나면 승인하지 않는다.

90분 soak는 active 1/5/8, graph/eager, copy delay, abort timing, auxiliary stream 수를 바꾼다. 모든 allocation generation의 claim/release intervals가 겹치지 않고 event consumers가 유한 시간에 닫혀야 한다. shutdown drain 뒤 outstanding replay와 delayed buffers가 0인지 확인한다.

terminal 문서는 세 문장으로 시작한다. 첫째, E7 generation 16 completion을 K17 consumer completion으로 읽은 것이 최초 불일치였다. 둘째, K17을 DoneA17 전에 free해 K18과 A17 lifetime이 겹쳤다. 셋째, generation-bound Ready/Done records와 graph WAR slot lease로 겹침을 제거했다.

그 뒤 source evidence 표, six-time ledger, regression matrix, canary 결과를 붙인다. “event를 추가해 해결”이라고 쓰지 않는다. 어느 stream 어느 operation 뒤 event를 record하고 누가 wait/query하며 어떤 allocation generation release를 허용하는지 적는다.

독자가 terminal을 재검산할 수 있어야 완료다. trace에서 sequence 301→302→501→502→503 edge를 찾고, K18 claim이 503 completion 뒤임을 확인한다. source에서 같은 dependency를 만드는 call path를 찾고, fault fixture에서 edge 제거가 실패를 재현하는지 본다.

마지막으로 과동기화 반증을 한다. device synchronize를 제거해도 narrow edges 아래 output equality와 lifetime invariants가 유지되는지 확인한다. ITL이 회복되고 correctness counters가 0이면 안전성과 concurrency가 함께 닫힌다. 이 단계가 없으면 race를 global serialization로 숨긴 임시 완화에 머문다.

운영자가 trace 한 건을 읽는 순서도 고정한다. 먼저 buffer generation의 claim과 release를 찾고, 그 사이 readers/writers를 나열한다. 다음으로 각 consumer의 completion evidence를 찾는다. 마지막에 request status와 client cancellation을 겹친다. request log부터 보면 terminal status가 device lifetime까지 끝냈다는 선입견이 생길 수 있다.

buffer K17에는 S0 writer와 S1 reader가 있다. writer Ready17은 S1 start를 허용하고 reader DoneA17은 allocator release를 허용한다. 두 events가 같은 handle E7을 시간차로 공유해도 ledger record는 다르다. callback payload가 role과 generation을 잃지 않아야 한다.

output O17에는 graph writer와 S2 copy reader, sampling reader가 있을 수 있다. graph completion만으로 O17을 반환하지 않는다. S2와 sampling이 각자 done edge를 내거나 join owner가 둘을 모은다. sampling이 CPU로 결과를 복사하지 않고 같은 GPU stream에서 이어져도 실제 stream ordering을 확인한다.

workspace W17은 output과 다르게 graph exec/cache owner가 장기간 보유할 수 있다. replay 하나가 끝나도 exec가 다음 replay에 같은 workspace address를 기대한다면 allocator free 대상이 아니다. graph cache eviction 또는 exec destroy가 workspace lifetime 종료를 소유한다. request cleanup와 cache cleanup를 섞지 않는다.

event pool도 별도 object lifetime을 가진다. event record를 기다리는 callbacks가 남았는데 handle을 pool에 반환하면 next borrower가 record/query state를 바꿀 수 있다. event lease는 record generation consumer가 닫힌 뒤 반환된다. buffer completion과 event object 반환을 같은 bool로 두지 않는다.

graph cache eviction 사고도 검산한다. cache entry ref가 0이 됐다고 exec를 destroy했지만 replay G17이 in-flight라면 object lifetime 위반이다. entry는 logical eligible이 되고 replay completion 뒤 physical destroy한다. 이 구조는 KV block의 refcount 0과 delayed writer 구분과 같은 원리다.

shutdown은 새 graph launches를 먼저 막는다. 그 뒤 queued updates/captures를 terminal로 만들고 in-flight replays를 drain하며 buffer/event generations를 회수한다. timeout이 끝났다고 개별 pointers를 normal pool에 돌리지 않는다. completion frontier를 잃었으면 worker allocator epoch 전체를 폐기한다.

multi-GPU graph에서는 rank별 streams와 collective consumers가 추가된다. local kernel Done event가 complete해도 NCCL operation이 buffer를 읽고 있을 수 있다. communicator stream dependency와 collective completion까지 owner set에 넣는다. rank 0 request status만으로 모든 ranks buffers를 반환하지 않는다.

CUDA Graph 안에 collective node나 external semaphore가 있으면 captured dependency topology를 확인한다. graph node completion과 외부 system acknowledgment의 경계가 다를 수 있다. 이 장은 local CUDA 중심이므로 실제 connector/collective contract는 해당 source에서 추가 증명하고 추측하지 않는다.

host memory lifetime도 빠뜨리지 않는다. async H2D copy source가 pageable/pinned staging인지, API가 source reuse를 언제 허용하는지 공식 contract를 확인한다. Python tensor object가 scope를 벗어났다는 사실과 underlying storage copy completion이 다를 수 있다. binding layer가 keep-alive owner를 가지는지 본다.

graph node argument storage는 값 복사인지 pointer 참조인지 API별로 확인한다. temporary host structure를 capture/instantiate 호출 직후 버려도 되는지 문서 contract가 결정한다. device pointer가 가리키는 allocation lifetime과 host parameter structure lifetime을 구분한다.

memory visibility를 cache flush라는 막연한 말로 설명하지 않는다. producer/consumer operations와 synchronization edge를 적는다. host가 device 결과를 읽는다면 copy/synchronization contract를 확인한다. managed memory나 mapped host memory는 별도 consistency rules가 있으므로 이 fixture의 device buffers 규칙을 그대로 확대하지 않는다.

error path도 ledger에 남긴다. event record 실패, graph launch 실패, asynchronous execution error, update failure는 각각 어느 owners가 생성됐는지 다르다. enqueue 전 실패면 즉시 cleanup가 가능할 수 있고 enqueue 여부가 불명확하면 conservative drain이 필요하다. 오류 문자열 하나로 rollback를 고르지 않는다.

metrics의 `in_flight_graphs=0`도 단독 증거가 아니다. counter decrement가 launch return에 있는지 completion callback에 있는지 source를 본다. delayed buffers, auxiliary streams, output copies가 남을 수 있다. shutdown terminal은 owner sets와 completion frontier를 함께 검사한다.

최종 regression은 event pooling on/off, graph caching on/off를 교차한다. pooling off에서만 정상이라면 event generation 관리가 의심되고 graph off에서만 정상이라면 replay buffer/WAR를 본다. 둘을 동시에 끄면 root branch를 분리하기 어렵다. 최소 변화로 가설을 반증한다.

성능 승인에서는 추가 events 수와 wait duration을 기록한다. correctness fix가 모든 streams를 직렬화했다면 ITL tail이 커진다. producer-ready, consumer-done, WAR edges가 실제 data dependency만 연결하는지 확인한다. 독립 requests와 graph slots는 가능한 한 겹쳐 실행되어야 한다.

이 마지막 원장을 완성하면 “CUDA는 비동기라 위험하다”는 막연한 결론을 벗어난다. 위험한 것은 비동기 자체가 아니라 dependency와 lifetime identity가 빠진 것이다. 명시적인 generation과 edge가 있으면 host와 device가 겹쳐 실행돼도 정확한 재사용 frontier를 설명하고 검증할 수 있다.

독자는 마지막으로 E43 표의 합계를 확인한다. K17 generation에는 writer 하나, readers 하나 이상, allocator owner 하나가 있다. 각 owner는 생성 event와 unique release event를 가진다. Ready17은 reader start를 허용하지만 owner를 제거하지 않고, DoneA17은 attention reader를 제거한다. 모든 consumer가 사라진 뒤 allocator owner가 reusable queue로 handoff한다.

이 합계가 맞아도 시간 edge가 틀릴 수 있으므로 interval을 함께 본다. K18 claim interval이 K17 reader interval과 한 µs라도 겹치면 use-after-free다. event callbacks가 최종 refcount 0을 만들었더라도 overlap이 존재하면 실패다. 상태 conservation과 temporal ordering은 서로 대신하지 못한다.

반대로 events가 많아도 owner identity가 틀릴 수 있다. DoneA16 completion을 DoneA17로 읽거나 graph slot X0 generation 9를 X0 generation 10으로 읽으면 ordering primitive 자체는 정상 작동한다. 잘못 연결한 edge가 정확하게 실행될 뿐이다. 그래서 handle, pointer, request ID만 아니라 generation tuple을 기록한다.

리뷰 승인자는 source link가 실제 predicate와 mutation을 가리키는지 확인한다. option declaration이나 함수 이름만 pin하면 lifetime을 증명하지 못한다. graph selection, buffer update, launch, event record/wait, cleanup consumer의 줄을 연결한다. 공식 문서는 primitive semantics를, project source는 owner usage를, trace는 발생 instance를 각각 증명한다.

이 세 evidence가 일치하고 fault injection가 first divergence를 재현하며 rollback 뒤 90분 soak가 닫힐 때 사건은 완료다. 하나라도 빠지면 “재현되지 않음”이나 “동기화 추가”라는 임시 상태로 남긴다. 다음 release에서 같은 주소와 event가 재사용될 때 silent corruption가 돌아오지 않게 terminal contract를 회귀 suite에 보존한다.

terminal trace에는 정상 eager와 graph replay를 같은 generation ledger로 비교한 결과도 남긴다. graph를 끌 때만 안전하면 captured buffer 또는 WAR owner가 빠진 것이고, 둘 다 실패하면 공통 allocator/event completion 경계를 다시 본다. 이 분기가 수정 범위를 불필요하게 넓히지 않도록 지켜 준다.

다음 장은 이 시간 계약 아래 실행되는 code object를 본다. PTX, SASS와 cubin이 어떤 architecture와 toolkit/driver 계약에서 선택되는지 이해해야 같은 graph node 이름 아래 실제 instruction path가 달라지는 이유를 설명할 수 있다. 45장에서는 그 kernel 내부의 attention tile과 online softmax를 다시 연다.

## 43.11 한 요청의 시간 계약을 닫는다

### 43.11.1 증상에서 먼저 열 source를 고른다

wrong token이 graph replay에서만 생긴다면 CUDA 문서를 처음부터 읽지 않고 graph key→input update→launch→다음 overwrite guard 순서로 vLLM 또는 SGLang source를 연다. 같은 pointer가 다른 request generation을 가리키는 증상이면 llama.cpp의 graph update와 multi-stream event fork/join을 먼저 잇는다. graph를 꺼도 재현되면 capture 목록을 내려놓고 CUDA stream·event의 producer-ready와 consumer-done 범위를 확인한다.

질문은 세 가지면 된다. 실행 골격 선택이 틀렸는가, current metadata publish와 replay 사이 edge가 빠졌는가, 마지막 consumer 전에 storage를 재사용했는가. 아래 링크는 이 순서의 증거 좌표다. CUDA 공식 문서는 반환·완료·ordering 범위를, framework source는 graph 선택과 buffer owner를, llama.cpp source는 native graph와 event join의 실제 mutation을 확인할 때 연다.

### 43.11.2 질문별 Reference/source note

- [NVIDIA CUDA C++ Programming Guide 12.9.1 — asynchronous execution, streams, events, CUDA Graphs](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html)
- [NVIDIA CUDA Programming Guide 13.3.0 — asynchronous execution and CUDA Graphs](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [vLLM v0.27.1 — `CompilationConfig.post_init_cudagraph_sizes`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/config/compilation.py#L1106-L1132)
- [vLLM v0.27.1 — `CUDAGraphWrapper.__call__` capture/replay dispatch](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/compilation/cuda_graph.py#L145-L330)
- [vLLM v0.27.1 — graph candidate construction and capture](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu/cudagraph_utils.py#L179-L329)
- [SGLang v0.5.18 — `get_batch_sizes_to_capture`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py#L65-L102)
- [SGLang v0.5.18 — decode `can_run_graph`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L648-L735)
- [SGLang v0.5.18 — decode graph capture lifecycle](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L997-L1155)
- [SGLang v0.5.18 — scheduler stream and WAR barrier](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L1685-L1720)
- [llama.cpp v0.2.0 — graph update and instantiate](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/ggml-cuda.cu#L2618-L2648)
- [llama.cpp v0.2.0 — multi-stream event fork/join](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/ggml-cuda.cu#L4023-L4120)
- [llama.cpp v0.2.0 — stream capture, instantiate and graph launch](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/ggml-cuda.cu#L4188-L4220)
- [llama.cpp v0.2.0 — begin capture and native event wait](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/ggml-cuda.cu#L4288-L4315)

### 43.11.3 종합 회고: 한 요청의 시간을 끝까지 소유하라

이 장을 한 문장으로 압축하면 “GPU 비동기 실행은 호출 목록이 아니라 happens-before edge와 buffer lifetime의 이야기”다. 단일 요청 R43에서 scheduler가 metadata를 쓰고, copy stream이 KV generation을 생산하고, event가 그 producer frontier를 표시하고, compute stream이 wait 뒤 attention을 실행하며, 마지막 consumer completion 뒤에야 storage를 재사용한다. 어느 한 단계라도 request ID·buffer range·allocation generation·value generation이 다른 대상을 가리키면 event나 barrier가 존재해도 계약은 닫히지 않는다.

CUDA Graph replay도 별도의 마법이 아니다. 고정된 graph executable과 static address는 실행 골격을 재사용할 뿐, active batch·row validity·slot mapping·sequence length 같은 값의 세대를 자동으로 갱신해 주지 않는다. 그래서 capacity 8에 active 5를 넣는 fixture는 rows 0–4의 current generation, rows 5–7의 invalid sentinel, metadata write→launch edge, 이전 replay read→다음 write의 WAR edge를 모두 요구한다. graph hit라는 관측만으로 이 네 계약 가운데 어느 것도 대신 증명할 수 없다.

조사자는 최종 wrong token이나 긴 latency에서 거슬러 올라가기보다 최초로 모순되는 edge를 찾는다. producer operation 앞에 record된 event, consumer launch 뒤에 놓인 wait, captured P0와 writer P1의 불일치, divergent CTA barrier, 아직 in flight인 buffer의 조기 반환이 그 first divergence다. 임시 device-wide synchronize가 증상을 감춰도 정확한 범위의 edge와 release condition을 복구하지 못했다면 수정은 끝나지 않았다.

따라서 마지막 손검산 질문은 다섯 개면 충분하다. 누가 이 buffer generation을 썼는가, 어느 frontier에서 생산이 완료됐는가, 누가 어떤 stream과 graph generation에서 읽는가, 그 consumer는 무엇을 기다렸는가, 어떤 completion 뒤에 주소와 값을 다시 써도 되는가. 이 답을 source predicate와 request trace 양쪽에서 맞출 수 있을 때 barrier·stream·event·CUDA Graph는 서로 떨어진 기능명이 아니라 하나의 설명 가능한 시간 계약이 된다.
