# 38장. CPU offload와 계층형 cache: 용량을 늘리고 시간을 잃지 않는 법

GPU 메모리가 모자라면 일부 KV cache를 CPU 메모리로 옮기면 된다는 설명은 절반만 맞다. 옮긴 뒤에도 그 요청이 다시 선택될 수 있고, 그때 attention kernel은 CPU 주소가 아니라 GPU에서 읽을 수 있는 물리 주소를 요구한다. 따라서 offload는 단순한 `copy`가 아니다. 어느 블록을 내릴지 고르고, 복사가 끝날 때까지 원본을 보호하고, 호스트 사본의 소유권을 확정하고, 필요할 때 다시 올린 뒤, compute stream이 복원 완료를 관측하게 만드는 수명주기다.

이 장을 읽고 나면 독자는 “GPU OOM이 줄었다”는 한 줄짜리 성공 판정 대신 세 질문을 던질 수 있어야 한다. 내려 쓴 KV가 다시 쓰일 확률은 얼마인가. 다시 계산하는 것과 PCIe를 왕복하는 것 중 어느 쪽이 더 싼가. 그리고 복사가 끝났다는 사실을 어느 상태와 event가 증명하는가. 입문 경로의 독자는 38.1~38.4절을 먼저 읽고, 운영자는 38.7절에서 증상별 분기를 따라가면 된다. 소스 경로와 CUDA 경로의 독자는 38.5~38.6절에서 Python manager가 pointer, stream, event로 내려가는 지점을 확인한다.

## 38.1 OOM은 사라졌는데 응답이 더 나빠진 밤

한 서비스가 24GiB를 KV cache에 쓰고 있다고 하자. 긴 대화가 늘어 GPU block 여유가 거의 사라지자 운영자는 96GiB짜리 host pool을 붙였다. 대시보드에서 GPU OOM은 사라졌다. admission도 더 많은 요청을 받아들였다. 그런데 그날 저녁부터 inter-token latency, 즉 연속 토큰 사이 시간의 p99가 수백 밀리초씩 튀었다. GPU 사용률은 오히려 짧게 꺼졌다가 다시 올랐다. CPU 사용률도 높지 않았다.

처음 떠올리기 쉬운 가설은 “CPU가 느려서”다. 하지만 이 말은 조사에 거의 도움이 되지 않는다. CPU에서 attention을 실행한 것이 아니라, CPU 메모리에 있던 KV를 GPU로 복원하느라 compute가 기다렸을 수도 있다. 복원 자체는 비동기였지만 같은 PCIe 링크에서 여러 요청이 경쟁했을 수도 있다. scheduler가 host hit를 곧바로 runnable로 해석하여 아직 완료되지 않은 block을 기다리는 요청을 batch에 넣었을 수도 있다. 세 경우는 모두 ITL이 늘지만 고쳐야 할 상태와 관측점이 다르다.

사건을 더 구체화해 보자. 요청 A는 긴 prefix를 prefill한 뒤 잠시 선점되었다. cache manager는 A의 뒤쪽 KV 2GiB를 host tier로 내리기 시작했다. 동시에 요청 B와 C도 2GiB씩 내렸다. A가 다시 선택됐을 때 metadata lookup은 host에 key가 있다고 답했다. scheduler는 이것을 cache hit로 집계했다. 그러나 A가 실제로 decode하려면 2GiB를 다시 GPU에 올려야 했다. 세 요청의 store와 한 요청의 load가 같은 링크와 copy engine을 다투었다. “hit”라는 논리 판정과 “kernel이 지금 읽을 수 있다”는 물리 판정 사이에 수십~수백 밀리초가 숨어 있었다.

여기서 첫 번째 불변조건이 나온다. **host hit는 GPU-ready가 아니다.** host tier에서 logical key를 찾았다는 것은 복원 후보를 찾았다는 뜻일 뿐이다. GPU block을 예약하고 payload를 복사하고 block table 또는 이에 해당하는 metadata를 새 주소로 commit하고, compute stream이 copy 완료를 관측해야 비로소 사용할 수 있다. 이 중 하나라도 빠지면 느린 것이 아니라 오답이나 use-after-free가 된다.

두 번째 불변조건은 **GPU에서 block을 재사용할 수 있게 된 시점과 host store가 durable해진 시점을 뒤섞지 않는 것**이다. 비동기 복사를 launch하자마자 GPU block을 다른 요청에 배정하면 copy engine이 새 내용과 옛 내용을 섞어 읽을 수 있다. 반대로 복사는 끝났지만 host metadata commit이 안 됐다면 lookup은 존재하지 않는다고 답해 비싼 prefill을 다시 실행한다. payload와 metadata는 함께 하나의 전환을 이루지만, 완료 순서는 명시적으로 관리해야 한다.

이 문제를 도서관에 비유하면 첫 직관을 얻을 수 있다. 자주 읽는 책은 책상 위 GPU tier에, 덜 읽는 책은 서고 host tier에 둔다. 책상은 작지만 손이 바로 닿고, 서고는 크지만 가져오는 시간이 든다. 그러나 이 비유는 금세 한계에 닿는다. 컴퓨터에서 책을 옮기는 동안 원본 페이지가 바뀔 수 있고, 같은 책 제목이 새 판본 generation을 가리킬 수 있으며, 운반 완료를 기다리지 않은 독자가 반쯤 복사된 페이지를 읽을 수 있다. 실제 cache에는 logical identity, physical slot, generation, reference count, stream ordering이 모두 필요하다.

또 하나의 함정은 weight offload와 KV offload를 같은 기능으로 부르는 것이다. weight offload는 모델 가중치 tensor의 placement를 바꾼다. 대개 layer 실행 순서에 맞춰 weight를 가져오거나 CPU에서 일부 연산을 수행하는 선택과 연결된다. KV offload는 요청마다 자라고 줄어드는 runtime state를 다룬다. 요청 취소, prefix 공유, eviction, 재복원, block generation이 개입한다. 둘 다 GPU 메모리를 아끼지만 key, lifetime, reuse pattern과 correctness 조건이 전혀 다르다.

## 38.2 2GiB를 옮기는 데 드는 83밀리초의 의미

계산의 출발점은 단순하다. payload가 2GiB이고 PCIe 실효 대역폭이 24GB/s라면 전송 시간의 하한은 payload를 bandwidth로 나눈 값이다. 단위부터 맞춰야 한다. 2GiB는 `2 × 2^30 = 2,147,483,648`바이트다. 여기서 24GB/s를 십진 단위인 `24 × 10^9`바이트/초로 놓으면 다음과 같다.

```text
T_copy_lower = 2,147,483,648 byte / 24,000,000,000 byte/s
             ≈ 0.0895 s
             ≈ 89.5 ms
```

운영자가 2GB를 `2 × 10^9`바이트로 단순화하면 약 83.3ms가 나온다. 두 값 중 하나만 “정답”으로 외우기보다 용량 표기의 GiB와 링크 표기의 GB가 서로 다른 단위를 쓰는지 밝히는 것이 중요하다. 이 장에서는 흔히 말하는 83ms를 대략적인 직관으로 쓰되, 실제 fixture의 2GiB를 엄밀히 대입한 하한은 약 89.5ms라고 구분한다. 어느 쪽이든 한 decode step의 정상 ITL보다 훨씬 크다.

이 수치는 payload만 움직이는 이상적인 하한이다. GPU block을 찾는 시간, host slot 예약, descriptor 작성, page pin, kernel 또는 DMA launch, event 기록, 완료 polling, metadata commit은 포함하지 않았다. 같은 PCIe root complex에서 NIC, NVMe, 다른 GPU 전송이 경쟁하는 시간도 빠졌다. 따라서 프로파일에서 120ms가 보였다고 “24GB/s가 나오지 않는다”라고 곧바로 결론내릴 수 없다. 먼저 실제 bytes와 queue 대기, 전송 구간을 분리해야 한다.

왕복이면 비용은 더 커진다. 2GiB를 한 번 내렸다가 다시 올리면 payload 전송만 약 179ms다. store는 요청의 critical path 밖에서 겹칠 수 있지만 load는 그 요청이 다시 실행되기 전에 끝나야 한다. 운영상 더 중요한 값은 왕복 평균이 아니라 **복원 critical path에 남은 노출 시간**이다.

```text
T_exposed_restore = max(0, T_queue + T_H2D + T_commit - T_overlap)
```

다음으로 재계산과 비교한다. 어떤 prefix suffix를 GPU에서 다시 prefill하는 데 55ms가 걸리고 host에서 복원하는 데 queue를 포함해 110ms가 걸린다면, host hit는 계산을 피했어도 latency를 악화시킨다. 반대로 같은 suffix 재계산이 240ms이고 restore가 100ms라면 복원이 유리할 수 있다. 중요한 것은 “cache hit이므로 이득”이 아니라 같은 workload에서 `T_restore`와 `T_recompute`를 비교하는 것이다.

reuse 확률도 포함해야 한다. 내려 쓴 block이 다시 요청되지 않으면 store 비용과 host 용량만 들고 load는 없다. store가 background에서 완전히 겹쳤다고 하더라도 PCIe와 pinned memory라는 공유 자원을 소비한다. 간단한 기대 비용은 다음처럼 적을 수 있다.

```text
E(offload) = T_store_exposed + P(reuse) × T_restore_exposed
E(recompute) = P(reuse) × T_recompute
```

offload의 latency 측면 손익분기는 `E(offload) < E(recompute)`일 때다. 여기에 offload가 OOM을 막아 거절 요청을 줄이는 capacity 이득은 별도로 더해야 한다. goodput 관점에서는 약간 느려져도 더 많은 요청을 deadline 안에 끝낼 수 있다. 그러나 ITL SLO가 단단한 대화형 서비스에서 restore가 decode의 critical path에 노출되면 admission 증가가 goodput 증가로 이어지지 않을 수 있다.

24GiB GPU pool과 96GiB host pool을 합쳐 “cache가 120GiB가 됐다”고 말하는 것도 부정확하다. GPU-resident bytes만 kernel이 직접 소비할 수 있다. host pool은 대기실이며, 동시에 restore할 GPU destination block이 필요하다. GPU가 완전히 찬 상태에서 새 block을 올리려면 먼저 다른 block을 내리거나 버려야 한다. 이 과정에서 load와 store가 맞물려 thrashing이 생길 수 있다. 따라서 capacity는 단순 합이 아니라 resident working set, 이동 가능 block, destination reserve와 정책에 의해 제한된다.

## 38.3 demote와 restore를 상태 기계로 읽기

offload 구현을 읽을 때 `copy_to_cpu()` 같은 함수 이름만 찾으면 핵심을 놓친다. copy 앞뒤에서 누가 주소를 소유하며 scheduler는 어떤 상태만 실행 가능하다고 보는지를 따라가야 한다. 최소 상태를 다음처럼 둘 수 있다.

```text
GPU_RESIDENT
  └─ demote 선택 + host slot 예약 → STORE_PENDING
STORE_PENDING
  ├─ copy/event 성공 → HOST_RESIDENT
  └─ 실패/취소 → GPU_RESIDENT 또는 INVALID
HOST_RESIDENT
  └─ lookup + GPU slot 예약 → LOAD_PENDING
LOAD_PENDING
  ├─ copy/event 성공 + metadata commit → GPU_RESIDENT
  └─ 실패/취소 → HOST_RESIDENT 또는 INVALID
```

실제 구현의 이름은 다르지만 네 경계는 공통적으로 확인해야 한다. 첫째, **prepare**는 목적지 공간을 예약하고 transfer descriptor를 만든다. 아직 payload가 안전하게 존재한다는 뜻은 아니다. 둘째, **submit**은 stream에 복사를 넣지만 비동기 API라면 함수 반환이 완료를 뜻하지 않는다. 셋째, **complete**는 event 또는 worker 결과를 확인하고 metadata를 commit한다. 넷째, **release**는 이전 소유권을 버리되 outstanding transfer가 참조하는 동안에는 물리 메모리를 재사용하지 않는다.

demote를 한 요청 A의 block 17로 추적해 보자. lookup key는 단순한 GPU slot 17이어서는 안 된다. slot 17은 곧 다른 요청에 재할당될 수 있다. key에는 token block hash, model·adapter·cache group처럼 payload의 의미를 구분하는 정보가 필요하다. physical descriptor는 source GPU block, destination host offset, byte length와 stride를 가리킨다. transfer가 끝나면 host entry가 logical key와 destination을 결합한다. 그 후에야 GPU block의 eviction 또는 재사용이 안전하다.

restore는 역순처럼 보이지만 완전한 대칭이 아니다. host lookup이 hit를 반환해도 GPU destination을 확보하지 못하면 시작할 수 없다. GPU slot이 준비되면 host physical address에서 GPU cache layout에 맞는 위치로 옮긴다. copy 완료 후 block table을 commit하고 compute stream에 의존성을 연결한다. 그 다음 scheduler가 A를 runnable batch에 넣는다. scheduler가 LOAD_PENDING을 GPU_RESIDENT와 동일하게 취급하면 compute가 미완료 payload를 읽는다.

reference count는 이 전환을 더 까다롭게 만든다. 여러 요청이 같은 prefix block을 공유하면 한 요청이 끝났다고 payload를 지울 수 없다. demote 중에도 source reference가 필요하고, restore 중에도 host reference가 필요하다. 완료 callback이 reference를 넘겨받는 순간을 명시하지 않으면 double free 또는 leak이 된다. 취소 경로는 특히 중요하다. A가 LOAD_PENDING 동안 취소되더라도 이미 제출된 DMA가 host와 GPU 주소를 읽고 쓸 수 있다. callback이 끝날 때까지 양쪽 buffer를 보호한 뒤 결과를 폐기해야 한다.

metadata commit은 작은 연산이라 중요하지 않아 보이지만 correctness의 선형화 지점이다. payload 복사가 끝나기 전에 key→host slot mapping을 공개하면 lookup이 불완전한 사본을 찾는다. 반대로 mapping을 먼저 지우고 restore가 실패하면 유효한 host 사본을 잃는다. 구현을 검토할 때는 “복사를 하는가”보다 “어느 완료 신호 뒤에서 어떤 mapping을 공개하거나 제거하는가”를 묻는 편이 정확하다.

계층이 둘보다 많으면 lookup과 promotion이 추가된다. GPU L0, process-local host L1, file 또는 remote L2가 있다고 하자. L2 hit를 곧바로 GPU hit처럼 반환하지 않는다. L2→L1을 거쳐 L1→L0으로 올리거나 direct transfer가 가능한지에 따라 future가 달라진다. promotion을 하다가 상위 tier가 가득 차면 eviction과 load가 다시 얽힌다. 이 때문에 multi-tier manager는 단순한 dictionary가 아니라 pending future, tier별 allocator, promotion policy와 request finalization을 함께 다룬다.

## 38.4 무엇을 내리는가: 정책은 주소보다 먼저 비용을 결정한다

LRU는 오래 사용하지 않은 block을 먼저 내린다. locality가 강한 workload에서는 합리적이지만, 긴 요청이 round-robin으로 decode되는 서버에서는 “오래됨”의 시간 범위가 짧아 모든 요청의 tail block이 번갈아 밀릴 수 있다. 복원 직후 다시 쫓겨나는 ping-pong이 보이면 copy kernel을 튜닝하기 전에 candidate 정책과 protected working set을 확인해야 한다.

ARC 같은 적응형 정책은 최근 한 번 본 항목과 반복적으로 본 항목의 균형을 조정하려 한다. 그렇다고 전송 비용을 자동으로 아는 것은 아니다. 같은 block 수라도 layer 구성과 dtype에 따라 bytes가 다를 수 있고, 같은 bytes라도 재계산 시간은 prompt shape와 kernel 효율에 따라 달라진다. 정책의 hit rate가 높아도 큰 block의 restore가 critical path를 지배할 수 있으므로 byte-weighted hit와 exposed restore time을 함께 봐야 한다.

sliding-window layer는 오래된 token의 KV를 계속 보존할 필요가 없을 수 있다. full-attention layer와 같은 규칙으로 모든 layer를 offload하면 사용하지 않을 bytes까지 이동한다. Transformers가 non-sliding layer만 offload하도록 고르는 분기는 이 비용 차이를 코드 수준에서 드러내는 사례다. 다만 “sliding이면 항상 제외”도 일반 법칙은 아니다. 모델의 cache class와 layer policy가 실제 어떤 tensor를 유지하고 어떤 위치을 prefetch하는지 확인해야 한다.

정책 판단에는 최소 네 값이 필요하다. entry의 byte 크기, 다시 쓰일 확률, 다시 계산하는 비용, 복원 가능한 deadline이다. 여기에 공유 prefix의 reference count를 더한다. 여러 요청이 공유하는 큰 prefix는 재사용 확률이 높지만, 모든 consumer가 곧 필요로 한다면 내리는 순간 restore 폭주가 생긴다. 반대로 완료된 긴 요청의 private tail은 커도 다시 쓰이지 않을 가능성이 높으므로 host에 저장하는 것 자체가 낭비일 수 있다.

offload를 eviction의 동의어로 쓰지 않는 이유도 여기 있다. eviction은 어떤 tier의 residency와 mapping을 제거하는 수명주기 결정이다. demotion은 더 느린 tier에 유효한 사본을 남기는 이동이다. host store가 실패했는데 GPU entry만 evict하면 data loss다. host가 가득 차 candidate를 받아들이지 못하면 GPU eviction 정책은 recompute를 선택하거나 admission을 줄여야 한다. “evict하면 자동으로 CPU로 간다”는 가정은 코드에서 prepare/store/complete 경로를 확인하기 전에는 성립하지 않는다.

## 38.5 네 구현에서 같은 단어가 가리키는 다른 수명

이제 추상 상태 기계를 실제 코드에 대입한다. 제품 이름별 기능 목록을 외우는 대신 `lookup → protect → transfer → completion → publish → release`라는 질문을 들고 네 저장소를 걷는다. 같은 `offload`라는 낱말이 요청별 block 이동, layer별 tensor 이동, graph placement를 각각 가리킨다는 차이가 드러난다.

### 38.5.1 vLLM: 준비와 완료 사이를 인터페이스로 보존한다

vLLM의 `OffloadingManager` 추상 계약은 lookup 결과부터 단순한 참·거짓이 아니다. `HIT`, `MISS`뿐 아니라 발견했지만 아직 읽을 수 없는 `HIT_PENDING`, 나중에 다시 시도해야 하는 `RETRY`를 구분한다. 이 구분이 앞서 말한 “host hit는 GPU-ready가 아니다”라는 불변조건의 코드 표현이다. `prepare_load`는 선택한 key를 eviction에서 보호한 뒤 worker가 실제 위치를 찾을 `LoadStoreSpec`을 반환하고, `complete_load`가 와야 보호를 푼다. store도 `prepare_store`와 `complete_store(success)`로 나뉜다.

성공 completion 뒤에야 block이 loadable해지고 실패하면 미완료 entry를 제거한다. [vLLM v0.27.1 — `vllm/v1/kv_offload/base.py:220-320` — `OffloadingManager.lookup/prepare_load/prepare_store/complete_store`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/base.py#L220-L320)

이 인터페이스에서 가장 놓치기 쉬운 것은 요청 종료다. `on_request_finished`가 불렸다고 이미 제출한 transfer가 끝났다는 뜻이 아니다. 주석은 scheduler가 더 이상 새 submit-side 호출을 만들지 않지만, 기존 `complete_store`와 `complete_load` callback은 뒤늦게 올 수 있다고 명시한다. 여러 tier를 연쇄하는 manager는 아래 tier로 새 submit을 만들 가능성이 사라질 때까지 종료 전달도 늦춰야 한다.

요청 객체의 종료와 I/O lifetime의 종료를 같게 놓으면 callback이 해제된 context를 만지거나, 아직 저장 중인 payload를 지울 수 있다. [vLLM v0.27.1 — `vllm/v1/kv_offload/base.py:322-382` — `on_request_finished/has_pending_work`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/base.py#L322-L382)

구체적인 CPU manager로 내려가면 상태가 숫자로 보인다. `_num_evictable_cache_blocks`는 `ref_cnt == 0`인 block 수이고, `_num_write_pending_blocks`는 store가 날아갔지만 아직 완료되지 않은 block 수다. lookup은 policy에서 block을 찾은 뒤 `is_ready`가 거짓이면 `HIT_PENDING`을 반환한다. `prepare_load`는 refcount가 0에서 1로 갈 때 policy에 non-evictable로 표시하고, `complete_load`가 refcount를 다시 내린다.

따라서 load descriptor를 받은 순간부터 completion까지 host source slot은 eviction 후보가 아니다. [vLLM v0.27.1 — `vllm/v1/kv_offload/cpu/manager.py:30-164` — `CPUOffloadingManager`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/manager.py#L30-L164)

store 경로는 실패 의미까지 읽어야 한다. `prepare_store`는 이미 저장된 key를 제외하고 필요한 host block 수를 센다. free block이 모자라면 evictable 수와 policy의 protected 집합을 확인한다. eviction이 불가능하면 `None`을 반환하지, GPU 원본을 잃으면서 억지로 진행하지 않는다. 새 host block을 배정해 policy에 넣는 시점에는 아직 ready가 아니며 pending 수만 증가한다. 성공한 `complete_store`가 `ref_cnt = 0`으로 바꾸고 evictable로 공개한다. 실패 completion은 policy entry를 제거하고 physical block을 free list에 돌려놓는다.

여기서 metadata insert와 readiness publish가 분리되어 있으므로 lookup은 진행 중 entry를 식별할 수 있다. [vLLM v0.27.1 — `vllm/v1/kv_offload/cpu/manager.py:166-273` — `prepare_store/complete_store`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/manager.py#L166-L273)

manager의 block id는 아직 payload 주소가 아니다. GPU worker는 CPU shared region의 sub-block pointer와 GPU cache pointer를 transfer job으로 바꾸고, async copy를 제출하며, finished event를 수집한다. shared offload region은 mmap 영역을 만들고 tensor-page view를 제공한다. 이 구조를 읽을 때는 Python list의 block id가 최종 kernel에서 byte address와 stride로 어떻게 번역되는지 추적해야 한다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- `swap_blocks_triton` launcher는 source와 destination cache, block mapping을 받아 device/host copy를 실행한다.
- mapping 순서가 KV layout과 다르면 transfer는 성공해도 논리 block이 바뀐다.
- [vLLM v0.27.1 — `vllm/v1/kv_offload/cpu/gpu_worker.py:62-152` — worker와 sub-block 초기화](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/gpu_worker.py#L62-L152) [vLLM v0.27.1 — `vllm/v1/kv_offload/cpu/swap_blocks_triton.py:25-73` — block copy launcher](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/swap_blocks_triton.py#L25-L73)

### 38.5.2 SGLang: cache tree의 lock과 host page 수명을 같이 읽는다

- SGLang의 HiRadix 경로에서는 prefix tree의 logical node와 device·host page pool의 physical allocation이 만난다.
- tree node를 찾았다는 사실만으로 device residency가 보장되지 않는다.
- backup을 제출할 때 node 또는 value의 lock reference가 eviction을 막고, transfer acknowledgement 뒤에서 host residency를 공개해야 한다.
- load-back도 host page를 찾는 단계, device page를 예약하는 단계, copy 완료와 tree metadata 갱신 단계로 나눠 읽어야 한다.
- [SGLang v0.5.18 — `python/sglang/srt/mem_cache/hiradix_cache.py:841-1128` — backup과 acknowledgement](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/hiradix_cache.py#L841-L1128)

- 후반부의 device·host eviction과 load-back 경로는 “LRU node를 지운다”는 수준보다 많은 일을 한다.
- logical key가 살아 있는지, host value가 준비됐는지, lock이 남아 있는지에 따라 candidate가 달라진다.
- host에서 다시 읽는 동안 physical page를 재사용하지 않아야 하며, 완료 후에는 어느 tier의 value가 canonical인지 정리해야 한다.
- tree lock 누락은 즉시 crash하지 않고 경쟁 시점에만 stale prefix나 오답으로 나타나므로 평균 hit rate로는 찾기 어렵다.
- [SGLang v0.5.18 — `python/sglang/srt/mem_cache/hiradix_cache.py:1133-1455` — eviction과 load-back](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/hiradix_cache.py#L1133-L1455)

`memory_pool_host.py`는 이름 그대로 “CPU tensor 하나”가 아니라 logical host pool과 paged host pool의 layout을 구현한다. allocation과 free를 볼 때 total RSS만 보면 안 된다. 논리적으로 free된 page가 allocator나 pinned region에 reservation으로 남아 있을 수 있고, page metadata가 physical offset과 엇갈릴 수도 있다. backup·restore 함수는 device block index와 host page index의 대응을 copy 경로에 넘긴다.

- 따라서 host pool leak 조사는 live logical owner, free-list 회수, allocator reserved bytes를 분리해야 한다.
- [SGLang v0.5.18 — `python/sglang/srt/mem_cache/memory_pool_host.py:61-182` — host pool layout](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/memory_pool_host.py#L61-L182) [SGLang v0.5.18 — `python/sglang/srt/mem_cache/memory_pool_host.py:183-581` — allocation·backup·restore](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/memory_pool_host.py#L183-L581)

더 느린 storage tier가 붙으면 batch get/set의 반환값이 다음 전환의 근거가 된다. file tier에 write를 제출했다는 사실과 page가 다시 읽을 수 있다는 사실을 구별해야 한다. 부분 실패가 가능한 batch에서 전체 key를 성공 처리하면 stale 또는 빈 page를 나중에 hit로 공개한다. 이 장에서는 원격 protocol을 다루지 않지만, 공통 원칙은 같다.

하위 tier 결과를 key별로 확인하고, 성공 payload만 promotion 대상으로 만든다. [SGLang v0.5.18 — `python/sglang/srt/mem_cache/hicache_storage.py:95-328` — transfer result와 storage contract](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/hicache_storage.py#L95-L328)

### 38.5.3 Transformers: 요청 block이 아니라 다음 layer를 미리 가져온다

Transformers의 classic `Cache` offloading은 vLLM의 요청별 secondary block tier와 같은 문제를 풀지 않는다. layer list를 따라 현재 layer의 K/V를 쓰고, 다음 offloaded layer를 prefetch stream에서 device로 가져오며, 방금 쓴 layer를 CPU로 보낸다. `offloading=True`일 때 별도 prefetch stream을 만들고, 기본값인 `offload_only_non_sliding=True`는 sliding layer를 제외한다.

이유도 코드 주석에 드러난다. sliding cache는 보통 작으므로 옮기는 비용을 피한다. [Transformers v5.15.1 — `src/transformers/cache_utils.py:1275-1305` — `Cache.__init__`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L1275-L1305)

`prefetch`는 linear-attention layer를 제외하고, 옵션에 따라 sliding layer도 제외한 뒤 다음 대상 layer를 찾는다. 별도 stream 안에서 해당 layer의 `prefetch()`를 부른다. 핵심 ordering은 `update`에 있다.

현재 `key_states.device`의 default stream이 prefetch stream을 기다린 뒤 다음 layer prefetch를 시작하고, 현재 layer cache를 update하고, 끝으로 현재 layer를 offload한다. offload는 default stream에서 실행해 앞선 update 완료 순서를 보존한다. stream을 썼다는 사실만으로 overlap이 보장되는 것은 아니지만, 어느 의존성이 correctness에 필요한지는 명확하다. [Transformers v5.15.1 — `src/transformers/cache_utils.py:1317-1381` — `prefetch/offload/update`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L1317-L1381)

이 차이는 metric 해석도 바꾼다. vLLM식 host block hit rate를 Transformers layer offload에 그대로 적용할 수 없다. 여기서 질문은 어떤 요청 key가 재사용됐는지가 아니라 layer `i+1` prefetch가 layer `i` 계산과 얼마나 겹쳤고 default stream이 얼마나 기다렸는가다. `offload_only_non_sliding=False`로 바꿨을 때 host capacity는 늘 수 있지만, 작은 sliding layer까지 매 layer 순서에서 왕복하므로 transfer 수와 stream wait가 늘 수 있다.

### 38.5.4 llama.cpp: `--kv-offload`를 secondary cache로 오해하지 않는다

llama.cpp의 `--kv-offload`라는 옵션 이름은 특히 오해하기 쉽다. argument 경로는 사용자 입력을 `cparams.offload_kqv`로 옮긴다. 그러나 graph builder에서 이 값이 거짓일 때 KV store와 attention output 사이 node를 CPU backend에 배치한다. 즉 이 분기는 GPU-resident KV block을 가득 찼을 때 CPU tier로 evict하고 나중에 restore하는 manager 계약과 같지 않다.

K/Q/V 관련 graph node가 어느 backend에서 실행되는지 정하는 placement 경계다. [llama.cpp v0.2.0 — `common/arg.cpp:2404-2409` — `--kv-offload` option](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/arg.cpp#L2404-L2409) [llama.cpp v0.2.0 — `src/llama-graph.cpp:2635-2650` — `offload_kqv` consumer](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L2635-L2650)

이 옵션을 비교표의 “KV CPU offload 지원” 한 칸으로 축약하면 잘못된 운영 판단이 나온다. vLLM CPU manager에서 기대하는 host lookup, pending store, restore destination, request별 completion을 llama.cpp의 이 분기에서 찾을 수 없다. 반대로 llama.cpp에서는 model memory 구현 생성 때 선택한 buffer type과 graph scheduler의 backend placement를 따라가야 실제 주소와 실행 위치를 알 수 있다. 같은 이름을 비교하기 전에 **이동 단위가 요청 block인지, layer tensor인지, graph node인지**를 먼저 적어야 한다.

### 38.5.5 같은 2GiB를 네 경로에 넣어 본 왕복 기록

차이를 확실히 하려면 같은 2GiB라는 숫자를 네 구현에 넣어 보자. vLLM에서 이 2GiB는 여러 GPU block과 cache group에 걸친 요청별 KV 구간이다. manager가 logical key 집합을 `prepare_store`에 넘기면 CPU block을 배정하고, worker용 source·destination spec을 만든다. worker는 GPU 쪽 group size와 logical offset을 읽어 첫 CPU chunk의 중간부터 시작해야 하는 partial block을 계산한다. GPU block과 CPU chunk 크기가 다를 수 있으므로 단순히 `src[i]`를 `dst[i]`로 복사하지 않는다. 각 canonical KV tensor마다 sub-block pointer와 byte size descriptor를 채운다.

2GiB는 결국 여러 개의 `(src pointer, dst pointer, size)` 연산으로 풀린다. [vLLM v0.27.1 — `vllm/v1/kv_offload/cpu/gpu_worker.py:240-360` — `transfer_async` descriptor construction](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/gpu_worker.py#L240-L360)

GPU→CPU 제출 때 별도 transfer stream은 현재 compute stream을 기다린다. 이는 2GiB source가 attention 또는 cache write에 의해 아직 갱신되는 동안 copy가 앞질러 읽지 않게 한다. 같은 방향의 이전 transfer가 있다면 그 transfer의 end event도 기다린다. 즉 A, B, C의 store를 각각 다른 stream에 넣더라도 무제한 병행시키지 않고 제출 순서를 보존한다. descriptor batch copy를 호출한 앞뒤에는 timing event를 기록한다. 함수가 `True`를 반환하는 순간은 2GiB가 host에 존재하는 순간이 아니라 job이 유효하게 제출된 순간이다.

`get_finished`가 queue 맨 앞의 end event를 query하여 완료를 확인하고 transfer result를 만들 때 비로소 manager가 성공 completion을 처리할 수 있다. [vLLM v0.27.1 — `vllm/v1/kv_offload/cpu/gpu_worker.py:362-441` — stream ordering과 `get_finished`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/gpu_worker.py#L362-L441)

그 2GiB를 다시 올릴 때는 방향이 바뀐다. host pinned memory는 동시에 다른 GPU stream이 쓰지 않는다는 전제 아래 source access ordering을 완화할 수 있지만, destination GPU block은 load 완료 전 attention이 읽어서는 안 된다. worker가 완료 결과를 내놓은 뒤 manager가 load 보호를 풀고, scheduler 또는 실행 경로가 compute 의존성을 닫아야 한다. 여기서 event query는 CPU가 완료를 관측하는 방법이고, compute stream wait는 GPU가 data readiness를 관측하는 방법이다. 둘은 같은 역할이 아니다.

CPU가 event를 query했다고 해서 이미 제출된 다른 compute stream에 자동 의존성이 생기지 않으며, 반대로 stream wait만 걸어 놓고 manager metadata를 일찍 공개하면 다른 요청이 physical block을 잘못 소유할 수 있다.

SGLang에서 같은 2GiB는 HiRadix tree의 node·value와 host page pool 사이의 대응으로 풀린다. tree는 prefix identity와 공유 관계를 알고, memory pool은 device index와 host page index를 안다. backup을 시작할 때 tree lock reference가 logical entry를 보호하고, kernel 쪽에는 K/V source·destination pointer와 index가 전달된다. one-layer 경로는 cache tensor를 element dimension에 맞춰 평탄화한 뒤 element size와 unroll을 정하고 `launch_one`을 호출한다. all-layer 경로는 pointer tensor, source·destination index, 각 cache의 byte stride를 넘긴다.

그러므로 2GiB라는 총량만 같아도 layer별 tensor layout과 stride가 틀리면 성공적으로 끝난 kernel이 엉뚱한 위치를 복사한다. [SGLang v0.5.18 — `python/sglang/kernels/ops/kvcache/hicache.py:127-198` — one/all-layer transfer dispatch](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/ops/kvcache/hicache.py#L127-L198)

SGLang의 restore를 읽을 때는 “host page를 찾았다”에서 멈추지 않는다. device pool에 목적지 page를 배정하고, host·device index가 같은 logical token 범위를 가리키는지 확인하고, transfer completion 뒤에서 tree value를 갱신해야 한다. shared prefix를 여러 요청이 참조한다면 load-back 도중 lock count가 0이 되어 host node가 쫓겨나지 않아야 한다. 2GiB restore와 동시에 host eviction이 같은 page를 재사용하면 copy engine은 새 entry의 bytes를 읽을 수 있다. 이 경쟁은 작은 fixture에서는 잘 나타나지 않으므로, 운영자는 pending load 수와 eviction candidate 수가 동시에 증가한 시점의 generation을 기록해야 한다.

Transformers에 2GiB를 대입하면 단위가 달라진다. 2GiB가 한 요청의 block 집합이 아니라 여러 layer cache tensor의 합이라고 하자. forward가 layer 17을 실행할 때 layer 18의 offloaded cache를 prefetch stream에서 device로 올린다. default stream은 현재 layer가 사용할 cache의 prefetch 완료를 기다린다. layer 17 update가 끝나면 그 layer cache를 CPU로 보낸다. 여기서는 logical request key lookup이나 host hit 판정이 없고, model layer 순서가 다음 이동을 예고한다. 그래서 reuse 확률도 “이 요청이 다시 선택되는가”가 아니라 같은 forward에서 다음 layer가 곧 실행된다는 거의 확정된 순서다.

대신 layer compute가 2GiB 일부의 transfer를 충분히 가리지 못하면 매 layer boundary의 wait가 누적된다.

2GiB가 32개 layer에 균등하게 나뉘었다면 layer마다 약 64MiB다. 24GB/s 하한에서는 한 방향 약 2.8ms이지만 launch, contention과 synchronization을 더해야 한다. 한 layer compute가 1.5ms라면 이상적인 prefetch조차 약 1.3ms가 노출될 수 있고, 32개 layer에서 누적된다. 반대로 현재 layer compute가 5ms이고 copy engine이 독립적으로 진행하며 링크 경쟁이 없다면 상당 부분을 가릴 수 있다. 총 2GiB 전송 시간만으로 latency를 예측할 수 없는 이유다. layer pipeline에서는 `max(T_layer_compute, T_layer_copy)`에 가까운 steady state와 첫 prefetch·마지막 offload의 경계 비용을 나눠 봐야 한다.

llama.cpp에서 같은 2GiB를 “내렸다가 다시 올리는 요청 cache”라고 기록하면 조사 자체가 잘못된다. `offload_kqv`는 graph에서 KV store와 attention output 사이 node를 GPU에 둘지 CPU backend에 둘지를 정하는 데 쓰인다. 2GiB tensor의 buffer placement와 backend 실행 비용을 비교할 수는 있지만, host entry에 logical key를 commit하고 나중에 hit로 restore하는 왕복은 이 분기의 계약이 아니다. 따라서 vLLM의 store latency나 SGLang의 host cache hit metric을 같은 축에 놓지 않는다. llama.cpp에서는 해당 tensor buffer가 어디에 할당됐고 graph scheduler가 어느 backend로 node를 보냈으며 backend 사이 copy가 graph 실행에 어떻게 들어왔는지를 관측해야 한다.

이 왕복 기록은 제품 우열표가 아니다. 네 경로가 서로 다른 질문에 답한다는 경계표다. 요청이 쉬었다가 다시 runnable이 되는 동안 KV를 보존하려면 vLLM·SGLang의 request/block lifetime을 본다. 한 forward 안에서 layer cache residency를 회전시키려면 Transformers의 layer ordering을 본다. CPU와 GPU 중 attention 관련 graph node를 어디서 실행할지 묻는다면 llama.cpp의 backend placement를 본다. 2GiB라는 숫자가 같아도 key, 미래 사용 시점, completion owner가 다르면 같은 최적화가 아니다.

## 38.6 PCIe, pinned memory, stream과 event의 실제 비용

GPU와 CPU 사이의 전송을 설명할 때 “PCIe 대역폭” 하나만 쓰면 시간선의 절반이 사라진다. payload가 출발하기 전에 host memory가 GPU DMA에 적합해야 하고, source·destination descriptor가 준비돼야 하며, copy를 어느 stream에 넣을지 정해야 한다. 제출 뒤에는 dependency와 completion을 서로 다른 주체가 관측한다. 이 절의 질문은 “몇 GB/s인가”가 아니라 “2GiB가 runnable request의 attention input이 되기까지 누가 무엇을 기다리는가”다.

### 38.6.1 pinned memory는 공짜인 빠른 RAM이 아니다

일반 pageable host memory는 운영체제가 page를 다른 물리 위치로 옮기거나 swap할 수 있다. GPU DMA가 안정된 physical page를 대상으로 전송하려면 드라이버가 staging 또는 pinning 작업을 해야 한다. 미리 page-locked, 즉 pinned memory를 준비하면 반복 전송에서 이 준비를 줄이고 비동기 copy 경로를 사용할 수 있다. 그렇다고 host RAM 전체를 pin하는 것이 정답은 아니다. pin된 page는 운영체제의 회수와 이동 자유를 제한하고, 큰 영역 등록·해제 자체가 비용이며, NUMA 위치가 GPU와 멀면 inter-socket 경로를 탈 수 있다.

vLLM CPU worker는 mmap region을 쓸 때 해당 영역을 pin하고, 그렇지 않으면 `torch.zeros(..., pin_memory=...)`로 CPU tensor를 만든다. GPU cache tensor는 byte view로 바꾸고, CPU page 크기는 GPU page 크기에 `blocks_per_chunk`를 곱한다.

즉 host 96GiB라는 설정이 단일 contiguous object 하나를 뜻하지 않는다. canonical KV tensor별 view, block row stride와 worker 영역이 물리 layout을 이룬다. [vLLM v0.27.1 — `vllm/v1/kv_offload/cpu/gpu_worker.py:464-529` — `CPUOffloadingWorker.__init__`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/gpu_worker.py#L464-L529)

shared mmap region은 block row마다 worker slot을 배치한다. rank별 offset을 계산하고 각 canonical tensor view를 `(num_blocks, tensor_page_size)` 모양과 `(row_stride, 1)` stride로 만든다. dtype을 int8로 둔 까닭은 stride 값과 byte 주소 계산을 일치시키기 위해서다.

이때 view가 contiguous라는 가정으로 pointer를 계산하면 다음 row에서 다른 worker 영역을 읽는다. secondary tier용 memoryview도 원본 mmap과 같은 data pointer인지 assert한다. [vLLM v0.27.1 — `vllm/v1/kv_offload/cpu/shared_offload_region.py:120-171` — strided view와 zero-copy memoryview](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/shared_offload_region.py#L120-L171)

host RSS가 pool 한도를 넘었다는 사건에서는 세 숫자를 구분한다. 첫째는 live logical entry가 소유한 payload bytes다. 둘째는 allocator나 mmap이 예약했지만 free list에 있어 재사용 가능한 bytes다. 셋째는 page table, descriptor buffer, object와 fragmentation 같은 overhead다. 요청이 끝날 때 live bytes가 줄었는데 RSS가 유지되는 것은 반드시 leak이 아니다. 반대로 RSS가 설정 한도 안에 있어도 refcount가 해제되지 않아 evictable block이 0으로 굳으면 논리 leak이다. “프로세스 RSS가 안 줄었다”와 “새 store가 계속 실패한다”를 같은 결함으로 묶지 않는다.

초기 pin 비용도 steady-state 전송률에 숨기면 안 된다. 큰 pool을 시작할 때 page를 populate하고 등록하면 startup latency와 순간 CPU stall이 생길 수 있다. lazy fault를 허용하면 첫 store마다 tail latency가 튈 수 있다. 어느 쪽이 나은지는 배포 readiness와 첫 요청 SLO에 달려 있다. 측정할 때 pool 생성 구간, 첫 page touch 구간, warm 상태의 copy 구간을 분리해야 24GB/s와 무관한 비용을 링크 성능 저하로 오판하지 않는다.

### 38.6.2 stream은 순서를 담고 event는 완료 사실을 운반한다

CUDA stream 안의 작업은 순서대로 실행되지만 서로 다른 stream 사이에는 명시적 의존성이 없으면 겹칠 수 있다. 이 겹침이 offload의 이득이자 오답 위험이다. GPU→CPU store의 source KV를 compute stream이 막 갱신했다면 transfer stream은 compute 완료를 기다려야 한다. CPU→GPU load가 destination을 채우는 동안 compute stream은 그 block을 읽기 전에 transfer 완료를 기다려야 한다. 두 방향의 화살표를 같은 `synchronize()` 한 줄로 뭉개면 overlap을 잃거나 의존성을 빠뜨린다.

vLLM의 방향별 handler는 각 transfer에 stream과 start/end event를 붙인다. GPU→CPU일 때 transfer stream이 현재 compute stream을 기다리고, 같은 방향의 이전 transfer가 있으면 그 end event도 기다린다. 그 뒤 copy batch를 launch하고 end event를 기록한다. `get_finished`는 queue 선두 event를 query하여 완료된 job만 결과로 바꾸며 stream, event, descriptor buffer를 pool에 돌려준다. 이 순서는 event object의 재사용도 payload lifetime과 묶여 있음을 보여 준다. 완료 전에 event를 pool로 돌려 다른 job이 덮어쓰면 잘못된 job의 readiness를 관측한다.

“비동기”는 호출자가 기다리지 않는다는 뜻일 뿐, 하드웨어에서 반드시 다른 연산과 완전히 겹친다는 뜻은 아니다. GPU copy engine 수, 링크 양방향 경쟁, source·destination memory의 bandwidth, descriptor 수와 transfer 크기가 overlap을 제한한다. 아주 작은 block을 많이 보내면 payload 하한보다 launch와 descriptor 비용이 커진다. 반대로 2GiB 하나를 크게 보내면 bandwidth 효율은 좋아도 한 긴 job이 뒤의 urgent restore를 막는 head-of-line blocking이 생길 수 있다. chunk 크기는 throughput과 긴 꼬리 latency 사이의 정책 변수다.

event timing도 범위를 명확히 해야 한다. copy 전후 같은 stream에 기록한 event의 elapsed time은 그 stream에서 두 event 사이의 시간이다. manager queue 대기, scheduler가 restore를 발견하기까지 걸린 시간, completion을 CPU가 polling하는 간격, metadata commit 뒤 batch에 다시 들어가기까지의 시간은 빠질 수 있다. 그래서 `transfer_time`만으로 ITL 증가를 설명하지 않는다. 요청 timeline에는 적어도 lookup 시각, prepare 시각, submit 시각, copy start/end, completion commit, runnable 전환, attention launch를 함께 놓는다.

다음과 같은 간단한 시간선을 손으로 그리면 잘못된 가설을 빠르게 제거할 수 있다.

```text
t0 host lookup HIT
t1 GPU destination reserve
t2 H2D submit
t3 copy start
t4 copy end event
t5 manager completion commit
t6 request enters batch
t7 attention reads restored block
```

`t3→t4`만 길면 전송 bandwidth·contention을 본다. `t2→t3`가 길면 같은 방향 queue와 앞선 event 의존성을 본다. `t4→t5`가 길면 polling 또는 engine step 진행을 본다. `t5→t6`가 길면 scheduler budget과 priority를 본다. `t6→t7`에서 wait가 길면 compute stream dependency와 batch 내 다른 작업을 본다. 모든 구간을 “offload latency”라는 한 metric으로 합치면 어느 층을 고칠지 알 수 없다.

### 38.6.3 copy가 끝났는데도 오답이 나는 두 방식

첫 번째 오답 방식은 source ordering 누락이다. GPU→CPU transfer가 compute stream의 마지막 KV write보다 먼저 시작하면 host에는 old K와 new V 또는 일부 layer만 갱신된 혼합본이 남을 수 있다. DMA 자체는 성공하고 end event도 정상 완료된다. 나중에 restore했을 때만 token이 달라진다. 따라서 copy API의 success code나 checksum 없는 byte count로는 correctness를 증명하지 못한다. source generation과 마지막 writer의 stream 의존성을 함께 확인해야 한다.

두 번째 방식은 destination publish가 너무 이른 경우다. H2D end event 전에 block table이 새 GPU slot을 가리키거나 request가 runnable이 되면 attention kernel이 쓰는 중인 destination을 읽는다. 재현 빈도는 copy와 scheduler timing에 따라 달라져 부하가 낮으면 사라질 수 있다. 임시로 전역 synchronize를 넣었을 때 오답이 사라지면 ordering 가설이 강해지지만, 이것을 최종 수정으로 삼으면 모든 overlap이 사라진다. 정확한 event wait와 metadata commit 지점을 찾아 최소 dependency로 복구해야 한다.

generation이 틀린 경우에는 copy ordering이 완벽해도 오답이다. host key가 옛 request A의 block을 가리키는데 physical host slot은 이미 request B의 새 generation으로 재사용됐다고 하자. A restore는 B의 payload를 정확히 복사하고 event도 정확히 기다린다. 문제는 logical identity와 physical ownership의 결합이다. lookup 때 본 generation, prepare_load에서 보호한 generation, completion에서 commit할 GPU generation이 같은지 검증해야 한다. offload correctness는 stream만의 문제가 아니라 allocator와 cache key까지 걸친다.

이 경계를 검증하는 가장 작은 기록은 job마다 다섯 좌표를 남기는 것이다. logical key digest, source tier와 physical slot·generation, destination slot·generation, submit과 completion sequence, consumer batch id를 한 줄에 묶는다. payload 원문이나 token을 로그에 남기라는 뜻은 아니다. 개인정보 없이도 동일한 logical key가 서로 다른 generation으로 바뀌었는지, 완료 전에 consumer가 등장했는지 판별할 수 있는 상관관계가 필요하다. 정상 요청 하나에서는 `prepare → submit → end event → commit → consume` 순서가 닫혀야 한다. 취소 요청에서는 submit 뒤 completion이 오더라도 publish하지 않고 양쪽 reference만 회수해야 한다.

이 두 timeline을 먼저 확보하면 전역 동기화나 cache reset처럼 원인을 가리는 조치를 하기 전에 상태 기계의 어느 edge가 깨졌는지 좁힐 수 있다.

pinned bytes와 transfer bytes를 같은 단위로 비교한다. 96GiB를 pin했는데 실제 재사용 payload가 분당 수백 MiB뿐이라면 큰 pool은 capacity 보험보다 운영체제 압박이 클 수 있다. 반대로 pinned pool이 너무 작아 반복 등록과 staging copy가 일어나면 설정상 host hit가 실제 direct H2D로 이어지지 않는다. 관측해야 할 것은 설정값이 아니라 live host entry bytes, pinned registered bytes, direction별 submitted·completed bytes, queue depth, exposed restore time이다. 이 값들은 원인 자체가 아니라 앞서 나눈 시간선과 ownership 가설을 선택하는 증거다.

## 38.7 여섯 사건으로 좁히는 offload 장애

이제 정상 경로를 실제 장애에 적용한다. 여섯 사건은 모두 “offload를 켠 뒤 나빠졌다”로 시작하지만, 같은 체크리스트로 풀리지 않는다. 각 사건에서 먼저 증상을 시간축과 자원축으로 좁히고, 그럴듯하지만 틀린 가설을 하나씩 버린 뒤, 상태 기계의 어느 전환이 깨졌는지 확인한다.

### 38.7.1 사건 1: OOM은 줄었지만 ITL p99가 솟았다

증상은 명확했다. offload 전에는 긴 prompt가 몰릴 때 admission 실패와 GPU OOM이 발생했다. offload 후에는 요청이 받아들여졌지만 decode ITL p50은 거의 그대로인 반면 p99만 180~300ms씩 튀었다. 평균 GPU utilization은 낮아졌고 host hit rate는 높았다. 처음에는 host hit가 높으므로 cache가 잘 작동하고, GPU utilization 저하는 CPU scheduler overhead 때문이라고 추측했다.

평균값을 버리고 요청 timeline을 맞추자 다른 그림이 보였다. 튄 token 직전에 해당 요청은 preemption 뒤 `HOST_RESIDENT`에서 발견됐다. lookup부터 submit까지는 짧았지만 H2D queue가 길었다. 앞서 여러 GPU→CPU store가 같은 방향 handler 또는 링크 자원을 차지했고, urgent restore가 뒤에 섰다. copy 자체의 event time은 90ms 안팎이었지만 submit 전 대기와 completion polling, 다음 scheduler step까지 합쳐 200ms가 노출됐다. host hit가 높은 것은 이득의 증거가 아니라 restore 의존 요청이 많다는 신호이기도 했다.

CPU scheduler 가설은 두 관측으로 반증됐다. ITL spike는 CPU run queue가 아니라 H2D pending bytes와 정렬됐고, 같은 token의 `t2→t3` 및 `t4→t6` 구간이 길었다. offload를 끄고 admission을 낮춘 대조군에서는 CPU 부하는 비슷했지만 spike가 사라졌다. 해결은 단순히 copy thread를 늘리는 것이 아니었다. urgent load가 background store 뒤에 무한히 밀리지 않도록 방향·우선순위 queue를 조정하고, restore destination용 GPU reserve를 남기고, deadline이 가까운 decode 요청은 애초에 demote하지 않는 정책을 적용해야 했다.

검증 종료 조건도 “p99가 한 번 내려갔다”가 아니다. 같은 prompt·decode 길이 분포와 동시성에서 OOM·거절률, TTFT, ITL p50/p99, direction별 pending bytes, exposed restore time을 함께 비교한다. admission 이득이 유지되고 ITL deadline 위반이 기준선 이하로 돌아오며, store 폭주 때도 load가 bounded time 안에 시작해야 복구를 닫는다.

### 38.7.2 사건 2: host RSS가 pool 한도를 넘었다

설정은 host pool 96GiB인데 프로세스 RSS가 112GiB까지 올랐다. 요청이 끝나도 RSS가 곧바로 내려오지 않았다. “KV entry가 해제되지 않는 leak”이라는 신고가 들어왔고, 운영자는 cache reset을 주기적으로 호출하자고 제안했다. 하지만 reset은 live owner와 allocator reservation을 함께 지워 원인을 숨길 뿐 아니라 다음 요청의 miss와 prefill을 폭증시킨다.

먼저 112GiB를 세 부분으로 나눴다. manager가 ready 또는 pending으로 추적하는 block bytes, free list에 돌아왔지만 mmap·allocator가 계속 예약한 bytes, descriptor·page table·Python object와 다른 프로세스 메모리다. live logical block 수는 요청 종료 뒤 줄었고 `_num_evictable_cache_blocks`도 회복됐다. 같은 host block id가 새 store에 재사용됐다. 이 경우 96GiB pool이 RSS에서 즉시 사라지지 않는 것은 leak의 증거가 아니다. pool은 재사용을 위해 예약된 장기 수명 영역이다.

그러나 추가 16GiB는 별도 문제였다. 짧은 실험마다 새로운 pool을 만들고 옛 worker shutdown이 완료되기 전에 다음 engine을 시작했다. outstanding event를 기다리는 동안 mmap view와 descriptor buffer가 base storage reference를 붙들었다. cleanup에서 view를 base보다

먼저 놓아야 하는 이유가 여기 있다. base를 먼저 닫으려 해도 view가 mmap buffer export를 잡고 있으면 해제가 지연되거나 실패한다. [vLLM v0.27.1 — `vllm/v1/kv_offload/cpu/shared_offload_region.py:173-190` — `cleanup` ownership order](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/shared_offload_region.py#L173-L190)

이 사건의 강한 leak 증거는 RSS가 높다는 사실이 아니라, 요청과 transfer가 모두 끝난 뒤에도 특정 generation의 logical owner/refcount가 0으로 내려가지 않고 free candidate가 되지 않는 것이다. allocator reservation 문제의 강한 증거는 live bytes가 낮고 동일 physical slot이 재사용되는데 OS RSS만 유지되는 것이다. deferred cleanup 문제의 강한 증거는 shutdown 대기 job, 남은 view와 mmap instance 수가 배포 반복 횟수에 비례하는 것이다. 세 경우의 복구가 다르므로 하나의 “CPU cache memory” 그래프로 합치지 않는다.

수정 검증에서는 steady state와 rolling restart를 분리했다. steady state에서는 live block, ready/pending, evictable과 free block의 보존식을 확인한다. restart에서는 새 요청을 중단한 뒤 in-flight job이 0이 되고, handler가 event·stream·descriptor pool을 비우고, view→base→mmap 순서로 ownership을 놓는지 본다. RSS 회수 시각은 OS 정책에 따라 늦을 수 있으므로 physical slot 재사용과 file mapping 수를 함께 증거로 삼는다.

### 38.7.3 사건 3: host hit인데 재계산보다 느렸다

서비스는 cache hit ratio 82%를 성공 지표로 내세웠다. 그런데 짧은 공유 prefix 요청에서는 offload를 끈 쪽이 TTFT가 더 짧았다. 팀은 PCIe 링크 불량을 의심했다. 실제 event bandwidth는 예상에 가까웠기 때문에 조사도 막혔다. 문제는 hit ratio의 분모였다. block 개수 기준 hit는 높았지만 작은 block과 큰 block을 같은 한 건으로 셌고, 복원 queue와 metadata 비용을 포함하지 않았다.

요청별로 `T_restore_exposed`와 `T_recompute`를 짝지어 보니 짧은 suffix는 prefill 재계산이 35~60ms인 반면 restore는 payload 하한만 80ms를 넘었다. GPU에 destination을 만들기 위한 eviction까지 발생하면 150ms가 넘었다. 반면 매우 긴 suffix는 재계산이 수백 밀리초여서 restore가 유리했다. “host hit이면 사용”이라는 정책이 서로 다른 손익분기 영역을 하나로 처리한 것이 원인이었다.

여기서 링크 불량 가설은 copy 구간 bandwidth가 정상이고, payload가 작은 집단에서도 고정 queue·commit 비용 때문에 역전된다는 관측으로 반증됐다. 개선 정책은 logical hit를 곧바로 load로 바꾸지 않는다. 예상 restore bytes, 현재 H2D queue, destination 확보 비용, 동일 suffix의 추정 recompute cost와 deadline을 비교한다. 작은 suffix나 queue가 긴 시점에는 host entry가 있어도 recompute를 고른다. 이는 cache data가 틀렸다는 뜻이 아니라 사용할 가치가 없는 hit를 거부하는 것이다.

hit metric도 네 가지로 쪼갰다. lookup hit block 수, hit payload bytes, 실제 load를 선택한 bytes, 재계산 대비 절약된 compute time이다. 첫 두 값은 cache coverage를, 세 번째는 policy 결정을, 네 번째는 결과를 말한다. 복구 판정은 hit ratio 유지가 아니라 같은 goodput과 OOM 조건에서 TTFT·ITL이 개선되고, 선택된 restore의 `T_restore_exposed < T_recompute` 비율이 충분히 높아지는 것이다.

### 38.7.4 사건 4: 옛 host block이 새 GPU generation으로 돌아왔다

네 번째 사건은 성능이 아니라 correctness였다. 긴 대화를 반복 재개하면 아주 드물게 앞 문맥과 무관한 token이 나왔다. 같은 prompt를 단독으로 실행하면 재현되지 않았고, 높은 동시성에서 cache eviction과 restore가 겹칠 때만 나타났다. copy event에는 오류가 없었고 transferred bytes도 기대값과 같았다. 그래서 처음에는 sampling의 비결정성이나 GPU kernel race를 의심했다.

결정적 증거는 logical key와 physical slot generation을 함께 남긴 trace에서 나왔다. 요청 A가 host key `K`를 lookup했을 때 slot 91, generation 12를 보았다. load 준비 전에 다른 eviction 경로가 reference를 잘못 0으로 내려 slot 91을 free list에 넣었다. 요청 B의 store가 같은 slot을 generation 13으로 재사용했다. A의 transfer descriptor는 physical slot 91을 정확히 읽었지만 generation을 다시 검증하지 않았다. 결과적으로 B payload가 A의 GPU destination으로 정확하게 복사됐다. DMA와 event는 모두 정상이고, 잘못된 것은 identity였다.

sampling 가설은 offload를 거친 요청에서 첫 divergence 위치가 restore block 경계 직후로 고정되고, 같은 random seed와 logits processor에서도 host generation mismatch가 있을 때만 갈라진다는 관측으로 반증됐다. kernel race 가설은 transfer를 동기화해도 generation 재사용을 막지 않으면 오답이 남는다는 대조로 약해졌다. 전역 synchronize가 ownership 버그를 고치지 못하는 대표 사례다.

수정은 lookup 결과에 단순 slot id만 넘기지 않고 key와 generation을 묶어 prepare에서 보호하도록 했다. allocator는 refcount 또는 pending read가 있는 generation을 재사용하지 않는다. completion은 처음 보호한 host generation과 destination 예약 generation을 확인한 뒤에만 GPU mapping을 공개한다. 요청이 취소되면 copy 완료 후 payload를 publish하지 않고 두 reference를 회수한다. 검증은 높은 churn에서 slot 재사용을 강제로 늘리고, `lookup_gen == protected_gen == copied_gen` 불변조건과 첫 logits의 일치를 동시에 확인한다.

### 38.7.5 사건 5: offload stream을 켠 뒤 간헐적 오답이 났다

다섯 번째 사건은 generation이 모두 맞는데도 발생했다. 순차 copy 모드에서는 정상이었고 async stream을 켜면 수천 요청에 한 번씩 첫 decode token이 달라졌다. 팀은 pinned memory가 불안정하다고 추측해 pageable staging으로 바꿨다. 속도는 느려졌지만 오답은 완전히 사라지지 않았다.

timeline에서 H2D end event보다 attention launch가 먼저인 요청이 발견됐다. manager는 worker submit이 성공하자 host entry를 hit 처리했고 scheduler는 다음 step에서 요청을 batch에 넣었다. CPU completion callback이 뒤늦게 왔지만 block table은 이미 destination을 가리켰다. compute stream은 transfer stream의 end event를 기다리지 않았다. 작은 payload에서는 copy가 우연히 먼저 끝나 문제가 감춰졌고, 링크 contention이 있을 때만 kernel이 일부 갱신된 destination을 읽었다.

pinned memory 가설은 같은 pinned allocation을 유지한 채 compute stream에 정확한 event dependency를 넣자 오답이 사라진 것으로 반증됐다. pageable 모드에서 빈도가 낮아진 것은 copy timing이 달라져 race window가 바뀐 결과였다. 수정은 manager readiness와 device readiness를 분리하고, destination을 consumer에게 publish하는 지점에서 해당 job의 completion을 확인하거나 consumer stream에 wait를 연결했다. 모든 stream을 device-wide synchronize하지는 않았다. 그렇게 하면 correctness는 얻어도 offload의 overlap 이유를 없애기 때문이다.

복구 테스트는 출력 일치만 보지 않는다. transfer를 인위적으로 작게 쪼개고 store/load 경쟁을 높여 race window를 넓힌다. 각 destination generation에 last-writer event를 기록하고 consumer launch가 happens-after 관계를 만족하는지 검사한다. 취소와 timeout도 넣어 완료 callback이 늦게 왔을 때 이미 새 request에 배정된 destination을 publish하지 않는지 본다. 오답 0회와 함께 ITL이 전역 동기화 대안보다 유지돼야 수정이 완성된다.

### 38.7.6 사건 6: sliding layer까지 옮기자 전송만 늘었다

마지막 사건에서는 correctness 문제가 없었다. full-attention과 sliding-window layer가 섞인 모델에서 모든 layer cache를 CPU로 옮기도록 설정하자 GPU 사용량은 조금 줄었지만 transfer bytes가 크게 늘고 forward latency가 악화됐다. 담당자는 sliding layer도 KV이므로 동일하게 offload하는 것이 일관적이라고 주장했다.

layer별 byte contribution과 재사용 범위를 계산하자 sliding layer는 제한된 window만 보존해 resident footprint가 작았다. 이것을 매 forward에서 CPU로 내리고 다시 올리는 고정 비용이 절약한 GPU bytes에 비해 컸다. 더구나 linear-attention 계열 state는 classic K/V layer와 update·offload 계약이 같지 않을 수 있다. Transformers의 구현이 linear layer를 prefetch 대상에서 제외하고 기본적으로 non-sliding layer만 offload하는 이유를 실제 predicate에서 확인할 수 있었다.

“일관된 정책이 더 단순하다”는 가설은 operational simplicity와 cost simplicity를 혼동했다. 한 boolean으로 모두 옮기면 설정은 단순하지만 transfer와 stream wait는 증가한다. layer별 offload 여부를 기록했을 때 latency 증가가 sliding layer copy와 정렬됐고, 이를 resident로 남기자 GPU bytes 증가는 작으면서 wait가 줄었다. 수정 후에는 full layer의 memory 절감이 유지되고 sliding layer transfer bytes가 사라졌으며, 동일 batch에서 layer boundary stall이 줄었다.

다만 이 결과를 모든 hybrid 모델의 규칙으로 일반화하지 않는다. window 크기, layer 수, dtype, recurrent state shape와 GPU pressure가 달라지면 손익분기도 달라진다. 실제 cache class가 `is_sliding`, `is_linear`를 어떻게 채우며 옵션 predicate가 어느 layer를 소비하는지 확인한다. 검증은 layer별 resident bytes, direction별 transfer bytes, default-stream wait와 end-to-end latency를 함께 비교한다. 단지 총 GPU memory가 줄었다는 이유로 정책을 성공 처리하지 않는다.

## 38.8 언제 offload를 끄고, 무엇을 남길 것인가

offload는 메모리가 부족할 때 항상 켜는 안전장치가 아니다. capacity를 latency와 host·interconnect 자원으로 바꾸는 선택이다. 따라서 “지원한다”와 “이 workload에서 이롭다” 사이에는 명시적인 보류 조건이 필요하다.

### 38.8.1 끄는 편이 나은 네 조건

첫째, working set이 GPU cache 안에 안정적으로 들어가고 OOM·preemption·admission loss가 거의 없다면 host tier는 얻는 용량 없이 store traffic만 만든다. 특히 짧은 대화가 많고 prefix reuse가 GPU 안에서 끝나는 서비스에서는 demotion candidate를 만드는 정책 자체가 불필요할 수 있다. 이때 offload를 켠 대조군이 throughput을 유지하더라도 ITL tail과 host pressure만 늘면 끈다.

둘째, 재계산이 restore보다 싸면 host hit를 쓰지 않거나 기능을 끈다. 짧은 suffix, 효율적인 chunked prefill, PCIe 경쟁이 큰 노드에서는 이 역전이 흔하다. 평균 hit ratio가 높다는 이유로 유지하지 않는다. `T_restore_exposed`, `T_recompute`, deadline을 길이 구간별로 비교한다. crossover보다 짧은 구간이 트래픽 대부분이면 offload의 복잡성에 비해 이득이 없다.

셋째, restore를 SLO 밖으로 숨길 여유가 없으면 끈다. 대화형 decode 요청이 다시 runnable이 되는 즉시 다음 token을 내야 하고 prediction 가능한 idle window가 없다면 80ms 이상의 load는 치명적이다. store를 background로 가릴 수 있어도 load는 소비 전에 끝나야 한다. urgent load priority와 GPU destination reserve로도 p99를 지키지 못하면 admission을 줄이거나 GPU cache를 늘리는 편이 낫다.

넷째, host memory와 PCIe가 다른 핵심 기능과 경쟁한다면 전체 노드 관점에서 끈다. NIC의 GPU-direct traffic, NVMe loading, 다른 GPU의 host transfer가 같은 topology를 쓸 수 있다. offload 단독 benchmark에서 24GB/s가 나와도 실제 배포에서는 root complex와 NUMA hop 때문에 collective 또는 ingress가 악화될 수 있다. GPU OOM 감소가 네트워크 tail과 시스템 swap 위험보다 가치 있는지 goodput으로 판단한다.

이 조건은 영구적인 금지 목록이 아니다. traffic mix, 모델 cache shape, GPU generation, topology가 바뀌면 손익분기를 다시 센다. 기능 flag를 켜고 끄는 행위보다 중요한 것은 왜 그 상태를 선택했는지 재검증 가능한 수치와 timeline을 남기는 것이다.

### 38.8.2 켜기 전에 닫아야 할 운영 계약

첫 계약은 capacity다. GPU resident working set, restore destination reserve, host logical capacity와 실제 pinned reservation을 따로 정한다. GPU 24GiB와 host 96GiB를 120GiB usable로 더하지 않는다. 최소 GPU reserve가 없으면 promotion이 eviction을 다시 요구해 load/store ping-pong이 시작된다.

둘째 계약은 policy다. 어떤 block·layer가 candidate이며 protected 조건은 무엇인지 적는다. shared prefix refcount, pending transfer, imminent decode deadline, sliding layer 여부를 candidate predicate에 반영한다. LRU라는 이름만 기록하지 않고 key touch가 어느 사건에서 일어나며 restore 직후 보호 기간이 있는지 확인한다.

셋째 계약은 completion이다. submit 성공, device event 완료, manager metadata commit, scheduler runnable 전환을 서로 다른 시각으로 기록한다. 요청 종료 뒤 callback이 올 수 있다는 수명도 포함한다. shutdown은 pending job을 기다리고 stream·event·descriptor view와 mmap ownership을 올바른 순서로 놓아야 한다.

넷째 계약은 관측이다. direction별 submitted/completed bytes와 jobs, queue wait, copy event time, exposed restore time, host ready/pending/evictable blocks, pinned bytes, GPU destination wait를 수집한다. label에 raw request id나 cache key를 넣어 cardinality를 폭발시키지 않는다. 상세 generation 연결은 제한된 trace 또는 sampling log로 남기고 집계 metric은 workload bucket과 tier·direction 정도로 제한한다.

다섯째 계약은 rollback이다. offload를 끄면 host data를 무조건 GPU로 모두 올리는 것이 아니라 새 demotion을 중단하고 in-flight transfer를 안전하게 drain하며, host-only request는 recompute·restore·fail 중 명시된 정책으로 처리한다. process를 죽여 pending DMA와 mmap을 동시에 끊는 것을 rollback이라 부르지 않는다. 출력 correctness와 resource cleanup이 함께 닫혀야 한다.

### 38.8.3 배포 전 손계산과 관측 판정

배포 전에 최소 세 workload bucket을 만든다. 짧은 private prompt, 긴 shared prefix, 긴 private conversation이다. 각 bucket에서 suffix bytes, 재계산 시간, 예상 reuse 확률, restore deadline을 적는다. 2GiB/24GB/s 하한처럼 단위가 보이는 계산을 하고, queue와 contention이 없는 최선의 경우에도 SLO가 불가능하면 실험을 진행하지 않는다.

그 다음 source 좌표에서 실제 옵션 consumer와 대상 단위를 확인한다. vLLM이면 request block manager와 worker completion을, SGLang이면 tree lock과 host page, Transformers면 layer predicate와 stream ordering, llama.cpp이면 graph backend placement를 본다. 이름만 같은 기능을 비교 실험에 섞지 않는다. 특히 weight offload 결과를 KV tiering 효과로 보고하거나 `offload_kqv`를 host hit cache로 해석하지 않는다.

canary에서는 세 단계로 용량을 늘린다. host pool만 예약하고 transfer를 하지 않는 단계에서 startup·RSS·NUMA 영향을 본다. 제한된 candidate만 store하는 단계에서 G2H queue와 allocator 수명을 본다. load까지 허용하는 단계에서 H2D critical path와 출력 일치를 본다. 한 번에 최대 pool과 모든 요청을 켜면 어느 단계가 tail을 만들었는지 알 수 없다.

성공 판정은 다섯 문장으로 닫을 수 있어야 한다. admission 또는 OOM이 얼마나 개선됐는가. 그 대가로 TTFT·ITL의 어느 분위수가 얼마나 변했는가. restore가 재계산보다 유리한 요청 비율은 얼마인가. host pinned memory와 PCIe 경쟁이 다른 workload를 해치지 않았는가. 취소·eviction·restart 경쟁에서도 generation과 출력이 일치하는가. 하나라도 증거가 없으면 “용량이 늘었다”는 이유만으로 전면 배포하지 않는다.

### 38.8.4 2GiB 요청 한 건의 배포 판정표를 완성한다

마지막으로 첫 사건의 요청 A를 종이 위에서 끝까지 판정해 보자. A의 offload 대상 suffix는 2GiB, 엄밀한 payload 하한은 89.5ms다. 관측된 G2H queue는 12ms, copy event는 96ms, completion commit까지 4ms였다. store는 A가 선점된 140ms 동안 진행되어 `12 + 96 + 4 - 140`이 0보다 작으므로 요청 A의 직접 latency에는 노출되지 않았다. 그러나 링크와 pinned bandwidth를 썼다는 시스템 비용은 남는다.

A가 다시 선택됐을 때 H2D queue는 31ms, event copy는 94ms, completion에서 runnable까지 8ms였다. scheduler가 미리 복원을 시작해 다른 요청 계산과 45ms를 겹쳤다. 따라서 exposed restore는 `31 + 94 + 8 - 45 = 88ms`다. 같은 suffix를 다시 prefill한 대조군은 warm 상태에서 132ms였다. 이 요청 하나만 보면 restore가 약 44ms 유리하다. 여기서 89.5ms payload 하한과 exposed 88ms가 모순처럼 보이지만, 전자는 copy 자체 하한이고 후자는 queue·commit을 합한 뒤 45ms overlap을 뺀 critical-path 노출이다. 서로 다른 시간 범위를 비교해야 한다.

이제 확률을 넣는다. A와 같은 workload bucket에서 demote한 suffix가 다시 쓰일 확률은 0.35였다. store의 exposed time은 0이지만 shared-link interference를 request-equivalent 9ms로 추정했다고 하자. 단순 기대 비용은 `9 + 0.35 × 88 = 39.8ms`다. 재계산 기대 비용은 `0.35 × 132 = 46.2ms`다. 차이는 요청당 약 6.4ms로 작다. 측정 오차나 traffic 변화로 쉽게 뒤집힐 수 있으므로 “항상 offload”보다 queue threshold와 길이 threshold를 둔 조건부 정책이 합리적이다.

capacity 이득을 더하면 결론이 달라질 수 있다. offload하지 않은 대조군은 같은 동시성에서 요청 2%를 admission deadline 안에 받지 못했고, offload군은 0.2%만 놓쳤다고 하자. 완료 요청당 latency가 약간 늘어도 전체 goodput은 좋아질 수 있다. 반대로 deadline을 넘긴 restore가 3%라면 accepted request가 늘어도 유효한 응답은 늘지 않는다. 그래서 admission 성공률이 아니라 deadline 안에 완료된 요청 수를 분자로 삼는다.

관측 레코드에는 다음 상태가 연결되어야 한다. A의 logical key digest와 host generation, G2H job id와 end event, host readiness commit sequence, H2D job id와 destination GPU generation, runnable batch id, 첫 consumer attention launch다. 순서는 `store submit < store end < host publish < load submit < load end < GPU publish < consume`이어야 한다. 두 transfer가 방향별 queue에서 겹칠 수 있어 전체 job id가 단조로울 필요는 없지만, 같은 payload의 happens-before 관계는 깨지면 안 된다.

반증도 함께 설계한다. 첫 가설은 “PCIe가 느려 p99가 증가했다”다. event copy bandwidth가 안정적이고 queue·runnable gap이 spike를 설명하면 기각한다. 둘째 가설은 “host hit가 많으므로 이득이다”다. `T_restore_exposed >= T_recompute`인 bucket이 많으면 기각한다. 셋째 가설은 “출력이 달라진 것은 sampling noise다”다. 동일 seed에서 divergence가 restore boundary와 generation mismatch에 정렬되면 기각한다. 넷째 가설은 “RSS가 안 줄어 leak이다”다. live owner가 0이고 physical slot 재사용이 관측되면 allocator reservation으로 분기한다.

정책을 바꾼 뒤에는 같은 A를 세 번 관측한다. 정상 restore, load 중 취소, host slot churn과 동시에 일어난 restore다. 정상 경로는 88ms 노출과 출력 일치를, 취소 경로는 publish 없이 reference 회수를, churn 경로는 generation 보호를 증명한다. 평균 latency만 좋아지고 취소 때 pending reference가 남으면 배포하지 않는다. 반대로 correctness test만 통과하고 urgent restore가 store 뒤에서 무한히 기다려도 배포하지 않는다. offload의 완료 조건은 capacity·latency·lifetime 세 축이 동시에 닫히는 것이다.

이 판정표는 특정 metric 이름이나 로그 문자열에 의존하지 않는다. 구현이 바뀌어도 lookup, reserve, submit, device completion, metadata publish, consumer라는 사건은 남는다. 새 버전을 감사할 때 함수 이름이 달라졌다면 각 사건의 새 owner를 찾고, 사건 사이 edge가 여전히 증명되는지 확인한다. 이것이 옵션 도움말보다 오래가는 독법이다.

장애를 닫는 증거도 이 사건 열을 따라 서로 달라야 한다. ITL 사건은 load queue의 상한과 deadline 내 consumer 도달률이 회복됐을 때 닫는다. RSS 사건은 단순히 숫자가 내려갈 때가 아니라 live owner 보존식이 맞고 rolling restart 뒤 낡은 mapping과 view가 남지 않을 때 닫는다. 느린 hit 사건은 선택된 restore가 같은 요청의 재계산보다

실제로 싸고 goodput이 개선될 때 닫는다. generation 사건은 높은 slot churn에서도 lookup·protected·copied generation이 일치하고 logits divergence가 없을 때 닫는다. stream 사건은 전역 synchronize 없이 last-writer event와 consumer 사이 의존성이 성립할 때 닫는다. sliding layer 사건은 절약한 resident bytes와 추가 transfer·wait를 layer별로 비교해 선택 predicate가 손익분기를 반영할 때 닫는다.

이렇게 종료 조건을 나누면 cache reset, process restart, device synchronize 같은 강한 조치가 왜 불충분한지도 보인다. 세 조치는 증상을 잠시 없앨 수 있지만 policy의 재발 조건, reference 회수, generation 검증과 최소 stream dependency를 증명하지 않는다. 복구 뒤 같은 부하와 취소·churn 조건을 다시 주었을 때 상태 edge가 유지돼야 한다. 정상 경로만 한 번 통과한 결과는 회복이 아니라 재현 대기 상태다.

최종 손익분기는 한 숫자가 아니라 순서 있는 결정이다. 먼저 correctness와 cleanup 불변조건을 만족하지 못하면 기능을 끈다. 그다음 최선의 payload 하한이 SLO를 이미 넘으면 restore 대신 재계산 또는 admission 제한을 택한다. 하한은 가능하지만 queue가 문제라면 priority와 reserve로 노출 시간을 줄인다. 마지막으로 capacity가 늘어도 deadline goodput이 늘지 않으면 끈다. 이 순서를 지키면 높은 hit ratio나 낮은 OOM 하나가 다른 실패를 가리지 못한다.

운영 문서에는 마지막 선택과 함께 기각한 대안도 짧게 남긴다. 예를 들어 “2GiB restore를 사용”이라고만 적지 말고, 측정된 H2D queue·copy·overlap, 같은 suffix 재계산 시간, reuse bucket과 deadline 결과를 적는다. 다음 GPU나 CUDA·driver·모델 버전에서 대역폭과 cache shape가 달라지면 이 입력만 다시 측정해 결정을 갱신할 수 있다. 반대로 특정 서버 이름과 당시의 boolean 값만 남기면 새 버전에서 옵션이 같은 이름을 유지하더라도 실제 consumer와 이동 단위가 바뀐 사실을 놓친다. 좋은 운영 기록은 결론보다 결론을 재계산할 좌표를 보존한다. 그 좌표가 있어야 offload를 켜는 판단만큼 끄는 판단도 장애 대응이 아니라 검증 가능한 설계 변경이 된다.

### 38.8.5 판정: 느린 메모리를 더하는 일이 아니라 시간을 예약하는 일

CPU offload의 핵심은 GPU 메모리 옆에 더 큰 창고를 붙이는 것이 아니다. 미래에 다시 쓸 상태를 지금 어느 비용으로 보존하고, 필요해질 때 어느 deadline 안에 다시 실행 가능한 주소로 돌려놓을지 정하는 일이다. 그래서 capacity 계산은 transfer 시간, reuse 확률, recompute 비용과 함께 있어야 한다.

정확성은 prepare와 completion 사이에서 지켜진다. host hit는 ready와 다르고, submit 성공은 copy 완료와 다르며, request finish는 transfer lifetime 종료와 다르다. logical key·physical slot·generation·reference와 stream event가 같은 전환을 증명해야 한다. 이 연결을 잃으면 성능 저하보다 더 위험한 조용한 오답이 생긴다.

네 구현을 비교하며 얻은 가장 중요한 교훈은 이름보다 이동 단위를 먼저 보라는 것이다. vLLM과 SGLang은 요청·prefix block의 계층 수명을, Transformers는 forward의 layer cache 회전을, llama.cpp의 해당 옵션은 graph node placement를 다룬다. 같은 2GiB라도 누가 key를 갖고 언제 다시 소비하는지가 다르다.

다음 장에서는 이 정상 상태 기계를 진단 도구로 뒤집는다. cache miss가 정책상 정상인지, reference leak 때문에 생겼는지, stale generation이나 position 오류가 오답을 만들었는지를 lookup부터 release까지 분리한다. offload를 이해한 독자는 이제 “cache가 이상하다”가 아니라 어느 tier의 어느 전환에서 어떤 증거가 사라졌는지를 물을 수 있다.

## 38.9 pinned·pageable·NUMA를 하나의 byte/time 원장에 넣는다

용량을 96GiB에서 192GiB로 늘린 배포가 있었다. admission 실패는 1.8%에서 0.3%로 줄었지만 ITL p99는
42ms에서 118ms로 올랐다. 운영자는 PCIe 카드와 GPU를 바꾸지 않았으므로 전송 성능은 같을 것이라고
생각했다. 그러나 새 96GiB는 GPU와 가까운 NUMA node의 pinned pool이 아니라 원격 node의 pageable
allocation이었다. 대시보드의 `host cache bytes`는 두 종류를 하나로 더했고, hit counter도 host에 key가
존재한다는 사실만 셌다. capacity는 늘었지만 runnable KV가 되는 시간은 길어졌다.

먼저 byte 원장을 네 칸으로 나눈다. `B_logical`은 cache key가 가리키는 유효 KV payload다. `B_reserved`는
allocator가 확보했지만 entry가 쓰지 않는 여유와 fragmentation까지 포함한다. `B_pinned`는 DMA에 사용할 수
있도록 page-locked 또는 등록된 host 영역이다. `B_transfer`는 promotion 한 번에서 실제 descriptor가 옮긴
byte다. 192GiB 설정이 곧 192GiB pinned도, 한 hit가 192GiB를 옮긴다는 뜻도 아니다. logical entry 2GiB가
pageable이면 driver staging 때문에 host→host copy와 temporary pin이 추가될 수 있다.

시간 원장도 대응시킨다.

```text
T_ready = T_lookup + T_host_queue + T_page_fault_or_stage
        + T_pin_or_registration + T_numa_path + T_pcie_dma
        + T_completion_poll + T_metadata_commit + T_scheduler_gap - T_overlap
```

모든 항이 매번 존재하는 것은 아니다. 미리 pinned된 local pool은 page fault, staging과 반복 registration을
거의 없앨 수 있다. pageable memory는 API가 동작하더라도 내부 staging이 critical path에 나타날 수 있다.
NUMA 항은 source CPU memory controller에서 GPU가 붙은 root complex까지 가는 inter-socket traffic,
bandwidth 경쟁과 locality penalty를 묶은 관측 칸이다. PCIe 세대나 lane 일반론은 54장에 맡기고, 여기서는
cache promotion의 어느 byte가 어느 경로를 탔는지만 센다.

2GiB fixture를 대입한다. local pinned pool에서 lookup 0.3ms, queue 9ms, DMA 94ms, completion/commit 3ms,
scheduler gap 7ms, overlap 45ms라면 exposed ready time은 `68.3ms`다. remote NUMA pageable pool에서는 lookup
0.3ms, queue 18ms, first touch/staging 24ms, remote path contention 21ms, DMA 101ms, completion/commit 5ms,
scheduler gap 9ms, overlap 45ms로 `133.3ms`다. 같은 logical hit와 같은 2GiB인데 비용은 거의 두 배다.

pageable이 항상 나쁘고 pinned가 항상 좋다는 결론도 피한다. 192GiB를 모두 pin하면 운영체제가 page를
회수하거나 이동할 자유를 잃고 다른 process의 allocation, filesystem cache와 network buffer를 압박할 수
있다. pinned region을 만들고 해제하는 비용도 있다. 재사용률이 낮은 cold tier까지 pin하면 실제 전송하지
않는 byte가 시스템 자원을 묶는다. 흔히 쓰는 hot window만 pinned L1에 두고 cold pageable 또는 storage
tier는 promotion 때 bounded staging buffer를 쓰는 설계가 합리적일 수 있다.

그때 staging buffer는 숨은 capacity ceiling이다. pinned staging이 8GiB이고 2GiB restore 네 개가 진행 중이면
다섯째 요청은 host cache가 192GiB 비어 있어도 기다린다. cache occupancy만 보면 여유지만 promotion
concurrency는 4다. 원장에 staging reserved, in-flight bytes, largest contiguous extent와 waiter 수를 둔다.
fragmentation 때문에 총 2GiB free가 있어도 alignment를 만족하는 destination이 없을 수 있다.

NUMA placement는 process CPU affinity만으로 증명되지 않는다. pool page가 어느 node에서 first-touch됐는지,
GPU가 어느 PCIe root에 연결됐는지, transfer worker thread가 어디서 descriptor와 memory를 만지는지 확인한다.
main thread를 GPU-local CPU에 bind했어도 lazy allocator worker가 다른 node에서 처음 page를 touch하면 physical
pages는 원격일 수 있다. `numactl` 설정 문자열보다 actual page placement와 measured directional bandwidth가
증거다.

- vLLM 고정 source에서 이 원장은 실제 pointer consumer로 내려간다.
- CPU offloading worker는 canonical GPU cache tensor를 byte view로 만들고 pinned CPU tensor 또는 pinned mmap region을 준비한다.
- [vLLM CPU pool construction](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/gpu_worker.py#L464-L529) transfer path는 source·destination pointer와 size descriptor를 만들고 async batch copy를 제출한다.
- [vLLM transfer descriptor consumer](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/cpu/gpu_worker.py#L240-L441) 설정된 host GiB가 latency로 변하는 경계는 pool의 실제 allocation/placement와 descriptor byte, queue/event 순서다.
- option help만 읽어서는 pinned 여부나 remote first-touch를 증명할 수 없다.

- LMCache v0.5.4의 local CPU tier도 configured capacity를 ready capacity와 구분하게 해 준다.
- lazy allocator는 처음에 CPU tensor를 pageable하게 만들고 chunk 단위로 pin을 시도하며 pin 실패를 경고한다.
- [LMCache lazy allocator pinning](https://github.com/LMCache/LMCache/blob/3e11b8ed191631e6f098b8038235823f1a410b24/lmcache/v1/memory_allocators/lazy_memory_allocator.py#L111-L155) [LMCache pin chunk 경계](https://github.com/LMCache/LMCache/blob/3e11b8ed191631e6f098b8038235823f1a410b24/lmcache/v1/memory_allocators/lazy_memory_allocator.py#L289-L352) configured CPU capacity, allocated chunks, successfully pinned chunks와 in-use objects를 나눠야 한다.
- pin 실패 뒤 fallback semantics를 확인하지 않고 “CPU cache 192GiB”라고 보고하면 promotion 비용을 설명할 수 없다.

관측은 요청별 raw address를 metric label로 내보내지 않는다. pool별 configured/reserved/pinned/live bytes,
NUMA node cohort, direction, size bucket, queue와 stage/register/DMA/commit 시간을 집계한다. sampled trace에는
logical key digest, source pool/node, pinned 여부, byte count, destination GPU generation과 consumer batch를
연결한다.

## 38.10 Preview — lookup에서 promotion까지: LMCache와 vLLM의 readiness를 잇는다

이 절은 LMCache의 전체 lifecycle을 소유하지 않는다. CPU offload와 hierarchical cache의 공통 질문인 `lookup→host ready→GPU ready→scheduler commit` frontier가 어디서 갈리는지만 미리 본다. Chunk key 생성, storage lookup·promotion·eviction과 connector의 상세 state machine은 62장에서 같은 frontier를 제품 source에 고정한다.

LMCache의 async lookup은 scheduler가 모든 storage I/O를 기다리지 않도록 하는 경로다. scheduler 쪽은 token
chunk keys와 lookup ID를 보내고 worker/storage manager는 backend contains와 retrieval을 진행한다. 그러나
`contains=true`, retrieval future 생성, CPU object 도착, GPU injection 완료는 서로 다른 사건이다. “async
hit”라는 한 상태로 접으면 scheduler가 아직 payload-ready가 아닌 길이를 computed prefix로 올릴 수 있다.

고정 source의 `LMCacheEngine.async_lookup_and_prefetch`는 요청 token과 configs를 storage manager 경로로
넘긴다. [LMCache async lookup entry](https://github.com/LMCache/LMCache/blob/3e11b8ed191631e6f098b8038235823f1a410b24/lmcache/v1/cache_engine.py#L1322-L1380)
함수가 반환되거나 future가 생겼다는 사실을 GPU readiness로 해석하지 않는다. retrieved object의 lifetime,
local CPU allocation, GPU connector가 KV slot으로 옮기는 completion과 scheduler computed token 승인을 잇는다.

prefix 8GiB를 256MiB chunk 32개로 나누자. metadata lookup은 32개 모두 존재한다고 2ms에 답했다. local CPU
tier에는 12개가 resident이고 20개는 remote에서 내려와야 한다. `logical hit=8GiB`, `CPU ready=3GiB`,
`remote pending=5GiB`다. longest contiguous ready prefix가 2.5GiB라면 뒤의 scattered ready 0.5GiB는 causal
prefix를 건너뛰어 사용할 수 없다. GPU promotion destination과 copy 완료도 남는다.

frontier를 다섯 개로 둔다. `F_lookup`은 key가 연속 존재하는 token 경계다. `F_host_ready`는 payload와 schema가
local tier에 준비된 연속 경계다. `F_gpu_reserved`는 destination KV slot을 확보한 경계다. `F_gpu_ready`는
transfer completion과 generation validation을 끝낸 경계다. `F_committed`는 scheduler/runner가 computed
prefix로 소비해도 되는 경계다. 각 frontier는 이전 것보다 작거나 같아야 한다. 취소나 validation 실패에서는
provisional frontier를 폐기한다.

vLLM offloading manager는 lookup 결과를 pending/retry와 구분하고 prepare/complete로 수명을 나눈다.
[vLLM offloading manager contract](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_offload/base.py#L220-L320)
LMCache와 통합된 경로에서는 LMCache lookup ID와 vLLM request/block generation, host MemoryObj와 GPU
destination, scheduler computed frontier를 하나의 correlation record로 잇는다. 두 제품 metric의 timestamp를
겹치는 것만으로 같은 object를 증명하지 않는다.

promotion policy는 “찾으면 올린다”보다 복잡하다. 요청이 다시 scheduled될 확률, deadline, bytes,
destination pressure, remote remaining time과 recompute cost를 본다. 8GiB 전체를 기다리기보다 first 2GiB만
올리고 나머지를 recompute하는 혼합 정책은 backend가 partial ready prefix를 정확히 표현할 때만 가능하다.
source에 없는 speculative promotion을 기능처럼 단정하지 않는다.

promotion 도중 eviction은 두 tier를 흔든다. GPU destination을 만들려고 GPU block Q를 host로 demote하면 P의
H2D와 Q의 D2H가 링크와 staging pool을 경쟁한다. host가 가득 차 Q를 remote로 내리면 한 promotion이 연쇄
demotion 둘을 만든다. 원장에 cause job ID를 둬 `P promotion → Q demotion → host victim R remote store`를
한 tree로 묶는다. 각 job을 독립 hit/miss로 집계하면 amplification을 놓친다.

lookup race도 있다. t0 cache lookup을 보내고 t1 timeout으로 local recompute를 시작한다. t2 remote hit가
돌아와 H2D를 시작하고 t3 recompute가 같은 GPU slots에 KV를 쓴다. 하나를 winner로 정하고 loser의 transfer,
reference와 reservation을 취소해야 한다. generation fence 없이 둘을 완료시키면 마지막 writer에 따라 logits가
달라진다. lookup ID, request incarnation, destination generation과 winner state가 필요하다.

관측 화면에서는 promotion funnel을 본다. requested bytes 중 key-hit, schema-valid, host-ready, GPU-reserved,
GPU-ready, committed bytes가 얼마인지 센다. drop reason은 miss, timeout, pressure, validation mismatch, cancel,
recompute-won으로 bounded한다. ITL과 연결할 때는 `F_committed`가 consumer deadline 전에 도달했는지를 본다.

## 38.11 용량 증가가 ITL을 악화시킨 손익분기 사건

사건의 배포 전 상태를 고정한다. GPU KV pool은 24GiB, host hot tier는 local NUMA pinned 32GiB다. decode
요청 48개가 round-robin으로 실행되고, 각 요청의 cold tail candidate는 평균 512MiB다. host tier hit율은
58%, GPU OOM에 가까운 admission reject는 1.6%, ITL p50/p99는 23/47ms다. 팀은 reject를 없애려고 host
capacity를 128GiB로 늘렸다. local node에 남은 큰 contiguous memory가 부족해 추가 96GiB는 remote NUMA의
pageable pool로 배치됐다.

배포 뒤 host logical hit율은 58%에서 86%로 올랐고 reject는 0.2%로 줄었다. raw throughput도 4% 늘었다.
하지만 ITL p50은 25ms로 거의 같고 p99는 126ms가 됐다. 특히 512MiB와 1GiB restore가 decode 사이에 들어간
요청에서 spike가 몰렸다. “cache hit가 늘었으니 kernel이 문제”라는 첫 가설과 “요청을 더 받아 queue가
길어졌을 뿐”이라는 둘째 가설이 경쟁했다.

byte/time 원장으로 10분 trace를 다시 계산했다. 기존 local pinned 32GiB에서 promotion된 payload는 420GiB,
remote pageable tier에서는 690GiB였다. local 512MiB restore의 median은 lookup 0.2ms, queue 2.7ms, DMA 22ms,
commit/scheduler 3.1ms, overlap 15ms여서 exposed 13ms였다. remote pageable restore는 lookup 0.2ms, queue
11ms, staging/first-touch 8ms, NUMA contention 9ms, DMA 27ms, commit/scheduler 5ms, overlap 15ms여서 exposed
45.2ms였다. p99 queue와 staging을 넣으면 한 번에 91ms가 노출됐다.

한 요청이 decode 20 token 동안 평균 2.4회 remote promotion을 겪었다. 모든 token에 고르게 분산되지 않고
특정 token 앞에서 45~91ms가 추가되므로 p50은 거의 안 변하고 p99만 뛴다. scheduler는 host hit request를
admit한 뒤 GPU destination이 날 때까지 기다렸고, destination을 만들려고 다른 request의 GPU block을 host로
내렸다. urgent H2D 512MiB와 background D2H 512MiB가 같은 staging/링크 자원을 다투는 순간도 있었다.
capacity 확대가 lookup hit, promotion 빈도와 cascade demotion을 동시에 늘린 것이다.

kernel 가설은 같은 batch shape의 attention duration이 배포 전후 오차 범위였고 ITL spike가 kernel launch
전 `F_host_ready→F_committed` 구간에 놓인 것으로 반증됐다. 단순 queue 가설은 offload가 없는 canary에서
동일 active request 48개를 유지해도 p99 51ms였고, remote promotion byte와 spike가 request 단위로 정렬된
것으로 약해졌다. NUMA 가설은 remote pool을 사용한 요청만 p99가 높고 같은 key를 local pinned staging으로
복사한 대조군에서 60ms 아래로 내려간 것으로 지지됐다.

이제 손익분기를 식으로 닫는다. suffix 512MiB를 재계산하는 데 38ms가 걸린다고 하자. local pinned restore의
exposed 13ms는 25ms 이득이다. remote pageable median 45.2ms는 재계산보다 7.2ms 손해이며 p99에서는 53ms
이상 손해다. demote store가 background overlap 뒤에도 request-equivalent interference 4ms를 만든다면 reuse
확률 `p`에서 remote offload 기대 비용은 `4 + p×45.2`, recompute는 `p×38`이다. 두 식이 같아지는 양의
`p`가 없다. restore 자체가 재계산보다 느리므로 reuse가 늘수록 손해가 커진다.

local pinned tier는 다르다. store interference가 3ms라면 `3+p×13 < p×38`, 즉 `p>3/25=0.12`에서
offload가 유리하다. 이 workload의 local reuse 0.58은 threshold를 넘는다. 따라서 전체 offload on/off가
아니라 local pinned tier는 유지하고 remote pageable tier의 decode-critical promotion은 금지하는 결론이
나온다. remote tier는 긴 prefill이나 deadline이 느슨해 45ms를 overlap할 수 있는 workload, 혹은 재계산이
훨씬 비싼 suffix에만 조건부로 쓴다.

capacity 이득도 돈으로 환산한다. 배포 전 1.6% reject 중 deadline 내 재시도 성공을 포함한 lost goodput이
분당 12 requests였다. 확대 뒤 reject loss는 분당 1.5로 줄었지만 ITL deadline violation이 분당 19건
늘었다. accepted count는 늘었어도 SLO-valid completed goodput은 분당 8.5건 줄었다. raw throughput +4%가
잘못된 성공 신호였던 이유다. 제품 분자는 accepted가 아니라 deadline과 correctness를 만족한 completion이어야
한다.

수정 정책은 세 단계다. 첫째 restore destination용 GPU 2GiB를 reserve해 promotion이 즉시 demotion을 부르는
cascade를 줄인다. 둘째 H2D urgent queue를 background D2H store보다 먼저 처리하되 starvation 방지를 위한
store byte quantum을 둔다. 셋째 candidate마다 `T_restore_predicted < min(T_recompute, deadline_slack)`일 때만
remote promotion을 허용한다. predicted time은 size bucket, pinned/pageable, NUMA node, queue depth와 recent
bandwidth에서 얻는다. hit 여부 하나로 선택하지 않는다.

LMCache 쪽에는 local CPU object가 어느 allocator chunk와 pin 상태를 갖는지, async lookup/prefetch의 hit가
어느 readiness frontier까지 왔는지 기록한다. vLLM 쪽에는 offloading manager prepare/complete와 destination
generation, runnable transition을 기록한다. 두 trace를 lookup ID/request incarnation으로 잇는다. LMCache가
8 chunks hit라고 보고한 시각과 vLLM이 실제 8 chunks를 computed로 commit한 시각의 차이가 exposed promotion
time이다. integration 경계가 이 차이를 소유한다.

canary는 세 cohort다. A는 local pinned hit, B는 remote pageable hit, C는 forced recompute다. 같은 prompt
length, adapter/model, scheduler priority와 arrival을 쓴다. A에서는 restore bytes와 ITL 이득을, B에서는
prediction이 remote hit를 거절하거나 충분한 slack에서만 선택하는지를, C에서는 recompute baseline과 GPU
pressure를 본다. cache warmness가 다르면 hit와 recompute의 logical work가 달라지므로 prefill tokens와 output
contract를 함께 기록한다.

failure injection은 pin 실패, remote NUMA saturation, H2D queue backlog, destination allocation 실패와
lookup timeout을 각각 넣는다. pin 실패 뒤 entry를 pinned라고 label하지 않는지, timeout 뒤 recompute winner가
late promotion을 폐기하는지, allocation 실패가 host ref와 staging reservation을 되돌리는지 확인한다. fault
후 `pending bytes=0`, `provisional refs=0`, `destination generation unchanged`, output parity를 통과해야 한다.

rollback은 host capacity 값을 32GiB로 되돌리는 한 줄이 아니다. 새 remote admission과 promotion을 멈추고
in-flight lookup, staging과 DMA를 drain한다. remote-only payload를 가진 active request는 deadline과 state에
따라 완료 restore 또는 recompute로 명시한다. live refs가 있는 host objects를 pool shrink가 먼저 해제하지
않게 한다. NUMA pageable arena를 unmap하기 전에 transfer callbacks와 memoryviews가 끝났는지 확인한다.
마지막으로 local pinned cohort의 hit/ITL과 baseline goodput이 복원됐는지 본다.

사건 보고의 원인은 “CPU cache가 너무 컸다”가 아니다. 정확한 인과는 `remote pageable capacity 증가 →
logical hit와 accepted working set 증가 → staging/NUMA promotion 및 cascade demotion 증가 →
F_host_ready에서 F_committed까지의 critical path 노출 → decode token의 ITL tail 증가`다. capacity 자체가
아니라 느린 tier를 runnable capacity처럼 admission한 정책이 문제였다.

## 38.12 cache hierarchy decision을 판정하고 재검증한다

38장의 소유 범위는 bus 전체가 아니다. CPU socket과 GPU topology의 일반 구조, PCIe 세대별 lane 대역폭,
IOMMU와 chipset 설명은 54장에서 다룬다. 여기서는 cache manager가 어느 tier의 bytes를 어떤 deadline에
소비 가능한 GPU state로 바꾸며, 그 선택이 재계산과 admission보다 이로운지만 판정한다. 같은 물리 지식을
반복하기보다 decision owner를 분명히 한다.

독자가 처음 작성할 문서는 표가 아니라 요청 한 건의 장면이다. “A가 decode token 17 직전에 host hit를
얻었지만 91ms 동안 다음 token을 내지 못했다.” 이어 직관을 쓴다. “창고에 물건이 있다는 사실과 작업대에
도착했다는 사실은 다르다.” 그 다음에만 exact frontier, byte/time 식과 source consumer를 붙인다. 이렇게
하면 용어를 모르던 독자도 왜 lookup hit가 ITL을 보장하지 않는지 먼저 이해한다.

정확한 기계는 여섯 전환이다. lookup이 logical 후보를 찾는다. host backend가 schema-valid payload를 pin해
eviction에서 보호한다. GPU allocator가 destination generation을 예약한다. transfer worker가 byte descriptor를
제출하고 completion을 관측한다. manager가 physical mapping과 computed frontier를 commit한다. scheduler와
runner가 attention consumer를 launch한다. finish/cancel은 반대 방향으로 refs와 reservations를 회수한다.
submit과 complete, host-ready와 GPU-ready를 합치지 않는다.

source 산책은 이 전환마다 producer와 consumer를 찾는다. vLLM `OffloadingManager`의 lookup/prepare/complete,
CPU worker의 pinned pool과 transfer descriptors, direction handler의 event completion을 연결한다. LMCache
engine의 async lookup entry, storage manager의 retrieval, local CPU allocator의 pin chunk와 connector의
GPU injection을 연결한다. option parser나 문서 예제는 시작점이고, state mutation과 다음 consumer가
고정 source 근거다.

수치 worksheet는 본문을 대체하지 않고 사건을 검산한다.

```text
payload/request       = 512 MiB
host kind             = remote pageable
lookup/queue/stage     = 0.2 + 11 + 8 ms
NUMA/DMA/commit        = 9 + 27 + 5 ms
overlap                = 15 ms
exposed restore        = 45.2 ms
same-suffix recompute  = 38 ms
store interference     = 4 ms
decision               = decode-critical restore reject
```

이 worksheet의 숫자는 반드시 같은 request와 같은 workload bucket에서 와야 한다. 평균 DMA bandwidth와 p99
queue, 다른 길이의 recompute를 섞어 가상의 손익분기를 만들지 않는다. GiB/GB 단위를 명시하고 transfer bytes가
logical bytes와 같은지 compression, padding, multi-layer layout을 확인한다. overlap은 총 compute 시간이 아니라
실제로 restore와 동시에 진행된 구간만 뺀다.

capacity를 평가할 때 세 denominator를 둔다. configured host bytes는 운영자가 요청한 상한이다. usable
resident bytes는 allocator/alignment/metadata를 제외하고 entry가 쓸 수 있는 공간이다. deadline-ready bytes는
현재 queue와 tier에서 요청의 소비 시점 전에 GPU로 올릴 수 있는 양이다. 192GiB configured가 120GiB usable,
동시에 8GiB만 staging 가능하고 deadline-ready가 2GiB일 수 있다. admission은 마지막 값을 무시하면 안 된다.

observability는 counter보다 인과 edge를 보존한다. lookup ID와 request incarnation, key digest, tier/node,
pinned state, source/destination generation, bytes, queue/submit/end/commit/consume timestamp를 sampled trace에
둔다. metrics에는 size/tier/direction/reason 같은 bounded label만 쓴다. dashboard는 logical hit율 옆에 host-ready,
GPU-ready, committed ratio와 exposed restore distribution을 둔다. 높은 hit와 낮은 readiness가 한눈에 보여야
한다.

성능 최적화 전에 correctness terminal을 확인한다. source last writer 뒤 store가 시작됐는가. host payload가
complete된 뒤 lookup에 공개됐는가. H2D completion 뒤 GPU mapping이 commit됐는가. consumer stream이 readiness를
관측했는가. lookup/protected/copied generations가 같은가. cancel과 timeout의 late completion이 새 owner를
건드리지 않는가. 이 질문 중 하나라도 답이 없으면 더 큰 pool이나 더 많은 copy worker를 배포하지 않는다.

그 다음 latency terminal을 확인한다. restore size bucket별 `T_ready`가 recompute와 deadline slack보다 작은가.
urgent H2D가 background store 뒤에 무한히 서지 않는가. destination reserve가 cascade를 막는가. NUMA local
placement가 실제 page와 GPU topology에서 성립하는가. pinned pool이 작아 반복 staging을 만들거나 너무 커서
OS pressure를 만들지 않는가. 평균 대신 ITL spike 직전 promotion timeline을 본다.

capacity terminal은 OOM 감소만 보지 않는다. deadline-valid completion goodput이 늘었는가. host pool 증가로
admission한 working set이 promotion bandwidth를 초과하지 않는가. reuse하지 않을 entry의 store가 링크와
pinned bytes를 낭비하지 않는가. cache eviction과 promotion이 서로를 유발하는 amplification ratio는 bounded인가.
용량을 더했는데 이 세 답이 나빠지면 tier는 storage일 뿐 serving capacity가 아니다.

rollout은 local pinned hot tier부터 시작하고, 긴 shared prefix처럼 재계산 비용이 큰 bucket에만 연다. remote
pageable 또는 storage tier는 lookup만 shadow해 expected hit와 predicted restore를 측정한 뒤 실제 promotion을
canary로 연다. capacity, policy, priority, staging size를 동시에 바꾸지 않는다. 한 변화의 expected byte/time
mutation을 먼저 쓰고 관측이 맞는지 확인한다.

rollback terminal은 새 promotion 중단, in-flight drain, active host-only request 처리, refs와 generation 정리,
arena/mmap release와 baseline goodput 복원이다. process kill은 pending DMA와 shared mapping lifetime을 설명하지
못하므로 정상 rollback 절차의 대체가 아니다. emergency restart가 필요했다면 재기동 뒤 stale external
metadata와 shared region generation을 검증한다.

실제 review에서는 비용 모델의 각 항이 측정 가능한지 역으로 확인한다. `T_lookup`은 scheduler가 요청을 보낸 시각부터 prefix length 응답을 받은 시각이다. backend 내부 remote lookup만 재면 IPC와 queue가 빠진다. `T_host_queue`는 payload retrieval 또는 staging job이 runnable해질 때까지 기다린 시간이다. `T_stage`는 pageable source에서 pinned buffer로 복사하고 필요한 registration을 마칠 때까지다. `T_dma`는 같은 transfer stream의 start/end event 범위이며, `T_commit`은 CPU가 completion을 관측한 뒤 mapping을 공개할 때까지다. `T_scheduler_gap`은 commit 뒤 실제 selected batch까지다.

이름만 맞추지 말고 timestamp owner와 clock domain도 적는다.

CPU monotonic clock과 CUDA event time을 그대로 빼면 안 된다. CUDA event는 stream 구간 duration으로 쓰고,
CPU timeline에는 submit과 completion callback timestamp를 둔다. 둘 사이 차이는 queue, launch와 polling을
포함할 수 있다. 분산 LMCache worker의 clock이 다르면 absolute timestamp보다 lookup ID별 causal sequence와
각 process duration을 사용하거나 clock synchronization 오차를 기록한다. 3ms 개선을 주장하면서 clock skew가
5ms이면 결론을 보류한다.

byte도 같은 원칙을 쓴다. KV logical tensor byte는 layer, K/V, head, dimension, token, dtype의 곱이다. transfer
descriptor byte는 layout padding, block row stride와 선택 layer 때문에 다를 수 있다. host allocator reserved
byte는 alignment와 free chunk를 포함한다. PCIe profiler byte에는 staging copy가 없고 CPU memory bandwidth에는
나타날 수 있다. 서로 다른 byte를 한 denominator로 나누어 bandwidth를 만들면 pageable path가 실제보다
빠르거나 느리게 보인다.

NUMA incident를 재현할 때 production node 전체를 먼저 흔들지 않는다. 동일 512MiB object를 local pinned,
remote pinned, local pageable, remote pageable 네 cohort에 배치한다. lookup과 GPU destination 조건을 고정하고
promotion 1개, 4개, 16개 concurrency에서 queue와 exposed time을 잰다. local pinned와 remote pinned 차이는
주로 placement/competition을, local pinned와 local pageable 차이는 staging/pinning을 드러낸다. 네 효과가
상호작용하므로 단일 평균만으로 분리하지 않는다.

그 다음 실제 scheduler trace를 replay한다. microbenchmark의 최대 bandwidth가 좋아도 urgent 512MiB load가
2GiB background store 뒤에 서면 ITL은 나빠진다. copy chunk를 작게 하면 urgent job이 끼어들 기회는 늘지만
descriptor와 event overhead가 커진다. chunk 64MiB라면 2GiB store는 32개 scheduling point를 가지며, 256MiB면
8개다. policy는 aggregate GB/s와 urgent p99를 함께 최적화한다. source가 job 중간 preemption을 지원하지
않으면 chunk boundary만 우선순위 전환점이다.

destination reserve도 공짜가 아니다. GPU 2GiB를 restore 전용으로 비우면 resident KV capacity가 줄어 OOM 또는
eviction이 먼저 올 수 있다. reserve가 없으면 cascade demotion이 ITL을 늘린다. arrival trace에서 reserve
0, 1, 2, 4GiB를 비교해 deadline-valid completion이 최대인 지점을 찾는다. GPU free byte가 아니라 largest
allocatable block와 cache group별 destination availability를 본다. 한 group만 부족하면 전체 prefix promotion이
막힐 수 있다.

promotion 거절도 실패가 아니다. predicted remote restore 60ms, recompute 38ms라면 safe miss로 전환해
recompute하는 것이 올바른 선택이다. metric은 이를 backend miss와 구분해 `bypass_cost_model`로 센다. hit율이
낮아져도 ITL과 goodput이 좋아질 수 있다. 운영 목표가 cache hit 최대화에서 deadline-valid work 최대화로
바뀌어야 이 선택이 dashboard에서 회귀처럼 보이지 않는다.

반대로 long shared prefix는 remote restore가 유리할 수 있다. 8GiB restore exposed 310ms, recompute 920ms,
deadline slack 500ms라면 610ms compute를 아끼며 deadline도 지킨다. 같은 remote pageable tier라도 512MiB
decode tail은 bypass하고 8GiB prefix는 promote할 수 있다. threshold는 byte 하나가 아니라 predicted transfer,
recompute, slack과 link interference 함수다. workload bucket이 바뀌면 다시 학습·측정하되 correctness fence는
바꾸지 않는다.

마지막으로 capacity 확대 전후의 cohort identity를 고정한다. pool이 커지면 scheduler가 더 긴 요청과 더 많은
동시 요청을 받아 workload 자체가 달라진다. 같은 offered arrival를 closed replay한 controlled test와 새로운
admission을 포함한 product test를 분리한다. controlled test는 mechanism latency를, product test는 reject와
deadline goodput을 보여 준다. 둘을 섞으면 “같은 요청이 느려졌다”와 “더 어려운 요청을 받아 평균이
달라졌다”를 구분할 수 없다.

배포 승인 기록에는 threshold의 숫자뿐 아니라 유효 범위를 적는다. “512MiB 이하 decode tail은 remote
pageable promotion을 bypass한다”는 규칙은 당시 model KV layout, CUDA/driver, GPU와 CPU NUMA placement,
staging 8GiB, concurrency 48과 H2D priority에서 나온다. GPU, chunk 크기나 worker 수가 달라지면 `T_ready`
항을 다시 잰다. rule을 영구 상수로 두지 않고 versioned policy와 측정 artifact를 연결한다.

정책 예측이 틀릴 수도 있으므로 actual/predicted를 함께 남긴다. predicted 30ms인데 actual 85ms라면 bandwidth
모델보다 queue, staging, NUMA 또는 scheduler gap이 빠졌는지 본다. 실제가 반복해서 threshold를 넘으면 해당
tier·size cohort를 circuit-break하고 recompute로 보낸다. 한 번의 outlier로 전체 cache를 끄지 않지만,
generation이나 output mismatch는 비용 문제가 아니므로 즉시 unsafe namespace를 차단한다.

capacity를 줄일 때도 byte 원장이 필요하다. configured 128GiB를 32GiB로 내리기 전에 live 74GiB, pinned
96GiB, in-flight 6GiB라면 즉시 shrink할 수 없다. 새 store를 중단하고 cold unreferenced entries를 evict하며
in-flight completion을 기다린다. allocator가 arena 전체 pin을 유지하면 logical live가 32GiB 아래여도 pinned
reservation은 process restart나 pool rebuild 전까지 줄지 않을 수 있다. 이 차이를 leak으로 오판하지 않되
rollback에 재구성 시점을 명시한다.

마지막 검산은 conservation으로 한다. 창 시작 tier별 live bytes에 committed store를 더하고 ownership 이동과
eviction/free를 반영하면 창 끝 live bytes와 맞아야 한다. in-flight와 provisional은 별도 부채다. submitted
transfer bytes와 completed success/failure bytes도 보존돼야 한다. 차이가 남으면 성능 그래프보다 먼저 누락
event, ref leak 또는 double accounting을 찾는다.

당직자가 즉시 사용할 최소 대조군도 남긴다. 같은 request를 local pinned hit, remote/pageable hit, forced
recompute 세 경로로 보내 output과 selected prefix를 맞춘다. 첫 경로가 빠르면 host tier 자체는 유효하고,
둘째만 느리면 placement/staging 비용을 좁힐 수 있다. 셋째가 둘째보다 빠르면 hit를 보존하려 애쓰기보다
cost-model bypass가 정답이다. 세 경로 모두 느리면 scheduler queue나 model compute처럼 promotion 밖의
공통 경계를 본다.

이 대조는 실제로 기능을 실행하지 못하는 source review에서도 설계할 수 있다. pinned revision에서 각 경로의
option consumer, prepare/complete state와 fallback을 찾고 예상 trace를 쓴다. 나중에 환경이 준비되면 같은
fixture로 숫자만 채운다. source에서 지원하지 않는 fallback을 있다고 가정하거나, 실행하지 않은 결과를
성능 사실로 쓰지 않는다. 문서는 capability evidence와 runtime evidence를 분리한다.

결정 후에도 cold workload를 잊지 않는다. hot prefix가 많던 주간에는 local pinned tier가 이로웠지만 traffic이
private conversation으로 바뀌면 store만 늘 수 있다. reuse bucket, exposed restore/recompute ratio와 deadline
goodput을 지속 관측하고 threshold 재평가 조건을 둔다. cache hierarchy는 설치 후 고정된 용량 장치가 아니라
workload가 바뀔 때마다 미래 사용 시간을 다시 가격 매기는 정책이다.

마지막 회고 질문은 네 개면 충분하다. 이 entry는 몇 byte이며 어느 tier와 NUMA node에 실제 resident인가.
다음 소비 deadline 전에 pinned/staged/transfer/commit을 끝낼 수 있는가. 같은 suffix 재계산과 비교해 얼마나
이익이며 그 사이 다른 요청에 준 interference는 얼마인가. 취소·eviction·timeout에서도 logical identity와
physical generation이 정확히 한 owner에게 돌아가는가. 네 답을 같은 trace와 pinned source로 설명할 수 있을
때 host capacity 증가는 비로소 검증된 serving 최적화가 된다.
