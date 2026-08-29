# 37장. free처럼 보이는 block은 언제 다시 쓸 수 있는가: allocation·reference·eviction·rollback

새 요청 T가 KV blocks 세 개를 요구한다. pool에는 “free”라고 표시된 blocks가 세 개 보인다. allocator는 이를 T에게 주었지만 직후 다른 요청의 output이 깨진다. 하나는 prefix cache에 남아 있던 eviction candidate였고, 하나는 reference count가 잘못 0이 되었으며, 하나는 abort된 요청의 GPU write가 아직 끝나지 않은 delayed resource였다. 숫자는 셋 모두 free였지만 재사용 자격은 달랐다.

이 장의 중심 질문은 “메모리가 얼마나 필요한가”가 아니다. allocation이 어떤 owner를 만들고, reference·lock·pin이 무엇을 보호하며, 0이 된 resource가 언제 eviction candidate가 되고, policy가 무엇을 희생시키며, 중간 실패가 어떤 mutation을 되돌려야 하는가다. 33장의 byte 계산, 34장의 주소 translation, 35장의 prefix hash는 전제로 두고 수명 상태만 따른다.

고정 source는 vLLM `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang `71de97b264b04dcd514cf904003028aefe9775c8`, Transformers `550d7b3834670483a4df436541272c055dc364bf`, llama.cpp `bb4caa7540188872173c44d161602d9271386413`이다. 네 구현은 block pool, radix lock, initialized cache, cell metadata라는 서로 다른 owner model을 쓰므로 field 이름을 억지로 맞추지 않는다.

## 37.1 여덟 resources가 있는데 T는 왜 기다리는가

### 37.1.1 fixture의 owner를 먼저 적는다

물리 resources가 B0–B7 여덟 개라고 하자. cached free candidates C0,C1,C2가 B0–B2를 가리킨다. 실행 중 R은 shared prefix B3과 private tail B4를 갖고, S도 B3을 공유하며 B5를 갖는다. B6,B7은 content identity가 없는 uninitialized free다. B3 refcount는 2, B4와 B5는 1이다.

T가 세 blocks를 요구하면 B6,B7 두 개는 즉시 allocation할 수 있다. 세 번째는 cached candidates 가운데 eviction eligible한 하나를 policy로 선택해야 한다. B0–B2가 ref 0이고 in-flight writer나 external pin이 없다면 content metadata를 evict하고 physical block을 T generation으로 전환할 수 있다. R/S의 B3–B5는 live owner가 있으므로 후보가 아니다.

여기서 cached free라는 말은 물리 bytes가 비어 있다는 뜻이 아니다. prefix content가 남아 있고 lookup index가 이를 가리키지만 active request reference가 0이라 새 allocation 압력에서 희생될 수 있다는 뜻이다. eviction 전까지 새 matching request가 다시 touch해 active로 만들 수 있다. free queue membership과 content identity가 공존한다.

### 37.1.2 refcount 0은 정책 결정 전의 자격이다

refcount가 0이면 active reader가 없다는 중요한 조건을 만족한다. 그러나 allocator가 즉시 bytes를 덮었다는 뜻은 아니다. cached map과 eviction order에 남아 다음 hit를 제공할 수 있다. policy가 B1을 victim으로 골라 metadata를 제거하고 free queue에서 allocation owner로 넘길 때 physical reuse가 commit된다.

반대로 refcount 1이면 eviction policy가 오래되었다고 판단해도 live owner를 희생시키면 안 된다. policy는 eligible set 안에서 순서를 정할 뿐 eligibility predicate를 무시할 권한이 없다. LRU, FIFO, priority는 “누가 먼저”를 답하고 reference·lock·writer fence는 “누가 후보인가”를 답한다.

pin과 lock도 숫자 하나로 합치지 않는다. request active reference, radix node ancestor lock, connector transfer pin, async writer fence는 서로 다른 release event를 갖는다. aggregate ref가 1이라는 값만 남기면 누가 언제 decrement해야 하는지 알 수 없다. owner kind와 generation을 기록한다.

### 37.1.3 비유의 한계: 도서 대출보다 transaction에 가깝다

cache block을 도서관 책에 비유하면 refcount는 대출자 수, eviction은 서가에서 책을 빼는 일처럼 보인다. 하지만 KV block에는 GPU writer가 있고, 같은 physical ID가 새 generation으로 즉시 재사용되며, table metadata와 content index가 별도다. 반납 버튼을 눌렀어도 과거 kernel이 쓰고 있을 수 있다.

더 정확한 모델은 owner transition transaction이다. `cached eligible → selected victim → metadata detached → allocated to T → table committed → writer submitted` 단계가 있다. 어느 단계에서 실패했는지에 따라 rollback이 다르다. 책을 다시 꽂는 단일 동작으로 설명할 수 없다.

## 37.2 allocation은 block을 꺼내는 한 함수가 아니다

### 37.2.1 reserve, claim, publish를 나눈다

T allocation을 세 단계로 나눈다. reserve는 후보 resources가 충분한지 확인하고 B6,B7과 victim B1을 확보한다. claim은 old cached identity를 제거하고 T generation과 ref owner를 설치한다. publish는 T request table과 runner metadata에 `[6,7,1]`을 공개한다. publish 뒤에는 kernel이 이 IDs를 읽을 수 있다.

reserve 실패는 외부 state를 바꾸지 않거나 임시 selection을 놓아야 한다. claim 뒤 publish 실패는 ref, table, cached metadata의 native rollback 규칙을 따른다. kernel submit 뒤 실패는 단순 rollback할 수 없고 completion fence 뒤 cleanup해야 한다. allocation 성공 return 하나로 이 세 경계를 가리지 않는다.

multi-group KV에서는 group별 resources가 모두 필요하다. group A가 세 blocks를 얻고 group B가 두 개 뒤 실패하면 A만 publish할 수 없다. 이번 transaction이 새로 얻은 A/B resources를 역순 해제하고 touch/ref mutation을 되돌린다. 다른 request가 원래 가진 shared refs까지 감소시키지 않는다.

### 37.2.2 capacity check와 실제 claim 사이 경쟁을 본다

`has_enough`가 true였다고 뒤의 claim이 반드시 성공하는 것은 아니다. 같은 allocator를 concurrent path가 쓰거나 cached candidate가 lookup hit로 touch되어 active가 될 수 있다. 구현이 single scheduler thread로 serialization하는지 lock/atomic claim을 쓰는지 확인한다. check와 mutation이 분리됐으면 실패 반환을 정상 경로로 다룬다.

watermark와 reservation policy는 물리 free가 있어도 admission을 거절할 수 있다. 이는 allocation failure가 아니라 headroom을 지키는 policy decision일 수 있다. 운영자는 free count와 eligible candidates, reserved blocks, watermark를 함께 본다. “세 개가 보이는데 왜 T가 waiting인가”의 답이 correctness fence일 수도 정책일 수도 있다.

### 37.2.3 touch는 hit를 eviction 후보에서 active owner로 옮긴다

prefix lookup가 B0 hit를 찾은 뒤 allocation 단계에서 이를 touch해야 eviction queue에서 제거하고 ref를 올린다. lookup과 touch 사이 victim selection가 B0을 가져가면 stale hit가 된다. content hash equality만으로 physical lifetime을 보장하지 않는다.

vLLM single-type manager는 computed blocks가 free queue의 eviction candidate일 수 있음을 고려해 필요한 new blocks 수에서 evictable hit blocks를 계산하고, allocation commit에서 computed blocks를 touch한다. [allocation eligibility와 touch](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/single_type_kv_cache_manager.py#L200-L286)를 함께 읽어야 hit와 owner 전환을 볼 수 있다.

touch가 너무 이르면 뒤 allocation 실패 때 ref/pin을 rollback해야 한다. 너무 늦으면 victim selection와 경쟁한다. transaction ledger에는 touched existing blocks와 newly allocated blocks를 따로 기록한다. rollback은 new blocks free와 existing touch decrement를 각각 수행한다.

### 37.2.4 ownership commit의 정확한 순간을 찾는다

allocation 함수가 block IDs를 반환하는 순간과 request가 owner가 되는 순간은 구현 contract에 따라 같을 수도 다를 수도 있다. pool에서 ID를 pop하면 다른 allocator는 못 쓰지만 request table에 아직 없을 수 있다. 이 중간에는 allocator transaction가 owner다. request refcount를 먼저 올리는 구현이라면 request가 cleanup responsibility를 넘겨받는 commit marker가 필요하다.

fixture T에서 reserve owner가 B6/B7/B0을 잡은 상태는 externally invisible하다. request object에 임시 blocks list를 달더라도 scheduler running set과 runner table에는 없다. abort가 이때 오면 scheduler general free가 아니라 allocation transaction cancel가 처리하는 편이 자연스럽다. 두 cleanup가 동시에 같은 IDs를 소유하지 않게 handoff를 명시한다.

claim 뒤 request table publish 전에는 physical content identity가 이미 T generation으로 바뀌었을 수 있다. 이때 실패하면 old cached B0을 hit index에 되살릴 수 있는지는 native metadata log에 달렸다. bytes가 남았다는 이유로 hash만 복원하면 claim 과정의 write/zeroing가 content를 바꾸었을 수 있다. content validity를 증명하지 못하면 uninitialized로 둔다.

publish commit는 runner가 metadata를 관측할 수 있는 queue handoff와 연결된다. host request table list에 append한 것만으로 executor가 봤다는 뜻은 아니다. scheduler output object 생성, queue put, worker dequeue 가운데 cancellation가 가능한 boundary를 찾는다. queue에서 remove할 수 없는 뒤라면 submitted cleanup로 취급한다.

ownership commit를 event 로그로 남길 때 `allocated B6` 같은 한 문장 대신 transaction 21 reserve, T ref claim, scheduler generation 71 publish, worker generation 71 accept를 구분한다. failure handler가 어느 event까지 commit됐는지 보고 inverse path를 고른다. timestamp보다 monotonic IDs로 causal order를 잇는다.

multi-group에서는 final group 성공이 전체 commit 조건일 수 있다. group 0 request ref를 일찍 설치해도 group 2 실패 전에는 runner publish하지 않는다. coordinator가 partial results를 소유하다 전부 성공하면 request owner에게 한꺼번에 넘긴다. manager별 code가 ref를 즉시 올린다면 coordinator rollback가 정확히 그 refs를 알아야 한다.

ownership commit 뒤 metrics도 이동한다. reserved gauge는 줄고 active allocated가 늘며 cached eviction count가 확정된다. 이 metric order가 implementation mutation보다 앞서면 dashboard가 존재하지 않는 capacity를 보여 줄 수 있다. metric은 owner state commit 뒤 같은 transaction context에서 갱신하거나 sequence를 붙인다.

commit marker가 모호하면 defensive cleanup가 over-free 또는 leak 중 하나를 선택하게 된다. “혹시 allocation됐으면 free”는 live block을 두 번 놓을 수 있고 “확실하지 않으면 아무것도 안 함”은 orphan을 만든다. source audit의 핵심 산출물은 함수 목록보다 이 handoff predicate다.

### 37.2.5 eligibility snapshot은 claim 순간까지 유효해야 한다

B0이 ref 0이어서 victim eligible했지만 T claim 직전 U prefix lookup가 B0을 touch할 수 있다. single-thread scheduler라면 두 사건의 program order가 serialization를 제공한다. concurrent manager라면 victim selection에서 claim까지 lock 또는 compare-and-transition가 필요하다. old snapshot의 ref 0을 믿고 current ref 1을 evict하지 않는다.

radix node lock도 같다. policy list에서 unlocked였던 node가 request match로 lock될 수 있다. evict loop는 실제 deletion 직전 lock predicate를 다시 확인하거나 list mutation가 atomic하게 eligibility를 반영해야 한다. candidate collection와 free callback 사이 gap을 본다.

Transformers `has_enough_free_blocks()`가 initialized victims를 uninitialize하는 side effect는 snapshot보다 강한 claim에 가깝다. hash map에서 제거해 다른 lookup가 touch하지 못하게 만든 뒤 deque로 넘긴다. caller가 이를 pure capacity probe로 여러 번 호출하면 eligibility state를 소비할 수 있다. 이름이 아니라 mutation을 기준으로 owner를 판단한다.

llama.cpp `find_slot()` search도 cells empty snapshot 뒤 실제 ubatch placement까지 다른 mutation가 끼지 않는 event-loop serialization가 필요하다. chosen indices가 returned 뒤 다른 batch가 먼저 쓰면 double allocation이다. slot result consumer와 scheduler ordering을 따라간다.

eligibility 재검사는 성능 비용이 있지만 live eviction보다 싸다. global lock 대신 queue removal와 ref transition를 같은 critical section에 두거나 version counter로 compare할 수 있다. source가 실제로 선택한 mechanism만 설명하고 대체 설계를 구현 사실처럼 쓰지 않는다.

## 37.3 reference·lock·pin은 무엇을 금지하는가

### 37.3.1 reference는 active logical owner를 센다

R과 S가 B3을 공유하면 refcount 2다. R finish가 decrement하면 1이고 S가 계속 읽는다. S finish 뒤 0이 되어 cached eligible state로 갈 수 있다. decrement 두 번은 negative count 또는 premature free를 만들고 decrement 누락은 permanent pin처럼 보인다.

refcount는 request 수와 반드시 같지 않다. 한 request가 group·session·fork 구조에서 여러 logical owners를 만들 수 있고 special/null blocks는 일반 ref를 유지하지 않을 수 있다. source가 ref를 증가시키는 정확한 mutation과 감소시키는 cleanup을 쌍으로 찾는다.

### 37.3.2 radix lock은 node와 ancestor의 eviction을 막는다

SGLang radix tree는 matched node만 살아 있으면 되는 것이 아니다. compressed path와 ancestor 구조가 lookup mapping을 지탱한다. request가 node를 사용할 때 lock reference가 ancestor eviction eligibility와 evictable size accounting에 영향을 준다. unlock은 finish/abort 또는 prefix owner handoff에서 일어난다.

SWA radix variant는 `full_lock_ref`와 `swa_lock_ref`를 따로 두며 SWA lock이 있으면 full lock도 있어야 해 `full_lock_ref >= swa_lock_ref` 불변식을 둔다. [dual lock fields](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/swa_radix_cache.py#L60-L104)는 cache policy가 하나의 refcount로 충분하지 않은 사례다.

lock leak은 tree path가 eviction되지 않아 token pool shortage로 나타난다. physical allocation count만 보면 radix node Python object 문제처럼 보이지만 node value가 KV locations를 pin한다. 반대로 unlock이 빠르면 request attention 중 value locations가 eviction/reuse된다. node lock history와 token pool generation을 연결한다.

실제 radix [`inc_lock_ref()`와 `dec_lock_ref()`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/radix_cache.py#L622-L657)는 target에서 root 방향으로 path를 걸으며 lock transition가 evictable size에 미치는 delta를 반환한다. 단순히 target node counter 하나만 올렸다 내리는 구조가 아니다. fixture의 B3 prefix가 node N value라면 N과 ancestor path의 첫 0→1 transition이 candidate capacity를 줄인다.

R이 N을 lock한 뒤 S가 같은 N을 lock하면 counters는 더 늘지만 evictable→locked transition는 이미 R에서 일어났다. S finish decrement가 2→1일 때 evictable size가 회복되면 안 된다. R finish로 1→0이 될 때 path가 다시 eligible해진다. request count마다 value length를 더하고 빼면 metric가 실제 selector와 갈라진다.

`cache_finished_req()`와 `cache_unfinished_req()`는 request prefix node handoff와 lock release/acquisition가 만나는 경계다. [finished/unfinished cache mutation](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/radix_cache.py#L458-L567)에서 inserted/matched prefix, request token mapping, last node ref가 어떻게 바뀌는지 본다. T abort rollback는 이 handoff 중 어느 lock이 새로 생겼는지 알아야 한다.

### 37.3.3 pin과 in-flight fence는 refcount 0 뒤에도 남는다

remote transfer가 block을 source로 읽거나 GPU kernel이 write 중이면 active request ref가 0이어도 physical reuse를 지연해야 한다. 이를 refcount에 합칠 수도 별도 delayed owner로 둘 수도 있지만 release event가 명시되어야 한다. request map 삭제가 fence completion을 의미하지 않는다.

abort cleanup은 logical ref를 놓고 delayed list에 block generation과 last writer/transfer completion을 남긴다. completion 뒤 pool로 반환한다. pin을 ref로 표현했으면 crash 시 owner process가 사라졌을 때 이를 누가 정리하는지 recovery protocol이 필요하다.

## 37.4 eviction은 eligibility와 policy의 두 판정이다

### 37.4.1 vLLM free queue는 eviction order도 가진다

vLLM `BlockPool` free queue는 unowned blocks뿐 아니라 caching이 켜졌을 때 eviction candidates를 순서 있게 보관한다. 새 block을 가져올 때 cached metadata가 남은 candidate라면 `_maybe_evict_cached_block()`가 hash maps와 events를 정리한다. [eviction·touch·ordered free](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/block_pool.py#L641-L763)는 free와 cache eviction이 다른 mutation임을 보여 준다.

`touch()`는 ref 0 block을 free list에서 제거하고 ref를 증가시킨다. `free_blocks()`는 ordered iterable을 eviction priority에 맞춰 queue에 되돌린다. request blocks를 reverse order로 free해 tail이 먼저 eviction되도록 하는 manager 정책은 shared prefix 앞부분을 오래 남기는 효과가 있다. 이 효과를 모든 cache policy의 보편 법칙으로 확대하지 않는다.

pool transition을 B0 하나로 펼쳐 보자. 처음 B0은 ref 0, hash 있음, free queue member다. T의 prefix lookup가 B0을 찾으면 아직 candidate일 뿐이다. allocation commit의 touch가 queue에서 B0을 제거하고 ref를 1로 만든다. T table에 B0을 publish한 뒤 kernel reader가 생긴다. T finish가 ref를 0으로 만들면 hash가 유지된 채 queue로 돌아간다.

새 U가 physical block을 필요로 해 B0을 victim으로 선택하면 cached hash들을 제거하고 removal event를 만든다. 그 다음 B0은 U ref/owner를 얻는다. old bytes가 zero인지 여부는 correctness predicate가 아니다. old index가 제거되고 U가 write하기 전 누구도 B0을 cached prefix로 읽지 않아야 한다.

explicit eviction는 다른 경로다. invalid external KV load처럼 content identity를 더는 신뢰할 수 없으면 hash metadata를 제거할 수 있다. block ref가 양수면 active request가 물리 block을 계속 소유할 수 있어 pool free와 분리된다. 그 request는 invalid range를 recompute하거나 fail해야 한다. “eviction 완료” 로그로 allocator capacity가 늘었다고 계산하지 않는다.

free order가 tail-first인 이유는 request prefix 앞 blocks가 다른 prompt와 공유될 가능성이 높고 suffix는 request-specific일 가능성이 크다는 효과로 읽을 수 있다. 그러나 workload가 suffix repetition을 많이 가지거나 sliding group이면 실제 value가 다를 수 있다. source가 구현 ordering을 증명해도 최적 정책이라는 결론은 workload measurement가 필요하다.

free queue corruption를 찾을 때 block object ref와 linked-list membership을 함께 검사한다. ref 0인데 queue에 없으면 leak 후보이고 ref>0인데 queue에 있으면 live eviction 위험이다. special null block, delayed owner, external manager가 예외인지 분류한다. queue length 합계만 맞아도 같은 ID가 중복되고 다른 ID가 빠질 수 있다.

vLLM manager의 `free(request)`는 group coordinator에 request blocks를 넘기고 reverse order로 놓는다. request table에서 refs를 제거한 시각과 scheduler가 request map을 삭제하는 시각, async deferred-free fence가 물리 return을 허용하는 시각을 구분한다. block pool 함수만 읽으면 in-flight writer guard를 놓친다.

explicit `evict_blocks()`는 cached hash metadata를 제거하지만 refcount가 양수인 block을 물리 pool에서 free하지 않을 수 있다. “evicted” metric이 physical availability 증가와 같지 않을 수 있다. content index eviction, active owner, allocator queue를 각각 본다.

### 37.4.2 SGLang policy는 unlocked nodes에서 victim을 고른다

SGLang radix eviction은 lock ref가 0이고 policy list에 eligible한 leaf/node를 골라 value KV indices를 free하고 tree를 정리한다. locked node를 LRU가 오래됐다는 이유로 제거하지 않는다. root도 특별히 non-evictable하다. policy parser는 지원 문자열을 validation하고 strategy를 선택한다.

[`RadixCache.evict()`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/radix_cache.py#L592-L621)는 `EvictParams`가 요구한 token 수를 향해 policy-selected nodes를 처리한다. fixture T가 one block shortage라도 node value length가 더 크면 eviction result는 요구량을 넘을 수 있다. callback 또는 pool free와 node removal ordering을 읽고, returned count를 exact requested allocation와 동일시하지 않는다.

SWA dual path의 [`inc_lock_ref()`/`dec_lock_ref()`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/swa_radix_cache.py#L700-L783)는 full과 SWA lock kinds, UUID/path 조건에 따라 두 counters와 lists를 갱신한다. generic radix의 단일 delta 계산을 그대로 적용하지 않는다. B3이 full prefix owner인지 sliding-window owner인지가 inverse mutation를 바꾼다.

SWA/full dual lists에서는 같은 node가 두 visibility/lifetime 조건을 가질 수 있다. full tokens eviction와 sliding-window tokens eviction의 counts가 다를 수 있고 tombstone leaf cleanup이 뒤따른다. 한 `evicted_tokens` 수로 tree node removal과 두 pool 회수를 모두 단정하지 않는다.

radix node의 lock 변화는 evictable size bookkeeping까지 바꾼다. node value length가 16이고 lock이 0→1이면 policy가 희생할 수 있는 token 수에서 16을 빼야 한다. unlock 1→0이면 다시 더한다. ancestor가 잠긴 동안 child를 어떻게 세는지는 tree 구현 contract를 따른다. count만 고치고 LRU list membership을 고치지 않으면 selector와 metric이 갈라진다.

request R이 node N을 match하고 lock한 뒤 suffix allocation에 실패했다고 하자. R은 아직 runner table을 얻지 못했지만 N lock owner는 생겼다. admission rollback가 suffix locations만 free하고 N을 unlock하지 않으면 cache leak이다. 반대로 exception handler가 N과 ancestor를 두 번 unlock하면 다른 S가 쓰는 path가 eligible이 된다.

node split도 transaction surface다. compressed edge 안에서 new key가 divergence하면 old node value를 prefix/suffix nodes로 나누고 parent/children, locks, LRU membership을 재배치할 수 있다. split 중 allocation 실패가 가능한지, 실패하면 original topology를 유지하는지 본다. tree structure가 반쯤 바뀌면 lookup miss뿐 아니라 wrong KV indices가 생긴다.

eviction는 unlocked leaf부터 value KV indices를 pool에 돌리고 node를 제거하거나 tombstone을 정리한다. internal node를 먼저 지우면 live descendant path가 끊긴다. policy order와 tree structural eligibility를 분리한다. LRU tail이 internal/locked면 다음 eligible leaf를 찾는 native loop를 읽는다.

SWA variant의 `full_lock_ref >= swa_lock_ref`가 깨지는 사건을 생각하자. SWA unlock만 빠뜨리면 full ref 0, SWA ref 1 같은 불가능 state가 생길 수 있다. assertion가 일찍 잡으면 좋지만 production에서 disabled되면 eviction lists가 서로 다른 판단을 한다. dual counter mutation를 한 helper transaction으로 묶는지 확인한다.

정상 evictable size와 actual freed token count도 다를 수 있다. requested victim tokens보다 node value가 커 overshoot할 수 있고 full/SWA paths에서 한 eviction가 두 counts에 기여할 수 있다. allocation가 요구한 exact slots와 eviction result를 연결해 남은 shortage를 재평가한다.

### 37.4.3 Transformers initialized free는 content를 보존한다

Transformers `BlockManager`는 uninitialized free deque와 initialized cached ordered set을 나눈다. refcount가 0인 complete block은 initialized set으로 가 content/hash가 남는다. uninitialized blocks가 부족할 때 initialized victim을 pop하고 hash mapping을 제거해 재allocation한다. [state와 victim 전환](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache_manager.py#L58-L125)을 읽으면 free bytes를 zeroing과 동일시할 수 없다.

incomplete block은 ref 0에서 tracking을 제거하고 uninitialized deque로 간다. complete와 incomplete tail이 같은 free 경로를 타지 않는다. fork/COW 뒤 child rollback이 partial destination ref를 0으로 만들면 initialized cache가 아니라 uninitialized가 되어야 한다.

Transformers fixture에서 B0 complete initialized, B1 incomplete uninitialized라고 하자. 둘 다 `num_free_blocks`에는 포함되지만 B0 allocation에는 hash index removal이 필요하고 B1은 deque pop만 필요하다. allocation latency와 cache hit loss가 다르다. free count만으로 eviction work를 추정할 수 없다.

`has_enough_free_blocks(3)`는 uninitialized가 두 개면 initialized set에서 하나를 uninitialize한다. 이 mutation은 단순 dry-run이 아니다. hash-to-ID mapping을 제거하고 ID를 uninitialized deque로 옮긴다. 뒤 allocation transaction가 다른 이유로 실패하면 evicted content를 자동 복구하는지 source contract를 확인해야 한다.

함수 이름 `has_enough`만 보고 pure predicate로 호출하면 안 된다. option validation이나 speculative admission가 여러 번 호출할 때 cache eviction side effect가 반복될 수 있는지 caller를 본다. source walk에서는 return value뿐 아니라 내부 sets/deques mutation을 표시한다.

shareable allocation는 parent block ID와 group ID를 가진 `Block` 객체를 만들고 chain을 이어 간다. refcount가 1인 partial block은 in-use state다. complete 표시 전 rollback하면 object를 tracking map에서 제거하고 uninitialized queue로 돌린다. hash를 생성하지 않은 tail을 initialized cache로 넣지 않는다.

fork는 complete parents ref를 child 수만큼 먼저 올린 뒤 incomplete suffix destinations를 allocation한다. child 1 allocation 뒤 child 2가 실패하면 함수가 `None`을 반환하기 전에 이미 올린 refs와 destinations를 누가 rollback하는지 caller까지 추적한다. return tuple의 empty copy lists만 보고 side effect가 없었다고 단정하지 않는다.

Transformers [`fork_blocks()`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache_manager.py#L131-L193)는 complete prefix refs를 먼저 늘리고 remaining incomplete suffix마다 destination IDs와 copy pairs를 만든다. fixture에서 parent B3 complete, B4 partial이면 child는 B3 ref를 공유하고 B4 destination를 새로 받아야 한다. allocation failure가 난 정확한 child index까지 mutation ledger에 남긴다.

[`decrease_ref_count()`와 `free_blocks()`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache_manager.py#L194-L215)는 ref 0에서 complete block을 initialized set으로, incomplete를 uninitialized deque로 보낸다. T abort가 B3 shared ref와 new partial destination를 똑같이 free하면 state가 틀린다. inverse mutation는 content completeness predicate를 다시 사용한다.

request-level manager의 [`free_blocks(request_id, block_manager)`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache_manager.py#L300-L327)는 request block table owner에서 group manager로 release를 전달한다. request mapping 삭제, group별 free, async future state가 이미 FINISHED/PENDING인 guard를 한 timeline으로 읽는다. block manager helper 하나가 전체 abort cleanup를 소유한다고 생각하지 않는다.

free path에서 shareable false blocks는 ref object 없이 uninitialized queue에 extend될 수 있다. 같은 ID를 double free하면 deque 중복이 생겨 두 요청이 같은 block을 받을 수 있다. shareable true는 negative ref assertion나 missing object로 드러날 수 있지만 nonshareable은 더 조용할 수 있다. allocation generation uniqueness audit가 필요하다.

### 37.4.4 llama.cpp cell eligibility는 block cache eviction가 아니다

llama.cpp는 sequence-tagged KV cells와 ring search를 사용한다. `find_slot()`은 cell empty와 sequence position, contiguous requirement를 보고 ubatch가 들어갈 자리를 찾는다. [cell slot search](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L894-L1015)는 hash-indexed cached block victim을 고르는 vLLM/Transformers 정책과 다르다.

sequence association가 제거돼 cell이 empty가 되면 재사용 후보가 된다. fragmentation 때문에 충분한 empty cells 총량이 있어도 필요한 contiguous slot이 없을 수 있고 defrag/shift가 필요하다. refcount-0 cached block LRU와 같은 상태 기계로 번역하지 않는다.

llama.cpp [`seq_rm()`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L379-L435)는 sequence ID와 position range를 기준으로 각 stream의 cells association를 제거하고 실제로 cell이 비었는지에 따라 used accounting와 update flags를 바꾼다. R finish가 B3 같은 shared cell에서 R association만 제거했을 때 S association가 남으면 empty가 아니다.

position range가 일부라면 sequence 전체 cleanup와 다르다. sliding/shift나 context edit는 특정 range만 제거할 수 있다. broad abort handler가 `(-1,-1)` full removal semantics를 부분 eviction에 적용하거나 반대로 partial range를 full cleanup로 착각하면 cells가 남거나 live context가 사라진다. caller의 p0/p1 contract를 확인한다.

`seq_rm()` 뒤 `find_slot()`가 새 ubatch cells를 찾는 순서에서 old graph가 removed cells를 읽지 않는 synchronization가 필요하다. cell metadata상 empty가 된 시각과 physical writer completion를 구분한다. llama.cpp에 paged deferred-free 필드가 있다고 발명하지 않고 context scheduling/graph completion 경계를 실제 caller에서 확인한다.

R이 cells 4–7을 쓰고 S가 cell 6을 sequence-shared한다고 하자. R removal 뒤 cells 4,5,7은 empty가 될 수 있지만 6은 S association가 남는다. four-cell contiguous allocation는 4–7을 쓸 수 없다. total empty count가 충분하다는 계산과 `find_slot(cont=true)` 성공이 다르다.

ring head는 search 시작 hint이고 live ownership은 cells metadata가 결정한다. head 근처에서 slot을 못 찾으면 wrap하며 후보를 검사한다. stale head 때문에 search가 느린 것과 cell empty marking 누락으로 영원히 못 찾는 것은 다르다. debug에는 head, used count, longest empty run, sequence associations를 둔다.

defrag는 scattered live cells를 옮겨 contiguous free run을 만든다. 이동 graph가 완료되기 전에 old/new cells를 allocation하면 두 owner가 겹친다. defrag pending state와 allocator admission를 serialize하거나 event fence를 둔다. paged pool eviction와 달리 content copy/position metadata update가 핵심 비용이다.

sequence copy/fork는 cell metadata에 multiple sequence IDs를 추가할 수 있다. remove는 target sequence association만 지우고 count가 0일 때 empty로 만든다. 모든 cells를 request-private라고 가정해 R finish 때 clear하면 S context가 사라진다. block refcount와 이름은 달라도 shared writer/read ownership 질문은 같다.

llama.cpp crash cleanup은 process context teardown로 cache 전체를 폐기할 수 있지만 graceful sequence removal은 cell-level semantics를 지킨다. process crash 복구를 normal `seq_rm` 반복으로 흉내 내지 않는다. owner ledger 신뢰 여부가 다르다.

## 37.5 partial allocation 실패를 역순으로 되감는다

### 37.5.1 T가 두 blocks 뒤 실패한 사건

T가 B6, B7을 claim하고 세 번째 B1 victim metadata를 evict하려는 순간 connector reservation이 실패했다고 하자. 아직 publish 전이라면 T table에 IDs가 없어야 한다. B6/B7 owner를 제거해 원래 uninitialized free queue로 돌린다. B1 eviction가 commit 전이면 cached identity를 유지한다.

B1 hash metadata를 이미 제거했다면 rollback이 이를 복원할 수 있는지가 구현 contract에 달렸다. content와 hash가 안전하고 transaction log가 있다면 복원할 수 있다. native manager가 eviction을 irreversible로 정의하면 B1은 uninitialized free로 남기고 이전 cache hit를 잃을 수 있다. correctness는 유지하되 cache 효율이 줄어든다. 존재하지 않는 rollback을 발명하지 않는다.

### 37.5.2 existing touch와 new allocation을 따로 되돌린다

T가 cached prefix B0을 touch해 ref 1로 만들고 suffix B6/B7을 allocation했다가 실패했다고 하자. rollback은 B0 ref를 0으로 내려 eviction queue 위치를 native policy에 맞게 복구하고 B6/B7을 free한다. B0 content를 uninitialized로 만들면 다른 cached request의 기회를 잃고 hash map이 어긋난다.

여러 group에서 touch한 순서를 기록한다. group 0 B0, group 1 C0을 touch한 뒤 group 2 실패라면 두 refs를 모두 놓는다. 중간 exception handler가 group 0만 알고 있으면 group 1 pin leak이다. coordinator-level transaction가 group manager results를 모으는 이유다.

### 37.5.3 publish 뒤 실패는 rollback이 아니라 cleanup일 수 있다

runner metadata가 T blocks를 받아 kernel을 submit했다면 physical writes가 시작될 수 있다. 이때 request table을 old state로 되돌리고 blocks를 즉시 free하면 writer use-after-free가 된다. T를 terminal error로 만들고 in-flight completion fence 뒤 resource를 release한다. 외부에서는 allocation 실패처럼 보여도 내부 경계는 실행 cleanup이다.

output stream과 scheduler owner도 닫아야 한다. blocks만 회수하고 waiting/running container에 T가 남으면 retry가 stale table을 재사용한다. 반대로 request를 map에서 먼저 지우면 late completion owner를 잃는다. 28·31장의 수명을 여기서는 allocator transaction에 적용한다.

### 37.5.4 rollback ledger는 mutation 종류와 역연산을 가진다

T transaction ledger를 구체적으로 적는다. `touch_existing=[B0]`, `new_uninitialized=[B6,B7]`, `evicted_cached=[B1]`, `new_table_entries=[]`, `submitted=[]`라고 하자. failure 시 B0 touch를 release하고 B6/B7을 free한다. B1 cached identity 복원이 native contract에 없으면 uninitialized eligible로 남긴다. table과 submit은 비었으므로 runner fence는 없다.

다른 시점에는 `new_table_entries=[B6,B7,B1]`, `submitted=[step52]`까지 진행됐을 수 있다. 이때 entries를 old table로 되돌리는 것은 future schedule을 막는 데 필요하지만 resources는 step 52 completion 전 free하지 않는다. ledger가 `delayed=[B6,B7,B1; fence=52]` owner로 넘긴다. rollback이라는 이름 하나에 즉시 free와 delayed cleanup을 섞지 않는다.

역순 원칙에도 예외처럼 보이는 dependency가 있다. hash metadata restore가 physical block을 다시 cached owner로 만들려면 block이 아직 다른 request에 publish되지 않았어야 한다. table unpublish와 victim metadata restore ordering을 native lock/serialization 아래 수행한다. 단순 list reverse가 transaction correctness를 자동으로 보장하지 않는다.

ledger 자체가 allocation hot path overhead를 만들 수 있다. 구현은 local lists, context manager, coordinator return object처럼 더 가벼운 표현을 쓸 수 있다. 중요한 것은 failure handler가 “지금까지 무엇을 얻었는가”를 재구성할 수 있다는 점이다. 상태를 global maps 전수 scan으로 추정하면 concurrent owner를 잘못 free할 수 있다.

OOM exception과 normal `None` failure도 같은 rollback 의무를 가진다. 예상 가능한 capacity shortage는 return path에서 처리하고 unexpected exception은 finally/RAII가 처리할 수 있다. normal branch만 cleanup하고 exception branch를 빼먹으면 드문 fault injection에서 leak가 나타난다. 반대로 broad finally가 commit 뒤 resources까지 free하지 않게 commit marker가 필요하다.

### 37.5.5 retry는 이전 transaction generation을 재사용하지 않는다

T가 allocation 실패로 waiting에 돌아갔다가 다음 iteration retry한다고 하자. request identity는 같아도 allocation transaction generation은 새로 만든다. 이전 ledger의 B6/B7 references나 victim selection를 그대로 신뢰하지 않는다. 그 사이 다른 request가 blocks를 얻거나 B0이 touch될 수 있다.

retry가 old reserved IDs를 들고 있으면 ownership lease가 명시되어야 한다. reservation이 유지되는 동안 pool capacity에서 빠지고 abort가 release해야 한다. reservation을 release했으면 IDs는 hints도 아니다. integer block ID만 request object에 남아 새 generation content를 가리키는 stale reservation을 막는다.

backoff/fairness는 30장의 scheduler 주제이므로 여기서 반복하지 않는다. allocator 관점에서는 retry마다 owner partition 합과 leaked mutation가 없어야 한다. 열 번 실패해도 free/cached counts가 첫 시도 전과 같거나 policy상 irreversible evictions만큼만 달라야 한다.

## 37.6 abort와 crash cleanup은 같은 경로가 아니다

### 37.6.1 graceful abort는 owner ledger를 읽을 수 있다

abort는 request object, block table, refs, locks, in-flight steps가 살아 있어 native cleanup을 실행할 수 있다. waiting T는 reserved resources만 rollback하고, running T는 future membership을 막고 writer completion 뒤 free한다. prefix lock과 private tail ref를 구분한다.

double abort는 effect를 반복하지 않아야 한다. ref decrement가 두 번이면 live shared block을 victim으로 만든다. map에 request가 없다는 guard만으로 delayed owner cleanup가 완료됐는지 증명하지 못한다. terminal marker, released owner kinds, delayed generation을 본다.

abort가 allocation transaction 중 들어올 수도 있다. reserve 뒤 claim 전이면 transaction cancel flag를 보고 candidates를 놓는다. claim 뒤 publish 전이면 ledger rollback한다. publish 뒤 submit 전이면 table unpublish와 resources release가 가능하되 runner queue에 metadata가 전달되지 않았음을 증명한다. submit 뒤에는 fenced cleanup이다.

두 task가 abort와 allocation completion을 각각 처리한다면 commit 권한을 serialize해야 한다. allocation가 publish한 직후 abort가 old “not published” snapshot으로 immediate free하면 runner가 freed block을 받을 수 있다. request generation state machine과 allocator transaction marker를 atomic boundary에서 바꾼다.

natural finish와 abort 경쟁도 ref effect를 한 번만 수행한다. finish가 prefix lock과 private refs를 release했다면 abort는 collector close만 할 수 있다. delayed writer entry는 completion handler가 한 번 drain한다. terminal reason과 resource cleanup owner를 같은 bool 하나에 의존하지 않는다.

### 37.6.2 worker crash는 completion을 증명할 수 없다

process가 죽으면 host owner ledger 일부와 device context가 함께 사라질 수 있다. 같은 process pool을 계속 쓸 수 없으므로 일반 abort처럼 blocks를 global pool에 반환하는 문제가 아닐 수 있다. coordinator는 worker의 outstanding requests를 실패시키고 upstream refs/remote pins를 정리하며 해당 device allocator를 재초기화한다.

remote KV connector나 shared store가 있으면 crash한 producer가 잡던 pin lease를 누가 만료시키는지 필요하다. 영구 pin은 capacity leak이고 너무 짧은 lease는 느린 live transfer를 eviction한다. local refcount만 0으로 만든다고 distributed owner가 사라지지 않는다.

coordinator crash와 worker crash도 다르다. worker가 죽고 coordinator가 살아 있으면 outstanding request→worker mapping으로 caller를 실패시키고 worker-local pool을 폐기할 수 있다. coordinator가 죽으면 worker가 orphan work를 계속하는지 supervisor가 context를 종료하는지 deployment contract가 필요하다. local source 함수만으로 distributed recovery를 단정하지 않는다.

shared store lease에는 epoch가 필요할 수 있다. old coordinator의 delayed release가 새 coordinator generation의 pin을 감소시키면 live transfer가 eviction된다. lease owner identity와 process/session epoch를 결합한다. timeout cleanup은 동일 epoch owner에게만 적용한다.

crash 뒤 metrics가 refcount 0으로 초기화됐다는 사실은 clean recovery 증거가 아니다. allocator process가 새로 만들어져 old GPU context가 사라졌는지, remote registrations와 CPU tiers가 해제됐는지 본다. 단순 counter reset은 외부 pin leak를 숨긴다.

### 37.6.3 shutdown drain은 새 allocation를 먼저 막는다

graceful shutdown은 admission과 victim selection를 막고, launch 전 reservations를 rollback하고, in-flight writers를 drain한 뒤 active refs와 locks를 해제한다. allocator부터 파괴하면 late cleanup가 stale object를 건드리고 waiter가 hang할 수 있다.

drain timeout이면 불확실한 generation을 재사용하지 않고 process/device allocator teardown에 맡긴다. availability를 위해 free count만 복원하면 silent KV corruption 위험이 있다. crash recovery와 normal pool reuse를 분리한다.

shutdown state를 `accepting`, `draining`, `closed`로 나누면 allocation guard가 선명해진다. draining 진입 뒤 new reserve는 실패해야 하지만 existing submitted work completion와 cleanup allocation metadata access는 허용돼야 한다. allocator object를 closed로 너무 빨리 바꾸면 drain handler가 ref를 놓지 못한다.

grace period 동안 cached eligible content를 굳이 eviction할 필요는 없다. process teardown가 모두 회수할 예정이기 때문이다. active/delayed owner의 안전한 terminal 처리와 caller notification가 우선이다. eviction metric를 높여 free count를 만드는 작업은 shutdown completion을 증명하지 않는다.

## 37.7 네 사고를 first divergence에서 닫는다

### 37.7.1 leaked pin

증상은 active request가 줄어도 evictable capacity가 회복되지 않고 allocation failure가 늘어나는 것이다. aggregate refcount보다 owner-kind ledger를 본다. request ref는 0인데 radix lock, transfer pin, delayed writer owner가 남는지 확인한다.

first divergence는 finish/abort에서 matching decrement를 빼먹은 줄, allocation rollback에서 touched block을 놓지 않은 줄이다. 정상 cached eligible blocks가 content를 보존하는 상태를 leak으로 오인하지 않는다. eviction 압력에서 실제 후보가 되는지 반증한다.

수치로 보면 active requests 0인데 ref sum 5인 것만으로 leak이라고 말할 수 없다. cached eligible은 ref 0이고 connector pins나 delayed writers가 5일 수 있다. 각 owner의 deadline/completion condition을 확인한다. condition이 이미 충족됐는데 drain되지 않은 것만 leak다.

SGLang에서는 root lock이나 permanent structural refs가 baseline으로 존재할 수 있다. absolute lock sum보다 request 전후 delta와 evictable size 일치를 본다. vLLM null block도 일반 ref partition에서 제외한다. 구현 special owner를 leak detector가 예외 처리하되 arbitrary whitelist로 실제 leak를 숨기지 않는다.

### 37.7.2 double decrement와 live eviction

증상은 refcount negative assertion, free queue 중복 ID, 다른 request의 KV corruption이다. R/S shared B3에서 R cleanup 뒤 ref 1이어야 하는데 0이면 R 경로가 두 번 decrement했거나 S increment가 commit되지 않았다. ref before/after와 owner identity를 기록한다.

policy가 ref 0 B3을 victim으로 고른 것은 downstream 결과다. first divergence는 owner accounting다. victim selector에 request ID blacklist를 넣어 가리면 다른 shared block에서 재발한다. ref transaction를 고친다.

double decrement의 첫 증상이 allocation success일 수도 있다. premature candidate B3 덕분에 T가 기다리지 않고 admission되기 때문이다. latency는 좋아졌지만 S output이 깨진다. free capacity 증가를 항상 health로 해석하지 않는 이유다.

반증은 legitimate owner release다. S가 이미 finish했고 output corruption가 network buffering일 수 있다. S의 last reader completion과 B3 victim selection를 causal sequence로 맞춘다. 서로 다른 시간 snapshot을 합쳐 live eviction이라고 부르지 않는다.

### 37.7.3 partial allocation orphan

증상은 T가 waiting/failed인데 B6,B7이 pool에도 request table에도 없고 free count가 서서히 감소하는 것이다. allocation transaction ID로 reserved, claimed, published sets를 비교한다. first divergence는 third-block failure branch가 already claimed list를 잃은 지점이다.

복구는 runner submit 여부를 확인한다. submit 전 orphan은 owner를 검증해 free할 수 있다. submit 가능성이 있으면 completion을 모르는 resource로 격리한다. free count 회복만 목표로 즉시 pool에 넣지 않는다.

orphan detector는 `pool total - known partitions` 차이를 찾는다. partitions에는 active refs, cached eligible, uninitialized, reserved, delayed, special blocks가 들어간다. count가 맞아도 duplicate ID 때문에 다른 orphan이 숨을 수 있으므로 ID set union과 disjointness도 확인한다.

자동 복구가 orphan ID를 free하려면 generation과 no-writer를 증명해야 한다. 그렇지 못하면 process restart가 더 안전하다. 오래된 table이 해당 ID를 가리키는지 runner submissions를 검색한다. 단순히 request map에 없다는 것은 충분하지 않다.

### 37.7.4 stale victim metadata

증상은 cache hit가 evicted/reused block ID를 반환하거나 eviction metric은 증가했는데 hash/radix index가 계속 old location을 가리키는 것이다. victim selection, index removal, physical generation change, lookup touch 순서를 본다.

first divergence는 metadata detach와 pool claim 사이 transaction이 끊긴 지점이다. cache를 전부 끄면 증상은 사라질 수 있지만 allocator owner bug를 고친 것은 아니다. generation mismatch를 hit에서 거부하고 index cleanup 원인을 수정한다.

stale victim metadata는 반대 방향도 있다. physical block은 cached eligible인데 hash/radix index가 너무 일찍 제거되어 hit가 사라질 수 있다. correctness는 유지되지만 miss와 recompute가 증가한다. wrong-output 사건과 performance regression을 같은 severity로 다루지 않되 transaction invariant는 고친다.

index가 block ID와 generation을 함께 검증하면 stale hit를 reject할 수 있다. 하지만 reject counter가 늘어나는 것은 upstream removal race가 남았다는 신호다. guard는 silent corruption을 막는 방어선이지 root fix가 아니다.

## 37.8 수치 fixture로 eligibility와 rollback을 검산한다

초기 state에서 uninitialized free=2(B6,B7), cached eligible=3(B0–B2), active unique blocks=3(B3–B5)다. physical 합은 8이다. logical refs는 B3이 두 개라 4지만 physical count와 합산하지 않는다. T가 세 blocks를 성공하면 uninitialized/cached 후보에서 세 개가 active로 이동한다.

policy가 B0을 victim으로 골랐다면 cache hits potential은 하나 줄고 T physical owner는 B6,B7,B0이 된다. physical 합은 여전히 8이다. allocation 뒤 free count 0만 보고 leak이라고 하지 않는다. owner partition 합이 8인지 본다.

두 blocks 뒤 실패하고 완전 rollback되면 초기 partition으로 돌아와야 한다. irreversible cache eviction가 이미 commit됐다면 B0은 cached eligible이 아니라 uninitialized free가 될 수 있지만 physical 합과 active owner는 같아야 한다. cache metadata 효율 손실과 memory leak을 구분한다.

R finish 뒤 B4 partial은 uninitialized free, B3 shared ref는 1이다. S finish 뒤 B3 complete면 cached eligible로 갈 수 있다. 이 두 finish에서 physical free 증가는 같아 보여도 content identity state가 다르다. 다음 allocation가 어느 것을 먼저 희생시키는지 policy가 결정한다.

evictable count는 cached eligible tokens/blocks와 uninitialized free를 구분해 표시한다. watermark reservation, delayed pins, locked radix nodes를 별도로 뺀다. `total = active + delayed/pinned + cached-eligible + uninitialized + reserved` partition가 맞는지 표본 audit한다. 구현 special/null resources는 별 항으로 둔다.

### 37.8.1 성공 transaction을 숫자와 세대로 적는다

초기 generation을 붙이자. B6은 physical ID 6 generation 4, B7은 ID 7 generation 2로 uninitialized free다. cached B0은 generation 9이며 hash H0을 가진다. T allocation transaction 21은 B6/B7을 claim하고 B0 cached metadata를 evict한 뒤 B0 generation을 10으로 증가시킨다. T table에는 `(6,g4),(7,g2),(0,g10)`이 들어간다.

왜 uninitialized B6/B7 generation도 유지하거나 증가시키는지는 구현에 달렸다. 중요한 것은 과거 table이 같은 integer ID를 current content로 오인하지 않게 allocation epoch를 논리적으로 구분하는 것이다. 실제 field가 없다면 scheduler/table generation과 pool owner history로 추론할 수 있다. source에 없는 field를 있다고 쓰지 않는다.

partition before는 `active={3,4,5}`, `cached={0,1,2}`, `uninit={6,7}`, `reserved={}, delayed={}`다. reserve 뒤에는 `reserved={0,6,7}`이고 cached/uninit sets에서 빠진다. publish 뒤 reserved가 active T owner로 이동한다. 어느 snapshot에서도 ID set union은 0–7이고 intersections는 허용된 shared logical refs를 제외하면 없다.

shared logical refs는 physical partition를 중복시키지 않는다. B3 ref 2라도 active physical set에는 한 번만 센다. ref ledger에는 `(R,B3),(S,B3)` 두 edges가 있다. physical partition와 logical edge count를 섞으면 total이 9가 되어 false alarm이 난다.

T finish 뒤 blocks가 complete인지에 따라 상태가 갈린다. B6/B7/B0이 full hash content로 commit됐다면 cached eligible로 갈 수 있다. partial tails는 uninitialized가 된다. same three physical IDs가 free count 3을 만들더라도 future hit 가능성과 eviction cost가 다르다.

### 37.8.2 fixture를 allocation에서 abort까지 한 바퀴 돌린다

초기 R/S/T 장면을 한 timeline으로 합치자. `t0`에서 R과 S는 B3 shared prefix를 읽고 R은 B4, S는 B5 private tails를 쓴다. B0–B2는 cached eligible, B6–B7은 uninitialized다. T는 아직 waiting이며 owner edge가 없다. 이때 physical partition는 서로 겹치지 않고 B3에만 logical refs 두 개가 있다.

`t1`에서 T allocation transaction 21이 시작된다. allocator는 B6과 B7을 reserve하고, cached B0을 victim candidate로 고른다. reserve 상태는 다른 allocator가 이 IDs를 claim하지 못하게 하지만 runner는 아직 읽을 수 없다. T request table에도 없다. reservation gauge는 3, active T refs는 0이다.

`t2`에서 B0 content metadata eviction이 commit된다. hash index와 reverse map에서 B0 generation 9 identity가 사라진다. B0 bytes는 남아도 cached hit로 반환하면 안 된다. T claim이 B0 generation 10과 B6/B7 owner edges를 만든다. ref/pin/lock 가운데 여기서 생기는 것은 physical allocation ref다. radix lock이나 connector pin은 별 사건이다.

`t3`에서 T table `[B6,B7,B0]`가 scheduler output generation 71에 publish된다. publish는 reservation를 executable ownership으로 바꾼다. runner row 4가 table을 받고 compute stream에 step을 submit한다. 이후 allocation transaction를 “실패”로 되돌릴 수 없다. T를 abort하더라도 table generation과 writer completion을 처리하는 cleanup으로 넘어간다.

`t4`에서 T가 B6의 content를 remote store로 보내는 connector pin을 얻는다고 하자. 이제 B6에는 T request ref와 transfer pin 두 owner kinds가 있다. 같은 숫자 counter로 합치면 request finish 뒤 남은 1이 무엇인지 알 수 없다. ledger는 `(request,T)`와 `(transfer,X)`를 분리한다. B7/B0에는 request ref만 있다.

`t5`에서 S가 끝난다. B3 ref는 2→1이 아니라 R과 S 중 S만 놓아 1이 된다. B5 partial tail은 ref 1→0으로 uninitialized free가 된다. eviction policy는 B3을 candidate로 볼 수 없고 B5에는 evict할 cached metadata가 없다. aggregate free capacity가 하나 늘어난다.

`t6`에서 T client가 abort한다. scheduler는 future submissions를 막고 table owner를 terminal로 바꾼다. T request refs B6/B7/B0을 release할 준비를 하지만 latest step writer가 남았다. B7/B0은 delayed generation으로, B6은 delayed writer와 transfer pin 양쪽 조건을 가진다. output collector는 닫혀도 resource reuse는 아직 금지다.

`t7`에서 GPU completion이 도착한다. B7/B0 writer fence가 풀려 incomplete라면 uninitialized queue로 간다. B6은 GPU fence가 풀렸어도 transfer X가 읽는 중이라 pin owner가 남는다. free count가 두 개만 회복되는 것은 leak가 아니다. B6 release condition가 아직 false다.

`t8`에서 transfer completion가 B6 pin을 놓는다. 이제 active ref 0, writer 0, transfer pin 0이므로 physical reuse eligibility가 생긴다. content가 complete하고 native cache commit가 있었다면 cached eligible이 될 수 있고 abort policy상 invalid/incomplete라면 uninitialized가 된다. 어느 state인지는 data validity contract가 결정한다.

`t9`에서 R도 끝나 B3 ref가 0이 되고 complete shared prefix가 cached eligible로 들어간다. B4 partial은 uninitialized가 된다. 초기 cached B0 content는 T allocation 때 사라졌지만 B3라는 새 cached candidate가 생겼다. final partition가 initial과 똑같을 필요는 없다. 모든 IDs에 한 owner state가 있고 live refs가 0이며 delayed conditions가 해소됐는지가 완료 기준이다.

이 timeline은 refcount, pin, lock을 한 counter로 합치면 왜 안 되는지 보여 준다. radix cache를 쓴다면 R/S의 B3 content path에는 node lock도 있어 physical ref와 함께 release돼야 한다. node lock은 tree eviction를 막고 transfer pin은 bytes reuse를 막으며 request ref는 active logical reader를 나타낸다. release 함수와 condition가 다르다.

### 37.8.3 같은 timeline의 실패 위치를 왕복한다

`t1` reserve 중 세 번째 candidate가 없으면 B6/B7 reservations를 놓고 T는 waiting으로 돌아간다. B0 metadata는 아직 건드리지 않았으므로 initial partition가 완전히 복원된다. `t2` B0 eviction 뒤 connector validation이 실패하면 B6/B7을 놓고 B0을 uninitialized로 남길 수 있다. correctness는 복원되지만 cached opportunity는 복원되지 않는다.

`t3` publish 중 runner metadata serialization가 실패하면 kernel submit 여부가 분기점이다. queue에 generation 71이 들어가지 않았음을 증명하면 table unpublish와 immediate refs release가 가능하다. executor가 메시지를 받았는지 불명확하면 delayed owner로 격리한다. “serialization exception”이라는 같은 메시지도 handoff commit 전후가 다르다.

`t4` transfer pin 획득 뒤 request abort가 오면 transfer cancellation가 실제 completion을 보장하는지 본다. cancel API return가 remote DMA 중단과 같지 않을 수 있다. pin lease를 즉시 0으로 만들지 않고 callback/timeout epoch를 기다린다. worker crash라면 local callback가 오지 않아 supervisor lease recovery가 필요하다.

`t6` abort handler가 두 번 실행되면 request refs를 두 번 놓지 않게 effect ownership을 확인한다. 첫 handler가 delayed entries를 등록했고 두 번째가 request map 부재로 return해도 괜찮으려면 completion handler가 entries를 소유해야 한다. 첫 handler가 registration 전에 map을 지웠다면 두 번째 guard가 leak를 고정한다.

`t7` completion가 두 번 전달돼도 pool return effect는 한 번이어야 한다. delayed entry를 atomic remove/claim한 handler만 release한다. token output dedup과 block release dedup을 같은 flag로 묶지 않는다. output은 protocol policy로 drop될 수 있어도 resource completion은 반드시 처리된다.

이 왕복에서 inverse mutation가 대칭적이지 않은 지점은 B0 cached metadata eviction와 device submit이다. cached content를 잃는 것은 성능 state 손실로 허용될 수 있고 submit된 work는 completion 전 되돌릴 수 없다. rollback 설계는 모든 mutation을 문자 그대로 원상 복구하는 환상이 아니라 correctness owner partition를 회복하는 일이다.

### 37.8.4 세 번째 block 실패를 네 commit 위치에서 계산한다

위치 A는 capacity check 전이다. partition 변화가 0이어야 한다. 위치 B는 B6/B7 reserve 뒤다. rollback 후 reserved는 0, uninit에 6,7이 돌아간다. 위치 C는 B0 hash eviction 뒤다. rollback contract가 irreversible이면 uninit에 0까지 들어가고 cached는 1,2만 남는다. physical capacity는 보존되지만 one cached entry를 잃는다.

위치 D는 T table publish 뒤 runner submit 전이다. table generation을 invalid하고 row에서 T entries를 제거한 다음 blocks를 free한다. executor queue에 metadata가 없음을 증명해야 한다. 위치 E는 submit 뒤다. active request owner를 terminal-delayed로 바꾸고 completion까지 blocks를 reserved/delayed partition에 둔다. 즉시 initial partition로 돌아오지 않는 것이 정상이다.

각 위치의 expected metric도 다르다. A/B는 eviction count 변화가 없어야 한다. C는 eviction 1과 allocation failure 1이 함께 보일 수 있다. D는 table install/uninstall trace가 있고 device launch 0이다. E는 terminal request, in-flight 1, delayed blocks 3이며 completion 뒤 free가 증가한다. incident 재현 없이도 source branch에서 기대값을 정의할 수 있다.

위치 E에서 client가 retry해 T2를 만들면 old T delayed blocks를 새 T2에 넘기지 않는다. request payload가 같아도 transaction generation이 다르다. cache hit로 old completed blocks를 나중 획득할 수는 있지만 allocator eligibility를 다시 통과한다.

### 37.8.5 policy 선택의 비용을 fixture에 얹는다

B0–B2 cached candidates의 last-touch order가 B2 oldest, B0, B1 newest라고 하자. LRU면 B2를 victim으로 고른다. 그러나 B2가 long shared prefix의 first block이고 B0가 short request tail이면 future reuse value가 다를 수 있다. policy가 recency만 쓰면 이 정보는 고려하지 않는다. 코드가 구현하지 않은 optimality를 주장하지 않는다.

tail-first free ordering은 같은 request blocks가 queue로 들어갈 때 suffix가 먼저 victim이 되게 할 수 있다. candidate global order는 다른 requests touch와 섞인다. “tail always evicted first”가 아니라 free insertion가 주는 local priority 효과로 표현한다.

SGLang radix policy는 node value 길이가 달라 victim 하나가 요구 slots를 넘겨 free할 수 있다. request T가 한 page만 부족해도 compressed node 16 tokens가 evict될 수 있다. freed amount, chosen nodes, future hit loss를 함께 본다. exact-fit allocator처럼 해석하지 않는다.

Transformers initialized ordered set의 pop direction이 effective recency/order policy를 만든다. insertion/touch가 order를 어떻게 갱신하는지 확인한다. dict가 ordered라는 사실만으로 LRU라고 부르지 않는다. access 시 move가 없다면 insertion/finish order에 가깝다.

llama.cpp는 total free cells보다 contiguous run이 victim/defrag decision에 중요할 수 있다. empty six cells가 scattered되어 T four-token ubatch가 못 들어가면 cache count eviction 문제가 아니다. longest run과 slot search를 본다.

## 37.9 ABA-37: 반환된 B41을 과거 완료가 다시 반환한 밤

### 37.9.1 문제 장면: ID는 같았고 세대만 달랐다

이제 이 장의 조건을 실제 장애 하나로 압축한다. pool에는 block ID 0–63이 있고 B41은 요청 A의 마지막 KV block이다. allocator 기록은 `(block_id=41, generation=17, owner=A, refcount=1)`이다. A의 decode step 883은 stream 3에 KV write를 enqueue했고, host는 그 뒤 `cudaEventRecord(E17, stream3)`에 해당하는 완료 표지를 남겼다. 이 시점의 핵심은 함수 enqueue가 성공했다는 사실과 device write가 끝났다는 사실이 다르다는 데 있다.

`t=0 µs`에 A@17이 B41을 가진다. `t=8 µs`에 step 883의 write가 제출된다. `t=11 µs`에 preemption 판단이 A를 running에서 retracted로 옮긴다. 거의 동시에 `t=12 µs`에 client disconnect가 abort를 일으킨다. preemption 경로는 scheduler reference를 놓고, abort 경로는 request table을 제거한다. 두 경로가 같은 logical owner token을 공유하지 않으면 둘 다 `1→0` 감소를 수행하거나 하나는 이미 0인 값을 다시 감소시킨다.

문제가 난 구현은 refcount가 0이 된 순간 B41을 free list tail에 넣었다. E17은 아직 incomplete였지만 event owner는 block record가 아니라 abort된 request object에만 매달려 있었다. request object를 map에서 지우자 allocator는 writer가 남았다는 사실을 볼 수 없었다. `t=19 µs`에 eviction 압력이 cached B41의 old identity를 제거했고, `t=22 µs`에 요청 B가 같은 정수 ID를 꺼내 generation 18을 받았다. 기록은 이제 `(41,18,B,1)`이다.

`t=27 µs`에 B의 block table이 runner에 publish된다. `t=31 µs`에 과거 E17이 완료된다. 늦은 callback은 캡처해 둔 `block_id=41`만 읽고 `free(41)`을 호출했다. 현재 record의 generation을 확인하지 않았기 때문에 B@18의 ref를 감소시켜 free list에 다시 넣었다. 이것이 ABA다. 관측한 정수 상태는 `41 allocated → 41 free → 41 allocated`이고 마지막 ID가 처음과 같아 “아무것도 변하지 않았다”처럼 보이지만, A@17과 B@18은 다른 resource다.

더 나쁜 변형에서는 E17의 device write가 B가 초기화한 KV 뒤에 도착한다. 이 경우 allocator metadata는 정상이어도 bytes가 섞인다. B의 첫 몇 token은 정상이고 특정 head·offset부터 logits가 흔들린다. OOM도 exception도 없어서 cache hit나 sampling noise로 오인하기 쉽다. 그래서 ABA 검사는 refcount underflow만 찾으면 부족하다. `old writer completion < new generation first read/write`라는 시간 불변식도 함께 증명해야 한다.

이 사고에서 abort, preemption, eviction은 각각 합법적인 기능이다. 잘못은 셋이 겹쳤다는 사실이 아니라 effect owner가 불분명했다는 데 있다. preemption은 실행 자격을 회수하고, abort는 request 수명을 끝내며, eviction은 cached identity를 희생시킨다. 어느 경로도 미완료 device writer를 지울 권리는 없다. 세 경로가 공통 release ledger를 통해 “누가 logical ref를 이미 놓았는가”와 “누가 physical readiness를 기다리는가”를 봐야 한다.

### 37.9.2 직관: 열쇠 번호가 아니라 임대 계약을 비교한다

B41이라는 숫자는 호텔 객실 번호와 같다. A가 41호를 체크아웃했다고 프런트 시스템에 표시돼도 청소 작업자가 아직 안에 있을 수 있다. B에게 같은 방을 배정한 뒤 A의 늦은 체크아웃 메시지가 오면, 객실 번호만 본 시스템은 B의 계약을 취소한다. 해결은 방 번호를 더 크게 만드는 것이 아니라 `(41, 계약 17)`과 `(41, 계약 18)`을 서로 다른 capability로 취급하는 것이다.

이 비유에서도 청소 완료는 CUDA event completion과 완전히 같지 않다. event는 특정 stream에서 그 앞에 제출된 작업이 완료됐다는 ordering 증거이지 block의 모든 사용자를 자동으로 발견하는 garbage collector가 아니다. 다른 stream reader, NCCL transfer, host copy가 있다면 각각의 dependency가 record에 포함돼야 한다. event 하나를 기다렸다는 이유로 알려지지 않은 consumer까지 끝났다고 단정하지 않는다.

generation은 wrap되지 않는 이상 단조 증가하는 local identity다. free list node에는 block ID만 넣지 않고 free 당시 generation 또는 pool이 다음 claim에서 부여할 generation을 연결한다. delayed completion record는 `(block_id, generation, event_id, effect_kind)`를 가진다. callback은 current generation이 자신과 같을 때만 current record를 바꿀 수 있다. 다르면 “이미 새 계약”이므로 stale completion metric만 올리고 old delayed record를 닫는다.

generation check만 추가해도 old write가 새 bytes를 덮는 문제까지 자동 해결되지는 않는다. check는 늦은 host callback이 B@18을 free하지 못하게 하지만, allocator가 E17 완료 전에 B@18을 claim했다면 device memory overlap은 이미 허용됐다. 따라서 두 문이 필요하다. 첫째, physical reuse 전 old writer completion을 기다린다. 둘째, 늦은 completion consumer가 generation을 비교한다. 하나는 device happens-before, 다른 하나는 host metadata isolation이다.

refcount도 generation에 귀속한다. `refcount[41]=1`은 불충분하고 `refcount[(41,18)]=1` 또는 record 안의 epoch-bound counter여야 한다. A@17 release token이 B@18 counter를 감소시키지 못해야 한다. owner token은 `(request_id, allocation_tx, block_generation, role)`처럼 effect를 식별한다. abort와 preemption이 같은 role을 release하려 하면 첫 소비만 성공하고 두 번째는 idempotent no-op 또는 명시적 duplicate 경고가 된다.

free list membership은 또 하나의 owner다. B41@18이 active table에도 있고 free list에도 있으면 숫자 합계가 맞더라도 중복 allocation이 가능하다. queue 삽입은 `refcount==0`, `no logical lock`, `no transfer pin`, `writer_ready`, `not already_member`, `generation current`를 한 transition에서 검증한다. 각 predicate를 서로 다른 시점의 snapshot으로 읽으면 check와 insert 사이에 다시 경쟁이 생긴다.

### 37.9.3 상태 전이: logical release와 physical reuse 사이에 문을 둔다

ABA-37의 안전한 상태는 다섯 개로 읽을 수 있다. `ACTIVE(A@17)`에는 request ref와 writer E17이 있다. abort/preemption 뒤에는 `LOGICALLY_RELEASED(A@17)`로 가며 request ref는 0이지만 event owner는 남는다. eviction policy는 cached identity를 지워 `DELAYED_FREE(A@17)`로 만들 수 있으나 free list에는 넣지 않는다. E17 완료를 소비한 뒤에만 `REUSABLE(41,next=18)`이 된다. B가 atomic claim하면 `ACTIVE(B@18)`이다.

허용되지 않는 edge는 `LOGICALLY_RELEASED(A@17) → ACTIVE(B@18)` 직행이다. cached metadata를 지우는 것과 writer readiness는 독립 조건이다. content hash가 제거됐다는 사실은 bytes를 덮어도 된다는 증거가 아니다. 반대로 E17이 끝났어도 A ref가 1이면 B에게 줄 수 없다. eligibility는 logical, policy, temporal 조건의 conjunction이다.

숫자로 검산한다. `t11` preemption이 owner token `A/scheduler/17`을 release해 ref가 0이 됐다면 `t12` abort는 같은 token을 다시 감소시키지 않는다. abort가 별도 `A/request-table/17` token을 실제로 소유했다면 초기 ref가 2였어야 한다. counter의 초기값과 owner edge 수가 맞지 않으면 이미 모델이 틀렸다. “경로가 둘이므로 decrement 두 번”이 아니라 “생성된 edge마다 release 한 번”이다.

free list에는 B41이 최대 한 번 등장한다. intrusive list라면 `is_free` flag와 links가 generation 17에서 detach되고 generation 18 claim 시 초기화되는지 본다. deque와 별도 set을 쓰면 두 자료구조 mutation가 함께 commit돼야 한다. callback가 deque에 같은 object를 두 번 append하면 allocator가 연속 두 번 B41을 pop할 수 있다. 첫 request와 둘째 request가 모두 `(41,18)`을 받는 더 직접적인 double allocation이다.

event completion 소비도 상태 전이다. poller가 `cudaEventQuery`에 해당하는 결과를 보고 delayed entry를 claim한다. 여러 host threads가 같은 ready entry를 볼 수 있다면 atomic remove 또는 once token이 필요하다. 첫 consumer만 `DELAYED_FREE→REUSABLE`을 수행하고 둘째는 이미 닫힌 completion을 관측한다. query가 ready라는 사실은 queue insertion effect가 아직 실행되지 않았다는 사실과 구분한다.

abort와 preemption 순서를 뒤집어도 결과가 같아야 한다. abort가 먼저 request를 terminal로 만들면 scheduler preemption은 terminal generation의 release effect를 재실행하지 않는다. preemption이 먼저 delayed owner를 만들면 abort는 client/output owner만 닫고 block delayed record를 보존한다. eviction이 둘 사이에 들어와도 victim policy는 cached identity만 제거하며 E17 fence를 건너뛰지 않는다.

또 하나의 branch는 B가 allocation되기 전에 A가 다시 resume되는 경우다. A generation 17의 reservation이 유지되는 설계라면 resume은 같은 identity를 되찾을 수 있다. 이미 reusable queue에 publish해 다른 claimant가 가져갈 수 있는 상태라면 resume은 새 generation과 새 blocks를 받아야 한다. “같은 request ID”를 이유로 old generation을 부활시키지 않는다.

### 37.9.4 source walk: allocator와 완료 소비자를 한 쌍으로 읽는다

vLLM을 읽을 때 `BlockPool`의 ordered free queue만 보면 절반이다. [`touch`, `free_blocks`, cached eviction 경계](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/block_pool.py#L641-L763)에서 block이 queue에서 빠지고 ref가 늘며, ref가 0일 때 어떤 순서로 돌아오는지 먼저 고정한다.

이어 [`KVCacheManager.free`와 finished request 정리 경계](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_manager.py#L256-L286)를 따라 request block table 소유권이 pool로 언제 넘어가는지 확인한다.

여기서 중요한 독해법은 구현에 없는 CUDA event field를 있다고 쓰지 않는 것이다. pin한 source가 증명하는 것은 allocator/ref/free transaction와 scheduler cleanup 경계다. 실제 backend가 synchronous ownership contract를 두는지, runner가 별도 completion fence를 두는지는 호출자와 worker 경로에서 추가로 확인해야 한다. allocator가 block을 free한다고 해서 그 함수 자체가 모든 kernel completion을 기다린다고 추론하지 않는다.

vLLM fixture에서는 네 질문을 코드 옆에 적는다. free 대상 iterable에 같은 block이 두 번 들어올 수 있는가. ref 0 block만 ordered free queue에 넣는가. cached hash 제거와 physical allocation은 어떤 순서인가. request finished/aborted transition가 worker가 소비할 metadata frontier보다 앞설 때 누가 old work를 drain하는가. 각 답의 source owner가 다르면 handoff contract를 연결한다.

SGLang에서는 token/KV location allocator와 radix ownership을 함께 본다. [`RadixCache`의 finished·unfinished request mutation](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/radix_cache.py#L458-L567)은 request token mapping과 last node lock이 cache owner로 넘어가는 장면을 보여 준다.

[`inc_lock_ref`·`dec_lock_ref`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/radix_cache.py#L622-L657)는 node path가 eviction eligible해지는 조건을 제공한다.

retraction을 읽을 때는 “메모리 부족으로 요청을 뺐다”에서 멈추지 않는다. request-to-token slots, token-to-KV locations, radix prefix lock, scheduler batch membership 가운데 어느 owner를 즉시 반환하고 어느 state를 resume용으로 남기는지 적는다. abort가 뒤따르면 동일 location range를 다시 free하지 않는 guard가 어디 있는지 본다. allocator의 integer indices만 로그에 남는다면 allocation generation을 관측용으로라도 붙여 ABA를 재현 가능하게 만든다.

SGLang의 policy가 unlocked radix nodes를 victim으로 고른다는 사실도 writer completion과 동일하지 않다. radix unlock은 logical cache ownership 조건이다. CUDA kernel이나 transfer가 그 locations를 더 쓰지 않는다는 보장은 실행 경로의 completion contract에서 와야 한다. source walk는 radix `dec_lock_ref()`에서 끝나지 않고, free된 token locations를 다음 batch가 block table에 싣는 소비 지점까지 이어져야 한다.

두 구현을 비교할 때 자료구조 이름을 맞추지 않는다. vLLM block object/ref/free queue와 SGLang token locations/radix locks는 granularity와 cache policy가 다르다. 공통으로 비교할 것은 identity claim, logical release, candidate publication, next consumer, delayed work boundary다. 이 다섯 질문이면 구현별 차이를 보존하면서 같은 ABA 위험을 찾을 수 있다.

### 37.9.5 관측과 검증: 합계가 아니라 세대별 edge를 검사한다

재현 fixture는 block 64개 전체가 필요 없다. B41 하나, 요청 A/B, scheduler thread, completion consumer 두 개면 충분하다. event completion을 수동 barrier로 막고 A write를 submitted 상태에 둔다. 그 사이 preemption과 abort 순서를 각각 실행하고 eviction pressure로 B41 old identity를 제거한다. B allocation을 시도했을 때 E17 전이라면 B41을 받지 않아야 한다.

첫 invariant는 identity uniqueness다. 모든 active table, reservation, delayed list, free queue에서 `(block_id,generation)`의 owning state가 정확히 하나다. cached index는 content reference로 공존할 수 있으나 physical claim owner와 역할을 구분한다. 같은 pair가 두 active requests에 있거나 free queue와 active table에 동시에 있으면 즉시 실패한다.

둘째는 edge conservation이다. generation 17에서 생성된 logical ref tokens 수와 성공한 unique release effects 수를 센다. abort, preemption, finish, timeout을 모두 호출해도 각 token은 한 번만 소비된다. counter가 0이라는 결과만 검사하지 않는다. `+2,-1,-1`과 `+1,-1,-1,+1`은 최종값이 같아도 중간 live eviction 위험이 다르다.

셋째는 temporal safety다. `ready(A@17,E17)` 또는 그에 상응하는 backend completion 증거가 `claim(B@18)`보다 happens-before여야 한다. timestamps는 clock 오차와 logging delay가 있으므로 event sequence나 queue handoff ID로 증명한다. 서로 다른 streams라면 wait/event dependency edge가 실제로 연결됐는지 본다.

넷째는 stale callback isolation이다. B@18 claim 뒤 A@17 completion callback를 한 번, 두 번, 순서를 바꿔 전달한다. B refcount, free membership, hash, table row는 전혀 변하지 않아야 한다. A delayed record만 closed가 되고 duplicate delivery metric이 증가할 수 있다. callback가 current ID lookup 뒤 mutation한다면 이 test가 바로 잡아낸다.

다섯째는 queue uniqueness다. free list를 순회해 block object identity와 generation pair의 중복이 없는지 검사한다. queue count와 set cardinality가 같아야 하고 links가 reciprocal해야 한다. B41을 두 번 pop하려는 artificial callback race에서 둘째 claim은 실패해야 한다. lock-free structure라면 ABA-safe tag나 해당 구현의 reclamation contract를 테스트한다.

여섯째는 content safety다. A@17 old writer에 sentinel pattern `0x17`, B@18 initialization에 `0x18`에 해당하는 작은 deterministic tensor를 사용한다고 사고 실험한다. B의 first read 시점에는 오직 B generation의 expected pattern만 허용한다. 실제 GPU 실행을 요구하지 않는 unit test에서는 fake completion gate와 shadow buffer mutation으로 ordering contract를 검증할 수 있다.

test matrix는 최소 `abort first/preemption first`, `event before/event after logical release`, `eviction 있음/없음`, `completion once/twice`, `B claim 시도/미시도`를 곱한다. 모든 조합을 무작정 늘리지 말고 transition edge가 달라지는 대표 조합을 고른다. 특히 event가 먼저 끝난 happy path만 있으면 ABA를 전혀 검증하지 못한다.

운영 telemetry에는 `block_id`, `generation`, `request_id`, `allocation_tx`, `owner_role`, `ref_before/after`, `free_membership`, `event_seq`, `scheduler_step`을 표본화한다. raw KV 값은 필요 없다. `stale_completion_total`, `duplicate_release_total`, `generation_mismatch_total`, `delayed_free_age`, `free_queue_duplicate`는 0이어야 하는 correctness 지표다. 정상 workload histogram과 섞어 평균으로 숨기지 않는다.

관측 비용 때문에 전 block을 상시 추적하기 어렵다면 anomaly-triggered ring buffer를 둔다. ref underflow, queue duplicate, generation mismatch, delayed age 초과가 생기면 앞뒤 owner events를 보존한다. 장애 뒤 현재 snapshot만 덤프하면 이미 B@18로 바뀌어 A@17 흔적이 사라진다. ABA는 이력 없이 현재값만 봐서는 정의상 놓치기 쉽다.

### 37.9.6 rollback: 안전한 완화와 완료 판정

사고 중 가장 안전한 즉시 완화는 의심 generation이 속한 allocator/worker epoch를 격리하는 것이다. B41 하나를 free로 강제 표시하지 않는다. old writer 존재를 증명할 수 없으면 해당 pool의 신규 admission을 멈추고 known completions를 drain한다. generation history가 유실됐다면 cache hit를 보존하려 하지 말고 worker-local KV state를 폐기한다.

기능 flag로 prefix caching 또는 aggressive preemption을 끄면 race 빈도는 낮아질 수 있다. 그러나 이는 root fix가 아니다. abort만으로도 logical release와 delayed writer가 겹칠 수 있다. 완화 효과를 “오류 사라짐”으로 읽지 않고 재사용 window가 줄었다는 반증 자료로 쓴다. correctness patch는 owner token, generation check, delayed-free fence를 모두 닫아야 한다.

rollback 뒤 재시작 조건은 free count 회복이 아니다. active generation마다 owner edge가 보존되고, delayed entries가 completion 뒤 유한 시간에 닫히며, free list 중복이 0이고, stale callback가 current state를 바꾸지 않아야 한다. 5,000회 fixture와 90분 abort/preemption soak에서 generation mismatch와 wrong-answer sentinel이 0인지 확인한다.

성능 회귀도 별도로 본다. 모든 free에 device-wide synchronize를 넣으면 ABA는 감춰지지만 unrelated streams까지 막아 ITL과 throughput을 훼손한다. 정확한 fence scope는 해당 generation의 last writer/reader dependency다. narrow event, stream ordering, delayed reclamation 가운데 실제 backend contract에 맞는 것을 쓴다. correctness와 concurrency를 동시에 지키는 지점이 설계 목표다.

최종 회고에는 최초 불일치를 한 문장으로 남긴다. “A@17의 E17이 incomplete인데 refcount 0만으로 B41을 free queue에 publish했다.” abort나 eviction을 원인이라고 쓰면 다음 구현에서도 같은 문제가 반복된다. 뒤이어 “late E17 consumer가 generation 없이 free(41)를 실행해 B@18을 변경했다”는 두 번째 결함을 분리한다. early reuse와 stale completion mutation은 각각 독립 회귀 test가 필요하다.

### 37.9.7 함수 단위 감사표를 실제 호출 순서로 채운다

이 사고를 source에서 찾을 때 `free`라는 문자열부터 전역 검색하면 후보가 너무 많다. 먼저 요청 A가 running batch에 들어간 시점의 block table 생성자를 찾고, 그 table을 worker input으로 직렬화하는 함수, device work를 enqueue하는 함수, scheduler가 finished·preempted·aborted 결과를 소비하는 함수, 마지막으로 pool 반환 함수를 한 줄로 연결한다. 각 함수 옆에는 입력 identity, 새로 만든 owner edge, 넘겨준 effect, 실패 반환을 적는다. 호출 그래프가 아니라 소유권 그래프를 만드는 작업이다.

allocation 함수에서는 반환값만 보지 않는다. free queue pop이 block object를 반환하는지 integer ID를 반환하는지, refcount 증가는 pop 내부인지 caller인지, cached metadata eviction은 claim 전인지 후인지, exception이 날 수 있는 mutation이 어느 사이에 있는지 읽는다. 함수가 list 전체를 atomic하게 얻는지 block마다 부분 성공하는지도 중요하다. 세 번째 block에서 실패할 때 앞의 두 block generation을 누가 소유하는지가 rollback 범위를 결정한다.

request table 갱신 함수에서는 host list append와 runner visibility를 분리한다. scheduler output이 만들어졌지만 queue에 넣기 전이라면 취소가 가능한가. queue에 들어갔지만 worker가 아직 읽지 않았다면 retract message가 같은 ordering channel을 쓰는가. worker가 generation 17 table을 읽은 뒤 host가 row를 지워도 device launch는 취소되지 않는다. source의 Python container mutation을 device lifetime의 끝으로 읽지 않는다.

completion 경로에서는 callback closure가 무엇을 캡처하는지 본다. request object 전체를 캡처하면 abort가 object fields를 초기화한 뒤 callback가 잘못된 default를 읽을 수 있다. block ID list만 캡처하면 generation을 잃을 수 있다. event record가 allocator delayed entry를 직접 가리키는지, request map lookup으로 다시 찾는지, worker epoch가 바뀐 뒤 old callback를 거부하는지 적는다. callback signature는 lifetime contract의 압축본이다.

free 함수에서는 “ref가 0이면 append” 전후의 guard를 본다. underflow assertion이 release build에서도 유지되는가. 이미 free queue member인 object를 다시 넣는 것을 막는가. cached hash reverse map을 먼저 지우는가. queue node links를 새 generation claim에서 정리하는가. 여러 blocks를 free할 때 reverse order가 policy만 바꾸는지 duplicate input도 가리는지 확인한다. ordered queue는 ownership 검증 장치가 아니라 policy 구조일 수 있다.

abort handler에서는 상태 guard 하나만 보고 idempotent라고 결론내리지 않는다. handler가 request status를 terminal로 바꾸기 전에 block refs 일부를 놓고 exception이 나면 재호출 guard가 cleanup을 건너뛸 수 있다. 반대로 status를 먼저 terminal로 만들고 cleanup를 별도 worker에 넘겼다면 그 worker task가 durable하게 등록됐는지 확인한다. idempotence는 함수가 두 번 호출돼도 된다는 속성이 아니라 effect마다 unique consumption record가 있다는 속성이다.

preemption handler는 resume state를 남기므로 finish와 다르다. recompute preemption은 KV를 놓고 token progress만 보존할 수 있고 swap 계열은 다른 tier owner를 만들 수 있다. 어떤 방식이든 A@17의 writer completion 책임이 사라지지 않는다. resume가 같은 request ID를 사용하더라도 new allocation transaction이면 generation이 달라야 한다. request identity와 physical allocation identity를 합치지 않는다.

eviction 함수는 candidate selection과 destructive mutation을 나눠 읽는다. ref 0, radix unlocked, no pin이라는 snapshot이 selector에서 true였어도 mutation 직전에 다시 유효한가. selection list를 만드는 동안 prefix hit가 touch할 수 있는가. single-thread scheduler가 serialization를 보장한다면 그 event loop 바깥 callback가 allocator를 만질 수 있는지도 본다. “single threaded”라는 설명은 모든 completion consumer까지 같은 thread라는 증거가 아니다.

CUDA 경계에서는 API 이름보다 ordering 범위를 기록한다. event가 어느 stream에 record됐고, block의 last writer가 정말 그 stream 앞에 있는가. 다른 stream의 copy가 event 뒤에도 진행되는가. default stream semantics에 암묵적으로 기대는가. graph replay라면 event node와 memory address binding이 generation마다 안전한가. 공식 CUDA contract가 제공하는 happens-before보다 넓은 보장을 allocator가 가정하면 그 간극이 결함 후보다.

관측 함수도 source audit 대상이다. free count metric이 queue length를 읽는지 ref 0 block 수를 세는지, delayed list를 별도 category로 노출하는지 확인한다. generation mismatch를 단순 debug log로 버리면 production에서는 silent no-op가 leak로 남을 수 있다. metric update가 mutation보다 먼저 일어나 exception에서 되돌아가지 않으면 dashboard만 정상처럼 보인다. correctness state와 관측 state의 commit 순서를 맞춘다.

테스트 source에서는 mock가 실제 경쟁을 제거하지 않았는지 본다. fake allocator가 block ID를 영원히 재사용하지 않으면 ABA가 발생할 수 없다. completion callback를 함수 호출 직후 동기 실행하면 delayed window가 사라진다. abort와 preemption을 같은 helper로 합치면 double effect를 검증하지 못한다. 최소 fake는 ID reuse, controllable completion, independent terminal paths 세 가지를 지원해야 한다.

첫 fault injection은 allocation 직후, publish 전이다. 이 지점에서는 B41@18이 worker에게 보이지 않았으므로 rollback 뒤 generation 18 record가 active table에 없어야 한다. 그러나 generation counter 자체를 17로 되돌릴 필요는 없다. monotonic identity에 gap이 생겨도 안전하다. 숫자를 연속으로 만들겠다고 generation을 재사용하면 실패 transaction의 늦은 callback와 충돌한다.

둘째 fault injection은 publish 뒤 device submit 전이다. worker queue가 generation 18 metadata를 받았는지 명확해야 한다. 받지 않았음을 증명하면 immediate release가 가능하다. 알 수 없다면 conservative delayed state로 보낸다. timeout 뒤 그냥 free하는 정책은 증명이 아니다. worker epoch를 폐기하거나 acknowledgment protocol로 frontier를 확정한다.

셋째 fault injection은 submit 직후 abort다. event gate를 닫아 놓고 logical refs가 0이 되는 것을 허용하되 free queue membership은 false여야 한다. eviction pressure를 최대로 올려도 B41은 후보로 나오지 않는다. event gate를 연 뒤 completion consumer가 한 번만 reusable transition을 수행하고, 그 다음 B가 generation 18로 claim한다. 이 순서가 핵심 happy recovery다.

넷째 fault injection은 B claim 뒤 old callback 재전달이다. production queue의 at-least-once delivery, timeout retry, shutdown drain 중 중복 소비를 모사한다. callback는 `(41,17)` delayed record가 이미 closed임을 보고 끝나야 한다. current `(41,18)` object의 refcount와 membership를 읽거나 쓸 필요조차 없게 설계할 수 있다. current lookup를 한다면 generation equality 이전에는 mutation가 없어야 한다.

다섯째 fault injection은 abort handler 중간 crash다. owner token release 전, release 후 delayed entry 등록 전, 등록 후 request map 삭제 전을 나눈다. recovery는 durable ledger나 worker epoch 폐기로 각각의 불확실성을 닫는다. map에 request가 없다는 사실만으로 blocks가 free됐다고 간주하지 않는다. 반대로 delayed entry가 있다는 이유로 logical ref를 다시 감소시키지 않는다.

여섯째는 free list corruption 자체다. 동일 object를 두 번 append하고 validator가 어느 시점에 잡는지 본다. allocation pop에서만 검사하면 그 전 queue length metric은 틀리고, 두 concurrent claimant가 pop을 시작한 뒤면 늦다. insertion transition에서 membership compare가 가장 좁은 방어선이다. 주기적 full scan은 latent corruption 탐지용 보조 장치다.

일곱째는 generation wrap과 process restart다. 64-bit generation은 현실적으로 wrap 가능성이 낮아도 worker restart 뒤 counter가 0부터 시작하면 old remote callback와 값이 겹칠 수 있다. identity에 worker boot epoch나 allocator UUID를 포함한다. local integer generation만 비교하는 설계는 process boundary를 넘는 completion·connector message에 충분하지 않다.

여덟째는 remote KV transfer다. source B41@17을 읽는 DMA가 남아 있으면 local CUDA writer가 끝났어도 physical reuse가 위험할 수 있다. transfer lease가 별도 pin인지 unified completion record인지 확인한다. abort가 remote request를 취소했다는 control acknowledgment와 actual data movement completion을 구분한다. remote epoch가 낡으면 새 owner state를 변경하지 못하게 한다.

아홉째는 graph capture/replay다. captured graph가 block table pointer를 간접 참조하는지 physical address를 고정하는지에 따라 generation safety가 달라진다. host table만 generation 18로 바뀌었는데 replay가 old address binding을 쓰면 allocator metadata 검사 바깥에서 overlap이 생긴다. graph key와 replay input validation에 allocation identity 또는 안전한 indirection contract가 있는지 확인한다.

열째는 multi-GPU ownership이다. tensor-parallel ranks가 같은 logical request의 서로 다른 physical blocks를 가지면 rank 0 completion만으로 전체 generation을 reusable로 만들 수 없다. 각 rank fence와 collective dependency가 닫혀야 한다. 한 rank가 abort를 먼저 처리하고 free list에 넣으면 다음 request의 collective가 stale peer와 맞물릴 수 있다. per-rank generation과 coordinator commit frontier를 함께 기록한다.

이 감사표의 종료 조건은 모든 함수 이름을 수집하는 것이 아니다. block identity가 만들어지는 곳, ref edge가 만들어지고 소비되는 곳, writer/reader completion을 증명하는 곳, free list publication, next consumer claim 사이에 빈 handoff가 없어야 한다. 빈칸이 implementation bug라는 뜻은 아니지만, source나 contract test로 설명되지 않으면 운영자가 사고 때 확인해야 할 우선 가설이다.

마지막으로 문서의 설명도 같은 엄격함을 지킨다. “abort하면 KV를 해제한다”라고 쓰지 않고 logical ref, cached identity, delayed writer, physical reuse 가운데 무엇을 언제 해제하는지 말한다. “event를 기다린다”라고 쓰지 않고 어느 stream의 어느 generation 작업을 누구의 consumer가 기다리는지 말한다. 친절한 설명은 용어를 줄이는 일이 아니라 독자가 다음 함수와 다음 증거를 스스로 찾게 경계를 밝혀 주는 일이다.

독자가 직접 실습할 때는 먼저 종이에 여섯 열을 만든다. 첫 열은 sequence, 둘째는 `(block,generation)`, 셋째는 logical owners, 넷째는 pending device·transfer work, 다섯째는 queue membership, 여섯째는 다음 mutation을 허용하는 predicate다. source 한 branch를 지날 때마다 한 행만 추가한다. refcount 숫자만 복사하지 말고 그 숫자를 만든 owner token을 나열한다. 이 표에서 같은 generation이 active와 free 두 행에 동시에 나타나거나, pending work가 있는데 reusable predicate가 true가 되면 source를 더 내려갈 지점이 정해진다.

그 다음 반대 방향으로 읽는다. B가 B41@18을 처음 소비하는 함수에서 시작해 누가 ID를 공급했는지, queue insertion은 누가 했는지, 어떤 completion이 insertion을 허용했는지 역추적한다. forward walk는 누락 cleanup을 잘 찾고 backward walk는 근거 없는 reuse를 잘 찾는다. 두 경로가 동일한 handoff event에서 만나지 않으면 중간에 암묵적 계약이 있다. comment, assertion, test가 그 계약을 증명하는지 확인한다.

마지막 실습은 로그 한 줄을 개선하는 일이다. `free block 41`을 `release logical owner=A/scheduler generation=17 ref=1→0 writer=E17 pending queue_insert=false`로 바꾼다. 완료 뒤에는 `consume E17 generation=17 delayed_owner closed reusable_publish generation=18`을 남긴다. B가 받으면 `claim block=41 generation=18 tx=B92 free_member true→false`가 이어진다. 이 세 사건이 있으면 운영자는 정수 ID가 되돌아온 것을 정상 재사용과 ABA mutation으로 구별할 수 있다.

리뷰어는 마지막으로 세 질문에 답한다. B41@17을 반환하라고 명령할 수 있는 주체는 누구인가. 그 명령은 어떤 완료 증거 뒤에 실행되는가. B41@18이 된 뒤 과거 명령은 어디에서 거부되는가. 첫 답만 있으면 double release를 놓치고, 둘째가 없으면 early reuse를 놓치며, 셋째가 없으면 stale callback를 놓친다. 세 답에는 각각 고정 source 위치나 executable invariant가 붙어야 한다. 설명이 “allocator가 알아서 안전하게 처리한다”에서 끝난다면 아직 함수 경계를 충분히 내려가지 않은 것이다. 반대로 세 답이 이어지면 독자는 다른 pool, connector, graph runner에서도 같은 수명 결함을 스스로 탐색하고 반증 실험까지 독립적으로 설계할 수 있다.

soak 종료 뒤에도 delayed age와 duplicate release의 장기 꼬리를 다시 확인한다.

## 37.10 운영자는 증상에서 어느 owner를 먼저 보는가

OOM/admission stall인데 사용률이 낮으면 reserved·watermark와 fragmentation, group별 shortage를 본다. 사용률이 높고 active request가 적으면 cached eligible이 실제 eviction 가능한지 lock/pin을 본다. evictable이 충분한데 allocation가 실패하면 policy/queue corruption이나 multi-group constraint를 본다.

output corruption이면 free count보다 live eviction/use-after-free를 우선한다. damaged S block generation의 이전 owner R과 return sequence를 찾는다. refcount 0 시각, victim metadata detach, allocator claim, S table publish, old writer completion을 잇는다.

cache hit rate 급락은 무조건 capacity 부족이 아니다. rollback가 cached metadata를 불필요하게 irreversible eviction하거나 free order가 prefix 앞부분을 먼저 희생할 수 있다. eviction events와 allocation failures, tail/head block position을 함께 본다.

latency tail은 eviction 자체보다 victim cleanup과 copy/transfer fence에서 생길 수 있다. allocation wait를 eligibility 계산, victim selection, metadata removal, physical readiness로 나눈다. policy 이름 하나로 latency를 설명하지 않는다.

### 37.10.1 owner snapshot은 같은 scheduler step에서 모은다

active request count는 step 100, free queue는 step 101, pin list는 step 99처럼 서로 다른 시점이면 정상 handoff가 leak 또는 overlap으로 보인다. snapshot barrier나 monotonic generation을 사용해 같은 causal frontier에서 partition를 재구성한다. full global lock가 비싸면 subsystem sequence와 handoff events를 연결한다.

request R 하나를 drill-down할 때 status, block/table generation, ref edges, radix locks, connector pins, last writer sequence, stream openness를 묶는다. aggregate cache usage에서 출발해도 마지막에는 owner identity로 내려가야 first divergence를 찾는다.

free queue dump에는 order와 membership를 둘 다 남긴다. count만 맞아도 duplicate B6과 missing B7이 상쇄될 수 있다. cached index reverse map도 block→hash와 hash→block 양쪽을 대조한다. one-way stale metadata는 lookup과 eviction에서 다른 증상을 낸다.

### 37.10.2 증상→관측→분기→검증으로 읽는다

증상이 admission stall이면 첫 관측은 physical partition와 group별 shortage다. uninitialized+cached eligible이 요구량보다 작으면 실제 capacity pressure다. 충분한데 실패하면 lock/pin, watermark, contiguous requirement, transaction corruption으로 분기한다. policy tuning은 eligibility가 정상임을 확인한 뒤다.

증상이 cache hit decline이면 hash/radix lookup requests와 candidate residency, eviction reason을 본다. capacity pressure에 따른 정상 victim 증가인지 allocation rollback가 content metadata를 불필요하게 지우는지 나눈다. hit rate만 보고 memory를 늘리지 않는다.

증상이 wrong output이면 live owner eviction와 stale generation을 최우선으로 본다. free queue length가 낮다는 사실은 부차적이다. damaged physical ID의 owner history를 역추적하고 last reader/writer completion보다 victim claim이 앞섰는지 검증한다.

증상이 shutdown hang이면 outstanding allocation transactions, future submissions, delayed owners, connector leases, output waiters를 센다. cached eligible blocks가 많이 남은 것은 hang 원인이 아니다. release condition가 있는 owners 중 가장 오래된 것을 찾는다.

### 37.10.3 가설을 반증하는 최소 변화

leaked pin 가설은 새 admission를 멈추고 existing requests를 drain했을 때 pin owners가 release conditions와 함께 0으로 수렴하는지 본다. cache content는 남을 수 있으므로 total used bytes 0을 요구하지 않는다. evictable capacity가 회복되는지가 핵심이다.

eviction policy 가설은 eligibility set을 고정하고 victim order만 바꿔 hit/latency를 비교해야 한다. async writer fence나 block size까지 동시에 바꾸면 원인을 분리할 수 없다. workload prefix reuse distribution을 기록한다.

double decrement 가설은 ref mutation에 owner token을 붙여 같은 owner가 release를 두 번 호출했는지 본다. 함수 call count만으로 부족하다. idempotent wrapper가 두 번 호출돼도 effect는 한 번일 수 있고 한 호출이 loop bug로 두 번 decrement할 수 있다.

partial orphan 가설은 fault boundary별 partition invariants를 검사한다. runner submit 전 failure에서 delayed owner가 생기면 불필요하지만 안전할 수 있다. 어떤 partition에도 없는 ID가 생기면 leak이고 두 partitions에 같은 generation이 있으면 double ownership이다.

### 37.10.4 복구 종료 조건을 correctness로 정한다

leak를 수동 free한 뒤 free count가 올랐다는 것으로 종료하지 않는다. ID generation의 no-writer, no-table-reference, no-pin을 증명한다. 불확실하면 worker/cache를 폐기한다. 가용성보다 silent corruption 방지가 우선이다.

refcount bug 수정 뒤 shared-prefix concurrency와 abort race에서 owner edges가 정확히 한 번 release되는지 정적 branch와 test fixture로 검증한다. prefix caching off에서만 정상인 것은 root invariant가 아직 깨졌다는 신호다.

rollback 수정 뒤 success path 성능도 본다. 모든 allocation에 global synchronization를 넣어 correctness를 얻으면 throughput과 tail이 크게 악화될 수 있다. transaction-local ledger와 narrow fence로 scope를 줄인다.

policy 변경 뒤 cache hit만 보지 않고 TTFT, recompute tokens, eviction work, admission failures를 함께 본다. 더 높은 hit rate가 locked nodes/pins 증가로 allocation stalls를 만들 수 있다. policy의 목적함수를 하나의 metric으로 축약하지 않는다.

**할당 장애 결정 트리.** free block 수가 줄고 live owner 수도 늘면 실제 capacity 압력을, owner는 줄었는데 refcount가 남으면 leak를, free 뒤 old completion이 refcount를 바꾸면 generation/ABA를 의심한다. eviction 직후 miss만 늘면 정책 문제지만 wrong answer가 나면 주소 재사용 fence 문제다. rollback은 allocator를 비우는 것으로 끝내지 않고 old generation callback이 더는 state를 바꾸지 못하는지 검증한다.

운영 판정은 refcount를 강제로 0으로 만드는 것이 아니다. allocation·share·release 사건 합이 현재 refcount와 일치하고 terminal request가 owner 집합에서 사라졌으며, 늦은 callback을 주입해도 새 generation 값이 보존될 때 복구를 승인한다.

## 37.11 단일 회고: free는 owner가 바뀌는 과정이다

fixture의 마지막 장면으로 돌아가면 처음 보였던 “free 세 개”가 왜 위험한 표현인지 분명하다. B0은 ref 0 cached candidate라 policy eviction 뒤에야 T가 쓸 수 있었다. B6은 content 없는 uninitialized resource라 곧바로 reserve할 수 있었다. abort 뒤 B7은 request ref가 0이어도 writer completion 전에는 delayed owner였다. 세 block은 aggregate gauge에서 비슷해 보여도 허용 전이가 달랐다.

이 차이를 무시한 allocator는 두 방향으로 실패한다. 모든 ref 0을 즉시 reusable로 보면 live GPU/transfer work가 새 owner를 덮는다. 모든 content-bearing block을 영구 보존하면 eviction가 일어나지 않아 capacity가 마른다. 정확한 구현은 eligibility predicate로 safety를 지키고 policy로 reuse value를 선택하며 fence로 시간 관계를 지킨다.

allocation 성공도 free queue에서 ID를 pop한 한 줄이 아니었다. reserve owner가 candidates를 격리하고, cached identity를 detach하며, request ref를 claim하고, table을 publish한 뒤 runner가 generation을 소비한다. 각 handoff에는 failure inverse가 있다. publish/submit 뒤에는 과거 상태 복원보다 terminal cleanup가 정확하다.

refcount, lock, pin을 구분한 까닭도 숫자 precision 때문만은 아니다. request ref의 release event는 finish/abort이고 radix lock은 tree owner handoff이며 transfer pin은 DMA completion/lease, writer fence는 device completion이다. leak를 고칠 때 counter 하나를 0으로 강제하면 아직 끝나지 않은 다른 owner를 지운다.

eviction eligibility와 policy를 나눈 덕분에 운영 판단도 달라진다. locked/live resource가 victim이 된 사고는 LRU가 나쁜 것이 아니라 candidate set correctness가 깨진 것이다. eligible content가 너무 빨리 사라져 hit가 낮은 현상은 policy/order 문제일 수 있다. 둘을 같은 cache tuning 문제로 다루지 않는다.

rollback가 initial snapshot과 byte-for-byte 같지 않아도 정상일 수 있다. B0 hash eviction가 irreversible이면 B0은 uninitialized로 남지만 physical owner partition와 correctness는 회복된다. 반대로 hit metric을 보존하려 old hash를 무리하게 복원했다가 content generation이 달라지면 silent corruption가 된다. 성능 state 복원보다 identity truth가 우선이다.

crash cleanup은 이 원칙을 가장 엄격하게 드러낸다. owner ledger와 completion을 신뢰할 수 없으면 개별 blocks를 normal pool로 돌리지 않는다. worker/device allocator를 폐기하고 remote lease를 epoch에 맞춰 회수한다. 서비스 재개가 조금 늦더라도 불확실한 KV를 다음 request에 주지 않는다.

독자가 새 allocator source를 만났을 때 찾을 것은 `free()`라는 이름 하나가 아니다. reserve owner가 생기는 순간, live reader/writer를 표현하는 state, candidate eligibility가 true가 되는 predicate, victim policy가 개입하는 위치, publish commit, failure inverse와 crash boundary다. 이 일곱 관계를 찾으면 자료구조 이름이 달라도 수명 기계를 재구성할 수 있다.

수치 검산은 이 관계가 빠졌는지 빨리 드러낸다. B0–B7 ID union이 항상 여덟이고 active·reserved·delayed·cached·uninitialized partitions가 겹치지 않아야 한다. logical ref edges는 physical count와 별도로 센다. transaction 전후 합계가 맞아도 duplicate ID와 missing ID가 상쇄될 수 있으므로 set identity까지 본다.

시간 검산도 필요하다. request ref가 0이 된 시각, cached metadata eviction, new generation claim, writer completion, physical reuse 사이 happens-before를 적는다. aggregate snapshot가 맞더라도 claim이 old writer completion보다 앞서면 use-after-free다. 상태 partition와 event ordering은 서로 대신하지 못한다.

fixture에서 B6 transfer pin이 늦게 풀리는 장면은 정상 long tail과 leak를 구분한다. pin owner와 completion condition가 존재하고 완료 뒤 유한 시간에 release되면 정상이다. owner 문자열만 남고 condition를 찾을 수 없거나 완료 뒤에도 pin이 유지되면 leak다. timeout을 짧게 해 counter를 지우는 것은 검증이 아니다.

정책 실험은 candidate safety를 고정한 뒤 해야 한다. ref·lock·pin predicate가 흔들리는 상태에서 LRU와 다른 정책을 비교하면 live eviction 차이가 hit rate 차이처럼 섞인다. 먼저 owner invariant를 증명하고, 그 다음 같은 eligible sequence에서 victim order와 cache value를 비교한다.

마지막으로 정상 cleanup와 crash recovery를 같은 성공 기준으로 두지 않는다. 정상 path는 block 단위 identity와 cached content를 보존할 수 있다. crash path는 불확실한 local generation을 폐기하고 remote leases와 callers를 terminal로 만드는 것이 성공이다. cache hit 보존을 위해 신뢰할 수 없는 state를 되살리지 않는다.

운영 로그도 이 모델을 말할 수 있어야 한다. “B6 freed” 대신 “T request ref를 해제했으나 transfer X pin과 writer step 71이 남았다”라고 쓴다. 뒤에는 “step 71 completion 후 writer owner 제거”, “transfer X epoch 4 completion 후 uninitialized queue generation 5로 반환”이 이어진다. 이 세 줄이면 늦은 정상 release와 leak를 구별할 수 있다.

“T allocation failed”도 commit 위치를 포함한다. reserve 전 capacity rejection인지, B0 metadata eviction 뒤 rollback인지, runner publish 뒤 terminal cleanup인지가 없으면 같은 failure counter에 correctness와 efficiency 사건이 섞인다. transaction generation과 last committed mutation를 기록한다.

이 정도 관측은 모든 request의 상세 dump를 요구하지 않는다. allocation failure, 장기 pin, ref underflow, partition mismatch가 생긴 transaction과 표본 request에 집중한다. raw KV content 없이 owner IDs와 generation, sequence만으로 대부분의 수명 결함을 찾을 수 있다.

이 장에서 free는 bool이 아니었다. active ref가 0이 되어 cached eviction candidate가 되는 것, content metadata를 제거하는 eviction, physical ID를 새 generation에 claim하는 allocation, in-flight writer 뒤 실제 reuse가 각각 다른 사건이다. lock과 pin은 서로 다른 release condition으로 eligibility를 제한한다.

free-block count 하나가 높다고 안전한 것도, 낮다고 곧바로 누수인 것도 아니다. 어떤 generation을 누가 읽거나 쓰며 어느 조건 뒤 candidate가 되는지 설명할 수 있을 때만 그 숫자가 의미를 얻는다.

이 owner 설명이 완전히 닫혀야 allocator의 성능 수치와 정확성 판단을 같은 언어로 올바르게 비교할 수 있다.

vLLM은 cached candidates가 포함된 ordered free queue와 touch/ref, group coordinator를 쓴다. SGLang radix는 node lock과 token locations, policy lists를 함께 소유한다. Transformers는 complete initialized와 incomplete uninitialized free를 나눈다. llama.cpp는 sequence-tagged cells의 empty/slot eligibility를 찾는다. 같은 refcount 표로 덮으면 중요한 차이가 사라진다.

rollback은 무조건 이전 bytes를 복원하는 일이 아니다. 이번 transaction이 얻은 refs, locks, pins, reservations, table publication을 commit boundary에 맞춰 해소하는 일이다. cached metadata eviction가 irreversible이면 correctness를 유지한 채 cache hit opportunity를 잃을 수 있다. kernel submit 뒤에는 rollback보다 fenced cleanup이 필요하다.

38장은 이 local owner model을 CPU와 remote tier로 확장한다. 다음 질문은 eviction된 GPU content가 사라지는 대신 CPU tier로 이동할 때 source와 destination 중 누가 transfer 동안 owner이며, 실패한 copy가 어느 tier의 allocation와 pin을 되돌려야 하는가다.
