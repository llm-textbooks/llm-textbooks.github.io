# 36장. Sliding·hybrid cache와 physical address

sequence position 19가 sliding ring의 physical slot `19 mod 8=3`에 쓰인다고 하자. slot 3에는 이전 position 11이 있었다. writer가 value를 덮고 absolute-position metadata를 갱신하기 전에 reader가 slot 3을 보면 address는 유효하고 값도 finite하다. crash 대신 조용한 오답이 나온다.

이 장은 12 layers 모델을 따라간다. full attention 4 layers, sliding-window attention 4 layers(`W=8`), recurrent/SSM 4 layers다. length 20에서 full layers는 20 positions, sliding layers는 최근 window와 in-flight contract, recurrent layers는 token-indexed K/V 대신 sequence별 conv/state를 가진다. `12×L×KV-byte` 하나로는 세 lifetime과 address model을 모두 틀린다.

33장은 component bytes를, 34~35장은 pages·block tables·prefix sharing을 설명했다. 이 장은 layer group마다 logical position이 어떤 physical address/state slot로 번역되고 언제 재사용되는지 닫는다. 37장의 allocation/refcount/eviction lifecycle은 깊이 침범하지 않는다.

따라서 14장의 representation 수학이나 33장의 총 byte 식이 여기 다시 등장해도 재설명하려는 것이 아니다. 이 장이 소유하는 판정은 같은 logical position과 계산된 allocation이 full page, sliding ring, recurrent slot에서 서로 다른 address·overwrite·commit 규칙으로 보존되는가다.

고정 source는 vLLM `v0.27.1` commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang `v0.5.18` commit `71de97b264b04dcd514cf904003028aefe9775c8`, Transformers `v5.15.1` commit `550d7b3834670483a4df436541272c055dc364bf`, llama.cpp `v0.2.0` commit `bb4caa7540188872173c44d161602d9271386413`이다. runtime은 실행하지 않는다.

## 36.1 같은 cache라는 말이 세 주소 모델을 숨긴다

문제는 full, sliding과 recurrent state를 layer axis만 다른 같은 tensor로 보는 순간 시작한다. full attention은 absolute position 0…19의 K/V가 향후 query에서 필요할 수 있다. sliding layer는 오래된 positions를 재사용 가능한 slots로 바꾼다. recurrent layer는 history를 state vector에 접어 넣으므로 position별 K/V 주소 자체가 없다.

도서관 비유에서 full cache는 모든 과거 문서를 보관하고, sliding cache는 최근 서류만 남기는 회전 선반이며 recurrent state는 과거를 요약한 한 권의 장부다. 한계는 회전 선반이 단순 FIFO가 아니라 absolute position metadata와 mask contract를 필요로 하고, 요약장은 update ordering이 틀리면 되돌릴 원본 positions가 없다는 점이다.

주소는 pointer 하나가 아니다. request/sequence generation, layer group, logical absolute position, page/block 또는 ring slot, valid generation과 dtype/layout이 함께 identity를 만든다. 동일 slot number 3이라도 full group page 3, sliding ring slot 3과 recurrent sequence slot 3은 호환되지 않는다.

### 주소를 tuple로 쓰면 무엇이 보이는가

full layer의 logical address를 `(request_generation, layer, absolute_position)`으로 쓸 수 있다. block allocator는 이를 `(cache_group, block_id, offset)`으로 번역한다. block size 4에서 position 19는 logical block index 4, offset 3이다. block table이 physical block 27을 가리키면 backend address는 group tensor의 block 27, offset 3이다.

sliding layer는 position 19를 ring slot 3에 놓을 수 있다. physical tuple은 `(sliding_group, request_generation, slot=3, slot_version, stored_absolute=19)`다. slot number만으로는 position 11의 old state와 구별할 수 없다. mask/read mapping은 query absolute position과 stored absolute metadata가 visibility 범위에 드는지 확인해야 한다.

recurrent layer는 position 19 state를 `(recurrent_group, sequence_slot, request_generation, state_version=19)`처럼 볼 수 있다. state version은 19개 updates를 반영했다는 의미이지 token 19의 K/V가 별도 cell에 있다는 뜻이 아니다. rewind/resume가 있으면 restored version과 next expected version을 합의한다.

이 tuple은 실제 implementation struct를 요구하는 설계안이 아니다. source의 여러 arrays/objects에 흩어진 identity fields를 독자가 한 줄로 재구성하는 분석 도구다. 어떤 항이 source에 없으면 safe boundary나 implicit ownership이 그것을 대신하는지 찾는다.

### 같은 slot number가 만드는 잘못된 디버깅

log에 `slot=3`만 남으면 operator는 full block offset 3, sliding ring slot 3과 server sequence slot 3을 섞는다. recurrent state corruption을 sliding overwrite로 오진할 수 있다. metric은 layer group/type과 unit kind를 붙이고 request identity는 sampled trace에 둔다.

GPU pointer가 동일하다고 state identity가 같지도 않다. allocator가 request A 종료 뒤 같은 address를 B generation에 재사용할 수 있다. pointer equality는 physical storage reuse를 말할 뿐 logical lifetime equality가 아니다. generation과 commit event를 함께 본다.

## 36.2 12-layer W=8 fixture를 손으로 펼친다

positions 0…7까지 sliding slot은 absolute position과 같다. position 8은 slot 0을 재사용하고 old position 0을 대체한다. position 11은 slot 3, position 19도 slot 3이다. position 19 write 전 slot 3 metadata는 absolute 11, generation G다. write/commit 뒤에는 absolute 19, 같은 request generation G와 new slot version을 가리켜야 한다.

length 20에서 full layers each 20 positions, four layers 합 80 layer-positions다. sliding layers each 최대 8 retained positions라 단순 steady-state 합은 32다. recurrent layers는 four sequence states다. 하지만 chunked prefill in-flight query가 window 밖 prefix를 잠시 필요로 하면 sliding physical maximum이 정확히 `4×8`이라고 단정하지 않는다.

position 19 query가 sliding attention에서 읽을 logical keys는 contract에 따라 최근 positions 범위다. kernel mask가 absolute positions를 비교하는지, precomputed slot map/lengths를 쓰는지 확인한다. ring에 position 11과 19가 같은 slot을 쓴다는 사실만으로 visibility를 결정하지 않는다.

### fixture를 bytes와 write traffic까지 확장하기

full/sliding attention의 K/V position·layer payload를 B=4KiB라고 하고 recurrent state·layer·sequence payload를 R=64KiB라고 가정하자. 이 값은 계산 방법을 위한 fixture이며 특정 model 측정값이 아니다. length 20에서 full logical state는 `4×20×4KiB=320KiB`, sliding retained baseline은 `4×8×4KiB=128KiB`, recurrent state는 `4×64KiB=256KiB`다. 합은 704KiB다.

token 19 한 step의 new write traffic baseline은 full four positions 16KiB, sliding four overwritten positions 16KiB와 recurrent four state updates 최대 256KiB다. recurrent update가 full state를 읽고 쓰면 read+write traffic은 더 크고 fused kernel/partial state semantics에 따라 달라진다. retained bytes와 per-step traffic은 같은 숫자가 아니다.

hybrid manager disabled로 sliding layers가 full length 20을 reserve하면 sliding capacity component는 128KiB에서 320KiB 방향으로 늘고 total baseline은 896KiB다. recurrent 256KiB는 그대로다. exact physical bytes는 blocks/group padding/in-flight를 적용해야 하지만 어떤 component가 192KiB 증가 방향인지 설명할 수 있다.

W=16으로 바꾸면 sliding retained baseline은 256KiB이고 total은 832KiB다. position 19가 slot 3을 쓰더라도 old occupant는 position 3이며 retained logical range는 4…19다. W=8의 range 12…19와 다르다. modulo slot만 관측하면 visibility 의미를 놓친다.

block size가 4이고 sliding physical pages를 independently round한다고 단순화하면 W=8은 layer당 two blocks, W=16은 four blocks다. chunked prefill extra capacity가 6 tokens라면 required capacity가 W+6=14로 계산되는 spec은 layer당 four blocks가 필요할 수 있다. W=8이라는 config 아래 tensor capacity 16을 보는 이유가 된다.

full length 20은 layer당 five blocks다. fixture four layers에서 full blocks 20, sliding steady W=8 blocks 8, in-flight 14 blocks 16으로 달라진다. hybrid disabled promotion은 sliding도 20 blocks 방향이다. block table/group allocation detail은 인접 장에서 다루지만 address capacity 비교에는 이 count가 유용하다.

metric reconciliation은 logical retained bytes 704KiB, allocated page capacity, backend tensor reserved와 process reserved를 네 줄로 둔다. request 종료 뒤 first line과 used units가 줄어도 tensors는 유지될 수 있다. 반대로 position 19 오답은 bytes가 모두 맞아도 tag/version ordering에서 생긴다.

이 fixture는 memory 최적화와 correctness가 다른 축임을 보여 준다. sliding은 retained bytes를 줄이지만 overwrite address와 metadata commit을 추가한다. recurrent state는 context-linear memory를 피하지만 sequence-slot generation과 read-modify-write dependency를 추가한다. 절감 이유와 새 실패 표면을 함께 설명해야 한다.

독자는 B와 R을 자기 model cache spec fields로 바꾸면 된다. dense attention B는 KV heads/head dim/dtype에서, MLA면 latent components에서, recurrent R은 conv/state tensor shapes에서 얻는다. 모든 12 layers에 같은 B를 곱하지 않는다.

### position 0부터 19까지 ring을 전개하기

W=8이고 간단히 `slot=p mod 8`이라 두자. positions 0…7은 slots 0…7을 처음 채운다. position 8은 slot 0의 old absolute 0을 덮고, 9는 slot 1의 1을, 10은 slot 2의 2를, 11은 slot 3의 3을 덮는다. 이때 retained absolutes는 4…11이다.

position 12…15는 slots 4…7을 덮어 retained set을 8…15로 만든다. position 16은 slot 0의 8, 17은 slot 1의 9, 18은 slot 2의 10, 19는 slot 3의 11을 덮는다. commit 뒤 ring slots 0…7의 stored absolutes는 `[16,17,18,19,12,13,14,15]`다. physical slot order와 logical chronological order가 다르다.

query position 19가 keys 12…19를 볼 수 있는 contract라면 chronological gather order는 slots 4,5,6,7,0,1,2,3이다. kernel이 slots 0…7을 그대로 chronological로 해석하면 positions 16…19가 12…15보다 앞에 놓인다. absolute positions 또는 ring start pointer가 필요하다.

일부 causal contracts에서는 current position의 K/V write와 attention read ordering 때문에 query 19가 keys up to 19를 포함하거나 prior 18까지만 읽고 current를 별도 처리할 수 있다. 이 장의 modulo fixture는 identity를 보여 주며 exact inclusive boundary는 backend mask source에서 확인한다.

### 12 layers의 state count를 같은 단위로 세지 않기

full layers 0,3,6,9처럼 pattern이 interleaved될 수도 있고 처음 4개일 수도 있다. logical model layer order와 cache group order가 다르면 group index 1이 model layer 1이라는 가정이 깨진다. layer-name/spec mapping을 보존한다.

length 20에서 full four layers는 each 20 K/V positions다. sliding four layers는 steady-state each eight retained positions지만 in-flight query/prefill buffer가 더 있을 수 있다. recurrent four layers는 each one sequence state plus architecture-specific convolution buffer다. `80+32+4=116`은 unit이 다른 합이므로 memory bytes로 쓰지 않는다.

bytes를 계산하려면 full/sliding K/V position bytes와 recurrent state shape bytes를 각각 곱해 합한다. group allocator가 common page size로 padding하면 physical reservation을 다시 계산한다. 33장의 shape 장부를 이 address fixture에 가져온다.

### chunked prefill가 W보다 큰 physical interval을 만드는 장면

P prompt positions 0…19를 chunk `[0,12)`, `[12,20)`로 처리한다고 하자. first chunk의 later queries가 causal/window attention을 수행할 때 chunk 내부 tokens와 prior context를 함께 표현해야 한다. backend가 chunk query rows를 한 번에 처리하면 current chunk tokens W=8만으로 단순 cap하지 못할 수 있다.

vLLM sliding spec이 maximum memory를 계산할 때 in-flight chunked prefill tokens를 고려한다는 source note가 중요한 이유다. `window=8` config가 `physical slots exactly 8`을 뜻하지 않는다. maximum batched tokens, block alignment와 backend layout이 추가 capacity를 만들 수 있다.

운영자는 logical retained history W, current iteration query tokens, physical cache capacity를 세 값으로 기록한다. physical 16 slots가 보인다고 sliding이 꺼졌다고 단정하지 않는다. 반대로 full sequence 20 capacity가 보이면 fallback 승격인지 in-flight overhead인지 spec type과 sizing predicate를 본다.

### generation 사고를 네 interleaving으로 나누기

정상 update는 old slot metadata `(abs=11,version=v)`를 읽고 overwrite eligibility를 확인한다. writer는 new K/V for 19를 저장하고 new metadata `(abs=19,version=v+1)`를 publish한다. reader는 matching committed version만 소비한다.

사고 A는 value first, metadata late다. reader가 new K/V 19와 old absolute 11을 결합해 RoPE/mask 또는 output gather를 잘못 해석한다. 사고 B는 metadata first, value late다. reader가 old K/V 11을 position 19로 본다. 두 경우 모두 pointer와 shape는 유효하다.

사고 C는 request generation reuse다. A generation 7이 slot 3 update를 submit한 뒤 slot owner가 B generation 8로 바뀐다. late A write가 B state를 덮는다. slot version만 있고 request generation이 없으면 v+1이 정상처럼 보일 수 있다.

사고 D는 layer-group partial commit이다. full groups와 recurrent groups는 epoch t를 commit했지만 sliding group 하나가 old version이다. model output은 layers를 순서대로 통과하므로 later hidden state가 mixed temporal state를 반영한다. single request-level cache length가 모두 정상이라고 거짓말한다.

복구 invariant는 group별 expected input version과 committed output version이다. next model iteration은 required groups가 epoch t를 commit한 뒤 시작한다. asynchronous overlap을 허용해도 event dependencies가 happens-before를 보존한다.

## 36.3 full·sliding·recurrent의 lifetime과 commit

full KV commit은 new absolute position을 block/page slot에 쓰고 block table/logical length를 전진한다. release는 request terminal, eviction/offload와 sharing refcount에 걸린다. overwrite-by-window는 일반 full contract가 아니다.

sliding commit은 overwrite 대상 slot의 last reader가 끝난 뒤 new K/V와 absolute-position/generation metadata를 일관되게 publish해야 한다. value 먼저, metadata 나중 또는 반대 순서에서 concurrent reader가 intermediate state를 보지 않게 stream/event 또는 iteration boundary가 필요하다.

recurrent commit은 sequence slot의 old state를 읽어 new state를 쓰는 read-modify-write다. token 19 state가 token 18 state에서 유도된다. slot이 다른 request generation에 재사용되면 reset/restore가 first update보다 먼저 완료돼야 한다. position table보다 sequence-slot generation이 핵심이다.

hybrid model runner는 같은 request step에서 full KV append, sliding ring overwrite와 recurrent state update를 모두 수행할 수 있다. 하나의 `cache_updated=true` boolean은 어느 component가 commit됐는지 표현하지 못한다. layer group별 commit epoch와 readiness를 둔다.

### full append의 정상 commit

full layer에서 position 19를 쓰기 전에 request block table이 logical block 4를 physical block 27로 가리킨다고 하자. writer input에는 block 27, offset 3, absolute position 19, layer mapping과 request generation이 들어간다. K/V write 완료 뒤 logical cache length가 20으로 전진한다.

cache length를 write submit 전에 20으로 올리면 next reader가 incomplete slot을 볼 수 있다. 반대로 write는 끝났는데 scheduler length가 19면 same position을 다시 계산하거나 capacity를 과대 reserve한다. device completion과 host metadata update의 sync boundary를 찾는다.

prefix-sharing full block이면 partial tail write가 copy-on-write를 요구할 수 있다. 이 lifecycle은 35/37장에 맡기되, physical address가 share-safe unique block으로 resolve된 뒤 write한다는 precondition을 둔다.

### sliding overwrite의 정상 commit

slot 3을 overwrite할 때 old absolute 11이 current query들이 더 읽지 않는지 확인한다. iteration boundary가 old readers completion을 보장하거나 explicit event/refcount가 필요하다. overwrite eligibility와 attention visibility mask는 별도다. 쓸 수 있다고 해서 new position이 모든 queries에 보여야 하는 것은 아니다.

K/V와 metadata가 separate tensors라면 atomic multi-tensor write가 아니다. kernel 하나가 values와 tags를 순서대로 쓰더라도 other stream reader와 ordering이 필요하다. same stream order, CUDA event 또는 batch boundary가 contract를 만든다.

slot generation counter가 wrap하거나 dtype width가 작으면 ABA-like reuse가 생길 수 있다. 실용적으로 request generation+absolute position 비교가 충분한지 source field width와 lifetime을 본다. pointer/slot number 단독 validation은 부족하다.

### recurrent read-modify-write의 정상 commit

recurrent state S18에서 token 19 input을 받아 S19를 만든다. in-place update면 kernel이 old state를 모두 읽기 전에 다른 operation이 same slot을 reset/overwrite하면 안 된다. out-of-place double buffer면 active buffer selector를 commit한다.

convolution state와 recurrent state가 separate arrays라면 둘 다 same token version으로 전진해야 한다. conv S19, recurrent S18 같은 partial state는 shapes가 맞지만 output이 틀린다. component-version tuple을 둔다.

preemption/resume에서 state를 offload/restore하거나 recompute할 수 있다. resume scheduler가 slot을 running으로 표시하기 전에 restore completion과 generation match를 확인한다. request A old host snapshot을 B slot에 복사하는 사고를 tenant mixing으로 관측할 수 있다.

### 세 commit을 barrier 하나로 묶는 비용과 대안

모든 groups마다 global device synchronize를 하면 correctness는 단순하지만 overlap과 throughput을 잃는다. same stream ordering, per-group events와 iteration-level dependency로 필요한 edges만 만들 수 있다. 정확한 mechanism은 stack/backend마다 다르다.

관측에서는 synchronize call 유무보다 producer stream, completion event, consumer wait와 metadata publish 순서를 본다. asynchronous이라고 unsafe가 아니며 synchronous이라고 group mapping/generation 오류를 막는 것도 아니다.

fault injection conceptual fixture는 sliding metadata publish 지연, recurrent reset 지연과 full length early advance를 각각 분리한다. runtime을 실행하지 않더라도 source call order와 expected state table로 first unsafe window를 찾을 수 있다.

## 36.4 vLLM의 cache spec과 hybrid grouping

[`vllm/v1/kv_cache_interface.py:235-353`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_cache_interface.py#L235-L353)에서 FullAttentionSpec과 SlidingWindowSpec의 page-size·memory 계산을 비교한다. [`kv_cache_interface.py:559-756`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_cache_interface.py#L559-L756)은 sliding MLA와 Mamba state spec/merge 조건을 잇는다.

sliding maximum은 W만이 아니라 chunked prefill in-flight tokens 때문에 커질 수 있다. config window 8을 tensor slots 8로 바로 번역하지 않고 spec method가 계산한 maximum memory와 block alignment를 사용한다.

[`vllm/v1/core/kv_cache_utils.py:1144-1500`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_utils.py#L1144-L1500)과 [`kv_cache_utils.py:1501-1795`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_utils.py#L1501-L1795)에서 layer specs를 page size/pattern으로 group하고 common allocator tuple에 연결하는 경계를 본다. hybrid manager disabled path가 sliding spec을 full spec으로 승격하면 reservation이 늘 수 있다.

**position 19가 vLLM cache 계약을 통과한다**

caller는 model의 cacheable layer list에서 layer type과 dimensions를 읽어 spec object를 만든다. fixture의 full four, sliding four와 recurrent four가 logical model layer names에 연결돼야 한다. group order를 재배치해도 layer-name→spec mapping을 잃으면 correct bytes의 wrong state를 소비한다.

full spec은 heads, head size, block size와 dtype으로 page bytes를 만든다. block size 4에서 position 19는 logical block 4, offset 3이다. scheduler/manager가 physical block ID를 제공하면 runner input은 layer group, block과 offset을 attention backend slot mapping으로 내린다. mutation은 K/V write와 computed/cache progress commit이다.

sliding spec은 window 8뿐 아니라 maximum batched/in-flight tokens와 alignment 조건을 소비한다. predicate는 current prefill/query가 old history를 더 요구하는지와 bounded capacity가 얼마인지다. runner consumer는 scheduled positions와 block/slot metadata를 attention backend에 넘긴다. position 19가 old 11 storage를 재사용한다는 fixture는 이 bounded identity를 설명하지만 exact modulo implementation은 backend source를 따른다.

Mamba spec은 K/V heads 식 대신 conv/state shapes에서 page/state bytes를 만든다. position 19 input은 request의 sequence state version 18을 읽어 19로 갱신한다. common allocator tuple에 함께 들어가도 attention block ID와 Mamba state handle은 같은 unit이 아니다.

grouping 함수는 specs의 page size와 repeating layer pattern을 맞춰 groups를 만든다. group tuple component와 logical layer mapping이 runner까지 보존된다. full append, sliding bounded address와 recurrent slot이 같은 request schedule에서 함께 내려가지만 consumers는 서로 다른 metadata를 읽는다.

hybrid manager disable predicate가 true이면 sliding spec이 full로 승격될 수 있다. position 19는 bounded reuse 대신 full-history append path가 되고 correctness는 유지돼도 reservation이 증가한다. full position-byte를 B라 하면 attention capacity가 steady fixture의 `80B+32B=112B`에서 `80B+80B=160B` 방향으로 늘어난다. exact bytes에는 blocks와 in-flight capacity가 들어간다.

stream 비용은 device K/V/state writes, metadata upload와 group dispatch다. Python spec은 CUDA ordering을 증명하지 않는다. runner/backend caller에서 current stream, any async copy/event와 output completion을 더 따라간다. 역방향 debug는 first wrong layer group에서 selected spec, slot mapping과 layer-name mapping으로 올라온다.

## 36.5 SGLang·Transformers·llama.cpp의 주소 소유권

SGLang은 [`swa_radix_cache.py:1-420`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/swa_radix_cache.py#L1-L420)에서 full/SWA hybrid radix ownership과 sanity checks를 읽는다. SWA token-to-KV allocator의 logical position→pool slot 번역과 일반 allocator를 구분한다.

Transformers [`cache_utils.py:1702-1801`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L1702-L1801)은 config layer types에서 sliding/hybrid/linear-attention layer object를 선택한다. [`cache_utils.py:1278-1379`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L1278-L1379)은 offload/prefetch stream ordering과 non-sliding selection을 확인한다.

llama.cpp [`llama-model.cpp:2098-2383`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model.cpp#L2098-L2383)은 architecture/context에 따라 plain KV, recurrent, hybrid와 hybrid-ISWA memory factory를 선택한다.

[`llama-kv-cache.cpp:1036-1100`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L1036-L1100)과 [`llama-kv-cache.cpp:1665-1708`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L1665-L1708)은 SWA mask와 cell eligibility를 잇는다.

**position 19가 SGLang full/SWA pools를 통과한다**

caller는 scheduler/model-worker 초기화에서 일반 token-to-KV allocator와 SWA-aware cache/pool을 구성한다. position 19 work가 batch에 들어오면 request logical position과 physical pool-slot requirements가 metadata로 내려간다. full layers와 SWA layers가 same prompt progress를 공유해도 pool address가 같다고 가정하지 않는다.

full radix path는 prefix tokens와 pool indices를 node ownership/refcount에 연결한다. position 19가 new suffix면 new full KV slot에 쓰고 logical prefix ownership을 갱신한다. cache hit이면 existing slots를 참조하지만 partial tail write 전에 unique ownership이 필요하다.

SWA hybrid predicate는 full-prefix match뿐 아니라 bounded SWA state가 reusable absolute range/generation을 갖는지 확인해야 한다. full cache가 positions 0…19를 갖는다고 SWA storage가 position 19의 recent window와 current phase를 자동 보장하지 않는다. hybrid radix node의 full/SWA lengths와 sanity checks를 함께 읽는다.

SWA token-to-KV allocator는 logical position을 dedicated pool slot로 번역한다. exact modulo인지 indirection table인지는 source mutation을 따른다. old occupant invalidation, new position/generation association과 device K/V write가 한 commit chain을 이룬다.

recurrent pool은 request/sequence slot을 선택하고 conv/state tensor slice를 model worker에 준다. fused update가 old state와 token 19를 소비해 new state를 쓴다. request batch row reorder와 pool slot identity를 분리하고 cancellation/resume에서 generation/reset을 확인한다.

consumers는 full attention metadata, SWA attention/mask metadata와 recurrent state views다. 하나의 flattened row index를 세 addresses로 그대로 쓰지 않고 request-to-pool maps를 따른다. metadata H2D copy와 fused kernels가 current stream에서 실행되는지, cache ownership publish가 device completion 뒤인지 caller/result path를 왕복한다.

wrong output이 SWA boundary에서만 나타나면 full radix hit 가설보다 SWA slot/mask/compatible length를 본다. recurrent layer부터 갈리면 sequence pool generation을 본다. cache hit를 끄고도 재현되면 radix reuse보다 active update path가 강한 후보다.

**position 19가 Transformers layer cache object를 통과한다**

caller는 model config의 layer types로 each model layer에 full, dynamic sliding 또는 recurrent/linear state cache object를 배치한다. fixture의 model layer order와 cache object list가 같은 identity mapping을 유지해야 한다. constructor selection이 update/address semantics를 결정한다.

position 19 forward에서 attention layer는 K/V와 cache positions를 update API에 전달한다. full object는 sequence dimension append/static address write를 하고, sliding object는 warmup인지 W-filled 상태인지 판단해 bounded storage를 rotate, slice 또는 indexed write한다. exact mutation은 source branch를 읽으며 ring 비유를 구현 사실로 바꾸지 않는다.

sliding consumer는 update 뒤 returned K/V view와 causal/window mask를 즉시 attention에 사용한다. physical storage order가 chronological order와 다르면 object가 reorder/view metadata를 제공해야 한다. position 19에서 slots stored `[16,17,18,19,12,13,14,15]`를 그대로 읽지 않게 하는 branch를 찾는다.

recurrent/linear layer object는 token-indexed K/V tuple 대신 architecture state tensors를 갱신한다. batch compaction이나 request removal에서 state row가 request identity와 함께 이동하는지 continuous manager/model path를 잇는다. classic invocation batch와 service slot lifetime을 같은 것으로 설명하지 않는다.

offload predicate가 non-sliding layers만 CPU로 옮기면 position 19 compute 전 full layer buffer를 prefetch stream에서 device로 복사하고 compute/default stream이 completion event를 기다린다. 특정 layer hang은 correct layer event가 record됐는지, consumer가 wrong generation/event를 기다리는지 역추적한다.

memory cost는 classic Dynamic/Static cache와 continuous paged cache가 다르다. layer object source가 보여 주는 bounded update와 offload 사실을 paged pool allocation에 그대로 일반화하지 않는다. 역방향 debug는 first wrong model layer의 object type, cache position, update indices와 returned view로 올라간다.

**position 19가 llama.cpp memory factory와 cells를 통과한다**

caller는 model architecture/context로 plain KV, recurrent, hybrid 또는 hybrid-ISWA memory implementation을 선택한다. wrong factory branch는 bytes뿐 아니라 cell eligibility와 update semantics를 바꾼다. MTP context 예외 predicate도 일반 hybrid wrapper와 같은지 확인한다.

plain KV component는 cell capacity와 sequence ownership metadata를 갖는다. batch item의 sequence ID와 position 19가 eligible cell을 찾고 per-layer K/V tensor cell axis에 write한다. SWA mask/eligibility는 old cells 가운데 current query가 볼 positions와 reusable cells를 판정한다.

old position 11 cell을 position 19가 재사용할 수 있어도 sequence identity와 absolute position condition이 맞아야 한다. cell number만 전달하면 wrong tenant/position state를 읽는다. mutation은 cell position, sequence membership과 K/V write며 graph execution completion 전 host ownership을 새 request에 주지 않는다.

hybrid wrapper는 full/SWA/recurrent components의 address/update calls를 같은 external batch position에서 호출한다. component order와 model layer mapping을 factory construction에서 graph/model forward consumer까지 잇는다. recurrent component는 sequence state slot을 update하며 reuse 전 clear/restore가 필요하다.

backend graph가 K/V/state operations를 구성하면 execution scheduling과 backend buffer ownership이 happens-before를 만든다. GPU/CPU split이면 copy nodes/events도 lifetime의 일부다. output에서 역추적할 때 factory type, SWA eligibility, cell stored position/sequence와 recurrent generation을 차례로 본다.

네 구현의 object topology는 다르다. 그러나 caller가 layer/cache type을 선택하고, predicate가 position 19의 legal address를 정하며, mutation이 state/generation을 publish하고, model consumer가 visibility/mask와 함께 읽고, stream/event가 순서를 지킨다는 사슬은 같다. source 구조만으로 성능 순위를 단정하지 않는다.

**고정 source를 갱신할 때 보존할 의미**

version upgrade에서 line만 새 위치로 옮기지 않는다. vLLM은 spec subtype, window/in-flight predicate, group tuple mapping과 disabled fallback 의미를 본다. SGLang은 full/SWA ownership, allocator mapping과 recurrent generation을 본다. Transformers는 layer object factory, bounded update와 offload event ordering을 본다. llama.cpp는 memory factory, cell eligibility/SWA mask와 graph completion을 본다.

symbol 이름이 바뀌어도 input과 mutation이 같으면 새 commit에 맞춰 갱신한다. 이름은 같지만 ring이 copy/roll로 바뀌면 physical address와 비용 설명을 수정한다. option 이름이 유지돼도 fallback promotion semantics가 달라질 수 있다.

spec code는 memory contract를 증명하지만 CUDA event ordering까지 증명하지 않는다. layer factory는 selected object를 증명하지만 latency를 증명하지 않는다. mask predicate는 visibility를 증명하지만 output parity 측정은 아니다. 각 주장에 맞는 source 범위를 둔다.

review artifact에는 position 19, W=8, old 11, generation G7을 function inputs/fields에 대응시키고 expected mutation을 적는다. recurrent에는 sequence slot/version, offload에는 layer identity/event를 대응한다. generic class를 실제 branch에 대입한다.

관측 field가 expose되지 않으면 존재한다고 쓰지 않는다. 필요한 debug hook owner와 privacy/cardinality 제약을 제안한다. 이 원칙은 source 인용을 incident first divergence를 찾는 반증 도구로 만든다.

## 36.6 physical address를 관측 가능한 record로 만든다

iteration record에는 request generation, absolute position, layer group/type, logical cache length/window, physical block/ring/sequence slot, slot generation, writer/reader stream event와 commit epoch를 둔다. raw pointer를 metric label로 넣지 않고 sampled trace/debug record에 둔다.

12-layer fixture에서 position 19 event는 full groups의 append address, sliding groups의 ring slot 3 old/new absolute metadata, recurrent groups의 sequence slot state version을 동시에 남긴다. output 오답이 생기면 first divergent group을 찾는다.

memory 관측은 full pages, sliding retained/in-flight capacity, recurrent state bytes와 group padding을 분리한다. hybrid manager off/on 비교에서 logical layer pattern, spec selection, bytes/page와 total reserved blocks를 맞춘다.

### position 19 trace를 세 줄로 맞추기

첫 줄은 logical work다. request generation G7, query absolute 19, cache progress before 19와 layer pattern을 쓴다. 둘째 줄은 physical addresses다. full group block/offset, sliding group slot/old absolute/new absolute/slot version, recurrent group sequence slot/old state version을 쓴다. 셋째 줄은 execution ordering이다. plan epoch, producer stream, completion event, metadata publish와 next consumer wait를 둔다.

정상 record의 full entry는 `abs=19→block27/off3`, sliding은 `abs=19→slot3(old=11,newVersion=20)`, recurrent는 `seqSlot5 state18→19`처럼 읽힌다. exact IDs는 구현마다 다르지만 three-address distinction이 남는다. layer group mapping hash/version도 함께 둔다.

sliding reader가 wrong output을 냈다면 writer trace와 reader trace를 조인한다. reader expected abs range, physical slots, each stored absolute와 version을 비교한다. slot 3 stored 19인데 reader가 11로 해석하면 metadata consumer, stored 11인데 position 19로 해석하면 publish/write ordering, slot 3 자체가 다른 request면 generation ownership이다.

recurrent trace는 huge state values를 전부 log하지 않고 state generation/version, owner sequence, checksum 또는 sampled norm을 남길 수 있다. tenant mixing에서는 owner generation mismatch가 value anomaly보다 먼저 나타나야 한다. raw user data는 metric label에 넣지 않는다.

### memory record를 세 component로 reconcile하기

full attention position payload를 B라 하면 fixture logical full은 `4×20×B=80B`다. sliding steady retained는 `4×8×B=32B`다. recurrent one-sequence state bytes가 R이면 `4R`이다. logical hybrid baseline은 `112B+4R`이며 B와 R units가 bytes라 합산 가능하다.

physical reservation은 block rounding, sliding in-flight maximum, group common page size와 recurrent alignment를 더한다. full/sliding specs가 same page group에 묶여 maximum page capacity를 쓰면 `112B`보다 slack가 생긴다. recurrent state가 attention page size에 padding되면 `4R` 기대가 크게 늘 수 있다.

hybrid manager off에서 sliding four layers가 full length로 승격되면 attention logical-capacity direction은 `160B`다. on/off metrics에서 model layer pattern은 같고 spec types/bytes/page 또는 required blocks가 바뀌어야 한다. weights bytes 변화나 request length 분포 변화가 없음을 함께 확인한다.

process RSS는 이 component sum에 graph pools, workspace, weights와 allocator reservation을 더한 값이다. hybrid cache metric과 nvidia-smi delta를 바로 같게 놓지 않는다. group tensor bytes 또는 backend buffer bytes가 reconciliation anchor다.

### boundary-focused observation이 필요한 이유

ring 오류는 positions 0…7 warmup에서는 재현되지 않는다. first reuse position 8, next cycle 16, specific old/new pair 11→19를 포함해야 한다. random short prompt test가 모두 통과해도 boundary bug는 남는다.

window size를 8에서 16으로 바꿨더니 오류 시작점이 8→16으로 이동하면 modulo/overwrite 가설이 강해진다. layer outputs가 every W boundary에서 갈리고 full layers는 parity를 유지하면 model weights나 general RoPE 가설은 약해진다.

chunk size를 바꿔 오류가 이동하면 in-flight overwrite eligibility나 metadata layout을 본다. same W라도 prefill chunk 12와 decode one-token path가 different kernel/mask를 쓸 수 있다. path label과 query span을 trace에 둔다.

### stream event를 timestamp로만 추측하지 않기

host timestamps에서 metadata publish가 device write completion보다 빨라 보여도 asynchronous APIs의 return time은 completion time이 아니다. CUDA event record/wait relation과 stream sequence를 본다. same stream operations는 submission order를 갖지만 host metadata가 other thread/stream consumer에 보이는 계약은 별도다.

event ID/generation을 sampled trace에 넣으면 producer position 19 event E19를 consumer iteration 20이 기다렸는지 확인할 수 있다. wrong event reuse는 event object pointer가 같아도 generation이 다를 수 있다. request/slot reuse와 같은 ABA 위험이다.

global synchronize를 추가해 오류가 사라지면 ordering 가설이 강해지지만 최종 복구로 곧장 채택하지 않는다. 어떤 missing edge가 있었는지 좁히고 per-group/event dependency로 복구해 throughput 비용을 제한한다. sync로도 유지되면 mapping/generation logic을 본다.

## 36.7 다섯 장애를 first divergence에서 닫는다

window boundary 오답은 position 7→8, 15→16, 11→19 slot reuse에서만 나타날 수 있다. value, absolute metadata, mask visibility와 stream ordering 중 처음 어긋나는 지점을 찾는다. 일반 numerical instability는 modulo boundary와 정확히 일치하면 약한 가설이다.

### 사건 1: slot 3에서만 finite wrong answer가 나온다

증상은 NaN이나 crash가 아니라 특정 long prompt의 token 19부터 greedy output이 reference와 갈리는 것이다. positions 0…18의 selected tokens와 full-layer checkpoints는 맞다. sliding layer 5 attention output이 position 19에서 처음 다르다.

경쟁 가설은 RoPE absolute position, mask off-by-one, slot metadata race와 K/V overwrite ordering이다. writer record는 slot 3 K/V checksum이 new 19와 맞지만 stored absolute tag가 11이다. first divergence는 K/V write 뒤 tag publish 전 reader가 접근한 epoch다. general RoPE 가설은 input position IDs가 correct하고 full layers가 맞아 약해진다.

복구는 K/V와 tag를 무조건 atomic instruction 하나로 만들라는 뜻이 아니다. producer stream completion 뒤 tag/valid generation을 publish하고 reader가 committed generation만 선택하도록 한다. same iteration reader라면 kernel 내부 ordering/mask contract를 조정한다. W boundaries 8,16,24와 varied chunk sizes로 회귀한다.

hybrid manager disabled 뒤 OOM은 sliding spec이 full spec으로 승격돼 W=8 대신 full maximum length를 reserve했는지 본다. model weights나 traffic 증가 가설은 effective cache specs/page bytes 변화로 반증한다.

### 사건 2: 안전 옵션을 껐더니 시작 단계에서 OOM이 난다

사용자는 hybrid manager가 복잡해 보여 disable했다. traffic가 들어오기 전 cache initialization/profile에서 OOM이 난다. model length 32k, W=8인데 sliding four layers도 full 32k capacity spec으로 승격됐다면 reservation 증가가 매우 크다.

startup finalized spec list를 on/off로 비교한다. layer types는 같지만 `SlidingWindowSpec→FullAttentionSpec` 변화와 maximum memory/page calculation이 first divergence다. weights와 runtime request count는 allocation 전 동일하므로 반증된다.

복구는 memory fraction만 낮추는 것이 아니다. hybrid manager를 다시 켜고 supported grouping/path correctness를 검증하거나, fallback의 capacity requirement를 받아 block count/concurrency를 줄인다. disable option의 semantic effect를 “optimization off”가 아니라 address/reservation model change로 문서화한다.

resume 뒤 recurrent state가 다른 tenant와 섞이면 sequence slot ID뿐 아니라 generation/reset completion을 본다. old state free와 new assignment 사이 ordering이 first divergence다. full KV block table이 정상이어도 recurrent owner는 별도다.

### 사건 3: resume 첫 token에 다른 대화 흔적이 섞인다

request A가 preempt돼 sequence slot 5 state를 host에 저장했다. slot 5를 B가 사용한 뒤 A가 resume한다. resume 첫 recurrent layer output만 reference와 다르고 full/sliding state restore는 맞다. old snapshot copy, B reset과 A generation assignment 순서를 본다.

trace에서 slot owner가 A generation 9로 publish됐지만 async restore copy가 B generation 8 state buffer를 source로 사용했다. first divergence는 restore descriptor generation이다. tokenizer/prompt 가설은 resume 전 outputs와 full-layer cache parity로 반증된다.

복구는 `(sequence_slot,generation,state_version)`을 snapshot/copy descriptor와 destination validation에 붙인다. restore event completion 뒤 running membership을 publish한다. cancellation/slot reuse/resume 순서를 교차해 wrong generation copy가 submit 전에 실패하는지 본다.

cache byte metric 불일치는 layer-group padding, common page size, Mamba state replication과 attention payload를 분리한다. 하나의 token-bytes formula를 모든 groups에 곱하지 않는다.

### 사건 4: metric이 112B+4R과 맞지 않는다

fixture logical calculation은 full `80B`, sliding `32B`, recurrent `4R`인데 backend cache tensor bytes가 1.6배다. 첫 가설은 leak이지만 startup 직후 request 0에서도 같은 fixed delta가 있다. layer group specs와 allocated tensor shapes를 펼친다.

full/sliding page group이 common maximum stride로 padding되고 Mamba state가 TP ranks에 replicate됐다. logical component 합과 group physical capacity 사이 first divergence가 spec grouping/allocation에서 나타난다. free/request lifecycle 전부터 존재하므로 leak 가설은 탈락한다.

복구가 반드시 grouping을 제거하는 것은 아니다. common tuple packing이 kernel/allocator 단순성과 capacity fungibility를 제공할 수 있다. metric 이름을 logical component bytes와 physical group tensor bytes로 나누고, excessive padding을 줄이는 변경은 performance/correctness를 따로 검증한다.

다른 mismatch에서는 hybrid manager off가 sliding을 full로 승격했거나 metric이 block capacity를 logical retained positions로 표시할 수 있다. spec type, bytes/page, total blocks와 tensor numel을 단계적으로 reconcile한다.

offload 뒤 특정 layer hang은 non-sliding selection, copy/default stream event, prefetch completion과 next compute wait를 잇는다. memory 절감이 보인다고 synchronization correctness가 증명되지는 않는다.

### 사건 5: offload 뒤 layer 8에서만 forward가 멈춘다

Transformers offload를 켜자 GPU memory는 줄었지만 hybrid model이 항상 layer 8 진입에서 hang한다. CPU-GPU bandwidth 부족이라면 느려져도 event completion은 진행해야 한다. trace에서는 compute stream이 prefetch event E8을 기다리지만 E8 record operation이 제출되지 않았다.

layer type mapping에서 index 8 sliding layer는 offload 제외 대상인데 prefetch consumer는 old logical index table로 non-sliding으로 분류했다. producer는 copy를 skip하고 consumer는 wait했다. first divergence는 layer-selection predicate tables가 서로 다른 mapping을 쓴 지점이다.

PCIe/NVLink 성능 가설은 copy operation 자체가 없고 same layer에서 deterministic hang해 탈락한다. 복구는 producer/consumer가 같은 cache-layer object identity로 offload eligibility를 결정하고 event를 generation/layer와 묶는 것이다. no-copy branch는 wait도 만들지 않는다.

회귀는 full→sliding, sliding→recurrent처럼 type boundary layers를 포함한다. offload bytes 감소뿐 아니라 event record/wait pairing, returned cache view parity와 cancellation cleanup을 본다. global synchronize로 hang을 숨기지 않는다.

### 다섯 사건이 가리키는 공통 경계

사건 1은 sliding slot의 value/tag publish, 사건 2는 spec selection이 reservation model을 바꾸는 경계, 사건 3은 recurrent slot generation/restore, 사건 4는 logical component와 physical group stride, 사건 5는 offload layer selection과 stream event가 갈렸다.

모두 hybrid cache 총 bytes 하나만 보면 찾기 어렵다. request generation, layer type/spec group, absolute/state version, physical unit과 stream event를 유지하면 first divergence 전까지 정상인 components를 제외할 수 있다.

fault injection은 runtime 실행 없이 source call order table로도 설계할 수 있다. metadata publish를 write completion 전으로 가정하고 reader 가능 window를 표시하거나, wrong generation restore descriptor와 unmatched offload event를 constructed state로 검토한다. 실제 성능/재현 수치는 실행 전까지 주장하지 않는다.

실전 triage에서는 first bad absolute position을 먼저 찾는다. 8의 배수 경계면 sliding reuse, resume 직후면 recurrent restore, hybrid option 변경 직후 startup OOM이면 spec promotion, offload 특정 type boundary면 event selection을 우선한다. 이 symptom index는 원인을 확정하지 않고 source walk 시작점을 줄인다.

그다음 layer checkpoint를 full/sliding/recurrent groups로 나눈다. first bad layer가 나오기 전 cache groups는 반증되고, 그 layer의 address tuple과 producer event만 깊게 본다. final text 차이만으로 모든 12 layers를 동시에 의심하지 않는다.

복구 diff는 state invariant와 비용을 함께 적는다. generation/tag ordering을 강화했는지, extra event 또는 copy가 latency에 무엇을 더하는지, spec fallback을 되돌려 reservation이 어떻게 변하는지 밝힌다. correctness fix가 global synchronization으로 throughput을 불필요하게 잃지 않는지도 source call graph에서 검토한다.

관측이 부족하면 모른다고 남기고 존재하지 않는 metric이나 수치를 만들어 빈칸을 채우지 않는다. 대신 필요한 field, snapshot epoch와 수집 경계를 source owner에 연결한다. runtime 전 정적 분석은 expected transition과 반증 지점을 제공한다.

### 사건 1을 reader와 writer 양쪽에서 닫기

window-boundary 오답의 사용자 증상은 특정 context length부터 답이 의미상 어긋나는 것이다. 서버 error, NaN과 OOM은 없다. greedy decode에서도 position 19 이후 divergence가 반복되고 W를 16으로 바꾸면 first bad position도 뒤로 이동한다. 이 두 관측은 numerical noise보다 bounded-address reuse를 가리킨다.

writer 쪽 trace는 plan epoch, position 19, selected slot 3, previous occupant 11, request generation과 producer event를 가진다. K/V checksum 또는 sampled vector는 fp16 reference의 position 19와 맞는다. metadata tag가 old absolute 11이면 first bad mutation은 tag publish다. tag도 19라면 reader 쪽으로 간다.

reader trace는 query absolute, visible logical range, gathered physical slots와 each stored absolute를 가진다. slot 3 tag가 19인데 chronological gather가 slot 0부터 시작해 order가 `[16,17,18,19,12,13,14,15]`라면 view/reorder predicate가 first divergence다. gather order는 맞지만 mask가 11을 visible로 남기면 SWA mask boundary다.

RoPE off-by-one은 경쟁 가설이다. position IDs와 pre-rotation Q/K checkpoint를 비교하고 full layers의 same position output이 맞으면 general RoPE input은 약해진다. 그러나 sliding backend가 stored key positions를 별도 처리한다면 tag 오류가 RoPE effect로 나타날 수 있으므로 source consumer를 본다.

복구 검증은 one golden output만 쓰지 않는다. W=8에서 positions 7→8,11→19,15→16,23→24, W=16 boundary, prefill chunks 1/7/12와 decode path를 포함한다. same slot reuse가 different request generation에서 일어나는 fixture도 둔다. group별 layer output, tag/order와 final logits를 확인한다.

performance 비용도 기록한다. tag validation, gather/reorder 또는 event wait가 추가되면 launch, metadata bandwidth와 sync gap이 변한다. correctness를 위해 global device sync를 넣어 고친 뒤 exact dependency로 좁힐 여지가 있는지 본다.

### 사건 2의 option 사슬을 끝까지 따라가기

사용자가 hybrid manager disable flag를 켠 이유는 기능을 단순화하기 위해서였다. 입력 문자열/constructor field가 validated config에 들어가고, grouping utility의 branch가 hybrid management를 건너뛴다. 이 branch가 sliding specs를 full specs로 replace/promote하는 mutation까지 이어지는지 확인한다.

effective state record에는 requested disabled, finalized disabled, original layer pattern full4/sliding4/recurrent4, selected cache specs와 group page sizes를 둔다. startup OOM은 request scheduling 전이므로 traffic, prompt lengths와 prefix cache를 경쟁 가설에서 빠르게 제외할 수 있다.

W=8, maximum model length 32,768이라면 sliding retained capacity가 full로 승격될 때 단순 layer-position 비율은 4,096배 방향이다. 실제 spec은 blocks, max batched tokens와 common grouping으로 달라지지만 변화 규모가 크다. weight memory가 동일해도 cache arena profiling 단계에서 OOM이 가능하다.

first divergence는 flag parser가 아니다. finalized flag까지 의도대로다. spec factory/grouping branch가 `SlidingWindowSpec`을 `FullAttentionSpec`으로 바꾸는 mutation이 physical reservation 증가의 원인이다. “option이 먹지 않았다”가 아니라 option semantic을 오해한 것이다.

복구 A는 hybrid manager를 다시 활성화하고 position boundary correctness와 supported model pattern을 검증하는 것이다. 복구 B는 disabled fallback을 유지하되 cache block count/memory fraction/concurrency를 줄여 full reservation을 수용한다. B는 sliding memory 절감을 포기하는 명시적 tradeoff다.

검증은 process가 뜨는 것만 보지 않는다. selected specs, group bytes/page, total blocks/tensor bytes, W-boundary parity와 recurrent state mapping을 확인한다. enabled manager가 wrong grouping을 만들면 OOM은 사라져도 correctness가 깨질 수 있다.

### 사건 3에서 tenant mixing을 반증하는 순서

resume 첫 token에 다른 문맥 흔적이 보이면 data privacy incident로 다뤄야 한다. 먼저 output symptom만으로 tenant mixing을 확정하지 않고 request generation과 recurrent state owner를 확인한다. full KV blocks, sliding slots와 recurrent state 가운데 first mismatch를 찾는다.

A generation 9가 preempt될 때 recurrent state version 18 snapshot을 만든다. B generation 10이 same sequence slot 5를 사용한 뒤 A가 resume한다. expected transition은 B last writer completion→slot released/reset boundary→A restore copy version18→restore event complete→A running publish→token19 update다.

관측에서 A running publish가 restore event보다 먼저라면 consumer가 incomplete/old state를 읽는다. restore source descriptor가 generation 10이면 B state를 복사한다. destination owner tag가 generation 9인데 contents가 B라면 generation validation이 metadata만 확인하고 copy source를 확인하지 않은 것이다.

sampling nondeterminism 가설은 greedy mode와 first recurrent layer checkpoint divergence로 약해진다. tokenizer/prompt 오류는 resume 전 prefix outputs와 input IDs parity로 반증한다. full cache corruption은 full layers/checksums가 맞아 탈락한다.

source predicate는 free sequence slot selection, snapshot lookup key, restore enqueue와 running transition이다. mutation은 destination state tensors, owner generation과 version이다. consumer는 recurrent forward state slice다. stream boundary는 restore copy event와 model compute wait다.

복구는 snapshot key와 transfer descriptor에 request generation/model-state version을 포함하고 destination generation을 submit/completion 양쪽에서 검사한다. old request cancellation이 late callback을 보내도 new generation을 mutate하지 못한다. restore failure는 running publish 없이 cleanup/rollback으로 간다.

회귀는 A preempt→B reuse→A resume, A cancellation during restore, duplicate completion, slot ID same/generation different와 TP/PP worker restore를 포함한다. state checksum만 아니라 owner/version invariant와 output parity를 본다.

### 사건 4의 denominator를 한 줄씩 벗긴다

metric이 formula보다 크다고 할 때 formula denominator는 12 layers×20 positions가 아니다. full logical bytes는 4×20×B, sliding retained logical bytes는 4×min(20,W)×B, recurrent logical bytes는 4R이다. 첫 expected 합은 112B+4R이다.

allocator metric이 attention blocks capacity를 세면 full/sliding positions는 block rounding되고 in-flight sliding capacity가 추가된다. group common page size는 small spec을 max spec stride에 맞출 수 있다. recurrent state가 attention page tuple에 pack되면 padding 또는 replication이 생긴다.

backend tensor bytes는 total reserved units를 포함하며 request logical used와 다르다. process RSS는 graph pools/workspace/weights까지 포함한다. 네 denominator를 raw bytes로 같은 snapshot epoch에 놓는다. used attention positions metric에 recurrent active sequences를 token처럼 더하지 않는다.

first divergence가 logical→spec이면 layer type/window/state shape 가정이 틀렸다. spec→tensor이면 grouping/alignment/replication/extra buffer다. tensor→process면 workspace/allocator pools다. request terminal 뒤 used만 안 줄면 lifecycle/refcount이고 tensor reserved가 유지되는 것은 pool contract일 수 있다.

복구는 metric 값을 formula에 맞추도록 fudge factor를 넣는 것이 아니다. exporter 이름/unit을 logical retained, allocated capacity, group tensor reserved와 process reserved로 나눈다. layer group/type, device와 dtype을 bounded labels로 쓰고 request generation은 trace에 둔다.

fixture에서 hybrid manager on/off, W 8/16, request length 7/8/9/19/20과 recurrent active sequence 0/1/2를 바꿔 각 denominator의 expected direction을 확인한다. 실제 수치는 backend allocation source와 runtime 전에는 주장하지 않는다.

### 사건 5의 stream wait cycle을 그림 없이 복원하기

offload hang에서 compute stream C가 layer 8 buffer-ready event E8을 기다린다. copy stream P는 layer 8이 sliding이므로 offload/prefetch 대상이 아니라고 skip한다. C의 layer eligibility table이 layer 8을 full로 잘못 분류하면 E8은 never recorded다. wait cycle은 GPU utilization low와 forward stall로 보인다.

PCIe/NVLink bandwidth 부족은 copy가 제출돼 duration이 길어야 한다. trace에 copy node/event record 자체가 없으면 탈락한다. dead CUDA kernel 가설도 last submitted operation이 event wait이고 kernel launch가 없으면 약해진다. memory pressure 가설은 allocation success와 deterministic same-layer stall로 반증된다.

caller는 cache layer iteration과 prefetch scheduling이다. predicate는 non-sliding/offload eligible layer다. mutation은 buffer residency/active staging slot과 event record다. consumer는 next layer forward의 compute stream wait다. producer/consumer가 same layer object identity와 generation을 사용해야 한다.

복구는 shared eligibility source를 사용하고 no-copy branch가 wait를 만들지 않게 한다. event pool reuse에는 layer/buffer generation을 붙인다. cancellation/exception에서 recorded-but-unused event와 staging buffer를 cleanup하되 next request가 stale event를 ready로 오해하지 않게 한다.

검증은 type boundaries full→sliding, sliding→recurrent, recurrent→full, multiple offload buffers와 rapid request cancellation을 포함한다. event record/wait pair cardinality, layer identity, output parity와 actual memory reduction을 본다.

### option 변화는 address 모델 변화일 수 있다

hybrid manager, sliding window override, cache implementation selection, recurrent/offload enable은 단순 memory-size knobs가 아니다. validation 뒤 selected spec/object/factory가 바뀌고 address translation, update mutation과 stream dependencies가 달라질 수 있다.

옵션 기록은 requested value, finalized effective state, selected layer specs/objects, position 19 addresses, allocation bytes와 output/latency effect를 잇는다. config diff만 있고 spec/object diff가 없으면 no-op 가능성이 있다. spec이 바뀌었는데 metric만 그대로면 fixed arena가 capacity units를 바꿨는지 본다.

window 8→16은 logical visibility와 ring reuse boundary, retained capacity와 first-overwrite epoch를 바꾼다. hybrid disable은 sliding→full promotion, offload toggle은 residency와 stream event graph를 바꾼다. recurrent state dtype/placement는 state bytes와 copy/update path를 바꾼다.

성공 조건도 option마다 다르다. window change는 semantic/model contract가 허용해야 하고 boundary parity, hybrid enable은 grouping/address correctness와 OOM headroom, offload는 event pairing/output parity와 transfer stalls, recurrent placement는 generation isolation과 restore ordering을 본다.

**full metric의 분모**

full attention에서 request logical retained positions는 prompt+committed decode positions다. layer-positions는 full layer count를 곱한 값이고 payload bytes는 layer-specific K/V bytes를 곱한다. block allocator used capacity는 unique physical blocks×block slots이며 partial-tail slack를 포함한다.

prefix sharing이 있으면 logical request references와 unique full blocks가 다르다. active request length 합을 unique payload bytes로 읽지 않는다. full block refcount/retention lifecycle은 37장이 맡지만 metric 이름에는 logical refs와 unique units를 구분한다.

full tensor arena reserved는 free blocks까지 포함한다. request 종료 뒤 logical/used unique가 내려가도 arena는 유지될 수 있다. process memory와 비교할 때 weights/graph/workspace를 빼지 않으면 full cache leak을 오진한다.

**sliding metric의 분모**

sliding logical position은 model absolute progress와 retained visibility length가 다르다. request position 20이라고 retained 20은 아니다. steady W=8이면 recent bounded state지만 current chunk/in-flight contract, sink/global tokens와 alignment가 capacity를 늘릴 수 있다.

`ring_slots_used=8`도 eight unique historical positions가 valid하다는 뜻은 absolute tags/generation이 맞을 때뿐이다. valid retained count, physical capacity slots, overwrite count와 tag mismatch를 나눈다. overwrite counter 증가는 memory growth가 아니라 address reuse다.

sliding layer count를 full layer count와 합쳐 total cached tokens라 부르면 W effect를 잃는다. layer-group retained positions 또는 bytes를 합산하되 request logical absolute length는 별도다. hybrid manager promotion 뒤 sliding group 자체가 없어지고 full group capacity가 늘 수 있다.

**recurrent metric의 분모**

recurrent state는 active sequence slots, state/conv tensor bytes per slot와 replication을 곱한다. token progress 20을 state bytes에 곱하지 않는다. state version은 freshness/ordering metric이지 allocation count가 아니다.

one sequence slot에 conv and recurrent components가 있고 TP ranks에 replicate되면 cluster aggregate bytes와 rank-local bytes를 나눈다. parked/offloaded snapshot이 host와 device에 동시에 존재하면 transition 중 double residency가 있을 수 있다. active state, snapshot bytes와 transfer workspace를 구분한다.

owner generation mismatch, reset pending와 restore pending counts는 correctness/lifecycle 관측이다. state tensor bytes가 정상이어도 wrong generation이 내용을 소비할 수 있다. capacity metric만으로 tenant isolation을 증명하지 않는다.

**세 분모를 한 dashboard에서 읽는 법**

12-layer fixture snapshot은 request absolute progress 20, full logical layer-positions 80, sliding valid retained layer-positions 32, recurrent active state slots four라는 식으로 시작한다. physical full/sliding blocks/slots와 recurrent tensor bytes를 next row에 둔다. group tensors reserved와 process reserved는 그 아래다.

length가 20→21로 늘면 full logical은 four positions 증가하고 sliding valid retained는 steady 32일 수 있으며 recurrent allocation은 그대로, state version만 각 recurrent layer에서 전진한다. 이 delta signature가 three address models를 구분한다.

window 8→16이면 full delta는 없고 sliding retained/capacity upper가 늘며 first overwrite boundary가 이동한다. active sequences 1→2면 recurrent state bytes/slots가 늘지만 full/sliding lengths는 request별로 별도 변화한다. option/traffic intervention의 expected signature를 미리 적는다.

관측값이 signature와 다르면 first transition을 찾는다. sliding retained가 full처럼 계속 증가하면 spec promotion 또는 metric denominator, recurrent bytes가 tokens와 선형 증가하면 state accounting을 KV로 오해했거나 architecture가 실제 recurrent가 아닌 branch일 수 있다.

**source와 metric 사이의 공백을 정직하게 남기기**

source에서 page-size method와 tensor shape는 보이지만 exporter가 없을 수 있다. 그때 존재하지 않는 metric 이름을 책에 쓰지 않는다. 필요한 observation을 `layer-group spec type, physical units, absolute tag/version, producer event`처럼 field 수준으로 제안한다.

runtime을 실행하지 않았으므로 latency, memory saving percentage와 incident reproduction을 측정 사실로 쓰지 않는다. 112B→160B 같은 값은 stated fixture의 cost model이다. actual backend는 rounding/in-flight/grouping으로 달라질 수 있음을 adjacent 문장에 둔다.

고정 source가 보여 주는 fact와 inference도 나눈다. branch가 sliding spec을 full로 replace한다는 fact에서 reservation 증가 방향을 추론할 수 있다. exact OOM threshold는 available memory/block calculation과 workload/hardware 없이는 모른다.

### 운영 변경 전후를 같은 fixture로 비교하기

baseline은 W=8, hybrid manager on, offload off, request generation G7과 progress 20이다. selected specs는 full4/sliding4/recurrent4이고 position 19 addresses는 full append, sliding reusable slot과 recurrent state version이다. baseline이 있어야 option diff 의미를 읽는다.

hybrid manager off counterfactual은 architecture를 바꾸지 않지만 cache specs/capacity를 full-like로 바꿀 수 있다. expected signature는 sliding overwrite/reuse 감소 또는 소멸, reserved attention capacity 증가다. output semantics가 유지돼도 OOM/concurrency가 변한다.

offload on은 logical addresses와 result가 같아야 하지만 residency owner와 stream graph가 변한다. selected non-sliding buffers의 device residency 감소, copy/prefetch events 증가와 transfer wait가 expected signature다. exclusion predicate가 layer mapping과 맞아야 한다.

window 16은 model contract가 허용된다는 전제 아래 first overwrite를 8에서 16으로 옮기고 retained upper를 늘린다. position 19 slot 3의 old occupant가 W=16에서는 position 3일 수 있다. modulo는 같아도 visibility range가 다르다.

resume은 physical sequence slot이 같아도 generation/state version을 바꾼다. restored version 18에서 position 19 update가 이어져야 한다. full/sliding blocks가 correct해도 recurrent generation mismatch면 output이 깨진다.

각 diff는 option 문자열이 아니라 selected spec/object, address mutation, stream event, metric denominator와 correctness를 잇는다. expected signature가 변하지 않으면 no-op 또는 다른 branch다. 예상 밖 component가 변하면 option scope가 이해보다 넓다.

이 비교는 benchmark가 아니다. milliseconds나 GPU bytes를 발명하지 않고 source mutation이 어떤 observables를 움직여야 하는지 정한다. 이후 runtime 검증자가 same fixture로 inference를 확인하거나 반증한다.

upgrade review도 동일하다. option default가 같아도 spec grouping, layer object factory, SWA eligibility나 event selection predicate가 바뀌면 position 19 path가 달라진다. commit diff에서 caller inputs, selected object, mutation fields와 consumer expectations를 다시 채운다.

metric 이름이 유지돼도 denominator가 full blocks에서 hybrid tuple units로 바뀔 수 있다. exporter source와 unit description을 함께 고정한다. historical dashboard 비교에서 semantic discontinuity를 version boundary로 표시하지 않으면 가짜 memory regression을 만든다.

마지막으로 correctness fixture와 capacity fixture를 분리하되 같은 selected specs를 사용한다. boundary output parity는 address contract를, group bytes와 units는 reservation을 검증한다. 둘 중 하나만 통과하면 option rollout은 완료가 아니다.

## 36.8 한 request에 세 memory lifetime이 공존한다

좋은 hybrid 설명은 cache 총량 하나로 끝나지 않는다. full layer는 absolute history와 block address, sliding layer는 bounded visibility와 ring/page slot generation, recurrent layer는 sequence-owned state update를 가진다. 같은 request generation 아래 세 commit을 조정한다.

12-layer W=8 fixture에서 독자는 position 19가 full append, sliding slot 3 overwrite, recurrent state version advance로 갈라짐을 설명할 수 있어야 한다. physical slot이 같아도 absolute/generation metadata가 다르면 다른 address다.

37장에는 각 allocation의 refcount, eviction, rollback과 free ordering을 넘긴다. 이 장은 어떤 layer state가 어떤 address model과 commit boundary를 갖는지 닫는다.

### position 19로 네 stack을 같은 높이에서 비교하기

vLLM에서 position 19의 첫 결정은 cache spec과 group이다. full/sliding/Mamba layer specs가 page/state size와 grouping contract를 만들고 runner metadata가 각 backend consumer에 physical units를 전달한다. hybrid manager fallback은 sliding address model을 full capacity model로 바꿀 수 있다.

SGLang에서는 full/SWA radix ownership과 token/recurrent pools가 position 19을 나눠 가진다. full prefix node가 같은 token history를 가리켜도 SWA-compatible recent state와 recurrent sequence state는 별도 readiness를 가져야 한다. pool index mapping과 radix logical references를 구분한다.

Transformers에서는 layer cache object 선택이 position 19 update semantics를 정한다. full append, dynamic sliding update와 recurrent state layer가 model layer forward consumer에 직접 연결된다. offload option은 non-sliding object의 residency와 copy/compute stream dependency를 추가한다.

llama.cpp에서는 architecture-selected memory factory와 cells가 주소 소유권을 정한다. plain KV, recurrent, hybrid/hybrid-ISWA wrapper가 external batch position을 component-specific update로 내린다. cell eligibility와 SWA mask, sequence ownership과 backend graph completion이 commit boundary다.

공통점은 position 19이라는 logical work가 하나여도 physical state가 세 갈래라는 것이다. full component는 absolute-history append, sliding component는 bounded slot reuse와 visibility, recurrent component는 sequence state read-modify-write다. 모두 request generation과 layer mapping, commit epoch를 보존해야 한다.

차이는 이 invariant를 물질화하는 object다. spec/group tuple, radix/pools, cache layer objects, memory factory/cells가 서로 대응하지만 동일 자료구조는 아니다. 한 stack의 `slot`을 다른 stack의 `cell`과 숫자로 직접 비교하지 않고 caller→predicate→mutation→consumer→ordering을 비교한다.

비용도 object topology에 따라 다르다. group common page padding, radix/pool metadata, dynamic sliding rotate/copy, offload event, backend graph/cell scan과 recurrent fused update가 서로 다른 host/device 비용을 만든다. source 구조만으로 어느 stack이 빠르다고 결론내리지 않고 expected cost location만 표시한다.

### 증상에서 source branch로 내려가는 최종 workflow

첫 단계는 증상을 시간과 위치로 좁힌다. boundary-only 오답은 first bad absolute position과 W relation, OOM은 startup/profile인지 request growth인지, tenant mix는 resume/reset 직후인지, metric mismatch는 logical/allocated/reserved 어느 숫자인지, hang은 last submitted operation과 layer type을 기록한다.

둘째는 12-layer checkpoint를 full/sliding/recurrent groups로 나눈다. first bad layer 이전 groups는 반증한다. boundary token 19에서 first sliding layer가 갈리면 ring/tag/mask, first recurrent layer가 갈리면 state slot generation, 모든 layers 전부터 입력이 다르면 cache 밖을 본다.

셋째는 effective object를 고정한다. requested option 대신 finalized cache specs, layer objects, memory factory, hybrid/SWA/recurrent pool types와 offload eligibility를 기록한다. hybrid disabled OOM은 이 단계에서 sliding→full promotion이 보이고, wrong factory나 layer type mapping도 드러난다.

넷째는 position 19 address record를 만든다. full block/offset and stored absolute, sliding physical slot/old-new tags and version, recurrent sequence slot/owner generation/state version을 둔다. layer group mapping과 plan epoch가 모든 fields를 묶는다.

다섯째는 mutation ordering을 잇는다. producer input state, write/update operation, completion event, metadata/owner publish와 consumer wait를 순서대로 적는다. host API return timestamp를 CUDA completion으로 쓰지 않는다. same stream order인지 explicit event인지 caller source로 확인한다.

여섯째는 metric denominator를 맞춘다. full logical references/unique blocks/reserved arena, sliding absolute progress/valid retained/in-flight capacity, recurrent active sequence states/snapshots를 분리한다. process memory는 graph/workspace/weights가 더해진다는 것을 밝힌다.

일곱째는 competing hypothesis를 관측으로 지운다. W에 따라 first bad position이 이동하면 general numerical noise가 약하고, spec promotion이 request admission 전 보이면 traffic OOM 가설이 약하다. recurrent owner mismatch가 먼저면 tokenizer가 약하고, event record가 없는데 wait만 있으면 bandwidth 부족이 약하다.

마지막은 source predicate와 mutation에 first divergence를 고정한다. “sliding cache 문제”로 끝내지 않고 stored tag publish, chronological view, SWA eligibility, spec promotion, snapshot generation, group stride 또는 offload layer selection처럼 고칠 branch를 지목한다.

### 다섯 복구의 종료조건

boundary 오답 복구는 W=8/16 boundaries와 varied prefill chunks에서 K/V, tags, gathered order, mask와 layer outputs가 reference와 맞아야 한다. request generation reuse fixture에서도 late writer가 new owner를 mutate하지 않아야 한다. global sync 없이 exact dependency가 충분한지 확인한다.

hybrid disable OOM 복구는 selected specs와 bytes/page/total capacity가 intended state로 돌아오고 startup headroom이 확보돼야 한다. manager enable 뒤 position-boundary correctness와 recurrent mapping도 통과한다. 단순 memory fraction 감소로 process만 뜬 것을 semantic 복구라 하지 않는다.

recurrent tenant-mix 복구는 snapshot/restore descriptor, destination owner와 state version이 같은 request generation을 가져야 한다. restore event 전 running publish가 없어야 하고 cancellation, duplicate completion과 slot reuse에서 exactly-once cleanup을 보장한다. full/sliding parity와 recurrent layer checkpoint를 함께 본다.

metric mismatch 복구는 logical, allocated capacity, group tensor reserved와 process reserved가 raw bytes로 reconcile돼야 한다. exporter label/unit이 bounded하고 time snapshot이 일치해야 한다. 숫자를 맞추는 fudge factor가 아니라 spec/grouping/replication을 설명하는 component 합이어야 한다.

offload hang 복구는 each eligible layer copy event가 exactly one matching consumer wait를 갖고, excluded layer는 copy도 wait도 없어야 한다. layer type boundaries, event pool generation과 cancellation에서 dead wait/stale ready가 없어야 한다. output parity, memory reduction과 transfer wait cost를 같이 본다.

공통 종료조건은 세 가지다. 첫째, position 19의 three-address identity가 source와 trace에서 이어진다. 둘째, group별 commit과 stream happens-before가 next consumer 전에 닫힌다. 셋째, logical/physical/reserved memory와 correctness 결과가 same effective objects를 설명한다.

**36.8의 중간 회고.**

Hybrid cache가 어려운 이유는 cache 종류가 많아서가 아니다. 한 request와 한 token step이 서로 다른 lifetime과 address contract를 가진 memory components를 동시에 갱신하기 때문이다. full history는 append하고, sliding history는 overwrite/reorder하며, recurrent history는 state 안에 접힌다.

W=8 position 19는 이 차이를 압축한다. full layer에서는 새 absolute entry, sliding layer에서는 old 11이 있던 reusable slot 3, recurrent layer에서는 state version 18→19다. pointer나 slot number만으로 identity를 표현할 수 없고 request generation, layer group, absolute/state version과 commit event가 필요하다.

네 stack은 이 계약을 spec groups, radix/pools, cache layer objects와 memory factories/cells로 각각 구현한다. 독자는 이름을 암기하기보다 누가 object를 선택하고, 어떤 predicate가 address를 허용하고, 무엇을 mutate하며, 누가 읽고 어떤 stream edge가 순서를 지키는지 묻는다.

이 질문은 다섯 incident를 분리한다. boundary 오답은 slot/tag/mask, disable OOM은 spec promotion, tenant mix는 state generation/restore, byte mismatch는 group denominator, offload hang은 layer selection/event pairing에서 갈린다. 모두 hybrid cache라는 한 label 아래 있지만 first divergence는 다르다.

37장으로 넘기는 것은 추상적인 cache object가 아니다. full/sliding/recurrent component별 physical allocation unit, current owner generation, valid absolute/state version, references와 pending events가 붙은 state다. 다음 장은 이 units가 언제 refcount를 얻고 잃으며 eviction·rollback·free되는지 다룬다. 이 장은 주소와 commit이 안전하다는 전제까지 닫는다.

독자가 실제 codebase에서 이 장을 재현할 때는 position 19 하나만 고정해도 충분하다. layer factory/spec 선택, full block mapping, sliding old/new occupant, recurrent sequence state와 consumer event를 한 장에 연결한다. 이 작은 trace가 완성되지 않으면 더 큰 cache dashboard도 원인을 설명하지 못한다.

반대로 이 trace가 완성되면 model이나 window가 달라져도 질문은 유지된다. logical position은 어느 component 주소로 번역됐는가, old owner의 마지막 reader는 끝났는가, new generation은 언제 publish됐는가, model layer는 어떤 committed version을 읽었는가다. Hybrid cache의 복잡성은 이 네 질문으로 조사 가능한 형태가 된다.

## 36.9 sliding ring을 prefill부터 decode까지 손으로 걷는다

window `W=8`, sink tokens `S=2`, request generation `R7`, block abstraction 없이 token slots를 직접 보는 단순 fixture를 둔다. physical slots0~1은 sink positions0~1에 고정하고 slots2~7은 최근 6 tokens를 위한 ring이다. ring capacity `C=W-S=6`이다. position `p≥S`의 기본 slot은 `S+((p-S) mod C)`다.

prefill chunk `[0,5)`를 쓰면 positions0,1은 sink slots0,1이고 positions2,3,4는 ring slots2,3,4다. tags는 `[0,1,2,3,4,empty,empty,empty]`다. 다음 chunk `[5,10)`은 p5→slot5, p6→6, p7→7, p8→2, p9→3이다. p8은 old tag2를, p9는 old tag3을 overwrite한다.

position9을 계산할 때 attention-visible keys는 sink0,1과 recent positions4~9, 모두 8개다. physical chronological order는 slots0,1 다음 ring tags4,5,6,7,8,9이며 slots4,5,6,7,2,3에 흩어진다. raw memory order0,1,2,3,4,5,6,7을 그대로 읽으면 tags0,1,8,9,4,5,6,7이 되어 시간 순서가 깨진다.

mask가 logical positions를 기준으로 올바르더라도 gathered K/V rows가 wrong order면 finite wrong answer가 난다. 반대로 gather가 맞고 mask가 physical slot index를 position으로 해석해도 틀린다. address translation, tag validation, chronological gather와 mask coordinates를 각각 검사한다.

decode p10은 slot4를 재사용하고 old tag4를 덮는다. write-before-read 위험을 본다. p10 query가 keys5~9와 sinks를 읽은 뒤 p10 K/V를 쓰는지, write를 먼저 하더라도 old p4가 window 밖이므로 안전한지 kernel contract를 확인한다. attention variant가 current token을 포함하면 p10 write가 current key로 visible해야 한다.

prefill chunk가 `[8,12)`처럼 ring을 한 번에 여러 slots overwrite하면 element별 순서와 final tags를 계산한다. p8→2, p9→3, p10→4, p11→5다. 이전 chunk consumer가 slots2~5를 읽는 비동기 work를 끝냈는지 event로 fence한다. CPU가 progress를 11로 갱신했다는 사실은 device last reader 완료를 보장하지 않는다.

## 36.10 sink token은 단순히 window를 늘리는 것이 아니다

sink0~1은 sequence가 길어져도 유지되는 identity다. ring의 recent capacity는 W-S이므로 S를 늘리면 recent history는 줄어든다. W8에서 S0은 최근 8, S2는 sinks2+최근 6이다. “sink2를 추가했으니 총 10 tokens”가 아니다. 구현 option이 total window와 ring window 중 무엇을 뜻하는지 확인한다.

position19에서 S2/C6이면 slot은 `2+((19-2) mod6)=7`이다. visible recent는 14~19이고 sinks0,1이다. tags chronological은 0,1,14,15,16,17,18,19다. physical ring slots는 p14→4,15→5,16→6,17→7,18→2,19→3이라는 계산과 맞지 않는가를 다시 검산해야 한다.

식에 넣으면 p19→`2+(17 mod6)=7`이 아니라 `17 mod6=5`, slot7이 맞다. 위 mapping도 p14에서 `(12 mod6)=0` slot2여야 한다. 따라서 chronological physical slots는 2,3,4,5,6,7이다. 계산 중 발견한 모순을 숨기지 않고 formula에서 다시 시작한다. 이런 손검산이 source trace의 잘못된 offset을 잡는다.

sink cache가 별 tensor이고 ring만 slots0~5를 쓰는 구현도 있다. 이 경우 combined logical index와 physical buffer/offset pair를 둔다. 하나의 integer slot로 합치면 sink buffer slot1과 ring buffer slot1을 혼동한다. address identity는 component/storage ID와 offset을 포함한다.

sink tokens가 prefix sharing과 결합되면 request별 소유권도 본다. common sink K/V를 shared artifact로 참조하는지 request cache에 복사하는지, adapter/model identity가 key에 들어가는지 확인한다. request cancel이 shared sink storage를 free하지 않고 reference만 감소해야 한다.

## 36.11 layer별 hybrid state family와 address 표

12-layer 예제를 둔다. layers0~3은 full attention,4~7은 sliding W8/S2,8~11은 recurrent state다. position19에서 full layer address는 page1/offset3(block16), stored tag19다. sliding은 ring slot7과 tag19다. recurrent는 request sequence slot5, state version19다.

이 세 주소는 같은 integer19로 표현되지 않는다. full cache는 append-only logical page mapping, sliding은 overwrite 가능한 slot+absolute tag, recurrent는 fixed slot+monotonic version이다. layer factory가 layer index를 wrong family에 매핑하면 allocation도 맞고 tensor bounds도 유효한 채 의미만 틀릴 수 있다.

prefill `[0,16)` 뒤 full layers는 one full page, sliding layers는 sink2+recent14? 실제 W8이므로 sinks0,1과 recent10~15만 valid하다. recurrent layers는 16 steps를 접은 state version15 또는 processed-count16 convention을 갖는다. version이 last position인지 count인지 명시한다.

다음 prefill `[16,20)`에서 full은 page1 offsets0~3을 append한다. sliding은 p16→slot4,17→5,18→6,19→7을 overwrite한다. recurrent는 state를 네 번 순서대로 갱신하거나 fused scan으로 equivalent final version19를 만든다. 중간 recurrent outputs가 필요한 model이면 final state만 같다고 충분하지 않다.

decode p20은 full page1/offset4, sliding p20→slot2, recurrent version20이다. planner record에는 component group, logical interval, physical destination range, old occupant/version, new generation, write event와 consumer event를 둔다. group별 commit이 다르면 request-level progress를 가장 늦은 required component acknowledgment 뒤 publish한다.

hybrid state가 offload되면 address에 device tier와 buffer generation을 더한다. recurrent slot5가 CPU snapshot과 GPU live buffer를 동시에 가질 수 있다. restore 중 어느 copy가 canonical인지, running publish 전에 transfer event를 기다리는지 본다. pointer value 재사용만으로 owner를 판단하지 않는다.

## 36.12 logical position과 physical slot 혼동 사고

관측은 W8/S2 모델에서 position18부터 특정 layer output가 reference와 갈리고 text는 문법적으로 자연스럽지만 답이 틀리는 것이다. NaN, illegal access와 OOM은 없다. W16으로 바꾸면 first bad position이 34 부근으로 이동한다. tokenizer, sampling noise보다 sliding boundary가 강한 후보가 된다.

layer checkpoint는 full0~3은 reference와 같고 first sliding layer4에서 갈리며 recurrent layers는 이미 wrong input을 받는다. cache write K/V 값 자체는 position17까지 맞다. p18 write도 expected slot6에 있다. reader gather/mask branch로 범위를 줄인다.

source trace에서 writer는 `slot=S+((p-S)%C)`를 사용하지만 reader는 `slot=p%W`를 사용했다. p18은 writer slot6, reader slot2다. slot2에는 tag14가 있어 bounds와 dtype은 유효하다. reader가 absolute tag를 검사하지 않아 stale-but-valid K/V를 읽었다. first divergence는 reader address translation이다.

왜 p18에서 처음 드러났는지는 chunk/boundary와 sink offset 때문이다. 일부 이전 positions에서는 modulo 결과가 우연히 같거나 wrong row 영향이 작았다. W 변경과 S0 fixture로 first-bad relation을 검증한다. S0에서 writer/reader formulas가 같아 bug가 사라지면 sink-offset 누락 가설이 강해진다.

수정은 reader가 canonical address descriptor를 사용하고 slot tag가 expected absolute position인지 assert하도록 한다. chronological gather indices와 logical mask positions를 함께 생성한다. writer/reader가 formula를 각각 복제하지 않게 spec/helper를 공유하되 CUDA kernel의 실제 indexing도 동일한지 확인한다.

회귀는 W8/16, S0/1/2, prefill chunks1/5/8/9/16, positions boundary-1/boundary/boundary+1과 decode를 교차한다. expected slot, old/new tag, gathered chronological tags, mask와 layer output을 검증한다. full/recurrent neighbor groups가 변하지 않는지도 본다.

rollback은 sink-enabled layout generation admission을 fence하고 active requests를 drain한다. S0/S2는 ring capacity와 address mapping이 달라 handles를 그대로 이어 쓰지 않는다. live migration이면 logical tags를 읽어 new layout chronological order로 다시 써야 하고 parity를 검증한다. 지원하지 않으면 explicit drain/restart를 택한다.

## 36.13 그림과 source walk로 주소 계약을 고정한다

독자 그림은 세 줄이면 충분하다. 첫 줄은 absolute positions, 둘째는 physical slots, 셋째는 occupant tags다. W8/S2에서 p14~19를 쓴 뒤 다음처럼 읽는다.

```text
logical recent : 14 15 16 17 18 19
physical slots :  2  3  4  5  6  7
stored tags    : 14 15 16 17 18 19
sink           : slot 0→tag 0, slot 1→tag 1
```

p20을 쓰면 slot2 tag14가 20으로 바뀐다. visible chronological은 sinks0,1과 15~20이고 physical slots는 3,4,5,6,7,2다. raw slot order2~7은 20,15,16,17,18,19다. reader가 chronological permutation을 만들지 않으면 current p20이 recent history 맨 앞에 놓인다.

```text
logical recent : 15 16 17 18 19 20
physical slots :  3  4  5  6  7  2
raw slot order : 20 15 16 17 18 19
```

prefill `[14,21)`가 한 invocation에서 ring capacity6보다 큰 7 tokens라면 p14는 같은 invocation 안에서 p20에 덮인다. kernel이 모든 seven query outputs를 계산할 때 p14 K/V가 필요할 수 있다. 단순 final ring storage만 보고 in-place write하면 early query의 key가 사라진다. chunk를 capacity 이하로 제한하거나 staging/attention order가 old rows를 먼저 소비하도록 해야 한다.

이 사례는 final tags가 맞아도 intermediate outputs가 틀릴 수 있음을 보여 준다. source에서 prefill writer와 attention compute 순서, temporary staging과 supported maximum chunk를 확인한다. decode one-token fixture만으로 prefill correctness를 증명하지 않는다.

source walk 첫 카드는 layer/spec factory다. model layer index와 attention type, window/sink/recurrent parameters가 finalized cache component로 변하는 symbol을 찾는다. option/config 값이 validation과 fallback 뒤 실제 layer object에 들어가는지 본다. unsupported sink가 silently disabled되면 requested diagram이 실행과 다르다.

둘째 카드는 allocation/grouping이다. full block table, sliding ring/circular pool과 recurrent slots가 separate tensors인지 common grouped arena인지 기록한다. group offset/stride가 layer index에서 어떻게 계산되는지 본다. layer4가 sliding group row0인지 global row4인지 혼용하면 wrong component address를 읽을 수 있다.

셋째 카드는 scheduler/model input metadata producer다. request logical positions, block table, slot mapping, sliding window lengths, sink count와 state slot generation을 누가 만든다. prefill chunk와 decode에서 동일 helper를 쓰는지 branch가 다른지 확인한다. cached-prefix start가 modulo 계산의 origin을 잘못 바꾸지 않는지 본다.

넷째 카드는 CUDA/backend writer다. K/V row가 physical slot에 store되는 index expression과 tag/version publish 순서를 찾는다. tags가 host-only라면 device kernel이 stale row를 어떻게 피하는지 mask/mapping contract를 본다. asynchronous write completion event가 metadata publication 전에 signal되는지 확인한다.

다섯째 카드는 reader/gather/mask다. attention kernel이 physical ring을 직접 chronological하게 해석하는지 wrapper가 gathered view를 만드는지 확인한다. sink rows를 별 prefix로 붙이는지 same tensor offsets로 읽는지, expected absolute position tags를 검증하는지 본다. query position과 key logical positions가 mask에 전달되는 단위를 고정한다.

여섯째 카드는 recurrent update다. state slot lookup, old version read, update/scan, new version write와 publish를 잇는다. fused scan이면 intermediate outputs와 final state 모두 reference와 비교한다. restore/offload path가 같은 owner generation과 stream event를 쓰는지 본다.

일곱째 카드는 output/terminal이다. request progress가 어느 component acknowledgments 뒤 commit되는지, abort가 full blocks, ring owner와 recurrent slot을 각각 어떻게 닫는지 확인한다. 한 component cleanup 실패가 request ID reuse 뒤 다른 대화를 오염시키지 않게 generation을 증가시킨다.

vLLM source card는 hybrid cache specs/grouping, scheduler slot mapping, runner attention backend와 cache manager terminal을 연결한다. SGLang은 radix/token pool, schedule batch metadata, model runner/attention과 retract/free를 잇는다. Transformers cache layer object/update와 model forward consumer를, llama.cpp KV cells/slot positions와 CUDA backend view를 잇는다.

각 카드에는 revision, file/symbol/span, input unit, address expression, mutation, stream/event, expected tag/version, next consumer, failure rollback과 falsifier가 있다. “ring buffer를 쓴다”는 설명만으로는 p20 slot2를 재현할 수 없다. 숫자를 source expression에 대입할 수 있어야 한다.

## 36.14 관측·실패 주입·배포 terminal

관측 record의 primary key는 request generation, plan epoch, layer group과 absolute position이다. fields는 component kind, logical position/state version, physical storage/slot/offset, old/new tag, write event, reader event, commit status와 next owner다. request ID를 metric label로 내지 않고 sampled trace에 둔다.

metrics는 ring overwrites, tag mismatch, chronological gather failure, recurrent version mismatch, stale generation reject와 component cleanup gap counters를 둔다. gauges는 active full/ring/recurrent owners와 pending events, histograms는 overwrite-to-read fence, restore latency와 component commit lag다. window/sink/group/config generation은 bounded labels다.

boundary fixture는 W8/S2 positions7/8,13/14,19/20처럼 ring wrap 전후를 포함한다. sink count0/1/2와 prefill chunks1/5/6/7/8/16을 교차한다. chunk가 ring capacity보다 클 때 staged correctness 또는 explicit split을 검증한다. decode는 wrap을 여러 번 지나 request generation reuse까지 본다.

layer fixture는 full→sliding boundary layer3/4, sliding→recurrent7/8을 checkpoint한다. first bad layer로 group mapping을 찾는다. all layers output만 비교하면 later errors가 퍼져 first divergence를 잃는다. component-specific sentinel K/V/state와 reference attention/scan을 쓴다.

race injection은 old reader를 event barrier에서 멈추고 new position overwrite를 시도한다. expected는 overwrite가 wait하거나 separate generation storage를 쓰는 것이다. 반대로 new write를 멈추고 reader가 tag publication 전 접근하는 경로도 본다. host timestamp가 아니라 explicit CUDA/event dependency로 순서를 제어한다.

cancel injection은 prefill multi-token write 중, recurrent update 뒤 publish 전, restore copy 중과 output commit 뒤에 둔다. old generation late completion이 new owner tags/state를 바꾸지 않아야 한다. full/ring/recurrent resources가 exactly once terminal되고 shared sink/prefix는 reference policy를 따른다.

incident 검증은 observation→branch→cause를 같은 record로 닫는다. W에 따라 first bad position이 이동하고 first sliding layer에서 갈리며 writer slot/tag는 맞고 reader formula만 `p%W`라면 cause가 구체적이다. tokenizer나 numerical noise를 더 조사하지 않는다.

수정 후 parity는 final text만 보지 않는다. per-position gathered tag order, mask coordinates, K/V values, layer output와 recurrent version을 비교한다. output이 우연히 같아도 tag assertion이 틀리면 실패다. eager/graph, normal/overlap과 offload on/off를 교차한다.

성능도 본다. tag validation과 gather가 추가 synchronization/metadata copy를 만드는지 측정한다. canonical descriptor를 device-friendly form으로 precompute할 수 있지만 generation correctness를 유지한다. correctness를 위해 global synchronize를 넣은 임시 fix를 최종 최적화로 승인하지 않는다.

rollout 전에 effective layout fingerprint를 만든다. layer-family sequence, W/S, component storage IDs, group strides, dtype, address formula version과 kernel backend를 포함한다. old/new fingerprint가 다르면 active handle compatibility를 명시적으로 판정한다.

canary는 short pre-wrap만 쓰지 않고 여러 wrap long-context, sink-enabled, hybrid/recurrent, prefill chunk>capacity와 cancel/reuse를 포함한다. first wrap 이후에만 드러나는 bug를 startup smoke test가 놓치지 않게 한다. config generation별 tag mismatch와 output parity를 본다.

rollback은 new generation admission fence, inflight plan drain, full/ring/recurrent owner reconciliation, optional arena restart, self-check와 readiness 순서다. wrong-address output이 이미 client에 전달됐다면 내부 state rollback만으로 복구됐다고 쓰지 않는다. 명시적 client terminal/incident scope를 기록한다.

readiness는 model loaded뿐 아니라 layer factory/layout fingerprint, arena mapping, event loop, recurrent restore와 boundary self-test를 포함한다. p19/p20 fixture가 canonical tags와 layer outputs를 만들고 old generation pending이 0일 때 traffic을 연다.

독자 최종 산출물은 position19/20 address diagram, layer family table, pinned writer/reader source cards, race matrix와 rollback ledger다. 이 다섯 개가 있으면 finite wrong answer를 “sliding cache 이상”이 아니라 정확한 address/version branch로 좁힐 수 있다.

37장에 넘기는 physical unit은 storage pointer만이 아니다. component kind, owner generation, absolute tag/state version, chronological mapping, pending event와 terminal policy가 붙는다. eviction/refcount는 이 identity를 보존한 채 소유권만 바꿔야 한다.

**한 request를 prefill부터 네 decode까지 추적한다.**

R7은 prompt length18이고 chunks `[0,5)`, `[5,13)`, `[13,18)`로 실행된다. 첫 chunk 뒤 ring tags0,1,2,3,4이고 recurrent version은 4다. 둘째 chunk는 p5~12를 처리해 final visible sinks0,1과 recent7~12를 남긴다. 셋째는 p13~17을 처리해 recent12~17을 남긴다.

각 chunk의 plan record에는 planned interval과 previous committed prefix가 있다. chunk `[5,13)` 실행이 실패하면 scheduler progress를 13으로 publish하지 않는다. ring final tags가 일부 쓰였을 수 있으므로 generation을 폐기하거나 old snapshot에서 restore한다. recurrent state도 version4에서 12로 partial advance했으면 rollback/copy-on-write가 필요하다.

prefill completion 뒤 first decode position18은 slot6, 다음 19는 7, 20은 2, 21은 3이다. visible recent sets는 각각 13~18,14~19,15~20,16~21이다. sinks0,1은 고정이다. 네 steps에서 slot/tag/gather를 표로 남기면 wrap 두 번과 raw order rotation을 모두 본다.

full layer addresses는 block16에서 p18 page1/offset2,19/3,20/4,21/5다. recurrent state는 versions18~21로 전진한다. request progress commit은 full append, sliding tag publish와 recurrent update가 모두 해당 step generation에서 완료된 뒤 일어난다.

decode p20에서 cancellation을 넣는다. selected output가 client에 아직 전달되지 않았고 K/V/state update가 완료됐을 수 있다. 정책이 p20 commit-before-delivery인지 abort-discard인지 정한다. 어느 쪽이든 resume/new request가 half-committed components를 읽지 않는다. output cursor와 cache progress를 같은 의미로 가정하지 않는다.

slow consumer도 분리한다. client가 p19 뒤 읽지 않아도 scheduler/cache는 p20,p21을 계산할 수 있다. output queue backpressure가 generation reuse를 막거나 bounded buffering 뒤 abort해야 한다. ring overwrite safety는 client delivery가 아니라 model/cache reader lifetime과 연결된다.

**sink와 prefix cache가 만나는 사고.**

두 requests A/B가 동일 prompt sinks0,1을 공유하고 나머지는 다르다고 하자. cache key가 model/token IDs만 포함하고 adapter identity를 빠뜨리면 B가 A adapter로 계산된 sink K/V를 참조할 수 있다. physical slot과 tags는 맞고 ring boundary도 정상이라 finite semantic 오답만 난다.

first divergence는 address formula가 아니라 shared sink artifact identity다. A/B의 layer0 sink outputs부터 갈리고 non-shared run은 정상이다. adapter를 같게 하면 문제가 사라진다. sink sharing disable로도 사라진다. ring size 변경과 first bad position은 관계가 없어 앞 incident와 구분된다.

수정은 cache key에 model/adaptor/rope/cache dtype과 relevant execution generation을 포함하고 shared reference owner를 둔다. B cancel이 A artifact를 free하지 않고 reference만 줄인다. adapter unload/reload generation 뒤 old sink를 재사용하지 않는다.

fixture는 same/different adapter, same/different model revision, two tenants, cancel/evict/COW를 교차한다. shared hit 시 K/V parity와 refcount, miss 시 isolation을 검증한다. request ID나 tenant raw 값을 metric label로 노출하지 않는다.

이 사례는 모든 wrong answer를 modulo bug로 몰지 않게 한다. first bad layer/position relation, shared/non-shared neighbor와 source identity branch로 가설을 나눈다. 같은 physical slot correctness라도 semantic producer identity가 틀릴 수 있다.

**hybrid grouping stride 사고를 계산한다.**

full4, sliding4, recurrent4 layers가 있고 group-local row를 사용한다고 하자. sliding global layer4는 sliding group row0이다. metadata producer가 global index4를 그대로 group tensor stride에 넣으면 row4는 bounds 밖이거나 다음 allocation의 유효 row를 가리킬 수 있다. padded group capacity8이면 bounds 안이라 조용한 오답이 된다.

address는 `(group_id, local_layer, slot, component_offset)`이다. global layer에서 group/local로 변환하는 table을 cache spec factory가 제공한다. allocator, runner binding과 kernel wrapper가 같은 table generation을 써야 한다. `layer_idx % group_size` shortcut은 irregular pattern에서 틀린다.

예를 들어 layer sequence F,S,S,R,F,S,R,R처럼 불규칙하면 sliding global5의 local index2다. `5%4=1`은 wrong row다. regular four-layer groups fixture만으로 bug를 잡지 못한다. actual model layer pattern과 randomized family sequences를 property test에 넣는다.

first divergence는 layer4/5 checkpoint와 group mapping checksum으로 찾는다. slot tags는 expected positions인데 tensor base pointer/row가 wrong group/local이다. ring formula를 고쳐도 해결되지 않는다. source owner는 layer-to-cache mapping producer/consumer다.

수정 뒤 group tables, tensor offsets, kernel base pointers와 layer outputs를 검증한다. cache spec reorder/serialization, PP stage local layer numbering과 offload selected layers도 교차한다. global/local/PP indices 세 개를 이름으로 구분한다.

**운영 dashboard를 읽는 순서.**

첫 panel은 logical progress다. full committed prefix, sliding latest absolute/oldest retained, recurrent committed version을 request cohort별로 본다. 세 값이 같은 semantic step을 가리키는지 확인한다. planned progress와 committed를 섞지 않는다.

둘째 panel은 physical ownership이다. full used blocks, ring active owners/capacity/overwrites, recurrent active slots와 snapshots를 본다. arena reserved는 별 panel이다. request drain 뒤 used owners가 0이어도 arena capacity가 남는 것은 정상일 수 있다.

셋째 panel은 address assertions다. tag mismatch, wrong group/local mapping, stale generation, chronological gather invariant와 version mismatch counters를 본다. zero가 정상이다. sampling으로만 검사한다면 coverage와 first unchecked range를 표시한다.

넷째 panel은 synchronization이다. pending write/read/restore events, fence latency, event-pool generation mismatch와 dead waits다. overlap on/off와 offload cohort를 나눈다. host scheduler timestamp만으로 device happens-before를 단정하지 않는다.

다섯째 panel은 user effect다. boundary-position output parity sample, TTFT/ITL, abort/restore errors를 본다. address counter가 증가했는데 final text가 우연히 같아도 correctness failure로 처리한다. 반대로 latency만 나쁘고 address invariants가 맞으면 performance path를 조사한다.

incident 시작은 panel 순서와 반대일 수 있지만 first divergence를 찾을 때 logical→physical→address→sync→output으로 맞춘다. earliest unequal transition이 수정 owner다. downstream symptom이 가장 큰 곳을 먼저 고치지 않는다.

**최종 리뷰와 handoff.**

리뷰어는 W/S의 의미, ring capacity, modulo origin과 buffer identity를 묻는다. prefill chunk가 capacity보다 클 때 intermediate queries가 안전한지, reader가 chronological order와 logical mask를 어떻게 만드는지 묻는다. sink/prefix sharing identity와 group/local layer map도 확인한다.

recurrent state에는 version convention, fused scan intermediate output, snapshot/restore owner와 event를 묻는다. full/sliding/recurrent commit이 request progress에 합류하는 barrier를 확인한다. cancellation이 component별 partial mutation을 어떻게 폐기하는지 본다.

성능 옵션에는 effective layout/branch와 correctness fixture가 따라야 한다. hybrid manager disable, sliding/sink 변경, offload, graph/overlap은 memory나 speed만 바꾸지 않고 address object와 lifetime을 바꿀 수 있다. option→normalized spec→factory→consumer→effect→rollback을 잇는다.

승인 조건은 W8/16, S0/1/2, chunks와 wrap, irregular layer patterns, share/isolation, cancel/reuse, offload/overlap에서 address/tag/version과 layer outputs가 reference에 맞는 것이다. global synchronization 없이 exact fences로 통과하고 resources가 유한 시간 안에 닫혀야 한다.

최종 문장은 구체적으로 쓴다. “R7 p20에서 sliding writer·reader는 component ring, slot2, expected tag20과 generation G를 공유했고 chronological tags15~20 및 sink0~1, recurrent version20, full page1/offset4가 동일 commit epoch에 합류했다”처럼 재현 가능해야 한다.

이 문장을 source span과 trace에서 다시 만들 수 있으면 다음 장은 해당 physical units의 refcount/eviction을 안전하게 논의할 수 있다. 만들 수 없다면 pointer와 slot만 가진 불완전 identity이므로 먼저 이 장의 빈칸을 닫는다.

**배포 전 마지막 주소 worksheet.**

첫 행에는 model layer family pattern을 쓴다. layer index마다 full/sliding/recurrent/cross, group ID와 local index를 적는다. PP를 쓰면 global layer와 stage-local layer도 분리한다. factory가 만든 actual object class/spec fingerprint를 붙여 config 예상과 대조한다.

둘째 행에는 W, S, ring capacity와 modulo origin을 쓴다. p19와 p20의 expected component/slot/tag를 손계산한다. separate sink/ring buffers면 storage ID와 offset을 따로 쓴다. combined buffer면 reserved sink offsets가 modulo에 한 번만 더해지는지 본다.

셋째 행에는 prefill chunks를 쓴다. `[0,5)`, `[5,13)`, `[13,18)`처럼 실제 intervals와 각 chunk final tags를 적는다. chunk가 capacity를 넘으면 intermediate query가 overwritten rows를 필요로 하는지 표시하고 staging/split/source predicate를 붙인다.

넷째 행에는 decode p18~21의 full page/offset, sliding slot/tag/gather order와 recurrent versions를 나란히 쓴다. planned, written, acknowledged, committed timestamps/generations를 분리한다. client delivery는 별 cursor다.

다섯째 행에는 physical tensors를 쓴다. storage pointer generation, group/layer stride, dtype/component offsets, block table/slot mapping, tag/version arrays와 CUDA streams/events다. raw pointer는 재사용될 수 있으므로 owner generation 없이 identity로 쓰지 않는다.

여섯째 행에는 failure를 쓴다. writer delay, reader delay, cancellation, restore, offload와 old generation completion을 주입한다. expected wait/reject/cleanup을 component마다 적는다. 모든 경로가 하나의 global synchronize로 통과하는 것은 기능 확인일 뿐 최종 성능 설계가 아니다.

일곱째 행에는 terminal을 쓴다. output service terminal, full block refs, ring owner/tags, recurrent slot/snapshot, shared sink refs, events와 telemetry가 닫히는 시점을 기록한다. process arena가 남는 것은 request leak과 구분한다.

**wrong-answer 가설을 빠르게 줄이는 표.**

first bad position이 W/S boundary와 함께 움직이면 ring/tag/mask가 유력하다. first bad layer가 family boundary와 함께 움직이면 group mapping/factory가 유력하다. resume 첫 token만 틀리면 snapshot/owner generation이 유력하다. sharing에서만 틀리면 sink/prefix identity가 유력하다. offload에서 hang이면 event selection/pairing을 본다.

모든 positions/layers가 조금씩 다르면 tokenizer/model weights, numerical backend와 input preprocessing을 다시 고려한다. sliding이라는 이유만으로 cache를 범인으로 정하지 않는다. passing neighbor와 option toggles를 동일 inputs/generation에서 비교한다.

writer K/V와 tag가 이미 틀리면 input position, rotary coordinates와 store mapping으로 올라간다. writer는 맞고 reader gathered rows가 틀리면 address/chronology다. gathered rows는 맞고 attention output가 틀리면 mask/kernel numerical path다. layer output도 맞고 text만 틀리면 sampling/output mapping을 본다.

이 표는 관측 결과를 source owner에 연결한다. 각 가설에는 falsifier가 있고, falsified branch를 incident note에 남긴다. “cache를 초기화하니 해결”처럼 모든 state를 없애는 대응은 root cause를 증명하지 않는다.

**성능과 메모리까지 함께 승인한다.**

ring은 logical retained bytes를 full history보다 줄이지만 gather permutation, tags와 boundary handling 비용을 만든다. sink는 recent capacity를 줄이고 별 storage/identity를 추가한다. hybrid grouping은 arena 효율을 높이거나 padding을 만들 수 있다. recurrent state는 length-linear KV 대신 slot capacity를 가진다.

benchmark는 pre-wrap short context만 쓰지 않는다. multiple wrap, chunk>capacity, mixed layer pattern과 active request churn에서 epoch latency, gather/tag overhead, memory resident, TTFT/ITL을 측정한다. correctness fixtures를 같은 run generation에 붙인다.

optimization이 tag checks를 sampling으로 줄이면 unchecked coverage와 detection latency를 기록한다. debug mode의 full assertions와 production sampled trace가 동일 canonical formula를 써야 한다. 별도 debug implementation이 production bug를 재현하지 못할 수 있다.

memory dashboard는 full unique blocks, ring fixed capacity/owners, recurrent slots와 shared sinks를 component별로 보여 준다. logical progress와 physical capacity를 합친 하나의 utilization ratio를 만들지 않는다. process reserved에는 graph/workspace와 general allocator가 별 항으로 남는다.

최종 승인에는 source revision/layout fingerprint, p19/p20 worksheet, irregular layer mapping, race/cancel matrix, long-context parity, performance/memory와 rollback rehearsal이 있다. 이 artifact가 있어야 window/sink 또는 backend upgrade를 안전하게 diff할 수 있다.

**최종 손검산.**

W8/S2, p20을 다시 계산한다. recent capacity6, ring index `(20-2) mod6=0`, combined slot2다. overwrite되는 old tag는 14이고 write 뒤 expected tag20이다. visible chronological recent는 15~20, slots3,4,5,6,7,2다. sinks0,1을 앞에 붙여 logical key positions0,1,15,16,17,18,19,20을 만든다.

full group은 block16에서 page1/offset4이고 recurrent group은 state version20이다. layer family mapping이 F0~3/S4~7/R8~11이면 sliding global layer4는 group-local0, recurrent layer8도 local0이다. global index를 group stride에 직접 쓰지 않는다.

prefill이 `[14,21)` 한 chunk였다면 p14 state가 p20 write 전에 attention consumers에 필요했는지 확인한다. final tags만 맞는다고 성공이 아니다. source branch가 chunk split6+1을 만들거나 temporary staging/ordered kernel로 intermediate semantics를 보존해야 한다.

cancel이 p20 write 뒤 request progress commit 전에 오면 full page, ring tag와 recurrent state가 모두 generation G의 partial mutation이다. 구현 계약에 따라 G를 폐기하거나 component snapshots로 rollback한다. p20 output을 client에 commit하지 않고 p21/new owner가 G state를 읽지 않게 한다.

이 네 줄을 trace와 source에서 재현하면 address·ordering·terminal이 닫힌다. 하나라도 값이 다르면 first unequal row가 조사 시작점이다. 최종 text의 우연한 일치나 crash 부재는 주소 correctness를 대체하지 않는다.

운영 기록은 확정된 formula/source, effective branch에 의존하는 조건부 결론과 실행으로 확인해야 하는 timing 가설을 구분한다. 미검증 항목에는 fixture, metric, 담당 owner와 rollback trigger를 붙인다. 다음 조사자는 닫힌 줄을 반복하지 않고 위험한 빈칸부터 검증한다.

배포 뒤에도 boundary tag mismatch, stale generation, group-map checksum, recurrent version gap과 cleanup residue를 config generation별로 감시한다. 회귀가 보이면 새 admission을 즉시 fence하고 inflight component owners를 reconciliation한 뒤 검증된 layout으로 되돌린다. readiness는 p19/p20 self-test와 pending old generation0을 요구한다.

이 조건은 성능 개선보다 먼저 통과해야 하는 주소 안전성 gate다. 이후에만 gather overhead와 memory 절감을 평가한다.
