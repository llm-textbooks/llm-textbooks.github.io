# 34장. token 위치는 어떻게 실제 KV 주소가 되는가: page, block table, cell

요청 R의 일곱 번째 token이 어느 KV 주소에 저장되는가. “일곱 번째 칸”이라고 답하면 연속 tensor만 상상한 것이다. paged KV에서는 논리 token 위치를 block 크기로 나누어 논리 block 번호와 block 안 offset을 얻고, request의 block table에서 물리 block ID를 읽은 뒤 layer·KV head·vector 차원의 주소를 붙인다. 이 번역이 틀리면 kernel은 정상 실행하면서 다른 요청의 K와 V를 읽는다.

이 장은 block size 4인 작은 fixture를 손으로 계산한다. 요청 R의 logical block table이 `[7, 2, 11]`이라면 token position 0–3은 physical block 7, 4–7은 block 2, 8–11은 block 11에 놓인다. position 6은 logical block `6 // 4 = 1`, offset `6 % 4 = 2`, physical ID `table[1] = 2`다. 그 뒤 layer와 head, vector component를 더해 실제 cache element를 찾는다.

vLLM 기준은 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang은 `71de97b264b04dcd514cf904003028aefe9775c8`, Transformers는 `550d7b3834670483a4df436541272c055dc364bf`, llama.cpp는 `bb4caa7540188872173c44d161602d9271386413`이다. llama.cpp의 KV cell ring은 paged block table과 다른 주소 모델이므로 용어를 억지로 맞추지 않는다.

## 34.1 block size 4로 주소를 손으로 푼다

### 34.1.1 논리 위치와 물리 ID 사이에 table이 있다

R의 token positions가 0부터 9까지라고 하자. block size `B=4`이면 필요한 logical blocks는 `ceil(10/4)=3`개다. logical block 0은 positions 0–3, block 1은 4–7, block 2는 8–11의 주소 범위를 나타낸다. 마지막 block은 두 token만 유효하고 offsets 2–3은 아직 쓰이지 않았다.

allocator가 R에 physical IDs `[7, 2, 11]`을 주었다면 논리 순서와 물리 순서는 다르다. block 7 다음 memory block이 8이어도 R의 다음 논리 block은 physical 2다. kernel은 `physical_id = block_table[row, logical_block]`을 반드시 거친다. physical ID를 `first_id + logical_block`으로 계산하면 첫 block 뒤부터 다른 request의 cache를 읽는다.

position 6을 계산한다. `logical_block=1`, `offset=2`, `physical_id=2`다. cache layout을 단순히 `[physical_block, block_offset, kv_head, head_dim]`이라고 두면 K의 element `(head=3, component=5)`는 `K[layer][2][2][3][5]`다. V도 별도 tensor 또는 layout의 V 축에서 같은 block·offset을 쓴다. 실제 kernel layout은 vectorization과 backend에 따라 차원을 재배열할 수 있지만 table translation은 사라지지 않는다.

position 9는 logical block 2, offset 1, physical 11이다. valid sequence length가 10이므로 attention은 positions 0–9만 읽어야 한다. physical 11의 offsets 2–3에 오래된 값이나 다른 generation 흔적이 있어도 length mask가 막는다. block table만 맞고 valid length가 12로 잘못 전달되면 partial block의 쓰레기 두 token을 읽는다.

### 34.1.2 row stride까지 넣어 batch 주소를 계산한다

batch block table이 flat buffer이고 request당 최대 네 block entry를 예약한다고 하자. row stride는 4다. R이 row 1이면 logical block 2의 table entry 주소는 `base + 1*4 + 2`다. 그 값 11을 읽어 physical cache로 간다. 현재 R이 세 block만 쓴다고 stride가 3인 것은 아니다. static metadata buffer는 capacity stride를 쓸 수 있다.

다음 request S가 row 2이고 table `[5, 9, -1, -1]`이면 S의 first entry는 flat index 8이다. kernel이 active block count 3을 row stride로 잘못 쓰면 row 2 base를 6으로 계산해 R의 third entry와 padding을 S table로 읽는다. wrong block ID 사건은 allocator보다 metadata stride에서 시작할 수 있다.

row stride와 block size도 다른 단위다. block size는 physical KV block 하나가 담는 token 수다. table stride는 request row에 예약된 block ID entry 수다. byte stride는 table element dtype까지 곱한 값이다. 이 셋을 모두 `block_size`라 부르면 binding과 kernel 사이 ABI를 검토할 수 없다.

### 34.1.3 layer·head·vector 주소는 backend layout을 따른다

단순 row-major fixture에서 cache shape가 `[num_blocks, B, num_kv_heads, head_dim]`이고 element byte가 `e`라면 physical block `p`, offset `o`, head `h`, component `d`의 byte offset은 `((((p*B)+o)*H+h)*D+d)*e`다. layer마다 별도 tensor면 layer base를 먼저 고른다. K와 V가 별도 tensor면 각 base가 다르다.

일부 paged-attention kernel은 K를 vectorized load에 맞춰 `[block, head, head_dim/x, block_offset, x]`처럼 배치한다. 같은 논리 좌표도 stride 식이 달라진다. Python cache shape만 보고 native kernel 식을 추정하지 않고 backend metadata와 template specialization을 확인한다. block table은 physical page 선택을, kernel layout은 page 내부 element 선택을 담당한다.

tensor parallel에서는 local KV head 수와 global head 번호가 다를 수 있다. table의 physical block ID가 rank-local pool을 가리키는지, 동일 ID가 각 rank의 local shard를 가리키는지 확인한다. global head 6이 rank 1의 local head 2라면 head mapping을 적용한 뒤 address를 만든다. block translation과 head sharding은 직교하지만 최종 pointer에서 결합한다.

### 34.1.4 K와 V의 실제 linear offset을 두 layout으로 비교한다

fixture에 숫자를 더 넣자. physical blocks 16개, B=4, local KV heads `H=4`, head dimension `D=8`, FP16 `e=2 bytes`라고 하자. 단순 `[block, offset, head, dim]` layout에서 physical block 2의 token offset 2, head 3, component 5는 element index `(((2*4+2)*4+3)*8+5)=349`다. byte offset은 698이다. layer 0 K tensor base에 698 bytes를 더한다.

position 6이라는 논리 숫자는 계산 중간에 더는 직접 쓰이지 않는다. 먼저 table을 통해 `(p=2,o=2)`로 번역됐기 때문이다. position 6을 cache first dimension에 그대로 넣으면 element index `((6*4+3)*8+5)=221`이 되어 physical block 1의 다른 위치를 읽는다. contiguous-cache 공식을 paged tensor에 적용한 결과다.

K가 vector width `x=4`에 맞춰 `[block, head, dim/x, offset, x]`라면 component 5는 vector group 1, lane 1이다. coordinate는 `[2,3,1,2,1]`이고 stride 식도 달라진다. V는 `[block, head, dim, offset]`처럼 별도 layout을 쓸 수 있다. K와 V가 같은 logical token을 나타낸다고 pointer arithmetic이 같다고 단정하지 않는다.

layer dimension이 tensor 바깥 list라면 `key_cache[layer]`가 base를 고른다. 하나의 flat allocation에 layers를 pack했다면 layer stride를 더한다. KV cache group이 layers를 묶는 hybrid model에서는 group index와 group 안 layer index가 먼저 결정된다. vLLM `KVCacheBlocks`의 outer group이 주소 번역 사슬에 남는 이유다.

attention query의 head와 KV head도 같지 않을 수 있다. GQA에서 query head `qh`는 `kvh = qh // queries_per_kv_head`로 local KV head를 고른다. TP shard mapping을 그 전에 적용하는지 뒤에 적용하는지는 backend 계약을 따른다. wrong block ID와 wrong head mapping은 둘 다 타 요청의 vector처럼 보이므로 block trace에 local head도 남긴다.

이 손계산의 목적은 production kernel이 단순 layout을 쓴다고 주장하는 것이 아니다. kernel template에서 `block_table`, `block_size`, head mapping, cache strides를 발견했을 때 각 값이 논리 좌표의 어느 단계를 구현하는지 알아보는 것이다. 최적화된 pointer 식에서 division/modulo가 사라졌다면 power-of-two shift나 precomputed slot mapping으로 옮겨졌는지 찾는다.

### 34.1.5 layer·K/V·head stride를 끝까지 손으로 계산한다

앞 fixture에 layer와 K/V 축을 실제 숫자로 더한다. Layer 2개, physical block 16개, block size 4, local KV head 4개, head dimension 8, FP16 2 bytes이며 하나의 flat allocation을 `[layer, kv, block, offset, head, dim]` 순서로 pack한다고 하자. `kv` 축은 K=0, V=1이다. 한 token cell은 `4*8*2=64 bytes`, 한 block은 `4*64=256 bytes`, 한 K 또는 V plane은 `16*256=4096 bytes`, 한 layer는 `2*4096=8192 bytes`다.

R position 6은 logical block 1, offset 2, table entry 2다. Layer 1, V, local head 3, component 5의 byte 주소를 계산한다. Layer base는 `1*8192=8192`, V plane은 `1*4096=4096`, physical block은 `2*256=512`, token offset은 `2*64=128`, head는 `3*8*2=48`, component는 `5*2=10`이다. 합은 base에서 `12,986 bytes`다. 같은 position의 K라면 V plane 4096을 빼 `8,890 bytes`다.

이 계산에서 sequence position 6을 block stride에 직접 곱하지 않았다는 점이 핵심이다. Position은 table lookup 뒤 `(physical=2, offset=2)`로 사라진다. 잘못된 contiguous 식이 position 6을 block 내부 token처럼 쓰면 physical block 2 대신 연속 여섯 번째 token cell을 선택해 다른 block을 읽는다. 주소는 allocation 범위 안이므로 memcheck가 잡지 못할 수 있다.

Vectorized K layout을 `[layer, block, head, dim/x, offset, x]`, `x=4`로 바꾸면 같은 logical component 5는 group 1, lane 1이다. Stride 순서가 달라져도 physical block과 offset은 table에서 얻는다. Source에서 pointer 식을 읽을 때 `head_dim/x`, `block_offset`, lane이 어느 logical 좌표를 구현하는지 이 fixture로 대조한다. Python tensor의 보기 좋은 shape가 native specialization의 실제 stride라고 가정하지 않는다.

Tensor parallel rank도 넣자. Global KV heads가 8이고 TP=2이면 rank 1은 global heads 4–7을 local heads 0–3으로 가질 수 있다. Global head 7은 local head 3이므로 위 주소를 쓴다. Global head 3을 rank 1에서 local 3으로 잘못 해석하면 다른 shard에 존재하지 않는 의미를 읽는다. Block ID 2는 각 rank의 local pool에서 같은 정수일 수 있으므로 `(rank, pool generation, block id)`가 physical identity다.

Batch table stride는 KV stride와 별도다. Request row capacity가 4 entries, ID dtype int32라면 row byte stride는 16 bytes다. Row 1 logical block 1의 table byte offset은 `1*16+1*4=20`이고 그 값이 physical 2다. Active block count 3을 row stride로 쓰면 row 2부터 4 bytes씩 당겨져 valid하지만 다른 ID를 읽는다. Kernel launch는 성공하고 특정 batch compaction 뒤만 오답이 난다.

수치 검증은 manager table, runner host table, device table과 kernel-consumed ID 네 값을 기록한다. R row 1, logical block 1에서 모두 2여야 한다. 그 뒤 cache byte offset 12,986의 generation stamp 또는 표본 hash가 R의 KV generation과 맞는지 본다. Table이 맞고 content generation이 다르면 allocator/COW lifetime 문제다. Manager부터 ID가 다르면 allocation owner 문제다. Host는 맞고 device부터 다르면 H2D/static buffer generation 문제다.

이 손계산은 production tensor를 dump하라는 뜻이 아니다. 작은 fixture에서 stride와 coordinate를 검증하고 실제 incident에서는 `(row, logical block, physical id, layer, local head, offset, generation)`만 표본으로 남긴다. Prompt와 전체 KV를 수집하지 않아도 first wrong address를 함수 경계로 좁힐 수 있다.

## 34.2 partial last block은 allocated length와 valid length를 가른다

### 34.2.1 열 token은 세 block을 차지하지만 열두 token이 아니다

R length가 10이고 B=4이면 allocated capacity는 12다. 내부 fragmentation은 마지막 block의 unused slots 2개다. 이것은 block 두 개를 잃었다는 뜻이 아니라 한 request tail에서 최대 B-1 token slot이 비는 현상이다. 여러 짧은 request가 많으면 tail waste가 누적된다.

kernel metadata에는 block table length와 sequence length가 모두 필요하다. table은 세 physical blocks를 찾게 하고 length는 마지막 block에서 두 offset만 유효하게 한다. table entries가 세 개라는 사실로 valid token 12를 추론하지 않는다. 반대로 length 10인데 table이 두 entry뿐이면 position 8–9 translation가 table bounds를 넘는다.

decode로 position 10이 추가되면 같은 physical block 11 offset 2에 쓴다. 새 block allocation은 position 12, 즉 logical block 3에 진입할 때 필요하다. scheduler가 매 token 새 block을 요청하면 allocator overhead와 fragmentation이 커진다. position 10인데 새 block을 append하고 old partial block을 건너뛰면 logical continuity가 깨진다.

### 34.2.2 full block과 partial block은 공유 조건이 다르다

prefix cache는 일반적으로 full block 경계에서 재사용을 안정적으로 정의한다. partial tail은 뒤에 올 token에 따라 내용과 hash가 달라지고 두 request가 같은 prefix까지만 공유해도 각자 tail을 이어 써야 한다. read-only shared prefix와 writable tail의 ownership을 나눈다.

vLLM `get_computed_blocks()`는 computed cache block이 full이어야 한다고 명시한다. prompt 전체가 hit해도 logits를 얻기 위해 마지막 token을 재계산하고 block alignment 제약 때문에 한 block 전체를 다시 계산할 수 있다. [computed block 경계](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_manager.py#L202-L250)는 prefix hit length와 logical request length가 단순히 같지 않음을 보여 준다.

partial-page length 사고는 output quality로 조용히 나타난다. block IDs는 모두 pool 범위 안이고 kernel launch도 성공한다. 하지만 length가 capacity로 전달되면 attention denominator와 values에 stale positions가 들어간다. first divergence는 kernel이 아니라 scheduler/runner metadata가 valid length 대신 allocated length를 쓴 지점이다.

### 34.2.3 block size trade-off는 주소 metadata에도 나타난다

B가 작으면 tail waste 상한은 줄지만 같은 sequence에 더 많은 table entries가 필요하다. length 4096에서 B=4면 1024 IDs, B=16이면 256 IDs다. table H2D bandwidth, kernel indirection, manager object 수가 달라진다. B가 크면 table은 짧지만 짧은 request의 tail waste와 COW copy 단위가 커진다.

kernel은 지원하는 block size specialization을 가질 수 있다. config validator가 허용한 숫자라도 선택 backend가 같은 set을 지원하는지 확인한다. block size는 allocator만의 옵션이 아니라 cache tensor shape, block table capacity, kernel indexing constant, copy kernel 단위에 전파된다.

옵션 사슬은 `입력 block/page size → 최소·지원값 validation → num blocks와 table capacity 계산 → manager allocation granularity → runner metadata → kernel specialization`이다. 성능 표에서 block size 하나만 바꾸면 동일 memory budget에서 num physical blocks도 바뀔 수 있다. fragmentation과 metadata cost를 같은 workload에서 함께 측정한다.

### 34.2.4 fragmentation을 request 세 개로 계산한다

B=4에서 lengths 1, 5, 10인 A, R, S가 있다고 하자. 각각 allocated slots는 4, 8, 12이고 tail waste는 3, 3, 2다. valid 16 token을 위해 24 slots를 잡아 tail utilization은 `16/24=66.7%`다. B=8이면 allocations 8, 8, 16, waste 7, 3, 6으로 utilization은 `16/32=50%`다. 이 작은 workload에서는 큰 block이 불리하다.

하지만 table entries는 B=4에서 `1+2+3=6`, B=8에서 `1+1+2=4`다. 각 entry가 4-byte ID라면 raw IDs는 24 bytes와 16 bytes다. 실제 table은 request별 fixed row capacity와 alignment를 가져 차이가 더 클 수 있다. cache data에 비해 작아 보여도 매 step H2D와 kernel random lookup이 반복된다.

길이가 정확히 block boundary인 request가 많으면 tail waste 차이는 줄어든다. 길이 8 request는 B=4에서 두 blocks, B=8에서 한 block이고 둘 다 unused 0이다. COW에서는 B=8 partial tail copy가 최대 8 token KV를 복제하지만 B=4는 최대 4다. prefix cache hit granularity도 달라질 수 있지만 hash key의 상세는 35장으로 넘긴다.

sliding window나 recurrent/hybrid group은 모든 logical positions를 full-attention처럼 보존하지 않을 수 있다. group마다 effective retained blocks가 다르면 하나의 waste 공식으로 전체를 합치지 않는다. manager outer group과 backend spec을 기준으로 addressable window를 계산한다. old logical position이 table에서 null 또는 reused slot로 표현될 수 있다.

block size 선택의 최적점은 평균 length 하나로 정해지지 않는다. request length distribution, concurrent tails, prefix sharing, fork 빈도, table bandwidth, kernel supported sizes가 함께 결정한다. 옵션 문서는 기본값과 허용 범위뿐 아니라 어떤 downstream state가 바뀌는지 설명해야 한다.

## 34.3 vLLM: manager의 block이 runner table이 되기까지

### 34.3.1 `KVCacheBlocks`는 group과 block 순서를 보존한다

vLLM `KVCacheBlocks.blocks[i][j]`에서 바깥 `i`는 KV cache group, 안쪽 `j`는 그 group의 logical token block 순서다. 모든 group이 영원히 같은 block 수를 가진다고 가정하지 않으려고 group을 바깥 차원으로 둔다. `get_block_ids()`는 각 `KVCacheBlock.block_id`를 tuple of lists로 바꿔 scheduler와 runner 경계에 전달한다. [자료구조 계약](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_manager.py#L30-L83)을 flat list 하나로 축약하면 hybrid group mapping을 잃는다.

`KVCacheManager`는 config, model length, scheduler/hash block sizes를 받아 coordinator를 만들고 그 block pool을 노출한다. scheduler block size와 hash block size가 별도 인자라는 점은 주소 allocation 단위와 prefix hashing 단위가 개념상 다를 수 있음을 보여 준다. 이 장은 hash 계산을 35장에 넘기고 physical allocation과 table만 본다.

manager allocation 결과는 request의 logical order를 유지해야 한다. physical IDs가 `[7,2,11]`처럼 흩어져도 list append 순서는 positions의 block order다. allocator free queue 순서와 request table 순서를 섞지 않는다. pool은 global physical resource owner이고 request blocks는 logical view다.

### 34.3.2 block pool은 ID, reference, eviction 가능성을 관리한다

vLLM `BlockPool`은 `0..num_gpu_blocks-1` ID를 가진 `KVCacheBlock` 객체를 만들고 free queue를 구성한다. caching이 켜지면 queue에는 단순 zeroed free뿐 아니라 eviction 가능한 cached block도 포함될 수 있다. allocation가 block을 가져올 때 old hash metadata와 reference state를 올바르게 갱신해야 한다. [pool 초기화](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/block_pool.py#L131-L193)를 보면 physical ID namespace가 pool에서 정해진다.

null block은 block ID 0을 placeholder로 쓰지만 일반 ref count를 유지하지 않는 특별 객체다. kernel table에 0이 보였다고 반드시 request의 실제 first block이라고 해석하지 않는다. padding·null semantics와 valid metadata를 함께 본다. special ID를 일반 pool block처럼 free하면 free queue가 오염될 수 있다.

prefix cached full block은 여러 request가 참조할 수 있다. ref count가 0이어도 cached content로 eviction 후보가 될 수 있고 새 matching request가 다시 참조한다. “free”는 즉시 zeroing된다는 뜻이 아니다. table에서 제거된 block의 오래된 bytes는 length와 ownership metadata가 읽기를 막는다.

pool의 free queue를 단순 stack of unused memory로 이해하면 eviction을 놓친다. caching이 꺼진 경우 free block은 곧 allocation 가능한 unowned resource에 가깝다. caching이 켜지면 ref count 0인 complete block도 queue에 있으면서 content hash lookup 대상일 수 있다. 새 allocation가 이를 가져갈 때 hash map에서 old identity를 제거하고 block metadata를 새 generation으로 전환해야 한다.

R의 block 7 ref count가 2이고 R이 끝났다면 decrement 뒤 1이므로 S가 계속 읽는다. free queue에 넣어서는 안 된다. ref count가 1인데 allocator queue에도 존재하면 같은 physical block에 S reader와 T writer가 생긴다. table trace와 pool queue membership을 함께 보는 이유다.

ref count가 0이 된 complete block은 cached lookup으로 다시 살아날 수 있다. S prefix lookup가 ID 7을 선택하면 queue에서 제거하고 ref를 올린다. 같은 순간 eviction allocator가 7을 가져가지 않도록 pool mutation가 serialize되어야 한다. Python object reference가 존재한다는 사실만으로 physical allocation ownership이 보장되지 않는다.

null block은 더욱 특별하다. pool initialization가 free queue 첫 block을 빼 ID 0 null로 만들고 일반 ref count를 유지하지 않는다. padded group이나 absent cache mapping이 null을 사용할 수 있다. request cleanup가 table의 모든 IDs를 순회하며 0을 일반 free하면 null semantic이 무너진다. address debug에서 ID 0은 valid ordinary allocation인지 special null인지 config와 group metadata를 확인한다.

hash lookup가 동일 content blocks 중 임의 하나를 돌려줄 수 있어도 request table에는 선택된 정확한 ID가 들어간다. 이후 output trace에서 hash만 기록하면 어느 physical generation을 읽었는지 알 수 없다. content identity와 address identity를 둘 다 남긴다. 35장에서는 hash key가 content equality를 어떻게 증명하는지 다루지만 여기서는 선택 뒤 ID lifetime만 추적한다.

block event나 metrics가 켜져 있다면 allocation/free/cache event를 runner table generation과 연결할 수 있다. event timestamp만으로 ordering을 증명하지 않고 request ID, block ID, generation, ref before/after를 본다. event가 비활성일 때도 manager source에서 어느 mutation 직후 표본 trace를 넣을 수 있는지 정한다.

pool 사용률이 100%에 가까워도 모든 blocks가 running requests에 pinned됐다는 뜻은 아니다. eviction 가능한 cached blocks가 queue에 포함될 수 있고 watermark가 admission을 제한할 수 있다. 반대로 usage가 내려갔어도 deferred writer 때문에 실제 safe reuse가 늦을 수 있다. address safety는 aggregate usage보다 owner state에 있다.

### 34.3.3 scheduler output과 runner row가 table을 구체화한다

scheduler는 새 request와 cached/resumed request에 block IDs를 넣어 runner metadata를 만든다. runner는 request row별 block table buffer를 갱신하고 attention backend에 table, sequence length, slot mapping을 전달한다. logical request list order와 runner row order가 다르면 table row도 같은 permutation으로 이동해야 한다.

continuous batching에서 finished row를 compact할 때 Python list만 줄이면 stale table row가 남는다. table buffer가 static capacity라면 unused row를 clear 또는 valid-count로 mask해야 한다. 새 R이 old S row를 재사용할 때 이전 table suffix가 남아도 current block count와 length가 kernel read를 제한해야 한다.

async scheduling에서는 previous step과 current table generation이 겹칠 수 있다. runner buffer를 덮기 전에 이전 kernel이 table을 모두 읽었는지 stream ordering이 필요하다. block IDs 자체가 맞아도 stale table generation을 kernel이 읽으면 use-after-free와 같은 결과가 난다. 31장의 future state를 여기서는 address metadata lifetime으로 적용한다.

### 34.3.4 append-only table과 allocation rollback을 읽는다

vLLM block pool 주석은 동일 hash content가 있어도 새로 complete된 allocated block ID를 deduplicate해 바꾸지 않는 이유로 block table의 append-only 성질을 든다. request가 받은 physical IDs가 실행 중 갑자기 다른 cached ID로 교체되면 runner table update와 in-flight kernel ordering이 복잡해진다. content equality와 address identity 안정성은 별도 선택이다.

append-only는 table entry가 절대 삭제되지 않는다는 뜻이 아니다. request lifetime 안에서 새 logical tail blocks를 뒤에 붙일 때 기존 entry IDs를 안정적으로 유지한다는 의미다. preemption, finish, COW, cache compaction은 다른 table generation을 만들 수 있다. source 주석 범위를 넘겨 global 불변식으로 쓰지 않는다.

allocation transaction을 fixture로 본다. R table `[7,2]`에 position 8–9를 위한 block 11을 얻었다. scheduler output을 만들기 전에 다른 resource 준비가 실패하면 11을 request owner에서 떼고 pool state로 정확히 돌려야 한다. table에는 `[7,2,11]`이 남아서는 안 된다. runner가 old table을 쓴다면 physical copy도 제출되지 않아야 한다.

반대로 scheduler output에 11을 넣은 뒤 request blocks append가 실패하면 runner는 owner 없는 block을 쓴다. allocation result, request table mutation, scheduler output serialization의 commit order를 함께 본다. exception handler가 pool free만 해도 이미 만든 runner metadata를 전달하면 use-after-free다.

hybrid KV groups에서는 모든 group allocation이 성공해야 logical position이 addressable하다. group 0이 block 11을 얻고 group 1이 부족하면 group 0만 commit하지 않는다. `KVCacheBlocks` outer tuple이 group별 결과를 묶으므로 partial group rollback을 확인한다. 한 group table만 길면 layer마다 같은 position의 KV 존재가 달라진다.

runner metadata source walk는 new와 cached/resumed request를 나누어 본다. new path는 전체 IDs를 설치하고 cached path는 기존 row에 append할 수 있다. resume는 old runner row와 같은 row일 이유가 없으므로 request ID mapping을 기준으로 table을 옮긴다. row index를 lifetime identity로 쓰지 않는다.

### 34.3.5 block table의 상태 전이를 요청 R로 끝까지 잇는다

R이 처음 들어왔을 때 prompt length가 6이고 B=4라고 하자. allocator는 physical 7과 2를 주고 request table은 `[7,2]`, valid length는 6이다. runner row 3에 이를 설치한다. first prefill은 block 7 offsets 0–3과 block 2 offsets 0–1에 K/V를 쓴다. logical table은 두 entries지만 second block은 writable partial이다.

decode positions 6과 7은 같은 block 2 offsets 2와 3에 append된다. position 8을 schedule할 때만 새 physical 11을 allocation하고 request table을 `[7,2,11]`로 append한다. scheduler output에는 새 block과 scheduled position이 일관되게 들어간다. runner host table row 3의 third entry가 11로 바뀌고 H2D completion 뒤 compute가 offset 0에 쓴다.

이때 prefix cache가 blocks 7과 2를 complete로 표시할 수 있다. complete는 content identity와 read-only sharing 가능성을 뜻하지만 R table에서 사라진다는 뜻이 아니다. 새 S가 first 8 token을 공유하면 S table first two entries도 `[7,2]`가 되고 ref counts가 늘어난다. S의 private tail은 다른 ID를 받는다.

R이 preempt되면 정책에 따라 private blocks reference를 놓고 computed frontier를 reset한다. blocks 7과 2가 cached eviction candidates로 남을 수 있고 11 partial은 uninitialized free가 될 수 있다. R logical token history는 request object에 남아도 runner table row 3은 더 이상 R owner가 아니다. resume 때 row 1과 새 table generation을 얻을 수 있다.

resume prefix lookup이 first 8 token blocks 7과 2를 다시 찾고 private 12를 얻었다면 new table은 `[7,2,12]`다. old physical 11 bytes가 아직 device에 남아도 owner가 아니다. output corruption 조사에서 “R은 예전에 11을 썼다”는 로그로 current table을 복원하면 stale mapping을 만든다. table에는 generation과 commit 시점이 필요하다.

R이 자연 finish하면 runner membership을 제거하고 request references를 decrement한다. block 12가 full·complete인지 partial인지에 따라 cached initialized 또는 uninitialized free state가 달라질 수 있다. connector나 async writer가 있으면 pool 반환은 지연될 수 있다. table row를 새 request T가 재사용하는 시각과 physical blocks가 재사용되는 시각은 독립적이다.

이 전이를 owner 표로 읽으면 request table owner, runner metadata row owner, physical block ref owner, cache content identity가 같은 순간에 바뀌지 않음을 알 수 있다. table row 3은 R에서 T로 넘어가도 block 7은 R/S prefix reference로 남을 수 있다. R이 끝나도 block 12의 write completion fence가 남을 수 있다. 하나의 `freed` 로그로 모두 표현하지 않는다.

rollback은 각 전이의 역연산이다. block 11 allocation 뒤 scheduler commit 전 실패하면 11만 반환하고 `[7,2]`를 유지한다. table H2D 뒤 compute launch 전 cancel이면 generation을 invalid로 만들고 buffer reuse ordering을 지킨다. compute launch 뒤 abort면 block을 즉시 재사용하지 않고 completion 뒤 반환한다. failure position이 달라 rollback이 다르다.

block table capacity를 넘는 append는 allocation 전에 validation해야 한다. max blocks per request가 3인데 position 12를 schedule하면 logical block 3 entry가 필요하다. physical block을 먼저 얻고 table bounds에서 실패하면 rollback이 필요하다. 더 위험한 경우 bounds check 없이 row 다음 request entry를 덮는다. model max length와 table capacity validation를 연결한다.

runner가 CUDA graph static table buffer를 쓰면 active row capacity와 capture bucket을 구분한다. current batch 3 requests라도 captured table은 8 rows일 수 있다. unused rows는 trash/sentinel 또는 valid request count로 무효화한다. stale suffix를 zeroing하는 것만으로 row owner correctness가 증명되지는 않는다.

### 34.3.6 vLLM table generation을 allocator generation과 맞춘다

Physical ID는 재사용되는 작은 정수다. Block 7이 R에게 할당됐다가 ref count 0과 eviction을 거쳐 T에게 다시 할당돼도 table entry는 둘 다 7이다. 따라서 host table diff만 보면 stale R table과 current T table을 구분하지 못한다. Debug identity는 `(pool generation, block id, allocation epoch)`를 가져야 한다. Epoch가 코드의 실제 field가 아니라면 event sequence나 별도 표본 계측으로 만든다.

Incident P34에서 scheduler step 80은 R row 2에 logical table `[7,2,11]`, allocation epochs `[14,9,22]`를 만들었다. R은 preempt되어 blocks를 반환했고 block 2는 T의 epoch 10이 되었다. Step 83에서 R이 resume해 새 table `[5,13,9]`를 얻었지만 runner static buffer row 2의 second entry가 H2D update에서 누락돼 여전히 2였다. Kernel은 pool 범위 안의 block 2 epoch 10을 정상적으로 읽어 T의 KV를 섞었다.

Observation은 batch compaction 뒤 R의 token 품질이 흔들리고 CUDA error가 없다는 것이다. Branch는 allocator가 wrong ID를 준 경우, scheduler output이 old table을 만든 경우, runner host table update가 누락된 경우, device table copy ordering이 잘못된 경우, kernel row/stride가 틀린 경우다. Manager, scheduler output, host static row, device row와 consumed ID를 같은 batch generation에서 차례로 비교한다.

P34에서는 manager와 scheduler output이 `[5,13,9]`, host static row도 update 후 `[5,13,9]`였지만 device snapshot은 `[5,2,9]`였다. First divergence는 allocator가 아니라 host-to-device content generation이다. Static address가 같다는 사실은 graph replay에 유리하지만 내용이 current라는 증거는 아니다. Copy event와 graph replay stream 사이의 happens-before를 확인한다.

반증 fixture는 row 2만 바꾸지 않는다. Batch rows를 `[A,R,T]→[R,T]→[U,R,T]`로 compact·expand하며 R logical block 1의 ID와 epoch를 매번 바꾼다. Graph on/off, eager/reference backend를 교차한다. Eager에서도 device row가 stale이면 graph key 문제는 약하다. Graph replay에서만 stale이면 static buffer update·capture generation을 본다. Backend reference도 같은 device table을 소비한다면 둘 다 틀릴 수 있으므로 manager table을 reference 입력으로 별도 materialize한다.

Rollback은 table row를 current 값으로 한 번 덮는 데서 끝나지 않는다. New graph replay를 막고, old batch generation을 drain하며, device table content generation과 KV allocation epochs가 일치하는 canary를 실행한다. Epoch가 불명확한 physical blocks는 evictable cache로 남기지 않고 worker pool을 격리한다. R output이 정상으로 돌아와도 T의 block이 old reader에게 노출됐으므로 affected request 범위를 block event ledger로 찾는다.

vLLM source walk는 `KVCacheBlocks.get_block_ids()`에서 scheduler-facing logical order, scheduler output이 runner에 넘기는 IDs, runner block table update와 attention backend consumer까지 닫는다. `BlockPool` free queue와 cached block ref mutation은 content generation owner다. 각각의 함수가 따로 맞아도 generation handoff가 없으면 P34가 생긴다.

수정 검증은 ID parity와 content parity를 둘 다 본다. 모든 경계의 ID가 13이어도 block 13이 old epoch content라면 실패다. ID가 의도된 shared prefix로 같다면 epoch와 read-only status도 같아야 한다. Writable tail에서는 R과 T가 같은 physical generation을 가리키지 않아야 한다. 이 세 조건을 통과해야 stale table과 정상 공유를 구분했다.

## 34.4 shared prefix와 copy-on-write가 table을 갈라 놓는다

### 34.4.1 full prefix는 같은 physical ID를 가리킬 수 있다

R과 S가 첫 8 token이 같고 B=4라면 logical blocks 0–1을 공유할 수 있다. R table `[7,2,11]`, S table `[7,2,5]`처럼 first two physical IDs가 같다. block 7과 2는 read-only complete prefix이고 각 request tail은 11과 5로 갈린다. shared prefix는 table entry alias가 의도된 경우다.

attention read는 같은 block을 여러 request가 참조해도 안전하다. append write가 shared full block 안으로 들어가면 안전하지 않다. full block 다음 position은 새 tail block offset 0이므로 자연스럽게 갈린다. partial block을 공유해야 하는 fork나 beam path는 writable suffix를 copy-on-write해야 한다.

reference count는 physical block lifetime을 지킨다. R이 끝나 table에서 7을 빼도 S reference가 남으면 pool로 반환하지 않는다. ref count decrement와 table mutation ordering이 어긋나면 S가 읽는 block이 재할당된다. 반대로 ref를 올리고 table entry를 설치하지 못한 rollback은 block을 eviction하지 못하는 누수다.

### 34.4.2 COW는 alias를 없애고 내용은 복제한다

R과 child C가 partial block 11의 offsets 0–1까지 같은 context를 가졌다고 하자. 둘이 다음 token을 다르게 append하면 같은 physical block offset 2에 동시에 쓸 수 없다. allocator는 C에 새 block 13을 주고 11의 유효 prefix를 13으로 copy한 뒤 C table last entry를 13으로 바꾼다. R은 11을 유지한다.

copy source와 destination, valid copy length, table update, ref count decrement가 한 transaction이다. destination table을 먼저 공개하고 copy가 끝나기 전에 kernel이 읽으면 uninitialized KV다. copy 뒤 table update를 빼먹으면 새 block은 누수되고 C는 계속 alias를 쓴다. ref count를 먼저 줄여 source가 pool에 돌아가면 copy kernel이 재사용된 bytes를 읽는다.

Transformers `BlockManager.fork_blocks()`는 complete shareable blocks를 reference로 fork하고 incomplete suffix는 새 blocks를 allocation해 copy source/destination 목록을 만든다. 예제 자체가 `[0,1,2,3]`에서 complete 0–2는 공유하고 partial 3만 child별 새 ID로 복제하는 구조를 보여 준다. [fork와 COW 목록](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache_manager.py#L126-L208)은 alias와 copy를 같은 말로 부르면 안 되는 이유다.

### 34.4.3 COW alias 사고는 정상 공유와 구분한다

두 request table에 같은 block ID가 있다는 사실만으로 오류가 아니다. block이 complete read-only인지, partial writable인지와 reference count를 본다. 정상 shared prefix는 동일 ID·다중 ref·write 없음이다. COW alias는 동일 partial ID에 두 writer가 다른 logical suffix를 쓰는 상태다.

증상은 beam 두 개가 갑자기 같은 token history를 보거나 한 request 진행이 다른 request 품질을 바꾸는 것이다. first divergence는 sampler가 아니라 fork에서 incomplete block을 reference-only로 넘긴 지점 또는 table swap보다 copy event가 늦은 지점이다. source/destination generation과 write slot을 추적한다.

### 34.4.4 COW publish는 copy completion 뒤의 address commit이다

부모 R과 자식 C가 full prefix blocks `[7,2]`를 공유하고 partial tail block 11의 offsets 0–1까지 유효하다고 하자. C가 offset 2에 새 token을 쓰기 전에 writable destination 13을 얻고 valid bytes를 복사해야 한다. C table이 `[7,2,13]`이 되는 것과 copy가 끝나는 것은 다른 사건이다. Table publish만 먼저 보이면 child attention이 미완성 block 13을 읽을 수 있다.

앞 layout에서 한 layer의 K/V block은 512 bytes이고 layer가 2개면 partial physical block 전체는 1024 bytes다. Valid offsets가 2개뿐이라 논리적으로 필요한 payload는 절반이지만 copy primitive가 full block을 복사할 수 있다. Full copy는 stale unused offsets도 옮기지만 valid length가 2라면 correctness에는 영향이 없다. 다음 write가 offset 2를 덮는다. 비용 최적화와 address correctness를 구분한다.

COW transaction은 `reserve13 → retain source11 → enqueue copy(11→13) → record completion → publish/read table13 → release source write dependency`다. Host에서 table을 먼저 바꿔도 compute stream이 copy event를 기다리면 안전할 수 있다. 반대로 copy API가 return했어도 nonblocking enqueue라면 source ref를 즉시 줄이는 것은 위험하다. Python call 순서가 아니라 consumer happens-before를 본다.

Eviction이 끼면 위험이 커진다. Source 11의 ref count가 copy 시작과 동시에 0이 되어 eviction allocator가 block 11을 generation 30으로 재사용하면 copy kernel이 새 content 일부를 읽을 수 있다. Destination 13은 유효한 범위 안이고 copy도 성공하므로 조용한 provenance 오염이다. Source generation은 copy completion까지 read lease를 가져야 한다.

Fork 두 개 C1과 C2를 동시에 만들고 destination 13, 14를 예약한다. C1 copy는 성공하고 C2 allocation이 실패하면 API가 전체 fork atomicity를 요구하는지 부분 성공을 허용하는지 명시한다. 전체 rollback이면 destination 13과 shared prefix ref increments를 되돌리되 이미 launch된 copy completion 뒤 반환한다. C1 table publish를 지웠다고 destination writer가 사라진 것은 아니다.

Race fixture는 copy completion과 table publish, source eviction 순서를 바꾼다. 정상 case는 source lease 유지, copy complete, child publish, source release다. Case B는 publish가 먼저지만 compute가 event를 기다려 정상이다. Case C는 source eviction이 copy completion보다 빨라 corruption이 나야 한다. Case D는 destination table이 old batch row에 publish되어 다른 child가 읽는다. 각 case는 source/destination ID와 epoch, valid length, copy event, consumer batch generation을 남긴다.

Cause branch는 네 가지다. Incomplete tail을 complete/shareable로 잘못 분류했는가. Destination allocation과 table publish가 다른 generation인가. Copy event를 consumer stream이 기다리지 않았는가. Source read lease가 일찍 끝났는가. Shared prefix ID가 같다는 관측만으로 COW alias를 선언하지 않는다. Full read-only blocks의 alias는 정상이다. 최초 writable offset에서 physical generation이 분리되는지 본다.

Rollback은 모든 prefix sharing을 끄는 것으로 끝내지 않는다. Affected tail generations를 격리하고 child table을 safe copy로 재구성한다. Source content provenance가 불명확하면 parent와 children 요청을 오류 종료하고 해당 blocks를 cache hit 대상으로 쓰지 않는다. 수정 뒤 full prefix는 여전히 공유되어 memory 이득을 유지하고, partial writable tail만 destination generation으로 갈라지는지 검증한다.

## 34.5 SGLang: request pool index와 token pool 위치를 번역한다

### 34.5.1 request row와 token location은 두 단계 indirection이다

SGLang의 `Req`는 request pool에서 row index를 얻고 request-to-token pool은 그 row와 logical token position을 physical token/KV location으로 연결한다. radix cache hit로 얻은 prefix와 새 allocation suffix가 한 row mapping에 이어진다. vLLM block ID list와 자료구조 이름은 다르지만 logical sequence가 physical storage 위치를 직접 연속으로 가정하지 않는다는 점은 같다.

block/page 기반 backend에서는 token location을 block ID와 offset으로 다시 해석하거나 paged attention metadata를 만든다. scheduler의 `ScheduleBatch`는 request pool indices, sequence lengths, output cache locations와 attention backend용 정보를 같은 row order로 유지해야 한다. `filter_batch()`와 merge가 관련 tensor 전체에 같은 permutation을 적용해야 한다.

radix prefix가 shared physical locations를 제공하면 새 suffix allocation은 그 뒤 logical positions에 연결된다. prefix lock/reference를 놓는 시점과 request row table update가 어긋나면 eviction된 location을 읽을 수 있다. hash tree 탐색 자체는 다음 장에서 다루고 여기서는 hit 결과가 address mapping에 들어오는 경계만 본다.

### 34.5.2 paged attention metadata는 kernel ABI다

kernel은 Python `Req` 객체를 읽지 않는다. batch size, sequence lengths, request pool indices 또는 block table, slot mapping, page size 같은 device metadata를 받는다. backend마다 필드 이름과 layout이 다를 수 있다. scheduler object가 맞다는 사실로 native kernel input이 맞다고 단정하지 않는다.

prefill은 여러 query positions를 한 batch에 packing하고 decode는 대개 request당 적은 query를 쓴다. 같은 KV table도 query-to-request mapping이 다르다. flattened query index가 어느 request row와 logical position인지 metadata가 연결한다. wrong cumulative sequence length는 올바른 block table의 잘못된 row를 선택하게 한다.

overlap 경로에서는 future tensor와 schedule stream tensor의 lifetime도 table에 적용된다. 이전 forward가 table storage를 읽는 동안 다음 schedule이 같은 buffer를 compact하면 stale table 사건이다. forward_done/copy event가 어느 metadata storage를 보호하는지 확인한다.

### 34.5.3 token pool 누수와 wrong location을 구분한다

token pool free count가 줄어드는 누수는 mapping entry가 request cleanup 뒤 반환되지 않은 경우다. wrong location은 entry는 존재하지만 다른 request 또는 logical position을 가리킨다. 전자는 admission/OOM으로, 후자는 silent corruption으로 나타난다. count metric만으로 wrong mapping을 잡기 어렵다.

표본 trace에 `(request id, req_pool_idx, logical position, physical token loc, batch row, generation)`을 남긴다. output corruption 시 같은 physical loc에 겹친 두 writer를 찾는다. free 뒤 generation 없이 같은 integer loc가 재사용되는 것은 정상일 수 있으므로 시각과 generation을 붙인다.

### 34.5.4 radix hit 이후 suffix를 붙이는 주소 장부

R prompt 10 token에서 radix cache가 first 8 positions를 hit했다고 하자. B=4 backend라면 prefix physical blocks 또는 token locations가 logical 0–7에 대응한다. scheduler는 positions 8–9를 위한 suffix locations를 새로 allocation한다. request pool row는 shared prefix mapping과 private suffix mapping을 하나의 logical sequence로 보여 준다.

kernel query가 position 9일 때 causal attention은 0–9 KV를 읽는다. physical locations는 radix tree node 순서가 아니라 request row mapping 순서로 제공되어야 한다. tree가 content lookup을 담당하고 pool table이 execution address order를 담당한다. radix child order를 kernel table로 직접 쓰면 logical positions가 바뀐다.

prefix reference는 forward가 shared locations를 읽는 동안 eviction을 막는다. suffix allocation 실패 rollback에서 request-local reference를 놓되 다른 request reference까지 free하지 않는다. waiting request가 오래 prefix를 pin하면 eviction pressure가 생길 수 있으므로 owner와 pin age를 관측한다.

chunked prefill에서는 total request length, current chunk query length, computed prefix length, KV valid length가 다르다. block table은 전체 readable context를 가리키고 slot mapping은 이번 write positions를 가리킨다. current query count를 total KV length로 쓰면 과거 prefix를 읽지 못한다.

decode retraction이나 abort는 row를 filter하고 suffix locations를 release한다. overlap 경로에서 이전 forward가 row storage를 읽고 있다면 event를 기다린다. token pool free count가 먼저 회복됐다는 사실은 안전한 cleanup 증거가 아니다.

backend가 token-level pool mapping을 paged block metadata로 압축한다면 division/modulo와 table build 위치를 찾는다. 이미 physical token location을 flat slot로 제공하면 kernel이 block table을 다시 읽지 않을 수 있다. 이름이 아니라 binding 인자를 기준으로 주소 사슬을 그린다.

### 34.5.5 SGLang row와 token location generation을 함께 검증한다

SGLang에서는 request pool row가 곧 physical KV 주소가 아니다. Row는 요청별 metadata를 찾는 첫 indirection이고, token pool location 또는 paged metadata가 실제 cache storage를 찾는 다음 indirection이다. Batch compaction 뒤 request row만 맞고 token locations가 old permutation이면 kernel은 유효하지만 다른 요청의 cells를 읽는다.

작은 fixture에서 request pool rows는 R=4, S=9이고, R의 positions 0–5는 token locations `[20,21,22,23,44,45]`라고 하자. Block size 4로 보는 backend라면 first page는 locations 20–23의 contract, tail은 44–45와 valid length 2를 나타낸다. S가 끝나 batch row가 줄어들어도 R request pool index 4와 token locations는 바뀌지 않을 수 있다. Packed batch row와 persistent request pool row를 같은 숫자로 쓰면 wrong lookup이 된다.

Radix prefix hit 뒤 R의 first four locations가 shared generation 7이고 suffix 44–45가 writable generation 12일 수 있다. Hit count 4만 기록하면 실제 physical provenance를 잃는다. Prefix ref, suffix allocation, request pool row와 backend page metadata를 같은 request generation으로 묶는다. Eviction이 shared locations를 재사용하기 전에 radix reference가 0이고 in-flight reader가 없는지 확인한다.

Incident S34는 retraction 뒤 resume에서 발생한다. Old request pool row 4가 반환되고 새 R incarnation은 row 6을 얻었지만 overlap future가 row 4의 token-location vector를 device metadata에 썼다. Row 4는 이미 T가 사용 중이었다. Physical locations가 pool 범위 안이라 launch는 성공했다. First divergence는 attention kernel이 아니라 future batch가 persistent row generation을 보존하지 않은 지점이다.

Verification은 scheduler `Req` identity, request pool index와 generation, token locations, batch packed row, device metadata를 한 줄로 비교한다. `filter_batch()`가 살아남은 packed indices를 관련 tensor 전체에 적용했는지 확인한다. `release_req()` 뒤 old row를 참조하는 future가 있다면 resource return과 future invalidation ordering을 조사한다. Aggregate token pool free count로는 wrong location을 찾을 수 없다.

Rollback은 overlap admission을 닫고 old batch/future를 drain한 뒤 request pool과 token pool generation을 재구성한다. Radix cache content generation까지 불명확하면 단순 request resume를 하지 않고 해당 cache entries를 invalidate한다. Safe canary는 shared full prefix, new writable suffix와 batch compaction을 함께 거쳐 각 logical position의 location generation이 기대 owner와 일치해야 한다.

## 34.6 Transformers: cache tensor와 `BlockManager`의 특별 주소

### 34.6.1 cache shape는 두 extra blocks를 포함한다

Transformers `PagedAttentionCache`는 block size가 최소값보다 작은지 검사하고 model KV heads, head dimension, TP plan, attention layer group을 반영한다. physical cache shape는 `(num_blocks + 2, block_size, local_kv_heads, head_dim)`에 해당하는 flat first dimension을 만든다. 두 extra blocks는 allocator가 일반 request에 주지 않는 padding zone이다. [cache 초기화와 shape](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache.py#L89-L210)에 그 이유가 드러난다.

첫 extra block에는 read trash와 sentinel index가 있고 두 번째는 write trash다. padding token read가 정상 block을 오염시키지 않고, padding write도 실제 request KV에 닿지 않게 한다. physical index가 allocatable num blocks 이상이라고 무조건 out-of-bounds로 판정하지 않는다. special index 계약을 확인한다.

block size default나 backend 호환성은 버전에 따라 달라질 수 있으므로 고정 commit config를 읽는다. cache constructor validation, memory handler의 num blocks 계산, TP all-reduce minimum, tensor shape와 attention integration까지 option을 추적한다. config 값만 바꾸고 static tensor나 kernel metadata가 old shape를 유지하면 stale stride가 된다.

### 34.6.2 manager는 initialized와 uninitialized free를 구분한다

Transformers `BlockManager`는 uninitialized free deque와 initialized cached ordered set을 가진다. initialized block은 complete content와 hash를 보존하지만 ref count 0이라 eviction 가능하다. uninitialized가 부족하면 initialized block의 hash mapping을 제거하고 다시 allocation한다. [free block 확보](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache_manager.py#L58-L125)는 free가 zero content와 동의어가 아님을 보여 준다.

shareable block은 `Block` 객체와 parent ID, group ID, ref count를 가진다. incomplete block은 complete hash가 없고 reference가 0이 되면 uninitialized queue로 돌아간다. complete block은 initialized cache 후보가 된다. address ID lifecycle과 content validity lifecycle을 분리한다.

`get_free_blocks()` 실패는 table에 partial append를 남기지 않아야 한다. 여러 blocks를 요청할 때 enough check 뒤 일괄 deque pop을 한다. fork 중 child 일부 allocation 뒤 다음 child가 실패하면 caller가 이미 만든 destination을 rollback하는지까지 봐야 한다. 함수 한 단계 성공만으로 transaction 전체를 증명하지 않는다.

`has_enough_free_blocks()`는 uninitialized blocks가 부족하면 initialized cache blocks를 uninitialize한다. 필요한 수만큼 ordered set에서 꺼내 hash-to-ID mapping을 제거하고 uninitialized deque에 넣는다. 여기서 content bytes를 반드시 zero로 만들 필요는 없다. ownership과 hash validity를 지우면 새 writer가 덮는다. valid length와 table owner가 old bytes read를 막아야 한다.

`get_free_blocks()`는 enough check 뒤 uninitialized deque에서 IDs를 뽑는다. shareable이면 각 ID에 parent와 group을 가진 `Block` 객체를 만들고 last block을 새 parent로 이어 간다. 이 parent chain은 logical prefix identity 구축에 쓰일 수 있지만 kernel address 순서는 request block list가 제공한다. parent pointer를 table next pointer로 오해하지 않는다.

`increase_ref_count()`에서 ref가 0에서 1이 되면 initialized free set에서 block을 제거한다. `decrease_ref_count()`가 0을 만들면 complete는 initialized set으로, incomplete는 tracking map에서 제거해 uninitialized deque로 보낸다. 같은 ref=0이어도 content completeness에 따라 다음 상태가 다르다. COW tail cleanup과 shared full prefix cleanup이 갈리는 줄이다.

nonshareable blocks는 객체 ref tracking 없이 FIFO free 구조처럼 다뤄질 수 있다. layer group policy가 shareable 여부를 정하므로 request table의 동일 ID 반복을 모든 group에서 정상 prefix sharing으로 해석하지 않는다. sliding-window group이 같은 address를 재사용하는 것은 window policy일 수 있다.

free와 allocation가 같은 iteration에 이어지면 integer ID가 즉시 재사용될 수 있다. host trace가 ref mutation 전후 generation을 갖지 않으면 old request table과 new request table이 같은 ID라 정상처럼 보인다. current owner뿐 아니라 allocation epoch를 debug metadata로 구성한다.

### 34.6.3 eager·SDPA fallback은 cache update 비용을 바꾼다

모든 attention kernel이 paged cache를 직접 읽는 것은 아니다. Transformers 문서는 일부 kernel에 direct interaction mechanism이 없어 `PagedAttentionCache.update()`로 수동 read/write하며 sequence가 길어질수록 bottleneck이 될 수 있다고 설명한다. 주소 모델은 유지되지만 gather/scatter 또는 materialization 비용이 backend마다 달라진다.

paged allocation를 쓴다는 사실과 paged attention kernel을 쓴다는 사실을 분리한다. cache manager가 block table을 관리해도 backend integration가 contiguous view를 만들 수 있다. 옵션 효과는 allocation fragmentation뿐 아니라 attention integration의 gather와 copy까지 포함한다.

### 34.6.4 special padding 주소가 실제 block과 섞이지 않게 한다

allocatable `num_blocks` 뒤 두 blocks가 padding zone이다. `read_trash_index`는 first extra block 시작, `sentinel_index`는 그 다음 element, `write_trash_index`는 second extra block 시작이다. block size 최소 validation은 first extra block 안에 special indices를 둘 공간과 연결된다.

padding query가 read trash에서 읽으면 real request block을 참조하지 않는다. write trash는 padding output KV가 실제 cache를 덮지 않게 한다. sentinel은 sliding-window group의 새 store 위치가 없음을 나타낼 수 있다. 세 index는 같은 padding zone이어도 read, write, control 의미가 다르다.

metadata builder가 padding row에 ordinary block ID 0을 넣으면 실제 cache를 읽거나 쓸 수 있다. 반대로 real token에 sentinel을 넣으면 KV update가 누락된다. batch padding mask와 special slot mapping을 함께 감사한다. allocatable range 밖이더라도 전체 tensor shape 안의 합법 index다.

TP에서는 각 rank의 `num_blocks`와 max batch tokens를 all-reduce minimum으로 맞춘다. block ID namespace는 rank-local tensor를 가리켜도 table shape와 valid range는 일치해야 한다. 한 rank만 더 큰 ID를 쓰면 logical sequence가 rank마다 다른 KV를 본다.

layer grouping은 layer-to-group mapping을 만든다. full attention group은 shareable이고 sliding group은 다른 policy를 쓸 수 있다. 같은 logical block이 group마다 다른 physical ID를 가진다면 layer index로 group을 먼저 고른다. flat table 하나로 합칠 때 group stride가 필요하다.

static address marking과 compile graph는 cache tensor base를 고정한다. runtime block table content는 바뀌어도 base pointer는 같아야 한다. cache tensor를 reallocate하고 old graph를 replay하면 table IDs가 맞아도 stale base를 읽는다. block size change를 hot option으로 추정하지 않는다.

## 34.7 llama.cpp: block table이 아니라 연속 KV cell slot을 찾는다

### 34.7.1 `find_slot()`은 cell ring의 빈 구간을 찾는다

llama.cpp v0.2.0의 unified KV cache는 per-request physical block ID table 대신 KV cells와 head search position을 관리한다. `find_slot(ubatch, cont)`는 ubatch token 수와 sequence/stream 정보를 보고 cache cell ring에서 배치가 놓일 slot을 찾는다. [고정 source](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L894-L1015)는 cell empty 여부, sequence ownership과 position을 조사한다.

이 모델에서 logical position 6을 `table[1]=2`로 번역하는 단계는 없다. chosen slot/cell index와 sequence position mapping으로 K/V tensor 위치를 만든다. fragmentation은 free cell 구간과 ring search, shift/defrag 정책에서 나타난다. paged table의 흩어진 blocks를 gather하는 문제와 동일하지 않다.

`cont` 인자는 연속 slot 요구 여부에 관여한다. restore나 fragmented state에서 잘못된 contiguous 가정은 slot을 못 찾거나 기존 cell mapping을 깨뜨릴 수 있다. block size 옵션과 연결해 설명하지 않는다. llama.cpp 주소 granularity는 이 경로에서 cell이다.

### 34.7.2 cell은 sequence ID와 position 의미를 가진다

cell debug 출력은 empty, sequence count, sequence position을 보여 준다. 하나의 cell이 여러 sequence와 관계를 가질 수 있고 sequence copy/fork semantics가 metadata에 반영된다. physical cell index만으로 logical token을 알 수 없고 position과 seq ownership을 함께 본다.

ring head는 다음 search를 빠르게 하는 힌트이며 KV state 자체와 구분된다는 header 주석이 있다. head가 틀리면 search 비용 또는 선택 순서가 달라질 수 있지만 살아 있는 cell ownership을 무시해 덮어써서는 안 된다. allocator cursor와 content metadata를 같은 state로 보지 않는다.

K-shift와 defragmentation은 cell content나 positions를 이동·조정할 수 있다. 이동 뒤 graph input과 metadata가 stale cell index를 보지 않게 ordering해야 한다. paged block table update와 개념적으로 “address mapping 변경”이라는 공통점은 있지만 자료구조와 kernel ABI는 다르다.

### 34.7.3 cell fixture로 paged 식의 부적용을 확인한다

cache cells 0–11이 있고 head search가 8이라고 하자. positions 0–2의 R cells가 physical 8–10에 놓일 수 있다. 다음 search는 ring 끝을 만나 0부터 빈 구간을 찾는다. logical position 3이 cell 0이 되어도 `[8,0]` 같은 request block table을 가진다고 해석하지 않는다. cell metadata의 sequence ID와 position이 관계를 보존한다.

fragmentation으로 cells 8,10만 비고 9가 사용 중이면 two-token contiguous ubatch는 그 구간에 놓을 수 없다. `find_slot(..., cont=true)`는 다른 연속 구간을 찾는다. `cont=false` semantics는 returned `slot_info.idxs`와 caller graph input을 읽는다. arbitrary physical pages를 table로 잇는 것과 같다고 가정하지 않는다.

여러 sequence가 cell을 공유하면 `seq_count`와 sequence set이 ownership을 나타낸다. 한 sequence를 제거해도 다른 sequence가 남으면 empty가 아니다. paged ref count와 역할이 비슷하지만 metadata와 write policy가 다르다.

ring head는 search 시작점 hint다. stale head가 full scan으로 회복 가능한지 source를 보고, cell ownership metadata corruption과 구분한다. header가 head를 KV state 자체가 아니라고 설명하는 이유다.

defrag는 live cells를 이동한다. graph가 old cell indices를 참조하는 동안 이동하면 stale address다. defrag scheduling, graph input, compute completion 사이 fence를 본다. paged table에서는 entry update로 indirection을 바꾸지만 cell model은 content movement가 직접 필요할 수 있다.

이 비교는 llama.cpp가 뒤처진 paging이라는 뜻이 아니다. cell model은 block table H2D indirection이 없지만 contiguous search와 defrag 비용을 가진다. paged model은 scattered blocks를 잇는 대신 table metadata와 kernel lookup 비용을 가진다.

### 34.7.4 비교는 주소 질문으로만 한다

네 구현에 공통 질문을 던진다. logical token position의 owner는 어디 있는가. physical storage index를 누가 allocation하는가. batch/kernel에 어떤 mapping을 전달하는가. shared/fork 뒤 writable suffix를 어떻게 분리하는가. cleanup 또는 compaction 뒤 stale address를 무엇이 막는가.

답은 다르다. vLLM은 group별 block ID list와 runner block table, SGLang은 request/token pool mapping과 backend metadata, Transformers는 block manager와 special padded cache indices, llama.cpp는 sequence-tagged cell slot이다. “모두 paging”이라는 표보다 이 차이가 실제 디버깅에 유용하다.

### 34.7.5 같은 요청을 네 주소 모델에 넣어 본다

R length 10이라는 동일 입력을 넣어도 관측 장부는 달라진다. vLLM에서는 group별 `KVCacheBlocks`가 three logical blocks를 가지며 `get_block_ids()`가 runner-facing IDs를 만든다. runner row와 sequence length가 kernel table lookup을 완성한다. allocator는 global pool IDs와 refs를 소유한다.

SGLang에서는 request pool index를 먼저 찾고 그 row의 logical token positions가 token/KV pool locations에 연결된다. radix hit가 있으면 first positions는 shared locations이고 suffix는 new locations다. paged backend가 block metadata를 요구하면 이 mapping에서 runner ABI를 만든다. request row와 packed query row를 구분한다.

Transformers에서는 `PagedAttentionCache`의 group별 key/value tensors와 `BlockManager` allocation IDs가 table을 만든다. allocatable blocks 뒤 special padding indices가 같은 flat address namespace에 있다. request future state와 block table이 current/previous IO pair를 건너갈 때 row generation을 지킨다.

llama.cpp에서는 `find_slot()`이 ubatch를 위한 cell indices를 찾고 cells가 sequence IDs와 positions를 기록한다. length 10이 세 blocks라는 전제는 없다. contiguous slot을 얻거나 ring과 fragmentation 정책에 맞는 indices가 생긴다. graph는 이 cell mapping을 읽는다.

공통 계산은 logical position과 physical owner의 관계를 묻는 데서 끝난다. paged 구현은 division/modulo와 table indirection을 명시적으로 보여 주고 cell 구현은 slot result와 per-cell position metadata로 관계를 나타낸다. 두 표현 모두 generation 없이 integer address만 기록하면 reuse 뒤 stale reference를 구분하기 어렵다.

성능 비교도 주소 작업으로 나눈다. paged table은 scattered allocation와 tail fragmentation 감소를 얻는 대신 ID metadata, H2D, kernel lookup을 낸다. cell ring은 table lookup이 없지만 적절한 free slot search, fragmented gap, defrag/shift 비용을 낸다. workload 없이 어느 쪽이 항상 우월하다고 말하지 않는다.

공유와 fork도 표현이 다르다. block model은 complete physical IDs의 ref count와 partial COW가 선명하다. cell model은 cell sequence membership과 copy/remove semantics를 읽어야 한다. 동일한 `ref_count` 필드를 찾으려 하지 말고 writable physical location에 writer가 둘 생기지 않는지 확인한다.

cleanup에서 vLLM/Transformers는 table reference removal과 pool state transition을 보고, SGLang은 request/token locations와 radix ref를 보고, llama.cpp는 sequence association 제거와 cell emptiness를 본다. 어느 모델도 logical request finish만으로 즉시 physical reuse를 단정하지 않는다. in-flight compute와 공유 owner가 남을 수 있다.

이 비교는 32장의 scheduler 종합표를 반복하지 않는다. scheduler가 누굴 선택했는지가 아니라 선택된 R의 position이 kernel storage까지 어떻게 번역되는지만 비교한다. 선택 정책이 같아도 address model과 failure surface는 달라진다.

### 34.7.6 llama.cpp cell 이동은 table update가 아니라 content relocation이다

llama.cpp cell model에서는 stale table generation 대신 stale cell mapping을 본다. Sequence R의 positions 0–2가 cells 8–10에 있고 defrag가 이를 cells 1–3으로 옮긴다고 하자. Graph input이나 backend batch가 old indices 8–10을 이미 캡처했다면 content move와 consumer 사이에 fence가 필요하다. Integer cell 8은 이후 다른 sequence의 position을 담을 수 있어 bounds 오류 없이 wrong KV가 된다.

Cell identity는 `(cell index, cell generation, sequence id set, logical position)`으로 기록한다. Shared sequence semantics가 있다면 sequence set이 둘 이상일 수 있으므로 index 중복이 곧 alias bug는 아니다. Writable update가 어느 sequence와 position을 대상으로 하는지, copy/remove 뒤 membership이 어떻게 바뀌는지를 본다. Ring head는 search hint이지 content generation이 아니다.

수치 fixture에서 cache 12 cells, R은 8–10, S는 3–5를 사용한다. Defrag plan D4는 R을 0–2로 옮기고 S를 3–5에 유지한다. Backend batch B9는 D4 전 old mapping을 snapshot했다. D4가 content copy와 metadata publish를 마친 뒤 B9가 launch되면 old indices를 읽는다. D4가 B9 completion 뒤 실행되거나 B9 metadata가 new mapping으로 rebuild되어야 한다.

`find_slot(..., cont=true)` 실패는 paged allocator fragmentation과 같은 식으로 해석하지 않는다. Free cells 합계가 4여도 largest contiguous run이 1이면 two-token ubatch가 실패할 수 있다. `cont=false`가 허용된다면 returned indices consumer가 arbitrary locations를 처리하는지 source로 확인한다. Block table이 없으므로 `[first+offset]`과 returned idx list 가운데 어느 ABI인지 구분한다.

Cancel·slot reuse도 cell provenance와 연결된다. Task A가 cells 8–10에 writer를 제출한 뒤 slot을 release하고 task B가 같은 cells generation을 얻으면 old backend completion이 B content를 덮을 수 있다. Slot state cleanup과 cell writer completion을 같은 것으로 보지 않는다. Task·slot generation, cell generation과 backend batch identity를 함께 fence한다.

Rollback은 old graph와 batch admission을 막고 content movement 및 in-flight writer를 terminalize한 뒤 cell ownership을 다시 검사한다. Sequence membership이 불명확한 cells는 empty로 표시해 즉시 재사용하지 않고 context를 폐기하는 편이 안전하다. Defrag를 끄는 것은 임시 fallback이며 fragmented search 비용과 capacity 영향을 측정한다.

이 비교는 paged 구현을 llama.cpp에 투영하지 않게 한다. vLLM·SGLang은 indirection metadata generation이 stale할 수 있고 llama.cpp는 cell content와 mapping relocation generation이 stale할 수 있다. 공통 질문은 logical position의 current physical owner와 old consumer fence다. 자료구조 이름을 통일하는 것이 아니다.

## 34.8 네 주소 사고를 first divergence에서 닫는다

### 34.8.1 wrong block ID

증상은 특정 batch compaction 뒤 token 품질이 깨지거나 다른 request 내용이 섞이고 illegal access 없이 결과만 틀리는 것이다. physical ID가 pool 범위 안이면 bounds checker도 잡지 못한다. request row, logical block, table stride, physical ID, block generation을 기록한다.

first divergence는 allocator가 wrong ID를 준 경우도 있지만 table row permutation, flat stride 계산, stale H2D metadata일 수 있다. manager의 request blocks가 맞는지 먼저 보고 scheduler output, runner host table, device table, kernel row 선택 순으로 내려간다. 가장 먼저 값이 달라진 경계가 수정 지점이다.

반증은 의도된 shared prefix다. R과 S가 같은 physical ID를 가리켜도 complete shared block이면 정상이다. token content와 logical prefix, ref count, write 여부를 확인한다. 복구는 mapping이 불명확한 worker cache를 재사용하지 않고 해당 generation을 격리한다.

### 34.8.2 COW alias

증상은 fork·beam·parallel sample 가운데 한 branch 생성이 다른 branch를 바꾸는 것이다. 두 table의 partial tail ID와 write offsets를 대조한다. complete prefix alias는 정상이고 첫 divergent writable position부터 destination이 달라야 한다.

first divergence는 incomplete block을 complete로 표시한 함수, ref-only fork를 선택한 predicate, copy completion 전 destination table publish 중 하나다. copy source와 destination 목록, valid length, event ordering을 본다. alias를 발견했다고 shared prefix 전체를 복사하는 수정은 memory 이득을 없애므로 writable tail만 고친다.

copy transaction을 stream 순서로 적는다. allocator가 destination 13을 reserve하고 copy launcher가 source 11의 valid offsets 0–1을 13에 복사한다. copy completion event 뒤 attention compute가 child table의 13을 읽는다. host table H2D와 compute stream wait도 이 dependency를 반영한다. source ref는 copy가 끝날 때까지 유지한다.

table publish를 copy enqueue 전에 해도 compute가 completion을 기다리면 안전할 수 있다. 따라서 Python mutation 순서만으로 결함을 단정하지 않고 consumer happens-before를 본다. 반대로 copy 함수 return이 nonblocking enqueue인데 source ref를 즉시 놓으면 unsafe하다.

copy length는 block capacity와 valid tail length를 구분한다. primitive가 block 전체를 복사하면 unused stale bytes도 이동하지만 valid length가 읽기를 막으면 correctness는 유지된다. 다만 bandwidth 비용은 늘어난다. 모든 layers와 K/V groups가 같은 logical mapping으로 copy됐는지 확인한다.

fork rollback도 transaction이다. child C1 destination이 성공하고 C2가 실패하면 API가 partial fork를 허용하는지 전체 취소인지 본다. 전체 취소라면 C1 destination과 shared prefix ref increments를 되돌린다. ref 일부를 놓치면 cache block이 영구 pinned된다.

### 34.8.3 partial-page length

증상은 block boundary 근처에서만 attention output이 흔들리고 sequence length가 B의 배수일 때 사라지는 것이다. R length 10, capacity 12 fixture처럼 table entry count와 valid length를 비교한다. padding/trash index를 의도적으로 쓰는 backend인지도 확인한다.

first divergence는 scheduler length, runner cumulative lengths, kernel metadata 중 처음 capacity가 valid length로 바뀐 곳이다. block content zeroing으로 증상을 가리면 stale value 대신 zero를 attention에 넣을 뿐 mask 오류는 남는다. correct length predicate를 복원한다.

### 34.8.4 stale table

증상은 async 또는 CUDA graph replay, batch row reuse에서만 wrong ID가 나타난다. host table은 최신인데 device kernel이 이전 generation을 읽을 수 있다. host mutation, H2D copy event, compute stream wait, graph replay, previous kernel completion을 잇는다.

first divergence는 buffer overwrite가 이전 consumer completion보다 앞서거나 compute가 table copy event를 기다리지 않은 지점이다. block generation guard가 output에서 오류를 잡아도 이미 wrong KV를 읽은 뒤다. table storage에 double buffer 또는 정확한 stream fence를 둔다.

네 사고는 연결될 수 있다. stale table이 freed partial block을 가리키면 COW alias처럼 보이고, wrong length가 table suffix의 stale ID를 읽으면 wrong block ID가 된다. 증상 이름을 원인으로 쓰지 않고 logical position에서 physical address까지 translation을 다시 계산한다.

### 34.8.5 하나의 장애를 네 구현에서 같은 이름으로 오진하지 않는다

운영자는 “KV 주소가 꼬였다”는 동일한 증상에서 출발할 수 있지만 첫 확인점은 구현마다 다르다. vLLM에서는 request의 group별 block IDs와 runner device table generation을 대조한다. SGLang에서는 request pool row, logical position별 token location, `ScheduleBatch` permutation을 본다. Transformers에서는 `BlockManager` owner와 block table, special trash/sentinel index를 본다. llama.cpp에서는 cell의 sequence set, position, chosen slot을 본다.

llama.cpp에서 position 6의 wrong cell을 발견하고 `logical_block=1`의 table entry를 찾으려 하면 존재하지 않는 계층에서 시간을 낭비한다. 반대로 vLLM에서 physical cache tensor를 scan해 같은 sequence positions가 연속인지 찾으면 paging의 정상적인 scatter를 corruption으로 오인한다. 공통 진단 언어는 logical position과 physical address이지 공통 자료구조 이름이 아니다.

R의 output이 block boundary에서 깨졌다고 하자. vLLM/Transformers paged path에서는 position 4 또는 8에서 table entry와 offset reset을 먼저 계산한다. SGLang token pool에서는 chunk/row mapping이 boundary에서 새 allocation 구간으로 넘어가는지 본다. llama.cpp에서는 ubatch slot search가 ring wrap 또는 fragmented 구간에서 어떤 cell indices를 반환했는지 본다. boundary라는 증상도 서로 다른 branch를 통과한다.

shared prefix 뒤 첫 private token에서만 깨지면 COW와 tail mapping을 의심한다. vLLM은 complete prefix block ref와 new tail allocation, Transformers는 complete block reference와 incomplete copy list, SGLang은 radix hit locations와 suffix allocation, llama.cpp는 sequence-sharing cell semantics와 write separation을 본다. “prefix cache를 꺼서 해결”은 원인을 좁히는 반증일 수 있으나 최종 수정은 아니다.

async를 끄면 증상이 사라지는 stale table 사건도 주소 모델별로 다르다. paged path는 host/device block table buffer generation과 stream wait를 본다. SGLang은 pool mapping tensor와 batch snapshot lifetime을 본다. llama.cpp는 defrag/shift와 graph input cell indices ordering을 본다. 전체 synchronize로 사라졌다는 결과는 ordering 가설을 강화하지만 어느 storage dependency가 빠졌는지는 더 찾아야 한다.

### 34.8.6 first divergence를 찾는 계산 장부

사건 시점 R length는 10, B=4, host table `[7,2,11]`, expected row 1, stride 4였다고 하자. expected flat table indices는 4,5,6이고 position 9는 index 6에서 physical 11을 읽어 offset 1을 쓴다. device trace가 row 1 stride 3을 썼다면 flat index 5, physical 2를 읽는다. divergence는 allocator가 아니라 stride metadata다.

다른 사건에서 host와 device table 모두 `[7,2,11]`인데 kernel은 physical 11 offset 3까지 읽었다. valid length가 12로 전달됐음을 확인한다. sequence length host state가 10이면 cumulative-length packing 또는 kernel binding에서 12로 변한 첫 지점을 찾는다. block ID를 바꾸거나 cache를 zeroing하는 것은 수정이 아니다.

COW 사건은 parent `[7,2,11]`, child도 `[7,2,11]`, parent/child valid tail 2에서 시작한다. fork 뒤 child append가 physical 11 offset 2에 write됐다면 incomplete last block을 alias한 것이다. expected child `[7,2,13]`과 copy `(11→13, valid offsets 0–1)`가 어디서 빠졌는지 본다. table은 13인데 content가 틀리면 copy/event 경계다.

stale table 사건에서는 generation을 더한다. step 20 device table row 1은 R `[7,2,11]`이고 step 21 host는 row 1을 S `[5,9]`로 바꾼다. step 20 kernel이 table을 늦게 읽는 동안 H2D가 같은 storage를 overwrite하면 R kernel이 S blocks를 본다. first divergence는 wrong owner가 아니라 double-buffer/fence 없는 metadata reuse다.

physical ID generation도 적는다. block 11 generation 8이 R tail이었고 free 뒤 generation 9로 T에 배정됐다면 숫자 11이 같은 것은 정상 재사용이다. R의 old table generation이 11을 읽으면 generation 9 T content를 보게 된다. trace에 integer ID만 남기면 “R이 원래 11을 썼다”는 사실 때문에 stale use를 정상으로 오인한다.

주소 장부는 단계마다 expected와 observed를 한 쌍으로 둔다. logical `(block,offset)`, table `(row,stride,index)`, physical `(id,generation)`, cache `(group,layer,head,component)`, validity `(seq_len,write position)`다. 처음 달라진 열이 source walk의 시작점이다. 모든 cache를 dump하지 않아도 작은 좌표가 사고를 좁힌다.

복구는 divergence 범위에 맞춘다. host table serialization만 틀렸고 allocator owner가 일관되면 해당 batch/request를 실패시키고 metadata path를 고칠 수 있다. physical block generation owner를 증명하지 못하면 worker cache를 격리한다. COW ref count가 불명확하면 shared prefix 전체가 영향을 받을 수 있어 관련 worker pool을 재사용하지 않는다.

### 34.8.7 incident A34: stale table과 COW eviction이 만난다

A34의 observation은 fork한 두 branch 중 C만 block boundary 다음 token부터 달라지고, batch compaction이 없으면 재현되지 않는다는 것이다. Manager trace에서 parent와 child는 full prefix `[7,2]`를 공유하고 child tail은 destination 13이다. Runner device row에는 `[7,2,11]`이 남아 있었다. Physical 11은 copy source였지만 copy 뒤 ref가 0이 되어 다른 request T의 generation 30으로 eviction·reuse됐다.

첫 직관은 COW copy가 실패했다는 것이다. 그러나 destination 13 hash는 reference와 일치했고 copy event도 끝났다. 문제는 child device table이 13으로 publish되지 않은 stale content generation이다. 동시에 source 11이 T에게 재사용되어 잘못된 read가 곧바로 의미 있는 다른 KV를 돌려줬다. Source가 아직 old bytes였다면 오답이 덜 보이거나 우연히 맞아 사건이 숨었을 수 있다.

Branch는 다섯 개로 나눈다. Manager가 child table에 13을 넣지 않았는가. Scheduler output이 old 11을 보냈는가. Runner host row는 13인데 H2D update가 누락됐는가. Device row는 13인데 kernel stride·row가 11을 선택했는가. Device row가 11인 동안 source eviction generation 30이 시작됐는가. 같은 batch generation의 값으로 순서대로 비교한다.

A34 ledger는 다음과 같다.

```yaml
request: C
request_generation: 8
batch_generation: 91
logical_block: 2
valid_tail: 2
manager: {physical: 13, epoch: 25}
scheduler_output: {physical: 13, epoch: 25}
runner_host: {physical: 13, content_generation: 91}
runner_device: {physical: 11, content_generation: 88}
copy: {source: "11@29", destination: "13@25", completed: true}
eviction: {physical: 11, new_owner: T, epoch: 30}
first_wrong_coordinate: {layer: 6, kv_head: 3, token: 8, component: 5}
```

Cause는 device table update transaction의 세대 누락이다. Batch compaction은 row 2의 length와 first entries를 갱신했지만 tail entry는 active block count를 old length로 계산해 copy하지 않았다. Kernel은 stale 11을 읽었다. COW와 eviction은 source content를 바꾸어 증상을 뚜렷하게 했지만 최초 mapping divergence는 runner device row였다.

Verification은 three-way matrix를 쓴다. Eviction을 막고 table update bug를 유지하면 device ID는 11이지만 old source bytes를 읽는다. Table update를 고치고 eviction을 허용하면 13을 읽어 정상이다. 둘 다 고치면 정상이어야 한다. COW를 전부 disable해 full copy하면 증상이 사라질 수 있지만 stale row update branch를 반증하지 못한다. 다른 row change에서 다시 나타날 수 있다.

Address 손계산도 확인한다. Layer 1 V, physical 13, offset 0, head 3, component 5라면 앞 stride 식의 byte offset은 `8192+4096+13*256+0+48+10=15,674`다. Stale physical 11이면 `15,162`로 512 bytes 차이다. 두 주소 모두 allocation 안이다. 이 좌표의 generation stamp가 25와 30으로 달라 first wrong value의 provenance를 증명한다.

Rollback은 graph·runner batch admission을 막고 generation 91 이전 device table consumers를 drain한다. Affected child C와 block 11 generation 30을 읽었을 수 있는 request를 ledger에서 찾는다. Cache provenance가 불명확한 entries 11과 13을 invalidate하고 table buffer를 current generation으로 다시 채운다. Allocator 전체 uniqueness가 증명되지 않으면 worker cache를 폐기한다.

수정 뒤 fixture는 row compaction, fork, partial tail COW와 immediate source eviction을 5,000회 반복한다. Manager→scheduler→host→device→kernel ID와 epoch가 일치하고 writable child tail이 parent/source와 갈라져야 한다. Event와 table generation mismatch reject가 정상 workload에서 0이며 injected stale update에서 확실히 검출되어야 한다. Correctness가 돌아와도 H2D bytes와 graph replay latency가 performance budget 안인지 본다.

종료 조건은 output parity만 아니다. Affected generation이 terminal이고, stale device rows가 0이며, source/destination ref와 eviction owner가 일관되고, current pool에서 physical ID·epoch owner가 유일해야 한다. Safe rollback path와 canary가 남아 있어야 admission을 완전히 연다.

## 34.9 소스와 관측을 연결하는 주소 감사

### 34.9.1 한 token을 네 경계에서 확인한다

R position 6을 표본으로 고른다. scheduler의 logical length와 block list에서 `(1,2)`를 계산한다. runner host metadata에서 row와 stride로 table entry physical 2를 읽는다. device metadata copy generation을 확인한다. backend kernel layout에서 layer·local head·component address 식을 확인한다.

각 단계의 값을 모두 dump할 필요는 없다. request pseudonym, position, logical block/offset, physical ID/generation, row/stride, valid length를 표본 trace로 남긴다. kernel 내부 pointer는 debug build 또는 metadata replay로 검증할 수 있다. runtime 실행은 이 집필에서 하지 않지만 필요한 관측 계약은 정적으로 설계한다.

R position 9도 확인하면 partial block predicate를 검증할 수 있다. position 10 append가 같은 physical 11 offset 2를 쓰고 position 12에서만 새 table entry가 생기는지 source mutation을 따른다. boundary의 앞·뒤·정확한 배수 세 점이 off-by-one을 잘 드러낸다.

position 3, 4도 좋은 짝이다. position 3은 logical block 0 offset 3, physical 7의 마지막 slot이다. position 4는 logical block 1 offset 0, physical 2의 첫 slot이다. output 차이가 이 경계에서만 생기면 division/modulo, table next entry, RoPE position은 각각 따로 확인한다. position encoding 오류와 cache address 오류는 같은 boundary symptom을 만들 수 있다.

write와 read를 분리해 검증한다. slot mapping이 position 4 K/V를 physical 2 offset 0에 썼는지 먼저 보고, 다음 decode attention이 같은 table entry를 읽는지 본다. write는 맞고 read가 틀리면 attention metadata/kernel path다. write부터 틀리면 runner slot mapping이나 allocation path다. final output 하나로 둘을 합치지 않는다.

layer 표본도 하나보다 두 개를 잡는다. layer 0과 마지막 layer에서 같은 logical mapping이 group별 physical table을 올바르게 선택하는지 본다. 첫 layers만 정상이고 hybrid group 경계 뒤 깨지면 group index/stride 문제다. 모든 layer가 같은 wrong block이면 request row/table 문제 가능성이 크다.

head 표본은 GQA mapping 경계를 고른다. query head group의 first/last head가 같은 KV head를 가리키는지, 다음 group에서 local KV head가 증가하는지 계산한다. TP rank별 local head mapping도 붙인다. block table이 맞아도 head shard가 틀리면 다른 vector를 읽으므로 주소 감사를 block ID에서 멈추지 않는다.

이 표본은 production payload를 전부 기록하라는 뜻이 아니다. token content 없이 coordinates와 generation으로 대부분의 owner/stride 문제를 찾을 수 있다. 실제 K/V 값 비교가 필요하면 작은 synthetic fixture와 debug environment에서 제한한다. 사용자 prompt의 KV를 장기 로그로 남기는 운영 습관은 피한다.

fixture를 자동 assertion으로 옮길 때도 결과 token 문자열만 비교하지 않는다. B=4에서 positions 3,4,7,8의 expected `(logical block, offset)`을 검증하고 request row stride에 따른 flat table index를 확인한다. COW 뒤 parent와 child의 complete prefix IDs는 같고 partial tail ID는 달라야 한다. valid length 10이면 physical last block offsets 2–3은 attention read set에 없어야 한다.

allocator ID는 실행마다 달라질 수 있으므로 `[7,2,11]` 자체를 golden value로 고정하지 않는다. IDs의 관계와 owner를 검증한다. 세 entries가 서로 valid하며 logical order를 보존하고, shared prefix request끼리 앞 entries가 같고 private tails가 다르며, freed generation이 current table에 없다는 식이다. 이렇게 해야 free queue 순서 변경에 테스트가 깨지지 않으면서 correctness invariant는 지킨다.

llama.cpp fixture는 table assertion을 쓰지 않는다. returned cell indices가 ubatch token 수와 맞고 각 cell의 sequence/position metadata가 expected logical positions를 나타내며 live unrelated cell을 덮지 않는지 본다. 동일한 test helper를 강제하기보다 공통 logical-address 질문에 구현별 assertion을 둔다.

### 34.9.2 option은 downstream shape diff로 검증한다

block size를 4에서 8로 바꾸면 동일 length 10의 logical blocks는 3에서 2가 된다. tail unused는 2에서 6으로 늘고 table entries는 줄어든다. physical cache tensor second dimension 또는 flattened page shape, max blocks per request, slot mapping division/modulo, kernel specialization도 바뀌어야 한다.

config 출력만 달라지고 runner table capacity나 kernel constant가 4로 남으면 split-brain이다. constructor validation과 memory planner, manager, static IO tensor, attention integration를 순서대로 확인한다. compile/CUDA graph가 shape를 capture했다면 engine 재생성 없이 hot change를 지원한다고 추정하지 않는다.

fragmentation 측정은 allocated block count와 valid token count에서 tail waste를 계산한다. prefix sharing은 shared physical IDs를 request별로 중복 합산하지 않는다. table memory와 cache data memory를 분리한다. 33장의 전체 byte 공식을 반복하지 않고 주소 granularity가 waste와 metadata를 어떻게 바꾸는지만 본다.

옵션 비교에서는 동일 token capacity를 유지하려는지 동일 bytes budget을 유지하려는지 분명히 한다. B 변경으로 kernel padding/alignment가 달라지면 실제 num blocks가 단순 inverse scaling하지 않을 수 있다. memory planner가 출력한 allocatable blocks와 special/reserved blocks를 기록한다. Transformers의 two padding blocks나 vLLM null block을 request capacity로 세지 않는다.

table stride는 max model length와 block size에서 `ceil(max_len/B)` 형태로 정해질 수 있지만 sliding/hybrid, capture bucket, max blocks per request option이 상한을 바꿀 수 있다. 실제 constructor와 static tensor shape를 본다. request current length로 stride를 추정하지 않는다.

kernel specialization가 B를 compile-time constant로 쓰면 unsupported size는 validation에서 막혀야 한다. generic fallback이 있다면 선택 predicate와 성능 차이를 본다. 옵션 parser가 받아들인다는 사실과 fast kernel이 실행 가능하다는 사실은 다르다. fallback이 contiguous materialization를 요구하면 address indirection 비용 모델도 달라진다.

### 34.9.3 source coordinate는 table의 생성과 소비를 모두 가져야 한다

allocator 함수만 인용하면 physical ID가 어디서 생겼는지는 알지만 kernel이 어떤 row/stride로 읽는지 모른다. runner builder와 backend binding, kernel launcher까지 source chain을 잇는다. 반대로 kernel index 식만 보면 table entry가 stale해진 이유를 scheduler에서 찾을 수 없다.

각 구현에서 allocation, mapping mutation, metadata serialization, consumer, cleanup 다섯 좌표를 잡는다. COW가 있으면 copy source/destination과 event를 추가한다. llama.cpp는 block table serialization 좌표가 없으므로 cell slot selection, ubatch mapping, graph input, cell cleanup/shift 좌표를 대신 잡는다.

source를 읽는 순서도 사건에 따라 바꾼다. 정상 주소를 처음 배울 때는 allocator에서 runner와 kernel 방향으로 내려간다. 잘못된 output을 조사할 때는 kernel이 받은 row·length·table generation에서 scheduler와 allocator로 거슬러 올라간다. COW 사건은 fork predicate와 copy pair에서 양쪽 table로 펼친다. 하나의 고정 탐색 순서를 모든 사고에 적용하지 않는다.

line anchor는 mutation이 보이는 최소 범위를 잡는다. class 선언만 링크하면 `block_size` 필드는 찾을 수 있어도 언제 table entry가 append되는지 알 수 없다. 반대로 긴 파일 전체 링크는 first divergence를 재현하기 어렵다. validator, allocation, table build, binding, free 각각에 commit-pinned 좌표를 둔다.

코드 주석과 관측을 분리한다. “append-only를 유지한다”는 block pool 주석은 특정 dedup 선택의 근거다. 모든 table update가 영원히 append-only라는 저자 의도로 확대하지 않는다. `find_slot`의 debug print는 cells를 관찰할 수 있음을 보여 주지만 production에서 항상 출력된다는 뜻은 아니다. 독자가 무엇을 코드 사실로, 무엇을 진단 설계로 읽어야 하는지 밝힌다.

주소 trace를 만들 때 raw pointer를 장기 식별자로 쓰지 않는다. caching allocator가 같은 pointer를 다른 tensor generation에 재사용하고 process마다 address가 다르다. `(pool/block 또는 cell ID, allocation generation, table generation, request generation)`이 논리 owner를 더 잘 표현한다. pointer는 같은 generation 안에서 storage alias를 확인하는 보조값이다.

표본 검증은 request content를 저장하지 않고도 가능하다. logical token ordinal과 pseudonymous request ID, block/table coordinates, length를 기록한다. prefix equality를 확인해야 하면 content hash와 source generation을 쓰고 prompt text를 그대로 남기지 않는다. 디버깅 정확성과 데이터 최소화를 함께 지킨다.

한 단계 값이 없으면 그 사실도 결과다. runner가 device table generation을 노출하지 않는다면 host table과 kernel execution 사이를 직접 증명할 수 없다. 이 gap을 “아마 H2D가 됐을 것”으로 메우지 않고 debug instrumentation 후보로 남긴다. 정확한 책은 모르는 경계를 숨기지 않는다.

**주소 통제 실험과 판정.** 실험 A는 logical block 하나의 physical block ID만 바꿔 reference no-cache logits가 달라져야 하는지 본다. 실험 B는 cancel 뒤 같은 physical block을 새 generation에 재할당하고 old completion이 page table을 수정하지 못하게 한다. lookup miss인데 block content가 남으면 hash/index를, mapping은 맞고 값이 다르면 write position·generation을 진단한다. 이 결정 트리는 miss, stale hit와 use-after-free를 다른 owner로 분리한다.

## 34.10 주소 변환을 이해했다는 기준

주소 worksheet는 여섯 칸으로 제출한다. 첫 칸은 request generation과 logical token position이다. 둘째는 block size로 나눈 logical block과 offset이다. 셋째는 request row와 table stride, table content generation이다. 넷째는 physical block ID와 allocation epoch다. 다섯째는 layer·K/V·local head·component와 byte stride다. 여섯째는 consumer batch generation, stream/event와 first observed value다. 한 칸이 없으면 다음 칸의 pointer가 맞아도 provenance를 증명할 수 없다.

P34 fixture의 정답을 다시 적어 보자. R generation 8, token 6, B=4이므로 logical block 1과 offset 2다. Runner row 1, table stride 4, int32 table byte offset 20에서 physical ID 2 epoch 9를 읽는다. Layer 1 V, local head 3, component 5의 flat byte offset은 12,986이다. Batch generation 80이 이 좌표를 소비하고, current table content generation도 80이어야 한다. 이 계산을 손으로 할 수 있어야 optimized kernel 식의 어떤 항이 무엇을 뜻하는지 검산할 수 있다.

A34 fixture는 다른 결론을 준다. Child C generation 8, token 8의 logical block 2는 manager와 scheduler에서 destination 13 epoch 25다. Runner device table은 stale source 11 epoch 30을 가리켰다. Layer 6, local head 3, component 5의 first wrong value는 address 범위 안에 있었다. 따라서 illegal access가 없다는 사실은 mapping correctness를 지지하지 않는다. First divergence는 device table content generation 88이 batch 91에서 소비된 경계다.

Observation에서 rollback까지의 짧은 기록은 다음 순서를 지킨다. Observation은 특정 boundary·fork·compaction 뒤 silent output divergence다. Branch는 allocator, scheduler table, host/device update, row stride, COW copy와 valid length다. Cause는 first mismatching ID·epoch 또는 content generation이다. Verification은 graph/eager, eviction on/off, compaction yes/no와 generation assertion matrix다. Rollback은 old consumers drain, affected blocks quarantine, table rebuild와 output canary다.

성능 counter가 사건을 대신하지 않는다. KV usage, cache hit와 eviction count는 조사 범위를 줄이지만 어느 token이 어느 physical generation을 읽었는지 말하지 않는다. Hit가 높아도 stale table이면 wrong content를 읽고, eviction이 많아도 epoch fence가 맞으면 correctness는 유지된다. Aggregate signal에서 request coordinate ledger로 내려가야 한다.

Sanitizer도 범위를 이해한다. Out-of-bounds ID나 잘못된 alignment는 잡을 수 있지만 pool 안의 stale physical ID는 합법 pointer다. Race detector가 host table mutation을 볼 수 있어도 device content generation의 의미가 맞는지는 모른다. Tool 결과가 clean이어도 ID·epoch owner mismatch fixture가 실패하면 correctness bug다. 반대로 assertion 하나가 발동했다고 kernel pointer 식이 원인이라고 단정하지 않는다.

Safe fallback은 상황별로 다르다. Graph static table update가 의심되면 eager metadata path로 affected shape를 보낼 수 있다. COW completion이 의심되면 writable tail의 safe copy와 explicit event를 쓸 수 있다. Allocator epoch가 불명확하면 cache reuse를 끄고 worker pool을 폐기해야 한다. 모든 경우 fallback이 output parity를 회복하는지와 TTFT·memory budget 비용을 함께 기록한다.

Soak 종료는 block boundary와 race를 포함한다. Lengths 3,4,5,7,8,9를 섞어 partial/full transition을 반복하고, fork 직후 child write, eviction pressure, batch row compaction과 cancellation을 주입한다. 5,000 iteration 동안 manager→device ID·epoch mismatch가 0이고, writable alias가 0이며, pool owner uniqueness와 delayed copy lease가 drain 뒤 초기 bound로 돌아와야 한다. 정상 shared prefix는 계속 hit되어야 한다.

Rollback rehearsal에서는 신규 admission을 멈추고 old batch generation을 drain한다. Device table buffers를 current row generation으로 다시 채우고, affected cache entries를 invalidate하며, unknown epoch blocks를 일반 free queue에 넣지 않는다. New canary가 known physical IDs에 고유 epoch를 얻고, old completion이 table이나 content를 mutate하지 않는지 확인한 뒤 traffic을 연다. Process restart만으로 회복했다고 원인을 fragmention으로 확정하지 않는다.

네 구현의 차이는 worksheet의 어느 칸이 필요한지를 바꾼다. vLLM은 group·block table·pool epoch, SGLang은 request pool row·token location·radix generation, Transformers는 block manager와 special padding address, llama.cpp는 cell index·sequence membership·relocation generation을 쓴다. Paged 식을 cell model에 강요하지 않고, cell relocation을 table H2D update라고 부르지 않는다.

독자가 새 backend를 만났을 때도 먼저 layout 이름을 외우지 않는다. Logical token을 physical storage로 바꾸는 indirection, storage 내부 stride, current owner generation과 consumer fence를 찾는다. Backend가 slot mapping을 미리 계산해 division과 table lookup을 감췄다면 그 producer를 찾는다. Kernel이 page table과 length를 직접 받으면 binding에서 dtype·stride·capacity를 확인한다.

주소가 맞다는 결론은 네 증거가 합쳐질 때만 강해진다. Logical coordinate가 correct하고, mapping generation이 current하며, physical content owner가 expected이고, consumer ordering이 completion 뒤여야 한다. 어느 하나만 맞으면 조용한 오염이 가능하다. 이 네 조건이 35장의 content hash·prefix identity를 논할 수 있는 출발점이다.

회귀 테스트의 golden value는 physical ID 7이나 pointer 주소가 아니다. Allocator free-queue 순서가 바뀌면 정상 구현도 다른 ID를 받을 수 있다. Golden invariant는 logical order, ID 범위, epoch uniqueness, read-only sharing과 writable separation, valid length, consumer generation이다. 이 관계를 검사하면 allocator 정책 변경에는 견고하면서 주소 correctness 퇴행은 잡는다.

Fault injection은 한 번에 한 edge를 깨뜨린다. Host table tail entry copy를 생략해 stale generation을 만들고, COW source lease를 copy completion보다 먼저 풀고, valid length를 allocated capacity로 바꾸고, row stride를 active count로 잘못 계산한다. 네 fault가 서로 다른 first divergence를 내는지 확인한다. 모두 최종 token mismatch로만 보이면 계측이 너무 늦다.

Address trace의 privacy도 관리한다. Raw prompt나 KV vector를 기본 수집하지 않고 pseudonymous request, logical ordinal, ID·epoch·stride와 짧은 content digest를 쓴다. Digest는 content equality의 완전한 증거가 아니며 collision과 key 의미는 35장에서 다룬다. 여기서는 동일 physical generation을 잘못 join하지 않게 하는 보조값이다.

Multi-rank 환경에서는 rank마다 같은 integer block ID가 존재할 수 있다. Trace를 합칠 때 rank와 cache group, device generation을 빠뜨리면 서로 다른 physical storage를 같은 block처럼 연결한다. Global token과 local KV head mapping도 함께 기록한다. 한 rank만 stale table이면 collective는 정상이어도 final attention output이 갈라질 수 있다.

운영 종료 packet에는 affected requests와 blocks의 범위를 넣는다. Stale table 11을 읽은 batch generations, epoch 30 block 11의 실제 owner T, COW destination 13을 참조한 children과 old graph consumers를 열거한다. “cache reset 후 정상”은 범위 증거가 아니다. Reset 전 공개된 output과 재사용된 generations를 찾아야 한다.

수정이 table H2D copy 범위를 늘렸다면 성능 회귀도 측정한다. 매 step copied entries, bytes, copy stream duration, graph replay wait, TTFT와 ITL을 baseline과 비교한다. 모든 row capacity를 매번 복사하는 안전 수정은 correctness를 회복하지만 long-context table bandwidth를 키울 수 있다. Dirty rows·entries만 복사하더라도 generation bitmap 자체가 current인지 검증해야 한다.

마지막 질문은 단순하다. “token 6은 어디 있는가?”에 physical ID만 답하지 말고 request generation 8, row 1, logical block 1, offset 2, block 2 epoch 9, layer/KV/head/component stride, batch generation 80과 completion edge까지 답할 수 있는가. 그 답이 있어야 wrong value를 model의 신비가 아니라 검증 가능한 주소 번역 실패로 다룰 수 있다.

같은 질문을 rollback 뒤에도 반복한다. Rebuilt table이 새 ID를 얻었다면 manager와 device, kernel consumer가 모두 새 epoch를 보는지 확인한다. Old table buffer와 graph executable이 남아 있어도 admission generation에서 선택되지 않아야 한다. Old completion이 도착하면 accounting만 닫고 current content를 바꾸지 않아야 한다. 이 canary가 없으면 재시작이나 cache reset이 증상을 잠시 가린 것인지 address contract를 고친 것인지 구분할 수 없다.

주소 감사를 팀의 공통 언어로 만들 때도 용어보다 좌표를 우선한다. Scheduler 개발자는 logical block과 table generation을, runtime 개발자는 device row와 copy event를, kernel 개발자는 consumed ID와 internal stride를, cache 개발자는 allocation epoch와 ref/COW lease를 제출한다. 네 장부가 같은 batch generation에서 연결되어야 서로 책임을 미루지 않고 first divergence를 찾는다.

이 연결은 과도한 로그를 요구하지 않는다. Boundary fixture와 오류 표본에서만 generation-bearing coordinate를 남기고 정상 hot path에는 bounded counter와 mismatch assertion을 둔다. Mismatch가 생겼을 때 주변 generation을 확장 수집한다. 정확성에 쓰이지 않는 raw pointer와 repository trivia를 쌓는 대신 owner handoff를 재구성할 최소 evidence를 보존한다.

독자는 이제 “PagedAttention이 fragmentation을 줄인다”에서 멈추지 않아야 한다. R의 position을 block size로 나누고 table row stride를 적용해 physical block을 찾은 뒤 cache layout과 head shard로 element 주소를 계산할 수 있어야 한다. partial tail의 allocated capacity와 valid length를 구분하고 shared full prefix와 writable COW tail을 판별해야 한다.

이 능력은 구현 세부 암기가 아니다. block ID가 바뀌거나 backend layout이 달라져도 논리 위치, mapping generation, physical owner, element stride라는 네 단계를 다시 세우면 새 코드를 읽을 수 있다. 반대로 제품 이름과 기본 block size만 외우면 첫 버전 변경에서 주소 사슬을 잃는다.

vLLM의 group별 `KVCacheBlocks`, global block pool, runner table은 각기 다른 owner다. SGLang의 request/token pool과 paged backend metadata는 row permutation을 함께 지켜야 한다. Transformers는 two padding blocks의 special indices와 initialized/uninitialized free state를 가진다. llama.cpp는 cell ring slot을 찾으므로 block table 식을 적용하지 않는다.

장애 조사도 같은 번역을 거꾸로 걷는다. wrong output에서 kernel row와 valid length를 보고 device table generation, host runner metadata, request mapping, allocator owner까지 올라간다. first divergence가 table stride인지 COW publish ordering인지 partial length인지 stale buffer인지 분리한다. allocator를 재시작하기 전에 어느 주소 경계가 처음 틀렸는지 증명한다.

35장에서는 full block이 어떻게 content identity를 얻어 prefix lookup과 eviction에 쓰이는지 다룬다. 여기서는 hash 식과 radix 탐색을 앞당기지 않는다. 다음 장에 넘길 질문은 하나다. physical block ID가 우연히 같은 것이 아니라 token content와 extra context가 같다는 사실을 어떤 key가 증명하며, 그 key의 lifetime이 table reference와 어떻게 연결되는가.
