# 33장. KV tensor shape와 byte를 직접 계산하기

서버가 “KV cache 1GiB를 쓴다”고 말할 때 그 1GiB는 무엇인가. request가 의미상 보유한 K/V payload일 수도 있고, block rounding 뒤 배정된 token slots일 수도 있다. backend가 미리 잡은 tensor arena, CUDA Graph pool과 process RSS를 가리킬 수도 있다. 이 숫자를 섞으면 정상 reservation을 leak으로 오진하고, logical formula만 맞춘 채 OOM을 놓친다.

이 장은 네 장부를 분리한다. model/request에서 유도한 logical payload, block/page rounding 뒤 allocated capacity, kernel이 요구하는 physical layout, process가 reserved한 memory다. 34장이 block table과 allocator lifecycle을 깊게 다루므로 여기서는 shape와 bytes가 allocation unit으로 번역되는 경계까지만 닫는다.

14장이 KV·latent·recurrent component의 의미와 shape를 소유한다면, 이 장은 그 shape에 dtype·layer·rank·rounding을 곱해 “몇 byte이며 몇 요청을 수용하는가”에 답한다. 36장의 질문은 다시 달라진다. 계산된 allocation unit이 layer group별 page·ring·state slot의 어느 주소와 generation을 갖는지는 그 장에서 닫는다.

고정 source는 vLLM `v0.27.1` commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang `v0.5.18` commit `71de97b264b04dcd514cf904003028aefe9775c8`, Transformers `v5.15.1` commit `550d7b3834670483a4df436541272c055dc364bf`, llama.cpp `v0.2.0` commit `bb4caa7540188872173c44d161602d9271386413`이다. runtime은 실행하지 않는다.

## 33.1 1GiB를 만드는 곱셈을 한 항씩 읽는다

문제 장면부터 시작하자. capacity planner는 “8k context 한 요청이 1GiB”라고 계산했는데 서버는 네 requests만 받아도 cache 부족을 보고한다. 다른 팀원은 batch padding이 원인이라 하고, 누군가는 K/V를 두 번 계산했다고 주장한다. 이때 formula를 다시 외우는 것보다 각 숫자가 logical state인지 physical capacity인지 표시하는 일이 먼저다.

KV cache를 책장에 비유하면 token position마다 모든 attention layer에 K책과 V책 한 쌍을 꽂는 셈이다. GQA는 같은 책을 여러 query 독자가 함께 읽는다. 그러나 비유는 allocation에서 멈춘다. 실제 책장은 block 단위로 빌리고, kernel은 책의 배열 순서와 alignment를 요구하며, 비어 있는 선반도 process가 계속 reserve할 수 있다.

일반적인 dense self-attention layer에서 request 하나의 logical KV payload는 `2 × L_cache × N_kv × D_head × bytes(dtype)`다. 2는 K와 V, `L_cache`는 보존한 token positions, `N_kv`는 key/value heads, `D_head`는 head dimension이다. 모든 attention layers가 같은 구조라면 transformer layer 수 `N_layer`를 곱한다.

fixture는 `N_layer=32`, `N_kv=8`, `D_head=128`, bf16 2 bytes, `L_cache=8192`다. 한 token·한 layer의 K/V는 `2×8×128×2=4096 bytes`다. 32 layers는 token당 `4096×32=131,072 bytes`, 즉 128KiB다. 8,192 tokens는 `131,072×8,192=1,073,741,824 bytes`, 정확히 1GiB다.

이 결과는 request 하나의 logical dense payload다. block rounding, alignment, scale tensors, allocator metadata와 workspace가 없다. tensor parallel에서 KV heads가 ranks에 어떻게 partition/replicate되는지도 반영하지 않았다. 1GiB를 GPU free-memory 감소와 바로 비교하지 않는다.

batch payload는 padded maximum length에 request count를 곱하지 않는다. request lengths가 1,16,17,31이면 logical cached tokens 합은 65다. dense logical payload는 token당 bytes에 65를 곱한다. physical allocator는 다른 답을 낼 수 있다.

**단위가 바뀌는 지점마다 나누어 검산하기**

1GiB 식을 한 줄 calculator에 넣으면 factor 하나를 빠뜨려도 결과가 그럴듯해 보인다. 먼저 element count와 byte count를 나눈다. token·layer당 K elements는 `8×128=1,024`, V도 1,024이므로 2,048 elements다. bf16 두 bytes를 곱하면 4,096 bytes다. 여기까지가 token 하나가 attention layer 하나에 남기는 logical payload다.

32 layers를 곱하면 131,072 bytes다. 1KiB가 1,024 bytes이므로 정확히 128KiB/token이다. 이 중간값은 운영에서 유용하다. context가 1,000 tokens 늘 때 request logical cache가 약 125MiB 늘어난다는 것을 즉시 계산할 수 있다. 정확히는 `128KiB×1,000=125MiB`다. 8,192처럼 2의 거듭제곱을 곱하면 1GiB가 된다.

decimal GB와 binary GiB도 구분한다. 1,073,741,824 bytes는 1GiB지만 약 1.074GB다. dashboard가 GB를 10진으로 표시하고 source metric이 bytes를 GiB로 바꾸면 7%가량 차이를 leak처럼 볼 수 있다. 모든 장부는 raw bytes를 기본으로 저장하고 UI에서 단위를 명시한다.

formula review에서는 dimension table을 만든다. `N_layer=32 layers`, `L=8192 tokens/request`, `N_kv=8 heads/layer`, `D=128 elements/head`, `K/V=2 tensors`, `dtype=2 bytes/element`다. 각 factor에 unit을 붙이면 결과 unit이 bytes/request로 소거된다. query heads나 batch maximum length처럼 단위는 맞지만 의미가 다른 factor를 끼우는 오류를 잡는다.

**여러 request의 logical payload**

동일 모델에서 A length 1, B 16, C 17, D 31이면 logical tokens 합 65다. token당 128KiB이므로 `65×128KiB=8,320KiB=8.125MiB`다. maximum length 31을 네 requests에 곱한 padded 계산은 124 tokens, 15.5MiB로 거의 두 배다. paged/ragged cache의 logical ownership은 request별 actual length를 합한다.

prefix sharing이 없다면 request별 logical sum이 unique semantic positions sum과 같다. A/B/C/D 중 B와 D가 first 16-token prefix를 공유하면 logical ownership entries는 여전히 각 request length를 말할 수 있지만 unique physical cached positions는 16만 한 번 저장할 수 있다. 어떤 metric이 logical references인지 unique pages인지 밝힌다.

beam search나 parallel samples는 prompt prefix를 share하고 decode suffix가 갈라질 수 있다. 단순 `batch×length`는 prefix를 중복 계산하거나 sharing을 과대 가정한다. shared prefix positions, branch-local positions와 refcount를 나눈다. 이 lifecycle은 34장이 깊게 다룬다.

### model length와 cache length를 혼동하지 않기

model maximum length 32,768은 upper contract이고 모든 request가 그만큼 cache를 즉시 사용한다는 뜻이 아니다. static cache가 max shape를 미리 allocate하면 reserved capacity 계산에는 들어갈 수 있지만 dynamic/paged logical used에는 actual computed positions만 들어간다.

### calculator 결과가 맞는지 세 방향으로 확인하기

첫 방향은 token increment다. fixture에서 accepted token 하나가 commit되면 request logical KV가 128KiB 증가해야 한다. 100 tokens 증가면 12.5MiB다. source-derived metric의 logical delta가 이 근처가 아니면 layer/head/dtype 또는 cache component 가정이 틀렸다.

둘째는 context doubling이다. dense full layers에서 length 4,096→8,192는 logical payload 512MiB→1GiB다. sliding layers가 섞이면 full linear doubling이 깨진다. observed per-layer/state spec을 합산해 어느 component만 증가했는지 본다.

셋째는 dtype/head intervention이다. bf16→fp8 raw payload는 절반, KV heads 8→4도 dense raw payload 절반이다. 둘을 함께 바꾸면 raw 1/4이지만 scales, alignment, replication과 fixed reservation 때문에 physical/process delta는 1/4이 아닐 수 있다. intervention 뒤 첫 unchanged layer를 찾는다.

이 세 sanity checks는 runtime benchmark를 요구하지 않는 algebraic cross-check다. source의 tensor shapes와 allocation records만 있어도 token dimension, dtype element size와 head axis가 formula에 들어갔는지 확인할 수 있다. latency나 fragmentation은 별도 관측이다.

### bytes/token을 concurrency로 바꾸는 마지막 계산

usable KV payload budget이 20GiB이고 token당 128KiB라면 rounding 전 theoretical token slots는 `20GiB/128KiB=163,840`이다. block 16이면 exactly 10,240 dense pages다. 이 숫자는 maximum concurrent requests가 아니다. 각 request lengths, partial-block slack, reserved prefixes와 future decode growth를 나눠야 한다.

모든 request가 length 8,192라면 raw로 20 requests지만 allocator/policy headroom과 workspace 때문에 실제 admission은 적을 수 있다. request가 length 1이라도 each one block을 차지하면 10,240 requests가 block upper bound지만 sequence slots와 scheduler limits가 먼저 막을 수 있다.

GQA가 KV heads를 절반으로 줄여 bytes/token 64KiB가 되면 same 20GiB pool은 theoretical 327,680 slots다. fixed arena memory가 그대로여도 capacity가 두 배다. GQA 미절감 incident에서 RSS 대신 blocks/token capacity를 보는 이유를 이 계산이 보여 준다.

fp8 scale 포함 bytes/token이 66KiB라면 theoretical slots는 raw half 기대보다 작다. page alignment가 16-token page를 1.0625MiB에서 1.125MiB로 round하면 finalized blocks는 또 줄어든다. 최종 concurrency 계산은 cache spec의 actual bytes/page를 사용한다.

prompt length 8,192와 generated tokens 256인 request의 self-KV cache length는 일반적으로 8,448까지 증가한다. current query token의 K/V commit timing에 따라 metric sampling 순간 one-token 차이가 있을 수 있다. accepted output count와 cache committed length의 exact relation을 implementation source에서 확인한다.

sliding window가 4,096이면 logical model position은 8,448이어도 local layer physically retained context는 최대 window 근처다. global layer는 full 8,448을 유지할 수 있다. 하나의 `cache_length` metric으로 layer별 retained length를 복원하지 않는다.

## 33.2 MHA·GQA·MQA에서 어떤 head 수를 저장하는가

“query heads를 32에서 그대로 두고 KV heads만 8로 줄였으니 GPU memory가 1/4로 내려가야 한다”는 기대는 logical payload에는 맞다. 하지만 dashboard가 fixed-size cache arena를 보여 주면 memory는 그대로이고 block count가 네 배가 될 수 있다. 먼저 어떤 숫자를 비교했는지 묻는다.

여러 query heads가 같은 K/V head를 공유하는 모습을 여러 검색 창구가 하나의 문서 보관함을 읽는 것으로 생각할 수 있다. 보관함 수는 줄지만 검색 창구의 계산은 남는다. 또한 distributed backend가 각 rank에 같은 보관함을 복제하면 건물 전체 보관함 수는 single-device 식과 달라진다.

MHA에서는 보통 `N_kv=N_q`다. query heads 32, KV heads 32면 cache가 32 heads를 저장한다. GQA에서 query heads 32, KV heads 8이면 query groups가 같은 K/V heads를 공유하므로 cache 식에는 8을 넣는다. MHA 대비 logical payload는 1/4이다.

MQA는 KV head 하나를 모든 query heads가 공유하므로 `N_kv=1`이다. query projection/attention compute가 query heads만큼 존재해도 cache에 query tensor를 저장하지 않는다. query heads를 KV heads처럼 곱하면 GQA/MQA 절감을 잃는다.

그러나 config field 하나만 보고 절감을 확정하지 않는다. effective model architecture가 `num_key_value_heads`를 어떻게 default/override하는지, tensor parallel ranks에 KV heads가 분할되는지 또는 replication되는지, backend physical layout이 alignment capacity를 얼마나 갖는지 source에서 잇는다.

GQA가 8 heads라 logical 1/4이어도 rank당 minimum head replication과 alignment 때문에 small TP configuration에서는 정확히 1/4이 아닐 수 있다. cluster aggregate bytes, rank-local bytes와 per-process reservation을 분리한다.

### MHA에서 GQA로 바뀌는 수치를 끝까지 계산하기

query heads 32, head dimension 128, 32 layers, bf16을 유지한다. MHA `N_kv=32`이면 token·layer payload는 `2×32×128×2=16,384 bytes`, 전체 layers token당 512KiB다. 8,192 tokens는 4GiB다. GQA `N_kv=8`은 앞 fixture의 1GiB, MQA `N_kv=1`은 128MiB다.

비율은 MHA:GQA:MQA가 32:8:1, 즉 32 query heads가 같아도 KV payload는 4GiB:1GiB:128MiB다. attention output dimension과 query projection은 여전히 32 query heads를 반영한다. cache memory 절감이 model parameter와 attention compute를 같은 비율로 줄인다는 뜻이 아니다.

config에서 `num_key_value_heads`가 없을 때 `num_attention_heads`를 default로 쓰면 MHA다. loader가 architecture-specific field를 읽지 못하면 intended GQA model이 effective MHA처럼 allocation될 수 있다. 반대로 backend가 KV heads를 query groups로 expand하는 것은 compute view일 수 있으며 expanded tensor를 persistent cache로 저장하는지 source에서 확인해야 한다.

### tensor parallel에서 rank-local 계산

TP=4이고 KV heads 8이 균등 partition되면 rank당 2 KV heads를 저장해 aggregate 8이다. rank-local logical payload는 single-device 1GiB의 1/4, 256MiB다. 네 process RSS를 합하면 다시 1GiB plus per-rank overhead다.

MQA `N_kv=1`, TP=4에서는 한 head를 1/4 head로 단순 나눌 수 없는 backend가 각 rank에 replicate할 수 있다. rank당 payload 128MiB, aggregate 512MiB가 되어 single-device logical 128MiB보다 4배다. 이는 오류가 아니라 collective/attention implementation의 replication tradeoff일 수 있다.

GQA heads가 TP degree보다 작거나 나누어떨어지지 않을 때도 padding/replication이 생긴다. `ceil(N_kv/TP)`만으로 단정하지 않고 effective local heads와 allocation shape를 읽는다. user-facing model config, cluster aggregate semantic payload, rank-local physical tensor를 세 열로 둔다.

TP 변경 뒤 nvidia-smi per-GPU bytes가 줄지 않았다고 GQA가 적용 안 됐다고 말할 수 없다. fixed cache budget을 allocator가 더 많은 blocks로 바꿨을 수 있다. tensor shape/bytes per block과 number of blocks를 함께 본다.

## 33.3 MLA·sliding·hybrid state는 dense 공식을 깨뜨린다

새 model의 `num_key_value_heads`를 찾지 못한 운영자가 dense GQA 식에 임의 값을 넣었다. 예상은 1GiB인데 source의 cache spec은 latent tensor와 recurrent state를 만든다. formula가 틀렸다기보다 적용 대상이 사라진 것이다. cache가 실제로 무엇을 보존하는지 component inventory부터 다시 만든다.

MLA를 압축 서류철, sliding attention을 오래된 서류를 폐기하는 회전 선반, recurrent state를 누적 요약장으로 비유할 수 있다. 하지만 압축 latent는 단순 lossless zip이 아니며 projection을 compute에서 복원하는 architecture다. sliding layer도 sink/global positions를 남길 수 있고 recurrent state는 token별 페이지처럼 증가하지 않는다.

MLA는 full K/V heads를 그대로 저장하지 않고 compressed latent와 RoPE component 같은 implementation-specific state를 보존할 수 있다. dense `2×N_kv×D_head` 식을 억지로 적용하지 않는다. cached latent dimensions, separate positional component, dtype과 layer count를 source의 cache spec에서 읽는다.

sliding-window/local attention layer는 request 전체 length가 아니라 보존 window에 가까운 length를 가질 수 있다. global layers는 full context, local layers는 capped context를 저장한다면 layer별 `L_cache(layer)`를 합한다. 모든 layers에 maximum sequence length를 곱하면 과대 계산한다.

hybrid model은 full attention, sliding attention, SSM·GDN 또는 recurrent layers가 섞인다. attention KV payload와 recurrent/convolution state의 shape·lifetime이 다르다. recurrent state를 fake token slots로 padding해 unified allocator에 넣는 구현도 있으므로 logical state bytes와 physical grouped page bytes를 분리한다.

cross-attention/multimodal encoder cache는 decoder self-KV와 length 축, reuse boundary와 lifetime이 다르다. encoder sequence length와 decoder generated length를 한 `L_cache`로 합치지 않는다. request sharing이나 repeated beams에서 ownership도 달라질 수 있다.

### MLA를 component 합으로 다시 세기

MLA cache spec이 token·layer당 compressed latent dimension `D_latent`와 RoPE key dimension `D_rope`를 저장한다고 가정한다. 논리 elements는 구현 contract에 따라 `D_latent + D_rope` 또는 별도 K/V-like components의 합이다. dense 식의 앞 factor 2와 KV heads를 자동으로 유지하지 않는다.

예를 들어 `D_latent=512`, `D_rope=64`, bf16이면 단순 component payload는 token·layer당 `576×2=1,152 bytes`다. 32 layers와 8,192 tokens라면 288MiB다. 그러나 이것은 가상 fixture다. 실제 source spec이 latent에 어떤 components와 dtype을 포함하는지 확인해야 한다. absorbed projection이 cache에 없고 compute에서 재구성되는지도 구분한다.

backend가 alignment 때문에 latent width 512를 576이나 640 stride로 잡거나 RoPE component를 별도 tensor로 둔다면 physical bytes가 달라진다. workspace에서 full heads를 일시적으로 materialize해도 persistent cache payload와 섞지 않는다. OOM peak는 둘 다 영향을 받으므로 persistent와 transient를 함께 관측하되 합계 이름을 다르게 한다.

### sliding layers를 합산하는 방법

32 layers 중 global 8, local 24이고 request logical position 8,192, local window 4,096이라고 하자. GQA token·layer K/V는 4KiB다. global payload는 `8×8,192×4KiB=256MiB`, local payload는 `24×4,096×4KiB=384MiB`, 합은 640MiB다.

모든 layers full length로 계산하면 1GiB이므로 384MiB를 과대 계산한다. 반대로 모든 layers에 window를 적용하면 512MiB로 global cache 128MiB를 누락한다. layer pattern과 per-layer cache spec이 필요한 이유다.

window 경계의 implementation은 exactly 4,096 slots보다 sink tokens, block rounding과 current token 때문에 조금 클 수 있다. logical policy length와 allocated blocks를 구분한다. model config의 sliding window 값만으로 backend capacity bytes를 확정하지 않는다.

### hybrid attention과 recurrent state 장부

attention layers에는 token-indexed K/V가 있고 SSM/GDN layers에는 request/sequence별 recurrent state와 convolution buffer가 있을 수 있다. recurrent state가 context length에 선형 증가하지 않는다면 dense token formula를 적용하는 순간 과대 계산한다. state dimension, dtype, number of recurrent layers와 active sequences를 곱한다.

예를 들어 attention payload가 request당 640MiB이고 recurrent layers state가 request당 12MiB라면 logical hybrid state는 652MiB다. unified page allocator가 attention page 크기에 맞춰 recurrent state를 padding하면 physical capacity는 더 클 수 있다. page-size grouping이 서로 다른 specs를 같은 group에 넣는지 source를 본다.

GDN/SSM state는 preemption/recompute semantics도 KV와 다를 수 있지만 lifecycle 상세는 이후 장으로 넘긴다. 이 장에서는 state component 이름, logical shape, allocation stride와 metric이 attention blocks에 포함되는지만 기록한다.

### cross-attention을 별도 lifetime으로 세기

decoder self-KV는 prompt+generated length에 따라 증가한다. encoder cross K/V는 encoder length 1,024를 한 번 project해 decoder layers가 읽는다고 하자. cross-attention layers 16, KV heads 8, head dimension 128, bf16이면 K/V payload는 `2×16×1,024×8×128×2=64MiB`다.

self-KV가 1GiB라 total persistent attention state는 1.0625GiB지만 증가율은 다르다. output token이 늘 때 self-KV만 token당 bytes만큼 증가하고 cross cache는 고정이다. request 종료 또는 encoder context 교체 때 release boundary도 다르다.

multimodal encoder feature buffer 20MiB와 projected cross K/V 64MiB가 둘 다 존재할 수 있다. feature를 projection 뒤 release할 수 있는 architecture인지, generation 동안 재사용하는지 source contract를 확인한다. 둘을 같은 `encoder cache` metric으로 중복 또는 누락하지 않는다.

## 33.4 65 logical tokens가 96 slots가 되는 과정

formula가 정확한데 capacity가 부족한 가장 작은 사례가 block rounding이다. lengths 1,16,17,31의 합만 보면 65 slots면 충분하다. allocator는 request마다 last partial block을 다른 request에게 곧장 빌려주지 못해 96 slots를 배정한다. logical equation과 allocator equation이 모두 맞으면서 31 slots 차이가 난다.

호텔 객실 비유에서 block은 방, tokens는 투숙객처럼 보인다. 한 request의 마지막 방에 빈 침대가 있어도 다른 request가 들어갈 수 없다는 직관은 맞다. 그러나 prefix sharing은 같은 방을 여러 logical tables가 참조하고 copy-on-write는 방을 복제하므로 단순 occupancy 비유는 refcount와 physical uniqueness를 설명하지 못한다.

block size 16이고 request lengths가 1,16,17,31이다. 각 request가 독립 blocks를 가진다면 allocated slots는 ceiling division으로 `16,16,32,32`다. logical 합은 `1+16+17+31=65`, physical slots 합은 96이다. 마지막 partial blocks의 빈 slots는 각각 15,0,15,1로 합 31이다.

slot utilization은 `65/96≈67.7%`다. token당 logical KV가 128KiB인 fixture라면 logical 65 slots는 8.125MiB, allocated 96 slots payload capacity는 12MiB다. 약 3.875MiB 차이는 leak이 아니라 request별 last-block rounding이다.

shared prefix가 있으면 request별 allocated block 수를 더한 값과 unique physical blocks가 다르다. refcount가 같은 prefix page를 여러 logical block table entries가 가리킬 수 있다. 반대로 copy-on-write가 발생하면 shared tail이 분리돼 unique blocks가 늘어난다. 34장이 이 lifecycle을 맡는다.

### prefix sharing 뒤 분모가 바뀌는 과정

B length 16과 D length 31이 first 16-token block을 공유하면 sharing 전 B one, D two blocks인 세 references가 unique two blocks가 된다. A/C를 더한 fixture unique total은 six에서 five blocks, capacity는 96에서 80 slots로 줄 수 있다. logical request lengths 합 65는 그대로다.

`65/80=81.25%`를 physical slot fill이라고 부르면 주의가 필요하다. shared prefix 16 positions가 logical numerator에서 B와 D에 두 번 들어간다. unique semantic occupied positions를 numerator로 쓰면 다른 값이다. logical-reference efficiency와 unique-page occupancy를 서로 다른 metric으로 둔다.

request D가 shared tail을 수정하거나 branch가 갈려 copy-on-write하면 unique blocks가 다시 늘어난다. memory jump는 refcount/COW event와 연결한다. request 종료 뒤 prefix entry가 future reuse를 위해 retained되면 active logical tokens는 줄어도 unique blocks는 남는다. eviction lifecycle은 34장에 넘긴다.

### hybrid grouping의 page padding

attention page가 block당 2MiB이고 recurrent state logical page가 request당 320KiB라 하자. common page-size group이 2MiB로 normalize하면 recurrent logical 320KiB가 physical capacity 2MiB를 차지한다. dense attention 식에 없는 큰 slack다.

separate pools는 slack를 줄이지만 attention pool이 부족할 때 recurrent free pages를 빌리지 못하는 stranded capacity를 만든다. layer specs grouping 기준과 group별 bytes/page를 source에서 찾는다. 하나의 total blocks metric으로 component별 capacity를 설명하지 않는다.

SSM state가 sequence당 고정이면 context가 늘어도 attention처럼 선형 증가하지 않는다. unified cells로 표현되더라도 logical state formula는 state dimension×dtype×recurrent layers×active sequences다. token slot count를 그대로 곱하지 않는다.

alignment는 block rounding 위에 추가될 수 있다. page 한 개의 logical bytes가 backend allocation alignment의 배수가 아니면 physical stride가 커진다. kernel layout이 K/V를 별도 buffers나 packed tensors로 둘 수도 있다.

### 65→96을 bytes와 capacity loss로 번역하기

fixture token당 payload 128KiB에서 block 하나 16 slots의 logical page capacity는 2MiB다. A/B/C/D가 각각 1,1,2,2 blocks를 받아 unique blocks 여섯 개라면 reserved payload capacity는 12MiB다. logical used는 8.125MiB, unused last-block capacity는 3.875MiB다.

utilization 67.7%를 GPU memory utilization과 혼동하지 않는다. 이것은 이 네 requests에 배정된 KV slots 중 logical used 비율이다. allocator가 total 1,000 blocks를 미리 reserve했다면 process-level reserved utilization은 또 다르다. free pool 994 blocks는 unused지만 leak도 request rounding도 아니다.

request B length 16이 token 하나를 더 받아 17이 되는 순간 새 block을 요구해 allocated slots가 16→32로 뛴다. logical bytes는 128KiB만 늘지만 capacity는 2MiB 늘어난다. OOM이 token boundary에서 불연속적으로 나타나는 이유 중 하나다. metric에는 requested new slots와 newly allocated block을 함께 둔다.

C length 17이 31까지 늘어도 allocated slots 32는 그대로다. logical used가 늘며 internal slack가 줄어든다. length 32까지 같은 blocks, 33에서 세 번째 block이 필요하다. token당 formula는 smooth growth를 말하고 allocator는 staircase growth를 만든다.

block size를 8로 바꾸면 slots는 8,16,24,32 합 80이고 slack 15다. rounding은 줄지만 block-table entries, allocation operations와 metadata가 늘 수 있다. page size tuning의 lifecycle/lookup tradeoff는 34장으로 넘기고 여기서는 payload capacity 차이만 계산한다.

## 33.5 quantized KV와 cross-attention의 숨은 항

fp8을 켜고 OOM 빈도가 줄자 memory 최적화가 성공했다고 판단했다. 며칠 뒤 long-context 답변만 틀린다. payload bytes는 줄었지만 scale state와 read/write conversion이 cache semantics의 일부가 됐다. byte 계산과 accuracy 검증을 하나의 변경에서 함께 닫아야 한다.

압축 KV를 작은 숫자와 복원용 눈금표의 조합으로 생각할 수 있다. 숫자 본체가 절반이어도 눈금표, 정렬과 임시 복원 공간이 필요하다. 비유의 한계는 scale이 단일 표가 아닐 수 있다는 점이다. layer/head/token/group마다 다른 scale index가 physical page layout과 정확히 맞아야 한다.

bf16 2 bytes를 fp8 1 byte로 바꾸면 앞의 1GiB pure payload는 산술상 512MiB다. 그러나 scale tensor가 head/token/group 단위로 추가되고 alignment와 conversion workspace가 있다. backend가 일부 layers만 fp8을 지원하거나 fallback하면 실제 절감은 정확히 50%가 아니다.

scale granularity는 bytes와 accuracy를 함께 바꾼다. per-tensor scale은 작지만 dynamic range가 넓은 heads/layers에서 error가 클 수 있다. per-head/per-group scale은 accuracy를 지킬 수 있지만 metadata와 load가 늘어난다. write quantization과 read dequantization이 같은 scale indexing을 써야 한다.

### scale bytes를 payload 식에 넣기

fp8 raw payload가 token·layer당 K/V 2,048 elements라면 2,048 bytes다. per-token-per-head scale이면 K 8개와 V 8개 scale을 fp32 4 bytes로 저장해 64 bytes가 추가된다. raw 대비 3.125%다. 32 layers, 8,192 tokens에서는 `64×32×8,192=16MiB`이므로 512MiB raw와 합쳐 528MiB다.

per-group scale에서 group size 64 elements면 2,048 elements에 32 groups가 필요하다. fp16 scale 2 bytes면 역시 64 bytes다. zero point와 inverse scale을 함께 보존하면 더 커진다. 별도 scale tensor가 alignment 단위로 round되면 logical metadata보다 physical allocation이 더 크다.

static per-layer scale은 token length에 선형 증가하지 않지만 dynamic per-token scale은 KV positions와 lifetime을 공유한다. formula에서 fixed metadata와 per-token bytes를 나눈다. 동일 fp8 이름이라도 granularity, scale dtype과 K/V shared 여부가 달라 bytes가 다르다.

### conversion workspace와 fallback layers

attention kernel이 fp8 cache를 직접 읽지 못하면 bf16 dequantization staging을 쓸 수 있다. persistent payload는 528MiB여도 context chunk에 비례한 transient workspace가 더해진다. graph capture가 workspace를 private pool에 유지하면 request 종료 뒤 process reservation이 내려오지 않는다.

32 layers 중 24만 fp8, 8은 bf16 fallback이면 scale 전 payload는 full bf16 1GiB의 `24/32×1/2 + 8/32×1 = 62.5%`, 즉 640MiB다. global config flag만 보고 512MiB를 기대하면 128MiB 차이가 난다. effective layer spec dtype list에서 first fallback을 찾는다.

write quantization도 projection output에서 scale reduction과 conversion scratch를 쓸 수 있다. peak OOM 분석은 persistent, write workspace, read/dequant workspace와 graph pool을 나눈다. cache bytes metric에 workspace를 넣지 않되 process peak 장부에는 포함한다.

### accuracy가 scale indexing에서 갈리는 장면

token 257 K head 3이 write scale index `(layer,token,head)`를 사용했는데 read kernel이 `(layer,head,token-group)` stride로 해석하면 shapes와 bytes는 맞는다. crash 없이 attention만 틀어진다. output text보다 layer별 dequantized K/V와 fp16 reference의 first divergence를 본다.

write 전 projection은 맞고 stored fp8 code/scale부터 다르면 quantization write path다. stored values는 맞지만 reconstructed vector가 다르면 read scale indexing이나 conversion dtype이다. short context는 맞고 block 첫 token 257부터 깨지면 일반 quantization noise보다 page/scale stride가 강한 후보다.

weight quantization format과 KV cache dtype을 혼동하지 않는다. GGUF weight가 quantized됐어도 runtime KV가 fp16일 수 있고, model weights가 bf16이어도 KV만 fp8일 수 있다. config→cache spec→allocation dtype→kernel read/write path를 연결한다.

cross-attention cache는 encoder K/V를 한 번 만들고 many decoder steps에서 읽을 수 있다. decoder self-KV처럼 매 output token 증가하지 않을 수 있지만 encoder length와 layers에 따라 큰 고정 payload다. multimodal feature cache와 projected cross K/V를 중복 계산하지 않도록 source ownership을 확인한다.

## 33.6 네 구현에서 shape가 allocation으로 내려가는 길

네 codebase에서 `kv_cache` 검색 결과를 모두 읽지 않는다. model-derived dimensions가 cache spec을 만들고, spec bytes가 capacity count를 만들며, runner/backend가 tensor를 allocate하고, metric이 used/reserved unit을 노출하는 한 방향을 잇는다. 그다음 metric에서 역방향으로 올라가 denominator를 확인한다.

logical shape가 `[K/V,layers,blocks,block_size,heads,dim]`처럼 보이더라도 physical tensor axes가 같다는 뜻은 아니다. K/V separate buffers, layer-first, token-major, vector-packed와 transposed key layout이 가능하다. element-count baseline과 kernel stride를 분리한다.

### 같은 request R로 네 stack을 왕복하기

R은 GQA fixture의 8,192-token request다. 먼저 finalized model attributes에서 layers 32, local/effective KV heads, head dimension과 cache dtype을 적는다. user config가 아니라 cache constructor가 실제 소비하는 값을 사용한다. 이 네 값으로 token bytes 128KiB baseline을 만든다.

vLLM에서는 layer별 cache spec subclass와 page-size property를 찾는다. dense, sliding, MLA specs가 같은 식을 공유하는지 override하는지 본다. block size 16이면 expected dense page는 2MiB다. utils가 available bytes를 page bytes로 floor-divide하고 spec groups를 normalize하는 지점을 잇는다.

runner가 tensors를 allocate할 때 group/layer mapping, selected dtype, total block axis와 backend를 기록한다. `numel×element_size` 합이 spec bytes×blocks와 다르면 K/V packing, alignment, scales, common group page 또는 extra buffer를 좁힌다. attention layer가 올바른 cache tensor view를 받는 identity도 확인한다.

SGLang에서는 scheduler의 finalized total-token capacity가 어느 memory pool slots를 뜻하는지 먼저 본다. request pool capacity와 token-to-KV pool capacity를 혼동하지 않는다. memory pool constructor가 layer, token slot, local heads, dimensions와 dtype을 사용해 K/V buffers를 만드는 경로를 잇는다.

radix/prefix cache node는 logical prefix tree를 소유하지만 values는 token-pool indices일 수 있다. node token references와 unique occupied KV slots를 나눈다. MLA factory가 dense pool 대신 latent pool을 고르면 R의 128KiB/token baseline을 폐기하고 latent component shape로 다시 계산한다.

Transformers에서는 effective cache implementation을 먼저 확정한다. DynamicCache는 actual sequence tensor가 grow하는 lifetime, StaticCache는 maximum shape reservation, continuous paged handler는 shared pool blocks를 갖는다. 동일 model config라도 어느 constructor가 선택됐는지에 따라 used/reserved 의미가 달라진다.

model config의 KV heads와 head dimension이 cache class에 전달되는지, explicit head dimension과 hidden-size-derived 값 중 무엇을 쓰는지 본다. heterogeneous layer groups가 common page size를 쓰면 per-layer logical sum과 group-normalized physical sum을 따로 계산한다. exporter가 없다면 metric을 발명하지 않고 object fields/tensor shapes를 관측 제안으로 남긴다.

llama.cpp에서는 architecture-specific effective K/V embedding dimensions와 cache K/V `ggml_type`을 constructor에서 읽는다. query heads로 수동 유도한 값보다 constructor field가 allocation baseline이다. K와 V widths/types가 다르면 factor 2×same width shortcut을 버리고 각각 합한다.

cell axis capacity와 active used cells를 분리한다. sequence removal 뒤 cells metadata가 free돼도 backend tensor buffer는 context lifetime 동안 유지된다. CPU/GPU offload로 buffers가 나뉘면 device별 bytes를 합하고 pinned staging은 별도 항으로 둔다. recurrent factory가 선택되면 KV dense branch에서 빠져 state tensors를 다시 센다.

네 stack에서 같은 R의 표는 finalized dimensions, cache implementation/spec, bytes per page/cell unit, total units, tensor allocation bytes, logical used units, unique used units와 reserved arena를 열로 갖는다. blocks, token slots, pages와 cells라는 이름은 다르지만 raw bytes와 ownership lifetime으로 비교한다.

### source가 증명하는 것과 측정이 증명하는 것

constructor의 tensor shape product와 config-to-spec mutation은 source로 증명할 수 있다. allocator fragmentation, CUDA driver reservation과 actual kernel workspace peak는 실행 관측 없이 확정하지 않는다. source-derived number는 expected baseline이지 benchmark 결과가 아니다.

반대로 nvidia-smi delta만으로 local KV heads나 scale granularity를 역산하지 않는다. weights, graph pools와 other requests가 섞인다. backend tensor allocation metric 또는 object shapes로 cache component를 격리한 뒤 process reservation과 reconcile한다.

고정 commit line link는 주장별로 둔다. model dimension source, spec page bytes, capacity calculation, tensor allocation, metric/exporter가 서로 다른 증거다. broad file link 하나로 전체 chain을 증명하지 않는다. version upgrade 때 symbol inputs와 mutations가 유지되는지 diff한다.

vLLM은 [`vllm/v1/kv_cache_interface.py:1-420`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/kv_cache_interface.py#L1-L420)에서 KV cache spec, tensor shape와 page-size byte 계산을 찾는다. [`vllm/v1/core/kv_cache_utils.py:1-500`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_utils.py#L1-L500)은 layer specs grouping과 available memory 기반 block count를 잇는다.

[`vllm/v1/core/kv_cache_manager.py:1-520`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_manager.py#L1-L520)는 logical request positions를 blocks/free pool과 연결한다. 34장의 allocator lifecycle을 복제하지 않고 이 장에서는 spec의 page bytes가 manager accounting unit이 되는 경계까지만 본다.

SGLang은 scheduler effective `max_total_num_tokens`, page size와 token/request pool capacity를 확인한 뒤 memory pool의 layer/head/dtype shape와 K/V buffers로 내려간다. [`memory_pool.py:1-520`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/memory_pool.py#L1-L520)에서 token slots와 per-layer allocation을 찾는다.

Transformers는 [`continuous_batching/cache.py:1-520`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache.py#L1-L520)의 paged memory handler, layer grouping, footprint와 allocation을 잇는다. classic DynamicCache/StaticCache와 continuous paged cache의 lifetime을 같은 RSS 기대치로 비교하지 않는다.

llama.cpp는 [`src/llama-kv-cache.cpp:1-520`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L1-L520)에서 layer별 K/V tensor type, dimensions와 backend buffer allocation을 읽는다. unified cells, recurrent/hybrid factory와 sequence ownership을 구분한다.

**vLLM: model config에서 page bytes까지**

source walk는 model의 number of layers, attention heads, KV heads, head dimension과 cache dtype을 얻는 call에서 시작한다. cache dtype이 `auto`라면 weight dtype 문자열을 그대로 bytes로 바꾸지 않고 finalized cache dtype을 찾는다. MLA, sliding 또는 hybrid model이면 dense full-attention spec 하나가 아니라 layer별 spec factory 결과를 본다.

`kv_cache_interface.py`에서는 spec이 page/block 하나의 logical shape와 byte size를 어떻게 계산하는지 읽는다. page bytes가 K/V factor와 block size, local KV heads, head size와 element size를 포함하는지 각 항을 fixture에 대응한다. spec의 Python shape는 attention backend가 tensor를 보는 logical contract일 수 있으며 kernel 내부 vectorized layout과 동일하다고 단정하지 않는다.

`kv_cache_utils.py`는 available memory를 page size로 나누어 block count를 정하고 서로 다른 layer specs를 grouping할 수 있다. available bytes 전부가 cache payload가 되는지, alignment/group common page size와 profiling reservation을 빼는지 확인한다. page bytes가 맞는데 total capacity가 예상과 다르면 block-count numerator와 grouping을 본다.

manager는 request length를 block count로 ceiling-round한다. 이 장은 A/B/C/D가 six blocks를 요구한다는 계산까지만 manager에 연결한다. free list, prefix refcount와 eviction 순서는 34장에 맡긴다. metric에서 allocated logical blocks와 physical unique blocks를 구별할 필드가 있는지 기록한다.

runner allocation에서는 total blocks와 per-layer shape가 실제 device buffers가 되는 지점을 찾는다. K와 V separate tensors인지 packed leading dimension인지, dtype과 device/backend가 무엇인지 본다. allocation bytes와 spec page bytes×blocks가 다르면 alignment, grouping, scales 또는 hidden workspace를 좁힌다.

**SGLang: token pool capacity에서 layer buffers까지**

SGLang scheduler의 `max_total_num_tokens`는 사용자 request lengths 합과 같지 않고 token pool capacity 상한이다. page size가 1보다 크다면 effective token capacity와 page rounding을 구분한다. auto-tuned capacity가 memory fraction/profile 결과로 바뀔 수 있으므로 requested maximum과 finalized pool size를 함께 기록한다.

memory pool에서는 K/V buffer shapes의 layer, token slot, local head와 head dimension axes를 찾는다. code의 axis order가 `[layer,token,head,dim]`인지 다른 order인지는 logical element count에 영향을 주지 않지만 kernel stride와 physical layout 해석에는 중요하다. shape product×dtype bytes와 allocator-reported bytes를 따로 계산한다.

radix/prefix cache는 logical token prefix ownership을 pool slot indices와 연결한다. request별 prefix length를 더하면 shared slots를 중복 셀 수 있다. token pool used slots와 radix nodes의 logical references가 다른 denominator다. source review에서 scheduler capacity metric을 unique physical slots로 오해하지 않는다.

MLA backend는 dense K/V pool과 다른 latent buffers를 만들 수 있다. model config가 MLA인데 dense GQA 식과 memory-pool class를 따라가면 처음부터 잘못된 branch다. factory/dispatcher가 어떤 cache class를 선택했는지 effective state로 남긴다.

**Transformers: classic cache와 continuous paged cache**

classic `DynamicCache`는 generation이 진행되며 layer tensors를 append/grow하는 의미를 갖고, `StaticCache`는 configured maximum shape를 미리 준비할 수 있다. continuous paged memory handler는 many request lifetimes가 shared pool blocks를 사용한다. 같은 model shape라도 reserved/used 변화가 다르다.

`PagedAttentionMemoryHandler`의 memory-footprint 식에 layer grouping, KV heads, head dimension, dtype와 block size가 어떻게 들어가는지 1GiB fixture와 대조한다. sliding/global layer groups가 separate capacities를 갖는지 common maximum page로 padding되는지 본다. request cache object가 block IDs와 logical length를 어떻게 보유하는지는 다음 장 경계까지만 연결한다.

classic cache로 실행한 RSS curve와 continuous cache pool RSS를 직접 비교해 leak을 판단하지 않는다. former는 request lifetime에 grow/release될 수 있고 latter는 process lifetime arena를 유지한다. logical output parity와 memory ownership contract가 각각 다르다.

**llama.cpp: cells와 backend tensors를 분리해서 읽기**

llama.cpp KV cache constructor는 model layers를 순회해 K/V tensor types와 dimensions를 정하고 backend buffers를 할당한다. context/cell capacity와 tensor element count를 연결한다. layer마다 attention/recurrent 특성이 다르면 K/V tensor가 없거나 다른 state factory가 선택될 수 있다.

unified KV cell count는 sequence positions가 차지할 logical cells의 pool이다. tensor shape의 cell axis capacity와 active sequence ownership을 구분한다. cells가 free돼도 backend buffer는 context lifetime 동안 유지될 수 있으므로 process memory가 내려가지 않는 것이 정상일 수 있다.

GGUF model weight type은 KV tensor type의 증거가 아니다. cache K/V type configuration과 constructor가 선택한 `ggml_type`, backend buffer byte size를 본다. K와 V가 서로 다른 types/layout을 가질 가능성도 dense `2×same dtype` shortcut 전에 확인한다.

**네 source walk를 같은 표로 마무리하기**

공통 열은 model-derived dimensions, effective cache implementation/dtype, layer spec, page/cell bytes, total reserved pages/cells, tensor allocation shape, logical used units와 exposed metric이다. 한 구현에서 열이 비면 0이라고 쓰지 않고 source를 더 따라간다.

각 열에는 value뿐 아니라 unit을 적는다. KV heads는 global인지 rank-local인지, token capacity는 logical refs인지 unique slots인지, bytes는 payload인지 allocated tensor인지 표시한다. 이 표가 있어야 vLLM blocks, SGLang token pool, Transformers pages와 llama.cpp cells를 같은 숫자로 잘못 비교하지 않는다.

**vLLM metric 계보를 page 식에 붙이기**

cache spec이 한 page의 bytes를 만들고 utils가 profiled available memory로 number of blocks를 정하면 reserved payload baseline은 page bytes×blocks다. manager의 request tables는 이 pool 가운데 logical ownership을 표현한다. metric이 GPU cache usage ratio를 내보낸다면 numerator가 used blocks인지 free blocks의 complement인지 source exporter까지 찾는다.

request A length 1이 block size 16에서 한 block을 얻으면 logical bytes 128KiB지만 used-block metric은 page 2MiB 하나를 센다. usage ratio를 logical token utilization로 읽으면 16배 차이다. cache hit metric은 shared prefix reuse를 뜻할 수 있지만 bytes saved metric과 동일하지 않다.

available memory 기반 block 계산은 weight, activation profiling과 safety margin 뒤의 remainder를 쓸 수 있다. nvidia-smi total free를 page bytes로 직접 나눈 theoretical blocks와 finalized blocks가 다른 것이 정상이다. profiling 결과, cache memory budget, page bytes와 blocks를 한 startup record에 둔다.

hybrid layer specs가 group common page size를 사용하면 each group의 pages와 per-page layer contribution을 확인한다. global total block count 하나에 dense token bytes를 곱하면 recurrent/sliding group padding을 놓친다. spec group별 expected allocation과 actual tensor bytes를 합한다.

**SGLang token capacity metric을 physical slots에 붙이기**

`max_total_num_tokens`가 100,000이라면 그것이 request logical lengths sum hard limit인지 token pool slot count인지 확인한다. page size와 allocator rounding이 있으면 active logical 65가 used pool slots 96에 대응할 수 있다. scheduler admission은 capacity 단위로 판단하고 radix cache는 shared ownership을 별도로 갖는다.

memory pool의 allocation shape product는 layer count×pool slots×local heads×head dimension×dtype plus K/V다. MLA pool이면 axes가 latent dimensions로 바뀐다. exporter의 used tokens가 pool indices allocated를 세는지 active request tokens를 세는지 source에서 label/field를 따라간다.

radix prefix가 request 없이 retained되면 scheduler running tokens는 0이어도 pool used slots가 남는다. leak을 주장하기 전에 radix cache size, evictable entries와 refcounts를 본다. 34장이 eviction action을 다루고 이 장은 retained slots의 bytes만 계산한다.

**Transformers memory footprint와 cache class 선택**

model config가 same GQA shape라도 classic DynamicCache, StaticCache와 paged handler는 allocation moment가 다르다. DynamicCache tensor lengths는 generation과 함께 커질 수 있고 StaticCache는 maximum capacity를 미리 잡을 수 있으며 paged handler는 process pool을 request들이 공유한다.

memory handler footprint 식이 bytes/token을 계산할 때 layer groups와 local KV heads를 소비하는지 본다. configured maximum batch/sequence shape를 곱하는 static path와 blocks×page bytes를 쓰는 paged path를 섞지 않는다. 같은 API request라도 cache implementation effective selection이 denominator를 바꾼다.

metrics가 없다면 source-derived expected tensor shapes와 allocator object fields가 baseline이다. document에 metric이 있다고 가정해 이름을 발명하지 않는다. expose되지 않은 값은 tracing/logging proposal로 분리한다.

**llama.cpp cells와 buffer bytes를 연결하기**

llama.cpp context의 KV cell capacity는 logical positions를 담는 address space다. layer별 K/V tensors는 cell axis capacity에 head/embedding dimensions와 types를 곱해 backend buffer에 놓인다. cells used/free와 backend buffer bytes는 서로 다른 lifetime이다.

sequence removal 뒤 cells metadata가 free돼도 buffer allocation은 context가 살아 있는 동안 유지된다. RSS 사건 2와 같은 정상 retention이다. recurrent models에서는 cell factory가 recurrent state path를 선택할 수 있으므로 모든 layers에 K/V factor 2를 적용하지 않는다.

backend split/offload가 있으면 layer tensors가 CPU/GPU buffers에 나뉠 수 있다. GPU memory만 보고 total cache bytes를 계산하면 host portion을 누락한다. backend buffer별 device, bytes와 layer range를 합하고 pinned staging은 별도 transient/reserved 항으로 둔다.

### metric reconciliation을 실제 숫자로 수행하기

expected page 2MiB, total 512 blocks라면 KV tensor arena payload baseline은 1GiB다. active requests logical 5,000 tokens는 625MiB이고 block rounding used 330 blocks라면 allocated payload capacity는 660MiB다. free blocks 182는 364MiB capacity다. used+free가 total 512 blocks로 맞는다.

backend reports 1.02GiB라면 24MiB 차이를 즉시 leak이라 하지 않는다. scale tensors, alignment와 spec-group metadata를 합한다. process reserved 1.3GiB라면 추가 280MiB는 graph/workspace/general allocator를 분해한다. 네 합계가 어떤 lifetime에 속하는지 기록한다.

request 종료 뒤 logical 0, used 0, free 512, backend tensor 1.02GiB, process reserved 1.3GiB가 될 수 있다. 이 한 snapshot은 pool contract와 일치한다. 이후 동일 workload에서 used/free만 왕복하고 backend/process가 stable하면 정상 재사용이다.

반대로 used+free가 510으로 줄거나 backend tensors가 iteration마다 늘면 first divergence가 있다. block accounting gap인지 extra tensor allocation인지 source owner를 찾는다. raw bytes와 units가 있어야 두 문제를 구분한다.

reconciliation은 한 시점 snapshot보다 사건 전후 delta가 강하다. request A length 1을 admission했을 때 logical +128KiB, used blocks +1, free blocks -1, backend arena delta 0이어야 pool contract와 맞는다. A가 positions 2…16으로 자라는 동안 logical은 token마다 +128KiB지만 used/free block delta는 0이다. position 17에서 used +1, free -1이 된다.

A 종료 시 logical은 -17×128KiB, used blocks -2, free +2이고 backend arena는 유지될 수 있다. prefix retention이 있으면 logical active는 줄어도 used unique blocks가 모두 반환되지 않는다. retained reason/refcount가 있어야 한다. 이 delta table은 rounding과 lifecycle을 34장에 넘기는 정확한 접점이다.

quantized path에서는 logical delta가 fp8 payload plus scale bytes다. backend arena delta는 초기 allocation 뒤 0이고 conversion workspace high-water mark만 처음 증가할 수 있다. long context boundary에서 accuracy가 깨져도 bytes deltas가 정상이라면 scale mapping correctness branch로 이동한다.

four-stack 모두에서 같은 delta 질문을 던진다. 새 logical token이 어느 cache component bytes를 늘리는가, 어느 allocation unit을 처음 넘을 때 physical used가 증가하는가, release가 free capacity와 arena reservation에 각각 무엇을 하는가다. object 이름이 달라도 이 세 변화는 비교 가능하다.

delta가 예상과 다를 때는 가장 바깥 RSS부터 추측하지 않는다. model-derived token bytes가 맞는지, effective spec page bytes가 맞는지, used-unit transition이 block boundary에서 일어났는지, arena가 재사용되는지 순서대로 확인한다. first divergence 뒤 단계는 결과이고 앞 단계는 이미 반증된 조건이다.

이 방식은 monitoring label 설계도 단순하게 만든다. component, spec group, unit kind, dtype와 device처럼 제한된 차원으로 aggregate하고 request identity는 trace에 둔다. bytes counter에는 logical, allocated-capacity, tensor-reserved와 process-reserved 접두를 붙여 독자가 denominator를 이름만으로 알게 한다.

장애 중에는 source-derived expectation과 observed snapshot을 같은 단위로 변환한 worksheet를 보존한다. 수정 뒤 수치가 맞아도 output parity와 reuse cycle을 다시 확인한다. memory accounting fix가 kernel layout을 깨거나 metric만 고쳐 실제 allocation을 숨기지 않게 한다.

계산, allocation, reuse와 correctness 네 gate가 함께 닫혀야 복구다. 어느 하나라도 빠지면 숫자만 맞춘 수정일 수 있다.

마지막으로 source fact와 inference를 분리한다. constructor shape product는 source로 증명할 수 있다. allocator fragmentation이나 actual peak는 runtime 관측 없이는 단정하지 않는다. 이 장의 static fixture는 expected baseline과 first-divergence 위치를 주며 benchmark를 가장하지 않는다.

version upgrade에서는 option 이름보다 이 공통 열을 다시 채운다. cache factory가 바뀌었는지, local heads와 dtype이 달라졌는지, page-size 식과 tensor shape product가 같은지 비교한다. metric 이름이 유지돼도 denominator가 blocks에서 token slots로 바뀔 수 있으므로 exporter source도 확인한다.

독자는 한 구현의 숫자를 다른 구현의 knob로 곧장 번역하지 않는다. 동일 1GiB logical workload라도 fixed arena fraction, block size, shared prefix와 local-head replication이 달라 reserved capacity와 maximum concurrency가 달라진다. 비교의 공통 바닥은 raw dimensions와 bytes다.

## 33.7 네 장애에서 first divergence를 찾는다

첫 장애는 계산보다 실제 cache tensor bytes가 2배인 경우다. K/V factor를 빠뜨렸는지, layer 수를 두 번 곱했는지, dtype bytes가 effective allocation과 같은지, TP rank마다 heads가 replicated됐는지 순서대로 본다. first divergence는 formula input, spec page bytes, tensor shape 중 처음 달라지는 지점이다.

둘째는 request 종료 뒤 used blocks가 줄지만 process RSS가 그대로인 경우다. pool이 다음 request를 위해 arena를 reserved한 정상 동작일 수 있다. free block count가 회복되고 same pool에서 reuse되면 RSS 유지 자체는 leak 증거가 아니다. unique live blocks, reserved arena와 external allocator bytes를 나눈다.

셋째는 GQA heads를 줄였는데 memory가 안 줄어드는 경우다. effective model config가 여전히 MHA인지, loader가 architecture default를 덮었는지, TP replication/alignment가 절감을 삼켰는지, metric이 reserved capacity만 보는지 확인한다. config 파일 숫자와 tensor allocation shape 사이 first divergence를 찾는다.

넷째는 fp8 KV 뒤 OOM은 줄었지만 출력이 틀리는 경우다. layer별 write/read scale, granularity/index, fallback dtype과 dequantization path를 비교한다. greedy short fixture에서 first divergent layer/token을 찾고 fp16 KV parity와 반증한다. memory 절감이 correctness gate를 대신하지 않는다.

### 사건 1: formula보다 tensor allocation이 정확히 두 배다

GQA fixture로 logical payload 1GiB를 예상했는데 backend KV tensors 합은 2GiB다. process RSS는 더 크지만 먼저 exact 2×인 tensor shape를 조사한다. 경쟁 가설은 K/V factor 누락, dtype 4 bytes, layers duplicate, TP replication, alignment와 double buffering이다.

worksheet에는 K/V factor 2가 이미 있고 allocation dtype도 bf16이다. layer별 bytes가 expected 32MiB가 아니라 64MiB로 모든 layers에서 동일하다. tensor shape의 local KV-head axis가 8 expected가 아니라 16이다. first divergence는 model-derived global heads에서 cache spec local capacity를 만들 때 alignment/replication으로 16이 된 지점이다.

kernel contract가 16-head stride를 요구하면 오류가 아니라 physical overhead다. 기록을 logical global heads 8, rank-local physical capacity 16으로 고친다. 불필요 expansion이면 spec factory를 수정하고 attention parity와 allocation bytes를 함께 검증한다. ratio만 보고 K/V를 두 번 저장했다고 단정하지 않는다.

다른 2× 사례는 old/new cache double buffer일 수 있다. 두 allocation call의 lifetime과 active pointer를 본다. migration이나 graph address stability에 두 buffers가 필요하면 logical 1GiB와 physical reserved 2GiB를 분리한다. 한 buffer가 영원히 unreachable이면 그때 leak 후보다.

### 사건 2: request 종료 뒤 RSS가 그대로다

active request 0, logical tokens 0, used unique blocks 0인데 process memory는 peak다. free blocks는 total capacity까지 회복됐다. 첫 가설은 leak이지만 paged pool이 initialization/profile 뒤 maximum tensor arena를 process lifetime 동안 유지할 수 있다.

same workload를 다시 넣었을 때 external allocation 증가 없이 free blocks가 재사용되고 reserved peak가 안정되면 retention contract다. 반복마다 total-free gap이나 non-pool allocations가 증가하면 leak 가능성이 남는다. RSS가 안 내려온다는 symptom 하나는 둘을 구분하지 못한다.

backend KV tensor, graph private pool, general caching allocator와 host pinned staging을 나눈다. request free는 block IDs를 pool에 돌려도 tensors를 driver에 반환하지 않는다. 정상이라면 dashboard를 used/reserved로 고친다. 실제 leak이면 finished block table, retained prefix refcount, deferred event와 cache group handle 중 first unreleased owner를 찾는다.

### 사건 3: GQA head 수를 줄였는데 memory가 그대로다

MHA 32 heads에서 GQA 8 heads로 바꿨지만 reserved arena는 800MiB로 같다. bytes per page와 pages count를 먼저 나눈다. 이전 page 8MiB×100, 새 page 2MiB×400이면 reservation은 같아도 token capacity가 4배다. 절감이 RSS가 아니라 concurrency로 전환됐다.

page bytes도 같다면 effective config→cache spec을 따라간다. loader가 architecture field를 못 읽어 KV heads를 query heads로 default했을 수 있다. 또는 TP replication/alignment floor가 local shape를 그대로 유지했을 수 있다. config file 숫자보다 finalized local tensor shape가 증거다.

복구 성공은 RSS 감소만이 아니다. effective local KV heads, page bytes, block count, total token capacity와 output parity를 본다. fixed memory fraction policy에서는 page bytes가 줄고 block count가 늘어 reserved가 같은 것이 올바르다.

### 사건 4: fp8이 OOM을 줄였지만 long context를 망친다

fp8 뒤 page bytes가 줄고 blocks는 늘었지만 long-context retrieval이 layer 17 이후 깨진다. short prompts는 정상이다. precision loss, scale indexing, unsupported fallback과 page boundary mapping이 competing hypotheses다.

greedy reference에서 projection 직후 K/V는 맞지만 stored fp8를 dequantize한 vector가 token 257부터 크게 다르다. 257이 block 첫 slot이고 read scale가 previous block entry를 가리킨다. first divergence는 dequant scale table의 block stride다.

일반 precision-loss는 모든 positions에서 점진적으로 나타나지 않고 exact boundary에서 급증해 약해진다. layer dtype과 selected kernel이 fp8 supported라 fallback 가설도 탈락한다. correct scale로 fp8 code를 재구성하면 reference tolerance에 들어 write path도 반증된다.

복구 후 block boundaries 15→16,31→32,256→257, sliding wrap와 TP local-head fixtures를 검증한다. gate는 output text, dequantized vector tolerance, attention logits와 long-context task를 포함한다. memory page bytes/capacity도 예상대로인지 확인한다.

### 네 사건의 공통 조사 축

formula 2×는 dimensions→spec→tensor shape에서 갈렸다. RSS retention은 used→free pool은 정상이고 reserved lifetime이 달랐다. GQA 미절감은 reserved denominator 또는 effective spec에서 갈렸다. fp8 오답은 bytes가 아니라 kernel scale read layout에서 갈렸다.

항상 logical expected bytes, finalized cache spec, physical tensor shape/dtype, allocator units, used/free/reserved metrics, kernel read/write correctness 순으로 내려간다. OOM이라고 모두 allocator 문제도 아니고 오답이라고 모두 weight 문제도 아니다. first divergence 전까지 맞는 층은 후보에서 제외한다.

### incident room에서 실제로 묻는 질문

2× 사건에서 첫 질문은 “어떤 memory 숫자가 두 배인가”다. process RSS라면 weights/workspace까지 포함해 formula와 비교 대상이 아니다. backend KV tensor bytes가 두 배라면 tensor list를 layer와 device별로 펼친다. page capacity bytes가 두 배라면 spec 식으로 더 올라간다. 같은 2×도 first divergent artifact가 다르다.

expected table에는 global KV heads 8, rank-local expected 2(TP=4), actual local shape axis, K/V widths, dtype element bytes와 layer count를 둔다. actual heads가 4면 replication/alignment, dtype 4면 finalized cache dtype, tensors가 두 세트면 double buffer owner를 찾는다. factor를 하나씩 반증한다.

RSS retention 사건에서는 request finish 직전과 직후 delta를 맞춘다. logical tokens, active block references, unique used blocks와 free blocks가 기대대로 움직였는가. 이 네 값이 정상인데 backend tensor bytes가 고정이면 pool reservation이다. unique blocks가 안 줄면 prefix retention/refcount 또는 delayed free이고, external allocations가 반복 증가하면 다른 owner leak다.

“재시작하면 memory가 내려온다”는 모든 process-lifetime pool에서도 참이므로 leak 증거가 아니다. 같은 workload 두 번째 cycle이 existing arena를 재사용하는지 본다. high-water mark가 cycle마다 계단식 증가하는지 안정되는지가 더 강한 관측이다.

GQA 사건에서는 model file/config diff보다 startup finalized attributes와 spec shape를 우선한다. page bytes가 1/4인데 total reservation이 같으면 block count가 4배인지 본다. maximum concurrent tokens 또는 free blocks가 늘었다면 optimization이 capacity로 환전된 것이다. 운영 목표가 concurrency라면 RSS 불변은 성공일 수 있다.

page bytes도 불변이면 K/V heads axis와 dtype을 확인한다. local head replication이 같은 floor를 만들었는지, backend가 query-expanded K/V를 persistent 저장하는지, wrong cache implementation branch가 선택됐는지 본다. model output이 맞다는 사실만으로 cache memory architecture가 intended GQA라는 보장은 없다.

fp8 사건에서는 memory와 correctness timeline을 같은 deployment change에 둔다. expected raw page bytes, scale bytes와 total blocks가 맞는지 먼저 확인한다. 그다음 first bad token/layer/head를 찾는다. exact block first slot에서만 깨지면 page table/scale stride, 모든 large values에서 깨지면 scale range/granularity, unsupported layers에서만 깨지면 fallback/conversion contract가 후보가 된다.

dequantized cache vector를 fp16 reference와 비교할 때 random output text까지 기다리지 않는다. write-before-quant tensor, stored code, stored scale, read-after-dequant tensor와 attention output을 checkpoint로 둔다. first mismatch 뒤 state만 조사한다. write-before부터 다르면 KV quantization 자체를 반증하고 upstream projection/position을 본다.

복구 후에는 memory만 다시 재지 않는다. 1GiB/512MiB expected table, page boundaries, long/sliding contexts, TP local heads와 output/logit parity를 통과한다. scale metadata가 추가돼 expected reduction이 48.4%라면 “half” 대신 actual component 합을 문서에 남긴다.

### 관측 화면이 답해야 하는 네 문장

첫째, 현재 model/request의 logical state는 몇 bytes인가. model dimensions와 per-layer component specs, actual lengths로 계산한다. 둘째, allocator가 몇 physical units를 request/cache owners에 배정했는가. block rounding과 sharing 뒤 unique units를 센다.

셋째, backend가 cache tensors와 related scale state로 몇 bytes를 reserve했는가. free pool도 포함한 arena capacity다. 넷째, process 전체에서 그 밖의 memory는 무엇인가. weights, graphs, workspace, staging과 general allocator를 분리한다.

이 네 문장 합이 맞지 않으면 metric 간 time skew도 확인한다. scheduler used blocks와 device tensor allocation metric이 서로 다른 timestamp이면 admission/free 중 일시적으로 불일치한다. epoch/snapshot generation을 붙여 같은 순간을 비교한다.

Prometheus-style aggregate에는 request ID를 label로 넣지 않는다. cache spec group, device, dtype, unit kind와 state(used/free/reserved)처럼 제한된 axes를 둔다. individual request length, shared refcounts와 block tables는 sampled trace/debug dump에 둔다.

alert는 RSS 고정 자체보다 invariants에 둔다. used+free가 total units와 맞지 않음, tensor bytes가 spec bytes×units 허용 오차를 벗어남, request terminal 뒤 owner ref가 비정상 유지됨, fp8 read parity가 boundary fixture에서 실패함이 원인에 가깝다.

## 33.8 logical·allocated·reserved를 한 문장에 섞지 않는다

좋은 기록은 “request logical KV는 1GiB, block rounding 뒤 allocated payload capacity는 1.08GiB, backend tensor arena reservation은 1.5GiB, active used slots는 0.9GiB 상당”처럼 denominator를 붙인다. process RSS 하나로 cache efficiency를 말하지 않는다.

관측에는 model-derived bytes/token, logical cached tokens, allocated/unique blocks, used slots, free/reserved blocks, backend tensor bytes와 allocator arena를 둔다. shared refcount와 delayed free, graph/workspace/offload staging을 leak 전 반증한다.

독자는 1GiB fixture의 각 항을 model config/source field에 연결하고, 65→96 rounding을 block accounting metric과 연결할 수 있어야 한다. MLA/hybrid/quantized path에서는 dense 식을 멈추고 cache spec의 실제 state components를 다시 세야 한다.

34장에는 block table mapping, prefix sharing, refcount, eviction와 free lifecycle을 넘긴다. 이 장의 산출물은 page/block 하나의 byte 크기, request logical length와 allocated slots의 차이, backend layout/reservation을 구분한 장부다.

### 한 request를 admission하기 전의 capacity 판단

새 request R이 prompt 8,192 tokens로 들어온다. logical GQA payload는 1GiB지만 allocator는 prompt를 한 번에 모두 commit하지 않을 수 있다. block size 16이면 full prompt에 512 blocks가 필요하다. prefix hit, chunked prefill과 available blocks에 따라 admission 시점 reservation semantics가 다르므로 512 blocks가 즉시 used된다고 단정하지 않는다.

capacity planner는 먼저 bytes/page를 계산한다. token 128KiB×16=2MiB/page다. free unique blocks 600이면 raw suffix capacity는 9,600 slots, 1.171875GiB payload capacity다. R의 requested 512 blocks는 들어오지만 다른 active decode의 future growth와 workspace headroom은 별도다. 이 policy는 다른 장의 몫이고 이 장은 unit conversion을 제공한다.

prefix hit가 4,096 tokens, 정확히 256 shared blocks이면 R이 새로 요구하는 unique suffix blocks는 256일 수 있다. 그러나 logical cache length와 request table references는 full 512 blocks를 가리킨다. metric이 new allocated blocks 256을 보고 request KV가 512MiB라고 말하면 logical shared prefix를 누락하고, references 512를 unique allocation으로 보면 sharing 절감을 누락한다.

R이 끝나면 suffix unique 256 blocks가 free될 수 있고 shared prefix 256은 cache retention/refcount에 따라 남는다. process tensor arena 600 blocks×2MiB=1.171875GiB는 그대로다. logical request 1GiB, incremental unique allocation 512MiB, post-finish retained prefix 512MiB capacity와 arena reservation을 한 문장에 섞지 않는다.

### OOM 계산에서 workspace를 붙이는 시점

KV arena 1.171875GiB가 있고 model weights와 other persistent tensors가 20GiB라고 하자. GPU usable budget 24GiB에서 단순 headroom은 약 2.828GiB다. prefill attention workspace, logits, graph pools와 temporary buffers가 3GiB peak면 R admission 뒤 OOM이 날 수 있다. KV logical formula가 맞아도 전체 peak 식은 실패한다.

반대로 fp8 KV로 arena를 줄여 600MiB headroom을 얻었어도 dequant workspace가 400MiB 추가되면 net peak improvement는 200MiB다. exact numbers는 backend와 chunk shape 관측이 필요하다. 이 장의 역할은 persistent KV reduction 512MiB를 total process reduction 512MiB라고 단정하지 않게 하는 것이다.

CUDA Graph pool은 captured address stability를 위해 buffers를 high-water capacity로 유지할 수 있다. workload가 작아져도 reservation이 안 내려오는 것은 RSS 사건과 같은 lifetime 차이다. graph enabled/disabled 비교에서는 warmup/capture state와 steady state를 분리한다.

### formula를 source field에 주석처럼 붙이기

독자가 code review를 할 때 식의 각 항 옆에 symbol을 적는다. `N_layer`는 cacheable attention layer specs count이며 model layer count와 다를 수 있다. `N_kv`는 finalized rank-local persistent heads 또는 component width다. `D_head`는 effective K/V dimension이며 query head dim shortcut이 아닐 수 있다. dtype bytes는 allocation tensor type이다.

`L_cache`는 request logical position, layer window-retained length, pool capacity 또는 cell capacity 중 문맥을 밝힌다. block 식에서는 `ceil(L/block_size)`이고 sharing 식에서는 unique pages와 logical references를 분리한다. quantized path는 payload type size와 scale state를 더한다.

source review comment가 “KV bytes = layers×tokens×heads…”로 끝나면 MLA/hybrid subclasses에 guard를 요구한다. cache spec interface가 page bytes를 polymorphic하게 제공한다면 caller가 dense internals를 재계산하지 않고 spec value를 소비하는 편이 drift를 줄인다.

metric exporter도 spec-derived denominator를 재사용하는지 본다. hardcoded dense formula는 new cache type이 추가될 때 틀린다. used units×spec page bytes와 tensor allocation bytes를 함께 노출하면 logical accounting과 physical reconciliation이 가능하다.

### 독자가 손으로 완료하는 최종 worksheet

첫 줄에는 model component를 쓴다. full attention layers, sliding layers, MLA layers, recurrent layers와 cross-attention layers를 나눈다. 각 component에 retained length/state count, widths/heads, dtype와 K/V 또는 latent factors를 넣어 logical bytes를 합한다.

둘째 줄에는 request lengths와 allocation units를 쓴다. A/B/C/D 1,16,17,31은 block 16에서 1,1,2,2 blocks다. logical 65, references six blocks, sharing 없을 때 unique six, slots 96이다. shared prefix가 있으면 references와 unique를 별도 갱신한다.

셋째 줄에는 physical spec을 쓴다. local heads, alignment stride, K/V packing, scale tensors, common group page size와 dtype fallback을 반영한다. page bytes×total reserved pages를 tensor arena baseline과 비교한다.

넷째 줄에는 transient와 process 항을 쓴다. conversion/context workspace, graph pools, staging과 general allocator reservation이다. 이 줄은 logical KV efficiency와 별도로 유지한다. OOM은 모든 simultaneous peak를 보지만 bytes/token 공식은 첫 줄만 설명한다.

다섯째 줄에는 observations를 쓴다. logical tokens, block references, unique used/free/total units, backend tensor bytes, scale bytes와 process reserved다. raw bytes와 snapshot epoch를 붙인다. worksheet의 first unequal transition이 debugging 시작점이다.

마지막에는 correctness를 쓴다. fp8면 write/read dequant parity, hybrid/MLA면 correct component/layer mapping, sharing이면 COW isolation, cross cache면 lifetime identity를 확인한다. memory 숫자가 맞아도 다른 request/layer state를 읽으면 성공이 아니다.

**33.8의 중간 결론.**

KV memory를 이해하는 가장 짧은 길은 formula를 외우는 것이 아니라 formula의 적용 범위를 표시하는 것이다. dense MHA/GQA/MQA logical payload에는 `K/V×layers×length×KV heads×head dimension×dtype bytes`가 맞는다. MLA, sliding, hybrid, recurrent와 cross-attention은 component별 식으로 갈라진다. fp8은 scales와 conversion path를 더한다.

그다음 logical tokens를 physical allocation units로 번역한다. 65 tokens가 block rounding으로 96 slots가 되고 prefix sharing으로 unique blocks가 다시 줄 수 있다. local-head replication, alignment와 group common page가 bytes/unit을 바꾼다. tensor arena는 free slots까지 reserve하며 process memory에는 다른 pools가 더해진다.

그래서 “KV가 1GiB다”라는 문장은 아직 절반이다. request logical payload 1GiB인지, allocated page capacity인지, backend arena인지, process RSS인지 말해야 한다. dtype, device/rank, component와 lifetime을 붙여야 비교할 수 있다.

네 incident도 이 구분으로 닫힌다. 2×는 shape/spec first divergence, RSS 유지는 used와 reserved lifetime, GQA 미절감은 page bytes와 block count 또는 effective heads, fp8 오답은 scale/read mapping 문제였다. 같은 memory symptom이라도 다른 층을 고친다.

34장으로 넘기는 block에는 이미 정확한 bytes/page, component/spec group, request logical references와 unique physical identity가 붙어 있다. 다음 장은 그 blocks가 table에 어떻게 연결되고 shared/refcount/eviction/free되는지를 설명한다. 이 장은 무엇을 얼마나 저장하는가를 닫고, 다음 장은 누가 그것을 언제 소유하는가를 맡는다.

실무에서 이 구분은 capacity proposal의 문장도 바꾼다. “GPU 한 장에 8k requests 20개가 들어간다” 대신 “dense logical 기준 request당 1GiB이며, 20GiB KV arena는 rounding 전 163,840 token slots다. block size 16과 observed length distribution, shared-prefix unique pages, graph/workspace headroom을 적용한 admission capacity는 별도로 산정한다”고 쓴다. 가정이 드러나므로 workload가 바뀌면 다시 계산할 수 있다.

모델 교체 review도 같은 장부를 쓴다. KV heads가 줄어도 MLA latent가 추가됐는지, local/global layer pattern이 달라졌는지, cache dtype과 scale granularity가 무엇인지, TP local replication이 어떻게 변했는지 component diff를 만든다. parameter count나 weight file size는 KV bytes의 proxy가 아니다.

장애 대응자는 숫자가 큰 곳부터 고치지 않는다. logical 예상과 spec page가 처음 갈리면 model/config branch, spec과 tensor allocation이 갈리면 layout/grouping, used/free와 owner가 갈리면 allocator lifecycle, bytes는 맞고 output이 갈리면 kernel mapping을 본다. 이 순서가 탐색 범위를 줄인다.

독자가 이 장의 1GiB와 65→96 계산을 스스로 재현하고 자기 model fields로 치환할 수 있다면, KV cache는 더 이상 막연한 GPU memory 덩어리가 아니다. source에서 유도하고 metric으로 반증할 수 있는 구체적인 tensor state가 된다.

최종 review에는 계산식 자체보다 각 항의 provenance를 남긴다. 어느 finalized field가 layer/head/dimension/dtype을 제공했고, 어느 spec이 page bytes를 만들었으며, 어느 tensor allocation과 metric이 이를 확인했는지 연결한다. 새 backend가 추가돼도 이 연결을 다시 채우면 dense shortcut이 적용 가능한지 즉시 판단할 수 있다.

memory 최적화의 목표도 분명해진다. raw payload를 줄이는 것, rounding slack를 줄이는 것, sharing을 늘리는 것, reserved arena를 조정하는 것과 transient peak를 줄이는 것은 서로 다른 작업이다. 먼저 어떤 장부를 줄일지 선언한 뒤 그 장부와 correctness가 실제로 변했는지 검증한다.

## 33.9 하나의 worksheet로 GQA·MLA·sliding을 비교한다

공통 좌표를 먼저 고정한다. batch가 아니라 request R 하나, rank 하나, cacheable component 하나를 기본 단위로 둔다. `logical`은 R의 의미 있는 state payload, `allocated`는 page/block rounding 뒤 R이 참조하는 capacity, `reserved`는 allocator arena가 확보한 전체 capacity, `resident`는 현재 device에 실제 존재하는 arena·metadata·scale을 뜻한다. 구현 metric이 다른 뜻으로 resident를 쓰면 그 정의를 우선한다.

Llama 계열 GQA 예제로 layers32, query heads32, KV heads8, head dimension128, BF16 2B, context4096를 둔다. rank-local replication이 없고 TP가 KV heads를 4-way로 sharding하면 rank-local KV heads2다. dense persistent payload는 `2(K,V)×32×4096×2×128×2B=134,217,728B`, 즉 128MiB다. TP 전체 합은 512MiB다.

KV heads8 전체를 각 rank에 replicate하는 backend라면 rank당 512MiB이고 4 ranks 합은 2GiB다. model의 global GQA 식만 계산해 512MiB라고 쓰고 rank metric과 비교하면 4배 오차가 난다. finalized parallel mapping에서 local heads와 replication factor를 확인한다. query heads32는 KV payload 식에 직접 넣지 않는다.

block size16이면 request4096는 256 blocks로 정확히 나뉜다. 위 rank-local bytes/token은 `2×32×2×128×2=32,768B`, page bytes는 524,288B, 즉 512KiB다. request allocated도 128MiB다. length4097이면 logical은 128MiB+32KiB지만 257 pages를 예약해 128.5MiB다. logical/allocated gap은 약 480KiB다.

allocator가 rank당 40GiB KV arena를 startup에 reserve하면 R 하나의 allocated128MiB와 별개로 backend tensor reserved/resident가 40GiB일 수 있다. used pages256, free pages81664 같은 unit count와 arena bytes가 보존되는지 맞춘다. process RSS에는 weights, graph/workspace와 allocator bookkeeping이 더해진다.

MLA 예제는 dense K/V head 식을 버리고 component width 합을 쓴다. layers60, cached latent width512 BF16와 positional component width64 BF16를 token마다 저장하며 별 V가 latent에 통합됐다고 가정하자. rank sharding 후 실제 spec이 이 layout을 택한다는 전제에서 bytes/token은 `60×(512+64)×2=69,120B`, context4096 logical은 283,115,520B, 약 270MiB다.

여기에 scale이 token/layer당 two FP32 values라면 `60×2×4=480B/token`, 4096에서 1,966,080B가 추가된다. 실제 scale granularity가 page, head group 또는 tensor라면 식이 달라진다. 이름이 FP8 cache라고 payload1B만 곱하지 않고 source tensor shapes를 본다.

MLA latent가 TP ranks에 replicate되는지 shard되는지도 확인한다. width512가 rank-local128이면 logical은 약 4분의 1이지만 positional component64가 replicate될 수 있다. `60×(128+64)×2×4096≈90MiB`다. 전체 width를 균등 나누는 shortcut은 replicated component를 놓쳐 더 작게 계산한다.

sliding GQA는 retained length를 layer별로 바꾼다. layers32 중 full8, sliding24, global context4096, window1024, rank-local heads2, head dim128, BF16라 하자. bytes/token-layer-head pair는 K/V 포함 512B다. logical은 `8×4096×2×512 + 24×1024×2×512 = 56MiB`다. 모든 layer를 4096로 세면 128MiB여서 72MiB를 과대평가한다.

그러나 allocator가 full/sliding layers를 같은 common page group으로 묶고 max page bytes를 사용하면 reserved/allocated 절감이 logical 식만큼 나오지 않을 수 있다. separate pools인지 common group인지 cache spec builder와 allocator consumer를 확인한다. sliding eviction으로 logical old tokens가 사라져도 circular arena capacity는 high-water로 남을 수 있다.

## 33.10 hybrid/recurrent state를 같은 단위로 더한다

hybrid model에 full attention12 layers, sliding attention24 layers, recurrent state12 layers가 있다고 하자. full context8192, window2048, rank-local KV heads2, head dim128, BF16다. attention bytes는 `2×heads×dim×dtype=1024B/token/layer`다. full은 `12×8192×1024=96MiB`, sliding은 `24×2048×1024=48MiB`, 합 144MiB다.

recurrent layer가 request마다 state tensors `[conv_width=4,hidden=4096]` BF16와 `[state_dim=16,hidden=4096]` FP32를 유지한다고 가정한다. layer당 conv32KiB, recurrent256KiB, 합 288KiB이고 12 layers에서 3.375MiB다. 이는 sequence length에 선형 증가하지 않는다. attention bytes/token 식에 recurrent layers를 넣으면 context8192에서 거대한 과대평가가 난다.

반대로 recurrent state가 batch slot capacity만큼 arena에 preallocate되면 request logical3.375MiB와 process reserved는 다르다. slots128이면 432MiB가 예약될 수 있다. active request10개 logical33.75MiB만 보고 process memory와 비교하면 약 398MiB gap이 정상 capacity slack일 수 있다. slot reuse generation과 zero/reset correctness도 필요하다.

hybrid grouping은 component별 page/state unit 크기가 다르다. allocator가 common block abstraction을 위해 largest unit에 맞추면 작은 recurrent state 또는 sliding page에 padding이 붙는다. group page bytes가 attention512KiB인데 recurrent logical unit288KiB라면 unit당 224KiB slack이다.128 slots에서 28MiB다. 실제 grouping/packing을 source에서 확인한다.

cross-attention cache는 encoder input lifetime을 따른다. decoder context가 늘어도 cross K/V length는 encoder length에 고정될 수 있고 여러 beams/requests가 공유할 수 있다. self KV와 같은 `L_cache`를 넣지 않는다. request cancel, encoder artifact cache와 decoder terminal 중 누가 reference를 반환하는지 별 owner ledger를 둔다.

최종 worksheet 행은 component, layers, retained length/state count, rank-local width/heads, K/V factor, dtype, scale/metadata, logical units, allocation unit, allocated units, arena reserve와 owner generation이다. 합계 전에 각 행 단위가 bytes인지 elements인지 pages인지 검산한다.

## 33.11 stride·packing·alignment가 formula를 바꾸는 지점

논리 shape `[layers,blocks,2,heads,block_tokens,head_dim]`의 element 곱이 payload다. 하지만 tensor stride가 contiguous logical shape와 다르면 allocation storage size는 마지막 element offset과 alignment를 따라간다. K/V가 별 tensors인지 axis2로 packed되는지, head_dim이 vector width에 맞춰 padded되는지 확인한다.

head_dim120을 kernel이 128로 pad하고 KV heads3을 alignment4로 잡는다면 logical elements/token/layer는 `2×3×120=720`, storage는 `2×4×128=1024`다. BF16에서 logical1440B, storage2048B로 42.2% 증가한다. config의 head dimension과 allocated stride를 혼용하면 capacity를 과대예측한다.

page header, block table와 scale metadata도 별 항이다. pages100,000개에 header64B면 약 6.1MiB다. request block tables가 slots1024×max_blocks4096×4B면 16MiB다. KV payload 수십GiB에 작아 보여도 tight guard와 graph/workspace를 합치면 OOM 경계에 영향을 준다.

allocator의 소비 지점은 page bytes를 받아 total blocks를 정하는 startup과 request별 blocks를 차감하는 runtime으로 나뉜다. startup이 raw formula로 blocks를 계산하고 tensor builder가 aligned stride를 사용하면 arena tensor가 예상보다 커져 startup OOM 또는 blocks 축소가 생긴다. 두 consumer가 동일 cache spec을 쓰는지 본다.

vLLM에서는 finalized cache specs, KV cache manager의 blocks/bytes와 worker/runner tensor allocation을 연결한다. SGLang에서는 token-to-KV pool/page allocator의 cell size와 model runner cache tensors를 잇는다. Transformers continuous cache의 page spec/block manager와 model attention consumer를 잇고, llama.cpp에서는 KV cell layout/type와 backend buffer allocation, CUDA kernel view를 잇는다.

각 구현 source card에는 logical tensor shape, allocation shape/stride, dtype, rank mapping, page/cell bytes producer, allocator total-unit division, kernel consumer view와 release owner를 둔다. 문서 formula만 있고 allocation span이 없거나 tensor shape만 있고 allocator denominator가 없으면 source walk가 덜 닫힌 것이다.

## 33.12 byte 단위 혼합으로 capacity가 어긋난 사고

관측은 startup이 KV capacity131,072 tokens를 보고했지만 약 98,000 tokens에서 allocation failure가 반복된 것이다. GPU process memory는 예상 KV arena보다 약 8GiB 크고 output correctness는 정상이다. leak, fragmentation, graph pool과 replicated heads가 후보다.

config 계산은 GQA global KV heads8을 TP4로 나눠 local2라고 가정해 page bytes512KiB를 만들었다. 그러나 selected backend tensor는 최소 local heads4 alignment로 storage했고 page bytes1MiB를 소비했다. allocator total blocks는 512KiB denominator로 계산했지만 실제 tensor는 1MiB stride였다. logical bytes와 storage bytes를 섞은 것이다.

단순하면 capacity가 절반이어야 하지만 약 75%에서 실패한 이유는 arena가 실제 tensor capacity를 startup에 조정하고 일부 pool/guard가 보상했기 때문이다. first divergence는 model logical shape에서 spec storage stride로 갈 때다. request block leak이나 runtime fragmentation은 drain 뒤 owner counts가 기준선으로 돌아오고 failure token이 반복 가능하므로 약해진다.

branch trace는 normalized parallel config local heads2, cache spec page bytes512KiB, tensor builder aligned heads4, allocator total blocks와 kernel stride를 나란히 둔다. metric의 `kv_bytes_used`가 blocks×512KiB를 써 실제 tensor resident와 8GiB 차이를 만들었다. observability도 같은 잘못된 denominator를 공유했다.

수정은 tensor builder만 padding을 제거하는 것이 아니다. kernel capability가 alignment4를 요구한다면 cache spec이 storage heads4와 page1MiB를 canonical하게 제공하고 allocator와 metric이 이를 소비해야 한다. logical efficiency metric은 별도로 local logical heads2를 사용한다. 두 숫자의 이름과 분모를 분리한다.

verification은 heads2/3/4, head_dim120/128, TP1/2/4, block size 경계를 교차한다. spec page bytes, actual tensor storage bytes/blocks, allocator total units와 kernel view strides를 assert한다. fill-to-capacity fixture는 advertised usable blocks까지 성공하고 다음 block이 명시적으로 reject돼야 한다.

rollback은 새 spec generation으로 admission을 fence하고 old-layout active requests를 drain한다. page bytes가 다른 arena 사이에서 block handles를 그대로 옮기지 않는다. live conversion을 지원한다면 K/V contents와 scale mapping, generations를 검증한다. 지원하지 않으면 bounded drain/restart와 client impact를 명시한다.

## 33.13 allocator에서 kernel까지 source consumer를 왕복한다

source 탐색은 model config의 head 수에서 끝나지 않는다. 먼저 finalized model/cache config가 layer별 cache spec을 만드는 함수를 찾는다. 이 spec이 full/sliding/MLA/cross/recurrent type, local widths, dtype, block tokens와 page bytes를 어떤 field로 표현하는지 적는다. polymorphic spec이면 dense subclass의 식을 다른 subclass에 적용하지 않는다.

다음은 startup sizing consumer다. available device bytes에서 weights, non-KV profiling peak와 guard를 뺀 뒤 page bytes로 나누어 total units를 정하는 함수 span을 찾는다. integer division과 alignment, multiple pools 또는 pipeline/tensor ranks 사이 minimum을 확인한다. GB/GiB 변환과 decimal log formatting이 계산 내부 단위에 침투하지 않는지도 본다.

예를 들어 usable20GiB를 page512KiB로 나누면 40,960 pages다. log가 decimal GB로 20.0GB라고 보여도 내부 bytes는 21,474,836,480일 수 있다. 사용자가 `20.0×10^9/524,288≈38,147`로 역산해 capacity mismatch라고 판단하면 단위 표시 문제다. 내부 raw bytes와 IEC/decimal rendering을 분리한다.

세 번째는 arena tensor builder다. total units와 spec shape가 실제 allocation dimensions로 어떻게 변하는지 본다. layer-major separate tensors, unified `[blocks,layers,...]`, K/V separate/packed, per-layer pools와 hybrid grouping을 기록한다. `num_blocks`가 모든 layers에 같은지 spec group마다 다른지 확인한다.

storage bytes는 tensor `numel×element_size`만으로 충분하지 않을 수 있다. storage offset, padded stride, allocator alignment와 multiple backing buffers를 본다. view tensor의 numel은 작지만 backing storage는 클 수 있고, slices 여러 개가 같은 storage를 공유할 수 있다. unique storage pointer/size 기준과 logical view 합을 둘 다 기록한다.

네 번째는 runtime page allocator다. request가 token append 또는 prefill chunk를 받을 때 필요한 blocks를 계산하고 free list/refcount를 mutation하는 source를 찾는다. logical references, unique physical handles와 copy-on-write allocation을 구분한다. `can_allocate`와 commit 사이 generation reservation이 있는지 27장의 admission 계약과 연결한다.

sliding cache는 old logical position이 window 밖으로 나갈 때 same circular cells를 재사용할 수 있다. allocator free가 일어나지 않아도 logical retained state는 window로 제한된다. block table의 logical position→physical cell mapping과 kernel modulo/slot consumer를 연결한다. overwrite 전에 old generation의 last attention consumer를 fence한다.

MLA에서는 latent and positional components가 같은 page handle 안에 packed되는지 separate tensors/allocators인지 본다. separate라면 한 component allocation 성공 뒤 다른 component 실패의 rollback이 필요하다. page count는 같아도 bytes/page와 alignment가 다를 수 있다. metric이 latent pool만 세고 positional pool을 누락하지 않는지 확인한다.

recurrent state는 token page allocator가 아니라 request/sequence slot allocator를 사용할 수 있다. cache라는 공통 이름 때문에 blocks metric에 억지로 환산하지 않는다. slot owner, state tensors, reset/copy semantics와 preemption persistence를 source card에 별 행으로 둔다. hybrid total memory에서 bytes만 합치되 unit 보존식은 따로 유지한다.

다섯 번째는 model runner와 attention layer consumer다. cache tensor views를 layer에 bind하는 함수, forward metadata의 block table/slot mapping, backend kernel call의 expected shape/stride/dtype를 잇는다. allocator가 page bytes를 올바르게 계산해도 layer index가 wrong spec group을 참조하면 correctness가 깨진다.

kernel wrapper가 strides를 전달하는지 contiguous assumption을 하는지 확인한다. FP8 scale tensors는 pointer, shape와 granularity가 read/write kernel 양쪽에서 일치해야 한다. MLA component offsets와 sliding positions도 source span으로 고정한다. Python view shape만 보고 CUDA indexing을 추정하지 않는다.

여섯 번째는 free/reuse consumer다. finish, abort, preemption, prefix eviction과 process restart가 어떤 handles/refcounts/slots를 반환하는지 찾는다. request terminal과 arena release를 구분한다. preallocated arena는 request가 0이어도 resident로 남을 수 있고 정상이다. request-owned used units가 0으로 수렴해야 한다.

vLLM card는 cache spec 생성→worker memory/profile sizing→KV cache manager allocation→runner binding→attention backend→free/refcount를 한 줄로 잇는다. SGLang card는 model config/pool construction→token-to-KV pool allocator→schedule batch slot mapping→runner attention→free/retract를 잇는다. 각 pinned revision의 실제 symbol/span을 사용한다.

Transformers continuous card는 paged cache initialization→block manager allocation/refcount→batch input block tables→model attention cache update/consumer→request cleanup을 잇는다. llama.cpp card는 model hparams/cache type→KV cell/buffer sizing→slot/batch positions→backend CUDA graph/kernel view→slot erase/defrag를 잇는다.

구현 비교표의 공통 열은 spec producer, storage builder, runtime unit allocator, mapping producer, kernel consumer, terminal owner다. 하지만 구현마다 한 object가 여러 역할을 겸할 수 있다. 같은 class 이름을 요구하지 않고 동일 byte/identity가 변환되는 경계를 찾는다.

source card 마지막에는 falsifier를 둔다. actual tensor stride가 spec과 같다면 alignment mismatch 가설은 탈락한다. unique used pages가 예상보다 많다면 sharing/COW를 본다. used는 맞고 resident만 크면 arena/other pools를 본다. bytes와 mapping은 맞는데 output이 틀리면 kernel scale/indexing을 본다.

**왜 byte 계산이 admission보다 앞서는가.** KV는 요청마다 계속 자라므로 평균 prompt 길이만으로 capacity를 잡으면 tail request가 page slack과 함께 동시성 한계를 넘긴다. 왜냐하면 logical payload 외에도 layer, K/V, KV head, dtype, block padding, allocator metadata와 replica 수가 곱해지기 때문이다. 비용을 작게 잡으면 scheduler는 실행 가능한 batch라고 판정하지만 allocator가 뒤늦게 실패한다. 반대로 reserved byte만 쓰면 재사용 가능한 pool까지 사용 중으로 오해한다.

**두 반증 실험.** 같은 logical token 수에서 block size만 바꿔 payload는 일정하고 slack·metadata가 달라지는지 측정한다. 이어 GQA의 KV head 수만 바꿔 parameter width와 token당 cache byte가 어떤 비율로 달라지는지 계산한다. 이 실험은 OOM 원인이 모델 parameter라는 가설과 cache fragmentation이라는 가설을 분리한다.

왜 profiler의 peak만으로 식을 대체할 수 없는지도 분명하다. peak에는 workspace와 다른 요청이 섞이고, 왜 증가했는지 owner를 말해 주지 않는다. 반대로 식만으로는 allocator가 왜 reserved page를 즉시 반환하지 않는지 설명하지 못한다. 그래서 logical byte, physical allocation과 timeline을 함께 검산해야 왜 admission 한계가 달라졌는지 답할 수 있다.

## 33.14 capacity incident의 관측·검증·rollback terminal

사건 dossier 첫 표는 동일 snapshot epoch의 다섯 장부다. model logical payload, request allocated pages, allocator used/free/total, backend unique storage bytes와 process reserved/resident를 둔다. 서로 다른 시각의 high-water RSS와 current pages를 빼서 leak을 만들지 않는다. GPU rank와 device도 고정한다.

앞 사고의 구체적 숫자를 복원한다. spec은 page512KiB, advertised total16,384 pages라 8GiB arena를 예상했다. actual aligned tensor는 page1MiB equivalent라 같은 16,384 capacity를 만들면 16GiB storage가 필요하다. startup에서 다른 pool headroom8GiB를 잠식해 처음에는 성공했지만 request pages 약 12,288에서 workspace peak와 충돌해 실패했다.

advertised token slots는 `16,384×block16=262,144`였고 failure는 약 196,608 tokens, 즉 75%였다. 이 비율을 leak rate로 해석하지 않는다. request drain 뒤 pages는 free로 돌아왔고 동일 fill에서 같은 threshold가 반복됐다. persistent8GiB storage gap과 workspace4GiB peak가 합쳐진 deterministic capacity error다.

observation→branch는 네 갈래다. used pages가 예상보다 빨리 증가하면 block rounding/sharing/progress를 본다. page 수는 맞고 tensor bytes가 다르면 spec/layout/stride다. tensor bytes도 맞고 process reserved만 크면 graph/workspace/general allocator다. 모든 bytes는 맞는데 allocation이 실패하면 fragmentation, largest allocation과 pool fencing을 본다.

first divergence는 raw model config가 아니라 finalized spec과 tensor builder 사이에서 찾는다. config global KV heads8, normalized rank-local2까지는 맞다. spec storage heads2가 512KiB page를 만들었지만 builder/kernel capability가 aligned heads4를 선택했다. allocator와 metric이 spec을 믿어 false capacity를 발표했다.

cause를 한 문장으로 고정한 뒤 수정 owner를 정한다. canonical storage descriptor가 aligned heads, padded dimension, packed components, dtype/scale와 page bytes를 제공해야 한다. startup sizing, tensor builder, allocator metric과 kernel view가 동일 descriptor를 소비하게 한다. logical descriptor는 efficiency 계산용으로 별도 유지한다.

단위 type도 강화한다. raw integer 대신 `LogicalBytes`, `StorageBytes`, `PageBytes`, `PageCount`, `TokenCount`처럼 코드 수준 wrapper가 가능하면 사용한다. 최소한 variable names와 assertions에 suffix를 둔다. pages×tokens를 bytes와 직접 더하는 실수를 막는다. serialization/metrics boundary에서 raw bytes로 변환한다.

검증 첫 층은 pure calculation fixture다. GQA local heads2/aligned4, MLA latent+positional, sliding full/window, hybrid recurrent state와 quantized scales의 expected logical/storage/page bytes를 손계산 값과 비교한다. decimal/IEC 표시가 raw bytes를 바꾸지 않는지도 본다.

둘째 층은 tensor fixture다. allocated storages를 unique pointer로 deduplicate하고 storage nbytes, shapes, strides와 offsets를 descriptor 합과 비교한다. views 합을 backing storage와 잘못 더하지 않는다. K/V separate/packed와 hybrid groups를 모두 포함한다. no model execution이어도 construction path를 검증할 수 있다.

셋째 층은 allocator fill fixture다. advertised total pages까지 allocate하고 마지막 성공/첫 실패를 확인한다. page tail lengths15/16/17, prefix sharing/COW, cancel/free와 reuse generation을 교차한다. logical token capacity와 physical page capacity를 별 결과로 둔다.

넷째 층은 kernel parity다. page boundary, sliding wrap, MLA component offsets와 FP8 scale granularity에서 reference path와 selected backend output을 비교한다. memory 숫자만 맞추고 mapping을 틀리게 고치지 않았는지 확인한다. cross-request sentinel로 valid-but-wrong read를 잡는다.

다섯째 층은 peak integration이다. graph variants, attention/conversion workspace와 KV arena를 동시에 활성화해 combined device envelope를 검증한다. page capacity만 채운 synthetic test가 production OOM을 재현하지 못할 수 있다. 27장의 admission equation에 canonical storage bytes를 입력한다.

rollout은 model/cache layout generation으로 cohort를 나눈다. new descriptor로 만든 arena와 old handles를 섞지 않는다. startup canary에서 advertised pages, actual storage, fill ratio와 process headroom을 비교한다. long-context output parity와 abort/reuse도 통과한 뒤 traffic을 늘린다.

rollback trigger는 advertised/actual storage mismatch, capacity fill below guard, allocator conservation gap, output parity failure, FP8 scale/index error 또는 workspace collision이다. 발생하면 new admission을 fence한다. inflight old/new layout requests를 generation별로 drain 또는 terminal하고 handles를 reconcile한다.

arena를 즉시 free할 수 없는 graph/caching allocator라면 process restart가 필요할 수 있다. restart를 단순 복구 성공으로 쓰지 않는다. client terminal, request handles, prefix references와 old process resource closure를 기록한다. 새 process readiness는 descriptor/tensor self-check 뒤 연다.

관측 metric은 logical bytes, allocated unique pages/bytes, arena total/free/used, actual unique storage bytes, scales/metadata, workspace/graph와 process allocated/reserved를 둔다. model/component/layout/config generation을 bounded labels로 쓰고 request details는 trace에 둔다.

reconciliation dashboard는 `used_unique_pages×canonical_page_bytes`와 page-backed storage used-equivalent를 비교하되 arena 전체 storage와 혼동하지 않는다. arena utilization은 used/total, logical efficiency는 logical/allocated, device headroom은 all resident owners 합으로 각각 계산한다. 하나의 “KV utilization”로 합치지 않는다.

capacity proposal에는 workload length distribution과 block rounding을 적용한다. 평균 length만 쓰지 않고 page tail histogram, prefix sharing unique ratio, sliding/full layer patterns와 recurrent slots를 포함한다. rank별 최소 headroom이 cluster admission을 제한할 수 있다. TP 전체 합으로 rank OOM을 숨기지 않는다.

마지막 독자 worksheet는 GQA, MLA, sliding/hybrid 각 행에 logical formula와 actual source fields를 쓴다. 다음 열에 aligned storage, page bytes, request pages와 arena pages를 쓴다. 관측 tensor/allocator/metric을 붙이고 first unequal transition을 표시한다. 차이가 의도된 padding인지 bug인지 owner와 근거로 판정한다.

승인 문장은 “KV 계산이 맞다”가 아니다. “고정 revision의 cache descriptor가 rank-local/aligned/component layout을 반영했고 startup sizing·tensor storage·runtime allocator·kernel view가 동일 page bytes/generation을 소비하며 fill, boundary, cancel, long-context parity와 rollback terminal을 통과했다”라고 쓴다.

정적 source 분석으로 확정할 수 있는 shape/consumer와 실제 allocator peak·fragmentation·traffic sharing ratio를 구분한다. 후자는 controlled fixture와 production trace가 필요하다. 미검증 항목마다 필요한 metric, workload와 중단 조건을 남긴다.

이 장이 닫히면 34장에는 단순 blocks가 아니라 canonical bytes/page, logical/unique references, layout generation과 terminal owner가 넘어간다. cache eviction과 sharing 정책은 이 정확한 physical unit 위에서만 의미가 있다. 잘못된 denominator를 넘기면 뒤 장의 정교한 allocator 정책도 false capacity를 운영한다.

**세 모델을 한 표에서 끝까지 계산하는 연습.**

모델 A는 앞의 Llama GQA다. length4097, layers32, local KV heads2, head dim128, BF16, block16이다. logical bytes는 `4097×32768=134,250,496B`다. allocated pages257×524,288B=134,742,016B다. logical/allocated efficiency는 약 99.64%다. arena40GiB가 preallocated라면 resident denominator에서 request share는 약 0.31%다.

모델 B는 MLA60 layers, rank-local latent128과 replicated positional64, BF16, token-layer scale two FP32 values, block16이다. payload/token은 `60×192×2=23,040B`, scale/token480B, 합 23,520B다. length4097 logical은 96,373,440B다. page raw bytes는 376,320B지만 allocator가 384KiB로 align하면 page393,216B,257 pages에서 101,056,512B다. efficiency 약 95.37%다.

여기서 raw page376,320B로 total units를 계산하고 실제 aligned393,216B로 arena를 만들면 약 4.49% capacity 차이가 난다. 작은 차이처럼 보여도 20GiB pool에서 약 920MiB다. graph/workspace guard가 얇으면 특정 long-context cohort에서만 OOM이 난다. global memory 평균으로는 찾기 어렵다.

모델 C는 full12/sliding24/recurrent12 hybrid다. length8192/window2048 attention logical144MiB, recurrent request state3.375MiB, 합 147.375MiB다. full page512KiB×`12 layer grouping`처럼 구현별 page 정의를 다시 확인하고, sliding circular pages가 high-water48MiB를 유지하는지 본다. recurrent slots128 arena432MiB 중 request 하나는 3.375MiB다.

모델 C에서 active requests10이면 recurrent logical33.75MiB지만 resident arena432MiB다. attention pages는 각 request length/window와 sharing에 따라 증가한다. “hybrid cache request당 190MiB”처럼 recurrent arena 전체를 각 request에 나눠 붙이면 active count에 따라 값이 흔들린다. per-request logical과 process fixed/capacity reserve를 별 행으로 둔다.

세 모델의 비교 열은 logical bytes/request, allocated bytes/request, process reserved pool, current resident storage, transient peak다. A가 logical128MiB, B92MiB, C147MiB여도 process headroom 순위는 arena sizing과 graph/workspace 때문에 달라질 수 있다. 가장 작은 logical 모델이 가장 많은 request를 받는다고 단정하지 않는다.

**metric 이름을 단위 계약으로 만든다.**

`kv_cache_usage` 같은 ratio만 내지 않는다. `kv_logical_bytes`, `kv_allocated_unique_bytes`, `kv_arena_used_pages`, `kv_arena_total_pages`, `kv_storage_reserved_bytes`, `kv_scale_metadata_bytes`처럼 분자를 이름에 넣는다. ratio는 분자/분모 설명을 help text에 고정한다.

page size metric에는 raw bytes를 사용하고 KiB 문자열은 UI에서 변환한다. block tokens와 page bytes를 함께 노출한다. model/config/layout generation과 rank는 bounded labels다. component type은 full/sliding/MLA/recurrent/cross 정도로 제한한다. request나 layer index 전체를 labels로 쓰지 않는다.

snapshot은 epoch와 CUDA stream synchronization semantics를 갖는다. allocator counter는 CPU commit 시점, tensor memory는 async allocation 시점, process reserved는 caching allocator high-water일 수 있다. 동일 timestamp처럼 보여도 visibility boundary가 다르다. reconciliation window와 pending reservations를 포함한다.

alarm 예로 allocated unique bytes가 canonical pages×bytes와 1% 이상 지속적으로 다르면 spec/layout drift를 본다. logical efficiency가 낮으면 length/page rounding이나 grouping을 본다. arena used는 낮지만 process headroom이 줄면 graph/workspace/other pools를 본다. 각 alarm이 다른 runbook으로 연결된다.

**변경 전후 review checklist.**

모델 revision에서 layer type pattern, KV/query heads, head dimensions, MLA components, recurrent shapes와 cross cache를 diff한다. serving revision에서 cache spec, dtype/scale, block size, alignment, grouping과 parallel mapping을 diff한다. CUDA/backend revision에서 kernel expected stride와 supported cache types를 diff한다.

startup fixture는 descriptor bytes와 actual unique storages를 비교한다. runtime fixture는 1/15/16/17/4097 tokens, sharing/COW, sliding wrap, recurrent slot reuse를 포함한다. failure fixture는 partial component allocation, abort, preemption과 old generation late write를 포함한다. performance fixture는 workspace/graph combined peak를 포함한다.

option 변경도 mutation chain으로 쓴다. cache dtype은 spec element/scale bytes와 kernel selector를, block size는 rounding·page bytes·block table을, memory utilization은 arena total과 other headroom을, TP는 local heads/replication과 rank minimum을 바꾼다. 이름만 보고 한 항만 바뀐다고 쓰지 않는다.

rollback은 layout-compatible 여부를 먼저 판정한다. dtype, stride, page bytes 또는 component packing이 달라지면 active handles를 재사용하지 않는다. admission fence와 bounded drain, prefix artifact invalidation, arena destruction/restart와 readiness self-check를 순서대로 수행한다. request/resource/client terminal을 모두 기록한다.

최종 dossier에 calculation spreadsheet만 두지 않는다. 각 coefficient의 pinned source span, actual tensor storage 증거, allocator counters, kernel parity, capacity fill 결과와 rollback rehearsal을 연결한다. 이 연결이 있어야 다음 release에서 어느 항이 변했는지 빠르게 다시 계산할 수 있다.

독자는 이제 “GQA라 KV가 작다”, “MLA라 더 작다”, “sliding이라 window만 저장한다” 같은 문장을 조건 없이 쓰지 않는다. rank-local replication, component layout, alignment, allocator grouping과 resident lifetime을 확인한 뒤 logical·allocated·reserved·resident 중 정확한 장부를 말한다. 그것이 capacity와 correctness를 동시에 지키는 byte 계산이다.

**incident room의 20분 판정 순서.**

첫 5분에는 단위를 잠근다. 문제가 난 GPU rank, model/cache/layout generation과 snapshot epoch를 적고 모든 metric을 raw bytes/pages/tokens로 변환한다. GB 표기와 GiB, global과 rank-local, logical heads와 storage heads를 섞지 않는다. advertised capacity와 actual failure point도 pages와 tokens 두 단위로 쓴다.

다음 5분에는 worksheet의 인접 행만 비교한다. logical→allocated가 갈리면 rounding/alignment/grouping이다. allocated pages→arena storage가 갈리면 page descriptor/tensor layout이다. arena는 맞고 process resident가 갈리면 scales, workspace, graph 또는 general allocator다. 처음부터 모든 CUDA allocations를 뒤지지 않는다.

세 번째 5분에는 passing neighbor를 고른다. TP2에서는 정상이고 TP4에서 실패하거나 head dim128은 정상이고 120에서 실패한다면 local-head alignment branch를 본다. length4096은 정상이고 4097에서 실패하면 page tail과 capacity boundary를 본다. model A는 정상이고 MLA B만 실패하면 dense shortcut이 spec subclass에 침투했는지 본다.

마지막 5분에는 source consumer를 왕복한다. spec page bytes producer, startup division, tensor storage builder, runtime allocator와 kernel view가 같은 descriptor/generation을 쓰는지 표시한다. first divergence 뒤 consumer를 모두 고치고 fixture/rollback을 정한다. metric exporter만 고쳐 숫자를 맞추는 것은 capacity bug를 남긴다.

**memory 절감 제안을 반증하는 질문.**

KV dtype을 BF16에서 FP8로 바꾸면 payload는 절반인가. scale/metadata, aligned stride, unsupported fallback layers와 conversion workspace를 넣어야 한다. logical payload가 절반이어도 arena block count나 process peak가 절반이 아닐 수 있다. long-context output parity와 scale indexing을 hard gate로 둔다.

block size를 16에서 8로 줄이면 rounding slack는 줄지만 page count와 block-table metadata, allocator operations가 늘어난다. kernel/backend가 작은 page를 효율적으로 소비하는지 확인한다. observed length-tail histogram으로 saved payload와 added metadata/CPU cost를 계산한다.

prefix sharing을 늘리면 logical references는 같고 unique pages가 줄어든다. COW tail, tenant/model/adaptor identity와 eviction lifetime이 정확해야 한다. hit ratio만 높고 unique used bytes가 줄지 않으면 shared lookup은 성공했지만 physical reuse가 안 됐을 수 있다.

sliding window를 줄이면 sliding layers logical retained bytes는 감소하지만 full/global/recurrent/cross components는 그대로다. common pool이 high-water reserve를 유지하면 process resident도 바로 줄지 않는다. quality/context contract와 position wrap correctness도 함께 검증한다.

arena utilization을 높이면 usable pages는 늘지만 graph/workspace guard가 줄어 combined execution OOM을 만들 수 있다. page fill fixture와 실제 peak integration을 둘 다 통과해야 한다. advertised KV tokens 최대화가 serving goodput 최대화와 같지 않다.

이 질문들은 최적화를 막기 위한 것이 아니다. 어느 장부를 실제로 줄이는지, 비용이 다른 장부로 이동하는지, 그리고 capacity claim이 source와 tensor에서 재현되는지를 분명히 한다. 성공 문장은 saved logical bytes, changed allocated pages, arena/headroom effect와 correctness terminal을 함께 포함한다.

최종 검산에서는 request를 모두 drain한 뒤 logical/used pages는 0, free pages는 total, recurrent slots는 free-list로 돌아오는지 본다. arena reserved가 그대로인 것은 설계일 수 있으므로 leak으로 판정하지 않는다. prefix artifacts가 정책상 유지된다면 request owner가 아니라 cache owner ledger에 남아야 한다.

그다음 같은 workload를 재실행해 capacity threshold와 outputs가 반복되는지 확인한다. threshold가 매번 낮아지면 hidden owner 또는 fragmentation을 조사한다. 동일하다면 deterministic descriptor 문제라는 가설이 강해진다. fix 뒤에는 advertised 마지막 page까지 성공하고 next page가 controlled reject되는지 본다.

운영 문서는 확정·조건부·미검증을 구분한다. tensor shape/stride와 source consumer는 고정 revision에서 확정할 수 있다. 실제 traffic sharing ratio, allocator fragmentation과 peak overlap은 실행 측정이 필요하다. 미검증 항목에는 fixture, metric, 담당 owner와 중단 조건을 붙인다.

이렇게 남긴 worksheet는 계산표 이상의 역할을 한다. model architecture, parallel layout, allocator와 CUDA consumer가 같은 byte identity를 공유하는지 증명하는 계약이며, 다음 cache 정책 장에서 block lifetime과 sharing을 안전하게 논의할 기준선이다.

배포 승인에는 계산 revision, 실제 storage fingerprint, fill-test 결과, parity fixture, resource terminal과 rollback rehearsal을 함께 서명한다. 하나라도 비어 있으면 capacity 숫자는 아직 운영 계약이 아니다.
