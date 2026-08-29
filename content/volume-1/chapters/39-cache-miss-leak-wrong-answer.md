# 39장. cache가 이상하다는 말을 세 사건으로 분해하는 법

같은 2,048-token prompt를 세 번 보냈다. 첫 번째 환경에서는 cached-token counter가 0이고 TTFT가 길지만 답은 맞다. 두 번째에서는 답과 TTFT가 정상인데 요청이 끝날 때마다 available blocks가 조금씩 줄어 결국 OOM이 난다. 세 번째에서는 hit counter가 높고 TTFT도 짧지만 특정 tenant의 첫 output token부터 달라진다.

셋을 “cache 문제”로 묶으면 조사 순서가 무너진다. 첫 사건은 lookup identity·eligibility·eviction 또는 metric 분모의 문제다. 두 번째는 release 뒤 남은 owner·pin·pending job 또는 allocator accounting 문제다. 세 번째는 key isolation·position·physical generation·copy completion의 correctness 문제다. miss는 느리지만 맞을 수 있고 leak은 한동안 맞으며, hit가 높아도 오답일 수 있다.

이 장은 6편의 incident workbook이다. 앞 장의 KV byte, block table, hash, sliding/hybrid address, allocation 수명을 다시 설명하지 않는다. 대신 사건을 `lookup → reserve → write/load → compute read → release → metric` 여섯 경계에 놓고 첫 divergence를 찾는다. vLLM `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang `71de97b264b04dcd514cf904003028aefe9775c8`, Transformers `550d7b3834670483a4df436541272c055dc364bf`, llama.cpp `bb4caa7540188872173c44d161602d9271386413`를 고정 source로 쓴다.

## 39.1 세 증상은 어느 질문에서 갈라지는가

### 39.1.1 miss: 재사용하지 않았지만 계산은 맞다

첫 환경에서 logical prompt는 2,048 tokens다. 그러나 cache lookup 대상이 정확히 2,048이라는 뜻은 아니다. 마지막 token을 logits 때문에 재계산하거나 block/page alignment가 cached eligible length를 줄일 수 있다. chat template·adapter·multimodal identity가 달라 key가 달라졌을 수 있고 prefix caching predicate가 request 종류 때문에 lookup를 건너뛸 수도 있다.

답이 맞다는 사실은 cache correctness를 지지하지만 miss 원인을 말하지 않는다. key mismatch라면 정상 miss이고, eligible entry가 있었지만 너무 일찍 eviction됐다면 policy/capacity 문제다. hit는 있었는데 counter 분모가 external/local 또는 query/token을 섞어 0으로 보였을 수도 있다. 먼저 실제 lookup가 실행됐는지와 normalized identity를 본다.

### 39.1.2 leak: 계산 owner는 끝났는데 resource owner가 남는다

두 번째 환경에서 request 종료 뒤 available가 100→98→96처럼 내려간다. cache content를 유지하는 정상 prefix residency일 수 있으므로 available만 보아 leak라 하지 않는다. evictable candidates가 같은 폭으로 늘었다면 pressure 때 재사용 가능한 정상 cache다. available과 evictable이 모두 줄고 active requests도 0인데 lock/pin/pending owner가 누적되면 leak 가설이 강해진다.

CUDA allocator reserved VRAM이 유지되는 것도 곧 KV leak가 아니다. caching allocator가 tensor storage를 process에 보존할 수 있다. request KV block refs, live cache objects, device tensor allocated bytes, framework reserved bytes를 네 층으로 센다. OOM가 allocator fragmentation인지 logical owner leak인지 분리한다.

### 39.1.3 wrong answer: 빠른 hit가 잘못된 state를 공급한다

세 번째 환경은 성능 지표만 보면 성공이다. cached tokens와 TTFT가 개선됐다. 그러나 tenant B가 tenant A의 adapter/model/template identity로 만든 KV를 읽거나 stale block generation, wrong absolute position, incomplete copy를 읽으면 첫 token부터 달라진다. hit rate 상승은 correctness 증거가 아니다.

오답 조사에는 cache-off differential가 필요하다. 동일 token IDs, model revision, adapter, sampling seed/parameters와 backend 조건에서 cache off 결과를 reference로 두고 first-divergent token 또는 logit/layer checkpoint를 찾는다. stochastic sampling 차이를 cache corruption로 오인하지 않도록 결정 조건과 허용 비결정성을 기록한다.

## 39.2 여섯 경계에 같은 요청 R을 놓는다

### 39.2.1 lookup: 무엇을 query했고 무엇이 hit였는가

lookup ledger는 raw prompt 문자열이 아니라 normalized token IDs와 extra identity, cache group, position/window context를 가진다. query count와 query tokens, hit blocks와 hit tokens를 구분한다. first miss 이후 뒤 block hash가 존재해도 longest prefix 계약상 hit가 아닐 수 있다.

lookup predicate가 false면 cache index를 조사하기 전에 왜 skip됐는지 본다. prompt logprobs, pooling, explicit reset/skip, unsupported topology처럼 implementation branch가 있을 수 있다. counter 0은 cache가 비었다는 뜻이 아니라 query 자체가 0일 수 있다.

R의 lookup를 숫자로 적어 보자. logical input은 2,048 tokens지만 block size 16, final-token recompute 조건 때문에 최대 cache hit length가 2,047일 수 있다. full-block lookup가 2,032까지만 반환하면 16 tokens를 이번 prefill에서 계산한다. hit 2,032를 2,048로 나누면 99.2%, queried eligible 2,032로 나누면 100%다. 어느 rate가 옳다는 문제가 아니라 어떤 질문에 답하는지 다르다.

chat template revision이 system marker 하나를 바꾸면 token 0부터 key chain이 달라져 hit 0이 정상이다. prompt text가 화면상 같아도 BOS, special token, truncation side가 다르면 token IDs가 다르다. adapter extra identity가 key namespace에 들어가면 tenant별 adapter가 다른 것도 정상 miss다. miss 조사에서 namespace를 강제로 제거해 hit를 만들면 세 번째 오답 사건을 만든다.

eviction 가설은 lookup 직전 residency를 요구한다. 이전 request에서 entry를 만들었다는 로그와 현재 hit 0 사이에 pressure eviction가 있었을 수 있다. hash/radix index와 physical owner generation이 lookup 시각에 존재했는지 본다. 과거 cache-store success만으로 current miss를 key bug라 하지 않는다.

metric 가설은 lookup trace와 counter increment를 맞춘다. local query tokens 2,032인데 exported query 0이면 reset/scrape 또는 logger consumer 문제다. lookup가 skip되어 trace 0인데 denominator가 all prompt tokens이면 hit rate 0이 계산될 수 있다. implementation metric 정의를 바꾸기 전에 독자가 원하는 SLO 분모를 별 derived metric으로 만들 수 있다.

### 39.2.2 reserve: hit와 새 suffix가 execution owner가 되는가

lookup result는 아직 active owner가 아닐 수 있다. cached candidate를 touch/lock하고 suffix blocks/locations를 reserve해야 한다. lookup hit 뒤 allocation failure가 나면 computed prefix를 release하고 request가 waiting으로 돌아갈 수 있다. metric이 lookup hit만 세면 실행이 재사용하지 못했는데 hit가 높게 보인다.

reserve ledger에는 requested new tokens/blocks, touched existing IDs, new physical IDs, failed group, rollback를 둔다. hybrid cache에서 group별 common hit가 다르면 scheduler가 실제로 쓸 수 있는 aligned common prefix를 기록한다.

lookup에서 R blocks `[7,2]` hit를 얻었지만 suffix reserve가 실패했다고 하자. manager는 hit candidates를 touch/lock했다가 rollback에서 놓는다. lookup counter는 2,032 hit를 기록할 수 있지만 execution는 request를 다음 iteration까지 시작하지 못한다. TTFT는 그대로거나 더 길어진다. “hit 높음, TTFT 불변”의 first divergence는 metric가 아니라 lookup success와 schedule commit 사이일 수 있다.

hybrid group A는 2,032 tokens, group B sliding cache는 1,024, recurrent group C는 usable state 0일 수 있다. model forward가 공통 computed frontier 1,024 또는 다른 native reconciliation를 요구하면 A hit bytes가 있어도 2,032 prefill를 건너뛰지 못한다. group별 local hit를 단순 합해 “cache가 절반 일을 안 했다”고 결론 내리지 않는다.

reserve가 succeeded라도 remote load destination blocks를 allocation했을 뿐 content는 아직 없다. execution owner와 data-ready owner를 나눈다. runner table에 publish하는 시점이 load completion 전이면 compute stream fence가 필요하다. reserve success metric와 usable cached tokens metric을 분리한다.

allocation rollback가 cached hit metadata를 irreversible eviction할 수 있다. 첫 retry에서는 hit였지만 reserve failure 뒤 두 번째 retry가 miss가 된다. cache key가 변한 것이 아니라 rollback가 residency를 바꿨다. transaction ID와 eviction event를 연결한다.

### 39.2.3 write/load: bytes가 준비됐다는 completion이 있는가

local prefill write, COW copy, CPU/remote KV load는 모두 physical destination를 채운다. metadata가 destination ID를 publish했어도 stream/event 또는 transfer completion 전 compute가 읽으면 오답이다. copy/load 실패를 miss로 fallback해 recompute하는지 request를 fail하는지 native policy를 확인한다.

pending transfer는 release를 늦춰 leak처럼 보일 수 있다. owner와 completion condition가 있고 끝난 뒤 pin이 줄면 정상 지연이다. completion가 왔는데 pending map에서 제거되지 않으면 leak다. timeout으로 pin을 지우기 전에 DMA writer/reader가 끝났는지 증명한다.

write와 load는 방향이 달라도 destination validity를 commit한다. local prefill write는 scheduled token positions가 kernel completion 뒤 valid해지고, COW copy는 source generation과 valid prefix length를 destination에 복제하며, remote load는 descriptor key와 destination block generation을 맞춘다. 완료 event 하나가 어느 operation과 generation인지 기록한다.

copy completion보다 table publish가 먼저인 것은 compute가 event를 기다리면 안전할 수 있다. host log 순서만 보고 race라 하지 않는다. 반대로 Python copy call return을 completion으로 오인하면 nonblocking stream에서 wrong answer가 난다. producer stream record와 consumer stream wait를 본다.

load가 partial failure를 보고 invalid block IDs를 반환하면 downstream dependent prefix를 recompute하거나 cache entry를 evict할 수 있다. failure를 성공 hit로 counter에 남기면 hit와 computed 합계가 어긋난다. external requested, transferred, valid loaded, recomputed를 따로 센다.

checksum/hash가 맞아도 wrong destination position에 썼다면 오답이다. content identity와 address identity를 함께 검증한다. R key K가 block 11 generation 5를 의도했는데 descriptor가 block 11 generation 4를 가리키면 integer ID는 같아도 stale load다.

### 39.2.4 compute read: 어느 logical position이 어느 generation을 읽었는가

오답의 강한 증거는 request R의 first-divergent position을 physical address까지 잇는 것이다. logical block/offset 또는 token pool/cell index, physical generation, layer group, local KV head, valid length, table/mapping generation을 기록한다. cache off reference와 같은 position의 logits 또는 layer output을 비교한다.

첫 layer부터 divergence면 loaded embedding/position 또는 early attention KV mapping을 본다. 여러 layers 뒤 시작하면 group mapping, layer-specific cache, recurrent state를 본다. final token만 다르다는 사실로 sampler 문제를 먼저 단정하지 않는다.

first-divergent token은 output ordinal만이 아니라 context position을 가진다. prompt 2,048 뒤 첫 generated token을 만드는 logits가 다르면 cached prompt state가 이미 달랐을 가능성이 높다. first output은 같고 17번째부터 다르면 newly appended KV, window shift, partial-page boundary도 본다.

layer checkpoint를 모든 hidden states로 저장할 필요는 없다. norm, checksum 또는 selected coordinate를 동일 deterministic fixture에서 비교해 divergence layer를 좁힌다. numerical backend tolerance를 미리 정한다. 작은 floating difference가 sampling threshold를 넘긴 사건과 gross wrong block read를 구분한다.

position 원장은 RoPE absolute position, SWA window-local index, block offset, sequence length를 함께 가진다. logical token IDs가 같아도 position 2,048 state를 position 0으로 재사용하면 key hash가 같아 보일 수 있다. cache key에 position context가 들어가는지, admission predicate가 reuse를 금지하는지 본다.

tenant isolation는 request header가 아니라 effective model execution identity를 쓴다. gateway tenant ID가 달라도 same base model cache를 안전하게 공유할 수 있는 경우가 있고, same tenant라도 adapter revision이 다르면 공유하면 안 된다. correctness key namespace를 조직 경계와 기계적으로 같게 두지 않는다.

### 39.2.5 release: logical finish와 physical reuse를 나눈다

request finish/abort는 refs와 radix locks, private blocks, connector pins, async writer fences를 각각 release한다. cached complete content는 ref 0으로 남아 evictable일 수 있다. partial/uninitialized는 즉시 free queue로 갈 수 있다. aggregate available가 원래 값으로 즉시 돌아와야 한다는 기대가 틀릴 수 있다.

release ledger는 owner before/after, delayed conditions, pool transition와 table/map removal을 갖는다. double cleanup은 live shared block을 premature eviction하고 missing cleanup은 leak를 만든다. request map 부재만으로 모든 delayed owner가 끝났다고 말하지 않는다.

R 자연 finish와 abort를 나란히 적는다. 자연 finish는 final output commit 뒤 private tail refs를 놓고 shared complete blocks는 cached eligible로 넘긴다. abort는 in-flight output/load가 남아 delayed owner를 만들 수 있고 incomplete destination를 cacheable로 commit하지 않는다. 두 경로의 final partition가 달라도 owner 합은 닫혀야 한다.

leak 사건에서 가장 오래된 unreleased owner를 찾는다. request ref가 남으면 finish/abort inverse mutation를, radix lock이면 last-node handoff를, transfer pin이면 completion callback/lease를, reserved allocation이면 transaction rollback를 본다. “KV cache used”라는 합계에서 시작해도 root는 owner kind별 함수다.

double release는 available를 정상 이상으로 올리거나 free queue duplicate를 만든다. 당장 OOM와 반대 증상이라 health check가 놓칠 수 있다. 나중에 두 requests가 같은 physical ID를 받아 오답이 난다. leak audit는 missing owner뿐 아니라 duplicate availability도 본다.

framework tensor reference는 block manager ref와 별도다. Python closure, future, debug history가 cache object를 붙잡아 device storage를 살릴 수 있다. manager partition가 정상인데 live tensors가 증가하면 object graph를 본다. allocator reserved만 유지되면 live tensor가 아닐 수 있다.

### 39.2.6 metric: 어느 분모와 어느 시각을 센 값인가

hit rate가 `hit tokens / queried tokens`인지 `hit requests / requests`, local hits만인지 external transferred tokens까지 포함하는지 확인한다. lookup 시점 counter와 execution commit counter가 다를 수 있다. reset-on-read stats는 scrape 순서에 따라 interval을 나눈다.

usage gauge도 physical allocated, non-free, non-evictable, request-visible bytes 중 무엇인지 본다. 서로 다른 scheduler steps의 active, available, evictable을 합치지 않는다. metric은 사건의 증거지만 정의를 읽지 않으면 새로운 착시가 된다.

metric sampling cadence는 짧은 owner transition를 숨긴다. scrape 15초 사이 pin이 생겼다 사라지면 gauge에는 안 보이지만 latency에 영향을 줄 수 있다. leak처럼 지속 상태는 oldest age와 high-water mark가 유용하고 wrong answer는 request trace가 필요하다. metric 종류를 사건 시간 규모에 맞춘다.

reset-on-read counter는 두 collector가 읽으면 분모를 나눌 수 있다. exporter와 debug endpoint가 같은 stats를 consume하는지 source를 본다. 한 scrape에서 hit 0이 나온 것이 실제 interval miss인지 다른 reader가 먼저 reset했는지 확인한다.

label cardinality를 피하려 request ID를 metric label로 넣지 않는다. aggregate anomaly에서 trace exemplar나 sampled request ledger로 이동한다. tenant/model/cache group 같은 bounded labels도 deployment 규모를 계산한다. correctness 조사 detail은 로그/trace에 둔다.

## 39.3 세 원장이 분모와 owner와 정답을 분리한다

### 39.3.1 분모 원장

2,048-token prompt에서 logical tokens 2,048, lookup eligible 2,032, local queried 2,032, local hit 1,024, external requested 1,008, external loaded 768, recomputed 256이라고 하자. 숫자는 구현 alignment와 마지막-token 정책에 따라 달라질 수 있지만 합계 관계를 명시해야 한다.

`computed = local hit + valid external load + recompute`처럼 native metric 계약을 확인한다. external requested를 loaded로 세거나 last token을 eligible 분모에 넣으면 hit rate와 computed tokens가 모순된다. request-level 평균과 token-weighted rate도 결과가 다르다.

### 39.3.2 owner 원장

physical pool 100 blocks라면 `active + locked/pinned/delayed + cached-evictable + uninitialized + reserved + special = 100` partition를 만든다. shared ref edges는 physical count와 별도로 센다. ID set union과 intersection도 검증해 duplicate free와 orphan가 count에서 상쇄되지 않게 한다.

steady state 20회 반복에서 active가 매번 0으로 돌아오고 pending jobs가 0이며 evictable+uninitialized가 같은 범위로 돌아오면 정상 cache residency일 가능성이 크다. locked/pinned가 반복마다 +1이면 leak다. allocator reserved만 유지되고 live KV partition는 정상이라면 framework pool behavior다.

### 39.3.3 correctness 원장

reference run과 cache-on run의 normalized input identity, model/adapter revision, sampling condition를 고정한다. token position별 chosen token만 비교하지 않고 가능하면 first-divergent logits 또는 layer checkpoint를 찾는다. wrong key는 first attention부터, stale late-layer group은 해당 group 이후 divergence할 수 있다.

physical trace에는 key/hash, block/cell ID와 generation, absolute position, valid length, copy/load event를 붙인다. hash equality만으로 current physical generation을 증명하지 않는다. table ID가 맞아도 position/window metadata가 틀릴 수 있다.

세 원장의 canonical 진단표는 다음 하나다. 사건별 표를 새로 만들기보다 이 세 행에 request generation을 맞추고, 구현 위치는 `source` 열에서 시작한다. 링크는 정답 위치가 아니라 해당 원장의 producer와 consumer를 함께 걷기 위한 고정 출발점이다.

| 원장 | 반드시 닫을 등식·질문 | source |
|---|---|---|
| 분모 | eligible·queried·local hit·valid external load·recompute가 실제 computed frontier와 합의하는가 | [vLLM lookup·통계](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_manager.py#L202-L337), [SGLang cache metric](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/observability/metrics_collector.py#L285-L370) |
| owner | active·locked/pinned·cached-evictable·uninitialized·reserved·special partition와 generation별 inverse mutation이 닫히는가 | [vLLM allocation transaction](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_manager.py#L338-L489), [SGLang allocation 전 eviction](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/common.py#L110-L145) |
| correctness | normalized identity·position·physical generation·copy completion이 reference의 최초 일치/불일치 tensor와 합의하는가 | [Transformers continuous cache update](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache.py#L525-L598), [llama.cpp KV cell 상태](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L681-L700) |

### 39.3.4 세 원장을 하나의 2,048-token timeline에 맞춘다

request A가 prompt P를 먼저 계산해 full-block eligible 2,032 tokens를 cache에 남겼다고 하자. 분모 원장에는 logical 2,048, eligible/store 2,032가 들어간다. owner 원장에는 physical blocks generation 12, ref 0 cached-evictable가 들어간다. correctness 원장에는 token IDs, model/adapter/template identity, absolute positions 0–2,031이 들어간다.

request R lookup가 시작된다. 정상 miss 사건에서는 R identity가 adapter revision B라 A entry를 query하지 않거나 different key로 miss한다. 분모는 query/hit 2,032/0, owner는 A blocks 그대로 cached, correctness는 R이 all 2,048을 recompute해 reference와 맞는다. 느리지만 세 원장은 모순되지 않는다.

metric miss 사건에서는 R이 A와 same identity로 physical generation 12를 touch하고 2,032 computed frontier를 얻는다. runner compute는 suffix 16만 처리한다. 분모 trace는 hit인데 exporter가 reset되어 0이다. owner와 correctness는 정상이고 metric 원장만 execution 사실과 어긋난다. first divergence는 metric consumer다.

leak 사건에서는 R이 정상 hit/compute/answer를 마치지만 finish 뒤 block refs는 0이어도 transfer pin X 또는 radix lock L이 남는다. 분모와 correctness는 정상이다. owner partition에서 cached-eligible로 돌아가야 할 blocks가 locked/pending에 머문다. 반복마다 L owner가 하나씩 증가하면 release 경계가 first divergence다.

wrong-answer 사건에서는 R tenant identity가 B인데 lookup key가 A와 충돌해 generation 12를 touch한다. 분모 원장은 2,032/2,032로 완벽하고 owner partition도 합이 맞는다. correctness 원장에서 effective adapter와 KV producer adapter가 다르다. first layer checkpoint부터 divergence한다. owner leak detector와 hit metric은 모두 green일 수 있다.

stale-generation wrong-answer에서는 identity도 같지만 cached index가 physical ID7 old generation 12를 가리키고 pool current owner는 generation 13이다. owner partition는 generation을 빼고 integer IDs만 세면 맞아 보인다. correctness 원장과 generation-aware owner 원장이 lookup 경계에서 모순된다. ID set count만으로 충분하지 않다.

load-race wrong-answer에서는 key와 generation도 맞다. reserve가 destination generation 14를 만들고 transfer X가 bytes를 쓴다. compute read가 X completion event보다 앞서 first divergence가 생긴다. 세 원장에 event ordering를 넣지 않으면 모두 맞아 보인다. correctness 원장의 copy-complete 항이 필요한 이유다.

이 timeline은 조사 우선순위를 정한다. 답이 맞고 hit 0이면 correctness checkpoint보다 lookup predicate/분모를 먼저 본다. 답이 맞고 owner가 누적되면 key를 바꾸지 말고 release inverse를 본다. 답이 틀리고 hit가 높으면 성능 개선을 롤백한 뒤 identity·generation·event를 좁힌다. 같은 여섯 경계를 쓰되 출발 원장이 다르다.

### 39.3.5 first divergence 뒤의 연쇄를 원인으로 오인하지 않는다

lookup key collision 뒤 wrong KV를 읽으면 output token이 달라지고 그 token을 append한 새 KV도 reference와 달라진다. 뒤 layers와 다음 steps 전체가 divergence한다. 마지막 sampler에서 처음 눈에 띄었다고 sampler가 원인이 아니다. earliest checkpoint를 찾는다.

release lock leak 뒤 available가 줄면 eviction pressure와 preemption, TTFT 상승, 마지막에 OOM가 이어진다. OOM allocator stack은 root가 아니다. 첫 request finish에서 lock delta가 baseline으로 돌아오지 않은 것이 divergence다.

metric denominator 오류는 운영자가 cache option를 바꾸게 해 실제 miss를 만들 수 있다. incident timeline에서 자동 policy action와 config rollout도 포함한다. 처음에는 관측 오류였는데 잘못된 remediation가 execution behavior를 바꾼다. 변경 전후 generation을 나눈다.

강한 증거는 원인 경계 바로 양쪽 state다. lookup 전 normalized identity와 lookup result, release 전 owner ledger와 release 후 partition, compute read 전 event/table generation과 first layer output처럼 붙인다. 멀리 떨어진 OOM 또는 final text만으로 causal jump하지 않는다.

**구현 위치는 진단표의 source 열로 읽는다.**

**vLLM: lookup predicate와 세 token 합계를 함께 읽는다**

vLLM `KVCacheManager`는 prefix lookup enable predicate와 stats recording를 분리하고, prompt 전체 hit여도 logits를 위해 마지막 token을 재계산하는 상한을 둔다. [`get_computed_blocks()` 경계](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_manager.py#L202-L337)를 읽으면 prompt length를 그대로 hit denominator로 쓰면 안 되는 이유가 보인다.

hybrid coordinator는 groups가 공통으로 제공할 수 있는 longest hit, alignment와 uncached common prefix를 조정한다. 한 group hit가 2,048이어도 다른 group이 1,024이면 실행 computed frontier는 제한될 수 있다. [hybrid lookup 조정](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_coordinator.py#L685-L842)을 group별 ledger에 연결한다.

vLLM stats는 local hit, external transfer, computed token 관계를 별 필드로 가진다. [`PrefixCacheStats`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/metrics/stats.py#L255-L343)의 update invariant와 logger consumer를 함께 읽는다. 대시보드의 prefix hit 하나로 remote load와 recompute를 합치지 않는다.

logger는 stats를 Prometheus-facing counters/gauges로 옮기는 또 하나의 semantic boundary다. [usage와 prefix metric 소비](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/metrics/loggers.py#L978-L1045)에서 누적 counter인지 interval sample인지, local/external labels가 어디서 갈리는지 본다. manager trace는 맞는데 dashboard가 틀리면 이 consumer에서 first divergence를 찾는다.

residency sample은 request hit와 다른 질문에 답한다. physical cached content가 얼마나 오래 남았는지 sampled distribution를 cache hit numerator처럼 해석하지 않는다. cache pressure가 낮아 residency는 길어도 identity mismatch 때문에 R hit는 0일 수 있다. 반대로 churn가 커도 hot prefix가 반복 touch되어 hit rate는 높을 수 있다.

prefix reset 경로도 miss 사건에 영향을 준다. reset가 cached metadata를 비우는 데 실패하거나 특정 active state 때문에 거절될 수 있다. 운영자가 “cache reset 완료”라고 기록했는데 old generation가 남으면 wrong-answer fixture와 연결된다. reset 요청, validation result, actual evicted IDs를 구분한다.

leak는 block pool refs/free queue와 deferred-free owners로 내려간다. wrong answer는 request block table generation, hash extra identity, hybrid group/position로 내려간다. vLLM이라는 이유로 모든 incident를 manager 한 파일에서 찾지 않는다.

vLLM의 분모 원장을 더 구체적으로 읽자. request가 prefix lookup를 skip하면 `record_prefix_cache_stats()`도 조건에 맞춰 count를 다룬다. stats 객체가 interval마다 reset될 수 있으므로 request trace의 actual hit blocks와 exporter interval counter를 대조한다. logger의 metric family가 local query/hit, external query/hit 또는 residency sample 중 무엇을 소비하는지 확인한다.

`max_cache_hit_length = prompt_length - 1`은 all-hit prompt에서도 forward logits를 얻기 위한 마지막 token 경계다. block alignment 때문에 one token이 아니라 whole last block recompute가 될 수 있다. prompt 2,048, block 16이면 2,032 cached가 구현상 full hit일 수 있다. 이를 16-token miss라고 alert하면 false positive다.

pooling/prompt-logprob 같은 lookup-disable request를 all requests denominator에 넣을지 운영 metric 목표를 결정한다. implementation local hit rate와 product-level “전체 요청 중 cache로 절약한 tokens”는 다른 metric이다. 원본 counter 의미를 바꾸지 않고 derived SLO를 만든다.

external transfer가 invalid blocks를 보고하면 scheduler는 affected requests의 computed frontier를 조정하고 recompute할 수 있다. local hit counter가 높고 external loaded도 높지만 final computed savings가 낮을 수 있다. stats 합계에서 requested transfer와 valid transferred를 구분한다. failure 뒤 hash metadata eviction가 있었는지도 owner 원장에 넣는다.

leak source walk는 request `free()`에서 block groups와 pool queue로 내려간 뒤 async processed-step fence까지 이어 간다. pool ref 0인데 queue에 없는 ID, queue에 ref>0 ID, delayed entry condition 충족 뒤 잔류를 찾는다. aggregate usage logger는 이 structural invariant를 대신하지 않는다.

wrong-answer source walk는 key generation에서 lookup block ID, request table row, runner device table, attention cache group까지 한 request generation으로 잇는다. local hit라는 fact가 hash namespace와 physical generation correctness를 동시에 보장하지 않는다. extra identity와 address owner를 별도로 검증한다.

**SGLang: full·SWA·Mamba pool 분모를 나눈다**

SGLang metrics collector는 cache hit와 available/evictable/used를 cache 종류별로 다룰 수 있다. [metric 수집 경계](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/observability/metrics_collector.py#L285-L370)에서 full pool shortage를 SWA pool available로 상쇄하지 않는다.

allocation 공통 경로는 available가 부족한 만큼 eviction를 요청한다. [allocation 전 eviction](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/common.py#L110-L145)은 total eviction count보다 어느 pool 부족분을 채웠는지 보게 한다. lock leak이면 evictable가 낮아 eviction 호출이 capacity를 못 만든다.

allocation failure 진단은 request row, full/SWA/Mamba 요구량과 각 pool availability를 같이 남겨야 한다. 한 pool의 “available=100” 로그만 보면 다른 pool 0 때문에 실패한 사실을 놓친다. request가 실제로 요구한 group set과 allocation result를 같은 transaction ID로 잇는다.

SWA radix cache의 sanity는 tree walk로 계산한 evictable size와 LRU list accounting를 비교한다. 두 값이 다르면 available 감소를 실제 pin leak로 단정하기 전에 list/metric mutation divergence를 본다. 그러나 sanity assertion를 끄고 metric만 맞추는 수정은 victim selector structure를 고치지 않는다.

cache hit metric는 matched logical tokens와 physical reusable tokens가 page/window alignment 때문에 다를 수 있다. radix match length가 길어도 allocator가 usable prefix를 줄일 수 있다. lookup trace, request token mapping, actual forward query length를 함께 본다.

hit가 높은데 오답이면 radix path identity만 보지 않는다. request-to-token KV location, SWA absolute position/window, recurrent/Mamba slot generation을 correctness 원장에 넣는다. full/SWA dual lock과 LRU sanity가 metric partition와 일치하는지도 확인한다.

SGLang 사건에서는 full pool available 20, evictable 30인데 SWA available 0, evictable 0일 수 있다. aggregate 50을 보고 allocation 가능하다고 결론 내리면 안 된다. request가 두 pools 모두 필요하면 bottleneck pool이 admission를 막는다. Mamba recurrent state가 별 slot을 요구하는 hybrid model도 같은 원리다.

common allocation path가 shortage만큼 eviction를 요청해도 locked nodes밖에 없으면 실제 freed가 부족하다. eviction result와 retry allocation를 연결한다. eviction counter 증가가 allocation success를 뜻하지 않는다. compressed node size 때문에 overshoot하거나 dual pool 중 하나만 회복될 수 있다.

radix cache sanity는 tree-derived evictable size와 LRU-list-derived size를 비교할 수 있다. 둘이 다르면 metric bug가 아니라 structural list/lock mutation divergence일 수 있다. leak 조사에서 available 단조 감소와 sanity mismatch가 함께 나면 missed inc/dec lock 또는 node removal를 본다.

wrong answer가 window boundary에서만 나면 full radix key가 맞아도 SWA evicted sequence length와 absolute positions가 틀릴 수 있다. logical hit tokens를 full context reusable tokens로 해석하지 않는다. recurrent state는 token blocks가 아니라 slot generation이 correctness identity다.

cancel leak는 request pool row와 token locations, last radix node locks를 함께 본다. abort가 row를 filter했지만 last node unlock를 빼먹으면 evictable가 줄고 token locations가 pin된다. 반대로 node를 unlock했지만 in-flight batch가 locations를 읽으면 오답이다. batch completion ordering를 붙인다.

**Transformers: live cache와 allocator reserved를 가른다**

continuous path는 `PagedAttentionCache`와 `BlockManager`의 initialized, uninitialized, shared refs를 owner 원장에 넣는다. request finish 뒤 complete initialized blocks가 남는 것은 정상 cache residency다. partial blocks와 request mapping이 남으면 leak 후보다.

classic Cache에서는 logical sequence length, tensor capacity, layer offload/prefetch state, Python live references를 센다. crop이 logical length만 줄이고 storage capacity를 즉시 allocator에 반환하지 않을 수 있다. VRAM graph 하나로 KV leak를 단정하지 않는다.

continuous manager의 error path가 future request states를 실패시키고 scheduler refs를 정리하는지 본다. batch error에서 current pair만 실패하고 previous pair state가 남으면 cancel/exception 뒤에만 leak가 난다. manager active/waiting maps와 IO pair futures, block refs를 한 owner timeline에 둔다.

classic cache offload에서는 GPU allocated가 줄고 CPU live tensors가 늘 수 있다. GPU-only dashboard는 leak가 해결된 것처럼 보이지만 total hierarchical owner가 누적될 수 있다. 38장의 tier 수명을 여기서는 incident evidence로만 사용한다. prefetch completion 뒤 old CPU/GPU copy owner가 둘 남지 않는지 본다.

wrong answer는 layer별 cache update position과 offload/prefetch completion, batch reorder를 본다. continuous path의 special trash/sentinel addresses가 real token에 적용됐는지도 확인한다. classic과 continuous의 cache state를 같은 block counter로 맞추지 않는다.

classic `Cache.get_seq_length()`가 2,048이라고 tensor capacity가 정확히 2,048이거나 모든 layers가 동일 materialized state라는 뜻은 아니다. dynamic/static/offloaded cache implementation에 따라 capacity와 residency가 다르다. crop 뒤 semantic length 1,024로 줄어도 backing allocation가 남을 수 있다. 답이 맞고 repeated grow/crop에서 live object count가 안정적이면 reserved memory를 leak로 부르지 않는다.

offloaded cache는 current layer와 prefetch layer tensor가 device에 겹치는 순간이 있다. 순간 VRAM peak를 steady leak로 오인하지 않는다. prefetch future가 error/abort 뒤 남아 layer tensors를 붙잡으면 leak가 될 수 있다. layer index, event completion, Python owner를 기록한다.

continuous `BlockManager`에서 initialized ref0 blocks는 free count에 포함되면서 hash content를 보존한다. active requests 0인데 initialized bytes가 남는 것은 정상이다. incomplete blocks나 request block maps, future states가 반복마다 증가하는지 본다. allocator reserved와 manager free states를 같은 stacked graph에 단순 합하지 않는다.

wrong answer가 batch compaction 뒤 나타나면 `FutureRequestState`와 new token/logprob row mapping, block table pair generation을 본다. finished/PENDING request를 skip하면서 token index는 소비해야 뒤 rows가 맞는다. cache 자체가 정상인데 output row reconciliation가 틀린 사건도 “cache 오답”처럼 보일 수 있다.

**llama.cpp: slot reuse와 RAM prompt cache를 섞지 않는다**

llama.cpp server의 slot-local common-prefix reuse, RAM prompt cache, unified KV cells는 서로 다른 reuse 경계다. global block hash hit rate처럼 하나의 counter로 만들면 denominator가 없다. request가 어느 slot/context를 재사용했고 cell sequence/position metadata가 무엇인지 본다.

KV cache buffer memory breakdown은 backend buffers별 bytes를 보여 주지만 live sequence ownership를 직접 말하지 않는다. [KV memory breakdown](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L681-L700)을 cell used/empty와 연결한다. context 전체 memory에는 model, KV/recurrent와 compute buffers가 섞이므로 RSS 유지와 KV leak를 동일시하지 않는다.

context memory breakdown도 model weights, KV/recurrent memory와 compute buffers를 분리해 읽어야 한다. 동일 context가 고정 compute buffer를 유지하는 것은 prompt cache leak가 아니다. server slot 수와 context count가 늘어 buffer instances가 증가하는지, 한 context 안 cells used만 증가하는지 나눈다.

slot-local prefix reuse가 실패해도 answer는 full recompute로 맞을 수 있다. miss ticket에서 global KV hash index를 찾지 말고 slot prompt tokens와 common-prefix detection, context shift/reset를 본다. RAM prompt cache restore가 성공했어도 compute buffer preparation 때문에 TTFT가 그대로일 수 있다.

wrong answer fixture는 same token IDs와 sampling 조건에서 first divergence를 찾고 seq-id, position, cell index와 context shift 전후를 비교한다. paged block generation field를 llama.cpp에 있다고 가정하지 않는다. cell ownership과 graph update ordering이라는 native 증거를 쓴다.

llama.cpp 메모리 breakdown에서 KV buffer가 일정하게 크게 보이는 것은 context가 고정 capacity cache를 미리 잡았기 때문일 수 있다. request 종료마다 buffer 자체를 반환하는 설계라고 가정하지 않는다. cell used count와 sequence associations가 baseline으로 돌아오는지 본다.

server slot common-prefix reuse는 같은 slot의 prompt token 비교와 context state reuse일 수 있고 global cross-request block hash query와 분모가 다르다. RAM prompt cache도 serialized state restore 비용과 identity를 가진다. 세 경로를 합쳐 hit tokens로 내면 TTFT와 computed work 관계를 해석할 수 없다.

context shift 뒤 오답이면 first-divergent position과 cell `pos`, sequence ID association, shift update completion을 본다. long context에서만 발생하는 boundary fixture가 유용하다. total KV memory나 prompt cache hit rate는 position mapping correctness를 증명하지 않는다.

slot release 뒤 response serialization가 남는 정상 수명과 active cell associations가 남는 leak를 구분한다. task/slot owner는 끝났는데 seq cells가 계속 used라면 cleanup path를 본다. fixed buffer bytes가 남되 cells가 empty면 allocator residency다.

**성능 cache와 정답 state를 섞지 않는 중간 확인.**

2,048-token fixture의 세 실행을 마지막으로 나란히 놓자. miss 실행은 lookup hit가 0이고 all tokens를 다시 계산했지만 correctness reference와 일치했다. leak 실행은 lookup과 계산, 답이 모두 맞았지만 release 뒤 owner partition가 닫히지 않았다. wrong-answer 실행은 lookup와 TTFT, owner count까지 정상처럼 보였으나 effective identity와 physical generation 또는 read ordering가 달랐다.

이 세 실행에서 공통 cache hit rate 하나는 진단력이 거의 없다. miss는 낮고 leak는 정상이며 wrong-answer는 높다. available gauge도 miss에서는 정상, leak에서는 감소, wrong-answer에서는 정상일 수 있다. final text correctness는 miss/leak에서 정상이고 wrong-answer에서만 깨진다. 그래서 세 원장을 별도로 유지한다.

분모 원장은 성능 cache가 실제로 어느 compute를 건너뛰었는지 말한다. logical prompt length에서 eligible alignment와 lookup queries를 빼고, local hit·valid external load·recompute를 합쳐 actual computed frontier와 맞춘다. query도 하지 않은 token을 miss denominator에 넣거나 failed load를 hit savings로 세지 않는다.

owner 원장은 memory capacity가 어느 수명에 묶였는지 말한다. request refs와 radix locks, transfer pins, writer fences, cached eligible, uninitialized, reservations를 physical generation별로 partition한다. aggregate count뿐 아니라 ID union와 disjointness를 본다. ref edge 수와 physical resources를 섞지 않는다.

correctness 원장은 재사용 state가 현재 execution identity와 position에 맞는지 말한다. token IDs 외 model·adapter·template/multimodal identity, block/cell generation, absolute/window position, layer group, copy/load event를 연결한다. final sampling output 전에 first-divergent checkpoint를 찾는다.

여섯 경계는 이 원장들이 만나는 위치다. lookup에서 identity와 query denominator가 만나고 reserve에서 hit candidate와 physical owner가 만난다. write/load에서 destination owner와 completion가 만나며 compute read에서 physical generation와 correctness가 만난다. release에서 logical terminal과 resource lifetime가 만나고 metric에서 이 모든 사건이 축약된다.

first divergence는 원장끼리 처음 모순되는 곳이다. execution trace hit 2,032인데 exporter 0이면 metric다. request terminal 뒤 pin owner condition가 완료됐는데 entry가 남으면 release다. key와 owner는 맞지만 compute가 copy event보다 먼저 읽으면 write/load→read ordering다. 뒤의 OOM나 final wrong token은 first divergence가 아니다.

복구 우선순위도 세 사건이 다르다. wrong answer는 suspect generation를 즉시 격리하고 cache reuse를 안전하게 disable할 수 있다. leak는 new allocation를 제한하고 owner를 drain해 추가 capacity/corruption 위험을 막는다. miss는 correctness와 capacity가 안전한 상태에서 identity, policy와 critical path를 최적화한다.

임시 대응과 완료를 구분한다. cache disable은 wrong answer를 피하고 miss를 강제하는 안전 조치다. process restart는 leak owner를 폐기해 capacity를 되찾는 조치다. metric alert mute는 관측 noise를 줄인다. 어느 것도 root key·release·denominator mutation를 고쳤다는 증거는 아니다.

독자는 incident ticket에 “cache 문제”라고 쓰는 대신 첫 관측을 명시할 수 있어야 한다. “lookup query 2,032/hit 0이며 normalized identity 동일, eviction 없음”, “20 cancels 뒤 transfer completion 20인데 pins 20 잔류”, “tenant B cache-on layer 0 checksum가 reference와 다르고 hit physical generation가 A owner”처럼 쓴다. 이 문장은 바로 다음 source boundary를 가리킨다.

실전에서 모든 evidence가 한 번에 있지는 않다. generation trace가 없으면 gap을 인정하고 표본 instrumentation를 추가한다. prompt/KV content를 그대로 기록하지 않고 pseudonymous request, digest, coordinates와 events를 쓴다. 불확실한 owner를 free하거나 cache entry를 정상으로 선언하지 않는다.

6편의 결과는 KV cache를 단순 메모리 최적화로 보지 않는 것이다. 재사용은 compute를 줄이는 성능 계약이고, physical ownership를 유지하는 lifetime 계약이며, 같은 model state를 공급하는 correctness 계약이다. 세 계약은 함께 성공해야 cache hit가 가치가 있다.

6편에서 KV cache는 tensor bytes이면서 주소 table, content identity, owner lifetime였다. 이 장의 세 사건은 그 축이 서로 독립적으로 실패할 수 있음을 보여 준다. miss는 identity lookup가 실패해도 정답을 재계산한다. leak는 owner release가 실패해도 한동안 정답을 만든다. wrong hit는 성능이 좋아도 정답 state를 오염시킨다.

조사자는 여섯 경계를 따라가되 세 원장을 섞지 않는다. 분모 원장은 무엇을 query·hit·load·compute했는지, owner 원장은 어느 physical generation을 누가 붙잡는지, correctness 원장은 같은 input이 어디서 처음 달라졌는지 말한다. 세 원장이 처음 모순되는 함수와 state mutation이 first divergence다.

다음 편은 CUDA execution으로 내려간다. cache incident가 address와 owner까지 정상인데도 wrong result나 stall이 남으면 stream/event, kernel layout, memory hierarchy의 실행 계약을 조사해야 한다. 39장에서 만든 request·generation·position 상관관계가 CUDA trace와 kernel source를 잇는 출발점이 된다.

그때도 “kernel 문제”라는 새 포괄어로 되돌아가지 않는다. cache table을 소비하는 stream, launcher가 전달한 stride와 valid length, kernel이 계산한 physical address, completion event를 같은 request generation에 붙인다. cache 원장과 CUDA 실행 원장이 처음 갈리는 지점이 다음 first divergence다.

이 장의 최종 산출물은 checklist가 아니라 세 개의 닫힌 incident narrative다. 정상 miss는 identity 차이를 증명하고, leak는 남은 owner와 release condition를 특정하며, 오답은 first-divergent checkpoint에서 wrong identity·generation·position·event 중 하나로 좁힌다. 증상만 지운 reset나 평균 metric 개선은 이 narrative를 대신하지 못한다.

세 narrative가 모두 같은 2,048-token fixture와 고정 source 좌표로 명확하고 독립적으로 다시 재현 가능해야 6편의 cache 수명 논의가 실제 운영 판단과 기술적으로 안전하고 검증 가능한 복구 종료 기준으로 완전히 닫힌다.

이 기준은 다음 버전에서도 증상 이름이 아니라 최초로 어긋난 분모·owner·정답 경계를 다시 찾게 한다.

## 39.4 일곱 사건을 첫 divergence에서 닫는다

### 39.4.1 hit=0, memory 정상, 답 정상

첫 분기는 lookup 전이다. normalized tokens와 adapter/model/template identity가 동일한지, caching predicate가 true인지 본다. query=0이면 option/skip path고 query>0 hit=0이면 key mismatch, eviction, cold cache로 간다. memory 정상이라는 사실은 index identity를 설명하지 않는다.

강한 검증은 expected eligible prefix만 hit하고 computed/recompute tokens가 함께 변하는 것이다. hit counter만 올랐는데 compute가 그대로면 metric 또는 execution commit 경계가 어긋났다.

R의 사건을 시간순으로 재구성한다. request A가 같은 normalized tokens로 cacheable full blocks를 만들고 ref 0 cached state로 남겼다. R lookup 직전 pressure eviction event는 없고 physical candidates도 존재한다. 그런데 R trace에는 lookup query 자체가 없다. 이 경우 hash map을 뒤질 이유가 없다. caching predicate 또는 ingress option/요청 특성이 first divergence다.

query tokens 2,032, hit 0이 기록됐다면 key 원장을 비교한다. token IDs는 같지만 adapter revision extra가 다르면 정상 miss다. extra도 같은데 cached index에 entry가 없다면 store/eviction 경계를 본다. entry와 physical generation은 valid인데 lookup가 miss면 group namespace나 traversal bug다.

metric만 hit 0이고 trace는 2,032 tokens를 touch해 runner에 commit했다면 metric 경계다. output 정답과 computed tokens 감소가 execution reuse를 뒷받침한다. 이때 cache logic를 바꾸면 정상 경로를 망친다. exporter/reset consumer를 고친다.

miss 복구는 R만 hit하게 하는 것이 아니다. adapter가 다른 S는 계속 분리된 entry를 써야 하고 final partial/uneligible tokens는 recompute해야 한다. isolation와 eligibility를 유지한 채 expected prefix만 reuse하는 것이 종료 조건이다.

### 39.4.2 hit 높음, TTFT 불변

lookup는 성공했지만 reserve rollback, external load critical path, graph/input preparation가 prefill savings를 상쇄할 수 있다. local hit, external requested/loaded, recomputed tokens와 first GPU start 시각을 맞춘다. token-weighted hit가 높아도 아주 짧은 suffix가 느린 transfer를 기다릴 수 있다.

first divergence는 성능 기대와 실제 critical path가 갈린 경계다. “cache가 느리다”가 아니라 lookup 뒤 어느 dependency를 기다렸는지 증명한다.

R은 local 1,024 hit와 external 1,008 request를 얻었다고 하자. external load가 20ms, recompute 1,024 tokens가 12ms라면 hit tokens가 많아도 remote dependency가 critical path를 늘릴 수 있다. hit rate가 compute savings만 세고 restore latency를 세지 않으면 TTFT 불변은 모순이 아니다.

reserve가 destination blocks를 기다리거나 graph/input preparation가 cache-off baseline에 없던 overhead를 만들 수 있다. GPU start가 cache-on에서 더 늦고 model compute duration은 짧다면 host/load critical path다. start와 duration이 모두 baseline과 같다면 cached frontier가 runner input에 반영되지 않았을 가능성이 있다.

external requested 1,008 중 768만 valid load되고 240을 recompute했다면 hit numerator를 1,008로 둘지 768로 둘지 metric contract에 따라 다르다. product SLO에서 실제 skipped compute를 보려면 derived savings를 별도로 만든다.

복구는 hit rate가 아니라 TTFT decomposition로 판정한다. load overlap이나 tier 선택으로 critical path가 줄었는지, computed tokens와 GPU work가 줄었는지 본다. 작은 workload에서 transfer overhead가 더 크면 cache를 쓰지 않는 policy가 옳을 수도 있다.

### 39.4.3 available가 단조 감소한다

owner 원장에서 active, locked, pending, cached eligible, uninitialized를 step별로 본다. cached eligible 증가면 정상 residency, lock/pin 증가면 leak 후보, reserved 증가면 unfinished allocation transaction다. release completion 뒤에도 owner가 남은 첫 request를 찾는다.

20회 반복에서 active는 0으로 돌아오지만 locked가 `0,1,2,...,20`으로 늘고 available가 같은 폭으로 줄며 evictable는 늘지 않는다고 하자. first leaked owner는 첫 R의 last radix node lock이다. OOM 시점 allocation failure는 결과이며 원인은 첫 finish/abort unlock 누락이다.

다른 run에서는 pending transfer가 concurrency depth 4까지 늘었다가 completion 뒤 0으로 내려가고 available가 sawtooth로 회복된다. 순간 감소는 leak가 아니다. oldest pending age가 transfer latency 안이고 generation별 completion가 있음을 확인한다.

또 다른 run에서는 cached eligible이 매번 늘어 available는 줄지만 pressure allocation 시 candidates가 evict되어 T가 성공한다. 이것도 정상 residency다. content cache가 free headroom을 사용했을 뿐 physical owner partition는 닫힌다.

복구 후 natural finish뿐 아니라 cancel, allocation failure, load timeout를 반복한다. locked/pending/reserved가 release conditions 뒤 baseline으로 돌아오고 ID partition가 중복·orphan 없이 닫혀야 한다. restart로 count를 0으로 만드는 것은 검증이 아니다.

### 39.4.4 RSS/VRAM만 유지된다

live cache object와 tensor allocated bytes가 안정적인데 allocator reserved만 높은 것은 재사용 pool일 수 있다. 동일 workload 반복에서 새 allocation/OOM 없이 reserved plateau면 leak 가설이 약해진다. live objects와 owner IDs가 증가하면 강해진다.

PyTorch allocator reserved가 8GiB로 남고 allocated live tensors는 3GiB에서 안정적일 수 있다. 다음 iteration가 pool을 재사용해 driver allocation가 늘지 않으면 정상 caching behavior다. `nvidia-smi`만 보면 8GiB가 계속 보여 leak처럼 보인다.

manager free blocks는 baseline인데 Python future list가 old cache tensors를 보존해 allocated live가 3→4→5GiB로 늘 수도 있다. block owner ledger만 맞아도 object lifetime leak다. cache object strong references와 pending callbacks를 본다.

llama.cpp fixed context KV buffer는 cells가 empty여도 backend buffer bytes를 유지할 수 있다. context 재사용 설계에서는 정상이다. context count가 요청마다 늘거나 cell associations가 baseline으로 안 돌아오면 leak 가설이 강해진다.

복구 종료 조건은 VRAM 0이 아니다. steady plateau, live owners baseline, repeated workload reuse, OOM absence를 함께 본다. process allocator를 매번 비우면 graph는 낮아져도 allocation overhead가 악화될 수 있다.

### 39.4.5 첫 token부터 오답이다

cache-off reference와 first layer attention output부터 비교한다. key isolation, position 0/absolute offset, block/cell generation, copy completion 순서로 분기한다. sampling randomness를 고정하지 못했다면 logits distribution와 deterministic prefix checkpoint를 쓴다.

tenant A가 먼저 same tokens를 adapter A로 계산했고 tenant B는 adapter B로 요청했다고 하자. B lookup가 A physical generation 12를 hit하고 TTFT가 짧다. cache-off B reference와 layer 0 attention output이 즉시 다르다. extra identity에 adapter revision이 없거나 isolation predicate가 빠진 lookup 경계가 first divergence다.

key는 올바른데 physical generation가 stale일 수도 있다. index는 `(hash→ID7)`을 가리키지만 ID7이 eviction 뒤 generation 13으로 다른 content를 가진다. B table은 ID7을 받고 first attention부터 다르다. key를 더 넓히는 수정은 도움이 없다. index removal과 physical reuse ordering를 고친다.

copy/load race는 같은 key·generation metadata를 보일 수 있다. destination는 맞지만 compute stream이 copy event를 기다리지 않아 일부 layers가 old bytes를 읽는다. 반복할 때 divergence 위치가 timing에 따라 흔들릴 수 있다. producer/consumer event와 first divergent layer를 연결한다.

cache-off/on 문자열이 같아도 logits가 달랐지만 같은 argmax였을 수 있다. deterministic fixture에서 selected logits/checkpoint tolerance와 tenant cross-matrix A→A, A→B, B→A, B→B isolation를 확인한다.

### 39.4.6 긴 context boundary에서만 오답이다

SWA/window start, partial page valid length, table/cell wrap boundary 앞뒤 positions를 추적한다. B-1, B, window-1, window positions를 fixture로 삼는다. short prompt 정상이라는 사실로 key identity 문제를 완전히 배제하지 않지만 boundary metadata 가설을 우선한다.

R이 length 2,047에서는 정상이고 2,048 또는 2,049에서 틀린다고 하자. page size 16이면 2,047 offset 15, 2,048 next block offset 0이다. SWA window 2,048이면 동시에 oldest-position eviction 경계다. 두 가설이 겹치므로 table entry와 window start를 각각 기록한다.

table이 correct new ID를 가리켜도 valid length가 capacity로 전달되면 partial-page stale slots를 읽을 수 있다. window start가 off-by-one이면 correct blocks에서 wrong range를 읽는다. physical mapping와 mask/length를 함께 비교한다.

llama.cpp ring wrap에서는 total cells가 맞아도 position/seq association가 old shift 값을 가질 수 있다. paged assertion를 적용하지 않고 logical absolute position에서 native cell metadata를 본다.

복구는 boundary 앞뒤 네 점에서 divergence가 사라져야 한다. window와 page 크기가 서로소인 추가 fixture로 두 경계를 분리할 수 있다.

### 39.4.7 cancel 뒤에만 leak가 난다

abort timeline에 last submitted step, output/transfer completion, request ref release, lock decrement, delayed block return을 넣는다. 자연 finish와 비교해 빠진 inverse mutation를 찾는다. late transfer가 진행 중인 정상 pin인지 completion 뒤 누락인지 구분한다.

R cancel이 step 71 write와 remote load X 사이에 들어왔다고 하자. collector는 닫히지만 blocks는 delayed writer와 transfer pin을 가진다. step 71 completion 뒤 일부가 돌아오고 X completion 뒤 나머지가 돌아오면 정상이다. cancel return 시 available가 즉시 회복되지 않았다는 이유로 leak라 하지 않는다.

X completion 로그는 있는데 pin map entry가 반복마다 +1이면 callback cleanup가 first divergence다. completion가 없고 oldest age가 timeout를 넘으면 transfer hang 또는 lost response다. pin decrement를 강제로 호출하기 전에 DMA safety와 lease epoch를 본다.

natural finish에는 leak가 없고 abort만 있으면 두 path diff가 강한 source map이다. request ref free는 공통인데 radix unlock, connector cancel, reserved rollback 중 빠진 mutation를 찾는다.

복구 뒤 cancel timing를 lookup 전, reserve 뒤, load 중, compute submit 뒤, final output 직전으로 바꿔도 partition가 닫혀야 한다. 한 timing에서만 OOM가 미재현된 것으로 닫지 않는다.

## 39.5 진단 행렬은 사건을 이해한 뒤 쓴다

행렬을 읽기 전에 “첫 분기”의 의미를 고정한다. 이것은 가장 가능성이 높은 원인을 찍는 칸이 아니라, 다음 두 가설이 서로 다른 값을 예측하는 가장 앞 경계다. hit=0에서 key mismatch와 eviction은 lookup result만으로 같아 보이지만 current normalized key와 residency/eviction history에서 갈린다. source를 읽을 위치가 여기서 정해진다.

“강한 증거”도 단일 metric 이름이 아니다. normalized identity에는 실제 token IDs와 effective model/adapter/template가 함께 있어야 하고, pool 합계에는 같은 generation ID partition가 필요하다. first-divergent layer에는 cache-off reference와 deterministic 조건이 필요하다. 필요한 조건이 빠지면 표의 evidence를 충족하지 못한다.

경쟁 가설은 하나를 맞히면 끝나는 양자택일도 아니다. key mismatch와 eviction가 동시에 있을 수 있고 missing decrement 뒤 metric scrape 지연이 섞일 수 있다. 먼저 관측된 divergence를 고친 뒤 세 원장을 다시 맞춰 다음 divergence가 남는지 본다. 한 incident에 여러 causal stages가 있을 수 있다.

hit=0 row에서 memory 정상은 owner leak 가능성을 낮추지만 key correctness를 증명하지 않는다. available 감소 row에서 답 정상은 현재 request가 correct KV를 읽었다는 뜻이지 다음 victim allocation가 안전하다는 뜻은 아니다. first token 오답 row에서 hit 높음은 reuse가 실행됐음을 시사하지만 어느 tier/path hit인지 확인해야 한다.

RSS/VRAM row는 allocator라는 별 seventh execution boundary를 추가하려는 것이 아니다. release 뒤 live logical owner와 framework storage owner가 갈리는 관측 층이다. cache manager free가 끝났지만 tensor reference나 reserved pool이 남는지 본다. 여섯 경계 model 안에서는 release와 metric 사이에 놓인다.

긴 context row는 position boundary를 first 분기로 삼는다. page boundary와 SWA window, cell wrap이 같은 length에 겹치면 추가 fixture로 분리한다. short/long 두 점만으로는 어느 transition인지 알 수 없다. boundary 양옆 최소 네 점을 사용한다.

cancel row는 natural finish를 control run으로 둔다. 두 paths의 shared cleanup와 abort-only cleanup를 diff하면 누락 inverse mutation가 좁혀진다. 다만 cancel이 in-flight transfer를 만들고 natural finish는 그렇지 않다면 final partition가 즉시 같을 필요는 없다. completion 뒤 비교한다.

행렬은 우선순위를 강제하지 않는다. wrong answer가 함께 있으면 correctness 격리가 먼저고, capacity가 임계면 new admission를 막을 수 있다. 표는 안정화 뒤 root first divergence를 찾는 지도다. incident response authority와 rollback policy는 배포 환경 계약을 따른다.

| 증상 | 첫 분기 | 경쟁 가설 | 강한 증거 |
|---|---|---|---|
| hit=0, memory 정상, 답 정상 | lookup 전 | key/template/option mismatch vs eviction | normalized identity와 lookup predicate |
| hit 높음, TTFT 불변 | lookup 후 | 분모 오류 vs restore/copy critical path | computed/local/external tokens와 GPU 시작 |
| available 단조 감소 | release | live owner vs metric 지연 | refs/pins/pending과 pool partition |
| RSS/VRAM만 유지 | allocator | reserved pool vs live KV | live storage와 allocator reserved 분리 |
| 첫 token부터 오답 | compute read | wrong key/position/generation | first divergent checkpoint와 physical mapping |
| 긴 context에서만 오답 | boundary | SWA mask vs partial length | boundary 양옆 position trace |
| cancel 뒤에만 leak | cleanup | late transfer vs missing decrement | abort timeline과 completion owner |

행렬은 source를 건너뛰는 답지가 아니다. 첫 분기에서 얻은 evidence가 경쟁 가설을 나누지 못하면 다음 경계로 이동한다. 증상 하나에 모든 metric을 무작정 수집하지 않는다.

### 39.5.1 miss 사건의 실제 조사 기록

운영자는 먼저 “두 번째 요청인데 hit=0”이라는 ticket를 받는다. 동일하다는 주장을 prompt text 비교로 받아들이지 않고 ingress 이후 token IDs, special tokens, chat-template digest, model/adapter effective identity를 두 requests에서 대조한다. 첫 token부터 다르면 정상 key miss 후보로 종료 방향이 보인다. 모두 같으면 lookup predicate로 간다.

R trace에서 prefix lookup enabled가 false라면 index dump를 하지 않는다. 어떤 request flag나 model mode가 disable했는지 source predicate와 config normalization를 찾는다. 의도된 skip이면 alert denominator를 고치고, 의도하지 않은 skip이면 option→request field mutation를 고친다. predicate true면 query/hit record로 내려간다.

query 2,032/hit 0이고 직전 cached entry store가 있었다면 eviction events와 current index를 본다. pressure victim으로 정상 제거됐으면 capacity/policy 사건이지 key bug가 아니다. event 없이 index가 사라졌다면 store commit/rollback 또는 metadata corruption다. index는 있는데 lookup miss면 group/hash traversal를 본다.

query trace hit 2,032인데 exported hit 0이면 logger/reset 경계다. runner computed frontier가 2,032이고 actual prefill 16이라는 증거가 execution hit를 확정한다. cache algorithm는 건드리지 않는다. scrape order와 counter consumer를 수정하고 historical metric discontinuity를 표시한다.

마지막으로 TTFT를 본다. hit와 computed savings가 맞아도 external restore, reserve wait, graph preparation가 critical path면 hit incident와 performance incident를 분리한다. ticket 제목을 “miss”에서 “warm path TTFT”로 바꾸는 것도 조사 성과다. 한 root cause에 맞지 않는 증상을 억지로 합치지 않는다.

### 39.5.2 leak 사건의 실제 조사 기록

available 감소 ticket에서는 먼저 같은 scheduler frontier의 pool partition를 만든다. active requests, cached eligible, uninitialized, locked/pinned, pending transfers, reserved transactions, special resources의 ID union을 센다. active=0이라는 한 숫자로 leak를 확정하지 않는다.

첫 10회는 cached eligible이 증가하고 pressure allocation에서 eviction되어 available가 회복된다면 정상 warm-up residency다. 다음 10회부터 locked가 request마다 하나씩 증가하고 evictable가 줄면 first anomalous request를 찾는다. 그 request의 lookup lock inc, finish/abort path, last-node handoff를 source와 맞춘다.

natural finish는 정상인데 cancel에서만 lock이 남으면 두 paths의 inverse mutation diff가 source 좌표를 좁힌다. cancel timing가 transfer 중이면 pin도 별도로 남을 수 있다. lock release를 추가하면서 pending transfer block을 즉시 free하지 않는다. owner kinds마다 fix를 나눈다.

manager owner partition가 정상인데 VRAM만 늘면 live tensor allocated와 allocator reserved를 분리한다. future/callback object count가 늘면 framework-level reference leak고 reserved plateau만 있으면 caching allocator behavior다. process memory graph를 KV block leak와 같은 ticket에서 떼어 낼 수 있다.

수정 검증은 100회 같은 반복에서 count가 “대략 안 늘어남”이 아니다. request delta lock/pin/reservation가 release condition 뒤 baseline이고 ID set disjointness가 유지되며 pending oldest age가 bounded인지 본다. abort와 timeout를 포함한다. OOM threshold까지 도달하지 않았다는 것은 약한 증거다.

### 39.5.3 wrong-answer 사건의 실제 조사 기록

정답 사건은 먼저 cache를 임시 disable해 reference를 얻되 그것을 영구 fix로 삼지 않는다. 동일 input IDs와 effective execution identity, deterministic sampling 조건을 고정한다. output first token이 다르면 prefill final logits를 비교하고, 같다면 first divergent output position 직전 state로 이동한다.

prefill logits가 다르면 layer checkpoints를 binary-search하듯 좁힌다. layer 0 attention부터 다르면 lookup key, position, physical table/cell mapping와 load completion을 우선한다. layer group 경계에서 시작하면 hybrid/sliding/recurrent group metadata를 본다. 모든 layer는 같고 logits processor 뒤 다르면 cache 외 경계를 분리한다.

tenant B가 A blocks를 hit했다면 key extra identity와 admission isolation를 본다. identity가 같은데 physical owner generation가 다르면 stale index cleanup를 본다. key와 generation가 맞는데 timing-dependent면 stream/event completion를 본다. 세 가설은 모두 high hit/short TTFT를 만들므로 performance metric로 나뉘지 않는다.

boundary-only 오답은 positions 2,047, 2,048, 2,049와 page/window가 분리되는 추가 points를 trace한다. table entry, valid length, window start, absolute position, cell sequence metadata를 한 row에 둔다. wrong physical ID와 correct ID/wrong mask를 구분한다.

수정 뒤 cache-on/off cross-matrix와 first checkpoint가 맞고 stale reject, copy wait, owner metrics도 정상이어야 한다. 모든 hit를 drop해 reference와 같아졌다면 correctness guard는 생겼지만 intended reuse는 복구되지 않았다. root index/key/event ordering를 계속 고친다.

### 39.5.4 세 사건이 동시에 보일 때의 순서

실제 장애에서는 hit가 떨어지고 available도 줄며 일부 답이 다를 수 있다. correctness를 먼저 보호해 suspect cache entries나 worker를 격리한다. 불확실한 physical generation를 재사용하지 않는다. 그 다음 owner leak를 멈춰 capacity와 추가 corruption를 막고, 마지막에 miss/TTFT 최적화를 회복한다.

이 우선순위는 miss가 중요하지 않다는 뜻이 아니다. wrong answer는 이미 전달한 결과를 되돌릴 수 없고 live eviction/use-after-free는 다른 requests를 오염시킬 수 있어 blast radius가 크다. performance cache hit를 보존하려 correctness quarantine를 미루지 않는다.

격리 후에도 evidence를 지우지 않는다. reset 전에 request→key→physical generation→event와 owner partition snapshot을 보존한다. raw prompt/KV content 대신 pseudonymous identity와 coordinates를 사용한다. reset가 root state를 지운 뒤에는 first divergence를 재구성하기 어렵다.

## 39.6 같은 TTFT 악화를 miss·leak·wrong reuse로 가른다

사용자는 세 사건 모두 “캐시를 켰는데 느리고 이상하다”고 말할 수 있다. miss는 재계산 때문에 TTFT가 늘어난다. leak는 free capacity 감소와 eviction/recompute를 거쳐 나중에 TTFT가 늘어난다. wrong reuse는 빠른 hit 뒤 validation/retry 또는 잘못된 답 때문에 체감 실패를 만든다. latency 하나로 원인을 정하지 않는다.

동일한 2,048-token prompt에서 eligible aligned prefix가 2,032 tokens라고 하자. 정상 hit 실행은 lookup queried2,032, local valid hit2,032, computed16이다. miss 실행은 queried2,032, hit0, computed2,048이다. leak 실행은 첫 요청에서 hit2,032/computed16이지만 반복 cancel 뒤 free blocks가 줄어 eviction과 computed tokens가 점차 늘어난다.

wrong reuse 실행은 hit2,032/computed16으로 성능 숫자가 정상처럼 보인다. 그러나 layer0 cached K/V checksum 또는 first forward hidden-state checksum이 no-cache reference와 다르다. 같은 hit count라도 effective identity 또는 physical generation이 틀렸다. correctness evidence가 없으면 가장 위험한 사건을 건강한 hit로 오판한다.

첫 분기표는 `queried/hit/computed`, `used/free/owners after terminal`, `first checkpoint parity` 세 열이다. miss는 첫 열에서, leak는 둘째 열에서, wrong reuse는 셋째 열에서 처음 갈린다. 뒤의 TTFT/OOM/final text는 결과다.

구현 metric 이름을 공통 의미로 번역한다. prefix-cache query/hit tokens, KV cache usage/available blocks, request/radix/pin owners와 eviction/recompute events가 어느 snapshot/event인지 확인한다. metric 이름이 `hit_rate`라고 분모가 eligible queried tokens라는 보장은 없다. exporter source까지 읽는다.

Prometheus는 cohort와 시간 변화를 보여 주는 첫 사다리다. trace는 request generation, identity digest, lookup/reserve/load/read/release 사건을 연결하는 둘째 사다리다. tensor evidence는 selected cached rows/tags/checksum과 layer output가 reference와 갈리는지 보는 셋째 사다리다. 세 층을 일반 관측 개론으로 확장하지 않고 이 cache 경계에만 적용한다.

## 39.7 miss 사건을 재현하고 rollback까지 검증한다

관측은 cache enabled cohort의 TTFT가 no-cache와 같고 hit token counter가 0인 것이다. memory used/free와 request terminal은 정상이고 output parity도 맞다. leak/wrong-answer보다 안전하지만 비용 절감 계약은 실패했다.

Prometheus 단계에서 query counter가 증가하는지 먼저 본다. query0이면 cache lookup branch 자체가 실행되지 않았거나 eligible prefix가 0이다. query2,032/hit0이면 lookup은 실행됐지만 key/policy/content가 맞지 않는다. query/hit2,032인데 computed2,048이면 metric 또는 scheduler/runner 소비가 갈린다.

분모를 검산한다. prompt2,048, block16, 마지막 block은 output 시작 경계 때문에 재사용 제외라고 하면 eligible2,032다. exporter가 total prompt2,048을 denominator로 쓰면 정상 hit2,032도 99.2%로 보인다. query하지 않은 16을 miss로 세지 않는다. cache disabled 요청도 denominator cohort에서 제외한다.

trace 단계는 normalized identity digest, queried block digests, lookup result handles/generations, scheduler computed-prefix와 actual runner input interval을 잇는다. model revision, adapter, chat template/token IDs, multimodal features, cache dtype/layout와 tenant namespace 중 어느 field가 달랐는지 본다.

사건 예로 writer key는 `(model M7, adapter A3, tokens digest X, block16)`인데 reader key는 adapter default A0을 사용했다. lookup query는 있었지만 key가 달라 hit0이다. cache content가 evicted된 것이 아니다. writer entry가 resident이고 same-A3 control request는 hit한다. first divergence는 request identity→lookup key normalization이다.

tensor 단계는 miss 사건에서 주로 반증용이다. no-cache recompute K/V와 writer cached K/V가 같은 input에서 일치한다면 content corruption 가설이 약해진다. output parity가 맞고 computed2,048이므로 wrong reuse도 약하다. cache를 끄고 같은 latency가 나는 것은 miss의 결과와 일치하지만 원인 key mismatch를 증명하지는 않는다.

수정은 reader key에 adapter generation을 포함하고 writer/reader가 canonical key builder를 공유하게 한다. key namespace를 좁혀 hit를 올리는 대신 tenant/model isolation을 약화하지 않는다. unknown/missing adapter field를 default와 동일하다고 silent 처리할지 reject할지 계약을 정한다.

회귀 fixture는 same/different model, adapter, template, token IDs, multimodal identity, block alignment와 layout generation을 pairwise로 만든다. same identity는 hit2,032/computed16, one-field different는 intentional miss/computed2,048이어야 한다. output parity와 owner cleanup도 함께 본다.

rollback은 새 key generation admission을 fence하고 old/new namespaces를 혼용하지 않는다. 새 sharing이 suspect면 cache reuse를 disable해 안전한 miss로 전환할 수 있다. old entries는 generation별 quarantine/evict하고 request refs를 닫는다. hit rate 복원만 보고 승인하지 않는다.

**miss 사건의 복구 결말.**

key/template mismatch를 고쳤다면 동일 identity R의 lookup query 2,032와 hit 2,032가 expected이고 adapter가 다른 S는 별 key를 얻어야 한다. local hit가 runner computed frontier에 반영되어 prefill query와 measured model work가 줄어야 한다. hit counter만 바뀌면 metric 수정일 뿐 execution miss는 남았다.

eviction policy를 고쳤다면 pressure 없는 repeated R에서 residency가 유지되고 pressure에서는 eligible victim order가 의도대로 작동해야 한다. live/locked blocks를 보존하면서 cache hit opportunity가 개선됐는지 본다. 모든 blocks를 pin해 hit를 100%로 만들면 allocation stall이라는 새 leak성 문제를 만든다.

metric denominator 수정은 historical dashboard와 alert threshold도 재해석해야 한다. token-weighted와 request-weighted rate를 label/name으로 구분하고 local/external/computed 관계를 검증한다. reset/scrape concurrency에서 interval 합이 유지되는지 본다.

TTFT는 같은 hardware, concurrency, prompt/suffix distribution에서 warm/cold를 나눠 본다. hit path의 host/load overhead와 actual GPU prefill 감소를 함께 측정한다. 이 집필에서는 runtime을 실행하지 않지만 독자가 검증할 denominator와 boundary를 명시한다.

## 39.8 leak 사건에서 잔여 owner를 없애고 rollback을 확인한다

관측은 cache hit와 output가 정상인데 cancel20회 뒤 available blocks가 매번 한 block씩 줄어드는 것이다. steady traffic을 멈춰도 회복하지 않고 process restart 뒤에만 돌아온다. arena reserved가 유지되는 정상 caching allocator와 request-owned used units 잔류를 구분한다.

Prometheus 단계에서 total, free, used, cached eligible, request refs, radix locks, transfer pins, writer fences와 pending reservations을 같은 snapshot generation에 맞춘다. `free+owned partitions=total` 보존식의 gap을 찾는다. RSS/VRAM high-water만 보면 request leak인지 arena reserve인지 알 수 없다.

예로 total10,000, free8,000, cached eligible1,500, active request400, transfer pins100이면 합 10,000이다. all requests drain 뒤 expected는 free8,400+cached1,500+pins100=10,000처럼 정책상 cache/pins가 남을 수 있다. transfer completion 뒤 pins도 0이어야 한다면 free8,500+cached1,500이다. owner condition을 먼저 정한다.

사건에서는 cancel20, transfer completions20인데 pin owners20과 blocks20이 남았다. request refs0, writer fences0이다. trace는 cancel terminal, transfer completion callback와 pin-release mutation을 연결한다. callback이 old request map을 찾지 못해 early return하면서 release를 건너뛰었다.

first divergence는 free gauge 감소가 아니라 completion callback의 owner lookup failure 뒤 `unpin` 미호출이다. allocator 자체는 요청받은 free를 올바르게 처리한다. restart가 회복시키는 것은 process lifetime에 묶인 orphan pins를 폐기하기 때문이다. root fix 증거가 아니다.

tensor evidence는 leaked block content가 wrong owner로 재사용되지 않았음을 확인한다. pin 때문에 재사용 자체가 막혔다면 capacity leak이고 correctness는 당장 안전하다. 강제 free로 문제를 숨기면 늦은 transfer writer가 new owner block을 오염시킬 수 있다. last writer event와 generation을 확인하지 않고 free하지 않는다.

수정은 pin handle이 request-map lookup과 독립적으로 self-contained release owner/generation을 갖게 하고 completion/cancel이 idempotent finalizer로 수렴하게 한다. cancel이 client terminal을 먼저 보내도 transfer terminal이 background에서 exactly once pin을 반환한다.

회귀 fixture는 normal completion, cancel-before-transfer, cancel-during, completion-before-cancel, duplicate callback, process/worker restart와 block reuse를 교차한다. N cycles 뒤 used/free/owner union이 baseline으로 돌아오고 late writer가 reused generation을 mutate하지 않아야 한다.

rollback은 새 allocations를 제한하고 suspect generation blocks를 quarantine한다. safe last-consumer completion을 확인한 handles만 release한다. 필요하면 worker drain/restart를 하되 trace/owner snapshot을 먼저 보존한다. capacity 복귀, no stale write와 client terminal을 모두 닫는다.

**leak 사건의 복구 결말.**

수정 전 first R abort가 lock L을 남겼다면 수정 뒤 같은 boundary에서 L inc와 dec가 owner ID로 한 쌍이어야 한다. absolute lock sum은 root/special baseline을 포함할 수 있으므로 request 전후 delta를 본다. evictable size와 tree/list sanity도 일치해야 한다.

transfer pin leak라면 completion 또는 cancel lease expiry 뒤 pin map entry와 destination/source block owner가 함께 정리되어야 한다. pin만 0으로 만들고 pending future가 tensor를 붙잡으면 live storage leak가 남는다. 반대로 future만 제거하고 block writer가 진행 중이면 correctness 위험이다.

steady-state run은 충분한 반복 후 active, locked/pending, reserved, cached eligible, uninitialized partition가 주기 범위로 돌아오는지 본다. cached set 내용은 workload에 따라 바뀌어도 physical union와 owner conditions가 닫혀야 한다. process restart 없이 검증한다.

double-free 방어도 확인한다. missing decrement를 고치며 natural finish와 abort가 모두 release하면 ref를 두 번 놓을 수 있다. free queue ID uniqueness, ref nonnegative, shared request correctness를 함께 본다. available가 더 많이 회복됐다는 것은 성공이 아니다.

## 39.9 wrong-answer 사건의 최초 오답과 rollback을 확인한다

관측은 cache-on에서 TTFT가 빠르고 hit2,032인데 첫 generated token부터 no-cache reference와 다르다. cache-off는 정상이고 owner/free 보존식도 맞다. miss나 leak가 아니라 cached state identity/address/order가 후보다.

Prometheus 단계는 cache-on/off error rate, hit cohort, model/adapter/layout generation과 boundary counters를 좁힌다. aggregate final quality는 탐지 신호일 뿐 first divergence가 아니다. tag mismatch, stale generation reject 또는 cross-identity hit counter가 있으면 source branch로 내려간다.

trace는 lookup key와 returned physical handles, writer owner identity, load/copy events, compute read event, absolute/window positions와 layer group을 잇는다. key가 맞아도 handle generation이 old일 수 있고, generation이 맞아도 copy completion 전 읽을 수 있으며, address가 맞아도 chronological order가 틀릴 수 있다.

tensor evidence는 가장 이른 layer/checkpoint를 비교한다. token embedding은 같고 layer0 attention input은 같지만 cached K row checksum부터 다르면 cache content/identity다. K/V는 맞고 layer attention output부터 다르면 mask/stride/order 또는 kernel이다. all layer outputs가 맞고 token만 다르면 sampling/output을 본다.

사건 예는 tenant B lookup key에 adapter generation이 있었지만 returned handle cache entry의 physical generation이 A의 evicted/reused block이었다. block table update는 new handle index를 썼으나 async external load completion이 old destination generation을 덮었다. final key metadata는 B라 metric상 hit가 정상이다.

first divergence는 external load completion의 generation check 누락이다. copy event가 signaled된 뒤 B compute가 읽어 ordering은 맞지만 내용 producer가 old A transfer다. tenant mixing, scheduler output demux와 model numerical noise를 tensor/layer checkpoint로 분리한다.

수정은 destination generation을 transfer descriptor와 completion callback에 넣고 mismatch 시 publish/drop·cleanup한다. compute는 matching generation의 copy completion 뒤에만 block table entry를 visible하게 본다. old load failure는 B request를 miss/recompute 또는 explicit error로 전환한다.

회귀 fixture는 same physical block reuse, delayed load, cancel/reallocate, duplicate completion, adapter/tenant/model changes, sliding boundary와 layer groups를 교차한다. cache-on/off per-layer parity, tags/generations와 exactly-once cleanup을 검증한다.

rollback은 suspect cache/layout generation을 즉시 격리하고 reuse를 disable한다. 이미 전달된 wrong outputs는 internal eviction으로 회수되지 않으므로 affected request scope와 client terminal을 기록한다. old transfers를 drain/drop하고 owners를 reconcile한 뒤 clean generation으로 readiness를 연다.

**wrong-answer 사건의 복구 결말.**

key namespace 수정은 tenant/model/adapter/template cross-matrix를 통과해야 한다. same identity는 reuse하고 different effective execution identity는 분리한다. cache-off/on first-layer checkpoint와 logits가 허용 tolerance 안에서 맞아야 한다. 한 final 문자열 equality는 약하다.

generation 수정은 eviction/reuse를 강하게 일으키는 fixture에서 stale index가 current physical owner를 반환하지 않는지 본다. generation mismatch guard가 hits를 drop하면 correctness는 지킬 수 있지만 stale reject counter가 계속 늘지 않아야 root ordering가 고쳐졌다.

event ordering 수정은 copy/load delay가 달라도 consumer가 completion 뒤 읽고 whole-device synchronize 없이 overlap을 유지해야 한다. 전체 synchronize로만 맞으면 진단 patch이지 최종 성능 계약이 아니다. 정확한 producer-consumer storage event를 연결한다.

boundary 수정은 page, SWA window, cell wrap 각각 앞뒤 positions에서 검증한다. page size와 window가 겹치는 fixture뿐 아니라 분리되는 lengths를 사용한다. one off-by-one를 고치며 반대 boundary를 깨지 않았는지 본다.

## 39.10 실제 metric과 source state를 같은 사건에 붙인다

vLLM incident card는 scheduler가 보고하는 computed/cached token progress, KV cache manager의 blocks/refs와 runner input/output generation을 연결한다. prefix cache hit token metric이 lookup 결과인지 실제 scheduled savings인지 exporter source에서 확인한다. request finish/abort와 block free boundary도 붙인다.

SGLang card는 radix/prefix lookup result, lock/reference ownership, token-to-KV pool free/used와 schedule batch의 cached/extend lengths를 잇는다. full attention, sliding pool과 recurrent/Mamba state를 같은 denominator로 합치지 않는다. retract/cancel과 async transfer completion cleanup을 본다.

Transformers continuous card는 cache lookup/update, block manager refs/free blocks, request state progress와 error cleanup을 연결한다. process reserved memory는 framework allocator와 paged cache live blocks를 분리한다. hit metric이 없다면 trace에서 queried/reused intervals를 재구성하고 없는 metric을 있다고 가정하지 않는다.

llama.cpp card는 server slot prompt-match/cache reuse, KV cells ownership/generation, slot cancel/reset과 optional RAM prompt cache를 구분한다. matched tokens가 실제 compute skip으로 이어지는지 batch construction을 본다. stable slot ID가 physical cell generation을 보장한다고 가정하지 않는다.

세 incident 모두 source card 열은 predicate, state before, mutation, acknowledgment, next consumer, rollback과 metric exporter다. miss는 key builder/lookup와 computed frontier, leak는 owner finalizer/free list, wrong reuse는 handle generation/load-read ordering이 canonical owner다.

Prometheus query 예시는 이름을 고정하지 않고 의미를 고정한다. 같은 config generation에서 `rate(queried_tokens)`, `rate(valid_hit_tokens)`, `rate(actual_computed_tokens)`을 비교한다. owner는 current gauges의 합과 total을 같은 scrape epoch에 맞춘다. correctness는 error/tag counters를 cohort로 좁히되 tensor parity는 trace artifact다.

counter interval과 gauge snapshot을 직접 뺄 때 주의한다. 5분 hit counter와 현재 free blocks는 같은 보존식 항이 아니다. restart counter reset, rolling replicas와 scrape gaps를 고려한다. 사건 timeline의 exact worker/rank를 trace로 고정한 뒤 aggregate trend를 사용한다.

high-cardinality identity는 metric label로 넣지 않는다. model/config/cache generation, reason, pool/component와 outcome은 bounded labels다. request/key/block IDs와 digest, absolute position, layer checkpoint는 sampled trace에 둔다. prompt/KV raw content 대신 safe digest와 controlled synthetic sentinel을 사용한다.

source exporter가 wrong denominator를 쓸 수 있다. hit counter가 lookup candidate를 세고 validation failure도 hit로 남기면 performance savings를 과대평가한다. `candidate`, `validated`, `loaded`, `consumed`를 별 events로 두거나 metric 의미를 정확히 문서화한다. actual computed frontier와 맞춘다.

owner exporter가 ref edges를 physical blocks로 합산하면 sharing에서 used가 부풀 수 있다. unique physical handle count와 reference edge count를 따로 낸다. radix lock, transfer pin과 writer fence는 한 block에 동시에 붙을 수 있으므로 mutually exclusive partition인지 overlapping edge counts인지 metric 계약을 밝힌다.

wrong-answer exporter도 silent corruption 전체를 포착하지 못한다. tag/generation assertions는 알려진 invariant violation을 잡지만 semantic key 누락은 값이 internally consistent할 수 있다. cache-on/off shadow parity 또는 controlled checksum sampling을 사용하되 overhead와 privacy를 관리한다.

source walk의 종료는 metric 이름을 찾는 것이 아니다. 사건 row 하나에 lookup decision, returned handle, owner mutation, compute read, release와 exporter increment를 동일 generation으로 연결해야 한다. 어느 arrow가 비면 instrument gap을 인정하고 추가한다.

**왜 세 증상을 먼저 분리하는가.** miss는 정답을 다시 계산해 latency만 늘 수 있지만 leak는 미래 admission capacity를 잠식하고 wrong answer는 이미 잘못된 state를 소비했다는 뜻이다. 비용과 복구 범위가 다르므로 hit ratio 하나로 묶으면 위험하다. miss에서 content checksum이 맞으면 index·policy를, leak에서 terminal owner가 없으면 refcount·cleanup을, wrong answer에서 mapping은 맞고 generation이 다르면 stale reuse를 조사한다.

**반증 실험.** cache를 완전히 끈 no-cache reference와 동일 token logits를 비교하면 model 자체 오답 가설을 분리할 수 있다. 같은 key를 유지한 채 physical page generation만 바꾸는 실험은 hash collision과 stale page를 가른다. cancel 폭주 뒤 live request를 0으로 만들고 allocated/refcount가 0으로 수렴하지 않으면 단순 workload 증가 가설을 기각한다.

왜 rollback 범위도 다른가. miss는 cache bypass로 정답을 지킬 수 있지만 wrong answer는 해당 generation의 응답과 후손 prefix를 폐기해야 한다. 왜냐하면 이미 소비된 stale KV는 hit counter를 고쳐도 되돌아오지 않기 때문이다.

## 39.11 반증·회귀·rollback terminal을 한 표로 닫는다

같은 증상에서 miss를 반증하려면 query/hit와 actual computed frontier를 본다. valid hit2,032가 runner compute skip16으로 이어졌다면 pure miss가 아니다. latency가 느려도 graph/kernel/network를 본다. hit counter만 높고 computed2,048이면 metric/consumer gap이라 miss 계열이 남는다.

leak를 반증하려면 requests를 drain하고 pending transfers/fences가 완료된 뒤 owner union과 allocator baseline을 본다. request-owned units가 0으로 돌아오고 repeated cycles threshold가 안정적이면 leak가 약하다. process reserved가 유지되는 것만으로 leak라 하지 않는다.

wrong reuse를 반증하려면 cache-on/off deterministic fixture의 earliest cache-consuming layer parity, handle/tag/generation과 copy-read event를 본다. final text 하나가 같아도 hidden checkpoint mismatch가 있으면 반증되지 않는다. sampling randomness은 fixed logits/token or layer parity로 제거한다.

세 원인이 동시에 나타날 수 있다. key mismatch로 miss가 늘어 capacity pressure가 커지고 cancel path pin leak가 드러나며, 강제 free 대응이 late write wrong reuse를 만들 수 있다. 안전 우선순위는 wrong-answer 격리, unsafe allocation/reuse 차단, owner reconciliation, 그 뒤 miss 최적화다.

regression matrix의 공통 축은 cache on/off, same/different identity, hit/miss, normal/cancel, async delay, generation reuse, prefix alignment와 full/sliding/recurrent component다. 각 사건에 필요한 상호작용을 full coverage하고 나머지는 pairwise로 줄인다.

miss expected table은 same identity hit/savings, one-field difference intentional miss, evicted miss와 external-load fallback을 구분한다. hit count, actual computed interval, output parity, owner cleanup을 assert한다. key normalization upgrade에서 old/new namespaces mixed rollout도 시험한다.

leak expected table은 N normal/cancel/retry cycles 뒤 owner counts와 free baseline, pending age0, late callbacks0을 요구한다. shared cache policy상 cached eligible가 남으면 reference owner와 eviction 가능성을 확인한다. 강제 GC/restart 없이 수렴해야 root fix다.

wrong-answer expected table은 cache-on/off per-layer parity, delayed copy/reuse, cross adapter/tenant/model, sliding boundary, quantized scale와 group mapping을 포함한다. expected intentional miss가 wrong hit보다 안전하다. generation mismatch는 hard reject/recompute 또는 explicit error로 닫는다.

failure injection은 lookup validation failure, reserve failure, external load timeout, cancel-before/after-copy, duplicate completion, release callback exception과 worker restart를 둔다. 서비스 terminal, resource terminal과 telemetry terminal을 따로 확인한다. error response만 보낸 것은 owner cleanup 증거가 아니다.

rollback matrix도 사건별로 다르다. miss 변경은 old/new key namespace를 격리하고 sharing을 줄여 안전한 recompute로 돌아간다. leak 변경은 suspect owners를 quarantine하고 last consumer 뒤 release하며 필요하면 bounded drain/restart한다. wrong reuse는 즉시 cache generation을 격리하고 reuse를 disable한다.

rollback 중 inflight request semantics를 명시한다. cached prefix를 버리고 recompute할 수 있는지, already delivered outputs 뒤 retry가 idempotent인지, wrong results affected scope를 어떻게 알릴지 정한다. internal cache flush는 client-visible corruption을 되돌리지 않는다.

readiness는 process alive가 아니다. cache key/layout generation, allocator conservation, transfer/output loops, controlled same/different identity fixture와 cache-on/off parity를 포함한다. old generation pending owners/transfers가 0이고 baseline capacity가 회복된 뒤 admission을 연다.

승격 조건은 miss의 savings, leak의 finite cleanup, wrong reuse의 parity를 동시에 만족하는 것이다. hit rate가 올라도 isolation/correctness가 깨지면 실패다. memory가 줄어도 premature free/late write가 있으면 실패다. parity가 맞아도 모든 요청을 recompute해 performance 계약이 사라지면 miss 개선은 실패다.

incident 문장은 구체적으로 쓴다. miss는 “adapter generation 누락으로 queried2,032/hit0”; leak는 “cancel20/completion20 뒤 pin owners20 잔류”; wrong reuse는 “old transfer generation이 reused block을 publish해 layer0 checksum부터 divergence”처럼 첫 state mutation을 포함한다.

최종 artifact는 세 narrative, Prometheus cohort query, request-generation trace, tensor/layer checkpoint, pinned source card, regression matrix와 rollback record다. 관측 일반론을 반복하지 않고 cache lookup·owner·read 경계의 실제 증거만 남긴다.

독자는 이 artifact를 통해 동일 TTFT/OOM/품질 신고에서도 어느 원장이 처음 깨졌는지 선택한다. 그 결과 cache를 무조건 끄거나 process를 무조건 재시작하는 대신 안전한 임시 대응과 root fix 완료를 구분할 수 있다.

CUDA 편으로 넘기는 것은 cache 문제가 아니라는 막연한 결론이 아니다. key·owner·address·generation과 copy/read event까지 cache 원장에서 맞았는데 kernel output부터 갈린다는 명확한 handoff다. 다음 조사는 launcher stride, stream ordering과 kernel physical address에서 시작한다.

**세 사건을 30분 안에 분류하는 실전 순서.**

첫 5분에는 symptom cohort를 고정한다. worker/rank, model·adapter·cache generation, prompt length/alignment, cache on/off와 cancel 여부를 적는다. rolling deployment의 old/new 요청을 합치지 않는다. passing neighbor 하나를 같은 좌표에서 고른다.

다음 5분에는 세 숫자를 얻는다. eligible queried/hit/actual computed tokens, terminal 후 used/free/owner partitions, cache-on/off earliest checkpoint parity다. metric이 없으면 모른다고 표시하고 trace instrumentation을 추가한다. hit rate 하나로 빈칸을 채우지 않는다.

세 번째 5분에는 분기한다. query0은 branch/eligibility, query>0 hit0은 key/content/policy, hit>0 computed-all은 consumer/metric이다. owner gap은 release/pin/fence, owner 정상은 leak를 약화한다. first checkpoint mismatch는 identity/generation/address/order, parity는 wrong reuse를 약화한다.

네 번째 5분에는 source mutation을 찾는다. miss는 canonical key builder와 lookup return, leak는 terminal callback과 finalizer, wrong reuse는 transfer descriptor/block-table publish/read wait를 걷는다. exporter increment가 실제 mutation 전후 어디에 있는지 붙인다.

다섯 번째 5분에는 competing hypotheses를 지운다. resident entry와 same-identity control hit가 있으면 eviction miss가 약하다. completion20/pin20이면 general fragmentation보다 release 누락이 강하다. layer0 checksum divergence와 cache-off parity면 sampling/network가 약하다.

마지막 5분에는 containment와 완료 조건을 나눈다. miss는 안전한 recompute가 가능하다. leak는 allocation fence/quarantine이 필요하다. wrong reuse는 cache generation 즉시 격리가 우선이다. root fix는 regression/owner/parity와 rollback terminal이 통과해야 한다.

**Prometheus에서 흔히 만드는 오판.**

hit ratio 분모를 total prompt tokens로 쓰면 unaligned tail, never-queried tokens와 cache-disabled traffic이 miss로 들어간다. 분자는 candidate hit인지 validated/consumed hit인지도 확인한다. 실제 compute savings와 관계없는 ratio는 용량 계획에 쓰지 않는다.

free blocks가 낮다고 leak라 하지 않는다. cached eligible, active refs, pins/fences, provisional reservations와 allocator policy reserve가 있을 수 있다. 반대로 process VRAM이 일정하다고 leak가 없다고 하지 않는다. preallocated arena 안에서 free-list units만 새고 process bytes는 변하지 않을 수 있다.

counter 증가와 current gauge를 한 보존식에 더하지 않는다. `rate(cancel_total[5m])`와 현재 pins는 관계를 보는 지표이지 동일 단위의 합이 아니다. request-generation trace에서 cancel→completion→unpin event를 맞춘다.

label cardinality를 줄이려고 원인들을 `cache_error` 하나로 합치면 runbook이 사라진다. bounded reason은 key_mismatch, validation_fail, stale_generation, load_timeout, release_gap처럼 source mutation에 대응하게 한다. request/key/block는 exemplar/trace다.

metric이 0인 것도 두 뜻이다. event가 없거나 exporter가 branch에 연결되지 않았다. controlled fixture로 lookup/cancel/mismatch를 발생시키고 expected counter가 증가하는지 검증한다. instrumentation 자체를 회귀 시험한다.

**tensor evidence를 안전하고 싸게 수집한다.**

production prompt/KV 전체를 dump하지 않는다. controlled synthetic requests에서 layer·position별 checksum, shape/stride, block generation과 event IDs를 쓴다. 실제 incident에서는 safe digest와 sampled boundary rows만 수집하고 privacy policy를 따른다.

checksum은 dtype/NaN/ordering을 숨길 수 있다. sum 하나보다 multiple moments/hash와 tags를 쓰고, 충돌이 의심되면 controlled reproduction에서 exact tensor comparison을 한다. floating backend 허용 오차와 identity mismatch를 구분한다.

earliest checkpoint를 선택한다. tokenizer IDs, embeddings, layer0 Q/current input, cached K/V read, attention output, later layers와 logits 순이다. 전부 dump할 필요 없이 binary search처럼 checkpoint를 좁힌다. cache-off reference와 same seeds/config를 사용한다.

sharing 사건은 writer A와 reader B의 identities를 함께 기록한다. returned handle metadata만 B로 갱신됐을 수 있으므로 content producer generation과 last write event를 본다. physical storage reuse fixture가 필요하다.

sliding/recurrent/quantized components는 component-specific evidence를 쓴다. sliding absolute tags/chronological order, recurrent state version/owner, FP8 scale granularity를 본다. dense K/V checksum 하나로 모든 cache family를 검증하지 않는다.

**회귀가 통과했어도 남는 운영 질문.**

key fix가 cache namespace를 세분화해 hit를 낮출 수 있다. correctness isolation이 우선이며 새 건강한 denominator에서 hit/TTFT baseline을 다시 잡는다. old 높은 hit가 cross-identity sharing이었다면 성능 회귀가 아니라 오류 제거다.

leak fix가 pins를 더 빨리 release하면 late consumer safety를 검증해야 한다. memory baseline 회복만 보고 premature free를 승인하지 않는다. barrier-controlled late write/read와 block reuse generation fixture를 유지한다.

wrong-reuse fix가 mismatch를 모두 recompute로 fallback하면 정답은 맞아도 miss storm이 생길 수 있다. mismatch rate와 reason을 감시하고 key/layout rollout 문제를 고친다. safe fallback은 containment이지 영구 성능 설계가 아닐 수 있다.

restart가 필요한 rollback은 snapshot을 먼저 보존한다. owner graph, pending transfers, cache generation과 affected request scope가 사라지면 root cause를 다시 증명하기 어렵다. 단, wrong output 확산을 막는 격리보다 증거 수집을 우선하지 않는다.

readiness fixture는 same identity hit, intentional identity miss, cancel during async load와 stale generation reuse 네 개를 포함한다. hit/savings, owner cleanup, parity와 mismatch rejection을 한 번에 확인한다. service health endpoint만으로 cache 안전성을 판정하지 않는다.

**최종 독자 worksheet.**

행 1은 관측이다. TTFT/OOM/wrong text와 cohort를 쓴다. 행 2는 분모로 queried/hit/computed intervals를 쓴다. 행 3은 owner partitions와 terminal timeline, 행 4는 identity/handle/event와 earliest tensor checkpoint다.

행 5는 first divergence source span과 state before/after다. 행 6은 반증된 가설과 passing neighbor다. 행 7은 containment, 수정, regression fixture와 rollback terminal이다. 빈칸은 추측으로 채우지 않고 필요한 instrumentation을 적는다.

miss 행의 완료 예시는 key generation A3 복원, hit2,032/computed16, output parity와 owner cleanup이다. leak 행은 cancel20/completion20/unpin20, owner gap0과 reuse safety다. wrong-answer 행은 stale completion reject, per-layer parity와 affected generation 격리다.

세 행을 같은 표에 넣되 성공 기준을 합치지 않는다. hit 개선은 miss, finite cleanup은 leak, tensor/output parity는 wrong reuse의 기준이다. 모든 사건이 service/resource/telemetry terminal을 가져야 한다.

최종 승인 문장은 first divergence를 포함한다. “cache 문제가 해결됐다”가 아니라 어느 normalized key, owner finalizer 또는 generation fence가 어긋났고 어떤 fixture에서 닫혔는지 쓴다. 이 문장만으로 다른 엔지니어가 고정 revision에서 source와 trace를 다시 따라갈 수 있어야 한다.

이 장의 실용성은 metric 개수에 있지 않다. 같은 증상에서 세 원인 중 안전하게 먼저 다룰 것을 선택하고, 잘못된 강제 free나 넓은 sharing 같은 2차 사고를 피하며, 완료를 재현 가능한 증거로 선언하는 데 있다.

**배포 변경의 인과 카드.**

cache key field를 추가하면 parser option보다 identity builder, entry namespace, mixed-version lookup와 old-entry eviction이 바뀐다. rollout 중 writer old/reader new와 writer new/reader old 조합을 시험한다. protocol/version prefix가 없다면 silent cross-lookup보다 intentional miss가 안전하다.

refcount/finalizer 변경은 request terminal, async transfer/writer completion, radix/prefix ownership과 allocator free 순서를 바꾼다. idempotent release가 duplicate callback을 견디는지, early release가 last consumer를 앞서지 않는지 본다. lock 하나 추가했다는 설명보다 owner transition table을 남긴다.

generation fence 변경은 destination handle allocation, copy descriptor, completion publish와 compute visibility를 바꾼다. mismatch 때 drop만 하는지 recompute/error로 request를 전진시키는지 정한다. dropped completion의 pin/buffer도 terminal돼야 한다.

metric exporter 변경은 runtime semantics를 고치지 않는다. denominator 수정으로 dashboard가 좋아져도 actual computed tokens는 그대로일 수 있다. exporter와 scheduler/allocator mutations를 별 commit으로 검증하고 release note에서 관측 의미 변경을 밝힌다.

cache disable/fallback option은 wrong reuse containment에 유용하지만 memory와 TTFT를 바꾼다. disabled path가 정말 lookup/load를 건너뛰고 recompute하는지, old cached owners가 drain되는지 확인한다. flag false가 기존 transfers를 취소하지 않을 수 있다.

canary cohort는 same/different identities, hit/miss, cancel/load delay와 block reuse를 포함한다. 정상 happy path만 흘리면 key isolation, finalizer와 generation fence를 실행하지 않는다. synthetic fixture를 작은 비율로 유지한다.

승격 gate는 cache savings, owner conservation, layer/output parity, no stale generation와 acceptable TTFT/memory다. 하나라도 깨지면 원인별 rollback을 수행한다. rollback 뒤 baseline을 새 config generation에서 다시 측정한다.

**장애 후 회고에서 남겨야 할 것.**

timeline은 user symptom보다 먼저 lookup query, reserve, load/write, compute read, output, cancel/release와 metric scrape를 둔다. first divergence와 downstream amplification을 다른 색으로 표시한다. OOM이나 wrong token만 root cause로 쓰지 않는다.

source evidence는 revision/file/symbol/span, branch input과 mutation을 포함한다. trace evidence는 request/cache/physical generation과 event ordering을 포함한다. tensor evidence는 earliest checkpoint coordinates와 reference를 포함한다. 각각이 주장하는 범위를 넘지 않는다.

반증된 가설을 기록한다. resident entry와 control hit로 eviction을, owner baseline 회복으로 leak를, cache-on/off parity로 wrong reuse를 약화한 근거처럼 쓴다. 다음 incident의 조사 시간을 줄인다.

affected scope는 model/adapter/cache generation, worker/rank와 시간 창으로 표현한다. raw prompts를 incident artifact에 넣지 않는다. wrong outputs가 전달됐다면 client-visible 영향과 후속 조치를 숨기지 않는다.

임시 대응 종료 조건도 둔다. cache disable을 언제 다시 켤지, quarantined entries/transfers를 언제 폐기할지, conservative allocation cap을 언제 해제할지 정한다. root regression과 canary가 통과하기 전 자동 복귀하지 않는다.

마지막으로 runbook owner와 다음 검증 날짜를 남긴다. cache 구현이나 default가 바뀌면 same 2,048-token fixture와 cancel/reuse barriers를 다시 실행한다. 문서가 고정된 결론이 아니라 revision에 묶인 재검증 절차가 된다.

**최종 네 줄 검산.**

miss 행은 queried2,032, valid hit0, computed2,048이며 output parity와 owner baseline이 정상이다. first divergence는 normalized adapter generation이 reader key에서 빠진 지점이다. 수정 뒤 same identity는 hit2,032/computed16이고 different identity는 intentional miss다.

leak 행은 cancel20, transfer completion20, request refs0인데 pin owners20과 blocks20이 남는다. first divergence는 completion callback의 early return이 self-contained unpin을 건너뛴 지점이다. 수정 뒤 all interleavings에서 unpin20, owner gap0이며 reused block late write도 0이다.

wrong reuse 행은 hit2,032/computed16, owner 합 정상인데 cache-on layer0 K/V checksum부터 reference와 다르다. first divergence는 old transfer generation이 reused destination에 publish된 지점이다. 수정 뒤 mismatch completion은 publish하지 않고 recompute/error terminal로 닫히며 layer parity가 맞는다.

이 세 줄이 같은 dashboard의 hit rate 하나로 축약되지 않게 한다. 각 줄은 performance denominator, lifetime owner, correctness checkpoint를 하나씩 소유한다. symptom이 겹쳐도 first divergence와 안전한 containment는 서로 다르다.

마지막 줄은 rollback이다. miss는 safe recompute와 namespace 격리, leak는 allocation fence·quarantine·last-consumer release, wrong reuse는 generation 격리와 cache disable을 사용한다. old owners/transfers, client terminal과 telemetry가 0으로 수렴한 뒤 readiness를 연다.

독자가 자신의 구현 metric과 source state로 이 네 줄을 다시 채울 수 있을 때 장이 완료된다. 채울 수 없는 칸은 새로운 일반론이 아니라 정확한 instrumentation 또는 source walk 과제다.

각 과제에는 고정 revision, 재현 fixture, expected counter/state/checkpoint, 담당 owner와 중단 조건을 붙인다. 측정이 추가되면 같은 request/cache generation으로 원장에 편입하고 first divergence를 다시 판정한다. 추정치로 빈칸을 닫지 않는다.

완료 판정에는 passing neighbor와 실패 cohort를 함께 보존한다. 수정이 원래 정상 hit·cleanup·parity를 깨지 않았다는 증거까지 있어야 재현가능하고 안전한 복구다.

**세 사건에 공통인 복구 원칙.**

miss 수정은 예상 eligible tokens만 hit해야 한다. key namespace를 넓혀 tenant/model isolation를 깨뜨리며 hit를 올리면 실패다. hit와 함께 recompute tokens와 TTFT critical path가 workload 조건에서 줄어야 한다. cold start와 eviction pressure에서도 정상 miss가 유지돼야 한다.

leak 수정은 steady-state 반복 뒤 physical owner partition가 같은 범위로 돌아오고 pending jobs, refs, locks가 release conditions 뒤 0/baseline으로 수렴해야 한다. process reset 한 번으로 OOM이 사라진 것은 종료 증거가 아니다. abort, natural finish, allocation rollback를 각각 확인한다.

correctness 수정은 cache on/off differential에서 first divergence가 사라지고 tenant/model/adapter/template isolation가 유지돼야 한다. boundary fixture와 shared prefix/COW, async copy/load completion를 다시 본다. 평균 hit rate나 한 답 문자열만으로 닫지 않는다.

복구 뒤 guard가 root cause를 숨기지 않는지도 본다. stale hit를 모두 drop하면 답은 맞아도 miss와 TTFT가 악화될 수 있다. pin timeout을 강제로 줄이면 leak는 사라져도 live transfer use-after-free가 생긴다. 수정은 first divergence의 identity·owner·ordering를 고쳐야 한다.

**incident를 다시 열어야 하는 신호.**

평균 hit가 회복됐지만 computed tokens가 줄지 않으면 miss incident는 닫히지 않았다. OOM가 사라졌지만 locked/pending owner가 단조 증가하면 workload가 짧아졌을 뿐 leak가 남았다. final output이 맞지만 first-layer divergence가 지속되면 sampling 우연이 증상을 가렸다.

cache off로 영구 fallback해 세 증상을 모두 지울 수 있다. 하지만 이 장의 복구 목표는 정상 cache contract를 회복하는 것이다. 안전을 위해 임시 disable한 상태와 root fix 완료를 release note에서 구분한다.

새 version에서 metric 이름이나 manager 구조가 바뀌면 기존 dashboard green을 completion evidence로 재사용하지 않는다. source definition와 owner partition를 다시 고정한다. 같은 incident label이 버전별로 다른 boundary를 가질 수 있다.

복구 보고서는 수정한 코드보다 관측 사슬을 먼저 남긴다. miss는 expected query·hit·computed/recompute와 TTFT decomposition, leak는 반복 횟수별 owner partition와 oldest pending age, wrong answer는 first checkpoint와 identity/generation/event를 기록한다. 다음 release에서 같은 symptom이 나왔을 때 regression인지 다른 boundary인지 비교할 수 있다.

종료 범위도 명시한다. local prefix path를 고쳤다고 remote load와 CPU offload, SWA/recurrent state까지 correctness가 증명된 것은 아니다. 수정이 닿은 cache group, tier, backend와 topology를 적고 나머지는 별 fixture로 확인한다. 한 green test를 제품 전체 cache contract로 확대하지 않는다.

성능 회귀 허용 범위는 correctness와 분리해 승인한다. narrow event wait 추가로 wrong answer가 사라졌지만 ITL이 변했다면 두 결과를 모두 보고한다. correctness fix를 되돌려 benchmark를 회복하지 않고, synchronization scope를 더 좁히는 후속 최적화를 한다.

leak fix가 cache residency를 줄여 hit rate를 낮출 수도 있다. 이전 높은 hit가 leaked lock/pin으로 content를 영구 보존한 결과였다면 그 수치는 건강하지 않다. owner invariant가 정상인 새 baseline에서 policy를 다시 조정한다.

miss fix가 새로운 sharing을 허용하면 tenant isolation audit를 필수로 붙인다. key namespace를 좁혀 hit를 올린 변화는 wrong-answer blast radius를 키울 수 있다. cache-on/off correctness와 cross-identity miss/hit matrix가 함께 통과해야 한다.

마지막으로 rollback 계획을 갖는다. suspect cache generation를 quarantine하거나 cache feature를 disable하는 안전 조치는 가능하지만, process reset 전에 trace/owner snapshot를 보존한다. 데이터와 사용자 영향 범위가 불명확하면 결과를 재사용하지 않고 해당 worker를 격리한다.

이 장이 닫은 것은 cache의 logical owner, generation과 address다. 40장은 그 주소를 실제로 소비하는 CUDA stream과 kernel launch로 내려가 producer completion, consumer wait와 launch generation이 같은 cache lifetime을 가리키는지 확인한다.
