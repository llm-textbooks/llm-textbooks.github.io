# 35장. prefix sharing의 정확한 경계: hash, full block과 partial COW

두 요청의 앞 token이 같다고 KV를 바로 공유하면 안 된다. 같은 model과 adapter가 만들었는지,
multimodal embedding이 같은지, 어느 cache group과 position인지, block이 완전히 계산됐는지와 physical
owner가 아직 살아 있는지를 확인해야 한다. prefix sharing은 문자열 비교가 아니라 identity와 lifetime
transaction이다.

이 장의 fixture는 block size 4다.

```text
A = [10,11,12,13,20,21]
B = [10,11,12,13,30,31]
```

첫 block `[10,11,12,13]`은 identity가 모두 같다면 공유할 수 있다. A의 두 번째 block에는 `[20,21]`
KV가 있고 아직 두 slot이 비었다. B가 이 physical partial block을 참조한 채 `[30,31]`을 append하면
A의 suffix를 덮는다. cached length는 4, prompt length는 6이며 두 값을 합치면 안 된다.

## 35.1 A/B fixture로 full과 partial ownership을 계산한다

block 0은 token range `[0,4)`, block 1은 `[4,8)`이다. A가 prompt 6 token을 prefill하면 block 0은 full,
block 1은 occupancy 2다. prefix cache가 full block만 shareable로 commit한다면 A의 cached prefix length는
4다. A request는 physical block 0과 private partial block 1을 소유한다.

B lookup은 first block hit를 얻는다. B remaining prefill은 `[30,31]` 두 token이다. B는 shared block 0의
refcount를 올리고 새 private block 2에 suffix를 쓴다. A block 1과 B block 2는 logical block index가
둘 다 1이지만 physical ID가 달라야 한다.

### COW가 필요한 정확한 조건

partial physical block을 여러 sequence가 참조하고 어느 sequence가 다른 suffix를 append하려 할 때
copy-on-write 또는 새 block allocation이 필요하다. 기존 filled prefix를 새 block에 copy하고 writer만
새 owner로 전환하거나, 처음부터 partial은 공유하지 않는다.

```text
before: phys1=[A20,A21,_,_] refs={A,B}
B append without COW: phys1=[A20,A21,B30,B31]  # B logical prefix도 틀림
B overwrite offset2가 아니라 offset0이면 A까지 즉시 오염
```

B의 correct suffix는 logical block `[30,31]`이므로 A partial 내용 `[20,21]`과 공통 prefix가 아니다.
이 fixture에서는 copy할 filled partial prefix조차 없다. first full block까지만 share하고 B는 fresh block을
받아야 한다. C가 `[10,11,12,13,20,99]`라면 A partial의 첫 token 20만 공통이지만 block-granular cache는
그 one-token hit를 공유할지 별 policy다.

### chained hash가 과거를 포함해야 한다

block hash를 현재 block token만으로 만들면 `[10,11,12,13]` 뒤의 `[20,21,_,_]`과 전혀 다른 과거
`[7,8,9,0]` 뒤 동일 full block `[10,11,12,13]`이 같은 key가 된다. attention KV는 이전 position/
prefix와 model position transformation에 의존하므로 unsafe하다.

```text
h0 = H(namespace, [10,11,12,13], extra0)
h1 = H(h0, [20,21,22,23], extra1)
```

parent hash가 chain을 만든다. block index/position, cache group과 extra identity가 namespace에 어떻게
반영되는지는 구현 source를 본다. hash equality는 candidate lookup일 뿐 physical content와 owner가
유효하다는 보장은 아니다.

### extra identity가 token equality를 제한한다

같은 token IDs라도 model revision, LoRA/adapter, prompt embedding, multimodal image/audio, cache salt,
attention/cache group이나 processor revision이 다르면 K/V가 달라질 수 있다. key에 영향을 주는 identity와
admission에서 공유를 금지하는 predicate 중 하나가 필요하다.

adapter A와 base model이 같은 text token을 쓰는데 adapter가 Q/K/V projection을 바꾸면 shared KV는
틀리다. multimodal placeholder token이 같아도 image embedding이 다르면 틀리다. extra key 누락은 hash
collision보다 훨씬 구조적인 namespace collision이다.

### hash collision과 stale physical generation

충분히 큰 hash도 collision 가능성이 0은 아니다. internal Python hash, stable cryptographic/external
hash는 재현성과 비용이 다르다. collision 검증/secondary identity가 있는지, cross-process cache key가
stable한지 구분한다.

hash index가 block ID 7을 가리키지만 block 7이 evict/reuse되어 generation 12의 다른 content라면 hash
entry는 stale하다. hash map removal과 physical free/reuse ordering, refcount/generation fence가 필요하다.
lookup hit counter만으로 correctness를 증명하지 않는다.

cache warmup과 hit test는 quality parity를 포함한다. no-cache reference, cache miss first run, hit second run의
logits를 deterministic 조건에서 비교한다. hit path만 final text가 같아도 small logits divergence가 sampling에서
숨을 수 있다. boundary layer/head fingerprint는 diagnostic lab에서만 사용한다.

H/L ratio만으로 T_saved를 계산하지 않는다. cached prefix가 long context attention read를 여전히 요구하는
decode path, prefill compute의 layer FLOP와 kernel efficiency를 반영한다. hit 12k가 hit 1k의 정확히 12배
이득은 아니다.

fallback이 remote miss/timeout 뒤 local recompute를 한다면 correctness는 유지되지만 double lookup cost가
생긴다. remote payload 일부 load 뒤 fallback할 때 provisional block/ref를 rollback해야 한다. connector
details를 별 장으로 넘기더라도 prefix transaction의 commit/abort는 이 사건 기록에 남긴다.

adapter hot reload는 same adapter name에 new weights가 붙는 ABA다. key에 name만 넣으면 old KV가 hit한다.
immutable revision/content digest가 필요하다. unload/reload 때 affected namespace entries를 invalidate하고
active requests가 old revision을 끝낼 fence를 둔다.

multimodal processor revision은 media digest가 같아도 embedding이 바뀔 수 있다. processor/model revision을
key에 넣거나 cache를 격리한다. placeholder position과 media ordering도 extra key에 반영돼야 한다. image1,
image2 순서가 바뀐 fixture를 둔다.

physical content fingerprint는 전체 KV dump가 아니다. 허가된 lab에서 selected layer/head/dim의 safe digest,
finite ratio와 norm을 쓴다. A block0과 B shared read digest가 같고 A/B suffix digest는 달라야 한다. production
always-on content hashing은 GPU traffic/보안 비용 때문에 신중하다.

공유 길이 L이 uniform modulo16이면 full-block usable은 `floor(L/16)×16`, 평균 tail loss7.5다. 실제 system
prompt는 특정 boundary에 몰릴 수 있어 histogram을 쓴다. tokenizer/template 변경이 boundary를 이동시켜
hit율을 바꿀 수 있다.

multi-group cache에서 same logical tokens가 group0 hit, group1 miss이면 request 전체 computed frontier를
group1에 맞춰 줄여야 할 수 있다. group0 blocks는 provisional ref를 얻었다면 rollback한다. “평균 group
hit”로 computed tokens를 정하지 않는다.

hybrid/sliding group의 detailed address/lifetime은 36장으로 넘기지만 key group ID와 all-groups consistency는
여기서 검증한다. model layer group이 바뀐 upgrade에서 old external hashes를 재사용하면 schema version
gap이다.

adapter mixture도 layer별이다. LoRA가 only LM head를 바꾸면 KV sharing은 safe할 수 있고, attention Q만
바꾸면 K/V cache는 base와 같을 가능성이 있지만 attention output/residual 이후 future-layer K/V가
바뀔 수 있다. layer0 K/V가 같아도 later layers가 달라져 full-cache sharing은 unsafe할 수 있다. adapter
target과 layer dependency를 전체 model forward로 판단한다.

이 제약 때문에 selective per-layer cache sharing은 복잡하다. implementation이 cache group별 identity를 지원하지
않으면 adapter revision 전체를 namespace에 넣는 보수 정책이 낫다. hit loss와 correctness를 교환한다.
source에 없는 selective optimization을 추측하지 않는다.

multimodal embedding은 placeholder가 포함된 block 이후뿐 아니라 attention을 통해 later token K/V에 영향을
준다. media extra key를 only placeholder block에 넣고 parent chain이 이후 blocks로 전파하면 identity가
이어진다. parent hash가 없으면 later text-only blocks가 잘못 collide할 수 있다. chained extra propagation의
핵심 사례다.

tenant salt가 parent root에 들어가면 모든 descendant key가 격리된다. block마다 salt를 반복 serialize할
필요 없이 chain에 포함될 수 있다. salt rotation 후 active request는 old namespace를 끝내고 new request는
new salt를 사용하도록 generation boundary를 둔다.

scan 자체가 active mutation과 race하지 않도록 snapshot/lock 또는 quiescent maintenance mode를 쓴다.
production에서 무거운 full scan을 자주 돌리지 않고 sampled invariants와 shutdown/offline audit를 조합한다.

abort boundary도 같다. A가 block0 device write 후 host complete mark 전 abort되면 block0을 shareable로
남길지 source policy를 확인한다. content가 complete라도 request semantics/extra key가 확정됐는지 본다.
partial block abort는 private free가 기본 안전선이다.

새 cache implementation을 읽을 때 class 이름보다 A/B fixture를 먼저 대입한다. `lookup(A/B)` 결과, physical
tables, write indices와 finish refs를 손으로 쓸 수 없다면 아직 이해하지 못한 것이다. feature brochure를
더 읽는 대신 owner/caller를 추적한다.

completion event가 CUDA stream에 있고 hash insert가 CPU에서 일어나면 host가 event를 기다리는지 본다.
graph replay 반환이나 kernel enqueue를 content-ready로 오인하면 B가 incomplete KV를 읽는다. cache commit
fence는 output token commit과 같거나 다를 수 있다. source caller와 event ordering을 연결한다.

same template prefix에서 adapter가 request 중간에 바뀌는 session은 더 어렵다. first segment KV는 old
adapter, later segment는 new adapter가 만들었다. request-global adapter ID 하나로 whole chain을 key하면
old segment safe hit를 잃거나 mixed state를 잘못 label할 수 있다. implementation이 mid-session adapter
change를 허용하는지 먼저 확인하고 unsupported라면 reject가 안전하다.

multimodal interleaving도 block별 extra key placement를 시험한다. media placeholder가 block1에 있고 block0은
pure text라면 block0은 images가 달라도 share 가능할 수 있다. 그러나 model architecture가 media embedding을
앞 token representation에 소급하지 않는지, parent chain 이후 blocks가 media identity를 유지하는지 본다.
보수적으로 request 전체 namespace를 달리하는 설계도 가능하다.

payload header는 token count, block/page size, group/layer/head/dtype와 checksum을 포함할 수 있다. source에
없는 exact format을 주장하지 않고 필요한 validation 질문으로 남긴다. index hit 뒤 schema mismatch는
safe miss/fallback이어야 한다.

template revision rollout에서 old entries를 유지해도 token IDs와 model same이면 safe할 수 있지만 이를
증명하기 어렵거나 side-channel policy가 바뀌면 namespace bump한다. decision과 근거를 release record에
남긴다.

운영 incident에서 prompt token을 재현할 수 없으면 length/block boundary와 safe hash digest만으로 source
path를 좁힌다. customer consent/secure lab에서 최소 fixture를 합성한다. production KV dump를 요구하지
않는다.

event retention이 없으면 보수적으로 worker/cache namespace 전체를 invalidate하고 affected time/model/
tenant output을 검토할 수 있다. 이는 운영 정책 결정이며 source만으로 impact를 확정하지 않는다.

### partial block의 writer 권한과 COW를 A/B에 적용한다

이번에는 adapter와 tenant가 모두 같다. A prompt는 `[10,11,12,13,20,21]`, C prompt는
`[10,11,12,13,20,99]`다. logical LCP는 5이지만 block size 4에서 full shareable prefix는 4다. 초보자는 A의
두 번째 partial block `[20,21,_,_]`을 C가 참조하고 마지막 token 21만 99로 고치면 한 token 계산을 아낄
수 있다고 생각하기 쉽다. 그러나 KV는 token ID 배열이 아니다. position 5의 K/V를 덮는 순간 A의 future
decode가 같은 physical byte를 읽을 수 있다. writer isolation이 먼저다.

위 fixture의 partial payload는 block 전체 512 KiB이고 filled two-position payload는 256 KiB다. COW가
filled common position 20 하나만 보존한다고 해도 모든 32 layer에서 K와 V의 해당 slot을 destination으로
복사해야 하므로 128 KiB 이동이다. 그 뒤 position 5를 새로 계산해 다른 128 KiB를 쓴다. one-token prefill을
아끼려고 allocation, 128 KiB device copy, stream dependency와 metadata update를 추가하는 셈이다. model과
page layout에 따라 copy kernel 비용이 saved compute보다 클 수 있다. “partial hit token 1”만 보고 이득을
판단할 수 없다.

안전한 full-only 경로는 단순하다. A와 C는 first full block만 공유해 refs 2를 만들고, A partial과 C partial은
각각 physical 51과 52를 받는다. A write slots는 `(51,0),(51,1)`, C write slots는 `(52,0),(52,1)`다. C가
끝나 refs가 내려가도 A의 51은 영향받지 않는다. hash granularity가 4이면 partial block은 hash lookup
candidate조차 되지 않는다. vLLM request hasher가 “full blocks only”에서 멈추고 block pool이 full range만
commit하는 것은 lost-tail과 안전성의 명시적 교환이다.

COW를 지원한다면 transaction은 다섯 단계여야 한다. destination 53을 provisional allocate한다. A partial
51의 공통 position 0 payload를 copy stream에 enqueue한다. event가 완료되기 전 C block table을 reader-visible
commit하지 않는다. 완료 뒤 C logical block 1 mapping을 53으로 바꾸고 position 1에 token 99 KV를 쓴다.
마지막으로 old shared reference를 감소시킨다. allocation 실패, copy 실패, cancel이 어느 순간 와도 53과
provisional ref를 rollback하고 A mapping은 변하지 않아야 한다. source가 이 상태를 표현하지 않으면
“partial COW가 있을 것”이라고 추정하지 않는다.

경쟁 조건을 숫자로 보자. t0 A/C가 physical 51을 refs2로 공유했다고 가정한다. t1 C가 destination 53을
allocate해 refs1 provisional 상태다. t2 128 KiB copy가 stream S1에 enqueue된다. t2.2 scheduler가 C를
취소하고 53을 free queue에 넣는다. t2.4 D가 53을 재할당한다. t2.8 늦은 S1 copy가 완료되면 D의 KV를
덮는다. 이 결함은 C output이 이미 취소돼 정상처럼 보이며 D에서 늦게 나타난다. 해결에는 allocation
generation과 CUDA completion fence가 모두 필요하다. completion `(block53,generation7)`은 현재 owner가
generation8이면 mutation이나 free를 수행하면 안 된다.

다른 경쟁은 reader 생존이다. A가 먼저 finish해 block 51의 ref를 2→1로 내리고 C가 계속 읽는 동안 cache
eviction이 51을 victim으로 고르면 안 된다. cache owner convention과 live reader ref가 합쳐져 eviction
eligibility를 결정해야 한다. C가 먼저 finish하고 A가 decode를 이어 가는 반대 순서도 시험한다. 두 순서에서
output parity와 final refs 0, hash entry/cache owner 상태가 같아야 한다. 한 순서만 검사하면 asymmetric
cleanup bug를 놓친다.

관측값은 COW 횟수 하나가 아니다. `logical_lcp=5`, `accepted_shared=4 또는 5`, `copied_positions=1`,
`copied_bytes=131072`, `destination_generation=7`, `copy_ready_at`, `writer_commit_at`, `old_ref_after`,
`rollback_reason`을 한 trace에 둔다. production metric에는 raw block ID와 request ID를 label로 넣지 않고
copy byte, latency, failure reason과 sampled trace link만 둔다. accepted shared length가 5인데 copied byte가
0이고 physical writer가 같다면 즉시 correctness 경보다.

### identity transition 사고: 같은 문장인데 답이 바뀌었다

오전 9시 12분, tenant Blue의 고객 지원 봇이 갑자기 tenant Red의 제품명을 답했다. 두 tenant는 같은
base model과 같은 system prompt 문장 “당신은 친절한 상담원입니다”를 쓰지만 adapter가 다르다. Blue는
adapter `support-blue`, Red는 `support-red`다. 운영 화면은 오류 직전 요청에서 prefix-cache hit 100%,
TTFT 41 ms라는 좋은 숫자를 보여 주었다. cache를 끈 재시도는 TTFT 83 ms였고 답은 정상이었다. 이 장면의
위험은 “hash가 우연히 충돌했다”는 데 있지 않다. 서로 달라야 할 입력이 애초에 같은 preimage로 만들어진
namespace collision일 수 있다는 데 있다.

fixture를 block size 4, layer 32, KV head 8, head dimension 128, FP16으로 고정하자. 공통 token은 12개여서
full block 세 개다. block 하나의 KV payload는 `32×8×128×4×2(K,V)×2 byte = 524,288 byte`, 즉 512 KiB다.
세 block을 잘못 공유하면 1.5 MiB의 다른 adapter KV가 Blue 요청에 붙는다. 요청마다 suffix 두 token만 새로
계산하므로 output은 정상적인 문법을 유지할 수 있다. 바로 이 점 때문에 crash보다 위험하다. shape와 dtype,
pointer는 모두 유효하고 내용의 의미만 틀리다.

직관은 우편함 열쇠와 비슷하다. token IDs는 거리 주소이고 adapter·model·position·tenant 정책은 동과 호수다.
거리 이름만 같다고 같은 우편함을 열면 안 된다. 반대로 모든 software version 문자열을 무조건 열쇠에 넣으면
안전해 보이지만, KV를 바꾸지 않는 로깅 패치에도 캐시가 전부 무효화된다. 올바른 질문은 “무엇이 같아야
하는가”가 아니라 “어느 상태 차이가 이 block의 K/V byte를 바꾸거나 공유 정책을 금지하는가”다.

**identity를 넣는 순서는 처리 pipeline과 hash tuple을 구분한다.** 먼저 HTTP text와 message array가
tokenizer와 chat template를 지나 token IDs와 special-token 배치를 만든다. tokenizer/template identity는
이 단계의 결과에 반영된다.

다음으로 model deployment가 weight revision, architecture, RoPE scaling과 KV
layout을 확정한다. adapter manager가 request에 LoRA identity를 붙이고 multimodal processor가 media
feature identifier와 block 안 offset을 만든다. session 또는 cache manager가 position/context parent와
tenant salt를 결정한다. 마지막에 block hasher가 parent hash, 현재 full-block token tuple과 native extra
keys를 직렬화해 digest를 만든다. 이 pipeline 순서와 Python tuple 안의 필드 순서는 같은 개념이 아니다.

vLLM 고정 source에서 실제 tuple은 더 좁고 구체적이다. `generate_block_hash_extra_keys`는 LoRA name,
multimodal `(identifier, relative offset)`, 첫 block의 cache salt, prompt-embedding digest 순서로 `extra_keys`를
결합한다. 이어 `hash_block_tokens`는 `(parent_block_hash, curr_block_token_ids_tuple, extra_keys)`를 hash
function에 넘긴다. [extra key와 chained tuple](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_utils.py#L517-L623)

그래서 “model revision→tokenizer→template→adapter→position→tenant를 한 tuple에 순서대로 넣는다”라고
쓰면 source 사실이 아니다. LoRA는 native extra key에 명시되지만 model revision은 이 함수의 tuple에
보이지 않는다. tokenizer와 template는 산출 token IDs를 통해 간접 반영될 뿐 revision 자체가 들어가지
않는다.

position은 root에서 시작한 parent chain과 token boundary에 암묵적으로 걸리지만 arbitrary absolute
offset field가 명시된 것은 아니다. tenant isolation은 caller가 `cache_salt`를 공급할 때 첫 block에서 chain
전체로 전파된다.

이 “없음”은 곧바로 취약점이라는 뜻도 아니다. 한 worker의 cache가 단일 immutable model deployment에만
속한다면 process/cache namespace 자체가 model identity다. 서로 다른 model revision이 같은 physical
block pool을 공유하지 않으면 매 block key에 model digest를 반복할 필요가 없다. tokenizer/template가
달라도 결과 token IDs가 완전히 같고 model inputs의 다른 부분도 같다면 KV도 같을 수 있다. 반대로 external
cache를 여러 deployment가 공유하거나 mutable alias가 hot reload되면 process 경계가 더 이상 identity를
보장하지 않는다.

그때는 connector namespace 또는 payload schema에 immutable model/config digest가
필요하다. key 함수만 보고 전체 isolation을 판정하면 안 되는 이유다.

consumer 산책은 hash 생성에서 멈추지 않는다. request hasher는 full `hash_block_size`만 순회하며 이전
block hash를 다음 parent로 넘긴다. KV manager의 `get_computed_blocks`는 request hashes를 coordinator의
`find_longest_cache_hit`에 주고 모든 관련 group에서 가능한 longest prefix를 받는다. [longest hit consumer](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_manager.py#L229-L295)

block pool의 `cache_full_blocks`는 resolved hash에 cache group ID를 결합하고 physical block map에 삽입한다.
[full block commit consumer](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/block_pool.py#L225-L341)

identity는 `request field → extra tuple → chained hash → longest lookup → group-qualified physical block →
computed frontier`로 소비된다. 어느 화살표가 끊겨도 “key에 넣었다”는 말은 충분하지 않다.

수치 사고를 시간축으로 재구성한다. t0에 Red 요청 R이 token 12개를 계산해 physical blocks 40, 41, 42를
full commit하고 refs는 각각 1이다. R이 끝난 t1에는 refs가 0이지만 cache owner convention으로 entries가
남는다.

t2에 Blue 요청 B가 같은 token IDs와 같은 `lora_name="support"`를 갖고 들어온다. 실제 weights는
서로 다르지만 mutable alias가 같았다고 하자. 세 chained hash가 모두 같아 longest hit 12가 되고 refs는
0→1이다. B는 마지막 prompt token을 logits 때문에 일부 재계산하고 suffix decode로 넘어간다. t3에 첫
output이 41 ms에 나오지만 layer 1부터 Red adapter가 만든 residual history의 KV를 읽는다. 잘못된 identity는
lookup에서 처음 드러났으나 의미 오염의 뿌리는 t2 이전 adapter alias resolution이다.

반증은 네 갈래로 한다. 첫째 cryptographic collision이면 서로 다른 fully serialized preimage가 같은 digest다.
둘째 schema collision이면 preimage가 같지만 빠진 identity가 있다. 셋째 stale generation이면 key는 맞지만
hash map이 재사용된 physical block을 가리킨다. 넷째 write corruption이면 lookup과 owner는 맞고 suffix
write index가 shared range를 덮는다. incident trace에서 Red와 Blue의 serialized extra tuple이 동일하고
adapter content digest가 다르면 둘째다. stronger SHA를 켜도 고쳐지지 않는다. immutable adapter revision을
namespace에 넣거나 alias reload 때 해당 namespace를 invalidate해야 한다.

tenant isolation은 correctness와 confidentiality를 분리한다. 두 tenant가 같은 model·adapter·prompt를
사용하면 KV byte는 수학적으로 같아 cross-tenant reuse가 correctness상 가능하다. 그러나 hit/miss에 따른
TTFT 차이가 다른 tenant의 prompt 존재를 드러낼 수 있고, 운영 정책상 물리 공유 자체가 금지될 수 있다.
tenant salt를 root extra key에 넣으면 첫 hash가 달라지고 parent chain 때문에 모든 descendant가 분리된다.
tenant별 salt rotation은 새 요청만 새 generation으로 보내고 old active request는 old chain을 끝내게 한다.
salt 문자열을 raw tenant ID로 로그에 남기지 않고 bounded cohort와 safe digest를 쓴다.

containment는 cache 전체 재시작이 가장 단순하지만 영향이 크다. 우선 unsafe adapter alias와 tenant namespace의
새 admission에서 prefix read를 끈다. active hits는 output을 멈추고 generation을 구분해 늦은 completion을
버린다. affected hash descendants와 blocks 40–42의 reference requests를 추적하고, entries를 invalidate한 뒤
refcount가 0이 될 때 physical free를 허용한다. adapter를 immutable content revision으로 다시 등록하고 cold
miss parity를 확인한 다음 tenant 하나의 canary에서 cache를 연다. TTFT 회복보다 no-cache logits parity,
wrong-hit 0, refs baseline 복귀가 먼저다.

## 35.2 vLLM은 chained full-block hash와 physical block map을 분리한다

vLLM의 block hash utility는 internal/external hash, group identity와 extra keys를 다룬다.
[`kv_cache_utils.py` hash 생성](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_utils.py#L549-L748)은
request token을 full hash-granularity chunks로 나누고 parent hash와 multimodal/prompt embedding extra를
결합한다.

### hash granularity와 cache block size

hash block size와 concrete cache-group block size가 다를 수 있다. source는 divisibility/compatibility를
검증하고 lazy view에서 target granularity에 맞는 last chained hash를 고른다. 4-token hash가 두 개
모여 8-token physical block을 나타내면 8-token boundary hash만 lookup key가 될 수 있다.

granularity conversion을 무시하면 cached length를 4로 보고 실제 physical block 8을 share하거나 반대로
hit를 놓친다. option→validation→resolved sizes→hash view→lookup length를 닫는다.

### longest computed prefix lookup

[`KVCacheManager.get_computed_blocks`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_manager.py#L229-L274)는
request block hashes에서 cache groups가 모두 만족하는 longest prefix blocks와 token count를 찾는다.
첫 miss 이후 뒤 hash가 우연히 존재해도 prefix chain을 건너뛰지 않는다.

A/B에서 h0 hit, h1은 full hash가 없으므로 cached tokens=4다. B scheduler는 computed frontier를 4로
시작하고 suffix 2를 새 block에 쓴다. hit length, returned block IDs와 allocation commit을 따로 본다.

### full block cache commit

[`BlockPool.cache_full_blocks`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/block_pool.py#L225-L347)는
already-cached와 newly-full range를 구분하고 hash map에 full blocks를 넣는다. partial→full promotion이면
old shorter hash metadata를 제거하고 새 token boundary hash를 삽입하는 경로가 있다.

hash map insert는 block ownership transaction의 일부다. block hash, group ID, refcount/free queue와 events가
일관돼야 한다. null/masked cache groups와 external event hash도 native semantics를 유지한다.

**A/B physical ownership**

B lookup으로 block 0을 얻었다고 즉시 writer가 되는 것이 아니다. cached block refs/active request
assignment이 증가하고 suffix allocation은 별도다. block pool은 cached hash→block과 block→hash metadata를
관리한다. free queue에 있으면서 cached인 block이 재사용될 때 hash entry 제거/eviction ordering을 본다.

partial block을 hash cache에서 제외하면 A/B fixture COW 위험을 구조적으로 줄인다. 구현의 partial
promotion/append semantics를 allocator call까지 확인한다. “vLLM은 COW한다”처럼 source 범위를 넘는
일반화 대신 full-cache predicate와 private suffix allocation을 고정한다.

**option과 효과**

prefix caching/hash algorithm/block size option은 validation을 거쳐 request hash generation, pool lookup과
allocation consumer가 된다. mutation은 cached token frontier, refcount/hash map와 new blocks다. 물리
효과는 skipped prefill compute/KV reuse, hash CPU work와 fragmentation이다.

hit count가 늘어도 B suffix/lookup/connector/graph overhead 때문에 TTFT가 줄지 않을 수 있다. actual
cached tokens, new query, cache load readiness와 first model start를 본다.

hash CPU 비용을 대략 계산한다. prompt 16,384, hash block4면 4,096 chained operations, block16이면 1,024다.
cryptographic digest가 operation당 cost를 더해도 prefill GPU saved work와 비교해야 한다. full token bytes를
매번 serialize하는지 incremental digest를 쓰는지 source를 본다. 작은 prompts에서는 lookup/hash overhead가
saved compute보다 클 수 있다.

vLLM `cache_full_blocks` partial→full promotion도 시간축으로 본다. A partial occupancy2에서 두 token을 더
써 full이 된다. old shorter/partial hash metadata가 있었다면 remove하고 8-token boundary hash를 insert한다.
lookup thread가 old/new 사이를 볼 수 있는 ordering을 manager serialization/ref map으로 막아야 한다.

A/B request hit=100%, token hit=4/6≈66.7%, usable full-block hit=66.7%다. block size8이면 logical LCP4가
있어도 usable physical hit0일 수 있다. remote payload가 timeout이면 index hit는 66.7%지만 ready hit0이다.
dashboard label에 numerator/denominator와 owner를 쓴다.

prefix key schema version은 serialized/external cache header와 startup effective log에 남긴다. code deploy가
key inputs를 바꿀 때 version을 bump하고 old namespace를 읽지 않는다. rolling deploy에서 old/new workers가
같은 external cache를 공유하면 version isolation이 특히 중요하다.

hash granularity를 4→16으로 키우면 16k prompt의 hash operations는 4,096→1,024로 줄지만 suffix divergence
평균 reusable tail을 최대 15 token 잃을 수 있다. block metadata/hash map entry도 줄고 internal fragmentation/
miss granularity가 바뀐다. prompt shared-prefix length distribution으로 expected lost hit를 계산한다.

block size와 hash size가 다르면 conversion이 additional rule을 만든다. hash4, physical16이면 h at 16-token
boundary를 선택해야 한다. physical16 group 하나라도 missing이면 multi-group longest hit가 더 짧아질 수
있다. groups별 block sizes/availability를 원장에 둔다.

external KV cache에서는 producer와 consumer model/config digest가 일치해야 한다. hash key가 같아도 tensor
layout, dtype/scale, layer grouping이나 CUDA architecture-specific format이 다르면 payload를 직접 읽을 수
없다. key namespace와 payload schema/version을 분리한다. connector가 validation/fallback을 갖는지 본다.

block size를 줄여 hit granularity를 높이면 block table/hash/node metadata와 allocator work가 늘고 attention
kernel page support가 제한될 수 있다. address translation/kernel details는 34장에 두고 이 장에서는
option validation과 usable hit/ownership effect만 기록한다.

## 35.3 SGLang radix tree는 token path, node lock과 unfinished cache를 소유한다

SGLang prefix cache는 radix tree node/path로 token sequence prefix를 찾는다. hash-indexed fixed full-block
map과 같은 자료구조가 아니다. request token path, node value/KV indices, lock reference와 eviction
ownership을 세로로 읽는다.

[`RadixCache`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/radix_cache.py#L1-L260) 계열의
match/insert/lock 흐름을 고정 revision에서 확인한다. page-aligned variant와 일반 radix의 partial key/value
semantics를 동일시하지 않는다.

### A/B radix match

A finished/cache insert 뒤 root에서 `[10,11,12,13]` path가 존재한다. node가 `[10,11,12,13,20,21]`
처럼 더 긴 compressed edge를 가진다면 B match는 edge 안에서 divergence를 찾아 split할 수 있다.
matched length는 4이고 last node/value indices가 B prefix ownership의 출발점이다.

fixed block hash와 달리 token-granular radix는 partial edge match를 표현할 수 있다. 그러나 KV allocator가
page-aligned라면 실제 reusable physical range는 page boundary로 제한될 수 있다. logical matched tokens와
allocatable shared KV indices를 구분한다.

### lock reference는 eviction을 막는 lifetime이다

matched prefix node/ancestors를 request가 사용할 동안 lock/refcount를 올려 eviction victim이 되지 않게
해야 한다. request finish/abort 또는 prefix owner 전환에서 감소한다. tree path가 존재한다는 것과
physical KV가 안전하게 pin됐다는 것은 다른 상태다.

A/B가 같은 node를 lock하면 shared prefix KV는 두 request lifetime에 걸쳐 보존된다. B suffix indices는
private이어야 한다. unlock이 너무 빠르면 B attention이 읽는 동안 eviction/reuse될 수 있고, 너무
늦으면 cache capacity leak이다.

### unfinished request cache

`cache_unfinished_req`는 진행 중 request의 computed prefix를 tree/cache에 반영하는 경계다. finished
request cache와 달리 last partial suffix, request-to-token mapping과 lock owner가 아직 변할 수 있다.
insert length와 cacheable aligned prefix를 확인한다.

A가 6 token에서 unfinished라면 full/page-aligned 4까지만 global shareable인지, 6까지 radix node에
insert하되 physical indices를 어떻게 보호하는지 native implementation을 읽는다. B가 divergence 4에서
새 suffix를 쓸 때 A last node/index를 재사용하지 않아야 한다.

**policy와 prefix cache를 분리한다**

SGLang LPM/DFS schedule policy가 radix match length를 ordering에 사용할 수 있지만 cache correctness와
fairness는 별도다. match_prefix가 hit를 반환해도 PrefillAdder allocation/lock commit이 실패할 수 있다.
hit metric, queue priority와 physical KV reuse를 세 경계로 둔다.

**extra identity와 namespace**

Req extra key/cache salt, adapter/model, multimodal padded/hash identity가 radix lookup key에 어디서 포함되는지
확인한다. token path만 같다면 unsafe한 modality가 cache sharing에서 excluded되는 predicate가 있을 수
있다. 없는 identity는 gap이다.

**recovery effect**

radix cache를 flush하면 stale node/KV mapping을 제거할 수 있지만 active locked owners와 allocator state를
안전한 idle boundary에서 처리해야 한다. wrong namespace entries만 선택 격리할 수 있는지도 본다.
flush 후 hit=0은 correctness 복구의 임시 증거이지 root cause 해결이 아니다.

radix match는 prompt 길이와 edge traversal, split/insert cost가 있다. compressed edge가 길면 common prefix
comparison byte/token cost가 있고 waiting LPM policy가 여러 request를 match하면 scheduler CPU가 늘 수 있다.
tree match와 actual cache reuse를 분리해 host lookup duration을 기록한다.

SGLang node lock도 ancestor/descendant lock counts와 evictable size accounting을 검산한다. matched prefix
lock이 root-to-last path에 걸리는지 last node만인지, insert/split 때 lock을 새 nodes로 어떻게 옮기는지
source를 본다. node tree 존재와 allocator indices lifetime이 함께 유지되어야 한다.

SGLang lock reference도 root/cache-owned baseline이 있을 수 있다. node lock count0가 evictable 의미인지,
negative/ancestor propagation과 total evictable size update를 source로 확인한다. common refcount 식을 네
stack에 강제하지 않는다.

radix token path는 extra identity를 root namespace/extra key로 분리하지 않으면 parent chain만으로 media
차이를 모른다. Req extra key가 match_prefix와 tree key에 실제 소비되는지 caller까지 본다. scheduler
policy용 simulated radix와 physical tree cache를 혼동하지 않는다.

stale hash scan은 hash→block과 block→hash를 양방향 비교하고 refs/complete/generation, token boundary를
검증한다. radix는 parent/child key edge와 value KV indices가 allocator live set에 있는지, lock/evictable
accounting을 본다. Transformers는 block parent/ref/complete와 allocator tables, llama RAM은 entry header/
model/context identity를 본다.

cache capacity pressure에서 어떤 entry를 evict할지도 hit performance를 바꾼다. LRU, radix evictable size,
free queue와 priority는 owner가 다르다. 36장의 eviction hierarchy를 반복하지 않고, active refs/locks가
victim 제외되고 removal transaction이 key/physical state를 함께 갱신한다는 조건만 둔다.

## 35.4 Transformers는 prefix match와 block refcount/fork를 transaction으로 잇는다

Transformers continuous batching의 `PagedAttentionCache.search_prefix_match`는 request tokens와 cache block
state에서 reusable prefix를 찾고 scheduler가 remaining prefill을 계산한다. `BlockManager`는 physical
blocks의 refcount, complete/hash state와 fork/free를 소유한다.

[`BlockManager`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache_manager.py#L37-L295)는
`fork_blocks`, increase/decrease refcount, free와 shareable-complete marking을 제공한다.

**A/B prefix lookup과 complete block**

A block 0이 complete/shareable로 commit됐고 block 1은 partial/private라고 하자. B search는 block 0을
match하고 refcount를 올린다. scheduler는 B remaining tokens 2를 query로 만들고 allocator가 private
suffix block을 준다.

complete flag는 hash/key 존재와 별도다. 아직 device write/host update가 끝나지 않은 block을 complete로
노출하면 B가 partial content를 읽는다. output/cache write commit 뒤 mark-shareable ordering을 본다.

**fork와 refcount**

fork는 parent blocks를 child request에 연결하고 block refcount를 증가시키거나 필요한 copy를 수행한다.
child가 finish/cancel하면 decrease refcount하고 0에서 free pool로 돌아간다. parent finish가 먼저여도
child ref가 있으면 physical block은 살아야 한다.

A/B fixture의 shared block 0 refcount가 2이고 A가 끝나면 1, B도 끝나면 0이다. partial A/B blocks는
각각 refcount 1이어야 한다. total refcount가 맞아도 B가 A partial ID를 잘못 가진다면 corruption이므로
request block table 내용도 검사한다.

### partial COW

fork가 partial block을 공유할 수 있는 path라면 append 전에 COW가 필수다. complete full blocks만 fork
대상이라면 invariant가 COW 필요를 제거한다. source에서 `fork_blocks`의 block range와 allocator copy,
unshared initialization을 확인한다.

C가 A prefix 5 token을 fork하려 해도 block-granular share length는 4일 수 있다. cached logical length,
query remaining과 physical block table을 함께 기록한다.

### hash와 parent relation

Block은 parent/group relation과 complete 상태를 가진다. prefix match key/hash, block manager mapping과
physical cache tensor가 같은 generation을 가리켜야 한다. free 후 reused ID의 old parent/hash가 남으면
stale hit다.

**scheduler/output commit**

prefix hit로 FutureRequestState의 query length가 줄고 static IO read indices/block table이 shared blocks를
가리킨다. model write는 B private suffix indices에만 가야 한다. update 뒤 newly complete blocks를
shareable로 표시한다. lookup→refcount→IO mapping→write→complete의 순서다.

**options**

allow block sharing/prefix config, block size와 safety margin은 validation 후 cache allocator/search/fork가
소비한다. mutation은 matched length, refs, block table과 query length다. 효과는 prefill work/KV 절약,
refcount metadata와 fragmentation이다. hit만 보고 output row correctness를 추론하지 않는다.

### physical ownership

request table/mapping, ref/lock, complete, free queue와 reverse map을 적는다.

```text
block0 owners={A,B}, refs=2, complete=true
blockA1 owners={A}, occupancy=2, content=[20,21]
blockB1 owners={B}, occupancy=2, content=[30,31]
```

llama slot-local이면 time-ordered owner A→B, kept `[0,4)`, removed `[4,6)`을 쓴다.

### write/COW와 completion

suffix write indices, exclusive predicate, allocate/copy/remove와 table update를 적는다. COW async면 fence를
본다. full-only sharing이면 invariant assertion을 적는다.

device KV write, host progress, complete/hash insert와 event ordering을 잇는다. unfinished/abort cache 경계를
분리한다. content ready 전 hit 노출이 없어야 한다.

refcount overhead는 block 수와 request sharing fanout에 비례한다. 1,000 requests가 common system prefix
8,000, block16을 공유하면 shared blocks 500개의 refs/owner mapping을 관리하고 request block tables는
각각 prefix references를 가진다. unique KV는 크게 줄지만 metadata와 finish decrement storm이 생길 수
있다. request count와 block refs를 모두 profile한다.

finish 1,000개가 동시에 refcount를 내릴 때 마지막 owner transition만 physical free/cache eviction eligibility를
바꾼다. atomic/serialized manager semantics와 event emission 비용을 본다. ref underflow는 duplicate finish/
cancel, missed decrement는 leak이다. 합 invariant는 `refcount == live request tables referencing block + cache
owner convention`처럼 구현 정의를 반영한다.

Transformers fork에서 parent/child finish order 네 가지를 fixture로 둔다. A before B, B before A, simultaneous
cancel, child fork failure다. fork failure 뒤 provisional refs/copies를 rollback해야 한다. free block count와
request tables 합이 원상 복구되는지 expected를 쓴다.

partial COW corruption은 다른 request 응답까지 바꾸므로 severity가 높다. detection 후 cache hit를 disable하고
affected worker를 drain/restart해 physical pool을 clean slate로 만든다. 단순 hash namespace bump는 이미 active
corrupted block content를 제거하지 않는다.

unique KV saved는 logical hit token×per-token byte에서 shared physical blocks와 TP replication을 반영한다.
100 requests가 4-token block 하나를 공유하면 logical reused tokens 400이지만 physical unique saved는
99 blocks×block byte다. refcount metadata와 block table references는 추가된다. cluster byte는 rank별
KV head partition/replication을 적용한다.

refcount invariant를 더 구체화한다. cached block이 free queue에 refs0로 남아 hash hit 후보가 될 수 있는
설계에서는 refs0가 곧 invalid가 아니다. hash cache owner convention과 eviction eligibility를 포함해야
한다. active request refs와 cached status를 각각 기록한다. generic `refs>0` assertion을 강제하면 정상
cache를 깨뜨릴 수 있다.

Transformers BlockManager에서 complete/shareable block이 free request 뒤 cache에 남는지 free pool로 가는지는
`free_blocks(...,shareable)` 호출 인자와 policy에 달렸다. request finish reason과 complete blocks count가
정확히 전달되는지 본다. abort partial은 shareable로 표시하면 안 된다.

llama slot prompt cache는 refcount가 없어도 slot active/idle owner와 task ID가 lifetime fence다. active slot을
new task가 선택하지 않는 predicate가 concurrent overwrite를 막는다. RAM entry는 별 owner/ref/LRU state를
가질 수 있다. block refcount metric 부재를 correctness 부재로 오독하지 않는다.

partial boundary lengths 3,4,5,7,8을 테스트한다. full boundary 직전/직후에 cached length, allocations와
write indices가 expected step function인지 본다. COW/off-by-one bug가 여기서 드러난다.

fork failure fault injection expected도 문서화한다. shared refs 일부 증가 후 new suffix allocation이 실패하면
all increments rollback되고 B table은 제거돼야 한다. A refs/content는 원상이다. retry가 새 B generation으로
안전하게 시작한다.

책에서 원본 source 일부를 인용할 때 hash/key algorithm의 핵심 branch와 refcount/COW 경계만 짧게 쓴다.
긴 함수 전체를 복제하지 않고 pinned link와 해설로 독자가 따라가게 한다. version 변화 가능성을 명시한다.

마지막 선택은 correctness gate 이후 performance다. identity/COW/stale-generation evidence가 없으면 hit율
이득과 무관하게 보수적으로 sharing을 끈다. evidence를 갖추고 workload repeated prefix가 충분할 때
granularity/hash/tier를 최적화한다.

실제 source review 보고서는 한 request diagram보다 두 request concurrency diagram을 포함해야 한다.
A lookup/compute/cache commit과 B lookup이 겹칠 때 어느 lock/manager thread가 serialize하는지 적는다.
single-thread unit test만으로 concurrent ref/eviction 안전을 증명하지 않는다. 반대로 manager event loop가
완전히 serialized라면 unnecessary atomic/COW complexity를 상상하지 않는다.

cache salt는 accidental cross-tenant hit를 막지만 wrong write indices/COW를 막지 않는다. refcount/generation
invariant가 여전히 필요하다. 하나의 보호를 전체 안전성으로 확대하지 않는다.

root cause 수정 후 observability를 추가한다. namespace digest, usable hit, block generation와 COW counter가
향후 같은 사고 first divergence를 잡는다. high-cardinality raw hash를 metric label로 넣지 않고 sampled
trace/event store를 쓴다.

## 35.5 llama.cpp slot-local LCP와 RAM prompt cache는 global block hash cache가 아니다

llama.cpp server의 slot은 자기 prompt tokens와 context KV를 보존할 수 있다.
[`server-context.cpp` prompt cache 처리](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L3128-L3170)는
old slot prompt와 new token의 longest common prefix를 비교하고 divergent suffix KV를 제거해 prefix를
재사용한다.

**A/B slot-local reuse**

slot이 A `[10,11,12,13,20,21]`을 가졌고 B를 같은 slot에 배치하면 LCP=4다. slot KV에서 position
4 이후 A suffix를 remove하고 B `[30,31]`을 새로 decode한다. global hash map/refcount로 A와 B가 동시에
block 0을 공유하는 그림이 아니다. 한 slot state를 새 task로 전환하는 reuse다.

A task가 아직 active인데 B가 same slot을 쓰지 않는 available predicate가 있어야 한다. LCP selection은
idle/available slot placement와 결합된다. concurrent shared physical block COW와 같은 semantics로 쓰지
않는다.

**cached-token metric의 의미**

LCP 4이면 cached prompt token 4로 보고할 수 있다. 실제 TTFT는 suffix 2 compute, slot selection, KV
remove, batch packing과 graph/model time을 포함한다. batch size/processing order에 따라 cached token과
timing이 달라질 수 있는 문서 주의를 source behavior와 연결한다.

**RAM prompt cache**

별도 RAM prompt-cache entry는 slot/context state를 serialize/save/load하고 entry limit/LRU와 LCP-based
selection을 가질 수 있다.
[`server-task.cpp` RAM prompt cache](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-task.cpp#L1691-L1872)는
entry size/limit, eviction과 longest-common-prefix load 좌표다.

RAM entry reuse는 GPU-resident paged block refcount와 다르다. serialization byte, load time와 context
restore correctness가 있다. slot-local in-memory KV reuse와 RAM saved state를 metric에서 분리한다.

**extra identity**

RAM/slot cache key가 tokens 외 model/context parameters, adapter/control vector와 multimodal embeddings를
어떻게 격리하는지 본다. unsupported combination에서 cache를 disable하는 predicate가 있을 수 있다.
token LCP만으로 semantic equality를 가정하지 않는다.

**recovery**

wrong slot cache가 의심되면 affected slot/RAM entry를 invalidate하고 new context에서 parity를 본다.
global paged hash namespace flush와 다른 target이다. active task result/slot owner를 안전하게 끝낸 뒤
reuse한다.

RAM prompt cache serialization은 model/context binary compatibility를 요구한다. entry가 tokens와 state byte를
저장해도 new model build/context parameter가 이를 읽으면 unsafe할 수 있다. header/version/model digest와
load validation을 source에서 찾고 없으면 cache directory를 deployment revision별 격리한다.

model hot swap도 동일하다. process model pointer가 바뀌었는데 global cache object가 재사용되면 weights
revision namespace가 필수다. llama model-specific context/slot lifetime, manager cached processor와 cache
owner를 source에서 확인한다. restart가 가장 명확한 boundary일 수 있다.

RAM prompt cache도 disk/file corruption, partial write와 process crash를 고려한다. atomic temp+rename/header
validation이 있는지 source를 본다. corrupt entry를 skip/delete하고 cold compute로 fallback할 수 있어야
service 전체가 죽지 않는다. correctness error로 silent load하는 것보다 miss가 낫다.

## 35.6 다섯 사고를 first divergence에서 복구까지 닫는다

prefix cache 사고는 hit rate와 final text만으로 찾기 어렵다. key construction, logical match, physical
owner commit, read/write indices와 output을 순서대로 비교한다.

### 사고 1: adapter가 다른데 hash hit가 난다

base request A와 LoRA request B가 token IDs `[10,11,12,13]`을 공유한다. adapter가 attention projection을
바꾸면 K/V가 다르다. key가 tokens+parent만 포함하고 adapter/model revision이 없으면 h0가 같다.

first divergence는 tokenization이 아니라 extra identity generation이다. request key input에 adapter ID/
revision이 있는지, block hash/radix namespace/cache salt에 전달되는지 본다. lookup이 이미 separate key인데
physical block이 같다면 allocator map corruption이다.

관측은 safe model/adapter digest, hash extra digest, resulting block hash, lookup block ID/generation과
K/V fingerprint다. prompt content를 로그할 필요는 없다. base와 adapter hash가 같다는 사실이 root
evidence다.

복구는 prefix cache flag를 끄는 임시 containment 뒤 affected namespace entries와 physical blocks를
idle fence에서 폐기한다. key schema에 adapter revision을 넣고 old/new namespace version을 다르게 한다.
이미 생성된 output은 correctness incident로 분류하고 재처리 범위를 검토한다.

반증은 adapter가 Q/K/V 이전에 영향을 주지 않는 경우다. adapter 위치가 attention KV를 바꾸지 않는다면
sharing이 가능할 수도 있다. architecture/adapter target modules를 source로 확인하고 모든 adapter를
무조건 cache miss로 만들지 않는다. 안전한 보수 정책과 정확한 selective key 사이 trade-off다.

### 사고 2: multimodal placeholder가 같아 image KV가 섞인다

A/B text token에는 같은 `<image>` placeholder가 있지만 실제 image/embedding이 다르다. processor output
token만 hash하면 hit다. prompt embedding/multimodal extra key가 media digest와 position을 block별로
결합해야 한다.

vLLM hash extra generation이 multimodal/embedding data를 parent chain에 어떻게 넣는지 확인한다. SGLang
Req extra/cache salt/radix key, Transformers sharing eligibility와 llama slot/RAM state에서도 modality
isolation을 찾는다. 지원하지 않으면 caching disable이 safe answer다.

first divergence ladder는 processor outputs→extra key→block/node key→lookup→physical K fingerprint다.
text tokens부터 같다는 것은 정상이다. extra key가 다르고 hash도 다른데 hit면 hash map/group lookup,
hash가 같은데 extra가 다르면 hash construction/collision을 본다.

복구에는 processor/model revision을 key namespace에 넣는다. media content digest 비용과 privacy를 고려해
stable identity를 쓰되 raw media를 log하지 않는다. remote/external cache는 process-independent stable
hash가 필요하다.

### 사고 3: partial block COW 누락으로 A 답이 변한다

A가 block 1 `[20,21,_,_]`을 가진 상태에서 B가 이를 참조하고 `[30,31]`을 쓴다. refcount는 2로 정상,
hash hit metric도 1이다. 그러나 block table이 A/B 모두 physical 1을 가리키고 writer가 B suffix를
같은 slots에 쓴다.

first divergence는 lookup cached length가 4가 아니라 6으로 과대 계산됐는지, fork가 partial block을
포함했는지, append 전 `refcount>1 && partial` COW predicate와 new block mapping이 있었는지 순서로 본다.
cache write index에서 A/B overlap이 최초로 나타날 수 있다.

sentinel fixture는 physical block을 직접 검증한다. A block0=100, block1 prefix=200, B suffix=300 pattern을
사용하고 B write 뒤 A readback이 그대로인지 본다. runtime 실행은 이 장에서 하지 않지만 expected
block IDs/indices를 unit test로 명시한다.

복구는 active owners를 중지하고 corrupted blocks/descendants를 폐기한다. refcount만 다시 맞추는 것으로
내용 오염이 사라지지 않는다. full-only sharing invariant를 enforce하거나 correct COW를 구현하고 A/B/C
partial divergence tests를 넣는다.

### 사고 4: hit rate는 높은데 TTFT가 줄지 않는다

dashboard hit 90%인데 TTFT가 같다. hit가 request 수 비율인지 token 비율인지 먼저 확인한다. 짧은
32-token prompts의 4-token hit 90%는 compute 절약이 작다. P 16k에서 12k hit와는 다르다.

cached logical tokens와 block-aligned usable tokens가 다를 수 있다. hash granularity 4 hit라도 physical
block 16 boundary 때문에 usable 0/16일 수 있다. external connector는 hash index hit 뒤 payload load를
기다릴 수 있다. host radix lookup/sort가 GPU saved compute를 상쇄할 수도 있다.

timeline은 lookup start/end, match length, lock/ref assignment, connector load, remaining query, first model
start/end와 output commit을 둔다. first divergence가 no-cache reference와 비교해 remaining Q가 줄지
않으면 alignment/consumer 문제다. Q는 줄었지만 model start가 늦으면 lookup/load/queue다. model time은
줄고 TTFT가 같으면 tokenizer/output/other wait다.

복구는 hit metric 정의를 token-weighted usable hit와 payload-ready hit로 나눈다. block size/hash algorithm을
바꾸기 전에 saved GPU ms와 added CPU/transfer를 측정한다. 높은 hit 자체를 목표로 최적화하지 않는다.

### 사고 5: stale hash가 reused physical block을 가리킨다

hash map h0→block7이 남았지만 block7 refcount 0에서 evict/free되고 new generation에 다른 request data가
쓰였다. B lookup h0가 old entry로 block7을 받으면 token/identity가 맞아도 content가 틀리다.

first divergence는 eviction transaction이다. hash reverse map removal/event, free queue insertion,
allocation generation increment와 new hash insertion ordering을 본다. lookup은 hash와 block current hash/
generation/refcount가 일치하는지 확인해야 한다.

race timeline을 쓴다.

```text
t0 h0 -> block7 gen11 refs0 cached
t1 eviction chooses block7
t2 free/reuse block7 gen12 for X
t3 old h0 entry still visible
t4 B lookup h0 -> block7 gen12  # stale
```

복구는 cache index를 physical allocator와 함께 quiesce하고 rebuild/flush한다. hash map만 지우면 active
refcounts, block만 zero하면 stale index가 남을 수 있다. invariant checker로 every hash entry→live
block generation/hash와 reverse map을 검증한다.

### hash collision 사고를 namespace 사고와 구분한다

서로 다른 full token/extra chain이 같은 hash value를 낼 수 있다. 그러나 실제 사고 대부분은 missing
identity/stale mapping처럼 deterministic schema bug일 수 있다. collision이라고 부르기 전에 preimage
inputs를 safe digest/length로 비교한다.

cryptographic hash option은 collision risk와 cross-process stability를 개선할 수 있지만 CPU cost와
key size가 있다. Python randomized hash는 process restart/external cache 재현성이 다를 수 있다. internal
in-process lookup과 external connector key 요구를 구분한다.

secondary token/extra verification이 가능하면 collision을 detect할 수 있지만 full token storage/privacy와
cost가 있다. correctness requirement에 맞게 설계한다.

### wrong position/cache group 사고

같은 tokens가 다른 absolute position, sliding/cache group 또는 layer representation에서 같은 key가
되면 unsafe할 수 있다. parent chain length가 position을 간접 encode할 수 있지만 position offset/context
semantics와 cache group ID를 확인한다.

vLLM `BlockHashWithGroupId`, Transformers block group parent, SGLang allocator group와 llama context parameters를
native source로 본다. 36장의 hybrid/sliding 구현을 반복하지 않고 group identity가 key/lookup에 포함되는
계약만 다룬다.

### incident 공통 원장

request/model/adapter/media safe identity, token range, parent/current hash or radix path, logical match,
usable-aligned match, physical block/node/slot ID+generation, ref/lock, writer indices, content fingerprint,
output step을 한 행에 둔다.

expected chain→lookup→ownership→read/write→output 순서에서 first divergence를 찾는다. key가 틀리면 kernel을
보지 않고, key/owner가 맞고 write indices부터 겹치면 COW를 본다. output만 다르면 attention/model로
넘긴다.

### free/eviction/reuse

ref/lock decrement, zero-owner, hash/node removal, free queue와 generation reuse 순서를 적는다. active lookup과
eviction 사이 pin acquisition, failed allocation rollback을 본다.

evidence surface도 owner에 붙인다.

safe request/namespace digest, logical/usable hit, hash/path digest, physical ID+generation, refs/locks, lookup/load
duration, remaining Q, first model/output와 eviction event를 둔다. raw token/media는 기본 로그가 아니다.

tenant isolation에서 cache salt는 sharing domain을 의도적으로 좁힌다. same public prompt라도 tenant별 salt면
hit를 포기해 side-channel/cross-tenant contamination risk를 줄인다. salt rotation은 old entries namespace를
orphan으로 만들 수 있어 eviction/flush와 capacity를 고려한다.

recovery verification은 no-cache parity, A/B concurrent fixture, ref/free invariant와 stale-generation scan을
모두 통과해야 한다. hit rate가 다시 올라왔다는 것은 correctness recovery 증거가 아니다.

lookup false positive와 false negative도 센다. false positive는 semantic identity가 다른데 hit 또는 stale
physical content다. correctness incident다. false negative는 safe same prefix인데 namespace/granularity 때문에
miss다. performance 손실이다. false positive zero를 우선하고 false negative를 최적화한다.

hash collision detector가 없으면 false positive 원인을 schema/stale/collision로 완전히 분류하기 어렵다.
test build에서 token+extra secondary fingerprint를 보관하거나 deterministic fixture로 collision path를
검증할 수 있다. production privacy/메모리와 분리한다.

cache event audit은 added/removed 순서를 block generation과 묶는다. event에 hash, parent, token range와
group, block ID/generation, reason full/evict/free가 있으면 external index를 재구성할 수 있다. event loss/
reorder가 있으면 snapshot/reconciliation protocol이 필요하다. source가 제공하는 범위를 넘겨 보장하지
않는다.

cache flush operation은 new admission을 잠시 막고 active refs가 끝날 때까지 기다리거나 safe invalidation을
해야 한다. hash map만 clear해 active block tables는 계속 읽을 수 있지만 finish event/reverse map이 old
entry를 가정할 수 있다. source flush transaction과 concurrency guard를 확인한다.

flush가 prefix cache hits만 disable하고 physical KV request blocks를 유지하는지, 전체 KV를 reset하는지
구분한다. wrong-key incident containment에서 영향 범위와 in-flight request correctness가 달라진다. broad
destructive reset은 service interruption을 동반한다.

cache poisoning recovery 후 namespace를 즉시 다시 공유하지 않고 canary tenant/model에서 hit path parity를
본다. A/B/C/D, adapter/media variants와 concurrent finish/cancel을 통과한다. hit rate, TTFT보다 logits/
content and ownership invariants를 먼저 승인한다.

eviction race expected는 lookup이 block pin을 획득하기 전/후로 나눈다. before pin eviction wins→lookup miss/
retry, after pin lookup wins→eviction skips block이다. 둘 다 안전하지만 stale hit/active reuse는 실패다.

hash collision fault는 test hash function을 작은 bit width로 대체할 수 있다. secondary verification/chain
behavior가 safe miss/error를 만드는지 본다. production algorithm collision을 기다리지 않는다. source에
test hook이 없다면 unit layer에서 map semantics를 검증한다.

observability reason enum에는 lookup miss, identity mismatch, alignment truncation, payload unavailable, lock/
allocation failure와 stale rejection을 둔다. 모두 miss counter 하나면 optimization과 incident를 구분하지
못한다. hit도 local-ready, remote-pending, slot-LCP와 RAM-load를 분리한다.

eviction은 inverse transaction이다. candidate 선택, hash/node visibility 제거, active pin 확인, physical
free queue insertion과 allocator reuse 순서가 있다. visibility를 먼저 제거하면 new lookup은 miss하고
active refs가 끝날 때 free할 수 있다. physical free를 먼저 하면 stale lookup window가 생긴다.

free queue의 cached block reuse 정책이 hash map에서 block을 lazy remove한다면 allocation path가 old
hash를 제거하고 events를 내야 한다. vLLM block pool의 reverse hash set이 이 cleanup을 돕는다. 모든
free refs0 block이 즉시 content zeroed된다고 가정하지 않는다. correctness는 visibility/generation, security
zeroization은 별 requirement다.

tenant 간 memory zeroization이 필요하면 prefix sharing policy와 allocator reuse policy를 함께 감사한다.
logical key isolation이 있어도 physical block에 old bytes가 남고 kernel mask/index bug가 읽을 수 있다.
zero-on-allocation/valid length guards와 cost를 본다. 이는 hash correctness를 넘어선 defense-in-depth다.

checksum failure는 hash collision과 다르다. key identity는 맞지만 transport/storage content가 corrupted된
것이다. first divergence는 payload load/checksum이고 local hash construction을 고치지 않는다. incident
reason을 분리한다.

hot prefix가 refs0 cached state로 free queue tail/front에 놓이는 정책은 재use chance와 allocation victim
order를 바꾼다. metrics에는 entry age/frequency보다 actual evicted reusable tokens와 subsequent miss를
둔다. hit admission policy와 eviction policy를 같은 “prefix caching option”으로 합치지 않는다.

hash algorithm option을 바꿀 때 live entries migration을 고려한다. old/new hash가 다르면 dual lookup,
lazy rebuild 또는 flush가 필요하다. dual lookup은 namespace confusion/collision surface를 늘릴 수 있다.
rolling upgrade에서 version-prefix key가 가장 명확할 수 있다.

### A/B fixture의 마지막 검증: collision·tenant·COW를 재현하고 rollback한다

최종 lab에는 여섯 요청이 등장한다. R은 Red adapter revision r7, B는 Blue b3, X는 Red와 같은 r7이지만 다른
tenant, C는 R과 첫 5 token만 같고 여섯째가 다르다. D는 현재 block token은 R block0과 같지만 앞 parent가
다르다. E는 모든 identity가 R과 같지만 position 100에서 시작한다. block size 4, KV capacity 12 blocks,
tenant별 동시 요청 2, hash는 production algorithm과 강제-collision test double 두 가지를 쓴다.

정상 기대를 먼저 쓴다. R cold run은 full blocks 0–2를 계산하고 commit한다. 같은 tenant·adapter·model·position의
R2 warm run은 12-token prompt에서 logits용 마지막 boundary 제한을 적용한 native usable prefix를 얻고 output
parity를 지킨다. B는 token이 같아도 adapter revision이 달라 miss다. X는 byte equality가 가능하지만 tenant
isolation policy가 strict라 salt 때문에 miss다. C는 first block 4만 안전하게 공유하고 private partial을

쓴다. D는 parent chain 때문에 miss다. E는 position/context parent가 다르므로 miss 또는 explicit unsupported
reject다. 이 expected result를 runtime을 본 뒤 쓰면 시험이 아니다.

첫 fault는 hash function이 모든 입력에 같은 digest `00`을 반환하게 한다. 올바른 설계가 cryptographic
충돌을 secondary token/parent/identity 비교로 검출하는지, 아니면 collision probability를 위험 모델상
수용하는지 source contract를 기록한다. secondary validation이 없다면 test는 wrong hit 가능성을 드러내지만
production에서 실제 collision이 발생했다는 뜻은 아니다. 반면 B와 R의 serialized preimage가 adapter alias
누락으로 애초에 같다면 test double 없이도 deterministic wrong hit다. 두 결과를 같은 “hash collision”로
보고하지 않는다.

둘째 fault는 tenant salt 전달을 gateway에서 engine DTO 사이에서 지운다. R 뒤 X를 넣으면 token과 model
identity가 같아 hit가 난다. output은 정확할 수 있으므로 logits parity만으로 isolation 시험을 통과시키면
안 된다. expected tenant miss, observed hit, TTFT 82→39 ms의 timing channel, shared block refs와 source
producer/consumer gap을 함께 보고한다. 수정 뒤 salt가 h0에 들어가고 h1/h2로 전파돼 세 hash가 모두 달라지는지
확인한다. salt rotation generation이 active R의 lookup과 섞이지 않는지도 본다.

셋째 fault는 C partial destination allocation 직후 copy completion을 20 ms 지연하고 C를 cancel한다. block
53이 completion 전에 free/reuse되지 않아야 한다. 강제로 D에 재할당했을 때 old completion은 generation
mismatch로 무시돼야 하며, D selected layer/head digest와 no-cache logits가 같아야 한다. 시험 후 provisional
blocks, reader refs, CUDA events와 pending callbacks가 모두 0이어야 한다. 단순히 C가 response를 내지 않았다는
것은 성공 조건이 아니다.

넷째 fault는 block 41의 hash entry 제거를 지연한 채 physical ID를 새 generation에 재사용한다. lookup이
old `(hash, block41,generation6)`을 읽었지만 allocator current generation은 7이다. safe path는 stale entry를
miss 처리하고 map을 정리한 뒤 fresh compute한다. unsafe path는 valid pointer 때문에 crash 없이 wrong
answer를 낸다. 관측에는 hash-map generation, allocator generation, last write completion과 live reader count를
붙인다. stronger digest나 tenant salt는 이 lifetime 결함을 고치지 않는다.

다섯째 fault는 model hot reload다. deployment alias `qwen-prod`는 같지만 weights revision m17에서 m18로
바뀌었다. process/cache pool을 새로 만들었다면 natural namespace isolation이 있는지 확인한다. shared external
cache라면 connector key prefix 또는 payload header에서 immutable model/config digest와 KV layout schema를
검증한다. miss fallback은 안전해야 하며, m17 payload를 절반 restore한 뒤 mismatch를 발견했다면 provisional
device blocks와 refs를 rollback한다. `model_name` label이 같다는 이유로 hit를 인정하지 않는다.

여섯째는 성능 반례다. R2는 12-token hit로 prefill 3.6 ms를 아꼈지만 key 생성 0.4 ms, remote metadata 1.2
ms, payload load 2.8 ms, pin 대기 0.7 ms가 들었다. 순효과는 `3.6-(0.4+1.2+2.8+0.7)=-1.5 ms`다. hit rate는
100%인데 TTFT가 악화한다. local resident hit라면 load 2.8 ms가 없어 +1.3 ms 이득이다. dashboard는 두
경로를 같은 hit로 합치지 않고 logical hit, resident-ready hit, remote-ready hit와 fallback을 분리한다.

반대로 8,192-token system prefix가 local resident이고 prefill saved 46 ms, lookup/hash 1.7 ms, pin 0.3 ms라면
약 44 ms를 아낀다. 따라서 prefix caching이 좋거나 나쁘다는 결론은 workload prefix 길이, locality tier,
block tail loss와 queue interaction에 조건부다. COW도 평균 lost tail 7.5 token을 회수하기 위해 request당
128 KiB 이상을 copy한다면 short prefix에서는 손해일 수 있다. 수치는 특정 hardware의 보편 성능값이 아니라
측정 원장을 닫는 예다.

관측 화면은 세 층이면 충분하다. 첫 층은 correctness/isolation으로 namespace-miss expectation, wrong-hit,
stale-generation rejection, no-cache logits parity와 tenant policy violation을 둔다. 둘째는 ownership으로
matched/accepted tokens, pinned blocks, live refs, COW bytes, provisional rollback과 late completion rejection을
둔다. 셋째는 performance로 hash, lookup, transfer, pin, saved prefill, suffix compute, queue delay와 TTFT를
둔다. 세 층을 하나의 hit-rate 숫자로 접지 않는다.

회고 문서는 증상부터 쓴다. “Blue가 Red 제품명을 답했고 cache-off parity는 정상”이라는 독자 언어 다음에
“동일 mutable adapter alias가 native LoRA extra key가 되어 서로 다른 weights가 같은 preimage를 만들었다”는
기계 원인을 쓴다. 이어 pinned producer/hasher/lookup/commit consumer, affected namespace/time/request와
containment를 적는다. 마지막으로 immutable adapter revision, strict tenant salt, generation fence와 fault
tests를 재발 방지로 연결한다. 단순히 “hash key를 강화했다”라고 쓰면 어떤 차원을 추가했는지 검토할 수 없다.

선택과 rollback terminal도 명확해야 한다. 새 identity schema는 versioned namespace에서 canary로 시작한다.
old unsafe keys에 dual lookup하지 않는다. cache miss first run과 hit second run, B/X/C/D/E negative fixtures,
두 finish order와 cancel/reuse race가 통과한 뒤 cohort를 넓힌다. wrong hit, stale reject 급증, ref leak,
payload mismatch 또는 p99 lookup budget 초과가 나면 새 admission의 cache read를 끄고 active generation을
drain한다. physical blocks를 무조건 즉시 free하지 않고 live readers와 completion fence 뒤 invalidate한다.

독자가 이 장을 닫을 때 외워야 할 field 목록은 없다. 대신 순서를 설명할 수 있어야 한다. 먼저 tokenizer와
template가 model input token을 만든다. deployment와 adapter/media/session/tenant가 KV 의미와 공유 정책을
고정한다.

native source가 지원하는 identity만 parent/current/extra chain에 넣는다. longest-prefix consumer가
full·group·residency 조건을 적용한다. physical owner를 pin하고 divergent suffix에는 exclusive writer를 준다.
finish·abort·eviction은 generation fence 뒤 ownership을 돌려준다. 누락된 identity는 namespace에서 보완하거나
sharing을 금지한다. 이 인과 순서가 한 요청 trace에서 이어질 때 prefix cache는 빠른 우연이 아니라 검증 가능한
최적화가 된다.

마지막으로 실제 당직자가 사용할 15분 판단 흐름을 장면으로 복습한다. wrong answer 신고가 오면 먼저 해당
요청을 격리하고 cache-off 동일 입력을 재현한다. cache-off도 틀리면 prefix cache에서 출발하지 않는다.
cache-off는 맞고 hit만 틀리면 최종 token digest와 model deployment generation을 비교한다. 둘이 다르면
key construction 이전의 tokenizer/template/routing 문제다. 둘이 같으면 resolved adapter revision,
media/prompt-embedding identity, initial position과 tenant salt를 비교한다. 의미가 다른 field가 hash preimage나
namespace에서 사라진 최초 경계가 identity root cause 후보가 된다.

preimage까지 다르면 실제 digest collision 가능성을 보고 secondary data를 안전한 lab에서 비교한다. preimage와
digest가 모두 맞으면 lookup이 반환한 physical ID와 allocation generation, group, refcount를 본다. generation이
다르면 stale index다. 그것도 맞으면 A/B block table과 suffix write slots를 겹쳐 본다. shared full prefix
밖의 writer address가 같다면 COW 또는 private allocation 결함이다. address도 맞으면 completion fence와
runner input을 거쳐 model/backend 쪽으로 내려간다. 이 순서는 가능성 순위가 아니라 first divergence를
보존하기 위한 순서다.

운영 메모에는 민감한 prompt나 raw KV를 붙이지 않는다. safe token digest, length, block boundaries, bounded
model/adapter generation, salted tenant cohort, sampled hash prefix, physical generation과 transition timestamp면
대부분의 경계를 재구성할 수 있다. 더 깊은 byte 비교는 승인된 격리 lab에서만 수행한다. 디버깅 가능성을
높이려다 고객 prompt와 cross-tenant identity를 새로운 로그 유출면으로 만들면 안 된다.

수정 뒤에는 positive hit만 확인하지 않는다. 동일 identity R2는 기대 길이만큼 hit하고, 다른 adapter B와
strict tenant X는 miss하며, parent가 다른 D와 position이 다른 E도 miss 또는 명시적 reject여야 한다. C는
native granularity만큼 공유하되 divergent writer가 분리된다. collision test double은 safe miss 또는 문서화한
risk contract를 보여야 하고, stale generation과 late copy completion은 새 owner를 건드리지 않아야 한다.
모든 fault 뒤 refs, locks, provisional blocks와 pending events가 baseline으로 돌아와야 rollout을 다시 연다.

이렇게 보면 prefix cache key의 “순서”는 단순한 필드 배열이 아니다. 의미를 만드는 upstream 순서,
고정 source의 serialization 순서, parent chain이 identity를 후손에게 전달하는 순서, lookup과 physical pin이
그 결정을 소비하는 순서, finish가 ownership을 반납하는 순서가 겹쳐 있다. 책의 목표는 필드명을 외우게 하는
것이 아니라 이 다섯 순서 중 어디에서 계약이 끊겼는지 독자가 스스로 찾아내게 하는 것이다.

예를 들어 “adapter를 key에 넣었다”는 검토 답변을 받으면 네 번 더 묻는다. API alias인가 resolved immutable
revision인가, 모든 block에 직접 들어가는가 root parent로 전파되는가, local pool과 external connector가
같은 namespace를 쓰는가, hot reload 중 old reader와 new admission은 어느 generation으로 갈리는가. “tenant
salt를 넣었다”에도 같은 질문을 적용한다. salt가 빈 값으로 default된 요청은 sharing 허용인가 reject인가,
gateway retry가 salt를 보존하는가, rotation 때 old cache를 dual lookup하는가, metric에서 tenant 원문이
노출되지 않는가를 확인한다.

partial COW 리뷰에서는 “copy했다” 다음을 묻는다. 어느 layer와 K/V range를 몇 byte 복사했는가, destination
allocation generation은 무엇인가, 어떤 CUDA event 뒤 block table을 commit했는가, cancel 시 callback과
provisional ref를 누가 회수하는가, source reader가 끝날 때까지 eviction을 무엇이 막는가. 답이 Python
metadata 이동에서 끝나면 실제 tensor ownership은 아직 설명되지 않았다.

마지막 승인 회의에서 hit rate 그래프는 맨 뒤에 둔다. 먼저 negative fixture 여섯 개의 결과, owner/refcount
보존, no-cache logits parity와 tenant isolation을 본다. 그다음 saved prefill time에서 hash·lookup·transfer·pin·
COW·queue 비용을 뺀 critical-path 이득을 본다. correctness와 isolation을 통과하지 못한 빠른 hit는 성능
성과가 아니라 사고의 가속기다. 이 판단 순서를 지키는 것이 prefix cache를 안전하게 운영하는 가장 현실적인
최적화다.

그리고 source revision이 바뀌면 이 결론을 그대로 이식하지 않는다. producer field, tuple order, full-block
predicate, group qualifier, lookup boundary, ref/free ordering을 다시 diff한다. 이전 fixture가 새 semantics에서도
같은 이유로 통과하는지 확인해야 upgrade가 완료된다. 결과만 같고 branch가 달라졌다면 새 실패 모드를 별도로
시험한다.
## 35.7 네 구현을 동일 prefix transaction으로 비교한다

공통 기능표 대신 A/B의 logical match→physical reuse→private suffix→free를 네 번 적는다.

### vLLM A/B vertical trace

A Request가 full-block chained hash h0를 만든다. block pool이 block0 full을 hash map에 commit하고 A
partial block1은 private로 남는다. B hash generation은 same namespace에서 h0가 같고 suffix full hash는
없다. KV manager lookup은 cached tokens=4와 block0을 반환한다.

allocator는 B ref를 붙이고 fresh suffix block2를 준다. runner write indices는 B positions4,5를
block2 slots0,1에 쓴다. B finish에서 block0 ref를 줄이고 block2 complete/cache/free를 처리한다.

검산은 hash chain, returned group blocks, block0 refs, A/B block tables, B write indices와 reverse map이다.
hit event와 state mutation을 분리한다.

### SGLang A/B vertical trace

A tokens/KV indices가 radix path에 insert되고 unfinished/finished 경계에서 node/lock이 갱신된다. B
match는 prefix length4와 last node/KV indices를 준다. PrefillAdder는 B suffix2와 matched lock/allocator
demand를 계산한다.

B request-to-token mapping은 shared prefix와 private suffix indices를 이어야 한다. finish/abort에서 lock과
private mapping을 release하고 cache insert/merge한다. logical match와 page-aligned usable range를 구분한다.

### Transformers A/B vertical trace

A block0을 complete/shareable로 만들고 block1 partial을 private table에 둔다. B search는 block0을 match하고
refcount/fork로 table에 붙인다. B suffix2는 FutureRequestState query와 new block2 write indices를 얻는다.

IO가 block0 read와 block2 write를 model에 넘기고 update 뒤 occupancy/complete를 갱신한다. A/B finish에
refs 2→1→0이 된다. complete timing, fork range, tables와 write indices를 검산한다.

### llama.cpp A/B vertical trace

A slot cache가 남고 B가 same available slot을 선택하면 LCP=4다. position4 이후 A suffix KV를 제거하고
B suffix를 decode한다. concurrent A/B shared refs가 아니라 slot owner A→B 전환이다.

RAM prompt cache를 쓰면 entry selection/LCP/load와 slot restore를 별도 추적한다. task/slot availability,
kept/removed range, final prompt와 RAM entry identity를 본다.

**같은 cached length4의 다른 의미**

vLLM은 full hash blocks, SGLang은 radix logical/aligned match, Transformers는 complete shareable blocks,
llama는 slot LCP positions다. 동일 metric으로 ownership/concurrency를 추론하지 않는다.

vLLM/Transformers는 concurrent physical prefix refs를 명시적으로 가질 수 있다. SGLang은 node locks/
indices와 page variant를 본다. llama slot-local reuse는 sequential owner transition일 수 있다.

### full/partial 비교

vLLM `cache_full_blocks`, Transformers complete marking은 shareable boundary를 명시한다. SGLang radix는
token edge와 allocator page, unfinished insertion을 함께 봐야 한다. llama는 block fullness보다 LCP
position remove/keep다.

“partial COW 지원” 대신 partial logical match 노출, physical sharing, writer exclusivity와 divergence
copy/remove/allocation을 질문한다.

### eviction/reuse 비교

vLLM hash map/free queue, SGLang radix eviction/lock, Transformers refcount/free와 llama slot/RAM LRU는
victim owner가 다르다. hash stale-generation과 RAM stale-entry를 같은 field로 만들지 않는다.

### performance 비교

saved query, unique KV byte, lookup CPU, payload load, suffix work, graph shape와 TTFT를 둔다. hit ratio는
입력일 뿐이다. 동일 model/kernel이 아니면 absolute TTFT 순위로 cache algorithm을 결론내리지 않는다.

external cache event는 local state보다 늦게 전달될 수 있다. block added event와 removed event ordering,
parent hash와 token range가 downstream에 충분한지 본다. remote index가 hit를 반환해도 payload delivery/
generation validation 전 computed frontier를 commit하지 않는다. local hit와 remote-available hit metric을
분리한다.

hit correctness가 맞은 뒤 TTFT 효과를 분해한다. total prompt L, usable hit H, remaining query L-H, lookup
T_lookup, optional load T_load, saved compute T_saved라면 단순 효과는 `ΔTTFT≈T_saved-T_lookup-T_load-
schedule_effect`다. prefix hit 때문에 batch composition/graph shape가 달라 schedule_effect가 양/음일 수 있다.

high hit/no gain 사고에서 cache lookup이 CPU event loop를 block하면 다른 request TTFT까지 악화될 수 있다.
per-request lookup과 scheduler step CPU duration을 본다. external connector가 network tail을 추가하면 hit
request p99가 miss local compute보다 느릴 수도 있다. timeout/fallback semantics를 확인한다.

timing side channel도 있다. cross-tenant hit가 TTFT를 줄이면 다른 tenant prompt 존재를 추측할 수 있다.
correct KV identity만으로 보안 policy가 완성되지 않는다. isolation requirement에 따라 tenant salt 또는
sharing disable을 선택한다. 책은 legal/security 결론을 단정하지 않고 technical leakage channel을 적는다.

prefix cache observability는 네 분모를 분리한다. request hit rate는 한 token이라도 hit한 request 비율,
token hit rate는 matched tokens/prompt tokens, usable hit rate는 실제 skipped query tokens/prompt tokens,
payload-ready hit rate는 model start 전에 physical KV가 준비된 hit 비율이다. 같은 90%라도 의미가 다르다.

rolling deploy의 model revision이 같아도 hash algorithm이 Python process salt에 의존하면 cross-worker key가
달라 hit를 잃을 수 있다. correctness는 유지되지만 performance가 흔들린다. stable external hash option과
internal fast hash를 분리한다. source의 InternalBlockHash/ExternalBlockHash type 구분을 활용한다.

cryptographic hash를 켰는데 CPU가 느려졌다면 hash generation을 profiler range로 분리한다. prompt 길이,
block count와 extra serialization byte에 대한 slope를 본다. TTFT 증가가 lookup이 아니라 tokenizer/chat
template일 수 있으므로 first divergence를 유지한다.

TTFT benchmark는 cold miss, warm same-request, warm concurrent share, RAM/remote restore를 나눈다. cache-on
first run을 hit 성능으로 포함하지 않는다. warmup/model graph effects를 분리하고 same prompt reuse가
실제로 expected key를 사용했는지 event로 확인한다.

concurrent share benchmark에서 A/B start timing을 고정한다. A block0 complete commit 전 B lookup이면 miss가
정상일 수 있다. commit 후 B lookup에서 hit를 기대한다. race window를 성능 variability로 오판하지 않는다.

metric cardinality는 hash/tenant/request raw label을 피한다. aggregate reason/count와 sampled trace safe digest를
쓴다. adapter/model revisions는 bounded configured label 또는 digest sampling으로 관리한다. privacy와
Prometheus cardinality를 함께 고려한다.

시간축 t0에 A block0 allocate refs1, t1 device write, t2 host progress, t3 complete/hash insert라고 하자.
B가 t1.5 lookup하면 miss가 정상이고 fresh compute할 수 있다. B가 t3.5 lookup하면 hit/ref increment를
기대한다. t2와 t3 사이 content는 ready해도 index에 아직 없으므로 performance miss이지 correctness
failure가 아니다. hit-before-t1 completion이 위험하다.

cache size를 늘려 hit율이 올랐지만 TTFT가 그대로일 수 있다. workload prompts가 짧거나 GPU prefill이
이미 작은 비중, lookup CPU/lock contention이 늘거나 output/network가 지배할 수 있다. unique KV saved와
critical-path saved time을 본다. capacity만으로 결과를 약속하지 않는다.

dual lookup을 한다면 old hit payload를 new identity schema로 secondary validate해야 한다. schema bug를
고치면서 old unsafe key를 그대로 fallback하면 복구가 무효다. performance continuity보다 correctness를
우선한다.

응답 오염 범위를 평가할 때 affected namespace/hash descendants와 time window, physical block generations,
requests that referenced blocks를 사용한다. hash chain 때문에 wrong parent entry의 descendants도 unsafe할
수 있다. metrics/event retention이 충분해야 한다.

### 각 transition의 source reference를 producer에서 physical reader까지 잇는다

캐시 문제를 읽을 때 흔한 실수는 hash 함수 한 개를 찾고 조사를 끝내는 것이다. hash는 후보 이름일 뿐이다.
그 이름을 누가 만들고, 누가 longest prefix로 해석하고, 어떤 physical owner를 pin하며, runner가 어느
address를 읽는지까지 이어져야 실제 공유가 된다. 여기서는 앞의 Blue/Red 사건을 breakpoint 다섯 개로
다시 걷는다.

첫 breakpoint는 tokenizer와 template가 끝난 직후다. user text가 같아도 template revision에 따라 role
marker, BOS/EOS, whitespace token이 달라질 수 있다. 원문 문자열 대신 최종 token count, special-token
positions와 safe token digest를 기록한다. 두 요청의 token digest가 다르면 cache key divergence가 정상이고
adapter를 볼 필요가 없다. digest가 같다면 tokenizer/template revision이 직접 KV 차이를 만들었는지,
아니면 단지 같은 model input으로 수렴했는지 구분한다. revision 문자열이 다르다는 이유만으로 wrong hit라고
부르지 않는다.

둘째 breakpoint는 request identity가 engine DTO로 고정되는 순간이다. model deployment generation,
resolved adapter content revision, multimodal processor/feature identifier, prompt embedding digest, tenant
sharing salt와 initial cache position을 적는다. API가 받은 adapter alias가 아니라 loader가 실제로 resolve한
immutable revision이어야 한다. Blue와 Red가 둘 다 alias `support`여도 resolved digest가 `ab31`과 `f902`라면
identity는 달라야 한다. DTO가 alias만 보존하면 hash utility에 도착하기 전에 정보가 이미 사라졌다. hasher에
필드를 추가하는 수정만으로는 producer loss를 복구하지 못한다.

셋째 breakpoint는 vLLM `generate_block_hash_extra_keys`와 `hash_block_tokens`다. block 0에서 native extra
tuple을 손으로 쓴다.

```text
extra0 = (lora_name,
          (media_identifier, relative_offset)...,
          cache_salt,
          prompt_embed_sha256...)
h0 = H(NONE_HASH, token_ids[0:4], extra0)
h1 = H(h0,        token_ids[4:8], extra1)
h2 = H(h1,        token_ids[8:12], extra2)
```

이 배열은 설명용 schema proposal이 아니라 해당 고정 revision의 결합 순서를 요약한 것이다. cache salt는
첫 block에만 직접 들어가도 h0가 h1과 h2의 parent이므로 후손에 전파된다. LoRA name은 각 block extra 생성에
포함된다. media identifier는 그 media range와 겹치는 block에 relative offset과 함께 들어가고 이후 차이는
parent chain으로 전달된다. prompt embedding은 해당 block slice의 tensor bytes를 SHA-256한 digest다. model,
tokenizer, template, arbitrary position field가 이 native tuple에 명시적으로 있다는 주장은 하지 않는다.

넷째 breakpoint는 lookup consumer다. `get_computed_blocks`는 prefix read가 enabled인지 확인하고, all-token
hit에서도 logits를 얻기 위해 마지막 token을 다시 계산하도록 maximum hit length를 prompt length minus one으로
제한한다. coordinator는 request block hashes와 group availability에서 longest hit를 반환한다. 여기서
`h0,h1,h2 존재`와 `physical blocks ready/pinnable`은 다를 수 있다. multi-group 중 하나가 h1까지만 가능하면
computed frontier는 더 짧은 공통 경계로 내려간다.

lookup event에 requested hash count, matched hash count,
accepted blocks, group miss reason과 ref transition을 남긴다.

다섯째 breakpoint는 commit과 reader mapping이다. `cache_full_blocks`는 request hashes를 physical group block
size에 맞춰 resolve하고 group ID를 hash와 결합해 map에 넣는다. partial→full promotion에서 기존 shorter
metadata가 있으면 제거하고 새 boundary hash를 삽입한다. scheduler가 computed frontier를 올린 뒤 runner의
block table과 slot mapping이 실제 cached block을 가리켜야 한다. source audit은 `_insert_block_hash` 호출에서

끝나지 않고 allocator 반환, scheduler output, worker input과 attention page address consumer까지 요청 ID와
generation을 연결한다. 이 마지막 연결이 없으면 index hit는 맞아도 runner가 fresh block을 읽거나 그 반대일
수 있다.

SGLang에서는 같은 질문을 radix 언어로 번역한다. token path의 root-to-leaf match가 logical identity를
표현하고, node의 KV indices와 lock reference가 physical residency/lifetime을 표현한다.

adapter/media/model
identity가 radix token path 바깥 namespace 또는 sharing-disable predicate에서 보장되는지 확인한다. root가
tenant별로 분리되지 않았는데 cross-tenant sharing 정책이 금지라면 token path가 정확해도 설계 gap이다.
unfinished request insertion은 matchable token 길이와 complete physical KV 경계를 혼동하기 쉬운 곳이다.
PrefillAdder가 받아들이는 matched prefix와 allocator가 실제로 reuse하는 indices를 나란히 본다.

Transformers continuous path에서는 request state의 cache blocks, complete predicate, fork/refcount와
write indices를 잇는다. hash 또는 parent relation이 같은 candidate를 찾아도 partial block을 fork한 범위와
새 suffix allocation이 writer-exclusive인지 확인한다. CPU offload가 끼면 key hit, host payload 존재,
device restore 완료가 서로 다른 frontier다. model run 전에 restore-ready fence가 없으면 logical hit는 맞지만
stale 또는 incomplete payload를 읽는다. cancellation은 host와 device refs를 모두 기준값으로 돌려야 한다.

llama.cpp에서는 global hash block map을 억지로 찾지 않는다. task가 available slot을 고르고 cached tokens와
new prompt의 LCP를 계산한 뒤 어느 position부터 제거하고 다시 decode하는지 본다. A가 끝난 slot을 B가
재사용하는 sequential ownership과 A/B가 동시에 같은 physical prefix를 읽는 concurrent ownership은 다르다.
RAM prompt cache가 있으면 entry identity와 slot restore generation을 추가한다. tenant 정책이 slot pool
분리로 구현되는지, cache key salt가 있는지, 아니면 application에서 sharing을 막는지 native 경계를 적는다.

소스 산책의 종료 조건은 다섯 문장이다. “이 token digest는 어디서 만들어졌다. 이 immutable model/adapter와
policy namespace가 여기서 고정됐다. 이 함수가 parent/current/extra를 이 순서로 hash했다. 이 consumer가
longest usable prefix와 physical owner를 pin했다. 이 runner mapping이 해당 generation을 읽고 finish가
reference를 돌려줬다.” 하나라도 source와 runtime trace로 채우지 못하면 cache hit의 correctness는 아직
증명되지 않았다.

**prefix reuse 판정 분기.** hash가 다르면 normalization·token IDs·model/adapter salt부터 본다. hash는 같지만 reuse가 거부되면 partial-block policy와 committed length를 확인한다. reuse는 됐는데 첫 새 token logits가 다르면 block generation, position과 copy-on-write를 조사한다. latency만 좋아지고 correctness control이 실패하면 sharing을 끄고 원인을 닫을 때까지 배포하지 않는다.

## 35.8 A/B/C/D를 새 버전에 다시 통과시키는 선택 기준

이 장의 source는 vLLM v0.27.1 `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang
v0.5.18 `71de97b264b04dcd514cf904003028aefe9775c8`, Transformers v5.15.1
`550d7b3834670483a4df436541272c055dc364bf`, llama.cpp v0.2.0 계열
`bb4caa7540188872173c44d161602d9271386413`에 고정했다. 다음 기록은 runtime 측정 결과를 가장하지
않는다. source에서 확인한 mutation과 실제 배포에서 확인할 관측 계획을 분리한다.

새 version을 읽을 때는 긴 체크리스트부터 펼치지 않는다. A/B가 같은 canonical identity를 만들고,
공통 full block만 함께 읽으며, 서로 다른 suffix는 private block에 쓰고, 마지막 owner가 사라진 뒤에만
eviction 후보가 되는 한 편의 시간축을 먼저 그린다. 어느 구현이든 이 네 장면을 설명한 뒤 stack별
자료구조 차이를 그 옆에 붙인다.

identity schema부터 시작한다.

model/weights revision, tokenizer/template, adapter revision, media/embedding digest, position/cache group,
tenant salt 중 KV를 바꾸는 항을 적는다. 각 항이 hash/radix/block/slot key 또는 sharing-disable predicate로
이어지는 pinned source를 붙인다.

없는 항은 불필요 근거 또는 gap이다. token equality로 pass하지 않는다.

granularity schema는 별도다.

hash chunk, allocator block/page, kernel page와 logical match granularity를 적는다. divisibility/alignment
validation, conversion view와 usable length 식을 쓴다. A/B logical match4가 physical match4/0 중 무엇인지
계산한다.

key chain과 lookup을 이어 쓴다.

parent/current inputs, extra keys와 algorithm stability를 적는다. external cache면 cross-process key/version을
쓴다. lookup owner, longest-prefix stop, group condition, lock/ref acquisition과 miss를 잇는다.

same current block/different parent, same tokens/different adapter, same placeholder/different image의 expected
different key를 fixture로 둔다.

upgrade diff에서는 다음을 본다.

old/new key schema version, algorithm/granularity, complete predicate, longest rule, COW/fork range, ref/free
ordering과 metric definition을 diff한다. schema가 바뀌면 old entry namespace migration/flush를 정한다.

선택 기준은 workload에서 나온다.

concurrent repeated prefixes가 중요하면 full-block hash/refcount/radix ownership과 distributed cache가 기준이다.
local sequential reuse면 slot LCP 단순성이 적합할 수 있다. adapter/multimodal isolation은 must-pass다.

hit rate보다 saved GPU work, KV byte, lookup/load overhead와 correctness evidence를 weight한다. RAM/remote
tier는 capacity 대신 transfer/version risk를 추가한다.

이 기록을 실제 A/B/C/D 확장 fixture에 적용해 보자. C=`[10,11,12,13,20,99]`,
D=`[7,8,9,0,10,11,12,13]`이다. C는 A와 token 5개가 같지만 first full block 4만 global shareable일
수 있다. D의 second block token은 A first block과 같지만 parent가 다르므로 chained hash가 달라야 한다.

A full h0, C first h0가 같고 C second partial은 fresh physical block이어야 한다. token-granular radix는
C와 A가 partial prefix token20을 logical match할 수 있지만 page/block allocator가 shared index를
안전하게 표현하는지 확인한다. llama slot LCP는 5 positions를 keep하고 position5 이후를 remove할 수
있다. 같은 fixture에서 usable cached length가 stack별로 4 또는 5일 수 있으며 어느 쪽도 source contract에
맞으면 정상이다.

D는 current block-only hash bug를 잡는다. D block1 `[10,11,12,13]`만 보면 A block0과 같지만 h_D1은
`H(h_D0,[10,11,12,13],extra)`라 h_A0와 달라야 한다. radix path도 root부터 `[7,8,9,0]`을 거쳐야 하므로
A root edge를 재사용하지 않는다. llama LCP(A,D)=0이므로 slot prefix를 keep하지 않는다.

이 네 fixture expected를 표로 쓰면 다음과 같다.

```text
       logical LCP(A,X)   full-block reusable   current-block collision test
B             4                  4              same h0 only
C             5             4 or native-safe5   partial divergence
D             0                  0              parent chain must differ
```

adapter B2와 image B3도 만든다. tokens는 B와 같지만 extra identity만 다르다. B2/B3의 logical token LCP는
6이어도 cache semantic reusable length는 0이어야 할 수 있다. share-disable predicate가 더 보수적으로
전체 miss를 만드는 것도 허용한다. expected hash/path namespace가 다름을 assert한다.

template/tokenizer revision도 key schema에 고려한다. 같은 user text가 revision 변경 뒤 우연히 같은 token
IDs를 만들더라도 special-token/chat-template semantics가 model KV에 영향을 주지 않는다면 sharing 가능할
수 있다. 그러나 model revision/weights가 다르면 절대 안 된다. 어떤 revision을 namespace에 넣을지는
KV equality를 실제로 바꾸는 state를 기준으로 한다. 모든 software version을 넣으면 안전하지만 hit를
과도하게 잃고 migration이 잦다.

position offset도 fixture로 만든다. A tokens가 position0부터, E same tokens가 session continuation
position100부터 시작한다. RoPE/position encoding 때문에 K가 다를 수 있다. parent chain length가
position을 암묵적으로 고정하는 request-local prefix cache에서는 E가 same root prefix로 조회되지 않는지,
explicit cache position namespace가 필요한지 본다. slot context shift도 native semantics를 따른다.

prefix cache disabled parity는 baseline이다. disabled path가 allocator/block layout까지 달라 graph/kernel
변수도 바꾼다면 pure cache effect가 아니다. selected backend/shape와 no-cache path를 기록한다. cache
lookup만 disabled하는 test hook이 있으면 더 좋은 differential이다.

cache-on miss first run도 baseline과 같아야 한다. miss path가 hash generation/extra metadata를 추가해도
model inputs/KV writes는 동일해야 한다. first divergence가 miss run부터 있으면 sharing hit가 아니라
instrumentation/layout bug다.

hit second run은 prefill queries가 줄지만 decode logits가 baseline과 같아야 한다. exact floating order/
backend가 달라 허용 오차가 있으면 greedy/logits tolerance를 명시한다. sampling text random difference를
corruption으로 오판하지 않는다.

A/B concurrent test는 B hit 뒤 A를 계속 decode해 A output이 변하지 않는지 본다. B finish/free 뒤 A가
shared prefix를 계속 읽을 수 있어야 한다. A finish first/B survives도 검증한다. refs와 ownership의 두
finish orders가 핵심이다.

C partial test는 C common token20을 어느 granularity에서 재사용하는지 expected를 native stack별로 둔다.
vLLM full-only이면 cached4, radix/slot may keep5, Transformers complete-block이면4다. different result를
bug로 만들지 않는다. 각 physical writer mapping이 safe한지가 기준이다.

D parent-chain test는 current block match를 절대 hit로 만들지 않는다. external cache index에서도 parent/
prefix fingerprint가 key에 포함돼야 한다. D hit가 발생하면 key construction first divergence다. allocator를
더 볼 필요가 없다.

E same tokens at position100 test는 position semantics에 따라 miss해야 한다. session prefix parent가 다르기
때문에 natural miss인지 explicit position key인지 기록한다. absolute-position independent model이라도
cache layout/causal context가 같아야 reuse 가능하다.

F different model weights test는 unconditional miss다. model name만 같고 revision differs case를 포함한다.
content-addressed weight digest 또는 immutable deployment namespace를 쓴다. mutable alias `latest`를 key로
쓰지 않는다.

fault test 결과를 한 evidence table에 모은다.

```text
fixture A/B: expected hit4, distinct suffix blocks, parity pass
fixture A/C: native hit4 or safe5, no shared divergent writer
fixture A/D: expected hit0, parent-chain pass
adapter/media/model variants: expected namespace miss
evict/reuse: old hash never resolves new generation
finish orders: refs/locks return to baseline
```

이 표는 반복 체크리스트가 아니라 한 key/ownership contract의 executable examples다. 새 version에서
expected가 바뀌면 source semantic diff와 migration을 설명해야 한다.

최종 source review에서 comments/doc과 mutation을 대조한다. “full blocks only” comment가 있는데 code가
partial promotion을 허용하면 exact condition을 읽는다. tests가 comment를 고정하는지 본다. documentation
문장만으로 invariant를 선언하지 않는다.

다른 cache library/새 framework에도 같은 fixture를 적용한다. hash map, radix, trie, slot snapshot 등
자료구조 이름보다 identity inputs, match granularity, physical owner, private suffix와 reuse fence를 찾는다.
이 다섯 좌표가 있으면 구현 언어가 달라도 비교할 수 있다.

책을 읽은 뒤 독자가 첫 source breakpoint를 어디에 둘지도 명확해야 한다. B request hash/key가 완성되는
함수, longest match가 cached length를 반환하는 함수, physical ref/lock을 획득하는 함수, suffix write
indices를 만드는 함수와 finish decrement/free 함수다. 다섯 경계의 input/output을 같은 request trace로
잇는다.

key construction부터 다르면 lookup 구현을 디버깅하지 않는다. key/match는 맞지만 physical table부터
다르면 allocator/fork/lock이다. table까지 맞고 write overlap이면 COW/index다. write/readback이 맞고
logits부터 다르면 cache sharing 범위 밖 model/backend를 본다. first divergence가 investigation 순서를
결정한다.

성능에서도 같은 규칙을 쓴다. key generation 시간이 늘면 hash/extra serialization, lookup이 늘면 map/
radix/remote, match는 빠른데 model start가 늦으면 allocation/scheduling, model Q가 안 줄면 usable alignment,
model은 빨라졌는데 TTFT가 같으면 output/other critical path다. hit rate로 원인을 대신하지 않는다.

option 변경은 하나씩 한다. block/hash granularity, algorithm, prefix-enable, RAM/remote tier, cache capacity와
tenant salt를 동시에 바꾸면 hit와 correctness 원인을 분리할 수 없다. 먼저 key/ownership parity, 다음
granularity/performance, 마지막 capacity/tier를 검증한다.

cache capacity를 늘리는 변화는 eviction frequency와 hit를 바꾸지만 key/COW correctness를 고쳐 주지
않는다. collision/stale incident에서 capacity 확대는 stale entry lifetime을 늘릴 수도 있다. root cause
수정 전 tuning을 중단한다.

hash를 stronger algorithm으로 바꾸는 것도 missing adapter/media identity를 해결하지 않는다. 같은 wrong
preimage는 더 강한 hash에서도 같다. schema completeness와 collision resistance를 분리한다. stale physical
generation도 algorithm과 무관하다.

full-only sharing에서 partial hit를 더 얻고 싶다면 COW 구현 전에 benefit upper bound를 계산한다. block
size16, 평균 lost tail7.5, prefill token cost와 request rate로 saved GPU time을 추정한다. metadata/copy/
test complexity와 correctness risk보다 작은 이득이면 보수 invariant를 유지한다.

partial COW를 구현한다면 copy byte는 layers×KV heads×head dim×filled slots×K/V×dtype다. block metadata
복사만이 아니라 모든 layer KV payload를 copy하거나 backend가 logical indirection을 제공해야 한다.
CUDA stream ordering과 destination allocation failure rollback을 설계한다. Python list copy로 설명을
끝내지 않는다.

copy 대신 split ownership을 지원하는 allocator도 있을 수 있다. 한 physical page의 slot ranges를 다른
request가 공유하면 refcount가 block이 아니라 subrange/validity bitmap 의미를 가져야 하고 kernel write
mask가 겹치지 않아야 한다. source가 이를 명시하지 않으면 block-granular exclusivity를 가정한다.

RAM state load는 COW 대신 whole context clone 비용이 있을 수 있다. slot-local LCP remove/recompute는
concurrent sharing 위험을 피하지만 같은 prefix KV를 여러 slot에 duplicate한다. memory와 simplicity의
trade-off다. global shared-block hit율과 직접 비교하지 않는다.

최종 선택 문서는 조건부다. “우리 workload는 8k system prefix가 70% request에서 반복되고 adapter/media
namespace가 분리되며 concurrent sharing evidence가 있으므로 full-block cache를 사용한다”처럼 쓴다.
“prefix cache가 빠르다”는 일반 문장을 쓰지 않는다.

운영자가 cache를 끌 수 있는 kill switch와 namespace selective invalidation을 준비하되 이것을 correctness
대체물로 삼지 않는다. kill switch 적용 범위, active request behavior와 restart 필요성을 문서화한다.
incident 때 빠른 containment가 가능해야 한다.

마지막으로 owner 보존식을 한 줄로 쓴다. 모든 shareable physical state는 유효한 identity key와 current
generation을 가지며, live readers/locks와 cache owner convention의 합이 lifetime을 설명하고, divergent
writer는 exclusive destination을 가진다. 이 식이 깨지면 hit를 허용하지 않는다.

현장에서 이 보존식을 검증할 때는 단일 hit ratio 그래프에서 출발하지 않는다. 같은 요청에 request ID,
normalized token fingerprint, adapter identity, multimodal digest, matched logical length, accepted physical
length, 마지막으로 pin한 block 또는 radix node, COW 발생 위치, allocation generation을 연결한 상관관계
기록을 만든다. 원문 prompt 전체나 이미지 바이트를 로그에 남기라는 뜻은 아니다. 개인정보를 노출하지
않는 안정적인 fingerprint와 길이, namespace만으로도 두 요청이 어느 판단 경계까지 같았는지 재구성할 수
있다.

첫 요청은 miss이고 두 번째 요청은 hit인 A/A 재생, 마지막 token만 다른 A/B 재생, adapter만 다른
A/B 재생을 나란히 놓으면 key construction, logical match, physical acceptance 중 최초로 갈라지는 지점이
보인다. 이 최초 불일치가 디버깅의 시작점이다. 최종 출력이 달라진 뒤 모든 레이어를 훑는 방식은 너무
늦고, prefix hit가 기록됐다는 이유만으로 scheduler나 kernel을 의심하는 방식은 너무 이르다.

성능 문제도 동일한 경계를 사용하되 질문을 바꾼다. logical matched tokens가 큰데 accepted tokens가 작으면
block 정렬, 미완성 node 정책, residency 또는 allocator 세대 검사를 먼저 한다. accepted tokens도 큰데
TTFT가 줄지 않으면 cache lookup 시간, 원격 metadata 왕복, KV 이동, pin 대기, prefix 이후의 queue delay를
분리한다.

TTFT는 줄었지만 처리량이 떨어지면 공유 prefix가 오래 pin되어 free capacity를 압박하는지,
COW가 작은 suffix마다 allocation과 copy를 유발하는지, scheduler가 cache-aware locality를 좇다가 batch
형성을 망치는지 본다. 이때 hit token 수, 실제 생략한 prefill 연산량, lookup 및 transfer 시간, pinned
bytes, COW bytes, eviction 후 재계산량을 한 요청의 타임라인에 겹쳐야 한다. 서로 다른 집계 창의 비율을
나란히 놓으면 원인과 결과가 쉽게 뒤집힌다.

회귀 시험은 성공 경로만 확인해서는 안 된다. hash 충돌을 강제로 주입하고, 같은 token에 서로 다른
adapter와 media identity를 붙이며, partial block의 공유 직후 두 요청을 분기시킨다. reader가 살아 있는
동안 eviction을 요청하고, free된 physical ID를 즉시 재할당하며, remote lookup 응답이 돌아오기 전에
namespace generation을 올린다. 각 시험은 단순히 crash가 없음을 통과 조건으로 삼지 않는다. 잘못된 hit가
거절되는지, 거절 뒤 recompute가 정확한지, pin과 refcount가 기준값으로 복귀하는지, stale completion이 새
owner를 해제하지 않는지까지 확인한다. fault injection 뒤 메모리가 남는다면 안전한 실패처럼 보여도 장기
운영에서는 용량 고갈로 바뀐다. 반대로 정리 과정이 성급하면 조용한 use-after-free가 된다.

구현을 바꿀 때는 옵션을 한 번에 하나만 움직인다. block size를 바꾸면 hash 및 match granularity와 내부
단편화가 동시에 변한다. partial COW를 켜면 accepted prefix와 copy traffic, ownership state가 함께 변한다.
강한 digest를 추가하면 충돌 확률은 낮아지지만 누락된 adapter 차원은 복구되지 않는다. cache capacity를
늘리면 eviction 빈도는 줄어도 stale generation 검사는 생기지 않는다. 각 변경의 예상 인과를
“어떤 키가 달라지고, 어떤 상태 전이가 추가되며, 어느 metric이 어느 방향으로 움직이는가”라는 문장으로
먼저 적는다. 관측값이 그 문장과 어긋나면 최적화 효과를 선언하지 말고 숨은 경로 또는 측정 오류를 찾는다.

최종 운영 승인에는 세 종류의 증거가 필요하다. source evidence는 실제 key, match, lock/refcount, COW,
free 경로의 함수와 고정된 revision을 가리킨다. invariant evidence는 A/B fixture와 경쟁 조건 시험이 owner
보존식을 지켰음을 보인다. performance evidence는 동일 workload에서 계산 생략 이득이 lookup, 이동,
pinning, 복사 비용보다 컸음을 보인다. 셋 중 하나라도 없으면 그 결과는 흥미로운 관측일 수는 있어도
재현 가능한 설계 결론은 아니다. 이 구분을 지키면 독자는 코드 변경 뒤 어디를 다시 읽고 어떤 실험을
추가해야 하는지 스스로 결정할 수 있다.

이 사건에서 닫은 invariant를 다음 주소 문제로 넘겨 보자.

prefix cache는 identity equality, longest usable prefix, physical owner pin, private suffix write와 free/reuse가
모두 닫혀야 한다. full-only sharing은 granularity를 잃지만 COW 복잡성을 줄인다. partial sharing은 더
많은 hit와 더 강한 fencing을 요구한다.

36장에는 hybrid/sliding representation이 들어간다. 이 장은 address translation을 반복하지 않고 group
identity가 key/ownership에 포함돼야 한다는 계약만 넘긴다. 독자는 A/B fixture로 새 implementation을
검산할 수 있어야 한다.
