# 51장. 한 GPU에서 여러 LoRA를 안전하게 섞는 법

어댑터 A와 B를 각각 단독으로 실행하면 답이 정확하다. 두 요청을 같은 서버에 순서대로 보내도 대개 정확하다. 그런데 traffic이 늘어 prefill과 decode가 한 batch에 섞이고, 방금 끝난 A의 자리에 B가 적재되는 순간부터 아주 드물게 B 요청이 A의 말투를 보인다. 서버는 오류를 내지 않는다. tensor shape도 맞고 CUDA kernel도 끝까지 실행된다. 이런 사고는 “LoRA 품질이 좋지 않다”거나 “GPU 수치 오차가 있다”는 말로 설명할 수 없다. 요청의 identity가 어딘가에서 다른 요청의 weight와 연결된 것이다.

이 장은 그 연결을 처음부터 끝까지 따라간다. API에 들어온 adapter 이름은 artifact path로 해석되고, artifact는 base model과 target module에 맞는 A/B tensor로 검증된다. tensor는 TP rank별로 나뉘어 resident slot에 들어간다. scheduler는 서로 다른 adapter를 요구하는 request를 한 batch로 만들고, flattened token마다 어느 slot을 사용할지 mapping을 만든다. low-rank kernel은 mapping에 따라 delta를 계산해 base output에 더한다. 끝난 요청은 mapping과 cache reference를 놓고, adapter는 drain·evict·free 단계를 거친다.

중요한 질문은 “LoRA가 로드됐는가?”가 아니다. 이름, artifact hash, internal id, resident slot, token mapping, cache identity가 같은 adapter generation을 가리키는가가 질문이다. slot 번호는 주소일 뿐 정체성이 아니다. LRU가 slot 2에서 A를 내리고 B를 넣으면 같은 숫자의 의미가 달라진다. stale mapping이 여전히 2를 가리키면 shape가 완벽한 wrong answer가 만들어진다.

여기서는 작은 Llama fixture를 사용한다. base linear weight는 고정하고 rank 2인 adapter A와 B의 값은 서로 구별되는 상수로 둔다. batch에는 A token 둘, B token 하나, base-only token 하나를 넣는다. 다음 iteration에서는 request가 끝나고 순서가 바뀌며 slot도 재사용된다. 이 fixture 하나로 load, TP slice, token permutation, kernel scatter, cache key, unload를 차례로 심사한다.

## 51.1 hot-reload 중 같은 이름의 tenant adapter가 충돌한 사건

대표 사건은 tenant A의 `support` adapter가 slot 3에서 실행되는 동안 tenant B가 같은 public name의 새 artifact를 hot-reload한 경우다. 요청 배열의 이름은 같았지만 artifact digest와 generation이 달랐고, batch compaction 뒤 일부 token row가 새 slot을 읽었다. 이 장은 이 한 사건을 `load 준비 → generation이 붙은 slot publish → request-to-token batch mapping → drain 뒤 unload` 순서로 좁힌다.

운영자가 보는 이름을 `customer-a-tone`이라고 하자. API gateway는 이 문자열을 model alias로 받을 수 있다. registry는 alias를 `/models/adapters/a`라는 path로 바꾼다. loader는 그 디렉터리에서 config와 weight를 읽고 engine은 integer id 17을 부여한다. GPU manager는 현재 빈 slot 2에 tensor를 넣는다. kernel은 path나 이름을 모르고 token mapping의 값 2만 본다.

이 다섯 값은 한동안 같은 것을 가리키지만 역할은 다르다. public name은 routing key다. path는 위치다. artifact hash는 내용이다. internal id는 engine의 논리 key다. slot id는 현재의 물리 위치다. path의 파일이 교체되어도 이름과 path는 그대로일 수 있다. process restart 뒤 internal id가 달라질 수 있다. eviction 뒤 slot 2의 owner도 달라진다.

그래서 trace에는 tuple을 남긴다. base revision, adapter public name, adapter file revision 또는 hash, adapter config hash, target module set, rank와 scaling convention, internal id, slot id와 slot generation, tenant scope다. 모든 log line에 긴 path를 붙이라는 뜻은 아니다. secret이 될 수 있는 path는 노출하지 않고 bounded stable identifier를 사용한다. 하지만 내부적으로 서로 다른 generation을 합쳐 세지 않아야 한다.

### public identity에서 artifact identity로

요청의 이름을 path로 바꾸는 순간에는 authorization도 함께 닫혀야 한다. 사용자가 임의 local path나 hub repository를 넣을 수 있다면 tenant boundary를 건너거나 실행 환경이 예상하지 않은 artifact를 읽을 수 있다. registry는 허용된 public name을 immutable revision으로 해석하고, 요청 trace에 resolution 결과를 기록해야 한다.

이 단계의 실패는 GPU failure가 아니다. unknown name, forbidden tenant, revision mismatch, hash mismatch는 weight allocation 전에 거절하는 것이 좋다. 일부 worker가 이미 파일을 읽은 뒤 authorization이 실패하면 불필요한 자원 사용과 cleanup 부담이 생긴다. ingress에서 가능한 validation과 worker에서만 가능한 tensor validation을 구분한다.

vLLM의 [`LoRARequest`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/request.py#L8-L71)는 name, integer id와 path를 운반하며 equality와 hash의 기준도 정의한다. 이 기준을 읽지 않고 Python object가 다르니 서로 다른 adapter라고 생각하면 안 된다. 반대로 name만 같으면 내용도 같다고 확대해서도 안 된다. 현재 class가 제공하는 identity contract와 배포 registry가 보장해야 할 content immutability는 별도다.

### slot에는 generation이 필요하다

GPU slot은 제한된 A/B buffer의 index다. adapter A가 slot 2에 들어간 뒤 요청이 끝나고 B가 같은 slot을 재사용할 수 있다. 이때 `(slot=2,generation=41)`과 `(slot=2,generation=42)`는 다른 resource다. code가 명시적 generation counter를 갖지 않더라도 loader transaction, active reference와 mapping refresh 순서로 같은 불변식을 만들어야 한다.

관찰할 값은 세 가지다. slot owner가 바뀐 시각, mapping tensor가 갱신된 시각, 이전 graph나 running batch가 마지막으로 그 slot을 참조한 시각이다. 안전한 순서는 이전 reference drain, slot inactive 전환, 새 tensor copy 완료, owner commit, 새 mapping publish다. copy 중인 slot을 먼저 publish하거나 owner를 바꾼 뒤 이전 graph를 replay하면 부분 tensor 또는 새 tensor를 잘못 소비할 수 있다.

이 절에서 확정한 것은 이름이 storage address가 아니라는 사실이다. 아직 A/B tensor가 base layer의 어느 좌표에 더해지는지는 확인하지 않았다. 다음 절에서 작은 행렬을 손으로 계산해 identity가 지켜졌을 때 기대할 delta를 만든다.

## 51.2 LoRA 한 좌표를 손으로 계산한다

Base linear를 `y=xWᵀ`라고 하자. `W`의 shape는 `[out,in]`이다. rank `r`인 LoRA의 한 convention에서 A는 `[r,in]`, B는 `[out,r]`이며 `Δy=(xAᵀ)Bᵀ×s`다. scale `s`는 흔히 `alpha/r`이지만 모든 variant에 무조건 같은 식을 적용하지 않는다. 실제 config와 consumer가 rank-stabilized scaling이나 다른 convention을 쓰는지 확인한다.

Fixture는 `in=3`, `out=2`, `r=2`, input `x=[1,2,1]`이다. Adapter A의 `A_A=[[1,0,0],[0,1,0]]`, `B_A=[[1,0],[0,2]]`, scale 1이면 shrink 결과는 `[1,2]`, expand 결과는 `[1,4]`다. Adapter B의 `A_B=[[0,0,1],[1,0,0]]`, `B_B=[[3,0],[0,1]]`이면 shrink는 `[1,1]`, expand는 `[3,1]`이다. base-only token의 delta는 `[0,0]`이다.

Batch token 네 개가 `[A,A,B,base]`라면 기대 delta rows는 `[[1,4],[1,4],[3,1],[0,0]]`다. kernel이 adapter별로 token을 정렬해 `[base,A,A,B]` 순서로 계산할 수는 있다. 그러나 최종 scatter는 원 row 순서를 회복해야 한다. `[0,0]`가 첫 row에 남으면 shape는 맞지만 첫 A token이 base-only가 된다.

### orientation과 scaling을 shape만으로 판정하지 않는다

Square fixture는 transpose 오류를 숨긴다. 따라서 A는 `[2,3]`, B는 `[2,2]`처럼 적어도 한 축이 비대칭이어야 한다. target linear가 QKV packed parameter라면 logical output 폭과 physical packed segment를 분리한다. `q_proj` adapter는 Q segment에 들어가야 하고 TP rank는 그 Q output axis의 local slice만 소유할 수 있다.

Scaling 오류는 모든 output이 일정 배율로 어긋나는 형태로 보인다. rank 2와 alpha 4라면 `alpha/r=2`인지 다른 rule인지 source에서 확인한다. loader가 scale을 A에 미리 fold했는지, runtime kernel argument로 넘기는지에 따라서도 관찰 지점이 달라진다. A/B file 값만 비교해 scale이 누락됐다고 단정하지 않는다.

Base가 quantized되어 있어도 LoRA A/B는 별도 dtype으로 계산될 수 있다. base GEMM output과 delta를 어느 dtype에서 더하는지, adapter dtype cast가 언제 일어나는지 본다. “INT4 base이므로 adapter도 INT4”가 아니다. 반대로 FP16 delta를 더하기 위해 base weight 전체를 dequantize한다고 자동 가정하지도 않는다.

### target module resolution은 이름 바꾸기 이상의 일이다

Checkpoint는 `q_proj.lora_A`처럼 logical name을 가질 수 있지만 serving model은 QKV를 하나의 packed parameter로 만들 수 있다. Resolver는 name을 찾고 끝나는 것이 아니라 Q segment offset, local TP slice, A와 B 중 shard되는 축을 알아야 한다. gate/up packed MLP도 같은 문제를 가진다.

Unknown target를 조용히 무시하면 일부 layer만 adapter가 적용된다. 서버는 load success를 반환하고 품질만 이상해진다. 허용 정책이 있다면 applied, ignored, missing target inventory를 노출해야 한다. 32개 layer에서 동일 target가 모두 빠지면 regex 또는 naming generation 문제를 먼저 본다.

이제 우리는 identity가 맞을 때 나와야 할 수치와 shape를 갖고 있다. 다음은 artifact가 이 logical A/B로 변환되어 device slot에 들어가는 lifecycle이다.

### 51.2.1 Transformers의 active adapter와 multi-LoRA serving은 다르다

Transformers의 PEFT integration을 읽으면 `load_adapter`, `add_adapter`, `set_adapter`, `disable_adapters`, `enable_adapters`라는 익숙한 이름이 나온다. [`load_adapter`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/peft.py#L80-L238)는 config와 state dict를 해석하고 model에 adapter를 주입한다.

[`set_adapter`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/peft.py#L430-L470)는 model object의 active adapter state를 바꾼다.

이 API가 여러 adapter를 보유할 수 있다는 사실을, 한 forward batch의 token마다 서로 다른 adapter를 선택할 수 있다는 뜻으로 읽으면 안 된다. Model-global active state와 per-token mapping은 다른 concurrency contract다. Web handler 두 개가 같은 model object에 대해 A와 B를 번갈아 `set_adapter`하고 동시에 forward하면 lock이나 replica isolation 없이 어느 forward가 어느 state를 보게 될지 조사해야 한다.

Race fixture는 thread 또는 async task 두 개로 구성한다. Task A가 adapter A를 set한 뒤 barrier에서 기다리고, task B가 B를 set한 뒤 두 forward가 겹치게 한다. 단독 reference와 output을 비교한다. 이 실험을 실제 실행하지 않더라도 code review에서는 active state owner, mutation lock, forward가 adapter를 snapshot하는 지점을 찾는다. 요청마다 `set_adapter`를 호출하는 recipe라면 serving scheduler가 제공하는 token mapping과 같은 안전성을 자동으로 얻지 못한다.

Adapter list를 활성화하는 composition도 구분한다. 한 token에 A와 B delta를 모두 합성하는 것은 token 1은 A, token 2는 B를 쓰는 것과 다르다. API가 list를 받는다는 사실만으로 multi-tenant batching 지원이라고 쓰지 않는다. Composition order와 weight coefficient가 있으면 별도 effective adapter identity로 cache key에 반영해야 한다.

Disable은 tensor를 unload한다는 뜻이 아닐 수 있다. Forward 적용을 끄지만 module과 weights는 model에 남을 수 있다. Enable은 이전 active set을 다시 적용할 수 있다. 따라서 memory reclamation, request selection과 state mutation을 각각 확인한다. 운영자가 disable 뒤 GPU memory가 즉시 줄 것으로 기대한다면 API 의미가 맞지 않을 수 있다.

### 51.2.2 merge는 빠른 토글이 아니라 새 base generation이다

Runtime overlay는 `W`를 보존하고 요청마다 `ΔW`를 선택한다. Merge는 `W' = W + ΔW`를 물리적으로 만든다. Merge된 model은 low-rank kernel과 token mapping이 필요 없을 수 있지만, A와 B를 같은 base replica에서 요청별로 바꾸기는 어려워진다. 원래 W 사본이 없으면 unmerge의 수치와 reversibility도 구현에 종속된다.

Offline merge artifact는 새로운 base revision으로 취급한다. Tokenizer와 architecture가 같더라도 weight hash가 다르고 그 weight로 만든 KV cache도 원 base와 섞으면 안 된다. Cache key에 adapter가 없더라도 merged base identity가 달라 충돌을 막아야 한다. “adapter를 merge했으니 adapter-aware key가 필요 없다”는 말은 맞을 수 있지만 base revision을 그대로 두어도 된다는 뜻은 아니다.

In-place merge 중 요청을 받으면 일부 layer만 merge된 순간을 노출할 수 있다. Serving replica를 drain하고 merge를 완료한 뒤 새 generation을 publish하거나, 별도 replica에서 만들어 원자적으로 routing을 전환해야 한다. Merge failure rollback은 layer별 delta가 어디까지 적용됐는지 알아야 한다. 파일 저장 성공만 보고 memory model이 완전하다고 판단하지 않는다.

Quantized base는 더 까다롭다. GPTQ/AWQ packed tensor에 dense delta를 elementwise로 바로 더할 수 없다. Dequantize, add, requantize는 code와 scale을 바꾸며 original quantization error와 다른 artifact를 만든다. Runtime overlay는 quantized base output에 dense low-rank delta를 더할 수 있지만 backend와 dtype 지원을 확인해야 한다. 둘의 성능과 정확도를 같은 “LoRA 비용”으로 합치지 않는다.

### 51.2.3 llama.cpp에서 architecture LoRA와 external adapter를 가른다

llama.cpp source에는 이름에 `lora`가 들어간 두 종류가 보인다. 일부 model architecture는 원래 구조 안에 low-rank projection을 포함한다. 예를 들어 metadata의 attention LoRA rank는 DeepSeek 계열 MLA 같은 architecture field일 수 있다. 이것은 사용자가 나중에 올린 PEFT adapter가 아니다. Shape가 low-rank라는 이유만으로 hot-swappable tenant adapter로 분류하면 안 된다.

별도의 external adapter는 base model graph에 delta 연산을 추가하고 scale을 적용한다. 조사자는 adapter object의 load API, model binding, target tensor lookup, graph build에서 적용되는 조건과 lifetime을 찾는다. Server option이 여러 adapter를 읽을 수 있어도 한 request 또는 한 token마다 다른 조합을 고르는지 별도로 확인한다.

[`llama-arch.cpp`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-arch.cpp#L380-L390)의 adapter metadata keys는 file 의미의 한 경계일 뿐 concurrency 지원 증거가 아니다. Granite Switch의 slot-axis embedded A/B tensors도 model 자체가 학습한 routing mechanism이다. vLLM/SGLang의 external resident adapter slot과 이름이 비슷해도 owner, request mapping과 unload lifecycle이 다르다.

현재 server denominator에서 per-token external multi-LoRA가 입증되지 않으면 기능이 불가능하다고 일반화하지 않고 “감사한 경로에서 확인하지 못했다”고 쓴다. 대신 external adapter 한 개 또는 global set을 적용할 때 model context 간 isolation, scale change, cache invalidation과 graph rebuild 여부를 확인한다.

### 51.2.4 artifact 파일을 logical A/B inventory로 바꾼다

Adapter directory에는 config, safetensors 또는 binary weights와 optional tokenizer/module metadata가 있을 수 있다. Loader는 먼저 adapter type, rank, alpha, target modules, base hint를 읽고 expected inventory를 만든다. 그 뒤 file key를 logical module과 A/B role에 mapping한다. 이 순서는 base loader와 같다. Config 오류가 weight corruption처럼 보일 수 있다.

Inventory ledger에는 source key, target logical module, layer index, role A/B/bias, global shape, TP-local shape, dtype, scale owner, consumed 여부를 둔다. Unexpected key가 모두 harmless하다고 넘기지 않고 pattern을 본다. 모든 `k_proj`만 unexpected라면 model wrapper target support가 빠졌을 수 있다. 마지막 두 layer가 missing이면 adapter 제작 당시 layer count가 base와 다를 수 있다.

Base compatibility를 repository 이름만으로 판정하지 않는다. Hidden width, intermediate width, Q/K/V output width, vocab rows와 target naming generation을 비교한다. 같은 Llama family라도 GQA head 수나 packed module convention이 다를 수 있다. Adapter A의 B matrix output rows가 checkpoint base의 Q width와 맞는지 직접 확인한다.

Safetensors header shape가 맞아도 content가 올바른 target를 뜻하지는 않는다. Distinct-pattern conversion fixture로 logical module mapping을 test한다. A/B를 all-zero로 두면 mapping 오류가 output에 나타나지 않으므로 target별로 서로 다른 low-rank basis를 사용한다.

## 51.3 load는 여러 worker가 함께 commit하는 transaction이다

동적 adapter load API가 200을 반환했다고 모든 TP worker가 같은 tensor를 갖는 것은 아니다. Coordinator가 요청을 fan-out하고 rank 0은 성공했지만 rank 1이 OOM으로 실패할 수 있다. 성공 결과를 `any`로 합치거나 첫 응답만 반환하면 process group은 서로 다른 resident inventory를 가진다. 다음 collective는 끝나더라도 rank별 delta가 달라져 output이 틀릴 수 있다.

적재 상태를 `resolving → validated → host-loaded → sharded → device-resident → active`로 나눈다. active 이전 상태는 요청이 선택할 수 없어야 한다. 실패하면 새 allocation과 registry reservation을 역순으로 되돌린다. retry가 같은 id로 들어와도 중복 slot을 만들지 않는 idempotency가 필요하다.

### vLLM의 load와 LRU activation 경계

vLLM의 [`WorkerLoRAManager._load_adapter`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/worker_manager.py#L105-L171)는 request path를 해석하고 model-compatible LoRA object를 만드는 경계다. local file을 찾는 것, model configuration에 맞추는 것, device에 resident하게 만드는 것을 한 단어 “load”로 뭉치지 말고 branch별로 기록한다.

[`LRUCacheWorkerLoRAManager`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/worker_manager.py#L241-L319)는 요청된 adapter 집합을 적용하고 필요하면 load한 뒤 activate한다. LRU는 무엇을 내릴지 정하는 capacity policy다. running request의 reference safety는 별도 불변식이다. “가장 오래 안 썼다”는 사실만으로 아직 graph나 persistent batch가 참조하는 slot을 덮어써서는 안 된다.

`max_loras`, host cache capacity와 rank cap은 같은 숫자가 아니다. 하나는 동시 resident 또는 active 다양성을, 하나는 재적재를 줄이는 host-side cache를, 하나는 buffer shape 상한을 결정할 수 있다. 옵션 사전으로 나열하지 않고 세 번째 adapter가 들어왔을 때 어떤 state transition이 일어나는지 source에서 확인한다.

### SGLang의 fan-out과 rollback을 읽는다

SGLang의 dynamic load/unload control은 [`tokenizer_control_mixin.py`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_control_mixin.py#L575-L780)에서 여러 결과를 합치는 경계를 제공한다. 여기서 질문은 함수 이름이 아니라 성공 조건이다. 모든 scheduler/worker가 성공해야 전체 성공인지, partial failure에서 이미 load한 worker를 unload하는지, timeout 뒤 늦은 success가 orphan resident를 만드는지 본다.

Transaction log에는 operation id, adapter content identity, target worker set, rank별 prepare/commit/rollback 결과, slot allocation, 최종 registry publication을 둔다. 재시도는 이전 operation이 resolving인지 rollback 중인지 구별해야 한다. 같은 public name으로 다른 hash를 동시에 load하는 요청은 serialization하거나 새 generation으로 분리한다.

적재가 끝났어도 아직 token이 adapter를 선택한 것은 아니다. 다음 절에서는 request identity가 scheduler batch와 flattened token vector로 변하는 순간을 살핀다.

## 51.4 slot generation을 request mapping에 고정한다

Continuous batching에서 한 request가 항상 한 row인 것은 아니다. Prefill request는 여러 token을 내고 decode request는 보통 step당 한 token을 낸다. Chunked prefill이 있으면 같은 request의 일부 token만 현재 iteration에 들어온다. 따라서 adapter mapping을 request 수 길이로 만들고 token kernel에 넘기면 길이와 순서가 맞지 않는다.

Batch가 request A의 prompt token 3개, request B의 decode token 1개, base request의 prompt token 2개를 포함하면 flattened mapping은 `[slotA,slotA,slotA,slotB,-1,-1]`이어야 한다. 다음 iteration에서 A가 끝나고 B, base 순으로 compact되면 mapping도 같은 permutation으로 다시 만들어야 한다. 이전 buffer의 tail을 zeroing하지 않거나 valid length를 잘못 넘기면 stale slot이 읽힌다.

### vLLM의 mapping과 segment metadata

vLLM의 [`LoRAMapping`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/layers/utils.py#L27-L43)은 prompt mapping과 token mapping을 구분한다. Sampling에 필요한 mapping과 hidden-state token mapping이 왜 다른 길이를 가질 수 있는지 forward와 output path를 연결해 읽는다.

GPU kernel은 같은 adapter의 token을 모아 효율적으로 처리할 수 있다. [`LoRAKernelMeta.prepare_tensors`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L109-L171)는 token mapping을 device buffer에 복사하고 정렬과 unique counts를 준비한다. 이 정렬은 semantic order 변경이 아니라 compute grouping이다. 결과는 inverse mapping 또는 scatter index를 통해 원 token row로 돌아와야 한다.

Kernel entry가 mapping length `M`을 assert해도 identity correctness는 증명하지 못한다. A와 B가 뒤바뀐 vector도 길이는 같다. Distinct constant fixture로 row별 delta를 비교해야 한다. mapping buffer의 logical adapter id가 resident slot index로 변환되는 시점도 기록한다.

### SGLang의 batch diversity gate와 memory pool

SGLang ingress는 [`_validate_and_resolve_lora`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L3297-L3338)에서 요청 adapter를 해석하고 unique adapter 수 조건을 확인한다. 요청 batch 안의 다양성과 scheduler iteration의 active diversity는 같지 않을 수 있다. 여러 API request가 합쳐지는 뒤쪽에서 상한을 다시 지켜야 한다.

Scheduler의 [`_can_schedule_lora_req`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L3453-L3495)는 현재 running adapter와 새 request의 관계를 admission에 반영한다. 용량이 찼을 때 무조건 reject하는지, 기다리게 하는지, drainer로 전환하는지가 latency와 fairness를 바꾼다. 하지만 어떤 정책이든 running token의 slot이 중간에 바뀌지 않는 불변식은 지켜야 한다.

[`prepare_lora_batch`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/lora/mem_pool.py#L740-L785)는 active/pinned adapter와 buffer slot을 준비한다. Backend의 [`prepare_lora_batch`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/lora/backend/triton_backend.py#L265-L351)는 indices, ranks와 scalings를 device batch info로 내린다. Rank와 scaling은 adapter마다 다를 수 있으므로 단일 batch-wide scalar로 축약하면 안 된다.

이 절에서 request name은 실제 token row의 slot 선택으로 내려왔다. 다음 절은 그 mapping이 shrink·expand 계산과 TP shard에서 어떻게 사용되는지 본다.

## 51.5 batch에서 tenant별 slot을 token row로 펼친다

LoRA linear은 개념적으로 shrink `xAᵀ`, expand `zBᵀ`, scale과 base output addition으로 나뉜다. 여러 adapter를 한 batch에서 처리하려면 각 token row가 선택한 A/B를 사용해야 한다. 단순히 adapter마다 별도 GEMM을 실행하고 scatter할 수도 있고, grouped/segmented kernel을 사용할 수도 있다. 구현 차이보다 중요한 것은 mapping과 weight slot의 합의다.

Fixture `[A,A,B,base]`에서 shrink output의 row는 A 두 개, B 하나, zero 하나다. Base-only sentinel은 흔히 `-1` 같은 값일 수 있다. 이 값이 unsigned index로 변환되어 마지막 slot을 읽지 않는지 확인한다. Kernel이 no-adapter row를 skip하더라도 output buffer를 초기화하거나 base output만 보존해야 한다.

### sort와 scatter 사이의 두 permutation

Token을 adapter id로 stable sort하면 original→sorted permutation과 sorted→original inverse가 생긴다. TP 또는 MoE까지 결합하면 expert grouping permutation이 추가될 수 있다. 같은 index buffer를 의미가 다른 permutation에 재사용하지 않는다. Trace에는 각 coordinate system의 이름과 valid length를 남긴다.

첫 divergence를 찾기 위해 shrink 직전의 token row, selected slot과 A hash, shrink output, expand 직전 B hash, scaled delta, scatter destination을 기록한다. 전 layer 전체를 dump할 필요는 없다. 첫 layer의 작은 sample과 distinct constant fixture면 mapping 오류를 빠르게 찾을 수 있다.

### TP에서 A와 B의 소유권

Column-parallel base linear와 row-parallel base linear는 LoRA A/B shard 방식이 다를 수 있다. Q projection의 output axis를 rank별로 나누면 B의 output row를 shard하고 A는 replicate하는 설계가 자연스러울 수 있다. Row-parallel에서는 A의 input 축 shard와 collective가 연관된다. 실제 layer wrapper와 loader를 읽어야 하며 이름만 보고 공식처럼 적용하지 않는다.

TP=1에서 정상이고 TP=4에서만 틀리면 global A/B shape, rank-local slice, scale, collective 전후 delta를 비교한다. Packed QKV라면 logical Q offset과 rank-local packed offset을 모두 적는다. `q_proj`의 global slice를 packed destination 전체에 그대로 적용하면 K/V 경계를 침범할 수 있다.

Fully sharded LoRA option은 모든 tensor가 단순히 TP로 나뉜다는 뜻으로 읽지 않는다. 어떤 A/B를 shard하고 어떤 communication을 추가하는지 current consumer를 확인한다. Memory 감소와 latency 증가는 workload, rank와 backend에 따라 달라지므로 옵션 이름만으로 효과를 단정하지 않는다.

### quantized base와 dtype 경계

Base weight가 GPTQ/AWQ/FP8이어도 adapter delta는 별도 dense A/B로 계산될 수 있다. Base output dtype, shrink/expand accumulator, scale dtype, addition dtype과 final output을 분리한다. Adapter를 활성화했을 때만 kernel fallback이 생기는지 관찰한다.

Merge는 완전히 다른 경로다. Delta를 base weight에 물리적으로 더하면 매 token mapping이 필요 없지만 base storage의 identity가 바뀐다. Quantized base에 merge하려면 dequantize·add·requantize 또는 지원되는 별도 방식이 필요할 수 있다. Runtime overlay와 offline merged artifact를 모두 “adapter on”이라고 부르면 memory, reversibility와 isolation을 설명할 수 없다.

이제 계산 자체는 닫혔다. 하지만 correct delta가 잘못된 KV나 prefix cache와 결합하면 여전히 다른 adapter의 결과가 나온다.

## 51.6 KV와 prefix cache도 adapter identity를 알아야 한다

같은 prompt token이라도 adapter가 Q/K/V projection에 delta를 더하면 layer가 만든 K/V가 달라진다. Adapter A로 계산한 prefix cache를 B 요청이 재사용하면 이후 token의 low-rank 계산이 정확해도 attention은 A의 과거 상태를 읽는다. 증상은 첫 token보다 cache hit 이후에만 나타날 수 있다.

Cache key에는 base revision과 token prefix뿐 아니라 output에 영향을 주는 adapter content identity가 반영되어야 한다. Public name만 넣으면 같은 이름의 artifact가 hot replacement됐을 때 stale hit가 생긴다. Slot id만 넣으면 eviction 뒤 다른 adapter가 같은 slot을 차지할 때 충돌한다. Internal id가 process lifetime에만 unique하다면 persistent external cache의 key로 부족할 수 있다.

### prefix cross-adapter fixture

Prompt `P`를 A로 한 번 prefill해 cache를 만든다. 같은 `P`를 B로 요청하고 의도적으로 cache lookup을 수행한다. 정상이라면 miss이거나 B identity에 맞는 entry를 사용해야 한다. Hit가 났다면 첫 layer cached K/V sample과 B reference를 비교한다. Mapping과 adapter kernel이 모두 맞는데 cached K/V부터 다르면 cache-key divergence다.

Base-only도 하나의 명시적 identity다. Adapter 없음 sentinel을 빈 문자열, zero id와 missing field로 제각각 표현하면 key normalization에서 adapter 0과 다른 의미가 합쳐질 수 있다. Cache writer와 reader가 같은 canonical tuple을 사용해야 한다.

Cache entry가 adapter tensor를 직접 참조하지 않더라도 provenance는 필요하다. Entry 생성 generation, base와 adapter hashes, positional/config state를 남긴다. Adapter unload가 cache entry를 반드시 삭제해야 하는지는 cache가 content-addressed인지 slot-addressed인지에 따라 달라진다. Immutable content key라면 adapter가 재적재될 때 안전하게 재사용할 수 있지만 mutable slot key라면 invalidate가 필요하다.

### CUDA graph의 주소 안정성과 의미 안정성

CUDA graph replay는 같은 device buffer 주소를 재사용할 수 있다. Slot 2 주소가 유지되므로 pointer contract는 맞아 보인다. 하지만 그 주소의 owner가 A에서 B로 바뀌었다면 의미는 달라졌다. Mapping tensor가 B를 가리키는 새 값으로 replay 전에 갱신되는지, 이전 graph 실행이 끝난 뒤 slot copy가 시작되는지 확인한다.

Adapter rank와 active diversity가 capture shape에 포함되는지도 본다. 새 rank가 buffer 상한을 넘거나 mapping segment 수가 capture bucket을 벗어나면 eager fallback 또는 recapture가 필요할 수 있다. Correctness를 지키기 위한 fallback과 성능 regression을 분리해 기록한다.

이 절은 계산 밖의 상태도 identity를 보존해야 함을 보였다. 다음은 종료와 eviction에서 reference를 안전하게 끊는 법이다.

## 51.7 unload는 참조를 drain한 뒤 generation을 폐기한다

Adapter unload API를 받았다고 즉시 device buffer를 덮어쓰면 running request가 깨진다. Scheduler queue, persistent batch, captured graph, in-flight CUDA work, prefix cache가 각각 adapter를 참조할 수 있다. 안전한 lifecycle은 `active → draining → inactive → evictable → freed`다.

Draining에서는 새 request admission을 막고 기존 request가 끝나거나 정책에 따라 abort되기를 기다린다. Inactive는 새 mapping에 나타나지 않지만 아직 in-flight use가 끝났는지 synchronization이 필요할 수 있다. Evictable은 reference가 없음을 증명한 상태다. Freed 뒤 slot generation을 올리고 새 owner를 publish한다.

### unload while running 사건

Long decode request A를 실행하고 중간에 A unload를 호출한다. 허용 정책은 reject, wait/drain, request abort 가운데 하나일 수 있다. 무엇이든 silent overwrite는 안 된다. API response time, scheduler state, request result, slot owner와 CUDA completion event를 함께 기록한다.

Timeout이 있으면 timeout 후 작업이 백그라운드에서 계속되는지 확인한다. Client가 실패를 받았는데 worker 일부가 늦게 unload를 commit하면 registry와 resident state가 다를 수 있다. 다음 retry가 operation generation을 확인하고 수렴해야 한다.

### eviction과 abort cleanup

LRU eviction은 explicit unload와 trigger가 다르지만 reference safety는 같다. Capacity가 찼다는 이유로 running owner를 고르면 안 된다. Pinned adapter는 eviction 후보에서 빠질 수 있지만 pinned 수가 전체 capacity를 채우면 새 request를 어떻게 처리하는지 명시해야 한다.

Request abort는 token mapping, batch row, sampler mapping, adapter reference count를 함께 정리한다. KV block만 반환하고 adapter reference를 놓지 않으면 eviction이 영원히 막힌다. 반대로 reference를 먼저 놓고 persistent batch에서 row를 늦게 제거하면 slot reuse leakage가 생긴다. Cleanup 순서를 source와 event trace로 확인한다.

종료 경로까지 닫았으므로 이제 실제 사고 여섯 개를 같은 조사법으로 정리할 수 있다.

### 51.7.1 prepare와 commit 사이를 일부러 실패시킨다

분산 적재의 정상 경로만 시험하면 rollback은 영원히 검증되지 않는다. Fixture는 rank 1의 device allocation 직전에 의도적인 failure를 넣는다. Rank 0은 host tensor를 읽고 device slot까지 확보했으며 rank 1은 config validation만 끝낸 상태라고 하자. Coordinator는 각 rank의 prepare result를 모아 하나라도 실패하면 commit을 publish하지 않아야 한다. Rank 0에는 allocation 해제와 provisional registry entry 제거를 지시하고, rank 1에는 생성된 host cache reference가 있다면 놓도록 한다.

이때 단순히 `False` 응답 수를 세는 것으로 부족하다. Rank별 operation generation, provisional slot, owner, bytes allocated, active mapping exposure 여부를 확인한다. Rollback RPC가 timeout되면 operation을 완결됐다고 표시하지 말고 reconciliation 대상에 둔다. 다음 retry가 들어왔을 때 orphan slot을 새 slot과 별개로 만들면 누수가 누적된다. Retry는 같은 artifact identity와 operation id의 이전 state를 조회하고 cleanup을 마친 뒤 다시 prepare해야 한다.

Commit 순서도 중요하다. 모든 rank의 device copy 완료가 확인되기 전에 ingress registry가 adapter를 available로 표시하면 새 request가 일부 rank에서만 resident한 adapter를 선택할 수 있다. 따라서 external availability는 distributed commit 이후다. 반대로 commit은 됐지만 응답 전 connection이 끊겼다면 client retry가 duplicate load가 되지 않도록 idempotent lookup을 제공해야 한다.

Unload rollback은 load보다 어렵다. 일부 rank가 이미 free한 뒤 다른 rank가 running reference 때문에 unload를 거부하면 이전 상태로 완전히 되돌리기 어렵다. 그래서 unload는 먼저 drain과 reference check를 모든 rank에서 prepare하고, 모두 free 가능하다는 합의 뒤 commit하는 편이 안전하다. Current implementation이 이 exact protocol을 쓰지 않는다면 실제 ordering과 failure window를 문서화하고 보완 monitor를 둔다.

### 51.7.2 resident capacity와 batch diversity를 나눈다

GPU에 네 adapter가 resident할 수 있다고 한 batch에서도 네 adapter를 동시에 처리할 수 있는 것은 아니다. Kernel metadata buffer나 captured graph가 batch당 두 adapter만 지원할 수 있다. 반대로 batch diversity가 네 개여도 LRU manager가 여덟 adapter를 resident하게 두어 load thrash를 줄일 수 있다. 두 capacity를 같은 `max_loras`라는 말로 설명하면 admission failure를 잘못 진단한다.

예를 들어 resident A, B, C가 있고 batch active limit가 2라고 하자. Running batch가 A와 B를 사용 중일 때 C request가 도착한다. C는 이미 resident이므로 load latency는 없지만 즉시 batch에 들어갈 수 없다. Scheduler는 기다리거나 A/B 중 하나가 빠지는 경계를 선택해야 한다. 이 wait를 adapter load wait로 metric에 기록하면 잘못된 최적화를 하게 된다.

반대로 active limit에는 여유가 있지만 C가 resident하지 않고 모든 slot이 pinned이면 admission은 slot capacity에서 막힌다. Pinned adapter가 request를 처리하지 않는다고 inactive로 간주해 덮어쓰면 pin 계약을 어긴다. Operator에게 pinned count, resident count, active diversity, waiting request adapter set을 별도로 보여 주어야 한다.

Fairness도 adapter-aware다. 인기 adapter A 요청이 계속 도착하면 batch가 A로 채워지고 희귀 B가 diversity slot을 오래 기다릴 수 있다. 무조건 adapter별 batch를 만드는 것은 효율적일 수 있지만 tail latency를 악화시킨다. 이 장은 특정 fairness policy를 정답으로 고르지 않는다. 다만 scheduler가 adapter diversity 때문에 기다린 시간을 일반 token budget wait와 구분하고, starvation fixture에서 정책을 관찰하도록 요구한다.

### 51.7.3 adapter 교체와 hot reload의 정확한 의미

같은 public name A에 새 artifact A2를 배포하는 hot reload를 생각하자. Registry entry의 path만 바꾸고 internal id와 cache namespace를 그대로 쓰면 old running request와 new request가 한 identity로 합쳐진다. A1로 시작한 decode는 중간 layer step부터 A2 slot을 읽을 수 있고, A1 prefix cache를 A2가 hit할 수 있다.

안전한 방식은 A2를 새 immutable generation으로 prepare하고, 새 요청 resolution을 A2로 원자적으로 전환하며, A1 running reference가 drain될 때까지 A1 slot과 cache provenance를 유지하는 것이다. Public alias는 이동하지만 content identity는 공존한다. A1과 A2의 internal id 또는 generation은 달라야 한다.

Artifact가 같은 hash인데 path만 이동했다면 content-addressed identity는 유지할 수 있다. 그러나 loader config, target resolution code 또는 base revision이 바뀌면 같은 adapter files도 effective graph가 달라질 수 있다. Hash 범위에 config와 relevant engine normalization을 포함하거나 deployment generation으로 분리한다.

Hot reload test는 긴 A1 decode 중 alias를 A2로 전환하고, 전환 전후 새 request 두 개를 넣는다. A1 request는 끝까지 A1 reference와 일치해야 하고 전환 뒤 request는 A2와 일치해야 한다. Slot 수가 부족하면 전환을 거절하거나 drain해야 하며 A1을 즉시 덮어쓰지 않는다.

## 51.8 first divergence로 여섯 사건을 가른다

**사건 1 — slot reuse leakage.** A가 slot 2를 쓰고 끝난 직후 B를 slot 2에 load한다. 다음 batch의 한 token이 여전히 이전 mapping buffer를 읽는다. Expected mapping, published owner generation, device tensor hash를 비교한다. Mapping generation이 뒤처졌으면 kernel을 수정하지 않는다. Drain과 publish 순서를 고치고 A/B를 빠르게 교대하는 stress fixture를 둔다.

**사건 2 — partial distributed load.** TP rank 0은 load 성공, rank 1은 rank cap 또는 OOM으로 실패한다. Coordinator가 성공을 반환한다. Rank별 resident inventory와 operation result를 비교해 first divergence를 prepare/commit 사이에서 찾는다. All-worker commit과 rollback을 구현하고 같은 operation id retry를 검증한다.

**사건 3 — mixed-batch permutation.** `[A,A,B,base]`를 adapter별로 정렬한 뒤 inverse scatter가 오래된 batch order를 사용한다. Mapping length와 shape는 맞다. Distinct delta fixture에서 처음 틀린 row를 찾고 request compaction permutation과 LoRA sort permutation을 별도 buffer로 둔다.

**사건 4 — packed target mismatch.** `q_proj` A/B가 packed QKV의 K segment offset에 load된다. Load success와 output shape는 정상이다. Q/K/V segment별 constant pattern과 TP rank-local offsets를 비교한다. Logical target resolver와 physical packed loader의 mapping을 한 test에서 검증한다.

**사건 5 — prefix cross-adapter hit.** 동일 prompt 때문에 B가 A의 cached K/V를 받는다. Adapter kernel output은 B reference와 맞지만 first cached layer K/V가 다르다. Key에 immutable adapter content identity를 추가하고 hot replacement, base-only, unload/reload fixture를 넣는다.

**사건 6 — unload while running.** API가 device tensor를 즉시 free하고 in-flight graph가 같은 주소의 새 adapter를 읽는다. Scheduler reference, CUDA completion과 slot generation을 시간축에 놓는다. Drain protocol과 reference pin을 추가하고 abort, timeout, worker failure에서도 같은 cleanup을 검사한다.

이 사건들은 모두 응답이 틀릴 수 있지만 최초 divergence는 서로 다르다. Identity resolution, artifact load, slot ownership, token mapping, packed target, cache key, cleanup 중 첫 다른 지점을 찾으면 경쟁 가설을 빠르게 기각할 수 있다.

### 51.8.1 slot leakage를 값으로 증명한다

Leakage 재현은 실제 고객 adapter 없이도 가능하다. 모든 layer를 넣은 거대한 fixture 대신 첫 linear target 하나에 A는 output 첫 좌표에 `+1`, B는 둘째 좌표에 `+2`가 나오도록 low-rank basis를 만든다. Base-only는 zero다. Slot 0과 1을 번갈아 재사용하면서 batch order와 request completion을 무작위로 바꾼다.

각 iteration에서 CPU reference는 immutable content identity로 delta를 계산한다. Engine trace는 logical id vector, resolved slot/generation vector와 kernel sorted indices를 저장한다. Output mismatch가 나면 먼저 request identity와 CPU reference가 맞는지, 그 다음 logical→slot resolution, slot→tensor digest, kernel row를 순서대로 비교한다. GPU output부터 거꾸로 추측하지 않는다.

Slot owner가 B로 바뀌었는데 mapping snapshot은 A의 slot 번호를 그대로 썼다면 두 설계가 가능하다. Mapping에 generation을 포함해 mismatch를 reject하거나, running mapping이 모두 drain된 뒤에만 owner를 바꾼다. Generation check가 kernel hot path 비용 때문에 어렵다면 host scheduler와 stream ordering이 동일한 안전성을 제공하는지 증명해야 한다.

Race window를 넓히기 위해 copy와 graph replay 사이에 event를 삽입할 수 있다. Production에서 sleep을 넣는 것이 아니라 test hook으로 `owner reserved`, `copy enqueued`, `copy complete`, `mapping published`, `old replay complete` 경계를 제어한다. 모든 합법적 ordering과 실패 ordering에서 invariant를 assert한다.

### 51.8.2 partial load를 상태 수렴 문제로 본다

Distributed failure fixture는 rank마다 다른 지점에서 한 번씩 실패시킨다. Config read, safetensors open, shape validation, host allocation, device allocation, H2D copy, registry prepare, commit response에서 failure를 주입한다. 각 경우 최종 상태가 “모든 rank active” 또는 “모든 rank absent” 중 하나로 수렴해야 한다.

중간 state가 남으면 reconciliation loop가 필요하다. Coordinator가 operation journal을 읽고 worker inventory를 조회해 missing commit을 완료할지 rollback할지 결정한다. Client request timeout과 server operation cancellation을 동의어로 보지 않는다. Client가 기다리지 않아도 operation은 명시적으로 success 또는 rolled-back state에 도달해야 한다.

Error message에는 어느 rank가 실패했는지와 retry 가능성을 담되 internal path와 credential을 노출하지 않는다. Metric에는 stage별 failure count와 rollback incomplete count를 둔다. 단순 load failure rate가 낮아도 incomplete rollback 한 건은 correctness 위험이 크므로 별도 alert다.

### 51.8.3 mixed batch는 세 좌표계를 함께 기록한다

Mixed batch에는 request index, flattened token index, adapter-sorted index 세 좌표가 있다. Sampling은 sequence/output index를 하나 더 가질 수 있다. Bug report에서 “row 3”이라고만 쓰면 어느 좌표인지 알 수 없다. Ledger는 `request[2] → token[7:11] → sorted[4:8] → output[2]`처럼 edge를 기록한다.

Prefill, chunked prefill, decode, speculative token이 섞이면 request당 scheduled token 수가 매 step 바뀐다. Mapping repeat count는 prompt 전체 길이나 total computed length가 아니라 이번 forward에 실제 들어가는 token 수와 합의해야 한다. Padding token이 kernel M에는 들어가지만 semantic output에는 없을 수도 있으므로 valid mask도 확인한다.

Stable sort는 같은 adapter 안에서 token order를 유지할 수 있지만 correctness가 stable에 의존하는지 source를 읽는다. Segment kernel이 결과를 sorted buffer에 쓰고 inverse scatter가 있으면 내부 order는 바뀌어도 된다. Scatter가 없고 grouped operation이 original order를 전제한다면 sort 자체가 허용되지 않는다.

Abort/finish compaction fixture는 매 iteration 한 request를 제거하고 새 request를 추가한다. Base-only sentinel을 첫, 중간, 마지막 위치에 둔다. A와 B의 prompt/decode 역할을 교대한다. 모든 row에 unique expected delta를 주어 permutation이 틀린 첫 순간을 찾는다.

### 51.8.4 packed target는 load-time semantic test로 잡는다

Packed target mismatch는 runtime 품질 평가까지 기다릴 필요가 없다. Q, K, V logical adapter B tensor를 서로 다른 constant stripe로 만들고 load 후 resident packed buffer segment sample을 검사한다. TP rank마다 expected stripe와 offset을 계산한다. Gate/up도 같은 방식으로 두 pattern을 쓴다.

Source name mapping, packed segment mapping, TP slice를 한 함수가 수행하더라도 test assertion은 세 단계로 나눈다. Name이 올바른 destination logical role을 찾았는가, role이 올바른 segment를 골랐는가, global segment에서 현재 rank slice를 골랐는가. 이 분해로 TP=1에서 숨는 offset bug를 찾는다.

Adapter rank가 target마다 다르거나 rank padding이 있으면 physical buffer rank dimension과 effective rank를 분리한다. Kernel은 padded columns을 mathematical zero로 보거나 effective rank를 인자로 받아야 한다. 이전 slot의 padded tail이 남아 있으면 새 low-rank adapter가 stale contribution을 받을 수 있으므로 load 시 zeroing policy를 확인한다.

### 51.8.5 cache 사고는 fresh run을 대조군으로 둔다

Wrong answer가 cache hit에서만 발생하는지 가장 빠르게 확인하려면 동일 request를 cache disabled 또는 unique prefix로 실행한다. Fresh run이 reference와 맞고 hit run만 다르면 adapter load와 kernel 전체를 먼저 의심할 이유가 줄어든다. 그러나 sampler randomness가 비교를 흐리지 않도록 deterministic logits 또는 first-layer K/V checkpoint를 사용한다.

Key audit는 serialize된 key material을 semantic field로 decode해 본다. Base id, adapter content id, token block, position regime와 relevant multimodal/template state가 들어가는지 확인한다. Hash digest만 두 개 비교하면 어느 field가 누락됐는지 알 수 없다. Test 환경에서는 pre-hash tuple을 안전하게 logging한다.

Hot reload는 같은 public name, 다른 content hash를 사용한다. Alias-only key가 충돌하는지 확인한다. Slot reuse fixture는 다른 public name이 같은 slot을 차지하게 한다. Slot-only key가 충돌하는지 확인한다. Base-only와 adapter id 0 표현도 별도 case로 둔다.

Cache invalidation으로 문제를 숨길 수는 있지만 reuse 가치가 사라질 수 있다. Content-addressed provenance를 정확히 만들면 immutable same-content adapter는 unload/reload 뒤에도 안전하게 hit할 여지가 있다. Correctness를 먼저 닫고 retention과 sharing policy를 그 위에 얹는다.

### 51.8.6 unload 사건은 시간축으로 판정한다

Unload race는 log line 순서만으로 부족할 수 있다. Host timestamp와 GPU event completion을 한 timeline에 놓는다. `new admissions disabled`, `last batch mapping built`, `last kernel launched`, `last kernel complete`, `reference zero`, `free enqueued`, `free complete`, `new owner copy`, `new mapping publish`를 기록한다.

안전한 partial order는 last use completion이 free/reuse보다 앞서고, new copy completion이 new mapping publish보다 앞서는 것이다. Host call return 순서가 GPU execution 순서를 자동 보장하지 않는다. 다른 stream을 쓰면 explicit event dependency가 필요할 수 있다. 같은 stream이라도 manager thread가 어느 stream에 enqueue하는지 확인한다.

Unload API가 wait하지 않고 operation id를 반환한다면 status endpoint 또는 event로 completion을 확인할 수 있어야 한다. 즉시 200을 “memory freed”라고 metric에 기록하지 않는다. Drain wait, GPU free와 registry removal latency를 분리한다.

Abort policy도 명확히 한다. Unload가 running request를 강제 abort한다면 client error, KV release, output queue close, adapter reference와 mapping cleanup이 하나의 terminal path로 수렴해야 한다. Request가 성공처럼 끝나거나 partial output 뒤 이유 없이 끊기면 운영자가 adapter race를 알아차리기 어렵다.

## 51.9 Llama fixture를 요청에서 GPU row까지 수직 추적한다

이제 앞의 사건을 하나의 연속된 trace로 묶자. Base는 hidden width 4096, query heads 32, KV heads 8인 Llama이고 TP=4다. Adapter A와 B의 rank는 각각 8과 16이며 둘 다 `q_proj`와 `v_proj`를 target으로 한다. A는 tenant red, B는 tenant blue에만 허용된다. A의 public alias는 `red-style`, artifact content id는 `sha256:a1`, internal id는 17, 현재 slot은 2 generation 41이다. B는 `blue-style`, `sha256:b1`, internal id 23, slot 5 generation 9다.

수직 추적의 첫 관문은 ingress에서 선택을 고정하는 일이다.

Red request가 model field에 `red-style`을 넣으면 router는 tenant scope를 확인하고 registry snapshot에서 `sha256:a1`을 얻는다. Registry가 path만 반환한다면 resolution 직후 file manifest hash를 확인해 immutable content id를 만든다. Request object에는 public name과 content id를 모두 보존한다. 이후 alias가 A2로 이동해도 이 request는 A1을 계속 가리킨다.

Request validation은 engine capability도 확인한다. LoRA가 disabled인데 request가 adapter를 요구하면 admission 전에 거절한다. Rank 8이 configured maximum을 넘는지, target model이 adapter를 지원하는지, 한 request가 허용된 adapter 수를 넘는지 본다. 이 단계에서 GPU slot이 없어도 당장 unknown adapter라고 말하지 않는다. Identity와 capacity failure를 다른 error로 유지한다.

Batch API가 여러 prompts를 받으면 adapter field가 scalar인지 prompt별 list인지 normalization 규칙을 확인한다. Scalar A는 모든 prompt에 A를 적용할 수 있지만 list `[A,B]`는 tenant authorization과 length를 각각 검증해야 한다. Length mismatch를 마지막 token mapping 단계까지 미루면 어느 prompt가 어느 identity인지 잃는다.

Trace checkpoint는 `request_id`, tenant bounded id, public alias, resolved content id, base revision, resolution registry generation, authorization decision이다. Path와 credential은 trace attribute로 그대로 내보내지 않는다. 이 값이 worker까지 전달되는 DTO에서 누락되면 internal id만 남아 hot reload를 구분하지 못할 수 있다.

### 51.9.1 resident lookup은 logical id와 slot owner를 대조한다

Worker는 internal id 17을 table에서 찾는다. Hit라고 바로 사용하지 않고 table entry의 content id가 `sha256:a1`인지 확인한다. Name-based equality만 사용하면 alias replacement 뒤 A2 request가 A1 resident를 hit할 수 있다. Content mismatch는 새 generation load 또는 explicit conflict다.

Slot 2의 owner record에는 internal id 17, content id a1, generation 41, load operation id, tensor inventory digest, active reference count가 있다. Model layer wrapper는 slot 2의 A/B buffer offsets를 알고 있다. Table entry와 slot owner가 다르면 inconsistent state다. 한쪽을 자동으로 truth로 삼아 계속하지 않고 request admission을 멈추고 reconciliation한다.

Miss이면 capacity policy가 victim을 고른다. Victim은 LRU tail이어도 pinned 또는 referenced이면 제외한다. CPU cache에 artifact가 있다면 disk/hub resolution을 생략할 수 있지만 config hash와 base compatibility 결과를 재사용해도 되는 generation인지 확인한다. Engine upgrade나 base reload 뒤 old validation cache를 그대로 쓰지 않는다.

Copy는 provisional slot에서 수행한다. A/B 모든 target layer와 rank-local slice가 들어간 뒤 inventory digest를 계산하고 ready flag를 세운다. Ready 이전 slot은 mapping에 나타나지 않는다. CUDA asynchronous copy를 사용했다면 host enqueue 완료와 device visibility 완료를 구분하고 appropriate event/dependency를 건다.

### 51.9.2 global adapter shape를 TP-local shape로 내린다

Fixture의 Q projection global output width는 4096이고 V는 1024다. Rank 8 adapter에서 `q_proj` A가 `[8,4096]`, B가 `[4096,8]` convention이라 하자. Column-parallel Q가 output rows를 네 rank에 나누면 B는 rank당 `[1024,8]` slice가 되고 A는 각 rank가 `[8,4096]`을 가질 수 있다. V의 B는 rank당 `[256,8]`이다.

이것은 illustrative contract이며 concrete wrapper source가 다른 sharding을 쓸 수 있다. Ledger에는 global source shape, partition dimension, rank offset/length, local destination shape, replicate 여부를 적는다. Rank별 B hash가 다르고 A hash가 같다는 pattern을 예상할 수 있다. Fully sharded mode라면 A도 나뉘고 collective가 추가될 수 있으므로 별도 expected pattern을 만든다.

Packed QKV runtime parameter에서 Q segment local rows 1024, K 256, V 256이 한 buffer에 배치될 수 있다. Q adapter B는 Q segment의 local offset에, V adapter B는 Q와 K 뒤의 V offset에 들어간다. Global V offset 5120을 local buffer에 그대로 쓰면 범위를 벗어나거나 잘못된 padding을 건드린다. Resolver는 logical target를 local packed coordinate로 변환해야 한다.

Target inventory는 layer마다 반복된다. 한 layer만 offset이 다르면 layer-specific fused layout을 확인한다. 모든 V target가 같은 양만큼 어긋나면 global/local offset conversion 문제를 의심한다. Constant pattern A/B로 loaded slot의 segment sample을 읽으면 forward 전에도 검출할 수 있다.

### 51.9.3 scheduler request 배열을 token 배열로 펼친다

Iteration 시작 시 running request는 A decode 두 개, B decode 하나다. Waiting에서 A prefill prompt 5 token과 base-only prompt 3 token이 admission된다. Scheduler request order를 `[Adec1,Bdec1,Adec2,Apref,basepref]`라고 하자. Model runner가 만드는 flattened tokens는 각각 1,1,1,5,3개다.

Logical adapter ids vector는 request 단위로 `[17,23,17,17,none]`이지만 token vector는 `[17,23,17,17,17,17,17,17,none,none,none]`이다. Slot resolution 뒤 `[2,5,2,2,2,2,2,2,-1,-1,-1]`이며 generation sidecar 또는 batch snapshot은 slot 2가 generation 41임을 보장한다. Kernel이 internal id를 직접 받는지 slot index를 받는지는 구현에 따라 다르므로 두 vector를 혼동하지 않는다.

Chunked prefill이 Apref에서 처음 2 token만 선택하면 vector 길이는 8이 아니라 5+가 아니라 실제 scheduled tokens에 맞춰 줄어든다. Request prompt length 전체로 mapping을 repeat하면 뒤 token offsets가 밀린다. Query start locations, scheduled token counts와 mapping range를 같은 ledger에 둔다.

Persistent batch가 finished Adec1을 제거할 때 request state array, token counts, block table, sampling metadata와 adapter mapping을 같은 swap/remove permutation으로 갱신한다. Mapping만 rebuild한다면 source request order가 already compacted됐는지 확인한다. Double permutation도 stale mapping만큼 위험하다.

### 51.9.4 sort·shrink·expand·scatter를 한 row로 확인한다

Kernel metadata가 slot id로 stable sort하면 original vector `[2,5,2,2,-1]`은 예를 들어 `[-1,2,2,2,5]`로 바뀐다. Sorted token indices가 `[4,0,2,3,1]`라면 inverse scatter는 output sorted row 1을 original row 0으로 돌려야 한다. Segment starts와 counts는 base-only 1, A 3, B 1이다.

Shrink kernel은 A segment에 slot 2의 A tensor를, B segment에 slot 5의 A tensor를 사용한다. Base-only segment는 skip한다. Expand는 같은 segment owner의 B tensor와 scaling을 사용한다. Shrink에서 slot 2를 쓰고 expand에서 stale slot 5를 쓰는 일이 없도록 batch info가 두 kernel 사이에서 immutable해야 한다.

Layer wrapper가 base output을 먼저 계산한 뒤 delta를 add할 때 output buffer alias와 accumulation order를 확인한다. Base-only row에는 delta buffer가 zero이거나 add를 skip해야 한다. 이전 iteration의 delta buffer를 reuse하고 valid rows만 일부 overwrite하면 base-only row에 stale A delta가 남을 수 있다. Fixture는 base-only row를 batch 중간과 끝에 번갈아 배치한다.

Sampling logits adapter가 별도로 있다면 hidden token mapping과 prompt/sampler mapping을 구분한다. 한 request에서 last token row만 logits head로 가기 때문에 sampler mapping 길이는 sequence 수일 수 있다. Hidden mapping을 그대로 logits kernel에 넘기거나 반대의 경우 shape mismatch 또는 wrong adapter head가 된다.

### 51.9.5 prefill에서 만든 cache provenance를 decode가 이어받는다

Apref의 first chunk가 A1으로 K/V를 만들면 sequence state에는 adapter content identity a1이 붙는다. 다음 chunk와 decode가 같은 identity인지 assert한다. API가 mid-request adapter change를 허용하지 않는다면 request mutation을 reject한다. 허용하려면 이전 KV를 어떻게 처리할지 명시해야 하며 단순 mapping 변경만으로는 과거 K/V가 바뀌지 않는다.

Prefix lookup key는 base revision, token block, relevant position/config state와 a1을 포함한다. Internal id 17은 process restart에서 재사용될 수 있고 slot 2는 eviction에서 재사용되므로 persistent key material로 적합하지 않다. Content hash 전체가 길면 collision-resistant digest를 사용하되 registry alias만 사용하지 않는다.

Cache hit 뒤에도 provenance를 response trace에 남긴다. Wrong answer 조사에서 adapter kernel mapping이 맞다는 사실만 보고 끝내지 않고 hit entry의 adapter identity를 대조한다. Cache hit ratio metric은 adapter별 bounded bucket으로 볼 수 있지만 unbounded public names를 label로 쓰지 않는다.

A1 unload 뒤 content-addressed cache entry를 보존할지는 policy다. Weight가 나중에 같은 hash로 다시 적재되고 base/config도 같다면 재사용 가능할 수 있다. 그러나 tenant scope나 data retention policy가 cache sharing을 막을 수 있다. Correctness identity와 authorization/lifecycle policy를 모두 만족해야 한다.

### 51.9.6 abort와 unload가 reference를 역순으로 놓는다

Apref client가 disconnect하면 scheduler는 앞으로의 token admission을 막고 batch에서 request row를 제거한다. In-flight forward가 있다면 completion 전 slot reference를 놓지 않는다. Output queue, KV blocks, prefix writer transaction, adapter reference를 어느 순서로 cleanup하는지 기록한다.

Adapter reference count는 request admission 시 잡고 final completion/abort cleanup에서 정확히 한 번 놓아야 한다. Retry나 duplicated abort가 두 번 decrement해 negative 또는 premature zero가 되지 않게 idempotent state transition을 둔다. Exception path가 normal finalizer를 건너뛰는지 확인한다.

Explicit unload A1은 registry alias를 새 request에서 숨기고 running references가 zero가 되기를 기다린다. Prefix entries가 slot pointer를 보유하지 않는 content-addressed data라면 device adapter reference와 별개다. CUDA graph executable이 slot buffer address만 capture하고 mapping을 runtime update한다면 graph object 자체가 owner reference인지 current design을 확인한다.

Reference zero 뒤 device free를 enqueue했어도 stream에서 완료되기 전 slot을 새 copy에 재사용하면 write/read overlap이 생길 수 있다. Event 또는 stream ordering으로 completion을 보장하고 그 후 generation을 increment한다. Host manager state만 보고 GPU work 완료를 추정하지 않는다.

이 연쇄를 끝까지 이으면 비로소 다음과 같은 판정문을 쓸 수 있다.

정상 판정은 “A가 로드됐다”보다 길지만 모호하지 않다. “Red request는 registry generation 12에서 alias `red-style`을 content a1으로 resolution했고 internal id 17, slot 2 generation 41을 snapshot했다. TP=4 각 rank는 Q/V B rows 1024/256을 올바른 packed offsets에 commit했다. Scheduler의 8 scheduled tokens는 slot vector와 같은 permutation을 사용했고 shrink/expand/scatter sample은 hand fixture와 일치했다. Prefix entry는 base revision과 a1을 key로 사용했으며 abort 뒤 reference zero와 CUDA event completion 후 slot generation이 42로 전환됐다.”

실패 판정도 최초 divergence를 포함한다. “Request와 resident tensor는 B1이었지만 prefix lookup은 public alias만 key에 사용해 hot reload 전 B0 entry를 hit했다. First-layer cached K/V부터 B1 reference와 달랐고 fresh-cache run은 일치했다. Content identity를 key와 provenance에 추가하고 B0/B1 coexistence fixture를 통과했다.” 이런 문장은 누가 무엇을 수정해야 하는지 알려 준다.

## 51.10 운영 원장과 관측값

한 request의 원장은 ingress correlation id로 시작한다. Public adapter name과 authorization result, resolved immutable artifact identity, engine internal id, worker별 load generation, resident slot generation, scheduler request position, flattened token range, token-to-slot mapping, kernel segment, cache key digest, output metadata, final reference release를 잇는다.

Metric은 load latency와 success/failure, rollback count, host/device resident count, active/pinned count, eviction reason, batch unique adapter count, token per adapter, mapping refresh generation, adapter-aware cache hit를 포함할 수 있다. Tenant 이름과 path를 unbounded label로 쓰면 cardinality와 정보 노출 문제가 생긴다. Bounded id와 trace lookup을 조합한다.

### 무엇을 alert로 만들 것인가

가장 강한 alert는 state invariant 위반이다. Registry에는 active인데 일부 worker에 resident하지 않음, running mapping이 inactive generation을 참조함, active reference가 있는데 slot owner가 바뀜, rollback 뒤 allocation이 남음, cache hit identity가 request identity와 다름 같은 사건이다. 단순 resident count 증가는 capacity signal이지 correctness 위반은 아니다.

Latency는 resolution, host load, H2D copy, distributed commit, admission wait, kernel overhead로 분해한다. Adapter diversity가 늘어 latency가 증가했을 때 load thrash인지 batch grouping overhead인지 구분한다. Eviction rate와 reload byte, scheduler wait와 active diversity를 함께 본다.

### 재현 bundle

재현 자료에는 secret path 대신 hashes, current pins, base/adapter config, target inventory diff, worker state snapshots, mapping vectors와 slot generations, 작은 input token ids, cache hit/miss와 first-layer sample을 넣는다. 전체 customer prompt나 weight를 보고서에 복사하지 않아도 first divergence를 재현할 synthetic fixture를 만들 수 있다.

재현 bundle의 시간 범위는 load API 호출부터 마지막 cleanup까지다. Wrong output 한 줄만 수집하면 adapter가 언제 resident가 됐고 어떤 request가 slot을 먼저 참조했는지 알 수 없다. Coordinator와 worker clock 차이가 있으면 operation sequence number와 causal parent를 함께 둔다. GPU event는 host wall clock과 정확히 같지 않으므로 stream event order를 별도 edge로 표현한다.

Weight 자체를 공유할 수 없는 사고에서는 tensor digest와 작은 승인된 sample statistic을 쓴다. Layer·target·A/B별 shape, dtype, finite count, norm, first/last small slice hash를 남기면 rank 간 다른 shard나 stale slot을 찾을 수 있다. 전체 tensor hash는 TP rank마다 달라질 수 있으므로 global expected slice와 연결한다. Replicated A tensor는 rank 간 같아야 한다는 식의 expected relation을 함께 기록한다.

Mapping vector도 customer token 내용을 요구하지 않는다. Request의 bounded synthetic index, scheduled token count, adapter content digest, logical id, slot/generation과 permutation만 있으면 된다. Token ids가 target behavior에 꼭 필요할 때는 최소 fixture로 치환한다. 운영 trace와 재현 fixture 사이에 동일한 coordinate transformation이 적용되는지를 test한다.

### 용량 문제와 correctness 문제를 같은 dashboard에서 가르지 않는다

Adapter load latency가 길고 eviction이 많아도 mapping이 정확하면 성능 문제다. Resident owner와 running generation이 다르거나 partial rollback이 남으면 correctness 문제다. Dashboard는 두 범주를 시각적으로 분리하되 correlation할 수 있게 한다. Eviction spike 직후 generation mismatch가 발생한다면 capacity pressure가 race를 촉발한 증거가 된다.

Capacity panel에는 host cache hit/miss, device load bytes, resident/pinned/active counts, wait queue의 unique content ids, victim selection reason을 둔다. Batch panel에는 scheduled requests/tokens, unique adapters, base-only rows, sorted segment counts, eager/graph path를 둔다. Correctness panel에는 unresolved registry identity, worker inventory disagreement, stale generation reference, rollback incomplete, cache provenance mismatch를 둔다.

Adapter별 latency를 public name label로 직접 내보내면 tenant 수에 따라 Prometheus cardinality가 폭증할 수 있다. 상위 제한된 cohort 또는 hashed bounded bucket을 metric에 쓰고 정확한 identity는 trace/log에서 찾는다. `adapter_loaded=1{name=...}`처럼 모든 adapter를 permanent series로 유지하는 방식은 unload 뒤에도 series가 남아 실제 resident state와 혼동될 수 있다. Resident inventory는 bounded gauges와 별도 query endpoint를 조합한다.

### 배포 전 회귀 matrix

회귀 test는 adapter 수 0,1,capacity,capacity+1과 rank 1,최대 rank를 교차한다. TP=1/2/4, eager/graph, prefix cache off/on, quantized base off/on, prefill/decode/mixed batch를 모두 무작정 Cartesian product로 만들 필요는 없다. 각 risk edge를 적어 최소 pairwise fixture와 몇 개의 high-risk full combination을 고른다.

가장 중요한 조합은 slot eviction 중 mixed batch, hot reload 중 cache hit, TP packed QKV target, abort와 unload 동시 발생, partial worker load retry다. 각 test는 단순 “요청 성공”이 아니라 row별 hand-calculated delta, content/slot generation, cache provenance와 final zero reference를 assert한다.

성능 회귀도 correctness fixture와 같은 workload identity를 사용한다. Adapter diversity 1에서 4로 늘 때 tokens/s와 latency가 어떻게 바뀌는지 보되 load warm/cold를 분리한다. Graph fallback, mapping sort와 H2D metadata copy 시간이 증가했는지 trace로 설명한다. Faster 결과 하나로 격리 검사를 생략하거나 correct 결과 하나로 production capacity를 단정하지 않는다.

Release review에서는 adapter 관련 option default가 바뀌었는지뿐 아니라 request DTO, registry equality/hash, slot manager, batch compaction, cache key와 graph dispatcher의 diff를 함께 본다. 이들은 서로 다른 디렉터리에 있어도 하나의 identity chain을 구성한다. 변경된 edge가 있으면 fixture의 expected ledger도 갱신하고 old/new generation coexistence를 다시 시험한다.

## 51.11 네 요청·여덟 token을 artifact에서 kernel row까지 추적한다

hot reload 직후 tenant Red와 Blue가 같은 public alias `style`을 사용했다. authorization scope와 artifact는
각각 Red A1, Blue B1로 다르다. 동시에 Red는 A1을 A2로 교체했다. 대부분 요청은 정상이었지만 reload 전
admission되고 이후 decode된 요청만 다른 스타일을 섞어 답했다. load API와 GPU copy는 모두 성공했다.

작은 batch를 고정한다. R0 base-only 2 tokens, R1 Red A1 prefill3, R2 Blue B1 decode1, R3 Red A2 decode2다.
flattened token rows8개의 expected identity는 `[base,base,A1,A1,A1,B1,A2,A2]`다. base sentinel=-1,
A1 slot2/gen41, B1 slot5/gen9, A2 slot3/gen1이면 slot vector는 `[-1,-1,2,2,2,5,3,3]`이고 generation
vector 또는 동등한 lifetime fence도 필요하다.

slot index만 넘기면 hot reuse를 못 잡는다. slot2가 A1 gen41에서 C1 gen42로 바뀐 뒤 stale request가 2를
읽으면 pointer와 shape는 유효하지만 다른 delta다. admission은 immutable content와 resident slot generation을
snapshot하고 GPU completion까지 reference를 유지해야 한다.

LoRA 수치는 input `x=[1,2]`, rank1, A `[3,4]`, B column `[5,6]`, scaling0.5로 둔다. `Ax=11`,
`B(Ax)=[55,66]`, delta `[27.5,33]`이다. base `[10,20]`이면 final `[37.5,53]`이다. A1/B1/A2를 서로
다른 상수 pattern으로 만들면 어느 token row가 어느 slot을 썼는지 역검산할 수 있다.

kernel이 adapter별 rows를 sort하면 original `[0..7]`은 base0,1→A1 2,3,4→A2 6,7→B1 5 순서로
재배열될 수 있다. A/B matmul 뒤 inverse scatter가 original token positions를 복원해야 한다. slot mapping이
맞아도 inverse permutation이 틀리면 올바른 delta가 다른 request row에 더해진다.

vLLM per-token metadata는 request mapping을 token segments로 확장하는 source 경계다.
[vLLM token metadata](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L109-L171)

logical request identity는 request source에서, device residency와 LRU는 worker manager에서 잇는다.
[vLLM request identity](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/request.py#L8-L71)
[vLLM worker manager](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/worker_manager.py#L105-L319)

registry는 Red alias generation7→A1, generation8→A2를 소유한다. R1이 generation7에서 admission됐으면
decode까지 A1을 쓴다. 실행 중 alias를 다시 resolve하면 prefill A1, decode A2가 섞이고 이미 만든 KV와
identity가 달라진다. new alias generation은 new admission에만 적용한다.

SGLang ingress와 distributed control도 identity resolution, worker fan-out과 rollback을 잇는다.
[SGLang ingress](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L3297-L3338)
[SGLang load control](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_control_mixin.py#L575-L780)

일부 TP worker만 commit했는데 API success를 반환하거나 scheduler admission을 열면 rank별 delta가 갈린다.

SGLang scheduler와 memory pool/backend는 resident diversity에서 batch row metadata로 내려간다.
[SGLang admission](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L3453-L3495)
[SGLang pool](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/lora/mem_pool.py#L740-L785)

[SGLang Triton metadata](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/lora/backend/triton_backend.py#L265-L351)

Transformers PEFT의 model-wide active adapter 변경은 continuous mixed-request row mapping과 다르다.
[Transformers PEFT load](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/peft.py#L80-L238)
[Transformers activation](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/peft.py#L390-L509)
기능 이름이 adapter라도 state owner와 concurrency contract를 같다고 쓰지 않는다.

capacity도 수치화한다. Llama q_proj H4096/out4096, rank16, fp16 LoRA는 A와 B 합 256KiB/layer다. v_proj
out1024는 160KiB/layer다. q+v, layers32면 약 13MiB이고 metadata/alignment/other targets를 더한다. slots8은
단순 약 104MiB지만 active/pinned references 때문에 evictable slots가 더 적을 수 있다. resident cap, batch
diversity cap과 admission capacity를 구분한다.

## 51.12 종합 판정: adapter는 요청에 붙은 작은 파일이 아니다

LoRA의 수학은 짧다. 하지만 serving에서 `BA`를 더하는 일은 이름 해석, artifact 검증, distributed load, resident slot, continuous batch mapping, grouped kernel, cache provenance와 drain protocol의 합이다. 어느 단계도 adapter identity를 임의로 축약할 수 없다.

새 engine을 조사할 때는 먼저 merge인지 runtime overlay인지 묻는다. Runtime overlay라면 public name이 immutable content로 어떻게 해석되는지, internal id와 slot이 어떻게 분리되는지 본다. A/B shape와 scaling을 손으로 계산하고 packed target와 TP slice를 확인한다. 그 다음 request mapping이 flattened token mapping으로 확장되는 지점, sort와 scatter, base-only sentinel을 찾는다.

마지막으로 계산 밖의 상태를 심사한다. KV와 prefix cache key가 adapter content를 구분하는가, CUDA graph replay가 slot 의미 변경과 동기화되는가, unload와 LRU eviction이 running reference를 drain하는가, distributed partial failure가 rollback되는가를 확인한다. Load 성공이나 첫 token 생성은 이 계약 중 일부만 통과했다는 뜻이다.

좋은 장애 판정은 구체적이다. “같은 public name의 새 artifact가 기존 internal id를 재사용했고 prefix key는 name만 포함했다. B request는 mapping과 slot tensor는 올바랐지만 A generation의 cached K/V를 hit했다. Content hash를 key에 넣고 hot-replacement generation을 분리한 뒤 첫-layer K/V와 reference logits가 일치했다.” 이 문장은 증상, 최초 divergence, 원인, 수정과 검증을 모두 가진다.

이제 어댑터를 작은 부가 weight로만 보지 않게 됐다. 그것은 request부터 GPU row, cache와 cleanup까지 따라다니는 실행 identity다. 그 identity의 generation과 소유권을 끝까지 보존할 때만 한 GPU에서 여러 tenant와 여러 LoRA를 빠르고 안전하게 섞을 수 있다.

실무에서 가장 먼저 버려야 할 습관은 장애를 adapter file 하나로 축소하는 것이다. File hash가 맞다는 사실은 request가 그 file을 선택했다는 증거가 아니고, device에 그 tensor가 있다는 사실은 현재 token이 그 slot을 골랐다는 증거가 아니다. Mapping이 맞다는 사실도 과거 KV와 graph replay가 같은 generation이라는 증거가 아니다. 각 경계는 독립된 관찰값으로 확인해야 한다.

반대로 모든 layer output을 무작정 dump할 필요도 없다. Identity tuple, worker commit state, slot owner generation, 한 target의 A/B pattern, flattened mapping과 첫 delta row, cache provenance, final reference count를 먼저 모으면 대부분의 경쟁 가설을 빠르게 줄일 수 있다. 첫 divergence가 load 이전이라면 kernel profile은 도움이 되지 않는다. Cache entry에서 갈라졌다면 LoRA matmul을 다시 구현할 이유가 없다.

설계 review에서는 state owner를 문장으로 말하게 한다. Alias generation은 registry가, artifact validation은 loader가, resident slot은 worker manager가, request membership은 scheduler가, token permutation은 model runner와 kernel metadata가, cache provenance는 cache manager가, drain과 free는 lifecycle coordinator가 소유한다. 한 state를 둘이 소유하거나 아무도 소유하지 않는 구간이 race와 leak의 후보가 된다.

마지막 승인 문서에는 정상 경로보다 실패 경로를 더 분명히 적는다. Rank 하나가 load에 실패하면 이미 복사된 slot은 누가 되돌리는가, client timeout 뒤 operation은 누가 수렴시키는가, running request 중 unload가 오면 누가 admission을 막고 reference zero를 기다리는가, graph replay 중 hot reload가 오면 어느 event가 owner publish를 지연하는가에 답해야 한다. 답이 callback 이름 하나라면 아직 충분하지 않다. State transition, predicate와 관찰 가능한 completion을 함께 제시해야 한다.

이 조사법은 LoRA 외 adapter에도 이어진다. Prefix tuning, prompt adapter, multimodal projection patch처럼 적용 위치와 state가 달라도 public identity가 immutable content로 해석되고, request가 runtime resource를 선택하며, batch와 cache가 그 identity를 보존하고, cleanup이 reference를 닫아야 한다. 달라지는 것은 shape와 compute 경계이지 isolation을 증명하는 질문의 순서는 아니다.

완료 조건도 한 문장으로 닫을 수 있다. 마지막 running token과 GPU work가 끝났고 모든 worker가 같은 inactive generation을 보고하며, provisional allocation과 mapping row가 사라지고, 재사용 가능한 cache에는 immutable provenance가 남아 있고, slot이 다음 generation에 안전하게 넘어갔다면 unload가 끝난 것이다. API handler가 반환됐거나 dictionary에서 이름이 지워진 시점만으로는 충분하지 않다.

낯선 engine에서도 이 종료 조건을 먼저 적으면 source 탐색이 쉬워진다. Ingress에서 identity owner를 찾고, load transaction과 resident owner를 찾고, batch가 token mapping을 만드는 곳과 kernel이 소비하는 좌표를 찾은 뒤, cache provenance와 reference release를 역방향으로 따라간다. 이름이 달라도 이 ownership chain이 닫히는지를 확인하면 기능 표보다 훨씬 정확하게 multi-adapter serving의 안전성을 판단할 수 있다.

## 51.13 대표 사건의 상세 장부와 조건 카탈로그

t0 R1은 A1 slot2/gen41을 잡고 t1 prefill KV를 만든다. prefix key가 public alias `style`만 포함했다고 하자.
t2 A2가 같은 alias generation8, slot3/gen1에 commit되고 t3 R3가 A2로 들어온다. t4 R1 decode가 alias만
비교해 A2 또는 Blue B1 cache entry를 구분하지 못한다. compute slot mapping은 맞아도 cached K/V가 틀린다.

cache-off fresh run에서 A1 reference와 맞고 hit에서만 first-layer K/V가 다르면 LoRA kernel보다 prefix
provenance를 본다. key는 base revision, adapter immutable content, position/layout을 포함하고 tenant sharing
policy가 strict면 tenant salt도 필요하다. 같은 content의 cross-tenant hit도 timing leakage 정책상 금지될 수
있다.

unload A1은 registry generation7을 new admission에서 숨기되 running R1 reference가 zero가 될 때까지 slot2를
free하지 않는다. forward, output/KV commit과 cache writer completion 뒤 ref0, device free/overwrite event 뒤
slot gen42로 전환한다. dictionary delete와 API return은 terminal이 아니다.

race fixture는 unload 직후 slot2에 C1을 copy하면서 R1 old kernel completion을 지연한다. expected는 old GPU
work drain, ref0, generation increment, C1 copy/commit 순서다. old completion이 C1 ref를 release하거나 R1이
C1 tensor를 읽으면 실패다. device-wide synchronize로 숨기지 않고 exact stream/event fence를 둔다.

TP ranks0~3 중 rank2 A2 load가 실패하면 ranks0,1,3 provisional slots를 inactive rollback하고 alias generation8을
publish하지 않는다. retry는 새 operation/generation을 사용하고 old late success를 무시한다. worker inventory가
수렴하기 전 scheduler admission을 막는다.

slots8 중 active6, pinned-load1이면 evictable1이다. 동시에 A2/D1 loads가 오면 victim reservation을 원자적으로
해야 한다. 같은 slot을 두 operations가 선택하거나 capacity failure를 다른 resident adapter로 fallback해선
안 된다. wait/reject는 성능 결과이고 wrong identity 적용은 correctness 실패다.

검증 matrix는 cold/warm, LRU eviction, same-alias replacement, cross-tenant alias, prefill A1→reload→decode,
cache off/on, TP partial failure, abort/unload와 graph/eager를 포함한다. expected registry/content/slot generation,
eight-token mapping, delta sample, cache provenance와 final refs를 runtime 전에 쓴다.

rollback은 alias generation8 admission을 막고 A2 in-flight를 drain한다. A1/A2 slot, graph mapping, KV/prefix와
output generations을 content identity로 분리한다. A2 unload 전에 kernels/cache writers completion을 확인한다.
registry만 A1로 돌리고 A2 cache가 남으면 완료가 아니다.

terminal은 모든 workers가 같은 inactive/active inventory를 보고 provisional0, running refs0이며 late completion이
새 owner를 건드리지 않고 cache provenance가 request/tenant policy와 맞고 eight-token delta/reference logits가
일치하는 상태다. resident capacity, eviction/reload, latency와 goodput도 guardrail로 복원한다.

TP와 packed projection을 더 정확히 계산한다. base Llama fixture는 H4096, Q output4096, K/V output1024,
LoRA rank16이다. q_proj A는 logical `[16,4096]`, B는 `[4096,16]`; v_proj A `[16,4096]`, B
`[1024,16]`이다. fp16 payload는 q A/B 각각 128KiB, v A128KiB/B32KiB, 합 416KiB/layer다. 32 layers면
13MiB다. scaling, metadata와 alignment는 별도다.

TP4 column-parallel q/v에서 B output rows는 Q1024, V256씩 shard될 수 있다. 각 rank B payload는 q32KiB,
v8KiB다. A가 replicated라면 rank마다 q/v A256KiB가 있어 per-rank total296KiB/layer, cluster aggregate에서
replicated A가 네 번 존재한다. A도 input-parallel shard되는 implementation이면 식이 달라진다. native
parallel layer와 loader를 확인한 뒤 정확한 strategy를 manifest에 쓴다.

packed QKV destination은 global output width6144이고 TP4 local width1536일 수 있다. Q local1024, K256,
V256 segments다. source q/k/v LoRA B가 각각 올바른 local offsets에 들어가야 한다. Q/V만 target인 adapter에서
K segment는 base-only다. loader가 Q,V를 contiguous two-target buffer로 생각해 V를 Q 뒤 offset1024에 넣으면
packed layout상 K segment를 덮고 V segment는 비게 될 수 있다.

constant-pattern fixture로 잡는다. Q B rows는 1, K target 없음은 0, V B rows는 3으로 채운다. expected local
packed digest는 `[1×1024 rows,0×256,3×256]`이다. shape total1280 target rows만 검사하면 offset bug를 놓친다.
rank별 pattern에 rank+1을 더해 global/local slice 교차도 잡는다. kernel에서 first Q/V delta row를 sampling해
loader source와 연결한다.

row permutation fixture를 실제 배열로 쓴다. original slot vector `[-1,-1,2,2,2,5,3,3]`을 stable group
order base,A1,A2,B1로 정렬하면 permutation `p=[0,1,2,3,4,6,7,5]`다. sorted row j는 original row
`p[j]`에서 온다. inverse는 original0→0,1→1,2→2,3→3,4→4,5→7,6→5,7→6이다.

sorted kernel outputs를 `[b0,b1,a10,a11,a12,a20,a21,b10]`이라 쓰면 scatter 결과는
`[b0,b1,a10,a11,a12,b10,a20,a21]`이어야 한다. inverse 5와 6을 바꾸면 R2 Blue row에 A2 delta가 가고 R3
첫 row에 B1 delta가 간다. adapter tensors는 모두 올바르고 slot vector도 맞지만 output isolation은 깨진다.

prefill chunking이 있으면 한 request가 이번 step에서 연속 row 하나를 갖는지 확인한다. scheduled token 배열이
request order와 다르거나 logits rows만 subset이면 mapping은 scheduler output의 exact permutation을 따라야 한다.
padding/inactive graph rows에는 base sentinel 또는 inactive predicate가 필요하다. 이전 replay의 slot 값이 남아
있으면 inactive row가 실제 output에 기여하는 consumer가 있는지 본다.

base-only sentinel도 단순 slot0으로 만들지 않는다. slot0이 실제 adapter resident slot이면 의미가 충돌한다.
kernel metadata가 base-only를 어떤 sentinel/mask/segment로 표현하는지 source를 따른다. shrink가 unique active
adapters만 만들 때 base segment가 포함되는지, expand/scatter가 row count를 보존하는지 검산한다.

adapter rank가 서로 다르면 batching shape가 복잡해진다. A1 rank8, B1 rank16, A2 rank32라고 하자. resident
storage가 max rank32로 padding하거나 rank별 kernels를 group할 수 있다. padding은 payload/capacity를 늘리고
invalid rank lanes가 delta에 기여하지 않게 mask해야 한다. configured maximum rank와 per-adapter actual rank를
같은 값으로 덮지 않는다.

q+v layers32의 logical payload는 rank에 비례해 A1 약 6.5MiB, B1 13MiB, A2 26MiB다. slots8을 max-rank26MiB로
고정 예약하면 208MiB이고 actual 세 adapter는 45.5MiB지만 free slots reserve가 남는다. dynamic allocator면
fragmentation과 largest extent가 admission을 제한할 수 있다. resident count만으로 capacity를 설명하지 않는다.

host cache와 device pool도 나눈다. artifact A/B tensors가 host에 deserialized됐지만 device slot이 없으면 warm
host hit이지 runnable resident가 아니다. H2D copy와 all-rank commit이 남는다. load latency는 registry lookup,
artifact read/validation, host cache, device allocation/copy, distributed commit과 scheduler wait로 분해한다.

capacity incident 수치를 둔다. device budget208MiB, resident A1 6.5, B1 13, A2 26, 다섯 max-rank adapters
각 26이면 총 175.5MiB다. free32.5MiB로 D1 rank32 26MiB는 들어가지만 graph metadata/workspace12MiB를 함께
예약하면 부족하다. adapter pool만 보고 admit하면 graph capture 또는 kernel setup에서 실패한다.

active A1/B1 refs와 pending A2 unload가 있으면 LRU victim도 제한된다. cold resident count가 많아도 pinned
bytes가 budget 대부분이면 new load wait가 맞다. running identity를 희생해 slot을 덮어 성공률을 높이지 않는다.
admission은 atomic reservation 또는 bounded wait/reject를 선택한다.

batch diversity cost도 resident capacity와 다르다. resident8이어도 step unique adapters4가 sort segments와
metadata/kernel launches를 늘릴 수 있다. tokens per adapter가 `[2,3,1,2]`처럼 작으면 grouping overhead가
delta compute보다 클 수 있다. 하지만 diversity를 줄이려고 A1/A2를 alias 기준 한 group으로 합치면 wrong
weights다. performance grouping key는 immutable content/slot generation과 일치해야 한다.

graph capture는 address 안정성과 semantic generation을 분리한다. slot buffers 주소가 고정돼 graph를 replay할
수 있어도 slot2 contents가 A1→C1으로 바뀌면 request mapping generation이 updated되고 old in-flight replay가
끝났어야 한다. graph object가 slot buffer를 참조하는 동안 owner reference를 누가 갖는지 source/runtime를 본다.

mapping H2D도 generation을 가진다. host가 new vector를 만들었지만 graph replay가 old device metadata buffer를
읽거나 async copy completion 전 kernel이 launch되면 rows가 섞인다. mapping generation, copy event와 consumer
launch edge를 trace한다. slot weight generation과 mapping generation 둘 다 맞아야 한다.

prefix provenance는 K/V가 만들어진 시점의 adapter content를 따른다. R1 prefill A1이 만든 cache를 R1 decode가
읽을 때 registry current alias가 A2여도 A1 provenance가 맞다. request snapshot이 A1이기 때문이다. 반대로
R3 A2가 alias만 같다고 A1 prefix를 hit하면 틀리다. current alias보다 request immutable content를 key에 쓴다.

adapter가 Q/V만 target해도 cache sharing을 단순 허용하지 않는다. Q 변화는 current attention output/residual을
바꾸고 이후 layers K/V를 바꿀 수 있다. V 변화도 residual을 바꾼다. target이 LM head뿐인 특수 adapter는 KV가
base와 같을 가능성이 있지만 implementation이 selective layer provenance를 지원하지 않으면 conservative
adapter-wide namespace를 쓴다. source 없는 selective sharing을 발명하지 않는다.

tenant policy는 별 축이다. Red/Blue가 byte-identical content digest를 사용해도 cross-tenant cache hit latency가
prompt 존재를 누설하거나 isolation policy를 위반할 수 있다. key에 tenant salt를 넣거나 cache sharing을 disable한다.
authorization success와 KV mathematical equality를 같은 predicate로 합치지 않는다.

hot reload는 replace가 아니라 coexistence다. A1 running references가 남는 동안 A2를 별 content ID/slot에 load하고
alias generation만 new admission에 바꾼다. capacity가 coexistence 두 copies를 못 담으면 load를 wait/reject하거나
drain window를 잡는다. A1 slot을 즉시 overwrite하는 in-place reload는 stronger fencing과 no-running proof가
필요하다.

same content reload도 generation을 검토한다. content digest와 base/config가 같다면 device tensor를 재사용할 수
있지만 operation/registry generation은 바뀔 수 있다. no-op dedup을 지원하면 refcounts와 API result가 일관돼야
한다. path mtime만 바뀌었다고 slot을 churn할 필요는 없지만 content validation 없이 same alias를 no-op 처리하면
stale artifact가 남는다.

distributed rollback은 state conservation으로 검증한다. prepare 전에 free slots F, host bytes H, registry active
set R이 있다. partial load 실패 뒤 provisional slots/bytes가 0이고 F/H/R이 baseline으로 돌아와야 한다. 이미
active adapters refs는 변하지 않는다. late worker success는 operation generation mismatch로 discard/cleanup한다.

timeout은 failure 확정과 다를 수 있다. client가 timeout됐지만 coordinator가 load를 계속 commit하면 retry가
duplicate operation을 만들 수 있다. idempotency key와 status query, explicit cancel contract를 둔다. operator가
timeout 후 unload를 보내 load completion과 교차할 때 final desired generation을 state machine으로 수렴시킨다.

unload 중 abort도 역순 cleanup을 요구한다. request row를 다음 batch에서 제거하고 output/KV/cache writer를
terminal한 뒤 adapter ref를 놓는다. duplicated abort가 두 번 decrement하지 않는다. unload waiter는 ref0와 GPU
completion을 관측한다. scheduler queue에서 사라진 시각만으로 slot을 free하지 않는다.

wrong-output incident의 source walk는 ingress resolution에서 시작해 worker manager, scheduler admission,
token metadata, kernel delta, cache lookup과 unload finalizer로 내려간다. 각 단계의 content/slot generation을 같은
R1 correlation trace로 잇는다. repo별 함수명이 달라도 producer와 consumer를 둘 다 붙인다.

여기서 조사자가 실제로 남길 표는 추상적인 구성 요소 목록이 아니다. 첫 열에는 관찰 시각, 둘째 열에는 request와
tenant, 셋째 열에는 public alias와 immutable content digest, 넷째 열에는 registry generation, 다섯째 열에는
worker별 slot과 slot generation, 여섯째 열에는 token row 범위, 일곱째 열에는 cache provenance, 마지막 열에는
GPU completion과 release 상태를 적는다. R1의 한 행이 prefill과 decode 사이에서 content digest만 바뀐다면
재해석 문제이고, digest는 같은데 slot generation만 바뀐다면 lifetime 문제이며, 둘 다 같은데 K/V provenance가
다르면 cache key 또는 cache metadata 문제다. 이렇게 열마다 가능한 원인을 제한해야 로그의 양이 아니라 판별력이
커진다.

artifact 검증은 파일이 열린다는 사실에서 끝나지 않는다. manifest가 선언한 base model revision, target module,
rank, alpha, dropout의 inference 의미, fan-in/fan-out orientation, bias 정책, tensor dtype과 shape를 실제 tensor와
대조한다. q_proj용 A가 `[r,H]`인지 `[H,r]`인지 loader가 transpose하는지, B output dimension이 TP shard 이전의
global width인지 이후 local width인지 기록한다. scaling이 `alpha/r`인지 rank-stabilized variant인지도 runtime이
기대하는 식과 맞춰야 한다. 같은 이름과 같은 shape라도 scaling convention이 다르면 모든 row가 그럴듯하게 틀린다.

검증 실패는 publish 이전에 원자적으로 끝나야 한다. 예를 들어 32 layers 중 layer19 v_proj B만 width가 1025라면
앞선 18 layers를 device에 복사했더라도 active inventory에는 나타나지 않아야 한다. provisional allocation id와
operation generation을 붙여 rollback이 어느 buffer를 회수했는지 확인한다. 단순히 예외 문자열만 보존하면 late
copy completion이 회수된 주소를 건드렸는지 판단할 수 없다. copy stream event와 allocator free event의 순서를
함께 남긴다.

artifact digest의 범위도 명시한다. weight 파일 bytes만 hash하고 adapter config를 제외하면 동일 tensor에 다른
scaling이나 target mapping을 적용한 두 artifact가 같은 identity가 된다. 반대로 archive의 timestamp와 파일 순서까지
hash하면 의미가 같은 artifact가 불필요하게 다른 identity가 된다. canonical identity는 inference 의미를 결정하는
tensor content, config, base compatibility와 loader interpretation version을 포함한다. operational provenance에는
원본 URI, checksum, 서명, 수집 시각을 별도로 둔다. 의미 identity와 공급망 기록을 하나의 문자열에 우겨 넣지 않는다.

registry entry는 최소한 public scope, alias generation, immutable content identity, lifecycle state, desired replicas,
operation id를 가진다. worker inventory는 content identity, device, slot, slot generation, dtype/layout, committed state,
reference count와 last-use event를 가진다. 둘은 같은 표가 아니다. registry에 A2가 active라는 사실은 rank2 device에
A2가 있다는 증거가 아니며, worker에 tensor가 resident라는 사실은 새 request가 그것을 선택해도 된다는 뜻이 아니다.
publish barrier가 두 상태를 연결한다.

요청 snapshot에는 alias 문자열만 저장하지 않는다. authorization이 끝난 tenant scope, resolved content identity,
registry generation, base compatibility와 선택 정책을 불변 값으로 저장한다. queue wait 동안 reload가 발생해도 이
snapshot을 다시 해석하지 않는다. 다만 capacity 부족으로 아직 resident slot이 정해지지 않았다면 content identity는
고정하고 slot binding만 admission 단계에서 수행한다. content 선택과 물리 배치를 분리해야 eviction 뒤 다른 slot에
재적재돼도 의미는 유지된다.

admission에서 확인할 predicate를 순서대로 쓴다. 요청 snapshot이 아직 허용되는가, 해당 content가 모든 필요한
worker에 committed됐는가, base revision과 parallel layout이 맞는가, batch diversity와 metadata capacity가 남았는가,
slot generation reference를 원자적으로 획득했는가를 본다. 하나라도 실패하면 base model로 조용히 fallback하지 않는다.
명시적으로 base-only를 요청한 R0와 adapter를 요구했지만 준비되지 않은 요청은 의미가 전혀 다르다.

8-token fixture의 각 경계에서 기대값을 고정하면 최초 divergence를 빠르게 찾는다. ingress 뒤 request identities는
`[base,A1,B1,A2]`, scheduler 뒤 request token counts는 `[2,3,1,2]`, flatten 뒤 content rows는
`[base,base,A1,A1,A1,B1,A2,A2]`, residency bind 뒤 slots는 `[-1,-1,2,2,2,5,3,3]`다. stable sort 뒤
original row indices는 `[0,1,2,3,4,6,7,5]`, inverse scatter 뒤 request slices는 R0 `[0:2]`, R1 `[2:5]`,
R2 `[5:6]`, R3 `[6:8]`이어야 한다. 어느 배열이 다음 함수의 입력인지까지 trace에 넣는다.

첫 divergence가 flatten 단계라면 kernel을 의심하지 않는다. flatten은 맞고 device mapping buffer만 틀리면 async
copy 또는 double buffering을 본다. mapping도 맞고 sorted delta가 틀리면 slot tensor, scaling, packed offset과
kernel mask를 본다. sorted delta는 맞고 final rows만 틀리면 inverse scatter를 본다. final hidden rows가 맞지만
logits가 틀리면 이후 adapter target, logits processor와 cache reuse를 본다. 조사 순서를 이런 이분법으로 만들면
거대한 trace를 처음부터 끝까지 눈으로 훑지 않아도 된다.

reference 계산은 전체 모델 복제만을 뜻하지 않는다. 선택한 한 layer와 한 target에 대해 CPU 또는 단순 eager
matmul로 `base(x)+scale*B(Ax)`를 계산하고 runtime의 delta와 비교한다. fp16 오차를 고려한 tolerance를 정하되 wrong
adapter pattern이 tolerance에 묻히지 않도록 입력과 상수를 고른다. A1은 양수, B1은 부호 교대, A2는 서로 다른 소수
패턴을 쓰면 slot 교환, transpose와 scaling 오류가 다른 signature를 만든다. 모두 1인 tensor는 여러 오류를 같은 결과로
가릴 수 있다.

토큰별 검증에서는 평균 오차만 보지 않는다. 여덟 row 각각의 max absolute error, expected adapter identity와 observed
delta digest를 기록한다. R2 한 row의 오류는 batch 평균에서 작아 보이지만 tenant 격리 관점에서는 완전한 실패다.
prefill의 세 row만 맞고 decode 한 row가 틀리면 phase별 mapping producer 또는 graph path가 다를 가능성이 크다.
eager와 graph, prefill과 decode를 따로 교차해야 하는 이유다.

수용량을 throughput 식으로도 본다. 한 step이 총 512 tokens이고 unique adapters가 1일 때 metadata 준비 40µs와 LoRA
kernel80µs가 든다고 하자. unique adapters8에서 segment 준비가 adapter당 15µs, 작은 grouped launch가 각 25µs라면
추가 비용은 대략 105µs와 175µs가 된다. 실제 수치는 측정해야 하지만 구조는 분명하다. token 수가 같은데 diversity가
늘면 compute arithmetic보다 sorting, metadata와 launch 고정비가 지배할 수 있다. 따라서 단순 tokens/s 그래프에
unique adapter count 축이 없으면 병목을 잘못 해석한다.

반대로 한 adapter가 480 tokens, 나머지 일곱 adapter가 각 4~5 tokens인 skewed batch와 여덟 adapter가 64 tokens씩인
balanced batch는 unique count가 같아도 다르다. 작은 segment를 합치거나 지연시키면 throughput은 좋아질 수 있지만
per-tenant latency와 fairness가 악화된다. scheduler 정책은 최대 diversity, 최소 segment size, 대기 시간 budget과
SLO를 함께 드러내야 한다. 어느 정책도 content identity를 합치는 허가가 되지는 않는다.

coexistence capacity의 break-even도 계산한다. 기존 A1 6.5MiB가 pinned이고 새 A2 26MiB, transaction workspace12MiB가
필요하면 reload 순간 추가 38MiB가 필요하다. free32.5MiB인 앞 fixture에서는 5.5MiB가 부족하다. 선택지는 cold victim
하나를 evict해 여유를 만들기, A1 drain 뒤 교체하기, host에 A2를 준비하고 짧은 commit window를 잡기, 요청을 reject하는
것이다. A1을 제자리 overwrite하는 것은 capacity 해결책처럼 보이지만 in-flight correctness proof가 없으면 허용하지
않는다.

eviction cost는 bytes만이 아니다. host cache hit이면 26MiB H2D와 distributed barrier, host miss이면 artifact read와
deserialize까지 붙는다. PCIe bandwidth를 25GB/s로 가정하면 순수 26MiB copy 하한은 약 1ms지만 allocator, many-tensor
launch, synchronization과 TP fan-out이 실제 tail을 키운다. NVLink가 빠르다는 일반론으로 이 경로를 지우지 않는다.
tensor가 CPU에서 각 GPU로 가는지 한 rank에서 broadcast되는지에 따라 병목 link와 failure boundary가 달라진다.

메트릭은 원인을 보존하도록 설계한다. `adapter_load_seconds`는 phase label을 제한된 enum으로 나누고, success/failure와
reason을 붙인다. `adapter_resident_bytes`, `adapter_pinned_bytes`, `adapter_provisional_bytes`, `adapter_slots_free`,
`batch_unique_adapters`, `adapter_mapping_build_seconds`, `adapter_kernel_seconds`, `adapter_cache_hit`를 본다. 그러나
content digest나 tenant를 그대로 metric label로 넣으면 cardinality와 정보 노출이 커진다. 고유 identity는 sampled
trace/log에 넣고 metric에는 bounded class를 쓴다.

alert도 load failure 한 개로 끝내지 않는다. provisional bytes가 operation 종료 뒤 0으로 돌아오지 않음, worker inventory
generation 불일치, inactive generation reference 증가, slot owner generation mismatch, adapter-aware cache provenance
miss 급증, reload 뒤 reference logits mismatch를 각각 감시한다. correctness canary는 작은 고정 prompt와 adapter pattern을
주기적으로 실행하되 실제 tenant artifact를 노출하지 않는 합성 adapter를 사용한다.

운영 대시보드의 첫 화면은 resident count보다 lifecycle funnel이 유용하다. resolve 요청 수, admission wait/reject,
host hit, device load, distributed prepare, commit, active requests, drain, unload complete가 한 흐름으로 보여야 한다.
reload 시각에 admission wait가 늘고 provisional bytes가 남으면 capacity/rollback 문제를 의심한다. latency만 튀고 identity
canary가 정상이라면 correctness와 performance incident를 분리할 수 있다.

cache 대시보드에서는 overall hit ratio만 보면 안 된다. base-only, adapter content, tenant policy, prefill/decode와 prefix
길이 bucket별 hit/miss를 본다. alias reload 직후 old content hit가 지속되는 것은 running old generation에는 정상일 수
있다. 새 A2 request가 A1 provenance entry를 hit한 횟수가 0이어야 한다. current alias와 cache content가 다르다는 이유만으로
모든 old hit를 오류로 세면 coexistence를 오탐한다.

장애 재현은 t0부터 terminal까지 결정적으로 만든다. t0 A1 commit, t1 R1 prefill admission, t2 prefill kernel launch를
event로 멈춤, t3 A2 prepare/commit, t4 R3 admission, t5 A1 unload 요청, t6 R1 kernel release, t7 cache writer completion,
t8 ref0와 slot free, t9 C1 reuse 순서다. 각 지점에서 barrier를 제어하고 expected inventory를 적는다. 임의 sleep은 race를
재현하지 못하고 느린 CI만 만든다.

t3 직후 기대 inventory는 A1 active-but-draining ref1 slot2/gen41, A2 active ref1 또는 admission에 따라 0 slot3/gen1이다.
t5 뒤 A1은 new admission unavailable이지만 resident다. t6 뒤에도 cache writer가 reference를 갖는 설계라면 free하면 안
된다. t7 뒤 ref0가 되고 retire event가 완료된 뒤에만 t8 free가 가능하다. 이 표가 실제 implementation ref owner와
다르면 누락된 lifetime edge를 찾은 것이다.

부분 TP 실패 fixture에서는 rank2만 allocation failure를 주입한다. coordinator가 prepare 결과를 모으기 전 rank0이
alias를 publish하지 않는지, ranks0/1/3의 provisional buffer가 rollback되는지, rank2 late retry가 old operation을
commit하지 않는지 본다. 두 번째 동일 idempotency request는 첫 operation status를 반환하거나 명시한 새 generation으로
재시도해야 한다. duplicate slot과 이중 refcount가 없어야 한다.

client disconnect fixture도 필요하다. load API 연결이 끊겼다고 coordinator task를 즉시 취소하면 worker RPC 일부만
남을 수 있다. 반대로 무조건 계속하면 client가 실패로 알고 같은 artifact를 재요청한다. API contract가 disconnect 후
continue인지 cancel인지 선언하고 operation status를 조회할 수 있게 한다. 어느 쪽이든 최종 inventory가 단일 generation으로
수렴한다는 검증이 핵심이다.

rollback 판단은 먼저 new admission을 멈춘다. 이어 A2 request 수와 GPU work를 drain하고 A2 cache writer를 닫은 뒤
alias를 A1 또는 known-good content로 전환한다. 이미 A2로 생성한 응답을 A1 응답처럼 재개하지 않는다. 스트리밍 응답이
중간에 잘못된 generation을 썼다면 continuation보다 abort/retry 정책이 안전할 수 있다. 사용자-visible semantics를
명시적으로 선택한다.

복구 뒤에는 단순 성공률뿐 아니라 의미 보존을 증명한다. cold cache와 warm cache 각각에서 A1, A2, B1, base-only의
고정 입력 logits 또는 선택 layer delta를 reference와 비교한다. cross-tenant 동일 alias와 동일 digest 두 경우를 분리한다.
worker inventory digest가 모든 ranks에서 같고 provisional0, retired generation refs0, free bytes가 예산과 일치하며 late
completion counter가 증가하지 않아야 한다.

성능 terminal은 baseline과 비교한다. p50만 복원돼도 reload tail이 악화될 수 있으므로 load p95/p99, admission wait,
first-token latency, inter-token latency, goodput, eviction rate와 graph fallback 비율을 본다. correctness fix로 모든 cache와
graph를 영구 disable해 숫자를 맞췄다면 안전한 임시 완화일 수는 있어도 최종 capacity terminal은 아니다. 어떤 최적화를
언제 재활성화할지 canary 단계와 rollback threshold를 둔다.

코드 review 체크리스트는 ownership 질문으로 닫는다. public alias를 누가 immutable identity로 바꾸는가, request가 그
결과를 언제 snapshot하는가, all-worker commit 전 publish를 누가 막는가, slot generation ref를 누가 얻고 놓는가,
request rows를 token rows로 누가 확장하는가, sort와 inverse scatter의 계약은 무엇인가, cache key와 entry metadata에
어떤 provenance가 있는가, abort와 unload가 GPU completion을 어떻게 기다리는가를 묻는다. 함수 이름만 나열하는 대신
각 답에 producer, stored state, consumer와 failure predicate를 붙인다.

새 버전을 검토할 때는 source line이 이동하므로 링크의 숫자만 신뢰하지 않는다. pinned commit에서 함수와 invariant를
확인하고, upgrade commit에서는 같은 state owner와 producer-consumer chain을 다시 찾는다. 필드 이름이 바뀌어도 request
identity가 kernel row까지 보존되는지, lifetime fence가 cache writer까지 닫히는지를 검증한다. 반대로 이름이 그대로여도
semantic contract가 바뀔 수 있으므로 diff와 작은 수치 fixture를 함께 쓴다.

마지막으로 이 사고를 한 문장으로 축약하면 “adapter를 파일이나 alias가 아니라 생성 번호를 가진 실행 provenance로
취급하지 않아 생긴 혼합”이다. 하지만 좋은 보고서는 그 문장에 머물지 않는다. 어느 경계에서 A1이 A2로 바뀌었는지,
왜 기존 검사가 그것을 놓쳤는지, 여덟 row 중 어떤 row가 처음 달라졌는지, 어떤 cache entry가 잘못 재사용됐는지,
rollback이 모든 provisional state와 reference를 닫았는지까지 수치로 보여 준다. 그래야 다음 구현을 읽는 독자도 같은
방법으로 새로운 engine과 kernel을 파고들 수 있다.

독자가 처음 현장에 들어갔을 때 사용할 최소 조사 순서도 남겨 두자. 먼저 잘못된 응답 하나를 request id, tenant,
base revision, adapter alias와 요청 시각으로 고정한다. 다음으로 당시 alias가 가리킨 content generation을 registry
history에서 복구한다.

이어 모든 TP worker의 slot owner generation과 load operation 상태를 비교하고, scheduler가 만든
request-level mapping과 model runner의 token-level mapping을 확보한다. 그 뒤 선택 layer의 input, A 중간값, B delta와
scatter 후 row를 sampling한다. 마지막으로 prefix lookup key와 entry provenance, abort/unload reference를 확인한다.
앞 단계에서 divergence를 찾으면 다음 단계는 원인 확인용으로만 좁힌다.

증거를 수집할 때 개인정보와 모델 자산도 지킨다. 원문 prompt 전체나 adapter weight를 로그에 남기지 않고 request의
통제된 hash, 길이, synthetic probe 결과와 tensor digest를 쓴다. tensor digest는 dtype, shape, logical target과 함께
기록한다. raw bytes hash만 같아도 orientation interpretation이 다를 수 있고, digest만 다르면 어느 tensor가 다른지
모르기 때문이다. 필요한 full tensor dump는 접근 통제된 단기 artifact로 분리하고 보존 기간을 둔다.

로그 clock도 주의한다. rank별 wall clock이 어긋나면 load commit이 kernel launch보다 먼저인지 뒤인지 잘못 읽을 수 있다.
coordinator operation sequence, CUDA event 관계와 request-local monotonically increasing step을 사용한다. “12:00:01에
free, 12:00:00.9에 kernel” 같은 timestamp 비교보다 `kernel_done -> cache_write_done -> ref_release -> free -> reuse`
happens-before edge가 강한 증거다. 분산 trace span도 이 edge를 보조해야지 대신하지 않는다.

sampling 자체가 race를 가릴 수 있다. 모든 요청에 device synchronize와 tensor copy를 넣으면 lifetime bug가 사라진다.
평상시에는 가벼운 identity와 generation만 기록하고, 재현 fixture에서는 기존 stream ordering을 유지한 비동기 digest나
선택 row capture를 사용한다. debug mode가 실행 순서를 어떻게 바꾸는지 문서화하고, 관찰을 켠 경우와 끈 경우 모두에서
합성 canary가 실패하는지 확인한다.

모델별 target 이름 차이도 registry validation에 반영한다. 어떤 checkpoint는 `q_proj`와 `v_proj`, 다른 모델은 fused
`qkv_proj`, 또 다른 구현은 gate/up projection을 packed parameter로 노출한다. 문자열 suffix가 같다는 이유로 target을
찾으면 중복 또는 누락될 수 있다. base model의 canonical module map과 packed replacement table을 만들고 adapter manifest의
logical target을 runtime parameter slice로 변환한다. 변환 결과의 layer 수, global/local shape와 byte 합계를 load 전에
예측하고 실제 allocation과 비교한다.

vocabulary 또는 LM head adapter는 별도의 주의를 요구한다. tokenizer/base vocabulary revision이 다르면 B output rows의
뜻이 달라질 수 있다. shape가 우연히 같아도 token id semantics가 다르면 잘못된 logits delta가 적용된다. embedding과
LM head가 tied인지, tensor parallel vocabulary padding이 어떻게 들어가는지, added tokens를 runtime이 지원하는지 확인한다.
“LoRA shape valid”와 “serving tokenizer compatible”은 서로 다른 검증 항목이다.

quantized base 위 LoRA도 dtype 경계를 적는다. base projection은 int4/int8 packed weight와 scale로 계산되지만 LoRA A/B는
fp16 또는 bf16으로 남고 accumulator에서 합쳐질 수 있다. delta가 quantized output에 더해지는지 dequantized accumulator에
더해지는지에 따라 reference tolerance와 kernel fusion 위치가 달라진다. adapter를 양자화했는지, base만 양자화했는지,
activation dtype과 scaling dtype이 무엇인지 manifest와 trace에 보존한다. 단순히 “4-bit model”이라고 쓰지 않는다.

speculative decoding이나 pipeline parallel을 사용하면 identity 전파 경계가 더 늘어난다. draft와 target이 같은 adapter
semantics를 지원하는지, 검증된 tokens의 KV provenance가 어느 model/content 조합인지 확인한다. pipeline stage마다 필요한
target tensors가 다르므로 일부 stage에 adapter가 없다는 사실이 정상일 수도 있다. 그러나 stage inventory digest는 기대
target set을 기준으로 비교해야 한다. 모든 rank의 resident byte가 같아야 한다는 잘못된 invariant를 만들지 않는다.

continuous batching에서는 R1의 decode가 매 step 다른 동료와 묶인다. 첫 step은 A1만, 다음은 A1/B1, 다음은 base/A2와
함께일 수 있다. request snapshot은 같지만 flattened row offset, permutation과 unique adapter table은 매번 달라진다.
따라서 request 첫 step만 검사해서 isolation을 승인하지 않는다. 동료 조합을 바꾼 여러 decode step에서 R1 delta가 같은
reference를 유지하는지 본다. 특히 batch에서 마지막 한 row와 segment boundary를 집중 검증한다.

fairness terminal도 필요하다. max-rank adapter가 작은-rank tenant를 계속 밀어내거나 hot tenant가 slots를 모두 pin하면
correctness는 맞아도 서비스는 사용할 수 없다. tenant별 admission wait, rejection, eviction 유발량과 resident quota를
본다. quota는 public alias count가 아니라 실제 bytes, pinned lifetime과 batch diversity 비용을 고려해야 한다. 동일
content dedup을 허용해도 accounting과 isolation policy를 분리한다.

배포는 synthetic 한 번으로 끝내지 않는다. 첫 단계는 eager, cache-off, single worker에서 수치 reference를 맞춘다.
둘째는 TP와 packed targets, 셋째는 graph와 cache, 넷째는 hot reload/unload race, 다섯째는 제한된 production canary다.
각 단계에는 mismatch0, inventory convergence, provisional0와 latency threshold라는 승격 조건이 있다. 실패하면 직전
known-good 설정으로 돌아가며 cache namespace와 resident generations도 함께 정리한다.

최종 incident terminal 표에는 다섯 종류의 “0”이 들어간다. wrong identity token rows0, cross-provenance cache hits0,
retired generation 신규 admissions0, terminal operation 뒤 provisional allocations0, ref0 이후 late writes0이다. 여기에
모든 worker inventory convergence, reference delta/logits 합격, capacity와 SLO 복원을 더한다. 이 조건 중 하나를 측정할
수 없다면 해결됐다고 선언하기 전에 instrumentation debt를 별도 owner와 기한으로 남긴다.

이렇게 닫으면 adapter 서빙은 더 이상 옵션 하나를 켜는 기능이 아니다. artifact 의미를 검증하고, 분산 registry와
resident resource를 transaction으로 연결하고, 매 scheduling step에서 token row까지 identity를 보존하며, cache와 GPU
lifetime을 같은 provenance로 닫는 작은 분산 시스템이다. 그 관점이 있어야 `max_loras`, rank limit, CPU offload, fully
sharded mode와 prefix caching 같은 옵션을 서로 독립된 knob가 아니라 capacity, latency, isolation을 함께 바꾸는 설계
결정으로 읽을 수 있다.

마지막 승인 전에 작은 반증 실험을 한 번 더 한다. A1과 A2의 alias를 서로 바꾸지 않고 slot 번호만 의도적으로 교환한
negative fixture, slot은 그대로 두고 cache provenance만 잘못 주입한 fixture, inverse permutation만 한 칸 회전한 fixture를
각각 실행한다. 검사가 세 오류를 모두 잡아야 한다. 정상 fixture가 통과한다는 사실만으로는 관측 장치가 실제 혼합을
감지하는지 알 수 없다. 경보와 trace가 어느 경계의 오류인지 서로 다른 first-divergence를 가리키는지도 확인한다.

문서 review에서도 같은 반증을 적용한다. “alias가 unique하므로 안전하다”는 문장에는 tenant scope와 generation 반례를,
“slot이 pinned라 안전하다”에는 stale device mapping 반례를, “cache key에 adapter id가 있다”에는 id 재사용과 base revision
반례를 붙인다. 설명이 반례를 견디지 못하면 독자는 운영 중 처음 만난 변형에서 다시 길을 잃는다. 안전성 주장은 반드시
불변식, 깨지는 실행 순서, 관찰값과 복구 terminal까지 한 묶음으로 쓴다.

소스 인용도 장식이 아니라 탐색 출발점이다. request identity 링크에서는 생성자 필드와 equality/hash 의미를, worker
manager에서는 add/remove와 LRU·pin 전이를, scheduler에서는 admission 시점과 실패 반환을, metadata/backend에서는 row
확장과 consumer shape를 따라간다. 링크 줄 주변만 읽고 결론내리지 않고 caller와 callee, tensor producer와 kernel
consumer를 왕복한다. pinned commit은 독자가 같은 상태 기계를 재현하게 하는 좌표다.

결국 이 장의 실용적 산출물은 특정 engine의 함수 암기가 아니다. 새 코드베이스를 만나도 artifact에서 request, slot,
token row, kernel delta, KV provenance와 terminal release까지 하나의 identity chain을 그리는 능력이다. chain 중 빈 화살표가
있는 곳이 다음 source digging 지점이고, 두 owner가 같은 state를 갱신하는 곳이 race 후보이며, 숫자로 expected를 쓸 수
없는 곳이 아직 이해하지 못한 계약이다. 이 기준을 만족해야 hot reload가 빠르다는 benchmark와 tenant isolation이
안전하다는 주장 모두를 믿을 수 있다.

그러므로 릴리스 승인자는 마지막으로 한 가지를 묻는다. “이 요청의 어느 token이 어떤 immutable adapter content를,
어느 slot generation과 cache provenance로 사용했는지 사후에 재구성할 수 있는가?” 답이 alias나 현재 inventory뿐이면
승인하지 않는다. 여덟-token fixture의 모든 좌표와 lifetime terminal을 재구성할 수 있을 때 비로소 기능과 성능을 함께
검토할 토대가 생긴다.

## 51.14 소스 노트

아래 근거는 adapter artifact의 load, request identity, scheduler admission, per-token mapping과 kernel metadata를 같은 generation으로 잇는 데 쓴다. API에서 활성화됐다는 사실만으로 모든 worker와 batch row가 같은 adapter를 소비한다고 단정하지 않는다.

### Adapter가 load·activate되지 않는가

API는 성공했는데 adapter가 worker의 active set에 없을 때는 여기서 시작한다. PEFT의 load/enable state와 vLLM request identity·worker activation을 잇는다.

- [Transformers v5.15.1 — PEFT adapter load](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/peft.py#L80-L238)
- [Transformers v5.15.1 — add/set/enable/disable](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/peft.py#L390-L509)
- [vLLM v0.27.1 — LoRA request identity](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/request.py#L8-L71)
- [vLLM v0.27.1 — worker load와 LRU activation](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/worker_manager.py#L105-L319)

### Mixed batch에서 다른 adapter가 섞이는가

단독 요청은 맞지만 mixed batch나 reorder 뒤에만 출력이 틀릴 때는 여기서 시작한다. Per-token metadata, ingress resolution, scheduler admission과 memory-pool row mapping을 같은 request order로 비교한다.

- [vLLM v0.27.1 — per-token LoRA metadata](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L109-L171)
- [SGLang v0.5.18 — ingress LoRA resolution](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L3297-L3338)
- [SGLang v0.5.18 — distributed load/unload control](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_control_mixin.py#L575-L780)
- [SGLang v0.5.18 — LoRA scheduler admission](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L3453-L3495)
- [SGLang v0.5.18 — memory-pool batch preparation](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/lora/mem_pool.py#L740-L785)

### Kernel metadata와 artifact contract가 맞는가

Admission과 row mapping은 맞지만 kernel 결과 또는 다른 format에서만 차이가 날 때는 여기서 시작한다. Triton batch metadata, llama.cpp metadata key와 LoRA/Punica의 계산 계약을 함께 읽는다.

- [SGLang v0.5.18 — Triton backend batch metadata](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/lora/backend/triton_backend.py#L265-L351)
- [llama.cpp v0.2.0 — adapter metadata keys](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-arch.cpp#L380-L390)
- [LoRA paper v2](https://arxiv.org/abs/2106.09685v2)
- [Punica paper v1](https://arxiv.org/abs/2308.16369v1)
