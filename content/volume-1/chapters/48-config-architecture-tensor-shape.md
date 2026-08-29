# 48장. config의 숫자가 tensor의 모양이 되기까지

## 48.1 config shape와 effective graph가 갈린 도입 사건

Llama checkpoint를 올리자 loader가 `q_proj.weight`의 shape가 예상과 다르다고 멈췄다. 운영자는 weight 파일이 깨졌다고 생각했다. 그러나 파일의 bytes는 원본과 같았다. 문제는 `num_key_value_heads`를 잘못 고친 config였다. model class는 그 config를 믿고 K/V projection와 KV cache를 다른 폭으로 만들었고, checkpoint tensors는 원래 폭을 유지했다. 오류가 발견된 곳은 weight copy였지만 첫 divergence는 config였다.

다른 사건에서는 `tie_word_embeddings=true`가 됐다. loader는 별 `lm_head.weight`를 요구하지 않았고 embedding parameter를 output projection에 재사용했다. config가 false였다면 `[vocab_size,hidden_size]` LM-head tensor가 따로 필요하다. boolean 하나가 parameter inventory와 memory ownership, missing-weight 판정을 바꾼다.

config를 “모델 설정값 목록”으로 읽으면 이런 인과를 놓친다. `model_type`은 config class registry를 고르고 `architectures`는 causal-LM 같은 concrete model class 선택에 쓰일 수 있다. `hidden_size`, query/KV head counts와 intermediate size는 parameter dimensions를 만든다. RoPE fields는 position transform를 고르고 vocab와 tie state는 embedding/LM-head inventory를 만든다. loader는 이렇게 만들어진 expected inventory에 checkpoint names와 shapes를 맞춘다.

이 장은 교육용 Llama fixture 하나를 네 스택에 수직으로 통과시킨다. Transformers v5.15.1 commit `550d7b3834670483a4df436541272c055dc364bf`, vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, llama.cpp v0.2.0 commit `bb4caa7540188872173c44d161602d9271386413`을 고정한다.

Qwen·Gemma·Llama·MoE의 가로 비교는 52장으로 넘긴다. 여기서는 dense Llama 하나를 깊게 읽고 MoE fields는 잘못된 architecture/tensor inventory를 드러내는 경계로만 쓴다. artifact format와 shard mechanics는 49–50장, quantized packing은 46장이 소유한다.

**도입 사건: config shape는 맞았지만 effective graph가 달랐다.**

두 pod A/B의 artifact는 동일했다. raw JSON에는 H4096, L32, Nq32, Nkv8, I11008, V32000과
`tie_word_embeddings=false`가 있었다. checkpoint q/k/v와 MLP shapes도 예상과 같았고 missing/unexpected는 0이었다.
그런데 A는 정상, B는 KV allocation이 약 7% 컸고 첫 logits가 달랐다. 처음에는 loader byte corruption 또는
GPU backend numerical difference를 의심했다.

다섯-state snapshot을 비교했다. raw와 Transformers config default는 같았다. constructor attention/MLP
parameters도 같았다. B의 serving compatibility hook은 kernel tile 요건을 맞추려고 intermediate_size를
11008→11264로 config object 자체에 mutate했고, 다른 hook은 RoPE scaling absence를 새 backend default로
채웠다. MLP constructor는 mutation 전 이미 만들어져 logical weight shape11008이었지만 backend workspace와
fused kernel metadata는 mutated11264를 읽었다. RoPE consumer는 mutation 후 default를 읽었다.

즉 한 mutable config object 안에서 시간에 따라 effective graph가 갈렸다. parameter shape는 A/B 동일하고
loader도 성공했지만 B fused MLP는 physical padding256을 기대했다. loader glue가 destination padding을 만들지
않은 채 kernel metadata만 11264로 넘기면 OOB 또는 stale values 위험이 있다. 이 배포에서는 wrapper가 temporary
padded buffer를 만들어 memory가 늘었고 output divergence의 직접 원인은 별도 RoPE default였다.

tensor 수치를 계산한다. batch tokens256에서 gated MLP gate/up activation padding은 각각
`256×256×2 byte=128KiB`, 두 tensors256KiB/layer다. 32 layers에서 sequential reuse 여부에 따라 peak는 한
layer workspace일 수 있지만 graph가 layer별 static buffers를 잡으면 8MiB가 될 수 있다. weight padding이면
세 matrices의 추가 elements는 gate/up `2×256×4096`, down `4096×256`, 합 3,145,728 elements/layer,
fp16 약 6MiB/layer, 32 layers 약 192MiB다. mutation 종류를 모르면 “7% 증가”를 설명할 수 없다.

RoPE fixture는 position p와 head pair를 작은 수치로 비교한다. theta/scaling default가 달라지면 Q/K rotation
각도가 바뀌고 attention score가 첫 layer부터 갈린다. tensor shapes와 parameter bytes는 그대로다. first
divergence가 embedding 이후 Q/K RoPE output이라면 loader segment보다 position/config consumer를 본다.

첫 competing hypothesis는 corrupted weights였다. parameter digest와 q/k/v constant-pattern segment mapping이
같아 기각됐다. 둘째는 attention kernel rounding이었다. RoPE 직후, QK kernel 전부터 divergence해 기각됐다.
셋째 derived default 차이는 raw absence/value manifest와 effective RoPE parameters가 달라 지지됐다. 넷째 MLP
mutation은 output root cause는 아니지만 capacity/performance 회귀 원인이었다.

수정은 config object를 한 값으로 되돌리는 데서 끝나지 않았다. logical semantic config를 immutable하게 두고
backend physical padding을 별도 `backend_layout`/resolved state로 만들었다. constructor, loader expected inventory,
forward semantic I는 11008을 계속 사용한다. kernel storage I'=11264와 padding/mask는 backend object가 소유한다.
raw/default/derived/effective manifest에서 둘을 별 field로 기록한다.

RoPE default는 artifact absence를 backend별로 해석하지 않게 했다. architecture/config class의 공식 default를
resolved config generation에 materialize하고 모든 pods가 같은 digest를 소비한다. 새 backend가 다른 scaling을
필요로 하면 silent default가 아니라 explicit option과 compatibility validation으로 거절하거나 새 model
generation으로 승인한다.

validation fixture는 raw presence 네 경우를 둔다. explicit standard value, explicit nonstandard value, absence에
documented default, null/invalid다. default owner가 exactly once 적용되고 mutation log가 남는지 본다. 같은 raw
artifact/options에서 resolved digest가 deterministic해야 한다. process import order나 backend initialization
순서가 값을 바꾸면 실패다.

constructor fixture는 resolved config snapshot으로 parameter names/shapes/count를 생성한다. backend layout
fixture는 logical→physical padding mapping, valid slice와 kernel metadata를 생성한다. loader fixture는 original
11008 rows를 correct padded segments에 넣고 padding values를 정의한다. forward fixture는 padded lanes가
logical output에 기여하지 않고 A reference tolerance를 지키는지 예상 relation을 둔다.

TP fixture도 넣는다. global I11008, TP2 local logical5504, physical padded local5632인지, global physical11264를
나눈 5632인지 명확히 한다. rank마다 independently round하면 global padded sum이 달라질 수 있다. loader slice
range는 logical axis를 덮고 destination placement는 physical axis를 사용한다. overlap/gap과 padding mask를
검산한다.

Nkv fixture는 global8, TP1/2/4/8과 replication case를 둔다. constructor q/k/v shapes, rank-local weights,
runtime K/V heads, cache per-token bytes와 kernel num_kv_heads가 같은 resolved partition에서 나오는지 본다.
parameter loader만 맞고 cache planner가 raw/global value를 쓰면 split-brain이다.

vocabulary fixture는 V32000, padded32128, tied/untied와 missing output tensor 조합이다. embedding/LM head logical
rows, physical rows, tokenizer max ID와 sampler valid limit를 잇는다. padded logits rows는 mask돼야 하고 tie
ownership은 artifact/exporter evidence에 맞는다. load 성공만 통과 조건이 아니다.

observability에는 raw/effective full config를 metric label로 넣지 않는다. resolved-config digest, model/code/
artifact generation과 bounded mutation reason을 둔다. sampled trace에는 field-level provenance, constructor
snapshot, backend layout와 first divergent forward checkpoint를 연결한다. mutable object ID보다 immutable
generation을 쓴다.

rollout은 parser/resolver만 shadow해 old/new resolved manifest와 expected inventory/layout을 비교한다. shadow는
실제 constructor/kernel side effect를 증명하지 못한다. synthetic small model과 one-layer canary에서 mapping/
forward를 확인하고, 전체 model cohort에서 KV bytes, graph workspace, logits parity와 SLO를 본다.

자동 중단선은 unexpected resolved field diff, parameter inventory diff, loader missing/unexpected/segment mismatch,
cache byte equation mismatch, first-layer divergence, non-finite, graph/backend fallback과 SLO다. expected physical
padding 증가는 allowlist와 byte budget을 가져야 한다. “호환 mutation” 이름으로 모든 diff를 허용하지 않는다.

rollback은 effective generation을 이전 manifest로 돌리고 새 admission을 중단한다. mutated layout으로 capture한
graphs/workspaces와 caches를 old generation에 섞지 않는다. parameter storage가 같아도 RoPE/cache metadata와
physical padding contract가 다르면 active requests를 drain한다. remote/offloaded KV는 position/layout schema를
검증하고 incompatible entry는 recompute한다.

rollback terminal은 A/B resolved digest 일치, parameter/inventory closure, semantic/physical shape equations,
selected backend metadata, first-layer Q/K/RoPE와 final logits parity, KV/workspace bytes와 SLO 복원이다. JSON이
원래 값으로 보이는 것만으로는 완료가 아니다.

incident 보고의 정확한 원인은 “default가 잘못됐다”가 아니다. `raw artifact 동일 → mutable config를 constructor
전후 서로 다른 owner가 읽음 → backend가 semantic I를 physical padding 값으로 덮음 + RoPE absence를 다른
default로 resolve → parameter shape는 같지만 workspace와 forward graph 분기`다. 이 인과가 재발 방지 owner를
정한다.

최종 독자 절차는 field 하나를 세로로 완주하는 것이다. raw presence와 config class default를 찾고 derive/mutate
owner를 찾는다. constructor parameter equation을 계산하고 loader checkpoint shape/slice를 대조한다. forward와
backend가 읽는 effective value, physical layout과 cache/kernel metadata를 확인한다. 처음 값이 갈라진 consumer를
고친 뒤 downstream을 재검증한다.

이 절차를 `tie_word_embeddings`에도 대입한다. raw false면 embedding과 LM head가 독립 parameters여야 한다.
checkpoint 둘이 모두 있으면 loader는 각 owner에 넣는다. raw absence에 config default true라면 constructor가
공유 parameter 또는 tie 후처리를 만들 수 있다. exporter가 output tensor를 생략하는 convention과 loader
mutation이 결합될 수 있으므로 raw flag, tensor presence와 selected class를 함께 본다.

같은 values를 가진 두 matrices도 ownership이 다르면 tie가 아니다. pointer/shared Parameter identity,
serialization과 adapter update가 달라진다. loader가 embedding bytes를 LM head에 복사해 두 independent
parameters를 만들면 initial logits는 맞을 수 있지만 hot weight patch나 LoRA target에서 갈라진다. forward
한 번의 parity보다 lifetime mutation까지 architecture contract에 포함한다.

`attention_bias`도 shape와 의미를 나눈다. false면 q/k/v/o linears에 bias parameters가 없어 inventory가 줄고,
true면 output widths에 맞는 bias가 생긴다. loader가 bias keys를 무시하거나 zero initialize해 load를 통과할 수
있지만 checkpoint architecture와 다르다. backend fused projection이 bias를 지원/적용하는지 consumer를 본다.

`hidden_act`는 parameter shape를 바꾸지 않는다. silu와 gelu 모두 gate/up/down inventory는 같을 수 있다.
config/default가 달라져도 loader report는 완전히 동일하다. MLP intermediate checkpoint에서 activation output이
first divergence다. 그래서 shape chapter도 weight 없는 consumer checkpoints를 소유한다.

`rms_norm_eps` 역시 norm weight `[4096]`은 같다. epsilon이 raw absence에서 class version별 default로 달라지면
low-variance hidden에서 output이 달라질 수 있다. config library upgrade가 default를 바꾸었는지 pinned source
diff를 본다. 값이 작아 “수치 오차”라고 넘기지 않고 reference에 같은 epsilon을 적용한다.

`max_position_embeddings`는 weight shape보다 validation, RoPE cache와 serving admission을 바꿀 수 있다.
backend가 extrapolation을 허용하거나 CLI override로 늘리면 raw maximum과 effective accepted context가 다르다.
RoPE scaling 없이 length만 늘린 것과 scaling을 적용한 것은 다른 graph다. scheduler context limit, position IDs,
rotary consumer를 연결한다.

`pretraining_tp` 같은 compatibility field는 Transformers reference forward에서 slicing/aggregation을 바꿀 수
있고 serving native class는 무시하거나 별 TP를 사용할 수 있다. training-time semantic field와 runtime TP
world size를 같은 숫자라고 합치지 않는다. source consumer가 없으면 ignored-with-evidence로 기록한다.

field가 ignored라는 결론도 pinned class와 path에 한정한다. config object에 값이 존재하지만 constructor,
forward와 backend 어디에서도 읽지 않는지 static/source search로 확인한다. custom remote class를 켜면 같은
field가 소비될 수 있다. concrete code identity가 provenance의 일부다.

raw JSON parser가 unknown fields를 보존하는지 버리는지도 중요하다. 보존된 field가 `to_dict`에는 나타나지만
model은 읽지 않을 수 있고, strict schema가 버리면 downstream custom consumer가 정보를 잃는다. serialization
round trip 전후 raw/default를 비교한다. unknown preserved를 supported로 오해하지 않는다.

dtype fields도 state chain을 갖는다. artifact config dtype, checkpoint tensor dtype, loader target dtype,
quantization config, autocast/compute dtype와 KV dtype이 다를 수 있다. `torch_dtype=bf16` 한 field가 weights,
accumulator와 cache를 모두 뜻하지 않는다. 각 tensor/storage/kernel consumer를 나눈다.

예를 들어 fp16 weights, bf16 compute, fp8 KV이면 parameter shapes는 Llama fixture와 같지만 bytes와 scale
tensors가 다르다. config shape audit만으로 memory를 계산하지 않는다. loader가 cast하는 시각, quant scales,
backend supported dtype와 fallback을 effective manifest에 둔다.

quantization mutation은 module class 자체를 바꿀 수 있다. linear `[out,in]`의 logical shape는 같아도 packed
qweight/scales/zeros storage shapes와 loader mapping이 달라진다. raw architecture fields와 quant config를 별
schema로 두고 backend selection까지 잇는다. dense fp checkpoint shape를 packed parameter shape와 직접 비교하지
않는다.

adapter는 base config를 바꾸지 않아도 effective projection 연산과 batch grouping을 바꾼다. LoRA target q/v가
같은 base parameter shape에 low-rank delta를 적용한다. adapter revision, rank와 target modules는 request/runtime
identity다. base constructor manifest와 adapter effective graph를 구분한다.

model constructor audit의 산출물은 단순 `print(model)`이 아니다. semantic module path, parameter logical axes,
global/local/physical shapes, dtype/storage, tie alias와 source field provenance를 둔다. 같은 shape parameters가
여러 개면 module path와 segment owner로 구분한다.

loader audit은 checkpoint index와 shard 경계까지 내려가지만 49장이 byte/container를 자세히 소유한다. 48장은
expected name/shape/dtype와 mapping/slice가 architecture 계약에 맞는지까지만 다룬다. safetensors mmap과 GGUF
encoded byte mechanics를 반복하지 않는다. 장 사이 owner를 분리한다.

backend audit도 kernel 내부를 반복하지 않는다. cache heads, head dimension, layout, dtype/scales, positions와
logical/physical widths가 launcher metadata로 정확히 전달되는지 확인한다. kernel math와 transaction은 40~47장
좌표를 참조한다. 여기서는 잘못된 config가 backend predicate/input을 바꾸는 경계를 소유한다.

원장의 diff는 release artifact가 된다. old/new raw hashes가 같아도 resolved manifest가 다르면 code/options
generation change다. raw가 달라도 effective가 같을 수 있지만 왜 normalization됐는지 기록한다. silent equality도
provenance가 없으면 재현할 수 없다.

config resolver를 deterministic pure function처럼 설계하면 감사가 쉬워진다. 입력은 raw artifact, pinned code,
explicit options와 environment capabilities이고 출력은 immutable resolved semantic config와 backend layout plan이다.
runtime mutable global이나 initialization order에 의존하면 같은 입력에서 다른 digest가 나와 gate가 실패해야
한다.

capability fallback은 resolved 결과에 들어간다. requested fp8 KV가 unsupported라 fp16으로 fallback했다면
effective KV dtype/bytes와 reason을 기록한다. configured fp8 label만 dashboard에 남기면 capacity 차이를 설명할
수 없다. reject가 필요한 correctness feature와 safe performance fallback을 구분한다.

최종 source walk는 field마다 필요한 파일만 연다. class selection이면 auto config/factory, module shape면 Llama
constructor, packed mapping이면 loader, cache/kernel metadata면 serving backend를 본다. source note의 열두 링크를
매 field마다 다시 나열하지 않는다. 이 선택적 깊이가 reference-card 과밀을 막으면서 근거를 유지한다.

검증 matrix는 field 종류별로 관측 terminal이 다르다. shape-producing field인 H, Nq, Nkv, D, I, V는 parameter
inventory, local shard, packed storage와 runtime tensor/cache axes까지 확인한다. weightless-semantic field인 RoPE,
epsilon, activation, scale/window는 constructor/forward/backend checkpoint를 본다. ownership field인 tie는 alias와
update/serialization을 본다. placement field인 TP/EP와 padding은 global/logical/local/physical 네 shape를 본다.

H fixture는 4096 정상, raw absence, incompatible4100을 둔다. H가 Nq로 나누어지는지와 explicit D 관계를
validation한다. embedding, norm, q/o와 MLP input/output axes가 모두 같은 H를 소비한다. 한 module만 rounded
physical H를 쓰면 backend layout으로 분리하고 semantic H를 덮지 않는다.

Nq/Nkv fixture는 `(32,8)`, `(32,32)`, `(32,1)`, invalid `(30,8)`을 둔다. GQA group ratio가 integer인지,
q/k/v shapes와 repeat mapping, TP local heads와 KV cache byte equation을 검산한다. invalid 조합을 loader에서
늦게 발견하지 않고 resolved validation에서 거절한다. family가 non-divisible mapping을 지원하면 native source
contract를 명시한다.

D fixture는 derived128, explicit128, explicit nonstandard64와 H/Nq quotient mismatch를 둔다. explicit field를
지원하는 config/model이면 q/k projection output과 reshape가 D를 반영해야 한다. H/Nq만 다시 계산하는 backend는
capability gap이다. RoPE dimension이 full D인지 partial인지 별 field/consumer를 둔다.

I fixture는 logical11008, backend padded11264, checkpoint rows11008과 TP local5504/physical5632를 둔다. gate/up/down
source constant patterns을 packed destination logical slice에 넣고 padding이 zero 또는 kernel contract value인지
확인한다. output slice에서 padded lanes가 제거되는지 본다. graph workspace byte가 equation과 맞는지도 확인한다.

V/tie fixture는 `(32000,false,output present)`, `(32000,true,output omitted)`, contradictory combinations와 runtime
padding32128을 둔다. tokenizer highest ID, special IDs와 sampler valid range가 logical V 안에 있다. padded logits
128 rows는 masked된다. tie true에서 embedding/head alias와 adapter update가 일관되고, false에서 independent
payload를 보존한다.

RoPE fixture는 positions0,1,127,8191과 maximum boundary를 둔다. resolved theta/scaling digest, generated
position IDs와 Q/K rotary output relation을 A reference와 비교한다. tensor shape가 같다는 이유로 생략하지
않는다. cache hit/offload payload도 position/scaling generation과 호환돼야 한다.

loader mapping matrix는 shape-equal permutation을 잡는다. q source는 상수 1, k2, v3, gate4, up5로 채운다고
생각한다. packed qkv destination segments와 merged gate_up의 정확한 위치를 source mapping에서 기대한다.
TP rank별로 상수에 rank offset을 더하면 wrong slice/overlap을 찾을 수 있다. 이는 실행 결과를 주장하는 것이
아니라 구현/CI가 채워야 할 deterministic fixture다.

class matrix는 native Transformers Llama, remote custom class, vLLM native registry, SGLang native registry와
llama.cpp GGUF architecture를 같은 artifact semantics 위에 둔다. 모든 class가 같은 Python field spelling을
가질 필요는 없다. normalized semantic axes와 expected inventory/forward relation이 같아야 한다. unsupported
feature는 silent fallback보다 reject 또는 documented gap으로 둔다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- llama.cpp 경로에서는 HF raw/default manifest를 GGUF architecture keys와 hparams로 변환한 provenance를 붙인다.
- GGUF loader가 architecture를 resolve하고 model builder가 heads, embedding, feed-forward, RoPE/MoE validation을 소비한다.
- [llama.cpp architecture resolution](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model-loader.cpp#L540-L590) [llama.cpp hparams validation](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model.cpp#L1100-L1258)

HF→GGUF 변환에서 field가 absent/default였는지 converter가 materialize한 것인지 남긴다. transformed hparams가
같아도 source provenance가 다를 수 있다. output tensor omission/tie, RoPE scaling과 expert metadata는 tensor
directory와 함께 검증한다. encoded quant bytes는 49~50장 소유이고 여기서는 logical axes를 닫는다.

incident terminal matrix의 첫 행은 identity다. raw artifact/config/tokenizer/weights/code/options hashes와 selected
classes가 expected generation이다. 둘째는 resolution이다. 다섯-state manifest와 mutation reasons가 pod A/B에서
같다. 셋째 constructor다. names, logical/local/physical shapes와 alias ownership이 expected다. 넷째 loader다.
모든 source tensors/slices/segments가 closure를 이룬다.

다섯째 forward다. embedding, q/k/v before/after RoPE, cache write/read, attention output, MLP activation과 logits
shape/digest가 reference tolerance를 지킨다. 여섯째 backend다. effective head/layout/dtype/padding/position metadata와
selected kernel/fallback가 resolved manifest와 일치한다. 일곱째 serving은 KV/workspace bytes, TTFT/ITL,
deadline goodput과 terminal resource가 guardrail을 통과한다.

각 행의 failure는 다음 조사 owner를 가리킨다. identity가 다르면 배포/supply chain, resolution이면 parser/default/
mutation, constructor면 class/module, loader면 mapping/shard, forward면 weightless semantics 또는 placement,
backend면 adapter/capability, serving만 다르면 scheduler/kernel/resource를 본다. final token에서 역추측하지 않는다.

split-brain 사고의 canary 수치를 닫는다. 수정 전 B effective manifest는 semantic I11264로 잘못 표기되고 RoPE
scaling `backend_auto`였으며 KV/workspace resident가 baseline보다 7.1% 컸다. 수정 후 semantic I11008,
physical I11264가 별 field이고 RoPE는 A와 같은 explicit standard generation이다. parameter count/digest는
전후 같지만 workspace byte는 예상 physical padding 예산만 남고 first-layer RoPE/logits가 A tolerance로 돌아온다.

이 수치는 실제 실행 결과가 아니라 incident fixture의 acceptance 값임을 명시한다. production에서는 같은 열을
실측으로 채운다. source-derived expected와 runtime-observed를 구분한다. 실행하지 않은 canary가 통과했다고
쓰지 않는다. 이 책은 어떤 값을 어디서 얻고 어떻게 판정하는지 설계한다.

rollback drill은 mutation 직후, model constructed 후, weights loaded 후, graph captured 후와 active requests가
있는 시점으로 나눈다. constructor 전이면 resolved object를 폐기하면 된다. weights 후에는 parameter storage가
semantic-compatible한지 확인한다. graph/cache 후에는 layout/position generation을 drain/rebuild한다. active
requests의 old KV를 new semantics에서 재사용하지 않는다.

emergency fallback으로 previous container를 올려도 shared external KV/cache와 converted artifacts가 남을 수
있다. model/config/layout generation을 key/schema에서 확인하고 unsafe entry를 invalidate한다. old binary가 new
resolved manifest를 읽지 않게 deployment bundle을 원자적으로 고정한다. mixed pods에서는 admission을 열지
않거나 cohort를 분리한다.

rollback 뒤 raw/effective diff가 0이라는 확인만으로 부족하다. constructor inventory, loader mapping, first-layer
checkpoint와 backend/cache byte equations이 baseline으로 돌아와야 한다. mutation logs와 dashboard metric
definition도 이전 generation 의미로 맞춘다. config JSON을 되돌리고 stale graph가 남으면 rollback이 아니다.

회고 문서는 reader-first 순서를 지킨다. 먼저 “같은 config와 weights인데 pod B만 첫 token과 KV memory가
달랐다”는 장면을 쓴다. 다음 “설계도 한 장을 여러 작업자가 서로 다른 시점에 고쳐 읽었다”는 직관을 준다.
그 뒤 다섯 state, tensor equations와 source walk를 제시한다. 마지막에 검증/rollback matrix와 source note를 둔다.

reference card를 줄이는 대신 source note의 역할을 명확히 한다. note는 commit/path/symbol의 탐색 좌표다.
본문 link는 field의 생산자에서 소비자로 넘어가는 결정적 경계만 가리킨다. 독자가 모든 링크를 먼저 읽지
않아도 사건과 수치 fixture를 이해하고, 더 파고들 때 source note로 내려갈 수 있어야 한다.

최종 선택 invariant는 다음과 같다. 동일 artifact/code/options는 하나의 immutable resolved semantic config를
만들고, constructor·loader·forward·backend는 그 generation 또는 명시적으로 파생된 physical layout만 소비한다.
모든 mutation은 owner/predicate/reason과 downstream shape/rollback을 갖는다. semantic 값을 physical tuning 값으로
덮어쓰지 않는다.

독자가 이 invariant를 Nkv 하나에 적용해 raw presence, default/derive/mutation, q/k/v weight, TP slice, cache
heads/bytes와 kernel metadata까지 설명하면 장의 목표를 달성한다. I, V/tie, RoPE에도 같은 방식으로 반복할 수
있어야 한다. config는 정적 JSON이 아니라 실행 graph를 생성하는 provenance-bearing schema다.

마지막으로 실제 리뷰 대화를 재현한다. 리뷰어가 “config의 Nkv는 8이고 k_proj shape도 1024이니 정상”이라고
말하면 세 질문을 더 한다. cache planner와 attention launcher가 읽은 Nkv도 8인가. TP rank-local K/V head
정책은 partition인가 replication인가. cache byte와 block metadata가 그 정책 식과 맞는가. 세 답이 있어야
field가 constructor를 넘어 backend까지 도달했다.

“intermediate_size는 11008인데 kernel은 11264를 쓴다”는 답도 곧 오류는 아니다. semantic logical width11008과
physical padded width11264가 별 state로 존재하고 loader placement, mask/slice와 capacity가 맞으면 안전한 tuning일
수 있다. 오류는 physical 값을 semantic config에 덮어써 다른 consumer가 checkpoint나 forward 의미를 11264로
해석하게 만드는 것이다. 이름보다 owner와 axis를 본다.

“RoPE는 tensor shape를 바꾸지 않는다”는 답에는 first-layer forward checkpoint를 요구한다. raw/default theta와
scaling, position IDs, rotary dimension, backend implementation과 cache generation이 맞는지 본다. weight digest와
loader closure가 완벽해도 이 행이 다르면 output은 달라진다. shape audit을 semantic audit으로 확장하는 대표
field다.

“tie는 파일 크기만 바꾼다”는 답에는 alias ownership과 update를 요구한다. embedding/head values, shared storage,
adapter target, serialization과 sampler valid vocabulary를 잇는다. output tensor absence가 intentional convention인지
corruption인지 config flag 하나로 단정하지 않는다. exporter/converter와 model class의 합의를 본다.

source diff 절차는 old/new pinned revision에서 field producer와 consumers를 `rg`로 찾고 caller를 따라간다.
constructor signature default가 바뀌었는지, derived property가 추가됐는지, compatibility mutation predicate가
넓어졌는지, backend가 config object 대신 resolved object를 읽는지 비교한다. 문서 default만 diff하지 않는다.

새 field가 추가됐는데 checkpoint inventory가 같다면 weightless semantic 또는 backend tuning field일 수 있다.
읽는 곳이 없다면 inert metadata일 수 있다. 반대로 field가 삭제돼도 derived default가 같은 effective 결과를
만들 수 있다. field count나 JSON diff 크기로 위험을 평가하지 않는다. consumer impact radius가 기준이다.

resolved manifest에는 각 field value 외에 `origin=raw|default|derived|override|compatibility`, owner symbol,
source revision과 derivation inputs를 둔다. physical layout에는 logical source field와 padding/partition equation을
둔다. digest는 deterministic serialization에서 만든다. 비밀 경로나 host-specific pointer를 포함해 pod마다
digest가 달라지지 않게 한다.

manifest를 reader-facing 책에 그대로 덤프할 필요는 없다. 독자는 한 Nkv/I/RoPE 사건의 필요한 행만 본다.
운영 artifact는 전체 manifest를 저장하고 source note가 재구성 좌표를 제공한다. tutorial과 exhaustive reference의
경계를 이렇게 나눈다. 본문은 질문과 인과를, machine artifact는 전수 필드를 소유한다.

metric은 resolved-generation별 load success, mutation reason, fallback, cache byte mismatch와 forward anomaly를
집계한다. raw field values나 model-private config를 고 cardinality label로 넣지 않는다. detailed diff는 trace/
artifact link로 이동한다. generation cohort가 섞이면 A/B 성능과 correctness를 분리할 수 없다.

incident 영향 범위는 same artifact 전체가 아니라 divergent resolved generation을 소비한 requests다. 다만 trace
retention이 부족하면 deployment time, pod/code/options와 external cache sharing을 기준으로 보수 확대한다.
source 가능성만으로 customer impact를 확정하지 않고 output/reference evidence를 붙인다.

수정 후 새 default를 raw config에 materialize하는 선택은 재현성을 높일 수 있지만 upstream artifact 변경이다.
기존 model publisher semantics와 맞는지 확인하고 artifact revision/hash를 올린다. engine-local resolver 수정이면
raw artifact는 유지하고 code/options generation을 올린다. 두 수정을 동시에 해 원인을 숨기지 않는다.

backend padding을 manifest로 분리했더라도 kernel upgrade가 padding multiple을 256→512로 바꾸면 physical bytes와
graph capacity가 달라진다. semantic model generation은 같고 backend layout generation만 달라질 수 있다.
cache payload가 semantic layout인지 packed backend-specific인지에 따라 호환 범위를 정한다.

최종 verification lab은 config-only dry resolution, synthetic constructor/inventory, constant-pattern loader mapping,
one-layer semantic checkpoints와 serving metadata/byte equations 순서다. 앞 단계가 실패하면 뒤 거대 model
benchmark를 돌리지 않는다. 실행 환경이 없으면 source-derived expected와 필요한 probe를 남기고 runtime pass를
주장하지 않는다.

lab의 exit condition은 같은 input manifest가 pod A/B에서 같은 resolved digest와 module inventory를 만들고,
field별 semantic/physical relation과 loader/backend consumers가 closure를 이루는 것이다. deliberate mutation
fixture에서는 expected digest/shape/byte가 정확히 바뀌고 rollback 뒤 baseline이 복원돼야 한다.

이제 “config는 맞다”는 문장을 더 정확히 바꿀 수 있다. raw artifact가 의도한 값을 담았고, absence/default가
고정 class에서 해석됐으며, derived/mutation이 provenance를 갖고, constructor와 loader가 같은 semantic graph를
만들고, backend는 명시적 physical layout만 파생해 forward/cache/kernel이 동일 generation을 소비한다. 이 긴
문장이 짧은 JSON diff보다 실제 서빙 모델을 설명한다.

upgrade 승인자는 이 문장을 field 세 개로 샘플링하지 않는다. shape와 weightless semantics, ownership,
parallel/physical layout에서 대표 field를 고르고 전체 resolved-manifest diff를 자동 검사한다. 예상 diff는 owner와
effect budget을 가져야 하고 예상하지 않은 diff는 값이 같아 보여도 provenance 변화를 조사한다. default owner가
native class에서 remote class로 바뀌면 현재 값이 같아도 다음 absent field의 해석 위험이 달라진다.

config와 code revision의 조합도 artifact다. Transformers package만 pin하고 remote repository revision을
`main`으로 두거나, serving registry plugin을 floating으로 두면 resolved graph를 재현할 수 없다. imported
source digest와 native extension/backend version을 manifest에 포함한다. security/supply-chain 상세는 76장으로
넘기지만 architecture code identity는 이 장의 shape provenance다.

같은 field가 여러 곳에서 계산되면 single source of truth를 만들되 verification equation은 독립적으로 유지한다.
예를 들어 cache planner가 model attention object의 effective Nkv/D를 읽도록 통일해 split-brain을 줄인다.
동시에 raw config와 checkpoint k_proj shape에서 expected Nkv를 역검산한다. 한 잘못된 resolved object를 모든
consumer가 일관되게 읽는다고 correctness가 보장되는 것은 아니다.

역검산은 모호할 수 있다. k_proj output1024와 D128이면 Nkv8을 얻지만 packed/quantized layout, transpose,
latent projection에서는 단순 division이 틀릴 수 있다. concrete class/loader convention을 먼저 확인한다.
equation이 적용되지 않는 architecture에 Llama fixture를 강제하지 않는다. 52장의 family 비교에서 축을
확장하는 이유다.

incident 회고에서 MLP physical padding과 RoPE semantic default를 분리한 것이 중요하다. 둘은 같은 mutable
config mutation에서 발견됐지만 하나는 capacity/performance, 다른 하나는 correctness의 root cause였다. 한
증상 묶음에 root cause 하나만 강제하지 않는다. 각 first divergence와 downstream effect를 별 인과 사슬로
닫고 rollback이 둘 모두를 제거했는지 본다.

독자가 새 config를 받으면 먼저 모든 field를 외우려 하지 않는다. embedding/QKV/MLP/logits/cache라는 주요
tensor 축과 position·norm·activation 같은 weightless 축을 고른다. 각 축에서 raw→effective→constructor→loader→
forward/backend를 한 번 완주한다. 그 후 관련 fields를 확장하면 reference-card 더미보다 빠르게 model의 설계
의도와 위험 경계를 파악할 수 있다.

마지막 handoff에는 unresolved도 남긴다. source에서 mutation consumer는 찾았지만 runtime option resolution을
확인하지 못했거나, packed extension 내부 segment order를 binary에서 확인하지 못했다면 gap과 필요한 probe를
명시한다. expected source behavior를 observed deployment fact로 승격하지 않는다. 이 정직한 빈칸이 다음
조사자의 breakpoint가 된다.

반대로 검증된 행은 재사용할 수 있다. 같은 resolved generation과 concrete class, inventory/layout digest가
유지된다면 다음 성능 사건에서 config shape 가설을 빠르게 낮출 수 있다. 단, adapter, quantization, TP와 backend
generation이 바뀌면 영향 축을 다시 연다. 증거의 scope를 넘겨 상속하지 않는다.

48장의 종료 조건은 오류 없이 load되는 모델이 아니다. 어떤 raw 사실과 absence가 어떤 code에서 해석되고,
그 결과가 어느 semantic tensor와 physical storage, loader slice, forward/cache/backend consumer를 만들었는지
수치로 재구성되며, mutation과 rollback 뒤 같은 사슬을 다시 검증할 수 있는 모델이다.

## 48.2 사건의 raw config를 effective field로 확정한다

### 48.2.1 같은 파일에도 세 generation이 있다

첫 generation은 artifact의 raw metadata다. Hugging Face checkpoint에는 `config.json` fields가 있고 GGUF에는 typed key/value metadata와 tensor directory가 있다. 둘째는 library가 defaults, compatibility normalization와 user overrides를 적용한 effective config다. 셋째는 selected model class가 effective config로 만든 actual modules, parameters와 derived shapes다.

세 generation를 섞으면 “config 파일에는 false인데 model은 tied다” 같은 모순이 생긴다. loader가 in-memory config를 mutation했을 수 있고 command-line override가 적용됐을 수 있다. raw 파일이 바뀐 것이 아니라 effective state가 달라졌다. incident record에는 raw value, mutation owner/reason와 final value를 모두 둔다.

checkpoint tensor inventory는 넷째 독립 generation처럼 다룬다. constructed parameters가 기대한 names/shapes와 artifact가 제공하는 names/shapes를 loader가 reconcile한다. config와 weights가 같은 release에서 왔다는 가정이 깨지면 class construction는 성공해도 load에서 실패한다.

### 48.2.2 교육용 Llama fixture

fixture는 `model_type="llama"`, `architectures=["LlamaForCausalLM"]`, vocab `V=32,000`, hidden `d=4,096`, intermediate `d_ff=11,008`, layers `L=32`, query heads `Hq=32`, KV heads `Hkv=8`, untied embeddings와 default RoPE theta 10,000을 쓴다.

head dimension는 `Dh=d/Hq=4096/32=128`이다. GQA group size는 `Hq/Hkv=4`다. query heads 네 개가 한 KV head group에 대응하는 형태다. 이 관계를 구현이 허용하고 tensor-parallel partition와 맞는지 별 검증한다.

이 숫자는 특정 상용 checkpoint의 완전한 config가 아니다. shape 계산을 닫기 위한 fixture다. activation function, norm epsilon, BOS/EOS와 quantization 같은 다른 fields는 해당 shape 질문에 필요할 때만 추가한다.

### 48.2.3 default는 정보의 부재를 지우지 않는다

Transformers `LlamaConfig`는 `num_key_value_heads`가 없으면 query head count로 채울 수 있다. 이는 legacy multi-head attention compatibility를 제공하지만 raw field가 없었다는 사실은 보존해야 한다. explicit 32와 missing→default 32는 effective shape는 같아도 provenance가 다르다.

RoPE base/scaling도 optional defaults를 가질 수 있다. llama.cpp는 missing KV-head count를 total head count로 default하고 RoPE base를 10,000으로 시작할 수 있다. 서로 다른 formats/libraries가 같은 default를 갖는지 확인한다. config conversion에서 field를 누락해도 우연히 같은 default가 나오는 경우와 의미가 달라지는 경우가 있다.

default를 적용한 effective config를 로그에 남기지 않으면 later library upgrade에서 default change를 checkpoint regression로 오인할 수 있다. raw/effective diff를 release artifact로 보존하는 이유다.

## 48.3 effective config가 config class와 model class를 고른다

### 48.3.1 model_type은 config class를 고른다

Transformers `AutoConfig.from_pretrained`는 config dict를 읽고 `model_type`이 local mapping에 있으면 대응 config class를 선택한다. fixture의 `llama`는 `LlamaConfig`로 resolve된다. unknown type이면 mapping failure가 나거나 remote auto-map 조건을 검토한다.

config class는 fields를 typed attributes와 defaults/validation로 바꾼다. LlamaConfig는 hidden/head divisibility를 검사하고 missing KV heads를 default한다. raw JSON dictionary가 곧 model parameters가 아니라 config constructor가 normalized state를 만든다.

### 48.3.2 architectures는 model head/class preference다

`architectures=["LlamaForCausalLM"]`은 base decoder만이 아니라 causal language-model head를 가진 class를 선택하는 힌트다. Auto model factory는 config class mapping 안에서 architectures names를 살펴 compatible concrete class를 고를 수 있다. sequence classification나 base model이면 expected output head와 parameter inventory가 달라진다.

`model_type=llama`와 architectures가 모순되면 어느 mapping가 우선하는지 stack별 source를 읽는다. config class는 Llama인데 serving registry가 Qwen class를 선택하면 attribute expectations와 weight names가 충돌할 수 있다. raw strings만 보고 성공을 예측하지 않는다.

vLLM과 SGLang은 Transformers model class를 그대로 실행한다고 쓰지 않는다. HF config와 architecture spelling을 입력으로 native serving registry에서 class를 고른다. selected vLLM/SGLang Llama implementation가 TP-aware parameters와 KV cache-facing shapes를 만든다.

### 48.3.3 trust_remote_code는 code identity를 바꾼다

`trust_remote_code` 입력은 local code와 remote auto-map availability를 검사하는 resolved trust state가 된다. remote code가 있고 trust가 허용되며 explicit local path가 우선하지 않으면 dynamic module에서 config/model class를 가져올 수 있다. flag는 단순 지원 switch가 아니라 실행할 Python code의 identity와 parameter construction를 바꾼다.

option 연쇄를 닫는다. user flag→resolved trust decision→local/remote predicate→dynamic/native class selection→constructed parameter names/shapes→loader result와 imported code revision이다. observability에는 repository/revision, selected qualified class와 trust decision를 남긴다.

security와 reproducibility도 직접 연결된다. remote class는 checkpoint repository의 code를 실행할 수 있으므로 revision pin과 review가 필요하다. native class와 같은 config fields를 읽더라도 custom parameter naming/forward/cache layout를 만들 수 있다. trust on/off 결과를 같은 class라고 가정하지 않는다.

### 48.3.4 모순 fixture를 class 선택 단계에서 멈춘다

raw config가 `model_type=llama`, architectures `Qwen3ForCausalLM`이라고 하자. Transformers config는 LlamaConfig를 만들 수 있지만 model factory/native registry는 architectures를 따라 다른 class 후보를 찾을 수 있다. 첫 관측은 selected config class와 model class다.

selected model가 Qwen attributes를 요구하는데 LlamaConfig에 없다면 construction에서 실패할 수 있다. construction가 defaults로 통과해도 parameter names/shapes가 checkpoint와 달라 load에서 missing/unexpected가 난다. load error를 weight corruption로 바로 분류하지 않는다.

수정은 오류가 사라지는 class를 임의로 고르는 것이 아니다. artifact producer가 의도한 architecture와 weights inventory를 확인하고 raw metadata를 바로잡는다. remote code를 켜 우회할 때는 code revision와 schema contract가 맞는지 증명한다.

## 48.4 선택된 class가 embedding·QKV tensor shape를 만든다

### 48.4.1 embedding는 vocab×hidden이다

token embedding table는 logical shape `[V,d]=[32,000,4,096]`이다. checkpoint orientation와 framework parameter convention를 확인한다. token ID는 row index가 되고 output hidden vector 폭은 4,096이다.

vocab 32,000은 tokenizer가 낼 수 있는 IDs와 맞아야 한다. tokenizer vocabulary가 더 크면 out-of-range lookup가 날 수 있고 checkpoint embedding rows가 32,064처럼 padded돼 있으면 loader가 padding를 허용/trim하는 정책인지 확인한다. 단순 unequal만으로 corruption를 단정하지 않는다.

tensor parallel에서 embedding rows 또는 hidden dimension가 shards로 나뉠 수 있다. global expected shape와 local parameter shape를 구분한다. checkpoint full tensor를 loader가 shard parameter에 slice하는지 pre-sharded artifact를 읽는지도 49–50장 format/loader 경계와 연결한다.

### 48.4.2 Q는 4096, K/V는 각각 1024 폭이다

query projection output width는 `Hq×Dh=32×128=4,096`이다. key와 value는 각각 `Hkv×Dh=8×128=1,024`다. separate linear weights의 logical output widths는 Q 4,096, K 1,024, V 1,024다.

packed QKV parameter가 이 outputs를 한 축에 이어 붙이면 fused output width는 `4,096+1,024+1,024=6,144`다. input width는 hidden 4,096이다. weight storage orientation가 `[out,in]`인지 loader/linear abstraction convention를 확인한다. 숫자 6,144만 맞아도 Q/K/V segment order가 틀리면 wrong result다.

GQA가 KV cache bytes를 줄이는 이유도 shape로 보인다. MHA Hkv=32였다면 token/layer당 K+V elements는 `2×32×128=8,192`다. fixture Hkv=8이면 `2×8×128=2,048` elements다. dtype bytes와 layers/pages를 곱해야 physical cache bytes가 된다.

### 48.4.3 TP partition는 global heads를 local heads로 바꾼다

TP size 4를 교육용으로 두면 query heads는 rank당 8, KV heads는 rank당 2로 나눌 수 있다. local Q width 1,024, local K/V width 각각 256이다. packed local output width는 1,536이다. 실제 parallel layer의 partition/replication policy를 source에서 확인한다.

Hkv가 TP size보다 작거나 나누어지지 않으면 KV heads를 replicate하거나 validation failure/다른 mapping가 필요할 수 있다. `Hq=32,Hkv=7,TP=4`는 단순 integer division로 local heads를 만들 수 없다. silent truncation하면 projection/cache shape가 깨진다.

vLLM과 SGLang attention constructors는 total heads, KV heads, TP group를 읽어 local counts와 QKV parallel projection를 만든다. global config 값과 runtime local cache shape를 같은 숫자로 쓰지 않는다.

### 48.4.4 attention output projection는 hidden으로 돌아온다

attention heads output를 concat한 query hidden width는 4,096이고 output projection는 model hidden 4,096으로 보낸다. logical dense shape는 `[d,d]`지만 storage orientation, TP row/column parallel split를 source에서 확인한다.

K/V heads가 8이라고 attention output width가 1,024인 것은 아니다. query heads 32개가 output features를 만들고 GQA는 K/V를 그룹 공유한다. Hkv를 output projection input width에 잘못 쓰면 constructed shape부터 틀린다.

### 48.4.5 KV cache layout는 parameter shape의 downstream 결과다

per layer/token logical K shape는 `[Hkv,Dh]=[8,128]`, V도 같다. batch/sequence/page/block axes와 dtype/layout는 backend cache manager가 더한다. 예를 들어 page capacity P라면 conceptual block에 K/V `[P,Hkv,Dh]` axes가 있을 수 있지만 actual physical order는 backend source를 따른다.

KV head mismatch는 loader에서 K/V weight shape error로 발견될 수도 있고 load를 억지로 통과하면 cache allocation/write/read mismatch로 이어질 수 있다. config correction는 constructed weights뿐 아니라 cache byte calculation와 attention backend specialization를 함께 갱신한다.

## 48.5 RoPE와 MLP·LM head까지 shape 사슬을 닫는다

### 48.5.1 RoPE dimension는 head dimension와 맞아야 한다

fixture head dimension 128에서 default Llama RoPE는 key/query head의 rotary dimensions에 적용된다. exact full/partial rotary dimension는 config/source를 따른다. llama.cpp Llama path는 invalid `n_rot`가 expected key head dimension와 다르면 오류를 낼 수 있다.

theta/base 10,000과 scaling type/factor/original context는 parameter matrix shape를 바꾸지 않을 수 있지만 position transform와 supported kernel path를 바꾼다. “shape 장”에서 다루는 이유는 effective head/rotary dimensions와 dispatch contract에 묶이기 때문이다.

legacy `rope_scaling` dictionary와 newer rope parameters spelling가 충돌하면 normalization precedence를 확인한다. raw fields 둘을 동시에 소비하지 않는다. effective rope type/base/factor를 한 객체로 기록한다.

### 48.5.2 gated MLP는 intermediate width를 세 tensor에 반영한다

Llama gated MLP에는 gate와 up projections가 hidden 4,096에서 intermediate 11,008로 가고 down projection가 11,008에서 4,096으로 돌아온다. checkpoint `[out,in]` convention라면 gate/up `[11,008,4,096]`, down `[4,096,11,008]`이다.

vLLM/SGLang은 gate와 up를 packed parallel parameter로 만들 수 있다. packed output width는 `2×11,008=22,016`이지만 checkpoint에는 separate names가 있을 수 있다. loader mapping가 shard ID와 packed segment를 정확히 지정해야 한다.

intermediate_size를 잘못 바꾸면 세 weights 모두 shape mismatch를 낸다. 첫 reported tensor만 고치지 않는다. constructed MLP inventory와 checkpoint three tensors를 함께 비교한다.

### 48.5.3 untied LM head와 tied LM head

fixture `tie_word_embeddings=false`이면 LM head logical shape `[V,d]=[32,000,4,096]`가 별 parameter로 필요하다. final hidden `[tokens,d]`를 vocab logits `[tokens,V]`로 projection한다.

tie true면 embedding weight를 LM head에 reuse할 수 있다. loader는 separate `lm_head.weight` missing를 정상으로 취급하거나 checkpoint에 duplicate tensor가 있으면 skip/alias policy를 적용할 수 있다. tie는 값이 우연히 같다는 뜻이 아니라 storage/parameter ownership contract다.

SGLang GGUF loader가 output tensor 부재를 보고 effective `tie_word_embeddings=true`로 mutation하는 path가 있다. raw config→file tensor presence predicate→effective tie state→constructed/load inventory 변화로 기록한다. 원본 metadata가 자동 수정됐다고 쓰지 않는다.

### 48.5.4 MoE fields는 다른 tensor rank를 요구한다

dense fixture에는 expert count/top-k/group fields가 absent 또는 zero baseline이다. expert count가 positive면 router와 per-expert gate/up/down tensors, selected expert count 등의 validation가 필요하다. dense Llama class에 expert tensors를 남기면 unexpected weights가 되고 MoE class인데 tensors가 없으면 missing weights다.

llama.cpp hparams는 expert-used count가 total experts를 넘지 않고 group divisions가 맞는지 assertions를 가진다. config 숫자만 유효해도 selected architecture class가 expert tensor naming/rank를 지원해야 한다. 52장에서 families를 비교하고 여기서는 mismatch 경계만 고정한다.

## 48.6 Transformers에서 serving registry까지 class가 갈라진다

### 48.6.1 Transformers는 config class와 model class를 나눠 고른다

AutoConfig 단계는 `model_type`에서 LlamaConfig를 만든다. AutoModelForCausalLM 같은 factory는 config class mapping와 `architectures` preference로 LlamaForCausalLM을 고른다. base LlamaModel과 causal-LM wrapper는 embedding/layers는 공유해도 output head inventory가 다르다.

factory가 remote class를 고르면 qualified class와 module revision가 달라진다. `architectures` string가 같아도 auto-map가 custom class path를 제공할 수 있다. selected class를 로그에 남기지 않고 model name만 기록하면 재현할 수 없다.

class construction 뒤 pretrained loader는 checkpoint state dict와 parameters를 맞춘다. missing/unexpected keys와 size mismatch는 config-derived inventory와 artifact inventory 사이 차이다. 오류 목록은 결과이지 원인을 config, class, conversion와 corrupted shard 중에서 더 좁혀야 한다.

### 48.6.2 vLLM registry는 native serving class를 고른다

vLLM loader utility는 `hf_config.architectures` 목록을 model registry에 전달하고 resolved model class/architecture를 얻는다. fixture는 vLLM의 `LlamaForCausalLM` implementation로 resolve돼야 한다. supported 목록에 없거나 task가 다르면 resolution failure 또는 다른 fallback가 생길 수 있다.

vLLM Llama MLP는 gate-up parallel projection와 down projection를 만들고 attention는 total query/KV heads와 TP size로 parallel QKV parameters를 만든다. config shape가 native serving parameter layout로 변환되는 지점이다.

LlamaModel은 vocabulary embedding와 decoder layer stack을 만들고 LlamaForCausalLM은 output head/tie behavior를 추가한다. config tie가 true일 때 last pipeline rank의 embedding/LM-head ownership가 달라질 수 있다. global model inventory와 PP rank별 local inventory를 구분한다.

### 48.6.3 vLLM weight mapping는 separate checkpoint를 packed parameter로 넣는다

checkpoint는 `q_proj`, `k_proj`, `v_proj` separate names를 가질 수 있지만 serving parameter는 packed QKV일 수 있다. mapping는 name를 target packed parameter와 shard identifier로 바꾸고 weight loader가 correct segment/TP slice에 복사한다.

MLP gate/up도 같은 구조다. two checkpoint tensors가 one gate-up parameter의 서로 다른 segments에 들어간다. loaded parameter set를 단순 name count로 비교하지 않는다. source-to-target many-to-one mapping와 packed offsets를 기록한다.

tie true이면 `lm_head.weight`를 skip할 수 있다. checkpoint에 tensor가 있어도 duplicate/unused로 처리될 수 있고 없더라도 정상이다. tie false인데 skip하면 logits head가 uninitialized다. predicate는 effective config tie를 사용해야 한다.

### 48.6.4 SGLang registry와 native Llama class

SGLang loader utility도 HF architectures를 읽고 native registry support를 검사한 뒤 model class를 resolve한다. transformers backend fallback나 architecture rewrites가 있을 수 있으므로 raw first string와 effective architecture list를 나눈다.

SGLang LlamaAttention는 config query/KV heads와 TP를 소비해 local KV heads, QKV projection와 attention layer를 만든다. MLP는 intermediate width를 packed gate-up/down에 반영한다. causal-LM wrapper는 tie condition에 따라 output head를 구성한다.

load_weights는 q/k/v와 gate/up mappings, PP missing layers와 tie skip를 다룬다. post-load hook가 있는 class는 weights copy 뒤 derived state를 만들 수 있다. load function return만으로 model-ready를 선언하지 않는다.

### 48.6.5 loader별 후처리 경계

SGLang default loader의 `load_weights_and_postprocess`는 weights iterator를 model-specific `load_weights`에 전달하고 post-load path를 실행한다. quantized/remote/sharded loaders가 model load_weights를 bypass하면 explicit post-load fixup가 필요할 수 있다.

vLLM의 loader도 model class가 제공하는 mapping/loader helpers와 checkpoint iterator를 결합한다. format별 reader는 다르지만 constructed parameter inventory를 충족해야 한다. 49–50장은 file discovery/shards/serialization를, 이 장은 resulting name/shape contract를 본다.

## 48.7 llama.cpp에서 GGUF metadata가 hparams와 tensors가 된다

### 48.7.1 GGUF에는 model_type과 architectures 대신 architecture key가 있다

llama.cpp는 `general.architecture` string를 읽어 internal `llm_arch` enum를 고른다. fixture GGUF의 value `llama`는 `LLM_ARCH_LLAMA`가 된다. enum는 architecture-specific metadata key templates와 tensor names를 고르는 dispatch state다.

HF JSON의 `model_type="llama"`, architectures list와 의미가 비슷한 부분은 있지만 동일 fields가 아니다. 변환 도구가 HF config/weights를 GGUF keys와 canonical tensor names로 옮긴다. conversion correctness는 49–50장에서 다룬다.

### 48.7.2 key template가 hparams를 채운다

llama.cpp architecture tables는 embedding length, block count, attention head/KV-head count, FFN length와 RoPE keys 같은 format strings를 정의한다. selected arch가 `%s`에 들어가 `llama.embedding_length` 같은 concrete key를 찾는다.

model hparams loader는 embedding/layer/head counts를 읽고 optional KV-head count가 없으면 query heads로 default한다. RoPE base/scaling와 key/value head dimensions도 defaults/metadata로 채운다. expert fields를 읽어 dense/MoE invariants를 검사한다.

head arrays가 layer별 values를 허용할 수 있으므로 scalar HF fixture와 차이를 주의한다. effective hparams가 per-layer인지 global인지 log에서 확인한다. cache shape도 layer별 KV heads/SWA layout가 달라질 수 있다.

### 48.7.3 hparams가 expected tensor names와 shapes를 만든다

architecture-specific switch는 token embedding, per-layer attention norm/Q/K/V/O와 FFN tensors, final norm/output tensors를 `create_tensor`로 요구한다. required/optional flags가 missing tensor 판정을 정한다. expected shape는 hparams에서 계산된다.

GGUF tensor directory에 name가 있어도 dimensions가 expected와 다르면 load가 실패해야 한다. quantized type는 logical dimensions와 storage bytes가 다르지만 shape invariant는 유지한다. quant block compatibility는 46/50장으로 넘긴다.

output tensor가 absent한 architecture/checkpoint에서 token embedding를 tied output으로 사용할 수 있는지 loader/model source를 확인한다. SGLang의 GGUF reader가 tie config를 mutation하는 것과 llama.cpp native GGUF semantics를 동일 구현으로 쓰지 않는다.

### 48.7.4 llama.cpp validation는 늦은 crash를 막는다

Llama에서 rotary dimension가 key head dimension와 맞지 않으면 runtime_error가 날 수 있다. expert used/total/group relations는 assertions로 막는다. fused QKV tensor가 있는 architecture에서는 expected partition sum가 actual tensor axis와 맞는지 검사할 수 있다.

이 validation가 없거나 우회되면 graph construction 또는 kernel에서 dimension mismatch가 늦게 드러난다. metadata load 직후 effective hparams와 expected inventory를 dump해 file tensors와 비교하는 편이 원인을 빨리 찾는다.

## 48.8 mismatch 사건을 첫 shape divergence로 조사한다

**사건 1 — `model_type`와 `architectures`가 다르다.**

증상은 unknown architecture, missing attributes 또는 대량 missing/unexpected weights다. raw `model_type`, architectures/auto_map와 selected config/model qualified classes를 기록한다. remote trust decision와 serving registry resolution도 포함한다.

config class부터 예상과 다르면 registry mapping/remote code를 본다. config는 Llama인데 model가 Qwen이면 architectures metadata를 본다. classes는 모두 Llama인데 tensors가 다르면 conversion/revision/shape fields로 이동한다.

고치는 방법은 architectures string 하나를 성공할 때까지 바꾸는 것이 아니다. checkpoint tensor naming와 producer model class를 확인하고 coherent metadata set를 복구한다. selected class와 load inventory가 함께 맞아야 한다.

**사건 2 — hidden size가 heads로 나누어지지 않는다.**

`d=4096,Hq=30`이면 integer head dimension가 없다. config constructor 또는 model validation가 construction 전에 실패해야 한다. Python floor division 136으로 진행하면 Q heads width 4,080이 hidden 4,096과 맞지 않는다.

관측은 raw/effective d/Hq와 validation function다. user override가 Hq만 바꿨는지, remote config class가 다른 head_dim field를 explicit 제공하는지 본다. explicit head_dim가 있어도 projection widths와 RoPE/cache contract를 맞춰야 한다.

**사건 3 — GQA와 TP가 맞지 않는다.**

`Hq=32,Hkv=7,TP=4`는 local KV heads를 균등 분할하기 어렵다. implementation가 KV heads를 replicate하는 지원 path가 있는지, divisibility를 요구하는지 source를 본다. unsupported면 선명한 validation error가 나야 한다.

load를 통과했지만 wrong local mapping이면 K/V projection shard widths와 KV cache local heads가 처음 달라질 수 있다. rank별 expected/actual shapes와 head ownership를 기록한다. aggregate global shape만 보면 문제를 숨긴다.

**사건 4 — vocab rows가 다르다.**

config V=32,000인데 embedding tensor rows가 32,064라고 하자. 일부 checkpoints는 hardware/partition alignment를 위해 padded vocab를 가질 수 있다. loader가 extra rows를 허용/trim하고 effective vocab/logits를 32,000으로 유지하는지 확인한다.

tokenizer maximum ID가 32,010이면 config 32,000은 안전하지 않다. tokenizer files, config vocab, embedding/LM-head rows와 output slicing policy를 함께 본다. tie true면 embedding mismatch가 LM-head에도 동시에 영향을 준다.

**사건 5 — tie state와 LM-head tensor가 모순된다.**

tie true인데 separate `lm_head.weight` shape가 다르면 loader가 skip하는지 conflict로 실패하는지 정책을 본다. tie false인데 lm_head가 없으면 missing required tensor다. GGUF loader mutation가 true를 만들었다면 tensor absence predicate와 effective log를 확인한다.

weight values가 embedding와 동일하다는 hash만으로 tied ownership를 증명하지 않는다. constructed model가 같은 parameter/storage를 참조하는지, pipeline/shard ranks에서 어떻게 소유하는지 본다.

**사건 6 — RoPE fields가 충돌한다.**

legacy scaling field와 new rope parameters가 동시에 있고 type/base/factor가 다르면 normalization precedence가 필요하다. effective rope state와 selected kernel/position code를 기록한다. load는 성공해도 context position가 커질 때 output가 달라질 수 있다.

first divergence는 position 0이 아니라 scaling가 작동하는 boundary 뒤일 수 있다. same weights/config except rope effective state로 deterministic hidden checkpoint를 비교한다. 단순 long-context 품질 저하를 cache bug로 분류하지 않는다.

**사건 7 — dense와 MoE inventory가 섞였다.**

architectures는 dense Llama인데 config expert count가 positive이거나 artifact에 expert tensors가 있다면 selected class와 config schema가 모순된다. dense class는 expert fields를 무시해 unexpected tensors를 만들 수 있고 MoE class는 router/experts를 요구한다.

expert count, used/top-k와 group counts를 validation하고 expert tensor rank/dimensions를 본다. experts 일부만 누락된 shard 사건과 wrong architecture를 구분한다. 52장의 family 비교로 넘길지 incident fix로 metadata를 복구할지 결정한다.

**사건 8 — `trust_remote_code` on/off가 다른 model을 만든다.**

trust off에서는 native mapping로 Llama class가 선택되고 trust on에서는 repository custom class가 선택될 수 있다. 두 runs의 qualified config/model class, code revision, parameter inventory와 cache interface를 비교한다.

trust on으로 load가 성공했다고 native metadata mismatch가 고쳐진 것은 아니다. custom class가 intended implementation인지 검토한다. 반대로 trust off failure를 unsupported라고 단정하기 전에 current native registry support와 explicit local code priority를 본다.

**사건 9 — GGUF metadata와 tensor dimensions가 다르다.**

GGUF architecture가 llama이고 heads/embedding metadata를 성공적으로 읽었어도 tensor directory dimensions가 다른 checkpoint에서 왔을 수 있다. hparams-derived expected shape와 actual tensor info를 name별로 비교한다.

첫 mismatch가 Q/K/V면 head counts/head dimensions, FFN이면 intermediate/expert state, embedding/output이면 vocab/tie를 역추적한다. 모든 mismatch를 converter bug라고 하지 않고 raw GGUF keys와 tensor inventory가 어느 source에서 왔는지 확인한다.

## 48.9 한 Llama 체크포인트를 네 구현에서 수직으로 추적하는 실습

지금까지의 설명을 실제 조사 절차로 바꾸어 보자. 대상은 앞에서 정의한 교육용 Llama fixture다. `vocab_size=32000`, `hidden_size=4096`, `num_hidden_layers=32`, `num_attention_heads=32`, `num_key_value_heads=8`, `intermediate_size=11008`, `tie_word_embeddings=false`라고 하자. dtype은 BF16이고 tensor parallel size는 4다. 이 숫자들은 단순한 예제가 아니다. 각 숫자가 어느 객체의 어느 필드로 들어가고, 어떤 tensor shape를 만들며, checkpoint의 어느 이름과 대조되는지를 끝까지 기록하기 위한 기준선이다.

### 48.9.1 먼저 숫자가 아니라 불변식을 적는다

조사를 시작할 때 바로 소스 검색창에 `4096`을 넣으면 안 된다. 같은 숫자가 hidden width, projection input, output width, reshape의 곱 등 여러 의미로 나타나기 때문이다. 먼저 의미를 가진 기호를 만든다. 이 fixture에서 vocabulary row 수를 `V`, hidden width를 `H`, query head 수를 `Nq`, KV head 수를 `Nkv`, head dimension을 `D`, MLP intermediate width를 `I`, layer 수를 `L`이라고 하자. 그러면 `V=32000`, `H=4096`, `Nq=32`, `Nkv=8`, `D=H/Nq=128`, `I=11008`, `L=32`다.

첫 번째 불변식은 `H % Nq == 0`이다. 이 식이 깨지면 query를 `[tokens, Nq, D]`로 재해석할 수 없다. 두 번째는 `Nq % Nkv == 0`이다. 이 fixture의 각 KV head는 네 query head에 공유된다. 세 번째는 TP rank마다 query head를 균등하게 나눌 수 있어야 한다는 것이다. TP=4이므로 rank당 query head는 8개, KV head는 2개다. 네 번째는 attention output의 마지막 폭이 다시 `H`여야 residual branch와 더할 수 있다는 것이다. 다섯 번째는 MLP의 gate와 up projection이 같은 `I`를 만들고, down projection이 `I`를 `H`로 돌려놓아야 한다는 것이다.

이 불변식 목록은 config parser의 산출물이 아니다. 독자가 loader와 forward를 함께 심사하기 위한 별도 원장이다. 구현이 default를 채우거나 compatibility mutation을 수행해도 원장의 식은 변하지 않는다. 달라지는 것은 식에 대입되는 effective value다. 따라서 장애 보고서에는 “32 heads”라고만 쓰지 말고 “raw `Nq=32`, effective `Nq=32`, TP-local `Nq=8`”처럼 generation과 scope를 함께 적어야 한다.

### 48.9.2 parameter inventory를 shape와 소유권으로 펼친다

Embedding table은 논리적으로 `[V, H]`, 즉 `[32000, 4096]`이다. row 하나는 token id 하나의 hidden vector다. BF16이라면 원소 하나가 2 byte이므로 padding과 allocator overhead를 제외한 논리 payload는 `32000×4096×2=262,144,000 byte`다. 그러나 이 계산만으로 실제 GPU 점유량을 단정하면 안 된다. vocabulary parallelism이 row를 분할할 수 있고, loader가 padded vocabulary를 만들 수 있으며, tied head가 같은 storage를 공유할 수 있기 때문이다. shape 원장에는 logical shape, local shape, storage owner를 서로 다른 열로 둔다.

Attention의 query weight를 PyTorch식 linear weight 표기 `[out_features, in_features]`로 읽으면 `[Nq×D, H]=[4096,4096]`이다. key와 value는 각각 `[Nkv×D,H]=[1024,4096]`이다. output projection은 `[H,Nq×D]=[4096,4096]`이다. 이때 checkpoint가 `q_proj`, `k_proj`, `v_proj`를 별도 tensor로 저장하더라도 serving engine은 세 tensor를 하나의 packed QKV parameter로 보유할 수 있다. packed physical tensor를 보고 query, key, value가 원래부터 같은 폭이었다고 역추론하면 GQA를 놓친다. packed 축의 segment boundary는 `[4096 | 1024 | 1024]`이고 단순한 3등분이 아니다.

TP=4에서는 query projection의 output 축이 rank당 1024, key와 value는 각각 256이 된다. rank-local packed 폭은 1536이다. 반면 input hidden 폭 4096은 각 rank가 모두 받거나 preceding collective 결과로 준비된다. output projection은 input 쪽 head shard를 rank별로 받고 결과를 합치는 row-parallel 형태가 된다. 여기서 global checkpoint shape와 local parameter shape가 다르다는 사실이 중요하다. loader validation이 global tensor `[4096,4096]`을 local parameter `[1024,4096]`과 그대로 비교한다면 올바른 모델도 실패한다. 정상 loader는 shard id와 partition dimension을 알고 필요한 slice를 선택한다.

MLP에는 gate, up, down 세 weight가 있다. gate와 up은 각각 `[I,H]=[11008,4096]`, down은 `[H,I]=[4096,11008]`이다. 많은 serving implementation은 gate와 up을 `[2I,H]` 형태로 pack한다. 이때도 checkpoint 이름 두 개가 runtime parameter 하나로 수렴한다. shard mapping에는 “어느 checkpoint tensor가 packed parameter의 어느 segment에 들어가는가”가 들어 있어야 한다. `gate_proj`와 `up_proj`의 순서를 뒤집어도 shape validation은 통과할 수 있지만 activation 의미는 달라진다. 이것이 loader 성공이 semantic correctness를 보장하지 못하는 대표 사례다.

각 layer의 input norm과 post-attention norm은 각각 `[H]`다. bias가 없는 Llama 변형이라면 projection bias 이름이 없어야 한다. 마지막 norm도 `[H]`다. untied LM head는 `[V,H]=[32000,4096]`이며 embedding과 shape는 같지만 별도 소유권을 가진다. tied model에서는 head parameter가 없거나 embedding storage를 alias할 수 있다. 따라서 “같은 shape”와 “같은 parameter”와 “같은 storage”는 서로 다른 판정이다. checkpoint에 두 tensor가 모두 있더라도 runtime이 tie할 수 있고, checkpoint에 head가 없더라도 loader가 embedding을 head로 연결할 수 있다.

### 48.9.3 KV cache는 weight 목록이 아니라 실행 shape의 결과다

한 layer에서 token 하나가 만드는 K 원소 수는 `Nkv×D=1024`, V도 1024다. BF16이라면 K와 V를 합쳐 token·layer당 `1024×2×2=4096 byte`의 논리 payload가 필요하다. 32 layer 전체에서는 token당 131,072 byte, 즉 128 KiB다. sequence 8,192 token 하나가 모두 resident라면 단순 논리 계산으로 1 GiB다. 이것은 page table, alignment, allocator metadata, fragmentation, graph capture reserve를 제외한 수치다. cache dtype을 FP8로 바꾸면 payload 원소 크기는 줄지만 scale metadata와 kernel 지원 조건이 생긴다.

여기서 `num_attention_heads=32`를 잘못 사용하면 token당 cache를 네 배로 추정한다. 반대로 checkpoint가 실제 MHA인데 config만 `Nkv=8`로 바꾸면 loader에서 K/V projection row mismatch가 먼저 나타나거나, permissive한 변환을 거쳤다면 cache write에서 head axis가 맞지 않는다. 즉 KV cache OOM은 allocator만의 문제가 아닐 수 있다. config의 GQA cardinality가 잘못되어 cache capacity planner가 틀린 경우도 있다.

Paged cache의 physical layout은 구현마다 다를 수 있다. 예를 들어 block size가 16 token이면 하나의 logical sequence가 여러 physical block에 매핑된다. 그러나 layout이 `[blocks, block_tokens, kv_heads, head_dim]`인지, vectorization 축을 추가한 형태인지와 무관하게 원소 수의 의미는 `tokens×layers×2×Nkv×D`에서 출발한다. shape 조사는 semantic axes와 physical axes를 분리한다. semantic axis가 맞고 physical stride만 다르면 kernel adapter의 문제다. semantic axis 자체가 다르면 config 또는 class 선택 단계로 더 올라가야 한다.

### 48.9.4 Transformers 경로에서는 두 번의 선택을 표시한다

Transformers를 읽을 때 첫 화살표는 config factory다. `AutoConfig`는 저장소의 config data를 읽고 `model_type` 또는 auto map을 근거로 concrete config class를 고른다. 이 시점의 결과는 아직 causal LM class가 아니다. 두 번째 화살표는 model factory다. `AutoModelForCausalLM` 계열은 config class에 대응하는 mapping 또는 remote auto map을 사용해 concrete model class를 고른다. `architectures`는 저장 당시 class 이름을 기록하고 여러 fallback과 tooling에 힌트를 주지만, 그것 하나가 모든 auto resolution을 독점하지 않는다.

따라서 로그에는 `config.__class__`, `config.model_type`, `config.architectures`, 최종 `model.__class__`를 별도로 남긴다. 네 값 가운데 앞의 셋만 보고 “Llama가 선택됐다”고 결론내리지 않는다. custom repository가 `auto_map`을 제공하고 `trust_remote_code=true`라면 repository code가 새로운 config/model class를 제공할 수 있다. 같은 JSON이라도 trust flag, revision, cached module 상태가 다르면 최종 Python class identity가 달라진다.

Fixture를 Transformers native Llama로 읽었다면 model constructor는 effective config에서 embedding, layer 수, attention head 수, KV head 수, intermediate width를 읽어 module tree를 만든다. 그 뒤 checkpoint loader가 key와 parameter를 대조한다. 이 순서 때문에 config의 작은 오류가 “weight 파일이 깨졌다”는 모습으로 나타난다. 예를 들어 `intermediate_size=14336`으로 잘못 적으면 모든 layer의 gate/up/down shape가 체계적으로 어긋난다. 32개 layer에서 같은 축이 반복해 다르면 개별 tensor corruption보다 config architecture mismatch를 먼저 의심해야 한다.

`ignore_mismatched_sizes`류의 우회는 조사 도구이지 정상화 수단이 아니다. vocabulary resize처럼 의도된 변경에는 유용할 수 있지만, attention 또는 MLP 핵심 weight를 random initialization으로 남길 수 있다. 경고를 숨긴 채 서비스하면 loader는 성공하고 응답 품질만 붕괴한다. 우회를 사용했다면 어느 key가 재초기화됐는지, 변경된 tensor가 forward의 어느 연산에 쓰이는지, 원래 checkpoint와의 호환성을 어떻게 검증했는지까지 기록한다.

### 48.9.5 vLLM 경로에서는 registry와 packed loader를 분리한다

vLLM의 첫 질문은 “이 Hugging Face architecture label을 어떤 vLLM model implementation이 담당하는가”다. registry가 native class를 찾으면 그 class의 constructor가 parallel layer와 quantization-aware parameter를 만든다. 지원되지 않는 class가 fallback 경로를 택할 수 있으므로 최종 선택 경로를 확인해야 한다. 이름에 Llama가 들어간다는 사실보다 실제 import된 class와 그 class의 `load_weights`가 중요하다.

Fixture의 separate `q_proj`, `k_proj`, `v_proj` key는 vLLM 내부의 packed QKV parameter로 들어갈 수 있다. mapping table은 checkpoint suffix를 runtime parameter name으로 바꾸는 동시에 shard label을 전달한다. parameter-specific weight loader는 label에 따라 query segment, key segment, value segment를 선택하고 TP rank에 해당하는 slice를 복사한다. 따라서 단순 key rename과 tensor partition을 하나의 단계로 뭉뚱그리면 장애 위치를 찾기 어렵다. 조사자는 name mapping, packed segment selection, TP slicing, dtype/quantization conversion을 네 단계로 나누어 본다.

QKV mismatch가 났을 때 runtime packed parameter 전체 shape만 출력해서는 부족하다. checkpoint source shape, logical unsharded destination segment shape, TP-local destination segment shape를 함께 출력해야 한다. 이 fixture라면 K source의 global output rows는 1024, rank-local rows는 256이다. destination packed parameter의 전체 local output rows가 1536이라는 이유로 K tensor가 1536 row여야 하는 것은 아니다. segment loader가 K의 offset과 length를 알고 있어야 한다.

vocabulary parallel embedding과 LM head도 같은 방식으로 본다. runtime이 vocab rows를 hardware-friendly multiple로 pad할 수 있으므로 local storage row 수가 `V/TP`와 정확히 같지 않을 수 있다. token id의 유효 범위는 원래 vocabulary semantics를 따르고 padded rows는 routing과 masking의 대상이다. checkpoint row mismatch를 발견했을 때 먼저 tokenizer vocabulary, config vocabulary, checkpoint rows, padded runtime rows를 네 값으로 분리한다.

Quantized checkpoint에서는 shape와 storage representation이 더 멀어진다. 논리적으로 `[out,in]`인 weight가 packed integer words, group scales, zero points 같은 여러 parameter로 저장될 수 있다. 이때 원본 dense shape는 quantization config와 metadata로 복원된다. “tensor rank가 다르다”는 사실만으로 architecture mismatch라고 판정하지 않는다. 반대로 quantization method가 다르면 이름과 바깥 shape가 비슷해도 unpack 의미가 다르므로 method selection을 class selection과 같은 수준으로 기록한다.

### 48.9.6 SGLang 경로에서는 호환 mutation 전후를 보존한다

SGLang 역시 architecture label에서 serving class를 찾고, 그 class가 기대하는 parameter inventory를 만든다. vLLM과 유사한 parallel primitive를 사용하거나 호환되는 weight-loading 관례를 공유하는 부분이 있더라도 선택 로직과 fallback, 지원 범위가 완전히 같다고 가정하면 안 된다. 같은 repository가 한 engine에서는 native class, 다른 engine에서는 fallback class로 열릴 수 있다.

특히 loader가 GGUF나 특정 checkpoint family를 수용하기 위해 config를 보정한다면 raw JSON만 보관해서는 재현이 되지 않는다. mutation 직전과 직후 snapshot, mutation을 수행한 함수, 조건식, 선택된 architecture, 최종 tensor inventory를 함께 남긴다. 예를 들어 vocabulary size를 metadata row 수에 맞추거나 tie state를 보정했다면 downstream embedding과 head shape가 함께 달라진다. 보정은 단순 경고가 아니라 model graph를 바꾸는 사건이다.

SGLang에서 `tie_word_embeddings`가 loader mapping에 반영되는지 확인할 때는 세 질문을 던진다. constructor가 별도 head parameter를 만들었는가, loader가 missing head를 embedding key로 대체하는가, forward에서 logits 계산이 어느 storage를 참조하는가. 세 답이 일치해야 한다. constructor는 untied인데 loader만 embedding을 복사해 넣었다면 초기 값은 같아도 storage identity는 다르다. serving만 할 때 결과가 같아 보일 수 있지만 memory accounting과 adapter 적용에는 차이가 생긴다.

Adapter가 projection weight 위에 적용되는 경우 packed mapping은 더 중요해진다. LoRA target이 `q_proj`라고 적혀 있어도 runtime에는 packed QKV parameter 하나만 있을 수 있다. adapter loader는 logical module name을 packed segment로 다시 매핑해야 한다. base checkpoint가 정상인데 adapter 적용 후에만 품질이 무너지면 base shape뿐 아니라 adapter rank, target mapping, segment offset, TP shard를 추적한다. adapter 설명 자체는 별도 장의 대상이지만, 여기서는 architecture-to-shape 계약이 adapter에도 그대로 전파된다는 점을 놓치지 않는다.

### 48.9.7 llama.cpp 경로에서는 JSON 대신 GGUF key와 tensor directory를 대조한다

llama.cpp에서 동일한 질문은 다른 표면을 가진다. GGUF의 general architecture key가 어느 model architecture enum으로 해석되는지 확인하고, 그 architecture에 맞는 key template로 block count, embedding length, feed-forward length, attention head count, KV head count, rope fields를 읽는다. 그 결과가 model hparams가 되고, hparams가 expected tensor names와 dimensions를 만든다.

여기서 metadata dump는 config dump와 같은 역할을 하지만 완전히 같은 schema는 아니다. `hidden_size`라는 JSON 이름을 그대로 찾기보다 GGUF vocabulary에 정의된 key를 사용해야 한다. 변환기가 Transformers config를 GGUF metadata로 옮길 때 default와 derived field를 materialize할 수도 있다. 따라서 원본 HF config와 변환 후 GGUF metadata를 비교할 때 field name이 아니라 의미를 정규화해야 한다.

Fixture의 attention K/V shape는 GGUF tensor directory에서도 `Nkv×D`의 폭을 반영해야 한다. quantization type에 따라 byte layout과 block packing은 달라지지만 logical dimensions는 hparams와 맞아야 한다. loader가 metadata로 기대한 dimension과 directory entry의 dimension을 비교하는 이유는 kernel에 잘못된 stride를 넘기기 전에 실패시키기 위해서다. 이 validation을 제거하고 억지로 load하면 오류가 첫 token의 matrix multiplication이나 reshape까지 지연될 뿐이다.

GGUF 변환 전후를 검증할 때 최소한 embedding rows, block count, Q/K/V logical dimensions, MLP dimensions, norm vectors, output head 존재 여부를 비교한다. tensor 이름이 달라졌다는 사실보다 inventory cardinality와 logical axes가 보존되었는지가 핵심이다. quantized representation에서는 원소별 값 비교가 어렵더라도 dequantized sample 또는 conversion tool의 reported error를 이용할 수 있다. metadata만 맞고 tensor payload가 잘못된 경우를 배제하려면 이 단계가 필요하다.

### 48.9.8 한 사건을 네 구현에서 같은 언어로 기록한다

가령 모델이 Transformers에서는 열리지만 vLLM에서 K projection shape mismatch로 실패했다고 하자. 나쁜 보고서는 “vLLM이 이 모델을 지원하지 않는다”로 끝난다. 좋은 보고서는 다음 순서를 가진다. raw config의 `H,Nq,Nkv,D`, Transformers가 선택한 config/model class, Transformers module의 K weight shape, checkpoint K tensor shape, vLLM registry가 선택한 class, vLLM이 만든 global logical K shape, TP-local K segment shape, 실제 loader가 선택한 source slice를 한 표에 놓는다.

Transformers K weight와 checkpoint K tensor가 `[1024,4096]`로 일치하는데 vLLM expected global K가 `[4096,4096]`라면 vLLM 쪽 effective `Nkv`가 32가 된 경로를 찾는다. raw config에 `num_key_value_heads=8`이 있는데도 그랬다면 config conversion 또는 fallback class가 field를 잃었을 가능성이 있다. 반대로 Transformers도 `[4096,4096]`을 만들었다면 raw field가 읽히지 않았거나 custom class semantics가 다를 수 있다. 같은 증상이라도 최초 divergence 위치가 다르다.

네 구현을 비교할 때 공통 vocabulary는 “파일 이름”이 아니라 “의미 축”이다. `V,H,L,Nq,Nkv,D,I`, tie state, dense/MoE state, RoPE regime를 공통 열로 둔다. 각 구현의 field와 class와 tensor name을 그 열에 매핑한다. 이렇게 해야 한 구현의 내부 명칭이 바뀌어도 비교 표가 무너지지 않는다. 또한 upstream metadata 오류와 downstream implementation bug를 분리할 수 있다.

### 48.9.9 로더를 읽을 때 사용하는 다섯 개의 중단점

첫 번째 중단점은 metadata parse 직후다. raw source와 parsed value가 같은지 확인한다. 문자열 enum, null, 누락 field, alias가 이 단계에서 드러난다. 두 번째는 config normalization 직후다. default, compatibility rule, quantization rule, CLI override가 반영된 effective 값을 기록한다. 세 번째는 class resolution 직후다. registry key와 최종 class identity, fallback 이유를 남긴다.

네 번째는 module construction 직후이자 weight copy 전이다. 이때 expected parameter inventory와 global/local shape를 덤프하면 checkpoint 영향 없이 config와 class가 만든 graph를 볼 수 있다. 다섯 번째는 weight mapping과 copy 직전이다. source key, destination parameter, packed segment, shard axis, source slice, destination slice, conversion dtype를 기록한다. copy 이후 checksum이나 sample statistic도 유용하지만, 잘못된 segment 순서처럼 shape가 같은 오류는 별도 semantic test가 필요하다.

이 다섯 중단점 사이에서 최초로 값이 달라진 구간이 조사 범위다. parse 전부터 잘못됐다면 모델 제작 또는 배포 artifact 문제다. normalization에서 바뀌었다면 engine policy 문제다. class resolution에서 갈라졌다면 registry/support 문제다. construction shape가 틀리면 class implementation 문제다. construction은 맞고 copy만 틀리면 mapping/sharding/quantization loader 문제다. 이 분류는 “일단 trust flag를 켜 보라”거나 “config 숫자를 맞춰 보라” 같은 무차별 처방을 줄인다.

### 48.9.10 forward shape probe로 의미를 닫는다

Weight가 모두 들어갔다면 짧은 token batch로 forward shape를 검사한다. batch와 sequence를 합친 token-major 표현을 쓰든 `[batch,sequence,hidden]`을 유지하든 embedding 출력의 마지막 semantic width는 H다. Q는 `Nq×D`, K/V는 `Nkv×D`로 갈라진다. RoPE는 query와 key의 rotary subspace에 같은 position convention을 적용한다. attention 결과는 head들을 합쳐 H로 돌아오고 output projection과 residual addition을 통과한다.

MLP에서는 norm 출력 H가 gate/up 두 projection을 지나 I 두 개가 된다. activation과 elementwise product 뒤에도 I이며 down projection 뒤 H가 된다. 마지막 norm 뒤 LM head가 vocabulary logits를 만든다. untied fixture의 logits 마지막 축은 V다. runtime padded vocabulary가 있다면 physical logits buffer가 더 넓을 수 있지만 sampling 전에 invalid padded rows가 선택되지 않도록 처리해야 한다.

Probe는 shape만 assert하고 끝내지 않는다. Q/K/V의 segment별 norm, finite 여부, layer별 activation scale, tied storage identity, logits의 valid vocabulary 범위를 확인한다. gate/up이 뒤바뀐 경우처럼 shape는 완벽하지만 값의 의미가 틀린 장애가 있기 때문이다. 두 engine을 비교한다면 같은 tokenizer output, 같은 prompt, 같은 dtype 허용 오차, 같은 position과 cache 상태에서 intermediate sample을 비교한다. 첫 layer부터 어긋나는지, attention 이후인지, MLP 이후인지가 loader mapping의 어느 부분을 볼지 알려 준다.

## 48.10 shape mismatch 운영 워크북

이 절은 장애가 발생했을 때 그대로 복사해 사용할 수 있는 조사 기록이다. 핵심은 오류 메시지를 고치는 것이 아니라, 어느 세대의 사실이 처음 달라졌는지를 밝히는 데 있다. 아래 질문에 답할 수 없다면 아직 원인을 찾은 것이 아니다. 임시로 모델이 열렸더라도 다음 engine upgrade, quantization 변환, adapter 적용, TP 변경에서 같은 문제가 다른 형태로 돌아온다.

**사건 기록의 첫 장은 artifact identity다.**

먼저 repository 이름만 적지 않는다. revision commit, config file hash, tokenizer 관련 파일 hash, weight index hash, shard 목록, GGUF file hash를 기록한다. remote code를 쓴다면 code revision과 cache에서 실제 import한 module path도 포함한다. `main` branch 이름은 시간이 지나면 같은 artifact를 가리키지 않는다. “어제는 됐는데 오늘 안 된다”는 보고의 상당수는 실행 코드만 바뀐 것이 아니라 model artifact generation도 함께 바뀐 경우다.

그 다음 engine identity를 적는다. Transformers, vLLM, SGLang, llama.cpp의 version 문자열만으로 부족하면 commit을 기록한다. CUDA와 kernel 문제를 조사하는 장은 따로 있지만, 이 장의 loader 판정에서도 dtype과 quantization backend는 필요하다. 같은 logical architecture라도 quantization loader가 다른 parameter inventory를 요구하기 때문이다. 다만 GPU 이름이나 scheduler option을 무작정 붙여 원인을 흐리지 않는다. architecture-to-shape 경로에 영향을 주는 값부터 남긴다.

입력 artifact가 Hugging Face repository라면 `config.json` 원문을 보존하고, parser 이후 config serialization도 따로 보존한다. 두 파일의 diff가 normalization 기록이다. GGUF라면 metadata key/value dump와 tensor directory의 name, logical dimensions, quantization type을 보존한다. binary payload 전체를 보고서에 넣을 필요는 없지만, 원본을 다시 식별할 hash와 source URI는 있어야 한다.

모델 이름에는 architecture가 암시돼 보이지만 이름을 증거로 쓰지 않는다. `Llama`, `Qwen`, `Gemma` 같은 문자열이 repository 이름에 있어도 config와 code가 fork일 수 있다. 반대로 custom 이름이어도 native Llama-compatible graph일 수 있다. artifact identity는 마케팅 이름, schema identity, code identity를 구분한다.

**config 표에는 raw·default·override·effective 네 열을 둔다.**

각 field마다 raw value를 적는다. field가 없으면 빈칸이 아니라 `absent`라고 쓴다. parser default가 적용됐다면 default source를 적는다. CLI나 serving option이 덮었다면 override source를 적는다. compatibility code가 mutation했다면 함수와 조건을 적는다. 마지막 열이 module constructor가 실제로 읽은 effective value다.

예를 들어 `num_key_value_heads`가 raw에서 absent이고 config class default가 `num_attention_heads`를 사용했다면 effective GQA가 MHA가 된다. 단지 effective 값 32만 남기면 원 제작자가 32를 명시했다고 오해한다. 반대로 raw에 8이 있는데 engine conversion 결과 32가 됐다면 field 전달 누락이나 alias mismatch다. 두 경우는 결과 숫자가 같아도 수정 주체가 다르다.

`rope_scaling`처럼 구조체인 field는 문자열 한 줄로 축약하지 않는다. type, factor, original maximum position, low/high frequency factor 등 구현이 읽는 하위 key를 펼친다. unknown key가 버려졌는지, validation이 reject했는지, custom config가 보존했는지도 적는다. RoPE는 weight shape를 거의 바꾸지 않으면서 position semantics를 바꿀 수 있으므로 load 성공만으로 검증할 수 없다.

`torch_dtype` 또는 dtype 관련 metadata도 raw recommendation과 actual parameter dtype를 분리한다. config의 dtype은 loader 정책에 의해 override될 수 있다. quantization config가 있으면 weight storage dtype, compute dtype, scale dtype을 한 칸에 섞지 않는다. shape 문제와 dtype 문제는 오류 메시지가 비슷해질 수 있지만 byte stride와 kernel eligibility가 다르다.

`tie_word_embeddings`는 boolean 하나로 끝내지 않는다. raw flag, constructor의 head 생성 여부, checkpoint head key 존재 여부, loader alias/copy 동작, 최종 storage identity를 기록한다. “tied=true”인데 head key가 있는 것은 반드시 오류가 아니다. exporter가 중복 저장했을 수 있다. 중요한 것은 선택한 implementation이 중복 key를 어떻게 처리하고 최종 forward가 어느 storage를 사용하는가다.

**class 선택 표에는 후보가 탈락한 이유도 남긴다.**

Transformers에서는 config class 후보와 model class 후보를 구분한다. `model_type`, config `auto_map`, model `auto_map`, `architectures`, task-specific auto mapping이 각각 어떤 후보를 제공했는지 적는다. 최종 class만 적으면 trust flag를 바꿨을 때 왜 다른 결과가 나오는지 설명할 수 없다. remote class가 선택됐다면 import source revision을 class 이름 옆에 쓴다.

vLLM과 SGLang에서는 architecture string 목록을 어느 순서로 검사했는지, native registry hit인지, fallback인지, unsupported 판정인지 적는다. 여러 architecture 이름이 들어 있는 config에서 첫 번째 이름이 지원되지 않고 두 번째가 선택될 수도 있다. 이름 하나만 복사한 로그는 이 결정을 숨긴다. registry alias가 있다면 alias에서 concrete class로 가는 mapping도 기록한다.

llama.cpp에서는 GGUF architecture value에서 내부 model architecture로 가는 해석을 기록한다. 동일 family 안에서도 tensor naming template와 optional tensor inventory가 variant에 따라 달라질 수 있다. metadata key가 알려져 있다는 사실과 model graph가 지원된다는 사실은 구분한다. parser가 key를 읽을 수 있어도 expected tensor builder가 해당 variant를 모를 수 있다.

후보 탈락 사유는 중요하다. dependency가 없어서 remote class import가 실패했는지, engine registry가 architecture를 모르기 때문인지, quantization method가 class와 호환되지 않기 때문인지, multimodal processor 조건이 부족한지에 따라 복구가 달라진다. class를 강제로 다른 이름으로 바꾸는 처방은 후보 탈락 이유를 지우므로 최후의 실험으로만 사용한다.

**inventory diff는 missing과 unexpected만 세지 않는다.**

parameter inventory를 비교할 때 각 row에 semantic role, checkpoint key, runtime parameter, global logical shape, local physical shape, required/optional, packed segment, tie owner를 둔다. missing key 수와 unexpected key 수만 세면 중요한 패턴이 사라진다. 32개 layer에서 K/V만 동일하게 mismatch라면 GQA field 문제일 가능성이 높다. 모든 MLP projection이 같은 intermediate 축에서 mismatch라면 `intermediate_size` 문제다. layer 수 이후의 모든 key가 unexpected라면 `num_hidden_layers`가 다르다.

Embedding과 LM head에서만 row 수가 다르면 vocabulary generation을 본다. tokenizer vocabulary length, added tokens, config `vocab_size`, checkpoint embedding rows, checkpoint head rows, runtime padded rows를 나열한다. tokenizer length가 더 크면 새 token id가 embedding 범위를 벗어날 수 있다. checkpoint rows가 더 크면 reserved 또는 padding rows일 수 있다. 무조건 작은 쪽으로 자르면 special token이 사라질 수 있다.

Norm tensor가 `[H]`가 아니라 다른 폭이라면 hidden width mismatch를 의심한다. 하지만 quantized format의 auxiliary scale tensor는 원래 weight와 다른 rank를 가질 수 있다. semantic role이 norm인지 quantization metadata인지 먼저 구분한다. 이름 substring만으로 role을 분류하면 `weight_scale`을 실제 projection weight로 오해할 수 있다.

Packed parameter는 반드시 segment 단위로 펼쳐 비교한다. QKV destination 하나를 Q, K, V 세 logical row로 펼치고, gate-up destination 하나를 gate와 up 두 row로 펼친다. source와 destination의 합계 shape만 일치하면 segment 순서 오류를 놓친다. 각 segment에 source key, offset, length, TP slice를 기록하면 동일 shape 오배치를 찾을 수 있다.

MoE inventory에서는 expert axis를 별도 열로 둔다. dense MLP의 gate/up/down 세 tensor와 MoE의 expert별 tensor는 이름이 유사할 수 있지만 rank와 cardinality가 다르다. expert count, top-k, shared expert 존재 여부, router weight shape를 함께 본다. config에 expert field가 우연히 남아 있다고 MoE로 판정하지 말고 선택 class가 실제 expert modules를 만들었는지 확인한다.

**TP를 바꾸면 같은 checkpoint의 local truth가 달라진다.**

Fixture를 TP=1로 읽으면 Q/K/V output rows는 4096/1024/1024다. TP=2에서는 2048/512/512, TP=4에서는 1024/256/256이다. TP=8에서는 512/128/128이다. 이 계산은 `Nkv=8`이 TP에 균등 분할된다는 가정에서 성립한다. TP가 KV head 수보다 커지면 implementation은 KV head replication을 사용하거나 조합을 거부할 수 있다. 따라서 `Nkv % TP == 0`만 보편적 진리로 적으면 안 된다. concrete parallel layer의 replication rule을 읽어야 한다.

Query head와 KV head의 local cardinality가 다르므로 attention kernel에 넘기는 metadata도 달라진다. loader가 weight를 올바르게 나눴더라도 forward가 global head count를 local buffer에 적용하면 reshape가 깨진다. 반대로 kernel이 replication을 지원하지만 cache allocator가 replicated KV head 수를 반영하지 않으면 cache size 또는 indexing이 틀어진다. architecture field는 loader에서 끝나지 않고 runtime metadata까지 전파된다.

Vocabulary parallelism도 TP에 따라 local rows를 바꾼다. V=32000은 4로 나누어지지만 implementation이 padding multiple을 요구하면 각 rank의 storage가 8000보다 클 수 있다. LM head와 embedding이 같은 partition convention을 쓰는지 확인한다. tied model에서 두 모듈이 다른 padding convention을 쓰면 storage alias가 불가능하거나 logits gather가 잘못될 수 있다.

Row-parallel projection은 checkpoint slicing 축과 runtime collective를 함께 읽는다. down projection `[H,I]`에서 어떤 축을 TP로 나누는지는 linear primitive의 convention에 달려 있다. tensor shape만 보고 column-parallel인지 row-parallel인지 이름을 붙이지 않는다. constructor argument와 weight loader, forward collective가 같은 partition contract를 사용하는지 세 군데를 대조한다.

**`trust_remote_code` 사건은 보안 플래그 이상의 shape 사건이다.**

`trust_remote_code=false`에서 native fallback이 선택되고 true에서 repository class가 선택된다면 두 실행은 같은 모델이 아니다. custom config가 새로운 field를 해석할 수 있고, custom model이 projection packing이나 activation, norm 순서를 바꿀 수 있다. 따라서 false에서 발생한 missing key를 true로 해결했다고 해서 native implementation bug가 고쳐진 것은 아니다. 단지 code identity가 바뀌었다.

원격 코드를 허용할 때는 revision을 고정하고 실제 다운로드한 source hash를 남긴다. config와 weight revision만 고정하고 code revision을 놓치면 재현성이 없다. import 시 실행되는 code라는 운영 위험도 있지만, 이 장에서는 특히 architecture semantics가 code에 들어 있다는 점이 중요하다. JSON schema만 보관하면 custom forward를 복원할 수 없다.

원격 class를 native serving class로 이식하려면 field 이름을 복사하는 것보다 graph 계약을 비교한다. module inventory, projection shapes, tensor name mapping, RoPE application 위치, cache interface, logits head, tie behavior를 확인한다. native class가 weight를 모두 받아들였다는 사실은 충분하지 않다. custom activation이나 scaling이 weight 없는 연산이면 inventory diff에 나타나지 않기 때문이다.

**tie와 vocabulary 사건은 메모리 최적화로만 보면 실패한다.**

Embedding과 LM head를 tie하면 parameter payload를 줄일 수 있지만, 목적은 단순 절약만이 아니다. 학습 시 두 역할이 같은 parameter를 공유했다는 architecture 계약이다. serving에서 임의로 tie하거나 untie하면 초기 weight 값이 같더라도 adapter 적용, weight patch, serialization semantics가 달라진다. base model이 untied라면 두 matrix가 실제로 다른 값을 가질 수 있다.

Checkpoint에 LM head가 없고 config가 untied라면 세 가능성을 구분한다. exporter가 tied storage를 중복 저장하지 않았는데 flag만 잘못됐을 수 있다. loader가 embedding을 head로 사용하는 family-specific convention을 가질 수 있다. 또는 artifact가 정말 불완전할 수 있다. embedding을 복사해 head를 만들어 load를 통과시키기 전에 원본 architecture와 exporter 규칙을 확인한다.

Vocabulary resize는 row 수뿐 아니라 token-id 의미를 바꾼다. 두 tokenizer가 모두 length 32000이어도 token-to-id mapping이 다르면 같은 embedding row가 다른 token을 뜻한다. shape 원장은 통과하지만 모델 의미는 깨진다. 따라서 vocabulary mismatch 사건에는 tokenizer hash와 special token ids를 포함한다. logits 마지막 축과 sampling mask가 맞는지까지 확인한다.

Padded rows를 가진 runtime에서 top-k 또는 softmax가 전체 physical rows를 그대로 보게 두면 invalid token이 선택될 수 있다. 일반적으로 masking이나 vocab-range 제한이 필요하다. loader가 padding rows를 zero로 채웠다는 사실만 믿으면 안 된다. zero logit도 다른 valid logits가 음수일 때 선택될 가능성이 있다. architecture vocabulary와 sampler vocabulary의 계약을 닫아야 한다.

**dense/MoE 혼동은 한 field 수정으로 끝나지 않는다.**

Dense Llama fixture에 `num_local_experts` 같은 field를 추가했다고 곧바로 MoE model이 되지는 않는다. config class가 field를 보존하더라도 model class가 dense Llama이면 layer inventory는 gate/up/down 한 벌이다. 반대로 MoE class가 선택되면 router와 여러 expert weight를 기대한다. class와 inventory가 동시에 바뀌어야 한다.

MoE checkpoint를 dense class로 억지 mapping하면 expert tensor가 대량 unexpected key로 남는다. 첫 expert weight 하나를 dense weight에 복사해 loader를 통과시키는 것은 원 architecture를 보존하지 않는다. routing과 expert aggregation이라는 weight 없는 연산도 사라진다. 이런 변환은 호환 패치가 아니라 새로운 모델 변환이며 품질 검증이 별도로 필요하다.

Expert parallelism이 들어가면 global expert count와 rank-local expert inventory가 달라진다. TP의 head shard와 expert shard를 같은 축으로 취급하지 않는다. 한 rank에 없는 expert key가 missing처럼 보여도 정상 partition일 수 있다. loader report에는 global required inventory와 rank-local owned inventory를 구분한다.

**GGUF 변환 사건은 변환 전후 두 schema를 함께 심사한다.**

HF에서 GGUF로 변환한 뒤 llama.cpp만 실패한다면 먼저 converter version과 source revision을 고정한다. 원본 config field가 어느 GGUF metadata key로 갔는지 mapping 표를 만든다. derived `head_dim`, KV head count, RoPE scaling, tie/output presence처럼 이름이 일대일이 아닌 항목을 우선 본다.

그 다음 tensor transpose와 naming을 본다. PyTorch linear weight의 logical orientation과 GGML tensor convention이 다를 수 있으므로 dimension 순서가 뒤집혀 보이는 것만으로 오류라고 하지 않는다. converter와 loader가 합의한 convention에서 matmul semantic axes가 맞는지를 확인한다. tensor directory의 dimension과 hparams 기반 expected dimension을 같은 convention으로 정규화한 뒤 비교한다.

Quantization block은 logical element count와 encoded byte count를 분리한다. 파일 byte 길이를 dtype byte 수로 단순 나누어 shape를 추정하면 block quantization에서 틀린다. quantization type별 block size와 type size를 사용해야 한다. 그러나 architecture 검증의 첫 단계는 여전히 logical dimensions다. metadata의 H와 I가 payload tensor logical shape와 맞지 않으면 quantization 오차가 아니라 schema 또는 conversion 오류다.

Output tensor 존재 여부도 tie semantics와 함께 본다. GGUF가 output weight를 생략하고 token embedding을 재사용하는 convention을 쓴다면 hparams와 model builder가 그 생략을 허용해야 한다. 원본이 untied인데 converter가 output을 빠뜨렸다면 파일 크기는 줄지만 logits가 바뀐다. 변환 로그에서 tie 결정을 명시적으로 남긴다.

**수정안은 한 번에 하나의 generation만 바꾼다.**

원인을 찾은 뒤 config JSON, loader option, registry mapping, weight converter를 동시에 바꾸지 않는다. 한 번에 여러 generation을 바꾸면 어떤 수정이 필요했는지 알 수 없다. 먼저 가장 upstream의 잘못된 사실을 고친다. raw artifact가 틀렸다면 artifact를 새 revision으로 만들고 hash를 바꾼다. engine parser가 field를 잃었다면 parser를 고치고 raw artifact는 유지한다.

Compatibility mutation이 필요하다면 조건을 좁힌다. 특정 architecture, exporter version, field absence, tensor evidence가 모두 맞을 때만 적용하도록 한다. mutation 전후 값을 경고 또는 structured log로 남긴다. 모든 Llama 계열에 vocabulary나 KV head를 강제하는 global patch는 미래 variant를 망가뜨린다.

Registry 수정은 class label만 추가하고 끝내지 않는다. 새 label이 가리키는 class가 checkpoint inventory와 forward semantics를 지원하는지 test fixture를 둔다. 최소 fixture에는 config, 작은 synthetic tensor inventory, expected mapping과 shape assertions가 필요하다. 실제 거대 weight를 CI에 넣지 않아도 architecture contract 상당 부분을 검증할 수 있다.

Loader mapping 수정에는 같은 shape의 오배치를 잡는 test를 넣는다. Q, K, V와 gate, up source tensor를 서로 다른 상수 패턴으로 채우고 destination packed segment에 올바른 패턴이 들어갔는지 확인한다. shape assert만으로는 segment permutation을 검출할 수 없다. TP rank별 slice도 서로 다른 패턴을 사용하면 offset 오류를 찾기 쉽다.

최종 승인에는 load 성공, parameter inventory closure, forward finite check, reference implementation 비교, tokenizer/logits vocabulary check를 포함한다. 모든 출력을 완전히 같게 요구할지는 dtype과 kernel에 따라 달라지지만, 허용 오차와 비교 지점을 먼저 정해야 한다. 결과가 다르면 “GPU 수치 오차”로 넘기기 전에 최초 layer와 최초 연산을 찾는다.

## 48.11 한 field가 다섯 상태를 거쳐 tensor가 되는 tutorial

사건은 config diff가 0이라는 보고에서 시작한다. 두 serving pod는 같은 `config.json`, 같은 weight hash와
tokenizer를 사용했다. 둘 다 Q projection `[4096,4096]`, K/V `[1024,4096]`을 성공적으로 load했다. 그런데
pod B의 KV cache byte는 A의 두 배였고 decode output도 첫 layer부터 달랐다. raw config와 parameter shape가
맞으므로 팀은 CUDA kernel 문제라고 판단했다. 실제 차이는 raw에 없던 derived default와 backend mutation이었다.

값을 다섯 칸으로 나눈다. raw는 artifact에 직렬화된 값 또는 absence다. default는 config class가 absence를
채우는 값이다. derived는 다른 field에서 계산한 값이다. mutated는 compatibility, CLI, backend 또는 loader가
바꾼 값이다. effective는 concrete model/loader/backend consumer가 실제 사용한 최종 값이다. “config value”라는
한 칸에 다섯 값을 덮어쓰지 않는다.

첫 field는 `num_key_value_heads`다. raw가 absent이고 config class default가 `num_attention_heads`라면 Nq=32에서
Nkv=32가 된다. 다른 compatibility layer가 model family 관례로 Nkv=8을 derive하거나 override하면 effective가
8이다. JSON diff에는 둘 다 absence지만 K/V projection width와 KV bytes가 4배 달라진다. absence는 값이 같다는
증거가 아니다.

fixture는 `hidden_size H=4096`, `num_attention_heads Nq=32`, `head_dim D=128`, layers L=32, fp16이다.
Nkv=8이면 Q width4096, K/V width1024다. attention weight orientation을 `[out,in]`으로 쓰면 q_proj
`[4096,4096]`, k/v 각각 `[1024,4096]`, o_proj `[4096,4096]`이다. 한 token 한 layer KV는
`2(K,V)×8×128×2 byte=4096 byte`, 32 layers면 131,072 byte다.

Nkv=32라면 k/v `[4096,4096]`, token/layer KV16,384 byte, 32 layers524,288 byte다. context8192, batch16의
logical KV는 각각 약 16GiB와 64GiB다. parameter shapes가 artifact에서 `[1024,4096]`인데 constructor effective
Nkv32라면 loader shape mismatch가 나야 정상이다. 그런데 backend가 parameter constructor에는 Nkv8을 쓰고
cache planner만 raw/default Nkv32를 쓰면 weights는 성공하면서 cache layout만 두꺼워질 수 있다.

이 split-brain이 사건의 핵심이다. “constructor shape가 맞았다”는 모든 downstream consumer가 같은 effective
field를 썼다는 증거가 아니다. model attention module, weight loader, KV cache spec, attention backend metadata와
parallel shard planner가 각각 어느 object/value를 읽는지 추적한다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- Transformers pinned `LlamaConfig`에서 constructor signature와 assignments를 읽어 raw/default boundary를 찾는다.
- [Transformers LlamaConfig](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/configuration_llama.py#L31-L102) `AutoConfig.from_pretrained`가 model_type와 remote config class를 고르는 경계도 별도다.
- [Transformers AutoConfig](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/auto/configuration_auto.py#L278-L435)

source note에서는 signature를 나열하지만 tutorial 본문에서는 한 field의 consumer를 따라간다. config object가
선택된 뒤 Llama model constructor가 heads와 hidden을 읽어 attention projections를 만든다. parameter inventory가
생기고 loader가 checkpoint names/shapes를 대조한다. forward는 local heads와 repeat/group mapping을 만들고 cache
update에 K/V shape를 넘긴다. serving backend는 이를 block/page layout과 kernel metadata로 변환한다.

두 번째 field는 `head_dim`이다. raw head_dim이 있으면 그것을 쓰는 family도 있고, absent면 `hidden_size /
num_attention_heads`로 derive할 수 있다. H4096/Nq32면 128이다. backend가 explicit head_dim을 무시하고 quotient를
다시 계산하면 H3584/Nq28 같은 divisible case는 우연히 맞지만 nonstandard partial-RoPE/latent attention model은
틀릴 수 있다. constructor와 kernel metadata가 같은 D를 쓰는지 본다.

세 번째는 `intermediate_size I=11008`이다. gated MLP에서 gate/up weights는 `[11008,4096]`, down은
`[4096,11008]`이다. TP2에서 column-parallel gate/up local out5504, row-parallel down local in5504가 될 수
있다. loader가 global tensor를 slice하는 axis와 constructor local parameter axis가 맞아야 한다. raw I가 맞아도
backend padding이 I를 11264로 mutate하면 physical storage와 kernel tile은 달라진다.

padding은 logical/effective를 다시 나눈다. model semantic I는 11008이고 checkpoint rows도 11008이다. backend
physical padded I'=11264이면 loader는 11008 rows를 지정 segment에 넣고 padding256을 정의된 값으로 채운다.
forward/kernels는 logical mask 또는 output slice를 보존해야 한다. mutation을 config.intermediate_size 자체에
덮어쓰면 checkpoint expected shape가 11264로 바뀌어 artifact가 틀린 것처럼 보인다.

네 번째는 `vocab_size V=32000`과 `tie_word_embeddings`다. embedding `[32000,4096]`, untied LM head도
`[32000,4096]`이다. runtime이 vocab을 32128로 padding할 수 있지만 tokenizer/logits valid range는 32000이다.
loader와 sampler가 physical/logical V를 구분한다. tie default가 raw absence를 채우고 backend가 output tensor
presence를 근거로 mutate하면 parameter ownership과 missing-key 판단이 달라진다.

다섯 번째는 RoPE다. `rope_theta`, scaling config와 rotary dimension은 parameter shape를 만들지 않을 수 있지만
forward position transformation을 바꾼다. weight inventory가 완전히 같아도 output이 달라질 수 있다. shape
audit가 통과했다는 사실로 config semantics 전체를 승인하지 않는다. derived inv_freq, maximum positions와
backend-specific scaling consumer를 trace한다.

raw/default/derived/mutated/effective 원장의 한 행은 `field, raw presence/value, config-class default owner,
derive equation/owner, mutation predicate/owner, constructor value, loader expectation, backend/kernel value,
forward observation`이다. 표는 결과 요약이고 본문 source 산책을 대신하지 않는다.

## 48.12 constructor에서 loader와 backend까지 source call graph를 잇는다

class 선택은 첫 갈림길이다. `model_type`은 config class mapping을, `architectures`는 auto model 또는 serving
registry preference를, `trust_remote_code`는 code identity를 바꿀 수 있다. 세 문자열이 모두 Llama처럼 보여도
concrete config/model class와 revision을 기록한다. auto factory의 remote/native 선택을 고정 source에서 본다.
[Transformers auto model factory](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/auto/auto_factory.py#L184-L395)

constructor breakpoint에서는 config object snapshot을 immutable manifest로 복사한다. Python object가 뒤에서
mutate되면 constructor 직전과 cache/backend 초기화 직전 값을 비교할 수 있다. object repr만 저장하지 않고
field presence, source owner와 type을 보존한다. `None`, absence, false와 zero는 다른 상태다.

vLLM registry는 Transformers auto model class와 별도로 native serving architecture를 resolve한다.
[vLLM architecture resolution](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/model_loader/utils.py#L195-L220)
선택된 native Llama class constructor가 MLP와 attention modules를 만든다.
[vLLM Llama construction](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/llama.py#L79-L247)

parameter inventory 다음은 weight mapping이다. separate q/k/v source tensors가 packed destination parameter의
어느 segment로 들어가는지, gate/up도 같은 방식인지 본다. model/loader source는 architecture와 packed mapping을
함께 보여 준다. [vLLM Llama weight loading](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/llama.py#L344-L542)
shape만 맞아도 q와 k segment가 바뀌면 의미가 틀리므로 constant-pattern fixture를 둔다.

SGLang도 자체 registry와 Llama implementation을 가진다. [SGLang architecture resolution](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_loader/utils.py#L188-L234) [SGLang Llama model](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/llama.py#L70-L335) Transformers config class가 같아도 SGLang compatibility mutation, parallel layer와 backend consumer가 다른 effective values를 만들 수 있다.

제품 이름이 아니라 field owner를 비교한다.

loader 이후 backend breakpoint는 Q/K/V runtime shapes, local head counts, cache tensor shape와 attention plan
metadata다. TP2, Nq32,Nkv8에서 rank-local q heads16, KV heads가 partition되면 4일 수 있지만 replication policy면
8일 수 있다. `max(1,Nkv/TP)` 같은 식을 source 없이 일반화하지 않는다. global/local/storage/kernel heads를
각각 기록한다.

parameter count도 검산한다. q weight16,777,216 elements, k/v 각각 4,194,304, o16,777,216으로 attention
projections 합 41,943,040 elements/layer다. MLP gate/up 각각 45,088,768, down45,088,768로 합 135,266,304다.
norm과 bias 여부를 더해 layer inventory를 만든다. count가 맞아도 transpose/segment order는 별 gate다.

loader report는 loaded/missing/unexpected뿐 아니라 expected global/local shape, checkpoint shape, slice axis/range,
destination segment와 bytes를 둔다. TP rank0/1의 slice가 overlap/gap 없이 global tensor를 덮는지 보존식을
쓴다. replicated tensors는 의도된 중복으로 표시한다.

forward probe는 실제 거대 모델 실행을 요구하지 않는다. 작은 synthetic config와 tensor shapes로 constructor와
mapping contract를 검증하거나, 실행 가능한 환경에서는 selected layer checkpoint를 수집한다. 사용자는 실행이
필수라 하지 않았으므로 이 장은 source-derived expected probe를 설계하고 미실행 성능/출력 사실을 주장하지
않는다.

probe checkpoints는 input hidden `[B,T,4096]`, Q `[B,T,32,128]`, K/V `[B,T,8,128]`, cache write local
layout, attention output `[B,T,4096]`, MLP gate/up `[B,T,11008]`, down `[B,T,4096]`, logits
`[B,T,32000]`이다. backend packed/flattened physical shape가 다르면 semantic axes로 normalize한다.

derived field가 weight 없는 연산을 바꿀 때는 tensor inventory로 잡히지 않는다. RoPE scaling, norm epsilon,
activation, attention bias, sliding/window와 logits scaling이 예다. field ledger에 parameter-shape consumer가
없어도 forward/backend consumer를 둔다. “shape에 영향 없음”과 “모델에 영향 없음”은 다르다.

source note는 chapter 끝의 pinned 링크 목록처럼 탐색 좌표만 제공한다. tutorial 본문에서는 한 field가 어느
state를 거쳐 어떤 tensor와 consumer를 만드는지 서사로 걷는다. 함수마다 reference card를 반복해 독서 흐름을
끊지 않는다. 중요한 링크는 producer→consumer 전환점에만 둔다.

## 48.13 config→class→shape→effective graph를 함께 판정한다

복구를 승인하려면 먼저 shape 원장을 닫는다. raw value와 effective mutation, constructed module,
checkpoint inventory와 forward tensor를 같은 행에 놓는다.

원장의 첫 열은 artifact hash/revision와 raw metadata다. 둘째는 mutation/override log와 effective config/hparams다. 셋째는 selected config/model class와 registry path다. 넷째는 global/local expected parameter inventory와 shapes다. 다섯째는 checkpoint/GGUF tensors와 loaded/missing/unexpected 결과다. 여섯째는 forward/cache derived shapes다.

fixture 정상 행은 embedding `[32000,4096]`, Q/K/V widths 4096/1024/1024, packed 6144, gate/up/down shapes, LM head `[32000,4096]`, KV per token/layer 2048 elements를 가진다. TP가 있으면 local row를 별로 적는다.

그다음 loader 성공을 forward correctness와 분리한다. 모든 byte가 복사됐다는 사실은 position,
head partition과 cache shape가 model 의미에 맞다는 충분조건이 아니다.

loader가 모든 names/shapes를 복사해도 segment order, transposition, TP slice와 cache head mapping가 틀릴 수 있다. 작은 deterministic forward에서 embedding, Q/K/V projected shapes, attention/cache write와 logits width를 확인한다.

실행 수치는 이 장에서 만들지 않는다. 독자가 수집할 checkpoints와 expected relations를 정한다. weight count/bytes만으로 correct mapping를 증명하지 않는다.

config mutation은 owner와 이유, downstream shape와 rollback 조건을 기록해야 승인 가능한 변화가 된다.

mutation는 raw→effective diff, owner function, predicate와 downstream effect를 로그/manifest에 남긴다. silent compatibility mutation를 없앨 수 없다면 최소한 reproducible해야 한다. 같은 artifact와 options가 같은 effective config를 만들어야 한다.

SGLang GGUF tie mutation처럼 file tensor presence가 state를 바꾸면 absence가 intentional tied checkpoint인지 corrupt file인지 validation를 붙인다. mutation 뒤 constructed inventory가 actual files와 맞는지 확인한다.

마지막 종합 판정은 네 단계를 다시 나열하지 않고 최초 divergence와 복구 뒤 보존된 불변식을 답한다.

복구 완료는 오류 메시지가 사라지는 것이 아니다. raw/effective metadata가 intended architecture를 표현하고 네 registry가 예상 class를 고르며 parameters와 artifact tensors가 name/shape/dtype/packing contract로 맞아야 한다. forward에서 QKV/MLP/logits와 KV cache shapes도 derived equations와 맞아야 한다.

48장에서 config는 설명용 JSON가 아니라 model memory와 계산을 생성하는 schema임을 확인했다. 다음 49–50장은 이 expected inventory가 safetensors shards와 GGUF bytes에서 어떻게 발견·검증·배치되는지 본다. 52장은 Llama에서 확립한 읽기법으로 Qwen·Gemma·MoE families를 가로 비교한다.

독자가 마지막으로 기억할 것은 특정 숫자가 아니라 질문의 순서다. 이 값은 원본 artifact에 있었는가, 누가 default를 채웠는가, 어떤 option과 compatibility mutation을 거쳤는가, 어느 concrete class가 그 값을 읽었는가, 그 class는 어떤 logical tensor inventory를 만들었는가, parallel layer는 그것을 어떤 local shape와 storage로 바꾸었는가, loader는 checkpoint의 어느 slice를 어느 packed segment에 넣었는가, forward는 그 결과를 어떤 semantic axis로 소비했는가. 이 질문이 한 번도 끊기지 않으면 처음 보는 architecture도 조사할 수 있다.

반대로 어느 한 단계라도 “대략 호환될 것”이라고 건너뛰면 뒤의 관찰은 모호해진다. config 숫자를 고쳐 load가 됐다는 사실은 원 architecture를 복구했다는 증거가 아니다. missing key를 없앴다는 사실도 올바른 source tensor가 올바른 segment에 들어갔다는 증거가 아니다. 첫 token이 생성됐다는 사실도 tokenizer와 vocabulary 의미, position semantics, tie ownership이 보존됐다는 증거가 아니다. 각 단계는 다음 단계의 전제이고, reference forward는 앞선 계약을 마지막으로 심사한다.

이 장의 Llama fixture는 정답을 외우기 위한 예제가 아니라 측정 자다. 다른 모델을 만나면 `V,H,L,Nq,Nkv,D,I`에 그 모델 고유의 축을 더하고 dense/MoE, tie, RoPE, quantization, adapter 계약을 확장한다. 구현 이름과 field spelling이 달라져도 raw metadata에서 effective config, concrete class, expected inventory, physical shard, forward semantic shape로 내려가는 사슬은 남는다. 이 사슬과 최초 divergence를 문서와 로그에 보존하는 팀은 loader 오류를 시행착오가 아니라 재현 가능한 구조 분석으로 바꿀 수 있다.

리뷰에서도 같은 원칙을 쓴다. 수정 diff가 field 하나뿐이라고 작은 변경으로 분류하지 않는다. 그 field가 embedding row, projection width, cache capacity, kernel metadata, logits range 가운데 어디까지 전파되는지 영향 반경을 그린다. 반대로 코드 diff가 여러 loader에 걸쳐 있어도 하나의 semantic axis를 보존하기 위한 반복 변경일 수 있다. 줄 수가 아니라 계약의 변화로 위험을 판단한다. 배포 전에는 raw/effective diff와 inventory diff를 자동 산출하고, 예상하지 않은 변화가 한 줄이라도 있으면 새 artifact generation으로 승인받는다. 그래야 호환 패치가 조용히 모델 정체성을 바꾸는 일을 막을 수 있다.

좋은 조사 기록은 다른 사람이 같은 revision과 option으로 같은 effective graph를 재구성하고, 같은 shape 표에서 같은 최초 불일치를 지목할 수 있게 한다. 그것이 이 장에서 말하는 완결성이다.

## 48.14 소스 노트

이 목록은 config 해석, architecture class 선택, module construction과 weight mapping을 한 단계씩 교차 검증하는 순서로 읽는다. 같은 field 이름이 보여도 각 stack의 resolver가 어떤 class와 tensor shape를 만들었는지는 별도로 확인한다.

### Config가 다른 class를 골랐는가

Model type과 architecture 이름은 맞아 보이는데 예상한 implementation이 만들어지지 않을 때는 여기서 시작한다. Config parsing과 auto/remote-code class resolution을 먼저 잇는다.

- [Transformers v5.15.1 — `AutoConfig.from_pretrained`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/auto/configuration_auto.py#L278-L435)
- [Transformers v5.15.1 — auto model class와 remote-code 선택](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/auto/auto_factory.py#L184-L395)
- [Transformers v5.15.1 — `LlamaConfig`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/configuration_llama.py#L31-L102)

### Module 또는 tensor shape가 다른가

Class 선택은 맞지만 head, MLP, RoPE나 MoE shape가 처음 어긋날 때는 여기서 시작한다. Resolver에서 constructor와 hparam validation까지 같은 field가 어떻게 소비되는지 비교한다.

- [vLLM v0.27.1 — architecture resolution](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/model_loader/utils.py#L195-L220)
- [vLLM v0.27.1 — Llama MLP와 attention construction](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/llama.py#L79-L247)
- [vLLM v0.27.1 — Llama model, causal LM과 weight loading](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/llama.py#L344-L542)
- [SGLang v0.5.18 — architecture resolution](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_loader/utils.py#L188-L234)
- [SGLang v0.5.18 — native Llama attention/MLP/model loader](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/llama.py#L70-L335)

### Weight mapping, tie 또는 format 변환이 다른가

Constructed graph는 맞지만 load 뒤 row identity나 tied representation이 달라질 때는 여기서 시작한다. Causal-LM loader, GGUF mutation과 llama.cpp tensor table을 artifact 이름부터 validation까지 잇는다.

- [SGLang v0.5.18 — Llama causal LM, tie와 weight mapping](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/llama.py#L496-L813)
- [SGLang v0.5.18 — GGUF tie mutation](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_loader/loader.py#L3190-L3260)
- [llama.cpp v0.2.0 — architecture/key/tensor tables](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-arch.cpp#L1-L170)
- [llama.cpp v0.2.0 — GGUF architecture resolution](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model-loader.cpp#L540-L590)
- [llama.cpp v0.2.0 — hparams, head/RoPE/MoE validation](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model.cpp#L1100-L1258)
