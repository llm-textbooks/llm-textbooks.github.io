# 53장. 설정한 backend와 실제 실행된 kernel 사이

서버 설정에는 FlashInfer를 적었다. Python에서 package import도 성공한다. Startup log에도 FlashInfer라는 이름이 찍힌다. 그런데 Nsight Systems에서 decode 구간을 보면 기대한 kernel이 없다. 어떤 요청은 다른 attention kernel을 실행하고, prompt가 길어지면 또 다른 경로로 바뀐다. Quantized linear도 비슷하다. Marlin을 기대했지만 TP를 4로 올린 뒤 generic kernel이 나타나거나, 더 나쁘게는 repack된 weight를 맞지 않는 fallback이 읽어 응답만 조용히 틀린다.

이 사건을 “backend가 fallback했다”는 한 문장으로 끝내면 다음 장애를 막을 수 없다. 요청된 이름, registry에 등록된 후보, capability 검사를 통과한 후보, 최종 선택된 method, 실제 dispatch된 kernel은 서로 다른 사실이다. 그 사이에는 package와 build, CUDA toolkit, GPU SM, dtype과 quant format, local shape, memory layout, prefill/decode phase, CUDA graph와 feature composition이 있다.

이 장은 한 요청을 그 사슬 끝까지 따라간다. Raw config에서 시작해 normalization과 override를 기록하고 후보 목록을 만든다. 후보마다 predicate input과 reject reason을 확인한다. 선택이 tensor를 repack하거나 scale을 permute하면 representation owner가 바뀐 사실을 기록한다. Runtime shape가 다시 후보를 거절할 때 fallback consumer가 현재 representation을 읽을 수 있는지도 확인한다. 마지막에는 실제 CUDA symbol, copy와 workspace, output reference를 대조한다.

핵심 원칙은 간단하다. Import는 code가 보인다는 뜻이지 이 요청을 처리할 수 있다는 뜻이 아니다. Registration은 후보가 됐다는 뜻이지 선택됐다는 뜻이 아니다. Selection은 effective method가 정해졌다는 뜻이지 모든 phase와 layer가 같은 kernel을 실행했다는 뜻이 아니다. Kernel launch는 계산이 수행됐다는 뜻이지 input representation이 올바르다는 뜻이 아니다.

## 53.1 forced와 auto 요청 하나를 effective backend까지 추적한다

같은 Llama 요청을 두 번 보낸다. 첫 실행은 backend를 `forced=X`로 고정하고, 둘째는 `auto`로 둔다. 두 실행 모두 `요청·모델 representation의 capability → 설치·빌드 상태 → selector가 기록한 effective backend → runtime guard → fallback 또는 reject`의 다섯 칸을 채운다. forced는 capability를 만들어 내지 않으며 auto는 선택 사실을 숨기는 기본값이 아니다. 이후 조건 목록은 이 요청의 어느 칸을 반증하는지 찾기 위한 참고 카탈로그다.

Python import는 module file과 shared library를 찾고 필요한 symbol을 resolve하는 단계다. 여기서 성공해도 extension binary에 현재 GPU용 SASS/PTX가 있는지, driver가 그것을 load할 수 있는지, kernel이 BF16과 head dimension 128을 지원하는지, page layout과 causal mask가 맞는지는 아직 묻지 않았다.

따라서 조사 원장에는 availability를 여러 칸으로 둔다. Package discovered, module imported, extension loaded, registry candidate created, hardware accepted, dtype/format accepted, shape/layout accepted, feature composition accepted, selected, dispatched, completed다. 각 칸을 boolean과 evidence로 기록한다. “available=true” 하나로 합치지 않는다.

**같은 import 성공 뒤 서로 다른 탈락.**

Fixture A는 H100, BF16, head dimension 128, decode `q_len=1`이다. Fixture B는 같은 환경에서 head dimension만 96으로 바꾼다. Package availability는 같다. GPU SM도 같다. Candidate registration도 같다. 그런데 backend가 지원하는 head dimension 집합이 다르면 shape predicate에서 갈라진다.

Fixture C는 shape가 같지만 KV cache dtype을 바꾼다. 이 경우 dtype/layout predicate에서 갈라질 수 있다. Fixture D는 모든 runtime 값이 같지만 prebuilt binary가 현재 CUDA major를 포함하지 않거나 JIT가 disabled다. Package import는 성공해도 executable kernel materialization에서 실패할 수 있다.

이 네 경우의 증상은 “FlashInfer를 쓰지 않았다”로 같아 보인다. 수정은 다르다. Shape를 바꿔야 하는 경우, dtype을 바꿔야 하는 경우, matching wheel을 설치해야 하는 경우, supported fallback을 명시해야 하는 경우를 reject reason으로 구분해야 한다.

**forced와 auto의 실패 정책.**

Auto mode는 후보를 priority 순서로 시도하고 첫 지원 후보를 고를 수 있다. User가 특정 backend를 강제했을 때도 똑같이 조용히 다음 후보로 내려갈지는 policy다. 강제의 의미가 “선호”인지 “반드시 사용”인지 문서와 source에서 확인한다.

Strict forced mode라면 unsupported predicate에서 startup 또는 request를 reject하는 편이 예측 가능하다. Permissive mode라면 effective backend와 reject reason을 사용자에게 노출해야 한다. Raw config만 metrics에 남기면 operator는 원하는 backend가 실행된다고 오해한다.

강제 backend가 일부 phase만 지원할 수도 있다. Prefill은 forced candidate를 쓰고 decode는 다른 candidate를 쓰는 것이 허용되는지 확인한다. “attention_backend=F”라는 global name이 phase별 effective class를 숨기지 않게 한다.

## 53.2 선택에는 다섯 generation이 있다

첫 generation은 requested다. CLI, config file, environment variable, model metadata와 code default 가운데 무엇이 값을 제공했는지 기록한다. 둘째는 normalized다. Alias가 canonical enum으로 바뀌고 deprecated option이 새 field로 이동하며 model capability나 server override가 값을 조정할 수 있다.

셋째는 candidate set이다. Registry에 등록됐고 현재 selector가 고려하는 class 목록과 priority다. 넷째는 effective다. Predicate를 통과해 layer와 phase가 실제로 보유한 method/backend object다. 다섯째는 dispatched다. Runtime shape와 state를 보고 실제 branch가 호출한 operator와 CUDA kernel family다.

**Raw와 effective를 한 log line에 섞지 않는다.**

Operator가 `auto`를 요청했다면 effective가 FlashAttention이어도 raw는 auto다. 반대로 FlashInfer를 요청했는데 override가 FlashAttention으로 바꿨다면 raw와 effective가 다르다. Log가 “backend=flashinfer”라고만 쓰면 그것이 request인지 selection인지 알 수 없다.

Structured trace는 `requested_source`, `requested_value`, `normalized_value`, `override_owner`, `candidate_order`, `selected_class`, `phase`, `layer_kind`, `representation`, `dispatch_op`를 분리한다. Dynamic dispatch라면 request마다 또는 shape bucket마다 effective dispatch가 달라질 수 있으므로 startup metric만으로 충분하지 않다.

**Cache된 selector 결과의 key를 읽는다.**

Selector가 expensive predicate를 피하려고 결과를 cache할 수 있다. Cache key가 dtype, head dimension, block size, sliding window, phase 같은 모든 relevant input을 포함해야 한다. 누락된 field가 있으면 첫 fixture의 결과가 다음 fixture에 잘못 재사용된다.

Cache hit 자체는 correctness 증거가 아니다. Key material과 effective backend를 함께 기록한다. Config mutation 뒤 selector cache를 invalidate하는지도 본다. Process-global registry나 cached function이 여러 model instance 사이에 state를 공유하면 model A의 선택이 B에 번질 수 있다.

## 53.3 vLLM attention selector를 따라간다

vLLM의 [`get_attn_backend`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/selector.py#L101-L207)는 attention config와 layer parameters에서 backend를 결정하는 중심 경계다. Selected backend가 요구하는 KV cache layout에 맞춰 config를 조정하는 부분도 있으므로 selection이 class 이름만 바꾸는 일이 아니다.

**Selector input snapshot.**

Trace에는 head size, dtype, KV cache dtype, block size, attention type, sliding window, model architecture, platform과 requested backend를 둔다. Cached selector key가 이 중 어떤 값을 포함하는지 확인한다. 같은 model 안에서도 MLA와 일반 attention, encoder와 decoder 또는 layer type이 다르면 class가 달라질 수 있다.

Candidate class가 import 가능하다는 사실보다 `supports_*` 또는 platform selection predicate가 무엇을 보는지 읽는다. Reject reason이 warning string으로만 존재한다면 structured form을 추가하거나 test에서 log를 capture한다.

**Phase별 backend.**

Prefill은 긴 query와 많은 K/V를 처리하고 decode는 query length 1에서 paged cache를 읽는다. 하나의 backend class가 두 phase를 모두 처리할 수 있지만 내부 kernel branch가 다를 수 있다. MLA처럼 prefill selector가 별도로 존재할 수도 있다.

Requested/effective metric에는 phase를 넣는다. Startup에서 “using X”를 한 번 기록하고 모든 runtime을 X라고 집계하지 않는다. Nsight trace는 prefill과 decode NVTX range를 나눠 actual symbols를 확인한다.

**KV layout mutation.**

Backend가 특정 KV layout을 요구해 selector가 config/layout을 조정하면 downstream allocator와 cache writer가 같은 effective layout을 사용해야 한다. Selection cache가 바뀌었는데 이미 cache tensor가 이전 layout으로 할당됐다면 pointer ABI가 깨진다.

Backend hot change를 지원하지 않는다면 model initialization 이후 immutable해야 한다. Dynamic per-request selection이 있다면 representation/layout이 공존 가능한지 또는 request마다 별도 cache pool이 필요한지 확인한다.

## 53.4 quant method는 representation owner를 바꾼다

Quant backend 선택은 attention보다 위험한 추가 단계를 가진다. Candidate가 선택된 뒤 qweight를 repack하고 scales와 zero points를 permute할 수 있다. Conversion이 끝나면 tensor가 checkpoint-native layout이 아니다. 이후 fallback은 현재 representation을 읽을 수 있어야 한다.

**Marlin shape gate.**

vLLM의 [`verify_marlin_supports_shape`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L172-L230)는 partitioned N/K와 group size가 kernel 제약을 만족하는지 검증한다. Global hidden size가 아니라 layer parameter의 local partition이 input이다.

Fixture는 N=4096, K=4096, group size 128에서 TP=1/2/4와 irregular output projection을 비교한다. 각 rank의 local N/K, padding과 group boundary를 표로 만든다. Support가 false면 어떤 generic method가 checkpoint-native tensor를 소유하는지 확인한다.

**Conversion 전후를 별도 generation으로.**

Loader가 qweight를 읽은 직후 hash와 logical unpack sample을 남긴다. Repack 뒤 representation tag, shape/stride, scale permutation과 inverse test를 남긴다. 같은 parameter name을 in-place replace하더라도 owner가 Marlin method로 바뀐 사실을 기록한다.

Runtime M이나 workspace 조건 때문에 Marlin apply가 거절되면 fallback이 repacked layout을 직접 읽는지, inverse conversion을 하는지, 별도 native copy를 보존하는지 본다. 아무 증거 없이 generic GPTQ kernel에 같은 pointer를 넘기면 outer shape가 맞아도 wrong answer가 된다.

**MoE와 layer별 method.**

Dense linear와 MoE expert는 같은 quant config를 받아도 method가 다를 수 있다. Expert count, top-k, EP layout, routed token count와 backend package가 predicate에 추가된다. Model-wide `quantization=fp8` 하나로 모든 layer actual kernel을 설명하지 않는다.

Layer inventory에는 layer type, quant method class, effective backend, representation owner와 dispatch family를 둔다. Router, shared expert, routed expert와 LM head가 서로 다른 path를 쓸 수 있다.

## 53.5 SGLang override와 effective backend를 구분한다

SGLang에는 global attention backend뿐 아니라 prefill/decode backend와 MoE runner, A2A, GEMM backend가 있다. Raw option을 나열하는 대신 하나의 config가 normalization과 hardware/model override를 거쳐 무엇으로 바뀌는지 추적한다.

**Override는 mutation log를 남긴다.**

NVFP4/MXFP4와 device 조건이 특정 MoE backend를 요구하거나 금지할 수 있다. [`overrides.py`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/arg_groups/overrides.py#L1270-L1305)는 이런 조정의 고정점이다. Raw requested, predicate inputs, effective value, owner function과 reason을 기록한다.

사용자 강제값을 override한다면 warning과 final config dump가 필요하다. Auto default를 materialize한 것과 incompatible forced value를 바꾼 것을 같은 mutation으로 세지 않는다. Forced 값이 의미상 strict라면 reject가 더 맞을 수 있다.

**FP4 runner backend.**

[`fp4_utils.py`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/quantization/fp4_utils.py#L124-L166)의 effective runner backend는 FlashInfer 내부에서도 CUTLASS와 CuteDSL 같은 구현을 구분할 수 있다. “FlashInfer 사용”이라는 상위 label만으로 actual kernel family를 알 수 없다.

Package availability, SM과 format/layout predicate를 따라 runner가 어떤 data/scale interleave를 기대하는지 확인한다. Backend 변경이 conversion owner를 바꾸면 load-time preprocess와 runtime runner가 같은 effective value를 사용해야 한다.

**Prefill/decode와 graph path.**

Eager와 CUDA graph runner가 같은 selector state를 소비하는지 확인한다. Capture 때 effective backend와 replay 때 runtime metadata가 합의해야 한다. Larger shape가 capture bucket을 벗어나 eager fallback하면 backend까지 달라질 수 있다.

Trace에는 execution mode, graph bucket, phase, effective class와 kernel을 함께 둔다. Graph fallback count만으로 backend fallback이라고 단정하지 않는다. 같은 backend의 eager path일 수도 있다.

## 53.6 llama.cpp는 graph op부터 device kernel까지 두 번 고른다

llama.cpp model builder는 GGML graph op를 만든다. Backend scheduler는 tensor buffer와 backend가 op를 지원하는지 묻고 graph를 device별 split한다. CUDA backend가 op를 받으면 CUDA 내부 dispatch가 MMVQ, MMQ, cuBLAS 또는 FlashAttention kernel을 다시 고른다.

**supports_op와 kernel selector.**

첫 선택은 “이 op를 CUDA backend가 실행할 수 있는가”다. Input type, output type, shape와 op parameters가 predicate다. False이면 다른 backend 배치나 graph split/copy가 생길 수 있다. True라고 특정 CUDA kernel이 정해진 것은 아니다.

둘째 선택은 CUDA 내부다. Matmul은 weight quant type, rows/columns, batch, compute capability와 kernel support에 따라 갈린다. Actual branch와 symbol을 profile에서 확인한다. GGML op name `MUL_MAT` 하나로 cuBLAS 실행을 단정하지 않는다.

**Fallback과 tensor ownership.**

GGUF-native tensor가 backend-neutral buffer에 있거나 device-specific buffer로 upload될 수 있다. CUDA path가 rejected돼 CPU가 실행할 때 어떤 copy와 conversion이 필요한지 본다. CUDA-specific repacked representation을 CPU consumer가 그대로 읽는다고 가정하지 않는다.

Scheduler가 cross-device copy node를 삽입하면 latency와 memory traffic이 늘어난다. Backend fallback metric에는 selected device/backend뿐 아니라 copy bytes와 sync를 포함한다. Correctness는 original logical tensor coordinate와 CPU/CUDA output을 비교한다.

## 53.7 silent reinterpretation은 crash보다 위험하다

Representation mismatch는 dtype와 outer shape가 같을 때 가장 잘 숨는다. `uint8`는 FP8 code, packed INT8, FP4 data 또는 scale byte view일 수 있다. Consumer가 다른 decoder를 쓰면 pointer와 byte count는 맞고 결과만 틀린다.

**한 coordinate를 끝까지 복원한다.**

Fixture는 logical weight의 서로 다른 coordinate에 구별 가능한 code와 scales를 둔다. Checkpoint-native unpack reference, TP slice, repack, runtime dequant sample을 각각 계산한다. First divergent coordinate가 conversion 전인지 후인지 찾는다.

Scale direct/inverse, group axis, zero bias, nibble/interleave를 별도 열로 둔다. Backend name이 같아도 runner variant가 scale layout을 다르게 요구할 수 있다. Validation이 dtype/shape만 검사하는지 layout tag까지 검사하는지 본다.

**Fallback representation matrix.**

각 effective method에 input representation과 compatible fallback 목록을 만든다. Checkpoint-native GPTQ를 읽는 generic kernel, Marlin-repacked를 읽는 Marlin variants, NVFP4-interleaved를 읽는 runner는 서로 다른 row다. Conversion edge가 없으면 다른 row로 이동할 수 없다.

Fallback에서는 code의 존재보다 호출 시점이 중요하다. Conversion 전 fallback과 conversion 후 fallback은 같은 class 이름이어도 input state가 다를 수 있다. Owner와 representation을 runtime assertion 또는 type로 표현하는 것이 안전하다.

## 53.8 관측 원장과 재현 fixture

한 요청의 selection ledger는 raw config hash, normalized diff, device capability, package/build manifest, candidate order, reject reasons, layer/phase effective method, representation transitions, runtime shape, graph/eager mode, dispatch op/kernel symbol, workspace/copy와 output reference를 가진다.

Metric labels는 backend, phase, method family와 bounded reject code 정도로 제한한다. Model name, exact shape와 arbitrary error text를 label로 넣지 않는다. 세부 predicate inputs는 trace와 debug snapshot에 둔다.

### Profile과 selector log를 함께 읽는다

Nsight kernel symbol은 actual launch evidence다. 하지만 같은 symbol이 여러 representation을 받을 수 있고 wrapper가 preprocessing kernel을 먼저 실행할 수 있다. Selection trace, conversion event와 timeline을 correlation id로 잇는다.

No kernel이 보이면 CPU fallback, graph fusion, profiler range 누락과 request가 실제로 해당 layer를 통과했는지를 구분한다. Expected symbol absence만으로 selector bug라고 결론내리지 않는다.

### Correctness matrix

지원 fixture와 비지원 fixture를 한 축만 다르게 만든다. SM, dtype, head dim, TP local N, group size, phase, graph bucket, scale layout을 각각 바꾼다. Expected accepted/rejected candidate와 output reference를 assert한다.

Backend 사이 reduction order가 달라 small numerical difference가 생길 수 있다. Dtype별 tolerance를 미리 정하고 first coordinate/layer를 기록한다. Constant factor나 structured channel error는 rounding으로 넘기지 않는다.

Correctness matrix의 첫 열은 build와 device다. 동일 source commit으로 CUDA 12.x와 13.x wheel을 만들었을 때 candidate registration, compiled architecture와 lazy JIT 경로가 달라지는지 본다. GPU는 고정하고 toolkit/build만 바꾸는 fixture와 build를 고정하고 SM만 바꾸는 fixture를 나눈다. Driver도 manifest에 남기지만 여러 축을 한 번에 바꿔 원인을 흐리지 않는다.

둘째 열은 attention dtype과 cache dtype이다. BF16 query와 BF16 cache, BF16 query와 FP8 cache, FP16 query와 matching cache처럼 한 값씩 바꾼다. Backend가 cache scale metadata를 요구하면 scale presence, shape와 convention까지 fixture에 포함한다. Dtype enum acceptance만 확인하지 않는다.

셋째 열은 head dimension과 GQA ratio다. Head dim 64/96/128/256, query/KV head ratio를 바꾸고 candidate predicate와 actual kernel을 기록한다. Unsupported head dim이 generic fallback으로 가는지 explicit error인지 확인한다. GQA가 kernel 내부 replication을 요구하는지 metadata가 local heads를 제공하는지도 본다.

넷째 열은 sequence/page shape다. Prefill q_len, decode q_len, total KV length, page size, last-page valid count를 경계값 전후로 둔다. Empty/one-token, exact tile, tile+1, maximum supported를 포함한다. Last partial page에서만 틀리면 general backend selection보다 mask/page layout predicate와 kernel boundary를 본다.

다섯째 열은 quant M/N/K와 group size다. Exact tile, one short, padded, TP-divided와 expert-local shapes를 둔다. Selector accepted 결과뿐 아니라 padding/conversion bytes와 valid output trimming을 assert한다. Padding된 quant code가 mathematical zero를 만들지 않으면 output channels가 오염된다.

여섯째 열은 representation이다. 같은 logical weight를 native, correctly repacked, intentionally wrong interleave 세 artifact로 만든다. Parameter class가 wrong representation을 construction에서 reject하는지 확인한다. Wrong fixture가 kernel까지 도달해 output만 다르면 validation gap이다.

일곱째 열은 feature composition이다. Sliding window 단독, LoRA 단독, CUDA graph 단독이 지원돼도 조합을 pairwise로 시험한다. Prefix cache hit와 decode backend, speculative tokens와 graph bucket, MoE routing과 quant runner 같은 high-risk 조합을 우선한다. 모든 Cartesian product 대신 selector predicate edge coverage를 기준으로 조합을 고른다.

각 test의 expected result는 backend 이름 하나가 아니다. Requested/normalized, candidate reject codes, effective method, representation transitions, dispatched op, output tolerance와 cleanup state를 가진다. 이 expected ledger가 release diff에서 바뀌면 의도된 policy change인지 review한다.

### Selection manifest를 배포 artifact로 다룬다

Model을 load한 뒤 layer별 selection manifest를 생성한다. Base/model revision, engine commit, package/build, device capability, normalized config, layer role, parameter shape/dtype, method class, representation owner와 supported runtime guard domain을 포함한다. 같은 replica group의 rank별 manifest를 비교한다.

Manifest가 다르면 즉시 모두 실패해야 하는지는 차이의 의미에 달려 있다. Rank-local shape/hash는 정상적으로 다를 수 있지만 method ABI와 collective output dtype은 합의해야 한다. Expected rank variance schema를 정의하고 예상 밖 차이만 alert한다.

Hot configuration change를 허용하면 manifest generation을 올린다. Old in-flight request와 CUDA graph가 어느 generation을 snapshot하는지 기록한다. Backend 이름만 바꾸고 parameter representation을 그대로 둘 수 있는지, 재conversion과 graph recapture가 필요한지 transition plan을 만든다.

Rolling deployment에서는 old/new replicas가 다른 effective backend를 쓸 수 있다. Output tolerance와 cache portability를 확인한다. External KV transfer나 shared cache가 있다면 backend layout이 wire/storage contract에 영향을 주는지 별도 검사한다. Backend 변경이 local compute detail이라는 가정은 cache representation이 canonical일 때만 성립한다.

Manifest에는 negative capability도 남긴다. Candidate X가 head dim 96 때문에 rejected됐다는 사실은 다음 upgrade에서 regression/확장 여부를 비교하는 기준이 된다. Support matrix 문서를 수작업으로 유지하기보다 audited fixture의 predicate 결과를 근거로 삼는다.

### Reject reason을 설계하는 법

Reject reason은 사용자 메시지, metric code와 debugging payload를 분리한다. User message는 actionable하지만 secret path와 internal stack을 숨긴다. Metric code는 bounded cardinality다. Debug payload는 observed shape/dtype/device와 required range, source predicate를 가진다.

여러 predicate가 실패하면 첫 failure만 남길지 모두 평가할지 결정한다. Fast selector는 short-circuit할 수 있지만 debug mode에서는 후보 하나의 모든 relevant 실패를 수집하면 수정 후 다음 실패가 나타나는 시행착오를 줄인다. 다만 package import 자체가 불가능하면 이후 runtime predicate는 unknown이지 false로 표시한다.

Candidate priority와 reject reason을 함께 보여 준다. 지원 후보 B가 선택됐더라도 더 높은 A가 왜 탈락했는지 알아야 performance tuning이 가능하다. 선택된 B가 fallback인지 정상 auto choice인지 raw intent와 priority로 판정한다.

Runtime guard reason은 initialization reason과 namespace를 분리한다. `STATIC_SHAPE_UNSUPPORTED`, `DYNAMIC_M_THRESHOLD`, `WORKSPACE_ALLOCATION`, `GRAPH_BUCKET_MISS`, `RUNTIME_ALIGNMENT`처럼 단계가 드러나야 한다. 모든 것을 `backend_unavailable`로 세면 remediation이 불가능하다.

### Representation owner의 불변식

Parameter마다 현재 representation을 정확히 한 owner가 책임진다. Owner는 loader가 만든 object, quant method 또는 backend-specific parameter class일 수 있다. Consumer는 owner가 제공하는 apply/API를 통해서만 pointer를 받는다. Raw tensor attribute를 꺼내 다른 kernel에 넘기는 우회는 representation contract를 깨기 쉽다.

소유권 전환은 atomic해야 한다. Conversion output이 완성되기 전에 effective method를 새 owner로 publish하지 않는다. 실패하면 old owner가 계속 valid하거나 model load 전체가 실패한다. Half-converted parameter inventory를 runtime에 노출하지 않는다.

Serialization 또는 CPU offload가 있으면 representation을 보존하는지 canonical로 되돌리는지 명시한다. Repacked device tensor를 CPU로 copy했다고 checkpoint-native가 되는 것은 아니다. Reload consumer가 representation metadata 없이 byte만 읽으면 silent reinterpretation이 생긴다.

Workspace도 owner가 있다. Kernel A workspace를 B가 같은 shape라서 재사용할 수 있다고 가정하지 않는다. Size, alignment, initialization, stream lifetime과 graph capture ownership을 확인한다. Fallback 전환 시 workspace allocation/free가 sync를 만들 수 있다.

Scale와 auxiliary tensors도 main weight와 같은 generation으로 전환한다. Qweight만 R2이고 scales는 R1이면 outer shapes가 맞아도 결과가 틀린다. Conversion transaction digest는 related tensor set 전체를 포함한다.

### 성능 효과를 원인별로 분해한다

Backend 변경 뒤 tokens/s만 비교하면 왜 달라졌는지 모른다. Selection/plan time, conversion startup time, per-request metadata preparation, H2D copy, workspace allocation, kernel duration, number of launches, synchronization과 fallback transfer를 분리한다.

Cold run에는 JIT compile, autotune, cubin load와 conversion이 포함될 수 있다. Warm run과 섞지 않는다. Dynamic shape가 새 specialization을 만들면 warm-up coverage가 부족할 수 있다. Shape bucket별 cold/warm state를 trace에 둔다.

Primary kernel이 빠르더라도 padding과 repack overhead, small-M inefficiency 때문에 workload 전체에서 fallback이 나을 수 있다. Auto priority가 어떤 objective를 가정하는지 확인한다. Correctness predicate와 performance heuristic을 같은 boolean으로 섞지 않는다. Unsupported는 절대 선택하지 않지만 supported 후보의 우선순위는 workload tuning 문제다.

Phase별 arrival distribution과 TP size를 반영한다. Benchmark fixture의 global shape만 보고 production local shape 선택을 예측하지 않는다. Prefill/decode 비율과 batch M distribution에 따라 dispatch coverage가 바뀐다.

Backend fallback이 CPU 또는 다른 device로 가면 copy와 synchronization이 critical path가 된다. Kernel duration만 보면 “fallback도 빠르다”고 오판할 수 있다. End-to-end phase latency와 GPU/CPU timeline을 함께 본다.

### Wrong answer를 빠르게 localize한다

Output이 틀리면 final text부터 비교하지 않는다. Tokenization과 sampling을 고정하고 first affected layer의 input을 reference와 맞춘다. Projection output, dequantized weight sample, attention output 또는 cache write 중 첫 divergence를 찾는다.

Quant linear에서는 basis input으로 output channel을 분리한다. One-hot 또는 small deterministic vector가 packing/scale axis 오류를 잘 드러낸다. Random activation은 여러 오류가 합쳐져 pattern을 숨긴다. TP에서는 rank-local partial output과 collective 뒤 output을 모두 본다.

Attention에서는 Q/K/V projection이 맞는지 먼저 확인한다. 맞다면 backend input metadata, mask, position, cache address와 output을 본다. Backend 선택 문제와 upstream projection quant 문제를 섞지 않는다. Same attention backend가 raw Q/K/V부터 다르면 attention kernel을 의심할 이유가 없다.

Primary reference와 fallback 실행을 같은 intermediate coordinate에서 비교한다. 최초 불일치가 fallback decode/unpack이면 representation mismatch다. Kernel output까지 같고 final logits가 다르면 downstream method selection을 본다.

NaN/Inf는 dtype/scale/workspace uninitialized 후보지만 finite wrong answer도 심각하다. Validation이 finite check만 한다면 structured misinterpretation을 놓친다. Coordinate reference와 checksum/norm pattern을 조합한다.

### Release upgrade에서 다시 확인할 것

Engine upgrade는 candidate priority, default, predicate 범위, conversion format과 kernel package dependency를 바꿀 수 있다. Config file이 같아도 effective selection이 달라진다. Old/new manifest diff를 release artifact로 검토한다.

Package dependency가 optional에서 bundled로 바뀌거나 prebuilt cubin set이 달라질 수 있다. Import availability change와 runtime predicate change를 분리한다. CUDA toolkit major upgrade는 compile target과 library ABI를 바꿀 수 있지만 model format까지 자동으로 바꾸는 것은 아니다.

New backend가 priority 앞에 추가되면 auto mode 사용자는 별도 option 변경 없이 새 kernel을 쓴다. Correctness fixture와 production-like shape coverage를 통과하기 전 auto rollout을 제한할 수 있다. Forced old backend는 계속 가능한지 확인한다.

Conversion format version이 바뀌면 persisted cache나 serialized repacked tensor의 compatibility를 본다. Runtime-only parameter라면 reload로 해결될 수 있지만 graph cache나 external tensor cache가 representation을 보존하면 invalidation이 필요하다.

Metrics 이름과 effective label도 migration한다. Dashboard가 old backend enum만 알고 new value를 drop하지 않는지 확인한다. Unknown을 other로 합치더라도 alert와 trace에서 exact value를 찾을 수 있어야 한다.

이 operational workbook의 목적은 더 많은 정보를 수집하는 것이 아니다. Requested에서 completed kernel까지 각 generation의 owner와 변환을 최소 evidence로 잇는 것이다. Candidate reject card, layer manifest, representation transition, runtime dispatch와 first-coordinate reference가 있으면 대부분의 backend 사건을 재현할 수 있다.

## 53.9 forced/auto Llama 요청의 effective backend를 재구성한다

이제 하나의 fixture를 startup부터 첫 decode까지 따라가자. Model은 Llama 계열, BF16 activation, GQA, head dimension 128, KV cache BF16이다. Attention은 auto, quantization은 GPTQ 4-bit, group size 128, TP=4다. GPU는 SM90이고 matching CUDA extension package가 설치돼 있다고 하자. Prefill은 1024 tokens, decode는 한 token이다.

수직 재구성은 raw source와 precedence를 고정하는 데서 시작한다.

첫 표에는 CLI, config file, environment, model metadata, compiled default를 모두 적는다. 같은 field를 둘 이상이 제공하면 precedence code를 확인한다. CLI `auto`, environment forced backend와 model override가 동시에 있으면 최종 normalized 값만 보고 원인을 알 수 없다.

Raw source에는 값의 부재도 기록한다. Absent라서 auto default가 적용된 것과 사용자가 명시적으로 auto를 적은 것은 operational intent가 다를 수 있다. Deprecated alias가 새 enum으로 변환되면 warning과 source location을 남긴다.

Normalization 뒤 snapshot을 serialization한다. Object repr가 dynamic state를 생략할 수 있으므로 selector가 실제 읽는 fields를 명시한다. Device discovery 전 normalized config와 device-aware override 후 effective config를 분리한다.

그다음에는 candidate마다 reject card를 만들어 선택되지 않은 이유까지 보존한다.

Attention candidate A, B, C가 있다면 priority와 predicate를 카드로 만든다. Card에는 import status, extension/build id, device capability, supported dtypes, head dims, cache layouts, mask/features, phase와 graph support가 있다. Selector가 short-circuit해 B 이후를 검사하지 않아도 candidate order는 보존한다.

Reject card는 `unsupported` 하나로 끝나지 않는다. `HEAD_DIM_NOT_IN_SET`, `KV_DTYPE_UNSUPPORTED`, `PACKAGE_SYMBOL_MISSING`, `SM_TOO_LOW`, `FEATURE_SLIDING_WINDOW`, `LAYOUT_PAGE_SIZE`처럼 bounded code와 observed/required 값을 둔다. Human message는 추가하되 metric label로 쓰지 않는다.

Accepted card도 조건부다. Prefill accepted와 decode accepted를 분리하고 maximum sequence, page size, graph capture 가능 범위를 적는다. Startup acceptance가 모든 runtime M과 q_len을 포괄하지 않으면 dynamic guard를 명시한다.

### 53.9.1 quant method와 attention을 독립 원장으로 둔다

한 request의 attention backend와 quant GEMM backend는 서로 다른 selector다. FlashInfer attention이 선택됐다고 GPTQ linear도 FlashInfer가 처리하는 것은 아니다. Layer마다 attention op, QKV projection, output projection, MLP, LM head의 method를 별도 row로 둔다.

QKV projection은 GPTQ checkpoint-native tensor를 loader가 읽고 Marlin eligibility를 평가한다. TP=4 뒤 local output widths와 packed segment가 tile 조건을 만족하는지 확인한다. MLP gate/up/down은 다른 N/K shape를 가져 같은 model에서도 eligibility가 다를 수 있다.

Layer table은 raw quant method, concrete parameter class, global/local shape, conversion, representation owner, apply method, kernel family를 가진다. “Model uses Marlin”이라는 하나의 flag로 축약하지 않는다.

### 53.9.2 prefill timeline을 해부한다

Request가 admitted되면 model runner가 prefill batch tensors와 attention metadata를 만든다. 이 시점의 q_len, sequence lengths, block table, causal/sliding state와 graph/eager mode를 selector/dispatch trace에 붙인다. Startup selected class와 runtime branch가 같은지 확인한다.

QKV projection은 quant method의 apply를 호출한다. Repacked qweight pointer, scales, zero point/g_idx, M/N/K, workspace와 output dtype을 기록한다. Actual CUDA symbol과 preprocessing/copy kernels를 timeline에 연결한다. Attention은 selected backend wrapper, plan/workspace, kernel symbol과 KV write layout을 기록한다.

Long prefill이 chunked되면 각 chunk의 q_len과 effective branch가 다를 수 있다. 첫 chunk와 마지막 chunk가 같은 backend인지 확인한다. Backend change가 허용돼도 KV layout과 numerical state가 compatible해야 한다.

### 53.9.3 decode timeline과 phase split을 비교한다

Decode는 M과 q_len이 작아 projection kernel과 attention kernel 선택이 달라질 수 있다. Weight representation은 같은 resident parameter이므로 dynamic fallback consumer가 repacked layout을 읽을 수 있어야 한다. Prefill Marlin, decode another kernel이라면 두 kernel이 같은 repacked representation을 지원하거나 별도 native copy가 있어야 한다.

Attention decode는 paged cache layout, block table, KV dtype와 page size를 predicate로 본다. Prefill backend가 contiguous temporary를 썼더라도 decode가 읽는 persistent cache format은 합의되어야 한다. First decode에서 cache read sample을 reference와 비교한다.

Phase metric은 prefill/decode를 분리하고 actual token/launch count를 집계한다. Request-level label 하나로 두 backend를 표현하지 않는다. TTFT regression은 prefill, ITL regression은 decode timeline과 연계한다.

### 53.9.4 TP-local rejection을 손으로 계산한다

Global linear weight `[N=4096,K=4096]`가 있다고 하자. Column-parallel TP=4면 local N=1024다. Group size가 K axis 128이면 K group boundary는 유지될 수 있다. 다른 layer가 N=11008이라면 padding/partition convention에 따라 local N이 2752 또는 padded value가 된다.

Marlin shape predicate가 N/K tile family와 group size를 어떻게 검사하는지 대입한다. Padding이 허용되면 mathematical zero가 되는 packed code와 scale/ZP를 사용하고 logits invalid rows를 mask하는지 본다. Padding cost와 workspace도 기록한다.

Uneven shard라면 rank마다 candidate 결과가 달라질 수 있다. Global method는 모든 rank가 지원하는 공통 backend를 골라야 하거나 rank별 다른 kernel 뒤 collective correctness를 증명해야 한다. Rank 0만 selection log를 남기지 않는다.

### 53.9.5 conversion transaction을 기록한다

Checkpoint-native parameter를 `R0`라고 하자. TP slice 뒤 `R1`, Marlin repack과 scale permutation 뒤 `R2`다. 각 transition은 input/output representation, owner function, tensor hash/sample, reversibility와 native copy 보존 여부를 가진다.

Conversion 중 실패하면 parameter가 일부 layer만 R2가 되지 않게 transaction boundary를 본다. Model-wide rollback이 어렵다면 layer별 owner가 정확히 표시되어 dispatch가 각 representation에 맞는 consumer를 고르게 해야 한다. 같은 class field가 반쯤 갱신된 상태를 publish하지 않는다.

Post-load memory accounting은 R0/R1 사본이 해제됐는지 포함한다. Fallback을 위해 native copy를 보존하면 memory cost가 늘지만 compatibility가 넓어진다. 사본을 없애면 runtime fallback 범위를 R2-compatible kernels로 제한해야 한다.

### 53.9.6 runtime guard가 다시 거절할 때

Initialization predicate는 static N/K/format을 확인했지만 runtime M, token count, workspace와 graph state는 실행 때 알 수 있다. Apply method가 M threshold 또는 workspace allocation 실패로 primary kernel을 거절할 수 있다.

이 guard는 initialization selection과 같은 representation contract를 알아야 한다. Candidate list를 `primary R2`, `fallback R2`, `convert R2→R1 then native`, `reject`로 명시한다. Generic 이름만 가진 fallback list는 위험하다.

Workspace OOM을 backend unsupported와 같은 code로 처리하지 않는다. Capacity를 늘리거나 batch를 줄이면 해결되는 transient failure와 format/layout incompatibility를 구분한다. Retry가 다른 backend를 선택하면 representation과 output tolerance를 다시 검증한다.

### 53.9.7 CUDA graph capture와 replay

Capture는 특정 batch shape, pointers, workspace와 dispatch branch를 고정한다. Capture 때 primary kernel A를 썼다면 replay input이 같은 predicate domain에 있는지 graph bucket admission이 보장해야 한다. Larger shape를 억지로 같은 graph에 넣지 않는다.

Graph miss로 eager path를 쓰면 selector 또는 apply guard가 B를 고를 수 있다. Metric에는 graph hit/miss와 effective kernel change를 별도 event로 둔다. Graph miss가 늘어 latency가 증가한 것과 backend B 자체가 느린 것을 분리한다.

Representation pointer는 graph lifetime 동안 안정해야 한다. Post-load conversion이나 hot backend reconfiguration으로 parameter storage가 교체되면 graph를 invalidate/recapture해야 한다. 주소만 같고 content representation이 바뀌는 경우도 generation으로 막는다.

### 53.9.8 fallback의 수치 동등성을 판정한다

Primary와 fallback output을 같은 reference와 비교한다. Exact bit equality를 항상 요구하지 않지만 dtype, accumulation과 reduction order에 맞는 absolute/relative tolerance를 사전 정의한다. Random prompt final token만 비교하지 않고 작은 basis input과 first linear output을 본다.

Structured error는 representation bug 신호다. Output channel마다 일정 scale이 틀리면 scale axis/direct-inverse를, 일정 주기로 값이 섞이면 packing/tile permutation을, TP rank boundary에서만 틀리면 local slice를 본다. Small unstructured error는 accumulation order 후보지만 first divergence를 확인한다.

Fallback이 CPU로 이동하면 dtype conversion과 transfer를 포함해 reference를 본다. CPU 결과가 맞아도 repeated H2D/D2H와 synchronization이 latency를 지배할 수 있다. Correctness와 performance 판정을 분리한다.

이 내부 선택 사슬은 effective state로 번역해 운영자에게 보여 줘야 한다.

Startup manifest에는 requested/normalized backend와 device/build facts, layer/phase selected methods, parameter representations를 둔다. Runtime metrics에는 dispatch family, phase, fallback/reject code, graph mode와 counts를 둔다. Exact shape와 model identity는 high-cardinality label 대신 trace에 둔다.

정상 auto selection과 장애는 fallback 비율만으로 구분하기 어렵다. Candidate reject reason, primary attempts, compatible fallback success, explicit reject와 wrong-output guard failure를 별도 count로 둔다. Forced backend mismatch는 별도 warning/health state로 노출한다.

Profile capture에는 correlation id와 NVTX ranges를 사용해 request phase, layer category, method와 kernel timeline을 잇는다. Production metric이 주장하는 effective backend와 sampled profile symbol이 주기적으로 일치하는지 audit한다.

마지막으로 재현 bundle을 닫고 최종 판정을 쓴다.

Bundle에는 current revisions, package/build manifest, device capability, raw/effective config diff, candidate cards, layer inventory, R0/R1/R2 samples, runtime shape, graph state, kernel symbols와 reference output이 들어간다. Weight 전체나 customer prompt 없이 synthetic coordinate fixture로 재현한다.

좋은 판정은 최초 divergence를 말한다. “TP=4 MLP down projection의 local K가 Marlin gate를 통과하지 못해 initialization은 native GPTQ method를 유지했다. Attention과 다른 linear는 Marlin이어서 model-wide label만 잘못됐다. Layer별 effective metric을 추가했고 output은 native reference와 일치했다.”

Wrong answer 판정은 더 엄격하다. “R2 Marlin repack 뒤 decode M threshold가 generic R0 consumer를 호출했다. First unpack coordinate에서 달랐고 kernel 이후가 아니었다. R2-compatible fallback만 허용하고 없을 때 reject하도록 수정했으며 M boundary와 TP fixture를 통과했다.”

## 53.10 선택 사슬을 실제 운영에 적용한다

새 stack을 만났을 때 option 문서부터 외우지 않는다. Requested 값이 어디서 오고 누가 normalize하는지 찾는다. Candidate registry와 priority를 찾고 package, SM, dtype/format, shape, layout, feature predicate를 펼친다. Effective class가 정해지면 tensor representation의 owner가 바뀌는지 확인한다.

그 다음 runtime dispatch를 본다. Phase, layer, graph/eager와 dynamic shape가 actual branch를 바꾸는지 확인한다. Fallback이 있다면 현재 representation을 읽을 수 있는지, conversion edge 또는 native copy가 있는지 증명한다. 마지막에 kernel symbol과 reference output을 연결한다.

좋은 판정은 “FlashInfer가 안 됐다”가 아니다. “Package와 extension은 available했지만 decode head dimension 96이 candidate predicate를 통과하지 못해 effective backend가 X가 됐다. KV representation은 conversion 전이라 X와 compatible했고 phase metric과 profile symbol이 일치했다.” 또는 “Marlin repack 뒤 dynamic M fallback이 native GPTQ consumer를 호출해 first unpack coordinate가 달라졌다. Repacked-compatible fallback으로 제한하고 unsupported shape는 reject했다.”처럼 쓴다.

설정 이름, import, registry, selection, conversion, dispatch와 completion을 분리하면 backend 문제는 운이 아니라 재현 가능한 의사결정 문제가 된다. 성능을 높이는 선택도 correctness를 지키는 fallback도 그 사슬이 보일 때만 설명할 수 있다.

이 사슬을 code review에 적용할 때는 selector 함수 하나만 보지 않는다. Config field를 선언하는 곳, default와 override를 만드는 곳, candidate를 등록하는 곳, predicate가 device와 shape를 읽는 곳, method가 parameter를 만드는 곳, loader가 conversion하는 곳, forward가 apply를 부르는 곳, runtime guard와 fallback branch, wrapper 아래 native op registration까지 이동한다. 한 파일의 local reasoning으로는 owner transition을 볼 수 없다.

첫 번째 review 질문은 값의 세대다. Raw option이 enum으로 normalize된 뒤 다른 field에 복사되는지, model-specific adjustment가 overwrite하는지, phase-specific field가 global field를 inherit하는지 확인한다. Final config dump가 selector가 읽은 object와 같은 generation인지도 본다. Dump 이후 lazy mutation이 있으면 운영자가 보는 값과 실행 값이 다르다.

두 번째 질문은 predicate input의 진실성이다. Device capability를 current rank device에서 읽는지, global hidden size가 아니라 TP-local parameter를 보는지, configured dtype가 아니라 actual loaded tensor dtype를 보는지, page size와 cache layout이 allocator effective value와 같은지 확인한다. Stale 또는 derived-before-mutation 값을 읽으면 predicate 자체가 정확해도 결과는 틀린다.

세 번째 질문은 predicate의 범위다. Initialization에서 확인한 static facts와 runtime guard가 확인할 dynamic facts를 나눈다. Static acceptance를 “모든 batch 지원”으로 확대하지 않는다. Runtime guard가 새로운 rejection을 만들 수 있다면 candidate fallback list와 representation domain을 construction 때 준비한다.

네 번째 질문은 선택 결과의 소비다. Effective backend enum이 설정됐지만 layer constructor가 raw field를 다시 읽거나 graph runner가 별도 default를 사용하면 split brain이 된다. 모든 consumer가 canonical effective state 또는 명시적 phase override를 읽는지 찾는다. Metric도 같은 state에서 파생해야 한다.

다섯 번째 질문은 conversion ownership이다. Candidate가 선택되기 전 speculative conversion을 하는지, selection 후 정확히 한 owner가 conversion을 commit하는지, auxiliary scale/ZP/g_idx가 같은 transaction에 들어가는지 본다. Failure와 retry가 conversion을 두 번 적용하지 않는 idempotency도 필요하다. Scale permutation을 두 번 하면 shape는 그대로이고 값만 틀릴 수 있다.

여섯 번째 질문은 fallback의 방향이다. Primary가 거절된 시점의 current representation을 적고 fallback consumer의 required representation과 비교한다. 둘 사이 edge가 없으면 reject한다. “Generic”이나 “torch”라는 이름은 portable representation을 보장하지 않는다. Wrapper가 내부 custom op를 호출할 수도 있고 parameter object가 이미 backend-specific일 수 있다.

일곱 번째 질문은 실제 dispatch다. Python method 호출만 보지 않고 registered op, dispatcher key, extension wrapper와 CUDA launcher를 잇는다. Lazy compilation, autotune과 kernel specialization이 있으면 first call과 steady state를 구분한다. Profiler symbol이 wrapper name과 다를 수 있으므로 source launch registration을 확인한다.

여덟 번째 질문은 완료와 오류다. Kernel launch API가 성공해도 asynchronous error는 later sync에서 나타날 수 있다. 어느 owner가 error를 request failure로 귀속하는지 본다. Fallback이 launch failure 뒤 재시도되는지, partial output/workspace를 초기화하는지 확인한다. Failed primary output buffer를 fallback이 누적 add하면 wrong answer가 된다.

아홉 번째 질문은 동시성이다. Process-global registry mutation, selector cache, autotune cache와 workspace pool이 여러 model/request를 공유할 수 있다. Key가 model/device/stream/shape/representation을 충분히 구분하는지 본다. Model A가 custom backend를 등록하거나 global attention implementation을 바꾼 뒤 B에 영향을 주는지 fixture를 둔다.

열 번째 질문은 teardown이다. JIT module, graph executable, converted parameter와 workspace가 어느 generation까지 살아 있는지 본다. Model reload 후 old selector cache가 new parameter representation을 가리키지 않아야 한다. Destruction과 in-flight kernel completion의 ordering도 확인한다.

Incident review template는 짧게 유지할 수 있다. 증상과 workload shape, requested/effective state, first rejected candidate와 reason, selected method, representation transition, actual dispatch, first numerical divergence, fallback compatibility, 수정과 boundary fixture다. 이 열을 채울 수 없다면 “backend bug”라는 issue title만 있는 셈이다.

Operator recipe도 같은 순서를 쓴다. 먼저 final effective manifest를 얻는다. Requested name과 비교하고 phase/layer별로 나눈다. 예상 후보가 탈락했다면 package를 재설치하기 전에 reject reason과 input을 본다. Shape/dtype/SM가 지원되면 representation과 runtime guard를 본다. Actual symbol이 다르면 graph/eager와 dynamic branch를 본다. Output이 틀리면 first coordinate로 돌아간다.

성능 recipe는 correctness가 닫힌 뒤 실행한다. 동일 artifact와 output tolerance에서 candidate coverage, conversion cold cost, warm kernel time, planning/workspace/copy, phase latency를 비교한다. Backend를 강제해 더 빠른 kernel symbol을 얻었더라도 fallback rate와 feature disable, cache layout 변화가 end-to-end를 악화시킬 수 있다.

Rollback recipe는 effective state를 이전 값으로 되돌리는 것만이 아니다. Parameter가 이미 repack됐다면 old method가 그 representation을 읽을 수 있는지 확인하고 필요하면 reload/reconvert한다. CUDA graph와 autotune cache를 invalidate하고 runtime manifest generation을 갱신한다. Config string만 되돌리면 mixed generation이 된다.

Multi-node에서는 rank별 capability와 selection을 collect한다. Homogeneous cluster라고 inventory에 적혀 있어도 driver/package 또는 visible device가 다를 수 있다. Effective method와 representation ABI가 collective group에서 합의되는지 startup barrier 전에 검증한다. 한 rank만 fallback해도 output collective가 수치적으로 합쳐질 수 있지만 performance skew와 representation mismatch 위험이 있다.

MoE에서는 expert placement와 backend selection을 함께 본다. Rank-local expert 수, routed token M, expert quant representation과 A2A output layout이 runner predicate에 들어갈 수 있다. Dense fallback이 expert tensor를 같은 방식으로 읽는다고 가정하지 않는다. Empty expert/token case와 load imbalance boundary를 fixture에 넣는다.

Adapter가 활성화되면 base quant kernel과 LoRA delta kernel의 composition이 graph support를 바꿀 수 있다. Base method는 supported지만 adapter wrapper가 eager fallback을 요구할 수 있다. Effective backend metric을 base kernel 하나로 표시하지 않고 composition path를 trace한다. Adapter off/on 한 축만 바꿔 candidate와 symbol을 비교한다.

Prefix cache hit는 prefill 계산을 건너뛰어 observed backend coverage를 바꾼다. Hit-heavy workload에서 prefill backend symbol이 적다고 misconfiguration으로 판단하지 않는다. Cache hit/miss와 scheduled phases를 denominator로 사용한다. Correctness 비교에서는 fresh/hit를 분리하고 cache representation provenance를 확인한다.

Speculative decode는 target/draft와 verify path의 shapes를 바꾼다. Backend가 normal decode를 지원해도 multi-token verify를 다른 kernel로 처리할 수 있다. Requested attention backend 하나를 두 model과 phase에 복사하지 않고 effective selector state를 각각 기록한다.

CUDA graph capture는 fallback policy를 제한할 수 있다. Replay 중 arbitrary branch를 바꿀 수 없다면 bucket admission이 unsupported shape를 eager로 보내야 한다. Graph node 안 primary launch 실패 뒤 다른 kernel을 호출하는 것이 가능한지 source를 확인한다. 불가능하면 capture 전에 predicate를 완전히 평가한다.

Memory pressure도 selection과 섞일 수 있다. Workspace allocation 실패가 slower low-workspace fallback으로 이어질 수 있지만 parameter representation은 compatible해야 한다. OOM retry가 batch를 줄이는지 backend를 바꾸는지, 기존 output/workspace를 cleanup하는지 trace한다. OOM을 hardware unsupported로 permanent cache하지 않는다.

Autotune 결과는 support predicate가 아니다. Candidate들이 모두 correct/supported인 domain 안에서 성능을 고른다. Autotune failure나 missing profile 때문에 unsupported kernel을 고르거나 supported path를 영구 배제하지 않는다. Cache key에 SM, dtype, shape, representation과 version이 들어가는지 확인한다.

마지막으로 문서의 backend 지원표는 출발점이지 runtime evidence가 아니다. Current pinned source predicate와 actual build/device manifest가 더 구체적인 denominator다. 문서가 지원한다고 해도 local package가 빠질 수 있고, source가 branch를 갖고 있어도 wheel이 kernel을 포함하지 않을 수 있다. 반대로 newer JIT plugin이 추가 capability를 제공할 수 있으므로 observed extension identity를 남긴다.

완성된 selection trace를 열면 숫자와 이름이 아니라 인과가 보여야 한다. User intent가 어떤 normalization을 거쳐 어떤 후보를 만들었고, 각 후보가 어느 정확한 사실 때문에 탈락했으며, 선택된 owner가 tensor를 어떻게 바꾸었고, runtime이 어떤 kernel을 실행했으며, fallback이 왜 안전했거나 거절됐는지가 이어져야 한다. 그 인과가 끊기지 않을 때 backend tuning과 장애 복구가 같은 언어를 쓴다.

### 53.10.1 대표 옵션을 state transition으로 읽는 실전 표

옵션 설명은 “어떤 backend를 사용한다”로 끝나면 쓸모가 없다. 값을 바꿨을 때 parser state, normalized enum, selector inputs, parameter representation, runtime dispatch와 observable effect 가운데 무엇이 달라지는지를 써야 한다. 다음 표의 행은 제품 전체 지원표가 아니라 고정 source에서 실제 추적할 질문이다. 배포자는 current build와 model에 대해 observed 열을 채운다.

| raw option 또는 artifact field | normalized state | selector predicate | runtime guard | 성공 시 변화 | 안전한 fallback 조건 |
|---|---|---|---|---|---|
| vLLM attention backend | canonical backend enum 또는 auto | platform·dtype·head size·KV dtype·block/layout·attention type | phase·query shape·graph/eager | attention impl, metadata builder, KV layout와 kernel family | 같은 logical Q/K/V·mask·cache contract |
| vLLM quantization method | layer별 quant config/method | bits·group·act-order·ZP·local K/N·SM/build | runtime M·workspace·operator domain | parameter class, repack, scale/ZP permutation, GEMM family | current representation을 읽는 consumer |
| vLLM cudagraph mode | capture policy와 size buckets | operation/feature capture safety | actual batch/token bucket·pointer generation | graph replay 또는 eager dispatch | eager path의 method·representation 동등성 |
| SGLang attention backend | global 또는 phase-specific effective backend | model architecture·device·dtype·cache·feature override | prefill/decode·ragged shape·graph state | metadata/layout, plan/run wrapper와 kernel | phase 경계 cache 좌표가 동일 |
| SGLang MoE runner | effective runner backend | quant format·EP/A2A·expert shape·SM·dependency | routed token M·empty expert·workspace | expert layout/interleave, runner와 communication | expert/scale representation compatible |
| SGLang FP4 runner | CUTLASS/CuteDSL 등 concrete runner | NVFP4 recipe·SM·scale layout·alignment | activation shape·padding·workspace | scale interleave와 FP4 GEMM ABI | runner layout id가 같은 consumer |
| llama.cpp GPU layers/device | graph tensor placement policy | buffer support·op support·memory capacity | scheduler split와 copy readiness | CPU/GPU placement와 transfer graph | canonical GGUF block type를 지원하는 backend |
| llama.cpp flash attention | operation preference/effective path | head/type/mask/architecture support | sequence shape와 workspace | fused attention 또는 decomposition | 동일 mask·position·accumulator contract |

첫 행을 예로 들면 raw가 `auto`일 때 normalized도 단순 문자열 auto로 남을 수 있지만 selector는 priority 후보를 구성한다. Raw가 특정 enum이어도 platform class가 support check를 수행한다. Selected backend가 KV layout을 바꾸면 allocator와 writer가 그 state를 소비해야 한다. 실제 request에서 내부 branch가 prefill/decode kernel을 다시 나눌 수 있다. 따라서 option effect는 “FlashInfer 활성화”가 아니라 이 state mutation 전체다.

두 번째 행에서 quant method는 더 강한 side effect를 가진다. Static selector가 Marlin을 고르면 loader 이후 qweight만이 아니라 scale, ZP와 g_idx ordering까지 바뀔 수 있다. Runtime M guard가 generic path로 내려갈 때 method name만 바꾸면 안 된다. Parameter가 어느 generation인지에 맞는 consumer edge가 필요하다. 이 edge가 없으면 unsupported error가 올바른 결과다.

CUDA graph 옵션은 단순 performance toggle처럼 보이지만 dispatch lifetime을 바꾼다. Capture는 kernel nodes, parameter/workspace addresses와 method generation을 고정한다. Bucket miss가 eager로 내려가면 동일 representation을 소비하는지 확인해야 한다. Option을 off로 바꾼 실험은 launch mechanism뿐 아니라 selector/runtime branch까지 달라질 수 있으므로 변경 집합을 기록한다.

SGLang phase backend는 prefill과 decode를 별도로 설정할 수 있어 상위 label 하나가 더 위험하다. Prefill writer가 만든 KV page를 decode reader가 다른 backend로 읽는다면 logical token/head/dim에서 physical page/offset으로 가는 식이 합의돼야 한다. Backend 이름이 다르더라도 canonical cache layout을 공유하면 안전할 수 있고, 이름이 같더라도 runner version과 layout generation이 다르면 안전하지 않을 수 있다.

MoE runner에서는 routed token M이 request마다 변한다. Static model shape가 supported여도 expert별 M=0, 작은 M, tile boundary와 overload에서 runtime branch가 달라질 수 있다. EP rank가 보유한 local expert와 quantized scale interleave도 predicate input이다. Fallback이 dense GEMM을 여러 번 부른다면 expert tensor view와 output scatter가 같은 routing semantics를 보존해야 한다.

llama.cpp의 GPU option은 모든 연산을 한 kernel family로 보내는 스위치가 아니다. Graph tensor placement, backend `supports_op`, split/copy와 CUDA 내부 kernel selection이 순서대로 일어난다. GPU layer 수를 늘린 뒤 CPU fallback이 줄었는지 보려면 node별 placement와 copy를 봐야 한다. CUDA kernel 하나가 보였다는 사실로 graph 전체가 GPU라고 쓰지 않는다.

### 53.10.2 vLLM selector의 raw와 effective를 분리해 기록한다

고정된 vLLM selector source를 읽을 때 public option 이름에서 시작하되 selector function의 실제 arguments로 내려간다. Head size, dtype, KV cache dtype, block size, attention type, sliding window와 platform이 어떤 object에서 공급되는지 찾는다. Default 또는 global config가 layer-specific parameter로 바뀌는 지점을 기록한다. 함수 call이 cached라면 cache key에 모든 semantic input이 있는지도 본다.

Candidate class가 생성되기 전에 optional module import check가 있을 수 있다. 이 결과는 installation state다. Class의 support predicate가 head size나 dtype을 거절하는 것은 capability state다. Class가 선택된 뒤 metadata builder가 cache layout을 잘못 해석하는 것은 numerical contract다. Selector 한 파일을 읽더라도 세 판정을 다른 열에 둔다.

예를 들어 head dimension 96이 candidate A에서 reject되고 B가 선택됐다고 하자. Raw/effective mismatch는 정상 auto policy일 수 있다. 이때 A의 reject code, B의 selected class, B가 요구하는 KV cache layout와 actual kernel을 기록한다. B output이 reference와 맞으면 correctness fallback이고, latency만 바뀌면 performance/SLO policy 문제다. B가 mask 또는 scale semantics를 다르게 읽으면 correctness incident다.

`attention_backend=A`처럼 강제한 경우는 policy를 추가한다. Strict라면 predicate false에서 startup을 실패시키고, permissive라면 B 선택과 이유를 final config 및 metric에 노출한다. Source가 실제로 어느 정책을 구현하는지 확인하며 사용자 기대를 임의로 덧붙이지 않는다. Distributed worker마다 다른 policy result가 나오면 request admission 전에 coordinator가 합의한다.

Selector가 config object를 mutate하면 호출 순서도 중요하다. KV layout가 A에 맞게 바뀐 뒤 runtime guard가 B로 fallback할 수 있다면 B가 A layout을 읽는지 확인한다. Allocation 전 fallback은 새 layout로 만들 수 있지만 allocation/capture 뒤 fallback은 migration 또는 compatible consumer가 필요하다. “같은 attention tensor”라는 말은 physical cache ABI를 증명하지 않는다.

### 53.10.3 vLLM quant 옵션의 consumer까지 내려간다

Quant config는 model 전체 label이지만 effective method는 layer별이다. Dense projection, stacked QKV, gate/up, LM head, embedding과 MoE expert가 서로 다른 parameter class와 backend를 가질 수 있다. Inventory에는 qualified layer name, global/local K/N, bits/group/ZP/act-order, method class와 representation owner를 쓴다.

Marlin shape support source는 local partition을 검사한다. TP를 바꾸면 global model config는 같아도 local N/K와 group/tile boundary가 달라진다. Predicate의 observed와 required 값을 rank별로 계산한다. False일 때 conversion 전 native method를 유지하는지, 일부 parameter를 만든 뒤 fallback하는지 source lifecycle을 본다.

Option 하나의 상태 변화 예시는 다음과 같다. Raw GPTQ config의 `desc_act=true`가 normalized method에 보존된다. Parameter creator가 g_idx storage를 만든다. Loader가 g_idx를 채운다. Post-load method가 g_idx를 sort하고 qweight row permutation을 만든다. Runtime GEMM이 activation permutation 또는 g_idx argument를 소비한다. 어느 한 단계가 빠지면 option이 parse됐어도 효과가 완성되지 않는다.

`group_size`도 단순 accuracy knob가 아니다. Scale tensor rows, g_idx lookup, packed shard alignment, kernel specialization과 metadata traffic을 바꾼다. `-1` 같은 sentinel은 normalized 단계에서 channel-wise 의미가 될 수 있다. Reader는 constructor validation, parameter shape 식, selector predicate와 dequant consumer를 이어야 “값을 바꾸면 무슨 일이 생기는가”에 답할 수 있다.

Backend option과 quant option이 충돌하면 precedence를 본다. User가 Marlin을 강제했지만 format/group/SM이 unsupported라면 override, fallback 또는 reject 가운데 source policy가 있다. Override가 tensor conversion 전인지 후인지에 따라 rollback 비용이 다르다. Final log에는 requested method, effective per-layer method와 conversion generation이 있어야 한다.

### 53.10.4 SGLang override를 option 무시로 오해하지 않는다

SGLang override source는 device, model과 quant format 조건으로 server arguments를 조정할 수 있다. 이 코드를 “사용자 값을 무시한다” 또는 “자동 최적화한다”로 먼저 평가하지 않는다. Predicate가 correctness capability를 지키는지 performance preference를 적용하는지 구분한다. Required ABI 때문에 한 runner만 가능한 경우와 benchmark 우선순위 때문에 고르는 경우는 실패 정책이 달라야 한다.

Raw, normalized, overridden, effective, dispatched의 다섯 값을 log schema에서 구별한다. Default materialization은 raw가 unset일 때 effective를 채운다. Alias normalization은 의미를 보존한다. Correctness override는 incompatible 조합을 다른 값으로 바꾼다. Performance override는 supported 후보 중 priority를 바꾼다. Dynamic dispatch는 request shape에서 operator를 바꾼다.

FP4 runner option은 quant artifact와 load-time conversion에 연결한다. ModelOpt/NVFP4 data와 block/global scales가 어느 canonical shape로 load되고, 어느 helper가 runner-specific interleave를 만드는지 본다. Effective runner가 바뀌면 conversion cache key와 graph generation도 바뀌어야 한다. 같은 `uint8` data와 scale numel은 layout compatibility가 아니다.

MoE option은 communication backend와 compute runner 조합을 본다. A2A output layout, local expert ordering, routed token counts와 quantized expert representation이 runner input ABI와 맞아야 한다. 각각 단독 supported라는 사실로 조합을 승인하지 않는다. Empty expert, top-k boundary와 EP rank mapping fixture를 둔다.

Prefill/decode option은 attention phase뿐 아니라 graph capture path를 포함한다. Capture manifest가 effective backend와 workspace를 고정하고 replay admission이 current metadata generation을 검사하는지 본다. Graph miss가 eager로 갈 때 backend가 달라지면 별도 fallback event다. 같은 backend의 eager variant면 graph fallback과 backend fallback을 구분한다.

### 53.10.5 metric이 option intent를 실행 사실처럼 보이게 하지 않는다

Metric에는 requested와 effective를 다른 이름으로 노출한다. Requested value는 configuration inventory에 적합하고 자주 바뀌지 않는다. Effective backend는 layer/phase가 많아 고카디널리티가 될 수 있으므로 bounded family와 reason code를 metric에, exact layer와 shape는 sampled trace에 둔다. Actual dispatch coverage는 scheduled operation을 denominator로 계산한다.

`fallback_total` 하나는 부족하다. Static selector reject, runtime guard, graph bucket miss, workspace failure, installation/materialization과 execution error를 나눈다. Correctness-compatible fallback인지도 diagnostic state에 둔다. 그러나 compatibility를 매 request마다 긴 label로 내보내지 않고 validated edge id/version을 사용한다.

Latency histogram은 backend label만으로 해석하지 않는다. Prefill/decode/verify, graph/eager, shape bucket과 fallback reason을 bounded cohort로 나눈다. Exact M/N/K를 label로 넣지 않는다. Rare shape tail은 trace exemplar에서 확인한다. Expected backend symbol count를 request count가 아니라 실제 scheduled layer-operation 수와 비교한다.

Numerical monitoring은 모든 intermediate tensor를 저장하지 않는다. Canary fixture의 exact coordinates, sampled layer checksum/norm, first-token logit parity와 non-finite count를 단계적으로 사용한다. Structured channel error가 tolerance 안에 숨을 수 있으므로 식별 fixture는 별도로 exact tuple을 검사한다. User content와 raw weight를 운영 log에 남기지 않는다.

Fallback이 latency만 바꿨다고 판정하려면 numerical contract가 먼저 닫혀야 한다. Reference tuple과 output tolerance를 통과한 뒤 copy bytes, workspace, launch count와 kernel duration을 비교한다. Correctness unknown 상태에서 SLO만 복구됐다고 incident를 닫지 않는다. 반대로 correctness fallback이 느리다는 이유로 unsupported primary를 강제하지 않는다.

## 53.11 fallback은 성공했지만 계약이 바뀐 사건

### 53.11.1 raw option에서 실제 kernel까지 한 줄로 잇는다

뒤에서 해부할 사건의 결과부터 보자. 운영자는 Marlin을 요청했고 startup manifest도 Marlin을 가리켰지만, 긴 prompt의 speculative verify 구간에서는 runtime guard가 generic operator를 골랐다. 응답은 성공했으나 첫 dequantized weight coordinate와 ITL tail이 함께 달라졌다. 이제 이 결과를 만든 여섯 경계를 거꾸로 펼친다.

사고를 보기 전에 옵션 하나를 끝까지 추적하는 표준 문장을 만든다. `raw attention backend=flashinfer`는 사용자가 입력한 의도다. Parser와 alias normalization 뒤 canonical enum이 만들어진다. Model, platform와 feature override가 effective preference를 바꿀 수 있다. Selector는 installed candidate 가운데 dtype, head dimension, cache layout와 phase predicate를 통과한 class를 고른다. Runtime guard는 실제 q_len, graph bucket과 workspace 상태로 다시 branch를 고른다. 마지막 wrapper와 extension이 CUDA kernel을 launch한다. 이 여섯 문장을 생략하지 않는다.

Quant 옵션도 같은 구조지만 representation mutation이 추가된다. `quantization=gptq`와 `bits=4`, `group_size=128`, `desc_act=true`가 raw 또는 checkpoint state에서 들어온다. Normalizer가 method/version을 확정하고 layer별 selector가 generic GPTQ, Marlin 또는 다른 구현의 후보를 만든다. Static shape predicate가 local K/N과 group alignment를 검사한다. 선택된 method가 qweight·scale·ZP·g_idx를 repack하면 representation generation이 바뀐다. Runtime M guard가 어느 operator를 호출하는지와 그 operator가 현재 generation을 읽는지가 마지막 계약이다.

SGLang에서는 raw attention backend, prefill/decode backend, MoE runner와 FP4 runner를 한 필드처럼 합치지 않는다. Global raw 값이 phase별 effective 값으로 분화할 수 있고, model/format override가 user preference를 바꿀 수 있다. `FlashInfer`라는 상위 이름 아래 CUTLASS와 CuteDSL runner가 다른 scale layout을 요구할 수도 있다. 최종 원장에는 상위 backend와 실제 runner를 모두 둔다.

각 단계에는 값뿐 아니라 소유자를 적는다. CLI parser, default provider, override function, selector, quant method, runtime wrapper와 native launcher다. “값이 바뀌었다”보다 누가 어떤 predicate와 reason으로 바꿨는지가 중요하다. Cache된 selection이라면 cache key와 generation owner도 넣는다. 관측되지 않은 단계는 추측으로 채우지 않고 unknown으로 남긴다.

### 53.11.2 지원 가능성·설치 상태·수치 계약을 세 장으로 나눈다

첫 장은 support capability다. Current pinned source가 특정 SM, dtype, shape, layout와 feature 조합을 처리하는 branch를 갖는지 묻는다. Source에 branch가 있으면 “지원 가능” 후보지만 local artifact가 그 code를 포함한다는 뜻은 아니다. Predicate가 false인 조합은 package를 다시 설치해도 지원되지 않는다.

둘째 장은 installation/materialization이다. Python module, native shared object, required symbols, compiled CUDA targets, JIT toolchain/cache와 current device가 실제로 준비됐는지 묻는다. Import success는 이 장의 첫 칸일 뿐이다. Lazy extension load, first-use compile와 driver JIT는 첫 request에서 실패할 수 있다. Wheel version 문자열 대신 extension hash와 embedded architecture manifest를 남긴다.

셋째 장은 numerical contract다. 선택된 code가 input tensor의 dtype·shape뿐 아니라 scale direction, group axis, zero bias, packed order, page layout, mask와 output accumulator를 올바르게 해석하는지 묻는다. Kernel이 launch되고 finite output을 냈다는 사실은 이 장을 통과했다는 증거가 아니다. Independent reference의 first affected coordinate와 비교해야 한다.

세 장은 다음처럼 서로 다른 판정을 만든다.

| 상태 | source capability | local install | numerical contract | 조치 |
|---|---:|---:|---:|---|
| A | false | true | 미평가 | 다른 supported method 또는 명시적 거절 |
| B | true | false | 미평가 | matching artifact/build를 설치하고 재검증 |
| C | true | true | false | adapter·layout·consumer bug를 수정하고 traffic 차단 |
| D | true | true | true | 성능과 SLO를 비교할 수 있는 후보 |
| E | unknown | true | unknown | source/build identity를 고정할 때까지 승인 보류 |

상태 A를 “FlashInfer가 설치되지 않았다”고 보고하면 잘못된 remediation이 나온다. 상태 B를 “head dimension이 unsupported”라고 보고하면 model shape를 불필요하게 바꿀 수 있다. 상태 C를 “quantization 오차”라고 tolerance로 덮으면 correctness incident가 지속된다. 따라서 metric code도 `UNSUPPORTED_*`, `UNMATERIALIZED_*`, `CONTRACT_MISMATCH_*` namespace를 분리한다.

### 53.11.3 fallback 성공 뒤 수치와 지연이 함께 바뀌었다

사고 환경은 H100 replica와 4-bit GPTQ model이었다. Operator는 Marlin을 선호했고 startup manifest도 대부분 layer에서 Marlin을 effective method로 기록했다. Standard decode canary는 정상이고 latency도 기준 안이었다. 긴 prompt 뒤 speculative verify가 만든 M 구간에서만 runtime guard가 primary path를 거절했다. Request는 실패하지 않았고 generic fallback이 응답을 반환했다.

첫 증상은 error rate가 아니었다. Verify cohort의 ITL tail이 늘고 일부 prompt의 첫 다른 token 비율이 상승했다. Service metric에는 `backend=marlin`이라는 startup label만 있어 fallback coverage가 보이지 않았다. Generic operator도 CUDA에서 정상 완료됐으므로 exception과 NaN counter는 0이었다. “서버가 성공했다”는 관측이 수치 계약과 SLO 변화를 모두 숨겼다.

팀은 latency부터 최적화하지 않았다. 같은 prompt, seed, tokenizer와 sampling을 고정하고 quantized projection의 basis fixture를 실행했다. Checkpoint-native unpack과 Marlin conversion inverse는 reference와 같았다. Standard M에서는 primary output도 같았다. Guard를 넘는 M에서 generic fallback의 first dequantized weight coordinate가 달랐다. Fallback은 checkpoint-native packing을 기대했지만 parameter는 이미 Marlin-repacked generation이었다.

왜 crash하지 않았을까. 두 representation의 storage dtype과 outer allocation size가 같았고 fallback wrapper는 dtype, numel와 alignment만 검사했다. Bit shift와 tile permutation은 semantic metadata였지만 argument ABI에 없었다. Generic kernel은 유효한 주소에서 유효한 bytes를 읽고 다른 logical weight를 계산했다. 결과는 finite였고 output shape도 맞았다.

Latency가 함께 나빠진 이유도 분리했다. Primary reject 뒤 fallback wrapper가 input을 contiguous copy하고 larger workspace를 할당했다. Repacked weight를 native라고 오인했으므로 correctness는 이미 깨졌지만, copy·allocation과 더 많은 kernel launch가 ITL tail까지 늘렸다. Wrong answer와 latency regression은 같은 fallback transition에서 나왔지만 원인은 각각 representation mismatch와 execution overhead였다.

first-divergence ledger는 다음과 같았다.

| 경계 | primary M | fallback M | 판정 |
|---|---|---|---|
| raw/normalized config | 동일 | 동일 | 원인 아님 |
| static selector | Marlin accepted | Marlin accepted | 원인 아님 |
| post-load owner | Marlin R2 | Marlin R2 | 동일 |
| runtime M guard | primary | generic fallback | 최초 control divergence |
| fallback expected input | 사용 안 함 | native R0 | ABI mismatch |
| first decoded code | reference와 동일 | tile 주기로 불일치 | 최초 numerical divergence |
| output/logits | tolerance 안 | structured mismatch | 증상 |
| latency | baseline | copy·workspace·launch 증가 | SLO 증상 |

이 표는 runtime guard가 잘못됐다는 뜻은 아니다. M domain 밖에서 primary를 거절하는 것은 올바를 수 있다. 잘못은 R2 owner에서 R0 consumer로 conversion edge 없이 이동한 fallback graph였다. Guard threshold를 무조건 넓혀 unsupported primary를 실행하는 것도 수정이 아니다.

### 53.11.4 선택 가능하지만 설치되지 않은 경우와 비교한다

같은 날 다른 replica에서는 source capability상 FlashInfer attention을 지원했지만 optional package의 native extension build가 맞지 않았다. Python namespace import는 성공했으나 first native operator lookup이 실패했다. Selector는 fallback attention을 골랐고 numerical reference는 일치했다. 다만 decode latency tail이 달라졌다. 앞 사건과 증상 일부는 같지만 numerical contract 판정은 달랐다.

두 사건을 비교하면 remediation이 선명해진다. Installation 사건은 matching build를 배포하고 extension/materialization fixture를 통과시키면 된다. Representation 사건은 package를 재설치해도 해결되지 않는다. Fallback compatibility와 parameter generation을 고쳐야 한다. 두 사건 모두 effective backend와 dispatch coverage metric이 필요하지만 correctness severity는 다르다.

Install fixture는 import, shared object load, symbol lookup, minimal supported call과 actual device kernel launch를 단계별로 둔다. Minimal call은 production model을 실행할 필요가 없지만 current ABI의 dtype, shape와 workspace를 만족해야 한다. Compile/JIT가 필요하면 cold/warm 상태와 artifact digest를 남긴다. 단순 import test를 readiness probe로 사용하지 않는다.

Capability fixture는 source predicate 경계를 본다. Head dimension, cache dtype와 SM을 한 축씩 바꿔 accepted/rejected card를 만든다. Installation fixture와 섞지 않으려면 동일 build/device에서 shape만 바꾼다. 반대로 build 비교에서는 input shape를 고정한다. 독립 변수를 지키는 것이 세 상태를 구별하는 가장 단순한 방법이다.

### 53.11.5 수정은 compatible edge와 runtime guard를 함께 바꾼다

가능한 수정은 세 가지였다. 첫째 R2를 직접 소비하는 fallback을 선택한다. 둘째 native R0 copy를 보존하고 fallback에 그것을 명시적으로 전달한다. 셋째 guard가 탈락할 때 검증된 R2→R0 inverse conversion을 수행한다. 첫째는 구현이 있으면 가장 직접적이고, 둘째는 steady memory를 늘리며, 셋째는 latency와 temporary memory를 늘린다. 어느 선택도 공짜가 아니다.

이번 수정은 supported M 범위를 가진 R2-compatible operator로 fallback edge를 제한했다. 그런 operator가 없는 shape는 request가 output buffer를 건드리기 전에 명시적으로 거절했다. Method owner가 runtime guard와 fallback table을 함께 제공하게 해 외부 wrapper가 raw parameter pointer를 임의 consumer에 넘기지 못하게 했다.

Validation descriptor에는 representation id/version, qweight pack, scale/ZP ordering, g_idx state, logical/local K/N과 generation을 넣었다. Hot path에서 긴 문자열을 비교하지 않고 construction 때 compatible dispatch table을 만들었다. Runtime에는 M bucket, graph/eager와 compact generation assertion만 남겼다. 성능을 위해 semantic validation을 삭제한 것이 아니라 안전한 시점으로 옮겼다.

Fallback이 쓰는 output buffer도 새로 검토했다. Primary가 일부 preprocessing 또는 launch를 수행한 뒤 실패할 수 있으므로 fallback이 output을 overwrite하는지 accumulate하는지 확인했다. Workspace와 temporary scale은 failure 뒤 초기화/해제하고, asynchronous launch error가 later sync에서 발견되면 해당 request와 generation에 귀속했다. 실패한 primary graph를 다음 request가 재사용하지 않게 했다.

### 53.11.6 generation-safe rollback과 재승인

배포 rollback은 CLI 값을 이전 backend로 되돌리는 작업이 아니었다. 새 replica에는 R2 converted parameters, selector cache, graph executable, autotune result와 workspace pool이 있었다. 이전 method가 R0를 기대한다면 config만 되돌린 mixed state는 더 위험하다. Model generation 전체를 drain하고 canonical checkpoint에서 이전 representation으로 다시 load했다.

Rollback manifest는 binary/package digests, normalized config, layer별 effective method, representation generation, graph generation과 selector/autotune cache generation을 포함했다. 모든 TP rank가 같은 compatible method generation에 합의한 뒤 readiness를 열었다. Timeout되거나 manifest를 제출하지 못한 rank는 success가 아니라 unknown으로 처리했다.

Traffic은 verify cohort와 standard decode cohort를 분리해 canary했다. Exact quant code/index와 first dequant coordinate, layer output tolerance, logits/token agreement를 확인했다. 다음으로 ITL/TTFT, copy bytes, workspace allocation, launch count와 fallback reason을 확인했다. Correctness와 SLO가 모두 이전 terminal로 돌아온 뒤에만 incident를 닫았다.

회귀 matrix에는 runtime guard 경계의 `M-1`, `M`, `M+1`, TP local N/K 경계, graph hit/miss, speculative verify on/off를 넣었다. Primary와 fallback을 모두 실제 converted representation으로 실행했다. Mock으로 primary를 바로 실패시키면 post-load owner state를 재현하지 못하므로 사용하지 않았다.

새 release 재승인에서는 raw option이 같은지보다 effective manifest diff를 본다. Candidate priority나 threshold가 바뀌어 auto mode가 새 backend를 선택할 수 있다. Old/new source에서 selector predicate, conversion version, fallback edge와 graph key를 비교하고 golden fixture를 재실행한다. “fallback reliability improved” 같은 release note를 numerical compatibility 증거로 쓰지 않는다.

### 53.11.7 독자가 현장에서 쓰는 20분 조사 순서

조사 worksheet는 다음 열을 고정한다. `request generation`, `model generation`, `raw source`, `raw value`, `normalized value`, `override owner`, `candidate`, `installation state`, `capability predicate`, `reject code`, `effective method`, `representation owner`, `runtime guard`, `dispatch op`, `kernel family`, `output checkpoint`, `latency contribution`이다. 빈 칸은 정상으로 간주하지 않고 unknown으로 둔다.

Candidate별 판정 카드는 반복 가능해야 한다. Package 이름, Python version, extension digest, CUDA build, embedded SM, current device, dtype, logical/local shape, layout version, enabled features와 phase를 기록한다. Accepted 카드도 predicate가 무엇을 보지 않았는지 적는다. Shape만 검사한 accept를 numerical compatibility 승인으로 확대하지 않는다.

Runtime guard 카드는 static selection과 별개다. Actual M, q_len, KV length, graph bucket, workspace availability, stream/capture state와 parameter generation을 적는다. Branch result가 primary, compatible fallback, converted fallback, eager 같은 어느 edge인지 표시한다. Edge가 representation descriptor를 검사하지 않으면 validation gap이다.

수치 카드는 code/index exact equality, dequant tolerance, layer output tolerance와 logits/token policy를 분리한다. Code, ZP, g_idx와 permutation에는 floating tolerance를 적용하지 않는다. FP16/BF16 accumulator output에는 사전 정의한 tolerance를 쓴다. Tolerance 변경은 backend 변경과 별도 review를 받는다.

성능 카드는 selector/plan, conversion, metadata preparation, copy, workspace, launch gap, kernel과 synchronization을 나눈다. Cold JIT/autotune과 warm request를 분리한다. Fallback cohort의 denominator는 전체 request가 아니라 해당 runtime guard domain에 들어온 operation이다. 평균만 보고 rare tail을 지우지 않는다.

복구 카드는 차단, 수정, canary, rollout과 rollback terminal을 가진다. 차단은 incompatible edge를 금지한다. 수정은 owner와 consumer invariant를 말한다. Canary는 boundary fixture와 production-like cohort를 함께 쓴다. Rollback terminal은 모든 rank, cache와 graph generation이 이전 compatible state로 수렴했음을 증명한다.

첫 3분에는 요청 cohort와 phase를 고정한다. Prefill, decode, speculative verify, graph hit/miss 가운데 어디서 latency나 output이 갈리는지 본다. Model-wide 평균으로 시작하지 않는다. Raw config, normalized final config와 effective manifest를 같은 generation에서 가져온다.

다음 4분에는 candidate card를 읽는다. 기대한 backend가 registry에 있었는지, package/build·SM·dtype·shape·layout·feature 중 어느 predicate에서 탈락했는지 찾는다. Import를 다시 실행하는 것만으로 이 단계를 대신하지 않는다. Reject reason이 없다면 source predicate input을 작은 표로 복원한다.

다음 4분에는 representation owner를 찾는다. Load 후 conversion이 있었는지, qweight와 auxiliary tensor가 어느 generation인지, fallback consumer가 기대하는 representation이 무엇인지 확인한다. Dtype와 shape가 같다는 이유로 compatible이라고 표시하지 않는다. 한 logical coordinate를 native, converted, consumer decode에서 비교한다.

다음 4분에는 actual dispatch를 확인한다. Wrapper class 이름이 아니라 operator와 kernel family, preprocessing copy, workspace와 sync를 timeline에 놓는다. Graph replay면 captured manifest와 runtime admission을 비교한다. Kernel symbol 부재를 곧바로 CPU fallback이라고 결론내리지 않는다.

마지막 5분에는 correctness와 SLO를 따로 판정한다. Reference coordinate와 layer output이 다르면 traffic을 차단하고 first divergence를 고정한다. 같다면 fallback latency를 copy·workspace·launch·kernel로 분해한다. 수정 후보마다 representation edge, memory와 latency 비용, rollback generation을 적는다. 증상만 보고 backend를 강제하거나 tolerance를 넓히지 않는다.

### 53.11.8 소스 노트

소스 좌표를 읽을 때는 link가 가리키는 함수의 역할을 과장하지 않는다. vLLM attention selector는 candidate와 backend class를 결정하는 증거지만, 그 request가 어느 native symbol을 launch했는지는 runtime wrapper와 profiler가 증명한다. Marlin shape helper는 static local K/N/group gate를 증명하지만 dynamic M fallback과 input representation은 apply path에서 확인한다. SGLang override는 effective argument mutation을 증명하지만 runner가 materialize됐는지는 method construction과 call trace가 필요하다.

#### selector에서 launcher까지 source walk를 끊지 않는 법

첫 고정점은 option consumer다. CLI나 config model이 값을 받아들이는 line, default/alias가 normalized state를 만드는 line, override가 value를 바꾸는 line을 분리한다. Parser가 field를 보존하는 것만으로 실행 효과가 있다고 쓰지 않는다. Field가 candidate order, predicate 또는 parameter constructor의 argument가 되는 다음 consumer를 찾는다.

둘째 고정점은 selector predicate다. Predicate 함수 전체를 “지원 검사”라고만 인용하지 않고 observed input과 branch result를 적는다. 예를 들어 local K/N divisibility, head size membership, KV dtype equality, SM lower bound, package availability가 어느 순서로 short-circuit되는지 본다. 첫 false만 log한다면 뒤 predicate는 false가 아니라 미평가다.

셋째 고정점은 method construction과 state mutation이다. Selected class가 parameter를 새로 만들거나 post-load conversion을 등록하는지 확인한다. qweight assignment, scale permutation, ZP conversion, g_idx sort와 workspace creation이 각각 언제 일어나는지 적는다. Attention backend라면 metadata builder, KV cache layout와 allocator/writer consumer를 잇는다. Selection이 이름만 바꾸는지 data contract까지 바꾸는지 여기서 갈린다.

넷째 고정점은 runtime guard다. Static selector가 승인한 domain 안에서도 actual M, q_len, sequence/page length, graph bucket, workspace와 feature state가 branch를 바꿀 수 있다. Guard의 true/false branch가 어떤 operator를 호출하고 어떤 representation을 기대하는지 나란히 둔다. Fallback branch가 멀리 떨어진 helper에 있더라도 current parameter owner를 따라간다.

다섯째 고정점은 native binding과 launcher다. Python custom op, C++ registration, dispatch macro, architecture/dtype specialization과 CUDA launch argument로 내려간다. Wrapper 함수 이름과 profiler symbol이 다를 수 있으므로 registration과 generated kernel naming을 확인한다. M/N/K, stride, scale pointer, workspace, stream과 output pointer가 source 의미와 call trace에서 합의하는지 본다.

여섯째 고정점은 completion이다. CUDA launch는 비동기이므로 launch return과 request-visible completion이 다르다. Error check, event/stream dependency와 output consumer를 찾는다. Primary가 launch 뒤 later error를 내고 fallback한다면 partial output과 workspace를 cleanup하는지 확인한다. “fallback succeeded”는 이전 work가 결과에 섞이지 않았다는 것까지 포함해야 한다.

이 여섯 고정점을 표로 만들면 source 이동에도 견딜 수 있다. Upgrade로 함수 이름과 line이 바뀌어도 option consumer, predicate, mutation, guard, launcher와 completion이라는 semantic role을 다시 찾는다. Diff에서는 값의 이름보다 predicate domain, representation transition과 fallback edge가 바뀌었는지를 우선한다.

#### incident terminal을 수치와 SLO 두 축으로 닫는다

수치 terminal의 첫 조건은 canonical tuple이다. Checkpoint-native q/scale/ZP/order 또는 attention Q/K/V·mask·cache coordinate가 independent reference와 맞아야 한다. 둘째는 conversion checkpoint다. Optimized representation의 inverse sample 또는 direct logical decoder가 같은 tuple을 내야 한다. 셋째는 primary와 실제 fallback output이 각자의 supported 경계에서 tolerance를 통과해야 한다.

넷째는 경계 반증이다. Runtime guard threshold의 바로 아래·정확한 값·바로 위, TP local tile/group 경계, graph bucket hit/miss와 phase split을 넣는다. Primary가 통과하는 중앙값만 검증하지 않는다. 이전 사고를 만든 rare cohort가 회귀 fixture의 첫 행이 되어야 한다.

SLO terminal은 kernel duration만 보지 않는다. Selector/plan, preprocessing copy, workspace allocation, launch gap, kernel, synchronization과 cleanup 시간을 phase별로 합친다. TTFT, ITL과 goodput 가운데 어떤 service objective가 바뀌었는지 적는다. Fallback coverage의 denominator는 해당 guard를 만난 operation이다.

Correctness가 복구됐지만 fallback이 느리면 두 선택지가 있다. Supported primary domain을 근거 있게 넓히거나, compatible fallback의 copy/workspace를 줄인다. 느리다는 이유로 incompatible consumer를 다시 허용하지 않는다. 반대로 latency가 돌아왔어도 output reference가 unknown이면 incident는 닫히지 않는다.

Generation terminal은 모든 rank와 replica가 하나의 manifest family에 수렴하는 상태다. Binary/build, normalized config, layer method, parameter representation, selector cache, graph와 autotune generation을 비교한다. Expected rank-local shape 차이를 제외한 method ABI 불일치는 request admission 전에 실패시킨다.

Rollback terminal은 old traffic이 끝날 때까지 old parameters/graphs가 살아 있고 new traffic만 새 generation을 보는 상태다. Address allocator가 같은 pointer 값을 재사용해도 generation이 다르면 graph를 공유하지 않는다. Cancellation된 conversion worker가 뒤늦게 parameter나 cache를 publish하지 않도록 join과 completion marker를 확인한다.

관측 terminal은 requested/effective mismatch, reject reason, dispatch coverage와 fallback edge가 dashboard에서 재현 가능한 상태다. Exact shapes와 layer names는 sampled trace에 두고 metric cardinality를 제한한다. Source claim, installed artifact claim과 execution claim을 각각 commit line, build manifest와 trace로 증명한다.

마지막 승인 문장은 구체적이어야 한다. “Raw auto는 normalized auto로 유지됐고 candidate A는 local N predicate에서 탈락했다. Candidate B는 native R0을 소비하며 reference와 일치했다. Runtime M 경계의 R2 fallback은 제거됐고 R2-compatible C만 허용된다. 모든 rank가 generation 17에 합의했으며 verify cohort ITL과 output이 rollback terminal 안이다.” 이 문장을 쓸 수 있어야 선택과 복구가 닫힌다.

#### 옵션 하나를 바꿀 때 반드시 다시 묻는 질문

첫째, 이 값은 preference인가 strict requirement인가. Auto priority를 바꾸는 값과 unsupported 상태를 금지하는 값은 실패 정책이 다르다. 둘째, normalization이 alias만 바꾸는가, default를 채우는가, model/hardware 사실로 override하는가. Raw와 effective diff에 owner와 reason을 붙인다.

셋째, selector가 process 시작 때 한 번 실행되는가, layer마다 실행되는가, request shape에서 다시 실행되는가. Cache가 있다면 key와 invalidation generation을 찾는다. 넷째, 선택이 tensor, cache layout, workspace나 graph를 mutate하는가. 그렇다면 option hot change는 단순 enum 교체가 아니다.

다섯째, capability false와 installation false를 구별했는가. Source에 지원 branch가 없으면 재설치로 해결되지 않고, branch는 있지만 extension이 없으면 shape 변경으로 해결되지 않는다. 여섯째, selected consumer의 numerical contract를 확인했는가. Dtype·shape·launch success는 scale, packing, mask와 cache address의 의미를 보장하지 않는다.

일곱째, runtime guard가 primary를 거절할 수 있는 domain은 무엇인가. M, q_len, page tail, graph bucket, workspace, adapter, speculative verify와 expert token 수를 본다. 여덟째, 각 guard branch는 현재 representation에서 검증된 edge인가. Conversion 뒤 native fallback을 호출하지 않는지 확인한다.

아홉째, 실제 실행은 무엇으로 관측되는가. Effective class log, custom operator trace, CUDA symbol과 completion을 연결한다. 하나만으로 나머지를 추정하지 않는다. 열째, 성능 변화의 denominator는 무엇인가. 전체 request 평균 대신 phase·shape·fallback cohort와 scheduled operation을 사용한다.

열한째, 수정 뒤 경계 양쪽이 검증됐는가. Accepted와 rejected fixture, primary와 fallback reference, graph hit/miss와 TP rank를 함께 둔다. 열두째, rollback이 config 외 converted parameters, selector/autotune cache, workspace와 graph generation까지 되돌리는가. 모든 rank가 terminal을 보고한 뒤 admission을 연다.

이 열두 질문은 옵션 문서의 부록이 아니라 코드 리뷰 순서다. 답이 source coordinate, manifest 또는 trace로 고정되지 않으면 “그럴 것”이라는 가정으로 표시한다. Unknown을 success로 바꾸지 않는 것만으로도 fallback이 조용히 수치 계약을 바꾸는 사고 상당수를 launch 전에 멈출 수 있다.

Review 결과는 layer family별로 남긴다. Attention 한 layer의 acceptance가 MLA, sliding-window, encoder attention과 KV-quantized layer를 대표하지 않는다. Dense quant projection의 Marlin 결과가 fused MoE, stacked QKV와 LM head를 대표하지 않는다. 같은 option이 각 family에서 다른 method와 representation을 만들 수 있기 때문이다.

또한 positive evidence와 negative evidence를 균형 있게 기록한다. Kernel symbol을 관측한 것은 launch의 positive evidence다. Expected symbol이 없다는 사실은 request가 그 branch를 통과했다는 denominator가 있을 때만 negative evidence가 된다. Fallback counter 0도 layer가 처음부터 다른 method로 만들어졌다면 primary coverage를 증명하지 않는다.

성공한 fallback에는 expiry를 둔다. 현재 build·SM·shape·representation에서 검증됐다는 의미이지 다음 release에도 영구 호환된다는 뜻이 아니다. Selector predicate, conversion version, runner ABI나 graph key가 바뀌면 fixture와 edge validation을 다시 실행한다. Cache manifest가 previous validation result를 재사용한다면 이 revision tuple을 key에 포함한다.

마지막으로 운영자는 option 변경 전후의 effective manifest를 보관한다. 장애가 발생했을 때 raw configuration diff만 보면 auto priority와 model override, dependency availability 변화를 놓친다. Layer·phase별 chosen method, reject reason, representation과 dispatched family의 diff가 있어야 “설정은 그대로인데 왜 kernel이 바뀌었는가”에 답할 수 있다.

이 diff는 성능 회귀뿐 아니라 재현성의 기준이다. 동일 revision과 device에서 결과가 다르면 environment override, selector cache, lazy JIT artifact, visible device 또는 stale graph처럼 아직 manifest에 들어오지 않은 숨은 입력을 찾는다. 입력을 발견하면 다음 비교부터 정식 key로 승격한다.

- [vLLM v0.27.1 — attention backend selector](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/attention/selector.py#L101-L207)
- [vLLM v0.27.1 — Marlin shape support](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L172-L230)
- [SGLang v0.5.18 — backend overrides](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/arg_groups/overrides.py#L1270-L1305)
- [SGLang v0.5.18 — FP4 runner backend](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/quantization/fp4_utils.py#L124-L166)
- [llama.cpp v0.2.0 — CUDA backend support boundary](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/ggml-cuda.cu#L3267-L3310)

## 53.12 종합 회고: 실제 kernel까지 닫힌 선택만 믿는다

이 장의 출발점은 간단한 오해였다. Package import가 성공하고 config에 backend 이름을 썼으니 그 kernel이 실행될 것이라는 믿음이다. 실제 serving stack은 그렇게 짧지 않다. Raw request는 normalize되고 model과 platform override를 거친다. Registry의 후보는 package, build, SM, dtype과 quant format, shape, layout, phase와 feature 조합을 심사받는다. 선택된 method는 parameter representation을 바꿀 수 있고 runtime guard는 실제 batch에서 다시 후보를 가른다.

따라서 backend를 확인하는 최소 단위는 이름 하나가 아니라 generation chain이다. Requested, normalized, registered candidate, accepted/effective, representation owner, dispatched operator, completed kernel과 reference output이다. 앞 generation과 뒤 generation을 한 log field에 섞으면 operator는 어느 단계가 달라졌는지 알 수 없다.

새 engine을 조사한다면 config 선언을 읽고 곧바로 kernel directory로 내려가지 않는다. 먼저 selector consumer와 precedence를 찾는다. Candidate order와 reject predicate를 표로 펼치고 predicate가 실제 local tensor와 current device를 읽는지 확인한다. 그 다음 method construction과 loader conversion을 따라 representation owner를 정한다. 마지막으로 forward runtime guard, fallback branch와 native launcher를 잇는다.

Wrong answer에서는 더욱 이 순서가 중요하다. Primary kernel이 실행됐다는 사실도, fallback이 crash하지 않았다는 사실도 representation compatibility를 증명하지 않는다. Native coordinate, conversion output과 consumer decode를 차례로 비교해 first divergence를 찾는다. Dtype와 outer shape가 같은 silent reinterpretation은 이런 coordinate fixture가 없으면 품질 문제로만 남는다.

성능 문제도 같은 ledger를 쓴다. Candidate reject 때문에 slower backend가 선택됐는지, graph miss가 eager branch를 열었는지, conversion과 workspace가 cold latency를 만들었는지, CPU/device copy가 fallback 시간을 지배했는지 분리한다. Kernel duration만 빠른 path가 end-to-end로 빠르다는 보장은 없다.

좋은 fallback은 기능이 아니라 증명된 edge다. Current representation을 직접 읽는 consumer, 명시적 conversion 뒤의 consumer, 원 native copy를 소유한 consumer 가운데 하나여야 한다. Edge가 없으면 reject가 correct behavior다. Error를 피하기 위해 incompatible pointer를 다른 kernel에 넘기는 것은 resilience가 아니라 silent corruption이다.

마지막 판정은 이렇게 쓸 수 있어야 한다. “Requested FlashInfer candidate는 import와 registration을 통과했지만 decode KV dtype predicate에서 탈락했다. Effective backend B는 unchanged cache representation을 지원했고 actual symbol과 output reference가 일치했다.” 또는 “Marlin R2 conversion 뒤 dynamic guard가 R0 fallback을 호출했다. First decoded code부터 달랐으며 R2-compatible fallback으로 제한한 뒤 TP와 M 경계 fixture가 일치했다.”

이 정도로 구체적이면 config를 바꿀지, package를 교체할지, shape를 조정할지, conversion/fallback code를 고칠지 결정할 수 있다. Backend 선택은 더 이상 숨은 heuristic이 아니다. 입력 사실, predicate, owner transition과 dispatch로 설명되는 실행 계약이다. 그 계약이 실제 kernel과 reference output까지 닫힐 때만 “이 backend를 사용한다”는 문장이 기술적으로 참이 된다.

완료 조건은 startup 성공이 아니다. 모든 rank가 같은 compatible method generation을 보고, layer별 parameter와 auxiliary tensor가 그 method의 representation owner에게 완전히 commit됐으며, graph와 eager runtime guard가 현재 representation에서 안전한 후보만 선택하고, actual launch와 reference output이 합의되어야 한다. Failure가 있었다면 provisional conversion, workspace와 cached selection도 rollback되어야 한다.

Rollback을 검증할 때는 config enum만 이전 값으로 돌리지 않는다. Post-load repack을 수행했다면 old method에 필요한 native copy가 남아 있는지 확인한다. 없다면 model parameter를 다시 load하거나 inverse conversion을 검증해야 한다. Graph executable, selector cache와 autotune result도 old/new representation generation을 구분하도록 invalidate한다. 일부 layer만 되돌아간 mixed state는 전체 model reject 대상으로 삼는다.

Distributed rollback은 rank별 manifest를 다시 비교한다. Rank 0이 primary, rank 1이 fallback을 계속 보유한 채 collective를 시작하지 않게 한다. 모든 rank가 previous generation 또는 new generation 중 하나로 수렴한 뒤 request admission을 연다. Timeout된 worker는 unknown으로 남기고 성공으로 간주하지 않는다.

운영 중에는 fallback 횟수보다 unexpected transition을 우선 alert한다. Auto policy가 의도한 phase split은 정상일 수 있지만 forced/effective mismatch, representation-compatible edge가 없는 fallback, rank별 method disagreement와 unknown dispatch symbol은 즉시 조사한다. Latency regression은 correctness invariant가 유지된다는 확인 뒤 tuning queue로 보낸다.

새 backend를 도입할 때도 같은 절차를 거친다. Registration test, supported/unsupported predicate boundary, conversion identity, primary와 fallback reference, TP/phase/graph composition, failure rollback과 metrics를 함께 추가한다. Fast path benchmark 하나만 넣고 fallback contract를 비워 두면 production의 rare shape가 가장 위험한 경로를 처음 실행하게 된다.

결국 selector는 단순한 factory가 아니다. Model artifact, hardware, runtime workload와 installed binary를 한 execution method로 묶는 policy engine이다. Policy의 입력과 탈락 이유, state mutation과 소비자가 관찰 가능해야 upgrade와 장애에서 같은 결론을 재현할 수 있다. 이 투명성이 확보되면 “왜 이 kernel인가”와 “왜 fallback했는가”에 code, trace와 수치로 답할 수 있다.

최종 검증에서는 지원 경계의 양쪽을 반드시 실행 가능한 fixture로 남긴다. Predicate가 `head_dim=128`을 받고 96을 거절한다면 두 case 모두 expected candidate card, effective method와 output reference를 가진다. TP-local tile, group size, graph bucket과 runtime M도 경계값 바로 아래·정확한 값·바로 위를 둔다. 그래야 predicate refactoring이 조용히 support domain을 바꾸는 일을 찾을 수 있다.

Fallback을 검증하는 fixture는 primary를 단순 mock failure로 건너뛰지 않는다. 실제 converted representation을 만들고 runtime guard를 의도적으로 탈락시켜 fallback consumer에 전달한다. Consumer가 현재 owner와 representation을 확인하고 reference output을 만드는지, incompatible하면 output buffer를 건드리기 전에 reject하는지 검사한다. 실패 뒤 workspace, graph와 cached selection이 다음 요청에 남지 않는지도 본다.

이 두 종류의 fixture가 함께 있어야 선택의 앞과 뒤가 닫힌다. Boundary fixture는 왜 candidate가 선택됐는지를, representation fixture는 선택이 바뀐 뒤에도 왜 안전한지를 증명한다. Actual kernel symbol과 numerical checkpoint를 더하면 config intent에서 GPU completion까지의 설명이 완성된다.

운영자는 이 설명을 재현할 수 있어야 한다. 같은 revision, build manifest, device capability, normalized config와 fixture를 사용한 다른 사람이 같은 candidate 탈락 이유, 같은 representation generation과 같은 dispatch를 관찰해야 한다. 결과가 다르면 selector cache, environment override, lazy JIT artifact 또는 device visibility 가운데 숨은 입력이 있다는 뜻이다. 그것을 manifest에 추가한 뒤 다시 비교한다.

재현성은 성능 숫자를 고정한다는 뜻이 아니다. 부하와 clock에 따라 시간은 달라질 수 있다. 대신 선택의 인과와 correctness domain을 고정한다. 어떤 입력이 어느 predicate를 통과했고 어떤 byte layout을 어느 consumer가 읽었는지 같다면 성능 차이도 profiler로 정직하게 분석할 수 있다.
## 53.13 참고: capability 조건 카탈로그

Backend support를 묻는 질문은 하나의 `is_available()`로 끝나지 않는다. Package/build, hardware, dtype/format, shape, layout/state, feature composition 여섯 축으로 나눈다. Predicate가 한 함수에 섞여 있어도 조사 표에서는 분리한다.

**Package와 build.**

Python package version, shared object soname, compiled CUDA major, embedded SM targets, JIT compiler와 cache 상태를 본다. Pure Python wrapper가 import돼도 native extension이 lazy load될 수 있다. 첫 kernel request에서만 symbol error가 날 수 있다.

Prebuilt cubin이 있으면 current SM을 포함하는지, PTX fallback이 있으면 driver JIT가 가능한지 확인한다. Source JIT는 toolkit과 compiler flags, architecture target, writable cache와 process permission에 의존한다. Wheel filename만 보고 runtime compatibility를 단정하지 않는다.

**Hardware.**

Compute capability는 tensor core instruction과 dtype 지원, shared memory와 architecture-specific kernel availability를 제한한다. “Hopper”나 “Blackwell”이라는 제품군 이름보다 runtime capability 값과 code predicate를 기록한다. MIG 또는 device visibility 때문에 selector가 예상과 다른 device를 조회하는지도 본다.

Multi-GPU에서는 rank마다 GPU가 다를 수 있다. 모든 rank가 같은 candidate를 지원해야 collective graph가 일관될 수 있다. Rank 0 capability만 보고 global selection을 broadcast하면 heterogeneous cluster에서 다른 rank가 실행하지 못할 수 있다.

**Dtype과 quant format.**

Activation dtype, weight storage dtype, scale dtype, KV cache dtype와 output dtype을 분리한다. FP8이라는 이름도 E4M3/E5M2, direct/inverse scale, per-tensor/channel/block granularity가 다르다. `uint8` container가 같다고 같은 format이 아니다.

Quant config가 method를 GPTQ라고 분류해도 bits, group size, zero point, act order와 packing version이 backend capability를 바꾼다. Candidate가 config name만 확인하고 tensor representation을 검증하지 않으면 silent reinterpretation이 가능하다.

**Shape.**

Attention은 batch tokens, query length, head count, KV head count, head dimension, page size, sequence length와 mask shape를 본다. Quant GEMM은 M/N/K, group size, tile divisibility와 expert/token counts를 본다. 중요한 것은 TP/EP 뒤 local shape다.

Global N=4096이 Marlin tile에 맞더라도 TP=3이면 rank-local partition과 padding rule이 달라질 수 있다. Uneven shard나 packed group boundary가 허용되는지 확인한다. Model config만 보고 support를 선언하지 않고 actual parameter shape와 runtime M을 predicate에 넣는다.

**Layout와 state.**

Contiguity, strides, packed order, interleave, page table와 block layout, alignment, device와 stream을 본다. Shape와 dtype가 같아도 scale permutation이 다르면 다른 ABI다. CUDA graph capture 중 pointer stability와 workspace ownership도 state predicate다.

Representation owner를 이름으로 남긴다. `checkpoint_native`, `marlin_repacked`, `nvfp4_interleaved`, `gguf_block_native`, `dense_dequantized`처럼 의미를 구분한다. 실제 code에 enum이 없어도 owning parameter/method class와 conversion lifecycle로 복원한다.

**Feature composition.**

Sliding window, ALiBi/RoPE variant, speculative decode, LoRA, multimodal mask, prefix cache, CUDA graph, chunked prefill와 MoE routing이 backend 지원을 바꿀 수 있다. 각각 단독 지원된다는 사실은 조합도 지원된다는 뜻이 아니다.

Predicate 표에는 accepted뿐 아니라 어느 feature 조합에서 탈락했는지 남긴다. Auto fallback이 feature를 조용히 비활성화해 backend를 유지하는지, feature를 유지하고 backend를 바꾸는지도 구분한다.

## 53.14 참고: fallback 조건별 first-divergence 카탈로그

**사건 1 — import success지만 runtime unsupported.** Module과 extension은 load됐지만 head dimension predicate가 false다. Package를 재설치하기 전에 candidate reject log를 본다. Supported shape fixture와 한 축만 다른 fixture로 predicate를 고정한다.

**사건 2 — silent reinterpretation.** FP8 scale byte layout을 다른 runner가 읽는다. Kernel은 완료되지만 first dequant coordinate부터 reference와 다르다. Representation owner와 scale interleave conversion을 수정한다.

**사건 3 — TP-local shape fallback.** TP=1에서 Marlin, TP=4에서 local N/tile gate가 실패한다. Global config가 아니라 rank-local parameter와 group boundary를 기록한다. Compatible generic method 또는 padding policy를 검증한다.

**사건 4 — phase split.** Prefill profile에는 FlashAttention, decode에는 FlashInfer가 보인다. 이것이 bug인지 intended selector인지 phase별 effective config와 support predicate를 대조한다. Metric을 phase별로 나눈다.

**사건 5 — post-repack fallback mismatch.** Marlin conversion 후 runtime shape가 fallback을 유발하고 generic consumer가 native layout으로 해석한다. Native/repacked sample과 fallback ABI를 비교한다. Compatible consumer가 없으면 명시적으로 fail한다.

**사건 6 — forced backend가 조용히 바뀐다.** Raw forced 값과 effective 값이 다르지만 startup은 성공한다. Override owner, reason과 strict/permissive policy를 확인하고 final config와 metric에 노출한다.

**사건 7 — graph와 eager가 갈라진다.** Captured bucket은 backend A, larger batch eager path는 B다. Representation이 공존 가능한지, workspace와 output tolerance가 맞는지 확인한다. Graph miss와 backend change를 별도 event로 남긴다.

**사건 1을 실제로 재현하는 순서.** 먼저 package import만 수행한 상태와 extension의 native symbol을 처음 호출한 상태를 분리한다. `import flashinfer`가 성공했어도 lazy loader가 `.so`를 열지 않았을 수 있다. Module path, version, loaded shared objects와 symbol resolution 결과를 남긴다. 그 다음 device capability와 candidate predicate를 같은 process에서 수집한다.

Supported control fixture와 failing fixture는 head dimension 하나만 다르게 한다. 두 run의 raw/effective config, package/build와 device facts는 같아야 한다. Candidate reject reason만 달라져야 한다. 여러 option을 동시에 바꾸면 어느 predicate가 원인인지 알 수 없다.

Selector가 rejected candidate 다음 후보를 고르면 output reference와 actual kernel을 확인한다. Forced mode라면 expected policy가 reject인지 fallback인지 test 이름에 명시한다. Startup log에 requested 이름만 남는다면 effective class를 structured log에 추가한다.

Package 문제를 고칠 때도 reinstall로 끝내지 않는다. 새 wheel의 CUDA major, SM artifact, extension hash와 JIT state를 manifest에 남기고 same fixture에서 candidate acceptance와 dispatch를 다시 확인한다. Import 성공만 재검사하면 같은 오판을 반복한다.

**사건 2의 byte-level 증거.** Silent reinterpretation fixture는 모두 zero나 random weight를 쓰지 않는다. Logical coordinate `(k,n)`마다 순차 code를 두고 scale block마다 1,2,4,8처럼 구별 가능한 값을 둔다. Native reference decoder로 몇 좌표를 손으로 복원한다.

Loader 직후 source tensor byte/sample, conversion 직후 output sample, kernel 직전 pointer의 representation owner를 비교한다. First native decode부터 틀리면 artifact/schema 문제다. Native는 맞고 conversion 뒤 틀리면 converter 문제다. 둘 다 맞고 consumer reference만 틀리면 argument/layout ABI다.

`uint8` dtype와 shape equality는 약한 validation이다. Representation tag, quant scheme/version, scale direct/inverse, block/group axis, interleave와 padding을 검사한다. Runtime assertion 비용이 크면 construction time에 method owner type과 parameter class를 묶고 unsafe cast를 없앤다.

수정 test는 primary뿐 아니라 fallback consumer도 같은 logical fixture를 읽게 한다. Conversion inverse identity만 통과해도 kernel argument가 scale을 다른 방향으로 적용할 수 있으므로 output basis test를 별도로 둔다.

**사건 3의 rank별 ledger.** TP-local fallback에서는 각 rank의 global range와 local physical shape를 나열한다. Padding이 있으면 valid range, padded range, scale/ZP rows와 collective output trimming을 적는다. Rank 0 shape만 보고 전체를 판정하지 않는다.

TP=1/2/4/8을 비교하되 global model과 artifact는 고정한다. 어느 TP에서 candidate가 처음 탈락하는지 찾는다. Predicate input이 global N을 잘못 사용해 accepted됐는데 kernel에서 local N assert가 나는 경우도 있고, 반대로 padding 가능하지만 selector가 raw local N만 보고 과도하게 reject할 수도 있다.

Uneven vocabulary 또는 MoE expert partition은 rank별 shape가 다를 수 있다. Backend를 rank별로 다르게 허용한다면 collective dtype/layout과 performance skew를 검증해야 한다. 보통 공통 compatible method가 단순하지만 current source policy를 확인한다.

Fallback이 선택되면 native/repacked ownership을 다시 본다. Initialization에서 local shape gate가 일찍 실패하면 repack 전 native를 유지할 수 있다. Runtime에 늦게 실패하면 이미 repack됐을 가능성이 있다. 같은 “shape unsupported”라도 failure 시점이 representation safety를 바꾼다.

**사건 4의 phase 표.** Prefill과 decode row에 requested backend, q_len, cache state, selected class, internal branch, workspace, kernel symbols, output dtype와 latency를 둔다. 같은 wrapper가 내부에서 다른 kernel을 호출하면 selected class는 같고 dispatched family만 다르다. 별도 selector면 effective class부터 다르다.

Chunked prefill은 third phase처럼 보일 수 있다. Chunk q_len과 prefix cache hit 상태에 따라 decode-like 또는 prefill kernel을 쓸 수 있다. Metric에서 단순 request type보다 actual phase/branch를 사용한다.

Phase split 자체는 오류가 아니다. 문제는 cache layout, mask/position semantics와 numerical state가 호환되는지다. Prefill 마지막 K/V를 sample하고 first decode reader가 같은 logical coordinate를 읽는지 확인한다. Backend 이름이 다르다는 이유만으로 output equality를 요구하거나 동일 이름이라는 이유로 호환을 가정하지 않는다.

Performance 판정은 TTFT와 ITL을 분리한다. Prefill backend 변경이 TTFT를, decode backend 변경이 ITL을 주로 건드릴 수 있다. End-to-end average만 보면 두 효과가 상쇄된다. Kernel duration과 planning/copy/workspace overhead도 분리한다.

**사건 5의 representation failure window.** Model initialization log에서 method A가 selected되고 post-load repack이 완료된다. Runtime M이 threshold 밖이라 branch B로 내려간다. 이때 B가 어떤 parameter class를 받고 어떤 decode function을 호출하는지 source를 따라간다.

Parameter name과 `.dtype`가 같아도 R0와 R2는 다르다. B가 native packing shift/mask를 사용하면 first code부터 달라진다. Shape assert가 통과한다는 사실이 오히려 silent wrong answer 가능성을 높인다.

안전한 수정은 세 가지다. R2를 직접 읽는 B2 fallback을 사용한다. R0 native copy를 보존하고 B에 명시적으로 전달한다. R2→R0 inverse conversion을 수행한 뒤 B를 호출한다. 어느 것도 비용과 memory가 공짜가 아니다. Compatible 경로가 없으면 explicit reject가 맞다.

판단 카드에는 fallback representation type을 포함한 test를 둔다. Mock method name이 아니라 실제 converted fixture를 전달하고 primary rejection을 강제한다. Reference dequant와 output을 비교하고 fallback count/reason도 assert한다.

**사건 6의 operator contract.** User가 backend X를 강제했는데 normalization override가 Y로 바꾸면 startup final config와 health status에 표시한다. Warning 한 번이 log aggregation에서 사라질 수 있으므로 requested/effective mismatch metric을 둔다.

Forced semantics가 strict라면 override 대신 configuration error를 낸다. 다만 distributed startup에서 일부 worker만 reject하지 않도록 coordinator가 capability denominator를 모은 뒤 일관되게 fail해야 한다. Error에는 unsupported predicate와 observed/required 값을 담는다.

Forced가 preference semantics라면 이름과 문서가 그것을 분명히 해야 한다. Fallback priority와 feature preservation policy를 노출한다. Backend를 유지하려고 sliding window를 끄는 것과 feature를 유지하고 backend를 바꾸는 것은 model semantics가 다르다. Silent feature disable은 단순 성능 fallback이 아니다.

Runtime dynamic guard에서 forced backend가 탈락할 수 있다면 request별 fallback을 metric에 기록한다. Startup effective X만 표시하면 operator가 실제 coverage를 알 수 없다. Workload shape distribution과 X dispatch ratio를 함께 본다.

**사건 7의 graph generation.** Capture manifest에는 shape bucket, selected method, parameter representation generation, workspace pointers와 kernel nodes를 둔다. Replay admission은 runtime inputs가 이 manifest predicate를 만족하는지 확인한다.

Graph miss로 eager가 되면 eager selector가 현재 representation을 기준으로 candidate를 고른다. Graph A와 eager B가 같은 R2를 읽는다면 안전할 수 있다. B가 R0를 기대하면 graph miss가 correctness incident를 촉발한다.

Parameter conversion 또는 backend override가 바뀌면 old graph generation을 invalidate한다. Pointer address가 allocator 우연으로 같아도 representation owner generation이 다르면 replay하지 않는다. Graph cache key에 relevant method/layout generation이 포함되는지 본다.

Test는 capture size 안과 밖을 교대하고 output을 reference와 비교한다. Graph hit/miss, backend A/B, conversion count와 memory를 기록한다. Larger batch에서 latency가 튀면 recapture, eager fallback, backend change와 workspace allocation 중 어느 단계가 원인인지 timeline으로 가른다.

이 일곱 사건을 관통하는 질문은 하나다. 최초로 다른 generation은 어디인가. Raw/normalized config, candidate predicate, selected class, representation transition, runtime guard, dispatch와 output coordinate를 순서대로 비교하면 backend 이름에 매달리지 않고 원인을 찾을 수 있다.

선택된 backend와 representation은 아직 resident byte가 아니다. 54장은 이 결정을 CPU 메모리, PCIe 전송과 HBM 배치의 실제 경로에 대입해, 어느 owner가 언제 payload를 옮기고 해제하며 kernel이 어떤 generation을 읽는지 이어서 확인한다.
