# 14장. RoPE·GQA·MLA가 cache shape를 바꾸는 법

긴 문맥에서 메모리가 갑자기 모자라거나 첫 prefill은 맞는데 decode부터 답이 갈릴 때 사람들은 흔히 “KV cache 문제”라고 말한다. 그러나 이 표현에는 서로 다른 세 질문이 뭉쳐 있다. token은 어떤 논리 위치를 갖는가. 그 위치를 반영한 K와 V는 어떤 feature shape로 저장되는가. 그 logical row는 어느 physical page와 slot에 기록되는가. 셋 중 하나만 틀려도 shape는 멀쩡한데 내용이 틀릴 수 있다.

12장에서 Q·K·V와 query head/KV head의 좌표를 만들었고, 13장에서 score·mask·softmax·PV가 그 좌표를 소비하는 법을 보았다. 이 장은 그 사이에 시간축을 넣는다. RoPE는 Q와 K가 위치 차이를 표현하도록 회전한다. GQA와 MQA는 여러 query head가 더 적은 KV head를 공유하게 해 저장 shape를 줄인다. MLA는 K/V의 비위치 성분을 저차원 latent로 압축하고 위치 성분을 별도로 다루므로 단순히 “KV head가 하나”라고 설명할 수 없다.

독자가 장을 마칠 때는 `max_model_len` 하나로 cache 용량을 말하지 않아야 한다. layer family, logical committed length, physical resident rows, KV 또는 latent width, storage dtype, quant scale, page metadata, rank ownership을 한 원장에 적어야 한다. cache policy와 eviction·분산 전송은 33~39장에서 확대한다. 여기서는 model이 요구하는 shape와 position invariant를 정확히 만든다.

소유 경계를 먼저 고정하자. 이 장은 RoPE·GQA·MLA가 **무엇을 저장해야 하는지**를 정한다. 33장은 그 logical component를 byte와 capacity로 환산하고, 36장은 full·sliding·recurrent group이 그 state를 어느 physical address에 놓고 언제 재사용하는지를 소유한다. 뒤의 두 장이 같은 shape 이름을 다시 쓰더라도 질문은 계산과 주소 수명으로 바뀐다.

## 14.1 RoPE는 위치를 더하지 않고 2차원 쌍을 회전한다

head vector의 두 성분 `(x0,x1)`을 각도 θ만큼 회전하면 다음과 같다.

```text
x0' = x0 cos θ - x1 sin θ
x1' = x0 sin θ + x1 cos θ
```

position `p`, pair index `i`, base `b`, rotary dimension `drot`를 쓰면 흔한 주파수는 `ω_i=b^(-2i/drot)`이고 각도는 `θ=pω_i`다. 구현마다 pair layout과 scaling contract가 다르므로 이 식은 기준축이지 config 없이 사용할 완전한 사양은 아니다. 중요한 성질은 같은 orthogonal rotation이 vector norm을 보존하고, query와 key의 내적이 절대 위치보다 상대 각도 차이를 반영한다는 점이다.

작게 계산하자. query pair가 `(1,2)`, key pair가 `(3,-1)`이고 position이 각각 2와 1이며 주파수 `ω=π/2`라고 하자. query 각도는 π라서 `q'=(-1,-2)`다. key 각도는 π/2라서 `k'=(1,3)`이다. 회전 뒤 내적은 `-1×1 + -2×3 = -7`이다. 상대 회전으로 쓰면 원래 q를 `θk-θq=-π/2`만큼 변환한 key와 내적한 것과 같다. 실제 모델의 주파수는 이런 단일 π/2가 아니지만 위치 차이가 score에 들어가는 원리를 손으로 확인할 수 있다.

또 다른 pair `(1,0)`에 θ=π/6을 적용하면 `(√3/2,1/2)`가 된다. norm 제곱은 전후 모두 1이다. 따라서 RoPE 자체가 magnitude를 키우는 연산이라고 설명하면 틀린다. 다만 finite precision, scaling variant, 후속 attention 계산 때문에 수치 오차는 생길 수 있다. Q/K normalization과 RoPE의 순서도 architecture source에서 확인한다.

### pair layout은 같은 수식의 메모리 계약이다

두 가지 흔한 구현이 있다. interleaved layout은 `(x0,x1),(x2,x3)`처럼 이웃 성분을 pair로 묶는다. half-split layout은 앞 절반과 뒤 절반을 `(x0,x[d/2]), (x1,x[d/2+1])`처럼 짝짓는다. 둘 다 2D 회전이지만 같은 flat array에 다른 permutation을 적용한다. cos/sin shape가 broadcast되어도 layout이 불일치하면 모든 값이 finite이고 shape도 맞는데 결과가 틀린다.

Transformers Llama의 [rotary module](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/modeling_llama.py#L73-L137)과 [apply helper](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/modeling_llama.py#L138-L166)는 cos/sin 생성과 `rotate_half` 계열 적용의 기준점이다.

backend가 `interleaved` flag를 받는다면 model helper와 동일한 pair convention인지 확인한다. flag 이름이 같다는 사실보다 입력 tensor의 마지막 차원 permutation이 증거다.

`rotary_dim`이 head dimension보다 작을 수도 있다. 이때 Q/K의 앞 또는 지정된 부분만 회전하고 나머지 non-positional component는 그대로 둔다. `[q_rot,q_pass]`의 slice boundary가 config, weight packing, kernel ABI에서 같아야 한다. 전체 head를 회전했다고 가정하면 MLA의 decoupled positional component나 partial RoPE 모델을 설명하지 못한다.

### position, cache position, physical slot은 다른 좌표다

논리 position은 token이 sequence에서 갖는 위치다. cache position은 cache API가 어느 logical index를 update하는지 나타낼 수 있다. physical slot은 paged allocator가 실제 메모리 block의 어느 row에 쓰는지 나타낸다. dense dynamic cache에서는 세 값이 우연히 비슷해 보이지만 paged serving과 sliding window에서는 갈라진다.

request A의 logical position 513이 physical block 9의 slot 1에 들어갈 수 있다. 다음 request B의 position 7은 block 2 slot 3일 수 있다. RoPE는 513과 7을 사용해야지 physical slot 1과 3을 사용하면 안 된다. block을 재배치하거나 prefix를 공유해도 이미 회전된 K의 논리 위치 의미는 바뀌지 않는다.

Transformers cache layer의 추상 계약은 [CacheLayerMixin](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L27-L112), 동적 append는 [DynamicLayer.update](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L113-L174),

정적 주소 update는 [StaticLayer](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L398-L503)에서 비교할 수 있다. cache object 이름이 같아도 logical append와 indexed write의 lifetime은 다르다.

## 14.2 prefill에서 decode까지 position 계약이 끊기면 첫 token 뒤부터 갈린다

길이 4 prompt의 position이 0,1,2,3이라고 하자. prefill은 네 token의 Q/K를 해당 각도로 회전하고 K/V를 commit한다. sampler가 token t4를 고르면 첫 decode input의 logical position은 4다. cache에는 네 과거 row가 있고 새 K/V는 position 4 의미로 기록된다. 다음 step은 5다. 단순 text causal model의 기준 invariant는 `next_position = committed_logical_tokens`다.

off-by-one 오류가 prefill 마지막 position을 4로 만들거나 첫 decode를 다시 3으로 만들면 첫 prefill logits가 맞을 수도 있다. prefill 내부의 상대 위치가 일관되면 마지막 hidden은 reference와 우연히 맞거나 차이가 작을 수 있지만 decode query와 cached key의 상대 각도가 어긋난다. “첫 응답은 맞고 두 번째 호출부터 틀린다”는 증상에서 tokenizer만 반복 조사할 이유가 없다.

### chunked prefill은 호출 횟수가 position을 reset하지 않는다

8-token prompt를 3,3,2 token chunk로 처리한다고 하자. 첫 chunk position은 0~2, 둘째는 3~5, 셋째는 6~7이어야 한다. 각 forward의 local row index 0,1,2를 position으로 쓰면 chunk boundary마다 RoPE가 reset된다. tensor shape와 causal mask는 정상이고 cache row 수도 8이지만 cached K의 회전이 틀린다.

ledger에는 `logical_start`, `query_length`, `committed_before`, `positions`, `write_slots`, `committed_after`를 둔다. 둘째 chunk라면 `committed_before=3`, positions `[3,4,5]`, successful commit 뒤 6이다. 실패하거나 취소되면 committed length가 부분 update와 어떤 transaction contract를 갖는지 확인한다. 단순히 allocated rows 수를 logical committed length로 쓰면 rollback에서 어긋난다.

### speculative decoding은 scheduled와 accepted position을 분리한다

position 20에서 draft가 token 네 개를 제안해 20~23을 검증했지만 앞 두 개만 accept했다고 하자. 다음 committed position은 22다. 검증용 temporary K/V 네 row를 모두 permanent cache length로 세면 rejected token 두 개가 ghost state가 된다. 구현은 rollback하거나 accepted prefix만 commit하거나 별도 staging buffer를 사용할 수 있다.

원장에는 proposed, executed, accepted, committed token count를 나눈다. allocator가 네 slot을 예약했다가 두 slot을 free해도 logical length가 24에 남아 있으면 틀리고, logical length는 22인데 stale page-table entry가 read set에 남아도 틀린다. 다음 decode의 position, mask KV length, physical read slots가 모두 accepted count에 합의해야 한다.

### sliding window에서는 물리 길이와 누적 위치가 갈라진다

window 4096인 local attention layer가 logical position 10000을 처리해도 resident K/V row는 최근 4096개뿐일 수 있다. physical tensor 길이를 4096으로 잘랐다고 다음 position을 4096으로 reset하면 안 된다. 누적 logical length는 10001로 진행하고, mask는 resident tensor의 왼쪽 offset이 어느 absolute position인지 알아야 한다.

Transformers의 dynamic sliding layer는 [update와 crop 계약](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L175-L290)에서 resident tensor와 cumulative length를 구분한다. [get_mask_sizes](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L263-L280)는 KV length와 offset을 함께 반환한다. offset을 잃으면 physical row 0을 logical position 0으로 오해한다.

### long-context scaling은 base 하나를 임의로 바꾸는 레시피가 아니다

RoPE scaling은 원래 학습 문맥보다 긴 위치에서 주파수를 조정하는 여러 계약을 포괄한다. linear scaling, dynamic NTK 계열, YaRN 같은 이름이 있지만 지원 field와 계산은 model config 및 library version에 고정된다. `factor` 하나가 모든 구현에서 같은 수식을 뜻한다고 가정하지 않는다. original maximum position, attention factor, 저주파/고주파 경계 같은 추가 parameter가 있을 수 있다.

공식 Transformers RoPE utility와 각 model config가 허용하는 `rope_parameters` 또는 `rope_scaling`을 source로 확인한다. Gemma3 config는 [local/full attention별 RoPE parameter 구성](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/configuration_gemma3.py#L91-L151)을 가진다.

Qwen3.5 config의 [RoPE validation 예외와 mRoPE field](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/configuration_qwen3_5.py#L74-L122)를 모델 독립 field로 일반화하지 않는다.

scaling 설정이 바뀌면 같은 position의 cos/sin과 회전된 K가 바뀐다. 이전 설정으로 만든 prefix cache를 새 설정에서 재사용해서는 안 된다. cache identity에 model artifact와 RoPE parameter digest가 포함되어야 하는 이유다. option parser가 새 값을 받아들였다는 사실만으로 backend가 해당 scaling을 지원했다는 증거도 아니다. selector, fallback, 실제 cos/sin producer를 확인한다.

## 14.3 GQA와 MQA는 query 수가 아니라 저장할 KV head 수를 줄인다

multi-head attention에서 query head 수 `Hq`와 KV head 수 `Hkv`가 같으면 MHA다. GQA는 `Hkv < Hq`이고 여러 query head가 한 KV head를 공유한다. MQA는 흔히 `Hkv=1`인 극단이다. query head h가 읽는 KV head는 균등 group일 때 `floor(h / (Hq/Hkv))`로 매핑할 수 있다. 구현은 repeat view, broadcast 또는 kernel 내부 mapping으로 표현한다.

`Hq=8`, `Hkv=2`라면 query 0~3은 KV head 0, query 4~7은 KV head 1을 읽는다. K/V를 물리적으로 네 번 repeat하면 attention input은 MHA 같은 shape로 보이지만 cache는 원본 두 KV head만 저장하면 된다. optimized backend는 repeat tensor를 materialize하지 않고 head mapping으로 읽는다. profiler에서 repeat kernel이 없다고 GQA가 적용되지 않았다고 판단하면 안 된다.

### dense KV byte 하한을 계산한다

full-attention layer 수 `L`, committed token 수 `N`, KV head 수 `Hkv`, head dimension `Dh`, element byte `e`라면 K와 V 저장의 dense 하한은 다음과 같다.

```text
KV_data_bytes = 2 × L × N × Hkv × Dh × e
```

예를 들어 L=32, N=8192, Hkv=8, Dh=128, BF16 e=2라면 `2×32×8192×8×128×2 = 1,073,741,824 bytes`, 정확히 1 GiB다. 같은 Hq=32에서 MHA라면 Hkv=32이므로 4 GiB다. MQA Hkv=1이면 128 MiB다. 이것은 batch 한 sequence의 data 하한이며 allocator page slack, block table, scale, workspace, replication을 제외한다.

batch의 request i마다 resident length `Ni`가 다르면 N 대신 `ΣNi`를 쓴다. prefix page를 물리적으로 공유한다면 두 logical request가 같은 data row를 참조할 수 있으므로 단순 합은 physical bytes를 과대평가한다. 반대로 TP rank마다 KV를 replicate하면 cluster total은 rank local 합보다 커진다. logical chargeback, unique physical data, rank aggregate를 별도 열로 둔다.

### TP는 head divisibility와 replication 정책을 바꾼다

TP=4이고 Hkv=8이면 rank마다 KV head 두 개를 shard할 수 있다. rank local cache data는 global의 1/4이다. 그러나 Hkv=2인데 TP=4라면 단순하게 head를 0.5개씩 나눌 수 없다. 구현은 KV head를 rank group에 replicate하거나 TP 제약을 reject하거나 다른 partition을 사용한다. MQA Hkv=1은 각 rank가 동일 K/V를 보유하는 설계가 흔하다.

replication factor `R`을 쓰면 cluster data는 `dense_global_bytes × R`로 볼 수 있다. 여기서 R은 무조건 TP가 아니다. Hkv와 TP의 gcd/grouping, backend 구현에 따라 달라진다. rank-local byte를 계산하려면 각 rank의 `local_Hkv`와 replica group을 source에서 읽는다. 옵션 `tensor_parallel_size`를 늘리면 weight는 줄어도 KV가 같은 비율로 줄지 않을 수 있다.

vLLM의 Qwen attention construction과 KV head partition은 [Qwen2 attention](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen2.py#L111-L235), Qwen3 layer는 [Qwen3 decoder](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3.py#L173-L263)에서 출발한다.

SGLang의 대응 경계는 [Qwen3 attention/layer](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3.py#L180-L434)다. line span은 fixed revision symbol과 함께 검산한다.

### quantized cache는 data 이외에 scale과 metadata를 가진다

KV data를 FP8 1 byte로 저장하면 BF16 대비 data 하한은 절반이다. 그러나 dequantization scale이 tensor별, head별, block별 또는 token group별로 필요하다. scale count `Ns`, scale element byte `es`, zero-point와 auxiliary metadata `M`이 있다면 `total = KV_data + Ns×es + M + page_slack`이다. “FP8이므로 정확히 절반”은 data portion에만 맞는다.

K와 V가 서로 다른 scale을 쓰는지, scale이 layer와 head마다 있는지, calibration 또는 dynamic calculation을 누가 하는지 확인한다. cache dtype option이 accepted되어도 selected attention backend가 해당 dtype을 지원하지 않으면 reject, fallback, conversion path 중 하나가 된다. weight quantization과 cache quantization은 다른 boundary다. residual과 Q projection은 BF16인데 cache storage만 FP8일 수 있다.

## 14.4 MLA는 K/V head를 공유하는 대신 latent와 위치 성분을 저장한다

MLA를 “더 강한 MQA”라고 부르면 중요한 구조를 잃는다. DeepSeek 계열 MLA의 기준 그림에서는 input hidden에서 compressed KV latent `c_KV`를 만들고, 이를 나중에 key의 non-positional component와 value에 필요한 representation으로 사용한다. key의 rotary positional component `k_R`은 별도로 만든다. query도 non-positional `q_C`와 rotary `q_R`로 나뉜다. cache는 full per-head K/V 대신 compressed latent와 shared positional component를 저장할 수 있다.

Transformers DeepSeek V3 config의 [kv_lora_rank와 RoPE head dimension](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/deepseek_v3/configuration_deepseek_v3.py#L70-L126), attention의 [compressed KV projection](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L362-L422),

[latent/rotary split과 reconstruction](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L423-L470)을 함께 읽는다. source의 eager/reference materialization과 optimized serving backend의 latent cache ABI를 구분한다.

### MLA cache 하한을 모델 field로 계산한다

full-attention MLA layer 수 `Lmla`, token 수 N, KV latent rank `Rkv`, shared rotary key dimension `Dr`, storage byte e를 쓰자. token당 latent와 rotary component를 한 벌씩 저장하는 단순 하한은 다음과 같다.

```text
MLA_data_bytes = Lmla × N × (Rkv + Dr) × e
```

예를 들어 Lmla=32, N=8192, Rkv=512, Dr=64, BF16이면 `32×8192×576×2 = 301,989,888 bytes`, 약 288 MiB다. dense GQA 예의 1 GiB보다 작지만 이 비교는 model width와 layer 수, backend materialization이 같은 조건일 때만 의미 있다. MLA가 runtime에 reconstructed per-head K/V를 장기 cache로 함께 보존하면 실제 bytes가 달라진다. workspace와 projection weight도 이 식에 없다.

TP에서 latent가 replicated인지 shard되는지, shared `k_R`이 rank마다 복제되는지 확인한다. latent dimension을 shard해도 attention score 계산이 어떤 collective를 요구하는지 backend 설계에 달려 있다. cluster total과 rank peak를 모두 계산한다. “head 수가 1이므로 MQA와 같다”는 식은 Rkv=512 latent를 누락한다.

### decoupled RoPE가 cache identity를 바꾼다

MLA의 non-positional latent와 rotary key component는 서로 다른 변환과 lifetime 의미를 가진다. `k_R`은 position과 RoPE config에 의존하고, latent는 hidden projection과 quantization/normalization에 의존한다. prefix reuse identity는 둘 모두 일치해야 한다. rotary scaling만 바뀌었는데 latent가 같다고 cache 전체를 재사용하면 positional component가 stale하다.

optimized backend는 weight absorption을 이용해 full K/V reconstruction을 피하거나 계산 순서를 바꿀 수 있다. 그러나 semantic score와 value aggregation은 reference와 일치해야 한다. backend가 MLA head dimension, dtype, GPU capability 또는 quantized latent를 지원하지 않으면 eager-like materialization으로 fallback할 수 있다. 이 경우 cache data는 작아도 step workspace와 compute가 예상보다 커진다.

fallback incident에서는 option과 model config만 보지 않는다. selected backend symbol, latent cache tensor shape, reconstructed temporary shape, kernel workspace를 확인한다. latency가 늘고 memory peak가 커졌지만 long-lived cache bytes는 그대로라면 full reconstruction workspace 가설이 살아 있다. long-lived bytes 자체가 per-head 크기로 늘었다면 cache ABI가 달라졌을 가능성이 있다.

## 14.5 Gemma와 Qwen은 “layer마다 같은 cache”라는 가정을 깨뜨린다

Gemma3 계열은 local sliding attention과 global attention layer를 조합할 수 있다. local layer는 최근 window만 resident하게 유지할 수 있지만 global layer는 더 긴 과거를 요구한다. 따라서 model 전체의 KV byte를 `모든 layer × max length`로만 계산하면 local crop을 과대평가하고, `모든 layer × window`로 계산하면 global layer를 과소평가한다.

Gemma3 config의 [sliding window와 layer pattern](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/configuration_gemma3.py#L70-L151), model의 [rotary embedding](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L156-L235),

[attention apply helper](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L236-L264)를 layer type과 함께 읽는다. local과 global이 다른 RoPE parameter set을 가질 수 있으므로 cos/sin cache identity도 layer family를 포함한다.

`Lg` global layer, `Ll` local layer, logical N, window W라면 dense data 하한은 `2×Hkv×Dh×e×(Lg×N + Ll×min(N,W))`다. N=32768, W=4096, Lg=8, Ll=24, Hkv=8, Dh=128, e=2라면 token-layer row는 `8×32768 + 24×4096 = 360448`이고 K/V data는 약 1.375 GiB다. 모든 32 layer가 global이면 4 GiB, 모두 window면 512 MiB다. layer mix를 빼면 이 차이를 설명할 수 없다.

### Qwen mRoPE는 하나의 scalar position을 여러 modality 축으로 확장한다

Qwen3.5 multimodal 경로는 temporal, height, width 좌표를 반영하는 position IDs를 구성한다. text-only에서는 축들이 같은 증가 좌표처럼 보일 수 있지만 image/video placeholder 구간에서는 grid coordinate가 다르다. Transformers의 [text rotary module과 mRoPE section](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L84-L170),

[multimodal position 작성](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1290-L1377), [prefill/decode rope delta 연결](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1450-L1490)을 한 계약으로 본다.

text token IDs와 placeholder 수가 같아도 image grid와 `mrope_position_deltas`가 다르면 decode position이 갈릴 수 있다. 첫 multimodal prefill에서 만든 delta는 이후 한-token decode가 modality 구간 뒤의 올바른 좌표를 이어 가도록 보존된다. KV cache만 checkpoint하고 rope delta를 잊으면 prefill은 맞고 decode부터 틀릴 수 있다.

mRoPE section은 head dimension의 어느 channel group이 temporal/height/width frequency를 사용할지 결정한다. section 합과 interleaving convention이 config와 kernel에서 같아야 한다. 일반 text RoPE kernel에 tensor shape만 맞춰 넣고 축 selection을 잃으면 이미지 요청에서만 오답이 난다. 따라서 backend 지원 검사는 dtype과 head dim뿐 아니라 mRoPE layout을 포함해야 한다.

### hybrid attention/recurrent 모델은 KV와 다른 state를 동시에 가진다

Qwen3.5의 일부 layer는 full attention이 아니라 gated delta/recurrent 계열일 수 있다. 그 layer는 모든 과거 token의 K/V 대신 convolution/recurrent state를 보존한다. 총 state는 attention layer KV, recurrent layer state, page metadata와 runner workspace의 합이다. cache dtype을 낮췄다고 recurrent state dtype까지 자동으로 바뀐다고 단정하지 않는다.

Transformers cache abstraction도 alternating attention/linear layer를 구분한다. [LinearAttentionLayer](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L886-L1080)와 cache의 [layer-aware

sequence length](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L1485-L1584)는 모든 layer에 동일 `get_seq_length`를 호출하는 가정을 경계한다. recurrent state shape와 reset contract는 15장의 owner다.

## 14.6 logical append는 page write·read metadata로 번역된다

수학에서는 새 K/V를 sequence 축 뒤에 붙인다고 말한다. serving engine은 매 step tensor를 concat하면 복사와 fragmentation이 커지므로 고정 크기 block/page를 할당하고 logical token을 physical slot에 쓴다. block size가 16이면 logical positions 0~15가 한 block에 들어갈 수 있지만 prefix sharing과 eviction 때문에 block ID는 연속일 필요가 없다.

write metadata는 새 token row마다 destination slot을 정한다. read metadata는 request별 block table, context length, last-page valid count로 과거 K/V를 모은다. logical order는 block table 순서와 slot offset으로 복원된다. physical memory 주소 순서가 sequence order라고 가정하지 않는다.

vLLM의 paged cache 관리와 runner input은 [KV cache manager](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/kv_cache_manager.py#L1-L220)와 [GPU model runner](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu_model_runner.py#L1-L260)에서 owner를 찾는다.

정확한 write kernel은 selected attention backend로 더 내려가야 한다. SGLang의 radix/pool 구조는 [token-to-KV pool](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/memory_pool.py#L1-L260)과 model runner metadata를 연결한다.

### prefix sharing은 logical owner가 둘이고 physical data가 하나다

두 request가 동일 prompt prefix 1000 token을 공유하면 logical chargeback은 각 request 1000 row지만 physical page는 한 벌일 수 있다. 공유 조건은 token IDs만이 아니다. model/adapter, position/RoPE config, multimodal embedding, cache dtype/scale, layer state contract가 같아야 한다. tenant isolation과 mutation policy도 포함한다.

copy-on-write가 필요한 마지막 partial page를 두 request가 동시에 extend하면 어느 시점에 분리하는지 본다. 공유 page를 직접 덮어쓰면 한 request의 새 K/V가 다른 request에 보인다. refcount가 2인데 free하면 use-after-free가 되고, refcount가 내려가지 않으면 memory leak처럼 보인다. prefix hit count만으로 saved bytes와 correctness를 증명할 수 없다.

### page slack과 metadata를 byte 원장에 더한다

block size P에서 각 sequence 마지막 block의 unused slots가 slack이다. request 수가 많고 평균 length가 짧으면 data 하한보다 예약 bytes가 크게 늘 수 있다. `allocated_slots × per_token_layer_bytes`와 `committed_tokens × per_token_layer_bytes`를 나눠 utilization을 계산한다. free block reserve와 graph capture용 고정 pool도 별도다.

block table index, sequence length, slot mapping, refcount, hash, scale metadata는 data보다 작아도 correctness owner다. 매우 짧은 request가 수십만 개면 CPU/GPU metadata와 transfer 비용이 무시할 수 없을 수 있다. 그러나 source나 측정 없이 metadata가 병목이라고 단정하지 않는다. 원장에 크기와 update frequency, 이동 경로를 적는다.

llama.cpp는 paged serving engine과 동일한 allocator를 쓰지 않으며 context/KV abstraction이 다르다. [context graph build](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-context.cpp#L1320-L1435), [KV cache implementation](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-kv-cache.cpp#L1-L220),

[model graph builder](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model.cpp#L2461-L2550)을 통해 logical token positions, cache cells와 graph input을 대응시킨다. vLLM block table 용어를 llama.cpp에 그대로 투사하지 않는다.

## 14.7 장애 워크북은 position·shape·lifetime을 따로 반증한다

워크북은 model이나 CUDA를 이 집필 과정에서 실행하라는 뜻이 아니다. 고정 source와 config로 예상 invariant, 허용된 별도 환경에서 수집할 field, 결과별 분기를 설계한다. 전체 K/V dump는 메모리와 개인정보를 크게 교란하므로 synthetic fixture와 bounded slice, digest를 우선한다.

### 사건 1: prefill은 맞고 첫 decode부터 틀린다

먼저 prefill 마지막 final hidden과 selected token이 같은지 확인한다. 같다면 첫 decode input ID와 embedding, logical position, committed cache length를 비교한다. Q projection은 같고 rotated Q/K부터 다르면 position 또는 scaling/layout을 본다. 새 K는 같지만 attention update가 다르면 cached K/V read slots와 mask offset을 본다.

경쟁 가설은 decode position off-by-one, chunk commit length 오류, stale page table이다. position IDs가 reference와 같으면 첫 가설이 약해지고, block table을 dense logical order로 재구성한 digest가 같으면 세 번째가 약해진다. 수정 후 prompt length 1, block boundary 전후, chunk boundary, 여러 decode step을 검증한다.

### 사건 2: TP를 늘렸더니 OOM이 줄지 않거나 오답이 난다

weight shard는 줄었지만 KV head가 replicate되었는지 계산한다. rank별 local Hkv, replica group, cluster total을 적는다. OOM이 cache capacity인지 workspace/graph pool인지 allocator metric으로 나눈다. 오답이면 local Q/K/V slice, head→KV mapping, collective 전후 update를 비교한다.

Hkv가 TP로 나누어지지 않을 때 backend가 어떤 mapping을 선택하는지 source에서 확인한다. rank마다 동일 KV를 보유하는 것이 설계라면 중복 자체는 bug가 아니다. 서로 달라야 할 shard가 동일하거나 동일해야 할 replica가 다른 것이 bug다. TP=1과 TP=2를 비교할 때 global head order로 재조립한다.

### 사건 3: 긴 문맥 옵션 뒤 prefix cache가 hit하지만 출력이 틀린다

cache key에 RoPE scaling config와 original max position, adapter/model revision이 포함되는지 본다. 이전 base로 회전된 K를 새 scaling에서 읽으면 token IDs가 같고 prefix hit도 정상으로 보인다. prefix를 비운 cold run이 맞고 warm reuse만 틀리면 identity 가설이 강해진다.

fix는 hit를 끄는 데서 끝나지 않는다. cache identity schema를 바꾸고 incompatible entry가 공유되지 않는지 검증한다. 같은 scaling의 두 request는 여전히 공유되어야 한다. hit rate, saved compute, correctness를 함께 본다. long-context option이 실제 backend에 적용되지 않아 fallback했다면 별도 성능 incident로 분리한다.

### 사건 4: image 요청만 decode에서 갈린다

text IDs, placeholder 위치, vision grid, multimodal position IDs, mRoPE section, rope delta를 비교한다. text-only fixture가 맞는다는 negative evidence는 일반 RoPE와 weight 오류를 약화하지만 완전히 기각하지 않는다. prefill layer checkpoint가 같고 decode rotated Q부터 다르면 saved delta와 다음 position 계산을 본다.

kernel이 scalar position만 지원해 fallback했는지, interleaved mRoPE를 다른 layout으로 적용했는지 구분한다. fallback이면 값은 맞고 성능만 다를 수 있다. layout 오류면 shape와 runtime 성공에도 값이 틀린다. selected backend symbol과 small pair/channel fixture가 서로 다른 질문에 답한다.

### 사건 5: MLA 모델이 예상보다 메모리를 많이 쓰고 느리다

먼저 long-lived cache tensor가 `[N,Rkv+Dr]`인지 reconstructed `[N,H,D]`인지 확인할 관측을 설계한다. latent cache는 작지만 매 step reconstructed workspace가 크면 peak와 latency가 늘 수 있다. backend eligibility가 head dim, dtype, GPU capability, TP layout 때문에 실패했는지 source selector를 본다.

config의 MLA field가 존재한다는 사실은 optimized MLA kernel이 실행되었다는 증거가 아니다. model reference path가 full head tensor를 materialize할 수 있다. requested backend, eligibility result, effective backend, cache ABI, workspace shape를 인과 사슬로 쓴다. fix 후 cache bytes뿐 아니라 output parity와 fallback log를 확인한다.

### 사건 6: speculative reject 뒤 다음 token이 간헐적으로 틀린다

proposed/accepted/committed length, staged slots, free list, read table을 step별로 적는다. rejected row의 cache data가 남아 있어도 read set에서 제외되고 나중에 안전하게 overwrite되면 correctness 문제는 아니다. logical length가 전진했거나 stale slot이 read table에 남으면 문제다.

request cancellation과 slot reuse까지 결합해 incarnation ID를 확인한다. 같은 numeric request ID나 slot ID가 재사용되어도 이전 state와 구별되어야 한다. next position과 mask length, block table valid count가 accepted count에 합의하는지 본다. 단순 allocated byte가 원상복구되었다는 사실은 충분하지 않다.

### differential checkpoint 표

| checkpoint | 같은데 다음이 다르면 | 우선 owner |
|---|---|---|
| token IDs와 embeddings | raw Q/K/V가 다름 | weight shard·quant·adapter |
| raw Q/K | rotated Q/K가 다름 | position·RoPE config·pair layout |
| rotated new K/V | cache read가 다름 | write slot·page table·prefix identity |
| logical cache rows | backend output이 다름 | mask offset·kernel ABI·numeric |
| rank local output | global update가 다름 | TP mapping·collective |
| accepted token count | next decode state가 다름 | rollback·commit·incarnation |

한 checkpoint가 같다는 판정에는 logical row ordering과 dtype metadata가 포함되어야 한다. physical array를 그대로 hash하면 page permutation 때문에 semantic equality를 놓친다. 반대로 norm만 비교하면 row swap을 놓친다. request incarnation, logical position, layer, KV component를 key로 재정렬한 뒤 digest와 bounded values를 비교한다.

## 14.8 position·representation·lifetime을 하나의 canonical 표로 고정한다

한 요청의 cache는 다음 한 표로 설명한다. 별도 position 원장, byte 원장과 lifetime dossier를 만들지 않는다.

| canonical key·사건 | position | representation·component byte | physical lifetime·owner | 완료·rollback 관측 |
|---|---|---|---|---|
| request generation·layer·logical row | scalar 또는 T/H/W·delta, pair layout | GQA `K/V×Hkv×Dh`; MLA `latent Rkv + rotary Dr`; dtype·scale | rank shard/replica, page·slot, refcount | accepted count만 committed length와 next position에 반영 |
| prefill/decode write | committed before→positions→after | data + scales + page slack + metadata | reserved→written→read-visible | failure/cancel이면 staging 폐기, generation 유지 |
| prefix/sliding read | absolute position·resident offset | component offset과 backend ABI | shared page 또는 ring owner | digest 불일치는 miss, old event 뒤 reclaim |
| capacity·성능 | logical rows와 unique resident rows | persistent + temporary + workspace | rank peak와 cluster unique 분리 | parity 뒤 byte·latency guardrail 판정 |

첫 열의 layer family는 global attention, sliding attention, MLA, recurrent/conv를 구분한다. Position 열에는
token 좌표, multimodal axes, cache update index와 physical slot을 섞지 않는다. Representation 열은
Hkv·Dh 또는 Rkv·Dr와 scale granularity를 보존한다. Lifetime 열은 allocated, written, committed,
shareable, reclaimable을 generation과 event로 잇는다.

이 장이 답한 것은 model shape 계약이다. 33~39장은 이 계약을 받아 block allocation, prefix identity, eviction, CPU/NVMe 계층, LMCache·Mooncake와 P/D 전송에서 state가 어떻게 이동하는지 다룬다. 그때도 RoPE가 적용된 K와 logical position, MLA latent/rotary component, recurrent state를 모두 “KV bytes”라는 한 상자로 뭉개지 않는다.

고정 source를 다시 찾을 때는 Transformers [Cache.update와 layer dispatch](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L1262-L1600), [QuantizedCache](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L1877-L1939), DeepSeek MLA [attention

body](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L362-L470), Qwen mRoPE [position preparation](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1290-L1490)을 의미 좌표로 사용한다.

독자는 이제 다음 질문에 답할 수 있어야 한다. 같은 logical position을 RoPE producer, mask, cache manager가 공유하는가. GQA에서 query head가 어느 KV head를 읽고 그 head는 어느 rank에 저장되는가. MLA cache가 latent와 rotary component를 어떻게 나누며 backend가 이를 그대로 유지하는가. sliding window의 physical length와 cumulative position이 왜 다른가. rejected token과 취소된 request의 state가 언제 commit되거나 폐기되는가.

이 중 하나라도 모르면 cache hit율이나 GPU memory 숫자를 최종 설명으로 쓰지 않는다. 먼저 shape와 lifetime owner를 찾는다. 그래야 긴 문맥 OOM, decode 오답, TP replication, prefix 오염, backend fallback을 서로 다른 문제로 분기하고, 수정 뒤 무엇을 검증해야 하는지도 정확히 정할 수 있다.

## 14.9 canonical 표를 채우는 열 가지 계산 사례

이 절은 앞의 개념을 한 장짜리 계산으로 끝내지 않는다. 같은 config라도 MHA·GQA·MLA, local/global mix, TP replication, quant scale, page slack에 따라 숫자가 어떻게 달라지는지 여러 번 다시 계산한다. 각 계산에는 증명 범위를 붙인다. 모델 config가 data 하한을 증명할 수는 있지만 allocator가 실제로 몇 block을 예약했는지는 engine state가 필요하다. source가 fallback branch를 보여 줄 수는 있지만 특정 배포가 그 branch를 탔다는 사실은 effective log나 trace가 필요하다.

**사례 A: MHA에서 GQA로 바뀔 때 무엇이 줄고 무엇이 그대로인가**

기준 모델은 hidden width 4096, query head 32, head dimension 128, 32 attention layer, context 16,384, BF16이다. MHA는 Hkv=32다. token·layer당 K/V element는 `2×32×128=8192`, byte는 16,384다. sequence 전체는 `32×16384×16384 bytes = 8,589,934,592 bytes`, 즉 8 GiB다. 한 request의 dense data 하한만 이만큼이다.

GQA Hkv=8로 바꾸면 token·layer당 K/V byte는 `2×8×128×2=4096`이다. 전체는 2 GiB다. MQA Hkv=1이면 token·layer당 512 byte, 전체 256 MiB다. 이 차이는 query projection과 query head 수가 줄었다는 뜻이 아니다. Q는 여전히 32 head이고 현재 token마다 4096 feature를 만든다. attention output도 32 query head에서 나온다. 저장하고 읽는 과거 K/V의 unique head 수가 줄었다.

따라서 prefill projection FLOP이 정확히 1/4이 된다고 말하면 틀린다. K/V projection output 폭은 줄지만 Q와 O, MLP는 그대로이고 attention score의 query head 수도 그대로다. decode의 KV read bytes는 크게 줄 수 있지만 실제 ITL 개선은 weight read, launch, O/MLP GEMM, collective, scheduler 구성에 달렸다. cache capacity와 end-to-end latency를 같은 비율로 연결하지 않는다.

TP=8을 넣어 보자. MHA Hkv=32는 rank당 네 KV head로 나누기 쉽다. GQA Hkv=8은 rank당 한 head가 가능하다. MQA Hkv=1은 여덟 rank에 동일 head를 replicate한다면 rank local cache가 256 MiB이고 cluster aggregate는 2 GiB다. 단일 rank 기준 capacity는 MQA 이득이 남지만 cluster total은 replication 때문에 GQA Hkv=8 shard와 같아질 수 있다. weight memory와 rank capacity, cluster cost를 한 숫자로 합치지 않는다.

TP=16에서 Hkv=8이라면 두 rank가 한 KV head replica group을 이룰 수 있다. global unique data는 2 GiB지만 cluster aggregate는 4 GiB다. rank local은 256 MiB다. 구현이 이 mapping을 지원하지 않고 reject할 수도 있다. config divisibility validation, local KV head 계산, attention backend mapping을 source로 연결한 뒤 실제 정책을 적는다.

**사례 B: 여러 request와 prefix sharing을 unique bytes로 계산한다**

같은 GQA 모델에서 request A 길이 8192, B 길이 4096, C 길이 2048이라고 하자. per-token 전체 layer KV byte는 `32×4096=131072`, 즉 128 KiB다. 공유가 없으면 committed logical data는 `(8192+4096+2048)×128 KiB = 1.75 GiB`다.

A와 B가 첫 3072 token prefix를 공유하고 physical page도 재사용한다면 logical charge는 그대로 1.75 GiB지만 unique token rows는 `8192+4096+2048-3072=11264`다. unique data 하한은 1.375 GiB다. saved data는 384 MiB다. B의 prefix hit가 3072라는 metric만으로 이 saved byte를 확정하려면 모든 layer/cache component가 실제 공유되었는지 확인해야 한다.

block size가 16이면 세 sequence의 길이가 우연히 block boundary에 맞아 slack이 없다. 길이를 8193, 4097, 2049로 바꾸면 각 request는 마지막 block에 15개 unused slot을 예약한다. 공유가 없는 경우 slack은 45 token row이고 전체 layer KV는 5.625 MiB다. request가 세 개라 작아 보이지만 백만 개의 매우 짧은 sequence에서는 마지막 page slack이 data 이용률을 지배할 수 있다.

prefix의 마지막 공유 block이 partial이면 copy-on-write 계산이 필요하다. 공통 prefix 3073 token은 193번째 block의 첫 slot만 사용한다. A와 B가 서로 다른 다음 token을 append하면 같은 block을 계속 공유해서는 안 된다. implementation이 full block만 공유하면 shared unique row 절감은 3072 token이고, partial block을 COW하면 3073까지 공유할 수 있지만 새 block 복제와 refcount transition이 필요하다.

cache chargeback 표에는 request logical tokens, shared prefix logical tokens, unique physical blocks, last-page valid slots, replica factor를 둔다. tenant별 logical charge와 GPU allocator physical bytes를 구분한다. “B가 4096 token을 사용하므로 512 MiB”와 “B가 추가로 유발한 unique bytes는 128 MiB”가 동시에 참일 수 있다. 어떤 질문에 답하는 숫자인지 이름을 붙인다.

**사례 C: Gemma local/global layer mix를 context sweep으로 계산한다**

앞에서 Lg=8, Ll=24, window=4096을 썼다. 이번에는 context N을 2048, 4096, 8192, 32768로 바꾼다. N≤W에서는 모든 layer가 N rows를 보존하므로 local/global 구분이 capacity를 줄이지 않는다. N=2048이면 layer-token rows는 `32×2048=65536`이다. N=4096이면 131072다.

N=8192부터 local layer가 4096에 고정된다. layer-token rows는 `8×8192 + 24×4096 = 163840`이다. 모든 layer global인 경우 262144이므로 37.5% 줄었다. N=32768에서는 mixed가 360448, all-global은 1,048,576이므로 약 65.6% 줄었다. context가 window를 얼마나 넘는지에 따라 이득 비율이 달라진다.

per layer-token K/V byte가 4096인 GQA BF16이라면 N=8192 mixed data는 640 MiB, all-global은 1 GiB다. N=32768 mixed는 1.375 GiB, all-global은 4 GiB다. 그러나 local layer가 physical crop을 하지 않고 mask만 window로 제한한다면 resident bytes는 all-global에 가까울 수 있다. model semantics의 window와 cache implementation의 resident crop은 다른 주장이다.

local layer의 cumulative position은 N까지 진행한다. N=32768에서 physical resident index 0이 logical position 28672일 수 있다. mask offset은 이 대응을 보존한다. layer별 RoPE parameter가 local/full로 다르면 같은 token position이라도 cos/sin table 또는 scaling contract가 다를 수 있다. cos/sin cache를 model-wide key 하나로 공유할 때 layer family가 identity에 포함되는지 본다.

global layer와 local layer의 block size가 같아도 eviction 가능 시점은 다르다. local layer의 window 밖 row는 그 layer에서 버릴 수 있지만 global layer의 같은 logical token row는 유지해야 한다. allocator가 layer별 block을 독립 관리하는지, mixed page가 함께 묶여 global lifetime을 강제하는지에 따라 실제 saved bytes가 달라진다. config 식은 하한을 주고 allocator layout이 실현 가능성을 결정한다.

**사례 D: MLA data·scale·workspace를 분리한다**

DeepSeek 예로 attention layer 60, N=32768, latent rank Rkv=512, rotary component Dr=64, BF16을 사용한다. long-lived data 하한은 `60×32768×576×2 = 2,264,924,160 bytes`, 약 2.109 GiB다. 비교용 GQA가 Hkv=8, Dh=128이면 `2×60×32768×8×128×2 = 8 GiB`다.

latent와 rotary component를 FP8로 저장하면 raw data는 약 1.055 GiB다. layer별로 K-latent와 rotary component 각각 FP32 scale 하나만 둔다면 scale bytes는 `60×2×4=480 bytes`로 미미하다. 그러나 64-token block마다 component별 scale을 두면 block 수 512, scale bytes는 `60×512×2×4=245,760 bytes`다. head/group별 scale이나 per-token scale이면 더 커진다. 정확한 granularity를 source 없이 고르지 않는다.

reconstruction workspace를 계산해 보자. active decode query 64개, query head 128, reconstructed non-positional key dimension 128을 한꺼번에 materialize하면 단일 current-query 쪽 temporary는 작지만 과거 N 전체를 `[64,128,N,128]`로 펼치는 구현은 비현실적으로 거대하다. optimized MLA가 weight absorption과 latent-domain 계산을 사용하는 이유다. reference source가 full key/value view를 표현해도 backend가 실제로 전체 4D 과거 tensor를 materialize하는지 확인해야 한다.

fallback path가 한 chunk씩 reconstruct하면 workspace는 chunk size에 비례하고 launch/compute가 늘 수 있다. long-lived cache metric은 2.1 GiB로 예상과 맞는데 peak가 훨씬 크고 latency가 느리다면 workspace를 본다. 반대로 allocator의 long-lived block 자체가 8 GiB에 가깝다면 latent ABI 대신 expanded K/V를 저장했는지 본다.

TP=8에서 Rkv를 64씩 shard하고 Dr=64를 모든 rank에 replicate한다고 가정해 보자. rank token당 stored width는 128이고 cluster aggregate width는 `8×128=1024`다. unique semantic width 576보다 replication/partition aggregate가 크다. 다른 설계는 latent를 replicate하고 projection weight를 shard할 수 있다. rank local capacity와 communication을 source backend별로 계산한다.

**사례 E: RoPE frequency와 scaling을 작은 표로 검산한다**

rotary dimension 8, base 10000이면 pair index i=0,1,2,3의 frequency는 `10000^(-2i/8)`이다. 값은 1, 0.1, 0.01, 0.001이다. position p=10에서 각도는 10, 1, 0.1, 0.01 radian이다. 고주파 pair는 빠르게 회전하고 저주파 pair는 천천히 변한다. 이 여러 scale이 상대 위치 범위를 표현한다.

linear scaling factor 2를 단순 예로 들면 effective position을 p/2로 쓰는 variant가 있을 수 있다. p=10 각도가 5,0.5,0.05,0.005가 된다. 그러나 이것을 YaRN의 정의로 부르면 안 된다. YaRN은 주파수 구간별 보간과 attention scaling 등 더 많은 contract를 가질 수 있다. library의 공식 rope type dispatcher와 config field를 기준으로 설명한다.

원래 max position 8192로 만든 model을 32768에 사용하려고 factor 4를 넣었다고 해서 품질과 backend 지원이 자동 보장되지 않는다. config validation을 통과했는지, model architecture가 해당 type을 허용하는지, cos/sin producer가 factor를 적용했는지, fused backend가 같은 결과를 내는지, prefix cache identity가 바뀌었는지 확인한다.

작은 검산 fixture는 position 0,1,원래 max-1,원래 max,확장 max-1을 고른다. pair별 cos/sin, rotated Q/K의 bounded slice를 reference helper와 backend input에서 비교한다. position 0은 scaling 차이가 드러나지 않으므로 그것만 테스트하면 약하다. 고주파와 저주파 channel을 모두 포함한다.

pair layout 검산은 feature가 한 곳만 nonzero인 basis vector를 쓴다. half-split과 interleaved는 nonzero가 이동하는 destination index가 다르다. random vector norm 비교만으로는 permutation 오류를 놓칠 수 있다. basis fixture는 어느 pair가 묶였는지 직접 보여 준다. mRoPE는 temporal/height/width section마다 다른 basis를 넣는다.

**사례 F: mRoPE의 prefill→decode 상태를 복원한다**

text token 두 개, 2×2 image grid 네 개, 뒤 text token 세 개를 단순 fixture로 두자. 실제 processor의 placeholder와 grid ordering은 source를 따르지만 여기서는 상태 열을 설명한다. text 앞 구간은 scalar처럼 증가하고 image 네 token은 temporal/height/width 좌표를 가진다. 뒤 text가 시작할 위치는 flat token count만으로 정하지 않고 multimodal position의 최대와 delta contract를 따른다.

원장에는 token index, token type, T/H/W position, channel section, flat cache row, rope delta를 둔다. prefill이 끝날 때 반환·저장되는 delta와 첫 decode가 만드는 position을 연결한다. 세 구현을 비교할 때 `position_ids` tensor axis order가 다를 수 있으므로 semantic T/H/W로 정규화한다.

image resize나 patch merge가 달라 grid가 2×2 대신 1×4가 되면 placeholder token 수는 같아도 H/W 좌표가 다르다. token IDs만 비교해서는 찾지 못한다. vision embedding이 같아도 mRoPE부터 layer output이 갈릴 수 있다. 반대로 prefill부터 vision embedding이 다르면 position보다 processor/model input을 먼저 조사한다.

decode step에서 image grid를 다시 전달하지 않는 구현은 saved delta/state에 의존할 수 있다. request migration이나 P/D 분리에서 KV만 옮기고 delta를 빼면 destination decode가 잘못된 position을 만든다. transfer manifest에 model-specific auxiliary state가 포함되는지 60~65장 P/D 경로와 연결한다.

**사례 G: source에서 option→effective cache shape를 따라간다**

첫 field는 model config의 `num_attention_heads`, `num_key_value_heads`, `head_dim`이다. loader가 이를 model class에 전달하고 attention constructor가 local query/KV heads를 계산하는 곳을 찾는다. TP world size와 divisibility validation을 연결한다. constructed projection output width와 cache spec이 동일 Hkv를 사용하는지 본다.

두 번째 field는 `max_position_embeddings`와 RoPE parameter다. config validation, rotary module initialization, cos/sin forward, backend adapter까지 내려간다. long-context override가 model config를 mutate하는지 별도 runtime field를 쓰는지 기록한다. source에서 parameter가 설정되지만 consumer가 읽지 않으면 effective state가 아니다.

세 번째 field는 cache dtype이다. CLI/parser에서 enum으로 바뀌고 cache config, allocator byte size, attention backend 지원 검사, write/read kernel dtype으로 전달되는 경로를 잇는다. scale calculation option과 scale tensor lifetime을 찾는다. unsupported 조합에서 error인지 fallback인지 확인한다. 시작 로그 문자열만 source symbol 증거로 사용하지 않는다.

네 번째 field는 block size 또는 page size다. cache spec의 bytes per block, allocator block count, slot mapping, backend page ABI를 잇는다. block size가 capacity utilization과 kernel specialization을 함께 바꿀 수 있다. 값을 키우면 block table은 작아질 수 있지만 last-page slack은 커질 수 있다. latency 효과는 workload length distribution과 backend에 의존한다.

다섯 번째 field는 sliding window다. model mask semantics, cache layer crop, engine cache spec, backend window argument가 같은 inclusive/exclusive boundary를 공유하는지 본다. config가 window total을 나타내는데 kernel API가 left tokens count를 요구하면 `window-1` 변환이 필요할 수 있다. 이 보정을 제거하면 shape는 맞고 한 token을 더 보거나 덜 본다.

각 경로는 `field → validator → derived shape/state → allocator or tensor → backend argument → metric/correctness effect → probe`로 기록한다. field와 effect 사이에 source edge가 없으면 레시피를 보류한다. config에 존재하지만 model family가 사용하지 않는 field도 있다. unknown은 실패가 아니라 다음 조사 좌표다.

**사례 H: cache lifetime을 request state machine에 붙인다**

request가 WAITING에서 RUNNING으로 갈 때 필요한 block을 예약한다. prefill chunk가 실행되면 write가 발생하지만 scheduler가 결과를 commit하기 전 취소나 failure가 날 수 있다. FINISHED 또는 ABORTED에서 refcount를 내리고 free list로 돌아간다. prefix 공유 block은 한 request 종료로 즉시 free되지 않는다.

상태마다 `allocated`, `written`, `logically committed`, `visible to attention read`, `shareable`, `reclaimable`을 boolean 또는 count로 적는다. allocated되었지만 아직 written되지 않은 slot, written되었지만 speculative reject로 committed되지 않은 row, committed이지만 window 밖이라 local layer에서 reclaimable한 row는 서로 다르다.

preemption에서 recompute 정책은 KV를 버리고 prompt/token history로 다시 계산할 수 있다. swap/offload 정책은 data와 metadata ownership을 다른 tier로 옮긴다. 어느 경우든 logical sequence position은 보존되어야 한다. 재개 시 physical slot이 달라져도 RoPE position과 committed length는 이어진다.

request ID 재사용은 incarnation을 요구한다. ID 42가 끝난 뒤 새 request도 42가 될 수 있다면 cache table과 auxiliary mRoPE/recurrent state는 세대 번호로 구별해야 한다. stale asynchronous completion이 새 incarnation의 block table을 갱신하지 못하게 한다. numeric ID가 같다는 이유로 prefix나 state를 공유하지 않는다.

분리 serving에서 prefill node가 cache를 만들고 decode node로 넘긴다면 transfer 완료와 visibility가 commit 경계다. data bytes뿐 아니라 block order, logical positions, dtype/scale, layer type, RoPE identity, mRoPE delta를 전달한다. destination allocator slot은 source와 달라도 된다. transfer manifest가 logical order를 재구성해야 한다.

**사례 I: 여덟 가지 잘못된 설명을 고친다**

첫째, “RoPE는 position embedding을 hidden에 더한다”는 설명은 틀리다. RoPE는 Q/K channel pair를 position-dependent angle로 회전한다. 둘째, “cache index가 position이다”는 설명은 paged/sliding에서 틀리다. logical position과 physical slot을 나눈다.

셋째, “GQA가 head 수를 줄인다”는 말은 모호하다. query head는 유지하고 unique KV head를 줄인다. 넷째, “MQA cache는 TP만큼 자동으로 작아진다”는 말은 replication을 누락한다. rank-local과 cluster aggregate를 계산한다.

다섯째, “MLA는 KV head 하나인 MQA다”는 말은 latent rank와 decoupled rotary component를 잃는다. 여섯째, “sliding attention이면 cache도 window만큼만 쓴다”는 말은 implementation crop 여부와 mixed global layer를 누락한다.

일곱째, “prefix hit면 계산과 메모리가 모두 절약된다”는 말은 partial page, layer family, model-specific state, unique physical ownership을 누락한다. 여덟째, “FP8 cache는 메모리가 정확히 절반”이라는 말은 scale·metadata·slack·workspace와 fallback을 누락한다.

교정 문장은 늘 조건을 포함한다. 예를 들어 “backend가 원본 KV head를 cache하고 scale overhead가 작으며 TP replication이 같다면 BF16에서 FP8 data portion은 절반이 된다”라고 쓴다. 조건을 붙이는 것은 설명을 약하게 만드는 것이 아니라 어떤 관측이 결론을 바꾸는지 보여 준다.

**사례 J: 완성 조건을 판정한다**

RoPE 판정은 작은 2D 계산, pair layout, rotary dimension, base/scaling, prefill/decode continuity, sliding offset, mRoPE auxiliary state를 모두 설명해야 통과다. position ID와 physical slot을 같은 열에 쓰면 미통과다.

GQA 판정은 Hq→Hkv mapping, dense byte 식, TP shard/replication, quant scale overhead를 숫자로 계산해야 통과다. “KV가 줄어든다”만 쓰면 미통과다. MLA 판정은 latent와 rotary component를 분리하고 optimized ABI와 reference materialization을 구분해야 통과다.

model 사례 판정은 Gemma local/global layer별 resident length와 Qwen multimodal/hybrid state를 일반 법칙과 구분해야 한다. 모든 layer에 같은 cache tuple을 가정하면 미통과다. source 판정은 Transformers·vLLM·SGLang·llama.cpp의 의미 owner를 fixed revision file과 span으로 연결해야 한다.

운영 판정은 position off-by-one, rotation layout, TP replication, prefix identity, speculative rollback, MLA fallback incident마다 증상→관측→분기→원인→검증을 닫아야 한다. dashboard 숫자 하나로 원인을 확정하면 미통과다. cache data를 실행 허가 없이 덤프하거나 사용자 입력을 노출하는 절차도 미통과다.

이 원장을 통과하면 독자는 cache를 단순 메모리 절약 장치로 보지 않는다. position semantics를 다음 step으로 운반하고, model architecture의 head/latent/state shape를 물리 tier에 보존하며, scheduler의 commit/rollback과 allocator의 lifetime을 연결하는 시스템으로 본다. 이 관점이 있어야 33~39장의 eviction·reuse·offload·분산 cache를 정확히 읽을 수 있다.

## 14.10 종합 사건: 32K에서만 느리고 두 번째 decode부터 간헐 오답이다

마지막으로 여러 축이 동시에 얽힌 사건을 순서대로 푼다. 서비스는 Qwen 계열 multimodal model을 TP=4로 제공한다. 8K 이하 text 요청은 정상이다. 32K image 요청은 첫 token이 대체로 맞지만 두 번째 decode부터 간헐적으로 다른 token을 낸다. 배포 변경에는 long-context RoPE override, cache FP8, attention backend upgrade, block size 16→32가 함께 들어갔다. latency도 30% 늘었다.

이 상태에서 가장 먼저 할 일은 네 변경을 한꺼번에 설명하는 이야기를 만드는 것이 아니다. 재현 identity를 고정하고 correctness와 performance 증상을 분리한다. artifact와 tokenizer/processor revision, rendered IDs, image grid, mRoPE position/delta, model config, TP, cache dtype, requested/effective backend, block size를 기록한다. 첫 token이 “대체로” 맞는다는 말은 fixture마다 exact 여부를 다시 분류한다.

### 1단계: input과 prefill의 마지막 의미점을 고정한다

text-only short, text-only 32K, image short, image 32K의 네 cohort를 만든다. 같은 semantic prompt를 억지로 쓰기보다 각 cohort에서 고정 token/grid fixture를 둔다. embedding splice 뒤 row identity, 첫 layer와 중간·마지막 layer residual, final hidden, selected first token을 비교하도록 설계한다.

image 32K도 prefill final hidden과 첫 token이 reference와 같다면 processor·weight·prefill attention의 큰 오류 가능성은 낮아진다. 그러나 scaling/layout의 미세 차이가 sampling에서 우연히 같은 token을 냈을 수 있으므로 rotated Q/K bounded slice와 layer digest를 확인한다. 첫 token 문자열만 parity 증거로 쓰지 않는다.

text 32K는 맞고 image 32K prefill부터 다르면 mRoPE axis, section layout, grid/placeholder를 본다. text와 image 모두 original max position을 넘을 때 다르면 long-context scaling과 backend를 본다. short image도 다르면 길이보다 multimodal processor와 mRoPE 기본 경로가 우선이다. cohort는 가설을 가르는 장치다.

### 2단계: 첫 decode handoff를 다섯 상태로 나눈다

prefill이 만든 상태를 `selected token`, `logical committed length`, `next text/mRoPE position`, `layer별 cache rows`, `physical block table`로 나눈다. 첫 decode input embedding이 맞는지, raw Q/K가 맞는지, rotated Q/K가 맞는지, cache read를 logical order로 재구성했을 때 맞는지, attention output이 맞는지 coarse-to-fine으로 간다.

raw Q/K까지 같고 rotated 값부터 다르면 saved rope delta 또는 next position, pair section, effective scaling이 원인 후보다. rotated new K가 같고 cached rows가 다르면 cache write/read와 block table을 본다. logical cached values도 같은데 backend output만 다르면 mask offset, FP8 dequant scale, backend numeric/ABI를 본다.

“두 번째 decode부터”가 정확히 무엇인지 명시한다. prefill 뒤 생성 token 1을 계산하는 호출을 decode step 1로 부를지, 첫 token을 반환한 뒤 다음 호출을 step 2로 부를지 팀마다 다르다. ledger는 generated-token index와 model invocation index를 모두 적는다. 명칭 혼동으로 off-by-one을 다시 만들지 않는다.

### 3단계: 네 변경을 한 축씩 반증한다

long-context scaling control은 같은 backend/cache dtype/block size에서 old와 new RoPE parameter만 비교한다. cache identity를 공유하지 않고 cold run으로 시작한다. prefill cos/sin과 decode next position을 확인한다. old가 맞고 new가 틀리면 scaling 지원 또는 identity가 살아 있다. 둘 다 맞으면 다른 축으로 간다.

FP8 control은 같은 scaling/backend/block에서 BF16 cache와 비교한다. raw projection은 같고 cache write 뒤 read/dequant부터 다르면 scale granularity, K/V scale order, backend 지원을 본다. 오차가 허용 범위인데 sampling만 갈리면 logit margin과 품질 contract도 기록한다. non-finite나 특정 block의 큰 오차는 단순 quantization noise로 넘기지 않는다.

backend control은 effective backend를 확인한 eager/reference와 fused path를 비교한다. 요청 이름이 아니라 실제 symbol과 eligibility result를 쓴다. mRoPE와 FP8 조합이 unsupported여서 fallback하면 correctness는 맞고 latency가 늘 수 있다. 지원한다고 잘못 선택되어 layout이 틀리면 correctness가 깨질 수 있다. 두 증상을 한 원인으로 단정하지 않는다.

block size control은 logical values가 같은지 먼저 본다. 16에서는 맞고 32에서 block boundary 직후만 틀리면 slot mapping, last-page valid count, block table stride를 본다. 32에서 latency가 늘었다면 slack 증가, page table 감소, kernel specialization 변화를 분리한다. 평균 sequence length 분포 없이 block size 효과 방향을 일반화하지 않는다.

### 4단계: TP replication을 숨은 다섯 번째 축으로 확인한다

배포가 TP=4였다고 해서 변경 전후 local KV mapping이 같다고 보장할 수 없다. backend upgrade가 local head layout이나 cache ABI를 바꿀 수 있다. model의 Hkv와 TP를 적고 rank별 local head와 replica group을 복원한다. 각 global KV head가 어느 rank storage에 있는지 표를 만든다.

rank 0~3의 replicated K가 write 직후 같아야 하는 설계인데 한 rank만 scale이 다르면 그 rank에서 attention output이 갈리고 collective 뒤 모두에게 퍼질 수 있다. 반대로 shard여야 할 head가 중복 mapping되면 일부 query group이 잘못된 K/V를 읽는다. global head order로 재조립한 digest를 reference와 비교한다.

TP=1 control이 맞고 TP=4만 틀리면 replication/shard/backend parallel path 가능성이 커진다. 그러나 TP 변경은 batch capacity와 scheduler shape도 바꾸므로 performance 비교에는 동일 token composition을 고정한다. correctness의 teacher-forced fixture와 서비스 latency experiment를 분리한다.

### 5단계: cache commit과 rollback을 요청 lifetime에 붙인다

간헐성은 cancellation, speculative reject, prefix reuse가 있을 때 강해질 수 있다. 단독 fresh request에서는 맞고 이전 image request가 취소된 뒤만 틀리는지 cohort를 만든다. old slot의 mRoPE delta, FP8 scale, block refcount가 새 request incarnation으로 새어 들어오는지 본다.

FP8 data block을 재사용할 때 scale metadata도 함께 overwrite되어야 한다. data는 새 값인데 scale은 이전 block 값이면 read 결과가 틀린다. prefix shared block은 immutable이어야 하며 partial last block extend는 COW해야 한다. block free가 data와 scale metadata의 lifetime을 동일하게 관리하는지 source state transition을 읽는다.

speculative path가 있다면 accepted count와 next mRoPE position, committed KV/scale rows가 같은 count만큼 진행하는지 확인한다. reject된 rows가 allocator에는 남아도 read-visible하지 않으면 괜찮다. logical length 또는 mask가 이를 포함하면 ghost context가 된다. cancellation 뒤 async write completion이 free/reallocated block을 덮지 않도록 incarnation/event를 확인한다.

### 6단계: 성능 저하를 saved-work와 extra-work로 쪼갠다

32K에서 latency가 30% 늘었다는 사실을 cache capacity 문제로 바로 연결하지 않는다. per-step query rows, KV resident/read rows, selected backend, reconstructed temporary, dequant kernel, collective와 page-table preparation 시간을 나눈다. cold/warm prefix와 prefill/decode phase도 분리한다.

long-context scaling 계산 자체의 cos/sin 비용이 늘었는지, backend fallback이 score materialization을 만들었는지, FP8 dequant가 추가되었는지, block size가 read locality/kernel specialization을 바꿨는지 경쟁 가설을 둔다. cache FP8로 HBM read가 줄어도 dequant와 fallback 때문에 느릴 수 있다. saved byte와 end-to-end latency는 동일 metric이 아니다.

MLA model이었다면 effective latent kernel과 expanded fallback을 확인해야 하지만 이 사건의 Qwen GQA/hybrid model에 MLA 식을 억지로 적용하지 않는다. hybrid recurrent layer 시간과 state도 별도다. 한 장에서 배운 모든 기술을 모든 모델에 동시에 적용하는 것이 아니라 config가 선택한 layer family만 사용한다.

### 7단계: 결론과 수정 검증을 분리한다

가령 조사 결과 correctness 원인은 image decode에서 rope delta가 request migration 중 누락된 것이고, latency 원인은 FP8+mRoPE 조합의 fused backend 미지원 fallback이었다고 하자. 하나의 배포 변경에서 두 root cause가 나온다. block size와 TP replication은 경쟁 가설 control에서 정상이라는 negative evidence로 남긴다.

rope delta 수정은 image short/long, prefill→여러 decode step, request migration, cancellation/reuse에서 position과 output parity를 검증한다. backend 지원 수정 또는 configuration rollback은 effective symbol, latency phase, cache bytes, correctness를 다시 본다. fallback 경고를 숨기는 것으로 성능 문제가 해결되지는 않는다.

rollback 기준도 사전에 둔다. cross-request state contamination, non-finite, position discontinuity는 즉시 안전 rollback 대상이다. 성능 fallback은 SLO와 capacity 영향에 따라 판단하되 correctness fix와 묶어 무리하게 배포하지 않는다. 수정 하나가 두 증상을 모두 해결할 것이라고 기대하지 않는다.

issue evidence에는 고정 revision, model/cache/RoPE config, exact input IDs와 공개 가능한 synthetic image/grid, mRoPE state, step별 logical/physical length, rank mapping, effective backend, first divergence와 negative controls를 넣는다. 거대한 사용자 K/V dump는 넣지 않는다. small basis fixture와 bounded digest로 layout과 position을 증명한다.

### 이 사건이 남기는 조사 순서

첫째 입력과 logical position을 고정한다. 둘째 raw projection과 rotary output을 나눈다. 셋째 logical cache data와 physical slot mapping을 나눈다. 넷째 model config가 요구하는 GQA/MLA/local/hybrid shape와 rank storage를 계산한다. 다섯째 requested backend와 effective backend를 나눈다. 여섯째 commit/rollback과 incarnation을 확인한다. 마지막에 latency와 capacity 효과를 workload별로 측정한다.

이 순서는 cache 문제를 무조건 model에서 allocator로 내려가는 직선으로 만들지 않는다. first divergence와 metric phase에 따라 owner를 선택한다. raw Q부터 틀리면 cache manager로 갈 이유가 없고, logical values는 맞는데 physical read만 틀리면 RoPE 수식을 다시 증명할 이유가 없다. evidence가 조사 비용을 줄이는 방식이다.

장말 invariant는 한 문장으로 압축할 수 있다. **같은 request incarnation의 모든 consumer는 같은 logical position과 committed length에 합의해야 하며, model-specific K/V 또는 latent/state shape가 rank와 physical slot을 오가더라도 그 의미가 보존되어야 한다.** 이 문장을 position 원장, byte 원장, lifetime 원장 세 개로 증명할 수 있을 때 cache 설명이 닫힌다.

### 코드 리뷰에서 이 invariant를 실제로 확인하는 순서

첫 출발점은 model config다. `num_attention_heads`, `num_key_value_heads`, head dimension이 명시 field인지 hidden width에서 유도되는지 적는다. RoPE parameter가 단일 dict인지 local/full layer별 dict인지, rotary dimension과 mRoPE section이 어디에 있는지 확인한다. MLA model이면 `kv_lora_rank`, non-positional Q/K dimension, rotary dimension을 별도 열로 옮긴다. config field를 읽은 뒤 아직 byte를 확정하지 않는다.

다음은 attention constructor다. global Hq/Hkv가 TP world size로 어떻게 local 값이 되는지 본다. projection output size와 local head mapping, KV replication factor를 계산한다. constructor가 backend-independent logical shape를 만들고 wrapper가 backend를 선택하는지, model class 자체가 specialized implementation을 구성하는지 표시한다. validation failure가 error인지 fallback인지도 이 경계에서 시작한다.

그다음 projection forward에서 raw Q/K/V 또는 compressed KV가 어떤 packed tensor로 나오는지 본다. split offset과 reshape/stride를 기록한다. fused QKV에서는 Q width, K width, V width가 같다고 가정하지 않는다. MLA에서는 compressed latent와 rotary key component의 split을 확인한다. quantized projection이면 weight storage와 activation output dtype을 구분한다.

rotary producer로 이동해 position tensor의 source를 역추적한다. dense text에서는 cache length에서 생성될 수 있고 serving runner가 별도 position을 제공할 수도 있다. multimodal이면 processor/grid와 saved delta가 개입한다. cos/sin producer가 config scaling을 소비하는지, local/global layer type을 key로 받는지, output dtype이 무엇인지 기록한다.

apply helper 또는 fused kernel adapter에서 pair layout과 broadcast axis를 확인한다. Q와 K 모두 회전하는지, K의 rotary slice만 회전하는지, cache write가 회전 전인지 후인지 본다. 일반적으로 회전 후 K를 저장하면 과거 position의 K를 매 decode마다 다시 회전할 필요가 없다. 그러나 source가 실제로 무엇을 cache하는지 확인하지 않고 이 일반성을 구현 사실로 쓰지 않는다.

cache spec 생성으로 내려가 layer 수와 per-token bytes가 어떤 field에서 계산되는지 본다. local/global/hybrid layer가 서로 다른 spec을 갖는지, MLA latent spec이 dense KV spec과 구분되는지, cache dtype scale가 포함되는지 확인한다. spec byte와 allocator block count를 곱한 capacity가 실제 reserved pool과 맞는지 검증할 관측을 설계한다.

write path에서는 scheduler/runner token row와 slot mapping의 길이가 같은지 본다. speculative token과 padding, multimodal placeholder 중 실제 K/V를 쓰는 row를 구분한다. layer index가 cache tensor index와 동일한지 hybrid layer mapping table이 있는지 확인한다. asynchronous write라면 data와 scale metadata의 completion dependency를 함께 추적한다.

read path에서는 block table, context length, window offset, head mapping이 backend ABI로 변환되는 곳을 찾는다. inclusive/exclusive window, last-page valid count, prefix shared block order를 기록한다. MLA는 latent와 rotary component pointer/stride가 별도인지 packed인지 본다. backend wrapper가 unsupported case를 detect한 뒤 어느 fallback object를 호출하는지 끝까지 따라간다.

attention output이 residual width로 돌아온 뒤에는 10장의 checkpoint로 합류한다. cache 조사라고 해서 residual differential을 버리지 않는다. raw/rotated Q·new K/V·logical cache read·attention update의 네 경계 중 first divergence를 찾고, global-complete update가 맞으면 이후 문제를 15장의 layer update, 16장의 LM head, 17장의 logits owner로 넘긴다.

마지막으로 cleanup과 rollback path를 읽는다. normal finish, abort, preemption, speculative reject, prefix refcount decrement, worker failure가 같은 함수로 수렴하는지 별도 경로인지 표시한다. free list에 block을 돌려보내기 전에 async consumer가 끝나는지, auxiliary scale·rope delta·recurrent state도 같은 incarnation과 함께 제거되는지 본다.

이 source walk는 파일 이름을 많이 아는 시험이 아니다. config에서 유도한 logical shape가 projection, rotary, cache spec, allocator, backend ABI, cleanup을 지나도 동일 의미를 유지하는지 증명하는 과정이다. 중간에 field 이름이 바뀌어도 `(request, layer, logical position, component, rank, physical slot, incarnation)`이라는 canonical key로 다시 번역한다.

### 독자가 다음 장에서 유지할 세 개의 질문

33~39장의 cache policy를 읽을 때 첫 질문은 “무엇을 저장했는가”다. dense rotated K/V인지, quantized data와 scale인지, MLA latent와 rotary component인지, hybrid recurrent state인지 답한다. 둘째는 “누가 언제까지 소유하는가”다. request 전용인지 prefix 공유인지, prefill node에서 decode node로 이동하는지, window 밖에서 reclaim 가능한지 적는다.

셋째는 “어떤 identity가 같아야 재사용 가능한가”다. token prefix만으로 끝나지 않고 model/adapter, processor, logical positions와 RoPE/mRoPE config, dtype/scale, layer state contract, tenant policy를 포함한다. 이 세 질문이 닫히지 않으면 hit율, offload bandwidth, eviction 정책을 최적화해도 saved work의 의미를 알 수 없다.

반대로 세 질문이 닫히면 기술 이름이 바뀌어도 분석할 수 있다. paged cache, radix cache, CPU offload, NVMe tier, P/D transfer는 physical lifetime과 이동 방식을 바꾼다. model이 요구한 logical position과 component shape는 보존되어야 한다. 바로 그 불변식이 이후 구현을 비교하는 공통 기준이다.

처음의 32K 사건으로 돌아가 보면 이 구분이 왜 필요한지 선명하다. 운영자는 처음에 GPU memory 그래프와 “KV cache 사용량”만 보았다. 그래프는 cache pool이 가득 차지 않았다고 말했다. 그래서 cache는 무죄라고 생각하고 sampling과 model weight를 뒤졌다. 하지만 pool byte는 logical position 연속성이나 rope delta의 존재를 말해 주지 않는다. capacity metric이 정상이어도 semantic state는 틀릴 수 있다.

반대로 latency가 늘었다는 이유로 FP8 cache를 실패라고 부르는 것도 성급하다. raw cache data는 실제로 줄었을 수 있다. 다만 mRoPE 조합이 optimized backend 조건을 벗어나 expanded fallback이 선택되었고, saved HBM bytes보다 추가 workspace와 연산 비용이 컸을 수 있다. 한 옵션이 capacity에는 이득이고 latency에는 손해일 수 있다. 어떤 상태를 바꾸었는지 알아야 서로 모순처럼 보이는 metric을 함께 설명할 수 있다.

독자에게 필요한 습관은 모든 내부 tensor를 외우는 것이 아니다. 증상이 보이면 먼저 좌표를 묻는 것이다. 현재 token의 logical position인가, cache의 committed length인가, physical slot인가. 현재 shape는 query head인가 unique KV head인가, MLA latent인가 reconstructed temporary인가. 현재 byte는 logical charge인가 unique physical data인가, rank peak인가 cluster aggregate인가. 질문 하나가 모호한 문제를 검증 가능한 문제로 바꾼다.

이 좌표를 확보한 뒤에야 최적화의 “왜”가 기술적으로 설명된다. RoPE는 position 정보를 score에 넣되 residual width를 늘리지 않기 위해 회전을 쓴다. GQA는 query 표현력을 그대로 단정적으로 보존한다고 보장하지는 않지만 serving에서 과거 K/V 저장과 read pressure를 낮추는 구조적 선택이다. MLA는 per-head K/V를 latent로 압축해 긴 문맥 state를 줄이면서 이를 소비할 특수 계산 경로를 요구한다. page cache는 logical append를 복사 없이 관리하지만 identity·commit·fragmentation이라는 새 책임을 만든다.

최적화는 공짜 삭제가 아니라 표현의 변경이다. 줄어든 data 대신 head mapping이, latent 대신 reconstruction/backend 계약이, 공유 page 대신 refcount와 COW가, sliding crop 대신 absolute offset이 필요하다. 이 교환 관계를 원장에 쓰면 옵션을 올리고 내리는 레시피가 아니라 설계 의도와 실패 조건을 설명할 수 있다.

## 14.11 RoPE scaling 이름을 실제 주파수 producer로 번역한다

`linear`, `dynamic`, `yarn`은 “문맥을 늘리는 세 강도”가 아니다. 같은 position과 pair index를 서로
다른 inverse frequency와 attention scale로 바꾸는 별도 계약이다. Transformers 고정 revision의
[`ROPE_INIT_FUNCTIONS`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_rope_utils.py#L665-L676)는
문자열을 default·linear·dynamic·yarn·longrope·llama3 등의 계산 함수에 연결한다. Registry에 이름이
있다는 사실은 model config가 필요한 field를 모두 가졌거나 serving backend가 같은 variant를
소비한다는 뜻이 아니다.

Linear scaling의 핵심 직관은 position 또는 frequency를 factor만큼 늘여 쓰는 것이다. Factor 4라면
position 16K가 원래 frequency에서 4K에 해당하는 위상 진행을 갖게 할 수 있다. 모든 pair의 주파수에
같은 비율을 적용하므로 단순하지만 학습 범위 안의 위치까지 위상을 바꾼다. “Context 4배”가 원래
4K 구간의 logits를 bitwise 보존한다는 뜻은 아니다.

Dynamic NTK 계열은 현재 sequence length가 original maximum을 넘을 때 base 또는 inverse frequency를
다시 계산한다. Transformers의
[`dynamic_frequency_update`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_rope_utils.py#L82-L108)는
position에서 sequence length를 얻어 frequency를 갱신하고, 짧은 sequence로 돌아오면 original
frequency를 복구하는 상태를 가진다.

따라서 cos/sin table은 config digest만 아니라 effective
sequence regime과 producer generation에 의존한다. 긴 request 뒤 짧은 request가 같은 mutable
frequency state를 잘못 공유하면 cross-request contamination이 될 수 있다.

YaRN은 저주파와 고주파 pair에 동일한 처리를 하지 않고 보간·외삽 영역을 ramp로 섞으며 attention
factor도 가질 수 있다. Transformers의
[`_compute_yarn_parameters`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_rope_utils.py#L345-L490)와
validation은 factor뿐 아니라 original maximum과 beta 경계를 소비한다.

vLLM의
[`get_rope`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/rotary_embedding/__init__.py#L33-L101)는
normalized `rope_type`을 읽고, linear·dynamic·yarn 등의 concrete class로 분기한다. Parser가 dict를
받았다는 사실에서 kernel-compatible cos/sin까지 건너뛰지 않는다.

작은 fixture는 position `0, original_max-1, original_max, 2×original_max`와 rotary pair의 첫·중간·
마지막 index를 교차한다. 각 cell에 inverse frequency, cos, sin, attention factor를 저장한다. Position
0은 많은 variant가 같은 `(cos=1,sin=0)`을 내므로 parity 증거가 약하다. Original boundary와 낮은·
높은 frequency pair가 variant mismatch를 더 빨리 드러낸다.

Cache identity에는 `rope_type`, factor, original maximum, theta, rotary dimension, pair layout,
attention factor와 model/layer family를 포함한다. Config key 이름이 `type`에서 `rope_type`으로 alias된
것은 동일 의미일 수 있지만 digest를 raw JSON 문자열로 만들면 alias만으로 cache miss가 난다.
반대로 normalized semantic fields를 빠뜨리면 다른 YaRN beta나 layer별 rope parameter가 같은 key로
충돌한다. Identity는 raw spelling이 아니라 실제 frequency producer 입력을 canonicalize한다.

Reader worksheet는 세 줄이면 시작할 수 있다. 첫 줄은 raw model config와 normalized fields, 둘째는
selected producer class와 table generation, 셋째는 attention consumer와 cached K convention이다.
이 중 하나가 unknown이면 “long context가 지원된다”가 아니라 “config가 accepted됐다”, “producer가
선택됐다”, “해당 request에서 parity가 확인됐다” 가운데 마지막 confirmed boundary까지만 쓴다.

## 14.12 MLA latent cache는 byte를 줄이고 표현 복원 책임을 남긴다

MLA의 이득을 dense GQA와 같은 모델 숫자로 비교해 보자. Layer 32개, context 32,768, BF16,
query head 128, non-positional Q/K dimension 128, value dimension 128, KV latent rank 512, rotary key
dimension 64라고 하자. Full per-head K/V를 장기 저장하면 token·layer당 원소는
`128×(128+64+128)=40,960`이다. K의 rotary 64를 head마다 물리 repeat하지 않는 optimistic dense
layout이라 해도 non-positional K와 V만 `128×(128+128)=32,768` 원소이고 shared rotary 64가
추가된다.

Optimistic dense byte는 다음과 같다.

```text
32 layers × 32768 tokens × (32768 + 64) elements × 2 bytes
= 68,853,694,464 bytes ≈ 64.12 GiB
```

MLA가 token·layer마다 latent 512와 shared rotary 64만 저장하면 다음과 같다.

```text
32 × 32768 × (512 + 64) × 2
= 1,207,959,552 bytes = 1.125 GiB
```

이 fixture의 long-lived data ratio는 약 57:1이다. 이 숫자는 MLA가 모든 모델에서 57배라는 뜻이
아니다. Dense 비교의 head 수와 dimension, latent rank, layer 수, dtype, TP replication이 고정된
경우의 장부다. Dense backend가 MQA/GQA로 더 적은 KV head를 저장하면 격차는 작아지고, latent를
FP8로 저장하거나 scale을 추가하면 분모가 달라진다.

SGLang의 고정 source는 이 저장 표현을 직접 드러낸다.
[`DeepseekV2AttentionMLA`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/deepseek_v2.py#L1708-L1925)는
`kv_lora_rank + qk_rope_head_dim` 폭의 cache-facing representation을 만들고, forward 경로의

[`latent_cache` pack`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/deepseek_v2.py#L2234-L2253)은
앞 slice에 non-positional latent, 뒤 slice에 rotary key component를 둔다.

Page-major layout의
[`mla_entry_bytes`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/mem_cache/layout/page_major.py#L112-L119)는
layer 수, cache dimension과 item size를 곱해 한 physical token slot의 byte를 계산한다.

이 source walk에서 중요한 mutation은 `latent_cache[..., :Rkv]`와 `[..., Rkv:]`의 slice boundary다.
Model config, projection output, cache spec, page view와 attention backend가 모두 같은 Rkv·Dr에
합의해야 한다. 합은 576으로 맞아도 512/64 경계가 448/128로 해석되면 shape와 allocation은 정상이고
내용만 틀린다. Cache ABI에는 total width뿐 아니라 component offset, dtype, scale, pair layout과
producer generation이 들어간다.

### decode reconstruction compute는 두 경로로 나누어 센다

기준 경로는 compressed `kv_a`를 normalize한 뒤 `kv_b_proj`로 per-head non-positional K와 V를
만들 수 있다. vLLM 고정 source의
[`DeepseekV2Attention.forward`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/deepseek_v2.py#L568-L615)는

Q를 non-positional/rotary로 나누고, `kv_a_proj_with_mqa`, normalization, `kv_b_proj`, rotary 적용 뒤
K/V를 attention에 전달한다. 이는 계산 의미를 확인하는 좋은 기준이지만 optimized serving backend가
과거 모든 latent를 매 step full K/V로 materialize한다는 보편 증거는 아니다.

Full reconstruction을 단순 계산해 보자. Context N=32,768의 latent `[N,512]`를 per-head K/V output
폭 `128×(128+128)=32,768`로 투영하면 dense multiply FLOP은
`2×32768×512×32768≈1.10×10^12`, layer당 약 1.10TFLOP이다. 이 계산을 32 layer와 매 decode
token마다 수행하는 경로는 cache byte 절감을 압도할 수 있다. 따라서 optimized MLA는 weight
absorption, query-side transformation이나 latent-aware attention kernel로 full historical K/V
materialization을 피하려 한다.

반면 새 token 하나의 latent를 만들고 K/V projection을 하는 비용은 M=1이라 약
`2×1×512×32768≈33.6MFLOP`이다. 과거 N 전체 reconstruction과 새 row projection은 약 32,768배
차이다. Profiler에서 `kv_b_proj`가 보였다는 사실만으로 어느 쪽인지 알 수 없다. Input M이 1인지
N인지, output temporary shape가 `[1,heads,...]`인지 `[N,heads,...]`인지 확인해야 한다.

Latent-aware 경로도 공짜가 아니다. Decode query를 absorbed weight로 변환하고, N개의 latent row와
score를 계산하며, rotary component를 별도로 결합하고 value 결과를 원래 head space로 되돌린다.
Long-lived cache byte는 줄어도 projection weight, workspace, latent read와 specialized kernel
instruction이 남는다. 비용 장부는 `persistent cache`, `per-step reconstructed temporary`, `projection
FLOP`, `latent attention FLOP`, `workspace peak` 다섯 행으로 분리한다.

### TP와 backend가 byte 이득의 실제 소유자를 바꾼다

Latent가 rank마다 replicate되면 rank-local 1.125GiB가 TP 수만큼 cluster에 존재할 수 있다. 반대로
latent dimension을 shard하면 score 또는 output을 합치는 collective가 필요할 수 있다. Shared rotary
component도 모든 query head·rank가 소비하므로 replication 방식이 다를 수 있다. Global formula를
TP로 무조건 나누지 않고 constructor의 local rank와 cache spec을 읽는다.

Backend selector가 latent-aware kernel을 선택하지 못하면 세 상태가 가능하다. 명시 오류로 배포를
막거나, reference-like expanded path로 fallback하거나, 다른 supported dtype/layout으로 conversion한다.
첫 상태는 capacity 계산 전에 드러나고, 둘째는 persistent cache는 작지만 step peak와 duration이
커지며, 셋째는 conversion buffer와 accuracy contract가 추가된다. “MLA cache가 1.125GiB”라는 수학만
으로 실제 peak memory와 ITL을 예측하지 않는다.

독자 fixture는 context 1K·8K·32K에서 persistent allocated byte, temporary peak, selected symbol과
attention duration을 함께 기록한다. Persistent byte만 N에 선형이고 temporary가 작으면 latent-aware
후보다. Temporary가 dense head width×N으로 늘면 expanded fallback 가설이 강하다. Duration은 늘지만
temporary가 작다면 latent read·score, launch, collective나 kernel efficiency를 분기한다.

## 14.13 mixed-generation prefix가 만든 조용한 wrong answer

서비스는 같은 model weight와 tokenizer를 유지한 채 long-context RoPE를 linear factor 4에서 YaRN
factor 4로 바꿨다. Prefix cache key에는 model ID, token IDs와 adapter가 있었지만 normalized RoPE
producer fields는 없었다. 배포는 rolling 방식이었고 old worker가 만든 prefix page를 shared cache에
남긴 채 new worker가 읽었다. Short prompt와 cold request는 정상이었지만 8K를 넘는 공통 system
prompt의 warm hit에서만 첫 decode logit이 간헐적으로 달랐다.

처음 dashboard는 cache hit가 높고 memory error가 없으므로 sampling nondeterminism을 의심했다.
하지만 teacher-forced fixture에서 동일 next token을 입력해도 first layer attention output부터
reference와 달랐다. Raw hidden과 Q projection은 같고 새 query에 적용한 cos/sin도 new YaRN reference와
같았다. 최초 불일치는 cache에서 읽은 prefix K의 rotary slice였다. Non-positional latent slice는
일치했고 rotary slice만 old linear producer 결과와 같았다.

사건을 다섯 generation으로 쪼갰다.

| generation | owner | 사건의 값 |
|---|---|---|
| artifact | model·tokenizer·adapter | old/new 동일 |
| rope producer | normalized type·factor·theta·beta | linear와 YaRN 상이 |
| cache entry | page header와 component offsets | producer generation 누락 |
| request | selected worker·effective config | new YaRN |
| physical page | old write incarnation | linear K rotary slice |

Token IDs가 같다는 사실은 latent projection 입력이 같다는 강한 단서지만 rotated K의 identity까지
보장하지 않는다. Old와 new 모두 width 576, BF16이고 slice 512/64가 같아 allocator·kernel ABI와
memory checker는 정상이다. 오답은 out-of-bounds가 아니라 **같은 shape에 다른 의미의 값**을 넣은
semantic compatibility 실패다.

경쟁 가설을 한 축씩 반증했다. Cache를 cold miss로 강제하면 new worker의 output은 reference와
일치했다. Old worker가 old page를 읽어도 일치했다. New worker가 old page를 읽을 때만 갈렸다.
Sampling을 greedy로 고정해도 logit divergence는 남아 RNG 가설을 기각했다. Page copy와 block size를
바꿔도 같은 old/new 조합에서 재현돼 physical address 가설도 내렸다.

RoPE variant mismatch를 더 직접적으로 확인하려고 prefix position `original_max-1`, `original_max`,
`2×original_max`의 첫·중간·마지막 rotary pair digest를 저장했다. Position 0 digest는 두 variant가
같아 control로 약했지만 original boundary 바깥에서 old page가 linear curve와 일치했다. Full K/V를
dump하지 않고 3 positions×3 pairs의 bounded fixture로 producer provenance를 식별했다.

수정은 cache key에 raw config JSON을 통째로 붙이는 방식이 아니었다. Alias `type`/`rope_type`, field
순서와 default spelling이 달라도 같은 producer라면 hit할 수 있도록 normalized semantic record를
만들었다. Record에는 producer class, factor, theta, original maximum, rotary dimension, pair layout,
attention factor, variant-specific beta/scale와 layer family를 넣고 digest했다. Page header에는 이
digest와 component ABI generation을 기록했다.

Reader는 lookup 때 request의 effective producer digest와 page header를 비교한다. 다르면 miss로
처리하고 old page의 refcount를 건드리지 않은 채 new representation을 계산한다. Rolling upgrade는
old generation을 drain하고 TTL이 지난 뒤 제거한다. Key namespace만 바꾸고 page header validation을
생략하면 잘못 라우팅된 page나 metadata corruption을 잡지 못하므로 producer와 consumer 양쪽에서
검증한다.

### partial reuse는 latent만 살리는 최적화인가

이론적으로 non-positional latent가 model weight·input·normalization과 dtype까지 같다면 old latent를
재사용하고 rotary component만 new producer로 다시 만들 수 있어 보인다. 그러나 rotary K는 hidden에서
나온 unrotated component 또는 이를 다시 만들 input state가 필요하다. Cache가 rotated `k_pe`만
보존하고 unrotated source를 버렸다면 token IDs만으로 cheap rewrite할 수 없고 해당 prefix forward를
다시 계산해야 한다.

또한 latent와 rotary component를 한 page row로 묶은 ABI에서 절반만 교체하면 shared readers,
COW와 commit ordering이 추가된다. New rotary slice를 쓰는 동안 old request가 같은 page를 읽으면
한 physical row가 두 semantic generation을 동시에 만족할 수 없다. Partial migration을 지원하려면
immutable source component, destination generation, atomic visibility와 rollback protocol이 명시돼야
한다. 그렇지 않으면 safe miss가 올바른 선택이다.

이 사건에서는 partial rewrite를 구현하지 않았다. Cold recompute의 비용이 일시적이고, silent wrong
answer의 위험이 더 컸다. Cache identity mismatch는 miss counter와 reason `rope_generation`으로
노출하고, expected cold capacity와 TTFT를 rollout budget에 넣었다. Hit율 하락은 correctness fix의
예상 비용이지 새 성능 장애로 오진하지 않았다.

### incident terminal은 output parity만으로 닫지 않는다

수정 검증은 old→old, new→new, old→new, new→old의 producer/consumer matrix를 사용했다. Same
generation 두 cell은 hit와 output parity, mixed 두 cell은 explicit miss와 recompute 후 parity를
요구했다. Linear·dynamic·YaRN variant, original boundary의 안팎, prefill chunk, 첫 세 decode step,
TP rank와 prefix partial block을 교차했다.

Rollback fixture는 lookup 뒤 request cancel, page refcount decrement와 new generation allocation을
포함했다. Mismatch page가 hit count나 logical committed length를 늘리지 않아야 하고, async producer가
취소된 destination page를 뒤늦게 visible하게 만들지 않아야 한다. Output이 맞아도 refcount leak이나
mixed page visibility가 남으면 사건은 종료되지 않는다.

Canary terminal은 wrong-answer 0만 보지 않았다. Generation별 hit/miss reason, cold recompute TTFT,
page population, old generation drain, output/checkpoint parity와 stale header rejection을 함께 봤다.
Old page가 0이 되고 mixed-generation lookup이 모두 명시 miss이며 new-only hit에서 latency guardrail을
통과한 뒤 rollout을 완료했다.

## 14.14 canonical 표를 mixed-generation 사건에 다시 적용한다

이 장의 최종 설명은 “KV cache 크기” 한 칸이 아니다. 14.8의 canonical key로 position,
representation, physical lifetime을 연결한다. 먼저 model artifact,
normalized RoPE producer, Hq/Hkv 또는 MLA fields, TP rank, cache dtype·scale, backend와 page ABI를
적는다. Raw config와 effective normalized field를 나란히 두어 alias와 default resolution을 숨기지
않는다.

Mixed-generation 사건의 position 행은 다음 값으로 채운다.

| 사건 | logical start | executed | accepted | committed after | resident offset | physical slots |
|---|---:|---:|---:|---:|---:|---|
| prefill chunk 1 | 0 | 1024 | 1024 | 1024 | 0 | pages 4–67 |
| prefill chunk 2 | 1024 | 1024 | 1024 | 2048 | 0 | pages 68–131 |
| speculative verify | 2048 | 4 | 2 | 2050 | 0 | staging 9–12→9–10 |
| sliding decode | 10000 | 1 | 1 | 10001 | 5905 | ring slot 1808 |

`executed`와 `committed`를 같게 쓰지 않는다. Speculative reject, cancel과 failed chunk는 physical work를
수행했어도 semantic length를 진행시키지 않을 수 있다. Resident offset은 sliding/local layer가
physical row 0을 어느 absolute position으로 해석하는지 보존한다. RoPE producer는 logical position을,
mask/cache backend는 committed length·resident offset·slot을 각각 소비한다.

같은 표의 representation 열에는 Dense GQA의 layer, unique KV head, head dimension, K/V와 dtype을
쓴다. MLA는 latent rank, rotary component dimension, component offset·dtype·scale와 reconstructed
temporary를 쓴다. Hybrid layer는 recurrent state family를 별도 행으로 둔다. 모든 것을 `KV bytes`로
합산하기 전에 의미와 owner를 보존한다.

```text
persistent_data = unique_physical_rows × per_row_component_bytes
page_slack = reserved_rows - committed_unique_rows
rank_peak = persistent + scale/metadata + temporary + workspace
cluster_unique ≠ sum(rank_peak) when prefix or rank replicas exist
```

수치 검산은 세 관점으로 한다. Logical charge는 request별 committed row를 합쳐 tenant가 요구한 state를
말한다. Unique physical은 prefix sharing 뒤 실제 data capacity를 말한다. Rank peak는 OOM을 설명한다.
Shared prefix를 request마다 합산하면 capacity를 과대평가하지만 chargeback에는 의도된 값일 수 있다.
TP replica를 unique global로 한 번만 세면 rank OOM을 과소평가한다.

Lifetime 열의 consumer walk는 config constructor에서 global field와 local rank를 얻고, projection에서
raw Q/K/V 또는 latent split을 확인하고, rotary producer에서 effective frequency와 pair layout을,
cache spec에서 per-row byte를, write/read adapter에서 component offset과 slot을, attention kernel에서
selected ABI를, cleanup에서 generation/refcount/event를 확인한다. 각 화살표에는 state mutation과
next consumer를 하나씩 쓴다.

완료·rollback 열에는 first-divergence checkpoint를 둔다. Input hidden, raw Q, raw latent 또는 K/V, rotated Q/K,
cache write 직후 logical reconstruction, decode read, attention output, residual update 순서로 작은
basis slice와 digest를 둔다. Full tensor를 무조건 저장하지 않는다. Position boundary, page boundary,
첫·중간·마지막 head/component처럼 가설을 가르는 좌표를 고른다.

### MLA capacity 주장을 승인하는 계산 카드

Card 첫 줄에는 비교 대상을 쓴다. “Dense GQA Hkv=8”과 “MLA latent 512+rotary64”처럼 실제 alternate
representation을 명시한다. MHA 128 head와 MLA를 비교해 큰 비율을 얻고 이를 현재 GQA baseline의
절감으로 발표하지 않는다. Layer 수, N, dtype, TP replication과 page utilization을 동일하게 맞춘다.

둘째 줄은 persistent byte다. Formula와 대입 값을 binary unit로 계산하고 scale·metadata·slack을
따로 더한다. 셋째는 per-step temporary와 workspace다. Expanded fallback이면 `[N,H,D]` temporary,
latent-aware면 absorbed query·score와 backend workspace shape를 적는다. 넷째는 compute와 traffic이다.
새 row projection과 historical reconstruction을 구분하고, context sweep에서 duration·DRAM byte의
기울기를 본다.

다섯째는 selected path다. Requested MLA flag가 아니라 actual backend class, wrapper, native symbol,
input stride·component offsets와 fallback reason을 기록한다. Persistent cache가 계산식과 맞아도
expanded temporary가 생기면 “capacity 절감은 확인, latency 이득은 기각”처럼 verdict를 분리한다.

여섯째는 correctness다. Reference와 raw projection, rotated component, attention update, logits를
비교한다. Quantized latent는 저장 전후 tolerance와 scale axis를, prefix reuse는 producer generation과
page header를, TP는 global head/component 재조립을 검증한다. 잘못 계산해 작고 빠른 cache는 최적화
후보가 아니다.

### 호환성 matrix는 지원 여부보다 state 전이를 묻는다

RoPE variant×cache generation×backend×dtype 조합마다 `supported` 체크 하나만 두지 않는다. Config
validation, producer construction, cache lookup, backend selection, write, read, cleanup terminal을
각각 둔다. Config가 accepted돼도 old cache를 잘못 hit할 수 있고, backend가 supported여도 특정
component dtype에서 fallback할 수 있다.

| 경계 | 정상 | 안전한 비지원 | 위험한 실패 |
|---|---|---|---|
| config | normalized producer 생성 | 명시 validation error | field silent ignore |
| lookup | generation match hit | mismatch miss | mixed generation hit |
| backend | latent-aware symbol | explicit fallback/error | wrong ABI selection |
| write | data·scale·header atomic visibility | request abort rollback | partial visible row |
| read | same component offsets·generation | stale page reject | finite wrong slice |
| cleanup | refcount/event 뒤 reclaim | generation drain | early reuse·leak |

이 matrix의 목적은 조합 수를 늘어놓는 것이 아니다. 장애 증상에서 최초로 위험한 state transition을
고르는 것이다. Wrong answer인데 validation부터 mismatch면 runtime kernel로 내려가지 않는다. Config와
lookup이 맞고 write 직후 logical reconstruction부터 다르면 projection·packing을 본다. Logical cache는
맞고 attention output만 다르면 backend ABI·numeric을 본다.

### 최종 판단문은 세 문장으로 끝낸다

첫 문장은 representation이다. “고정 model/config에서 layer당 cache row는 latent 512와 rotary 64의
BF16 576원소이며 SGLang page spec과 selected MLA backend가 같은 component offsets를 소비했다.”
둘째는 비용이다. “Context 32K에서 persistent rank byte는 계산과 일치했고 dense expanded temporary는
관측되지 않았으며 attention duration은 N sweep의 latent-aware 기울기를 보였다.” 셋째는 lifetime과
correctness다. “RoPE producer digest가 다른 prefix는 miss했고 same-generation reuse·cancel·rollback에서
checkpoint와 logits parity를 통과했다.”

어느 문장을 쓸 증거가 없으면 unknown으로 둔다. Persistent byte만 확인했으면 capacity까지만
accepted이고 reconstruction compute와 latency는 unverified다. Output parity만 확인했으면 sampled
fixture correctness이지 모든 cache generation의 안전성은 아니다. Dossier는 강한 결론을 만드는
장치가 아니라 증거보다 강한 결론이 나가지 못하게 하는 장치다.

### 작은 tensor로 component offset을 손으로 검산한다

실제 512+64는 눈으로 보기 어렵다. Fixture에서는 latent rank 4, rotary dimension 2로 줄이고 한
token row를 `[10,11,12,13 | 20,21]`로 둔다. 앞 네 값은 normalized non-positional latent, 뒤 두
값은 position을 반영한 rotary K다. Page size 4, layer 2라면 한 page envelope의 원소 수는
`4 tokens×2 layers×6=48`, BF16 byte는 96이다.

Page-major layout에서 physical token t=5는 page 1, page-local slot 1이다. Layer envelope를 interleave한
dense index 규칙이 `dense(t)=(t//4)×(4×2)+t%4`라면 dense row는 9다. Layer 0 view의 row 9와 layer 1
view의 row 9는 서로 다른 storage offset을 기준으로 같은 logical token의 각 layer component를
가리킨다. Block table entry가 단순 page 1이 아니라 layer-folded base를 기대하는 backend라면 adapter가
이를 변환해야 한다.

잘못된 reader가 split을 3+3으로 해석하면 `13`이 rotary 쪽으로 넘어가고 `20`까지 latent에 들어간다.
Total width 6, row count와 byte는 모두 맞는다. Attention output만 달라진다. 이 fixture는 memory
sanitizer보다 component digest가 필요한 이유를 보여 준다. Writer 직후 `latent=[10,11,12,13]`,
`rope=[20,21]`을 logical coordinate로 재구성해 assert한다.

Scale metadata가 있다면 fixture를 한 단계 늘린다. Latent FP8 scale `sL=0.5`, rotary BF16 no-scale,
value reconstruction weight generation `g=7`을 row header에 둔다. Reader가 scale axis를 per-row가
아니라 per-page로 해석하거나 generation 6 weight와 결합하면 data byte가 맞아도 값이 다르다.
Representation identity는 dtype 문자열뿐 아니라 scale granularity와 reconstruction weight generation을
포함한다.

### byte 절감이 capacity 증가로 이어지는 계산

Rank에서 cache pool로 24GiB를 예약하고 page utilization을 90%, metadata·scale overhead를 data의
3%로 잡자. Dense GQA가 token·layer당 `2×8×128×2=4096` byte라면 32 layer의 token당 data는
128KiB다. Overhead와 utilization을 적용한 resident token capacity는 대략
`24GiB×0.9/(128KiB×1.03)≈164,700` token이다.

MLA 576 BF16 원소는 layer당 1,152 byte, 32 layer에서 token당 36KiB다. 같은 가정이면
`24GiB×0.9/(36KiB×1.03)≈593,000` token이다. 이 비교는 약 3.6배 capacity 이득이다. 앞의 MHA-like
64GiB 비교가 57배였던 것과 다른 이유는 실제 baseline을 Hkv=8 GQA로 바꾸었기 때문이다. 어떤
dense 대상을 기준으로 삼는지가 headline ratio를 결정한다.

평균 resident length가 8K이고 prefix sharing을 무시하면 dense는 약 20 request, MLA는 약 72
request를 담는다. 그러나 concurrency를 3.6배 올리면 scheduler batch, projection M, attention query와
network output이 함께 변한다. Cache capacity 증가가 같은 ITL의 concurrency 증가를 보장하지 않는다.
Admission limit은 byte capacity, step-time budget과 SLO를 함께 소비한다.

Prefix sharing이 40%라면 unique physical token은 logical charge보다 작다. Dense와 MLA가 같은
sharing policy를 쓰더라도 page boundary, partial block과 identity mismatch rate가 달라 effective
capacity ratio가 달라질 수 있다. Rolling RoPE migration 동안 forced miss가 늘면 MLA data가 작아도
cold prefill work와 page churn이 커진다. Capacity spreadsheet에 hit율을 넣되 semantic mismatch를
억지 hit로 바꾸지 않는다.

### reconstruction 가설을 N·M sweep으로 반증한다

Expanded historical reconstruction이면 temporary byte와 projection FLOP이 context N에 비례한다.
Latent-aware attention도 N에 비례하는 read·score가 있지만 dense `[N,H,D]` temporary는 없어야 한다.
따라서 context 1K·4K·16K에서 peak allocation delta와 kernel input shape를 본다. Peak가 4배씩 늘고
temporary width가 head×K/V dimension이면 expanded 가설이 강해진다.

Decode batch M도 독립적으로 바꾼다. 새-token projection만 있다면 `kv_b_proj` input rows는 M이고,
historical expansion이면 대략 `ΣN_i`다. Batch 8, context 8K에서 profiler input M이 8인지 65,536인지
구분하면 같은 symbol 이름의 의미가 갈린다. Kernel 이름만으로 reconstruction 범위를 추정하지 않는다.

Weight absorption 경로는 precomputed absorbed weight의 artifact generation을 가진다. Adapter hot swap,
quantized weight reload나 TP reshard 뒤 absorbed weight를 재생성하지 않으면 cache latent는 맞아도
query transform부터 갈릴 수 있다. Incident matrix에 cache producer generation뿐 아니라 absorbed
weight generation을 별도 축으로 둔다. Cache를 cold로 해도 오답이 남으면 이 축이 살아 있다.

Performance terminal은 three-point slope를 남긴다. N 4K→8K→16K에서 persistent bytes, DRAM bytes,
temporary peak와 attention duration을 기록하고 선형·계단형·고정 overhead를 구분한다. 한 context의
tokens/s만으로 latent-aware 여부를 판정하지 않는다. Backend가 8K까지만 optimized이고 16K에서
split/fallback할 수도 있으므로 selected symbol과 workspace generation을 각 point에 붙인다.

### 배포 승인과 rollback을 representation 단위로 쓴다

RoPE config rollout, MLA backend upgrade와 cache dtype 변경을 한 release에 묶지 않는다. 세 변경은
producer semantics, compute path, storage representation이라는 서로 다른 generation을 바꾼다. 묶으면
wrong answer의 first divergence와 latency regression의 owner가 달라도 하나의 rollback밖에 할 수 없다.

각 rollout은 cold correctness, same-generation warm reuse, mixed-generation rejection, cancel/rollback,
context sweep과 TP matrix를 통과한다. RoPE 변경은 cache namespace drain, backend 변경은 selected
symbol·temporary peak, dtype 변경은 scale·tolerance가 중심 guardrail이다. 공통 output parity만으로
세 계약을 대체하지 않는다.

즉시 rollback 조건은 mixed-generation hit, non-finite, component offset mismatch, cross-request page
contamination이다. Latency 5% 회귀는 capacity와 SLO에 따라 판단할 수 있지만 semantic mismatch는
traffic 비율이 작아도 안전 rollback한다. Wrong answer가 관측되기 전에 stale header rejection
counter가 증가하면 예방적으로 rollout을 멈출 수 있어야 한다.

종료 뒤에는 old generation page가 0이 되었는지와 allocator reserved가 회복됐는지 확인한다. Output이
정상이어도 old pages가 refcount leak으로 남으면 다음 migration의 capacity를 갉아먹는다. 새 generation
hit율이 회복돼도 cold recompute backlog가 drain되지 않으면 TTFT tail이 남을 수 있다. Correctness,
capacity와 latency terminal을 각각 닫는다.

### 최종 회귀를 canonical 표의 한 요청으로 채운 예

Synthetic request `mla-rope-boundary-07`은 original maximum보다 17 token 긴 4,113-token prefix와
greedy decode 네 token을 가진다. Model revision, normalized YaRN fields, latent 512/rotary 64 split,
TP=4, BF16 cache, page size 16을 고정한다. Prefix 마지막 두 page가 boundary 안팎의 position을 함께
포함해 scaling과 page addressing을 동시에 자극한다.

Cold prefill에서는 positions 0…4,112, committed length 4,113, page 258의 valid count 1을 기록한다.
Layer 0·15·31에서 raw latent 네 성분, unrotated rotary 두 성분, rotated rotary 두 성분과 page read-back
digest를 reference와 비교한다. 첫 decode position은 4,113이고 physical slot은 새 page의 slot 1이
아니라 기존 partial page의 slot 1이다. Logical position과 page-local slot 숫자가 우연히 다르므로
둘을 뒤바꾼 오류가 드러난다.

Warm same-generation run은 prefix hit 뒤 cached component digest와 첫 네 decode logits가 cold run과
같아야 한다. Old linear generation page를 주입한 mixed run은 lookup에서 `rope_generation` miss가
나고, old page의 hit·refcount·committed length를 바꾸지 않아야 한다. Recompute된 new page가 visible된
뒤에만 request가 진행한다. Old page를 억지로 읽어 output divergence를 재현하는 negative control은
격리된 fixture에서만 사용한다.

MLA path 검증은 persistent byte와 temporary를 함께 본다. 4,113 token×32 layer×576×2의 logical
data는 144.6MiB이고 page slack·metadata를 더한 allocator 관측이 계산 범위에 들어야 한다. Decoder
step의 temporary에 `[4113,128,256]` full K/V가 없어야 하며 selected latent-aware symbol과 input
stride가 component ABI와 맞아야 한다. 없다면 capacity claim은 통과해도 optimized-path claim은
unsupported로 둔다.

Cancel fixture는 mixed miss recompute가 절반 진행됐을 때 request를 abort한다. Destination generation은
visible하지 않아야 하고 async write completion 뒤 page가 free list로 돌아가야 한다. 곧바로 같은
physical page를 새 request에 할당해 incarnation header, latent, rotary와 scale이 모두 overwrite됐는지
확인한다. Data만 새 값이고 generation이 old면 lookup 안전성이 깨지고, header만 새 값이고 data가
old면 더 위험하다.

Performance 결과는 cold prefill, same-generation hit와 mixed-generation safe miss를 분리한다. Safe
miss의 TTFT 증가는 migration 동안 예상된 비용이며 warm steady-state 수치에 섞지 않는다. Latent-aware
decode의 N sweep이 guardrail을 통과하고 old generation population과 recompute queue가 0이 된 뒤
incident를 종료한다.

Reviewer는 마지막으로 세 질문에 답한다. 왜 이 page를 재사용할 수 있었는가—normalized producer와
representation generation이 같기 때문이다. 왜 byte가 줄었는가—per-head dense K/V가 아니라 576폭
latent/rotary row를 저장하기 때문이다. 왜 여전히 느릴 수 있는가—reconstruction·latent attention,
fallback, collective와 migration recompute가 남기 때문이다. 세 답이 source, 계산과 사건 terminal에
각각 연결돼야 장의 “왜”가 완성된다.

실패 보고서에는 negative evidence도 남긴다. Token IDs, hidden input, raw latent와 page mapping이
같았고 rotated slice만 달랐다는 사실은 tokenizer·weight·allocator를 후순위로 내린 근거다. Cold
new→new와 warm new→new가 모두 맞았다는 사실은 YaRN 자체가 항상 틀렸다는 가설을 기각한다. Mixed
generation만 갈렸다는 교차표가 cache compatibility를 root cause로 좁힌다.

성능 root cause는 별도로 판정한다. Mixed-generation safe miss의 cold recompute는 예상 TTFT 증가지만
same-generation steady state에서도 느리면 backend path를 본다. Persistent byte가 계산대로 작고
temporary가 dense N×H×D로 커지면 expanded reconstruction fallback이다. Temporary는 작지만 duration이
N에 따라 과도하게 늘면 latent kernel, split 수, DRAM·collective를 본다. Wrong answer 수정과 latency
수정을 하나의 원인으로 묶지 않는다.

Capacity 판정도 allocation success 하나로 끝내지 않는다. Page utilization, scale·header overhead,
rank replica와 reserved fragmentation을 포함한 usable token 수를 계산한다. 새 capacity 때문에
scheduler가 더 많은 request를 admission하면 batch와 context distribution이 달라지므로 기존 ITL과
직접 비교하지 않는다. 동일 workload의 representation 절감과 늘어난 concurrency의 서비스 효과를
두 실험으로 나눈다.

새 release에서는 producer registry, required fields, MLA component offset, cache spec과 backend
eligibility를 semantic diff한다. Function line이 이동해도 이 다섯 anchor가 같으면 기존 fixture를
재사용할 수 있다. 하나가 바뀌면 그 boundary부터 downstream digest와 terminal을 다시 검증한다.

이렇게 canonical 행이 닫히면 운영자는 “RoPE factor를 바꿨다”나 “MLA라 cache가 작다”에서 멈추지 않는다.
어떤 producer가 어느 position의 rotary component를 만들었고, 어떤 latent representation이 어느
generation page에 commit됐으며, 어떤 backend가 그 row를 어떤 compute로 소비했는지 설명할 수 있다.
그 설명이 capacity, correctness와 latency가 서로 다른 방향으로 움직이는 이유를 동시에 보여 준다.

마지막 sanity check는 단위에서 시작한다. Decimal GB와 binary GiB, element와 byte, token과
token-layer row, rank-local과 cluster aggregate를 표 머리에 쓴다. 576을 byte로 착각하거나 layer
32를 두 번 곱하면 capacity가 수십 배 어긋나도 식 모양은 그럴듯해 보인다. 계산기 결과 옆에 한
token·한 layer의 1,152 byte를 먼저 적으면 규모 오류를 빨리 잡을 수 있다.

Position도 같은 방식으로 경계를 검산한다. Committed length 4,113이면 다음 scalar position은
0-origin 기준 4,113이고 마지막 cached position은 4,112다. Page size 16이면 4,112는 page 257의
slot 0이고 4,113은 slot 1이다. Logical position, page ID와 slot을 한 숫자로 축약하지 않는다.

최종 review는 source의 component split, byte 계산의 576폭, fixture의 page row와 cache header
generation이 같은 계약을 말하는지 확인한다. 세 표현 중 하나라도 다르면 성능 측정을 승인하지
않고 최초 불일치 owner로 돌아간다. 이 작은 검산이 긴 문맥에서만 드러나는 silent corruption을
배포 전에 막는 마지막 방어선이다.
