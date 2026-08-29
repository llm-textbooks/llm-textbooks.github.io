# 23장. Transformers classic generate의 단일 호출 수명

`model.generate()` 한 줄을 실행하면 model은 prompt를 받고 다음 tokens를 반복 생성해 tensor나 structured output을 반환한다. 이 한 줄 안에는 GenerationConfig 병합, input 준비, generation mode 선택, 첫 forward, cache 갱신, logits processor, stopping criteria와 output 조립이 있다. 겉보기에는 server request와 닮았지만 classic generate는 caller가 한 invocation의 loop와 state를 직접 소유한다.

이 장은 Transformers `v5.15.1`, commit `550d7b3834670483a4df436541272c055dc364bf`의 단일 text-generation 호출 하나를 함수·state 단위로 따라간다. HTTP route, global admission, continuous batch scheduler와 paged request allocator를 자동으로 제공한다고 가정하지 않는다. streamer와 `synced_gpus`, cancellation의 한계도 이 ownership 위에서 설명한다.

source만 읽고 model이나 CUDA를 실행하지 않는다. 실제 latency와 selected device kernel은 관찰 field로 남긴다. 먼저 greedy-like 대표 호출을 완주하고 beam·assisted 같은 mode catalogue는 필요한 분기에서만 언급한다.

## 23.1 generate 입구에서 effective config가 만들어진다

문제 장면부터 보자. caller는 `max_new_tokens=32`, `temperature=0`을 넘겼는데 warning이 나오거나 예상과 다른 mode가 선택됐다. model forward를 보기 전에 `generate`가 model-level config, passed GenerationConfig와 keyword arguments를 어떻게 합치는지 읽어야 한다.

공개 entry는 [`GenerationMixin.generate`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2100-L2500)다. exact signature와 긴 body의 범위는 고정 source에서 확인한다. 이 method는 input과 config를 받고 model kwargs를 준비하며 generation mode와 implementation으로 분기한다.

`generation_config`를 명시하지 않으면 model의 default generation config 또는 model config에서 유도된 state를 사용할 수 있다. call kwargs가 이를 override한다. deep copy나 update 방식과 unused kwargs 반환을 확인한다. caller가 넘긴 object가 mutate되는지, invocation-local copy인지가 concurrent calls 안전성에 영향을 준다.

config class와 validation은 [`generation/configuration_utils.py:100-700`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/configuration_utils.py#L100-L700)에서 fields, defaults와 `validate`를 읽는다. 같은 option 이름이 있어도 mode에서 의미가 없으면 warning이나 error가 날 수 있다.

`max_length`와 `max_new_tokens`는 대표적인 충돌이다. 전자는 input을 포함한 total length로 해석될 수 있고 후자는 새 output cap이다. 둘 다 있으면 precedence와 warning을 source에서 확인한다. prompt length가 달라져도 `max_new_tokens`는 같은 output budget을 주지만 `max_length`는 remaining budget이 달라진다.

temperature 0은 probability division에 0을 넣는 뜻이 아니다. sampling 여부와 processor/warper preparation에서 greedy mode로 가거나 invalid combination warning이 날 수 있다. `do_sample`과 temperature를 함께 본다. API adapter가 temperature 0을 어떤 kwargs로 바꾸는지도 외부 owner다.

unused model kwargs validation도 중요하다. typo가 silently ignored되면 실험을 잘못 해석한다. model forward signature와 prepare-input method가 소비할 kwargs를 어떻게 판정하는지 본다. custom model override는 허용 fields를 바꿀 수 있다.

대표 call state는 raw `input_ids`, caller kwargs, base config identity, effective config snapshot, model kwargs와 validation result다. config object를 run 중 바꾸지 않도록 invocation-local ownership을 확인한다.

first divergence가 effective config 이전이면 caller/adapter, config는 같고 mode부터 다르면 mode selection, mode는 같고 forward inputs가 다르면 input preparation owner다.

### config precedence 사건: max_new_tokens가 무시된 것처럼 보인다

model artifact의 generation config에는 `max_length=20`이 있고 caller는 `max_new_tokens=32`를 넘겼다고 하자. prompt 길이는 10인데 output이 10 tokens에서 끝났다. 첫 반응은 stopping criterion bug일 수 있다. 그러나 effective config와 derived length를 먼저 기록한다.

base config, explicit `generation_config`, direct kwargs의 세 층을 구분한다. direct kwargs가 invocation-local copy를 update하는지, config object에 이미 set된 값과 conflict warning을 만드는지 source에서 읽는다. `max_new_tokens`가 있으면 input length를 더해 effective max length를 계산하는 지점도 찾는다.

prompt length가 10이고 max new 32라면 expected total cap은 42다. actual effective max length가 20이면 precedence/update가 다르다. effective는 42인데 20에서 끝나면 EOS, custom criteria 또는 model-specific cap을 본다. output length만 보고 config bug를 단정하지 않는다.

config warning은 stdout noise가 아니라 contract signal이다. sampling false인데 temperature가 set됐거나 both max fields가 conflict하는 경우 option이 inactive할 수 있다. production wrapper가 warnings를 숨기면 user는 적용됐다고 오해한다. typed effective config를 observability에 안전하게 노출할 방법이 필요하다.

mutable config 공유도 사건을 만든다. thread A가 shared object의 max tokens를 바꾸는 동안 B가 generate를 호출하면 invocation snapshot 여부에 따라 race가 난다. library가 copy를 만들더라도 caller가 copy 전 concurrent mutation하면 안전을 보장하지 않는다. per-call immutable config를 권장하는 이유다.

model default config는 artifact 일부다. 같은 weights라도 generation_config JSON이 다르면 output length, EOS와 sampling defaults가 달라진다. checkpoint identity에 weight뿐 아니라 generation config를 포함한다. server adapter가 자체 defaults로 덮는 경우 owner를 분리한다.

unknown kwargs incident는 typo `max_new_token`을 생각한다. strict validation이 error를 내면 빨리 찾는다. custom model forward가 `**kwargs`를 넓게 받으면 typo가 validation을 통과하고 무시될 가능성을 source에서 확인한다. warning/error policy가 model class에 따라 달라질 수 있다.

복구 fixture는 base only, explicit config, direct kwargs, conflict, shared object reuse를 나눈다. output을 실행하지 않는 source 감사에서는 expected effective fields와 warnings를 전개한다. runtime test가 있다면 returned sequence length와 actual stop cause를 함께 본다.

### do_sample과 temperature를 분리한다

temperature는 sampling distribution을 조절하지만 mode selection은 `do_sample`에 의존할 수 있다. `do_sample=False, temperature=0.7`은 greedy loop에서 temperature가 사용되지 않을 수 있다. 사용자는 option을 넘겼지만 consumer가 없는 상태다.

반대로 `do_sample=True, temperature=0`은 invalid하거나 special handling이 필요하다. logits를 0으로 나누는 구현이어서는 안 된다. API adapter가 temperature 0을 greedy로 normalize하는 것과 direct Transformers call의 validation은 같은 contract가 아니다.

effective processor list를 보면 option이 실제 state가 되었는지 알 수 있다. temperature warper/transform이 list에 들어갔는지, selected generation mode가 sample인지 기록한다. raw config string만 보지 않는다.

seed도 config와 caller RNG state를 나눈다. `torch.manual_seed` 같은 global state를 쓰면 concurrent calls ordering이 random results를 바꿀 수 있다. explicit generator parameter가 지원되는지 source를 확인한다. deterministic server request-local RNG와 classic call global RNG를 같은 것으로 가정하지 않는다.

same output이 나왔다고 sampling config parity가 증명되는 것도 아니다. logits top1 margin이 크면 temperature 차이가 token을 바꾸지 않을 수 있다. processed scores와 mode identity를 비교한다.

## 23.2 model inputs와 cache는 첫 forward 전에 준비된다

`generate`는 input tensor를 그대로 model forward에 넣기만 하지 않는다. model input name, attention mask, position/cache state, encoder outputs, cache implementation과 device를 준비한다. decoder-only text fixture는 `input_ids[B,S]`에서 시작한다.

input preparation helpers는 [`generation/utils.py:500-900`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L500-L900)에서 `_prepare_model_inputs`, attention mask와 special token preparation symbols를 따라간다. exact lines는 고정 source에서 좁힌다.

input ID가 없고 `inputs_embeds`가 있으면 generation의 첫 step과 subsequent IDs ownership이 달라진다. model class가 이를 지원하는지, returned sequences가 어떤 IDs에서 시작하는지 확인한다. 두 inputs를 동시에 넘길 때 policy도 본다.

padding side와 attention mask는 마지막 token logits selection과 cache positions에 영향을 준다. decoder-only batched generation에서 right padding warning이 나올 수 있다. padding token과 EOS ID 관계도 attention mask inference를 바꿀 수 있다.

special token IDs는 config와 tokenizer artifact에서 일치해야 한다. `pad_token_id`가 없을 때 EOS로 fallback하는 warning이 있을 수 있다. 이 convenience가 semantic padding과 terminal을 같게 만드는 것은 아니다. effective special IDs를 기록한다.

cache preparation은 classic call 안에서 이루어진다. Dynamic, Static, Sliding 등 implementation이 config와 model capability에 따라 선택될 수 있다. cache object는 invocation state이며 layer K/V와 seen length를 가진다. server-wide paged cache allocator와 다르다.

cache class contracts는 [`cache_utils.py:1-500`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L1-L500)에서 update, sequence length와 storage를 읽는다. static cache는 capacity를 미리 할당하고 dynamic cache는 sequence 축을 늘릴 수 있다. physical shape와 logical length를 나눈다.

`use_cache=False`이면 each decode step에서 full prefix를 다시 model에 넣을 수 있다. output semantic은 같아야 하지만 compute와 model input shape가 크게 달라진다. cache가 켜졌다는 config와 model forward가 actual cache를 반환/사용했다는 사실을 연결한다.

prepare checkpoint는 input IDs/embeds, attention mask, positions/cache positions, special IDs, cache class/identity, initial length와 model kwargs다. shape와 dtype/device도 기록한다.

### cache update를 길이 3 prompt와 두 decode steps로 계산한다

batch 1, prompt IDs `[11,12,13]`, max new 2를 가정하자. 첫 iteration input IDs shape는 `[1,3]`, attention mask `[1,3]`, cache logical length는 0이다. model forward는 three query rows를 처리하고 layer별 K/V length 3을 cache에 만든다.

last logits에서 token 21을 고르면 logical sequence는 `[11,12,13,21]`이다. second forward에서 cache를 쓰는 model input은 보통 token `[21]` shape `[1,1]`이고 cache logical length 3이다. attention key length는 cached 3+current 1인 4다. cache update 뒤 length 4가 된다.

token 22를 고르면 output sequence는 length 5가 되고 max-new-2 criterion이 true다. 다음 forward를 호출하지 않는다. returned IDs에는 prompt 3과 new 2가 모두 포함될 수 있다. streamer는 prompt를 skip하는 option에 따라 only new text를 낼 수 있다.

cache off라면 second forward input은 `[11,12,13,21]` shape `[1,4]`이고 cache는 없다. attention math output이 같은 tolerance 안에 있어야 하지만 compute가 prefix를 반복한다. `use_cache` option은 output cap이 아니라 model input/state shape를 바꾼다.

attention mask도 `[1,3]`에서 `[1,4]`로 늘어나야 한다. cache position은 current token absolute position 3을 나타낸다. mask만 늘고 position이 0으로 reset되면 second step부터 logits가 다를 수 있다. first step parity가 이 incident의 특징이다.

Static cache capacity가 8이면 physical K/V sequence axis는 8일 수 있지만 logical committed length는 3→4다. unused slots 4~7은 mask가 막아야 한다. Dynamic cache는 physical length가 3→4로 늘 수 있다. cache length 하나의 word로 둘을 합치지 않는다.

sliding cache라면 physical retained length가 window에서 멈춰도 cumulative position은 계속 증가한다. model input slice와 RoPE/cache position은 absolute progress를 보존한다. classic cache object가 getter로 physical/logical values를 나누는지 읽는다.

cache object identity도 기록한다. model output이 same object를 mutate하는지 new cache를 반환하는지, kwargs update가 어느 reference를 next step에 넣는지 본다. wrong old reference를 유지하면 cache length가 늘지 않는다.

legacy tuple cache와 new Cache class conversion이 있다면 compatibility adapter가 length와 layer order를 보존하는지 확인한다. config implementation을 바꿀 때 selected cache type과 conversion warning을 기록한다.

first divergence fixture는 first forward K/V, after update length, second prepared input IDs, cache positions와 mask를 비교한다. second logits부터 틀리면 이 경계가 model weight보다 먼저다.

### inputs_embeds로 시작할 때 sequence ownership

caller가 input IDs 대신 precomputed embeddings를 넘기면 model은 first forward를 수행할 수 있지만 returned token history와 processors가 참조할 IDs가 필요하다. implementation이 placeholder/input IDs를 준비하거나 explicit IDs를 함께 요구할 수 있다.

first iteration에서 embeddings를 쓰고 later iterations는 selected token IDs를 embedding layer에 넣을 수 있다. `prepare_inputs_for_generation`이 embeddings를 first step에만 사용하도록 slicing하는지 본다. 계속 full embeds를 넣으면 cache/state가 중복된다.

attention mask length는 embedding sequence와 맞아야 한다. prompt token usage나 decoded prompt text를 embeddings만으로 복원할 수 없을 수 있다. classic library output semantics와 server accounting이 다르다.

custom model이 `inputs_embeds` generation을 지원하지 않으면 early validation이 error를 내야 한다. signature에 parameter가 있다는 사실만으로 cache slicing과 returned sequences가 올바르다고 단정하지 않는다.

## 23.3 generation mode는 서로 다른 loop ownership을 선택한다

effective config는 greedy/sample, beam, assisted, contrastive 등 generation mode를 정한다. mode는 단순 token selector가 아니라 batch expansion, candidate state, cache reorder와 output type을 바꿀 수 있다.

mode derivation은 [`GenerationConfig.get_generation_mode`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/configuration_utils.py#L700-L900)와 `generate` dispatch를 연결한다. `num_beams`, `do_sample`, constraints와 assistant model flags의 조합을 source로 판정한다.

대표 호출은 beam을 쓰지 않는 단일 sequence greedy 또는 sample path로 둔다. 이렇게 해야 prefill→one-token loop의 공통 수명이 보인다. beam path는 hypotheses와 cache reorder가 추가된다는 차이를 이후 source note로 둔다.

mode selection 뒤 logits processors와 stopping criteria를 준비하고 input batch를 `num_return_sequences` 등에 맞게 expand할 수 있다. expansion은 request 여러 개가 scheduler에 들어가는 것과 다르다. 하나의 caller invocation과 loop가 expanded rows를 함께 소유한다.

generation mode method가 external or compiled implementation으로 위임될 수도 있다. config name만 보고 actual loop symbol을 단정하지 않는다. dispatch branch와 callable identity를 고정한다.

`synced_gpus`는 distributed ranks가 한 rank의 sequence finish 때문에 loop collective ordering을 깨지 않게 끝까지 참여하게 하는 계약과 연결된다. request-level continuous batching fairness option이 아니다. 뒤 절에서 termination을 자세히 본다.

mode checkpoint는 resolved mode enum, selected callable, expanded batch/return count, processor list, criteria list와 cache reorder requirement다. mode가 달라졌다면 forward numeric을 비교하기 전에 config를 되돌아본다.

### mode incident: num_return_sequences 하나가 batch를 늘린다

caller가 하나의 prompt와 `num_return_sequences=4`, sampling을 요청했다고 하자. HTTP requests 네 개가 admission되는 것이 아니라 invocation 내부에서 input batch가 네 rows로 expand될 수 있다. model forward batch와 cache batch가 4가 된다.

GPU memory가 갑자기 늘었다고 server concurrency를 의심하면 owner가 틀리다. generation mode와 expand helper가 local tensor와 cache state를 복제/expand했다. output sequences shape도 `[4,prompt+new]`가 된다.

random streams가 rows별로 독립인지 global multinomial call에서 소비되는지 본다. one row 추가가 다른 rows output random sequence를 바꿀 수 있다. request-level deterministic semantics와 classic batch RNG semantics를 구분한다.

beam mode에서는 num beams만큼 expand하고 each step beam scores, parents와 hypotheses를 유지한다. selected token 하나씩 append하는 simple loop와 cache reorder가 추가된다. 이 장의 representative loop를 beam detail로 덮지 않지만 memory/ownership 차이는 기록한다.

mode가 external implementation으로 위임되면 streamer, output flags와 criteria 지원 범위가 다를 수 있다. configuration validation이 incompatible options를 막는지 확인한다. same `generate` entry가 same inner loop를 뜻하지 않는다.

fixture는 base batch 1과 expanded 4의 input IDs/cache shape, output row identity와 criterion results를 적는다. continuous server의 request count metric과 직접 비교하지 않는다.

## 23.4 첫 forward와 decode loop는 같은 caller가 소유한다

classic generation의 첫 forward는 prompt 전체를 처리하는 prefill이다. cache가 켜졌다면 이후 step은 보통 newest token과 past cache만 model에 넘긴다. 한 invocation의 Python loop가 forward, token selection과 termination을 반복한다.

현재 고정판의 decoder generation loop symbol은 `generate`가 선택한 implementation 안에서 찾는다. [`generation/utils.py:2800-3400`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2800-L3400)에서 no-beam loop, `prepare_inputs_for_generation`, model call, next-token logits와 state update를 잇는다. source 변화에 따라 exact method name을 확인한다.

loop 첫 iteration에서 `input_ids` length S, cache empty, attention mask length S다. model output logits는 `[B,S,V]`일 수 있고 next token decision은 last relevant position logits를 쓴다. right padding이면 last array position이 실제 token이 아닐 수 있어 warning과 mask contract가 중요하다.

next token ID를 append하면 `input_ids` logical history는 S+1이 된다. cache가 있다면 next model input은 appended token one row와 past K/V다. cache가 없으면 full S+1 history가 다시 들어간다. `prepare_inputs_for_generation` override가 slicing과 cache positions를 소유한다.

model kwargs update helper는 cache, attention mask, token type IDs와 positions를 step마다 갱신한다. [`_update_model_kwargs_for_generation`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L900-L1100)을 caller loop와 잇는다.

prefill과 decode를 명시적으로 scheduler phase로 등록하는 continuous server와 달리 classic loop는 input/cache shape로 phase가 드러난다. 다른 requests를 중간 step에 admission해 batch를 재구성하지 않는다. invocation batch rows는 일반적으로 loop 시작에 고정된다.

한 batch row가 EOS를 먼저 만나면 unfinished mask로 이후 token을 pad하고 다른 rows가 끝날 때까지 loop가 계속될 수 있다. sequence별 finish와 invocation loop finish를 구분한다. cache와 model forward는 finished rows를 계속 포함할 수 있다.

loop state는 current input IDs, unfinished mask, model kwargs/cache, scores/logits optional history, streamer, criteria state와 step count다. 이 state를 invocation caller가 소유한다. server scheduler request map이 아니다.

첫 forward incident는 input IDs와 config가 맞는데 output이 첫 step부터 다르다. model inputs, attention mask, cache empty state와 last-logit index를 본다. second step부터 다르면 cache update, position/mask extension과 prepare override를 본다.

### early-finish batch 사건: A는 끝났는데 forward는 계속된다

batch에는 A와 B prompts가 있다. 첫 decode에서 A는 EOS를 고르고 B는 ordinary token을 고른다. A의 unfinished flag는 false, B는 true가 된다. invocation loop는 B 때문에 계속된다.

다음 step에서 A row에 pad token을 append할 수 있다. model forward input에는 A row가 여전히 존재할 수 있고 cache도 batch dimension을 유지한다. A가 새 semantic tokens를 생성하는 것은 아니지만 compute row가 완전히 제거된다고 가정하지 않는다.

returned A sequence는 EOS 뒤 pads를 포함할 수 있고 decode `skip_special_tokens`가 보이지 않게 한다. output tensor rectangular shape와 semantic sequence length를 나눈다. usage를 계산하는 wrapper가 pad positions를 completion tokens로 세면 틀린다.

attention mask와 cache update가 finished row를 어떻게 처리하는지 source loop와 model implementation을 본다. fixed batch forward가 K/V를 pad row에도 쓸 수 있지만 downstream unfinished mask가 output selection을 차단한다. classic generate는 server scheduler처럼 row A slot을 새 request C로 교체하지 않는다.

이 difference는 latency에도 영향을 준다. 한 batch에 매우 긴 B가 있으면 A result 반환도 invocation end까지 기다릴 수 있다. streamer는 A finish를 per-row로 알려주는지, final tensor는 all rows 뒤에만 반환되는지 확인한다. server continuous batching과 tail ownership이 다르다.

negative fixture는 A EOS at step1, B at step3이다. each step input batch shape, unfinished mask, appended IDs, cache lengths와 returned sequences를 적는다. EOS와 pad ID가 같다면 finish detection과 padding representation을 조심한다.

bug incident는 A가 finish 후에도 non-pad sampled IDs를 output에 포함하는 경우다. raw selected next tokens가 unfinished mask로 pad-replaced되는 line을 찾는다. mask application 전/후 streamer put order도 본다. streamer가 replaced token이 아니라 raw token을 받으면 visible leak이 될 수 있다.

### prefill과 decode 사이 hidden state 보존 비용

`output_hidden_states=True`와 `return_dict_in_generate=True`를 켜면 first forward의 all prompt positions hidden states와 later step states를 tuple로 보존할 수 있다. long prompt와 many layers에서 memory가 크다.

attention outputs/scores도 마찬가지다. debug option은 execution path와 return state를 바꾼다. fused attention backend가 attentions 반환을 지원하지 않아 fallback할 수 있다. generate config option이 kernel selection까지 간접 영향을 줄 수 있다.

output histories는 caller가 returned object를 놓을 때까지 살아 있다. cache cleanup을 했는데 GPU memory가 내려오지 않는 incident에서 returned scores/hidden references를 본다. allocator reserved memory와 live tensor memory도 구분한다.

debug probe를 production call에 켜기 전에 작은 fixture에서 필요한 layer/step만 수집하는 hook을 고려한다. observer effect와 개인정보를 기록한다. source에 output collection list append가 어느 loop에 있는지 확인한다.

## 23.5 logits processors와 stopping criteria는 순서가 있는 state machine이다

model forward가 logits를 내면 next-token scores를 고르고 processors/warpers를 적용한다. repetition penalty, min length, bad words, constraints와 sampling transforms는 순서에 따라 결과가 달라진다. raw logits와 processed scores를 분리한다.

processor construction은 [`generation/utils.py:1100-1500`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L1100-L1500)의 `_get_logits_processor`, stopping construction은 [`:1500-1750`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L1500-L1750)을 고정 source에서 확인한다.

processor는 current full `input_ids`를 볼 수 있어 repetition/history-dependent state를 계산한다. custom processor가 CPU sync나 Python work를 넣으면 decode ITL에 들어간다. thread-safe인지, shared mutable state를 갖는지 concurrent generate calls에서 중요하다.

sampling mode는 processed scores에서 probabilities를 만들고 multinomial을 쓸 수 있다. greedy는 argmax다. seed와 generator가 caller/global RNG 중 무엇을 쓰는지 확인한다. classic call 두 개를 threads에서 동시에 실행하면 random stream과 model state safety가 별도 문제다.

selected token을 append하고 streamer에 전달한 뒤 stopping criteria를 평가하는 순서를 본다. max length, EOS와 custom criteria가 어느 step에서 true가 되는지 public output length를 정한다. stop string criteria가 decoded state를 가질 수 있다.

stopping criteria는 batch rows별 bool tensor를 돌려 unfinished mask를 갱신할 수 있다. 모든 rows가 finished일 때 invocation loop가 끝난다. synced GPUs에서는 local all-finished와 global loop termination이 다르다.

scores/output attentions/hidden states를 반환하도록 요청하면 loop가 history tuples를 보존해 memory가 증가한다. return dict와 output flags가 단순 response formatting option이 아니다. long generation에서 큰 state를 만든다.

first divergence ledger는 raw next logits, processed scores, selected ID, appended history, criteria result다. selected ID부터 같은데 loop length가 다르면 criteria/unfinished mask owner다.

### processor order 사건: 같은 logits에서 다른 token을 고른다

raw logits는 identical한데 direct generate와 server wrapper output token이 다르다고 하자. wrapper가 repetition penalty와 logit bias, grammar를 다른 순서로 적용할 수 있다. raw logits parity는 model/kernel을 기각한다.

세 candidates logits `[3,2,1]`을 생각하자. token 0이 previous history에 있어 repetition penalty가 score를 낮추고 token 2에 bias +3을 주면 order가 달라진다. penalty와 bias가 서로 다른 tokens면 order가 commute할 수 있지만 normalization/top-p와 masking을 섞으면 일반적으로 commute하지 않는다.

top-k는 score rank 밖을 제거하고 top-p는 normalized cumulative mass를 쓴다. temperature 전에/후 top-k rank는 같을 수 있지만 top-p set은 달라질 수 있다. grammar mask를 sampling 뒤 적용할 수는 없다. source processor list의 construction order와 loop call을 본다.

custom processor가 in-place로 scores를 mutate하는지 new tensor를 반환하는지, shared object가 step state를 갖는지 확인한다. same processor instance를 concurrent generate calls에 재사용하면 history가 섞일 수 있다.

logprobs를 wrapper가 반환할 때 raw logits, processed scores와 sampling distribution 중 어느 것을 썼는지 명확히 한다. generate output scores option이 어느 stage tensor를 보존하는지 source docstring과 code를 확인한다.

fixture는 raw logits, after-each-processor snapshots, selected token을 small vocabulary로 고정한다. 실행 없는 source 감사에서는 order와 consumer를 증명하고 numeric values는 설명 예시로 표시한다.

### stopping criterion의 off-by-one을 step timeline으로 본다

prompt length 3, max new 2에서 first new token을 append한 뒤 total length 4다. criterion은 false다. second를 append해 length 5가 되면 true다. criterion을 token selection 전에 평가하면 only one token을 만들 수 있다.

EOS도 selected token append와 unfinished update 순서를 가진다. returned sequence에 EOS가 포함되는지, streamer가 EOS ID를 받는지, decode가 skip하는지 나눈다. terminal cause와 visible text는 다르다.

custom criterion이 wall-clock deadline이나 external cancel event를 본다면 evaluation granularity는 token steps다. long model forward 중간에는 interrupt하지 못하고 forward 뒤 criterion에서 멈춘다. cancellation latency bound는 step duration을 포함한다.

batch criteria가 row별 bool을 반환할 때 shape와 device가 맞아야 한다. scalar bool legacy behavior와 per-row tensor contract가 version마다 다를 수 있다. custom implementation은 고정 version signature를 따른다.

synced ranks에서는 local criteria true라도 global peers가 끝날 때까지 loop control이 계속될 수 있다. semantic tokens append를 멈추고 collective participation만 유지하는 branch를 본다.

## 23.6 streamer와 cancellation은 HTTP lifecycle을 자동 제공하지 않는다

streamer를 넘기면 generated token IDs 또는 decoded text가 callback/queue로 전달된다. 이는 user가 final tensor 전에 intermediate output을 소비하게 하지만 OpenAI SSE, request abort와 resource scheduler를 자동 제공하지 않는다.

streamer base와 implementations는 [`generation/streamers.py:1-320`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/streamers.py#L1-L320)에서 `put`, token cache, text finalization과 `end`를 읽는다. `TextIteratorStreamer`는 queue를 consumer iterator에 연결한다.

`generate`는 보통 synchronous call이다. UI/server wrapper가 separate thread에서 generate를 실행하고 main thread에서 streamer를 iterate할 수 있다. worker exception이 streamer consumer에 어떻게 전달되는지 wrapper가 관리해야 한다. core streamer만으로 HTTP error object를 만들지 않는다.

slow consumer가 streamer queue를 읽지 않으면 queue가 bounded인지, `put`이 block하는지, memory가 늘어나는지 확인한다. callback이 무거우면 generation loop ITL에 직접 들어간다. continuous server의 shared output handler와 구조가 다르지만 backpressure 질문은 남는다.

cancellation은 streamer iterator를 그만 읽는 것만으로 model loop를 멈추지 않을 수 있다. stopping criteria에 external event를 넣거나 generation worker를 cooperative cancel하도록 wrapper가 설계해야 한다. Python thread를 kill하는 안전한 일반 primitive라고 가정하지 않는다.

exception이나 cancel에서도 `streamer.end()`가 호출되어 consumer가 terminal sentinel을 받아야 한다. loop의 normal cleanup과 wrapper exception handling을 잇는다. consumer가 무한히 queue wait하지 않게 한다.

stream text는 incomplete Unicode와 word-boundary buffering을 가질 수 있다. token put마다 visible character가 나온다는 보장은 없다. offline decode와 streaming chunks의 concatenation policy를 확인한다.

classic streamer는 request ID map을 기본 제공하지 않는다. 여러 calls를 동시에 service하면 wrapper가 thread/task와 streamer를 request identity로 묶어야 한다. 한 streamer를 concurrent calls에 재사용하면 outputs가 섞일 수 있다.

### streamer backpressure 사건: 생성 thread가 멈춘다

worker thread에서 `generate`를 실행하고 main thread가 `TextIteratorStreamer`를 읽는 UI를 생각하자. consumer rendering이 멈추자 GPU utilization도 끊기고 decode loop가 진행하지 않는다. model scheduler가 아니라 streamer queue put이 worker를 block했을 수 있다.

queue가 bounded이고 full이면 `put`이 기다린다. unbounded면 generate는 계속되지만 text chunks가 memory에 쌓인다. queue timeout이 있으면 exception이 worker 또는 consumer 어느 쪽에 나타나는지 확인한다. backpressure를 없애는 것이 아니라 위치와 bound를 정한다.

streamer는 token cache를 decode해 printable text boundary에서 chunk를 낼 수 있다. token 하나마다 queue item 하나가 아닐 수 있다. queue depth를 generated token count로 직접 환산하지 않는다. CJK, newline와 byte fallback에서 flush behavior가 다를 수 있다.

consumer가 iterator를 break해도 worker generate는 자동으로 알지 못할 수 있다. queue는 계속 차고 worker가 block하거나 generation을 완료한다. external cancel event를 stopping criterion이 읽도록 wrapper가 연결해야 한다.

cancel event가 set돼도 current forward가 끝난 뒤 criterion이 평가된다. long prefill 중 즉시 stop되지 않는다. thread interrupt나 CUDA context teardown으로 억지로 끊으면 model state와 distributed peers cleanup 위험이 있다. cooperative boundary를 명시한다.

first divergence timeline은 model forward end, streamer put start/end, consumer get, render를 둔다. put wait가 길면 streamer backpressure, put은 빠르고 render가 늦으면 client/UI, forward gap이면 model/loop owner다.

복구는 queue size를 늘려 UI가 다시 보인다는 것으로 끝내지 않는다. bounded memory, cancel propagation, final sentinel, worker thread join과 subsequent generation 정상까지 본다.

### streamer exception 사건: consumer는 영원히 기다린다

model forward가 exception을 던졌지만 worker thread만 stack trace를 남기고 종료했다. streamer consumer는 terminal sentinel을 받지 못해 queue get에서 기다린다. core streamer가 worker exception을 자동 전달하지 않는 path가 있을 수 있다.

wrapper는 worker function을 `try/except/finally`로 감싸 exception object나 sentinel을 queue에 넣고 thread를 join해야 한다. 정상 loop에서 `streamer.end()`를 호출하는 source와 exception path를 비교한다. exactly-once terminal이 필요하다.

consumer timeout은 hang을 피하지만 root exception을 보존해야 한다. timeout만 raise하면 model OOM, invalid input과 programmer error가 같은 모습이 된다. thread-safe exception channel을 둔다.

partial chunks를 이미 소비했으면 caller가 결과를 폐기, 표시, retry 중 무엇으로 할지 policy가 필요하다. classic streamer는 OpenAI error event contract를 정하지 않는다. wrapper/application owner다.

같은 streamer를 재사용하면 previous terminal과 cached tokens가 next call에 남을 수 있다. instance per invocation이 안전한 기본이다. reuse 지원을 주장한다면 reset method와 thread synchronization을 source로 확인한다.

### cancellation이 stopping criteria로 구현될 때

external `threading.Event` 또는 async-compatible flag를 custom criterion이 읽는다고 하자. criterion object는 loop마다 current IDs/scores를 받아 event set 여부를 batch rows bool로 반환한다. cancel target은 invocation 전체일 수 있다.

batch 중 request A만 취소하고 B는 계속하려면 row-specific mask와 output handling이 필요하다. classic batch는 independent server requests가 아니므로 application identity를 rows에 매핑해야 한다. simple scalar cancel은 all rows를 끝낸다.

criterion이 CPU event를 읽는 것은 cheap할 수 있지만 per-step callback에서 locks/IO를 하면 ITL에 들어간다. cancel check는 nonblocking이어야 한다. source loop의 criterion call 위치가 latency granularity를 정한다.

criterion true 뒤 streamer end, returned partial sequence와 cache release가 어떻게 되는지 확인한다. cancel reason이 built-in finish enum으로 반환되지 않을 수 있다. wrapper가 cancellation과 normal length를 구분하는 metadata를 추가해야 한다.

HTTP disconnect를 이 event로 연결하려면 server transport coroutine과 worker thread 사이 safe signal, request map과 join/timeout을 구현해야 한다. classic `generate` 단독 기능으로 부르지 않는다.

## 23.7 synced_gpus와 cleanup은 loop 종료 조건을 넓힌다

distributed generation에서 한 rank만 loop를 일찍 빠져나가면 다른 ranks가 다음 collective/model forward를 기다리며 hang할 수 있다. `synced_gpus`는 ranks가 모두 finished를 합의할 때까지 loop participation을 유지하는 경로다.

generation loop에서 peer-finished flag와 distributed reduction을 사용하는 branch를 [`generation/utils.py`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2800-L3400)에서 찾는다. local sequence가 finished여도 peer가 active면 loop iteration/collective order를 보존할 수 있다.

이는 tensor-parallel server scheduler의 request batching과 다르다. fixed distributed invocation의 ranks가 같은 control flow를 유지하는 문제다. new requests를 admission하거나 finished request slot을 다른 request로 채우지 않는다.

synced flag를 끄고 distributed backend가 collective를 요구하면 early exit hang이 날 수 있다. 반대로 불필요하게 켜면 finished rank가 extra iterations에 참여해 latency/work가 늘 수 있다. model/sharding environment와 generation code의 official condition을 따른다.

cleanup은 cache object와 output histories, streamer와 temporary model kwargs references를 놓는 과정이다. local Python references가 사라져도 CUDA asynchronous work 완료와 allocator reuse는 framework ordering이 소유한다. 이 장은 실행하지 않고 source lifetime만 본다.

exception이 loop 중 발생하면 streamer terminal, thread wrapper error, distributed peers와 cache cleanup이 모두 필요하다. 한 rank만 exception이면 다른 ranks가 collective에서 hang할 수 있어 distributed error propagation owner를 확인한다.

caller가 returned tensor를 보존하면 generated sequences memory는 당연히 남는다. `return_dict_in_generate`와 scores/hidden histories도 caller-owned output이다. cleanup leak과 intentional retention을 구분한다.

### synced ranks 사건: rank 0은 끝났고 rank 1은 계속한다

두 ranks가 distributed model forward에 참여한다고 하자. rank 0 local batch는 EOS를 모두 만났고 rank 1은 아직 active다. rank 0이 Python loop를 break하면 rank 1의 다음 forward collective가 상대를 기다리며 hang할 수 있다.

`synced_gpus` path는 local finished flag를 peers와 합의한다. all ranks가 finished가 아니면 rank 0도 loop control에 참여한다. 이미 local finished인 rank가 token semantic을 더 append하는지, model forward를 skip할 수 있는지 exact source branch를 본다.

global synchronization 자체도 step latency에 들어간다. 한 rank가 느리면 others가 기다린다. 이것은 continuous server scheduler fairness가 아니라 one distributed invocation의 collective ordering이다.

rank 한 곳에서 exception이 나면 normal finished flag exchange에 참여하지 못한다. other ranks가 hang하지 않도록 distributed launcher/error propagation이 process group을 abort하거나 peers에 failure를 알려야 한다. generate source만으로 전체 fault tolerance를 약속하지 않는다.

timeout을 걸어 process를 kill하면 cleanup이 어떻게 되는지 launcher owner를 본다. cache Python object cleanup보다 distributed communicator와 process lifetime이 더 넓다. 이 장은 source contract를 기록하고 NCCL 실행을 하지 않는다.

fixture는 rank 0 early EOS, rank 1 two more steps와 rank 1 exception을 논리 timeline으로 적는다. each iteration finished flag, collective participation, output append와 loop terminal을 source에서 전개한다.

### cleanup 사건: 호출이 끝났는데 memory가 그대로다

generate가 반환된 뒤 device memory가 바로 operating-system free로 보이지 않을 수 있다. framework caching allocator가 reserved blocks를 보존한다. live tensors와 allocator reservation을 구분한다.

returned sequences, scores, logits, attentions, hidden states와 cache references 중 무엇을 caller가 보존하는지 본다. `return_dict_in_generate=True`와 output flags는 큰 histories를 의도적으로 반환한다. 변수 reference를 놓기 전에는 leak이 아니다.

streamer queue에 unconsumed chunks와 token cache가 남을 수 있다. worker thread와 closure가 model outputs를 참조할 수도 있다. thread join과 queue drain, streamer instance lifetime을 본다.

static cache가 model attribute로 재사용되는 path인지 invocation local인지 확인한다. compile/static cache optimization은 state reset이 필요할 수 있다. old sequence data가 next call mask에 노출되지 않도록 logical length/reset owner를 본다.

asynchronous device work가 남았다고 무조건 synchronize를 추가하지 않는다. PyTorch stream/allocator가 dependency를 관리할 수 있다. illegal reuse evidence가 없는데 global synchronization을 넣으면 latency만 악화된다. source와 runtime trace로 lifetime을 판정한다.

memory incident fixture는 minimal returned tensor, return scores, return hidden states, slow streamer와 exception path를 비교한다. source-only로 reference graph를 적고 actual allocated bytes는 실행 승인 뒤 측정한다.

## 23.8 classic generate와 continuous server의 경계를 닫는다

classic generate는 invocation 시작에 batch와 config를 정하고 caller의 loop가 모두 끝날 때까지 소유한다. continuous server는 독립 requests를 waiting/running state로 두고 매 step batch membership을 바꾸며 KV allocator와 output streams를 장기 process가 소유한다.

classic batch row A가 먼저 finish해도 보통 invocation의 B가 끝날 때까지 fixed batch structure 안에 남는다. continuous scheduler는 A resources를 free하고 new C를 admission할 수 있다. throughput과 fairness ownership이 다르다.

classic cache는 call-local cache object이고 token sequence와 layer state를 직접 담는다. server paged cache는 request IDs, blocks와 refcounts를 process-level allocator가 관리한다. `use_cache=True`를 prefix cache나 paged reuse와 같다고 부르지 않는다.

classic cancellation은 caller/criterion/thread wrapper가 cooperative하게 구현한다. server disconnect abort는 request map→scheduler→KV cleanup lifecycle이다. streamer consumer close만으로 server-style abort가 완성되지 않는다.

대표 workbook은 input IDs 길이 3, max new tokens 2, greedy, cache true다. effective config, initial mask/cache, prefill logits, selected ID, updated cache length, second decode, criteria true와 output 반환을 source state로 적는다.

negative fixture는 config typo, right padding batch, cache off, early EOS row, slow streamer, external cancel event, synced distributed rank exception을 둔다. 실행하지 않고 expected source branch와 cleanup owner를 적는다.

소스 지도는 다음 순서로 읽는다.

- 공개 진입점은 [`generate`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2100-L2500)다.
- 입력과 캐시의 정규화는 [input/cache helpers](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L500-L1100)에서 확인한다.
- 후보 변형과 종료 판정은 [processors/criteria](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L1100-L1750)가 맡는다.
- 반복 실행은 [loop](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2800-L3400)로 내려간다.
- 상태 보존과 외부 전달은 각각 [cache](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cache_utils.py#L1-L500)와 [streamer](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/streamers.py#L1-L320)에서 닫힌다.

이 장의 출구 질문은 effective config를 누가 만들고, first/model inputs와 cache를 누가 준비하며, 어느 mode loop가 forward와 token selection을 소유하고, criteria가 언제 loop를 닫고, streamer/caller가 cancellation과 output lifetime을 어떻게 관리하는가다.

classic generate를 정확히 이해하면 이를 server scheduler의 축소판으로 오해하지 않는다. model semantics reference와 small differential에는 강하지만 continuous admission, cross-request fairness, request-local abort와 process-wide paged cache는 별도 serving layer가 필요하다.

### 대표 호출을 두 tokens 끝까지 완주한다

지금까지 나눈 state를 한 시간축에 다시 놓자. caller는 `input_ids=[[11,12,13]]`, attention mask ones, `max_new_tokens=2`, `do_sample=False`, `use_cache=True`를 넘긴다. model은 eval mode이고 EOS는 2, pad는 0이라고 하자. 숫자는 source state를 설명하는 fixture다.

generate entry는 base config를 invocation copy로 만들고 direct kwargs를 병합한다. effective max total length는 input 3+new 2=5다. mode는 no-beam greedy 계열이다. logits processor와 stopping criteria lists가 준비된다. cache implementation이 model capability와 config에서 결정된다.

input preparation은 IDs와 attention mask를 model kwargs에 둔다. cache는 empty, cache position은 0..2 prompt positions를 나타낸다. first `prepare_inputs_for_generation`은 full prompt를 model input으로 만든다.

prefill forward는 logits `[1,3,V]`와 cache length 3을 반환한다. loop는 last position logits `[1,V]`를 고른다. processors를 순서대로 적용하고 argmax token 21을 선택한다. unfinished mask는 true이므로 21을 append하고 streamer가 있다면 put한다.

model kwargs update는 returned cache를 next kwargs에 넣고 attention mask에 one column을 append한다. criteria는 current length 4가 max 5보다 작고 token 21이 EOS가 아니므로 false다.

second preparation은 cache length 3과 last token 21을 보고 model input `[1,1]`을 만든다. cache position은 3이다. forward 뒤 cache length는 4다. last logits에서 token 22를 고르고 append해 full IDs length 5가 된다.

max length criterion은 true다. unfinished sequence가 false로 갱신되거나 all-finished 판정으로 loop가 끝난다. streamer `end`가 호출되어 buffered text와 terminal을 보낸다. returned tensor는 `[11,12,13,21,22]`다. cache와 temporary kwargs는 invocation references가 사라질 때 cleanup 가능하지만 returned outputs는 caller가 소유한다.

이 timeline에서 option 효과를 읽는다. cache false면 second input이 four IDs다. max new 3이면 third loop가 있다. token 2가 first step에 나오면 EOS로 early terminal한다. sampling true면 argmax 대신 probability sampling이다. streamer는 core arithmetic을 바꾸지 않아야 하지만 callback/backpressure가 loop timing을 바꿀 수 있다.

각 단계의 evidence를 source function에 붙인다. config merge와 mode는 `generate`, input/cache는 preparation helpers, forward slicing은 model `prepare_inputs_for_generation`, kwargs update는 update helper, scores/selection과 criteria는 selected loop, text delivery는 streamer source다.

### first-step correct, second-step wrong 사건을 푼다

cache off 실행은 두 steps 모두 reference와 맞고 cache on은 first token만 맞으며 second token이 다르다고 하자. weights, tokenizer와 first forward는 강하게 기각된다. cache update와 second-step model input으로 좁힌다.

first output cache object가 correct length 3인지 본다. kwargs update가 이 object를 next call에 넣었는지 확인한다. second `prepare_inputs_for_generation`이 full IDs를 last token으로 slice했는지, cache position 3과 attention mask length 4를 만들었는지 본다.

raw second-step Q/K/V를 보기 전에 model inputs parity를 고정한다. cache off의 full prefix semantics와 cache on의 `[token21]+past`가 같은 absolute positions와 mask를 나타내야 한다. position 0 reset이나 cache length off-by-one은 shape가 맞아도 logits를 바꾼다.

cache class conversion도 가설이다. model은 legacy tuple을 반환하고 update helper는 Cache class를 기대하거나 반대일 수 있다. layer order, sequence dimension과 reorder가 보존되는지 본다. custom model implementation의 declared cache support를 확인한다.

static cache에서는 capacity 전체를 mask가 막아야 한다. mask target length가 logical 4가 아니라 capacity라면 unused slots를 additive mask로 제거한다. eager and fused attention adapter가 같은 cache offset을 쓰는지 15~16장 경계로 넘긴다.

first divergence는 cache object update, prepared input IDs, cache position, attention mask, second raw logits 순서다. raw logits만 보고 model kernel issue라고 쓰지 않는다. cache off control이 의미를 고정한다.

복구는 prompt length 1, boundary lengths, multi-batch padding과 multiple cache implementations를 본다. cache on/off selected logits tolerance를 step별로 비교한다. output text가 우연히 같아도 numeric divergence를 숨기지 않는다.

### classic call을 HTTP server로 감쌀 때 새 owner가 생긴다

많은 간단한 serving examples는 request마다 worker thread에서 `generate`를 호출하고 streamer를 HTTP response에 연결한다. 이 구조는 이해하기 쉽지만 library가 없는 lifecycle을 wrapper가 만들어야 한다.

admission owner가 필요하다. concurrent HTTP requests마다 generate thread를 무제한 만들면 model forward concurrency와 memory가 폭증한다. semaphore/queue와 reject policy를 wrapper가 소유한다. Transformers generation config에는 global max concurrent requests가 없다.

model object thread safety도 확인한다. inference forward가 read-only weights를 공유할 수 있어도 static cache, hooks, mutable generation config, RNG와 streamer를 calls 사이 공유하면 race가 생긴다. invocation-local state와 model-global state를 구분한다.

batching owner도 없다. request A decode step 사이에 B를 same batch로 자동 admission하지 않는다. wrapper가 multiple prompts를 one generate batch로 묶을 수 있지만 early-finish와 output demultiplex를 직접 관리한다. continuous batch scheduler와 다르다.

disconnect owner도 wrapper다. transport close를 external cancel criterion에 신호하고 worker completion을 join하며 stream terminal/exception을 처리한다. consumer iterator break만으로 worker가 멈추지 않는다.

usage와 finish reason도 wrapper가 raw sequences/criteria에서 만들어야 한다. loop가 왜 끝났는지 public enum으로 보존하지 않는 path가 있을 수 있다. max length, EOS와 custom cancel을 구분하는 metadata를 설계한다.

health/shutdown도 wrapper owner다. new calls admission을 막고 active threads/calls를 drain 또는 cancel하며 model/distributed process를 닫는다. classic generate API 하나로 process-wide readiness가 생기지 않는다.

이 구분은 Transformers를 serving에 쓰면 안 된다는 뜻이 아니다. model semantics reference, simple low-concurrency service, custom batch workflow에 유용하다. 다만 필요한 owner를 명시적으로 추가해야 한다.

### continuous scheduler와 비용 회계가 다른 이유

classic fixed batch에서 prompt lengths 100과 10을 padding해 함께 generate하면 prefill tensor와 mask가 rectangular할 수 있다. shorter row의 padding work가 생길 수 있다. continuous serving은 packed tokens와 ragged metadata로 padding을 줄일 수 있다.

decode에서도 classic batch rows가 fixed이고 early finished rows가 남을 수 있다. continuous scheduler는 active tokens만 compact하고 freed slots에 requests를 넣을 수 있다. batch size라는 metric의 의미가 다르다.

classic cache bytes는 invocation batch, layers, heads와 current length로 계산하며 call 종료 뒤 references를 놓는다. server paged cache는 process capacity를 preallocate하고 requests가 blocks를 공유/반납한다. model `cache_implementation=static`과 server block allocator를 같은 option으로 설명하지 않는다.

classic call latency는 caller queue가 없다면 config/input prep, prefill, fixed decode loop, streamer/output다. server latency에는 API admission, tokenizer queue, scheduler waits와 cross-request contention이 추가된다. `generate` benchmark를 server TTFT/ITL로 직접 일반화하지 않는다.

classic throughput은 invocation batch tokens/time일 수 있다. server goodput은 requests SLO, cancellation과 delivery를 포함한다. early finished pad work와 returned sequences를 어떻게 세는지 명시한다.

classic output histories를 켠 memory profile은 debug flags의 call-local state를 포함한다. server는 logprobs/output queues와 process metrics가 추가된다. memory 비교에서 scope를 맞춘다.

### output 반환 형태도 lifecycle state다

`return_dict_in_generate=False`이면 sequences tensor가 주된 반환이다. true이면 mode별 output dataclass가 sequences와 optional scores, logits, attentions, hidden states, past key values를 가질 수 있다. caller code가 expected type을 알아야 한다.

output flags가 false인데 field를 기대하면 null/absent다. enabled histories는 each step tuples로 쌓여 sequence length와 layers에 따라 커진다. response serialization wrapper가 이를 JSON으로 직접 바꾸려 하면 huge payload와 unsupported tensors 문제가 생긴다.

`past_key_values`를 반환해 caller가 continuation에 재사용할 수 있는 path가 있다면 prompt IDs와 cache position identity를 함께 보존해야 한다. cache만 다른 text request에 붙이면 state contamination이다. invocation end가 cache semantic lifetime의 끝이라고 무조건 단정하지 않되 explicit reuse contract가 필요하다.

generated sequence에는 input prompt가 포함되는 decoder-only convention과 decoder output만 있는 encoder-decoder convention이 다를 수 있다. wrapper usage/text slicing은 model architecture를 알아야 한다. visible completion start index를 hardcode하지 않는다.

beam output에는 sequence scores와 beam indices가 있을 수 있다. simple greedy output과 fields가 다르다. mode selection을 output parser와 연결한다. same generate method가 one universal output schema를 보장하지 않는다.

streamer delivery와 final returned tensor를 double-deliver하지 않게 application design을 한다. streamer는 incremental UI, return은 final authoritative sequence로 쓸 수 있다. offline decode와 streamed concatenation parity를 fixture로 본다.

### source audit에서 custom model override를 놓치지 않는다

GenerationMixin common loop가 model-specific input semantics를 모두 알지는 못한다. `prepare_inputs_for_generation`, cache position preparation, model forward와 `_reorder_cache` 같은 overrides가 architecture contract를 채운다.

Llama/Qwen/Gemma model class에서 forward signature와 prepare override를 찾아 common loop가 넘기는 kwargs를 확인한다. custom remote code model은 behavior가 더 다를 수 있다. trust remote code가 code execution과 generation semantics를 함께 바꾼다는 점을 기록한다.

common source가 cache slicing을 호출하더라도 override가 full inputs를 반환할 수 있다. source audit는 actual model class method resolution order까지 내려가야 한다. generic docs만으로 actual path를 증명하지 않는다.

encoder-decoder는 encoder outputs preparation과 decoder start token이 추가된다. 이 장의 decoder-only representative path와 다르다. 새 architecture를 감사할 때 input owner와 cache axes를 다시 그린다.

multimodal generation은 processor outputs와 placeholder/embeddings, model kwargs update를 추가한다. chat/API preprocessing과 generate core 경계를 분리한다. same `generate` lifecycle이지만 first inputs가 text IDs만은 아니다.

compiled/fullgraph path나 external generation backend가 common Python loop를 대체하면 callable identity와 supported custom processors/streamers를 확인한다. config option만 보고 common loop source를 runtime path라고 단정하지 않는다.

### 장애 인계 문장은 first divergence를 포함한다

나쁜 문장은 “Transformers cache가 틀린다”다. 좋은 문장은 “prompt length 3의 first forward logits와 returned cache length 3은 cache-off oracle과 일치하지만 second preparation의 cache position이 0으로 reset되어 second logits부터 갈린다”다.

streamer incident는 “model forward와 selected IDs timing은 정상이나 bounded streamer queue put이 consumer pause 뒤 4초 block되고 worker loop가 멈추며 end sentinel이 생성되지 않는다”라고 쓴다. backpressure owner가 보인다.

distributed incident는 “rank 0 local criteria true 뒤 loop를 exit하고 rank 1이 next forward collective를 호출해 hang한다. synced termination branch가 disabled였다”처럼 local/global termination을 쓴다.

config incident는 “caller direct max-new 32가 effective config에서 total max 42가 되어야 하나 explicit config merge 뒤 max 20이 남으며 loop는 source-defined max criterion에서 정상 종료한다”처럼 loop bug와 precedence를 나눈다.

cleanup incident는 “generate returned 뒤 sequences만 보존한 control은 live tensor가 줄지만 return hidden states path는 step/layer tuples를 caller result가 참조한다”처럼 intentional retention을 구분한다.

수정 뒤에는 failing boundary와 adjacent cases를 본다. cache second step뿐 아니라 third, varying prompt lengths와 batch padding을 본다. streamer normal/error/cancel terminal을 본다. synced one/all ranks finished와 exception을 본다.

## 23.9 config·model·loop 소유권을 source audit로 되짚는다

고정 source로 config merge/validation, helper calls, loop ordering, state update와 streamer methods를 증명할 수 있다. model-specific override도 source로 따라갈 수 있다. 그러나 actual GPU kernel, queue delay, memory bytes와 distributed timing은 runtime evidence가 필요하다.

source에 `synced_gpus` branch가 있다고 actual call에서 true였다고 단정하지 않는다. streamer class에 queue가 있다고 bounded 여부와 consumer speed를 추측하지 않는다. cache option이 있다고 selected concrete class를 initialization evidence 없이 단정하지 않는다.

실행 report에는 commit, model class/revision, effective GenerationConfig, input shapes, selected mode/callable, cache class, output flags, streamer/wrapper와 distributed topology를 붙인다. 하나라도 빠지면 generic claim을 좁힌다.

이 장의 numeric timelines은 concepts를 위한 fixtures다. 실제 latency 결과가 아니다. runtime을 실행하지 않는다는 constraint를 지킨다. 독자는 같은 fixtures를 승인된 환경에서 observation plan으로 사용할 수 있다.

최종적으로 classic generate는 숨은 magic이 아니다. effective config가 invocation program을 만들고, caller-owned loop가 model/cache/processors/criteria를 step마다 갱신하며, streamer와 returned object가 output lifetime을 나눈다. 이 수명을 정확히 보면 server scheduler가 추가해야 할 owner도 선명해진다.

### 23.9.1 실제로 source를 읽는 30분 경로

처음 5분에는 public `generate` signature와 docstring에서 caller-visible inputs/outputs를 고정한다. 그다음 body에서 config preparation, model kwargs validation, special-token/input preparation, mode resolution과 selected method call에 표식을 붙인다. 모든 branch를 읽으려 하지 않는다.

다음 5분에는 effective config가 만들어지는 helper를 연다. base/default config가 어디서 오고 kwargs가 어떻게 update되는지, unused kwargs와 validation warnings가 어떻게 처리되는지 본다. representative fields max new, sample, cache와 output flags만 먼저 따라간다.

다음 5분에는 input/cache helpers를 읽는다. decoder-only IDs path에서 attention mask, cache position와 cache class를 찾는다. model-specific `prepare_inputs_for_generation` override로 내려가 first and cached steps가 어떤 inputs를 반환하는지 본다.

다음 10분에는 selected no-beam loop를 읽는다. while condition, model input preparation, forward call, last logits selection, processors, sample/argmax, unfinished mask, streamer put, kwargs/cache update, criteria와 break의 순서를 한 줄씩 적는다. source code를 그대로 복사하지 않고 state transition으로 번역한다.

마지막 5분에는 output construction과 error/streamer terminal을 본다. returned type과 histories, cache, sequences ownership을 적고 caller wrapper가 thread/cancel을 어떻게 추가하는지 범위를 확인한다.

이 30분 산책의 산출물은 function catalogue가 아니다. effective config 한 장, two-step state timeline, output/cancel ownership 그림이다. beam이나 assisted가 필요하면 같은 의미 slots에 그 mode-specific state를 추가한다.

### 23.9.2 mode selector를 과도하게 설명하지 않는 이유

generation modes는 많지만 독자는 한 번에 모두 기억하지 못한다. 더 중요한 것은 mode가 loop topology를 선택한다는 원리다. greedy/sample은 row별 next token, beam은 hypotheses/reorder, assisted는 draft/verification state를 추가한다. 공통 입구와 output contract를 유지하면서 state가 달라진다.

mode catalogue를 앞에 놓으면 prefill/cache/processors/criteria라는 공통 수명이 가려진다. 대표 loop를 이해한 뒤 mode-specific delta를 읽으면 새 mode도 분석할 수 있다. expanded rows와 owner, cache reordering, terminal condition을 묻는다.

같은 config object가 mode에 따라 일부 fields를 무시할 수 있다. beam-only option을 greedy에 넘겼을 때 warning/error를 보는 이유다. option 존재보다 selected consumer를 확인한다.

external implementation이나 compile path는 Python loop를 대체할 수 있으므로 actual callable을 기록한다. 이 경우 representative semantic checkpoints—effective config, prepared inputs, outputs와 terminal—가 같은지 확인하고 internal loop source를 별도로 읽는다.

### 23.9.3 early EOS와 streamer가 어긋나는 사건

batch A가 EOS를 선택했는데 streamer UI에는 EOS 뒤 이상한 문자가 한 조각 보였다고 하자. raw selected IDs, unfinished mask application, streamer put argument와 decode filtering 순서를 본다.

loop가 raw next tokens를 streamer에 put한 뒤 finished rows를 pad로 replace하면 A의 forbidden post-finish token이 보일 수 있다. 반대로 mask replacement 뒤 put하면 pad/EOS skipping policy가 적용된다. actual source order를 확인한다.

첫 EOS 자체를 streamer가 받는 것은 가능하다. tokenizer `skip_special_tokens`가 visible text에서 제거할 수 있다. UI가 EOS text를 보였다고 loop가 one extra iteration했다고 단정하지 않는다. raw IDs와 step count를 본다.

batch row B가 계속되는 동안 A padding IDs가 returned rectangular tensor에 쌓일 수 있다. streamer가 batch generation을 지원하는 범위와 row demultiplex를 확인한다. 기본 text streamer가 batch size >1을 제한할 수도 있다. wrapper가 지원한다고 가정하지 않는다.

negative fixture는 A EOS step1, B step3과 special-token skipping on/off다. streamer chunks, returned IDs, unfinished flags와 iteration counts를 비교한다. visible UI만 보지 않는다.

### 23.9.4 custom stopping criterion이 memory를 붙잡는 사건

custom criterion object가 prompt tensor, tokenizer와 large external state를 closure로 잡고 있다고 하자. generate가 끝난 뒤 caller가 criterion list를 reuse하며 references가 남아 memory가 해제되지 않는다. model cache leak처럼 보일 수 있다.

criterion은 each step input IDs와 scores를 받으므로 history를 자체 list에 append할 수도 있다. library output flags가 off여도 custom object가 all scores를 보존한다. caller-owned mutable state를 감사한다.

concurrent calls가 같은 criterion instance를 공유하면 cancel flags와 histories가 섞일 수 있다. invocation per-instance를 쓰거나 synchronization을 설계한다. GenerationMixin이 arbitrary custom object thread safety를 보장하지 않는다.

criterion exception은 loop를 빠져나가지만 streamer terminal과 distributed peers cleanup을 wrapper가 처리해야 한다. custom code를 trusted extension으로 보고 failure scope를 넓힌다.

복구는 criterion reference를 놓고 object history가 bounded인지, normal/cancel/error paths와 second call이 정상인지 본다. allocator reserved memory와 live references를 분리한다.

## 23.10 관측과 검증으로 classic 호출을 닫는다

classic call total latency는 config/input prep, first forward, repeated decode forwards, processors/criteria/streamer gaps와 output construction으로 나눈다. CUDA synchronization 때문에 Python timestamp가 어느 work를 포함하는지 주의한다.

first forward는 prompt length와 batch에 민감하고 decode forward는 cache length와 batch에 민감하다. processor나 streamer CPU work는 forward between gaps에 보인다. long gap을 model kernel time으로 귀속하지 않는다.

streamer callback이 network write를 직접 하면 each token ITL에 transport가 들어간다. queue relay를 쓰면 generate loop와 client delivery가 decouple되지만 buffering/backpressure가 생긴다. architecture를 기록한다.

`output_scores` histories append와 CPU conversion wrapper가 decode step마다 sync를 만들 수 있다. source에 tensor append가 있다고 CPU sync를 단정하지 않지만 runtime profiler observation field로 둔다.

synced ranks에서는 forward뿐 아니라 finished-flag collective/gap을 본다. one slow rank가 all ranks ITL을 정한다. classic distributed invocation metric과 server per-request ITL을 섞지 않는다.

### 23.10.1 correctness differential에서 classic generate의 역할

classic eager-like path는 server engine의 model semantics reference로 유용하다. same tokenizer/template IDs, model artifact, dtype, config와 seed를 고정하고 selected logits/tokens를 비교한다. 하지만 cache/layout/backend가 달라 bitwise equality는 필수 아닐 수 있다.

first-step logits가 다르면 inputs/model/backend, first는 같고 later가 다르면 cache/position, raw logits는 같고 selected tokens가 다르면 processors/RNG, IDs는 같고 text가 다르면 detokenizer/stream layer로 나눈다.

server가 continuous batching을 하므로 classic call과 batch neighbors가 다르다. deterministic greedy에서 model math는 같아야 하지만 low precision reduction와 backend shape가 tolerance 차이를 만들 수 있다. numeric contract를 사전에 정한다.

classic call이 맞다는 사실은 server lifecycle correctness를 증명하지 않는다. disconnect, request isolation, paged cache reuse와 goodput은 별도 tests다. reference 범위를 model generation semantics로 제한한다.

반대로 server output이 다르다고 classic library가 absolute truth라고 단정하지 않는다. model-specific server implementation이 valid alternate kernel을 쓸 수 있다. first semantic divergence와 source contract를 확인한다.

### 23.10.2 한 줄 API 뒤의 소유권을 다시 읽는다

`generate`는 convenient entry지만 호출자가 lifecycle을 소유한다. config와 inputs를 준비하고 synchronous loop가 끝날 때까지 기다리며 returned outputs와 streamer/thread wrapper를 관리한다. library는 process-wide request scheduler가 아니다.

cache는 model loop의 과거 state를 저장하지만 multi-request allocator가 아니다. streamer는 incremental output을 제공하지만 HTTP cancel protocol이 아니다. synced GPUs는 distributed loop ordering을 보존하지만 server fairness가 아니다. 이름이 비슷한 기능을 owner로 구분한다.

문제가 생기면 한 호출 state를 잡는다. effective config, prepared first inputs, each step cache/input, raw and processed scores, selected IDs, criteria와 output terminal을 본다. first divergence가 다음 source 위치를 정한다.

성능을 볼 때도 같은 timeline을 쓴다. config/input CPU, prefill, decode, processor/streamer gap과 caller delivery를 나눈다. model forward 외 시간을 GPU 문제로 묶지 않는다.

cleanup은 invocation references와 wrapper tasks, distributed peers까지 닫는 과정이다. returned histories를 caller가 의도적으로 보존하는 것과 exception path leak을 구분한다. cancel은 cooperative boundary와 terminal signaling을 설계한다.

이 장을 읽은 독자는 classic generate를 작은 serving system처럼 과장하지 않으면서도 source-level reference로 깊게 사용할 수 있다. 다음 continuous manager 장에서는 invocation-local owner가 process-wide request manager와 scheduler로 어떻게 이동하는지 비교하게 된다.

### 23.10.3 마지막 종이 실습: 증상에서 함수까지

독자가 직접 판정할 수 있는 종이 실습을 하나 더 두자. 증상은 prompt 세 tokens, max new 두 tokens 호출이 세 tokens를 생성하고 streamer는 마지막 chunk를 닫지 않는 것이다. 두 문제가 하나의 원인이라고 가정하지 않는다.

첫 번째 branch는 length state다. raw caller kwargs와 base config를 적고 effective max total length를 계산한다. prompt 3+new 2면 5다. returned sequence length가 6이라면 criteria가 wrong effective cap을 받았거나 evaluation order가 한 step 늦을 수 있다. EOS가 없었다는 사실만으로 max criterion을 기각하지 않는다.

effective max가 6이었다면 config merge/precedence가 owner다. effective는 5인데 loop가 length 6까지 갔다면 criterion construction, call timing 또는 selected implementation을 본다. common Python loop가 아니라 external callable이면 그 loop source로 이동한다.

두 번째 branch는 streamer terminal이다. worker generate가 정상 return했는지, common loop가 streamer `end`를 호출했는지, wrapper exception/finally가 terminal을 전달했는지, consumer가 sentinel을 해석했는지 본다. extra token과 terminal hang이 서로 다른 first divergence를 가질 수 있다.

worker가 criterion exception으로 실패해 extra token을 만들고 end를 못 보냈다면 common cause가 될 수 있다. 그러나 evidence 없이 묶지 않는다. returned sequence가 존재하면 worker normal return 가능성이 있고 consumer-only hang이면 separate assembler owner다.

state ledger에는 effective config, selected callable, lengths before/after each append, criteria bool, streamer put/end, worker exception, consumer terminal을 둔다. raw text 전체를 저장할 필요는 없다. IDs와 lengths, state enums로 충분하다.

source owner는 순서대로 config update/validation, criteria builder, generation loop, streamer class와 caller wrapper다. model forward는 raw logits/IDs부터 이상할 때만 본다. GPU profiler는 이 incident의 첫 도구가 아니다.

수정 검증은 max new 0/1/2 boundary, EOS at first token, normal and exception streamer paths를 본다. output length와 terminal exactly-once를 별도 assertions로 둔다. 한쪽 success가 다른 쪽을 증명하지 않는다.

### 23.10.4 API option이 internal state가 되는 다섯 단계

`use_cache=True`를 예로 들면 caller kwarg가 effective config가 되고 cache preparation branch를 열며 model kwargs에 cache object가 들어가고 next input slicing을 바꾸며 later forward cost와 state를 바꾼다. option string만 기록하면 effect chain을 모른다.

`return_dict_in_generate=True`는 output mode를 바꾸고 optional history collections를 허용하며 loop append state와 returned ownership을 바꾼다. JSON formatting option이 아니라 memory lifetime option이다.

`streamer` argument는 callback object를 loop state에 넣고 selected IDs 이후 `put`, terminal에 `end`를 호출하며 consumer concurrency와 backpressure를 추가한다. output text만 바꾸는 것이 아니라 loop timing과 cancellation wrapper를 요구한다.

`synced_gpus=True`는 local finished condition을 global peer agreement에 연결하고 loop break와 collective participation을 바꾼다. model output value option이 아니라 distributed control-flow option이다.

`max_new_tokens`는 config precedence를 지나 derived total cap이 되고 criterion state를 만들며 append 뒤 bool을 바꾸고 loop terminal과 output length를 정한다. 이 chain이 있어야 option이 왜 있고 어떤 값과 event를 바꾸는지 설명할 수 있다.

각 option을 같은 template로 읽는다. caller field, effective config, consumer branch, mutated state, observable output/cost와 counterexample이다. 하지만 본문에 전 옵션 표를 나열하지 않는다. representative cases로 method를 익히고 나머지는 source note에서 적용한다.

### 23.10.5 새 model architecture를 만났을 때

common GenerationMixin을 읽은 뒤 actual model의 `prepare_inputs_for_generation`과 forward를 연다. input name, cache class, cache positions, attention mask와 special kwargs를 적는다. common loop가 제공하는 state를 architecture가 어떻게 소비하는지 연결한다.

Qwen/Gemma/Llama가 모두 decoder-only라도 sliding layers, hybrid state와 cache formats가 다를 수 있다. config `use_cache` 하나가 identical cache bytes를 뜻하지 않는다. common generation lifecycle과 model-specific state shape를 분리한다.

remote custom code는 common methods를 override할 수 있다. selected class source revision을 고정하고 untrusted code security를 고려한다. library version만 고정해도 model repository code가 바뀌면 behavior가 달라질 수 있다.

new architecture differential은 cache off first, cache on step-by-step, output processor/stopping 순서로 간다. first logits와 second cache step을 고정한 뒤 advanced modes와 compile을 켠다. 여러 axes를 동시에 바꾸지 않는다.

이 절차는 runtime 실행을 권한 없이 수행하라는 뜻이 아니다. 이 장에서는 source path와 fixtures를 준비한다. 승인된 환경에서 실행한다면 artifact hashes와 effective config, selected callable을 함께 기록한다.

### 23.10.6 심층 실습 전에 확인하는 현재 exit condition

독자는 one invocation의 owner를 말할 수 있어야 한다. caller가 inputs/config와 returned objects를 소유하고 GenerationMixin loop가 intermediate cache, IDs, processors와 criteria를 소유하며 model이 forward/cache update semantics를 소유한다. streamer wrapper가 concurrency와 cancellation을 추가한다.

또한 first divergence를 찾을 수 있어야 한다. config, prepared inputs, prefill logits, cache update, processed scores, selected ID, criterion, streamer terminal과 returned output 순서다. 이전 checkpoint가 같으면 앞 owner를 반복 조사하지 않는다.

마지막으로 continuous server와 차이를 설명해야 한다. classic invocation에는 cross-request admission과 dynamic batch membership, process-wide paged allocator와 HTTP disconnect mapping이 없다. 이를 추가하는 manager가 다음 장의 주인공이다.

이 exit condition을 실제 review 문장으로 바꾸면 다음과 같다. “effective config에서 total cap과 selected mode가 확정되고, first preparation은 full prompt와 empty cache를, next preparation은 latest token과 updated cache를 만든다. loop는 processed scores에서 ID를 고르고 append 뒤 criteria를 평가하며 streamer와 returned object에 서로 다른 output lifetime을 제공한다.” 이 문장을 source 좌표로 뒷받침할 수 있어야 한다.

어느 한 구간을 아직 확인하지 못했다면 추측으로 채우지 않는다. custom model override, external generation callable, wrapper thread와 distributed launcher는 common source 밖의 owner일 수 있다. 미검증이라고 표시하고 actual class와 caller를 찾는다.

반대로 많은 option을 읽었다고 lifecycle이 닫힌 것도 아니다. config field가 consumer에 도달하지 않거나 exception path에서 streamer terminal이 빠지고 cache reference가 잘못 update되면 핵심이 깨진다. field 개수보다 state transition coverage가 중요하다.

이 장의 종이 fixtures는 실제 model 실행 없이 source ordering을 검산하기 위한 것이다. 승인된 runtime test에서는 같은 IDs, effective config와 model revision을 고정하고 step별 shapes/cache lengths, selected IDs와 terminal events를 수집한다. source expectation과 observation을 별도 열에 둔다.

독자가 이 습관을 익히면 `generate`의 긴 함수가 덜 위협적으로 보인다. 입구에서 모든 branch를 외우지 않고 대표 호출의 state를 붙잡는다. branch가 나타날 때 그 branch가 config, input, loop, output 중 무엇을 바꾸는지 묻는다. 그리고 first divergence가 생긴 owner까지만 내려간다.

그 결과 설명도 짧아진다. 어떤 option을 켰다는 말 대신 어느 effective state와 loop transition이 달라졌고 어느 output과 비용이 변했는지 source로 말할 수 있다. 이것이 실용적인 함수 단위 독해다.

## 23.11 GenerationConfig 우선순위를 값의 출처 원장으로 푼다

사용자가 `generate(max_new_tokens=32, temperature=0.7)`를 호출했을 때 최종 값은 함수 인자만의 결과가 아니다. 모델에 붙은 generation config, 명시적으로 전달한 `GenerationConfig`, 호출 kwargs와 library default가 합쳐진다. 같은 필드가 여러 층에 존재하면 마지막 effective config가 어느 값을 골랐는지 알아야 한다. 로그에 request payload만 남기면 실제 실행과 다른 설정을 보고 디버깅할 수 있다.

고정한 Transformers 소스의 `_prepare_generation_config`는 config를 선택하고 복사한 뒤 kwargs로 업데이트하며, generation config에 속하지 않는 값은 model kwargs로 남긴다. `generate`는 이 결과를 validation, input preparation, mode 선택, processor와 criteria 생성에 넘긴다.

이 경계가 중요한 이유는 “알 수 없는 kwargs”와 “generation option override”가 같은 dict에서 출발하지만 서로 다른 consumer로 갈라지기 때문이다. [Transformers effective generation config 준비](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L1771-L1834), [generate의 config 준비와 consumer 연결](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2472-L2667)

### 네 층이 충돌하는 수치 fixture

library default가 `max_length=20`, model의 저장 config가 `max_new_tokens=64, do_sample=false`, caller가 명시적 `GenerationConfig(max_new_tokens=48, temperature=0.8)`을 주고 kwargs에 `max_new_tokens=16, do_sample=true`를 넣었다고 하자. effective 결과는 호출 kwargs override가 적용돼 `max_new_tokens=16, do_sample=true, temperature=0.8`이 된다. 단, 정확한 precedence는 해당 revision의 준비 함수와 config update를 근거로 검증한다.

prompt 길이가 100이면 `max_new_tokens=16`은 total sequence cap 116을 만든다. `max_length=20`이라는 default 숫자를 그대로 stopping criterion에 쓰면 prompt가 이미 cap을 넘었다. preparation 단계가 new-token cap을 현재 input length와 합쳐 derived total max length로 만드는 이유다. 옵션 원장에는 원본 field뿐 아니라 `prompt_len=100 → derived_max_length=116`이라는 파생값을 기록한다.

`do_sample=true`는 temperature와 top-p를 실제 sampling path의 consumer로 보낸다. temperature 0.8이 config에 있어도 `do_sample=false`라면 경고 또는 비활성 의미가 될 수 있다. field 존재와 실행 branch 활성화를 분리한다. “temperature가 0.8이라 다양성이 늘었다”는 설명은 effective mode가 sampling임을 확인한 뒤에만 성립한다.

이 fixture의 표는 `(field, library default, model config, explicit config, kwargs, effective, derived, consumer)` 여덟 열을 가진다. 값 하나가 잘못되면 source에서 어느 merge가 처음 달랐는지 찾는다. model config가 오래된 artifact에서 생성됐는지, explicit object가 호출 전에 mutation됐는지, kwargs spelling이 config field가 아니라 model input으로 남았는지도 본다.

### config copy는 요청 격리와 memory lifetime을 만든다

effective preparation이 config를 복사하는 이유 중 하나는 한 호출의 kwargs가 모델의 공유 default를 영구 변경하지 않게 하는 것이다. 두 thread가 같은 model object를 사용하고 A가 temperature를 0.2로, B가 1.0으로 호출할 때 공유 object를 in-place 수정하면 race가 된다. copy 뒤 request-local effective config를 만드는지 확인한다.

그러나 custom callable이나 wrapper가 전달 전에 `model.generation_config`를 mutation하면 common 함수 밖에서 race가 생길 수 있다. source audit은 library 함수만 보고 끝내지 않고 caller의 config 생성과 보관 lifetime을 확인한다. 서버 wrapper라면 request payload에서 새 config를 만드는지, global template를 수정하는지 본다.

config 안에 nested object가 있을 때 shallow/deep copy 의미도 중요하다. watermark나 cache config 같은 mutable nested field가 공유되면 top-level copy만으로 격리되지 않을 수 있다. 고정 revision의 실제 copy 함수를 확인하고, fixture에서 nested ID와 mutation 전후를 검사한다. 실행하지 않는 source-only 단계에서는 가능한 alias를 표시하고 runtime pointer/digest probe를 설계한다.

### processor와 stopping list는 config의 실행 형식이다

`_get_logits_processor`는 repetition, no-repeat ngram, bad words, min length, forced token, diversity, watermark 등 effective config와 model/input 정보를 읽어 ordered processor list를 만든다. `generate`에 custom processor list가 전달되면 built-in과 merge된다. 같은 type을 중복 넣는 충돌을 어떻게 처리하는지 확인한다. config 값만 맞아도 list 구성이나 순서가 다르면 score path가 달라진다. [Transformers logits processor 구성](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L1123-L1356)

`_get_stopping_criteria`는 max length, max time, stop strings, EOS 같은 조건을 구성하고 custom criteria와 merge한다. condition list는 score processor처럼 다음 token의 분포를 바꾸는 것이 아니라 append된 sequence의 종료 bool을 만든다. 같은 config에서 파생되더라도 consumer와 mutation stage가 다르다. [Transformers stopping criteria 구성](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L1358-L1418)

option 설명은 이 변환을 끝까지 적는다. `max_new_tokens=16`은 request-local config override이고 prompt length와 합쳐 total cap 116이 되며 max-length criterion을 만들고 append 뒤 `unfinished_sequences`를 갱신해 loop를 끝낸다. `repetition_penalty=1.2`는 processor instance와 history input을 만들고 model raw logits가 아니라 processed score를 바꾼다. 이 정도 연결이 있어야 “왜 이 옵션이 있는가”가 실행 의미와 맞물린다.

## 23.12 한 decode step에서 input·mask·cache가 어떻게 자라고 잘리는지 계산한다

decoder-only batch B=2, prompt 유효 길이가 각각 5와 3이고 left padding으로 tensor shape가 `[2,5]`라고 하자. 첫 forward는 full prompt를 처리한다. attention mask는 두 row의 valid/pad를 구분하고 cache position은 실제 model contract에 맞춰 prompt 위치를 나타낸다. output cache는 각 layer에서 key/value 길이 5에 해당하는 state를 갖는다. 짧은 request의 padding이 cache 물리 length에 포함되는지, position/mask로 무효화되는지는 model과 cache 구현을 읽는다.

첫 토큰을 붙여 입력 ID 모양이 `[2,6]`이 되어도, 캐시를 쓰는 다음 순전파가 여섯 토큰을 전부 다시 계산하는 것은 아니다. `prepare_inputs_for_generation`은 캐시 위치와 이미 처리한 길이를 보고 아직 처리하지 않은 토큰 조각만 모델 입력으로 만든다.

공통 도우미의 주석도 이 자르기 작업을 갱신 함수가 아니라 입력 준비 단계가 맡는다고 명시한다. 근거는 [Transformers generation kwargs update 계약](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L940-L996)과 [Transformers input preparation](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L519-L641)에서 함께 확인할 수 있다.

step 1에서 model input은 보통 latest token `[2,1]`, attention mask는 누적 `[2,6]`, cache position은 새 위치 하나, cache는 과거 길이 5를 나타낸다. forward 뒤 cache logical length는 6이 된다. step 2에서는 IDs container는 `[2,7]`, model input `[2,1]`, mask `[2,7]`, 새 cache position과 cache length 7이 된다. “input length”라는 말이 full sequence container, current model slice, cache length 세 값을 가질 수 있으므로 이름을 붙인다.

### layer별 KV byte로 lifetime을 계산한다

layers L=32, KV heads Hkv=8, head dim Dh=128, BF16 두 byte, key와 value 두 tensor를 가정한다. token 하나의 KV는 `32×8×128×2×2=131,072 byte`, 즉 128 KiB다. batch 2의 prompt 물리 길이 5라면 약 1.25 MiB이고 decode token 하나마다 batch 전체에 256 KiB가 늘어난다. 16 new tokens이면 단순 증가량은 4 MiB다.

이 값은 dense contiguous cache의 logical operand 계산이다. actual cache class의 layout, alignment, sliding layer, quantized cache, offload는 다를 수 있다. 그래도 `return_dict_in_generate`나 score retention과 달리 KV는 future forward가 읽는 persistent state라는 점이 중요하다. update 뒤 old cache reference가 해제되는지 in-place object가 자라는지 cache class별로 본다.

`_update_model_kwargs_for_generation`은 model output에서 cache를 꺼내 model kwargs에 넣고 attention mask 또는 decoder attention mask를 한 열 늘리며 cache position을 갱신한다. loop가 다음 iteration에 이 kwargs를 다시 preparation으로 보낸다. cache update, mask append, cache position 이동이 하나의 request step을 형성한다. [Transformers cache·mask·position update](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L940-L996)

### cache_position off-by-one incident

custom model의 `prepare_inputs_for_generation`이 latest token을 정확히 slice했지만 wrapper가 `cache_position`을 append 전에 한 번, update 함수 뒤 한 번 진행했다고 하자. 첫 decode token은 position 5 대신 6으로 RoPE를 적용한다. input ID와 cache shape는 모두 정상이고 예외도 없다. 첫 generated logit부터 reference와 다르며 이후 cache 전체가 오염된다.

first divergence는 effective config나 prompt prefill이 아니다. prefill logits와 cache digest는 같고, 두 번째 forward의 model input token도 같다. cache position scalar만 reference보다 1 크고 q/k rotation 뒤 attention output이 갈린다. incident record에는 `(logical_generated_step, full_ids_len, model_slice_len, cache_seq_len_before, cache_position, mask_len)`을 둔다.

수정 뒤 cache on/off reference를 비교한다. cache off는 매 step full sequence를 다시 계산하므로 올바른 position 기준선이 된다. cache on에서 first logits, appended ID, 다음 model slice와 position을 맞춘다. sliding/hybrid cache에서는 cache physical length가 logical absolute position과 다를 수 있으므로 단순 equality 대신 model contract를 쓴다.

### finished row가 batch 안에 남을 때

classic generate는 batch row별 `unfinished_sequences` mask를 유지하면서 모든 row가 끝날 때까지 loop를 돌 수 있다. 먼저 끝난 row에는 pad token을 append하고 model forward에 계속 포함될 수 있다. 따라서 B=2에서 한 row가 EOS를 만났다고 곧바로 batch dimension이 1로 compact된다고 가정하지 않는다. 이는 다음 장의 continuous manager와 중요한 차이다.

finished row가 cache와 attention mask를 계속 늘리는지, model compute에서 완전히 제거되지 않는지는 실제 loop를 읽는다. batch 내부 길이 불균형이 커질수록 wasted work가 생길 수 있다. classic generate를 서버로 감싼다고 continuous batching이 자동으로 생기지 않는 이유다.

## 23.13 output_scores·hidden_states·attentions가 만드는 숨은 retention을 계산한다

`return_dict_in_generate=true`와 `output_scores=true`를 켜면 loop가 step별 score tensor를 tuple에 쌓을 수 있다. vocabulary V=152,064, batch B=8, FP32 score라면 한 step은 `8×152064×4=4,865,?` byte, 약 4.64 MiB다. 정확히는 4,866,048 byte다. 256 new tokens를 모두 보존하면 약 1.16 GiB다. KV cache와 model weight 외에 이 retention이 OOM을 만들 수 있다.

score가 processed인지 raw인지, generation mode에 따라 어떤 tensor가 저장되는지 source contract를 확인한다. top-N logprobs만 필요한데 full vocabulary score tuple을 요청하면 consumer 요구보다 훨씬 큰 정보를 보존한다. 디버깅 편의 옵션을 production default로 켜지 않는다.

`output_hidden_states`는 더 복잡하다. layer별 hidden을 step별로 보존할 수 있다. decode step은 current slice가 1 token이어도 layers 32, hidden D=4096, BF16, B=8이면 단순 hidden payload가 `33×8×1×4096×2≈2.06 MiB/step`이다. 256 step이면 약 528 MiB다. prefill은 sequence length가 들어가 훨씬 크다. tuple/object overhead와 dtype, returned layer count는 별도다.

attention output은 heads와 query/key length를 포함한다. prefill self-attention matrix를 보존하면 sequence length 제곱이 된다. serving 성능을 보려는 목적으로 attentions를 켜는 순간 관측 자체가 workload를 크게 바꿀 수 있다. profiler와 trace를 우선하고 full tensor는 작은 synthetic fixture에 제한한다.

### 24 GiB GPU에서 일어난 OOM 사건

모델과 runtime 기본 사용량이 20.5 GiB, KV가 생성 중 1.0 GiB, allocator/workspace 여유가 1.2 GiB라고 하자. 남은 안전 여유는 약 1.3 GiB다. 사용자가 debugging을 위해 B=8, 256-token generation에 output scores를 켜면 계산상 약 1.16 GiB가 step tuple에 남는다. fragmentation과 temporary peak를 더하면 후반 step에서 OOM이 난다.

증상은 “긴 출력에서만 OOM”이고 KV cache가 흔한 용의자다. 하지만 step별 KV 증가 metric은 예상과 같고, allocated memory가 매 step 약 4.64 MiB씩 추가로 증가한다. `scores` tuple length와 element byte를 원장에 넣으면 retention owner가 generation output임을 찾는다. `output_scores=false`에서 같은 workload가 통과하는 것은 가설을 지지하지만, 실제 반환 계약이 score를 요구하는지 확인해야 한다.

수정은 필요하지 않은 output retention을 끄거나 bounded observation으로 바꾸는 것이다. 사용자가 full scores를 명시적으로 요구한다면 무조건 삭제할 수 없다. max_new_tokens와 batch, vocabulary에 따른 사전 byte 추정, 요청 제한, CPU offload 또는 streaming-like extraction을 검토한다. 각각 latency와 API semantics가 다르다.

수정 완료는 OOM이 사라진 것만이 아니다. 요청한 output field가 정확히 반환되고, 필요하지 않은 tensor reference가 step 뒤 해제되며, peak memory와 증가 slope가 계산과 맞아야 한다. exception path에서도 streamer와 retained tuple, cache reference가 해제되는지 본다.

### `use_cache=false`도 memory가 항상 작다는 뜻은 아니다

cache를 끄면 persistent KV는 줄지만 매 step 전체 growing sequence를 다시 forward한다. attention temporary와 compute가 커지고, output attentions/hidden states를 함께 보존하면 큰 prefill-like tensor가 반복될 수 있다. capacity 한 축만 보고 최적화로 부르지 않는다. latency, temporary peak, retained output을 함께 계산한다.

`cache_implementation`이 dynamic, static, offloaded, quantized 등으로 바뀌면 allocation timing과 resident device byte, transfer가 달라진다. config field가 actual cache object를 어디서 만드는지 `_prepare_cache_for_generation` 경로를 따라간다. model이 자체 cache를 공급하거나 external callable이 다르면 common quick escape가 있을 수 있다. [Transformers cache preparation](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L1931-L2037)

**classic generate 통제 실험.** 실험 A는 greedy와 `do_sample=false`의 effective config를 비교한다. 실험 B는 EOS와 max-new-token이 같은 step에 도착할 때 finish owner를 확인한다. 실험 C는 streamer callback에서 예외를 내 cache와 worker cleanup을 본다. 실험 D는 `output_scores` on/off에서 token은 같고 retention byte만 달라지는지 측정한다. 이 네 실험은 generation 결과와 호출 수명주기를 분리한다.

## 23.14 exception·streamer·cache cleanup을 포함한 최종 호출 dossier

정상 경로만 읽으면 loop가 token을 append하고 criteria가 끝내며 output을 반환한다고 요약할 수 있다. 서버나 notebook에서 실제로 문제가 되는 것은 exception과 조기 중단이다. streamer가 별도 thread/queue로 소비될 때 model forward가 예외를 내면 consumer가 영원히 기다리지 않도록 terminal signal을 보내야 한다. common `generate`가 제공하는 범위와 wrapper 책임을 구분한다.

### 한 호출의 상태 표

입구 state에는 model/tokenizer revision, caller input IDs와 mask, explicit config와 kwargs가 있다. preparation state에는 effective config, derived total cap, mode, cache object, processor/criteria list가 있다. loop state에는 full IDs, current model slice, model kwargs, cache position, unfinished mask, generator, retained outputs가 있다. 출구 state에는 sequences, optional scores/hidden/attentions/cache, streamer terminal과 exception이 있다.

각 state에 owner와 lifetime을 붙인다. effective config와 processors는 invocation 전체, current model slice와 model outputs는 한 iteration, KV cache와 full IDs는 generation 전체, returned score tuple은 caller가 output object를 놓을 때까지 살 수 있다. Python reference가 남으면 CUDA tensor도 해제되지 않을 수 있다. 함수가 return했다고 즉시 device memory가 모두 줄어든다고 가정하지 않는다.

### processor exception 뒤 streamer hang incident

custom logits processor가 step 7에서 all `-inf`를 감지하고 exception을 던졌다. generation worker thread는 종료됐지만 wrapper가 streamer의 end를 `generate` 정상 반환 뒤에만 호출했다. HTTP iterator는 queue에서 다음 text를 기다리며 연결을 닫지 않았고 cache/output references도 thread future에 남았다. GPU compute는 멈췄지만 요청 latency와 memory가 누적됐다.

first divergence는 score가 틀린 것이 아니라 exception-to-terminal edge다. trace에는 request incarnation, worker future state, exception type, last committed step, streamer end sent, consumer task state, cache/output references를 둔다. model error counter만 있고 active stream gauge가 줄지 않으면 cleanup 가설이 강하다.

수정은 wrapper의 `try/except/finally` ownership에서 error와 terminal을 consumer에 전달하고 request references를 해제하는 것이다. 구체적인 API는 wrapper마다 다르다. 이미 terminal이 전송된 경우 중복 end를 막고, partial text 뒤 error contract를 정한다. thread를 강제 종료하는 것만으로 queue consumer가 깨어나는 것은 아니다.

회귀 fixture는 정상, processor exception, model forward exception, consumer cancellation을 나눈다. 각 경우 worker 종료, streamer terminal 정확히 한 번, returned/raised contract, cache reference release와 active gauge 0을 확인한다. CUDA operation이 비동기라면 host exception 시점과 device completion/lifetime을 구분한다.

### classic 호출의 모니터링 최소 집합

effective config digest, selected generation mode, prompt/new token cap, processor와 criterion type digest를 trace에 둔다. iteration별 full IDs length, current slice length, cache logical/physical length, unfinished count를 sampled trace에 둔다. output retention에는 score/hidden/attention tuple length와 estimated bytes를 둔다.

latency는 config/input preparation, prefill, decode iterations, processor/sampling, streamer wait를 나눈다. memory는 model, cache, retained outputs, temporary를 가능한 범위에서 나눈다. 모든 request ID를 metric label에 넣지 않고 model/config cohort와 bounded buckets를 쓴다. 상세 state는 승인된 trace로 연결한다.

### 30분 회귀 워크북

0~5분에는 source/model revision과 effective config를 고정한다. 5~10분에는 cache off/on으로 first two logits와 model input slice, position을 비교한다. 10~15분에는 identity processors와 하나의 non-identity processor로 list order를 확인한다. 15~20분에는 max new tokens와 EOS criteria의 append/terminal 순서를 본다.

20~25분에는 output scores를 켜 step당 byte slope가 계산과 맞는지 작은 fixture에서 본다. 25~30분에는 exception을 주입해 streamer terminal과 reference release를 본다. 실제 model 실행 권한이 없다면 동일 필드를 수집할 probe와 기대값만 준비하고 실행했다고 쓰지 않는다.

### 다음 장으로 넘기는 정확한 경계

classic generate의 invocation은 고정 batch membership을 전제로 한 caller-owned 생명주기다. batch 안에서 row별 unfinished mask는 있지만, 다른 request가 중간에 입장해 빈 row를 채우는 process-wide manager는 아니다. cache도 invocation model kwargs가 소유하며 global page allocator와 request table을 자동 제공하지 않는다.

다음 장의 continuous manager는 request table, admission queue, active batch rebuild, cache allocation과 cancellation을 장기 생존 객체로 끌어올린다. 비교 기준은 API 이름이 아니라 owner lifetime이다. 같은 processor와 model forward를 재사용해도 누가 state를 보관하고 언제 row를 compact하는지가 달라진다.

23장의 최종 invariant는 다음과 같다. **한 invocation의 effective config가 명시된 precedence로 확정되고, 그 config가 준비한 ordered processor·criteria와 cache/input state가 매 iteration 같은 logical sequence를 진행하며, optional output과 streamer가 정상·예외·취소에서 정확한 lifetime과 terminal을 가진다.**

이 문장을 precedence 표, 두-step shape 원장, cache position incident, output retention OOM, exception cleanup fixture로 설명할 수 있다면 `generate`는 더 이상 하나의 거대한 black box가 아니다. 독자는 문제가 설정, preparation, forward/cache, score policy, stopping, output lifetime 중 어디서 처음 생겼는지 찾아 정확한 함수로 들어갈 수 있다.

**mode별로 같은 field가 다른 loop를 선택하는 순간**

`do_sample`, `num_beams`, `num_beam_groups`, constraints, assistant model 같은 field는 scalar 연산 하나를 바꾸는 데 그치지 않고 generation mode를 선택할 수 있다. greedy/sample은 batch row당 다음 token 하나를 진행하지만 beam search는 beam axis와 beam score, ancestry를 가진다. assisted mode는 candidate proposal과 verification state를 추가한다. config precedence 오류가 mode를 바꾸면 tensor shape와 cache reorder, output type이 함께 달라진다.

prompt batch B=2, `num_beams=4`라면 model-facing batch가 8로 확장될 수 있다. 앞의 KV 계산에서 prompt cache 약 1.25 MiB였던 단순 fixture가 beam별 cache를 가진다면 네 배 수준으로 늘 수 있다. 실제 cache sharing과 reorder 구현에 따라 물리 byte는 달라지지만, beam axis가 생성 state를 확장한다는 하한을 원장에 넣는다. `num_return_sequences=2`는 최종 output 개수와 beam 선택을 바꾸지만 반드시 forward batch를 같은 비율로 더 늘리는지는 mode를 읽는다.

mode audit 표에는 effective fields, selected enum/callable, input expansion factor, cache reorder owner, score state, output shape를 둔다. `do_sample=false,num_beams=1`과 `do_sample=true,num_beams=1`, beam 4 세 fixture로 branch를 분리한다. 동일 text가 나왔어도 selected mode와 cost가 다를 수 있다. latency만 보고 mode semantics를 추측하지 않는다.

external generation callable이나 custom model override가 선택되면 common loop의 ordering이 그대로라는 보장이 없다. `generate` 입구에서 callable resolution이 일어나는 revision이라면 실제 callable identity를 기록한다. remote code model은 `prepare_inputs_for_generation`, cache update나 generate 자체를 override할 수 있다. source fingerprint는 library commit과 model repository revision 둘을 포함한다.

**processor list의 충돌을 배열로 검증한다**

raw logits가 `[3,2,1]`이고 config repetition penalty가 token 0을 1.5로 나눠 `[2,2,1]`을 만든다고 하자. caller custom processor도 token 0에 -1 bias를 적용하면 최종 `[1,2,1]`이다. custom list가 config processor보다 앞이고 bias로 score가 양수에서 음수로 바뀌는 더 강한 예에서는 sign-aware repetition 결과가 달라진다. list merge와 ordering은 observable semantics다.

같은 processor type이 config와 custom list에 둘 다 있으면 중복 적용할지 error로 거부할지 고정 소스를 확인한다. 조용한 중복은 penalty를 두 번 적용해 사용자가 예상한 1.5가 사실상 더 강해질 수 있다. custom callable class name만 로그에 남기지 말고 type과 parameter digest, list index를 남긴다.

processor가 CPU tensor나 Python state를 캡처하면 device mismatch와 lifetime도 생긴다. 매 step input IDs 전체를 CPU로 복사하는 custom processor는 model forward보다 작은 코드여도 synchronization 병목이 될 수 있다. source-only audit에서는 callable body의 device transfer, `.item()`, Python loop를 표시하고 approved profiler에서 processor 구간을 측정하도록 probe를 설계한다.

stopping criteria도 custom list와 built-in의 merge를 검증한다. criteria는 row별 bool tensor를 반환할 수 있고 batch 전체의 종료는 unfinished mask와 distributed sync 조건이 결합한다. custom criterion이 Python bool 하나만 반환하거나 device가 다른 tensor를 만들 때 broadcast semantics가 의도와 같은지 본다.

**attention mask가 자라는 동안 position을 따로 보는 이유**

left-padded 두 request의 attention mask는 `[0,0,1,1,1]`과 `[1,1,1,1,1]`처럼 다를 수 있다. position IDs를 mask cumulative sum에서 만드는 model은 짧은 row의 첫 valid token을 0으로 맞춘다. cache position은 cache write의 global sequence 좌표 또는 다른 의미를 가질 수 있다. 두 tensor 이름에 position이 들어가도 같은 값이라고 가정하지 않는다.

decode append 뒤 attention mask에 1을 붙이는 것은 새 token을 유효하게 만든다. finished row에 pad를 append할 때 mask를 계속 1로 붙이는지, unfinished mask로 token만 pad 처리하는지 loop contract를 본다. finished row output은 무시돼도 그 row의 cache와 compute가 계속 변할 수 있다. 이 wasted work를 continuous compaction과 비교할 때 correctness와 capacity를 분리한다.

encoder-decoder model에서는 encoder attention mask와 decoder attention mask, cross-attention cache가 분리된다. 이 장의 decoder-only 숫자를 그대로 적용하지 않는다. `_update_model_kwargs_for_generation`이 encoder-decoder 여부로 어느 mask를 늘리는지 source branch를 읽는다. 같은 `max_new_tokens`라도 input prompt length와 decoder start length의 derived cap 의미가 다를 수 있다.

multimodal model은 image tensor를 첫 step만 쓰거나 model kwargs에 유지할 수 있다. `prepare_inputs_for_generation` quick escape와 model override가 large pixel values를 반복 전달하거나 reference로 보존하는지 본다. cache memory만 측정하고 vision input retention을 놓치지 않는다.

**cache implementation 선택의 효과 카드를 작성한다**

dynamic cache는 decode 길이에 따라 storage를 늘릴 수 있어 실제 생성 길이에 비례하지만 allocation/copy와 fragmentation이 생길 수 있다. static cache는 최대 길이를 미리 잡아 compile-friendly address와 shape를 제공할 수 있지만 짧은 생성에서 예약 waste가 생긴다. offloaded cache는 device capacity를 줄이는 대신 CPU memory와 transfer, synchronization을 추가한다. quantized cache는 byte를 줄이는 대신 encode/decode kernel과 numerical 차이를 만든다.

각 카드는 field/default, cache factory branch, created class, allocation timing, logical/physical shape, device/dtype, model consumer, observable effect를 가진다. `cache_implementation="static"` 문자열이 있어도 model이 지원하지 않거나 supplied `past_key_values` 때문에 factory가 bypass될 수 있다. actual object class와 pointer/device를 반증 관측으로 둔다.

max cache length 4096, 앞의 token당 128 KiB, B=2라면 static logical reservation 하한은 약 1 GiB다. 실제 generated length가 prompt 5+16이어도 전체 capacity를 예약할 수 있다. dynamic logical used byte는 약 `21×2×128 KiB=5.25 MiB`다. 이 극단적 차이는 단순 모델이며 actual preallocation과 layer layout을 확인해야 하지만, static option이 왜 OOM 시작 시점을 앞당길 수 있는지 설명한다.

반대로 CUDA graph나 compile이 static address/shape를 요구하면 static cache가 launch와 allocator 변동을 줄일 수 있다. capacity cost와 latency 이득을 같은 원장에 둔다. 짧은 single request benchmark와 긴 concurrent workload에서 결론이 다를 수 있다.

**synced_gpus가 local finish 뒤에도 loop를 유지하는 경우**

distributed generation에서 한 peer가 먼저 끝났다고 collective 참여를 즉시 중단하면 다른 peer가 hang할 수 있다. synced mode는 peer completion 합의를 위해 local finished 이후에도 loop coordination을 수행할 수 있다. token 선택 semantics와 distributed liveness가 결합되는 지점이다. “EOS를 만났는데 왜 kernel이 더 실행됐는가”를 bug로 단정하지 않는다.

incident fixture는 rank 0이 step 3, rank 1이 step 5에 finish하도록 만든다. step별 `this_peer_finished`, global finished consensus, forward/collective participation과 emitted token을 기록한다. local finished row는 새 의미 token을 commit하지 않아야 하지만 liveness를 위해 collective는 유지될 수 있다. timeout과 rank exception에서는 cleanup owner가 더 복잡하다.

한 rank의 custom processor가 exception을 내고 다른 rank가 collective에 들어가면 streamer hang보다 더 큰 distributed hang이 된다. wrapper는 rank error propagation과 process group policy를 가져야 한다. common generate가 모든 launcher cleanup을 소유한다고 가정하지 않는다. source와 distributed runtime 경계를 명시한다.

**output object를 caller가 오래 보관하는 사건**

generation은 성공했고 함수 local은 끝났지만 notebook 사용자가 결과 객체를 전역 list에 쌓았다. 각 객체가 256-step score tuple 약 1.16 GiB를 참조해 두 번째 호출에서 OOM이 났다. memory snapshot에서 active generation은 0이지만 tensor reference가 살아 있다. library leak과 caller retention을 구분한다.

fixture는 output object를 유지한 경우와 `sequences`만 복사하고 객체를 놓은 경우를 비교한다. Python GC와 CUDA allocator reserved memory를 구분한다. allocated가 줄고 reserved가 남는 것은 allocator cache일 수 있다. 다른 allocation이 reserved block을 재사용하는지 본다. 무조건 `empty_cache`를 수정으로 제시하지 않는다.

API 문서는 optional output의 lifetime과 비용을 경고하고, server wrapper는 허용 field와 max length를 제한할 수 있다. debug endpoint에서만 full score를 허용하거나 bounded top-N/digest로 대체할 수 있다. 사용자가 명시적으로 요구한 정보의 의미를 바꾸면 versioned contract가 필요하다.

**source-only audit의 정직한 한계와 실행 계획**

고정 source는 merge 순서와 가능한 cache path, loop mutation을 증명한다. 특정 GPU에서 static cache가 몇 퍼센트 빠르거나 output score가 정확히 어느 peak를 만든다는 것은 실행 증거가 필요하다. 이 장의 byte는 operand·retention 계산이며 allocator, kernel workspace, compile cache를 모두 포함하지 않는다.

승인된 실행 계획은 작은 deterministic model fixture에서 시작한다. config precedence를 서로 다른 값으로 채워 effective 결과를 캡처하고, cache off/on 두 step score를 비교한다. memory는 output scores off/on과 static/dynamic을 분리한다. exception processor로 terminal을 본다. 한 번에 모든 축을 바꾸지 않는다.

실제 대형 model에서는 artifact digest와 CUDA/library, dtype, device map, compile 상태를 고정한다. prompt length, batch, new tokens를 sweep하고 step별 allocated/reserved, score tuple byte, cache object length를 기록한다. source 예상과 observation이 다르면 actual class override와 selected path를 먼저 찾는다.

### 최종 배포 판정표

configuration terminal은 모든 대표 field의 source, effective, derived value와 consumer가 이어진다. numerical terminal은 cache off/on first two logits, processor 순서, stopping append order가 reference에 맞는다. memory terminal은 cache와 optional outputs의 slope·peak가 예산과 원장에 맞는다. lifecycle terminal은 정상, exception, cancel, distributed peer finish에서 terminal과 reference release가 닫힌다.

observability terminal은 full 민감 tensor를 상시 저장하지 않아도 effective config, mode, shape, cache length, retained output와 first divergence를 복원할 수 있다. fallback terminal은 cache implementation이나 compile path를 되돌려도 same sequence semantics와 output contract를 유지한다. 하나라도 닫히지 않으면 server wrapper의 기본 path로 승격하지 않는다.

최종 incident 문장은 구체적이어야 한다. “긴 generation OOM” 대신 “B=8,V=152064,FP32 output score가 step당 4,866,048 byte씩 256개 retained되어 약 1.16 GiB가 KV 1.0 GiB와 workspace에 겹쳤고, score tuple 생성부터 allocated slope가 갈렸다”고 쓴다. “cache가 틀림” 대신 “prefill cache는 같고 decode step 0의 cache_position이 5가 아니라 6이라 q/k rotation부터 갈렸다”고 쓴다.

이 언어가 다음 장의 비교 기준이 된다. continuous manager가 classic invocation을 여러 request의 장기 생명주기로 확장할 때도 effective option, current token slice, cache frontier, selected token, terminal의 의미는 보존해야 한다. 달라지는 것은 request table과 batch membership, allocator, cancellation owner다.

**독자가 직접 작성하는 one-step transition record**

마지막 연습에서는 prompt 길이 7, batch 1, max new tokens 3인 호출을 고른다. prefill 전 record에는 full IDs 7, model slice 7, mask 7, cache length 0, cache position 0~6을 적는다. prefill 뒤 selected token 42, cache length 7, full IDs append 뒤 8을 적는다. 첫 decode 전에는 model slice 1, mask 8, cache before 7, 새 position 7이어야 한다.

forward 뒤 cache length 8, selected token 9, full IDs 9가 된다. criterion은 append된 길이와 EOS를 읽는다. 아직 max new tokens 2개만 생성했으므로 계속한다. 다음 step에서 세 번째 token을 append하면 derived total cap 10이 되어 max-length criterion이 끝낸다. streamer에는 세 token과 terminal이 순서대로 가고 returned sequences shape는 `[1,10]`이다.

이 표에 `output_scores=true`를 더하면 각 step score tensor 세 개가 output object까지 산다. V=152,064, FP32, B=1이면 `3×152064×4=1,824,768 byte`, 약 1.74 MiB다. KV는 앞의 128 KiB/token fixture에서 total 10 token 약 1.25 MiB다. 작은 호출에서도 score retention이 KV보다 클 수 있다.

두 번째 fixture는 step 1에서 EOS가 나오게 한다. append 뒤 criterion이 finished를 표시하고 max cap까지 불필요하게 두 token을 더 생성하지 않아야 한다. `min_new_tokens=3`이 있으면 EOS processor가 EOS score를 언제 mask하는지와 stopping criterion이 언제 허용하는지를 구분한다. EOS가 선택된 뒤 무시하는 것과 선택 전에 금지하는 것은 다른 실행이다.

세 번째 fixture는 custom stop criterion이 step 2에서 exception을 던지게 한다. 마지막 committed sequence, streamer error/terminal, cache reference, optional score tuple 두 개의 lifetime을 적는다. 정상 fixture와 같은 cleanup edge를 통과하는지 확인한다. 이 한 장부가 config, preparation, cache, processing, stopping, output lifetime을 모두 연결한다.

**회고 질문과 handoff**

독자는 `GenerationConfig` 값을 어디서 찾았는가가 아니라 어떤 precedence로 effective 값이 됐는지 답한다. cache가 켜졌는가가 아니라 첫/full slice, cache logical·physical length와 position을 답한다. output을 반환했는가가 아니라 어떤 tensor가 caller lifetime까지 retained되는지 답한다. 끝났는가가 아니라 criterion, unfinished mask, streamer terminal과 distributed consensus를 답한다.

이 네 답이 있으면 classic loop의 성능과 correctness를 같은 state record에서 본다. prefill과 decode compute, cache capacity, processor work, optional retention, finished-row waste를 각각 분리한다. 다음 장에서는 같은 필드를 request table의 여러 row와 동적 batch membership으로 옮겨, compaction과 cancellation이 identity를 보존하는지 검증한다.

최종 rollout note에는 지원한 model class와 cache class, generation mode, custom callable 유무를 적는다. decoder-only greedy fixture가 통과했다고 encoder-decoder beam, assisted generation, remote override까지 검증됐다고 쓰지 않는다. 미검증 branch에는 첫 source entry와 필요한 fixture를 남긴다.

release를 올릴 때는 config field 목록 diff만 하지 않는다. preparation의 merge·validation, selected mode, processor/criterion ordering, cache factory와 kwargs update, output tuple lifetime을 같은 transition record로 다시 걷는다. source line이 이동해도 의미 edge가 유지되는지 확인한다.

rollback은 cache나 compile option만 원복하고 in-flight request의 effective config와 output contract를 바꾸지 않아야 한다. 장기 서버 wrapper라면 새 request부터 이전 path를 선택하고 진행 중 호출은 drain한다. exception과 cancellation fixture로 전환 중 terminal 중복과 reference 누수를 검사한다.

이 마지막 검토까지 닫혀야 “Transformers generate를 이해했다”는 문장이 옵션 암기가 아니라 재현 가능한 호출 생명주기 분석을 뜻한다.

마지막으로 동일 fixture를 cache fallback과 exception 경로에서도 반복한다. 정상 결과만 같은 것이 아니라 effective config, logical position, retained output, streamer terminal이 같은 의미를 유지해야 한다. 이 증거가 다음 release의 회귀 기준선이 된다.

미검증 branch와 한계도 같은 기록에 명시한다.
