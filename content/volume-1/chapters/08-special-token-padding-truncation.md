# 8장. special token, padding과 truncation의 함정

7장에서 채팅 템플릿은 role과 tool, reasoning, multimodal 자리를 모델이 읽을 문자열로 직렬화했다. 그러나 그 문자열이 곧 model input은 아니다. tokenizer는 delimiter와 BOS·EOS 같은 표지를 정수 ID로 바꾸고, 서로 길이가 다른 요청은 padding으로 한 tensor에 놓이며, context 상한을 넘는 입력은 truncation을 거친다. 이 세 단계는 모양만 다듬는 후처리가 아니라 모델이 실제로 보는 token과 position을 정한다.

이 장의 중심 질문은 **“tokenizer가 만든 ID 배열이 어떻게 의미를 잃지 않은 model input이 되는가?”**다. 답을 얻기 위해 special token을 특수한 문자열이 아니라 tokenizer·model config·embedding·generation 사이의 ID 계약으로 본다. padding은 빈칸 채우기가 아니라 attention과 position의 유효 영역을 표시하는 일로, truncation은 배열 길이 자르기가 아니라 protocol의 어떤 부분을 포기할지 결정하는 정책으로 읽는다.

고정 source는 Transformers 5.15.1 commit `550d7b3`, vLLM 0.27.1 commit `6e448d0`, SGLang 0.5.18 commit `71de97b`, llama.cpp commit `bb4caa7`이다. 이 장은 source에서 상태·분기·오류 경계를 복원하지만 model이나 server를 실행하지 않는다. 성능 수치나 특정 model의 출력 결과를 주장하지 않는다.

6장은 문자열을 token 경계로 나누는 원리를 설명했고 7장은 구조화 messages를 prompt protocol로 바꾸었다. 이 장은 그 출력에 special ID, padding, truncation을 적용해 최종 `[B,S]` 입력을 만든다. 9장은 여기서 만들어진 ID와 position을 embedding으로 바꾸고, 10장은 마지막 hidden state가 logits가 되는 과정을 이어받는다.

## 8.1 같은 문장을 batch에 넣었더니 결과가 달라졌다

한 요청을 단독으로 보냈을 때는 정상인데 길이가 다른 요청과 batch로 묶었더니 첫 token부터 달라진다고 하자. model weight, prompt text, sampling 설정은 같다. 운영자는 scheduler나 floating-point reduction을 의심하기 쉽다. 그러나 먼저 보아야 할 것은 batch의 `input_ids`, `attention_mask`, `position_ids`와 마지막 유효 token 위치다.

길이 5와 길이 8인 두 sequence를 폭 8에 맞추는 방법은 두 가지다. 오른쪽 padding은 짧은 sequence 뒤에 PAD 세 개를 붙인다.

```text
request A ids : [a0 a1 a2 a3 a4 P  P  P ]
request A mask: [ 1  1  1  1  1  0  0  0]
request B ids : [b0 b1 b2 b3 b4 b5 b6 b7]
request B mask: [ 1  1  1  1  1  1  1  1]
```

왼쪽 padding은 PAD를 앞에 둔다.

```text
request A ids : [ P  P  P a0 a1 a2 a3 a4]
request A mask: [ 0  0  0  1  1  1  1  1]
request B ids : [b0 b1 b2 b3 b4 b5 b6 b7]
request B mask: [ 1  1  1  1  1  1  1  1]
```

두 tensor는 유효 token 순서가 같지만 마지막 column의 의미가 다르다. decoder-only generation 코드가 `logits[:, -1, :]`에서 다음 token 분포를 고른다면 right-padded A의 마지막 위치는 PAD다. model이 attention mask를 받더라도 마지막 row의 hidden state가 “마지막 유효 token의 hidden state”로 자동 교체되는 것은 아니다. left padding에서는 마지막 column이 모든 sequence의 마지막 유효 token이라 이 선택과 잘 맞는다.

그렇다고 decoder-only model은 항상 left padding이라는 문장을 절대 법칙으로 외우면 안 된다. serving engine은 variable-length sequence를 padding 없이 packed representation으로 만들 수 있고, explicit gather index로 마지막 유효 row를 고를 수도 있다. architecture와 generation implementation이 positions/mask를 어떻게 만드는지 확인해야 한다. 중요한 것은 padding side 이름이 아니라 **다음 logits를 어느 row에서 읽는가**다.

### 최초 probe는 text가 아니라 네 tensor다

batch-only divergence를 좁힐 때 최종 text부터 비교하면 scheduler와 kernel까지 모든 층이 후보가 된다. 단독과 batch에서 다음 네 상태를 앞에서부터 비교한다.

```text
input_ids
attention_mask
position_ids 또는 cache_position
next-token logits를 읽은 row index
```

`input_ids`의 유효 구간이 다르면 tokenizer/template/truncation을 본다. 유효 IDs는 같지만 mask가 다르면 padding construction을 본다. mask는 같지만 positions가 다르면 model input preparation을 본다. 세 상태가 같고 selected row도 같은데 logits가 다를 때 scheduler/backend 가설로 내려간다.

반례는 PAD와 EOS가 같은 ID인 model이다. `input_ids == pad_token_id`만으로 mask를 추론하면 실제 prompt 안의 EOS와 padding을 구별하지 못한다. explicit attention mask가 없다면 자동 추론이 불가능하거나 모호할 수 있다.

Transformers 고정 source의 [`_prepare_attention_mask_for_generation`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L776-L806)은 input에 PAD가 있는지와 PAD가 EOS와 다른지를 검사해 mask 추론 가능성을 판단한다. 함수가 있다는 사실은 모든 모호함을 해결한다는 뜻이 아니다. PAD=EOS이면 caller가 올바른 mask를 제공해야 한다는 경계를 보여 준다.

## 8.2 special token은 문자열이 아니라 네 객체의 ID 계약이다

`<s>`, `</s>`, `<pad>`, `<unk>`는 눈으로 보면 문자열이다. 실제 serving에서는 최소 네 객체가 같은 의미를 합의해야 한다. tokenizer는 문자열을 ID로 바꾸고 special-token으로 취급한다. tokenizer/model config는 BOS·EOS·PAD·UNK field를 ID와 연결한다. embedding table은 그 ID를 유효한 row로 가져야 한다. generation code는 BOS로 시작하거나 EOS에서 끝내고 PAD로 finished row를 채운다.

이 계약에서 한 곳만 바뀌어도 실패 방식이 달라진다. tokenizer에 token string을 추가했지만 embedding row를 늘리지 않으면 out-of-range lookup이 날 수 있다. embedding을 늘렸지만 새 row를 학습하지 않았다면 실행은 되지만 의미를 얻지 못한다. EOS ID가 tokenizer와 generation config에서 다르면 model이 종료 marker를 내도 generation이 계속될 수 있다. PAD가 없다고 EOS를 PAD로 임시 지정하면 batching은 가능할 수 있지만 mask 추론이 모호해진다.

special token을 공연장의 표에 비유할 수 있다. 일반 token은 좌석 번호이고 special token은 출입·시작·종료를 지시하는 운영 표지다. 같은 숫자를 서로 다른 표지로 해석하면 운영이 꼬인다. 하지만 model에서는 special token도 embedding row를 고르고 logits 후보가 된다는 점에서 비유가 깨진다. “제어 토큰”이라고 해서 계산 밖에 있는 metadata가 아니다.

### BOS는 언제 들어가고 누가 넣는가

BOS(beginning of sequence)는 sequence 시작을 표시할 수 있다. 문제는 이름보다 insertion owner다. tokenizer의 `add_special_tokens=True`가 BOS를 추가할 수 있고, chat template가 literal BOS variable을 출력할 수 있으며, server가 pretokenized IDs 앞에 붙일 수도 있다. 둘 이상이 동시에 owner가 되면 BOS가 중복된다.

7장에서 본 SGLang의 render/encode 분리는 이 경계를 드러냈다. template가 role과 special token을 이미 포함하면 encode가 다시 special token을 더하지 않도록 해야 한다. 반대로 template가 BOS를 포함하지 않는데 `add_special_tokens=False`를 강제하면 BOS가 사라진다. 옵션 이름만 보지 않고 template output의 시작 token, tokenizer의 default insertion, server encode kwargs를 함께 본다.

Transformers generation은 input IDs가 없을 때 BOS를 이용해 시작 배열을 만들 수 있다. 고정 source의 [`_maybe_initialize_input_ids_for_generation`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L708-L751)은 input이 없는 조건과 batch size를 계산하고 BOS가 없으면 오류를 낸다. 일반 chat prompt가 이미 IDs를 제공하는 경로와 동일하게 읽어서는 안 된다. BOS의 효과는 “항상 첫 token으로 추가”가 아니라 caller가 input을 제공했는지와 tokenizer/template 정책에 따라 달라진다.

### EOS는 종료 후보이면서 prompt 안의 실제 token일 수 있다

EOS(end of sequence)는 generation stopping criterion에서 사용된다. 그러나 prompt history 안에 과거 turn 종료 marker로 EOS-like token이 있을 수 있고, 여러 EOS ID를 허용하는 config도 있다. `input_ids`에 EOS가 있다는 사실과 새로 생성된 token이 EOS라는 사실을 구분해야 한다.

vLLM의 `SamplingParams`는 engine이 primary EOS를 설정하고 generation config의 추가 EOS ID를 stop set에 합칠 수 있다. 고정 source의 [`update_from_generation_config`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/sampling_params.py#L649-L683)는 primary EOS와 additional IDs를 다루는 상태 변화를 보여 준다. `ignore_eos`나 stop token 설정이 있다면 실제 종료 집합은 tokenizer field 하나보다 넓거나 좁을 수 있다.

EOS가 선택되면 finished sequence의 이후 batch step을 어떻게 채울지도 필요하다. Transformers generation loop는 unfinished mask를 갱신하고 finished row에 PAD를 쓸 수 있다. [`generation loop의 PAD/EOS 처리`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2826-L2935)는 stop criterion과 output fill이 연결되는 좌표다. EOS와 PAD가 같더라도 “종료 사건”과 “tensor를 채우는 값”의 의미는 구별한다.

### PAD는 빈 token이 아니라 무시해야 할 위치를 표시하는 값이다

PAD의 embedding이 0이라는 보장은 없다. PAD row를 model에 넣고 attention mask를 잘못 만들면 실제 context로 읽힐 수 있다. loss 계산에서는 ignore index가 별도로 필요할 수 있지만 이 책은 inference를 다룬다. inference에서 PAD의 핵심 역할은 batch tensor의 모양을 맞추면서 유효 token 영역을 mask와 position에 전달하는 것이다.

PAD token이 없는 tokenizer에 새 PAD를 추가할지, EOS를 재사용할지는 model과 serving path에 따라 달라진다. 새 PAD를 추가하면 vocabulary와 embedding shape가 바뀐다. EOS 재사용은 weight 수정 없이 가능할 수 있지만 automatic mask inference를 모호하게 한다. 어느 선택도 이름 하나로 안전하지 않다.

Transformers tokenizer는 padding을 요청했는데 PAD가 없거나 ID가 유효하지 않으면 오류를 낸다. [`padding strategy 검증`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L2385-L2405)은 설정 누락을 조기에 드러낸다. 이 오류를 보고 무조건 `pad_token=eos_token`을 붙이는 것은 증상을 숨기는 응급조치일 수 있다. attention mask 제공, generation selected row, decode skip 정책까지 함께 검토해야 한다.

### UNK는 모르는 의미가 아니라 tokenizer의 실패 경로다

UNK(unknown token)는 vocabulary로 표현할 수 없는 입력을 하나의 ID로 대체하는 전통적 경로다. byte fallback이나 byte-level tokenizer는 unknown을 줄일 수 있지만 모든 tokenizer가 같은 정책을 쓰는 것은 아니다. UNK가 나왔다는 것은 model이 “이 단어를 모른다”고 판단했다는 뜻이 아니라 tokenizer가 해당 문자열을 더 세밀한 known pieces로 분해하지 못했다는 뜻이다.

UNK는 정보 손실을 만든다. 서로 다른 원문이 같은 UNK ID로 합쳐지면 decode로 원문을 복원할 수 없다. input validation, security filtering과 cache identity에서 raw text와 token IDs를 혼동하면 안 된다. 같은 IDs가 나왔다고 의미가 같은 입력이 아닐 수 있는 대표 예외다.

unknown 경로를 검사할 때는 화면 글자, Unicode code point, normalized text, token pieces, IDs를 순서대로 본다. 6장의 tokenizer 알고리즘이 canonical owner다. 이 장에서는 UNK ID가 embedding row와 config에 어떻게 연결되고, skip/decode 정책이 장애 증거를 숨기는지만 이어받는다.

## 8.3 added token을 붙이면 model vocabulary도 함께 바뀌는가

운영 중 새 role marker나 PAD를 추가해야 한다는 요청이 자주 나온다. tokenizer에 `add_special_tokens`를 호출하면 새 문자열이 하나의 special ID가 될 수 있다. 여기서 흔한 오해는 “tokenizer가 ID를 만들었으니 model도 이해한다”는 것이다.

tokenizer vocabulary 크기가 `V_t`, input embedding row 수가 `V_e`, output LM head row 수가 `V_o`라 하자. 모든 tokenizer ID가 input으로 들어갈 수 있으려면 최소한 `max_id < V_e`여야 한다. model이 그 token을 생성할 수 있으려면 output projection에도 대응 row가 필요하다. tied weights model에서는 input/output weight가 연결될 수 있지만 resize와 tying이 실제로 유지되는지 확인해야 한다.

### `add_special_tokens`는 tokenizer state만 바꾼다

Transformers 고정 source의 [`PreTrainedTokenizerBase.add_special_tokens`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L1102-L1160)는 special-token dictionary를 tokenizer에 추가하고, 문서 예시에서 model embedding resize가 별도 단계임을 밝힌다. API 이름에 model이 들어 있지 않다는 사실을 가볍게 보아서는 안 된다. tokenizer object와 model parameter는 다른 owner다.

`special_tokens_dict`의 field는 의미도 다르다. `pad_token`, `bos_token`, `eos_token`, `unk_token` 같은 known role과 `additional_special_tokens`가 있다. replacement/extension 정책에 따라 기존 additional token list를 덮거나 늘릴 수 있다. string이 이미 vocabulary에 있다면 새 ID가 생기지 않을 수도 있다. 반환된 added count와 final mapping을 기록해야 한다.

field→분기→상태→효과를 예로 들면 이렇다. `pad_token="<pad>"`를 추가한다. tokenizer는 기존 vocabulary/added vocabulary에서 문자열을 찾고 없으면 새 ID를 배정한다. tokenizer의 `pad_token_id`가 그 ID를 가리킨다. padding code는 batch 빈 위치에 이 ID를 쓴다. model embedding이 짧으면 lookup이 실패한다. embedding을 늘려도 새 row의 의미는 초기화 정책에 달려 있고 model이 PAD를 학습한 것은 아니다. 반증 관측은 final tokenizer length, PAD ID, embedding/LM-head shapes, added row 초기화와 attention mask다.

### embedding resize는 shape와 weight identity를 바꾼다

Transformers model의 [`resize_token_embeddings`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_utils.py#L2710-L2768)는 새 token 수와 optional multiple, initialization 정책을 받아 input embedding을 바꾸고 필요하면 weight tying을 다시 처리한다.

내부 [`_resize_token_embeddings`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_utils.py#L2769-L2835)는 old/new embedding과 output embedding 관련 상태를 다룬다.

이 호출은 값 하나를 config에 쓰는 일이 아니다. parameter shape가 바뀌고 checkpoint identity가 달라진다. tensor parallel shard, quantized weight, compiled graph, adapter vocabulary dependency가 있으면 runtime resize가 지원되지 않거나 추가 변환을 요구할 수 있다. serving replica 하나에서만 즉흥적으로 resize하면 같은 model name 아래 서로 다른 artifact가 생긴다.

새 row 초기화도 correctness 계약이다. mean resizing 같은 정책은 기존 embedding 분포를 이용할 수 있지만 새 token의 학습된 의미를 만들지는 않는다. role delimiter를 새로 추가하고 random/mean row로 두면 tokenizer는 원자적으로 처리하지만 model이 해당 protocol을 이해하지 못할 수 있다. model training artifact에 없던 control token을 serving에서 추가하는 것은 단순 호환 patch가 아니다.

### tokenizer length, config vocab size, embedding rows를 한 장부에 둔다

배포 전에 다음 관계를 산문으로 설명할 수 있어야 한다.

```text
tokenizer가 반환할 수 있는 최대 ID
< input embedding row 수
output 후보로 허용할 ID
< LM head row 수
special-token config의 각 ID
∈ tokenizer와 필요한 model row 범위
```

`len(tokenizer)`와 `tokenizer.vocab_size`가 backend에 따라 added vocabulary 포함 여부에서 다를 수 있으므로 이름만 보고 같다고 가정하지 않는다. 실제 ID map과 embedding shape를 본다. model config의 `vocab_size`가 serialization 시점 값과 runtime resized shape 중 무엇을 반영하는지도 확인한다.

문제 장면을 생각해 보자. tokenizer update 뒤 일부 request만 embedding index error를 낸다. new token을 포함하지 않는 입력은 old ID 범위 안에서 정상이고, added token을 만난 입력만 새 ID를 만든다. 평균 health check는 통과할 수 있다. fixed fixture에 모든 special/added token을 각각 넣고 mapping과 model row range를 정적으로 검증해야 하는 이유다.

## 8.4 left/right padding은 attention과 position을 함께 바꾼다

padding을 tensor의 빈칸이라고 부르면 attention mask만 맞으면 끝이라고 생각하기 쉽다. 실제로는 유효 token이 어느 column에 놓이는지, position이 어떻게 매겨지는지, cache가 어느 위치부터 시작하는지, 다음 logits를 어느 row에서 읽는지가 함께 바뀐다.

짧은 sequence `[A,B,C]`를 길이 5에 right padding하면 `[A,B,C,P,P]`, left padding하면 `[P,P,A,B,C]`다. mask가 PAD를 완전히 가리더라도 raw column index는 달라진다. absolute position embedding을 단순 column index로 만들면 A의 position이 0에서 2로 바뀐다. 많은 decoder implementation은 attention mask의 cumulative sum으로 유효 position을 0,1,2로 재구성하거나 cache position을 별도로 넘기지만 모든 경로가 같지는 않다.

RoPE에서도 “PAD를 mask했으니 position은 무관하다”는 설명은 부족하다. 유효 token에 적용되는 rotary position이 같아야 단독 실행과 padded batch를 비교할 수 있다. PAD 앞쪽을 포함한 raw column index를 그대로 쓰는지, mask에서 유효 position을 계산하는지, continuous batching이 padding 없는 packed positions를 만드는지를 확인한다.

### tokenizer field는 batch builder의 분기를 바꾼다

Transformers tokenizer의 기본 class field는 `padding_side`와 `truncation_side`를 가진다. 고정 source의 [초기화·validation](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L1035-L1065)은 kwargs에서 side를 읽고 `left/right` 외 값을 거부한다. field가 유효하다는 것은 model path에 적합하다는 뜻이 아니다. validation은 문자열 enum만 확인한다.

실제 padding mutation은 [`_pad`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L2764-L2836)에서 확인할 수 있다. right branch는 required input 뒤에 PAD를 넣고 attention mask와 token type/special mask의 뒤를 채운다. left branch는 앞에 넣는다. 어느 auxiliary field가 함께 이동하는지 읽어야 한다. custom tokenizer가 base implementation을 override할 가능성도 있으므로 concrete class를 확인한다.

field→분기→상태→효과 사슬은 다음처럼 닫힌다. `padding_side="left"`가 tokenizer state에 저장된다. batch padding이 left branch를 선택한다. `input_ids`, attention/special/token-type masks의 앞에 fill이 들어간다. model input preparation은 이 mask로 positions 또는 cache position을 만든다. decoder-only generation의 마지막 column이 유효 token이 되므로 selected-logits row와 잘 맞을 수 있다. 반증 관측은 단독/배치의 유효 IDs, positions, selected row와 첫 logits다.

### attention mask는 PAD ID의 별명이 아니다

attention mask는 어느 key/value 위치를 볼 수 있는지를 표시하는 실행 상태다. PAD ID는 input tensor의 fill value다. 흔히 `mask = input_ids != pad_id`로 만들지만 두 개념은 동치가 아니다. PAD와 EOS가 같은 ID일 수 있고, 실제 content에 pad-like ID가 들어갈 수도 있으며, packed input은 PAD 없이 길이 metadata를 사용할 수 있다.

Transformers generation의 mask 추론 코드가 PAD와 EOS가 다른 조건을 확인하는 이유가 여기 있다. PAD=EOS이면 ID 비교만으로 실제 EOS content와 padding을 가를 수 없다. caller가 explicit mask를 제공했는지 확인해야 한다. 경고가 없다는 사실은 mask가 맞다는 증거가 아니다. 해당 path가 warning 조건을 건너뛰었거나 engine이 다른 representation을 사용할 수 있다.

mask shape도 중요하다. `[B,S]` padding mask가 model 내부에서 causal mask와 결합되어 `[B,1,Q,K]` 또는 backend-specific metadata가 될 수 있다. left padding에서 query positions와 key positions의 offset이 일치해야 한다. attention kernel을 조사하기 전에 model-level mask semantics를 검증해야 한다.

### right-padding warning을 원인 판정으로 쓰지 않는다

Transformers 고정 source의 [`generate` 입력 preparation 경고 구간](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2510-L2550)은 decoder-only model에서 batch input 마지막 위치의 PAD를 검사하고 left padding을 권한다. source 주석은 PAD가 EOS/BOS와 같을 때 false positive 가능성을 고려한다.

이 경고는 유용하지만 root cause 판정기가 아니다. PAD=EOS인 prompt가 실제 EOS로 끝났을 수 있다. engine이 마지막 유효 index를 별도로 gather하면 right padding도 정확할 수 있다. 반대로 left padding을 썼어도 position_ids가 raw column index라면 결과가 달라질 수 있다. warning은 “selected row와 padding semantics를 확인하라”는 first probe다.

### padding 비용은 유효 token과 padded token을 나누어 센다

dense batch가 폭 `S_max`로 계산되면 유효 token 수 `ΣL_i`보다 `B×S_max`에 가까운 work를 만들 수 있다. optimized attention이나 packed sequence가 PAD work를 건너뛸 수 있지만 model의 모든 op가 자동으로 건너뛴다고 단정하지 않는다. embedding lookup, norm, projection과 attention backend의 representation을 따로 본다.

padding waste의 간단한 비율은 다음과 같다.

\[ \rho_{pad}=1-\frac{\sum_i L_i}{B\,S_{max}} \]

길이 `[128,128,128,2048]`을 한 dense batch로 묶으면 유효 token은 2,432이고 padded slot은 8,192다. `ρ_pad≈0.703`이다. 이 값이 latency 낭비율과 같다는 뜻은 아니다. backend가 varlen을 쓰거나 긴 sequence가 attention의 대부분을 지배할 수 있다. 그러나 length-aware batching 가설을 세우는 출발점은 된다.

성능 때문에 padding side를 바꾸는 경우 correctness를 먼저 지킨다. same valid IDs와 effective positions, masks, first logits를 비교한 뒤 padded work와 batch throughput을 본다. 결과 text만 같다고 충분하지 않다. greedy first token은 같아도 작은 logits 차이가 sampling과 긴 generation에서 커질 수 있다.

## 8.5 truncation은 무엇을 버릴지 결정하는 정책이다

7장은 chat template가 role/tool/multimodal protocol을 만들고 message-level truncation이 그 문법을 보존해야 한다고 설명했다. 이 장에서는 tokenizer와 server가 실제 token 배열을 자를 때 생기는 ID·position·generation 문제에 집중한다.

right truncation은 sequence 끝을 버린다. chat prompt에서는 가장 최근 user question이나 assistant generation suffix가 끝에 있을 가능성이 높다. left truncation은 시작을 버린다. system instruction, BOS와 tool schema가 앞에 있을 가능성이 높다. 어느 쪽도 의미적으로 자동 안전하지 않다.

### context 상한은 model max length 하나가 아니다

실제 prompt budget에는 model positional limit, server maximum, reserved output, multimodal expansion, prefix/cache alignment와 backend restriction이 겹친다. `tokenizer.model_max_length`가 매우 큰 sentinel이거나 artifact metadata와 다를 수 있다. server가 더 작은 상한을 둘 수도 있다.

입력 길이를 `S_in`, 예약 output을 `S_out`, model/server가 허용하는 전체 길이를 `S_cap`이라 하면 기본 불변식은 `S_in + S_out ≤ S_cap`이다. 그러나 image placeholder가 processor 뒤 `Δ_mm` 위치를 더 만들고 assistant suffix가 `S_suffix`라면 raw text token만 세어서는 안 된다.

\[ S_{model}=S_{rendered}+\Delta_{mm}+S_{special}-S_{removed} \]

각 항의 owner를 기록한다. template가 넣은 suffix와 delimiter, tokenizer가 자동 추가한 BOS/EOS, processor expansion, server truncation이 서로 중복되지 않아야 한다. 계산식이 복잡한 이유가 아니라 pass ownership이 흩어져 있기 때문에 장부가 필요하다.

### auto-truncate는 오류 정책을 상태 mutation으로 바꾼다

server option이 “자동 잘라내기 허용”을 켠다고 하자. 입력 문자열이 길다는 validation error가 사라지는 대신 request의 token state가 바뀐다. 어떤 side와 alignment로 잘랐는지, 원본 ID와 final ID를 어디에 보존하는지, cache lookup이 truncation 전후 어느 배열을 쓰는지가 downstream correctness를 결정한다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- SGLang scheduler source에는 `allow_auto_truncate`와 untruncated fill IDs를 소비하는 경계가 있다.
- 고정 source의 [`scheduler.py` request preparation 구간](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L2580-L2625)과 [untruncated IDs·matched prefix를 다루는 구간](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L2680-L2720)을 함께 읽으면 truncation이 cache/scheduler state와 만나는 지점을 찾을 수 있다.

exact behavior는 surrounding function과 request type을 확인해야 한다.

field→분기→상태→효과를 적어 보자. `allow_auto_truncate=true`가 serving config에 들어간다. request length validation이 reject 대신 truncation branch를 허용한다. request는 full/untruncated IDs와 실제 fill IDs 또는 truncate size를 구분해 보존할 수 있다. scheduler와 prefix cache는 final compute range를 만든다. 사용자는 성공 응답을 받지만 일부 input 의미가 사라진다. 반증 관측은 raw/rendered/final IDs 길이, removed source span, cache matched length와 usage다.

### system/tool/multimodal 문법은 token 하나씩 자를 수 없다

system block의 start/end delimiter, tool JSON의 braces/tags, multimodal placeholder 묶음이 중간에서 잘리면 tensor 길이는 유효해도 protocol은 불완전하다. 특히 tokenizer-level truncation은 message source 구조를 모르므로 원자성을 보장할 수 없다.

따라서 순서는 message-level selection→render→tokenize→final safety truncation이 되어야 한다. final safety truncation이 발생했다면 이는 정상 policy가 message budget을 맞추지 못했다는 신호다. silent success로 숨기기보다 trace에 failure category를 남기거나 request를 거부하는 설계가 필요할 수 있다.

멀티모달은 두 길이를 맞춰야 한다. text token sequence에 placeholder가 남았는지와 image feature batch가 함께 남았는지다. placeholder만 잘렸는데 pixel tensor가 남으면 count mismatch가 생긴다. image를 message-level로 제거하면 이를 참조하는 text와 tool metadata도 함께 처리해야 한다.

### truncation side를 바꾼 A/B는 의미가 다른 실험이다

left와 right truncation을 성능 option처럼 A/B 비교하면 서로 다른 prompt를 model에 넣게 된다. output 품질이나 latency가 달라져도 side 자체의 runtime 비용인지 보존된 content 차이인지 구분할 수 없다. 먼저 synthetic sequence처럼 의미가 통제된 fixture로 implementation behavior를 확인하고, 실제 chat에서는 product policy 관점에서 평가한다.

반증 fixture는 context limit 주변에서 한 token씩 길이를 바꾸고 BOS/system/tool/user/generation suffix의 source span이 어디까지 남는지 기록한다. tokenizer ID array만 보지 말고 source message→rendered byte→token range map을 사용한다. 7장의 compiler trace를 이어받는 이유다.

복구 종료 조건은 단순히 exception이 사라지는 것이 아니다. final input이 budget 아래 있고, required protocol atom이 완전하며, removed content가 policy와 일치하고, cache/usage가 final IDs를 설명하며, boundary fixture가 같은 규칙으로 처리되어야 한다.

## 8.6 generation과 detokenization에서 special ID는 다시 의미를 바꾼다

input 단계에서 BOS/EOS/PAD는 model context와 mask를 만들었다. generation 단계에서는 EOS가 종료 사건이 되고 PAD가 finished rows의 fill value가 되며, decode 단계에서는 `skip_special_tokens`가 화면에서 이 ID들을 숨길 수 있다. 같은 ID가 pipeline 위치에 따라 다른 역할을 수행한다.

### generated EOS와 stop string을 구별한다

EOS ID는 sampler가 고른 정수 token이다. stop string은 decoded byte/text stream에서 찾는 문자열일 수 있다. stop string이 여러 token 경계를 가로지를 수 있고, EOS-like literal이 일반 token pieces로 나올 수도 있다. 이 장은 special ID 계약만 다루며 stop automaton의 전체 의미는 11장이 소유한다.

여기서 필요한 연결은 종료 identity다. engine의 primary/additional EOS set, request의 stop token IDs, `ignore_eos`, output processor의 stop strings를 따로 기록한다. “EOS를 무시한다”는 option이 primary EOS만 무시하는지 additional stop ID까지 바꾸는지 source에서 확인한다.

vLLM의 `SamplingParams` source는 EOS를 internal stop set에 넣는 mutation을 보여 준다. option help만 보고 종료 의미를 설명하지 말고 engine이 tokenizer EOS를 언제 주입하고 generation config의 IDs를 어떻게 합치는지 본다. 실제 selected token과 finish reason을 함께 관측해야 한다.

### `skip_special_tokens`는 계산을 바꾸지 않고 증거를 숨길 수 있다

decode의 `skip_special_tokens=True`는 model이 special ID를 생성하지 못하게 하는 option이 아니다. token IDs는 이미 선택되었고 decode가 text로 바꿀 때 일부를 건너뛴다. 따라서 visible output이 빈 문자열이라고 model이 아무 token도 생성하지 않았다고 결론 내릴 수 없다.

장애 조사에서는 raw output IDs와 decoded-with-special, decoded-without-special을 구분한다. role marker나 EOS가 예상 밖으로 생성되었는데 skip이 이를 숨기면 final text만으로는 prompt/generation 경계 오류를 찾기 어렵다. production response에서는 숨기더라도 protected trace나 재현 bundle에는 ID-level evidence를 남긴다.

llama.cpp의 common helper는 tokenize/detokenize 경계를 명확히 보여 준다. [`common_tokenize`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/common.cpp#L1836-L1868)는 `add_special`과 `parse_special`을 구분해 `llama_tokenize`로 내려간다.

[`common_detokenize`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/common.cpp#L1890-L1912)는 special rendering flag와 byte piece 수집을 다룬다. `parse_special`과 `special`은 방향이 다른 계약이며 이름이 비슷하다고 같은 boolean으로 묶어서는 안 된다.

### decode는 encode의 단순 역함수가 아니다

normalization, UNK, byte fallback, cleanup, special-token skipping 때문에 `decode(encode(text)) == text`가 항상 성립하지 않는다. added token의 left/right strip 속성이나 tokenizer cleanup이 공백을 바꿀 수 있다. output streaming은 완성되지 않은 UTF-8 byte piece를 다음 token까지 보류할 수도 있다.

이 사실은 billing과 stop, cache identity에 영향을 준다. generated token 수와 visible character 수는 다른 단위다. special ID가 skip되거나 byte pieces가 합쳐져 한 문자가 되면 1:1 대응이 없다. 10장의 logits는 ID 선택 전 점수를, 11장은 selected IDs가 stop과 streaming으로 commit되는 과정을 다룬다.

## 8.7 네 구현에서 같은 계약을 끝까지 따라간다

이제 이름이 같은 option을 나란히 놓는 대신, 한 요청의 ID가 어느 함수에서 태어나고 어느 상태에 기록되는지 따라가 보자. 네 구현은 모두 “문자열을 token으로 바꾸어 생성한다”고 말할 수 있지만, padding과 truncation의 책임 경계가 다르다. Transformers는 일반적인 tensor batch를 만드는 tokenizer와 generation mixin의 경계가 선명하다. vLLM과 SGLang은 request별 sequence를 scheduler가 다루므로 dense batch의 오른쪽 PAD를 반드시 만들 필요가 없다. llama.cpp는 GGUF metadata와 vocabulary object가 model별 special ID의 근거가 된다. 이 차이를 무시하면 한 구현의 처방을 다른 구현에 그대로 옮겨 장애를 키운다.

### Transformers: tokenizer가 만든 모양을 generation이 해석한다

Transformers에서 첫 번째 추적점은 model forward가 아니라 tokenizer 호출이다. [`PreTrainedTokenizerBase`의 초기화와 side 검증](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L1035-L1065)은 `padding_side`와 `truncation_side`가 허용된 방향인지 확인한다. 이 값은 단순 표시용 config가 아니다. 이후 `_pad`가 어느 쪽에 pad ID와 0 mask를 붙일지, truncation helper가 어느 문맥을 버릴지 결정하는 policy input이다.

실제 padding mutation은 [`_pad`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L2764-L2836)에서 읽을 수 있다. 필요한 길이 차이를 계산하고, attention mask와 token type IDs와 special token mask를 input IDs와 같은 방향으로 확장한다. 여기서 중요한 것은 “PAD 하나를 붙인다”가 아니라 서로 관련된 여러 배열의 좌표계를 동시에 바꾼다는 점이다. custom tokenizer나 전처리기가 `input_ids`만 수동 padding하면 겉보기 shape는 맞아도 mask와 position의 의미가 어긋난다.

호출자는 padding을 요구하기 전에 pad token이 존재하는지도 확인해야 한다. [`pad token 검증 경로`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L2385-L2405)는 padding 요청과 pad ID의 부재를 연결해 실패시킨다. 이 실패를 피하려고 무조건 `pad_token = eos_token`을 대입하는 예제가 많지만, 그것은 기술적 필수 조건이 아니라 선택한 표현 계약이다. causal LM 추론에서 mask가 PAD 위치를 완전히 가리고 종료 판정이 prompt의 PAD를 읽지 않는다는 조건 아래 유용할 수 있지만, label masking이나 sequence classification까지 같은 tokenizer를 공유하면 의미가 달라진다.

generation으로 들어가면 special ID들은 device tensor로 정규화된다. [`special token tensor 준비 경로`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L776-L806)는 PAD와 EOS 관계를 이용해 attention mask를 추론할 수 있는지 판단한다. PAD ID와 EOS ID가 같으면 input ID만 보고 어떤 EOS가 실제 내용이고 어떤 위치가 padding인지 식별할 수 없다. caller가 attention mask를 명시하지 않은 순간 이 모호성이 실행 의미가 된다.

따라서 `pad=eos`가 항상 잘못이라는 결론도, 항상 안전하다는 결론도 틀렸다. 안전 조건은 explicit mask가 있고, prompt 끝의 EOS를 content로 보존해야 하는 경로와 batch fill을 구별하며, 종료 판정이 generated region을 기준으로 작동하는지다. 테스트는 ID equality만 assert할 것이 아니라 동일 prompt의 single 실행과 mixed-length batch 실행에서 첫 generated token logits가 허용 오차 안에 있는지 비교해야 한다.

decoder-only generation의 오른쪽 padding 경고는 이 계약 위반을 조기에 드러낸다. [`right-padding 검사`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2510-L2550)는 batch 마지막 열의 pad 여부를 보고 caller에게 left padding을 요구한다. 이유는 attention mask 하나로 과거 PAD를 가리는 것과, generation loop가 “현재 step의 마지막 위치”에서 어떤 logits를 집어드는 것이 별개의 문제이기 때문이다. model이 PAD를 attend하지 않아도 선택된 row가 PAD query의 row라면 결과는 달라질 수 있다.

token을 추가하는 순간에는 tokenizer와 model이 서로 다른 객체임을 기억해야 한다. [`add_special_tokens`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L1102-L1160)는 tokenizer vocabulary와 role mapping을 바꾸지만 model weight를 자동으로 키운다는 보장은 없다.

[`resize_token_embeddings`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_utils.py#L2710-L2768)는 embedding row 수를 바꾸고 tied weights 처리로 이어진다. 즉 “token이 encode된다”와 “그 ID를 model이 계산할 수 있다” 사이에 명시적인 배포 단계가 있다.

새 row가 생겼다는 사실만으로 학습된 의미가 생기지는 않는다. resize initialization은 수치적으로 유효한 vector를 제공할 뿐이다. 새 role marker가 model 행동을 안정적으로 제어하려면 training이나 adapter가 그 marker를 배운 적이 있어야 한다. 추론 서비스가 임의로 `<|assistant|>`를 추가해 out-of-range를 고쳤더라도, 출력 품질 문제가 해결되지 않는 이유가 여기 있다. 7장의 chat template 계약과 이 장의 vocabulary 계약이 만나는 지점이다.

### vLLM: PAD 배열보다 request 상태와 scheduler 경계를 본다

vLLM을 Transformers와 비교할 때 가장 흔한 오해는 GPU batch라면 반드시 `[batch, max_len]` 모양의 PAD-filled input이 존재할 것이라는 가정이다. continuous batching engine은 서로 다른 길이의 sequence가 가진 logical token block과 scheduler metadata를 묶어 실행할 수 있다. kernel 내부의 block/table layout과 API에서 보이는 padding side는 같은 개념이 아니다. 따라서 batch-only divergence를 조사할 때 먼저 실제 engine이 dense padding을 만들었는지 확인해야 한다.

OpenAI-compatible chat request에는 tokenizer와 출력 표현을 바꾸는 option이 함께 들어온다. [`ChatCompletionRequest의 관련 필드`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/protocol.py#L274-L328)는 `skip_special_tokens`, prompt truncation, 허용 token ID, special-token 추가 같은 요청 표면을 정의한다. 이 필드가 존재한다는 사실보다 중요한 질문은 어느 것이 prompt compiler에 전달되고 어느 것이 sampling/output processor에 전달되는가다.

예를 들어 prompt truncation은 text를 보기 좋게 자르는 formatter option이 아니다. request가 tokenizer 단계에서 몇 ID까지 남길지를 제한하여 cache key, prefill work, usage count를 동시에 바꾼다. [`API utility의 길이 제한 경로`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/serve/utils/api_utils.py#L176-L179)는 명시된 truncate limit이 허용 범위를 넘는지 검사한다. limit이 유효하다는 것과 system/tool grammar를 보존한다는 것은 다른 보장이다. 전자는 숫자 validation이고 후자는 application policy다.

`allowed_token_ids`도 special token 문제를 우회하는 만능 열쇠가 아니다. [`SamplingParams 검증`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/sampling_params.py#L861-L883)은 전달된 ID 집합을 validation하고 logits processing에 사용할 상태로 만든다. tokenizer에 없는 ID의 의미를 창조하거나 model embedding을 resize하지 않는다. vocabulary 범위와 model artifact가 먼저 일치해야 그 위에서 sampling constraint가 의미를 갖는다.

EOS 처리도 request 입력을 그대로 보존하는 수동적 과정이 아니다. [`SamplingParams의 종료 상태 갱신`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/sampling_params.py#L649-L683)은 model/tokenizer에서 온 EOS 정보를 internal stop-token set과 연결한다. 그래서 장애 bundle에는 사용자가 보낸 JSON만 저장해서는 부족하다. validation과 generation-config merge 뒤의 effective EOS set, `ignore_eos`, stop IDs를 함께 남겨야 한다.

출력에서는 [`skip_special_tokens와 공백 보존 필드`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/sampling_params.py#L295-L297)가 detokenization 표현을 바꾼다. 이 option을 토글해 답변 text가 달라졌다고 logits가 달라졌다고 말할 수 없다. raw output token IDs가 동일한지 먼저 비교하고, 동일하다면 차이는 decode layer에 있다. 반대로 IDs부터 다르면 prompt IDs, sampling state, scheduler 실행을 위로 거슬러 올라간다.

vLLM 검증의 핵심 fixture는 padding side를 무작정 바꾸는 것이 아니라 request isolation이다. 짧은 prompt A를 단독으로 실행한 trace와, 훨씬 긴 prompt B와 같은 scheduler window에 들어간 A의 trace를 비교한다. A의 final prompt IDs, computed-token count, first-step selected logits identity, output IDs가 같다면 dense-padding 가설은 약해진다. 다르면 prefix cache hit, multimodal placeholder, speculative path, sampling seed 등 scheduler가 공유하거나 변형한 상태를 차례로 좁힌다.

### SGLang: 잘린 원본과 실행용 ID를 별도 상태로 읽는다

SGLang은 long-input 처리와 cache-aware scheduling을 함께 읽어야 한다. tokenizer manager에서 입력을 만들고 scheduler가 prefix/cache 상태를 계산하며 detokenizer manager가 incremental output을 조립한다. “자동 truncation을 켰다”는 한 문장만으로는 어느 시점의 어느 배열이 잘렸는지 알 수 없다.

요청 표면의 `allow_auto_truncate`는 [`tokenizer manager의 요청 전달 경로`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tokenizer_manager.py#L1207-L1228)를 거쳐 실제 입력 처리 정책이 된다. [`long input utility`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/utils.py#L194-L207)는 길이 초과 때 허용 여부에 따라 오류 또는 절단을 선택한다. 이 분기는 availability와 semantic integrity 사이의 정책 결정이다.

자동 절단이 서비스 성공률을 높일 수는 있다. 그러나 가장 오래된 token을 버리는 행위가 system policy, few-shot delimiter, tool call의 여는 marker를 제거하면 정상 HTTP 응답이 더 위험한 오답을 만든다. 그러므로 production default를 정할 때는 “긴 요청을 거부하면 UX가 나쁘다”와 “조용히 의미를 바꾸면 audit가 불가능하다”를 함께 평가해야 한다. 대화 요약이나 구조 단위 제거가 가능한 application compiler가 engine의 blind truncation보다 먼저 개입하는 편이 대개 낫다.

scheduler는 실행용으로 축약된 sequence만 보고 끝나지 않을 수 있다. [`Req의 전체·절단 전 ID 상태`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_batch.py#L831-L869)와 [`full untruncated IDs 유지 경로`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_batch.py#L1269-L1303)를 읽으면 cache와 grammar 같은 후속 기능 때문에 원본 identity가 왜 필요한지 보인다.

debugger는 `origin_input_ids`, 실행 시점 IDs, cache prefix 이후 남은 IDs를 섞지 않아야 한다.

EOS 판정은 scheduler request 상태에 붙는다. [`EOS match 경로`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_batch.py#L1500-L1501)처럼 작은 비교도 어떤 EOS 집합과 어떤 token을 비교하는지 주변 상태를 읽어야 의미가 드러난다. 추가 special EOS, stop IDs, grammar termination이 서로 다른 finish reason으로 합쳐질 수 있으므로 단일 boolean만 로그에 남기면 원인을 잃는다.

detokenization은 batch 안에서도 option별로 실행군을 나눌 수 있다. [`detokenizer manager의 grouping`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/detokenizer_manager.py#L233-L280)은 skip-special과 space cleanup 같은 설정이 같은 요청을 묶어 decode하는 조건임을 보여 준다.

[`incremental decode 경로`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/detokenizer_manager.py#L328-L355)는 streaming text가 전체 ID 배열의 단순한 매 step decode가 아닌 이유를 드러낸다.

따라서 SGLang에서 “마지막 글자가 사라졌다”는 신고를 받으면 sampling부터 의심하지 않는다. raw IDs가 존재하는지, decode window와 read offset이 어디인지, unfinished byte가 보류됐는지, special skipping group이 맞는지 확인한다. raw IDs가 없다면 scheduler/stop으로, IDs는 있는데 text만 없다면 detokenizer state로 조사 범위를 좁힐 수 있다.

### llama.cpp: GGUF metadata에서 호출 flag까지 이어 본다

llama.cpp에서는 tokenizer 역할 ID가 외부 JSON 한 장에만 있지 않다. [`llama-arch의 GGUF tokenizer key 정의`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-arch.cpp#L357-L370)는 BOS, EOS, UNK, PAD token ID와 자동 추가 flag가 model artifact metadata의 일부임을 보여 준다. 잘못 변환된 GGUF는 weight가 정상이어도 prompt boundary를 다르게 만들 수 있다.

vocabulary가 tokenization 앞뒤에 BOS/EOS를 넣는 과정은 [`llama-vocab의 add-special 처리`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-vocab.cpp#L571-L595)에서 확인할 수 있다. 이미 BOS/EOS가 들어 있는 sequence와 자동 추가 flag가 만날 때 duplicate를 어떻게 피하는지 읽어야 한다. template가 marker text를 출력했다는 사실과 tokenizer가 special ID를 자동 추가한다는 사실은 독립된 층이다.

public vocabulary accessor인 [`special token ID 조회 경로`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-vocab.cpp#L3971-L4051)는 application이 hard-coded 숫자 대신 loaded model의 계약을 조회해야 하는 이유를 보여 준다. 같은 family 이름을 가진 변환물이라도 ID mapping이나 add flag가 다를 수 있다. 로그에는 “EOS=2라고 가정”이 아니라 model fingerprint와 accessor 결과를 남긴다.

상위 helper에서는 `add_special`과 `parse_special`을 따로 전달한다. 앞서 본 `common_tokenize`에서 전자는 BOS/EOS 같은 자동 경계 추가를, 후자는 입력 문자열 속 special-looking piece를 실제 special ID로 해석할지를 제어한다. untrusted user text가 role marker 문자열을 포함할 때 `parse_special`을 잘못 켜면 prompt injection의 표면이 달라질 수 있다. 반대로 trusted template marker를 일반 text로만 tokenize하면 학습 때의 단일 special ID 대신 여러 ordinary pieces가 들어갈 수 있다.

detokenize의 `special` flag는 또 다른 방향이다. 생성된 special ID를 사람이 보는 text에 표시할지 결정한다. 따라서 세 flag는 “special 처리”라는 하나의 switch로 추상화하면 안 된다. 입력 경계 자동 추가, 문자열의 marker 파싱, 출력 marker 표시라는 세 질문으로 API를 설계해야 한다.

model을 다시 저장하거나 변환할 때 metadata 보존도 검증 대상이다. [`model saver의 tokenizer metadata 기록`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model-saver.cpp#L341-L351)은 special ID와 add flag가 artifact에 다시 쓰이는 지점을 보여 준다. 원본 HF tokenizer의 config와 변환된 GGUF를 비교할 때 token strings뿐 아니라 ID, add flags, chat template까지 diff해야 한다.

speculative decoding을 붙이면 target과 draft tokenizer 계약이 더 엄격해진다. [`target/draft vocabulary compatibility 검사`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/speculative.cpp#L84-L95)는 BOS/EOS 같은 기본 경계가 맞지 않는 조합을 왜 경계해야 하는지 보여 준다. draft가 제안한 정수 ID를 target이 다른 token으로 해석하면 acceptance 확률 이전에 의미 identity가 깨진다. token count만 같거나 vocabulary size만 같다는 검사는 충분하지 않다.

네 구현을 관통하는 결론은 단순하다. Transformers에서는 dense batch tensor의 정렬과 generation row selection이 중요하고, serving engine에서는 request별 logical sequence와 scheduler state가 중요하다. SGLang은 절단 전·후 상태를 명시적으로 추적하며, llama.cpp는 artifact metadata와 flag 조합이 경계를 결정한다. 공통 질문은 “PAD가 왼쪽인가”가 아니라 “최종 prompt ID와 그 좌표·종료·표시 계약을 누가 언제 확정했는가”다.

## 8.8 장애를 재현하고 계약을 닫는 워크북

이 절의 목적은 모든 incident에 똑같은 체크리스트를 던지는 것이 아니다. 관측된 현상에서 가장 싼 반증 실험을 골라 원인 층을 줄이는 방법을 익히는 것이다. 재현 bundle의 최소 단위는 model/tokenizer artifact fingerprint, template revision, 원문 message 구조, rendered prompt의 hash와 안전한 발췌, final input IDs, mask/position 요약, effective truncation과 special-token config, output IDs와 finish reason이다. 민감한 원문을 저장할 수 없다면 span 길이와 role/tool 경계, salted hash를 남겨 identity를 보존한다.

### 사건 1: special token을 추가한 뒤 GPU에서만 index 오류가 난다

개발자는 tokenizer에 PAD를 추가했고 encode 결과도 정상이라 배포했다. 짧은 단일 요청은 PAD를 사용하지 않아 성공한다. 길이가 다른 요청이 batch가 되는 순간 새 pad ID가 input에 들어가고 embedding lookup이 vocabulary row 밖을 읽으려 하면서 실패한다. 증상이 “batch에서만 GPU index error”라 scheduler bug처럼 보이지만, 원인은 artifact 원자성 위반이다.

첫 반증은 tokenizer 길이, input/output embedding row 수, 실제 batch에 등장한 최대 ID를 한 줄에 놓는 것이다. `max_id < num_embeddings`가 거짓이면 더 아래 kernel을 볼 필요가 없다. tied output head가 있다면 input embedding만이 아니라 output projection row도 확인한다. quantized model은 resize가 지원되지 않거나 별도 재양자화가 필요할 수 있으므로 runtime에서 즉흥적으로 row를 늘리는 것을 배포 전략으로 삼지 않는다.

복구는 새 tokenizer와 resize된 model, generation config와 template를 하나의 versioned artifact로 다시 묶는 것이다. resize만 해서 새 PAD row의 의미가 학습되어야 한다고 착각하지 않는다. PAD는 mask로 계산에서 배제할 수 있지만 새 role token은 model이 학습하지 않았다면 기존 marker와 동등하지 않다. canary에서는 padding이 실제 발생하는 mixed-length batch를 포함하고, tokenizer-only rollback이 model과 version skew를 만들지 않도록 원자적으로 전환한다.

종료 조건은 예외가 사라진 것보다 강해야 한다. 모든 special ID가 vocabulary 범위 안에 있고, tokenizer role mapping과 generation config가 같은 ID를 가리키며, saved/reloaded artifact에서도 동일하고, PAD가 포함된 batch와 포함되지 않은 단독 요청의 semantic token logits가 일치해야 한다. 이 네 조건이 닫혀야 재발 방지가 된다.

### 사건 2: 같은 질문이 혼자일 때와 batch일 때 다른 답을 낸다

먼저 deterministic 조건을 만든다. sampling을 끄거나 동일 seed와 확정적 설정을 사용하고, adapter와 prefix-cache 상태를 고정한다. prompt A를 단독으로 실행한 뒤 길이가 긴 B, 짧은 C와 함께 실행한다. visible text만 비교하지 말고 A의 final IDs, attention mask의 1 영역, position IDs, 첫 decode step의 선택 대상 position, 첫-step logits checksum을 비교한다.

Transformers dense batch에서 A가 오른쪽 padding되고 generation loop가 마지막 tensor 열의 logits를 선택한다면, A의 semantic last token과 선택 row가 다르다는 증거가 나온다. 이 경우 left padding으로 바꾸되 position IDs가 mask 누적에 맞게 생성되는지 확인한다. 단순히 PAD를 왼쪽으로 옮기고 absolute position을 전체 tensor index로 두면 다른 종류의 drift를 만들 수 있다.

vLLM이나 SGLang에서 dense PAD tensor가 관측되지 않으면 같은 처방을 적용하지 않는다. A가 단독과 batch에서 같은 prompt IDs를 가지는지, prefix-cache match가 달라졌는지, scheduler가 prefill/decode chunk를 나누면서 computed-token state가 어긋났는지 본다. batch 구성에 따라 sampling RNG consumption이 달라지는지도 분리한다. “batch가 원인”은 층을 가리키지 않는 현상명일 뿐이다.

반증 행렬은 길이와 위치를 독립적으로 바꾼다. B의 내용은 유지한 채 PAD 길이만 늘릴 수 있는 synthetic fixture, A 앞뒤의 scheduling 순서만 바꾸는 fixture, cache를 끈 fixture를 만든다. A의 결과가 PAD 길이에만 따라 바뀌면 tensor 좌표 가설이 강해지고, cache on/off에 따라 바뀌면 prefix identity 가설이 강해진다. 이런 식으로 한 번에 한 계약만 흔든다.

복구 후에는 single-vs-batch equivalence를 regression test로 남긴다. 모든 floating-point logits가 bitwise 같아야 한다고 무리하게 요구하기보다, 동일 실행 경로에서 first-token distribution의 허용 오차와 greedy selected ID, finish condition이 안정적인지 정의한다. 병렬 kernel의 수치 순서 차이는 17장에서 별도로 다루되, 이 장에서는 ID와 좌표 계약 차이를 먼저 제거한다.

### 사건 3: 긴 tool 대화가 성공 응답을 내지만 protocol을 어긴다

관측된 prompt가 context limit을 넘었고 auto truncation이 켜져 있다. 서버는 200을 반환했지만 model은 존재하지 않는 tool name을 호출하거나 JSON 중간에서 일반 문장을 생성한다. 단순 token tail을 보면 출력 문제처럼 보일 수 있다. compiler trace를 복원하면 tool schema의 여는 부분이나 이전 tool result의 role marker가 잘린 채 closing marker만 남아 있는 경우가 있다.

재현은 limit보다 2~3 token 작은 fixture에서 시작해 한 token씩 늘린다. 각 단계마다 message와 template segment가 차지한 token span을 표시하고, 잘린 뒤 남은 atom의 완전성을 검사한다. system block, tool declaration, assistant tool-call, tool result, generation suffix는 내부 일부만 남길 수 없는 원자 단위로 취급한다. multimodal이면 placeholder count와 image/video feature count도 함께 assert한다.

숫자 budget은 다음처럼 역산한다. model context를 `C`, 예약할 generation을 `G`, template와 mandatory system/tool overhead를 `H`, safety margin을 `M`이라 하면 보존 가능한 대화 budget은 `B = C - G - H - M`이다. `H`를 평균값으로 두면 tool schema가 커질 때 다시 넘친다. 실제 요청을 render/tokenize한 값으로 계산하고, 제거 후 다시 render하여 길이를 검증한다. 문자열 글자 수의 비례 추정은 multibyte와 tokenizer merge 때문에 경계에서 실패한다.

복구 policy는 product 의미에 따라 선택한다. 오래된 conversational turn을 완전한 묶음으로 제거하거나, 별도 모델/규칙으로 summary를 만들고 그 provenance를 표시하거나, tool schema를 필요한 subset으로 줄일 수 있다. mandatory system instruction과 현재 user turn, generation suffix는 보존한다. 아무 policy도 안전하지 않으면 명시적인 context-too-long 오류가 조용한 절단보다 낫다.

검증에서는 HTTP 성공률만 보지 않는다. truncation 발생률, 제거된 role별 token 수, protocol-atom rejection 수, tool parse failure, placeholder mismatch, cache hit 변화, output finish reason을 함께 본다. 절단을 켠 뒤 latency가 좋아졌어도 tool failure가 늘었다면 serving 최적화가 아니라 의미 손실을 비용으로 지불한 것이다.

### 사건 4: 화면에는 빈 답인데 token 과금과 종료 기록은 남는다

response text가 빈 문자열이고 usage에는 output token 1이 기록되었다고 하자. 첫 질문은 “왜 model이 아무것도 생성하지 않았나”가 아니다. raw output ID가 무엇이었고 detokenizer가 어떤 option으로 이를 표시했는가다. ID가 EOS나 special role marker이고 `skip_special_tokens=True`였다면 생성과 종료는 정상인데 표현 단계가 숨긴 것이다.

같은 IDs를 special 표시 on/off로 decode하고, byte pieces도 기록한다. EOS라면 effective EOS set과 finish reason이 일치하는지 확인한다. 일반 byte token인데 text가 없다면 incomplete UTF-8 보류, streaming offset, cleanup을 조사한다. IDs 자체가 비어 있다면 prefill-time stop이나 request cancellation처럼 더 앞 층으로 이동한다.

이 사건은 모니터링 설계를 바꾼다. 사용자 응답 text를 그대로 로그에 저장하는 것은 개인정보 문제도 있고 원인 규명에도 불충분하다. 대신 output token count, 마지막 몇 ID의 보호된/샘플링된 trace, finish reason, skipped-special count, pending-byte length를 metric 또는 secure trace로 남긴다. text가 비었지만 EOS 1개가 생성된 요청과 engine이 token을 하나도 commit하지 못한 요청을 구별할 수 있어야 한다.

speculative decoding 중 빈 답이 늘었다면 target과 draft의 special ID 계약도 확인한다. BOS/EOS ID가 다르거나 한쪽 vocabulary가 같은 ID를 다른 piece로 해석하면 draft acceptance 이전부터 경계가 깨진다. tokenizer fingerprint, special-token map, vocabulary piece hash를 target/draft pair의 startup validation에 넣는다. 성능을 위해 붙인 draft model이 의미 identity를 바꾸지 않는다는 것을 먼저 증명해야 한다.

### 운영 dashboard는 원인 층을 보존해야 한다

이 장의 metric은 높은 수준의 request latency만으로 충분하지 않다. 입력 shaping을 관측하기 위해 prompt token histogram, truncation 요청/실행/거부 수, 잘린 token 수, special-token 추가 여부, PAD 비율 또는 engine의 실제 scheduled-token 비율을 본다. 출력에서는 generated token 수, skipped-special 수, EOS/length/stop/cancel finish reason, detokenizer pending-byte와 decode error를 본다.

label cardinality를 통제해야 한다. raw token ID나 model-generated tool name을 Prometheus label로 넣으면 시계열이 폭발한다. model revision, tokenizer revision, template revision, engine version, padding/truncation policy처럼 bounded dimension을 label로 두고, 개별 ID sequence는 sampling된 secure trace에 둔다. dashboard에서 이상 cohort를 찾은 뒤 trace ID로 재현 bundle을 찾아가는 구조가 좋다.

경보도 원인과 가까워야 한다. truncation rate만 높다고 경보하면 정상적인 긴 문서 workload에도 울린다. truncation 이후 tool parse failure 상승, placeholder mismatch, empty-text-with-output-tokens 비율, single/batch canary divergence처럼 의미 손실과 결합한 신호가 더 강하다. 배포 전후 tokenizer/model fingerprint mismatch는 요청을 받기 전에 startup hard failure로 막는 편이 낫다.

### 지금까지의 계약을 재현 질문으로 회수한다

reader가 실제 시스템을 열었을 때 첫 번째로 답해야 할 질문은 “special token 목록이 무엇인가”가 아니다. 최종 prompt ID 배열을 누가 만들었고, template가 출력한 marker와 tokenizer가 자동 추가한 경계가 중복되지 않는가다. 이어서 ID가 embedding과 output head 범위 안에 있는지, PAD와 EOS를 같은 ID로 썼다면 explicit mask와 종료 경계가 모호하지 않은지 확인한다.

두 번째는 sequence shaping이다. dense tensor라면 PAD 방향, mask, position, selected-logits row를 함께 본다. request-aware scheduler라면 logical sequence, computed/cache prefix, chunked-prefill state를 본다. truncation은 단순 side가 아니라 어떤 protocol atom을 어떤 우선순위로 보존했는지 묻는다. model context 안에 들어갔다는 사실은 의미가 보존됐다는 증명이 아니다.

세 번째는 출력 identity다. raw selected IDs, EOS/stop 판정, detokenizer option, streaming commit text를 분리한다. `skip_special_tokens`로 보이지 않는다고 생성되지 않은 것이 아니며, stop string으로 잘렸다고 EOS가 선택된 것도 아니다. usage count는 text 길이가 아니라 token commit 규칙과 연결한다.

마지막으로 artifact와 trace를 재현 가능하게 묶는다. tokenizer, model, generation config, chat template, adapter, quantization/format 변환물이 하나의 compatibility manifest를 가져야 한다. request JSON만 저장한 재현은 effective config와 scheduler 변형을 놓친다. final IDs와 정책 결정을 남긴 trace가 있어야 “왜”를 코드 지점까지 되돌릴 수 있다.

### 재현 번들을 읽는 순서가 조사 비용을 결정한다

재현 자료가 많아도 읽는 순서가 나쁘면 kernel trace부터 파다가 하루를 잃는다. 가장 먼저 artifact identity를 비교한다. model weight revision, tokenizer files의 content hash, special-token map, tokenizer config, generation config, chat template, adapter와 quantization metadata가 incident와 재현 환경에서 같은지 본다. 여기서 다르면 동일 요청을 실행했다는 전제가 아직 성립하지 않는다.

다음은 문자열 이전의 구조다. 원래 message role 순서, content part 유형, tool definition과 selected tool policy, multimodal asset count를 확인한다. 이 자료에서 rendered prompt로 내려가는 mapping을 본다. system delimiter가 어디에서 열리고 닫히는지, generation prompt가 붙었는지, user text 속 marker-like 문자열이 escape되었는지를 segment 단위로 표시한다. 이 단계가 7장의 소유 범위이며, 여기서 발견된 오류를 padding 문제로 오인하지 않는다.

세 번째로 tokenizer 출력의 identity를 본다. 각 segment의 token span, 자동 BOS/EOS 추가 전후 ID, added/special token mask, UNK 발생 위치를 비교한다. 같은 rendered bytes가 다른 IDs가 되었다면 normalization, tokenizer revision, `add_special_tokens`, `parse_special` 같은 입력 flag가 원인 후보가 된다. 반대로 IDs가 같으면 tokenizer 위쪽은 잠시 닫고 sequence shaping으로 내려간다.

네 번째는 truncation 전후다. original IDs와 final IDs의 longest common prefix/suffix만 출력하면 방향은 알 수 있지만 protocol atom이 완전한지는 알 수 없다. 제거된 ID 범위를 source segment에 역매핑하여 어떤 role과 delimiter가 손실됐는지 표시한다. context budget 계산에 사용한 최대 길이, generation reserve, multimodal expansion, cache prefix 길이도 함께 적는다. “128 token 잘림”보다 “tool schema opening과 함수 두 개가 제거됨”이 훨씬 행동 가능한 진단이다.

다섯 번째는 실행 좌표다. dense batch이면 row별 semantic length, PAD side와 count, attention-mask 합, position-ID 시작/끝, logits selection index를 기록한다. paged/request-aware engine이면 request의 logical length, cached/computed count, 새로 schedule된 token 수, block/table identity의 요약을 기록한다. 둘을 같은 `padding_length` 필드에 억지로 넣으면 구현 차이를 잃는다.

여섯 번째부터 model 이후를 본다. first-step selected ID와 top candidates, effective EOS/stop ID sets, finish reason, raw output IDs, detokenizer offsets와 visible text를 시간 순서로 맞춘다. 이 순서에서 first-step logits가 이미 다르면 output formatting을 볼 이유가 없다. IDs는 같은데 text만 다르면 GPU attention kernel을 볼 이유가 없다. 증거가 갈라지는 최초 지점이 조사 시작점이다.

이 판독법은 “아래부터 위까지 전부 로그로 남기라”는 뜻이 아니다. 평소에는 bounded metric과 fingerprint를 저장하고, canary나 sampling trace에서만 더 상세한 ID-level 자료를 남긴다. incident가 발생하면 같은 artifact로 synthetic fixture를 재생해 상세 trace를 얻는다. 개인정보가 포함된 production text를 무기한 저장하지 않고도 계약의 갈라짐을 찾을 수 있다.

## 8.9 경계값·배포 manifest·9장 handoff로 계약을 승인한다

이 절은 tokenizer 분할 규칙과 chat template 문법을 다시 정의하지 않는다. 문자열→ID와 template compiler의 canonical 설명은 6·7장을 참조하고, 여기서는 그 결과 ID가 padding·truncation·mask·position 계약을 통과했는지만 승인한다. 승인된 `input_ids`, `attention_mask`, `position_ids`가 embedding row와 첫 residual로 바뀌는 순간부터는 9장의 소유다.

special-token과 truncation 버그는 평균 길이 테스트에서 잘 숨는다. model context limit을 `C`라고 할 때 `C-1`, `C`, `C+1` 길이만 보는 것도 부족하다. template가 BOS나 generation suffix를 자동으로 붙인다면 user payload 길이와 final sequence 길이가 다르기 때문이다. fixture는 최종 render/tokenize 결과를 기준으로 목표 길이를 맞춘다.

첫 fixture family는 빈 입력과 최소 입력이다. 빈 system, 빈 user, whitespace-only user, special marker 문자열만 포함한 user를 각각 넣는다. BOS만 남는지, BOS와 EOS가 중복되는지, generation suffix 뒤의 selected position이 유효한지 본다. 빈 답을 허용하지 않는 application validation과 model tokenizer의 빈 sequence 처리를 구분한다.

두 번째 family는 길이가 한 token씩 다른 pair다. `[A]`, `[A,B]`, `[A,B,C]`처럼 의미 없는 통제 token을 사용해 padding side와 position을 확인한다. 같은 A를 긴 sequence와 묶을 때 A의 semantic positions가 어떻게 변하는지 계산하고, mask된 PAD에 대한 logits를 우연히 읽지 않는지 확인한다. 이 fixture는 자연어 품질 평가가 아니라 좌표 불변성 검사다.

세 번째 family는 marker collision이다. user가 literal `<|assistant|>`, model-specific BOS 문자열, JSON 속 EOS-like text를 보낸다. trusted template 영역과 untrusted content 영역이 같은 parse-special policy를 공유하는지 확인한다. ordinary pieces로 남아야 하는 문자열이 단일 privileged special ID로 변하면 injection boundary가 달라진다. 반대로 template marker가 ordinary pieces로 쪼개지면 학습 prompt와 달라진다.

네 번째 family는 truncation atom이다. system block, tool schema, assistant tool call, tool result, current user turn 각각의 바로 앞과 내부와 바로 뒤가 limit에 걸리도록 만든다. policy가 atom 내부 절단을 거부하거나 전체 제거하는지, 제거 후 delimiter balance가 맞는지 본다. 문자열 괄호 balance만으로는 tokenizer protocol을 증명할 수 없으므로 role/segment graph에서 완전성을 검증한다.

다섯 번째 family는 special ID alias다. PAD와 EOS가 다른 artifact, 같은 artifact, PAD가 없는 artifact를 준비한다. explicit attention mask의 유무를 교차시켜 결과를 본다. alias가 같은 것 자체가 실패 조건이 아니라 mask 추론과 종료 판정의 모호성이 실패 조건임을 확인할 수 있다. classification이나 training collator를 공유한다면 label ignore policy까지 별도 fixture로 둔다.

여섯 번째 family는 round-trip과 streaming이다. normalization이 있는 text, combining character, emoji, byte fallback이 필요한 입력, special token 주변 공백을 encode/decode한다. 전체 decode와 한 token씩 incremental decode의 최종 byte sequence가 같은지, 중간 chunk가 유효한 UTF-8인지, cleanup이 공백을 언제 바꾸는지 본다. `decode(encode(x))` equality가 제품 요구라면 tokenizer가 이를 보장하는 범위를 명시해야 한다.

### 8.9.1 배포 전에는 호환성 manifest를 실행 가능한 주장으로 만든다

manifest에 version 문자열만 적어서는 부족하다. tokenizer vocabulary size와 hash, model input embedding row 수, output head row 수, tied-weight 여부, BOS/EOS/PAD/UNK와 additional special ID set, 자동 add flags, padding/truncation side, chat template hash를 machine-readable field로 둔다. quantized artifact와 draft model이 있다면 각각 같은 필드를 가진다.

startup validator는 먼저 범위를 검사한다. 모든 emitted special ID가 `[0, num_embeddings)`에 있고, sampler가 고려하는 vocabulary와 output head가 호환되는지 본다. tokenizer가 추가한 ID가 model row보다 크면 요청을 받기 전에 실패한다. “그 token을 쓰는 요청만 실패할 것”이라며 부분 가동하면 batch와 template 조건에 따라 잠복 장애가 된다.

그다음 alias와 role을 검사한다. PAD=EOS 같은 의도적 alias는 허용 목록과 근거를 요구하고, mask가 언제 명시되는지 manifest에 적는다. BOS와 EOS가 같은 family도 있으므로 무조건 uniqueness를 요구하지 않는다. 핵심은 각 equality가 artifact의 정상 설계인지 잘못된 config merge인지 판별 가능한 것이다.

template/tokenizer smoke compile은 실행 서버 없이도 할 수 있다. 대표적인 system-user, multi-turn, tool-call, multimodal message를 render하고 tokenize하여 expected boundary IDs와 segment spans를 확인한다. 이 책의 검토 과정처럼 model forward를 실행하지 않아도 상당수 계약 오류를 잡을 수 있다. 다만 numerical generation equivalence까지 증명했다고 과장하지 않는다.

serving startup에서는 engine이 merge한 effective config를 다시 읽는다. command-line option, model generation config, request default가 합쳐진 EOS, max length, truncation, skip-special 정책을 canonical form으로 출력한다. 선언 파일의 값과 effective state가 다르면 우선순위도 함께 기록한다. 사용자는 option을 설정했다고 믿지만 낮은 우선순위라 적용되지 않는 사고를 막는다.

canary는 최소 한 개의 mixed-length batch와 context-boundary request를 포함한다. 단일 짧은 prompt만으로 health check를 구성하면 PAD와 truncation path를 전혀 통과하지 않는다. raw output ID와 expected finish reason을 assert하고 visible text만 비교하지 않는다. model update로 표현이 조금 달라져도 계약 경계가 깨졌는지를 더 정확히 잡을 수 있다.

rolling update에서는 old tokenizer/new model과 new tokenizer/old model이 잠깐이라도 조합될 수 있는지 본다. frontend가 tokenize하고 backend가 IDs를 받는 분리 구조라면 revision handshake가 필수다. backend가 기대한 tokenizer fingerprint와 request의 fingerprint가 다르면 재tokenize 가능한 text 경로로 돌리거나 명시적으로 거부한다. 정수 ID는 tokenizer revision을 잃는 순간 self-describing data가 아니다.

cache도 revision 경계를 넘어 재사용하지 않는다. prompt IDs가 우연히 같더라도 embedding weights나 rope/config가 달라지면 KV의 의미가 다르다. 이 장에서는 tokenizer/template identity를 강조하지만 cache namespace에는 model weight, adapter, quantization과 실행 의미를 바꾸는 config도 포함되어야 한다. 14장에서 prefix/KV cache key를 더 깊게 다룬다.

rollback 계획은 artifact 묶음을 단위로 한다. model만 되돌리고 tokenizer는 새 버전에 남기거나, template만 CDN cache에 남기는 rollback은 새 장애를 만든다. manifest digest로 배포와 cache namespace, trace를 연결하면 한 digest를 원자적으로 이전 digest로 돌릴 수 있다. 장애 후 “어느 조합이 실제로 서비스됐는가”도 digest 하나에서 복원된다.

### 8.9.2 성능 최적화와 의미 보존을 같은 실험에서 측정한다

padding을 줄이고 truncation을 늘리면 throughput 숫자는 쉽게 좋아진다. 그러나 이 장의 최적화 목표는 처리 token을 줄이는 것 자체가 아니다. 사용자 의도를 보존한 채 불필요한 계산을 줄이는 것이다. 따라서 벤치마크는 tokens/s와 time-to-first-token 옆에 protocol validity, answer consistency, tool success, truncation-induced rejection을 놓아야 한다.

dense batch의 padding waste는 `1 - 실제 semantic token 합 / (batch_size × padded_length)`로 근사할 수 있다. 이 값이 크면 length bucketing이나 dynamic batching의 이득이 보인다. 반면 paged continuous batching에서는 같은 공식을 그대로 적용하면 실제 kernel work를 잘못 설명한다. scheduled token, active sequence, KV block utilization, chunked prefill 크기 같은 engine-native 지표를 쓴다.

left padding으로 correctness를 고친 뒤 성능이 변했다면 두 효과를 분리한다. 동일 semantic IDs에 대해 kernel이 처리한 shape 차이, position construction 비용, batch composition을 측정한다. 자연어 output만 비교하면 선택 token이 달라 workload 자체도 달라진다. 먼저 고정 길이 generation이나 teacher-forced logits fixture로 shape 비용을 보고, 다음에 end-to-end 품질을 본다.

truncation은 input token 수와 prefill latency를 직접 줄이지만 cache hit를 낮출 수도 높일 수도 있다. 공통 system prefix를 보존하면 prefix cache에 유리하고, blind left truncation이 prefix를 제거하면 request마다 다른 시작점이 되어 재사용을 잃는다. tool schema를 request별로 다르게 축약해도 cache identity가 파편화된다. semantic policy와 cache policy를 함께 설계해야 하는 이유다.

special-token skipping은 전송 text를 줄일 뿐 model compute를 되돌리지 않는다. EOS가 한 token 빨리 선택되도록 학습·prompt·stop을 개선하는 것과, 이미 생성된 marker를 detokenizer에서 숨기는 것은 비용 효과가 다르다. metric에서 generated IDs와 visible characters를 분리하지 않으면 decode 최적화를 generation 최적화로 잘못 보고한다.

최종 승인에서는 성능 회귀와 의미 회귀를 별도 문턱으로 둔다. throughput이 목표를 넘더라도 batch equivalence나 tool grammar fixture가 실패하면 승인하지 않는다. 반대로 correctness가 맞아도 PAD waste와 context rejection이 운영 목표를 넘으면 scheduling이나 prompt policy를 다시 설계한다. “왜 빠른가”와 “왜 같은 의미인가”를 모두 설명할 수 있어야 serving 최적화가 완성된다.

### 8.9.3 독자가 직접 구현을 파고들 수 있으면 이 장은 끝난다

완료 기준은 option 이름을 외운 것이 아니다. 낯선 runtime을 받아도 tokenizer entrypoint에서 final IDs까지 call path를 찾고, special role mapping의 저장 위치와 model embedding row의 관계를 확인할 수 있어야 한다. 그다음 batch builder나 scheduler에서 sequence가 어떤 단위로 묶이고, truncation과 cache가 어느 상태를 바꾸는지 식별한다. 마지막으로 selected ID가 종료 판정과 detokenizer를 지나 text가 되는 경로를 연결한다.

코드를 읽을 때 변수 이름만 추측하지 않는다. `pad`, `eos`, `length`, `offset` 같은 이름은 층마다 다른 단위를 가진다. 선언부에서 type과 초기값을 확인하고, 모든 mutation site를 검색하며, consumer가 무엇과 비교하는지 본다. request field가 내부 state로 복사되는 지점, default나 model config와 merge되는 지점, kernel metadata로 변환되는 지점을 한 장의 trace에 놓는다. option의 “효과”는 이 mutation chain으로 설명해야 한다.

의도가 불명확하면 blame과 commit history, 관련 issue와 test를 보조 증거로 사용하되 현재 코드의 행동과 섞지 않는다. comment가 오래되었을 수 있고 test가 모든 backend를 덮지 않을 수 있다. 먼저 pinned revision의 실행 가능한 branch와 assertion으로 행동을 고정하고, 변경 이력은 왜 그 guard가 추가되었는지 설명하는 데 쓴다. 추론과 확인된 사실을 문장에서 구별한다.

새 backend를 검토할 때도 이 장의 지도는 유지된다. artifact metadata, prompt compiler, tokenizer, sequence shaping, scheduler, model input, sampler, stop, detokenizer라는 경계를 찾는다. 구현이 이들을 한 함수에 섞었더라도 책임 자체가 사라지는 것은 아니다. 각 경계에서 들어온 ID와 나간 ID, 좌표, policy, provenance를 적으면 비교 가능한 설명이 된다.

독자는 incident 뒤에 “padding 문제 같음”이라고 끝내지 않고 최초로 달라진 상태를 제시해야 한다. 예컨대 “right-padded row에서 mask는 정확했지만 generation이 tensor 마지막 열의 logits를 선택했다”, “auto truncation이 tool declaration의 앞부분을 제거했다”, “raw EOS ID는 생성됐지만 skip-special decode가 빈 text로 표현했다”처럼 반증 가능한 문장이어야 한다. 그 수준에 도달하면 다음 장의 embedding과 attention 계산을 잘못된 입력 위에서 분석하는 일을 피할 수 있다.

이 장은 6장의 tokenizer가 만든 ID를 받아 7장의 prompt compiler와 결합했고, padding/truncation이 그 sequence의 좌표와 의미를 어떻게 바꾸는지 추적했다. 다음 9장은 이 final IDs가 embedding row와 position representation을 만나 hidden state가 되는 순간을 다룬다. 10장은 마지막 hidden state가 vocabulary logits로 투영되는 계산을 잇는다. 그러므로 이 장의 종료 산출물은 깨끗한 문자열이 아니라 provenance가 있는 final token sequence다.

## 8.10 padding side가 position·mask·logit row를 바꾼 회귀와 truncation control-loss 사건

Padding incident P8은 단순했다. 같은 두 prompt를 각각 실행하면 같은 답을 냈지만 batch로 묶으면 짧은 요청의 첫 token이 달라졌다. Token IDs를 비교하니 실제 문장 부분은 같고 PAD가 붙은 쪽만 달랐다. 운영자는 right padding을 left padding으로 바꾸어 증상을 없앴다. 그러나 이 결과만으로 tokenizer 설정이 원인이라고 할 수 없다. Padding side는 ID 배열뿐 아니라 attention mask, position IDs, cache position, last-logit row 선택과 graph shape를 동시에 바꾸기 때문이다.

두 prompt A=[BOS,11,12,13], B=[BOS,21]를 길이 4로 맞춘다고 하자. Right padding B는 [BOS,21,PAD,PAD], mask는 [1,1,0,0]이다. Left padding은 [PAD,PAD,BOS,21], mask [0,0,1,1]이다. 두 배열은 유효 token 순서가 같지만 physical row가 다르다. Causal model이 “마지막 배열 row의 logits”를 무조건 고르면 right-padded B는 PAD row를 읽고, left-padded B는 실제 마지막 token row를 읽는다. Correct owner는 attention mask 또는 sequence length에서 last valid row를 계산해야 한다.

Position IDs는 모델 계약에 따라 달라진다. Mask cumulative sum을 사용해 pad를 고정 값으로 만드는 helper라면 right B의 유효 positions는 [0,1], left B도 [0,1]로 복구할 수 있다. 단순 `arange(4)`를 쓰면 left B의 유효 token은 positions 2,3이 되어 single-run positions 0,1과 다르다. RoPE가 position에 의존하므로 attention 값이 달라지고, pad를 attention에서 mask했더라도 유효 token hidden이 달라질 수 있다. “PAD는 mask되니 안전하다”는 가설이 여기서 깨진다.

Fixture는 `input_ids`, `attention_mask`, `position_ids/cache_position`, `logit_row_index` 네 tensor를 batch 직전과 model forward 직전에 비교한다. Token IDs만 같다는 parity는 첫 tensor만 닫는다. Position까지 같고 logit row만 다르면 output gather owner를 본다. Position이 다르면 model helper 또는 caller가 mask를 position으로 변환하는 경계를 본다. Mask부터 다르면 tokenizer/batch collation 또는 PAD ID equality 가정이 먼저다.

PAD ID와 mask를 분리해야 하는 이유도 수치로 드러난다. PAD가 EOS와 같은 ID를 공유하는 모델이 있을 수 있다. Prompt 내 실제 EOS ID까지 `input_ids != pad_id`로 mask하면 의미 있는 token을 지운다. 반대로 PAD가 새 ID인데 embedding row를 resize하지 않았다면 gather OOR가 난다. Mask는 padding operation이 만든 structural metadata여야 하며 ID equality는 최적화 힌트일 뿐 진실의 원천이 아닐 수 있다.

P8의 2×2 matrix는 padding side와 output gather를 교차한다. Left/right batch에 mask-derived last-valid gather와 physical-last-row gather를 각각 적용한다고 가정한다. Physical gather에서만 right padding이 실패하고 mask-derived gather는 둘 다 맞으면 tokenizer side 자체보다 logit row consumer가 root edge다. 두 gather 모두 right에서 실패하면 position/mask 또는 backend padding support를 더 본다. Single request 정상은 batch shape와 gather가 달라지는 예측에 부합하지만 원인을 혼자 증명하지 않는다.

Backend도 같은 logical tensor를 다르게 pack할 수 있다. Flash/paged attention path가 unpadded query rows와 cumulative sequence lengths를 소비하면 PAD row가 kernel에 들어가지 않을 수 있다. Dense fallback은 padded tensor와 mask를 소비할 수 있다. 따라서 left/right 변경이 effective backend를 바꾸면 결과 개선을 padding semantics만으로 설명하지 않는다. Requested/selected backend, packed row map과 output scatter/gather를 fixture에 넣는다.

Graph replay는 또 다른 generation을 추가한다. Capture bucket이 batch 8×length 2048인데 현재 live rows가 짧다면 static position, mask와 last-index buffer를 replay 전에 갱신해야 한다. IDs buffer만 current generation이고 last-index가 이전 batch라면 valid address에서 다른 row logits를 읽어 silent wrong answer가 난다. Padding 회귀 matrix에 eager/graph와 buffer content generation을 넣는 이유다.

두 번째 사건 T8은 truncation이 성공 응답을 만들었지만 system control을 잃은 경우다. Context budget 128에서 system control 30 tokens, tool schema 50, 대화 history 60, 새 user 20, generation prompt 3이면 합 163이다. 단순 left truncation으로 앞 35 tokens를 자르면 system control 전부와 tool schema 앞부분이 사라질 수 있다. Server는 128-token valid tensor를 만들고 model은 정상 답을 생성하지만 protocol correctness는 실패한다.

Truncation의 경쟁 가설은 capacity, policy와 bookkeeping으로 나눈다. Capacity 가설은 model/backend hard limit 때문에 163을 실행할 수 없다는 사실이다. Policy 가설은 어떤 semantic atom을 제거할지 결정한다. Bookkeeping 가설은 실제로 버린 span과 usage/observability가 일치하는지다. Capacity가 참이어도 oldest raw tokens를 자르는 policy가 옳다는 결론은 나오지 않는다. 세 질문을 한 `max_length` 숫자로 접지 않는다.

Atomic unit을 먼저 정의한다. System instruction, tool definition 한 개, role delimiter pair, multimodal placeholder와 feature bundle, assistant generation marker는 중간에서 자르면 안 되는 protocol atom이다. History message는 whole-turn 단위로 버릴 수 있고 긴 user document는 명시적 chunk/summarize policy를 가질 수 있다. Token budget allocator는 각 atom의 priority, minimum complete size와 dependency를 읽어 inclusion plan을 만든다.

위 163-token 예에서 반드시 남길 system 30, 새 user 20, generation marker 3은 53이다. Tool 호출이 허용돼야 한다면 schema 50도 남아 103, history에는 25만 남는다. History turn 하나가 30이라면 반쪽을 넣지 않고 전체를 버려 103으로 실행할 수 있다. Tool 사용이 선택 사항이고 deadline이 더 중요하면 tool atom 전체를 빼고 mode를 `tools_disabled_by_budget`으로 명시한다. Schema 절반 25를 남기는 것과는 의미가 다르다.

Stop/control loss는 출력만 보고 찾기 어렵다. Model이 우연히 안전한 답을 할 수 있기 때문이다. Compile manifest는 각 required control atom의 final token range와 digest를 보존하고 truncation 뒤 present/removed/replaced를 기록한다. Control atom이 제거됐는데 success response를 냈다면 correctness pass가 아니라 policy hard fail이다. Output safety evaluation은 추가 terminal이며 input contract 보존을 대신하지 않는다.

Truncation side A/B도 공정하게 설계한다. Left와 right는 서로 다른 prompt semantics를 만들므로 latency 비교만으로 winner를 고르지 않는다. 동일 semantic atom policy를 구현한 message-aware control과 raw-token left/right를 비교한다. Primary는 required atom preservation과 final IDs, guardrail은 TTFT·ITL·usable history다. 더 빠르지만 control을 잃은 lane은 performance 후보에서 제외한다.

Stop condition도 함께 본다. Prompt 안 EOS, generated EOS, stop string과 max tokens는 서로 다른 terminal이다. Truncation이 assistant prefix 끝 delimiter를 잘라 generation prompt를 새로 붙이면 prompt EOS 위치와 generation boundary가 달라질 수 있다. Decoder가 `skip_special_tokens`로 화면에서 이를 숨겨도 model input과 stop reason에는 남는다. Final response text가 같다는 사실은 stop/control trace parity가 아니다.

Pinned source walk에서는 tokenizer padding/truncation configuration에서 batch encoding output까지, generation input preparation에서 position/cache position과 logits row selection까지 caller-consumer를 잇는다. Warning 문자열은 source predicate를 찾는 단서일 뿐 원인 증거가 아니다. vLLM/SGLang은 request별 unpadded token count가 scheduler/runner metadata로 변하는 지점, llama.cpp는 batch sequence IDs와 output index를 만드는 지점을 확인한다. 네 stack에 같은 `padding_side` field가 있다고 가정하지 않는다.

관측은 bounded하게 한다. Fleet metric에는 padding side, length bucket, effective backend, gather mode와 result reason처럼 작은 enum을 둔다. Raw IDs, full prompt와 system text는 sampled secure trace에 둔다. `last_valid_index`, valid/padded rows, position min/max와 control-atom preservation은 digest와 수치로 기록할 수 있다. Wrong-answer incident에서는 failing request와 가까운 passing neighbor의 네 tensor를 보존한다.

복구 순서는 의미를 먼저 지킨다. P8은 unsupported padding/backend 조합을 safe gather path로 fence하고 stale graph metadata를 quarantine한다. T8은 required control atom을 보존할 수 없는 request를 자동 성공시키지 않고 명시적 reject 또는 safe mode로 보낸다. Cache는 old padding/truncation policy generation과 namespace를 공유하지 않는다. Position/KV가 다른 prompt를 같은 prefix hit로 세면 fix 뒤에도 wrong state가 남는다.

Regression matrix는 empty/one-token, left/right, mixed lengths, PAD=EOS, added PAD with resized/unresized embedding, eager/graph, dense/packed backend를 포함한다. Truncation은 limit-1/limit/limit+1, system/tool/media atom boundary, multi-turn whole removal과 generation prompt를 포함한다. 각 cell은 final IDs뿐 아니라 mask, positions, logit row, preserved atoms, stop reason과 first output을 판정한다.

최종 terminal은 두 문장이다. Padding terminal은 동일 logical sequence가 supported batch layout에서 같은 valid positions와 last-logit semantics를 가져 single/batch reference를 만족한다는 뜻이다. Truncation terminal은 context 상한을 지키면서 required control/protocol atoms를 보존하거나 불가능할 때 명시적 policy outcome을 내고, 버린 내용을 usage와 trace가 정확히 설명한다는 뜻이다. 둘을 “더 이상 warning이 없다”로 합치지 않는다.

이 두 사건을 통과하면 9장으로 넘길 state가 정확해진다. Embedding layer는 PAD가 어느 ID인지보다 어떤 rows가 valid하고 어떤 position을 가지며 어느 token ID가 실제 gather되는지를 받는다. Truncation은 model이 보지 못한 control을 복원하지 않는다. 따라서 embedding OOR나 첫 hidden divergence를 조사할 때 tokenizer/batch contract가 이미 닫혔다는 강한 기준선을 가질 수 있다.

### 8.10.1 네 tensor를 표로 펼쳐 single과 batch를 비교한다

fixture를 더 구체화하자. PAD=0, BOS=1이고 두 요청의 유효 ID가 각각 A=`[1, 41, 42, 43]`, B=`[1, 51]`라고 하자. A와 B를 따로 실행할 때 position은 A=`[0,1,2,3]`, B=`[0,1]`이다. 길이 4 right-padding batch에서 B의 physical IDs는 `[1,51,0,0]`, mask는 `[1,1,0,0]`이다. mask cumulative sum에서 1을 뺀 뒤 pad 위치를 0으로 채우면 position은 `[0,1,0,0]`, last-valid index는 1이다. left-padding이면 IDs=`[0,0,1,51]`, mask=`[0,0,1,1]`, positions=`[0,0,0,1]`, last-valid physical index는 3이다.

이제 output tensor가 `[batch, sequence, vocab]`이라고 가정한다. 단순 `logits[:, -1, :]`는 right B에서 index 3의 PAD query row를 읽고 left B에서는 index 3의 실제 token을 읽는다. 올바른 gather는 각 row의 `last_valid_index`를 사용해야 한다. 다만 일부 구현은 forward에 마지막 유효 token만 넘기거나 packed query를 사용해 output shape 자체가 `[num_query_tokens, vocab]`일 수 있다. 그러므로 코드를 읽을 때 `-1`이라는 표현만 검색해서는 안 된다. scheduler가 선택한 query indices, runner의 flatten/pack, model output scatter, sampler input selection까지 같은 logical row가 어떻게 이동하는지 걷는다.

표의 각 cell에는 값과 provenance를 함께 적는다. `attention_mask`가 tokenizer collator에서 왔는지 serving batch builder에서 왔는지, `position_ids`가 caller에서 명시됐는지 model forward가 생성했는지, `last_valid_index`가 sequence length인지 mask sum인지, graph buffer generation이 무엇인지 기록한다. 값이 맞아도 owner가 다르면 다음 backend에서 계약이 깨질 수 있다.

```
case        ids          mask        valid-pos   physical-last  chosen-row
single-B    [1,51]       [1,1]       [0,1]       1              1
right-B     [1,51,0,0]   [1,1,0,0]   [0,1]       3              1
left-B      [0,0,1,51]   [0,0,1,1]   [0,1]       3              3
```

`chosen-row`가 physical-last와 같아야 한다는 규칙은 right padding에서 틀린다. 규칙은 마지막 유효 query의 logits여야 한다. packed backend에서는 chosen row가 원래 physical index가 아니라 pack map의 index일 수 있으므로 `(sequence_id, logical_position)`으로 판정한다.

### 8.10.2 position과 cache position을 같은 숫자로 오해하지 않는다

position ID는 모델의 위치 표현에 들어가는 논리 좌표이고, cache position은 KV cache의 어느 slot에 쓰거나 읽을지를 가리키는 실행 좌표일 수 있다. 두 값이 흔히 같아서 같은 것으로 보이지만 left padding, prefix cache hit, sliding window, chunked prefill, decode continuation에서 갈릴 수 있다. prefix 100 token이 이미 cache에 있고 새 suffix 20 token만 계산한다면 query tensor의 local row는 0부터 19여도 논리 position은 100부터 119일 수 있다.

padding 회귀에서 position만 고치고 cache position이 stale이면 첫 forward는 맞아 보이다 decode에서 잘못된 KV를 읽을 수 있다. 반대로 cache slot은 맞지만 RoPE position이 pad를 포함한 physical index라면 attention score가 달라진다. fixture에는 logical token ordinal, position ID, cache slot, KV block/offset을 별 열로 둔다. 다음 token decode 때 각각이 어떻게 1 증가하는지도 확인한다.

graph replay에서는 host-side length가 새 값이어도 captured device buffer가 이전 값을 가질 수 있다. 그래서 trace에는 단순 배열 dump 외에 buffer update event와 generation을 남긴다. eager가 통과하고 graph만 실패하면 graph 자체를 원인으로 확정하지 말고 어떤 mutable metadata가 replay 전에 갱신되지 않았는지 찾는다. IDs, mask, positions, slot mapping, selected row 중 최초 stale buffer가 root edge다.

### 8.10.3 PAD=EOS 조합을 의미와 구조로 분리한다

decoder-only 모델에서 별도 PAD가 없어 EOS ID를 padding 값으로 재사용하는 경우가 있다. 이것은 vocabulary row를 공유한다는 뜻이지 모든 EOS occurrence를 padding으로 취급하라는 뜻이 아니다. prompt 내부의 실제 EOS와 batch builder가 추가한 structural pad는 같은 정수라도 mask와 provenance가 다르다.

예를 들어 IDs=`[1,70,2,2]`, pad_id=eos_id=2이고 실제 prompt가 `[1,70,2]`라면 마지막 2 하나만 padding이다. `ids != pad_id`로 mask를 만들면 세 번째 실제 EOS까지 0이 되어 유효 길이를 2로 잘못 계산한다. 올바른 mask는 padding operation이 원래 길이 3을 알고 `[1,1,1,0]`으로 만든다. generation helper가 input IDs만 받아 mask를 추론해야 한다면 이 모호성을 해결할 정보가 없다. caller가 explicit mask를 제공하거나 pad와 eos를 분리하는 artifact 정책이 필요하다.

새 PAD token을 추가하는 해결도 비용이 있다. tokenizer vocabulary 길이는 늘지만 model embedding과 lm-head row가 자동으로 늘어난다는 보장은 없다. resize와 weight initialization, tied-weight 재결합, checkpoint 저장, quantized artifact 지원을 확인해야 한다. 8장은 mismatch를 manifest에서 차단하고 9장은 실제 gather와 projection row를 더 깊게 추적한다.

### 8.10.4 truncation plan을 token slice가 아닌 의존 그래프로 만든다

긴 요청을 자를 때 각 atom은 다른 atom에 의존할 수 있다. tool result message는 앞선 assistant tool call과 tool declaration에 의존하고, image placeholder는 feature bundle에 의존한다. assistant continuation은 이전 assistant prefix와 종료되지 않은 문법 상태에 의존한다. 단순 priority 목록만으로는 의존 대상이 제거된 고아 atom이 남을 수 있다.

plan에는 atom ID, token range, required 여부, dependency, removable group, replacement policy를 둔다. history turn을 제거할 때 그 turn의 tool call/result 쌍을 함께 제거한다. image를 제거하면 placeholder와 feature를 함께 제거하거나 명시적 textual replacement를 넣고 template를 다시 compile한다. 이미 tokenized된 배열에서 placeholder ID만 삭제하면 주변 delimiter와 generation mask가 맞지 않을 수 있다.

128-token 예를 다시 계산하자. system S=30, tool T=50, history H1=30, H2=30, user U=20, generation G=3으로 총 163이다. S,U,G가 필수이고 T가 현재 `tool_choice=required`라면 필수 합은 103이다. 남은 25에는 H1이나 H2 전체가 들어가지 않으므로 둘 다 제거한다. 결과 103은 상한보다 25 작지만 문법적으로 완전하다. 25를 채우려고 H2 일부를 넣는 것은 utilization을 높이는 대신 protocol을 깨뜨린다.

반대로 tool choice가 `none`이면 T와 관련 tool history group을 제거할 수 있다. 그 결정은 모델이 임의로 하지 않는다. API policy가 `tools_disabled_by_budget` 같은 명시적 outcome으로 기록하고 user에게 알려야 한다. required atom 합 자체가 128을 넘으면 silent truncation 대신 reject 또는 더 큰 context lane으로 route한다.

### 8.10.5 stop과 control loss를 output 이전에 검출한다

control token이 잘렸는지 자연어 답변을 읽어 판정하면 비결정성과 안전 위험이 생긴다. compile 단계에서 required atom의 final span을 확인한다. system safety block digest, tool delimiter pair, assistant generation marker, multimodal placeholder count가 expected manifest와 일치하는지 model admission 전에 검사한다.

stop도 네 층으로 나눈다. 입력 prompt 안의 EOS는 context token이다. generated EOS는 sampler가 선택한 output token이다. stop token set은 선택 직후 종료 조건이다. stop string은 detokenized text의 문자 조건이다. `max_new_tokens`는 budget terminal이다. truncation이 prompt 끝을 바꾸면 generated continuation의 문법 상태가 달라질 수 있지만, 이 다섯 terminal을 하나의 `finished` boolean으로 접으면 원인을 찾을 수 없다.

사건에서 assistant header의 종료 delimiter 절반이 잘린 뒤 generation prompt가 중복 삽입됐다고 하자. 첫 generated token이 EOS가 되어 화면에는 빈 답이 보인다. detokenizer의 `skip_special_tokens`는 EOS를 숨긴다. 조사자는 “빈 문자열”에서 멈추지 않고 final prompt suffix IDs, generation boundary, first selected ID, stop reason, visible text를 순서대로 본다. first ID가 EOS라면 sampler 이전 logits와 prompt state를 조사하고, ID는 정상인데 visible text만 비면 decode/filter를 조사한다.

### 8.10.6 pinned source를 producer와 consumer 쌍으로 읽는다

Transformers에서는 tokenizer의 padding/truncation 구현이 `input_ids`와 `attention_mask`를 어떻게 만드는지 본 뒤 generation helper가 mask와 special IDs를 어떻게 준비하는지 잇는다. right-padding warning이 있는 줄은 행동의 일부일 뿐이다. 실제 logits selection과 cache position consumer까지 내려가야 경고 조건이 correctness 문제와 같은지 판정할 수 있다.

vLLM과 SGLang에서는 일반적인 dense-padding mental model을 강요하지 않는다. 요청별 token IDs와 길이가 scheduler metadata, packed batch, query length, slot mapping, sampler row로 변환되는 지점을 찾는다. API의 `truncate_prompt_tokens`나 유사 option은 parse에서 끝내지 않고 실제 list slice 또는 policy consumer, usage accounting, cache key까지 걷는다. llama.cpp에서는 batch token, position, sequence ID, logits request flag가 graph input과 output row를 어떻게 정하는지 본다.

각 source note에는 symbol 하나만 적지 않는다. producer symbol, emitted value와 unit, next consumer, falsifier fixture를 적는다. 예를 들어 “mask producer가 원래 길이로 structural mask를 만든다”는 주장은 PAD=EOS fixture에서 실제 EOS가 valid로 남는지로 반증한다. “sampler가 last-valid row를 고른다”는 주장은 mixed-length right-padding fixture에서 single/batch first logits parity로 반증한다.

### 8.10.7 회귀 artifact와 종료 판정을 보존한다

artifact에는 원문 전체가 필요하지 않은 경우에도 재현 가능한 최소 정보를 남긴다. tokenizer/model/template revision, special ID map, padding/truncation policy, raw/final token length, IDs의 안전한 fixture, mask, positions, cache positions, selected row, preserved atom manifest, first output ID와 stop reason을 포함한다. 실제 사용자 prompt는 보안 저장소에 두고 일반 trace에는 digest와 길이만 남긴다.

회귀 matrix는 축을 너무 많이 한 표에 넣지 않는다. padding 표는 side×batch shape×backend×eager/graph를, truncation 표는 budget boundary×atom policy×tool/media mode를 쓴다. 공통 판정 열은 final ID digest, valid position parity, selected logit row, required atom status, expected error다. 실패 cell에는 최초 다른 tensor와 owner symbol을 연결한다.

수정 뒤에는 passing fixture만 추가하지 않는다. 이전에 실패한 배열을 그대로 보존하고, 잘못된 구현을 되살렸을 때 test가 실제로 실패하는지 mutation 관점으로 확인한다. 경고 문구가 사라졌다는 assertion보다 tensor와 semantic atom assertion이 오래 간다. backend가 packed layout으로 바뀌어 physical 배열이 달라져도 logical `(sequence, position, token)` 계약으로 판정할 수 있다.

운영 종료는 canary에서 single/batch logits parity가 회복되고, control-loss rejection과 truncation outcome이 기대 범위에 있으며, stale graph/cache generation이 더 이상 소비되지 않을 때다. throughput이 회복됐다는 사실만으로 닫지 않는다. correctness 회복과 성능 회복, 관측 회복을 별 terminal로 기록한다.

## 8.11 처음 보는 batch 오류를 30분 안에 분류하는 실전 절차

첫 5분에는 재현 조건을 축소한다. 실패 요청 하나와 가장 가까운 성공 요청 하나를 고른다. text, template, tokenizer, sampling을 고정하고 single 실행과 two-row mixed-length batch만 비교한다. 실행이 금지된 정적 조사라면 test fixture와 call path에서 두 shape가 만들어지는 분기를 찾는다. 이 단계에서 GPU kernel을 먼저 바꾸지 않는다. final IDs부터 다르면 6~8장 경계이고, IDs가 같은데 first logits가 다르면 mask·position·row selection 이후다.

다음 5분에는 네 tensor 장부를 채운다. row별 raw length, physical length, `input_ids`, `attention_mask`, `position_ids` 또는 그 생성 규칙, sampler가 읽은 logical row를 기록한다. 배열 전체가 너무 크면 최초·마지막 16개와 digest, min/max, nonzero count를 남긴다. 단, 실패가 경계에 있으므로 suffix를 생략하지 않는다. PAD=EOS인지, left/right인지, explicit mask인지 inferred mask인지도 적는다.

세 번째 5분에는 2×2 반증을 한다. padding side를 left/right로, row selection을 physical-last/last-valid로 교차한다. 실제 코드를 임의 수정하지 못한다면 source predicate와 기존 test로 예상 결과를 작성한다. physical-last+right에서만 실패하면 row selection 가설이 강하다. 모든 right case가 실패하면 mask/position/backend support를 본다. graph에서만 실패하면 metadata generation을 추가 축으로 둔다.

네 번째 5분에는 truncation을 분리한다. raw prompt length, effective context limit, reserved output budget, final input length를 적는다. 잘린 token 수만 기록하지 말고 제거된 semantic atom을 표시한다. required system/tool/media/generation atom 중 하나가 부분 제거되면 model output을 기다리지 않고 policy failure로 분류한다. 아무 atom도 깨지지 않았는데 stop이 다르면 prompt suffix와 special ID, stopping criteria를 본다.

다섯 번째 5분에는 owner를 source에 고정한다. request option parser, tokenizer/collator, batch builder, model input preparation, runner pack, sampler gather, stop checker 중 최초 mutation과 next consumer를 한 쌍으로 적는다. “Transformers 문제”, “vLLM batching 문제”처럼 repository 전체를 owner로 쓰지 않는다. 값이 만들어진 symbol과 잘못 해석한 symbol 사이의 edge가 수정 단위다.

마지막 5분에는 임시 완화와 근본 수정을 분리한다. left padding 강제, eager fallback, 긴 요청 reject는 blast radius를 줄일 수 있지만 계약 위반의 원인을 없앴다는 뜻은 아니다. 완화에는 적용 cohort, 성능 비용, 해제 조건을 붙인다. 근본 수정에는 실패 tensor fixture, PAD=EOS 반례, graph generation, atom-aware truncation test를 붙인다. rollback은 tokenizer·template·model·cache generation을 같은 manifest로 되돌린다.

이 절차가 항상 30분 안에 root cause를 찾는다는 뜻은 아니다. 30분 안에 조사 층을 좁히고, 다음 담당자에게 재현 배열과 최초 경계를 넘기는 것이 목표다. 좋은 handoff는 “batch일 때 이상함”이 아니라 “right-padded short row의 valid positions는 single과 같지만 sampler가 physical row 2047을 선택하며, eager last-valid reference는 row 37을 선택한다”처럼 측정 가능한 문장이다.

운영 metric만으로 이 장부를 완성하려 해서는 안 된다. fleet metric은 길이 bucket, padding/truncation policy, backend, graph 여부, rejection reason의 분포를 보여 준다. 개별 IDs와 mask는 high-cardinality이며 민감할 수 있다. sampled secure trace나 synthetic canary에서 tensor artifact를 얻고 metric으로 범위를 추정한다. Prometheus label에 prompt digest나 token array를 넣는 것은 cardinality와 정보보호 양쪽에서 나쁘다.

synthetic canary는 작은데도 강해야 한다. 길이 1과 최대 길이, PAD=EOS, 별도 PAD, 두 sequence 길이 차가 큰 batch, left/right, graph bucket 경계, context limit-1/limit/limit+1을 포함한다. tool canary는 declaration과 result를 한 atom으로 보존하고, multimodal canary는 placeholder와 feature count를 확인한다. 기대 자연어는 느슨하게 두되 first logits 또는 selected ID, stop reason, manifest는 엄격히 비교한다.

성능 숫자도 같은 artifact에 연결한다. right에서 left로 바꾼 뒤 처리량이 5% 떨어졌다면 correctness 수정의 비용인지 batch composition 변화인지 분리한다. 동일 final IDs와 고정 decode length에서 scheduled token, kernel shape, graph hit를 비교한다. truncation 정책 변경 뒤 TTFT가 늘었다면 보존한 token 수가 늘어난 효과와 cache hit 변화, CPU planning 비용을 나눈다. 의미를 지키는 데 든 비용을 숨기지 않아야 다음 최적화가 올바른 층을 겨냥한다.

검토자는 특히 세 가지 쉬운 결론을 경계한다. 첫째, warning이 사라졌으니 해결됐다는 결론이다. warning predicate와 wrong output predicate가 같다는 증거가 필요하다. 둘째, single request가 맞으니 model weight는 정상이라는 결론이다. batch-only metadata나 graph path가 다른 weight view를 선택할 수도 있다. 셋째, output text가 같으니 truncation이 안전하다는 결론이다. control이 우연히 지켜졌거나 hidden tool capability가 이미 사라졌을 수 있다.

최종 판정 레코드는 짧아도 된다. `first_divergence`, `producer`, `consumer`, `failing_fixture`, `passing_neighbor`, `mitigation`, `fix`, `rollback_generation`, `correctness_terminal`, `performance_terminal` 열을 채운다. 각 값은 이 장의 source anchor나 보존 artifact를 가리킨다. 독자는 이 레코드만으로 어느 함수를 더 읽고 어느 배열을 다시 비교할지 결정할 수 있어야 한다.

8장의 핵심은 padding side나 truncation option의 권장값을 하나 정하는 데 있지 않다. option이 배열, 좌표, semantic atom, scheduler metadata, sampler와 stop state를 어떤 순서로 바꾸는지 설명하는 데 있다. 이 mutation chain을 갖고 있으면 모델과 backend가 달라져도 같은 질문을 재사용할 수 있다. 다음 장은 이 장에서 승인한 token ID와 position이 실제 embedding row와 위치 표현으로 변환되는 순간부터 시작한다.

인수인계 전에 실제 수치 장부를 한 번 더 검산한다. batch row 세 개의 유효 길이가 5, 17, 64이고 dense length 64라면 physical slot은 192, semantic token은 86이다. 단순 padding waste는 `(192-86)/192 = 55.2%`다. 그러나 packed attention이 86 query만 계산한다면 이 비율을 kernel waste라고 보고하면 틀린다. host tensor materialization, graph bucket, KV allocation, attention query 가운데 어디가 192를 소비하고 어디가 86을 소비하는지 분리한다. 같은 “padding 55.2%”가 메모리 복사 문제일 수도, 실제 compute와 무관한 표현일 수도 있다.

truncation 장부도 보존량만 계산하지 않는다. 입력 163에서 103으로 줄였으므로 60 token, 36.8%를 제거했다. 하지만 제거한 것이 두 개의 완전한 history turn이면 protocol validity는 유지된다. 반대로 35 token만 제거해 128을 꽉 채우면서 system 30과 tool delimiter 5를 잘랐다면 utilization은 높지만 제어 계약은 무너졌다. 최적화 지표에는 retained useful history, required atom preservation, rejected request, fallback lane을 함께 둔다.

비동기 serving에서는 admission과 execution 사이의 정책 generation도 확인한다. request가 queue에 들어갈 때 tokenizer/template generation G1으로 128 token이었는데 실행 전에 G2가 로드돼 special marker가 늘어나면, 이미 승인된 길이와 실제 input이 달라질 수 있다. queue state가 final IDs를 소유하는지 raw messages를 소유해 worker가 다시 compile하는지에 따라 책임이 갈린다. 재compile한다면 budget과 cache key, usage도 같은 generation에서 다시 계산해야 한다.

streaming 요청 취소도 edge case다. 첫 token 전에 취소된 request의 padded row와 slot mapping이 다음 batch에서 재사용될 때 length와 selected-row metadata가 함께 초기화돼야 한다. tombstone만 남고 graph buffer의 last index가 재사용되면 다른 sequence의 valid logits를 읽을 수 있다. 회귀 fixture에는 cancellation 직후 동일 bucket 재사용을 넣고 buffer generation과 sequence ID를 확인한다.

speculative decoding이 켜지면 selected row와 stop trace가 한 단계 더 늘어난다. draft가 여러 token을 제안하고 target이 일부를 accept할 때 prompt의 last-valid row, draft positions, target verification positions, accepted length가 일관되어야 한다. padding 오류가 draft에서만 발생하면 acceptance pattern이 바뀌어 성능 문제처럼 보일 수 있다. truncation으로 stop marker가 사라지면 draft가 길게 제안하고 target이 반복 reject할 수도 있다. baseline과 speculative lane을 나눠 최초 tensor 분기를 찾는다.

adapter와 quantization도 manifest 열에 들어가지만 8장의 owner로 성급히 끌어오지 않는다. LoRA가 embedding이나 lm-head를 수정할 수 있고 quantized kernel이 padding shape 제약을 가질 수 있으므로 비교 cohort는 고정해야 한다. 네 tensor가 이미 다르면 adapter 수치 오차를 조사하기 전에 sequence-shaping 경계를 닫는다. 네 tensor와 model identity가 같고 logits만 다를 때 다음 층으로 넘긴다.

분산 실행에서는 tensor-parallel rank마다 동일한 logical metadata를 보는지 확인한다. token IDs는 broadcast됐지만 position이나 selected indices가 rank-local stale buffer라면 collective는 정상 종료해도 logits가 잘못될 수 있다. 모든 rank의 전체 배열을 상시 로그할 필요는 없다. generation, shape, digest, min/max와 small failing fixture를 비교해 divergence rank를 좁힌다. collective hang이 아니라 silent semantic mismatch라는 점을 명확히 한다.

이 모든 확장은 “가능한 원인을 많이 나열”하려는 것이 아니다. 조사 순서는 여전히 좁다. final IDs, mask, logical position, cache position, selected row, first logits, selected token, stop, visible text다. speculative·graph·distributed 같은 기능은 이 사슬에 producer/consumer를 추가할 뿐이다. 최초로 달라진 값을 찾으면 뒤의 현상은 결과로 접고 앞의 후보만 조사한다.

따라서 독자가 남겨야 할 최종 문장은 구체적이다. “G2 right-padding graph lane에서 B의 logical last position은 1이고 mask도 정확하지만 sampler index buffer가 이전 batch의 physical 63을 유지한다. eager와 fresh graph capture는 index 1을 사용해 single reference와 일치한다.” 이 문장은 원인, 반증, 완화 범위를 동시에 준다. 또는 “raw left truncation이 required tool atom 50개 중 앞 35개를 제거했으며 validator generation은 새 schema를 기대한다. atom-aware plan은 history 60개를 제거하고 required 103개를 보존한다.”처럼 정책 사건을 닫는다.

코드 리뷰에서는 option 선언과 default만 보고 승인을 내리지 않는다. `padding_side`가 tokenizer 객체에 저장된 뒤 어느 collator가 읽는지, server가 unpadded list로 다시 바꾸는지, runner가 position과 slot을 다시 만드는지 확인한다. `truncation=True`도 실제 limit 결정, slice 방향, overflow return, error mapping, usage accounting을 잇는다. option이 존재한다는 사실과 현재 serving 경로에서 소비된다는 사실은 다르다.

테스트 이름도 계약을 대신하지 않는다. `test_left_padding`이 output text 하나만 비교한다면 mask와 cache position의 stale generation을 놓칠 수 있다. assertion이 어느 중간값을 고정하는지 읽고 빈 경계를 새 fixture로 보완한다. 반대로 구현 세부의 physical row 번호만 고정하면 packed backend 전환을 불필요하게 막는다. logical sequence ID, token ordinal, first logits parity처럼 의미 경계를 assertion으로 선택한다.

문서화에는 지원 조합과 비지원 조합을 명시한다. 특정 backend가 right padding을 지원하지 않으면 자동으로 left로 바꾸고 조용히 계속하기보다 effective policy와 reason을 trace에 남긴다. graph capture가 특정 length bucket에서만 안전하다면 fallback 여부와 성능 영향을 노출한다. 사용자는 requested option이 아니라 실제 선택된 경로를 알아야 회귀를 재현할 수 있다.

마지막으로 실패가 재현되지 않을 때도 artifact를 버리지 않는다. failing deployment generation, batch neighbors, queue ordering, cache hit, graph bucket이 빠졌을 가능성을 기록한다. 단독 replay의 성공은 원래 사건의 반증이 아니라 batch context가 원인의 일부라는 단서일 수 있다. 가까운 passing neighbor와 failing cohort를 함께 보존하면 재현 조건을 다시 조립할 수 있다.

이렇게 8장은 단어의 정의에서 끝나지 않는다. special token과 padding, truncation이 실제 배열과 상태를 어떻게 mutate하고, 그 변화가 position·cache·sampler·stop으로 어떻게 소비되는지 한 방향으로 읽게 한다. 다음 장에서 embedding OOR나 첫 hidden-state 차이를 만났을 때 이미 닫힌 입력 계약을 기준선으로 삼을 수 있다.

독자가 직접 확인할 마지막 연습은 의도적으로 모호한 장애 문장을 고치는 일이다. “긴 요청이 batch에서 가끔 틀린다”를 길이 bucket, padding side, effective backend, graph generation, final ID digest, first divergent tensor가 들어간 문장으로 바꾼다. 그 뒤 raw-token truncation과 atom-aware plan의 final manifest를 나란히 놓고 어떤 control이 보존됐는지 설명한다.

답을 쓸 때 추천 option 하나로 끝내지 않는다. 요청한 값이 어느 parser에서 받아지고, 어떤 state를 mutate하며, 어느 consumer가 배열이나 index로 바꾸고, 어떤 metric과 fixture가 효과를 판정하는지 적는다. 지원되지 않는 조합에서는 reject, fallback, silent coercion 중 무엇이 일어나는지도 구별한다. 이 연습을 통과하면 새로운 serving stack에서도 option 문서와 실제 실행 의미 사이의 간극을 스스로 좁힐 수 있다.

마지막 산출물에는 확정 사실과 아직 관찰하지 못한 경계를 따로 표시한다. 빈칸을 정상값으로 간주하지 않아야 다음 조사자가 같은 추측을 반복하지 않는다. 관찰 불가한 state에는 필요한 probe와 예상 owner를 붙여 handoff를 닫는다.

## 8.12 소스 노트와 다음 장 handoff

Transformers의 기준 revision은 `550d7b3834670483a4df436541272c055dc364bf`다. tokenizer side validation, `_pad`, special-token 추가, embedding resize, generation mask와 right-padding 검사를 같은 revision에서 읽었다. 문서 option 설명보다 이 호출 흐름을 우선한 이유는 config 값이 실제 배열과 generation state를 어디서 바꾸는지 고정하기 위해서다.

vLLM의 기준 revision은 `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`다. OpenAI protocol의 입력 표면, `SamplingParams`의 EOS/출력 상태, prompt length validation을 연결했다. vLLM의 continuous batching을 Transformers식 dense padding과 동일시하지 않았으며, 실제 logical request와 scheduler trace를 확인하도록 범위를 제한했다.

SGLang의 기준 revision은 `71de97b264b04dcd514cf904003028aefe9775c8`다. tokenizer manager와 long-input utility, schedule batch의 원본·실행 IDs와 EOS, detokenizer manager의 grouping/incremental decode를 따라갔다. 자동 truncation이 grammar-aware 보존을 자동 보장한다는 주장은 하지 않았다.

llama.cpp의 기준 revision은 `bb4caa7540188872173c44d161602d9271386413`다. GGUF metadata key, vocabulary의 BOS/EOS 처리, common tokenize/detokenize helper, saver와 speculative compatibility 경로를 사용했다. 파일/줄 링크는 해당 revision의 의미를 고정하며, 이후 revision에서는 이름과 책임 경계가 이동할 수 있다.
