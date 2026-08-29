# 6장. 토크나이저는 문자열을 어떻게 경계 짓는가

어느 날 한국어 질문 하나가 새 모델 배포 뒤 갑자기 세 토큰 길어졌다고 하자. 답의 품질은 조금 달라졌고, 같은 system prompt를 공유하던 요청의 prefix cache hit도 줄었다. GPU trace에는 이상이 없다. 모델 weight도 바뀌지 않았다. 운영자는 흔히 “토크나이저가 단어를 조금 다르게 잘랐나 보다”라고 말하고 넘어간다. 그러나 이 문장은 원인을 설명하지 못한다. 어느 Unicode 문자열을 어떤 byte로 보았고, 무엇을 정규화했으며, 어느 경계에서 후보 조각을 만들고, 어떤 subword 알고리즘이 어느 vocabulary ID를 골랐는지를 하나도 말하지 않기 때문이다.

이 장의 질문은 단순하다. **사람에게 같은 글처럼 보이는 입력이 왜 다른 token ID 열이 되며, 그 최초 차이를 코드에서 어떻게 찾는가?** 답을 얻고 나면 토크나이저를 모델 앞의 편의 도구가 아니라 실행 identity를 만드는 구성 요소로 보게 된다. token ID가 달라지면 embedding lookup 행이 달라지고, 모든 뒤 layer의 hidden state가 달라진다. token 수가 달라지면 prefill 계산량과 KV cache block 수가 달라진다. 공통 prefix가 달라지면 cache reuse도 달라진다.

## 6.1 원문 byte에서 시작해 최초 차이를 찾는다

텍스트 편집기에 `café`가 보인다고 하자. 첫 번째 문자열은 `é`라는 code point 하나를 쓸 수 있다. 두 번째 문자열은 `e` 뒤에 결합 악센트 code point를 붙일 수 있다. 화면 renderer는 둘을 거의 같게 그린다. Python의 문자열 비교와 UTF-8 byte 열은 같지 않을 수 있다. 정규화기가 둘을 합치면 같은 token 열이 될 수 있고, 정규화를 하지 않으면 다른 경로를 갈 수 있다.

이 차이를 이해하는 데 “우편물을 분류한다”는 비유가 도움이 된다. 눈에 보이는 문장은 봉투의 겉모습이고, Unicode code point는 주소를 적는 문자 체계이며, UTF-8 byte는 실제 운송 라벨이다. normalizer는 주소 표기를 표준화하고, pre-tokenizer는 우편물을 묶음으로 나누고, subword model은 각 묶음에 vocabulary 번호를 붙인다. 다만 비유에는 한계가 있다. 현실의 우편물은 표준화 뒤에도 같은 물건이지만, 텍스트 정규화는 대소문자·악센트·공백 같은 정보를 실제로 없앨 수 있다. 모델이 받는 입력 의미가 달라질 수 있다는 뜻이다.

토큰화 경로를 다음처럼 여섯 층으로 그릴 수 있다.

```text
사용자에게 보이는 문자열
→ Unicode code point 열
→ 정규화된 문자열
→ pre-tokenized 조각과 경계
→ subword/byte token 조각
→ vocabulary integer ID 열
```

이 장에서는 이 순서를 하나의 고정 관측선으로 사용한다. 입력 byte와 JSON 해석 뒤 문자열을 먼저 고정하고, normalization 결과, pre-tokenization 조각, vocabulary ID, decode 결과를 차례로 비교한다. 뒤 단계가 다르더라도 앞 단계가 같다고 확인되기 전에는 원인을 뒤로 넘기지 않는다. 이 원칙이 Unicode·fast/slow·cache 사고를 하나의 회귀 양식으로 묶는다.

| 단계 | 반드시 남길 값 | 같으면 다음에 볼 곳 | 다르면 소유자 |
|---|---|---|---|
| 입력 | UTF-8 byte 길이·digest·허용된 bounded 표본 | normalization | HTTP·JSON·template 경계 |
| 정규화 | 정책 digest·정규화 문자열 digest | pre-tokenization | normalizer artifact·구현 |
| 경계 | 조각·원문 offset·offset 단위 | vocabulary lookup | pre-tokenizer·AddedToken |
| ID | token 문자열·integer ID·vocabulary digest | decode | subword model·merge·vocabulary |
| 복원 | 누적 ID별 확정 byte와 출력 문자열 | serving 소비자 | decoder·stream buffer·cleanup 정책 |

여기에 special token을 붙이는 post-processing과 길이 제한이 더해질 수 있다. 이 장에서는 special token, padding, truncation의 정책 자체는 8장으로 넘긴다. 다만 added token이 base tokenizer보다 먼저 경계를 가로채는 부분은 여기서 다룬다. 경계 알고리즘을 이해하려면 그 예외를 빼놓을 수 없기 때문이다.

### “문자 하나”라는 단위는 하나가 아니다

운영 로그에서 `len(text)`만 남기면 어느 단위를 셌는지 모호하다. Python `str` 길이는 대체로 Unicode code point 수를 세지만, 사용자가 한 글자로 느끼는 grapheme cluster 수와 다를 수 있다. UTF-8 byte 수는 또 다르다. token 수는 vocabulary와 알고리즘에 달렸다. 다음 네 길이를 분리해서 생각해야 한다.

```text
grapheme cluster 수  — 사람이 대략 글자라고 느끼는 단위
code point 수        — Unicode scalar sequence
UTF-8 byte 수        — byte-level tokenizer가 보는 기초 재료
token ID 수          — 모델이 실제로 보는 sequence 길이
```

가령 한글 음절 `가`는 완성형 code point 하나로 쓸 수도 있고, 자모 `ᄀ`과 `ᅡ`의 조합으로 쓸 수도 있다. NFC 정규화는 조합 가능한 sequence를 합성하고, NFD는 분해한다. NFKC/NFKD는 compatibility 문자를 더 적극적으로 접거나 분해한다. 어느 쪽이 “정답”인지는 언어학만으로 정해지지 않는다. 모델이 학습될 때 쓴 tokenizer artifact와 같은 정책을 적용하는 것이 serving correctness의 기준이다.

Transformers의 fast backend는 serialized tokenizer 안의 normalizer, pre-tokenizer, model, post-processor, decoder를 함께 사용한다. `TokenizerBackend`가 tokenizer JSON을 다루며 precompiled SentencePiece charsmap을 normalizer 설정에서 읽는 경로는 [Transformers v5.15.1 `tokenization_utils_tokenizers.py:110-178`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L110-L178)에서 확인할 수 있다.

여기서 보이는 사실은 “SentencePiece이면 항상 같은 정규화”가 아니라 artifact 안의 normalizer 구성이 실제 backend에 전달된다는 점이다.

### 첫 divergence를 찾는 네 지문

두 환경의 token ID가 다르면 원문을 눈으로 대조하는 대신 네 지문을 남긴다.

1. 원래 문자열의 Unicode code point 표기와 UTF-8 digest
2. normalizer 뒤 문자열 또는 정규화 정책 digest
3. pre-tokenized 조각과 원문 offset
4. 최종 token IDs와 tokenizer artifact/revision digest

개인정보가 있는 원문 전체를 production log에 남기라는 뜻이 아니다. 승인된 재현 fixture에서는 code point와 byte를 볼 수 있고, 운영에서는 길이·digest·token count·bounded prefix를 쓸 수 있다. 중요한 것은 “같은 prompt”라는 사람의 판정을 기계적으로 검증 가능한 identity로 바꾸는 것이다.

이 네 지문을 비교했을 때 UTF-8부터 다르면 입력 수집이나 normalization 전 문제다. byte는 같고 normalized text가 다르면 normalizer config 또는 구현 차이다. normalized text는 같고 조각이 다르면 pre-tokenizer다. 조각까지 같고 ID가 다르면 vocabulary/model/merge 또는 added token 상태를 본다. 이 분기만으로도 embedding이나 CUDA까지 무턱대고 내려가는 일을 피할 수 있다.

### 눈으로 같은 입력을 손으로 해부해 보기

`가`, `é`, `🙂`를 한 줄에 놓고 세 단위를 직접 적어 보면 추상적인 경계가 금세 구체화된다. `가`의 완성형 code point는 `U+AC00`이고 UTF-8에서는 세 byte다. 같은 모양을 현대 한글 자모 `U+1100 U+1161`로 쓰면 code point가 둘이고 UTF-8 byte도 여섯 개다. NFC가 적용되면 두 번째 표현은 첫 번째와 합쳐질 수 있다. 그러나 tokenizer가 byte mapping을 먼저 수행하거나 정규화 정책이 다르면 처음부터 후보 symbol 수가 달라진다.

`é`도 같다. 합성형 `U+00E9`는 UTF-8 `C3 A9`이고, 분해형 `U+0065 U+0301`은 `65 CC 81`이다. NFD를 적용하면 합성형이 분해형으로 가고 NFC를 적용하면 반대 방향으로 간다. accent stripping이 뒤따르면 결합 악센트가 제거되어 평범한 `e`만 남을 수 있다. 여기서 “NFC는 글자를 보존한다”는 표현에도 한계가 있다. 사람이 느끼는 글꼴 모양을 대체로 보존할 뿐, byte identity와 code point identity는 바꾼다. 서명 검증이나 원문 byte span이 필요한 시스템에서는 이 차이가 correctness다.

`🙂`는 code point 하나지만 UTF-8 네 byte다. byte-level vocabulary가 네 byte 각각을 초기 symbol로 표현하고 merge가 충분히 학습되지 않았다면 token 네 개 가까이로 남을 수 있다. 반대로 vocabulary에 대응되는 결합 조각이 있으면 하나 또는 더 적은 token이 된다. 피부색 modifier나 zero-width joiner로 연결한 emoji sequence는 code point가 여러 개면서 화면에서는 하나의 grapheme cluster로 보일 수 있다. UI의 “마지막 한 글자를 지운다”는 동작과 token stream의 “마지막 token을 지운다”는 동작은 서로 바꿔 쓸 수 없다.

좀 더 까다로운 fixture는 보이지 않는 문자다. 일반 공백 `U+0020`, no-break space `U+00A0`, thin space `U+2009`, zero-width space `U+200B`는 화면에서 같거나 거의 보이지 않는다. compatibility normalization이나 whitespace pre-tokenizer가 일부를 같은 것으로 접을 수 있지만 모두가 항상 같은 취급을 받지는 않는다. JSON pretty printer, 웹 브라우저 복사, PDF 추출기가 이 문자를 바꾸면 사용자는 같은 prompt라고 믿고 token ID는 달라진다. 그래서 재현 자료에는 화면 캡처보다 `unicode_escape`에 해당하는 표기와 byte digest가 더 강한 증거가 된다.

잘못된 UTF-8도 경계를 시험한다. Python `str`까지 올라온 입력은 대개 이미 decoder가 invalid byte를 거부하거나 replacement character로 바꾼 뒤다. 반면 C API가 byte pointer와 길이를 직접 받거나, 프록시가 임의 byte body를 처리하면 원래 byte가 더 오래 살아 있을 수 있다. replacement character `U+FFFD`로 바뀐 뒤에는 어떤 invalid byte였는지 되돌릴 수 없다. “byte fallback이 있으니 임의 입력을 무손실 보존한다”는 말은 tokenizer가 원래 byte를 실제로 받았을 때만 성립한다. 그 전에 HTTP·JSON·언어 runtime decoder가 정보를 버렸다면 tokenizer는 복구할 수 없다.

이 손계산의 목적은 Unicode 표를 외우는 데 있지 않다. 사고가 났을 때 어느 층의 관찰값을 요구해야 하는지 익히는 데 있다. 사용자가 보낸 HTTP body의 byte, JSON parser 뒤 string, template renderer 뒤 string, tokenizer normalizer 뒤 표현은 같은 객체가 아니다. 각 단계의 owner와 digest가 있어야 “토크나이저가 바꿨다”와 “토크나이저에 도착하기 전에 바뀌었다”를 구별할 수 있다.

## 6.2 normalization 뒤에 pre-tokenization 경계를 만든다

정규화의 질문은 “다른 표기를 같은 것으로 취급할 것인가?”다. pre-tokenization의 질문은 “subword model이 어느 범위를 서로 합칠 수 있게 할 것인가?”다. 둘은 순서도, 정보 손실도, 디버깅 방법도 다르다.

lowercase normalizer를 쓰면 `Apple`과 `apple`은 같은 normalized text가 될 수 있다. 이는 vocabulary 효율을 높일 수 있지만 고유명사·약어·코드의 case 정보를 없앤다. accent stripping은 검색용 모델에는 유리할 수 있으나 언어 구별을 흐릴 수 있다. Unicode compatibility normalization은 전각/반각이나 ligature를 접지만, 원문 보존이 필요한 작업에는 손실이다. 이런 정책은 일반적으로 좋거나 나쁜 것이 아니라 학습 artifact와 일치해야 한다.

pre-tokenizer는 공백, 구두점, byte boundary, metaspace 같은 규칙으로 입력을 조각낸다. 조각 경계 밖으로 BPE merge를 허용하지 않는 구현이라면, 공백을 사이에 둔 두 단어는 merge table에 pair가 있어도 합쳐지지 않는다. GPT-2 계열 byte-level 방식은 공백을 다음 단어의 표식처럼 보존하는 경우가 있어 `hello`와 ` hello`가 다른 token이 될 수 있다. SentencePiece의 metaspace 표식도 공백 정보를 조각에 넣는다.

### 순서가 바뀌면 왜 결과가 달라지는가

다음 입력을 생각하자.

```text
"  Café\tCAFÉ  "
```

여기서 마지막 `É`는 `E`와 결합 문자를 쓴 것으로 가정한다. 먼저 NFC와 lowercase를 적용한 뒤 whitespace split을 하면 두 단어가 모두 `café`가 될 수 있다. 먼저 byte-level pre-tokenization을 하고 각 조각을 제한적으로 정규화한다면 byte offset과 경계가 달라질 수 있다. normalizer가 공백을 collapse하는지, pre-tokenizer가 leading whitespace를 별도 표식으로 보존하는지에 따라 최종 입력도 달라진다.

이것은 함수 순서의 사소한 구현 차이가 아니다. offset mapping의 의미와 cache identity를 바꾼다. 원문 character span을 token span에 매핑하는 NER·scoring·highlight 기능은 normalized text와 original text 사이 alignment를 필요로 한다. 정규화가 한 code point를 두 개로 늘리거나 여러 code point를 하나로 줄이면 단순 index equality가 성립하지 않는다.

Transformers fast backend가 batch encoding에서 각 encoding의 `offsets`를 `offset_mapping`으로 내보내는 곳은 [Transformers v5.15.1 `tokenization_utils_tokenizers.py:739-784`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L739-L784)다. 반면 base class의 API 계약은 slow tokenizer가 offset mapping을 제공하지 못할 수 있음을 감안해야 한다. fast/slow를 속도 차이로만 부르면 안 되는 첫 이유다.

### offset은 byte인가, character인가

`(start, end)`라는 tuple만 보고 단위를 추측하면 안 된다. backend API가 original string의 character offset을 주는지, normalized string이나 byte offset을 주는지 확인해야 한다. emoji와 한글처럼 UTF-8에서 여러 byte를 쓰는 입력을 fixture에 넣으면 단위 혼동을 빨리 드러낼 수 있다.

예를 들어 원문 `A가B`의 Python code point index에서 `가`는 `[1,2)`지만 UTF-8 byte에서는 `[1,4)`다. token offset `[1,2)`를 byte slice에 쓰면 중간 byte를 잘라 invalid UTF-8을 만든다.

반대로 byte offset `[1,4)`를 Python string slice에 쓰면 엉뚱한 범위를 잡는다. vLLM scoring utility는 fast tokenizer의 `offset_mapping`을 이용해 원문을 정확한 character 경계에서 자르는 경로를 둔다. [vLLM v0.27.1 `pooling/scoring/utils.py:38-59`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/pooling/scoring/utils.py#L38-L59)에서 max token 위치의 end offset으로 원문 slice를 정하는 코드를 볼 수 있다. 이 코드는 offset contract가 틀리면 scoring 입력 자체가 바뀌는 소비자 사례다.

### normalizer를 독립적으로 버전 관리해야 하는 이유

vocabulary 파일과 merge 파일이 같아도 normalizer가 다르면 token 열이 달라질 수 있다. tokenizer JSON digest만 기록하면 bundle 전체를 잡을 수 있지만, 원인 분석에는 component digest도 유용하다. model update에서 다음을 원자적으로 묶는다.

```text
normalizer config
pre-tokenizer config
subword model/vocabulary/merges
added/special token map
post-processor
decoder config
```

### 정규화와 경계의 교환법칙은 대개 성립하지 않는다

두 함수 `N`을 normalization, `P`를 pre-tokenization이라고 쓰자. 많은 입력에서 `P(N(x))`와 각 pre-token 조각에 `N`을 적용한 결과는 같지 않다. 정규화가 공백이나 구두점을 만들거나 없애고, 한 code point를 여러 code point로 분해하기 때문이다. 이 사실은 수학 기호를 쓰기 위한 장식이 아니라 configuration migration에서 반드시 확인할 불변조건이다.

가령 pre-tokenizer가 ASCII apostrophe를 경계로 삼고 normalizer가 curly apostrophe `’`를 ASCII `'`로 바꾼다고 하자. 먼저 정규화하면 `don't`의 경계 후보가 생긴다. 먼저 pre-tokenize하면 curly apostrophe를 일반 문자로 보고 단어 전체를 한 조각으로 넘길 수 있다. 이후 각 조각 안에서 apostrophe가 바뀌어도 이미 닫힌 경계 밖으로 subword merge가 나갈 수 없다. 최종 vocabulary와 merge가 완전히 같아도 ID 열이 달라지는 이유다.

공백 collapse도 순서에 민감하다. 입력 `A··B`에서 `·`를 공백이라고 표시하자. normalizer가 연속 공백을 하나로 줄이고 metaspace pre-tokenizer가 각 단어 앞 공백을 `▁`로 보존하면 결과는 `A`, `▁B`에 가깝다. metaspace 변환이 먼저 두 공백을 각각 표식으로 만들고 뒤 normalizer가 그 표식을 공백으로 보지 않으면 두 개가 남을 수 있다. 실제 구현의 정확한 규칙은 artifact를 읽어야 하지만, “같은 구성요소 집합이면 순서와 무관하다”는 가정이 틀렸다는 점은 일반적이다.

정규화의 멱등성도 시험할 가치가 있다. 이상적인 canonical normalizer `N`에는 `N(N(x)) = N(x)`를 기대하지만, custom replacement sequence나 서로 충돌하는 rule을 연결하면 그렇지 않을 수 있다. API gateway와 model server가 같은 normalizer를 각각 실행한다고 해서 늘 안전한 것이 아니다. double normalization이 결과를 바꾸면 gateway를 거친 요청과 direct request가 다른 token IDs를 만든다. 한 계층만 normalization owner로 정하고, 다른 계층은 원문을 보존하는 편이 원인 추적에 유리하다.

pre-tokenizer의 경계 또한 단순한 split 결과보다 많은 정보를 가진다. 어떤 backend는 공백을 버리는 대신 다음 조각에 붙이고, 어떤 backend는 byte-level alphabet으로 치환하며, 어떤 backend는 원문 offset alignment를 유지한다. 조각 문자열만 로깅하면 공백이 어디에 귀속되었는지 잃을 수 있다. 디버그 fixture에서는 최소한 `(piece, original_start, original_end)`를 함께 본다. normalization으로 길이가 바뀌면 normalized span과 original span의 alignment도 따로 필요하다.

정규식 pre-tokenizer를 평가할 때 평균 문장만 쓰면 pathological input을 놓친다. 긴 구두점 열, 긴 숫자, 소스 코드의 snake_case와 camelCase, URL, base64, JSON schema, 반복 공백을 넣는다. 이들은 자연어와 다른 후보 분포를 만들고 CPU 시간과 token 수를 동시에 늘릴 수 있다. 다만 정규식이 보인다고 곧바로 catastrophic backtracking을 주장해서는 안 된다. 실제 regex engine과 pattern, 선형화 여부를 확인해야 한다. 여기서 필요한 태도는 “가능한 원인”과 “소스로 확인한 원인”을 분리하는 것이다.

offset의 기준 문자열도 계약으로 고정한다. 원문 `Cafe\u0301`가 NFC 뒤 `Café`가 되었다면 `é` token의 원문 span을 `[3,5)`로 돌려줄 수도 있고, normalized span `[3,4)`를 돌려줄 수도 있다. 두 tuple은 모두 내부적으로 일관될 수 있지만 소비자가 기대한 기준과 다르면 버그다. UI highlight는 원문 span이 필요하고, normalized buffer를 직접 slice하는 내부 scorer는 normalized span이 필요할 수 있다. 필드 이름이 단지 `offsets`라면 문서와 구현을 함께 읽어야 한다.

llama.cpp의 GGUF conversion 경로도 tokenizer normalizer 정보를 metadata로 옮길 수 있다. `vocab.py`는 `Lowercase`, `StripAccents`, `BertNormalizer`, nested `Sequence`를 해석해 normalizer flags를 만든다. 고정 구현은 [llama.cpp v0.2.0 `gguf-py/gguf/vocab.py:161-181`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/gguf-py/gguf/vocab.py#L161-L181)에 있다. 이 경로는 GGUF가 weight만 담는 그릇이 아니라 tokenization 의미 일부도 운반한다는 증거다. 다만 모든 upstream normalizer의 임의 동작을 두 boolean flag로 완전히 표현할 수 있다고 해석해서는 안 된다.

변환기가 보존하는 metadata 범위를 검증해야 한다.

## 6.3 경계 안에서 vocabulary ID를 선택한다

subword 알고리즘을 “자주 나오는 문자열을 token으로 만든다”라고 묶으면 운영에 필요한 차이를 잃는다. BPE는 merge 순서에 따라 조각을 합친다. WordPiece inference는 보통 현재 위치에서 vocabulary에 있는 가장 긴 조각을 탐욕적으로 고르고 continuation 표식을 쓴다. Unigram은 후보 조각에 score를 주고 전체 경로의 비용이 좋은 segmentation을 고른다. SentencePiece는 BPE 또는 Unigram 같은 model과 raw-text 중심 처리·normalization·공백 표식을 묶는 toolkit/model format이라는 점을 구별해야 한다.

### BPE: merge 순서가 프로그램이다

작은 vocabulary를 만들자. 초기 symbol은 문자 또는 byte 단위라고 가정한다.

```text
입력: lower
초기: l o w e r
merge 순서:
1. l + o  → lo
2. lo + w → low
3. e + r  → er
4. low + er → lower
```

모든 merge가 적용되면 `lower` 하나가 된다. 하지만 merge 4가 없으면 `low`, `er` 두 token이다. merge table에 `o+w`도 있더라도 rank가 뒤이고 앞 merge가 `l+o`를 먼저 바꾸면 적용 가능한 pair의 상태가 달라진다. 구현은 매 순간 가장 우선순위 높은 pair를 고르거나 동등한 결과를 내는 자료구조를 쓴다.

이제 `lowest`를 넣어 보자.

```text
l o w e s t
→ lo w e s t
→ low e s t
```

`er`는 없고 `est` 관련 merge가 없다면 나머지는 잘게 남는다. BPE는 사전에서 가장 긴 문자열을 무조건 고르는 방식이 아니다. 학습된 merge 순서가 segmentation program이다. tokenizer update에서 vocabulary token 목록은 같아 보이는데 merge rank가 달라지면 결과가 바뀔 수 있다.

byte-level BPE에서는 Unicode string을 UTF-8 byte로 바꾼 뒤 모든 byte를 안전하게 표현할 alphabet에 매핑할 수 있다. 장점은 임의 byte를 unknown 하나로 붕괴시키지 않는다는 것이다. 대가는 사용자가 느끼는 한 글자가 여러 byte token으로 갈라질 수 있고, 중간 token만 decode하면 invalid UTF-8 조각이 될 수 있다는 점이다. byte fallback은 “모든 언어를 의미 있는 단위로 잘 이해한다”는 보장이 아니라 “입력 byte를 표현할 탈출구가 있다”는 보장이다.

### WordPiece: longest match와 unknown 붕괴

다음 vocabulary를 가정하자.

```text
["play", "player", "##er", "##ing", "##s", "un", "##known", "[UNK]"]
```

단어 `players`를 처리할 때 처음 위치에서는 continuation 표식이 없는 후보를 찾는다. `player`가 가장 길게 맞고, 남은 `s`는 `##s`가 맞는다. 결과는 `player`, `##s`다. `play`, `##er`, `##s`도 가능한 segmentation이지만 greedy longest-match라면 선택되지 않는다.

`playing`은 `play`, `##ing`이 된다. 그러나 `playx`에서 `##x`가 없고 fallback을 더 작은 문자 조각으로 허용하지 않는 구성이라면 단어 전체가 `[UNK]` 하나로 붕괴할 수 있다. 원문의 어느 부분이 문제인지 token 열만 보고 알기 어려워진다. byte fallback과 대조되는 중요한 failure mode다.

WordPiece의 `##`는 사용자가 입력한 실제 문자라기보다 단어 내부 continuation을 나타내는 vocabulary convention일 수 있다. decoder는 이를 제거하고 조각을 붙이는 cleanup을 할 수 있다. token string을 단순히 빈 문자열로 join하는 것은 정식 decode와 같지 않다.

### Unigram: 전체 경로의 score를 본다

Unigram 예제를 위해 token cost가 다음과 같다고 하자. 값이 작을수록 좋은 경로라고 가정한다.

```text
"서울"  1.2
"서"    0.8
"울"    0.9
"역"    0.5
"서울역" 2.0
```

입력 `서울역`에는 최소 세 경로가 있다.

```text
[서울역]       cost 2.0
[서울, 역]     cost 1.2 + 0.5 = 1.7
[서, 울, 역]   cost 0.8 + 0.9 + 0.5 = 2.2
```

이 숫자에서는 `[서울, 역]`이 선택된다. 가장 긴 token `[서울역]`이 vocabulary에 있어도 전체 score가 더 나쁘면 고르지 않는다. 실제 SentencePiece Unigram은 log probability와 dynamic programming을 쓰며 unknown 처리·normalization·sampling 설정이 더 있다. 이 손계산은 전체 경로 최적화라는 차이만 보여준다.

Unigram은 subword regularization처럼 여러 segmentation을 표본화할 수 있는 여지도 있다. serving에서는 deterministic inference 설정과 학습 augmentation 설정을 혼동하지 않아야 한다. 같은 text의 cache identity를 안정적으로 만들려면 production encode가 어떤 deterministic 조건을 쓰는지 고정한다.

### SentencePiece는 네 번째 segmentation 알고리즘이라는 말이 부정확하다

SentencePiece model은 BPE 또는 Unigram model type을 사용할 수 있다. raw sentence에서 공백을 metaspace 표식으로 바꾸고 normalization rule을 포함하는 방식으로 많이 쓰인다. 따라서 “BPE 대 WordPiece 대 Unigram 대 SentencePiece”라는 네 칸 표는 축이 섞여 있다. 앞 셋은 주로 segmentation/model 선택을 가리키고, SentencePiece는 trainer/runtime/model serialization과 normalization·pretokenization 정책을 함께 제공하는 체계다.

Transformers의 SentencePiece backend는 model proto를 읽고 normalizer의 dummy prefix 같은 설정을 조정하는 경로를 갖는다. [Transformers v5.15.1 `tokenization_utils_sentencepiece.py:45-96`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_sentencepiece.py#L45-L96)를 보면 SentencePiece processor를 생성하기 전에 proto 설정을 다루는 부분을 확인할 수 있다. 이 좌표는 “`.model` 파일만 같으면 wrapper 동작도 항상 같다”는 가정을 경계하게 한다.

### BPE에서 우선순위를 한 단계씩 추적하는 법

조금 더 현실적인 BPE 손계산을 해 보자. 입력 symbol이 `a b a b a`이고 merge rank가 `(a,b)=1`, `(b,a)=2`, `(ab,a)=3`, `(ab,ab)=4`라고 하자. 처음에는 `(a,b)`가 두 곳에 나타난다. 구현은 겹치지 않는 같은 pair를 처리해 `ab ab a`에 해당하는 상태를 만들 수 있다. 이제 원래 있던 `(b,a)`는 그대로 존재하지 않는다. 대신 `(ab,ab)`와 `(ab,a)`가 후보가 되고 rank 3인 `(ab,a)`가 rank 4보다 먼저 선택될 수 있다. 결과는 `ab aba`가 된다. vocabulary에 `abab`가 있어도 그 token으로 끝난다고 보장할 수 없다. 현재 후보와 rank의 상호작용이 경로를 결정한다.

이 예제에서 확인할 구현 세부는 세 가지다. 첫째, merge table의 숫자가 큰 것이 우선인지 작은 것이 우선인지다. 대개 작은 rank가 먼저지만 artifact 표현을 읽어야 한다. 둘째, 같은 pair가 여러 번 나타날 때 겹침을 어떻게 해결하는지다. 셋째, 새 symbol이 만들어졌을 때 이웃 후보를 어떻게 갱신하는지다. heap을 쓰는 구현과 linked structure를 쓰는 구현이 같은 결과를 내더라도 tie 처리나 dropout 같은 옵션이 있으면 차이가 날 수 있다.

byte-level BPE의 초기 alphabet 변환도 추적한다. 모든 byte를 그대로 printable Unicode로 취급하기 어려워 reversible한 byte-to-Unicode mapping을 쓰는 계열이 있다. 여기서 token string에 보이는 낯선 glyph는 원문 문자 자체가 아니라 특정 byte를 표시하는 내부 symbol일 수 있다. 디버거가 token string을 원문 substring처럼 표시하면 운영자가 잘못된 결론을 내린다. 원문 offset, internal token surface, decoded surface를 세 열로 나눠 보는 이유다.

BPE 장애를 파고들 때 가장 빠른 질문은 “최종 token이 왜 없나?”가 아니라 “최종 결과를 막은 최초 merge rank는 무엇인가?”다. 두 artifact의 vocabulary와 merge를 diff하고, 입력 하나에 대해 각 단계의 active pair와 선택 rank를 비교한다. 첫 rank divergence가 나오면 그 뒤 token 열 전체가 달라도 원인은 하나일 수 있다.

### WordPiece의 탐욕 선택을 실패 지점까지 계산하기

입력 `unplayers`와 vocabulary `un`, `##play`, `##player`, `##s`, `u`, `##n`, `[UNK]`를 가정하자. 첫 위치에서 `unplayers` 전체부터 길이를 줄여 가며 continuation 표식 없는 token을 찾는다. `un`이 맞으면 cursor는 두 글자 뒤로 간다. 이제 단어 내부이므로 후보 표면 `players`에 continuation prefix를 붙인 `##players`부터 찾고, 줄여서 `##player`를 고른다. 마지막 `s`는 `##s`다. 결과는 `un`, `##player`, `##s`다.

여기에 `##s`가 없다면 구현이 이미 고른 앞 token을 유지하고 마지막 글자만 `[UNK]`로 바꾸는지, 원래 단어 전체를 `[UNK]`로 만드는지 확인해야 한다. 전형적인 WordPiece 설명은 한 단어에서 어느 위치도 match하지 못하면 단어 전체를 unknown으로 처리한다. 그러면 앞의 성공한 탐욕 선택도 결과에서 사라질 수 있다. unknown rate 하나만 보지 말고 unknown이 덮은 원문 span 길이를 봐야 하는 이유다.

`max_input_chars_per_word` 같은 안전 한계가 있다면 vocabulary에 모든 조각이 있어도 지나치게 긴 공백 없는 문자열을 곧바로 unknown으로 보낼 수 있다. URL, hash, base64, minified code가 자연어보다 이 한계에 쉽게 걸린다. 모델 배포 뒤 `[UNK]`가 늘었다면 vocabulary 손상 외에도 pre-tokenizer가 긴 덩어리를 만들었는지, word-length guard가 달라졌는지 본다. 이 경우 GPU의 embedding lookup은 정상이다. 잘못된 ID는 훨씬 앞에서 확정되었다.

### Unigram을 동적 계획표로 읽기

앞의 `서울역` 비용 예제를 실제 dynamic programming 모양으로 펼쳐 보자. 문자열 경계를 0, 1, 2, 3이라 하고 `best[i]`를 위치 `i`까지 도달하는 최소 비용이라고 하자. `best[0]=0`에서 시작한다. 위치 0에서 `서` 후보는 위치 1에 0.8을, `서울`은 위치 2에 1.2를, `서울역`은 위치 3에 2.0을 제안한다. 위치 1에서는 `울`을 붙여 위치 2에 1.7을 제안하지만 기존 1.2보다 나쁘므로 버린다. 위치 2에서는 `역`을 붙여 위치 3에 1.7을 제안하고 기존 2.0을 바꾼다. backpointer를 따라가면 `서울`, `역`이다.

실제 score가 log probability라면 부호와 최적화 방향이 예제와 다를 수 있다. 중요한 것은 숫자 표기보다 후보 graph와 backpointer다. 후보 하나의 score를 바꾸면 그 token이 포함된 지역만이 아니라 뒤 전체 최적 경로가 달라질 수 있다. 가장 긴 token 탐욕 선택으로 Unigram 결과를 예측할 수 없는 이유다.

unknown 후보 비용도 중요하다. vocabulary 밖의 구간을 어떤 길이로 묶고 어떤 penalty를 주는지에 따라 rare script가 하나의 unknown으로 붕괴하거나 byte fallback 조각으로 풀린다. artifact conversion에서 fallback token type이나 score가 유실되면 평범한 입력은 같고 희귀 입력만 달라질 수 있다. parity corpus에 드문 script와 emoji를 넣는 이유가 여기에 있다.

Unigram sampling이나 `nbest`는 학습 augmentation에는 유용하지만 cache가 있는 serving 경로에서 무심코 켜면 같은 문자열이 서로 다른 ID 열이 될 수 있다. 난수 seed를 고정하는 것만으로 충분하지 않을 수 있다. request ordering과 worker 수가 난수 소비 순서를 바꿀 수 있기 때문이다. production identity를 요구하면 deterministic path를 명시하고 sampling은 별도의 학습·분석 endpoint로 격리한다.

### SentencePiece 공백 표식을 원문 공백으로 오해하지 않기

SentencePiece 계열에서 자주 보이는 `▁`는 일반적으로 word boundary 또는 공백 정보를 표현하는 metaspace 표식이다. token surface가 `▁서울`이라고 해서 원문에 실제 `U+2581` 문자가 있었다는 뜻은 아니다. decoder는 이 표식을 공백으로 복원한다. 반대로 원문에 진짜 `▁`가 들어오면 escaping 정책에 따라 구별이 필요하다. 내부 표식과 사용자 문자를 같은 glyph로 표시하는 UI는 혼동을 만든다.

dummy prefix 설정은 문장 첫 단어에도 앞 공백이 있는 듯한 표현을 붙여 문두와 문중 단어의 형태를 맞추는 데 쓰일 수 있다. 이 설정이 바뀌면 첫 token만 달라지고 나머지는 같아 보이는 현상이 생긴다. cache는 맨 앞 ID가 다르면 긴 공통 문장이 뒤에 있어도 prefix를 공유하지 못한다. “첫 token 하나쯤”이 serving cost에는 작은 차이가 아닐 수 있다.

whitespace suffix/prefix 처리와 trailing whitespace는 chat template와 만나 더 민감해진다. template가 role marker 뒤 newline을 하나 추가했는데 normalizer가 이를 접는지, metaspace가 별도 token으로 보존하는지에 따라 system prompt 전체의 prefix identity가 갈린다. SentencePiece 문제를 볼 때 model proto만 보지 않고 wrapper의 encode flags와 template가 만든 정확한 문자열까지 함께 본다.

이 절의 네 알고리즘을 닫는 관찰법은 이렇다. BPE에서는 선택된 merge rank 열, WordPiece에서는 각 cursor의 longest-match와 최초 실패 위치, Unigram에서는 후보 lattice와 최종 backpointer, SentencePiece에서는 normalization·dummy prefix·metaspace 설정을 남긴다. 최종 IDs만 비교하는 것보다 이 중간 증거가 있어야 다음에 artifact의 어느 부분을 파야 할지 알 수 있다.

## 6.4 fast·slow backend의 관찰 계약을 맞춘다

Transformers에서 fast tokenizer는 대개 Rust `tokenizers` backend를 감싼다. slow tokenizer는 Python과 model-specific library로 구현될 수 있다. fast가 빠르다는 사실은 중요하지만 serving 디버깅에서는 더 중요한 차이가 있다. fast encoding은 token별 original span, sequence ID, word ID 같은 alignment 정보를 제공할 수 있다. slow 구현과 동일한 token IDs를 내더라도 관찰 가능한 상태와 edge-case behavior가 완전히 같다고 가정하면 안 된다.

Transformers `TokenizerBackend.tokenize`는 backend의 `encode_batch`를 호출하고 첫 encoding의 tokens를 반환한다. 고정 소스는 [Transformers v5.15.1 `tokenization_utils_tokenizers.py:847-866`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L847-L866)이다.

`_encode_plus`는 더 넓은 encoding contract를 다루고 [같은 파일 `:925-1008`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L925-L1008)에 있다. 호출자가 `tokenize`의 문자열 조각만 필요한지, offset과 attention mask를 포함한 full encoding이 필요한지 구별해야 한다.

### AddedToken은 vocabulary 뒤에 한 행을 붙이는 것보다 강하다

added token은 base subword model로 분해되기 전에 특정 문자열을 하나의 token으로 가로챌 수 있다. `lstrip`, `rstrip`, `single_word`, `normalized`, `special` 같은 속성이 매칭 경계를 바꾼다. 예컨대 `<image>`를 added token으로 등록하면 `<`, `image`, `>`로 나뉘던 입력이 하나의 ID가 될 수 있다. 주변 공백을 token이 흡수하는지에 따라 다음 token의 표면도 달라진다.

Transformers fast backend는 special token map을 순회하며 `AddedToken` 속성을 보존하고, plain string special token은 `special=True`인 AddedToken으로 바꾼다. [Transformers v5.15.1 `tokenization_utils_tokenizers.py:431-466`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L431-L466)에서 이 처리를 확인할 수 있다.

실제 added token 등록 경계는 [같은 파일 `:794-846`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L794-L846)다.

이 동작은 cache와 embedding에 두 가지 영향을 준다. 첫째, token ID sequence가 달라져 prefix cache key가 달라진다. 둘째, added token ID가 model embedding row 범위를 넘는데 embedding matrix를 resize하지 않았다면 lookup이 실패하거나 config가 어긋난다. special ID와 embedding row 계약은 8장과 9장에서 더 깊게 다룬다. 여기서는 경계 가로채기가 base algorithm 앞에서 일어난다는 사실을 기억하면 된다.

### fast와 slow의 parity를 어떻게 판정하는가

정상적인 비교는 평균 token 수 하나가 아니다. 다음 corpus를 포함한다.

- NFC/NFD가 다른 문자열
- leading/trailing/multiple whitespace
- punctuation과 code snippet
- emoji와 결합 문자
- added/special token 주변 공백
- vocabulary에 없는 rare script와 invalid byte를 표현한 fixture

각 입력에서 IDs, token strings, decoded text, offset mapping을 비교한다. offset을 제공하지 않는 slow backend에는 “불일치”가 아니라 capability 차이라고 기록한다. IDs가 다르면 normalizer→pre-tokenizer→model→added token 순서로 first divergence를 찾는다. cleanup 옵션이 다르면 decode 차이는 별도 축으로 본다.

vLLM renderer는 tokenizer가 offset을 만들 수 있는지 확인하고, 필요한 경우 `_build_tokens_prompt`에 offset mapping을 넣는다. [vLLM v0.27.1 `renderers/base.py:431-487`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/renderers/base.py#L431-L487)는 fast tokenizer capability가 serving request representation으로 올라오는 실제 소비 경계다. 이 때문에 tokenizer mode 변경은 CPU 속도 옵션만이 아니라 일부 API 기능의 자료형을 바꾸는 선택일 수 있다.

### offset 불일치를 실제 사고처럼 재현한다

문서 highlight API가 `"A가éZ"`에서 model이 선택한 token을 굵게 표시한다고 하자. 마지막 `é`는 `e`와 결합 악센트다. 원문 code point index는 `A=[0,1)`, `가=[1,2)`, `é=[2,4)`, `Z=[4,5)`다. UTF-8 byte index는 각각 `[0,1)`, `[1,4)`, `[4,7)`, `[7,8)`이다. NFC normalized string에서는 `é`가 한 code point가 되어 normalized index `[2,3)`이 된다.

backend가 original character offset `[2,4)`를 반환했는데 소비자가 UTF-8 byte buffer를 slice하면 byte 2에서 4까지를 자른다. 이는 `가`의 중간 byte를 포함해 invalid sequence가 된다. backend가 byte offset `[4,7)`을 반환했는데 Python string에 쓰면 길이 5인 string에서 `[4,7)`가 `Z`만 가리킬 수 있다. backend가 normalized offset `[2,3)`을 주는데 원문 string에 적용하면 결합 악센트를 빼고 `e`만 선택한다. 세 버그는 tuple 모양이 모두 `(int,int)`라 type checker가 잡지 못한다.

재현 fixture는 offset tuple만 assert하지 않는다. 각 token에 대해 계약상 기준 buffer를 slice하고 그 substring을 저장한다. original-character 계약이라면 `[2,4)` slice가 `é`인지 본다. normalized-character 계약이라면 normalized buffer와 함께 저장한다. byte 계약이라면 byte slice가 valid UTF-8일 것이라는 불필요한 가정을 두지 않고 hex도 기록한다. token surface와 substring이 항상 같다고 assert해서도 안 된다. metaspace, continuation prefix, normalization 때문에 다를 수 있다.

pair input과 special token은 offset을 한층 복잡하게 한다. `[CLS] sentence A [SEP] sentence B [SEP]` 같은 post-processing에서 special token은 원문 span이 없으므로 `(0,0)` 같은 sentinel을 쓸 수 있다. 두 sentence는 각각 독립 index space를 가질 수 있고 sequence ID가 어느 입력인지 알려 준다. offset만 정렬해 전체 문자열 하나를 만들면 서로 다른 sentence의 `[0,n)`이 충돌한다. fast encoding의 sequence/word metadata가 단순한 편의 기능이 아닌 이유다.

added token `<image>`가 주변 공백을 흡수한다면 offset은 문자열 `<image>`만 가리킬지 흡수한 공백까지 가리킬지 contract에 달렸다. UI는 visible marker만 highlight하고 싶고 decoder는 공백을 복원해야 할 수 있다. `lstrip`과 `rstrip`을 바꾼 뒤 token IDs만 parity 검사하면 offset regression을 놓친다. added token의 ID, original span, decoded 주변 공백을 한 fixture에서 묶어 검증한다.

장애 조사에서는 first bad consumer도 찾는다. tokenizer가 올바른 original offset을 냈는데 scoring utility가 byte offset으로 해석했다면 tokenizer를 고치는 것은 더 큰 회귀를 만든다. 반대로 backend 교체 뒤 offset 기준이 달라졌는데 consumer contract가 문서화되어 있다면 adapter에서 변환해야 한다. producer와 consumer 각각의 단위, 기준 문자열, end-exclusive 여부를 적으면 책임 경계가 선명해진다.

### AddedToken 플래그를 문장으로 시험한다

`single_word=True`인 token `cat`을 등록했다고 하자. 독립 단어 `a cat sleeps`에서는 match하지만 `concatenate` 내부의 `cat`까지 가로채서는 안 된다는 의도다. 여기서 “word” 경계가 Unicode-aware인지 ASCII 기준인지 backend 구현을 확인해야 한다. 한국어 조사처럼 공백 없이 붙는 언어에서는 영어식 single-word 직관이 그대로 맞지 않는다.

`lstrip=True`인 `<assistant>`는 앞 공백을 함께 소비할 수 있고 `rstrip=True`는 뒤 공백을 소비할 수 있다. 이 옵션은 decode 표면뿐 아니라 다음 base token의 leading-space 표식을 바꾼다. `"hello <assistant> world"`에서 marker 앞뒤 공백을 흡수하면 `world`는 문두형 token처럼 보일 수 있다. flags 변경 하나가 marker token 하나만 바꾼다고 생각하면 안 된다.

`normalized=True`이면 added token match가 normalized text를 대상으로 할 수 있고, false이면 원래 표면을 기준으로 할 수 있다. case-folding normalizer가 있을 때 `SPECIAL` 등록이 `special`도 잡는지 여부가 갈린다. protocol marker는 exact surface를 요구할 수 있으므로 이 선택은 보안과도 닿는다. 사용자가 일반 text로 marker를 흉내 냈을 때 special parsing이 허용되는지, API가 message structure에서만 marker를 주입하는지까지 본다.

이 절을 마친 독자는 fast/slow를 한 숫자로 비교하지 않는다. IDs parity, original alignment capability, added-token matching, special/post-processing, decode policy를 각각 판정한다. 문제가 offset이면 backend encoding과 최초 consumer로 가고, marker 주변에서만 ID가 다르면 AddedToken flags와 template 문자열로 간다. 전체 tokenizer를 교체하기 전에 divergence 범위를 좁힐 수 있다.

## 6.5 ID 열을 byte와 문자열로 안전하게 decode한다

많은 설명이 `encode(text) -> ids`, `decode(ids) -> text`를 서로 역함수처럼 그린다. 실제로는 일반적으로 다음 식이 성립하지 않는다.

```text
decode(encode(text)) == text
```

normalization이 정보를 버리면 원문을 복원할 수 없다. lowercasing 뒤에는 원래 대문자 위치를 모른다. accent stripping 뒤에는 악센트를 복구할 수 없다. whitespace cleanup은 여러 공백을 하나로 바꿀 수 있다. unknown token은 여러 원문 조각을 하나의 ID로 합친다. special token skip은 ID 열에 있던 protocol marker를 visible text에서 제거한다.

더 정확한 기대는 tokenizer artifact가 정의한 canonicalization 안에서 의미 있는 round trip을 얻는 것이다. 어떤 모델은 `decode(encode(text))`가 normalized/canonical text와 같도록 설계된다. byte-level tokenizer는 임의 byte를 보존할 수 있어 더 강한 round trip을 제공할 수 있지만, special token parsing과 cleanup 옵션이 끼면 여전히 원문 동일성을 보장하지 않는다.

### 부분 decode가 깨지는 이유

UTF-8에서 한 Unicode code point는 여러 byte일 수 있다. byte-level token 하나가 그중 일부 byte만 표현한다면 token 하나를 즉시 decode했을 때 replacement character가 나오거나 빈 text가 될 수 있다. 여러 token이 모여 complete byte sequence가 된 뒤에야 글자 하나가 확정된다.

따라서 streaming detokenizer는 매 token을 독립 문자열로 바꿔 이어 붙이는 방식과 같지 않다. 누적 token/byte state를 가지고 새로 확정된 suffix만 내보내야 한다. stop string 처리까지 더해지면 stop prefix일 가능성이 있는 text를 잠시 보류할 수도 있다. 이 자세한 streaming commit은 11장에서 다룬다. 여기서 얻을 결론은 token boundary가 visible character boundary가 아니라는 것이다.

Transformers fast backend의 `_decode`는 backend `decode`를 호출하고 cleanup 정책을 적용한다. [Transformers v5.15.1 `tokenization_utils_tokenizers.py:1086-1111`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L1086-L1111)을 보면 `skip_special_tokens`와 cleanup이 별도 인자임을 알 수 있다.

SentencePiece backend의 decode는 added token과 SentencePiece 조각을 구분해 연속 subtext를 processor로 decode한다. [Transformers v5.15.1 `tokenization_utils_sentencepiece.py:266-304`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_sentencepiece.py#L266-L304)가 그 경계다.

llama.cpp public C API도 tokenize와 detokenize를 별도 계약으로 노출한다. `llama_tokenize` 선언과 필요한 buffer 크기를 음수로 반환하는 규약은 [llama.cpp v0.2.0 `include/llama.h:1153-1178`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/include/llama.h#L1153-L1178), detokenize 선언은 [같은 파일 `:1180-1202`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/include/llama.h#L1180-L1202)에 있다.

`common_tokenize` wrapper는 작은 예상 buffer로 먼저 호출한 뒤 부족하면 필요한 크기로 다시 할당한다. [llama.cpp v0.2.0 `common/common.cpp:1836-1866`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/common.cpp#L1836-L1866)가 이 two-call pattern을 보여 준다. 이 패턴은 token 수가 byte 수와 단순 비례하지 않으므로 API가 필요한 output capacity를 명시적으로 알려 주는 사례다.

### round trip을 네 등급으로 나눈다

`decode(encode(x))` 하나를 성공/실패 boolean으로 기록하면 어느 정보가 사라졌는지 알 수 없다. 첫째는 byte identity다. 원래 UTF-8 byte가 완전히 같은가. 둘째는 Unicode identity다. code point 열이 같은가. 셋째는 visible identity다. grapheme과 공백이 사용자에게 사실상 같게 보이는가. 넷째는 model identity다. 다시 encode했을 때 같은 token IDs를 얻는가. normalizer가 있는 tokenizer는 byte와 Unicode identity를 잃어도 model identity는 안정적일 수 있다.

예를 들어 NFD 입력 `Café`를 NFC로 정규화한 뒤 decode하면 합성형 `Café`가 나올 수 있다. byte identity와 code point identity는 실패하지만 visible identity와 재-encode ID identity는 성공할 수 있다. lowercase tokenizer가 `NASA`를 `nasa`로 내보내면 visible 의미까지 달라졌다고 느낄 수 있으나 model identity는 같을 수 있다. whitespace cleanup이 코드 indentation을 바꾸면 자연어에서는 사소해 보여도 프로그램 text에서는 의미를 훼손한다. “round trip 성공”이라는 문장은 요구한 등급을 붙여야 완전하다.

token 단위 debug 출력도 두 종류를 나눈다. `convert_ids_to_tokens`에 해당하는 표면은 vocabulary의 내부 symbol을 보여 주고, 정식 `decode`는 decoder와 cleanup을 거친 사용자 text를 보여 준다. BPE continuation marker, WordPiece `##`, SentencePiece metaspace, byte alphabet이 있으면 둘은 다르다. 내부 token surface를 이어 붙여 원문이라고 주장하거나, 정식 decode 결과만 보고 어느 ID가 어느 byte를 담당했는지 추측하지 않는다.

### streaming에서는 아직 확정되지 않은 byte를 보류한다

UTF-8로 `가`를 만드는 세 byte가 서로 다른 token 조각에 걸쳤다고 하자. 첫 token 뒤에는 완전한 code point가 없고, 둘째 뒤에도 없으며, 셋째가 와야 decoder가 한 글자를 확정한다. 매번 전체 누적 ID를 decode하고 이전 text와의 suffix 차이를 내는 구현은 개념적으로 안전할 수 있지만, 매 step 비용과 cleanup의 비국소성 문제가 있다. incremental decoder는 incomplete byte suffix와 decoder state를 보존한다.

이때 replacement character를 즉시 내보내면 나중 byte가 와도 이미 client에 보낸 문자를 되돌릴 수 없다. streaming output은 append-only인 경우가 많기 때문이다. 따라서 “아직 invalid”와 “최종적으로 invalid”를 구별해 incomplete sequence를 보류해야 한다. flush 또는 sequence 종료 시점에는 남은 byte를 어떤 오류 정책으로 처리할지도 계약이다.

stop string `</tool>`을 찾는 경우도 비슷하다. 현재 text가 `</to`라면 일반 출력인지 stop prefix인지 아직 모른다. client에 먼저 보냈다가 뒤에 `ol>`이 오면 protocol marker가 노출된다. detokenizer/stop matcher는 가능한 stop prefix 길이만큼 text를 보류할 수 있다. token ID stop은 더 빠르게 판정할 수 있지만 stop 문자열이 여러 token segmentation을 가질 수 있고 added token 여부도 영향을 준다.

cleanup이 punctuation 앞 공백을 지우는 식으로 이전 text를 수정할 수 있다면 단순 suffix streaming은 더 어렵다. 전체 decode 결과가 앞 step 결과의 strict prefix라는 보장이 없을 수 있기 때문이다. 구현이 streaming-safe decoder를 따로 두는지, cleanup을 마지막에만 하는지, client protocol이 replacement event를 허용하는지 본다. tokenizer의 offline decode parity만 통과해도 streaming text가 맞다고 결론 내릴 수 없다.

decode 장애를 조사할 때는 누적 IDs, 내부 token surfaces, 누적 decoded bytes, 새로 commit한 text, 보류 buffer를 step별로 남긴 fixture가 강하다. 처음으로 offline full decode와 streaming concatenation이 달라지는 step을 찾는다. 그 token이 incomplete UTF-8인지, special token인지, cleanup 경계인지 분류하면 다음 소스 위치가 결정된다.

## 6.6 같은 ID 흐름이 CPU 병목과 cache identity가 된다

GPU가 빠른데 TTFT가 길다고 해서 scheduler와 CUDA부터 볼 필요는 없다. 긴 tool schema, 많은 동시 HTTP request, slow Python tokenizer, remote artifact access가 tokenization CPU를 포화시킬 수 있다. 이때 GPU는 입력을 기다리고, engine queue는 낮게 보이며, API process의 render/tokenize 시간이 늘어난다.

토크나이저 비용은 대략 다음 성분으로 나뉜다.

```text
Unicode/normalization scan
+ pre-tokenization과 regex/byte mapping
+ subword model lookup/merge or path search
+ special/post-processing
+ Python↔native object/serialization
+ padding/truncation과 output object 생성
```

문자열 길이만으로 비용을 예측하기 어렵다. BPE는 조각과 merge 자료구조에, Unigram은 lattice 후보 수에, regex pre-tokenizer는 입력 pattern에 영향을 받는다. 반환하는 offset/attention mask 같은 부가 자료도 비용과 allocation을 늘린다. batch API는 native 호출 overhead를 줄일 수 있지만 작은 요청이 batch를 기다리는 queue latency를 더할 수 있다.

### 틀리기 쉬운 첫 가설

“CPU utilization이 높으니 fast tokenizer로 바꾸면 해결된다”는 그럴듯하지만 불충분한 가설이다. 먼저 API render와 tokenize를 분리해야 한다. tool JSON 직렬화가 병목일 수 있다. fast backend를 이미 쓰는데 Python에서 request별 object를 과도하게 만들 수 있다. 많은 slow client 때문에 output event loop가 막힌 것을 tokenization으로 오인할 수도 있다.

반증에는 다음 timeline이면 충분하다.

```text
request parsed
render completed
tokenizer call entered
tokenizer returned IDs
engine submission
```

같은 process monotonic clock에서 구간을 측정하고 prompt byte/token 분포와 함께 본다. tokenizer 구간만 늘어났다면 fast/slow mode, thread contention, batch size, normalizer/model artifact를 본다. render 구간이면 chat template와 tool schema owner로 이동한다. engine submission 뒤가 느리면 이 장을 떠난다.

### cache key의 진짜 입력

prefix cache가 재사용하는 것은 사람이 “같은 대화”라고 부르는 JSON이 아니다. 결국 model이 받은 token ID prefix와 model/cache identity다. raw JSON이 같아도 template 또는 tokenizer가 달라 IDs가 바뀔 수 있다. 반대로 서로 다른 Unicode 원문이 normalization 뒤 같은 IDs가 될 수도 있다.

cache identity를 다음 tuple로 생각하면 안전하다.

```text
(model/adapter/cache domain,
 tokenizer bundle revision,
 rendered token ID prefix,
 relevant multimodal/embedding identity)
```

실제 엔진이 이 tuple을 문자 그대로 key로 쓰는 것은 아니다. 여기서는 어떤 의미가 달라지면 reuse를 재검토해야 하는지를 보여 준다. tokenizer 업데이트를 weight와 독립적으로 배포하면서 기존 KV cache를 유지하면, 같은 raw prompt가 다른 IDs를 가리키거나 같은 IDs의 의미가 달라질 위험이 있다. cache를 version domain으로 분리하거나 drain/reset하는 정책이 필요하다.

vLLM의 renderer는 single prompt를 token화하고 token prompt를 만든 뒤 engine input으로 넘긴다. 동기/비동기 public 경계는 [vLLM v0.27.1 `renderers/base.py:614-649`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/renderers/base.py#L614-L649)에 있다. 이 함수 이름만으로 tokenizer CPU가 어느 thread/process에 있는지 단정할 수는 없다. renderer concrete class, executor, API deployment mode를 계속 따라가야 한다. 그러나 token sequence가 engine request 전에 확정되는 ownership 경계는 확인할 수 있다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- SGLang은 OpenAI tokenize endpoint에서 text 또는 messages를 token IDs로 만든다.
- chat request는 template manager를 사용할 수 있고, plain text는 tokenizer encode로 간다.
- [SGLang v0.5.18 `serving_tokenize.py:20-116`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_tokenize.py#L20-L116)은 API가 tokenization 결과를 노출하는 경계다.
- detokenize는 별도 serving class로 [같은 파일 `:118-154`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_tokenize.py#L118-L154)에 있다.
- tokenize/detokenize endpoint 결과가 model serving path의 모든 template·special-token 정책과 자동으로 같다고 가정하지 말고 실제 parameter를 비교해야 한다.

### CPU 병목을 queueing 문제로 다시 본다

동시 요청이 적을 때 tokenizer 한 번이 2ms이고 많을 때 20ms라면 알고리즘 자체가 갑자기 열 배 복잡해졌다고 결론 내리지 않는다. CPU run queue, native thread pool, Python GIL 구간, allocator contention, NUMA placement, request batching이 wall time을 늘릴 수 있다. 필요한 것은 평균 하나가 아니라 queue wait와 service time의 분리다.

API process가 request를 받은 시각, tokenizer work item을 enqueue한 시각, worker가 실제 시작한 시각, 끝난 시각을 잡을 수 있다면 `enqueue→start`가 대기이고 `start→end`가 service다. 대기만 늘면 worker 수와 upstream concurrency·admission을 본다. service도 늘면 input distribution, backend mode, CPU frequency, memory pressure, contention을 본다. 둘을 합친 `tokenization latency` histogram만 있으면 잘못된 처방을 내리기 쉽다.

batching은 throughput과 latency를 맞바꾼다. 여러 문자열을 native backend 한 호출로 보내면 Python↔native overhead와 vectorized work를 줄일 수 있다. 그러나 첫 요청은 batch가 찰 때까지 기다리고, 한 batch 안의 긴 prompt가 짧은 prompt completion을 늦출 수 있다. batch size를 키운 뒤 p50이 좋아지고 p99가 나빠지는 현상은 이상하지 않다. arrival rate와 max-wait policy를 함께 기록한다.

thread 수를 CPU core 수만큼 늘리는 것도 자동 해법이 아니다. API server worker 여러 개가 각자 native tokenizer thread pool을 만들면 총 runnable thread가 core를 크게 넘을 수 있다. context switching과 shared cache pressure가 늘고, NUMA를 가로지르는 memory access가 생긴다. tokenizer artifact와 merge table이 read-only여도 각 process가 별도 copy를 갖는지, page sharing이 되는지, worker locality가 어떤지 본다. GPU utilization 저하는 이 CPU oversubscription의 결과일 수 있다.

metric label에는 원문이나 token ID 전체를 넣지 않는다. cardinality와 개인정보 문제가 생긴다. 대신 model/tokenizer revision, endpoint, input kind, backend mode처럼 bounded label을 쓰고 prompt byte와 token count는 histogram으로 둔다. rare script 여부나 normalization 변화가 필요하면 offline sampled trace에서 승인된 digest와 fixture ID로 조사한다. observability를 위해 시스템을 더 느리고 위험하게 만들지 않는 설계다.

### cache hit 급락 사고를 처음부터 끝까지 걷는다

금요일 배포 뒤 prefix cache hit ratio가 78%에서 31%로 떨어졌다고 하자. GPU kernel, scheduler, block allocator에는 변경이 없고 weight revision도 같다. 동시에 한국어 요청의 평균 prompt token 수가 3% 늘었다. 이 두 관찰은 tokenizer 또는 template 변화와 일치하지만 아직 증명은 아니다.

첫 단계는 cache metric의 분모가 바뀌었는지 본다. traffic mix가 달라져 unique prompt가 늘었을 수 있다. endpoint·model·tenant처럼 bounded dimension에서 전후 cohort를 맞춘다. 같은 canonical fixture 요청을 구·신 배포에 보내되 model execution 없이 render/tokenize 결과만 비교할 수 있다면 raw rendered byte digest, token count, ID digest를 얻는다. 이 정적/endpoint 비교는 GPU 실행을 요구하지 않는다.

rendered byte부터 다르면 chat template, tool schema serialization, newline owner로 올라간다. byte는 같고 IDs가 다르면 tokenizer bundle을 component별로 비교한다. normalizer config, pre-tokenizer, vocabulary, merge/model score, added token map, decoder 중 encode에 영향을 주는 부분을 본다. 첫 divergent fixture가 NFD 한국어에서만 나타난다면 normalizer가 강한 후보지만, artifact diff로 확인하기 전에는 단정하지 않는다.

IDs가 같다면 tokenizer 가설은 기각한다. cache key에 adapter, multimodal input, block hash seed, cache salt, model domain 같은 다른 identity가 추가되었을 수 있다. token count metric이 늘어난 것은 traffic mix의 별도 현상일 수 있다. 상관관계를 한 원인으로 묶지 않는 것이 집요한 분석의 핵심이다.

원인이 tokenizer revision으로 확인되었다고 하자. 즉시 이전 파일 하나만 되돌리는 대신 bundle 원자성을 본다. vocabulary만 rollback하고 added token map이나 template를 새 버전으로 두면 ID collision과 marker 분해가 생길 수 있다. tokenizer·template·model config를 검증된 묶음으로 rollback하거나, 새 cache domain으로 분리해 old cache를 drain한다. cache를 강제로 재사용하는 것이 hit ratio 숫자는 회복해도 correctness를 깨뜨릴 수 있다.

사후 검증에서는 동일 fixture의 rendered byte, IDs, decoded canonical text, offsets를 golden artifact로 남긴다. production metric에는 revision별 cache hit와 prompt token 분포를 둔다. 다음 배포 gate는 평균만 보지 않고 언어·공백·added token·tool schema fixture를 포함한다. 이 사고에서 얻은 교훈은 “tokenizer를 업데이트하지 말라”가 아니라 identity를 보이지 않는 부속품으로 배포하지 말라는 것이다.

### 한 요청의 비용을 어디에 귀속할 것인가

긴 JSON tool schema가 있는 chat request는 template rendering에서 큰 문자열을 만들고, tokenizer가 다시 전체를 scan하며, engine이 긴 prefill을 수행한다. 세 단계 모두 prompt 길이에 따라 늘지만 owner와 최적화는 다르다. render가 반복 serialization을 한다면 canonical tool schema를 재사용할 수 있고, tokenization 결과를 안전한 identity key로 cache할 수 있으며, engine은 prefix cache를 활용할 수 있다. 어느 cache도 key가 불완전하면 correctness를 잃는다.

raw text tokenization cache의 key에 text만 넣으면 tokenizer revision과 encode flags가 빠진다. `add_special_tokens`, truncation side, max length, added-token state가 결과를 바꿀 수 있다. chat-level cache에는 template revision, message structure, tool choice 같은 rendering identity가 더 필요하다. prefix KV cache는 최종 IDs와 model execution domain이 핵심이다. 서로 다른 층의 cache를 하나의 “prompt cache”로 부르면 invalidation 책임이 흐려진다.

cache lookup 자체도 CPU 비용이다. 긴 raw string을 매번 hash하면 tokenization 일부를 대체하지만 공짜는 아니다. collision-safe verification, memory capacity, eviction, tenant isolation도 필요하다. 반복률이 낮은 사용자 prompt까지 무조건 cache하면 memory와 lock contention만 늘 수 있다. system prompt나 tool schema처럼 반복성과 안정성이 높은 부분부터 관찰하는 편이 합리적이다.

이 절을 닫는 판단 규칙은 명확하다. render 구간이 느리면 template·serialization owner로, tokenizer queue가 길면 concurrency와 worker topology로, tokenizer service가 길면 입력 분포와 backend로, engine submission 뒤가 길면 scheduler·GPU 장으로 이동한다. cache miss라면 raw text의 유사성이 아니라 최종 ID prefix와 revision domain을 비교한다.

## 6.7 다섯 경계를 네 서빙 스택의 함수로 연결한다

이제 `" Café 한글🙂 "`이라는 짧은 fixture가 있다고 하자. 실행하라는 뜻이 아니라, 소스를 읽을 때 이 입력이 어떤 객체를 지나갈지 머릿속으로 추적해 보는 것이다. 기록할 상태는 original text, normalized text, pre-tokenized spans, tokens, IDs, offsets, decoded text 일곱 가지다.

### Transformers: API 계약에서 native backend까지

Transformers base `encode`는 text를 받아 token ID list를 반환하는 상위 API다. 실제 구현은 `_encode_plus`로 내려가며 special-token/padding/truncation 옵션을 함께 전달한다.

고정 좌표는 [Transformers v5.15.1 `tokenization_utils_base.py:2241-2303`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L2241-L2303)이다. base `decode` 계약은 [같은 파일 `:2853-2915`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L2853-L2915)에 있다.

fast concrete 경로에서는 `_encode_plus`가 batch encode machinery를 통해 native backend encoding을 얻고, tokens/IDs/offsets 같은 필드를 Python `BatchEncoding`으로 바꾼다. 이 경계를 볼 때 token ID만 보지 말고 `overflowing_tokens`, `special_tokens_mask`, `offset_mapping`, length 같은 선택적 필드가 언제 생기는지 확인한다. 같은 IDs라도 downstream이 요구하는 alignment contract가 다를 수 있다.

SentencePiece slow/backend 경로에서는 wrapper가 AddedToken과 processor를 조합한다. `_add_tokens`가 token의 normalization/special 속성을 보존하려고 별도 처리를 하는 구간은 [Transformers v5.15.1 `tokenization_utils_sentencepiece.py:110-166`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_sentencepiece.py#L110-L166)이다. 이는 fast/slow parity 문제에서 added token 속성을 반드시 비교해야 한다는 구체적 근거다.

### vLLM: renderer가 engine input identity를 만든다

vLLM `BaseRenderer`는 tokenizer를 소유하고 prompt/messages를 rendering한 뒤 token prompt를 만든다. `_tokenize_prompt`와 `_build_tokens_prompt`는 [vLLM v0.27.1 `renderers/base.py:452-488`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/renderers/base.py#L452-L488)에 있다. offset이 필요하고 backend가 지원하면 결과 representation에 넣는다. plain prompt, tokens prompt, embeds prompt는 뒤에서 서로 다른 처리 경로를 가지므로 “vLLM이 항상 문자열을 tokenize한다”는 말도 정확하지 않다. caller가 이미 token IDs 또는 embeddings를 제공할 수 있다.

별도의 serving tokenize endpoint는 요청을 renderer 또는 tokenizer 경계로 보내고 결과를 API response로 만든다. [vLLM v0.27.1 `entrypoints/serve/tokenize/serving.py:32-125`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/serve/tokenize/serving.py#L32-L125)을 읽을 때 text와 messages 입력의 분기, count만 요구하는 경우와 token strings까지 요구하는 경우를 구분한다. endpoint 자체의 latency를 model engine TTFT와 같은 metric으로 부르지 않는다.

### SGLang: API tokenization과 manager ownership

SGLang `OpenAIServingTokenize`는 plain text와 chat messages를 분기한다. chat은 template manager가 선택한 conversation을 거칠 수 있고, 그 뒤 tokenizer가 IDs를 만든다. 이 경로는 다음 장의 chat template로 이어진다. 여기서 중요한 경계는 renderer 결과가 tokenizer 입력이 되기 때문에 template revision과 tokenizer revision을 분리해 기록해야 한다는 점이다.

HTTP route는 tokenize/detokenize request를 serving object로 위임한다. 고정 source 좌표는 [SGLang v0.5.18 `http_server.py:1758-1796`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/http_server.py#L1758-L1796)이다. 이 route가 있다는 사실은 main generation request와 동일한 process/thread scheduling을 증명하지 않는다. deployment mode와 manager wiring을 더 따라가야 한다.

### llama.cpp: byte buffer와 capacity 계약이 표면에 드러난다

llama.cpp C API는 UTF-8 text pointer와 byte length, output token buffer와 capacity, special token 추가/파싱 flag를 받는다. C++ `common_tokenize`는 먼저 추정 capacity로 호출하고 부족하면 반환된 필요 크기로 다시 호출한다. Python high-level API보다 buffer ownership이 선명하다.

detokenize wrapper도 작은 text buffer로 시도한 뒤 필요한 byte 수가 더 크면 resize한다. [llama.cpp v0.2.0 `common/common.cpp:1890-1907`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/common.cpp#L1890-L1907)을 보면 output token 수와 decoded byte 수가 다르다는 사실이 API shape에 드러난다. token 하나가 visible 글자 하나라는 오개념을 버리게 하는 좋은 source다.

### 네 경로를 비교하는 올바른 좌표

프로젝트별 class 이름을 나란히 놓는 것으로 비교를 끝내지 않는다. 다음 의미 좌표를 맞춘다.

```text
tokenizer artifact owner
normalizer/pre-tokenizer/model implementation
added/special token interception
encode input type(text/messages/IDs/embeds)
returned IDs와 offset capability
decode cleanup/special policy
CPU execution/process boundary
engine/cache identity handoff
```

Transformers는 범용 tokenizer API와 fast/slow backend abstraction을 제공한다. vLLM과 SGLang은 이를 serving renderer/manager 수명에 배치한다. llama.cpp는 GGUF vocabulary와 C buffer API에서 같은 의미를 구현한다. 기능 이름이 같다고 default flags, special parsing, cleanup, offset capability가 같지는 않다.

### 실패 사례를 소스 분기까지 닫아 보기

배포 B에서 `" 도구를 호출해 줘"`라는 요청만 첫 token부터 달라졌다고 하자. 증상은 평균 prompt token 수 증가와 prefix cache hit 하락이다. 먼저 A와 B의 message JSON이 같다는 사실로는 충분하지 않다. renderer 뒤 UTF-8 digest를 비교한다. digest가 다르고 B에 role marker 뒤 newline 하나가 더 있다면 tokenizer source를 파기 전에 template 분기로 올라가야 한다. tokenizer는 받은 문자열을 충실히 처리했을 뿐이다.

반대로 rendered byte가 완전히 같은데 IDs가 다르면 tokenizer 상태 장부를 연다. normalized text가 같고 pre-token piece에서 leading space 귀속만 다르다면 pre-tokenizer 또는 added-token whitespace flag가 최초 divergence다. A에서는 `<assistant>`가 앞 공백만 흡수하고 B에서는 뒤 공백까지 흡수했다고 하자. 그 결과 다음 한국어 조각이 leading-space 표식을 잃고 다른 BPE merge 경로를 간다. 최종 ID 열 수십 개를 diff하는 대신 marker 직후 첫 piece가 달라진 위치를 증거로 삼는다.

이제 source 분기를 좁힌다. Transformers backend에 AddedToken을 구성하는 wrapper에서 `lstrip`, `rstrip`, `normalized`, `special` 값이 어떤 serialized field에서 들어오는지 보고, vLLM 또는 SGLang renderer가 어떤 tokenizer object와 flag로 encode를 호출했는지 이어서 본다. serving endpoint의 tokenize 결과와 generation path를 비교할 때도 text/messages 분기와 special-token option을 같게 맞춘다. endpoint 이름이 같다는 이유만으로 동일 경로라고 가정하지 않는다.

수정 후보는 세 가지일 수 있다. 새 AddedToken flag를 되돌리거나, template 공백을 artifact가 기대한 형태로 맞추거나, 의도한 새 semantics라면 tokenizer/template/cache domain을 함께 versioning한다. 어느 선택이 맞는지는 모델 학습 artifact와 API protocol 의도에 달렸다. cache hit를 빨리 회복하려고 old KV를 새 ID 의미에 억지로 연결하는 선택은 제외한다. 같은 text처럼 보여도 ID prefix가 다르면 cached hidden state는 재사용 대상이 아니다.

검증 fixture는 문제 문장 하나로 끝내지 않는다. marker 앞뒤 공백 없음·한 칸·newline, NFC/NFD 한국어, marker와 일반 문자가 붙은 경우를 포함한다. 각 case에 rendered byte, AddedToken match span, pieces, IDs, decode 결과를 저장한다. fast/slow backend를 둘 다 지원한다면 capability 차이는 따로 기록하고, 요구하는 IDs와 marker span parity를 판정한다. 이 fixture가 rollback과 forward fix 모두에서 기대한 결과를 내야 한다.

다른 실패 사례로 scoring highlight가 한글 중간에서 깨진다고 하자. token IDs와 model score는 정상이고 UI span만 깨진다. 이때 vocabulary나 merge를 조사하면 시간을 낭비한다. backend offset `[1,4)`가 UTF-8 byte 기준인데 consumer가 Python character slice에 넣는 것이 최초 divergence라면 producer tokenization은 맞다. vLLM scoring utility처럼 offset을 소비해 원문을 자르는 함수에서 기대 단위를 확인하고, adapter 경계에서 명시적으로 변환하거나 일치하는 original-character offset을 요청한다.

검증은 ASCII만으로 하지 않는다. `A가éZ`의 byte·original code point·normalized code point index를 손으로 고정하고, 실제 slice 결과를 assert한다. special token의 sentinel span, pair input의 sequence ID, AddedToken이 흡수한 공백도 넣는다. tuple 숫자만 비교하지 않고 기준 buffer를 slice한 substring까지 확인해야 같은 종류의 회귀를 막는다.

세 번째 실패는 GPU utilization이 40%로 떨어지고 TTFT가 늘어난 경우다. tokenization을 의심하되 CPU utilization 하나로 확정하지 않는다. request parse, render 완료, tokenize enqueue, tokenize start, tokenize end, engine submit 시각을 같은 monotonic clock에서 잡는다. enqueue→start만 늘면 tokenizer algorithm보다 worker queue와 oversubscription을 판다. start→end가 특정 긴 JSON에서만 늘면 input distribution과 pre-tokenizer, native backend를 판다. render 구간이 늘면 tool serialization owner로 이동한다.

worker 수를 늘린 뒤 queue wait는 줄었지만 service time과 p99가 더 나빠진다면 native thread pool 중첩과 CPU run queue를 확인한다. batch를 키워 throughput이 좋아졌으나 작은 요청 latency가 나빠졌다면 max-wait와 length mixing을 본다. GPU가 기다린다는 현상은 동일해도 source owner는 서로 다르다. timeline의 최초로 늘어난 구간이 다음 조사 위치를 정한다.

이 세 사례의 공통 종료 조건은 명확하다. 사용자 증상을 만든 최종 차이만 말하지 않고, 처음 달라진 관찰 상태를 찾는다. 그 상태를 만든 artifact와 함수 분기를 고정 source에서 가리킨다. 이어서 그 차이를 실제 증상으로 소비한 다음 경계를 찾는다. 마지막으로 원인과 같은 축을 자극하는 회귀 fixture를 만든다. 최초 divergence, owner, consumer, 검증이 한 줄로 연결되지 않으면 분석은 아직 완성되지 않았다.

여기에는 중요한 반증 습관이 하나 더 있다. normalized text와 pre-token pieces, IDs가 모두 같다면 “보이지 않는 Unicode 차이”라는 매력적인 가설을 버린다. decode만 다르면 decoder와 cleanup으로 범위를 옮기고, engine에 전달된 IDs까지 같으면 embedding 이전 tokenizer 계층은 원인이 아니다. 깊게 판다는 말은 한 가설을 오래 붙드는 것이 아니라, 관찰값으로 틀린 층을 빠르게 제외하고 남은 경계의 source를 끝까지 따라간다는 뜻이다.

보고서에도 이 순서를 보존한다. “토크나이저 문제였다”라고 요약하지 않고, 예컨대 “동일 rendered bytes에서 AddedToken `rstrip` 값이 달라 marker 다음 pre-token span이 최초로 갈렸고, 그 결과 ID prefix와 cache key가 달라졌다”라고 쓴다. 이 문장은 재현 입력, 바뀐 상태, 구성 owner, serving 효과를 한 번에 담는다. 다음 운영자는 같은 증상을 보았을 때 그대로 믿는 대신 네 항목을 다시 측정해 같은 원인인지 판정할 수 있다.

마지막으로 fix가 성능 숫자만 회복했는지 보지 않는다. token count와 cache hit 외에 decoded canonical text, offset alignment, special marker isolation도 함께 검증한다. 공백 flag를 되돌려 cache hit가 올라갔지만 tool marker가 사용자 text에서 special ID로 parse된다면 수정은 실패다. serving 최적화는 identity와 protocol correctness를 보존할 때만 최적화다.

따라서 최종 승인은 구 버전과 신 버전의 golden fixture 차이를 사람이 읽을 수 있는 형태로 남긴 뒤 내린다. 의도한 차이는 어느 정책 변경 때문인지 설명하고, 의도하지 않은 차이는 배포를 막는다. fixture에 없는 언어와 형식이 존재한다는 한계도 기록한다. golden corpus는 완전한 증명이 아니라 알려진 경계를 반복 검증하는 안전망이며, production의 bounded metric과 sampled trace가 그 바깥의 변화를 감지한다.

## 6.8 하나의 회귀 양식으로 byte부터 decode까지 닫는다

이 절의 여러 표와 실험은 별도 workbook이 아니다. 모두 아래 한 행 형식을 채우기 위한 확대경이다. fixture마다 원문 byte, normalization, pre-tokenization, ID, decode의 기대값과 실제값을 한 행 묶음으로 기록하고, 최초 불일치 단계 하나에만 소유자를 배정한다. fast/slow 비교와 Unicode offset 비교도 같은 fixture ID를 공유해야 서로 다른 실험 결과가 섞이지 않는다.

```text
fixture_id | input_sha256 | normalizer_sha256 | normalized_sha256
pieces+offset_unit | token_ids | cumulative_decode_bytes
first_divergence | owner | pinned_source | expected | actual | verdict
```

이 양식을 처음부터 끝까지 기계적으로 채우지 않는다. 증상에 따라 세 갈래 중 하나를 먼저 완성한다. ID나 cache key가 달라졌다면 `raw bytes→normalizer→pre-token pieces→model IDs`를, highlight·scoring span만 깨졌다면 `offset unit→alignment→consumer slice`를, TTFT·GPU idle·cancellation이 달라졌다면 `tokenizer queue→effective backend→engine admission`을 먼저 잇는다. 첫 갈래가 의미 동일성을, 둘째가 좌표 동일성을, 셋째가 수명 주기와 비용을 판정하므로 서로의 결과를 대신할 수 없다.

각 갈래에서 최초 불일치를 찾은 뒤에만 나머지 열을 회귀 범위로 확장한다. 예를 들어 IDs까지 같다면 merge 알고리즘을 더 파지 않고 offset·decode consumer로 이동한다. enqueue→start만 늘었다면 Unicode 규칙보다 worker queue를 먼저 본다. 이 stop rule이 아래의 긴 검산 목록을 전수 조사 명령이 아니라 선택 가능한 증거 사다리로 바꾼다.

### 6.8.1 ID·cache key가 달라진 갈래: Unicode 회귀를 두 backend에서 재현한다

#### 단계별 parity contract를 실제 review 질문으로 바꾼다

첫 갈래는 같은 tokenizer program이 같은 의미 ID를 만드는지 묻는다. backend 선택과 직렬화 상태를 먼저 고정해야 뒤의 normalizer·AddedToken·BPE/WordPiece/Unigram 차이가 artifact 차이인지 구현 회귀인지 구별할 수 있다.

Fast와 slow라는 이름은 구현 위치를 말할 뿐 자동으로 우열을 정하지 않는다. Review의 첫 질문은 어느 backend가 빠른가가 아니라 두 backend가 같은 serialized tokenizer program을 소비하는가다. Tokenizer JSON 안에는 normalizer, pre-tokenizer, model, post-processor와 decoder가 있고 Python 쪽에는 AddedToken, special-token map, truncation/padding configuration이 덧붙을 수 있다. Native object를 만들 때 이 상태가 모두 직렬화되는지, 일부 Python override가 fast path에서 무시되는지 확인한다. 같은 vocabulary digest만으로 같은 program을 주장하지 않는다.

Backend selection도 입력 state다. `use_fast=True` 요청이 native artifact 부재, unsupported custom subclass 또는 load failure 때문에 slow object로 내려갈 수 있다. 반대로 slow class를 요청했지만 내부 method가 native engine을 호출할 수도 있다. Manifest에는 requested backend, constructed concrete class, serialized backend generation과 fallback reason을 적는다. 성능 cohort는 실제 constructed path로 나누고 correctness fixture는 두 path가 실제 실행됐다는 증거 뒤 판정한다.

Normalizer parity에는 문자열만 아니라 alignment가 포함된다. 두 backend가 같은 NFC 문자열을 내더라도 one-to-many alignment 처리 방식이 다르면 offsets가 갈린다. Normalizer output row는 raw byte interval, normalized byte interval, raw scalar interval과 mapping cardinality를 가진다. Mapping cardinality가 one-to-one인 ASCII만 통과한 결과를 Unicode parity로 확대하지 않는다. Many-to-one과 deletion, insertion fixture가 각각 필요하다.

Deletion fixture는 control character 또는 accent strip을 사용한다. Raw `a`+combining mark+`b`가 normalized `ab`가 될 때 두 normalized characters 사이 original 경계가 어디인지 정책을 정한다. Token `ab`의 original span은 제거된 mark까지 덮을 수도 있고 visible characters만 덮을 수도 있다. Highlight와 edit consumer는 다른 정책을 요구할 수 있으므로 tokenizer mapping과 consumer cover policy를 나눈다. Backend parity는 먼저 primitive mapping이 같은지를 본다.

Insertion fixture는 whitespace marker나 virtual prefix를 사용한다. Byte-level model이 문장 시작에 virtual space를 넣으면 normalized/model input에는 원문에 없는 symbol이 생긴다. 이 symbol의 offset을 `(0,0)`, 첫 글자 span 또는 undefined로 둘 수 있다. 어느 선택이든 API contract에 맞고 consumer가 처리하면 되지만 fast/slow가 다르게 보고하는데 숫자만 비교하면 false mismatch가 된다. Virtual origin이라는 상태를 별 enum으로 보존한다.

Pre-tokenizer parity는 span 순서와 coverage를 검사한다. 모든 raw/normalized non-virtual 구간이 누락이나 의도하지 않은 중복 없이 span에 덮이는지, separator를 output token에 포함하는지, empty span을 생성하는지 본다. `sum(span lengths)=text length`는 overlap과 gap이 상쇄될 수 있어 충분하지 않다. Interval union, overlap count와 declared discarded interval을 함께 검사한다. Whitespace를 버리는 pre-tokenizer라면 discarded reason이 있어야 한다.

Punctuation fixture는 ASCII apostrophe, curly apostrophe, full-width punctuation과 emoji sequence를 섞는다. 언어별 단어 분절 정답을 가정하지 않고 configured rule이 각 code point class를 어떻게 취급하는지 본다. Backend가 서로 다른 Unicode property database version을 사용하면 같은 설정 이름 아래 class membership이 달라질 수 있다. Native library와 Python runtime의 Unicode data version도 artifact identity 후보가 된다.

AddedToken은 normalizer와 pre-tokenizer보다 먼저 또는 특정 조건에서 우선 매치될 수 있다. `single_word`, `lstrip`, `rstrip`, `normalized` flag의 조합을 각각 시험한다. AddedToken `<tool>`이 `x<tool>y` 안에서 single-word일 때 매치되지 않아야 한다면 fast/slow가 같은 boundary predicate를 써야 한다. 앞 공백을 흡수하면 offset은 token literal보다 넓을 수 있다. Token ID만 같고 span이 다를 수 있으므로 match extent와 consumed whitespace를 기록한다.

AddedToken collision도 본다. Literal이 vocabulary piece와 같거나 두 AddedToken이 prefix 관계일 때 우선순위가 필요하다. `<image>`와 `<image_1>`이 있을 때 짧은 literal을 먼저 잡으면 긴 token이 분해된다. Registry insertion order, longest-match와 special status가 selector를 바꿀 수 있다. Serialized artifact를 재로드했을 때 순서가 보존되는지까지 fixture에 넣는다.

BPE parity는 최종 tokens만 비교하지 않고 merge trace의 first divergence를 찾는다. Initial symbols가 byte fallback을 거쳐 달라졌는지, 같은 pair에 rank가 같은지, stale priority queue entry를 어떻게 무효화하는지 본다. 한 backend가 tie를 leftmost, 다른 backend가 insertion order로 풀면 특수 artifact에서 결과가 갈릴 수 있다. Merge artifact에 duplicate/conflicting rank가 없다는 validator와 runtime tie policy를 구분한다.

WordPiece parity는 unknown policy가 중요하다. 한 위치에서 longest prefix를 찾지 못할 때 whole word를 UNK 하나로 접는지, 이미 찾은 prefix를 보존하고 나머지만 unknown 처리하는지 구현 계약을 확인한다. Maximum input chars per word가 byte, code point 또는 character count인지 Unicode 긴 단어에서 갈릴 수 있다. Limit-1, limit, limit+1과 combining sequence를 fixture로 둔다.

Unigram parity는 floating score와 pruning을 분리한다. 동일 lattice와 scores라도 tie tolerance, unknown edge와 BOS/EOS cost가 다르면 path가 갈릴 수 있다. 최종 score 차이가 0에 가까운 synthetic fixture에서 deterministic tie rule을 확인하고, production artifact에서는 selected path와 runner-up margin을 sample한다. Native와 Python floating dtype이 다르면 정상적인 rounding인지 semantic path drift인지 사전 tolerance를 둔다.

Post-processor는 model algorithm 뒤 special tokens와 pair sequence layout을 만든다. Single/pair 입력에서 type IDs, BOS/SEP 순서와 offsets의 virtual span을 비교한다. Chat serving은 pair API를 잘 쓰지 않아도 regression fixture에는 남길 가치가 있다. 같은 tokenizer artifact가 embedding·reranker 등 다른 endpoint에서 pair mode를 사용할 수 있기 때문이다. Scope 밖이면 production approval이 아니라 library parity gap으로 명시한다.

Decode parity는 token-by-token streaming과 full decode를 나눈다. Byte fallback pieces가 UTF-8 code point 중간에서 끝나면 streaming decoder는 incomplete bytes를 보류해야 한다. Replacement character를 먼저 방출했다가 다음 token에서 되돌릴 수는 없다. Per-stream pending byte buffer의 owner와 request incarnation을 기록한다. 두 request가 buffer를 공유하면 텍스트가 섞이는 보안/correctness 사고가 된다.

Cleanup policy도 decode output을 바꾼다. `clean_up_tokenization_spaces` 같은 후처리는 punctuation 주변 space를 바꾸지만 model token IDs를 바꾸지 않는다. Offset/highlight consumer가 cleaned text에 raw token offsets를 적용하면 mismatch가 난다. UI text generation과 evidence text generation을 분리하고 어느 좌표에서 offsets가 유효한지 적는다. 화면이 예쁘다는 이유로 원문 증거를 덮어쓰지 않는다.

### 6.8.2 span만 깨진 갈래: offset mapping을 세 좌표의 대수로 검산한다

둘째 갈래는 token ID가 아니라 그 token이 가리키는 원문 좌표를 판정한다. 숫자 tuple만 같아도 기준 buffer와 단위가 다르면 UI와 scoring consumer는 다른 문자를 자르므로 byte·scalar·grapheme 변환을 명시한다.

Raw UTF-8 byte 구간을 B, Unicode scalar 구간을 S, grapheme cluster 구간을 G라고 하자. Tokenizer primitive가 normalized byte interval N을 반환한다면 consumer가 요구하는 S 또는 G로 가는 mapping이 필요하다. `N→B→S→G` 각 변환은 단조일 수 있지만 일대일이라고 보장되지 않는다. 정규화가 순서를 바꾸거나 문자를 삭제하면 interval endpoint만 옮기는 방식은 내부 gap 정보를 잃는다. Fixture는 적어도 coverage set 또는 alignment edge를 보존한다.

UTF-8에서 ASCII는 byte 수와 scalar 수가 같아 잘못된 변환도 통과한다. 한국어 완성형은 scalar 하나에 세 bytes, emoji는 네 bytes일 수 있으며 variation selector와 skin-tone modifier는 별 scalar지만 한 grapheme에 속할 수 있다. 따라서 `len(text)`와 `len(text.encode())`가 다른 fixture가 offset test의 최소 조건이다. Python slice는 scalar-like code point index를 쓰지만 JavaScript DOM과 일부 언어는 UTF-16 code unit을 쓸 수 있어 client boundary에 네 번째 좌표가 생긴다.

Cross-language API에서는 offset unit을 schema에 명시한다. Server가 UTF-8 byte를 반환하고 browser가 UTF-16 code unit을 기대하면 중간 conversion library의 Unicode version과 malformed input policy도 identity다. Offset 숫자를 그대로 전달하는 gateway는 단순 proxy가 아니라 semantic bug owner가 될 수 있다. Client SDK별 fixture가 같은 raw bytes와 highlighted grapheme를 가리키는지 본다.

Zero-width joiner emoji는 좋은 경계 fixture다. 여러 scalar와 bytes가 하나의 grapheme를 만들지만 tokenizer는 내부에서 여러 pieces로 나눌 수 있다. 각 token offset이 grapheme 일부를 가리키는 primitive contract는 가능하다. UI highlight가 grapheme를 찢으면 안 된다면 consumer가 union span을 grapheme boundary로 확장한다. Tokenizer에게 모든 token을 grapheme 원자로 만들라고 요구하면 model artifact와 다른 segmentation을 강제하게 된다.

Bidirectional control과 invisible characters도 다룬다. 화면 순서와 logical string index가 다를 수 있으므로 visual x-coordinate를 token offset으로 역산하지 않는다. Security UI는 invisible span을 명시적으로 표시할 수 있지만 model input은 configured normalization을 따른다. Raw bytes digest와 logical offset을 보존해 incident에서 보이는 문장만으로 입력을 재구성하지 않는다.

Malformed UTF-8을 받을 수 있는 byte API는 text API와 별 contract다. Python string entry는 이미 decode가 끝났으므로 invalid byte가 들어올 수 없지만 llama.cpp 같은 byte buffer path나 file ingestion은 가능하다. Reject, replacement, byte fallback 중 정책을 고정한다. Replacement 뒤 offset은 원 bytes로 many-to-one mapping되며, silently dropping invalid byte는 security와 cache identity를 바꿀 수 있다.

Offset round-trip 검산은 token span을 raw substring으로 잘라 재-encode했을 때 같은 token이 나오는지 보는 보조 시험이다. Context-dependent BPE와 leading-space 규칙 때문에 개별 substring 재-encode가 원 token과 다를 수 있으므로 hard invariant로 쓰지 않는다. 대신 span이 원문 coverage를 설명하는지, 전체 ordered spans와 declared virtual/discarded 영역을 합치면 input mapping이 복원되는지를 본다.

### 6.8.3 latency·수명 주기가 달라진 갈래: cache와 scheduler까지 추적한다

셋째 갈래는 앞의 작은 의미 차이가 queue와 KV 비용으로 증폭되는 순간을 본다. 여기서는 tokenizer service time과 downstream token-count shape를 분리해야 worker 최적화가 cache·scheduler 회귀를 가리지 않는다.

Tokenizer 결과가 한 token 달라지면 prompt length, block boundary와 scheduler chunk가 달라질 수 있다. 16-token block에서 길이 256과 257은 physical block 수가 16과 17로 갈린다. Unicode normalization 하나가 token을 추가하면 KV allocation과 graph bucket, prefix hash가 모두 바뀔 수 있다. TTFT 차이를 tokenizer CPU 시간만으로 설명하지 않고 downstream shape transition을 함께 기록한다.

Prefix cache는 final IDs가 같을 때만 model-state identity 후보가 된다. Raw text가 달라도 normalization 뒤 IDs와 positions가 같으면 KV를 공유할 수 있지만 tenant, adapter, multimodal feature와 cache policy가 추가 predicate다. 반대로 raw digest만 key로 쓰면 안전하지만 Unicode equivalent 입력 사이 miss가 늘어난다. 최적화는 parity proof 뒤 단계적으로 key projection을 줄인다.

Response cache는 더 넓은 identity가 필요하다. Safety policy가 raw confusable을 구분하거나 audit가 원문을 보존해야 하면 같은 model IDs라도 response reuse를 제한할 수 있다. Model computation key와 product response key를 하나로 합치지 않는다. Cache hit율 숫자를 비교할 때 어느 key space인지 명시한다.

Scheduler admission estimate가 client character length를 사용하고 실제 work는 token length를 사용하면 언어별 bias가 생긴다. 한국어/emoji 요청은 character 대비 bytes와 tokens 관계가 영어와 다르다. Admission은 가능한 한 selected tokenizer generation의 token count 또는 calibrated bound를 사용하고, estimate error를 language label 대신 length/byte/token bucket으로 관측한다. Raw language를 고카디널리티 label로 넣지 않는다.

Tokenizer worker batching도 공정성을 바꾼다. 긴 문서 한 건이 native batch를 오래 점유하면 짧은 chat이 head-of-line blocking을 겪을 수 있다. Batch 최대 item 수보다 total bytes/code points와 expected tokens budget이 중요하다. Slow/fast backend의 service time 분포가 달라지면 같은 batching policy가 다른 queue tail을 만든다. Selector 변경과 worker scheduler 변경을 한 실험으로 합치지 않는다.

Cancellation은 tokenization lifetime을 닫아야 한다. Client가 끊겼는데 native batch 안 작업을 취소할 수 없으면 결과가 늦게 돌아올 수 있다. Late result를 reused request ID나 cache entry에 붙이지 않도록 incarnation fence가 필요하다. CPU work를 실제로 중단할 수 없는 경우에도 result publish와 downstream admission은 차단해야 한다. Cancel success metric이 compute stop을 뜻하는지 구분한다.

#### pinned source를 caller와 consumer까지 읽는 순서

Transformers source에서는 high-level encode 호출이 fast/slow 공통 base contract를 어떻게 정의하고 concrete backend가 어떤 method를 override하는지 본다. `return_offsets_mapping` validation, AddedToken conversion, truncation/padding setup과 batch encoding 결과 construction을 각각 찾는다. Base docstring은 public contract를 말하지만 native engine의 실제 alignment와 error behavior는 concrete call과 pinned tokenizer library artifact가 필요하다.

Fast implementation이 backend encoding object의 offsets를 그대로 노출한다면 그 offsets의 reference text/unit은 backend documentation/source에서 확인한다. Python wrapper가 batch/overflow encodings를 재배열하며 sequence IDs와 offsets를 합치는지도 본다. Overflow window가 있으면 같은 original span이 여러 window에 나타날 수 있다. 중복은 bug가 아니라 stride contract일 수 있으므로 overflow mapping을 보존한다.

Slow implementation이 offsets를 지원하지 않는 경우 fake parity를 만들지 않는다. `unsupported`와 `unverified`를 구분한다. 운영 consumer가 offsets를 필수로 요구하면 slow fallback은 correctness capability를 만족하지 않는다. Highlight 기능을 끄고 model serving만 계속하는 degraded mode가 가능한지 제품 contract가 정한다. Backend latency만으로 fallback을 승인하지 않는다.

vLLM source walk는 OpenAI/rendered prompt가 engine token prompt로 normalize되는 경계를 잡는다. Tokenizer group/pool이 request와 adapter별 tokenizer를 선택할 수 있다면 model name 하나로 generation을 합치지 않는다. Async tokenization result가 core request로 들어갈 때 request incarnation, token count와 multimodal placeholder metadata가 같은 owner에 연결되는지 확인한다.

SGLang에서는 tokenizer manager process가 public request, tokenization과 scheduler IPC를 소유한다. API tokenize endpoint와 generation path가 같은 helper/flags를 쓰는지 확인한다. Debug endpoint가 낸 IDs를 production generation의 증거로 쓰려면 template, special-token, truncation과 image handling 조건이 같아야 한다. IPC send 성공은 scheduler가 해당 IDs를 admission했다는 terminal이 아니다.

llama.cpp에서는 byte input과 vocabulary tokenization API의 capacity/retry contract를 읽는다. Caller가 먼저 필요한 token 수를 질의하고 buffer를 재할당하는지, negative/return code를 length로 오독하지 않는지 본다. Special-token parsing flag와 byte fallback, add-special behavior는 호출마다 달라질 수 있다. GGUF vocabulary metadata와 runtime flags가 함께 tokenizer generation을 만든다.

네 stack crosswalk는 함수 이름 표가 아니다. Raw input owner, template owner, tokenizer artifact selector, normalization/model backend, final IDs consumer와 cancellation lifetime을 같은 의미 좌표로 놓는다. 어떤 stack에 server queue가 없으면 `not owned`라고 쓰고 가상의 manager를 만들지 않는다. Exact link는 이 좌표의 대표 mutation을 증명할 때만 남긴다.

#### 승인용 regression matrix와 종료 문장

Regression matrix의 행은 ASCII baseline, NFC/NFD pair, compatibility character, whitespace variants, AddedToken boundary, emoji grapheme, invalid-byte policy, long overflow와 streaming split이다. 열은 old/new×fast/slow, single/batch와 offsets on/off다. 모든 cell을 production traffic 가중치로 평균내지 않는다. Boundary correctness는 한 건 실패해도 hard fail이고 성능은 representative cohort로 별 판정한다.

각 cell은 normalized digest, pre-token spans, tokens/IDs, offset unit/reference, decode pending bytes와 effective backend를 낸다. Expected equality relation을 사전에 쓴다. NFC/NFD가 같아야 하는지는 configured normalizer에 달렸고 raw-preserving tokenizer라면 달라야 할 수 있다. “Unicode equivalent”라는 외부 직관을 model artifact contract 위에 놓지 않는다.

첫 divergence가 normalizer라면 downstream ID와 cache 차이는 결과다. Pre-tokenizer라면 normalizer parity를 닫고 boundary rule을 수정한다. Model trace에서 처음 갈리면 vocabulary/merge/unknown을 본다. IDs는 같고 offsets만 갈리면 mapping과 consumer 좌표를 본다. 모든 tokenizer output이 같으면 UI, retrieval 또는 security consumer를 조사한다. 이 stop rule이 source search를 제한한다.

Rollback은 package version만 되돌리지 않는다. Serialized tokenizer artifact, AddedToken registry, template/special-token config, worker process와 cache namespace를 같은 generation으로 복원한다. New generation이 만든 prefix cache가 final IDs parity를 통과한다면 재사용 가능하지만 증거가 없으면 분리한다. In-flight request result는 old/new incarnation에 정확히 귀속한다.

Canary는 selected backend를 증명해야 한다. Fast bug를 고쳤는데 traffic이 모두 slow fallback을 탔다면 거짓 pass다. Fast/slow 각 lane에 최소 boundary fixture가 들어가고 effective class와 native artifact digest가 trace에 보여야 한다. Batch와 streaming path도 실제 consumer를 선택했는지 확인한다. Telemetry가 없으면 unknown이지 성공 0건이 아니다.

Correctness terminal은 configured normalization·segmentation·special-token program이 backend와 관계없이 expected final IDs를 만들고, offset capability가 선언한 좌표에서 원문 consumer 요구를 만족하며, decode streaming이 request별 pending bytes를 안전하게 닫는 상태다. Cache terminal은 공유 predicate가 이 parity generation과 model/product scope를 정확히 반영하는 상태다. Performance terminal은 correctness를 통과한 lane에서 CPU, queue tail과 downstream shape 비용이 budget 안인 상태다.

최종 incident 문장은 다음 형태가 된다. “New-fast lane은 NFD fixture에서 token IDs는 reference와 같았지만 normalized UTF-8 byte offset을 raw Python scalar offset으로 노출했고 retrieval consumer가 그대로 slice해 highlight와 security coverage를 한 grapheme 앞으로 이동시켰다. Alignment-based conversion과 schema unit 명시 뒤 old/new fast/slow boundary matrix가 통과했고 cache namespace는 parity가 증명된 generation만 공유했다. Fast queue p99도 별 performance budget을 통과했다.” 이 문장은 backend 이름보다 최초 잘못된 계약과 복구 범위를 알려 준다.

독자가 이 matrix를 새 tokenizer에 적용할 수 있으면 특정 library API를 외우지 않아도 된다. Raw bytes에서 normalizer alignment, pre-token span, model piece/ID, post-processing, offset consumer와 cache/scheduler까지 owner를 따라가면 된다. Source가 이동해도 serialized stage type, mutation, output object와 consumer라는 re-search key가 같은 의미를 다시 찾게 한다.

실제 review 회의에서는 이 긴 matrix를 세 장의 카드로 압축할 수 있다. 첫 카드는 의미 카드다. Raw fixture, configured normalization, expected token IDs와 offset consumer를 적는다. 둘째는 실행 카드다. Requested/constructed backend, serialized artifact, single/batch/streaming path와 단계별 digest를 적는다. 셋째는 운영 카드다. Queue population, cache namespace, rollback generation과 correctness/performance terminal을 적는다. 세 카드가 같은 tokenizer generation을 가리켜야 한다.

의미 카드가 없으면 library가 낸 값끼리만 비교해 두 backend가 똑같이 틀린 경우를 놓친다. 실행 카드가 없으면 설정상 fast를 실제로 fast path가 실행된 것으로 오인한다. 운영 카드가 없으면 unit fixture 수정 뒤 stale cache와 old worker가 production에 남는다. 문서가 길어지는 이유는 필드를 많이 모으기 위해서가 아니라 이 세 질문을 서로 대신하지 않게 하기 위해서다.

리뷰어는 대표 fixture 하나를 역방향으로 읽는다. UI가 highlight한 raw grapheme에서 시작해 consumer offset unit, tokenizer가 반환한 reference text, encoding object의 span, pre-token substring, normalized alignment와 raw bytes로 돌아간다. Forward trace와 reverse trace가 같은 interval relation에서 만나면 offset 설명이 닫힌다. 중간에 단순 integer pair만 남으면 그 경계가 다음 instrumentation 대상이다.

성능 회귀도 역방향으로 읽을 수 있다. Client TTFT 증가에서 tokenizer queue wait, native/Python execution, offset construction, downstream prompt length와 scheduler wait를 분해한다. Fast execution은 줄었는데 total이 늘었다면 batch fill이나 token count 변화가 원인 후보이고, tokenization phase 자체가 늘었다면 Unicode pathological input, fallback 또는 artifact drift를 본다. CPU utilization 한 숫자로 결론 내리지 않는다.

Artifact upgrade에서는 old/new tokenizer JSON의 text diff만 보지 않는다. Normalizer/pre-tokenizer/model/post-processor/decoder를 semantic stage로 diff하고 vocabulary·merge·AddedToken digest를 연결한다. 같은 stage type이라도 option default가 바뀌었는지, 같은 option이라도 native dependency가 의미를 바꿨는지 확인한다. Old source evidence가 없으면 current behavior를 과거에 역투영하지 않고 boundary fixture를 compatibility 근거로 둔다.

Offset schema 변경은 API versioning 대상이 될 수 있다. Byte에서 character로 unit을 바꾸면서 field 이름을 유지하면 기존 consumer가 조용히 틀린 slice를 만든다. 새 field 또는 explicit unit, rollout matrix와 dual-read 기간을 설계한다. Dual-read는 두 숫자를 로그에 남기는 것으로 끝나지 않고 같은 raw span을 가리키는지 comparator가 판정해야 한다. 불일치는 tenant/raw text를 노출하지 않는 digest와 bounded coordinate로 조사한다.

Fast/slow fallback 정책도 오류 종류별로 다르다. Native artifact load failure는 slow가 같은 semantic capability와 performance budget을 통과하면 안전 fallback일 수 있다. Offset unsupported인 slow path는 annotation endpoint에서 fail closed해야 할 수 있다. Tokenizer parity failure가 알려진 fixture에서는 slow가 known-good이면 affected cohort만 fence할 수 있다. 모든 예외를 slow로 보내면 defect와 fallback 비용이 숨고 capacity가 갑자기 무너진다.

마지막으로 training tokenizer와 serving tokenizer의 관계를 경계로 남긴다. 이 권은 training 과정을 다루지 않지만 serving artifact가 model weights가 기대한 vocabulary·normalization program과 같은지는 확인해야 한다. Vocab size만 같아도 ID→piece mapping이 다르면 embedding은 유효한 다른 row를 읽어 silent wrong answer를 만든다. Model bundle은 tokenizer artifact digest와 special-token IDs를 포함하고 server override는 explicit compatibility fixture 없이는 허용하지 않는다.

이 terminal을 다음 장의 chat template가 이어받는다. 7장은 messages와 tools를 rendered text로 만드는 compiler를 다루지만, 그 출력이 IDs가 되는 경계에서는 여기서 확정한 tokenizer generation, special-token producer와 offset contract를 사용한다. Template 회귀와 tokenizer 회귀를 분리하려면 rendered bytes checkpoint가 필요하다. Bytes가 다르면 template 쪽, bytes는 같고 IDs가 다르면 이 장의 stage 쪽에서 first divergence를 찾는다.

이제 지금까지의 경계를 하나의 회귀 사건으로 묶어 보자. 모델과 vocabulary 파일은 바뀌지 않았는데 tokenizer package를 올린 뒤 한국어와 결합 문자가 섞인 요청에서만 인용 구간이 한 글자씩 밀렸다. 생성 token 자체는 대부분 같았지만 retrieval 문서의 highlight가 틀렸고, 금칙어 검사기는 어떤 요청에서 문자열의 절반만 검사했다. Fast backend를 끄면 highlight는 돌아왔지만 처리 시간이 늘었다. 이때 “Rust tokenizer의 Unicode 버그”라고 결론 내리면 아직 너무 넓다. 정규화 결과, pre-token 경계, token ID, offset 단위와 원문 복원 가운데 어느 계약이 처음 달라졌는지 나누지 않았기 때문이다.

Fixture는 눈으로 비슷하지만 내부 좌표가 다른 조각을 의도적으로 포함한다. 첫 조각은 완성형 `가`와 자모 `ᄀ`+`ᅡ`, 둘째는 `é`와 `e`+combining acute, 셋째는 emoji 뒤 variation selector, 넷째는 non-breaking space와 일반 space, 다섯째는 한글·ASCII·emoji가 붙은 `GPU가✅좋다`다. 각 조각에 원문 UTF-8 bytes, Unicode scalar index, 사용자에게 보이는 grapheme cluster index를 미리 적는다. 이 세 좌표를 마련하지 않으면 offset 3이 세 번째 byte인지 세 번째 code point인지 세 번째 화면 글자인지 판정할 수 없다.

첫 비교점은 backend 출력이 아니라 normalizer 출력이다. Slow와 fast가 모두 NFC를 적용한다면 decomposed `e`+accent는 같은 normalized text가 되어야 한다. 한쪽만 strip accent 또는 compatibility normalization을 적용하면 token boundary 이전에 이미 다른 프로그램이다. 이 경우 token ID 차이를 BPE merge 차이라고 부르면 안 된다. Normalizer class, serialized configuration, 입력·출력 bytes와 normalized-to-original alignment를 한 행에 둔다. Normalized 문자열이 같을 때만 다음 pre-tokenizer 비교로 넘어간다.

두 번째 비교점은 pre-token span이다. Whitespace split이 non-breaking space를 separator로 보는지, punctuation rule이 emoji variation selector 사이를 자르는지, byte-level pre-tokenizer가 leading-space marker를 언제 넣는지 확인한다. 이 단계의 경쟁 가설은 두 가지다. 경계 자체가 다르거나, 경계는 같지만 offset을 다른 좌표계로 보고할 수 있다. Span의 normalized substring과 byte slice를 다시 잘라 원본과 대조하면 둘을 가를 수 있다. 동일 substring인데 숫자 offset만 다르면 segmentation보다 mapping 계약을 먼저 본다.

세 번째 비교점은 model algorithm 입력이다. BPE라면 initial symbols와 merge rank를, WordPiece라면 각 위치의 longest-match 후보와 continuation prefix를, Unigram이라면 lattice edge와 누적 score를 기록한다. Fast/slow가 동일 pre-token과 동일 vocabulary·merge artifact를 소비했다면 token piece와 ID는 같아야 한다는 강한 parity 가설을 세울 수 있다. 다르면 unknown 처리, AddedToken 우선순위, byte fallback 또는 artifact resolution을 조사한다. “같은 tokenizer 이름”은 merge file digest와 AddedToken state가 같다는 증거가 아니다.

Offset은 token ID parity와 별 terminal이다. 예를 들어 normalized `é` 한 code point가 원문 `e`+accent 두 code points에서 왔다면 normalized span `(0,1)`을 original character span으로 단순 복사할 수 없다. Alignment map은 한 normalized 위치가 원문 여러 위치에 대응하는 many-to-one 관계를 보존해야 한다. UTF-8 byte로는 `é`가 두 bytes이고 decomposed sequence는 세 bytes이므로 byte offset도 달라진다. Highlight consumer가 Python string slice를 기대하는데 tokenizer가 byte offset을 반환하면 ASCII fixture는 통과하고 한국어·emoji에서만 실패한다.

따라서 offset row에는 `reported_unit`, `reference_text`, `start/end`, `slice_result`, `round_trip_to_original`을 쓴다. Reference text는 raw, normalized, pre-token 중 하나다. `start=5,end=8`만 저장하면 단위와 대상 문자열을 잃는다. Consumer가 offset을 어떤 문자열에 적용하는지도 source에서 확인한다. Tokenizer가 정확한 normalized byte offset을 냈어도 UI가 raw character offset으로 오독하면 defect owner는 integration boundary다.

Fast backend가 slow보다 빠른 이유를 “Rust라서” 한 단어로 끝내지 않는다. Native backend는 normalizer, pre-tokenizer와 model loop를 한 pipeline에서 실행하고 batch allocation과 Python object 왕복을 줄일 수 있다. 반면 slow path는 override 가능한 Python method, 사용자 subclass와 단계별 object를 거칠 수 있다. 이 차이는 성능뿐 아니라 관찰 가능성과 extension surface를 바꾼다. Fast path가 offset mapping을 제공하는 반면 slow path는 offset을 제공하지 않거나 다른 근사 구현을 쓸 수도 있다. API field 이름이 같아도 `None`, absent와 실제 mapping은 다른 상태다.

성능 비교는 text 길이 하나로 하지 않는다. Short chat 64 bytes, 긴 한국어 문서 64KiB, AddedToken이 많은 tool prompt, pathological combining sequence를 별 cohort로 둔다. 시간은 normalization, pre-tokenization, model encode, offset materialization, Python result construction으로 나눈다. Fast total이 짧아도 offset materialization이 전체의 큰 비중이면 offsets를 요구하지 않는 serving path와 요구하는 annotation path를 분리할 수 있다. 하지만 두 path의 token ID가 같다는 parity gate 없이 빠른 path를 채택하지 않는다.

Queueing 관점도 필요하다. 평균 encode가 2ms에서 1ms가 되어도 single tokenizer worker 앞 arrival rate가 service rate에 가까우면 tail은 크게 줄 수 있다. 반대로 native call이 길게 GIL을 놓더라도 batch를 지나치게 크게 묶어 짧은 요청이 긴 문서 뒤에서 기다리면 p99는 나빠질 수 있다. `backend latency`와 `tokenizer queue wait`, batch fill, downstream scheduler wait를 분리한다. Fast/slow switch가 worker concurrency와 memory footprint를 함께 바꾸면 순수 algorithm 비교가 아니다.

회귀 실험은 네 lane으로 만든다. Old-slow, old-fast, new-slow, new-fast를 동일 raw bytes와 동일 serialized artifact에 적용한다고 가정한다. Old/new 모두 fast만 틀리면 native backend 또는 binding 가설이 강하다. New의 두 backend가 같이 틀리면 normalizer/vocabulary artifact 또는 shared wrapper 변화가 가깝다. New-fast ID는 맞고 offset만 틀리면 mapping/consumer boundary를 본다. 모든 tokenizer output이 같지만 highlight만 틀리면 UI slice와 normalization 이후 원문 선택을 본다. 이 matrix가 package rollback 하나보다 더 좁은 결론을 준다.

각 lane은 다섯 checkpoint를 남긴다. `raw_bytes_digest`, `normalized_bytes_digest`, `pretoken_spans`, `token_ids`, `offsets(unit,text_generation)`이다. Decode 결과는 여섯 번째 참고점이지 encode mapping의 완전한 역증명이 아니다. Normalization이 정보를 잃었거나 unknown/byte fallback을 썼다면 같은 화면 문자열이 돌아와도 원문 위치는 복구되지 않을 수 있다. 반대로 decode 문자열이 다르더라도 intended token IDs와 model semantics가 맞고 whitespace cleanup만 다를 수 있다. 어떤 contract를 승인하는지 분리한다.

금칙어 incident에서는 security terminal이 추가된다. 검사기가 raw text를 검사하는지 normalized text를 검사하는지, token offsets로 어느 span을 추출하는지 정책을 고정한다. Unicode confusable과 normalization을 tokenizer에 전부 맡기지 않는다. Tokenization parity는 security equivalence를 뜻하지 않는다. 다만 offset 단위가 틀려 검사 범위가 누락되는 correctness defect는 fixture의 모든 non-ASCII span이 완전히 덮이는지로 반증할 수 있다.

Cache identity도 같은 matrix로 검토한다. Raw bytes가 다르지만 normalization 뒤 token IDs가 같을 때 cache를 공유할지, 같은 token IDs지만 template/security policy가 다를 때 분리할지 결정해야 한다. Model KV reuse의 최소 의미 identity는 실제 input token IDs와 model/adapter/position 계약이지만, response/security cache에는 raw/normalized policy와 tenant가 추가될 수 있다. Tokenizer backend 이름을 무조건 key에 넣으면 안전하지만 불필요한 miss를 만들고, 빼면 parity가 깨진 backend 사이 wrong hit를 만들 수 있다. `backend-independent parity proven` generation에서만 key 축을 제거한다.

수정 후보가 fast binding에서 original offset conversion을 빠뜨렸다고 하자. Fix는 모든 offset에 character 길이를 더하는 식이 아니다. Normalizer alignment를 이용해 normalized byte span을 raw consumer가 요구하는 coordinate로 변환하고, many-to-one/one-to-many 경계에서 cover 정책을 정의한다. Highlight는 원문 grapheme 전체를 덮는 conservative span을, exact edit API는 ambiguous mapping을 오류로 처리할 수 있다. 동일 mapping이라도 consumer 목적에 따라 terminal policy가 다르다.

회귀 bundle에는 위 Unicode fixture 외에 empty text, only-special-token, invalid UTF-8을 허용하는 byte API, very long combining run과 streaming chunk boundary를 넣는다. Streaming에서 UTF-8 code unit이 chunk 사이에 갈라졌다면 decoder가 완성 전 byte를 tokenization에 넘기는지, buffer가 다음 chunk까지 보류하는지 확인한다. Batch path와 single path가 같은 pipeline을 호출하는지도 본다. Fast single만 고쳤는데 batch encode가 옛 conversion을 쓰면 production 회귀가 남는다.

최종 판정은 “fast와 slow가 같다”보다 구체적이다. 동일 raw fixture와 serialized tokenizer generation에서 normalized bytes, pre-token substring, token IDs가 같고, 각 backend의 offset은 선언한 reference text/unit에서 같은 원문 span을 가리킨다. AddedToken과 byte fallback branch도 같은 우선순위를 지닌다. 성능 lane은 이 correctness terminal 뒤에 queue와 CPU budget을 통과한다. 한 항이라도 unknown이면 fast path를 전면 default로 승격하지 않고 영향 cohort만 기존 backend로 보낸다.

이 종합 실습이 중요한 이유는 tokenizer를 “문자열을 ID로 바꾸는 함수”에서 끝내지 않기 때문이다. 실제 serving에서는 normalization과 segmentation이 cache key를 만들고, offset이 retrieval·security·annotation을 연결하며, backend 선택이 latency와 관찰 계약을 함께 바꾼다. 원문→normalized text→pre-token span→piece/ID→consumer offset의 다섯 화살표를 같은 generation으로 닫으면 Unicode 회귀를 특정 언어나 native library의 막연한 탓으로 돌리지 않고 최초 어긋난 계약을 수정할 수 있다.

이 절의 종료 조건은 단순히 “두 backend의 ID가 같다”가 아니다. fixture마다 원문 byte, 정규화 문자열, pre-token span, 최종 ID, offset의 단위와 기준 문자열을 함께 보존하고, 불일치가 최초로 나타난 경계를 기록해야 한다. 그 기록으로 cache key가 갈라지는지, 길이 산정과 admission이 달라지는지, decode 결과의 공백과 byte 복원이 달라지는지까지 소비자 방향으로 추적해야 한다. 반대로 downstream 증상에서 시작했다면 동일한 경로를 거꾸로 올라가 최초의 표현 차이까지 닫는다. 그래야 회귀 테스트가 우연히 같은 출력 하나를 확인하는 스냅샷이 아니라, tokenizer 교체와 버전 상승을 견디는 계약이 된다.

실무 검토표에는 비교 결과만 적지 말고 판정 불능도 명시한다. 예를 들어 slow backend가 offset을 제공하지 않는다면 빈 배열을 동일하다고 처리하지 않는다. 대신 `unsupported`로 남기고, token piece를 원문에 재정렬해 얻은 대체 좌표인지 backend가 직접 낸 좌표인지 provenance를 분리한다. Unicode normalization form, invalid byte 처리, `add_special_tokens`, truncation, 앞뒤 공백 보존도 fixture의 입력 열이어야 한다. 마지막으로 실패 행에는 최초 분기 함수와 그 분기의 소비자 한 곳을 함께 적는다. 이 두 좌표가 있어야 담당자가 tokenizer 구현 문제와 serving integration 문제를 서로 떠넘기지 않고 수정 범위를 정할 수 있다.

배포 승인 때는 대표 ASCII 문장만 통과시키지 않는다. 결합 문자, 한글 자모, emoji sequence, 잘못된 UTF-8, 문장 경계의 연속 공백을 고정 corpus로 두고 backend별 결과를 버전과 함께 보존한다. 새 실패가 의도된 변화라면 fixture를 지우지 말고 변경 이유와 cache 무효화 범위를 같이 기록한다.

## 6.9 장말 소스 노트와 짧은 회고

이 장의 마지막에서는 글자 하나가 모델 입력이 되는 경로를 한 번만 다시 걷는다. 화면에는 같아 보이는 glyph도 byte·normalization·pre-token 경계에서 달라질 수 있고, 그 첫 차이가 ID, embedding row, cache identity와 실행 비용으로 전파된다. 아래 화살표는 외울 순서가 아니라 최초 불일치를 멈출 순서다.

### 6.9.1 표현의 차이가 serving 비용으로 전파되는 경로

이 장에서 가장 중요한 문장은 “문자열은 곧 token이 아니다”가 아니다. 더 구체적으로 다음 인과를 말할 수 있어야 한다.

```text
같아 보이는 glyph
→ 다른 code point/byte 또는 normalization
→ 다른 pre-token 경계
→ 다른 subword 경로와 integer IDs
→ 다른 embedding row·sequence length·cache prefix
→ 다른 correctness와 serving cost
```

BPE는 merge rank 프로그램이고, WordPiece는 continuation vocabulary 위의 greedy match이며, Unigram은 전체 segmentation 경로 score를 비교한다. SentencePiece는 이 중 model type과 normalization·공백 처리·serialization을 묶는 체계다.

fast tokenizer는 단지 빠른 구현이 아니라 offset/alignment라는 관찰 계약을 더 제공할 수 있다. AddedToken은 base model 뒤에 ID만 추가하는 것이 아니라 base segmentation 앞에서 문자열 경계를 가로챈다. decode는 normalization·cleanup·unknown·special skip 때문에 encode의 완전한 역함수가 아니다.

### 6.9.2 증상에서 가설을 버리는 순서

운영에서 tokenization을 의심할 때는 GPU profiler부터 켜지 않는다. 원문 byte digest, tokenizer bundle revision, token count/ID digest, render/tokenize interval을 먼저 비교한다. first divergence가 tokenizer 뒤라면 그때 embedding과 model로 이동한다. normalized text와 IDs가 같다면 tokenization 가설을 버릴 수 있다.

### 6.9.3 독자 회수 뒤에 여는 고정 소스 좌표

이 장의 구현 관찰점은 Transformers v5.15.1 commit `550d7b3834670483a4df436541272c055dc364bf`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, llama.cpp v0.2.0 commit `bb4caa7540188872173c44d161602d9271386413`에 고정했다. 아래 좌표는 이 소스를 정적으로 읽은 근거이며, 서버나 모델을 실행해 성능을 측정했다는 뜻은 아니다.

- Transformers v5.15.1 — [`TokenizerBackend`와 tokenizer JSON 구성](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L101-L178)
- Transformers v5.15.1 — [`tokenize`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L847-L866), [`_encode_plus`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L925-L1008), [`_decode`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_tokenizers.py#L1086-L1111)
- Transformers v5.15.1 — [SentencePiece `_tokenize`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_sentencepiece.py#L204-L265)
- vLLM v0.27.1 — [`BaseRenderer` tokenization/offset handoff](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/renderers/base.py#L431-L488)
- vLLM v0.27.1 — [public prompt tokenization methods](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/renderers/base.py#L614-L649)
- vLLM v0.27.1 — [tokenize/detokenize serving](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/serve/tokenize/serving.py#L32-L154)
- SGLang v0.5.18 — [OpenAI tokenize/detokenize serving](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_tokenize.py#L20-L154)
- SGLang v0.5.18 — [HTTP tokenize/detokenize routes](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/http_server.py#L1758-L1796)
- llama.cpp v0.2.0 — [`llama_tokenize`/`llama_detokenize` API](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/include/llama.h#L1153-L1202)
- llama.cpp v0.2.0 — [`common_tokenize`/`common_detokenize` buffer lifecycle](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/common.cpp#L1836-L1907)
- llama.cpp v0.2.0 — [GGUF tokenizer normalizer metadata parsing](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/gguf-py/gguf/vocab.py#L161-L181)

다음 장에서는 messages와 role, tool schema가 어떤 문자열로 compile되는지 본다. tokenizer는 그 문자열의 경계를 정하지만, 어떤 문자열을 만들 것인지는 chat template의 책임이다. 이 두 artifact를 한 버전으로 뭉치면 cache miss와 의미 변화의 최초 원인을 구별할 수 없다.
