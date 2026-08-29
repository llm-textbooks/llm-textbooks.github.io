# 7장. 채팅 템플릿이 실제 프롬프트를 바꾸는 순간

채팅 API에 `system`, `user`, `assistant`라는 역할을 담아 보냈다고 해서 모델이 그 JSON 객체를 읽는 것은 아니다. 모델이 받는 것은 정수 토큰의 한 줄짜리 배열이다. 역할 이름, 메시지 경계, 도구 명세, 이미지 자리, “이제 assistant가 대답할 차례”라는 신호는 모두 그 배열 안의 토큰으로 번역되어야 한다. 이 번역을 맡는 것이 채팅 템플릿이다.

이 장의 중심 질문은 단순하다. **같은 messages 배열이 왜 서버와 템플릿에 따라 다른 모델 입력이 되는가?** 답을 얻기 위해 채팅 템플릿을 문자열 꾸미기 도구가 아니라 작은 컴파일러로 본다. 입력 언어는 role과 content를 가진 구조화 메시지이고, 출력 언어는 모델이 학습한 제어 토큰 프로토콜이다. 도구 호출과 reasoning, 멀티모달 content는 이 컴파일러의 확장 문법이며, generation prompt와 assistant mask는 입력 구간과 생성 구간의 경계를 표시한다.

컴파일러라는 비유는 유용하지만 완전하지 않다. 일반 컴파일러에는 비교적 명시적인 언어 명세와 타입 시스템이 있다. 채팅 템플릿의 실제 명세는 모델 학습 데이터의 직렬화 관습, tokenizer의 special token 설정, Jinja 코드, API 서버의 전처리 정책에 흩어져 있다. 문법적으로 렌더링에 성공해도 모델이 학습한 형식과 다를 수 있다. 반대로 두 문자열이 눈으로 비슷해도 줄바꿈 하나나 special token 한 개가 다른 토큰열을 만들 수 있다. 따라서 이 장에서 “컴파일이 성공했다”는 말은 Jinja가 예외 없이 문자열을 만들었다는 뜻에 그치지 않는다. 구조화 입력의 의미가 모델 입력 토큰과 서버 회계까지 일관되게 보존되었다는 뜻으로 사용한다.

이 장은 실행 결과를 주장하지 않는다. 소스는 Transformers 5.15.1 commit `550d7b3`, vLLM 0.27.1 commit `6e448d0`, SGLang 0.5.18 commit `71de97b`, llama.cpp commit `bb4caa7`에 고정한다. 코드를 읽어 상태와 분기를 복원하지만 모델이나 서버를 실행하지 않는다. 독자는 장을 마친 뒤 렌더링 전 messages, 렌더링 문자열, token IDs, generation 경계, cache·usage identity를 서로 다른 산출물로 기록하고, 최초 불일치가 어느 경계에서 생겼는지 설명할 수 있어야 한다.

## 7.1 같은 질문인데 한 서버에서만 답이 달라지는 이유

금요일 오후, 모델 교체 없이 API gateway만 바꾼 뒤 답변 품질이 흔들렸다고 해 보자. 사용자가 보낸 JSON은 같고 temperature도 0이며 model 이름도 같다. 운영자는 흔히 GPU kernel, quantization, random seed부터 의심한다. 그러나 첫 번째로 비교할 것은 GPU가 아니라 **렌더링된 prompt**다. 한 서버는 system 메시지 뒤에 줄바꿈을 두 번 넣고, 다른 서버는 한 번 넣었을 수 있다. 한쪽은 마지막 user 메시지 뒤에 assistant 시작 토큰을 붙였고 다른 쪽은 붙이지 않았을 수 있다. 도구 schema의 key 순서나 reasoning 설정이 달라졌을 수도 있다.

이 차이가 사소하지 않은 이유는 transformer가 문자열의 “뜻”을 먼저 해석한 뒤 토큰을 만드는 시스템이 아니기 때문이다. tokenizer가 만든 각 token ID는 embedding 행을 고르고, 이후 모든 layer의 attention은 앞선 위치들의 K와 V를 읽는다. prefix의 첫 role delimiter가 달라지면 그 위치의 embedding부터 달라지고, 뒤의 모든 hidden state가 다른 조건에서 계산된다. 마지막 user 본문이 똑같아도 `P(next token | rendered prefix)`의 조건부 사건 자체가 달라진다.

작은 예를 보자. API 입력은 다음처럼 동일하다고 하자.

```json
{
  "messages": [
    {"role": "system", "content": "답은 한 문장으로 써라."},
    {"role": "user", "content": "달은 왜 밝아 보이나?"}
  ]
}
```

템플릿 A가 다음 문자열을 만든다고 하자.

```text
<|im_start|>system
답은 한 문장으로 써라.<|im_end|>
<|im_start|>user
달은 왜 밝아 보이나?<|im_end|>
<|im_start|>assistant
```

템플릿 B는 role 이름을 평문으로 쓰고 assistant 접두사를 생략할 수 있다.

```text
System: 답은 한 문장으로 써라.
User: 달은 왜 밝아 보이나?
```

두 문자열은 사람이 읽으면 같은 대화처럼 보인다. 모델 관점에서는 다르다. 제어 토큰의 존재, role 구간의 길이, 질문의 position, 마지막 suffix가 모두 다르다. A 형식으로 instruction tuning된 모델에 B를 넣으면 “질문의 의미”가 보존되었다는 보장이 없다. 이때 두 서버의 첫 logits가 갈라지는 것은 sampling 재현성 문제가 아니라 입력 identity 문제다.

### prompt identity를 네 층으로 나눈다

문제를 좁힐 때 `prompt`라는 단어 하나를 쓰면 서로 다른 상태를 섞게 된다. 적어도 네 층을 구별해야 한다.

첫째는 **요청 의미**다. role, content, tools, response format, reasoning 관련 field를 가진 구조화 객체다. 둘째는 **렌더링 문자열**이다. 템플릿이 delimiter와 schema, generation suffix를 넣은 결과다. 셋째는 **token identity**다. tokenizer revision과 special-token 정책을 거친 정수 배열이다. 넷째는 **실행 입력**이다. truncation, padding, multimodal feature splice를 거쳐 model runner가 받는 IDs, positions, masks다.

같은 요청 의미가 같은 렌더링 문자열을 보장하지 않는다. 같은 렌더링 문자열도 tokenizer가 다르면 같은 IDs를 보장하지 않는다. 같은 IDs도 truncation과 multimodal merge가 다르면 같은 실행 입력을 보장하지 않는다. 따라서 “두 서버에 같은 prompt를 보냈다”는 문장은 어느 층을 비교했다는 것인지 없으면 증거가 아니다.

이 구분은 비용 설명에도 필요하다. tools를 추가했더니 모델이 실제로 tool을 호출하지 않았다고 해도 schema가 prompt에 직렬화되었다면 prompt token 수와 prefill 계산, KV 용량은 이미 늘었다. gateway가 동적 timestamp나 request ID를 system prompt에 삽입했다면 공통 prefix cache가 깨질 수 있다. 렌더링 문자열이 달라진 효과가 GPU에서 나타나지만 원인은 템플릿 앞단에 있다.

### 최초 불일치가 가장 값싼 증거다

두 서버의 답이 다를 때 최종 text만 비교하면 가능한 원인이 너무 많다. renderer, tokenizer, scheduler batch, backend, floating-point reduction, sampling 순서가 모두 후보가 된다. 반면 다음 산출물을 앞에서부터 비교하면 최초 불일치를 빠르게 찾을 수 있다.

```text
messages의 canonical JSON
→ 선택된 template의 digest와 이름
→ rendered UTF-8 bytes
→ input token IDs와 special-token 표지
→ truncation 뒤 token IDs
→ model input positions와 multimodal binding
→ 첫 raw logits
→ processed scores와 selected token
```

rendered bytes에서 이미 갈라졌다면 CUDA profiler를 열 이유가 없다. rendered bytes는 같은데 IDs가 다르면 tokenizer bundle을 본다. IDs까지 같은데 첫 logits가 다르면 그때 model artifact, position/mask, backend를 조사한다. 이 순서는 하위 층을 무시하는 것이 아니라 원인 공간을 가장 싼 증거로 자르는 방법이다.

반례도 있다. 렌더링 문자열이 같아도 tokenizer가 special token을 일반 문자 조각으로 처리하면 의미가 달라진다. 문자열 diff가 0이라는 이유로 템플릿을 무죄로 판정해서는 안 된다. 템플릿이 출력한 delimiter가 tokenizer vocabulary에서 원자적인 special ID인지까지 이어서 확인해야 한다. 이 경계는 다음 장에서 special token과 padding을 다룰 때 더 깊게 본다.

**메시지에서 token ID까지 실제 경계를 걷는다.**

템플릿 디버깅이 자주 막히는 이유는 사람이 보는 문자열과 모델이 받는 정수열 사이를 한 번에 건너뛰기 때문이다. `apply_chat_template(..., tokenize=True)` 같은 편리한 API는 선택, Jinja 렌더링, special token 정책, tokenizer 호출을 한 결과로 접는다. 정상 경로에서는 이것이 옳은 추상화다. 그러나 회귀 조사에서는 중간 산출물이 사라져 최초 분기를 찾기 어렵다. 따라서 동일 입력을 적어도 네 산출물로 펼친다.

1. 선택된 template의 원문과 digest
2. Jinja가 만든 rendered text의 UTF-8 byte
3. tokenizer에 넘기기 직전 special-token 정책과 실제 입력 문자열
4. 최종 token ID, attention mask, generation boundary

이 네 줄은 로그 장식이 아니라 컴파일 trace다. 첫 줄이 다르면 artifact resolution 문제이고, 둘째 줄이 다르면 template context나 Jinja 실행 문제다. 둘째 줄은 같은데 셋째 줄이 다르면 BOS/EOS 같은 special token ownership이 갈린 것이다. 셋째 줄까지 같은데 ID가 다르면 tokenizer artifact나 normalizer/backend 문제다. 원인을 이 순서로 좁히면 “템플릿 문제 같다”는 막연한 결론이 수정 가능한 함수 경계로 바뀐다.

**template selection과 Jinja context를 먼저 고정한다**

같은 template 문자열이라도 context가 다르면 다른 프로그램이다. `messages` 외에 `tools`, `documents`, `add_generation_prompt`, `continue_final_message`, 날짜, 모델별 custom variable이 분기를 바꾼다. 그러므로 digest를 template text에만 매기면 부족하다. 실행 identity는 template bytes와 context schema, 실제 분기 입력을 함께 설명해야 한다.

예를 들어 template가 `tools`의 존재 여부로 system block을 추가한다고 하자. 도구가 없는 요청 A와 도구가 하나인 요청 B의 user text가 같아도 rendered prompt는 다르다. 더 미묘한 경우는 도구 목록이 비어 있지만 `tools=[]`가 전달된 요청과 `tools` 자체가 생략된 요청이다. Jinja의 undefined 처리와 truthiness에 따라 같은 결과가 나올 수도, 다른 marker가 나올 수도 있다. cache key가 둘을 무조건 같다고 가정해서는 안 된다. 먼저 실제 render 결과가 같은지 확인하고, 향후 template 변경에도 동일성이 보존되는지를 별도 정책으로 결정한다.

고정 소스를 읽을 때는 public API의 인자 목록에서 멈추지 않는다. template 선택 함수가 tokenizer config의 어느 필드를 읽는지, named template와 default template를 어떻게 구분하는지, Jinja environment에 어떤 filter와 global을 주입하는지, 예외를 어느 API 오류로 바꾸는지까지 caller에서 consumer 방향으로 걷는다. 이때 중요한 것은 함수 이름 암기가 아니라 값의 소유권이다. 누가 template를 골랐는가, 누가 context를 만들었는가, 누가 rendered result를 tokenization 대상으로 선언했는가를 적는다.

**렌더링 결과는 문자보다 byte로 비교한다**

화면에서 같은 문자열처럼 보여도 byte가 다를 수 있다. 줄바꿈이 LF인지 CRLF인지, 결합 문자가 NFC인지 NFD인지, 눈에 보이지 않는 공백이 일반 space인지 non-breaking space인지에 따라 tokenizer 결과와 cache key가 갈린다. 따라서 incident fixture에는 pretty-printed prompt만 두지 않는다. UTF-8 길이, byte digest, escape된 suffix/prefix, 가능하면 code point 열을 함께 둔다.

가령 두 결과가 모두 `assistant:`로 끝나는 것처럼 보이지만 하나는 뒤에 공백 하나, 다른 하나는 줄바꿈 하나를 붙였다고 하자. BPE vocabulary에 `assistant:`과 뒤 공백을 포함한 merge가 있다면 차이는 마지막 token 하나에 그치지 않는다. suffix의 여러 ID가 바뀌고 prefix cache에서 공유할 수 있는 block 경계도 앞당겨질 수 있다. “문자 하나 차이”가 scheduler가 보는 input length와 KV block 수를 바꾸는 이유다.

비교 절차는 단순하다. rendered byte의 최초 불일치 index를 찾고, 그 주변 32 byte를 escape해 저장한다. 그 다음 두 token 열의 longest common prefix를 구한다. byte 분기와 token 분기가 같은 논리 구간인지 확인한다. byte는 앞에서 달라졌는데 token ID는 한동안 같을 수 있고, 반대로 정규화 때문에 여러 byte 차이가 하나의 token 경계에서 드러날 수도 있다. 두 좌표를 혼합하지 않는 것이 핵심이다.

**special token insertion의 owner를 하나로 만든다**

템플릿이 BOS 문자열을 직접 출력하면서 tokenizer 호출에도 `add_special_tokens=True`를 남기면 BOS가 두 번 들어갈 수 있다. 반대로 template가 BOS를 출력하지 않는데 serving wrapper가 이미 처리했다고 가정해 `False`로 호출하면 BOS가 사라진다. 이 문제는 special token이 “있다/없다”보다 누가 삽입 책임을 갖는지가 불명확해서 생긴다.

검산표에는 `template_emits_bos`, `tokenizer_add_special_tokens`, `observed_prefix_ids` 세 열을 둔다. 기대 BOS가 한 번이라면 `(true,false)`와 `(false,true)`는 후보가 될 수 있지만 `(true,true)`와 `(false,false)`는 즉시 조사 대상이다. 다만 문자열로 출력된 marker가 반드시 special ID 하나가 된다고 가정해서는 안 된다. AddedToken 등록, normalization, tokenizer backend에 따라 marker가 일반 subword 여러 개가 될 수도 있다. 최종 판정은 문자열 존재가 아니라 ID와 token metadata로 한다.

EOS도 같은 표로 다루되 generation 시작점과 종료 정책을 섞지 않는다. 입력 끝의 EOS, assistant header 뒤의 generation prompt, decoder가 생성 중 만나는 stop ID는 서로 다른 생명주기를 갖는다. 한 ID가 여러 역할을 겸할 수 있어도 어느 단계에서 삽입되고 어느 단계에서 소비되는지는 별도로 기록해야 한다.

**rendered text에서 token ID까지 손으로 검산한다**

작은 fixture를 하나 정한다. system 한 개, user 한 개, 빈 assistant 시작 marker로 이루어진 요청이다. 단계별로 byte 길이와 token 수를 기록한다. 예를 들어 rendered text가 184 byte이고 tokenizer 결과가 47 token이며 BOS가 별도 한 개라면 engine input은 48 token이다. 서버 A가 47, 서버 B가 48을 보고했다면 평균 token 길이 같은 통계로 덮지 말고 첫 ID부터 비교한다.

두 열의 longest common prefix가 0이고 첫 ID만 BOS 차이라면 owner 가설이 강하다. 공통 prefix가 31에서 끝나고 그 위치가 user content 내부라면 template보다 normalization/tokenizer artifact를 먼저 의심한다. 공통 prefix가 46이고 마지막 header에서 갈리면 `add_generation_prompt`나 trailing whitespace가 후보가 된다. 이 계산은 간단하지만 조사 순서를 결정한다는 점에서 중요하다.

```
template_sha256 = sha256(template_utf8)
render_sha256   = sha256(rendered_utf8)
token_sha256    = sha256(pack_uint32_le(input_ids))
lcp             = longest_common_prefix(ids_a, ids_b)
```

정수열 digest에는 dtype과 byte order를 명시한다. JSON 문자열 `[1, 23]`의 digest는 serializer 공백과 숫자 표현에 영향을 받는다. canonical binary 표현을 쓰거나 canonical JSON 규칙을 고정한다. digest는 증거를 대체하지 않는다. 원본을 찾기 위한 index이고, 충돌이나 직렬화 변경을 판정하려면 length와 artifact version도 함께 보존해야 한다.

**서버별 convenience API를 같은 단계에 맞춰 비교한다**

Transformers의 reference API, vLLM renderer, SGLang OpenAI serving, llama.cpp의 chat template 경로는 추상화 경계가 같지 않다. 한쪽 함수가 render와 tokenize를 함께 수행한다고 해서 다른 쪽의 render 함수와 곧바로 비교하면 안 된다. 각 구현에서 template selection, context assembly, render, special-token policy, encode, engine ingress의 여섯 pass를 찾아 같은 행에 놓는다.

비어 있는 행도 결과다. 예를 들어 어떤 경로가 assistant mask를 만들지 않거나 offset을 노출하지 않는다면 `없음`과 `관찰 불가`를 구분한다. 전자는 구현 계약이고 후자는 현재 계측의 한계다. 이 구분 없이 빈칸을 같은 값으로 취급하면 parity가 거짓으로 통과한다.

승인 조건은 최종 답변 문장이 비슷하다는 것이 아니다. 선택된 template digest, rendered byte digest, special-token owner, input ID digest, generation boundary가 기대값과 일치해야 한다. sampling 이전의 입력 identity를 닫은 뒤에야 출력 차이를 logits와 sampling 문제로 넘길 수 있다.

## 7.2 role·tool·reasoning은 어떻게 한 줄의 토큰열이 되는가

role serialization의 목적은 사람이 보기 좋은 대화록을 만드는 것이 아니다. 학습 때 모델이 보았던 경계 표지를 재현하여 “누가 말했고 다음에 누가 말해야 하는가”를 token context로 표현하는 것이다. system, user, assistant, tool은 API metadata에서 token protocol로 lowering된다. 이때 delimiter, 줄바꿈, block 순서가 모두 의미를 가진다.

role delimiter를 봉투의 발신자 표시에 비유할 수 있다. 내용이 같아도 발신자가 법원인지 친구인지에 따라 수신자가 다르게 해석한다. 그러나 모델에서 delimiter는 사람이 이해하는 label이 아니라 학습된 token embedding이다. delimiter 문자열이 vocabulary에서 여러 일반 token으로 쪼개지거나 학습 때와 다른 순서라면 봉투 비유는 깨진다.

### system과 user 경계는 모든 뒤 위치에 영향을 준다

system instruction은 보통 prefix의 앞쪽에 놓인다. 앞쪽 token은 causal attention 때문에 뒤의 모든 위치가 볼 수 있다. system block의 종료 delimiter 하나가 빠지면 뒤의 user content가 system 안에 계속 있는 것처럼 직렬화될 수 있다. 반대로 user content 안에 delimiter와 같은 문자열이 들어오고 tokenizer가 이를 special token으로 받아들이면 role 경계가 조기 종료될 가능성을 검토해야 한다.

이 문제를 prompt injection과 혼동하지 않는 것도 중요하다. template delimiter가 있다고 해서 사용자 지시가 system 지시를 절대 덮지 못하는 것은 아니다. delimiter는 구조를 표현할 뿐 모델의 지시 우선순위를 강제하는 접근 제어 장치가 아니다. 반대로 escaping을 무조건 HTML 방식으로 적용하면 모델이 학습하지 않은 문자열을 보게 된다. 보안 정책은 API validation, content escaping, special-token 허용 정책, model behavior를 나누어 설계해야 한다.

role sequence validation은 오류를 앞당긴다. system이 중간에 나타나거나 assistant가 두 번 연속되고, tool result가 대응 call 없이 들어오는 요청을 그대로 렌더링하면 모델이 이해할 수 없는 protocol을 만든다. 다만 모든 model template가 user/assistant의 엄격한 교대를 요구하는 것은 아니다. 여러 tool result를 연속으로 받는 형식도 있다. validator는 일반 도덕률이 아니라 선택된 template의 문법을 따라야 한다.

### tools는 prompt 앞에 숨어 있는 큰 프로그램이다

tool calling 요청에서 사용자가 실제로 선택한 tool만 비용을 만든다고 생각하기 쉽다. 그러나 많은 template는 사용 가능한 tool 전체의 이름, 설명, JSON schema와 호출 문법을 system 영역에 넣는다. tool이 스무 개이고 각 schema가 길면 사용자 질문보다 tool 명세가 훨씬 큰 prefix가 된다. 이것은 prefill token, KV byte, TTFT, position을 모두 바꾼다.

도구 직렬화는 세 identity를 동시에 만든다. 첫째, model input identity다. schema key 순서, 공백, description 문구가 token IDs를 바꾼다. 둘째, cache identity다. 시각적으로 같은 JSON이라도 canonicalization이 다르면 공통 prefix를 공유하지 못한다. 셋째, security identity다. tenant마다 허용된 tool 집합이 다른데 key에서 tool set을 누락하면 다른 tenant의 prefix를 잘못 재사용할 수 있다.

따라서 tool schema를 안정적으로 정규화할 필요가 있다. 하지만 모든 dictionary를 key 정렬하면 된다는 결론도 성급하다. JSON Schema에서 배열 순서가 의미를 가질 수 있고, template가 받은 order 자체를 모델 학습 형식으로 사용했을 수 있다. canonicalization은 의미 보존 규칙을 명시해야 한다. schema를 바꾼 gateway와 model repository의 template 중 누가 최종 직렬화 owner인지도 하나로 정해야 한다.

실제 문제 장면을 따라가 보자. tools를 붙였더니 답이 나빠졌지만 tool은 호출되지 않았다. 첫 가설은 model이 tool mode에서 품질이 낮다는 것이다. 경쟁 가설은 tool schema가 질문을 뒤로 밀어 context position과 prefill 부담을 바꾼 것, server가 다른 chat template variant를 선택한 것, reasoning flag가 tools와 함께 달라진 것이다. rendered bytes와 selected template name, prompt token count를 비교하면 이 가설들을 GPU 실행 전에 가를 수 있다.

### reasoning은 숨겨진 출력이 아니라 다음 입력의 일부가 될 수 있다

reasoning을 사용자에게 보여 주지 않는 정책과 다음 turn의 prompt에서 reasoning을 제거하는 정책은 같은 말이 아니다. 서버 A는 과거 assistant의 visible content만 messages에 넣고, 서버 B는 `reasoning_content`를 별도 field로 보존하여 template가 다시 직렬화할 수 있다. 사용자가 보는 대화는 같아도 다음 model input은 달라진다.

reasoning serialization에는 적어도 세 상태가 있다. reasoning이 아직 시작되지 않은 assistant prefix, reasoning이 열려 생성 중인 상태, reasoning이 닫히고 visible answer가 이어지는 상태다. template의 `enable_thinking`이나 `reasoning_effort`가 suffix와 과거 turn 직렬화를 바꾸면 first model token과 first visible token의 의미도 달라진다. TTFT를 어느 사건까지 재는지에도 영향을 준다.

vLLM과 SGLang 같은 서버는 request별 chat-template kwargs를 전달한다. 이 kwargs는 harmless metadata가 아니라 compiler flag다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- SGLang 고정 소스의 OpenAI protocol은 request의 [`chat_template_kwargs`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/protocol.py#L811-L844)를 모델링하고, serving 경로는 default와 request 값을 합쳐 rendering에 사용한다.
- [`OpenAIServingChat`의 template kwargs 처리](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_chat.py#L1048-L1060)를 보면 default가 request에 채워지는 경계를 찾을 수 있다.
- 같은 JSON messages라도 이 dict가 다르면 같은 prompt가 아니다.

reasoning을 cache와 billing에 연결할 때는 raw request, rendered prompt, generated model token, visible token을 따로 센다. 과거 reasoning이 prompt에 다시 들어가면 input token과 cache key에 포함된다. 현재 turn reasoning을 숨기더라도 model은 token을 생성했고 KV도 늘었다. visible output만 세면 compute와 billing 정책의 차이를 설명할 수 없다.

### 멀티모달 content는 placeholder와 feature의 결합 계약이다

이미지를 template에 넣는다는 말은 이미지 byte를 Jinja 문자열 안에 붙인다는 뜻이 아니다. 보통 template는 image 위치를 나타내는 special placeholder를 텍스트 열에 놓고, processor가 별도로 읽은 pixel tensor나 vision feature를 그 위치에 결합한다. 따라서 renderer output에는 실제 이미지 내용이 아니라 **binding site**가 생긴다.

이 경계에서 세 순서가 같아야 한다. messages에서 image가 등장한 순서, 렌더링된 placeholder 순서, processor batch의 image tensor 순서다. 두 이미지가 있고 placeholder가 뒤바뀌면 tensor shape는 정상이어도 첫 질문이 두 번째 이미지 feature를 볼 수 있다. correctness bug가 illegal memory access가 아니라 그럴듯한 오답으로 나타나는 이유다.

또한 placeholder 하나가 token ID 하나라는 보장은 없다. template 출력 문자열, tokenizer special-token 설정, processor의 expansion 규칙을 함께 보아야 한다. vision encoder가 만든 여러 feature token으로 placeholder를 확장하면 최종 sequence length와 positions, attention mask가 바뀐다. truncation이 placeholder marker의 일부를 자르거나 text token만 줄이고 image feature는 그대로 두면 count invariant가 깨진다.

Transformers의 multimodal 진입점은 tokenizer API와 별도다. 고정 소스의 [`ProcessorMixin.apply_chat_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/processing_utils.py#L1976-L2235)는 conversation에서 media를 읽고 template kwargs와 tokenize kwargs를 나눈 뒤 processor 호출로 이어진다. 이 함수가 tokenizer의 같은 이름 method와 동일하다고 가정하면 media loading과 feature-related output을 놓친다.

다음 절에서는 이 role/tool/reasoning/multimodal program의 끝에 붙는 generation prompt와 assistant mask를 본다. 입력 직렬화가 맞더라도 “어디까지가 prompt이고 어디서부터 model이 이어 써야 하는가”를 잘못 표시하면 조건부 분포와 회계가 다시 갈라진다.

## 7.3 템플릿을 작은 컴파일러로 읽으면 무엇이 보이는가

채팅 템플릿을 `format(messages)`라고 생각하면 입력 검증과 상태 전이를 놓친다. 실제 템플릿은 role 순서를 검사하고, content 타입에 따라 분기하고, tool schema를 다른 위치에 삽입하며, 과거 assistant reasoning을 다시 직렬화하고, 마지막에 generation suffix를 붙인다. Jinja loop와 condition은 출력 문자열을 만드는 프로그램이다.

컴파일러 관점에서는 다음 대응이 생긴다. messages와 tools는 소스 프로그램, role/content schema는 입력 문법, Jinja template와 template kwargs는 compiler 및 compiler flag, 렌더링 문자열은 중간 표현, tokenizer는 다음 lowering pass, token IDs는 model이 소비하는 기계어에 가깝다. `add_generation_prompt`, `continue_final_message`, `enable_thinking`은 단순 장식 옵션이 아니라 출력 문법을 바꾸는 compiler flag다.

이 비유가 주는 가장 큰 이점은 **artifact와 flag를 버전 관리해야 한다**는 사실이다. model weight만 고정하고 chat template를 floating `main`에서 받아오면 입력 언어의 compiler가 바뀐 셈이다. tokenizer revision과 template digest, server-side override, request별 template kwargs를 함께 보존해야 같은 모델 입력을 재현할 수 있다.

하지만 compiler 비유에는 세 가지 한계가 있다. 첫째, 출력 형식의 정답이 형식 명세 하나에 완전히 적혀 있지 않다. 학습 시 사용한 대화 serialization이 사실상의 ABI다. 둘째, template가 만들어 낸 문자열이 tokenizer의 normalization과 special-token 정책을 거치므로 한 pass만 독립적으로 검증할 수 없다. 셋째, API server가 template 전후에 content normalization, multimodal placeholder expansion, reasoning parser 같은 별도 pass를 삽입할 수 있다. 따라서 실제 pipeline을 소스에서 끝까지 따라야 한다.

### 입력 문법: role보다 content 타입이 먼저 깨질 수 있다

OpenAI 계열 messages에서 content는 언제나 문자열이라고 가정하기 쉽다. 멀티모달 요청에서는 text, image, audio 같은 항목의 배열일 수 있다. tool 응답은 tool-call ID와 content를 추가로 가진다. 서버가 이 객체를 평문으로 바꾸어 template에 넘기는지, template가 배열을 직접 순회하는지에 따라 지원 범위와 오류 위치가 달라진다.

잘 설계된 입력 검증은 예상하지 않은 타입을 조용히 문자열화하지 않는다. Python dictionary가 그대로 `{'type': ...}` 형태로 출력되면 Jinja 렌더링은 성공하지만 모델 protocol은 깨진다. 반대로 template가 엄격하게 예외를 내더라도 서버가 이를 500으로 바꾸면 사용자는 model 장애로 오인한다. API schema가 허용한 content와 renderer가 실제로 처리하는 content의 교집합이 serving contract다.

role 순서도 단순 loop 이상의 문법이다. system 메시지를 첫 위치에만 허용하는 template가 있고, 연속된 tool response를 하나의 user block으로 묶는 template가 있으며, 과거 assistant message 안에서 reasoning과 visible content를 나누는 template가 있다. 빈 messages, 마지막 role이 assistant인 prefill, tool call 뒤 tool result 누락은 각각 다른 문법 상태다.

여기서 좋은 진단 질문은 “Jinja가 렌더링했는가?”가 아니다. “어느 입력 타입과 role sequence가 어떤 출력 block으로 lowering되었는가?”다. 이를 답하려면 messages의 각 원소에 source index를 붙이고, 렌더링된 byte range와 token range로 이어지는 source map을 만들어야 한다. compiler가 source location을 보존하듯 prompt compiler도 최소한 장애 조사에서 이 대응을 복원할 수 있어야 한다.

### template selection은 artifact resolution이다

서버에는 template 출처가 여러 개일 수 있다. tokenizer config에 포함된 기본 template, model repository의 `chat_template.jinja`, command-line override, server builtin registry, request에서 허용한 template가 서로 경쟁할 수 있다. 우선순위가 명확하지 않으면 같은 image digest로 배포해도 request flag나 model metadata에 따라 다른 compiler를 쓴다.

우선순위는 제품 이름만으로 외우지 않고 현재 revision의 resolver가 실제로 검사한 순서로 남긴다. 다음 표의 `후보 값`에는 설정 문서의 예상값이 아니라 resolver 진입 시 읽은 값을, `선택 결과`에는 끝내 열린 파일·builtin key·tokenizer field를 쓴다. 사용하지 않은 후보도 `없음`, `거부`, `후순위`로 구별해야 같은 기본값이라는 말에 숨은 차이를 찾을 수 있다.

| 판정 순서 | 후보 값 | 선택 결과에 남길 필드 |
|---|---|---|
| request override | 요청 template 값과 허용 정책 | `request_template`, trust 판정, 거부 이유 |
| server override | CLI/config literal·path·builtin name | `source_kind`, canonical file path 또는 builtin key, content digest |
| tokenizer/model artifact | named/default `chat_template`와 repository file | tokenizer revision, 실제 field/name, file member, content digest |
| compatibility fallback | builtin·legacy renderer와 활성 조건 | effective renderer, fallback reason, Jinja/legacy mode |

이 표는 모든 구현이 위 순서를 공유한다는 주장이 아니다. 각 서버의 고정 source에서 존재하지 않는 행은 `not supported`로 닫고, 순서가 다르면 실제 resolver 순서로 행을 바꾼다. 핵심은 요청자가 지정한 이름과 최종 template bytes 사이에 선택 기록을 남기는 것이다.

vLLM의 고정 소스에서는 template validation과 loading이 분리되어 있다. [`validate_chat_template`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/chat_utils.py#L1315-L1343)는 값의 타입과 path/builtin 가능성을 검사하고, [`_load_chat_template`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/chat_utils.py#L1346-L1394)는 literal, file, builtin name을 실제 template 문자열로 해석한다. 검증 함수 이름만 보고 request별 template가 안전하다고 단정해서는 안 된다.

`trust_request_chat_template` 같은 정책과 어떤 호출자가 어떤 값을 넘기는지를 함께 읽어야 한다.

artifact resolution을 운영 장부로 바꾸면 model ID만 적어서는 부족하다는 결론이 나온다. 최소한 model revision, tokenizer revision, template source kind, template SHA-256, server default kwargs, request override kwargs를 기록해야 한다. template가 tools 존재 여부에 따라 여러 variant 중 하나를 고르면 선택된 variant 이름도 필요하다. 이 장부는 재현성뿐 아니라 prefix cache namespace와 billing audit에도 쓰인다.

### 렌더링과 tokenization을 한 호출로 묶을 때 잃는 것

편의 API는 messages를 받아 바로 token IDs를 반환할 수 있다. production에는 유용하지만 장애 조사에서는 중간 산출물인 rendered bytes가 사라진다. 반대로 `tokenize=False`로 얻은 문자열만 비교하면 tokenizer special-token 처리와 truncation을 놓친다. 따라서 정상 관측은 둘 중 하나를 고르는 것이 아니라 두 경계를 모두 보존한다.

Transformers 고정 소스의 [`PreTrainedTokenizerBase.apply_chat_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L2989-L3223)는 template 선택, Jinja rendering, tokenize 여부, padding/truncation과 generation mask 반환을 한 API에 모은다.

실제 Jinja environment와 rendering helper는 [`_compile_jinja_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/utils/chat_template_utils.py#L420-L496)와 [`render_jinja_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/utils/chat_template_utils.py#L498-L590)에 있다. API 한 줄이 짧다고 실행 의미도 짧은 것은 아니다.

장애를 좁힐 때는 동일 messages와 kwargs로 렌더링 문자열과 IDs를 각각 기록하고, rendered text를 다시 tokenizer에 넣은 결과가 direct `tokenize=True` 결과와 같은지 본다. 다르면 special token 추가 정책이나 API 내부 옵션을 의심한다. 단, 이 비교를 production 호출 두 번으로 수행하면 mutable counter나 dynamic field가 template에 들어가는 경우 결과가 달라질 수 있다. 같은 frozen input과 artifact를 사용해야 한다.

## 7.4 generation prompt와 assistant mask는 경계의 두 표현이다

대화가 마지막 user 메시지에서 끝났을 때 사람은 자연스럽게 assistant가 다음에 말하리라고 안다. 모델에는 이 순서 감각을 token context로 알려 주어야 한다. 많은 template에서 `add_generation_prompt=True`는 assistant role의 시작 delimiter나 reasoning 시작 marker를 suffix에 붙인다. 이 suffix는 빈 장식이 아니라 다음 token이 어느 역할과 구간에 속하는지를 조건으로 제공한다.

문제 장면부터 보자. 동일 model과 greedy decoding을 사용했는데 offline Transformers에서는 정상적으로 답하고 server endpoint에서는 user 메시지를 흉내 내거나 빈 문자열을 반환한다. rendered prompt를 비교하니 offline 쪽에는 assistant prefix가 있고 server 쪽에는 없다. 이때 “model이 불안정하다”는 가설은 첫 logits 이전에 반박된다. 두 실행은 애초에 다른 조건부 분포를 계산했다.

### 이어 쓰기와 새 답변 시작은 같은 연산이 아니다

마지막 메시지가 user이면 일반적인 chat completion은 새 assistant block을 시작한다. 마지막 메시지가 assistant이고 그 내용을 이어 쓰려는 prefill 요청이면 이미 열린 assistant block을 계속해야 한다. 전자는 generation prompt, 후자는 `continue_final_message`와 같은 의미가 필요하다. 둘을 동시에 쓰면 assistant prefix를 중복하거나 이미 닫힌 block을 이어 쓰는 모순이 생긴다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- Transformers 고정 소스의 `apply_chat_template`는 이 두 선택을 별도 인자로 받고 함께 참일 때의 오류를 처리한다.
- [`apply_chat_template`의 인자·검증 구간](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L2989-L3075)과 [rendering·continue 처리 구간](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L3076-L3165)을 나누어 읽으면 API flag가 실제 Jinja 변수와 rendered suffix로 이어지는 방식을 볼 수 있다.
- 정확한 line 안의 구현은 revision에 종속되므로 다른 버전에서는 symbol과 조건을 다시 찾아야 한다.

새 답변과 이어 쓰기를 우편 문서에 비유하면, generation prompt는 새 문서의 발신자 칸을 미리 채워 놓는 일이고 continue는 이미 쓰다 만 문장의 마지막 위치에서 펜을 다시 대는 일이다. 그러나 이 비유는 reasoning과 tool call 같은 중첩 문법을 숨긴다. assistant block 안에서도 reasoning 구간, visible content, tool-call 구간의 open/closed 상태가 따로 있을 수 있다. 단순히 마지막 문자열을 잘라 “이어 쓰기”를 구현하면 닫는 delimiter나 multi-byte 문자 경계를 잘못 찾을 수 있다.

따라서 prefill 계약은 마지막 role만으로 결정하지 않는다. 마지막 message의 content 타입, template가 block을 닫았는지, tool call이 미완성인지, reasoning marker가 열린 상태인지, server가 visible suffix를 어느 field에서 가져오는지를 기록한다. 이어 쓰기가 지원되지 않는 template에 flag를 전달했을 때 조용히 무시하는지 예외를 내는지도 API contract다.

### assistant mask는 byte 길이로 만들 수 없다

assistant mask는 token sequence의 각 위치가 assistant가 생성한 구간에 속하는지를 나타낸다. 학습 데이터 준비나 평가에서 loss를 어느 token에 적용할지 결정할 수 있고, 분석에서는 prompt token과 assistant token의 경계를 복원하는 데 유용하다. 중요한 점은 이 mask가 “렌더링 문자열의 마지막 30%” 같은 길이 규칙이 아니라 template가 표시한 generation block과 tokenizer offset mapping의 합성 결과라는 것이다.

Jinja extension은 `{% generation %}`와 `{% endgeneration %}` 사이의 문자 범위를 기록할 수 있다. renderer가 이 범위를 반환하면 tokenizer는 문자 위치를 token 위치에 대응시켜 assistant mask를 만든다. 다바이트 UTF-8, normalization, byte fallback, special token이 개입하면 문자 index와 byte index, token index는 서로 다르다. `len(rendered_text)`에서 suffix 길이를 빼는 방식은 한국어와 emoji, 결합 문자에서 쉽게 틀린다.

Transformers의 Jinja compile 경계는 [`_compile_jinja_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/utils/chat_template_utils.py#L420-L496)다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- 이곳에서 sandboxed environment와 generation tracking extension이 준비된다.
- rendering helper는 [`render_jinja_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/utils/chat_template_utils.py#L498-L590)에서 conversations를 순회하고 rendered chat와 generation indices를 만든다.
- tokenizer method의 후반부는 char-to-token mapping을 사용해 mask를 token 배열에 맞춘다.
- source walk의 핵심은 extension 이름을 외우는 것이 아니라 generation range가 어느 representation의 좌표이고, 다음
- pass에서 어떤 좌표로 변환되는지 보는 것이다.

assistant mask가 비어 있다고 곧바로 tokenizer bug라고 결론 내리면 안 된다. template가 generation block marker를 사용하지 않았을 수 있다. slow tokenizer나 특정 backend가 필요한 offset mapping을 제공하지 않을 수도 있다. padding과 batch가 mask shape를 바꿀 수도 있다. 경쟁 가설을 구분하려면 template source에 generation tag가 있는지, renderer가 character span을 반환했는지, tokenization 결과가 char-to-token mapping을 제공하는지 차례로 본다.

### generation 경계는 usage와 streaming의 기준점이다

서빙에서 assistant mask 자체를 반환하지 않더라도 같은 경계 개념이 필요하다. usage는 input token과 output token을 나눠야 하고, prefix cache는 어느 token까지 요청 입력인지 알아야 하며, streaming은 model이 새로 생성한 token만 보내야 한다. 이미 prompt에 포함된 assistant prefill token을 completion으로 다시 세거나 전송하면 회계와 사용자 경험이 모두 틀어진다.

다음과 같은 token ledger를 생각해 보자.

```text
rendered prompt tokens: 812
그중 과거 assistant tokens: 290
현재 assistant prefill suffix: 6
이번 실행에서 새로 accepted된 model tokens: 41
stop 처리 뒤 visible tokens: 37
```

API input usage는 대개 812를 기준으로 하지만 product 정책에 따라 cached/computed prompt token을 별도로 나눌 수 있다. output usage는 새로 accepted된 41과 visible 37 중 어느 것을 노출하는지 명시해야 한다. reasoning token을 숨기거나 stop marker를 제외하면 둘은 달라진다. 과거 assistant 290은 현재 output이 아니다. role이라는 말만으로는 이 회계를 설명할 수 없고, compile/commit 경계를 함께 기록해야 한다.

generation suffix가 달라지면 prefix cache identity도 달라진다. 같은 user messages라도 `enable_thinking=true`가 열린 reasoning marker를 붙이고 false가 닫힌 빈 reasoning block을 붙인다면 token tail이 다르다. 이 두 요청을 같은 cache key로 묶으면 잘못된 KV를 재사용할 수 있다. key를 raw JSON messages에만 묶는 설계가 위험한 이유다.

### 잘못된 generation prompt를 어떻게 반증하는가

대표 증상은 첫 token부터 role marker가 나오거나, 답변이 빈 채 EOS가 선택되거나, thinking on/off에서 first-visible latency가 예상 밖으로 바뀌는 것이다. 첫 probe는 최종 text가 아니라 rendered suffix의 마지막 수십 token ID와 첫 raw logits다. suffix가 다르면 backend 가설을 보류한다.

같은 rendered IDs를 강제했을 때 두 경로의 첫 logits가 같아진다면 template/suffix 가설이 지지된다. 반대로 rendered IDs가 같은데도 divergence가 남으면 positions, attention mask, model artifact로 내려간다. `add_generation_prompt`를 켰더니 문제가 사라졌다는 사실만으로 원인을 닫지 않는다. 이미 assistant prefix가 있는 요청에서 중복될 수 있으므로 user-last, assistant-last prefill, tool-result-last, empty conversation을 각각 검증해야 한다.

## 7.5 truncation은 길이를 줄이는 일이 아니라 문법을 다시 쓰는 일이다

긴 tool schema와 대화 기록을 가진 요청이 model context를 넘었다고 하자. 서버가 가장 오래된 token부터 잘라 길이를 맞추면 OOM은 피할 수 있다. 그러나 남은 sequence가 유효한 chat protocol이라는 보장은 없다. system block의 시작은 잘리고 끝 delimiter만 남을 수 있고, tool schema의 JSON이 중간에서 끊길 수 있으며, image placeholder와 feature 수가 달라질 수 있다.

문서를 페이지 수에 맞춰 앞에서 가위로 자르는 비유가 직관을 준다. 문서 분량은 줄지만 첫 장의 제목과 정의가 사라져 나머지를 해석하기 어려워진다. 채팅 protocol에서는 더 심각하다. delimiter와 tool tag는 괄호와 비슷해서 한쪽만 남으면 문법 상태가 바뀐다. 다만 model은 formal parser가 아니므로 문법 오류를 항상 예외로 보고하지 않는다. 그럴듯하지만 틀린 답을 낼 수 있다는 점에서 비유보다 위험하다.

### token-level truncation과 message-level policy를 분리한다

tokenizer의 일반 truncation은 지정 방향에서 token을 제거한다. 이는 tensor shape를 상한에 맞추는 마지막 안전장치로는 유용하다. 하지만 어느 message와 protocol unit을 보존할지는 알지 못한다. application/server는 먼저 message-level budget 정책을 적용해야 한다. system instruction, tool schema, 최근 user turn, 대응되는 assistant/tool-call pair, multimodal placeholder를 어떤 우선순위와 원자성으로 보존할지 결정한다.

예를 들어 context 상한이 8,192 token이고 최대 output 1,024를 예약한다면 prompt budget은 단순히 8,192가 아니다. model의 정확한 position limit, server가 요구하는 safety margin, multimodal expansion 뒤 길이, generation suffix를 고려해야 한다. rendered text가 7,000 token이더라도 image placeholder 하나가 processor에서 1,500 feature position으로 확장되면 최종 model input은 예산을 넘을 수 있다.

message-level policy는 “오래된 turn 삭제”만으로 끝나지 않는다. assistant tool call과 그 결과는 한 쌍으로 제거해야 할 수 있다. 과거 reasoning을 제거하면 다음 answer 의미가 달라질 수 있다. system instruction과 tools가 너무 커서 최근 user question을 위한 공간이 없다면 silent truncation보다 명시적 요청 오류나 tool 축약 정책이 나을 수 있다. 이 결정은 model library가 아니라 product/server owner의 책임이다.

### protocol atom을 정의하지 않으면 반쪽짜리 delimiter가 남는다

role block은 시작 delimiter, role name, newline, content, 종료 delimiter로 구성될 수 있다. tool block은 opening tag, canonical JSON, closing tag를 가진다. image placeholder는 vision start, pad, end token의 묶음일 수 있다. 이들을 protocol atom으로 정의하면 truncation이 atom 내부를 자르지 않도록 할 수 있다.

그렇다고 모든 message를 하나의 거대한 atom으로 보면 budget 활용이 나빠진다. tool schema 안에서 description을 축약할 수 있는지, 오래된 assistant visible content와 reasoning을 다르게 보존할 수 있는지 product semantics에 따라 세분화할 수 있다. 핵심은 byte나 token offset을 임의로 자르는 대신 **의미를 보존하는 단위와 허용된 변환**을 명시하는 것이다.

truncation 뒤에는 다음 invariant가 성립해야 한다. role delimiter가 균형을 이루고, tool call과 result의 참조 ID가 유효하며, multimodal placeholder 수와 bound feature 수가 맞고, 마지막 user/task state가 generation suffix와 양립하며, final token count가 output reservation을 포함한 상한 아래에 있어야 한다. 이 invariant는 긴 checklist로 외우기보다 하나의 실패 사례로 이해하는 편이 낫다.

### 실패 사례: system은 남았는데 실제 지시는 사라졌다

한 서버가 left truncation으로 오래된 token을 제거한다고 하자. system block이 아주 길고 끝부분에 안전 정책이 있으며, user turn이 뒤에 있다. truncation 결과 system 시작과 앞 문장은 사라졌지만 끝 delimiter와 일부 정책 문장만 남는다. token 수와 delimiter 균형은 우연히 맞을 수 있다. 서버 metric에는 오류가 없지만 답변 품질과 안전 동작이 달라진다.

운영자는 “system message가 요청에 있었다”는 raw JSON만 보고 template를 무죄로 볼 수 있다. 그러나 model이 받은 것은 truncation 후 IDs다. 증거는 raw system 존재 여부가 아니라 system content 중 몇 token이 어느 위치에 남았는지다. message source span을 token range에 연결했다면 잘린 구간을 설명할 수 있다. 그렇지 않으면 최종 IDs를 decode해 추측해야 하고 special token/normalization 때문에 원문 대응이 모호해진다.

복구는 무조건 right truncation으로 바꾸는 것이 아니다. right truncation은 최근 user 질문이나 generation suffix를 자를 수 있다. 올바른 선택은 product contract에 따라 system/tools/recent turns의 보존 정책을 정하고, 불가능한 요청을 명시적으로 거부하며, truncation event를 usage와 trace에 기록하는 것이다.

### cache key는 truncation 전이 아니라 실행 입력을 설명해야 한다

raw messages가 같더라도 server의 max context, reserved output, tool filtering, multimodal expansion 정책이 다르면 truncation 결과가 달라진다. prefix cache key를 raw JSON digest에만 묶으면 다른 실행 token sequence가 같은 identity를 가질 수 있다. 반대로 rendered final token IDs 전체만 key로 쓰면 correctness는 명확해지지만 partial prefix sharing과 template artifact audit를 별도로 설계해야 한다.

좋은 identity는 계층을 드러낸다. raw request digest는 API 재현용, template artifact와 kwargs digest는 compiler 재현용, rendered token prefix와 model/tokenizer revision은 KV reuse용이다. tenant와 adapter, multimodal feature identity도 필요한 범위에 포함한다. 어떤 field가 어느 key에 들어가는지는 cache 장에서 더 깊게 다루지만, template 장에서 중요한 결론은 명확하다. **렌더링과 truncation은 cache 앞의 의미 변환이며 key가 이를 건너뛰어서는 안 된다.**

### truncation과 billing은 같은 token을 다른 이름으로 센다

사용자는 원문 전체를 보냈지만 model은 일부만 받았을 수 있다. API usage가 원문을 token화한 수, 렌더링 후 수, truncation 후 실제 input 수 중 무엇을 보고하는지 명시되지 않으면 비용과 성능을 비교할 수 없다. prefix cache가 hit한 token과 실제로 compute한 prompt token도 다르다.

이 문제를 해결하려면 token ledger에 최소 네 값을 둔다. rendered-before-truncation, model-input-after-truncation, cached-input, computed-input이다. product billing이 이 중 하나를 사용하더라도 내부 trace에는 네 상태를 구분하는 편이 좋다. 단, request ID와 원문을 metric label로 넣어서는 안 된다. 높은 cardinality와 개인정보 노출을 피하기 위해 상세 identity는 trace나 접근 통제된 log에 둔다.

## 7.6 API와 서버는 어디까지 책임져야 하는가

template bug를 model repository 탓으로만 돌리면 서버가 삽입한 변환을 놓친다. 반대로 server가 모든 model template 의미를 자체적으로 재구현하면 model update와 쉽게 어긋난다. 책임 경계를 compiler pipeline의 각 pass로 나누면 소유권이 선명해진다.

API layer는 messages schema, role/content/tool field의 타입과 권한을 검증한다. template resolver는 model/tokenizer artifact와 server override에서 정확한 template를 고른다. renderer는 template kwargs와 messages를 deterministic하게 문자열 또는 token protocol로 바꾼다. tokenizer/processor는 special token과 multimodal feature를 결합한다. server는 context budget, cache key, usage, error mapping을 책임진다. model runner는 최종 IDs, positions, masks가 model contract에 맞는지 검증한다.

이 분할은 책임 떠넘기기가 아니다. 각 pass의 입력과 출력을 기록해 최초 불일치 owner를 찾기 위한 것이다. 예를 들어 unknown role을 API가 허용했지만 template가 예외를 내면 API/template contract mismatch다. renderer가 올바른 image placeholder를 만들었지만 processor가 feature 순서를 바꾸면 processor owner다. server가 truncation 후 token count를 usage에 반영하지 않으면 billing owner다.

### request별 template는 코드 실행과 비슷한 신뢰 경계다

Jinja template는 loop와 condition, helper를 사용한다. sandbox가 있어도 request가 임의 template를 제공하도록 허용하는 것은 단순 문자열 옵션보다 큰 권한이다. CPU를 많이 쓰는 loop, 매우 큰 출력, 예외를 유발하는 입력, server policy 우회 가능성을 검토해야 한다. vLLM의 `trust_request_chat_template` 같은 이름은 이 선택이 신뢰 경계임을 드러낸다.

보안 정책은 “Jinja sandbox이므로 안전” 한 문장으로 닫히지 않는다. 허용된 filter/global, template 크기, render timeout 또는 budget, 출력 길이 상한, file path resolution, builtin registry, audit digest를 살핀다. request별 template가 tenant마다 다르면 prefix cache namespace와 observability correlation에도 포함해야 한다. 그렇지 않으면 서로 다른 compiler output을 같은 model workload로 집계한다.

template override가 필요한 경우도 있다. model repository의 template가 누락되었거나 tool-use variant를 명시적으로 선택해야 할 수 있다. 안전한 운영은 override를 금지하는 것이 아니라 배포 artifact로 승격한다. review된 file을 image에 넣고 digest를 고정하며 startup에서 선택을 기록한다. request payload 안의 untrusted template와 deployment-controlled override를 같은 옵션으로 취급하지 않는다.

### error mapping은 prompt compiler의 사용자 인터페이스다

빈 messages, 잘못된 role, content type mismatch, tools schema 오류, context 초과는 서로 다른 요청 오류다. renderer exception을 모두 내부 500으로 바꾸면 운영자는 model/backend 장애로 오인하고 client는 수정 방법을 알 수 없다. 반대로 template 내부 세부와 path를 그대로 노출하면 정보 노출 위험이 있다.

서버는 오류를 입력 validation, unsupported template feature, context budget, internal render failure로 분류할 수 있어야 한다. 응답에는 수정 가능한 범위의 설명을 주고, 내부 trace에는 selected template digest, stage, exception class와 message index를 남긴다. 원문 content 전체를 log에 남기지 않고도 어느 role/item에서 실패했는지 기록할 수 있다.

### cache·billing·security identity는 같은 digest 하나가 아니다

모든 목적에 `hash(messages)` 하나를 쓰고 싶어질 수 있다. 그러나 cache는 model execution equivalence를, billing은 product accounting을, security audit는 tenant와 policy decision을 증명해야 한다. 목적이 다르므로 필요한 field와 보존 기간도 다르다.

cache identity에는 final token prefix, model/tokenizer/template revision, adapter와 multimodal binding이 중요하다. billing identity에는 rendered/truncated/cached/computed/generated/visible token의 전이가 중요하다. security identity에는 tenant, allowed tool set, selected policy/template, redaction과 override authority가 중요하다. 이 셋을 연결하는 request correlation ID는 필요하지만 metric label로 직접 넣지 않는다.

실제 장애에서는 이 구분이 힘을 발휘한다. cache hit가 올랐는데 다른 tenant의 tool prefix가 섞였다는 의심이 들면 cache key와 security identity를 대조한다. usage가 갑자기 줄었지만 GPU prefill은 그대로면 billing count와 computed token을 대조한다. template rollout 뒤 cache hit가 떨어지면 template digest와 rendered prefix를 대조한다. 하나의 “prompt hash”만 있으면 어느 의미가 달라졌는지 알 수 없다.

### server 비교는 endpoint 이름보다 pass ownership을 맞춘다

vLLM, SGLang, llama.cpp가 OpenAI-compatible endpoint를 제공하더라도 chat processing pass의 소유 위치는 다를 수 있다. 어떤 server는 Transformers tokenizer의 `apply_chat_template`에 크게 의존하고, 어떤 경로는 자체 renderer나 Python encoder를 사용하며, llama.cpp는 GGUF metadata와 C++ Jinja/legacy path를 가진다. endpoint가 같다는 사실은 compiled token IDs가 같다는 증거가 아니다.

공정한 비교는 raw request만 같게 두지 않는다. selected template bytes, template kwargs, content-format normalization, tokenizer/special tokens, truncation policy와 final IDs를 맞춘다. 이들을 의도적으로 다르게 둘 경우에는 차이를 결과의 설명 변수로 기록한다. 다음 절은 네 구현의 소스를 따라 이 pass ownership이 실제로 어디에 놓이는지 확인한다.

## 7.7 네 구현의 소스에서 prompt compiler를 따라간다

소스 산책은 함수 이름을 많이 모으는 일이 아니다. 동일한 요청이 어느 지점에서 conversation으로 정규화되고, template가 선택되며, kwargs가 합쳐지고, rendered text와 token IDs가 만들어지는지를 연결하는 일이다. 각 구현에서 같은 다섯 질문을 던진다.

1. 구조화 메시지를 누가 만들고 소유하는가.
2. template artifact와 variant는 어디서 고르는가.
3. tools·reasoning·generation flag는 어느 객체에 합쳐지는가.
4. rendering과 tokenization 사이의 중간 표현을 관찰할 수 있는가.
5. 오류와 final token IDs는 어느 경계에서 engine 입력으로 넘어가는가.

이 다섯 질문은 비교를 위한 공통 좌표일 뿐 표를 채우는 것이 목적은 아니다. 구현마다 pass가 합쳐지거나 갈라지는 이유와 그 결과를 읽어야 한다.

### Transformers: reference API 안에 선택·render·tokenize가 겹쳐 있다

Transformers는 많은 server가 기대는 기준 구현이지만 “template renderer 하나”로만 보면 안 된다. tokenizer와 processor에 같은 이름의 진입점이 있고, 실제 Jinja compile/render helper가 별도 파일에 있으며, template 선택도 tokenizer state와 요청 인자에 따라 달라진다.

읽기는 [`get_chat_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L3225-L3275)에서 시작할 수 있다. 이 함수는 explicit template, 저장된 string 또는 여러 template 중 어느 것을 사용할지 결정한다. tools가 있다고 해서 모든 버전과 model에서 같은 variant를 자동 선택한다고 가정하지 말고 고정 source의 분기를 읽는다. 선택 결과는 `apply_chat_template`로 들어가고, Jinja helper가 conversation과 tools를 rendering한다.

중요한 상태는 rendered string 하나가 아니다. batch conversation인지, template kwargs에 무엇이 들어갔는지, `add_generation_prompt`와 `continue_final_message`가 양립하는지, `return_assistant_tokens_mask`가 요청되었는지, tokenization/padding/truncation 결과가 어떤 container로 반환되는지를 본다. [`apply_chat_template` 후반의 tokenize·mask 경계](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L3165-L3223)는 renderer가 만든 generation span을 token mask로 바꾸는 좌표다.

멀티모달은 processor 진입점으로 옮겨 간다. [`ProcessorMixin.apply_chat_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/processing_utils.py#L1976-L2070)는 conversation과 media load/render kwargs를 다루고, 이어지는 구간이 images/videos/audio와 tokenizer/processor 출력을 결합한다. tokenizer method를 monkey-patch했는데 multimodal server 결과가 바뀌지 않는다면 processor 경로가 bypass 원인일 수 있다.

Transformers source walk에서 얻는 교훈은 편의 API와 관측 API를 구분하는 것이다. production은 한 번의 `tokenize=True` 호출을 쓸 수 있지만, 회귀 조사에는 selected template digest, rendered text, final IDs를 별도로 보존하는 wrapper가 필요하다. 이 wrapper가 원본 의미를 바꾸지 않는지도 differential로 확인해야 한다.

### vLLM: chat parsing과 renderer, engine ingress를 분리해 읽는다

vLLM의 OpenAI chat endpoint는 HTTP handler 안에서 즉시 tokenizer를 부르고 끝나지 않는다. serving class가 model/engine 상태를 검사하고 online renderer에 chat rendering을 위임한 뒤, 반환된 engine input을 scheduler 쪽으로 넘긴다. [`OpenAIServingChat.render_chat_request`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L192-L219)는 이 delegation 경계를 보여 준다.

이어지는 [`_create_chat_completion`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L240-L330)은 effective template kwargs, parser와 rendering 결과를 engine 요청 수명에 연결한다.

messages normalization은 별도 유틸리티에서 찾는다. [`parse_chat_messages`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/chat_utils.py#L1954-L1992)와 [비동기 media 경로](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/chat_utils.py#L1993-L2035)는 structured content가 conversation과 multimodal data로 나뉘는 좌표다. 이 단계를 지나면 원래 request object만 비교해서는 content normalization의 결과를 알 수 없다.

vLLM에서 특히 조심할 것은 content format과 template trust다. 문자열 content를 기대하는 template와 OpenAI-style content list를 기대하는 template가 다르며 auto detection이 개입할 수 있다. auto가 언제나 model author의 의도를 맞힌다는 보장은 없다. startup에 선택된 content format과 경고, request별 override 허용 여부를 artifact manifest에 포함해야 한다.

template loading도 source/path/builtin/literal을 구별한다. 앞서 본 `validate_chat_template`와 `_load_chat_template` 사이에는 validation과 actual resolution의 차이가 있다. file path가 startup 이후 바뀌었을 때 cached loader가 무엇을 반환하는지, literal과 이름이 모호할 때 어느 분기를 타는지도 재현성에 영향을 준다. “명령행에 같은 문자열을 넣었다”보다 load된 bytes digest가 강한 증거다.

이 구조에서 관측점은 세 곳이다. parse 뒤 conversation과 media binding, renderer 뒤 engine input token IDs, serving이 scheduler로 넘기기 직전 prompt components다. privacy 때문에 content 전체를 항상 log하지 않더라도 길이, digest, role/type sequence, placeholder count와 template identity를 trace field로 남길 수 있다.

### SGLang: OpenAI serving이 render와 encode를 의도적으로 나눈다

SGLang 고정 source에는 prompt compiler의 경계를 설명하기 좋은 주석이 있다. [`serving_chat.py`의 render/encode 구간](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_chat.py#L1300-L1355)은 `apply_chat_template(tokenize=True)`를 한 번에 쓰지 않고 먼저 `tokenize=False`로 렌더링한 뒤 tokenizer `encode`를 호출한다. 주석은 template가 role/special token을 이미 포함하므로 tokenizer가 BOS 같은 special token을 다시 추가하지 않게 해야 하는 경계를 설명한다.

이 분리는 관측성에도 이점이 있다. rendered prompt와 prompt IDs 사이를 직접 비교할 수 있다. 하지만 분리했다고 자동으로 안전한 것은 아니다. `_tokenizer_auto_adds_specials`에 따른 `add_special_tokens=False` 선택이 tokenizer backend별 의미와 맞아야 한다. template가 BOS를 포함하지 않는 model에 무조건 false를 전달하면 필요한 token을 잃을 수 있고, 포함하는데 true를 쓰면 중복될 수 있다.

또 하나의 흥미로운 분기는 tools 형식 fallback이다. 첫 rendering이 실패하면 OpenAI wrapper 안의 function을 평탄화해 다시 시도하는 구간이 이어진다. 이것은 호환성을 높이지만 같은 request가 어느 attempt에서 성공했는지에 따라 compiler input이 달라진다. fallback counter나 trace field가 없으면 template가 native OpenAI tool wrapper를 지원한 것처럼 오해할 수 있다. 성공 여부뿐 아니라 selected tool representation을 기록해야 하는 이유다.

reasoning kwargs도 rendering 직전에 합쳐진다. [`serving_chat.py`의 assistant prefill·reasoning kwargs 처리](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_chat.py#L1280-L1325)는 마지막 assistant message 처리, request reasoning effort와 template kwargs가 같은 compile pass에 들어가는 위치를 보여 준다. protocol model의 default 주입과 serving의 merge가 모두 있으므로 effective kwargs를 최종 지점에서 관찰해야 한다.

SGLang에는 Hugging Face tokenizer가 아닌 backend도 있다. [`TikTokenTokenizer.apply_chat_template`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/tokenizer/tiktoken_tokenizer.py#L106-L157)은 자체 Jinja template와 rendering 후 encode를 제공한다. “SGLang은 항상 Transformers의 template semantics를 그대로 쓴다”는 설명이 틀린 이유다. served model과 tokenizer backend까지 source path를 고정해야 한다.

### llama.cpp: GGUF template artifact와 Jinja·legacy 두 실행 경로

llama.cpp에서는 template가 GGUF model artifact 안에 들어갈 수 있다. [`llama_model_chat_template`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model.cpp#L2844-L2868)는 model metadata에서 default 또는 named template를 얻는 public 경계다. 변환 단계에서는 Hugging Face tokenizer config나 Jinja file의 template를 GGUF metadata에 넣을 수 있으므로 model file digest가 template identity까지 포함할 수 있다. 그러나 runtime override가 있다면 GGUF digest만으로는 충분하지 않다.

[`common_chat_templates_init`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/chat.cpp#L753-L850)은 override, model default, `tool_use` named template와 fallback source를 선택하고 BOS/EOS token을 연결해 template 객체를 만든다. 이 구간은 template 선택이 string 하나를 읽는 행위가 아님을 잘 보여 준다. model vocab에 필요한 BOS/EOS가 없는데 template가 변수를 사용하면 경고하는 분기도 있다. model artifact, vocab와 template가 하나의 ABI인 셈이다.

적용 단계에는 Jinja와 legacy 경로가 공존한다. [`common_chat_templates_apply_jinja`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/chat.cpp#L3600-L3742)는 tools를 OpenAI-compatible JSON으로 바꾸고, tool-use template 선택과 generation prompt, reasoning/tool parser 관련 상태를 만든다. [`common_chat_templates_apply_legacy`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/chat.cpp#L3743-L3807)는 알려진 legacy template 적용 API로 내려간다.

마지막 [`common_chat_templates_apply`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/chat.cpp#L3809-L3814)가 `use_jinja`에 따라 둘을 고른다.

이 이중 경로는 중요한 반례를 만든다. 같은 template source를 지정했다고 해도 Jinja와 legacy가 지원하는 기능과 출력이 완전히 같다고 가정할 수 없다. tools/reasoning parser와 generation suffix, unsupported custom template 오류가 다를 수 있다. 서버 비교에서는 `--jinja` 또는 대응 설정, chosen path와 actual rendered prompt를 함께 기록해야 한다.

### 네 구현 비교에서 얻는 결론

Transformers는 reference API 안에 selection/render/tokenize/mask를 모으고 processor가 multimodal 확장을 맡는다. vLLM은 chat parsing과 online renderer, engine ingress를 분리한다. SGLang OpenAI path는 render와 encode를 명시적으로 나누고 tool representation fallback을 갖는다. llama.cpp는 GGUF metadata/override에서 template를 정하고 Jinja와 legacy execution을 나눈다.

어느 구조가 무조건 옳다는 결론은 없다. pass를 합치면 API는 단순하지만 중간 산출물 관측이 어려울 수 있다. 나누면 관측과 정책 삽입은 쉬워지지만 special-token option을 정확히 전달해야 한다. fallback은 호환성을 높이지만 effective input identity를 숨길 수 있다. 중요한 것은 해당 revision의 pass ownership을 알고, 같은 요청 비교에서 실제 compiled IDs까지 맞추는 것이다.

## 7.8 하나의 compile trace로 회귀를 끝까지 닫는다

이제 template rollout 뒤 tools를 붙인 요청만 답이 달라졌다는 사건을 조사해 보자. 상황은 이렇다. model weight와 tokenizer는 그대로이고 server image를 교체했다. 일반 대화의 greedy output은 기존과 같지만 tools가 있는 요청은 first token부터 다르다. prompt token 수도 평균 23개 늘었다. GPU backend와 scheduler 설정은 변하지 않았다.

처음 떠오르는 가설은 새 server가 tool parser를 바꿔 output 후처리가 달라졌다는 것이다. 그러나 first raw logits부터 다르다면 output parser는 원인이 될 수 없다. 두 번째 가설은 tool schema JSON key 순서가 달라져 rendered prefix가 바뀌었다는 것이다. 세 번째는 새 version이 `tool_use` template variant를 선택하거나 다른 tool representation fallback을 사용했다는 것이다. 네 번째는 reasoning default kwargs가 tools와 함께 주입된 것이다.

조사는 final answer diff가 아니라 compile pipeline을 앞에서부터 걷는다. raw messages와 tools의 canonical capture가 같은지 본다. selected template digest와 effective kwargs를 비교한다. rendered UTF-8 bytes의 최초 차이와 그 차이가 속한 source message/schema field를 찾는다. token IDs의 최초 차이와 generation suffix를 본다. 여기서 이미 차이가 나면 GPU kernel 가설은 보류한다.

가령 새 server가 tools list를 `[{type:function, function:{...}}]`에서 내부 function dict만 남긴 형태로 template에 넘겼다고 하자. template가 둘 다 받아 렌더링에 성공했지만 출력 JSON wrapper가 달라져 23 token이 늘었다. first logits divergence는 이 차이로 설명할 수 있다. 같은 rendered IDs를 두 server engine에 직접 넣었을 때 first logits가 허용 오차 안에서 같다면 causal chain이 닫힌다.

복구는 구버전 전체 rollback만 있는 것이 아니다. tool representation을 명시적으로 고정하고 template artifact/kwargs digest를 deployment contract에 넣을 수 있다. 그러나 “답이 예전과 같아졌다”만으로 종료하지 않는다. 일반 chat, one/many tools, tool result, assistant prefill, reasoning on/off, multimodal content의 regression matrix에서 compiled IDs 또는 의도된 차이를 검증한다. prefix cache namespace와 prompt-token usage가 새 identity를 반영하는지도 본다.

**워크북: 두 번 적용된 template를 정적으로 찾는다**

실행 없이도 강한 검사를 할 수 있다. API gateway가 이미 rendered string을 `prompt`로 만들었는데 downstream server가 이를 user content로 감싸 다시 template를 적용하는 경우를 생각하자. 최종 문자열에는 role delimiter가 중첩되고 prompt token이 비정상적으로 늘어난다.

첫째, gateway와 server 사이 payload schema를 읽어 messages인지 prompt string인지 구분한다. 둘째, 양쪽 source에서 `apply_chat_template` 또는 renderer call을 찾는다. 셋째, downstream 입력 타입 검증이 rendered prompt를 raw user content로 받을 수 있는지 본다. 넷째, 각 pass의 출력 예시를 손으로 합성하여 delimiter가 두 번 나타나는지 확인한다.

이 검사는 “delimiter 문자열이 두 번 보이면 무조건 bug”라는 규칙이 아니다. 과거 assistant content 안에 quoted delimiter가 있을 수 있고 model protocol 자체가 반복 marker를 쓸 수 있다. 중요한 것은 동일한 구조화 message가 두 compiler pass를 순차 통과했는지 ownership을 증명하는 것이다.

**워크북: truncation이 protocol을 깨뜨리는 최초 길이**

template source와 tokenizer artifact를 고정하고 길이가 점차 늘어나는 messages fixture를 설계한다. 실행은 이 책의 현재 작업 범위 밖이지만 검증 조건을 소스 수준에서 정의할 수 있다. 각 길이에서 message-level policy가 선택한 turn, rendered-before-truncation token 수, final IDs, delimiter/tool/placeholder balance, reserved output을 기록한다.

경계값은 평균 길이가 아니라 `limit-1`, `limit`, `limit+1`과 protocol atom 직전·중간·직후다. 단순 token truncation이 atom 중간을 자르는 최초 길이를 찾고, message-level policy가 같은 budget에서 어떤 의미를 보존하는지 비교한다. 복구 종료 조건은 모든 입력을 억지로 성공시키는 것이 아니라, 보존 가능한 경우 유효 protocol을 만들고 불가능한 경우 명시적 요청 오류를 반환하는 것이다.

**워크북: tools·reasoning·multimodal이 동시에 있을 때**

각 기능을 따로 검증했는데 세 기능을 함께 켜면 실패하는 경우가 있다. user content에는 이미지 두 장과 질문이 있고, system에는 tools schema가 있으며, 이전 assistant turn에는 reasoning과 tool call이 있다고 하자. 이 fixture는 복잡해 보이지만 prompt compiler의 진짜 책임 경계를 드러낸다.

messages를 읽는 첫 pass는 text와 image item의 순서를 보존해야 한다. tool schema를 system block에 넣는 pass는 user content를 손상시키지 않아야 한다. 과거 assistant를 직렬화하는 pass는 reasoning, visible content와 tool call을 model protocol 순서로 배치해야 한다. generation suffix는 마지막 tool result 뒤 assistant가 다시 답할 차례라는 상태를 나타내야 한다. 마지막 processor pass는 placeholder와 image tensors를 같은 순서로 묶어야 한다.

이 요청에서 final text가 틀렸다는 사실만으로는 어느 pass가 원인인지 알 수 없다. 정적 검증은 산출물을 여섯 좌표로 나눈다. message source index와 content item index, rendered byte span, token span, multimodal placeholder ordinal, feature batch index, generation boundary다. 예를 들어 `messages[2].content[1]`의 두 번째 image가 rendered placeholder ordinal 1, feature batch index 1로 이어지는지 확인한다. 0-based/1-based 표기를 섞지 않는다.

tool call도 source map이 필요하다. 과거 assistant의 `tool_calls[0].id`와 다음 tool role의 `tool_call_id`가 렌더링된 protocol에서 같은 호출을 가리키는지 본다. template가 여러 tool result를 하나의 user block으로 합친다면 각 결과의 경계와 순서를 기록한다. reasoning을 visible content에서 분리하는 과정이 tool call 앞뒤 delimiter를 제거하지 않는지도 본다.

경쟁 가설은 적어도 네 개다. template가 content list를 평문으로 문자열화했다. tools variant가 선택되면서 multimodal-aware default template를 대체했다. reasoning parser가 assistant content를 잘못 잘라 tool call을 잃었다. placeholder 순서는 맞지만 processor feature batch가 달라졌다. rendered bytes와 selected variant에서 갈라지면 앞의 세 가설을 좁힐 수 있고, token/placeholder까지 같고 merged embedding에서 갈라지면 processor binding을 우선한다.

이 fixture를 모든 request에 logging할 필요는 없다. release approval에서 고정 artifact로 사용하고 production에서는 template digest, role/content-type sequence, tool count/schema digest, placeholder count와 generation suffix digest 같은 저비용 field를 남긴다. 장애 request가 발생하면 접근 권한이 있는 환경에서 원문을 redaction한 재현 bundle로 확장한다.

**정적 승인 시험: template 변경을 model 변경처럼 다룬다**

template rollout은 weight를 바꾸지 않으므로 가벼운 설정 변경으로 취급되기 쉽다. 그러나 compiler output이 바뀌면 model input distribution이 바뀐다. 승인 시험도 model artifact 변경과 비슷한 엄격함이 필요하다.

먼저 artifact identity를 고정한다. 이전·새 template bytes와 digest, tokenizer/model revision, server rendering code revision, default/request kwargs를 manifest에 둔다. 다음으로 대표 conversation corpus를 고정한다. 단일 user, system+user, multi-turn, 마지막 assistant prefill, one/many tools, tool result, reasoning on/off, multimodal single/multiple, context boundary를 포함한다.

각 fixture에서 raw messages부터 final model input까지 semantic diff를 만든다. 모든 token diff를 실패로 보지는 않는다. 의도한 새 delimiter나 tool syntax가 있다면 diff가 기대값이다. 중요한 것은 변경 이유와 영향을 분류하는 것이다. role boundary diff, tool schema diff, generation suffix diff, placeholder binding diff, truncation diff를 서로 다른 범주로 기록한다.

의도하지 않은 diff가 0이라고도 끝내지 않는다. 새 renderer가 예외 mapping이나 performance를 바꿀 수 있다. static code path에서 template compile cache key와 size, request override trust policy, fallback branch를 비교한다. template syntax error가 startup에서 발견되는지 첫 request에서 발견되는지, tools representation fallback이 새로 활성화되는지도 본다.

정확성 승인의 중심은 first divergence다. token IDs가 의도대로 달라졌다면 그 뒤 logits와 output이 달라질 수 있다는 사실을 변경 영향으로 받아들인다. token IDs가 같아야 하는 fixture에서 다르면 rollout을 멈춘다. token IDs가 같은데 output이 다르면 template 변경만으로 설명되지 않으므로 model/backend state를 조사한다.

성능 승인은 rendered prompt token distribution을 비교하는 데서 시작한다. 평균만 보면 긴 tools 요청의 tail을 놓친다. fixture별 prompt token, multimodal expansion length, common prefix length, expected cache namespace를 계산한다. dynamic field가 새로 들어가 공통 prefix가 깨지는지 확인한다. template compile/render CPU 비용도 경로상 존재하지만 실측 없이 빠르거나 느리다고 단정하지 않는다.

보안 승인은 allowed tools와 tenant identity, request override 권한, template output에 포함된 민감 field, log redaction을 본다. 새 template가 과거 hidden reasoning을 다시 prompt에 넣는다면 privacy retention 정책이 바뀐 것이다. 사용자가 볼 수 없는 내용이라고 보안 영향이 없는 것이 아니다.

rollback은 template file만 되돌리는 것으로 끝나지 않을 수 있다. prefix cache namespace가 template digest를 포함하지 않았다면 새·옛 compiled prefix가 섞였을 가능성이 있다. tokenizer config나 server default kwargs도 함께 배포되었다면 atomic bundle로 되돌려야 한다. 승인 종료 조건은 old/new artifact의 선택이 replica마다 일치하고, cache namespace가 분리되며, 대표 corpus의 intended diff와 error behavior가 문서화된 상태다.

**source review에서 묻는 것은 함수 이름보다 불변식이다**

네 구현의 line link를 열 때 코드를 위에서 아래로 읽기만 하면 세부 분기에 빠지기 쉽다. 먼저 불변식을 적고 해당 mutation을 찾는다.

첫 번째 불변식은 deterministic selection이다. 같은 model/tokenizer/template source와 effective kwargs가 같은 template bytes를 선택해야 한다. 두 번째는 role/content preservation이다. 입력 message 순서와 content item binding이 rendered protocol에서 추적 가능해야 한다. 세 번째는 single lowering이다. template가 role/special token을 넣었다면 tokenizer가 같은 BOS나 delimiter를 중복 추가하지 않아야 한다.

네 번째는 generation state consistency다. `add_generation_prompt`, assistant prefill, reasoning/tool state가 양립해야 한다. 다섯 번째는 atomic protocol truncation이다. 역할·도구·멀티모달 atom의 일부만 model input에 남아서는 안 된다. 여섯 번째는 identity completeness다. cache와 audit key가 의미를 바꾸는 artifact·flag·tenant field를 누락하지 않아야 한다.

이 불변식을 source에 대입하면 review가 구체적이 된다. vLLM loader의 cache key가 file content 변경을 어떻게 다루는지, SGLang render/encode 분리에서 special token 추가 여부가 어디서 결정되는지, llama.cpp Jinja/legacy 선택이 effective path에 기록되는지, Transformers generation span이 char-to-token으로 변환되지 못할 때 어떤 결과가 되는지 묻는다.

의도를 코드에서 과장해 읽지 않는다. fallback branch가 있다는 사실은 작성자가 모든 template 호환성을 보장한다는 뜻이 아니다. cache decorator가 있다는 사실은 file mutation을 지원한다는 뜻이 아니다. 경고가 있다는 사실은 운영자가 반드시 이를 수집한다는 뜻이 아니다. source가 보여 주는 predicate와 mutation, 반환값을 먼저 쓰고 설계 이유는 문서·주석이 뒷받침하는 범위에 한정한다.

**장말 산출물은 긴 체크리스트가 아니라 한 장의 compile trace다**

이 장을 실제 문제에 적용했다면 가장 가치 있는 산출물은 option 표가 아니라 request 하나의 compile trace다. trace에는 raw messages schema digest, selected template source/digest, effective kwargs, rendered bytes digest와 길이, token IDs digest와 길이, generation boundary, truncation 전후 길이, multimodal binding count, cache namespace와 usage counts가 시간순으로 놓인다.

모든 값을 production metric으로 만들지는 않는다. low-cardinality aggregate는 metric에, request-level identity와 span은 trace에, 접근 통제가 필요한 content diff는 redacted evidence bundle에 둔다. 이 층을 구분해야 관측 자체가 cardinality와 개인정보 문제를 만들지 않는다.

compile trace가 있으면 “같은 prompt”라는 모호한 문장을 버릴 수 있다. 어느 representation이 같고 어디서 달라졌는지를 말할 수 있다. 또한 하위 장과의 인터페이스가 생긴다. 8장은 final IDs 안의 special token과 padding/truncation, 9장은 embedding/position 입력, 10장은 first logits, 11장은 generated/visible token commit을 이어 받는다.

**손으로 계산하는 prompt 비용과 cache 공유 경계**

compiler trace가 실제 serving 비용으로 어떻게 이어지는지 작은 숫자로 확인해 보자. system과 두 turn의 대화가 180 token이고 generation suffix가 3 token이라고 하자. tools가 없으면 model input은 183 token이다. tool schema 열 개를 직렬화했더니 schema와 설명, 호출 문법이 1,240 token을 추가하면 input은 1,423 token이 된다. tool을 실제로 호출하지 않아도 prefill은 추가된 1,240 위치를 처리하고 각 layer의 attention state를 만든다.

이 숫자는 정확한 성능 예측이 아니라 비용 장부의 출발점이다. prefill FLOP과 KV byte는 model architecture에 따라 달라지고 GQA나 hybrid layer가 영향을 준다. 그러나 “tool을 호출하지 않았으므로 비용 0”이라는 가설은 prompt token ledger만으로 반박된다. GPU 시간이 늘지 않았다면 prefix cache hit, chunked prefill overlap 또는 다른 병목이 추가 비용을 가렸는지 다음 장에서 조사한다.

이제 열 개 tool의 배열 순서가 tenant마다 무작위라고 하자. 의미상 tool 집합이 같더라도 rendered prefix의 이른 위치부터 token sequence가 달라진다. block prefix cache가 16-token 단위로 공유한다고 가정하면, 차이가 첫 tool 이름에서 시작된 뒤의 긴 schema는 공유되지 않을 수 있다. key 정렬로 순서를 안정화하면 공유 길이가 늘 가능성이 있지만 tool preference가 배열 순서에 의존하는 model이라면 의미가 바뀔 수 있다. 최적화 전에 order가 protocol 의미인지 단순 serialization accident인지 증명해야 한다.

dynamic timestamp를 system prompt 첫머리에 넣는 경우는 더 분명하다. 매 요청마다 첫 block부터 token이 달라져 뒤의 공통 instructions와 tools를 prefix cache가 재사용하지 못할 수 있다. timestamp가 정말 model 입력에 필요하다면 이를 제거할 수 없다. 필요하지 않고 audit log에만 있으면 prompt 밖으로 옮기는 편이 의미와 cache를 함께 보존한다. “cache hit를 높이기 위해 timestamp를 제거했다”가 아니라 “model이 소비할 필요가 없는 request metadata를 compiler 입력에서 분리했다”가 더 정확한 설명이다.

multimodal expansion도 ledger에 따로 적는다. rendered text 단계에서 image placeholder 두 개가 각각 한 special token으로 보이더라도 processor가 image당 576 feature position으로 확장한다면 model input은 단순 text token 수보다 1,150 위치가량 늘 수 있다. 정확한 값은 processor/config에 달려 있다. truncation budget을 rendered tokenizer 길이만으로 계산하면 expansion 뒤 context를 넘을 수 있으므로 pre-expansion과 post-expansion 길이를 모두 기록한다.

회계 예시를 마무리해 보자. raw messages의 text만 180, template delimiter와 suffix가 23, tools가 1,240, image expansion의 순증가가 1,150이면 final model input은 2,593 위치가 된다. server가 usage에 1,443만 보고한다면 text/tokenizer 기준일 수 있고, engine은 2,593 위치를 처리했을 수 있다. 어느 수치가 API contract상 옳은지는 제품 정책의 문제지만, 둘을 같은 `prompt_tokens`라는 이름으로 비교하면 GPU 비용과 billing이 맞지 않는 것처럼 보인다.

**반례를 품은 설명만 운영에서 살아남는다**

이 장의 compiler 관점도 만능은 아니다. compiled IDs가 다르면 output이 반드시 크게 달라진다고 단정할 수 없다. 공백 token 하나의 차이를 model이 사실상 무시할 수 있다. compiled IDs가 같아도 positions, masks, multimodal embeddings가 달라 output이 갈릴 수 있다. template digest가 다르지만 대표 fixture에서 동일 output을 만드는 두 template도 있을 수 있다.

따라서 identity diff는 원인 후보를 좁히는 증거이지 품질 결과의 자동 판결이 아니다. rendered/token diff가 발견되면 그 차이가 의도된 protocol 변화인지 먼저 분류하고, first logits 또는 model input의 다음 경계에서 영향을 확인한다. 반대로 결과가 같다는 이유로 template rollout이 안전하다고 말하지 않는다. rare role/tool/multimodal/context-boundary fixture에서만 차이가 나타날 수 있다.

cache에서도 같은 주의가 필요하다. token prefix가 같다는 것은 해당 prefix KV를 공유할 correctness 조건의 일부지만 tenant isolation, model/adapter revision, position/cache layout 같은 조건이 더 있다. token이 다르다는 것은 exact prefix share가 불가능하다는 강한 신호지만 semantic cache처럼 다른 계층은 별도 계약을 가질 수 있다. 이 장은 exact model-input compiler의 경계를 다루며 의미 기반 cache의 안전성을 대신 증명하지 않는다.

마지막으로 source path가 같다는 사실도 실행 path의 증거가 아니다. server config와 tokenizer backend, Jinja/legacy flag, request content type에 따라 다른 branch가 선택된다. source note의 permalink는 가능한 계약을 보여 준다. 실제 request가 어느 branch를 탔는지는 selected artifact와 trace, rendered output으로 확인해야 한다. 이 구분을 지켜야 source reading이 뇌피셜을 줄이는 도구가 된다.

## 7.9 tool·multimodal 요청에서 template cache identity를 검증하는 종합 incident

한 팀이 prompt rendering cache를 도입한 뒤 text-only 대화는 빨라졌지만 tool과 image가 섞인 요청에서만 간헐적으로
엉뚱한 tool schema가 모델에 들어갔다. Cache key에는 model name, message JSON digest와 template 이름이 있었다.
운영자는 같은 message면 같은 prompt라고 생각했지만, 실제 compiler input에는 template source, Jinja environment,
special-token policy, tool schema normalization, multimodal placeholder binding과 generation-prompt mode가 더 있었다.
이 사건은 cache miss 성능 문제가 아니라 서로 다른 compiler program을 같은 identity로 접은 correctness 문제다.

첫 fixture는 system 한 개, user text 한 개, tool 두 개와 image 한 개를 가진다. Tool A와 B는 JSON object key 순서만
다른 schema, image는 같은 bytes지만 request-local ordinal이 다르다. 네 lane을 만든다. Template revision만 old/new,
tool schema ordering만 canonical/raw, image placeholder binding만 explicit/implicit, `add_generation_prompt`만 on/off로
바꾼다. 각 lane은 rendered Unicode string, UTF-8 bytes, special-token annotated segments, final IDs와 assistant
generation boundary를 남긴다. 최종 ID가 같기 전에는 cache 공유 후보가 아니다.

Jinja 단계는 문자열 치환기로만 읽지 않는다. Template는 message list를 순회하고 role/content type에 따라 branch하며,
tool definitions를 serialization하고 delimiter를 배치한다. Undefined variable 정책이 strict인지 silent인지도 output을
바꾼다. `tools`가 빠졌을 때 empty list와 undefined가 같은 branch인지, multimodal content item이 mapping인지 object인지,
custom filter가 deterministic한지 확인한다. Template source digest만 같아도 environment와 helper/filter generation이
다르면 program identity는 다르다.

Rendered string 다음에는 special-token insertion owner가 온다. Template가 문자 형태의 BOS/EOS를 직접 출력하고 tokenizer
호출도 `add_special_tokens=True`라면 BOS가 두 번 들어갈 수 있다. 반대로 template가 delimiter만 만들고 tokenizer가
generation BOS를 붙이는 계약이라면 `False`로 바꿀 때 첫 token이 사라진다. 따라서 compile trace는 각 special ID마다
`producer=template/tokenizer/server`, logical role, string span과 final index를 적는다. ID 배열 끝만 비교하지 않고 최초
중복·누락 index에서 멈춘다.

작은 계산을 해 보자. System 18 tokens, tool preamble 240, user text 32, image placeholder 4, delimiter와 role tokens 11,
generation prompt 3이라면 실행 prompt는 308 tokens다. Cache가 tool preamble 없는 text-only variant 68 tokens를 잘못
재사용하면 240-token 절약처럼 보이지만 모델은 tool contract를 받지 않았다. TTFT 개선은 invalid work omission이다.
Goodput 계산에서는 syntax/correctness gate를 통과하지 못한 response를 useful completion으로 세지 않는다.

Tool schema canonicalization은 의미 보존 여부를 따져야 한다. JSON object key order는 보통 의미가 없지만 required 배열,
enum order를 prompt에서 어떻게 설명하는지, description whitespace와 numeric default 표현은 token IDs를 바꿀 수 있다.
Canonicalizer가 schema digest만 만들고 template에는 raw schema를 넣으면 같은 key 아래 다른 prompt가 생긴다. 반대로
canonicalized schema를 실제 rendering에도 사용하면 key와 compiler input이 일치한다. `key input`과 `render input`을 한
장부에 두는 이유다.

Multimodal placeholder는 네 번째 identity 층이다. Rendered text의 `<image>` 한 개가 feature tensor 한 개와 자동으로
결합된다고 가정하지 않는다. Processor가 한 image를 여러 tiles와 newline feature로 확장할 수 있고 placeholder token
수는 model architecture와 resolution에 따라 달라진다. Cache entry가 text token IDs만 저장한다면 feature digest,
processor revision, placeholder expansion과 request-local binding을 별 key 또는 non-cacheable predicate로 관리해야 한다.
같은 URL 문자열도 fetch 결과와 authorization context가 다를 수 있으므로 URL만 identity로 쓰지 않는다.

Cache는 compiler 단계별로 나누는 편이 설명 가능하다. Template render cache는 normalized messages, selected template, environment, tool representation을 key로 하고 문자열을 output으로 가진다. Tokenization cache는 rendered bytes, tokenizer artifact와 special-token policy를 key로 하고 IDs/offsets를 output으로 가진다. Multimodal processor cache는 media content digest, processor config와 transformation parameters를 key로 하고 feature bundle을 output으로 가진다. 최종 prompt bundle cache는 이 세 generation의 relation과 position/binding metadata를 묶는다.

한 digest로 전부 숨기면 first drift를 찾기 어렵다.

Assistant mask도 cache entry의 의미를 바꾼다. 같은 final IDs라도 training-style mask, response extraction mask 또는
tool reasoning span이 다를 수 있다. Serving에서 logits를 계산할 위치와 prompt/output usage를 mask로 정한다면 mask digest를
bundle identity에 포함한다. Mask를 쓰지 않는 경로에서는 `not consumed`라고 명시해 불필요한 miss를 피한다. Field가 API에
존재한다는 이유만으로 모든 consumer가 의미 있게 사용한다고 가정하지 않는다.

Incident의 경쟁 가설은 네 개다. H1은 template selection drift이며 selected source/environment digest가 failing request에서
다를 것을 예측한다. H2는 double special-token insertion이며 rendered marker와 tokenizer-added ID가 같은 logical boundary에
두 번 나타날 것을 예측한다. H3은 cache key 누락이며 miss lane은 맞고 hit lane만 다른 bundle generation을 받을 것을
예측한다. H4는 multimodal binding drift이며 text IDs는 같지만 feature count·placeholder range가 다를 것을 예측한다.
각 가설은 다른 checkpoint에서 반증된다.

Pinned source walk는 public `apply_chat_template` entry에서 template selection, compiled template cache, render context와 tokenizer call을 구분해 따라간다. 함수 이름이나 module-level cache가 있다는 사실만으로 cache key가 충분하다고 말하지 않는다.

실제 key constructor가 source string과 environment option을 어떻게 묶는지, caller가 tools/documents와 tokenize flag를 넘기는지, 반환 object가 mask를 포함하는지 consumer까지 읽는다. vLLM/SGLang은 OpenAI request parsing과 renderer, engine prompt handoff 사이에서 동일 trace를 만든다. llama.cpp는 GGUF template resolution과 Jinja/legacy path 선택이 effective template identity를 만드는 지점까지 확인한다.

정적 regression은 cache hit/miss를 강제로 실행하지 않아도 설계할 수 있다. 두 compiler input을 canonical serialization하고
expected-equal/expected-different relation을 선언한다. Expected-equal pair는 object key order처럼 의미가 같고 actual render에도
같은 canonical form을 쓰는 경우다. Expected-different pair는 template revision, generation-prompt mode, tool set,
media feature bundle 또는 special-token policy가 다른 경우다. Key projection이 expected-different pair를 같게 만들면 collision
fixture가 실패한다.

Truncation까지 들어오면 cache identity는 최종 실행 prompt를 설명해야 한다. 같은 untruncated messages가 context budget과
atomic policy에 따라 다른 tool definition이나 history를 보존할 수 있다. Raw request digest만 key로 쓰면 8K와 32K lane,
text-only와 image-expanded lane이 충돌한다. `compiler input identity`, `truncation policy/budget`, `compiled output identity`를
나누고 cache가 어느 단계 output을 저장하는지 적는다. Truncation 뒤 IDs가 같다면 상위 raw 차이가 model KV reuse에는
불필요할 수 있지만 audit/security cache에는 보존할 수 있다.

복구는 cache flush 한 번으로 끝나지 않는다. 먼저 새 hit admission을 막고 affected template/tool/media generation을
격리한다. Entry가 final bundle을 포함하면 잘못 결합된 feature와 mask도 함께 폐기한다. 수정된 key schema는 version을 올려
old writer와 new reader가 namespace를 공유하지 않게 한다. Canary는 text, tool-only, image-only, tool+image와 truncation
boundary를 포함하고, rendered bytes→special IDs→feature binding→generation boundary가 reference와 같은지 확인한다.

성능 terminal은 correctness 뒤에 온다. Render cache hit율이 올라가도 key 생성과 canonicalization CPU가 커질 수 있고,
tool schema가 큰 요청은 serialization cost가 지배할 수 있다. Hit/miss별 render, tokenize, processor, queue 시간을 나누고
entry 크기와 eviction churn을 본다. Tenant/tool authorization이 cache 공유 범위를 제한하면 낮은 hit율은 안전 policy의
결과일 수 있다. 보안 경계를 풀어 hit율을 높이지 않는다.

최종 compile trace는 한 줄의 prompt가 아니다. Selected template/environment, normalized message/tool/media inputs, rendered
bytes, logical special-token producers, final IDs, assistant/generation boundary, truncation result와 feature binding을 같은
generation으로 잇는다. Cache entry는 이 trace의 어느 구간을 재사용하는지와 어떤 identity predicate를 요구하는지 말한다.
이 구조가 있으면 “Jinja가 달라졌다”, “tool prompt가 길다”, “multimodal cache가 이상하다”라는 막연한 설명을 최초
불일치와 안전한 무효화 범위로 바꿀 수 있다.

### 7.9.1 cache identity를 합성하는 필드를 실제 byte로 계산한다

이제 사건을 숫자로 닫아 보자. 요청 R1과 R2는 같은 user message를 갖고, tool 이름과 설명도 같다. 차이는 JSON Schema의 `required` 순서와 image preprocessing revision뿐이다. R1의 canonical message bytes가 412, canonical tool schema가 638, media manifest가 96 byte라고 하자. R2는 schema key ordering을 정규화한 뒤 638 byte로 같지만 processor revision 문자열이 달라 media manifest가 101 byte다. 단순히 세 문자열을 이어 붙이면 경계가 모호해진다. 길이 prefix와 field tag를 포함한 canonical envelope를 만든다.

```
identity_input =
  tag("template")  || u64(len(template_bytes)) || template_bytes ||
  tag("messages")  || u64(len(message_bytes))  || message_bytes  ||
  tag("tools")     || u64(len(tool_bytes))     || tool_bytes     ||
  tag("media")     || u64(len(media_bytes))    || media_bytes    ||
  tag("flags")     || u64(len(flag_bytes))     || flag_bytes
cache_id = sha256(identity_input)
```

`ab`+`c`와 `a`+`bc`가 같은 concatenation이 되는 식의 구조 충돌을 length prefix가 막는다. field tag는 schema evolution 때 필드 순서가 바뀌어도 의미를 보존한다. 실제 digest를 계산할 때는 template revision, canonicalizer version, tokenizer artifact, processor revision을 envelope header에 넣는다. hash algorithm 이름도 저장한다. digest 32 byte만 남기면 어떤 규칙으로 만들어졌는지 복구할 수 없기 때문이다.

여기서 canonicalization은 “아무 JSON이나 key 정렬”이 아니다. tool 배열 순서는 모델에게 제시되는 도구 우선순위를 바꿀 수 있고, enum 배열 순서는 template가 그대로 출력한다면 prompt byte를 바꾼다. object key는 의미상 무순서여도 template가 입력 iteration 순서를 관찰할 수 있다. 따라서 raw request identity, render identity, semantic tool identity를 분리한다. render cache는 실제 render 결과가 같아야 하므로 template가 관찰하는 순서를 보존한다. schema validation cache는 의미 동등성을 허용할 수 있지만 그 규칙을 별도 version으로 관리한다.

### 7.9.2 tool schema를 cache key에서 빼면 생기는 stale prompt

사건 시각 10:03에 `weather` 도구의 schema가 바뀌었다. 이전에는 `city`만 필수였고 새 버전은 `city`와 `unit`을 필수로 한다. gateway는 tool 이름 목록만 cache key에 넣었다. 이름이 같으므로 09:58에 렌더한 prompt가 hit 되었고, 모델은 구 schema를 본 채 `unit` 없는 call을 생성했다. validator는 새 schema로 검사해 요청을 거부했다. 겉으로는 모델의 tool-call 정확도가 갑자기 떨어진 사건처럼 보인다.

최초 불일치는 logits가 아니다. cache lookup에 사용한 identity와 validator가 사용한 schema generation이 갈린 지점이다. 조사자는 request trace에 `tool_schema_digest`, `render_cache_generation`, `validator_schema_digest`를 나란히 놓는다. 세 값 중 cache generation만 이전 값이면 stale reuse가 증명된다. 복구는 temperature를 낮추거나 retry prompt를 추가하는 것이 아니라 해당 generation을 무효화하고 cache admission에 schema digest를 포함하는 것이다.

재발 방지는 두 단계다. 첫째, cache entry가 자신을 만든 template, schema, tokenizer generation을 소유하도록 한다. 둘째, lookup 시 현재 generation과 모두 일치할 때만 hit로 인정한다. 배포 중 old/new worker가 공존하면 단일 전역 version만으로 부족하다. worker가 실제 로드한 artifact digest를 response trace에 남기고, router가 compatible generation으로만 요청을 보낸다.

### 7.9.3 multimodal placeholder와 feature cache는 같은 identity가 아니다

이미지 URL이 같다고 image input이 같은 것은 아니다. URL 뒤의 객체가 교체될 수 있고, HTTP content negotiation이 다른 bytes를 돌려줄 수 있으며, decode library와 resize/crop policy가 feature tensor를 바꿀 수 있다. 반대로 원본 byte가 같아도 template가 삽입한 placeholder 개수와 위치가 다르면 모델 입력은 다르다. 따라서 media identity를 세 층으로 나눈다.

- retrieval identity: resolved content digest, MIME type, byte length
- preprocessing identity: processor class/revision, resize·crop·normalization parameters, output shape와 dtype
- binding identity: message 내 media 순서, placeholder token 위치, feature slot mapping

사건 R1은 이미지 두 장 `[A,B]`, R2는 `[B,A]`를 갖는다. content digest의 정렬된 set만 key로 쓰면 둘은 충돌한다. feature tensor 자체는 두 개 모두 cache에 있어도 binding 순서는 달라야 한다. 모델이 첫 번째 placeholder를 A로 해석해야 하는데 B feature가 연결되면 오류 없이 전혀 다른 답을 낼 수 있다. 그래서 binding identity에는 ordinal과 message/content index가 필요하다.

또 다른 사건에서는 placeholder 하나가 truncation으로 사라졌지만 feature 두 개가 그대로 engine에 전달되었다. template cache는 truncation 이전 rendered bytes를 재사용했고, processor cache는 image 둘의 feature를 hit했다. 각각의 cache는 자기 층에서는 올바르지만 최종 결합 계약이 깨졌다. admission 직전에 `placeholder_count == feature_slot_count`와 ordinal bijection을 확인해야 한다. 불일치가 나면 조용히 마지막 feature를 버리지 말고 request를 거부하고 compile trace를 보존한다.

**Hash collision보다 흔한 것은 identity omission이다.**

운영 회의에서 “SHA-256 collision 가능성”이 먼저 언급되곤 한다. 그러나 실제 위험은 암호학적 충돌보다 key 입력에서 중요한 필드를 빼먹는 논리 충돌이다. `add_generation_prompt`, reasoning mode, tool choice, tokenizer revision, processor revision 중 하나라도 누락되면 서로 다른 실행이 같은 key로 합쳐진다. 이를 collision이라고만 부르면 hash algorithm 교체라는 잘못된 처방으로 이어진다.

두 종류를 구분한다. hash collision은 canonical input bytes가 다른데 digest가 같은 경우다. identity collision은 정책상 달라야 할 요청이 canonical input 단계에서 이미 같아진 경우다. stale hit는 key는 과거 규칙상 맞지만 entry generation이 현재 consumer와 호환되지 않는 경우다. 관측에는 `canonical_input_length`, 각 field digest, combined digest, canonicalizer version, producer generation, consumer generation을 남긴다. combined digest만으로는 세 사건을 구분할 수 없다.

보안 때문에 원문 schema와 prompt를 로그에 남길 수 없다면 조사 가능성을 포기할 필요는 없다. field별 keyed digest, byte length, element count, sensitivity class, 보존 기간을 남긴다. 제한된 incident vault에는 암호화된 원문을 짧게 보존하고 접근을 감사한다. 일반 metric에는 digest 전체를 label로 넣지 않는다. cardinality 폭발과 정보 노출 위험이 있으므로 trace/span attribute나 표본화된 사건 레코드로 보낸다.

**Cache hit의 이득과 검증 비용을 함께 계산한다.**

template render가 요청당 0.35 ms, tool schema serialization이 0.20 ms, media preprocessing이 14 ms라고 하자. 모든 것을 하나의 cache에 묶으면 media가 바뀔 때 값싼 render까지 miss하고, template만 바뀌어도 비싼 feature를 다시 만든다. 층별 cache가 필요한 이유다. render cache는 template/context identity를, token cache는 rendered bytes와 tokenizer identity를, media feature cache는 content와 processor identity를, prefix cache는 최종 token IDs와 model/KV 호환성을 사용한다.

초당 1,000 요청에서 render hit 80%라면 절약 상한은 `1000 × 0.8 × 0.55 ms = 440 CPU-ms/s`다. media 요청 비율 10%, feature hit 60%라면 `1000 × 0.1 × 0.6 × 14 ms = 840 CPU-ms/s` 상당의 preprocessing을 줄인다. 하지만 lookup, digest 계산, locking이 각각 0.08 ms라면 작은 prompt의 render cache 이득은 줄어든다. 평균만 보지 말고 hit/miss별 latency와 payload 크기, contention tail을 측정한다.

정확성 gate가 성능보다 먼저다. cache disabled 결과와 enabled miss 결과와 enabled hit 결과의 compiled token IDs, feature binding, generation boundary가 같아야 한다. 그 다음에 CPU time과 TTFT가 개선되는지 본다. 정확성 parity 없이 hit rate만 높이는 최적화는 오류를 더 빠르게 재사용할 뿐이다.

### 7.9.4 incident를 재현하고 닫는 최소 fixture

fixture는 네 요청으로 충분히 작게 시작한다. F1은 tool 없음·text only, F2는 schema v1, F3는 같은 이름의 schema v2, F4는 F3에 image 순서를 뒤집은 요청이다. 각 요청을 cold cache와 warm cache로 두 번 실행한 compile trace를 비교한다. 실행 자체가 불가능한 정적 리뷰에서는 source-level expected trace를 만들고 cache key builder와 consumer가 읽는 필드를 대조한다.

판정표의 행은 request이며 열은 template digest, context digest, rendered digest, tokenizer digest, token digest, media content digest, processor digest, binding digest, cache result, producer generation이다. F2와 F3의 tool digest가 같으면 schema omission이다. F3와 F4의 binding digest가 같으면 ordering omission이다. cold와 warm의 token digest가 다르면 cache payload 또는 post-cache pass가 비결정적이다.

rollback은 cache 전체 삭제로 끝내지 않는다. 어떤 generation과 field가 잘못되었는지 알아야 다음 배포에서 선택적으로 무효화할 수 있다. 종료 조건은 old entry 격리, 새 key version 배포, four-fixture cold/warm parity, mixed worker generation에서의 route 검증, stale-hit metric 0, 두 관측 window 동안 validator rejection 회복이다. 실패 artifact 하나는 regression corpus에 영구히 남긴다.

이 사건이 주는 가장 중요한 교훈은 template cache가 문자열 memoization이 아니라 compiler artifact cache라는 점이다. 입력 언어와 compiler version과 downstream consumer contract가 identity에 들어간다. tool과 media가 추가되면 key가 길어지는 것이 문제가 아니라, 어떤 변화가 어느 cache 층을 무효화해야 하는지 설명할 수 있어야 한다.

## 7.10 변경 검토를 한 장의 운영 판정으로 압축한다

마지막으로 개발자와 운영자가 같은 증거를 보도록 review sheet를 만든다. 첫 칸에는 변경된 파일 목록보다 의미 변화부터 쓴다. template selection 규칙이 바뀌었는지, context에 새 변수가 들어왔는지, whitespace와 special token 출력이 달라졌는지, tool/media binding이 달라졌는지, tokenizer 호출 flag가 바뀌었는지를 적는다. 그 다음에 그 의미를 구현한 pinned symbol과 caller, consumer를 연결한다. 코드 diff가 작아도 실행 identity가 바뀌면 큰 변경이고, 파일 diff가 커도 산출물 byte가 보존되면 serving 영향은 제한적일 수 있다.

판정의 첫 질문은 “기존 cache를 계속 읽어도 되는가”다. template byte만 바뀌었지만 모든 지원 fixture의 rendered byte가 같다면 render cache compatibility를 논의할 수 있다. 그러나 미래 입력에 대한 동등성을 증명하지 못했다면 version을 올리는 편이 안전하다. rendered byte가 같아도 tokenizer artifact가 바뀌었다면 token cache와 prefix cache는 별도 판단이 필요하다. media processor가 바뀌면 text token cache는 유지할 수 있어도 feature cache는 무효화해야 한다. 하나의 global purge 스위치는 응급 복구에는 유용하지만 정상 배포 정책을 대신하지 못한다.

두 번째 질문은 “비용과 길이가 어디서 달라졌는가”다. tool schema를 추가한 뒤 평균 prompt token이 300 늘었다면 Jinja 자체가 느려졌다고 결론 내리지 않는다. render CPU time, rendered bytes, token count, accepted prefix, scheduled prefill token을 나란히 본다. 1,000 요청에서 prompt가 300 token 늘면 총 prefill work는 300,000 token 증가한다. prefix cache가 80%를 공유한다면 실제 신규 계산은 단순 상한보다 작지만, schema가 request마다 key order나 description timestamp를 바꾸면 공유율이 무너질 수 있다. 따라서 byte 안정성과 token 비용을 함께 검토한다.

세 번째 질문은 “관측이 원인을 식별할 만큼 충분한가”다. metric에는 template version별 request count, render error, cache hit/miss, render duration histogram을 두되 raw digest를 label로 넣지 않는다. trace에는 선택된 template revision, field별 짧은 digest, rendered byte/token length, special-token owner, cache generation을 둔다. incident sample에는 접근 통제된 compile trace를 연결한다. metric은 이상을 찾고, trace는 요청을 좁히며, compile artifact는 byte와 ID의 최초 분기를 증명한다. 세 층을 한 저장소에 억지로 합칠 필요는 없다.

네 번째 질문은 “오류가 사용자에게 어떤 형태로 보이는가”다. template compile error처럼 즉시 4xx/5xx가 되는 실패는 발견하기 쉽다. 더 위험한 것은 성공 응답 안의 semantic corruption이다. 잘못된 tool schema를 본 모델이 validator에서 반복 거부되거나, image binding이 뒤집혀 그럴듯한 오답을 내거나, assistant header가 중복돼 모델이 빈 답을 내는 경우다. 성공률만 보면 놓친다. tool validation rejection, empty generation, stop-at-first-token, media binding assertion, cache-hit/cold parity 같은 증상별 신호가 필요하다.

다섯 번째 질문은 “rollback이 artifact generation을 되돌리는가”다. application image만 이전 버전으로 돌려도 shared cache가 새 generation entry를 계속 제공하면 복구되지 않는다. 반대로 cache만 비워도 old/new worker가 서로 다른 template를 로드한 상태라면 다시 오염된다. rollback manifest에는 server image, model/tokenizer revision, template digest, processor revision, cache key version, 허용 generation을 함께 기록한다. router와 worker, cache가 같은 generation fence를 보는지 확인한다.

승인 회의에서 다음 여섯 문장을 채우면 모호함이 크게 줄어든다. “이 변경의 최초 mutable boundary는 ___이다.” “그 경계의 산출물은 ___ byte와 ___ token으로 관찰한다.” “기존 entry와 호환되는 조건은 ___이다.” “호환되지 않을 때 무효화할 cache 층은 ___이다.” “사용자 영향의 earliest signal은 ___이다.” “rollback 완료는 ___ generation이 더 이상 소비되지 않을 때다.” 빈칸을 채울 수 없다면 코드가 틀렸다고 단정할 수는 없지만, 안전하게 배포할 증거가 아직 없다는 뜻이다.

최종 regression corpus는 happy path만 모으지 않는다. tool 없음과 빈 tools, 같은 tool 이름의 schema 변경, Unicode가 든 description, 아주 긴 enum, reasoning on/off, assistant 이어 쓰기, 이미지 0·1·2개, 동일 이미지의 순서 교환, placeholder truncation 직전과 직후를 포함한다. 각 fixture는 expected answer가 아니라 selected template, rendered digest, token IDs의 경계, feature binding을 기대값으로 가진다. 모델 확률 출력은 바뀔 수 있어도 compiler 계약은 더 엄격하게 고정할 수 있다.

이 review sheet의 목적은 템플릿을 영원히 고정하는 것이 아니다. 변경이 어느 표현을 바꾸고, 어느 비용을 만들며, 어느 cache를 무효화하고, 어떤 관측으로 안전성을 확인하는지 설명 가능하게 만드는 것이다. 그 설명이 있으면 새로운 모델 template를 도입할 때도 기존 서버의 편의 API에 끌려가지 않고 동일한 compiler pass로 비교할 수 있다.

독자는 실제 변경 한 건을 골라 이 판정을 역방향으로도 수행해 볼 수 있다. 먼저 user-visible symptom에서 시작해 response trace의 generation을 찾고, compiled token digest와 media binding을 확인한 뒤, render digest와 selected template까지 거슬러 올라간다. 순방향 source walk와 역방향 incident walk가 같은 최초 경계에서 만나야 한다. 만나지 않는다면 관측이 빠졌거나 서로 다른 요청을 비교한 것이다.

예를 들어 tool validation rejection이 늘었는데 token digest가 기준과 같다면 template cache를 범인으로 고정하지 않는다. validator schema generation, downstream parser, model output을 다음 경계로 넘긴다. 반대로 token digest가 다르고 render digest도 다르지만 selected template digest는 같다면 context 또는 Jinja variable을 조사한다. 이처럼 각 digest는 “정답”이 아니라 다음 조사 분기를 선택하는 표지다.

마지막 승인에는 owner와 기한도 붙인다. template artifact owner는 selection과 render fixture를, tokenizer owner는 special ID와 token parity를, multimodal owner는 feature binding을, platform owner는 cache generation fence와 metric을 책임진다. 한 팀이 모든 층을 소유하지 않아도 하나의 compile trace를 공유하면 handoff에서 증거가 사라지지 않는다. 장애 종료 뒤에는 실패 fixture, 최초 분기, 잘못된 가설, 실제 수정, rollback 기준을 짧은 레코드로 남긴다. 다음 template 변경은 그 레코드를 배포 전 반례로 다시 사용한다.

또한 여러 tenant가 request별 template를 허용하는 서비스라면 공유 범위를 명시한다. 같은 rendered digest가 나왔다는 이유만으로 tenant 경계를 넘어 cache를 공유하면 template 원문이나 tool schema의 민감한 구조가 timing과 hit 여부로 새어 나갈 수 있다. authorization scope, tenant salt, data classification을 identity policy와 함께 검토한다. 반대로 모든 entry를 tenant별로 격리하면 안전하지만 hit율과 메모리 비용이 달라진다. 이 선택은 성능 flag가 아니라 보안 계약이다.

배포 후 첫 관측 window에서는 전체 평균보다 generation별 cohort를 본다. old worker와 new worker의 render error, token length, cache hit, tool rejection, empty response를 따로 비교한다. 혼합 cohort의 평균은 한 generation의 회귀를 숨길 수 있다. canary가 종료된 뒤에도 old generation entry가 소비되는지 확인하고, 관측 가능한 최장 cache expiry와 in-flight request 수명을 지난 뒤에야 migration을 닫는다. 이 마지막 기다림까지 rollback 계획에 포함해야 한다.

## 7.11 장말 소스 노트

이 장을 닫을 때 먼저 남겨야 할 장면은 JSON이 token protocol로 바뀌는 순간이다. 같은 messages라도 selected template, kwargs, tools·media binding, generation prompt와 truncation이 다르면 rendered bytes와 token IDs가 달라진다. 답이 달라졌다면 아래 링크를 모두 여는 대신 raw request→selected template→rendered bytes→token IDs→first logits에서 첫 불일치를 먼저 고른다.

### 7.11.1 판정을 재현하는 고정 소스 좌표

본문의 인과 흐름을 끊지 않기 위해 고정 좌표를 여기 다시 모은다. 링크는 각 revision의 탐색 시작점이며 다른 version에서는 symbol과 surrounding predicate를 다시 확인해야 한다.

- Transformers 5.15.1 `550d7b3` — [`tokenization_utils_base.py:2989-3223`, `PreTrainedTokenizerBase.apply_chat_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L2989-L3223): template selection 결과를 rendering, tokenization, truncation, assistant mask로 잇는다.
- Transformers 5.15.1 `550d7b3` — [`tokenization_utils_base.py:3225-3275`, `get_chat_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/tokenization_utils_base.py#L3225-L3275): explicit/stored template 선택 경계다.
- Transformers 5.15.1 `550d7b3` — [`chat_template_utils.py:420-496`, `_compile_jinja_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/utils/chat_template_utils.py#L420-L496): sandboxed environment와 generation tracking을 준비한다.
- Transformers 5.15.1 `550d7b3` — [`chat_template_utils.py:498-590`, `render_jinja_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/utils/chat_template_utils.py#L498-L590): conversation별 rendered output과 generation span을 만든다.
- Transformers 5.15.1 `550d7b3` — [`processing_utils.py:1976-2235`, `ProcessorMixin.apply_chat_template`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/processing_utils.py#L1976-L2235): multimodal conversation과 media/processor 경계다.
- vLLM 0.27.1 `6e448d0` — [`chat_utils.py:1315-1394`, validation/loading](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/chat_utils.py#L1315-L1394): file, literal, builtin template resolution이다.
- vLLM 0.27.1 `6e448d0` — [`chat_utils.py:1954-2035`, chat parsing](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/chat_utils.py#L1954-L2035): structured content와 multimodal data를 conversation으로 바꾼다.
- vLLM 0.27.1 `6e448d0` — [`serving.py:192-330`, OpenAI chat rendering/ingress](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/entrypoints/openai/chat_completion/serving.py#L192-L330): online renderer 결과를 engine request로 잇는다.
- SGLang 0.5.18 `71de97b` — [`serving_chat.py:1280-1355`, reasoning/render/encode](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/serving_chat.py#L1280-L1355): effective kwargs, assistant prefill, render와 encode 분리다.
- SGLang 0.5.18 `71de97b` — [`protocol.py:811-844`, request template kwargs](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/entrypoints/openai/protocol.py#L811-L844): request schema의 compiler flag 경계다.
- SGLang 0.5.18 `71de97b` — [`tiktoken_tokenizer.py:106-157`, `TikTokenTokenizer.apply_chat_template`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/tokenizer/tiktoken_tokenizer.py#L106-L157): 비-Hugging Face tokenizer backend의 자체 template 경로다.
- llama.cpp `bb4caa7` — [`llama-model.cpp:2844-2868`, `llama_model_chat_template`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model.cpp#L2844-L2868): GGUF model metadata의 default/named template 조회다.
- llama.cpp `bb4caa7` — [`common/chat.cpp:753-850`, `common_chat_templates_init`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/chat.cpp#L753-L850): override, default/tool template, BOS/EOS와 template object를 결합한다.
- llama.cpp `bb4caa7` — [`common/chat.cpp:3600-3742`, Jinja apply](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/chat.cpp#L3600-L3742): tools/reasoning/generation prompt를 포함하는 Jinja 경로다.
- llama.cpp `bb4caa7` — [`common/chat.cpp:3743-3814`, legacy dispatch](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/common/chat.cpp#L3743-L3814): legacy 적용과 final Jinja/legacy 선택 경계다.

### 7.11.2 독자 회고와 다음 경계

채팅 템플릿은 JSON을 보기 좋게 펼치는 문자열 서식이 아니다. role과 content를 model이 학습한 token protocol로 낮추는 compiler다. tools와 reasoning, multimodal content는 compiler의 입력 언어를 확장하고, generation prompt는 다음 출력의 문법 상태를 정한다. assistant mask는 그 경계를 token 좌표로 되돌린다. truncation은 길이 조정이 아니라 protocol-preserving rewrite 문제다.

이 관점은 장애 조사 순서도 바꾼다. 같은 model의 답이 달라졌을 때 GPU부터 보지 않는다. raw request, selected template/kwargs, rendered bytes, token IDs, post-truncation model input, first logits 순서로 최초 불일치를 찾는다. cache, billing, security는 각각 다른 identity를 요구하지만 template artifact와 compiled token sequence를 공통 증거로 사용한다.

다음 장에서는 template가 출력한 delimiter와 BOS/EOS/PAD/UNK가 tokenizer에서 어떤 special ID가 되는지, padding과 truncation이 position/mask를 어떻게 바꾸는지 살펴본다. template compiler의 출력은 아직 model input의 완성본이 아니다. special-token protocol과 sequence-shaping pass를 통과해야 embedding lookup에 들어갈 수 있다.
