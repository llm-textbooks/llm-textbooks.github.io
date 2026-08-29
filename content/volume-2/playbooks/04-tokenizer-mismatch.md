# Playbook 04. tokenizer·template mismatch

## 실행 순서

### 최초 divergence
1. 원본 UTF-8 bytes와 normalization 결과를 비교한다.
2. 렌더링된 chat bytes, token IDs, special token, padding/truncation을 비교한다.
3. embedding/head vocab size와 added-token row checksum을 확인한다.
4. trainer와 serving의 first-step logits를 greedy로 비교한다.

## 분기

### 판정
- bytes부터 다르면 template/normalizer, IDs부터 다르면 tokenizer revision, logits부터 다르면 model artifact/backend다.
- decoded text가 같다는 이유로 ID 차이를 허용하지 않는다.

### 한 prompt를 다섯 경계로 쪼갠다

비교는 입력 object에서 시작한다. message role, content type, tool schema, image·audio placeholder, generation prompt flag를 canonical JSON으로 직렬화해 hash를 만든다. 이어 template가 만든 정확한 UTF-8 bytes에서 공백, newline, role delimiter, BOS/EOS와 tool JSON key 순서를 보존한다. 그다음 tokenizer의 ID·offset·special-token mask, collator 이후 `input_ids`·attention mask·position ID·labels·loss mask, embedding output과 first logits을 차례로 비교한다. 앞 경계까지 같고 다음 경계에서 달라진다면, 그 사이의 구성 요소가 우선 조사 대상이다.

각 경계에 `TokenizerID`, tokenizer files과 chat-template checksum, library revision, option 전체, 입력·출력 shape·dtype·hash를 남긴다. 학습과 serving에서 같은 이름의 model repository를 쓰더라도 mutable branch를 읽으면 다른 tokenizer를 받을 수 있으므로 commit·artifact digest로 고정한다.

### 증상이 가리키는 실패를 구분한다

token 수가 항상 하나 많으면 BOS/EOS 중복 삽입, generation prompt, 끝 newline을 먼저 본다. 특정 Unicode에서만 다르면 NFC/NFKC, byte fallback, invalid UTF-8 replacement, 제로 폭·조합 문자를 본다. tool call에서만 다르면 template branch, JSON serialization, escaping·key order, assistant prefix를 본다. batch size에 따라 다르면 padding side, pad ID, truncation side, max length와 position 재계산을 본다.

ID는 같은데 logits이 다르다면 tokenizer 사고로 단정하지 않는다. embedding·LM head row checksum, weight tying, added-token resize 순서, adapter base revision, attention mask, position ID, cache 여부, dtype·backend를 차례로 확인한다. ID mapping은 다른데 decoded text가 같은 경우가 오히려 더 위험하다. 서로 다른 embedding row를 읽고도 표면 문자열이 같아서 parity test를 통과한 것처럼 보일 수 있기 때문이다.

### vocabulary 변경의 영향 반경을 찾는다

added token을 추가했다면 tokenizer 길이, config `vocab_size`, input embedding row, output head row, weight tying, checkpoint tensor, optimizer moment, adapter와 quantized export가 같은 새 index를 알아야 한다. row를 뒤에 덧붙였는지 기존 vocabulary를 재정렬했는지는 영향 범위가 완전히 다르다. 재정렬은 shape가 같아도 모든 row의 의미를 바꾸므로 개별 token→ID fixture와 row checksum permutation으로 검출한다.

shrink/truncate는 더 엄격히 막는다. corpus·template·adapter·checkpoint에 제거된 ID가 남았는지 reverse scan하고, 없다는 증거 없이 row를 잘라내지 않는다. quantization 후의 head padding row와 logical vocab size도 구분한다. physical row가 더 많다고 tokenizer가 그 ID를 배출해도 된다는 뜻은 아니다.

## 재현 실험과 복구

### golden fixture matrix

일반 대화, system message 없음, 빈 assistant, 여러 turn, tool call·tool result, literal special-token 문자열, emoji·조합형 한글·공백 변형, 최대 길이 경계, left/right padding batch를 포함한다. 멀티모달 model은 여러 image, audio silence, variable-length video, placeholder 수 불일치를 추가한다. 각 fixture에 exact rendered bytes, IDs, offsets, mask, labels, first-logit checksum과 기대 실패를 둔다.

학습 processor/collator와 serving renderer를 각각 실행해 첫 divergence를 출력한다. template를 변경하고 old cache를 재사용하는 실패, tokenizer file 하나만 교체하는 실패, pad side를 바꾸는 실패, added-token row를 permutation하는 실패를 주입한다. 각 경우 예상한 경계의 assertion이 model forward 전에 작동해야 한다.

임시 복구에서는 학습 때 고정한 tokenizer·template·processor artifact를 serving release에 함께 배포하고 mutable 자동 로드를 차단한다. 다만 mismatch 상태에서 이미 만든 tokenized cache, packed dataset, adapter, evaluation result는 파생 artifact까지 영향 범위를 조회한 뒤 격리해야 한다. tokenizer만 교체해 서빙을 정상화해도, 잘못된 ID로 학습한 adapter는 복구되지 않는다.

## assistant mask는 문자열이 아니라 문자 구간과 token offset의 합성물이다

### 고정 소스에서 최초 불일치 경계를 읽는다

Transformers `550d7b3834670483a4df436541272c055dc364bf`의 `src/transformers/tokenization_utils_base.py:3079-3080`은 assistant mask를 요청하려면 `tokenize=True`와 `return_dict=True`가 모두 필요하다고 강제한다. `:3113-3151`은 template가 돌려준 assistant 문자 구간을 `char_to_token`으로 token 구간에 옮기고 그 위치만 1로 채운다.

```python
start_token = out.char_to_token(i, assistant_start_char)
end_token = out.char_to_token(i, assistant_end_char - 1)
if start_token is None:
    break
for token_id in range(start_token, end_token + 1 if end_token else len(input_ids[i])):
    current_mask[token_id] = 1
```

그러므로 rendered text가 눈으로 같아 보여도 normalizer, pre-tokenizer, offset mapping, truncation 위치 가운데 하나가 달라지면 loss-bearing token이 달라진다. 즉시 보존할 상태에는 template checksum과 IDs뿐 아니라 rendered UTF-8 bytes, 문자 구간, offset mapping, truncation length, `assistant_masks`가 들어가야 한다. decoded text 비교는 이 경계를 닫지 못한다.

### 테스트 좌표를 현장 gate로 바꾼다

같은 커밋의 `tests/test_tokenization_common.py:1131-1210`은 batch와 tensor 반환에서 assistant mask shape가 `input_ids`와 같고, assistant 문자 구간에 대응한 token만 1인지 검사한다. `:1324-1425`는 assistant 응답 중간에서 truncation했을 때 남은 assistant token이 끝까지 1인지 별도로 단언한다. 이 테스트는 라이브러리의 offset→mask 계약을 닫지만, 우리 template checksum과 train/serve artifact parity까지 보장하지는 않는다.

최소 분리 실험은 같은 canonical message를 학습과 서빙 경로에 넣고 `bytes → generation 문자 구간 → IDs·offset → assistant mask`를 순서대로 비교한다. 각 단계는 exact equality를 원칙으로 한다. first logits만 backend dtype 때문에 근사 비교가 필요하며, 그 허용 오차는 golden fixture의 고정 backend·dtype별로 사전 등록한다. 처음 다른 경계 뒤의 구성 하나만 교체해 divergence가 이동하거나 사라지는지 확인한다.

안전한 복구는 tokenizer 파일 하나를 덮는 작업이 아니다. tokenizer·template·processor를 하나의 digest 묶음으로 rollback하고, 그 digest로 만든 tokenized cache와 adapter를 함께 선택한다. 수정 뒤에는 일반 대화, truncation, left/right padding, tool call, 조합형 한글 fixture가 byte부터 mask까지 exact match하고 first logits도 등록된 tolerance를 통과해야 한다. 이어 잘못된 template checksum과 offset mapping을 넣은 negative fixture가 forward 전에 실패해야 폐루프가 닫힌다.

## 종료 조건

### 통과
golden prompts에서 bytes→IDs→label mask→first logits가 정한 tolerance 내 일치한다.

통과 묶음에는 raw object에서 logits까지의 경계별 hash, train/serve option diff, tokenizer·template·processor revision, vocabulary row 대응, 음성·영상을 포함한 negative fixture 결과, 영향받은 cache·adapter·evaluation·serving release 목록을 넣는다. greedy output 문자열만 같거나 일부 prompt만 통과한 경우는 종료하지 않는다. IncidentID에 최초 불일치 경계, 수정 artifact, 회귀 gate와 rollback 담당자를 남겨 다음 tokenizer·template 교체가 같은 사고를 반복하지 않게 한다.

### byte fallback·empty target 긴급 판별표

한글 조합 문자나 emoji에서만 token 수가 달라지면 먼저 raw bytes와 normalized bytes를 나란히 둔다. 다음으로 OOV 한 문자의 모든 `<0xHH>` piece가 vocabulary에 존재하는지 검사한다. 일부 byte만 존재하면 “부분적으로 보존됐다”고 판정하지 않는다. BPE와 Unigram 모두 완전한 byte 후보 집합이 없으면 UNK 경로를 선택할 수 있으며 `fuse_unk`에 따라 offset 모양도 달라진다.

loss가 갑자기 0이거나 NaN이면 `assistant_masks.sum()`과 `labels.ne(-100).sum()`을 batch·sample별로 확인한다. chat template digest, `{% generation %}` span, truncation 직전/직후 token과 마지막 assistant content를 보존한다. 빈 conversation, 빈 assistant target, truncation으로 target이 사라진 row를 같은 “empty”로 합치지 않는다. 각각 parser-valid/render-valid/train-invalid disposition을 따로 둔다.

복구 판정에는 NFC/NFD 쌍, 일부 byte vocabulary, literal special marker, 같은 assistant content가 앞 turn에도 있는 decoy와 assistant 첫 token 좌우의 truncation fixture를 반드시 포함한다. 정상 fixture가 통과하는 것뿐 아니라 고의로 stale template·tokenizer를 넣었을 때 bytes, IDs, generation span, mask 또는 valid count의 예상 경계에서 실패해야 종료한다.
