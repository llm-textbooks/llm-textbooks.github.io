# 실습: SourceRow에서 committed UpdateID까지 한 원장으로 검산하기

이 실습은 모델을 학습하지 않는다. 여섯 개 문자열과 순수 산술만으로 데이터 선택부터 fresh-process resume까지의 identity를 검산한다. 목적은 loss 값을 잘 맞히는 것이 아니라, 장애가 났을 때 어느 ID와 분모부터 비교해야 하는지 몸에 익히는 것이다.

## 실습 계약

권위 test vector는 `research/training/source-to-commit-golden-ledger-2026-08-29.json`이다. 생성 규칙은 `scripts/build_training_source_commit_golden.py`, 독립 검증은 `scripts/verify_training_source_commit_golden.py`에 고정돼 있다. oracle은 model forward, autograd, CUDA와 distributed process를 호출하지 않는다.

입력 universe는 `row-a=ab`, `row-b="  cd "`, `row-c=ab`, `row-d=e◌́`, `row-e=xy`, `row-f=zz`다. 앞의 다섯 row만 첫 checkpoint 전에 읽고 `row-f`는 재개 뒤 첫 입력으로 남긴다.

```python
ROWS = [
    ("row-a", "ab"), ("row-b", "  cd "), ("row-c", "ab"),
    ("row-d", "e\u0301"), ("row-e", "xy"), ("row-f", "zz"),
]
normalized = " ".join(unicodedata.normalize("NFC", raw).split())
dedup_key = sha256(normalized.encode()).hexdigest()
```

`row-d`는 두 code point가 NFC의 `é`로 합쳐지고 `row-b`는 양끝과 중복 공백이 정리된다. `row-c`의 normalized digest는 `row-a`와 같으므로 `duplicate_of=row-a`로 탈락한다. raw hash와 normalized hash를 둘 다 남기기 때문에 detector가 본 동치류와 원문 bytes를 혼동하지 않는다.

## 토큰과 pack을 손으로 읽는다

tokenizer는 교육용 `fixture-utf8-byte-v1`이다. UTF-8 byte에 3을 더하고 EOS 2를 붙인다. production tokenizer를 흉내 내기 위한 것이 아니라 token ID를 누구나 독립 계산하게 만들기 위한 선택이다. revision digest가 row와 checkpoint 양쪽에 들어가므로 규칙을 바꾸고 예전 ID를 재사용할 수 없다.

survivor는 `row-a,row-b,row-d,row-e`이고 둘씩 pack한다. 각 segment의 첫 token은 이전 segment가 예측해서는 안 되므로 valid-target mask가 0이다. 나머지 byte와 EOS는 1이다.

```text
pack-0: row-a [100,101,2] mask [0,1,1]
        row-b [102,103,2] mask [0,1,1]  denominator=4
pack-1: row-d [198,172,2] mask [0,1,1]
        row-e [123,124,2] mask [0,1,1]  denominator=4
```

segment ID 없이 token만 이어 붙이면 `row-b`의 첫 token을 `row-a`의 EOS가 예측하는 cross-example target이 생긴다. labels 개수가 같더라도 mask 위치가 다르면 다른 objective다.

## rank와 accumulation 분모를 합친다

rank 0은 `pack-0`, rank 1은 `pack-1`을 소유하고 두 microbatch는 `accum-0`에 속한다. 교육용 numerator는 valid 위치 token ID의 합이다. rank별 값은 `208/4`, `300/4`다. local mean 둘의 평균을 일반식으로 가정하지 말고 numerator와 denominator를 먼저 합친다.

\[
N=208+300=508,\qquad D=4+4=8,\qquad g=N/D=63.5.
\]

parameter scalar 100.000과 learning rate 0.010의 정적 SGD oracle은 99.3650을 만든다. 이 수치는 실제 language-model gradient가 아니다. `UpdateID=update-1`이 어떤 분모·parent state·child state를 원자적으로 소유해야 하는지를 검산하는 checksum이다.

## commit 뒤에만 cursor를 확정한다

checkpoint는 `committed_update_id`, update digest, source cursor 5, next PackID 2, next UpdateID 2, dedup seen map, tokenizer revision과 parameter 99.3650을 묶는다. backward 뒤 crash에서 cursor만 5로 저장하면 update가 빠진다. optimizer step 뒤 checkpoint 전 crash에서 cursor를 되감으면 같은 update가 두 번 적용된다.

fresh process가 checkpoint digest를 읽은 뒤 기대하는 첫 상태는 다음과 같다.

```json
{
  "next_source_row_id": "row-f",
  "next_pack_id": "pack-2",
  "next_update_id": "update-2",
  "next_token_ids": [125, 125, 2]
}
```

## 실패를 한 축씩 주입한다

다음 변경을 한 번에 하나씩 적용하고 verifier가 어느 assertion에서 처음 멈춰야 하는지 예상한다.

1. NFC를 끄면 `row-d` token ID와 tokenizer-stage digest가 먼저 달라진다.
2. dedup을 raw bytes 기준으로 바꾸고 공백 변형 row를 추가하면 survivor가 달라진다.
3. 두 번째 segment 첫 mask를 1로 바꾸면 pack과 rank/global denominator가 연쇄 변경된다.
4. rank별 valid count를 2와 6으로 바꾸면 local-mean 평균과 global reference가 어긋난다.
5. checkpoint cursor를 4 또는 next UpdateID를 1로 바꾸면 fresh resume identity가 깨진다.

합격 조건은 최종 parameter 하나가 맞는 것이 아니다. 각 stage digest가 기대값과 같고, 실패 주입의 최초 불일치가 예상 stage와 같으며, fresh process의 다음 SourceRow·PackID·UpdateID가 uninterrupted ledger와 모두 같아야 한다.

## 공개 구현과 연결해 읽기

4장의 DataTrove/HF filter·dedup 좌표는 selection operator의 실제 의미를, 5장은 byte fallback과 tokenizer revision을, 6장은 packing·mixture를 제공한다. Trainer/Accelerate의 accumulation과 item-count 시험, OLMo Core의 trainer state와 model/optimizer round-trip은 뒤쪽 상태 전이를 고정한다. 이 실습은 서로 다른 upstream test가 한 production stack의 종단 parity를 자동 증명한다고 주장하지 않는다. 동일 ID universe 위에 그 계약들을 투영해 빠진 join을 드러내는 최소 oracle이다.

이제 3장의 작은 GPT loop에서는 `batch` 대신 `SourceRowID·PackID`를 출력하고, 17장의 checkpoint 표에는 `UpdateID·cursor·dedup/tokenizer state`를 추가하며, 30장의 release manifest에는 ledger digest를 포함한다. 어느 하나라도 없으면 “resume 됐다”가 아니라 “일부 tensor를 다시 읽었다”고 기록한다.

## 최초 불일치 표를 작성한다

실습 보고서에는 최종 PASS 한 줄 대신 단계별 비교표를 남긴다. source 단계에서는 `source_row_id`, raw digest, normalized digest와 decision reason을 비교한다. tokenizer 단계에서는 revision digest, token ID 길이, 처음 다른 token offset을 비교한다. pack 단계에서는 PackID, source-row 순서, segment boundary, mask bitmap과 valid denominator를 비교한다.

ownership 단계에서는 rank, microbatch ID, accumulation window와 local numerator·denominator를 비교한다. commit 단계에서는 UpdateID, parent digest, normalized gradient, child digest와 commit state를 비교한다. resume 단계에서는 checkpoint digest, source cursor, next SourceRowID·PackID·UpdateID를 비교한다.

첫 차이가 source selection에 있는데 최종 parameter만 비교하면 tokenizer와 optimizer를 불필요하게 의심하게 된다. 반대로 token과 pack이 같은데 child digest가 다르면 rank reduction, accumulation scaling, learning rate 또는 commit ordering부터 본다. next row만 다르고 parameter가 같다면 checkpoint가 sampler/dedup cursor를 빠뜨렸을 가능성이 높다. 이 분류는 대규모 trace 없이도 책임 경계를 빠르게 줄인다.

반사실 보고에는 예상 failure stage와 관측 failure stage를 함께 쓴다. 두 값이 다르면 verifier 자체가 너무 늦은 결과만 검사하는 것이다. 예를 들어 segment mask mutation을 넣었는데 update digest에서 처음 실패한다면 pack-stage denominator assertion이 빠진 셈이다. source cursor mutation이 next sample에서만 발견되고 checkpoint schema에서는 통과한다면 저장 시점 admission이 약하다. 좋은 fixture는 오류를 발견할 뿐 아니라 가능한 한 생산자 가까이에서 거부한다.

마지막으로 이 oracle의 한계를 명시한다. byte tokenizer는 BPE나 Unigram의 normalization·fallback 의미를 대신하지 않는다. scalar update는 실제 autograd, AMP overflow, optimizer moment와 parameter group을 대신하지 않는다. 두 rank record는 collective liveness나 topology를 실행하지 않는다. 이 fixture가 증명하는 것은 선택한 필드의 identity와 산술 보존이다. 실제 stack에 이식할 때는 각 단순 필드를 production artifact와 canonical upstream test로 교체하되 SourceRowID에서 next UpdateID까지의 join 구조는 그대로 유지한다.
