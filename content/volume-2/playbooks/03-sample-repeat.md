# Playbook 03. 조용한 sample repeat

## 실행 순서

### 원장 비교
1. rank별 첫 128개 `DocumentID/packed-sample ID`를 수집한다.
2. checkpoint 전후, epoch/shard 경계, worker 재시작 직후의 중복·누락을 센다.
3. sampler cursor, epoch, shuffle seed, worker/rank partition을 덤프한다.
4. 중단 없이 실행한 run과 resume한 run의 ordered ID stream을 비교한다.

## 분기

### 판정
- rank 간 동시 중복은 shard partition, resume 직후 반복은 cursor, 주기적 반복은 epoch/seed reset을 의심한다.
- loss가 정상이어도 ID repeat가 있으면 실패다.

### 중복의 단위를 먼저 고정한다

`DocumentID`만 같다고 모두 불법 중복은 아니다. 긴 문서를 서로 다른 token span으로 잘랐을 수 있고, replacement sampling이 설계의 일부일 수도 있다. 반대로 packed row ID가 달라도 같은 문서 span이 두 번 들어갔을 수 있다. 따라서 최소 식별자를 `(corpus release, shard, document, byte/token span, transform revision)`으로 정의하고, packing 후에는 구성 span의 ordered list와 separator·loss mask를 hash한 `PackedSampleID`를 더한다.

세 가지 중복률을 따로 본다. `exact packed repeat` 는 배치 전체가 반복된 경우, `span exposure repeat`는 같은 소스 span이 반복된 경우, `near duplicate exposure`는 다른 DocumentID이지만 corpus dedup cluster가 같은 경우다. incident에서는 첫 두 가지를 배송 상태 문제로 취급하고, 세 번째는 data construction 문제로 분리한다.

### 반복률을 기대 노출과 비교한다

replacement sampling이나 curriculum oversampling이 의도된 경우에도 관측 반복률이 recipe와 맞는지 검증해야 한다. source `s`의 목표 sampling mass와 전체 token budget에서 기대 exposure count를 계산하고, 실현 document·span count와 confidence interval을 비교한다. 의도된 재표본과 cursor 결함을 같은 “중복” 숫자로 합치지 않는다.

긴 문서가 여러 span으로 나뉘면 DocumentID 빈도는 높아도 span은 고유할 수 있다. 반대로 random crop이 매번 조금씩 달라 exact span hash는 다르지만 대부분의 token이 겹칠 수 있다. exact ID, token overlap과 dedup component 세 층을 본다. 반복 허용 정책에는 층마다 별도의 상한을 둔다.

데이터 품질 weight가 높은 sample을 반복하는 curriculum은 optimizer에 더 큰 영향력을 준다. 반복 횟수뿐 아니라 valid token과 loss weight를 곱한 effective exposure를 계산한다. sample이 반복됐지만 mask 때문에 loss가 0인 경우와 answer span이 매번 학습된 경우는 다르다.

rank별 stream을 합칠 때 단순히 local position으로 정렬하지 않는다. global draw sequence나 committed UpdateID·microbatch·position tuple이 필요하다. 서로 다른 rank가 같은 span을 같은 update에서 소비한 동시 중복과 여러 epoch에 걸친 의도 노출을 분리한다.

### 순서 스택을 위에서 아래로 해부한다

corpus manifest가 shard 집합과 가중치를 정한다. mixture scheduler가 다음 source를 고르고, shard permutation과 document permutation이 자료 순서를 만든다. distributed sampler가 global stream을 rank로 나누고, worker pool이 prefetch하며, packer가 여러 span을 하나의 sequence로 합친다. batcher와 gradient accumulation은 이를 UpdateID에 묶는다. 어느 계층이든 cursor를 복원하지 못하면 중복이 생긴다.

checkpoint에 epoch만 넣는 것은 부족하다. mixture schedule position, source별 shard permutation과 cursor, shuffle RNG state, rank/world size, worker assignment, worker 내 prefetch queue, packer의 미완성 buffer, batch·accumulation cursor, 최종 committed UpdateID가 필요하다. 이 중 일부를 저장하지 않는 설계라면 sample-exact resume를 주장하지 말고, 어떤 분포 동치까지 보장하는지 등급을 낮춘다.

prefetch cursor와 commit cursor를 따로 둔다. worker가 읽어 queue에 넣은 sample은 장애 전에 optimizer effect에 쓰이지 않았을 수 있다. checkpoint는 queue 자체를 저장하거나, durable commit cursor 이전 상태로 돌아가 prefetched sample을 결정적으로 재생해야 한다. 두 cursor를 하나로 쓰면 중복 또는 누락 중 하나가 생긴다.

packing buffer에는 아직 완성되지 않은 span 조각, document boundary, loss mask 상태가 남아 있다. 이를 버리고 resume하면 다음 packed sequence 구성이 바뀐다. 전체 span multiset이 같아도 predecessor context와 position, loss denominator가 달라질 수 있다. PackedSampleID와 구성 span list를 비교한다.

mixture scheduler가 loss·quality metric에 따라 동적으로 weight를 바꾼다면 controller state와 관측 window도 checkpoint한다. weights만 저장하고 누적 통계를 잃으면 resume 직후 다른 source를 고를 수 있다. source choice RNG와 global draw counter를 함께 둔다.

### 반복 패턴으로 원인 계층을 좁힌다

resume 직후 모든 rank가 같은 구간을 반복하면 global cursor를 저장한 뒤 rank-local offset을 재계산하는 경로를 본다. rank 하나만 반복하면 worker/rank partition과 elastic membership change를 본다. worker 수만큼의 주기로 반복하면 worker seed와 modulo partition을 본다. shard 크기와 같은 주기라면 shard iterator reset, epoch 경계와 같은 주기라면 `set_epoch` 호출·seed 갱신을 본다.

checkpoint 주기만큼 반복하면 asynchronous save 시점과 commit 시점을 구분한다. model state는 Update 1000 후인데 data cursor는 Update 996이면 재개 시 네 update가 재사용된다. prefetch queue를 저장하지 않고 committed cursor를 queue 뒤로 밀어 놓았다면 반대로 누락이 생긴다. cursor는 “읽을 예정”이 아니라 “loss에 committed”된 경계를 가리켜야 한다.

## 재현·복구 절차

### 반복이 모델에 들어간 영향 범위를 계산한다

중복 span을 발견했을 때는 어느 checkpoint가 그 파일을 포함하는지만 확인해서는 안 된다. 해당 span이 들어간 microbatch가 backward에 기여했고 optimizer effect까지 commit됐는지 확인한다. AMP overflow나 abort로 step이 skip됐다면 읽기 기록은 남아도 parameter에는 영향을 주지 않았을 수 있다. accumulation의 일부로 쓰였다면 해당 update 전체를 영향 범위에 넣는다.

최초 잘못된 UpdateID부터 descendant checkpoint, adapter, reward·preference data와 evaluation을 계보로 찾는다. 반복된 sample로 생성한 synthetic response도 간접 descendant다. 데이터 cursor만 고쳐 현재 run을 계속하면 이미 변한 optimizer moments와 parameter는 남는다.

영향의 크기를 loss 하나로 추정하지 않는다. repeated span의 token weight, 횟수, update 위치와 gradient contribution probe를 기록하고, clean replay child run과 selected delta·evaluation slice를 비교한다. exact rollback이 불가능하면 불확실성과 보장 등급을 명시한다.

### uninterrupted/resume 교차 실험

작은 deterministic fixture로 1,024개 span을 만들고 각 payload에 눈으로 읽을 수 있는 ID를 넣는다. 기준 run은 256개 batch를 연속 소비한다. 실험 run은 같은 초기 state에서 worker prefetch가 찬 시점, accumulation 중간, shard 경계 직전, asynchronous checkpoint 중간에 각각 종료한 뒤 재개한다. ordered `PackedSampleID` stream과 UpdateID별 구성을 byte 단위로 diff한다.

같은 world size에서 sample-exact를 약속했다면 최초 차이가 하나라도 나오는 즉시 실패다. world size를 바꾸는 elastic resume은 global ordered stream 보존, 전체 multiset 보존, 사전 정한 분포 동치 중 어떤 등급을 약속하는지 먼저 정한다. 보장하지 않는 경우는 중복·누락 상한과 mixture drift를 수치로 보고한다.

### elastic world-size 변경의 보장 등급을 선택한다

world size가 바뀌면 rank-local cursor를 그대로 복원할 수 없다. global ordered stream을 stable draw ID로 정의해 새 rank에 재분할하면 전체 순서를 보존할 수 있지만 worker·packing 병렬성 비용이 생긴다. 전체 multiset만 보존하는 설계는 순서와 packing context가 바뀌므로 sample-exact가 아니다.

통계적 resume을 허용한다면 source mixture, span exposure, length와 dedup cluster 분포의 drift budget을 사전에 둔다. 중복·누락을 평균 mixture가 비슷하다는 이유로 숨기지 않는다. 이미 committed global draw ID는 새 membership에서 다시 배정하지 않도록 fencing한다.

elastic 재개 fixture는 world size 2에서 3, worker 수 2에서 1로 바꾸고 shard 경계와 unfinished pack buffer를 포함한다. 약속한 등급에 따라 ordered stream, multiset 또는 분포 검사를 선택하며 지원하지 않는 등급을 PASS로 표시하지 않는다.

### 임시 격리와 영구 수정

중복을 발견하면 자동 resume loop를 멈추고 해당 checkpoint generation을 격리한다. cursor를 임의로 앞으로 밀면 중복을 누락 문제로 바꿀 뿐이므로 허용하지 않는다. 마지막 sample-consistent checkpoint로 rollback하고, 중복 구간이 이미 update에 반영됐다면 해당 checkpoint의 model·optimizer·scheduler state도 함께 폐기한다.

영구 수정은 state schema에 누락된 cursor/queue/packer field를 추가하고 save generation과 data commit 경계를 원자적으로 묶는다. regression test에 worker kill, rank kill, coordinator kill, incomplete checkpoint, shard rollover, world-size change를 넣는다. 각 실패 지점은 예상한 최초 gate에서 중단되고, 재개 후 ID stream이 약속한 등급을 만족해야 한다.

### 상시 탐지기를 고카디널리티 폭발 없이 설계한다

모든 DocumentID를 metric label로 내보내지 않는다. rank·source bucket별 repeat count, duplicate distance와 cursor lag를 bounded metric으로 두고, 실제 ID tuple은 접근 제어된 incident artifact에 저장한다. rolling Bloom filter나 approximate distinct counter는 조기 경보에 쓸 수 있지만 false positive·negative를 calibration한다.

canary span을 각 shard와 epoch 경계에 배치하면 iterator reset과 partition 오류를 빠르게 찾을 수 있다. canary가 실제 loss에 영향을 주지 않도록 masked synthetic metadata나 별도 test run을 사용한다. production corpus에 비밀 표식을 무단 삽입하지 않는다.

경보는 repeat rate 하나가 아니라 commit cursor 후퇴, rank overlap, prefetch-commit gap 급증과 mixture drift를 조합한다. 경보 뒤에는 exact ledger로 재검증한다. approximate detector가 조용하다는 사실을 무결성 증명으로 쓰지 않는다.

## iterator state와 optimizer commit 사이의 suffix를 검증한다

### 고정 구현이 저장하는 것과 저장하지 않는 것

TorchTitan `b482babc5f1d5d718e1719a735f9a2d86d1b9aff`의 `torchtitan/components/data/loader.py:142-174`는 thread prefetch queue를 포함한 iterator state를 rank별로 저장한다. state에는 schema version과 `dp_world_size`가 함께 들어가며, 재개 시 DP 크기가 달라졌거나 현재 rank의 state가 없으면 복원을 거부한다.

```python
return {"version": 1, "dp_world_size": self._dp_world_size,
        self._rank_id: self._iterator.get_state()}

if state_dict["dp_world_size"] != self._dp_world_size:
    raise ValueError("cannot resume after changing the effective data-parallel degree")
```

이 방어선은 rank-local iterator suffix를 잘못된 partition에 적용하는 사고를 막는다. 그러나 `get_state()`가 곧 optimizer에 반영된 commit cursor라는 뜻은 아니다. trainer가 batch를 꺼낸 뒤 accumulation 도중 죽거나 AMP가 step을 건너뛰면 consumption은 전진했지만 parameter update는 commit되지 않는다. checkpoint manifest에는 iterator state digest와 함께 마지막 완료 `UpdateID`, 그 update의 ordered `PackedSampleID`, accumulation index, optimizer·scheduler step을 기록해야 두 시계를 접합할 수 있다.

### source→packing→resume suffix의 판정 oracle

같은 커밋의 `tests/unit_tests/cpu/test_dataset_checkpointing.py:30-65`는 두 source 구현과 두 rank에서 반복 경계를 지난 뒤 state를 저장하고, 원래 iterator와 resumed iterator의 다음 8개 input·position·label tensor가 exact equality인지 검사한다. `tests/unit_tests/cpu/test_text_dataset_packing.py:89-121`은 미완성 packing state가 있는 시점에 저장한 뒤 다음 5개 packed batch가 같은지를 별도로 검사한다. chat 경로도 `tests/unit_tests/cpu/test_chat_dataset.py:266-295`에서 rank별 suffix 4개를 같은 방식으로 닫는다.

이 테스트들의 oracle은 loss 유사도가 아니라 ordered tensor suffix의 완전 일치다. 현장 fixture는 tensor equality에 sample identity를 더한다. `source span tuple → PackedSampleID → rank·microbatch → UpdateID → checkpoint generation`을 append-only ledger로 남기고, uninterrupted run과 resume run의 checkpoint 이후 suffix를 exact diff한다. sample-exact resume을 약속한 구성에서는 최초 ID·position·label·loss-mask 차이 하나가 곧 실패다.

### 최소 분리와 안전한 복구

경쟁 원인은 source shuffle state, rank partition, prefetch queue, packing buffer, consumption/commit 시계, 비원자적 checkpoint로 둔다. 같은 fixture에서 장애 지점 하나만 `batch 인출 전 → 인출 후 → backward 후 → optimizer commit 후`로 이동한다. iterator suffix는 같은데 checkpoint parameter가 다르면 data iterator보다 commit 경계 문제이고, raw span suffix는 같지만 packed tensor가 다르면 packing state 문제다. rank 하나에서만 suffix가 달라지면 rank-local state 또는 partition을 우선 본다.

복구는 cursor를 감으로 앞으로 미는 작업이 아니다. 마지막으로 model·optimizer·scheduler·iterator ledger가 같은 `UpdateID`에서 닫힌 checkpoint로 함께 rollback한다. 수정 후에는 repeat 경계와 packing 중간에서 재개한 exact suffix test, 누락 rank state와 DP 크기 변경을 넣은 rejection test, AMP skipped step과 accumulation 중간 장애 test를 통과해야 한다. world-size 변경을 지원하지 않는 구현에서 발생한 명시적 거부는 계약 준수이며, 이를 sample-exact 성공으로 바꾸어 기록해서는 안 된다.

## 종료 조건

### 통과
정책상 허용한 replacement sampling 외 중복 0, 누락 0이며 world-size 변경 시 의미론을 별도로 문서화한다.

종료 증거에는 기준·resume ordered stream diff, rank/worker/shard별 중복·누락 집계, checkpoint의 data-state schema, 실패 주입 결과, 수정 commit과 새 CheckpointID를 포함한다. 단순히 최종 loss가 비슷하거나 전체 샘플 수가 같다는 사실은 순서·소유권·span 무결성을 증명하지 못한다. 다음 교대는 IncidentID에서 문제가 처음 시작된 UpdateID, 영향받은 checkpoint 범위, 현재 활성 재개 등급과 monitor threshold를 바로 찾을 수 있어야 한다.
