# Playbook 09. partial checkpoint

## 실행 순서

### generation 검증
1. completion marker, manifest, shard count/size/hash부터 확인한다.
2. model뿐 아니라 optimizer/RNG/sampler/scheduler state 존재를 확인한다.
3. 최신 generation이 incomplete라면 건너뛰고, 그보다 앞선 complete generation을 load한다.
4. resume 첫 batch/LR/RNG와 baseline을 비교한다.

## 분기

### 판정
- marker가 없으면 저장 미완료다. marker가 있는데 hash가 틀리면 commit protocol 실패이며, load는 성공했지만 sample이 달라지면 state coverage 실패다.

### “있다”와 “내구적이다”를 구분한다

디렉터리가 보이거나 object-store prefix에 shard가 있다고 checkpoint가 완료된 것은 아니다. generation ID와 parent CheckpointID, writer world size·rank mapping, schema version, 예상 shard 목록, logical tensor→shard 영역, byte size·checksum, 상태 범위, commit timestamp를 manifest에 넣는다. completion marker는 모든 shard가 내구적으로 기록되고 manifest 검증이 끝난 뒤에만 publish한다.

writer가 temporary generation에 shard를 쓰고, 각 rank의 byte·hash를 모은 뒤 top manifest를 쓰며, 마지막에 committed pointer를 원자적으로 바꾸는 protocol을 쓴다. object store의 listing 순서·지연에 의존해 “가장 큰 번호”를 고르지 않는다. reader는 committed pointer가 가리킨 manifest만 신뢰하고 manifest 밖 shard를 무시한다.

### 저장 매체별 원자성 가정을 명시한다

로컬 파일시스템의 같은 mount 안에서 rename이 원자적이라고 해서 분산 파일시스템이나 object store의 조건부 갱신도 같다고 가정하지 않는다. checkpoint protocol은 실제 사용하는 저장소의 `create`, overwrite, rename·copy, listing, read-after-write와 conditional put 의미를 문서로 고정한다. object copy가 끝난 뒤 원본을 지우는 방식은 중간에 두 generation이 동시에 보일 수 있다. reader가 목록 순서로 최신을 고르면 아직 commit되지 않은 대상을 읽을 수 있다.

committed pointer를 갱신할 때는 이전 pointer의 version이나 ETag를 조건으로 사용해 두 coordinator가 동시에 성공하지 못하게 한다. 실패한 writer는 새 generation을 committed로 주장하지 않고 orphan으로 표시한다. pointer 본문에는 CheckpointID와 manifest digest를 함께 넣어 같은 이름의 manifest가 사후에 바뀌는 것을 잡는다. marker 파일의 존재만 보지 않고 marker가 지시한 digest와 실제 manifest를 비교한다.

내구성 확인은 rank가 `write()`를 반환받은 순간이 아니다. host page cache, client buffer, filesystem journal, object multipart upload 가운데 어느 계층까지 완료됐는지 구분한다. 사용하는 저장소가 강한 durability 확인 API를 제공하지 않으면 completion marker가 보장하는 범위를 낮춰 쓰고, 별도 read-back과 hash 검증을 둔다. 장애 복구 문서에 없는 내구성을 추정해서는 안 된다.

암호화와 압축을 쓰면 hash 대상도 고정한다. logical tensor bytes, compressed payload, encrypted object 중 무엇을 검증하는지 명시하고 가능하면 계층별 digest를 둔다. 전송 중 payload는 정상인데 복호화 뒤 tensor layout이 틀릴 수 있다. key revision과 codec version은 checkpoint schema의 일부다.

### 저장 범위를 학습 상태 기계와 대조한다

model parameter만 있다고 학습을 정확히 resume할 수 있는 것은 아니다. optimizer moment·step counter, scheduler clock, AMP scaler, gradient accumulation 상태, RNG(CPU/GPU/rank/worker), data mixture·sampler·shard·packer cursor, pending microbatch, callback·early-stop·metric state, rollout queue·PolicyVersion을 recipe의 equality grade에 맞게 저장한다. adapter·multimodal·MoE는 base revision, projector, expert/router, tokenizer·processor를 추가한다.

각 field에는 owner, save 호출 시점, mutation 시점, load 순서, 누락 시 기본값을 기록한다. load가 성공했다는 로그는 누락 field가 default로 초기화되지 않았음을 증명하지 못한다. load 전·후 state manifest를 diff하고 허용한 migration이 아닌 missing/unexpected key를 fail closed한다.

### tensor shard와 논리 상태를 따로 검증한다

분산 checkpoint는 하나의 논리 tensor를 여러 저장 shard로 나눌 수 있고, 저장 파일 하나가 여러 tensor 조각을 담을 수도 있다. manifest에는 logical tensor name, global shape·dtype, shard axis와 half-open range, replica 여부, owner group, payload offset을 기록한다. range를 정렬했을 때 겹침과 빈틈이 없어야 하며 replicated tensor는 값 digest가 일치해야 한다. shard 파일 개수만 세면 잘못된 영역 교환을 발견하지 못한다.

optimizer는 parameter 이름이나 stable ID와 moment를 연결해야 한다. 로컬 parameter 순서에만 의존하면 model code 변경이나 wrapping 순서 변경 뒤 moment가 다른 tensor에 붙을 수 있다. 각 parameter의 global identity, shape와 optimizer group을 저장하고 load 때 현재 graph와 대조한다. tied parameter는 두 이름이 같은 storage를 가리키는지, 별도 복사로 풀렸는지도 확인한다.

FSDP·ZeRO·tensor parallel·expert parallel을 함께 쓰면 저장 shard와 실행 shard가 다르다. world size 변경은 파일을 읽는 문제가 아니라 global tensor를 새 mesh에 재분할하는 변환이다. expert의 global ID, router column, optimizer moment와 FP32 master weight가 함께 이동해야 한다. 변환 전후에 전체 합계만 비교하지 말고 ID별 canary와 첫 optimizer delta를 비교한다.

gradient accumulation 중간 저장을 지원한다면 이미 계산된 gradient buffer와 loss numerator/count, microbatch cursor를 포함해야 한다. 지원하지 않는다면 save는 optimizer boundary에서만 commit되고 pending accumulation은 재실행된다는 계약을 둔다. 일부 rank는 accumulation 3, 다른 rank는 4인 시점에 각각 저장해서는 하나의 일관된 checkpoint가 되지 않는다.

### 손상 패턴으로 commit 경계를 추적한다

marker가 없고 shard 일부만 있으면 writer 중단이지만 reader가 이를 무시하면 protocol은 정상일 수 있다. marker가 있는데 shard가 없거나 hash가 틀리면 marker가 durability 전에 publish되었거나 후속 mutation이 있었다. shard는 정상인데 manifest 영역이 겹치거나 빈틈면 distributed planner·reshard mapping을 본다.

model logits은 같지만 resume 첫 LR이 다르면 scheduler/optimizer step 소유권을 본다. 첫 batch가 다르면 sampler·worker·packer/prefetch cursor를 본다. batch는 같지만 dropout·augmentation이 다르면 RNG state와 restore 순서를 본다. 첫 forward는 같지만 optimizer step 후 다르면 moment·master weight·scaler·gradient accumulation state를 본다.

### retention과 save가 경합하는 순간을 의심한다

오래된 checkpoint를 지우는 cleanup 작업이 새 manifest를 읽기 전에 시작되면 아직 참조 중인 shard를 삭제할 수 있다. generation별 reference와 commit 상태를 기준으로 보존하고 파일 수정 시각만으로 삭제하지 않는다. pointer 교체와 retention 대상 계산 사이에는 generation fencing이 필요하다. reader가 load 중인 generation을 지우지 않도록 lease나 보수적인 보존 window를 둔다.

incremental checkpoint가 이전 generation의 unchanged shard를 참조한다면 부모 삭제가 자식 손상으로 이어질 수 있다. manifest의 dependency graph를 따라 reachability를 계산하고, 삭제 전 독립 검증을 한다. deduplicated content-addressed object는 reference count가 틀리면 여러 checkpoint가 동시에 깨질 수 있으므로 mark-and-sweep의 기준 root와 generation snapshot을 기록한다.

비동기 save에서는 학습 thread가 다음 update로 parameter를 바꾸는 동안 writer가 tensor를 읽을 수 있다. snapshot buffer, copy-on-write, stream event 또는 명시적 save barrier 중 어떤 방법으로 일관된 시점을 만드는지 확인한다. shard별 hash가 모두 맞아도 서로 다른 update의 tensor가 섞인 generation은 논리적으로 손상됐다. 저장 시점의 UpdateID를 rank별로 all-gather해 하나인지 확인한다.

## 재현·복구·파기 절차

### save 상태 기계의 모든 전이에서 장애를 주입한다

**DCP의 metadata 공개 시점과 저장소 내구성을 같은 것으로 읽지 않는다**

PyTorch DCP 고정 구현의 filesystem writer는 coordinator가 metadata를 임시 경로 `.metadata.tmp`에 쓰고 flush·`fsync`한 다음 최종 `.metadata`로 rename하는 순서로 공개한다. 관련 실패 전파 test는 writer 한 곳의 예외가 coordinator future와 호출자까지 올라오는지를 검사한다. 이 근거로 말할 수 있는 것은 선택한 filesystem 경로에서 metadata가 payload 뒤에 보이고, 일부 오류가 조용한 성공으로 바뀌지 않는다는 범위다.

다음 세 주장은 별도다. directory 자체의 `fsync`가 없으면 전원 장애 뒤 rename의 crash durability는 추가 검증이 필요하다. object store의 copy·conditional put은 filesystem rename과 같은 계약이 아니다. metadata가 보여도 model·optimizer·sampler·RNG의 의미적 coverage와 rank별 UpdateID 일치는 자동으로 성립하지 않는다.

causal triage는 `payload write → payload durability 확인 → metadata temp write → metadata flush/fsync → metadata rename 또는 commit object → reader selection → state coverage → first update` 순서로 걷는다. 각 경계에 process kill을 하나씩 넣고, reader가 볼 수 있는 generation을 미리 적는다. `.metadata.tmp`만 남은 경우를 최신 checkpoint로 고르거나, 최종 metadata가 가리키지 않는 orphan shard를 정상 state에 섞으면 실패다. timeout은 어느 단계가 deadline을 넘었다는 증상일 뿐 partial commit의 원인이 아니다. 같은 digest의 retry가 성공했는지, 두 coordinator가 서로 다른 manifest를 commit했는지부터 가른다.

upstream failure-propagation test PASS는 object-store 원자성, directory crash consistency, sampler cursor와 첫 optimizer delta를 증명하지 않는다. 이 네 범위는 미검증 경계로 운영 원장에 남기고, 실제 backend kill matrix와 28장의 next-update oracle을 통과하기 전에는 “atomic checkpoint”라고 부르지 않는다.

rank shard를 쓰는 중간, rank 보고 직후, manifest를 쓰는 중간, marker 기록 직전과 직후, committed pointer 교체 중간, retention cleanup 중간에 writer, coordinator, node를 차례로 종료한다. 다음 reader는 최신 번호가 아니라 최신 durable generation을 골라야 한다. incomplete generation은 quarantine하되 forensics가 끝나기 전에 삭제하지 않는다.

silent corruption test에서는 shard 한 byte, manifest size, tensor layout, schema version, rank mapping을 각각 바꾴 reader가 적절한 경계에서 거부하는지 본다. 누락 state test에서는 RNG, sampler, scheduler, optimizer shard를 하나씩 빼 load success만으로 통과하지 않는지 본다. 검증기가 각 mutation을 잡지 못하면 checkpoint 파일이 아니라 검증 계약을 먼저 고친다.

kill matrix는 단순히 프로세스를 종료하는 시점만 바꾸지 않는다. coordinator와 data rank를 따로 죽이고, object upload 완료 응답 유실, pointer 조건부 갱신 충돌, 느린 rank 보고, manifest truncate, cleanup 동시 실행을 주입한다. 각 주입마다 기대하는 durable generation과 orphan 목록을 미리 적는다. 시험 결과를 보고 기대값을 바꾸면 protocol의 결함을 합리화하기 쉽다.

schema migration 시험에서는 old reader가 new manifest를 명확히 거부하는지, new reader가 지원하는 old schema를 정확히 변환하는지 본다. 알 수 없는 field를 조용히 버리는 것이 안전한지 field별로 결정한다. optimizer나 sampler 의미가 바뀐 migration은 tensor shape가 같아도 trajectory 동치가 아니다. 변환 도구의 revision과 입력·출력 digest를 새 artifact로 남긴다.

### last durable generation에서 일관되게 재개한다

복구 관리자는 후보 generation을 목록화하고 pointer→manifest→shard→state coverage 순서로 검증한다. 완전한 최신 generation을 골라 모든 rank에 같은 CheckpointID·schema·topology mapping을 broadcast한다. 일부 rank만 다른 generation을 읽지 않도록 load barrier 전에 manifest digest를 all-gather한다.

재개 첫 2개 batch의 ordered sample/span ID, rendered IDs·mask, LR·scaler·optimizer counter, RNG probe, first logits·loss·gradient checksum을 uninterrupted baseline과 비교한다. 요구 등급이 sample-exact인지 numerical-tolerance인지 statistical인지 사전에 정한다. baseline이 없다면 동일 checkpoint를 독립한 두 프로세스에서 불러 첫 state를 교차하되, 이것을 uninterrupted 동치와 같다고 과장하지 않는다.

partial generation은 삭제 정책에 따라 일정 기간 격리한 뒤 파기한다. forensic manifest, missing/corrupt shard, writer log, incident timeline은 접근권한과 retention을 붙여 남긴다. 파기 전에 committed pointer가 해당 generation을 가리키지 않음을 다시 확인한다.

### 수리 가능한 상태와 폐기해야 할 상태를 구분한다

manifest만 유실됐고 모든 shard의 logical mapping을 독립된 rank report에서 재구성할 수 있어도 원래 generation을 제자리에서 수정하지 않는다. 복구 도구가 새 CheckpointID와 새 manifest를 만들고 부모로 손상 generation을 가리키게 한다. 재구성에 추정이 들어간 field는 명시하며, scheduler·sampler처럼 정확히 복원할 수 없는 상태가 있으면 허용 가능한 복구 등급을 낮춘다.

replica가 있는 tensor의 shard 하나가 깨졌다면 다른 replica에서 복구할 수 있지만, 값이 같은 logical state였다는 저장 시점 증거가 있어야 한다. parity나 erasure coding을 썼다면 복원 뒤 payload hash뿐 아니라 tensor schema와 end-to-end probe를 확인한다. optimizer shard처럼 유일한 state를 인접 rank 값이나 0으로 채우는 것은 수리가 아니라 새 학습 경로다.

복구 후보가 여러 개라면 가장 최근 번호보다 가장 최근의 완전하고 검증 가능한 generation을 고른다. 더 새 checkpoint의 model weight만 살리고 이전 checkpoint의 optimizer·sampler를 섞는 방식은 별도의 warm start다. 이를 exact resume로 표시하지 않고 새로운 run lineage와 데이터 재소비 범위를 계산한다.

서비스 배포용 weight export와 학습 재개 checkpoint도 구분한다. inference artifact가 logits를 재현하더라도 optimizer, RNG와 cursor가 없어 학습을 이어갈 수 없다. 반대로 distributed training shard는 바로 서빙할 수 있는 포맷이 아닐 수 있다. manifest의 artifact role을 확인하지 않고 서로 대체하지 않는다.

## 종료 조건

### 공급망 admission을 복구 경로 앞에 둔다

복구 checkpoint를 찾았다는 이유로 곧바로 deserialize하지 않는다. manifest가 가리키는 checkpoint·config·tokenizer·dataset cursor의 digest를 다시 계산하고, 서명과 in-toto subject가 그 digest를 가리키는지 확인한 다음 bounded parser로 넘긴다. safetensors header length·UTF-8·JSON·offset·shape arithmetic 중 하나라도 실패하면 해당 generation을 quarantine하고 이전의 완전한 generation으로 돌아간다. 서명 성공은 tensor 의미, 데이터 권리, SBOM 완전성이나 모델 행동 안전성의 대체 검사가 아니다.

복구 훈련에는 subject mismatch, truncated shard, trailing bytes, offset overlap과 거대 shape fixture를 넣는다. 각 fixture가 load 이전에 거부되고, 오류가 `signature/provenance/parser/layout` 중 정확한 단계로 기록되는지 확인한다. 실제 registry와 worker startup에서 이 mutation을 실행하지 않았다면 canonical upstream test 통과만으로 production admission을 검증했다고 표시하지 않는다.

### 통과
partial generation이 선택되지 않고 last durable checkpoint에서 요구한 sample/numerical equivalence로 재개한다.

종료 증거에는 save-state 간선별 kill 실험, corrupted/missing state mutation, reader 선택 로그, manifest·shard 검증, load 전후 state diff, uninterrupted/resume 교차 결과를 넣는다. 새 CheckpointID와 parent, 복구 등급, 영향받은 UpdateID 범위, incomplete generation 격리·파기 상태를 IncidentID에 묶는다. load API가 예외 없이 끝났거나 loss가 비슷하다는 사실만으로는 통과하지 않는다.
