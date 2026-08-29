# Golden Lab 20. PolicyVersion과 exactly-once 장애 주입

## L20.1 최소 상태 머신

### L20.1.1 저장소

작은 policy와 synthetic reward를 쓰되 상태 의미론은 축소하지 않는다. ledger는 `rollouts`, `optimizer_commits`, `weight_publications`, `checkpoints` 네 테이블로 구성한다. `rollouts.id`와 `optimizer_commits.rollout_id`에 uniqueness를 두고 commit transaction 안에서 consumed 전이를 기록한다.

## L20.2 정상 경로

### L20.2.1 version 17→18

version 17 weight hash를 replica 세 개에 확인한 뒤 네 rollout을 lease한다. `(token_ids, action_mask, old_logprobs, reward, version, seed)`를 저장한다. advantage와 policy loss의 numerator/denominator를 따로 기록하고 update commit 18을 만든다. replica별 hash ACK가 모두 온 뒤 published version을 18로 바꾼다.

## L20.3 장애 주입

### L20.3.1 duplicate delivery

같은 RolloutID를 ACK 전후 두 번 전달한다. 두 번째 전달은 조회는 가능해도 optimizer gradient 집합에 들어가면 안 된다. 최종 parameter hash가 single delivery reference와 같아야 한다.

### L20.3.2 partial publication

replica 둘만 version 18을 받은 시점에 셋째를 죽인다. published version은 17에 머물거나 healthy published set을 명시해야 한다. version 18로 표시하면서 version 17 replica가 요청을 받는 상태는 실패다.

### L20.3.3 checkpoint cut

optimizer commit 뒤 publication 전, publication 뒤 queue ACK 전 각각 checkpoint한다. resume 뒤 `committed`, `published`, `consumed`, `acked`를 재구성하고 같은 optimizer effect가 두 번 생기지 않는지 본다. 외부 environment 상태를 저장하지 않았다면 결과를 `sample-exact`라 부르지 않는다.

## L20.4 판정표

### L20.4.1 필수 assertion

- rollout의 모든 token은 하나의 PolicyVersion에서 생성된다.
- old log-prob 재계산은 고정 token/mask에서 tolerance를 만족한다.
- 하나의 RolloutID는 optimizer commit 하나에만 속한다.
- published replica의 weight hash는 published PolicyVersion manifest와 같다.
- checkpoint parent, optimizer commit, queue cursor, dedup ledger가 역행하지 않는다.
- stale discard/correction 수와 reward·길이 분포 변화를 함께 보고한다.

이 랩은 대규모 RL 성능을 재현하지 않는다. 작은 상태 시스템에서 “무엇이 정확히 한 번이어야 하는가”를 검증한다.

## L20.5 목적·전제와 증거 등급

### L20.5.1 이 실습이 검증하는 것

목표는 RL 알고리즘의 benchmark 성능이 아니라 rollout 생성, learner 소비, optimizer commit, weight publication과 checkpoint가 서로 다른 version을 섞지 않는지 검증하는 것이다. 이 상태 전이만 떼어 검증할 때는 policy network를 결정적인 작은 tensor state와 synthetic tokens/rewards로 대신해도 된다.

`manifest.json`에는 다음을 고정한다.

- policy state schema와 initial `PolicyVersion=17`, weight digest
- tokenizer/action-mask schema와 synthetic prompt IDs
- rollout worker·learner·publisher code revision
- advantage/return, policy loss와 KL 계산식
- queue/ledger database schema와 transaction isolation
- random seed/counter, lease·timeout·retry 설정
- evidence level: `Proposed`, `LocallyExecuted`, `ExternallyReproduced`

실행하지 않은 단계는 명령과 판정 규칙이 있어도 `Proposed`다. 원본 ledger snapshot, event log, parameter reports가 있어야 `LocallyExecuted`다. 실제 대규모 cluster semantics를 작은 process test가 증명한다고 쓰지 않는다.

## L20.6 ledger schema와 불변식

### L20.6.1 최소 tables

```text
policies(version PK, weight_hash, parent_version, status)
rollouts(id PK, policy_version, token_hash, mask_hash,
         old_logprob_hash, reward_hash, seed, lease_state)
optimizer_commits(id PK, parent_version, child_version UNIQUE,
                  rollout_set_hash UNIQUE, parameter_hash)
consumption(rollout_id UNIQUE, commit_id, status)
publications(version, replica_id, weight_hash, ack_state,
             PRIMARY KEY(version, replica_id))
checkpoints(id PK, commit_id, published_version, queue_cursor, ledger_hash)
```

실제 DB syntax는 선택한 engine에 맞춘다. 중요한 것은 같은 RolloutID가 두 commits에 들어가지 않고 child PolicyVersion이 한 optimizer commit에서만 태어나는 것이다. event timestamp보다 primary/foreign key와 monotonic version을 신뢰한다.

rollout lease와 optimizer consumption을 구분한다. worker/learner가 죽으면 lease는 만료되어 다시 전달될 수 있지만 committed consumption은 되돌아가지 않는다. duplicate delivery는 허용할 수 있어도 duplicate optimizer effect는 금지한다.

## L20.7 결정적 synthetic rollout

### L20.7.1 입력과 저장 상태

네 prompts와 action tokens를 작은 정수 배열로 고정한다. policy version 17의 logits 또는 직접 정의한 categorical probabilities에서 old log-probs를 계산한다. sampling을 실제로 하면 seed/counter와 generated token hashes를 저장한다.

각 rollout record에는 다음 필드를 기록한다.

```python
record = {
    "rollout_id": stable_id(prompt_id, policy_version, seed),
    "policy_version": 17,
    "prompt_ids": prompt_ids,
    "action_ids": action_ids,
    "action_mask": action_mask,
    "old_logprobs": old_logprobs,
    "reward": synthetic_reward(action_ids),
    "seed": seed,
}
```

판정 기준은 특정 reward 숫자가 아니다. 같은 manifest에서 record hash가 결정적으로 재현되고, 모든 action position의 log-prob이 하나의 PolicyVersion에서 나와야 한다. prompt/padding position은 policy denominator에서 제외한다.

## L20.8 learner update와 commit protocol

### L20.8.1 Proposed 의사코드

```python
with transaction():
    rows = lease_rollouts(expected_version=17, count=4)
    assert unique(r.id for r in rows)
    assert all(r.policy_version == 17 for r in rows)
    assert none_consumed(rows)

loss_sum, valid_count = policy_objective(rows, current_version=17)
gradient = backward(loss_sum / valid_count)
new_state = optimizer_preview(gradient)  # durable commit 전 임시 상태

with transaction():
    assert leases_still_owned(rows)
    commit = insert_optimizer_commit(parent=17, child=18,
                                     rollout_set_hash=hash_ids(rows),
                                     parameter_hash=hash_state(new_state))
    mark_consumed(rows, commit.id)
    insert_policy(version=18, parent=17, status="committed")
```

실제 optimizer mutation이 DB transaction과 동일 atomicity를 갖는다고 가정하지 않는다. 작은 lab에서는 copy-on-write `new_state`를 만든 뒤 durable commit 성공 시 active pointer를 바꾸는 방식으로 partial mutation을 피한다. production learner는 마지막 checkpoint rollback 등 별 protocol이 필요하다.

관측 항목은 rollout IDs/version, old/current log-prob difference, advantage sum/count, unclipped/clipped ratio, policy/KL loss numerator와 denominator, gradient digest, pre/post parameter hash와 optimizer state hash다.

## L20.9 publication을 quorum과 serving readiness로 분리한다

### L20.9.1 version 18 배포

publisher는 committed policy 18의 immutable artifact를 만들고 각 replica에 전송한다. replica는 load, config/tokenizer compatibility와 weight hash를 검증한 뒤 ACK한다. global `published_version=18`은 manifest가 정한 all-replica 또는 quorum 조건을 만족한 뒤에만 commit한다.

quorum을 허용하면 published set과 routing layer가 version 18 ACK replicas에만 requests를 보내야 한다. version 17 replica가 healthy하더라도 version 18 rollout pool에 섞이면 안 된다. replica별 readiness metric과 request PolicyVersion을 기록한다.

rollout에는 worker가 요청한 version뿐 아니라 실제 serving replica가 확인한 weight hash를 저장한다. version 번호 재사용을 금지한다. rollback은 새 policy event 또는 명시 pointer transition이며 과거 17 artifact를 19처럼 식별할지 protocol을 정한다.

## L20.10 장애 주입 전체 행렬

### L20.10.1 주입과 최초 detector

| 장애 | 주입 지점 | 통과 판정 |
|---|---|---|
| duplicate before ACK | queue redelivery | consumption uniqueness, 동일 final hash |
| duplicate after commit | stale message | 이미 consumed로 조회만 하고 skip |
| learner crash pre-commit | gradient 계산 뒤 | commit/consumption 없음, 안전 재시도 |
| learner crash post-commit | DB commit 뒤 ACK 전 | 같은 commit 재확인, 재-update 없음 |
| partial replica load | publication 중 | published pointer 유지 또는 healthy set 제한 |
| wrong weight hash ACK | replica | publication 거부 |
| version 17/18 mixed rollout | router | rollout seal validation 실패 |
| stale old-logprob | learner | version/hash parity 실패 |
| checkpoint pre-publication | cut A | committed 18, published 17 복원 |
| checkpoint post-publication | cut B | published 18과 ACK set 복원 |
| queue cursor ahead | corrupted checkpoint | load validation 거부 |
| dedup ledger missing | corrupted checkpoint | exact-once 등급 거부 |

각 failure는 정상 database와 artifacts를 복사한 child run에 주입한다. process가 죽었다는 사실만 확인하지 말고, restart 뒤 ledger와 parameter hash가 reference와 맞는지 검사한다.

## L20.11 stale rollout 정책과 판정

### L20.11.1 discard·importance correction

current learner version이 rollout version보다 앞설 때 discard, bounded lag, off-policy correction 중 정책을 고정한다. 이 실습은 어느 정책이 최상인지 결정하지 않는다. 실제 objective가 manifest의 정책과 같은지 검증한다.

discard이면 discarded RolloutID가 optimizer set에 없고 discard reason/counters가 있다. correction이면 old/current log-probs, ratios, clipping과 mask를 독립 재계산한다. version 차이 숫자만으로 correction을 가정하지 않는다.

stale 처리 전후 reward·length·domain 분포를 비교한다. 느린/긴 rollout이 더 자주 discard되면 realized RL data distribution이 바뀐다. stale rate 하나만 보고하지 않는다.

## L20.12 checkpoint·resume 절차

### L20.12.1 두 cut에서 재구성한다

checkpoint에는 policy/optimizer states, committed/published versions, replica ACK set, queue cursor, lease/consumption ledger digest, RNG/controller state를 넣는다. external queue snapshot을 보존하지 못하면 지원 resume 등급을 낮춘다.

resume은 다음 순서로 한다.

1. checkpoint schema와 모든 referenced artifact hashes 검증
2. optimizer commit과 policy parent/child chain 복원
3. publication pointer와 replica readiness 재검증
4. consumed RolloutIDs와 queue cursor/dedup 복원
5. outstanding leases를 정책에 따라 expire/reconcile
6. next rollout·commit IDs와 parameter hash를 reference와 비교

cut A에서는 policy 18이 committed지만 published pointer가 17일 수 있다. resume 후 18을 재학습하지 않고 publication만 재개해야 한다. cut B에서는 18이 이미 published이며 queue ACK 재전송이 optimizer duplicate를 만들지 않아야 한다.

## L20.13 관측 지표와 실패 분기

### L20.13.1 metrics

항상 기록할 counters/gauges는 generated/leased/consumed/discarded rollouts, duplicate deliveries, commits, committed/published version, replica hash disagreement, version lag, stale rate, valid action count와 reward/length buckets다.

PolicyVersion과 RolloutID를 무제한 metric labels로 넣지 않는다. current version은 gauge, 상세 IDs는 trace/ledger artifact에 둔다. commit과 publication latency, queue age는 histogram으로 본다.

parameter hash가 reference와 다르면 rollout set과 loss/gradient부터 본다. parameter는 같고 publication이 다르면 ACK/quorum/pointer를 본다. resume 뒤 rollout IDs가 다르면 queue/sampler state를 본다. IDs는 같고 old log-probs가 다르면 replica weight/config를 본다.

## L20.14 artifact와 종료 조건

### L20.14.1 보존 파일

`manifest.json`, initial/final policy states, rollout JSONL, DB schema와 before/after snapshots, transaction/event trace, optimizer report, publication report, checkpoint manifests, failure별 child reports를 보존한다. 민감 prompts는 opaque IDs와 접근 통제 artifact로 분리한다.

종료 조건은 다음과 같다.

1. 정상 17→18 commit/publication이 ledger에서 재구성된다.
2. 네 rollouts가 정확히 한 optimizer commit에만 기여한다.
3. duplicate와 두 crash cuts가 reference parameter hash를 유지한다.
4. partial publication에서 mixed-version serving이 차단된다.
5. checkpoint 두 cuts가 committed/published/consumed/acked를 정확히 복원한다.
6. stale 정책의 실제 sample 분포 영향이 보고된다.
7. 모든 단계의 evidence level과 미실행 범위가 명시된다.

대형 rollout cluster, 실제 reward model 성능과 네트워크 장애를 실행하지 않았다면 그대로 `NOT_RUN`이다. 작은 lab의 완료는 상태 의미론과 detector가 검증되었다는 뜻이지 production RL의 전체 성공을 뜻하지 않는다.

## L20.15 독립 재현과 evidence 승격

다른 검토자는 정상 ledger와 failure child 하나를 받아 final policy hash를 독립 계산한다. 작성자의 in-memory 객체나 설명이 필요하면 artifact가 불완전하다. SQL/query 결과와 raw event trace에서 RolloutID→CommitID→PolicyVersion을 왕복해야 한다.

`LocallyExecuted` 승격에는 실제 실행 command, database engine/version, source digest, exit codes와 생성 files의 hashes가 필요하다. 일부 failure만 실행했다면 matrix cell별로 evidence level을 둔다. 전체 표를 하나의 PASS로 올리지 않는다.

새 queue나 DB, rollout backend로 바꾸면 exactly-once claim을 자동 승계하지 않는다. lease, transaction, ACK와 checkpoint semantics를 다시 매핑하고 같은 duplicate/crash suite를 실행한다. 허용한 at-least-once delivery와 금지한 duplicate optimizer effect는 끝까지 구분한다.

마지막 보고서에는 정상 결과보다 실패가 어디서 차단되었는지를 같은 비중으로 남긴다. detector가 최종 parameter divergence 뒤에야 울렸다면 상태 gate가 늦다. mutation/commit 전에 차단하도록 protocol과 test를 보강한 뒤 실습을 봉인한다.

## L20.16 고정 소스에서 dry-run oracle을 복원한다

이 실습의 직접 출발점은 AReaL commit `94ce1655…`의 [`remote_inf_engine.py:966-1016`](https://github.com/areal-project/AReaL/blob/94ce16558b31ebf114f1d6d469e58e3af6d7ea59/areal/infra/remote_inf_engine.py#L966-L1016)이다. 함수는 generation segment를 보내기 직전에 `request_version = self.get_version()`으로 version을 고정하고, 응답 token 수만큼 같은 값을 `accumulated_versions`에 붙인다. 따라서 synthetic fixture의 예상 상태는 token IDs `[u₀,u₁,u₂]`, old log-probs `[ℓ₀,ℓ₁,ℓ₂]`, version vector `[17,17,18]`처럼 길이가 모두 같은 세 열이다.

코드를 실행하지 않는 dry-run에서는 두 segment 사이에 current version을 17에서 18로 바꾸고, 각 request payload·response token·누적 version 열을 종이에 전개한다. 최종 current version 18을 세 token에 덮어쓰는 변형은 세 번째 열이 `[18,18,18]`이 되는 순간 실패한다. 최초 불일치는 reward나 parameter가 아니라 첫 segment의 token-version pair다. 그 뒤에만 stale 판정, importance ratio와 optimizer contribution을 조사한다.

pass oracle은 token·log-prob·version 세 열의 길이가 같고 segment boundary와 version 변화가 일치하며, duplicate delivery가 같은 RolloutID의 두 번째 optimizer contribution을 만들지 않는 것이다. 실패하면 먼저 위 고정 함수의 version pin 위치와 backend payload를 읽고, 다음으로 AReaL `tests/test_functional.py:11-85`의 PPO actor-loss sequence fixture에서 token mask·old log-prob 축을 대조한다. 이 두 좌표가 설명하는 것은 segment 귀속과 loss 배열 계약이지 DB transaction의 crash atomicity가 아니다.
