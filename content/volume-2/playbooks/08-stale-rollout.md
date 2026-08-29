# Playbook 08. stale rollout

## 실행 순서

### version 원장 확인
1. prompt dispatch, token span, queue enqueue/dequeue, optimizer commit에 `PolicyVersion`을 기록한다.
2. `current-behavior` age histogram, stale discard, queue wait, rollout throughput을 함께 본다.
3. mixed-version prefix/suffix와 old logprob가 있는 trajectory를 격리한다.
4. queue size와 freshness threshold를 각각 하나씩 바꾼다.

## 분기

### 판정
- age가 queue에서 늘면 backpressure를 조사한다. rollout 도중 늘면 긴 environment interaction을, publication 뒤 replica마다 달라지면 weight sync를 조사한다.
- discard를 0으로 만들기 위해 threshold만 키우면 해결이 아니다.

### version이 무엇을 고정하는지 정의한다

`PolicyVersion`은 사람이 임의로 붙인 배포 이름이 아니라 trajectory의 확률을 결정하는 artifact 묶음의 해시여야 한다. base/checkpoint·adapter weight, tokenizer·chat template, generation options, sampling RNG 규칙, reward/constraint policy가 포함된다. weight는 같아도 temperature·top-p·stop sequence가 다르면 behavior distribution이 다르므로 다른 version으로 다룬다.

trajectory에는 prompt ID, dispatch·generation-start·generation-end·queue-enqueue·learner-dequeue 시각, behavior `PolicyVersion`, token ID, token별 old logprob, mask, reward/advantage 산출 version을 남긴다. learner의 current version과 behavior version의 차이를 optimizer update 수, wall-clock, KL/log-ratio drift로 각각 표현한다. “2 version old”는 update 크기가 다르면 같은 의미가 아니다.

### age를 세 좌표로 측정한다

behavior policy를 `\pi_b`, learner가 update에 사용하는 current policy를 `\pi_c`라 하자. token `a_t`의 importance ratio는 `r_t=exp(log \pi_c(a_t|s_t)-log \pi_b(a_t|s_t))`다. PolicyVersion 간 update 수가 작아도 한 token의 log-ratio가 클 수 있고, update 수가 커도 실제 분포 변화가 작을 수 있다. 따라서 version distance, wall-clock age와 token별 log-ratio·prompt-level KL을 함께 본다.

trajectory age는 generation, queue 대기, learner의 batch 조립 구간으로 나눈다. `A_total=A_generation+A_queue+A_assembly`로 기록하면 threshold 초과만 보는 대신 어느 제어면이 시간을 만들었는지 알 수 있다. 긴 environment interaction은 generation age가 크고, 느린 learner는 queue age가 크다. 두 경우의 수정은 다르다.

age budget은 prompt family와 horizon에 따라 다를 수 있지만 사후에 유리한 범주만 늘리지 않는다. calibration run에서 policy update norm, ratio tail, KL과 품질 변화를 연결해 사전에 정한다. threshold 바깥 trajectory는 무조건 나쁘다는 뜻이 아니라 현재 estimator의 보장 범위를 벗어났다는 뜻이다.

### mixed-version trajectory를 fail closed한다

replica가 generation 중에 weight를 교체하면 prefix는 version A, suffix는 version B일 수 있다. request 시작에 immutable model handle을 pin하고 종료 후에 해제한다. 교체가 in-place copy라면 tensor 일부만 새 weight인 순간도 있으므로 staging→digest 검증→atomic pointer swap의 publication protocol을 쓴다.

old logprob는 behavior policy가 실제로 샘플링한 같은 token, mask, temperature과 vocabulary에서 얻어야 한다. current policy로 재계산한 logprob를 old로 저장하거나, tokenizer mismatch로 token span을 바꾸면 importance ratio가 의미를 잃는다. trajectory의 version·token·old-logprob checksum을 queue 입구와 learner 입구에서 둘 다 검증한다.

tokenizer와 template는 weight sync 바깥의 독립 상태다. replica가 새 tokenizer를 먼저 받거나 stop rule만 바뀌면 같은 PolicyVersion 이름 아래 다른 token sequence가 생길 수 있다. prompt의 rendered bytes, input IDs, completion IDs, mask와 stop reason을 version bundle에 넣고 queue consumer가 다시 tokenize하지 않게 한다.

reward와 constraint에도 버전이 있다. behavior rollout 뒤 reward model이 바뀌면 같은 trajectory의 advantage가 달라질 수 있다. reward를 enqueue 전에 계산했는지 learner에서 계산했는지, RewardVersion과 normalization window를 기록한다. policy freshness와 reward freshness를 하나의 age로 합치지 않는다.

tool execution trajectory는 observation 자체가 외부 상태에 의존한다. environment revision, tool schema, response digest와 retry를 포함하지 않으면 같은 prompt·policy라도 다른 상태 전이를 old logprob와 결합할 수 있다. policy ratio는 action distribution을 보정할 뿐 environment drift를 보정하지 않는다.

### age가 늘어나는 구간을 시간축으로 찾는다

prompt dispatch 전에 늘면 scheduler backlog이다. generation 중에 늘면 long-tail output, tool/environment latency, slow replica와 retry를 본다. enqueue 후에 늘면 queue capacity, priority, learner throughput과 backpressure를 본다. learner가 dequeue했지만 update에 들어가기 전 늘면 minibatch assembly, reward inference, advantage computation을 본다. publication 직후 replica별 digest가 다르면 rollout 속도가 아니라 weight distribution protocol이 원인이다.

age 평균은 긴 tail을 숨긴다. prompt family, environment, output length, replica, queue partition별 p50/p95/p99와 discard rate를 본다. discard된 trajectory의 reward·length·domain 분포도 남긴다. 긴·어려운 trajectory만 stale로 버려지면 accepted batch의 분포가 쉽게 바뀌고 학습 bias가 생긴다.

### queue의 전달 의미를 명시한다

at-least-once queue는 consumer 재시작 때 같은 trajectory를 두 번 줄 수 있다. exactly-once처럼 보이게 하려면 TrajectoryID와 learner commit ledger로 deduplicate한다. dequeue acknowledgment를 optimizer commit 전에 보내면 장애 때 sample을 잃고, commit 뒤 너무 늦게 보내면 재전달될 수 있다. queue cursor와 UpdateID의 transaction 경계를 정한다.

priority queue는 freshness를 개선할 수 있지만 prompt distribution을 바꾼다. 짧은 trajectory나 높은 reward가 먼저 소비되면 FIFO와 다른 objective가 된다. priority key와 tie policy, starvation 한계를 기록하고 accepted distribution을 target prompt measure와 비교한다.

backpressure가 없다면 rollout worker가 learner보다 빠를 때 age는 필연적으로 증가한다. queue byte·token·trajectory 상한, producer lease와 admission 정책을 둔다. full queue에서 새 trajectory를 버릴지 producer를 멈출지 결정하며, 버린다면 어떤 분포가 손실됐는지 원장에 남긴다.

rollout retry가 같은 prompt에 새 RNG를 쓰면 두 trajectory가 생길 수 있다. 첫 시도가 실제로 실패했는지, 늦게 완료돼 duplicate로 들어왔는지 구분한다. prompt attempt ID와 generation lease를 두고 expired attempt가 queue에 들어오면 격리한다.

## 제어 실험과 복구

### freshness–throughput–bias 교환을 분리한다

queue size를 줄이면 freshness는 좋아지지만 rollout worker가 idle하거나 learner가 input starvation을 겪을 수 있다. threshold를 낮추면 importance-ratio tail은 줄지만 discard bias와 wasted rollout compute가 커진다. learner update batch를 줄이면 publication이 자주 일어나 sync 비용이 커질 수 있다. 각 변경을 하나씩 적용하고 accepted/discarded distribution, token/s, learner idle, rollout idle, KL·ratio clip fraction, reward·safety metric을 함께 본다.

freshness threshold는 결과를 본 뒤 늘리지 않는다. 고정 fixture에서 behavior/current logprob difference와 policy update norm을 바꾸어, age가 실제 ratio·KL 위험을 어느 한계로 예측하는지 먼저 calibration한다. update count만으로 부족하면 실현 KL, importance weight, checkpoint distance를 gate에 함께 쓴다.

### estimator가 stale data를 어떻게 쓰는지 검산한다

PPO-style clipped objective는 ratio가 clip 범위를 넘으면 일부 방향의 gradient를 제한하지만 arbitrary staleness를 안전하게 만드는 장치는 아니다. clip fraction, unclipped ratio quantile과 sign별 advantage를 본다. ratio가 모두 clip boundary에 몰리면 rollout compute가 있어도 유효 update 정보가 줄어든다.

reference-policy KL과 behavior-current KL은 다른 양이다. reference는 정책 drift 규제의 기준이고 behavior는 importance correction의 생성 분포다. 세 policy ID를 명시하고 logprob cache가 어느 policy에서 왔는지 구분한다. 이름이 비슷하다고 reference logprob를 old behavior logprob로 재사용하지 않는다.

advantage normalization을 accepted trajectory만으로 다시 계산하면 stale discard가 평균·분산을 바꾼다. normalization population과 distributed numerator/count, 빈 rank 처리를 기록한다. 동일 raw queue에서 threshold만 바꾼 두 run의 accepted measure와 gradient estimand가 어떻게 달라지는지 작은 fixture로 계산한다.

### 장애 주입과 rollback

**고정 upstream oracle로 version 귀속과 freshness 허용을 분리한다**

AReaL 고정 revision `94ce16558b31ebf114f1d6d469e58e3af6d7ea59`의 race fixture는 요청이 version 10에서 시작된 뒤 응답 처리 중 engine이 11로 바뀌어도 세 생성 token의 `output_versions`가 `[10,10,10]`인지 직접 검사한다. 이 test가 고정하는 것은 요청 시작 version의 **귀속**이다. version 10 rollout을 learner version 11이 받아도 되는지, replica가 실제로 version 10 weight로 계산했는지, publication이 원자적이었는지는 증명하지 않는다.

따라서 stale incident에서는 다음 순서를 바꾸지 않는다.

1. `len(output_tokens)==len(output_logprobs)==len(output_versions)`인지 검사한다. 다르면 trajectory serialization 경계에서 멈춘다.
2. 요청 중 engine version을 한 번 전진시켜 모든 token이 요청 시작 version을 유지하는지 본다. prefix/suffix version이 섞이면 queue age를 조사할 단계가 아니다.
3. `old_logp`, current `logprobs`, token별 `versions`를 같은 shape로 놓고 proximal 보간값을 손계산한다. 공개 수치 test는 단일·혼합 version 배치의 forward 값을 고정하지만 gradient와 허용 lag를 고정하지 않는다.
4. 결손 completion이 있으면 실제 `group_sizes`로 advantage population을 다시 자른다. 고정 group 크기를 쓰면 서로 다른 prompt의 reward가 섞인다.
5. 그 뒤에야 generation·queue·assembly age를 나누고 admission threshold를 적용한다.

age와 identity는 다른 변수다. version tag가 틀린 trajectory는 age가 0이어도 무효이고, tag가 정확한 오래된 trajectory는 estimator 정책에 따라 격리·보정·폐기할 후보이지 자동 corruption은 아니다. timeout이나 queue wait을 줄여 discard가 사라진 사실도 policy freshness의 증명이 아니다. 귀속 test PASS, proximal forward PASS, admission policy PASS, optimizer-update PASS를 네 칸으로 나누고 마지막 두 칸에는 프로젝트의 직접 assertion이 없으면 `미검증`으로 둔다.

의도적으로 한 rollout replica의 sync를 지연시키고, queue consumer를 멈추고, 긴 environment response를 만들고, generation 중 publication을 시도한다. 각 장애에서 version mismatch metric, age histogram, mixed-version assertion, backpressure, discard/quarantine가 예상 순서로 발생해야 한다. 조용히 current version으로 태그를 바꾸거나 old logprob를 재계산해 살리면 실패다.

stale batch가 이미 optimizer에 반영됐다면 영향받은 UpdateID와 여기서 파생된 PolicyVersion을 격리한다. 정책이 sample-exact rollback을 요구하면 마지막 정상 checkpoint·queue/data cursor로 돌아간다. statistical recovery만 가능하면 discarded/accepted 분포 차와 추가 evaluation을 명시하고 더 낮은 보장 등급으로 재개한다.

publication은 모든 replica가 새 artifact를 staging하고 digest를 보고한 뒤 coordinator가 같은 generation을 commit할 때만 활성화한다. 일부 replica가 준비되지 않았다면 old generation을 유지하거나 해당 replica를 격리한다. 요청 단위 pinning을 통해 generation 중 정책이 바뀌지 않게 한다.

### 복구 후 첫 update를 비교한다

rollback checkpoint에는 policy·optimizer·scheduler뿐 아니라 queue cursor, accepted/discarded ledger, normalization window, replica generation과 RNG가 들어가야 한다. queue만 이전 상태로 돌아가거나 learner만 앞으로 남으면 trajectory가 중복 또는 누락된다. CheckpointID와 QueueSnapshotID를 하나의 commit으로 다룬다.

재개 뒤 첫 batch의 TrajectoryID, behavior version, token·mask, old logprob, reward·advantage와 denominator를 정상 대조 실행과 비교한다. 첫 parameter delta가 같아야 sample-exact 복구를 주장할 수 있다. queue를 비우고 새 rollout만 받는 선택은 안전할 수 있지만, 버린 분포와 compute를 기록한 별도 child run이다.

장기 회귀에는 빠른 prompt와 긴 tool trajectory, sync가 느린 replica, learner pause, queue consumer restart와 publication 직전·도중 장애를 넣는다. 각 경우 mixed version은 0이어야 하며, stale discard와 retry duplicate가 예상 detector에서 잡혀야 한다.

## 종료 조건

### 통과
학습 batch의 모든 trajectory가 허용 age와 version/logprob 계약을 만족하고 replica artifact digest가 일치한다.

종료 묶음에는 trajectory lifecycle 시간축, accepted/discarded age·domain·length 분포, behavior/current logprob·KL, replica별 artifact digest, queue·publication 장애 주입, 영향받은 UpdateID·PolicyVersion을 넣는다. 동일 prompt·RNG fixture에서 pin된 behavior version의 token·old logprob이 재현되고 mixed-version mutation이 queue 입구 전에 실패해야 한다. 장기 run에서 age tail과 discard bias가 budget 안에 있으며 reward·safety·throughput 회귀가 없을 때만 IncidentID를 닫는다.

global valid count가 0인 batch와 rank별 valid count 1:3 batch를 추가한다. 전자는 어떤 update도 commit하지 않아야 하고, 후자는 concatenated single-process reference와 gradient·optimizer delta가 맞아야 한다. singleton group, mixed RewardRevision과 stale PolicyID도 각각 별도 거부 사유로 기록한다. 이 fixture는 설계만으로 PASS가 되지 않으며 실제 multi-rank run log와 no-commit 증거가 있어야 닫힌다.

로컬 상태 기계의 기대 동작은 `python scripts/verify_p03_static_oracle.py`로 먼저 확인한다. 이 검사는 정상 계약 10개와 mutation 10개를 실행해 고정 ledger digest를 비교한다. 여기서 PASS해도 live queue, NCCL collective나 분산 optimizer가 검증된 것은 아니다. 운영 IncidentID를 닫으려면 같은 mutation을 실제 runtime 경계에 주입한 별도 증거가 필요하다.
