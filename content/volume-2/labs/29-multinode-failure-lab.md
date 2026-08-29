# Lab 29. 멀티노드 확장과 장애 주입

실제 클러스터에서 실행하기 전에는 `설계 검토 완료/실행 예정`으로 표시한다. 장애 명령은 격리된 실험 job에서만 사용한다.

## 기준 run

### topology와 비용표

rank→node→GPU→NUMA→NIC, DP/TP/PP/EP group, local/global batch, gradient accumulation을 manifest에 쓴다. 예상 collective bytes와 실제 profiler bytes를 비교한다.

### 정상 fingerprint

20 step 동안 rank별 batch ID, loss denominator, step time decomposition, collective sequence hash, memory peak와 checkpoint latency를 저장한다.

### 실행 등급과 안전 승인

각 실험은 `Proposed`, `StagingExecuted`, `ExternallyReproduced` 등급을 갖는다. 실제 command·timestamp·raw log·exit code·metric·artifact가 없으면 `Proposed`다. 장애 주입 대상은 전용 cluster/job/namespace로 제한하고 production route·shared storage·다른 tenant와 분리한다. 실행 전에 owner, 승인자, 시작·자동 종료 시각, 영향 범위, abort command, cleanup 확인자를 받는다.

fault 명령은 host·interface·process ID를 명시적으로 검증한 뒤에만 생성한다. 정규식·glob·미해석 환경 변수로 범위를 결정하지 않는다. 사전 건전성 검사에서 모든 node·GPU·NIC·storage가 정상이고 baseline 편차가 budget 안에 있을 때만 주입한다. 이미 장애가 있는 cluster에 두 번째 장애를 주입하지 않는다.

### baseline을 수치로 고정한다

20 step의 warmup과 measurement 창을 구분한다. rank별 data wait, forward, backward, collective, optimizer, checkpoint 시간을 나누고 p50/p95/max와 slowest rank를 남긴다. collective마다 process-group ID, sequence, op, tensor count·dtype, enqueue/completion 시각을 지문화한다. NCCL debug/RAS, GPU XID·ECC, NIC error/drop/retry, NVLink counter, CPU/NUMA, filesystem latency의 시계를 맞춘다.

기준 checkpoint는 model·optimizer·scheduler·scaler·RNG·sampler·packer state, shard manifest, completion marker와 checksum을 갖는다. 독립한 새 process에서 load하여 첫 batch·LR·logits·loss와 한 update를 비교한 뒤 last-good로 승격한다. 파일이 있다는 사실만으로 recovery point라고 표시하지 않는다.

## 장애 주입 매트릭스

### F1 rank kill

경계 A는 forward 전, B는 all-reduce 중, C는 optimizer step 직후다. 각 경계에서 `SIGTERM→grace`, 별도 run에서 `SIGKILL`을 주입한다. launcher 종료 범위, last committed step, incomplete gradient/update 노출 여부를 기록한다.

### F2 network stall

한 rank/NIC 경로에 제한된 지연을 넣고 NCCL timeout·watchdog·RAS·launcher가 보고하는 최초 timestamp를 맞춘다. 복구 불가능한 communicator를 재사용하지 않는지 확인한다.

### F3 partial checkpoint

shard 쓰기 중 writer를 종료한다. completion marker 부재와 latest resolver의 이전 generation 선택을 확인한다. 별도 run에서는 complete marker 뒤 shard bytes를 변조해 load hash 검증을 시험한다.

### F4 straggler와 asymmetric input

한 rank의 data worker만 일정 시간 지연하거나 최장 sequence·특정 expert route를 배정한다. collective sequence는 같은데 도착 시간만 벌어지는지 본다. straggler를 hang으로 오인하지 않도록 heartbeat, kernel timeline, input 시간과 barrier wait를 함께 비교한다. 자동 sample skip이 rank별 제어 흐름을 바꾸면 fail-fast해야 한다.

### F5 GPU·node·storage degradation

장비를 실제로 손상시키지 않는 범위에서 GPU clock 제한, CPU contention, 읽기 지연, 허용된 error simulator를 쓴다. GPU RAS/XID, node health agent, scheduler, trainer watchdog 중 누가 먼저 이상을 포착하는지 본다. 문제 node를 격리한 뒤 spare node로 재배치할 때 topology·rank mapping·checkpoint reshard·data ownership이 새 generation에서 일관되는지 확인한다.

### F6 control-plane split brain

trainer process의 생존 여부뿐 아니라 launcher·rendezvous·checkpoint coordinator의 lease 만료와 중복 승계를 시뮬레이션한다. 두 coordinator가 같은 optimizer epoch에 서로 다른 membership·batch·checkpoint를 commit하지 못해야 한다. generation/lease token이 오래된 writer의 checkpoint·metric·model publication effect를 거부하는지 본다.

## 실행 순서

1. 정상 fingerprint와 last good `CheckpointID`를 확정한다.
2. fault 하나만 활성화한다.
3. alert 전후 60초 metric·trace·log를 보존한다.
4. 자동 재시작 횟수를 1회로 제한한다.
5. recovery 후 첫 32개 batch ID와 LR/RNG를 비교한다.
6. 동일 topology와 변경 topology 복구를 분리한다.
7. final parameter·optimizer state를 허용 tolerance로 비교한다.

## 실패 판정

- 완료 marker가 incomplete shard를 승인하면 즉시 실패다.
- 같은 optimizer epoch에 서로 다른 batch manifest가 있으면 split-brain이다.
- sample repeat/drop을 기록하지 못하면 sample-exact를 주장하지 않는다.
- 한 inference/worker rank만 old artifact를 쓰면 version publication 실패다.

## 분석 절차

### 첫 신호와 마지막 증상을 나눈다

watchdog timeout은 종종 마지막 증상이다. 시간축에서 첫 heartbeat 유실, 첫 collective sequence divergence, 첫 NIC/NVLink error, 첫 XID, 첫 checkpoint commit 불일치를 찾는다. 모든 rank가 all-reduce에서 세월을 보낸다고 NCCL을 즉시 원인으로 지목하지 않는다. rank 하나가 input decode, OOM, kernel에서 collective에 도달하지 못했을 수 있다.

rank 로그에 세 시각을 사용한다. host wall-clock은 node 간 정렬, monotonic clock은 process 내 간격, GPU event와 profiler correlation ID는 device timeline에 쓴다. clock offset·drift 추정을 같이 남겨 서로 다른 host의 로그 한 줄 순서를 과장하지 않는다.

### 통제 run으로 가설을 기각한다

토폴로지·tensor shape·collective 순서를 같게 유지한 작은 재현을 만든다. fault를 하나씩 적용하고 예상한 첫 metric·trace·fail-fast 경계·recovery state를 사전에 적는다. fault 없이도 장애가 나면 baseline이 불안정하므로 해당 비교를 무효로 한다. timeout·retry 횟수를 늘려 통과시킨 결과는 원인 해결이 아니다.

### 복구 등급을 판정한다

sample-exact는 resume 후 ordered batch/span ID, transform RNG, loss 분모가 baseline과 같아야 한다. numerical-tolerance는 배치·state는 같지만 backend 차이를 사전 tolerance로 허용한다. statistical recovery는 순서가 다르지만 mixture·metric 분포와 품질을 통계적으로 검사한다. world-size/topology 변경은 자동으로 sample-exact라고 표시하지 않는다.

## 출력

fault별 `IncidentID`, command, injected boundary, trace, first signal, RCA, recovery checkpoint, batch diff, parameter tolerance, 실행 여부를 한 row로 남긴다.

결과 디렉터리에는 topology/config/environment manifest, baseline fingerprint, fault specification·approval, raw rank logs, metric·trace, checkpoint manifests, recovery comparison, cleanup report, `verdict.md`를 넣는다. top manifest가 각 파일의 checksum과 evidence grade를 참조한다. 실행하지 않은 fault는 결과 칸을 비워 두지 말고 `NotExecuted`와 이유를 명시한다.

통과는 장애가 예상한 첫 신호에서 포착되고, 전 rank가 한 failure generation으로 수렴하며, partial effect 없이 last durable checkpoint로 복구될 때만 가능하다. recovery 후 배치·LR·RNG·parameter·optimizer 비교가 선언한 등급을 만족하고, 이전 communicator·lease·incomplete generation이 재사용되지 않아야 한다. cleanup 후에 network rule·clock limit·killed worker·temporary storage가 남지 않았음을 별도 검증한다.

## 정적 fixture로 복구 oracle을 먼저 증명한다

장애 명령 전에 TorchTitan commit `b482babc…`의 [`test_checkpoint.py:245-272`](https://github.com/pytorch/torchtitan/blob/b482babc5f1d5d718e1719a735f9a2d86d1b9aff/tests/unit_tests/cpu/test_checkpoint.py#L245-L272)를 읽는다. 이 테스트는 model weight·bias와 optimizer의 가짜 parameter를 저장하고 모두 다른 값으로 바꾼 뒤 load하여 세 값이 원본과 같은지 확인한다. 실습의 최소 fixture도 이 구조를 따른다. rank별 model shard, optimizer shard, scheduler step, sampler cursor를 작은 배열과 정수로 적고, 저장 snapshot·의도적 mutation·복구 예상값을 세 열로 만든다.

seed는 data order, model stochasticity와 fault schedule에 각각 별도 값을 두며 topology와 accumulation을 config digest에 넣는다. pass oracle은 load 호출 성공이 아니라 다음 BatchID, LR, model/optimizer tensor checksum과 첫 update가 기준표와 맞는 것이다. optimizer 항목 하나를 manifest에서 빼는 변형에서는 load 직후 optimizer checksum이 최초로 갈라져야 한다. completion marker만 확인하고 이를 놓치면 첫 update의 parameter delta에서 뒤늦게 드러난다.

정적 검토 순서는 `checkpoint inventory → shard owner → completion generation → restored tensor/state → next SampleID → first update`다. inventory부터 다르면 writer/schema, owner에서면 reshard planner, sample에서면 sampler/RNG, update에서만 다르면 optimizer·scheduler restore를 판다. 위 upstream 테스트는 단일 process의 값 복원을 증명할 뿐 torn write, world-size 변경과 NCCL communicator 재생성을 증명하지 않는다. 그 공백은 17장의 DCP 좌표와 이 실습의 F1~F6 별 `NotExecuted` 칸으로 남긴다.
