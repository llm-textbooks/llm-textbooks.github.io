# Playbook 06. rank hang

## 실행 순서

### collective 호출 순서
1. 모든 rank에서 마지막으로 도달한 step, microbatch, collective sequence를 모은다.
2. launcher, NCCL watchdog/RAS, XID, NIC/NVLink metric의 최초 timestamp를 맞춘다.
3. 가장 늦게 로그를 남긴 rank가 아니라 최초로 다른 collective를 호출한 rank를 찾는다.
4. 작은 tensor와 같은 topology에서 collective 순서를 재현한다.

## 분기

### 판정
- 호출 sequence가 다르면 control flow와 data shape를 조사한다. sequence는 같은데 한 rank만 늦으면 straggler, kernel, input을 확인하고, transport error가 있으면 link, NIC, GPU부터 확인한다.
- timeout 증가만으로 해결 판정하지 않는다.

### hang을 “느림”과 분리한다

특정 collective의 지연이 길어진 것과 rank들이 서로 다른 collective를 기다리는 것은 다른 사고다. rank별 progress heartbeat에 UpdateID, microbatch, pipeline stage, collective type, process-group ID, sequence number, tensor shape·dtype·count, enqueue/completion timestamp를 넣는다. watchdog timeout은 증상을 알릴 뿐 최초 원인을 말해 주지 않는다.

현재 시각의 stack trace만으로는 부족하다. 모든 rank가 collective에서 기다리고 있어도 원인은 몇 분 전 data loader, CUDA kernel, Python exception일 수 있다. 따라서 rank·host·GPU 시계 오차를 보정하고 첫 heartbeat 누락, 첫 collective sequence divergence, 첫 transport/RAS/XID 이벤트, 최종 watchdog을 하나의 시간축에 놓는다.

### 최초 대응자는 프로세스를 무작정 죽이지 않는다

장비 손상이 이어지는 XID나 전력·온도 이상이 없다면, 우선 각 rank의 생존 여부와 마지막 진척 지점을 한 번에 채집한다. Python stack, native stack, CUDA stream의 마지막 제출 작업, process group별 sequence counter, GPU utilization·memory, NIC port counter를 같은 IncidentID 아래 모은다. rank를 차례로 재시작하면 collective 대기 상태가 사라져 최초 불일치의 증거도 함께 사라진다.

반대로 GPU reset 반복, uncorrectable memory error, NIC가 지속적으로 link flap을 일으키는 경우에는 증거 채집보다 격리가 우선일 수 있다. 이때도 어느 보호 규칙이 작동해 어떤 rank generation을 종료했는지 기록한다. 안전 조치와 원인 판정은 같은 것이 아니다. 노드를 제외하고 재실행이 성공했다는 사실은 그 노드가 원인이었다는 강한 단서지만, kernel race나 placement-sensitive 순서 결함도 노드 변경으로 잠시 사라질 수 있다.

시간축은 wall clock 하나에 의존하지 않는다. host별 monotonic timestamp와 coordinator가 관측한 heartbeat sequence를 함께 쓴다. NTP 보정 전후에 시간이 역행할 수 있고, GPU event timestamp와 host log 시각의 기준점도 다르다. 정상 구간에서 CPU 제출과 GPU 완료 사이의 대응점을 주기적으로 기록하면 장애 구간의 사건을 같은 축으로 맞추기 쉽다.

### collective 시퀀스를 지문으로 만든다

각 process group에서 collective를 `(group, sequence, op, count, dtype, root/peer)`로 표현한다. 전 rank의 지문을 모아 처음 다른 sequence를 찾는다. 한 rank는 all-reduce, 다른 rank는 all-gather를 호출했다면 topology와 timeout을 바꿔도 해결되지 않는다. op는 같지만 count가 다르면 dynamic shape, empty batch, expert token count, conditional parameter usage를 본다.

시퀀스가 전 rank에서 같은데 하나가 enqueue에 도달하지 못했다면 그 rank의 직전 CPU/GPU work를 본다. data decode, filesystem stall, page fault, kernel infinite loop, GPU reset, OOM 예외, Python GC가 후보다. 모두 enqueue했지만 completion이 없으면 link, NIC, switch, NVLink/NVSwitch, NCCL transport·channel state를 본다.

sequence 번호만 같다고 충분하지 않다. 같은 번호의 collective가 서로 다른 process group을 가리키면 rank 집합이 맞지 않는다. group을 생성한 순서에 따라 로컬 객체 ID가 달라질 수 있으므로, 정렬된 global rank 목록과 group generation의 digest를 함께 비교한다. tensor count는 element 수와 byte 수를 모두 남긴다. FP32와 BF16이 같은 element count를 가져도 전송 byte와 datatype contract는 다르다.

비동기 collective에서는 `enqueue`와 `wait`를 분리한다. 모든 rank가 같은 작업을 제출했지만 한 rank만 다른 stream에서 dependency event를 기다릴 수 있다. 반대로 Python stack이 다음 연산에 있어도 GPU stream은 이전 collective에서 멈췄을 수 있다. work handle 생성 시각, stream ID, 선행·후행 event와 `wait` 호출자를 기록해야 host 진행과 device 진행을 혼동하지 않는다.

collective trace가 너무 많아 상시 보존이 어렵다면 ring buffer를 쓴다. 최근 N개 작업의 metadata만 메모리에 유지하고 watchdog이 경보를 내면 덤프한다. tensor 내용은 기본적으로 저장하지 않고 shape·dtype·count·checksum처럼 민감도를 낮춘 정보를 쓴다. 다만 checksum도 입력 identity와 연결될 수 있으므로 보존 기간과 접근 권한을 정한다.

### 소유권과 위상이 바뀌는 경계를 점검한다

**ProcessGroupNCCL 사건을 detector와 원인으로 분리한다**

고정 PyTorch revision `3691693263d2b66a68867e39b7449876844e06cf`에서 `ProcessGroupNCCL.cpp:975` 부근은 `TORCH_NCCL_ASYNC_ERROR_HANDLING`을 process-group 상태로 읽는다. source에서 환경 변수가 소비된다는 사실은 abort 정책의 입력을 고정하지만, 어느 rank의 어느 연산이 먼저 잘못됐는지는 말하지 않는다. watchdog·monitoring thread·NCCL RAS는 각각 outstanding work, heartbeat와 communicator 상태를 관측한다. 이 셋의 마지막 오류 문자열을 곧 root cause로 쓰지 않는다.

경쟁 가설은 최소 다섯 개를 동시에 연다.

| 가설 | 가장 작은 판별식 | PASS oracle | FAIL 뒤 소유자 |
|---|---|---|---|
| rank별 제어 흐름이 다름 | 같은 group generation에서 `(seq, op, count, dtype)` 비교 | 전 rank 지문 동일 | model/scheduler control flow |
| 한 rank가 enqueue 전 멈춤 | 직전 CPU span과 CUDA 제출 event 비교 | 모든 rank가 같은 work를 enqueue | data loader/kernel/runtime |
| stream dependency deadlock | work event와 선행 stream event를 동기 대조군에서 제거 | 동기 대조군 완료, overlap군에서만 정지 | stream/overlap owner |
| transport·device fault | 최초 NCCL RAS·XID·NIC counter가 sequence divergence보다 앞서는지 확인 | fault 없는 fabric control 완료 | GPU/NIC/fabric |
| 정상 long tail·observer starvation | 동일 phase latency 분포와 watchdog heartbeat를 분리 | work는 완료되고 detector만 늦음 | timeout/observer policy |

PyTorch upstream의 monitored-barrier fixture(`torch/testing/_internal/distributed/distributed_test.py:8829-8993`)는 정상 barrier, rank timeout, all-reduce에 멈춘 rank와 failure order를 나눠 검사한다. 이는 failure category를 구별하는 참고 oracle이지만 CPU/Gloo 조건을 CUDA/NCCL fabric의 보증으로 넓힐 수 없다. 반대로 고정 NCCL source에서 `ncclGroupStart/End`와 communicator lifecycle을 찾았다는 사실도 서로 다른 rank의 collective 순서를 의도적으로 뒤틀어 fail-fast하는 canonical test가 있다는 뜻은 아니다. 이 빈칸은 `TestCoverageUnverified`로 남긴다.

최소 회귀 fixture는 네 rank와 세 collective로 만든다. 정상군은 `all_reduce(A) → all_gather(B) → barrier`를 모두 같은 group에서 호출한다. 변형군은 한 번에 하나만 바꾼다. rank 2의 둘째 op를 `broadcast`로 바꾸고, rank 1의 `B.numel`을 바꾸고, rank 3의 enqueue를 bounded delay하며, 마지막에는 rank 0을 enqueue 직전에 종료한다. 첫 두 변형은 sequence/payload contract에서, delay는 latency discriminator에서, kill은 liveness/failure-generation에서 처음 갈라져야 한다. 네 경우가 모두 같은 “timeout” 한 줄로만 끝나면 진단 fixture는 실패다.

안전한 복구는 timeout 값을 올리는 일이 아니다. 모든 rank가 같은 failure generation을 인식하고 outstanding communicator를 폐기한 뒤 last committed UpdateID·CheckpointID로 함께 돌아가야 한다. old communicator handle을 새 rendezvous에서 재사용하지 않고, 재개 첫 SampleID·loss numerator/count·parameter delta를 대조군과 비교한다. node 격리 뒤 성공했다는 사실은 hardware 가설을 강화하지만 placement-sensitive race를 반증하지 못하므로 수정된 fixture를 같은 topology와 교체 topology에서 모두 반복한다.

gradient accumulation의 마지막 microbatch에서만 reduce하는 설계는 모든 rank가 같은 accumulation counter를 가져야 한다. pipeline은 stage별 send/recv schedule, expert parallel은 dispatch/combine, context parallel은 ring step, FSDP는 parameter all-gather/reduce-scatter 순서가 맞아야 한다. rank-local sample skip, conditional loss, unused parameter, MoE zero-token expert가 제어 흐름을 바꾸면 shape가 정상인데도 hang이 생긴다.

elastic membership 변경 시에는 old process group과 new process group의 generation을 구분한다. stale rank가 이전 group에 collective를 보내거나 두 coordinator가 다른 membership을 commit하면 재시작으로 일시 해결되어도 다시 발생한다. rendezvous generation, rank mapping, topology manifest와 CheckpointID를 함께 고정한다.

### MoE와 가변 길이에서 빈 작업을 합법 상태로 다룬다

MoE에서는 어떤 expert rank가 이번 microbatch에 token을 하나도 받지 않을 수 있다. 이 rank가 dispatch나 expert gradient collective를 건너뛰고 다른 rank는 zero-count collective를 호출하면 순서가 갈린다. 반대로 backend가 zero-count tensor를 허용하지 않는데 형식만 맞추려 호출하면 다른 오류가 난다. dispatcher 계약에 빈 assignment의 shape, collective 참여 여부와 dummy buffer 정책을 명시하고, all-zero routing fixture를 둔다.

sequence packing과 dynamic batching에서도 rank별 유효 token 수는 다를 수 있다. loss가 없는 rank가 backward 전체를 생략하면 DDP·FSDP의 parameter collective가 어긋난다. 유효 count가 0이어도 graph와 reduction 순서를 유지할지, batch를 전 rank에서 다시 구성할지 결정한다. `find_unused_parameters` 같은 옵션은 단순한 hang 해제 스위치가 아니라 graph discovery와 collective 순서를 바꾸므로, 어떤 parameter가 왜 unused인지 먼저 증명한다.

activation checkpoint와 conditional layer drop은 recompute에서도 같은 branch를 선택해야 한다. forward와 backward 재계산의 RNG나 data-dependent 조건이 달라지면 어떤 rank만 특정 parameter gradient를 만들 수 있다. branch decision digest를 microbatch와 layer별로 샘플링하고, checkpoint 복원 뒤 최초 divergence를 확인한다.

## 재현·격리·복구

### 가장 작은 재현으로 축소한다

원래 tensor 내용은 필요하지 않을 수 있다. 같은 process-group 생성 순서, collective sequence, tensor count·dtype, stream 관계만 남긴 작은 program으로 줄인다. 제어 흐름 문제라면 rank 하나에 빈 batch·conditional branch·exception을 주입해 지문 불일치가 같은 sequence에서 나오는지 본다. transport 문제라면 동일 topology에서 작은 반복 collective로 link·channel별 지연과 오류를 본다.

fault injection은 rank kill, worker stall, GPU XID, NIC link down, packet loss, delayed kernel, 한 rank의 shape 변경을 각각 분리한다. 안전한 스테이징 환경에서만 수행하고, 주입 전에 기대한 첫 metric·trace, fail-fast 한계, recovery 조건, cleanup을 적는다. 어떤 주입이든 시간만 지난 뒤 watchdog이 죽이는 것보다 최초 원인을 식별하는 것이 목표다.

재현 프로그램은 네 단계로 줄인다. 먼저 model과 data를 제거하고 기록된 collective 지문만 재생한다. 다음으로 process group과 topology는 유지하되 모든 payload를 동일 크기의 deterministic tensor로 바꾼다. 셋째, stream overlap을 끄고 동기 실행해 순서 결함과 race를 분리한다. 마지막으로 원래 overlap을 한 경계씩 되살린다. 처음부터 모든 환경 변수를 바꾸면 어느 변경이 원인을 제거했는지 알 수 없다.

NCCL debug·RAS 정보는 incident 기간에 필요한 범위로 활성화하고 로그 용량과 민감 정보를 관리한다. algorithm이나 protocol을 강제하는 실험은 원인 격리를 위한 대조군이다. 특정 강제값에서 hang이 사라져도 기본 cost model의 결함, topology discovery 문제, timing-sensitive race 가운데 무엇인지 추가로 나눠야 한다. 영구 설정으로 승격하려면 실제 message-size 분포와 장시간 회귀 시험이 필요하다.

네트워크 대조군은 같은 GPU 집합에서 intra-node만, 같은 노드 사이에서 단순 point-to-point, 실제 process group의 collective 순으로 넓힌다. port별 symbol·replay·discard counter와 collective tail을 시간축에서 맞춘다. 평균 대역폭이 정상이어도 한 rail이나 한 방향에서만 오류가 날 수 있으므로 rank pair matrix의 최대값을 본다.

### 부분 살아남을 막고 일관되게 rollback한다

rank 하나만 종료하고 나머지가 계속 checkpoint·optimizer effect를 남기지 않게 한다. coordinator가 failure generation을 commit하고 전 rank의 work를 cancel/격리한 뒤, 마지막 완전한 CheckpointID와 data cursor로 rollback한다. partial optimizer step이 가능한 구조라면 해당 generation을 재사용하지 않는다.

임시 운영 조치로 문제 link·GPU·node를 격리할 수 있지만 topology와 rank mapping이 바뀌면 performance·collective algorithm·checkpoint reshard 계약도 바뀐다. timeout을 늘리는 조치는 정상적인 최악 지연이 한계보다 길었음을 분포로 증명했을 때만 쓴다. sequence divergence와 장비 오류를 긴 timeout으로 숨기지 않는다.

### 재개 성공은 첫 optimizer effect로 판정한다

마지막 checkpoint가 읽힌다는 사실만으로 복구를 승인하지 않는다. 저장된 global sample cursor, accumulation phase, optimizer·scheduler·scaler, RNG와 process-group generation을 확인한다. 장애 직전 microbatch가 일부 rank에서만 backward를 끝냈더라도 optimizer effect가 commit되지 않았다면 전 rank가 같은 경계에서 다시 처리해야 한다. 부분 update 가능성이 있으면 해당 checkpoint generation을 폐기한다.

재개 후 첫 batch의 SampleID 집합과 순서, loss numerator/count, gradient norm, clipping coefficient와 parameter delta를 정상 대조 실행과 비교한다. topology나 world size가 바뀌었다면 exact trajectory가 가능한지 먼저 계약을 정한다. shard reshard가 성공했어도 optimizer moment가 다른 parameter에 붙으면 첫 delta에서 갈린다. 이 검증이 끝나기 전에는 장시간 실행의 처리량 회복을 성공 증거로 쓰지 않는다.

회귀 시험은 hang이 났던 조건만 반복하지 않는다. empty batch, zero-token expert, 마지막 partial accumulation, evaluation 전환, checkpoint 저장, membership 재구성처럼 collective sequence가 달라지는 경계를 조합한다. 각 경우에 정상 완료하거나 정해진 시간 안에 모든 rank가 같은 failure generation으로 종료해야 한다.

## 종료 조건

### 통과
동일 fault 조건에서 모든 rank가 같은 collective 순서를 유지하거나 명확히 fail-fast하고 recovery checkpoint가 유효해야 한다.

정상·오류 주입 run의 rank별 collective 지문, 시간축, NCCL/RAS·XID·NIC·NVLink 증거, 첫 불일치 rank·sequence, 수정 commit, recovery CheckpointID를 IncidentID에 묶는다. fail-fast는 전 rank가 한정된 시간 안에 같은 failure generation을 인식하고 부분 effect를 남기지 않았음을 의미한다. 재개 후 첫 batch·optimizer·scheduler state가 약속한 복구 등급에 맞고, 장시간 반복에서 지연·sequence 불일치·transport error가 재발하지 않아야 종료한다.
