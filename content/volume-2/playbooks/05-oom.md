# Playbook 05. OOM과 fragmentation

## 실행 순서

### 메모리 시간축
1. OOM 직전의 allocated/reserved/active/inactive split, peak와 요청 byte를 저장한다.
2. 실패 batch의 실제 text/media token 수, activation checkpoint 설정, sequence outlier를 확인한다.
3. microbatch, accumulation, sequence, checkpointing 가운데 한 옵션만 바꾸어 실험한다.
4. snapshot에서 수명이 비정상적으로 긴 tensor와 그 소유 경로를 찾는다.

## 분기

### 판정
- allocated가 capacity에 가까우면 진짜 용량, reserved≫allocated면 fragmentation/cache, step마다 증가하면 leak/retained graph다.
- 평균 sequence가 아니라 실패 batch의 최대 shape를 본다.

### OOM 직전의 상태를 잃지 않는다

OOM handler가 allocator cache부터 비우면 가장 유용한 증거가 사라진다. 예외 직전의 device, rank, UpdateID, microbatch index, accumulation index, 요청 allocation byte, free byte, allocated·reserved·active·inactive split, peak, allocator snapshot을 먼저 보존한다. 증거 수집 자체가 추가 OOM을 일으키지 않도록 snapshot 용량과 실패 경로를 사전에 시험한다.

배치 원장에는 sample ID, 샘플별 text token, image patch, audio frame, video token, padding token, valid-label 수, packing 구성, expert route 집계를 남긴다. 설정의 `max_length`나 평균 token은 실제 메모리를 대표하지 못한다. variable-resolution image, tool schema, chat template, MoE route skew가 실현 shape를 바꿘다.

### 첫 10분의 질문을 고정한다

먼저 어느 rank의 어느 phase에서 처음 실패했는지 확인한다. 다른 rank에서 발생한 NCCL timeout은 최초 OOM에서 파생된 증상일 수 있다. 이어 allocator가 요구한 단일 allocation 크기를 확인한다. free memory가 그보다 작으면 용량 문제지만, 총 free가 충분해도 요청을 담을 알맞은 block이 없을 수 있다. 실패 batch의 실현 shape가 정상 분포 안에 있는지, 이전 step 뒤에 메모리 바닥값이 회복됐는지도 확인한다. 이 네 가지에 답하기 전에는 batch size와 환경 변수를 동시에 바꾸지 않는다.

관측 handler는 OOM rank만 기록하고 끝나지 않는다. 모든 rank가 같은 UpdateID의 allocated·reserved·peak, batch shape, expert count와 phase를 내보내야 비대칭을 볼 수 있다. 중앙 수집을 기다리다가 collective hang이 길어지지 않도록 각 rank가 로컬 artifact를 먼저 내구적으로 저장하고 coordinator가 사후 결합한다.

같은 batch를 같은 상태로 즉시 반복하면 같은 OOM이 날 가능성이 높다. 자동 microbatch 축소가 있다면 새 BatchRevision과 accumulation 계획, loss denominator가 만들어진다. 실패 batch를 조용히 건너뛰는 것은 가장 긴·복잡한 샘플을 체계적으로 제거하므로 허용하지 않는다.

### 정상적으로 필요한 byte와 누수를 나눈다

정적 항목으로 parameter, gradient, optimizer master weight·moment, persistent buffer, communication bucket을 세고, 동적 항목으로 activation, temporary workspace, attention·loss intermediate, collective staging buffer를 센다. dtype과 sharding 소유권을 반영해 rank별 expected byte를 손으로 계산한다. 관측값이 이 상한을 크게 넘으면 graph retention, duplicate optimizer state, 의도하지 않은 full-parameter materialization을 의심한다.

step가 반복될수록 allocated floor가 오르면 Python list에 loss/tensor를 graph와 함께 보관했는지, hook이 output을 detach 없이 쌓는지, `retain_graph=True`가 남았는지, evaluation output과 generation cache가 삭제되는지 본다. allocated floor는 안 오르지만 reserved block이 파편화되면 shape 변동, stream 간 lifetime, split size·expandable segment 정책을 본다.

### byte 원장을 live interval로 바꾼다

parameter 수를 `P`, 저장 dtype byte를 `b_w`, gradient byte를 `b_g`, master weight와 두 moment를 `b_m`, `b_1`, `b_2`라 하면 복제된 Adam 계열의 persistent 근사치는 `P(b_w+b_g+b_m+b_1+b_2)`다. sharding stage에 따라 각 항의 분모가 다르므로 전체를 world size로 한 번에 나누지 않는다. frozen parameter에는 gradient·moment가 없을 수 있고 일부 group만 FP32 master를 가질 수 있다. optimizer 생성 뒤 group별 tensor inventory로 식을 수정한다.

activation은 단순히 `B·S·H` 하나가 아니다. layer별 norm 입력, attention Q·K·V와 score 또는 flash-attention saved state, MLP gate/up 중간값, dropout mask, residual과 checkpoint boundary를 센다. sequence가 두 배일 때 어떤 항은 선형이고, 명시적 attention score는 제곱으로 늘 수 있다. 사용하는 kernel이 실제로 무엇을 저장하는지 source와 profiler로 확인한다.

각 tensor에 생성 시점과 마지막 소비 시점을 붙이면 합계가 아니라 동시 peak를 계산할 수 있다. parameter all-gather와 activation, optimizer temporary가 겹치는 순간이 peak다. 개별 phase의 최대를 더하거나 모든 tensor 크기를 한꺼번에 합치는 방식은 모두 실제 peak를 잘못 예측할 수 있다.

### 어느 구간에서 peak가 생겼는지 찾는다

forward에서만 터지면 sequence length, attention 구현, multimodal feature, MoE dispatch, activation checkpointing을 본다. backward 시작에서 터지면 저장 activation과 gradient allocation이 겹치는 지점, fused backward workspace, checkpoint recomputation을 본다. optimizer step에서만 터지면 master weight·moment lazy initialization, foreach/fused temporary tensor, full parameter gathering을 본다. evaluation/generation에서 터지면 KV cache, 전체 logits 보관, metric 수집, train graph이 아직 살아 있는지 본다.

rank 하나만 OOM이면 해당 rank의 sequence/expert workload, shard ownership, topology-sensitive workspace를 비교한다. 모든 rank가 같은 순간 터지면 전역 shape·algorithm change를 본다. 한 rank의 OOM 후 다른 rank가 collective에 hang되면 두 incident로 나누지 말고 최초 OOM을 원인, hang을 파생 효과로 묶는다.

### 분산 wrapper가 만드는 순간적 복제를 찾는다

FSDP나 ZeRO에서는 steady-state shard가 작아도 module forward 전에 full parameter all-gather가 생길 수 있다. prefetch 거리, wrapping granularity와 reshard-after-forward 정책이 두 module의 full parameter를 동시에 살게 할 수 있다. profiler에서 all-gather buffer의 생성·해제와 activation peak를 겹쳐 본다. shard 크기만 보고 충분하다고 판단하면 이 순간적 복제를 놓친다.

checkpoint 저장과 evaluation 전환도 full state materialization을 만들 수 있다. rank 0에 full model이나 optimizer를 모으는 옵션, state dict 변환, CPU staging 전 device copy를 확인한다. 저장 중 OOM은 학습 microbatch를 줄여도 해결되지 않을 수 있다. 저장 경로 자체를 sharded·streaming 방식으로 바꾸고 manifest 검증을 추가한다.

MoE에서는 expert parameter 소유량과 routed activation 부하가 다른 축이다. expert 수가 고르게 배치돼도 token이 한 rank에 몰리면 dispatch receive buffer와 grouped GEMM workspace가 커진다. capacity·dropless 정책, send/receive count의 rank max, workspace 상한을 함께 본다. load balance 계수를 높이는 것은 메모리 응급 조치가 아니라 objective 변경이다.

pipeline parallel은 stage마다 layer 수와 activation lifetime이 달라 peak가 다르다. 1F1B schedule의 warmup microbatch, interleaving과 activation checkpoint 경계를 반영한다. 가장 많은 parameter를 가진 stage가 항상 가장 큰 peak를 갖는 것은 아니다.

## 통제 실험과 복구

### 메모리–계산–통신 교환을 수치로 남긴다

microbatch를 줄이면 activation peak는 줄지만 accumulation 횟수와 all-reduce 시점, valid-token 분모가 바뀌 수 있다. sequence를 줄이면 attention byte는 크게 줄지만 긴 문맥 분포와 truncation 정책이 바뀐다. activation checkpointing은 저장 byte를 재계산으로 바꾸지만 RNG·autocast parity와 throughput에 비용을 준다. optimizer sharding·CPU offload는 device 용량을 통신·bus·host memory로 옮긴다.

통제 조건은 한 번에 하나만 바꾸고 peak allocated/reserved, step time, recompute time, collective time, valid token/s, loss parity를 함께 보고한다. OOM이 사라졌다는 사실만으로 성공을 선언하지 않는다. 분모, sample mixture, optimizer step semantics와 최종 tensor가 같은지, 새 병목이 허용 범위 안인지 확인한다.

### 옵션을 상태 변화와 연결한다

microbatch를 `m`에서 `m/2`로 줄이고 accumulation을 두 배로 늘리면 nominal global batch는 같을 수 있다. 그러나 dropout RNG, token count weighting, gradient clipping 시점, scheduler의 step 정의와 communication overlap이 달라질 수 있다. 각 microbatch loss가 단순 평균인지 valid-token 합계인지 확인하고 one-step parameter delta를 대조한다.

activation checkpointing 범위를 넓히면 저장 activation은 줄지만 backward 중 forward kernel과 workspace가 다시 열린다. recompute 구간의 autocast, RNG와 conditional routing이 원 forward와 같아야 한다. peak가 backward의 다른 지점으로 이동할 수 있으므로 총 최대값뿐 아니라 시간축을 비교한다.

attention kernel 변경은 saved tensor와 workspace를 바꾼다. 선택한 dtype, mask, head dimension과 sequence shape에서 실제 어느 backend가 dispatch됐는지 기록한다. fallback이 특정 outlier batch에서만 일어나 OOM을 만들 수 있다. 설정 이름이 아니라 runtime branch를 본다.

optimizer의 fused·foreach 선택은 temporary tensor list와 launch 수를 바꾼다. 첫 optimizer step에서 moment가 lazy allocation되면 warmup 몇 step 동안만 peak가 높을 수 있다. 빈 model 직후가 아니라 moment가 모두 만들어진 안정 상태와 checkpoint resume 직후를 각각 측정한다.

CPU offload는 GPU byte를 pinned host memory, PCIe traffic과 synchronization으로 옮긴다. host OOM이나 page fault, transfer tail로 장애 형태가 바뀔 수 있다. GPU 통과만 보고 승인하지 않고 host RSS·pinned bytes·bus throughput과 step tail을 제한한다.

### fragmentation과 cache를 오해하지 않는다

reserved가 allocated보다 크다는 사실만으로 fragmentation을 증명할 수는 없다. allocator가 재사용하려고 빈 block을 cache한 정상 상태일 수 있다. 실패 allocation을 만족할 연속 block이 없는지, inactive split의 크기 분포, segment 재사용, shape 변경과 시간 상관을 snapshot으로 본다. cache 전체를 반복해 비우는 대신 실현 shape bucket, lifetime과 stream 계약, allocator 설정을 최소 변경한다.

임시 복구 과정에서 문제 batch를 삭제하지 않는다. token-budget batcher로 격리해 원래 sample ID·loss 분모를 유지하거나, 명시적 초과 정책에 따라 split하고 lineage를 남긴다. 구성 변경 기록에는 적용 checkpoint, 기대 peak, throughput 하한, rollback threshold와 담당자를 명시한다.

### graph capture와 stream lifetime을 확인한다

CUDA graph capture는 고정 address와 private memory pool을 요구할 수 있다. capture 전 warmup shape와 실제 batch shape가 다르거나 여러 graph pool이 누적되면 reserved가 커진다. graph instance와 shape bucket, pool lifetime을 원장에 넣는다. graph를 끄면 OOM이 사라지는 것은 대조군일 뿐이며 eager 경로의 함수 parity와 처리량을 함께 본다.

stream이 여러 개인 경우 allocator가 block을 안전하게 재사용하려면 tensor가 어느 stream에서 쓰이는지 알아야 한다. lifetime annotation이나 event가 빠지면 corruption이 생길 수 있고, 지나치게 보수적이면 block이 오래 묶인다. synchronization으로 OOM을 숨기면 overlap도 사라진다. stream·event trace를 따라 수명 관리 책임이 있는 경로를 고친다.

shape bucketing은 fragmentation을 줄일 수 있지만 padding과 데이터 mixture를 바꾼다. bucket 경계별 sample 수, padding token, valid token/s와 peak를 측정한다. 긴 샘플을 강제로 절단하면 allocator 최적화가 아니라 curriculum 변경이다.

### 종료 전 실패 주입과 회귀 행렬

최대 길이 text, 최대 해상도 image, 최대 frame video, empty-label sample, MoE hot expert와 checkpoint-save 경계를 각각 만든다. 두 극단을 한 batch에 넣는 조합은 단일 축 시험 뒤에 수행한다. train forward, backward, optimizer, evaluation, generation과 save 경로를 따로 통과시킨다.

의도적인 작은 memory limit이나 synthetic workspace로 failure handler가 증거를 남기는지 시험한다. handler 자체가 큰 snapshot을 만들다 다시 OOM에 빠져 원본 예외를 잃지 않아야 한다. 한 rank의 OOM이 다른 rank에서 무한 collective 대기로 바뀌지 않고 정해진 시간 안에 같은 failure generation으로 종료되는지도 본다.

회귀 표에는 shape bucket별 peak allocated·reserved, inactive split 분포, step p50·p99, valid token/s, host memory, collective tail과 loss·delta parity를 넣는다. 첫 optimizer state 생성, evaluation 전환, checkpoint 저장, resume와 curriculum phase 전환을 포함한 window를 고른다.

## allocator 통계와 tensor 소유권을 같은 시간축에 놓는다

### `reserved - allocated`를 곧바로 fragmentation이라 부르지 않는다

PyTorch `3691693263d2b66a68867e39b7449876844e06cf`의 `torch/cuda/memory.py:242-317`은 `allocated_bytes`를 allocator가 할당한 byte, `reserved_bytes`를 `cudaMalloc()` segment에서 예약한 byte, `inactive_split_bytes`를 inactive이면서 반환할 수 없는 split block으로 따로 정의한다. `requested_bytes`와 `allocated_bytes`의 차이는 rounding overhead를 보는 좌표다.

```text
allocated_bytes      allocator가 할당한 메모리
reserved_bytes       CUDA segment로 예약한 메모리
inactive_split_bytes 비활성이지만 반환할 수 없는 split block
requested_bytes      client가 실제 요청한 메모리
```

따라서 `reserved_bytes - allocated_bytes`에는 재사용 가능한 cache도 들어간다. 이 차이가 크다는 이유만으로 fragmentation을 확정하지 않는다. 같은 phase에서 `inactive_split_bytes`, OOM의 단일 요청 byte, 가장 큰 free block, `num_alloc_retries`, `num_ooms`, memory snapshot의 block 상태를 함께 봐야 한다. 반대로 allocated가 device capacity에 가까우면서 inactive split이 작으면 allocator knob보다 live tensor 소유권을 줄이는 편이 맞다.

같은 커밋의 `test/test_cuda.py:549-583`은 1 KiB tensor 생성 뒤 `active_bytes.all.current`가 정확히 1,024 늘고, 삭제·GC·cache 비우기 뒤 기준값으로 돌아오며 freed counter가 1,024인지 단언한다. 이 oracle은 통계 counter의 기본 증감을 닫을 뿐 실제 training graph의 소유자를 알려 주지 않는다. `empty_cache()` 뒤 reserved가 줄었다는 결과도 live tensor가 사라졌다는 증거가 아니다.

### activation·optimizer·collective 소유권을 분리한다

고정 batch를 `forward 직전 → forward 직후 → backward 직후 → optimizer step 직후 → collective 종료 후` 다섯 fence에서 측정한다. phase별 current와 peak를 같은 window에 섞지 말고 peak counter를 fence마다 reset한다. forward에서만 증가했다가 backward 뒤 회수되면 activation, optimizer 첫 step에서 새 바닥값이 생기면 moment·master state의 lazy materialization, collective 전후에만 rank별 peak가 벌어지면 bucket·all-gather·MoE dispatch buffer를 우선 조사한다.

PyTorch의 `test/distributed/_composable/fsdp/test_fully_shard_memory.py:300-344`는 unsharded parameter 수를 world size로 나눈 shard byte에 작은 buffer를 더한 상한으로 FSDP의 current active memory를 검사하고, model 삭제와 GC 뒤 기준값으로 돌아오는지도 단언한다. 이 좌표는 parameter shard 소유권과 회수의 oracle이다. activation peak, optimizer moment, communication workspace의 총 상한까지 보장하지 않으므로 각 owner를 tensor inventory와 profiler range로 따로 계상한다.

### 최소 분리, 판정, 복구의 폐루프

경쟁 원인은 실제 live-set 용량, allocator fragmentation, retained graph, activation shape outlier, optimizer 초기화, collective 순간 복제로 고정한다. 같은 checkpoint와 batch에서 한 번에 하나만 바꾼다. no-step forward로도 터지면 optimizer를 기각하고, reference attention에서만 사라지면 saved activation/workspace를 좁히며, collective를 생략한 단일 rank에서 사라지면 통신 ownership 가설을 지지한다. 매 실험은 peak 감소뿐 아니라 어느 phase·owner의 byte가 줄 것인지 먼저 예측한다.

판정선은 `reserved/allocated` 비율 하나가 아니다. 요청 byte가 live capacity를 넘는지, inactive split과 largest-free-block이 그 요청을 설명하는지, step 종료 current active가 기준선으로 복귀하는지, rank별 owner inventory 합이 allocator active와 허용 오차 안에서 맞는지를 함께 본다. 허용 오차는 CUDA·PyTorch·allocator backend·shape bucket별 golden run에서 미리 등록한다.

복구는 원인 owner에만 적용한다. activation이면 checkpoint 경계나 shape bucket, optimizer면 sharding·offload, collective면 prefetch와 bucket lifetime을 바꾼다. allocator 설정은 fragmentation 증거가 있을 때만 조정한다. 수정 뒤 문제 batch와 정상 batch를 모두 재생해 OOM 부재, phase별 current의 기준선 복귀, inactive split·retry의 예산 준수, loss denominator·parameter delta·throughput의 비회귀를 확인한다. 가장 긴 shape를 제외하거나 실패 rank만 batch를 줄인 run은 통과가 아니다.

## 종료 조건

### 통과
같은 실패 batch가 통과하고 100 step의 peak가 안정돼야 한다. allocator cache를 매 step 비워 숨기지 않는다.

동일 checkpoint와 batch에서 수정 전 OOM을 재현하고, 수정 후에는 메모리 시간축이 예측한 지점에서 낮아졌음을 증명한다. 100 step의 allocated floor와 peak에 증가 추세가 없고, token/s·loss·realized mixture가 허용 범위에 있어야 한다. 다른 shape bucket·rank·eval 경로와 checkpoint resume에서도 재발하지 않아야 한다. IncidentID에 allocator snapshot, batch ledger, byte 손계산, 가설별 통제 실험, 수정 commit과 장기 monitor threshold를 묶는다.
