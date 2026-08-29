# 2장. TTFT와 ITL은 왜 서로 싸우는가

두 사용자가 같은 서버에 접속했다고 생각해 보자. 민지는 긴 문서를 붙여 넣고 요약을
요청했다. 준호는 이미 답변을 받고 있으며 화면에는 token이 한두 개씩 나타나고 있다. GPU는
한 번에 큰 일을 받으면 효율적으로 계산할 수 있으므로 민지의 긴 prompt를 가능한 크게 묶어
처리하고 싶다. 하지만 그 계산이 끝날 때까지 준호의 다음 token을 미루면 화면이 멎은 것처럼
보인다.

이 장의 질문은 바로 이것이다. GPU를 바쁘게 만드는 선택이 왜 사용자의 응답을 더 매끄럽게
만들지는 못할까? 답을 찾으려면 “응답 시간”이라는 한 단어를 여러 개의 시계로 나누고,
prefill과 decode가 같은 모델을 통과하면서도 서로 다른 비용 구조를 가진다는 사실을 이해해야
한다.

## 2.1 사용자는 하나의 지연을 느끼지만 서버에는 여러 시계가 있다

사용자가 전송 버튼을 누른 시각을 `t_0`, 첫 번째 text delta를 받은 시각을 `t_1`, 그 뒤
token `i`와 `i+1`을 받은 시각을 각각 `t_i`, `t_{i+1}`라고 하자. 가장 널리 쓰는 두 지표는
다음처럼 정의할 수 있다.

\[
\mathrm{TTFT}=t_1-t_0
\]

\[
\mathrm{ITL}_i=t_{i+1}-t_i
\]

TTFT(time to first token)는 질문을 보내 첫 token을 보기까지 기다린 시간이다. ITL(inter-token
latency)은 답변이 시작된 뒤 인접한 token 사이의 시간이다. 여러 ITL을 평균낸 값은 TPOT라고
부르기도 하지만, 평균 하나만 보면 가끔 발생하는 긴 멈춤이 사라지므로 분포와 tail을 함께
봐야 한다.

여기서 조심할 점이 있다. client가 잰 `t_0`과 `t_1`에는 network, gateway, server queue,
tokenization과 socket buffering이 모두 들어간다. engine이 기록한 첫 sampled-token 시각은
client가 첫 byte를 받은 시각과 다르다. “TTFT가 500ms”라는 숫자는 어느 clock의 어느 사건을
뺐는지 밝히기 전에는 원인을 가리키지 못한다.

독자가 처음 만들어야 할 것은 거대한 dashboard가 아니라 다음처럼 단순한 timeline이다.

```text
client send
  → API receive
  → render/tokenize complete
  → engine admission
  → first schedule
  → prefill GPU start/end
  → first token commit
  → first byte send/receive
```

이 사건들을 같은 request identity와 연결하면 TTFT가 길다는 막연한 현상을 작은 구간으로
나눌 수 있다. API receive 전이 길면 network와 gateway를 보고, schedule 전이 길면 queue와
admission을 보며, GPU 구간이 길면 batch shape와 kernel을 본다. 첫 token commit 뒤가 길면
detokenization, output queue와 socket을 본다.

## 2.2 prefill과 decode는 같은 forward를 서로 다른 모양으로 실행한다

모델은 새 요청의 prompt를 먼저 읽어 각 layer의 KV를 만든다. 이 단계를 prefill이라고 한다.
prompt 길이를 `L`, hidden size를 `d`라고 할 때 projection과 MLP에는 대략 `L`에 비례하는 큰
행렬곱이 생기고, 일반적인 full attention의 score 계산에는 `L²` 관계가 나타난다. 실제 kernel은
큰 score matrix를 HBM에 그대로 쓰지 않는 등 IO를 줄이지만, 긴 prompt가 짧은 prompt보다
많은 계산과 memory traffic을 요구한다는 사실은 남는다.

prefill의 장점은 token 축에 병렬로 처리할 일이 많다는 것이다. GPU의 많은 core에 큰 tile을
나누어 줄 수 있고 tensor core가 좋아하는 행렬곱 shape가 나온다. 그래서 작은 조각 여러 개보다
큰 prefill batch 하나가 device 처리량만 보면 효율적일 수 있다.

이를 이삿짐 트럭에 비유할 수 있다. 상자를 하나씩 승용차로 나르는 것보다 트럭을 채워 한 번에
옮기는 편이 연료와 운전 시간을 아낀다. 그러나 이 비유는 두 가지를 감춘다. 첫째 token별
attention 비용은 동일한 크기의 상자가 아니다. 둘째 같은 도로를 decode 요청도 사용한다.
트럭 한 대가 하역장을 오래 점유하면 뒤에서 작은 택배를 기다리는 사람이 생긴다.

prefill이 TTFT에 직접 들어가는 이유는 첫 token을 고르기 전에 prompt 전체의 조건부 상태를
만들어야 하기 때문이다. prefix cache가 prompt 일부의 KV를 재사용하거나 prompt를 여러
chunk로 나누지 않는 한, 이 단계가 끝나기 전에 정상적인 첫 decode 결과를 낼 수 없다.

### decode는 token 하나를 만들지만 과거 전체를 읽는다

첫 token을 만든 뒤에는 방금 선택한 token 하나가 다음 query가 된다. cache를 사용하면 과거
token의 K와 V를 다시 projection하지 않아도 된다. query 길이는 보통 1이지만 attention은
지금까지 쌓인 KV를 읽어야 한다. context가 길어질수록 decode 한 step의 KV traffic도 커진다.

decode는 prefill과 반대의 어려움을 가진다. 한 요청만 보면 작은 matrix-vector 성격의 일이
많아 GPU 전체를 채우기 어렵다. 여러 request의 decode token을 같은 step에 모아야 병렬성이
늘어난다. 그래서 continuous batching은 이미 실행 중인 request들을 매 step 다시 묶는다.

사용자가 decode에서 민감하게 느끼는 것은 평균 처리량보다 규칙적인 진행이다. token이 평균
50ms마다 나오더라도 대부분 20ms이고 가끔 500ms가 걸리면 화면은 버벅인다. tail ITL을
숨긴 평균은 좋은 대화 경험을 설명하지 못한다.

또한 model이 token 하나를 계산한 시간과 ITL은 같지 않다. scheduler가 다음 step에 넣기를
기다린 시간, 다른 rank와의 collective, sampling, output routing, detokenization과 client
backpressure가 모두 두 visible token 사이에 들어올 수 있다. GPU kernel duration만 최적화했는데
ITL이 그대로인 상황은 충분히 가능하다.

## 2.3 큰 batch가 처리량을 높이고 지연을 망치는 과정

서버가 단위 시간에 처리할 수 있는 token 수를 높이려면 GPU가 놀지 않도록 충분한 일을 모아야
한다. 하지만 일을 모으는 동안 request는 queue에서 기다린다. batch가 커지면 device 효율은
좋아질 수 있지만 한 step의 실행 시간도 길어질 수 있다.

이를 아주 단순한 식으로 생각해 보자. scheduler가 일을 모으는 시간을 `W(B)`, batch 크기
`B`의 GPU 실행 시간을 `G(B)`, 해당 batch에서 유효하게 처리한 token을 `T(B)`라고 하자.

\[
\text{token throughput}(B)=\frac{T(B)}{W(B)+G(B)}
\]

`B`가 커질 때 `T(B)`가 실행 시간보다 빠르게 늘어나는 구간에서는 처리량이 좋아진다. 그러나
특정 request가 느끼는 지연은 분자에 관심이 없다. 그 request는 admission을 기다린 시간과
자기보다 앞선 step의 실행 시간을 모두 경험한다. 처리량의 최적점과 tail latency의 최적점이
다른 이유다.

실제 serving에서 batch size라는 말도 충분히 정확하지 않다. decode request 32개가 token
하나씩 내는 batch와 prompt token 4천 개를 처리하는 batch는 request 수가 같지 않거나 같아도
비용이 전혀 다르다. 그래서 scheduler는 흔히 sequence 수와 함께 step token budget을 둔다.
관측할 때도 request count만 보지 말고 scheduled query token, KV length 분포와 prefill/decode
구성을 기록해야 한다.

## 2.4 chunked prefill은 두 시계 사이의 타협이다

민지의 긴 prompt를 한 step에 전부 처리하면 GPU 효율은 좋을 수 있지만 준호의 decode가 오래
기다린다. 반대로 prompt를 지나치게 작은 조각으로 자르면 매 조각마다 scheduler와 kernel
launch 비용을 내고, 민지의 첫 token은 더 늦어진다. chunked prefill은 이 두 극단 사이에서
한 step의 prefill token 수를 제한한다.

예를 들어 prompt 8,192 token을 2,048 token씩 네 chunk로 나눈다고 하자. scheduler는 chunk
사이에 decode request를 넣을 기회를 얻는다. 준호의 최악 ITL은 한 번의 8,192-token prefill
step보다 줄어들 가능성이 있다. 하지만 민지의 prefill은 네 번의 admission과 launch를 거치며,
각 chunk가 작은 shape가 되어 kernel 효율이 낮아질 수도 있다.

“chunked prefill을 켜면 latency가 좋아진다”는 문장은 너무 짧다. 어느 latency가
좋아지는지, prompt 길이와 concurrency가 어떤지, chunk가 backend tile과 graph shape에 어떻게
맞는지를 함께 말해야 한다.

좋은 실험은 한 값만 바꾸고 다음을 함께 측정한다.

- 긴 prompt request의 TTFT 분포
- 동시에 decode 중인 request의 ITL과 tail ITL
- step별 prefill·decode token 구성
- scheduler waiting과 실행 시간
- kernel 수, GPU duration과 launch gap
- activation peak와 KV pressure

이 목록은 매 장마다 반복할 운영 체크리스트가 아니다. 이 장의 중심 주장인 “prefill chunk가
두 시계의 시간을 재배분한다”를 검증하는 최소 관측 묶음이다.

## 2.5 평균이 좋아졌는데 사용자는 불만인 세 가지 장면

첫 번째 장면은 긴 prefill이 드물게 들어오는 workload다. 평균 request 대부분은 빨라 보여도
긴 prompt가 들어온 순간 decode tail이 튄다. 전체 평균 ITL은 희귀한 spike를 희석하므로
prompt-length cohort와 spike step을 따로 본다.

두 번째 장면은 queue가 짧지만 step이 긴 경우다. dashboard의 waiting request 수는 낮은데
이미 실행 중인 큰 mixed batch가 GPU를 오래 점유한다. queue length만 보고 admission 문제가
아니라고 결론 내리면 틀린다. request별 `scheduled→GPU end` 구간과 batch token 구성을 본다.

세 번째 장면은 GPU 계산이 빨라졌지만 output 경로가 따라오지 못하는 경우다. sampling kernel을
fuse하거나 graph replay로 launch gap을 줄였는데 detokenizer, Python output handler 또는 느린
client가 병목이 된다. engine token commit은 빨라졌지만 visible ITL은 그대로다. 최적화가
병목을 없애기보다 다음 계층으로 옮긴 사례다.

이 세 장면은 모두 “GPU utilization이 높다”나 “throughput이 늘었다”는 한 숫자로 설명되지
않는다. serving 최적화의 단위는 부품이 아니라 사용자의 요청 수명이다.

## 2.6 어느 시계를 보호할지 먼저 결정한다

모든 workload에 같은 scheduler 설정을 추천할 수는 없다. 긴 문서를 batch로 처리하는 offline
서비스는 TTFT보다 총 처리량을 중시할 수 있다. 대화형 서비스는 첫 반응과 매끄러운 token
진행을 보호해야 한다. 여러 tenant가 섞이면 평균보다 tail과 fairness가 중요해질 수 있다.

최적화 전에 workload와 SLO를 문장으로 적는다.

> prompt 길이 분포가 이렇고, 동시성은 이 범위이며, 첫 token의 p95와 decode ITL의 p99를
> 이 한계 안에 두면서 시간당 완료 token을 최대화한다.

이 문장이 있어야 batch token budget, chunk 크기와 admission policy의 변화가 성공인지
판정할 수 있다. 처리량이 10% 늘었지만 ITL p99가 SLO를 넘었다면 대화형 서비스에서는
실패다. 반대로 tail이 조금 늘어도 여전히 SLO 안이고 완료량이 크게 늘었다면 합리적인
교환일 수 있다.

다음 장에서는 이 논의를 goodput으로 확장한다. server가 많은 token을 계산했다는 사실과
사용자가 SLO 안에서 유효한 결과를 받았다는 사실은 다르다. preemption으로 다시 계산한
token, speculative decoding에서 버린 token, 취소 뒤 계산한 token까지 throughput 분자에
넣으면 시스템이 바빠질수록 좋아 보이는 이상한 지표가 만들어진다.

## 2.7 원문에서 두 시계를 다시 찾는 좌표

시계 이름보다 먼저 느린 사용자 하나를 고른다. 첫 응답이 느리면 request arrival에서 first visible token까지의 시작·종료 사건을 찾고, 답변이 끊기면 인접한 visible token 두 개 사이에 어느 scheduler step이 들어왔는지 찾는다. 두 경로를 같은 `latency` 로그 하나로 합치지 않는다.

### 2.7.1 metric 이름 전에 시작·종료 사건을 고정한다

구현에서는 `TTFT`와 `ITL`이라는 이름 하나가 모든 구간을 자동으로 측정하지 않는다. metric이
어느 사건에서 시작하고 끝나는지 source를 읽어야 한다. 아래 좌표는 뒤 장의 상세 분석으로
가는 입구다.

### 2.7.2 구현별 source producer를 찾는다

- vLLM의 request latency와 token timing은 고정 revision의
  [`vllm/v1/metrics/loggers.py:796-856`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/metrics/loggers.py#L796-L856)와
  scheduler output commit 경계를 함께 읽는다.
- vLLM의 step별 token admission은
  [`Scheduler.schedule`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L439)에서 확인한다.
- SGLang의 scheduling loop는
  [`Scheduler.event_loop_normal`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L1719),
  새 prefill batch 계획은
  [`get_new_batch_prefill`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L3157)에서 시작한다.
- Transformers continuous batching의 scheduler contract는
  [`Scheduler.schedule_batch`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/scheduler.py#L67)에서
  token budget과 cache budget을 분리한다.

metric 이름은 release 사이에 바뀔 수 있다. 좌표를 읽는 목적은 이름을 암기하는 것이 아니라,
어떤 상태 전이가 timestamp를 기록하는지 확인하는 데 있다.
같은 이름의 metric이라도 기록 지점이 다르면 서로 다른 지연 구간을 뜻하므로, 비교 전에 시작·종료 사건을 먼저 맞춘다.

## 2.8 숫자 하나를 바꾸기 전에 작은 시간표를 그린다

여기까지 읽고 곧바로 chunk 크기나 batch token 한도를 바꾸고 싶을 수 있다. 그러나 설정을
먼저 바꾸면 결과 숫자는 얻어도 원리는 배우기 어렵다. 우선 가상의 한 step을 손으로 배치해
보자. 이미 decode 중인 요청 세 개 `D1`~`D3`가 있고, 6,000-token prompt인 `P1`이 막
도착했다. GPU에서 decode step은 18ms, `P1` 전체 prefill은 240ms가 걸린다고 가정한다.
수치는 실제 장비의 보편값이 아니라 두 정책의 차이를 드러내기 위한 예다.

정책 A가 `P1`을 한 번에 실행하면 시간표는 단순하다.

```text
0ms        240ms  258ms  276ms
| P1 prefill | D step | D step | ...
```

`P1`은 240ms 만에 prefill을 마친다. 반면 직전에 token을 내보낸 `D1`~`D3`는 다음 token을
최소 240ms 기다린다. 이 한 번의 정지는 평균 ITL보다 p99 ITL에 훨씬 선명하게 나타난다.
정책 B가 prompt를 1,500-token chunk 네 개로 나누고 chunk 사이에 decode를 넣는다고 하자.
작아진 prefill의 효율과 추가 launch 때문에 각 chunk가 68ms 걸린다면 다음과 같다.

```text
0       68  86      154 172      240 258      326ms
| P1-1 |D| | P1-2 |D| | P1-3 |D| | P1-4 |D|
```

decode 요청이 경험하는 가장 긴 공백은 대략 `68+18`ms 근처로 줄었다. 그러나 `P1`은
326ms가 지나야 prompt를 모두 읽는다. 이것이 chunked prefill의 대가다. 실제 시스템에서는
chunk마다 같은 시간이 걸리지 않는다. attention의 KV 길이가 늘고, mixed batch의 row 수가
달라지며, CUDA Graph로 포착된 shape인지도 달라질 수 있다. 그러므로 이 계산은 예측 공식이
아니라 trace를 읽을 때 무엇을 비교할지 알려 주는 기준선이다.

실제 trace를 받으면 먼저 step마다 다음 네 값을 한 줄에 적는다.

| step | prefill query token | decode query token | GPU 구간 | 기다린 요청 |
|---:|---:|---:|---:|---|
| 1041 | 1,500 | 3 | 71ms | `D1`~`D3`, `P2` |
| 1042 | 0 | 3 | 17ms | `P1`, `P2` |
| 1043 | 1,500 | 4 | 73ms | `D1`~`D4` |

이 표에서 첫 질문은 “71ms가 느린가?”가 아니다. `P1`의 TTFT를 위해 누구의 ITL을 얼마나
미뤘고, 그 지연이 설정한 SLO 안에 있는가가 질문이다. 다음으로 동일 shape의 kernel 시간이
평소보다 길었는지, 아니면 scheduler가 애초에 큰 shape를 골랐는지를 분리한다. 전자는
kernel·clock·memory 문제일 수 있고 후자는 정책 문제다. 두 원인을 섞으면 profiler에서
정상인 kernel을 붙잡고 scheduler 문제를 찾게 된다.

### 네 번만 반복해도 드러나는 실험 행렬

무작정 여러 옵션을 sweep하지 말고 workload 축 두 개를 먼저 고정한다. 짧은 prompt와 긴
prompt, 낮은 동시성과 높은 동시성을 조합하면 최소 네 workload가 된다. 각 workload에서
baseline과 변경안을 같은 입력·같은 출력 길이 분포로 비교한다.

| workload | 먼저 드러나는 한계 | 반드시 함께 볼 값 |
|---|---|---|
| 짧은 prompt·낮은 동시성 | launch·CPU overhead | queue, GPU gap, TTFT |
| 긴 prompt·낮은 동시성 | 한 요청의 prefill 비용 | prefill duration, HBM 사용량 |
| 짧은 prompt·높은 동시성 | decode batching과 KV 압력 | ITL tail, running 수, KV 여유 |
| 긴 prompt·높은 동시성 | prefill/decode 간섭 | cohort별 TTFT, ITL, step 구성 |

네 칸을 만들면 “우리 환경에서는 빨라졌다”는 문장을 더 정확하게 고칠 수 있다. 예를 들어
chunk를 줄였더니 긴 prompt·높은 동시성에서 ITL p99는 좋아졌지만 긴 prompt의 TTFT가
나빠졌고, 나머지 두 칸에는 차이가 없었다고 말할 수 있다. 이 정도로 범위를 좁혀야 다음에
소스의 어느 분기를 읽을지도 결정된다.

## 2.9 설정 이름이 아니라 상태 변화를 번역한다

서빙 프레임워크의 명령행 옵션은 친절한 설명문처럼 보이지만, 실제로는 여러 상태 전이를
한꺼번에 바꾸는 손잡이다. `max_num_batched_tokens` 같은 이름을 보고 “batch 크기”라고만
기억하면 부족하다. 그 값이 schedule loop의 token budget을 줄이면 다음과 같은 연쇄가 생길
수 있다.

```text
step token budget 감소
  → 긴 prefill이 더 많은 chunk로 분할됨
  → 한 step의 GPU 점유 시간이 짧아질 가능성
  → decode가 끼어들 schedule 경계 증가
  → 긴 prompt가 terminal prefill에 도달하는 step 수 증가
  → graph shape·kernel 효율·CPU schedule 횟수도 함께 변화
```

그래서 옵션을 설명할 때는 기본값과 추천값만 적어서는 안 된다. 이 책에서는 이후 각 옵션을
다음 여섯 질문으로 번역한다.

1. 어느 객체의 어느 필드로 들어가는가?
2. 그 필드를 읽는 조건문과 계산식은 무엇인가?
3. waiting·running·preempted 중 어느 집합의 이동이 달라지는가?
4. query token, KV block, workspace 중 무엇의 예산이 달라지는가?
5. 그 변화가 TTFT·ITL·goodput 가운데 무엇을 어떤 방향으로 움직일 가능성이 있는가?
6. 효과를 반증하려면 어느 timestamp와 batch shape를 기록해야 하는가?

여기서 “가능성”이라는 표현은 회피가 아니다. 같은 token budget도 model 구조, prompt 분포,
GPU 세대, kernel backend와 다른 제한값 때문에 결과가 달라진다. 소스는 상태 변화의 방향을
알려 주고, trace는 실제 workload에서 그 분기가 얼마나 자주 실행됐는지 알려 준다. 둘 중
하나만으로는 운영 결론을 내릴 수 없다.

### 첫 진단을 끝내는 질문

TTFT가 나쁘다는 신고를 받았을 때 이 장을 제대로 사용했다면 곧바로 “batch를 줄이자”고
답하지 않는다. 먼저 client와 engine의 clock 경계를 맞추고, 느린 요청을 prompt 길이와
도착 당시 동시성으로 나누며, 그 요청이 기다린 step의 prefill/decode 구성을 찾는다. 그런
다음 GPU 구간이 길었는지 schedule 전 대기가 길었는지 확인한다.

ITL이 나쁠 때도 순서는 같다. spike 직전의 긴 prefill, collective stall, KV 부족에 따른
preemption, output backpressure 중 최초로 시간이 벌어진 경계를 찾는다. 발견한 경계가
scheduler라면 그제야 token budget과 chunk 정책을 읽는다. CUDA kernel이라면 동일 shape의
정상 step과 비교한다. output 경로라면 GPU 옵션을 건드리지 않는다.

이 절차의 목적은 모든 장애를 열 단계로 복잡하게 만드는 것이 아니다. 오히려 조사 범위를
가장 싼 증거로 줄이는 데 있다. 독자가 이 장에서 가져가야 할 문장은 하나다. **TTFT와 ITL은
서로 싸우는 두 숫자가 아니라, scheduler가 누구의 시간을 어디에 배치했는지를 보여 주는 두
관측창이다.**

## 2.10 SLO를 쓰면 scheduler의 목적 함수가 달라진다

“빠르게 응답한다”는 목표는 scheduler가 계산할 수 있는 조건이 아니다. 예를 들어 대화형
서비스의 목표를 다음처럼 정했다고 하자.

```text
prompt 0~2K cohort: TTFT p95 ≤ 800ms
prompt 2K~8K cohort: TTFT p95 ≤ 2.5s
active decode: ITL p99 ≤ 120ms
모든 조건을 만족한 완료 요청의 비율 ≥ 99%
```

prompt 길이 cohort를 나눈 이유는 100-token 요청과 8,000-token 요청의 prefill 비용을 같은
분포에 섞지 않기 위해서다. 짧은 요청이 많으면 전체 TTFT p95는 좋아 보이면서 긴 문서
사용자는 계속 실패할 수 있다. ITL도 request별 평균을 다시 평균내지 말고, 어느 요청과 어느
step에서 tail이 생겼는지 보존해야 한다.

이 조건 아래에서 scheduler 변경의 성공은 throughput 최대가 아니다. 1분 동안 GPU가
120,000 token을 계산했더라도 취소 뒤 계산한 token, speculative branch에서 거부된 token,
SLO를 넘겨 사용자에게 너무 늦게 도착한 결과가 많다면 유효한 완료량은 작다. 이를 개념적으로
다음처럼 분리한다.

\[
\mathrm{goodput}=\frac{\text{SLO 안에서 완료된 유효 작업}}{\text{wall-clock time}}
\]

분자의 “유효 작업”은 제품 계약에 따라 request일 수도, output token일 수도 있다. 두 정의를
섞으면 짧은 답변을 많이 끝낸 시스템과 긴 답변을 몇 개 끝낸 시스템을 공정하게 비교할 수
없다. 지표 이름보다 분자·분모와 제외 조건을 먼저 고정해야 한다.

### 평균 처리량이 12% 올랐는데 배포를 되돌리는 경우

baseline이 초당 10,000 output token, TTFT p95 720ms, ITL p99 105ms였다고 하자. 변경안은 큰
batch를 허용해 초당 11,200 token을 냈지만 TTFT p95 760ms, ITL p99 190ms가 됐다. 평균
처리량은 12% 올랐고 TTFT도 목표 안이다. 그러나 active decode의 ITL SLO 120ms를 어겼으므로
대화형 계약에서는 실패다.

여기서 batch 한도를 원래 값으로 즉시 되돌리는 것만이 유일한 답은 아니다. tail step이 긴
prefill과 겹쳤다면 prefill chunk 정책을 조정할 수 있고, 특정 prompt cohort만 원인이라면
admission class를 나눌 수 있다. 중요한 것은 190ms를 만든 step을 찾은 뒤 최소 상태 변화를
선택하는 것이다. 여러 옵션을 동시에 바꾸면 어떤 분기가 tail을 줄였는지 알 수 없다.

반대로 ITL p99가 105ms에서 115ms로 늘었지만 SLO 안이고 goodput이 12% 늘었다면 합리적인
배포일 수 있다. “latency는 낮을수록 좋다”는 말만으로는 이 선택을 설명할 수 없다. 이미
충족한 지연 여유를 처리량과 교환한 것이다. 다만 workload가 변할 때 5ms의 여유가 사라질
수 있으므로 prompt 길이와 동시성 drift를 함께 감시해야 한다.

## 2.11 tail이 생긴 한 step을 scheduler 상태로 번역한다

ITL spike가 310ms였다는 trace를 찾았다고 하자. 다음 일은 그 시각의 GPU timeline을 무작정
확대하는 것이 아니라 spike 사이에 어떤 schedule 결정이 있었는지 복원하는 것이다. 최소
장부에는 다음이 필요하다.

| 항목 | 질문 |
|---|---|
| running set | spike 전후 같은 요청들이 남아 있었는가? |
| waiting set | 긴 prompt가 언제 admission됐는가? |
| scheduled query token | prefill과 decode 몫이 각각 얼마였는가? |
| KV 상태 | allocation 실패나 preemption이 있었는가? |
| execution mode | eager·compile·graph capture·replay 중 무엇인가? |
| output 상태 | device 완료 뒤 commit·stream 지연이 있었는가? |

예를 들어 step 88에서 decode 40 token과 prefill 2,048 token을 섞었고 GPU 구간이 230ms,
뒤 output 처리에 12ms가 걸렸다고 하자. 해당 사용자의 이전 token은 step 87 끝에 나왔고 다음
token은 step 89에 포함됐다. 310ms는 자기 token 하나의 kernel 시간이 아니라, step 88의
혼합 작업과 두 scheduling 경계, step 89의 실행 일부를 포함한다.

이 사실은 최적화 위치를 바꾼다. decode attention kernel 하나를 5% 줄이는 것보다 step 88의
prefill 몫을 나누는 편이 tail에 더 큰 영향을 줄 수 있다. 반대로 step 88의 GPU가 40ms인데
scheduler process가 200ms 멈췄다면 chunk를 바꿔도 해결되지 않는다. Python CPU saturation,
IPC backpressure, garbage collection 또는 synchronization을 조사해야 한다.

### preemption은 대기만 늘리는 것이 아니라 이미 한 일을 지운다

KV 공간이 부족해 running 요청을 선점하면 해당 요청은 다음 차례를 기다릴 뿐 아니라 일부
구현·정책에서 과거 계산을 다시 해야 할 수 있다. 이때 device throughput counter에는 최초
계산과 재계산이 모두 들어가 GPU가 바빠 보인다. 하지만 사용자가 받은 새 token은 늘지 않는다.

KV pressure가 있는 실험에서는 preemption 횟수만 세지 않는다. 어떤 길이의 요청이
선점됐는지, swap·remote restore·recompute 가운데 어느 경로였는지, 재계산 token이 몇 개인지,
그 요청의 TTFT·ITL tail에 얼마가 더해졌는지 연결한다. scheduler 정책이 만든 낭비를 kernel
throughput 개선으로 가릴 수 있기 때문이다.

공정성도 같은 상태 장부에서 읽힌다. 짧은 decode가 계속 들어와 긴 prefill이 끝없이 미뤄지면
decode ITL은 좋아도 긴 요청의 TTFT가 굶는다. 반대로 먼저 들어온 긴 prompt를 무조건 끝내면
대화형 요청이 멎는다. fairness는 추상적인 미덕이 아니라 request가 runnable인데도 몇 step
동안 service를 받지 못했는지, tenant와 cohort별 slowdown이 얼마나 다른지로 측정해야 한다.

## 2.12 metric 이름에서 구현의 timestamp까지 거슬러 간다

Prometheus에 `time_to_first_token_seconds`라는 histogram이 있다고 해서 앞에서 정의한 `t_1-t_0`과
자동으로 같지는 않다. server는 client의 전송 시각을 모를 수 있고, 첫 token이라는 말도
sample이 끝난 순간, output queue에 넣은 순간, 첫 SSE frame을 쓴 순간 가운데 하나일 수 있다.
지표를 신뢰하려면 수집 코드를 다음 방향으로 역추적한다.

```text
dashboard panel
  ← PromQL과 label filter
  ← histogram/counter 등록
  ← observe/inc 호출
  ← timestamp를 저장한 상태 전이
  ← request 수명의 실제 사건
```

dashboard에서 시작하는 이유는 사용자가 실제로 본 집계 조건을 보존하기 위해서다. 같은 metric도
route, model, replica, success label을 어떻게 필터링했는지에 따라 다른 모집단을 만든다. 그다음
bucket 경계와 단위를 확인한다. p99가 가장 큰 bucket에 몰렸다면 실제 tail 크기는 dashboard가
보여 주는 숫자보다 부정확할 수 있다.

코드에서는 `observe(value)`만 찾고 끝내지 않는다. `value`의 두 항이 어디서 기록됐는지 본다.
요청 생성 시각을 API process의 wall clock으로, first token을 engine process의 monotonic
clock으로 기록했다면 직접 뺄 수 없다. process를 넘는 trace에서는 clock domain과 propagation을
명시하고, 가능한 한 한 구간은 같은 monotonic clock에서 잰다.

### histogram은 한 요청의 이야기를 보존하지 않는다

histogram은 fleet의 분포와 SLO 위반을 보는 데 좋지만 “이 ITL spike 앞에 어떤 prefill이
있었는가”에는 답하지 못한다. label에 request ID를 넣으면 cardinality가 폭발한다. 그래서
metric과 trace의 역할을 나눈다.

- metric은 model·route·길이 cohort처럼 제한된 label로 분포와 추세를 본다.
- trace는 sampling된 요청의 phase timestamp와 step identity를 잇는다.
- structured event는 schedule 구성과 preemption처럼 고차원 상태를 짧은 기간 보존한다.
- profiler는 trace가 가리킨 좁은 device 구간에만 붙인다.

예를 들어 ITL p99 alert가 울리면 histogram에서 영향받은 replica와 cohort를 고른다. exemplar나
sampling trace로 request 하나를 찾고, 그 request의 두 token 사이 step ID를 얻는다. step
event에서 prefill/decode token과 KV 상태를 복원한 뒤, GPU 구간이 비정상일 때만 profiler를
연다. 도구가 많아지는 것이 아니라 싼 관측에서 비싼 관측으로 범위를 줄이는 순서다.

### label 하나가 결론을 뒤집는 사례

전체 ITL p99가 90ms에서 140ms로 올랐지만 model별로 나누니 새로 배포한 대형 model만 260ms,
기존 model은 92ms였다고 하자. fleet 평균만 보고 공용 gateway나 network를 의심하면 조사
범위가 너무 넓다. 반대로 `model` label이 사용자가 요청한 alias인지 실제 resolve된 artifact인지
확인하지 않으면 두 revision이 한 bucket에 섞일 수 있다.

prompt-length label도 raw 길이를 그대로 넣지 않고 경계가 고정된 cohort를 쓴다. tenant ID나
request ID 같은 무한 label은 metric에 넣지 않는다. 상세 상관관계는 trace attribute나 제한된
event store에서 찾는다. 관측성 설계 역시 GPU memory와 마찬가지로 예산을 배분하는 문제다.

## 2.13 이 장을 운영에서 사용하는 30분 조사 순서

실제 장애에서는 완벽한 trace가 갖춰질 때까지 기다릴 수 없다. 다음 순서는 제한된 증거로도
잘못된 층을 최적화하는 일을 줄인다. 시간은 엄격한 제한이 아니라 조사 우선순위를 나타낸다.

### 0~5분: 증상과 clock을 고정한다

“느리다”를 TTFT, ITL, total latency 가운데 하나로 바꾸고 p50인지 tail인지 적는다. client
측정인지 engine 측정인지, 시작·끝 사건이 무엇인지 확인한다. 영향받은 model, route, prompt
길이, 동시성과 시작 시각을 좁힌다. 이 단계에서 원인을 선언하지 않는다.

### 5~10분: queue와 실행을 나눈다

API receive→admission→first schedule→GPU end→token commit→client receive 가운데 확보 가능한
timestamp를 놓는다. first schedule 전이 길면 admission과 scheduler를 우선하고, GPU 구간이
길면 batch shape와 backend를 본다. commit 뒤가 길면 output 경로를 본다. timestamp가 없다면
가장 중요한 경계 한두 개만 임시 계측한다.

### 10~20분: 느린 step의 구성을 복원한다

spike 시점의 running/waiting 수, scheduled prefill/decode token, free KV block, preemption,
execution mode를 정상 step과 비교한다. 정상 비교군은 같은 model과 비슷한 query/KV shape여야
한다. 트래픽이 달라졌는지와 같은 workload에서 시스템이 느려졌는지를 구분한다.

### 20~30분: 하나의 가설을 반증한다

긴 prefill 간섭이 가설이면 작은 canary에서 chunk 한도 하나만 바꾸고 TTFT·ITL 두 분포를
동시에 본다. kernel 회귀가 가설이면 같은 shape와 backend의 duration을 정상 revision과
비교한다. output backpressure가 가설이면 local fast sink와 실제 client를 비교한다. 결과가
가설과 반대로 움직이면 설정을 더 돌리지 말고 분기점으로 돌아간다.

조사 기록의 마지막에는 반드시 부작용을 적는다. chunk를 줄여 ITL을 보호했지만 긴 요청의
TTFT와 launch 수가 늘었는지, batch를 키워 throughput을 얻었지만 KV pressure와 preemption이
늘었는지 확인한다. serving 최적화는 비용을 없애기보다 시간·memory·복잡성을 다른 위치로
옮기는 경우가 많다.

이제 독자는 `TTFT가 느리다`는 신고를 `어느 clock의 어느 구간이, 어떤 workload에서, 어느
schedule 결정 때문에 길어졌는가`라는 질문으로 바꿀 수 있다. 다음 장의 goodput은 여기서 한
걸음 더 나아간다. 빨리 계산한 token이 실제로 완료됐는지, SLO 안에 도착했는지, 취소와
재계산으로 버려지지 않았는지를 분자에서 다시 따진다.

### 재현 workload는 평균 길이 두 개로 만들지 않는다

운영 trace에서 prompt 평균 1,200 token, output 평균 180 token이라는 숫자를 얻었다고 하자.
모든 요청을 정확히 1,200/180으로 만든 benchmark는 평균은 보존하지만 scheduler가 겪는
간섭은 지운다. 실제 서비스에는 50-token 대화와 8,000-token 문서가 섞이고, 일부 사용자는
10 token 뒤 취소하며, streaming client의 소비 속도도 다르다. tail은 이 혼합에서 생긴다.

재현 workload에는 적어도 다음 분포를 보존한다.

- prompt와 output 길이의 joint distribution
- 요청 도착 간격과 burst 크기
- 동시 active decode 수
- prefix 중복률과 cache hit 가능성
- 취소 시점과 stop reason
- model·adapter·tenant 혼합

joint distribution이 중요한 이유는 긴 prompt가 항상 긴 답을 요구하지 않기 때문이다. 두
길이를 독립적으로 무작위 생성하면 운영에 거의 없는 조합이 늘어난다. arrival도 고정 concurrency
하나로 대체하지 않는다. closed-loop benchmark는 응답이 끝나야 다음 요청을 보내므로 서버가
느려질수록 입력 부하도 줄어든다. open-loop arrival은 정해진 시간에 요청을 보내 queue가
쌓이는 모습을 드러내지만 과부하 때 무한히 밀릴 수 있다. 무엇을 재현하는지 명시해야 한다.

민감한 prompt 원문을 저장할 수 없다면 token 길이, 안전한 prefix class, modality, sampling
계약과 arrival timestamp를 익명화해 replay shape를 만든다. 다만 임의 token ID는 실제 언어의
token frequency와 special-token 배치를 보존하지 않을 수 있다. prefix cache나 tokenizer
비용을 보려는 실험에서는 synthetic ID만으로 충분하지 않다. workload를 익명화하는 방식도
검증하려는 계층에 맞춰야 한다.

### 변경 전후를 같은 결정 기록에서 비교한다

설정 하나를 바꿀 때 다음과 같은 작은 장부를 먼저 쓴다.

| 항목 | 변경 전 가설 | 변경 뒤 예상 | 틀렸음을 보일 증거 |
|---|---|---|---|
| token budget | 긴 prefill step이 decode를 막음 | step GPU 시간이 감소 | step 구성은 같고 CPU gap만 증가 |
| chunk 수 | decode 삽입 경계가 부족함 | ITL tail 감소 | spike가 output 구간에 그대로 남음 |
| 긴 prompt TTFT | chunk overhead가 늘 수 있음 | 일부 증가 허용 | SLO를 넘거나 goodput 급락 |
| launch/graph | shape 수가 늘 수 있음 | gap·capture 증가 가능 | replay 비율과 gap 변화 없음 |

이 표의 핵심은 좋은 결과뿐 아니라 가설을 버릴 조건을 미리 적는 데 있다. 결과를 본 뒤 설명을
만들면 거의 모든 변화가 그럴듯해 보인다. 예상과 반대인 지표가 나오면 소스의 다른 분기를
읽거나 최초 divergence를 다시 찾아야 한다.

변경은 canary에서 시작하되 canary의 traffic mix가 본 fleet와 같은지 확인한다. 작은 replica는
KV pool 크기와 tensor parallel topology가 달라 scheduler 분기가 달라질 수 있다. 같은 옵션
값도 총 block 수가 다르면 preemption threshold를 다르게 만난다. 따라서 config diff와 함께
model artifact, GPU topology, cache capacity와 traffic cohort를 기록한다.

### 차이가 없다는 결과도 원인을 좁힌다

chunk 크기를 절반으로 줄였는데 step 구성과 TTFT·ITL이 모두 같다면 옵션이 무효라는 결론을
서두르지 않는다. 해당 workload의 prompt가 원래 chunk보다 짧았거나, 더 작은 별도 한도가
먼저 적용됐거나, feature 조건 때문에 chunked path가 선택되지 않았을 수 있다. config가
parse됐는지, effective field가 무엇인지, 조건문이 실행됐는지를 차례로 본다.

반대로 schedule output은 달라졌는데 사용자 지표가 같다면 scheduler 구간이 critical path가
아니었을 가능성이 커진다. network tail이 지배하거나 GPU가 작은 shape에서 효율을 잃어 얻은
시간을 상쇄했을 수 있다. “효과 없음”은 실패한 실험이 아니라 상태 변화와 최종 관측 사이의
다음 병목을 찾는 증거다.

### 이 장에서 독자가 설명할 수 있어야 하는 것

TTFT는 첫 token 이전의 모든 시간을 하나로 부르는 이름이지만, 원인을 찾을 때는 render,
queue, scheduling, prefill, commit과 전달 구간으로 쪼개야 한다. ITL은 decode kernel 시간과
같지 않으며, token 사이에 끼어든 다른 step과 output 경로를 포함한다. prefill은 token 축의
병렬성이 크고 decode는 과거 KV를 반복해 읽으므로 같은 model forward라도 scheduler가 다루는
비용 모양이 다르다.

batch와 chunk 옵션의 의미도 이제 숫자 하나가 아니다. 그것은 이번 step의 token 구성,
GPU 점유 시간, schedule 경계, KV 압력과 shape 집합을 바꾸는 상태 전이다. 효과는 workload와
SLO에 따라 달라지고, TTFT 하나만 개선됐다고 성공할 수 없다. ITL tail, goodput, preemption과
부작용을 함께 봐야 한다.

마지막으로 source와 metric은 서로를 대체하지 않는다. source는 옵션이 어떤 조건에서 어떤
상태를 바꾸도록 구현됐는지 말한다. metric과 trace는 운영 workload가 그 조건을 실제로
얼마나 밟았고 시간이 어디서 벌어졌는지 말한다. 둘을 request와 step identity로 연결했을 때만
“왜 빨라졌는가”와 “왜 느려졌는가”를 같은 언어로 설명할 수 있다.

다음 장에서 goodput을 별도로 다루는 이유도 여기에 있다. scheduler가 계산한 token 중에는
사용자에게 늦게 도착한 것, 취소 뒤 만들어진 것, rollback된 것과 재계산된 것이 섞일 수 있다.
GPU를 바쁘게 만드는 일과 유효한 요청을 SLO 안에 끝내는 일을 분리해야 최적화의 목적지가
선명해진다.

### 설정 리뷰에서 말로 끝내지 않는 방법

운영 변경안에 “TTFT 개선을 위해 batch 설정을 조정한다”라고만 쓰여 있다면 아직 리뷰할 수
없다. 다음과 같이 상태와 관측을 포함한 문장으로 고친다.

> 긴 prompt가 들어온 mixed workload에서 한 step의 prefill token 몫을 낮춘다. 그러면 긴
> prefill이 더 많은 schedule 경계로 나뉘어 active decode가 그 사이에 선택될 기회가 늘어난다.
> 예상 효과는 ITL p99 감소이고, 예상 비용은 긴 prompt TTFT·scheduler 호출·kernel launch
> 증가다. step event의 prefill/decode 구성과 두 latency cohort로 가설을 검증한다.

이 문장에는 값 자체가 빠져 있다. 값은 hardware와 workload 측정으로 채워야 하기 때문이다.
반면 인과 사슬은 소스에서 확인할 수 있다. config가 scheduler field로 전달되는지, budget
계산에서 어떤 상한과 `min`을 이루는지, 남은 prompt가 다음 waiting/running 상태에 어떻게
보존되는지 읽는다. 만약 다른 상한이 항상 더 작다면 바꾸려는 값은 현재 workload에서 아무
효과가 없다.

옵션 두 개가 같은 결과를 제한할 때는 effective constraint를 계산한다. 예를 들어 step token
한도와 sequence별 chunk 한도가 함께 있다면 실제 prefill 몫은 두 값, 남은 prompt, KV 여유와
다른 scheduled token의 함수다. 문서의 기본값 두 개만 비교해서는 어느 제한이 active인지 알
수 없다. 느린 step에서 계산 직전의 입력값과 계산 뒤 schedule output을 함께 남긴다.

변경 뒤에는 설정이 적용됐다는 로그만으로 끝내지 않는다. config parse 성공, scheduler의
effective field, 해당 조건문 통과, schedule output 변화, device shape 변화, 사용자 metric
변화를 순서대로 확인한다. 이 다섯 단계 중 최초로 차이가 사라진 곳이 “왜 효과가 없었는가”의
답이다.

```text
config diff
  → effective scheduler field
  → branch·budget 계산
  → scheduled token/KV 상태
  → runner launch shape
  → TTFT·ITL·goodput
```

이 사슬은 vLLM과 SGLang의 옵션 이름이 달라도 그대로 쓸 수 있다. Transformers continuous
batching이나 llama.cpp server의 batch 제한을 볼 때도 마찬가지다. 동일 이름을 찾는 것이 아니라
어느 예산을 누가 읽고, 어느 collection의 어느 요청을 전진시키는지 찾는다.

마지막으로 rollback 조건을 변경 전에 정한다. ITL은 좋아졌지만 긴 prompt TTFT p95가 SLO를
넘거나 preemption과 오류가 증가하면 자동 또는 수동으로 되돌린다. rollback 자체가 cache나
graph state를 즉시 원상복구하는지도 확인한다. config만 되돌렸는데 이미 capture된 shape나
queue backlog가 남아 잠시 다른 동작을 할 수 있기 때문이다.

이제 옵션 설명은 추천 숫자의 목록이 아니라 작은 실행 모델이 된다. 독자는 값을 바꾸기 전에
어떤 상태가 달라질지 예측하고, 값이 적용되지 않은 경우 어느 경계에서 사슬이 끊겼는지 찾고,
효과가 나온 경우 그 대가가 다른 사용자의 시계로 옮겨 가지 않았는지 검증할 수 있다.

간단한 구두 시험으로 장을 마쳐 보자. “GPU utilization이 60%라 batch를 키웠더니 utilization은
90%가 됐지만 대화가 끊긴다”는 보고를 받았다. utilization이 좋아졌다는 사실은 device가 더
오래 일했다는 뜻이지 유효한 token을 더 제때 보냈다는 뜻은 아니다. 먼저 끊김을 ITL spike로
정의하고, spike 사이 step의 prefill/decode 구성과 GPU duration을 찾는다. 큰 batch가 한 step의
점유 시간을 늘렸다면 원인 사슬이 맞는다. output queue가 벌어졌다면 다른 사슬이다.

원인 사슬이 맞을 때도 “batch를 줄인다”로 끝내지 않는다. decode의 ITL SLO를 보호하면서
prefill을 어느 크기로 나눌지, 긴 prompt의 TTFT를 얼마나 양보할지 결정한다. 처리량은 completed
goodput으로 다시 재고, 설정 변경이 preemption이나 graph fallback을 늘리지 않았는지 확인한다.
이 설명을 trace의 request·step ID와 함께 제시할 수 있어야 재현 가능한 운영 지식이 된다.

반대로 질문에 답하면서 어느 timestamp가 없는지 발견했다면 그것도 성과다. 다음 장애 전에
hot path를 망치지 않는 작은 event를 추가할 수 있다. `request_id`, `step_id`, phase, query
token, KV length, monotonic timestamp만으로도 많은 가설을 가를 수 있다. 원문 prompt나 tensor
전체를 남기지 않아도 된다.

TTFT와 ITL의 긴장은 제거해야 할 버그가 아니라 공유 GPU에서 서로 다른 작업을 함께 서비스할
때 생기는 기본 제약이다. 좋은 scheduler는 제약을 숨기지 않는다. workload와 SLO에 따라 어느
시간을 보호할지 명시하고, 선택의 비용을 metric과 trace에서 보이게 만든다. 좋은 운영자 역시
한 숫자를 최적화하지 않는다. 요청의 전체 여행에서 시간이 누구에게서 누구에게로 이동했는지
설명한다.

이 장의 실전 산출물은 추천 설정표가 아니다. clock 정의가 적힌 timeline, 느린
request와 step을 잇는 상관 키, workload cohort, 변경 전의 반증 가능한 가설, 변경 뒤 함께
확인할 TTFT·ITL·goodput 장부다. 이 다섯 가지가 있으면 새로운 GPU나 release에서 기본값이
바뀌어도 다시 판단할 수 있다. 없으면 우연히 잘 맞은 숫자를 다른 환경에 복사하게 된다.

## 2.14 평균 arrival rate가 같아도 tail은 달라진다

지금까지의 시간표에 arrival distribution을 넣어야 실제 chunk 크기를 결정할 수 있다. 평균
10 requests/s인 두 workload를 비교하자. A는 100 ms마다 한 건이 거의 일정하게 도착한다. B는
9초 동안 조용하다가 1초에 100건이 몰린다. 10초 평균은 같지만 B의 burst 직후 waiting queue는
최대 100건 가까이 쌓인다. Scheduler가 평균만 보고 token budget을 정하면 A에서는 안정적이던
chunk가 B에서 decode를 여러 step 기다리게 한다.

첫 계산의 산출물은 평균 prompt 길이가 아니라 도착 시각과 prompt token을 함께 가진 trace다.
각 request를 `(arrival, prompt_tokens, expected_output, priority, deadline)`로 적고, scheduler step
마다 admitted prefill token과 running decode token을 놓는다. 이 trace에 128, 256, 512-token
chunk를 적용해 TTFT p99와 ITL deadline miss가 어디서 갈리는지 손으로 닫아 보자.

한 step의 token budget은 1,024, running decode는 64건, step은 10 ms라고 하자. Decode를 한 번씩
진행하면 prefill budget은 960이다. Burst의 100개 prompt는 모두 1,024 token이다. 완성 chunk만
선택하는 단순 fixture에서 512는 step당 한 request, 256은 세 request, 128은 일곱 request에 첫
service를 준다. p99 request가 첫 chunk를 받는 queue wait는 대략 1,000 ms, 340 ms,
150 ms다. 작은 chunk가 burst의 첫 service를 넓게 나누는 효과다.

대신 prompt 하나를 끝내는 round는 각각 2, 4, 8회다. Chunk 경계마다 request당 metadata와 launch
overhead가 35 μs라면 100건의 추가 비용은 7, 14, 28 ms다. 또 512 두 개를 atomic하게 묶어
budget을 쓰면 decode가 한 step 밀려 ITL이 20 ms가 된다. 256/128이 매 경계에서 decode를 다시
admit하면 10 ms를 지킬 수 있지만 step 자체가 10.5/11 ms로 늘 수 있다. TTFT SLO 500 ms,
ITL SLO 15 ms라면 이 가정에서 512는 탈락하고 256은 첫 실험 후보이며 128은 overhead 여유가
작다. 이 결과는 추천값이 아니라 가정이 바뀌면 다시 계산할 기준선이다.

같은 평균 도착률이 왜 다른 chunk를 요구하는지 작은 창 두 개로 확인하자. 창 A에는 100 ms마다
요청 하나가 들어와 1초 동안 정확히 10개가 도착한다. 창 B에도 1초 동안 10개가 오지만 첫 100 ms에
8개, 나머지 900 ms에 2개가 온다. 둘 다 평균은 10 req/s다. 하지만 A의 첫 요청은 앞에 대기자가
없고, B의 여덟 번째 요청은 앞선 일곱 prompt와 같은 scheduler budget을 놓고 경쟁한다. 평균만
고정한 benchmark가 burst의 TTFT tail을 숨기는 이유다.

각 prompt가 512 token이고 step budget에서 prefill 512 token을 한 번에 처리하면 8개 burst의 첫
서비스 시각을 단순화해 `0, 20, 40, 60, 80, 100, 120, 140 ms`라 하자. Network와 decode에 고정
360 ms가 더해지면 TTFT는 `360…500 ms`다. 여덟 표본에서는 p50이 가운데 두 값의 평균인 430 ms,
p75는 대략 465 ms, 최대는 500 ms다. 표본이 작으므로 이 숫자를 운영 p99라고 부르지 않는다.
여기서는 같은 총량 안에서도 순서 통계가 어떻게 생기는지 손으로 보는 것이 목적이다.

Chunk를 256으로 줄여 각 요청이 첫 절반을 번갈아 받게 하면 first-service를 `0, 11, 22, 33, 44,
55, 66, 77 ms`로 앞당길 수 있다. 대신 각 prompt는 두 차례 scheduler를 지나고 마지막 절반의 완료는
늦어진다. 고정 360 ms를 더한 TTFT 후보는 `360…437 ms`로 좁아지지만, decode가 매 step 사이에
끼어 step 비용이 10 ms에서 11 ms로 늘었다면 ITL도 10 ms에서 11 ms로 오른다. 128 chunk는 첫
서비스를 더 촘촘히 만들 수 있으나 launch·queue bookkeeping이 네 번 발생한다.

결정은 평균 하나가 아니라 tail 계약을 적용한 뒤 내린다. TTFT 한도가 450 ms이고 ITL 한도가 12 ms라면
512는 burst 뒤쪽 세 요청이 위험하고 256은 두 계약 안에 남는다. 128의 실제 step이 12.5 ms라면
TTFT가 더 좋아도 ITL 계약 때문에 탈락한다. 반대로 도착이 창 A처럼 고르게 퍼지고 512의 최대 TTFT가
410 ms라면 256의 추가 bookkeeping을 지불할 이유가 없다. Chunk는 모델의 영구 성질이 아니라 도착
분포와 두 percentile 계약 사이의 선택이다.

| arrival 창 | chunk | first-service 범위 | 계산된 TTFT 범위 | ITL 후보 | 판정 이유 |
|---|---:|---:|---:|---:|---|
| A: 100 ms 간격 | 512 | 0–50 ms | 360–410 ms | 10 ms | 두 SLO 안, 분할 이득 작음 |
| B: 첫 100 ms에 8건 | 512 | 0–140 ms | 360–500 ms | 10 ms | burst 뒤쪽 TTFT 초과 |
| B: 첫 100 ms에 8건 | 256 | 0–77 ms | 360–437 ms | 11 ms | 두 SLO를 함께 만족 |
| B: 첫 100 ms에 8건 | 128 | 0–60 ms | 360–420 ms | 12.5 ms | ITL guardrail 초과 |

이 표의 숫자는 production 예측기가 아니다. 실제 검증에서는 prompt 길이 분포, cache hit, decode row,
graph bucket과 step duration을 trace에서 다시 채운다. 특히 percentile은 요청별 최종 TTFT 표본에서
계산해야 한다. Step 평균 11 ms에 p99 40 ms stall이 섞이면 위의 균등 간격 가정은 깨진다. 그러므로
후보 256을 적용하기 전에 burst trace를 replay하고, TTFT p50·p95·p99와 ITL p95·p99, prompt completion을
같은 cohort에서 비교한다.

선택 뒤에는 counterfactual도 남긴다. 동일 arrival timestamp와 token 길이를 512·256·128 정책에
각각 재생하고, first service가 앞당겨진 요청과 completion이 늦어진 요청을 같은 request ID로 잇는다.
256이 이긴 이유를 “작아서”라고 쓰지 않는다. Burst의 여덟 요청 사이에 prefill frontier를 나누어
TTFT tail을 줄였고, 증가한 step 비용이 ITL 12 ms 안에 남았기 때문이라고 쓴다. Arrival 분포나 kernel
시간이 바뀌면 이 인과를 다시 계산해야 한다.

승인 기록에는 평균 도착률만 쓰지 않고 burst 폭, burst 간격, 표본 수와 percentile 계산법을 함께
남긴다. 그래야 다음 release에서 같은 비교를 재현할 수 있다.

소스 walk는 옵션 이름이 아니라 budget 소비 지점에서 시작한다. vLLM commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`의
[`Scheduler.schedule`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L439)에서
running과 waiting이 budget을 소비하는 순서와 scheduled token 절단을 읽는다. SGLang commit

`71de97b264b04dcd514cf904003028aefe9775c8`의
[`event_loop_normal`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L1719)과
[`get_new_batch_prefill`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L3157)에서는
iteration entry와 새 prefill batch 선택을 잇는다.

두 구현의 이름을 같다고 하지 않고 step마다
budget before, decode 소비, request age/frontier, scheduled tokens와 remainder를 같은 표에 둔다.

| step | decode | oldest waiting age | selected prefill | remainder | step ms |
|---:|---:|---:|---|---:|---:|
| 0 | 64 | 0 ms | R0–R2×256 | 192 | 10.5 |
| 1 | 64 | 10.5 ms | R3–R5×256 | 192 | 10.5 |
| 2 | 64 | 21 ms | R6–R8×256 | 192 | 10.5 |

마지막 request까지 이 결정 기록을 만든 뒤 first-service와 prompt-completion을 따로 percentile한다.
First-service는 실제 TTFT가 아니다. 남은 prefill, 첫 decode, detokenization과 network 시간이 더해진다.
다만 first-service p99가 이미 SLO를 넘으면 뒤 구간이 이를 되돌릴 수 없어 scheduler 원인을 빠르게
확정할 수 있다.

### 사건 A: 평균 부하는 같았지만 burst의 99번째 요청이 SLO를 넘었다

배포 전 benchmark는 100 ms 간격으로 10 rps를 만들었고 256 chunk에서 TTFT p99 410 ms, ITL p99
12 ms였다. 운영 traffic도 10분 평균 10 rps였으므로 같은 결과를 기대했다. 그러나 운영에서는
매분 정각 batch job이 100건을 1초 안에 보냈고 interactive request와 같은 queue를 사용했다.
분당 평균만 보면 약 1.67 rps가 추가됐지만 그 1초의 offered load는 100 rps였다.

Trace를 복원하니 burst의 99번째 request가 first service를 347 ms에 받고, 1,024-token prompt의
나머지 세 round와 첫 decode·detokenize에 238 ms를 더 써 TTFT 585 ms가 됐다. Regular workload의
p99는 418 ms였다. GPU kernel p99와 평균 utilization은 거의 같았다. 최초 불일치는 burst 시작
직후 `oldest_waiting_age`와 아직 한 번도 service받지 못한 request 수가 함께 상승한 지점이었다.

512에서 이 사건을 replay하면 first service만 약 1초라 즉시 탈락한다. 128은 first service를
155 ms로 줄였지만 step duration이 11.1 ms로 늘고 8 round를 돌아 prompt completion이 510 ms,
최종 TTFT는 672 ms였다. 256은 585 ms였다. 어느 고정 chunk도 500 ms SLO를 만족하지 못했다.
따라서 수정은 128로 낮추는 것이 아니라 burst cohort를 admission에서 분리하고 waiting age가
250 ms를 넘은 새 prompt에 first-chunk 우선권을 주는 것이었다. 이후 round는 다시 정상 fair
queue로 돌아갔다.

수정 뒤 burst p99 TTFT는 468 ms, regular p99는 426 ms, decode ITL p99는 13.2 ms였다. 평균
throughput은 1.4% 낮아졌다. 이 변경은 세 SLO를 통과했지만, 오래 기다린 request가 계속 first
chunk만 받고 completion이 굶는지 확인해야 했다. `first_service_age`뿐 아니라
`prompt_completion_age` p99와 round count 분포를 acceptance에 포함한 이유다.

### 사건 B: TTFT를 지킨 설정이 decode tail을 주기적으로 멈췄다

두 번째 서비스는 burst prompt에 512-token first chunk를 두 개 연속 실행하는 fast-prefill
경로를 사용했다. TTFT p50은 18% 좋아졌지만 streaming ITL p99가 14 ms에서 31 ms로 나빠졌다.
느린 token은 무작위가 아니라 burst 직후 두 step마다 나타났다. Scheduler 결정 기록에서 decode
budget이 64로 기록됐지만 실제 batch publish에는 prefill atomic group이 먼저 1,024를 소비해
decode row가 다음 launch로 미뤄졌다. Config snapshot의 `chunk=512`만 보면 찾을 수 없는 상태
변화였다.

여기서 consumer는 parser가 아니라 scheduler의 budget decrement와 batch builder다. Source
review에서는 requested chunk, normalized maximum, request frontier에 적용된 actual scheduled
tokens, atomic grouping 후 remainder를 나눈다. 값 512가 저장됐다는 사실보다 그 값 때문에
`decode_admitted=false`가 된 branch가 중요하다. vLLM과 SGLang 좌표에서 같은 이름을 찾는 대신
각 구현이 이 네 상태를 어디서 갱신하는지 표시한다.

수정은 atomic group 사이에 decode checkpoint를 넣었다. TTFT p50 개선은 18%에서 12%로 줄었지만
ITL p99는 14.8 ms로 회복됐다. Acceptance는 TTFT 평균이 아니라 regular/burst cohort별 TTFT p50·p99,
active decode 수별 ITL p99, prompt completion p99, scheduled-but-not-launched token을 함께 본다.
Burst만 고친 뒤 regular short prompt가 퇴행하거나 decode 수 256에서 checkpoint overhead가 커지는
경우를 막는다.

### arrival에서 decision까지 같은 기록으로 잇는다

Incident bundle에는 원본 prompt를 넣지 않아도 된다. 다음 열이면 같은 scheduler 결정을 재생할
수 있다.

```text
arrival_ns, request_seq, prompt_tokens, output_bucket, deadline_ns
step_seq, budget_before, active_decode, oldest_waiting_age
candidate_request, frontier_before, requested_chunk, scheduled_tokens
atomic_group, budget_after, admitted_or_rejected, reason
step_start_ns, step_end_ns, first_service_ns, prompt_done_ns, first_token_ns
```

요약 percentile은 이 원장에서 다시 계산할 수 있어야 한다. p99 표본이 100개뿐이면 사실상 가장
느린 한두 건이 결론을 정하므로 sample count와 quantile method도 기록한다. Histogram bucket이
500 ms 바로 아래와 1초뿐이라면 585 ms와 990 ms를 구분하지 못한다. 이때 sampled trace로 tail
shape를 보강하고 dashboard 값만으로 chunk 후보를 선택하지 않는다.

최종 판정문에는 workload shape와 state mutation이 함께 들어간다. “평균 10 rps는 같았지만
1초 100-request burst가 waiting age를 만들었고 256 chunk에서 99번째 request의 first service와
네 round가 TTFT를 585 ms로 늘렸다. First-chunk age guard와 round fairness를 적용해 burst p99를
468 ms로 줄였으며 regular TTFT, decode ITL과 prompt completion guard를 함께 통과했다.” 이 문장이
있으면 다른 배포에서 arrival distribution이 바뀌었을 때 숫자를 복사하지 않고 계산을 반복할 수
있다.

### 분당 평균과 고정 histogram이 burst를 지우는 방식

Arrival distribution을 추정할 때 scrape interval당 request 수만 저장하면 간격 안의 순서를 잃는다.
60초 counter 증가가 600이면 10 rps지만, 600건이 고르게 왔는지 마지막 2초에 몰렸는지 알 수 없다.
두 workload의 scheduler queue는 전혀 다르다. Burstiness를 보려면 짧은 bucket의 arrival count,
inter-arrival time 표본, 동시에 열린 request 수 중 적어도 하나가 필요하다.

1초 bucket도 항상 충분하지 않다. 200 ms에 40건이 몰리고 나머지 800 ms가 비어 있으면 1초
평균은 40 rps지만 순간 arrival은 200 rps다. Scheduler가 10 ms step마다 세 prompt에 first chunk를
줄 수 있다면 40건을 한 번씩 service하는 데 최소 14 step, 약 147 ms가 걸린다. 1초 평균만으로
capacity 40 rps와 비교하면 queue가 없을 것처럼 보인다. Admission과 chunk 판단에 쓰는 시간
해상도는 scheduler가 결정을 바꾸는 step과 burst 지속시간보다 충분히 작아야 한다.

p99도 표본 수 없이 읽지 않는다. 5분 창에 request가 80개라면 nearest-rank p99는 사실상 가장
느린 한 건이다. 그 한 건이 network handshake였는지 scheduler queue였는지 cohort를 확인하지
않고 chunk를 바꾸면 원인이 아닌 경로를 튜닝한다. 10,000개 표본에서 p99는 느린 100개를
대표하므로 상대적으로 안정적이지만, release 전후 population과 prompt length가 같아야 비교할
수 있다.

Histogram 경계가 0.25, 0.5, 1, 2초일 때 501 ms와 999 ms는 모두 같은 bucket이다. 10,000건 중
9,850건이 0.5초 이하, 120건이 0.5~1초, 30건이 1~2초라면 p99는 0.5~1초 bucket 안에 있다.
Prometheus식 bucket interpolation이 약 0.625초를 내더라도 실제 100번째 tail이 0.51초인지
0.94초인지 알 수 없다. SLO가 600 ms라면 이 histogram만으로 pass/fail을 승인하지 않는다.
SLO 주변에 더 촘촘한 bucket을 두거나 bounded sampled trace로 정확한 값을 보강한다.

### adaptive chunk는 percentile을 읽고 scheduler state를 바꾸는 제어기다

고정 chunk 세 개가 workload 전체를 만족하지 못하면 adaptive policy를 고려할 수 있다. 하지만
“queue가 길면 chunk를 줄인다”는 문장만으로 구현하면 oscillation을 만든다. 입력, state, action,
guard와 rollback을 명시해야 한다.

입력은 최근 30초의 first-service p95/p99, active decode ITL p99, oldest waiting age, prompt frontier
분포다. Controller state는 current chunk, last transition time과 consecutive violation count다.
Action은 512→256→128처럼 한 단계만 이동한다. Guard는 ITL p99 15 ms와 prompt completion p99
800 ms다. Rollback은 두 guard 중 하나가 연속 세 창 위반하거나 scheduled-token goodput이 기준선
대비 3% 이상 떨어질 때다.

예를 들어 현재 512에서 first-service p99 720 ms가 세 창 연속 관측되고 ITL은 12 ms라고 하자.
256으로 내린 뒤 first-service는 390 ms, ITL은 13.5 ms, completion은 710 ms라 통과한다. 다시
128로 내리면 first-service는 210 ms지만 ITL 16.2 ms와 completion 845 ms가 되어 rollback한다.
Controller가 first-service 하나만 최소화하면 128을 선택하지만 전체 SLO는 256을 선택한다.

Source에서 확인할 mutation은 config object의 숫자가 아니다. 다음 scheduling iteration이 실제
읽는 effective chunk, 이미 partial prefill 중인 request의 frontier에 새 limit을 적용하는 시점,
graph/batch bucket 재선택, metric의 policy generation이다. Hot update가 새 request에만 적용된다면
old/new cohort를 나눠야 한다. 모든 running request에 즉시 적용한다면 batch shape 변화와 fairness를
검증한다. vLLM `Scheduler.schedule`과 SGLang `get_new_batch_prefill`의 scheduled token 결과가
controller generation과 같은 trace에 나타나야 “설정이 적용됐다”고 말할 수 있다.

Adaptive 실험도 같은 기록에 다음 한 행을 매 control window마다 남긴다.

```text
window, arrival_p50/p99, active_decode_p99, oldest_age_p99,
first_service_p99, prompt_done_p99, itl_p99, goodput,
requested_chunk, effective_chunk, policy_generation,
transition_reason, guard_result, rollback_reason
```

승인은 최소 regular, microburst, sustained overload 세 replay를 통과해야 한다. Sustained overload에서
모든 SLO를 지키는 불가능한 약속 대신 admission reject가 어느 cohort에 적용되는지 기록한다.
Controller가 128에 붙어 있어도 queue가 계속 자라면 chunk 문제가 아니라 offered load가 capacity를
넘은 것이다. 이때 reject/load shedding으로 넘어가며 chunk를 더 작은 임의값으로 밀지 않는다.

완료 판정은 “adaptive가 p99를 낮췄다”가 아니다. “30초 창 세 번의 first-service 위반이 policy
generation 7의 512→256 mutation을 만들었고 실제 scheduled token trace가 이를 확인했다. Burst
TTFT p99는 720→390 ms, ITL은 12→13.5 ms, completion은 710 ms로 guard를 통과했다. 128 실험은
ITL 16.2 ms로 자동 rollback했고 regular·overload replay에서 oscillation과 starvation이 없었다”다.
관측, consumer, state mutation, 효과와 철회가 한 문장에 이어진다.

### chunk 효과처럼 보이는 confounder를 먼저 제거한다

정책 전환 시점에 model revision, CUDA graph warmup, prefix cache hit, prompt length와 output
concurrency가 함께 바뀌면 p99 개선을 chunk에 귀속할 수 없다. 특히 burst 뒤에는 동일 prefix가
반복돼 cache hit가 늘 수 있다. 512 baseline은 cold miss, 256 candidate는 warm hit라면 first-service
이후 prompt completion 차이가 scheduler 정책보다 cache reuse에서 왔을 수 있다.

Replay는 arrival timestamp와 request shape뿐 아니라 cache disposition, graph hit/miss, adapter,
backend와 model revision을 고정한다. Cache를 완전히 끄는 실험 하나와 동일 hit map을 재생하는
실험 하나를 나눈다. 첫 실험은 scheduler 효과를 격리하고 둘째는 production 효과를 검증한다.
둘 중 하나만으로 승인하지 않는다.

Clock confounder도 있다. 첫 token timestamp가 server enqueue에서 찍히다가 release 뒤 socket write
완료로 바뀌면 TTFT가 늘어도 scheduler는 같을 수 있다. Metric source에서 시작·끝 event를 pin하고
old/new revision이 같은 clock을 쓰는지 확인한다. Histogram reset, scrape gap과 rolling deployment의
두 population 혼합도 policy transition처럼 보일 수 있다.

작은 counterfactual 표로 귀속을 닫는다.

| replay | chunk | cache map | graph map | arrival trace | TTFT p99 | ITL p99 |
|---|---:|---|---|---|---:|---:|
| B0 | 512 | fixed miss | fixed hit | burst-17 | 721 ms | 12.1 ms |
| C1 | 256 | fixed miss | fixed hit | burst-17 | 394 ms | 13.6 ms |
| C2 | 256 | production hit | production hit | burst-17 | 331 ms | 13.4 ms |
| X1 | 512 | production hit | production hit | burst-17 | 650 ms | 12.0 ms |

B0와 C1의 327 ms 차이가 chunk mutation의 격리 효과다. C1과 C2의 63 ms는 cache mix 효과다.
B0와 C2만 비교해 390 ms가 전부 chunk 효과라고 보고하면 과장한다. X1은 production cache가
512를 일부 돕지만 SLO 500 ms를 만족시키지 못한다는 반증이다.

### rollback도 하나의 terminal state다

Rollback을 “원래 숫자를 다시 쓴다”로 끝내면 running request와 metric population이 섞인다.
Policy generation 8의 128 decision을 철회할 때 controller는 generation 9의 256을 publish하고,
scheduler가 이를 처음 소비한 step을 기록한다. Generation 8에서 시작한 partial prefill이 128
round를 계속 쓰는지 다음 round부터 256을 쓰는지 계약을 고정한다. Old generation request가 모두
끝난 시점까지 rollback은 drain 중이다.

종료 조건은 네 가지다. Requested와 effective chunk가 256으로 합의되고, generation 8 request
수가 0이며, ITL/prompt completion guard가 연속 세 창 정상이고, oscillation cooldown이 끝난다.
Config API 200이나 controller log 한 줄은 첫 조건조차 보장하지 못할 수 있다. Scheduler trace의
effective generation과 cohort counter가 필요하다.

Rollback 중 admission도 명시한다. Overload가 계속되는데 128을 철회하면 queue가 더 빨리 자랄 수
있다. 이때 256 복귀와 함께 burst tenant reject watermark를 켜야 할 수 있다. Rollback 때문에
거절된 request를 chunk 정책 실패와 섞지 않고 `admission_guard` terminal reason으로 회계한다.

### SLO에는 값을 지킬 owner와 양보할 owner가 있다

Product owner는 TTFT 500 ms와 ITL 15 ms 같은 외부 약속, workload cohort와 예외를 정의한다.
Gateway owner는 arrival shaping과 overload reject, scheduler owner는 first-service·frontier·decode
admission, worker owner는 step duration과 graph/backend, observability owner는 clock과 population을
책임진다. Scheduler가 network TTFT까지 단독으로 보장할 수 없고 gateway가 device ITL을 직접
고칠 수도 없다.

SLO 위반 record에는 `first_divergent_owner`와 `handoff_owner`를 나눈다. Waiting age가 먼저
늘었지만 원인이 gateway burst admission이면 scheduler는 증상이 처음 보인 owner이고 수정 owner는
gateway일 수 있다. 반대로 offered load는 정상인데 atomic prefill이 decode를 미루면 scheduler가
둘 다 소유한다. 이 구분이 없으면 가장 눈에 띄는 team에게 설정 변경을 요구한다.

재사용 가능한 decision record는 다음 질문에 답한다.

```text
workload revision과 arrival trace hash는 무엇인가?
TTFT·ITL clock의 시작과 끝, cohort와 표본 수는 무엇인가?
requested/effective chunk와 policy generation은 무엇인가?
어느 scheduler consumer가 어떤 frontier·budget을 바꿨는가?
512/256/128 counterfactual의 first-service·completion·ITL은 얼마인가?
cache·graph·backend confounder를 어떻게 고정했는가?
승인 guard, rollback trigger와 terminal 증거는 무엇인가?
first divergent owner와 최종 수정 owner는 누구인가?
```

이 record가 있으면 새 GPU에서 step duration이 달라지거나 새 release에서 default chunk가 바뀌어도
같은 판단을 반복할 수 있다. 숫자를 복사하지 않고 arrival trace와 budget consumer에 다시
대입한다. 2장의 최종 산출물은 256이라는 추천값이 아니라 workload, SLO, state mutation과
rollback을 재현하는 이 결정 기록이다.

### 사후 분석에서 decision record를 실제로 다시 쓴다

한 달 뒤 CUDA와 serving runtime을 함께 올린 뒤 burst TTFT p99가 390 ms에서 540 ms로
되돌아왔다고 하자. 운영자는 이전 결론인 “256이 최적”을 그대로 적용하지 않는다. 저장한 arrival
trace hash를 재생하고 policy generation, effective chunk와 step 기록을 먼저 비교한다. Requested와
effective chunk는 모두 256이고 arrival distribution도 같았다. 하지만 step duration이 10.5 ms에서
13.1 ms로 늘었으며 graph miss가 burst 첫 여섯 step에 집중됐다.

Counterfactual에서 graph map을 old hit pattern으로 고정하자 TTFT p99는 402 ms로 회복됐다. Chunk를
128로 낮추면 first service는 빨라졌지만 graph bucket 종류와 capture miss가 더 늘어 completion
p99가 870 ms가 됐다. 최초 불일치는 scheduler의 chunk decision이 아니라 worker의 graph selection
generation이었다. 이전 record가 없었다면 같은 증상만 보고 chunk를 다시 바꾸어 원인을 가렸을
것이다.

수정은 burst 시작 전 자주 쓰는 batch bucket을 warmup하고 release 뒤 graph eligibility predicate를
고정하는 것이었다. Acceptance replay에서 TTFT p99 406 ms, ITL p99 13.7 ms, prompt completion
742 ms였고 effective chunk는 계속 256이었다. Rollback terminal과 같은 방식으로 old graph
generation request가 0이 될 때까지 관측했다. 이 사건은 decision record가 추천 설정 보관함이
아니라 “무엇이 같고 무엇이 달라졌는가”를 찾는 counterfactual 도구임을 보여 준다.

마지막으로 p99를 지켰다는 사실만으로 서비스가 좋아졌다고 결론 내리지 않는다. First-chunk
priority가 tail을 줄이는 동안 이미 계산한 prefill을 더 자주 중단하거나 admission reject를 늘릴
수 있다. TTFT·ITL guard를 통과한 뒤에는 성공적으로 전달된 token, 취소 뒤 계산된 token,
preemption으로 버린 work와 reject population을 같은 workload identity로 센다. 바로 다음 장에서
이들을 strict goodput의 분자와 분모로 옮긴다.

다음 장부터는 이 판단법을 더 세밀하게 확장한다. 계산됐지만 버려진 token을 세어 goodput을
정의하고, prefill과 decode의 연산·memory 모양을 나누며, 마지막에는 실제 scheduler 함수의
budget 계산과 상태 collection까지 내려간다. 지금은 두 시계를 정확히 구분하고 하나를 줄인
비용이 다른 시계에 나타날 수 있다는 사실을 잊지 않으면 된다.

이 구분이 서면 profiler를 켜야 할 때와 켜지 않아야 할 때도 자연스럽게 갈린다. GPU 구간이
정상인데 queue나 socket에서 시간이 벌어졌다면 더 깊은 kernel 분석은 정밀한 오답일 뿐이다.
먼저 틀어진 경계를 고르고, 그 경계를 소유한 상태에서 아래로 내려간다.
그 순서가 가장 빠르고 재현 가능한 디버깅 경로다.
