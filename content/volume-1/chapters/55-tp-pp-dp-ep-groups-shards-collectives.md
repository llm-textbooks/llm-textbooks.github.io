# 55장. 한 tensor의 소유권으로 읽는 TP·PP·DP·EP

분산 추론 장애를 설명할 때 “all-reduce가 꼬였다”, “TP가 안 맞는다”는 말을 자주 듣는다. 그러나 collective 이름은 원인이 아니다. Collective 직전에 각 rank가 무엇을 소유했고 그 값이 완전한 feature shard인지, 더해야 하는 partial sum인지, 다른 expert에게 보내야 할 token인지 모르면 all-reduce와 all-gather 가운데 무엇이 필요한지도 판단할 수 없다.

이 장은 병렬화 용어를 정의하는 사전이 아니다. `X[2,4]`와 `W[6,4]`라는 작은 행렬 하나를 rank 두 개에 놓고, 한 output 좌표가 어디에서 계산되고 누구에게 전달되는지 손으로 따라간다. 같은 request를 pipeline stage와 data-parallel replica, expert group에 넣어 owner가 바뀌는 순간을 기록한다.

핵심 질문은 매 경계에서 같다. Global logical tensor는 무엇인가. 현재 rank가 소유한 coordinate range는 어디인가. Local op의 결과는 완전한 값인가 partial sum인가. 다음 consumer는 replicated tensor와 shard 중 무엇을 요구하는가. 그 요구를 만족시키는 collective 뒤 새 owner는 누구인가.

NCCL이 ring을 쓰는지 tree를 쓰는지, NVLink와 RDMA가 byte를 어떻게 운반하는지는 다음 장의 질문이다. 여기서는 all-reduce, all-gather, reduce-scatter, all-to-all, send/recv가 tensor 의미를 어떻게 바꾸는지만 본다. 의미가 먼저 닫혀야 transport profile도 정확히 읽을 수 있다.

## 55.1 여덟 rank에서 partial tensor를 complete로 오인한 사건

대표 fixture는 rank 0–3이 TP group, rank 4–7이 두 번째 TP group을 이루고, 두 group이 DP replica이며 각 TP group 안에서 PP·EP owner가 다시 갈리는 배치다. rank 5가 가진 `[T,H/4]` tensor를 complete activation으로 publish하면서 값은 유한하지만 replica별 logits가 갈렸다. 이 장은 tensor마다 `global coordinate, local owner, partial/complete 상태, 다음 collective, 다음 owner`를 붙여 TP→PP→DP→EP 상태 변화를 복원한다.

`X=[[1,2,3,4],[2,1,0,1]]`이고 `W`의 여섯 row를 다음처럼 두자.

```text
W0=[1,0,0,0]  W1=[0,1,0,0]  W2=[0,0,1,0]
W3=[0,0,0,1]  W4=[1,1,0,0]  W5=[0,0,1,1]
```

`Y=XWᵀ`는 `[[1,2,3,4,3,7],[2,1,0,1,3,1]]`이다. 이 작은 답을 reference로 두면 feature shard와 partial sum을 눈으로 구별할 수 있다.

### 완전한 shard와 partial sum

Column parallel에서 rank 0은 W row 0–2, rank 1은 row 3–5를 소유한다. Rank 0 output은 `[[1,2,3],[2,1,0]]`, rank 1은 `[[4,3,7],[1,3,1]]`이다. 각 값은 완전하다. 다만 N feature axis가 두 rank에 나뉘어 있다.

Row parallel에서는 rank 0이 K 0–1, rank 1이 K 2–3을 소유한다. Rank 0 partial은 `[[1,2,0,0,3,0],[2,1,0,0,3,0]]`, rank 1 partial은 `[[0,0,3,4,0,7],[0,0,0,1,0,1]]`이다. 두 tensor shape는 모두 `[2,6]`이지만 값은 불완전하다. 같은 좌표를 더해야 reference Y가 된다.

Column output에 all-reduce를 적용하면 서로 다른 feature를 같은 coordinate라고 착각해 더한다. Row output에 all-gather를 적용하면 partial feature 두 벌을 이어 붙여 `[2,12]`가 된다. Collective 선택은 shape나 layer 이름이 아니라 local value의 의미에서 나온다.

### ownership ledger

각 tensor에 logical name, global shape, local shape, shard axis, global range, partial/complete, owner group, next consumer layout을 기록한다. `local_output`처럼 의미 없는 이름을 피한다. `Y_feature_shard[N=0:3]`와 `Y_partial_from_K[0:2]`는 다음 연산을 알려 준다.

Collective 전후에는 owner가 바뀐다. All-gather 전 column output은 rank별 disjoint feature owner다. 이후 모든 rank가 replicated Y owner가 된다. All-reduce 전 row output은 같은 logical coordinates의 partial contributor다. 이후 모든 rank가 complete replicated Y를 소유한다.

## 55.2 여덟 rank의 owner 사슬로 최종 판정을 내린다

이 장의 시작에서 column-parallel shard와 row-parallel partial은 작은 숫자로 구별됐다. 둘은 비슷한 local tensor처럼 보이지만 하나는 서로 다른 feature의 완전한 값이고, 다른 하나는 같은 output 좌표에 더해야 할 기여다. 이 차이가 all-gather와 all-reduce를 결정했다.

Parallelism을 이해한다는 것은 TP, PP, DP, EP 정의를 외우는 일이 아니다. Global tensor와 request state를 누가 소유하고, local operation이 그 소유권을 어떤 상태로 바꾸며, 다음 owner에게 무엇을 넘기는지 말할 수 있다는 뜻이다. Group은 그 transition에 참여하는 ranks의 범위다.

TP에서는 parameter feature를 shard하고 local matmul이 feature shard 또는 partial sum을 만든다. PP에서는 layer owner가 바뀌며 activation과 microbatch generation을 send/recv한다. Serving DP에서는 request와 KV state가 replica에 귀속된다. EP에서는 router가 token을 expert owner에게 보내고 inverse permutation이 original token owner를 복원한다.

Collective 이름은 이 semantic 뒤에 온다. All-gather는 disjoint shards를 ordered replicated tensor로 만든다. All-reduce는 same-coordinate partials를 complete replicated value로 만든다. Reduce-scatter는 partial을 합치며 new shard ownership을 만든다. All-to-all은 peer-directed entries를 destination owner에 재배열한다. Send/recv는 stage 사이에 단일 ownership을 인계한다.

장애 조사도 같은 순서를 따른다. 먼저 effective rank grid와 group member/order를 확인한다. 그 다음 parameter loader가 만든 local range와 local operation의 reference 값을 본다. Collective input annotation과 operation을 비교하고 output range/order를 확인한다. 마지막으로 next consumer가 그 placement를 예상했는지 본다.

Hang에서는 모든 rank의 group generation과 last collective sequence를 모은다. Rank가 다른 call, group 또는 branch에 있으면 NCCL transport로 내려갈 이유가 없다. 모든 rank가 같은 semantic call에 들어왔는데 completion만 늦다면 다음 장에서 topology, protocol과 stream을 본다.

Wrong answer에서는 basis fixture가 강하다. Feature shard marker, K partial sum, vocab range, PP microbatch constant와 expert output marker를 사용하면 first divergence가 눈에 보인다. Shape와 finite check만으로는 gather order, bias duplication과 inverse permutation 오류를 찾을 수 없다.

좋은 판정은 owner transition을 포함한다. “Rank별 Q feature shards는 정확했지만 wrapper가 complete shards를 same-coordinate partial로 표시해 all-reduce했다. First divergence는 collective input semantic이고 N-axis gather로 수정했다.” 또는 “Expert compute와 A2A payload는 맞았지만 previous batch inverse generation을 써 token 1/2 owner가 바뀌었다.”처럼 쓴다.

Optimization 역시 owner contract를 보존해야 한다. Gather를 없애려면 next layer가 shard를 직접 읽어야 한다. All-reduce를 reduce-scatter로 바꾸려면 next placement가 그 shard를 받아야 한다. Async overlap은 complete chunk range와 buffer lifetime을 증명해야 한다. 빠르다는 이유로 partial state를 complete처럼 읽을 수는 없다.

이제 다음 장으로 내려갈 준비가 됐다. 우리는 collective가 왜 필요한지, input이 무엇이며 output owner가 누구인지 알고 있다. 56장에서는 이 이미 올바른 collective가 NCCL algorithm, GPU topology, channel, protocol, stream과 network를 통해 어떻게 byte로 이동하는지 살핀다. Semantic ownership이 닫혀 있으므로 transport 병목과 correctness 문제를 혼동하지 않을 수 있다.

최종 승인에는 정상 forward만 넣지 않는다. Rank 하나가 empty expert batch를 받고, microbatch가 stage boundary에서 abort되며, DP replica가 unavailable하고, TP collective가 async 상태인 순간을 각각 시험한다. 모든 rank가 같은 terminal generation으로 수렴하고 temporary buffer, KV와 request reference가 남지 않아야 한다.

Group configuration을 바꾸는 배포에서는 old/new ownership manifest를 diff한다. TP size 변경은 parameter range와 KV head owner를, PP size 변경은 layer와 activation boundary를, DP size 변경은 routing/KV locality를, EP size 변경은 expert placement와 token split을 바꾼다. Option 숫자만 바뀐 것이 아니라 model state의 물리적 주인이 다시 배치된 것이다.

재배치 뒤 checkpoint load, cache restore와 graph reuse가 새 ranges를 따르는지 확인한다. Old TP shard를 new local rank에 그대로 붙이거나 old expert permutation을 재사용하면 group은 정상적으로 통신해도 값이 틀린다. Manifest generation을 cache/graph/loader key에 반영하고 incompatible state를 invalidate한다.

독자는 낯선 code에서 `all_reduce` 호출을 발견했을 때 이제 곧바로 성능을 평가하지 않는다. 호출 전 tensor가 어떤 global equation의 partial인지, group members가 그 equation의 contributors인지, reduction 뒤 누가 complete value를 읽는지 묻는다. 이 세 질문에 답이 없으면 collective가 맞는지조차 아직 모르는 것이다.

반대로 세 답이 닫히면 구현 이름이 달라도 비교할 수 있다. Custom fused collective, DTensor redistribution, process-group wrapper와 graph op는 모두 같은 ownership transition 위에 놓인다. 그 공통 좌표가 이 장이 남기는 가장 실용적인 도구다.

Deadlock 회귀의 종료 조건도 분명히 한다. 모든 rank가 expected group generation과 collective sequence를 보고, empty-token rank까지 같은 protocol에 참여하며, timeout 없이 completion을 관찰해야 한다. 종료 뒤 in-flight handle, PP recv와 EP dispatch buffer가 다음 fixture에 남지 않아야 한다. 단순히 test process를 kill해 hang을 끝낸 것은 복구가 아니다.

Rank grid를 바꾸어 같은 fixture를 반복한다. TP×EP coordinate ordering, PP stage 수와 DP replica 수가 달라져도 global reference와 owner transition이 유지되어야 한다. Rank 번호가 바뀌어 output이 달라진다면 group-local order, shard range 또는 permutation이 hidden assumption을 가진 것이다.

이 마지막 fixture까지 통과하면 collective의 의미, 참여 범위, output ownership과 lifecycle이 함께 닫힌다. 다음 장의 NCCL 분석은 이 증명된 semantic 위에서 byte가 왜 늦거나 멈추는지를 묻게 된다.

그때도 기준점은 이 장의 작은 행렬이다. Transport가 달라져도 전달해야 할 좌표와 새 owner는 달라지지 않는다. 값의 의미를 고정한 뒤에야 통신 최적화가 모델 동작을 보존했는지 확실하게 판정할 수 있다.

여기서 한 단계 더 나아가, global tensor에서 최종 sampling owner까지 실제 여덟 rank를 한 번도 생략하지 않고
걸어 보자. 이번 fixture의 world size는8, DP=2, PP=2, TP=2이고 EP는 TP와 별도 곱으로 추가하지 않고 각 PP
stage 안의 두 model-parallel rank를 expert owner로 재사용한다고 정한다. 이 규칙을 먼저 적는 이유는 `2×2×2×2=16`을
기계적으로 기대하는 실수를 막기 위해서다. Engine마다 EP가 DP 또는 TP 축을 재해석하는 방식이 다르므로 effective
mesh가 source of truth다.

global rank 식은 `g=((dp×PP)+pp)×TP+tp`로 둔다. 좌표와 rank는 `(0,0,0)=0`, `(0,0,1)=1`,
`(0,1,0)=2`, `(0,1,1)=3`, `(1,0,0)=4`, `(1,0,1)=5`, `(1,1,0)=6`, `(1,1,1)=7`이다.
TP groups는 `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`; PP groups는 `[0,2]`, `[1,3]`, `[4,6]`,
`[5,7]`; replica request owners는 ranks0–3과4–7이다. 같은 local TP rank0인 global0,2,4,6은 같은
parameter slice 번호를 가질 수 있지만 서로 다른 stage 또는 replica owner다.

vLLM pinned source는 rank tensor를 `ExternalDP×DP×PP×PCP×TP` 순서로 reshape하고, TP는 마지막 축을 그대로
묶고 PP와 DP는 해당 축을 transpose한 뒤 마지막 축으로 펼친다.
[vLLM model-parallel group construction](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/distributed/parallel_state.py#L1746-L1915)
따라서 이 fixture는 이름에서 추측한 것이 아니라 실제 layout rule을 축소한 것이다. Source가 PCP/DCP 또는 elastic EP를
포함하면 우리가 적은 3차원 좌표를 그대로 강요하지 않고 그 축을 manifest에 추가한다.

stage0은 embedding과 layers0–1, stage1은 layers2–3, final norm과 LM head를 소유한다고 하자. Global hidden
`H=4`, intermediate `I=8`, vocab `V=10`이다. TP0은 hidden 또는 intermediate의 앞 절반, TP1은 뒤 절반을
소유한다. 단, 어느 축을 나누는지는 layer contract마다 바뀐다. `gate_proj [8,4]` column-parallel에서는 rank0이
output rows0:4, rank1이4:8을 가지며 local result는 complete feature shard다. `down_proj [4,8]` row-parallel에서는
rank0이 input columns0:4, rank1이4:8을 가지고 local result `[tokens,4]`는 partial이다.

DP replica0에 두 요청 A와 B가 들어오고 scheduled hidden을 `A=[1,2,3,4]`, `B=[2,0,1,3]`으로 둔다. Stage0
TP ranks0/1은 embedding 뒤 complete hidden을 공유하거나 이전 placement에 맞게 가진다. Gate/up local shards가 각각
네 feature를 만들고 activation 뒤 down projection partial `D0`, `D1`을 만든다. TP group `[0,1]` sum 뒤에만
`D=D0+D1`이 complete다. Rank0 partial과 rank1 partial이 우연히 같은 `[2,4]` shape라는 이유로 하나를 residual에
더하면 전체 hidden의 절반 기여가 사라진다.

complete stage0 output은 PP peers0→2와1→3으로 전달된다. 여기에는 tensor bytes만이 아니라 replica0, pipeline
generation12, microbatch7, forward step0, request rows `[A,B]`가 붙는다. Rank2가 rank0 payload를, rank3이 rank1
payload를 받는 것은 TP-local lane을 유지하는 pipeline contract다. Rank2가 rank1 payload를 받아도 local shape는
`[2,4]`로 같을 수 있다. Group membership과 message generation을 빼면 값만 보고 잘못된 lane 교환을 찾기 어렵다.

stage1 MoE router의 global experts를 E0–E3으로 둔다. EP owner는 tp0이 even experts E0/E2, tp1이 odd experts
E1/E3을 가진다고 명시한다. Local physical table은 rank2에서 global→local `E0→0,E2→1`, rank3에서
`E1→0,E3→1`이다. Round-robin placement를 contiguous `E0/E1→rank2`, `E2/E3→rank3`로 잘못 해석하면 tensor
shape와 expert count는 똑같지만 다른 weight가 실행된다. Expert map generation을 router와 loader 양쪽에서 비교해야 한다.

각 request에서 두 tokens씩, 총 네 tokens `t0=A0,t1=A1,t2=B0,t3=B1`을 두고 top-k=2 routing을 만든다.
Assignments와 weights는 `t0:(E0,.75),(E3,.25)`, `t1:(E1,.60),(E2,.40)`,
`t2:(E2,.80),(E0,.20)`, `t3:(E3,.55),(E1,.45)`다. Entry는8개다. 원본 entry order를
`(token,choice)`로 쓰면 `[t0E0,t0E3,t1E1,t1E2,t2E2,t2E0,t3E3,t3E1]`이다.

EP owner별 stable dispatch를 하면 rank2 destination에는 `[t0E0,t1E2,t2E2,t2E0]`, rank3에는
`[t0E3,t1E1,t3E3,t3E1]`이 간다. Destination counts는 `[4,4]`이고 local expert별 재정렬까지 하면 rank2는
`E0:[t0,t2], E2:[t1,t2]`, rank3은 `E1:[t1,t3], E3:[t0,t3]`이다. Token t2가 같은 rank의 E2와 E0에
두 번 존재한다는 점을 놓치면 dedup 최적화가 top-k contribution 하나를 지운다.

수치를 더 눈에 보이게 expert가 input scalar marker에 각각10,20,30,40을 더한다고 하자. Token base markers는
`[1,2,3,4]`다. Expert outputs는 t0 E0=11, E3=41; t1 E1=22, E2=32; t2 E2=33, E0=13;
t3 E3=44, E1=24다. Gating combine reference는 t0=`.75×11+.25×41=18.5`, t1=`.60×22+.40×32=26`,
t2=`.80×33+.20×13=29`, t3=`.55×44+.45×24=35`다. 이 네 값이 inverse permutation 뒤 original
token rows에 와야 한다.

Dispatch-order output vector를 destination/expert order대로 쓰면 `[11,13,32,33,22,24,41,44]`처럼 구현의
local packing 규칙에 따라 정해진다. 각 entry에는 original index `[0,5,3,4,2,7,1,6]`이 대응한다. Combine은
단순 inverse scatter가 아니라 같은 token의 두 contributions을 gating weight로 reduce한다. Original index와 token id,
choice id를 분리하지 않으면 t2 E0/E2처럼 같은 destination rank 내부 재정렬에서 choice weight가 바뀔 수 있다.

첫 번째 wrong-result fixture는 expert map generation만 한 rank에서 이전 값으로 남긴다. Generation20은 even/odd,
generation19는 contiguous placement라고 하자. Rank2 router는 E2를 local1로 보내지만 loader inventory generation19에서는
local1이 E1 weight일 수 있다. Input/output shape, local expert count와 all-to-all byte count는 모두 맞는다. 위 marker에서
t1 E2 expected32가 E1 marker22로 바뀌어 combined t1은22가 된다. First divergence는 communication이 아니라
global expert id→physical local row 해석이다.

두 번째 wrong-result fixture는 inverse mapping generation만 이전 microbatch 것을 재사용한다. Current original-index
array가 `[0,5,3,4,2,7,1,6]`인데 stale array가 `[0,3,5,4,2,7,1,6]`이면 t1 E2와 t2 E0 contribution이
교환된다. 두 entry가 같은 scalar shape이고 destination rank2 안에 있으므로 NCCL은 완벽히 성공한다. Expert kernel도
각 local row 기준으로 맞다. Original token/choice scatter에서 처음 갈라진다.

세 번째 fixture는 split counts가 다르다. Rank2가 send counts `[4,4]`를 믿고 rank3가 recv counts `[3,5]`를
믿으면 backend contract에 따라 count mismatch로 hang 또는 truncation이 된다. Global entry count8만 비교하면 놓친다.
Source×destination send matrix의 각 cell과 receiver column sum을 대조한다. Empty destination도 count0으로 protocol에
참여해야 하며 conditional branch로 collective 자체를 건너뛰지 않는다.

vLLM MoE runner는 naive dispatch 조건에서 EP group의 `dispatch_router_logits`로 hidden과 router logits를 옮기고,
compute 뒤 `combine`을 호출하며 PCP fallback에서는 all-gather와 reduce-scatter를 짝짓는다.

[vLLM MoE dispatch/combine boundary](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L783-L909)
이 링크는 모든 backend가 같은 permutation 구현을 쓴다는 뜻이 아니다. 오히려 dispatch producer와 combine consumer 사이에
backend-specific metadata contract가 있다는 탐색 좌표다.

SGLang에서도 group coordinator와 MoE backend를 따로 읽는다. Group wrapper가 내놓는 local/global rank, communicator와
all-to-all API를 확인하고, backend가 만든 send/recv counts, top-k indices와 restore order를 잇는다.
[SGLang group coordinator](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/distributed/parallel_state.py#L1-L260)
Server option의 EP size가 같다는 사실만으로 vLLM과 동일한 expert placement나 token packing을 가정하지 않는다.

Transformers TP plan은 비교용으로 중요하다. Module에 colwise/rowwise strategy를 적용하면서 desired input/output
placement를 선언하고 DTensor redistribution이 필요한 transition을 만든다.

[Transformers tensor-parallel strategies](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/tensor_parallel.py#L1-L220)
Serving engine의 custom communicator와 구현 모양은 다르지만 global tensor placement→local op→redistribution→next
placement라는 질문은 동일하다. API 이름을 억지로 일치시키지 않고 semantic contract를 비교한다.

이제 final LM head와 sampling owner까지 간다. Vocab10, TP=2이므로 stage1 ranks2/3이 각각 token ids0:5와5:10
logit rows를 소유한다. 한 decode row의 local logits를 rank2 `[1,4,2,3,0]`, rank3 `[5,2,6,1,3]`으로 두자.
Rank3 local index2의 score6은 global token id7이다. Distributed top-k가 `(score,global_id)` pair를 merge하면7을
선택한다. Local index2만 broadcast하면 stage0 embedding이 token2로 오해한다.

Full gather 방식이라면 concat plan이 `[rank2 range0:5,rank3 range5:10]`임을 보장한다. Group local order가 `[3,2]`로
생성됐어도 global range order로 재배열해야 한다. Gather output `[5,2,6,1,3,1,4,2,3,0]`을 token ids0:10으로
해석하면 score6이 id2로 바뀐다. Shape `[10]`, softmax normalization과 top-1 score는 정상이라 quality metric만
서서히 악화될 수 있다.

Sampling owner는 stage1의 지정 coordinator rank 2 또는 rank 3 중 하나로 명시한다. 모든 TP ranks가 후보를 계산하더라도 RNG
state와 sampling 결과를 한 owner가 commit하고 `(request,decode_step,model_generation)`과 함께 peer 및 PP first stage로
전달한다. 두 rank가 독립 RNG를 advance한 뒤 한 결과만 선택하면 다음 step RNG state가 갈라진다. Greedy fixture만으로
sampling ownership을 검증했다고 말할 수 없는 이유다.

두 requests A/B에서 temperature sampling을 한다면 RNG counter도 request-local이어야 한다. Batch reorder가 A/B의 random
draw 소유권을 바꾸면 동일 seed 재현성이 깨진다. Sample result ledger는 logits generation, valid vocab range, sampler
owner global rank, request-local RNG generation, chosen global token id와 broadcast completion을 가진다. Streaming owner는
선택 token을 client에 한 번만 내보내고 stage0 embedding owners는 같은 token/step을 받아야 한다.

PP feedback에서 rank2 sampling owner가 token7을 rank0에만 보내고 rank1이 old token2를 유지하면 다음 embedding/TP
forward가 rank별 다른 input으로 시작한다. 첫 row-parallel all-reduce는 서로 다른 logical request coordinate의 partial을
더하게 된다. Collective shape와 group은 맞지만 값이 의미를 잃는다. TP input token digest를 collective 직전에 비교하면
NCCL hang이 아니라 upstream sampling broadcast divergence임을 찾을 수 있다.

NCCL source는 semantic group을 스스로 발명하지 않는다. `ncclGroupStart`와 `ncclGroupEnd`는 grouped submission 범위를
관리하고, enqueue check는 communicator readiness와 arguments를 검사한 뒤 task를 append한다.
[NCCL group entry](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/group.cc#L96-L120)

[NCCL enqueue check](https://github.com/NVIDIA/nccl/blob/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/enqueue.cc#L3124-L3169)
따라서 application rank들이 서로 다른 communicator나 collective sequence를 선택한 오류를 transport가 올바른 tensor
ownership으로 고쳐 주지 않는다. NCCL 로그의 opCount, count, datatype, communicator와 stream을 위 owner ledger에 붙인다.

Hang incident를 구체화하자. 배포가 mesh generation31에서32로 바뀌며 PP=1→2가 됐지만 rank3 model runner만 old
group accessor를 캡처했다. New TP group은 `[2,3]`인데 rank3 wrapper는 old `[0,1,2,3]` communicator에서 all-reduce
sequence88을 enqueue한다. Rank2는 new communicator sequence4에 들어간다. 두 buffers 모두 `[2,4]` fp16이고 count8이라
shape 검사로는 갈리지 않는다. All ranks의 `(group generation,member digest,sequence,call site)`를 모으면 첫 차이가 보인다.

이 사고에서 rank2가 timeout했다고 NCCL network를 먼저 바꾸면 안 된다. Rank2 log에는 new TP sequence4, rank3에는 old
world sequence88이 있다. Expected group members가 다르므로 transport path 이전에 application ownership이 갈렸다. Fix는
old request drain, graph/model wrapper/accessor destruction, new groups all-rank prepare, manifest comparison, atomic admission
publish다. Process 일부만 group을 재초기화하지 않는다.

Wrong-result incident는 더 조용하다. Mesh ranks와 communicator는 generation32로 모두 같지만 rank3 loader가 old expert
placement generation19를 restore했다. All-to-all entry counts와 completion은 정상이고 hang도 없다. Four-token reference에서
t1만26→22, t2는29와 다른 marker로 바뀐다. Router mapping digest는 generation20인데 parameter inventory digest가 rank3만19다.
첫 divergence를 local expert lookup 전에 잡는다.

두 incident는 “분산 설정 drift”라는 한 이름으로 합치지 않는다. 첫 사고는 group lifetime과 communicator participation
오류라 hang을 만들고, 둘째는 state generation/semantic owner 오류라 shape-correct wrong result를 만든다. Rollback terminal도
다르다. Group incident는 old/new in-flight collectives와 communicators를 닫아야 하고, expert incident는 parameter inventory,
router map, graph와 dispatch metadata를 한 generation으로 수렴시켜야 한다.

Group reconfiguration transaction을 prepare, validate, publish, drain, retire로 나눈다. Prepare에서 모든 ranks가 new coordinate,
member lists와 communicator를 만들되 new request는 보지 못한다. Validate에서 group member digest, local ranks, layer/expert
inventory, collective canary를 합의한다. Publish는 scheduler admission generation을 한 번에 바꾼다. Old requests는 old
groups/weights/graphs를 끝까지 쓰고 drain한다. Ref0와 device work completion 뒤 retire한다.

Prepare 중 rank6 실패 시 이미 만들어진 ranks의 new communicators와 provisional buffers를 destroy하고 publish generation은31로
남긴다. Retry operation33은 old late success32를 무시한다. Rank마다 단순 boolean initialized만 보면 late rank6이32를
active로 만들 수 있다. Operation id와 desired mesh generation을 group owner와 model runner 모두 비교한다.

Publish 뒤 일부 rank가 acknowledgment를 잃었다고 즉시 old group으로 되돌리는 것도 위험하다. New requests가 이미
generation32 collective에 들어갔다면 alias-like pointer만31로 바꾸면 같은 요청 안에서 group이 섞인다. 먼저 new admission을
막고 gen32 requests와 device collectives를 drain/abort protocol로 닫는다. 이어 graph, KV, PP mailboxes와 EP inverse buffers의
generation references가0인지 확인한 뒤 desired generation을 전환한다.

Collective retry는 특히 조심한다. Rank2 partial buffer가 all-reduce에 일부 사용된 뒤 async error가 났을 때 같은 buffer를
그대로 재사용해 contribution을 다시 더하면 결과가 달라질 수 있다. NCCL call이 error를 반환했다는 사실만으로 device-side
buffer가 pristine이라는 보장은 없다. Fresh generation buffer로 전체 layer step을 재실행하거나 request를 terminal failure로
끝내는 명시적 policy가 필요하다. Sampling token이 client에 이미 stream됐다면 투명 retry 범위는 더 좁다.

PP rollback은 mailbox ownership을 닫는다. Microbatch7 generation32 activation을 stage1이 받기 전에 abort되면 sender buffer,
send handle, receiver reservation과 metadata entry가 모두 terminal해야 한다. 다음 request가 microbatch id7을 재사용해도
generation33이므로 stale recv가 연결되지 않는다. Receiver가 payload를 받았지만 compute 전 abort라면 discard acknowledgment
뒤 buffer를 해제한다. Host queue delete만으로 send/recv lifecycle을 끝내지 않는다.

EP rollback은 eight routing entries의 cardinality를 보존한다. Dispatch 뒤 rank3 failure가 나면 rank2가 가진 E0/E2 결과만
원 token에 부분 combine해 응답하지 않는다. Top-k contribution set이 불완전하므로 request layer output이 정의되지 않는다.
All destination completion 또는 backend가 증명한 redundancy가 없으면 layer/request를 실패시킨다. Temporary sorted indices,
send/recv splits, expert outputs와 inverse map generation을 함께 회수한다.

Sampling rollback은 committed token 경계를 가진다. Token7이 sampler owner에서 정해졌지만 stage0 broadcast 전에 실패하면
client stream도 commit하지 않는 순서가 단순하다. 이미 client에 보냈다면 다음 decode state를 다른 replica로 넘기려면 token7,
RNG counter, KV/PP stage state와 model generation을 모두 transfer/commit해야 한다. Token 문자열만 재전송해 failover됐다고
판정하지 않는다.

수용량도 mesh owner로 계산한다. Global gate weight `[8,4]` fp16은64 bytes, TP2 local은32 bytes다. Down weight `[4,8]`도
64 bytes, row shard32 bytes다. Tiny fixture라 작지만 production에서 같은 식을 layer, quant scale, adapter와 expert 개수에
곱한다. PP는 stage별 layer bytes를 나누고 DP는 replica별 복제하며 EP는 expert weight를 나눈다. `global model bytes/GPUs`라는
평균은 embedding/head imbalance, shared experts와 temporary collective buffers를 숨긴다.

Activation peak는 ownership transition에서 생긴다. Row partial 두 개가 all-reduce input으로 존재하고 complete output이
in-place 또는 별 buffer로 생기는 순간, PP send buffer가 receiver completion까지 pinned되는 순간, MoE dispatch가 original
hidden과 packed duplicate entries를 동시에 보유하는 순간을 센다. Top-k2는 네 tokens를 여덟 entries로 늘린다. Padding과
alignment를 더하면 expert GEMM input은 원 hidden의 두 배 이상일 수 있다.

위 scalar fixture를 hidden4 fp16으로 바꾸면 original four tokens는32 bytes, top-k2 dispatched eight entries는64 bytes다.
Original indices int32 32 bytes, weights fp32 32 bytes, split metadata와 output64 bytes가 더해진다. Tiny compute보다 metadata가
더 클 수 있다. Production hidden4096에서는 entry payload가 지배하지만 extreme small decode batches에서는 sorting, counts
exchange와 launch 고정비가 중요하다.

관측 메트릭은 bounded labels와 trace를 나눈다. Metrics에는 `parallel_group_generation`, generation mismatch count,
collective timeout/error count, TP/PP/EP logical bytes, PP mailbox depth, expert token imbalance, sampling-owner failover count를
둔다. Exact member list, request id, expert permutation과 tensor range는 high-cardinality trace/debug bundle에 둔다. Global rank만
label로 넣고 coordinate를 잃지 않되 elastic generation을 무제한 label로 만들지도 않는다.

Rank-local last-operation ring에는 monotonic record id, group semantic name, member digest, generation, communicator identity,
collective kind, sequence, input logical tensor id, local range/partial flag, shape/dtype/count, stream, enqueue와 completion을 넣는다.
Watchdog가 timeout을 감지하면 모든 ranks에서 같은 snapshot epoch를 수집한다. 한 rank의 Python stack만으로 custom/fused
communicator 내부 sequence와 peer member를 추측하지 않는다.

Wrong-result trace에는 full activation 대신 basis marker와 coordinate digest가 유용하다. Loader local range digest, router
global→physical map digest, dispatch original-index digest, post-expert marker, inverse map digest, vocab offset과 sampled global id를
남긴다. 민감한 token hidden을 저장하지 않아도 어느 ownership transition에서 처음 달라졌는지 판별할 수 있다. Debug capture가
synchronize를 넣어 race를 가리지 않는지도 확인한다.

검증 matrix의 첫 축은 mesh다. TP2/PP2/DP2 fixture, TP4/PP1, uneven vocab, empty expert destination, top-k1과2, contiguous와
round-robin placement를 시험한다. 둘째 축은 execution path다. Eager/graph, prefill/decode, naive/custom all-to-all,
full-gather/distributed sampling을 교차한다. 셋째 축은 lifecycle이다. Normal, abort, rank prepare failure, timeout, reload와
late completion을 넣는다.

각 fixture는 실행 전에 expected owner table을 생성한다. Global parameter range, rank coordinate, layer/expert owner, input/output
placement, collective group/order, PP peer, dispatch/inverse arrays, vocab range와 sampling owner를 적는다. Runtime observation을
이 표와 diff한다. “모든 rank 결과가 같다”만 검사하면 모두가 같은 잘못된 gather order를 공유하는 오류를 놓친다. Global
dense/reference 결과와 비교한다.

Negative fixture도 필수다. TP gather order를 뒤집고, rank 하나의 group generation을 늦추고, expert map만 stale하게 만들고,
inverse permutation 두 entries를 교환하고, padded vocab mask를 끄고, sampler local→global offset을 제거한다. Test가 각각
loader/group/dispatch/combine/sampling의 다른 first divergence를 보고해야 한다. 정상 case 통과만으로 instrumentation이 실제
오류를 볼 수 있는지 알 수 없다.

Reader가 source를 파고들 때는 config parser부터 시작해도 충분하지 않다. Effective sizes가 runtime config에 들어간 뒤 rank
mesh를 reshape/transpose하는 construction, wrapper가 member lists로 process group을 만드는 곳, model loader가 local ranges를
계산하는 곳, forward가 collective를 선택하는 곳을 이어야 한다. MoE에서는 router indices→expert map→dispatch splits→kernel
local expert→combine inverse를 한 chain으로 읽는다. Sampling에서는 LM-head shard→candidate global offset→RNG owner→broadcast를
잇는다.

Producer와 consumer를 반드시 둘 다 본다. `all_reduce` wrapper의 함수 정의만 읽으면 input이 partial인지 모른다. Caller의
linear contract와 다음 residual consumer를 봐야 한다. Expert dispatch producer만 읽으면 restore mapping이 choice weight와
어떻게 합쳐지는지 모른다. Sampling candidate producer만 읽으면 어느 rank가 commit하고 stage0/client가 어떻게 받는지
모른다. 함수 목록 대신 state transition을 그리는 이유다.

소스 갱신에서는 line 이동보다 semantic diff를 본다. Rank layout axis order, EP nesting, group accessor caching, custom
collective fallback, expert placement default, dispatch metadata type, distributed sampling owner가 바뀌었는지 비교한다. TP/PP/EP
option 값이 같아도 이 중 하나가 바뀌면 old graph/cache/checkpoint interpretation과 호환되지 않을 수 있다. Manifest schema
version과 generation을 올리고 incompatible state를 invalidate한다.

운영 incident timeline은 관찰에서 시작한다. “일부 요청이 멈추고 일부는 조용히 다른 token을 냈다”를 hang과 wrong-result
두 cohort로 나눈다. Hypothesis는 transport congestion, group mismatch, expert placement drift, inverse map reuse, sampler offset이다.
모든 ranks의 mesh/inventory와 four-token fixture를 수집하면 hang cohort는 group31/32 mismatch, wrong cohort는 expert19/20
mismatch로 갈린다. 같은 배포 사건이 두 독립 failure mode를 만들었음을 증명한다.

Containment는 new admission을 막고 known-good generation31 replica로 traffic을 제한한다. 이미 generation32에 들어간 requests는
섞어 continuation하지 않고 상태에 따라 drain 또는 abort한다. Cache와 graph key에 mesh/model generation이 없다면 재사용을
막고 보수적으로 비운다. Network 환경 변수 변경이나 무작정 timeout 증가로 증상을 가리지 않는다.

Verification은 tiny mesh reference부터 production canary로 올라간다. Four-token MoE combined values `[18.5,26,29,35]`, vocab
top token7, stage feedback token7, rank-local parameter markers와 all-reduce dense reference를 확인한다. Empty-token rank와
top-k duplicate, padded vocab도 통과한다. 이어 all ranks member/inventory digests가 같고 last-operation ring이 같은 semantic
sequence를 보이며 timeout/mismatch counters가0인지 본다.

Rollback terminal은 구체적인 목록이다. New admission generation31로 수렴, generation32 active requests0, in-flight collective
handles0, provisional communicators0, PP mailbox/send/recv reservations0, EP dispatch/combine buffers0, stale graph references0,
sampling uncommitted tokens0이어야 한다. All worker model/expert inventory가31과 맞고 KV/cache entries도 compatible generation만
남아야 한다. Process restart로 memory를 지웠다면 복구는 됐지만 graceful rollback path가 검증된 것은 아니다.

Capacity terminal도 함께 본다. Stage별 resident bytes와 peak activation/collective workspace가 budget 아래이고, PP queue와
EP dispatched entries가 configured bounds로 돌아오며, DP routing imbalance가 SLO를 깨지 않아야 한다. Correctness를 위해
모든 collective overlap, graph와 distributed sampler를 disable한 상태는 안전한 임시 mode일 수 있으나 최종 성능 terminal은
아니다. 기능을 한 단계씩 canary로 되살리고 mismatch0 조건을 유지한다.

리뷰 회의에서 “shape가 맞는데 왜 틀릴 수 있나?”라는 질문에는 세 표로 답한다. 첫 표는 `[2,4]`가 feature shard인지 partial인지,
둘째는 local expert row0이 generation19의 E0인지 generation20의 E1인지, 셋째는 local vocab index2가 global2인지7인지 보여 준다.
Shape는 storage extent만 말한다. Group, topology coordinate, semantic range와 state generation이 붙어야 logical tensor identity가
된다.

“Topology가 달라지면 값도 달라지는가?”에는 transport topology와 logical mesh를 분리해 답한다. Ring/tree, NVLink/PCIe는
올바른 collective의 byte 경로와 성능을 바꿀 수 있지만 exact arithmetic order에 따른 작은 부동소수 오차 외에는 logical
owner를 바꾸면 안 된다. 반면 rank mesh member/order, expert placement와 shard range는 tensor 의미를 직접 바꾼다. 두 종류의
topology를 같은 단어로 섞으면 장애 층위를 잘못 고른다.

“NCCL이 성공했는데 왜 wrong answer인가?”에는 NCCL이 받은 communicator, count, dtype와 operation을 수행했을 뿐 application의
global coordinate를 알지 못한다고 답한다. E1 weight를 E2라고 부르거나 gathered rank order를 vocab order라고 해석한 오류는
유효한 byte transfer다. 그래서 NCCL success는 transport completion 증거이고 model semantic correctness 증거가 아니다.

“모든 rank가 같은 output이면 안전한가?”에도 아니다. Wrong gather order를 모든 rank가 동일하게 받거나 bias를 TP배 더한
all-reduce 결과를 모두 공유할 수 있다. Replica 간 equality는 divergence 탐지에는 유용하지만 dense global reference, valid
vocab mapping과 request semantics를 대신하지 않는다. Equality와 correctness를 별도 predicate로 둔다.

최종 코드 review 질문은 간결하지만 답은 수치여야 한다. Global W의 어느 좌표를 이 rank가 load했는가. Local matmul output은
complete shard인가 partial인가. Collective members는 정확히 그 contributors인가. Output rank order가 global coordinate order와
어떻게 연결되는가. PP payload의 request generation은 무엇인가. Expert entry의 original token/choice와 physical expert는
누구인가. Sampler가 local token id를 global로 언제 바꾸고 누가 commit하는가. Abort 뒤 각 owner는 언제 ref0가 되는가.

이 질문에 `[rank0 rows0:4]`, `partial [tokens,4]`, `TP group [2,3] generation32`, `dispatch original indices
[0,5,3,4,2,7,1,6]`, `sampler owner rank2/global token7`처럼 답할 수 있어야 한다. “Framework가 처리한다”, “all-reduce를
쓴다”, “EP backend가 알아서 한다”는 답은 source digging의 종료 조건이 아니다.

이 확장 fixture가 남기는 결론은 분명하다. Global tensor는 local buffer가 되는 순간 의미를 잃지 않는다. Rank mesh coordinate,
shard range, partial/complete 상태와 generation으로 그 의미가 압축된다. Collective는 이 상태를 다른 owner 상태로 바꾸며,
MoE permutation과 sampling은 token identity를 다시 request owner로 돌려놓는다. 어느 한 경계라도 generation 없는 pointer나
shape만으로 연결하면 hang 또는 조용한 wrong result가 된다.

반대로 artifact와 group manifest, local tensor ledger, collective record, token permutation과 sampling commit을 하나의 request
trace로 연결하면 원인을 빠르게 좁힐 수 있다. Local reference 이전이면 loader/model, collective participation 이전이면 group,
post-collective order면 placement, expert kernel 이후면 inverse combine, logits 이후면 sampler owner를 본다. 이 순서는 거대한
production tensor에서도 작은 mesh와 똑같다.

마지막 승인 문장은 이렇게 쓸 수 있다. “Mesh generation32의 ranks0–7은 `(DP2,PP2,TP2)` 좌표와 group member digest에
합의했고, global parameter ranges와 stage owners가 manifest와 일치했다. Four-token top-k2 dispatch 여덟 entries는 expected
split과 expert generation20을 사용해 `[18.5,26,29,35]`로 복원됐으며, vocab shards0:5/5:10에서 sampler owner가 global
token7을 commit했다. Abort/reload 뒤 old group, mailbox, permutation, graph와 collective references가0이었다.” 이 정도로
구체적이어야 분산 서빙이 단지 멈추지 않았다는 사실을 넘어 올바른 값을 계산했다고 말할 수 있다.

한 번 더 rank별 장부를 펼쳐 보자. Request A/B가 replica0에 있으므로 ranks4–7은 같은 model weights를 가진 별 replica지만
이 step의 request/KV owner는 아니다. Ranks0/1은 stage0, ranks2/3은 stage1이다. Rank0은 stage0 TP0 parameter shard,
rank1은 stage0 TP1 shard, rank2는 stage1 TP0과 even expert weights, rank3은 stage1 TP1과 odd expert weights를 가진다.
Global coordinate가 같아도 stage와 replica가 다르면 local buffer를 교환할 수 없다. Debug dump 이름에 `rank0`만 넣지 않고
`mesh32/dp0/pp0/tp0`처럼 owner coordinate를 붙인다.

Stage0 row-parallel partial에 basis 값을 둔다. A row에서 rank0 contribution `[1,10,100,1000]`, rank1 contribution
`[2,20,200,2000]`이면 complete `[3,30,300,3000]`이다. B row는 rank0 `[4,40,400,4000]`, rank1
`[5,50,500,5000]`, complete `[9,90,900,9000]`이다. All-reduce 직전 두 rank의 input checksum이 다른 것은 정상이다.
완료 뒤 checksum이 같아야 하며 dense reference와도 맞아야 한다. “Rank checksum mismatch” 경보를 collective 전후 구분 없이
쓰면 정상 partial을 장애로 오인한다.

같은 값을 PP boundary marker로 사용하면 lane swap이 즉시 보인다. Rank0→2 payload가 rank1→3 payload와 바뀌면 stage1의
두 TP lanes가 complete hidden 두 벌을 받는 문제가 아니라 서로 다른 semantic contributor를 받는다. Stage0에서 이미
all-reduce해 complete replicated hidden을 만들었다면 두 lanes payload는 같아야 하고 swap은 harmless할 수 있다. 반대로
reduce-scatter 또는 sequence-parallel shard를 유지했다면 swap은 치명적이다. PP payload contract에 replicated/sharded 상태를
명시해야 일반론을 피할 수 있다.

Mesh 구성 검증은 product check보다 강해야 한다. World8과 DP2×PP2×TP2가 맞아도 rank array를 `(PP,DP,TP)`로 reshape하면
TP groups는 같아 보일 수 있지만 PP와 DP groups가 바뀐다. Request A가 replica0이라고 생각했는데 rank2가 다른 DP replica에
속할 수 있다. 각 축에서 coordinate 하나만 바뀌고 나머지는 고정되는지 property test하고, 모든 global rank가 각 semantic
group에 정확히 한 번 속하는지 검사한다.

Ordered member digest는 set digest와 달라야 한다. `[2,3]`과 `[3,2]`는 같은 participants라도 local rank와 shard concat
order가 바뀐다. Digest input에 group semantic, generation, ordered global ranks와 backend를 넣는다. Communicator pointer는
process마다 값이 다르고 재시작에서 바뀌므로 cross-rank identity가 아니다. Control plane이 동일 digest에 합의하고 local
wrapper가 자신의 global→local mapping을 보고한다.

상태 generation은 하나로 뭉치지 않을 수도 있다. Mesh generation32, model weight revision7, expert placement20, graph capture9,
request step104가 함께 실행될 수 있다. 이 tuple을 매 tensor에 전부 복사할 필요는 없지만 compatibility relation을 manifest로
정의한다. Graph9가 mesh32/weight7/expert20에 대해 capture됐고 request104가 이를 reference한다는 식이다. 단일 `version=32`로
덮으면 독립 update와 rollback을 표현하기 어렵다.

Shape-correct 오류를 잡는 invariant를 경계마다 둔다. Loader는 local slice를 global basis와 비교한다. TP collective는
partial/feature annotation과 dense reference를 비교한다. PP는 microbatch generation과 request-row digest를 비교한다. EP는
entry cardinality, destination owner와 inverse bijection을 검사한다. Sampling은 valid global vocab interval과 commit owner를
검사한다. 모든 경계에서 full tensor 비교를 켜지 않아도 작은 deterministic canary가 이 invariant를 실행할 수 있다.

Inverse bijection은 top-k에서는 정확히 말해 entry-level bijection이다. Eight dispatched entries 각각은 하나의 original
`(token,choice)`로 돌아와야 하고, token-level에서는 두 entries가 weighted reduction된다. Entry ids 집합이 0:8을 정확히 한 번
포함하는지, token마다 choices0/1이 있는지, weights 합이 정책상 기대 범위인지 확인한다. Token id만 unique라고 assert하면
top-k2의 정상 duplicate를 제거한다.

Capacity overflow에서 token drop이 허용되는 MoE라면 dropped entry도 provenance를 가진다. Capacity factor 때문에 t3 E1이
drop됐다면 combine이 missing contribution을 어떤 정책으로 처리하는지 reference에 반영한다. Silent buffer overflow와 deliberate
capacity drop을 구분한다. Dropped count, expert capacity, renormalization 여부와 original entry id를 기록한다. Backend마다
정책이 다를 수 있으므로 top-k 수식만으로 결과를 단정하지 않는다.

Expert load balancing이 placement generation을 바꾸는 경우 이동 transaction을 model reload와 같은 엄격함으로 본다. New
physical expert copy가 모든 owner에 준비되고 router map이 publish되기 전에는 old mapping으로 처리한다. In-flight dispatch가
old mapping인데 compute가 new local rows를 보면 marker 사고가 난다. Dispatch metadata에 placement generation을 넣고 expert
kernel admission에서 local inventory와 일치시키며, old entries completion 뒤 old weights를 retire한다.

Graph capture는 communicator와 pointers를 숨긴다. New mesh가 같은 buffer address를 재사용해도 captured collective handle이나
expert map pointer가 old generation일 수 있다. Capture compatibility key에 mesh/group, parameter/expert placement와 maximum
dispatch layout을 포함한다. Reconfiguration 뒤 graph replay를 막고 eager canary로 owner table을 검증한 다음 new graph를
capture한다. Address equality는 semantic compatibility가 아니다.

DP failover에서도 sampling commit을 기준으로 한다. Replica0가 token7을 commit한 뒤 죽고 replica1이 request를 이어받으려면
prompt/decode tokens, all-stage KV, RNG state와 model/mesh generations이 token7 이후 상태로 일치해야 한다. Replica1이 같은
weights를 가졌다는 사실만으로 충분하지 않다. State transfer가 없다면 명시적 client retry가 맞고, 다른 replica에서 조용히
다음 step을 생성하면 ownership이 끊긴다.

모니터링에서 `NCCL operation timeout`과 `parallel semantic mismatch`를 별 alert stream으로 둔다. Group/member/sequence가
불일치하면 semantic mismatch가 먼저 울리고 transport tuning runbook으로 가지 않는다. 모든 semantic records가 맞고 enqueue는
됐지만 completion만 늦으면 56장의 topology, channel, protocol과 network counters를 연다. 이 triage 경계가 없으면 application
bug에 NIC와 NCCL 환경 변수를 반복 변경하게 된다.

성능 실험도 correctness canary를 같은 run에 둔다. Custom all-reduce, fused reduce-scatter, alternate all-to-all backend와
distributed sampler를 바꿀 때 throughput/TTFT 옆에 dense layer error, four-token MoE error, global token-id mismatch와 group
sequence mismatch를 함께 보고한다. Fast path가 wrong fixture에서만 빠르게 성공한다면 benchmark가 최적화를 승인하지 못하게
한다.

Rollback drill은 장애가 없을 때도 한다. Mesh32 prepare 중 rank6 실패, expert20 publish 직전 timeout, PP microbatch abort,
sampler commit 직후 replica loss를 순서대로 주입한다. 각 drill 후 generation별 active requests와 resource refs가0으로 닫히고
known-good fixture가 재실행되는지 본다. Restart에 기대지 않는 rollback과 restart-based recovery를 결과에서 분리한다.

독자가 직접 구현을 검토하는 한 시간짜리 실습은 다음처럼 끝난다. 첫15분에 effective mesh와 ordered groups를 손으로 쓴다.
다음15분에 한 linear의 global/local range와 collective equation을 계산한다. 다음15분에 네-token MoE entry table과 inverse를
만든다. 마지막15분에 vocab local/global id, sampler owner와 abort terminal을 잇는다. 어느 칸에서 source로 값을 채울 수
없는지가 다음 digging 목록이다.

이 실습에서 diagram을 그린다면 화살표 label에 API 이름보다 상태를 쓴다. `complete feature shard`, `same-coordinate partial`,
`generation20 expert entry`, `weighted original-token contribution`, `global token7 committed`가 좋은 label이다. `all-reduce`,
`A2A`, `broadcast`만 쓰면 왜 필요한지와 잘못됐을 때 무엇이 깨지는지 보이지 않는다. Transport operation은 상태 transition
아래에 보조 label로 둔다.

최종 terminal 증거는 재시작 후에도 재현 가능해야 한다. Pinned engine/source revision, effective configs, ordered group manifests,
model/expert generation inventories, tiny fixture inputs와 expected arrays, rank-local observations, collective records와 rollback
resource counts를 bundle로 보존한다. Production customer tensor를 복사하지 않고도 ownership bug를 다시 검증할 수 있어야
release upgrade에서 같은 class의 회귀를 잡는다.

마지막으로 이 fixture를 암기할 필요는 없다. 중요한 것은 어떤 새로운 parallel axis가 생겨도 global coordinate에 축을 추가하고,
group이 어느 축을 변화시키는지 쓰고, local result의 complete/partial 상태와 next consumer를 계산하는 습관이다. Context
parallel, sequence parallel, speculative worker가 추가돼도 같은 방법을 확장한다. 이름보다 owner와 transition을 먼저 적으면
서로 다른 engine의 구현을 같은 기준으로 비교할 수 있다.

## 55.3 group은 rank 목록이 아니라 통신 범위다

World size가 16이고 DP=2, PP=2, TP=2, EP=2라고 하자. Global rank를 숫자 하나로만 기록하면 어느 rank와 collective해야 하는지 알 수 없다. Rank coordinate를 `(dp,pp,tp,ep)`로 펼치고 group membership을 만든다.

### group construction

TP group은 dp, pp, ep coordinate를 고정하고 tp만 변화시킨다. PP group은 dp, tp, ep를 고정하고 pp가 변한다. DP group은 나머지를 고정하고 dp가 변한다. EP group은 dp, pp, tp를 고정하고 ep가 변한다. 실제 engine이 TP와 EP를 독립 축으로 두는지, 하나의 model-parallel world 안에서 재해석하는지는 source를 확인해야 한다.

Group ledger에는 global ranks의 ordered list, current global/local rank, group size, creation generation과 owner object를 둔다. `[0,1]`과 `[1,0]`은 같은 set처럼 보여도 collective rank order와 split interpretation에 영향을 줄 수 있다. 모든 process가 같은 order로 group을 만들고 같은 sequence로 collective를 호출해야 한다.

vLLM의 process group 구성은 [`parallel_state.py`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/distributed/parallel_state.py#L1-L260), SGLang의 group coordinator는 [`parallel_state.py`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/distributed/parallel_state.py#L1-L260)에서 current pin의 실제 member construction과 accessor를 따라가야 한다. 이름이 같다고 coordinate ordering까지 같다고 가정하지 않는다.

### 첫 deadlock 조사

Rank 0은 TP group `[0,1]`에서 all-reduce를 호출하고 rank 1은 잘못된 EP group `[1,3]`에서 all-to-all을 호출한다고 하자. 두 호출은 기다리기만 한다. NCCL transport를 보기 전에 rank별 group id, member list, collective sequence number, tensor shape/dtype와 call site를 비교한다.

Config product가 world size와 맞는지도 확인한다. DP×PP×TP×EP를 무조건 곱할 수 없는 engine에서는 nesting rule을 사용한다. 잘못된 공식으로 예상 group을 만들고 source를 버그라고 판단하지 않는다. Effective parallel config와 actual group object를 기준으로 한다.

## 55.4 column parallel은 feature ownership을 나눈다

Column-parallel linear는 weight의 output feature N축을 나눈다. PyTorch weight `[out,in]` 표기에서 rank 0이 W `[0:3,:]`, rank 1이 `[3:6,:]`를 가진다. X는 두 rank에 replicated되거나 앞 연산의 collective가 그 layout을 제공해야 한다.

### local matmul과 gather

각 rank의 local matmul 결과는 완전한 feature shard다. 다음 layer가 같은 N partition을 input shard로 받을 수 있다면 gather를 미룰 수 있다. 예를 들어 Q projection의 head shard가 attention local heads로 바로 들어간다면 global Q를 만들 필요가 없다.

다음 consumer가 전체 hidden feature를 요구하면 all-gather한다. Gather axis와 rank order가 W shard order와 같아야 한다. Rank 1 shard가 먼저 이어지면 shape는 맞고 feature semantics만 뒤집힌다. Constant basis fixture로 발견한다.

Transformers의 TP plan과 DTensor placement는 [`tensor_parallel.py`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/tensor_parallel.py#L1-L220)에서 colwise/rowwise strategy가 input/output layout을 어떻게 선언하는지 읽는다. DTensor라는 type 이름만 보고 physical collective를 추정하지 않고 placement transition consumer를 본다.

### vocabulary와 uneven shard

Vocabulary embedding/LM head도 output 또는 row axis를 나눌 수 있다. Vocab size가 TP로 나누어지지 않으면 padded rows와 valid range가 생긴다. Gather 결과의 padded logits를 sampling 전에 mask해야 한다. Rank별 local row count가 같아도 valid semantic range는 다를 수 있다.

Checkpoint loader는 global rows에서 rank range를 slice한다. Runtime output mapping과 loader shard order가 같아야 한다. Weight는 `[0:V/2]`, logits gather는 반대 rank order면 token ids가 바뀐다.

## 55.5 row parallel은 partial sum을 만든다

Row-parallel linear는 input feature K축을 나눈다. 각 rank는 X shard와 W의 같은 K slice를 곱해 full N shape partial을 만든다. Rank가 만든 값은 서로 다른 feature가 아니라 동일 Y 좌표에 대한 기여다.

### all-reduce와 reduce-scatter

다음 consumer가 replicated Y를 요구하면 sum all-reduce를 사용한다. 모든 rank가 같은 complete Y를 얻는다. 다음 consumer가 Y의 특정 axis shard를 받는다면 reduce-scatter로 sum과 shard를 한 번에 만들 수 있다.

Reduce-scatter의 input은 rank별 logical global partial이고 output은 reduced shard다. “Reduce한 뒤 아무 축이나 나눈다”가 아니다. Scatter order와 output placement가 next consumer의 shard contract와 맞아야 한다.

Bias ownership도 본다. 모든 rank partial에 bias를 더한 뒤 all-reduce하면 bias가 TP번 더해진다. Bias를 한 rank만 더하거나 reduction 뒤 더하는 current layer contract를 확인한다. Llama 계열 bias 없음이라는 사실을 다른 architecture에 일반화하지 않는다.

### sequence parallel과 혼동하지 않는다

Reduce-scatter output이 hidden feature가 아니라 token/sequence axis shard일 수 있다. Tensor parallel과 sequence parallel이 결합되면 input/output placement를 식으로 적는다. Collective 이름만 보고 shard axis를 추정하지 않는다.

vLLM의 parallel linear는 [`linear.py`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/linear.py#L1-L260)에서 parameter loader와 forward collective를 함께 읽는다. Weight shard 방식과 output reduction이 한 계약인지 확인한다.

## 55.6 transformer layer 하나에서 TP collective를 배치한다

Llama attention의 Q/K/V projection은 output heads를 TP rank에 나누고 local attention을 수행할 수 있다. Attention output projection은 local head features를 input shard로 받아 row-parallel partial을 만들고 sum한다. MLP gate/up은 column parallel, down은 row parallel 조합이 흔하다.

### QKV에서 output projection까지

Global Q heads 32, KV heads 8, head dim 128, TP=4라면 rank당 Q heads 8, KV heads 2다. Q/K/V projection 뒤 local tensor는 complete head shards다. Attention kernel은 local heads와 local/cache contract를 사용한다.

Local attention output `[tokens,8,128]`을 flatten하면 local hidden width 1024다. Output projection weight는 K/input width의 해당 slice를 소유하고 global hidden output `[tokens,4096]` partial을 만든다. All-reduce 후 residual과 더할 complete hidden이 된다. Parallel residual design이면 reduction placement가 달라질 수 있으므로 current model source를 본다.

### MLP

Gate/up projection은 intermediate size I를 rank별로 나눈다. 두 local results에 activation과 elementwise product를 적용한다. Down projection은 local I slice를 입력으로 받아 hidden partial을 만들고 reduce한다. Gate/up shard order가 다르면 shape는 맞아도 elementwise pairing이 틀린다.

Packed gate/up weight loader, TP range와 activation pairing을 함께 test한다. Constant stripe fixture로 rank별 local intermediate와 reduction 뒤 hidden을 비교한다.

### collective를 합치거나 미루는 최적화

Implementation은 all-reduce를 fused op, custom communicator 또는 deferred reduction으로 표현할 수 있다. Source에서 `all_reduce` 문자열이 없다고 collective semantics가 없다고 판단하지 않는다. 다음 consumer가 complete tensor를 요구하는 경계와 custom op의 output ownership을 확인한다.

## 55.7 pipeline parallel은 layer owner를 이동시킨다

PP는 weight feature axis를 나누기보다 layer range를 stage에 할당한다. Stage 0은 embedding과 layer 0–15, stage 1은 layer 16–31과 final norm/head를 소유할 수 있다. Stage boundary에서 activation과 metadata를 send/recv한다.

### boundary payload

Hidden activation `[scheduled_tokens,H]`, residual state가 분리돼 있다면 함께 전달한다. Position, request/token mapping, cache ownership은 stage가 독립 scheduler를 갖는지 shared schedule을 쓰는지에 따라 전달 방식이 다르다. Tensor만 맞고 microbatch identity가 틀리면 다른 request의 activation을 소비할 수 있다.

Payload ledger에는 pipeline generation, microbatch id, forward step, sender/receiver stage, tensor names/shapes/dtypes, request/token ranges와 completion을 둔다. Send가 끝났다는 host return과 receiver가 안전하게 reuse할 수 있다는 사실을 구분한다.

### microbatch timeline

Microbatch A가 stage 1에서 실행되는 동안 stage 0은 B를 처리할 수 있다. Sequence diagram에서 `(stage,microbatch,step)`을 coordinate로 둔다. A send와 B recv tag가 충돌하지 않도록 current runtime matching rule을 확인한다.

Abort가 발생하면 downstream stage에 cancellation 또는 terminal state가 전달돼야 한다. Stage 1이 A recv를 영원히 기다리거나 A buffer를 B로 해석하지 않게 한다. Pipeline bubble 성능보다 먼저 message ownership을 닫는다.

## 55.8 serving DP는 request replica ownership이다

Training에서 DP는 gradient를 all-reduce하는 의미로 익숙하다. Serving replica DP는 각 replica가 complete model-parallel group과 독립 request/KV state를 갖고 요청을 분배받는 구조일 수 있다. Gradient collective를 기본 전제로 설명하지 않는다.

### request routing

Ingress 또는 DP coordinator는 request를 replica에 배정한다. 그 순간 request state, KV cache, streaming output과 abort owner가 정해진다. 다음 decode step이 다른 replica로 이동하려면 KV state transfer나 shared canonical storage가 필요하다. 아무 메커니즘 없이 round-robin을 step마다 적용하면 state가 갈라진다.

Ledger는 request id, selected DP replica, replica-local scheduler id, KV owner, output owner와 terminal cleanup을 가진다. Replica failover가 있다면 state를 언제 어떤 generation으로 인계하는지 본다.

### DP와 TP coordinate

DP replica 0의 TP rank 1과 DP replica 1의 TP rank 1은 같은 TP local rank지만 다른 model replica와 request state를 소유한다. Metric과 trace가 local rank만 쓰면 두 owner를 합친다. `(dp_rank,tp_rank,global_rank)`를 함께 기록한다.

Weight는 동일 revision이어야 하지만 physical storage와 graph/cache는 replica별일 수 있다. Weight loading broadcast나 synchronization이 있으면 current source 경계를 확인한다. Serving DP의 token-level load balancing과 model parameter DP를 용어 하나로 합치지 않는다.

## 55.9 expert parallel은 token owner를 바꾼다

MoE layer에서 router는 각 token의 expert id와 weight를 만든다. Expert weights가 EP ranks에 나뉘어 있다면 token hidden을 expert owner로 보낸다. All-to-all은 같은 좌표의 합이나 feature gather가 아니라 peer-directed token permutation이다.

### 네 token과 두 expert

Token `t0,t1,t2,t3`의 expert assignment를 `[E0,E1,E1,E0]`라 하자. EP rank 0은 E0, rank 1은 E1을 소유한다. 원 input order가 rank/sequence별로 나뉘어 있어도 dispatch는 E0 tokens `[t0,t3]`을 rank 0, E1 `[t1,t2]`를 rank 1로 보낸다.

All-to-all input에는 peer별 split sizes와 token indices가 있다. Receiver는 expert-local packed batch를 계산한다. Reverse communication과 scatter는 결과를 `[t0,t1,t2,t3]` 원 order로 복원하고 top-k weight가 있으면 합친다.

### split size와 inverse permutation

Split sizes가 한 token 어긋나면 communication byte count mismatch로 hang하거나, shape가 우연히 맞으면 다른 expert result가 섞인다. Send counts, recv counts, sorted token ids, expert owner, inverse indices와 valid/padded token count를 기록한다.

Top-k=2이면 token 하나가 두 expert destination에 복제되고 결과를 weighted reduce한다. 이 reduction은 TP partial sum과 다른 semantic이다. Expert output contribution을 token coordinate로 모은다.

### EP와 TP 결합

Expert weight 하나가 TP로도 shard되면 token dispatch 뒤 expert 내부 column/row parallel collective가 추가된다. EP group과 TP group을 섞지 않는다. Coordinate ledger에서 token owner 이동과 feature partial reduction을 두 단계로 둔다.

## 55.10 collective 뒤 새 owner를 판정한다

여기서는 collective를 tensor 상태를 바꾸는 edge로만 정의한다. communicator enqueue·stream completion·오류·teardown의 상세 계약은 56장의 대표 all-reduce에서 계산한다.

All-reduce는 동일 logical coordinate partial을 sum하고 모든 member에게 complete value를 준다. All-gather는 disjoint shard를 ordered concat하여 모든 member에게 replicated tensor를 준다. Reduce-scatter는 partial global tensors를 reduce하고 disjoint output shard ownership을 준다.

All-to-all은 각 rank가 peer별 chunk를 보내 destination ownership으로 재배열한다. Send/recv는 stage owner가 단일 receiver에게 tensor를 인계한다. 같은 bytes를 이동해도 semantic transition이 다르다.

### collective assertion

호출 직전 global logical id, local shape/range, partial/complete, split/reduce axis와 group을 assert한다. 호출 직후 output shape/range, complete 여부와 next consumer expectation을 assert한다. Performance build에서 모든 값 검사를 켤 필요는 없지만 debug fixture가 이 계약을 실행해야 한다.

Collective wrapper가 async handle을 반환하면 output ownership은 completion 뒤에 유효하다. 다음 stream/op가 dependency 없이 읽지 않도록 한다. Stream/event 내부는 56장에서 다루되 semantic completion boundary는 여기서 기록한다.

## 55.11 일곱 장애를 first divergence로 가른다

**Column shard에 all-reduce.** Rank별 feature shard를 같은 coordinate로 더한다. First divergence는 collective input semantic이다. Gather 또는 downstream sharded consumer로 수정한다.

**Row partial에 all-gather.** Partial `[M,N]` 두 개를 concat해 `[M,2N]`으로 만든다. Reduction이 필요하다는 hand fixture로 반증한다.

**Vocab gather order.** Shape는 `[M,V]`로 맞지만 rank order와 token-id range가 뒤집힌다. Basis logits로 first wrong token row를 찾는다.

**PP microbatch mismatch.** Stage 1이 A recv buffer를 B metadata와 연결한다. `(pipeline_generation,microbatch,step)`에서 최초 divergence를 찾고 abort cleanup을 검증한다.

**DP replica drift.** Decode request가 다른 replica scheduler로 이동해 KV가 없다. Routing trace와 KV owner가 갈라진 순간을 찾는다.

**EP inverse permutation.** Expert compute는 맞지만 original token scatter가 틀린다. Sorted indices와 inverse mapping을 basis token으로 확인한다.

**TP×EP group deadlock.** Member list 또는 collective call order가 rank마다 다르다. Transport tuning 전에 group generation과 sequence를 대조한다.

### 55.11.1 column shard 사고를 값으로 재현한다

Rank 0의 complete shard `A=[[1,2,3],[2,1,0]]`, rank 1의 `B=[[4,3,7],[1,3,1]]`를 다시 보자. 잘못된 code가 두 tensor shape가 같다는 이유로 all-reduce를 호출하면 elementwise sum `[[5,5,10],[3,4,1]]`이 두 rank에 replicated된다. Expected Y `[2,6]`과 shape부터 다르거나 local buffer `[2,3]`로 다음 layer가 받아 shape가 자연스럽게 맞을 수도 있다.

Shape가 맞는 후자가 더 위험하다. Next layer도 local width 3을 기대하면 값 `[5,5,10]`을 legitimate feature shard처럼 계산한다. Error는 logits까지 전달되고 collective assert는 모두 통과한다. Basis W fixture가 semantic error를 드러낸다.

최초 불일치는 local matmul이 아니다. 두 rank A/B는 dense reference의 disjoint columns와 정확히 맞는다. Collective input annotation이 `complete feature shard`인데 reduction을 요청한 순간이다. Fix는 next consumer가 global Y를 원하면 N-axis all-gather, local shard를 원하면 collective 제거다.

Regression은 random W뿐 아니라 disjoint basis를 사용한다. Rank 0과 rank 1 output에 서로 겹치지 않는 marker를 둔다. All-reduce가 들어가면 marker가 즉시 섞인다. TP=1에서는 collective가 identity라 bug가 숨으므로 TP=2 이상을 필수로 한다.

### 55.11.2 row partial 사고를 값으로 재현한다

Rank 0 partial P0와 rank 1 partial P1은 모두 `[2,6]`이다. All-gather를 N축으로 하면 `[P0|P1]`의 `[2,12]`가 된다. Wrapper가 이후 reshape/truncate로 `[2,6]`을 만들면 P1 기여가 사라지거나 좌표가 섞일 수 있다.

Reference는 `P0+P1`이다. Local partial sample과 sum을 손으로 계산해 all-reduce 또는 equivalent reduction을 요구한다. Next consumer가 sequence shard를 원한다면 reduce-scatter의 scatter axis/order를 명시한다.

Bias fixture는 bias vector를 nonzero unique values로 둔다. 각 rank가 bias를 local partial에 더한 뒤 all-reduce하면 `TP×bias`가 된다. Reduction 뒤 한 번 더하는 contract 또는 designated owner contribution을 test한다. Bias 없는 Llama만 시험하면 다른 architecture에서 bug가 남는다.

첫 divergence ledger는 W/X K shard와 local matmul, bias application, collective kind, post-output을 가진다. Local matmul까지 맞고 post-reduction만 틀리면 CUDA GEMM을 profile하지 않는다.

### 55.11.3 vocab order 사고

V=8, TP=2로 token rows 0–3과 4–7을 나눈다. Rank 0 logits는 `[10,11,12,13]`, rank 1은 `[20,21,22,23]`처럼 marker를 둔다. Correct gather는 token-id order `[10,11,12,13,20,21,22,23]`다.

Group local rank order가 반대거나 loader shard mapping과 gather order가 다르면 `[20..23,10..13]`이 된다. Softmax와 top-k는 정상 작동하지만 selected global token id 의미가 바뀐다. Shape와 finite check는 잡지 못한다.

Uneven V=7이면 one padded row를 둔다. Padded logit가 zero라고 안전하지 않다. Valid logits가 모두 음수이면 zero가 top candidate가 될 수 있다. Invalid range를 negative infinity mask하거나 distributed sampler가 valid range를 제한하는지 본다.

Loader source row range, rank local vocab start/end, gather concat order, sampler global offset을 한 표에 둔다. First divergence가 load인지 output gather인지 sampling offset인지 구분한다.

### 55.11.4 PP microbatch mismatch 사고

Stage 0은 microbatch 41과 42를 연속 처리한다. 41의 activation buffer address를 send enqueue한 뒤 42 입력으로 너무 일찍 재사용하거나, stage 1 recv가 tag 없이 “다음 buffer”를 41로 해석하면 state가 섞인다.

Fixture는 각 microbatch activation을 unique constant로 둔다. 41은 모든 값 41, 42는 42다. Stage boundary 전 checksum, send record, recv checksum과 model layer input을 비교한다. Tensor shape/dtype는 같아도 값 marker가 owner mismatch를 보여 준다.

Abort 41을 send 전, in-flight, recv 완료 후 세 지점에서 발생시킨다. Stage 1이 wait에서 풀리는지, stale payload를 discard하는지, 42가 같은 buffer/tag generation을 안전하게 쓰는지 확인한다. Abort frontend response만 test하지 않는다.

Pipeline generation은 model reload/restart를 구분한다. Old stage 0의 늦은 message가 new stage 1 microbatch 41과 id가 같아도 accept되지 않아야 한다. `(model_generation,microbatch,step)`을 match key에 둔다.

### 55.11.5 DP replica drift 사고

Request R의 prefill은 replica 0에서 실행되어 KV block table과 scheduler sequence가 replica 0 memory에 있다. 다음 decode HTTP call이 stateless load balancer 때문에 replica 1로 가면 replica 1은 R을 unknown으로 reject하거나 잘못된 동일 id를 찾을 수 있다.

Fixture는 replica별 KV에 구별 가능한 marker를 둔다. Routing decision, replica-local request id, KV owner generation과 decode input을 기록한다. First divergence는 compute 전 routing boundary다.

Fix는 sticky routing, frontend-owned session map, explicit KV/state migration 또는 canonical shared state 가운데 architecture가 제공하는 mechanism이다. 단순히 두 replicas에 같은 request id를 만들면 output/abort ownership이 중복된다.

Replica failure failover는 정상 load balancing과 다르다. KV가 external store에 commit됐는지, output stream offset과 sampler RNG state가 인계됐는지 확인한다. State transfer가 없으면 request failure가 honest behavior일 수 있다.

### 55.11.6 EP inverse permutation 사고

Dispatch permutation `[t0,t3,t1,t2]`과 expert output marker를 사용한다. E0이 `+100`, E1이 `+200`을 더하면 sorted result는 `[t0+100,t3+100,t1+200,t2+200]`이다. Correct inverse는 original `[t0+100,t1+200,t2+200,t3+100]`이다.

Previous batch inverse `[0,2,3,1]`을 재사용하거나 padding removal offset이 틀리면 token 1/2 또는 3이 바뀐다. Expert GEMM은 정확하고 all-to-all byte count도 맞는다. First divergence는 reverse scatter다.

Top-2 fixture에서는 `(token,topk_slot)`를 identity에 포함한다. 두 expert contribution이 같은 token으로 합쳐지고 router weights와 pairing되는지 본다. Token index만 저장하면 duplicate entries가 덮어써질 수 있다.

Dynamic token count가 buffer capacity보다 작을 때 stale tail을 valid로 읽지 않도록 valid length를 전달한다. CUDA graph fixed buffer와 logical token count를 구분한다.

### 55.11.7 TP×EP deadlock 사고

Rank 0/1은 expert 내부 TP all-reduce를 먼저 호출하고 rank 2/3은 EP all-to-all을 먼저 호출한다고 하자. Group이 다르더라도 resource/call dependency가 cycle을 만들거나 같은 ranks가 mismatched sequence를 호출하면 hang한다.

Rank별 last-N collective ring buffer를 모은다. Sequence number는 group-local이므로 group generation/id와 함께 비교한다. Input shape/dtype, call site와 enqueue 여부를 기록한다. 한 rank가 conditional branch로 empty-token collective를 skip했는지 확인한다.

MoE에서 특정 rank에 token이 0개여도 protocol이 collective participation을 요구할 수 있다. Empty tensor와 split sizes zero로 같은 all-to-all sequence에 들어간다. “할 일이 없으니 return” branch가 peer를 기다리게 하지 않는지 본다.

Fix 뒤 imbalance fixtures로 token counts `[4,0]`, `[3,1]`, `[0,4]`를 시험한다. TP/EP group member list와 collective order가 모든 rank에서 같고 zero-token output ownership이 명확해야 한다. NCCL channel tuning은 이 semantic 합의 뒤에 한다.

## 55.12 수직 운영 원장

한 request의 trace는 ingress DP replica 선택, PP stage/microbatch, layer의 TP parameter shard와 local op, collective group/sequence, EP router assignment와 all-to-all, reverse scatter, final stage/output owner를 잇는다.

Metric에는 group별 collective count/bytes/time이 유용하지만 exact member list와 shapes를 label로 넣지 않는다. Trace/debug manifest에 group id와 tensor contract를 둔다. Hang watchdog은 rank별 last collective sequence와 call site를 수집한다.

### 작은 fixture 회귀 matrix

TP 1/2/4, PP 1/2, DP 1/2, EP 1/2를 무작정 모두 곱하기보다 ownership edge를 pairwise로 cover한다. TP column→row, PP boundary abort, DP routing persistence, EP dispatch/inverse, TP×EP group nesting과 PP×TP activation layout을 우선한다.

각 fixture는 요청 성공만 assert하지 않는다. Rank-local tensor 값, collective 전/후 owner, group members, next consumer input과 final reference를 확인한다. Shape-only test는 gather order와 permutation wrong answer를 놓친다.

수직 추적은 rank-coordinate manifest를 만드는 데서 시작한다.

World ranks 0–15를 `(dp,pp,tp,ep)`에 매핑한다. 단순히 사전식 순서로 숫자를 나누지 않고 current group construction loop가 어느 coordinate를 fastest-changing dimension으로 두는지 읽는다. Fixture manifest에는 global rank마다 네 coordinate와 TP/PP/DP/EP group id, group-local rank를 적는다.

예를 들어 rank 5가 `(dp=0,pp=1,tp=0,ep=1)`이라고 가정하는 것은 source ordering이 그 식을 사용할 때만 맞다. Rank-to-coordinate 함수와 inverse coordinate-to-rank 함수를 서로 적용해 identity를 확인한다. Config mutation이나 elastic restart 뒤 group generation이 바뀌면 manifest generation도 올린다.

모든 rank의 manifest를 coordinator가 모아 expected invariant를 검사한다. 각 TP group size, PP first/last stage, DP replica member set, EP expert-owner coverage가 맞는지 본다. 한 rank가 다른 environment/default로 effective size를 계산하면 collective 시작 전에 실패시킨다.

Group creation order도 기록한다. Distributed library가 모든 process에서 같은 순서로 subgroup을 만들도록 요구하는 경우, condition branch 때문에 일부 rank가 EP group 생성을 건너뛰면 뒤 group handles가 어긋날 수 있다. “이 rank는 member가 아니므로 group을 만들 필요 없다”는 최적화가 API contract와 맞는지 확인한다.

### 55.12.1 model layer owner 표

PP stage마다 embedding, decoder layer range, final norm, LM head owner를 적는다. Tied embedding/head가 서로 다른 stage라면 shared weight 또는 communication이 어떻게 처리되는지 확인한다. Stage에 없는 layer parameter가 loader report에서 missing으로 오해되지 않게 global expected와 stage-local expected inventory를 나눈다.

각 stage 안에서 TP rank는 parameter slice를 소유한다. Layer별 table에 global tensor name/shape, shard dimension, global range, padding, local shape, replicated auxiliaries와 loader key를 둔다. EP가 있으면 expert id range를 추가한다.

Ownership table은 static parameter에만 쓰지 않는다. Forward activation, residual, KV cache, router scores, dispatched token buffer와 logits도 row로 둔다. Static weight owner와 dynamic request-state owner를 구분한다.

Data-parallel replica는 같은 logical parameter ranges를 복제할 수 있지만 다른 physical storage generation을 가진다. Hash는 같아야 할 수 있어도 pointer와 graph owner는 다르다. Replica별 model load 완료와 request admission generation을 기록한다.

### 55.12.2 request가 DP replica에 들어오는 순간

Request R의 prompt token 4개가 DP replica 1에 route됐다고 하자. Router decision은 request lifetime 동안 sticky해야 한다. Replica 1 안의 scheduler가 sequence id, KV owner와 streaming output owner를 만든다. DP group의 다른 replica는 R을 모를 수 있다.

Prefill 첫 step에서 PP stage 0의 TP/EP ranks가 R token metadata를 공유한다. Stage 0만 tokenizer output을 갖고 stage 1이 activation만 받는지, 모든 stage scheduler가 동일 request metadata를 갖는지는 implementation에 따라 다르다. Trace로 실제 owner를 적는다.

R의 decode step을 load balancer가 replica 0으로 보내면 replica 0에는 KV와 scheduler state가 없다. Request-id collision로 다른 state를 찾을 수도 있다. Sticky routing key, explicit state transfer 또는 shared cache 없이 step-level reroute를 허용하지 않는다.

Abort 역시 original replica owner로 가야 한다. Frontend가 replica id를 잃으면 모든 replica에 broadcast할 수 있지만 cleanup idempotency와 비용이 달라진다. Current ingress/core protocol에서 abort route와 final response owner를 확인한다.

### 55.12.3 stage 0에서 embedding과 첫 TP layer

Token ids는 vocab-parallel embedding을 통과할 수 있다. Rank별 vocab range 밖 token은 zero contribution을 만들고 reduction으로 embedding을 복원하거나 owner rank의 row를 gather하는 설계가 있다. Exact implementation의 local output 의미를 확인한다.

Embedding 결과 `[tokens,H]`가 replicated되면 첫 QKV column-parallel projection이 각 rank의 head shard를 만든다. Fixture에서 H를 4로 줄이고 TP=2로 두면 rank 0은 first two output features, rank 1은 last two를 계산한다. Basis values를 기록한다.

Attention local output은 complete head shard다. Output projection은 local head feature K slice를 사용해 global hidden coordinate partial을 만든다. Rank별 partial을 sum하기 전 residual을 더하는지, reduction 뒤 더하는지 source에서 확인한다. Residual이 replicated라면 각 partial에 더한 뒤 all-reduce하면 TP번 더해진다.

Custom all-reduce가 있으면 wrapper input/output contract를 본다. Buffer registration, communicator와 kernel 내부는 다음 장으로 넘기지만 semantic sum과 replicated output 여부는 assert한다. Disable custom path에서도 같은 reference를 얻어야 한다.

### 55.12.4 MLP에서 column과 row를 연결한다

Gate/up weight의 I output rows를 같은 partition rule로 나눈다. Rank 0 gate shard와 rank 0 up shard는 동일 intermediate coordinates를 가리켜야 elementwise product가 맞다. Loader가 packed gate-up segment를 다른 order로 slice하면 shape가 맞아도 pairing이 바뀐다.

Small fixture에서 gate output `[g0,g1|g2,g3]`, up `[u0,u1|u2,u3]`로 나누고 local activation product `[f(g0)u0,f(g1)u1]` 등을 계산한다. Down weight의 corresponding K columns를 각 rank가 소유하고 hidden partial을 만든다. Sum 뒤 reference dense MLP와 비교한다.

Rank 1 intermediate shard가 padded rows를 포함하면 down projection의 padded columns이 zero semantic을 가져야 한다. Padding row가 nonzero weight/activation이면 reduction 결과를 오염시킨다. Loader zeroing과 valid range를 함께 본다.

Collective를 residual add나 norm과 fuse하더라도 next layer가 받는 hidden은 complete replicated인지 sequence shard인지 명시한다. 이름에 fused가 붙었다는 이유로 semantic boundary를 건너뛰지 않는다.

### 55.12.5 PP stage boundary transaction

Stage 0이 마지막 local layer를 끝내면 microbatch R0 activation과 optional residual/metadata를 stage 1에 보낸다. Send record에는 pipeline generation, microbatch id, token range, tensor schema hash와 receiver가 있다. Receiver는 expected record와 match한 뒤 buffer를 forward에 publish한다.

Stage 0이 send enqueue 후 buffer를 다음 microbatch에 재사용하려면 transfer completion dependency가 필요하다. 여기서는 completion owner만 기록하고 stream/event 구현은 56장으로 넘긴다. Receiver가 copy 완료 전에 layer를 시작하지 않는 불변식도 둔다.

Microbatch R0가 abort되면 stage 0이 아직 send 전인지, in-flight인지, stage 1 queue에 들어갔는지에 따라 cleanup이 다르다. Cancellation generation을 보내거나 receiver가 stale activation을 drop해야 한다. 다음 R1이 같은 buffer/tag를 재사용해도 generation으로 구분한다.

Stage 1이 final logits를 만들면 streaming output owner가 frontend와 연결된다. Stage 0에 output을 되돌릴지 last stage가 직접 보낼지는 current engine protocol을 본다. Request terminal state가 모든 stages의 buffers/KV references를 닫는지 확인한다.

### 55.12.6 MoE router와 token dispatch

Stage 안 MoE layer에 R의 token 4개가 들어간다. Router logits와 top-k assignment는 TP replicated인지 shard인지 확인한다. Expert placement table은 expert id→EP rank와 local expert index를 제공한다.

Top-1 fixture `[E0,E1,E1,E0]`에서 dispatch permutation `[t0,t3,t1,t2]`, peer split `[2,2]`를 만든다. Source rank별 input token distribution이 있다면 all-to-all send matrix를 rank×peer로 펼친다. 각 cell의 token ids와 byte count를 적는다.

Receiver rank 0은 E0 local batch `[t0,t3]`, rank 1은 E1 `[t1,t2]`를 계산한다. Expert output에 unique constant를 더해 owner가 보이게 한다. Reverse all-to-all과 inverse permutation 뒤 original token order를 확인한다.

Top-2에서는 token 복제와 combine weight를 넣는다. Dispatch entry는 `(original_token,expert,topk_slot,weight)` identity를 가진다. Reverse 후 같은 token의 contributions를 weight sum한다. Expert result order만으로 original position을 추정하지 않는다.

### 55.12.7 TP와 EP가 중첩된 expert

Expert weight가 TP=2로 shard되면 EP dispatch로 token이 expert group에 도착한 뒤 expert owner 내부 TP ranks가 feature shard/partial 연산을 한다. Token dispatch all-to-all과 feature all-reduce의 group, axis와 sequence가 다르다.

Trace는 먼저 EP transition `token owner→expert owner`, 다음 TP transition `feature shard/partial→expert output`으로 기록한다. 하나의 `model_parallel_group` label로 합치지 않는다. Group local rank가 우연히 같아도 communicator id를 확인한다.

Deadlock fixture는 rank마다 TP all-reduce와 EP all-to-all 호출 order를 의도적으로 뒤집는다. Scheduler가 모든 rank에서 동일 MoE branch/token presence를 보장하는지 확인한다. Empty token rank도 collective에 참여해야 하는 protocol이면 zero-size input으로 같은 sequence를 호출한다.

Expert load balancing이 rank별 token counts를 다르게 만들어도 split sizes exchange가 합의되어야 한다. Fixed equal split을 가정하지 않는다. Padding을 쓰면 valid count와 inverse mapping에서 제거한다.

### 55.12.8 logits와 sampling owner

Final PP stage의 LM head가 vocab parallel이면 rank별 logits feature shard를 만든다. Sampling algorithm이 distributed top-k를 지원하면 full gather를 피할 수 있고, 단순 sampler가 replicated logits를 요구하면 gather한다. Next consumer contract가 collective를 결정한다.

Distributed top-k는 local candidate token ids에 global vocab offset을 붙여야 한다. Candidate score와 token id pair를 merge한다. Padded/invalid vocab rows를 mask한다. Gather order를 token-id range와 대조한다.

Sampled token은 next decode step의 request owner와 모든 relevant TP/PP stages에 전달돼야 한다. PP first stage가 embedding을 수행하면 last→first feedback path가 있다. Token value, request/microbatch/step generation을 기록한다.

Streaming text output은 한 owner만 client에 내보내야 한다. 모든 TP ranks가 같은 sampled token을 갖더라도 중복 stream을 보내지 않는다. Output rank selection과 failure failover를 확인한다.

### 55.12.9 종료와 resource release

Request R이 finish/abort되면 DP replica scheduler, 모든 PP stage의 request/microbatch state, TP-local KV blocks, EP temporary dispatch buffers와 output queue가 닫혀야 한다. Global completion을 한 rank의 EOS만으로 선언하는지 coordinator 합의를 쓰는지 본다.

Collective in-flight 중 abort가 오면 rank 일부만 call을 건너뛰지 않도록 한다. Current iteration collective를 완료하고 output을 discard하거나 group-wide cancellation/error protocol을 사용해야 한다. Host queue에서 request를 지웠다는 사실만으로 device collective가 사라지지 않는다.

Cleanup ledger는 resource owner와 final reference count를 rank별로 가진다. 다음 request가 same sequence/microbatch buffer를 재사용할 때 generation이 올라간다. Stale PP recv나 EP inverse mapping이 새 request에 남지 않게 한다.

실행을 재구성한 뒤에는 rank-local observation과 metrics를 같은 좌표계에 놓는다.

Hang 조사에는 모든 rank의 last collective record가 필요하다. Group id/generation, sequence, collective kind, input/output tensor contract, call site, enqueue/completion state를 bounded ring buffer에 둔다. 한 rank log만으로 peer mismatch를 알 수 없다.

Metrics는 TP/PP/EP group별 bytes/time/count, PP queue depth/bubble, DP request load, EP token imbalance를 제공할 수 있다. Exact group members, shapes와 request ids는 high-cardinality이므로 trace/debug manifest에 둔다.

Collective time이 길어도 semantic 오류와 transport 병목을 구분한다. 모든 ranks가 동일 call에 들어왔고 input ownership이 맞다면 56장의 NCCL/topology 분석으로 내려간다. Call/group/order가 다르면 transport tuning 전에 고친다.

그 증거가 모두 이어졌을 때 다음과 같이 완료를 판정한다.

정상 판정은 “TP=4였다”가 아니다. “QKV projection은 head-axis complete shards를 만들었고 local attention 뒤 output projection은 hidden partial을 생성했다. TP group 3 all-reduce sequence 18이 sum하여 replicated hidden을 만들었으며 MLP gate/up shard와 down partial도 reference와 일치했다.”처럼 쓴다.

Pipeline 판정은 “Microbatch 7 activation schema와 generation이 stage 0 send와 stage 1 recv에서 같았고 abort microbatch 6은 receiver에서 drop됐다.”고 쓴다. EP 판정은 dispatch split, expert owner, inverse permutation과 token reference를 포함한다.

최초 divergence가 owner table, local matmul, group selection, collective input, collective output order, inverse permutation 또는 next consumer 가운데 어디인지 말할 수 있어야 한다. 그래야 model code, group construction, scheduler와 transport 담당자가 같은 증거로 협업한다.

Source를 읽는 순서도 이 owner 사슬을 따른다. 먼저 command/config가 parallel sizes를 어느 object에 넣는지 확인한다. 그 다음 distributed initialization이 world rank와 local device를 정하고 TP/PP/DP/EP member lists를 생성하는 함수를 찾는다. Accessor 이름만 모으지 않고 member construction loop와 coordinate arithmetic을 손으로 재현한다.

Group object가 custom wrapper라면 underlying torch process group, device communicator, CPU control group과 rank mapping을 분리한다. 같은 semantic group이 control metadata와 CUDA tensor용 두 communicator를 가질 수 있다. Log의 object id 하나를 group identity로 쓰지 않고 stable generation/member digest를 만든다.

vLLM group source에서는 initialization entry, `_TP`, `_PP`, `_DP` 같은 global owner가 설정되는 위치, group rank lists와 accessor를 연결한다. Model parallel initialization을 두 번 호출할 때 idempotency/assertion과 destroy path도 본다. Old group이 남은 채 새 generation을 만들면 cached layer/communicator가 split될 수 있다.

SGLang coordinator는 group wrapper가 all-reduce/all-gather와 send/recv를 어떻게 노출하는지, custom all-reduce enable/disable이 semantic output을 유지하는지 본다. Server arguments의 sizes가 runtime context로 전달되고 model runner가 같은 group accessor를 쓰는지 확인한다. Scheduler-side rank coordinate와 model-side coordinate가 별도 계산이면 대조한다.

Transformers DeviceMesh에서는 mesh dimension name과 physical rank layout을 읽는다. TP plan entry가 module path pattern을 concrete module에 적용하고 input/output DTensor placement를 어떻게 바꾸는지 확인한다. `ColwiseParallel`이라는 이름만 보고 gather 여부를 추정하지 않고 output layout parameter와 next module plan을 본다.

DTensor는 placement transition에서 communication을 삽입할 수 있다. Source에 explicit all-gather가 없더라도 `Shard`에서 `Replicate` 또는 shard dimension 변경이 collective semantic을 요구한다. Runtime trace에서는 DTensor redistribution call과 output placement를 기록한다.

llama.cpp는 process group TP/PP와 다른 multi-device model을 가질 수 있다. Tensor split ratio, layer offload와 backend scheduler가 tensor/node를 device에 배치하고 copy dependency를 삽입하는 경로를 읽는다. 한 process의 multiple CUDA devices와 multiple processes의 collective group을 같은 TP로 부르지 않는다.

현재 audited llama.cpp path에 distributed all-reduce/all-to-all group이 없다면 negative scope를 명시한다. Tensor row split 또는 device offload가 있다고 vLLM-style TP를 주장하지 않는다. 반대로 device별 partial computation과 sum이 있다면 exact graph op와 ownership을 근거로 설명한다.

Parameter loader source는 forward와 함께 읽는다. Column-parallel layer가 output row를 shard한다고 선언했는데 loader가 input column을 slice하면 local matmul은 shape가 맞거나 transpose convention 때문에 늦게 실패할 수 있다. Global checkpoint shape, parameter storage orientation, shard dimension와 global offset을 한 표에 둔다.

Packed QKV와 gate/up은 logical segment마다 shard range가 다르다. Query heads와 KV heads의 global/local cardinality를 계산하고 packed physical offset으로 변환한다. Rank-local parameter 전체를 균등 slice하는 것이 아니라 segment-specific loader가 source tensor를 해당 destination segment에 넣는지 확인한다.

Quantized parameter는 packed coordinate에서 shard할 수 있다. Logical K/N range가 group/pack boundary를 자르지 않는지, padding과 scale/ZP shard가 main weight와 일치하는지 본다. 이 장은 byte packing을 50장에 맡기되 local owner range와 collective output 의미는 기록한다.

Adapter가 있을 때 base TP shard와 LoRA A/B shard가 같은 local linear contract를 만들어야 한다. Base output은 feature shard 또는 partial인데 delta만 replicated라면 add 위치와 collective가 달라진다. 51장의 adapter mapping을 가져오되 adapter lifecycle을 반복하지 않는다.

Attention KV cache ownership은 QKV parameter shard에서 파생된다. TP rank마다 KV heads를 소유하거나 replication할 수 있다. Sequence request owner는 DP replica에 있고 layer cache tensor는 TP ranks에 분산된다. “KV owner=replica 1”을 하나의 pointer로 축약하지 않고 `(replica,tp_rank,layer,block,head_range)`로 기록한다.

PP가 있으면 각 stage는 자기 layer KV만 소유한다. Request abort가 모든 stages와 TP ranks의 cache blocks를 해제해야 한다. Stage 0 cleanup만 성공하고 stage 1 block이 남으면 leak다. Terminal acknowledgment 또는 coordinator state를 확인한다.

EP expert weights도 stage-local이다. Expert id global range, EP rank owner, expert-local index와 TP shard가 parameter inventory를 결정한다. Router가 global expert id를 local index로 변환하는 table이 loader placement와 같아야 한다. Expert 5 weight가 rank 1 local expert 1인데 router가 local 0을 선택하면 shape는 맞고 wrong expert가 실행된다.

All-to-all split sizes는 token counts뿐 아니라 hidden payload stride/dtype와 맞아야 한다. Quantized communication이나 FP8 dispatch가 있다면 scale/metadata ownership을 함께 전달한다. 이 장에서는 transport encoding 내부를 다루지 않지만 receiver가 어떤 logical hidden을 얻는지 명시한다.

Collective fusion은 여러 semantic transition을 한 kernel/API에 넣을 수 있다. Reduce-scatter와 norm, all-gather와 matmul, expert dispatch와 permutation이 fused될 수 있다. Debug mode에서 fusion 전 logical inputs와 output placement를 복원할 reference path를 둔다.

Async collective는 handle owner를 명시한다. Enqueue 이후 input buffer를 누가 보유하고 언제 reuse할 수 있는지, output이 어느 stream/event 뒤 complete인지 기록한다. NCCL stream/event detail은 다음 장으로 넘기지만 request/layer owner가 completion 전에 lifecycle을 끝내지 않는지는 여기서 본다.

In-place collective는 input/output storage가 같아 owner transition이 숨는다. Before semantic `partial`, after completion `complete replicated`처럼 generation을 올린다. Pointer equality를 state equality로 쓰지 않는다. Captured CUDA graph에서 같은 address가 step마다 다른 logical sequence를 가질 수 있으므로 request/step generation도 둔다.

All-gather order는 group local ranks와 shard global ranges의 mapping이다. 단순 local-rank ascending이 global coordinate ascending이 아닐 수 있다. Mesh/group construction이 strided rank list를 만들면 각 local rank의 shard range를 explicit concat plan으로 기록한다.

Reduce-scatter도 output chunk index와 desired next placement를 연결한다. Rank 0이 first chunk를 받는다는 default가 next module's rank shard와 맞는지 본다. Custom implementation이 input chunks를 pre-permute하면 wrapper contract를 확인한다.

All-to-all은 send matrix를 사용한다. Row는 source rank, column은 destination rank, cell은 ordered token entries다. 각 rank의 send splits row sum은 local dispatched entries와 같고 recv splits column sum은 received entries와 같아야 한다. Global sums와 inverse mapping cardinality를 assert한다.

PP send/recv는 collective group 전체가 아니라 peer pair지만 pipeline group order를 사용한다. First/last stage와 next/prev rank accessor가 wrap-around하는지, inference forward만 일방향인지, token feedback이 별도 channel인지 본다. Training backward semantics를 serving forward에 투영하지 않는다.

DP synchronization이 serving에서도 있을 수 있다. Model weight update, load balancing stats, expert load balance 또는 shutdown control이 DP group collective를 사용할 수 있다. Request data path와 control path를 분리한다. “Serving DP에는 collective가 없다”고 절대화하지 않는다.

Rank failure에서 process group semantics를 본다. 한 rank가 죽으면 remaining ranks가 collective error를 받고 request를 terminal failure로 정리하는지, group을 elastic하게 rebuild하는지 current implementation scope를 확인한다. 새 group generation과 old in-flight request를 섞지 않는다.

Retry가 가능하면 local output buffer와 collective sequence를 reset한다. Partial all-reduce 후 error가 났는데 같은 buffer로 retry하면 contribution을 두 번 더할 수 있다. Fresh buffer 또는 well-defined rollback을 사용한다. Idempotency를 host request id만으로 보장할 수 없다.

Metric의 collective bytes를 손으로 검산한다. All-reduce logical payload는 `numel×element_size`지만 algorithm traffic은 다음 장의 문제다. 여기서는 wrapper input/output logical bytes와 calls/tokens를 기록한다. All-to-all은 actual send counts 합과 hidden byte stride를 곱한다.

PP throughput은 microbatch pipeline occupancy와 bubble을 본다. 그러나 stage imbalance metric 전에 activation owner와 ordering이 정확한지 확인한다. Wrong microbatch를 빠르게 처리하는 것은 성능 성공이 아니다. Stage별 queue depth, active microbatch generation과 completion을 trace한다.

EP load imbalance는 expert별 token counts로 본다. Token count 0 rank가 protocol에 참여했는지, padding tokens가 compute metric에 포함되는지 구분한다. A2A time과 expert GEMM time을 분리하되 routing correctness fixture를 항상 같이 둔다.

TP communication/computation overlap이 있으면 collective output을 next op가 언제 소비하는지 timeline에 둔다. Partial shard를 overlap 명목으로 너무 일찍 읽지 않는다. Chunked collective라면 chunk별 complete range와 next consumer range를 대응시킨다.

Sequence parallel은 token axis ownership을 추가한다. Norm/dropout 없는 inference에서도 activation을 token shard로 유지할 수 있다. TP feature shard와 sequence shard를 tuple placement로 적는다. `[tokens/TP,H]`와 `[tokens,H/TP]`는 numel가 같아도 collective가 다르다.

Context parallel은 long sequence attention의 query/KV axis를 분할할 수 있다. 이 장 범위에서 등장하면 group와 input/output ownership만 설명하고 attention communication algorithm은 별도 장으로 넘긴다. TP/CP/DCP 이름을 option 표로 확장하지 않는다.

Speculative decode는 draft/target requests와 verify tokens가 각 replica/stage group에서 일관된 owner를 가져야 한다. Accepted token compaction이 EP/TP mapping과 같은 permutation을 사용하는지 확인한다. 일반 decode fixture 하나로 모든 dynamic token path를 커버했다고 보지 않는다.

Prefix cache 공유가 DP replicas 사이에 있다면 cache block content와 ownership transfer를 명시한다. Logical hit가 다른 replica에서 발생해도 local TP/head layout으로 materialize돼야 한다. External KV transport 내부는 별도 장으로 넘기되 request가 어느 replica에서 decode할 수 있게 됐는지 commit boundary를 기록한다.

모델 reload는 group과 parameter generation을 함께 바꿀 수 있다. Same ranks/group handles를 재사용해도 layer owner ranges나 expert placement가 달라지면 new manifest다. Old graph, cache와 in-flight request가 new weights/groups를 참조하지 않게 drain한다.

Release upgrade diff는 option defaults뿐 아니라 group coordinate order, layer partition policy, collective fusion, async wrapper와 expert permutation을 본다. Same world/TP sizes여도 rank member order가 바뀌면 checkpoint shard/cache/trace interpretation이 달라질 수 있다.

운영 runbook의 첫 단계는 모든 ranks의 manifest 수집이다. Product/coordinate, group members, layer/expert shard inventory와 request replica를 비교한다. 둘째는 hang/wrong-output workload의 last collective records다. 셋째는 hand fixture의 rank-local value와 post-collective reference다.

Hang이면 call participation과 sequence를 먼저 본다. 모든 rank가 같은 semantic call에 들어왔고 completion만 늦다면 56장의 transport/topology로 내려간다. Rank 하나가 다른 group/call/shape에 있다면 그 first branch를 찾는다.

Wrong output이면 local op reference를 먼저 본다. Local이 틀리면 loader/shard/model code다. Local이 맞고 collective 뒤 틀리면 group, operation, reduction/concat order다. Collective 뒤 맞고 next layer부터 틀리면 placement annotation 또는 consumer expectation이다.

OOM이면 parameter shard bytes, KV owner, communication workspace와 temporary gathered/dispatch buffer를 owner별로 센다. All-gather로 replicated tensor가 생기는 lifetime, PP activation queues와 EP padding이 peak를 만들 수 있다. Static weight size만 보지 않는다.

성능이면 compute/communication overlap과 payload를 보되 semantic-correct reference와 동일 workload를 사용한다. Collective를 제거해 빨라졌다면 next consumer가 shard를 직접 지원하는지 확인한다. Complete tensor를 요구하는데 sync를 없앤 결과는 최적화가 아니다.

재현 bundle은 model/engine pins, effective parallel config, rank/group manifests, global/local tensor ledger, collective traces, small input/weight values, PP/DP/EP identities와 final reference를 포함한다. Customer weight 없이 synthetic matrix로 group/collective correctness를 재현할 수 있다.

결국 distributed source walk의 산출물은 거대한 call graph가 아니다. Global tensor coordinate가 어느 rank에서 어떤 local state가 되고, collective가 어떤 new owner를 만들며, next consumer가 그것을 어떻게 읽는지 연결한 ownership graph다. 이 graph가 있으면 unfamiliar wrapper와 fused op도 의미 좌표에 놓을 수 있다.

## 55.13 ownership graph를 운영에 적용한다

새 distributed engine을 읽을 때 먼저 TP·PP·DP·EP option 목록을 외우지 않는다. Effective rank-coordinate와 group construction을 찾는다. Model constructor와 loader가 global parameter를 어느 rank range로 나누는지 기록한다. Forward local op가 complete shard인지 partial sum인지 계산한다.

다음 consumer가 원하는 layout을 적으면 collective 의미가 결정된다. Feature shards를 모두 필요로 하면 gather, 동일 좌표 partial을 complete하게 만들면 reduce, reduced shard를 원하면 reduce-scatter, expert owner로 token을 보내면 all-to-all, layer owner를 넘기면 send/recv다.

좋은 판정은 구체적이다. “MLP down projection rank outputs는 K-shard partial `[tokens,H]`였지만 wrapper가 feature shard로 표시해 all-gather했다. First divergence는 collective input semantic이며 sum all-reduce 뒤 reference와 일치했다.” 또는 “EP compute는 맞았지만 reverse permutation generation이 이전 batch 것이어서 token 2/3이 바뀌었다.”처럼 쓴다.

NCCL은 이 의미를 byte transfer로 실현한다. 다음 장에서는 동일 collective가 topology, channel, protocol과 stream에서 어떻게 움직이는지 본다. 그러나 어떤 transport도 잘못된 owner와 collective를 고쳐 주지 못한다. Global coordinate, local state와 new owner를 끝까지 추적해야 분산 추론의 값과 lifecycle이 닫힌다.

이 원칙을 code review에 적용할 때는 함수 이름보다 입출력 placement를 먼저 적는다. `ColumnParallelLinear.forward`라는 이름을 봤다면 weight의 어느 global axis를 slice했고 input은 replicated인지 sharded인지, output은 complete shard인지 gathered replica인지 확인한다. `RowParallelLinear`도 local partial을 실제로 언제 reduce하는지, caller가 이미 input을 shard했는지 본다.

Flag가 `gather_output=false`라면 collective가 없다는 사실만 적지 않는다. Output feature shard의 next consumer와 global range를 기록한다. Caller가 complete tensor를 기대하면서 flag를 끄면 최적화 option이 correctness option으로 변한다. 반대로 next layer가 same shard placement를 지원한다면 불필요한 gather를 없앨 수 있다.

`input_is_parallel=true` 같은 flag도 누가 input shard를 만들었는지 provenance가 필요하다. Replicated input을 shard라고 잘못 표시하면 각 rank가 같은 full X를 local W slice와 곱하거나 slice 없이 사용해 shape/error가 달라진다. Flag 선언과 runtime tensor placement를 assert한다.

All-reduce wrapper는 operation이 sum인지 확인한다. Max/min/average semantics가 있는 generic API에서 default를 추정하지 않는다. Sum 뒤 scaling을 수행하는지, world/group size로 나누는지 확인한다. Inference linear partial은 보통 sum이지 mean이 아니다.

All-gather wrapper는 gather dimension과 output allocation order를 확인한다. Some API는 first dimension으로만 gather해 caller가 transpose/reshape할 수 있다. Logical N feature concat과 physical dim0 gather를 연결하는 view/permute가 correctness 일부다.

Reduce-scatter wrapper는 input layout이 rank chunks 순서로 준비됐는지 본다. Interleaved feature나 sequence chunks를 contiguous rank chunks로 rearrange하는 전처리가 있을 수 있다. Wrapper 안에서 숨은 transpose/copy가 발생하면 memory와 latency owner를 기록한다.

All-to-all wrapper는 fixed equal split API와 variable split API를 구분한다. MoE token counts가 불균등한데 equal split을 위해 padding한다면 padding token identity와 valid mask가 필요하다. Variable split이면 peer counts exchange와 capacity가 맞아야 한다.

Send/recv wrapper는 peer global rank인지 group-local rank인지 확인한다. Pipeline accessor가 local stage index를 global rank로 변환하는지 본다. TP rank가 여러 개면 stage 0 TP rank i가 stage 1의 matching TP rank i와 통신하는지, gather/scatter boundary가 있는지 source로 고정한다.

Group accessor가 반환하는 local rank와 model loader의 shard id가 같은 coordinate를 사용하는지도 확인한다. Loader는 tp_rank를 global rank modulo TP로 계산하고 communicator는 different mesh ordering을 쓰면 weight shard와 collective order가 어긋난다. Central rank-coordinate owner를 사용하는 것이 안전하다.

Model class의 layer assignment는 `start_layer`, `end_layer`와 missing layer placeholders로 나타날 수 있다. PP stage가 자기 범위 밖 module을 만들지 않는지, empty layers를 identity로 두는지, weight loader가 stage-local expected keys만 consume하는지 확인한다.

PP first stage는 input embeddings, last stage는 norm/LM head를 보통 소유하지만 tied weights나 multimodal encoder가 경계를 바꿀 수 있다. Architecture-specific owner table을 생성한다. “PP0=embedding, last=head”를 universal rule로 쓰지 않는다.

Intermediate tensor container가 hidden만이 아니라 residual, image embeddings, rotary/position state를 운반할 수 있다. Container field를 나열하기보다 다음 stage의 forward signature가 실제 소비하는 값과 lifecycle을 연결한다. Optional field absence/default도 schema generation으로 기록한다.

Data-parallel coordinator가 request batch를 scatter하거나 output을 gather할 수 있다. Serving DP replicas가 independent라고 해도 global metrics, health와 model update에 control collective가 있을 수 있다. Data plane request ownership과 control plane synchronization을 분리한다.

DP rank가 TP/PP group bundle을 소유한다면 replica failure 범위는 bundle 전체다. TP rank 하나만 다른 replica에 대체해 request를 계속할 수 있는지 current architecture를 확인한다. KV와 collective group generation 때문에 단순 rank substitution은 어려울 수 있다.

Expert placement가 dynamic load balancing으로 바뀌면 expert owner generation과 router mapping을 원자적으로 전환해야 한다. In-flight token dispatch는 old placement를 snapshot하고 new requests는 new placement를 사용한다. Weight migration 완료 전에 router mapping을 publish하지 않는다.

Expert replication이 있으면 expert id의 owner가 하나가 아닐 수 있다. Router/load balancer가 replica를 고르고 reverse mapping이 selected copy를 기억해야 한다. EP rank count만으로 owner를 계산하지 않는다.

MoE capacity/padding은 token identity를 보존해야 한다. Dropped token, padded token과 rerouted token의 policy를 기록한다. All-to-all byte counts가 맞아도 capacity drop이 model semantics를 바꿀 수 있다. Router reference와 final token output을 비교한다.

TP/EP combined quantized MoE는 expert-local weight representation도 rank별이다. All-to-all hidden dtype와 expert kernel input dtype 사이 conversion owner를 확인한다. Collective는 hidden을 옮길 뿐 weight shard를 복원하지 않는다.

PP와 EP가 결합되면 stage boundary 전/후 어느 stage가 router/expert layer를 완전히 소유하는지 본다. Expert를 stages 사이 나누는 별도 architecture가 아니라면 MoE all-to-all은 stage-local group이어야 한다. Wrong PP coordinate를 포함한 EP group은 stage activation lifecycle과 충돌한다.

Group construction test는 membership set뿐 아니라 intersection을 확인한다. 각 rank가 기대한 수의 TP/PP/DP/EP groups에 속하는지, 서로 다른 logical groups가 unintended identical process group으로 alias되지 않는지 본다. Alias가 intentional이면 semantic sequence 충돌을 방지하는 ordering을 확인한다.

Collective sequence test는 branch coverage를 포함한다. Normal dense layer, MoE with tokens, MoE empty tokens, abort, speculative verify와 graph/eager가 모든 ranks에서 compatible call sequence를 만드는지 확인한다. One rank의 early return이 peer hang을 만들지 않는다.

CUDA graph는 collective calls도 capture할 수 있다. Capture group, buffer shapes와 sequence가 replay workload에 맞는지 확인한다. Graph bucket 밖 workload가 eager로 갈 때 group sequence가 다른 graph replay와 interleave되어도 communicator ordering이 안전한지 다음 장과 연결한다.

Overlap은 ownership을 더 세밀하게 만든다. Tensor chunk 0은 collective 완료돼 next op가 읽고 chunk 1은 in-flight일 수 있다. Whole tensor boolean ready 대신 complete ranges와 dependency를 기록한다. Wrong range read는 race wrong answer가 된다.

Memory pool reuse도 collective completion과 결합된다. Async send/all-reduce input을 allocator가 다른 request에 주지 않도록 handle/event reference를 가진다. Host-side request finish가 device buffer free 조건과 같지 않다.

Hang watchdog은 timeout 후 모든 ranks를 동시에 dump해야 한다. 순차 dump 동안 state가 변하면 비교가 어렵다. At least group/sequence/call site snapshot generation을 맞춘다. Stack trace만 수집하면 custom op 내부 group identity가 빠질 수 있다.

Wrong-answer watchdog은 full tensor dump 대신 small coordinate checksums를 사용할 수 있다. Feature shard marker, partial sum norm, gather range hash, router/inverse permutation digest와 KV head range를 rank별로 남긴다. Sensitive activation 원문은 저장하지 않는다.

Fault injection은 group initialization failure, rank collective timeout, PP message drop, DP reroute, EP variable count mismatch와 buffer allocation failure를 포함한다. Expected terminal state와 rollback resources를 정의한다. “Process가 죽었다”만 assert하지 않는다.

Group destroy/reinitialize test에서는 old accessor와 layer wrapper가 stale group을 보유하지 않는지 본다. Global singleton을 reset해도 module field가 old communicator reference를 갖고 있을 수 있다. New request admission 전에 model runner와 graph를 rebuild해야 할 수 있다.

Observability label은 `tp`, `pp`, `ep` size 정도로 bounded할 수 있지만 group member lists와 sequence는 trace로 보낸다. DP replica id도 small bounded label일 수 있으나 elastic generation과 혼동하지 않는다. Exact request id는 metric label에 넣지 않는다.

성능 보고서는 collective payload denominator를 명시한다. Tokens, hidden, dtype, TP/EP sizes와 local shapes가 같아야 비교 가능하다. Global model 크기만 같고 batch/token distribution이 다르면 A2A/all-reduce bytes가 달라진다.

Optimization 제안은 next consumer contract를 포함한다. “All-gather 제거”는 next op가 sharded placement를 직접 소비하도록 바뀌는 code와 함께여야 한다. “All-reduce를 reduce-scatter로”는 next layout과 shard axis가 바뀌고 이후 loader/layer가 그 placement를 받아야 한다.

Pipeline microbatch 수를 늘려 bubble을 줄이면 activation buffers와 request generations이 늘어난다. Memory peak와 abort cleanup을 함께 검증한다. Throughput만 보고 buffer owner limit을 넘기지 않는다.

Expert batching을 늘려 GEMM 효율을 높이면 token waiting과 A2A split이 변한다. Scheduler fairness와 tail latency, inverse mapping buffer capacity를 본다. Empty/small expert cases fallback도 representation/ownership이 맞아야 한다.

Data-parallel scale-out은 request distribution뿐 아니라 model weight replicas, KV capacity와 cache hit locality를 바꾼다. Sticky routing이 cache locality를 높이지만 imbalance를 만들 수 있다. 이 장에서는 policy 정답보다 request/KV owner가 명시되는지를 본다.

교육용 matrix가 작은 이유는 toy라서가 아니다. Rank-local 값과 collective 결과를 사람이 계산할 수 있어 semantic error를 즉시 드러내기 때문이다. Large Llama tensor에서도 같은 shard/partial equation이 반복된다. Dimension만 커지고 kernel/transport가 최적화될 뿐 owner 논리는 같다.

모든 equation에는 orientation convention을 적는다. Weight `[out,in]`, input `[M,in]`, output `[M,out]`을 사용한다. Framework가 다른 storage order/view를 쓰면 logical axes로 normalize한다. Row/column이라는 명칭이 matrix 표기 convention에 따라 혼동될 수 있다.

Final review는 각 numbered section의 주장과 source claim을 연결한다. Group construction claim, col/row parameter/forward claim, PP send/recv, DP routing/state, EP dispatch/inverse와 collective wrapper claim이 필요하다. Evidence gap은 빈칸으로 숨기지 않고 explicit gap으로 남겨 본문 단정을 좁힌다.

완료의 기준은 모든 collective가 빨리 끝나는 것이 아니다. Global reference와 rank-local ledger가 맞고, group member/order와 call sequence가 합의되며, collective 뒤 new owner가 next consumer expectation과 맞고, finish/abort 뒤 모든 dynamic owner가 닫혀야 한다. 이 기준을 만족한 뒤 transport 성능을 최적화한다.

운영에 적용할 때는 배포 전 manifest와 실행 중 observation을 분리한다. Manifest는 기대하는 mesh axes, ordered group members,
layer와 expert ranges, vocab ranges, sampler owner와 compatibility generations를 선언한다. Observation은 실제 process가 보고한
global/local rank, communicator member digest, loaded parameter ranges, active graph와 request references를 담는다. 두 표의 diff가
0이어야 admission을 연다. Config 파일끼리 같다는 사실은 runtime objects가 같다는 증거가 아니다.

첫 canary는 값이 드문 basis tensor를 사용한다. 각 TP shard에는 겹치지 않는 powers-of-ten marker를, 각 expert에는 global
expert id marker를, vocab shard에는 global token offset marker를 넣는다. 이 pattern은 gather order, reduction omission, expert
row swap과 sampler offset을 서로 다른 signature로 만든다. Random tensor 하나는 오류를 발견할 수는 있어도 어떤 owner
transition이 틀렸는지 설명하기 어렵다.

두 번째 canary는 lifecycle을 겨냥한다. Collective enqueue 뒤 request abort, PP send 뒤 receiver delay, EP dispatch 뒤 placement
reload, sampling commit 직전 owner failure를 deterministic event로 만든다. Expected terminal은 단순 process exit가 아니라
generation별 request, communicator handle, mailbox, dispatch buffer, graph와 RNG commit reference가0인 상태다. 이 수치를
운영 dashboard와 test assertion에서 같은 이름으로 사용하면 재현과 현장 증거를 연결하기 쉽다.

세 번째 canary는 empty와 uneven case다. Token counts가 `[4,0]`, `[3,1]`, `[0,4]`일 때 empty rank도 protocol에 필요한
collective sequence를 유지하는지 본다. Vocab10처럼 TP2에 균등한 case뿐 아니라 Vocab11의 padded/uneven range를 시험한다.
PP 마지막 microbatch가 한 token뿐일 때 metadata shape가 이전 큰 batch의 stale entry를 읽지 않는지도 확인한다. 평균적인
balanced batch만으로는 boundary bug를 드러내기 어렵다.

배포 승격은 semantic canary, lifecycle canary, performance canary 순이다. Semantic mismatch가 하나라도 있으면 빠른 backend를
비교하지 않는다. Lifecycle이 닫히지 않으면 repeated reload/abort에서 결국 leak 또는 stale generation이 된다. 두 단계가
통과한 뒤에만 NCCL algorithm, custom collective, overlap과 batch policy의 throughput·tail latency를 비교한다. 이 순서가
correctness와 performance 원인을 섞지 않게 한다.

현장 runbook의 첫 질문도 단순하다. “멈춘 rank들이 같은 logical collective에 참여했는가?” Group generation, ordered members,
sequence와 input tensor identity가 다르면 application branch를 찾는다. 모두 같고 enqueue가 확인됐는데 completion만 없다면
transport 층으로 내려간다. Wrong output에서는 local reference, post-collective placement, post-expert combine, sampling global-id
순으로 first divergence를 찾는다. 이 두 갈래를 첫 화면에 두면 긴 분산 로그를 목적 없이 훑지 않는다.

이 운영 절차는 source link를 살아 있는 점검 좌표로 만든다. Upgrade 때 vLLM rank reshape, SGLang coordinator, Transformers
placement strategy, NCCL enqueue와 MoE dispatch/combine의 producer-consumer 관계를 다시 확인한다. 함수명이 유지돼도 axis
order나 generation ownership이 바뀌면 manifest와 fixture를 갱신한다. 반대로 line number가 이동했어도 invariant가 같다면
같은 작은 수치로 계약을 재검증할 수 있다.

Runbook에는 판정 책임자도 적는다. Scheduler owner는 request와 DP replica, model runner owner는 TP/PP step, MoE owner는
placement와 permutation, sampler owner는 token commit, distributed coordinator는 group lifecycle terminal을 승인한다. 같은
generation 값을 여러 component가 보고한다는 사실만으로 책임이 닫히지는 않는다. 각 owner가 어떤 predicate와 counter를
terminal로 선언하는지 명시해야 partial cleanup을 다음 팀에 떠넘기지 않는다.

Evidence bundle의 마지막 페이지에는 정상 수치와 실패 수치를 나란히 둔다. Dense hidden reference, rank partials, MoE
eight-entry table, four combined outputs, vocab global token7, group/member digests와 zero-resource terminal을 한눈에 비교한다.
이 표가 있어야 독자는 소스 함수에서 발견한 state가 실제 request 값과 어떻게 연결되는지 확인하고, 다음 release에서도
동일한 사고 class를 짧은 fixture로 재검증할 수 있다.

## 55.14 소스 노트

이 근거 묶음은 topology 이름을 나열하기 위한 목록이 아니다. 먼저 parallel state에서 rank가 속한 TP·PP·DP·EP group과 collective domain을 확인하고, linear layer에서 weight shard 축과 partial output의 reduction 여부를 잇는다. 이어 model loader의 실제 device placement가 논리 group과 일치하는지 확인한다.

비교할 때는 같은 `world_size`만 맞추지 않는다. 각 tensor의 global shape, local slice 범위, producer rank, collective 종류와 consumer shape를 한 행에 기록한다. Group membership이 맞아도 shard axis가 다르면 collective는 정상 종료하면서 틀린 값을 만들 수 있다. 반대로 shape가 달라도 PP stage 사이 activation처럼 의도된 경계일 수 있으므로 source의 producer/consumer 계약으로 판정한다.

- [Transformers v5.15.1 — tensor parallel integration](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/integrations/tensor_parallel.py#L1-L220)
- [vLLM v0.27.1 — parallel state](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/distributed/parallel_state.py#L1-L260)
- [vLLM v0.27.1 — parallel linear layers](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/linear.py#L1-L260)
- [SGLang v0.5.18 — distributed parallel state](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/distributed/parallel_state.py#L1-L260)
- [llama.cpp v0.2.0 — model loader placement boundary](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model-loader.cpp#L1-L220)
