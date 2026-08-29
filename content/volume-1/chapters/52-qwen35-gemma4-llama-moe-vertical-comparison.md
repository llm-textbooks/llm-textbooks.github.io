# 52장. 같은 토큰, 다른 상태: Llama·Qwen3.5·Gemma4·MoE 수직 비교

모델 이름을 아는 것과 모델을 서빙할 수 있는 것은 다르다. 새 architecture의 config를 읽고도 모든 layer를 “attention 뒤 MLP”로만 그리면 cache 용량, rollback, graph capture와 backend 선택에서 잘못된 가정을 하게 된다. 이 장은 제품 사양을 나열하지 않는다. prompt 8 token인 두 요청이 prefill을 마치고 세 번 decode되는 동안 layer 하나가 무엇을 읽고 무엇을 쓰는지 같은 원장으로 비교한다.

설명용 좌표는 `H=16, Nq=4, Nkv=2, D=4`다. 실제 release의 크기를 섞지 않는다. Llama fixture는 모든 layer가 full attention과 dense MLP를 쓴다. Qwen3.5 fixture는 네 layer 가운데 세 layer가 Gated DeltaNet, 한 layer가 full attention이다. Gemma4 fixture는 sliding과 full attention을 번갈아 두며 같은 attention type의 이전 layer를 shared KV source로 가리킬 수 있게 한다. MoE fixture는 expert 네 개와 top-k 2를 쓴다. 네 모델 모두 residual row의 shape는 `[token,16]`이지만 persistent state와 실행 경로는 같지 않다.

## 52.1 모델 이름을 보기 전에 세 layer를 손으로 계산한다

첫 계산은 같은 residual row를 세 layer에 넣는다. Llama dense GQA에서는 Q/K/V와 full KV frontier를, Qwen3.5 hybrid layer에서는 convolution window와 recurrent state 갱신을, Gemma4 alternating layer에서는 sliding/full 선택과 shared-KV source를 적는다. 입력 row·dtype·token position은 같게 두고 **새로 생기는 state, 다음 decode가 읽는 state, rollback 때 함께 폐기할 generation** 세 열만 비교한다. 이 계산이 끝난 뒤에야 모델별 구현 카탈로그를 연다.

운영자는 두 요청 모두 prompt 길이 8이고 decode budget이 3이므로 같은 token budget을 예약했다. Llama는 정상인데 Qwen3.5의 첫 decode token부터 reference와 달라진다. Gemma4는 짧은 prompt에서는 맞지만 sliding window보다 긴 문맥에서만 달라진다. MoE 모델은 batch 순서를 바꾸면 특정 요청의 답이 다른 요청과 뒤섞인다. GPU memory 부족도 없고 tensor shape도 맞다. 이 세 증상을 “새 모델 kernel이 불안정하다”로 묶으면 조사 비용만 커진다.

먼저 모든 layer에 같은 state ledger를 쓴다.

```text
request / step / layer index / layer type
input residual shape와 row identity
읽은 persistent state와 logical frontier
쓴 persistent state와 commit frontier
mask·window·position 좌표
dense 또는 expert route와 token count
선택 backend와 fallback 이유
state byte·주소·lifecycle owner
```

Llama layer 0은 token 여덟 개의 K/V를 쓰고 첫 decode에서 아홉 번째 K/V를 append한다. Qwen3.5 GDN layer 0은 prefill chunk에서 convolution state와 recurrent state를 만들고 첫 decode에서 둘을 갱신한다. Gemma4 sliding layer 0은 window 밖 token을 읽지 않도록 mask와 resident state 범위를 제한한다. MoE layer는 attention state와 별개로 현재 token의 router logits, selected expert IDs, permutation과 combine weights를 temporary로 만든다. “past length 8”만 기록하면 서로 다른 상태를 같은 것으로 보게 된다.

### shape가 맞는 silent failure

Qwen3.5 사건에서 convolution state의 마지막 activation은 position 7까지 반영됐지만 recurrent matrix는 position 6에서 멈췄다고 하자. 두 tensor의 shape와 device는 정상이다. 첫 decode position 8은 새로운 input과 최신 conv state를 사용하면서 오래된 recurrent state를 읽는다. 출력은 유한하고 이후 step도 진행된다. 오류는 allocation이 아니라 두 frontier가 같은 commit에 속하지 않는 데 있다.

Gemma4 사건도 shape로 잡히지 않는다. sliding layer가 `[batch,heads,capacity,D]` cache를 받아도 capacity가 full layer와 같을 수 있다. 잘못된 mask가 window 이전 token까지 열어 주면 valid byte를 읽어 그럴듯한 attention output을 만든다. shared KV source가 같은 type의 직전 layer가 아니라 단순히 직전 layer를 가리켜도 head shape가 같으면 실행된다. 확인할 것은 크기가 아니라 source layer ID와 허용 key position 집합이다.

MoE 사건에서는 token sort가 `[r0t0,r1t0]`을 expert별 순서로 바꾼 뒤 inverse permutation이 잘못됐다. expert output shape는 여전히 `[2,H]`다. combine scatter가 두 행을 바꿔 놓으면 각 요청은 다른 요청의 expert 결과를 받는다. attention cache나 router top-k 자체는 정상일 수 있다. row identity를 residual ledger에 유지하지 않으면 마지막 logits에서야 문제가 드러난다.

### 최초 divergence를 정하는 비교점

세 사건은 동일한 비교 원칙으로 좁힌다. normalized residual, projection 또는 router logits, state readback, state mutation 뒤 frontier, backend output, residual add를 순서대로 비교한다. Qwen3.5는 prefill 마지막 state와 첫 decode가 읽은 state를 비교한다. Gemma4는 layer별 `(source_layer, allowed_key_range)`를 비교한다. MoE는 `(original_row, expert_id, sorted_slot, inverse_slot, combine_weight)`를 비교한다. 마지막 logits tolerance를 넓히지 않는다.

정상 ledger는 오류를 찾는 로그만이 아니다. scheduler가 request를 admit할 때 어느 state byte가 늘어나는지, prefix reuse가 어느 layer에서 가능한지, preemption이 어떤 state를 옮겨야 하는지 알려 준다. backend 등록 여부도 ledger가 요구하는 state ABI를 구현하는지로 판정한다. class import 성공은 이 계약을 보장하지 않는다.

이제 Llama를 기준 좌표로 고정한다. 기준이 단순해야 Qwen3.5, Gemma4와 MoE의 차이를 “더 복잡하다”가 아니라 어느 상태 전이가 추가되거나 바뀌었는지 말할 수 있다.

## 52.2 dense MLP와 MoE route를 같은 residual row에서 비교한다

attention 또는 GDN이 끝나면 residual row는 다시 `[rows,H]`다. dense MLP는 모든 row가 같은 up/gate/down weight를 통과한다. MoE는 먼저 router가 row별 expert score를 만들고 top-k expert를 고른 뒤, row를 expert별 작업 묶음으로 재배열해 GEMM하고 원래 순서로 합친다. 입력과 최종 출력 shape가 같다는 사실이 중간 상태의 차이를 가린다.

canonical fixture에는 두 request의 decode row `A`,`B`, expert 네 개 `E0..E3`, top-k 2를 둔다. router softmax 이전 logits를 다음처럼 고정한다.

```text
A: [4.0, 3.0, 1.0, 0.0] -> E0, E1
B: [0.0, 2.0, 5.0, 4.0] -> E2, E3
```

top-k weight를 정규화하는 policy라면 선택된 두 값 사이에서 다시 정규화한다. 정규화하지 않는 architecture도 있으므로 이름만 보고 적용하지 않는다. route ledger 한 행은 `(original_row, expert_id, topk_slot, routing_weight, sorted_slot, local_expert, output_slot)`을 가진다. 이 일곱 좌표가 expert GEMM보다 먼저 닫혀야 한다.

### router에서 combine까지 다섯 경계

첫 경계는 router logits다. logits dtype, bias, jitter 또는 normalization이 달라지면 selected expert가 바뀐다. 두 번째는 top-k selection과 weight다. tie-breaking과 expert ID order가 deterministic한지 본다. 세 번째는 dispatch다. 선택 쌍 네 개를 expert별로 sort하고 alignment를 위해 padding할 수 있다. 네 번째는 expert GEMM이다. local expert weight가 올바른 global expert에 대응해야 한다. 다섯 번째는 combine scatter다. expert output을 original row와 top-k weight에 맞춰 더한다.

vLLM의 [`Mixtral router·FusedMoE와 decoder`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/mixtral.py#L69-L290)는 작은 canonical route를 읽기 좋은 고정 좌표다. Mixtral에서 확인한 argument와 mapping을 Qwen3.5 또는 Gemma4 MoE의 binary ABI로 복사하지 않는다. expert weight 이름, shared expert, normalization, grouping, routed scaling과 quantization scheme이 달라질 수 있다. 공통인 것은 route→dispatch→expert compute→combine이라는 semantic ledger다.

### expert constant로 permutation을 검증한다

각 expert가 입력과 무관하게 `E0=[10,10,...]`, `E1=[20,...]`, `E2=[30,...]`, `E3=[40,...]`을 출력한다고 하자. routing weight를 모두 0.5로 단순화하면 A 결과는 15, B 결과는 35여야 한다. sorted buffer가 `[A:E0,A:E1,B:E2,B:E3]`에서 다른 순서가 되어도 inverse scatter가 정확하면 결과는 같다.

combine 뒤 A가 35, B가 15라면 router나 expert GEMM을 조사할 필요가 없다. expert output과 sorted slot까지 정상이고 original row scatter에서 처음 갈라졌기 때문이다. A가 20이라면 한 route가 중복됐거나 weight가 적용되지 않았을 수 있다. sentinel은 최종 logits보다 훨씬 구체적인 owner를 가리킨다.

padding row는 output에 기여하면 안 된다. fused kernel이 expert별 token 수를 tile 배수로 맞추기 위해 dummy slot을 넣을 때 valid count와 mask가 필요하다. padding의 original row sentinel이 0으로 초기화되어 request 0에 scatter되면 shape는 맞지만 A 결과에 쓰레기 값이 더해진다. dispatch metadata와 combine kernel이 같은 padded capacity와 valid count를 해석해야 한다.

### EP와 TP는 서로 다른 소유권을 나눈다

TP는 expert 내부 weight의 input/output dimension을 나눌 수 있다. EP는 global expert 집합을 rank별 local expert로 나눈다. global expert 3이 rank 1의 local expert 0일 수 있다. router가 낸 global ID, all-to-all dispatch destination, local packed weight index와 combine source rank를 연결해야 한다.

expert 네 개, EP 2라면 rank 0이 E0/E1, rank 1이 E2/E3를 소유한다고 하자. A route는 rank 0에만 가고 B route는 rank 1에만 간다. 다른 batch에서 모든 token이 E0/E1을 고르면 rank 0만 바빠진다. total token 수가 같아도 communication과 GEMM 시간이 달라진다. dense MLP에는 이런 expert skew가 없다.

TP와 EP를 함께 쓰면 route payload가 remote rank로 이동한 뒤 local TP collective가 필요할 수 있다. scheduler의 token budget은 이 topology 비용을 직접 나타내지 못한다. admission 전에 정확한 router 결과를 알 수 없으므로 최근 expert distribution과 conservative workspace를 사용하되 correctness capacity와 성능 추정을 섞지 않는다.

### quantized expert와 loader mapping

quantized MoE에서는 expert 하나가 qweight, scale, zero point 또는 group metadata 여러 buffer를 가진다. global expert→local expert mapping이 qweight에만 적용되고 scale order가 global 그대로면 output이 틀린다. loaded parameter name이 존재한다는 검사는 expert slice coverage를 보장하지 않는다. 49장의 source→destination edge를 expert component별로 만든다.

shared expert가 있는 architecture는 routed top-k 결과에 별도 dense/shared branch를 더할 수 있다. shared output이 combine 전인지 후인지, gate가 있는지 model source로 확인한다. Mixtral fixture에 shared expert가 없다는 이유로 다른 model에서도 없다고 가정하지 않는다. 이 장의 비교에서 MoE라는 단어는 하나의 ABI가 아니라 동적 row ownership이 추가되는 family를 뜻한다.

### dense와 MoE의 관측 차이

dense layer는 `rows`, matrix shape, dtype와 kernel family로 작업량을 설명하기 쉽다. MoE layer는 여기에 selected pairs, active experts, max/mean tokens per expert, padding ratio, remote dispatch bytes와 combine 시간을 더해야 한다. expert ID를 metric label로 무제한 노출하면 cardinality가 커지므로 model instance와 layer group은 bounded label로 두고 expert distribution은 histogram 또는 top imbalance 값으로 집계한다.

GPU utilization 하나가 높아도 route가 균형 잡혔다는 뜻은 아니다. 한 rank는 compute로 높고 다른 rank는 collective wait로 낮을 수 있다. layer별 dispatch/GEMM/combine 시간을 분리하고 rank별 active token을 함께 본다. 이제 이 차이를 KV·conv·recurrent·shared-KV 상태의 byte와 같은 원장에 합친다.

## 52.3 KV·conv·recurrent·shared-KV는 byte 식과 수명이 다르다

서빙 용량을 모델 parameter 크기와 max sequence length만으로 잡으면 hybrid architecture에서 틀린다. persistent state를 네 family로 나누고 layer plan을 따라 unique allocation을 합해야 한다. 같은 request가 소유해도 growth axis, rollback 단위, prefix reuse와 eviction 비용이 다르다.

### full KV는 token frontier를 따라 자란다

full attention KV의 logical byte는 대략 `2 × L × Nkv_local × D × element_bytes`다. TP replication과 page padding, quant scale metadata는 별도다. 새 accepted token마다 K/V row가 append된다. prefix block을 공유할 수 있고 append는 tail copy-on-write를 요구할 수 있다. preemption 시 block table과 payload를 복원하면 된다.

fixture에서 request 하나, layer 하나, BF16, `L=11,Nkv=2,D=4`면 352 bytes다. batch 두 개와 full layer 수를 곱하되 shared source가 있으면 unique allocation만 센다. 실제 allocator는 16-token page 하나를 잡아 512 bytes 이상을 예약할 수 있다. logical live byte, reserved byte, resident byte를 구분한다.

### sliding KV는 logical range와 physical capacity가 갈릴 수 있다

window `W=4`인 sliding attention은 logical read set이 최근 네 token이다. ring buffer라면 state byte가 `2×W×Nkv×D×bytes`로 bounded될 수 있다. full backing과 mask 방식이면 reserved byte는 L에 따라 늘어날 수 있다. 두 구현 모두 semantic output은 같아야 하지만 rollback과 prefix reuse 비용이 다르다.

ring overwrite는 commit 전 mutation에 주의한다. position 8이 position 4의 physical slot을 덮고 speculative reject되면 old position 4가 다음 window에 필요 없는지 계산해야 한다. window상 이미 빠졌다면 복원하지 않아도 될 수 있지만 여러 draft token과 consumer frontier가 다르면 안전 조건이 달라진다. state owner가 accepted frontier와 overwrite generation을 기록해야 한다.

### convolution state는 kernel width가 수명을 정한다

convolution state는 최근 activation의 제한된 history를 보존한다. shape는 channel과 kernel width에 의해 정해지고 prompt 전체 token 수에 선형으로 늘지 않는다. decode마다 shift 또는 circular index mutation이 일어난다. rollback은 token block을 free하는 문제가 아니라 이전 lanes 또는 재계산 가능한 prefix state를 복원하는 문제다.

state byte가 작아도 update frequency는 매 token이다. 여러 request를 continuous batch로 모으면 request별 conv pointer와 logical cursor를 gather해야 한다. batch compaction 뒤 row index가 바뀌어도 state ownership은 request ID에 남아야 한다. dense tensor batch axis를 그대로 state slot ID로 쓰면 request 제거 뒤 다른 request가 이전 state를 읽을 수 있다.

### recurrent state는 matrix update를 보존한다

DeltaNet 계열 recurrent state는 head별 matrix 또는 accumulator family다. byte는 sequence length보다 head와 key/value dimension에 좌우된다. prompt가 길어도 bounded될 수 있지만 state update가 이전 모든 token의 요약이므로 한 byte corruption의 영향은 이후 모든 decode로 이어진다. KV 한 token block만 다시 계산하는 것과 복구 단위가 다르다.

prefix reuse는 terminal recurrent state snapshot을 요구한다. prefix 중간에서 분기하려면 그 frontier의 state가 있어야 한다. 모든 token frontier snapshot을 저장하면 bounded state의 장점이 사라질 수 있으므로 checkpoint 간격과 recompute trade-off가 생긴다. scheduler는 reuse hit를 KV block hit와 같은 숫자로만 표시하지 않는다.

### shared KV는 byte를 줄이고 dependency를 늘린다

Gemma4 shared KV에서 consumer layer는 source allocation을 참조한다. physical byte는 별도 K/V를 만들 때보다 줄지만 dependency edge와 reference count가 생긴다. source layer가 sliding인지 full인지에 따라 capacity 식이 달라진다. consumer layer index만 보고 state byte를 합산하면 중복 계산한다.

source를 evict하려면 모든 consumer의 execution과 prefix reference가 끝났는지 확인한다. graph가 source pointer를 capture했다면 buffer 재배치도 graph invalidation 조건이다. state sharing은 단순 메모리 최적화가 아니라 lifetime topology 변경이다.

### 하나의 용량표로 손계산한다

fixture request 하나의 committed length가 11이라고 하자. Llama 네 full layer는 logical KV `4×352=1408 bytes`다. Qwen3.5는 full layer 하나의 352 bytes에 GDN 세 layer의 conv/recurrent 고정 state를 더한다. conv가 layer당 설명용 64 bytes, recurrent가 128 bytes라면 `352+3×192=928 bytes`다. 이 숫자는 실제 제품 사양이 아니라 서로 다른 growth law를 보여 주는 계산이다.

Gemma4가 full source 하나와 sliding source 하나를 두 consumer가 공유한다면 unique KV는 full 352와 sliding 128, 합계 480 bytes다. 공유하지 않고 네 layer가 각자 allocation하면 두 full과 두 sliding로 960 bytes다. MoE route temporary는 request lifetime state가 아니므로 이 persistent 합계에 넣지 않지만 peak workspace에는 넣는다.

| state | growth axis | mutation | rollback | prefix reuse key |
|---|---|---|---|---|
| full KV | token length | append | tail crop/free | token block·position |
| sliding KV | window 또는 backing length | append/overwrite | crop 또는 slot restore | window frontier·slot generation |
| conv | kernel width·channel | shift/circular write | lane restore/recompute | terminal conv state |
| recurrent | head matrix dims | in-place matrix update | snapshot/recompute | terminal recurrent state |
| shared KV | source state와 consumer edge | source mutation | source policy 상속 | source layer·generation |

이 표는 allocator를 하나로 만들지 말라는 뜻이 아니다. 공통 interface 아래 서로 다른 state spec과 rollback method를 제공할 수 있다. 위험한 것은 모든 state에 `num_blocks`와 `crop(length)`가 같은 의미라고 가정하는 것이다.

실제 capacity planner는 이 표를 layer plan에 적용해 request 하나의 증분을 계산한다. `reserve(request, tokens=3)`이라는 호출이 들어오면 full KV layer에는 세 token의 tail block 여유를, ring sliding layer에는 overwrite 가능한 세 slot 또는 undo 여유를, GDN layer에는 기존 state를 안전하게 갱신할 temporary나 snapshot을 요구한다. shared consumer layer에는 payload를 중복 예약하지 않지만 source reference metadata를 확보한다. MoE route workspace는 request가 아니라 실행 batch와 top-k pair 수에 따라 예약한다. 같은 API가 이 차이를 숨겨도 내부 state spec은 보존해야 한다.

두 request를 한 decode batch로 합칠 때도 단순히 request당 byte를 더하는 것만으로 peak를 설명하지 못한다. Llama attention workspace, Qwen3.5 recurrent temporary, Gemma4 mixed backend workspace와 MoE dispatch buffer가 layer 순서에 따라 겹치거나 재사용될 수 있다. 서로 동시에 살아 있지 않은 workspace는 최대값으로 잡을 수 있지만 asynchronous stream이나 overlap이 있으면 lifetime이 겹친다. planner는 이름이 아니라 allocation interval을 본다. graph capture가 workspace 주소를 고정하면 다음 layer가 같은 buffer를 재사용할 수 있는지도 capture contract에 달렸다.

수명 원장은 `allocate`, `first_write`, `last_read`, `commit`, `release` 다섯 시점을 둔다. full KV page는 prefix가 공유되면 request가 끝나도 다른 owner 때문에 release되지 않을 수 있다. convolution temporary는 layer call이 끝나면 사라지지만 committed conv state는 request와 함께 남는다. recurrent old snapshot은 commit 또는 rollback 가능 시점이 지나면 해제한다. shared KV source는 마지막 consumer의 last read 뒤에만 해제한다. MoE sorted buffer는 combine이 완료되고 비동기 collective가 참조하지 않을 때 해제한다.

이 원장을 메모리 metric과 맞출 때 allocator reserved와 실제 resident를 혼동하지 않는다. CUDA allocator가 큰 segment를 보유하면 logical state가 해제돼도 process reserved byte는 즉시 줄지 않을 수 있다. 따라서 “request 종료 후 GPU memory가 줄지 않았다”만으로 state leak을 선언하지 않는다. state owner registry에 reference가 남았는지, reusable free block으로 돌아왔는지, allocator segment가 cache된 것인지 차례로 구분한다. 반대로 process memory가 일정해도 잘못된 request가 free block을 참조하면 use-after-free semantic bug가 존재할 수 있다.

rollback 손계산도 한다. speculative width가 3이고 한 token만 accept됐다고 하자. full KV는 두 tail row를 invalid로 만들고 block의 accepted count를 조정한다. sliding ring이 세 slot을 덮었다면 reject된 두 write만 무효화하는 것으로 충분한지, 그 전에 살아 있던 old positions가 다음 allowed window에 필요한지 확인한다.

GDN conv/recurrent는 generation 9만 남기고 10·11 update를 되돌리거나 generation 8에서 accepted token 하나를 재실행한다. shared KV consumer는 source rollback 결과를 따르되 captured generation을 갱신한다. MoE route temporary는 rejected step 결과를 버리지만 accepted token의 residual과 다음 router input은 generation 9여야 한다.

재계산 전략은 state별로 비용이 다르다. KV tail 두 row는 projection을 다시 수행해 복원할 수 있다. recurrent generation 8 snapshot이 없으면 prefix 0~8 전체를 scan해야 할 수 있다. ring buffer의 old payload를 잃었다면 window 크기만큼 재계산하면 될 수 있다. 비용 정책은 이 차이를 이용하지만 correctness는 항상 같은 accepted prefix state를 복원해야 한다. “rollback이 비싸다”는 이유로 stale state를 계속 쓰는 선택지는 없다.

### state metric은 세 byte를 분리한다

logical byte는 model semantics가 요구하는 유효 payload다. reserved byte는 allocator page, alignment와 workspace를 포함한다. resident byte는 현재 GPU에 있는 부분이다. offload나 eviction이 있으면 세 값이 갈린다. layer type별로 이 세 값을 집계하면 “Qwen3.5가 KV를 덜 쓴다” 같은 거친 문장을 실제 capacity 효과로 바꿀 수 있다.

state mutation 실패도 counter 하나로 묶지 않는다. allocation failure, stale frontier, rollback failure, shared source missing, backend unsupported를 분리한다. request ID나 prompt text를 metric label로 넣지 않고 trace나 bounded diagnostic sample에 둔다.

## 52.4 backend capability와 scheduler는 layer plan을 함께 읽어야 한다

model registration, Python class import, weight load 성공, optimized backend 지원은 네 개의 다른 관문이다. class가 존재해도 GDN recurrent state ABI가 runner에 없을 수 있다. full attention kernel이 있어도 sliding window와 shared KV source를 못 받을 수 있다. FusedMoE가 있어도 현재 quant scheme이나 EP topology를 지원하지 않을 수 있다. 지원하지 않는 조합은 의미 보존 fallback 또는 명시적 reject로 닫는다.

### capability matrix는 model 이름보다 구체적이어야 한다

capability key에는 operation family, dtype, head/key/value dimension, mask/window, state layout, quantization, TP/EP degree, graph mode와 platform을 넣는다. 모든 조합을 metric label로 노출하라는 뜻은 아니다. 초기화 때 선택 결과와 reject 이유를 bounded enum으로 기록한다.

Llama attention은 paged GQA decode backend가 지원할 수 있다. Qwen3.5 full layer는 같은 family를 쓸 수 있어도 GDN layer는 recurrent/chunk backend가 따로 필요하다. Gemma4 sliding layer는 window mask와 shared source address를 wrapper가 표현해야 한다. MoE는 top-k, expert packing, quant format과 EP dispatch를 맞춰야 한다. “CUDA backend 사용”이라는 한 줄은 아무것도 증명하지 않는다.

eager reference는 correctness oracle 역할을 한다. 다만 eager가 모든 state lifecycle을 자동으로 구현하는 것은 아니다. optimized op만 빠지고 cache update나 layer plan은 같은 의미를 유지해야 한다. fallback에서 state를 dense full-attention KV로 재해석하거나 MoE를 모든 expert 평균으로 바꾸면 느린 reference가 아니라 다른 model이다.

### token budget 하나로 admission하지 못하는 이유

두 request가 각각 11 token이어도 Llama full KV, Qwen3.5 bounded recurrent state, Gemma4 shared/sliding KV의 추가 byte가 다르다. scheduler는 layer state spec으로 incremental byte를 계산해야 한다. prompt prefill 중 temporary workspace와 decode persistent growth도 구분한다. GDN chunk size가 workspace peak를 바꾸고 MoE expert skew가 dispatch buffer와 longest expert GEMM을 바꾼다.

admission correctness는 최악의 필요한 state를 확보하는 문제다. 성능 scheduling은 어떤 kernel family와 batch shape를 함께 묶을지의 문제다. 둘을 하나의 heuristic score로 숨기면 OOM을 피하려고 지나치게 보수적이 되거나 throughput을 높이려다 rollback 공간을 잃는다.

### continuous batch에서 row와 state slot을 분리한다

step마다 finished request를 제거하고 새 request를 넣으면 compute batch row가 바뀐다. KV는 block table이 request identity를 보존한다. conv/recurrent state도 별도 slot map이 필요하다. MoE dispatch의 original row는 현재 step의 batch row이고 persistent request slot과 동일하지 않을 수 있다.

ledger에는 `request_id→state_slot`, `request_id→current_batch_row`, `sorted_route_slot→current_batch_row`를 분리한다. compaction 뒤 stale batch row를 state slot로 쓰는 사건은 Qwen3.5에서 다른 request recurrent state를 읽게 하고 MoE에서는 combine row를 바꾼다. shape가 같아 탐지하기 어렵다.

### graph bucket은 batch size만으로 결정되지 않는다

CUDA graph는 stable addresses와 control path를 요구한다. Llama decode는 batch/token bucket과 backend metadata shape로 capture할 수 있다. Qwen3.5는 full/GDN layer sequence와 recurrent state pointer table, chunk/decode branch가 graph 계약에 들어간다. Gemma4는 sliding/full wrapper와 shared source pointer가 고정되어야 한다. MoE는 token별 route가 동적이므로 padded expert capacity나 specialized graph policy가 필요할 수 있다.

같은 batch size 8이라도 active expert 분포가 다르면 expert별 rows가 달라진다. graph가 최대 capacity buffer를 capture하면 padding ratio와 reserved workspace가 늘어난다. eager fallback이 섞이면 launch 수가 달라진다. graph hit rate는 model 전체 한 숫자보다 layer family와 bucket reason별로 본다.

### preemption 비용도 state family마다 다르다

Llama KV block은 offload하거나 recompute할 수 있다. sliding ring은 작지만 overwrite generation을 보존해야 한다. conv/recurrent terminal state는 byte가 작아 이동하기 쉬울 수 있지만 없으면 prefix 전체를 scan해야 한다. shared KV source 하나를 옮기면 여러 consumer dependency를 함께 갱신한다. MoE route temporary는 step이 끝나면 버리지만 distributed expert weights는 model lifetime이라 request preemption 대상이 아니다.

scheduler는 victim 선택 때 freed resident byte뿐 아니라 restore/recompute cost를 본다. full KV가 크다고 항상 가장 나쁜 victim은 아니다. terminal recurrent snapshot이 없고 prompt가 길면 작은 state를 잃는 recompute 비용이 더 클 수 있다. 비용 추정은 correctness owner와 분리된 정책이며 틀려도 output 의미를 바꾸면 안 된다.

### 관측성은 bounded label과 trace를 나눈다

metric에는 layer family별 execution count와 latency, backend/fallback enum, logical/reserved/resident state bytes, graph hit/fallback, MoE active expert count·padding ratio·skew를 둔다. exact layer index를 모든 metric label에 넣으면 model depth만큼 series가 늘 수 있으므로 layer type과 backend family로 집계하고, 문제 layer index는 sampled trace에 남긴다.

Qwen3.5 stale incident에는 conv/recurrent frontier mismatch counter가 직접적이다. Gemma4에는 shared source generation mismatch와 window-range violation이 유용하다. MoE에는 dispatch pairs, valid/padded rows, combine permutation validation failure가 유용하다. GPU utilization은 이 원인 metric을 대체하지 않는다.

운영 dashboard는 요청량, scheduler, state, backend, device 네 층을 같은 시간축에 놓는다. 요청량에는 admitted/running/preempted와 prompt/decode token을 둔다. scheduler에는 queue wait, state reservation failure, graph bucket과 preemption 이유를 둔다. state에는 family별 logical/reserved/resident byte와 rollback/reuse를 둔다. backend에는 layer family별 selected/fallback과 execution latency를 둔다. device에는 compute, memory bandwidth, collective와 allocation pressure를 둔다. 상관관계가 보인다고 인과를 확정하지 않고 trace의 state ledger로 최초 전이를 확인한다.

예를 들어 Qwen3.5 ITL이 튀는 시점에 GPU utilization이 낮고 graph fallback이 늘었다고 하자. 원인은 recurrent state pointer table의 dynamic shape 때문에 capture bucket을 벗어난 것일 수 있다. 그러나 같은 시점에 request compaction이 늘었다면 state gather 비용이나 stale-slot 보호 copy가 원인일 수 있다. fallback enum, pointer-table rows, gather bytes와 layer latency를 함께 봐야 한다. 단순히 graph bucket을 더 늘리면 reserved memory만 커질 수 있다.

Gemma4에서는 context가 window를 넘을 때 sliding layer resident byte가 계속 증가한다면 full backing 정책일 수도 있고 eviction이 실패했을 수도 있다. cache spec의 expected physical policy와 비교한다. full backing이 의도됐다면 mask correctness가 우선이고 증가 자체는 leak이 아니다. ring policy인데 증가한다면 source sharing reference가 old slots를 붙잡는지 본다. shared source consumer count와 oldest live generation을 trace에 남기면 구분할 수 있다.

MoE에서는 average expert tokens가 균등해도 tail latency가 나쁠 수 있다. 같은 layer의 한 step 안에서 max expert tokens와 rank별 remote pairs가 중요하다. histogram 평균만 보면 순간 skew를 숨긴다. p95 padding ratio, max/mean load ratio, all-to-all wait와 combine latency를 함께 본다. expert ID별 series 대신 imbalance scalar와 sampled offending distribution을 trace event로 남겨 cardinality를 제한한다.

alert는 증상이 아니라 invariant에 가깝게 만든다. conv/recurrent frontier가 다르면 즉시 correctness alert다. shared source generation이 consumer보다 오래되면 alert다. combine valid rows가 input selected pairs와 맞지 않으면 alert다. backend fallback 비율 상승은 성능 alert이며 output 의미가 보존되는 한 요청을 실패시킬 이유는 아니다. reserved memory 증가도 threshold와 지속 시간을 보고 capacity alert로 다룬다. severity가 다른 사건을 한 “model unhealthy” 신호로 합치지 않는다.

배포 전에는 capability manifest를 생성한다. model revision과 layer plan hash, effective dtype/quantization, TP/EP, attention/GDN/MoE backend, graph 지원 bucket, state spec version을 묶는다. 각 worker가 같은 manifest hash를 확인한 뒤 serving registry에 publish한다. rank 하나가 다른 fallback을 선택하면 distributed collective 순서와 output이 달라질 수 있으므로 local 성공만으로 ready가 아니다. 불일치 이유를 구체적으로 노출하고 전체 group을 publish하지 않는다.

rolling upgrade도 manifest generation을 사용한다. old worker와 new worker가 같은 model 이름을 제공해도 state serialization이나 prefix cache key가 다를 수 있다. recurrent snapshot 또는 shared KV block을 세대 사이에 재사용하려면 명시적 compatibility가 필요하다. 그렇지 않으면 traffic drain 뒤 state를 새로 만든다. model weight checksum만 같다고 state ABI가 같다고 결론내리지 않는다.

## 52.5 세 first-divergence 실험으로 경쟁 가설을 제거한다

사건을 재현할 때 실제 거대 model 전체를 첫 도구로 삼지 않는다. 동일 state ledger를 유지한 작은 fixture가 source selection, state mutation과 permutation을 더 빨리 가른다. 세 실험 모두 증상, 정상 원장, 첫 차이, 반증, 수정, 재발 fixture 순서로 닫는다.

### 실험 A: Qwen3.5 prefill 마지막 state와 첫 decode

증상은 prompt logits가 reference와 같지만 첫 decode부터 다르고 이후 차이가 커지는 것이다. GDN layer 하나, conv width 3, recurrent head 하나의 작은 fixture를 만든다. prefill input을 position별 distinct basis로 두어 terminal conv lanes와 recurrent matrix를 손으로 계산한다.

정상 원장은 prefill 입력 frontier 0, chunk 출력 frontier 8, exported conv/recurrent generation 8을 기록한다. 첫 decode는 position 8과 generation 8 두 state를 읽고 generation 9를 commit해야 한다. 실제 ledger에서 conv는 8, recurrent는 7이면 첫 divergence는 prefill state export 또는 cache container update다.

attention mask, tokenizer와 later MLP는 반증된다. prefill layer output과 full attention layer output이 같고 GDN recurrent read에서 처음 다르기 때문이다. conv state까지 같으므로 convolution kernel 가설도 제외한다. optimized GDN을 eager로 바꿔 문제가 사라져도 backend math인지 wrapper mutation인지 구분하려면 eager/optimized 양쪽의 state readback을 비교한다.

수정은 두 state를 한 generation transaction으로 publish하고 decode가 동일 generation만 받게 한다. failure injection으로 conv update 뒤 recurrent update 전에 예외를 내고 retry 결과가 clean run과 같은지 본다. speculative 3 token 적용 뒤 1 token accept와 request batch compaction도 재발 fixture에 넣는다.

### 실험 B: Gemma4 source layer와 allowed key range

증상은 window 이하 context에서는 맞고 `W+1`에서 다르며 특정 alternating layer에서만 차이가 난다. 네 layer, window 4, position별 K sentinel을 만든다. layer 0과 2는 sliding, 1과 3은 full이며 2가 0을, 3이 1을 source로 쓰게 한다.

position 4 decode에서 sliding allowed set은 `{1,2,3,4}`, full은 `{0,1,2,3,4}`다. source layer ID와 physical slots를 dump하지 않고 bounded ledger로 기록한다. layer 2가 source 1을 선택했다면 source relation에서 처음 다르다. source 0이지만 position 0 기여가 있으면 window mask 또는 slot mapping이 첫 차이다.

KV projection weight와 RoPE는 source payload를 직접 비교해 반증한다. source layer 0 cache bytes가 reference와 같기 때문이다. full layer output이 맞으면 공통 attention scale과 tokenizer도 제외된다. eager sliding은 맞고 optimized만 틀릴 때 wrapper가 전달한 window와 kernel이 해석한 inclusive/exclusive bound를 비교한다.

수정 뒤 prompt lengths `W-1,W,W+1,2W+1`, speculative overwrite/reject, prefix cache hit와 shared source eviction을 검사한다. graph capture와 eager 양쪽에서 source pointer generation이 같아야 한다.

### 실험 C: MoE sort와 inverse combine

증상은 batch 1에서는 맞고 batch 2에서 request 순서를 바꾸면 output 행이 바뀌는 것이다. expert constant fixture를 사용한다. router logits, top-k IDs와 weights, sorted slot, expert output, inverse scatter를 각각 저장한다.

router 결과가 reference와 같고 expert constant output도 맞지만 combine 뒤 A/B가 바뀌면 inverse permutation owner가 첫 divergence다. quantization, expert GEMM과 network precision은 반증된다. EP를 끄면 문제가 사라져도 all-to-all 자체보다 global original-row ID가 rank-local row로 잘못 축약됐는지 확인한다.

수정은 route item에 request/step 범위의 stable row identity를 포함하고 dispatch permutation과 inverse를 같은 plan object에서 만들게 한다. padding slot은 invalid identity를 갖고 scatter 전에 제거한다. 재발 fixture에는 같은 expert로 몰림, 빈 expert, top-k tie, EP rank 교환, request cancellation로 batch row가 압축되는 경우를 넣는다.

### 실험 D: 등록됐지만 capability가 없는 model

네 번째 사건은 수치 오류 이전에 막아야 한다. model class와 weights는 load됐지만 selected backend가 GDN state layout 또는 Gemma4 shared KV를 지원하지 않는다고 하자. 잘못된 구현은 generic attention으로 계속 실행한다. 올바른 경계는 initialization capability check에서 eager semantic reference를 선택하거나 unsupported 조합을 구체적으로 거부하는 것이다.

fixture는 backend capability bit 하나를 의도적으로 끈다. expected 결과는 fallback enum과 같은 reference output 또는 deterministic reject다. server가 ready를 publish한 뒤 첫 request에서 obscure shape error를 내면 gate가 너무 늦다. fallback이 selected됐지만 optimized kernel trace가 남으면 effective path 기록이 틀렸다.

이 실험은 support matrix를 문서 표로 끝내지 않는다. 각 supported 조합은 작은 semantic fixture, unsupported 조합은 reject fixture를 가진다. dependency version이 바뀌면 import test만이 아니라 capability probe와 reference comparison을 다시 수행한다.

### 공통 조사 노트

모든 실험에서 model revision, framework commit, effective layer plan, dtype, parallel topology, backend와 graph mode를 기록한다. “Qwen3.5 최신” 같은 이름은 재현 identity가 아니다. state checksum은 mutation 전후와 logical frontier를 함께 적는다. 같은 bytes라도 frontier가 다르면 같은 state가 아니다.

관찰 hook이 mutation timing을 바꾸지 않게 한다. 거대한 tensor를 CPU로 복사하기보다 sentinel slice, shape/stride, generation과 bounded checksum을 쓴다. concurrency를 끄면 문제가 사라져도 race라고 바로 결론내리지 않는다. deterministic ordering이 stale overwrite를 숨겼을 가능성을 fixture 순서 변경으로 가른다.

세 실험을 CI에 넣을 때 framework 전체 server를 매번 띄울 필요는 없다. layer plan constructor, state container, backend wrapper와 route combiner를 작은 deterministic input으로 연결한다. source reference와 optimized backend가 모두 있으면 동일 ledger checkpoint를 비교한다. GPU가 없는 검사에서는 plan, shape, mapping과 rejection gate를 닫고, GPU job에서는 numerical output과 mutation ordering을 추가한다. 검사 범위를 명시해 CPU test 통과를 CUDA backend 증거로 과장하지 않는다.

fixture artifact에는 예상 tensor 전체보다 계산 가능한 규칙을 저장한다. Qwen3.5 state는 position별 basis가 어떤 lane과 matrix 항을 바꾸는지 식으로 둔다. Gemma4 K/V는 `100×layer+position` 규칙과 expected allowed set을 둔다. MoE expert는 constant output과 expected inverse permutation을 둔다. 구현이 dtype나 layout을 바꿔도 semantic 기대값을 재생성할 수 있다.

비결정성이 허용되는 지점도 좁힌다. top-k tie가 동일 점수일 때 expert order가 implementation별로 다를 수 있다면 fixture는 tie를 피하거나 허용 집합을 명시한다. floating reduction order 차이는 tolerance를 둘 수 있지만 row identity, selected source layer와 frontier에는 tolerance가 없다. semantic 좌표 오류를 numerical tolerance로 덮지 않는다.

failure injection은 cleanup까지 검사한다. Qwen3.5 recurrent update 중 실패하면 conv state와 generation이 원복되고 pending graph/stream 작업이 정리되는지 본다. Gemma4 source eviction 직전 실패하면 reference count가 음수나 dangling이 되지 않는지 본다. MoE all-to-all 일부 rank 실패 시 다른 ranks가 combine을 기다리며 hang하지 않고 batch를 abort하는지 본다. 부분 결과 model/request를 serving output으로 publish하지 않는다.

재시도는 깨끗한 state generation에서 시작한다. failed attempt의 destination state 일부를 덮어쓰는 방식은 누락된 lane이나 expert row에 이전 byte를 남길 수 있다. 새 request state를 만들거나 모든 family가 명시적 reset barrier를 통과한다. distributed attempt ID를 바꾸고 ranks가 같은 generation을 확인한다. 이 규칙은 loader의 atomic publish와 같은 형태지만 대상이 request-lifetime state라는 점이 다르다.

성능 fixture와 correctness fixture도 분리한다. sentinel과 checksum hook은 최적화 path를 방해할 수 있다. correctness가 닫힌 뒤 production-like input에서 TTFT, ITL, goodput, state peak와 skew를 잰다. 성능 회귀가 나오면 같은 ledger의 selected backend, workspace와 state bytes를 비교한다. 결과 숫자만 비교해 새로운 architecture가 느리다고 결론내리지 않는다.

## 52.6 새 architecture는 state ledger를 역방향으로 읽는다

새 모델을 만났을 때 모델 이름부터 검색해 feature list를 만들지 않는다. serving 결과가 정확해지려면 persistent state의 의미, layer materialization, backend capability와 scheduler ownership이 연결되어야 한다. 읽는 순서는 config에서 시작하지만 검증 질문은 최종 consumer에서 역방향으로 온다.

조사의 첫 한 시간에는 layer plan을 닫는다.

config에서 hidden size와 head 수만 뽑지 않는다. layer type sequence, full/sliding interval, GDN/SSM flag, expert 수와 top-k, shared source rule을 canonical plan으로 직렬화한다. 각 layer가 attention, recurrent operator, dense MLP 또는 MoE 가운데 무엇을 materialize하는지 constructor와 대조한다.

plan hash를 loader, state allocator, forward runner와 coverage validator가 같이 쓰는지 본다. 한 모듈이 stale config를 사용하면 내부 두 consumer가 우연히 합의해도 실행 semantics가 어긋날 수 있다. model class registration은 이 단계의 시작이지 완료가 아니다.

그다음 한 시간에는 한 prefill과 한 decode를 손으로 잇는다.

prompt 8과 첫 decode 하나만으로 layer별 read/write state를 적는다. Llama full KV append, Qwen3.5 conv/recurrent terminal state, Gemma4 window와 shared source, MoE route temporary를 같은 열에 놓는다. shape가 없는 의미 값인 source layer ID, logical frontier, allowed position set과 original row도 반드시 넣는다.

state byte는 logical/reserved/resident로 계산한다. allocator API 이름을 보고 모두 paged KV라고 부르지 않는다. mutation과 rollback method를 적고 speculative 3→1 accept에서 무엇을 되돌리는지 확인한다. prefix reuse는 필요한 terminal state가 모두 존재하는지 본다.

세 번째 시간에는 앞에서 확정한 state를 backend와 scheduler 소비자까지 연결한다.

각 layer가 실제 선택한 backend와 capability 이유를 기록한다. eager fallback이 의미를 보존하는지, graph capture가 state pointer와 route capacity를 포함하는지 확인한다. scheduler admission 식이 unique state allocation과 workspace를 세는지, preemption이 restore 가능한 단위로 victim을 고르는지 본다.

metric은 이 가설을 반증할 수 있어야 한다. backend selected/fallback, state family bytes와 frontier mismatch, expert skew와 padding, graph reason을 bounded하게 노출한다. model name별 GPU utilization만 있으면 새 architecture의 어느 layer가 비용을 만들었는지 알 수 없다.

### 네 모델을 한 표로 닫는다

| 질문 | Llama dense GQA | Qwen3.5 hybrid | Gemma4 mixed/shared | MoE 경로 |
|---|---|---|---|---|
| persistent state | full KV | full KV + conv + recurrent | full/sliding KV + source edge | attention state는 model별, route는 step temporary |
| token 증가 | KV append | full layer append, GDN state update | full append 또는 sliding overwrite | selected pair 수 증가 |
| rollback | KV tail | conv·recurrent transaction | tail 또는 ring/source generation | uncommitted route 폐기 |
| 주요 동적 좌표 | block/position | 두 state frontier | source layer·window range | original row·expert·permutation |
| 병렬 위험 | TP head/collective | state slot과 backend | shared lifetime·mixed kernel | EP dispatch·TP expert slice |
| silent failure | head/cache order | stale recurrent generation | wrong source/range | wrong inverse scatter |

표의 목적은 모델 순위를 매기는 것이 아니다. residual row가 같은 shape로 돌아와도 그 전에 어떤 persistent state와 permutation을 거쳤는지 보여 준다. 새 architecture가 이 네 열 어디에도 맞지 않으면 억지로 분류하지 않고 새 state family를 추가한다.

### 옵션을 효과가 아니라 state mutation으로 읽는다

TP degree를 바꾸면 local head와 weight shard, collective가 변한다. EP degree를 바꾸면 global→local expert mapping과 dispatch topology가 변한다. attention backend는 mask·cache ABI와 workspace를, GDN backend는 chunk/recurrent state layout을 바꿀 수 있다. graph mode는 padded capacity와 주소 lifetime을 바꾼다. cache dtype은 K/V byte와 writer/reader를 바꾸지만 recurrent state dtype까지 자동으로 바꾼다고 추측하지 않는다.

각 option 실험에는 바뀌어야 할 state, 실제 consumer, 예상 physical effect, correctness invariant와 반증 조건을 쓴다. option field만 바뀌고 selected path가 같으면 성능 차이를 그 option의 효과로 부르지 않는다. 반대로 optimized path가 바뀌었는데 output state ledger가 다르면 속도 측정을 중단하고 correctness부터 닫는다.

실전에서는 이 과정을 한 장의 vertical worksheet로 유지한다. 첫 칸은 immutable model identity와 framework commit이다. 두 번째는 layer plan hash와 layer family count다. 세 번째는 family별 parameter mapping과 persistent state spec이다. 네 번째는 prefill 마지막 generation과 첫 decode read generation이다. 다섯 번째는 backend capability와 effective fallback이다. 여섯 번째는 scheduler byte 식과 rollback method다. 일곱 번째는 bounded metrics와 재현 fixture다. 빈 칸은 “기본 attention과 같을 것”이라는 추측으로 채우지 않는다.

worksheet를 source code와 연결할 때 대표 함수 하나에 모든 책임을 몰지 않는다. config constructor는 layer plan 입력을 정하지만 backend 지원을 보장하지 않는다. model layer constructor는 module을 만들지만 scheduler allocation을 보장하지 않는다. forward는 state를 소비하지만 prefix cache serialization을 보장하지 않는다. backend wrapper는 kernel을 launch하지만 request rollback을 보장하지 않는다. 각 claim 뒤에 책임의 끝을 써야 다음 owner를 찾을 수 있다.

코드를 내려갈 때 residual row 하나를 계속 붙잡는다. layer 진입의 `(request,row,position,generation)`이 projection 또는 router, state read, backend, state commit과 residual add를 지나도 같은 요청을 가리키는지 본다. tensor shape만 기록하지 않고 source state layer와 original row identity를 함께 기록한다. 이 작은 습관이 hybrid cache와 MoE permutation을 별개의 어려운 주제가 아니라 같은 ownership 문제로 보이게 한다.

새 architecture가 multimodal input을 포함하면 text vertical에 무작정 vision tower를 끼워 넣지 않는다. image token 또는 cross-modal state가 text layer의 position, mask, cache와 layer plan을 바꾸는 경계만 표시하고 전처리·encoder 계산은 별도 vertical로 분리한다. text-only fixture가 통과했다고 multimodal support가 증명되는 것도 아니고, vision class가 등록됐다고 text recurrent/backend state가 준비된 것도 아니다.

serving engine 사이 비교도 같은 worksheet로 한다. Transformers reference가 의미를 보여 주고 vLLM이 optimized path를 제공한다고 단정하지 않는다. 각 current pin에서 실제 model registration, loader mapping, state allocator, runner와 backend gate를 확인한다. SGLang이나 llama.cpp에 해당 architecture 경로가 없으면 비슷한 이름의 model로 대체하지 않고 support boundary로 기록한다. 지원 부재는 오류가 아니라 silent reinterpretation을 막는 중요한 사실이다.

문서와 운영 recipe는 effective state를 중심으로 쓴다. “Qwen3.5 실행 명령”, “Gemma4 추천 옵션”을 독립 카드로 나열하면 버전이 바뀌면서 의미가 사라진다. 대신 option이 어느 field를 바꾸고 layer plan, state spec, backend selector와 scheduler 식 가운데 어디서 소비되는지 적는다. 사용자는 자기 환경의 effective manifest를 보고 책의 인과 사슬을 재검증할 수 있다.

worksheet를 실제 변경 검토에 적용해 보자. 한 개발자가 Qwen3.5의 decode backend를 새 fused op로 바꾸는 patch를 제출했다. 검토자는 kernel 이름보다 먼저 기존 state ledger의 입력과 출력이 어디서 연결되는지 본다. fused op가 conv와 recurrent state를 모두 입력으로 받고 새 state 둘과 output을 반환하는지, 아니면 pointer를 in-place로 바꾸는지 확인한다. in-place라면 실패와 speculative reject의 undo owner가 누구인지 묻는다. output numerical test만 있고 state generation test가 없으면 첫 decode 한 번은 통과해도 두 번째 step이나 retry에서 깨질 수 있다.

patch가 새 workspace를 요구한다면 allocation owner와 graph lifetime을 적는다. workspace 크기가 batch rows에 비례하는지, recurrent head dimension에 비례하는지, chunk prefill과 decode가 같은 buffer를 쓰는지 확인한다. scheduler capacity에 persistent byte로 더하지 않되 concurrent layer execution과 graph capture에서 peak가 겹치는지 본다. backend가 unsupported shape에서 기존 reference로 돌아갈 때 workspace와 partially mutated state가 남지 않아야 한다.

같은 검토를 Gemma4 최적화에 적용하면 질문이 달라진다. sliding kernel이 `window_size`만 받는지 actual allowed start position을 받는지, shared KV consumer가 source layer buffer를 직접 받는지 alias table을 받는지 본다. ring cache라면 logical position과 slot generation이 kernel metadata에 포함되어야 한다. full layer와 sliding layer가 같은 wrapper를 공유해도 mask mode와 source pointer가 layer plan대로 바뀌는지 trace fixture로 확인한다.

MoE patch에서는 GEMM TFLOPS보다 permutation contract를 먼저 본다. route item의 original row가 local batch row인지 global token row인지, EP all-to-all을 지나도 안정적인지 확인한다. expert별 padding count가 combine에 전달되는지, empty expert가 있을 때 offset prefix sum이 유지되는지 본다. quantized expert mapping은 qweight와 scale이 같은 local expert permutation을 사용하는지 loader coverage로 닫는다. 이후에야 tile과 communication overlap의 성능을 평가한다.

서빙 recipe를 만들 때도 이 검토 결과를 그대로 사용한다. workload가 긴 context 위주라면 Llama full KV의 page capacity와 prefix hit가 중요하다. Qwen3.5 hybrid는 full layer 비율과 GDN terminal state snapshot/recompute가 중요하다. Gemma4는 actual layer pattern, window와 unique shared source allocation이 중요하다. MoE는 top-k pair 수와 observed skew, EP topology가 중요하다. 단순 parameter count나 context limit만으로 GPU 수를 정하지 않는다.

작은 capacity 예를 더 계산한다. 동시에 request 100개가 length 1,000에 있고 fixture 차원을 그대로 쓴다고 하자. full KV layer 하나의 logical byte는 request당 `2×1000×2×4×2=32,000 bytes`, 전체 3.2 MB다. 네 Llama layer면 12.8 MB다. Qwen3.5의 full layer 하나와 layer당 192 bytes인 GDN 세 개라면 약 3.2576 MB다. 실제 모델에서는 dimension과 layer 수가 훨씬 크지만 growth law의 차이는 같다. context를 두 배로 늘리면 full KV 항은 두 배가 되고 GDN 고정 항은 그대로다.

Gemma4 window 128과 full/shared source가 하나씩이면 request당 unique state는 full 32,000 bytes와 sliding `2×128×2×4×2=4096 bytes`, 합계 36,096 bytes다. consumer layer가 두 개씩이어도 payload가 실제 공유되면 중복하지 않는다. 반면 구현이 semantic sharing만 하고 physical copy를 둔다면 reserved byte는 더 크다. source 코드를 보고 logical dependency와 physical allocation을 따로 기록해야 계산이 맞는다.

MoE persistent cache byte는 attention family가 결정하므로 “MoE라서 KV가 줄었다”고 말할 수 없다. MoE가 더하는 것은 expert parameters와 step workspace다. decode batch 100, top-k 2면 selected pair는 200개다. hidden 16 BF16 activation을 dispatch할 때 payload만 대략 6,400 bytes이며 metadata, alignment, all-to-all buffer와 output이 더해진다. expert skew가 심하면 padded capacity는 균등 분포 계산보다 커진다. persistent와 temporary를 같은 model memory 숫자로 합치면 admission과 peak 원인을 구분하지 못한다.

비용 모델은 측정으로 교정하되 의미 식을 버리지 않는다. observed allocator overhead와 backend workspace coefficient를 layer family별로 추정할 수 있다. 그러나 회귀 모델이 알려 준 총 byte만 저장하면 framework upgrade에서 왜 변했는지 설명할 수 없다. logical state 식, sharing graph, allocator unit과 measured correction을 분리한다. 예측과 실제 차이가 커질 때 어떤 계약이 바뀌었는지 조사할 수 있다.

정확도 검증도 model 전체 perplexity 하나로 끝내지 않는다. Llama 기준은 layer별 cache readback과 output을, Qwen3.5는 terminal state와 첫 decode를, Gemma4는 window boundary와 source layer를, MoE는 selected expert와 combine row를 표본으로 검증한다. end-to-end output은 최종 안전망이지만 first divergence를 설명하지 못한다. 작은 fixture와 실제 checkpoint 표본을 함께 사용하면 synthetic path만 맞고 loader mapping이 틀린 경우도 잡을 수 있다.

운영 장애 보고서에는 수정 option보다 깨진 invariant를 제목으로 쓴다. “GDN kernel disable로 해결”보다 “prefill recurrent generation 7을 decode position 8이 읽음”이 재사용 가능한 지식이다. “Gemma4 eager로 해결”보다 “sliding consumer layer 2가 full source layer 1을 선택함”이 정확하다. “EP off로 해결”보다 “remote dispatch 뒤 inverse row identity가 rank-local로 축약됨”이 다음 backend에서도 통한다.

그런 보고서는 경쟁 가설이 왜 탈락했는지도 남긴다. projection checksum이 같아서 loader를 제외했고, source cache bytes가 같아서 writer를 제외했으며, expert constant output이 같아서 GEMM을 제외했다는 식이다. 나중에 비슷한 증상이 나와도 같은 조사 순서를 적용하거나 다른 최초 divergence 때문에 즉시 갈라낼 수 있다. 단순히 최종 fix commit만 남기면 다시 처음부터 추측하게 된다.

마지막으로 support boundary를 사용자에게 정직하게 표현한다. 특정 current pin에 Qwen3.5 class가 있어도 원하는 quantization과 graph, prefix reuse가 모두 지원된다는 뜻은 아니다. Gemma4 text path가 있어도 multimodal vertical까지 증명하지 않는다. MoE model 하나가 동작해도 다른 router policy와 packed expert format이 호환된다고 말하지 않는다. effective capability manifest와 검증 fixture가 있는 조합만 지원으로 선언한다.

### 코드 읽기의 종료 조건

한 모델을 이해했다는 종료 조건은 architecture 이름을 설명하는 것이 아니다. config에서 layer plan을 재현하고, checkpoint parameter가 올바른 module과 expert slice에 들어가며, prefill이 만든 모든 persistent state를 첫 decode가 같은 generation으로 읽고, backend가 state ABI를 지원하며, scheduler가 byte와 rollback 비용을 계산하는 것을 설명할 수 있어야 한다.

또한 세 가지 실패를 작은 fixture로 재현할 수 있어야 한다. stale state frontier, wrong source/window, wrong route permutation이다. 경쟁 가설을 최초 divergence 이전과 이후로 나누고 수정 뒤 재발 fixture를 남겨야 한다. 이것이 실용적인 수직 비교다.

Qwen3.5 vertical fixture는 config의 layer pattern에서 시작한다. layer 7이 linear-attention/Gated Delta 계열이고 layer 8이 full attention이라고 하자. 동일 hidden row `[B=2,T=1,H=4096]`가 들어와도 layer 7은 convolution/recurrent state slot을 읽고 갱신하며 layer 8은 KV cache page를 읽고 쓴다. scheduler가 보는 token rows 2는 같지만 persistent state family가 다르다.

config parser가 pattern을 normalize하는지, decoder constructor가 layer index 7에 어느 module class를 넣는지, forward가 cache object의 어느 update method를 호출하는지, backend가 prefill/step kernel을 어떻게 선택하는지 한 열씩 적는다. class registration가 있다는 사실로 serving support를 결론내리지 않는다.

prefill fixture는 B=2, chunk T=4다. convolution width 4라면 layer 7은 각 sequence의 last relevant activations를 conv state에 남기고 recurrent matrix/state를 chunk scan 결과로 갱신한다. decode fixture T=1은 이 state를 이전 frontier로 읽고 next state를 쓴다. prefill output logits equality만으로 first decode state를 검증할 수 없다.

state sample은 request R0 layer7 slot3 generation11로 둔다. prefill 마지막 state의 selected scalar/hash가 S11이고 첫 decode kernel input도 S11이어야 한다. scheduler가 KV block table만 넘기고 recurrent slot generation을 빠뜨리면 shape는 `[2,1,4096]`로 정상이어도 state가 zero/stale로 시작한다.

wrong-backend Q35 사건은 model registry가 architecture name을 지원한다고 표시했지만 selected attention backend가 KV-only cache contract만 구현한 경우다. layer 8은 정상이고 layer 7은 generic attention fallback 또는 잘못된 KV path로 들어갔다. request 첫 tokens는 plausible하지만 long sequence와 preemption 뒤 divergence가 커졌다.

최초 불일치는 attention 수학이 아니다. layer plan이 layer7 state family `linear_recurrent`를 요구했는데 backend capability tuple이 `kv_attention`만 true인 상태로 model ready가 publish된 순간이다. capability check는 architecture string보다 layer kinds, prefill/step, dtype, cache state, graph/preemption support를 포함해야 한다.

Gemma vertical fixture는 layer 5가 sliding attention이고 layer 6이 full attention이며 일부 attention layer가 다른 source layer의 KV를 공유한다고 하자. config의 attention type sequence와 sliding window, shared-KV relation이 module construction과 mask/cache plan으로 내려가는지 본다.

layer5 query position 20, window 8이면 allowed key logical range를 fixture policy에 따라 `[13,20]`처럼 계산한다. 정확한 inclusive/exclusive convention은 pinned mask code로 확인한다. physical ring capacity 8과 logical positions를 분리한다. modulo slot만 mask upper/lower bound로 쓰지 않는다.

shared KV relation은 두 layer가 같은 pointer를 우연히 쓰는 것이 아니다. consumer layer6이 producer layer4의 projected K/V와 position provenance를 읽는 source edge다. cache bytes를 줄일 수 있지만 producer completion, eviction, graph buffer lifetime dependency가 생긴다. source layer index를 config/module/cache metadata에 보존한다.

Gemma wrong-source 사건은 shared-KV mapping이 layer index shift로 4 대신 3을 가리킨 경우다. K/V shapes와 head counts가 같아 kernel launch가 성공한다. output은 특정 prompts에서만 흔들린다. layer별 unique sentinel projection을 써 consumer5가 expected producer4를 읽는지 검증한다.

mask와 KV source를 독립적으로 반증한다. source K/V를 correct로 고정하고 allowed key range만 바꾼 test, mask를 correct로 고정하고 source layer만 바꾼 test를 둔다. 둘을 동시에 고치면 first divergence를 모른다. shared cache hit metric 하나로 두 원인을 합치지 않는다.

Llama dense fixture는 기준 GQA layer로 유지한다. config Hq=32, Hkv=8, D=128이면 query heads per KV head는 4다. module projections와 cache shape가 이 tuple을 보존하고 selected attention kernel이 GQA를 지원해야 한다. 공통 attention 설명을 반복하지 않고 다른 architectures의 state 추가분을 비교하는 기준점으로 쓴다.

Llama MoE fixture는 layer9에서 MLP 대신 router와 experts를 선택한다고 하자. tokens4, top-k2면 routing pairs8이고 expert sorting/padding 뒤 GEMM M은 8 이상이다. attention cache state는 dense Llama와 같아도 feed-forward kernel/workspace/EP communication state가 달라진다.

config의 num_experts, experts_per_token, expert intermediate size가 module constructor에서 router/expert arrays를 만들고 forward가 top-k/sort/fused MoE backend로 내려가는 경로를 잇는다. `enable_expert_parallel` option은 architecture를 바꾸지 않지만 expert ownership과 all-to-all/dispatch backend를 바꾼다.

MoE wrong-backend 사건은 fused kernel이 top-k2는 지원하지만 current quantized expert format 또는 expert placement를 지원하지 않는데 selector가 backend 이름만 보고 통과한 경우다. dense layers는 정상이고 expert layers만 wrong output 또는 fallback thrash를 보인다. effective backend reason과 local expert mapping을 기록한다.

세 vertical을 한 row format으로 비교한다. config field, derived layer plan, module class, input tensor shape, persistent state identity, cache update, kernel/backend capability, fallback, output invariant다. model marketing name과 parameter count는 이 표의 key가 아니다.

Qwen layer7 persistent state는 `(request,layer,state-kind,slot,generation,frontier)`로 식별한다. Gemma shared KV는 `(request,producer-layer,consumer-layer,logical-range,cache-generation)`이다. Llama KV는 `(request,layer,block-table-generation)`, MoE intermediate는 `(step,layer,expert-plan-generation)`이다. 같은 `past_key_values` 이름으로 덮지 않는다.

scheduler admission도 layer plan을 읽어야 한다. 요청 tokens와 KV bytes만 계산하면 Qwen recurrent/conv slots, Gemma shared dependency, MoE workspace/communication headroom을 빠뜨린다. 모든 state를 단순 합산하는 대신 persistent capacity와 transient peak, shared alias를 구분한다.

preemption fixture는 더 선명하다. Qwen request를 retract할 때 KV blocks뿐 아니라 recurrent frontier/conv state를 보존하거나 recompute해야 한다. Gemma shared producer KV를 consumer보다 먼저 evict하면 안 된다. MoE step workspace는 in-flight kernel completion 뒤 재사용한다. state family마다 release condition가 다르다.

graph capture key도 architecture string 이상이어야 한다. Qwen prefill/step state shapes, Gemma layer type/window/shared source, MoE token/expert plan과 backend specialization가 compatibility에 영향을 준다. batch size가 같다는 이유로 KV-only graph를 hybrid layer에 replay하지 않는다.

unsupported gap은 명시적으로 기록한다. Transformers reference module가 forward를 지원해도 vLLM/SGLang optimized serving path가 cache/preemption/graph/quant/TP 조합을 모두 지원한다고 추론하지 않는다. gap table은 config parse, eager forward, continuous batch, cache state, graph, distributed, quant kernel별 support를 나눈다.

fallback도 correctness contract를 가진다. unsupported hybrid layer가 eager reference로 갈 수 있으면 cache representation와 request lifecycle가 optimized layers와 호환되는지 확인한다. 중간 layer만 CPU/reference로 바꾸는 것이 가능한지 source가 증명해야 한다. 단순히 `attn_implementation=eager` label로 해결했다고 하지 않는다.

incident test A는 layer pattern을 alternating으로 만들고 각 layer state update counter를 센다. linear layers만 recurrent/conv generation이 증가하고 full attention layers만 own KV generation이 증가해야 한다. counter가 모든 layers에서 같으면 layer dispatch가 무시됐을 수 있다.

test B는 Gemma producer layers에 distinct K/V sentinel을 둔다. consumer outputs에서 expected source signature와 allowed window를 독립 검증한다. cache bytes가 줄었다는 사실보다 source edge correctness가 우선이다.

test C는 MoE experts에 distinct constants를 두고 router top-k pairs, sorted rows, local expert owner, inverse combine을 검산한다. expert backend fallback 전후 output equality를 본다. routing entropy 변화만으로 expert kernel 오류를 판단하지 않는다.

test D는 capability matrix의 한 축씩 끈다. graph unsupported, preemption unsupported, quant unsupported, TP unsupported일 때 selector가 명시적 fallback/rejection을 내는지 본다. silent model-ready publish를 실패로 처리한다.

rollback ladder는 wrong backend만 끄고 known-good path로 전환하는 것부터 시작한다. hybrid cache state가 이미 오염됐으면 in-flight requests를 drain하고 worker model/state epoch를 재생성한다. backend flag만 바꾼 뒤 stale recurrent/shared KV를 계속 쓰지 않는다.

model registration rollback와 traffic rollback를 분리한다. new admissions를 old known-good replica로 보내고 affected replica를 격리한다. model name이 같아 router가 섞지 않도록 effective capability/version을 replica identity에 넣는다.

90분 soak는 prefill chunk1/4/17, decode, preemption, graph/eager, short/long window, top-k experts skew를 섞는다. layer state generation, shared source, mask range, expert combine sentinel, fallback reason이 기대와 일치해야 한다. output equality와 TTFT/ITL을 함께 본다.

terminal의 첫 문장은 “새 architecture가 불안정했다”가 아니다. “layer7이 recurrent state를 요구했지만 KV-only backend capability가 model-name registration만으로 선택됐다” 또는 “Gemma consumer layer가 producer4 대신 3 cache provenance를 읽었다”처럼 최초 state mismatch를 적는다.

수정 뒤 config→derived plan→module→state→cache→kernel의 pinned source와 runtime trace가 같은 layer identity를 가리켜야 한다. unsupported 조합은 명시적으로 거부/fallback하고 supported 조합은 state sentinel와 first-decode reference를 통과한다. 이 수직 왕복이 architecture support의 완료 조건이다.

## 52.7 세 layer를 같은 수직 원장으로 손검산한다

첫 열은 config source field다. 둘째는 constructor가 만든 module class, 셋째는 forward input/output tensor, 넷째는 persistent state, 다섯째는 cache manager ownership, 여섯째는 backend/kernel selector tuple, 일곱째는 scheduler lifecycle, 여덟째는 output invariant다. 모델 이름만 쓰지 않고 layer index와 state kind를 key로 둔다.

Qwen fixture의 config normalized layer plan은 `[linear,linear,full,linear,...]`처럼 표현될 수 있다. 실제 pattern과 field 이름은 pinned Transformers/vLLM source에서 확인한다. 여기서는 layer7=`linear_delta`, layer8=`full_gqa`라는 fixture contract만 사용한다.

layer7 input X shape는 `[B=2,T=4,H=4096]` prefill이다. projection 뒤 gate/value/key-like tensors와 convolution inputs가 생길 수 있지만 exact names/shapes는 implementation source를 따른다. 중요한 serving 출력은 hidden rows뿐 아니라 request별 conv state와 recurrent state frontier다.

conv width4, state channels C라는 fixture에서 request R0의 chunk positions 0–3을 처리하면 next decode가 사용할 last window는 positions0–3이다. 다음 chunk positions4–5를 처리하면 state는 positions2–5에 해당하는 rolling content를 표현해야 한다. chunk boundary에서 zero로 초기화하면 full prefill과 chunked prefill outputs가 갈라진다.

recurrent state는 token count만큼 늘어나는 KV가 아니라 fixed-shape summary일 수 있다. byte capacity가 고정이어도 generation/frontier correctness가 필요하다. R0 frontier4 state를 R1 slot에 배정하거나 frontier3 상태를 첫 decode에 쓰면 memory bounds는 정상이고 output만 틀린다.

layer8 full attention은 같은 request/token frontier를 KV cache로 표현한다. layer7 recurrent slot3 generation11과 layer8 block table generation27은 같은 scheduler request에 속하지만 independent owners다. request preemption ledger가 둘을 모두 보존/해제/recompute해야 한다.

Qwen source trace는 config layer type resolver, decoder layer constructor/selection, linear module forward, cache update object, vLLM model implementation의 layer/backend plan, native kernel dispatch를 순서대로 pin한다. Transformers reference cache path가 있다는 사실과 vLLM continuous serving support를 같은 claim으로 쓰지 않는다.

support gap은 단계별로 적는다. config parse yes, reference eager prefill/decode yes, continuous batch state slots unknown/no, chunked prefill equivalence unknown, preemption restore no, CUDA graph no, quantized kernel combination no처럼 둔다. 빈칸을 model registration yes로 채우지 않는다.

Gemma fixture는 layer plan `[sliding,sliding,full,...]`과 shared source relation을 가진다. layer5 query positions `[20,21]`, window8, producer layer4 KV generation31을 사용한다고 하자. cache manager는 logical allowed range와 physical stored range, producer provenance를 함께 제공한다.

position20의 allowed range가 fixture convention에서 13–20, position21은 14–21이다. prefill chunk가 positions16–21만 전달돼도 earlier stored keys13–15가 필요할 수 있다. current chunk length만으로 mask를 만들지 않는다. global logical position과 cache frontier를 사용한다.

physical ring slot은 position modulo capacity일 수 있다. position20과 12가 같은 slot4를 공유해도 generation/tag가 current logical position20임을 증명해야 한다. shared consumer가 producer layer4 slot을 읽을 때 producer layer index와 logical tag를 확인한다.

shared KV가 layers4→5 관계라면 layer5는 own K/V projection을 생략하거나 다른 semantics를 가질 수 있다. implementation source가 실제로 어떤 tensors를 reuse하는지 pin한다. “shared”라는 config 이름으로 pointer alias 세부를 추정하지 않는다.

Gemma source trace는 config attention type sequence/window/shared relation, module init, mask composition, KV source selection, cache storage/update, serving backend capability로 잇는다. Transformers text model source와 vLLM optimized model source의 commit을 각각 고정한다.

wrong source fixture는 producer layers에 constants4와 3을 넣는다. consumer5 expected output signature4가 아닌 3을 내면 source mapping 오류다. mask를 all-allowed로 고정해 provenance만 검증하고, source를 correct로 고정해 window만 검증하는 두 test로 분리한다.

Llama dense fixture는 layer0 GQA Hq32/Hkv8/D128, B2/T4로 둔다. KV elements per layer/token은 `2×Hkv×D=2048`, FP16 bytes4096이다. 4 tokens×2 requests면 layer cache 신규 bytes32768이다. 이 기준은 Qwen recurrent fixed state와 Gemma shared/sliding capacity의 차이를 수치화한다.

Llama MoE layer9는 hidden rows `M=B×T=8`, experts4, top-k2라 routing pairs16이다. expert row alignment4이고 counts가 `[5,4,4,3]`이면 padded rows `[8,4,4,4]`, total20이다. attention KV bytes는 dense 기준과 같지만 transient expert GEMM M과 workspace가 token rows8보다 커진다.

router output indices/weights는 sort plan으로 변하고 local expert ownership을 통과해 fused MoE kernel arguments가 된다. combine는 original `(token,route-slot)`으로 되돌린다. expert constants0/10/20/30과 one-hot routes로 permutation/inverse를 검증한다.

EP2라면 experts0–1과 2–3 owner ranks를 정한다. scheduler/token dispatch, all-to-all backend, local expert kernel, return/combine completion이 lifecycle에 추가된다. `enable_expert_parallel`은 KV cache를 바꾸지 않지만 transient buffers와 collective owners를 바꾼다.

Llama/MoE source trace는 config/model constructor, decoder layer MLP selection, router/top-k, fused MoE method, backend option normalization/selector, expert placement, native kernel call로 잇는다. quantized experts라면 format/kernel ABI capability를 selector tuple에 추가한다.

세 fixture의 capacity를 비교한다. Llama full KV는 sequence 길이에 선형 증가한다. Gemma sliding KV physical capacity는 window로 제한될 수 있지만 shared source dependency가 있다. Qwen recurrent/conv state는 fixed-size일 수 있지만 chunk/preemption frontier가 있다. MoE workspace는 step routing distribution에 따라 transient peak가 변한다.

따라서 admission formula는 `persistent_KV + persistent_recurrent/conv + shared ownership adjustments + transient_MoE/workspace + graph buffers + safety headroom`이다. 모든 architectures에 max sequence×KV bytes 하나를 적용하지 않는다. 실제 implementation가 state를 어떻게 allocate하는지 source로 보정한다.

selector audit는 요구 capability set과 제공 set의 포함 관계로 쓴다. Qwen layer7은 recurrent prefill, recurrent step, chunk state, preemption, dtype/device를 요구한다. Gemma layer5는 sliding mask, shared source, position tags, graph compatibility를 요구한다. MoE layer9는 top-k, expert format, placement, workspace, collective mode를 요구한다.

backend 이름이 등록됐어도 required subset 하나가 false면 fallback/reject해야 한다. fallback가 semantic-equivalent인지 reference fixture로 검증한다. output shape만 같은 generic attention는 recurrent state update를 대신하지 못한다.

wrong-backend incident WB52는 mixed model replica가 capability cache를 architecture name으로만 key한 경우다. 이전 Llama replica의 `flash_backend_supported=true` entry를 Qwen3.5 replica가 재사용했다. layer8 full attention는 맞았지만 layer7 recurrent layer가 KV-only launch plan을 받았다.

model load health check는 short prompt logits만 검사해 통과했다. long chunked prefill 후 first decode와 preemption resume에서 divergence가 나타났다. capability cache key에 layer plan/state schema/model implementation version가 없었던 것이 최초 불일치다.

fix는 capability key를 model name 문자열이 아니라 normalized layer plan hash, state schema version, backend/dtype/quant/TP/graph tuple로 만든다. model ready 전 모든 unique layer kinds의 prefill/step fixture를 실행한다. unsupported layer kind가 하나라도 있으면 partial optimized publish를 금지한다.

fallback가 layer별 혼합을 지원하려면 hidden tensor/device/stream/cache state contract가 layers 사이에서 호환돼야 한다. source가 이를 명시하지 않으면 entire model/replica를 known-good backend로 보낸다. 편의상 unsupported layer 하나만 eager Python으로 호출한다고 가정하지 않는다.

rollback는 WB52 replica의 new admission을 막고 state epoch를 폐기한다. 이미 recurrent state가 잘못 갱신된 requests는 correct backend로 이어서 decode하지 않고 retry/terminal policy를 적용한다. stale state 위에 backend만 바꾸면 output correctness가 회복되지 않는다.

Gemma wrong-source replica도 cache provenance가 오염됐으면 affected requests와 cache entries를 격리한다. shared KV mapping fix 뒤 old cache generation을 재사용하지 않는다. Llama MoE wrong expert mapping도 step workspace/route plan generation을 폐기한다.

regression matrix는 Qwen full/chunked prefill→first decode→preemption, Gemma window boundary/source layer, Llama dense GQA, MoE top-k/expert skew를 포함한다. backend/capability flag를 한 축씩 바꿔 explicit selection/fallback reason을 검증한다.

telemetry는 architecture label 하나 대신 normalized layer-plan hash, current layer kind, state family, backend selector result, fallback reason을 trace에 둔다. metrics label은 bounded model/backend/state-kind 정도로 제한한다. 특정 request layer provenance는 anomaly trace에서 본다.

90분 soak는 mixed sequence lengths, chunk sizes, graph buckets, preemption, adapter/quant 조합, expert skew를 섞는다. recurrent first-decode hash, shared source tag/window, KV generation, expert combine sentinel가 0 mismatch여야 한다. selected backend가 request 중간에 설명 없이 바뀌지 않아야 한다.

terminal report는 세 vertical의 source pins와 fixture results를 나란히 둔다. config→module class→state object→cache update→kernel selector→output invariant가 각 layer에서 이어져야 한다. unsupported cells는 빈칸이 아니라 explicit gap/rejection/fallback evidence를 가진다.

이 비교의 목적은 모델 우열이 아니다. architecture 차이가 serving state와 lifetime, scheduler cost, kernel capability를 어디서 바꾸는지 찾는 것이다. 공통 attention 수식을 다시 설명하지 않고 divergence가 처음 생기는 layer/state boundary를 보여 준다.

layer-plan hash producer는 normalized config에서 시작한다. raw config field order나 JSON formatting을 hash하지 않고, layer index별 semantic kind, state schema version, relevant dimensions/window/shared source, MoE properties를 canonical tuple로 만든다. unknown field를 조용히 버리면 future architecture가 old plan으로 충돌할 수 있으므로 schema version과 unsupported marker를 둔다.

Qwen fixture의 canonical entries는 `(7,linear_delta,conv4,recurrent_schemaX)`와 `(8,full_gqa,Hq32,Hkv8,D128)`처럼 표현한다. Gemma는 `(5,sliding,window8,kv_source4)`를 포함한다. Llama MoE는 `(9,moe,E4,topk2,intermediate,expert_format)`를 포함한다. exact fields는 pinned implementation가 실제 state/kernel selection에 사용하는 값만 넣되 누락 test를 둔다.

hash consumer 첫째는 model implementation/runner initialization다. constructor가 만든 actual module classes와 plan entries를 대조한다. config가 layer7 linear라고 했는데 module list가 full attention이면 model-ready 전에 실패한다. Python class 이름만 비교하지 않고 declared state schema를 확인한다.

둘째 consumer는 cache manager allocation다. plan의 state families로 KV groups, sliding/full capacity, recurrent/conv slots, shared source edges를 만든다. allocator 결과에는 plan hash를 붙인다. 다른 hash의 cache object를 request나 graph replay가 재사용하지 못한다.

셋째 consumer는 backend selector다. unique layer kind마다 required capabilities를 계산하고 backend capability registry와 비교한다. selector result는 `(plan_hash,layer_kind,dtype,quant,tp/ep,graph,prefill_or_step)` key로 cache한다. architecture name 하나로 결과를 공유하지 않는다.

넷째 consumer는 scheduler cost model이다. token budget뿐 아니라 state slot availability, KV group blocks, MoE workspace/collective headroom을 plan에서 얻는다. cost model이 모르는 state kind는 zero cost로 처리하지 않고 admission gap으로 표시한다. reference-only fallback의 cost도 별도다.

다섯째 consumer는 graph capture inventory다. graph key는 batch/capture size와 함께 plan/state schema/backend specialization를 포함한다. Qwen full layer graph를 recurrent layer에 쓰거나 Gemma shared-source generation가 다른 replay를 재사용하지 않는다. layer execution이 한 graph에 묶이면 전체 plan compatibility를 본다.

여섯째 consumer는 model-ready health gate다. unique layer kinds의 minimal prefill/step/state transition fixtures와 source capability evidence를 모은다. registration import 성공이나 dummy forward 한 번은 충분하지 않다. gap이 있으면 replica를 ready로 publish하지 않거나 명시된 safe fallback 전체 경로를 검증한다.

WB52 timeline은 이를 숫자로 적는다. t0에 Llama plan hash HL이 backend cache key `architecture=causal_lm` 아래 flash supported를 저장한다. t1에 Qwen plan HQ가 같은 architecture label로 시작한다. t2 selector cache hit가 layer7 required recurrent capabilities를 계산하지 않고 flash KV plan을 반환한다. t3 model-ready short full-attention probe가 통과한다.

t4 chunked prefill R0 positions0–15가 layer7 recurrent state를 갱신하지 않는다. hidden outputs는 fallback approximation 때문에 plausible하다. t5 first decode position16이 state slot generation0을 읽고 reference와 처음 diverge한다. t6 preemption/resume에서 stale slot이 다른 request generation과 섞여 divergence가 확대된다.

최초 모순은 t2다. backend kernel이 계산을 틀린 t5가 아니다. selector가 HQ/layer7 required set을 보지 않고 HL의 cached result를 소비했다. source audit는 cache key producer, lookup, required-capability builder, runner consumer를 pin한다. trace는 current plan hash와 cached plan hash를 모두 남긴다.

regression은 두 models load order를 뒤집는다. Llama→Qwen, Qwen→Llama, concurrent init 모두 같은 effective results를 내야 한다. cache cold/warm도 교차한다. 이전 model의 selector entry가 다음 model layer plan에 영향을 주면 실패다.

hash collision test는 window, shared source, conv width, top-k 중 한 field씩 바꿔 key가 달라지는지 본다. semantic effect가 없는 metadata field 변경은 key를 불필요하게 바꾸지 않을 수 있다. 무엇이 key material인지 schema 문서와 test로 고정한다.

unsupported matrix의 행은 layer/state feature다. columns는 Transformers eager reference, vLLM eager/continuous, SGLang eager/continuous, graph, chunked prefill, preemption, TP, EP, quant backends다. cell은 yes/no/fallback/unknown과 pinned evidence를 가진다. model family 전체를 yes 한 칸으로 표시하지 않는다.

Qwen recurrent cell은 prefill scan, one-step update, chunk carry, state-slot batching, request reorder, preemption restore를 나눈다. prefill/step kernel이 있어도 scheduler reorder가 slot mapping을 안전하게 갱신하지 못하면 continuous batching support는 incomplete다.

Gemma cell은 attention type sequence, sliding mask logical positions, ring tags, shared KV source/provenance, graph buffer refresh를 나눈다. sliding attention 지원이 shared-source 지원을 자동 포함하지 않는다. window length 하나만 맞아도 source layer가 틀릴 수 있다.

Llama/MoE cell은 dense GQA와 router/expert path를 분리한다. top-k, expert count/placement, quantized expert format, fused kernel, EP communication, graph/workspace가 axes다. dense Llama 지원을 MoE support로 확대하지 않는다.

unknown cell은 지원으로 간주하지 않는다. source path를 더 추적하거나 small fixture로 확인할 backlog다. release가 급하면 known-good backend/architecture 조합만 allowlist한다. undocumented silent fallback에 기대지 않는다.

wrong-backend 반증 A는 recurrent state를 zero/known sentinel로 바꿔 first decode 민감도를 본다. correct backend는 prefill state를 사용해 sentinel reference와 일치한다. KV-only path는 state mutation counter와 output이 함께 다르다.

반증 B는 Gemma producer source만 swap한다. correct mask를 고정하고 producer4/3 outputs가 distinct signature를 내게 한다. selector/backend가 shared-source argument를 무시하면 두 runs가 같거나 wrong signature를 낸다.

반증 C는 MoE fused backend를 generic reference로 바꾼다. router indices/weights를 고정한다. generic은 맞고 fused만 틀리면 expert kernel/placement capability를 본다. 둘 다 틀리면 upstream router/name mapping을 본다.

반증 D는 graph를 끈다. graph-off만 정상이라면 state buffer refresh/key/lifetime를 조사한다. eager도 틀리면 module/backend state semantics를 본다. graph와 backend를 동시에 바꾸지 않아 root branch를 보존한다.

rollback 준비는 active requests를 state family별로 분류한다. Qwen recurrent wrong state requests는 restart/retry가 필요하다. Gemma cache provenance가 틀린 requests도 old cache를 폐기한다. Llama dense requests가 unaffected인지 plan hash로 분리한다. replica 전체 restart가 안전하고 단순할 수 있다.

backend cache invalidation는 code deploy만으로 충분하지 않을 수 있다. process/global selector cache, graph exec cache, model runner state plan, KV/recurrent caches를 새 schema epoch로 재생성한다. shared external cache가 있다면 plan hash namespace를 바꾼다.

canary는 unique layer kinds에 synthetic probes를 주기적으로 보낸다. Qwen chunk→first decode state hash, Gemma window/source sentinel, MoE expert constants를 확인한다. production text를 로그에 남기지 않고 deterministic fixture를 사용한다.

metrics는 unsupported selection attempts, explicit fallback, plan-hash mismatch, state-generation mismatch, graph-key rejection를 둔다. fallback count가 늘면 correctness는 유지돼도 capacity/SLO가 바뀔 수 있다. model-ready 여부와 effective backend distribution을 함께 본다.

source evidence는 config/module/cache/kernel 네 종류를 서로 대신하지 않는다. Transformers config와 reference forward는 architecture semantics를 보여 준다. vLLM/SGLang model runner와 cache manager는 serving lifecycle를 보여 준다. native backend source는 kernel capability를 보여 준다. 각각 고정 commit/line을 연결한다.

Qwen3.5 지원 gap이 실제 optimized source에 없으면 그 사실을 명시한다. 다른 Qwen version이나 비슷한 DeltaNet 구현을 직접 지원 증거로 대체하지 않는다. inference로 비교할 때는 “유사 state contract로 추정”이라고 표시하고 production support 판정은 보류한다.

Gemma 계열 version 차이도 보존한다. Gemma2/3/4 이름을 섞지 않고 fixture가 가리키는 config/model implementation의 exact class/commit을 쓴다. shared KV가 특정 version/variant에만 있으면 일반 Gemma 법칙으로 확대하지 않는다.

Llama MoE 역시 dense Llama와 별도 architecture/variant일 수 있다. 이 장의 비교 축은 Llama-like dense GQA 기준과 MoE feed-forward variant다. 실제 named model config가 어떤 modules를 만드는지 source로 고정한다. “Llama는 모두 MoE” 같은 오해를 막는다.

terminal source table은 layer7 Qwen, layer5 Gemma, layer0 Llama dense, layer9 MoE 네 rows를 가진다. 각 row에 config field URL, module constructor URL, state/cache consumer URL, backend selector/kernel URL, unsupported note를 둔다. 한 URL로 모든 cells를 채우지 않는다.

terminal runtime table에는 plan hash, layer kind, state generation before/after, selected backend/reason, output sentinel, preemption/graph result를 둔다. source가 가능한 경로를 증명하고 runtime은 canary가 실제 그 경로를 소비했음을 증명한다.

완료 조건은 WB52 재현, first divergence t2 확인, cache key fix, old state epoch 폐기, cold/warm/load-order regression, 90분 soak다. unsupported matrix unknown은 release allowlist 밖에 둔다. 이 조건 뒤에만 Qwen/Gemma/MoE optimized replica를 ready로 publish한다.

사후 회고에는 성능 향상보다 support claim의 범위를 적는다. 어느 layer kinds, state schema, chunk/preemption/graph/quant/TP/EP 조합이 검증됐는지 명시한다. 독자는 이 범위를 벗어난 요청이 fallback/reject되는 이유를 이해할 수 있다.

이 vertical audit를 익히면 새 model architecture를 만났을 때 공통 attention 수식을 다시 공부하는 데 시간을 쓰지 않는다. config가 module과 state를 어디서 바꾸고 그 state를 어떤 cache/kernel/lifecycle가 소유하는지 바로 찾는다. serving correctness와 backend capability가 만나는 실제 경계다.

실제 30분 source drill은 config class에서 시작한다. Qwen layer type pattern을 읽고 layer index7의 normalized value를 적는다. model constructor에서 index7에 생성된 module object를 찾는다. forward에서 그 object가 호출하는 cache/state update와 returned state를 찾는다. serving runner에서 같은 layer의 backend choice와 state slot metadata를 찾는다.

네 지점 가운데 하나가 source에 없으면 support gap이다. reference model에 module이 있어도 serving runner가 별도 implementation를 가지면 자동 연결하지 않는다. dynamic module dispatch나 generated code라면 registry/table과 runtime resolved class를 evidence로 남긴다.

Gemma drill은 config attention types에서 layer5를 선택하고 module init의 attention type/window/source relation을 확인한다. forward mask composer의 logical position inputs, cache update의 physical slots/tags, backend launcher의 window/source arguments를 잇는다. shared producer completion과 cache eviction owner도 찾는다.

Llama/MoE drill은 decoder layer9가 dense MLP인지 sparse expert module인지 constructor predicate를 확인한다. router output shapes, top-k normalization, sort/pad plan, local expert mapping, fused kernel arguments, inverse combine을 한 route pair로 추적한다. expert quant format이 있으면 loader/backend capability를 추가한다.

각 drill은 한 sample coordinate를 가진다. Qwen은 `(R0,layer7,frontier16,state_slot3,generation11)`, Gemma는 `(R0,layer5,qpos20,producer4,keys13..20,cache31)`, MoE는 `(token0,route1,expert2,sorted_row8,plan9)`이다. 함수 이름 목록보다 이 coordinate가 보존되는지 본다.

scheduler trace에는 current layer를 모든 kernel마다 상시 label로 넣을 필요는 없다. anomaly/canary에서 layer-plan hash와 state kind, selected backend, generation를 sample한다. bounded metrics에는 plan/backend/fallback totals를 둔다. cardinality와 디버깅 detail을 분리한다.

state byte 계산도 sample과 연결한다. Llama KV는 layer/token에 따라 증가하고 Qwen recurrent state는 slot/layer fixed allocation일 수 있다. Gemma sliding/shared KV는 physical capacity와 logical dependency가 다르며 MoE workspace는 step maximum이다. allocator metric가 이 categories를 표현하는지 본다.

같은 HBM 사용량이어도 lifecycle 위험이 다르다. KV block은 token ownership/eviction를 따르고 recurrent slot은 request frontier/reorder를 따르며 shared KV는 producer-consumer edge를 따른다. MoE workspace는 launch completion를 따른다. aggregate cache-used percent만으로 leak/pressure를 설명하지 않는다.

preemption cost model도 fixture로 측정한다. Qwen state를 preserve하면 fixed bytes copy/hold, recompute하면 prompt scan cost가 든다. Gemma sliding KV recompute는 window만으로 충분한지 shared full producer가 필요한지 본다. Llama full KV recompute는 prefix tokens, MoE는 prior step workspace를 보존할 필요가 없는지 source contract를 확인한다.

chunked prefill boundary matrix는 Qwen chunk sizes1/4/15/16/17, Gemma window boundary7/8/9, Llama page boundary, MoE token alignment 전후를 고른다. full prefill reference와 first decode를 비교한다. final prefill logits만 비교하지 않는다.

request reorder test는 batch rows R0/R1을 step 사이 swap한다. state slot mapping과 cache tables가 request identity를 따라야 한다. Qwen recurrent states, Gemma shared source caches, Llama KV blocks, MoE current route plan 가운데 persistent와 transient를 구분한다. row index를 owner identity로 쓰면 swap에서 실패한다.

abort test는 R0을 kernel enqueue 뒤 제거한다. persistent state slots/cache refs는 completion-safe cleanup를 따르고 R1 mappings는 유지된다. MoE workspace가 batch shared라면 R0 output drop가 workspace 조기 free를 일으키지 않는다. architecture support에는 failure lifecycle도 포함된다.

distributed test는 TP ranks가 동일 layer plan hash와 backend decision을 가졌는지 startup에서 확인한다. EP ranks는 local expert mapping digest를 비교한다. 한 rank만 fallback하면 collectives와 output contract가 갈라질 수 있으므로 replica ready를 거부한다.

quant test는 Qwen recurrent kernel이나 Gemma shared attention, MoE expert GEMM가 current dtype/format을 지원하는지 feature별로 본다. model 전체 quant method가 로드됐다는 사실로 모든 layer kernels support를 주장하지 않는다. unsupported layer가 safe dequant/reference path로 갈 수 있는지 검증한다.

adapter test도 layer state와 결합할 수 있다. LoRA가 projection weights를 바꾸면 Qwen state evolution과 Gemma producer K/V values가 adapter provenance를 가져야 한다. cache reuse가 adapter identity를 key에 포함하는지 본다. 이 장에서는 adapter kernel 세부를 반복하지 않고 support matrix 축으로만 기록한다.

wrong-backend dashboard는 first-decode mismatch, chunk equivalence, source provenance mismatch, expert basis mismatch를 correctness signals로 둔다. TTFT/ITL 변화는 보조다. 빠른 잘못된 backend가 selection success로 집계되지 않게 output/state probes를 둔다.

fallback storm은 selector가 request마다 unsupported path를 시도한 뒤 exception/fallback하는 경우다. model init에서 capability를 결정하고 stable effective plan을 만들 수 있는지 본다. runtime dynamic shapes만 request-level guard로 남긴다. 반복 warning/compilation이 tail latency를 만들지 않게 한다.

rollback 성능 계획은 known-good backend의 throughput/capacity를 계산한다. optimized replica를 격리하면 admission headroom을 낮추거나 replicas를 늘려야 할 수 있다. correctness 복구와 SLO 완화를 함께 운영하되 unsafe backend를 급히 재활성화하지 않는다.

source upgrade audit는 config/model implementations뿐 아니라 cache interfaces와 backend capability registry diff를 본다. layer-plan schema field가 추가됐으면 old cache/hash를 무효화한다. new architecture alias가 등록됐어도 unique layer fixtures가 없는 상태로 allowlist하지 않는다.

paper나 model card는 architecture intent를 제공하지만 current serving support를 증명하지 않는다. config와 reference source는 exact module/state semantics를, serving source는 lifecycle/backend를, tests/trace는 current build behavior를 증명한다. 세 evidence 역할을 구분한다.

terminal report의 unsupported matrix는 시간이 지나면 갱신된다. unknown이 yes가 되려면 pinned consumer source와 regression fixture가 추가돼야 한다. no가 fallback로 바뀌면 fallback state compatibility와 performance를 기록한다. 단순 release note 한 줄로 cell을 바꾸지 않는다.

WB52 fix의 final canary는 Llama를 먼저 load한 뒤 Qwen을 load해 selector warm-cache collision를 재현하지 않는지 확인한다. 반대 순서와 concurrent load도 통과한다. plan hashes와 cache entries가 distinct하고 unique layer probes가 expected backend/state generations를 보인다.

Gemma canary는 producer mapping table을 hot reload하거나 graph recapture할 때 old cache provenance가 섞이지 않는지 본다. MoE canary는 expert placement change 뒤 old route/kernel plan을 폐기한다. plan hash가 state/cache/graph identities 전체에 전파돼야 한다.

90분 terminal 뒤 worker restart를 한 번 포함한다. selector cache cold 상태와 restored/compiled artifacts가 같은 effective plan을 만든다. restart 뒤만 정상이라면 invalidation path가 누락됐을 수 있다. hot upgrade와 cold boot를 모두 검증한다.

최종 승인자는 네 sample coordinates를 source와 trace에서 직접 왕복한다. 하나라도 backend capability가 inferred/unknown이면 해당 combination을 allowlist 밖에 둔다. 지원 범위를 좁게 정확히 말하는 것이 model 이름 전체를 성급히 지원한다고 쓰는 것보다 운영에 유용하다.

이렇게 닫으면 독자는 새 모델을 보고 “attention이 무엇인가”부터 되풀이하지 않는다. 어떤 layer가 기준 Llama GQA와 달라지고, 그 차이가 persistent state·cache provenance·workspace·kernel selector 중 어디에 새 owner를 만드는지 조사한다. 바로 그 owner가 서빙 최적화와 실패 복구의 출발점이다.

최종 fault campaign은 layer-plan 생산 직후, module construction 뒤, cache allocation 뒤, backend selection 뒤, graph capture 뒤에 각각 실패를 넣는다. 어느 단계에서도 partial ready replica가 router에 노출되면 안 된다. created modules/state allocations/compiled graphs는 owner 역순으로 정리하고 이전 replica는 계속 serving한다.

module list와 plan mismatch는 load-time fatal이다. cache manager가 state kind를 모르면 allocation를 0으로 진행하지 않는다. backend capability 부족은 검증된 fallback가 있으면 effective plan에 명시하고 없으면 reject한다. graph만 unsupported면 eager path가 동일 state contract를 유지하는지 test한 뒤 선택한다.

runtime shape guard 실패도 reason을 보존한다. 특정 chunk length나 expert count에서만 kernel specialization이 없으면 generic safe path로 갈 수 있지만 state generation와 cache layout가 호환돼야 한다. request 중간 backend switch가 state representation를 바꾸면 새 request/replica boundary에서만 전환한다.

incident 종료 뒤 support 문구도 고친다. “Qwen3.5 지원” 대신 검증한 layer plan hash, dtype/quant, TP/EP, chunk/preemption/graph 조합을 release artifact에 기록한다. Gemma variant와 shared-KV relation, Llama/MoE expert format도 명시한다. 사용자는 자신의 config tuple이 allowlist에 있는지 확인할 수 있다.

reference fixture는 release artifact와 함께 version한다. config samples, deterministic states, expected first-decode/output sentinels, selector results를 저장한다. source commit가 바뀌면 fixture를 재생성하기 전에 semantic diff를 검토한다. golden output만 무비판적으로 갱신하지 않는다.

마지막 terminal 문장은 다음과 같다. “HQ layer7 required recurrent/chunk/preemption capabilities를 selector가 plan-hash key로 검증했고 KV-only HL cache entry를 거부했다. Gemma producer provenance와 MoE expert combine도 고정 fixture를 통과했으며 cold/warm/restart soak에서 mismatch 0이었다.” 이 증거가 있어야 vertical support가 닫힌다.

승인자는 마지막으로 네 layer sample의 config commit, module class, state/cache schema, selected native path를 한 표에서 대조한다. runtime trace의 plan hash가 source에서 계산한 canonical tuple과 같고, fallback cell은 명시적 reason과 reference equality를 가져야 한다. model name이나 output shape만 맞는 행은 통과시키지 않는다.

배포 뒤 새 config variant가 들어오면 기존 family allowlist를 자동 상속하지 않는다. canonical tuple을 다시 만들고 unsupported matrix를 조회한다. unknown layer/state feature는 격리된 reference test로 보내거나 load를 거부한다. 이 gate가 silent wrong-backend 선택을 model-ready 이전에 멈춘다.

이 최종 판정까지 갖추면 architecture 비교는 정적인 사양표가 아니라 실행 가능한 serving contract가 된다. 독자는 config 한 필드가 state owner와 kernel 요구를 어떻게 바꾸는지 source, fixture, 실제 운영 trace에서 끝까지 동일하게 확인하고 승인할 수 있다.

### 회고

Llama는 단순해서 기준이 된 것이지 보편 구현이어서 기준이 된 것이 아니다. Qwen3.5는 token별 KV history 대신 bounded하지만 매 step 일관되게 갱신해야 하는 conv/recurrent state를 추가한다. Gemma4는 layer type에 따라 허용 history와 KV source dependency를 바꾼다. MoE는 persistent activation history와 별개로 token row의 동적 소유권과 permutation을 추가한다.

서빙 최적화의 “왜”는 여기서 나온다. full KV는 context와 함께 커지므로 paging과 prefix reuse가 중요하다. recurrent state는 작아도 terminal snapshot과 atomic mutation이 중요하다. sliding state는 bounded capacity 대신 modulo 주소와 rollback이 중요하다. shared KV는 byte를 줄이는 대신 lifetime edge를 만든다. MoE는 모든 token에 모든 expert를 계산하지 않는 대신 dispatch, imbalance와 combine correctness를 감당한다.

따라서 새 모델을 빠르게 서빙하는 첫 단계는 가장 빠른 kernel을 고르는 일이 아니다. 그 모델이 실제로 보존해야 하는 state를 정확히 이름 붙이고, 한 token이 그 state를 어떻게 읽고 쓰는지 닫는 일이다. 그다음에야 allocator, scheduler, graph와 kernel이 같은 계약을 최적화할 수 있다.
## 52.8 모델 카탈로그 1: Llama dense GQA

Transformers v5.15.1의 [`LlamaAttention`과 decoder layer](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/modeling_llama.py#L197-L332)는 separate Q/K/V projection, rotary position, cache update, attention interface와 dense MLP가 만나는 기준을 제공한다.

[`LlamaModel.forward`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/modeling_llama.py#L376-L471)는 position과 cache를 layer loop에 전달한다. 이 좌표를 모든 모델의 공통 구현이라고 부르지 않고 비교용 baseline으로만 쓴다.

prompt 길이 8, batch 2라면 normalized hidden은 `[16,16]`으로 펼칠 수 있다. 각 token은 Q head 네 개, KV head 두 개를 만든다. Q는 현재 attention 계산 뒤 사라지는 step temporary다. K/V는 request와 layer가 소유하는 persistent state에 append된다. GQA에서 query head 두 개가 KV head 하나를 공유하지만 cache의 logical token frontier는 request마다 8이다.

### prefill에서 쓰고 decode에서 읽는 것

prefill layer는 여덟 position의 K/V를 만들고 causal mask 아래 attention을 계산한다. cache write가 성공하면 committed frontier는 8이다. 첫 decode는 position 8의 Q/K/V를 만들고 과거 0~7과 새 position 8을 읽는다. commit을 attention 이전에 표시하는지 이후에 표시하는지는 implementation과 speculative policy에 달렸지만, 실패나 reject 때 어느 범위를 되감는지는 명시되어야 한다.

BF16이고 batch 하나만 생각하면 layer별 K/V logical byte는 `2 × 8 × Nkv × D × 2`다. fixture에서는 `2×8×2×4×2=256 bytes`다. 첫 decode 뒤 288 bytes, 세 step 뒤 352 bytes다. allocator의 page padding과 metadata는 별도다. 이 식은 full attention KV에만 적용한다. 뒤에서 convolution state나 recurrent matrix에 token length를 그대로 곱하지 않는다.

Llama dense MLP는 각 token row를 같은 weight에 통과시킨다. scheduler 관점에서 token 수가 GEMM 행 수를 결정하며 expert별 skew나 permutation은 없다. TP를 적용하면 projection과 output/MLP weight ownership, collective가 생기지만 token이 어느 expert로 갔는지라는 동적 routing state는 없다.

### serving implementation에서 확인할 경계

vLLM v0.27.1의 [`LlamaAttention`부터 model과 loader까지](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/llama.py#L122-L441)를 읽을 때 class 이름보다 네 경계를 본다. config에서 local Q/KV head를 계산하는 지점, packed QKV weight가 rank-local destination에 들어가는 지점, attention layer가 cache config와 backend를 결합하는 지점, loaded parameter coverage를 닫는 지점이다. Transformers의 separate projection과 vLLM의 packed projection은 물리 표현이 달라도 동일한 logical head ledger를 보존해야 한다.

backend에는 Q pointer만 가는 것이 아니다. K/V cache address, block table 또는 sequence length, position·mask 정보와 head layout이 함께 간다. optimized kernel이 없다면 eager reference로 의미를 보존할 수 있어야 한다. fallback은 느릴 수 있지만 silent하게 MHA로 head를 복제하거나 cache를 생략해서는 안 된다.

Llama에서 prefix reuse는 full attention KV prefix가 model revision, adapter와 position 의미까지 같을 때 가능하다. block 단위 reference count와 copy-on-write는 33~37장의 책임이다. 여기서는 reuse 대상이 layer별 K/V라는 점만 고정한다. preemption이 GPU KV를 버리거나 CPU로 옮기면 재개 전에 같은 logical frontier를 복원해야 한다.

### 기준 원장의 불변식

각 layer 종료 시 residual shape는 `[rows,H]`로 돌아온다. 읽은 KV frontier는 attention이 허용한 마지막 position과 맞고, 쓴 frontier는 accepted token까지만 포함한다. dense MLP는 row order를 바꾸지 않는다. backend가 달라도 layer output과 cache readback의 logical head order는 같다. 이 네 가지가 다음 모델 비교의 기준이다.

Qwen3.5는 이 기준에서 attention layer 일부를 다른 연산으로 바꾼다. residual shape가 같다는 이유로 state까지 KV라고 부르면 안 된다. 차이는 layer type이 materialize되는 순간부터 시작된다.

## 52.9 모델 카탈로그 2: Qwen3.5 hybrid state

Transformers의 [`Qwen3_5TextConfig`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/configuration_qwen3_5.py#L29-L121)는 layer pattern과 attention 관련 구조를 config로 표현한다. model은 layer index에 따라 full attention 또는 Gated DeltaNet 계열 모듈을 materialize한다. 네 layer fixture에서 `[GDN,GDN,GDN,full]`을 선택했다고 하자. cache container는 “네 개의 같은 KV layer”가 아니라 서로 다른 state spec 네 개를 담아야 한다.

[`Qwen3_5GatedDeltaNet`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L388-L540)은 projection과 gating, convolution/recurrent state의 prefill·decode 경계를 읽는 고정 좌표다.

[`full attention과 layer 선택`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L604-L793)은 같은 decoder 안에서 두 family가 교대하는 것을 보여 준다. 여기서는 DeltaNet 수식을 다시 전개하지 않고 persistent state가 어떻게 달라지는지 추적한다.

### prefill chunk와 single-token decode는 같은 호출 모양이 아니다

prefill에서 GDN은 여러 token을 chunk 또는 recurrent scan으로 처리하고 마지막 convolution window와 recurrent accumulator를 남긴다. convolution state의 크기는 전체 prompt 길이에 선형으로 늘지 않고 kernel width와 channel 구조에 의해 정해질 수 있다. recurrent state도 `[head,key_dim,value_dim]` 같은 matrix family이며 full KV처럼 token별 vector를 모두 보관하는 것이 아니다. prompt가 8에서 8,000으로 늘어도 state byte가 1,000배가 된다고 쓰면 틀린다.

decode에서는 새 token 하나가 conv state의 오래된 lane을 밀어내고 새 activation을 넣는다. 이어 recurrent state를 읽어 output을 계산하고 accepted update를 반영한다. chunk path가 만든 마지막 state와 single-token path가 기대하는 layout, dtype, normalization이 같아야 한다. prefill output logits가 맞아도 마지막 state export가 틀리면 첫 decode부터 달라진다.

full attention layer는 여전히 token 축 K/V를 append한다. 따라서 한 request 안에 세 GDN state 묶음과 한 KV state가 공존한다. cache length API가 첫 attention layer의 token frontier를 대표값으로 반환하더라도 GDN state readiness는 별도다. scheduler가 length 하나만 보고 모든 layer가 준비됐다고 판단해서는 안 된다.

### 두 frontier를 하나의 transaction으로 묶는다

GDN layer의 decode mutation을 다음처럼 기록한다.

```text
before: position=8, conv_frontier=8, recurrent_frontier=8
read: conv_state@8, recurrent_state@8
compute: new_conv, delta_update, output
commit accepted token:
  conv_frontier=9
  recurrent_frontier=9
  position=9
```

conv만 in-place로 먼저 갱신한 뒤 recurrent kernel이 실패하면 request state가 반쯤 전진한다. 재시도는 같은 token을 새 conv state에 두 번 넣을 수 있다. 안전한 구현은 temporary 또는 undo 정보, commit ordering, 실패 시 복원 규칙 가운데 하나를 갖는다. speculative decode에서 draft token 세 개를 미리 적용했다가 한 개만 accept하면 두 state를 accepted frontier까지 함께 crop 또는 reconstruct해야 한다.

prefix reuse도 Llama와 다르다. full attention layer의 KV block을 공유하는 것만으로 hybrid model prefix가 완성되지 않는다. 동일 prefix가 만든 conv/recurrent terminal state도 있어야 한다. terminal state를 재계산하면 KV reuse가 줄인 TTFT 이점 일부가 사라진다. state hash와 adapter/model identity, position semantics를 포함한 reuse key가 필요하다.

### vLLM vertical에서 확인할 support boundary

vLLM의 [`Qwen3_5DecoderLayer와 model/load path`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5.py#L118-L286)를 읽을 때 등록된 class만 확인하지 않는다. layer type이 실제 attention 또는 GDN object를 만드는지, runner가 두 state spec을 allocate하는지, prefill과 decode metadata가 specialized backend에 전달되는지, load path가 projection 이름과 packed layout을 맞추는지 본다.

optimized GDN backend가 특정 dtype, head dimension, chunk size만 지원할 수 있다. 조건을 벗어나면 의미가 같은 eager 경로로 가거나 명시적으로 거부해야 한다. 일반 attention backend에 Q/K/V 모양으로 억지 변환하는 것은 fallback이 아니다. graph capture도 recurrent state 주소와 mutation order가 안정적일 때만 가능하다. decode batch bucket이 같아도 각 request의 state pointer table이 달라진다.

### stale state 사건을 닫는다

fixture에 channel마다 식별 가능한 sentinel을 넣는다. prefill 마지막 conv lane에는 `800+c`, recurrent matrix 대각에는 `80+h`를 둔다. 첫 decode 입력은 `900+c`다. reference와 conv output이 같지만 recurrent readback부터 다르면 conv 가설을 기각한다. 두 readback은 같은데 output이 다르면 gate/delta kernel 또는 projection을 본다. output은 같고 state commit 뒤부터 다르면 mutation/rollback owner를 본다.

실제 수정은 conv와 recurrent state가 같은 generation ID를 갖게 하고 layer call이 두 mutation의 commit record를 반환하도록 하는 식이다. 재발 fixture는 prefill→첫 decode, speculative 3→1 accept, kernel 실패 뒤 retry를 모두 포함한다. Qwen3.5를 “attention이 적어 빠른 모델”로 소개하는 것보다 이 transaction을 이해하는 편이 서빙에 훨씬 중요하다.

## 52.10 모델 카탈로그 3: Gemma4 sliding·full·shared KV

Gemma4의 차이는 window 숫자 하나가 아니다. [`Gemma4TextConfig`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma4/configuration_gemma4.py#L123-L270)는 layer type과 sliding/full 구성을 materialize하는 입력이다.

[`Gemma4TextAttention`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma4/modeling_gemma4.py#L298-L481)은 attention type, projection과 KV source를 읽는 좌표이며, [`Gemma4TextModel.forward`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma4/modeling_gemma4.py#L637-L749)는 layer별 mask와 cache가 실제 loop에서 결합되는 경계다.

fixture는 `[sliding,full,sliding,full]` 네 layer다. window를 4로 둔다. position 8의 첫 decode에서 sliding layer가 허용하는 key는 새 token을 포함해 최근 네 position, 예를 들면 `{5,6,7,8}`이다. full layer는 `{0..8}`을 읽는다. physical cache가 더 많은 token을 보유해도 mask가 허용 범위를 정한다. 반대로 allocator가 window만 보존하는데 full mask를 주면 존재하지 않는 old token 주소를 읽게 된다.

### shared KV는 alias가 아니라 source relation이다

같은 attention type의 layer끼리 KV를 공유하도록 설계된 경우, consumer layer는 자신의 K/V를 새로 쓰는 대신 지정된 source layer의 state를 읽을 수 있다. “이전 layer KV”라고 축약하면 sliding과 full이 교대할 때 틀린 source를 고를 수 있다. 원장에는 `consumer_layer=2, attention_type=sliding, source_layer=0, source_generation=g`처럼 적는다.

source relation은 cache allocation 수를 줄일 수 있지만 lifetime coupling을 만든다. source layer state는 마지막 consumer가 끝날 때까지 살아야 한다. preemption이나 prefix eviction이 consumer만 보고 source block을 해제하면 dangling reference가 된다. copy-on-write가 필요한 mutation인지 read-only reuse인지도 분리한다. 같은 byte 주소를 공유한다는 구현 세부와 같은 semantic K/V를 사용한다는 계약은 동일하지 않다.

full layer 3이 full layer 1의 KV를 source로 쓴다고 하자. source layer가 만든 head semantics와 consumer projection이 기대하는 semantics가 맞아야 한다. 단지 `[B,Nkv,L,D]` shape가 같다고 공유할 수 없다. model source가 계산한 `shared_kv_source_layer`가 단일 진실이어야 하며 loader, cache allocator와 forward consumer가 같은 layer plan generation을 사용해야 한다.

### sliding cache의 byte를 두 가지로 구분한다

logical attention 범위가 window 4라고 physical storage가 반드시 네 token뿐인 것은 아니다. 구현은 full-length backing에 mask만 제한할 수도 있고 ring/sliding buffer로 오래된 slot을 재사용할 수도 있다. 전자는 byte가 context와 함께 늘지만 주소 해석이 단순하다. 후자는 byte가 bounded지만 logical position→physical slot modulo mapping과 overwrite 안전성이 필요하다.

두 방식을 benchmark 이름으로 추측하지 않는다. cache spec의 capacity, writer의 slot mapping, reader의 key range와 mask를 함께 본다. speculative token이 window 경계를 넘었다가 reject되면 덮어쓴 old slot을 복원할 수 있는지도 확인한다. accepted frontier만 되돌려서는 이미 overwrite된 payload가 돌아오지 않는다.

fixture에서 BF16 full K/V는 position 0~8에 대해 request당 `2×9×2×4×2=288 bytes`다. sliding logical live set 네 token은 128 bytes다. shared KV가 두 consumer의 별도 allocation을 제거하면 physical byte는 줄지만 consumer metadata와 source lifetime은 남는다. scheduler capacity 식은 layer count에 일률적인 288을 곱하지 않고 layer별 state spec과 unique source allocation을 합해야 한다.

### vLLM Gemma4 vertical과 expert layer

vLLM의 [`Gemma4 attention·MoE·decoder`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/gemma4.py#L224-L798)는 attention뿐 아니라 layer type에 따른 dense/expert 경계를 함께 보여 준다. 읽는 순서는 config-derived layer plan, attention object와 KV sharing, MLP 또는 MoE object, forward branch, load mapping이다. Transformers에 있는 field 이름을 vLLM이 같은 방식으로 소비한다고 가정하지 않는다.

backend gate는 full과 sliding 각각에 대해 판정한다. full attention을 지원하는 kernel이 sliding mask나 shared source ABI를 지원하지 않을 수 있다. 일부 layer만 fallback하면 한 batch 안에서 kernel family가 교대한다. graph capture는 layer sequence와 workspace 주소를 포함해야 하며 “Gemma4 graph” 하나로 뭉뚱그릴 수 없다.

### wrong window와 wrong source 사건

K position마다 값 `100×layer+position`을 넣는 sentinel을 쓴다. layer 0 sliding source의 positions 5~8과 layer 1 full source의 0~8을 구별한다. layer 2가 source 1을 읽으면 score에 100대 값 대신 200대 값이 나타난다. source는 맞지만 mask가 full이면 positions 0~4의 sentinel 기여가 생긴다. 두 오류를 한 attention output 차이로 묶지 않는다.

비교 순서는 layer plan, source layer ID, physical slot→logical position, allowed key set, backend output이다. source ID에서 다르면 mask kernel을 조사하지 않는다. source와 range가 맞고 output만 다르면 backend stride·scale을 본다. 수정 뒤에는 window보다 짧은 prompt, 정확히 window, window+1, shared source가 교대 type을 건너뛰는 경우를 fixture로 남긴다.


