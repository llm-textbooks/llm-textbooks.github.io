# 26장. 정적 배치에서 continuous batching으로

같은 GPU에 요청 A와 B를 함께 넣었다. A prompt는 1,000 tokens이고 output은 10, B prompt는 20이고 output은 200이다. 정적 batch가 두 요청을 처음부터 끝까지 묶으면 A는 B의 긴 decode가 끝날 때까지 batch row를 차지하고, B는 A의 긴 prefill padding과 계산에 끌려간다. GPU가 바쁜 것과 유효한 request work를 효율적으로 처리하는 것은 다르다.

Continuous batching은 token을 더 큰 tensor에 넣는 기법 하나가 아니다. 각 model iteration 뒤 finished requests를 제거하고 waiting requests를 admission하며, 다음 iteration에 실행할 token rows와 cache metadata를 다시 만드는 service-level schedule이다. 이 장은 그 전환을 latency·padding·shape·lifetime 관점에서 설명한다.

27장은 구체적인 token/sequence budget과 policy를, 28장은 waiting/running/preempted/finished 상태 기계를 소유한다. 이 장에서는 왜 iteration-level 재배치가 필요한지, physical batch가 어떻게 바뀌며 어떤 비용과 오류를 만드는지만 닫는다.

고정 source는 vLLM `v0.27.1` commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang `v0.5.18` commit `71de97b264b04dcd514cf904003028aefe9775c8`, Transformers `v5.15.1` commit `550d7b3834670483a4df436541272c055dc364bf`, llama.cpp `v0.2.0` commit `bb4caa7540188872173c44d161602d9271386413`이다. runtime은 실행하지 않는다.

## 26.1 정적 batch가 만드는 두 종류의 낭비

문제 장면을 숫자로 보자. A prompt length 8, B prompt length 2를 padded tensor `[2,8]`로 prefill한다. attention mask가 padding의 의미 누출은 막아도 dense embedding·projection·MLP가 B의 padding rows에 work를 할 수 있다. 유효 prompt tokens는 10인데 physical rows는 16이다. 단순 row 기준 utilization은 62.5%다.

packed/varlen prefill은 유효 10 rows를 flatten하고 cumulative lengths `[0,8,10]`으로 request boundaries를 보존할 수 있다. padding 6 rows를 줄이지만 metadata와 layout preparation이 필요하다. 모든 operator가 ragged representation을 직접 받는 것은 아니므로 model runner가 어느 축을 flatten하고 backend가 lengths를 소비하는지 확인한다.

decode에서 A가 1 token 뒤 EOS, B가 100 tokens를 더 생성한다고 하자. fixed batch loop는 A row에 pad를 넣으며 B가 끝날 때까지 shape `[2,...]`를 유지할 수 있다. 99 iterations 동안 A row는 semantic work를 하지 않는다. finished-row mask가 correctness를 지켜도 compute slot은 회수되지 않는다.

head-of-line blocking은 두 형태다. 긴 prompt와 짧은 prompt를 같은 prefill batch로 묶으면 짧은 요청의 first token이 긴 prefill completion을 기다린다. 긴 output과 짧은 output을 fixed generation batch로 묶으면 짧은 요청의 final return이 긴 row를 기다릴 수 있다.

static batch가 항상 나쁘다는 뜻은 아니다. shape가 고정돼 kernel launch, compilation과 CUDA Graph 재사용이 쉽고 dense GEMM이 큰 regular shape를 얻는다. requests가 비슷한 lengths이고 offline throughput이 목표면 padding cost보다 regularity 이득이 클 수 있다.

비유로는 식당의 단체상을 생각할 수 있다. 모든 손님이 식사를 끝내야 다음 팀이 앉는 것이 static batch다. 자리가 비는 즉시 새 손님을 앉히는 것이 continuous다. 그러나 GPU batch row는 의자처럼 독립적이지 않다. attention cache, positions, collective와 graph shape가 함께 재구성되므로 비유는 여기서 멈춘다.

비용 장부는 valid tokens, padded/finished dummy rows, model iterations, cache bytes, metadata/copy, graph variant와 launch를 나눈다. batch size 하나로 효율을 설명하지 않는다.

### 세 요청을 직접 시간축에 놓아 보기

추상적인 `batch size=2`보다 요청별 작업표가 훨씬 많은 것을 말해 준다. A는 시각 0에 도착하며 prompt 8개와 output 1개를 요구한다. B도 시각 0에 도착하지만 prompt 2개와 output 5개를 요구한다. C는 두 번째 decode가 시작되기 직전에 도착하며 prompt 3개와 output 2개를 요구한다고 하자. 이해를 위해 prefill token row 하나와 decode row 하나의 비용이 같다고 잠시 가정한다. 실제 attention 비용은 sequence length에 따라 달라지므로 이 가정은 절대 성능 계산이 아니라 낭비의 위치를 드러내는 자다.

정적 구현이 A와 B를 `[2,8]`로 묶으면 첫 forward의 물리 row 수는 16이고 유효 row는 10이다. 그 뒤 두 row를 고정한 채 decode를 다섯 번 돌면 물리 decode row는 10개다. A는 첫 decode에서 끝나므로 유효 decode row는 A의 1개와 B의 5개, 모두 6개다. 따라서 이 단순 장부에서 물리 작업은 26 rows, 유효 작업은 16 rows다. 10 rows, 즉 38.5%가 prompt padding 또는 finished padding이다. C는 이 batch가 완전히 끝난 뒤에야 새 batch를 얻는다면, GPU에 자리가 의미상 비어 있는데도 네 번의 B decode 동안 문 앞에서 기다린다.

continuous 방식은 A와 B의 initial prefill 표현이 padded인지 packed인지에 따라 첫 비용이 달라진다. packed라면 10 prompt rows다. 첫 decode에서 A와 B가 각각 한 token을 얻고 A가 종료되면 다음 iteration의 active set에는 B만 남는다. 바로 이때 C를 admission할 수 있다. 다음 iteration은 C의 prompt 3 rows와 B의 decode 1 row를 함께 다루거나, backend 제약에 따라 두 실행으로 나눌 수 있다. 어느 쪽이든 C는 B의 output 다섯 개가 모두 끝나기를 기다리지 않는다. 이득의 핵심은 “항상 한 kernel에 섞는다”가 아니라 완료된 lifetime이 다음 lifetime의 입장을 막지 않는다는 데 있다.

응답 시간도 따로 계산한다. 각 iteration이 한 시간 단위이고 initial prefill도 한 단위라고 단순화하면 정적 batch에서 A의 token은 두 번째 단위에 이미 계산되어도 API가 batch 전체 반환만 지원하면 여섯 번째 단위까지 전달되지 않을 수 있다. streaming 반환을 지원하면 A의 사용자는 일찍 받을 수 있지만, A의 physical row는 여전히 남는다. 사용자 체감 지연과 GPU 낭비가 반드시 동시에 생기지는 않는다는 뜻이다. C의 first token은 새 batch의 prefill 뒤에 나오므로 적어도 일곱 번째 단위가 된다. continuous 방식에서는 C admission이 가능한 boundary가 세 번째 단위 부근이므로 first token이 훨씬 앞당겨진다.

이 계산에는 일부러 감춘 비용이 있다. 긴 prompt의 prefill은 decode 한 token보다 비싸며, attention의 읽기 양도 다르다. mixed prefill/decode가 서로 다른 kernel 경로를 요구할 수 있고, batch를 다시 만드는 CPU 시간이 추가된다. 그래서 실측에서는 row count를 latency로 곧장 환산하지 않는다. 다만 row 장부는 “GPU utilization이 95%인데 왜 짧은 요청이 느린가”라는 질문에 첫 단서를 준다. 높은 utilization이 padding과 이미 끝난 row를 계산한 결과일 수도 있기 때문이다.

### 길이 bucketing이 해결하는 것과 남기는 것

정적 배치도 무방비로 서로 다른 길이를 섞을 필요는 없다. prompt 길이가 1~64, 65~128처럼 가까운 요청을 bucket으로 모으면 prefill padding을 줄일 수 있다. output length 예측치까지 이용하면 decode 종료 시점도 가까워진다. offline 평가처럼 모든 입력을 미리 알고 있으면 정렬 후 batch를 만드는 것만으로 상당한 이득을 얻는다.

그러나 online traffic에서는 bucket을 채우려고 기다린 시간이 새 queueing delay가 된다. 정확한 output length는 생성 전에는 모르며 `max_new_tokens=512`가 실제 9 tokens 종료를 뜻할 수도 있다. prompt 길이가 같아도 tool call, stop string, speculative path와 sampling 결과 때문에 종료 시점이 갈라진다. bucketing은 분산을 좁히지만 lifetime을 독립시키지는 않는다. 따라서 “길이 bucket을 썼으니 continuous가 필요 없다”와 “continuous이니 bucket이 필요 없다”는 둘 다 성급하다. 전자는 finished row 회수를 놓치고, 후자는 매 iteration의 shape locality를 놓친다.

운영자는 bucket 대기 시간을 별도 지표로 둬야 한다. arrival부터 bucket sealing까지, sealing부터 device submit까지, submit부터 first token까지를 나누면 length grouping이 compute를 줄인 대신 queue를 늘렸는지 보인다. 평균만 보면 조용한 시간대의 긴 대기가 높은 부하 시간대의 처리량 이득에 묻힌다. p50, p95와 요청 길이 구간을 함께 봐야 하는 이유다.

### 같은 workload를 세 방식으로 끝까지 세기

이번에는 계산을 조금 더 현실적으로 만든다. 요청 A는 시각 0에 도착하고 prompt 8, output 1이다. B는 시각 0에 도착하며 prompt 2, output 5다. C는 첫 decode가 끝난 직후 도착하고 prompt 3, output 2다. D는 그다음 iteration 직전에 도착하며 prompt 7, output 1이다. prompt row 비용을 단순히 1, decode row 비용을 1로 놓되, padded prefill은 최대 길이만큼 모든 request row를 실행한다고 가정한다. 이 모형은 attention의 제곱 비용을 생략하므로 절대 FLOPs를 나타내지 않는다. 대신 어느 요청 때문에 물리 row가 생겼는지를 정확히 보존한다.

첫 번째 방식은 A와 B를 완전히 닫은 뒤 C와 D를 처리하는 fixed static batch다. A/B prefill은 `2×8=16` physical rows, valid는 10이다. decode는 두 rows를 다섯 번 실행해 10 physical rows, valid는 6이다. 첫 묶음은 physical 26, valid 16이다. C/D prefill은 `2×7=14`, valid는 10이다. decode는 두 rows를 두 번 실행해 physical 4, valid 3이다. 둘째 묶음은 physical 18, valid 13이다. 전체는 physical 44, valid 29, dummy 또는 padding 15다. row utilization은 약 65.9%다.

요청 완료 시점은 더 불리하다. iteration을 prefill 한 번과 decode 한 번씩의 논리 단위로 세면 A/B prefill은 step 0, decode는 step 1~5다. A는 step 1에 의미상 끝나지만 fixed tensor에는 step 5까지 남는다. C와 D는 arrival가 빨라도 둘째 묶음 prefill이 step 6에 시작되고, C는 step 8에 끝난다. D는 step 7에 끝나지만 batch 전체 반환 방식이면 C를 기다린다. streaming이면 final 전달은 앞당길 수 있어도 둘째 decode의 D row 하나는 dummy가 된다.

두 번째 방식은 prompt 길이 bucketing을 하되 batch lifetime은 고정한다. A와 D의 prompt 길이 8과 7을 한 bucket, B와 C의 2와 3을 다른 bucket으로 묶고 싶다. 그러나 D와 C는 나중에 도착한다. bucket을 채우기 위해 A/B를 기다리게 하지 않는다면 결국 첫 static 묶음과 같다. 반대로 모든 요청이 올 때까지 기다리면 prefill physical rows는 A/D가 16에 valid 15, B/C가 6에 valid 5여서 padding이 7에서 2로 줄어든다. decode physical은 A/D가 2에 valid 2, B/C가 10에 valid 7이다. 전체 physical 34, valid 29, utilization 85.3%다. compute 장부는 좋아졌지만 A와 B는 C/D arrival까지 queue에서 기다렸다. online latency 목표에서는 이 대기 비용이 padding 절감보다 클 수 있다.

세 번째 방식은 packed prefill과 iteration-level membership 교체를 허용한다. step 0에서 A/B prompt valid 10 rows만 처리한다. step 1에서 A/B decode 두 rows를 실행한 뒤 A가 종료한다. C가 도착했다면 step 2 plan은 B decode 한 row와 C prefill 세 rows, 모두 네 valid rows다. D도 준비됐다고 하자. step 3은 B/C decode 두 rows와 D prefill 일곱 rows, 모두 아홉 valid rows다. step 4에서 B/C/D decode 세 rows를 실행하고 C와 D가 끝난다. 이후 B 한 row만 남아 다섯 번째 output까지 진행한다. 이 단순 일정에서는 physical과 valid가 모두 29 rows다.

continuous 일정이 항상 29 physical에 머무는 것은 아니다. graph bucket이 1, 2, 4, 8, 16이면 step별 row 수 10, 2, 4, 9, 3, 1은 각각 16, 2, 4, 16, 4, 1 capacity로 실행될 수 있다. graph 기준 physical capacity 합은 43이고 dummy capacity는 14다. packed logical row 관점에서는 낭비가 사라졌는데 graph regularity를 위해 여유가 돌아온다. 다만 fixed batch의 dummy는 긴 request가 끝날 때까지 특정 lifetime을 가두지만 graph dummy는 해당 iteration의 capacity rounding이다. 둘은 같은 rows라도 latency와 memory lifetime 의미가 다르다.

graph size를 token rows가 아니라 sequence count로 잡는 backend라면 숫자가 달라진다. step 3의 D prefill 일곱 rows와 B/C decode 두 rows가 flattened token 9개여도 sequences는 셋이다. 어떤 차원을 capture key로 쓰는지 확인하지 않고 산술을 운영 예측에 적용하면 안 된다. 손계산의 목적은 config를 고르는 것이 아니라 source와 metric에서 `size`의 단위를 캐묻는 습관을 만드는 것이다.

세 방식의 결론도 한 줄 순위가 아니다. 모든 입력이 미리 있고 반환 지연이 중요하지 않다면 bucketing static이 낮은 관리 복잡도로 좋은 utilization을 낸다. arrivals가 흩어지고 short request의 TTFT와 독립 종료가 중요하면 continuous가 queue와 lifetime을 분리한다. graph bucket 때문에 physical capacity가 다시 늘 수 있지만 admission 기회와 finished resource 회수는 유지된다. 성능 리뷰에서는 compute row, queue wait, response completion과 resource holding을 네 장부로 따로 비교해야 한다.

## 26.2 iteration은 request lifecycle을 재배치하는 안전점이다

autoregressive decode는 one or more scheduled tokens를 forward하고 outputs를 받은 뒤 다음 step을 결정한다. 이 iteration boundary에서 finish reasons가 확정되고 next input IDs와 cache lengths를 알 수 있다. continuous engine은 여기서 active set을 바꿀 수 있다.

iteration `t`의 active requests가 `{A,B,C}`이고 A가 finish했다고 하자. output processor는 A terminal을 client stream으로 보내고 scheduler/cache owner는 A resources를 safe point에 반환한다. next iteration은 `{B,C,D}`로 구성할 수 있다. D는 waiting에서 새로 admission됐다.

중요한 것은 D가 B/C의 logical sequence에 섞이지 않는다는 점이다. physical token rows는 compact돼도 request ID, logical position, block table, sampling state가 함께 remap된다. row index는 lifetime identity가 아니다.

iteration 중간에 arbitrary admission하지 않는 이유는 model input tensor와 cache metadata가 이미 만들어졌고 device work가 진행 중이기 때문이다. boundary에서 runner output과 scheduler state를 합의한 뒤 next batch를 만든다. exact boundary는 engine architecture에 따라 다르지만 ownership transition이 필요하다.

prefill과 decode를 한 iteration에 섞을 수도 있다. A는 prompt chunk 여러 rows, B/C는 decode one row를 제공한다. flattened physical batch에서 each row phase와 request/position mapping이 필요하다. mixed batch는 prefill/decode를 단순 batch dimension 두 그룹으로만 표현하지 않을 수 있다.

finished removal은 client delivery와 resource free를 구분한다. scheduler가 A를 finished로 만들었어도 final output이 transport에 전달되지 않았을 수 있다. 반대로 client disconnect는 scheduler가 current step을 drain한 뒤 remove할 수 있다. 21장 lifecycle과 이 장 batch membership을 연결한다.

iteration ledger는 step ID, active IDs before/after, scheduled token counts, flattened row ranges, finished/admitted IDs, cache blocks and runner shape를 갖는다. 27장의 budget 숫자보다 여기서는 membership transition을 본다.

### iteration plan은 작은 실행 계약이다

한 iteration의 plan을 단순한 ID 목록으로 생각하면 오류를 찾기 어렵다. plan에는 적어도 plan epoch, request identity, 이번에 처리할 logical token interval, input buffer의 physical interval, 이전 cache 길이, 실행 뒤 기대 cache 길이, logits를 읽을 row와 output destination이 묶여야 한다. runner는 이 계약으로 device 입력을 만들고, output processor는 같은 epoch의 계약으로 결과를 되돌린다.

예를 들어 flattened input이 `[A7, A8, C0, C1, C2, B3]`라면 row 순서만 보고 phase를 추측해서는 안 된다. A가 chunked prefill의 마지막 두 tokens인지, C가 새 prompt인지, B가 decode인지 metadata가 말해야 한다. logits가 필요한 row도 전부가 아닐 수 있다. A8, C2와 B3 뒤의 hidden state만 next-token selection에 필요하다면 gather 위치가 각각 1, 4, 5다. compaction이 input interval만 바꾸고 logits gather index를 바꾸지 않으면 다른 sequence의 분포를 읽는다.

plan 작성 시점과 소비 시점 사이에는 비동기성이 있다. host가 iteration `t+1` 계획을 준비하는 동안 device가 `t`를 실행할 수 있다. 겹침 자체는 좋은 최적화지만 mutable 배열 하나를 두 세대가 공유하면 current output이 next membership을 보게 된다. double buffering이나 immutable snapshot, epoch 검사가 필요한 이유다. 성능을 위한 overlap이 identity isolation을 깨뜨리지 않는지 확인한다.

새 요청 admission도 plan에 들어온 순간과 cache가 실제로 준비된 순간을 구분해야 한다. prefix cache hit가 있으면 logical prompt 전체를 다시 계산하지 않고 일부 suffix만 scheduled될 수 있다. 반대로 multimodal encoder 결과나 adapter 준비가 아직 끝나지 않았다면 waiting queue의 앞에 있어도 이번 plan에는 못 들어간다. 이 장의 관심사는 어떤 정책으로 골랐는지가 아니라, 선택 결과가 완전한 실행 계약으로 내려왔는가다.

### 종료는 한 점이 아니라 네 개의 경계다

“A가 끝났다”는 말에는 적어도 네 시점이 섞인다. model이 EOS 후보를 만든 시점, sampling/stop processor가 terminal을 확정한 시점, final chunk가 transport owner에게 전달된 시점, cache와 slot이 재사용 가능해진 시점이다. 정상 경로에서는 가까워 보여도 cancellation, backpressure와 asynchronous copy가 끼면 벌어진다.

batch membership에서는 두 번째 경계가 중요하다. 더 생성할 token이 없다고 확정된 A는 next model plan에서 빠질 수 있다. 하지만 output bytes가 socket으로 모두 쓰였다는 뜻은 아니다. transport queue가 final event를 소유하게 한 뒤 model slot을 반환할 수 있는 설계도 있고, output processor가 cache-backed state를 참조한다면 더 늦게 반환해야 하는 설계도 있다. source에서 객체 lifetime을 따라가야지 함수 이름 `finish`만 보고 free 시점을 추정해서는 안 된다.

cancellation은 반대 방향의 사례다. client가 연결을 끊어도 이미 제출한 CUDA work를 request 하나만 즉시 취소하기는 어렵다. current iteration의 결과를 버리고 boundary에서 membership을 제거하는 편이 흔하다. 이 짧은 drain 동안 metrics의 active와 사용자 관점 active가 다를 수 있다. disconnect timestamp, cancellation accepted, last submitted plan, resource release를 함께 남겨야 누수를 오진하지 않는다.

### boundary ordering을 사건 기록으로 재구성하기

iteration `t`가 시작될 때 active는 `[A,B,C]`, waiting은 `[D,E]`라고 하자. scheduler가 immutable plan `P_t`를 만들고 plan epoch와 request-to-row map을 확정한다. runner가 `P_t`를 받아 device buffers를 채운 뒤 submit한다. 이 순간부터 host active collection을 바꾸더라도 `P_t`가 참조하는 mapping은 바뀌면 안 된다. client가 B를 취소해도 current output row 1의 identity는 B다.

device completion 뒤 runner는 row-ordered output `O_t`와 `P_t` epoch를 돌려준다. output processor는 먼저 current map으로 A/B/C에 결과를 귀속한다. 그다음 stop 조건을 적용해 A가 EOS, B가 cancellation terminal, C가 계속 실행임을 확정한다. terminal event를 transport owner에게 넘긴 뒤 next membership 후보에서 A/B를 제외한다. cache release는 current work가 끝났고 후속 consumer가 old state를 더 보지 않는 경계에서 수행한다.

그 뒤에야 D/E admission을 고려한다. capacity상 D 하나만 들어온다고 하자. next active는 `[C,D]`이고 old C row 2는 new row 0으로 compact된다. D는 new row 1과 fresh cache mapping을 받는다. next plan `P_{t+1}`을 만들면서 C의 logical position과 sampling history는 유지하고 D의 position은 prompt 시작점으로 초기화한다. 마지막으로 inverse map을 검증하고 runner에 넘긴다.

output demux보다 compaction을 먼저 해 active 배열을 `[C,D]`로 바꾸면 `O_t[0]`을 C로, `O_t[1]`을 D로 해석할 위험이 있다. finished resource를 device completion보다 먼저 반환하면 D가 A의 slot을 받는 동안 old kernel이 같은 주소를 쓸 수 있다. admission 뒤 finish 처리를 하면 capacity 계산이 A/B를 여전히 active로 세어 D를 불필요하게 미룰 수 있다. ordering은 취향이 아니라 identity, memory safety와 latency에 동시에 걸린 계약이다.

모든 구현이 이 문장 순서 그대로 동기 실행하는 것은 아니다. output 처리와 next plan 준비를 pipeline하거나 deferred free queue를 두고 transport delivery를 다른 thread에 맡길 수 있다. 그래도 happens-before 관계는 보존해야 한다. current output은 current map으로 귀속된 뒤 map이 폐기되고, old slot의 last consumer가 끝난 뒤 new generation이 재사용하며, terminal 판정 뒤 request가 next plan에서 빠진다. source review에서는 코드 줄 순서가 아니라 lock, event, future와 ownership transfer가 이 관계를 만드는지 본다.

경계 로그에는 wall-clock timestamp만으로 부족하다. plan epoch, mapping generation, slot generation, device submission/completion event와 request terminal sequence를 함께 남긴다. 서로 다른 thread clock ordering이 애매해도 epoch 관계로 first divergence를 찾을 수 있다. 정상 fixture에서 `P_t→O_t→terminal(A,B)→compact(C)→admit(D)→P_{t+1}` 사슬이 재구성되어야 한다.

## 26.3 compaction은 tensor row와 request identity를 함께 옮긴다

세 requests의 next-token rows가 physical slots 0,1,2에 있고 slot 1 request가 finish했다고 하자. next batch는 slot 2를 slot 1로 compact할 수 있다. request C의 row index는 2에서 1로 바뀌지만 logical position, RNG와 output stream identity는 같다.

model input IDs뿐 아니라 positions, attention metadata, slot mapping, block table row, multimodal/recurrent state와 logits sampling params가 같은 permutation을 따라야 한다. 하나만 old order를 유지하면 shape가 맞은 cross-request 오류가 난다.

compaction은 physical copy일 수도 있고 index/select view, metadata rebuild일 수도 있다. runner가 persistent input buffers를 쓴다면 active prefix에 rows를 rewrite한다. CUDA Graph는 fixed capacity buffer와 active counts를 사용해 shape variants를 줄일 수 있다.

request ID→batch index map과 batch index→request ID 배열을 분리한다. inverse mapping consistency를 assertion으로 둔다. output batch rows를 request streams로 되돌릴 때 same mapping generation을 써야 한다. next iteration map을 current output에 적용하면 IDs가 교차한다.

slot reuse에는 generation/incarnation이 필요하다. A가 slot 4를 비우고 D가 받았는데 previous asynchronous output or cache write가 늦게 도착하면 D state를 오염시킬 수 있다. safe completion과 map epoch를 확인한다.

RNG state도 request-owned여야 한다. batch compaction이 global RNG consumption order를 바꾸면 neighbor arrival에 따라 sampling이 변할 수 있다. engine contract가 request-local generators를 쓰는지 source에서 본다.

compaction defect fixture는 A/B/C 중 B finish, C move, D admission이다. each state tensor에 digit-coded request marker를 둔다. next selected token and output stream이 expected ID로 돌아가는지 source permutation을 전개한다.

### 함께 이동해야 하는 상태를 층별로 세기

첫 층은 model input이다. token IDs, position IDs, sequence lengths, attention row boundaries와 logits selection indices가 있다. 둘째 층은 cache 주소다. block table, slot mapping, prefix sharing reference와 cache length가 있다. 셋째 층은 생성 의미다. temperature, top-p, repetition history, stop automaton, grammar state와 request-local RNG다. 넷째 층은 서비스 의미다. request ID, stream destination, tracing context, arrival timestamp, tenant label과 cancellation handle이다.

이 네 층이 반드시 하나의 거대한 struct에 있어야 한다는 뜻은 아니다. hot device metadata와 cold service metadata를 분리하는 편이 효율적일 수 있다. 중요한 것은 같은 permutation generation을 공유하는 것이다. 각 배열이 독립적으로 `remove(index)`를 수행하면 한 곳의 조건 분기 누락이 조용한 교차 오염을 만든다. 중앙의 old-to-new map을 만들고 모든 owner가 그것을 소비하거나, request ID를 key로 새 view를 재구축하는 방식이 검증하기 쉽다.

가령 old slots가 `[A,B,C]`, new slots가 `[A,C,D]`라면 old-to-new는 `0→0, 2→1`, D는 fresh slot 2다. token buffer는 `[A_token,C_token,D_token]`이지만 stop automaton이 `[A_state,B_state,D_state]`라면 C가 B의 stop phrase에서 종료한다. shape도 dtype도 정상이라 crash가 없다. 출력 문장이 이상하게 짧아지는 드문 현상으로만 나타난다. 이런 오류가 메모리 접근 위반보다 위험한 이유다.

검증 fixture에서는 각 상태를 서로 다른 표식으로 채운다. A 계열은 100번대, B는 200번대, C는 300번대 값을 쓴다. B 제거 뒤 new slot 1의 모든 request-owned field가 300번대인지 검사한다. 실제 token ID에 임의 숫자를 넣어 model을 돌릴 필요는 없다. mapping 함수와 metadata builder의 결과만 검사해도 permutation 결함 상당수를 잡는다.

### stable slot과 dense row를 혼동하지 않기

일부 구현은 request에 stable logical slot을 주고 매 iteration dense compute rows를 따로 만든다. 이때 logical slot 17의 C가 physical row 1에서 실행될 수 있다. 다른 구현은 active 배열 자체를 압축해 index가 바뀐다. 어느 설계든 가능하지만 로그에서 `slot`, `row`, `cache block`을 같은 숫자로 출력하면 디버깅이 망가진다.

stable slot은 service metadata 주소를 유지하고 cancellation lookup을 단순하게 한다. 반면 slot capacity만큼 큰 tensor를 그대로 실행하면 빈 공간 낭비가 생기므로 gather된 dense rows가 필요하다. dense active array는 compute locality가 좋지만 removal과 inverse map 관리가 잦다. source를 읽을 때 컨테이너 이름보다 어떤 index가 어떤 lifetime을 갖는지 질문해야 한다.

slot generation은 `(slot_id, generation)` 쌍으로 표현할 수 있다. A가 `(4,11)`을 쓰다 끝나고 D가 `(4,12)`를 받았다면 늦은 A output은 generation mismatch로 폐기된다. generation 없이 slot 4만 비교하면 정상처럼 보인다. asynchronous callback, distributed result와 transport event가 있는 시스템에서는 이 작은 숫자가 use-after-reuse를 막는 방화벽이다.

## 26.4 shape flexibility와 CUDA Graph regularity는 긴장한다

continuous batching은 active row 수와 prefill/decode mix가 step마다 달라진다. eager execution은 arbitrary shapes를 받아들이기 쉽지만 launch overhead와 compile specialization이 늘 수 있다. CUDA Graph는 captured addresses, control path와 supported shapes를 요구한다.

engine은 common batch sizes를 capture하고 actual active rows를 next supported size로 pad할 수 있다. 예를 들어 active 13 rows를 graph size 16으로 실행해 3 dummy rows를 감수한다. static max batch padding보다 작지만 waste가 사라진 것은 아니다.

fixed input buffers는 address stability를 제공한다. scheduler/runner는 current IDs, positions와 metadata를 buffer prefix에 copy하고 unused tail을 안전하게 mask/reset한다. stale tail을 kernel이 읽으면 cross-request leak이다.

graph variant 수를 늘리면 padding waste가 줄지만 capture/warmup memory와 dispatch complexity가 늘어난다. variant가 적으면 regular하지만 dummy work가 늘어난다. actual workload batch-size histogram으로 선택해야 한다.

prefill long rows와 decode short rows가 같은 captured shape/path를 쓸 수 있는지 backend eligibility에 달렸다. mixed batch가 graph fallback을 만들 수 있다. config에서 graph를 켰다는 사실과 each iteration actual graph replay를 분리한다.

compaction copy도 비용이다. input IDs 같은 작은 metadata는 cheap해 보여도 many tensors와 host-device synchronization이 쌓일 수 있다. persistent buffer update가 graph replay critical path에 들어간다. kernel time만 측정하면 scheduling gap을 놓친다.

correctness gate는 eager와 graph selected IDs/output parity, unused-tail isolation, slot reuse generation과 metadata mapping이다. performance는 graph hit ratio, padding rows, copy/launch gaps와 capture memory를 본다.

### graph bucket을 손으로 계산하기

capture sizes가 1, 2, 4, 8, 16, 32라고 하자. active rows 13은 16 graph를 골라 dummy 3 rows, 81.25% row utilization을 만든다. 다음 step에 다섯 요청이 종료해 active 8이면 정확히 8 graph를 쓴다. 새 요청 하나가 들어와 9가 되면 다시 16을 골라 dummy 7 rows, 56.25%가 된다. membership을 빨리 채우는 것이 항상 compute를 줄이지 않는 흥미로운 경계다.

반대로 capture sizes를 모든 정수 1~32로 만들면 dummy row는 거의 사라진다. 하지만 각 graph가 필요한 memory pool, warmup 시간, executable 관리와 검증 범위가 늘어난다. 실제 active histogram에서 9가 드물고 8과 16이 많다면 9 전용 graph는 가치가 작다. graph set은 maximum batch size만으로 정하는 옵션이 아니라 traffic 분포를 압축한 설계다.

padding row가 안전하려면 token ID 하나를 0으로 쓰는 것으로 충분하지 않다. length가 0인지, attention metadata가 어느 cache도 가리키지 않는지, logits output이 gather되지 않는지, collective에 포함될 때 값이 의미를 오염시키지 않는지 확인한다. persistent buffer tail은 이전 request의 valid 값이 남기 쉬우므로 매번 전부 zeroing할지, valid count로 kernel을 제한할지, dummy descriptor를 명시할지 source contract가 필요하다.

graph fallback도 하나의 사건으로 기록한다. `eligible=false`만 남기면 이유를 모른다. batch shape 미capture, prefill path, adapter 조합, speculative mode, unsupported attention metadata처럼 원인을 구분해야 한다. replay hit가 높아도 가장 긴 prefill만 fallback해 tail latency를 지배할 수 있으므로 request phase와 duration을 함께 본다.

### regularity를 위해 continuous의 이득을 되돌려 주는 경우

active 9를 16으로 pad하고, 새 요청을 받지 않은 채 8까지 내려오기를 기다리는 구현을 생각하자. graph efficiency는 좋아질 수 있지만 queue의 짧은 요청은 기다린다. 반대로 9를 즉시 실행하면 dummy 7 rows가 생긴다. 이 선택은 scheduler의 capacity 숫자만이 아니라 latency objective, graph bucket과 expected arrivals가 얽힌다. 구체적인 admission 정책은 다음 장의 몫이지만, 이 장에서는 물리 비용이 어디로 이동하는지 보아야 한다.

또 다른 경우는 compaction을 줄이려고 hole을 유지하는 것이다. stable slots 덕분에 host metadata copy는 줄지만 kernel이 sparse slot mask를 처리하거나 큰 capacity tensor를 읽는다. 매 step dense하게 압축하면 device work는 줄지만 host preparation이 늘어난다. workload가 short outputs 위주라 membership churn이 크면 compaction 비용이 커지고, long outputs 위주면 amortize된다. 한 번의 microbenchmark로 결론내기 어려운 이유다.

## 26.5 네 구현에서 batch owner를 찾는다

vLLM은 EngineCore scheduler가 requests를 schedule하고 model runner input이 flattened scheduled tokens와 attention/cache metadata를 만든다. 고정 source의 [`vllm/v1/core/sched/scheduler.py:1-360`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L1-L360)에서 schedule entry, running/waiting collections와 output update를 찾는다. 27·28장에서 policy/state를 깊게 다룬다.

runner batch update는 [`vllm/v1/worker/gpu_model_runner.py:1-420`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu_model_runner.py#L1-L420)에서 cached request state, input preparation과 execute 경계를 따라간다. exact symbol과 lines는 fixed tree에서 좁힌다.

SGLang 스케줄러는 요청 풀과 실행 배치를 반복마다 갱신한다. [`python/sglang/srt/managers/scheduler.py:1-420`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L1-L420)에서는 이벤트 루프, 배치 선택, 결과 처리를 찾는다.

그다음 [`schedule_batch.py:1-420`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_batch.py#L1-L420)에서 배치 객체와 필터링·병합을 읽는다.

Transformers continuous batching manager는 classic `generate`와 다른 process-level owner를 추가한다. 고정 source의 [`generation/continuous_batching/continuous_api.py:553-900`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/continuous_api.py#L553-L900)에서 incoming/cancel queues, active slots와 iteration을 찾는다. exact package path와 symbol은 v5.15.1 tree에서 확인한다.

llama.cpp server는 slots와 ubatch scheduling으로 requests를 model decode work에 묶는다. [`tools/server/server-context.cpp:1040-1600`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1040-L1600)에서 task/slot update와 decode call을, [`src/llama-batch.cpp:1-300`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-batch.cpp#L1-L300)에서 batch/ubatch representation을 찾는다.

네 구현의 class 이름보다 공통 owner를 찾는다. incoming requests, active membership, iteration plan, runner input mapping, output-to-request mapping, finished removal과 slot/cache reuse다. 빈 owner가 있으면 lifecycle source를 더 따라간다.

### 네 구현의 차이는 membership을 어디에 물질화하는가에서 시작한다

vLLM 독해의 중심은 scheduler output과 runner의 cached request state 사이 경계다. scheduler가 “이번 step에 각 request가 몇 tokens를 전진하는가”를 정하면 runner는 기존 device-facing state에 delta를 반영해 flattened inputs를 만든다. 따라서 active collection의 Python 순서가 곧 tensor row 순서라고 가정하지 않는다. source walk에서는 scheduler request ID, runner cached state index, input row range와 sampled output index가 이어지는 지점을 찾는다. scheduler가 내린 논리 계획과 runner가 선택한 graph capacity도 서로 다른 단위다.

SGLang은 long-lived scheduler process와 `ScheduleBatch` 계열 객체의 변환이 눈에 띈다. waiting requests를 running batch에 합치고 finished를 filter하는 작업이 batch 객체의 metadata 재구성과 맞물린다. prefix/cache pool과 request pool의 index가 물리 model rows와 어떻게 연결되는지 보아야 한다. vLLM과 같은 이름의 continuous batching이라도 객체 lifetime과 pool indirection이 다르므로 한 구현의 mental model을 그대로 이식하지 않는다.

Transformers continuous manager는 library 사용자가 익숙한 `generate()`의 closed batch 밖에 arrival와 cancellation owner를 세운다는 점이 교육적으로 중요하다. classic generation은 호출 당시 tensor batch가 세계의 전부지만 manager는 여러 request lifetime을 slot에 올리고 내린다. 그래서 model generation utilities만 읽어서는 queue wait와 slot reuse를 볼 수 없다. 반대로 manager만 읽으면 cache update와 logits processing이 기존 generation machinery에 위임되는 경계를 놓친다. 두 층의 접합부가 continuous service가 library loop에 추가하는 책임을 드러낸다.

llama.cpp server에서는 server slot이 사용자 request lifetime을 대표하지만, `llama_batch`와 내부 micro-batch는 compute work를 대표한다. 한 slot의 prompt가 여러 micro-batches로 갈라질 수도 있고 여러 slots의 tokens가 public batch에 함께 들어갈 수도 있다. 따라서 slot `n_batch` 설정, prompt processing chunk와 concurrent slots를 하나의 batch-size 옵션처럼 설명하면 틀린다. CPU-side graph construction과 backend execution의 경계도 Python scheduler 기반 두 구현과 다르게 보이지만, output row를 slot identity로 되돌려야 한다는 불변식은 같다.

네 구현을 비교할 때 가장 좋은 질문은 “continuous batching을 지원하는가”가 아니다. 새 arrival가 어느 event loop에서 보이는가, current execution plan은 immutable한가, active removal은 어떤 객체가 수행하는가, cache ownership은 slot/request 중 어디에 붙는가, output order를 service identity로 누가 되돌리는가, graph 또는 micro-batch shape를 누가 결정하는가를 묻는다. 답의 위치가 다르면 metric을 심을 위치와 race가 생길 위치도 달라진다.

source 차이를 성능 순위로 곧장 번역해서도 안 된다. pool indirection은 추가 lookup처럼 보이지만 allocation 안정성을 줄 수 있고, cached request state는 update 복잡성을 늘리지만 persistent buffers를 가능하게 한다. manager abstraction은 overhead가 있어도 cancellation과 fairness의 명시적 owner를 준다. slot 기반 server는 단순해 보여도 micro-batch와 cache shift가 별도 축을 만든다. 구현 선택은 workload와 관측 가능성까지 묶어 평가한다.

### source를 읽는 순서는 호출 방향과 반환 방향을 왕복한다

첫 번째 독해에서는 요청이 들어오는 방향으로만 간다. queue에 들어간 request가 active collection에 편입되고, scheduled token descriptor가 runner input으로 바뀌며, model forward가 호출되는 지점을 잇는다. 두 번째 독해는 output에서 거꾸로 올라온다. device row의 logits가 어느 request 결과가 되고, finish 판단이 어디서 생기며, active collection에서 언제 제거되는지 찾는다. 두 경로가 같은 request identity와 iteration plan에서 만나야 한다.

vLLM에서는 스케줄러 파일 하나만 읽고 연속 배치를 이해했다고 말할 수 없다. 스케줄 결과를 실행기가 어떤 캐시 요청 상태와 합치는지, 실행기 출력을 스케줄러가 어떻게 반영하는지까지 이어야 한다.

[`scheduler.py:361-760`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py#L361-L760)에서 후속 스케줄링과 갱신 경계를 좁힌다. 이어 [`gpu_model_runner.py:421-900`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu_model_runner.py#L421-L900)에서 요청 상태 갱신과 입력 준비의 연결을 확인한다.

그래프 디스패치는 같은 파일의 [`gpu_model_runner.py:901-1320`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu_model_runner.py#L901-L1320)까지 내려가 실제 모양 선택과 버퍼 수명을 찾는다.

SGLang은 scheduler loop와 batch 객체의 책임을 나눠 읽는다. loop가 언제 새 batch를 만들고 running batch를 갱신하는지 [`scheduler.py:421-900`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/scheduler.py#L421-L900)에서 추적한다. batch filtering과 merge가 어떤 request-owned 배열을 옮기는지는 [`schedule_batch.py:421-900`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/schedule_batch.py#L421-L900)에서 확인한다.

이름이 `filter`라고 해서 Python list만 줄이는 것으로 가정하지 않고 cache와 sampling metadata의 동행을 본다. model worker로 내려가는 경계는 [`tp_worker.py:1-360`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/managers/tp_worker.py#L1-L360)에서 찾아 scheduler와 device runner 사이의 ownership transfer를 표시한다.

Transformers 매니저는 고전 생성 루프와 서비스형 매니저를 비교하기 좋은 기준점이다. [`continuous_api.py:901-1082`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/continuous_api.py#L901-L1082)에서 활성 모음의 갱신과 반복 결과 처리를 찾는다.

같은 디렉터리의 [`requests.py:1-320`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/requests.py#L1-L320)에서는 요청이 소유하는 상태의 범위를 확인한다. 고전 경로의 [`utils.py:2500-3100`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/utils.py#L2500-L3100)와 대조하면, 한 번의 호출에서 배치 차원을 유지하는 루프에 외부 도착 소유자가 왜 없는지 보인다.

llama.cpp에서는 HTTP 작업, 서버 슬롯, `llama_batch`, 내부 마이크로배치를 구분한다. [`server-context.cpp:1601-2200`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1601-L2200)에서 슬롯 진행과 결과 반환을 따라간다.

[`llama-batch.cpp:301-620`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-batch.cpp#L301-L620)에서는 공개 배치 표현이 내부 실행 단위로 어떻게 다뤄지는지 본다. 그래프 구축과 계산 스케줄링의 더 아래 층은 [`llama-context.cpp:1-420`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-context.cpp#L1-L420)을 연결한다.

서버 슬롯 수와 CUDA 커널의 배치 차원을 같은 값으로 읽지 않는 것이 핵심이다.

이 source walk에서 표로 남길 열은 구현 이름, arrival owner, active membership owner, physical row builder, cache mapping owner, output demultiplexer, finish detector, resource releaser, graph shape selector다. 한 symbol이 여러 열을 소유할 수도 있고 여러 파일에 분산될 수도 있다. 빈 칸은 “기능 없음”이 아니라 아직 호출을 덜 따라갔다는 표시다.

### 이름이 같은 batch가 같은 단위를 뜻하지 않는다

API 문서의 batch는 동시에 제출한 사용자 요청 묶음일 수 있다. scheduler의 batch는 이번 iteration에 선택한 sequence 묶음이며, runner의 batch는 flattened token rows다. attention backend의 batch는 cumulative sequence offsets로 정의된 ragged sequences일 수 있고, GEMM의 첫 차원은 그 rows를 다시 합친 token 수다. llama.cpp의 micro-batch는 graph나 memory 제약 때문에 public batch를 더 쪼갠 실행 단위다.

로그의 `batch_size=16`만 비교하면 네 구현을 잘못 비교한다. 16 requests가 각 한 decode token을 낸 것인지, 한 request의 16 prompt tokens인지, active capacity가 16인데 valid rows가 3인지 밝혀야 한다. source에서 변수 선언과 shape construction까지 내려가 단위를 붙인다. 문서에서는 `request batch`, `sequence set`, `scheduled token rows`, `graph capacity`처럼 의도적으로 긴 이름을 쓴다.

Transformers classic generate는 fixed invocation batch가 owner라 이 표의 incoming/cross-request admission이 없다. continuous manager가 이를 별도 queue와 slot state로 추가한다. 23장과의 경계가 여기다.

### source review를 한 request의 이름으로 고정하기

큰 codebase를 읽다가 길을 잃는 가장 흔한 이유는 모든 class를 위에서 아래로 읽기 때문이다. 대신 가상 request `req-C` 하나를 정하고 이름이 바뀌는 지점을 기록한다. ingress object의 external ID, scheduler request object, active collection key, runner cached state key, physical row interval, cache slot과 output destination이 한 줄의 계보를 이룬다. source가 ID 대신 index만 넘기면 inverse mapping owner를 찾는다.

vLLM walk에서는 `req-C`가 waiting에서 schedule result에 나타난 첫 epoch를 표시한다. runner가 이전 epoch cached state와 새 schedule delta를 합칠 때 token interval과 physical input slice를 적는다. forward 뒤 sampled token row가 어떤 배열을 거쳐 scheduler update의 `req-C`로 돌아오는지 역추적한다. finish가 되면 running collection, cache allocation과 runner cached state에서 각각 언제 사라지는지 서로 다른 선으로 표시한다.

SGLang walk에서는 `req-C`가 request pool index와 running batch 내부 위치를 얻는 순간을 나눈다. batch merge 전후 위치, filter 후 permutation, cache 관련 index와 logits result position을 이어 본다. scheduler loop가 새 request를 볼 수 있는 시점과 model worker가 이미 받은 batch의 불변 경계를 표시한다. pool index가 stable해도 batch position은 변할 수 있다.

Transformers manager walk에서는 incoming request가 active slot을 얻기 전후를 가른다. classic generation configuration에서 파생된 sampling/stop state가 slot과 함께 움직이는지, cancellation queue event가 current iteration과 next membership 중 어디에 반영되는지 본다. model/cache layer가 반환한 batch-ordered 결과를 manager가 request future나 stream으로 되돌리는 순간을 찾는다.

llama.cpp walk에서는 external task ID, server slot ID, `llama_batch` sequence identity와 micro-batch row를 네 칸에 둔다. `req-C` prompt가 여러 decode calls에 걸쳐 처리되면 같은 slot lifetime 아래 compute work가 어떻게 잘리는지 기록한다. 다른 slot token과 함께 batch에 들어갈 때 output logits가 어느 slot로 돌아오는지도 확인한다.

이 방식의 장점은 구현 간 class 대응표를 억지로 만들지 않는다는 것이다. 네 codebase의 객체 구조는 달라도 arrival, admission, physical execution, output attribution, terminal과 release라는 사건은 비교할 수 있다. 어떤 구현에서 사건 사이 owner가 둘로 나뉜다면 그 자체가 중요한 설계 차이다. asynchronous callback이 있다면 identity와 epoch가 그 경계를 어떻게 건너는지 본다.

리뷰 노트에는 주장과 증거를 분리한다. “finished request는 다음 iteration에서 제거된다”라는 주장 옆에는 terminal 판정 call site, active filter call site와 next input builder의 고정 source link를 각각 둔다. 단일 함수 이름만으로 전체 lifecycle을 증명하지 않는다. source가 변경될 때는 세 anchor가 여전히 같은 happens-before 관계를 만드는지 재검토한다.

### 계산표를 실제 trace와 대조할 때 생기는 차이

앞의 row 계산은 교육용으로 모든 token 비용을 같게 놓았다. 실제 trace에서는 prefill attention, projection과 MLP가 token rows에 비례하는 부분과 sequence length 구조에 민감한 부분을 나눈다. decode는 한 새 token만 입력해도 기존 KV를 읽으므로 active sequence 길이에 따라 memory traffic이 달라진다. valid row 수가 같아도 duration이 같지 않다.

packed prefill도 “padding 0”이라는 한 문장으로 끝나지 않는다. embedding과 MLP는 flattened valid rows만 보더라도 attention backend가 cumulative lengths를 소비하고, output gather와 cache write가 sequence boundary를 복원한다. metadata 준비와 alignment 여유가 있을 수 있다. operator별 tensor shape와 kernel path를 보고 logical padding 제거가 device work 어디까지 전달됐는지 확인한다.

continuous mixed iteration은 phase 간 간섭도 만든다. long prefill rows가 decode rows와 함께 들어가면 decode 사용자의 inter-token gap이 늘 수 있다. 반대로 decode만 우선하면 waiting prefill이 늙는다. 이 장은 구체 우선순위를 정하지 않지만, iteration record에 phase별 valid rows와 request completion을 함께 넣어야 다음 장의 policy 논의가 근거를 갖는다.

distributed execution에서는 한 rank만 active count를 다르게 이해해도 collective shape가 어긋난다. membership plan과 selected capacity가 workers에 일관되게 전달되는지, dummy rows가 모든 ranks에서 같은 의미인지 본다. service identity는 coordinator가 소유하더라도 device row contract는 participating workers가 공유해야 한다.

마지막 차이는 scheduler overhead의 상대 크기다. 큰 model에서는 수십 microseconds의 host mapping이 작아 보이지만 작은 model, 높은 decode rate와 작은 active set에서는 iteration마다 반복되어 비중이 커진다. 반대로 compaction을 생략해 dummy row가 늘면 큰 model에서 비용이 훨씬 커질 수 있다. 모델 크기와 workload 없이 “continuous overhead는 무시 가능하다” 또는 “compaction은 비싸다”라고 단정하지 않는다.

## 26.6 관측은 request 수보다 유효 token work를 본다

batch size 32가 높다고 효율적이라 말할 수 없다. 32 rows 중 finished dummy 20, padding 10이면 valid work는 2일 수 있다. physical scheduled rows, valid prefill/decode tokens, dummy/padding과 accepted outputs를 나눈다.

timeline에는 iteration start/end, active IDs before/after, admitted/finished, per-request scheduled rows, runner shape, graph size/hit와 output commit을 둔다. request-level TTFT/ITL과 iteration duration을 연결한다.

head-of-line 증거는 짧은 request가 어느 긴 operation을 기다렸는지다. arrival, batch formation, prefill start/end, first output을 본다. static grouping wait와 scheduler queue wait를 구분한다.

padding waste는 logical token count와 physical tensor/graph rows 차이다. attention backend가 varlen으로 padding score를 skip해도 MLP/GEMM rows가 padded인지 operator별로 본다. 하나의 padding ratio를 model 전체 FLOP waste로 직접 쓰지 않는다.

compaction cost는 host preparation, H2D metadata copy, layout kernels와 event-loop gap에 나타날 수 있다. model kernels 사이 CPU gap과 buffer update range를 본다. source에 copy가 있다고 expensive라고 단정하지 않는다.

graph tradeoff는 replay hit ratio, selected graph size-active rows, fallback reasons, capture memory와 latency를 함께 본다. average utilization만 보면 dummy work와 stable low latency를 구분하지 못한다.

fairness와 budget policy는 27장 owner다. 이 장에서는 active set 변화가 특정 request를 오래 제외하는 증상만 handoff한다. state transitions 상세는 28장으로 넘긴다.

### 한 번의 관측 세션으로 네 장부를 맞추기

운영자가 처음 보는 것은 흔히 request latency histogram이다. p95가 나빠졌다면 곧장 batch size를 바꾸기 전에 같은 시간 구간의 request, queue, iteration과 physical work를 연결한다. request record에는 arrival, tokenize 완료, waiting 진입, admission, first scheduled token, first output, terminal과 resource release가 있다. iteration record에는 epoch, active before/after, admitted/finished IDs, valid prefill/decode rows, selected execution capacity, graph/eager path와 duration이 있다.

시각 12:00:00~12:01:00에서 C의 TTFT가 800ms라고 하자. arrival부터 waiting 진입이 5ms이고 waiting부터 admission이 700ms라면 tokenizer나 model forward부터 파는 것은 순서가 틀렸다. 같은 구간 iteration은 40ms씩 정상이고 graph replay도 안정적이다. active before/after를 보면 finished IDs가 있는데 admitted가 세 epochs 뒤에야 나타난다. 이때 질문은 CUDA kernel이 느린가가 아니라 membership owner가 어느 경계에서 queue를 다시 보는가다.

반대로 waiting은 2ms인데 admission부터 first output이 700ms라면 physical plan을 본다. C의 prompt가 long prefill과 함께 padded됐는지, chunked/mixed plan에서 어느 rows를 받았는지, graph fallback과 runner preparation gap이 있었는지 나눈다. model forward duration이 길어도 C 자체 prompt 때문인지 neighbor의 physical rows 때문인지 schedule row range가 있어야 답할 수 있다.

terminal부터 release가 길면 사용자 latency histogram에는 안 보일 수 있다. final output은 이미 갔지만 cache blocks가 묶여 다음 request admission을 늦춘다. 그래서 resource holding time은 terminal timestamp와 allocator release timestamp 차이로 측정한다. 이 값이 증가할 때 transport queue, asynchronous device consumer와 deferred cleanup owner를 연결한다. “요청이 끝났다”는 한 counter로는 누수와 정상 drain을 구별할 수 없다.

physical utilization도 두 분모로 본다. `valid token rows / submitted token rows`는 padding·finished rows를 드러내고, `submitted rows / selected graph capacity`는 graph dummy를 드러낸다. valid 9, submitted 9, graph capacity 16이면 packing은 완벽하지만 graph occupancy는 56.25%다. valid 9, submitted 16, capacity 16이면 padding이 문제다. 두 경우의 remedy가 다른데 GPU utilization 하나로는 같아 보일 수 있다.

compaction overhead는 event-loop gap에 숨는다. iteration `t` device completion부터 `t+1` submit까지 시간을 재고, 그 안의 output processing, finish removal, mapping rebuild, metadata copy와 admission을 span으로 나눈다. active churn이 높은 구간에서 mapping rebuild만 커지는지 본다. 하지만 span이 길다는 이유로 compaction을 제거하지 않는다. 그 시간이 줄이는 dummy device work와 queue latency를 함께 비교한다. host 50µs를 아껴 device 500µs dummy work를 늘리는 최적화는 손해일 수 있다.

관측 label의 cardinality도 조심한다. request ID와 raw prompt를 Prometheus label에 넣지 않는다. request-level trace에는 sampling으로 identity를 보존하고, aggregate metric은 batch bucket, phase, execution path와 finish reason처럼 제한된 차원을 쓴다. first divergence 조사 때는 trace epoch와 structured log를 조인한다. 높은 cardinality가 metric backend를 무너뜨리면 정작 incident 순간 데이터가 사라진다.

### 반사실 비교가 옵션 이름보다 강하다

같은 captured request trace를 static, length-bucketed, continuous planner에 replay한다고 생각하자. model은 실행하지 않아도 arrival, prompt length, observed output length와 termination epoch로 membership simulation을 만들 수 있다. static 반사실은 batch sealing과 last-request completion을, bucketing은 bucket wait와 padding을, continuous는 iteration admission/removal과 graph rounding을 계산한다. 실제 content와 logits가 없어도 scheduling 손익의 상당 부분을 비교할 수 있다.

반사실 결과에서 continuous valid rows가 가장 적지만 queue p95가 더 길다면 planner가 new prefill을 거의 admission하지 않은 것이다. 이것은 continuous 개념의 실패가 아니라 구체 정책과 workload 상호작용이다. 반대로 static padding이 낮아도 short request final latency가 길면 batch 반환 또는 lifetime 결합을 의심한다. 하나의 throughput 숫자 대신 동일 arrivals에 대한 completion curve를 그리면 누가 누구를 기다리게 했는지 보인다.

검증에서 가장 중요한 것은 입력 trace의 동일성이다. 서로 다른 시간대 traffic을 비교하거나 cache hit가 다른 실행을 비교하면 batching 효과와 workload 효과가 섞인다. prompt length, output length, arrival gaps, adapter/model route, prefix hit와 cancellation을 고정한다. graph warmup과 capture 비용을 포함할지 steady-state만 볼지도 명시한다. 결과가 좋아도 어떤 장부에서 좋아졌는지 설명하지 못하면 설정 변경 근거로 약하다.

### 독자가 직접 그리는 두 개의 그림

첫 그림은 request Gantt다. 가로축은 iteration, 세로축은 A/B/C/D다. 각 칸에 waiting, prefill, decode, terminal, released를 색으로 나누고 static과 continuous를 위아래에 그린다. static 그림에서 A의 terminal 뒤에도 batch lifetime 막대가 B 끝까지 이어지는지, continuous 그림에서 C가 그 빈 경계에 들어오는지 본다. 이 그림은 head-of-line을 latency 언어로 보여 준다.

둘째 그림은 physical row strip이다. iteration마다 valid prefill은 진한 파랑, valid decode는 진한 초록, padding은 회색, graph dummy는 빗금으로 그린다. 첫 그림과 세로선을 맞추면 C의 빠른 admission을 위해 graph dummy가 늘었는지, compaction 후 submitted rows가 줄었는지 동시에 보인다. request count만 그린 dashboard보다 scheduler와 CUDA shape의 접점을 훨씬 잘 드러낸다.

그림에 cache block 수와 host gap을 모두 우겨 넣지는 않는다. 별도 작은 track으로 active cache bytes와 completion-to-next-submit gap을 둔다. 시각화의 목적은 모든 metric을 한 화면에 넣는 것이 아니라 causal ordering을 잃지 않는 것이다. terminal 선 뒤 cache가 내려오지 않으면 finish race가 보이고, active 변화 때 host gap이 튀면 compaction cost가 보인다.

## 26.7 결함 주입은 identity와 boundary를 깨뜨린다

첫 결함은 wrong permutation이다. B finish 뒤 C row를 compact했지만 sampling params는 old index에 남는다. C가 B의 temperature/stop을 사용한다. raw model output는 C지만 selected token/finish부터 다르다.

둘째는 stale graph tail이다. active rows 13을 graph 16 buffer에 넣고 unused 13~15를 reset하지 않는다. kernel이 length/mask 오류로 old request tail을 읽는다. digit-coded dummy values와 valid length assertion으로 잡는다.

셋째는 output map generation 오류다. current iteration outputs를 next iteration compacted map으로 demux해 C token을 D stream에 보낸다. output batch plan ID와 request IDs를 함께 운반해야 한다.

넷째는 early free다. A finished output을 받자 cache blocks/slot을 재사용했지만 device work or output processor가 old state를 참조한다. safe completion과 deferred free owner를 본다.

다섯째는 graph fallback storm이다. request mix가 captured sizes/paths를 계속 벗어나 eager/capture switching overhead가 늘어난다. graph enabled metric만으로 hit를 추정하지 않는다.

여섯째는 admission starvation이다. decode active set이 항상 capacity를 차지해 new prefill이 들어오지 못하거나 long prefill이 decode step을 늘린다. exact budget policy는 27장으로 넘기되 symptom과 membership timeline을 제공한다.

### 장애 1: 짧은 요청이 GPU가 한가한데도 줄에서 늙는다

증상은 명확하다. 요청 C는 prompt 12, output 3으로 짧고 도착 당시 GPU memory에도 여유가 있다. 그런데 TTFT가 4초를 넘는다. dashboard의 GPU utilization은 55%이고 active requests는 maximum보다 작다. 처음 떠오르는 가설은 network ingress 지연, tokenizer 정체, graph compile 또는 긴 prompt다. 하지만 C의 tokenize end는 arrival 직후이고 prompt 길이도 정상이다.

관측을 request timeline과 iteration timeline으로 겹친다. C는 waiting queue에 시각 10.1초에 들어갔지만 admission은 14.0초다. 그 사이 active set에는 A/B가 있고 A는 10.3초에 이미 EOS를 냈다. 그런데 membership 로그에는 A가 14.0초까지 남아 있다. GPU trace는 B의 decode one-row work와 A의 masked dummy row를 계속 실행한다. first divergence는 model latency가 아니라 A terminal 뒤 next plan에서 A가 제거되지 않은 boundary다.

“static batch라 원래 그렇다”는 가설을 반증하려면 설정 이름만 보지 않는다. source에서 scheduler가 new arrival를 iteration마다 검사하는지, active collection이 batch 전체 종료 조건에 묶였는지 본다. 이 사례에서는 wrapper가 service engine의 continuous API가 아니라 classic fixed-batch generation을 호출하고 있었다. server의 concurrency setting은 여러 fixed invocations를 허용했지만 invocation 내부 membership은 바뀌지 않았다. memory 여유와 active count가 있어 보인 이유다.

복구는 무조건 batch size를 키우는 것이 아니다. request lifetime을 독립적으로 소유하는 manager/engine 경로로 전환하고, A terminal 다음 plan에서 active removal과 C admission이 일어나는지 검증한다. 회귀 fixture는 A `(8,1)`, B `(2,50)`, C late arrival `(3,2)`를 사용한다. 합격 조건은 C admission epoch가 A terminal 다음 safe boundary를 넘겨 불필요하게 미뤄지지 않는 것이다. GPU utilization 상승은 부차적 결과이며 핵심은 queue age가 batch partner의 output length에 묶이지 않는 것이다.

### 장애 2: compaction 뒤 특정 stop 조건이 다른 사용자에게 옮겨 간다

증상은 crash가 아니라 드문 조기 종료다. 세 requests를 함께 처리할 때 B가 끝난 직후 C가 원래 갖지 않은 stop phrase에서 종료한다. 같은 C를 단독 실행하면 재현되지 않고, temperature를 0으로 해도 batch 조합에 따라 달라진다. model weight나 floating-point nondeterminism을 의심하기 쉽지만 raw logits top candidates는 단독 실행과 일치한다.

관측은 model output 전후를 나눈다. iteration 41의 raw logits row 2는 C에 맞고 selected token도 정상이다. iteration 42에서 C가 old row 2에서 new row 1로 이동했다. token IDs, positions와 block table은 C marker를 갖지만 stop automaton 배열의 row 1에는 B marker가 남았다. first divergence는 compaction plan 적용 때 sampling/stop state owner가 permutation consumer 목록에서 빠진 지점이다. 오류가 logits 뒤에서 생기므로 kernel과 cache corruption 가설을 반증할 수 있다.

추가 반증으로 C와 B의 stop configuration을 같게 하면 장애가 사라지고, B가 아닌 A를 종료시켜 C가 이동하지 않게 해도 사라진다. graph를 끄거나 attention backend를 바꿔도 유지된다. 이 결과는 shape나 CUDA execution보다 row identity permutation을 가리킨다. request ID만 찍은 로그로는 모든 줄이 C처럼 보여 놓칠 수 있으므로 각 state bundle의 mapping generation을 기록한다.

복구는 stop state 한 배열에 즉석 `index_select`를 추가하고 끝내지 않는다. request-owned state inventory를 만들고 중앙 permutation map을 모든 consumer에 적용한다. output demux가 current generation을 사용하는지, RNG와 grammar state도 같은 결함을 갖는지 함께 감사한다. digit-coded fixture에서 `[A,B,C]→[A,C,D]` 뒤 new row 1의 model/cache/generation/service 네 층이 모두 C generation인지 검사한다. 이후 fault injection으로 한 consumer를 일부러 old map에 남겨 assertion이 first bad iteration에서 실패하는지 확인한다.

### 장애 3: final token은 갔는데 slot이 풀리지 않는다

증상은 시간이 지날수록 admission capacity가 줄어드는 것처럼 보이는 현상이다. clients는 final response를 정상 수신하지만 active slots metric은 내려오지 않는다. queue wait가 점차 길어지고 재시작하면 회복된다. cache leak, transport backpressure, cancellation race와 metrics bug가 경쟁 가설이다.

request 하나의 terminal 사슬을 따라간다. stop processor는 epoch 88에서 finish를 확정했고 transport final event도 전송했다. scheduler active collection에서도 빠졌다. 그러나 cache allocator의 reference count와 slot generation은 그대로다. release callback log가 없고, 해당 request에서만 client disconnect와 final write completion이 거의 동시에 일어났다. first divergence는 두 terminal 경로가 모두 “상대 경로가 release할 것”이라고 판단해 deferred free enqueue를 생략한 분기다.

metrics bug를 반증하려고 allocator free-block count와 새 request allocation 실패를 함께 본다. 실제 block이 반환되지 않아 관측 오류가 아니다. device work가 늦게 끝난다는 가설은 completion event가 release 조건보다 먼저 signal된 것으로 반증한다. 단순 transport 지연도 final write 완료가 있으므로 탈락한다. source에서는 normal finish와 cancellation handler가 idempotent terminal owner를 공유하는지, slot generation을 누가 증가시키는지 확인한다.

복구 설계는 release를 두 번 호출해도 한 번만 효과가 있는 terminal finalizer로 모은다. normal, cancellation, disconnect가 모두 같은 owner에게 reason을 전달하고 owner가 last device consumer와 output handoff를 확인한 뒤 resource를 반환한다. 회귀 시험은 final write 직전, 직후와 device completion 사이에 disconnect 순서를 교차한다. 합격 조건은 final response 여부와 무관하게 각 admitted generation이 정확히 한 terminal resource event를 갖고 allocator 기준선이 반복 workload 뒤 돌아오는 것이다.

### 장애 4: graph를 켠 뒤 처리량은 올랐지만 tail latency가 흔들린다

증상은 평균 iteration latency 개선과 p99 TTFT 악화가 함께 나타나는 것이다. graph replay hit ratio는 92%라 성공처럼 보인다. 하지만 active rows가 8에서 9로 넘어갈 때마다 iteration이 길어지고 waiting queue age가 계단처럼 상승한다. compile storm, memory pressure와 large prefill 유입이 후보 가설이다.

관측을 graph bucket별로 나누면 active 8은 capacity 8, active 9는 capacity 16을 사용한다. dummy rows가 일곱 개로 늘고 persistent metadata tail update도 두 배 범위를 만진다. 더 중요한 것은 mixed prefill이 있는 9-row steps 일부가 graph eligibility를 벗어나 eager fallback한다는 점이다. first divergence는 global graph hit가 아니라 workload phase별 bucket transition이다.

memory pressure 가설은 allocator headroom과 page fault 부재로 약해지고, compile 가설은 새로운 compile event가 없으므로 탈락한다. prefill 자체 비용은 영향을 주지만 같은 prompt length에서도 active 8과 9 경계에서 불연속이 남는다. graph enabled라는 boolean 대신 requested rows, selected capacity, dummy rows, path와 fallback reason을 한 event에 기록했기 때문에 찾을 수 있다.

복구 후보는 9~12 graph variant 추가, admission을 이용한 shape locality, mixed path 분리 또는 eager 유지다. 어느 하나가 보편 답은 아니다. traffic histogram에서 9~12가 충분히 잦고 capture memory가 감당되면 variant가 맞다. 드물다면 variant 비용이 더 크다. 검증은 평균 throughput만 보지 않고 bucket boundary 양쪽의 TTFT, iteration duration, dummy ratio와 queue age를 비교한다. 정책 선택의 상세는 다음 장으로 넘기되 물리 원인은 이 장에서 닫는다.

fixture는 A/B/C active, B finish, D admission과 active sizes graph boundary `N-1,N,N+1`을 둔다. model execution 없이 source permutation과 expected state를 적는다.

복구는 final text 하나보다 request-to-row identity, no stale read, output demux, deferred free와 graph/eager parity를 검증한다. active membership이 변하는 consecutive iterations를 본다.

## 26.8 static과 continuous를 선택하는 판단

offline homogeneous workload, similar lengths와 maximum throughput이면 static/dynamic bucketing이 단순하고 효과적일 수 있다. fixed shapes와 large dense kernels, graph reuse가 강점이다.

online heterogeneous arrivals, streaming latency와 independent completion이 중요하면 continuous batching이 head-of-line과 finished-slot waste를 줄인다. 대신 scheduler/metadata/allocator와 identity correctness가 복잡해진다.

둘은 binary만은 아니다. length bucketing static batches, chunked prefill, graph size padding, limited continuous slots처럼 혼합할 수 있다. option 이름보다 membership change boundary와 physical work를 기록한다.

reader workbook은 요청 A `(prompt=8,output=1)`, B `(2,5)`, C arrival at iteration 2를 쓴다. static padded rows와 loop iterations, continuous active set과 valid token rows를 손계산한다. graph sizes 1,2,4가 있으면 padding도 계산한다.

source 지도는 vLLM scheduler/runner, SGLang scheduler/batch, Transformers manager, llama.cpp slot/batch에서 common owners를 연결한다. policy details를 복제하지 않고 next chapters로 handoff한다.

이 장의 출구 질문은 정적 batch에서 padding과 finished-row waste가 얼마인지, iteration boundary에서 membership이 어떻게 바뀌는지, compaction이 어떤 state를 함께 permute하는지, graph regularity와 flexible shape가 어떤 tradeoff인지다.

### 선택을 문장으로 설명하는 법

좋은 설계 기록은 “continuous batching을 활성화했다”로 끝나지 않는다. 예를 들어 이렇게 쓴다. “우리 traffic은 prompt p50 180, p99 6,000이고 output 분산이 크며 streaming TTFT가 목표다. fixed batch에서 finished rows가 submitted decode rows의 31%였고 short requests가 longest partner completion을 기다렸다. iteration boundary admission으로 lifetime 결합을 끊되 active 9~12가 잦아 graph variants를 보강한다. mapping epoch와 slot generation assertion을 correctness gate로 둔다.” 이 문장은 workload, observed waste, mechanism, 새 비용과 안전 장치를 연결한다.

static을 유지하는 결정도 같은 수준으로 설명할 수 있다. “입력이 사전에 준비된 offline embedding workload이고 sequence lengths를 정렬할 수 있으며 독립 streaming completion이 없다. bucket sealing wait는 사용자 latency가 아니며 fixed shapes가 compile/graph reuse를 단순화한다. padding ratio가 4% 이하이므로 continuous membership machinery의 복잡성을 추가하지 않는다.” continuous가 최신이라는 이유보다 훨씬 강한 결정이다.

hybrid 선택에서는 경계를 밝힌다. prefill은 길이 bucket과 packed rows로 묶고 decode는 iteration-level로 교체할 수 있다. graph는 common capacities만 capture하고 rare shapes는 eager로 보낼 수 있다. server slot은 stable하게 유지하되 compute rows만 dense gather할 수 있다. 각 혼합은 queue, host preparation, dummy device work와 identity complexity 중 어디에 비용을 지불하는지 적는다.

옵션을 바꾼 뒤에는 성공 조건과 rollback 조건을 미리 둔다. 성공은 같은 request trace에서 TTFT/ITL 개선, valid-to-submitted ratio, release delay와 graph fallback이 허용 범위에 있는 것이다. rollback은 cross-request parity 실패, mapping assertion, allocator 기준선 미복귀, queue tail 악화다. throughput이 올라도 correctness gate 하나가 깨지면 채택하지 않는다.

마지막으로 독자는 active set을 숫자 하나로 보지 않는다. 그 안에는 서로 다른 prompt phase, decode age, cache lifetime과 output destination이 있다. continuous batching의 본질은 그 requests를 많이 모으는 것이 아니라 매 iteration 유효한 work만 다시 계약하고, 끝난 lifetime을 안전하게 분리하며, 새 lifetime이 들어올 자리를 만드는 것이다. CUDA Graph와 compaction은 그 계약을 빠르게 만드는 도구이지 identity와 종료 경계를 흐릴 면허가 아니다.

장애 대응에서도 같은 관점을 유지한다. queue가 길면 먼저 GPU가 느리다고 말하지 않고 arrival부터 admission까지 어느 membership boundary를 통과하지 못했는지 찾는다. 출력이 섞이면 model 정확도를 의심하기 전에 current plan의 row와 request identity가 output commit까지 유지됐는지 본다. memory가 줄어들면 allocator 총량만 보지 않고 terminal, last consumer와 slot generation 교체 사이의 순서를 복원한다. graph가 느리면 enabled 여부가 아니라 actual capacity, dummy rows와 fallback phase를 본다.

이 네 질문은 서로 분리된 요령이 아니다. 모두 logical request lifetime과 physical execution shape를 구별하는 한 원리에서 나온다. scheduler는 둘을 매 iteration 연결하고, compaction은 연결을 재배치하며, runner는 그 순간의 physical contract를 실행하고, output processor는 결과를 logical lifetime으로 되돌린다. 어느 경계든 mapping generation을 잃으면 correctness 문제가 되고, 경계를 너무 늦게 넘으면 latency와 utilization 문제가 된다.

따라서 chapter의 계산을 자기 workload에 적용할 때는 먼저 네 requests만 뽑아도 충분하다. prompt가 긴 것, output이 긴 것, 빨리 끝나는 것, 늦게 도착한 것을 고른다. 실제 arrival와 terminal을 보존한 채 static, bucketed, continuous timeline을 손으로 그린다. valid rows, submitted rows, graph capacity와 resource release를 따로 합산한다. 이 작은 표를 source의 membership owner와 맞추면 추상적인 “batch 최적화”가 조사 가능한 engineering 문제로 바뀐다.

다음 장에서 token budget과 admission policy가 등장해도 이 물리 장부를 버리지 않는다. policy가 공정해 보이더라도 dummy work와 graph fallback을 키울 수 있고, throughput이 좋아 보여도 short lifetime을 오래 붙잡을 수 있다. 그때 판단의 기준은 option 이름이 아니라 어떤 request가 어느 iteration에 왜 들어왔고, 어떤 rows가 실제 계산됐으며, 종료 뒤 무엇이 언제 반환됐는가다.

27장으로 넘길 것은 admission/token budgets와 fairness rule이다. 28장으로 넘길 것은 request state enum과 transition. 이 장은 physical batch가 바뀌어야 하는 이유와 안전한 mapping을 소유한다.

독자가 이 장을 닫기 전에 마지막으로 확인할 것은 용어의 단위다. request 수, active sequence 수, scheduled token rows, submitted tensor rows와 graph capacity를 같은 batch size로 뭉개지 않는다. terminal output 전달과 slot/cache release도 같은 finish로 뭉개지 않는다. 이 두 구분만 유지해도 “GPU는 바쁜데 사용자는 왜 기다리는가”, “요청은 끝났는데 왜 새 요청이 못 들어오는가”, “graph replay는 성공했는데 왜 dummy work가 늘었는가”를 서로 다른 원인으로 조사할 수 있다.

좋은 continuous batching 구현은 active set이 자주 바뀐다는 사실을 숨기지 않는다. 변화할 때마다 identity를 증명하고 physical cost를 기록하며, 이전 generation의 마지막 consumer와 다음 generation의 첫 writer 사이를 안전하게 가른다. 성능은 그 correctness 위에서만 의미가 있다.

## 26.9 packed 배열은 하나가 아니라 함께 움직이는 열 묶음이다

`input_ids=[a0,a1,b0,c0,c1,c2]`만 보면 packed batch는 단순 연결처럼 보인다. 그러나 runner가 실제로 소비하는 것은 token IDs와 request row, position, sequence length, query start, block-table row, slot mapping, sampling row, multimodal span을 같은 순서로 묶은 column family다. 한 열만 compact하면 tensor shape는 정상이고 값 범위도 유효해서 즉시 crash하지 않는다. 대신 B의 token이 C의 cache position에 기록되는 식의 valid-but-wrong 결과가 생긴다.

작은 fixture를 만든다. iteration 40의 logical requests는 A, B, C이고 scheduled token counts는 `[2,1,3]`이다. flattened ranges는 A `[0,2)`, B `[2,3)`, C `[3,6)`이며 query starts는 `[0,2,3,6]`이다. request row는 `[A,A,B,C,C,C]`, positions는 `[5,6,9,2,3,4]`라고 하자. block-table rows와 sampling rows는 request 단위로 `[A,B,C]` 순서다. 이 상태의 checksum을 각 열과 request별 slice에 둔다.

B가 finish하고 D가 waiting에서 두 tokens로 들어오면 다음 plan의 logical order를 `[C,A,D]`로 선택할 수 있다. old request permutation은 `[2,0]`, new append는 D다. token ranges는 C `[0,3)`, A `[3,5)`, D `[5,7)`이고 starts는 `[0,3,5,7]`이다. positions, slot mappings와 token IDs는 token permutation `[3,4,5,0,1,new0,new1]`을 사용한다. block table과 sampling state는 sequence permutation `[2,0,new]`을 쓴다. token permutation과 sequence permutation을 같은 배열로 재사용하면 단위가 달라 틀린다.

그래서 source를 읽을 때 `remove_request`, `condense`, `swap_states`, `update_batch` 같은 이름 하나로 끝내지 않는다. 함수가 sequence-indexed arrays와 token-indexed arrays를 각각 열거하는지, helper가 current length와 capacity tail을 어디까지 복사하는지, in-place move의 source/destination overlap을 어떻게 다루는지 확인한다. GPU tensor와 host mirror가 둘 다 있으면 어느 쪽이 canonical이고 copy가 언제 일어나는지도 적는다.

in-place compaction은 순서가 중요하다. `[A,B,C]`에서 A를 제거하려고 last C를 slot 0으로 swap하면 physical order는 `[C,B]`다. stable remove를 기대한 output demux가 `[B,C]`로 해석하면 결과가 교차한다. swap-remove 자체는 올바른 O(1) 선택일 수 있다. 문제는 order contract를 consumer가 모르는 것이다. batch row를 request ID로 다시 찾는지, stable ordering을 요구하는 metadata가 함께 swap되는지 확인한다.

여러 row를 제거할 때 ascending index로 하나씩 shift하면 다음 index가 바뀐다. `[A,B,C,D,E]`에서 indices `[1,3]`을 순서대로 제거하면 B 제거 뒤 원래 D는 index 2가 된다. 여전히 3을 지우면 E가 빠진다. bitmap으로 survivors `[A,C,E]`를 한 번 gather하거나 indices를 descending으로 처리하고, 최종 permutation을 모든 열에 한 번 적용한다. 어떤 방법이든 fixture는 인접 제거, 첫/끝 제거, 모두 제거, remove와 add 동시 발생을 포함한다.

capacity tail은 의미 없는 공간처럼 보여도 CUDA Graph input buffer에서는 이전 iteration 값이 남아 있다. active rows가 8에서 5로 줄었는데 kernel이 graph capacity 8을 읽는다면 tail 5~7을 mask하거나 deterministic dummy로 덮어야 한다. length만 5로 바꾸고 stale slot mapping을 그대로 두면 backend가 padded rows를 실행할 때 이전 request의 KV를 건드릴 수 있다. padding은 계산 낭비뿐 아니라 쓰기 권한 문제다.

## 26.10 add·remove·swap을 한 iteration transaction으로 읽는다

iteration 77 시작 active set이 `[A,B,C,D]`라고 하자. model output 처리 결과 B와 D가 finish하고, scheduler는 waiting E와 F를 admission한다. runner가 먼저 add E/F를 tail에 쓴 뒤 B/D를 remove하는 구현도 있고, survivors를 compact한 뒤 append하는 구현도 있다. 최종 `[A,C,E,F]`만 같다고 중간 상태가 안전한 것은 아니다. add가 old tail D의 last device consumer보다 앞서면 F가 쓴 metadata를 late kernel이 읽을 수 있다.

transaction을 `observe completion → mark terminal → fence last consumer → build survivor permutation → reserve new slots → initialize all new columns → publish plan generation`으로 쓴다. 구현은 단계를 합칠 수 있지만 publish 전에는 consumer가 partial plan을 보면 안 된다. host arrays가 먼저 바뀌고 device copy가 비동기로 진행된다면 stream/event dependency가 publication fence다. mutex만으로 CUDA work의 완료를 보장한다고 가정하지 않는다.

두 세대를 둔다. batch generation G77은 old slot ownership을, G78은 new mapping을 가리킨다. output에는 execution generation과 row/request identity가 있어 G77 결과를 G78 map으로 해석하지 않는다. slot 자체도 reuse generation을 갖는다. request ID를 재사용하지 않더라도 late write가 같은 physical slot을 공격할 수 있기 때문이다. `slot=2,generation=14`와 `slot=2,generation=15`는 다른 소유권이다.

사고를 구체화한다. overlap mode에서 G77 compute가 끝났다는 event는 signal됐지만 output copy의 last consumer는 끝나지 않았다. CPU scheduler가 B/D를 제거하고 slot 3을 F에 재할당해 G78 metadata를 썼다. 늦은 G77 output packer가 slot 3의 request ID를 읽어 D 결과를 F mailbox로 보냈다. token은 vocabulary 범위 안이고 F도 실행 중이라 오류가 조용히 통과했다.

first divergence는 model logits가 아니다. G77 selected token과 G77 device row checksum은 맞다. output packer가 읽은 routing metadata generation만 G78이다. 수정은 output packer가 G77 immutable routing snapshot을 소유하게 하거나 slot 재사용을 output last-consumer event 뒤로 미룬다. 단순히 request ID 비교를 추가할 경우 mismatch 뒤 D의 terminal과 F의 progress를 어떻게 복구하는지도 정해야 한다.

race fixture는 event barrier를 코드로 제어한다. G77 compute 완료 뒤 output pack을 멈추고, CPU에 remove/add를 허용한 다음 G77 pack을 재개한다. expected는 D 결과가 D terminal 경로로만 가고 F cursor는 변하지 않는 것이다. normal/overlap, stable compact/swap-remove, graph/eager를 교차한다. model 실행이 없어도 synthetic row payload와 generation으로 mapping logic을 검증할 수 있다.

rollback도 transaction의 일부다. survivors compact 후 E slot 초기화가 실패하면 G77을 되살릴 수 있는지, 아니면 A/C만 담은 새 G78을 publish하고 E/F를 reject하는지 구현 계약을 택한다. partial E metadata가 active length 밖에 남더라도 다음 graph padded tail이 읽지 않게 zero/mask한다. reservation과 initialization을 분리하면 실패한 reservation 반환도 확인한다.

## 26.11 padding 이득은 세 층에서 따로 계산한다

첫 층은 prompt padding이다. lengths `[1024,64,60,52]`를 dense `[4,1024]`로 만들면 submitted rows 4096, valid rows 1200, padding 2896, row utilization 29.3%다. packed prefill이 정확히 1200 rows를 실행한다면 70.7%의 row를 제거한다. 하지만 attention과 MLP 비용이 row마다 같지 않고 varlen metadata 비용이 있으므로 wall-time 70.7% 개선을 뜻하지 않는다.

둘째 층은 decode finished padding이다. 네 요청 output lengths가 `[1,4,20,100]`이고 fixed batch가 100 iterations를 네 rows로 실행하면 submitted decode rows 400, valid rows 125, dummy 275다. continuous removal이 이상적으로 valid 125만 실행하면 row 기준 68.75%를 줄인다. 새 요청을 빈 자리에 넣으면 submitted rows는 다시 커질 수 있지만 그것은 dummy가 아니라 다른 request의 유효 work다. throughput 분모에 completed useful tokens를 써야 구분된다.

셋째 층은 graph capacity padding이다. active sequence histogram이 size 1~8에 70%, 9~12에 25%, 13~16에 5%이고 buckets가 8과16뿐이라고 하자. 구간 대표 평균을 각각 6,10.5,14.5로 잡으면 expected valid rows는 `0.70×6+0.25×10.5+0.05×14.5=7.55`다. expected capacity는 `0.70×8+0.30×16=10.4`, dummy는 2.85, capacity utilization은 약72.6%다.

bucket 12를 추가하면 expected capacity는 `0.70×8+0.25×12+0.05×16=9.4`, dummy 1.85, utilization 약80.3%다. iteration당 capacity 한 row를 줄이는 셈이다. 그러나 새 graph는 capture memory, warm-up time와 code path를 늘린다. 9~12 traffic이 하루 중 짧게만 나타나면 이득이 비용을 갚지 못할 수 있다. histogram을 시간대·phase·model별로 봐야 한다.

혼합 prefill/decode에서는 token row와 sequence row가 다르다. 세 decode requests 각1 token과 한 prefill request 128-token chunk를 합치면 sequences4, scheduled tokens131이다. graph bucket이 sequence count 기준이면 capacity4가 맞지만 token buffer는131을 담아야 한다. token count 기준 graph라면 128 다음 bucket256을 골라125 dummy capacity가 생길 수 있다. source의 capture key와 buffer dimensions를 확인하지 않고 `batch=4`로 비용을 말하지 않는다.

padding gain의 break-even을 단순 모형으로 쓴다. dense execution time이 submitted row당 `c`, packed metadata/gather 비용이 fixed `h`와 valid row당 `m`이라면 packed가 유리한 조건은 `P·c > h+V·m`이다. `P`는 제거한 padding rows, `V`는 valid rows다. `c=8µs`, `m=0.5µs`, `h=200µs`, `V=1200`이면 오른쪽 800µs이고 최소 padding 100 rows를 넘어야 한다. 앞 예제 P=2896이므로 모형상 이득이 크다. 실제 kernel 병렬성과 launch overlap 때문에 계수는 benchmark로 구한다.

## 26.12 source에서 permutation의 producer와 consumer를 고정한다

vLLM에서는 고정 revision의 scheduler step ordering, runner batch removal/new-state와 output reconciliation span을 연결한다. scheduler가 request를 finished/admitted로 판단하는 것과 runner가 physical arrays를 바꾸는 것은 다른 소유자일 수 있다. remove list와 new requests가 어떤 구조로 runner에 전달되고 runner가 어느 persistent state를 갱신하는지 따라간다. 이름이 같은 request count라도 scheduler와 runner snapshot 시점이 다를 수 있다.

SGLang에서는 normal loop와 overlap loop가 batch/result lifetime을 어떻게 다르게 겹치는지 본다. schedule batch의 request ordering, output processing과 next batch construction 사이에 어떤 object가 살아 있는지 기록한다. overlap은 “동시에 실행한다”는 설명보다 이전 result의 last consumer와 다음 batch metadata first writer가 공유 buffer에서 어떻게 fencing되는지가 핵심이다.

Transformers continuous manager에서는 generation loop가 next batch를 준비하고 processor가 batch를 update하는 span을 잇는다. 전통적 `generate`의 stable tensor batch와 continuous API의 membership 변화 경계를 구분한다. request state completion이 어느 시점에 batch removal로 소비되고 error handler가 partial update를 어떻게 닫는지 본다.

llama.cpp에서는 server slot lifetime과 compute batch construction을 분리한다. slot이 stable해도 `llama_batch`에 들어가는 token rows는 iteration마다 바뀔 수 있다. slot ID를 physical row로 가정하지 않고 positions, sequence IDs와 logits selection이 batch builder에서 어떻게 채워지는지 확인한다. CUDA Graph 사용 여부는 실제 captured dimensions와 update 가능 metadata를 별도로 읽는다.

각 source card에는 revision, file/symbol/span, caller, input collection, survivor/new permutation, mutated arrays, async stream/event, output consumer, rollback과 falsifier를 둔다. “batch를 갱신한다”는 claim은 Grade를 얻지 못한다. 어떤 배열의 어느 단위를 어떤 generation에서 바꾸는지까지 있어야 한다.

## 26.13 permutation incident dossier와 관측 설계

운영 metric은 `active_requests` 하나로 부족하다. iteration generation별 active sequences, scheduled valid tokens, submitted tensor rows, selected graph capacity, dummy rows, removed/admitted counts와 permutation size를 둔다. gauges와 event counters를 섞지 않는다. active는 snapshot, admitted/finished는 interval event다.

permutation 자체를 request ID label로 노출하지 않는다. trace에 old/new row와 generation을 넣고 metric에는 permutation length, moved rows, stable/swap mode, validation failure reason만 bounded label로 둔다. production에서 전체 배열을 로그로 남기면 높은 비용과 민감 정보 위험이 있다. 실패 sampling trace에 hash와 소수의 anonymized row만 둔다.

보존식은 `old_active - terminal - preempted + resumed + admitted = new_active`다. token 열에는 `sum(scheduled_tokens)=last(query_starts)`를 둔다. 각 request의 range length가 scheduled count와 같고 ranges가 겹치지 않으며 `[0,total)`을 덮는지 검사한다. sequence-indexed table rows는 new active identities와 일대일이어야 한다. graph tail은 active와 겹치지 않는 dummy ownership을 갖는다.

incident dossier에는 G77/G78 plan, old/new identity table, token/sequence permutations, host/device buffer generations, CUDA events, selected output와 demux cursor를 넣는다. first divergent column을 표시한다. 최종 text가 틀렸다는 결과만 저장하면 model, sampling, mapping 중 어디서 갈렸는지 다시 조사해야 한다.

회귀 matrix는 no-change, add-only, remove-only, adjacent/multiple remove, swap first-last, add+remove same iteration, all-finish then new batch, preempt/resume, graph boundary, overlap delayed output을 포함한다. 각 fixture는 expected final text보다 intermediate mapping과 terminal resources를 먼저 검증한다. random property test는 unique identities와 arbitrary scheduled counts를 만들고 permutation 후 round-trip을 검사한다.

성능 회귀는 moved bytes와 CPU preparation, H2D metadata copy, graph replay/fallback, iteration latency를 나눈다. stable compact가 bytes를 많이 옮기고 swap-remove가 consumer complexity를 늘릴 수 있다. traffic에 output ordering 요구가 없다면 swap이 유리할 수 있지만 deterministic trace와 debugging 비용도 결정에 포함한다.

## 26.14 실전 리뷰와 다음 장 handoff

리뷰 첫 질문은 “batch size가 몇인가”가 아니다. request count, sequence count, scheduled token rows, tensor capacity 중 어느 단위인지 묻는다. 둘째는 membership을 누가 결정하고 physical arrays를 누가 materialize하는지다. 셋째는 old output의 last consumer와 reused slot의 first writer 사이 fence다. 넷째는 모든 parallel columns가 같은 permutation generation을 쓰는지다.

옵션 리뷰에서는 continuous enable, max sequences, graph sizes, padding/capture, overlap이 어느 constructed component와 branch를 바꾸는지 걷는다. requested option과 effective path를 구분한다. graph를 요청했지만 unsupported shape가 eager로 가거나 overlap을 요청했지만 capability 때문에 normal loop가 선택될 수 있다. metric은 effective mode를 보여야 한다.

배포 fixture는 네 요청을 쓴다. A prompt가 길고 빨리 끝나며, B output이 길고, C는 늦게 도착하고, D는 graph boundary를 넘긴다. baseline과 candidate에 동일 arrival/stop을 넣고 active membership, rows, capacity, output cursor, release event를 비교한다. stochastic sampling이면 deterministic tokens를 주입하거나 mapping layer를 model과 분리한다.

승격 조건은 identity/terminal assertion 0 failures, valid-token 결과 parity, allocator 기준선 복귀, 그리고 workload 목표에 맞는 TTFT/ITL/goodput 개선이다. 평균 GPU utilization만으로 승격하지 않는다. rollback 조건은 cross-request output, stale generation write, unreleased slot/cache, graph boundary tail 악화다. rollback 뒤 old/new generation resources를 reconcile한다.

27장에는 combined admission budget을 넘긴다. 이 장의 `valid tokens`, `active sequences`, `graph capacity`, metadata/workspace와 cache holding이 다음 장 식의 입력이다. 28장에는 G77/G78 membership transition과 terminal ownership을 넘긴다. 29장에는 prefill chunk가 iteration rows와 TTFT/ITL을 어떻게 바꾸는지를 넘긴다.

독자가 26장을 닫으며 만들 산출물은 세 개다. 첫째, 실제 source span으로 연결된 add/remove/swap permutation table이다. 둘째, prompt·finished·graph padding을 분리한 수치 장부다. 셋째, delayed output과 slot reuse를 교차한 race fixture다. 이 셋이 있으면 continuous batching은 홍보 문구가 아니라 측정하고 반증할 수 있는 실행 계약이 된다.

**배열 감사표를 실제로 채우는 순서.**

먼저 source에서 persistent batch object의 모든 field를 열거한다. 이름으로 분류하지 말고 leading dimension과 index consumer를 확인한다. `[num_reqs]`라도 request indexed일 수 있고 sequence indexed일 수 있으며 beam/speculative child가 있으면 둘이 같지 않다. `[num_tokens]`는 flattened scheduled-token indexed일 가능성이 크지만 padded capacity를 포함할 수도 있다. `[num_reqs,max_blocks]` block table은 row와 유효 block length가 함께 움직여야 한다.

각 field에 단위, dtype, device, capacity, active length, producer, first consumer, last consumer를 적는다. 예를 들어 host `req_ids`는 sequence indexed이고 output demux가 last consumer일 수 있다. device `slot_mapping`은 token indexed이고 attention kernel이 소비하지만 비동기 debug/output copy가 뒤에 남을 수 있다. host object를 바꿀 수 있는 시점과 device buffer를 덮을 수 있는 시점이 같다고 가정하지 않는다.

그다음 add path만 단독으로 읽는다. capacity가 충분할 때 tail append인지 free-list slot인지, 새 request의 모든 열이 초기화되는지 본다. position과 block table은 채웠지만 sampling row가 이전 occupant 값을 유지할 수 있다. default zero가 안전한 필드와 반드시 request에서 계산해야 하는 필드를 구분한다. initialization completion 전에 active length를 늘리면 consumer가 partial row를 볼 수 있다.

remove path에서는 terminal marking, output drain과 physical removal을 분리한다. logical finished가 즉시 array removal을 뜻하지 않을 수 있다. last token logprob나 usage 계산이 row state를 더 읽는다면 deferred removal이 필요하다. 반대로 output object가 필요한 값을 복사했다면 client network write 완료까지 device row를 붙잡을 이유는 없을 수 있다. 정확한 last consumer를 찾아야 latency와 memory를 동시에 줄일 수 있다.

swap path에서는 두 방향을 본다. destination에 source row를 복사하는 것뿐 아니라 row-to-request와 request-to-row 역색인이 모두 갱신되는지 확인한다. inverse map 하나가 stale하면 다음 remove가 잘못된 row를 찾는다. swap 두 번이 identity가 되는 property, permutation 뒤 inverse lookup round-trip, 모든 active request가 정확히 한 row를 갖는 uniqueness assertion을 둔다.

batch update 함수 밖의 숨은 배열도 찾는다. sampling metadata cache, grammar state, LoRA/adaptor selection, multimodal encoder offsets, speculative draft cursor, prefix-cache reference가 별 object에 있을 수 있다. main batch의 order를 바꾸고 auxiliary object가 request ID lookup을 쓰는지 positional lookup을 쓰는지 확인한다. positional consumer만 permutation 대상이다. request keyed map까지 불필요하게 재배열하면 오히려 bug를 만든다.

**메모리 이동 비용을 손으로 상한 계산한다.**

active capacity 256이고 request row마다 block table 128 entries×4B=512B, sampling metadata 256B, positions/lengths/IDs 64B, 기타 host/device index 192B라면 row당 약1,024B의 직접 metadata라고 하자. stable compaction으로 평균128 rows를 옮기면 iteration당128KiB다. 초당200 iterations면 약25MiB/s여서 순수 bandwidth는 작아 보인다.

그러나 비용은 bytes만이 아니다. 여러 작은 tensor에 각각 gather kernel을 launch하거나 host slice와 H2D copy를 수행하면 launch와 synchronization이 지배한다. 12개 arrays에 각8µs launch/dispatch overhead가 있으면 iteration당96µs, 200 iterations/s에서 CPU/GPU scheduling time 19.2ms/s다. 절대값은 환경마다 다르지만 “metadata가 작으니 공짜”가 아닌 이유를 보여 준다.

swap-remove는 finished k개에 대해 k rows만 옮길 수 있다. capacity256에서 매 step2개가 끝나면 stable gather128KiB 대신 약2KiB가 될 수 있다. 대신 physical ordering이 불안정하고 여러 remove에서 last survivor 선택과 inverse map update가 복잡하다. output ordering을 row order에 의존하는 consumer가 하나라도 있으면 그 consumer를 고치거나 stable compact를 유지해야 한다.

GPU gather로 모든 columns를 fused하게 옮기면 launch를 줄일 수 있지만 dtype/shape가 다른 arrays와 host-only objects를 다루기 어렵다. struct-of-arrays를 array-of-structs로 바꾸면 이동은 단순해져도 kernels의 coalescing과 selective access가 나빠질 수 있다. 자료구조 선택은 compaction 한 함수의 속도가 아니라 hot consumers 전체의 access pattern으로 판정한다.

복사하지 않는 indirection도 후보다. logical order가 physical slots의 index vector를 참조하면 add/remove는 index만 바꾼다. 그러나 attention과 sampling kernels이 extra gather/indirection을 매 token 지불하고 graph capture가 dynamic index를 받아야 한다. fragmentation으로 memory access locality가 나빠질 수 있다. compaction cost를 없앤 대신 every-iteration consumer 비용을 만든다.

break-even은 `C_compact < I_remaining × C_indirection`으로 생각할 수 있다. request가 남은 iterations `I_remaining` 동안 매번 indirection 비용을 지불한다면 긴 decode에서는 한 번 compact하는 편이 낫다. 반대로 membership churn이 매우 높고 request가 짧으면 indirection이 유리할 수 있다. 실제 분포와 kernel profiler로 계수를 구하고 source에서 어느 path가 선택되는지 고정한다.

**padding과 cache를 함께 보아야 하는 이유.**

finished row를 compute batch에서 제거해도 KV block reference가 즉시 풀리지 않으면 새로운 request admission은 늘지 않는다. 반대로 cache를 먼저 풀고 output/runner가 old block table을 읽으면 use-after-free가 된다. compute compaction과 allocator release는 같은 terminal generation을 공유하되 서로 다른 last consumer 조건을 가질 수 있다.

A가 block10개를 쓰고 끝났고 next iteration에 D가10개를 요구한다고 하자. allocator가 A blocks를 D에 재할당한 뒤 old graph tail이 A slot mapping으로 write하면 D cache가 오염된다. graph dummy rows를 mask했다는 assertion과 allocator generation check가 모두 필요하다. dummy compute를 줄이는 성능 작업이 ownership fencing과 직접 연결되는 사례다.

prefix cache가 A blocks 일부를 공유한다면 request terminal이 곧 physical free는 아니다. A reference만 감소하고 shared prefix entry는 남는다. metric에서 request-held blocks released와 allocator free blocks increase가 항상 같다고 가정하지 않는다. reference owner별 ledger를 둔다. 그렇지 않으면 정상 공유를 leak으로 오판하거나 실제 request reference 잔류를 cache hit 자산으로 오판한다.

preemption도 remove와 동일하지 않다. running batch에서 빠지지만 logical request는 terminal이 아니며 KV를 유지하거나 swap/recompute 상태로 바꿀 수 있다. resume 시 sampling/RNG/grammar cursor와 output commit position이 이어져야 한다. finished removal fixture만으로 preempt/resume permutation을 검증하지 못한다.

**교차 요청 오염 사고를 단계별로 판정한다.**

증상은 C가 생성할 수 없는 토큰 하나를 냈다는 것이다. 먼저 tokenizer/model logits 문제와 mapping 문제를 분리한다. C logical input과 model runner에 제출된 token/position checksum을 본다. input이 이미 A 값을 포함하면 plan/materialization 경계다. input은 맞고 selected token row가 다르면 kernel/output row mapping이다. selected token도 맞고 client delivery만 다르면 demux/stream 경계다.

사건에서 G103 old active `[A,B,C]`, B terminal, G104 new `[C,A,D]`였다고 하자. C expected new row0인데 block-table permutation은 `[A,C,D]`로 남았다. G104 token IDs와 positions는 C지만 KV row는 A다. attention 결과는 유효 floating-point 값이며 NaN도 없다. 이 때문에 generic health metric은 모두 정상이다. row별 identity hash가 token metadata와 cache table에서 다르게 나타나는 것이 첫 증거다.

원인은 survivor list를 만드는 두 helper가 서로 다른 order를 사용한 것이다. 하나는 stable `[A,C]`, 다른 하나는 free-slot 최적화를 위해 `[C,A]`를 선택했다. 각각 독립적으로 합리적이지만 공통 permutation artifact가 없었다. 수정은 canonical new order를 한 번 만들고 모든 arrays가 그것을 소비하게 한다. helper가 자체 정렬하지 못하게 API를 바꾸거나 returned permutation hash를 assertion한다.

회귀 fixture는 값 범위를 일부러 모두 유효하게 만든다. request별 token IDs, positions, block IDs가 겹치지 않지만 bounds 안에 있게 한다. out-of-range crash에 의존하면 valid-but-wrong을 놓친다. 결과는 request-specific sentinel과 row hash로 검증한다. batch size1에서는 permutation이 드러나지 않으므로 최소 세 survivors와 remove/add가 필요하다.

사고가 낮은 확률이라면 timing과 shape 조건을 보존한다. overlap on, active size graph boundary, two simultaneous finishes, one admission, output copy delay가 재현 조건일 수 있다. 각 조건을 하나씩 제거해 failure가 사라지는지 본다. 모두 묶은 stress만 반복하면 어느 mechanism이 필요한지 알 수 없다.

**성능 실험을 correctness 실험과 같은 표에 놓는다.**

baseline과 candidate마다 logical workload fingerprint, request arrival, prompt/output lengths, stop conditions를 고정한다. correctness 열에는 per-request tokens, finish reason, output cursor, mapping assertions와 resource terminal을 둔다. performance 열에는 queue, TTFT, ITL, goodput, valid/submitted rows, compaction time, graph capacity/fallback과 allocator headroom을 둔다.

candidate가 submitted rows를30% 줄였지만 compaction CPU가 iteration당0.4ms 늘고 GPU step이1ms라면 scheduler가 새 bottleneck이 될 수 있다. GPU utilization은 내려가도 goodput이 오를 수 있고, 반대로 dummy rows가 줄어도 host gap 때문에 device가 쉰다. timeline에서 plan build, H2D metadata, graph launch, compute, output processing을 나눈다.

batch 평균만 보지 않고 churn cohort를 만든다. iteration당 add/remove0, 1~2, 3+ 그룹에서 비용을 비교한다. stable long-running batch에서는 continuous machinery overhead만 보이고, high churn에서 padding 회수 이득이 나타난다. traffic weighted 결과가 deployment 결론이다. synthetic worst case 하나를 전체 workload 결론으로 쓰지 않는다.

TTFT와 ITL도 요청 길이별로 본다. 새 prefill을 즉시 넣어 short request TTFT가 좋아졌지만 기존 decode의 ITL이 흔들릴 수 있다. policy는 다음 장이 소유하지만 이 장에서는 새 rows가 물리 iteration duration을 얼마나 늘렸는지 기록한다. scheduling fairness와 kernel cost를 분리할 수 있어야 한다.

실험 결과에는 confidence와 미검증 조건을 남긴다. runtime을 실행하지 않은 source 분석이면 가능한 mutation chain과 fixture 설계까지만 확정한다. 실제 coefficient, graph eligibility, traffic histogram은 미검증이다. 실행 결과가 있더라도 한 GPU/model의 수치를 다른 architecture에 일반화하지 않는다.

**최종 학습 체크리스트.**

독자는 packed라는 말을 들으면 먼저 flatten axis와 boundary metadata를 묻는다. compaction이라는 말을 들으면 canonical survivor order와 token/sequence permutation을 분리한다. graph padding이라는 말을 들으면 active 단위와 capture key 단위를 묻는다. finish라는 말을 들으면 logical terminal, output last consumer, cache/slot release를 분리한다.

소스 탐색은 scheduler policy에서 시작해도 되고 runner update에서 역추적해도 된다. 어느 방향이든 membership decision→permutation artifact→parallel array mutation→device consumption→output inverse mapping→resource terminal의 닫힌 고리를 만든다. 중간에 “framework가 알아서 한다”는 화살표를 남기지 않는다.

수치 계산은 valid prompt, prompt padding, valid decode, finished dummy, graph dummy, metadata move를 별 행으로 둔다. 비율의 분모를 명시한다. valid/submitted token ratio와 graph active/capacity ratio, request completion/resource release delay는 서로 대체하지 않는다. 한 숫자가 좋아졌다고 전체 최적화가 성공한 것은 아니다.

incident 문장은 첫 divergence를 포함한다. “출력이 섞였다”보다 “G104 survivor order는 `[C,A]`였지만 block-table gather가 G103 stable indices `[A,C]`를 사용해 token row0의 identity C와 cache row0의 identity A가 처음 갈렸다”가 낫다. 이 문장은 수정 owner와 회귀 fixture를 거의 자동으로 정한다.

마지막 승인에는 rollback terminal도 쓴다. 새 compaction path를 끄면 inflight Gnew plans를 drain할지 abort할지, old runner buffer와 new mapping generation을 언제 폐기할지 정한다. feature flag false만으로 이미 제출된 graph와 output을 old consumer가 해석할 수 없다. generation boundary에서 요청과 자원을 닫아야 한다.

이 체크리스트를 통과하면 27장의 admission equation은 추상 숫자가 아니다. 그 식의 token과 sequence는 여기서 만든 유효 rows와 identities이고, workspace와 graph bucket은 여기서 계산한 physical capacity이며, KV blocks는 terminal fencing 뒤에만 반환되는 자원이다. 연속 배치는 바로 이 여러 단위를 매 iteration 다시 합의하는 토대다.

**한 장짜리 실무 워크시트.**

워크시트 첫 칸에는 고정 revision과 effective options를 쓴다. 두 번째 칸에는 iteration G의 old active identities, scheduled counts와 physical capacities를 쓴다. 세 번째에는 terminal/preempt/resume/admit events를 쓴다. 네 번째에는 canonical new order와 token/sequence permutations를 쓴다. 다섯 번째에는 각 array의 mutation span과 last consumer fence를 쓴다. 마지막에는 service·resource·telemetry terminal을 쓴다.

숫자 예제로 G9 old `[R1,R2,R3,R4]`, counts `[1,2,1,4]`, total tokens8을 둔다. R2 terminal, R4 preempt, R5 count3 admission이면 new `[R1,R3,R5]`, counts `[1,1,3]`, starts `[0,1,2,5]`다. sequence survivor permutation은 `[0,2]`, token survivor permutation은 old ranges R1 `[0]`, R3 `[3]`이고 R5 새 range가 뒤따른다. old R4 tokens `[4,5,6,7]`은 resume state로 옮기거나 폐기 규칙을 따른다.

capacity bucket8을 고르면 active tokens5에 dummy3이다. block table은 sequence rows3 plus padded capacity가 있는지, sampling metadata는 sequences3인지 tokens5인지 실제 shape로 기록한다. R2 cache blocks는 terminal fence 뒤 반환하고 R4 blocks는 preemption 정책에 따라 유지·swap·recompute 중 하나다. 둘을 같은 removed count로 합치면 allocator 예측이 틀린다.

output fixture에는 G9 selected results를 늦게 도착시킨다. G10 map이 publish된 뒤 G9 R2 terminal result와 R4 partial result가 들어와도 G9 routing snapshot으로 처리돼야 한다. R2 final은 exactly once이고 R4 partial cursor는 resume state에 반영되되 client에 이미 commit한 범위를 중복하지 않는다. R5 mailbox는 G9 결과로 변하지 않는다.

배열 검증은 세 층이다. 구조 검증은 lengths, starts monotonicity, bounds와 uniqueness다. identity 검증은 각 column의 request/generation hash 일치다. 의미 검증은 position progression, block ownership, RNG/grammar/output cursor가 request history와 이어지는지다. shape만 맞는 검사는 첫 층만 통과한다.

성능 열에는 old submitted8, new valid5, graph capacity8, dummy3, moved survivor rows2, initialized rows1을 적는다. preemption serialization bytes와 cache release blocks도 별도다. 다음 iteration에서 R4가 resume하면 그 restore cost를 현재 optimization 이득에서 빼야 한다. 한 step만 빠른 정책이 전체 request를 더 느리게 만들 수 있다.

falsifier는 구체적이어야 한다. “permutation bug가 아니다”의 증거는 모든 consumed columns의 generation hash가 일치하고 selected token이 이미 잘못됐다는 것이다. “graph padding 탓이 아니다”는 bucket 양쪽에서 dummy ratio를 맞춰도 latency step이 남는 것이다. “cache release 지연이 아니다”는 필요한 blocks headroom이 충분한데 admission이 멈춘 것이다.

source review가 끝나면 워크시트의 빈칸을 센다. mutation span은 찾았지만 last consumer가 비어 있으면 race 위험이 미해결이다. option은 찾았지만 effective branch가 비어 있으면 배포 검증이 미해결이다. 계산은 있지만 실제 단위가 비어 있으면 추정일 뿐이다. 빈칸을 숨기지 않아야 다음 측정이 정확해진다.

이 워크시트는 구현 네 개를 같은 이름으로 평준화하지 않는다. vLLM, SGLang, Transformers와 llama.cpp가 서로 다른 object와 process boundary를 가져도 동일 질문으로 비교하게 한다. 답은 각 pinned symbol과 배열에 남긴다. 공통점은 request lifetime과 physical row 사이의 mapping 계약이고, 차이는 그 계약을 누가 언제 materialize하고 fence하는가다.

마지막 판정은 한 문단이면 된다. workload의 padding/finished waste, 선택한 membership boundary, 실제 permutation owner, graph capacity 비용, 확인한 incident fixture와 남은 미검증 조건을 연결한다. 이 문단을 쓰지 못하면 옵션을 고를 준비가 덜 된 것이다. 쓸 수 있다면 독자는 다음 장에서 여러 admission budget이 충돌할 때도 어떤 physical work와 lifetime이 식의 항을 만드는지 잃지 않는다.

**리뷰에서 자주 나오는 오답을 교정한다.**

“packed이므로 padding이 0이다”는 틀릴 수 있다. prompt padding은 없어도 graph capacity tail, alignment, block table capacity와 kernel tile padding이 남는다. 어느 층의 padding인지 써야 한다. “remove는 Python list에서 끝난다”도 틀릴 수 있다. persistent device tensor, inverse index와 allocator ownership이 별도로 남는다. 함수 반환만 terminal로 보지 않는다.

“request ID가 있으니 reorder는 안전하다”도 충분하지 않다. consumer가 모든 열에서 ID lookup을 하는지, 일부 kernel이 positional row를 쓰는지 확인해야 한다. request ID는 metadata가 함께 잘못 permute되면 틀린 row에 붙은 채 스스로 일관돼 보일 수도 있다. 독립적으로 생성한 generation hash나 sentinel fixture가 필요한 이유다.

“CUDA event가 signal됐으니 slot을 재사용해도 된다”는 event가 무엇을 fence하는지 빠뜨린다. compute completion, D2H output copy, host demux 중 어느 마지막 consumer까지 포함하는지 source stream dependency로 확인한다. 한 event가 전체 lifetime을 대표한다고 이름만 보고 믿지 않는다. overlap path에서는 특히 다른 stream과 host callback이 남는다.

“처리량이 올랐으니 compaction이 성공했다”도 correctness와 tail을 누락한다. cross-request corruption은 낮은 확률이어도 허용할 수 없고, short request TTFT가 악화될 수 있다. goodput 분자에는 정확히 전달된 유효 tokens만 넣고, 실패·중복·dummy work를 제외한다. resource terminal이 관측 창 안에 닫혔는지도 승격 gate다.

마지막 오답은 모든 구현에 같은 최적화를 강요하는 것이다. persistent batch representation, graph capture 방식, slot semantics와 output pipeline이 다르면 최적 permutation 방법도 다르다. 공통 체크리스트는 비교 좌표이지 동일 코드 구조를 요구하는 표준이 아니다. pinned source에서 실제 owner와 consumer를 확인하고 workload 수치로 선택한다.

결국 조사자는 한 iteration을 재현 가능한 회계 단위로 만든다. 들어온 logical lifetime, 실제 제출한 rows, 선택한 capacity, 이동한 bytes, 남은 last consumer와 반환한 resources를 같은 generation에 묶는다. 이 ledger가 연속된 두 iteration에서 보존되면 add/remove/swap은 안전한 변환이다. 하나라도 설명되지 않으면 높은 처리량 수치는 아직 승인 근거가 아니다.

승인 뒤에도 generation mismatch, stale tail write, inverse-map failure와 allocator release delay를 상시 감시한다. 회귀가 발견되면 해당 generation admission을 막고 inflight output과 resources를 reconciliation한 뒤 이전 path로 되돌린다.

복구 완료도 동일 ledger로 다시 엄격히 증명한다.
