# 이 책을 읽는 법 — 목적에 따라 경로를 바꿔라

처음부터 마지막 장까지 읽는 길만 있는 책은 실무에서 오래 살아남기 어렵다. 지금 손에 잡힌 문제가 loss인지, 데이터인지, 분산 hang인지에 따라 출발점이 달라야 한다. 다만 어느 길을 택해도 실제 산출물의 연결은 끊지 않는다.

## 처음 학습 코드를 읽는 독자

### 계산의 최소 폐회로

1장부터 3장까지 읽고 단일 batch의 forward, backward, optimizer step, save, resume를 직접 대조한다. 이어서 5장과 7–10장을 읽으면 문자열이 logits와 gradient로 바뀌는 경로가 닫힌다. 이때 처음부터 대규모 클러스터를 흉내 내지 않는다. 작은 실행에서 확인하지 못한 불변조건은 GPU 수를 늘린다고 생기지 않는다.

### 첫 번째 통과에서 남길 기록

token ID와 label, 유효 loss 토큰 수, 주요 activation shape, gradient norm, 파라미터 한 개의 update 전후 값, optimizer step, RNG 상태, checkpoint ID를 남긴다. 이 기록은 이후 모든 최적화의 대조군이다.

## 파인튜닝 레시피를 검증하려는 독자

### 데이터에서 배포까지

5–6장, 18–20장, 24–25장, 30장 순서로 읽는다. chat template와 response-only mask를 먼저 고정하지 않은 채 SFT 성능을 논하지 않는다. adapter를 merge하거나 quantize한 뒤에는 학습 직후 모델과 같은 prompt, token ID, decoding 조건으로 parity를 확인한다.

### 옵션을 읽는 네 가지 질문

모든 옵션에 네 질문을 붙인다. 무엇을 소유하는가. 언제 상태가 바뀌는가. 체크포인트에 무엇이 남는가. 실패하면 어떤 관측값이 먼저 달라지는가. 옵션 이름을 외운 사람과 시스템을 설명하는 사람은 여기서 갈린다.

## 멀티노드 장애를 다루는 독자

### 소유권부터 그린다

14–17장과 26–29장을 잇는다. parameter, gradient, optimizer state, activation, batch, RNG, sampler cursor를 rank별로 그린 다음 collective와 checkpoint commit을 얹는다. hang이 발생하면 마지막 로그 줄보다 마지막으로 모든 rank가 동의한 상태를 찾는다.

### 복구의 세 등급

복구 성공은 하나가 아니다. 같은 표본을 같은 순서로 소비하는 sample-exact, 허용 오차 안에서 같은 결과에 도달하는 numerical-equivalent, 바이트까지 같은 bitwise-identical을 구분한다. 요구 등급을 정하지 않으면 재현성 논쟁은 끝나지 않는다.

## 연구 논문과 새 아키텍처를 읽는 독자

### 주장과 구현 사이를 왕복한다

8–14장과 21–23장을 중심으로 읽는다. 목적함수와 복잡도 표만 옮기지 말고, 공식 코드에서 새 state가 어디에 생기는지, backward가 무엇을 저장하는지, 분산 환경에서 누가 통신하는지 확인한다. 논문에 없는 운영 비용은 구현의 state dict와 collective 경로에서 드러나는 경우가 많다.

### 장마다 사용할 디깅 질문

입력의 최소 단위는 무엇인가. 출력의 의미와 shape은 무엇인가. 학습되는 상태와 단순 buffer는 무엇인가. stochastic한 선택은 어디서 일어나는가. 동일성을 깨뜨릴 수 있는 비결정적 연산은 무엇인가. 이 다섯 질문은 새로운 모델을 만났을 때도 그대로 쓸 수 있다.

## 표기와 근거를 읽는 법

### 코드 인용

코드 조각은 설명에 필요한 최소 범위만 싣는다. 주변 제어 흐름이 중요한 경우에는 호출자와 피호출자, 테스트를 함께 안내한다. 행 번호는 고정 리비전과 짝을 이루며, 최신 브랜치의 이동 가능한 행 번호를 영구 좌표처럼 취급하지 않는다.

### 수식과 텐서

수식의 첨자는 가능한 한 코드의 차원 이름과 맞춘다. `B`는 batch, `T`는 sequence length, `H`는 hidden size처럼 장마다 다시 정의한다. reduction이 나오면 합산 축과 분모를 쓰고, stop-gradient나 mask가 있으면 계산 그래프에서 끊기는 지점을 밝힌다.

### 장 끝의 인계물

각 장 마지막에는 다음 장으로 넘기는 실제 파일, ID, checksum, tensor snapshot을 적는다. “연관이 있다”는 말만으로 연결하지 않는다. 인계물을 재생성해 두 결과를 대조할 수 있을 때 비로소 연결이 완성된다.
## 원전에서 실패 계약까지 역방향으로 읽는 법

이 책의 인용은 장식이 아니라 추적 시작점이다. 가장 안전한 읽기 순서는 `원전의 불변 식별자 → 보존 원문의 chunk와 hash → 수식의 적용 범위 → 현재 코드의 tensor·상태 전이 → 반례와 실패 fixture`다. 제목이나 개념어가 같다는 이유만으로 원 논문과 현재 구현을 같은 것으로 취급하지 않는다.

8장의 scaled dot-product attention을 예로 들자. 먼저 *Attention Is All You Need*의 arXiv ID `1706.03762`와 30books에 보존된 text content digest를 확인한다. 그다음 `QK^T/√d_k`, mask, row-wise softmax, `PV`라는 수식으로 내려간다. 현재 Llama·Qwen 계열 코드는 GQA head 공유, RoPE, backend dispatch와 dtype 조건을 더한다. 따라서 원 논문은 수학적 선행 근거지만 현재 함수 전체의 구현 명세는 아니다. 모든 위치가 mask된 행에서 NaN이나 구현별 fallback이 생기는 반례가 이 확대 해석을 막는다.

13장의 scaling law도 같은 방식으로 읽는다. `2001.08361`의 관측식은 모델·데이터·compute 축의 경험적 power law다. 데이터 혼합기나 curriculum 코드가 그 식을 “구현”하는 것은 아니다. 현재 코드에서는 어떤 token·sample을 실제 소비했는지, optimizer와 stopping rule이 무엇인지 별도로 추적한다. 관측 범위 밖 외삽, 데이터 질 변화와 downstream metric 전환은 실패 계약이다.

15장의 GPipe는 `1811.06965`의 micro-batch flush pipeline에서 시작한다. bubble 식을 읽고 activation rematerialization과 batch semantics를 분리한 뒤 현재 Megatron schedule의 1F1B·interleaving·virtual stage 상태로 내려간다. 둘은 역사적으로 연결되지만 같은 scheduler가 아니다. microbatch 수가 pipeline stage 수보다 작을 때, uneven sequence가 들어올 때, 마지막 partial batch가 생길 때의 bubble·denominator fixture가 구현 차이를 드러낸다.

역연결이 없다는 기록도 정보다. 현재 30books graph에는 paper-like entry 27개와 concept 164개가 있지만 arXiv ID가 없는 entry가 9개이고, 원문 전체를 사용할 수 없다고 표시된 entry도 1개다. 이 경우 제목 유사도나 concept label만으로 `exactMatch`를 만들지 않는다. 독자는 링크 수보다 chunk hash, 문자 좌표, 관계 종류와 “무엇을 증명하지 않는가”를 먼저 확인해야 한다.

### 164개 개념의 두 번째 감사표를 읽는 법

기존 역색인에는 164개 중 17개 concept가 한 번 이상 연결되어 있었다. 두 번째 감사에서는 그 링크를 자동 승인하지 않고 원문 chunk의 hash·문자 offset·발췌문과 2권의 수식·코드·실패 계약을 다시 맞췄다. 이 엄격한 기준에서 독립 승인한 항목은 scaled dot-product attention, GPipe pipeline parallelism, micro-batch 세 개다. 나머지 161개는 삭제한 것이 아니라 `검토필요`로 남겼다.

표의 `mappingBasis`는 두 노드가 공유하는 정확한 명제를 말한다. `confidence`는 그 명제 연결의 확신이지 현대 구현 전체가 원 논문과 같다는 확률이 아니다. `boundary`는 반드시 반례처럼 읽는다. 예를 들어 GPipe의 micro-batch가 현재 1F1B scheduler의 선행 개념이라는 사실은 두 상태 기계가 동일하다는 뜻이 아니다.

미연결 reason taxonomy의 0도 의미가 있다. 자동 감사로 원문없음·동명이의·범위밖·현재 엔티티없음을 확정할 근거가 없으면 억지로 분류하지 않고 `검토필요`에 둔다. 독자는 먼저 exact chunk를 읽고, 로컬 수식의 변수와 전제를 대조하고, 코드 symbol이 그 수식을 어느 조건에서 구현하는지 확인한 다음, 실패 계약이 현대적 차이를 막고 있는지 검토한다.
