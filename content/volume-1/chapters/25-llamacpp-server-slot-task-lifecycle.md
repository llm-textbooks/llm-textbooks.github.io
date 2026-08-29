# 25장. llama.cpp 서버의 slot과 task: 요청 한 건이 살아서 돌아오는 전 과정

밤 11시, 단일 GPU에 올린 `llama-server`가 이상해졌다. GPU 사용률은 간헐적으로만 치솟고 `/health`는 정상인데, 새 채팅 요청은 첫 글자를 받기까지 오래 기다린다. 어떤 연결은 취소했는데도 한동안 계산이 계속되는 것처럼 보인다. 운영자는 `--parallel 8`을 주었으니 여덟 요청이 항상 동시에 전진할 것이라고 생각한다. 로그의 `slot 3`을 GPU 스트림 번호로 이해하고, slot이 비었는데 요청이 기다리는 현상을 KV 메모리 부족으로 단정한다. 세 해석은 모두 위험하다.

이 장은 이런 밤에 소스 코드를 어디서부터 읽어야 하는지를 설명한다. 출발점은 REST API 목록이 아니라 요청 한 건의 수명이다. HTTP 핸들러가 JSON을 해석하고 `server_task`를 만든다. task는 `server_queue`에 들어간다. 주 루프가 task를 받아 사용 가능한 `server_slot`에 결합하거나 deferred 큐로 보낸다. `update_slots()`는 여러 slot의 프롬프트와 다음 토큰 작업을 한 모델 배치로 조립한다. 결과는 `server_response`로 돌아가고, HTTP 쪽의 `server_response_reader`가 스트림 조각이나 최종 응답으로 변환한다. 연결이 끊기면 reader의 정리가 취소 task를 게시하고, 서버는 대기 큐와 실행 slot을 서로 다른 방법으로 청소한다.

여기서 slot은 “HTTP 연결 하나”, “OS 작업 스레드 하나”, “CUDA 스트림 하나”, “독립 모델 하나”가 아니다. 현재 요청의 프롬프트 상태, sampler, 생성 문자열, 중지 조건, 통계, KV sequence 식별자를 묶어 두는 서버 실행 자리다. 반대로 task는 큐를 건너 slot으로 이동하는 작업 명세다. 모델의 실제 forward는 여러 slot의 토큰을 합친 `server_batch`와 llama.cpp context가 수행한다. 이 세 객체를 분리해 읽어야 동시성, 캐시 재사용, 취소, 스트리밍의 원인을 정확히 찾을 수 있다.

이 장의 근거는 llama.cpp 커밋 `bb4caa7540188872173c44d161602d9271386413`으로 고정한다. 이후 코드가 달라졌다면 함수 이름이 아니라 여기서 제시하는 소유권 질문을 다시 적용해야 한다. 누가 요청을 소유하는가, 누가 실행 자리를 소유하는가, 누가 결과를 기다리는가, 어느 상태 전이가 다음 계산을 허용하는가가 핵심이다.

## 25.1 먼저 세 객체를 갈라 놓는다: HTTP 요청, task, slot

### 25.1.1 세 객체를 구분하는 세 질문

#### 같은 요청을 세 번 표현하는 이유

클라이언트가 보낸 JSON은 외부 계약이다. `messages`, `prompt`, `stream`, `n`, `max_tokens`, sampler 설정처럼 사용자가 이해하는 필드를 담는다. 그러나 JSON 객체를 그대로 계산 루프에 들고 가면 네트워크 수명과 추론 수명이 강하게 결합한다. 느린 클라이언트의 소켓 쓰기, 연결 종료, 배치 요청의 여러 자식 결과가 모델 실행 상태를 오염시킬 수 있다. 그래서 서버는 외부 표현을 내부의 `task_params`, 토큰열, task 종류로 정규화한다.

`server_task`의 `id`는 queue가 부여하는 내부 식별자다. `index`는 한 HTTP 요청에 여러 프롬프트가 있을 때 결과 순서를 되찾는 데 쓰인다. `id_target`은 취소 task가 취소할 원래 task를 가리킨다. `id_slot`은 특정 slot을 요구하는 관리 작업이나 명시적 선택에 쓰인다. `id_parent`와 `child_tasks`는 한 프롬프트에서 여러 completion을 만드는 병렬 sampling을 표현한다. `params`와 `tokens`가 실제 추론 입력이며, `type`이 completion, embedding, rerank, cancel, metrics 같은 처리 경로를 고른다.

이 필드 배치는 [server-task.h의 `server_task`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-task.h#L137-L222)에서 확인할 수 있다.

slot은 task보다 오래 산다. `server_slot::task`는 현재 결합된 작업이고 `task_prev`는 디버깅을 위해 직전 작업을 잠시 보존한다. `prompt.tokens`와 `common_memory mem`은 프롬프트 및 sequence 메모리와 연결된다. `smpl`은 sampler 상태다. `generated_text`, `generated_tokens`, `n_sent_text`는 생성과 스트림 전송의 진행을 추적한다. `has_next_token`, `stop`, `stopping_word`, `n_predict_max`는 종료 판정을 담는다. `stats`는 이 자리에서 처리한 프롬프트와 생성 구간의 시간을 기록한다.

정의 전체는 [server-context.cpp의 `server_slot`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L196-L360)에 있다.

따라서 요청이 끝나도 slot 객체는 사라지지 않는다. `reset()`은 생성 문자열, 토큰 확률, stop 상태, sampler 연결, 통계를 다음 작업에 맞게 초기화하고 현재 task를 `task_prev`로 옮긴다. 프롬프트 KV를 언제 지울지는 별도 정책이다. 이 차이가 중요하다. “slot이 idle이다”는 “그 slot에 캐시가 없다”는 뜻이 아니다. 반대로 “캐시 토큰이 있다”는 “그 slot이 처리 중이다”라는 뜻도 아니다.

#### slot은 스케줄러도 KV page도 아니다

vLLM 문맥에서 scheduler는 여러 sequence group의 실행 우선순위와 token budget, block 할당을 결정한다. SGLang 문맥에서는 scheduler process가 request 상태, radix cache, running batch를 관리한다. llama.cpp의 `server_slot`은 그 전체 역할과 등가가 아니다. slot은 서버 레이어가 유지하는 요청 실행 자리이며, `update_slots()`와 lower-level llama context가 함께 실제 배치를 만든다. KV sequence id로 slot id가 사용되더라도 slot 자체가 물리 KV 블록이나 GPU 페이지는 아니다.

이 구분은 장애 판단을 바꾼다. 모든 slot이 processing이면 새 inference task가 deferred 되는 것은 slot admission 문제다. idle slot은 있는데 decode가 실패한다면 batch 용량, context 공간, 모델 backend 오류를 따로 봐야 한다. 공통 prefix가 긴 idle slot이 선택되었다고 해서 필요한 모든 KV가 즉시 재사용된다고 단정할 수 없다. 프롬프트 비교, memory sequence 조작, 실제 decode가 각각 성공해야 한다.

간단한 소유권 표를 머릿속에 둔다.

| 객체 | 주된 소유 내용 | 수명 경계 | 오해하면 생기는 진단 오류 |
|---|---|---|---|
| HTTP 요청/response reader | 연결, 기다릴 task id 집합, 스트림 조립 상태 | 핸들러 시작부터 응답·취소까지 | 느린 소켓과 느린 추론을 같은 원인으로 본다 |
| `server_task` | 정규화된 파라미터, 토큰, 종류, 관계 id | 생성 후 큐를 거쳐 slot 또는 즉시 처리까지 | 큐 대기와 slot 실행을 합쳐 본다 |
| `server_slot` | 현재 task, prompt, sampler, 생성 상태, 통계 | 서버 초기화부터 종료까지 재사용 | slot 수를 GPU 실행 스레드 수로 본다 |
| `server_batch`/llama context | 이번 encode/decode에 투입할 토큰과 실제 계산 상태 | update 반복과 context 수명 | admission 정책을 CUDA kernel scheduler로 본다 |
| `server_response` | task id별 결과 객체 | 서버 수명, 결과는 요청별 소비 | 스트림 조각이 곧 네트워크 전송이라고 본다 |

#### 상태를 로그 문장으로 번역하는 법

상태 이름만 외우지 말고 “다음에 무엇을 할 수 있는가”로 번역한다. idle slot은 새 task를 받을 수 있다. started 계열 상태는 task가 결합되어 프롬프트 준비가 필요하다. prompt 처리 상태에서는 아직 sampling 결과를 일반적인 decode 토큰처럼 취급하면 안 된다. generating 상태는 다음 입력 토큰을 batch에 넣고 결과 logits에서 sampling할 수 있다. release는 결과를 보낸 뒤 slot을 재사용 가능하게 만들며 deferred task 하나를 되살릴 수 있다.

로그를 볼 때는 `id_task`, `id_slot`, OpenAI 호환 completion id를 한 줄에 묶어야 한다. HTTP request id만 추적하면 parent/child task를 놓친다. slot id만 추적하면 같은 slot을 순차 사용한 서로 다른 요청을 합쳐 버린다. 최소 상관 키는 시간, task id, parent id, slot id, 외부 completion id, result 종류다.

## 25.2 HTTP 경계에서 task가 만들어지는 과정

### 25.2.1 HTTP 입력이 task가 되는 세 경계

#### 라우트는 입구이고 실행기는 아니다

`server.cpp`는 `/completion`, `/v1/completions`, `/chat/completions`, `/v1/chat/completions`, `/v1/responses`, embedding, rerank, tokenize 같은 경로를 `server_routes`의 핸들러에 연결한다. 이 목록은 [라우트 등록부](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server.cpp#L234-L274)에서 보인다. 그러나 라우트 등록을 읽고 “각 엔드포인트가 독립 추론 루프를 실행한다”고 생각하면 안 된다. completion 계열은 공통 구현으로 수렴하고 task/result 통로를 공유한다.

`handle_completions_impl()`은 외부 요청을 task들로 바꾸는 중심 경계다. completion id를 만들고, 파싱 결과에서 task를 구성하고, task별 state를 만들고, `server_response_reader`에 게시한 뒤 결과를 반복해서 받는다. 스트림이면 partial result를 SSE 형식으로 내보내고 final 또는 error에서 닫는다. 비스트림이면 필요한 결과를 모아 한 응답으로 만든다. 해당 흐름은 [completion 공통 핸들러](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L4175-L4310)에 고정되어 있다.

이 분리의 의도는 계산 스레드가 HTTP 쓰기 속도에 직접 종속되지 않게 하는 것이다. 다만 무한한 격리는 아니다. 결과 큐에 소비되지 않은 객체가 쌓이거나 HTTP 계층이 오랫동안 반환하지 못하면 메모리와 연결 자원이 압박받는다. “GPU가 계산을 끝냈다”와 “사용자가 응답을 받았다” 사이에는 result 생성, queue 전달, JSON 직렬화, SSE write가 남는다.

#### 채팅은 문자열 하나로 갑자기 변하지 않는다

채팅 요청에서는 chat template 적용, 도구 호출 표현, reasoning 구간, 응답 형식이 task 생성 전후에 걸쳐 있다. 입력 messages는 모델이 소비할 prompt 형태로 변환되고 tokenization을 거친다. 출력에서는 partial 문자열을 단순히 그대로 보내는 대신 `task_result_state`가 tool call과 chat message의 점진적 구조를 갱신한다. 그래서 스트림 조각의 경계는 토큰 경계, UTF-8 문자 경계, 도구 호출 JSON 경계와 항상 같지 않다.

`server_response_reader`가 `states` 벡터를 갖는 이유가 여기에 있다. task가 slot으로 move되더라도 HTTP 호출자가 응답 조립 상태를 잃으면 안 된다. `server_task::create_state()`가 chat parser parameter를 바탕으로 별도 상태를 만들고, partial/final result의 `update()`가 그 상태를 갱신한다. 모델 실행 상태와 프로토콜 조립 상태를 분리한 것이다.

운영에서 “모델은 정상인데 tool call 스트림이 깨진다”면 slot의 sampler부터 의심하지 않는다. 먼저 chat template과 parser 설정, partial result의 `update`, JSON/SSE 직렬화 경계를 본다. 반대로 첫 토큰 자체가 늦다면 template 시간, tokenize 시간, deferred 대기, prompt decode 시간을 구간별로 나눈다.

#### 배치 요청과 `n`은 task 수를 바꾼다

하나의 HTTP 요청이 task 하나라는 가정도 깨진다. 여러 prompt를 한 요청에 넣거나 `n`을 늘리면 parent와 child task가 생길 수 있다. 각 child는 부모의 params와 tokens를 복제하지만 slot을 명시하지 않으며, 고정 seed를 쓴 경우 child 순서에 따라 seed를 달리한다. 결과는 `index`와 task 관계를 이용해 다시 요청 수준으로 모인다.

따라서 `--parallel 4`인 서버에 `n=4` 요청 하나를 넣으면 “클라이언트 한 명이 slot 하나만 쓴다”는 보장이 없다. 자식들이 실행 자리를 차지해 다른 사용자의 head-of-line 대기를 늘릴 수 있다. 용량 계획에서는 HTTP RPS가 아니라 생성 branch 수, 평균 점유 시간, prompt 길이를 함께 본다. 제한 정책도 단순 연결 수보다 `n`, batch prompt 개수, 최대 생성 토큰에 적용해야 한다.

## 25.3 queue의 두 단계: 즉시 접수와 slot 대기

### 25.3.1 `post()`가 보장하는 것과 보장하지 않는 것

`server_queue::post()`는 mutex를 잡고 id가 없으면 새 id를 부여하고, 앞이나 뒤에 task를 넣고 condition variable을 깨운다. cancel task이면 아직 실행되지 않은 대상 task를 대기 자료구조에서 먼저 제거한다. 여기까지는 admission 완료가 아니라 “주 루프가 볼 수 있는 큐에 등록됨”이다. GPU 메모리 확보, slot 선택, prompt decode 성공은 보장하지 않는다. 구현은 [server-queue.cpp의 게시 경로](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-queue.cpp#L20-L82)에 있다.

`queue_tasks`는 지금 처리할 주 큐다. `queue_tasks_deferred`는 사용할 slot이 없어 미룬 inference task를 둔다. `queue_tasks_unhandled`는 `yield_to_queue()` 동안 worker가 받아 보았지만 그 시점에 안전하게 처리할 수 없어 잠시 제쳐 둔 task다. 세 큐는 이유가 다르다. deferred 증가를 HTTP ingress 폭주라고만 부르면 안 되고, unhandled를 영구 거절로 해석해서도 안 된다.

`defer()`는 task id를 바꾸지 않고 deferred 뒤에 넣는다. slot이 release될 때 `pop_deferred_task(id_slot)`이 호출되면 그 slot을 명시한 task를 먼저 찾고, 없으면 가장 앞 task를 주 큐 앞으로 옮긴다. 이 정책 때문에 특정 slot을 지정한 관리 또는 추론 요청과 일반 요청의 순서가 단순 FIFO와 다를 수 있다. [deferred 복귀 코드](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-queue.cpp#L76-L113)를 읽을 때 우선순위의 범위를 정확히 보아야 한다.

### 25.3.2 주 루프는 접수와 계산을 교대로 수행한다

`start_loop()`는 `process_new_tasks(false)`로 현재 주 큐를 비우고, `callback_update_slots()`를 호출한 뒤 새 task를 기다리는 구조다. callback은 초기화 때 `process_single_task()`와 `update_slots()`에 연결된다.

즉 queue는 모델 내부 스케줄러가 아니라 서버 이벤트 루프의 조정자다. wiring은 [초기화 코드](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1364-L1380), 반복은 [queue 주 루프](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-queue.cpp#L278-L366)에 있다.

중요한 결과가 두 가지다. 첫째, 새 task를 post했다고 실행 중 update 한가운데 임의로 slot 구조를 바꾸지 않는다. 정해진 callback 경계에서 처리한다. 둘째, 긴 encode/decode 작업이 주 루프를 오래 잡으면 metrics나 cancel 접수가 늦어질 수 있다. 이를 완화하기 위해 `yield_to_queue()`는 계산을 원래 주 스레드에서 수행하면서 별도 worker가 안전한 종류의 새 task를 처리할 기회를 준다.

`yield_to_queue()`는 일반적인 두 번째 추론 스레드가 아니다. ggml 계산은 원래 스레드에 남고 worker는 새 task callback을 돈다. yielding 동안 처리할 수 없는 task는 callback이 false를 반환하여 unhandled로 보내며, yield가 끝나면 원래 순서를 보존하도록 주 큐 앞에 되돌린다. 예외가 나도 task를 복원한 뒤 다시 던진다. 이 설계는 responsiveness를 높이되 slot 자료구조를 무제한 병렬 변경하지 않으려는 절충이다.

### 25.3.3 queue 정체를 네 구간으로 측정한다

첫째는 HTTP parse/tokenize 전 구간이다. 여기서 느리면 queue 크기는 작아도 요청 지연이 크다. 둘째는 post부터 `process_single_task()`까지의 주 큐 대기다. 셋째는 slot 부재로 인한 deferred 대기다. 넷째는 slot에 결합된 뒤 prompt/decode batch에서 실제로 전진하지 못하는 실행 대기다. 한 개의 “queue latency”로 합치면 해결책이 뒤집힌다.

주 큐 대기가 길고 slot은 남는다면 긴 update, lock 경쟁, 과도한 관리 task, CPU 전처리를 본다. deferred만 늘고 모든 slot 점유가 길다면 `--parallel`, context 분할, 생성 길이, 요청 branch 수를 본다. slot은 processing인데 토큰 진행이 없다면 batch 조립 실패, context 부족, backend 오류, 지나치게 큰 prompt chunk를 본다. result는 생성됐는데 클라이언트가 늦다면 response queue와 네트워크 write를 본다.

실용 로그에는 다음 시점을 넣는 것이 좋다. HTTP 수신, parse 완료, tokenize 완료, task post, task callback 진입, slot 선택, prompt 첫 batch 투입, prompt 완료, 첫 sampled token, 첫 partial result 생성, 첫 socket write, final result, slot release, reader stop이다. 이 시계열은 “TTFT가 느리다”를 실행 가능한 원인으로 바꾼다.

## 25.4 slot 선택: 명시적 id, prefix 유사도, LRU

### 25.4.1 선택 함수의 실제 우선순위

`get_available_slot()`은 먼저 task가 `id_slot`을 지정했는지 본다. 다음으로 `slot_prompt_similarity`가 켜져 있으면 processing 중이 아니고 prompt token이 남은 slot들의 longest common prefix를 계산한다. 현재 prompt 길이에 대한 공통 prefix 비율이 임계값보다 크며 기존 최선보다 큰 slot을 고른다. 선택하지 못하면 processing 중이 아닌 slot 가운데 `t_last_used`가 가장 오래된 자리를 고른다. 코드는 [slot 선택 함수](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1500-L1613)에 있다.

여기서 prefix 유사도는 문자열 유사도가 아니라 토큰열의 앞부분 일치다. chat template, system prompt, whitespace, tokenizer가 달라지면 사람이 비슷하다고 보는 요청도 공통 prefix가 짧다. 반대로 긴 공통 system prefix를 공유하는 요청은 사용자 질문이 달라도 재사용 후보가 된다. 따라서 cache hit를 높이려면 요청 라우팅뿐 아니라 template 안정성과 prefix 배치가 중요하다.

`f_keep`은 새 요청과 공통인 길이를 기존 slot prompt 길이로 나눈 값에 해당한다. 새 요청에 잘 맞는 slot이라도 기존 context의 큰 부분을 잃게 된다면 prompt cache 갱신 후보가 된다. cache 저장과 load가 실패하면 prompt를 clear한다. 선택 정책과 캐시 보존 정책이 한 함수에서 만나는 이유다.

### 25.4.2 LRU가 공정성 정책 전체는 아니다

LRU는 “사용 가능한 slot 중 가장 오래 안 쓴 자리”를 고른다. 실행 중 요청을 선점하지 않으며, 가장 짧은 남은 생성 시간을 추정하지도 않는다. 긴 생성이 모든 slot을 차지하면 짧은 요청은 deferred에서 기다린다. 그래서 이 서버의 slot LRU를 continuous batching scheduler의 preemption이나 fairness와 같다고 부르면 안 된다.

공정성을 개선하려면 먼저 외부 admission과 요청 제한을 생각한다. 사용자별 동시 branch 제한, `n` 제한, prompt와 output token 상한, 긴 요청 전용 풀, timeout과 cancel propagation이 효과적이다. slot 수를 무조건 늘리면 slot당 context 여유와 batch 효율, 메모리 압력이 달라질 수 있다. 선택 정책 하나가 전체 SLO를 해결하지 않는다.

### 25.4.3 캐시 옵션은 소비 지점까지 읽는다

`--cache-prompt`류 요청 동작, prompt similarity, RAM prompt cache, idle slot cache, unified KV 여부를 이름만 보고 해석하면 위험하다. 초기화 코드에서는 `cache_idle_slots`가 RAM cache 없이 요청되면 비활성화한다. unified KV이면 새 task 시작 때 idle slot을 저장하고 비워 재사용 공간을 만들 수 있지만, unified가 아니면 slot KV를 비워도 다른 slot이 쓸 공간이 생기지 않아 RAM 사본만 게시하고 VRAM KV는 남기는 경로가 있다. 이 조건은 [cache idle slot 초기화](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1382-L1397)에 명시되어 있다.

옵션을 설명할 때는 네 문장을 채운다. 어떤 필드에 저장되는가. 어느 함수가 읽는가. 어떤 분기가 달라지는가. 그 결과 latency, memory, correctness 중 무엇이 바뀌는가. 예컨대 prompt similarity 임계값은 막연히 “캐시 성능”을 올리지 않는다. 너무 높으면 재사용 후보를 놓치고 LRU로 간다. 너무 낮으면 공통 prefix가 짧은 slot을 고르는 비용과 cache churn이 커질 수 있다. 실제 workload의 prefix 분포와 선택 로그를 함께 봐야 한다.

## 25.5 task를 slot에 결합할 때 일어나는 일

### 25.5.1 `process_single_task()`는 종류별 dispatcher다

queue callback이 받은 모든 task가 inference slot을 요구하지는 않는다. metrics, slot save/restore/erase, LoRA 변경, cancel, next response 같은 제어 task는 각자의 즉시 처리 경로가 있다. completion, infill, embedding, rerank 같은 추론 task는 token 준비를 마친 뒤 slot을 찾는다. 사용할 자리가 없으면 `queue_tasks.defer()`로 이동한다. 이 분기를 [process_single_task](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L2305-L2390)에서 따라가면 “큐에 들어온 일”과 “slot을 쓰는 일”을 분리할 수 있다.

yielding worker가 callback을 실행할 때는 더 조심한다. 해당 순간 안전하게 처리할 수 없는 작업을 거절하면 queue 계층이 unhandled에 보관했다가 되돌린다. false 반환은 사용자 오류 응답이 아니라 잠정적인 처리 유예다. 이 의미를 모르고 로그의 declined를 실패율로 세면 잘못된 경보가 생긴다.

### 25.5.2 `launch_slot_with_task()`는 단순 포인터 대입이 아니다

slot에 task를 붙이는 함수는 요청별 LoRA 구성을 비교한다. adapter 변경이 cache 유효성을 깨는지 판단하고, 일반 LoRA와 aLoRA 조건을 구분한다. 여러 aLoRA를 잘못 요청하거나 activation sequence 조건을 만족하지 못하면 실행 전에 오류를 보낼 수 있다. 그 뒤 generation limit, sampler, prompt 상태, 통계, task 소유권을 설정한다. 시작 실패가 반드시 GPU decode 실패는 아닌 이유다.

이 경계에서 task는 move되어 slot의 `unique_ptr<const server_task>`가 된다. HTTP 쪽은 원본 task 객체를 붙잡아 결과를 기다리지 않는다. 대신 task id와 별도로 만든 `task_result_state`를 보존한다. move 이후 `child_tasks` 원소 접근이 유효하지 않다는 주석도 이 소유권 이동을 강조한다.

LoRA 옵션을 운영할 때는 모델 weight 적용만 보지 않는다. adapter 조합 변경이 이전 prompt KV 재사용을 허용하는지, slot별 adapter 상태가 어떤지, 요청별 adapter가 admission 시간을 늘리는지 확인한다. 동일 텍스트라도 adapter가 달라지면 의미 있는 hidden state가 달라지므로 잘못된 KV 재사용은 성능 문제가 아니라 정합성 문제다.

### 25.5.3 slot 시작 실패의 진단 순서

첫째, task parse 오류와 tokenization 오류를 확인한다. 둘째, 특정 slot id가 존재하며 사용 가능한지 본다. 셋째, LoRA/aLoRA 제약과 prompt 길이, context 한도를 본다. 넷째, prompt cache load 또는 memory sequence 조작이 실패했는지 본다. 다섯째, batch에 prompt token을 넣을 공간과 backend decode 반환을 본다. 모든 실패를 “slot 부족”으로 묶지 않는다.

오류 응답에는 task id와 가능하면 slot id가 있어야 한다. slot 결합 전 오류는 id_slot이 없을 수 있다. 결합 후 오류는 release가 반드시 뒤따르는지 확인한다. 오류를 보냈지만 slot이 processing에 남으면 deferred 큐가 계속 늘어나는 2차 장애가 생긴다.

## 25.6 `update_slots()` 안의 prefill과 decode

### 25.6.1 한 번의 update는 모든 slot을 한 토큰씩만 돌린다는 뜻이 아니다

`update_slots()`는 processing slot을 순회하며 상태에 맞는 입력을 `server_batch`에 넣고 llama decode/encode를 호출하고 결과를 다시 slot에 배분한다. 프롬프트가 긴 slot은 prompt token 여러 개를 batch에 넣을 수 있고, generation slot은 보통 직전 sampled token을 다음 위치의 입력으로 넣는다. 여러 slot의 입력이 하나의 batch 호출에 공존할 수 있다. 이것이 서버 수준의 병렬 진행이다.

prefill과 decode의 비용 구조는 다르다. prefill은 많은 토큰의 attention과 KV 생성을 한꺼번에 수행해 처리량 중심이다. autoregressive decode는 slot마다 새 위치 하나를 반복하며 기존 KV를 읽어 memory bandwidth와 반복 지연에 민감하다. 긴 prompt가 큰 chunk로 들어오면 decode 중인 다른 slot의 다음 token 시간이 늘 수 있다. 반대로 prompt chunk를 지나치게 작게 하면 호출 오버헤드와 전체 prefill 시간이 늘 수 있다.

따라서 “continuous batching을 지원한다”는 한 문장만으로 latency 특성을 설명할 수 없다. batch capacity, physical batch 설정, slot 수, context 분배, prompt chunking, speculative decoding, backend가 함께 결정한다. 코드 독자는 `update_slots()`에서 어떤 상태가 token을 batch에 넣는지, `i_batch`가 결과 위치와 어떻게 연결되는지, decode 후 어느 slot이 sampling되는지를 순서대로 추적해야 한다.

### 25.6.2 prompt cache hit도 계산이 0이 되는 마법은 아니다

새 task의 tokens와 slot prompt의 공통 prefix를 찾고 memory sequence가 보유한 부분을 재사용하면 그 prefix의 forward를 줄일 수 있다. 하지만 바뀐 suffix는 처리해야 하며, sampler와 생성 상태는 새 요청에 맞게 초기화해야 한다. context shift, keep 정책, LoRA 변경, multimodal 입력, draft context가 개입하면 재사용 범위가 더 복잡해진다.

캐시 관측에는 최소한 입력 token 수, cache된 prompt token 수, 실제 처리 token 수, cache load 시간, 첫 decode 시각을 둔다. hit 여부 boolean 하나는 효과를 설명하지 못한다. 4,096 토큰 중 50 토큰 재사용과 4,000 토큰 재사용은 같은 hit가 아니다. `result_prompt_progress`와 final result의 `n_prompt_tokens_cache`, slot stats를 함께 보아야 한다.

### 25.6.3 sampling과 다음 batch 사이의 고리

모델 호출 결과 logits는 slot의 `i_batch`가 가리키는 결과 위치와 연결된다. sampler가 token을 고르고 `process_token()`이 텍스트 조각, stop word, EOS, 최대 생성 길이, 확률 출력, UTF-8 전송 가능 범위를 갱신한다. `has_next_token`이 유지되면 선택한 token이 다음 update의 입력 후보가 된다. 종료 조건이면 final result를 만들고 slot을 release한다.

여기서 streaming은 모델이 문자열을 직접 소켓에 쓰는 기능이 아니다. sampling 결과를 token text로 누적하고, 아직 보내지 않은 안전한 문자열 범위를 계산하여 partial result 객체를 만든다. HTTP reader가 그 객체를 프로토콜 형식으로 직렬화한다. 그래서 token 생성 시간과 stream write 시간은 별도 지표여야 한다.

`process_token()`과 응답 생성 경계는 [토큰 처리와 응답 함수](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1779-L2075)에서 볼 수 있다. 생성이 멈췄을 때는 `stop`, `stopping_word`, `has_next_token`, `n_predict_max`, sampler 상태를 함께 읽는다.

### 25.6.4 긴 계산 중 제어 요청이 살아남는 방법

`update_slots()` 내부의 일부 encode/decode 작업은 `queue_tasks.yield_to_queue()`로 감싼다. 계산 자체는 주 스레드에서 계속하지만 worker가 metrics, cancel처럼 허용되는 task를 받아볼 수 있다. 이 장치가 없다면 큰 prompt batch 하나가 끝날 때까지 HTTP 쪽이 취소 task를 게시해도 주 callback이 보지 못한다.

그러나 yield가 즉각적인 kernel 중단을 뜻하지는 않는다. 이미 backend에 제출된 계산을 token 단위로 임의 선점하는 장치가 아니다. 취소 반응 시간의 하한은 현재 실행 단위, yield 위치, backend 호출 시간에 좌우된다. 운영 timeout을 정할 때 “소켓이 닫힌 시각”과 “slot이 release된 시각”의 차이를 측정해야 한다.

## 25.7 partial, final, error가 결과 큐를 건너는 법

### 25.7.1 결과는 다형 객체이고 id로 라우팅된다

`server_task_result` 기본형은 `id`, `id_slot`, batch `index`를 갖고 `to_json()` 계약을 제공한다. completion partial은 새 content와 token, progress, 확률, begin 여부를 담는다. final은 전체 content, 생성 token, stop 이유, prompt/cache token 수, generation params, 통계를 담는다. error는 오류 유형과 메시지를 담는다. 정의는 [result 타입 계층](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-task.h#L272-L430)에 있다.

`server_response`는 기다리는 task id 집합과 결과 벡터를 mutex와 condition variable로 보호한다. `send()`는 실제로 누군가 기다리는 id의 결과를 넣고 깨운다. `recv()`는 요청이 기다리는 id 중 하나에 해당하는 결과가 올 때까지 막힌다. 이 구조 덕분에 여러 HTTP 스레드가 하나의 서버 실행 루프에서 돌아온 결과를 자기 요청별로 소비한다.

결과 큐는 생성 순서와 네트워크 수신 순서 사이의 완충이다. 하지만 per-client 무한 버퍼로 설계했다고 추정하면 안 된다. 느린 소비자, 큰 logprobs, 매우 잦은 stream chunk가 메모리를 키울 수 있는지 소스와 부하에서 확인해야 한다. backpressure 정책이 명시적이지 않은 구간은 reverse proxy의 write timeout, 최대 연결, 요청 제한으로 방어한다.

### 25.7.2 `is_begin`, `is_progress`, `is_stop`을 섞지 않는다

stream 시작을 알리는 partial, prompt 진행을 알리는 partial, 실제 생성 text partial은 목적이 다르다. `is_begin`은 SSE에서 성공 상태를 시작할 시점을 표현할 수 있고, `is_progress`는 prompt 처리 진척을 담는다. `is_stop()`은 reader가 해당 task의 결과 수명을 닫을지 판단하는 계약이다. final은 stream 모드에서도 stop으로 취급된다.

클라이언트 구현은 빈 content chunk를 무시한다는 이유로 begin이나 usage 정보를 버리면 안 된다. 서버 디버깅에서도 partial 개수만 token 수로 세면 prompt progress와 구조 이벤트 때문에 틀릴 수 있다. token throughput은 slot stats와 decoded token 수에서, 전송 chunk throughput은 response 계층에서 따로 계산한다.

### 25.7.3 최종 응답은 slot release와 같은 사건이 아니다

서버는 final result를 구성해 queue에 보낸 뒤 slot 상태를 정리하고 release한다. HTTP reader가 final을 받아 JSON을 쓰는 것은 더 나중일 수 있다. 따라서 세 종료 시각을 둔다. 모델 생성 종료, slot 재사용 가능, 클라이언트 write 완료다. 첫째와 둘째의 차이가 크면 cleanup/cache 저장을 보고, 둘째와 셋째의 차이가 크면 result 소비와 네트워크를 본다.

오류도 마찬가지다. parse 오류는 task post 전에 HTTP에서 끝날 수 있다. slot 시작 오류는 result queue로 전달되지만 계산은 시작하지 않았을 수 있다. decode 오류는 여러 slot에 영향을 줄 수 있고 각 task에 오류를 보내며 상태를 복구해야 한다. status code 하나로 원인을 분류하지 말고 발생 경계와 정리 결과를 기록한다.

## 25.8 취소, 연결 종료, 오류, cleanup

### 25.8.1 reader의 RAII가 중요한 이유

`server_response_reader`는 기다리는 task id 집합을 소유하고 destructor에서 `stop()`을 부른다. 정상 응답뿐 아니라 HTTP 핸들러가 예외로 빠지거나 연결이 끊겨 scope를 벗어날 때도 남은 task를 취소하는 안전망이다. reader가 결과 waiting set에서 id를 제거하고 cancel task를 게시해야 서버가 더 이상 소비자가 없는 생성을 계속하지 않는다. 선언은 [server queue와 response reader](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-queue.h#L15-L220)에 있다.

RAII는 즉시 계산 중단 보장이 아니다. cancel task가 queue에 도착하는 시간, pending cleanup, 실행 slot 탐색, 현재 backend 호출 종료가 필요하다. 그래도 예외 경로마다 수동 취소를 빠뜨리는 것보다 수명 경계를 구조적으로 묶는다. 코드 리뷰에서는 모든 early return이 reader 생성 전인지 후인지, stop이 id 집합을 중복 처리해도 안전한지 본다.

### 25.8.2 대기 중 취소와 실행 중 취소는 경로가 다르다

`post()`가 cancel task를 받으면 `cleanup_pending_task(id_target)`이 주 큐, deferred 큐, yield 중 unhandled 큐에서 아직 실행되지 않은 대상 task를 제거한다. 그 뒤 cancel task 자체는 callback으로 가서 실행 slot에 결합된 대상을 찾고 중단·release한다. 즉 pending 제거와 active slot 정리는 한 연산이 아니다. [pending cleanup](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-queue.cpp#L368-L380)은 큐 자료구조만 다룬다.

경쟁 조건을 생각해 보자. cancel이 게시되기 직전에 원 task가 주 큐에서 빠져 slot으로 move되었다면 pending cleanup은 찾지 못한다. cancel callback이 slot의 task id를 찾아야 한다. 반대로 아직 deferred라면 slot 탐색만으로는 지워지지 않는다. 두 경로가 모두 필요한 이유다.

parent/child가 있으면 취소 대상 집합도 중요하다. HTTP 요청이 여러 task id를 기다리는 경우 reader는 남은 id 전체를 정리해야 한다. 한 child의 final을 받았다고 sibling을 취소하면 `n` 응답이 불완전해지고, 연결 종료 때 parent만 취소하고 child를 남기면 고아 생성이 slot을 점유한다. `server_task::get_list_id()`와 reader의 id set을 함께 추적한다.

### 25.8.3 cleanup의 불변식

정상, stop word, token limit, cancel, backend error 어느 경로든 다음 불변식을 확인한다. 기다리는 id는 결과 소비 또는 취소 후 제거된다. 실행 task는 slot에서 빠진다. sampler와 요청별 생성 버퍼는 reset된다. memory sequence는 정책에 따라 보존 또는 제거되며 우연히 다음 요청과 섞이지 않는다. slot은 재사용 가능한 상태가 된다. release callback이 deferred task를 주 큐로 되돌릴 기회를 준다.

장애 주입을 실행하지 않더라도 코드 리뷰로 각 return 경로를 표로 만들 수 있다. 오류를 보내는 함수 뒤에 release가 호출되는가. release가 callback과 reset 순서를 지키는가. cache save 실패가 slot을 processing에 남기지 않는가. result를 보낼 waiting reader가 이미 사라졌을 때 객체가 누적되지 않는가. terminate가 queue와 response wait를 모두 깨우는가. 이 표가 취소 안정성의 최소 증거다.

## 25.9 옵션을 상태 변화로 읽는 운영 해설

### 25.9.1 옵션을 다섯 상태 변화로 번역한다

#### `--parallel`은 무엇을 늘리는가

`--parallel`은 서버가 준비하는 slot 수와 동시 sequence 수의 상한에 영향을 준다. 값을 늘리면 더 많은 task가 slot에 결합될 가능성이 있지만, GPU가 그 수만큼 독립 kernel을 동시에 실행한다는 뜻은 아니다. 여러 slot의 token이 공통 batch와 context에 들어간다. 모델 context와 KV 설정에 따라 slot당 이용 가능한 문맥, 메모리 사용, batch 구성 효율이 바뀐다.

값을 올릴 때 확인할 효과는 deferred 대기 감소, active slot 증가, slot당 prompt/decode 진행 속도 변화, KV 메모리 여유, TTFT와 inter-token latency의 꼬리다. 평균 throughput만 좋아지고 긴 요청과 짧은 요청의 꼬리가 나빠질 수 있다. 반대로 너무 작으면 GPU 여유가 있어도 긴 generation이 자리를 독점하여 admission이 막힌다.

#### batch와 micro-batch 설정은 admission 수가 아니다

논리 batch 크기와 물리 micro-batch 설정은 한 llama decode 호출에 구성할 token 작업량과 backend 실행 단위를 제한한다. slot 수와 같은 축이 아니다. slot이 많아도 batch capacity가 작으면 한 update에서 일부 입력만 전진하거나 prompt가 여러 조각으로 나뉜다. batch를 키워도 slot이 하나이고 decode만 한다면 새 token 후보가 제한되어 이점이 작을 수 있다.

설정 변경 전에는 workload를 prompt-heavy, decode-heavy, 혼합으로 나눈다. prompt-heavy에서는 prefill throughput과 첫 토큰 꼬리를 함께 본다. decode-heavy에서는 active sequence 수에 따른 token latency와 memory bandwidth를 본다. 혼합에서는 긴 prefill이 기존 decode를 얼마나 방해하는지 본다. OOM만 피했다고 최적값은 아니다.

#### context와 KV 설정은 slot의 의미를 바꾼다

전체 context, unified KV, KV data type, offload와 관련된 설정은 slot들이 사용할 memory 공간과 재사용 방식을 바꾼다. non-unified 구성에서는 한 idle slot의 영역을 비워도 다른 slot이 곧바로 쓸 수 없을 수 있다. unified 구성에서는 sequence별 memory 조작과 공간 공유가 가능하지만 fragmentation, cache 정책, context shift를 함께 봐야 한다.

운영자는 “모델 최대 context가 128K이므로 parallel 8에서도 요청마다 128K”라고 가정하면 안 된다. 실제 server context 배분과 시작 로그, slot의 `n_ctx`, 요청 제한을 확인한다. 긴 prompt 허용 정책은 단일 요청 성공뿐 아니라 다른 slot의 KV와 batch 시간을 잠식한다.

#### streaming과 timeout은 계산 옵션이 아니다

`stream=true`는 partial result 생성과 HTTP 전송 방식을 바꾸지만 모델이 autoregressive하게 계산한다는 사실은 같다. stream을 꺼도 내부 token loop는 돈다. 다만 전체 결과를 모으는 메모리와 사용자가 체감하는 응답 시점이 달라진다. reverse proxy buffering이 켜져 있으면 서버가 partial을 만들어도 사용자는 늦게 볼 수 있다.

polling interval, HTTP write timeout, proxy idle timeout, client cancel은 서로 다른 시계다. timeout이 너무 짧으면 정상적인 긴 prefill 중 연결이 끊기고 cancel이 몰린다. 너무 길면 고아 요청이 오래 자원을 잡는다. prompt progress 또는 heartbeat가 실제 네트워크를 통과하는지 확인하고, cancel-to-release 지연을 별도 SLO로 둔다.

#### prompt cache와 slot save는 같은 기능이 아니다

자동 prefix 재사용, RAM prompt cache, idle slot 저장, `/slots/:id_slot` 관리 API는 목적과 수명이 다르다. 자동 재사용은 admission 시 기존 prompt와 새 task를 비교하는 정책이다. RAM cache는 상태를 다른 저장 층에 보존한다. slot save/restore는 명시적 관리 작업이며 파일 경로와 slot id를 다룬다. 이들을 모두 “KV cache 켜기”로 설명하면 보안과 성능 판단이 흐려진다.

명시적 저장은 파일 I/O 시간, 경로 검증, 모델·adapter 호환성, 복원 중 slot 가용성을 고려해야 한다. 자동 cache는 hit 분포와 eviction을 본다. unified memory는 공간 공유 의미를 본다. 각각의 consumer 함수와 오류 결과를 따로 문서화한다.

## 25.10 장애를 역추적하는 실전 워크북

### 25.10.1 증상에서 최소 조사 경로를 고른다

#### 증상 A: GPU 사용률은 낮은데 첫 토큰이 늦다

먼저 요청 한 건의 여섯 시간을 잰다. parse, template/tokenize, 주 큐, deferred, prompt processing, 첫 result-to-write다. deferred가 대부분이면 active slot의 생성 길이와 branch 수를 본다. 주 큐가 길면 update 호출 시간과 yield 가능 구간을 본다. prompt processing이 길면 cache된 token 비율, prompt 길이, batch 설정, CPU/GPU offload를 본다. write만 길면 proxy buffering과 느린 client를 본다.

낮은 평균 GPU 사용률은 CPU tokenization, 짧은 반복 kernel, 동기화, 작은 batch, slot starvation을 모두 가릴 수 있다. `--parallel`을 먼저 올리지 않는다. idle slot 수, batch token 수, decode 호출 시간, CPU core 사용, backend 로그를 같은 시간축에 놓는다.

#### 증상 B: 모든 slot이 찼고 짧은 요청이 굶는다

각 slot의 task 시작 시각, prompt 길이, 생성 token 수, 최대 생성 한도, parent/child 관계를 덤프한다. 긴 무제한 generation과 큰 `n`이 자리를 차지하는지 확인한다. deferred 큐의 대기 나이 분포를 본다. LRU 선택은 실행 중 slot을 빼앗지 않으므로 이 상황을 해결하지 못한다.

해결 후보는 output 상한, branch 제한, 요청 class별 인스턴스, 외부 queue deadline, 사용자별 concurrency다. parallel 증가는 메모리와 token latency 검증 뒤 적용한다. cancel이 release까지 실제 도달하는지도 확인한다. 연결은 사라졌는데 slot task가 남는다면 reader 수명과 cancel callback을 추적한다.

#### 증상 C: 스트림이 뭉쳐서 오거나 중간에 깨진다

partial result 생성 시각과 socket write 시각을 분리한다. 서버가 일정하게 partial을 만들면 HTTP library, proxy buffering, TLS, client parser를 본다. 생성 자체가 뭉치면 UTF-8 안전 경계, stop word 보류, tool call parser, prompt progress와 text chunk 구분을 본다. logprobs를 크게 요청했을 때 serialization 크기도 측정한다.

마지막 chunk가 없으면 final result 생성, reader의 `is_stop`, SSE 종료 marker, 연결 예외 경로를 차례로 본다. slot은 release됐는데 client만 기다리면 계산 계층 문제가 아니다. client는 끝났는데 slot이 남으면 cleanup 계층 문제다.

#### 증상 D: 취소 후에도 계산이 오래 간다

client disconnect, reader stop, cancel post, cancel callback, active slot 발견, backend 호출 반환, slot release 시각을 기록한다. pending task라면 `cleanup_pending_task`가 어느 deque에서 제거했는지 본다. active라면 현재 yield 지점까지의 시간이 핵심이다. parent/child id 전체가 취소됐는지도 확인한다.

긴 단일 prefill 호출이 지연의 원인이면 prompt chunk와 batch 설정이 취소 반응성에 주는 영향을 평가한다. 하지만 작은 chunk는 throughput을 낮출 수 있다. 취소 즉시성을 공짜 기능으로 보지 말고 계산 granularity와의 trade-off로 문서화한다.

#### 증상 E: cache를 켰는데 더 느려졌다

hit boolean 대신 공통 prefix token 수와 cache 처리 시간을 잰다. similarity 임계값 때문에 작은 prefix를 가진 slot이 자주 선택되는지, 기존 긴 context를 저장하느라 I/O가 생기는지, LoRA 변경으로 cache가 지워지는지, template 변동으로 token prefix가 흔들리는지 본다. cache load 실패 후 clear 경로도 센다.

워크로드가 거의 공통 prefix를 공유하지 않으면 검색과 저장 비용만 늘 수 있다. 반대로 system prompt가 길고 안정적이면 라우터가 같은 모델·adapter·template 요청을 같은 인스턴스로 보내는 것이 유리하다. cache 정책은 서버 내부 옵션과 상위 라우팅 키가 함께 만든다.

#### 최소 관측 필드와 경보

요청 수준에는 request/completion id, task id 집합, model, stream, prompt token, output limit, `n`, adapter, 도착·완료·취소 이유를 기록한다. slot 수준에는 id, state, task id, prompt/cache/processed token, generated token, last-used, prompt/decode 시간, release 이유를 기록한다. queue 수준에는 main/deferred/unhandled 길이와 가장 오래 기다린 나이를 둔다. result 수준에는 partial/final/error 개수, 생성부터 소비까지 지연, waiting id 수를 둔다.

경보는 deferred 길이 하나보다 조건 조합이 낫다. deferred 나이가 증가하면서 모든 slot이 processing이고 generation age가 높으면 admission 포화다. main queue age가 증가하지만 idle slot이 있으면 event loop 지연이다. result 소비 지연만 증가하면 네트워크 또는 client backpressure다. cancel-to-release가 길면 yield granularity나 cleanup 결함을 본다.

#### 소스 읽기 체크리스트

1. 라우트에서 공통 handler까지 실제 호출을 잇는다.
2. JSON 필드가 `task_params`의 어느 필드가 되는지 기록한다.
3. task 수가 prompt 개수와 `n` 때문에 어떻게 늘어나는지 계산한다.
4. reader가 등록한 waiting id 집합을 확인한다.
5. post 시 id 부여와 cancel pending cleanup을 확인한다.
6. `process_single_task()`에서 task type별 분기를 표로 만든다.
7. slot 부재 시 deferred되고 release 시 어떻게 복귀하는지 잇는다.
8. `get_available_slot()`의 명시 id, prefix, LRU 순서를 확인한다.

9. `launch_slot_with_task()`의 adapter, sampler, limit, ownership 이동을 확인한다.
10. `update_slots()`의 상태별 batch 입력과 `i_batch`를 추적한다.
11. decode 결과에서 sampler와 `process_token()`을 잇는다.
12. partial/final/error가 어느 id로 `server_response`에 들어가는지 본다.
13. reader가 stream state를 갱신하고 언제 stop하는지 본다.
14. 정상·취소·오류 각각에서 release/reset/deferred wake-up을 확인한다.
15. 옵션마다 저장 필드, consumer, 분기, 관측 효과를 적는다.

## 25.11 다른 서빙 스택과의 정확한 경계

### 25.11.1 vLLM과 비교할 때

vLLM의 OpenAI API server도 HTTP 요청을 engine request로 넘기고 비동기 결과를 stream한다. 그러나 vLLM scheduler의 sequence scheduling, token budget, KV block manager와 llama.cpp server slot을 일대일 대응시키면 안 된다. llama.cpp slot은 요청 실행 상태를 담는 고정 자리 성격이 강하고, 실제 llama batch/context가 계산을 수행한다. vLLM은 다수 request를 scheduler iteration마다 선택하고 paged KV block 관리와 더 직접 결합한다.

비교 단위는 이름이 아니라 질문이다. admission을 누가 결정하는가. 실행 중 preemption이 있는가. KV의 논리 sequence와 물리 block을 누가 매핑하는가. prefill과 decode를 어떤 budget으로 섞는가. 결과 stream의 취소가 scheduler에 언제 도달하는가. 이 질문에 답한 뒤에야 `slot`, `sequence group`, `request`를 대응시킬 수 있다.

### 25.11.2 SGLang과 비교할 때

SGLang은 tokenizer manager와 scheduler 사이의 IPC, running batch, radix prefix cache, model worker 실행 경계가 더 명시적으로 분리된다. llama.cpp server는 한 프로세스의 queue callback과 slot update 안에 많은 조정이 모여 있다. SGLang의 radix cache node를 llama.cpp slot과 같다고 보면 안 된다. radix node는 공유 prefix 구조이고 slot은 현재 요청 실행 자리다.

SGLang에서 ingress 지연을 볼 때 tokenizer manager queue와 scheduler queue를 나누듯, llama.cpp에서도 HTTP 전처리, main task queue, deferred slot queue를 나눠야 한다. 차이는 프로세스 토폴로지이지 관측 질문의 필요성이 아니다.

### 25.11.3 Transformers와 비교할 때

Transformers의 `generate()`는 일반적으로 한 호출의 입력, generation configuration, logits processor, stopping criteria, cache를 Python 호출 범위에서 조정한다. 자체적으로 HTTP admission, 여러 연결의 waiting id, reusable server slot을 제공하는 서버 스케줄러는 아니다. llama.cpp의 sampler와 stop 처리를 Transformers generation loop와 비교할 수는 있지만, `server_queue`까지 `generate()`에 대응시키면 층이 맞지 않는다.

Transformers 기반 서버를 만들 때 이 장의 객체 분리가 설계 체크리스트가 된다. request protocol state, normalized generation job, reusable execution state, shared model batch, async result channel, cancellation ownership을 누가 맡는지 명시해야 한다. 라이브러리 함수 하나를 호출한다고 이 문제가 사라지지 않는다.

### 25.11.4 경계 비교표

| 질문 | llama.cpp server | vLLM | SGLang | Transformers 단독 |
|---|---|---|---|---|
| 외부 요청 정규화 | route와 task params | API server와 engine request | tokenizer manager | 호출자 코드 |
| admission 자리 | fixed server slots와 deferred queue | scheduler 대상 request/sequence | scheduler running/waiting batch | 기본 제공 없음 |
| prefix 재사용 대표 구조 | slot prompt와 prompt cache/memory sequence | block 기반 prefix cache | radix cache | model cache를 호출자가 관리 |
| 실제 계산 묶음 | `server_batch`와 llama context | scheduler output과 worker batch | running batch와 model worker | `generate()` 내부 batch |
| 결과 전달 | `server_response`와 reader | async output stream | manager/scheduler 반환 통로 | 함수 반환 또는 streamer |
| 연결 종료 정리 | reader가 cancel task 게시 | engine abort 경로 | request abort 경로 | 호출자가 구현 |

이 표는 우열표가 아니다. 같은 문제를 어느 객체와 프로세스 경계에서 푸는지를 보여준다. llama.cpp를 분석한 경험을 다른 엔진에 옮길 때는 함수 이름이 아니라 수명과 소유권을 옮긴다.

### 25.11.5 대표 요청 한 건을 시계열로 복원한다

구조를 실제 장면에 대입해 보자. 클라이언트 A가 `stream=true`, 2,000개 prompt token, 최대 300개 output token으로 `/v1/chat/completions`를 호출한다. 같은 시각에 네 slot은 모두 generation 중이다. HTTP 핸들러는 요청을 파싱하고 chat template을 적용하며 토큰열을 만든다. 이때 아직 slot은 점유하지 않는다. task가 만들어지고 reader가 그 id를 waiting set에 등록한 뒤 queue에 post한다. 주 큐 대기는 짧지만 `process_single_task()`가 사용 가능한 slot을 찾지 못하므로 task는 deferred로 간다.

이 시점의 정확한 관측 문장은 “A는 tokenization을 끝냈고 내부 task id를 받았지만 실행 자리를 얻지 못했다”이다. “A가 GPU queue에 들어갔다”라고 쓰면 너무 많은 층을 건너뛴다. GPU backend에는 A의 graph가 제출되지 않았다. `server_slot::task`에도 A가 없다. A의 토큰은 `server_task`가 소유한 채 deferred deque에 있다. client connection과 reader는 결과를 기다린다.

기존 요청 B가 stop token을 생성한다. `process_token()`이 더 이상 다음 token이 없다고 판정하고 final result를 보낸다. slot release 과정은 B의 요청별 상태를 reset하고 callback을 통해 `pop_deferred_task(released_slot_id)`를 부를 기회를 만든다. deferred의 A가 특정 slot을 요구하지 않았다면 queue 앞쪽으로 돌아간다. HTTP A의 스레드가 task를 다시 post한 것이 아니다. 같은 id와 params를 가진 move된 task가 내부 큐 사이를 이동한다.

다음 주 루프에서 A가 다시 `process_single_task()`에 들어간다. `get_available_slot()`은 이제 idle인 B의 옛 slot을 포함해 후보를 본다. 다른 idle slot에 A와 공통 prefix가 충분히 긴 prompt가 남아 있다면 similarity 정책이 그 자리를 고를 수 있다. 그렇지 않으면 last-used 시간이 오래된 idle slot을 고른다. 선택된 slot의 이전 요청과 A의 adapter 구성이 다르면 `launch_slot_with_task()`는 cache 보존 가능성을 다시 판단한다. A의 task 소유권이 slot으로 이동한 뒤에야 admission이 끝난다.

첫 `update_slots()`에서 공통 prefix로 재사용할 수 없는 A의 suffix가 prompt batch에 들어간다. batch에는 다른 generation slot의 한 token 입력들도 함께 있을 수 있다. 따라서 A의 prefill은 혼자 실행되는 별도 HTTP worker가 아니다. backend decode가 반환하면 A의 prompt progress가 갱신되고, 아직 prompt가 남았다면 다음 chunk를 준비한다. 기존 slot들은 자기 logits를 sampling하고 다음 입력을 준비한다. 이 반복에서 A의 긴 prefill 때문에 다른 사용자 token 간격이 늘어날 수 있다.

A의 prompt가 모두 처리된 뒤 첫 logits가 sampling된다. token text가 완전한 UTF-8 조각이고 stop word 탐지를 위해 보류할 필요가 없다면 partial result가 `server_response`에 들어간다. reader가 id로 결과를 받고 chat parser state를 갱신하고 OpenAI 호환 SSE JSON으로 만든다. 이때 first sampled token, first partial enqueue, first socket write는 서로 다른 시각이다. 정확한 TTFT 분석은 세 값을 모두 기록한다.

200번째 output token에서 client A가 브라우저 탭을 닫는다. HTTP handler scope가 끝나 reader destructor가 `stop()`을 실행한다. cancel task가 post되고 pending queue에서 같은 id를 청소한다. 하지만 A는 이미 slot에서 실행 중이므로 pending 제거만으로 끝나지 않는다. cancel callback이 active slot의 task id를 찾아 release해야 한다. 현재 llama backend 호출이 길면 cancel callback이 실행될 수 있는 yield 경계까지 기다린다. client disconnect와 GPU 계산 정지가 동시에 일어나지 않는 이유다.

이 시계열에서 객체별 상태를 표로 남기면 race를 찾기 쉽다.

| 순간 | HTTP/reader | task queue | slot | model batch | result queue |
|---|---|---|---|---|---|
| parse 완료 | 연결 유지, state 준비 | 아직 없음 | 변화 없음 | 변화 없음 | 변화 없음 |
| post | waiting id 등록 | main에 A | 변화 없음 | 변화 없음 | 없음 |
| slot 포화 | 기다림 | deferred에 A | 기존 네 task 실행 | 기존 task 입력 | 기존 결과 |
| B release | 기다림 | A가 main으로 복귀 | 한 자리 idle | 다음 update 대기 | B final |
| A launch | 기다림 | A 제거 | A task 소유 | prompt 입력 예정 | 없음 |
| 첫 sampling | 기다림 | 변화 없음 | A 생성 상태 | A logits 반환 | A partial |
| disconnect | reader stop | cancel task | A가 아직 있을 수 있음 | 현재 호출이 끝나는 중 | 미소비 partial 가능 |
| cancel 처리 | handler 종료 | cancel 소비 | A release/reset | 이후 A 제외 | waiting id 제거 |

이 표를 운영 사고마다 채우면 “요청이 어디 있다”라는 말이 모호하지 않다. 특히 task 객체가 move된다는 점 때문에 같은 순간 main queue와 slot 양쪽에 정상적으로 동시에 존재해서는 안 된다. 관측 코드가 복사본을 만들어 보여 준다면 snapshot 시각을 표시해야 한다. 서로 다른 시각의 메트릭을 한 화면에 놓고 중복 실행으로 오해하지 않도록 한다.

### 25.11.6 세 경쟁 조건을 코드 리뷰로 검증한다

첫 번째는 release와 새 admission의 경쟁이다. slot B가 final을 보내고 release되는 동안 새 task A가 도착한다. queue와 slot이 아무 잠금 규칙 없이 다른 HTTP 스레드에서 직접 바뀐다면 A가 반쯤 reset된 slot을 볼 수 있다. 이 구현은 HTTP가 task를 post하고 주 loop의 callback 경계에서 slot을 변경하도록 좁힌다. `callback_on_reset`이 reset 전에 호출되어야 한다는 주석도 순서가 계약임을 보여 준다. 리뷰에서는 result send, release callback, task move, state idle 전환의 실제 순서를 한 줄씩 확인해야 한다.

두 번째는 cancel과 admission의 경쟁이다. 대상 task가 main deque에 있을 때 cancel이 오면 `cleanup_pending_task()`가 제거한다. 대상이 이미 pop되어 callback 지역 변수에 있고 아직 slot에 move되지 않았다면 어느 정리 경로가 책임지는지 확인해야 한다. yielding 경로에서는 거절된 task가 `queue_tasks_unhandled`에 있을 수 있어서 cleanup이 그 deque도 검사한다. 이것은 사소한 자료구조 추가가 아니다. 그 deque를 빼먹으면 취소된 요청이 yield 종료 후 되살아나는 결함이 된다.

검증 방법은 대상 task 위치를 상태 공간으로 나누는 것이다. 위치는 main, deferred, unhandled, callback local, slot, completed/result sent다. cancel이 각 위치에서 도착했을 때 다음 owner가 누구인지 적는다. main/deferred/unhandled는 queue cleanup, slot은 context cancel handler, completed는 waiting id와 결과 소비 정리가 담당한다. callback local의 짧은 창은 callback 실행 모델과 cancel task 처리 순서를 읽어 확인한다. “cancel 함수가 있다”가 아니라 모든 위치를 덮어야 한다.

세 번째는 final result와 disconnect의 경쟁이다. 서버가 final을 `send()`하는 순간 reader가 waiting id를 제거할 수 있다. response queue는 mutex 아래 waiting set을 확인하고 결과를 넣어야 한다. reader stop은 id 제거와 cancel 게시가 중복되어도 이미 완료한 task를 다시 위험하게 release하지 않아야 한다. final을 받은 뒤 scope 종료가 cancel을 무조건 만들면 불필요한 cancel task가 생길 수 있으므로 `cancelled`, received count, id set 갱신을 본다.

이 경쟁은 스트림에서 더 잘 드러난다. 마지막 partial을 받은 뒤 final이 오기 전에 network write가 실패할 수 있다. reader destructor는 아직 기다리는 id를 취소해야 한다. 반면 final을 이미 받아 stop 상태로 id를 제거했다면 소켓 write 실패가 모델 실행을 다시 취소할 대상은 없다. protocol write 성공과 inference 완료를 한 boolean으로 표현하면 이 둘을 구분하기 어렵다.

오류 경로도 같은 상태 공간으로 검토한다. backend decode가 batch 전체 오류를 반환하면 그 batch에 참가한 모든 processing slot을 어떻게 처리하는지 본다. 한 slot의 invalid request가 launch 전에 발생하면 다른 slot의 update를 막지 않는지 본다. response serialization이 예외를 던지면 model task가 이미 완료됐는지, reader가 waiting set을 정리하는지 본다. queue worker의 예외는 `yield_to_queue()`가 unhandled task를 복원한 뒤 주 스레드에서 다시 던지는지 확인한다.

리뷰 기록에는 다음과 같은 불변식 문장을 쓴다.

- task id 하나의 inference 소유자는 어느 순간 main/deferred/unhandled/callback/slot 중 최대 하나다.
- processing slot에는 유효한 current task가 있어야 하고 idle로 돌아가기 전에 요청별 sampler 연결과 생성 상태가 정리되어야 한다.
- waiting id가 제거된 뒤 같은 id의 새 result가 무제한 보관되어서는 안 된다.
- cancel은 pending 위치와 active 위치를 모두 덮고, 완료된 task에 대해서는 안전한 no-op이어야 한다.
- release된 slot은 deferred task를 깨울 수 있지만, 그 task를 release callback 안에서 곧바로 backend 실행까지 밀어 넣지는 않는다.
- yield 중 거절된 task는 유실되거나 순서가 역전되지 않고 main queue로 복원되어야 한다.

이 불변식은 특정 테스트 이름보다 오래 간다. 코드가 리팩터링되어 deque가 다른 자료구조로 바뀌어도 한 task의 단일 소유권, cancel의 위치 완전성, 결과의 소비자 확인은 유지되어야 한다.

### 25.11.7 옵션 변경을 실험 기록처럼 문서화한다

실행하지 않고 소스를 분석하는 단계에서도 옵션 표는 가설과 검증 지점을 명확히 해야 한다. 예를 들어 `parallel`을 4에서 8로 바꾼다고 쓰지 말고 다음처럼 쓴다. 변경은 구성 필드를 통해 slot 초기화 개수를 늘린다. 예상 직접 효과는 동시에 task를 소유할 수 있는 slot 상한 증가다. 예상 간접 효과는 deferred 감소 가능성, 한 update의 active sequence 증가, KV와 host 상태 증가, sequence별 token latency 변화다. 변하지 않는 것은 HTTP worker 수와 GPU kernel 동시 실행 수를 직접 지정하지 않는다는 점이다. 검증 지표는 active/deferred, slot별 `n_ctx`, batch token 수, prompt/decode 시간, 메모리다.

prompt similarity도 같은 형식으로 쓴다. 변경 필드는 `slot_prompt_similarity`다. consumer는 `get_available_slot()`이다. 값이 0이면 prefix 후보 탐색 분기가 사실상 비활성이고 LRU 선택으로 간다. 양수이면 idle slot token과 새 task token의 longest common prefix 비율을 계산한다. 직접 효과는 선택 기준 변화다. 기대 성능 효과는 재처리 prompt 감소 가능성이다. 부작용은 후보 탐색 비용, 낮은 질의 cache 선택, 기존 context 저장과 load 비용이다. 검증은 선택 이유 로그와 공통 prefix token, cache 처리 시간으로 한다.

`cache_idle_slots`는 단독 boolean으로 문서화하면 부족하다. 초기화 consumer가 RAM prompt cache 크기를 확인하고, 없으면 비활성화한다. unified KV 여부에 따라 idle slot을 clear하여 공유 가능 공간을 만들 수 있는지 달라진다. 그러므로 “idle slot cache를 켜면 VRAM이 절약된다”는 보편 문장은 틀릴 수 있다. 저장 사본의 위치, 기존 KV의 잔류, 다음 task 시작 때 수행되는 save/load/clear를 조건별로 쓴다.

`n_predict` 또는 요청별 최대 output은 slot의 `n_predict_max`에 도달하여 종료되는 경로에 영향을 준다. 직접 효과는 한 task의 최대 점유 시간과 결과 길이 상한이다. 간접 효과는 slot turnover, deferred tail, response memory, 사용자 완결성이다. 값 `-1` 같은 무제한 의미가 허용되면 서비스 정책에서 별도 상한을 두는 이유를 설명한다. 악의적이지 않은 요청도 stop token을 늦게 내는 모델에서 자리를 오래 점유할 수 있다.

`stream`은 task의 generation 계산량 상한을 바꾸지 않지만 partial result 생성과 reader loop, response formatting을 바꾼다. 직접 효과는 사용자가 토큰을 일찍 보는 것과 result 객체 수 증가다. 간접 효과는 JSON 직렬화, proxy flush, 느린 소비자의 압력이다. 검증은 sampled-token-to-enqueue와 enqueue-to-write를 나눈다. stream을 껐을 때 throughput이 달라졌다면 모델 계산 외 response overhead도 계측한다.

`n`은 response formatting 옵션처럼 보이지만 task graph와 slot 점유를 바꾼다. parent가 child task를 만들고 seed와 id 관계가 생긴다. 직접 효과는 completion branch 수다. 간접 효과는 한 사용자 요청이 차지하는 slot 수, batch 다양성, 결과 조립 대기다. 검증은 HTTP request당 task id 수와 동시에 active인 child 수다. API gateway 비용과 rate limit이 request 수만 세면 이 부하를 놓친다.

요청별 LoRA 설정은 slot launch의 consumer까지 추적한다. adapter scale 목록을 만들고 이전 slot adapter와 비교하며 cache를 지울지 결정한다. 직접 효과는 적용 weight 경로다. 간접 효과는 prefix reuse 무효화와 launch 오류다. 검증에는 model 이름만 아니라 adapter 조합을 cache routing key와 로그에 넣는다. 동일 prompt라는 이유로 다른 adapter 요청을 같은 cache 결과로 취급해서는 안 된다.

idle sleep 설정은 queue의 `time_last_task`, sleeping callback, model/context 생명주기와 연결된다. metrics task처럼 idle timer를 reset하지 않는 task가 있을 수 있다. update 실행 시간을 idle로 잘못 세지 않도록 주 루프가 시각을 보정한다. 직접 효과는 유휴 자원 상태 전환이고, 간접 효과는 다음 요청의 wake-up 지연이다. health가 응답한다는 사실만으로 model context가 깨어 있다고 단정하지 않는다. sleep 진입, callback 완료, wake 요청, load 완료, 첫 task 처리 시각을 기록한다.

마지막으로 옵션 표에는 “상호작용” 열이 있어야 한다. parallel과 context, batch와 prompt chunk, similarity와 template 안정성, idle cache와 unified KV, stream과 proxy buffering, timeout과 cancel yield는 서로 독립이 아니다. 한 번에 여러 값을 바꾸면 어떤 consumer 분기가 효과를 냈는지 알 수 없다. 소스 분석 단계에서는 한 옵션씩 상태 전이 가설을 만들고, 나중에 실행 검증이 허용되는 환경에서 교호작용을 작은 행렬로 확인한다.

| 변경 축 | 직접 consumer | 먼저 달라질 상태 | 반드시 함께 볼 조건 | 잘못된 단축 해석 |
|---|---|---|---|---|
| parallel | slot/context 초기화 | slot 수와 admission | context, KV, batch | GPU thread 수다 |
| prompt similarity | `get_available_slot()` | 선택 이유 LCP/LRU | template, token prefix | 항상 cache hit가 오른다 |
| idle slot cache | 초기화와 launch cache 경로 | save/load/clear | RAM cache, unified KV | idle이면 KV가 사라진다 |
| output limit | launch와 token 종료 판정 | `n_predict_max`, stop | model EOS, client policy | 응답 JSON만 짧아진다 |
| stream | partial/result handler | chunk 생성과 소비 | proxy flush, client speed | 모델 decode 방식이 바뀐다 |
| `n` | child task 생성 | task·slot 점유 수 | parallel, rate limit | 응답 배열 크기뿐이다 |
| LoRA | `launch_slot_with_task()` | adapter와 cache 유효성 | activation, prefix | weight만 바뀐다 |
| idle sleep | queue loop와 callbacks | context sleeping/wake | health, cold latency | process가 멈춘다 |

이 방식으로 문서화하면 옵션 설명이 명령행 사전에서 운영 모델로 바뀐다. 독자는 값을 바꾼 뒤 무엇을 관측해야 하는지, 기대와 다른 결과가 나오면 어느 consumer 함수로 돌아가야 하는지 알 수 있다.

## 25.12 심층 추적 전에 서버와 backend의 현재 경계를 묶는다

요청 한 건의 전체 경로를 다시 한 문장씩 연결하자. HTTP route가 외부 JSON을 검증하고 template/tokenization과 task 구성을 수행한다. reader가 기다릴 task id와 프로토콜 조립 상태를 소유한다. queue `post()`가 id를 부여하고 주 루프를 깨운다. `process_single_task()`가 task 종류를 분기하며, inference가 slot을 얻지 못하면 deferred한다. `get_available_slot()`은 명시 id, token prefix 유사도, LRU 순으로 사용 가능한 자리를 찾는다. `launch_slot_with_task()`가 adapter와 sampler, limit, prompt 상태를 준비하고 task 소유권을 slot으로 옮긴다.

`update_slots()`가 여러 slot의 prompt 또는 다음 token을 batch에 넣어 llama context를 실행한다. sampling과 `process_token()`이 partial 또는 final result를 만든다. response reader가 id에 맞는 결과를 받아 HTTP 형식으로 보낸다. 정상, 취소, 오류 모두 release와 reset을 거쳐 slot을 다시 사용할 수 있게 해야 한다.

독자가 반드시 남겨야 할 통찰은 세 가지다. 첫째, slot은 GPU 스케줄러나 KV page가 아니라 재사용되는 서버 실행 상태다. 둘째, queue 대기, deferred 대기, batch 실행 대기, result 전송 대기는 서로 다른 지연이다. 셋째, 옵션의 의미는 이름이 아니라 필드와 consumer 함수, 상태 전이, 관측 효과로 설명해야 한다.

실제 코드 탐색을 시작할 때는 거꾸로 읽는 방법도 유용하다. 사용자가 받은 마지막 JSON에서 `to_json()` 구현을 찾고, 그 result를 만든 `send_partial_response()` 또는 final 생성 지점으로 올라간다. 거기서 slot의 stop 이유와 통계를 찾고, `process_token()`의 호출자를 따라 decode 결과와 `i_batch`로 이동한다. 다시 `update_slots()`의 batch 조립을 거슬러 올라가 slot이 어떤 task를 소유했는지 보고, `launch_slot_with_task()`와 `get_available_slot()`을 지나 `process_single_task()`와 queue post까지 돌아간다. 이 역방향 추적은 API 문서에서 시작해 계산 함수를 막연히 찾는 것보다 누락이 적다. 결과 필드 하나가 어느 상태에서 채워졌는지를 계속 묻기 때문이다.

반대로 지연 사고에서는 순방향 추적을 쓴다. request 도착 시각부터 각 소유권 이전을 따라가며 마지막으로 전진한 경계를 찾는다. task id가 발급되지 않았다면 parse/template/tokenize 이전이다. id는 있지만 queue callback 로그가 없다면 main queue 또는 loop responsiveness다. deferred에 있다면 admission이다. slot에 task가 있지만 prompt processed 수가 멈췄다면 batch 조립이나 backend다. decoded 수는 늘지만 partial이 없다면 stop 문자열 보류, UTF-8, response 생성이다. result가 있으나 write가 없다면 HTTP 계층이다. 이렇게 마지막 정상 상태를 찾으면 조사 범위가 한 함수군으로 줄어든다.

성능 보고서에는 평균만 쓰지 않는다. prompt 길이 구간별 queue, deferred, prefill, first-result 지연의 p50·p95·p99를 나눈다. generation에서는 active slot 수별 inter-token latency를 나눈다. cache는 공통 prefix 비율 구간별 절약된 처리 token과 cache 관리 시간을 나눈다. cancel은 task 위치별로 post-to-release를 나눈다. 같은 서버라도 짧은 채팅, 긴 문서 요약, `n` 병렬 sampling, LoRA 요청이 전혀 다른 분포를 만들기 때문이다.

정확성 보고서에는 재사용 경계를 넣는다. task가 바뀔 때 sampler와 stop state가 초기화되는지, adapter 변경 때 cache가 무효화되는지, child seed가 의도대로 갈리는지, partial tool call이 final 구조와 일치하는지, 취소된 task의 token이 다음 slot 사용자에게 섞이지 않는지를 확인한다. 처리량 개선이 이 불변식을 깨면 최적화가 아니다. 특히 prompt cache 문제는 잘못된 답이 그럴듯하게 나올 수 있어 단순 crash보다 발견이 어렵다.

코드 변경 리뷰에서는 자료형에 필드를 추가한 것만 보지 않는다. 새 요청 옵션이라면 parse, default, task move/copy, child 복제, slot launch, result echo, metrics, cancel cleanup까지 전달되는지 본다. 새 task type이라면 yielding worker에서 안전한지, slot이 필요한지, deferred될 수 있는지, response waiting id를 끝내는지 확인한다. 새 slot state라면 `update_slots()`의 모든 switch, release, metrics serialization이 그 상태를 이해하는지 본다. 수명주기 코드는 한 분기 누락이 고아 task나 영구 점유로 나타난다.

마지막으로 재현 기록에는 고정 커밋과 구성뿐 아니라 요청 형태를 남긴다. endpoint, stream, prompt token 수, output limit, `n`, slot 지정, adapter, cache 관련 flag, parallel, context, batch를 기록한다. 로그 일부만으로는 선택 분기를 재구성할 수 없다. 같은 바이너리라도 template과 tokenizer 결과가 달라지면 prefix 선택이 바뀌므로 적용 template과 실제 token 수를 남긴다. 이 기록이 있어야 다음 릴리스에서 구조가 달라졌을 때 같은 증상을 비교할 수 있다.

한 요청을 설명하는 문장마다 현재 owner와 다음 owner를 적어 보는 습관도 좋다. “서버가 처리한다” 대신 “deferred deque가 task를 보유하고 release callback이 main queue로 옮긴다”라고 쓴다. “GPU가 응답한다” 대신 “backend 결과를 slot이 sampling하고 response queue가 reader를 깨운다”라고 쓴다. 이런 문장 교정만으로도 층을 건너뛴 인과 설명과 근거 없는 성능 주장을 상당수 제거할 수 있다.

여기까지는 서버 조정 계층이다. 아직 `llama_decode()` 아래에서 graph가 어떻게 만들어지고 tensor가 backend buffer에 놓이며 CUDA kernel과 CPU backend가 어떻게 선택되는지는 열지 않았다. 26장에서는 이 정확한 경계를 넘는다. `server_batch`가 llama API의 batch로 변환되는 지점, sequence id와 position이 attention graph에 반영되는 지점, backend scheduler가 graph node를 장치별로 배치하는 지점을 추적한다. 그때도 같은 원칙을 쓴다. slot id를 CUDA stream으로 번역하지 않고, 논리 sequence와 물리 memory를 갈라 놓으며, 서버에서 본 prompt/decode 차이가 graph와 kernel 수준에서 왜 다른 비용으로 나타나는지를 연결한다.

**고정 소스 메모**

- 기준 커밋: `bb4caa7540188872173c44d161602d9271386413`.
- task와 result 자료형: [`tools/server/server-task.h`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-task.h#L137-L430).
- queue, response, reader 계약: [`tools/server/server-queue.h`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-queue.h#L15-L220).
- queue 구현과 yield: [`tools/server/server-queue.cpp`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-queue.cpp#L20-L366).
- slot과 실행 수명주기: [`tools/server/server-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L196-L3956).
- completion handler와 result loop: [`tools/server/server-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L4149-L4310).

이 소스 메모는 줄 번호를 결론 대신 쓰기 위한 목록이 아니다. 릴리스가 바뀌면 각 링크에서 같은 소유권 질문을 다시 검증하기 위한 출발점이다. route가 task를 몇 개 만드는가, queue가 어느 상태에서 defer하는가, slot이 무엇을 reset하고 무엇을 보존하는가, update가 계산 중 queue에 언제 양보하는가, reader가 어떤 id를 언제 취소하는가를 다시 확인하면 구조 변화도 빠르게 찾을 수 있다.

## 25.13 두 slot의 HTTP task를 `llama_decode()`와 CUDA graph 경계까지 내려보낸다

slot을 CUDA stream으로 생각하면 첫 단추부터 틀린다. slot은 HTTP task를 받아 prompt progress, sampler, stop state, output 통계와 sequence identity를 보존하는 서버 객체다. 여러 active slot의 token은 하나의 shared batch에 섞여 `llama_decode()` 한 번으로 내려갈 수 있다. llama context와 backend scheduler가 tensor graph를 실행하며 CUDA backend가 배치된 node를 kernel로 내린다. server slot 수와 CUDA stream 수는 같은 축이 아니다.

고정 소스의 `server_context`는 queue callback에서 `process_single_task`와 `update_slots`를 호출한다. inference task가 들어오면 `get_available_slot`이 사용 가능한 slot을 고르고 `launch_slot_with_task`가 task state를 slot에 결합한다.

`update_slots`는 active slot들을 순회해 prompt token 또는 이전 sample의 다음 token을 shared batch에 넣고 `llama_decode`를 호출한다. [llama.cpp task 분기와 slot launch](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L1500-L1708), [llama.cpp update와 decode 호출](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-context.cpp#L2723-L3620)

### slot 0과 slot 1의 세 decode 원장

server batch capacity가 6 token, slot 0의 prompt A가 5 token, slot 1의 prompt B가 3 token이라고 하자. 첫 update에서 fairness와 chunking policy가 A 네 token, B 두 token을 batch에 넣는다고 가정한다. 각 row에는 token ID뿐 아니라 logical position과 sequence ID, logits output flag가 필요하다. flat batch row 0~3은 A sequence, row 4~5는 B sequence다.

`llama_decode()`는 이 six-token batch를 model graph로 실행한다. A와 B는 아직 prompt가 남았으므로 sampling하지 않고 progress를 각각 4와 2로 갱신한다. 다음 update는 A 마지막 prompt token과 B 마지막 prompt token을 넣는다. 두 row에서 다음-token logits가 필요한지 output flag를 설정한다. decode 뒤 slot별 logits row index를 정확히 연결해 sampler가 A distribution을 B에 쓰지 않게 한다.

두 slot이 각각 token a1, b1을 sample했다면 세 번째 update batch에는 두 token이 들어간다. logical positions는 A 5, B 3이다. sequence ID는 slot object의 재사용 가능한 index와 같아 보일 수 있지만 KV sequence ownership 계약을 source로 확인한다. slot이 release/reuse될 때 이전 KV sequence가 정리되거나 의도한 prompt cache만 남아야 한다.

이 원장의 열은 `(update_generation, batch_row, task_id, slot_id, sequence_id, token_id, logical_pos, prompt_or_decode, logits_requested, output_row)`다. batch row와 output row가 항상 같다고 가정하지 않는다. 일부 row만 logits를 요청하면 compact output index가 별도로 생길 수 있다. `i_batch` 같은 mapping을 sampling까지 추적한다.

### batch size가 graph shape와 kernel을 바꾸는 이유

prefill six-token batch와 decode two-token batch는 같은 model weight를 쓰지만 graph tensor shape와 attention work가 다르다. prefill은 각 sequence의 여러 query와 causal 관계를 처리하고 KV를 여러 칸 쓴다. decode는 sequence당 새 query 하나가 긴 KV context를 읽는다. backend가 선택하는 GEMM/GEMV, attention kernel과 launch 구성이 달라질 수 있다.

prompt cache를 켜서 A 공통 prefix 네 token을 재사용하면 첫 update의 A unseen suffix는 한 token으로 줄 수 있다. shared batch 구성은 A 1+B 3=4 token이 되고 prefill graph shape도 달라진다. README는 backend와 batch size에 따라 cache on/off logits가 bit-for-bit 동일하지 않을 수 있다고 경고한다. cache가 model semantics를 의도적으로 바꾸는 것은 아니지만 numerical execution shape가 달라질 수 있다. [llama.cpp cache_prompt 계약](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/README.md#L571-L590)

성능 원장에는 batch token, busy slot, prompt/decode composition, context lengths, `llama_decode` 횟수와 backend graph/kernel 구간을 둔다. server README가 `n_decode_total`과 busy slots per decode metric을 제공하는 것도 이 이유다. slot utilization 하나로 CUDA utilization을 추측하지 않는다.

### `llama_decode()` 아래의 owner를 구분한다

server context는 llama API batch를 구성하고 호출 결과를 slot에 되돌리는 owner다. llama context는 model, KV context와 graph build inputs를 소유한다. ggml backend scheduler는 graph node와 backend buffers, compute scheduling을 소유한다. CUDA backend는 배정된 operation을 CUDA kernel/graph capture 실행으로 번역한다. 어느 층의 “graph”인지 이름을 붙인다.

HTTP stream이 느린 것은 CUDA graph 문제일 수 있지만 output queue나 stop buffer 문제일 가능성도 있다. `llama_decode` return 전이면 model/backend, return 뒤 sampling까지면 sampler, partial result 생성 뒤면 queue/HTTP로 checkpoint를 나눈다. CUDA profiler만 열기 전에 server timeline을 만든다.

fixed commit의 `llama_decode` entry는 context implementation에 batch를 넘기는 C API 경계다. 실제 graph build와 backend schedule은 context source의 caller 아래로 이어진다. [llama.cpp llama_decode API 경계](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-context.cpp#L4120-L4155)

### logits row mapping incident

slot A가 prompt 마지막 row에서 logits를 요구하고 slot B는 prompt 중간이라 요구하지 않았다고 하자. compact logits output에는 A 한 row만 있다. 다음 update에서 B도 prompt를 끝내 두 row가 필요하다. server가 이전 `i_batch` mapping을 재사용하면 B sampler가 A의 이전 logits row를 읽을 수 있다. token은 정상 범위이고 crash가 없다.

first divergence는 llama graph output 값 자체가 아니다. output row 0의 score는 A reference와 맞지만 consumer slot이 B다. trace에는 batch generation, row token/seq/pos, logits flag, output index, sampler slot/task를 둔다. slot ID만 기록하면 reuse generation을 구분하지 못한다.

수정은 batch build 때 output mapping을 generation별로 만들고 decode completion까지 snapshot을 보존하는 것이다. update loop가 yielding 동안 queue task를 처리해 slot state가 바뀔 수 있다면 mapping lifetime과 mutation barrier를 본다. async backend라면 `llama_decode` return이 synchronization을 의미하는지도 실제 API contract를 확인한다.

회귀 fixture는 A만 logits, B만 logits, 둘 다, slot 중간 cancel을 조합한다. synthetic distinguishable logits 접근이 가능하면 각 slot winner를 다르게 만든다. 실행 권한이 없으면 source probe와 expected mapping table을 준비하고 실행했다고 주장하지 않는다.

### HTTP task부터 backend까지 timestamp를 잇는다

timestamp는 HTTP parsed, task posted, queue selected, slot launched, batch row added, decode begin/end, sample, partial/final enqueued, socket write로 나눈다. deferred queue 시간과 busy slot wait를 구분한다. decode begin/end 안을 graph build, backend schedule, CUDA kernel로 확장할 때 동일 batch generation을 trace correlation key로 쓴다.

GPU event 시간과 host wall time을 섞지 않는다. llama decode가 async work를 submit한 뒤 내부 sync가 다른 지점에 있을 수 있다. 고정 source 주석도 decode timing이 sync 뒤에만 유효한 경로가 있음을 암시한다. host function duration만 kernel total로 부르지 않는다.

metric cardinality를 제한한다. slot ID는 작은 bounded label일 수 있지만 task ID는 trace에 둔다. prompt length, batch token, active slot은 bucket으로 둔다. CUDA kernel 이름은 profiler artifact에 두고 Prometheus label로 무한히 늘리지 않는다.

## 25.14 context shift·prompt cache·abort를 slot generation과 함께 닫는다

context window가 가득 차면 서버는 요청을 실패시키거나 context shift를 수행할 수 있다. shift는 단순히 오래된 text를 버리는 것이 아니다. 유지할 prefix와 discard할 중간 token, KV position을 정하고 remaining sequence가 새 좌표에서 계속 attention할 수 있게 cache를 이동·재배치한다. sampler와 stop history가 어느 범위를 유지하는지도 별도다.

### n_ctx=16의 context shift 손계산

slot A가 prompt 10 token과 generated 6 token으로 context 16을 채웠다고 하자. 다음 token을 위해 한 칸이 필요하다. server가 system prefix `n_keep=4`를 유지하고 중간 6 token을 discard, 최근 6 token을 남긴다고 가정하자. logical conversation total은 16이지만 resident KV는 keep 4+recent 6=10으로 줄고 다음 write position은 backend shift contract에 따라 10 부근이 된다.

사용자-visible generated length와 stop/penalty history는 자동으로 10으로 줄지 않는다. repetition penalty가 전체 generated history를 보존할지 resident context만 볼지는 sampler state 계약이다. context shift counter, discarded token range, KV resident position과 full output token history를 분리한다.

RoPE absolute position을 cache physical index로 단순 재설정하면 model 의미가 달라질 수 있다. llama.cpp의 KV sequence shift operation이 position delta를 어떻게 적용하고 model이 shift를 지원하는지 source에서 확인한다. 이 책의 손계산은 owner와 좌표 질문을 만들며 모든 architecture에서 동일 결과를 주장하지 않는다.

### prompt cache hit를 prefix 길이와 artifact identity로 판정한다

이전 slot prompt가 `[S0,S1,U0,U1,U2]`, 새 prompt가 `[S0,S1,U0,V1,V2]`이면 common prefix는 세 token이다. `cache_prompt=true`에서 unseen suffix 두 token만 eval할 수 있다. 하지만 tokenizer/template/model/adapter와 relevant sampling/cache representation이 호환돼야 한다. token 문자열이 같아도 adapter가 바뀌면 KV는 재사용할 수 없다.

slot 선택이 prefix 유사도를 사용하면 cache hit optimization과 admission policy가 결합한다. free slot 0은 prefix 3, slot 1은 prefix 0이지만 LRU가 더 오래됐다고 하자. selector가 prefix를 우선하는지 명시 slot/LRU와 어떤 순서인지 source를 읽는다. latency 이득과 fairness를 분리한다.

cache hit metric은 requested prompt tokens, reused prefix tokens, evaluated suffix tokens과 validation identity를 둔다. hit boolean만으로 1 token 재사용과 10k token 재사용을 같게 보지 않는다. cache lookup/sequence manipulation 비용도 측정한다.

#### adapter가 바뀌었는데 prompt cache가 재사용된 사건

slot 0이 LoRA A로 prompt P를 처리하고 release됐다. 다음 task는 같은 P지만 LoRA B를 요청했다. launch가 adapter를 바꿨지만 prompt cache key/invalidator가 token prefix만 보아 old KV를 유지했다. 첫 suffix/decode logits부터 full recompute reference와 달랐고 답은 문법적으로 정상이라 오래 발견되지 않았다.

first divergence는 HTTP parse나 tokenization이 아니다. tokens와 slot selection은 같고, cache reuse decision에서 artifact/adapter identity가 누락됐다. trace에는 model digest, adapter set/digest, template/token digest, cached prefix length와 cache generation을 둔다. cache hit rate 상승과 품질 회귀가 동시에 나타날 수 있다.

수정은 compatibility key에 model context와 adapter state, KV-affecting option을 포함하거나 adapter 변경 때 slot KV를 clear한다. 무엇이 KV를 바꾸는지 source와 model contract를 근거로 한다. sampling-only temperature는 KV identity가 아닐 수 있지만 prompt processor가 token을 바꾸면 token digest에서 달라진다.

회귀는 same prompt/same adapter hit, same prompt/different adapter miss, one-token suffix, template/tokenizer revision mismatch를 둔다. hit/miss뿐 아니라 first logits와 evaluated token count를 reference와 비교한다. cache disabled full recompute를 semantic oracle로 둔다.

### abort 위치별 상태 전이

task가 아직 main queue에 있으면 queue cancellation으로 reader를 깨우고 slot state는 없다. deferred라면 deferred owner에서 제거한다. slot에 launch됐지만 batch에 row가 없으면 slot release와 sampler/stop state reset, KV policy를 적용한다. decode batch에 들어가 in-flight면 result discard와 resource fencing이 필요하다.

HTTP disconnect는 task ID cancel을 queue/server context까지 전달해야 한다. response reader만 끝내면 slot은 generation을 계속해 GPU와 KV를 소비한다. 반대로 slot을 release했지만 reader terminal을 보내지 않으면 handler가 기다린다. cancel semantic terminal과 backend completion/resource release를 구분한다.

slot reuse에는 generation을 붙인다. slot ID 0의 task A가 cancel되고 task B가 들어올 때 late A partial/final과 decode output이 B로 attach되지 않게 한다. `(slot_id,slot_generation,task_id)` tuple을 result와 batch mapping에 둔다. 외부 task ID가 재사용될 수 있다면 request incarnation도 필요하다.

#### abort late-output incident

A slot 0 generation 17이 two-token decode batch에 들어간 직후 client가 disconnect했다. server가 slot을 IDLE로 바꾸고 B를 generation 18로 즉시 launch했다. decode return 뒤 update는 current slot.task를 참조해 A logits를 B sampler에 넘겼다. B 첫 token이 틀리고 A reader는 이미 없어 exception이 없었다.

first divergence는 batch snapshot의 generation 17과 consumer current generation 18이 다르다는 점이다. 수정은 in-flight batch가 보유한 slot generation을 completion에서 검증하고 mismatch output을 discard하며, backend가 slot/KV memory를 쓸 수 있는 동안 reuse하지 않는 것이다. slot object 재사용과 KV sequence reuse를 함께 fence한다.

회귀 fixture는 decode completion을 지연하고 A cancel/B launch를 사이에 넣는다. B first logits가 clean reference와 같고 A output이 어느 queue에도 전달되지 않으며 slot generation mismatch counter가 예상대로 증가하는지 본다. cancel storm에서 slot이 영구 busy가 되거나 deferred가 굶지 않는지 soak한다.

#### context shift와 abort가 겹치는 사건

A가 context shift용 KV manipulation을 예약한 뒤 cancel됐다. slot을 B가 재사용했고 늦은 shift operation이 같은 sequence ID에 적용되면 B cache positions가 이동한다. model compute가 없어도 cache control operation이 in-flight일 수 있다. backend completion fence는 decode kernel뿐 아니라 KV sequence mutation도 포함해야 한다.

trace에는 cache op type, seq ID, slot generation, logical range, submit/complete event를 둔다. allocator/free 상태만으로 control operation race를 잡지 못한다. B의 input position과 KV cell position digest를 clean launch와 비교한다.

### ggml CUDA graph와 server slot generation의 경계

CUDA graph capture가 static buffers와 operation topology를 재사용하더라도 논리 task/slot generation을 캡처해서는 안 된다. replay 전에 current batch data, positions, sequence mapping과 output mapping을 올바른 static buffer에 채운다. graph executable lifetime은 여러 task에 걸치고 batch descriptor lifetime은 한 update다.

graph fallback이 shape나 unsupported operation 때문에 eager로 바뀌면 slot/cache semantics는 같아야 한다. cache shift나 prompt-cache sequence operation이 graph 밖 control path라면 happens-before를 명시한다. graph replay가 old mapping을 읽는 incident는 server trace와 CUDA trace correlation이 필요하다.

### 최종 배포 dossier와 terminal

첫 표는 HTTP request→task→queue/deferred→slot generation→batch row→sequence/position→decode output→result queue→HTTP write다. 둘째는 two-slot three-decode 수치 원장이다. 셋째는 context shift의 full history/resident KV/physical position이다. 넷째는 prompt cache compatibility key와 reused/evaluated tokens다. 다섯째는 abort 위치별 cleanup과 fence다.

correctness terminal은 slot/batch/output mapping과 clean reference logits가 맞다. cache terminal은 prompt hit가 compatible artifact에서만 발생하고 shift 뒤 logical/model position 계약이 맞다. lifecycle terminal은 queue/deferred/launched/in-flight abort에서 reader terminal과 slot/KV reclaim이 정확히 한 번이다. backend terminal은 graph/eager와 batch shape가 바뀌어도 같은 logical sequence를 처리한다.

performance terminal은 queue/deferred/prefill/decode/sample/route 시간을 나누고 cache reused tokens와 shift cost, busy slots/decode, graph selected/fallback을 설명한다. observability terminal은 task ID를 고-cardinality metric에 넣지 않고도 sampled trace에서 slot generation과 batch mapping을 복원한다.

rollback은 cache_prompt, context shift, graph fast path를 독립적으로 끌 수 있어야 한다. in-flight task는 generation fence와 drain 뒤 전환한다. cache를 끄면 full recompute reference semantics가 복원되고, graph를 끄면 eager path가 동일 IDs/positions/output mapping을 소비해야 한다.

25장의 최종 invariant는 다음과 같다. **HTTP task는 하나의 slot generation에 결합되고, update마다 만들어진 batch snapshot이 token·position·sequence·logits row를 llama/ggml/backend completion까지 보존하며, prompt cache와 context shift는 compatible cache identity와 logical position을 유지하고 abort된 generation의 late work는 다음 slot 사용자에게 절대 commit되지 않아야 한다.**

이 문장을 two-slot batch 계산, logits mapping, adapter cache incident, abort late-output과 shift race로 설명할 수 있으면 llama.cpp 서버와 CUDA backend 사이를 추측 없이 걸을 수 있다.

**slot launch에서 reset하는 값과 보존하는 값을 나눈다**

새 task를 slot에 결합할 때 모든 필드를 0으로 만드는 것도, 이전 상태를 그대로 두는 것도 정답이 아니다. task ID, sampling params, generated token, stop matcher, partial UTF-8, timing과 error state는 새 generation에 맞춰 reset한다. prompt cache를 재사용한다면 compatible prefix tokens와 KV sequence 일부는 의도적으로 보존한다. slot statistics 중 lifetime cumulative와 request-local counters도 구분한다.

source review 표에는 `field, old generation, launch assignment, reset/preserve condition, first consumer, release action`을 둔다. sampler object가 이전 repetition history를 들고 있다면 같은 prompt cache를 쓰더라도 새 request output history와 섞일 수 있다. stop string matcher의 pending prefix가 남으면 새 response 첫 bytes가 보류되거나 잘린다. seed/generator가 child task마다 의도대로 분리되는지도 본다.

LoRA와 multimodal projection 같은 model-affecting state는 prompt KV compatibility를 바꿀 수 있다. task launch가 adapter를 적용하는 순서와 cache reuse decision을 연결한다. 먼저 prefix hit를 계산하고 나중에 adapter를 바꾸면 old identity로 잘못 hit할 수 있다. compatible key를 launch 준비 초기에 확정하거나 adapter change가 cache를 invalidate해야 한다.

**prompt suffix 절약량을 실제 token으로 계산한다**

이전 prompt 1,024 token과 새 prompt 1,100 token이 앞 900 token을 공유한다고 하자. cache off는 1,100 token을 eval한다. cache hit는 unseen suffix 200 token을 eval하므로 saved work token은 900이다. “이전 prompt 길이 1,024”를 모두 재사용하는 것이 아니다. divergence index가 경계다.

batch capacity가 256이면 cache off는 최소 다섯 prompt chunks, suffix는 한 chunk일 수 있다. `llama_decode` 호출 수가 5→1로 줄고 TTFT가 개선될 가능성이 있다. 실제 chunking은 다른 slot과 batch 공유, n_batch/n_ubatch와 backend에 달렸으므로 최소 호출 수를 측정과 구분한다.

KV byte도 계산한다. 앞 장의 128 KiB/token 예시를 그대로 적용하면 900-token cached prefix는 약 112.5 MiB의 logical KV를 재사용한다. slot이 cache를 유지하는 동안 이 capacity는 다른 prompt에 쓸 수 없다. saved compute와 resident capacity tradeoff를 함께 적는다.

prefix lookup 자체도 비용이 있다. token vector 비교와 KV sequence manipulation, slot selection이 TTFT에 들어간다. 짧은 8-token prompt에서 3 token hit를 위해 복잡한 cache operation을 하면 이득이 작을 수 있다. prompt length/hit length bucket별로 eval saved time과 management time을 본다.

**context shift의 position ledger를 더 엄밀히 쓴다**

full conversation token index를 `g`, resident KV index를 `r`, model position을 `p`라 하자. shift 전에는 우연히 `g=r=p`일 수 있다. keep prefix와 recent tail을 재배치하면 r은 compact되지만 p는 backend operation이 정한 delta를 반영한다. output history index g는 사용자 conversation 기록을 위해 계속 증가할 수 있다.

n_ctx 16 예에서 global indices 0~3 keep, 10~15 recent를 남긴다면 resident rows는 `[0,1,2,3,10,11,12,13,14,15]` 열 개다. cache operation이 recent positions에서 6을 빼면 model positions는 `[0,1,2,3,4,5,6,7,8,9]`가 된다. 다음 token은 p=10에 쓴다. 어떤 model/strategy가 이 shift를 허용하는지 확인한다.

stop matcher와 response text는 discarded context token을 사용자 출력에서 삭제하지 않는다. sampler repetition history가 full generated tokens를 보존한다면 g를 읽고, resident-only policy면 별도 window를 읽는다. 한 `n_past` 필드로 세 좌표를 모두 설명하지 않는다.

shift metric에는 shifts count, discarded token, kept prefix, resident before/after, cache operation time, next logical/model position을 둔다. shift가 잦으면 decode latency spike와 quality 변화가 생길 수 있다. 평균 inter-token latency는 spike를 숨기므로 shift cohort p99를 본다.

**prompt cache와 context shift가 만나는 compatibility 문제**

shift된 slot의 cached token vector는 full original prompt와 동일하지 않을 수 있다. 새 task prefix matching이 visible token history만 보고 hit를 선언하면 resident KV layout/position과 맞지 않을 수 있다. cached token sequence, resident ranges, position transform generation을 key에 포함하거나 shifted cache를 reusable prefix 후보에서 제외할 수 있다.

fixture는 unshifted exact prefix, shifted keep-only prefix, shifted recent overlap, adapter change 네 경우다. expected reusable token count와 position mapping을 적는다. full recompute first logits를 oracle로 사용한다. cache hit인데 evaluated suffix가 0이라는 사실만으로 correctness를 증명하지 않는다.

사건을 가정하자. slot A가 shift generation 3을 거친 뒤 IDLE이 됐다. B prompt가 A의 visible first 10 token과 같아 prefix hit 10으로 계산됐다. 그러나 resident KV는 keep 4+recent 6이고 positions가 compact된 상태였다. B suffix logits가 틀렸지만 token digest hit는 맞았다. first divergence는 cache compatibility가 resident layout generation을 무시한 decision이다.

수정 뒤 cache key/digest뿐 아니라 resident range/position transform을 검증하고 unsupported 상태는 miss로 보낸다. hit rate는 낮아질 수 있지만 full recompute semantics가 우선이다. drain 없이 policy를 바꿀 때 기존 shifted slot cache를 invalidate한다.

**queue yielding과 slot mutation의 happens-before**

server main loop는 task queue를 처리하고 `update_slots`가 오래 걸릴 때 responsiveness를 유지하려 yield할 수 있다. yield가 있다는 사실은 thread가 동시에 slot을 mutate한다는 뜻과 같지 않지만, callback/re-entry와 deferred task가 어느 지점에서 처리되는지 확인해야 한다. batch snapshot을 만든 뒤 cancel task가 처리되면 current slot status와 snapshot generation이 다를 수 있다.

세 checkpoint를 둔다. batch build sealed, decode completion, slot update commit이다. cancel이 sealed 전이면 row에서 제외할 수 있다. sealed 뒤 completion 전이면 output discard flag와 deferred release를 설정한다. completion 뒤 commit 전이면 generation check로 discard한다. commit 뒤면 이미 emitted token과 cancel ordering 계약을 따른다.

task queue의 `update_slots` callback과 yield 주석은 responsiveness 시간이 idle accounting에 섞이지 않게 하는 코드도 가진다. queue wait, update compute, yield handling을 별도 latency로 본다. [llama.cpp queue update callback](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/tools/server/server-queue.cpp#L300-L326)

**ggml CUDA graph까지 내려갈 때 확인할 실제 데이터**

server batch의 token, position, sequence ID와 logits flag는 llama context graph input으로 변환된다. graph node는 embedding, norm, matmul, RoPE, attention, MLP와 output head를 나타내며 backend allocation/scheduler가 supported backend에 배치한다. CUDA graph capture/replay가 사용되면 device pointer와 topology 안정 조건을 가진다.

독자는 server `slot_id`가 graph tensor dimension으로 직접 들어간다고 가정하지 않는다. sequence ID와 batch metadata가 attention mask/KV view를 만드는 의미 경계다. graph backend split과 device selection은 model tensor placement와 operation support를 따른다. HTTP task priority가 CUDA stream priority로 자동 번역되지 않는다.

source trace는 `update_slots` batch add, `llama_decode`, context graph build, ggml backend graph compute, CUDA backend op implementation으로 이어간다. chapter 26이 backend 세부를 확대하더라도 25장 incident에는 batch generation correlation을 남겨 위아래 trace를 연결한다.

graph replay failure나 unsupported shape에서 eager/backend fallback이 일어날 수 있다. selected path를 runtime log/trace로 확인한다. config에 CUDA가 켜졌다는 사실만으로 모든 node가 CUDA kernel이라는 결론을 내리지 않는다. CPU fallback과 transfer가 critical path가 될 수 있다.

**두 node·네 GPU가 아니라 한 서버의 NUMA 경계도 기록한다**

llama.cpp가 CPU tokenization/sampling과 GPU offload를 함께 쓸 때 HTTP thread, queue loop, model context, GPU device 사이 host memory와 synchronization이 있다. prompt token buffer가 어느 NUMA node에 있고 pinned transfer를 쓰는지에 따라 long prompt TTFT가 달라질 수 있다. 이 장에서는 경로를 기록하고 구체 최적화는 hardware 장으로 넘긴다.

multi-GPU tensor split이 있으면 하나의 `llama_decode`가 여러 device/backend 작업을 만든다. slot parallel과 tensor split을 혼동하지 않는다. active slots 두 개가 GPU 두 개에 하나씩 고정된다고 가정하지 않는다. model graph partition과 batch sequence는 독립 축이다.

관측에는 model offload layers, tensor split/device identity, batch token, graph backend node count/transfer와 server slot mapping을 함께 둔다. 하나의 평균 GPU utilization로 slot fairness나 prompt cache 효과를 설명하지 않는다.

**abort failure drill을 위치별로 실행한다**

첫 drill은 queue before slot이다. task를 post한 직후 cancel하고 reader가 terminal을 받으며 active slot/KV가 변하지 않는지 본다. 둘째는 deferred다. 모든 slot을 busy로 만들고 cancel해 deferred length가 줄고 새 task가 starvation 없이 들어오는지 본다.

셋째는 launched before batch다. sampler/stop/adapter state가 reset되고 cache policy가 의도대로 clear/preserve되는지 본다. 넷째는 batch sealed/in-flight다. decode delay를 넣고 generation mismatch output discard와 resource fence를 본다. 다섯째는 partial emitted 뒤다. delivered prefix, final reason, reader close와 slot release가 한 번인지 본다.

여섯째는 context shift op in-flight, 일곱째는 prompt cache sequence manipulation 중 cancel이다. control operation completion까지 sequence ID를 재사용하지 않는다. 여덟째는 HTTP socket close만 하고 cancel bridge를 끊어 orphan detection metric이 경보하는지 본다.

각 drill은 task/slot/sequence/batch/cache generation과 queue/reader state를 기록한다. 최종 string만 검사하지 않는다. unrelated slot B의 inter-token latency와 output이 유지되는지도 본다. global main loop block이나 잘못된 shared context cleanup을 잡는다.

**메모리·시간 conservation 원장**

한 request의 token은 received, tokenized, reused prefix, evaluated prompt, sampled, accepted/committed, routed, delivered로 나눈다. `prompt_tokens = reused + evaluated`가 compatibility miss/shift 예외 정의 안에서 성립하는지 본다. generated는 delivered보다 클 수 있지만 terminal 뒤 orphan gap은 수렴해야 한다.

시간은 parse/template/tokenize, queue, deferred, slot launch/cache prep, prefill decode calls, first sample/route, generation decode/sample, final route/write로 나눈다. context shift와 cache operation을 별도 event로 둔다. backend graph/kernel은 decode 내부 subspan이다.

memory는 model/backend buffers, KV context, prompt/token histories, sampler/grammar, shared batch, output queue로 나눈다. slot count를 늘리면 KV capacity와 per-slot state가 증가하지만 model weight가 slot마다 복제되는 것은 아닐 수 있다. 실제 context architecture를 확인한다.

**친절한 option 카드 다섯 개**

`--parallel`은 available slot 수와 shared batch의 동시 sequence 기회를 바꾼다. request state와 sampler, output queue가 slot별로 늘고 KV context capacity가 분할/공유되는 방식을 확인한다. 효과는 concurrency와 per-request context/capacity tradeoff다. 반증은 busy slots/decode, queue wait, per-slot context limit이다.

`--ctx-size`는 context capacity와 KV memory, shift 시작점을 바꾼다. 값만 키우면 긴 prompt를 받지만 token당 KV byte에 비례해 capacity가 늘고 slot parallel과 결합된다. 반증은 actual context size, KV allocation byte, shift count와 OOM이다.

`--batch-size`와 micro batch 관련 값은 한 decode call에 넣는 prompt token과 graph shape, temporary를 바꾼다. output quality를 직접 조절하는 값은 아니지만 numerical kernel path 차이가 있을 수 있다. 반증은 batch token, decode calls, selected backend/kernel과 first logits differential이다.

`cache_prompt`는 compatible prefix evaluation을 줄이고 slot cache retention/selection을 사용한다. 효과는 TTFT와 resident KV, numerical batch-shape 차이다. 반증은 token digest·artifact identity, reused/evaluated token, cache generation과 full recompute logits다.

context shift option은 capacity 초과 때 fail 대신 KV/history coordinate를 변환한다. 효과는 더 긴 visible generation과 shift latency/quality risk다. 반증은 discard/keep ranges, resident/model position, next logits와 stop/sampler history다. 단순 “무한 context”라고 설명하지 않는다.

### 최종 release gate

source gate는 pinned commit의 route/task/queue/slot/update/decode/context/backend/output/cancel edge를 닫는다. numeric gate는 two-slot batch row와 output mapping, prompt suffix, context shift positions를 손으로 재현한다. differential gate는 cache off full recompute, eager fallback과 clean slot launch를 oracle로 둔다.

lifecycle gate는 여덟 abort 위치와 slot generation late work를 통과한다. performance gate는 target prompt/concurrency에서 queue·prefill·decode·cache/shift·route와 backend subspan이 예산 안이다. observability gate는 full prompt나 high-cardinality task ID를 metric에 노출하지 않고 sampled trace로 first divergence를 복원한다.

rollout은 cache_prompt, context shift, CUDA graph/backend option을 cohort별로 분리한다. 한 번에 모두 켜지 않는다. regression이면 in-flight generation을 drain/fence하고 새 task부터 검증된 path를 선택한다. 기존 slot cache의 compatibility generation을 invalidate할 필요가 있는지 확인한다.

최종적으로 독자는 “llama.cpp가 GPU에서 생성한다”가 아니라 구체적으로 말할 수 있다. HTTP task A가 slot 0 generation 17을 얻고, update 42의 batch rows와 sequence/position으로 `llama_decode`에 들어가며, backend graph completion 뒤 snapshot mapping으로 logits를 A sampler와 response queue에 돌려준다. prompt cache와 shift는 cache identity/position ledger를 통과하고 cancel은 generation 17의 late work를 discard·fence한다.

이 문장이 source anchor와 수치 원장, incident trace로 성립하면 25장이 닫힌다. 다음 장은 같은 batch generation을 받아 ggml graph allocation, backend scheduler와 CUDA kernel의 실제 node/stream/memory 경계로 더 내려간다.

**마지막 incident rehearsal**

운영자는 일부러 slot 0의 generation 17 decode를 지연하고 그 사이 HTTP disconnect, context shift 예약, slot generation 18의 cache-compatible task launch를 순서대로 넣는다. 올바른 구현은 generation 17의 sampled output과 shift operation을 discard 또는 completion fence하고, generation 18은 clean cache decision과 position ledger에서 시작한다. task reader는 cancel terminal을 정확히 한 번 받고 새 reader는 이전 partial bytes를 보지 않는다.

rehearsal record의 첫 행에는 외부 request와 task ID, slot generation을 둔다. 둘째에는 batch generation과 token/position/sequence/logits mapping을 둔다. 셋째에는 prompt cache key, reused extent와 resident shift generation을 둔다. 넷째에는 decode/shift submit과 completion event를 둔다. 다섯째에는 sample, routed/delivered cursor, terminal과 slot/KV release를 둔다.

첫 divergence가 cache decision이면 adapter/template/token/shift identity를 본다. batch row부터 다르면 update snapshot을 본다. backend output은 맞고 sampler owner가 다르면 logits mapping을 본다. selected token까지 맞고 stream만 다르면 response queue와 reader를 본다. release 뒤에만 cache가 변하면 late control operation과 generation fence를 본다.

수정 뒤 동일 task를 cache off full recompute, cache on unshifted, shifted miss, eager backend와 graph path에서 비교한다. exact bit parity가 제품 계약이 아니라면 top candidate와 bounded logit tolerance, committed token semantics를 구분한다. cache on/off batch shape가 numerical path를 바꿀 수 있다는 고정 문서의 한계도 결과에 남긴다.

성능 판정은 correctness fixture와 별도다. reused 900 token이 TTFT에서 얼마나 saved work를 만들었는지, cache resident 112.5 MiB와 management time은 얼마인지, abort fence가 slot reclaim을 얼마나 늦췄는지, graph path가 실제 선택됐는지를 측정한다. 하나의 throughput 숫자로 cache와 graph 이득을 합치지 않는다.

최종 rollback rehearsal은 새 task admission을 막고 in-flight generation과 cache control operation을 drain한 뒤 cache/shift/graph option을 이전 값으로 돌린다. active reader에 terminal이 중복되지 않고 slot generation이 연속되며 새 task가 clean reference path를 타는지 본다. rollback 버튼 자체보다 이 상태 migration이 중요하다.

독자가 이 rehearsal을 설명할 수 있다면 “slot이 꼬였다”는 증상을 task, slot generation, batch snapshot, KV sequence, backend completion, response route의 정확한 edge로 바꿀 수 있다. 이것이 25장이 제공해야 할 실전 디깅 능력이다.

release note에는 검증한 endpoint, stream 여부, parallel/context/batch, cache와 shift, adapter, offload/backend 조합을 적는다. completion 한 종류가 통과했다고 embeddings, rerank, multimodal task의 slot path까지 같다고 주장하지 않는다. 새 task type은 slot 필요 여부, batch row와 output mapping, cancel/reader terminal을 별도 fixture로 갖는다.

관측이 부족한 배포에서는 추측으로 kernel 병목이나 cache corruption을 단정하지 않는다. 먼저 task-post, slot-launch, batch-seal, decode-complete, sample, route, write checkpoint를 추가한다. backend subspan은 decode가 실제 first divergence 또는 critical path일 때 확대한다. 이렇게 위에서 아래로 좁혀야 CUDA trace의 많은 kernel 중 요청 증상과 관련된 실행을 고를 수 있다.

마지막으로 slot ID만 남긴 오래된 로그를 개선한다. slot generation, task incarnation, batch generation과 sequence ID를 같이 기록하고 cache operation에도 같은 generation을 붙인다. 이 네 좌표가 있으면 reuse 자체는 정상 최적화로 유지하면서 late work만 정확히 격리할 수 있다.

수정 후 다시 검증한다.

Slot과 task의 세대를 닫았으면 다음 질문은 여러 활성 작업이 한 실행 묶음에 들어갈 때 그 정체성이 보존되는가다. 26장은 이 generation 원장을 batch membership, row mapping과 결과 반환 순서로 확장해 동시성이 커져도 한 요청의 상태가 다른 요청으로 새지 않는지 추적한다.
