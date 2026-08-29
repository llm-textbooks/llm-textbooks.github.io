# 24장. Transformers continuous batching을 서버처럼 해부하기

Transformers 5.15.1에는 한 호출 안에서 생성 루프를 도는 `GenerationMixin.generate()`뿐 아니라 여러 요청을 background generation loop에서 함께 처리하는 continuous batching 구현도 있다. 둘을 “같은 generate의 빠른 모드”라고 합치면 안 된다. request queue, scheduler, paged cache, offload, CUDA Graph와 output router라는 새로운 owner가 생기기 때문이다.

이 장의 고정 소스는 Hugging Face Transformers 5.15.1, commit `550d7b3834670483a4df436541272c055dc364bf`, Apache-2.0이다. 실험적·변경 가능성이 큰 surface는 해당 고정 revision의 사실로만 설명한다. 일반적인 production 안정성이나 이후 release의 API 호환성을 추론하지 않는다.

23장에서 본 고전 `generate()`는 호출 하나가 이미 고정된 tensor batch, cache와 종료 loop를
소유한다. 이 장의 manager는 그 loop를 빠르게 만든 옵션이 아니다. 오래 사는 background thread가
서로 다른 시각에 들어오고 나가는 요청을 `RequestState`로 보존하고, 매 iteration에 실행 batch를
다시 만들며, request보다 오래 사는 paged cache와 static IO buffer를 운영한다. 같은 model
`forward`를 호출하더라도 ownership과 동시성 계약이 바뀐다.

이 차이를 대표 요청 `R7`로 따라간다. `R7`은 11-token prompt, 최대 4개 새 token, streaming=true인 텍스트 요청이다. 처음에는 manager의 bounded input queue에 있고 cache block은 없다. processor가 이를 waiting domain으로 옮기고 scheduler가 선택하면 running request가 되어 prompt 일부 또는 전체가 static input buffer의 row를 점유한다. cache allocator가 물리 page를 붙이고 model step이 새 token을 계산한다. CPU update가 token을 `RequestState`에 commit한 뒤에야 streamer가 외부로 내보낸다. EOS 또는 length limit가 참이면 terminal output을 전달하고 scheduler와 cache allocator가 R7의 소유권을 제거한다.

이 순서 가운데 GPU 계산 완료와 request 완료는 같은 사건이 아니다.

원문은 다음 세 소유자를 구분하며 읽는다.

- 전체 수명 주기의 시작점은 [`ContinuousBatchingManager`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/continuous_api.py#L553-L1081)다.
- 개별 요청의 상태는 [`RequestState`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/requests.py#L83-L361)가 소유한다.
- 실행 배치 조립은 [`ContinuousBatchProcessor`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/continuous_api.py#L190-L551)가 맡는다.

이 장의 모든 설명은 이 리비전의 호출과 필드에 한정한다.

| 수명 질문 | 고전 `generate()` | continuous manager |
|---|---|---|
| 새 요청을 받는 시점 | 함수 호출 전에 batch 확정 | background loop가 계속 admission |
| 실행 row의 정체 | 호출 동안 대체로 같은 sequence 집합 | step마다 waiting/running에서 재구축 |
| KV owner | generation 호출과 sequence row | processor의 cache allocator와 request ID |
| 종료 판정 | decoding loop의 unfinished mask | request별 update와 manager stop을 분리 |
| 외부 전달 | 반환 tensor 또는 호출 streamer | output router·callback·request iterator |
| 취소 | 표준 loop의 일반 per-request API가 아님 | cancel queue 뒤 안전한 scheduler 경계 |
| static buffer | 호출 내부 tensor 중심 | manager session의 IO pair와 graph가 재사용 |

따라서 23장의 processor·warper·stopping 순서는 일부 재사용되더라도 그 state container는 그대로
옮겨오지 않는다. continuous path는 지원하는 processor를 batch-aware 형태로 구성하고, 서로 다른
길이의 요청을 flat token layout에 맞춰 적용한다. 고전 generate에서 가능했던 beam search,
assisted generation, arbitrary custom stopping·streamer가 이 경로에도 자동으로 제공된다고 가정할
수 없다. 마지막 절에서 CLI가 실제로 제한하는 surface를 source로 닫는다.

고전 경로에서 호출 stack을 소유한 application thread는 `generate()`가 반환될 때 tensor batch와
cache lifetime이 끝났다고 이해할 수 있다. continuous 경로의 application thread는 `add_request`
뒤 즉시 돌아오고, 실제 owner는 background generation thread와 processor다. caller의 Python scope가
끝나도 R7은 queue, scheduler와 GPU cache에서 살아 있다. 이것이 client disconnect가 곧 취소가
아닌 이유이며, manager stop과 request finish를 분리해야 하는 이유다.

동시성도 “여러 thread가 model.forward를 호출한다”로 설명하지 않는다. 여러 application coroutine이
동시에 request를 제출할 수 있지만 manager는 이를 queue로 serialize하고, processor가 한 step의
flat batch로 만든다. async IO는 CPU preparation과 GPU work를 겹치지만 request state commit은
future-state row 순서를 따라야 한다. 외부 동시성, scheduler batch concurrency와 CUDA stream overlap은
서로 다른 층이다.

cache lifetime의 차이는 더 물리적이다. classic dynamic cache는 sequence tensor와 함께 step마다
성장하고 호출 종료 때 owner가 사라지는 그림으로 읽을 수 있다. continuous paged cache는 manager
session이 큰 pool을 먼저 소유하고 request는 block mapping과 refcount를 빌린다. R7 종료는 pool
allocation을 해제하는 사건이 아니라 R7의 mapping을 반환하는 사건이다. prefix sharing이 있으면
일부 block은 다른 request 때문에 계속 살아 있다. GPU reserved memory가 줄지 않았다는 이유만으로
finish가 실패했다고 판단하지 않는다.

반대로 static pool이 존재한다는 사실은 request data가 안전하게 지워졌다는 증명도 아니다. 새
request가 block을 재사용하기 전 refcount, complete flag와 generation이 올바르게 전환돼야 한다.
운영 보안 요구가 tenant 간 physical memory sanitization까지 포함한다면 allocator의 논리 ownership
외에 overwrite 정책을 별도로 감사해야 한다. 이 고정 소스의 기능과 배포자가 요구하는 isolation
정책을 섞지 않는다.

## 24.1 첫 15분: 요청 하나의 사건 원장을 만든다

요청을 넣기 전에 model/tokenizer revision, token IDs, max new tokens, EOS IDs, streaming 여부를 고정한다. manager 설정에서는 max batch tokens, cache block/page 수, block size, prefix sharing, offloading, async batching, compile/CUDA Graph를 기록한다. GPU에서는 UUID, SM, driver/런타임, allocated/reserved memory를 남긴다.

관측 순서는 `add_request`→input queue→scheduler waiting/active→`prepare_next_batch`→block allocation→tensor packing→`ModelRunner.compute_batch`→`update_batch`→output router→finish/free다. 각 단계에 request status, current length, allocated blocks, scheduled Q tokens, max KV read, selected fast path와 graph replay 여부를 기록한다.

이 원장은 같은 요청이 queue의 논리 상태, 이번 iteration의 임시 batch row, cache의 물리 block을 차례로 점유한다는 사실을 드러내야 한다. 세 좌표를 request ID 하나로 뭉개면 GPU 계산 완료, token commit, 외부 전달, block 반환 중 어디서 지연됐는지 구별하지 못한다. 앞에서 따라온 `R7`의 status와 owner가 바뀌는 순간마다 시각과 세 좌표를 함께 적는다.

## 24.2 admission: backpressure가 API 계약이 되는 지점

`generation/continuous_batching/continuous_api.py:787-818`, `ContinuousBatchingManager.add_request`의 핵심은 다음과 같다.

```python
if not self.is_tp_driver:
    return None
if self.background_thread_status.local_status >= BackgroundThreadStatus.FLUSH_AND_STOP:
    return None

state = RequestState(
    request_id=request_id,
    initial_tokens=list(input_ids),
    max_new_tokens=max_new_tokens,
    eos_token_id=eos_token_id,
    streaming=streaming,
    logit_processor_kwargs=logit_processor_kwargs,
)
self.input_queue.put(state, block=True, timeout=10)
self._has_new_requests.set()
```

TP driver가 아닌 process에서 no-op인 이유는 여러 rank가 동일 외부 요청을 독립 admission해 중복 sequence를 만드는 것을 막기 위해서다. 반환 `None`은 성공한 request ID가 아니므로 caller가 rank 역할을 모른 채 결과를 기다리면 영원히 오지 않는다.

flush/stop 상태에서 새 요청을 버리는 분기는 drain의 admission gate다. 이미 들어온 요청을 어떻게 끝낼지는 background status와 별도다. rolling shutdown 시험은 new request 거부, active request drain, hard stop 실패 전달을 구분한다.

`initial_tokens=list(input_ids)`는 caller가 이후 원래 list를 수정해도 request state가 변하지 않게 ownership을 넘긴다. queue put은 최대 10초 block한다. 따라서 application thread latency에 queue backpressure가 직접 드러난다. timeout 뒤 request가 들어갔는지 아닌지 모호하지 않은지, retry가 duplicate ID를 만들지 않는지 시험한다.

### R7이 queue item에서 길이를 가진 request state가 되는 순간

[`add_request`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/continuous_api.py#L763-L818)는 prompt를 model에 바로 보내지 않는다. driver rank인지, manager가 stop admission 경계를 넘었는지 확인하고 request ID를 정한 뒤 `RequestState`를 만든다. R7의 `initial_tokens`는 caller input에서 list로 복사된다. 이 복사는 HTTP handler가 재사용하는 input list와 background worker가 읽는 state의 ownership을 분리한다.

`RequestState`에는 “길이”가 하나만 있지 않다. initial token 수, 남아 있는 prefill token, 이미 처리된 position offset, 생성된 token 수와 현재 논리 길이가 서로 다른 field와 property에서 계산된다. prefix hit나 chunked prefill, offload rollback이 없으면 비슷해 보이지만 그 경우만 보고 합치면 안 된다. R7 prompt가 11개이고 아직 계산하지 않았다면 cache가 보존한 token은 0일 수 있다. 8-token prefix hit가 적용되면 새 query는 3개지만 attention이 읽는 logical 문맥은 11개다. 첫 생성 token이 commit되면 generated length는 1이고 다음 decode의 logical position은 11이다.

상태 전이는 [`RequestStatus`와 `RequestState`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/requests.py#L83-L351)로 표현된다. status는 UI label이 아니라 admission 대기, prefill, decoding, terminal 처리를 구분하는 control state다. 특정 status에 있다는 사실만으로 GPU가 지금 그 request를 실행 중이라는 뜻은 아니다. async mode에서는 device batch가 in-flight인 동안 CPU future state가 다음 logical 상태를 미리 표현할 수 있다.

`FutureRequestState`는 batch를 준비한 순간의 request가 model step 뒤 어떤 상태가 될지를 snapshot하고 `has_new_token`, complete block 수, query length를 함께 둔다. 그 사이 cancellation이나 다음 batch 준비가 일어날 수 있으므로 output row를 mutable request 목록에 위치로만 대응시키면 안 된다. update는 계산을 시작할 때 고정한 future state와 device output row를 맞춰 commit한다.

```text
HTTP thread의 input_ids
→ add_request의 list 복사와 RequestState
→ manager input_queue
→ scheduler waiting map/order
→ active map과 cache block
→ FutureRequestState가 이번 device row를 고정
```

`add_request` 반환은 GPU admission 완료가 아니다. queue에 넣었다는 뜻에 가깝고 cache capacity나 token budget 때문에 R7은 여러 iteration waiting일 수 있다. client-facing queue latency, scheduler waiting과 first model execution을 분리해야 TTFT가 어디서 생겼는지 설명된다.

request ID는 외부 caller가 결과를 찾는 routing key이면서 scheduler/cache ownership key다. 같은 ID를 동시에 재사용하면 output queue뿐 아니라 block mapping과 cancellation target도 모호해질 수 있다. application retry는 queue timeout을 “실패했으니 새로 넣어도 된다”로 단정하지 않는다.

### OutputRouter는 계산 결과와 소비자를 분리한다

[`OutputRouter`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/continuous_api.py#L84-L129)는 request별 callback과 공용 result queue 사이에서 `GenerationOutput`을 전달한다. streaming R7은 각 commit에서 output이 올 수 있지만 non-streaming 요청은 내부 token이 전진해도 terminal 전까지 외부 전달을 미룰 수 있다. 따라서 “output이 없다”는 사실이 request가 실행되지 않았다는 뜻은 아니다.

callback은 generation background thread와 application event loop의 경계를 건넌다. callback registration이 token 생성보다 늦으면 빠른 요청의 terminal output을 놓칠 수 있다. CLI non-streaming 경로가 future를 먼저 등록하고 그다음 `add_request`를 호출하는 이유가 이 race다. terminal output은 status와 optional error를 함께 운반한다. consumer는 completion과 FAILED를 같은 정상 종료로 취급하지 않아야 한다.

## 24.3 background loop: busy-spin을 피하면서 마지막 async 결과를 잃지 않는다

`continuous_api.py:922-984`, `_run_generation_loop`는 `prepare_next_batch`→`_generation_step`→`update_batch`를 반복한다. 요청이 없으면 event를 0.1초 timeout으로 기다린다. 이 timeout은 token polling interval이 아니라 idle busy-spin을 피하면서 stop/new request를 다시 확인하는 cadence다.

async batching에서는 첫 batch를 bootstrap한 뒤 compute와 이전 결과 update가 겹칠 수 있다. loop가 끝날 때 IO pair를 바꾸고 마지막 in-flight 결과를 한 번 더 update하는 이유다. 이 tail drain이 빠지면 device에는 계산된 마지막 token이 있어도 request state와 output queue에는 commit되지 않는다.

모든 예외는 critical error 경로로 모이고 `finally`에서 background status를 stopped로 바꾼다. “thread가 죽었다”와 “모든 request가 정상 완료됐다”는 다르다. 남은 request에는 실패 output이 전달돼 waiter가 유한하게 종료돼야 한다.

## 24.4 scheduling: cache full은 단일 boolean이 아니다

`ContinuousBatchProcessor.prepare_next_batch`는 cancellation을 먼저 지우고 CPU-offloaded cache도 해제한다. 다음으로 scheduler가 max batch tokens와 cache page 수 안에서 request, decode fast path, Q token 수와 최대 KV read를 반환한다.

```python
requests_in_batch, use_decode_fast_path, num_q_tokens, max_kv_read = (
    self.scheduler.schedule_batch(self.max_batch_tokens, self.cache.num_pages)
)
while requests_in_batch is None:
    if self.offloading_manager.offload_requests() == 0:
        raise RuntimeError(
            "No requests can be scheduled and no requests can be offloaded."
        )
    requests_in_batch, use_decode_fast_path, num_q_tokens, max_kv_read = (
        self.scheduler.schedule_batch(self.max_batch_tokens, self.cache.num_pages)
    )
```

`None`과 빈 list의 의미가 다르다. `None`은 cache 때문에 schedule이 불가능해 offload/retry가 필요하다는 신호이고, 빈 list는 처리할 요청이 없다는 신호다. 둘을 Python truthiness 하나로 합치면 cache pressure를 idle로 잘못 처리한다.

prefix를 완전히 공유하는 request를 offload해도 physical page가 하나도 늘지 않을 수 있으므로 loop는 실제로 영향 있는 victim을 찾을 때까지 반복한다. 더 이상 offload할 수 없으면 명시적으로 실패한다. 무한 retry보다 낫지만 manager 전체 critical error로 번지는지 request-local 실패인지 운영 정책을 확인해야 한다.

schedule 뒤 CPU-offloaded request를 restore하고 static input이면 Q/KV 크기를 padding한다. padding은 수학적 token 수를 늘리는 것이 아니라 compile/graph shape를 맞추는 storage·compute overhead다. metric은 logical scheduled tokens와 padded execution tokens를 분리한다.

### batch rebuild는 request를 tensor row로 번역하는 commit 전 단계다

continuous batching의 핵심은 “여러 요청을 batch로 묶는다”보다 매 step batch가 다시 만들어진다는
사실이다. R7이 prefill 중일 때는 prompt의 여러 token이 flat query 구간을 차지할 수 있다. 다음
iteration에 decode로 넘어가면 한 query token만 차지한다. 그 사이 새 요청 R8의 prefill이 들어오면
flat input은 `[R7의 decode 1개, R8의 prefill k개]`처럼 서로 다른 query length를 이어 붙인다.
model이 보는 첫 axis만으로 request boundary를 복원할 수 없으므로 cumulative sequence length,
position, cache read/write index와 logits row가 함께 조립된다.

[`prepare_batch_tensors`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/input_outputs.py#L319-L461)는 선택된 요청과 캐시 메타데이터를 정적 텐서에 넣는다.

[`get_model_kwargs`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/input_outputs.py#L468-L535)는 이를 모델 서명에 맞는 인자로 만들고, `get_cb_kwargs`는 블록 테이블 계열 입력을 분리한다. 따라서 값의 정합성을 검사할 때 텐서 모양만 비교해서는 부족하다.

R7 query 1개, R8 query 3개라면 logical total은 4다. cumulative query boundary는 개념상 `[0,1,4]`이고 position은 각 request의 cache frontier를 따라야 한다. logits가 각 sequence의 마지막 query에서만 필요하면 logits index는 R7의 0, R8의 3을 가리킨다. 이 가운데 하나가 이전 batch의 `[0,3,4]`를 재사용하면 allocation 안의 유효 주소를 읽으면서 다른 request의 logits를 commit할 수 있다.

static buffer의 capacity와 used extent도 분리한다. graph bucket이 query 8개를 요구해 physical input
tensor가 길이 8이어도 이번 logical total은 4다. 나머지 네 row는 padding이며 request owner가 없다.
model runner가 padded size로 실행할 수는 있지만 output update는 logical future-state row만 소비해야
한다. padding token의 sampled ID를 다음 request에 주는 것은 shape error가 아니라 ownership error다.

batch rebuild 경계에서 독자가 확보할 최소 원장은 다음과 같다.

```text
step epoch
selected request IDs와 각 query length
flat token range와 logical position range
cumulative query/KV lengths
cache read/write indices와 block table generation
logits index와 FutureRequestState output row
logical extent / padded extent / static capacity
```

이 원장은 scheduler 정책을 설명하려는 것이 아니다. FIFO와 prefill-first가 어느 요청을 먼저 고르는지는
뒤 장의 주제다. 여기서는 어떤 정책이 선택했든 그 선택을 tensor와 cache 좌표로 손실 없이 번역하고,
update가 같은 epoch를 소비하는지를 확인한다.

### block은 byte 조각이 아니라 참조 수와 complete 상태를 가진다

[`BlockManager`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache_manager.py#L37-L295)는 free block만 세지 않는다. block에는 parent/group 관계, refcount와 complete 의미가 있고 prefix sharing은 이 상태를 사용한다. partial block을 다른 request가 공유 가능한 prefix로 보려면 그 token 내용이 완전히 commit됐다는 보장이 필요하다.

block size가 4이고 R7 prompt가 11 token이라고 하자. cache가 처음부터 전부 계산한다면 세 block이
필요하며 마지막 block은 3 slot만 채운다. 첫 두 block은 완전한 8-token prefix가 될 수 있지만
세 번째 block은 다음 token이 오기 전 partial이다. hash가 같다는 이유만으로 partial block을 공유하면
unused slot과 이후 write ownership이 충돌한다. source의 complete marking과 shareable free 경계를
함께 읽어야 한다.

prefix hit가 8 token이면 R7은 공유 block 두 개의 refcount를 올리고 새 tail block을 소유할 수 있다.
R7이 끝나 `free_blocks`를 호출해도 공유 두 block은 refcount가 남아 있으면 physical free pool로
즉시 돌아가지 않는다. 따라서 “request finished 1건”과 “free blocks 3개 증가”는 동치가 아니다.
다른 request가 같은 prefix를 붙잡는 동안 cache byte는 살아 있다.

allocator는 logical past length와 query length에서 read/write index를 만든다. full attention과 sliding
attention allocator는 같은 계산을 쓰지 않는다. full attention은 과거 전체를 읽는 반면 sliding은
window 밖 위치를 재사용하거나 읽기 집합에서 제외할 수 있다. R7의 logical position 11이 physical
slot 11이라는 가정은 paged·sliding layout에서 성립하지 않는다. model에는 logical position을,
cache kernel에는 allocator가 정한 physical index를 전달한다.

allocation failure의 의미도 request 상태에 따라 다르다. waiting R8은 아직 block을 갖지 않은 채
다음 step을 기다릴 수 있다. 이미 active인 R7이 다음 token을 위한 block을 얻지 못하면 progress를
위해 offload 또는 victim 처리가 필요할 수 있다. scheduler 세부 정책을 외우기 전에 “누가 이미
물리 cache를 소유하고 있고, 실패 뒤 그 ownership이 유지되는가”를 묻는다.

cache correctness를 확인할 때 block table만 dump해서는 부족하다. request ID, logical token range,
physical block/slot, refcount, complete flag, allocator group과 generation을 한 행에 둔다. free pool에서
꺼낸 block ID가 같더라도 generation이 달라지면 이전 request의 관측과 구분해야 한다. CUDA pointer가
같다는 사실은 같은 논리 KV라는 뜻이 아니다.

### D2H/H2D offload는 request state rollback까지 포함한다

`OffloadingManager._offload_to_cpu`는 victim별 GPU block index를 모으고 GPU view에서 한 번 gather한 뒤 pinned CPU pool의 연속 run에 non-blocking copy한다. copy는 compute stream과 같은 ordering domain을 사용하므로 host-side 명시적 synchronize 없이도 뒤의 restore가 앞선 D2H를 보게 한다. 이 주장은 arbitrary stream에서도 안전하다는 뜻이 아니다. cache write, D2H와 H2D가 동일한 ordering 계약을 지키는지 확인해야 한다.

async batch가 이미 실행 중인 decoding request를 다음 schedule을 위해 offload할 수 있다. 이때 in-flight batch의 placeholder 또는 partial KV를 진짜 token으로 복원하지 않도록 `position_offset`을 한 칸 되돌리고 마지막 참 token을 `remaining_prefill_tokens`에 넣는 분기가 있다. CPU copy만 보고 logical rollback을 빼면 byte는 복원돼도 sequence 위치가 틀린다.

restore는 request별 CPU block ownership을 먼저 pop하고 새 GPU block table의 앞부분에 맞춰 H2D scatter한다. source의 TODO가 지적하듯 H2D 도중 예외가 나면 이미 pop한 CPU entry가 free pool로 돌아가지 못할 수 있다. fault injection은 H2D copy failure를 넣어 request mapping, free CPU blocks, GPU allocation과 `is_cpu_offloaded`가 원자적으로 복구되는지 확인한다.

관측 ledger에는 victim ID, starved demand, GPU block IDs, CPU pool IDs, D2H/H2D enqueue stream, `position_offset`, rollback token, reallocated GPU IDs, copy 완료 전후 owner를 둔다. `offloaded_requests_total` 하나로는 soft reset과 실제 CPU swap, logical rollback과 byte transfer를 구분할 수 없다.

### block allocation: 필요한 token과 physical block을 혼동하지 않는다

`generation/continuous_batching/scheduler.py:122-143`, `_allocate_blocks_if_needed`는 현재 길이와 이미 할당한 block capacity의 차이인 occupancy를 계산한다. 다음 token이 들어가지 않거나 block이 전혀 없으면 필요한 block을 요청한다.

allocation 실패 시 active request만 `starved_requests`에 physical block 수와 함께 남긴다. waiting request는 아직 cache를 소유하지 않으므로 기다리면 된다. 같은 allocation failure라도 active sequence는 앞으로 나아가기 위해 victim/offload가 필요하고 waiting sequence는 admission을 늦출 수 있다는 차이다.

block size를 크게 하면 metadata와 allocation 빈도는 줄 수 있지만 마지막 block의 internal fragmentation이 커진다. 작게 하면 block table과 allocator work가 늘 수 있다. prompt length histogram과 branching/prefix sharing을 넣어 free block count, wasted slots, allocation latency와 ITL을 함께 측정한다.

## 24.5 compute: forward, logits, sampling과 CUDA Graph가 한 step에 묶인다

`generation/continuous_batching/model_runner.py:101-143`, `ModelRunner.compute_batch`는 device-resident IO pointer를 얻고, model이 지원하면 LM head 전에 hidden state를 필요한 위치만 남기도록 `logits_to_keep`을 전달한다. 이어 block table 사용 여부와 compile 상태로 forward 함수를 고르고 CUDA Graph를 사용할지 정한다.

```python
forward_fn, use_cuda_graph = self._get_forward_fn(
    use_block_table=self.inputs_and_outputs.use_block_table
)
if not use_cuda_graph:
    with maybe_stream:
        forward_fn(model, batch_data, carry_over_ids, prev_output_ids, output_ids)
else:
    graph = self.inputs_and_outputs.get_graph()
    if graph is not None:
        with torch.cuda.stream(compute_stream):
            graph.replay()
    else:
        self._capture_graph(forward_fn, compute_stream, *args)
```

`use_block_table`은 단순 cache option이 아니라 decode/varlen forward 선택과 graph pool identity에 영향을 줄 수 있다. graph가 없으면 먼저 warm-up forward를 수행하고 thread-local capture mode에서 다시 실행해 저장한다. 첫 iteration 비용에는 정상 계산과 capture가 겹치므로 steady replay와 분리한다.

IO tensor 주소와 shape가 replay 계약을 만족해야 한다. request 내용은 static buffer 안에서 갱신하되 pointer와 captured operation topology는 유지한다. batch shape가 bucket을 벗어나면 padding, 다른 graph 또는 eager path가 필요하다. graph replay가 보였다는 사실만으로 현재 logical request가 올바른 slot을 읽었다고 보장되지 않으므로 output token과 block table generation을 함께 검증한다.

### model step은 forward 한 번이 아니라 logits row 선택과 sampling까지 닫힌다

[`ModelRunner.compute_batch`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/model_runner.py#L101-L133)는 input/output owner에게 device pointer를 받아 forward path를 고른다. model이 `logits_to_keep`을 지원하면 모든 query row의 vocabulary logits를 만들지 않고 sequence별 필요한 위치만 LM head에 남기도록 요청할 수 있다. 이 최적화는 output row mapping과 결합된다. 잘못된 logits index는 GPU memory를 절약하면서 정확히 다른 token의 hidden state를 분류하는 조용한 오류를 만든다.

R7 decode 1 row와 R8 prefill 3 row가 함께 있고 각 sequence의 마지막 row만 logits가 필요하다면
LM head의 logical 입력은 두 row다. R7은 flat row 0, R8은 flat row 3이다. `logits_to_keep=2` 같은
개수와 `[0,3]` 같은 위치 의미를 혼동하면 안 된다. model별 forward signature가 어떤 표현을
받는지 runner의 capability check와 실제 kwargs를 잇는다.

forward 결과는 [`_forward_process_and_sample`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/model_runner.py#L159-L227)에서 score 처리와 sampling으로 이어진다. batch-aware logits processor는 request마다 다른 history 위치나 kwargs를 flat batch에 맞춰 적용해야 한다.

고전 generate는 호출이 가진 `input_ids` matrix와 ordered `LogitsProcessorList`를 매 step 처리하지만, continuous path에서는 서로 길이가 다른 request state에서 필요한 token history를 static `output_ids`와 carry-over 영역으로 옮긴다.

processor 순서는 확률 의미다. repetition penalty, minimum length에 따른 EOS 억제, temperature,
top-k/top-p 같은 변환은 일반적으로 교환 가능하지 않다. 이 고정 revision이 지원하는 processor를
`ContinuousBatchingLogitsProcessorList`로 만드는 경로를 확인하고, classic generate에서 임의로
추가한 custom processor가 자동 전달된다고 쓰지 않는다. request-level `logit_processor_kwargs`는
지원 processor가 요구하는 per-request 값의 통로이지 arbitrary Python callable 전체를 보장하는
표현이 아니다.

sampling 결과가 `output_ids` device buffer에 쓰였다는 것은 아직 R7의 token이 아니다. async D2H와
`update_batch`를 거쳐 `RequestState.update_and_check_completion`에 들어갈 때 논리 token으로 commit된다.
이 구분은 critical error와 cancellation에서 중요하다. GPU가 token 42를 계산한 뒤 worker가
실패하면 외부 stream이 42를 받지 못할 수 있다. exactly-once token delivery를 주장하려면 device
compute, host commit과 router delivery의 세 경계를 별도 epoch로 추적해야 한다.

### stopping은 token을 commit하면서 request마다 판정한다

[`update_and_check_completion`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/requests.py#L228-L267)은 sampled token과 optional logprob를 state에 넣고 EOS와 max-new-token 조건을 판정한다. R7의 최대 새 token이 4라면 네 번째 token은 생성·commit된 뒤 terminal이 된다. length limit가 네 번째 token 계산 자체를 금지하는 것으로 이해하면 output 길이를 한 칸 틀리게 센다.

EOS도 processor와 stopping의 두 위치를 구분한다. minimum length 정책은 logits 단계에서 EOS 확률을
억제할 수 있고, 실제 EOS가 sampled되면 request update가 종료를 판정한다. EOS ID가 여러 개인
model, EOS가 지정되지 않은 경우의 sentinel, prompt 안에 EOS가 이미 있는 경우를 config normalization과
request state source에서 확인한다. classic generate의 모든 `StoppingCriteria` callable이 여기서
호출된다고 추정하지 않는다.

streaming R7은 token commit마다 `GenerationOutput`이 전달될 수 있다. output object의 token 목록이
이번 delta인지 누적 sequence인지 consumer가 source contract로 확인해야 한다. CLI streamer는 token
level output을 tokenizer decode state로 바꾸고 SSE text queue에 넣는다. byte-level 또는 subword
token은 한 token마다 완성된 Unicode 문자열이 아닐 수 있으므로 streamer가 decoder state를 소유한다.
request finish 뒤 decoder flush와 queue end marker가 모두 필요하다.

non-streaming 요청은 매 step state와 cache가 전진하지만 router delivery를 terminal까지 미룬다.
이때 client가 연결을 끊으면 handler는 `cancel_request`를 호출해야 GPU work와 cache ownership이
유한하게 끝난다. 단순히 asyncio future를 버리면 manager는 소비자 없는 request를 계속 생성할 수
있다. 반대로 cancel 반환 직후 block을 재사용됐다고 가정하면 안 된다. cancel queue를 processor가
drain하는 안전 경계가 cleanup commit이다.

### 최초 divergence를 한 request의 여섯 경계에서 찾는다

classic generate와 continuous manager의 첫 token이 다를 때 최종 text만 비교하지 않는다. 같은
model revision, token IDs, effective sampling config와 RNG 조건을 고정하고 greedy부터 시작한다.
그 뒤 다음 경계를 순서대로 비교한다.

```text
R7의 logical input token·position
→ paged cache를 반영한 model kwargs
→ selected logits hidden row
→ raw logits와 processor 후 scores
→ sampled device output row
→ RequestState에 commit된 token과 router output
```

첫 경계부터 다르면 tokenizer가 아니라 CB tensor packing·prefix position을 본다. model kwargs는
같고 logits부터 다르면 paged attention backend와 cache read/write를 본다. raw logits는 같고 processed
score부터 다르면 supported processor와 history packing이다. sampled row는 같고 committed token이
다르면 async row alignment다. commit은 같고 client text만 다르면 router·streamer decode 경계다.

sampling mode에서는 RNG owner도 기록한다. classic generate의 generator 전달 방식과 continuous
batch에서 request가 batch row를 옮길 때 random draw 순서가 같다고 보장할 수 없다. stochastic output
문자열이 다르다는 사실만으로 correctness failure를 선언하지 않는다. greedy parity, score 분포와
request별 seed 지원 계약을 먼저 확인한다.

## 24.6 update: GPU 결과가 request의 token이 되는 commit 경계

`continuous_api.py:424-463`, `update_batch`는 device output을 request별 future state와 맞춘다. async mode에서는 schedule 뒤 finish/offload된 request의 token slot도 소비해 index 정렬을 유지하되 state에는 적용하지 않는다. 이 index 소비가 빠지면 이후 request가 앞 request의 token을 받는다.

첫 새 token이 나오면 PREFILLING에서 DECODING으로 바꾸고 `update_and_check_completion`으로 token·logprob·EOS·length를 commit한다. 완료 block을 shareable로 표시하는 시점도 이 결과와 연결된다. 미완성 KV를 prefix hit로 노출하면 다른 request가 아직 쓰는 block을 읽을 수 있다.

finish면 scheduler에서 request를 제거하고 새 admission block을 해제한다. streaming 또는 finished request만 output에 넣어 batch로 전달한다. non-streaming request는 중간 token을 외부에 내보내지 않지만 내부 state와 cache는 매 step 전진한다.

fork는 update 뒤 수행된다. free block이 충분한 child만 기존 cache를 공유/복사하고 나머지는 equivalent initial request로 waiting queue에 넣는다. cache copy는 한 CUDA stream context에서 모아 수행된다. source의 FIXME가 밝히는 async+sliding-window race 가능성은 고정 revision의 중요한 한계다. production fixture에서 fork, sliding window, async batching을 동시에 켜고 sequence별 first divergent token과 cache generation을 검사한다.

### cancellation과 cleanup

public `cancel_request`는 cancel queue에 ID를 넣고 event를 깨운다. 실제 제거는 scheduler `clear_cancelled_requests`가 cancellation lock 아래 active 또는 waiting map에서 state를 꺼내고 waiting order를 정리하며 cache block을 해제할 때 일어난다. 호출 반환은 cleanup 완료가 아니다.

CPU offload copy가 진행 중이거나 async compute가 이미 해당 request를 포함하면 lifetime fence가 더 필요하다. processor는 취소 state의 CPU cache도 free하고 async update에서는 이미 끝난 row의 output index를 소비한다. fault test는 cancel 호출 시각, scheduler 제거, GPU/CPU cache free, 마지막 output과 다음 block reuse를 순서대로 기록한다.

### 한 step을 함수 세 개로 끊어 관찰한다

`ContinuousBatchProcessor._generation_step`은 이 구현의 transaction 경계다. `prepare_next_batch`가 scheduler 선택과 cache allocation을 static IO tensor에 반영하고, runner가 model forward와 sampling을 수행하며, `update_batch`가 결과 token을 request 상태에 commit한다. 세 함수 사이에서 오류가 났을 때 rollback 책임이 다르므로 “batch step 실패” 한 줄로 합치지 않는다.

IO 쪽의 `get_model_kwargs`는 token·position·mask 같은 고정 buffer를 model signature로 바꾸고, `get_cb_kwargs`는 block table과 cumulative sequence metadata를 continuous-batching attention 계약으로 분리한다. scheduler가 올바른 request를 선택했어도 이 조립에서 row, offset 또는 block table generation이 어긋나면 다른 request의 KV를 읽을 수 있다.

관측 표에는 step마다 scheduled request IDs, logical query length, cache block generation, model-kwargs tensor shape와 주소, CB kwargs의 cumulative lengths, sampled row index, commit된 token을 한 행에 둔다. CUDA Graph를 쓰면 주소 안정성이 필요하므로 값만 비교해서는 부족하다. 반대로 주소가 안정적이어도 logical row mapping이 맞다는 보장은 없다. pointer 계약과 request ledger를 동시에 검사해야 한다.

### paged backend 전환은 manager session에 속하는 가역 상태다

`continuous_api.py:631-663`, `switch_to_cb_friendly_attn`은 현재 backend가 flash도 paged도 아니고 model이 flash를 지원할 때 FA3, 다음 FA2 순서로 가용성을 확인한다. 둘 다 없으면 원 backend를 유지하되 최종 이름에는 `paged|`를 붙인다. 따라서 `sdpa`는 환경에 따라 `paged|flash_attention_3`, `paged|flash_attention_2`, 또는 `paged|sdpa`가 된다. 이미 `paged|...`이면 아무것도 바꾸지 않아 재호출이 안전하다.

manager는 변경 전 값을 `_original_attn_impl`에 보존하고 stop에서 복원한다. persistent manager 재사용 시에는 stop이 복원했을 가능성이 있어 다시 전환한다. 이 상태를 process-global option처럼 설명하면 안 된다. model config와 manager session lifetime이 만나는 가역 mutation이다.

paged 전환 자체가 CUDA Graph를 capture하는 것은 아니다. paged backend는 비연속 KV block table 계약을 제공하고, mask 필요 여부와 workload 조건은 별도의 graph 정책이 판단하며, warmup이 실제 graph를 준비한다. “Flash를 선택했으므로 graph가 켜진다”는 인과를 쓰지 않는다.

## 24.7 async IO의 happens-before를 event 세 개로 읽는다

`input_outputs.py:657-820`, `ContinuousBatchingAsyncIOs`는 두 host-device pair를 만든다. 각 pair의 host와 device static tensor는 CUDA Graph가 요구하는 주소 안정성을 유지하고, 두 벌을 번갈아 써 batch N의 compute·D2H와 batch N+1의 CPU 준비·H2D를 겹친다. 대가는 static device buffer와 graph set을 두 벌 보유하는 VRAM이다.

```text
host pack N
  -> H2D copy(N) -> record h2d_over(N)
  -> compute waits h2d_over(N) -> compute(N)
  -> record compute_over(N) -> D2H waits compute_over(N)
  -> D2H copy(N) -> record d2h_over(N)
  -> switch pair -> CPU waits d2h_over(N) -> request-state update
```

`HostDeviceIOPair.transfer_inputs_h2d`와 `transfer_outputs_d2h`는 `non_blocking=True` copy를 enqueue한다. Python 함수가 반환됐다는 사실은 copy 완료가 아니다. `get_model_kwargs`가 H2D event를 기록하고 compute stream에 wait를 걸며, `retrieve_device_outputs`가 compute→D2H ordering을 만든다. pair를 바꾼 뒤 `prepare_batch_update`가 해당 pair의 `d2h_over.synchronize()`를 통과해야 host output을 읽는다.

host tensor는 CPU에 있다고 자동으로 pinned인 것이 아니다. `_setup_static_tensors`는 CPU device이면서 accelerator가 존재할 때 `pin_memory`를 요청한다. pinned memory는 DMA와 non-blocking transfer를 가능하게 하는 자원이지만 allocation·resident pressure 비용도 있다. 이 조건과 실제 tensor pinned 상태를 독자 실험 항목으로 분리한다.

session reset은 두 pair의 event와 H2D·D2H·compute stream을 synchronize한다. 이것이 pending work가 끝난 뒤 buffer를 다시 쓰는 명시적 fence다. 반면 manager `join`은 Python generation thread만 기다린다. thread가 종료됐다는 사실과 모든 CUDA work가 완료됐다는 사실을 혼동하지 않는다.

### 두 buffer가 겹치는 것은 두 request가 동시에 같은 model을 실행한다는 뜻이 아니다

async batching의 목표를 정확히 말하면 batch N의 GPU compute·output transfer와 batch N+1의 CPU
packing·input transfer 일부를 겹치는 것이다. 두 CUDA graph가 같은 model weights에 무제한 concurrent
forward를 수행한다고 단정할 수 없다. compute stream은 ordering을 유지하고 IO pair가 번갈아
producer와 consumer 역할을 바꾼다. overlap의 단위는 request가 아니라 batch epoch와 buffer pair다.

pair A가 R7/R8 batch N을 담고 있다고 하자. host A에서 input이 완성되면 H2D stream에 copy하고
`h2d_over(A,N)` event를 기록한다. compute stream은 이 event를 기다린 뒤 A의 device pointer로
forward/sampling을 실행하고 `compute_over(A,N)`을 기록한다. D2H stream은 compute event를 기다려
output을 host A로 옮기고 `d2h_over(A,N)`을 기록한다. 그동안 CPU는 pair B에 batch N+1을 pack할
수 있다. 다음에 A를 재사용하려면 CPU update가 A의 D2H 완료를 확인하고 이전 future-state와 output
row를 모두 소비한 뒤여야 한다.

여기서 세 종류의 잘못된 “완료”가 자주 섞인다. `copy_(non_blocking=True)` Python 호출 반환은
enqueue 완료다. CUDA event record도 event 앞 stream work가 이미 끝났다는 뜻이 아니라 그 위치에
completion marker를 놓았다는 뜻이다. `prepare_batch_update`가 host 값을 안전하게 읽으려면 해당
D2H event synchronize가 끝나야 한다. wall-clock profiler range가 닫혔다는 사실만으로 host buffer
visibility를 주장하지 않는다.

tail drain은 이 pipeline의 필수 단계다. 새 request가 없고 stop 조건이 참이 되어 loop를 바로
끝내면 마지막 pair의 compute 결과가 아직 D2H/update를 통과하지 않았을 수 있다. loop가 pair를
swap하고 마지막 update를 한 번 수행하는 source 흐름은 pipeline stage 수만큼 남은 결과를 비우는
drain이다. R7이 마지막 batch의 유일한 request라면 이 한 번을 빼먹을 때 GPU trace에는 정상 token이
있지만 client는 끝없이 기다린다.

double buffering의 비용도 명시한다. host staging과 device static tensors, graph lookup owner가 두
벌 필요하다. max batch token, mask, block table와 output capacity가 큰 설정에서는 pair 배수가
reserved memory에 직접 반영된다. overlap을 켜서 ITL이 줄 가능성과 static footprint·warmup capture가
늘어나는 비용을 함께 측정한다. 짧고 작은 workload에서는 transfer가 충분히 길지 않아 overlap
이득보다 synchronization과 두 buffer 비용이 클 수도 있다. source는 가능한 중첩 구조를 증명하지만
특정 GPU에서의 이득은 trace로 확인해야 한다.

### CUDA Graph key는 request identity가 아니라 실행 shape를 식별한다

[`ContinuousBatchingIOs._get_graph_key`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/input_outputs.py#L542-L561)와
async wrapper의 graph access는 현재 static shape에 맞는 graph를 찾는다. R7이라는 ID나 prompt text는
graph key가 아니다. 서로 다른 request 집합도 padded query/KV shape와 forward path가 같으면 같은
captured operations를 replay할 수 있다. request별 값은 주소가 안정된 static buffer에 새로 써 넣는다.

주소 안정성과 의미 안정성은 별개다. pair A의 input pointer가 capture 때와 같아도 block table이
batch N-1 값이면 replay는 정확히 잘못된 cache를 읽는다. 반대로 logical values가 맞아도 pointer를
새 tensor로 바꾸면 capture가 참조한 옛 storage를 읽는다. graph 진단에는 pointer, capacity와 함께
used extent, batch epoch, row→request map과 cache generation을 넣어야 한다.

`ModelRunner._get_forward_fn`은 block table 사용 여부와 compile/graph 조건에서 concrete forward를
고른다. decode fast path와 variable-length path는 같은 graph topology가 아닐 수 있다. query/KV
shape가 bucket에 맞지 않거나 sliding attention이 fast path를 막으면 eager 또는 다른 graph가
선택될 수 있다. startup log의 “CUDA graph enabled”는 모든 runtime batch가 replay된다는 뜻이 아니다.
step별 selected forward, graph key, hit/capture/fallback reason을 관측해야 한다.

첫 capture는 steady replay와 비용 구조가 다르다. `_capture_graph`는 필요한 warmup forward와 capture
과정을 수행하고 graph buffer에 저장한다. warmup helper는 여러 decode bucket과 varlen shape를
시도하지만 일부 실패를 warning으로 흡수할 수 있다. 상위 `warmed_up=True`는 시도를 기록할 뿐 모든
key가 준비됐다는 증명서가 아니다. 첫 실요청에서 capture가 나타나면 TTFT outlier가 될 수 있으므로
준비된 key 집합을 readiness 관측으로 둔다.

graph와 pair lifetime도 구분한다. request finish는 graph를 free하지 않는다. processor reset은
request/cache·IO의 동기화를 수행하면서 session 재사용을 준비할 수 있고 graph buffer는 더 오래
살 수 있다. manager destroy와 Python owner 해제가 graph reset에 도달하는 시점은 background thread
join과 stream synchronize 뒤여야 안전하다. reserved CUDA memory가 request 종료 뒤 유지되는 현상은
graph/session pool일 수 있으므로 active request leak과 같은 지표로 판단하지 않는다.

async parity를 검증할 때는 sync mode, async eager, async graph replay의 세 경로를 같은 greedy
request set으로 비교한다. R7 단독 decode만 쓰면 퇴화 shape가 row-alignment 버그를 숨길 수 있으므로
길이가 다른 prefill과 decode를 섞고, finish되는 request와 계속되는 request가 같은 batch에 있게 한다.
첫 divergence가 device output 전이면 packing/graph이고, device output은 같지만 host commit부터
다르면 D2H event 또는 pair swap이다. client output만 다르면 router 경계다.

## 24.8 stop, join, destroy에서 반드시 확인할 비대칭

`start`는 status와 fatal error를 초기화하고 generation thread를 만든다. `stop`은 flush 또는 hard-stop 신호를 보내고 `block=True`일 때만 `join`한다. timeout 뒤 thread가 살아 있으면 warning만 남고 reference도 유지된다. 그러므로 timeout 반환은 teardown 성공이 아니다.

`stop`은 session 보존 여부에 따라 batch processor를 버리거나 model에 cache하고 원 attention backend를 복원한다. `destroy`는 실행 중이면 blocking stop한 뒤 CPU communication group을 해제한다. async IO 객체에는 별도의 public destroy가 없으므로 안전한 해제는 loop tail drain, blocking join, reset/synchronization과 Python ownership 종료의 조합에 의존한다.

운영 체크리스트에는 stop mode, timeout, thread alive, 마지막 batch update, 세 stream의 pending 여부, pair index, event 완료, batch processor reference와 attention backend 복원 값을 둔다. timeout을 강제로 발생시킨 뒤 즉시 새 session을 시작하는 반례는 독자용 런타임 실험이다. 정적 소스만으로 race가 실제 발생한다고 주장하지 않는다.

### classic generate와 attention control plane의 위치가 다르다

continuous manager는 이미 생성된 model의 config dispatch key를 바꾸고 `paged|` decorator를 합성할
수 있다. classic generate는 호출마다 선택된 model attention implementation과 cache object를 사용하지만,
manager는 session 시작에서 paged-friendly backend를 선택하고 stop에서 원 값을 복원한다. 따라서
같은 model object를 classic 호출과 manager session이 동시에 공유하면 config mutation의 owner와
시간 범위를 명확히 해야 한다. 이 고정 source는 manager의 저장·복원 흐름을 보여 주지만 arbitrary
동시 혼용이 안전하다고 보증하지 않는다.

### classic 호출의 output과 async manager output은 다른 pipeline이다

classic generate는 호출 stack 안에서 forward 결과를 CPU loop가 소비하고 streamer 또는 return
tensor로 보낸다. continuous path는 두 IO pair와 H2D·compute·D2H event 뒤에 future-state update와
OutputRouter를 둔다. 비교 좌표는 단순 copy API가 아니라 producer, destination buffer owner,
completion primitive, request state mutation, buffer reuse fence와 consumer dispatch다. 이 여섯 필드를
채워야 동일 model의 첫 token 차이가 model 수치인지 manager overlap인지 가를 수 있다.

### warm-up 플래그는 graph 준비 성공의 증명서가 아니다

manager의 `warmup()`은 processor가 없으면 `_create_batch_processor()`로 만들고 `ContinuousBatchProcessor.warmup`을 거쳐 `ModelRunner.warmup`에 위임한 뒤 `warmed_up=True`를 기록한다. persistent manager의 다음 context는 이 플래그를 보고 warm-up을 건너뛴다. 문제는 하위 `run_one_warmup()`이 block allocation 실패나 prepare/forward/capture 예외를 warning으로 낮추고 `0.0`을 반환한다는 점이다. 상위 호출은 실패 개수를 받지 않으므로 `warmed_up=True`는 “시도했다”에 가깝고 “모든 graph bucket이 capture됐다”는 증명이 아니다.

warm-up shape는 임의 예제가 아니다. varlen 경로는 `max_batch_tokens`와 cache 전체 capacity에서 query를 뺀 최대 KV read로 먼저 실행해 pool의 큰 allocation을 앞당긴다. decode는 request 수 `1, 2, 4, ...`를 `max_requests_per_batch`까지 훑는다. runtime의 padding bucket lattice와 맞추려는 선택이다. sliding attention 때문에 decode fast path가 꺼졌으면 이 축은 생략된다. async IO는 pair마다 static tensor와 graph buffer가 독립이므로 두 pair를 swap하며 각각 warm-up한다.

단일 warm-up은 fake request와 실제 cache block을 할당하고 static IO packing, model kwargs 조립, `compute_batch`와 capture 경계를 통과한다. `finally`는 fake request block을 반드시 해제한다. 그러나 부분적으로 커진 graph memory pool이나 fragmentation까지 rollback하지는 않는다. warning 문구가 “load 중 OOM 가능”을 직접 언급하는 이유다.

운영 readiness는 `warmed_up` 하나가 아니라 다음 벡터로 판단한다.

- varlen 최대 shape의 성공 여부와 duration
- decode bucket별 성공·실패와 captured graph identity
- async pair별 graph 존재 여부
- warm-up 전후 free cache block 수
- graph pool reserved bytes와 allocation 실패
- 첫 실요청의 capture/fallback 여부

실행 검증을 하지 않는 이 책의 정적 결론은 실패가 nonfatal로 흡수되는 control flow까지다. 실제 capture coverage는 독자가 선택적으로 위 벡터를 관측해야 한다.

### processor 재사용은 reset이지 config 재해석이 아니다

`_create_batch_processor()`의 cold path는 concrete `PagedAttentionCache`를 만든 뒤 실제 block 수, 최대 batch token과 prefix-sharing 가능성으로 continuous-batching config를 다시 해석한다. sliding attention group이 있으면 decode fast path를 끄고, scheduler 이름이 등록되지 않았으면 warning 뒤 FIFO를 고른다. 이어 cache, scheduler, IO, offload manager, logits policy와 model runner를 하나의 execution lifetime으로 묶는다.

warm-up이 먼저 processor를 만들었다면 background loop의 factory 호출은 새 객체를 만들지 않는다. 기존 processor에 `reset()`을 호출해 돌려준다. persistent session도 같은 경로다. reset은 offload state와 scheduler request maps를 비우고, async IO stream/event를 synchronize하며, request cache block을 해제한다. 하지만 cache allocation, graph topology, scheduler class와 constructor 때 snapshot한 `do_sample`, logprob, async policy는 유지한다.

따라서 새 session에 다른 workload hint나 config object를 넘겼다고 persistent processor의 sizing과 실행 정책이 자동으로 바뀌지 않는다. `ContinuousMixin.init_continuous_batching()`은 cached manager가 있으면 새 generation config, CB config와 workload hints를 적용하지 않고 paged attention backend만 다시 켜 반환한다. cold init에서 EOS가 없으면 전달된 `GenerationConfig` 자체를 `-1` sentinel로 변경한다. 원 config가 immutable snapshot이라고 가정하면 이 mutation도 놓친다.

session ledger에는 manager/processor identity, cold 또는 reuse, 요청된 새 config digest, 실제 processor snapshot, scheduler class, cache capacity, graph set과 reset completion을 둔다. 설정은 바뀌었는데 behavior가 그대로라면 option parser보다 persistent ownership부터 확인한다.

### 오류는 input queue·active·waiting의 세 ownership domain을 지난다

request-local `_handle_request_error`는 state를 FAILED로 만들고 active request라면 이미 생성된 token도 수집해 terminal output을 보낸다. processor의 `fail_all_requests()`는 active request마다 scheduler `finish_request`를 호출하고, waiting request에는 CPU-offloaded cache를 먼저 해제한 뒤 map과 ordering deque를 비운다. 아직 manager input queue에만 있는 요청은 이 함수가 소유하지 않는다.

manager의 `_fail_all_remaining_requests()`가 input queue를 nonblocking drain한 뒤 processor의 active/waiting failure로 이어 준다. 그런데 processor construction 이전에 치명 오류가 나면 queue는 비워지지만 `_handle_request_error`를 호출할 processor가 없다. 이 경로에서는 queued caller에게 terminal output이 전달되지 않을 수 있다. output waiter의 유한 종료를 보장하려면 fatal error와 thread liveness를 별도로 확인해야 한다.

### 한 request 오류와 worker 전체 사망을 같은 recovery로 다루지 않는다

R7의 잘못된 processor kwargs처럼 request 하나의 검증에서 잡을 수 있는 오류는 R7을 FAILED로 만들고
다른 request를 계속 진행할 여지가 있다. 반면 model forward, static IO alignment나 TP control-plane에서
예외가 background loop 밖으로 올라가면 어느 output row와 cache write가 commit됐는지 확정하기 어렵다.
이 경우 고정 revision의 top-level handler는 `fatal_error`를 보존하고 hard stop을 요청하며 남은
ownership domain을 실패 처리한다. catch 범위가 recovery의 신뢰 경계다.

“remaining requests”는 단일 collection이 아니다. 아직 manager input queue에만 있어 processor가
보지 못한 request, scheduler waiting map의 request, active map과 GPU block을 소유한 request가 있다.
CPU offload map에 state가 있을 수도 있다. failure cleanup이 하나의 map만 비우면 caller 또는 cache
ownership이 유실된다. 제출 총수는 terminal success, terminal failure, 명시적 cancellation과 아직
살아 있는 request의 합으로 보존되어야 한다.

processor가 존재하는 경로에서 active R7을 실패시키면 이미 commit된 generated token을 포함한
terminal `GenerationOutput`을 만들고 scheduler finish를 통해 block을 해제한다. waiting request는
GPU block이 없을 수 있지만 prefix refcount나 CPU-offloaded state가 있다면 그 owner도 정리해야 한다.
input queue request는 processor의 helper가 알 수 없으므로 manager가 별도로 drain한다. 이 계층 분리가
함수 중복이 아니라 ownership domain의 차이다.

processor 생성 전 실패라는 반례는 특히 중요하다. request가 queue에 들어간 뒤 factory가 cache
capacity 또는 backend 초기화에서 실패하면 manager는 queue item을 꺼낼 수 있지만 processor의
`_handle_request_error`와 output router delivery 경로가 준비되지 않았을 수 있다. source가 terminal
output을 보장하지 않는 경로라면 CLI는 manager fatal state를 감시해 waiter를 별도로 깨워야 한다.
“queue가 비었으니 cleanup 완료”라고 보고하면 소비자는 영원히 대기할 수 있다.

TP에서는 한 rank의 오류가 다른 rank의 loop를 collective에 남겨 두지 않도록 stop state를 합의한다.
`BackgroundThreadStatus`가 local과 TP 상태의 최대를 취해 flush 요청이 이미 발생한 hard stop을
약화시키지 않는 이유다. 하지만 status 합의는 CUDA collective와 Python thread가 모두 종료됐다는
barrier가 아니다. error timestamp, rank별 local/TP status, 마지막 완료 batch epoch, thread liveness와
stream event completion을 함께 봐야 한다.

재시작도 자동 recovery로 가정하지 않는다. `fatal_error`를 초기화하고 새 thread를 만들 수 있는
코드가 있어도 이전 processor, graph, attention backend mutation과 cache allocator가 일관된 reset을
마쳤는지 별도다. `destroy` docstring의 restart 금지와 기계적으로 막는 guard가 일치하는지도 읽는다.
서비스 supervisor가 process restart를 택하는 이유는 단순 편의가 아니라 partially committed GPU
state를 가장 명확한 lifetime 경계에서 버리기 위함일 수 있다.

장애 관측은 traceback 하나로 끝나지 않는다. 최초 예외 함수와 batch epoch, R7이 속한 ownership
domain, device compute·D2H·host commit 중 마지막 완료 경계, terminal output 전달 여부, CPU/GPU block
회수와 worker status를 한 incident timeline으로 묶는다. 이 정보가 있으면 duplicate token, lost
request, cache leak과 단순 client disconnect를 서로 다른 사고로 분류할 수 있다.

`ContinuousBatchProcessor.handle_batch_error()`라는 batch-local helper도 있지만 이 고정 revision의 repository 내부 generation loop caller는 없다. `_generation_step`이나 update에서 예외가 나면 loop 최상위 `except`가 `_handle_critical_error`로 보내 manager-wide hard stop과 전체 request failure를 수행한다. 함수가 정의돼 있다는 사실을 recovery policy가 연결돼 있다는 증거로 쓰지 않는다.

오류 주입 ledger에는 exception 최초 함수, processor 생성 여부, input queue drain 수, active/waiting 수, CPU/GPU block 해제, terminal output delivery, hard-stop TP 합의와 waiter 종료를 둔다. 세 domain의 합이 제출 request 수와 맞지 않으면 lost request다.

### stop 상태는 TP 전체에서 단조 증가하지만 teardown은 별도 barrier다

`BackgroundThreadStatus`는 `DONT_STOP < FLUSH_AND_STOP < HARD_STOP < STOPPED`의 정수 상태를 쓴다. `request_stop()`은 flush와 hard만 허용하고 lock 아래 local·TP 값의 최대를 취한다. TP collective 결과를 반영하는 `update_with_tp_status()`는 이전 TP 상태보다 낮아지는 값을 거절하고 local과 다시 MAX 결합한다. 한 rank가 hard stop을 요청한 뒤 다른 rank의 flush 신호가 이를 약화시키지 못하는 단조성이다.

그러나 stop 합의와 resource teardown은 같은 사건이 아니다. `stop(block=False)`는 신호를 보낸 직후 processor를 `None`으로 만들거나 persistent cache로 저장하고 원 attention backend를 복원한다. `block=True, timeout=N`도 join 뒤 thread가 아직 살아 있다는 warning만 낸 후 같은 정리를 계속한다. worker가 아직 `_generation_step()`에 들어갈 수 있다면 `self.batch_processor` 제거 또는 backend 복원과 경합할 수 있다. 정적 소스는 이 가능성을 보이며 실제 발생 빈도를 주장하지 않는다.

안전한 teardown invariant는 `stop()` 반환이 아니라 generation thread가 살아 있지 않고, async pair의 마지막 update와 stream/event synchronization이 끝났으며, processor와 attention backend의 owner 전환이 그 뒤에 일어났다는 순서다. `destroy()`는 실행 중이면 timeout 없는 blocking stop 뒤 CPU communication group을 폐기하지만 별도 destroyed flag가 없다. docstring은 restart 불가라고 해도 `start()`에 이를 막는 guard가 없다. 문서 계약과 기계적 invariant를 구분한다.

context manager의 보호 범위도 정확히 읽는다. init, optional warm-up과 `start()`는 `try` 바깥이고, `yield manager` 이후만 `finally`의 stop/destroy가 보호한다. yield 이전의 동기 실패에는 같은 cleanup이 자동 적용되지 않는다. background thread 내부 초기화 실패는 `start()` 호출자의 동기 예외가 아니라 fatal state와 output 경로로 나타난다.

기존 결함 주입 목록에 네 가지를 더한다: warm-up capture 하나를 실패시켜 `warmed_up`과 graph coverage가 갈라지는지, persistent reuse에 다른 config를 넘겨 effective policy가 고정되는지, processor 생성 전 오류에서 queued waiter가 종료되는지, join timeout 직후 processor/backend state와 thread liveness가 경합하는지 확인한다. 이들은 성능 benchmark가 아니라 lifetime 계약 검증이다.

### cache capacity는 KV byte 하나가 아니라 네 항의 방정식이다

`PagedAttentionMemoryHandler`는 `max_batch_tokens=M`과 cache page 수 `N`을 따로 정하지 않는다. LM head에서 hidden activation과 항상 FP32인 `[M,V]` logits를 peak로 잡고, attention에서는 Q와 신규 K/V의 M 항 및 기존 cache 전체 read의 N 항을 잡는다. 여기에 persistent KV, bulk input/output, block table과 read/write index가 더해진다. explicit eager/SDPA mask는 `[1,1,M,N+M]`이므로 `M·N`과 `M²` 항을 만든다. async batching은 host/device IO tensor 두 벌 때문에 해당 계수의 multiplier가 2다.

둘 다 미지정이면 cache의 10%를 한 batch가 채운다는 관계로 quadratic equation을 풀어 VRAM upper bound를 구하고 기본 batch token 값과 최소 한도를 적용한다. 하나만 지정되면 나머지를 linear 또는 quadratic solve한다. 마지막에는 LM-head와 attention peak별 해의 보수적인 최소를 취하고 footprint가 available memory를 넘거나 값이 0 이하이면 실패한다.

| 항 | 대표 storage | 커지는 조건 | 진단 값 |
|---|---|---|---|
| `M` | logits, Q/K/V, bulk IO, block/write index | vocabulary·batch token·async IO 증가 | dtype별 bytes/token |
| `N` | persistent KV, read index, attention cache read | context capacity·group 증가 | pages, block size, KV dtype |
| `M·N` | explicit attention mask | paged eager/SDPA와 큰 batch/cache 병존 | mask backend와 group 수 |
| `M²` | mask의 current-batch 구간 | 큰 prefill chunk | max batch tokens² |

OOM issue에는 최종 두 숫자만 쓰지 않는다. available memory 산식, `max_memory_percent`, activation peak별 coefficient, async multiplier와 선택 backend의 mask 필요 여부를 보존한다. 그래야 “KV block을 줄여야 하는가, prefill M을 줄여야 하는가, mask-free backend가 필요한가”를 구분할 수 있다.

## 24.9 static IO·control plane·graph lifetime의 소유자를 검산한다

`ContinuousBatchingIOs._setup_static_tensors`는 작은 int32 입력을 128-byte aligned bulk tensor에 묶고 그 view를 input IDs, positions, cumulative lengths, logits indices와 carry-over IDs로 나눈다. explicit mask가 필요하면 attention group마다 `[1,1,M,N+M]`, block table은 decode fast-path capacity, read/write index는 각각 `N+M`과 M 크기로 선할당한다. CPU staging이면서 accelerator가 있으면 pinned memory를 요청한다.

따라서 profiler의 런타임 batch가 작다고 static reserved memory도 작다고 가정하지 않는다. allocation 표에는 logical used extent와 physical capacity, host/device, pinned 여부, async pair 개수, graph key를 함께 둔다. capacity solver의 polynomial coefficient와 이 tensor inventory가 byte 단위로 맞는지 검산한다.

### 24.9.1 TP control-plane은 NCCL model collective와 다른 ordering domain이다

TP continuous batching에서는 driver rank만 request와 cancellation queue를 drain한다. 먼저 payload size와 stop status 두 정수를 synchronous CPU MAX all-reduce로 합의하고, payload가 있으면 Gloo `broadcast_object_list`로 Python request state를 전파한다. 그 뒤 모든 rank가 같은 순서로 logit processor validation, waiting admission과 cancellation flag를 적용한다.

이 경로가 느리거나 한 rank에서 예외 의미가 달라지면 CUDA kernel은 정상이어도 rank별 scheduler state가 갈라질 수 있다. issue 표에는 driver 여부, local queue counts, reduced payload/stop 값, object broadcast latency, rank별 request ordering digest를 둔다. GPU NCCL trace만 보고 이 control-plane stall을 찾으려 하지 않는다.

### 24.9.2 CUDA graph의 capture lifetime과 해제 lifetime을 분리한다

`CudaGraphBuffer`는 shape key별 graph를 저장하지만 explicit reset method를 제공하지 않고 destructor에서 dictionary를 비우며 각 `graph.reset()`을 호출한다. 같은 key에 `set_graph`를 다시 호출할 때도 기존 graph를 먼저 reset하는 명시 경로가 없다. Python object 교체가 곧 CUDA graph resource 회수 완료라는 보장은 이 wrapper에 없다.

session reset이 graph를 재사용하는 것과 manager/object 파괴 때 graph를 해제하는 것은 다른 사건이다. 장기 worker에서는 graph key 수, replacement 횟수, buffer owner identity, destructor 도달과 CUDA reserved bytes를 함께 기록한다. teardown 완료 조건에 generation thread 종료와 stream synchronization뿐 아니라 graph buffer의 명시적 owner 해제를 포함한다.

## 24.10 interface 경계와 25장 handoff를 닫는다

Transformers CLI serving은 model manager를 직접 HTTP handler에 노출하지 않는다. [`CBGenerateManager`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cli/serving/utils.py#L809-L977)가 continuous API의 token-level result와 asyncio·text streaming 사이를 번역한다. 첫 CB 요청에서 `init_cb`는 model의 `init_continuous_batching`을 호출하고 manager를 start한다. 이미 `_cb`가 있으면 돌아오므로 첫 요청의 generation config가 shared sampling 설정을 정하는 구조와 TODO를 반드시 읽어야 한다.

이후 요청이 다른 temperature나 top-p를 보냈다고 classic generate처럼 완전히 독립된 per-request config가 되는 것은 이 고정 revision의 CLI 계약이 아니다.

streaming 경로는 먼저 `add_request(..., streaming=True)`로 request ID를 받고 `CBStreamer`와 callback을
연결한다. callback은 `GenerationOutput` token을 tokenizer decoder에 넣고 text queue로 전달하며,
error가 있으면 `_StreamError`를 넣고 stream을 닫는다. HTTP client가 중단되면 streamer cancellation이
manager의 `cancel_request`로 이어져야 한다. application queue를 닫는 것과 GPU request cleanup은
별도 사건이므로 두 경계를 로그에서 구분한다.

non-streaming 경로는 더 중요한 race를 드러낸다. 짧은 R7이 `add_request` 직후 한 step에 끝날 수
있으므로 result future 또는 callback을 먼저 등록하고 request를 admission한다. 순서를 거꾸로 하면
worker가 terminal output을 deliver한 뒤 consumer가 생겨 영원히 기다릴 수 있다. source 주석의
“Register future BEFORE add_request”는 스타일 선호가 아니라 completion-before-subscription race를
막는 happens-before다.

CLI의 manager 선택도 lifetime을 바꾼다. `GenerateManager`는 persistent inference thread에서 `model.generate()`를 실행한다.

`CBGenerateManager`는 하나의 continuous manager와 background worker를 공유한다. [`GenerationState`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/cli/serving/utils.py#L983-L1056)는 continuous batching flag, model capability와 LLM modality를 보고 CB 사용 여부를 정한다. model ID가 바뀌면 기존 CB manager를 stop하고 새 wrapper를 만든다. model object와 model ID mapping이 바뀌는 동안 old worker가 완전히 join됐는지, stop timeout을 어떤 의미로 쓰는지는 서비스 teardown 감사의 대상이다.

server health는 HTTP process가 살아 있다는 뜻과 다르다. `_check_alive`는 manager의 `fatal_error`가
있으면 새 요청을 빠르게 실패시켜 죽은 worker의 queue에 넣지 않는다. server 쪽 CB dead handler가
이를 응답으로 바꾼다. background critical error 뒤 이미 queued된 요청, active request와 새 request가
각각 어떤 terminal 결과를 받는지 나눠 본다. readiness probe는 endpoint의 TCP 성공만이 아니라
CB worker fatal state와 manager 초기화 상태를 반영해야 한다.

### 24.10.1 classic generate에서 가능하던 기능이 그대로 넘어오지 않는 경계

continuous manager를 `generate()`의 drop-in concurrent mode로 소개하면 가장 위험한 오해가 생긴다.
classic path는 generation mode에 따라 greedy/sample, beam 계열, contrastive, assisted generation과
custom generation을 고르고, ordered logits processor와 stopping criteria, streamer를 한 호출의
config로 조립한다. CB path는 `RequestState`, paged cache, batch-aware processor와 한 token output
계약에 맞춘 별도 구현이다. source에 명시적으로 연결되지 않은 classic feature를 지원한다고
추론하지 않는다.

첫째, beam search는 한 외부 요청이 여러 hypothesis row로 fork되고 score·parent ancestry를 유지해야
한다. CB cache manager에 block fork 기능이 있다는 사실만으로 classic beam search가 구현됐다는
뜻은 아니다. fork는 cache/request cloning primitive일 수 있지만 beam scorer, hypothesis pruning,
length penalty와 finalization 호출 사슬이 연결되어야 beam feature다. public add-request 인자가
`max_new_tokens`, EOS, streaming과 processor kwargs 중심이라면 classic `num_beams`를 넣을 자리가
실제로 있는지 확인한다.

둘째, arbitrary stopping callable은 background thread에서 user Python code를 실행하고 request별
history·state를 보존해야 한다. CB의 `update_and_check_completion`이 EOS와 length를 판정한다는 사실은
모든 classic `StoppingCriteriaList`를 지원한다는 뜻이 아니다. 문자열 stop sequence도 token EOS와
같지 않다. CLI request schema가 이를 받아 어떤 processor/state로 컴파일하는지 source 연결이 없으면
지원 범위 밖으로 기록한다.

셋째, per-request sampling 다양성은 manager construction 때 snapshot한 generation config와
request-level kwargs를 분리해 본다. CLI docstring과 TODO는 shared temperature, top-p, do-sample을
첫 init config에서 가져오는 경계를 밝힌다. 서로 다른 tenant가 같은 manager에서 서로 다른 sampling
policy를 요청할 때 무시, 거부 또는 제한된 override 중 무엇이 실제인지 handler의 config builder와
`add_request` signature를 대조한다. classic generate처럼 호출별 `GenerationConfig`가 전체 loop를
소유한다고 가정하지 않는다.

넷째, multimodal 입력은 tensor input IDs 외에 pixel values, audio features, encoder outputs와 modality별
cache lifetime을 요구한다. CLI의 `use_continuous_batching`이 LLM modality를 조건으로 보는 것은 중요한
capability gate다. model class에 method가 있다는 사실만으로 모든 modality가 CB를 쓸 수 있는 것은
아니다. handler가 CB일 때 tokenizer 결과를 Python token list 형태로 만들고 classic일 때 tensor를
만드는 분기도 input ownership이 다름을 보여 준다.

다섯째, custom streamer와 return-dict surface를 구분한다. classic generate는 caller가 streamer를
직접 넘기고 scores, attentions, hidden states와 cache를 포함한 다양한 반환을 요청할 수 있다. CLI CB
wrapper는 `GenerationOutput` token/logprob/error를 text streamer로 바꾸는 제한된 surface다. model
runner가 내부 logits를 가진다는 사실이 arbitrary hidden-state 반환 API를 의미하지 않는다. static
buffer와 graph 계약에 새 output을 추가하면 memory sizing과 capture topology도 바뀐다.

이 경계들은 단점 목록이 아니라 두 실행기의 목적 차이다. classic generate는 한 호출의 표현력과
generation algorithm 조합을 제공한다. continuous manager는 request admission·paged cache·batch rebuild와
overlap을 위해 더 좁은 상태를 오래 사는 session 안에서 관리한다. 기능을 요구할 때는 “generate에서
되었다”가 아니라 continuous public API→state field→processor/runner→output까지 연결된 source가
있는지를 증명한다.

### 24.10.2 R7의 끝은 finish, output, free 세 사건이 모두 닫힐 때다

R7이 네 번째 token에서 length limit를 만났다고 하자. device buffer에 token이 쓰인 시점은 compute
완료다. D2H와 `update_and_check_completion`이 끝나 state가 terminal이 되는 시점은 logical finish다.
router가 callback/future에 terminal output을 전달한 시점은 consumer visibility다. scheduler가
active map에서 R7을 제거하고 cache refcount를 낮추며 private block을 free한 시점은 resource cleanup이다.
이 네 시점을 한 `request_done` timestamp로 합치면 tail latency와 leak을 동시에 설명할 수 없다.

prefix block은 R7 종료 뒤에도 다른 request refcount 때문에 남을 수 있다. graph와 static IO buffer는
request보다 오래 사는 manager session 자원이라 R7 finish 때 free되지 않는 것이 정상이다. 반대로
R7 전용 output handler와 waiting/active entry, private cache mapping은 유한하게 사라져야 한다.
owner별 기대 수명을 구분해야 reserved VRAM을 leak으로 오판하지 않고 실제 ghost request를 놓치지
않는다.

정상 종료의 증거는 다음 사건 사슬이다.

```text
sampled token(device)
→ D2H 완료
→ R7 state commit와 terminal 판정
→ terminal GenerationOutput 전달
→ scheduler request 제거
→ CPU/GPU private cache ownership 해제
→ 다음 request가 새 generation으로 block을 재사용
```

취소는 sampled token이 없거나 in-flight output을 버리는 갈래를 가질 수 있고, 실패는 terminal
error output을 전달해야 한다. 세 갈래 모두 마지막에는 request map과 cache owner가 사라져야 하지만
외부에 보이는 token semantics는 다르다. 이 구분이 request lifecycle의 결론이다.

### 24.10.3 다음 장으로 넘기는 상태

이 장은 scheduler가 어떤 fairness 정책으로 R7과 R8을 골라야 하는지 결정하지 않았다. 선택된 request가
queue에서 state가 되고, batch tensor와 paged cache 좌표로 번역되며, model output이 다시 같은
request에 commit되고 free되는 ownership 사슬을 닫았다. 다음 scheduling 장에서는 이 고정된 state와
resource cost를 입력으로 token budget, prefill/decode 우선순위와 starvation을 다룬다.

독자가 장애를 만났을 때 첫 질문은 “continuous batching이 느린가”가 아니다. R7이 input queue,
waiting, active, in-flight device row, future update, terminal output, freed cache 중 어디에 있는지 묻는다.
그 위치를 특정하면 queue stall, cache pressure, graph fallback, async alignment와 consumer race가 서로
다른 조사로 갈라진다. 오래 사는 서버를 이해한다는 것은 결국 각 state의 owner와 commit 경계를
말할 수 있다는 뜻이다.

마지막으로 성능 수치도 같은 lifetime 위에 놓는다. admission부터 첫 schedule까지는 queue 대기,
첫 schedule부터 첫 device output까지는 prefill·runner 비용, device output부터 첫 전달까지는
D2H·update·router 비용이다. ITL에는 다음 schedule 대기, cache allocation, model step과 commit이
반복해서 들어간다. 평균 kernel 시간만으로 client latency를 설명할 수 없는 이유다. graph replay가
빨라도 request가 cache pressure로 waiting에 머물면 TTFT는 줄지 않고, async overlap이 성공해도
callback이 generation thread를 막으면 전달 간격이 늘 수 있다.

그래서 trace의 request span과 CUDA range를 request ID 하나로 무리하게 직접 연결하기보다 batch
epoch와 row mapping을 중간 join key로 둔다. `R7 → batch N row 0 → graph key G → device output row 0
→ update epoch N → R7 output` 사슬이 있어야 GPU 시간과 사용자 시간을 정확히 잇는다. 이 사슬은
관측 도구의 장식이 아니라 continuous manager의 동적 batch를 해석하는 최소 데이터 모델이다.

<!-- reader-sources:begin -->

### 24.10.4 이 장의 원문 확인 지점

- [고정 소스: Transformers v5.15.1](https://github.com/huggingface/transformers/tree/550d7b3834670483a4df436541272c055dc364bf)

`continuous_api.py`에서 manager 생성·start·admission·generation loop·stop을 잇고, 같은 파일의
processor에서 prepare→compute→update transaction을 확인한다. `requests.py`는 status, 다섯 길이와
terminal output의 의미를 고정한다. `scheduler.py`에서는 이 장에 필요한 waiting/active ownership과
allocation 호출 경계까지만 읽고 정책 평가는 다음 장에 남긴다.

`cache_manager.py`는 block refcount, complete marking, full/sliding allocator의 read/write index를,
`cache.py`는 실제 paged tensor group을 소유한다. `input_outputs.py`는 static capacity, flat row packing,
cumulative length, IO pair와 CUDA event를 연결한다. `model_runner.py`는 forward 선택, logits row,
processor·sampling과 graph capture를 닫는다. `offloading_manager.py`는 D2H/H2D byte 이동뿐 아니라
request position rollback과 restore ownership을 보여 준다.

CLI를 조사할 때는 `cli/serving/utils.py`의 `CBGenerateManager`, `CBStreamer`, `GenerationState`를
보고 completion·chat-completion·response handler가 CB capability와 tokenization 형태를 어떻게
선택하는지 이어 읽는다. source link는 symbol 이름 검색의 대체물이 아니다. 새 revision에서는
field와 호출 순서가 달라질 수 있으므로 이 장의 줄 번호를 그대로 적용하지 말고 pinned commit과
새 commit의 owner transition을 diff한다.

실행하지 않은 정적 검토는 가능한 경로와 명시된 한계를 증명한다. 실제 배포가 async, graph,
prefix sharing 또는 특정 paged backend를 선택했다는 사실은 effective config, selected forward와
runtime trace로 별도 확인해야 한다. 이 증거 수준을 구분해야 source 설명이 운영 환경에 대한
근거 없는 단정으로 변하지 않는다.

<!-- reader-sources:end -->

## 24.11 request table과 physical batch row를 세 요청으로 분리한다

classic `generate`에서는 함수 인자 batch가 호출 동안 거의 고정된다. continuous manager에서는 request A가 decode 중일 때 B가 끝나고 C가 입장한다. 그러므로 request identity와 GPU tensor row가 같은 번호라고 믿으면 안 된다. request table은 장기 생존하는 논리 상태이고, batch row는 이번 iteration을 위한 임시 실행 좌표다.

고정 소스의 `RequestState`는 요청 ID, 입력·출력 토큰, 상태와 길이 정보를 소유한다. `update_and_check_completion`은 새 토큰과 로그 확률을 반영한 뒤 완료 여부를 판단한다.

매니저의 `add_request`는 요청 상태를 만들어 입력 큐로 넘기고, `cancel_request`는 ID를 취소 큐에 넣는다. 프로세서는 드라이버에서 새 요청과 취소를 꺼낸 뒤 텐서 병렬 각 랭크에 방송하고 스케줄러에 반영한다.

이 연결은 [Transformers request state와 token commit](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/requests.py#L123-L269)과 [manager admission과 cancel](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/continuous_api.py#L763-L862)에서 확인한다.

### A·B·C 네 iteration 원장

capacity가 두 row이고 block size가 4 token이라고 하자. A prompt 길이는 6, B는 3, C는 5다. iteration 0에서 waiting table은 A,B,C이고 scheduler가 A와 B를 active로 고른다. logical request→physical row mapping은 `{A:0,B:1}`이다. A는 두 block, B는 한 block이 필요하다. C는 capacity 때문에 waiting에 남는다.

prefill 뒤 A가 token 10, B가 EOS를 생성했다고 하자. update 단계에서 A의 generated length는 1, logical total length는 7이다. B는 generated length 1, total 4이고 terminal이 된다. B의 physical row 1과 cache block refcount는 cleanup 대상이다. 그러나 GPU kernel 완료 전에 row와 block을 즉시 C에 주면 늦은 write가 C state를 오염할 수 있다. completion event와 allocator release 순서를 같이 본다.

iteration 1 preparation은 B를 제거하고 C를 admit한다. compact 방식이 A를 row 0에 유지하고 C를 row 1에 놓으면 mapping은 `{A:0,C:1}`이다. 다른 구현은 active list order에 따라 `{C:0,A:1}`을 만들 수도 있다. 정답은 특정 row가 아니라 모든 per-request tensor와 metadata가 같은 permutation을 적용하는 것이다.

iteration 1에서 A는 decode token 하나만 model input으로 내고 C는 prompt 다섯 token을 prefill한다. 하나의 flat batch에 decode와 prefill row가 섞일 수 있으며 cumulative sequence lengths와 cache slot mapping이 각 segment를 설명한다. `[batch,seq]` rectangular mental model만으로는 부족하다.

IO preparation은 selected states에서 flat token, positions, block mapping을 만든다. [Transformers continuous batch tensor preparation](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/input_outputs.py#L319-L418)

iteration 2 전에 A가 streaming client disconnect로 취소됐다고 하자. cancel queue의 ID A가 scheduler state를 cancellation으로 표시하고 `prepare_next_batch`가 cancelled states를 clear한다. CPU-offloaded cache가 있다면 별도 cleanup도 수행한다. C가 row 0으로 compact될 수 있다.

C의 generator, processor state, grammar, block table, prompt/generated lengths와 output route가 모두 새 row 0을 가리켜야 한다. [Transformers cancellation 수집과 cleanup](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/continuous_api.py#L309-L377)

iteration 3에서 D가 들어온다면 A의 request ID나 incarnation과 무관한 새 state다. string ID를 재사용할 수 있는 API라면 generation number를 추가하거나 완료 table에서 중복을 거부해야 한다. 늦은 A output이 D route에 전달되지 않게 한다. physical row 재사용과 external request ID 재사용은 별도의 위험이다.

### classic batch와 같은 점, 다른 점

같은 점은 각 request의 logical token sequence다. prompt 뒤 accepted token이 append되고, next position과 cache가 진행하며 stopping condition이 terminal을 만든다. processor와 sampler의 의미도 request별이어야 한다. classic fixture의 step record를 request table row로 옮길 수 있다.

다른 점은 ownership lifetime이다. classic invocation은 caller가 batch 전체와 cache kwargs를 소유한다. continuous manager는 queue, scheduler, block manager, output router와 background loop가 서로 다른 lifetime을 가진다. request가 끝나도 manager와 static IO/CUDA graph는 산다. batch tensor는 매 iteration 다시 packing되며 membership이 바뀐다.

classic에서 먼저 끝난 row가 pad로 남아 wasted compute를 만들 수 있지만 continuous manager는 terminal row를 제거하고 waiting request로 capacity를 채울 수 있다. 대신 compaction permutation과 cache block mapping, async output routing이라는 correctness 비용을 낸다. “continuous가 빠르다”는 설명은 saved finished-row work와 admission opportunity를 계산하고 새 관리 비용을 함께 본 뒤에만 성립한다.

### request table 최소 불변식

request ID는 정확히 하나의 live `RequestState`와 output route를 가리킨다. state의 prompt length, generated length, computed/cache frontier와 visible/terminal frontier는 구분된다. physical batch row는 iteration generation과 함께만 유효하다. cache block list와 slot mapping은 logical request position을 보존한다. terminal 또는 cancel은 정확히 한 번 output/cleanup transaction을 시작한다.

관측 record에는 `(iteration, request_id, request_incarnation, status, physical_row, prompt_len, generated_len, computed_len, block_ids, output_cursor)`를 둔다. 모든 iteration에 전체 table을 production 로그로 남기지 않고 sampled trace와 bounded counts를 사용한다. queue length, active/waiting/cancelled/finished count는 metric으로 둔다.

## 24.12 compaction은 tensor 하나가 아니라 request 상태 묶음의 permutation이다

A,B,C가 rows 0,1,2에 있고 B가 끝났다고 하자. compact 결과가 A,C rows 0,1이면 permutation은 old `[0,2]`다. token IDs와 positions만 `index_select([0,2])`하고 sampling temperature가 old `[0,1]`을 사용하면 C가 B의 설정을 받는다. output token은 정상 vocabulary 범위이고 예외도 없어 품질 이상으로만 보인다.

permutation 대상은 current input segments, positions, sequence lengths, block table/slot mapping, logits row, sampling params, generator, grammar/penalty state, request ID, output route와 async IO offsets다. 일부는 dense tensor, 일부는 Python list, 일부는 device-resident static buffer다. 같은 permutation이라는 의미를 각 representation에 맞춰 구현해야 한다.

### 수치 fixture로 wrong-row를 드러낸다

A temperature=0 greedy, B temperature=1.0, C temperature=0.5라고 하자. 세 request raw logits를 의도적으로 `[5,4]`, `[0,0]`, `[1,0]`으로 둔다. B가 끝난 뒤 C가 row 1로 이동했는데 temperature tensor가 `[A,B]`로 남으면 C는 0.5 대신 1.0을 쓴다. uniform 0.3에서 두 분포가 우연히 같은 token을 낼 수 있으므로 processed probabilities와 effective temperature를 직접 비교한다.

cache block은 A `[10,11]`, B `[20]`, C `[30,31]`이라 하자. compaction 뒤 row 1 block table이 `[20]`로 남으면 C latest token은 B cache를 읽는다. C prompt length 5와 block size 4에는 두 block이 필요하므로 metadata assertion으로 잡을 수 있다. 길이가 우연히 같으면 numerical cache digest와 owner request를 비교해야 한다.

output route도 A queue QA, B QB, C QC로 구분한다. sampled token과 state update는 C에 맞는데 physical row 1을 QB에 deliver하면 끝난 B stream에 late token이 가거나 새 consumer가 없는 queue에 쌓인다. GPU correctness test는 통과하고 사용자 stream만 hang한다. commit tuple에 request identity를 포함해야 한다.

### static capacity buffer와 active prefix

CUDA graph를 위해 IO buffer가 max batch capacity로 고정돼도 active rows는 매 iteration 달라진다. inactive tail을 zero/pad하고 graph가 full capacity를 실행할 수 있다. physical address가 안정적이라는 사실과 logical row identity가 안정적이라는 사실은 다르다. static buffer row 1에 이번에는 C metadata를 덮어쓴다.

async H2D copy가 이전 iteration buffer를 읽는 동안 host가 같은 pinned row를 새 mapping으로 덮어쓰면 race가 난다. IO pair와 CUDA event가 producer/consumer happens-before를 보장해야 한다. double buffering은 address 세대를 분리하지만 어느 pair가 iteration n인지 기록해야 한다. source의 IO pair/event ownership과 `prepare_batch_tensors` 호출을 연결한다.

### compaction 사건: token은 C 것인데 logprob와 stream은 B 것

실제 incident narrative를 만들자. B가 step 4에서 EOS, C가 old row 2에서 row 1로 compact됐다. model input과 block table, sampler logits는 `[0,2]` permutation을 적용해 C token 77을 올바르게 선택했다. 그러나 logprob output buffer와 output router list는 old prefix `[0,1]`을 썼다. C stream은 token 없이 기다렸고 B stream에는 EOS 뒤 token 77과 C의 logprob가 도착했다.

first divergence는 model/cache가 아니다. device sampler record의 `(request=C,token=77)`까지 맞고, D2H result row를 request ID에 attach하는 update edge에서 B가 나온다. trace에는 preparation mapping, device row generation, D2H completion generation, update mapping과 route ID를 둔다. physical row 숫자만 기록하면 두 generation의 row 1을 구분하지 못한다.

수정은 batch descriptor에 immutable request incarnation vector와 iteration generation을 넣고 모든 output을 그 descriptor로 inverse map하는 것이다. current scheduler active list를 D2H 완료 시점에 다시 참조하면 그 사이 다음 batch가 준비돼 mapping이 달라질 수 있다. launch 때의 mapping snapshot을 completion까지 보존한다.

회귀 fixture는 서로 다른 logit/token/logprob와 output queue를 가진 A,B,C를 사용한다. middle row finish, first row cancel, simultaneous two finish, new request admit을 조합한다. token, logprob, terminal reason, cache owner와 stream route가 request identity를 따라가는지 본다. throughput soak만으로는 rare ordering을 재현하기 어렵기 때문에 event delay injection을 쓴다.

## 24.13 cache position과 block lifetime을 cancel·reuse까지 닫는다

continuous cache에는 적어도 세 길이가 있다. request가 논리적으로 가진 total token length, model이 이미 계산해 cache에 commit한 length, 현재 iteration이 계산할 query length다. block allocator는 physical capacity와 refcount를 가진다. prefix sharing이나 offload가 있으면 resident frontier와 shared complete block도 추가된다.

A prompt 6이 block size 4에서 block `[10,11]`을 쓴다고 하자. block 10은 네 token으로 complete, block 11은 두 token만 valid하다. first decode token position 6은 block 11 offset 2에 쓴다. 다음 token은 offset 3, 그다음 position 8은 새 block 12 offset 0이 필요하다. allocator는 model launch 전에 capacity를 확보하고 slot mapping을 만들어야 한다.

cache position을 physical offset `block_id×block_size+offset`과 혼동하지 않는다. position 6은 RoPE와 mask가 이해하는 logical coordinate고 physical slot은 allocator 선택에 따라 달라진다. prefix sharing이면 같은 logical prefix가 다른 request에서 shared block을 참조할 수 있다. source에서 request block list가 slot mapping으로 변환되는 producer를 찾는다.

### cancellation의 세 시점

첫 시점은 waiting이다. 아직 block을 할당하지 않았다면 request table과 output route를 terminal/cancel 처리하고 queue에서 제거하면 된다. tokenization이나 CPU state가 있으면 해제한다. 둘째는 scheduled but not launched다. block과 static IO row를 예약했지만 GPU가 읽기 전이므로 reservation rollback이 필요하다.

셋째는 in-flight다. kernel과 D2H가 해당 row와 cache slot을 사용할 수 있다. cancel flag가 왔다고 refcount를 즉시 0으로 만들고 새 request에 block을 주면 late write/read가 겹친다. output은 버릴 수 있어도 resource는 completion event까지 격리해야 한다. cancel semantic completion과 device resource completion을 구분한다.

고정 processor는 cancellation을 scheduler에 전달하고 `clear_cancelled_requests` 뒤 offloaded cache를 정리한다. block manager의 `decrease_ref_count`는 shared ownership을 하나씩 줄인다. refcount 0에서 free list로 돌아가는 정확한 조건과 complete/shared block policy를 본다. [Transformers block refcount 감소](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/cache_manager.py#L170-L218)

### late-write cache reuse incident

A가 row 0, block 40 offset 3에 decode write를 launch했다. client disconnect로 A가 cancel됐고 host scheduler가 block 40을 free list에 즉시 반환했다. C가 같은 iteration overlap에서 block 40 offset 3을 새 prompt tail로 할당받았다. A kernel이 늦게 완료되며 C key/value를 덮었다. C first prefill output은 우연히 정상이고 다음 decode attention부터 이상해졌다.

allocator metric은 refcount가 정상적으로 0→1로 바뀌어 leak이 없다. 단순 memory audit는 통과한다. first divergence는 C cache write 직후가 아니라 A late completion 뒤 C read다. block owner generation과 write event를 trace에 넣어야 한다. `(block_id,allocation_generation,request_incarnation,logical_range,last_write_event)`를 debug record로 둔다.

수정은 in-flight generation completion 전 block을 재사용하지 않는 deferred free 또는 event fencing이다. cancellation output은 즉시 사용자에게 전달할 수 있지만 physical reclaim은 늦출 수 있다. 이 지연이 capacity를 잠시 줄이므로 cancelled-inflight blocks와 reclaim latency를 metric으로 둔다.

회귀는 A kernel completion을 인위적으로 늦추고 C allocation을 압박한다. C cache digest와 reference logits, block generation을 검사한다. cancellation storm에서 deferred free가 무한히 쌓이지 않고 device completion 뒤 회수되는지 soak한다. driver와 TP rank가 block lifecycle 합의를 공유하는지도 본다.

### prefix sharing refcount와 partial tail

A와 B가 complete prefix block 10을 공유하면 refcount 2다. A cancel은 1로 줄이고 B가 계속 읽을 수 있어야 한다. partial tail block은 request-private일 수 있다. complete marking 전에 shared index에 넣으면 아직 쓰이는 block을 다른 request가 읽는다. block complete 상태와 valid token extent를 cache key/lifetime에 포함한다.

context shift나 sliding cache에서는 logical absolute position이 늘어도 resident window length는 고정될 수 있다. request total length와 cache physical used block 수가 같지 않다. eviction이 position mapping과 mask를 어떻게 갱신하는지 해당 backend에서 확인한다. 이 장은 general invariant를 제시하고 실행하지 않은 backend를 동일하다고 단정하지 않는다.

**continuous manager 통제 실험.** 실험 A는 서로 다른 길이 세 요청의 row compaction 뒤 request→row, cache position과 output owner가 함께 permutation되는지 본다. 실험 B는 한 요청 cancel과 같은 tick의 batch append를 겹쳐 freed row가 재사용되기 전에 generation fence가 서는지 확인한다. 실험 C는 empty active batch와 새 admission race를 주입한다. 실험 D는 slow stream consumer가 manager scheduler를 막는지 queue depth와 tick latency로 반증한다.

## 24.14 streaming cancellation을 manager transaction과 classic 호출에 대조한다

continuous manager에서 생성 token은 model output에서 바로 사용자 socket으로 가지 않는다. GPU result가 request state에 commit되고 `GenerationOutput`이 output router로 전달되며 streamer/handler가 consume한다. client disconnect는 반대 방향으로 cancel queue를 지나 scheduler와 cache cleanup에 도달한다. 두 방향이 비동기라 late output과 cancel이 교차한다.

### generated·delivered·cancelled frontier

A가 token 5개를 generation state에 commit했고 output router가 4개를 queue에 넣었으며 client가 3개를 읽은 시점에 disconnect했다고 하자. generated=5, routed=4, delivered=3이다. cancel 뒤 token 5가 late route될 수 있다. API가 partial output을 보존하는지 버리는지와 무관하게 counters를 섞지 않는다.

classic threaded streamer도 producer/consumer frontier가 있지만 invocation caller가 thread와 streamer를 감싼다. continuous manager에서는 background loop가 여러 request output을 route하고 한 client cancellation이 다른 request를 멈추면 안 된다. output queue와 cancel queue의 request identity, terminal exactly-once가 핵심이다.

### cancel과 natural finish race

step n token이 EOS이고 GPU completion 직전 cancel이 도착했다고 하자. scheduler가 cancel을 먼저 처리하면 finish reason cancelled, update가 먼저 처리하면 stop/EOS가 될 수 있다. 제품 계약이 어느 event를 우선하는지 정하고 state transition을 atomic하게 만든다. terminal을 두 번 deliver하거나 cache를 두 번 free하면 안 된다.

state machine은 live→finishing→terminal 같은 compare-and-set 의미를 가질 수 있다. cancel과 natural finish 중 승자가 terminal reason을 publish하고 cleanup ownership을 가진다. 패자는 이미 terminal임을 보고 no-op한다. 실제 구현이 lock, queue order, single background thread로 직렬화하는지 source를 읽는다. “Python이라 안전”이라고 가정하지 않는다. GPU/D2H callback과 handler thread가 있다.

### stream cancel incident: output은 멈췄지만 cache가 남는다

HTTP handler가 disconnect를 감지해 local async iterator만 닫고 manager `cancel_request`를 호출하지 않았다. output router는 consumer 없는 queue에 token을 계속 넣고 scheduler는 request를 active로 유지했다. 사용자 관점에서는 취소 성공이지만 GPU와 cache block은 max_new_tokens까지 사용됐다. active request와 orphan output queue가 누적돼 admission이 막혔다.

증거는 disconnect count 증가와 cancel queue 증가가 일치하지 않고, request table active count와 cache used blocks가 연결 종료 뒤에도 유지되는 것이다. source owner는 HTTP wrapper와 manager cancel edge다. model forward나 scheduler policy를 먼저 고치지 않는다.

수정은 handler `finally`에서 request incarnation이 terminal이 아니면 cancel을 enqueue하고, output consumer 종료와 manager terminal acknowledgment를 연결하는 것이다. 네트워크 close를 기다리느라 generation loop를 block하지 않는다. cancel enqueue 실패나 manager stop 중이면 별도 cleanup policy가 필요하다.

회귀 fixture는 client가 첫 token 전, 중간 token 뒤, EOS와 동시에 disconnect하는 세 경우다. cancel queue, scheduler removal, output terminal, cache refcount, active/waiting counters가 eventually 정합하는지 본다. unrelated B request의 token cadence가 유지되는지도 검사한다.

### backpressure와 느린 consumer

output queue가 unbounded면 느린 client가 generated-routed gap과 memory를 키운다. bounded면 `deliver`가 block해 background generation loop 전체를 멈출 수 있다. per-request drop/cancel, 별도 routing worker, bounded buffering 전략 중 무엇인지 확인한다. 하나의 느린 request가 다른 active request의 GPU scheduling을 소유하지 않게 한다.

buffer byte는 token 수만으로 계산하지 않는다. token ID/logprob metadata와 decoded text, Python object overhead가 있다. B=100 slow streams가 각 1,000 token metadata를 쌓으면 상당한 host memory가 된다. queue depth, oldest age, routed-delivered gap을 metric으로 둔다. request ID는 high-cardinality label이 아니라 trace에 둔다.

**source audit와 rollout terminal**

source walk는 manager `add_request/cancel_request`, processor `prepare_next_batch/update_batch`, scheduler admission/cancel clear, IO tensor packing, model runner compute, output router deliver, block refcount를 하나의 transaction으로 잇는다. 특정 scheduler policy 평가는 뒤 장으로 미루되 state ownership은 닫는다.

numeric fixture는 A/B/C mapping, block list, position, temperature/generator와 output queue를 표로 둔다. iteration마다 old→new permutation을 명시한다. incident injection은 middle completion, in-flight cancel late write, output mapping delay, disconnected consumer를 포함한다.

correctness terminal은 token/cache/option/output route가 request incarnation을 따른다. lifecycle terminal은 waiting, scheduled, in-flight cancel과 natural finish race에서 terminal과 free가 정확히 한 번이다. performance terminal은 finished-row saved work, packing/graph/queue overhead와 deferred free capacity를 함께 본다. observability terminal은 request table과 iteration mapping, cache generation, output frontier를 민감 데이터 없이 복원한다.

rollback은 continuous path를 classic 호출로 바꾸는 것만으로 끝나지 않는다. in-flight manager request를 drain/cancel하고 cache와 output terminal을 닫은 뒤 새 요청을 classic path로 보낸다. 두 path의 finish reason, partial stream과 sampling semantics가 호환되는지 검증한다.

24장의 최종 invariant는 다음과 같다. **장기 생존 request table의 각 incarnation이 매 iteration 임시 batch row와 cache blocks, sampling state, output route에 정확히 매핑되고, completion·compaction·cancel 뒤에도 논리 token/cache/visible frontier가 같은 request를 가리키며 resource는 device completion 뒤 정확히 한 번 회수되어야 한다.**

이 문장을 세-request 원장, permutation incident, late-write cache reuse, stream cancel leak의 네 증거로 설명할 수 있으면 classic과 continuous의 차이가 “batch를 계속 채운다”는 구호를 넘어 실제 코드와 운영 상태로 이해된다.

### prepare→compute→update transaction을 함수별로 다시 걷는다

`prepare_next_batch`는 단순 tensor 생성 함수가 아니다. driver가 input/cancel/stop payload를 수집하고 TP rank에 전달한 뒤 cancelled state를 clear하며 scheduler가 다음 active set을 만든다. cache offload가 있다면 cancelled CPU state도 정리한다. 이 단계가 반환한 batch descriptor는 다음 compute와 update가 공유할 generation snapshot이어야 한다.

compute는 descriptor가 가리키는 flat input, position, cache mapping과 sampling metadata로 model runner를 호출한다. CUDA graph가 선택되면 capacity와 shape에 맞는 captured executable과 static buffers를 사용한다. eager fallback이 선택되면 같은 logical mapping을 다른 callable이 소비한다. graph/eager 전환에서 row order와 inactive tail semantics가 같아야 한다.

update는 GPU/D2H 결과를 descriptor의 request mapping으로 되돌린다. token과 logprob를 `RequestState`에 반영해 terminal을 판정하고, unfinished state는 다음 scheduler candidate로 남긴다. finished output은 router에 전달하고 cache refcount와 offload state를 정리한다. 이 순서는 “sampled token ID가 있다”에서 “request가 한 step 진행했다”로 바뀌는 commit 경계다. [Transformers batch preparation과 update](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/generation/continuous_batching/continuous_api.py#L360-L456)

transaction record에는 prepare generation, active request vector, cache reservation, compute launch/event, sampled output, update completion을 둔다. 다음 prepare가 이전 compute와 overlap된다면 mutable scheduler active list를 completion mapping으로 재사용하지 않는다. descriptor snapshot 또는 generation-indexed buffer가 필요하다.

### mixed prefill·decode 비용을 token budget으로 계산한다

A decode query 1 token, C prompt 5 token이 같은 iteration에 들어가면 forward token 수는 6이다. attention work는 동일하지 않다. A query는 cache length 7을 보고, C prefill 다섯 query는 causal prefix 안에서 서로 다른 key extent를 본다. scheduler의 token budget 6이 곧 동일 latency 6단위라는 뜻은 아니다. prefill과 decode composition을 metric으로 나눈다.

A,C 모두 hidden D=4096, BF16이면 input activation payload 자체는 `6×4096×2=49,152 byte`로 작다. 그러나 layer attention/MLP work와 C의 cache write 5 token이 따른다. 앞의 KV 128 KiB/token fixture라면 이 iteration은 A 128 KiB와 C 640 KiB, 총 768 KiB의 new KV logical payload를 만든다. block allocator는 C prompt가 block boundary를 넘는지에 따라 capacity를 미리 확보한다.

classic으로 A와 C를 별도 호출하면 launch와 weight traversal이 둘로 나뉜다. mixed batch는 같은 model step에서 모아 weight reuse와 GPU occupancy 기회를 만들지만 ragged metadata, packing, cache mapping과 latency coupling을 추가한다. C 긴 prefill이 A inter-token latency를 늘릴 수 있다. scheduler 정책은 throughput과 decode tail을 조절하며 다음 장의 주제지만, 이 장에서는 composition과 owner를 관측할 수 있어야 한다.

metric에는 scheduled requests, prefill/decode token, total query token, max/mean cache context, allocated block, packing/compute/update 시간을 둔다. 평균 batch size 하나로는 긴 prefill 한 개와 짧은 decode 여러 개를 구분하지 못한다. output queue backpressure 시간도 compute와 분리한다.

### admission 실패와 retry ownership

manager input queue에 넣었다고 GPU admission이 보장되는 것은 아니다. waiting state에서 timeout이나 queue capacity 제한, invalid config, cache capacity 부족으로 거절될 수 있다. API layer가 이미 request ID와 stream을 만들었다면 reject output과 terminal을 정확히 한 번 전달해야 한다. caller가 retry할 수 있는 오류인지, 같은 request ID를 재사용할 수 있는지 계약을 정한다.

cache가 일시적으로 부족한 waiting은 scheduler가 보류할 수 있지만 영구적으로 prompt가 capacity보다 크면 계속 기다려서는 안 된다. required blocks와 maximum capacity를 비교해 terminal reject를 낼 수 있다. waiting age, required blocks, rejection reason을 metric/trace에 둔다.

retry가 새 request incarnation을 만들면 이전 cancel/late output과 구분한다. 외부 ID가 같더라도 internal generation이 달라야 한다. idempotency는 API gateway 책임일 수 있지만 manager가 중복 live ID를 어떻게 다루는지 확인한다. silent overwrite는 request table과 output route를 잃는다.

### natural finish와 cancel race의 deterministic fixture

single background update thread가 모든 state transition을 직렬화한다고 가정해도 queue drain 순서가 결과를 결정한다. iteration n compute가 EOS를 만들었고 cancel ID가 다음 prepare queue에 있다. 구현이 update 후 prepare라면 EOS terminal이 먼저 확정되고 cancel은 이미 finished ID로 no-op할 수 있다. prepare가 overlap돼 cancel flag를 먼저 세우면 cancelled reason이 이길 수 있다.

fixture는 device completion event와 cancel enqueue 사이에 barrier를 둬 두 interleaving을 재현한다. 제품 계약이 “commit된 EOS 우선”이면 compute launch만으로는 부족하고 update commit 시점을 기준으로 한다. cancel이 commit 전에 도착하면 output discard, 이후면 natural finish 같은 rule을 정의한다. 어느 rule이든 terminal·cleanup exactly-once가 핵심이다.

검사 값은 request status transition sequence, terminal reason, committed token count, delivered token count, block refcount decrement 횟수와 output terminal count다. 두 interleaving에서 허용 결과 집합을 명시한다. 허용되지 않은 중복 free와 double terminal은 schedule에 관계없이 실패다.

**shutdown은 request cancel의 합이 아니다**

manager stop은 새 admission을 막고 background loop, TP rank, IO events와 graph lifetime을 닫는다. active request 각각을 cancel하는 것만으로 thread와 static buffers가 해제된다고 가정하지 않는다. stop signal broadcast, last async compute/update drain, output terminal, block/cache manager destroy의 순서를 읽는다.

graceful shutdown은 일정 deadline 안에서 active request를 완료하거나 취소한다. forced shutdown은 partial output과 error reason, device work fencing을 가진다. manager object를 destroy한 뒤 handler가 `cancel_request`를 호출하는 late callback도 안전하게 거부해야 한다.

shutdown fixture는 waiting-only, active in-flight, output backpressured, one TP rank error를 나눈다. join이 끝나고 background status, active/waiting table, cache refcount, output router task와 static buffer reference가 terminal인지 본다. GPU memory reserved가 allocator cache인지 live tensor reference인지 구분한다.

**classic 대비 migration 체크리스트를 인과 문장으로 쓴다**

classic wrapper에서 continuous로 바꾸면 request config를 invocation-local object에서 request table field로 옮긴다. full batch IDs에서 ragged/flat IO와 block mapping으로 바뀐다. cache kwargs에서 block manager ownership으로 바뀐다. caller return에서 output router와 per-request stream으로 바뀐다. thread cancellation에서 manager cancel queue와 scheduler cleanup으로 바뀐다.

각 이동에는 parity fixture가 필요하다. 동일 prompt/config의 first token과 short sequence가 classic reference와 맞는지 본다. cache position과 processor history를 step별로 비교한다. streaming chunk segmentation은 다를 수 있어 committed token sequence와 terminal reason, visible bytes 의미로 정규화한다. 성능 비교는 동일 output contract와 target concurrency에서 한다.

continuous가 지원하지 않는 classic generation mode나 custom processor가 있다면 admission에서 명시적으로 거부하거나 검증된 fallback을 사용한다. 조용히 option을 무시하면 effective config parity가 깨진다. fallback이 classic thread를 만들면 resource limit과 cancel bridge를 별도 설계한다.

**최종 독자 dossier**

첫 페이지는 request table snapshot과 iteration mapping이다. 둘째는 A/B/C permutation과 block/position 원장이다. 셋째는 prepare-compute-update event timeline이다. 넷째는 generated-routed-delivered와 terminal/free conservation이다. 다섯째는 incident first divergence와 수정 뒤 failure injection이다.

source anchor는 manager admission/cancel, processor preparation/update, scheduler selection, IO packing, model runner, request commit, block refcount, output deliver를 가리킨다. runtime evidence는 effective config, selected graph/eager path, iteration descriptor, event와 counters를 제공한다. source 가능성과 실제 선택을 구분한다.

배포 terminal은 동일 request가 batch reorder와 unrelated cancellation에도 같은 token semantics를 유지하고, compact row가 모든 state bundle에서 일치하며, cancelled in-flight resource가 completion 뒤 회수되고, 느린/disconnected stream이 다른 request를 막지 않는 것이다. target load에서 queue age와 decode tail, deferred free capacity도 예산 안이어야 한다.

이 dossier를 가진 독자는 “continuous batching이 느리다”를 prefill mix, packing, graph fallback, output backpressure 또는 cache pressure로 나눈다. “답이 가끔 다르다”를 request mapping, cache owner, sampling metadata, output route 중 최초 divergence로 나눈다. 이것이 운영 가능한 continuous manager 설명이다.

**마지막 20분 failure drill**

0~5분에는 A/B/C fixture에서 B middle completion을 강제한다. prepare descriptor의 old→new mapping과 모든 request-scoped field digest를 비교한다. token input, cache blocks, temperature/generator, output route 가운데 하나라도 다른 permutation이면 compute를 시작하기 전에 실패시킨다. assertion이 production fast path에 너무 비싸면 canary/debug mode에 둔다.

5~10분에는 A in-flight cancel과 block reuse 압박을 만든다. cancel terminal은 빠르게 나가도 block generation이 completion event 전 free list에 들어가지 않는지 본다. C가 capacity 때문에 잠시 waiting하는 것은 허용할 수 있지만 A late write가 C allocation과 같은 generation을 가져서는 안 된다. deferred free queue age와 count가 completion 뒤 0으로 수렴해야 한다.

10~15분에는 C output consumer를 느리게 하고 B 정상 stream cadence를 본다. output router가 C queue put에 block되어 generation loop 전체를 세우면 request 격리가 깨진다. 설계가 의도적으로 global backpressure를 택했다면 capacity와 timeout, cancel 정책을 문서화한다. token correctness와 routing liveness를 별도 terminal로 본다.

15~20분에는 EOS completion과 cancel race, manager shutdown을 겹친다. 허용된 하나의 finish reason과 terminal event, refcount 감소 한 번, output route close 한 번을 확인한다. stop/join 뒤 input/cancel queue의 late callback이 새 state를 만들지 않아야 한다. 이 drill은 평균 throughput benchmark가 보지 못하는 lifecycle edge를 짧게 재현한다.

**배열과 표가 엉키지 않게 쓰는 법**

본문 표의 행은 physical row가 아니라 request incarnation을 기준으로 정렬한다. iteration마다 별도 `physical_row` 열을 둔다. cache block과 current input segment, output cursor는 같은 행에 둬 독자가 수평으로 ownership을 검산할 수 있게 한다. active list 순서만 나열하면 compaction 전후 동일 request를 눈으로 잇기 어렵다.

event timeline은 prepare, H2D ready, compute launch, device complete, D2H ready, update commit, deliver, reclaim의 세로 순서를 쓴다. overlap이 있으면 iteration n과 n+1을 두 열로 놓는다. 화살표는 실제 happens-before만 그린다. host call order가 CUDA completion order를 보장한다고 임의로 연결하지 않는다.

metric 표는 count, byte, time, error를 나눈다. active/waiting/cancelled count, allocated/deferred block byte, packing/compute/update/route time, mapping-generation mismatch와 empty capacity error가 서로 다른 열이다. 하나의 “batch latency” 숫자로 상태를 압축하지 않는다.

incident 표는 symptom, first matching checkpoint, first divergent checkpoint, competing hypotheses, source owner, fix, regression, rollback을 가진다. “scheduler 문제” 같은 넓은 결론을 피한다. 예를 들어 token/logit은 C와 일치하지만 route ID가 B로 갈린다면 scheduler selection과 cache를 negative evidence로 내리고 update inverse mapping을 연다.

**완료 뒤에도 남기는 미검증 범위**

고정 소스는 FIFO와 prefill-first scheduler, paged/full/sliding cache, graph/eager 등 여러 가능성을 포함할 수 있다. 이 장의 fixture가 모든 조합에서 실행됐다고 주장하지 않는다. actual deployment의 selected scheduler, cache class, offload, TP와 graph capability를 effective record로 확인한다.

새 model architecture가 custom continuous input preparation이나 hybrid cache를 요구하면 common mapping 외의 state가 추가된다. recurrent state, encoder outputs, multimodal inputs도 permutation bundle에 포함돼야 한다. 지원 선언 전에 middle completion/cancel fixture를 그 state에 확장한다.

CUDA/library upgrade는 graph capture와 event/allocator behavior를 바꿀 수 있다. source call graph가 같아도 runtime path와 timing이 달라질 수 있으므로 failure drill과 capacity ledger를 재실행한다. 실행하지 못한 환경은 명시적으로 pending evidence로 남긴다.

이 정직한 범위 표시가 책의 설명을 약하게 만들지 않는다. 독자가 어떤 invariant는 일반적이고 어떤 성능·backend 결론은 특정 revision과 환경에 묶였는지 알게 한다. 다음 장의 llama.cpp slot manager도 같은 request identity, temporary execution row, cache frontier, output/cancel 질문으로 비교할 수 있다.

마지막 handoff record에는 continuous request A의 external ID와 incarnation, 마지막 committed token, logical position, cache block generation, visible output cursor와 terminal reason을 남긴다. 같은 prompt를 classic reference로 재실행할 때 비교할 semantic coordinate다. physical row, CUDA graph buffer 주소, scheduler iteration은 실행 증거이지만 cross-engine identity는 아니다.

25장에서는 이 record를 llama.cpp의 task ID와 server slot, `llama_batch` row, KV context와 HTTP response channel에 번역한다. 이름은 달라도 slot reuse와 context shift, prompt cache, abort의 위험은 같은 질문으로 읽을 수 있다. 어떤 object가 장기 request를 소유하고 어떤 배열이 한 decode iteration만 사는지 먼저 찾는다.

24장을 닫는 최종 질문은 하나다. “지금 보고 있는 row·block·output이 어느 request incarnation의 어느 logical step인가?” 준비, compute, update, stream, reclaim 어느 단계에서도 한 문장으로 답할 수 있어야 한다. 답할 수 없다면 성능 튜닝 전에 mapping과 generation observability를 보강한다.

수정 뒤에는 같은 세-request fixture를 eager와 graph, cache pressure와 정상 여유, 빠른 consumer와 disconnect 조건에서 반복한다. 결과뿐 아니라 terminal과 resource conservation도 같아야 한다. 이 검증 범위와 남은 예외를 release record에 고정한다.
