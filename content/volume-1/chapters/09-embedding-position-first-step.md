# 9장. embedding lookup과 위치 정보의 첫걸음

토크나이저가 `[17, 2048, 31]`이라는 정수 열을 만들었다고 하자. 이 숫자 자체에는 아직 방향도 거리도 문맥도 없다. 모델의 첫 계산은 각 정수를 거대한 표의 행 주소로 사용해 벡터를 꺼내는 일이다. 그런데 이 단순해 보이는 단계에서 vocab 행 수, dtype, device, tensor parallel shard, padding mask, position 좌표, multimodal placeholder 가운데 하나만 어긋나도 첫 decoder layer에 들어가기 전에 이미 결과가 틀린다.

이 장의 독자 문제는 “embedding이 단어의 의미를 담는다”는 비유를 배우는 것이 아니다. **token ID가 어느 parameter 행을 읽고, 어느 위치 좌표와 어느 modal feature를 만나 첫 hidden state가 되는지 소스와 shape로 증명하는 것**이다. embedding은 문맥화 이전의 시작점이다. 같은 ID는 같은 weight revision에서 같은 행을 읽지만, 위치와 주변 token이 다르면 뒤 layer의 hidden state는 달라진다. 이 비유의 한계부터 분명히 해야 “embedding 공간에서 가까우니 모델도 같은 판단을 한다”는 성급한 결론을 피할 수 있다.

## 9.1 정수 ID 하나가 벡터 한 행이 되는 순간

embedding weight를 `E ∈ R^(V×H)`라고 쓰자. `V`는 행 수, `H`는 hidden size다. input ID `i`의 embedding은 행 선택 `E[i,:]`다. batch input이 `[B,S]`이면 결과는 `[B,S,H]`다. 행렬곱처럼 보이게 하려면 ID `i`를 길이 `V`인 one-hot row `o_i`로 만들고 `o_i E`를 계산할 수 있지만 실제로 거대한 one-hot을 만드는 것은 낭비다. 구현은 gather 또는 row lookup을 사용한다.

작은 표를 손으로 읽어 보자.

```text
E = [[ 1.0,  0.0,  0.5],   # id 0
     [-1.0,  2.0,  0.0],   # id 1
     [ 0.2,  0.3, -0.4],   # id 2
     [ 4.0, -1.0,  1.0]]   # id 3

input_ids = [[3, 1, 3],
             [0, 2, 1]]
```

lookup 결과의 첫 row는 `[[4,-1,1],[-1,2,0],[4,-1,1]]`이다. ID 3이 두 번 나오면 같은 parameter row를 두 번 읽는다. batch axis와 sequence axis가 보존되고 hidden axis가 새로 붙는다. 여기에는 평균도 softmax도 없다. 같은 ID의 두 위치가 같은 시작 벡터를 갖는다는 사실과, 뒤 attention을 지난 결과도 같다는 주장은 전혀 다르다.

### ID 범위가 correctness의 첫 guard다

모든 ID에는 `0 ≤ id < V`가 성립해야 한다. tokenizer의 `vocab_size`, `len(tokenizer)`, config의 `vocab_size`, 실제 embedding weight의 첫 dimension은 서로 같다고 가정하지 않는다. added token은 tokenizer 길이와 최대 ID를 늘릴 수 있고, model embedding이 함께 resize되지 않았을 수 있다. 8장은 special ID와 config 계약을 소유한다. 이 장은 그 계약이 깨졌을 때 실제 row lookup에서 무슨 일이 생기는지를 이어 받는다.

`max_id < weight.shape[0]`이면 index 범위는 안전하지만 의미까지 안전한 것은 아니다. 새 행이 무작위 초기화되었거나 잘못된 checkpoint row가 들어갈 수 있다. 반대로 `max_id ≥ weight.shape[0]`이면 lookup은 범위 오류를 내야 한다. 오류가 난다는 것은 오히려 silent corruption보다 낫다. ID를 modulo로 접거나 unknown ID로 몰래 바꾸어 실행을 계속하면 protocol token 의미가 사라진다.

Transformers Qwen3.5 text model은 `nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)`를 만들고 forward에서 `inputs_embeds`가 없을 때 `embed_tokens(input_ids)`를 호출한다. 고정 소스는 [Transformers v5.15.1 `modeling_qwen3_5.py:1130-1164`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1130-L1164)다. `padding_idx`를 constructor에 넘긴다는 사실과 pad row가 attention에서 자동으로 완전히 사라진다는 주장은 다르다. mask가 별도 소비자다.

### dtype은 index와 row에 서로 다른 계약을 준다

`input_ids`는 정수 index다. 일반적인 PyTorch embedding은 `torch.long` 같은 정수 dtype을 기대한다. fp16으로 바꾸면 메모리가 줄 것이라는 생각은 범주 오류다. 부동소수점 2048.0은 table address 계약이 아니다. embedding weight와 출력 activation은 fp32, fp16, bf16 또는 양자화와 관련된 표현일 수 있다.

ID tensor byte는 대략 `B×S×index_bytes`이고, lookup 출력은 `B×S×H×activation_bytes`다. `B=2`, `S=4`, `H=8`, bf16이라면 input IDs가 int64일 때 64 byte, output은 128 byte다. 실제 모델의 `H`가 수천이면 output이 훨씬 커진다. 이 계산은 allocator overhead와 padding, temporary를 제외한 논리 payload다. “embedding은 단순 lookup이므로 메모리 비용이 없다”는 설명이 틀린 이유다.

weight dtype과 output dtype이 늘 같다고도 단정하지 않는다. quantized embedding은 packed weight를 읽어 dequantize output을 만들 수 있고, 일부 모델은 embedding을 다른 layer와 별도 dtype으로 유지한다. 어떤 kernel과 module이 선택되는지는 format·backend owner 장에서 더 깊게 다룬다. 여기서는 source에서 실제 module class, `weight.dtype`, 반환 `hidden_states.dtype`을 구분해 기록하는 습관을 세운다.

**one-hot 비유가 설명하는 것과 숨기는 것**

one-hot `o_i E`는 lookup의 수학적 결과를 잘 설명한다. 선택된 행만 결과에 기여하고, ID 사이에 1.5 같은 중간 주소가 없다는 점도 보인다. 그러나 실제 성능을 설명하는 비유로는 부족하다. 구현은 `[B,S,V]` one-hot tensor를 materialize하지 않으며, dense GEMM의 규칙적인 memory access를 그대로 갖지도 않는다. token IDs가 가리키는 row를 gather하므로 access pattern과 weight layout이 중요하다.

연속된 IDs가 반드시 메모리에서 의미상 가까운 token이라는 뜻도 아니다. vocabulary ID 배치는 tokenizer artifact가 정하며, semantic similarity 순서로 정렬된 좌표축이 아니다. embedding vector 사이 cosine similarity는 학습 결과의 한 단면이지만 ID 숫자 차이 `|i-j|`에는 일반적으로 의미가 없다. ID 17과 18이 인접 행이라는 사실을 “비슷한 단어”로 읽으면 안 된다.

embedding row 자체도 고정된 사전 정의가 아니다. checkpoint revision이 바뀌면 같은 token string과 ID가 유지되어도 row 값이 달라질 수 있다. adapter가 embedding module을 수정하거나 prompt embedding을 삽입하는 구성도 있다. 따라서 cache나 differential test identity는 tokenizer IDs만으로 닫히지 않는다. model weight와 adapter domain이 함께 필요하다.

문맥화된 token 표현을 보고 “embedding”이라고 부르는 관행도 source 읽기를 흐린다. 이 장에서는 `embed_tokens(input_ids)` 직후의 row를 input embedding이라 부르고, decoder layer를 지난 tensor는 hidden state라고 부른다. 마지막 layer의 hidden을 embedding이라고 부르는 downstream API가 있을 수 있지만, 어느 layer와 pooling을 뜻하는지 별도 계약이다.

### row lookup을 주소 계산으로 해부한다

논리적으로 contiguous한 dense table에서 row `i`의 byte 시작 주소는 `base + i×row_stride`다. `row_stride`는 최소 `H×element_bytes`지만 alignment와 packing 때문에 다를 수 있다. quantized table은 여러 값이 block 단위로 packed되고 scale/zero-point metadata가 붙어 단순 곱셈만으로 element를 읽지 않을 수 있다. 그래도 global ID가 어느 logical row를 선택하는지는 보존되어야 한다.

`input_ids=[[3,1,3]]`이면 같은 row 3을 두 번 읽는다. hardware cache가 두 번째 read를 재사용할 가능성은 있지만, 이것을 반드시 L2 hit라고 단정할 수 없다. 다른 warps와 layers가 cache를 경쟁하고 table이 크기 때문이다. 소스 감사는 access와 allocation 경로를 확정할 수 있지만 실제 cache hit rate는 profiler가 필요하다. 이 장에서는 실행하지 않으므로 성능 가능성과 측정 사실을 구별한다.

row lookup 뒤 output layout도 본다. logical shape `[B,S,H]`가 contiguous인지, packed tokens 때문에 `[T,H]`인지 engine마다 다를 수 있다. `T=sum(valid_lengths)`인 packed representation은 padding row를 아예 model input에서 제거할 수 있다. 같은 의미의 sequence를 dense batch와 packed batch로 표현하므로 tensor index를 논리 position으로 오인하지 않는다.

**resize가 단순히 행 수 하나를 바꾸지 않는 이유**

tokenizer에 새 token을 추가하고 embedding을 `V`에서 `V+k`로 늘리면 old rows를 복사하고 new rows를 초기화해야 한다. tied LM head가 있다면 output rows도 일치해야 한다. checkpoint serialization, optimizer state는 학습 책의 범위지만 serving artifact에는 최종 행 값과 tie 관계가 정확히 들어 있어야 한다.

새 행을 기존 token 평균으로 초기화하는 전략이 있어도 그 token 의미가 학습되었다는 뜻은 아니다. initialization은 오류를 피하는 시작값일 뿐이다. protocol marker가 새 row를 사용하면 첫 hidden이 학습 분포 밖일 수 있고 model output이 불안정해진다. “index error가 사라졌다”와 “새 token이 유효하다”를 분리한다.

quantized checkpoint나 tensor-parallel packed artifact는 runtime resize를 지원하지 않을 수 있다. 각 shard의 physical padding, quant block, LM-head tie가 다시 만들어져야 하기 때문이다. 운영 서버에서 tokenizer만 hot reload하는 방식은 특히 위험하다. 승인 bundle에서 tokenizer max ID와 checkpoint embedding rows를 함께 검증하고 원자적으로 배포한다.

**padding row의 특별함을 과장하지 않는다**

PyTorch `nn.Embedding`의 `padding_idx`는 학습 중 해당 row gradient를 다루고 초기화 시 row를 zero로 둘 수 있는 의미가 있다. 이미 checkpoint가 해당 row에 nonzero 값을 담았는지, loading 뒤 어떻게 유지되는지는 확인해야 한다. 추론에서는 index가 pad이면 여전히 output 위치가 생긴다. attention mask와 packing이 이를 제거하지 않으면 graph를 흐른다.

pad ID가 EOS와 같게 설정되는 decoder-only serving도 있다. 그렇다면 “pad row는 항상 zero”라는 가정은 EOS 의미를 훼손할 수 있다. ID 숫자 공유와 각 위치의 mask 역할을 구별해야 한다. 8장의 protocol 설정을 이어받되, 이 장의 관측에는 실제 pad 위치의 gathered row와 valid mask를 함께 둔다.

이 절을 닫는 판정은 다음과 같다. IDs와 weight revision이 같고 raw gathered row가 같다면 lookup 의미는 일치한다. raw row가 다르면 row mapping, checkpoint, dtype/dequantization, shard를 본다. raw row는 같고 첫 layer input이 다르면 scale, position addition, multimodal splice로 이동한다. “embedding 문제”라는 넓은 표현을 최초 달라진 연산으로 줄이는 것이 목적이다.

## 9.2 device와 병렬 소유권은 lookup의 의미를 바꾼다

단일 GPU에서는 ID와 embedding weight가 같은 device에 있고 output도 그 device에 생긴다고 생각하기 쉽다. 분산 serving에서는 embedding table이 vocabulary axis로 나뉘거나 pipeline 첫 rank에만 존재할 수 있다. `E[i]`라는 수식은 같지만 누가 그 행을 소유하고 어떻게 partial result를 합치는지가 실행 의미다.

vocabulary parallel을 두 rank로 손계산하자. `V=8`, rank 0이 global IDs `[0,4)`, rank 1이 `[4,8)`을 소유한다고 하자. 입력 `[1,6,3]`에서 rank 0은 ID 1과 3의 local row를 읽고 ID 6 위치에는 zero partial을 둔다. rank 1은 ID 6만 읽고 나머지를 zero로 둔다. 같은 `[S,H]` shape의 partial을 all-reduce sum하면 global lookup 결과가 된다. ID range mask와 local offset `global_id-vocab_start`가 정확해야 한다.

rank 1이 ID 6을 local ID 6으로 그대로 사용하면 local table `[4,H]` 범위를 넘는다. shard 시작을 잘못 빼 ID 2를 읽으면 shape는 맞고 숫자만 틀린 silent failure가 된다. 그래서 분산 embedding 문제는 단지 collective가 성공했는지로 끝나지 않는다. global ID, shard `[start,end)`, local masked ID, partial nonzero owner, reduction 결과를 한 행에 둔다.

vLLM의 `VocabParallelEmbedding`은 padding된 vocabulary, tensor-parallel rank별 shard index, quantization method를 구성한다. class와 shard 계산 경계는 [vLLM v0.27.1 `vocab_parallel_embedding.py:198-270`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/vocab_parallel_embedding.py#L198-L270)에서 읽을 수 있다.

shard index 계산은 [같은 파일 `:351-377`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/vocab_parallel_embedding.py#L351-L377)에 있다. logical vocab과 padding된 physical rows를 구별해야 하는 소스 근거다.

Qwen3.5 vLLM model은 이 parallel embedding을 만들고 pipeline rank 경계를 고려한다. 고정 위치는 [vLLM v0.27.1 `qwen3_5.py:231-258`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5.py#L231-L258)이다. tied weight를 선택하는 wrapper 경계는 [같은 파일 `:316-337`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5.py#L316-L337)에 있다.

tied embedding의 출력 역할은 16장에서 LM head와 함께 본다. 여기서는 동일 storage를 기대할 때 실제 module wiring이 그렇게 되었는지 확인한다.

### pipeline parallel에서는 첫 stage가 row를 만든다

pipeline parallel의 중간 rank는 보통 raw IDs가 아니라 이전 stage hidden states를 받는다. 모든 rank가 embedding weight를 갖는다고 가정하면 memory budget과 weight loading 검증이 틀린다. 첫 stage는 IDs 또는 precomputed inputs embeddings에서 hidden state를 만들고, 중간 stage는 intermediate tensor를 이어 받는다. 마지막 stage는 LM head를 소유할 수 있다. tied weight 때문에 first와 last 사이 weight 전달·복제·tie 정책이 별도 문제가 된다.

SGLang Qwen3.5 model은 first pipeline rank에서 `embed_tokens`를 만들고, forward에서 `input_embeds`가 없을 때 lookup하며 그렇지 않으면 외부 embedding을 사용한다. construction은 [SGLang v0.5.18 `qwen3_5.py:1340-1362`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_5.py#L1340-L1362), 분기는 [같은 파일 `:1410-1434`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_5.py#L1410-L1434)에 있다.

`input_embeds`가 주어졌다는 이유로 그것이 올바른 model revision의 row라는 보장은 없다. lookup 우회는 identity 검증 책임을 caller로 옮긴다.

**설정을 필드에서 관측까지 닫는다**

`tensor_parallel_size=2`라는 필드 하나를 읽고 “embedding이 두 배 빨라진다”라고 설명하면 쓸모가 없다. 이 필드는 parallel group 크기를 바꾸고, vocab row shard 범위와 physical padding을 바꾸며, local gather와 collective 경로를 만든다. 효과 후보는 rank당 weight memory 감소와 collective 추가다. 실제 latency가 좋아지는지는 vocab size, interconnect, batch token 수, kernel에 달렸다. 반증 관측은 각 rank row 범위, local output, collective bytes와 timeline이다.

`dtype=bf16`도 같은 형식으로 읽는다. field는 weight/activation dtype 결정 경로에 들어가고, embedding module 또는 loader가 실제 storage dtype을 정하며, lookup output byte와 numeric representation이 바뀐다. 기대 효과는 memory bandwidth와 capacity 변화지만 embedding이 해당 dtype으로 실제 load되지 않거나 backend가 cast하면 요청만 하고 상태는 달라지지 않을 수 있다. config 문자열, loaded weight dtype, output dtype을 함께 보아야 한다.

**vocab padding과 sequence padding은 다른 문제다**

vocab parallel 구현이 행 수를 TP size나 quantization block 배수에 맞추려고 physical row를 추가하는 것을 vocabulary padding이라 부를 수 있다. batch의 짧은 문장을 같은 width로 맞추는 sequence padding과 이름만 같고 축이 다르다. 앞은 `[V,H]`의 V축을 늘리고, 뒤는 `[B,S]`의 S축을 늘린다.

logical vocab이 10이고 TP 4개에 균등 배치하려 physical rows를 12로 만들었다고 하자. rank별 physical row는 3개지만 global IDs 10과 11은 사용자 token이 아니다. tokenizer max ID가 11이라고 해서 physical table 범위 안이라는 이유로 허용하면 untrained padding row를 읽는다. guard는 `id < logical_vocab`, loader는 physical shard shape도 별도로 확인해야 한다.

sequence padding은 반대로 valid token mask와 position을 요구한다. 이 두 padding을 한 metric `padding_ratio`로 묶으면 어느 waste를 줄여야 하는지 모른다. vocabulary padding overhead는 weight bytes에, sequence padding은 activation/attention work에 주로 나타난다. engine packing이 sequence padding을 줄여도 vocabulary shard padding은 그대로다.

**ID와 weight가 다른 device에 있을 때**

PyTorch high-level 경로에서는 input ID와 embedding weight의 device가 맞지 않으면 오류가 나거나 명시적 이동이 필요하다. serving engine은 CPU에서 scheduler metadata를 만들고 GPU input buffer로 복사한 뒤 graph를 실행할 수 있다. `.to(device)` 한 줄이 보인다고 매 request 동기 copy라고 단정하지 않는다. buffer reuse, nonblocking copy, pinned memory, graph input update를 계속 따라가야 한다.

관찰에는 source device, destination device, dtype, byte 수, stream, synchronization consumer를 둔다. ID tensor는 weight보다 작지만 decode step마다 작은 copy와 sync가 critical path에 붙으면 ITL에 영향을 줄 수 있다. 반면 prefill의 긴 ID copy는 전체 model compute에 비해 작을 수 있다. 실제 비율은 측정 없이는 주장하지 않는다.

precomputed `inputs_embeds`는 IDs보다 훨씬 크다. `[T,H]` bf16 tensor를 다른 GPU로 옮기는 비용은 `[T]` int64 IDs보다 대략 `H/4`배 클 수 있다. pipeline boundary나 multimodal processor가 잘못된 device에서 embedding을 만들면 큰 transfer가 생길 수 있다. “lookup을 미리 해서 CPU를 줄인다”는 최적화가 transfer를 늘리는 반례다.

### vocab-parallel 손계산을 수치로 검증한다

각 row가 scalar라고 단순화하고 global table을 `[10,20,30,40,50,60]`이라 하자. rank 0은 IDs `[0,3)`, rank 1은 `[3,6)`을 소유한다. 입력 `[0,4,2,5]`에 대해 rank 0 partial은 `[10,0,30,0]`, rank 1 partial은 `[0,50,0,60]`이다. sum은 `[10,50,30,60]`이다.

rank 1의 start offset을 2로 잘못 계산하면 global ID 4가 local row 2를 읽어 60이 되고 global ID 5는 range 밖으로 mask될 수 있다. partial은 `[0,60,0,0]`이고 reduction 결과 shape는 여전히 `[4]`다. collective failure metric은 0인데 수치가 틀린다. shard boundary IDs를 포함한 golden fixture가 필요한 이유다.

TP size를 바꿀 때 field는 rank range와 collective group을 모두 바꾼다. checkpoint loader도 global weight를 새 shard로 나누거나 이미 sharded artifact를 맞춰 읽는다. weight loading은 성공했지만 process group이 다른 ranks를 묶으면 partial sum이 엉뚱해질 수 있다. 반증은 rank별 shard metadata와 known row slice, collective group membership을 함께 확인하는 것이다.

**pipeline과 tied weight의 소유권 충돌**

입력 embedding은 first stage, LM head는 last stage에 필요하다. weight tying을 물리적으로 유지하려면 같은 storage를 두 stage가 직접 공유하기 어렵거나 별도 통신/복제 정책이 필요할 수 있다. 구현은 weights를 복제하고 loading value를 같게 유지하거나, pipeline 구조에 맞는 tying helper를 사용할 수 있다. config flag 하나로 memory가 반드시 절반이라고 계산하지 않는다.

첫 stage에 embedding이 없거나 middle stage가 raw IDs를 받는다면 pipeline wiring이 깨진다. 반대로 모든 stage에 table이 로드되면 correctness는 맞아도 memory budget이 예상보다 커질 수 있다. state dict key 존재, local module type, parameter bytes를 rank별로 본다. tied 여부는 Python object 이름보다 actual storage 또는 loader semantics로 판정한다.

이 절에서 설정 설명의 종료 조건은 “TP를 켠다”가 아니다. 필드가 shard partition을 어떻게 만들고, rank local table과 masked partial을 어떻게 바꾸며, 어떤 collective가 global hidden을 복원하고, weight memory와 transfer에 어떤 후보 효과를 주는지 말해야 한다. 마지막으로 boundary ID fixture와 rank별 slice가 그 설명을 반증할 수 있어야 한다.

## 9.3 padding과 mask는 첫 hidden state를 어디까지 무시할지 정한다

padding ID도 유효한 정수라 embedding row를 읽는다. `padding_idx`가 있으면 그 row를 특별히 초기화하거나 학습 gradient에서 제외할 수 있지만, 추론 graph에서 해당 위치가 자동 삭제되는 것은 아니다. lookup 결과 tensor에는 pad 위치의 hidden row가 존재한다. attention mask가 key/query visibility를 막고, 위치 계산이 pad를 어떻게 건너뛰는지가 뒤 결과를 정한다.

두 문장 `[A,B,C]`와 `[D]`를 left-pad해 같은 width로 만들자.

```text
input_ids      [[A, B, C],
                [P, P, D]]
attention_mask [[1, 1, 1],
                [0, 0, 1]]
```

embedding output은 두 row 모두 `[3,H]`다. 둘째 row의 앞 두 위치에도 `E[P]`가 있다. 올바른 mask를 쓰면 실제 token D의 query가 pad key를 보지 않도록 해야 한다. mask가 전부 1이면 D는 P의 K/V를 문맥으로 읽고 첫 layer부터 달라진다. pad embedding이 zero여도 projection bias, position 처리, normalization, residual 경로 때문에 “zero니까 완전히 무해하다”고 일반화할 수 없다.

position IDs를 attention mask의 누적합으로 만들고 pad 위치를 별도 값으로 채우는 계열에서는 위 row가 대략 `[x,x,0]` 또는 mask convention에 따른 값이 된다. 단순 `arange(3)=[0,1,2]`를 쓰면 D의 논리 위치가 2가 된다. right padding이면 실제 token이 앞에서 시작하므로 단순 arange와 맞아 보일 수 있지만 decode batching과 cache position에서 다시 갈린다.

**세 가지 mask를 혼동하지 않는다**

tokenizer가 내는 0/1 padding mask, model이 만드는 causal visibility, attention backend가 받는 additive 또는 boolean mask는 다른 상태다. padding mask가 `[B,S]`이고 causal mask가 `[S,S]`라면 결합 결과는 broadcast된 `[B,1,Q,K]`일 수 있다. flash 계열 backend는 explicit dense mask 대신 sequence length와 causal flag를 받을 수 있다. 이 변환은 13장의 attention mask가 소유한다. 이 장에서는 lookup 뒤 pad hidden이 실제 문맥으로 흘러가지 않게 만드는 첫 입력 계약까지만 닫는다.

실패 재현은 간단하다. 내용 token IDs가 같은 두 batch를 left padding 폭만 다르게 만든다. 올바른 position/mask 정책이면 유효 token의 첫-layer 이후 결과가 허용 오차 안에서 같아야 한다. 다르면 embedding output의 유효 slice부터 비교한다. lookup부터 다르면 ID 또는 row 문제다. lookup은 같고 position 적용 뒤 다르면 위치 좌표다. Q/K/V 이후 다르면 mask와 attention 장으로 이동한다.

padding side 설정은 field→분기→상태→효과로 읽는다. `padding_side="left"`는 tokenizer batch assembly에서 pad 위치를 앞쪽으로 정하고 `input_ids`와 mask 배열을 바꾼다. model의 position helper가 mask-aware하면 유효 token 좌표를 다시 만들고, 그렇지 않으면 absolute position이 이동한다. 효과는 batch shape 통일과 decode의 마지막 token 정렬이지만, 잘못된 position/mask 조합은 logits parity를 깨뜨린다. 반증은 같은 문장을 서로 다른 pad 폭으로 구성한 유효 hidden 비교다.

**right padding도 항상 안전하지 않다**

right padding은 valid tokens의 tensor index와 logical position이 앞부분에서 일치해 이해하기 쉽다. 하지만 batch의 마지막 tensor column이 pad일 수 있어 “항상 `hidden[:,-1]`에서 다음 logits를 읽는다”는 구현은 짧은 row에서 틀린 위치를 고른다. generation wrapper가 valid length를 알고 마지막 유효 token을 선택하거나 left padding을 쓰는 이유가 된다. 이 출력 선택은 16장의 LM head 입력 경계로 이어진다.

prefill 뒤 decode에서는 각 request 길이가 다르다. dense batch width가 128이라도 짧은 request의 다음 logical position은 23일 수 있다. 공통 tensor width를 position으로 쓰면 짧은 request가 128에서 이어진다. continuous batching engine은 request별 sequence length 또는 position metadata를 유지해야 한다.

padding side를 바꿨는데 token IDs의 multiset이 같다는 검사는 아무것도 증명하지 못한다. sequence 순서와 mask, position을 함께 봐야 한다. 특히 EOS와 PAD ID가 같으면 ID 값만으로 어느 EOS가 실제 종료 token이고 어느 것이 batch filler인지 구별할 수 없다. mask와 valid length가 의미를 준다.

### mask가 잘못되었을 때 차이는 어떻게 전파되는가

pad key 하나가 visible해지면 첫 attention layer에서 query의 score 후보가 하나 늘어난다. softmax denominator와 가중합이 바뀌고 attention output이 residual stream에 더해진다. 이후 normalization과 다음 layer가 이 차이를 비선형적으로 전파한다. 마지막 logits가 조금 또는 크게 바뀌는지는 예측하기 어렵지만 최초 divergence는 첫 attention이다.

pad embedding이 zero라 해도 key/value projection에 bias가 있으면 zero input이 nonzero K/V가 될 수 있다. bias가 없어 value가 zero여도 pad score가 softmax probability 일부를 가져가 유효 value들의 합을 줄인다. 따라서 zero value가 visible해도 결과는 달라질 수 있다. 이 손계산이 “pad row zero면 mask가 필요 없다”는 설명을 반박한다.

유효 value가 `v=[2]`, pad value가 `[0]`이고 둘의 score가 같다면 올바른 mask output은 2지만 잘못된 두-key softmax output은 `0.5×2+0.5×0=1`이다. projection bias나 position이 없어도 이미 다르다. mask의 목적은 pad가 무의미한 값을 갖게 만드는 것이 아니라 후보 집합에서 제외하는 것이다.

### packed sequence에서는 pad row 대신 경계 metadata가 중요하다

serving engine이 여러 request valid tokens만 `[T,H]`로 이어 붙이면 explicit pad hidden은 줄어든다. 대신 각 sequence 시작 offset과 length, query/key boundary가 필요하다. request A의 token이 request B의 key를 보지 않도록 causal mask가 sequence boundary를 알아야 한다. padding mask 오류가 사라진 대신 cumulative length 또는 block table 오류가 같은 역할을 한다.

packed tensor index 10이 logical position 10이라는 보장도 없다. 앞 request 길이가 7이면 다음 request의 첫 token이 packed index 7이지만 logical position은 0이다. position IDs와 request mapping을 별도 유지한다. batch compaction이나 request reorder 뒤 metadata가 hidden row와 같이 이동하지 않으면 서로 다른 사용자의 위치·mask가 섞인다.

실패 재현은 서로 다른 길이의 두 request 순서를 바꾸는 방식이 강하다. request별 gathered row와 logical position은 순서와 무관하게 같아야 한다. packed offset은 달라져도 결과를 원래 request identity로 복원했을 때 first-layer input이 같아야 한다. 순서에 따라 달라지면 batch row mapping, cumulative length, position metadata 동기화를 본다.

### position helper와 mask helper의 결합을 읽는다

일부 generation 경로는 `attention_mask.long().cumsum(-1)-1`로 position을 만들고 pad 위치를 특정 값으로 채운다. 이 방법은 valid token만 순서를 증가시키는 직관을 준다. 그러나 model-specific code가 별도 `cache_position`을 요구하거나 multimodal position rank가 다르면 generic helper만으로 충분하지 않다.

mask dtype도 주의한다. original 0/1 integer mask를 bool로 바꾸는 것은 대개 visibility 의미를 보존하지만, additive mask는 허용 위치 0과 차단 위치 큰 음수를 쓴다. 0/1을 그대로 attention score에 더하면 차단이 아니라 1만큼 가산하는 전혀 다른 연산이다. backend 직전 ABI는 13장에서 다루되 input ledger에는 변환 전후 convention을 기록한다.

이 절의 handoff는 선명하다. gathered rows가 맞고 pad 폭에 따른 position state도 같다면 embedding 입구는 닫힌다. 첫 attention visibility가 다르면 13장으로 이동한다. decode step에서만 position이 갈리면 14장의 cache length와 RoPE를 본다. 마지막 유효 hidden 선택만 틀리면 16장의 LM head input indexing을 본다.

## 9.4 위치 정보는 “몇 번째 token인가”를 여러 방식으로 표현한다

embedding row만으로는 `[A,B]`의 A와 `[B,A]`의 A가 같은 시작 벡터다. 모델이 순서를 구별하려면 위치를 계산에 넣어야 한다. absolute learned embedding은 위치 `p`의 별도 row `P[p]`를 token embedding에 더할 수 있다. sinusoidal absolute encoding은 고정 함수로 좌표를 만든다. relative position bias는 query 위치와 key 위치의 차이를 attention score에 더한다. RoPE는 Q와 K의 channel pair를 위치 각도만큼 회전해 내적이 상대 위치 차이를 반영하게 한다.

이 네 방식을 “위치 벡터를 token embedding에 더한다”로 묶으면 RoPE와 relative bias를 잘못 이해한다. 많은 decoder-only 모델에서 `inputs_embeds` 자체에는 positional vector를 더하지 않고, layer 안에서 Q/K에 rotary transform을 적용한다. 그래도 첫 forward 전에 `position_ids` 또는 동등한 좌표를 준비해야 하므로 이 장에서 입구를 다룬다. 회전 수학과 KV cache 결합은 14장의 소유다.

### absolute lookup을 손으로 계산한다

token embedding이 `E[A]=[1,2]`, `E[B]=[3,4]`, learned position rows가 `P[0]=[0.1,0.2]`, `P[1]=[0.3,0.4]`라면 `[A,B]`의 합은 `[[1.1,2.2],[3.3,4.4]]`다. `[B,A]`는 `[[3.1,4.2],[1.3,2.4]]`다. 같은 A도 위치에 따라 시작 hidden이 달라진다. `max_position_embeddings` 밖의 index는 table 범위를 넘으므로 context extension이 단순 서버 max-length 옵션이 아닌 이유가 된다.

padding 때문에 A가 tensor index 5에 있어도 논리 position을 0으로 줄 수 있다. 어느 좌표가 맞는지는 학습과 cache 정책에 달렸다. tensor storage offset, logical sequence position, cache slot은 같은 숫자가 아니다. continuous batching에서는 서로 다른 요청 token이 packed buffer에서 인접해도 각 요청의 논리 position은 독립적으로 이어진다.

### RoPE 준비 단계의 shape ledger

일반 text RoPE position IDs가 `[B,S]`이고 hidden이 `[B,S,H]`라고 하자. Q/K projection 뒤 head shape와 rotary dimension이 정해지고 position별 cos/sin을 gather 또는 계산한다. prefill `[0,1,2,3]` 뒤 decode token은 논리 position 4를 받아야 한다. cache에 쓰는 physical slot이 97이어도 rotary position이 97인 것은 아니다. 두 좌표를 섞으면 cache read/write는 성공하면서 수치가 틀린다.

Transformers Qwen3.5 model이 `cache_position`이 없을 때 past length에서 arange를 만들고 `position_ids` 기본을 구성하는 경계는 [Transformers v5.15.1 `modeling_qwen3_5.py:1163-1181`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1163-L1181)다. 이 helper가 있다는 사실은 serving engine의 packed position과 같은 구현을 쓴다는 뜻이 아니다. 동일 logical 좌표를 다른 data structure로 만들 수 있다.

relative bias는 token embedding row를 바꾸지 않고 attention score의 `(q,k)` pair에 영향을 준다. 따라서 embedding output이 두 구현에서 같고 첫 attention score부터 다르면 relative position bucket과 bias를 조사한다. absolute addition 모델이라면 embedding-plus-position 직후부터 달라질 수 있다. first divergence 위치가 architecture family를 좁혀 준다.

`max_position_embeddings`, rope scaling, position offset 같은 설정은 field 이름만 나열하지 않는다. field가 position helper 또는 rotary module 생성 분기로 들어가고, cos/sin frequency나 허용 좌표 상태를 바꾸며, 긴 context에서 phase와 backend eligibility에 영향을 준다. 짧은 fixture가 같다고 설정이 적용되지 않았다고 단정할 수 없다. 경계를 넘는 위치와 source state를 관측해야 반증할 수 있다.

**sinusoidal absolute 좌표를 두 차원으로 맛본다**

sin/cos 위치 encoding을 단순화해 위치 `p`에서 `[sin(p),cos(p)]` 한 쌍을 만든다고 하자. `p=0`이면 `[0,1]`, `p=π/2`라면 `[1,0]`이 된다. 실제 transformer는 여러 frequency를 hidden dimensions에 배치해 짧고 긴 주기의 변화를 함께 표현한다. learned table과 달리 함수로 좌표를 만들 수 있지만 training 범위 밖 extrapolation이 자동으로 정확하다는 보장은 없다.

absolute encoding을 token embedding에 더하면 first decoder layer input에서 바로 관찰할 수 있다. token row가 같은데 addition 뒤 달라지면 position IDs 또는 encoding table/function이 원인이다. 반면 RoPE-only 모델은 embedding 직후가 같고 Q/K 회전 뒤 처음 달라질 수 있다. hook 위치를 architecture에 맞춰 정해야 하는 이유다.

position addition 앞뒤 dtype도 본다. fp32로 encoding을 계산한 뒤 bf16 hidden에 더할 때 cast가 어디서 일어나는지, table이 어느 device에 있는지 확인한다. 긴 context에서 작은 phase 차이가 생겼다고 무조건 position algorithm 오류라고 하지 않는다. 동일 좌표와 dtype에서 reference 결과를 비교해야 한다.

**relative bucket은 거리 전체를 그대로 저장하지 않는다**

relative position bias 계열은 `q_pos-k_pos`를 bucket으로 양자화할 수 있다. 가까운 거리는 세밀하게, 먼 거리는 로그 구간으로 묶는 방식이 가능하다. 그러면 거리 100과 101이 같은 bucket bias를 공유할 수 있다. bucket 수와 max distance 설정은 bias table shape와 mapping 분기를 바꾼다.

hand ledger에는 query position, key position, signed distance, bucket ID, selected bias row를 둔다. causal model에서는 미래 key가 mask되므로 distance 부호와 bucket convention을 특히 주의한다. bucket mapping 하나가 뒤집히면 tensor shape는 모두 정상이고 attention score만 체계적으로 틀린다.

relative bias가 layer마다 별도인지 공유되는지에 따라 weight owner와 memory가 달라진다. 이 차이는 embedding table과 무관하지만 position 정보를 준비하고 소비하는 첫 경계를 이해하는 데 필요하다. first embedding을 비교해 같다고 전체 position parity를 선언하지 않는다.

**RoPE 손계산은 회전 전 좌표까지만 잡는다**

2차원 pair `(x,y)`를 각도 `θp`만큼 회전하면 `(x cos θp-y sin θp, x sin θp+y cos θp)`다. position 0에서는 변하지 않고 position 1에서는 기본 각도 `θ`만큼 회전한다. Q와 K가 각각 위치 `p`와 `q`에서 회전하면 내적에 대체로 상대 각도 `θ(p-q)`가 나타난다는 것이 직관이다.

이 장에서 중요한 것은 `p`와 `q`가 어디서 왔는지다. prefill 네 token의 position `[0,1,2,3]`, 다음 decode `[4]`, prefix cache hit 뒤 새 suffix 시작 `[cached_length]`가 논리적으로 이어져야 한다. physical KV block index나 packed index를 넣으면 회전은 정상 실행되며 틀린 좌표를 사용한다.

RoPE scaling field는 base frequency, position transform 또는 cos/sin cache 생성에 들어간다. field가 config에 존재하는 것과 model implementation이 해당 type을 지원하는 것은 다르다. backend가 자체 rotary kernel을 쓰면 Python module state가 직접 실행되지 않을 수도 있다. config→selected rotary implementation→실제 frequency/position input→kernel symbol→long-context 결과를 연결해야 한다.

짧은 sequence에서는 두 scaling 설정 차이가 매우 작거나 의도적으로 같을 수 있다. position 0 fixture만으로는 어떤 회전도 일어나지 않아 검증력이 없다. position 1, training context 경계 근처, scaling이 시작되는 구간을 정적 계산 fixture로 둔다. 실행하지 않더라도 source formula와 expected coordinate를 손으로 만들 수 있다.

### 세 위치 좌표를 분리하는 incident ledger

continuous batching 요청 R의 다음 token이 논리 position 41이고, packed input row 7이며, KV cache physical slot 902라고 하자. 이 세 숫자는 각기 다른 owner가 필요하다. model RoPE는 41, input gather/scatter는 row 7, cache write는 slot 902를 쓸 수 있다. 모두 integer tensor라 잘못 연결해도 type과 shape가 맞는다.

request reorder 뒤 position tensor만 옛 row 순서를 유지하면 R이 다른 request 위치를 받는다. cache block table만 옛 순서면 rotary는 맞지만 KV를 잘못 쓰거나 읽는다. packed row만 틀리면 token embedding 자체가 다른 request와 섞인다. request ID별로 `(input_row, logical_pos, cache_slot)` triple을 기록하면 첫 divergence를 분류할 수 있다.

prefix cache hit도 논리 position을 생략하지 않는다. cached prefix KV는 원래 positions에서 계산되었고 새 suffix는 그 길이 뒤에서 이어져야 한다. 동일 token IDs라도 position offset이나 rope scaling domain이 다르면 cached KV 재사용이 안전하지 않을 수 있다. cache identity 상세는 뒤 편의 소유지만, 이 장은 위치 설정이 identity에 들어가는 이유를 제공한다.

### 위치 실패를 정적 fixture로 닫는다

fixture A는 unpadded `[A,B,C]`, fixture B는 left-padded `[P,P,A,B,C]`, fixture C는 prefix length 10 뒤 decode `[C]`다. A와 B의 유효 logical positions가 모두 `[0,1,2]`인지 확인한다. C의 new token position은 10이어야 하며 physical buffer index와 분리한다.

absolute-add 모델이면 token row, position row, 합 결과를 비교한다. RoPE 모델이면 token row는 같고 Q/K 직전 `position_ids`와 cos/sin selection을 비교한다. relative-bias 모델이면 query-key bucket을 비교한다. architecture별 최초 position consumer가 다르다는 사실을 fixture 설계에 반영한다.

이 관측이 모두 같으면 position 입구 가설을 버린다. RoPE 적용 수치가 다르면 14장, attention backend mask/bias가 다르면 13장, layer input 자체가 다르면 10장으로 이어 간다. position이라는 단어 하나로 세 장을 동시에 의심하지 않는다.

## 9.5 멀티모달 입력은 placeholder 자리에 다른 feature를 접합한다

이미지 placeholder ID는 이미지 내용이 아니다. text sequence 안에서 vision encoder가 만든 feature가 들어갈 위치와 protocol 경계를 표시한다. processor는 image를 pixel tensor와 grid metadata로 바꾸고 text에는 placeholder를 만든다. model wrapper는 text embedding을 만든 뒤 placeholder 위치를 vision feature로 대체하거나, model-specific merge 규칙으로 sequence를 확장한다.

작은 예를 보자. text IDs가 `[BOS, IMG, IMG, Q]`이고 hidden size가 3이라고 하자. text lookup 결과의 IMG 두 row가 각각 같은 `E[IMG]`다. vision encoder가 `v0=[0.5,1.0,-1.0]`, `v1=[0.2,0.1,0.7]`을 만들면 splice 뒤 hidden은 `[E[BOS],v0,v1,E[Q]]`가 된다. placeholder count 2와 feature row count 2가 맞아야 한다. feature가 3개이면 어느 row를 버릴지 임의로 정하지 말고 계약 오류를 내는 편이 안전하다.

placeholder 하나가 여러 feature row로 확장되는 설계라면 tokenizer sequence length와 model sequence length가 다르다. 이때 attention mask, position IDs, cache length, scheduler token accounting을 확장 뒤 길이에 맞춰야 한다. 텍스트 token 수만 보고 KV capacity를 예약하면 under-accounting이 생긴다. 정확한 회계 owner는 multimodal processor와 engine interface에 걸쳐 있다.

### 순서와 수량은 모두 binding 계약이다

이미지 두 장과 placeholder 두 묶음이 있어도 순서가 바뀌면 shape는 맞는다. 첫 질문이 두 번째 image feature를 보게 되는 silent semantic failure다. request content order, rendered placeholder order, pixel batch index, feature slice, final embedding span을 하나의 binding ledger로 둔다. 단순 count equality보다 강한 검증이다.

Transformers Qwen3.5 conditional generation 경로에서 text embedding과 vision feature가 만나는 source는 [Transformers v5.15.1 `modeling_qwen3_5.py:1668-1745`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1668-L1745) 부근이다. 실제 고정 코드에서 placeholder mask, feature shape 검증, scatter 방식을 함께 읽어야 한다. class 이름만 보고 모든 multimodal model이 같은 expansion 정책을 쓴다고 일반화하지 않는다.

Gemma3 conditional generation도 image token mask와 image embedding을 text embedding에 결합하는 경계를 가진다. [Transformers v5.15.1 `modeling_gemma3.py:770-890`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L770-L890)를 Qwen 경로와 비교할 때 placeholder ID, scale, scatter, mask update가 같은지 각각 본다. “둘 다 masked_scatter를 쓴다”보다 주변 shape contract가 중요하다.

### multimodal position은 1차원 arange로 끝나지 않을 수 있다

image grid는 temporal, height, width 좌표를 가질 수 있고, model은 text와 vision token에 서로 다른 position 구성 규칙을 적용할 수 있다. Qwen 계열 multimodal RoPE는 grid와 placeholder span에서 다축 position을 준비하고 decode가 이어 받을 delta를 보존할 수 있다. 이 장은 position tensor가 생성되는 입구와 binding을 다루고, 실제 rotary channel 분할은 14장으로 넘긴다.

텍스트-only가 맞고 image request만 틀리면 attention kernel부터 교체하지 않는다. processor output의 placeholder 위치와 grid, vision feature row 수, splice 뒤 sequence, position tensor shape, decode로 넘긴 delta를 순서대로 비교한다. splice 직후 hidden부터 다르면 vision/binding 문제다. splice는 같고 rotary 적용 뒤 다르면 position/RoPE owner로 간다.

truncation은 이 계약을 깨뜨릴 수 있다. placeholder 묶음 일부만 잘리면 feature는 남고 insertion slot은 부족해진다. 8장의 truncation 정책은 placeholder 원자성을 보존해야 하고, 이 장의 splice guard가 수량 불일치를 잡아야 한다. 오류를 mask padding으로 숨기면 잘못된 image가 잘못된 질문에 결합될 수 있다.

### masked scatter를 shape만으로 이해하지 않는다

boolean mask가 text hidden `[T,H]`에서 placeholder rows를 고르고 vision features `[N,H]`를 넣는다고 하자. mask true element를 feature element와 맞추는 구현은 row mask를 hidden dimension으로 expand할 수 있다. 이 경우 true 원소 수는 `N×H`여야 한다. 단순 placeholder token count `N`만 비교하고 expanded mask contract를 놓치면 오류 메시지를 잘못 해석한다.

feature dtype과 device도 text hidden에 맞아야 한다. vision tower가 fp32 CPU output을 만들고 text model이 bf16 GPU라면 splice 전에 cast/transfer가 필요하다. 어디서 변환하는지에 따라 memory와 synchronization이 달라진다. 자동 cast가 성공해 shape 오류가 없다고 효율적 경로인 것은 아니다.

in-place처럼 보이는 masked scatter가 실제 새 tensor를 만드는지, autograd 여부와 inference graph에서 alias가 있는지도 source로 확인한다. serving에서는 memory peak와 graph capture 가능성에 영향을 줄 수 있다. 다만 source만으로 allocator peak 수치를 단정하지 않고 allocation 가능 경로로 기록한다.

### feature 수는 image 수와 같지 않다

이미지 한 장은 patch embedding, spatial merge, special delimiters를 거쳐 여러 model tokens가 된다. 원본 width와 height, resize/crop, patch size, merge size가 feature row 수를 바꾼다. 따라서 `num_images=1`과 `num_image_tokens=1`을 혼동하면 scheduler accounting과 placeholder expansion이 모두 틀린다.

간단히 resize된 image가 `4×4` patch grid이고 `2×2` spatial merge를 한다면 merge 뒤 feature grid는 `2×2`, 즉 4 rows가 될 수 있다. 시작/끝 delimiter가 text token으로 따로 남으면 model sequence에는 그 4 rows 외 marker rows가 있다. 정확한 식은 model processor config와 source를 따라야 하며, 이 숫자는 설명용이다.

video는 temporal axis가 추가된다. frame sampling이 달라지면 같은 파일도 feature rows가 달라질 수 있다. request admission이 tokenizer placeholder count만 보고 token budget을 잡으면 실제 expanded length를 놓친다. processor가 확정한 grid와 merge 뒤 row count를 scheduler에 넘기는 경계를 확인한다.

### placeholder를 원자적 protocol로 검증한다

template가 `<vision_start>`, 반복 image pad, `<vision_end>`를 만든다고 하자. tokenizer가 marker를 각각 원자적 special ID로 내는지, 일반 text에서 사용자가 동일 표면을 입력했을 때 special parsing이 허용되는지는 8장의 계약이다. 이 장에서는 최종 IDs에서 start/end 균형과 pad span이 feature insertion contract를 만족하는지 본다.

truncation이 end marker만 자르거나 pad span 일부를 줄이면 feature row 수와 맞지 않는다. processor가 feature도 함께 crop하는 정책이 명시되어 있지 않다면 silent truncation을 해서는 안 된다. request validation에서 구조를 보존한 단위로 줄이거나 전체 image segment를 제거하는 선택이 필요하다.

placeholder ID가 ordinary learned embedding row도 가진다고 해서 그 row가 최종 hidden에 남는다고 가정하지 않는다. scatter가 완전히 대체하면 placeholder row는 임시 carrier다. 일부 architecture는 delimiter embedding을 남기고 content pad rows만 대체할 수 있다. splice 전후 span을 비교해 실제 정책을 확인한다.

### batch expansion과 image binding

batch row 0에 image 한 장, row 1에 image 세 장이 있으면 flat pixel batch와 text batch 사이 ragged mapping이 필요하다. 단순히 모든 image features를 같은 수로 repeat하면 row 경계가 무너진다. beam expansion이나 request duplication에서도 text row와 각 sample의 image count를 함께 확장해야 한다.

binding ledger에 `request_id`, `content_index`, `pixel_batch_range`, `grid`, `feature_row_range`, `placeholder_span`, `merged_span`을 둔다. 이 값으로 count와 order를 모두 검증한다. 두 requests의 이미지 수가 같을 때만 통과하는 버그를 잡으려면 `[1,3]`과 `[3,1]`처럼 ragged 순서를 바꾼 fixture가 필요하다.

shape error가 난다고 무조건 vision encoder output이 잘못된 것은 아니다. feature rows는 맞고 placeholder mask가 template에서 줄었을 수 있다. 반대로 mask는 맞고 spatial merge config가 다른 checkpoint에서 load되었을 수 있다. rendered IDs→processor grid→vision output→splice mask 순으로 first divergence를 찾는다.

### multimodal 위치 delta가 decode로 이어지는 이유

vision token이 다축 좌표를 사용하거나 text sequence accounting과 다른 좌표 범위를 쓰면 prefill 마지막 뒤 decode text position을 단순 `input_ids_length`로 정하지 못할 수 있다. model wrapper가 계산한 delta는 이후 decode의 position을 이어 주는 상태다. 이 state가 request reorder나 cache hit에서 다른 row와 섞이면 첫 decode token부터 달라진다.

prefill ledger에는 position tensor 전체를 거대하게 저장하기보다 rank, 각 axis의 시작/끝, placeholder span의 bounded slice, final delta를 둔다. decode ledger에는 previous delta, current logical length, 만들어진 next position을 둔다. text-only에는 delta가 default 경로를 타고 image path에서만 특별해질 수 있다.

텍스트-only parity가 좋다는 사실은 multimodal position을 검증하지 않는다. image 한 장 prefill, 그 뒤 한 token decode를 최소 경계로 둔다. splice hidden까지 같고 prefill output도 같은데 decode에서 처음 갈리면 delta/cache-position handoff가 강한 후보다.

### 보안과 격리도 feature splice의 일부다

서로 다른 request의 flat vision feature buffer를 공유할 때 offset 오류는 정확성뿐 아니라 tenant isolation 문제다. request A placeholder가 request B feature slice를 읽으면 다른 사용자의 image 정보가 섞일 수 있다. shape와 dtype은 완전히 정상일 수 있다. request identity와 buffer range를 검증해야 하는 이유다.

cache key에서도 동일 text IDs만으로 multimodal prefix를 같다고 보지 않는다. image content digest, processor revision, grid/feature identity, adapter/model domain이 다르면 hidden과 KV가 다르다. 구체 cache 설계는 뒤 편에서 다루지만, placeholder IDs가 같은 image identity를 뜻하지 않는다는 불변식은 이 장에서 확정한다.

이 절의 종료 조건은 feature tensor를 보았다는 것이 아니다. content item이 어느 placeholder span에 어떤 feature rows와 position state로 결합되었는지 request별로 연결해야 한다. splice 뒤 first-layer input이 같으면 multimodal 입구는 닫히고, 뒤 cross/self-attention 또는 cache로 이동한다.

## 9.6 Qwen과 Gemma를 나란히 놓아 공통 축과 차이를 본다

Qwen3.5와 Gemma 계열 사례를 feature 목록으로 외우지 않는다. 먼저 공통 shape ledger를 쓴다.

```text
input_ids 또는 inputs_embeds
→ text embedding [tokens, H]
→ optional vision feature [vision_tokens, H]
→ merged hidden [model_tokens, H]
→ logical positions / multimodal positions
→ first decoder layer
```

그 다음 각 모델의 config가 어느 module과 state를 선택하는지 채운다. `vocab_size`와 `hidden_size`는 embedding table shape를 만든다. pad/image token ID는 lookup 또는 placeholder mask 분기를 만든다. embedding scale 또는 normalizer가 있으면 raw row에 곱해 첫 hidden magnitude를 바꾼다. multimodal grid와 merge size는 vision feature row 수와 model token 수를 바꾼다.

vLLM Gemma3 text model은 `VocabParallelEmbedding`을 만들고 embedding output에 normalizer를 곱한다. construction과 lookup은 [vLLM v0.27.1 `gemma3.py:303-343`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/gemma3.py#L303-L343)에 있다. 이 곱을 누락하면 ID와 row는 맞지만 첫 hidden magnitude가 달라진다. first divergence가 lookup 직후 scaling에서 나타나는 좋은 사례다.

이 설정을 인과로 쓰면 다음과 같다. hidden size field가 embedding second dimension을 바꾸고 checkpoint weight shape와 loader 검증을 바꾼다. image token ID field가 placeholder boolean mask를 바꾸고 feature insertion 위치를 바꾼다. embedding normalizer field 또는 model 상수는 lookup 뒤 multiplication을 만들고 모든 layer input scale에 영향을 준다. 반증 관측은 raw gathered row, scale 값, scaled hidden의 작은 slice다.

Qwen3.5의 text-only path에서는 embedding row 뒤 position과 hybrid layer 구성이 이어진다. multimodal wrapper에서는 image/video feature와 position delta가 더해진다. Gemma 계열도 text와 vision을 결합하지만 동일한 placeholder 수, grid 해석, position tensor rank를 가정하지 않는다. 공통 질문은 유지하되 답은 각 pinned model source에서 얻는다.

### tied embedding을 입구에서만 이해한다

입력 embedding과 LM head가 weight를 공유할 수 있다. 입력에서는 ID로 행을 gather하고 출력에서는 hidden과 모든 vocab row의 내적을 계산한다. 같은 parameter가 두 연산 역할을 맡는다는 뜻이다. “입력 token과 출력 token의 의미가 완전히 대칭”이라는 보장은 아니다. transformer가 그 사이에서 hidden을 바꾸고 output projection에 scale이나 별도 norm이 있을 수 있다.

또한 config에 `tie_word_embeddings=True`가 있다고 실제 storage tie를 무조건 단정하지 않는다. loader, quantization, pipeline partition, resize가 module wiring을 바꿀 수 있다. input/output weight object, shape, shard, storage identity를 확인한다. 출력 projection 비용과 distributed logits는 16장의 소유다. 이 장에서는 input row owner와 tie가 loading·memory에 주는 경계까지만 남긴다.

### 모델 이름보다 첫 hidden을 만드는 식을 비교한다

두 architecture를 비교할 때 먼저 `h0 = lookup(ids)`인지, `h0 = scale×lookup(ids)`인지, absolute position addition이 있는지, placeholder replacement가 어느 wrapper에서 일어나는지 적는다. 이 네 줄이 같아야 “embedding 단계가 같다”라고 말할 수 있다. class 이름이나 config JSON field 목록이 같다는 것은 증거가 아니다.

Gemma 계열의 embedding normalizer가 `sqrt(hidden_size)` 성격의 scale이라고 가정해 손계산해 보자. `H=4`, raw row `[1,-1,0.5,0]`이면 scale 2를 곱한 `[2,-2,1,0]`이 layer input이 된다. Qwen path에 같은 곱이 없다면 동일 raw row를 강제로 넣어도 first hidden은 다르다. 이 차이는 weight corruption이 아니라 architecture 의도다.

scale을 dtype cast 전에 곱하는지 뒤에 곱하는지도 numeric parity에 영향을 줄 수 있다. reference는 model-specific forward source다. 두 engine 중 하나가 scale을 layer norm에 흡수하거나 fused kernel에서 수행할 수도 있으므로 Python tensor hook 하나만으로 누락을 단정하지 않는다. 실제 first-layer input과 selected backend source를 연결한다.

### Qwen 사례를 config→state로 읽는다

`vocab_size`는 text embedding logical rows와 LM head output domain에 들어간다. `hidden_size`는 row width, first hidden의 마지막 축, 뒤 projection input dimension을 정한다. `pad_token_id`는 embedding constructor와 batch policy에 관여할 수 있다. multimodal image/video token IDs는 placeholder mask와 feature splice 분기를 정한다.

position 관련 field는 rotary module과 multimodal position helper를 선택한다. text-only default position과 grid-aware position을 같은 `[B,S]`라고 가정하지 않는다. `rope_deltas` 같은 state가 prefill에서 decode로 이어진다면 request lifecycle에 저장되어야 한다. config가 맞아도 이 state handoff가 빠지면 image prefill은 맞고 다음 token부터 틀릴 수 있다.

vLLM과 SGLang의 Qwen model은 serving에 맞춘 packed tokens, TP embedding, pipeline ownership을 갖는다. Transformers reference와 tensor layout이 다르더라도 request별 logical IDs, raw row, post-splice hidden, position 좌표가 같아야 model semantics가 맞는다. packed storage index 자체를 elementwise 비교하지 말고 request mapping으로 canonical order를 복원한다.

### Gemma 사례에서 scale과 modality를 분리한다

Gemma text path가 embedding scale을 적용한다면 text-only mismatch는 raw row와 scale 뒤 row를 나누어 비교한다. raw row부터 다르면 checkpoint/shard, scale 뒤부터 다르면 normalizer, 첫 layer 이후면 forward owner다. image path에서는 여기에 placeholder mask와 vision feature가 추가된다.

Gemma3 vLLM source의 `return self.embed_tokens(input_ids) * self.normalizer`는 짧지만 중요한 상태 전환이다. field 또는 derived constant가 `normalizer` 값을 정하고 forward multiplication이 hidden magnitude를 바꾸며 모든 뒤 layer가 이를 소비한다. 기대 효과를 “안정화”라고 막연히 쓰기보다 reference semantics라고 표현하고, 반증은 곱 전후 bounded slice다.

멀티모달 wrapper가 text model 밖에서 feature를 넣으면 text class만 비교해서는 image parity를 증명할 수 없다. processor와 vision tower, projector, placeholder scatter, text model 호출을 하나의 경로로 잇는다. Qwen과 Gemma가 둘 다 image를 받더라도 grid 계산과 position 정책은 각각 고정 source에서 확인한다.

### architecture 차이와 구현 버그를 구별한다

두 모델의 embedding norm이 다르거나 position 방식이 다른 것은 expected difference다. 같은 모델 checkpoint를 Transformers와 serving engine에서 비교할 때 expected formula가 다르면 구현 parity 문제다. 비교 축에 model family와 implementation을 동시에 바꾸면 원인을 분리할 수 없다.

먼저 한 모델·한 checkpoint에서 implementations를 비교하고, 그 다음 reference implementation 안에서 Qwen/Gemma를 비교한다. 전자는 correctness differential, 후자는 architecture contrast다. dtype, padding, multimodal fixture도 한 번에 하나씩 바꾼다. 이 순서가 없으면 Gemma scale 차이를 vLLM 버그로, Qwen multimodal delta를 model quality 차이로 오인한다.

tied weight도 같은 방식이다. 한 architecture에서 input/output storage가 실제로 tied인지 implementations 사이에 비교하고, 서로 다른 architectures가 tying을 선택했는지는 별도 사실로 둔다. 16장에서 LM head logits를 비교할 때 이 장의 input row evidence를 다시 사용한다.

## 9.7 shape ledger와 실패 재현으로 최초 divergence를 잡는다

독자가 유지할 최소 ledger는 다음 순서다.

```text
input_ids: shape, integer dtype, device, min/max ID
embedding weight: logical/physical rows, H, dtype, device, shard range
raw gathered rows: shape, dtype, owner rank
post-scale/splice hidden: model-token length, H, dtype, device
attention mask: input convention과 유효 길이
position state: logical IDs, cache position, multimodal axes/delta
first-layer input digest 또는 bounded slice
```

표를 채우는 목적은 텐서 정보를 많이 모으는 것이 아니다. 두 구현의 first divergence를 찾는 것이다. raw IDs부터 다르면 6~8장으로 돌아간다. IDs는 같고 gathered row부터 다르면 checkpoint, shard, dtype, weight mapping을 본다. gathered row는 같고 scale 뒤 다르면 model-specific normalizer다. text는 같고 splice 뒤 다르면 multimodal binding이다. first-layer input까지 같으면 embedding 가설을 버리고 10장의 residual forward, 12장의 QKV, 13장의 attention visibility, 14장의 RoPE/cache position으로 이동한다.

**실패 1 — added ID가 마지막 행을 넘는다.**

증상은 특정 special token이 들어간 요청만 index error를 내는 것이다. 경쟁 가설은 tokenizer-model revision 불일치, truncation 뒤 corrupted ID, TP local offset 오류다. 먼저 global max ID와 logical embedding row 수를 비교한다. 단일 rank에서도 실패하면 revision/resize가 강하고, TP에서만 실패하면 shard mapping을 본다.

`vocab_size` 숫자 하나로 종료하지 않는다. tokenizer가 낼 수 있는 실제 최대 ID, embedding logical rows, padded physical rows, 각 rank shard 범위를 적는다. physical padding row가 있다고 그 ID를 사용자 vocabulary로 허용해서는 안 된다. 수정 뒤에는 마지막 정상 ID, 첫 비정상 ID, added special IDs를 fixture로 고정한다.

**실패 2 — left padding 폭에 따라 같은 문장 결과가 달라진다.**

유효 IDs가 같은 두 batch를 만들고 pad 폭만 바꾼다. gathered valid rows가 같음을 먼저 확인한다. position IDs에서 유효 token이 `[0..n-1]`로 맞는지, padding mask가 pad key를 차단하는지 본다. position까지 같고 첫 attention 뒤 다르면 backend mask ABI로 이동한다. pad row를 zero로 만드는 것은 mask 수정의 대체물이 아니다.

**실패 3 — TP에서만 embedding이 어긋난다.**

global IDs가 shard 경계를 가로지르는 fixture를 쓴다. rank별 `[start,end)`, masked IDs, local gathered partial의 nonzero 위치, reduction 뒤 row를 비교한다. collective가 완료되었다는 log만으로 수치 parity를 증명하지 않는다. shard offset이 하나 밀리면 all-reduce도 정상이고 shape도 정상이다.

**실패 4 — image 두 장의 답이 서로 바뀐다.**

색과 내용이 명확히 다른 두 공개 fixture를 사용한다고 가정하되 여기서는 실행하지 않는다. request content index, placeholder span, pixel batch index, encoded feature slice, final embedding span의 binding을 정적으로 추적할 schema를 만든다. count가 모두 2여도 순서 mapping이 바뀔 수 있다. placeholder와 feature를 pair identity로 검증해야 한다.

text-only path와 image path의 raw text embedding은 같고 splice 뒤 특정 span만 다르면 vision/binding이 원인 후보다. splice까지 같고 position tensor부터 다르면 multimodal position helper로 이동한다. 모든 first-layer input이 같다면 뒤 vision cross-attention 또는 decoder layer를 본다.

**실패 5 — `inputs_embeds` 우회가 다른 모델의 행을 넣는다.**

일부 API와 model forward는 `input_ids` 대신 `inputs_embeds`를 받을 수 있다. 이때 embedding lookup과 ID range guard를 건너뛴다. shape `[B,S,H]`와 dtype이 맞아도 다른 model revision 또는 adapter domain에서 만든 벡터일 수 있다. caller가 어느 weight revision으로 만들었는지 identity가 필요하다.

input IDs와 inputs embeds를 동시에 줄 때 어느 쪽이 우선인지, 둘을 금지하는지 model API를 읽는다. Transformers Qwen3.5의 분기처럼 둘 중 하나를 선택하는 contract를 source에서 확인한다. 반증은 동일 weight로 직접 lookup한 row와 supplied embedding을 비교하는 것이다. supplied path만 틀리면 tokenizer나 embedding table 자체를 고칠 이유가 없다.

**실패 6 — config는 BF16인데 embedding output은 FP32다.**

증상만 보면 dtype option이 무시된 것처럼 보인다. 먼저 loaded embedding weight dtype, quantization wrapper, autocast context, lookup output, post-scale output을 분리한다. embedding을 의도적으로 fp32에 두는 architecture 또는 quant backend일 수 있고, cast가 first layer 직전에 일어날 수 있다. config string 하나와 최종 logits dtype만 비교해서는 분기 위치를 모른다.

field가 loader dtype 분기로 실제 전달되었는지, module replacement가 이를 덮었는지, weight storage와 compute output이 어떤지 source와 state로 닫는다. 기대 효과는 memory 또는 bandwidth 변화의 후보일 뿐이다. output이 fp32여도 뒤 즉시 bf16 cast하면 activation residence가 제한적일 수 있고, 반대로 table 전체가 fp32면 weight memory가 늘어난다.

반증 fixture는 같은 몇 IDs의 raw row를 reference dtype으로 읽고 cast order별 expected slice를 만든다. 허용 오차를 dtype에 맞게 정하고 exact equality를 강요하지 않는다. first divergence가 dequantization인지 scale multiplication인지 확인하면 kernel 장으로 넘길 증거도 작아진다.

**실패 7 — prefix cache hit 뒤 첫 decode만 틀린다.**

prefill first-layer input과 output이 reference와 같고 cache를 재사용한 다음 token부터 다르면 embedding weight 손상 가능성은 낮다. 새 token raw row, logical position, cache position, rope scaling domain, multimodal delta를 비교한다. raw row는 같은데 logical position이 cached length 대신 physical slot을 사용하면 position handoff가 최초 divergence다.

cache miss path와 hit path에 동일 suffix ID를 넣고 `(raw row, logical pos, first-layer input)`을 비교한다. raw row와 position이 같으면 이 장을 떠나 KV read/cache metadata와 attention으로 간다. cache hit ratio 자체는 correctness 증거가 아니다. 틀린 domain의 KV를 잘 재사용해도 hit는 높다.

**실패 8 — mixed batch 순서에 따라 다른 사용자의 이미지가 섞인다.**

request A는 image 한 장, B는 세 장이라고 하자. `[A,B]`와 `[B,A]` batch 순서를 바꾸고 request identity로 결과를 다시 정렬한다. placeholder spans, flat pixel ranges, feature ranges, position deltas가 각 request에 동일하게 귀속되어야 한다. flat buffer offset만 비교하면 batch 순서 때문에 값이 달라지는 것이 정상일 수 있으므로 canonical request mapping이 필요하다.

count assertion은 두 순서에서 모두 통과할 수 있다. 잘못된 prefix sum이 A에 B의 첫 feature를 주고 B에 나머지를 주어도 total count는 맞는다. request별 range 시작/끝과 content index를 검증한다. 이 failure는 개인정보 격리 위험이므로 silent fallback보다 hard validation을 우선한다.

### 정적 source walk를 실제 작업 순서로 만든다

첫째, model config class와 checkpoint config에서 vocab, hidden, pad/image IDs, tie, position fields를 적는다. 둘째, model `__init__`에서 embedding module과 derived scale을 찾는다. 셋째, forward의 `input_ids`/`inputs_embeds` 분기와 lookup, scale, placeholder splice를 순서대로 적는다. 넷째, position default와 multimodal state를 찾는다. 다섯째, first decoder layer 호출 인자를 기록한다.

serving engine에서는 그 앞에 model runner가 만든 packed IDs, positions, multimodal metadata를 붙이고, module 안에서는 TP/PP owner와 collective를 붙인다. llama.cpp에서는 동일 의미를 graph input, `ggml_get_rows`, position tensor, layer graph로 번역한다. 함수 이름을 맞추지 않고 state transition을 맞춘다.

각 source 좌표 옆에는 읽은 사실과 아직 추론인 부분을 나눈다. 예컨대 `ggml_get_rows`가 호출된다는 것은 row gather graph를 증명하지만 실제 GPU kernel이나 cache hit rate를 증명하지 않는다. `VocabParallelEmbedding`이 all-reduce path를 가진다는 것은 collective 가능성을 증명하지만 특정 deployment latency를 증명하지 않는다. 실행 금지 조건에서도 정확성을 유지하는 방법이다.

### 정적 검증으로 마지막 source walk를 준비한다

최소 text fixture는 shard boundary IDs, pad가 있는 두 길이, repeated ID를 포함한다. multimodal fixture schema는 image 수가 다른 두 request와 placeholder order를 포함한다. runtime을 수행하지 않더라도 expected shapes, legal ID ranges, shard owner, logical positions, feature/span count 식을 손으로 계산할 수 있다.

source에서 얻은 module shape가 이 ledger와 맞는지 대조한다. config `V,H`, TP size, physical padding rule에서 rank rows를 계산하고 constructor 인자와 비교한다. forward 분기에서 supplied `inputs_embeds`가 lookup을 우회하는지, scale이 어느 쪽에 적용되는지, multimodal mask가 어느 tensor에 scatter되는지 확인한다.

종료 조건은 링크 수가 아니다. token ID에서 first decoder layer input까지 모든 tensor의 owner, shape, dtype/device, position 의미, optional modality binding이 끊기지 않아야 한다. 한 칸이 source로 확인되지 않으면 가설로 표시하고 다음 조사 파일을 남긴다. 확인되지 않은 값을 그럴듯한 기본값으로 채우지 않는다.

## 9.8 gather dtype·TP shard·tied weight·position owner를 한 사건으로 묶는 실습

E9 사건은 tokenizer를 교체한 뒤 새 special ID가 드물게 들어오는 요청에서만 발생했다. Single GPU에서는 명시적인 index error가 났지만 tensor parallel 4-way 배포에서는 오류 없이 첫 token이 달라졌다. Log에는 global vocab size 128,256, embedding rows 128,256과 새 ID 128,256이 찍혔다. 마지막 유효 row index는 128,255이므로 새 ID는 한 칸 OOR다. 그런데 TP path는 local range mask와 reduction 때문에 잘못된 row가 0처럼 흘러 silent divergence를 만들었다.

첫 guard는 정수 범위다. 입력 ID tensor dtype이 int64인지 int32인지보다 먼저 `0 ≤ id < num_embeddings`를 확인한다. Negative ID가 unsigned/native conversion에서 큰 양수로 보일 수 있고, int64 ID를 int32 launcher argument로 줄이면 매우 큰 vocabulary나 sentinel이 overflow할 수 있다. 보통 vocabulary 크기에서는 두 dtype가 같은 값을 표현하지만 custom op schema와 index arithmetic의 dtype 계약은 별도로 확인한다. “PyTorch tensor가 int64”는 kernel 내부 index type까지 증명하지 않는다.

Dense gather 주소는 `base + id × row_stride + d × element_bytes`다. V=128,256, hidden=4096, BF16 2 bytes라면 row 하나는 8,192 bytes다. 마지막 valid ID 128,255는 base에서 1,050,? bytes 규모 offset을 가진다. 정확한 총 byte는 128,256×4,096×2 = 1,050,673,152 bytes다. ID 128,256은 정확히 allocation end를 가리켜 첫 element부터 OOR다. Row padding과 alignment가 있어 물리 allocation이 더 커도 semantic row가 생긴 것은 아니다.

TP vocab shard는 global ID를 local coordinate로 바꾼다. 균등 4-way라면 rank별 nominal range는 32,064 rows다. Rank 0은 [0,32064), rank 1은 [32064,64128), rank 2는 [64128,96192), rank 3은 [96192,128256)이다. Global ID g를 소유한 rank는 `start≤g<end`, local ID는 `g-start`다. 소유하지 않는 rank는 output row를 zero mask하고 collective sum으로 owning rank의 embedding을 복원할 수 있다.

OOR g=128,256은 어느 rank도 소유하지 않아야 한다. 모든 rank가 zero를 내고 reduction이 zero vector를 만들면 CUDA error는 없지만 잘못된 first hidden이 생긴다. 다른 구현은 마지막 rank에서 `local=g-start=32064`를 계산하고 local table end를 읽을 수 있다. 따라서 global range assertion은 shard mask 이전 공통 owner가 수행해야 한다. Local kernel마다 안전하다는 사실은 global ID가 vocabulary contract 안이라는 증거가 아니다.

비균등 shard와 padding도 고려한다. Kernel 효율을 위해 local rows를 multiple로 올림할 수 있다. V=128,257을 4-way로 나누며 각 rank 32,128 rows를 물리 할당하면 총 physical rows 128,512가 된다. Padding rows는 주소상 유효하지만 model vocabulary에는 없다. Range check를 physical padded end에 맞추면 invalid global ID가 dummy row를 읽고 silent pass한다. Manifest에는 logical global vocab, shard logical range, physical rows와 dummy initialization을 따로 둔다.

Tokenizer `len(tokenizer)`, config `vocab_size`, input embedding rows와 LM-head output rows도 서로 다른 순간에 갱신될 수 있다. Added token 뒤 tokenizer는 128,257인데 config와 weights가 128,256이면 input gather가 먼저 실패한다. Input embedding만 resize하고 untied LM head를 그대로 두면 prompt는 통과하지만 새 token logit row가 없다. Tied model은 resize helper가 shared parameter를 어떻게 재결합하는지 확인해야 한다. 네 숫자의 equality를 startup manifest와 checkpoint save/load 뒤 모두 검증한다.

Weight tying은 Python object identity 한 줄보다 넓다. Input embedding `E[V,H]`와 LM head가 `E` 또는 `Eᵀ` view를 공유할 수 있다. TP에서는 input gather가 vocab rows를 shard하고 output projection도 vocab rows를 shard하되 collective pattern은 다를 수 있다. Quantization/repack은 논리적으로 tied여도 서로 다른 physical representation을 만들 수 있다. Adapter가 input 또는 output 한쪽만 수정하면 effective tying도 깨진다. `tied_word_embeddings=true`는 loader와 runtime consumer가 같은 generation을 쓴다는 증거가 아니다.

Tied resize 계산도 명확히 한다. 새 row를 하나 추가하면 BF16 dense input weight는 8,192 bytes 증가한다. Untied output도 같이 늘리면 16,384 bytes이고 optimizer state는 serving 범위 밖이다. TP physical padding 때문에 실제 증가가 한 row가 아니라 alignment block일 수 있다. Memory delta가 예상보다 큰 것을 leak로 부르기 전에 shard padding과 repack workspace를 분해한다. 새 row initialization은 correctness 의미를 갖는다. Random, mean-resizing 또는 copied special row가 첫 logits를 바꾼다.

Position owner는 embedding owner와 분리한다. Absolute position embedding 모델은 token row와 position row를 더한다. RoPE 모델은 input 단계에서 token embedding만 만들고 attention projection 뒤 Q/K에 position transformation을 적용한다. Multimodal 모델은 position IDs 또는 delta를 processor/merge 단계에서 만들 수 있다. `position_ids=None`이 언제 arange/cache_position으로 materialize되는지 caller와 model forward를 이어 읽는다.

E9의 competing hypothesis는 세 개다. H1 tokenizer/config drift는 invalid global ID가 model entry 전에 존재할 것을 예측한다. H2 TP shard range bug는 IDs가 logical vocab 안이지만 shard boundary 값에서만 single/TP가 갈릴 것을 예측한다. H3 position owner drift는 token embedding gather까지 같고 token+position 또는 first attention input에서 갈릴 것을 예측한다. Checkpoint를 `final IDs→global/local range→gather row→position state→first residual`로 놓으면 세 가설을 순서대로 반증할 수 있다.

경계 fixture는 각 shard의 `start-1,start,end-1,end`, global `V-1,V`, negative sentinel을 포함한다. Valid ID는 정확히 한 rank가 소유하고 local index가 logical range 안이어야 한다. Invalid ID는 collective 전에 동일한 bounded error가 나야 한다. Rank마다 서로 다른 순간 실패하면 collective hang으로 확대될 수 있으므로 host/common validation과 all-rank error protocol도 설계한다. 이 원고에서는 실행하지 않고 필요한 predicate와 expected outcome만 정의한다.

Gather parity는 row 값을 직접 비교한다. 각 vocabulary row에 `row_id + dimension_fraction` 같은 sentinel을 넣는 정적 fixture를 생각하면 transpose, wrong shard와 off-by-one이 어느 coordinate에서 생겼는지 알 수 있다. Production weight 전체를 dump하지 않고 selected safe test rows와 digest를 사용한다. Quantized table이면 dequantized reference tolerance와 scale axis를 명시한다. Output zero vector만 보면 valid padding row인지 masked non-owner인지 구분할 수 없으므로 owner bit도 기록한다.

Position fixture는 padding/truncation 장의 네 tensor를 이어받는다. Same logical sequence가 single/batch, left/right supported layout, prefill/chunk/decode에서 같은 logical positions를 가져야 한다. Absolute table에서는 `position < max_position_embeddings`, RoPE에서는 position scaling/config generation과 cache position continuity를 본다. Decode 첫 step이 prompt length가 아니라 padded length를 쓰면 single/batch가 갈린다. KV cache block index와 logical position을 같은 숫자로 오인하지 않는다.

Pinned source walk는 모델의 embedding accessor에서 끝나지 않는다. Loader가 weight를 shard/resize/tie하는 producer, model forward가 IDs와 positions를 준비하는 caller, parallel embedding이 global range를 local gather와 reduction으로 바꾸는 consumer, LM head가 같은 또는 별 representation을 읽는 지점까지 잇는다. Transformers/vLLM/SGLang의 class 이름이 비슷해도 resize ownership과 distributed validation 위치는 다를 수 있다. llama.cpp는 GGUF vocabulary metadata와 embedding tensor shape, graph get_rows op와 CUDA backend까지 같은 logical row 계약으로 연결한다.

관측은 raw token sequence 전체를 metric label에 넣지 않는다. `id_range_result`, shard boundary bucket, owner_count, embedding_generation, position_owner/mode와 first-divergence stage를 bounded enum으로 둔다. Sampled trace에는 pseudonymous request, selected IDs digest, offending ID, V, shard range와 local index를 보존한다. OOR는 correctness hard failure이며 retry로 다른 rank에 보내 숨기지 않는다.

복구는 tokenizer rollback만 선택하지 않는다. Tokenizer가 의도적으로 새 ID를 만들었다면 model bundle/config/embedding/LM-head를 원자적으로 새 generation으로 배포하거나, 새 ID가 production admission에 들어오지 못하게 fence한다. Partial resize artifact와 old cache를 폐기한다. Tied weight가 repack됐다면 input/output consumer가 같은 generation인지 canary에서 확인한다. Old request가 new allocation을 읽지 않도록 generation별 drain을 둔다.

Canary는 ordinary IDs만으로 부족하다. V-1, 새 added ID, 각 TP 경계, tied/untied config, quant/adapter on/off, left/right batch와 decode position continuity를 포함한다. First hidden과 selected safe LM-head logits를 reference와 비교한다. Intended new row는 초기화 policy 때문에 old model과 같을 필요가 없지만, loader가 선언한 reference와 rank parity는 맞아야 한다. Unknown dummy row가 선택되는 것은 허용하지 않는다.

최종 terminal은 네 층이다. Vocabulary terminal은 tokenizer가 낼 수 있는 모든 admitted ID가 logical embedding/LM-head contract에 있다. Shard terminal은 valid global ID가 정확히 한 logical owner와 올바른 local row를 가진다. Tying terminal은 input/output producer와 consumer generation이 의도한 공유 또는 분리를 보존한다. Position terminal은 padding, cache와 modality를 거쳐 logical position owner가 연속성을 지킨다. 이 네 terminal이 닫혀야 첫 residual divergence를 attention/MLP 쪽으로 안전하게 넘길 수 있다.

E9에서 global OOR validation을 추가하고 tokenizer/model bundle을 맞춘 뒤 TP silent zero는 사라졌다고 하자. 그것은 H1을 지지하지만 H2 boundary fixture를 생략할 이유는 아니다. 기존 valid shard end에서 off-by-one이 별도로 존재할 수 있다. Position probe도 token gather 수정과 독립이다. 한 incident가 여러 guard를 추가하게 만들 수 있지만 각각은 자기 falsifier로 승인한다.

이 실습의 핵심은 embedding을 단순 lookup으로 축소하지 않는 것이다. Serving에서 한 ID는 tokenizer artifact, logical vocabulary, shard range, physical padded table, dtype/index arithmetic, tied output representation과 position owner를 통과한다. Shape가 맞고 주소가 유효해도 의미 row가 틀릴 수 있다. 각 변환의 input, owner, generation과 output을 적으면 GPU 오류와 silent wrong answer를 같은 OOR label로 뭉치지 않고 최초 잘못된 계약에서 멈출 수 있다.

### 9.8.1 작은 sentinel table로 wrong row를 눈에 보이게 만든다

추상적인 `[V,H]` 대신 V=8, H=4인 표를 손으로 만든다. row `r`의 값은 `[10r, 10r+1, 10r+2, 10r+3]`으로 둔다. ID `[0,3,7]`을 gather하면 기대 결과는 `[[0,1,2,3],[30,31,32,33],[70,71,72,73]]`이다. 이 패턴은 한 행 off-by-one, dimension stride 오류, transpose를 서로 다른 모양으로 드러낸다.

BF16 저장이라면 row bytes는 `4×2=8`, 전체 표는 64 byte다. row 7의 시작 offset은 56이고 마지막 element는 byte 62~63이다. ID 8의 시작 offset은 64로 allocation end다. CUDA allocation이 alignment 때문에 256 byte를 확보했더라도 logical ID 8이 유효해지는 것은 아니다. 메모리 접근 가능성과 model vocabulary 의미를 분리하는 가장 작은 예다.

TP=2로 row를 나누면 rank 0은 `[0,4)`, rank 1은 `[4,8)`을 소유한다. ID 3에 대해 rank 0 local=3, rank 1은 non-owner다. ID 4는 rank 1 local=0이다. 각 rank가 `(owner_bit, local_id, partial_vector)`를 내고 sum이 reference row와 같은지 확인한다. OOR ID 8은 owner count가 0이어야 하고 collective 전에 공통 오류가 나야 한다. owner count 0인 zero vector를 정상 embedding으로 흘려보내면 silent corruption이다.

physical rows를 rank마다 8의 배수로 padding한다고 하자. rank 1은 logical 4 rows지만 physical 8 rows를 가질 수 있다. ID 8을 rank 1 local=4로 잘못 매핑하면 주소는 유효하고 dummy row를 읽는다. sentinel fixture는 dummy를 0이 아닌 poison pattern으로 채워 잘못된 성공을 즉시 드러낼 수 있다. production dummy initialization을 poison으로 바꾸라는 뜻은 아니다. 정적/테스트 artifact에서 logical guard가 physical padding에 기대지 않는지 확인하는 장치다.

### 9.8.2 index dtype의 범위와 산술 dtype을 분리한다

입력 tensor가 int64여도 custom CUDA kernel이 index를 int32로 cast할 수 있다. ID 자체가 int32 범위 안이어도 `id × row_stride`가 32-bit에서 overflow할 수 있다. V=128,256, row stride=8,192 byte일 때 마지막 row offset은 약 1.05GB라 signed int32 범위 안이다. H=16,384 BF16이면 stride=32,768이고 같은 vocab의 offset은 약 4.20GB라 32-bit signed byte offset을 넘는다. pointer arithmetic이 64-bit인지 확인해야 한다.

반대로 kernel이 element index를 사용하면 계산 경계가 달라진다. `id × H + d`의 최대값을 어떤 type으로 계산하는지 본다. launcher schema, C++ dispatch, CUDA template parameter, device local variable을 이어 읽는다. Python dtype 한 줄만으로 내부 산술을 추정하지 않는다. 음수 sentinel은 host validation에서 막혀야 하며 unsigned 변환 뒤 huge index가 되도록 두지 않는다.

dtype 회귀 fixture는 값 범위와 곱셈 범위를 나눈다. ID `-1,0,V-1,V`는 semantic range를, 큰 synthetic stride와 boundary ID는 address arithmetic을 시험한다. 실제 거대 weight를 실행하지 않아도 source의 type과 상한 식을 검산할 수 있다. 구현이 chunked address 또는 64-bit pointer를 쓰면 그 사실을 기록하고, 확인하지 못했으면 가설로 남긴다.

### 9.8.3 tied weight는 공유 선언이 아니라 양쪽 consumer 계약이다

input embedding은 ID로 row를 gather하고, LM head는 hidden vector와 vocabulary row의 내적을 계산한다. 같은 parameter를 공유해도 access pattern은 다르다. TP input은 한 owner의 row를 복원하기 위해 partial을 sum할 수 있고, output은 각 rank가 vocabulary shard logits를 만들고 sampling을 분산 수행할 수 있다. 따라서 “weight tied”는 collective와 layout까지 같다는 뜻이 아니다.

V=8,H=4 sentinel에서 hidden `[1,0,0,0]`을 LM head에 넣으면 logit r은 row의 첫 값 `10r`이 된다. input ID 7이 `[70,71,72,73]`을 내고 output logit 7이 70인지 확인하면 양쪽 representation이 같은 logical rows를 보는지 검산할 수 있다. output만 한 행 밀리면 input gather는 맞아도 token probability가 다른 ID에 귀속된다.

resize 전 V=8에서 새 token ID 8을 추가했다고 하자. input을 9 rows로 늘리고 LM head가 8에 남으면 prompt에 새 ID가 들어올 때 embedding은 성공하지만 모델은 새 token을 생성할 logit을 갖지 않는다. 반대로 output만 늘면 새 token을 생성할 수 있다고 보이지만 다음 step input gather가 실패한다. tied helper가 두 module을 다시 같은 storage/view로 묶는지, checkpoint load가 tying을 재적용하는지, quantization이 별 packed copy를 만드는지 확인한다.

adapter도 effective identity를 바꾼다. input embedding LoRA나 vocabulary extension module과 output adapter가 동일 정책으로 적용되는지 본다. base parameter object가 같아도 한 consumer 앞에만 delta가 붙으면 effective rows는 tied가 아니다. cache namespace와 trace에는 base weight generation뿐 아니라 adapter set과 merge state가 필요하다.

### 9.8.4 position owner를 token row의 owner와 별도로 추적한다

absolute embedding에서는 token row `E[id]`와 position row `P[pos]`를 더해 first hidden을 만든다. sentinel을 `E[id]=[10id,...]`, `P[pos]=[100pos,100pos,...]`로 두면 ID 3, pos 2의 첫 값은 230이다. 결과가 330이면 physical padded position 3을 썼다는 식으로 owner 오류가 보인다.

RoPE 모델에서는 first hidden에 position row를 더하지 않을 수 있다. position은 attention layer의 Q/K transformation에서 소비된다. 이 경우 embedding output이 같다고 position contract가 닫힌 것은 아니다. model runner가 만든 `positions` 또는 `cache_position`, model forward가 전달한 값, rotary module이 적용한 scaling과 section을 이어 읽는다. 9장에서는 owner와 handoff를 고정하고 구체 회전 수학은 뒤 장으로 넘긴다.

multimodal 모델은 image grid와 merge policy에서 position delta를 만들 수 있다. text token ordinal만으로 다음 decode position을 정하면 image span이 차지한 logical range와 어긋날 수 있다. processor, model wrapper, decode state 중 누가 delta를 소유하는지 확인한다. `position_ids=None`이라는 API 값은 position이 없다는 뜻이 아니라 downstream default owner에게 생성을 위임했다는 뜻일 수 있다.

### 9.8.5 wrong-row 사건의 first divergence를 단계별로 닫는다

E9를 재구성하면 요청은 새 ID V를 포함했다. tokenizer output에서 이미 `max_id == V`이므로 첫 위반은 model gather 이전 global range다. TP zero vector와 이상한 first token은 downstream 증상이다. 공통 validation을 추가한 뒤 오류가 모든 rank에서 collective 전에 동일하게 반환되는지 확인한다. 단순히 CUDA OOR가 사라졌다는 것으로 승인하지 않는다.

두 번째 fixture는 valid shard boundary ID를 넣는다. ID `start_r`은 rank r local 0, `end_r-1`은 마지막 logical row다. owner count가 정확히 1이고 reduced vector가 dense reference와 같아야 한다. 이 fixture가 실패하면 OOR 수정과 별개의 shard mapping 버그다. 세 번째 fixture는 same IDs에 position만 바꿔 raw gather가 같고 first position-dependent state가 기대대로 달라지는지 본다.

마지막으로 tied output을 확인한다. selected safe hidden에 대한 boundary logits를 input row reference와 비교하고, quantized/adapter lane은 각자의 허용 오차와 effective generation을 명시한다. `first hidden이 맞다`와 `LM head가 같은 row identity를 쓴다`는 별 terminal이다.

incident 보고서는 다음처럼 쓴다. “Tokenizer generation T2가 global ID 128,256을 생성했으나 model generation M1의 logical V는 128,256이었다. TP mask는 owner count 0을 error로 만들지 않고 zero vector를 all-reduce해 first residual의 최초 차이가 됐다. 공통 admission range guard와 atomic T2/M2 bundle을 적용했고, V-1/V 및 모든 shard 경계 fixture와 tied-output row identity를 통과했다.” 이 문장은 원인, 전파, 수정, 반증을 모두 담는다.

## 9.9 source walk와 배포 승인표를 하나의 실습으로 만든다

처음 보는 모델을 검토할 때 config에서 `vocab_size`, `hidden_size`, tying, maximum/rope fields를 적는다. tokenizer length와 special ID map을 옆에 둔다. checkpoint tensor index에서 input embedding과 LM-head shape, dtype, quantization/packing을 확인한다. config 숫자만 믿지 않고 실제 tensor shape와 loader가 resize/pad하는 경계를 찾는다.

그 다음 model constructor에서 embedding module을 만든 인자를 읽는다. PP first rank만 module을 소유하는지, TP가 logical/original/added vocab을 어떻게 나누는지, physical padding multiple이 무엇인지 기록한다. loader가 global weight를 shard하는 함수와 forward가 global IDs를 local gather로 바꾸는 함수를 잇는다. constructor와 forward 사이에 같은 range convention이 쓰이는지 boundary 식으로 검산한다.

forward에서는 `input_ids`와 `inputs_embeds` 분기를 분리한다. 후자는 gather를 우회하므로 ID range fixture가 실행되지 않을 수 있다. multimodal wrapper가 먼저 text embedding을 만든 뒤 placeholder에 feature를 scatter하는지, feature를 만든 뒤 model에 `inputs_embeds`로 넘기는지에 따라 guard owner가 달라진다. scale, cast, device 이동의 순서도 first hidden 수치를 바꾼다.

position은 default 생성 지점을 찾는다. caller가 explicit tensor를 주는 lane과 `None`에서 materialize하는 lane, prefill과 decode, cache hit/miss, graph/eager가 같은 logical position을 만드는지 표로 둔다. absolute table이면 range와 gather를, RoPE이면 Q/K consumer까지 handoff를, multimodal이면 delta와 grid provenance를 적는다.

output 쪽에서는 tied flag, `get_output_embeddings`, loader tie hook, LM-head parallel layout과 sampling consumer를 찾는다. input과 output의 logical vocabulary generation이 같은지, added vocab와 dummy rows를 동일하게 제외하는지 확인한다. weight object identity가 다르더라도 의도된 quantized representation일 수 있고, object identity가 같더라도 adapter가 한쪽에만 적용될 수 있다. 판정은 effective row와 consumer generation으로 한다.

승인표에는 여섯 equality와 두 intentional difference를 둔다. tokenizer admitted max ID < logical input rows, config logical vocab = loader logical vocab, shard union = global logical range, shard intersection = empty, valid ID owner count = 1, input/output token identity map = same이다. physical padded rows와 logical rows, input/output physical layout은 의도적으로 다를 수 있다. 차이를 오류로 지우지 말고 변환 함수와 guard를 적는다.

작은 계산도 함께 둔다. BF16 dense bytes는 `V×H×2`, untied input+output은 대략 두 배다. TP rank bytes는 padding과 quantization을 반영한다. 새 token N개가 늘 때 logical delta `N×H×element_bytes`와 physical alignment delta를 분리한다. 관측 memory가 이보다 크면 repack workspace, duplicate generation, temporary load buffer를 다음 후보로 본다.

canary는 보통 ID와 경계 ID를 섞는다. `[0,1,start_r-1,start_r,end_r-1,V-1]` 중 각 rank에 유효한 집합을 만들고 repeated ID로 stable gather도 본다. invalid `-1,V`는 bounded error를 기대한다. same logical sequence를 single/batch와 cache miss/hit로 보내 position owner를 비교한다. multimodal은 image 수가 다른 두 request의 순서를 바꾸고 request별 span을 canonical order로 복원한다.

관측에는 `logical_vocab`, `physical_rows`, `shard_start/end`, `owner_count`, `embedding_generation`, `position_mode/owner`, `tied_effective_generation`, `first_divergence_stage`를 bounded field로 남긴다. 전체 weight나 사용자 token열을 metric에 넣지 않는다. incident trace에는 offending ID와 safe fixture row digest, shape/dtype를 보존한다.

rollback은 tokenizer만, model만, cache만 따로 되돌리지 않는다. artifact manifest가 tokenizer, config, input embedding, LM head, adapter/quant state, position config를 한 generation으로 묶는다. old/new PP·TP worker가 섞이지 않도록 router fence를 두고, in-flight request와 prefix cache가 어느 generation을 소유하는지 확인한다. OOR fix 뒤 old KV를 재사용하면 position/weight identity가 달라질 수 있다.

이 실습의 종료 조건은 네 stack의 class 이름을 표로 많이 모으는 것이 아니다. 한 admitted ID가 어느 global contract에서 검증되고, 어느 rank가 어떤 local row를 읽고, 어떤 collective와 scale을 지나 first hidden이 되며, position state가 어디서 합류하고, 같은 logical vocabulary가 output projection에서 어떻게 소비되는지를 끊김 없이 설명하는 것이다. 빈 경계는 `unknown`으로 남기고 다음 source symbol을 지정한다.

검토 중 가장 흔한 오판은 checkpoint shape가 맞으니 runtime row도 맞을 것이라는 가정이다. loader는 tensor를 transpose, shard, pad, quantize, repack할 수 있다. lazy loading이나 memory mapping은 view의 stride와 device placement를 바꾼다. 따라서 artifact tensor `E[V,H]`, loaded logical view, kernel physical layout을 별 행으로 둔다. 각 변환의 입력 shape, 출력 shape, row identity 보존식, generation을 기록한다.

예를 들어 원본 BF16 `[8,4]`를 TP=2로 `[4,4]`씩 나눈 뒤 kernel alignment 때문에 `[8,4]` physical buffer로 repack한다고 하자. rank 1의 logical row 0은 global row 4이고 physical row 0에 있을 수도, metadata header 뒤 row 2에 있을 수도 있다. `local_id=0`이 곧 byte offset 0이라는 가정은 layout descriptor가 증명해야 한다. custom op caller가 stride와 offset을 어떻게 전달하는지 consumer까지 읽는다.

quantized embedding은 row마다 또는 group마다 scale/zero-point를 가질 수 있다. wrong row가 발생하면 code bytes뿐 아니라 scale row도 같은 logical owner에서 선택돼야 한다. code는 row 7, scale은 row 6을 읽으면 값은 유한하고 shape도 맞지만 first hidden이 조용히 틀린다. fixture에는 dequantized reference뿐 아니라 selected scale/group index를 넣는다. 허용 오차는 올바른 row 안의 양자화 오차를 허용할 뿐 wrong-row를 숨기는 용도가 아니다.

pipeline parallel에서는 embedding이 첫 stage에만 있고 뒤 stage는 `inputs_embeds` 또는 hidden state를 받는다. 잘못된 요청이 middle stage에 직접 들어가거나 pipeline routing generation이 어긋나면 first stage의 range guard가 우회될 수 있다. stage contract에는 payload type, logical sequence order, position state, model generation을 넣는다. 뒤 stage가 integer IDs를 받지 않는다면 거기서 vocabulary validation을 반복할 필요는 없지만, first-stage validation을 통과한 provenance는 보존해야 한다.

replica와 expert parallel도 관측을 흐릴 수 있다. 같은 request를 다른 replica에 retry했을 때 한쪽만 틀리면 weight generation, shard mapping, adapter merge가 replica별로 다른지 본다. MoE routing은 embedding 뒤이므로 raw gather와 first hidden이 같다면 그 층으로 넘긴다. 모든 분산 기능을 embedding 원인으로 끌어들이지 않고 최초 divergence를 기준으로 범위를 제한한다.

prefix cache와 embedding identity도 계산해 보자. cached prefix의 token IDs가 같아도 model weight generation M1에서 만든 KV를 M2가 읽으면 의미가 다르다. input embedding row 하나가 바뀌면 첫 layer부터 모든 cached K/V가 바뀐다. cache key 또는 namespace는 model/checkpoint, adapter, quantization, position/rope semantics를 포함해야 한다. tokenizer digest만 같다는 사실은 KV compatibility를 증명하지 않는다.

새 special row 초기화 정책은 품질과 안정성 양쪽에 영향을 준다. zero initialization은 OOR silent zero와 구분하기 어렵고, random initialization은 첫 logits에 큰 변동을 만들 수 있다. 기존 row 평균을 쓰는 정책도 구현과 dtype에 따라 결과가 다르다. manifest에는 initialization method와 seed 또는 source row, resize 후 tie/repack generation을 기록한다. canary expected value는 이 선언에서 만든다.

position fixture는 긴 context 경계도 포함한다. maximum-1, maximum, scaling threshold 전후, sliding-window eviction 전후를 선택한다. absolute table OOR는 명시적 error여야 하고, RoPE scaling lane은 config에 따라 연속된 logical position을 사용해야 한다. overflow를 modulo로 감싸거나 physical cache slot을 position으로 재사용하면 주소는 유효해도 의미가 틀린다.

chunked prefill에서는 100-token prompt를 40,40,20으로 나눴다고 하자. 각 chunk local row는 0부터 시작하지만 logical positions는 0~39,40~79,80~99여야 한다. 마지막 chunk 뒤 decode 첫 position은 100이다. cache hit 60이 있으면 계산 chunk는 60~99부터 시작할 수 있다. chunk-local offset, cached length, logical position, slot mapping을 한 식에 섞지 않는다.

multimodal position에서는 text token count와 model position increment가 다를 수 있다. image placeholder 하나가 feature 256개로 확장되거나 grid/merge 정책이 별 delta를 만들 수 있다. final text IDs만 보고 decode position을 정하면 어긋난다. processor가 만든 grid, placeholder span, merged feature length, model이 반환하거나 보존한 position delta를 source walk에 넣는다. batch reorder 뒤 request별 delta가 섞이지 않는지도 본다.

wrong-row를 모니터링하는 fleet metric은 직접적인 weight 비교를 상시 수행하지 않는다. startup/canary에서 logical/physical shape, shard coverage, tie generation을 검증하고 counter를 노출한다. runtime에는 range validation failure, owner-count violation, position discontinuity, feature binding rejection을 센다. first-hidden digest는 승인된 synthetic probe에서만 사용해 비용과 데이터 노출을 제한한다.

장애 시 full embedding weight를 dump하는 것은 느리고 민감하며 대개 불필요하다. offending ID와 양옆 safe rows의 shape/dtype, cryptographic digest, bounded slice를 각 rank에서 얻고 loader manifest와 비교한다. tied output도 같은 logical rows의 digest를 비교한다. quantized representation은 raw packed digest와 dequantized safe slice를 분리한다. 이 증거로 artifact corruption, mapping error, consumer error를 좁힌다.

성능 최적화도 correctness artifact 위에서 평가한다. fused gather+scale, quantized lookup, sharded reduction을 바꿀 때 dense reference와 boundary IDs의 first hidden을 비교한다. throughput과 memory bandwidth만 보고 승인하지 않는다. row deduplication이나 cache가 repeated ID를 최적화한다면 repeated/non-repeated fixture가 같은 logical output을 내는지 본다.

최종 handoff 문장은 다음 층의 시작점을 명확히 한다. raw gathered row가 다르면 loader/shard/gather에 남는다. raw row는 같고 scale/cast 뒤 first hidden이 다르면 embedding postprocess를 본다. first hidden과 positions handoff가 같고 Q/K부터 다르면 attention projection/rotary로 간다. 모든 decoder hidden이 같고 vocabulary score만 다르면 tied LM head와 logits 장으로 간다. 이 경계 이동 규칙이 무작정 CUDA kernel 전체를 뒤지는 일을 막는다.

독자가 검토표를 채울 때 `shape OK`라는 단일 체크는 금지한다. logical shape, physical shape, stride, storage offset, dtype, device, layout/repack version을 나눈다. `[4,4096]` 두 tensor도 한쪽은 contiguous BF16이고 다른 쪽은 quantized packed view일 수 있다. shape equality는 row identity와 값 equality의 필요조건 일부일 뿐이다.

주소 계산도 단위를 붙인다. row stride가 element인지 byte인지, local ID가 signed인지, pointer base가 shard allocation 시작인지 global view 시작인지 적는다. `offset=id*stride`만 쓰면 32-bit overflow와 element-byte 혼동을 놓친다. 작은 sentinel 계산과 production 상한 계산을 함께 두면 구현의 type 선택을 검토할 수 있다.

분산 error protocol은 correctness의 일부다. rank 0만 OOR를 발견하고 예외를 던지는 동안 다른 rank가 all-reduce에 들어가면 단순 입력 오류가 hang으로 확대된다. 가능한 한 collective 이전 공통 validation을 수행하고, rank-local 조건이라면 모든 rank가 일관된 terminal로 수렴하도록 설계한다. timeout은 원인을 해결하지 않고 증상을 늦게 끊을 뿐이다.

startup validation은 tokenizer가 낼 수 있는 전체 vocabulary와 model rows를 비교하되 dynamic added token과 request별 adapter를 고려한다. 단순 `len(tokenizer)==num_embeddings`가 항상 정답은 아니지만 admitted ID set이 logical rows의 부분집합이어야 한다는 조건은 유지된다. 의도적으로 미사용인 output-only row나 reserved row가 있다면 manifest에 방향과 소비자를 명시한다.

weight tying 검증에는 save/load round trip의 의미도 들어간다. runtime에서 두 module을 tie했어도 checkpoint saver가 하나만 기록하는지 둘을 복제하는지, loader가 다시 tie하는지에 따라 재시작 뒤 identity가 달라질 수 있다. binary artifact diff에서는 tensor name, shape, storage alias 정보와 loader hook을 함께 본다. serving 재시작 후에만 logits가 달라지는 사건은 이 경계를 의심할 근거가 된다.

position owner 변경은 model config diff에만 나타나지 않을 수 있다. generation helper나 runner가 default tensor를 만들도록 책임이 이동하면 같은 checkpoint에서도 결과가 달라진다. revision 비교에서는 config field, helper signature, default branch, cache update caller를 함께 본다. 변경 전후 same fixture의 `(token_id, logical_position, cache_position)` ledger가 가장 직접적인 증거다.

마지막 승인 회의에서는 네 문장을 읽는다. “admitted global ID는 collective 전에 logical range로 검증된다.” “valid ID는 정확히 한 shard owner와 올바른 local row를 가진다.” “input과 output vocabulary consumer는 선언된 tying/adapter/quant generation을 공유한다.” “position owner는 padding, chunk, cache hit, modality를 거쳐 logical continuity를 보존한다.” 한 문장이라도 source와 fixture로 뒷받침되지 않으면 해당 칸은 미확인이다.

이 네 문장을 option 설명과도 연결한다. tensor parallel size를 바꾸면 shard range와 collective가, quantization을 바꾸면 physical representation과 dequantization dtype이, adapter를 바꾸면 effective input/output row가, RoPE scaling을 바꾸면 position consumer가 변한다. option 이름이나 권장값만 적지 않고 parse된 값이 module construction과 forward metadata를 어디서 바꾸는지 mutation chain을 남긴다.

효과 판정은 각 option의 owner state에서 시작한다. TP 증가 뒤 메모리가 줄었는지 보기 전에 shard coverage와 boundary parity를 확인한다. quantized embedding이 빨라졌는지 보기 전에 sentinel row와 scale axis를 비교한다. adapter hot-swap이 성공했다는 응답 전에 input/output generation과 old cache 격리를 확인한다. position scaling으로 긴 context가 수용됐다는 사실 전에 threshold 전후 logical continuity와 first attention handoff를 검증한다.

마지막으로 failure fixture를 문서 예제로만 두지 않는다. artifact manifest와 함께 version control에 보존하고 source revision이 바뀔 때 expected owner와 symbol을 재확인한다. 함수가 이동해도 invariant가 유지되면 mapping만 갱신한다. invariant가 바뀌면 새 행동, migration, cache compatibility와 rollback 조건을 기록한다.

이제 9장의 독자는 ID 하나가 어느 byte 주소를 선택하는지 계산할 수 있고, TP에서 누가 그 row를 소유하는지 판정하며, tied output과 position handoff가 같은 generation인지 조사할 수 있다. 이 기준선이 있어야 다음 장의 logits 차이를 embedding 문제로 잘못 돌리지 않는다.

재현이 어려운 사건에서는 최소한 failing ID의 범위, rank별 logical interval, physical row 수, selected owner count, position generation을 보존한다. 값 원문을 남길 수 없다면 safe synthetic row와 동일 경계를 사용한다. passing neighbor는 평범한 ID가 아니라 같은 shard 끝의 `V-1`처럼 한 조건만 다른 사례를 고른다.

조사 종료 뒤에는 잘못된 첫 가설도 기록한다. 예를 들어 CUDA OOR만 예상했지만 physical padding 때문에 silent dummy row가 읽혔다면, 그 반례는 다른 padded table과 quantized kernel 검토에도 재사용된다. 지식은 정답 목록보다 어떤 관측이 가설을 깨뜨렸는지에서 더 잘 이전된다.

확정하지 못한 kernel 내부 산술은 추론으로 표시하고, 확인에 필요한 launcher signature와 device symbol을 다음 작업으로 남긴다. 정적 source walk의 한계를 숨기지 않는 것이 과도한 확신보다 정확한 인수인계를 만든다.

## 9.10 네 스택의 소스 경로와 장말 회고

Transformers에서는 model class의 `__init__`에서 embedding module이 어떤 config field로 만들어지는지 보고, forward에서 `input_ids`와 `inputs_embeds` 분기를 따라간다. position default가 어디서 생기고 multimodal wrapper가 어느 mask로 feature를 넣는지 이어서 본다. high-level `generate`부터 시작하면 첫 hidden의 owner를 놓치기 쉽다.

vLLM에서는 model-specific class만 읽지 않는다. `VocabParallelEmbedding`의 global/local vocabulary mapping과 forward reduction을 먼저 보고 model class가 이를 어떻게 구성하는지 본다. physical padding, original vocab, added vocabulary가 별도 범위로 취급되는지도 확인한다. engine의 packed input과 position은 model runner 경계에서 들어오므로 logical position과 cache slot을 구별한다.

SGLang에서는 pipeline first rank와 `input_embeds` 우회 분기, vocab-parallel module, multimodal processor/model wrapper를 잇는다. SGLang Qwen source의 embedding accessor와 pipeline ownership은 [SGLang v0.5.18 `qwen3_5.py:1390-1434`](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_5.py#L1390-L1434)에 고정된다. server option이 module class와 shard state를 실제로 바꿨는지는 loaded module과 rank range로 반증한다.

llama.cpp에서는 PyTorch `nn.Embedding` 이름을 찾는 대신 graph의 row gather를 본다. input token tensor를 만들고 `ggml_get_rows`로 token embedding weight에서 행을 고르는 경계는 [llama.cpp v0.2.0 `llama-graph.cpp:2285-2325`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L2285-L2325)에 있다. 위치 tensor와 batch input은 graph builder가 별도로 소유한다. API가 다르더라도 “정수 ID→weight row→first activation”이라는 의미 좌표로 비교할 수 있다.

### 고정 소스 노트

구현 좌표는 Transformers v5.15.1 commit `550d7b3834670483a4df436541272c055dc364bf`, vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, llama.cpp v0.2.0 commit `bb4caa7540188872173c44d161602d9271386413`에 고정했다. 다음 항목은 이 코드를 정적으로 읽은 좌표이며 모델이나 서버를 실행해 수치를 얻었다는 뜻은 아니다.

- Transformers v5.15.1 — [Qwen3.5 embedding construction과 first forward](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1130-L1185)
- Transformers v5.15.1 — [Qwen3.5 multimodal feature 결합](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1668-L1745)
- Transformers v5.15.1 — [Gemma3 text embedding 경계](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L560-L625)
- Transformers v5.15.1 — [Gemma3 conditional generation의 image feature 결합](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L770-L890)
- vLLM v0.27.1 — [`VocabParallelEmbedding` 생성](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/vocab_parallel_embedding.py#L198-L270)
- vLLM v0.27.1 — [vocab shard index 계산](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/vocab_parallel_embedding.py#L351-L377)
- vLLM v0.27.1 — [Qwen3.5 embedding과 weight tie](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5.py#L231-L337)
- vLLM v0.27.1 — [Gemma3 embedding scale](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/gemma3.py#L303-L343)
- SGLang v0.5.18 — [Qwen3.5 pipeline embedding construction](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_5.py#L1340-L1362)
- SGLang v0.5.18 — [Qwen3.5 input embedding 분기](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3_5.py#L1410-L1434)
- llama.cpp v0.2.0 — [token embedding row gather](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L2285-L2325)
- llama.cpp v0.2.0 — [embedding tensor callback과 graph input](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-graph.cpp#L1520-L1540)

이 장의 핵심 인과는 짧다. integer ID가 weight row를 선택하고, dtype/device/shard가 그 row의 물리적 소유권을 정한다. padding mask와 position state는 어느 row가 어느 순서의 문맥으로 들어갈지 정한다. multimodal wrapper는 placeholder span을 다른 modality feature와 결합한다. 이 상태가 모두 맞아야 첫 decoder layer의 입력이 맞다.

실무에서 이 인과를 한 문장으로 보고할 수 있어야 한다. “모델 입력이 다르다” 대신 “동일 global IDs에서 TP rank 1의 vocab start가 한 행 밀려 shard-boundary token의 raw gathered row가 최초로 달랐고, all-reduce 뒤 first-layer input에 전파되었다”라고 쓴다. padding 사고라면 “raw rows는 같았으나 left-pad 폭을 logical position에 포함해 유효 token position이 이동했다”라고 쓴다. multimodal 사고라면 “placeholder 수는 같았으나 request reorder 뒤 feature range prefix sum이 content order와 분리되어 다른 image span이 splice되었다”라고 쓴다.

좋은 설명은 수정과 반증도 포함한다. shard metadata를 바로잡은 뒤 경계 IDs에서 rank partial과 global result를 비교한다. position helper를 고친 뒤 pad 폭이 다른 fixture의 유효 first-layer input parity를 확인한다. feature binding을 고친 뒤 ragged image batch의 순서를 바꾸어도 request별 span identity가 유지되는지 본다. 최종 logits만 같다는 우연에 기대지 않고 원인과 같은 상태를 직접 검증한다.

반대로 first-layer input이 byte 단위로 또는 dtype 허용 오차 안에서 같다면 embedding을 더 오래 의심하지 않는다. Q/K/V projection에서 처음 갈리면 12장, mask와 attention score면 13장, rotary와 cache position이면 14장으로 이동한다. 마지막 hidden까지 같고 vocab score만 다르면 16–17장이다. 이 handoff가 있어야 source를 깊게 읽는 작업이 전체 repository를 무작정 헤매는 일로 변하지 않는다.

이때 digest 하나만 같다고 수치 parity를 선언하지 않는다. digest 생성 dtype과 layout, canonical request order를 고정하고, 작은 bounded slice와 shape ledger를 함께 보존한다. packed tensor는 request별 logical order로 복원한 뒤 비교한다. multimodal feature 원본을 공개할 수 없다면 승인된 synthetic fixture와 content digest, grid, span metadata를 남긴다. 증거는 재현 가능해야 하지만 사용자 데이터를 과도하게 보존해서는 안 된다.

또한 오류가 나지 않았다는 사실은 계약 충족이 아니다. physical vocab padding row를 읽거나 잘못된 image feature를 같은 shape로 삽입하면 graph는 정상 종료한다. 이 장의 guard가 range뿐 아니라 logical ownership, position identity, modality binding을 확인하는 이유다.

다음에 어디를 파야 하는지도 first divergence가 정한다. ID가 틀리면 6~8장으로 돌아간다. embedding row와 first-layer input이 맞으면 10장의 residual forward로 간다. position이 Q/K에 적용되는 구체 수학은 14장, attention visibility는 13장이다. 모든 layer를 지난 hidden이 맞는데 다음 token 점수가 다르면 16–17장의 LM head와 logits로 간다. 경계를 지키면 embedding이라는 한 단어에 tokenizer, model, attention, output 문제를 모두 뭉개지 않는다.

같은 작은 fixture를 유지해야 수정 뒤에도 이 최초 분기점과 ownership 계약을 다시 정확히 검증할 수 있다.
