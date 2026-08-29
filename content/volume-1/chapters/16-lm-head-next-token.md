# 16장. 마지막 hidden state가 다음 token이 되는 경계

모델의 마지막 decoder layer가 끝났다고 다음 token이 이미 나온 것은 아니다. 그 시점에 있는 값은 여전히 hidden width `D`를 가진 연속 벡터다. 이를 final normalization이 정돈하고, LM head가 vocabulary 크기 `V`의 score로 투영한다. processor가 허용 규칙과 반복 제약을 적용하고, sampler가 하나의 token ID를 선택한다. 분산 서버라면 그 선택을 모든 rank와 다음 scheduler step이 같은 사실로 받아들여야 한다.

세 장의 소유권은 score의 생애로 나눈다. 이 장은 hidden row를 vocabulary logits로 만들고 분산 shard를 한 logical vocabulary 좌표로 복원하는 경계를 소유한다. 17장은 logits의 정규화와 logprob 의미를, 18장은 ordered processor·constraint·RNG·stop을 거쳐 token을 visible commit하는 정책을 소유한다.

이 경계가 흐리면 이상한 진단이 나온다. 모든 layer hidden은 같은데 출력 token만 다르다는 이유로 attention kernel을 의심한다. TP rank마다 가진 vocab shard를 global token ID처럼 읽는다. logprobs 요청이 LM head 전체 비용을 바꾼다는 사실을 놓친다. rank 0만 sampling한 token을 다른 rank에 늦게 전달해 다음 decode의 input과 cache length가 갈라진다.

17장은 다음 장에서 logit의 수학과 확률이 언제 필요한지 설명하고, 18장은 이어서 temperature, top-k/top-p, repetition, grammar, stop 순서를 다룬다. 이 장은 두 장이 받을 raw logits와 request state의 경계를 먼저 닫는다. 마지막 residual row의 shape·dtype·소유권이 final norm, vocab projection, TP gather 또는 distributed sampling, output metadata를 지나 어떻게 다음 model input의 token ID가 되는지 코드 경계로 닫는다.

이 장의 모든 계산과 사건은 다음 네 행으로 돌아온다. 별도 next-step·수치·rank·logprob 원장을 만들지
않고 필요한 열만 사건에서 확장한다.

| canonical 흐름 | 입력·출력 좌표 | owner·분산 상태 | 완료 조건 | 대표 실패·관측 |
|---|---|---|---|---|
| selected row | request·position→hidden row `[B,D]` | runner row mapping | 필요한 row와 generation 확정 | padding·packed reorder, row digest |
| vocab projection | normalized row→local score `[B,Vr]` | tied/untied head, vocab shard | valid original·added 범위의 local logits | weight identity, padding mask, FLOP·byte |
| distributed choice | local score→global token ID | gather 또는 distributed candidate·sampling owner | 모든 rank가 같은 global ID에 합의 | local/global offset, tie-break, broadcast |
| next step | global ID→embedding input·position | request/scheduler generation | commit 뒤 다음 input과 cache frontier 일치 | stale slot, cancel, accepted count |

## 16.1 마지막 residual은 token이 아니라 vocabulary를 묻기 위한 query다

문제 장면부터 보자. 두 구현의 마지막 layer output checksum은 같다. 그런데 한쪽은 token 42, 다른 쪽은 314를 선택했다. “model forward가 같다”는 보고는 절반만 맞다. final norm weight, LM head weight, selected row, vocab mapping, processor state, RNG와 distributed ownership이 아직 남아 있다.

10장의 residual ledger를 마지막 layer `L-1`까지 가져오면 `H ∈ R[T,D]`가 있다. prefill에서 T는 prompt의 여러 token row일 수 있고 decode에서는 active request당 한 row인 경우가 많다. 생성에 필요한 것은 모든 row가 아니라 각 request의 다음 token을 예측할 selected row다. dense padded batch에서는 마지막 array row가 아니라 마지막 유효 token row이고, packed serving에서는 runner가 request→row mapping을 제공한다.

`T=9`, request A가 row 0~5, B가 row 6~8을 소유한다고 하자. A와 B의 logits source row는 5와 8이다. 모든 9 row를 vocab projection하면 `[9,V]`를 만들지만 생성에는 `[2,V]`만 필요하다. LM head 전에 selected rows를 gather하면 projection FLOP과 logits memory를 줄일 수 있다. 그러나 prompt logprobs나 speculative verification은 더 많은 row가 필요하므로 항상 마지막 row 두 개만 선택할 수는 없다.

### final norm은 마지막 layer norm과 다른 checkpoint다

pre-norm decoder의 각 layer가 residual update를 더한 뒤 모델 body는 final RMSNorm 또는 LayerNorm을 한 번 더 적용할 수 있다. 마지막 layer output `H`와 final normalized `Z=Norm_final(H)`를 구별한다. LM head는 보통 Z를 읽는다. layer checkpoint가 모두 맞고 Z부터 다르면 final norm weight, epsilon, dtype 또는 PP stage owner를 본다.

작은 row `h=[1,2,-1,0]`, RMSNorm scale g=1, epsilon을 생략하면 RMS는 `sqrt(1.5)≈1.225`, `z≈[0.816,1.633,-0.816,0]`이다. 이 계산은 11장과 같지만 여기서의 의미가 다르다. Z는 다음 residual로 전달되지 않고 vocab projection의 query가 된다. final norm을 두 번 적용하거나 마지막 layer의 pre-MLP norm output을 잘못 넘기면 shape는 `[T,D]`로 같아도 logits가 달라진다.

Transformers Llama의 model body는 [layer loop와 final norm](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/modeling_llama.py#L367-L420), causal LM wrapper는 [LM head와 row selection](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/modeling_llama.py#L422-L490)에서 연결된다.

Qwen3.5도 [text model forward](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1502-L1569)와 [causal LM wrapper](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1572-L1655)를 나눠 읽는다.

### selected row는 작은 optimization이자 correctness contract다

`logits_to_keep` 같은 field는 단순 출력 슬라이스가 아니다. LM head 전에 hidden row를 줄이면 큰 `[T,V]` projection과 materialization을 피한다. Transformers Gemma3의 [row slice와 LM head](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L600-L649)는 이 경계를 보여 준다.

Qwen3.5의 [slice_indices와 projection](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1590-L1644)도 같은 의미 좌표다.

하지만 row selection이 틀리면 model layer가 완벽해도 다음 token이 틀린다. padding side, packed reorder, chunked prefill, speculative verification row, prompt logprobs 요청이 selection contract를 바꾼다. ledger에는 request incarnation, logical position, hidden row index, 왜 logits가 필요한지를 기록한다. 전체 hidden checksum만으로 selection 오류를 찾을 수 없다.

## 16.2 LM head는 hidden width를 vocabulary score로 투영한다

vocabulary matrix `W ∈ R[V,D]`와 normalized hidden row `z∈R[D]`가 있으면 logit vector는 `l=Wz+b`다. token v의 logit은 해당 vocabulary row `W_v`와 z의 내적이다. logit은 확률이 아니라 상대 score다. 모든 logit에 같은 상수를 더해도 softmax 확률은 같고 argmax도 같다. 이 성질은 stable softmax가 max를 빼는 이유와 연결되지만 자세한 확률 해석은 17장에 맡긴다.

작은 예로 D=3, V=4를 쓰자.

```text
z = [1, -1, 0.5]
W0 = [ 1,  0,  0]  → l0 = 1
W1 = [ 0,  1,  1]  → l1 = -0.5
W2 = [ 1, -1,  0]  → l2 = 2
W3 = [-1,  0,  2]  → l3 = 0
```

argmax는 vocab row 2다. temperature 1의 softmax를 꼭 만들지 않아도 greedy token 2는 선택할 수 있다. logprobs를 요청하거나 stochastic sampling을 하면 normalization과 probability가 필요하다. LM head는 token을 고르지 않았다. 네 score를 만들었고 sampler가 계약에 따라 token ID를 고른다.

### weight tying은 두 module 이름이 같은 storage를 가리키는 계약이다

많은 causal LM은 input embedding weight와 output LM head weight를 묶는다. token ID v를 hidden row로 읽을 때 사용한 embedding row와, hidden을 vocab v score로 투영할 때 사용하는 row가 같은 parameter storage다. 이는 vocab semantic 좌표를 공유하고 parameter memory를 줄이지만, embedding lookup과 output projection의 연산 방향은 다르다.

`tie_word_embeddings=True` 또는 `_tied_weights_keys`가 있다고 실제 storage alias가 항상 완성되었다고 단정하지 않는다. model initialization, weight loading, quantization, adapter, PP partition이 tie를 구현하는 방식이 다를 수 있다.

Transformers Llama의 [tied weight와 LM head 선언](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/modeling_llama.py#L422-L437), Gemma3의 [tied/TP/PP plan](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L583-L599)을 source 기준점으로 삼는다.

vLLM Qwen2 MoE는 [LM head 생성과 tie assignment](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen2_moe.py#L418-L489)에서 serving parallel head와 embedding weight를 연결한다. tied model에서 embedding shard와 output vocab shard의 padding/loader mapping이 같아야 한다. 별도 LM head artifact가 존재하는 untied model에 tie를 강제하면 logits가 틀린다.

### vocab padding은 실제 token ID가 아닌 행을 만든다

TP와 kernel alignment 때문에 vocab size를 divisible/aligned 크기로 pad할 수 있다. 원래 V=100003인데 padded Vp=100096일 수 있다. extra 93 row는 사용자 tokenizer의 valid token이 아니다. local shard GEMM에는 포함될 수 있지만 global sampling 전에 mask하거나 제거해야 한다. padded row가 큰 임의 logit을 가지면 invalid token이 선택될 수 있다.

vLLM의 [ParallelLMHead 선언](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/vocab_parallel_embedding.py#L518-L579)과 [vocab shard indices](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/vocab_parallel_embedding.py#L70-L180)는 original/padded/adaptor vocab 좌표를 읽는 출발점이다. local column index와 global token ID를 구분한다.

**dtype는 final hidden과 LM head에서 다시 갈라진다**

Z가 BF16이고 head weight가 BF16이어도 accumulator 또는 output logits를 FP32로 만들 수 있다. quantized model이 모든 weight를 같은 format으로 저장하는 것도 아니다. LM head는 accuracy 또는 kernel 지원 때문에 unquantized로 남을 수 있다. `enable_fp32_lm_head` 같은 option은 단순 return cast가 아니라 projection compute와 weight conversion 비용을 바꿀 수 있다.

vLLM [LogitsProcessor의 head dtype 경로](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/logits_processor.py#L23-L139)는 quant method, FP32 head와 matrix multiplication 선택을 보여 준다. option이 true라는 사실보다 constructed head dtype, weight storage, output logits dtype을 확인한다. 매 step weight 전체를 변환하면 큰 비용이므로 source가 conversion lifetime을 어떻게 다루는지 본다.

## 16.3 vocabulary tensor parallel은 local score를 global token 좌표로 되돌린다

V가 매우 크면 LM head weight와 logits를 TP rank에 vocab row 방향으로 나눈다. TP=2, V=8이면 rank 0이 token 0~3, rank 1이 4~7을 소유할 수 있다. 각 rank는 같은 normalized hidden z와 local weight shard로 네 logit을 만든다. local index 1은 rank 0에서는 global token 1, rank 1에서는 token 5다.

greedy sampling의 단순 방법은 logits shard를 all-gather해 `[B,V]`를 만들고 global argmax를 구하는 것이다. 다른 방법은 rank마다 local max `(value,global_id)`를 구하고 distributed max reduction으로 승자를 정한다. top-k/top-p와 logprobs는 더 많은 global ordering 정보가 필요해 communication 계약이 복잡하다. implementation이 어떤 방법을 쓰는지 source를 확인한다.

### all-gather는 logits를 완성하지만 비용과 lifetime을 만든다

batch selected rows B=64, V=128000, FP32 logits라면 full tensor는 `64×128000×4=32,768,000 bytes`, 약 31.25 MiB다. 매 decode step 모든 rank에 gather하면 network traffic과 GPU memory가 반복된다. prompt logprobs로 T=2048 row를 projection/gather하면 약 0.98 GiB다. selected-row optimization과 distributed logits가 중요한 이유다.

vLLM [LogitsProcessor forward](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/logits_processor.py#L63-L99), [TP gather와 vocab trim](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/logits_processor.py#L140-L199)은 local projection 뒤 padded/adaptor vocab을 어떻게 다루는지 읽는 좌표다. gather 결과가 모든 rank에 있는지 driver rank에만 있는지도 caller까지 본다.

SGLang은 [ParallelLMHead](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/vocab_parallel_embedding.py#L591-L669), [LogitsProcessor forward](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/logits_processor.py#L300-L470),

[Sampler entry](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/sampler.py#L71-L180)을 이어 본다. TP LM-head all-to-all 또는 DP LM-head option이 있으면 default resolution과 validation까지 추적한다.

**pipeline parallel에서는 마지막 stage만 vocab weight를 소유할 수 있다**

PP stage 0이 embedding과 앞 layer, 마지막 stage가 final norm과 LM head를 소유하는 구성이 흔하다. stage 0은 logits를 만들지 않고 intermediate hidden을 보낸다. tied embedding/LM head가 서로 다른 stage에 논리적으로 위치하면 parameter storage를 물리적으로 공유할 수 없고 loader가 복제 또는 별도 mapping을 해야 할 수 있다.

PP=2에서 final hidden send/receive checksum은 맞는데 logits부터 다르면 마지막 stage final norm/head weight를 본다. stage 0 embedding storage pointer와 stage 1 LM head pointer가 다르다고 tie가 깨졌다고 단정하지 않는다. 두 복제 weight의 값과 update-free inference contract가 같으면 semantic tie를 구현할 수 있다.

**sampling owner와 다음-step owner가 합의해야 한다**

한 rank가 token ID를 선택했다면 다음 model invocation의 모든 TP rank가 같은 token을 embedding하고 같은 logical position을 진행해야 한다. driver rank가 token 42를 골랐는데 rank 1은 local argmax token 5를 사용하면 collective 이후 cache가 즉시 갈라진다. selected token broadcast는 작은 값이지만 correctness의 commit barrier다.

ledger에는 local logits ready, global sampling state ready, token selected, token broadcast complete, request state committed, next scheduler input built를 순서대로 둔다. 비동기 stream과 distributed group이 있으면 event dependency를 기록한다. CPU detour가 있는지 GPU-resident token path인지도 latency에 영향을 준다.

## 16.4 processor와 sampler는 score의 의미와 request state를 함께 읽는다

LM head가 만든 raw logits는 아직 사용자 계약을 반영하지 않는다. banned token, minimum length, repetition/history, grammar, temperature, top-k/top-p 같은 변환이 적용될 수 있다. 정확한 순서와 수식은 18장에서 설명한다. 여기서는 processor가 읽는 state와 logits ownership이 serving batch에서 어떻게 맞아야 하는지 본다.

mixed batch의 request A는 temperature 0 greedy, B는 temperature 0.8 top-p, C는 JSON grammar를 사용할 수 있다. logits tensor는 `[3,V]`지만 각 row의 parameter와 history, grammar state가 다르다. batch parameter array가 packed row reorder와 함께 이동해야 한다. A의 repetition history를 B row에 적용하면 shape와 dtype은 정상이고 결과만 틀린다.

### temperature와 stable softmax의 경계를 숫자로 확인한다

raw logits `[1000,1001,999]`에 naive exp를 적용하면 FP32에서도 overflow 위험이 있다. max 1001을 빼면 `[-1,0,-2]`이고 exp는 약 `[0.368,1,0.135]`, 합 1.503, 확률은 `[0.245,0.665,0.090]`이다. temperature 0.5라면 max를 빼기 전/후 일관되게 logits 차이를 두 배로 만들어 분포가 더 뾰족해진다.

greedy는 softmax 없이 argmax할 수 있다. 그러나 logprobs 반환은 normalization을 요구한다. raw logits output, processed logits, logprobs를 같은 이름 `scores`로 섞지 않는다. observability schema에 stage를 붙인다. processor 전 top token과 후 top token이 다르면 model bug가 아니라 요청 정책이 작동했을 수 있다.

SGLang sampler의 [forward와 probability preparation](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/sampler.py#L98-L249), [probability sampling path](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/sampler.py#L250-L415),

[logits/logprobs sampling variants](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/sampler.py#L416-L470)을 의미 좌표로 읽는다. source에 여러 path가 있어도 effective backend와 request mode를 확인한다.

### RNG state는 seed 숫자 하나보다 넓다

stochastic sampling은 probability와 random draw가 필요하다. 같은 seed여도 request batching 순서, generator ownership, consumed random count, speculative proposal/reject가 다르면 token이 갈릴 수 있다. request별 generator인지 batch/global generator인지 확인한다. retry가 같은 logical request의 RNG state를 재사용할지 새 attempt로 진행할지도 제품 계약이다.

분산에서 모든 rank가 독립 sampling하면 random draw가 달라질 수 있다. 한 owner가 sampling하고 broadcast하거나 counter-based RNG 좌표를 합의해야 한다. deterministic mode 주장은 artifact/input/processor/backend/RNG schedule을 고정한 범위에서만 한다. floating reduction order가 확률 경계에서 token을 바꿀 가능성도 있다.

### grammar와 stop은 token 선택 전후의 다른 책임이다

grammar mask는 보통 선택 가능한 token logits를 제한하므로 sampler 전 state다. stop token은 선택된 token ID를 보고 request 완료를 결정할 수 있다. stop string은 decode된 text boundary와 tokenizer byte sequence가 필요해 더 뒤의 output processor 책임일 수 있다. token이 선택되었지만 사용자에게 emit하지 않고 stop으로 소비하는 경우도 있다.

“stop이 한 token 늦다”는 증상에서 LM head를 바로 의심하지 않는다. selected token ID, grammar/stop state transition, detokenizer buffered bytes, emitted event를 나눈다. 이 장은 selected token과 distributed commit까지를 닫고 문자열 조립·streaming은 서비스 출력 장으로 넘긴다.

## 16.5 네 구현의 forward 종점을 같은 의미 좌표로 읽는다

Transformers의 causal LM `forward`는 model body hidden을 받고 필요한 row를 LM head에 투영해 logits를 반환한다. `generate`는 그 결과에 processor와 stopping criteria, cache update를 반복한다. reference를 만들 때 `forward` logits와 `generate` selected token을 같은 단계라고 부르지 않는다.

Llama의 [causal LM class와 forward](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/llama/modeling_llama.py#L422-L490), Qwen3.5의 [head declaration과 forward](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1572-L1648),

Gemma3의 [language model head](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/gemma3/modeling_gemma3.py#L583-L649)를 같은 축에 놓는다. tied key, TP plan, logits row selection과 output dtype을 비교한다.

vLLM model class는 hidden 계산과 `compute_logits`를 분리하는 경우가 많다. Qwen2 MoE의 [compute_logits](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen2_moe.py#L480-L495)는 model-specific wrapper에서 common LogitsProcessor로 handoff하는 예다. runner가 어떤 hidden rows를 넘기는지, sampler가 어느 worker에서 실행되는지는 caller로 올라가 확인한다.

vLLM [ParallelLMHead forward](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/vocab_parallel_embedding.py#L520-L580)는 vocab shard projection의 owner다. [LogitsProcessor](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/logits_processor.py#L23-L199)는 head dtype, gather, original/adaptor vocab trim을 담당한다. 이름이 processor라고 sampling policy까지 모두 소유한다고 오해하지 않는다.

SGLang도 model wrapper의 LM head, logits processor, sampler, runner state를 분리한다. [Qwen3 model wrapper](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/qwen3.py#L435-L560), [vocab parallel head](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/vocab_parallel_embedding.py#L591-L669),

[logits processor](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/logits_processor.py#L300-L470), [sampler](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/sampler.py#L71-L470)를 순서대로 잇는다.

llama.cpp는 architecture graph가 output norm과 output weight multiplication node를 만들고, sampling chain이 logits를 소비한다. [model graph builder](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model.cpp#L2461-L2550)에서 architecture별 graph로 내려가 `result_output` 의미를 찾는다. sampling 구현 파일은 vendored revision에서 symbol을 검색해 token data array, softmax, top-k/p, dist sampling chain을 연결한다. PyTorch의 `[B,V]` 표기를 ggml `ne[]`와 바로 동일시하지 않는다.

**source walk는 여섯 handoff를 닫는다**

첫째 final layer output에서 final norm input으로 간다. 둘째 normalized hidden에서 selected row를 고른다. 셋째 LM head local projection과 vocab shard를 찾는다. 넷째 global logits 또는 distributed top candidates를 만든다. 다섯째 request별 processor와 RNG/grammar state를 적용해 token ID를 선택한다. 여섯째 token을 request state와 모든 execution rank에 commit한다.

한 handoff라도 source에서 끊기면 “model이 token을 만들었다”라고 쓰지 않는다. 특히 model class가 logits만 반환한다면 token selection owner는 generation loop 또는 server sampler다. runner가 selected token을 CPU output queue로 보내는 것과 다음 GPU input buffer에 쓰는 것도 다른 edge다.

## 16.6 first-logit differential은 model과 serving 정책을 분리한다

전체 text가 달라졌을 때 첫 selected token부터 비교한다. teacher-forced exact token IDs로 같은 prefill과 decode input을 만들고, final layer residual, final norm output, selected hidden row, raw local/global logits, processed logits, random draw와 selected token을 coarse-to-fine으로 기록한다.

마지막 hidden까지 같고 final norm부터 다르면 norm owner다. norm까지 같고 local logits부터 다르면 LM head weight/dtype/quant/tie/shard다. rank local logits는 같은데 gathered global logits가 다르면 vocab offsets/padding/collective다. raw global logits가 같고 processed logits가 다르면 request policy/history/grammar row mapping이다. processed distribution까지 같은데 token이 다르면 RNG와 distributed sampling owner다.

### 작은 vocab fixture로 global ID를 검산한다

TP=2, V=8에서 rank 0 local logits `[1,3,0,2]`, rank 1 `[4,1,0,-1]`이라고 하자. rank 1 local argmax index 0은 global token 4다. global argmax는 token 4 score 4다. local index 0을 그대로 broadcast하면 token 0을 잘못 선택한다. pair `(score,global_id)` 또는 correct offset gather가 필요하다.

vocab padding으로 rank 1에 invalid rows 8~11이 있고 그중 score 100이 남아 있다면 trim/mask 전에 argmax하면 invalid token이 이긴다. projection weight padding 초기화가 0이라고 영원히 가정하지 않는다. quant loader나 uninitialized buffer 문제를 막기 위해 semantic vocab boundary에서 명시적으로 제외한다.

### logprobs만 느린 사건

일반 generation은 selected row 한 개와 sampled token만 필요할 수 있다. prompt logprobs는 prompt 여러 row의 LM head projection과 log-softmax, selected/ top-k value 수집이 필요하다. vocab V가 크면 `[T,V]` materialization과 TP gather가 지배할 수 있다. “sampling이 느리다”가 아니라 row count, vocab projection, normalization, output transfer를 나눈다.

request option이 `logprobs=5`라고 top 5만 계산하면 된다는 뜻도 아니다. 정확한 top 5를 얻으려면 global vocab score를 비교해야 한다. distributed top-k algorithm으로 full gather를 피할 수 있지만 local candidate merge와 normalization contract가 필요하다. prompt logprob와 generated token logprob의 row selection도 다르다.

### restart 뒤 seed divergence 사건

artifact와 input, raw logits가 restart 전후 같은데 sampled token만 다르면 seed initialization, request ID→generator mapping, batch order를 본다. raw logits가 미세하게 다르면 backend/dtype/collective도 남는다. 같은 seed 문자열이 log에 있다는 사실은 같은 random counter를 소비했다는 증거가 아니다.

greedy fixture가 restart 전후 맞고 stochastic만 다르면 model weight 오류 가능성은 낮아진다. 단, temperature/top-p processor가 달라져 distribution이 바뀌었는지 먼저 확인한다. generator state digest와 draw owner를 기록할 수 있는 synthetic test를 설계하되 실제 사용자 probability 전체를 무제한 덤프하지 않는다.

## 16.7 선택된 token은 다음 decode step의 입력이자 commit record다

token ID v가 선택되면 tokenizer decode를 기다리기 전에 model state는 다음 step을 준비할 수 있다. request generated token list에 v를 append하고, stop/length policy를 판정하며, next input buffer에 v를 쓰고, logical position과 cache commit을 진행한다. speculative path에서는 accepted tokens와 proposed tokens를 구분한다.

이 경계의 핵심은 exactly-once semantic이다. token이 사용자에게 emit되었지만 request state에 commit되지 않으면 retry에서 중복 또는 다른 continuation이 생길 수 있다. state에는 commit되었지만 output stream에 전달되지 않으면 사용자는 timeout을 보고 재시도할 수 있다. API delivery와 model token commit은 분산 transaction이 아니므로 attempt/request identity와 replay 정책이 필요하다.

### canonical 흐름의 네 번째 행: next-step commit

```text
step s final hidden ready
raw/processed logits identity
sample draw identity
selected global token ID
distributed broadcast complete
accepted/committed token count
stop decision
next input row + logical position
cache visible length
output emitted/detokenizer state
```

selected token과 next input row는 같아야 하지만 stop token이면 다음 model step을 만들지 않을 수 있다. speculative verification에서는 여러 accepted token이 한 번에 commit될 수 있다. beam search는 여러 branch와 parent index를 갖는다. 이 장의 단일 sequence 기준을 확장할 때 branch identity를 추가한다.

### broadcast 실패와 stale token incident

driver rank가 token 42를 선택했지만 rank 2가 이전 step token 17을 사용했다고 하자. rank 2 embedding/QKV가 처음부터 달라지고 collective가 오염된다. 다음 step에서만 first divergence가 생기므로 이전 step LM head와 sampler를 함께 봐야 한다. token broadcast completion과 next graph replay 사이 event를 확인한다.

shape와 CUDA graph address가 고정되어도 buffer value incarnation은 바뀐다. request slot을 재사용할 때 previous token과 sampling parameter가 남아 있으면 mixed batch에서 간헐 오답이 난다. slot ID 외에 request generation을 기록한다. cancellation 뒤 async sample/broadcast가 새 request slot을 덮지 못하게 한다.

**tokenizer와 문자열은 한 단계 뒤다**

token ID 42가 어떤 bytes/text가 되는지는 tokenizer vocab과 added token, byte fallback, streaming decode buffer에 달렸다. model은 문자열을 출력하지 않는다. selected IDs가 같은데 text가 다르면 tokenizer revision과 decode state를 본다. IDs부터 다르면 이 장의 경계로 돌아온다.

stop string이 token boundary를 가로지르면 detokenizer가 여러 token bytes를 버퍼링해야 할 수 있다. 사용자에게 token event를 보낼지 text delta를 보낼지도 API 계약이다. LM head latency와 detokenization/backpressure latency를 분리한다. 이 장의 종점은 global token commit이며 네트워크 delivery 전체가 아니다.

## 16.8 canonical 네 행으로 “model은 맞는데 답이 다르다”를 좁힌다

상황은 다음과 같다. vLLM과 SGLang에서 같은 Qwen artifact, token IDs, BF16, TP=4를 사용한다. greedy 요청은 같지만 temperature 0.7, top-p 0.9에서 seed를 같게 주어도 세 번째 token부터 갈린다. logprobs를 요청하면 한 서버만 ITL이 크게 늘어난다. 실행 결과를 미리 만들지 않고 조사 설계를 한다.

먼저 세 step의 teacher-forced raw logits를 비교한다. 첫 두 selected token이 같다는 이유로 그때의 raw logits가 같다고 가정하지 않는다. 각 step final hidden, final norm, local/global raw logits top values와 digest를 비교한다. raw logits가 허용 오차 밖이면 sampling seed를 조사하기 전에 model/head/collective로 돌아간다.

raw logits가 같으면 processor 결과를 비교한다. temperature, top-p, banned/repetition/grammar state와 application order를 fixed schema로 정규화한다. 두 서버 내부 class 이름을 맞추지 않고 raw logits→eligible set→normalized distribution이라는 의미를 맞춘다. history에는 emitted text가 아니라 committed token IDs를 사용했는지 확인한다.

distribution도 같으면 random draw 좌표를 본다. seed, request incarnation, step index, sample index, generator device/rank, consumed counter를 기록한다. 한 서버가 batch-global generator를 사용해 다른 concurrent request 때문에 counter가 진행할 수 있다. “동일 seed”의 제품 계약이 batch 독립 reproducibility까지 약속하는지 문서와 source로 확인한다.

### logprobs는 두 번째·세 번째 행의 추가 비용으로 읽는다

active request 64, V=152064, FP32 global logits면 한 selected-step full tensor는 약 37.1 MiB다. TP=4 local은 약 9.3 MiB지만 gather/merge와 logsumexp가 필요하다. prompt 1024 row의 logprobs를 모두 요구하면 raw materialization 하한은 약 594 MiB per request가 되어 현실적으로 chunk/selection이 필요하다.

여기서는 row 수, projection, collective와 반환 byte라는 서빙 경계만 다룬다. Logsumexp, 정규화된
확률과 exact logprob의 수학·수치 안정성은 17장으로 넘긴다.

두 서버가 같은 `logprobs=5` API를 제공해도 한쪽은 full logits gather 뒤 top-k, 다른 쪽은 distributed top-k와 logsumexp reduction을 사용할 수 있다. 정확성과 latency를 함께 비교한다. output으로 보내는 top-5 bytes는 작지만 이를 찾기 위한 global computation은 작지 않다. 네트워크 응답 크기만 보고 원인을 찾지 않는다.

selected row 수, projection dtype, local vocab width, gather/all-to-all bytes, softmax/logsumexp, CPU transfer, serialization 시간을 구분한다. logprobs off에서는 LM head 자체가 여전히 next token을 위해 필요하지만 prompt 전체 row projection은 생략될 수 있다. option→row selection→tensor shape→collective→ITL effect를 닫는다.

### weight tying/quantization incident를 반증한다

한 서버 loader가 tied embedding을 LM head에 alias하고 다른 서버는 별도 weight를 로드한다고 하자. 둘의 값이 같으면 output은 같을 수 있다. storage alias 여부만 bug 판정으로 쓰지 않는다. weight digest를 vocab shard/global row 좌표로 비교한다. quantized artifact가 LM head를 제외하는지 포함하는지도 loader config와 constructed quant method로 본다.

embedding output부터 다르면 tie보다 embedding loader다. hidden은 같고 logits만 다르면 LM head shard/dtype가 후보가 된다. 특정 vocab range에서만 logit이 다르면 shard offset, adapter vocab, padding row를 본다. 모든 vocab에서 작은 오차가 있으면 dtype/quantization을 본다.

### 수정과 완료 판정

원인이 request-global RNG였고 제품 계약이 request별 deterministic seed를 요구한다면 generator ownership을 바꾼다. 수정 후 concurrency와 batch reorder를 바꾸어도 동일 request fixture가 같은 token sequence인지 확인한다. raw logits와 processor distribution은 변경 전후 같아야 한다. performance와 generator state lifetime도 본다.

logprobs latency 원인이 full prompt logits materialization이라면 required rows를 먼저 선택하거나 distributed computation을 개선할 수 있다. 그러나 prompt logprob exactness, padded/adaptor vocab, TP merge, output ordering을 검증한다. 속도만 개선되고 값이 틀리면 완료가 아니다.

마지막으로 canonical 네 행을 닫는다. final residual과 final norm, selected hidden row, LM head weight/vocab shard, raw/processed logits, random draw, selected global token, broadcast/commit, next input의 owner가 모두 이어져야 한다. source link는 각 handoff가 실제로 존재함을 증명하고, differential checkpoint는 어느 handoff가 처음 갈렸는지 보여 준다.

이 장을 지나면 “모델 출력”이라는 말을 세분화할 수 있다. model body output은 hidden state다. LM head output은 logits다. sampler output은 token ID다. tokenizer/output processor는 text delta를 만든다. API server는 이를 사용자에게 전달한다. 경계를 구분해야 성능과 correctness의 책임을 올바른 stack으로 보낼 수 있다.

48장에서는 config와 artifact가 tied head, vocab size, dtype, quantization, TP plan을 어떻게 선택하는지 다시 본다. 52장에서는 Qwen·Gemma·Llama의 한 요청을 이 종점까지 수직으로 걷는다. scheduler와 cache 장에서는 selected token commit이 다음 step admission과 KV lifetime을 어떻게 바꾸는지 연결한다. 어느 방향으로 가든 global token ID와 request incarnation을 잃지 않는다.

## 16.9 두 번째·세 번째 행의 FLOP·byte·통신을 요청 모양으로 번역한다

왜 logit 경계가 serving 최적화의 독립 대상인지 수치로 보자. hidden width D=8192, vocabulary V=152064, selected row B=64, BF16 head weight와 hidden, FP32 logits를 가정한다. LM head weight element는 `V×D=1,245,708,288`개이고 BF16이면 약 2.32 GiB다. tied weight라면 input embedding과 같은 parameter를 가리켜 별도 2.32 GiB를 피할 수 있지만 projection 때 이 weight를 읽는 비용은 사라지지 않는다.

한 decode step의 matrix multiplication은 대략 `2×B×D×V` FLOP이다. 숫자는 약 159.5 TFLOP다. weight를 한 step에서 얼마나 재사용하는지는 B와 kernel tile에 달렸다. B가 1이면 거대한 weight를 읽어 vocab score 한 row만 만들기 때문에 memory/launch 관점이 강하다. B=64면 같은 weight tile을 여러 hidden row에 적용할 기회가 커진다. scheduler가 active rows를 모으는 이유가 LM head에도 연결된다.

출력 FP32 logits는 `64×152064×4≈37.1 MiB`다. BF16 raw logits라면 절반이지만 stable softmax/logsumexp와 API logprobs가 FP32를 요구할 수 있다. head weight 2.32 GiB와 비교하면 output은 작아 보이지만 TP gather와 매-step allocation, prompt row가 커질 때는 무시할 수 없다. memory capacity와 bandwidth, communication을 canonical 표의 두 번째·세 번째 행에 둔다.

**selected rows가 projection 비용을 어떻게 바꾸는가**

길이 2048 prompt 8개를 padded batch로 prefill했다고 하자. 모든 row logits를 만들면 T=16384다. 다음 token만 필요하면 각 request의 마지막 유효 row 8개만 투영할 수 있다. GEMM M축과 logits output이 2048배 차이 난다. model layers는 prompt token 전체를 처리해야 하지만 vocab projection은 선택적으로 줄일 수 있다.

prompt logprobs가 켜지면 선택 row가 다시 늘어난다. causal prediction에서 position j의 hidden이 token j+1의 logprob를 평가하므로 첫/마지막 row alignment를 정확히 정의한다. padding과 BOS/EOS, image placeholder, masked label을 제외할 수 있다. `T` 전체를 무조건 projection하는 구현과 필요한 row index만 gather하는 구현은 API 값이 같아도 비용이 다르다.

speculative verification은 draft token k개에 대한 logits row가 필요할 수 있다. 일반 decode처럼 request당 마지막 row 하나만 고르면 acceptance를 계산할 수 없다. runner가 `run_lm_head` 또는 logits indices를 mode별로 바꾸는 이유다. selected-row optimization은 execution mode의 correctness contract를 읽고 적용해야 한다.

### vocab TP=4에서 두 번째·세 번째 행을 rank별로 계산한다

V=152064가 TP=4로 정확히 나뉜다면 rank당 38016 vocab row와 약 594 MiB BF16 head weight를 가진다. B=64 local logits FP32는 약 9.28 MiB다. full gather 뒤 각 rank가 37.1 MiB를 보유할 수 있다. all-gather traffic과 storage가 모든 rank에 필요한지는 sampler placement에 달렸다.

original vocab V=151936이고 padded Vp=152064라면 128 invalid row가 있다. rank별 shard range를 original과 padded 좌표로 적는다. adapter가 extra vocab을 추가하면 original, added, padded range가 더 복잡해진다. tokenizer가 반환할 수 있는 global token IDs와 output head가 가진 row mapping이 일치해야 한다.

rank 0은 global 0~38015, rank 1은 38016~76031처럼 offset을 가진다. local top token `(score=12.3,index=10)`은 rank 2에서 global 76042다. distributed top-k merge는 score와 global ID를 함께 운반한다. tie score의 deterministic ordering도 구현 계약이다. score가 정확히 같을 때 작은 ID를 택할지 reduction order를 따를지 정하지 않으면 rank topology에 따라 greedy가 달라질 수 있다.

### all-gather를 피하는 distributed greedy를 손으로 계산한다

TP=3, 각 rank의 local maximum이 `(5.0, token 2)`, `(7.0, token 6)`, `(6.5, token 10)`이라고 하자. max reduction은 score 7.0과 global token 6을 선택한다. full logits를 모으지 않아도 greedy 정답을 얻는다. 하지만 temperature sampling에서 전체 probability normalization이 필요하면 global logsumexp가 필요하고, top-p는 cumulative ordering을 위해 후보를 더 모아야 한다.

distributed softmax는 각 rank local max를 구하고 global max m을 reduction한 뒤 local `sum exp(l_i-m)`을 계산해 global sum을 reduction할 수 있다. 특정 token logprob는 그 token owner의 logit과 global logsumexp로 계산한다. top-k는 rank마다 local top-k를 모아 global top-k를 다시 고를 수 있다. top-p는 필요한 후보 수가 분포에 따라 달라 local fixed-k만으로 정확하지 않을 수 있다.

그러므로 “sampler를 분산화하면 gather가 사라진다”는 말은 충분하지 않다. 어떤 API mode와 exactness를 지원하는지, communication이 all-gather에서 reduce/candidate exchange로 어떻게 바뀌는지 말해야 한다. logprobs top-n, full logits return, grammar mask가 vocab shard에 어떻게 적용되는지도 포함한다.

**final norm과 head dtype의 byte 경계를 계산한다**

B=64, D=8192의 selected hidden은 BF16 약 1 MiB다. final norm FP32 temporary도 약 2 MiB라 작다. 하지만 LM head weight를 FP32로 materialize하면 4.64 GiB가 필요하다. 매 step `.to(fp32)`로 복제하면 capacity와 latency가 크게 악화된다. source가 weight conversion을 매 호출 하는지, GEMM이 BF16 weight에 FP32 output accumulator를 지원하는지 구분한다.

quantized 4-bit LM head data 하한은 약 0.58 GiB지만 group scale/zero, unpack/dequant workspace, kernel eligibility가 더해진다. 많은 quantization config가 LM head를 제외하는 이유는 accuracy와 vocab projection kernel 지원, tied embedding storage 등 여러 조건이다. artifact가 INT4라고 LM head까지 INT4라고 쓰지 않는다.

final hidden의 dtype을 FP32로 올린다고 전체 model residual이 FP32가 되는 것은 아니다. selected rows만 cast할 수 있다. option→selected hidden cast→head GEMM dtype→logits dtype→sampler input의 사슬을 본다. 성능 효과는 B,V,hardware kernel에 따라 다르며 정밀도 이득은 logit margin cohort로 평가한다.

### logit margin이 작은 token에서 수치 차이가 선택을 바꾼다

top logits가 token A 10.000, B 9.999인 경우 0.002 오차가 argmax를 뒤집을 수 있다. margin이 5인 경우 같은 오차는 선택을 바꾸지 않는다. layer hidden allclose가 통과해도 near-tie cohort의 greedy token이 다를 수 있다. 반대로 token이 같다는 사실만으로 모든 logits가 같다는 뜻도 아니다.

correctness 계약을 bitwise logits, tolerance logits, greedy token parity, distribution distance, task quality 중 어디에 둘지 명시한다. backend와 dtype이 같다면 강한 parity를 요구할 수 있고 quantized serving은 별도 tolerance/quality 기준이 필요하다. stochastic token sequence exact match는 RNG schedule까지 고정하지 않으면 약한 비교다.

logit differential은 selected global indices의 raw value, max absolute/relative error, top-k overlap, margin을 함께 본다. softmax probability는 logit shift invariant이므로 raw logits에 constant offset만 있는 경우 distribution은 같다. processor가 absolute threshold를 쓰는 특수 계약이 없다면 token selection은 유지된다. 관측 목적에 맞는 metric을 고른다.

**output logprobs의 CPU·network 경계를 계산한다**

generated token마다 top-20 logprobs를 반환하면 값과 token ID, decoded representation metadata가 필요하다. 64 request에서 수십 KiB 수준일 수 있지만 GPU에서 top candidates를 찾고 CPU object로 변환하며 JSON을 만드는 overhead가 작은 decode step에서는 보일 수 있다. full `[B,V]`를 CPU로 복사해 top-20을 찾으면 37 MiB transfer가 매 step 발생한다.

GPU top-k 뒤 20개만 전송하는 경로와 full logits transfer를 구분한다. synchronization을 위해 `.cpu()` 또는 scalar `.item()`이 들어가면 GPU pipeline이 기다릴 수 있다. logprobs option이 ITL만 늘리고 model kernel 시간은 같다면 output extraction과 sync를 본다. metric에는 LM head, sampler, D2H, serialization, stream backpressure를 나눈다.

사용자가 full logits를 요청하는 별도 API는 capacity와 개인정보 위험이 다르다. giant tensor return을 기본 generation path와 같은 것으로 취급하지 않는다. source가 debug/output option에서 hidden/logits reference lifetime을 늘리는지도 확인한다. 관측 기능이 allocator peak를 바꾸는 원리는 10장의 hidden-state retention과 같다.

**weight tying의 세 가지 물리 구현을 구분한다**

단일 device eager에서는 embedding module과 LM head가 같은 parameter object/storage를 가리킬 수 있다. TP에서는 두 module이 같은 vocab shard storage를 가리킬 수 있다. PP에서 embedding과 head가 다른 stage라면 동일 storage는 불가능하고 동일 값의 replica를 가질 수 있다. 세 경우 모두 semantic tying을 구현하지만 memory와 loading 경로가 다르다.

checkpoint loader가 output weight를 생략하고 embedding에서 tie할 수 있다. 반대로 artifact에 둘 다 있는데 config가 untied라면 두 parameter를 각각 로드해야 한다. loader가 missing weight를 자동 tie하는 조건과 unexpected weight 처리, quant method assignment을 확인한다. weight filename만으로 constructed storage를 단정하지 않는다.

adapter가 output head에 붙을 수 있거나 added vocabulary row를 제공할 수 있다. embedding에는 adapter row가 있는데 head mapping에는 없으면 token을 input으로 읽을 수 있어도 output으로 선택할 수 없다. 반대도 가능하다. base/added vocab ranges와 adapter ID가 request row까지 보존되어야 한다.

### 네 행의 결과를 운영 metric으로 번역한다

LM head 시간을 `lm_head_seconds`, global communication을 `logits_collective_seconds`, sampling을 `sample_seconds`, output logprobs를 `logprob_extract_seconds`처럼 구분할 수 있다. 실제 metric 이름은 구현을 따른다. 전체 decode model time만 있다면 trace/span 또는 bounded instrumentation 계획을 세운다.

관측에는 selected rows B, original/padded vocab V/Vp, head dtype, TP/DP mode, requested logprobs rows/top-n, sampler backend를 label 또는 exemplars와 연결한다. 고 cardinality request ID를 Prometheus label로 직접 넣지 않고 trace correlation을 사용한다. metric 값과 source branch를 연결할 evidence가 필요하다.

LM head가 느리다는 결론 뒤에는 workload shape가 있어야 한다. B가 작아 weight read가 지배하는지, prompt logprobs로 T가 큰지, vocab이 크거나 adapter vocab이 있는지, FP32/quant fallback인지, TP gather인지 말한다. 그래야 batch 증가, row pruning, distributed sampler, dtype 변경 중 어느 최적화가 왜 맞는지 판단할 수 있다.

## 16.10 종합 incident: 값은 맞지만 사용자가 느리고, 빠르게 만들자 답이 달라졌다

한 팀이 `logprobs=5` 요청의 ITL이 높다는 문제를 해결하려 한다. profile을 대충 본 결과 LM head 뒤 GPU utilization이 낮아 보여 full logits gather를 제거하고 rank별 top-5만 모으도록 바꿨다. latency는 줄었지만 일부 top-p 요청의 token 분포가 달라지고 grammar 요청에서 invalid token이 선택됐다. 이 사건은 최적화의 semantic contract가 왜 필요한지 보여 준다.

처음 직관은 그럴듯하다. 각 rank에서 top-5를 골라 모으면 global top-5도 찾을 수 있다. rank local top-5를 모두 모아 global top-5를 다시 고른다면 반환용 top-5에는 맞다. 그러나 top-p는 global cumulative probability mass가 threshold를 넘을 때까지 후보가 얼마나 필요한지 미리 알 수 없다. 분포가 평평하면 rank당 5개보다 훨씬 많은 token이 nucleus에 들어간다.

grammar mask도 local shard에 정확히 적용되어야 한다. grammar engine이 global allowed-token bitset을 만들었는데 local offset 없이 각 rank가 처음 Vlocal bits를 읽으면 rank 1 이후 mask가 어긋난다. padded vocab row와 added adapter vocab도 boundary를 복잡하게 한다. shape가 맞고 kernel이 성공해도 semantic eligible set이 틀린다.

팀은 “top-5 API인데 왜 top-p 전체가 필요한가”라고 묻는다. 반환할 logprobs top-5와 sampling policy top-p는 다른 요구다. output top-5만 필요해도 sampler는 정확한 distribution에서 token을 선택해야 한다. 두 계산을 같은 후보 set으로 재사용하려면 그것이 exact한 조건을 증명해야 한다.

### 증상을 요청 mode별로 분리한다

greedy+no-logprobs, top-k=5+logprobs=5, top-p=0.95+logprobs=5, grammar+top-p의 네 fixture를 둔다. teacher-forced raw local logits는 변경 전후 같아야 한다. global selected token과 returned top logprobs, eligible count, normalization constant를 mode별로 비교한다.

greedy가 맞다는 사실은 local/global max merge가 맞음을 보여 주지만 top-p cumulative mass는 증명하지 않는다. top-k=5가 맞아도 grammar bitset offset은 증명하지 않는다. grammar 없는 top-p가 맞아도 padded/adapter vocab mask를 증명하지 않는다. fixture마다 어떤 contract를 검증하는지 명시한다.

raw local logits와 local vocab ranges를 먼저 비교한다. 다음으로 processor 전 global reference logits를 semantic coordinate로 재조립한 결과를 기준으로 둔다. grammar/repetition/banned mask 적용 뒤 eligible logits, temperature 뒤 logsumexp, top-p candidate set, random draw, selected token을 비교한다.

변경 path는 full tensor를 만들지 않으므로 동일한 full-logits checkpoint를 억지로 넣을 필요가 없다. 대신 global max와 logsumexp, merged candidate `(score,global_id)`, cumulative mass, cutoff score를 기록한다. reference full logits에서 계산한 scalar/set과 비교한다. implementation shape가 달라도 의미 checkpoint는 맞출 수 있다.

top-p mismatch가 cutoff 후보 수 부족에서 처음 나타나면 distributed candidate expansion algorithm이 필요하다. 충분한 upper bound를 한 번에 모으거나 iterative threshold/refinement를 할 수 있다. 어떤 설계를 택하든 exactness와 worst-case communication을 설명한다. 평평한 분포에서는 거의 full vocab이 필요할 수 있다는 한계를 숨기지 않는다.

grammar mismatch가 rank 1부터 나타나면 global token ID→local index 변환을 본다. allowed token 76042가 TP=4에서 어느 rank의 어떤 local index인지 계산한다. original/padded/adaptor range별 mask를 만든 뒤 invalid padded row를 항상 제외한다. grammar state는 request row reorder와 함께 움직여야 한다.

### latency 개선을 같은 byte 열에서 다시 계산한다

full gather가 B=64, V=152064 FP32 약 37.1 MiB였고 매 rank에 복제되었다고 하자. local candidate 64개씩만 모으면 score FP32와 token ID int64를 합쳐 12 byte로 단순 계산할 때 `64 request×64 candidate×12×4 ranks≈192 KiB`다. communication은 크게 줄 수 있다.

그러나 candidate selection kernel, iterative expansion, global logsumexp reduction, grammar mask distribution이 추가된다. top-p가 평평한 분포에서는 candidate 64가 부족해 재시도가 생길 수 있다. 평균과 p99를 분리한다. logprobs exact normalization은 global logsumexp가 필요하다. communication byte만 줄었다고 end-to-end 이득을 확정하지 않는다.

returned top-5 extraction과 sampler candidate set을 분리하면 output API 비용을 줄이면서 sampling exactness를 보존할 수 있다. selected token owner가 정해진 뒤 그 token logprob와 requested top-5만 CPU로 보낸다. full logits를 CPU로 복사하지 않는다. 이 변화는 GPU→CPU sync와 JSON overhead도 줄일 수 있다.

분산 softmax에서 global max와 global sum reduction을 정확히 수행하면 full gather 없이 exact logprob normalization이 가능하다. 하지만 processor가 rank-local mask를 적용하기 전에 max를 구하거나 padded row를 sum에 포함하면 normalization이 틀린다. stage order를 18장의 policy 의미와 맞춘다.

### 분산 commit에서 새 race가 생기지 않는지 본다

distributed sampler가 rank별 candidate를 비동기로 모으면 driver가 token을 선택하기 전에 모든 rank의 processor state가 같은 step을 가리켜야 한다. cancellation된 request row가 candidate buffer에 남거나 row compaction mapping이 바뀌면 다른 request의 token이 선택될 수 있다. batch row와 request incarnation을 candidate tuple에 연결한다.

selected token broadcast가 끝나기 전에 다음 graph replay가 시작되지 않아야 한다. candidate collective와 token broadcast가 서로 다른 communication group/stream을 쓰면 event를 둔다. debug synchronization을 넣어 race가 사라진다고 root cause를 해결한 것은 아니다. buffer lifetime과 step generation을 고친다.

DP LM head나 TP all-to-all mode에서는 hidden row가 head owner로 이동하고 token 결과가 원 request owner로 돌아올 수 있다. request row permutation과 inverse mapping을 ledger에 넣는다. 값은 맞는데 특정 request latency만 길면 load imbalance와 all-to-all routing을 본다. 값이 바뀌면 mapping을 먼저 본다.

request A가 greedy, B가 grammar, C가 top-p인 mixed batch에서 row reorder가 일어나면 sampling parameter와 grammar state, RNG owner가 같은 permutation을 따라야 한다. logits row만 reorder하고 state tensor를 그대로 두면 각 단독 fixture는 통과하고 mixed fixture만 틀릴 수 있다. aggregate distribution 통계로는 찾기 어렵다.

### 수정 검증은 기능 mode와 workload를 교차한다

정확성 축은 greedy, top-k, top-p sharp/flat distribution, grammar sparse/dense allowed set, repetition/history, original/padded/adapter vocab, logprobs on/off다. 성능 축은 B=1/작은/큰, V, TP, prompt rows와 generated rows다. 모든 조합을 무차별 실행하는 대신 원인 경계를 덮는 pairwise와 boundary fixture를 고른다.

flat distribution fixture는 실제 language model 없이 synthetic logits tensor로 sampler unit contract를 검증할 수 있다. 이 집필 작업에서는 실행하지 않지만 test 설계를 source에 대응시킨다. model end-to-end fixture는 hidden→head→sampler handoff를 검증한다. unit test와 integration test가 증명하는 범위를 구분한다.

수정이 완료되면 raw logits는 변경 전과 동일하고, 모든 정책 mode에서 selected token/logprobs가 reference와 맞으며, target workload의 LM-head/sampler/collective 시간이 개선되어야 한다. p99 iterative expansion이 SLO를 깨지 않는지 본다. metric label과 effective distributed mode를 기록한다.

### 다른 사건에도 같은 경계를 적용한다

특정 vocab 범위에서만 오답이 나면 global/local ID와 shard weight mapping을 본다. added token에서만 그러면 adapter vocab과 tokenizer range를 본다. TP=1은 맞고 TP>1만 틀리면 padding/collective/distributed sampler를 본다. greedy는 맞고 stochastic만 다르면 processor와 RNG를 본다.

모든 token ID는 같은데 text만 다르면 이 장을 빠져나가 tokenizer/output processor로 간다. raw logits부터 다르고 final hidden은 같다면 head로 돌아온다. final hidden부터 다르면 12~15장의 first divergence로 돌아간다. 이 분기가 있어야 모든 생성 차이를 sampler 탓이나 model 탓으로 뭉개지 않는다.

logprobs만 느리면 필요한 hidden rows와 vocab operation을 계산한다. LM head time은 같고 API latency만 느리면 D2H, detokenization, serialization과 backpressure를 본다. TP를 늘렸는데 더 느리면 head weight shard 이득과 collective/작은-batch 비효율을 함께 본다.

### 독자가 가져갈 최종 mental model

마지막 hidden z는 vocabulary에 질문하는 vector다. LM head의 각 vocab row는 그 질문에 score로 답한다. TP는 답안지를 여러 rank에 나눠 놓았으므로 global token 좌표로 다시 합의해야 한다. processor는 request 계약에 맞지 않는 답을 제거하거나 score를 조정한다. sampler는 남은 분포와 RNG에서 하나를 고른다. broadcast와 commit이 그 선택을 다음 step의 공동 사실로 만든다.

이 비유의 한계도 분명하다. vocab row가 사람이 읽는 의미를 독립적으로 저장한 사전 항목은 아니고 hidden과 학습된 내적 좌표다. processor 순서와 distributed algorithm은 단순 채점 이상의 state machine이다. 그래서 직관 뒤에는 shape, dtype, vocab range, row identity, RNG counter, commit event가 따라야 한다.

코드 리뷰에서 최종 질문은 세 개다. 어느 hidden row가 head에 들어갔는가. 어느 global vocab scores가 실제 policy를 통과했는가. 어느 token ID가 모든 rank와 request state에 exactly once로 commit되었는가. 세 답이 source와 관측으로 닫히면 forward의 종점이 닫힌다.

첫 질문이 비면 padding·packing·speculative row selection을 본다. 둘째가 비면 head shard, padding vocab, processor와 distributed probability를 본다. 셋째가 비면 sampler owner, broadcast, request lifecycle과 next input buffer를 본다. 질문이 owner를 직접 가리키도록 만든 것이 이 장의 실용적 목적이다.

최적화의 “왜”도 이 세 질문 안에서 설명된다. selected-row projection은 필요 없는 vocab 질문을 만들지 않기 위해서다. vocab parallel은 거대한 head weight를 shard하기 위해서다. distributed sampling은 full logits 이동을 줄이기 위해서다. 각각 row contract, global-ID contract, policy exactness라는 대가를 갖는다.

대가를 기록하지 않은 최적화는 단순한 속도 옵션으로 보이지만, 실제로는 정답의 소유권을 이동시킨다. source에서 state handoff를 닫고 canonical byte 열에서 saved work와 extra work를 계산하며 differential에서 first divergence를 찾을 때만 안전하게 적용할 수 있다.

**source audit를 실제 함수 호출 순서로 수행한다**

코드 리뷰는 model class에서 `lm_head` 문자열을 찾는 것으로 끝나지 않는다. 먼저 model body가 final hidden을 반환하는 지점을 찾고 final norm이 body 안인지 wrapper 안인지 확인한다. pipeline stage 분기가 있다면 마지막 stage가 아닌 rank가 어떤 intermediate object를 반환하는지도 기록한다. 이 단계의 산출물은 `[row,D]` tensor와 global layer/position identity다.

그다음 caller가 logits가 필요한 row를 고르는 지점을 찾는다. prefill last token, prompt logprobs, decode, speculative verify가 같은 index builder를 쓰는지 mode별 branch인지 본다. index가 CPU에서 만들어져 GPU gather에 전달되는지, packed row compaction 뒤 갱신되는지, cancellation된 row를 제거하는지 확인한다. row index tensor의 length가 LM head의 M축을 결정한다.

선택된 hidden이 final norm 전인지 후인지 확인하고 storage dtype과 head input dtype을 적는다. final norm source가 body에 있다면 wrapper가 다시 norm하지 않는지 본다. model별 logit scale, soft cap 또는 output multiplier가 있으면 LM head 전/후 어느 위치인지 기록한다. 단순 `Wz` 기준에서 벗어나는 architecture-specific transform은 52장 사례로 연결한다.

LM head constructor에서는 original vocab size, padded size, added vocab size, TP local range와 bias를 적는다. tied configuration이면 embedding weight assignment 또는 loader mapping을 찾는다. parameter object가 같다는 정적 증거와 서로 다른 PP stage에 값이 복제된다는 증거를 구분한다. quant method가 head에 적용되는 조건과 제외 조건도 constructor/loader 양쪽에서 확인한다.

projection call에서는 local logits shape, output dtype, bias와 scale 적용 순서를 기록한다. custom quant method가 generic matrix multiplication을 대체한다면 그 return이 global logits인지 local shard인지 확인한다. function 이름이 `forward`라고 자동 gather를 가정하지 않는다. head class와 logits processor 사이 책임을 나눈다.

TP collective 경계에서는 input local ranges와 output global ordering을 확인한다. all-gather라면 rank concat order와 padding trim, adapter vocab rearrangement를 본다. reduce/top-candidate 방식이면 global ID offset과 tie-breaking을 본다. DP LM head 또는 all-to-all이면 hidden row permutation과 inverse mapping, request owner를 추가한다.

processor 직전 raw logits checkpoint는 model correctness의 종점으로 쓰기 좋다. 그러나 이 tensor가 padded rows를 이미 제거했는지, FP32로 cast되었는지, scale/softcap이 적용된 뒤인지 stage 이름을 명시한다. 두 구현을 비교할 때 `raw`라는 이름만 맞추지 말고 포함된 변환을 맞춘다.

sampling parameter는 request state에서 batch row tensor로 materialize된다. temperature, top-k/p, min-p, penalties, grammar mask, seed/generator가 어떤 row reorder를 따르는지 본다. batch object가 CPU list와 GPU tensor를 함께 갖는다면 update timing도 확인한다. request가 finish/cancel되어 compact될 때 stale row가 남지 않아야 한다.

sampler output이 local token index인지 global token ID인지 확인한다. token score/logprob와 parent/accepted index 같은 auxiliary output도 의미를 적는다. driver-only 결과라면 broadcast function과 group, device/stream을 찾는다. 모든 model rank가 next input을 만들기 전에 completion을 기다리는 edge를 연결한다.

마지막으로 scheduler/request state update까지 올라간다. selected token을 generated list, stop checker, detokenizer, next model input, cache commit이 각각 언제 읽는지 본다. 하나의 token object를 공유하는지 복제 event인지, failure/retry에서 exactly-once가 어떻게 구현되는지 적는다. model executor source만 읽고 이 edge를 닫지 않으면 forward 종점이 사용자 요청 수명과 연결되지 않는다.

## 16.11 token commit 경계의 다섯 대표 사건

### 16.11.1 사건 A — 특정 added token이 절대 선택되지 않는다

새 adapter가 tool-call용 token 152100을 추가했다. tokenizer는 이를 input으로 encode하고 embedding도 정상적으로 읽는다. 그러나 model은 아무리 강한 prompt에서도 이 token을 output으로 선택하지 않는다. base token logits와 hidden은 reference와 같다. 이때 “모델이 새 token을 학습하지 못했다”는 가설만 세우면 loader/mapping 결함을 놓친다.

먼저 LM head global vocab range가 152100을 포함하는지 본다. base V, adapter extra vocab, padded V의 순서와 local shard owner를 계산한다. added embedding row가 output head에 연결되었는지, adapter head weight가 로드되었는지, logits processor가 original vocab으로 trim하면서 added range까지 잘라 버리지 않는지 확인한다.

synthetic hidden을 실행하지 않더라도 source에서 added vocab shard indices와 trim/rearrange를 추적할 수 있다. 허용된 test에서는 해당 row weight와 global logit 위치를 직접 검증한다. token 152100이 rank 3 local index로 존재하지만 gather 뒤 위치가 padding 영역으로 분류된다면 mapping이 first divergence다.

수정 후 input embedding과 output head의 added token range, TP=1/4, adapter on/off, mixed base/adapter request를 검증한다. base request가 added token을 선택할 수 있어야 하는지 mask해야 하는지는 제품 계약을 따른다. isolation을 정확성으로 정의한 뒤 검사한다.

### 16.11.2 사건 B — PP를 켜자 tied model의 logits만 달라진다

PP=1에서는 Transformers reference와 맞고 PP=2에서 final hidden send/receive까지 같다. final norm도 같다. logits부터 다르다. embedding은 stage 0, LM head는 stage 1에 있고 config는 tied다. storage pointer가 다르다는 사실은 예상되지만 두 stage의 weight 값은 같아야 한다.

stage 1 loader가 artifact에 없는 `lm_head.weight`를 missing으로 건너뛰고 embedding replica에서 복사하지 않았다면 head가 잘못 초기화될 수 있다. 반대로 stage 0 embedding은 quantized인데 stage 1 head는 unquantized semantic replica를 올바르게 만들 수도 있다. loader의 tied-weight mapping과 PP missing-weight rules를 읽는다.

vocab shard별 weight digest를 embedding semantic row와 head row로 비교한다. PP stage와 TP rank를 모두 key에 넣는다. global row 0, shard boundary 전후, last original token, padding/added row를 고른다. head weight가 맞고 logits만 다르면 head dtype과 collective를 본다.

수정 후 PP=1/2, tied/untied fixture, artifact에 head weight 존재/생략 case를 검증한다. memory가 증가했더라도 PP 물리 복제가 설계상 필요한지 측정한다. semantic tie와 physical alias를 혼동하지 않는다.

### 16.11.3 사건 C — stop token은 선택됐는데 한 step 더 계산한다

sampler trace에는 EOS global token ID가 step s에서 선택되어 있다. 사용자 stream은 정상 종료하지만 GPU는 step s+1에서 해당 request row를 한 번 더 실행한다. 이 현상은 LM head의 token 선택이 아니라 commit과 scheduler visibility 사이 문제다.

selected token broadcast가 driver에만 도착하고 scheduler worker의 finished mask가 다음 batch build 뒤에 갱신되는지 본다. stop checker가 detokenized text thread에만 있고 GPU scheduler는 max token count만 보는지도 확인한다. request state transition timestamp와 next batch row list를 연결한다.

한 번 더 실행된 token이 사용자에게 emit되지 않아 correctness가 겉으로 맞아도 goodput과 KV allocation, ITL tail에 비용이 남는다. cancellation과 stop을 scheduled/executed/accepted/delivered 원장으로 나누는 이유다. 3장의 goodput 회계와 연결한다.

fix 후 EOS, stop-token set, stop string, length finish, client cancellation을 구분한다. stop string은 tokenizer buffer 때문에 token 선택 시점에 알 수 없을 수 있다. 모든 경우를 무조건 sampler 직후 중단하도록 바꾸지 않는다. 어느 owner가 earliest safe finish를 알 수 있는지 정의한다.

### 16.11.4 사건 D — top-p만 batch composition에 따라 달라진다

같은 request와 seed를 단독으로 실행하면 일정하지만 다른 stochastic request와 batch되면 token sequence가 달라진다. raw/processed logits와 probability는 같다. 이때 global generator가 batch row 순서대로 random number를 소비한다면 concurrency가 request의 draw 좌표를 바꾼다.

제품이 동일 seed의 request-level reproducibility를 약속하지 않으면 이것이 bug가 아닐 수 있다. 그러나 약속한다면 request별 generator나 counter-based coordinate가 필요하다. `seed, request incarnation, generated position, sample branch`로 draw를 주소화한다. speculative reject가 random count를 어떻게 소비하는지도 정한다.

분산 rank가 각각 RNG를 소비하는 구조에서는 selected owner를 하나로 고정하거나 동일 counter를 사용한다. rank local floating difference가 후보 cutoff를 바꾸면 같은 uniform draw여도 token이 다를 수 있으므로 distribution parity도 확인한다. seed만 맞추고 모든 차이를 RNG로 돌리지 않는다.

### 16.11.5 사건 E — logprobs는 맞지만 순서가 뒤섞인다

returned top token IDs와 values 집합은 reference와 같지만 request B가 A의 logprobs를 받는다. DP LM head all-to-all에서 hidden rows를 head workers로 보냈다가 inverse permutation이 잘못된 경우를 생각할 수 있다. aggregate accuracy test는 통과하고 mixed batch에서만 나타난다.

candidate tuple에 request incarnation과 original row를 붙여 all-to-all send index, head compute row, return index를 연결한다. token ID global mapping과 request row mapping은 서로 다른 permutation이다. 둘을 하나의 offset으로 처리하지 않는다. cancellation로 batch가 compact될 때 generation counter가 같은지도 본다.

수정 후 서로 다른 sampling parameter와 쉽게 구분되는 synthetic logits를 가진 request를 섞는다. batch order와 completion order를 바꾸고 inverse mapping을 검증한다. 성능 trace에서 all-to-all load imbalance와 mapping correctness를 별도 metric으로 본다.

**source-only 작업에서 정직하게 남겨야 할 미확정점**

이 장은 source를 읽어 가능한 path와 invariant를 확인하지만 실제 배포의 effective branch를 관측하지 않았다. 따라서 특정 option이 몇 퍼센트 빠르다거나 어떤 GPU에서 반드시 어느 sampler를 선택한다고 주장하지 않는다. fixed source line은 구현 가능성과 default resolution을 증명하며 runtime config/log/trace가 실제 선택을 증명한다.

Canonical 표의 FLOP과 byte는 operand 하한 또는 단순 모델이다. kernel fusion, allocator, cache, interconnect topology, overlap을 포함한 측정값이 아니다. 예를 들어 37.1 MiB logits가 있다고 매 step 정확히 그만큼 network를 쓴다고 단정하지 않는다. collective algorithm과 distributed sampler가 바꿀 수 있다.

synthetic fixture는 개인정보 없이 boundary를 검증하기 좋지만 실제 language distribution과 workload tail을 대신하지 않는다. flat logits는 top-p worst case를 검증하고 real cohort는 후보 분포와 SLO를 검증한다. 두 증거를 서로 대체하지 않는다.

## 16.12 final token을 다음 serving state로 넘긴다

10장에서는 embedding에서 residual stream을 붙잡았다. 11장은 norm과 projection, 12장은 head partition, 13장은 attention 의미, 14장은 position과 cache, 15장은 dense/MoE/recurrent update를 확대했다. 이 장은 final hidden이 global token ID로 commit되는 순간까지 와서 그 stream을 닫았다.

한 token이 끝났지만 서비스는 끝나지 않았다. commit된 ID는 다시 embedding input이 되어 같은 경로를 반복하고, scheduler는 완료/취소/새 요청과 함께 다음 batch를 만든다. KV와 recurrent state는 새 position으로 진행한다. 그러므로 forward 종점은 다음 iteration의 시작점이다.

독자가 이 편에서 가져갈 핵심은 module 이름의 암기가 아니다. request-position-layer-row라는 좌표와 tensor owner, persistent state, partial/global completeness, commit이라는 다섯 질문이다. framework와 model architecture가 달라져도 이 질문으로 source를 다시 걸을 수 있다.

이제 48장에서 config가 실제 module/head/vocab/cache shape를 어떻게 선택하는지 읽고, 52장에서 한 모델을 수직으로 추적할 준비가 되었다. serving engine 편에서는 이 model path를 scheduler step, batch formation, CUDA kernel과 metric timeline에 연결한다. 최종 token ID를 잃지 않는 것이 다음 모든 분석의 기준점이다.

**같은 요청을 네 구현에서 비교할 때 번역해야 하는 것**

Transformers 기준선은 dense batch shape와 causal LM output object를 사용하기 쉽다. `logits_to_keep=1`이면 마지막 sequence row를 선택할 수 있지만 padding이 있는 여러 request에서 각자의 마지막 유효 row를 자동으로 의미하는지 API contract를 확인해야 한다. serving engine의 packed selected indices와 비교할 때 logical `(request, prediction_position)`으로 정규화한다.

vLLM은 model `compute_logits`와 common `LogitsProcessor`, runner/sampler 사이에서 책임을 나눈다. local head weight가 vocab parallel이고 processor가 gather/trim을 담당할 수 있다. 따라서 Transformers full `[B,V]`와 vLLM head 직후 local `[B,Vlocal]`을 직접 allclose하지 않는다. global ID로 재조립한 동일 stage를 비교한다.

SGLang은 logits processor output에 next-token logits 이외 prompt logprobs용 field와 normalization 결과가 함께 있을 수 있다. sampler backend와 DP/TP LM-head mode에 따라 global tensor가 materialize되는 위치도 달라질 수 있다. 내부 dataclass field 이름이 같아 보여도 raw/processed/normalized stage를 확인한다.

llama.cpp에서는 graph output이 어느 token rows의 logits를 materialize하는지 context/batch request가 결정한다. sampler chain은 mutable token candidate array에 transform을 적용할 수 있다. PyTorch processor list와 함수 이름을 맞추기보다 `selected hidden rows → vocab scores → policy transforms → selected global token`이라는 의미 순서를 맞춘다.

비교 fixture는 vocab 전체를 매번 보존하지 않아도 된다. first divergence를 찾기 위한 coarse 단계에서는 selected row hidden digest, final norm bounded slice, raw logits의 max/min/norm과 고정 global token indices, top-k set을 쓴다. 값이 갈리는 stage에서만 synthetic 작은 vocab 또는 full logits 접근이 허용된 test로 확대한다.

vocab revision도 비교 identity다. tokenizer vocab size가 같아도 token ID→bytes mapping이나 added token range가 다를 수 있다. LM head global row 42의 숫자가 같다는 사실과 그것이 같은 token 의미라는 사실을 분리한다. artifact tokenizer digest와 model config vocab, head original/padded ranges를 함께 고정한다.

### 독자가 직접 해 볼 마지막 손계산

TP=2, original vocab 6, padded vocab 8을 가정한다. rank 0은 global 0~3, rank 1은 4~7을 소유한다. token 6과 7은 invalid padding이다. rank 0 logits `[0,1,2,3]`, rank 1 `[4,5,100,90]`이면 trim 전 argmax는 invalid token 6이다. original range로 trim하면 valid argmax는 token 5다.

token 5에 grammar mask가 0이고 allowed set이 {1,4}라면 processed argmax는 token 4 score 4다. temperature와 top-p를 적용하기 전에 invalid/banned mask가 들어가는 계약이라고 가정한 계산이다. 실제 processor order는 18장과 implementation source를 따른다. 이 작은 예는 순서가 결과를 바꾼다는 사실을 보여 준다.

rank 1의 local index 0은 global token 4다. sampler가 local 0을 반환하면 driver가 offset 4를 더해야 한다. 이미 global 4를 반환하는 API에서 다시 더하면 token 8이라는 out-of-range가 된다. return contract를 source에서 확인하지 않고 caller가 추측하면 안 된다.

selected token 4를 모든 rank에 broadcast한 뒤 next embedding은 global row 4를 읽는다. rank-sharded embedding이 입력 token을 owner range에 따라 mask/reduce한다면 모든 rank가 같은 global ID를 받아야 한다. 이로써 output vocab parallel의 global-ID contract가 다음 input embedding contract와 연결된다.

이 손계산에서 hidden과 head GEMM은 주어졌고 문제는 padding, grammar와 global offset에 있었다. 실제 장애에서도 first divergence 앞의 맞는 경계를 보존하면 불필요하게 model layer 전체를 다시 조사하지 않는다. “답이 틀렸다”를 “rank 1 processed local winner를 global ID로 commit하는 edge가 틀렸다”로 바꾸는 것이 깊은 디깅이다.

### global vocabulary 확장 전에 묶어 두는 현재 경계

LM head는 종종 attention과 MLP 뒤에 붙은 단순 linear로 취급된다. 수학만 보면 맞다. serving 관점에서는 거대한 vocab weight, selected-row shape, TP/DP communication, request별 policy와 RNG, output logprobs, next-step commit이 만나는 교차점이다. 작은 linear라는 이름이 시스템 책임의 크기를 말해 주지 않는다.

그래서 최적화도 두 층으로 읽는다. 행렬 곱 자체의 dtype·kernel·weight shard를 개선하는 층과, 필요 row/candidate/communication만 남기는 시스템 층이다. 두 번째 층은 더 큰 saved work를 만들 수 있지만 semantic policy를 보존해야 한다. top-p incident처럼 전송을 줄인 결과 정답 집합을 줄여 버리면 최적화가 아니다.

반대로 모든 logits를 gather하고 CPU로 보내는 기준선은 이해하기 쉽지만 scale에서 비싸다. 기준선의 가치는 의미 oracle을 제공하는 데 있다. optimized path가 같은 global score와 policy 결과를 더 적은 materialization으로 만든다는 것을 differential로 증명한다. 이해하기 쉬운 reference와 빠른 serving path를 적으로 만들지 않는다.

최종적으로 model forward의 결과를 말할 때는 stage를 붙인다. `final_hidden`, `raw_global_logits`, `processed_distribution`, `selected_token`, `committed_token`, `emitted_text`다. stage를 붙이는 습관 하나가 관측과 source owner를 정렬하고, 독자가 다음에 어디를 파야 하는지 알려 준다.

장애 기록도 같은 언어를 사용한다. “세 번째 token이 다르다” 대신 “step 2의 final hidden과 raw global logits는 허용 오차 안에서 같고, grammar mask 뒤 eligible set에서 global token 76042가 누락되어 processed distribution부터 갈린다”라고 쓴다. 이 문장은 model layer, LM head weight와 RNG를 우선 가설에서 내리고 vocab-local grammar mapping으로 조사 범위를 좁힌다.

성능 기록은 “logprobs가 느리다” 대신 “prompt logprobs가 selected rows를 64에서 8192로 늘리고 FP32 vocab projection output과 TP gather가 생겨 LM-head 구간 p99가 증가했다”라고 쓴다. 그러면 row pruning, distributed normalization, output extraction 중 무엇이 saved work를 만드는지 계산할 수 있다.

수정 뒤에는 같은 stage로 돌아온다. eligible set과 selected global token이 reference와 일치하고, broadcast 뒤 모든 rank의 next input ID와 logical position이 같으며, 목표 workload에서 collective와 output materialization이 줄었는지 확인한다. 사용자 text만 우연히 같거나 평균 latency만 내려간 것은 충분하지 않다.

이 장의 최종 invariant는 다음과 같다. **동일한 selected hidden row와 model/head identity에서 만들어진 global vocabulary score가 request policy와 하나의 sampling state를 거쳐 정확히 하나의 global token ID로 결정되고, 그 ID가 모든 실행 owner와 request lifecycle에 같은 step으로 commit되어야 한다.** 이 문장을 source handoff와 canonical 네 행, first-divergence 증거로 설명할 수 있으면 마지막 hidden에서 다음 token까지의 경계가 완성된다.

다음 iteration에서는 바로 그 committed ID가 embedding row가 된다. 따라서 종점의 global-ID·incarnation·position 합의가 깨지면 다음 forward 전체가 오염되고, 합의가 맞으면 새 step도 같은 좌표계에서 다시 추적할 수 있다.

이 순환이 생성 서비스의 기본 심장박동이다.

같은 작은 vocab fixture로 수정 뒤 global token 좌표와 다음-step handoff도 다시 검증한다.

## 16.13 rank-local vocabulary를 global token으로 복원하는 전 과정을 펼쳐 본다

여기서는 앞에서 소개한 vocabulary parallel을 실제 디버깅에 쓸 수 있을 만큼 천천히 다시 걷는다. 핵심은 `local index`와 `global token ID`가 같은 정수가 아니라는 점이다. 단일 GPU에서는 우연히 둘이 같아서 이 구분이 보이지 않는다. TP를 켜는 순간 하나의 vocabulary 행렬은 여러 rank에 나뉘고, 각 rank가 출력하는 열 번호는 자기 조각 안의 좌표가 된다. 서버가 사용자에게 확정해야 하는 값은 tokenizer와 다음 embedding이 이해하는 전역 좌표다. 이 변환이 한 번 빠지거나 두 번 적용되면 행렬 곱이 완벽히 맞아도 다른 단어가 생성된다.

고정한 vLLM 소스의 `VocabParallelEmbeddingShardIndices`는 이 문제를 단순한 `start`, `end` 두 값으로 축약하지 않는다. 원래 vocabulary와 추가 vocabulary 각각에 대해 padded 범위와 실제 범위를 나눈다. 생성자는 원래 크기를 `pad_vocab_size`로 올린 뒤 TP rank별 구간을 계산하고, 실제 끝은 `min(..., org_vocab_size)`로 잘라 낸다. 이어지는 `ParallelLMHead`는 같은 분할 정보를 물려받는다.

즉 padding은 커널에 좋은 행렬 모양을 만들기 위한 저장 좌표이고, 실제 token 공간은 tokenizer가 의미를 부여한 의미 좌표다. [vLLM vocabulary 분할과 실제 범위 계산](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/vocab_parallel_embedding.py#L198-L377)

SGLang의 대응 클래스도 원본, 추가, padded 범위를 별도 필드로 보존한다. 특히 입력 embedding에서 token이 원래 구간에 속하는지 추가 구간에 속하는지 mask를 만들고, 유효한 local offset을 계산한다. output head만 읽고 “rank 번호 곱하기 local size를 더하면 끝”이라고 생각하면 adapter가 붙인 extra vocabulary에서 틀릴 수 있다. 원본 vocabulary 뒤에 padding hole이 있고 그 뒤에 추가 token이 배치되는 구현에서는 물리 행 번호와 의미 token ID 사이에 한 번 더 번역이 필요하기 때문이다.

[SGLang shard index와 입력 mask 계산](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/vocab_parallel_embedding.py#L80-L188), [SGLang padded·added 범위 구성](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/vocab_parallel_embedding.py#L263-L383)

### 열세 개 token을 네 rank에 나누는 손계산

의미 있는 token이 13개, TP가 4이고, 구현이 전체 행 수를 16으로 padding한다고 하자. 설명을 위해 각 rank가 연속된 네 행을 가진다고 놓는다. rank 0의 물리 구간은 0~3, rank 1은 4~7, rank 2는 8~11, rank 3은 12~15다. 실제 token ID는 0~12뿐이다. 따라서 마지막 rank의 local row 0만 token 12이고 local row 1~3은 계산 편의를 위한 빈 행이다.

한 decode row에서 local logits가 다음과 같다고 하자.

| rank | local logits | local argmax | 전역으로 번역한 후보 |
|---|---|---:|---:|
| 0 | `[1.0, 2.0, 0.5, -1.0]` | local 1, score 2.0 | token 1 |
| 1 | `[3.0, 1.5, 2.2, 2.8]` | local 0, score 3.0 | token 4 |
| 2 | `[0.0, 4.0, 3.9, -2.0]` | local 1, score 4.0 | token 9 |
| 3 | `[3.7, 90.0, 80.0, 70.0]` | local 1, score 90.0 | invalid padding |

물리 tensor에 그대로 `argmax`를 적용하면 padding row가 이긴다. “padding weight는 보통 0이니 괜찮다”는 주장은 안전 계약이 아니다. checkpoint loader가 쓰지 않은 행을 어떤 값으로 초기화하는지, quantization scale이나 uninitialized buffer가 무엇을 남기는지, fused kernel이 그 행을 계산하는지는 별개의 문제다. 올바른 계약은 실제 vocabulary 밖의 score를 선택 이전에 후보에서 제거하는 것이다.

vLLM의 고정 소스는 all-gather 경로에서 logits를 `org_vocab_size`까지 자르고, 별도 distributed 경로에서는 shard의 원본 끝을 기준으로 padding entry를 mask한다. [vLLM global trim과 shard padding mask](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/logits_processor.py#L63-L200)

padding을 제거하면 네 rank 후보는 `(2.0,1)`, `(3.0,4)`, `(4.0,9)`, `(3.7,12)`가 되고 global greedy 결과는 token 9다. 여기서 reduction payload가 score 하나뿐이면 winning rank는 알 수 있어도 token을 확정할 수 없다. 적어도 score와 global ID의 쌍, 또는 score와 rank-local ID 뒤의 별도 owner 전달이 필요하다. 동점도 정의해야 한다. rank 2 token 9와 rank 3 token 12의 score가 모두 4.0이라면, 작은 global ID를 고르는지 rank reduction 순서를 따르는지 정하지 않으면 TP topology나 collective 구현이 바뀔 때 greedy 출력도 바뀔 수 있다.

이 손계산을 테스트로 만들 때는 세 종류의 행을 의도적으로 넣는다. 첫째는 정상 token보다 훨씬 큰 padding score다. 둘째는 서로 다른 rank에 놓인 동일 score다. 셋째는 adapter 추가 token과 원래 token의 경계다. 테스트가 “argmax 값이 9다”만 검사하면 tie-break와 added-vocab 번역을 놓친다. 후보 tuple의 `(score, global_id, validity, owner_rank, local_id)`를 보존하여 각 변환을 검사한다.

### full all-gather가 제공하는 단순성과 치르는 비용

가장 이해하기 쉬운 구현은 각 rank의 local logits를 vocabulary 축으로 모두 모아 `[rows, padded_vocab]`을 만들고, 실제 vocabulary 크기로 자른 뒤 일반 sampler를 실행하는 것이다. vLLM의 `LogitsProcessor`는 platform 선택에 따라 all-gather를 사용하고, 모은 뒤 원래 vocabulary 크기로 trim한다. SGLang은 `skip_all_gather`, TP 크기, attention DP 크기를 보고 gather 여부와 group을 결정하며, DP-attention 경로에서는 tensor를 모은 뒤 permute와 reshape로 global vocabulary 배열을 만든다. “두 엔진 모두 logits를 모은다”는 요약은 너무 거칠다.

어느 group이, 어떤 row 배치로, 어느 단계에서 모으는지를 알아야 trace와 메모리를 해석할 수 있다. [vLLM logits gather 경로](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/logits_processor.py#L56-L152), [SGLang gather 선택과 global tensor 구성](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/logits_processor.py#L283-L328)

비용을 요청 모양으로 계산해 보자. 원래 vocabulary 151,936, padded vocabulary 152,064, decode row 64개, logits가 FP32라면 한 rank가 최종 full tensor를 보관하는 크기는 `64×152064×4 = 38,928,384 byte`, 약 37.1 MiB다. TP=4라면 각 rank local tensor는 약 9.28 MiB다.

ring all-gather의 단순 payload 관점에서 rank마다 local chunk의 세 배에 해당하는 데이터를 받고, 최종적으로 네 조각을 보유한다. 실제 링크 byte와 시간은 collective algorithm, topology, chunking, overlap에 따라 달라지므로 27.8 MiB 수신량을 latency라고 읽어서는 안 된다. 그래도 매 decode step마다 37.1 MiB full materialization이 생길 수 있다는 capacity·allocation 하한은 남는다.

row가 64가 아니라 1이면 full tensor는 약 594 KiB다. 작은 값처럼 보여도 매 token 반복되고 collective launch와 synchronization이 붙는다. 반대로 prompt logprobs 때문에 8,192 row를 투영하면 FP32 full logits는 약 4.64 GiB다. 실제 엔진이 모든 row를 동시에 materialize하지 않고 chunking하거나 필요한 값만 뽑는 이유가 여기 있다. 옵션 하나가 “logprobs를 반환한다”에서 끝나지 않고 selected-row 수, LM-head GEMM M축, gather payload, output serialization을 함께 바꾸는 것이다.

full gather의 장점도 명확하다. grammar, repetition penalty, arbitrary logit bias, top-p처럼 전역 score 배열을 기대하는 기존 processor를 그대로 적용하기 쉽다. 디버깅할 때 reference tensor를 저장하기도 쉽다. 따라서 이를 무조건 나쁜 구현으로 부르면 안 된다. workload에서 vocabulary와 row 수가 작거나, 복잡한 policy가 full distribution을 요구하거나, 단순한 correctness 경로가 더 중요한 경우에는 합리적이다. 최적화 질문은 “gather가 존재하는가”가 아니라 “요청 계약이 요구하는 정보보다 얼마나 더 많이 materialize하고 이동하는가”다.

### global greedy와 exact logprob는 요구하는 collective가 다르다

greedy는 각 rank의 유효 local maximum 하나를 모으면 된다. rank r의 시작 token을 `s_r`라 하고 local winner index를 `j_r`라 하면 후보는 `(m_r, s_r+j_r)`다. 전역에서 score 최대를 고르면 exact greedy가 된다. padding mask와 deterministic tie-break가 먼저 정의돼 있어야 한다. vLLM 고정 소스의 distributed greedy 쪽은 local maximum과 index를 pair로 만들고 gather한 뒤, shard의 `org_vocab_start_index`를 global ID 계산에 사용한다.

이 코드는 “full vocabulary가 없어도 greedy는 exact할 수 있다”는 구현 증거다. [vLLM distributed greedy pair와 global offset](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/logits_processor.py#L168-L203)

하지만 사용자가 선택 token의 정확한 logprob를 요구하면 score 최대 하나로는 부족하다. global log probability는 `z_t - log(sum_j exp(z_j))`다. 안정적으로 계산하려면 먼저 모든 shard의 local maximum에서 global maximum `m`을 구하고, 각 rank가 `sum exp(z_j-m)`을 계산한 뒤 그 합을 all-reduce한다. 마지막에 `log Z = m + log(global_sum)`을 얻는다. selected token의 logit은 그 token owner가 제공하거나 후보 결과에 이미 포함한다. 이 방법은 full logits 없이 선택 token의 exact logprob를 계산할 수 있지만 collective가 적어도 global max와 global sum이라는 두 reduction 의미를 가진다. fused reduction이더라도 수학적 소유권은 사라지지 않는다.

top-k logprobs는 각 shard가 local top-k를 보내고 전역에서 그 후보들을 다시 top-k로 줄이면 exact하다. 전역 top-k 안에 들어갈 token은 반드시 자기 shard의 local top-k 안에도 들어가기 때문이다. TP=8, k=5이면 full 152k score 대신 최대 40개 후보를 합칠 수 있다. 그러나 logprob 값을 정규화하려면 여전히 전체 vocabulary의 logsumexp가 필요하다. “top-k 후보만 정확하다”와 “그 후보의 확률도 정확하다”는 다른 계약이다.

top-p는 더 까다롭다. 누적 확률이 p에 도달할 때까지 몇 개 후보가 필요한지 logits 분포에 따라 바뀐다. 모든 logits가 거의 평평하고 p=0.95라면 vocabulary 대부분이 필요하다. 각 rank에서 고정 k개만 보내는 근사는 충분한 후보를 보장하지 못한다. adaptive threshold나 분산 selection 알고리즘을 쓸 수 있지만, 구현이 그런 보장을 제공하는지 source로 확인해야 한다. 단순히 “distributed sampler”라는 이름만 보고 full sampler와 같은 결과라고 가정하지 않는다.

grammar mask와 logit bias도 소유권 문제다. global token 9를 금지하려면 rank 2의 local row 1을 찾아 mask해야 한다. global mask 배열을 모든 rank에 복제할 수도 있고, 각 rank가 자기 range만 slice할 수도 있다. 후자가 통신과 메모리를 줄이지만 added vocabulary와 padding hole을 정확히 번역해야 한다. processor가 gather 전 local logits에 적용되는지, gather 후 global logits에 적용되는지에 따라 같은 mask 표현을 재사용할 수 없다. 옵션이 처리 순서를 바꾸면 mask 자료구조의 좌표계도 함께 바뀌어야 한다.

### observability는 tensor 크기보다 의미 좌표를 기록해야 한다

운영 지표에 `lm_head_ms` 하나만 있으면 projection과 collective, trim, processor, sampler가 한 덩어리로 보인다. 최소한 selected row 수, original/padded/added vocab 크기, TP 크기, local logits dtype·byte, gather 또는 distributed reduction 방식, 반환 logprob 수를 request cohort와 연결한다. full logits 값 자체를 production metric으로 내보내라는 뜻은 아니다. 민감하고 비싸다. shape와 ownership, timing, bounded digest만으로도 많은 문제를 좁힐 수 있다.

정확성 debug snapshot에는 고정된 소수 global token의 raw score를 넣는다. 각 global token에 대해 owner rank, local index, validity를 같이 기록한다. global token 9의 score를 요청했는데 rank 2 local 1을 읽었다는 사실이 보여야 한다. top-k에는 local 후보와 merge 뒤 global 후보를 따로 남긴다. 그러면 “local GEMM부터 값이 다름”, “padding mask 뒤 달라짐”, “global offset에서 달라짐”, “policy 뒤 달라짐”을 구분할 수 있다.

collective metric은 group identity가 중요하다. TP group, attention-TP group, DP scatter 경로가 섞인 환경에서 단순 `all_gather_count`는 원인을 말해 주지 않는다. group size, input/output shape, operation purpose를 label cardinality가 폭발하지 않는 범주형 값으로 기록한다. request ID를 Prometheus label에 직접 넣지 않고 trace exemplar나 sampled debug record로 연결한다. p50만 보면 작은 decode row가 지배하므로 prompt-logprobs cohort와 large-batch cohort의 p95·p99를 분리한다.

성능 수정의 완료 조건도 두 축이다. 의미 축에서는 동일한 raw global logits 또는 허용된 분산 알고리즘의 exact 결과, 동일 eligible set, 동일 selected global ID와 logprob를 확인한다. 비용 축에서는 local/full tensor byte, collective payload 또는 횟수, peak temporary memory, LM-head 구간 latency를 확인한다. full gather를 없앴지만 CPU synchronization을 추가해 p99가 악화되거나, top-k는 같지만 logprob normalization이 달라졌다면 완료가 아니다.

### 소스를 읽을 때 실제로 따라갈 질문

첫 질문은 head가 돌려주는 tensor의 마지막 축이 무엇인가다. `ParallelLMHead.forward` 직후라면 대개 local padded vocabulary다. 하지만 quantization method가 custom output을 만들거나 일부 platform이 다른 경로를 선택할 수 있다. 함수 이름이 `logits`라고 해서 global이고 trim된 값이라고 믿지 않는다. shape 계산과 다음 caller를 읽는다.

둘째는 원래 vocabulary 크기를 누가 아는가다. model config, tokenizer, embedding/head layer, logits processor가 서로 다른 필드로 가질 수 있다. adapter가 추가 token을 붙이면 `num_embeddings`, `org_vocab_size`, `num_embeddings_padded`가 모두 달라진다. 이 값들이 artifact revision과 함께 일치하는지 확인한다. tokenizer가 13개 token을 내는데 head는 12개만 실제로 로드했거나, head는 extra row를 가지는데 output processor가 원래 크기로 너무 일찍 자르면 정상 token이 사라진다.

셋째는 global tensor가 언제 생기는가다. SGLang 고정 소스는 `do_tensor_parallel_all_gather`를 `skip_all_gather`, TP/attention TP 크기와 함께 결정한다. 실제 gather helper는 DP-attention이면 global buffer를 만들고 rank 축과 row 축을 재배열한다. 따라서 line profiler에서 `forward`만 보면 통신이 projection 내부처럼 보일 수 있고, memory snapshot에서는 permute 뒤 contiguous 여부가 추가 비용을 만들 수 있다. [SGLang logits gather 구현](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/layers/logits_processor.py#L672-L803)

넷째는 sampler가 local 또는 global 중 무엇을 반환하는가다. caller가 offset을 더하는지, processor가 이미 global pair를 만드는지 읽는다. 이름이 `token_id`라는 이유로 global이라고 단정하지 않는다. 반환 dataclass와 다음 embedding 호출까지 따라가야 계약이 닫힌다. token stream serializer가 정상 문자열을 만들었다고 해서 안전한 것도 아니다. 잘못된 global ID가 우연히 유효 범위 안이면 전혀 다른 정상 token으로 decode되며 예외가 나지 않는다.

다섯째는 fallback이다. distributed greedy가 지원되지 않는 policy에서 full gather로 돌아가는지, 옵션 조합을 거부하는지, 조용히 근사하는지 확인한다. 좋은 문서는 빠른 path뿐 아니라 어떤 입력에서 빠른 path가 해제되는지 설명한다. 운영자는 feature flag를 켰는데 traffic 일부만 느린 이유를 fallback 조건에서 찾을 수 있다.

이 절의 결론은 단순하다. vocabulary parallel은 행렬을 나누는 기법이면서 동시에 token 의미를 분산했다가 다시 합의하는 프로토콜이다. 행렬 계산의 정확성, padding의 비의미성, local/global 좌표 변환, collective의 충분성, request policy의 적용 위치를 한 묶음으로 검증해야 한다.

**여섯 통제 실험.** 실험 A는 같은 hidden과 weight로 dense LM head와 vocabulary-parallel gather의 global logits를 비교한다. 실험 B는 공통 logit offset을 더해 argmax와 softmax가 불변인지 본다. 실험 C는 rank 경계의 최대 logit을 만들어 local argmax만 쓰는 오류를 드러낸다. 실험 D는 tied weight를 copy로 바꾸고 한 update 뒤 alias divergence를 확인한다. 실험 E는 dummy vocabulary row를 최대로 만들어 logical-vocab mask가 생성 전에 막는지 본다. 실험 F는 output projection dtype만 바꿔 first differing logit과 token margin을 기록한다. 각 실험은 정상 control, 의도한 실패와 복원 뒤 회귀를 함께 남긴다.

## 16.14 tied embedding의 로드·별칭과 “정상 범위의 틀린 token” 사건

입력 embedding과 LM head가 같은 weight를 공유하는 모델은 parameter 수를 줄이고 학습에서 두 표현 공간을 결속한다. 그러나 config의 `tie_word_embeddings=true`가 곧 runtime에서 언제나 하나의 storage만 존재한다는 뜻은 아니다. 모델 생성, checkpoint naming, quantization wrapper, pipeline rank, weight loader가 그 계약을 실제 객체와 storage로 구현해야 한다. 이 경계를 놓치면 메모리 중복만 생기는 경우도 있고, 더 위험하게는 embedding과 head 중 하나만 갱신되거나 잘못된 shard를 참조해 출력이 달라질 수 있다.

vLLM의 고정 Llama 구현은 output head를 만든 뒤 `tie_word_embeddings`가 참이면 `lm_head.tie_weights(model.embed_tokens)`의 반환값을 다시 head에 대입한다. `ParallelLMHead.tie_weights`는 quantization method에 위임한다. 즉 단순한 `head.weight = embed.weight` 한 줄로 일반화할 수 없다. quantization 방식이 alias, wrapper 또는 별도 표현을 어떻게 연결하는지가 계약의 일부다. loader는 tied 상태에서 별도 `lm_head.weight`가 checkpoint에 없을 수 있는 naming 상황도 처리해야 한다.

[vLLM Llama head 생성과 tie](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/llama.py#L485-L495), [vLLM ParallelLMHead tie 위임](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/vocab_parallel_embedding.py#L520-L578)

SGLang의 고정 Llama 구현은 tied config에서 `self.lm_head = self.model.embed_tokens`로 module 객체 자체를 연결하고, untied일 때 `ParallelLMHead`를 새로 만든다. 같은 파일의 loader에는 checkpoint와 실행 표현이 다를 때 embedding weight를 head 쪽 이름으로 복사해 로드하는 분기가 있다. 따라서 “SGLang은 항상 alias, vLLM은 항상 복사”처럼 한 문장으로 끝내면 틀린다. model class의 구성 단계, quantization path, loader 분기를 함께 읽어야 한다.

[SGLang Llama tied head 구성](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/llama.py#L529-L541), [SGLang Llama weight loader 분기](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/models/llama.py#L706-L792)

### 세 개의 “공유”를 구분한다

첫째는 의미 공유다. 학습 artifact가 input embedding과 output projection을 같은 parameter로 정의한다. 둘째는 객체 공유다. runtime module 두 경로가 같은 parameter 객체를 참조한다. 셋째는 storage 공유다. 실제 device memory의 data pointer가 같다. quantized runtime은 의미상 같은 weight에서 input lookup용 표현과 output GEMM용 packed 표현을 따로 만들 수도 있다. 이때 storage가 둘이어도 반드시 bug는 아니다. 반대로 Python 객체 이름이 같아 보여도 lazy materialization이나 wrapper 내부 buffer가 별도일 수 있다.

메모리 계산도 이 구분을 반영한다. V=152,064, D=8192, BF16 dense weight는 `152064×8192×2 = 2,491,449,344 byte`, 약 2.32 GiB다. TP=4이면 rank당 dense shard는 약 594 MiB다. 진정한 storage alias라면 rank마다 이 크기의 중복을 피한다. 하지만 quantized output head가 packed 4-bit weight와 scale을 별도로 만들고 input embedding은 BF16을 유지한다면 “tied 모델”이어도 runtime memory 절감량이 단순 2.32 GiB가 아니다. 운영 문서에는 config boolean 대신 실제 resident representation과 byte를 적는다.

로드 검증 fixture는 weight 이름 개수만 세지 않는다. global token 0, shard 경계 앞뒤, 마지막 실제 token, 추가 token에서 embedding row와 head row의 checksum을 비교한다. TP rank마다 local row 변환을 적용한 뒤 비교해야 한다. padding row는 의미 equality 대상이 아니다. 양자화가 있으면 dequantized bounded slice 또는 projection 결과를 허용 오차로 비교한다. pointer equality는 alias 증거지만 numerical identity를 대신하지 않고, numerical identity는 storage sharing 증거를 대신하지 않는다.

pipeline parallel도 질문을 늘린다. 첫 stage가 input embedding을 소유하고 마지막 stage가 LM head를 소유한다면 물리적으로 같은 GPU parameter를 alias할 수 없다. tied 의미를 유지하려면 checkpoint에서 같은 값을 각 stage에 로드하거나 별도 synchronization 전략이 필요하다. “weight tying은 메모리를 절반으로 줄인다”는 설명은 single-stage 배치에만 맞을 수 있다. PP topology, rank ownership, weight replication을 같이 적어야 한다.

### 사건: rank 2의 local token 3이 global token 3으로 commit됐다

TP=4, rank당 38,016개의 padded vocab row를 가진 실제적인 모양을 생각하자. rank 2의 시작점은 76,032다. 어떤 요청의 raw local logits에서 rank 2 local index 3이 score 18.4로 전역 최고다. 올바른 global token은 76,035다. 그런데 새 distributed greedy path가 candidate pair에 local index만 넣고, driver가 그것을 이미 global이라고 해석했다. 결과로 token 3이 commit됐다.

이 사고가 까다로운 이유는 token 3도 유효하기 때문이다. out-of-range exception도, CUDA fault도, tokenizer error도 없다. 모델은 전혀 다른 짧은 token을 정상적으로 출력하고 다음 step에서는 embedding global row 3을 읽는다. 한 번의 좌표 오류가 다음 residual, KV, logits 전체를 갈라 놓는다. 사용자는 “TP=1에서는 답이 좋은데 TP=4에서 문장이 이상하다”고 보고한다. 평균 latency와 GPU utilization은 정상이고, weight checksum도 모두 맞을 수 있다.

첫 대응에서 전체 transformer layer를 비교하면 step 1까지 hidden과 local logits는 정확히 일치한다. divergence는 local candidate를 global candidate로 바꾸는 edge에서 처음 나타난다. debug record는 다음처럼 읽혀야 한다.

```text
step=0 owner_rank=2 shard_start=76032
local_winner=(score=18.4, local_id=3, valid=true)
expected_global_id=76035 committed_id=3
next_embedding_global_id=3
```

이 네 줄이 있으면 attention kernel, KV cache, RoPE, LM-head GEMM을 조사 대상에서 내릴 수 있다. 반대로 selected token만 `3`으로 기록하면 그것이 local인지 global인지 알 수 없어 로그가 증거 역할을 하지 못한다. 필드 이름에 좌표계를 넣는 것이 단순 문체 문제가 아니라 디버깅 가능성이다.

수정은 candidate를 만드는 owner에서 global offset을 더하거나, return type을 명시적인 local pair로 유지하고 commit 직전에 단 한 번 변환하는 두 방식이 있다. 어느 쪽이든 변환 횟수가 타입과 assertion으로 드러나야 한다. `0 <= global_id < org_vocab_size + allowed_added_vocab`을 검사하고, owner range로 역변환했을 때 원래 local ID가 나오는 round-trip assertion을 debug build에 둔다. 단, 범위 assertion만으로 이번 사건은 잡히지 않는다. 잘못된 3도 범위 안이므로 round-trip과 reference fixture가 필요하다.

회귀 테스트는 TP=1과 TP=4의 최종 text만 비교하지 않는다. 같은 selected hidden과 head fixture에서 local logits, local winner, global merge 후보, selected global ID, broadcast ID, 다음 embedding owner를 비교한다. 각 rank의 local index 0과 마지막 실제 index를 강제로 승자로 만드는 사례를 넣는다. 마지막 rank에서는 padding row를 가장 크게 만들되 선택되지 않아야 한다. 동일 score tie도 넣어 topology 변화에 대한 결정성을 검사한다.

### tied load 문제가 같은 증상으로 보이는 경우를 분리한다

위 사건과 달리 rank 2 local logits 자체가 reference와 다르면 global offset 수정부터 하면 안 된다. final hidden이 같고 LM-head projection부터 다르면 head weight identity를 조사한다. tied checkpoint에서 loader가 `lm_head.weight`를 없다고 건너뛰었는데 runtime alias가 형성되지 않았다면 head는 초기값일 수 있다. 반대로 checkpoint에 embedding과 head가 둘 다 있고 loader 순서가 같은 storage에 서로 다른 tensor를 두 번 써서 마지막 값이 이길 수도 있다. artifact가 정말 두 tensor의 equality를 보장하는지 먼저 확인한다.

로드 원장은 `(checkpoint_name, runtime_parameter, tp_slice, transform, loaded/skip_reason)`을 한 행으로 둔다. tied parameter라면 embedding 이름과 head 이름이 같은 runtime storage로 수렴하는지, 하나가 의도적으로 skip되는지 기록한다. quantization pack, transposition, padding도 transform에 포함한다. 단순 “all weights loaded” 카운터는 같은 parameter에 두 번 쓴 것과 한 번도 쓰지 않은 것을 상쇄할 수 있다.

문제 분류는 첫 divergence로 한다. embedding lookup부터 다르면 shared source weight 또는 tokenizer/global-ID 입력을 본다. embedding은 맞고 final hidden도 맞지만 LM-head raw local logits가 다르면 head representation과 load/tie를 본다. local logits는 맞고 global raw logits가 다르면 shard order, padding trim, gather를 본다. raw global은 맞고 processed distribution이 다르면 policy를 본다. selected global은 맞고 다음 embedding이 다르면 broadcast/commit lifecycle을 본다.

이 순서는 “모델 답이 다르다”는 막연한 문제를 다섯 개의 소유권 경계로 나눈다. 독자가 실제 코드에서 어디부터 들어가야 하는지도 알려 준다. model class constructor에서 head 생성과 tie 분기를 확인하고, parameter layer의 shard indices와 tie method를 확인하고, loader의 skip/rename mapping을 확인하고, logits processor의 gather/trim을 확인하고, sampler return과 next input까지 닫는다.

### 친절한 운영 체크리스트: 무엇을 켰을 때 무엇이 바뀌는가

TP 크기를 1에서 4로 바꾸면 head weight는 vocabulary 축으로 나뉘고 raw logits는 local tensor가 된다. global result를 얻기 위한 collective 또는 distributed selection이 새로 필요하다. 메모리는 head shard 관점에서 줄지만 collective와 좌표 변환이 생긴다. 확인할 값은 shard start/end, padded range, local tensor shape, global token 변환이다.

logprobs 반환을 켜면 선택 token 하나만이 아니라 normalization과 상위 후보 정보가 필요해진다. prompt logprobs까지 켜면 projection row 수도 늘어난다. 확인할 값은 selected row indices, global logsumexp 방법, 반환 k, serialization byte다. latency 증가는 sampler 하나가 아니라 LM-head GEMM과 collective, output copy에서 올 수 있다.

adapter extra vocabulary를 켜면 원본 token 범위, padding, 추가 token 범위가 공존한다. 확인할 값은 tokenizer added-token ID, head의 added range, 각 rank local mapping, trim이 적용되는 시점이다. 원래 vocabulary 크기로 무조건 자르면 adapter token이 사라지고, padded 크기까지 허용하면 의미 없는 row가 후보가 된다.

distributed greedy 또는 gather 생략을 켜면 full logits materialization은 줄 수 있다. 대신 지원 policy 범위가 좁아지거나 별도의 exact reduction이 필요하다. 확인할 값은 fallback 조건, candidate payload, tie-break, padding mask, global offset이다. 같은 seed보다 먼저 eligible set과 global winner가 reference와 같은지 본다.

weight tying을 켜면 모델 구성과 loader의 parameter graph가 바뀐다. dense single-stage에서는 memory 중복을 줄일 수 있지만 quantization과 PP에서는 별도 representation이 필요할 수 있다. 확인할 값은 runtime object/storage identity, shard numerical equality, loader skip 이유, resident byte다. config boolean 하나로 성공을 판정하지 않는다.

quantization을 켜면 head projection weight의 저장 형식, scale, kernel과 tie 구현이 달라질 수 있다. embedding lookup과 LM-head GEMM이 같은 packed representation을 쓸 수 있는지 구현별로 다르다. 확인할 값은 quant method가 제공하는 `tie_weights`, head output dtype, dequantized slice, padding row 초기화다. “tied인데 메모리가 왜 두 벌인가”라는 질문은 representation 요구를 확인한 뒤 판단한다.

### canonical 네 행에 남겨야 할 최종 증거

한 장짜리 incident dossier에는 artifact digest와 config, TP·PP topology, original/padded/added vocab 크기, rank별 shard 범위가 먼저 온다. 다음에는 한 request-position의 selected hidden digest, local raw score fixture, mask 뒤 local 후보, global merge 후보, selected global ID, logprob, broadcast와 다음 embedding ID가 온다. 끝에는 first divergence, 수정한 함수와 불변식, 회귀 fixture, 성능 전후가 온다.

source 링크는 장식이 아니다. shard 범위를 생성하는 함수, head가 weight를 묶는 함수, model loader가 이름을 건너뛰거나 복사하는 분기, logits가 gather되고 trim되는 함수, token이 commit되는 caller를 각각 가리킨다. line이 이동할 수 있으므로 commit을 고정하고, 문서에는 그 코드에서 확인한 사실과 실제 배포에서 관측한 사실을 구분한다. 이 책의 수치 fixture는 전자를 이해시키지만 특정 GPU 배포의 실행 branch를 증명하지는 않는다.

마지막으로 수정 뒤 두 방향을 모두 걷는다. output 방향에서는 local score가 올바른 global token과 exact policy 결과가 되는지 본다. input 방향에서는 그 global token이 다음 embedding에서 정확한 owner와 local row로 되돌아가는지 본다. 이 round trip이 닫혀야 “한 token을 생성했다”고 말할 수 있다.

### 30분 source probe: 문서의 설명을 자기 배포 코드에 다시 대입한다

첫 5분에는 실행을 시도하지 않고 effective artifact와 source revision을 적는다. model config의 `vocab_size`, tokenizer의 실제 ID 범위, added token 수, `tie_word_embeddings`, quantization 방식, TP와 PP 크기를 한 표에 놓는다. 값이 다르면 어느 것이 오류라고 바로 판단하지 않는다. padding과 adapter 때문에 합법적으로 다를 수 있다. 대신 각 값의 producer와 consumer를 적어 “tokenizer가 만들 수 있는 ID”와 “head가 계산하는 물리 row”를 구분한다.

다음 5분에는 model constructor만 읽는다. input embedding을 만드는 줄, output head를 만드는 줄, tie 분기, `compute_logits` 호출을 표시한다. vLLM Llama에서는 head 생성 뒤 tie method가 호출되고, `compute_logits`는 logits processor에 head를 넘긴다. SGLang Llama에서는 tied branch가 embedding module을 head로 사용하고 forward가 logits processor로 이어진다. 이 단계의 질문은 numerical 값이 아니라 객체 graph다. 어느 pipeline rank가 어떤 module을 소유하며, head 직전 hidden의 shape와 dtype이 무엇인지 적는다.

세 번째 5분에는 parameter layer로 내려간다. `org_vocab_size`, padded size, rank별 `shard_indices`, partition row 수를 손으로 다시 계산한다. 실제 TP rank 하나를 골라 시작과 끝을 적고 global boundary token 세 개를 local row로 번역한다. 예를 들어 시작 바로 앞 token은 이 rank에서 invalid이고, 시작 token은 local 0이며, 실제 끝 이후의 padding은 selectable하지 않아야 한다. 코드의 `min`과 mask가 이 예상과 같은지 확인한다.

네 번째 5분에는 logits processor를 읽는다. local head output 이후 all-gather인지 distributed candidate인지, original size trim이 gather 전인지 후인지, output dtype이 무엇인지 표시한다. `skip_all_gather` 같은 boolean은 이름만 복사하지 않고 누가 설정하고 어떤 조건과 결합되는지 역추적한다. false에서 true로 바뀌면 사라지는 tensor, 새로 생기는 candidate pair, sampler가 요구하는 입력 계약을 적는다. 지원하지 않는 sampling policy에서 fallback하거나 거부하는지도 찾는다.

다섯 번째 5분에는 loader를 읽는다. checkpoint의 embedding/head 이름이 runtime parameter에 어떻게 매핑되는지, tied일 때 어떤 이름을 skip하는지, padding과 TP slicing이 어느 순서인지 적는다. `loaded_params` 집합만 보지 말고 실제 runtime storage별 write 횟수를 생각한다. 같은 storage에 embedding과 head가 각각 쓰이는 경우 마지막 write가 무엇인지, head 이름이 없을 때 alias가 이미 완성됐는지 확인한다.

마지막 5분에는 하나의 예상 incident를 쓴다. “TP=4 rank 2 local 3이 global 3으로 commit될 수 있다”처럼 symptom, competing hypotheses, 첫 checkpoint를 구체적으로 쓴다. 첫 checkpoint는 final hidden과 local logits다. 이것이 맞으면 weight·kernel 가설을 내리고 global merge로 이동한다. 이것이 다르면 tie/load와 projection으로 이동한다. probe의 산출물은 방대한 함수 목록이 아니라 다음에 열 정확한 함수와 반증 가능한 예상이다.

이 절차를 옵션 변경 리뷰에도 재사용할 수 있다. 리뷰어는 “통신을 줄인다”는 설명 대신 어떤 global 정보가 더 이상 materialize되지 않는지, 요청 기능에 필요한 정보가 어떤 reduction으로 보존되는지 묻는다. “메모리를 줄인다”면 tied storage와 quantized representation을 구분하고 rank·stage별 resident byte를 묻는다. “결정성을 보장한다”면 score tie와 collective order, global ID tie-break를 묻는다.

### 잘못된 설명을 걸러 내는 반례 다섯 개

“TP를 켜도 logits는 같은 `[B,V]`다”라는 설명에는 head 직후 local tensor라는 반례가 있다. global shape는 gather 뒤에만 성립할 수 있다. “padding row는 0이므로 선택되지 않는다”에는 모든 valid logits가 음수거나 padding buffer가 0보다 큰 경우가 반례다. 명시적 validity mask가 필요하다.

“local top-k만 모으면 확률도 정확하다”에는 normalization이 전체 vocabulary를 요구한다는 반례가 있다. 후보 순위와 확률 값의 exactness를 나눈다. “tied weight는 언제나 같은 pointer다”에는 pipeline stage 분리와 서로 다른 quantized representation이 반례다. 의미 공유와 storage 공유를 나눈다.

마지막으로 “잘못된 token ID는 범위 검사로 잡힌다”에는 rank-local ID가 우연히 정상 global 범위에 들어가는 이번 사건이 반례다. 범위 검사와 더불어 owner/local/global round trip, TP=1 reference, boundary fixture가 필요하다. 좋은 설명은 정상 흐름만 매끄럽게 말하는 것이 아니라, 독자가 그 설명을 과신하지 않도록 깨지는 조건도 함께 준다.

배포 전 terminal은 네 문장으로 닫는다. 모든 rank의 padding 후보가 제외됐고 local-to-global round trip이 boundary fixture에서 성립한다. exact logprob 요청은 global normalization을 통과하며 top-k 후보의 ID와 값이 reference에 맞는다. tied artifact는 rank·stage별 loader 원장과 numerical slice가 일치하고 의도한 storage byte를 보인다. 취소와 다음-step handoff까지 같은 request incarnation의 global token이 유지된다. 하나라도 증명되지 않으면 새 distributed path를 기본값으로 승격하지 않고, 검증된 gather 경로로 rollback한다.

rollback 뒤에도 실패 fixture와 trace는 지우지 않는다. 최적화 path가 다시 제안될 때 같은 rank boundary, padding winner, tied loader 조합을 재사용한다. 성공 조건은 “이번 요청이 자연스럽다”가 아니라 좌표·확률·storage·commit 네 계약이 독립적으로 닫히는 것이다. 이 기준이 있어야 성능 개선과 정답 보존을 같은 리뷰에서 다룰 수 있다.

마지막 검토자는 작은 vocabulary fixture와 실제 크기 byte 원장을 함께 본다. 작은 예제는 정답을 손으로 증명하고, 실제 모양은 최적화의 경제성을 보여 준다. 둘 중 하나만 있으면 설명은 각각 장난감이나 측정 없는 추측으로 기운다. source anchor는 가능한 실행 경로를 고정하고 runtime trace는 배포가 실제로 그 경로를 택했는지 확인한다. 세 증거가 일치할 때만 incident를 닫는다.

이 장의 긴 여정을 한 문장으로 압축하면 이렇다. LM head는 hidden vector를 큰 숫자 배열로 바꾸는 마지막 linear가 아니라, sharded parameter와 token 의미 공간, request policy, 분산 합의, 다음 iteration을 연결하는 commit boundary다. 이 경계를 읽는 독자는 답이 이상할 때 무작정 attention부터 뒤지지 않고, 값이 처음 달라진 좌표와 owner를 찾아 올바른 함수로 들어갈 수 있다.
