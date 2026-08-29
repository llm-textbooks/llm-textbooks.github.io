# 2장. 역전파에서 optimizer step까지

1장의 logit–target 오차가 어떤 parameter gradient로 나뉘는지가 이 장의 출발점이다. 11장은 여기서 만든 gradient를 AdamW state와 update로 바꾸고, 14장은 같은 backward가 저정밀·fused CUDA 경로에서 어디서 다르게 누적되는지 보여 준다.

이 장을 읽는 동안 붙들 상태 사슬은 하나다.

> `scalar loss` → `autograd graph` → `saved tensor와 version` → `local VJP` → `leaf .grad 누적` → `DDP reduction` → `unscale·finite 합의` → `global clip` → `optimizer commit`

화살표마다 값의 **단위와 소유자**가 바뀐다. loss는 token objective의 scalar이고, graph node가 운반하는 것은 upstream adjoint이며, leaf `.grad`는 여러 경로와 microbatch의 기여를 담는 mutable buffer다. DDP를 지나면 rank-local 기여가 collective 계약에 따라 합쳐지고, unscale 뒤에야 optimizer가 읽을 실제 단위가 된다. clip은 그 vector를 제약 집합으로 옮기며, optimizer step은 gradient뿐 아니라 parameter와 moment와 step counter를 함께 바꾸는 commit이다. 최종 loss가 같다는 사실만으로 이 중간 상태들이 같다고 추론하지 않는다.

| 경계 | 들어오는 상태 | 나가는 상태 | 이 경계에서 처음 드러나는 오류 |
|---|---|---|---|
| scalar loss | per-token contribution, mask, denominator | 0차원 objective | label shift·분모 오류 |
| autograd graph 생성 | tensor operation과 alias | 역방향 edge와 saved state | `detach`, grad mode, 잘못된 custom op |
| local VJP | upstream adjoint, local operand | operand별 adjoint contribution | transpose·broadcast reduction·backward 식 오류 |
| leaf accumulation | 여러 contribution | `.grad=None|tensor` | 덮어쓰기·중복 backward·zero 시점 오류 |
| collective | rank-local gradient | reduced gradient | 마지막 `no_sync`, 분모·world-size 오류 |
| unscale·finite | scaled gradient, scale | 실제 단위 gradient 또는 skip 결정 | rank별 scale·overflow 불일치 |
| clip | unscaled global vector | 제약된 gradient | local/shard norm·순서 오류 |
| optimizer commit | parameter, gradient, moment, LR | 새 parameter와 optimizer state | 부분 step·counter·group mapping 오류 |

이 표는 디버깅 순서이기도 하다. parameter delta가 다르면 optimizer부터 추측하지 않고, 동일한 `GoldenBatchID`에서 표의 위쪽 경계부터 checksum과 선택 원소를 비교해 **최초 불일치**를 찾는다. 뒤에서 생긴 차이는 대부분 앞선 차이의 결과다.

## 2.1 scalar loss에서 VJP와 leaf gradient까지

역전파는 gradient tensor가 갑자기 생기는 명령이 아니다. scalar loss의 seed가 operator별 VJP를 거꾸로 통과하면서 saved tensor를 읽고, 여러 경로의 기여를 leaf의 `.grad`에 합산하는 과정이다. 먼저 이 수학·메모리 계약을 닫아야 뒤의 분산 오류를 정확히 분리할 수 있다.

### 2.1.1 계산 그래프는 값을 재사용한다

**scalar autograd와 chain rule**

`y=x*x+x`에서 `x`는 세 경로로 결과에 영향을 준다. 따라서 `dy/dx=2x+1`이며 각 경로의 gradient는 대입이 아니라 누적되어야 한다. micrograd의 작은 `Value` 그래프는 production autograd의 축소판이다. 위상 역순으로 node를 방문하고, leaf의 `.grad`에 기여를 더한다. 같은 parameter를 embedding과 LM head가 공유할 때도 이 원리는 바뀌지 않는다.

**계산 그래프의 방향.** forward edge는 operand에서 result로, backward는 result의 adjoint에서 operand adjoint로 흐른다. scalar node `v`의 adjoint를 `v̄=∂L/∂v`라 쓰면 연산 `z=f(x,y)`는 `x̄+=z̄∂f/∂x`, `ȳ+=z̄∂f/∂y`를 수행한다. `+=`가 핵심이다. 한 leaf가 여러 branch에서 쓰이면 모든 경로의 기여를 합쳐야 total derivative가 된다.

**위상 정렬.** output에서 DFS로 parent를 방문해 topo list를 만들고 역순으로 local backward를 호출한다. cycle이 없는 계산 그래프라는 가정이 있다. Python control flow가 recurrent 계산을 여러 step 펼쳐도 graph 자체는 펼쳐진 DAG다. in-place mutation이 이전 값의 의미를 지우면 autograd version counter가 오류를 내는 이유다.

**작은 수치 검산.** `x=3`, `y=x²+x=12`, analytic gradient 7이다. 중앙차분 ε=1e-4는 `(f(3+ε)-f(3-ε))/(2ε)`로 약 7을 준다. forward 값만 맞고 gradient가 4처럼 나오면 `x*x`의 두 경로 중 하나를 대입으로 덮었을 가능성이 있다.

**반례 1—gradient 0은 bug가 아닐 수 있다.** ReLU 음수 영역, saturated gate, ignored loss, frozen parameter는 합법적으로 0을 만든다. expected path와 local derivative를 확인하지 않고 zero gradient를 corruption으로 단정하지 않는다.

**반례 2—finite difference도 항상 oracle은 아니다.** ε가 너무 작으면 rounding, 너무 크면 고차항 오차가 생긴다. 비매끄러운 max/top-k/ReLU 경계에서는 좌우 derivative가 다를 수 있다. 여러 ε와 방향 derivative를 쓴다.

**micrograd에서 production으로 넘기는 불변식.** leaf reuse는 누적되고, backward 시작 seed는 scalar loss에서 1이며, backward 전 gradient를 초기화하지 않으면 이전 graph 기여가 남는다. production의 accumulation은 이 마지막 성질을 의도적으로 이용하지만 update 경계에서 zeroing해야 한다.

**Jacobian을 만들지 않고 기울기를 운반한다**

벡터 함수 `y=f(x)`의 Jacobian을 `J_f=∂y/∂x`라 하자. reverse mode가 각 node에서 계산하는 것은 `J_f` 자체가 아니라, 뒤에서 온 covector `v̄=∂L/∂y`와의 곱 `x̄=v̄J_f`다. PyTorch의 `Tensor.backward()`는 scalar output이면 시작 covector 1을 암묵적으로 넣는다. output이 scalar가 아니면 어떤 선형 결합을 미분할지 나타내는 `gradient`를 제공해야 한다. `torch.autograd.grad(outputs, inputs, grad_outputs=...)`도 같은 VJP를 반환하지만, 결과를 어느 leaf `.grad`에 누적할지와 graph를 보존할지는 호출 계약에서 따로 읽어야 한다.

기하적으로 gradient는 “loss가 커지는 화살표”이기 전에 좌표계와 내적을 정했을 때 미분 covector를 vector로 표현한 것이다. 그래서 parameterization을 바꾸면 같은 함수 변화라도 Euclidean gradient와 norm이 달라진다. Adam이나 Muon이 gradient를 변환하는 이유, global norm clipping이 function-space trust region과 같지 않은 이유가 여기서 시작한다. 이 장에서는 우선 각 구현이 같은 parameter 좌표에서 같은 VJP를 만드는지를 확인하고, 11장에서 preconditioner가 그 방향과 크기를 어떻게 바꾸는지 본다.

작은 diamond graph `a=x²`, `b=3x`, `L=a+b`를 그려 보면 `x`에는 두 edge가 도착한다. local VJP는 각각 `2x`와 3이고 leaf accumulator는 둘을 더해 `2x+3`을 만든다. `detach`를 `b`에 넣으면 forward 값 `b=3x`는 그대로지만 그 edge의 VJP가 사라져 gradient는 `2x`가 된다. 이 사례는 forward parity가 backward connectivity를 증명하지 않는 가장 작은 반례다.

**값·storage·미분 이력을 서로 구분한다**

tensor를 조사할 때 세 질문을 분리한다. 숫자 값이 같은가, 같은 storage를 alias하는가, 같은 autograd history에 연결되어 있는가. `clone()`은 새 storage를 만들면서 미분 edge를 유지할 수 있고, `detach()`는 storage를 공유할 수 있지만 그 결과를 현재 graph의 상수처럼 취급한다. `detach().clone()`은 둘 다 끊는다. 단순히 checksum과 data pointer 하나만 비교해서 graph 연결성을 판단할 수 없다.

view에는 base tensor의 storage mutation이 보인다. backward가 과거 값을 필요로 하는데 그 storage가 in-place로 바뀌면 saved tensor의 version counter와 현재 version이 달라져 autograd가 오류를 낼 수 있다. 이 오류는 방해물이 아니라 “현재 값으로 과거 함수의 derivative를 계산할 수 없다”는 안전장치다. 다만 `.data`, graph 밖 custom extension, 잘못 선언한 alias/mutation contract는 이 보호를 우회할 수 있다. 따라서 “오류가 나지 않았다”를 gradient correctness의 증거로 쓰지 않고 out-of-place reference와 방향 미분을 함께 둔다.

saved tensor는 단순 cache가 아니다. local backward를 정의하는 과거 입력·출력 또는 충분통계이며, graph node가 해제될 때 그 lifetime도 끝난다. `retain_graph=True`는 gradient를 초기화하는 옵션이 아니라 이 backward-local 상태를 남기는 옵션이다. 반면 activation checkpoint는 일부 saved state를 애초에 보존하지 않고 backward 때 forward 구간을 재실행한다. 전자는 lifetime 연장이고 후자는 storage를 compute로 교환하는 것이므로 서로 반대에 가까운 조작이다.

### 2.1.2 tensor backward가 숨기는 것

**saved tensor와 fused backward**

행렬곱 `Y=XWᵀ`는 upstream gradient `G`에서 `dX=GW`, `dW=GᵀX`를 만든다. framework는 backward에 필요한 입력이나 통계를 saved tensor로 보관한다. fused attention과 fused cross entropy는 중간 행렬을 저장하지 않고 필요한 통계를 재계산할 수 있다. 같은 수학이라도 메모리 lifetime과 부동소수점 연산 순서는 달라진다.

golden hook은 각 module의 activation checksum, gradient checksum, dtype, shape만 저장한다. 전체 tensor를 무조건 덤프하면 메모리와 동기화 비용이 진단 대상을 바꾼다. 첫 비정상 layer를 찾은 뒤에만 좁은 slice를 보존한다.

**행렬곱을 index로 검산한다.** `Y_bto=Σ_i X_bti W_oi`에서 `dW_oi=Σ_bt dY_bto X_bti`, `dX_bti=Σ_o dY_bto W_oi`다. batch와 token 축이 reduction된다. gradient shape가 parameter shape와 같다는 확인만으로 reduction 축 오류를 잡을 수 없다. batch 하나·token 하나만 nonzero인 fixture로 정확한 원소를 예측한다.

**Broadcast backward.** bias `[C]`를 activation `[B,T,C]`에 더하면 bias gradient는 `[B,T]` 축 합이다. LayerNorm scale도 broadcast되지만 local normalized activation이 곱해진다. GQA의 repeated KV처럼 forward broadcast는 backward reduction을 동반한다. view/expand/copy의 storage 차이와 수학적 합산을 분리한다.

**Saved tensor가 필요한 이유.** GELU backward는 forward input, matmul backward는 상대 operand, softmax backward는 probability 또는 재계산 가능한 logsumexp를 필요로 한다. autograd Function의 `save_for_backward`는 tensor lifetime을 backward까지 늘린다. mutation version이 달라지면 잘못된 과거 값으로 gradient를 계산하지 않도록 error가 난다.

**Activation checkpointing.** 구간의 input만 저장하고 forward를 backward 때 재실행해 activation memory를 줄인다. 대가는 추가 compute와 RNG/mutation 재현 요구다. dropout mask가 recompute 때 달라지면 다른 함수의 gradient다. global cache나 counter를 forward에서 갱신하는 module은 checkpoint-safe한지 검사한다.

**Fused CE.** logits `[N,V]` 전체를 저장하는 대신 hidden과 vocab weight에서 logsumexp·target logit·gradient를 tile로 계산할 수 있다. loss reduction denominator와 tied weight gradient가 reference와 같아야 한다. vocab chunk별 max/LSE를 online combine하며 split mean을 평균하지 않는다.

**Fused attention backward.** 8장의 식대로 Q/K/V와 row LSE를 사용해 score/probability tile을 재계산한다. forward output parity와 dQ/dK/dV parity를 따로 test한다. deterministic 옵션은 reduction 순서와 atomic 사용을 바꿀 수 있다.

**Hook 설계.** module forward hook에는 `RunID,node_path,call_index`, shape/stride/dtype, finite count, RMS, checksum을 남긴다. tensor `register_hook`에는 backward gradient 통계를 남긴다. parameter hook은 alias 때문에 중복 등록되지 않게 canonical storage group에 건다. hook에서 `.item()`과 CPU copy는 sync를 만들므로 correctness run에만 쓴다.

```python
def grad_probe(name):
    def hook(g):
        assert torch.isfinite(g).all(), name
        ledger(name, g.shape, g.dtype, g.norm(), checksum(g))
        return g
    return hook

hidden.register_hook(grad_probe("block.0.hidden"))
```

이 코드는 관찰만 해야 한다. hook이 gradient를 반환하지 않거나 수정하면 graph 동작을 바꾼다. sparse gradient와 distributed tensor는 `.norm()` 자체가 collective를 요구할 수 있으므로 ownership에 맞춘다.

**Upstream test 해석.** PyTorch autograd gradcheck는 double precision finite difference와 analytic gradient를 비교하지만 nondeterminism·sparse·undefined gradient에 별도 조건이 있다. fused kernel test는 지원 dtype/hardware와 tolerance 안의 parity만 증명한다. nanoGPT snapshot에는 layer별 backward unit test가 없으므로 10장의 golden probe가 stack-local fixture다.

**반례 3—forward parity와 backward mismatch.** detached copy를 forward에 사용하면 값은 같아도 원 parameter로 gradient가 흐르지 않는다. tied storage 해제, KV repeat detach, custom Function backward 누락이 이런 실패를 만든다.

**반례 4—gradient norm이 같아도 방향이 다르다.** 두 vector가 같은 norm으로 직교할 수 있다. max-abs, relative error, cosine과 parameter delta를 함께 본다.

### 2.1.3 update는 순서가 있는 상태기계다

**accumulation·AMP·unscale·clip·step**

`K`개 microbatch를 큰 batch 하나처럼 만들려면 각 loss를 `K`로 나누거나 loss sum과 전역 분모를 끝에서 한 번 적용한다. FP16에서는 scaled loss로 backward한 뒤 gradient를 unscale하고, 그 다음 norm clipping을 하고, overflow가 없을 때 optimizer를 step한다. clipping 뒤 unscale하면 threshold의 단위가 달라진다.

nanoGPT의 고정 revision은 마지막 accumulation microstep에서만 DDP gradient sync를 켠다. 이어 `scaler.unscale_`, `clip_grad_norm_`, `scaler.step`, `scaler.update`, `zero_grad(set_to_none=True)`를 차례로 호출한다. 이는 옵션 목록이 아니라 상태 변경 순서다. 하나를 옮기면 다른 함수를 학습한다.

**호출 흐름과 상태.** nanoGPT `train.py:290-305`는 K microstep에서 forward와 scaled backward를 수행한다. `293-298`은 마지막 microstep에만 DDP sync를 켠다. `306-314`는 unscale, clip, optimizer/scaler step, zeroing이다. optimizer는 `train.py:198-202`에서 만들어지고 resume이면 moment state를 load한다. 이 line range가 보여 주지 않는 sampler/RNG/scaler checkpoint 결손은 3장에서 다룬다.

| 순서 | 읽는 상태 | 쓰는 상태 | 순서가 틀릴 때 |
|---|---|---|---|
| forward/autocast | parameter, batch | loss, saved tensors | dtype·objective 차이 |
| scale | loss, scale | scaled loss | underflow 방지 실패 |
| backward | graph | scaled gradient 누적 | overflow·누락 |
| DDP reduce | local gradients | global averaged/summed gradient | hang·K배 통신 |
| unscale | scale, gradients | true-unit gradients | clip 단위 오류 |
| clip | global norm | rescaled gradients | threshold 무효 |
| optimizer step | params, grads, moments, LR | params, moments | partial/overflow step |
| scaler update | found-inf, tracker | next scale | resume drift |
| zero | gradients | None/zero | sample 간 누적 |

**Loss scaling의 식.** scale `s`로 `L'=sL`을 backward하면 `g'=sg`다. FP16 표현에서 작은 g가 underflow하지 않게 한다. step 전 `g'/s`로 돌아온다. 하나라도 inf/nan이면 optimizer step을 건너뛰고 scale을 낮춘다. step이 skip됐는데 scheduler가 전진하면 parameter update와 LR time이 어긋난다.

**Global norm clipping.** 모든 parameter gradient를 이어 붙인 vector g의 norm이 c보다 크면 `g←g·c/(||g||+ε)`다. parameter별 clipping과 global clipping은 다르다. DDP reduction 전 local norm을 clip하면 rank마다 다른 scale이 적용돼 global gradient를 clip한 것과 같지 않다. FSDP shard는 global squared norm을 collective로 합쳐야 한다.

**Accumulation 수식.** microbatch loss sum `S_k`, valid count `N_k`에서 원하는 gradient는 `∇ΣS_k/ΣN_k`다. N이 같을 때만 각 mean을 K로 나눈 것과 같다. DDP rank까지 `S_rk,N_rk`를 합친다. dropout RNG 때문에 같은 sample을 한 큰 batch와 microbatch로 나눴을 때 exact parity가 아닐 수 있으므로 dropout off reference를 먼저 쓴다.

**Parameter group과 clipping.** frozen parameter, sparse embedding, matrix optimizer group이 섞일 수 있다. global norm에 어떤 group을 포함하는지 적는다. unscale은 해당 optimizer가 소유한 gradient에 적용된다. optimizer 두 개를 쓰면 각각 unscale/overflow/step의 atomicity를 설계한다.

**반례 5—loss를 K로 나눴는데 gradient가 작다.** framework나 Trainer가 이미 accumulation scale을 적용했을 수 있다. 중복 division은 K² 효과를 낸다. loss 함수, training_step, accelerator wrapper의 소유자를 추적한다.

**반례 6—clip norm log가 정상인데 실제 gradient는 overflow다.** scaled gradient를 unscale 전에 norm으로 읽거나 non-finite parameter를 norm 계산에서 빠뜨렸을 수 있다. found-inf와 parameter별 finite count를 함께 본다.

**실패 주입 2-A—clip-before-unscale.** 같은 batch에서 올바른 순서와 바꾼 순서를 비교한다. scale이 큰 만큼 거의 모든 gradient가 clip되는지 parameter delta로 확인한다.

**실패 주입 2-B—zero 누락.** update 뒤 `zero_grad`를 한 번 건너뛰어 다음 batch gradient에 이전 것이 더해지게 한다. loss는 정상이어도 update가 달라진다. gradient ledger의 step owner와 accumulation window ID로 잡는다.

**실패 주입 2-C—마지막 sync 해제.** DDP 마지막 microstep도 no-sync로 두어 rank parameter가 갈라지게 한다. step 뒤 parameter checksum을 rank별 비교한다. rank 0 metric만 보면 놓칠 수 있다.

**실험 2-D—큰 batch parity.** 동일 sample과 전역 분모, dropout 0에서 combined batch와 K microbatch의 loss sum/count/gradient/delta를 비교한다. optimizer momentum이 없는 첫 step부터 시작한 뒤 Adam state까지 확장한다.

### 2.1.4 첫 비정상을 이분 탐색한다

**NaN과 plateau 체크리스트**

먼저 `None`, 정확한 0, finite nonzero, non-finite를 네 상태로 나눈다. `grad is None`은 그 update에서 leaf accumulator에 기여가 도착하지 않았다는 뜻이다. frozen parameter라면 정상이고, trainable parameter라면 detach·unused branch·zeroing 시점·optimizer ownership을 조사한다. 값이 0인 tensor는 graph에는 참여했지만 local derivative나 상쇄 결과가 0일 수 있다. 두 상태를 `grad_norm=0` 하나로 합치면 graph 단절과 합법적인 zero gradient를 구별할 수 없다.

| 관측된 최초 차이 | 아직 의심하지 않을 것 | 바로 고정할 fixture | 다음으로 볼 상태 |
|---|---|---|---|
| loss numerator·count | optimizer·DDP | token 2~4개의 손계산 CE | target, mask, reduction |
| forward 중간값 | backward kernel | 동일 parameter·입력 FP64 eager | dtype, alias, RNG, selected backend |
| forward는 같고 VJP가 다름 | LR·scheduler | 한 방향의 central difference | saved tensor, transpose, detach, mutation |
| local VJP는 같고 leaf grad가 다름 | optimizer formula | shared leaf를 두 번 쓰는 diamond graph | accumulation, hook, 두 번째 backward |
| rank-local은 같고 reduced grad가 다름 | activation | two-rank numerator/count 표 | collective, `no_sync`, scale factor |
| reduced grad는 같고 unscale 뒤 다름 | data | 고정 scale의 overflow/no-overflow 쌍 | scaler state, found-inf 합의 |
| raw grad는 같고 clipped grad가 다름 | model graph | 선택 원소와 global squared norm | shard ownership, 중복 parameter |
| clipped grad는 같고 delta가 다름 | loss | FP64 optimizer 첫 두 step | group, moment, LR, step counter |

이 표에서 “같다”는 finite와 norm이 같다는 뜻이 아니다. shape·dtype·stride 같은 계약을 먼저 보고, 선택 원소·max absolute/relative error·방향 cosine·checksum을 목적에 맞게 조합한다. 특히 norm 하나는 직교한 두 vector를 구별하지 못한다.

**손계산에서 분산 update까지 올라가는 oracle 사다리**

첫 fixture는 kernel이 아니라 종이 위 식이어야 한다. 다음 예는 branch 누적, `detach`, 반복 backward와 zeroing을 한꺼번에 분리한다.

```python
import torch

torch.set_default_dtype(torch.float64)
x = torch.tensor(2.0, requires_grad=True)
loss = x * x + 3 * x
loss.backward()
assert x.grad.item() == 7.0              # 2x + 3

x.grad = None
detached_loss = x * x + (3 * x).detach()
detached_loss.backward()
assert x.grad.item() == 4.0              # 값은 같아도 한 edge가 끊김
```

여기서 `x.grad=None`은 buffer를 비운 것이지 graph를 끊은 것이 아니다. 첫 loss에 다시 `backward()`하면 기본적으로 saved state가 이미 해제되어 오류가 난다. 반대로 새 forward를 만들고 zero 없이 backward하면 leaf `.grad`에는 이전 값과 새 기여가 더해진다. graph lifetime과 gradient-buffer lifetime은 서로 다른 축이다.

두 번째 fixture는 VJP를 직접 손계산한다. `X=[[1,2]]`, `W=[[3,4],[5,6]]`, `Y=XWᵀ`, upstream `G=[[7,11]]`이면 `dX=GW=[[76,94]]`, `dW=GᵀX=[[7,14],[11,22]]`다. random tensor의 평균 오차보다 이 작은 비대칭 정수가 transpose와 reduction 축 오류를 더 선명하게 잡는다. bias를 추가하면 `db=[7,11]`이며 batch 행을 하나 더 넣었을 때 그 축이 합산되는지 확인한다.

세 번째는 central directional difference다. parameter vector `θ`와 정규화한 방향 `u`에서

`D_u L ≈ [L(θ+εu)-L(θ-εu)]/(2ε)`

를 autograd inner product `⟨∇L,u⟩`와 비교한다. 모든 parameter coordinate를 흔드는 full finite difference보다 싸고, 실제 update 방향을 선택해 검사할 수 있다. FP64에서 여러 `ε`를 로그 간격으로 움직여 truncation error가 우세한 구간과 rounding error가 우세한 구간 사이의 안정 영역을 찾는다. dropout·data augmentation을 끄고 parameter perturbation 사이에 buffer나 RNG가 변하지 않게 한다. ReLU kink, top-k routing 경계처럼 derivative가 정의되지 않거나 불연속인 점은 fixture에서 피하고 별도 subgradient 계약으로 다룬다.

그 뒤에만 FP32 eager, checkpoint/recompute, AMP, fused backward, accumulation, DDP 순으로 한 층씩 올린다. 각 승격 단계는 직전 단계와 변경점 하나만 가져야 한다. 예컨대 eager와 checkpoint를 비교할 때 동시에 dropout seed와 dtype을 바꾸면 gradient mismatch가 recompute 때문인지 rounding 때문인지 판정할 수 없다. 최종 분산 fixture도 single-process concatenated objective를 oracle로 유지한다.

**graph retention을 memory가 아니라 state로 진단한다**

unexpected graph retention은 보통 GPU memory 그래프가 계단처럼 오르는 것으로 보이지만 최초 불일치는 Python container나 callback이 graph-root tensor를 소유하기 시작한 순간이다. `losses.append(loss)`와 `running += loss`는 detach하지 않으면 이전 graph를 붙들 수 있다. logging에는 `loss.detach()` 또는 필요한 scalar 복사만 넘기며, 진단 시에는 container element의 `grad_fn`, saved-tensor pack/unpack count, update 뒤 live tensor owner를 기록한다.

activation checkpoint recompute 때문에 forward hook 호출 수가 늘어나는 것은 누수와 다르다. 원 forward와 recompute를 `phase`로 구분하고, update가 끝났을 때 이전 UpdateID의 graph-root가 남는지 본다. `retain_graph=True`가 필요한 multi-loss라면 어느 loss가 마지막 consumer인지 정하고 그 직후 graph가 해제되는 fixture를 둔다. 단지 OOM이 사라질 때까지 `empty_cache`를 호출하는 것은 live reference를 고치지 않는다.

NaN이면 loss에서 거꾸로 보지 말고 입력에서 앞으로 간다. token range, embedding finite, norm variance, QK score max, attention row sum, MLP activation, logit max, per-token CE, scaled gradient, unscaled gradient, parameter delta 순으로 최초 비정상을 찾는다. plateau면 label shift와 valid count, learning rate, `zero_grad` 시점, frozen parameter, 반복 batch ID를 먼저 확인한다.

통제 실험은 한 번에 하나만 바꾼다. AMP를 끄고 같은 `GoldenBatchID`의 loss·gradient cosine을 비교한다. accumulation `K=1`과 `K=4`는 동일 sample 집합과 분모를 써야 한다. 차이가 크면 dropout RNG, batch order, DDP reduction 또는 rounding을 분리한다.

**NaN 결정 트리.** 입력 ID와 mask가 유효한가? 아니면 data/tokenizer다. embedding부터 non-finite인가? weight/checkpoint다. norm 전은 finite인데 후가 깨지는가? variance/epsilon/dtype다. QK score에서 처음 커지는가? scale/QK norm/mask다. MLP에서 커지는가? activation/init/low precision이다. logits는 finite인데 loss가 inf인가? target range/log-softmax다. loss는 finite인데 backward가 깨지는가? fused backward/scaler다. unscale 뒤만 깨지는가? overflow detection과 scale state다.

**Plateau 결정 트리.** `GoldenBatchID`가 변하는지 확인한다. label shift와 valid count를 손검산한다. trainable parameter 수와 nonzero gradient 비율을 본다. learning rate와 scheduler owner를 확인한다. clip ratio가 계속 1보다 매우 작은지 본다. optimizer delta/weight ratio를 layer별로 본다. 작은 batch를 반복해 memorization 가능한지 시험한다. memorization도 안 되면 model/data plumbing 문제를 먼저 해결한다.

**Silent failure 지표.** loss 하나 외에 gradient RMS/zero/non-finite, clip coefficient, scaler scale·skipped steps, parameter delta RMS, update/weight ratio, batch ID repeat, valid-token count를 기록한다. rank별 min/max를 보지 않고 평균만 보면 한 rank의 corruption이 묻힌다.

**NaN 반례 7—NaN을 0으로 바꾸면 복구가 아니다.** `nan_to_num`은 최초 원인을 숨기고 잘못된 gradient를 step할 수 있다. fail-fast 후 마지막 valid CheckpointID로 돌아가 원인 실험을 한다.

**Plateau 반례 8—learning rate를 올리는 것이 항상 답이 아니다.** 모든 labels가 ignore이거나 optimizer에 parameter가 등록되지 않았다면 LR은 효과가 없다. 관측 가능한 parameter delta를 먼저 확인한다.

**Hook 기반 first-bad-tensor.** forward node 순서와 backward 역순에 monotonic call index를 부여한다. 두 run을 비교할 때 첫 non-finite 또는 첫 tolerance 위반 node만 자세히 덤프한다. module이 반복 호출되면 path만 아니라 call index가 필요하다. activation checkpoint recompute 호출은 phase를 표시한다.

**Test matrix.** scalar finite difference, matmul/broadcast analytic gradient, tied-gradient 합, manual/fused CE, manual/SDPA backward, K accumulation, AMP on/off, clip order, save/load 뒤 gradient를 층위별로 둔다. 아래 test가 실패하면 위의 복합 test 결과를 해석하지 않는다.

**분산 조사 체크리스트.** gradient collective owner를 찾는다. DDP 평균/sum semantics를 확인한다. no-sync window와 마지막 sync를 찾는다. valid count collective를 확인한다. global norm 계산 rank를 본다. overflow flag가 rank 전체에서 합의되는지 확인한다. optimizer/scaler/scheduler step이 모든 rank에서 같은 횟수인지 기록한다. step 뒤 parameter checksum을 비교한다.

**복구 조사 체크리스트.** checkpoint에 optimizer moment, scaler, scheduler, RNG, accumulation 중간 상태가 있는지 본다. 공식 save 경계가 gradient가 비워진 optimizer-step 직후인지 확인한다. overflow로 step이 skip된 직후 counter 의미를 적는다. resume 첫 batch에서 forward·gradient·delta를 기준 실행과 비교한다.

**재현 절차.** CPU FP64 toy gradient를 최하위 oracle로 만든다. FP32 eager golden batch로 module hook ledger를 만든다. fused kernel을 하나씩 켜 gradient parity를 비교한다. AMP를 켜 scaled/unscaled ledger를 남긴다. accumulation과 DDP를 추가한다. 마지막에 checkpoint round trip을 수행한다. 각 단계는 바로 앞 단계와 한 field만 다르다.

**2장의 인계물.** 3장에는 update state order, gradient ledger, overflow/skip counter를 넘긴다. 10장에는 hook schema와 first-difference algorithm을 넘긴다. 11장은 parameter별 gradient와 group manifest를 받아 optimizer geometry를 적용한다. 14장은 saved tensor/recompute와 precision failure를, 15장은 gradient/clip collective ownership을 받는다.

**종료 판정.** scalar chain rule부터 tied/fused/distributed gradient까지 같은 `GoldenBatchID`로 이어져야 한다. loss가 맞는 것, gradient가 맞는 것, optimizer delta가 맞는 것, resume 뒤 같은 update를 하는 것을 각각 검사한다. 하나의 성공으로 다른 셋을 추론하지 않는다.

**Vector-Jacobian product.** tensor output의 full Jacobian을 만들지 않고 upstream vector `v`와 `vᵀJ`를 계산하는 것이 reverse-mode다. scalar loss에서 시작하면 output dimension과 무관하게 parameter 전체 gradient를 한 번의 reverse traversal로 얻는다. 반대로 input 몇 방향의 sensitivity가 필요하면 forward-mode JVP가 유리할 수 있다. Hessian-vector product도 gradient에 다시 directional derivative를 적용해 full Hessian 없이 구한다.

**Graph retention과 memory leak.** backward 뒤 graph를 해제하는 것이 기본이다. loss tensor를 Python list에 graph째 보존하거나 `retain_graph=True`를 반복하면 activation이 계속 살아남는다. logging에는 detach된 scalar를 사용한다. 여러 loss에서 같은 graph를 backward해야 할 때만 retention 이유와 lifetime을 명시한다.

**Custom backward 계약.** forward input 수와 backward 반환 gradient 수가 맞아야 하며 gradient가 필요 없는 입력은 None을 반환한다. dtype/device/shape를 보존하고 higher-order gradient 지원 여부를 적는다. contiguous를 가정하는 CUDA kernel은 wrapper에서 stride를 검증하거나 copy를 명시한다. silent reinterpretation을 허용하지 않는다.

**In-place와 alias 반례.** residual `x += branch`가 backward에 필요한 pre-add x를 덮을 수 있다. framework version counter가 잡지 못하는 custom kernel에서는 잘못된 gradient가 finite하게 나온다. out-of-place reference와 gradcheck를 둔다. view의 base가 mutation되는 경우도 포함한다.

**Saved-tensor hook과 offload.** saved activation을 CPU로 옮기거나 압축해 GPU memory를 줄일 수 있지만 transfer synchronization, dtype 손실, lifetime이 추가된다. pack hook과 unpack hook은 같은 tensor identity와 shape를 복구해야 한다. 성능 이득과 gradient parity를 별도 측정한다.

**Gradient checkpoint 경계 선택.** block 전체를 checkpoint하면 attention·MLP를 재계산하고 residual input만 보존한다. 너무 작은 경계는 framework overhead가 늘고, 너무 큰 경계는 recompute 비용과 RNG 관리가 커진다. peak memory, step time, gradient parity를 같은 batch에서 비교한다.

**AMP autocast의 실제 의미.** 모든 연산을 낮은 dtype으로 강제하는 옵션이 아니다. operation policy에 따라 matmul은 낮은 precision, reduction/norm/loss 일부는 높은 precision을 사용할 수 있다. parameter storage dtype, compute dtype, accumulator dtype, optimizer master state를 구분한다. BF16은 FP16보다 exponent 범위가 넓어 보통 loss scaler가 필요 없지만 mantissa는 짧다.

**Overflow 합의.** DDP rank 하나에서 inf가 생겼다면 모든 rank가 같은 optimizer step을 skip해야 parameter가 일치한다. found-inf flag를 collective로 합의하는지 확인한다. 일부 rank만 step하면 다음 collective 전부터 model이 갈라진다. scaler state도 rank별로 같아야 한다.

**Scheduler 소유권.** optimizer가 overflow로 skip됐는데 loop counter 기준 scheduler가 step하면 LR만 전진한다. update-based schedule은 실제 successful optimizer step을 세는지 확인한다. accumulation microstep마다 scheduler를 부르면 K배 빠르게 진행된다. checkpoint에는 counter 의미를 함께 저장한다.

**Multiple optimizer atomicity.** actor의 일부 parameter나 multimodal tower에 별도 optimizer를 쓰면 A는 step하고 B는 overflow로 skip하는 상태가 가능하다. 전체 update를 원자적으로 보려면 overflow를 미리 합의하고 모두 step 또는 모두 skip해야 한다. 의도적으로 비동기라면 PolicyVersion에 부분 update 의미를 포함한다.

**NaN 조사 체크리스트.** 마지막 valid CheckpointID를 고정한다. failing GoldenBatchID를 보존한다. eager FP32 reference를 만든다. forward hook으로 first bad node를 찾는다. fused/compile을 끈다. backward hook으로 first bad gradient를 찾는다. scaled/unscaled를 비교한다. optimizer step 전 parameter를 보존한다. 한 옵션씩 복원한다. 원인 수정 뒤 같은 fixture와 주변 batch를 회귀 test로 둔다.

**Plateau 조사 체크리스트.** target/mask/valid count, repeated batch, trainable parameter, gradient zero ratio, LR, clip coefficient, skipped step, delta/weight ratio, optimizer moment RMS, validation contamination을 순서대로 확인한다. 작은 subset을 과적합할 수 없는 stack은 대규모 run으로 보내지 않는다.

**분산 hang 결정 트리.** 모든 rank의 마지막 collective sequence number를 비교한다. 한 rank가 backward에 진입하지 않았으면 data exception이나 divergent branch다. collective 종류/shape가 다르면 unused parameter·conditional MoE·no-sync 경계를 본다. 모두 같은 collective에서 멈췄으면 topology/NCCL/network를 조사한다. timeout으로 process를 죽이기 전 stack과 flight recorder를 남긴다.

**실험 2-E—gradcheck ladder.** scalar closed-form→FP64 tensor gradcheck→FP32 analytic reference→target dtype fused kernel 순으로 승격한다. 아래 단계가 실패하면 위 단계 benchmark를 중단한다. tolerance, seed, tensor range와 non-smooth point 회피를 test manifest에 둔다.

**실험 2-F—saved tensor budget.** 같은 block을 eager, checkpoint, fused attention으로 실행해 saved tensor bytes, allocator peak, recompute FLOP, step time을 기록한다. hook payload가 측정을 오염하므로 memory snapshot run과 detailed correctness run을 분리한다.

**실험 2-G—overflow recovery.** 입력 scale을 키워 FP16 overflow를 유도한다. 모든 rank가 step을 skip하고 parameter checksum이 유지되며 scale이 낮아지는지 확인한다. checkpoint/resume 뒤 scale과 growth tracker가 이어지는지 본다. BF16/FP32 reference에서는 같은 batch가 finite한지 확인해 data corruption과 precision 범위를 분리한다.

**실험 2-H—clipping geometry.** 같은 gradient에 global norm, parameter별 norm, value clipping을 적용해 방향 cosine과 parameter delta를 비교한다. 같은 threshold 숫자가 같은 제약을 뜻하지 않는다. recipe가 어떤 geometry를 의도하는지 명시한다.

**Test 실패를 읽는 순서.** value mismatch 전에 shape/dtype/device를 본다. forward mismatch가 있으면 backward 결과를 해석하지 않는다. forward는 맞고 gradient가 다르면 first gradient node를 찾는다. gradient가 맞고 update가 다르면 optimizer/clip/scaler다. update가 맞고 resume만 다르면 durable state와 next batch다.

**최종 state ledger.** graph node는 step-local, saved tensor는 backward-local, accumulated gradient는 update-window-local, parameter와 optimizer moment는 run-durable, scaler/scheduler/RNG/sampler는 resume-durable이다. lifetime이 다른 상태를 checkpoint payload 하나라는 말로 뭉개지 않는다.

**다음 장 검증 checkpoint.** 3장은 이 순서를 실제 loop에 배치한다. `loss_sum,valid_count,scale,found_inf,grad_norm,clip_coef,optimizer_step,scheduler_step`을 동일 UpdateID 한 행으로 기록한다. 이 ledger가 끊기면 loss curve로 상태 변이를 추론하지 않는다.

**코드 review 질문.** backward seed는 어디서 오는가. gradient는 대입인가 누적인가. 어떤 tensor가 saved되고 누가 해제하는가. in-place mutation이 있는가. checkpoint recompute가 RNG와 side effect를 보존하는가. autocast policy와 accumulator dtype은 무엇인가. scale/unscale/clip 순서는 어디에 있는가. overflow flag는 rank 전체에서 합의되는가. scheduler는 successful update를 세는가. 이 질문을 factory→training_step→wrapper→optimizer source로 추적한다.

**First-difference report 형식.** 기준/실험 RunID, GoldenBatchID, UpdateID, node path와 call index, forward/backward phase, shape/stride/dtype, expected tolerance, max abs/rel, cosine, checksum, producer source를 한 행에 둔다. first node 이전이 모두 통과했는지 명시한다. 뒤쪽 mismatch 수백 개는 원인이 해결될 때까지 secondary로 접는다.

**실패 주입 2-I—hook이 만든 오류.** gradient hook에서 실수로 `g.clamp_`를 실행하거나 None을 반환하는 변형을 만든다. 관측 도구가 parameter delta를 바꾸는지 reference와 비교한다. production profiler와 debug hook을 같은 run에 무분별하게 넣지 않는 이유다.

**실패 주입 2-J—RNG recompute 불일치.** dropout이 있는 block을 activation checkpoint하되 RNG preservation을 끈 변형을 reference와 비교한다. forward output은 원 실행에서 같지만 recompute mask가 달라 backward gradient가 갈라질 수 있다. dropout 0 test만으로 이 branch를 덮었다고 말하지 않는다.

**운영 경보로 연결한다.** `found_inf`, skipped update, scale 감소, clip coefficient, non-finite gradient count를 UpdateID에 묶는다. 단발 overflow와 지속 corruption을 window로 구분한다. 경보가 울리면 failing batch와 마지막 valid checkpoint를 보존하고 자동으로 LR만 낮추지 않는다. data anomaly와 numeric range를 통제 실험으로 분리한다.

**독자 확인 문제.** scale 1024인 gradient의 scaled norm이 5120이고 clip threshold가 1일 때 unscale 전 clip과 후 clip의 결과 차이를 계산한다. valid count가 2와 8인 microbatch의 mean을 반씩 더하는 objective와 token mean을 식으로 비교한다. DDP가 rank average를 수행할 때 필요한 scale을 framework semantics와 함께 설명한다.

**최종 인계 표.** 3장에는 step state machine, 10장에는 hook/tensor ledger, 11장에는 unscaled unclipped gradient와 clipped gradient 둘 다, 14장에는 saved tensor와 scaler state, 15장에는 local/global gradient ownership, 17장에는 durable optimizer·scaler·scheduler 상태를 넘긴다. consumer가 checksum과 UpdateID를 읽지 못하면 handoff는 실패다.

**2장 종료 점검.** 같은 golden loss에서 scalar analytic gradient, tensor autograd gradient, fused backward gradient, accumulation 뒤 gradient를 서로 구분해 설명한다. 각 값이 어느 dtype과 scale, reduction 상태인지 단위를 붙인다. unscale·clip·step 뒤 parameter delta를 예측한다. overflow와 checkpoint resume에서 어떤 상태가 durable해야 하는지 답한다.

테스트가 실패하면 허용 오차부터 늘리지 않는다. 입력 checksum, forward first difference, backward first difference, update state 순서로 원인을 좁힌다. nondeterministic kernel이면 반복 분포와 deterministic reference를 함께 남긴다. 고치지 못한 차이는 `NotExecuted`가 아니라 `Unresolved`로 구분한다.

모든 결과에는 source commit, runtime version, device, dtype, seed, batch ID와 tolerance policy를 붙인다. 이 metadata가 없으면 수치 하나가 우연한 실행인지 회귀 가능한 근거인지 판정할 수 없다. 성공한 test와 skip된 test를 같은 통과 수치로 합산하지 않는다.

마지막으로 hook을 모두 제거한 clean run에서 loss와 parameter delta를 다시 비교한다. 관측 도구가 만든 graph break, synchronization, gradient mutation이 사라진 뒤에도 결론이 유지돼야 한다.

**이 장이 넘기는 것.** gradient snapshot, valid-label denominator, parameter delta 이전 상태를 3장과 11장에 넘긴다.

**다음 장에서 깨질 수 있는 것.** checkpoint가 model만 저장하면 같은 gradient를 재현해도 다음 batch와 optimizer update는 달라진다.

**검증 체크포인트.** finite difference, autograd, 닫힌식 gradient가 일치하고 `K` microbatch와 결합 batch의 update가 허용 오차 안에서 같아야 한다.

## 2.2 한 optimizer update의 순서를 고정한다

leaf gradient가 존재한다는 사실과 parameter가 갱신됐다는 사실 사이에는 여러 관문이 있다. accumulation을 끝내고, scale을 복원하고, 모든 rank의 finite 여부를 합의하고, clipping과 optimizer write를 완료한 뒤에야 하나의 update가 commit된다.

### 2.2.1 한 optimizer update를 상태기계로 읽는다

**Trainer loop의 들여쓰기가 의미하는 것**

기준 Transformers commit `36deb0b53ed0863f4b4dfdea23dcaec7f3df3701`의 `src/transformers/trainer.py:1729-1814`를 읽으면 update 경계가 드러난다. 바깥 loop는 optimizer update 하나를, 안쪽 loop는 누적할 microbatch를 담당한다. `1737-1745`는 마지막 remainder를 포함한 batch 묶음과 `num_items_in_batch`를 준비한다. `1755-1774`는 microbatch별 forward/backward를 수행하고 마지막이 아니면 `no_sync`로 gradient collective를 미룬다.

`1793-1814`에서만 clip, optimizer step, scheduler step, zero-grad, global-step 증가가 일어난다. 따라서 log의 `step`이 microstep인지 update인지 먼저 확인한다. throughput은 microbatch를, learning-rate schedule은 update를 셀 수 있다. 같은 `step=1000`이 같은 학습 상태를 뜻하지 않는다.

| 전이 | 읽는 상태 | 쓰는 상태 | 대표 증상 |
|---|---|---|---|
| fetch | sampler cursor,RNG | batch IDs,count | resume sample 차이 |
| forward | params,batch | activation,loss | non-finite activation |
| backward | loss,graph,scale | accumulated grad | zero/non-finite grad |
| unscale | scaler,grad | true-scale grad | clip 기준 오류 |
| clip | global grad norm | clipped grad | update 폭 변화 |
| step | grad,moments,params | params,moments | weight divergence |
| schedule | update count | next LR | off-by-one |
| zero | grad buffers | None/zero | leakage·state 차이 |
| commit | 모든 상태 | checkpoint | resume divergence |

각 함수가 읽고 쓰는 tensor와 scalar를 적어야 failure injection의 관측점이 생긴다. `trainer.py:1776-1788`의 `logging_nan_inf_filter`가 표시용 누계를 보정해도 계산된 gradient를 되돌리지 않는다. dashboard가 매끈해졌다는 것을 학습 복구로 오독하지 않는다.

**backward에 들어가는 네 개의 loss**

`trainer.py:1907-1976`의 `training_step`은 inputs를 준비하고 `compute_loss`를 호출한 뒤 backend 의미에 맞춰 `accelerator.backward`로 넘긴다. 조사할 때 model raw scalar, 전체 valid target으로 정규화한 objective, backward 직전 scalar, scaler가 내부 증폭한 scalar를 구분한다. dashboard에 기록되는 값도 별도다.

사용자 정의 loss가 `num_items_in_batch`를 무시하거나 이미 나눈 loss를 다시 accumulation factor로 나누면 전체 scale이 바뀐다. DeepSpeed나 tensor-parallel backend가 divisor를 소유할 수 있어 무조건 `loss/K`가 답은 아니다. 같은 fixture에서 concat reference parameter gradient와 비교해 실제 의미를 판정한다.

### 2.2.2 mixed precision은 복구 프로토콜이다

FP16 dynamic loss scaling은 loss에 `s`를 곱해 작은 gradient를 표현 범위로 올린 뒤 step 전에 `1/s`로 되돌린다. non-finite가 발견되면 step을 건너뛰고 scale을 낮춘다. 핵심은 순서다. scaled gradient를 clip하면 실제 threshold가 달라지고, skip 결정 전에 weight decay나 moment update가 일어나면 parameter가 움직인다.

BF16은 exponent 범위가 넓지만 mantissa 정밀도가 낮다. “BF16이면 안전”이 아니라 activation, reduction, gradient collective, master weight, optimizer state dtype을 각각 확인한다. 1장의 loss가 logits만 FP32로 올려도 앞 activation과 뒤 optimizer가 모두 FP32라는 뜻은 아니다.

scaler checkpoint에는 scale과 growth tracker가 포함돼야 한다. 누락하면 resume 첫 batch에서 overflow skip 여부가 달라진다. 복구 test는 다음 batch IDs뿐 아니라 scale, found-inf, skipped-step, parameter delta를 비교한다. parameter가 load 시점에 같다는 것만으로 trajectory 동일성을 주장하지 않는다.

loss가 finite여도 gradient가 non-finite일 수 있고 gradient가 finite여도 optimizer moment가 overflow할 수 있다. forward boundary, backward parameter group, unscale 뒤, optimizer state 뒤를 따로 검사한다. `nan_to_num`은 목표를 바꾸는 조치이지 원인 해결이 아니다.

### 2.2.3 clipping을 기하와 소유권으로 이해한다

모든 gradient를 vector `g`로 보면 global L2 clip은 `g'=g·min(1,c/(||g||+ε))`다. 방향은 유지하고 길이만 줄인다. parameter별 clip은 block별 계수가 달라 전체 방향을 바꾼다. value clip은 좌표별 절단이다. 세 방법을 모두 “gradient clipping”이라는 이름으로 뭉치지 않는다.

sharded training의 global norm은 각 shard squared norm을 합쳐야 한다. local norm으로 clip하면 world size와 partition에 따라 effective threshold가 바뀐다. `trainer.py:2550-2560`의 `_clip_grad_norm`이 accelerator로 위임하는 이유는 backend가 소유한 shard와 precision 의미를 사용하기 위해서다. wrapper 밖에서 parameter를 순회한 custom 구현은 전체 shard를 보지 못할 수 있다.

clip 전 norm, clip 계수, clip 후 norm을 함께 남긴다. clip 후 값만 보면 폭발이 가려진다. clip 빈도가 증가하면 threshold부터 올리지 말고 어느 layer와 sample이 norm을 만들었는지 조사한다. 큰 norm은 오류일 수도 드문 어려운 sample의 유효 신호일 수도 있다.

microbatch마다 clip한 합과 모두 누적한 뒤 한 번 clip한 값은 다르다. effective batch objective라면 accumulation 뒤 update 직전에 clip한다. per-sample clipping은 differential privacy처럼 별도 알고리즘이다. no-clip, microbatch-clip, post-accumulation-clip의 gradient cosine과 parameter delta를 비교하는 통제 실험으로 차이를 확인한다.

### 2.2.4 optimizer state를 원장으로 만든다

optimizer는 gradient를 빼는 함수가 아니라 parameter별 상태 전이기다. SGD momentum은 velocity, AdamW는 first/second moment와 step counter를 읽고 쓴다. bias correction이 어느 step을 쓰는지, frozen 또는 sparse parameter가 언제 state를 얻는지 확인한다.

AdamW의 decoupled decay는 adaptive update와 별도로 parameter를 감쇠한다. loss에 L2 penalty를 더하면 그 gradient가 moment와 preconditioner를 통과하므로 같지 않다. bias와 norm을 no-decay로 두는 recipe는 이름 pattern 설명이 아니라 실제 parameter-group membership manifest로 검증한다.

각 group에는 parameter IDs와 alias, lr, betas, epsilon, decay, scheduler linkage, state dtype/device를 둔다. adapter 삽입이나 model surgery 뒤 누락·중복을 검사한다. tied parameter가 두 group에 들어가면 한 update에서 두 번 step될 수 있다. 같은 값의 복사본과 같은 storage alias를 구분한다.

step 전후 `Δθ`를 선택 parameter에 기록한다. gradient와 delta cosine, `||Δθ||/||θ||`, decay 기여, moment RMS를 본다. gradient가 정상인데 delta가 0이면 LR, frozen flag, skipped step, group 누락을 본다. Adam delta가 현재 gradient 반대 방향이 아니어도 momentum과 decay 때문일 수 있다.

작은 FP64 fixture에서는 첫 두 step을 손계산한다. 첫 step만 맞는 test는 bias correction과 resume counter 오류를 놓친다. 중간에 zero-gradient step을 넣어 decay와 momentum 지속도 검사한다.

**scheduler의 시간축과 skip 의미**

learning rate가 epoch, dataloader batch, optimizer update, consumed token 중 어느 축의 함수인지 적는다. accumulation이나 world size를 바꾸면 batch 기반 warmup은 같은 token 예산이 아니다. token schedule이면 input, supervised, unique token 중 어느 count인지 고정한다.

Trainer `trainer.py:1803-1809`는 optimizer step이 skip되지 않았을 때 일반 scheduler를 진행한다. overflow인데 LR 시간만 흐르는 것을 막는다. custom loop가 scheduler를 optimizer보다 먼저 호출하거나 skip에도 호출하면 off-by-one이 생긴다. metric 기반 scheduler에는 별도 state와 호출 경로가 있다.

resume에는 scheduler config만 아니라 내부 step과 metric history가 필요하다. 첫 resume update에서 실제 사용한 LR과 update 뒤 표시되는 next LR을 구분한다. dashboard 한 점의 timestamp가 update 전인지 후인지도 기록한다.

**zero-grad와 graph 생명주기**

`zero_grad(set_to_none=True)`와 zero tensor 채우기는 의미가 다를 수 있다. optimizer는 None gradient parameter를 skip하고 zero gradient parameter에는 decay나 step counter를 적용할 수 있다. 조건부 expert와 동결 parameter에서 차이가 커진다. 메모리 최적화 옵션으로만 설명하지 않는다.

`retain_graph=True`를 습관적으로 쓰면 saved tensor가 살아 memory가 증가한다. 여러 loss를 같은 graph에서 backward해야 하는지, 먼저 합칠 수 있는지 본다. activation checkpointing은 forward activation을 덜 저장하고 backward 중 재계산한다. dropout RNG와 side effect가 재계산에서 같아야 한다. activation checkpoint와 지속성 checkpoint를 혼동하지 않는다.

in-place version 오류를 `.data`로 우회하면 검사를 없앨 뿐 gradient를 고치지 않는다. anomaly detection은 조사 run에서 최초 backward node를 찾는 도구이며 정상 장기 run에는 비용이 있다. graph break 의심 시 selected parameter의 `requires_grad`, `grad_fn`, first-zero boundary를 기록한다.

## 2.3 DDP·checkpoint·parameter alias의 소유권

분산 환경에서는 같은 이름의 tensor가 어느 rank와 module에 속하는지를 먼저 알아야 한다. collective 제어 흐름, checkpoint cut, parameter·buffer·alias를 하나의 소유권 원장으로 묶으면 NaN과 plateau가 계산 오류인지 불완전 commit인지 구분된다.

### 2.3.1 분산 collective에서 제어 흐름을 맞춘다

DDP `no_sync`는 마지막 microbatch까지 all-reduce를 미룬다. rank마다 accumulation 길이나 skip 결정이 다르면 일부 rank만 collective에 들어가 hang이 난다. sampler exhaustion, zero-valid batch, exception을 rank-local 분기로 처리하지 않는다.

gradient bucket은 parameter readiness와 겹쳐 통신한다. unused parameter나 조건부 expert가 bucket completion을 지연시킬 수 있다. hang 조사에는 rank별 마지막 collective sequence, bucket parameter 목록, batch ID, accumulation phase가 필요하다. 여기서는 NCCL/CUDA를 실행하지 않고 이 증거 계약만 정의한다.

overflow skip 역시 group 합의가 필요하다. 한 rank만 non-finite여도 replica를 일치시키려면 모든 data-parallel rank가 같은 optimizer/scheduler 전이를 해야 한다. detector flag의 collective 범위가 DP group인지 world인지 topology에 붙인다.

### 2.3.2 checkpoint는 원자적 커밋이다

sample-exact resume에는 model뿐 아니라 optimizer moments와 step, scheduler, scaler, CPU/device RNG, sampler cursor, microstep과 accumulation phase, consumed sample IDs, callback state가 필요하다. microbatch 중간을 저장하려면 `.grad`까지 보존하거나 마지막 committed update로 되돌아간다.

payload를 임시 위치에 쓰고 checksum을 확인한 뒤 completion marker를 마지막에 publish한다. 일부 rank shard만 존재하는 checkpoint를 선택하면 안 된다. 모든 shard와 metadata가 같은 CheckpointID인지 검증한다. 저장 성공 응답 유실 뒤 retry도 중복 UpdateID를 만들지 않아야 한다.

uninterrupted run과 interrupted/resumed run의 다음 batch, raw loss, valid count, gradient checksum, clip norm, skip flag, LR, parameter/moment delta를 update별 비교한다. 최종 metric이 비슷한 것은 exact resume 증거가 아니다. 비결정 kernel을 허용하면 tolerance와 최초 divergence를 명시해 statistical equivalence와 exactness를 나눈다.

forward 뒤 종료, backward 중 종료, optimizer shard 일부 저장 뒤 종료, marker 직전 종료를 각각 주입한다. 미완성 artifact는 loader 후보가 되지 않아야 한다. world size 변경 load 성공은 trajectory 동일성과 별개이며 weights-only, optimizer-equivalent, sample-exact 복구 수준을 구분한다.

### 2.3.3 NaN과 plateau 결정 트리

loss가 처음 non-finite면 input IDs/mask/count, embedding, block boundary, logits, per-token CE 순으로 최초 tensor를 찾는다. loss는 finite이고 grad만 깨지면 backward와 scale/unscale을 본다. grad도 finite인데 parameter가 깨지면 optimizer state와 fused step 입력을 본다. 표시용 NaN filter 뒤가 아니라 raw metric을 사용한다.

plateau에서는 objective target과 valid count부터 확인한다. gradient 0이면 frozen/detach/underflow, 정상인데 delta가 작으면 LR·clip·preconditioner, delta도 정상인데 evaluation만 정체면 mixture·metric·contamination을 본다.

한 batch overfit은 유용한 분기다. 고정 batch와 RNG에서 loss가 내려가지 않으면 local training path를 의심한다. 내려가면 sampler, schedule, regularization, distributed boundary로 범위를 넓힌다. 여기서 실행하지는 않으며 실험의 독립변수와 판정만 정의한다.

### 2.3.4 update 소유권과 durable state 계약

한 update에 대해 batch IDs와 valid targets, raw/normalized/backward loss, scaler, grad sync, unscaled norm, clip 계수, optimizer group과 moments, 사용 LR, parameter delta, zero policy, UpdateID를 설명할 수 있어야 한다. 하나가 빠지면 “한 step”은 재현 단위가 아니다.

3장에는 이 원장을 연속 run과 dataloader cursor, checkpoint로 잇는다. 11·12장은 optimizer 수학을, 15~17장은 shard ownership과 recovery를 확장한다. 이 장의 목적은 특정 API 암기가 아니라 어느 stack에서도 상태 전이 순서를 증거로 복원하는 능력이다.

**옵션을 상태 변화로 번역하는 사전**

**accumulation과 batch 계열**

`gradient_accumulation_steps=K`는 optimizer step 사이 backward 호출 수의 목표를 바꾼다. 같은 per-device batch와 world size라면 nominal examples per update는 늘지만 valid targets는 mask와 길이에 따라 달라진다. 마지막 remainder, iterable exhaustion, zero-valid skip 때문에 실제 K도 다를 수 있다. config 값과 update별 observed microbatch count를 함께 남긴다.

`per_device_train_batch_size`는 rank마다 내는 example 수를 바꾼다. sequence length와 packing이 다르면 memory와 token 수가 선형이라고 단정하지 않는다. `dataloader_drop_last`는 마지막 batch를 버려 노출과 total steps를 바꾼다. worker와 prefetch는 성능뿐 아니라 worker RNG와 resume cursor에 영향을 준다.

`max_steps`와 `num_train_epochs` 중 무엇이 우선하는지 호출 stack을 확인한다. iterable dataset은 epoch 의미가 합성될 수 있다. warmup ratio가 total steps에서 계산되면 max steps 변경이 warmup 절대 길이도 바꾼다.

**precision과 compilation 계열**

`fp16`, `bf16`, `tf32`는 한 축의 세 값이 아니다. 앞 둘은 autocast/parameter/gradient 정책에, TF32는 CUDA matmul의 FP32 입력 계산 모드에 연결된다. 여기서는 runtime을 실행하지 않지만 recipe에는 backend capability와 허용 오차를 기록한다.

gradient checkpointing은 activation 저장과 backward 재계산을 바꾸며 지속성 checkpoint 주기와 무관하다. reentrant 선택은 graph와 RNG 제약을 바꿀 수 있다. compile은 graph capture와 fusion을 바꿔 hook과 anomaly detection의 경계를 바꾸므로 조사에는 eager reference가 필요하다.

**optimizer와 persistence 계열**

`learning_rate`는 base 값과 현재 group 값을 구분한다. `weight_decay`는 적용 group과 decoupled 여부를 확인한다. betas는 moment 시간 척도, epsilon은 작은 second moment 영역의 effective step을 바꾼다. `max_grad_norm=0`은 구현에서 clip 비활성인지 확인한다.

`logging_steps`는 원래 관측 주기지만 callback side effect가 있으면 state에 영향을 줄 수 있다. `save_steps`와 retention은 persistence를 바꾼다. `load_best_model_at_end`는 metric, direction, save/eval cadence를 통해 최종 checkpoint 선택을 바꾼다. 마지막 parameter와 best parameter를 혼동하지 않는다.

**parameter, buffer, alias를 함께 본다**

optimizer가 갱신하는 것은 parameter지만 model에는 running statistics, RNG-driven cache, quantization observer 같은 buffer도 있다. frozen parameter라도 train mode에서 buffer는 변할 수 있다. adapter training에서 `requires_grad=False`만으로 backbone 완전 고정을 주장하지 않는다. mode와 buffer checksum을 함께 본다.

EMA를 쓰면 live parameter와 shadow parameter라는 두 상태가 생긴다. evaluation과 checkpoint가 어느 쪽인지, step 뒤 언제 EMA를 갱신하는지 적는다. skipped optimizer step에도 EMA가 진행되면 시간축이 갈라진다.

weight tying은 두 이름이 한 parameter를 가리키는 alias다. flatten/shard wrapper는 flat buffer view로 바꿀 수 있다. name probe가 중복 집계하거나 stale reference를 잡지 않게 identity, storage range, logical name, shard owner를 기록한다.

optimizer를 만든 뒤 wrapper가 parameter object를 교체하면 optimizer가 옛 object를 들 수 있다. adapter injection, distributed wrapping, optimizer construction 순서는 API 취향이 아니라 소유권 계약이다. 각 단계 전후 parameter IDs와 group reference를 비교한다.

## 2.4 accumulation·통신·callback을 실행서로 검증한다

accumulation 크기나 통신 겹침을 바꾸면 단순히 속도만 달라지지 않는다. gradient의 합산 계수, optimizer state 메모리, evaluation callback의 시점까지 함께 움직이므로 통제 실험과 증상별 실행서에서 각각의 효과를 분리한다.

### 2.4.1 accumulation 대수와 통제 실험

microbatch i의 sum을 `S_i`, count를 `N_i`라 하면 원하는 gradient는 `∇ΣS_i/ΣN_i`다. 각 mean을 backward해 K로 나누면 `Σ∇S_i/(K N_i)`이고 N이 같을 때만 같다. 각 sum을 global count로 나누어 backward해야 token-global objective가 된다.

DDP rank 평균이 추가되면 rank factor도 고려한다. rank·microbatch별 S,N을 표로 만들고 single concat reference와 parameter gradient를 비교한다. scalar가 우연히 같아도 token별 방향이 달라 gradient가 다를 수 있다.

dynamic packing에서는 future N을 미리 모를 수 있다. update batch들을 prefetch해 count를 구하거나 gradient buffer를 사후 rescale하는 설계가 필요하다. collective가 이미 실행된 뒤 rescale 가능한지와 buffer precision을 함께 본다.

clip 실험에는 no clip, microbatch clip, post-accumulation clip을 둔다. 최종 gradient cosine, norm, parameter delta를 비교한다. 두 parameter group이 norm 99:1인 fixture에서 global과 group clip 차이도 본다. gradient clip 뒤 decay가 큰 delta를 만들 수 있어 update/weight ratio를 함께 기록한다.

### 2.4.2 optimizer state dtype과 메모리

BF16 parameter라도 Adam moments는 FP32일 수 있고 master parameter 복사본도 있을 수 있다. sharding이 parameter, gradient, optimizer state 중 무엇을 분할하는지 원장에 둔다. offload는 device/host 이동과 synchronization을 추가한다.

8-bit optimizer는 moment quantization block, scale, error와 update kernel을 바꾼다. “메모리만 절감한 Adam”이라고 단정하지 않는다. 작은 fixture에서 dequantized moment, FP32 reference, parameter delta error와 outlier block을 비교한다.

checkpoint에는 logical state와 physical shard/quantization metadata가 필요하다. world size나 implementation을 바꿔 load할 때 변환 범위와 손실을 명시한다. state key 이름이 같다는 사실만으로 호환을 주장하지 않는다.

### 2.4.3 통신과 계산 겹침

DDP는 gradient readiness 순서대로 bucket collective를 시작한다. bucket 크기는 launch 수, overlap, peak memory와 reduction order를 바꾼다. 성능 A/B에는 gradient 허용 오차와 convergence invariant를 함께 둔다.

`no_sync` accumulation은 microbatch 통신을 줄이지만 마지막 backward에 통신이 몰린다. bucket view와 grad buffer alias가 memory와 zero 의미를 바꿀 수 있다. profile은 compute gap, collective duration, exposed communication을 나눈다.

straggler를 네트워크 탓으로 바로 돌리지 않는다. 긴 sequence, loader stall, allocator retry, thermal throttling, recovery가 upstream 원인일 수 있다. rank별 batch token count와 마지막 event를 time-aligned trace로 잇는다.

### 2.4.4 callback과 evaluation의 숨은 전이

loop는 pre-step, post-step, step-end callback을 부른다. callback이 logging만 한다고 가정하지 않는다. early stopping, checkpoint, gradient modification, external scheduler가 state를 바꿀 수 있다. callback 순서와 serialized state를 manifest에 넣는다.

중간 evaluation은 train mode를 eval로 바꾸고 dataloader와 RNG를 소비할 수 있다. 종료 뒤 mode와 RNG가 복원되는지 확인한다. asynchronous evaluation은 어느 immutable CheckpointID를 평가했는지 연결한다.

metric이 늦게 도착해 best 선택이 달라지는 race도 있다. UpdateID, CheckpointID, EvalID와 metric revision을 분리한다. dashboard 시간순과 causal 순서를 혼동하지 않는다.

**증상별 실행서**

loss spike이면 직전 batch IDs와 valid count를 고정한다. 같은 checkpoint에서 forward atlas를 비교한다. forward가 같으면 backward scale, unscale grad, clip, moments, delta로 내려간다. 재현되지 않으면 RNG, cursor, async transform을 본다.

resume divergence이면 checkpoint marker와 checksum, 다음 samples, RNG, accumulation phase, scaler, optimizer/scheduler step을 비교한다. 최초 divergence가 input이면 loader, loss면 mode/state, gradient면 precision/collective, parameter면 optimizer다.

hang이면 rank별 마지막 batch와 microstep, no-sync 여부, collective sequence, skip flag, exception을 모은다. 한 rank만 zero-valid 또는 exhaustion인지 확인한다. timeout 재시작 전에 evidence를 보존한다.

throughput regression이면 token 수와 length를 정규화한다. loader, forward, backward, communication, optimizer, checkpoint 시간을 분해한다. correctness invariant가 같은 뒤에만 성능 옵션을 판단한다.

**update 경계 test matrix**

scalar 함수로 finite difference와 autograd를 맞춘다. 작은 linear model로 reduction과 accumulation을 맞춘다. tied model로 alias gradient 합을 맞춘다. 두-rank mock으로 global denominator와 average collective를 맞춘다. overflow fixture로 all-rank skip과 scheduler 정지를 본다.

optimizer 첫 두 step, zero-gradient+decay, clip 전후를 FP64 hand reference와 비교한다. checkpoint를 forward 뒤, backward 뒤, payload 중간에 끊어 마지막 committed UpdateID에서 복구하는지 본다. uninterrupted/resume의 다음 두 updates를 state별 비교한다.

CPU toy가 CUDA fused optimizer를 증명하지 않고 kernel unit test가 sample-exact recipe를 증명하지 않는다. 실행하지 않은 test에는 expected invariant만 기록하며 성공 결과를 꾸미지 않는다.

## 2.5 autograd engine에서 optimizer ledger까지

이제 engine의 ready queue에서 시작해 mixed precision, clipping, optimizer ledger로 이어지는 실제 호출 경계를 읽는다. 각 단계가 읽고 쓰는 state와 실패 시 재시도 범위를 나누면 scheduler clock이 optimizer effect보다 앞서가는 오류도 드러난다.

### 2.5.1 autograd engine을 graph와 queue로 읽는다

**leaf, view, saved tensor의 생명주기**

PyTorch의 `loss.backward()`는 식을 상징적으로 미분하는 한 줄이 아니다. forward에서 만들어진 grad function graph의 dependency를 세고 준비된 node를 queue에 넣어 vector-Jacobian product를 실행한다. leaf parameter에는 accumulator가 gradient를 모은다. intermediate tensor는 `.retain_grad()`를 요청하지 않으면 일반적으로 `.grad`를 보존하지 않는다. 그러므로 probe가 없다는 사실을 gradient가 흐르지 않았다는 증거로 읽으면 안 된다.

view는 base storage를 공유하고 backward에서 gradient를 원래 layout으로 되돌린다. in-place 수정에는 version counter가 있어 backward에 필요한 값이 바뀌면 오류를 내도록 한다. 그러나 `.data` 같은 우회로로 mutation하면 detector를 피할 수 있다. anomaly detection은 디버깅에 유용하지만 모든 step에 켜면 synchronization과 overhead가 크다. 최소 재현에서 first bad node를 찾고 production에는 저비용 finite probe와 selected hooks를 둔다.

saved tensor는 backward 식에 필요한 forward 값이다. activation checkpointing은 이를 덜 저장하고 forward 구간을 다시 계산해 memory와 compute를 교환한다. recomputation이 dropout RNG와 autocast state, model mode를 동일하게 복원하지 못하면 원래 graph와 다른 gradient가 된다. checkpoint on/off A/B는 logits뿐 아니라 selected gradient와 parameter delta를 비교한다.

multiple backward를 하려면 graph retention이 필요한 경우가 있지만 무심코 `retain_graph=True`를 쓰면 생명주기가 늘고 memory leak를 숨긴다. 서로 다른 loss 항이 같은 graph를 공유할 때 합쳐 한 번 backward할지 순차 backward할지 정한다. gradient 합은 선형이어도 hooks, scaling, clipping을 중간에 넣으면 state 전이가 달라진다.

**hook은 관측자일 수도 변경자일 수도 있다**

tensor·module·parameter hook은 gradient를 기록하거나 바꿀 수 있다. 호출 순서, 반환값, compilation/DDP wrapping 전후 등록 시점이 중요하다. logging hook이라도 `.cpu()`나 scalar 추출로 synchronization을 만들 수 있다. hook 목록과 revision을 run manifest에 넣고 baseline에서는 모두 끈 결과를 보존한다.

gradient checkpoint, DDP reducer, optimizer hook이 함께 있으면 “backward가 끝났다”의 경계가 단순하지 않다. parameter grad가 ready된 시점과 모든 bucket collective가 끝난 시점, optimizer가 읽어도 되는 시점을 구분한다. profiler event와 UpdateID를 연결해 first use가 reduction 완료 뒤인지 확인한다.

### 2.5.2 mixed precision을 숫자가 아니라 분산 프로토콜로 다룬다

**scale과 unscale의 정확한 순서**

FP16에서 작은 gradient가 underflow되는 것을 줄이기 위해 loss에 scale $s$를 곱해 backward한다. parameter grad는 $s g$가 되며 clipping과 optimizer가 원래 $g$를 보려면 먼저 unscale해야 한다. scaled gradient를 norm clip하면 threshold가 사실상 $1/s$로 바뀐다. gradient penalty나 직접 `.grad`를 읽는 custom code도 어느 scale의 값을 보는지 명시한다.

BF16은 exponent 범위가 넓어 FP16과 같은 dynamic loss scaling이 항상 필요하지 않지만 mantissa precision 문제는 남는다. “BF16이므로 안정적”이라고 단정하지 않는다. matmul input/output, accumulation, normalization, logits, loss, optimizer state가 각각 어떤 dtype인지 dtype ledger를 만든다. autocast는 모든 연산을 같은 dtype으로 바꾸는 스위치가 아니라 op별 정책이다.

GradScaler 계열 상태에는 scale, growth tracker, growth/backoff factor, interval이 포함된다. overflow가 발견되면 optimizer step을 건너뛰고 scale을 낮추며, finite step이 일정 횟수 이어지면 키운다. checkpoint에 scaler를 빼면 resume 직후 update skip 패턴이 달라진다. scalar scale 값뿐 아니라 해당 optimizer에서 unscale 또는 step이 호출되었는지를 나타내는 단계 상태도 정상 경계에서 관리한다.

**overflow 결정은 모든 rank가 공유해야 한다**

한 rank에서 inf가 검출되고 다른 rank는 finite이면 일부만 optimizer를 실행해서는 안 된다. parameters가 즉시 갈라지고 다음 collective가 무의미해진다. overflow flag를 적절한 process group에서 합의하고 모든 rank가 step·scheduler·UpdateID를 같이 skip한다. tensor/pipeline/data parallel group이 중첩될 때 합의 범위를 model replica 전체로 닫는다.

overflow fixture는 특정 parameter grad 하나에 inf를 주입한다. unscale 전 detector, global found-inf reduction, optimizer skip, moments 불변, scheduler 불변, scaler backoff, grad clear가 기대 순서인지 본다. rank 하나에만 주입한 경우에도 모든 replica checksum이 같아야 한다. detector가 metric에서만 울리고 optimizer가 이미 실행되었다면 너무 늦다.

### 2.5.3 clipping은 parameter partition을 아는 collective다

**global norm의 대수와 구현 소유권**

$p$-norm clipping은 전체 gradient norm $G=(\sum_i|g_i|^p)^{1/p}$를 구해 계수 $\min(1,M/(G+\epsilon))$를 곱한다. parameter group별, layer별, tensor별 clipping은 다른 연산이다. 문서에 `max_grad_norm=1`만 적으면 norm scope와 dtype, unscale 시점을 알 수 없다.

FSDP나 tensor sharding에서는 각 rank가 parameter 일부만 보므로 local norm은 global norm이 아니다. local power sum을 collective로 합친 뒤 root wrapper가 scale을 적용한다. PyTorch의 FSDP에는 root instance에서 호출해야 하는 `clip_grad_norm_` 경로가 따로 존재한다. 일반 `torch.nn.utils.clip_grad_norm_`를 local shard에 적용하면 world size와 partition에 따라 update가 달라질 수 있다.

sparse gradient, mixed dtype, nonfinite 값, duplicate alias도 처리해야 한다. tied parameter를 이름 두 개로 두 번 세면 norm이 부풀고 clip 계수가 작아진다. optimizer group의 unique storage 집합과 clip 대상 집합을 비교한다. `error_if_nonfinite` 정책은 overflow protocol과 중복되거나 더 이른 fail-fast가 될 수 있으므로 순서를 고정한다.

clipping 실험은 threshold 아래 gradient, 위 gradient, 정확히 0, inf, shard 불균등을 둔다. hand norm과 반환된 pre-clip norm, post-clip norm, parameter delta를 비교한다. foreach/fused와 scalar loop 결과를 tolerance 안에서 맞추되 reduction order 차이를 기록한다.

### 2.5.4 optimizer step을 write-ahead ledger로 만든다

**AdamW 한 step의 모든 읽기와 쓰기**

parameter $\theta$, gradient $g$, first moment $m$, second moment $v$, step $t$를 명시한다. Adam 계열은 $m\leftarrow\beta_1m+(1-\beta_1)g$, $v\leftarrow\beta_2v+(1-\beta_2)g^2$, bias correction을 거쳐 update를 만든다. AdamW의 decoupled weight decay는 gradient에 L2 항을 섞는 구현과 상태 전이가 다르다. gradient가 0이어도 decay로 parameter가 움직일 수 있다.

epsilon이 square root 안인지 밖인지, maximize, amsgrad, capturable, differentiable, foreach/fused 선택은 실제 식·state·dispatch를 바꾼다. 이름이 AdamW라고 동등하지 않다. optimizer defaults와 각 parameter group override를 완전히 직렬화한다. group 순서와 parameter identity가 load 뒤 유지되는지도 검증한다.

한 update ledger에는 pre-step parameter projection, unscaled grad projection, clip coefficient, learning rate, weight decay, moment 전후, optimizer step counter, post-step delta를 둔다. 전체 tensor 복사는 비싸므로 작은 hand model에서는 전부, production에서는 deterministic sample과 checksum을 쓴다. update 식으로 재계산한 expected delta와 실제 delta가 다르면 fused kernel을 의심하기 전에 group 옵션과 alias를 본다.

**step의 성공을 commit marker로 정의한다**

optimizer가 parameter 일부를 쓴 뒤 process가 죽으면 checkpoint는 원자적이지 않다. memory 안의 update도 논리적으로 pre-update와 post-update 사이에 있다. 저장 시에는 payload를 임시 revision에 쓰고 모든 shard checksum과 optimizer/scaler/scheduler/RNG/cursor를 검증한 뒤 manifest commit marker를 게시한다. loader는 marker 없는 payload를 선택하지 않는다.

`UpdateID`는 유효 optimizer commit 때만 증가시키고 microstep과 분리한다. overflow skip, zero-valid skip, gradient accumulation 중단은 같은 global step으로 세지 않는다. 로그의 step, scheduler step, checkpoint suffix가 각기 다른 의미를 갖지 않도록 state schema를 둔다.

**scheduler는 시간의 단위를 소유한다**

**warmup ratio가 실제 몇 update인지 계산한다**

total steps는 dataset rows만으로 정해지지 않는다. epochs, sampler length, world size, microbatch, accumulation, drop-last, remainder, skipped updates가 관여한다. warmup ratio를 integer steps로 반올림하는 규칙도 구현마다 다르다. 먼저 planned updates와 observed committed updates를 분리한다.

scheduler를 optimizer step 전에 호출하는지 뒤에 호출하는지에 따라 첫 update learning rate가 달라질 수 있다. overflow로 optimizer가 skip됐는데 scheduler만 진행하면 높은 loss scale 구간 동안 schedule 시간이 흐른다. evaluation 기반 scheduler는 비동기 metric의 CheckpointID와 도착 순서가 state를 바꾼다. scheduler state와 last applied metric ID를 checkpoint한다.

resume test는 checkpoint 직전·직후 learning rate를 비교하는 데 그치지 않는다. uninterrupted와 resumed run의 다음 세 committed updates에서 applied LR, optimizer counter, scheduler counter, parameter delta가 맞아야 한다. epoch 경계와 마지막 불완전 accumulation 직전에 중단하는 fixture를 별도로 둔다.

## 2.6 DDP reducer와 sample-exact resume

DDP reducer의 bucket 완료는 곧바로 학습 step의 완료가 아니다. accumulation window와 optimizer commit을 구분한 property test로 중단 전후의 표본, gradient, parameter delta가 요구한 동일성 등급을 만족하는지 검증한다.

### 2.6.1 DDP reducer와 accumulation의 제어 흐름

**bucket readiness와 `no_sync`**

DDP는 parameter gradient가 준비되는 순서에 따라 bucket을 collective에 보낸다. bucket 크기와 parameter order는 overlap과 memory, reduction order를 바꾼다. unused parameter detection이나 conditional branch가 있으면 rank별 graph가 달라질 수 있다. 한 rank에서만 사용되지 않은 parameter는 collective mismatch의 원인이 된다.

`no_sync`는 accumulation 중간 backward의 reduction을 미루고 마지막 backward에서 동기화한다. 마지막 microbatch가 예외, zero-valid, dataloader exhaustion으로 사라지면 accumulated gradient가 동기화되지 않은 채 남을 수 있다. accumulation state machine에는 `micro_index`, expected count, sync-required flag를 둔다. 공동 skip은 이 상태까지 모든 rank에서 같아야 한다.

communication hook은 default average를 압축·지연·다른 reduction으로 바꿀 수 있다. hook revision, process group, error feedback state도 checkpoint와 동등성 계약에 포함한다. 성능 변경 뒤 single-rank reference와 multi-rank gradient projection을 비교해 목적함수가 유지되는지 본다.

**hang triage는 마지막 collective부터 역추적한다**

rank별 마지막 completed event, current bucket, collective sequence number, batch ID, valid target count, exception을 time-aligned trace로 모은다. timeout은 원인이 아니라 관측이다. loader stall, CUDA error, conditional skip, OOM recovery가 한 rank의 다음 collective 진입을 막았을 수 있다.

모든 rank stack을 같은 시각에 수집하고 first diverging event를 찾는다. NCCL debug log만으로 application control flow를 복원할 수 없으므로 UpdateID와 microstep을 correlation field로 넣는다. 재시작 전에 evidence artifact를 committed location에 보존한다.

### 2.6.2 sample-exact resume를 property test로 만든다

**복원해야 하는 최소 상태**

model parameters와 optimizer moments만 저장하면 이어 학습은 되지만 같은 run의 연속은 아니다. sampler epoch/cursor, shuffled permutation 또는 생성 규칙, data worker seed, iterable source offset, consumed sample IDs, accumulation phase, current gradients, RNG, scaler, scheduler, callbacks를 고려한다. 중간 accumulation checkpoint를 지원하지 않으면 committed update 경계에서만 저장한다고 명시한다.

distributed sampler는 world size가 바뀌면 sample partition이 달라진다. elastic resume가 sample-exact를 포기하고 statistical continuation을 제공하는지 구분한다. dataset revision이나 filtering result가 바뀐 checkpoint를 자동으로 load하지 않는다. CorpusRevision과 ordered shard checksum을 compatibility gate로 둔다.

property test는 임의의 중단 지점을 생성한다. uninterrupted N updates와 K에서 중단 후 N까지 재개한 run을 비교한다. 다음 sample sequence, losses, gradients, optimizer/scaler/scheduler state, parameter projection이 계약 범위에서 같아야 한다. payload 쓰기 중 kill, marker 전 kill, marker 후 kill을 넣어 loader가 마지막 완전 commit만 고르는지 본다.

**corruption과 partial state를 조용히 받아들이지 않는다**

각 shard에 checksum과 logical ownership을 둔다. optimizer shard 하나가 없을 때 moments를 0으로 초기화해 진행하는 것은 복구가 아니라 새 recipe다. strict load가 실패하고 명시적 변환 도구만 migration을 수행하게 한다. dtype이나 world-size 변환에는 source/target schema와 손실 가능성을 기록한다.

checkpoint 검증은 파일 존재 여부보다 tensor key, shape, dtype, alias, finite, optimizer group mapping, counter consistency를 본다. manifest의 UpdateID와 scheduler/scaler 내부 counter가 모순이면 fail closed한다. 보관 정책으로 오래된 checkpoint를 지우기 전에 최신 checkpoint의 실제 restore drill을 통과시킨다.

### 2.6.3 실무 결정 트리를 test와 연결한다

**loss spike와 nonfinite**

첫째 같은 GoldenBatch를 checkpoint에서 다시 forward한다. logits가 다르면 parameter·mode·RNG·kernel을 본다. logits가 같고 loss가 다르면 reduction과 mask다. loss가 같고 gradient만 다르면 scaling, graph, collective다. unscaled grad가 같고 delta가 다르면 clipping·optimizer다. 각 분기에는 다음 probe와 중단 조건을 붙인다.

nonfinite이면 최초 tensor를 forward activation, scaled grad, unscaled grad, moment, parameter로 나눈다. scaler가 skip했는지 모든 rank에서 확인한다. alert가 난 뒤 자동으로 learning rate를 줄인 run은 원인을 가릴 수 있으므로 원본 artifact를 보존하고 recovery를 child revision으로 만든다.

**plateau와 지나치게 작은 update**

loss가 평평하면 valid targets와 data repetition, learning rate, grad norm, clip fraction, parameter delta, optimizer moment ratio를 본다. gradient가 있는데 delta가 작으면 LR, preconditioner, weight decay, precision을 본다. gradient 자체가 0이면 mask, detached graph, frozen parameters, saturated activation을 본다. evaluation만 plateau면 train/eval template와 mode 차이를 확인한다.

**throughput regression과 memory 증가**

loader, forward, backward, collective, optimizer, checkpoint 시간을 token length로 정규화한다. hooks와 anomaly detection, retained graph, logging synchronization, bucket change를 본다. memory가 step마다 늘면 Python reference, retained outputs, graph retention, callback artifact를 조사한다. allocator reserved 증가와 live tensor 증가를 구분한다.

### 2.6.4 reducer·resume 회귀 묶음

**단위 test에서 장애 drill까지**

scalar chain rule과 finite difference, view/in-place detector, tied alias gradient를 단위 test한다. accumulation remainder, all-ignore, overflow, clip threshold, AdamW 첫 두 step, scheduler skip을 작은 model에서 hand reference와 맞춘다. two-rank fixture로 uneven targets, rank-local overflow, unused branch, no-sync 마지막 microbatch를 검증한다.

checkpoint drill은 forward 직후, accumulation 중간, optimizer payload 중간, commit marker 전후에 중단한다. 지원하지 않는 중간 경계는 명시적으로 저장을 거부해야 한다. restore 뒤 첫 세 updates에서 sample IDs부터 parameter delta까지 비교한다. test 결과에는 revision, device/backend, tolerance와 실행 여부를 적는다.

**다음 장으로 넘기는 update manifest**

인계 artifact는 update가 실제로 일어났는지를 시간 순서대로 재구성할 수 있어야 한다. loss 단계에는 `LossNumerator`와 `ValidTargetCount`를, mixed-precision 단계에는 적용한 scale과 `found-inf`, 모든 rank가 합의한 skip 여부를 남긴다. gradient 단계에는 unscaled global norm과 clip coefficient를, optimizer 단계에는 group options, 학습률, moment projection과 `UpdateID`를 기록한다. 끝으로 scheduler·scaler counter와 checkpoint commit ID를 연결하고, 재개에 필요한 sample cursor·RNG·alias ledger·callback state를 함께 보존한다.

완료 판정은 optimizer가 호출되었다는 로그가 아니다. 모든 rank가 동일한 유효 update를 commit했고, 그 parameter delta를 loss와 optimizer 식에서 재구성할 수 있으며, 임의 중단 뒤 같은 다음 samples와 state로 돌아왔다는 증거다. 이 세 조건이 닫혀야 3장의 작은 GPT 실행을 신뢰할 수 있다.

## 2.7 관측성·형식 불변식·hand reference

대규모 trace를 모두 저장하지 않고도 잘못된 update를 찾으려면 비용별 관측 계층이 필요하다. 상태기계의 불변식을 assertion으로 바꾸고, 가장 작은 hand reference를 장기 보존해 source upgrade 뒤에도 의미가 바뀌지 않았는지 대조한다.

### 2.7.1 update 관측성을 비용 계층으로 설계한다

**항상 켜는 계측과 사건 때만 켜는 계측**

매 update마다 보존할 값은 global numerator/count, scale, skip reason, global grad norm, clip coefficient, learning rate, step duration, UpdateID처럼 작고 bounded한 scalar다. layer별 norm과 parameter projection은 낮은 주기로 sampling한다. 전체 activation과 gradient dump는 사건이 발생한 GoldenBatch를 격리한 재현에서만 쓴다. 모든 것을 항상 기록하면 synchronization, storage, 정보 노출이 학습 자체를 바꾼다.

metric timestamp만으로 인과 순서를 복원하지 않는다. microstep, UpdateID, CheckpointID, EvalID를 명시하고 rank clock 차이를 고려한다. optimizer duration이 길어진 시점과 checkpoint upload가 겹쳤다면 동일 host I/O contention인지 trace로 확인한다. dashboard annotation은 config revision과 자동 recovery action을 가리킨다.

gradient norm histogram에는 parameter 수나 tensor name을 고 cardinality label로 넣지 않는다. module class와 depth bucket처럼 제한된 차원을 사용하고 상세 이름은 artifact에 둔다. tied storage는 중복 집계하지 않는다. missing gradient, zero gradient, nonfinite gradient count를 구분한다. frozen parameter의 missing은 정상일 수 있으므로 expected-trainable set과 비교한다.

**alert에서 자동 조치까지의 안전 경계**

nonfinite 한 번에 job을 종료할지 scaler가 복구하도록 둘지는 precision 정책이다. 연속 skip 수, scale 하한, parameter nonfinite, replica checksum divergence에 서로 다른 severity를 둔다. 자동 LR 감소나 checkpoint rollback은 새 RunRevision을 만들고 원본 evidence를 보존한다. 조치가 성공해 loss가 정상화되어도 원인 규명이 완료된 것은 아니다.

hang watchdog은 rank별 heartbeat만 보지 않는다. 마지막 collective sequence와 microstep이 전진하는지, loader queue와 CUDA event가 전진하는지 본다. 느린 정상 batch를 kill하지 않도록 length와 checkpoint event를 고려한다. timeout 직전 stack·NCCL state·batch locator를 수집하되 민감한 원문은 복사하지 않는다.

### 2.7.2 상태기계의 형식 불변식

**한 update가 지켜야 할 순서**

`ACCUMULATING` 상태에서는 parameter와 optimizer moments가 바뀌지 않는다. `READY`에는 필요한 microbatch와 global count가 닫혀 있다. `UNSCALED` 뒤에만 norm과 clipping을 해석한다. `COMMITTED`에는 parameter, moments, scheduler, scaler, UpdateID가 정책상 일관된다. `SKIPPED`에는 skip reason과 unchanged state 범위를 기록한다. 허용되지 않은 전이를 assertion과 event log로 막는다.

예외가 어느 상태에서 났는지에 따라 재시도가 다르다. forward 실패는 같은 batch를 재시도할 수 있지만 nondeterministic transform을 다시 호출하면 입력이 달라질 수 있다. partial optimizer write 이후에는 in-memory rollback을 추정하지 말고 마지막 committed checkpoint에서 복구한다. checkpoint upload 실패는 학습 update commit과 storage commit을 구분해 처리한다.

state invariant property test는 임의 event 순서를 생성해 `step` 두 번, unscale 전 clip, skip 뒤 scheduler advance, marker 없는 restore가 거부되는지 본다. 정상 경로만 test하면 드물게 발생하는 장애가 가장 위험한 상태를 만든다. 실패 메시지에는 현재 state, 요청 event, UpdateID를 넣어 triage를 단축한다.

### 2.7.3 코드 리뷰용 질문을 실행 가능한 assertion으로 바꾼다

“gradient가 finite인가”는 unscale 뒤 unique trainable parameters의 nonfinite count assertion이 된다. “모든 rank가 같은가”는 UpdateID·skip flag·selected parameter checksum collective assertion이 된다. “resume이 정확한가”는 다음 sample IDs와 첫 세 delta의 equivalence test가 된다. 자연어 질문마다 owner, probe, expected value, failure artifact를 붙인다.

optimizer option을 추가하면 default만 문서화하지 않는다. 어느 parameter group field를 바꾸고, state allocation과 dispatch가 어떻게 달라지며, checkpoint schema와 fused eligibility에 어떤 영향이 있는지 test한다. 아무 변화가 관측되지 않으면 옵션이 ignored 되었는지 조건부 branch가 선택되지 않았는지 조사한다.

마지막 review에는 model code 담당자, data/sampler 담당자, distributed 담당자, observability 담당자가 같은 UpdateID dossier를 읽는다. 각자가 자기 metric만 보는 대신 loss에서 parameter delta와 checkpoint까지 한 경로를 왕복한다. 이 교차 검토가 통과해야 작은 예제에서 얻은 설명을 긴 학습 run의 운영 규칙으로 확장할 수 있다.

### 2.7.4 최소 hand reference를 오래 보존하는 이유

**fused 구현의 oracle은 느려도 명료해야 한다**

작은 linear model과 두 parameter group을 FP64로 계산하는 reference를 유지한다. 첫 두 update의 gradient accumulation, decoupled decay, bias correction, clipping, scheduler를 명시적 tensor 연산으로 쓴다. foreach·fused·compiled optimizer는 이 reference와 비교한다. 성능 구현을 reference로 삼으면 같은 버그를 공유할 수 있다.

fixture에는 zero gradient가 있는 parameter, weight decay 제외 group, tied alias, 마지막 불완전 accumulation을 포함한다. expected moments와 delta를 저장하되 framework upgrade 때 자동으로 덮어쓰지 않는다. 변화가 의도된 경우 식과 migration note를 사람이 검토한 뒤 golden 값을 갱신한다.

**테스트가 증명하는 범위를 표시한다**

CPU FP64 test는 상태 순서와 대수를 증명하지만 CUDA rounding과 fused dispatch를 증명하지 않는다. 단일 GPU test는 distributed collective를 증명하지 않는다. 두 rank toy는 수십 rank straggler와 장애 복구를 모두 증명하지 않는다. 각 test record에 `proves`, `does_not_prove`, backend, world size, revision을 둔다.

실행 환경이 없는 test는 expected invariant와 fixture만 남기고 `NOT_RUN`으로 표시한다. 나중에 실행 결과가 들어오면 산출물 checksum과 로그 locator를 연결한다. 계획된 test를 PASS처럼 세는 관행은 completion audit를 무력화한다.

이 원칙은 책의 코드 인용에도 적용한다. 함수 일부는 상태 전이를 설명하는 증거이지만 전체 recipe의 성공을 보장하지 않는다. 호출자, 옵션 조건, test, 분산 범위를 함께 읽어야 독자가 자기 환경에서 무엇을 다시 검증해야 하는지 알 수 있다.

최종 인계 전에는 optimizer option을 하나씩 기본값에서 바꾼 child fixture를 실행한다. 변화가 나타나야 할 state와 나타나지 않아야 할 state를 먼저 적는다. 예컨대 clipping threshold 변경은 unscaled gradient 자체를 바꾸지 않고 clip coefficient와 이후 delta를 바꿔야 한다. scheduler 변경은 같은 첫 backward gradient를 유지하면서 applied learning rate와 delta를 바꿔야 한다.

예상 영향 경계보다 앞에서 tensor가 갈라지면 실험 입력이나 RNG가 통제되지 않은 것이다. 아무 경계도 달라지지 않으면 옵션이 실제 optimizer group에 전달되지 않았거나 조건부 fused path에서 무시되었을 수 있다. 이 영향 반경 검사가 옵션 설명을 실행 가능한 지식으로 만든다.

검증 보고에는 성공한 경로뿐 아니라 의도적으로 실패시킨 assertion과 최초 detector도 남긴다. 실패를 감지하지 못한 test suite는 정상 run 수가 많아도 안전망으로 볼 수 없다. 복구 revision에서 같은 고장 주입이 예상 경계에서 차단될 때 비로소 해당 계약을 닫는다.

## 2.8 saved tensor·alias·AMP의 생애 주기

autograd memory는 forward가 끝났다고 사라지지 않는다. view와 alias가 저장된 값을 공유하고 version counter가 in-place 변경을 감시하는 동안, AMP scale state와 accumulation window도 별도 생애 주기를 갖는다.

### 2.8.1 autograd를 tape가 아니라 기여도 원장으로 읽는다

**고정 source에서 출발한다.** 로컬 micrograd snapshot은 commit `7bc720e951fe422b8f8814aa5aa1b64121d26b4c`다. `micrograd/engine.py:2`의 `Value`는 scalar 값, gradient, 이전 nodes, local backward closure를 보존한다. `:54`의 `backward`는 output에서 graph를 위상 정렬하고 역순으로 closure를 호출한다. production tensor engine보다 작지만 “한 값이 여러 경로에서 쓰이면 adjoint를 더한다”는 핵심을 숨기지 않는다.

작은 source를 읽을 때 production과 같은 기능을 억지로 찾지 않는다. micrograd에는 CUDA stream, view metadata, mixed precision, distributed reducer, saved-tensor hook이 없다. 대신 local derivative가 closure에 캡처되고, `self.grad += ...`로 기여를 합치며, output seed를 1로 놓는 구조를 정확히 본다. 이 reference는 chain rule의 oracle이지 PyTorch의 성능 구현 설명서가 아니다.

하나의 parameter `w`가 `a=w*x`, `b=w*y`, `L=a+b`에 쓰이면 `∂L/∂w=x+y`다. backward closure가 한 branch의 값을 대입하면 다른 기여를 잃는다. tied embedding/head, recurrent unrolling, residual fan-out에서도 같은 원리다. gradient buffer의 `+=`와 training loop의 accumulation은 모두 합산이지만 소유 경계가 다르다. graph 내부 fan-out 합산은 한 backward의 미분이고, microbatch accumulation은 여러 loss의 미분을 update window에 합치는 정책이다.

leaf `.grad`가 `None`인 것과 0 tensor인 것도 다르다. `None`은 아직 materialize되지 않았거나 해당 graph에 경로가 없음을 뜻할 수 있다. 0은 경로가 있었지만 derivative가 0일 수 있다. optimizer의 `zero_grad(set_to_none=True)`는 memory write를 줄이고 다음 backward가 새 buffer를 만들게 한다. 일부 optimizer는 `grad is None`인 parameter를 skip하지만 zero gradient가 있는 parameter에는 weight decay나 state update를 적용할 수 있다. zeroing 옵션이 optimizer semantics에 영향을 줄 수 있으므로 source로 확인한다.

**saved tensor는 backward의 입력 계약이다.** 연산 `z=f(x)`의 backward가 x나 z를 필요로 하면 autograd node는 해당 값을 보존하거나 재계산할 방법을 가진다. matmul은 상대 operand, sigmoid는 output, normalization은 통계와 normalized input, dropout은 mask가 필요하다. saved tensor는 forward가 끝났다고 즉시 해제할 수 없는 activation memory의 주요 원인이다.

in-place 연산이 위험한 이유는 단순히 “autograd가 싫어해서”가 아니다. backward가 과거 x를 기대하는데 storage가 새 값으로 덮이면 다른 함수의 derivative를 계산한다. version counter는 저장 당시와 backward 당시 mutation version을 비교해 조용한 오염을 막는다. view는 base storage와 mutation 관계를 공유하므로 복사처럼 다루면 안 된다.

activation checkpointing은 saved tensors를 줄이는 대신 forward 일부를 backward 때 다시 계산한다. 정확성 조건은 재계산 함수가 같은 입력과 parameter, RNG, global state에서 같은 값을 내는 것이다. dropout RNG가 달라지거나 forward가 cache·counter를 mutation하면 다른 graph가 된다. autocast context와 train/eval mode도 같아야 한다. checkpoint boundary마다 input checksum, RNG state policy, side-effect 목록을 둔다.

### 2.8.2 tensor autograd의 layout과 alias를 추적한다

**view와 copy를 값만으로 구분하지 않는다.** transpose, reshape, expand, narrow는 조건에 따라 view를 만들고 stride를 바꾼다. contiguous 호출은 필요하면 새 storage를 만든다. forward 값과 shape가 같아도 backward accumulation의 kernel, memory, alias 관계가 달라진다. custom fused Function은 non-contiguous input을 지원하는지 또는 명시적으로 contiguous copy를 만드는지 test한다.

`expand`는 stride 0 view로 한 값을 여러 위치에 보인다. backward에서는 확장된 축의 gradient를 원 storage로 합쳐야 한다. GQA에서 KV heads를 repeat하는 구현, bias broadcast, normalization scale에 같은 reduction이 나타난다. forward parity만 확인하면 backward reduction 누락을 놓친다. batch·token 중 한 위치만 nonzero인 fixture로 정확한 합산 배율을 계산한다.

parameter alias ledger에는 Python name만 넣지 않는다. canonical parameter ID, object identity, storage identity, offset, shape, stride, requires-grad, optimizer group owner를 기록한다. embedding/head tie, shared experts, adapter weight sharing을 표현한다. checkpoint save/load와 FSDP wrapping 전후에 ledger를 비교한다. alias가 의도적으로 materialize되는 변환이면 이유와 새로운 update semantics를 명시한다.

gradient hook은 관찰 자체가 graph를 바꿀 수 있다. hook이 새 gradient를 반환하면 값을 수정한다. `.item()`이나 CPU copy는 device synchronization을 일으킨다. distributed tensor의 norm은 collective가 필요할 수 있다. 항상 켜는 probe는 finite count와 sampled projection처럼 작게 만들고, 전체 dump는 격리 재현에서만 쓴다.

**custom autograd Function은 forward와 backward 두 API다.** forward가 맞는다는 것은 절반의 증거다. backward의 입력 개수, `None` 반환, higher-order gradient 지원, saved tensor lifetime, autocast decoration을 확인한다. `gradcheck`는 double precision finite difference로 작은 smooth fixture를 검사하지만 discrete top-k 경계, low-precision rounding, distributed side effect를 증명하지 않는다.

directional derivative test는 모든 parameter를 원소별 finite difference하는 비용을 줄인다. 임의 방향 v에 대해 `(L(θ+εv)-L(θ-εv))/(2ε)`와 `g·v`를 비교한다. 여러 ε를 사용해 truncation과 rounding 구간을 찾는다. ReLU kink나 routing boundary에서는 방향을 경계에서 떨어뜨린다. fused backward는 output, input gradients, parameter gradients를 별도 projection으로 비교한다.

### 2.8.3 backward 한 번과 accumulation window를 분리한다

**목표는 gradient 합이 아니라 같은 global objective다.** 1장에서 microbatch `k`의 loss numerator를 `S_k`, valid count를 `N_k`라 했다. 한 update의 목표가 `ΣS_k/ΣN_k`이면 backward 기여도 같은 분모를 써야 한다. 각 `S_k/N_k`를 K로 나누는 방식은 N이 같을 때만 등가다. dynamic padding과 assistant-only mask에서는 N이 달라진다.

window를 시작하기 전에 total count를 알 수 있으면 각 numerator를 total count로 normalize한다. streaming 때문에 미리 모르면 numerator gradient를 누적하고 마지막에 global count로 rescale하는 설계가 가능하지만 optimizer·AMP·DDP와 맞물린다. framework가 `num_items_in_batch`를 model loss에 넘기는 이유는 이 계약을 loss 계산 시점에 닫기 위해서다. wrapper가 다시 K로 나누는지 확인한다.

accumulation state에는 `window_id`, target microsteps, completed microsteps, sample IDs, numerator sum, valid count, sync pending, gradient scale, partial gradient ownership이 있다. 중간 checkpoint를 허용하지 않으면 update boundary에서만 save한다. 허용하면 partial gradients와 microstep cursor, scaler context까지 저장해야 정확히 재개할 수 있다. “gradient accumulation steps=8”만 저장해서는 부족하다.

마지막 remainder window가 K보다 짧을 때를 정의한다. drop할지, 짧은 window로 step할지, 다음 epoch과 합칠지에 따라 sample consumption과 gradient scale이 달라진다. scheduler가 update 기준이면 step 수도 바뀐다. 데이터 병렬 rank마다 remainder 길이가 다르면 collective order를 맞춰야 한다. dataset length가 world size와 batch size로 나누어떨어진다는 암묵 가정을 제거한다.

**`no_sync`는 통신 최적화이면서 상태 전이다.** DDP accumulation에서는 중간 microsteps의 gradient reduction을 미루고 마지막 backward에서 accumulated gradient를 reduce한다. 마지막에도 `no_sync`가 남으면 rank parameters가 갈라진다. 매 microstep sync하면 수학은 같을 수 있으나 통신 횟수와 rounding order가 달라진다. unused parameter 탐색과 bucket rebuild가 있을 때 control flow도 확인한다.

로컬 nanoGPT snapshot은 commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`다. `train.py:295` 부근은 공식 context manager 대신 `model.require_backward_grad_sync`를 마지막 microstep에서만 true로 둔다. 이 좌표는 해당 snapshot의 구체적 선택이다. 일반 DDP API의 영구 계약이라고 확대하지 않는다.

같은 파일 `train.py:308`은 clipping을 쓰는 경우 `scaler.unscale_(optimizer)`를 먼저 호출하고 `:309`에서 `clip_grad_norm_`, `:311`에서 `scaler.step`, `:314`에서 `zero_grad(set_to_none=True)`를 수행한다. 몇 줄의 순서가 loss scale 단위, clip 단위, step skip, buffer 생애를 정의한다. 이 source에는 완전한 crash-consistent checkpoint protocol이 없으므로 3장의 별도 설계가 필요하다.

### 2.8.4 AMP를 dtype 목록이 아니라 동적 상태기계로 이해한다

**autocast와 GradScaler는 다른 문제를 푼다.** autocast는 연산별로 선택한 dtype에서 forward를 실행해 처리량과 memory를 개선한다. GradScaler는 FP16 backward에서 작은 gradient가 underflow하는 위험을 줄이려고 loss에 scale `s`를 곱한다. BF16에서는 넓은 exponent 범위 때문에 scaler를 쓰지 않는 경우가 많지만, autocast policy와 hardware 지원은 별도 확인한다.

scaled loss `L'=sL`의 gradient는 `g'=sg`다. optimizer step 전 `g=g'/s`로 되돌려야 한다. unscale 전에 global norm을 읽으면 scale 단위의 norm이고 clip threshold가 사실상 `c/s`가 된다. unscale은 같은 optimizer에 대해 update당 한 번만 수행해야 한다. 이후 gradient를 추가 accumulation하면 scale이 섞인다.

로컬 PyTorch snapshot은 commit `3691693263d2b66a68867e39b7449876844e06cf`다. `torch/amp/grad_scaler.py:53`의 `GradScaler`, `:302`의 `unscale_`, `:375`의 `step`이 현재 확인한 핵심 좌표다. line number보다 commit과 symbol을 함께 보존한다. `step`은 optimizer별 state를 읽고 inf/NaN 검사 결과에 따라 실제 optimizer step을 호출하거나 건너뛴다.

GradScaler state에는 scale, growth tracker, growth/backoff factor, growth interval과 optimizer별 stage·found-inf가 포함될 수 있다. 정확한 schema는 revision source와 state dict로 확인한다. checkpoint에서 scale만 저장하고 growth tracker를 잃으면 resume 뒤 scale evolution이 달라진다. optimizer가 여러 개면 found-inf와 step atomicity가 각각 어떻게 처리되는지 설계한다.

**overflow skip은 update 전체의 공동 결정이어야 한다.** 한 rank에서만 non-finite가 발견되면 모든 rank가 같은 optimizer update를 skip해야 parameters가 일치한다. found-inf를 collective로 합치는 경로를 확인한다. optimizer step만 skip하고 scheduler와 UpdateID, EMA, data cursor를 전진시키는지 정책을 정한다. data를 소비한 채 update만 skip할 수도 있지만 그 사실이 재현 원장에 남아야 한다.

scale growth는 연속된 유효 steps 뒤 증가하고 overflow에서 감소한다. “scale이 크다” 자체는 품질 지표가 아니다. model, loss normalization, gradient distribution에 적응한 수치 상태다. scale이 계속 하한으로 내려가거나 skip streak가 길면 최초 non-finite layer를 조사한다. scaler가 NaN을 치료하는 것이 아니라 잘못된 update를 막고 표현 범위를 조절한다.

FP16, BF16, TF32, FP32를 한 `precision` 문자열로 설명하지 않는다. parameter storage, master weight, activation, matmul input/output/accumulator, reduction, optimizer state dtype을 표로 만든다. fused optimizer가 moments를 FP32로 유지하는지, low-precision state를 쓰는지 확인한다. dtype 변경의 first-difference가 어느 tensor인지 GoldenBatch로 찾는다.

**clipping을 안정화 장치와 목적함수 변경 사이에서 본다**

**global norm을 정확히 정의한다.** trainable unique parameters의 gradients를 하나의 vector로 이어 norm `||g||_p`를 구한다. L2 global clipping은 `||g||_2>c`일 때 `g←g·c/(||g||_2+ε)`다. parameter별 norm clipping, value clipping, adaptive gradient clipping은 서로 다른 함수다.

tied parameter를 두 name으로 중복 집계하면 norm이 부풀고 clip coefficient가 작아진다. sparse gradient는 dense와 다른 norm 경로를 쓸 수 있다. FSDP·ZeRO shard에서는 local shard squared norm을 합산해 global norm을 구해야 한다. DDP reduction 전 local gradient를 clip하면 각 rank가 다른 방향·배율로 변해 global gradient clip과 같지 않다.

#### 분산 loss의 실패 계약은 평균값 하나로 닫히지 않는다

rank별 유효 토큰 수가 다르면 올바른 목적함수는 rank loss의 평균이 아니라 `Σ_r numerator_r / Σ_r valid_count_r`다. 이를 검증할 때는 valid count를 1:7처럼 일부러 비대칭으로 만들고, 분산 gradient와 Adam update를 모든 sample을 이어 붙인 FP64 reference에 비교한다. logged loss만 맞는 시험은 DDP의 평균 배율과 optimizer delta가 동시에 맞음을 증명하지 못한다.

전 rank의 `valid_count=0`은 특별한 transaction이다. 0으로 나눈 NaN을 뒤늦게 scaler가 발견하게 두지 말고, 모든 rank가 같은 skip reason에 합의한 뒤 parameter, optimizer step, scheduler step과 global token clock을 하나도 전진시키지 않아야 한다. count collective 전·중·후 rank 하나가 사라지는 시험도 같은 no-commit oracle을 사용한다. bounded timeout과 명시적 process-group failure 없이 surviving rank가 무한히 기다리면 수치 문제가 아니라 liveness 실패다. 현재 고정 upstream의 정상 parity 시험에는 이 세 장애 assertion이 없으므로 제안 fixture와 실제 증거를 구분한다.

clipping은 outlier update를 제한하지만 원인을 제거하지 않는다. 매 step clip coefficient가 매우 작으면 effective update가 scheduler 설계와 달라진다. clipping 비율, unclipped norm, coefficient, layer/group별 contribution을 계측한다. threshold를 올려 loss가 안정된다는 이유만으로 좋은 수정이라 결론내리지 않는다.

adaptive gradient clipping은 gradient norm을 parameter norm과 비교할 수 있다. 작은 norm parameter와 bias, normalization scale 처리에 epsilon과 exclusion 정책이 중요하다. Muon처럼 matrix update를 orthogonalize하는 optimizer와 global clipping의 순서도 명시한다. optimizer-specific transform 전 raw gradient를 clip하는지 transform 후 update를 clip하는지는 다른 알고리즘이다.

**failure fixture를 만든다.** 모든 gradient를 알려진 vector로 두고 expected global norm과 coefficient를 손으로 계산한다. alias 하나, `grad=None` 하나, zero tensor 하나, non-finite 하나를 넣는다. distributed fixture는 두 ranks에 서로 다른 shard norm을 둔다. unscale 전후 scale을 2의 거듭제곱으로 두면 순서 오류를 명확히 볼 수 있다.

## 2.9 collective에서 checkpoint 복구까지

optimizer step은 parameter·moment·scheduler·effect ledger를 함께 바꾸는 분산 트랜잭션이다. 장애 지점을 collective 전후와 checkpoint 경계에 주입해 partial write가 노출되지 않는지 확인하고, 저장 메모리와 재계산 메모리를 별도로 계산한다.

### 2.9.1 DDP reducer를 collective 호출 이상의 계약으로 읽는다

**bucket은 gradient ready 순서와 통신을 연결한다.** DDP는 parameters의 gradients를 bucket에 모아 준비되는 대로 all-reduce를 시작해 backward와 통신을 겹친다. bucket order, size, unused parameter 탐색, static graph 설정이 overlap과 control flow에 영향을 준다. model parameter 등록 순서를 바꾸면 성능이 달라질 수 있지만 목적함수는 유지되어야 한다.

all-reduce가 sum을 만들고 framework가 world size로 나누는지, reduce-scatter를 사용하는지 revision에서 확인한다. communication hook은 압축, low precision, PowerSGD 같은 변환을 추가할 수 있어 수치와 state를 바꾼다. hook 이름만으로 정확성을 가정하지 않고 error feedback buffer와 checkpoint 여부를 본다.

unused parameter는 합법적 conditional graph일 수 있지만 rank마다 사용 여부가 다르면 collective ordering 문제가 된다. `find_unused_parameters`는 graph traversal 비용을 갖고 static graph 최적화와 상호작용한다. mixture-of-experts나 modality branch처럼 data-dependent path에서는 어떤 parameters가 어느 rank에서 gradient를 갖는지 명시한다.

gradient accumulation에서 intermediate `no_sync`는 local buffer에 기여를 쌓는다. 마지막 backward가 모든 expected buckets를 ready로 만들지 못하면 reducer가 hang하거나 stale gradient가 남을 수 있다. 예외로 window가 중단될 때 gradient buffer와 reducer state를 버리고 coordinated restart한다. 한 rank만 retry하지 않는다.

**rank parity를 좁은 checksum으로 감시한다.** 모든 parameter를 매 step hash하면 비싸다. 고정된 selected parameters와 random projection을 낮은 주기로 비교하고, skip flag와 UpdateID는 매 step collective invariant로 둔다. divergence가 감지되면 마지막 일치 checkpoint와 첫 불일치 update 사이를 재생한다.

floating reduction order 때문에 bitwise checksum이 항상 같아야 하는지는 backend와 deterministic policy에 따라 다르다. 동일 rank replicas라면 DDP step 뒤 parameters는 보통 같은 reduction 결과를 받지만 asynchronous mutation이나 rank-local optimizer state가 있으면 달라질 수 있다. 허용 오차 projection과 exact state ID를 함께 쓴다.

hang 조사에는 마지막 entered collective sequence, bucket index, parameter names, microstep, no-sync state, found-inf decision을 남긴다. NCCL timeout 메시지만으로 네트워크 고장이라 결론내리지 않는다. rank별 control-flow divergence, loader exception, OOM으로 한 rank가 collective에 도달하지 않은 경우가 흔하다.

### 2.9.2 optimizer step을 트랜잭션으로 모델링한다

**read set과 write set을 적는다.** optimizer step은 parameter, gradient, learning rate, group options, step count, moments를 읽는다. parameter와 moments, step count를 쓴다. scheduler는 applied LR과 자기 counter를, scaler는 found-inf와 growth tracker를 쓴다. EMA나 SWA, callback도 parameter update 뒤 상태를 쓸 수 있다.

UpdateID를 commit하려면 이 mutation들이 정책상 일관되어야 한다. process가 parameter 절반을 갱신한 뒤 죽으면 in-memory state를 신뢰할 수 없다. optimizer kernel이 foreach/fused로 여러 tensors를 묶어도 crash atomicity를 보장한다고 가정하지 않는다. durable recovery는 마지막 완전 checkpoint에서 시작한다.

gradient가 non-finite면 parameter와 moments가 바뀌지 않아야 한다는 불변식을 둔다. scheduler와 data cursor가 어떻게 되는지는 명시적 정책이다. 일부 시스템은 batch를 소비하고 update를 skip하며 scheduler도 skip한다. 다른 시스템은 attempted step을 시간축으로 셀 수 있다. 이름보다 checkpoint에 저장되는 counters와 resume 동작을 본다.

parameter group마다 optimizer 옵션과 state shape가 다르다. weight decay 제외, adapter-only LR, embedding LR, matrix optimizer와 scalar/vector fallback을 표현한다. parameter가 정확히 한 group에 속하는지, frozen parameters가 state를 할당하지 않는지 확인한다. config를 바꿔 resume할 때 group ordering과 parameter identity mapping을 검증한다.

**delta를 식에서 재구성한다.** SGD라면 momentum과 weight decay를 포함한 buffer를, AdamW라면 first/second moment, bias correction, epsilon 위치, decoupled decay를 기록한다. selected parameter 원소에 대해 pre-weight, raw gradient, clipped gradient, moments before/after, applied LR, decay, post-weight를 저장한다. framework scalar와 손계산 delta가 맞아야 한다.

foreach·fused·capturable·differentiable 옵션은 단순 속도 flag가 아니다. 지원 device/dtype, tensor grouping, step counter placement, CUDA graph capture, autograd-through-step 여부를 바꿀 수 있다. 옵션을 켰는데 실제 dispatch가 fallback했는지 profiler/source로 확인한다. 결과 parity와 state dict compatibility를 별도로 test한다.

### 2.9.3 checkpoint와 resume를 update 경계에서 닫는다

**checkpoint는 weight 파일이 아니다.** 정확한 재개에는 model parameters와 buffers, optimizer state, scheduler state, AMP scaler, random number generators, sampler/data cursor, gradient accumulation state, parameter alias, framework/config/source revisions가 필요하다. 어떤 항목이 빠지면 “가중치에서 계속 학습”은 가능해도 uninterrupted trajectory의 재현은 주장할 수 없다.

Python, NumPy, CPU torch, 각 CUDA device의 RNG를 구분한다. model parallel rank와 data parallel rank가 서로 다른 stream을 소비할 수 있다. dropout, data augmentation, random masking, sequence packing shuffle가 어느 generator를 쓰는지 owner를 기록한다. worker process의 RNG와 prefetch queue까지 exact resume에 포함할지 정책을 정한다.

sampler cursor는 epoch와 batch index만으로 충분하지 않을 수 있다. dynamic packing, filter, weighted mixture, streaming dataset은 이미 소비한 document/sample IDs와 shard offsets, buffer state가 필요하다. data loader가 미리 가져온 batches 중 checkpoint commit 시점에 아직 학습되지 않은 것을 어떻게 처리하는지 정의한다. duplicate와 skip을 모두 consumption ledger로 탐지한다.

optimizer state는 parameter 이름 대신 순서나 object mapping으로 serialize될 수 있다. model wrapping, group 재정렬, freeze 정책 변경 뒤 load하면 moments가 잘못된 parameter에 붙을 위험이 있다. canonical parameter ID와 shape/dtype, group signature를 checkpoint manifest에 넣고 load 전 검증한다. adapter를 추가하거나 vocabulary를 resize한 migration은 신규 state 초기화 규칙을 명시한다.

**저장 protocol을 두 단계 commit으로 만든다.** 각 rank가 자기 shard를 임시 위치에 쓰고 checksum과 size를 보고한다. coordinator는 모든 expected shards와 metadata가 준비되었을 때 manifest를 commit한다. durable marker가 없는 directory는 불완전 checkpoint로 간주한다. object storage의 rename·listing semantics를 로컬 filesystem과 같다고 가정하지 않는다.

manifest에는 world topology, shard ownership, format/schema version, parent CheckpointID, committed UpdateID, next sample cursor를 둔다. parameter shard와 optimizer shard가 서로 다른 UpdateID에서 나온 조합을 거부한다. save 도중 학습이 계속되면 copy-on-write 또는 state snapshot consistency를 보장하는지 확인한다.

load는 fail-closed다. checksum, expected shard count, tensor metadata, parameter mapping, alias, source compatibility를 검사한 뒤 model을 READY로 전이한다. 일부 optimizer state가 없다고 조용히 0 초기화하지 않는다. warm-start가 의도라면 exact resume과 다른 RunRevision을 만들고 missing/new state 목록을 남긴다.

**resume parity test를 세 단계로 한다.** 첫째, save 직후 load한 state dict를 값과 metadata로 비교한다. 둘째, 동일한 다음 GoldenBatch에서 logits, loss numerator/count, unscaled gradient를 비교한다. 셋째, 첫 세 parameter delta와 optimizer moments, scheduler/scaler counters를 비교한다. 첫 step만 같고 둘째부터 달라지면 RNG 또는 state counter 결손일 가능성이 있다.

distributed topology가 바뀌는 elastic resume은 bitwise 재현과 별도 목표다. global batch와 sample order, optimizer state resharding을 보존해 수학적 equivalence를 노릴 수 있지만 reduction order와 RNG partition이 달라질 수 있다. 무엇을 보존하고 어떤 tolerance를 허용하는지 명시한다. world size 변경을 단순 resume이라고 부르지 않는다.

### 2.9.4 장애가 update 중간에 발생했을 때의 복구 표

**forward 전 실패.** parameter와 optimizer state는 아직 바뀌지 않았다. batch fetch가 실패했다면 sample cursor와 retry semantics를 본다. transient I/O를 같은 SampleID로 재시도할 수 있지만 nondeterministic transform이 다시 실행되면 input checksum이 달라질 수 있다. poison sample을 skip하면 dataset objective가 바뀌므로 reason과 ID를 기록한다.

**backward 중 실패.** 일부 gradients와 reducer buckets가 채워졌을 수 있다. 같은 process에서 buffer 일부만 지우고 계속하는 것보다 모든 ranks가 accumulation window를 폐기하고 마지막 clean boundary에서 재시도하는 편이 명료하다. OOM 뒤 CUDA allocator와 stream 상태, communication work가 안전한지 확인한다. rank 하나만 batch size를 줄이면 collective shape와 objective가 갈라진다.

**unscale·clip 중 실패.** gradients가 scaled와 unscaled 상태로 섞일 수 있다. optimizer별 stage를 원장에 두고 재호출 가능 여부를 source로 확인한다. 안전하지 않으면 window를 폐기한다. norm collective timeout이면 모든 ranks가 같은 stage였는지 trace한다. clip coefficient를 계산했지만 적용 중 실패한 partial mutation도 고려한다.

**optimizer step 중 실패.** parameter와 moments가 부분적으로 쓰였을 가능성을 배제할 수 없다. process memory를 계속 사용하지 않고 마지막 durable checkpoint로 rollback한다. fused kernel launch가 비동기라 예외 시점과 실제 실패 연산이 다를 수 있으므로 synchronization과 device error 상태를 본다.

**checkpoint 중 실패.** training update commit과 durable checkpoint commit을 구분한다. 비동기 checkpoint writer가 실패해도 in-memory training은 전진할 수 있지만 recovery point objective가 오래될 수 있다. checkpoint age와 failure alert를 둔다. incomplete artifacts는 garbage collection하되 manifest가 가리키는 committed shards를 지우지 않는다.

**rank failure와 elastic replacement.** membership이 바뀌면 process group, sampler partition, model/optimizer shards를 재구성한다. replacement rank가 동일 CheckpointID와 RunRevision을 load했는지 collective handshake한다. old group의 outstanding work와 new group을 섞지 않는다. recovery 뒤 selected parameter checksum과 next SampleIDs를 비교한다.

장애 표에는 `state_before`, `possible_partial_writes`, `safe_retry_boundary`, `states_to_restore`, `evidence_to_collect`가 있다. “재시작하면 된다”는 문장 대신 어떤 mutation이 완료되었는지 알 수 없으므로 어디까지 되돌아가는지 쓴다. 이 표가 실제 on-call playbook과 연결되어야 한다.

**autograd memory를 생애 주기로 계산한다**

**메모리 항목을 분리한다.** parameters, gradients, optimizer states, persistent buffers, saved activations, temporary workspaces, communication buckets, allocator fragmentation을 각각 센다. mixed precision에서는 low-precision parameter와 FP32 master copy가 함께 있을 수 있다. Adam moments 두 개가 FP32이면 parameter보다 optimizer state가 더 클 수 있다.

activation은 batch, sequence, hidden, layer에 따라 증가하고 attention score materialization 여부에 따라 크게 달라진다. FlashAttention 같은 fused path는 full score matrix를 저장하지 않고 row statistics와 Q/K/V를 이용해 backward에서 재계산한다. activation checkpointing은 layer 구간의 내부 saved tensors를 줄이는 대신 forward FLOPs를 추가한다.

peak memory는 단순 합보다 lifetime overlap으로 결정된다. backward 초반에는 일부 forward activations와 새 gradients, communication buckets가 겹친다. optimizer step에서 foreach temporary tensor가 추가될 수 있다. checkpoint serialization이 device copy를 만들면 peak가 달라진다. memory snapshot을 state machine 단계와 연결한다.

`zero_grad(set_to_none=True)`는 gradient storage를 allocator에 반환하거나 재사용 가능하게 하고 다음 backward가 필요 시 할당한다. zero fill 방식은 storage를 유지한다. 성능과 memory fragmentation, optimizer의 `None` 처리 차이를 작은 fixture로 본다. gradient accumulation 중간에는 당연히 zero하면 안 된다.

saved-tensor hook로 CPU offload나 compression을 할 수 있지만 transfer synchronization과 numerical change를 만든다. 어떤 tensor를 offload하고 prefetch하는지, pinned memory와 PCIe/NVLink 경로, backward stall을 계측한다. correctness reference와 gradient parity를 먼저 확인한다.

**OOM을 용량 부족 하나로 설명하지 않는다.** sequence length outlier, allocator fragmentation, graph retention, logging hook, accidental `retain_graph=True`, Python list에 loss tensor 보존, compile workspace, checkpoint overlap이 원인일 수 있다. allocated/reserved/active bytes와 tensor lifetime을 본다. batch size를 무조건 줄이기 전에 최초 증가 UpdateID와 object owner를 찾는다.

OOM recovery로 microbatch를 동적으로 줄이면 accumulation count와 denominator, scheduler, sample grouping이 바뀐다. 같은 global batch를 유지하려면 microsteps를 늘리고 count normalization을 맞춘다. 실패 batch를 재포장하면 RNG와 dropout 차이가 생길 수 있다. adaptive batch 정책을 RunRevision과 manifest에 기록한다.

## 2.10 DDP·AMP·compile·adapter의 결합 경계

DDP와 AMP의 순서가 맞더라도 compile, CUDA Graph, fused optimizer가 mutation과 capture 제약을 추가한다. adapter처럼 trainable subset이 바뀌면 reducer bucket과 optimizer state도 달라지므로 option 변경을 source caller와 runtime state까지 추적한다.

### 2.10.1 DDP와 AMP가 만나는 순서를 작은 식으로 검산한다

**두 rank, 두 microstep fixture.** rank r, microstep k의 unscaled gradient numerator를 `g_rk`라 하고 global valid count를 N이라 하자. loss scale s를 곱해 local buffer에는 `sΣ_k g_rk/N`에 해당하는 값이 쌓인다. 마지막 backward에서 DDP가 rank 평균을 한다면 world-size factor를 어떻게 보정했는지 포함해야 원하는 `Σ_rk g_rk/N`이 된다.

DDP reduction 뒤 unscale하면 모든 ranks가 같은 averaged scaled gradient를 `s`로 나눈다. 일부 rank가 다른 scaler scale을 갖고 있으면 reduction 전에 단위가 섞인다. scaler state를 rank 간 일치시키거나 found-inf/scale policy가 collective로 조정되는지 확인한다. rank-local scale divergence assertion을 둔다.

unscale 뒤 global norm clipping을 한다. DDP replicated parameters라면 각 rank가 같은 gradient와 norm을 가져 같은 coefficient를 적용한다. sharded gradients라면 squared norm collective가 필요하다. 그 뒤 scaler/optimizer step의 skip 결정을 모든 rank에서 맞춘다. step이 실행되면 zeroing과 UpdateID commit으로 간다.

fixture는 rank 0에 큰 gradient와 count 7, rank 1에 작은 gradient와 count 1을 둔다. scale s=128, clip threshold c를 손으로 계산 가능한 값으로 정한다. 올바른 delta와 다음 scaler state를 reference로 저장한다. local mean, clip-before-unscale, rank-local skip, double accumulation division을 각각 주입한다.

**왜 순서가 바뀌면 다른 함수인가.** clip은 비선형이다. `clip(mean(g_r))`와 `mean(clip(g_r))`는 일반적으로 다르다. unscale은 선형이지만 scale이 rank마다 다르면 reduction과 교환되지 않는다. overflow skip은 조건 분기여서 일부 rank만 실행하면 state가 갈라진다. zeroing은 이전 기여를 파괴하므로 commit 전 호출하면 재시도가 불가능하다.

이 식을 코드 review 표로 옮긴다. 각 연산에 input unit, owner, collective 여부, mutation, retryability를 적는다. `backward`라는 한 줄 아래에서 reducer hook이 실행되고, `scaler.step` 내부에서 unscale 여부 확인과 optimizer 호출이 조건부로 일어날 수 있으므로 표면 호출만 읽지 않는다.

### 2.10.2 compile·CUDA graph·fused optimizer의 상태 제약

**graph capture는 동적 상태를 정적으로 만들려 한다.** CUDA graph는 반복 launch overhead를 줄이지만 capture 동안 memory address, control flow, 일부 scalar와 collective 조건이 고정되어야 한다. GradScaler scale, found-inf, optimizer step counter를 device tensor로 두는 `capturable` 경로가 필요한 이유다. host scalar `.item()`은 capture를 깨거나 synchronization을 만든다.

dynamic sequence shape, zero-valid branch, overflow skip, variable accumulation remainder가 graph specialization과 어떻게 상호작용하는지 본다. 여러 graph를 shape bucket별로 보유할 수 있지만 state mapping과 memory 비용이 생긴다. eager path와 captured path의 GoldenBatch delta를 비교한다.

`torch.compile`은 forward/backward graph를 분할하거나 optimizer step을 compile할 수 있다. graph break 위치와 recompilation count를 계측한다. Python callback이나 gradient hook이 specialization을 유발할 수 있다. compiled path가 custom autograd backward나 communication hook을 우회하지 않는지 source/trace로 확인한다.

fused optimizer는 parameter groups를 device kernel로 묶어 launch를 줄이고 vectorized update를 수행한다. 지원하지 않는 dtype, sparse gradient, differentiable/capturable 옵션에서 fallback할 수 있다. 일부 groups만 fused되고 나머지가 foreach/single tensor이면 update 순서와 error handling을 확인한다.

성능 검증에는 step latency뿐 아니라 end-to-end tokens/sec, peak memory, compilation warmup, graph cache size를 넣는다. correctness에는 first two deltas와 moments, skip behavior, checkpoint round trip을 넣는다. 빠른 경로와 reference 경로가 같은 source option을 공유해 같은 버그를 가질 수 있으므로 독립 hand reference를 유지한다.

### 2.10.3 trainable subset과 adapter가 update state를 바꾼다

**freeze는 `requires_grad=False` 한 줄보다 넓다.** frozen parameter가 optimizer group에서 제거되는지, gradient가 materialize되지 않는지, weight decay가 적용되지 않는지 확인한다. BatchNorm 같은 running buffers나 dropout mode는 parameter freeze와 별개다. LLM의 normalization과 adapters에서도 train/eval mode와 buffer mutation을 확인한다.

LoRA는 base weight를 고정하고 low-rank factors의 곱을 forward에 더한다. backward는 A와 B로 흐르고 base W에는 흐르지 않는다. initialization에서 한 factor를 0으로 두면 첫 step에 다른 factor gradient가 0일 수 있다. 이를 dead gradient bug로 오인하지 않고 식으로 예상한다. adapter scaling과 dropout은 gradient magnitude와 RNG state를 바꾼다.

여러 adapters를 동시에 학습하면 parameter names, groups, learning rates, checkpoint selection을 명확히 한다. merge/unmerge는 forward weight와 optimizer state 의미를 바꾼다. 학습 중 merge를 반복해 rounding drift가 생기지 않는지 본다. checkpoint에는 base model revision과 adapter config·weights가 함께 식별되어야 한다.

quantized base와 adapter 학습에서는 dequantized compute path, quantization scale, low-precision gradient, optimizer state dtype을 구분한다. base quantization buffer가 mutation되는지 확인한다. paged optimizer나 CPU offload는 state ownership과 synchronization을 바꾼다. “4-bit 학습”이 모든 tensors가 4-bit라는 뜻은 아니다.

trainable subset이 작으면 global grad norm에서 frozen base를 제외하고 adapters만 센다. clipping threshold를 full fine-tuning과 같은 값으로 그대로 쓰는 것이 타당한지 distribution을 측정한다. DDP unused parameter와 adapter conditional routing도 bucket behavior를 바꿀 수 있다.

### 2.10.4 update 관측에서 흔한 거짓 신호를 제거한다

**gradient norm 하나는 방향을 말하지 않는다.** 같은 norm의 직교 vectors가 있다. raw·clipped gradient norm, parameter delta norm, gradient–delta cosine, update/weight ratio를 selected layers에서 본다. Adam과 weight decay 때문에 delta가 raw gradient와 평행하지 않을 수 있다. 그 차이를 오류가 아니라 optimizer 식과 비교한다.

평균 norm은 한 layer의 폭발을 숨긴다. depth·parameter type·group bucket별 p50/p95/max를 본다. 그러나 tensor name을 metric label로 모두 내보내지 않는다. 사건 artifact에 상세 row를 저장한다. tied parameter는 한 번만 센다.

loss scale이 증가하는 것을 학습이 안정됐다는 뜻으로 보지 않는다. clip coefficient가 1인 것도 좋은 gradient라는 보장이 없다. zero gradient 비율은 sparse activation과 frozen parameters 때문에 높을 수 있다. 모든 metric은 expected state와 대조한다.

step time 증가가 optimizer 문제처럼 보여도 gradient all-reduce, checkpoint I/O, allocator GC가 겹쳤을 수 있다. forward, backward compute, communication wait, unscale/clip, optimizer, zeroing, checkpoint enqueue를 trace span으로 나눈다. CUDA event와 host clock의 의미를 구분한다.

rank 0만 관측하면 다른 rank의 overflow, count imbalance, straggler를 놓친다. 매 step에는 rank aggregate min/max와 disagreement flag를 낮은 cardinality로 남긴다. 상세 per-rank trace는 alert의 correlation ID로 조회한다. 모든 ranks가 동시에 많은 로그를 쓰며 I/O 폭주를 만들지 않게 sampling한다.

**소스 코드 갱신을 안전하게 검토하는 방법**

**세 snapshot의 역할을 혼동하지 않는다.** micrograd commit은 chain rule의 교육 reference, nanoGPT commit은 작은 training loop의 구체적 순서, PyTorch commit은 production AMP/autograd API 구현 근거다. 한 repository의 test가 다른 stack 조합을 보증하지 않는다. 책의 local integration fixture가 세 경계를 연결한다.

PyTorch upgrade에서는 autograd, AMP scaler, optimizer, DDP reducer의 changed symbols를 찾는다. public release note만 읽지 않고 source diff와 tests를 본다. default dtype/foreach/fused 선택, state dict schema, reduction semantics, warning-to-error 변화가 있는지 확인한다. CUDA·NCCL 조합의 지원 범위는 공식 compatibility 문서와 build metadata로 별도 검증한다.

Transformers Trainer upgrade는 loss denominator와 accumulation scale을 바꿀 수 있고, Accelerate/DeepSpeed/FSDP wrapper가 다시 감쌀 수 있다. 호출 stack을 실제 configuration별로 복원한다. 동일 옵션명이 서로 다른 owner에서 중복 적용되지 않는지 본다. source grep 결과를 runtime dispatch 증거로 과장하지 않는다.

upgrade canary는 첫 두 updates를 old/new environment에서 비교한다. input manifest, logits, normalized loss, unscaled gradient projection, clip coefficient, delta, moments, scaler/scheduler state를 본다. 예상 numerical tolerance와 첫 difference boundary를 사전에 정한다. throughput 측정은 correctness canary 뒤에 한다.

rollback에는 binary/package만 되돌리는 경우와 checkpoint까지 되돌리는 경우가 있다. 새 optimizer semantics로 이미 update했다면 package rollback만으로 old trajectory에 돌아가지 않는다. canary가 production checkpoint를 오염시키지 않도록 child RunRevision에서 실행한다.

## 2.11 한 update를 수치·시간·원자성으로 재구성한다

앞선 계약을 하나의 GoldenUpdateRun에 모은다. loss seed에서 graph traversal, reduction, unscale·finite·clip, optimizer effect까지 수치 좌표와 사건 시간을 함께 기록하면 성능 최적화가 정확성을 침범한 최초 지점을 찾을 수 있다.

### 2.11.1 한 update를 수치·시간·원자성으로 재구성한다

**한 parameter의 두 update를 손으로 재구성한다**

**SGD에서 상태의 필요성을 본다.** scalar parameter `θ=2`, gradient `g_1=3`, learning rate `η=0.1`인 plain SGD라면 첫 update 뒤 `θ=1.7`이다. momentum `β=0.9`를 추가하고 buffer 초기값을 0으로 두면 구현 정의에 따라 `v_1=3`, `θ_1=1.7`이 된다. 둘째 gradient가 -1이면 `v_2=0.9·3-1=1.7`, `θ_2=1.53`이다. 현재 gradient가 음수여도 누적 momentum 때문에 parameter가 같은 방향으로 더 움직인다.

Nesterov와 dampening, maximize 옵션은 식을 바꾼다. 이름만 보고 hand reference를 쓰지 않는다. framework source가 buffer를 처음 만들 때 dampening을 적용하는지, update에 `g+βv`를 쓰는지 확인한다. weight decay가 gradient에 결합되는 L2 방식인지 parameter에 분리 적용되는 decoupled 방식인지도 구분한다.

**Adam의 첫 두 moments를 계산한다.** `m_t=β_1m_{t-1}+(1-β_1)g_t`, `v_t=β_2v_{t-1}+(1-β_2)g_t²`다. zero initialization의 초기 편향을 보정하려면 `m̂_t=m_t/(1-β_1^t)`, `v̂_t=v_t/(1-β_2^t)`를 쓴다. update는 흔히 `θ_t=θ_{t-1}-η m̂_t/(sqrt(v̂_t)+ε)` 꼴이다.

epsilon이 square root 바깥인지 안인지, step size 계산에 bias correction을 합치는지에 따라 저차원 수치가 달라질 수 있다. 실수 대수에서 등가인 재배열도 low precision에서는 rounding이 다르다. fused kernel의 hand reference는 해당 revision의 정확한 식을 구현하되 FP64로 계산한다.

AdamW의 decoupled decay는 adaptive gradient update와 별도로 `θ←θ-ηλθ` 또는 등가 multiplicative form을 적용한다. gradient에 `λθ`를 더해 second moment에 넣는 L2 regularization과 다르다. bias와 normalization scale을 decay에서 제외하는 recipe는 parameter grouping 단계의 정책이다. name pattern이 새 model architecture에서 잘못 매칭되지 않는지 parameter type과 shape로 감사한다.

첫 두 updates의 ledger에는 `θ_before,g_raw,g_after_clip,m_before,v_before,step,lr,decay,θ_after`가 있다. 모든 parameter를 저장하지 않고 selected elements와 random projections를 쓴다. 그러나 처음 구현을 검증할 때는 아주 작은 vector 전체를 FP64 JSON fixture로 보존한다. optimizer upgrade 때 golden expected를 자동 재생성하지 않는다.

**gradient accumulation과 Adam의 비선형성을 구분한다.** microbatch마다 Adam step을 두 번 하는 것과 gradients를 평균해 한 번 step하는 것은 같지 않다. moments와 bias-correction time이 두 번 전진하고 parameter가 중간에 바뀐다. accumulation은 optimizer update 횟수를 줄이는 것이므로 scheduler와 weight decay 적용 횟수도 달라진다.

global batch를 두 배로 늘리고 accumulation steps를 두 배로 늘려 sample/update가 같아졌는지 계산한다. data-parallel world size, per-device microbatch, accumulation, sequence valid ratio를 모두 곱해야 한다. “effective batch size”는 sequences인지 input tokens인지 valid targets인지 명시한다.

**scheduler의 시간은 무엇을 세는가**

**step 단위부터 고정한다.** scheduler는 optimizer updates, attempted updates, consumed tokens, wall-clock 가운데 하나를 시간으로 삼을 수 있다. 대부분의 framework recipe는 optimizer step을 세지만 overflow skip과 gradient accumulation remainder에서 의미가 흔들릴 수 있다. scheduler 호출 위치와 skip 조건을 source로 확인한다.

warmup steps를 고정하면 global batch가 바뀔 때 warmup 동안 본 tokens가 달라진다. warmup ratio는 total planned updates가 정확해야 한다. streaming이나 early stop, elastic world size에서는 planned total이 바뀔 수 있다. token-based scheduler는 valid count 원장을 이용하지만 구현과 checkpoint state가 더 복잡하다.

cosine, linear, constant-with-warmup, inverse-sqrt는 같은 peak LR에서도 다른 integrated step size를 만든다. 그래프만 보여주지 않고 `lr_t`를 계산하는 함수와 boundary `t=0,warmup_end,total_end`를 fixture로 둔다. off-by-one은 첫 update 전에 scheduler를 호출하는지 뒤에 호출하는지에서 생긴다.

optimizer state의 step counter와 scheduler counter가 resume 뒤 일치하는지 본다. parameter group별 LR이 있으면 base schedule에 multiplier를 적용하는지 각 group scheduler가 별도 state를 갖는지 확인한다. adapter와 base group을 동결/해제하는 curriculum에서는 새 group의 warmup 기준을 정한다.

overflow skip에서 scheduler가 전진하면 실제 parameter update 없이 LR 시간이 흐른다. 간헐적 한 번은 작아 보여도 불안정 구간에서 skip streak가 길면 warmup을 소모할 수 있다. `attempted_step`, `committed_update`, `consumed_valid_targets`, `scheduler_step`을 각각 로그해 정책과 맞는지 본다.

**gradient noise와 batch equivalence를 과장 없이 측정한다**

**큰 batch parity는 deterministic 조건의 국소 검사다.** 동일 samples, 동일 parameter, dropout off, 같은 numerator/count에서 concatenated batch와 accumulated microbatches의 gradient가 가까워야 한다. floating reduction 순서로 작은 차이는 있을 수 있다. 이 fixture가 통과한다고 dropout on과 BatchNorm, data-dependent routing까지 exact equivalence인 것은 아니다.

dropout mask는 batch shape와 call order에 따라 RNG 소비가 달라진다. 큰 batch 한 번과 작은 batch K번이 같은 random numbers를 각 sample에 배정하지 않을 수 있다. sample-keyed RNG를 설계하거나 statistical equivalence만 목표로 할 수 있다. 어떤 기준을 쓰는지 test metadata에 적는다.

gradient noise scale을 논할 때도 raw batch variance, parameter projection, optimizer-preconditioned update를 구분한다. token correlation 때문에 valid token 수가 독립 sample 수와 같지 않다. 같은 document의 인접 tokens와 duplicated examples는 강하게 상관된다. batch size scaling rule을 경험 법칙으로 제시할 때 dataset과 optimizer 조건을 함께 둔다.

batch가 커지면 communication efficiency가 좋아질 수 있지만 update frequency가 줄고 memory가 늘어난다. LR linear scaling은 특정 regime의 heuristic이지 보존 법칙이 아니다. warmup과 optimizer, gradient clipping, data order를 함께 조정하고 validation·compute efficiency로 검증한다.

**multi-optimizer와 복합 update의 원자성**

**optimizer가 둘이면 skip도 둘이다.** generator/discriminator, actor/critic, multimodal towers, separate adapter groups는 optimizer를 여러 개 쓸 수 있다. 각 optimizer가 다른 loss와 accumulation 주기를 가지면 UpdateID hierarchy를 만든다. 한 optimizer가 step하고 다른 하나가 overflow로 skip했을 때 허용되는지 algorithm이 결정한다.

shared parameter가 두 optimizers에 동시에 등록되면 두 번 update될 수 있다. 의도한 alternating method가 아니라면 금지한다. parameter ownership assertion을 둔다. shared forward graph에서 첫 backward 뒤 graph가 해제되어 둘째 loss backward가 실패하거나 `retain_graph`로 memory가 폭증할 수 있다. losses를 합쳐 한 backward할지 순차 backward할지 식으로 결정한다.

각 optimizer의 GradScaler를 공유할지 별도 사용할지도 중요하다. shared scale은 한 loss의 overflow가 다른 optimizer를 skip시킬 수 있다. 별도 scale은 shared gradients의 단위를 섞지 않게 경계를 설계해야 한다. unscale과 clipping, zeroing을 optimizer owner별로 추적한다.

alternating update에서 상대 model parameter가 중간에 바뀌므로 two losses를 같은 snapshot에서 계산하는지 순차 snapshot에서 계산하는지 다르다. checkpoint는 phase와 next optimizer owner를 저장한다. resume가 cycle 중간을 처음부터 반복하지 않게 한다.

**distributed failure를 네트워크와 control flow로 나눈다**

**timeout은 결과다.** 한 rank가 OOM, dataloader exception, assertion, 긴 compilation에 걸리면 다른 ranks는 다음 collective에서 기다리다 timeout된다. NCCL 오류를 네트워크 원인으로 단정하기 전에 rank별 마지막 state와 stack, collective sequence를 맞춘다.

collective fingerprint에는 operation type, process group, sequence number, tensor shape/dtype/device, caller state를 넣는다. 민감한 tensor 값은 필요 없다. rank 간 fingerprint가 다르면 control-flow 또는 shape divergence다. 모두 같고 일부 transport만 멈추면 topology, link, driver, NIC를 조사한다.

asynchronous collective work를 launch하고 기다리기 전에 input storage를 mutation하면 race가 생길 수 있다. CUDA stream dependency와 work handle lifetime을 확인한다. communication overlap에서 compute stream이 reduced gradient를 너무 일찍 읽지 않게 event를 둔다. debug mode에서 synchronization을 강화해 race가 사라지는지 비교한다.

straggler는 hang과 다르다. sequence length, token routing imbalance, host I/O, thermal throttling, ECC recovery, network retransmission을 rank별 trace로 나눈다. slowest-rank time이 전체 step을 결정한다. 평균 GPU utilization은 tail latency를 숨긴다.

elastic restart는 장애 rank만 바꾸는 기능처럼 보여도 global sample assignment와 optimizer shard ownership을 바꾼다. membership change를 새로운 topology revision으로 기록하고 checkpoint barrier에서 재구성한다. partial current update를 버릴지 재생할지 모든 ranks가 동의한다.

**numerical drift의 허용 범위를 설계한다**

**bitwise와 functional equivalence를 구분한다.** deterministic algorithms와 동일 hardware/software에서도 일부 parallel reduction은 order에 따라 bitwise 차이가 날 수 있다. 목표가 정확한 debugging replay인지 통계적 training equivalence인지 정한다. exact requirement를 지원하지 않는 backend에 거짓 보장을 쓰지 않는다.

tensor 비교는 `atol+rtol·|reference|`만으로 끝내지 않는다. near-zero 영역의 absolute error, 큰 값의 relative error, cosine, norm ratio, non-finite pattern을 본다. layer depth에 따라 작은 차이가 증폭될 수 있으므로 first-difference와 growth curve를 기록한다.

parameter delta는 weight 자체보다 작은 경우가 많아 post-weight 비교가 update 차이를 숨긴다. `θ_after-θ_before`를 직접 비교한다. Adam moments와 scaler state처럼 미래에 영향을 주는 hidden state도 비교한다. 첫 step에서 가까워도 state가 다르면 이후 divergence가 커진다.

stochastic training run 두 개의 final metric만 비교해 code parity를 주장하지 않는다. GoldenBatch deterministic fixture, short controlled trajectory, full statistical experiment를 계층화한다. 각 층이 증명하는 범위를 표시한다.

**update 상태를 Prometheus 지표로 번역한다**

**counter와 gauge를 구분한다.** committed updates, skipped updates, consumed valid targets, non-finite events는 monotonic counter가 적합하다. current loss scale, LR, grad norm, clip coefficient, queue depth는 gauge다. restart에서 counter reset을 RunID와 구분한다. UpdateID를 metric label로 넣어 cardinality를 폭발시키지 않고 exemplar나 trace link로 연결한다.

histogram에는 step duration, backward duration, collective wait, checkpoint duration을 둔다. bucket은 실제 SLO와 분포에 맞춘다. layer name, parameter name, sample ID를 label로 쓰지 않는다. 상세 진단은 artifact store와 trace에 둔다.

alerts는 단일 threshold보다 상태 조합을 본다. `nonfinite_updates_total` 증가와 loss scale 하강, skip streak를 결합한다. grad norm 상승과 clip coefficient 하강이 지속되는지 본다. committed UpdateID 정지인데 heartbeat만 살아 있으면 progress stall이다. rank disagreement flag는 즉시 높은 severity로 다룬다.

metric 수집 자체가 `.item()` synchronization을 만들 수 있다. 이미 계산된 scalar를 재사용하고 sampling 주기를 둔다. global norm을 logging만 위해 추가 collective하지 않는다. 성능 회귀가 발생하면 관측을 끄는 대신 비용 계층과 aggregation 위치를 고친다.

**2장의 통합 실습**

**실습 1: scalar graph에서 alias까지.** micrograd snapshot으로 한 leaf가 세 branch에 쓰이는 graph를 만들고 손미분과 비교한다. 한 closure의 `+=`를 대입으로 바꾼 failure를 주입한다. 이어 PyTorch tensor에서 tied parameter 두 사용 경로의 gradient 합과 storage identity를 확인한다.

**실습 2: accumulation denominator.** valid targets가 1,3,8인 microbatches를 만든다. FP64 concatenated reference, microbatch-mean 평균, numerator/global-count 방식을 비교한다. gradient projection과 첫 SGD·AdamW delta를 저장한다. 마지막 remainder window도 넣는다.

**실습 3: AMP 순서.** scale 128을 고정한 toy gradient에서 unscale→clip과 clip→unscale을 비교한다. inf 하나를 주입해 optimizer parameter와 moments, scheduler, UpdateID가 정책대로 유지되는지 본다. 두 logical ranks 중 하나만 inf인 상황에서 공동 skip을 확인한다.

**실습 4: DDP sync.** dropout off인 작은 model과 unequal target counts로 single-process reference를 만든다. intermediate no-sync와 마지막 sync를 사용한 결과를 비교한다. 마지막 sync를 의도적으로 끄고 parameter projection disagreement detector가 울리는지 본다.

**실습 5: crash/resume.** 두 updates 뒤 checkpoint하고 세 updates를 더 진행한다. 별도 process가 checkpoint를 load해 같은 samples로 세 updates를 수행한다. logits, loss, gradient, delta, moments, LR, scale, sample IDs를 step별 비교한다. scaler나 sampler state를 하나씩 누락시켜 최초 차이를 기록한다.

**실습 6: source 영향 반경.** nanoGPT의 unscale/clip 순서를 reference로 읽고 local fixture에서 순서를 바꾼다. PyTorch GradScaler snapshot의 `unscale_`와 `step` stage 제약을 source 좌표로 연결한다. source가 증명하는 것과 fixture가 추가로 증명하는 것을 표에 나눈다.

실습 결과에는 PASS 한 줄이 아니라 input manifest, expected invariant, observed first difference, source coordinates, environment metadata를 담는다. 실행하지 않은 CUDA/distributed case는 `NOT_RUN`으로 남기고 정적 분석을 실행 증거로 둔갑시키지 않는다.

**3장으로 넘기는 update dossier**

**한 update를 재생할 수 있어야 한다.** dossier는 1장의 loss manifest를 포함하고 microstep별 scalar와 count, autograd graph probe, raw/scaled/unscaled gradient projection, reducer state, global norm, clip coefficient를 덧붙인다. optimizer에서는 groups, options, moments before/after, applied LR, selected parameter delta가 있다.

상태 전이에는 `ACCUMULATING→READY→UNSCALED→CLIPPED→COMMITTED` 또는 정의된 skip 경로가 기록된다. 각 event의 owner와 timestamp보다 UpdateID ordering을 우선한다. checkpoint manifest는 committed UpdateID와 next sample cursor를 가리킨다. 모든 ranks의 agreement evidence가 있다.

source evidence는 micrograd `Value.backward`, nanoGPT accumulation/unscale/clip/step/zero 순서, PyTorch `GradScaler.unscale_`와 `step`을 고정 commits로 가리킨다. production stack의 wrapper와 optimizer symbol은 실제 선택한 recipe에서 추가한다. 함수 하나가 전체 동작을 보증한다고 말하지 않는다.

완료 판정은 세 질문이다. 같은 global objective가 world size와 accumulation partition이 달라도 같은 gradient를 만드는가. overflow·예외·rank failure에서 partial update를 commit하지 않고 정해진 경계로 복구하는가. checkpoint에서 돌아온 첫 세 updates가 허용한 재현 수준에서 uninterrupted run과 맞는가.

셋 중 하나라도 증거가 없으면 optimizer가 “돌아간다”는 사실만 확인한 것이다. 세 질문이 닫히면 3장은 작은 GPT 전체를 실행하며 data cursor에서 checkpoint까지 이어진 상태기계를 한 run으로 검증할 수 있다.

**한 번의 update가 실패하는 여섯 가지 서로 다른 방식**

**목적함수는 맞지만 graph가 끊긴다.** loss scalar 값은 reference와 같은데 parameter 일부가 detach되어 gradient를 받지 못할 수 있다. frozen 설정, `.data` 사용, 새 tensor 생성, custom backward 누락, alias 파괴를 본다. expected-trainable parameter set과 actual `grad is not None`, selected directional derivative를 비교한다. norm 합만 보면 작은 adapter 하나의 누락을 놓친다.

**gradient는 맞지만 단위가 틀린다.** loss scale이 아직 곱해진 gradient를 clip하거나 accumulation division이 두 번 적용될 수 있다. direction cosine은 1이라도 magnitude와 delta가 다르다. raw loss numerator/count, scaled loss, unscaled norm을 함께 보존해야 한다. learning rate를 임시로 조정해 증상을 숨기지 않는다.

**전역 gradient가 아니다.** no-sync가 마지막 microstep까지 남거나 unequal rank denominator를 local mean으로 평균할 수 있다. rank 0 curve는 정상처럼 보인다. single-process concatenated reference와 rank별 parameter projection을 비교한다. collective trace에서 expected bucket 수와 순서를 확인한다.

**optimizer가 다른 parameter를 갱신한다.** parameter group mapping이 checkpoint load 뒤 어긋나거나 tied weight가 두 groups에 중복될 수 있다. names, canonical IDs, shapes만으로도 일부 오류를 찾지만 동일 shape parameters는 checksum과 mapping manifest가 필요하다. selected moment가 예상 parameter의 gradient history를 따르는지 본다.

**step은 건너뛰었는데 시간이 전진한다.** AMP overflow로 optimizer가 skip됐지만 scheduler, EMA, UpdateID, callback, data cursor 중 일부가 전진할 수 있다. 이것이 의도한 정책인지 명시한다. 각 counter를 따로 관측하고 공동 commit record로 묶는다. loss scale 하강만 보고 update가 없었다고 추정하지 않는다.

**parameter는 맞지만 재개 상태가 다르다.** weight checksum은 같아도 optimizer moments, scaler tracker, RNG, sampler cursor가 다르면 다음 update부터 갈라진다. checkpoint smoke test가 forward만 비교하면 놓친다. 최소 세 update의 trajectory parity를 요구하는 이유다.

이 여섯 실패는 표면적으로 모두 “loss가 이상하다”거나 “재현이 안 된다”로 나타날 수 있다. 조사 순서는 loss manifest, graph reachability, gradient unit, collective, optimizer mapping, commit/resume state다. 앞 경계가 같다는 증거가 있을 때만 다음 경계로 내려간다.

**옵션을 상태 변화로 번역하는 표**

**`gradient_accumulation_steps`.** update당 microstep 목표, no-sync 횟수, loss normalization, remainder policy, scheduler update 빈도를 바꾼다. sample 수만 곱해 effective batch라고 쓰지 말고 valid targets와 global world size를 계산한다. 변경 전후 첫 gradient와 delta를 비교한다.

**`max_grad_norm`.** unscaled global gradient 자체는 바꾸지 않고 clip coefficient와 이후 optimizer input을 바꿔야 한다. 0이나 None이 clipping 비활성인지 즉시 0으로 만드는지 API를 확인한다. sharded optimizer에서는 norm collective owner도 바뀔 수 있다.

**mixed-precision mode.** autocast dtype, loss scaling 사용 여부, master parameters, optimizer dispatch, kernel 선택과 numerical tolerance를 바꾼다. BF16을 선택했다고 모든 연산과 state가 BF16이 되는 것이 아니다. first-difference tensor를 기록한다.

**DDP bucket options.** bucket size와 ready order, overlap, view semantics를 바꾸며 수학적 global gradient는 유지되어야 한다. `find_unused_parameters`, static graph, communication hook은 control flow와 state를 더 바꾼다. 성능 효과와 parity를 분리한다.

**optimizer `foreach`·`fused`·`capturable`.** dispatch와 temporary memory, device step tensor, supported dtype, checkpoint representation에 영향을 줄 수 있다. 같은 식을 목표로 해도 rounding order가 다르다. 실제 선택된 path와 fallback reason을 확인한다.

**`zero_grad(set_to_none)`.** gradient buffer lifetime과 optimizer의 missing-gradient branch를 바꾼다. accumulation window 내부에서 호출 위치가 잘못되면 기여를 잃는다. weight decay가 `None`과 zero에서 같은지 source/fixture로 확인한다.

**checkpoint interval과 async save.** recovery point objective, I/O overlap, state snapshot consistency를 바꾼다. update boundary 이외 저장을 허용하면 partial accumulation state를 포함해야 한다. duration 최적화와 crash consistency를 함께 test한다.

옵션 설명의 완료 조건은 default와 권장값을 쓰는 것이 아니다. 그 옵션이 읽는 config field, 실제 branch와 symbol, 바꾸는 tensor/state, 예상 first difference, 성능·정확성 효과, checkpoint 영향, 실패 detector를 한 행으로 연결하는 것이다.

### 2.11.2 autograd graph와 분산 backward의 실행 경로를 검증한다

**update 인과를 설명하는 기준**

독자는 scalar loss에서 시작해 graph의 모든 기여가 leaf gradient에 합쳐지는 과정을 설명할 수 있어야 한다. saved tensor와 recomputation, view와 alias, in-place mutation이 backward에 미치는 영향을 구분한다. 큰 batch와 accumulation이 같은 조건과 같지 않은 조건을 식으로 말한다.

AMP에서는 autocast와 scaling을 분리하고 unscale, finite check, global clipping, optimizer step, scaler update의 순서를 복원한다. DDP에서는 gradient bucket, no-sync, reduction semantics, uneven denominator, rank agreement를 복원한다. optimizer에서는 selected parameter의 moments와 delta를 hand reference로 재계산한다.

장애가 나면 현재 state와 possible partial writes를 식별해 안전한 retry boundary를 고른다. checkpoint가 weights뿐 아니라 trajectory state를 닫는다는 것을 증명한다. resume 첫 세 updates에서 input부터 delta까지 비교한다. 실행하지 않은 경로와 증명하지 못한 범위를 정직하게 표시한다.

이 기준은 framework API를 많이 외우는 것보다 엄격하다. API는 revision에 따라 바뀌지만 gradient 기여, 단위, collective, mutation, durable commit이라는 질문은 남는다. 고정 source 좌표와 작은 oracle을 함께 보존하면 독자는 새 optimizer와 wrapper를 만나도 같은 방식으로 해부할 수 있다.

2장의 마지막 산출물은 재현 가능한 UpdateID다. 그것은 loss가 어느 samples에서 왔고, 어떤 graph와 gradient 단위를 거쳐, 어떤 ranks에서 합쳐지고, 어떤 clipping과 optimizer state로 parameter를 바꾸었으며, 어디에 durable하게 commit되었는지를 가리킨다. 이 계보가 닫히면 다음 장의 end-to-end 실행은 단순 데모가 아니라 각 계약을 한 번에 반증할 수 있는 통합 시험이 된다.

**현장에서 바로 쓰는 첫 차이 탐색표**

**loss 전까지 같은가.** 동일 GoldenBatch의 IDs, masks, positions, parameter checksum, logits probe, numerator/count를 비교한다. 하나라도 다르면 backward를 조사하기 전에 그 경계를 고친다. loss scalar만 같고 numerator/count가 다를 수 있으므로 둘을 별도로 본다.

**backward 직후 같은가.** selected logits와 early/middle/late hidden의 gradient projection을 비교한다. expected parameter에 gradient가 도달했는지, tied alias 기여가 합쳐졌는지 본다. scaled gradient라면 scale을 함께 기록하고 다른 run과 비교하기 전에 같은 단위로 환산한다.

**reduction 뒤 같은가.** microstep별 no-sync 상태, 마지막 bucket collective, rank별 valid count를 본다. single-process concatenated reference와 global gradient를 비교한다. rank 간 parameters가 step 전부터 다르면 reducer 이전의 state broadcast와 checkpoint load를 조사한다.

**clip 뒤 같은가.** unscaled global norm과 coefficient를 비교한다. norm은 같은데 gradient 방향이 다를 수 있으므로 projection도 본다. alias 중복, sharded norm, sparse parameter exclusion을 확인한다. threshold 변경의 영향이 unscaled gradient보다 앞에서 보이면 실험 통제가 깨졌다.

**step 뒤 같은가.** optimizer group, applied LR, step counter, moment projections, decay, parameter delta를 비교한다. post-weight checksum만 보지 않고 delta를 직접 본다. fused와 reference path의 dispatch를 확인한다. 일부 parameter만 다르면 group mapping과 grad-None branch를 좁힌다.

**commit 뒤 같은가.** scheduler, scaler, EMA, callback, UpdateID, next sample cursor, checkpoint parent를 비교한다. parameter가 같아도 이 상태가 다르면 다음 update에서 갈라진다. rank agreement와 durable manifest까지 확인한다.

이 표는 순차적으로 사용한다. 앞 단계의 동일성이 강한 증거로 닫히지 않았는데 뒤 단계의 옵션을 바꾸면 원인 후보가 늘어난다. 반대로 최초 차이를 찾으면 그 연산의 입력과 source branch를 작은 fixture로 축소한다. 큰 학습을 다시 돌리는 일은 마지막 검증이지 첫 조사 수단이 아니다.

**독립 검토자가 update를 승인하는 조건**

독립 검토자는 작성자가 고른 정상 예제만 보지 않는다. accumulation counts를 불균등하게 만들고, scale을 바꾸고, 한 rank에 non-finite를 넣고, checkpoint 항목 하나를 제거한다. 각각 denominator reference, 공동 skip, load validation, first-difference detector가 예상대로 실패해야 한다.

source 좌표는 로컬 checkout에서 다시 확인한다. micrograd `Value.backward`가 chain contributions를 어떻게 누적하는지, nanoGPT가 마지막 microstep sync와 unscale·clip·step·zero 순서를 어떻게 표현하는지, PyTorch GradScaler가 optimizer별 stage와 skip을 어떻게 다루는지 읽는다. 책의 문장이 source보다 넓은 보장을 하면 범위를 줄이거나 integration evidence를 추가한다.

검토자는 test 결과를 `PASS`, `FAIL`, `NOT_RUN`, `OUT_OF_SCOPE`로 구분한다. CUDA 장비가 없어 실행하지 않은 distributed fixture를 정적 source 분석만으로 PASS로 바꾸지 않는다. 반대로 대규모 모델을 실행하지 않아도 scalar·작은 tensor fixture와 source audit로 함수 계약을 깊게 검증할 수 있다.

마지막에는 임의의 UpdateID를 골라 loss contribution 하나에서 parameter delta와 checkpoint manifest까지 정방향으로 걷고, checkpoint에서 sample과 loss로 역방향으로 걷는다. edge마다 owner, revision, checksum, state transition이 있어야 한다. 설명이 끊기는 edge는 미완료 항목으로 남긴다.

이 봉인 절차를 통과하면 2장은 optimizer 사용법 목록이 아니다. 수학적 미분이 실제 buffer와 collective, mixed precision, mutation, 장애 복구를 거쳐 durable한 한 번의 학습 변화가 되는 과정을 독자가 직접 재구성할 수 있는 장이 된다.

**autograd engine의 node·edge·ready queue를 읽는다**

PyTorch autograd graph에서 tensor는 값이고 `grad_fn`은 그것을 만든 backward node로 향하는 입구다. 각 node는 incoming gradient를 받아 parent tensor에 대한 vector-Jacobian product를 계산한다. scalar loss에서 시작해 `.backward()`가 암묵적으로 1을 seed하는 이유다.

여러 downstream path가 같은 parameter에 도달하면 기여도가 누적되어야 한다. micrograd의 topological reverse와 `_backward` closure는 이 원리를 작은 graph에서 보여 준다. production engine은 device stream, ready dependency count와 worker queue를 관리하지만 chain rule의 합산 contract는 같다.

leaf parameter의 `.grad`는 graph node output과 별도 accumulator가 소유할 수 있다. hook이 보는 시점이 개별 contribution인지 누적 완료 뒤인지 구분한다. DDP reducer hook은 parameter gradient가 ready해지는 순간 bucket 상태를 바꾼다. hook registration 순서와 graph reuse를 source에서 확인한다.

`retain_graph=True`는 편의를 위한 memory-free option이 아니다. backward 뒤 saved tensors를 해제하지 않아 다음 backward를 허용하며 activation lifetime을 늘린다. 같은 graph에 의도치 않게 반복 backward하면 gradient가 중복 누적될 수 있다. higher-order gradient의 `create_graph`와 구분한다.

in-place mutation은 version counter로 검출되지만 모든 semantic 오류가 자동 검출되는 것은 아니다. `.data`나 graph 밖 storage mutation, custom extension의 alias contract가 잘못되면 조용한 오염 가능성이 있다. functional reference와 saved tensor hook, anomaly fixture를 사용한다.

**실험.** diamond graph에서 두 path의 gradient 합, shared parameter 두 호출, detached branch와 view mutation을 tiny FP64로 계산한다. `.grad` 초기화 전후와 두 번 backward를 비교한다. expected graph와 node count를 artifact로 보존한다.

**fused linear-cross-entropy backward를 기본 식과 대조한다**

LM head와 cross entropy를 fuse하면 full logits를 오래 저장하지 않고 vocabulary chunk별로 normalization과 gradient를 계산할 수 있다. memory 이득은 크지만 output gradient와 weight·hidden gradient를 정확히 보존해야 한다.

기본 경로에서 \(G_z=p-q\), output projection \(z=hW^\top\)라면 \(G_h=G_zW\), \(G_W=G_z^\top h\)다. fused kernel은 이 세 식을 chunk·shard와 low precision에서 재조립한다. loss numerator/count와 reduction scale도 같은 위치에서 적용된다.

full reference와 비교할 때 scalar loss만 보지 않는다. selected logit gradient, hidden gradient, output weight gradient와 tied embedding의 합산을 본다. vocabulary edge ID, extreme logits, all-ignore, uneven valid count와 non-contiguous hidden fixture를 넣는다.

chunk size는 workspace와 launch 수, accumulation ordering을 바꾼다. 작은 chunk에서 rounding error가 누적될 수 있다. exact/numerical tolerance를 dtype·shape별로 정한다. chunk option을 성능 flag로만 설명하지 않는다.

vocabulary-parallel fused path는 global max·sum과 target owner collective를 보존해야 한다. backward의 hidden contribution도 rank 사이 합의한다. 한 rank zero-valid와 target shard boundary를 주입한다. 1장의 sharded CE, 15·29장의 collective ledger와 연결한다.

compiled graph나 custom autograd Function이 fused path를 감싸면 backward registration, autocast와 saved tensor lifetime을 source에서 확인한다. fallback이 발생하면 actual dispatch를 profiler와 branch marker로 기록한다. 성능 비교는 parity gate 뒤 수행한다.

**profiler로 backward의 임계 경로를 찾는다**

backward trace에서 가장 긴 node만 찾으면 안 된다. gradient computation, DDP bucket readiness·collective와 optimizer 준비가 겹친다. parameter gradient가 ready된 시각과 bucket all-reduce enqueue·complete, 마지막 exposed communication을 같은 timeline에 둔다.

CPU gap은 Python hook, `.item()`, anomaly/debug, allocator나 graph compilation에서 올 수 있다. 한 CPU call의 시간이 길어도 이전 CUDA work를 기다린 synchronization일 수 있다. 해당 call 제거·비동기화 control에서 gap이 이동하는지 본다.

`torch.profiler.schedule`의 wait·warmup·active는 capture state를 만든다. backward 첫 step의 allocator·compile과 steady state를 분리한다. stack·shape·memory 옵션은 overhead를 추가한다. profiler on/off의 loss·gradient, step p95와 peak memory를 비교한다.

NVTX 또는 record function range는 microstep, forward/backward, bucket, unscale·clip·optimizer와 checkpoint를 표시한다. label cardinality를 제한하고 SampleID 원문을 넣지 않는다. rank별 clock alignment와 membership epoch를 보존한다.

trace가 full gradient tensor를 저장할 필요는 없다. boundary의 shape·dtype, norm·finite, selected projection과 event를 기록한다. incident window에서만 상세 stack을 켠다. 관측 hook이 tensor reference를 잡아 graph lifetime을 늘리지 않는지 memory snapshot으로 확인한다.

**분산 분모와 gradient 평균을 하나의 식으로 검산한다**

rank \(r\)의 loss numerator를 \(S_r\), valid target 수를 \(N_r\)라 하자. 원하는 global token mean gradient는 \(\sum_r \nabla S_r / \sum_r N_r\)다. 각 rank가 \(S_r/N_r\)를 backward하고 DDP가 rank mean을 만들면 일반적으로 다르다.

DDP가 gradient sum을 world size \(P\)로 나누는 계약이면 각 rank backward scalar를 \(P S_r/N_{global}\)로 만들 수 있다. 실제 reducer와 wrapper가 추가 scaling을 하는지 revision에서 확인한다. framework의 `num_items_in_batch`가 accumulation local/global 중 무엇인지 추측하지 않는다.

gradient accumulation에서도 모든 microbatch numerator와 count를 window 단위로 합쳐야 한다. microbatch별 valid count가 다른데 각 mean을 \(1/K\)로 더하면 objective가 바뀐다. uneven rank와 uneven microbatch를 동시에 넣은 fixture로 concatenated reference와 parameter delta를 비교한다.

zero-valid rank는 graph-connected zero로 collective 순서를 유지하거나 전역 합의로 update를 skip한다. 그 rank만 backward를 건너뛰면 hang 가능성이 있다. \(N_{global}=0\)이면 parameter·moment·scheduler와 committed clock이 모두 멈춰야 한다. scaler 정책은 별도 expected state를 갖는다.

communication hook이 compressed gradient를 반환하면 분모 scaling 이전·이후와 error-feedback state를 확인한다. hook의 numerical error와 denominator error를 분리한다. selected gradient projection과 residual buffer를 checkpoint/resume fixture에 넣는다.

**Transformers의 전역 token 평균은 어디서 닫히는가**

고정 revision `36deb0b53ed0863f4b4dfdea23dcaec7f3df3701`의 `Trainer._get_num_items_in_batch`는 누적 window에 속한 batch들의 valid label을 세고, causal LM이면 예측 대상이 아닌 첫 번째 위치를 빼기 위해 `labels[..., 1:]`를 사용한다. `average_tokens_across_devices=True`이고 world size가 1보다 크면 이 count를 gather한 뒤 합한다. 즉 `num_items_in_batch`는 현재 microbatch의 local count가 아니라, 이 경로에서는 accumulation window와 data-parallel rank를 걸친 global valid-target count다.

그러나 count를 전역화하는 것만으로 gradient가 맞지는 않는다. `compute_loss`는 model이 그 count를 소비하는 분기에서 loss에 process 수를 다시 곱한다. DDP reducer가 rank gradient를 평균하면 이 계수가 상쇄되어 \(\sum_r \nabla S_r/\sum_r N_r\)가 남는다. TP·EP-as-TP처럼 batch를 복제해 보는 rank는 `tp_size`로 divisor를 조정한다. 따라서 option의 의미는 “metric을 잘 평균한다”가 아니라 **loss owner의 global count division과 reducer owner의 world-size division을 서로 상쇄한다**는 것이다.

이 해석은 upstream 테스트의 범위에서만 받아들여야 한다. `test_loss_averaging`은 단일 GPU의 큰 batch와 multi-GPU의 작은 per-device batch를 비교하고, 전역 token averaging을 끈 경로는 step loss가 크게 다르며 켠 경로는 최대 차이 0.005 미만임을 검사한다. 다만 이는 지정된 tiny Qwen2, padding, drop-last, 정상 종료 rank의 10 step loss 동등성이다. rank 하나가 collective에 참여하지 않는 사건, process-group membership 변경, 전 rank의 valid count가 0인 step은 증명하지 않는다.

accumulation 테스트는 또 다른 축을 닫는다. `test_gradient_accumulation_grad_norm_with_num_items_in_batch`는 batch 8을 `1×8`과 `4×2` microstep으로 나눠 손실과 gradient norm을 큰-batch baseline과 비교한다. custom loss와 label smoothing도 같은 count를 소비할 때의 parity를 별도로 검사한다. 반면 count를 소비하지 않는 model 분기는 microbatch별 mean이므로 완화된 tolerance를 쓴다. 이 차이는 API compatibility와 objective parity가 같은 개념이 아님을 보여 준다.

**빈 rank와 사라진 rank를 같이 취급하지 마라.** valid target이 0인 rank도 process group에 살아 있으면 graph-connected zero numerator와 count 0을 collective에 낼 수 있다. 반면 rank process가 사라졌거나 다른 membership epoch의 group을 쓰면 이는 분모 문제가 아니라 collective liveness 문제다. 전자의 oracle은 global count가 양수면 정상 update, 0이면 all-rank no-commit이다. 후자의 oracle은 timeout·abort·rendezvous 사건이며 어떤 rank도 optimizer effect를 commit하지 않아야 한다.

TorchTitan의 validation 경로는 이 분리를 보조하는 구현 예다. pipeline microbatch loss는 reduction-sum의 합으로 모으고, DP·CP가 활성화되면 numerator를 `dist_sum`한 뒤 `total_global_valid_tokens`로 나눈다. 이는 validation metric의 분자·분모 계약을 보여 주지만 training gradient parity 테스트는 아니다. 또 현재 코드의 division 앞에 global count 0 guard가 보이지 않으므로 all-ignore validation batch는 별도 negative fixture로 남겨야 한다.

**saved tensor와 activation checkpoint의 상태 계약**

activation checkpoint는 forward 중 일부 intermediate를 저장하지 않고 backward에서 구간을 재실행한다. memory를 줄이는 대신 compute, RNG와 side effect를 추가한다. reference forward와 recompute forward가 같은 pure function이어야 gradient parity가 가능하다.

dropout RNG를 보존하는지, CPU·CUDA generator와 device set을 어떻게 다루는지 source에서 확인한다. custom random operation이나 새로운 device 이동은 보존 범위 밖일 수 있다. checkpointing on/off의 forward output, gradient와 RNG trace를 비교한다.

reentrant와 non-reentrant 경로는 autograd graph recording, backward API, detached tensor와 early stop 동작이 다를 수 있다. 사용하는 revision의 공식 source·test를 읽는다. option 이름만 보고 등가라고 가정하지 않는다.

hook과 profiler는 recompute에서 두 번 호출될 수 있다. forward-only metric을 중복 count하지 않게 invocation phase를 표시한다. saved tensor hooks로 offload/compress하면 serializer와 lifetime·stream dependency가 새 failure surface다.

checkpoint segment를 늘리면 activation memory는 줄 수 있지만 recompute와 kernel schedule이 변한다. peak, step p95와 numerical parity를 함께 측정한다. OOM 해결이 objective·BatchID 변경 없이 이뤄졌는지 28장 golden package로 검증한다.

**여러 optimizer와 loss의 commit을 조정한다**

actor·critic, discriminator·generator, modality-specific heads처럼 여러 optimizer가 있으면 하나의 `step`이 아니다. 각 optimizer의 parameter ownership, loss, backward·unscale·clip, commit clock과 scheduler를 manifest로 둔다. shared parameter가 어느 optimizer에 속하는지 명시한다.

한 loss의 backward가 shared graph를 해제하기 전에 다른 loss가 필요할 수 있다. loss 합을 한 번 backward할지 순차 backward와 retain을 쓸지 수식과 memory로 결정한다. 기여도 scale과 stop-gradient boundary를 작은 graph로 검산한다.

GradScaler를 공유하면 optimizer A의 overflow가 B를 어떻게 막는지, 별도 scaler면 shared gradient scale을 어떻게 맞추는지 정책이 필요하다. 일부 optimizer만 commit한 상태를 허용하는 algorithm인지, 전체 transaction으로 rollback해야 하는지 정한다.

GAN/RL처럼 alternating update는 phase와 policy/reward generation을 checkpoint한다. resume가 다른 optimizer phase에서 시작하면 trajectory가 바뀐다. scheduler와 data cursor도 optimizer별 또는 global clock에 연결한다.

fault fixture는 A step 후 B 전 process kill, 한 optimizer only overflow, shared parameter duplicate ownership과 state mapping swap을 포함한다. recovery assertion은 단순 model load가 아니라 phase, moment와 next delta다.

**gradient hook과 custom Function의 리뷰 절차**

hook은 gradient를 관찰하거나 바꿀 수 있다. 반환값이 원 gradient를 대체하는지, registration order와 once/per-use semantics를 확인한다. logging-only hook도 `.item()` sync나 tensor retention으로 성능·memory를 바꿀 수 있다.

gradient clipping·noise·mask를 hook에서 적용하면 optimizer 전 어느 단위에서 동작하는지 명시한다. parameter별 hook과 global norm은 등가가 아니다. DDP reducer hook 전후에 따라 local gradient와 reduced gradient 중 무엇을 바꿀지 달라진다.

custom `Function.backward`는 forward inputs 수만큼 gradient 또는 `None`을 반환하고 shape·dtype/device를 맞춰야 한다. saved tensor와 non-tensor metadata를 구분한다. in-place/dirty와 non-differentiable marking contract를 읽는다.

`gradcheck`와 `gradgradcheck`는 작은 double smooth fixture에 유용하다. low-precision kernel, nondifferentiable boundary, distributed side effect와 alias correctness는 별도 test가 필요하다. finite difference가 통과해도 production dispatch가 custom backward를 실제 사용했는지 branch marker로 확인한다.

upgrade 때 hook firing order, compile compatibility와 custom op registration이 바뀌는지 test한다. silent fallback과 graph break를 profiler에 기록한다. source 좌표는 public wrapper, registered operator, backward implementation과 upstream test를 모두 연결한다.

**3장으로 넘길 UpdateManifest를 확장한다**

manifest에는 LossManifest, graph boundary, saved tensor·checkpoint policy, parameter alias와 trainable ownership이 있다. microstep별 raw/scaled/unscaled gradient, reducer bucket·collective, global norm, clip coefficient와 commit vote를 기록한다.

optimizer별 group/options, moments·step before/after, parameter delta, scheduler와 scaler state를 둔다. attempted/committed update, zero-valid·overflow와 exception reason을 분리한다. checkpoint generation과 next BatchID를 연결한다.

performance evidence에는 backward phase, exposed collective, profiler state, peak activation/saved tensor와 observer overhead가 있다. fused/compiled actual dispatch와 fallback을 기록한다. 실행하지 않은 CUDA path는 `hardware-pending`이다.

3장은 이 manifest를 tiny GPT의 embedding→attention→MLP→LM head와 결합해 한 row가 실제 parameter 변화와 generation으로 이어지는지 실험한다. 11~15장은 optimizer·precision·parallel detail을 확장하고 26·28장은 관측과 golden resume를 재사용한다.

### 2.11.3 DDP·fused·CUDA Graph의 update state를 인수한다

**DDP `no_sync`와 bucket rebuild를 검증한다**

gradient accumulation에서 중간 microstep의 all-reduce를 피하려고 `no_sync` 계열 context를 사용할 수 있다. 마지막 microstep만 synchronization을 허용해야 window 전체 local contribution이 reduce된다. context 범위가 forward를 포함해야 하는지 사용하는 wrapper source에서 확인한다.

중간 microstep에서 sync가 실수로 발생하면 correctness가 우연히 맞더라도 communication이 늘고 scaling이 달라질 수 있다. 반대로 마지막 microstep까지 `no_sync`면 replica gradient가 갈라진다. collective ledger와 selected parameter delta로 두 오류를 잡는다.

DDP bucket은 parameter size·order와 readiness에 따라 구성되고 dynamic graph/unused parameter 조건에서 rebuild될 수 있다. bucket mapping generation, parameter FQN, dtype와 size를 기록한다. compile이나 adapter injection 뒤 mapping이 바뀌는지 본다.

`find_unused_parameters` 같은 option은 autograd traversal과 reducer readiness를 바꾼다. 필요 없는 경우 overhead가 있고, 필요한 dynamic branch에서 끄면 hang 가능성이 있다. static-graph 가정은 actual control flow와 fixture로 검증한다.

gradient-as-bucket-view 같은 memory optimization은 `.grad` storage alias와 zeroing/detach contract를 바꿀 수 있다. optimizer·hook이 gradient tensor를 교체하거나 in-place 수정할 때 source/test를 확인한다. option→alias state→memory/effect를 manifest에 넣는다.

**fused optimizer의 state를 hand reference와 대조한다**

foreach·fused 구현은 여러 parameter update를 묶고 device-side step과 found-inf를 사용할 수 있다. scalar loop와 같은 수학을 목표로 해도 operation ordering, rounding, intermediate dtype와 memory peak가 다르다. actual dispatch와 fallback을 기록한다.

tiny parameter 두 개를 서로 다른 group에 넣어 LR, decay와 epsilon을 다르게 한다. 첫·두 번째 step의 moment, bias correction와 delta를 FP64 reference로 계산한다. one parameter의 gradient를 `None`, 다른 하나를 zero로 만들어 skip·decay semantics를 본다.

AMP overflow에서 fused optimizer가 전달받는 scale/found-inf와 step state를 확인한다. parameter·moment·scheduler가 모두 멈추는지, device-side step counter가 증가하지 않는지 test한다. 반환값이나 kernel launch 존재만으로 commit을 추론하지 않는다.

state dict round-trip은 foreach/fused/scalar 사이 migration을 시험한다. parameter group order와 FQN mapping을 보존한다. unsupported dtype/device에서 fallback하면 numerical/performance support row를 분리한다.

**CUDA graph capture에서 update state를 고정한다**

CUDA graph replay는 같은 memory address와 launch topology를 반복해 CPU overhead를 줄인다. input, gradient, optimizer state와 scalar buffer가 stable address를 가져야 한다. dynamic allocation, Python branching과 host `.item()`은 capture 경계를 깨뜨릴 수 있다.

GradScaler의 scale, found-inf, optimizer step과 LR가 host scalar면 replay마다 바뀌는 state를 graph가 보지 못할 수 있다. capturable optimizer는 device tensor state를 사용한다. 실제 revision에서 어떤 option과 device가 지원되는지 본다.

shape가 바뀌는 variable sequence는 padding/bucket 또는 여러 captured graph가 필요할 수 있다. wrong-shape replay를 거부하고 graph key에 model/config/dtype/shape를 넣는다. capture cache가 다른 model generation과 섞이지 않는다.

overflow·zero-valid·checkpoint callback 같은 rare branch를 capture가 어떻게 처리하는지 시험한다. 정상 fast path만 capture하고 exceptional path는 graph 밖으로 나갈 수 있지만 state transition과 synchronization을 명시한다. failure가 조용히 stale update를 replay하지 않게 한다.

reference eager와 captured replay의 forward, gradient, two-step delta와 scheduler를 비교한다. warm-up/capture 비용과 steady state를 분리한다. 실제 CUDA 장비에서 실행하지 않았다면 source-confirmed/hardware-pending을 구분한다.

**backward 장애를 세 사례로 복원한다**

**사례 A: loss는 같고 gradient만 절반이다.** 두 microbatch valid count가 달랐고 wrapper가 각 mean을 accumulation 수로 다시 나눴다. scalar logging은 재계산돼 맞아 보였지만 backward scale이 틀렸다. microbatch numerator/count와 concatenated gradient projection이 원인을 드러냈다.

수정은 global window denominator로 scaling하고 framework의 추가 division과 중복되지 않게 했다. unequal-length fixture와 selected parameter delta를 regression에 넣었다. learning rate를 두 배로 올리는 완화는 Adam·clip과 batch별 scale을 보존하지 못하므로 거부했다.

**사례 B: profiler를 켜면 OOM이 난다.** backward hook이 activation tensor를 Python list에 저장해 graph lifetime을 늘렸다. reserved-minus-allocated만 보고 allocator fragmentation으로 오진했다. profiler on/off snapshot에서 살아 있는 stack이 hook으로 이어졌다.

수정은 detach된 bounded summary와 ring buffer를 사용하고 hook을 incident window 뒤 제거했다. observer on/off numerical parity, peak와 step p95를 검증했다. 관측 도구가 만든 장애를 모델 memory 요구로 기록하지 않았다.

**사례 C: 한 rank overflow 뒤 replica가 갈라졌다.** rank-local GradScaler가 한 rank optimizer만 skip했고 다른 rank는 commit했다. 다음 all-reduce에서 gradient가 다시 평균돼 loss는 잠시 자연스러웠지만 parameter checksum이 이미 달랐다.

수정은 found-inf commit vote를 data-parallel group에서 합의해 모든 rank가 함께 skip하게 했다. parameter·moment·scheduler와 scaler expected state를 rank별로 검사했다. network timeout이 없더라도 distributed correctness failure가 될 수 있음을 29장 fault matrix에 연결했다.

**update를 관측할 최소 metric schema**

상시 metric에는 attempted/committed update, microstep, valid token, loss numerator/denominator, LR, grad norm, clip coefficient, scaler scale·overflow/skip와 checkpoint age를 둔다. counter/gauge와 reset semantics를 정의한다.

module group별 gradient·parameter·update norm은 bounded taxonomy로 sampling한다. parameter 이름 전체를 label로 쓰지 않는다. histogram과 full tensor는 incident artifact로 보존한다. `None`, zero와 nonfinite count를 분리한다.

분산에서는 rank local value의 mean·max·min과 max-rank owner를 제한적으로 집계한다. global loss는 numerator/count, norm은 수학에 맞는 reduction을 쓴다. rank mean 하나가 token·parameter weight를 왜곡하지 않는다.

metric event에는 UpdateID, membership epoch와 checkpoint generation을 기록한다. event time과 ingestion time을 분리한다. logger failure가 optimizer commit을 막을지 계속할지 정책을 정하고 telemetry incomplete를 기록한다.

alert는 loss nonfinite, gradient nonfinite, repeated overflow, committed-update stall, trainable-zero-delta와 replica checksum mismatch를 서로 다른 producer에 연결한다. threshold만 주지 않고 첫 probe와 safe action을 포함한다.

**2장의 독립 인수 절차**

인수자는 임의 UpdateID에서 BatchID와 LossManifest를 확인하고 raw/scaled/unscaled gradient, reducer completion, clip과 optimizer delta를 재계산한다. scheduler·scaler와 checkpoint generation이 commit 결과와 맞는지 본다.

다음으로 wrong denominator, overflow, partial optimizer state, view mutation와 rank-local nonfinite fixture를 하나씩 실행한다. expected first detector가 작동하고 parameter·moment·clock이 정책대로 멈추는지 본다. 변조 제거 후 clean control을 다시 실행한다.

profiler trace에서 forward/backward, bucket, optimizer critical path와 observer overhead를 확인한다. 실제 dispatch가 source ledger의 eager/fused/compiled path와 맞는지 본다. fallback과 미실행 CUDA combination을 숨기지 않는다.

checkpoint clean resume에서 next BatchID, gradient와 delta를 uninterrupted branch와 비교한다. model weight만 같은 warm-start를 resume으로 승인하지 않는다. multi-optimizer면 phase와 각 state를 확인한다.

결과에는 source commit·symbol/test, environment, fixture, exact/numerical tolerance와 reviewer를 넣는다. 3장은 승인된 UpdateManifest만 받아 tiny GPT 종단 실험을 구축한다.

**custom CUDA operation의 backward를 승인한다**

custom operation은 Python schema, dispatcher registration, device implementation, autograd formula와 meta/fake implementation을 하나의 bundle로 검토한다. forward CUDA kernel만 빠르다고 production training에 넣지 않는다. compile과 autocast registration도 실제 path를 바꾼다.

shape, stride, dtype, alignment, empty tensor와 non-contiguous input contract를 명시한다. kernel이 contiguous를 요구하면 wrapper가 copy하는지 거부하는지 기록한다. silent contiguous copy는 memory·performance를 바꾼다.

backward는 input별 gradient, accumulation dtype와 atomic/reduction ordering을 확인한다. race condition은 작은 정상 input에서 재현되지 않을 수 있다. deterministic fixture, repeated stress와 sanitizer/tool evidence를 support 상태에 맞게 둔다.

FP64 CPU/reference와 selected CUDA shape의 forward·gradient를 비교한다. gradcheck 가능한 smooth core, low-precision tolerance, boundary/empty와 large-stride를 분리한다. fused production shape에서 actual dispatch를 profiler로 확인한다.

오류 뒤 CUDA asynchronous exception이 다른 API에서 surfaced될 수 있다. fault location과 observed location을 구분하고 bounded synchronization debug mode를 사용한다. 대규모 runtime을 실행하지 않았다면 hardware result를 만들어내지 않는다.

**nonfinite 최초 발생 지점을 이분 탐색한다**

loss NaN이 보인 step보다 activation, gradient나 optimizer state가 먼저 nonfinite일 수 있다. 마지막 정상 checkpoint와 최초 비정상 UpdateID를 고정하고 같은 BatchID·RNG에서 재현한다. 데이터·config·environment identity를 먼저 확인한다.

forward boundary를 embedding, attention, MLP, norm, logits와 loss로 나눠 finite·norm을 검사한다. 전부 정상이라면 backward를 output gradient, layer group, accumulation, unscale과 clip로 나눈다. gradient도 정상이면 optimizer moment·denominator·epsilon과 parameter update를 본다.

모든 tensor에 hook을 달지 않고 binary search처럼 구간을 좁힌다. anomaly detection과 sync는 위치 정보를 주지만 overhead와 execution ordering을 바꿀 수 있다. debug on/off의 재현 여부를 기록한다.

의심 연산만 FP32/FP64 reference로 실행해 overflow, underflow, cancellation과 invalid domain을 구분한다. dtype을 전체적으로 올려 증상을 없애는 것과 root cause를 설명하는 것을 구분한다. mask all-invalid, zero variance, division denominator와 exponential range를 확인한다.

수정 뒤 failure batch와 정상 neighboring batches, several next updates를 검증한다. nonfinite가 사라졌어도 clipping fraction·update ratio와 quality가 달라지지 않았는지 본다. regression fixture에는 first detector와 no-commit assertion을 둔다.

**update contract의 교차 판정**

autograd는 graph를 따라 gradient 기여를 정확히 합산해야 한다. saved tensor와 alias는 backward가 기대한 forward state를 보존해야 한다. accumulation과 DDP는 1장의 global objective denominator와 같은 gradient를 만들어야 한다.

AMP는 scale·unscale·found-inf를 상태기계로 관리하고 overflow에서 group 전체 commit을 막아야 한다. clipping은 unscale·accumulation 뒤 한 번 적용된다. optimizer·scheduler와 여러 optimizer phase는 명시한 transaction에서 함께 움직인다.

fused, compiled, CUDA graph와 custom op에는 reference parity, actual dispatch와 failure fixture가 필요하다. 속도 향상으로 numerical·durability 실패를 상쇄하지 않는다. observer는 state를 바꾸지 않고 first divergence를 찾을 만큼 충분해야 한다.

checkpoint는 UpdateManifest의 model·optimizer·scheduler·scaler·RNG·data와 phase를 보존한다. clean resume의 다음 update가 uninterrupted branch와 declared grade에 맞는다. 분산 fault 뒤 old membership이 commit하지 않는다.

이 조건이 통과하면 독자는 loss scalar에서 parameter delta까지 각 state의 소유자와 전이를 설명할 수 있다. 3장은 이 update를 tiny GPT 전체에 적용해 모델 구성 요소와 generation behavior까지 추적한다.

**clipping을 parameter-space 투영으로 해석한다**

global L2 norm clipping은 gradient \(g\)가 radius \(c\) 밖이면 \(g\cdot c/(\|g\|+\epsilon)\)로 줄인다. 방향은 유지하고 크기만 경계에 투영하는 직관이다. 그러나 optimizer의 adaptive preconditioning 전 gradient 공간에서 적용되므로 최종 parameter delta 방향과 norm은 그대로 비례하지 않을 수 있다.

parameter group별 clip은 전체 global clip과 다른 방향을 만든다. layer별 threshold, value clipping과 adaptive gradient clipping도 서로 다른 geometry다. 이름을 모두 “clip”으로 묶지 않는다. 구현의 norm scope, p-norm, epsilon과 sparse/nonfinite 처리를 확인한다.

AMP scale된 gradient를 clip하면 threshold가 loss scale에 따라 달라진다. 반드시 unscale 뒤인지 source와 state marker로 검증한다. accumulation 중간마다 clip하면 contribution 합을 clip한 결과와 다르다. 마지막 microstep·global reduction 뒤 한 번 적용하는 reference와 비교한다.

clip fraction이 높으면 폭발을 막고 있다는 뜻이지 안정적인 학습의 증거가 아니다. pre-clip norm, coefficient, post-clip norm, update-to-weight와 loss·LR를 함께 본다. threshold를 낮춰 NaN을 숨기지 않는다.

fixture는 두 parameter group에 직교 gradient를 주고 global/group clip 결과를 손계산한다. 한 rank에만 큰 gradient를 넣어 reduction 전 clip과 후 clip이 다른 결과를 보이게 한다. declared policy와 parameter delta를 exact 비교한다.

**update 변경의 영향 반경을 계산한다**

loss scaling, clipping, optimizer나 reducer 코드를 고치면 어떤 RunID와 checkpoint가 영향을 받는지 configuration과 source digest로 찾는다. metric logging만 잘못됐는지 실제 backward·commit이 달랐는지 분리한다. 전자는 dashboard 재계산으로 고칠 수 있지만 후자는 trajectory와 descendants를 다시 평가해야 한다.

affected checkpoint에서 만든 adapter, merge, quantized export와 deployment를 Artifact DAG로 찾는다. 수치 차이가 작다는 이유로 영향 query를 생략하지 않는다. evaluation과 safety budget 안인지 새 subject별로 판정한다.

hotfix는 old source를 inplace 교체하지 않고 new revision과 fixture generation을 만든다. same GoldenBatchID에서 expected first divergence가 수정 경계와 맞는지 본다. unrelated boundary가 먼저 달라지면 dependency/config drift를 조사한다.

rollback은 last unaffected checkpoint와 data cursor, optimizer/scheduler/scaler compatibility를 확인한다. code만 되돌리고 새 schema state를 읽어도 되는지 test한다. 지원되지 않으면 clean branch나 migration을 사용한다.

incident record에는 direct cause, contributing condition, detection gap, affected artifacts, mitigation와 regression source/test를 넣는다. “gradient 문제”처럼 계층 이름으로 닫지 않는다. 어떤 state가 언제 잘못 전진했는지 명시한다.

**loss seed에서 optimizer effect까지 완전한 원장을 만든다**

**독립 검토의 event 재생**

검토자는 SampleID에서 loss contribution, autograd node, gradient accumulator, reducer bucket, clip, optimizer delta와 checkpoint generation까지 정방향으로 걷는다. 이어 checkpoint의 moment·step에서 parameter, gradient contribution과 원 sample로 역방향 추적한다.

각 edge에는 owner, source revision, shape·dtype, checksum 또는 numerical summary와 state transition이 있다. 합산이나 alias 때문에 일대일 대응이 불가능한 지점은 contribution set과 reduction 식을 기록한다. 추측으로 한 sample이 특정 weight에 저장됐다고 쓰지 않는다.

wrong denominator와 overflow fixture를 다시 실행해 detector와 no-commit을 확인한다. profiler·hook을 제거한 clean control이 같은 update를 내는지 본다. reviewer, environment와 결과 digest를 인수 report에 넣는다.

이 왕복이 끊기지 않으면 UpdateID는 다음 장의 tiny GPT 실험에 사용할 수 있다. 끊기면 모델 크기를 키우거나 최적화 옵션을 추가하지 않고 누락된 state·evidence부터 보강한다.

인수자는 source-confirmed, upstream-test-confirmed, local-synthetic-executed와 hardware-pending을 구분한다. fused CUDA·NCCL 경로를 실행하지 않았다면 reference 설명을 측정 결과처럼 쓰지 않는다. 다음 실행 command, 예상 invariant와 failure artifact를 명시해 미검증 항목을 재현 가능한 작업으로 남긴다.

모든 예외는 책임자, 범위와 만료 시점을 가지며 만료 뒤 자동으로 gate를 다시 실패시킨다. 과거의 우연한 PASS를 새 revision이나 topology에 상속하지 않는다.

최종 report에는 canonical UpdateManifest digest, 검토자, 실제 실행 시각, environment와 재검증 조건을 함께 고정해 후속 실험의 유일한 기준점으로 사용한다.

**운영 인계에 남길 state 메모**

새 학습을 시작하기 전에는 첫 update를 특별 취급한다. optimizer moments가 초기 상태이고 gradient history가 짧아 손계산과 비교하기 가장 쉽다. 첫 batch의 source IDs, loss manifest, raw·unscaled gradient, clip coefficient, parameter delta를 보존한다. resume와 framework upgrade에서 같은 fixture를 다시 사용한다.

정상 장기 run에서도 낮은 주기로 selected update dossier를 남긴다. 장애가 발생한 뒤에만 계측을 켜면 최초 원인을 만든 state를 잃을 수 있다. 다만 전체 tensors를 항상 덤프하지 않고 bounded projection과 counters를 보존한다. 민감한 sample은 접근 통제 artifact로 분리한다.

운영자는 loss spike, overflow, hang, checkpoint 실패를 서로 독립 alert로 보되 같은 UpdateID로 연결한다. 하나의 원인이 여러 증상을 낼 수 있다. rank failure가 collective timeout과 checkpoint 누락을 만들고, 비정상 batch가 overflow와 scaler skip을 만들 수 있다. 시간 근접성보다 state ordering을 우선한다.

복구 후에는 dashboard가 정상으로 돌아왔다는 이유로 사건을 닫지 않는다. 마지막 정상 checkpoint에서 failure batch를 격리 재생하고 최초 차이를 확인한다. 수정 revision에서 고장 주입 test가 예상 detector를 울리는지 본다. 재발 방지 evidence가 source와 fixture에 남아야 한다.

이 메모의 핵심은 단순하다. update는 한 함수 호출이 아니라 여러 상태의 합의된 commit이다. 어느 rank와 어느 optimizer도 절반만 전진해서는 안 되며, 재개는 같은 다음 입력과 state를 복원해야 한다. 이 기준을 유지하면 규모가 커져도 조사 질문은 흐려지지 않는다.

독립 인계자는 마지막으로 선택한 parameter 하나의 delta를 raw gradient에서 다시 계산한다. 필요한 moment와 bias correction, clipping, learning rate, decay가 dossier에 없으면 인계를 거부한다. 동시에 다음 SampleID와 scheduler·scaler counter를 확인한다. 숫자가 맞아도 source revision이 없으면 우연한 일치일 수 있고, source가 맞아도 fixture가 없으면 실행 경로를 증명하지 못한다. 양쪽이 함께 닫혀야 UpdateID를 승인한다.

승인된 기록은 이후 장애 조사와 재개 검증의 기준점이 된다. 변경된 옵션은 이 기준점에서 최초로 달라져야 할 상태를 명시하며, 예상 밖의 앞선 차이는 즉시 새로운 원인 후보로 등록한다. 이 규율이 시행착오를 재현 가능한 공학으로 바꾼다.
**loss seed에서 AccumulateGrad까지 추적한다**

1장의 `LossEnvelope`가 scalar tensor `L`을 넘기면 `.backward()`는 scalar 출력에 1을 seed한다. non-scalar 출력에는 명시적 upstream gradient가 필요하다. mean loss와 sum loss는 seed가 같아도 내부 normalization이 다르다.

각 forward operation은 backward node와 필요한 saved tensor를 만든다. `grad_fn.next_functions`는 Python tensor 객체 목록이 아니라 gradient edge를 보여준다. leaf parameter에는 일반 연산 grad_fn 대신 AccumulateGrad 경로가 연결된다.

branch 두 개가 같은 parameter를 사용하면 gradient contribution은 합쳐져야 한다. hook 호출 순서나 graph traversal 순서를 최종 수학 합과 혼동하지 않는다. FP64 tiny graph에서 각 branch VJP와 합을 손으로 계산한다.

view, transpose, slice는 storage alias와 stride를 만든다. backward는 view gradient를 base layout에 scatter하거나 reshape한다. overlapping view의 in-place mutation은 version counter error 또는 잘못된 결과 위험이 있다.

saved tensor version은 forward 후 in-place 변경을 감지한다. `.data`나 비정상 custom kernel로 검사를 우회하지 않는다. anomaly detection은 유용하지만 항상 켤 production 기능과 동일하지 않다.

parameter `.grad`가 None인 것과 zero tensor인 것은 다르다. unused parameter는 None일 수 있고 `zero_grad(set_to_none=True)`도 None을 만든다. optimizer가 None과 zero를 decay·step에서 다르게 처리하는지 확인한다.

gradient accumulation은 `.grad` buffer에 microstep contribution을 더한다. window 시작 전 zeroing, 마지막 commit 후 zeroing, exception 중 abort 규칙을 고정한다. stale gradient가 다음 window로 넘어가지 않아야 한다.

`retain_graph=True`는 graph를 재사용하지만 saved tensor memory를 유지한다. 무심코 켜 memory leak을 숨기지 않는다. 동일 graph에서 두 backward가 의도적으로 gradient를 누적하는지 시험한다.

`create_graph=True`는 gradient 계산 자체를 higher-order graph로 기록한다. 일반 training에 필요하지 않다면 꺼야 한다. gradient penalty나 meta-learning 경로는 second derivative fixture를 둔다.

hook은 tensor grad, node, parameter post-accumulate 위치마다 의미가 다르다. 관측 hook이 gradient를 반환해 수정하지 않는지 확인한다. distributed reducer hook과 사용자 hook의 순서도 runtime trace로 본다.

최초 차이 탐색은 loss scalar, selected logit gradient, last-layer weight gradient, 중간 block gradient, embedding gradient 순서로 내려간다. 마지막 동일 edge와 최초 다른 edge를 기록한다.

**autograd source·함수·state·test 묶음**

PyTorch 고정 commit `3691693263d2b66a68867e39b7449876844e06cf`에서 Python 진입, engine 실행, AccumulateGrad, saved variable, anomaly 경로를 symbol과 function hash로 고정한다. line number만 영구 계약으로 쓰지 않는다.

Python `Tensor.backward`에서 C++ engine까지 caller chain을 적는다. 실제 build가 local source commit과 같은지 `torch.__version__`, build config, shared library hash로 확인한다.

engine ready queue는 dependency count가 0이 된 node를 실행한다. CPU와 device stream, reentrant backward, multithreading이 순서를 바꿀 수 있지만 의존성 의미는 같아야 한다.

node 실행 전후 selected tensor metadata를 probe한다. shape, dtype, device, stride, storage alias, finite count를 기록한다. 모든 tensor를 CPU로 복사해 timing을 망치지 않는다.

test A는 `y=x²+x²`의 shared leaf다. 기대 gradient `4x`를 확인한다. test B는 transpose와 slice가 있는 matmul이다. noncontiguous gradient를 FP64 reference와 비교한다.

test C는 in-place saved tensor mutation이다. 명시적 error를 기대한다. test D는 unused branch다. None gradient와 DDP unused detection을 확인한다.

test E는 custom Function의 forward/backward와 gradcheck다. double precision, 작은 입력, noncontiguous input, empty tensor, boundary 값을 넣는다. 필요하면 gradgradcheck도 한다.

test F는 compiled/eager graph다. graph break count, selected backward, gradient와 parameter delta를 비교한다. source branch 존재와 actual dispatch를 분리한다.

**mixed precision을 네 dtype과 세 상태로 분리한다**

parameter storage dtype, forward compute dtype, accumulation dtype, optimizer master/state dtype을 따로 적는다. “FP16 training” 한 단어로 이 네 상태를 합치지 않는다.

autocast는 operation policy에 따라 matmul, convolution, normalization, reduction의 dtype을 고른다. 입력 dtype만 보고 kernel accumulator dtype을 확정하지 않는다. profiler와 source dispatch를 본다.

FP16은 exponent 범위가 좁아 작은 gradient underflow와 큰 값 overflow에 취약하다. BF16은 exponent가 넓지만 mantissa가 짧다. TF32는 FP32 storage와 다른 matmul precision 정책이다.

loss scaling은 `L_s=sL`로 만들고 backward gradient `sg`를 얻은 뒤 optimizer 전 `g`로 unscale한다. clipping을 scaled gradient에 적용하면 threshold 의미가 `s`에 따라 바뀐다. 순서는 unscale→finite check→clip→step이다.

GradScaler state에는 scale, growth tracker, growth/backoff factor, interval, optimizer별 stage와 found-inf가 있다. checkpoint resume가 scale만 복원하면 evolution이 달라질 수 있다.

한 optimizer에 `unscale_`를 두 번 호출하는 것은 stage violation일 수 있다. 여러 optimizer가 shared parameter를 가지면 소유권을 명시한다. optimizer별 found-inf와 global atomicity 정책을 둔다.

한 rank에서만 overflow가 나면 모든 replica가 step을 skip해야 parameter가 같게 남는다. found-inf를 collective로 합치는지 integration test한다. rank-local skip은 다음 step부터 이미 다른 model이다.

scale growth는 성공 step 수에 따라 달라진다. accumulation microstep마다가 아니라 optimizer commit마다 tracker가 증가해야 한다. skipped step과 scheduler advance를 함께 정의한다.

FP32 master weight를 쓰는 optimizer는 model parameter와 master copy의 관계를 checkpoint에 저장한다. cast-back rounding과 parameter checksum을 본다.

mixed precision fixture는 같은 tiny network를 FP64 oracle, FP32, BF16, FP16+scaler로 실행한다. loss, raw/scaled/unscaled grad, clip coefficient, update를 비교한다.

near-underflow fixture는 작은 input과 loss coefficient를 쓴다. scaling 없이 gradient가 0이 되고 scaling 후 회복되는지 본다. near-overflow fixture는 scale backoff와 synchronized skip을 시험한다.

normalization과 softmax는 민감한 reduction이다. fused kernel이 accumulator를 승격하는지 본다. output parity만 아니라 gradient parity를 검사한다.

**accumulation의 global objective를 식으로 보존한다**

microbatch `k`의 loss numerator를 `N_k`, valid count를 `C_k`라 하면 window 목표는 `L=Σ_k N_k/Σ_k C_k`다. 각 `N_k/C_k`를 같은 비중으로 평균하면 counts가 다를 때 다른 objective다.

backward 전에 각 microbatch numerator를 global `C=ΣC_k`로 나누거나, local mean gradient에 `C_k/C`를 곱혀야 한다. framework가 accumulation step 수로 자동 나누는지 source에서 확인한다.

마지막 remainder window가 계획보다 적은 microbatch를 가지면 denominator와 scheduler commit은 실제 소비량에 맞춰야 한다. drop_last 정책을 manifest에 둔다.

zero-valid microbatch는 numerator 0, count 0으로 참여할 수 있지만 division은 하지 않는다. window global count가 0이면 optimizer step을 명시적으로 skip하고 scheduler/RNG 정책을 정한다.

multiple loss component가 pair count, token count, pixel count처럼 다른 denominator를 가지면 각각 sum/count 후 coefficient로 결합한다. local scalar를 먼저 더하지 않는다.

gradient accumulation과 DDP를 함께 쓰면 rank·microstep 이중 합이 있다. global objective는 `Σ_rΣ_k N_rk / Σ_rΣ_k C_rk`다. local mean의 rank 평균은 일반적으로 같지 않다.

DDP reducer가 gradient를 world size로 나눈다면 loss scale에서 그 factor를 보정해야 할 수 있다. actual reducer semantics와 communication hook을 확인한다.

fixture는 counts `(1,3,8)`과 두 ranks의 불균형 배치를 쓴다. concatenated FP64 reference, accumulation, DDP 결과의 selected gradient와 SGD delta를 비교한다.

**DDP reducer와 collective state를 관측한다**

DDP는 parameter를 bucket으로 묶고 autograd hook에서 gradient ready를 표시한다. bucket이 ready되면 collective를 launch해 communication과 backward를 overlap할 수 있다.

bucket order, size cap, parameter registration order가 timing을 바꾼다. correctness와 performance를 분리한다. rebuild가 발생하면 새 mapping을 ledger에 저장한다.

`no_sync`는 중간 accumulation microstep의 collective를 미룬다. 마지막 microstep forward/backward가 sync context 밖에 있어야 하는지 revision source를 확인한다.

마지막 microstep도 no_sync이면 replica gradient가 다르다. 중간마다 sync하면 통신이 늘고 scale 조합이 달라질 수 있다. collective trace와 parameter checksum으로 잡는다.

unused parameter는 bucket readiness를 막을 수 있다. `find_unused_parameters`, static graph option, dynamic branch를 실제 control-flow fixture로 검증한다.

gradient-as-bucket-view는 `.grad` storage가 bucket view일 수 있다. `detach_`나 resize 같은 operation 제약을 확인한다. optimizer와 zeroing이 alias를 보존하는지 본다.

communication hook은 all-reduce 대신 compression이나 custom reduction을 할 수 있다. hook state, error feedback, division factor를 checkpoint에 포함한다.

collective sequence는 모든 rank에서 같아야 한다. 한 rank의 exception이나 conditional loss가 call order를 바꾸면 hang한다. coordinated abort와 restart를 설계한다.

NCCL kernel 경계에서 stream, event, async work completion을 본다. optimizer가 reduction 완료 전에 grad를 읽지 않아야 한다. profiler timestamp만 아니라 dependency event를 확인한다.

test는 world size 1/2, unequal counts, unused branch, no_sync 마지막 오류, rank-one overflow, custom hook을 포함한다. global FP64 reference와 delta를 비교한다.

**optimizer effect를 parameter 한 원소에서 재생한다**

SGD는 `θ←θ−ηg`다. momentum은 buffer `v←μv+g`, update `θ←θ−ηv`처럼 구현되지만 dampening, Nesterov convention을 source에서 확인한다.

AdamW는 `m←β1m+(1−β1)g`, `v←β2v+(1−β2)g²`, bias correction과 `m/(sqrt(v)+ε)`를 사용하고 decoupled weight decay를 적용한다. epsilon이 sqrt 안인지 밖인지 구현 식을 확인한다.

step counter가 0/1 어느 시점에서 bias correction에 들어가는지 본다. checkpoint step off-by-one은 첫 resumed update를 바꾼다.

weight decay가 gradient moment에 들어가는 L2와 decoupled AdamW는 다르다. bias, norm scale, embedding을 제외하는 parameter group policy를 logical parameter type으로 감사한다.

gradient clipping은 optimizer 입력을 바꾼다. global norm clipping은 모든 parameter gradient norm의 합에서 coefficient를 만든다. 분산 shard에서는 global sum-square collective가 필요하다.

value clipping, adaptive clipping, per-group clipping은 다른 투영이다. option 이름과 threshold 단위를 manifest에 넣는다.

foreach와 fused optimizer는 tensor list grouping과 kernel을 바꾼다. capturable은 step과 scalar state를 device tensor로 둘 수 있다. differentiable은 optimizer step graph를 만든다.

fallback이 조용히 일어나면 기대 performance가 없다. actual dispatched kernel과 tensor grouping을 profiler/source로 확인한다. eager reference와 update parity를 먼저 닫는다.

selected parameter element에 pre-weight, raw grad, unscaled grad, clipped grad, m/v before/after, step, LR, decay, post-weight를 저장한다. 손계산 delta와 맞춘다.

parameter alias가 optimizer group에 두 번 들어가면 update가 중복될 수 있다. storage identity와 logical parameter ID로 dedup한다.

frozen parameter, None grad, zero grad가 decay에서 어떻게 처리되는지 test한다. None을 zero와 같은 것으로 가정하지 않는다.

overflow skip 시 optimizer moment, parameter, step counter, scheduler가 모두 commit되지 않는지 확인한다. 일부 state만 전진하면 재현이 깨진다.

**CUDA kernel 경계를 forward·backward·optimizer로 나눈다**

Python operation 하나는 dispatcher, composite op, ATen kernel, vendor library 또는 generated fused kernel로 내려간다. profiler name만으로 source 함수를 확정하지 않는다.

linear backward는 input gradient, weight gradient, bias gradient GEMM을 만들 수 있다. fusion과 필요 gradient flags에 따라 일부만 실행한다. shape·layout·transpose가 kernel selection을 바꾼다.

cross-entropy backward는 softmax probability를 materialize하지 않는 fused kernel일 수 있다. FP64 reference `p−q`, ignore, smoothing, denominator와 비교한다.

normalization backward는 reduction과 saved mean/rstd를 사용한다. recompute path와 saved path의 dtype을 확인한다. epsilon은 forward와 backward에 일관돼야 한다.

attention backward는 recompute, dropout mask, causal mask, sequence metadata를 사용한다. flash backend와 math backend gradient tolerance를 고정한다.

custom CUDA op는 schema, device guard, stream, launch check, autocast, autograd, meta/fake implementation을 갖춰야 한다. forward kernel만 등록해 training에 넣지 않는다.

kernel launch는 비동기다. error가 뒤 synchronization에서 보일 수 있다. debug sync가 최초 failing launch를 좁히지만 timing을 바꾸므로 별도 run으로 사용한다.

illegal memory access, race, uninitialized read는 output NaN과 다르다. sanitizer와 deterministic tiny fixture를 사용한다. 오류 후 CUDA context를 계속 신뢰하지 않는다.

atomic reduction은 합 순서 비결정성을 만들 수 있다. bitwise 기준과 numerical tolerance 기준을 구분한다. deterministic mode가 지원하지 않는 op를 명시한다.

CUDA graph capture는 pointer, shape, control flow, collective sequence를 고정한다. GradScaler found-inf나 dynamic clipping이 capture와 어떻게 연결되는지 actual branch를 시험한다.

compiled fusion에는 graph break와 recompilation state가 따른다. input shape, stride, dtype guard가 새 kernel을 생성할 수 있다. compile cache revision을 RunID에 넣는다.

performance gate는 kernel parity 뒤 실행한다. launch 수, achieved bandwidth, occupancy만 아니라 valid target/update당 시간을 측정한다.

**activation checkpoint와 RNG 재계산 계약**

activation checkpoint는 forward activation을 덜 저장하고 backward에서 구간 forward를 재실행한다. parameter memory를 줄이지 않으며 compute를 늘린다.

dropout이 있으면 원 forward와 recompute가 같은 RNG mask를 사용해야 한다. RNG state preserve option과 device generator를 확인한다.

reentrant와 non-reentrant 경로는 graph recording, detached tensor, backward API, early stop semantics가 다를 수 있다. revision source와 upstream test 범위를 읽는다.

checkpoint boundary input 중 gradient가 필요한 tensor가 정확히 연결되는지 본다. frozen embedding과 adapter 조합에서 input grad가 끊길 수 있다.

in-place op와 mutable cache가 recompute 결과를 바꿀 수 있다. forward side effect를 제거하거나 checkpoint 밖으로 둔다.

mixed autocast context가 recompute에도 동일하게 적용되는지 본다. forward BF16, recompute FP32처럼 달라지면 gradient가 달라진다.

distributed collective를 checkpointed region에 넣으면 recompute가 collective를 재실행할 수 있다. 모든 rank control flow와 engine semantics를 검증한다.

fixture는 dropout 있는 two-layer network를 checkpoint on/off로 비교한다. loss, RNG state, gradients, one-step delta, peak memory를 기록한다.

**nonfinite 최초 발생을 단계별로 찾는다**

loss가 finite인지 먼저 본다. loss가 NaN이면 backward가 아니라 logits, target, reduction, input부터 조사한다. loss가 finite이고 scaled loss만 inf면 scale이 너무 크다.

backward node별 output gradient finite count와 max를 probe한다. 모든 activation 저장 대신 layer midpoint를 이분한다. probe가 synchronization을 추가하는 점을 기록한다.

raw scaled grad가 inf이고 unscaled도 inf면 overflow다. raw scaled는 finite인데 unscale 후 NaN이면 scaler state나 memory corruption을 본다.

global norm 계산만 inf면 square accumulation dtype과 매우 큰 finite gradient를 본다. FP32 sum-square overflow를 FP64 reference와 비교한다.

clip 뒤 NaN이면 zero norm division, inf coefficient, foreach/fused bug를 본다. optimizer moment에서 최초 NaN이면 epsilon, state dtype, step, stale state를 본다.

parameter update 뒤 최초 nonfinite면 learning rate, decay, master cast, kernel을 본다. 한 rank만 발생하면 found-inf collective와 replica checksum을 본다.

AMP scale을 낮춰 사라져도 root가 data outlier인지 kernel인지 구분한다. offending BatchID와 selected tensor를 보존한다.

NaN batch를 조용히 skip하면 sample measure와 scheduler가 달라진다. 명시적 policy와 counter를 둔다. 반복되면 release를 막는다.

## 2.12 finite difference에서 resume 판정까지

마지막 대절은 설명을 독립 검증으로 바꾼다. operator별 VJP를 유한차분으로 반증하고, gradient 소유권과 compiled backward를 작은 fixture로 대조한 뒤, checkpoint에서 재개한 첫 update가 같은 effect를 만드는지 판정한다.

### 2.12.1 checkpoint·GoldenUpdateRun·장애 판정을 교차 검증한다

**checkpoint를 update transaction의 commit record로 본다**

optimizer step 전 checkpoint는 old parameter와 accumulated grad를 어떻게 다룰지 명확해야 한다. 일반적으로 completed optimizer step 경계에서 저장한다.

checkpoint에는 parameter, optimizer moments/step, scheduler, scaler, RNG, sampler cursor, accumulation position, reducer/communication hook state가 필요하다.

partial accumulation을 저장한다면 `.grad` buffers와 no_sync window, global count도 저장해야 한다. 그렇지 않으면 window를 abort하고 cursor를 안전 boundary로 되돌린다.

분산 shard checkpoint에는 global logical tensor mapping과 topology metadata를 기록한다. world-size 변경 resume에서 reshard를 시험한다.

save completion은 모든 shard의 generation과 manifest가 atomic하게 게시될 때다. 일부 rank file만 새롭고 나머지가 옛 generation인 cut을 거부한다.

clean process load 뒤 next BatchID, loss, raw/unscaled grad, collective count, optimizer delta를 uninterrupted run과 비교한다.

overflow 직전/직후, scheduler boundary, accumulation remainder, bucket rebuild 이후 checkpoint를 failure injection한다.

weight-only artifact는 inference에는 쓸 수 있지만 training resume checkpoint가 아니다. 이름과 lineage를 분리한다.

**한 update의 source-state-test 인수표**

loss source는 1장의 selected callable과 denominator다. state는 scalar/dtype/count다. test는 manual numerator와 selected logit gradient다.

autograd source는 engine/node/backward formulas다. state는 graph edges, saved tensors, version counters다. test는 FP64 VJP와 gradcheck다.

AMP source는 autocast policy와 GradScaler functions다. state는 op dtype, scale, found-inf, stage다. test는 underflow/overflow와 synchronized skip이다.

accumulation source는 training loop와 backward scaling이다. state는 microstep, numerator/count, grad buffers다. test는 concatenated reference다.

DDP source는 reducer와 no_sync path다. state는 buckets, ready order, collective factor다. test는 unequal ranks와 replica checksum이다.

clipping source는 norm calculation과 tensor grouping이다. state는 raw norm, coefficient, clipped grad다. test는 hand global norm이다.

optimizer source는 selected SGD/AdamW/foreach/fused function이다. state는 groups, moments, step, master weight다. test는 one-element replay다.

CUDA source는 dispatched kernel and build revision이다. state는 stream, accumulator dtype, workspace다. test는 eager oracle, sanitizer, tolerance다.

checkpoint source는 save/load connector다. state는 generation, shards, cursor, RNG다. test는 uninterrupted next-step parity다.

한 행이라도 source만 있고 runtime state가 없거나, state만 있고 oracle test가 없으면 미검증이다. option echo나 successful exit code는 update correctness 증거가 아니다.

**3장에 넘기는 UpdateEnvelope**

`UpdateEnvelope`에는 LossEnvelope parent, AutogradGraphID, AccumulationWindowID, CollectiveLedgerID, OptimizerStateID, CheckpointID를 기록한다.

microstep별 BatchID, loss numerator/count, scale, backward completion을 넣는다. selected parameters의 raw/scaled/unscaled/clipped gradients를 넣는다.

collective에는 bucket membership, sequence, sum/average factor, rank counts를 넣는다. optimizer에는 groups, LR, decay, moments, step과 applied delta를 넣는다.

transaction 결과는 committed, skipped-overflow, aborted-error 중 하나다. scheduler와 sampler가 어느 상태까지 전진했는지 함께 기록한다.

3장은 이 update가 memory capacity와 throughput 조건에서 지속 가능한지 분석한다. 수학적으로 맞지만 memory peak나 communication hang이 있는 경로는 운영 가능한 update가 아니다.

최종 합격은 FP64 tiny oracle, mixed precision tolerance, unequal-count distributed equivalence, optimizer replay, checkpoint resume, injected failure 검출을 모두 요구한다.

**GoldenUpdateRun 완전 기록**

FP64 tiny network는 shared parameter, branch, view, normalization, nonlinear activation을 포함한다. 1장의 fixed logits loss를 입력으로 사용한다. 모든 parameter와 intermediate gradient를 손계산 또는 finite difference로 비교한다.

microbatch counts는 1, 3, 8로 둔다. numerator/global-count backward와 concatenated reference를 비교한다. naive microbatch mean 평균을 expected failure로 둔다.

FP32 run은 oracle과 가까워야 한다. BF16 run은 사전 tolerance 안이어야 한다. FP16 run은 scaler off/on, overflow/backoff를 포함한다.

autocast op별 actual dtype을 기록한다. loss scale, scaled loss, raw scaled grad, unscaled grad, found-inf를 기록한다. clipping 전후 global norm과 coefficient를 기록한다.

두 ranks에 counts를 불균등 배치한다. no_sync 중간 microstep과 final sync를 trace한다. bucket membership, collective order, division factor, replica checksum을 기록한다.

rank 하나에 overflow를 주입한다. 모든 ranks가 optimizer와 scheduler를 함께 skip해야 한다. parameter와 moment checksum이 동일하게 남아야 한다.

SGD, momentum, AdamW 각각 selected parameter 원소를 재생한다. m/v, bias correction, epsilon, decay, step, LR로 post-weight를 손계산한다.

foreach, fused, capturable 경로를 eager reference와 비교한다. actual dispatch와 fallback을 기록한다. CUDA graph capture replay의 pointer와 state 안정성을 검사한다.

activation checkpoint on/off에서 dropout RNG, gradients, delta를 비교한다. peak memory와 recompute time을 측정한다. side-effect가 있는 고장 function을 주입한다.

step 직후 checkpoint를 저장한다. clean process에서 next batch, loss, gradient, collective, optimizer delta를 재현한다. world size 변경 reshard도 별도 시험한다.

GoldenUpdateRun은 scalar 한 개가 아니라 update transaction 전체다. source functions, runtime states, oracle assertions, profiler trace, artifact hashes를 하나의 ID에 묶는다.

**backward 최초 차이 질문 백 개**

loss tensor가 같은가. dtype이 같은가. device가 같은가. backward seed가 같은가. scale 전 값이 같은가. count가 같은가.

graph node가 같은가. edge가 같은가. saved tensor가 같은가. version counter가 같은가. view stride가 같은가. alias가 같은가.

parameter가 leaf인가. requires-grad가 맞는가. unused인가. grad가 None인가. zero인가. stale buffer가 있는가.

branch contribution이 모두 합쳐졌는가. hook이 grad를 바꾸는가. custom backward가 맞는가. gradcheck가 통과하는가. higher-order가 필요한가.

autocast가 켜졌는가. op dtype이 맞는가. accumulator dtype이 맞는가. master weight가 있는가. cast-back이 맞는가.

scale이 맞는가. growth tracker가 맞는가. found-inf가 맞는가. unscale이 한 번인가. optimizer stage가 맞는가. overflow가 synchronized인가.

microstep index가 맞는가. window count가 맞는가. remainder가 맞는가. zero-valid 처리가 맞는가. global denominator가 맞는가.

no_sync 범위가 맞는가. final sync가 있는가. bucket이 ready인가. unused traversal이 맞는가. collective sequence가 같은가. division factor가 맞는가.

communication hook이 같은가. hook state가 복원됐는가. error feedback이 같은가. reduction 완료 전에 step하지 않는가. replica checksum이 같은가.

raw grad가 finite인가. scaled grad가 finite인가. unscaled grad가 finite인가. norm sum이 finite인가. clip coefficient가 finite인가.

global norm인가. per-group norm인가. threshold 단위가 같은가. scaled grad를 clip하지 않았는가. shard norm이 reduce됐는가.

parameter group이 같은가. alias가 중복됐는가. frozen parameter가 빠졌는가. weight decay 제외가 맞는가. LR가 맞는가.

momentum buffer가 맞는가. first moment가 맞는가. second moment가 맞는가. step counter가 맞는가. bias correction이 맞는가. epsilon 위치가 맞는가.

decoupled decay인가. update 순서가 맞는가. foreach grouping이 맞는가. fused kernel이 선택됐는가. fallback이 있었는가.

CUDA stream이 맞는가. event dependency가 있는가. launch error가 지연됐는가. race가 있는가. deterministic policy가 같은가.

checkpoint boundary가 commit 후인가. 모든 shard generation이 같은가. optimizer state가 있는가. scaler가 있는가. RNG가 있는가. cursor가 있는가.

accumulation position이 복원됐는가. grad buffer가 복원됐는가. reducer mapping이 복원됐는가. topology reshard가 맞는가.

next BatchID가 같은가. next loss가 같은가. next gradient가 같은가. next delta가 같은가. scheduler가 같은가. sampler가 같은가.

eager와 compiled가 같은가. checkpoint on/off가 같은가. single/distributed가 같은가. source/runtime branch가 같은가. tolerance가 사전 선언됐는가.

최초 nonfinite가 어디인가. 최초 replica divergence가 어디인가. 최초 state skip이 어디인가. 복구가 한 변수인가. 회귀 시험이 추가됐는가.

**update 옵션 변경 승인표**

batch size 변경은 denominator, noise, communication을 바꾼다. global tokens와 update count를 맞춰 비교한다.

accumulation steps 변경은 commit 빈도와 scaler growth, scheduler를 바꾼다. remainder와 no_sync를 다시 시험한다.

precision 변경은 op dtype, accumulator, scaler, tolerance를 바꾼다. loss curve만 비교하지 않는다.

checkpointing 변경은 RNG, recompute, memory를 바꾼다. gradient parity와 side effect를 시험한다.

DDP option 변경은 graph traversal, bucket, reduction을 바꾼다. unequal-count oracle과 hang injection을 실행한다.

communication hook 변경은 gradient 값과 state를 바꾼다. error feedback checkpoint와 full-precision reference를 본다.

clipping 변경은 parameter-space update 방향을 바꾼다. raw norm distribution과 selected delta를 비교한다.

optimizer 변경은 moment geometry, decay, state memory를 바꾼다. 동일 gradient에서 one-step과 multi-step trajectory를 비교한다.

foreach/fused 변경은 dispatch와 rounding을 바꾼다. parity 뒤 성능을 측정한다. silent fallback을 실패로 분류한다.

CUDA graph 변경은 pointer와 dynamic state를 제한한다. overflow, scheduler, variable shape 처리 범위를 명시한다.

compile 변경은 graph break, fusion, cache를 바꾼다. eager oracle과 backward node coverage를 비교한다.

world size 변경은 sampler, denominator, optimizer shard를 바꾼다. global sample ledger와 resume를 검증한다.

새 option은 config→runtime object→tensor/state→effect→failure→recovery를 한 행으로 작성한다. config echo에서 멈추지 않는다.

승인은 GoldenUpdateRun, performance, held-out training behavior, checkpoint resume가 모두 통과해야 한다. 미검증 backend는 지원 범위 밖으로 표시한다.

**update effect 승인 조건**

봉인 파일에는 LossEnvelope parent, source commits, build hashes, resolved options, tensor probes, collective ledger, optimizer replay, checkpoint manifest를 담는다.

독립 검토자는 sample에서 loss, graph edge, parameter gradient, bucket, clip, moment, delta까지 정방향으로 걷는다. post-weight에서 moment, gradient, loss contribution, sample까지 역방향으로 걷는다.

overflow skip과 exception abort도 정상 transaction처럼 기록한다. 아무 state도 부분 commit되지 않았음을 증명한다. scheduler와 cursor를 포함한다.

성능 수치는 correctness artifact를 parent로 참조한다. 빠르지만 다른 gradient를 만드는 kernel을 같은 run의 최적화로 부르지 않는다.

3장 handoff에는 peak activation, optimizer state memory, communication bytes, recompute FLOPs와 update latency를 더한다. memory와 throughput 분석이 exact update를 기준으로 하게 한다.

최종 판정은 같은 sample measure, 같은 loss seed, 같은 global gradient, 정확히 한 optimizer commit, 재현 가능한 next step이다. 이 다섯 조건이 닫히면 2장이 끝난다.

**상태별 최소 관측값 사전**

loss 상태: scalar, numerator, count, dtype, device, scale before backward.

graph 상태: root, node IDs, edges, saved tensors, version counters, retain flags.

leaf 상태: parameter ID, requires-grad, alias, grad None/zero, accumulator.

AMP 상태: autocast policy, op dtype, scaler value, growth tracker, found-inf.

accumulation 상태: window ID, microstep, consumed batches, global count, grad buffers.

reducer 상태: bucket map, ready order, no-sync, collective sequence, division factor.

clipping 상태: raw norm, global sum-square, threshold, coefficient, clipped norm.

optimizer 상태: groups, LR, decay, moments, step, master weights, dispatch.

kernel 상태: operation, device function, stream, accumulator, workspace, build hash.

checkpoint 상태: generation, shards, RNG, cursor, scheduler, scaler, topology.

transaction 상태: started, backward-complete, reduced, unscaled, clipped, committed/skipped.

검증 상태: FP64 oracle, tolerance, selected probes, expected failure, verdict.

각 state transition에는 owner function이 있다. config key가 owner가 아니다. instantiated callable과 runtime branch를 기록한다.

각 tensor에는 logical ID가 있다. Python name이 바뀌어도 alias와 shard를 추적한다.

각 collective에는 sequence number가 있다. rank별 trace가 같은 순서를 가져야 한다.

각 optimizer commit에는 parent checkpoint와 consumed BatchIDs가 있다. 중복 소비와 누락을 막는다.

각 skip에는 reason이 있다. overflow, zero count, explicit policy, exception을 구분한다.

각 recovery에는 new RunID가 있다. bad state를 덮어쓰지 않는다.

**장애별 즉시 판정**

loss가 다르면 1장으로 돌아간다. loss가 같고 logit grad가 다르면 loss backward를 본다.

last head grad부터 다르면 CE backward, dtype, scale을 본다. 중간 layer부터 다르면 autograd edge와 saved tensor를 본다.

한 branch contribution이 없으면 detach, conditional, unused parameter를 본다. zero grad를 정상 수렴으로 오판하지 않는다.

grad가 microstep마다 작아지면 accumulation 자동 division을 본다. counts가 다른 window를 사용한다.

FP16에서만 0이면 underflow와 scale을 본다. BF16에서만 불안정하면 mantissa와 reduction을 본다.

unscale 뒤만 이상하면 GradScaler stage와 optimizer ownership을 본다. 두 번 unscale하지 않는다.

clip 뒤만 이상하면 global norm dtype과 coefficient를 본다. scaled gradient clip 여부를 확인한다.

한 rank만 다르면 no_sync, count, overflow collective, unused branch를 본다.

hang이면 collective sequence와 bucket readiness를 본다. timeout 재시작 전에 rank traces를 보존한다.

SGD는 맞고 AdamW만 다르면 moment, step, bias correction, epsilon, decay를 본다.

eager optimizer는 맞고 fused만 다르면 dispatch, grouping, state dtype을 본다.

첫 step은 맞고 resume만 다르면 moments, scaler, RNG, cursor, scheduler를 본다.

checkpoint on에서만 다르면 RNG replay, autocast context, side effect, detached input을 본다.

compile에서만 다르면 graph break, custom Function, mutation, guards를 본다.

CUDA에서만 다르면 kernel accumulator, race, stream dependency, unsupported layout을 본다.

parameter는 finite인데 moment가 NaN이면 optimizer state update를 이분한다.

parameter update 뒤 replica가 갈라지면 synchronized skip과 all-reduce completion을 본다.

scheduler만 앞서면 skip transaction atomicity를 고친다. LR 보정으로 숨기지 않는다.

sampler만 앞서면 abort/rollback cursor 정책을 고친다. sample 중복을 ledger로 찾는다.

복구는 tiny fixture, one-step distributed, resume, long canary 순서로 확대한다.

**source 검토 체크**

public API signature를 읽는다. caller를 읽는다. selected dispatch를 읽는다. backward formula를 읽는다. state mutation을 읽는다. error path를 읽는다.

upstream test assertion을 읽는다. 빠진 integration 경계를 적는다. local fixture로 채운다. 실행 branch를 trace한다.

commit과 function hash를 저장한다. binary build가 source와 같은지 확인한다. compiler, CUDA, NCCL revision을 저장한다.

default option을 resolved value로 바꾼다. environment override를 저장한다. fallback warning을 failure로 승격할지 정한다.

source가 보증하는 범위와 관측된 runtime을 분리한다. 성능 claim과 수치 claim을 분리한다. single-device와 distributed claim을 분리한다.

**장간 state 연결**

1장의 LossEnvelope가 update의 부모다. 3장은 activation, optimizer, communication memory를 계산한다. 11장은 AdamW 세부 geometry를 확장한다.

12장은 다른 optimizer의 parameter-space 효과를 비교한다. 15장은 tensor/pipeline/data parallel collective를 확장한다. 17장은 checkpoint 세대를 확장한다.

18장은 adapter parameter만 optimizer group에 넣는 경로를 사용한다. 19장은 pair/token denominator를 사용한다. 20장은 online rollout update 원자성을 사용한다.

각 장은 UpdateEnvelope의 parameter IDs, global denominator, collective ledger, commit status를 직접 부모로 참조한다.

최종 독자는 한 sample의 loss contribution이 어느 parameter delta에 기여했는지 추적하고, 한 parameter delta를 sample·gradient·collective·moment로 역추적할 수 있어야 한다.

이 양방향 추적과 next-step resume가 통과하면 backprop 한 step은 설명이 아니라 재현 가능한 transaction이 된다.

### 2.12.2 Jacobian 없는 역전파를 operator별 기하로 이해한다

**독립 재현 시험**

독립 검토자는 sealed LossEnvelope와 checkpoint를 받는다. 작성자의 메모 없이 next BatchID를 예측하고 실제 sampler cursor와 대조한다.

backward 전 parameter, optimizer moments, scaler, RNG checksum을 기록한다. backward 후 raw gradient, unscale 후 gradient, clip 후 gradient를 selected elements에서 기록한다.

두 ranks의 local numerator/count와 gradient contribution을 공개하기 전에 global expected를 FP64로 계산한다. collective 뒤 값이 expected와 맞는지 본다.

optimizer step은 selected parameter에 대해 손으로 재생한다. LR, decay, moments, bias correction, epsilon, clip coefficient를 사용해 post-weight를 계산한다.

overflow variant에서는 parameter, moments, step, scheduler가 모두 그대로여야 한다. 한 state라도 전진하면 transaction atomicity 실패다.

exception variant에서는 partial accumulation과 reducer state를 폐기하거나 완전히 복원해야 한다. 한 rank만 재시작하지 않는다.

commit 뒤 새 checkpoint를 clean process에서 load한다. 다음 loss, gradient, collective, delta가 uninterrupted run과 맞아야 한다.

eager, fused, compiled 경로는 같은 oracle을 사용한다. 성능은 parity 통과 뒤 비교한다. fallback은 trace에 명시한다.

최종 report는 최초 divergence, state owner, source function, tensor evidence, recovery test를 포함한다. “seed 차이”나 “수치 오차”로 근거 없이 닫지 않는다.

이 시험을 통과한 UpdateEnvelope만 다음 장이 memory와 throughput 분석의 기준으로 사용한다.

검토자는 마지막으로 두 counterfactual을 실행한다. 첫째, loss scalar는 유지하되 computation graph의 두 branch 기여를 바꾼다. scalar 일치가 parameter gradient 일치를 보증하지 않음을 확인한다. 둘째, global gradient는 유지하되 optimizer moment와 step을 바꾼다. gradient 일치가 update 일치를 보증하지 않음을 확인한다.

또한 한 rank overflow, final no-sync, stale moment, partial checkpoint를 각각 주입한다. 네 오류는 finite loss를 만들 수 있으므로 replica checksum, transaction status, next-step parity가 검출해야 한다.

최종 artifact에는 성공 run과 expected-failure run을 함께 둔다. library upgrade가 failure를 조용히 허용하면 gate regression이다. 수정 뒤 tiny oracle, distributed one-step, clean resume를 다시 실행한다.

이로써 backprop step은 chain rule, dtype, denominator, collective, optimizer, CUDA kernel, checkpoint가 하나의 원자적 update로 닫힌다.

**Jacobian을 만들지 않고 Jacobian을 쓰는 법**

역전파를 “미분을 자동으로 해 주는 기능”이라고만 기억하면 복잡한 모델에서 길을 잃는다. 더 정확한 그림은 이렇다. forward의 각 연산은 입력의 작은 변화가 출력의 작은 변화로 어떻게 옮겨 가는지 나타내는 선형 사상, 곧 Jacobian을 암묵적으로 가진다. backward는 그 거대한 행렬을 메모리에 만들지 않고, 출력 쪽에서 도착한 벡터를 그 Jacobian의 전치에 곱한다. 이것이 vector–Jacobian product, VJP다.

연산 `y=f(x)`에서 `x`가 `n`차원, `y`가 `m`차원이면 Jacobian `J`의 모양은 `[m,n]`이다. 최종 loss `L`이 scalar이면 출력 쪽 adjoint `ȳ=∂L/∂y`는 길이 `m`인 행벡터이고, 입력 adjoint는 `x̄=ȳJ`다. Transformer의 parameter 수가 수십억이어도 full Jacobian을 만들 필요가 없는 까닭이다. scalar 하나에서 출발해 graph를 한 번 거꾸로 훑으면 모든 parameter에 대한 gradient를 동시에 얻는다.

이 관점은 디버깅에도 바로 쓰인다. 어떤 module의 forward 값은 맞는데 입력 gradient가 다르다면 세 후보가 있다. upstream adjoint가 이미 달랐거나, 그 module의 local VJP가 다르거나, 두 기여를 합치는 지점에서 누적이 달랐다. 그러므로 module 경계에서 activation만 비교해서는 backward 오류를 찾을 수 없다. 같은 activation, 같은 upstream gradient, 같은 local VJP라는 세 조건을 따로 확인해야 한다.

예를 들어 `y=Ax+b`의 VJP는 `x̄=Aᵀȳ`, `Ā=ȳxᵀ`, `b̄=ȳ`다. batch가 붙으면 `Ā`와 `b̄`는 batch 축을 합한다. 이 합이 바로 여러 sample이 같은 parameter에 보내는 기여의 총합이다. mean loss는 여기에 분모를 한 번 곱한 결과이지, 각 sample의 목적함수가 사라지는 것이 아니다. sample별 gradient를 보고 싶다면 이 reduction 전의 per-sample VJP를 보존하거나 `vmap` 같은 변환을 이용해야 한다.

JVP는 반대 방향이다. 입력 방향 `ẋ`을 정하고 `ẏ=Jẋ`을 계산한다. parameter 전체의 gradient가 필요한 일반 학습에는 reverse mode가 유리하지만, 특정 perturbation이 출력에 미치는 영향이나 Hessian–vector product를 볼 때 JVP가 유용하다. 한 방향의 finite difference `(f(x+εv)-f(x))/ε`와 JVP를 비교하면 full Jacobian 없이 local linearization을 시험할 수 있다.

**기하학적 독해.** gradient는 “내리막 방향” 그 자체가 아니라 현재 좌표계의 Euclidean inner product 아래에서 loss 증가율을 나타내는 covector를 vector로 식별한 것이다. parameterization이나 metric을 바꾸면 같은 함수라도 update 방향이 달라진다. Adam, Muon, natural gradient가 서로 다른 까닭을 이해하려면 이 구분이 필요하다. 이 장에서는 우선 raw VJP를 정확히 보존하고, 11장과 12장에서 optimizer가 이 정보를 어떤 geometry로 변형하는지 이어 본다.

**방향 미분 fixture.** 무작위 normalized vector `v`를 하나 고르고 analytic 값 `g·v`와 중앙차분 `(L(θ+εv)-L(θ-εv))/(2ε)`을 비교한다. parameter마다 좌표 finite difference를 하면 비용이 parameter 수에 비례하지만 방향 검사는 두 번의 forward로 넓은 오류를 잡는다. 다만 여러 오류가 우연히 한 방향에서 상쇄될 수 있으므로 seed가 다른 방향 여러 개를 쓴다. ε sweep으로 truncation error와 rounding error가 만나는 구간도 확인한다.

**softmax backward는 확률을 다시 정규화한다**

softmax를 원소별 지수 함수처럼 미분하면 틀린다. `p_i=exp(z_i)/Σ_j exp(z_j)`이므로 한 logit의 변화가 모든 확률에 영향을 준다. Jacobian은 `∂p_i/∂z_j=p_i(δ_ij-p_j)`이고, upstream gradient를 `u_i=∂L/∂p_i`라 하면 VJP는 다음처럼 정리된다.

`∂L/∂z_i = p_i (u_i - Σ_j p_j u_j)`

즉 upstream 신호에서 확률 가중 평균을 빼고 다시 `p_i`를 곱한다. 모든 logit에 같은 상수를 더해도 softmax가 변하지 않으므로 gradient 합은 0이다. 이 불변식은 강력한 unit test다. 한 row의 `dz.sum()`이 허용 오차를 크게 벗어나면 reduction 축이나 mask 처리가 잘못되었을 가능성이 높다.

cross entropy와 결합하면 식이 더 단순해진다. one-hot target `y`에 대해 `L=-Σ_i y_i log p_i`이므로 `∂L/∂z=p-y`다. target class에는 `p_t-1`, 나머지에는 `p_i`가 흐른다. label smoothing이면 `y`가 one-hot이 아니라 완만한 target distribution이 되고, class weight나 ignore mask가 있으면 각 row의 기여와 전역 분모가 함께 달라진다.

이 단순한 식이 “fused cross entropy라면 무조건 안전하다”는 뜻은 아니다. kernel은 보통 row maximum, exponent sum, target logit, valid mask를 tile 단위로 처리한다. vocabulary parallel이면 각 rank의 local maximum을 먼저 구한 뒤 global maximum을 합의하고, shifted exponential sum을 다시 collective로 합쳐야 한다. local softmax를 만든 뒤 평균하면 전혀 다른 분포다. backward의 `p-y`도 target token을 소유한 rank만 `-1`을 적용하되 모든 rank가 global denominator를 동일하게 써야 한다.

**세 token 손검산.** logits `[0, log 2, log 3]`의 확률은 `[1/6,2/6,3/6]`이다. target이 세 번째이면 gradient는 `[1/6,2/6,-3/6]`이고 합은 0이다. 두 번째 token을 ignore하면 그 row 전체 gradient가 0이어야 한다. 두 valid row의 mean을 쓰면 각 row는 `1/2`가 곱해진다. 이 작은 fixture 하나가 class 축, target subtraction, ignore mask, denominator를 동시에 드러낸다.

**수치 안정성.** `exp(1000)`을 직접 만들지 않고 row maximum을 빼서 log-sum-exp를 계산한다. maximum을 빼도 softmax가 같은 이유는 분자와 분모에 같은 상수가 소거되기 때문이다. backward에서는 확률이 0이나 1에 가까워 gradient가 작아질 수 있다. 이것은 반드시 underflow bug는 아니다. FP32 reference와 비교해 수학적 saturation인지 낮은 dtype의 조기 rounding인지 분리한다.

**attention softmax의 차이.** attention에서는 causal/padding mask가 금지된 score를 사실상 `-∞`로 만든다. 금지 위치의 probability와 gradient는 0이어야 하고, 완전히 mask된 row의 정책은 구현마다 확인해야 한다. 잘못된 all-masked row가 NaN을 만들 수 있다. fused attention은 probability matrix를 저장하지 않고 row log-sum-exp를 이용해 backward에서 tile별로 재구성하므로 forward의 LSE와 mask 의미가 backward 계약의 일부다.

**정규화 backward를 투영으로 읽는다**

LayerNorm과 RMSNorm은 forward 식보다 backward geometry를 이해할 때 차이가 선명하다. LayerNorm은 feature vector에서 평균 방향을 제거하고 분산으로 크기를 맞춘다. 따라서 입력 gradient는 upstream gradient를 그대로 scale하는 것이 아니라, 평균 방향과 normalized activation 방향의 성분을 제거한 투영에 가깝다.

feature 수를 `D`, `μ=mean(x)`, `r=1/sqrt(var(x)+ε)`, `x̂=(x-μ)r`, `y=γx̂+β`라 하자. `q=ȳ⊙γ`이면 입력 gradient는 다음 꼴로 쓸 수 있다.

`x̄ = (r/D) [Dq - Σq - x̂ Σ(q⊙x̂)]`

첫 번째 합은 상수 방향, 두 번째 합은 normalized activation 방향의 성분을 뺀다. `γ̄=Σ(ȳ⊙x̂)`, `β̄=Σȳ`이며 batch와 token 축을 합한다. constant vector를 입력에 더해도 LayerNorm 출력이 거의 같으므로 이상적인 입력 gradient의 feature 합은 0에 가깝다. ε와 finite precision 때문에 정확한 0을 강제하지는 않되 reference tolerance를 둔다.

RMSNorm은 평균을 빼지 않는다. `r=1/sqrt(mean(x²)+ε)`, `y=γxr`이므로 상수 shift에 불변하지 않다. 입력 gradient에서는 `x` 방향 성분이 조정되지만 상수 방향을 별도로 제거하지 않는다. 두 norm을 이름만 바꿔 끼우고 checkpoint weight를 load하면 forward와 backward가 모두 달라진다. 특히 residual stream의 평균 성분이 다음 layer로 전달되는 방식이 바뀐다.

**ε의 위치.** `sqrt(var+ε)`와 `sqrt(var)+ε`는 다른 함수다. 작은 variance에서 차이가 커지고 backward의 분모 거듭제곱도 달라진다. source를 읽을 때 config의 epsilon 이름만 보지 말고 실제 native/fused kernel 수식을 확인해야 한다. mixed precision에서는 mean과 variance를 어느 dtype으로 누적하는지도 함께 본다.

**작은 fixture.** `[1,2,3,4]` 한 row에 임의 upstream `[1,-1,2,-2]`를 넣어 FP64 수식으로 `dx,dγ,dβ`를 계산한다. constant shift를 더한 LayerNorm 결과와 gradient 불변성을 확인하고, RMSNorm에서는 달라져야 한다는 negative assertion을 둔다. feature가 모두 같은 row, 매우 작은 variance, 큰 magnitude가 섞인 row도 포함한다.

**fused kernel의 저장 선택.** backward에 input 전체 대신 mean과 reciprocal standard deviation, 또는 normalized output을 저장할 수 있다. 어떤 값을 저장하는지는 memory와 재계산량뿐 아니라 rounding 경로를 바꾼다. source 좌표를 남길 때 forward entry point, saved statistics dtype, backward entry point, parameter-gradient reduction kernel을 하나의 묶음으로 기록한다.

**attention backward를 세 경로로 분해한다**

한 head를 `S=QKᵀ/√d`, `P=softmax(S+M)`, `O=PV`로 쓰자. output gradient `G=∂L/∂O`가 들어오면 먼저 `dV=PᵀG`, `dP=GVᵀ`를 얻는다. softmax VJP로 `dS=P⊙(dP-row_sum(P⊙dP))`를 만들고, 마지막으로 `dQ=dSK/√d`, `dK=dSᵀQ/√d`를 계산한다.

이 분해는 장애 위치를 즉시 좁힌다. `dV`만 틀리면 probability 재구성이나 output matmul을 본다. `dV`는 맞고 `dQ,dK`가 틀리면 softmax VJP, mask, scale을 본다. causal mask 위의 `dS`는 0이어야 한다. 한 query row의 `dS` 합 역시 softmax 불변성 때문에 0에 가깝다.

GQA와 MQA에서는 여러 query head가 같은 KV head를 공유한다. forward에서 K/V를 논리적으로 repeat했으면 backward에서 대응 query head들의 `dK,dV`를 원래 KV head로 합쳐야 한다. repeat된 view 각각에 gradient를 남기고 원 storage로 reduction하지 않으면 finite하지만 작은 gradient가 나온다. 반대로 kernel과 wrapper가 둘 다 합치면 head group 수만큼 커진다.

RoPE는 Q와 K에 위치별 회전을 적용한다. 이상적인 2차원 rotation의 transpose가 inverse이므로 backward는 upstream을 반대 방향으로 회전시킨다. sin/cos table의 position index, interleaved/half-split layout, scaling variant가 forward와 같아야 한다. forward output이 우연히 비슷해도 gradient의 pair mapping이 틀릴 수 있으므로 basis vector fixture를 쓴다.

Flash 계열 attention은 `P` 전체를 HBM에 쓰지 않는다. backward는 Q, K, V, O, row LSE와 upstream gradient를 tile로 읽어 probability를 재구성한다. 이때 핵심은 “같은 수학”과 “같은 비트”를 구분하는 것이다. tile 크기, reduction 순서, accumulator dtype, atomic 사용에 따라 작은 차이는 생길 수 있다. 하지만 mask된 위치의 0, GQA reduction, scale factor, dropout RNG 같은 의미적 불변식은 tolerance로 넘길 문제가 아니다.

dropout이 attention probability에 적용되면 backward는 forward와 정확히 같은 mask와 scaling을 써야 한다. activation checkpoint가 attention forward를 재실행할 때 RNG state를 보존하는 이유다. seed만 같아도 kernel launch shape나 counter offset이 달라지면 mask가 달라질 수 있다. fixture에는 seed뿐 아니라 RNG algorithm, offset, tensor shape, dropout probability, selected backend를 기록한다.

**2×3 fixture.** query 두 개, key 세 개, head dimension 둘인 아주 작은 tensor를 만든다. 첫 query는 마지막 key를 causal mask로 금지한다. FP64로 S, P, O, dP, dS, dQ, dK, dV를 모두 저장한다. eager composition, framework SDPA, fused backend를 각각 비교한다. forward O만 아니라 여섯 중간 불변식과 세 입력 gradient를 비교해야 backend 변경의 영향 반경을 알 수 있다.

**residual과 tied weight는 gradient의 합류점이다**

Transformer block의 residual을 `y=x+F(x)`라 쓰면 입력 gradient는 `x̄=ȳ+J_Fᵀȳ`다. 하나는 identity 경로, 다른 하나는 attention이나 MLP 경로다. residual이 깊은 network의 최적화를 돕는 직관은 gradient가 매 block의 복잡한 Jacobian만 연속으로 통과하지 않고 identity 경로를 가진다는 데 있다. 그렇다고 gradient가 그대로 보존된다는 뜻은 아니다. 두 경로가 방향상 상쇄하거나 증폭할 수 있고, pre-norm과 post-norm은 Jacobian의 곱 순서를 바꾼다.

residual 디버깅 fixture는 branch output을 0으로 만드는 것에서 시작한다. 그러면 `dy/dx`는 identity여야 한다. 다음에는 linear branch `F(x)=Wx`를 넣어 `dx=dy+Wᵀdy`를 손으로 계산한다. in-place add, detached branch, stochastic depth mask, tensor-parallel collective를 한 번에 넣지 않는다.

embedding과 LM head가 weight를 공유하는 tied model에서는 같은 storage가 두 graph 위치에서 사용된다. embedding lookup 경로는 선택된 token row에 sparse한 기여를 보내고, LM head 경로는 모든 vocabulary row에 dense한 기여를 보낼 수 있다. 최종 `.grad`는 두 기여의 합이다. 두 이름이 같은 `Parameter` object를 가리키는지, 서로 다른 object가 storage만 공유하는지, checkpoint load가 alias를 보존하는지에 따라 optimizer 동작이 달라질 수 있다.

검증할 때 세 run을 만든다. 첫째 embedding 경로만 살려 `g_embed`를 얻는다. 둘째 LM head 경로만 살려 `g_head`를 얻는다. 셋째 전체 graph에서 `g_total`을 얻는다. `g_total≈g_embed+g_head`가 되어야 한다. optimizer parameter list에는 shared weight가 한 번만 있어야 한다. 두 번 들어가면 moment와 decay가 두 번 적용될 수 있다.

adapter도 합류점을 만든다. LoRA linear를 `Wx+sBAx`로 쓰면 frozen base W에는 gradient가 없더라도 input gradient는 base와 adapter 두 경로의 합이다. A나 B를 0으로 초기화하는 방식에 따라 첫 step에 어느 factor가 gradient를 받는지가 달라진다. “trainable parameter 수가 맞다”는 검사만으로 adapter gradient plumbing을 증명할 수 없는 이유다.

MoE에서는 router가 선택한 expert 경로와 auxiliary loss 경로가 합쳐진다. top-k dispatch가 discrete하여 선택 index 자체에는 보통 직접 gradient가 없더라도 selected gate weight와 router objective에는 gradient가 흐른다. dropped token, capacity overflow, load-balancing denominator를 포함해 어떤 branch가 graph에 남는지 확인해야 한다. 9장은 이 구조를 forward 관점에서, 21장은 학습 objective 관점에서 확장한다.

**PyTorch autograd를 source에서 읽는 순서**

PyTorch에서 `loss.backward()` 한 줄을 따라갈 때 Python 함수 하나만 읽어서는 부족하다. 공개 API는 backward할 tensor와 optional gradient를 정규화하고, C++ execution engine으로 GraphTask를 넘긴다. graph의 node는 각 연산이 만든 backward function이고 edge는 다음 node와 input slot을 가리킨다. leaf parameter에는 `AccumulateGrad`가 연결되어 최종 기여를 `.grad`에 누적한다.

많은 native operation의 derivative 식은 손으로 작성된 Python backward가 아니라 `tools/autograd/derivatives.yaml`과 code generation 경로에서 정의된다. 따라서 operator를 조사할 때는 네 층을 묶는다. native schema와 forward implementation, derivative formula, 생성된 backward node, 그 식을 검증하는 test다. composite operation이면 다른 differentiable operation으로 분해되어 derivative가 자동 구성될 수 있고, custom autograd Function이면 저자가 직접 backward를 책임진다.

현재 보존한 PyTorch snapshot에서는 `torchgen/api/autograd.py`가 derivative metadata를 해석하는 경로를 보여 준다. 이것은 runtime backward 자체가 아니라 build-time code generation 층이다. runtime의 ready queue와 GraphTask, generated function, leaf accumulation을 뒤섞지 않는 것이 중요하다. source 인용에는 repository revision과 file path뿐 아니라 symbol, 주변 line, 호출자, runtime에서 실제 선택된 backend를 같이 남긴다.

`grad_fn`은 non-leaf tensor가 어떤 backward node에서 만들어졌는지 보여 주고 `next_functions`는 다음 edge를 탐색하는 단서를 준다. 그러나 이를 production hot path에서 전부 순회하며 logging하면 overhead가 커진다. tiny fixture에서 graph topology를 덤프하고, 실제 run에서는 선택한 경계의 hook과 profiler event만 남긴다.

`torch.autograd.grad`와 `.backward()`도 목적이 다르다. 전자는 지정한 input에 대한 gradient를 반환하며 기본적으로 leaf `.grad` 누적을 피할 수 있다. 후자는 graph를 따라 leaf accumulator까지 기여를 보낸다. per-sample gradient나 gradient penalty처럼 graph를 다시 미분할 때 `create_graph` 의미를 명확히 해야 한다. 불필요한 `retain_graph=True`는 correctness fix가 아니라 lifetime 누수를 숨기는 경우가 많다.

`no_grad`는 그 context에서 새 연산의 reverse-mode graph 기록을 끄지만 parameter의 `requires_grad` 자체를 영구 변경하지 않는다. `inference_mode`는 더 강한 최적화를 적용하고 version/view tracking 제약이 다르므로 학습 graph로 되돌려 쓰는 tensor의 경계를 조심한다. `detach`는 같은 storage를 공유할 수 있는 graph 단절이다. 값을 복사했다는 뜻이 아니므로 detached view의 mutation도 별도 문제를 만들 수 있다.

**source 검토 질문.** 어느 함수가 tensor를 저장하는가? saved tensor는 원본인가 output인가 통계인가? version counter는 언제 검사되는가? undefined gradient를 0으로 취급하는가? higher-order gradient가 가능한가? sparse, complex, noncontiguous input은 지원하는가? CUDA와 CPU가 같은 formula를 쓰는가? deterministic mode에서 backend가 바뀌는가? 이 질문에 답하지 못하면 “PyTorch autograd가 처리한다”는 말은 증거가 아니다.

**CUDA stream 위에서 backward가 실행될 때**

CUDA 연산은 host 호출이 끝났다고 계산이 끝난 것이 아니다. kernel은 stream에 enqueue되고, 같은 stream 안의 순서는 보존되지만 서로 다른 stream 사이에는 event나 명시적 dependency가 필요하다. autograd engine은 forward operation이 사용한 stream 의미를 존중하며 backward work를 배치해야 한다. custom Function이 별도 stream에서 work를 내보내고 lifetime이나 event를 잘못 관리하면 CPU에서는 재현되지 않는 race가 생긴다.

gradient가 finite하다는 사실은 race가 없다는 증거가 아니다. 이전 step의 buffer 일부가 섞여도 값은 finite할 수 있다. 동일 seed 반복에서 checksum이 간헐적으로 달라지거나 profiler timeline에서 consumer가 producer보다 먼저 실행되면 stream dependency를 의심한다. 디버그 목적으로 전역 synchronize를 넣어 문제가 사라질 수 있지만, 그것은 원인을 고친 것이 아니라 race를 가린 위치 단서다.

`AccumulateGrad`가 여러 branch의 기여를 받을 때 CUDA add의 reduction 순서도 비트 수준 결과에 영향을 준다. atomic reduction은 실행 순서가 고정되지 않을 수 있다. deterministic algorithm을 요청하면 느린 kernel이나 다른 implementation이 선택되거나 unsupported 오류가 날 수 있다. 재현성 계약에는 “같은 수학적 결과”, “tolerance 안의 결과”, “bitwise 동일” 중 무엇을 요구하는지 적는다.

CUDA graph capture는 allocation 주소와 launch topology가 안정적이어야 한다. backward graph, gradient buffer, optimizer state가 capture마다 달라지면 replay가 성립하지 않는다. `set_to_none=True`로 gradient storage를 매번 새로 만들지, 고정 buffer를 재사용할지에 따라 capture 설계가 바뀐다. warmup에서 만들어진 autograd node와 stream association도 실제 capture run과 같은지 확인한다.

NCCL collective는 계산 stream과 communication stream 사이 dependency를 더한다. gradient bucket이 ready되면 all-reduce를 enqueue하고 optimizer가 그 결과를 읽기 전에 completion을 기다려야 한다. overlap은 이 기다림을 늦추는 것이지 없애는 것이 아니다. profiler에서 kernel 시간이 겹친다는 사실과 parameter가 올바른 reduced gradient를 읽는다는 correctness를 별도 검증한다.

**race fixture.** 두 stream에서 producer와 consumer를 실행하되 event를 의도적으로 제거한 expected-failure test를 둔다. 반복 checksum이 흔들리는지 확인하고, 올바른 event를 넣은 뒤 안정화되는지 본다. 이 fixture는 실제 모델 전체보다 작아야 하며 compute-sanitizer나 framework anomaly 도구를 쓸 때도 동일 입력을 유지한다.

**한 operator의 backward를 승인하는 표준 절차**

새 fused operation이나 custom CUDA extension을 학습 경로에 넣을 때 forward benchmark부터 보는 습관을 버린다. 먼저 의미 계약을 적는다. 입력 shape·stride·dtype·device, broadcasting, mask, reduction, empty tensor, non-finite, deterministic behavior, 필요한 gradient input을 명시한다.

그다음 FP64 compositional reference를 만든다. production kernel과 독립적인 primitive 조합이어야 한다. production과 reference가 같은 helper를 공유하면 같은 bug를 공유할 수 있다. 작은 hand-computable case, random typical case, 극단값, noncontiguous view, boundary shape, expected error를 구분한다.

forward parity 뒤에는 scalar objective를 붙여 모든 differentiable input의 VJP를 비교한다. upstream gradient를 all-ones 하나만 쓰지 않는다. all-ones는 softmax 같은 연산에서 특수하게 0을 만들 수 있다. seed가 고정된 비대칭 upstream을 여러 개 쓴다. parameter gradient뿐 아니라 input gradient도 본다.

`gradcheck`는 double precision 중앙차분으로 강한 신호를 주지만 만능은 아니다. stochastic operation은 RNG를 고정해야 하고, nondifferentiable boundary는 피하거나 subgradient 계약을 정해야 한다. overlapping storage나 sparse output에는 별도 제약이 있다. 낮은 dtype kernel은 FP64 gradcheck 경로와 다른 dispatch를 탈 수 있으므로 실제 BF16/FP16에서 FP32 reference와 tolerance 비교를 추가한다.

두 번째 derivative가 필요한 gradient penalty, meta-learning, 일부 optimizer를 지원한다면 `gradgradcheck`나 명시적 Hessian–vector fixture를 둔다. custom backward 내부에서 tensor를 detach하거나 non-differentiable native kernel만 쓰면 첫 backward는 맞아도 higher-order graph가 끊긴다. 지원하지 않는다면 silent zero 대신 명확히 문서화하고 오류를 내는 편이 낫다.

성능 승인은 correctness matrix 이후다. warmup, synchronization, input reuse, allocator 상태를 통제하고 forward-only와 forward+backward를 나눠 잰다. 저장 activation bytes, recompute FLOPs, temporary workspace, kernel launch 수, HBM traffic을 함께 본다. forward가 빨라도 backward workspace가 커서 전체 batch size를 줄이면 학습 throughput은 나빠질 수 있다.

마지막으로 fallback을 시험한다. unsupported shape가 reference path로 명시적으로 돌아가는지, 조용히 잘못된 kernel을 타는지 확인한다. compile, autocast, activation checkpoint, DDP/FSDP wrapper 아래에서도 같은 operation이 선택되는지 trace한다. version upgrade 뒤에는 성공 fixture와 expected-failure fixture를 모두 실행한다.

**loss에서 parameter delta까지 한 줄로 추적한다**

이 장의 모든 설명을 한 문장으로 압축하면 “어느 sample의 어느 valid token이 만든 logit 오차가 어느 graph edge와 reduction을 지나 어느 parameter의 어느 stateful update가 되었는가”다. 이 질문에 답하려면 평균 loss 하나가 아니라 lineage가 필요하다.

한 golden token을 고른다. 그 token의 target, valid flag, unreduced negative log-likelihood, global denominator를 기록한다. selected vocabulary logits와 `p-y`를 손으로 계산한다. LM head를 거쳐 hidden gradient와 weight row 기여를 계산한다. 마지막 block에서 attention·MLP·residual 세 경로를 분리해 upstream VJP가 합쳐지는 것을 본다. embedding까지 내려가 tied contribution을 합친다.

그 뒤 local gradient가 accumulation window의 어느 microstep에 더해졌는지 표시한다. rank-local numerator와 count, DDP reduction, unscale, global norm과 clip coefficient를 기록한다. selected parameter 한 원소에 대해 raw gradient, moment 전 값, bias correction, decay, learning rate, post-update 값을 손으로 재생한다.

이 추적은 모든 원소를 dump하라는 뜻이 아니다. 전체 tensor에는 shape, dtype, finite count, norm, checksum을 남기고, 고정된 selected index 몇 개에만 값을 남긴다. index는 값이 큰 원소만 사후 선택하지 말고 run 전에 고정한다. 그렇지 않으면 실패를 보고 유리한 증거를 고르는 selection bias가 생긴다.

반대 방향 추적도 한다. checkpoint의 한 parameter delta에서 시작해 어느 optimizer group과 moment가 만들었는지, 어떤 clipped gradient가 들어왔는지, 어느 ranks와 microbatches가 기여했는지, 어느 loss denominator와 sample IDs였는지 거슬러 올라간다. forward lineage와 backward lineage가 같은 RunID, BatchID, UpdateID에서 만날 때 비로소 update가 설명 가능하다.

독자는 이 절차를 거친 뒤 loss가 감소한다는 사실만으로 학습이 맞다고 결론내리지 않는다. 잘못된 label shift도 loss를 낮출 수 있고, 누락된 rank도 안정적인 곡선을 만들 수 있으며, stale optimizer state도 finite update를 만든다. 정확성은 결과 곡선과 더불어 상태 전이의 provenance로 증명한다.

**독자가 직접 만드는 다섯 개의 역전파 실험**

첫 실험은 scalar branch다. 하나의 변수에서 두 경로가 갈라졌다 합쳐지는 함수를 만들고 analytic, autograd, finite difference를 비교한다. 한 branch를 detach한 expected-failure variant를 추가한다. forward 값이 같아도 gradient가 달라지는 입력을 찾아 설명한다.

둘째는 softmax와 cross entropy다. vocabulary 세 개, token 두 개로 logits를 직접 정하고 one-hot, label smoothing, ignore token을 차례로 적용한다. unreduced loss와 `p-y`, denominator를 손으로 계산한다. 모든 logit gradient의 row sum이 0인지 본다. mean-of-means가 틀리는 길이가 다른 두 microbatch도 만든다.

셋째는 작은 attention이다. causal mask와 GQA sharing을 넣고 compositional FP64 reference와 선택된 SDPA backend의 `dQ,dK,dV`를 비교한다. KV head의 gradient가 query group에서 합쳐지는지 본다. RoPE pair index 하나를 바꾼 expected-failure fixture로 forward와 backward 최초 차이를 찾는다.

넷째는 tied embedding이다. embedding-only, head-only, full graph의 세 gradient를 저장해 합 법칙을 확인한다. checkpoint save/load 뒤 alias identity와 optimizer parameter 중복을 검사한다. 의도적으로 clone해 tie를 끊었을 때 loss 첫 step이 같아도 다음 update가 갈라지는 것을 관찰한다.

다섯째는 update transaction이다. 두 microbatch, AMP scale, clipping, AdamW 한 step을 아주 작은 model에 적용한다. overflow를 주입한 run에서는 parameter, moment, scheduler, sampler cursor 중 정책상 전진하면 안 되는 상태를 검증한다. clean checkpoint에서 다음 batch까지 재생해 uninterrupted run과 비교한다.

각 실험 보고서는 hypothesis, controlled variable, expected invariant, observed tensor, source symbol, failure interpretation, recovery test를 한 표에 둔다. “오차가 작았다”가 아니라 어떤 dtype과 tolerance에서 어떤 불변식이 통과했는지 쓴다. expected-failure가 실제로 실패하는지 확인해야 test가 오류를 검출할 힘이 있다는 것도 증명된다.

이 다섯 실험이 연결되면 역전파는 더 이상 검은 상자가 아니다. 확률분포의 오차가 local VJP로 분해되고, residual과 shared parameter에서 합쳐지고, CUDA와 collective 위에서 누적된 뒤, optimizer state와 함께 원자적인 다음 상태로 commit되는 하나의 추적 가능한 과정이 된다.

**gradient 소유권·reduction·AMP·memory의 경계를 세운다**

**leaf, non-leaf, view의 gradient 소유권**

초보자가 가장 먼저 마주치는 혼란은 “분명 `requires_grad=True`인데 왜 `.grad`가 비어 있는가”다. `requires_grad`는 이 tensor를 포함한 연산을 reverse-mode graph에 기록할지 결정하는 조건이고, `.grad`에 결과를 영구 누적할지는 별도 문제다. 보통 사용자가 만든 trainable parameter 같은 leaf tensor가 gradient buffer를 소유한다. 연산 결과인 non-leaf tensor는 backward를 이어 가기 위한 adjoint를 순간적으로 받지만 기본적으로 `.grad`를 보존하지 않는다.

`x`가 leaf이고 `y=x*2`, `z=y.sum()`이면 `z.backward()` 뒤 `x.grad`는 채워지지만 `y.grad`는 기본적으로 보존되지 않는다. `y.retain_grad()`를 forward 중 호출하면 조사 목적으로 남길 수 있다. 이것은 graph를 유지하는 `retain_graph=True`와 전혀 다르다. 앞의 것은 특정 non-leaf의 gradient 관측이고, 뒤의 것은 같은 graph를 다시 backward할 수 있도록 node와 saved tensor의 lifetime을 연장한다.

parameter를 dtype이나 device로 옮기는 코드도 leaf 소유권을 바꿀 수 있다. module을 정상적으로 이동시키는 것과, 이미 만든 parameter에 연산을 적용해 그 결과를 새 변수로 잡는 것을 구분한다. optimizer에는 실제 forward가 읽는 leaf parameter가 들어 있어야 한다. optimizer가 오래된 object를 소유하고 model이 새 tensor를 읽으면 loss와 backward는 정상인데 step 뒤 model weight가 변하지 않는 기묘한 실패가 생긴다.

view는 새 storage가 아니라 base storage의 다른 해석일 수 있다. transpose, slice, reshape 일부가 이에 해당한다. view를 이용한 연산의 gradient는 view backward가 stride와 index mapping을 거꾸로 적용해 base leaf로 보낸다. expanded view는 여러 논리 위치가 같은 storage 원소를 가리키므로 backward에서 그 기여를 합한다. “forward에서 복사하지 않았다”와 “backward에서도 아무 일도 하지 않는다”는 같은 말이 아니다.

in-place mutation은 이 관계를 어렵게 만든다. backward formula가 과거 input을 필요로 하는데 그 storage를 덮으면 저장한 version과 현재 version이 달라진다. framework가 version counter 오류를 내면 안전장치가 작동한 것이다. `.data`나 잘못된 custom kernel로 이 안전장치를 우회하면 오류 대신 잘못된 finite gradient가 나올 수 있다. 그래서 out-of-place reference와 mutation negative fixture가 필요하다.

`zero_grad(set_to_none=True)`는 buffer를 0으로 채우는 대신 gradient가 아직 없다는 상태로 되돌린다. 다음 backward에서 새 buffer를 할당하거나 효율적으로 대입할 수 있다. 반면 zero tensor를 남기면 optimizer가 “gradient가 존재하지만 모두 0”인 parameter로 볼 수 있다. weight decay, sparse update, optimizer skip 정책에서 None과 zero의 효과가 같다고 가정하지 않는다.

**소유권 점검표.** parameter name, Python object identity, storage pointer, leaf 여부, requires-grad, optimizer group index, alias group, gradient state(None/zero/nonzero)를 update 전후에 기록한다. 모든 step에서 전체 pointer를 노출할 필요는 없지만 tiny fixture와 checkpoint round trip에서는 이 표가 shared/frozen/stale parameter 문제를 빠르게 찾는다.

**loss reduction은 gradient의 단위를 정한다**

loss reduction은 보기 좋은 scalar를 만드는 후처리가 아니다. gradient의 단위를 정하는 objective 일부다. token negative log-likelihood `ℓ_i`와 valid mask `m_i`에서 sum objective는 `S=Σm_iℓ_i`, token mean은 `L=S/N`, `N=Σm_i`다. backward는 모든 valid token 기여에 `1/N`을 곱한다. N을 local batch 크기, padded length, sequence 수로 바꾸면 optimizer가 보는 함수가 달라진다.

packing에서는 한 row 안에 여러 document가 들어가고 padding이 거의 없을 수 있다. conversational SFT에서는 prompt token을 mask하고 assistant token만 학습할 수 있다. preference loss는 pair 수로, reward-model auxiliary term은 valid score 수로, multimodal loss는 modality별 element 수로 정규화할 수 있다. 여러 loss를 더할 때 각 항의 분모와 coefficient가 gradient scale의 물리적 단위다.

microbatch 평균을 단순 평균하는 오류를 수치로 보자. 첫 microbatch에 valid token 2개, loss sum 2가 있고 둘째에 valid token 8개, loss sum 16이 있다. 올바른 token mean은 `18/10=1.8`이다. microbatch mean은 각각 1과 2이고 둘의 평균은 1.5다. 두 번째 microbatch의 token 기여가 의도보다 작아졌다. gradient도 같은 가중 왜곡을 받는다.

DDP가 rank gradient를 평균한다면 local loss scaling과 world size가 결합된다. 각 rank가 같은 N을 가질 때 local mean gradient의 rank 평균은 global mean과 같다. N이 다르면 그렇지 않다. 안전한 방식 하나는 각 rank에서 loss sum을 backward하고 global valid count로 정규화하되 DDP 평균이 추가하는 world-size factor를 명시적으로 보정하는 것이다. 다른 방식도 가능하지만 numerator, denominator, reducer semantics를 한 식으로 써야 한다.

sequence mean과 token mean은 서로 다른 공정성 가정을 담는다. token mean은 긴 sequence가 더 많은 gradient 질량을 갖는다. sequence mean은 각 sequence loss를 자체 길이로 평균한 뒤 sequence를 평균해 짧은 sequence와 긴 sequence의 총 가중치를 같게 할 수 있다. 어느 것이 옳은지는 과제와 sampling 정책에 달려 있다. metric 이름에 `loss` 하나만 쓰면 이 선택이 사라진다.

label smoothing, class weight와 z-loss도 분모에 들어온다. auxiliary loss를 batch mean으로 만들고 main CE를 token mean으로 만들면 sequence-length mixture가 변할 때 상대 coefficient의 실효값이 바뀔 수 있다. coefficient를 고정했다는 사실이 gradient 비율을 고정했다는 뜻은 아니다. 각 loss 항의 raw sum, count, reduced scalar, gradient norm contribution을 따로 관측한다.

**denominator conservation test.** 동일한 sample 집합을 한 batch, 길이가 다른 microbatch, 두 ranks의 불균등 partition으로 각각 표현한다. dropout을 끄고 global numerator와 denominator를 동일하게 만든다. loss, selected gradient, parameter delta가 같아야 한다. 일부 token mask를 바꾼 negative fixture에서는 예상한 비율로 달라져야 한다. 이 test가 통과해야 accumulation step 수를 performance 옵션으로 취급할 수 있다.

**AMP overflow는 분기 하나가 아니라 합의 protocol이다**

mixed precision의 dynamic loss scaling은 scale 숫자 하나를 조절하는 편의 기능이 아니다. backward 결과가 이 update를 commit할 수 있는지를 판정하는 protocol이다. scaled gradient를 unscale하면서 non-finite를 찾고, 하나라도 발견되면 parameter와 optimizer state를 변경하지 않은 채 step을 건너뛴다. scale tracker는 정책에 따라 낮아지며 다음 시도를 준비한다.

single optimizer에서는 이 흐름이 단순해 보인다. 그러나 optimizer가 둘이거나 parameter group이 서로 다른 device에 있거나 distributed shard를 가지면 found-inf의 소유권이 분산된다. 한 optimizer만 step하고 다른 optimizer가 skip하면 model state는 어떤 명확한 objective의 update도 아니다. 의도적으로 독립 update를 허용하지 않는 한 commit decision을 공유해야 한다.

DDP에서도 한 rank만 overflow를 발견하고 그 rank만 skip하면 즉시 replica가 갈라진다. found-inf를 rank 전체에서 합의하고 모든 rank가 같은 branch를 실행해야 한다. agreement collective 자체가 gradient collective 순서와 섞여 hang을 만들지 않도록 call sequence를 고정한다. timeout 뒤 재시작할 때는 어느 rank가 어떤 tensor에서 non-finite를 발견했는지 보존한다.

scheduler, EMA, global step, sample cursor, checkpoint generation도 commit decision을 알아야 한다. optimizer step이 skip됐는데 scheduler와 EMA만 전진하면 다음 성공 update의 LR과 teacher weight가 기준 실행과 달라진다. sample cursor는 재시도 정책에 따라 전진할 수도 있고 같은 batch를 다시 쓸 수도 있다. 어느 쪽이든 명시적인 `AttemptID`와 `CommittedUpdateID`를 분리한다.

gradient accumulation 중 마지막 microstep에서 overflow를 발견하면 앞서 누적한 finite gradient도 이번 window와 함께 폐기된다. 중간 partial gradient를 다음 window에 섞으면 sample lineage와 denominator가 깨진다. abort 뒤 gradient buffer, reducer bucket readiness, no-sync state가 깨끗한지 확인한다.

BF16은 FP16보다 exponent 범위가 넓어 dynamic scaling이 필요하지 않은 경우가 많지만, non-finite가 불가능해지는 것은 아니다. attention score 폭주, 잘못된 division, optimizer moment overflow는 여전히 생긴다. “BF16이므로 scaler가 없다”와 “finite gate가 필요 없다”를 혼동하지 않는다.

**원자성 fixture.** 정상 update와 동일한 초기 state에서 selected gradient 한 원소에 inf를 주입한다. parameter, first/second moment, optimizer step counter, scheduler, EMA, committed update counter의 checksum이 모두 예상대로 유지되는지 본다. 다음 정상 batch를 실행해 clean reference의 해당 정책 지점과 parity가 회복되는지 확인한다. 단순히 exception이 없다는 것은 합격 조건이 아니다.

**activation checkpoint는 시간을 바꾸어 메모리를 산다**

activation checkpointing의 핵심 교환은 명확하다. forward에서 backward에 필요할 모든 중간 tensor를 저장하지 않고 구간의 경계 입력만 보존한다. backward가 그 구간에 도착하면 forward를 다시 실행해 중간값을 복원한다. 메모리를 줄이는 대신 추가 계산을 지불한다. 하지만 stochastic state와 side effect가 있으면 “같은 forward를 다시 실행한다”는 조건이 깨질 수 있다.

dropout은 같은 mask가 필요하다. CPU와 CUDA RNG state, 여러 device, custom generator를 모두 쓰면 무엇을 보존하는지 source 계약을 확인한다. default device 추론이 실제 tensor가 있는 모든 device를 포괄하지 못하면 한 device의 mask만 달라질 수 있다. distributed sequence parallel 영역에서 rank별 RNG offset 규칙도 원래 forward와 같아야 한다.

BatchNorm running statistics 같은 mutation, cache append, router counter, logging side effect는 recompute에서 두 번 실행될 수 있다. Transformer에서 LayerNorm은 running state가 없지만 custom module이나 MoE telemetry는 그렇지 않을 수 있다. forward가 순수 함수라는 가정을 깨는 모든 write를 inventory로 만든다.

reentrant와 non-reentrant checkpoint는 autograd graph 기록과 API 제약이 다르다. 어느 variant가 input/output nested structure, detached tensor, `autograd.grad`, early-stop recomputation, compile context를 지원하는지 현재 framework revision에서 확인한다. 오래된 경험으로 default를 추론하지 않고 호출부의 resolved `use_reentrant` 값을 기록한다.

selective checkpoint는 operation마다 저장과 재계산 정책을 달리할 수 있다. matmul처럼 계산량이 큰 결과는 저장하고 elementwise 연산은 재계산하는 선택이 가능하다. 그러나 compiler가 fusion을 바꾸면 operation 경계와 비용 모델이 달라진다. peak allocated memory 하나만 보지 말고 saved bytes, recompute FLOPs, step time, graph break, gradient parity를 함께 비교한다.

checkpoint 경계가 너무 작으면 Python/autograd overhead가 커지고, 너무 크면 expensive operation까지 반복한다. pipeline parallel에서는 recompute가 stage schedule의 bubble과 memory lifetime에 영향을 준다. tensor parallel collective를 recompute하는지, communication 결과를 저장하는지에 따라 네트워크 비용도 달라진다. 그래서 checkpoint policy는 module 이름 목록이 아니라 실행 graph와 topology에 종속된 성능 계약이다.

**recompute parity fixture.** dropout off exact reference, dropout on RNG-preserving reference, intentional RNG mismatch의 세 run을 둔다. forward loss, layer별 gradient, parameter delta를 비교하고 profiler에서 해당 구간이 예상 횟수만큼 재실행됐는지 확인한다. side-effect counter를 하나 넣은 expected-failure module로 detector가 이중 실행을 잡는지도 시험한다.

**backward hook은 관찰 장치이자 잠재적 개입이다**

hook은 편리하지만 관찰만 한다고 자동 보증되지 않는다. tensor hook이 새 gradient를 반환하면 downstream 값을 바꾼다. module full backward hook은 input/output gradient tuple의 의미와 호출 시점이 있고, 같은 module이 여러 번 호출되거나 checkpoint recompute될 때 횟수가 늘어난다. distributed wrapper가 parameter hook을 자체 등록한다는 사실도 고려한다.

hook 안에서 `.item()`, CPU copy, 파일 쓰기를 하면 CUDA synchronization과 Python serialization이 들어간다. race의 timing을 바꾸거나 overlap을 없애 문제를 가릴 수 있다. 상시 관측은 device-side reduction으로 작은 통계만 만들고 비동기 buffer에 기록한다. first-bad-tensor 조사에서만 선택한 tensor slice를 동기화한다.

gradient norm 하나로는 방향 오류를 잡지 못한다. 최소 통계는 shape, stride, dtype, finite count, zero fraction, min/max, RMS 또는 L2, checksum이다. 두 run 비교에는 max absolute, scale-aware relative error와 cosine을 추가한다. 매우 작은 reference 원소의 relative error가 폭발하므로 absolute와 relative threshold를 결합한다.

parameter hook의 호출 순서를 collective 순서로 사용하면 안 된다. autograd scheduling과 bucket rebuild에 따라 ready 순서가 바뀔 수 있다. DDP reducer가 정의한 bucket과 collective sequence가 protocol이고, 관측 hook은 이를 읽기만 해야 한다. hook에서 임의 collective를 호출하면 rank마다 다른 graph path에서 deadlock할 수 있다.

anomaly detection은 backward에서 NaN이나 오류가 난 node의 forward traceback을 제공해 유용하지만 큰 overhead가 있다. 모든 silent wrong gradient를 검출하지는 못한다. finite하지만 틀린 custom backward는 analytic reference, gradcheck와 negative fixture가 필요하다. profiler 역시 실행되었다는 사실은 보여 주지만 수학적 정확성을 증명하지 않는다.

**probe ladder.** 먼저 loss와 update-level metric으로 이상을 발견한다. 다음에는 block boundary checksum으로 최초 다른 block을 찾는다. 그 block에서 attention/MLP/norm 경계로 좁힌다. 마지막에 operator input, upstream gradient, local VJP를 dump한다. 처음부터 전체 graph 모든 tensor를 저장하지 않는다. 관측 비용이 원래 장애를 바꾸지 않는지 probe off/on parity도 둔다.

**한 번의 update를 법의학적으로 복원하는 사례**

상황을 가정하자. 네 ranks에서 accumulation 4로 학습하는 run이 재시작 뒤부터 loss는 비슷하지만 validation이 서서히 나빠진다. gradient norm 평균은 정상이고 NaN도 없다. “수치적 흔들림”이라고 닫기 쉬운 유형이다.

첫 단계는 같은 CheckpointID와 다음 BatchID를 clean process에서 재생하는 것이다. rank별 sample IDs와 valid-token counts를 비교하니 rank 3만 마지막 microbatch의 count가 다르다. padded sample을 재구성하는 data cursor가 checkpoint에 완전히 저장되지 않았고, resume에서 shard 끝 처리 정책이 달라졌다.

두 번째로 global loss numerator와 denominator를 재계산한다. UI에 표시된 loss는 rank-local mean을 평균한 값이라 차이가 작았지만 실제 global token mean은 기준 실행과 달랐다. DDP gradient 평균은 불균등 local denominator를 이미 반영했으므로 gradient 방향도 조금 바뀌었다. 평균 norm은 이 방향 차이를 숨겼고 selected layer의 cosine 비교가 최초 신호를 보였다.

세 번째로 update lineage를 본다. accumulation window의 네 microsteps와 rank별 counts, reducer completion, unscale, clip coefficient는 모두 finite하다. optimizer 구현은 잘못이 없었다. 최초 차이는 data cursor→valid mask→local denominator였고 backward와 optimizer는 주어진 다른 objective를 정확히 계산했다.

수정은 LR이나 clipping 조정이 아니다. checkpoint에 shard cursor, epoch permutation, partial-buffer state와 end-of-shard policy를 포함하고, resume 첫 global batch의 sample multiset과 valid count를 uninterrupted run과 비교하는 gate를 추가한다. 과거 실패 checkpoint는 조용히 load하지 않고 sample-exact resume을 보장하지 않는 generation으로 표시한다.

negative fixture는 rank 하나의 cursor를 한 sample 앞당긴다. detector는 loss threshold가 아니라 BatchID multiset 또는 denominator mismatch에서 즉시 실패해야 한다. 다른 fixture는 sample은 같고 mask 한 token만 바꾼다. 이때 data identity checksum만으로 부족하고 token-level supervision checksum이 필요하다는 것을 확인한다.

이 사례의 교훈은 “backward bug처럼 보이는 현상”이 graph 이전에서 시작할 수 있다는 점이다. 반대로 data가 같다고 optimizer가 맞는 것도 아니다. 조사 순서는 BatchID와 objective, activation과 local VJP, reduction과 scale, optimizer state, checkpoint next-step parity를 잇는다. 최초 divergence보다 뒤의 정상 값은 원인을 면책하지 않는다.

**3장으로 넘기는 실행 가능한 update 계약**

3장의 작은 GPT run은 이 장의 개념을 말로만 참조하지 않는다. 입력으로 `LossEnvelope`를 받아야 한다. 여기에는 sample/token identity, target shift, supervision mask, loss별 numerator·denominator, selected logits와 reduction policy가 있다.

backward 뒤에는 `GradientEnvelope`를 만든다. graph generation, accumulation window, microstep IDs, scaler scale, found-inf agreement, reducer semantics, clip norm과 coefficient, parameter alias/group manifest, selected gradient statistics가 포함된다. raw, unscaled, clipped gradient를 같은 이름으로 덮어쓰지 않고 phase를 구분한다.

optimizer가 commit하면 `UpdateEnvelope`를 만든다. AttemptID와 CommittedUpdateID, optimizer/scheduler generation, parameter와 state의 pre/post checksum, skip/abort 이유, sampler cursor policy, checkpoint eligibility를 기록한다. checkpoint는 이 envelope가 가리키는 post-state와 다음 BatchID를 함께 복원해야 한다.

세 envelope는 거대한 tensor dump가 아니라 재계산 경로다. 전체 통계와 selected index, source revision, config resolved value, artifact digest를 이용해 독립 검토자가 같은 tiny run을 재생할 수 있어야 한다. 민감한 원본 data를 공개하지 못하면 stable sample hash와 공개 가능한 synthetic fixture를 함께 둔다.

3장에서 model의 parameter 수와 activation memory를 계산할 때도 gradient와 optimizer state의 ownership을 이 계약에서 읽는다. tied parameter를 두 번 세거나 frozen base의 gradient memory를 포함하지 않는다. activation checkpoint와 fused backward가 저장하는 tensor를 실제 selected path에 맞춰 산정한다.

이제 한 step의 질문은 닫힌다. loss scalar만 맞는가가 아니라 동일한 token objective가 동일한 local VJP, 동일한 collective와 scale, 동일한 stateful optimizer commit을 거쳐 다음 step에서도 같은 함수를 계산하는가를 묻는다. 이 조건을 충족한 update만 작은 GPT 전체 실행의 기준점이 된다.

**embedding backward는 lookup의 역연산이다**

token embedding forward는 weight matrix `E[V,D]`에서 input ID가 가리키는 row를 모아 `X[B,T,D]`를 만든다. backward는 이 gather를 거꾸로 실행한다. 각 position의 upstream vector를 해당 token ID의 weight row에 더한다. 같은 token이 batch와 sequence에서 여러 번 나오면 모든 기여가 합쳐진다.

작은 예로 IDs가 `[[2,1,2]]`이고 upstream이 세 vector `g0,g1,g2`라면 `dE[2]=g0+g2`, `dE[1]=g1`, 나머지 row는 0이다. index에는 미분하지 않는다. token ID는 discrete address이지 연속 parameter가 아니다. tokenizer 선택에 gradient가 직접 흐르지 않는 이유와 embedding weight에는 흐르는 이유가 여기서 갈린다.

framework embedding은 dense gradient를 만들 수도 있고 sparse gradient를 만들 수도 있다. sparse mode는 방문한 row만 저장해 큰 vocabulary에서 메모리와 update를 줄일 수 있지만 지원 optimizer와 distributed reduction에 제약이 있다. sparse tensor의 coalesce 여부, duplicate index 합산, weight decay 의미를 확인한다. dense optimizer가 sparse gradient를 조용히 dense로 바꾸면 예상한 메모리 절감이 사라진다.

padding index가 지정되면 해당 row의 gradient를 0으로 유지하는 계약이 있을 수 있다. 그러나 LM에서 pad token이 실제 target vocabulary class로도 쓰이거나 embedding과 output head가 tied되면 head 경로의 gradient는 별도로 들어올 수 있다. “padding index이므로 shared weight row가 절대 변하지 않는다”고 일반화하지 않는다.

frequency scaling 옵션은 같은 token이 여러 번 등장할 때 gradient를 빈도로 나눌 수 있다. 이것은 optimizer learning rate 편의가 아니라 objective contribution을 바꾸는 동작이다. 기본값과 resolved option을 기록하고, 반복 token fixture로 기대 합 또는 평균을 검증한다.

embedding backward는 duplicate ID 때문에 parallel scatter-add를 쓸 수 있다. CUDA atomic 합산 순서가 bitwise nondeterminism을 만들 수 있다. deterministic mode가 다른 algorithm을 선택하는지, 큰 vocabulary와 Zipf 분포에서 hot row contention이 성능 병목인지 분리해 본다. correctness fixture는 반복 ID, padding ID, out-of-range error, empty input, tied head를 포함한다.

한 batch에서는 방문한 row만 embedding lookup gradient를 받지만, training 전체에서는 token frequency가 update 횟수와 noise scale을 결정한다. rare token row는 적은 관측으로 큰 변화가 생길 수 있고, frequent token은 많은 문맥의 상충하는 기여를 평균한다. tokenizer와 data mixture가 optimizer geometry에 연결되는 지점이다.

**MLP backward를 gate와 value 두 길로 나눈다**

SwiGLU MLP를 단순화해 `a=xW_g`, `b=xW_u`, `h=silu(a)⊙b`, `y=hW_d`라 쓰자. output gradient `dy`에서 `dh=dyW_dᵀ`, `dW_d=hᵀdy`가 먼저 나온다. elementwise product 때문에 `da=dh⊙b⊙silu'(a)`, `db=dh⊙silu(a)`로 갈라지고, 입력에서는 `dx=daW_gᵀ+dbW_uᵀ`로 다시 합쳐진다.

gate와 value projection을 하나의 fused matrix로 계산하는 구현은 weight layout을 `[gate;up]` 또는 `[up;gate]`로 둘 수 있다. split 순서가 바뀌면 shape는 맞지만 다른 함수를 계산한다. checkpoint conversion과 tensor parallel shard가 이 순서를 보존해야 한다. basis input과 비대칭 weight로 gate/up swap을 잡는다.

SiLU의 derivative는 `σ(a)+aσ(a)(1-σ(a))`다. 큰 음수에서는 activation과 derivative가 작아지고, 큰 양수에서는 거의 선형이 된다. approximate activation이나 fused kernel이 다른 식을 쓰면 saturation 영역에서 gradient 차이가 커질 수 있다. 0 근처와 양·음의 큰 input을 포함한 scalar fixture를 둔다.

MLP intermediate dimension은 activation memory와 GEMM shape를 좌우한다. checkpointing 없이 backward하면 gate/up preactivation 또는 activation을 저장할 수 있다. fused SwiGLU가 어떤 tensor를 저장하고 어떤 것을 재계산하는지에 따라 memory가 다르다. “fused라서 memory가 절반” 같은 주장은 saved-tensor inventory 없이 하지 않는다.

tensor parallel에서는 first projection output을 shard하고 down projection 뒤 reduce할 수 있다. backward collective는 forward의 반대 ownership 흐름을 따른다. `dW_d`, local `dh`, gate/up local gradients, input-gradient reduction의 순서를 적는다. bias 유무와 sequence parallel이 reduction 축을 바꿀 수 있다.

MoE expert 하나의 MLP backward 식은 비슷하지만 token routing이 앞에 붙는다. expert가 받은 token 수가 0이면 empty GEMM과 gradient state 정책이 필요하다. selected expert의 weight만 gradient를 받고 router gate는 별도 경로를 가진다. expert parallel all-to-all의 reverse permutation이 틀리면 다른 token gradient가 원래 position으로 돌아간다.

hidden 2, intermediate 3, token 2의 FP64 tensor로 모든 중간값과 gradient를 저장한다. unfused composition, fused activation, fused projection 경로를 비교한다. gate/up swap과 reverse permutation 오류를 주입해 norm이 비슷해도 selected element와 cosine이 실패하는지 확인한다.

**한 pre-norm block을 역순으로 걷는다**

pre-norm decoder block을 `u=x+Attn(Norm1(x))`, `y=u+MLP(Norm2(u))`라 하자. backward는 쓰인 식의 정확한 역순이다. `dy`는 마지막 residual에서 identity 경로 `du_identity=dy`와 MLP 경로로 갈라진다. MLP backward를 통과한 뒤 Norm2 VJP를 거쳐 두 기여가 `du`에서 합쳐진다.

이 `du`가 첫 residual에서 다시 identity와 attention 경로로 갈라진다. attention output projection, P·V, softmax, QK와 RoPE의 backward를 통과하고 Norm1 VJP를 거쳐 최종 `dx`에 합쳐진다. parameter gradient는 각 local operation에서 생성되고 input gradient는 이전 block으로 흐른다.

post-norm이면 Jacobian 곱 순서가 다르다. residual 합 뒤 norm을 통과하므로 identity 경로도 norm Jacobian의 영향을 받는다. pre/post 이름만 알고 gradient 흐름이 같다고 가정하면 initialization과 stability 설명이 피상적이 된다. 작은 linearized block에서 두 Jacobian을 직접 곱해 singular value가 어떻게 달라질 수 있는지 본다.

block backward 조사에는 forward call index가 필요하다. weight sharing이나 recurrent block이면 같은 module path가 여러 layer 위치에서 호출될 수 있다. activation checkpoint recompute도 forward hook을 다시 실행한다. `layer_index`, `call_index`, `phase=original|recompute|backward`를 구분한다.

최초 차이를 찾을 때 최종 `dx`만 보면 어느 residual branch가 틀렸는지 알기 어렵다. `du_identity`, `du_mlp`, `dx_identity`, `dx_attention`의 norm과 selected slice를 따로 비교한다. 합 결과가 맞아도 두 branch 오류가 상쇄될 수 있으므로 branch contribution을 보존한 fixture가 필요하다.

stochastic depth나 residual scaling이 있으면 identity와 branch coefficient가 달라진다. DeepNorm, residual multiplier, gated residual, mHC 같은 구조는 단순 `x+F(x)`가 아니다. 7장과 10장의 model config에서 실제 식을 가져와 local VJP를 다시 유도한다. 이름이 residual이라는 이유로 coefficient 1을 가정하지 않는다.

norm 단독, MLP 단독, attention 단독을 통과시킨다. residual을 붙인 뒤 tied/shared parameter 기여를 확인한다. eager FP32 reference에서 시작해 BF16 autocast, fused backend, checkpoint, compile, tensor parallel 순으로 하나씩 추가한다. 각 단계는 바로 전 단계와 선택 경로 하나만 달라야 한다.

**clipping·compiled backward·성능을 정확성 뒤에서 판정한다**

**gradient clipping의 기하와 한계**

global norm clipping은 gradient vector가 반지름 `c`인 ball 밖에 있을 때 경계로 radial projection한다. `||g||>c`이면 방향은 유지하고 크기만 `c`로 줄인다. 따라서 clipping 전후 cosine은 이상적으로 1이다. 방향이 달라지면 parameter별 clip, non-finite 처리, group 누락 같은 다른 동작이 섞였다.

그러나 parameter space의 Euclidean norm은 function space의 변화량과 같지 않다. layer마다 scale과 sensitivity가 다르고 reparameterization에 따라 같은 함수의 gradient norm이 달라진다. clipping은 폭주 방지에 유용한 운영 장치지만 trust region을 정확히 구현하거나 안정적 학습을 보증하지 않는다.

adaptive gradient clipping은 parameter norm에 비례한 threshold를 쓰기도 한다. unit-wise인지 tensor-wise인지, zero-norm parameter에 floor를 어떻게 두는지에 따라 다르다. global clipping과 동시에 쓰면 순서와 중복 효과를 명시해야 한다. library option 이름만 보고 같은 algorithm이라고 판단하지 않는다.

FSDP나 tensor parallel에서 global norm은 shard-local squared norm을 합쳐 square root를 취해야 한다. replicated parameter를 rank마다 중복 집계하면 norm이 커지고 과도하게 clip된다. shared parameter도 한 번만 세야 한다. sparse gradient와 expert-local parameter가 global group에 포함되는지 정의한다.

clip coefficient가 오랜 기간 매우 작으면 실제 effective learning rate가 scheduler보다 clipping에 의해 결정된다. dashboard에는 pre-clip norm, threshold, coefficient, clipped-step fraction, layer/group별 norm을 둔다. parameter 수가 많은 group의 norm이 자연히 큰 경향과 진짜 폭주를 구분한다.

같은 방향에서 크기만 다른 gradient 두 개는 threshold 밖에서 같은 clipped vector가 될 수 있다. 이 때문에 upstream 폭주 정도가 사라질 수 있으므로 pre-clip 값을 보존한다. 반대로 norm은 같지만 직교하는 두 gradient는 clipping 뒤에도 서로 다르다. norm 안정성을 convergence 증거로 쓰지 않는다.

**compiled와 fused backward의 의미 경계**

compiler는 Python graph를 포착하고 operation을 재배치·fusion하거나 custom kernel을 생성할 수 있다. forward와 backward graph를 함께 최적화하면 saved tensor 선택, recomputation, memory planning이 eager와 달라진다. API가 같아도 실행 node와 kernel 수가 달라지므로 source와 profiler를 함께 읽는다.

graph break가 생기면 앞뒤 구간만 compile되고 중간은 eager로 실행될 수 있다. 성능이 예상보다 낮은 문제와 gradient가 틀린 문제를 구분한다. graph break reason, compiled region count, guard와 recompilation count를 기록한다. 입력 shape가 바뀔 때 새 graph가 생기는지 dynamic shape policy도 본다.

fusion은 수학적으로 연속한 연산 사이 intermediate rounding을 줄이거나 reduction 순서를 바꿀 수 있다. 허용 오차 안의 차이일 수 있지만 loss denominator, mask, RNG와 mutation 같은 semantic state는 보존해야 한다. `fast_math`나 TF32 설정은 approximation 범위를 별도로 기록한다.

custom backward가 compiler transform을 지원하려면 fake/meta kernel, functionalization, alias/mutation schema, autograd registration이 필요할 수 있다. eager에서만 통과한 extension을 compiled training에 바로 넣지 않는다. forward-only compile 성공 역시 backward compile을 증명하지 않는다.

CUDA graph와 compiler를 함께 쓰면 static address, warmup, capture-safe allocator와 optimizer step이 결합된다. overflow처럼 control flow가 달라지는 update는 capture 안팎의 정책을 확인한다. step skip branch가 capture된 parameter mutation을 부분 실행하지 않아야 한다.

eager composition을 기준으로 eager fused, compiled composition, compiled fused의 네 칸을 만든다. forward, input/parameter gradient, delta, peak memory, step time을 비교한다. compile failure가 eager fallback으로 숨는지 selected backend trace로 확인한다. correctness를 통과한 칸만 성능 표에 넣는다.

**backward memory를 byte 단위로 계산한다**

학습 메모리는 parameter bytes만으로 설명되지 않는다. parameter storage, gradient, optimizer state, saved activation, temporary workspace, communication buffer, allocator fragmentation이 함께 peak를 만든다. backward는 saved activation을 소비하면서 gradient와 temporary를 만들기 때문에 peak 위치가 model마다 다르다.

간단한 dense BF16 training에서 parameter가 `P`개라면 BF16 weight `2P` bytes, FP32 gradient나 master weight를 쓰면 각각 `4P`, Adam moments는 `8P`가 추가될 수 있다. 실제 optimizer와 precision policy에 따라 다르므로 고정된 “parameter당 16 bytes” 공식을 맹신하지 않는다. fused optimizer, quantized state, sharding이 ownership을 바꾼다.

activation은 대략 batch `B`, sequence `T`, hidden `H`, layer `L`에 비례하지만 attention probability를 materialize하면 `BT²` 항이 생긴다. Flash attention은 이 matrix를 HBM에 저장하지 않아 memory scaling을 바꾼다. backward 재계산을 위해 어떤 Q/K/V/O/LSE를 저장하는지 실제 backend에서 센다.

gradient accumulation은 activation을 K배 동시에 저장하는 것이 보통은 아니다. 각 microbatch backward가 끝나면 activation은 해제되고 parameter gradient만 누적된다. 반면 pipeline schedule은 여러 microbatch activation이 in-flight라 peak가 달라진다. `retain_graph` 누수는 accumulation과 달리 이전 graph를 계속 살려 둔다.

activation checkpoint는 saved activation을 줄이고 recompute temporary를 만든다. peak allocated와 peak reserved를 구분하고 memory snapshot으로 어떤 allocation이 살아 있는지 본다. allocator cache가 reserved memory를 유지해도 live tensor 누수와 같지는 않다. OOM 직전 free memory 하나만 보는 대신 allocation history와 tensor lifetime을 본다.

DDP bucket은 gradient와 별도 buffer를 만들거나 gradient view를 사용할 수 있다. FSDP는 all-gather한 full parameter와 reduce-scatter buffer의 lifetime이 겹칠 수 있다. overlap prefetch 옵션이 성능을 높이면서 peak memory도 높일 수 있다. 15장의 ownership 표와 이 장의 update phase를 결합해 시간축 메모리 도표를 만든다.

각 allocation class에 owner, shape, dtype, element bytes, count, lifetime start/end, sharding factor, alias 여부를 기록한다. 계산 합과 framework memory snapshot의 차이를 workspace·allocator·untracked library allocation으로 분해한다. 이 ledger가 있어야 batch size를 줄이는 임시 처방보다 어떤 tensor lifetime을 줄여야 하는지 판단할 수 있다.

**역전파를 이해했다는 판정 기준**

수식을 외웠다는 것은 첫 단계다. 독자는 scalar branch의 누적에서 tensor VJP로 이동하고, softmax·norm·attention·MLP·residual의 local backward를 upstream gradient와 연결해 설명할 수 있어야 한다. full Jacobian을 만들지 않는 이유와 reverse mode가 scalar loss에 유리한 이유도 설명해야 한다.

코드에서는 public API, generated derivative, runtime node, CUDA kernel과 upstream test를 서로 다른 층으로 찾을 수 있어야 한다. 어떤 tensor를 backward까지 저장하며 mutation/version/RNG가 그 계약을 어떻게 깨뜨리는지 판단해야 한다. fused나 compiled라는 이름을 correctness의 증거로 쓰지 않는다.

실험에서는 hand FP64, finite-direction difference, eager FP32, mixed precision, fused/compiled, distributed 순으로 oracle 사다리를 세울 수 있어야 한다. 정상 fixture와 함께 detach, wrong denominator, gate swap, one-rank overflow, stale moment, RNG mismatch 같은 expected-failure를 넣어 detector의 민감도를 확인한다.

운영에서는 loss, gradient와 update를 하나로 뭉뚱그리지 않는다. sample/token objective, local VJP, accumulation/collective, unscale/clip, optimizer state, checkpoint next-step parity의 최초 divergence를 찾는다. 평균 norm과 finite 여부만으로 방향과 state correctness를 추론하지 않는다.

마지막 판정은 재시작 뒤 다음 update다. 동일한 parent state와 BatchID에서 loss numerator/denominator, selected activation, raw·unscaled·clipped gradient, optimizer delta와 next state가 기준 실행의 계약과 맞아야 한다. bitwise 동일성이 필요하지 않은 backend라도 허용 오차와 의미적 불변식을 명시한다.

이 기준을 통과하면 3장의 작은 GPT는 단지 돌아가는 script가 아니다. input byte에서 다음-token objective가 만들어지고, Transformer graph를 거쳐 parameter와 optimizer state가 바뀌며, 그 결과를 중단 뒤에도 재현하는 완전한 학습 상태기계가 된다.

**matmul backward의 transpose는 암기보다 index로 찾는다**

Transformer backward 오류 가운데 transpose와 reduction 축 오류는 shape가 우연히 맞아 조용히 지나갈 수 있다. `Y=XWᵀ`에서 `X[N,I]`, `W[O,I]`, `Y[N,O]`라 두고 원소 식 `Y_no=Σ_i X_ni W_oi`부터 시작한다. upstream `G[N,O]`가 오면 `dX_ni=Σ_o G_no W_oi`, `dW_oi=Σ_n G_no X_ni`다. 행렬식은 `dX=GW`, `dW=GᵀX`가 된다.

batch와 token을 `N=B×T`로 평탄화했다면 weight gradient는 두 축 모두를 합친다. packing mask가 있더라도 hidden이 loss로 연결되지 않은 position에는 upstream이 0이어야 한다. mask를 matmul backward에 다시 적용하는 것이 아니라 graph 앞의 loss/attention mask가 만든 upstream 의미를 추적한다.

QKV fused projection은 output 축을 Q, K, V 구간으로 나눈다. backward에서는 `dQ,dK,dV`를 그 layout에 맞춰 합친 뒤 fused weight gradient를 만든다. GQA에서는 Q와 KV의 크기가 다를 수 있어 세 equal chunk 가정이 틀린다. config의 head 수와 actual projection sizes로 split한다.

column/row parallel linear는 local matmul 식과 collective가 결합된다. input이 replicated인지 sharded인지, output gradient가 어느 rank에 있는지에 따라 `dX` all-reduce와 `dW` local reduction이 달라진다. transpose가 맞아도 collective owner가 틀리면 local unit test는 통과하고 distributed run만 갈라진다.

**basis fixture.** X와 G에서 원소 하나씩만 1로 두면 dW의 어느 원소가 1이어야 하는지 바로 알 수 있다. random tensor의 norm 비교보다 index mapping 오류를 잘 잡는다. batch·token·head 축을 하나씩 움직이며 expected nonzero coordinate를 기록한다.

**higher-order gradient가 필요한 순간**

일반적인 language-model training은 first-order gradient로 충분하지만 모든 fine-tuning objective가 그렇지는 않다. gradient penalty, meta-learning, influence approximation, curvature-vector product는 gradient 계산 자체를 다시 미분할 수 있다. 첫 backward를 만들 때 `create_graph=True`가 필요한 이유다.

gradient `g(θ)=∇L(θ)`와 vector `v`에서 Hessian-vector product는 `∇θ[g(θ)·v]`로 구할 수 있다. full Hessian `P×P`를 만들지 않고 두 번의 automatic differentiation으로 곡률 방향을 얻는다. 값은 싸지 않지만 full matrix보다 현실적이다.

custom autograd Function이 backward 안에서 detached 값이나 미분 불가능한 kernel을 쓰면 first-order result는 맞아도 second-order graph가 끊긴다. 지원 contract에 first-order only인지 명시한다. 필요한 objective에서만 `gradgradcheck`를 수행하고, 일반 학습에 무분별하게 create-graph를 켜 activation lifetime을 늘리지 않는다.

finite difference of gradient로 HVP를 검산할 수 있다. `(g(θ+εv)-g(θ-εv))/(2ε)`와 analytic HVP를 비교한다. 역시 ε sweep과 FP64 작은 fixture가 필요하다. ReLU 경계나 top-k routing처럼 매끄럽지 않은 지점은 해석 범위를 따로 둔다.

**backward 성능 병목을 정확성 뒤에 읽는다**

GPU utilization이 낮다고 곧바로 backward kernel을 fusion하지 않는다. 먼저 timeline에서 CPU launch gap, data wait, allocator synchronization, communication wait, 작은 kernel 과다, GEMM shape 비효율을 구분한다. backward는 forward보다 더 많은 GEMM과 reduction, gradient collective를 포함해 병목 소유자가 다를 수 있다.

큰 GEMM은 Tensor Core를 잘 쓰지만 작은 batch·짧은 sequence·좁은 adapter에서는 launch와 memory traffic 비중이 커진다. LoRA는 base linear backward 중 weight gradient를 생략할 수 있지만 input gradient와 adapter A/B gradient는 계산해야 한다. frozen parameter 수만 보고 backward FLOPs가 같은 비율로 줄 것이라 추정하지 않는다.

profiler의 kernel 이름을 수식 단계에 연결한다. 어떤 GEMM이 dX, dW인지, 어떤 reduction이 bias/norm/softmax인지, 어느 NCCL call이 어느 gradient bucket인지 표시한다. correlation ID와 tensor shape 없이 kernel 시간만 나열하면 optimization 영향 반경을 판단하기 어렵다.

성능 변경 전후에는 같은 GoldenBatchID, precision, backend와 synchronization policy를 쓴다. step time뿐 아니라 tokens/s, peak memory, recompute FLOPs, gradient parity를 함께 본다. 빠른 경로가 fallback되거나 일부 parameter gradient를 빼먹어 빨라진 것은 개선이 아니다.

**독자의 첫 backward 코드 리뷰**

코드 리뷰는 `loss.backward()` 줄에서 시작하지 않는다. loss가 어떤 unreduced 항과 분모에서 만들어졌는지, target shift와 mask owner가 누구인지 먼저 찾는다. 이어 autocast/scale, accumulation loop, no-sync, backward entry를 순서대로 표시한다.

그 다음 trainable parameter manifest를 만든다. frozen, shared, adapter, sparse, expert-local parameter를 구분하고 optimizer group과 대조한다. backward 뒤 None/zero/nonzero gradient 비율과 selected tensor를 본다. 예상하지 않은 None은 graph 단절이나 group 오류이고, 예상한 None은 frozen contract다.

unscale, non-finite agreement, clipping, optimizer step, scheduler와 zeroing 순서를 적는다. 각 함수가 읽고 쓰는 state를 표로 만든다. exception이나 overflow branch가 어느 state를 원복하고 어느 counter를 전진시키는지도 읽는다.

마지막에는 checkpoint save 지점을 찾는다. accumulation 도중인지 commit 뒤인지, gradient·moment·scaler·scheduler·RNG·cursor를 어디까지 보존하는지 확인한다. clean process에서 next batch와 next update를 예측할 수 없다면 resume 계약은 아직 완성되지 않았다.

리뷰 결과는 막연한 “autograd 사용”이 아니라 여섯 좌표로 남긴다. objective 생성 함수, backward entry, local/fused derivative 구현, collective owner, optimizer mutation, checkpoint serialization이다. 각각 source revision과 test 또는 제안 fixture를 연결한다.

좋은 리뷰는 오류가 없다고 선언하는 문서가 아니다. 어떤 input·dtype·shape·backend·topology에서 무엇을 검증했고, 무엇은 실행하지 않아 제안으로 남았는지를 선명히 구분한다. 이 경계가 다음 코드 변경에서 다시 확인할 정확한 출발점이 된다.

**gradient accumulation을 큰 batch와 같게 만드는 조건**

accumulation이 큰 batch와 같은 update를 만든다는 문장은 조건부다. sample multiset, supervision mask, global denominator, parameter 시작값이 같고 모든 microbatch가 끝날 때까지 optimizer가 parameter를 바꾸지 않아야 한다. dropout과 기타 stochastic operation의 random draw도 비교 목표에 맞게 통제해야 한다.

각 microbatch의 mean loss를 `K`로 나누는 관용구는 valid count가 모두 같을 때만 정확하다. 길이가 다른 packed sample에서는 raw loss sum을 누적하고 window 전체 valid count를 한 번 적용한다. 또는 각 microbatch sum에 예상 global denominator의 역수를 곱한다. denominator를 backward 뒤 gradient에 적용할 때는 DDP reducer의 평균 계수와 AMP scale 순서를 함께 계산한다.

parameter가 microstep 사이 바뀌면 gradient들이 서로 다른 함수 지점에서 계산된다. pipeline이나 online learner에서 weight version이 바뀔 수 있으므로 단일 accumulation window는 하나의 ParameterGeneration을 가져야 한다. actor rollout의 policy version 문제와 같은 계보 원칙이다.

BatchNorm 같은 batch-dependent state가 있으면 microbatch forward와 combined batch forward 자체가 다를 수 있다. 일반 Transformer의 LayerNorm은 각 token feature 안에서 정규화하므로 이 문제는 작지만 MoE capacity, batch-level contrastive loss, in-batch negatives는 microbatch 분할에 민감하다. objective가 batch interaction을 가지면 gradient만 더해 큰 batch를 복원할 수 없다.

dropout RNG도 exact comparison을 어렵게 한다. combined tensor와 microbatch가 같은 random stream을 같은 logical element에 배정하지 않을 수 있다. correctness의 첫 oracle에서는 dropout을 끄고 exact/tight parity를 본다. 그 뒤 dropout을 켜 distributional equivalence나 명시적 counter mapping을 검증한다.

gradient clipping은 accumulation이 끝난 뒤 한 번 적용해야 큰 batch의 global gradient clipping과 같다. microbatch마다 clip한 vector를 더하면 각 방향의 상대 크기가 바뀐다. optimizer moment와 weight decay 역시 window마다 한 번만 적용한다. scheduler의 step 단위가 microstep인지 committed update인지 명시한다.

**검산 절차.** sample 네 개를 하나의 batch와 `1+3`, `2+2`, `3+1` microbatch로 나눈다. valid-token 수가 서로 다르게 mask를 구성한다. FP64 또는 FP32, dropout off, momentum 없는 첫 step에서 loss sum/count와 gradient를 비교한다. 이어 AdamW state, AMP, DDP를 한 층씩 추가한다. 어느 층에서 최초 차이가 생겼는지 기록한다.

expected-failure에는 microbatch mean의 단순 평균, 중간 optimizer step, microbatch별 clipping, 마지막 no-sync 누락, denominator world-size 중복 보정을 넣는다. 다섯 오류가 서로 다른 gate에서 검출되어야 한다. 그래야 accumulation factor를 바꿀 때 throughput만 달라지고 학습 의미는 보존된다는 주장을 할 수 있다.

실제 리뷰에서는 accumulation factor만 기록하지 말고 `micro_batch_size`, sequence-length distribution, valid-token count, world size, reducer 평균 규칙과 committed update당 sample/token 수를 함께 적는다. 같은 effective batch라는 이름 아래 서로 다른 token objective가 숨을 수 있기 때문이다. OOM 회피를 위해 microbatch를 줄였을 때 scheduler와 warmup이 sample clock인지 token clock인지도 다시 계산한다. 처리량 개선과 수렴 변화가 동시에 보이면 먼저 이 clock과 denominator가 보존되었는지 확인한 뒤 kernel 성능을 해석한다.

resume이 accumulation window 중간을 허용한다면 partial gradient, 이미 소비한 microbatch IDs, numerator/count, no-sync/reducer state와 RNG를 모두 저장해야 한다. 대부분의 시스템은 이 복잡성을 피하려고 committed update 경계에서만 checkpoint한다. 어느 정책이든 장애 직후 sample을 중복 또는 누락하는지 ledger로 검증하며, loss 곡선이 매끄럽다는 이유로 partial-window 복구를 정상 판정하지 않는다.

독립 검토자는 window 경계를 하나씩 이동한 재시작 fixture로 같은 다음 parameter delta와 sample lineage가 복원되는지 확인한다. 이 시험이 accumulation의 운영 의미를 닫는다.

결국 역전파의 산출물은 `.grad` tensor 하나가 아니라 손실에서 committed update까지 이어지는 사건열이다. 최초 차이가 loss·VJP·누적·collective·unscale·clip 가운데 어디서 생겼는지 말할 수 있어야 optimizer 설정을 바꿀 근거도 생긴다. 3장에서는 이 계약을 작은 GPT의 실제 batch와 checkpoint에 대입해, 한 update를 정방향으로 재생하고 장애 지점에서 역방향으로 추적한다.
