# 9장. MLP·MoE·residual

8장의 attention output이 residual stream으로 돌아온 뒤 expert·MLP로 이동하는 상태를 추적한다. 15장은 expert dispatch/combine의 global ID·all-to-all 소유권을 받아 분산하고, 25장은 특정 domain·안전 prompt가 특정 expert로 쏠리는 현상을 red-team slice로 검증한다.

이 장을 읽는 동안 붙들어 둘 질문은 하나다. **한 토큰의 residual state가 어느 함수에서 어떤 모양으로 바뀌고, 그 결정을 역전파가 어떻게 되짚는가.** 이름이 MLP든 sparse MoE든 출발점과 종착점은 `[B,T,C]` residual이다. 차이는 그 사이의 계산을 모든 토큰이 공유하느냐, router가 선택한 expert에만 맡기느냐에 있다. 아래 실행 원장은 뒤의 수식·코드·통신·장애 절을 한 줄로 관통한다.

| 단계 | 논리 상태와 shape | 만드는 연산 | 다음 단계가 의존하는 사실 |
|---|---|---|---|
| residual input | `R∈ℝ[B,T,C]` | attention branch 뒤의 상태 | dtype·stride·TokenID·RNG step |
| normalized input | `X=Norm(R)`, `[B,T,C]` | RMSNorm/LayerNorm | 실제 epsilon, reduction dtype, pre/post-norm 위치 |
| dense gate/up | `G=XW_gᵀ`, `U=XW_uᵀ`, 둘 다 `[B,T,F]` | 두 linear 또는 packed GEMM | gate/up 저장 순서와 logical slice |
| dense activation | `H=SiLU(G)⊙U`, `[B,T,F]` | activation·원소곱 | backward에 저장하거나 재계산할 `G,U` |
| dense down | `D=HW_dᵀ`, `[B,T,C]` | down projection | TP reduction·dropout·residual add 순서 |
| MoE score | `Z=X_fW_rᵀ`, `[N,E]`, `N=B·T_valid` | router linear | softmax dtype, mask, correction bias |
| selection | `(I,A)`, 각각 `[N,k]` | score 변환·top-k·선택 weight 정규화 | `I`는 불연속 결정, `A`는 연속 mixture weight |
| acceptance | assignment rows `≤N·k` | capacity/drop/reroute | offered·accepted·dropped 보존식 |
| dispatch | expert별 ragged `[n_e,C]` | stable permutation·EP all-to-all | global ExpertID, source TokenID, inverse permutation |
| expert compute | expert별 `[n_e,F]→[n_e,C]` | grouped SwiGLU/GEMM | zero-token expert, accumulation dtype, saved tensor |
| combine | `M∈ℝ[N,C]` | reverse all-to-all·inverse permutation·weighted sum | contribution 수와 selected weight 적용 위치 |
| shared/residual | `R'=R+Dropout(M+S)` | shared expert·branch scaling·residual add | 최종 `[B,T,C]`와 identity path |
| backward | 위 표의 역방향 | combine→expert→dispatch→router와 residual identity | gradient owner, collective 역순, loss별 분모 |

여기서 `C`는 hidden size, `F`는 expert intermediate size, `E`는 routed expert 수, `k`는 토큰당 선택 수다. `N`을 단순히 `B·T`라고 놓으면 padding과 sequence packing이 라우팅 통계를 오염시킨다. 실제 원장에는 valid-token mask를 적용한 `N`과 물리적으로 flatten한 row 수를 모두 둔다. top-k 뒤에는 토큰 수가 아니라 **assignment 수**가 `N·k`까지 늘어난다는 점도 중요하다. unique token, offered assignment, accepted assignment, expert row는 서로 다른 분모다.

## 9.1 residual stream에서 dense와 sparse branch를 나눈다

MLP와 MoE를 이해하는 기준선은 residual stream의 입력과 출력이다. norm 뒤 hidden이 dense MLP 또는 router·expert 경로로 들어갔다가 같은 shape로 돌아오는 지점을 고정하면 서로 다른 구현도 동일한 함수 계약으로 비교할 수 있다.

pre-norm decoder block을 단순화하면 dense 경로는 다음과 같다.

`X=Norm(R)`, `G=XW_gᵀ`, `U=XW_uᵀ`, `H=SiLU(G)⊙U`, `D=HW_dᵀ`, `R'=R+D`.

MoE는 `X`까지 같고, 그 뒤 `Z=Router(X)`와 expert 함수 `f_e(X)`로 갈라져 `M_i=Σ_{e∈S_i}a_{i,e}f_e(X_i)`에서 다시 `[N,C]`로 모인다. shared expert `s(X_i)`가 있으면 `M_i+s(X_i)`인지, learned gate를 곱한 `M_i+g_i s(X_i)`인지 실제 forward에서 확인한다. 마지막 residual add가 동일해 보여도 routed branch의 분산, token drop, shared branch가 `D`의 의미를 바꾼다.

SwiGLU의 기하를 “gate가 정보를 거른다”로 끝내지 말자. `U`는 `F`개의 후보 방향에 대한 좌표이고 `SiLU(G)`는 같은 입력에서 계산한 위치 의존적 대각 행렬 `diag(SiLU(G))`이다. 따라서 한 토큰에 대한 중간 변환은

`H=diag(SiLU(W_gX))W_uX`

로 읽을 수 있다. 고정된 선형 subspace를 통과하는 것이 아니라, 입력이 자기 자신에게 적용할 채널별 scale을 만든다. `W_down`은 이 조건부 basis 좌표를 residual 공간으로 되돌린다. gate 값이 0 근처이면 해당 up direction과 그 weight gradient가 함께 약해지고, 큰 양수 영역에서는 SiLU가 거의 선형이 되어 두 projection의 곱이 지배한다. 음수 포화 영역에서는 `SiLU'(G)`가 작아 gate branch gradient가 약해져도 `U` branch의 gradient는 `SiLU(G)`를 통해 별도로 남는다. 그러므로 “SwiGLU가 gradient를 막았다”는 진단은 `dG`, `dU`, `dX_g`, `dX_u`를 나누어 보지 않으면 성립하지 않는다.

upstream을 `Q=∂L/∂D`라 놓으면 `∂L/∂H=QW_d`, `∂L/∂U=(∂L/∂H)⊙SiLU(G)`, `∂L/∂G=(∂L/∂H)⊙U⊙SiLU'(G)`다. 같은 `X`로 돌아오는 branch gradient는 `dX=dGW_g+dUW_u`로 합쳐지고, norm backward를 지난 값에 residual identity gradient가 더해진다. fused kernel은 이 식을 바꾸는 것이 아니라 중간 tensor의 물질화·저장·재계산 위치를 바꾼다. forward parity만 맞고 custom backward의 한 branch가 빠진 오류를 잡으려면 gate/up/down weight gradient와 input gradient를 각각 oracle과 비교해야 한다.

### expert selection과 weighting을 서로 다른 상태로 기록한다

router가 내놓는 `Z`에서 score `P`를 만들고 `I=topk(P,k)`를 고르는 과정과, 고른 expert output에 곱할 `A`를 만드는 과정은 서로 다른 상태 전이다. 예컨대 softmax 전체 분포에서 top-k를 고른 뒤 선택된 값만 합이 1이 되도록 다시 정규화할 수도 있고, 원래 확률을 그대로 쓸 수도 있다. correction bias가 선택 순위에만 쓰이고 mixture weight에는 쓰이지 않는 구현도 있다. 이 세 경우는 expert index가 같아도 output scale과 router gradient가 다르다.

선택 집합 `S_i`를 고정한 작은 영역에서는 `M_i=Σ_e a_{i,e}f_e(X_i)`가 연속이고 `∂L/∂a_{i,e}=⟨∂L/∂M_i,f_e(X_i)⟩`가 router로 돌아간다. 그러나 top-k index 자체는 경계에서 불연속이며 보통 autograd 대상이 아니다. 따라서 router 실패를 다음 셋으로 나눈다.

1. **선택 실패**: 기대 expert가 `I`에 없다. logit·bias·mask·top-k tie와 dtype margin을 본다.
2. **weight 실패**: `I`는 맞지만 `A`의 정규화·scale·detach가 틀리다. selected-weight 합과 main-loss router gradient를 본다.
3. **실행 실패**: `(I,A)`는 맞지만 capacity·dispatch·expert·combine이 그 결정을 보존하지 못한다. assignment ledger를 본다.

load-balance loss는 이 구분을 없애지 않는다. hard assignment fraction `f_e`와 probability mass `P_e`를 결합한 auxiliary objective는 router에 균형 신호를 주지만, selected expert의 실제 output을 잘못 섞은 combine 오류를 고치지 못한다. z-loss는 logit scale을 제어하지만 expert count를 직접 균등화하지 않는다. main, balance, z-loss의 numerator·denominator·coefficient를 각각 기록해야 router gradient를 원인별로 분해할 수 있다.

### norm·branch·residual owner를 framework별 source에서 찾는다

Transformers Qwen3 MoE 고정 revision `550d7b3834670483a4df436541272c055dc364bf`에서는 `Qwen3MoeMLP`, packed `Qwen3MoeExperts`, `Qwen3MoeTopKRouter`, `Qwen3MoeSparseMoeBlock` 순으로 계산 계약을 읽는다. 이 경로는 gate/up/down 식, router softmax·top-k·renormalization, expert execution과 `index_add_` combine을 확인하는 단일-process oracle이다. 그것만으로 capacity drop이나 EP all-to-all을 구현했다고 결론내리면 안 된다.

Megatron 계열에서는 decoder layer가 넘긴 normalized hidden을 `MoELayer`가 받고, router가 routing map/probability를 만든 뒤 token dispatcher의 `token_permutation`이 통신 전 layout과 count를 만든다. expert module이 local expert rows를 계산하면 `token_unpermutation`이 reverse collective와 inverse mapping으로 원래 token layout을 복원한다.

실제 revision에서 `router.py`의 router class, `token_dispatcher.py`의 `MoEAlltoAllTokenDispatcher`, `moe_layer.py`의 `MoELayer`, `experts.py`의 grouped/sequential expert 경계를 함께 고정한다. 이름은 revision마다 이동할 수 있으므로 최종 증거는 repository·commit·path·symbol·body fingerprint와 test fixture를 한 묶음으로 둔다.

두 framework를 연결하는 join key는 클래스 이름이 아니다. 동일한 normalized input `X`, global ExpertID, selected slot, weight, expert output, combined output이다. 먼저 Transformers oracle에서 `(TokenID,ExpertID,A)` 표를 만들고 Megatron dispatcher가 같은 accepted assignments를 어느 rank·local slot으로 옮겼는지 비교한다. EP가 꺼진 parity, frozen routing을 사용한 multi-rank roundtrip, router까지 켠 full backward 순으로 범위를 넓힌다. 그래야 수학 함수 오류와 collective 오류가 섞이지 않는다.

### token·score·position·count 보존식을 먼저 세운다

source rank `s`가 destination rank `d`로 보내는 row 수를 `C_sd`라 하면, collective 진입 전에 `send_count_s[d]=recv_count_d[s]`가 모든 peer pair에서 맞아야 한다. dispatch 뒤 expert `e`가 받은 row 수 `n_e`의 합은 accepted assignment 수와 같아야 한다. reverse all-to-all 뒤 contribution 수 역시 같아야 하며, inverse permutation을 적용한 각 contribution은 원래 `(source rank,TokenID,selected slot)`로 돌아와야 한다.

* `Σ_e offered_e=N·k`는 mask·sentinel을 제외한 정의에서만 성립한다.
* `offered_e=accepted_e+dropped_e`는 capacity 정책의 첫 보존식이다.
* `Σ_e accepted_e=Σ_{s,d}C_sd`는 local hit를 통신 행렬에 포함하는 방식까지 고정해야 한다.
* combine 뒤 token `i`의 contribution 수는 accepted된 selected slot 수와 같아야 한다.

이 네 식 가운데 count 식만 맞아도 permutation swap은 남을 수 있다. payload checksum도 동일한 두 row의 교환을 놓칠 수 있다. 그래서 `source_rank,TokenID,selected_slot,global_expert,destination_rank,local_expert,recv_slot,reverse_slot`을 한 행으로 묶은 원장이 필요하다. forward에서 만든 permutation과 inverse는 backward에서도 expert-output gradient를 같은 expert input row로 돌려보낸다. backward collective 순서가 forward의 단순 텍스트 역순이라는 뜻은 아니다. autograd graph의 dependency와 process-group collective sequence가 모든 rank에서 일치해야 한다.

**imbalance·hang·gradient failure를 한 번에 고치지 않는다**

| 관측 | 첫 분기 | 분리 실험 | 판정에 필요한 증거 |
|---|---|---|---|
| 특정 expert로 요청이 몰림 | router/data인가 capacity인가 | frozen hidden으로 router만 replay | offered histogram·entropy·top-k margin |
| offered는 균등, accepted만 치우침 | local/global capacity domain | 같은 routing으로 capacity만 끄거나 크게 설정 | offered/accepted/dropped와 rounding rule |
| assignment는 같고 rank tail만 증가 | expert compute인가 network인가 | frozen permutation·identity expert roundtrip | pairwise bytes, collective duration, expert GEMM `M` histogram |
| 일부 rank가 collective에서 멈춤 | count/order/group mismatch | payload 전 preflight count·sequence 검사 | rank별 last collective, send/recv matrix, process-group ID |
| expert gradient가 0 | 미선택인가 backward 단절인가 | forced routing으로 해당 expert에 token 주입 | accepted rows, expert output grad, wgrad |
| router는 aux gradient만 받음 | selected weight가 detach됐는가 | main-only·aux-only loss backward 비교 | `∂L_main/∂Z`와 `∂L_aux/∂Z` |
| loss는 정상인데 한 domain 품질 저하 | shared/residual이 failure를 숨기는가 | domain별 forced expert와 branch ablation | domain drop, branch ratio, expert visit age |
| backward에서만 hang | reverse count 또는 gradient support 차이 | forward-only 뒤 expert-output scalar backward | backward collective sequence·zero-token rank |

capacity factor를 올려 hang이 사라졌다고 router 문제가 해결된 것은 아니다. buffer가 커져 timing만 바뀌었거나 drop이 줄어 collective shape가 달라졌을 수 있다. 반대로 auxiliary coefficient를 높여 utilization이 균등해져도 특정 link의 지연은 남는다. 진단 순서는 `X→Z→I/A→accepted ledger→dispatch→expert→combine→gradient`에서 **처음 달라진 상태**를 찾는 것이다. timeout, learning rate, capacity를 동시에 바꾸면 최초 원인을 잃는다.

이 실행 원장을 확보한 뒤에야 옵션의 뜻도 구체화된다. `top_k`는 assignment 수·combine 식·bytes·router gradient support를 바꾸고, `capacity_factor`는 buffer 상한과 학습 objective를 함께 바꾼다. `expert_model_parallel_size`는 global ExpertID의 owner map·collective group·checkpoint reshard를 바꾸며, router compute dtype은 top-k 경계의 churn을 바꾼다. 설정 파일에 값이 들어갔다는 사실과 downstream tensor가 실제로 달라졌다는 effect test를 구분한다.

## 9.2 dense MLP와 SwiGLU의 forward·backward를 계산한다

routing을 추가하기 전에 모든 token이 같은 weight를 통과하는 dense 경로를 완전히 닫는다. projection shape, activation derivative와 saved tensor를 계산해야 sparse expert 안의 MLP와 fused kernel을 올바르게 검산할 수 있다.

### GELU·SiLU·SwiGLU의 함수와 derivative를 비교한다

nanoGPT MLP는 `C→4C→C`와 GELU를 쓴다. GLU 계열은 두 projection 중 하나를 gate로 사용하고, SwiGLU는 `silu(W_gx)⊙W_ux`를 down projection에 넣는다. attention이 token 사이 정보를 섞는다면 MLP는 각 token 위치 안에서 채널을 변환한다. intermediate width가 parameter·activation·FLOP를 함께 바꾼다.

### attention 뒤에 channel-wise nonlinear mixing이 필요한 이유

attention은 value의 가중합으로 token 사이 정보를 이동하지만 같은 layer에서 channel별 nonlinear feature를 만드는 역할은 제한된다. position-wise MLP는 각 `[C]` vector에 같은 함수를 적용해 feature를 확장·선택·압축한다. token 간 통신은 없지만 parameter와 activation FLOP의 큰 부분을 차지한다.

GELU는 `gelu(x)=xΦ(x)`이며 standard normal CDF Φ로 입력 크기에 따라 통과 비율을 부드럽게 바꾼다. derivative는 `Φ(x)+xφ(x)`다. ReLU처럼 음수 전체를 0으로 만들지 않으며 exact erf와 tanh approximation이 있다. source와 port의 approximation을 맞춘다.

nanoGPT `3adf61e`, `model.py:78-92`는 `C→4C` Linear, GELU, `4C→C`, dropout을 연결한다. education config C=32에서 intermediate 128, up/down weight는 각각 `[128,32]`, `[32,128]`이다.

GLU는 두 projection `a=W_ax`, `g=W_gx`에서 `h=a⊙σ(g)` 계열이다. SwiGLU는 `h=silu(g)⊙u`, `y=W_down h`다. Llama Transformers `550d7b3`, `modeling_llama.py:163-189`은 `down(act(gate(x))*up(x))`를 고정한다.

backward는 upstream `dy`에서 `dh=W_downᵀdy`, `du=dh⊙silu(g)`, `dg=dh⊙u⊙silu'(g)`다. 한 branch의 activation이 다른 branch gradient를 scale한다. gate saturation이나 u outlier가 gradient를 막거나 키울 수 있다.

| architecture | projections | intermediate activation | parameter 대략 |
|---|---|---|---:|
| GELU FFN | up, down | `[B,T,F]` | `2CF` |
| GLU/SwiGLU | gate, up, down | 두 `[B,T,F]` | `3CF` |

같은 F에서 SwiGLU가 parameter가 많으므로 model은 F를 조정한다. “4C 대 8/3C” 같은 rule은 alignment와 model family에 조건부다. 실제 config·tensor shape를 쓴다.

**FLOP/bytes.** Linear forward matmul은 대략 `2BTCF`, down도 `2BTFC`다. backward는 input/weight gradient로 더 많은 matmul을 수행한다. activation checkpoint/fusion은 saved gate/up tensor와 HBM traffic을 바꾼다.

**Residual.** pre-norm block `U=H+MLP(Norm(H))`에서 branch를 0으로 만들면 output은 H다. down projection/ dropout 뒤 add 순서를 확인한다. in-place residual은 saved input alias를 깨뜨릴 수 있다.

**반례 1.** exact GELU와 tanh approximation은 보통 가깝지만 bitwise 같지 않다. backward parity도 별도다.

**반례 2.** SwiGLU F를 GELU F와 같게 두고 품질/속도를 비교하면 parameter와 FLOP budget이 다르다.

**실험 9-A.** FP64 scalar/vector에서 GELU/SwiGLU analytic·finite-difference backward를 비교한다.

**실험 9-B.** GELU와 SwiGLU를 같은 parameter 또는 FLOP budget으로 각각 맞춘 두 표를 만든다. quality 실행 전 shape/bytes를 검산한다.

**실패 주입 9-C.** gate/up branch를 바꾸거나 elementwise 곱 대신 합을 사용해 shape-only test가 놓치는 semantic 오류를 golden output으로 잡는다.

## 9.3 router에서 top-k·capacity·accepted assignment를 만든다

MoE의 첫 불연속 경계는 router logit 자체가 아니라 어느 expert 선택이 수락됐는지 결정되는 순간이다. score 계산, top-k, normalization, capacity와 drop을 단계별 ledger로 분리한다.

### router logits에서 accepted assignment까지 보존한다

MoE router는 token hidden state에서 expert score를 만들고 top-k를 고른다. 선택 index와 weight는 forward 값일 뿐 아니라 expert parameter로 gradient를 보내는 경로다. capacity를 넘는 token을 drop하면 throughput은 안정될 수 있지만 그 token의 표현과 loss가 달라진다. drop을 padding처럼 숨겨서는 안 된다.

**Sparse parameter, dense routing.** E experts 전체 parameter를 저장하지만 token 하나는 k experts만 실행한다. router logits `r=xW_r∈R^E`, probability `p=softmax(r)`, selected set `S=topk(p,k)`다. output은 보통 `y=Σ_{e∈S}w_e Expert_e(x)`다. selected weight를 다시 normalize할지 config에 따라 다르다.

Transformers Qwen3 MoE `550d7b3`의 `modeling_qwen3_moe.py:249-290`에는 fp32 router softmax, top-k와 optional renormalization이, `210-247`에는 packed expert 3-D weights와 `index_add_` combine이 구현돼 있다. `316-345`는 sparse layer placement를 config에서 정한다.

### hard top-k의 연속 gradient와 불연속 선택 경계를 나눈다

선택 index 자체에는 일반적으로 gradient가 없다. selected probability weight를 통해 router로 gradient가 흐르지만 선택되지 않은 expert 경로는 main loss gradient가 없다. boundary에서 작은 score 변화가 expert set을 불연속적으로 바꾼다. aux objective와 noise가 exploration/load를 돕는다.

**Dispatch ledger.** token을 flatten한 stable TokenID, selected expert IDs, routing weights, 송신 rank, 수신 rank, slot, accepted/dropped reason, returned position을 기록한다. combine 뒤 원 token order로 정확히 돌아와야 한다.

| state | shape/단위 | owner |
|---|---|---|
| router logits/prob | `[N,E]` | source rank |
| top-k indices/weights | `[N,k]` | source rank |
| expert counts | `[E]` | global/window |
| dispatch permutation | `N·k` entries | all-to-all plan |
| expert input | `[tokens_e,C]` | expert rank |
| expert output | same | expert rank |
| combine output | `[N,C]` | source rank |

**Capacity.** 흔한 capacity는 `ceil(capacity_factor·N·k/E)` 계열이지만 group/rank와 padding policy에 따라 식이 다르다. 초과 token을 drop, reroute, overflow expert, pad-to-capacity 중 무엇으로 처리하는지 읽는다. capacity factor는 memory/communication predictability와 token preservation의 trade-off다.

token drop이 residual skip으로 이어지면 해당 branch output 0일 수 있다. LM loss는 계속 계산되지만 expert update 기회와 representation이 달라진다. dropped token count를 loss denominator 밖으로 숨기지 않는다.

**Expert permutation invariant.** router columns, expert weights와 IDs를 같은 permutation으로 바꾸면 output은 같아야 한다. dispatch/combine mapping test에 유용하다.

**Grouped routing.** DeepSeek류는 expert group을 먼저 제한하고 group 안 top-k를 고르며 shared expert를 모든 token에 적용할 수 있다. Transformers DeepSeek V3 `modeling_deepseek_v3.py:131-230`에는 group-limited top-k, bias와 shared expert 경계가 구현돼 있다. routing bias가 selection을 바꾸는 값과 output mixture weight인지 구분한다.

**반례 3.** k experts를 선택해도 weights를 renormalize하지 않으면 합이 1이 아닐 수 있다. 이름 `top_k`만 보고 convex combination으로 가정하지 않는다.

**반례 4.** token drop 0이라고 expert 균형이 좋은 것은 아니다. capacity가 크면 한 expert collapse도 drop 없이 처리될 수 있다.

**실험 9-D.** uniform router logits에서 probabilities, tie top-k set, expert count와 gradient를 검사한다. tie order는 backend별 달라도 set/output contract를 정한다.

**실험 9-E.** capacity factor sweep에서 accepted/dropped/rerouted, expert utilization, peak buffer, output error와 step time을 기록한다.

**실패 주입 9-F.** dispatch permutation의 두 token을 교환하고 stable TokenID/combine checksum이 탐지하는지 본다.

**균등 분배와 좋은 예측은 같은 목표가 아니다**

**auxiliary loss와 expert imbalance**

load-balancing loss, router z-loss, capacity factor는 main next-token objective 바깥의 제어장치다. 계수를 올리면 균형은 좋아져도 specialization을 약화할 수 있다. expert별 routed token count, accepted/dropped count, probability mass, gradient norm을 함께 기록해야 한다.

**Load balance의 두 통계.** token assignment fraction `f_e`와 평균 router probability `P_e`를 결합하는 Switch-style auxiliary objective가 흔하다. 형태는 model마다 scale이 다르지만 큰 `f_eP_e` concentration을 벌한다. hard count f는 top-k 선택, P는 differentiable probability path를 제공한다.

Transformers Qwen3 MoE `modeling_qwen3_moe.py:514-590`은 attention-mask-aware load balance를 계산하고 `667-686`은 `L_total=L_LM+λL_aux`를 연결한다. padding mask를 빼면 pad token이 expert count와 aux gradient를 왜곡한다.

**Router z-loss.** `z=logΣ_e exp(r_e)`의 제곱 평균 계열은 router logits magnitude가 커지는 것을 벌해 softmax saturation과 낮은 precision 문제를 줄인다. coefficient는 main/aux loss scale과 함께 기록한다. z-loss는 expert count 균형과 다른 목표다.

**Loss denominator.** LM은 supervised token, balance는 routed nonpad token, z-loss는 router-evaluated token을 분모로 쓸 수 있다. 세 scalar를 더하기 전에 reduction과 accumulation/DDP scaling을 맞춘다. aux loss가 microbatch별 mean이면 valid token이 다른 accumulation에서 weighting이 왜곡될 수 있다.

**Gradient 경로.** LM main loss는 selected expert output과 routing weights를 통해 흐른다. aux/z loss는 router parameter에 별도 gradient를 더한다. expert weights는 보통 aux loss를 직접 받지 않는다. total router gradient를 main/aux로 분해해 coefficient 효과를 본다.

**Expert imbalance의 원인.** router 초기화/temperature, data/domain, capacity, group restriction, stale weights, rank-local normalization이 있다. network straggler와 routing skew를 구분한다. expert token histogram이 균등한데 rank 시간이 다르면 compute/network/topology 문제일 수 있다.

**Metrics.** expert별 requested/accepted/dropped, probability mass, mean/max capacity utilization, entropy, Gini/CV, router logit max/z, expert forward/backward time, gradient norm을 window와 rank min/max로 기록한다.

**반례 5.** aux loss가 낮아도 semantic specialization이 좋은지 나쁜지 알 수 없다. 균등성 지표다.

**반례 6.** 평균 expert count가 균등해도 각 microbatch·rank에서 심한 burst가 발생해 all-to-all buffer와 straggler를 만들 수 있다. time-local distribution을 본다.

**실험 9-G.** aux coefficient sweep에서 LM loss, balance/z, entropy, drop, expert gradient와 validation을 같은 token budget으로 비교한다.

**실패 주입 9-H.** padding mask를 aux calculation에서 제거해 sequence length별 expert bias가 생기는지 본다.

**실패 주입 9-I.** router logits scale을 크게 해 saturation, z-loss와 top-k stability를 FP32/BF16에서 비교한다.

## 9.4 assignment를 dispatch·expert·combine 함수로 실행한다

accepted assignment는 token을 expert owner 순서로 재배열하는 통신 계획이 된다. dispatch permutation, expert-local MLP와 inverse combine을 하나의 합성 함수로 적어 token identity와 weight가 왕복하는지 검증한다.

### EP all-to-all의 count와 payload를 같은 원장에 둔다

EP에서는 token을 owner rank로 보내고 결과를 되돌리는 all-to-all이 critical path가 된다. 한 expert가 몰리면 compute와 network가 함께 skew된다. 먼저 router imbalance인지 link 문제인지 분리한다. 동일 routing histogram에서 rank time만 벌어지면 topology/collective를, histogram 자체가 치우치면 router/capacity를 본다.

**EP 소유권.** E experts를 ep_size ranks에 나누고 expert e의 owner를 mapping한다. source rank는 local tokens를 destination별 pack하고 all-to-all로 보낸다. owner는 expert forward/backward를 하고 reverse all-to-all로 output/gradient를 돌려준다. TokenID와 slot permutation이 두 방향에서 보존돼야 한다.

통신 bytes는 대략 dispatch accepted token copies `N·k·C·b`를 forward와 return에 각각 전송하고 backward도 유사 경로를 가진다. metadata/count exchange, padding-to-capacity와 duplicated shared expert는 추가된다. 실제 bytes는 local expert hit와 topology에 따라 다르다.

### 송신지·수신지 count와 buffer allocation을 맞춘다

각 송신지→수신지 send count와 reciprocal receive count가 맞아야 한다. collective에 들어가기 전 count exchange와 buffer allocation을 한다. rank 하나가 다른 token count/collective order를 가지면 hang 또는 buffer corruption이다.

**Megatron 경계.** Megatron-LM의 MoE router, token dispatcher, expert-parallel all-to-all 경로는 Transformers 단일-device reference가 보여주지 않는 ownership·collective를 구현한다. 책의 source note에서는 registry 고정 Megatron revision의 router/dispatcher와 tests를 정확히 고정해야 한다. 여기서는 Transformers 계산 계약과 Megatron 분산 계약을 혼합해 한 구현처럼 쓰지 않는다.

**TP×EP.** expert MLP 내부를 TP로 나누면 dispatch 뒤 expert group 안 TP collective가 추가된다. sequence parallel과 expert tensor parallel의 token layout을 확인한다. DP replica별 experts는 gradient sync group이 일반 dense DP와 다를 수 있다.

**Overlap.** dispatch communication과 local expert compute를 chunk로 overlap할 수 있지만 token ordering, stream/event, buffer lifetime이 state가 된다. overlap 옵션은 단순 속도 flag가 아니라 chunk schedule과 in-flight memory를 바꾼다. correctness reference에서 overlap을 끈다.

**Checkpoint.** expert parameter는 owner rank shard, router/shared expert는 복제 또는 다른 shard 정책을 가진다. checkpoint manifest에 expert global ID→rank/key mapping을 둔다. ep_size 변경은 reshard가 필요하다. optimizer moment와 router state, aux counters도 이동한다.

**Resume.** optimizer-step 경계에서 in-flight token dispatch가 없어야 단순 checkpoint가 가능하다. mid-microbatch recovery를 지원하면 dispatch lease/TokenID와 partial expert results가 필요하다. 일반 공개 trainer가 이를 보장하지 않으면 last committed step으로 rollback한다.

**Failure 1—router collapse.** histogram·entropy·drop이 먼저 변한다. expert ranks 일부만 과부하된다. aux coefficient, router gradient, data shift를 본다.

**Failure 2—network/link.** histogram은 비슷하지만 특정 source-destination transfer와 rank time이 느리다. NCCL trace, topology, NIC/NVLink counters를 본다.

**Failure 3—collective mismatch.** rank별 last collective sequence, send/recv counts, conditional branch를 본다. timeout만 늘리지 않는다.

**Failure 4—expert NaN.** 특정 expert output/gradient만 non-finite면 routed TokenIDs와 input RMS, expert weight를 보존한다. combine 뒤 모든 token으로 NaN이 퍼지기 전에 owner에서 fail-fast한다.

**실패 주입 9-J.** 한 rank send count를 1 줄여 collective preflight가 mismatch를 거부하는지 본다.

**실패 주입 9-K.** 한 expert compute에 delay를 넣어 router skew와 network 없이 straggler detection을 검증한다.

**실패 주입 9-L.** checkpoint expert global ID mapping을 교환해 load 뒤 golden routing/output mismatch를 잡는다.

**실험 9-M.** 같은 frozen routing plan에서 EP on/off reference output·gradient를 비교한다. 그 뒤 routing을 켜 communication/compute 시간을 분리한다.

**Upstream test 범위.** Transformers common/model tests가 single-device router/expert output과 aux loss를 검사해도 EP collective를 증명하지 않는다. Megatron dispatcher tests가 permutation/count를 검사해도 특정 cluster topology 성능을 보장하지 않는다. model·revision·backend·dtype별로 범위를 적는다.

**조사 체크리스트.** MLP activation/width/approximation과 backward를 찾는다. router dtype, top-k, normalization, group/bias를 적는다. capacity/drop/reroute를 찾는다. aux/z formula·coefficient·denominator를 검산한다. TokenID dispatch/combine mapping과 expert ownership을 본다. EP/TP/DP groups, collectives, checkpoint reshard와 tests를 확인한다.

**장애 결정 트리.** output mismatch면 dense expert oracle→router indices/weights→dispatch permutation→expert output→combine 순이다. imbalance면 requested probability→selected count→capacity/drop→rank time을 본다. hang이면 collective sequence/count/group이다. resume mismatch면 expert ID mapping→optimizer shard→router/aux state다.

**실제 인계.** 10장에 nanoGPT dense MLP atlas와 residual checksum, 11장에 dense/expert/router parameter roles, 14장에 fused activation/backward dtype, 15·16장에 EP dispatch bytes/groups, 17장에 expert shard/reshard mapping, 26장에 imbalance/drop/collective metrics를 넘긴다.

**Dense tensor ledger.** MLP input, norm output, gate/up preactivation, activation/gated product, down output, dropout output, residual sum을 각각 기록한다. `[B,T,C]→[B,T,F]→[B,T,C]` shape뿐 아니라 stride·dtype·RMS·finite ratio와 saved tensor를 남긴다. fused SwiGLU가 gate/up을 interleave한 `[B,T,2F]`를 쓸 수 있어 checkpoint layout과 kernel layout을 분리한다.

**Dense backward ledger.** residual upstream gradient는 identity와 branch로 갈린다. down weight/input gradient, gated product의 u/g branch gradient, gate/up weight와 MLP input gradient를 기록한다. 두 branch가 같은 input x로 돌아오므로 input gradient는 합이다. 한 branch가 detach되면 forward는 맞을 수 있어 gradient oracle이 필요하다.

**Activation checkpoint.** MLP intermediate `[B,T,F]` 두 개는 memory가 크다. checkpoint/recompute나 fused activation은 저장을 줄이고 compute를 늘린다. dropout RNG와 MoE routing top-k를 recompute할 때 같은 decision을 사용해야 한다. router tie/nondeterminism으로 expert set이 달라지면 다른 backward다.

**Expert parameter 회계.** dense MLP parameter `~3CF`인 SwiGLU와 E experts의 `~3ECF` total, token당 active `~3kCF` compute를 구분한다. router `CE`, shared expert와 biases가 추가된다. “active parameter”와 checkpoint/optimizer bytes는 다른 분모다.

**Optimizer state.** expert parameter가 한 rank에만 있으면 moment도 owner shard에 있다. router와 shared parameter는 DP/TP policy에 따라 복제·shard된다. expert가 어떤 batch에서 token을 받지 않으면 gradient가 None/zero일 수 있다. optimizer가 sparse participation과 weight decay를 어떻게 처리하는지 본다.

**Router precision.** hidden은 BF16이어도 router logits/softmax를 FP32로 계산할 수 있다. 작은 score 차이가 top-k boundary를 바꾸기 때문이다. cast 위치, top-k input dtype, selected weight cast를 ledger에 둔다. FP32 softmax만으로 deterministic tie가 보장되지는 않는다.

**Top-k tie.** equal logits에서 backend의 index order가 다를 수 있다. deterministic policy가 필요하면 stable tie-break를 정의한다. output invariant는 expert가 서로 다르면 index order/selection에 민감하다. test fixture는 intentionally unique score와 exact tie를 별도로 둔다.

**Capacity의 분산 분모.** capacity를 global N으로 계산하는지 EP group/rank local N으로 계산하는지에 따라 drop이 달라진다. padding 때문에 rank token 수가 다르면 local capacity는 source-rank bias를 만든다. global count collective와 integer rounding/tie rule을 기록한다.

**Token duplication.** top-k에서 token 하나가 k expert로 복제된다. dispatch count 합은 accepted expert assignments이며 unique token 수와 다르다. combine weight sum, dropped assignment와 fully dropped token을 별도 센다. top-2 한 branch drop 후 남은 weight를 renormalize할지도 policy다.

**Shared expert.** 모든 token이 통과하는 shared expert는 dense capacity를 제공하고 routed expert와 합쳐진다. routed collapse가 있어도 shared path가 output을 만들어 loss가 급격히 깨지지 않을 수 있다. router failure가 loss curve에 숨는 반례다.

**Aux-loss-free balancing.** selection bias를 조정하는 per-expert bias를 별도 update할 수 있다. bias가 router selection에만 쓰이고 mixture weight에는 직접 더해지지 않는지 source를 읽는다. update rule과 checkpoint state, distributed count 집계를 저장한다. “aux loss가 없다”는 balance state가 없다는 뜻이 아니다.

**Router gradient clipping.** 전체 model global clip에 router가 포함되면 expert/main gradient 규모가 router update를 제한할 수 있다. 별도 clip/group을 쓰면 optimizer semantics가 달라진다. aux coefficient sweep은 clip activation 비율도 함께 본다.

**Z-loss 수치.** logsumexp는 stable하게 계산해야 한다. 이미 router logit이 inf이면 z-loss도 inf다. z-loss coefficient를 높여 증상을 가리는 대신 hidden/router weight outlier의 최초 원인을 찾는다.

**실험 9-N—finite-difference router.** top-k set이 변하지 않는 작은 perturbation 구간에서 selected weight를 통한 router gradient를 finite difference와 비교한다. selection boundary에서는 비미분성을 test expectation에 명시한다.

**실험 9-O—expert permutation.** router columns, expert parameter, global IDs와 ownership mapping을 같은 permutation으로 바꾸어 output·gradient 불변을 검사한다. 하나만 permutation해 failure를 확인한다.

**실험 9-P—dispatch conservation.** unique token N, requested assignments Nk, accepted+dropped assignments, destination send/receive, combine outputs의 보존식을 검사한다. padding capacity slot은 real assignment와 분리한다.

**실험 9-Q—dense equivalence.** k=E이고 모든 expert가 같은 함수, weights sum 1이면 weighted output이 그 함수와 같은지 본다. dispatch/combine reference다. 일반 MoE quality equivalence 주장이 아니다.

**실패 주입 9-R—stale routing.** forward router indices와 backward/recompute indices를 다르게 만들어 checkpointed MoE가 fail하도록 한다. selected IDs checksum을 saved state로 비교한다.

**실패 주입 9-S—fully dropped token.** capacity 0/극소로 token 모든 assignment를 drop해 residual-only behavior와 drop reason metric을 확인한다. silent zero output을 정상 expert output으로 세지 않는다.

**실패 주입 9-T—aux denominator.** padding 많은 microbatch와 적은 microbatch의 aux mean을 단순 평균해 global nonpad reference와 차이를 재현한다.

**실패 주입 9-U—expert optimizer 누락.** reshard/load 뒤 한 expert parameter가 optimizer group에서 빠지게 한다. routed gradient는 있지만 delta가 0인 expert를 update ledger로 잡는다.

**EP 통신 스케줄.** count exchange, input permutation/pack, all-to-all, grouped expert GEMM, reverse all-to-all, unpermute/weighted combine 순이다. backward는 output gradient를 selected weights와 expert outputs에 분기하고 역 dispatch한다. overlap 구현은 chunk마다 이 순서를 pipeline한다.

**Grouped GEMM.** expert별 token 수가 가변이므로 작은 GEMM E개 대신 grouped GEMM으로 실행할 수 있다. expert offsets/counts와 weight pointer 배열이 kernel contract다. capacity padding은 shape regularity를 주지만 wasted FLOP를 만든다. tokens/expert histogram과 GEMM tile utilization을 함께 본다.

**Topology placement.** 같은 node/NVSwitch 안에 EP group을 둘지 NIC를 건널지 all-to-all 비용이 크게 다르다. TP와 EP group 축을 어떻게 배치하는지 rank map으로 그린다. expert imbalance와 network oversubscription이 겹치면 평균 bandwidth 하나로 원인을 분리할 수 없다.

**Expert redundancy와 fault.** 한 expert owner가 죽으면 일반 checkpoint 없는 live reroute는 model 함수를 바꾼다. replica expert를 두는 design이 아니라면 마지막 committed checkpoint에서 전체 group을 복구한다. partial expert output을 다른 rank가 재사용할 exactly-once protocol이 공개되지 않았다면 지원한다고 쓰지 않는다.

**Checkpoint reshard test.** ep_size 2에서 저장하고 4로 load해 global expert ID별 parameter/moment checksum, router column mapping, golden dispatch/output을 비교한다. load 성공만으로 expert identity 보존을 증명하지 않는다. shared expert와 router replica도 확인한다.

**성능 지표 분모.** tokens/sec는 unique token인지 expert assignment인지 명시한다. expert TFLOP는 accepted assignments, network bytes는 remote assignments, drop rate는 requested assignment 또는 unique token 분모를 구분한다. capacity padding FLOP를 useful FLOP와 나눈다.

**모니터링 rule.** expert CV/Gini, max/mean tokens, fully dropped token, aux/z, router entropy, per-expert forward/backward time, all-to-all p50/p99, send matrix skew, expert gradient/delta를 window로 본다. threshold는 model/E/k/capacity/topology 조건에 붙인다.

**NaN 결정 트리.** residual/norm input이 finite인지 본다. router logits/prob와 selected weights를 본다. 특정 expert input/activation/weight/output을 본다. combine에서 first NaN인지 확인한다. aux/z reduction을 분리한다. BF16 grouped GEMM을 reference dense GEMM과 비교한다. failing TokenIDs를 보존한다.

**Loss plateau 결정 트리.** fully dropped/selected k, router entropy와 expert update count를 본다. shared expert만 학습되는지 확인한다. aux coefficient와 clip을 본다. expert optimizer/moment mapping을 본다. source/domain별 routing과 data mixture를 연결한다.

**Throughput 결정 트리.** requested/accepted histogram, capacity padding과 grouped GEMM utilization을 본다. send matrix와 topology를 본다. all-to-all/compute overlap timeline을 본다. 특정 expert straggler와 link straggler를 frozen routing plan으로 분리한다. dense fallback/CPU sync metric을 확인한다.

**Hang 결정 트리.** 모든 rank의 EP group, collective sequence, send/recv counts를 비교한다. conditional empty-token rank가 collective를 skip했는지 본다. TP/EP nested collective ordering과 stream event를 본다. timeout 전 flight recorder와 routing count를 보존한다.

**복구 결정 트리.** CheckpointID/ep_size/rank map을 확인한다. global expert ID→key/owner와 optimizer mapping을 비교한다. router/shared state와 aux-balance bias/counter를 본다. golden routing/output 뒤 one-step delta를 검사한다.

**Test pyramid.** scalar activation derivative, dense MLP forward/backward, expert function, router top-k/weight, capacity/drop, permutation/conservation, single-process MoE, multi-rank dispatcher, reshard checkpoint, failure injection, topology benchmark 순이다. 아래 단계 실패를 성능 test로 덮지 않는다.

**Source/test note.** Transformers 고정 source는 Qwen3 MoE packed experts/router/aux-total loss와 DeepSeek group/shared routing의 계산 reference다. Megatron 고정 checkout에서는 token dispatcher, all-to-all, grouped GEMM, router tests와 checkpoint mapping을 별도 source note로 고정한다. 정확 line을 확인하지 않은 Megatron 동작은 본문에서 특정 함수 사실로 승격하지 않는다.

**최종 manifest.** 첫 묶음은 계산 형태를 고정한다. MLP 종류와 중간 폭 F, activation 근사법, dense·expert·router·shared parameter group을 기록한다. 둘째 묶음은 routing 정책을 고정한다. top-k, 재정규화, capacity와 drop 정책, aux/z 식의 계수와 분모를 적는다. 셋째 묶음은 실행 상태를 고정한다. 전역 expert 소유권, dispatch·combine checksum, routing metric, collective group과 전송 byte, optimizer·checkpoint mapping을 연결한다.

**9장 완료 조건.** dense activation의 수식과 backward를 손검산한다. token 하나가 router에서 k expert, EP ranks, combine을 거쳐 원 position으로 돌아오는 ID chain을 추적한다. main/aux/z gradient와 분모를 구분한다. imbalance·network·collective·checkpoint failure를 서로 다른 관측으로 격리할 수 있어야 한다.

**독자 확인 문제.** N=1024, E=8, k=2, capacity factor 1.25에서 정의한 capacity 식으로 expert slot을 계산하고 requested assignment 2048과 padded/accepted/drop 분모를 구분한다. C=4096, BF16에서 remote assignment 1000개의 한 방향 activation bytes를 계산한다. aux loss가 균형을 개선해도 validation을 보장하지 않는 이유를 적는다.

**회귀 gate.** dense MLP FP64 derivative, SwiGLU branch gradient, router stable top-k, expert permutation, dispatch conservation, capacity drop, padding-aware aux, EP reference parity, reshard one-step delta를 순서대로 실행한다. 각 gate는 source commit, config, dtype, seed, expected invariant와 `NotExecuted/Passed/Failed`를 가진다.

**마지막 인계.** 10장이 읽을 dense residual checksum, 15장이 읽을 global expert ownership/send matrix, 17장이 읽을 expert/optimizer shard map이 같은 model revision과 RunID를 가리켜야 한다. 한 manifest에서 이 세 view가 reconciliation되지 않으면 MoE stack을 출판 가능한 상태로 승인하지 않는다.

## 9.5 dense baseline에서 sparse 함수 합성으로 확장한다

SwiGLU의 gate와 MoE router는 모두 ‘선택’처럼 보이지만 하나는 연속 곱이고 다른 하나는 token-level assignment다. 두 경로를 같은 tensor atlas에 놓되 gradient와 자원 소유권이 갈라지는 지점을 명시한다.

### gate·up·down projection의 gradient를 branch별로 추적한다

일반적인 gated MLP는 `u=xW_up`, `g=xW_gate`, `y=(φ(g)⊙u)W_down`으로 쓸 수 있다. hidden이 `[N,C]`, intermediate가 F라면 up과 gate는 `[C,F]`, down은 `[F,C]`다. `N`은 batch와 sequence를 평탄화한 token 수다. SwiGLU는 `φ(g)=SiLU(g)=gσ(g)`를 사용한다. 구현은 gate와 up projection을 하나의 `[C,2F]` matrix로 합친 뒤 split할 수 있다. split 순서가 바뀌어도 shape는 맞으므로 checkpoint conversion에서 흔한 조용한 오류다.

backward를 쓰면 두 branch의 역할이 선명해진다. `a=φ(g)⊙u`, upstream `dA=dY W_downᵀ`라 하면 `dU=dA⊙φ(g)`, `dG=dA⊙u⊙φ'(g)`다. 이어 `dW_up=xᵀdU`, `dW_gate=xᵀdG`, `dX=dU W_upᵀ+dG W_gateᵀ`이며 down 경로도 더해진다. gate가 포화되거나 u가 극단적으로 작으면 한 branch gradient가 약해질 수 있다. activation output만 관측하지 않고 gate/up 값과 각 branch gradient RMS를 보는 이유다.

SiLU derivative는 `σ(g)+gσ(g)(1-σ(g))`다. GELU는 exact erf 식과 tanh approximation이 있어 source와 kernel이 같은 변형을 쓰는지 확인한다. random normal 입력에서는 차이가 작아 보일 수 있으므로 큰 양수, 큰 음수, 0 근처와 dtype 경계 fixture를 쓴다. fused activation kernel은 forward뿐 아니라 saved tensor와 backward 식을 reference에 대조한다.

intermediate size는 parameter와 FLOP를 결정한다. gated MLP는 projection이 세 개이므로 같은 C와 F에서 두 projection을 쓰는 GELU MLP보다 parameter가 많다. 많은 architecture는 이 차이를 상쇄하려 F를 조정하고 hardware multiple로 round한다. config의 `intermediate_size`를 일반적인 `4C`로 추측하지 않고 실제 값을 읽는다. tensor parallel shard divisibility와 grouped GEMM tile 조건도 round 선택에 영향을 준다.

tensor parallel에서는 gate/up output F축을 나누고 down projection에서 partial output을 reduce하는 column-parallel/row-parallel 조합을 자주 쓴다. 정확한 collective는 framework layout에 달려 있다. sequence parallel이 함께 있으면 input activation의 token 축 owner도 바뀐다. module 이름만 보고 all-reduce 위치를 단정하지 않고 weight shard 축, local activation shape, collective 전후 checksum을 그린다.

residual add는 MLP output shape가 `[B,T,C]`로 돌아왔다는 것 이상이다. dropout 또는 stochastic depth RNG가 rank와 recomputation에서 같은 mask를 써야 하고, mixed precision add가 어느 dtype에서 일어나는지 확인한다. in-place add는 activation lifetime을 줄이지만 checkpoint recomputation이나 hook이 원래 residual을 필요로 하면 alias 문제를 만든다. `h_before`, normalized branch input, branch output, `h_after` checksum과 storage 관계를 기록한다.

### fused kernel을 eager oracle과 shape·dtype별로 비교한다

gate와 up GEMM을 합치면 weight layout이 `[2F,C]` 또는 `[C,2F]`로 저장될 수 있다. 그 뒤 fused SwiGLU가 split·activation·multiply를 한 kernel에서 수행한다. down GEMM까지 완전히 fuse하기는 중간 reduction과 weight reuse 때문에 다른 tradeoff가 있다. quantized 또는 FP8 path는 scale granularity와 amax history를 추가한다. 같은 module class가 shape/dtype에 따라 서로 다른 backend로 dispatch될 수 있다.

성능 분석은 GEMM M/N/K, stride, alignment, tile, epilogue, intermediate materialization byte를 적는다. M=N_tokens가 작으면 큰 F/C에서도 GEMM utilization이 낮다. packing과 microbatch가 M을 바꾸고 sequence parallel이 local M을 줄인다. forward TFLOP만 보고 backward weight-gradient GEMM과 activation-gradient GEMM을 빼지 않는다. activation recomputation은 memory를 줄이지만 gate/up GEMM을 다시 실행한다.

correctness oracle은 작은 `[N,C,F]`에서 unfused FP64 계산이다. fused BF16/FP8 output, dX, 세 weight gradient를 비교한다. gate/up concatenation order, transpose, stride, bias 유무, activation approximation, scale 적용 시점을 한 축씩 바꾼다. kernel이 non-contiguous input을 지원하지 않아 암묵 copy하거나 잘못 읽는지도 확인한다. profiler의 빠름은 함수 동일성을 증명하지 않는다.

### MoE output을 weighted expert 함수 합으로 정의한다

**dispatch permutation과 combine inverse를 증명한다**

token hidden `x_n`에 router logits `r_n=x_nW_r`를 계산하고 top-k expert 집합 `S_n`을 고른다. 선택 확률을 전체 softmax에서 가져올지 top-k 안에서 다시 normalize할지는 서로 다른 함수다. 출력은 일반적으로 `y_n=Σ_{e∈S_n} α_ne E_e(x_n)`다. capacity 때문에 assignment가 거부되면 renormalize, zero, residual fallback, shared expert 가운데 무엇을 하는지 명시한다.

nonpadding token이 N개면 requested assignment 수는 `N·k`다. expert별 capacity를 `ceil(capacity_factor·N·k/E)`처럼 정의할 수 있으나 framework마다 분모와 rounding이 다르다. sequence 단위인지 global DP group 단위인지도 확인한다. local rank가 자기 token만 보고 capacity를 정하면 world size와 partition에 따라 drop이 달라질 수 있다. 실제 source 식과 counter를 읽고 dense oracle fixture에 맞춘다.

router top-k는 동률과 낮은 precision에 민감하다. stable expert ID tie-break가 없으면 device/backend가 assignment를 바꿀 수 있다. noise나 jitter를 학습 중 추가한다면 RNG state와 eval 비활성화를 확인한다. softmax를 FP32로 계산하는지, logits clip이나 z-loss가 있는지 기록한다. router logits가 finite여도 한 expert로 쏠릴 수 있으므로 entropy와 assignment histogram이 필요하다.

dispatch permutation은 `(token_id, slot, expert_id, weight)` 원장을 만든다. expert별로 정렬해 contiguous input을 만들고 all-to-all을 거쳐 owner rank에 보낸다. expert output은 역 permutation으로 원 token/slot에 돌아와 weighted sum된다. 보존식은 requested=accepted+dropped, accepted send=accepted receive, 각 accepted assignment가 정확히 한 output 또는 명시적 failure를 갖는다는 것이다. unique token 수와 assignment 수를 섞지 않는다.

shared expert는 모든 token에 적용되는 dense 경로이고 routed experts와 합쳐진다. DeepSeek 계열처럼 shared expert와 routed expert를 함께 쓰는 구조에서는 FLOP, parameter group, residual scaling을 별도로 센다. shared path가 정상이라 routed path가 망가져도 loss가 완전히 붕괴하지 않을 수 있다. routed output을 zero하는 failure injection으로 shared-only baseline과 비교한다.

fine-grained experts는 큰 expert를 더 작은 여러 expert로 나눠 선택 수를 늘릴 수 있다. 총 activated intermediate capacity, routing 조합, communication assignment 수가 함께 바뀐다. expert 수 증가를 곧바로 model capacity 증가로 해석하지 않는다. Qwen/DeepSeek/GLM 계열 config에서 expert 수, top-k, shared count, intermediate size, group routing 옵션을 실제 checkpoint source와 연결한다.

**prediction loss와 balancing objective를 별도 scalar로 둔다**

load-balancing auxiliary loss는 expert별 token fraction과 평균 router probability의 곱을 사용해 균형을 유도하는 변형이 흔하다. 정확한 상수 E, k, denominator와 padding mask를 source에서 확인한다. top-k discrete assignment fraction에는 직접 gradient가 없거나 stop-gradient로 다뤄질 수 있고 probability 경로가 router를 움직인다. microbatch local 통계인지 global group 통계인지에 따라 gradient와 balance가 다르다.

z-loss는 `logsumexp(router_logits)^2` 같은 항으로 logits 규모를 제한할 수 있다. load balance와 목적이 다르다. coefficient가 지나치면 main task보다 routing regularizer가 update를 지배할 수 있다. main loss sum/count, balance numerator/denominator, z loss와 각각의 scaled contribution을 기록한다. total scalar 하나만 보면 원인을 찾을 수 없다.

auxiliary-loss-free balancing은 expert bias를 routing 선택에 사용하되 main score 또는 gradient와 다른 update rule로 조정할 수 있다. 이 bias가 model parameter인지 runtime state인지, checkpoint되는지, 어느 counter로 갱신되는지 확인한다. 학습 resume에서 bias가 빠지면 같은 model weight라도 다음 assignment가 달라진다. 이름만 보고 auxiliary loss가 완전히 없다고 단정하지 않고 다른 balance controller를 찾는다.

균등 assignment는 품질의 필요조건도 충분조건도 아니다. domain 또는 token 유형에 따라 전문화가 생기면 불균형이 유용할 수 있다. 반대로 dead expert는 capacity를 낭비하고 optimizer state가 stale해진다. entropy, Gini, max/mean뿐 아니라 expert별 token semantics, validation ablation, gradient/delta, overflow/drop을 함께 본다. balance metric 개선만으로 task metric 개선을 주장하지 않는다.

**expert parallelism은 collective state machine이다**

**all-to-all의 count와 payload를 동일한 원장으로 검증한다**

EP rank마다 destination별 send count를 만든다. count exchange가 끝나야 receive buffer 크기를 알 수 있다. 입력 activation, token metadata, routing weight가 같은 permutation과 count를 사용해야 한다. 한 payload만 다른 dtype byte 계산을 쓰거나 empty destination을 생략하면 collective hang 또는 조용한 misalignment가 생긴다. count checksum과 first/last assignment ID를 rank별 flight recorder에 남긴다.

communication byte는 accepted remote assignments×hidden width×element size가 기본이지만 metadata, scale, alignment padding, backward payload가 더해진다. top-k가 커질수록 같은 token이 여러 destination으로 복제될 수 있다. local expert assignment는 network를 쓰지 않으므로 send matrix와 rank placement가 중요하다. 평균 remote fraction만으로 hot pair와 NIC oversubscription을 숨기지 않는다.

overlap 구현은 permutation chunk, count readiness, NCCL collective, grouped GEMM, reverse collective를 stream event로 연결한다. compute가 통신을 완전히 숨긴다는 표현 대신 critical path에서 겹친 시간과 exposed tail을 측정한다. 작은 expert batch나 심한 imbalance는 일부 rank GEMM이 빨리 끝나도 가장 느린 rank를 기다리게 한다. timeline을 routing histogram과 연결한다.

empty-token expert와 empty-token rank는 정상 edge case다. 어떤 rank도 token을 받지 않았어도 collective sequence에는 참여해야 할 수 있다. conditional branch로 all-to-all을 skip하면 다른 rank가 기다린다. zero-count payload를 backend가 지원하는지, dummy buffer가 필요한지 고정 test로 확인한다. gradient accumulation microbatch마다 routing이 달라지므로 한 번 성공한 topology test로 충분하지 않다.

EP, TP, DP가 함께 있으면 process group membership과 collective order를 명시한다. router 또는 shared expert parameter는 어느 group에서 gradient reduce되는가. routed expert는 DP replica가 있는가. TP-sharded expert의 내부 GEMM collective와 EP dispatch가 어떤 순서인가. rank map을 표로 만들고 각 parameter의 global ID, shard axis, replica group을 연결한다.

**checkpoint와 장애 복구에서 expert identity를 보존한다**

checkpoint key의 local index를 global expert ID로 착각하면 EP size 변경에서 expert가 뒤섞인다. manifest는 global expert ID, owner rank/group, local slot, tensor shard range, optimizer moment와 router column을 연결한다. reshard는 parameter만 옮기지 않고 moment, FP32 master, balance bias, update counter를 함께 옮긴다. load 뒤 global checksum과 golden routing을 검사한다.

장애가 collective 도중 발생하면 일부 rank가 expert output을 계산했어도 update는 commit되지 않았다. exactly-once partial reuse protocol이 없다면 마지막 consistent CheckpointID에서 전체 group을 재시작한다. rank 하나만 새 process로 바꾸고 다른 rank의 in-flight state를 유지하면 collective sequence와 RNG가 갈릴 수 있다. recovery 문서에 실제 지원 범위를 적는다.

expert parameter corruption은 global loss만으로 늦게 발견될 수 있다. expert별 weight checksum, output finite, gradient norm, delta norm을 저빈도로 검사한다. 특정 expert에 routing이 적으면 corruption이 오랫동안 노출되지 않는다. golden probe token을 expert별로 강제 routing하는 진단은 production routing 함수와 분리된 health check로 사용할 수 있다. 강제 routing 결과를 실제 품질의 증거로 과장하지 않는다.

world-size나 EP size 변경 뒤 loadable과 numerically equivalent를 구분한다. assignment tie-break, capacity domain, all-reduce order가 바뀌면 동일 weight에서도 routing과 수치가 달라질 수 있다. topology-portable 계약은 global expert identity와 데이터 안전을 보장하되 exact trajectory는 보장하지 않을 수 있다. 요구 수준을 명시하고 golden one-step delta로 검증한다.

**실제 모델 해부로 넘어가기 전의 종합 판정**

**dense·SwiGLU·MoE를 동일 입력의 oracle로 비교한다**

교육 fixture는 `B=2,T=4,C=8,F=16,E=4,k=2`처럼 손으로 추적 가능한 크기를 쓴다. token ID와 position을 7장 manifest에서 받아 norm output을 고정한다. dense MLP에서는 gate/up preactivation, activated product, down output, residual sum을 저장한다. MoE에서는 router logits/probability, selected expert/weight, capacity decision, dispatch position, expert output, combined output을 저장한다.

첫 oracle은 expert 네 개가 서로 다른 단순 선형 변환을 하게 만든다. 그러면 token이 잘못된 expert나 position으로 돌아왔을 때 값으로 식별할 수 있다. 둘째 oracle은 모든 expert weight를 같은 dense MLP로 복사하고 top-1 routing을 사용해 routed output이 dense reference와 같은지 본다. 셋째는 top-2 weight 합과 renormalization을 검증한다. 복잡한 실제 checkpoint 전에 permutation과 combine을 닫는다.

backward atlas는 router, selected weight, 각 expert parameter, input residual gradient를 포함한다. top-k index 자체는 미분 불가능하지만 selected probability 경로는 gradient를 가진다. dropped assignment와 fully dropped token이 어떤 gradient를 받는지 정책과 맞춰 본다. aux/z loss를 하나씩 켜 main-loss-only gradient와 차이를 분해한다.

실제 모델에서는 config, model source, checkpoint tensor 세 출처를 대조한다. Qwen 계열의 dense/MoE layer 선택, DeepSeek 계열의 routed/shared expert와 group routing, Gemma 계열 gated MLP의 projection 명명, GLM 계열 residual/norm 배치를 근거가 확인된 범위에서 읽는다. 이름이 같은 `gate_proj`라도 concatenated tensor인지 별도 tensor인지 source와 shape로 확인한다. model card 요약만으로 함수 경로를 채우지 않는다.

옵션 change sheet에는 `num_experts`, `top_k`, `capacity_factor`, `drop_tokens`, `renormalize`, router dtype/noise, aux/z coefficient, shared expert 수, expert intermediate size, EP/TP size, grouped GEMM, overlap를 넣는다. 각 옵션이 config validation, module factory, tensor shape, mutable state, collective, checkpoint schema, metric과 test를 무엇을 바꾸는지 적는다. 효과 설명은 throughput 또는 quality 한 단어로 끝내지 않는다.

현장 실험은 dense baseline, MoE function parity fixture, single-rank routing, multi-rank dispatch, imbalance injection, empty-rank, expert corruption, checkpoint reshard 순으로 승격한다. 대규모 학습 런타임을 실행하지 않는 조건에서도 source와 test를 감사하고 필요한 실행 계약을 구체적으로 남길 수 있다. 실행하지 않은 backend 조합은 통과로 표시하지 않는다.

10장으로 넘길 최종 handoff는 model revision, layer index, input/output residual checksum, norm 위치, MLP type와 activation, router formula와 dtype, expert ownership, dispatch ledger, main/aux/z loss denominator, parameter/optimizer group이다. 10장은 이 자료로 config의 한 줄이 실제 token forward와 backward, checkpoint state에 어떤 함수를 만드는지 끝까지 추적한다.

**고정 소스 좌표.** Qwen3 MoE의 router와 packed expert 경로는 [Transformers 고정 revision의 `modeling_qwen3_moe.py:210-290`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L210-L290)에서 확인한다. 이 좌표가 보여 주는 단일 구현과 다른 framework의 capacity·dispatcher·EP 정책을 동일하다고 가정하지 않는다.

**이 장이 넘기는 것.** MLP/MoE output, routing ledger, auxiliary-loss denominator, residual checksum을 10장에 넘긴다.

**다음 장에서 깨질 수 있는 것.** 개별 module invariant가 모두 맞아도 config, checkpoint key, tied weight가 서로 다른 revision이면 전체 모델은 달라진다.

**검증 체크포인트.** dense MLP shape, gate finite range, routed/accepted/dropped token 보존식, expert별 gradient, residual shape identity를 확인한다.

## 9.6 residual·router·expert 구현을 실제 모델에서 찾는다

논문식 이름을 model class에 그대로 투영하지 않는다. Qwen·DeepSeek·GLM 계열의 norm 위치, shared expert, router state와 residual 결합을 실제 caller와 tensor shape에서 복원한다.

### gated MLP의 기하를 residual gradient까지 추적한다

SwiGLU 계열을 `a=xW_g`, `b=xW_u`, `m=SiLU(a)⊙b`, `y=mW_d`로 쓰면 두 상향 projection은 역할이 다르다. `W_g`는 좌표별 통과율을 만들고 `W_u`는 전달할 값을 만든다. 두 weight를 checkpoint conversion에서 바꾸어 넣어도 shape는 같지만 함수는 같지 않다. 작은 비대칭 weight와 음수·0·양수 입력을 넣어 preactivation부터 비교해야 한다.

backward는 `dm=dy W_d^T`, `db=dm⊙SiLU(a)`, `da=dm⊙b⊙SiLU'(a)`다. gate가 큰 음수이면 value 경로까지 약해지고, 0 근처에서는 SiLU derivative가 단순 binary gate와 다르다. activation 이름만 확인하지 않고 정확한 식, approximation 여부, compute dtype을 확인한다. fused kernel이 sigmoid approximation을 쓴다면 허용오차를 saturation 영역과 0 근처에서 따로 정한다.

`gate_up_proj`처럼 두 projection을 하나의 packed weight로 저장할 수 있다. split order가 `[gate,up]`인지 `[up,gate]`인지, output dimension이 어느 축인지, quantized scale이 두 절반에 공유되는지 확인한다. checkpoint key가 하나라고 수학적으로 하나의 projection인 것은 아니다. packing은 저장과 kernel dispatch 계약이며 semantic tensor atlas에는 두 logical tensor를 유지한다.

intermediate size는 parameter 수뿐 아니라 tensor-core tile, TP 분할, activation peak에 영향을 준다. hidden `C`, intermediate `F`일 때 activation은 대략 두 개의 `[tokens,F]` preactivation과 product를 거친다. fusion 여부에 따라 저장되는 tensor가 달라진다. activation checkpointing이 gate/up을 재계산하는지, product만 저장하는지에 따라 backward peak와 compute가 바뀐다.

tensor parallel에서 column-parallel gate/up은 F축을 shard하고 local product를 계산할 수 있다. down projection이 row-parallel이면 partial `[tokens,C]`를 reduce한다. gate와 up이 서로 다른 shard mapping을 가지면 elementwise product가 다른 좌표를 곱하는 silent bug가 난다. 두 packed half의 global intermediate index를 같은 manifest로 검증한다.

gradient accumulation에서 MLP parameter gradient는 microbatch마다 더해지지만 residual gradient는 즉시 앞 layer로 흐른다. FP32 gradient accumulator, BF16 activation recompute, fused backward의 reduction order가 one-step delta에 영향을 준다. forward parity만 통과한 kernel을 학습 지원으로 인정하지 않고 `dx`, `dW_g`, `dW_u`, `dW_d`를 모두 비교한다.

**residual은 덧셈이 아니라 깊이 방향의 상태 전달 계약이다**

pre-norm block `h'=h+F(N(h))`에서 identity 경로는 hidden state를 그대로 전달한다. 그러나 branch scale이 layer마다 누적되면 residual RMS가 커질 수 있다. post-norm, sandwich norm, residual multiplier, DeepNorm류 scaling은 Jacobian과 초기화 가정을 바꾼다. model config의 `rms_norm_eps`만 보고 residual architecture가 같다고 판정할 수 없다.

parallel residual에서는 attention과 MLP가 같은 normalized input을 받고 `h'=h+A(N(h))+M(N(h))`처럼 합쳐진다. sequential residual은 attention 결과가 MLP 입력에 영향을 준다. parameter 이름과 shape가 같아도 계산 graph가 다르다. attention branch를 0으로 둔 fixture와 MLP branch를 0으로 둔 fixture, 둘 다 켠 fixture로 호출 순서와 합산 위치를 분리한다.

mixed precision에서 두 branch가 BF16이고 residual만 FP32일 수도 있다. fused add-norm kernel이 어느 dtype에서 합하고 무엇을 저장하는지 확인한다. residual을 activation dtype으로 매 layer 반올림하는 경로와 FP32 buffer를 유지하는 경로는 깊어질수록 갈린다. layer 1만 보는 test가 아니라 동일 단순 block을 여러 번 반복한 drift fixture가 필요하다.

mHC 같은 residual mixing 구조를 볼 때 이름이나 모델 카드 도식에서 멈추지 않는다. hidden stream이 몇 개인지, mixing matrix가 input-dependent인지, 어떤 축에서 normalize/constraint되는지, 기존 residual과 호환되는 초기화가 무엇인지, backward에서 mixing parameter와 stream에 gradient가 어떻게 분기되는지를 묻는다. 일반 `h+F(h)`와 같은 tensor shape라는 이유로 drop-in equivalent라 부르지 않는다.

GLM 계열의 특정 release가 mHC를 사용한다고 주장하려면 해당 모델 revision의 config key, module constructor, forward symbol, checkpoint tensor를 모두 고정한다. 로컬에 근거가 없으면 “가능한 설계”와 “그 모델의 구현”을 분리해 쓴다. 모델 카드의 용어를 Transformers 구현에 자동 투사하지 않는다. source가 추가되면 golden layer fixture로 stream 수와 mixing 식을 닫는다.

### router를 score가 아닌 allocation state machine으로 읽는다

**logit에서 accepted assignment까지 모든 상태를 보존한다**

router는 보통 `r=xW_r`에서 expert logit을 만들지만 이후 단계가 모델마다 다르다. softmax 전 top-k인지 후 top-k인지, group 제한을 먼저 적용하는지, selection weight를 다시 normalize하는지, bias가 selection에만 쓰이는지 weight에도 쓰이는지에 따라 함수가 달라진다. 한 줄의 `top_k=2`로 이 차이를 표현할 수 없다.

token `i`와 expert `e` 사이 assignment를 `(i,e,w,slot,status)`로 기록한다. `status`는 selected, capacity accepted, dropped, rerouted, shared-expert-only 등을 구분한다. 보존식은 selected assignment 수가 accepted+dropped+rerouted로 분해되는지, combine 뒤 각 token이 몇 경로를 받았는지 검사한다. token count만 비교하면 top-k 복제 하나가 사라져도 숨을 수 있다.

capacity가 expert별 `ceil(capacity_factor·tokens·k/E)`라면 tokens의 domain이 local rank인지 EP group 전체인지가 중요하다. local capacity는 rank partition에 따라 결과가 바뀐다. global capacity는 count exchange가 필요하다. padding token과 masked token을 capacity denominator에 포함하는지도 명시한다. variable-length batch에서 configured token 수를 쓰면 실제 valid token이 적을수록 capacity가 느슨해진다.

tie-break는 재현성의 일부다. 동일 router score에서 작은 expert ID를 우선하는지, stable sort가 token order를 보존하는지, atomic slot allocation이 실행 순서에 의존하는지 확인한다. 모든 score가 같은 fixture는 production에서는 드물지만 tie 정책을 가장 잘 드러낸다. TP/EP world size를 바꿨을 때 exact routing을 요구하는지 통계적 동등성만 요구하는지 계약에 적는다.

auxiliary load-balancing loss는 routing 결과와 같은 denominator를 써야 의미를 해석할 수 있다. token fraction과 mean probability의 곱, sequence별 보조 손실, bias 기반 balancing은 서로 다른 제어 장치다. coefficient 0에서 main gradient와 동일한지, coefficient를 켰을 때 router와 expert 중 어디에 추가 gradient가 가는지 검사한다. detach 위치가 한 칸 다르면 같은 scalar value여도 gradient가 다르다.

z-loss는 router logits의 log-sum-exp 크기를 억제하지만 selection entropy와 동일하지 않다. logit에 상수를 더하면 softmax는 같아도 z-loss는 변한다. 이 불변식 반례로 구현이 probability에 잘못 적용되었는지 찾을 수 있다. FP32 router가 필요한 이유도 큰 logit에서 softmax와 z-loss 안정성을 따로 보기 위함이다.

shared expert는 routed expert의 fallback과 다르다. 보통 모든 token에 적용되는 dense 경로이며 routed combine과 더해진다. shared expert 수, intermediate size, gate 여부, scaling이 parameter와 compute를 바꾼다. routed token이 drop되어도 shared output이 남는지, capacity 통계가 shared 경로를 세는지 확인한다.

DeepSeek 계열의 group-limited routing은 expert를 group으로 묶고 group score로 후보군을 제한한 뒤 expert top-k를 고를 수 있다. group score 정의, correction bias, renormalization을 정확히 읽는다. Qwen3 MoE 고정 revision의 `Qwen3MoeTopKRouter`와 expert module은 비교 기준이지 DeepSeek 정책의 증거가 아니다. 두 구현을 같은 표에 두되 source 좌표를 섞지 않는다.

**expert parallel dispatch의 함수 경계를 CUDA와 collective까지 내린다**

dispatch는 routing 결과를 destination rank 순서로 permutation하고 activation을 보낸다. metadata에는 원 token index, selected slot, expert ID, combine weight가 필요하다. 일부 구현은 weight를 source에 남기고 combine 때 적용하며, 다른 구현은 destination으로 보낸다. forward와 backward가 동일 permutation 계약을 공유해야 한다.

DeepEP 로컬 고정 소스 `sources/training-deepseek-deepep/deep_ep/impls/dispatch.cuh`와 `combine.cuh`는 dispatch/combine kernel 경계를 읽는 좌표다. `deep_ep/utils/comm.py`와 buffer wrapper에서 Python 호출이 어떤 handle과 event를 넘기는지 위로 추적한다. 파일 이름만 인용하지 않고 entry symbol, shape assertion, count tensor, returned handle을 적는다.

intranode NVLink 경로와 internode RDMA/NVSHMEM 경로는 같은 API 아래 다른 staging과 synchronization을 가질 수 있다. low-latency decode용 경로와 throughput 학습용 경로를 혼동하지 않는다. 학습 backward에 필요한 inverse dispatch와 gradient combine이 지원되는지 별도로 확인한다. inference benchmark의 dispatch 성공은 training correctness 증거가 아니다.

count exchange가 잘못되면 흔히 hang으로 보이지만 더 위험한 경우는 byte 수는 맞고 token 순서가 다른 경우다. activation payload와 metadata가 같은 permutation digest를 공유하는지 확인한다. 각 assignment에 64-bit diagnostic ID를 붙인 작은 fixture로 destination expert와 reverse position을 roundtrip한다.

expert GEMM은 token 수가 expert마다 달라 grouped GEMM으로 묶는다. group descriptor에는 input pointer, weight pointer, M/N/K, leading dimension이 들어간다. zero-token expert를 descriptor에서 제거하는 정책과 길이 0의 빈 descriptor를 두는 정책이 있다. weight pointer와 expert ID가 한 칸 밀리면 다른 expert 계산을 하면서 shape는 정상이다. identity-scaled expert fixture가 이를 드러낸다.

backward는 expert input gradient를 원 token으로 reverse combine하고, top-k 여러 경로의 gradient를 더한다. combine weight gradient는 expert output과 upstream의 내적에서 나오며, selected probability 경로를 통해 router로 간다. weight를 forward에서 destination에 cast했다면 backward saved dtype과 recomputation 차이를 확인한다.

overlap은 event dependency graph로 검증한다. permutation 완료 전 send가 시작되면 안 되고 receive chunk가 준비되면 해당 expert GEMM은 시작할 수 있다. reverse collective는 그 expert backward가 끝난 뒤다. default stream의 우연한 동기화에 의존한 코드는 stream 변경에서 race가 난다. CUDA event와 NCCL group 호출의 happens-before를 trace한다.

**실제 모델 세 계보를 동일한 질문으로 해부한다**

### Qwen·DeepSeek·GLM의 공통 계약과 고유 state를 나눈다

첫 질문은 layer schedule이다. 모든 layer가 MoE인지, 처음 또는 주기적으로 dense layer가 있는지, layer index로 factory가 어떤 class를 고르는지 확인한다. config의 expert 수만 읽으면 dense/MoE 혼합 비율을 놓친다. 특정 layer 0, 경계 전후, 마지막 layer를 instantiate하는 정적 fixture로 module class와 parameter inventory를 저장한다.

둘째는 projection layout이다. gate/up/down이 개별 Parameter인지 packed인지, expert dimension이 weight의 첫 축인지, grouped GEMM용 transpose가 저장 형식인지 runtime view인지 확인한다. conversion script가 expert를 stack하는 순서를 읽고 checkpoint tensor의 몇 개 slice를 source expert와 대조한다.

셋째는 router 식이다. input normalization 뒤 어느 dtype으로 matmul하는가, bias가 있는가, softmax와 top-k 순서는 무엇인가, weight normalization과 scaling은 무엇인가. source forward를 식으로 옮기고 손계산 가능한 `E=4,k=2` fixture에 대조한다. “top-2 router”라는 공통 명칭은 이 세부를 보장하지 않는다.

넷째는 expert ownership이다. Transformers 단일-device reference는 expert list를 local module로 가질 수 있다. Megatron류 학습 runtime은 EP/TP/DP mesh에 배치한다. reference model의 수학 함수와 distributed runtime의 dispatch를 분리한 뒤 end-to-end parity로 결합한다. 한 저장소의 함수가 다른 저장소의 통신 정책을 증명하지 않는다.

다섯째는 residual과 normalization이다. MoE output에 shared expert를 더한 뒤 residual을 더하는지, scaling을 어느 지점에 적용하는지, post-attention norm과 pre-MLP norm이 별도인지 확인한다. mHC류 다중 stream이 있으면 expert output이 어느 stream으로 들어가고 mixing이 어느 순서인지 tensor atlas에 추가한다.

여섯째는 loss와 metric이다. aux loss가 model output에 별도 field로 나오고 trainer가 total loss에 더하는지, model 내부에서 이미 더해지는지 확인한다. 두 번 더하는 통합 버그가 가능하다. gradient accumulation에서 aux denominator를 microbatch별 평균 후 평균하는지 global token으로 다시 정규화하는지도 살핀다.

일곱째는 checkpoint portability다. global expert ID와 local stack index, router column, optimizer state가 함께 이동해야 한다. expert 수가 같은데 EP size만 달라지는 reshard와 expert 수 자체가 달라지는 architecture surgery를 구분한다. 후자는 단순 reshard가 아니며 initialization과 품질 재검증이 필요하다.

**성능·품질·안정성을 한 숫자로 섞지 않는다**

MoE는 token당 활성 parameter를 줄여 compute 대비 total capacity를 키울 수 있지만 router, dispatch, imbalance 비용이 생긴다. FLOP만으로 throughput을 예측하지 않는다. expert별 token histogram, remote fraction, all-to-all bytes, grouped GEMM occupancy, exposed communication tail을 함께 측정한다.

품질 비교는 total parameter, active parameter, training token, optimizer, data mixture를 명시한다. dense와 MoE가 같은 wall time이라는 이유로 같은 compute budget인 것은 아니다. auxiliary objective와 dropped token도 학습 함수를 바꾼다. 비교표에 해당 축을 모두 둔다.

안정성은 router logit magnitude, entropy, expert utilization, dropped rate, expert gradient norm, residual RMS를 본다. 평균 utilization이 균등해도 sequence 또는 domain별로 collapse할 수 있다. 언어, 데이터 source, 길이 bucket별 conditional histogram을 offline sample로 분석하되 고카디널리티 production metric은 피한다.

debug 순서는 수학 reference, single-rank routing, distributed roundtrip, expert GEMM, combine, backward, checkpoint다. 처음부터 전체 cluster loss를 보면 원인을 좁힐 수 없다. 각 단계는 이전 단계의 digest를 입력으로 받아 최초 불일치를 표시한다.

최종 `SparseBlockCard`는 model revision, layer schedule, projection packing, router equation, selection/capacity/tie 정책, shared expert, residual/mHC graph, EP owner map, collective sequence, aux loss denominator, checkpoint schema를 가진다. 10장은 이 카드와 실제 checkpoint shape를 결합해 모델 전체 한 layer를 재구성한다.

## 9.7 함수 oracle에서 cluster test까지 검증 사다리를 만든다

검증은 한 번의 end-to-end loss 비교로 끝내지 않는다. dense function, routing ledger, single-process dispatch, EP collective와 optimizer update를 차례로 추가해 실패 범위를 매 단계 절반으로 줄인다.

### eager function에서 distributed step까지 한 축씩 확장한다

첫 단계는 dense gated MLP FP64 oracle이다. 작은 행렬을 명시적으로 곱하고 SiLU exact 식을 쓴다. packed projection을 unpack한 reference와 forward/backward를 비교한다. activation approximation과 dtype 차이는 이 단계를 통과한 뒤 별도 허용오차로 다룬다.

둘째는 router oracle이다. logit, softmax, group filtering, top-k, renormalization, capacity를 Python의 명시적 loop로 계산한다. production sort/top-k와 다른 algorithm을 써 같은 버그 공유를 줄인다. tie와 NaN logit 정책도 명시한다.

셋째는 single-rank expert dispatch다. expert마다 `output=(expert_id+1)·input` 같은 식별 변환을 쓴다. combine 결과에서 잘못된 expert와 weight를 즉시 알 수 있다. 모든 expert가 같은 weight인 fixture는 permutation 오류를 숨긴다.

넷째는 two-rank roundtrip이다. token assignment ID를 payload에 넣고 send count, receive position, reverse position을 비교한다. 한 rank가 empty인 경우, 모든 token이 한 rank로 몰리는 경우, 균등한 경우를 포함한다. collective sequence와 zero-count 처리도 확인한다.

다섯째는 grouped GEMM이다. 서로 다른 M/N/K와 zero-token group을 넣고 개별 GEMM reference와 비교한다. descriptor pointer와 leading dimension을 검사한다. CUDA graph capture가 있다면 dynamic token count의 padding/bucket 정책을 확인한다.

여섯째는 full backward다. input gradient, expert weight gradient, router selected-weight gradient, aux/z loss gradient를 분리한다. main loss만, aux만, z만, 합산을 각각 실행해 선형 합과 detach 위치를 확인한다. dropped assignment의 gradient 정책도 고정한다.

일곱째는 residual block parity다. attention output을 고정하고 dense/MoE branch, shared expert, residual/mHC mixing, norm 순서를 비교한다. layer를 여러 번 반복해 작은 dtype 차이가 누적되는 양상을 본다. 한 layer tolerance만으로 깊은 drift를 승인하지 않는다.

여덟째는 checkpoint roundtrip과 EP reshard다. global expert ID와 optimizer moment를 저장하고 다른 owner map으로 load한다. load 뒤 router column, expert weight, golden token output, one-step delta를 확인한다. loadable과 trajectory-equivalent를 구분한다.

### source·kernel·collective 좌표의 증명 범위를 구분한다

Qwen3 MoE 좌표는 Transformers commit `550d7b3834670483a4df436541272c055dc364bf`의 모델 파일과 해당 config/test에 한정한다. DeepSeek routing 식은 DeepSeek 모델 또는 공식 학습 코드의 별도 revision에서 가져온다. DeepEP는 통신 kernel 증거다. 한 저장소가 다른 저장소의 model semantics까지 증명한다고 쓰지 않는다.

`sources/training-deepseek-deepep/deep_ep/impls/dispatch.cuh`, `combine.cuh`, `deep_ep/utils/comm.py`에서 Python wrapper부터 CUDA entry까지 call chain을 만든다. handle이 어떤 buffer와 event를 소유하는지, count tensor dtype과 device, returned tensor layout을 적는다. header 일부 인용만으로 호출 조건을 추정하지 않는다.

Transformers reference는 보통 모든 expert를 local에서 loop하거나 packed op로 계산할 수 있다. distributed training runtime은 별도 dispatcher를 주입한다. reference의 selected expert/weight 결과를 canonical ledger로 삼고 runtime의 dispatch/combine이 이를 보존하는지 비교한다.

GLM과 mHC는 근거 수준을 특히 엄격히 나눈다. 모델 카드가 구조를 설명하는 단계, config에 key가 있는 단계, forward code와 checkpoint tensor가 있는 단계, training backward/test까지 검증한 단계에 서로 다른 신뢰 표시를 둔다. marketing 명칭을 구현 완료로 승격하지 않는다.

Qwen과 DeepSeek도 family name만으로 release를 일반화하지 않는다. dense와 MoE variant, generation-only implementation, training-capable implementation을 구별한다. model ID, config hash, code commit이 한 묶음이다.

### collapse·drop·hot rank를 원인별 metric으로 번역한다

collapse는 한두 expert가 token 대부분을 받는 현상이다. 전체 utilization entropy뿐 아니라 batch, sequence, domain별 분포를 본다. 평균은 균등하지만 각 언어가 서로 다른 한 expert로 몰리는 specialization일 수도 있다. 이를 무조건 collapse로 판단하지 않고 품질·capacity·failure risk와 연결한다.

dropped rate가 오르면 capacity 부족, imbalance, padding denominator 오류를 구분한다. expert별 offered/accepted/dropped count를 본다. offered가 균등한데 dropped가 특정 rank에 몰리면 placement 또는 local capacity domain 문제다.

router logit scale이 커지면 softmax saturation, z-loss, top-k margin이 변한다. max logit, logsumexp, entropy, top1-top2 margin을 샘플링한다. FP16/BF16 overflow가 있으면 router FP32 경로와 cast 지점을 본다.

expert gradient가 0이면 token을 못 받았는지, shared path만 사용됐는지, dispatch backward가 끊겼는지 구분한다. accepted assignment count와 gradient norm을 함께 본다. token은 받았는데 gradient가 0이면 activation saturation이나 loss path를 추가로 본다.

통신 hang은 마지막 완료 collective sequence ID, rank별 send count vector, CUDA event readiness를 수집한다. 모든 rank가 같은 collective를 호출했는지 먼저 확인한다. 네트워크 문제로 단정하기 전에 conditional skip과 process group mismatch를 배제한다.

성능 저하는 expert imbalance와 network imbalance를 나눈다. max/mean expert tokens, rank별 GEMM time, pairwise bytes, exposed communication tail을 같은 timeline에 둔다. grouped GEMM kernel만 최적화해도 straggler rank가 그대로면 step time은 줄지 않는다.

**학습 recipe의 옵션을 상태 변화표로 바꾼다**

**옵션 하나가 바꾸는 객체를 끝까지 따라간다**

`num_experts`는 router output dimension, expert weight stack, checkpoint key, optimizer state, dispatch owner map을 바꾼다. 기존 checkpoint에 값을 바꾸는 것은 단순 runtime 튜닝이 아니라 architecture surgery다. initialization과 conversion 규칙이 필요하다.

`top_k`는 assignment 수, combine weight, communication byte, capacity pressure, backward path를 바꾼다. throughput과 품질을 동시에 바꿀 수 있다. top-k 변경 실험에서는 capacity와 token budget을 통제하고 selected/accepted 보존식을 다시 계산한다.

`capacity_factor`와 drop policy는 tensor shape와 objective를 바꾼다. 높은 capacity는 padding/workspace를 늘릴 수 있고 낮은 capacity는 token 경로를 버린다. “메모리 옵션”으로만 다루지 않는다. drop token이 residual/shared path를 통해 어떤 신호를 유지하는지 확인한다.

router dtype은 module parameter dtype과 matmul/softmax compute dtype을 각각 바꿀 수 있다. autocast가 router까지 낮추는지, explicit FP32 cast 뒤 output weight를 다시 activation dtype으로 내리는지 본다. state dict dtype만으로 compute dtype을 추정하지 않는다.

aux coefficient는 total loss scalar와 router gradient를 바꾸지만 logging loss에 포함되는 방식도 확인해야 한다. trainer가 model-returned aux loss를 다시 더하지 않는지 통합 test를 둔다. gradient accumulation에서 global denominator를 쓴다.

EP size는 owner map과 collective group, local expert count를 바꾸며 checkpoint reshard가 필요하다. TP size는 expert 내부 weight shard와 GEMM/collective를 바꾼다. 두 축을 동시에 바꾸는 migration은 단계별 reference를 둔다.

grouped GEMM, communication overlap, CUDA graph는 성능 경로를 바꾸지만 dynamic shape와 saved tensor 계약을 건드릴 수 있다. 각각 off reference를 유지한다. fallback 발생을 metric으로 내보내고 production shape에서 실제 선택을 확인한다.

**최종 인계는 한 token의 생애를 재현해야 한다**

probe token 하나의 normalized input, gate/up activation, router logits, selected expert, capacity slot, destination rank, expert output, combine weight, shared output, residual/mHC output을 저장한다. backward에는 각 지점의 gradient를 붙인다.

이 trace는 모든 production token을 저장하라는 뜻이 아니다. 작은 deterministic fixture와 승인된 probe에서 만든다. 운영에서는 digest와 histogram만 수집하고 anomaly 때 재현 bundle로 내려간다.

10장 수신자는 config와 checkpoint만 받아도 이 token trace를 다시 만들 수 있어야 한다. checksum이 처음 갈리는 node가 architecture port의 실패 위치다. layer 전체 output 하나만 비교하는 것보다 훨씬 강하다.

최종 승인 질문은 네 가지다. gate/up packing 순서를 어떻게 증명했는가. assignment 보존식은 empty rank에서도 성립하는가. router auxiliary gradient의 denominator는 무엇인가. EP size 변경 뒤 global expert identity를 무엇이 보존하는가. 답은 고정 source, fixture, manifest를 가리킨다.

답이 “framework가 처리한다”면 미완성이다. framework도 revision과 옵션에 따라 다른 dispatcher와 kernel을 선택한다. 이 장의 목적은 이름을 암기하는 것이 아니라 token 하나가 dense 또는 sparse branch를 거쳐 어떻게 상태와 통신을 만들고 다시 원 위치로 돌아오는지 설명하는 데 있다.

**장애 대응 playbook을 구성한다**

loss가 첫 step부터 NaN이면 dense gate/up preactivation과 router logit을 먼저 본다. gate만 비정상이면 packing, initialization, dtype을 확인한다. router만 비정상이면 input norm, router dtype, softmax/z-loss를 본다. 둘 다 정상이면 expert output과 residual 합을 따라간다.

single rank는 정상이고 EP에서만 틀리면 assignment ledger를 비교한다. source token ID, destination rank, local expert, receive slot, reverse slot이 roundtrip하는지 본다. count가 맞아도 순서가 틀릴 수 있으므로 payload digest와 metadata digest를 결합한다.

특정 expert만 gradient가 없으면 accepted token count를 확인한다. count가 0이면 routing/capacity 현상이고, count가 양수면 expert backward나 combine 경로 문제다. router gradient까지 없으면 selected weight detach와 aux loss 통합을 본다.

step time의 tail이 길면 가장 느린 rank의 offered token, grouped GEMM shape, pairwise communication을 본다. 평균 tokens/s만 보지 않는다. expert placement를 바꾸기 전 router distribution이 data domain과 결합된 현상인지 확인한다.

resume 뒤만 품질이 갈리면 global expert ID, router column, optimizer moment, balance bias, RNG와 counter를 확인한다. weight checksum만 같아도 moment가 다른 expert에 붙을 수 있다. golden routing과 one-step delta가 load 검증의 종료 조건이다.

EP size 변경에서 hang이 나면 새 process group membership과 collective order를 본다. loadable checkpoint가 runtime mesh를 자동으로 올바르게 만든다고 가정하지 않는다. zero-token rank도 모든 required collective에 참여한다.

OOM은 activation, dispatch staging, grouped GEMM workspace, saved routing metadata, backward buffer로 분해한다. capacity를 줄여 해결하면 dropped assignment와 objective가 바뀐다. activation checkpointing, chunking, overlap buffer 수 같은 대안을 기능 변화와 분리해 비교한다.

throughput 최적화 뒤 validation이 갈리면 fused activation approximation, router dtype, capacity/tie determinism, overlap race를 차례로 끈다. 모든 최적화를 한꺼번에 eager로 돌리는 것보다 한 경로씩 reference로 교체해 최초 원인을 찾는다.

모니터링에는 layer별 모든 expert label을 무제한 노출하지 않는다. 선택된 probe layer와 expert bucket, max/mean imbalance, dropped rate, exposed tail을 쓴다. 상세 expert trace는 offline artifact로 보존한다.

코드 리뷰 체크리스트는 shape assertion에 그치지 않는다. logical expert ID가 어디서 local slot으로 바뀌는가, permutation inverse가 어디서 만들어지는가, weight가 어느 dtype에서 적용되는가, backward saved tensor는 무엇인가, failure 전 commit은 어디인가를 묻는다.

최종 실험 보고서는 성공 run뿐 아니라 의도적으로 깨뜨린 fixture와 detector 반응을 포함한다. count 하나를 바꾸면 보존식이 실패하고, expert stack을 바꾸면 golden output이 실패하고, scheduler 없이 aux loss를 두 번 더하면 gradient oracle이 실패해야 한다. detector가 실패를 못 잡으면 gate가 불완전하다.

이 playbook을 통과해야 MoE의 장점을 논할 자격이 생긴다. sparse capacity는 단지 적은 FLOP이 아니라 확률적 할당, 가변 계산, 통신, 복구 가능한 identity의 결합이다. 이 결합을 함수와 상태로 이해하면 새 모델 이름이 나와도 같은 질문으로 해부할 수 있다.

마지막 판정표는 dense와 sparse를 같은 기준으로 비교한다. 입력 residual checksum, norm 위치, projection packing, activation 식, intermediate dtype, router 식, selected/accepted/dropped 보존식, shared expert, residual 또는 mHC mixing, backward gradient, process group, checkpoint owner를 행으로 둔다. 각 행에는 source commit과 symbol, 손계산 oracle, upstream test, local 반례, production metric을 연결한다. Qwen 구현에서 확인한 사실은 Qwen 행에만, DeepSeek routing에서 확인한 사실은 DeepSeek 행에만, DeepEP 통신에서 확인한 사실은 dispatcher 행에만 둔다.

근거의 경계를 지키면 서로 다른 stack을 결합할 때 무엇을 새로 검증해야 하는지가 드러난다. 새 model을 이 표에 넣을 때 family 이름으로 빈칸을 채우지 않는다. config와 checkpoint shape, forward code가 있을 때 함수 행을 채우고, backward test가 있을 때 학습 지원 행을 채우며, multi-rank fixture가 있을 때 분산 행을 채운다. 성능 수치는 동일 token distribution과 topology에서만 비교한다. capacity나 top-k가 달라 objective가 바뀌면 별 실험으로 취급한다. 장애 복구는 global expert ID와 next-step delta까지 맞아야 통과다. 이 엄격한 표가 있어야 MoE 설명이 구조 소개를 넘어 실제 구현을 파고드는 지도 역할을 한다.

미검증 backend와 topology도 표에서 지우지 않는다. CUDA kernel을 정적으로 읽었지만 backward를 실행하지 않았다면 학습 경로는 미검증이다. 두 rank fixture만 통과했다면 multi-node RDMA와 failure recovery를 보장하지 않는다. model card만 존재하면 구조 가설이지 함수 증거가 아니다. 이 경계를 분명히 적으면 독자는 무엇이 사실이고 무엇이 다음 조사 과제인지 곧바로 구별한다. 깊이는 많은 용어가 아니라 근거의 범위와 반례가 정확한 데서 나온다.

각 미검증 행에는 필요한 rank 수, topology, 입력 assignment, expected invariant, 실패 detector를 적는다. 다음 실행자는 모호한 재조사 대신 정확한 실험을 곧바로 수행할 수 있다.

## 9.8 MLP·router·load balance의 수학을 함께 유도한다

dense MLP의 basis expansion에서 시작해 SwiGLU의 곱, router softmax와 load distribution으로 확장한다. 품질 objective와 자원 균형 controller가 같은 숫자로 뭉개지지 않도록 각각의 분모와 gradient를 적는다.

**attention과 다른 축을 섞는다.** self-attention은 token positions 사이에서 정보를 모으고 MLP는 각 token 위치에 같은 함수를 독립적으로 적용해 channels를 변환한다. 입력 `x∈R^C`, intermediate width `I`에서 일반 FFN은 `y=W_down φ(W_up x+b_up)+b_down`이다. batch·token 축은 병렬 examples이고 weight는 공유된다.

parameter와 FLOPs는 주로 `C×I`와 `I×C` 두 projections에서 온다. `I≈4C`라는 고전적 비율은 architecture별로 달라지고 gated MLP는 projections 수와 width를 조절해 budget을 맞춘다. config의 `intermediate_size`와 tensor shapes를 실제 checkpoint에서 확인한다.

ReLU는 음수 영역을 0으로 만들고 GELU는 Gaussian gate 직관을 가진 smooth activation이다. exact erf form과 tanh approximation이 있을 수 있다. approximation 선택은 forward·backward numerical difference와 kernel fusion eligibility를 바꾼다. model config 또는 source branch를 고정한다.

MLP가 “지식을 저장한다”는 직관은 activation neurons와 facts의 일대일 대응을 뜻하지 않는다. 특정 directions와 sparse activations가 behavior와 연결될 수 있으나 distributed representations와 context, attention이 함께 작동한다. 기하 직관은 intervention evidence와 실제 projection source에 붙인다.

**backward를 matrix별로 분리한다.** preactivation `a=W_up x`, hidden `h=φ(a)`, output `y=W_down h`에서 upstream g가 오면 `dW_down=g hᵀ`, `dh=W_downᵀg`, `da=dh⊙φ'(a)`, `dW_up=da xᵀ`, `dx=W_upᵀda`다. batch·token positions의 outer products가 weight gradient에 합쳐진다.

activation checkpointing은 a나 h 저장을 줄이고 backward에서 projections/activation을 재계산할 수 있다. fused MLP는 intermediate를 HBM에 쓰지 않고 tiles 안에서 처리할 수 있다. forward output뿐 아니라 dX와 모든 matrix gradients를 reference와 비교한다.

tensor parallel에서는 up projection output channels를 shard하고 down projection input을 대응시킬 수 있다. row/column parallel collectives와 activation placement를 표로 둔다. bias가 있으면 어느 rank가 소유하고 어떻게 reduce하는지 본다. sequence parallel과 함께 token axes도 이동할 수 있다.

### SwiGLU gate의 Jacobian과 saved tensor를 유도한다

**세 projections의 역할을 구분한다.** 흔한 SwiGLU는 `a=W_gate x`, `b=W_up x`, `h=SiLU(a)⊙b`, `y=W_down h`다. 이름이 `gate_proj`, `up_proj`, `down_proj`인 경우가 많지만 source에서 order와 activation을 확인한다. 일부 fused storage는 gate/up weights를 concat한다.

SiLU(a)=aσ(a)이고 derivative는 `σ(a)+aσ(a)(1-σ(a))`다. upstream `u=∂L/∂h`에서 `∂L/∂a=u⊙b⊙SiLU'(a)`, `∂L/∂b=u⊙SiLU(a)`다. gate branch와 value branch가 서로의 값을 곱하므로 한 branch saturation/zero가 다른 gradient를 막을 수 있다.

두 branches를 하나의 `[2I,C]` matmul로 계산하고 split할 수 있다. checkpoint가 `[gate,up]` 순서인지 `[up,gate]`인지 converter가 알아야 한다. shapes가 같아 swap이 조용히 통과한다. 서로 다른 simple weights와 input으로 branch-order fixture를 만든다.

fused SiLU-mul kernel은 a,b를 읽고 h를 쓴다. backward는 upstream과 a,b 또는 재계산 값을 필요로 한다. in-place variant가 어느 buffer를 덮는지, activation checkpointing과 호환되는지 본다. BF16/FP16 extreme a에서 sigmoid saturation과 approximation tolerance를 검사한다.

intermediate size는 gated projections 세 개의 parameter/FLOP budget과 alignment를 고려해 선택될 수 있다. round-to-multiple은 tensor core tile과 TP divisibility를 맞춘다. config 숫자가 이론 비율과 조금 다른 이유를 “임의”라 하지 않고 hardware·parameter budget과 source construction으로 검증한다.

**MoE를 네 단계의 보존식으로 쪼갠다**

**router score.** token hidden x에서 logits `r=W_rx`를 만들고 softmax 또는 sigmoid 계열 score를 계산한다. router dtype을 FP32로 올릴 수 있다. noise, bias, group routing, expert mask가 어느 시점에 적용되는지 본다.

**selection.** top-k indices와 selected weights를 만든다. weights를 selected experts 사이에서 다시 normalize하는지, raw probabilities를 쓰는지에 따라 output scale과 gradient가 달라진다. tie-breaking과 deterministic policy를 고정한다. masked/unavailable expert가 선택되지 않는지 본다.

**capacity/acceptance.** offered assignments `N_tokens×k` 중 expert capacity와 priority에 따라 accepted 또는 dropped를 정한다. token drop을 residual로 우회하거나 zero expert output으로 두는지 정책을 본다. dropless routing은 drop 대신 variable load와 communication/memory tail을 감수한다.

**dispatch/combine.** accepted assignments를 expert owner로 보내 grouped expert MLP를 수행하고 원 token order로 되돌려 selected weights로 합한다. shared expert가 있으면 별 dense branch를 더할 수 있다. permutation과 inverse, duplicate top-k destinations를 보존한다.

보존식은 `offered=accepted+dropped`, rank별 send counts와 peer receive counts 일치, expert input rows와 accepted assignments 일치, combine contributions와 accepted outputs 일치다. top-k=-1 sentinel과 masked assignments를 count 정의에서 명시한다.

token identity는 `source_rank,source_token,selected_slot,global_expert` tuple로 둔다. dispatch 후 `destination_rank,local_expert,recv_slot`을 연결한다. combine handle이 이 inverse mapping을 보존한다. payload checksum만으로 순서 swap을 잡기 어려워 metadata와 결합한다.

### selected score gradient와 top-k 결정 경계를 구분한다

**selected weight 경로.** top-k indices를 고정하면 output `y=Σ_{e∈S} α_e f_e(x)`이고 `∂L/∂α_e=g·f_e(x)`다. softmax-normalized selected weights를 통해 router logits로 gradient가 흐른다. expert output 차이가 router 학습 신호를 만든다.

top-k set 자체의 변화는 discrete하다. 일반 backward는 선택되지 않은 indices 변화에 대한 derivative를 직접 다루지 않고 선택된 경로에서 gradient를 계산한다. noisy routing, auxiliary loss와 load balance가 exploration/assignment를 보완한다. top-k가 미분 가능하다는 과도한 설명을 피한다.

capacity drop priority가 router score에 의존하면 threshold 경계에서 불연속이 있다. finite difference를 tie/boundary 바로 위에서 쓰면 불안정하다. selected set을 고정한 local gradient test와 routing-decision test를 분리한다.

selected weights를 dispatch 전에 detach하거나 combine kernel backward가 weight gradient를 반환하지 않으면 router는 auxiliary loss만 배울 수 있다. expert output과 total loss는 내려갈 수 있어 놓치기 쉽다. router logit gradient를 main loss와 aux loss로 분해한 fixture를 둔다.

router z-loss는 logits/logsumexp scale을 제어하고 load-balance loss는 assignment/probability distribution을 조절한다. 두 항의 numerator, denominator, coefficient, schedule을 별도 ledger에 둔다. total loss 하나만 보면 어느 신호가 router를 지배하는지 알 수 없다.

### expert load를 평균이 아닌 histogram과 tail로 본다

**importance와 load를 구분한다.** expert e의 probability mass 합과 hard assignment count는 다르다. auxiliary loss는 둘의 곱 또는 다른 통계를 사용할 수 있다. exact formula와 normalization을 model source에서 확인한다. top-k와 group routing이 statistics axes를 바꾼다.

perfect uniform load가 항상 최적 예측은 아니다. domain specialization은 비균등 routing을 만들 수 있다. 문제는 capacity overflow, straggler, undertrained experts와 품질 사이의 trade-off다. balance coefficient를 높이면 router가 예측보다 균등성에 더 많은 weight를 둘 수 있다.

layer별 expert counts의 mean, max/mean, coefficient of variation, entropy, zero-token experts, dropped rate를 본다. batch 평균은 순간 hot expert와 rank skew를 숨긴다. p95/p99와 domain/language/length bucket을 offline artifact에서 연결한다.

expert placement가 global expert IDs를 ranks에 배치하므로 count 균등만으로 network load가 균등하지 않을 수 있다. 같은 rank에 함께 hot experts가 있으면 inter-rank traffic과 grouped GEMM tail이 커진다. placement와 routing 분포를 공동 최적화하되 checkpoint identity를 보존한다.

bias-based auxiliary-loss-free balancing 같은 방법은 expert bias state를 token statistics로 갱신할 수 있다. 이 bias가 model parameter gradient로 학습되는지 별 controller update인지 구분한다. update interval, rate, checkpoint/resume과 distributed aggregation을 기록한다.

### capacity·dropless·token drop의 memory와 objective를 계산한다

**capacity factor의 단위를 계산한다.** tokens N, top-k k, experts E에서 평균 assignments/expert는 `Nk/E`다. capacity factor c를 곱하고 rounding/alignment해 expert capacity를 정할 수 있다. exact formula와 minimum capacity, group/local scope를 source에서 확인한다.

capacity를 높이면 drop은 줄지만 dispatch buffers와 expert compute tail이 늘어난다. 낮추면 throughput/memory는 좋아질 수 있으나 selected contribution이 사라져 model function이 바뀐다. capacity tuning을 순수 성능 옵션으로 취급하지 않는다.

priority는 router weight, token order, random 또는 batch-priority routing을 쓸 수 있다. 같은 assignments라도 survivor가 달라진다. distributed ranks에서 global priority인지 local capacity인지 본다. deterministic rerun과 data order sensitivity를 test한다.

dropless kernel은 accepted assignments 수만큼 variable buffers와 grouped GEMM shapes를 만든다. worst-case hot expert와 zero-token expert를 지원해야 한다. capacity buffer가 없어도 all-to-all receive limits와 workspace sizing이 있다. OOM recovery가 silent drop으로 fallback하지 않는지 본다.

capacity/dropless 비교는 같은 router assignments에서 output parity 가능한 영역과 overloaded 영역을 나눈다. overloaded fixture에서는 expected dropped IDs와 residual behavior를 명시한다. validation effect를 token categories와 experts로 분해한다.

## 9.9 dispatch kernel과 network topology의 비용을 계산한다

expert computation은 동일 weight GEMM의 단순 batch가 아니라 expert별 M이 다른 ragged workload다. descriptor, tile scheduling, all-to-all message histogram과 physical topology를 연결해 compute와 network tail을 분리한다.

**ragged batches를 한 launch에 묶는다.** 각 expert가 받은 token 수 `n_e`가 달라 개별 GEMM을 launch하면 작은 matrices와 launch overhead가 많다. grouped GEMM은 expert별 `X_e W_e`를 descriptors/offsets로 묶어 실행한다. input permutation이 expert-contiguous layout을 만든다.

zero-token expert는 descriptor에 빈 matrix로 들어가거나 건너뛴다. backward에서는 expert별 dW와 dX를 같은 offsets로 계산한다. counts prefix sum과 offsets가 하나만 틀려도 다음 expert payload가 섞인다. canary values와 expert-specific weights로 fixture를 만든다.

weight layout과 quantization/precision은 kernel마다 다르다. `[E,I,C]`, packed gate/up `[E,2I,C]`, transposed storage를 확인한다. converter와 checkpoint shard mapping이 global expert ID와 matrix axes를 보존해야 한다.

load imbalance가 심하면 가장 큰 n_e가 kernel/step tail을 결정하고 작은 experts는 tensor cores를 충분히 채우지 못한다. padding to block, sorting experts, token/expert parallel hybrid를 쓸 수 있다. padding computations가 output/gradient에 들어가지 않게 masks와 counts를 유지한다.

expert tensor parallelism을 함께 쓰면 한 expert matrix가 여러 ranks에 shard되고 expert parallel dispatch 뒤 추가 collectives가 필요하다. EP, TP, DP process groups와 rank coordinate를 분리한다. “expert가 rank에 있다”는 설명만으로 ownership이 충분하지 않다.

### DeepEP handle의 count·buffer·event 수명을 추적한다

**고정 snapshot을 명시한다.** 로컬 DeepEP commit은 `01dc3aaac82068020353dce2c302e38153c0bfaa`다. `deep_ep/buffers/legacy.py:14`의 `Buffer`가 runtime buffer와 process topology를 감싼다. `:293` `get_dispatch_layout`, `:322` `dispatch`, `:408` `combine`이 normal path의 직접 좌표다.

`get_dispatch_layout`은 top-k indices에서 rank/expert별 counts와 membership layout을 준비한다. input shape, num experts, group/world mapping을 읽는다. layout 결과가 dispatch allocation과 peer counts에 연결된다. invalid expert sentinel과 masked assignments 처리도 본다.

`dispatch`는 first call에서 routing metadata로 payload를 보내고 재사용 가능한 handle을 반환할 수 있다. cached handle path는 same routing layout을 전제로 할 수 있으므로 batch가 달라졌을 때 어떤 fields를 갱신하는지 source를 확인한다. handle은 opaque 성능 object가 아니라 inverse combine에 필요한 semantic state다.

`combine`은 expert outputs를 source token order로 되돌리고 top-k weights를 적용하거나 관련 metadata를 사용한다. dispatch permutation의 정확한 inverse인지 reference와 비교한다. same destination expert로 중복 assignment, zero-token rank, masked slots를 포함한다.

`deep_ep/utils/refs.py:10`의 reference `dispatch`와 `:177`의 `combine`은 optimized runtime을 검증할 명료한 출발점이다. `:126`의 `generate_pre_combine_data`도 combine metadata 생성 의미를 읽는 좌표다. reference가 지원하는 dtype/layout 범위와 tests를 확인한다.

### intra-node·inter-node·low-latency branch를 구분한다

**normal intranode path.** `legacy.py:390` 부근은 runtime intranode dispatch 호출을 보여 준다. NVLink/NVSwitch topology와 peer buffers, count exchange를 source/config에서 확인한다. 실제 hardware bandwidth 주장은 실행 evidence가 필요하다.

**internode path.** `legacy.py:458` `internode_dispatch`, `:509` `internode_combine`은 node 사이 RDMA와 node 내부 전달을 결합한다. send RDMA/NVL heads, receive metadata, process groups가 더해진다. 모든 ranks가 같은 topology view를 가져야 한다.

**low-latency path.** `:553` `low_latency_dispatch`, `:624` `low_latency_combine`은 decoding 같은 작은 token traffic을 겨냥한 별 경로다. training의 large token dispatch와 같은 buffer/overlap 가정을 쓰지 않을 수 있다. 1권 serving path와 연결하되 training support를 별도 검증한다.

config helpers `:233` `get_dispatch_config`, `:263` `get_combine_config`, `:176` low-latency RDMA size hint가 num ranks와 hidden/experts에서 어떤 sizes를 정하는지 읽는다. default는 특정 topology/performance heuristic일 수 있다. 함수 존재를 모든 cluster의 최적값으로 과장하지 않는다.

elastic buffer에는 `deep_ep/buffers/elastic.py:855` `dispatch`, `:1046` `combine`이 있다. legacy와 handle schema, rank mapping, failure/membership semantics를 diff한다. 이름이 elastic이라고 arbitrary mid-update recovery가 자동 보장되는 것은 아니다.

### overlap을 stream event와 buffer ownership DAG로 증명한다

**communication과 compute를 겹치려면 lifetime이 길어진다.** dispatch buffer를 network가 읽는 동안 producer가 다음 batch로 덮어쓰면 race가 난다. double buffering 또는 event dependency가 필요하다. buffer index와 UpdateID/LayerID를 연결한다.

CUDA stream에서 dispatch completion event가 expert GEMM stream의 dependency가 되고 combine은 GEMM output ready를 기다린다. host return 시점과 device completion 시점을 구분한다. asynchronous handle을 scope 밖에서 잃거나 너무 일찍 synchronize하면 correctness 또는 overlap이 깨진다.

backward는 forward routing metadata와 permutations를 필요로 한다. activation checkpointing으로 expert forward를 재계산할 때 same routing decision과 RNG를 보존해야 한다. load-balancing bias가 중간에 갱신되면 recompute가 다른 assignments를 만들 수 있다. routing output을 저장하거나 controller update boundary를 분리한다.

overlap correctness fixture는 모든 expert가 서로 다른 recognizable transform을 하게 만든다. stream synchronization을 강제한 reference와 async path output/gradient를 비교한다. many iterations에서 buffer reuse race를 stress한다. average PASS만 보지 않고 rare mismatch와 hang을 기록한다.

performance trace는 routing, layout/count exchange, payload send/recv, grouped GEMM, combine, wait를 분리한다. overlap ratio를 wall-clock span 겹침으로 보되 hidden synchronization을 profiler/source로 찾는다. 빠르지만 payload가 stale한 path는 즉시 실패다.

**residual stream을 identity path와 learned mixing으로 비교한다**

**기본 residual은 `x+F(x)`다.** gradient에는 identity path와 sublayer Jacobian path가 더해진다. 깊은 model에서 identity가 information/gradient highway를 제공한다. pre-norm placement와 residual dtype, scaling을 7장과 연결한다.

residual scaling `αx+βF(x)`이나 learned gates는 깊이 방향의 mixing을 바꾼다. α,β가 scalar, channel, layer, dynamic인지 본다. initialization이 identity에 가까운지, constraints가 있는지 확인한다. checkpoint와 optimizer state에 새 parameters가 포함된다.

mHC 계열처럼 여러 residual streams/channels을 mixing matrix로 결합하는 architecture에는 단일 vector residual보다 넓은 state가 필요하다. mixing의 row/column stochastic 또는 doubly stochastic constraints, parameterization과 projection을 논문·source에서 확인한다. 이름만으로 수학을 추정하지 않는다.

로컬 `sources/training-mhc-megatron-pr2943`는 Megatron 변경 제안 snapshot으로 검토한다. commit/revision과 changed modules, configs, tests를 고정하고 forward mixing, initialization, backward, tensor parallel, checkpoint mapping을 함수별로 추출한다. PR source는 merged production 보장과 다르므로 상태를 명시한다.

mixing matrix가 stochastic constraints를 가지면 residual contributions의 scale과 information flow를 조절할 수 있다. 하지만 constraint 만족이 training stability와 품질을 자동 보장하지 않는다. singular values, row/column sums, gradient, layerwise stream norms를 fixture와 run metrics로 본다.

**mHC를 기존 residual과 비교하는 반증 표**

**state shape.** standard residual `[B,T,C]`와 hyper-connection streams `[B,T,K,C]` 또는 실제 implementation layout을 비교한다. K축이 어디에 있고 어떤 함수가 expand/reduce하는지 source에서 확인한다. shape 추정을 책의 사실로 쓰지 않는다.

**mixing coefficients.** input-dependent인지 learned static인지, normalization/constraint가 어떻게 적용되는지 본다. softmax, Sinkhorn-like projection 또는 다른 parameterization이면 iterations, epsilon, dtype이 function과 cost를 바꾼다. exact source를 인용한다.

**identity recovery.** 특정 initialization 또는 K=1에서 standard residual과 같아지는지 작은 fixture로 확인한다. 같아야 한다는 가정이 논문 equation과 code에서 성립하는지 본다. output뿐 아니라 gradient와 parameter inventory를 비교한다.

**parallelism.** K streams가 hidden shard와 함께 배치되는지, 추가 activation memory와 collectives가 있는지 본다. sequence/context parallel에서 mixing statistic이 local인지 global인지 확인한다. checkpoint reshard mapping을 둔다.

**failure.** row/column sums drift, mixing saturation, one stream collapse, non-finite projection, wrong checkpoint permutation을 주입한다. detector에는 constraint residual, singular spectrum, stream utilization, dense reference output이 있다. 평균 activation RMS만으로 collapse를 놓치지 않는다.

GLM 계열 model card에서 mHC 사용 주장이 있다면 exact model/revision과 공개 code availability를 연결한다. Hugging Face model implementation이 inference forward만 제공하는지 training backward/tests까지 있는지 구분한다. 논문, model card, PR implementation의 evidence scope를 따로 둔다.

**mHC source를 함수와 호출자로 고정한다**

**snapshot 상태를 명시한다.** 로컬 Megatron mHC snapshot은 commit `e7e1a13ab6ed4d1cebe927bd8b43f2416e6590d2`다. `megatron/core/transformer/hyper_connection.py`가 핵심 구현이고, 이 snapshot이 upstream의 어느 PR/branch 상태인지 source registry에서 별도 기록한다. merged release와 동일하다고 가정하지 않는다.

같은 파일 `:25` 부근의 `SinkhornKnopp`와 `:31` `_sinkhorn_normalize`, `:70` forward path는 행·열 normalization iteration을 읽을 좌표다. `:112` 부근의 `HyperConnectionModule`, `:221` mapping 계산, `:381` full forward, `:670` 이후 checkpoint utilities를 call graph로 연결한다. line은 commit과 함께 사용한다.

Sinkhorn normalization은 양수 matrix를 반복적으로 row/column normalize해 doubly stochastic 근처로 보낸다. finite iterations와 epsilon, input parameterization, FP32 cast가 constraint residual을 결정한다. exact projection과 근사 iteration을 구분한다. zero/negative entries 처리와 gradient path를 source에서 확인한다.

row sums와 column sums가 1에 가까운 것은 necessary invariant지만 output function 전체를 증명하지 않는다. stream expansion, input-dependent mappings, residual merge, sublayer application 순서가 맞아야 한다. identity/simple diagonal fixture와 random small K reference를 만든다.

`transformer_block.py:804` 부근은 block 시작에서 hyper connections를 위해 hidden states를 expand하는 branch를, `:838` 이후는 training/recompute 조건을, `:931` 부근은 final layernorm interaction을 검토할 좌표다. module source만 보지 않고 block caller가 shape/lifetime을 어떻게 관리하는지 본다.

tests의 `tests/unit_tests/models/test_gpt_layer_specs.py:25` 이후는 `enable_hyper_connection` config가 attention/MLP layer specs를 어떤 module type으로 바꾸는지 확인한다. 이것은 construction evidence다. numerical forward/backward, distributed, long training stability까지 증명하지 않는다.

**recompute와 checkpoint를 별도 상태로 본다.** K residual streams와 mappings는 activation memory를 늘릴 수 있다. mHC-specific recompute manager가 어떤 intermediates를 저장/재계산하는지 caller와 checkpoint utilities에서 본다. RNG와 mapping parameter state가 같아야 backward recompute가 같은 함수다.

checkpoint schema에는 standard model에 없던 mixing parameters와 K dimension config가 들어간다. mHC-disabled checkpoint에서 enabled model로 warm start할 initialization, enabled checkpoint를 standard model로 변환하는 policy를 명시한다. missing keys를 조용히 무시하지 않는다.

## 9.10 backward·optimizer·checkpoint에서 ExpertID를 보존한다

forward routing이 맞아도 backward의 inverse communication이나 optimizer owner가 틀리면 expert가 다른 token의 gradient를 받는다. global ExpertID와 TokenID를 gradient, moment와 checkpoint shard까지 운반한다.

**Qwen3-MoE 좌표.** Transformers commit `550d7b3834670483a4df436541272c055dc364bf`에서 `src/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193` `Qwen3MoeMLP`, `:270` `Qwen3MoeSparseMoeBlock`이 직접 생성된 implementation 좌표다. modular source에는 `modular_qwen3_moe.py:51` MLP, `:63` sparse block이 있다. generated와 modular source의 관계를 build system과 함께 본다.

MLP에서 gate/up/down projections, activation, bias를 확인한다. sparse block에서 router linear, logits dtype, top-k selection, weight normalization, expert loop/vectorized dispatch, shared expert 여부를 확인한다. training backward가 일반 PyTorch ops로 이어지는지, custom kernel decorator가 있는지 본다.

**DeepSeek V3 좌표.** 같은 snapshot의 `deepseek_v3/modeling_deepseek_v3.py:115` `DeepseekV3MLP`, `:212` `DeepseekV3MoE`가 direct implementation이다. modular source `modular_deepseek_v3.py:44`와 `:129`에서는 parent classes를 통해 재사용되는 관계가 드러난다. copied code와 inheritance를 둘 다 읽는다.

routed experts와 shared experts, group-limited routing, top-k normalization, routing bias/score correction이 config에서 실제 branch로 어떻게 들어가는지 함수 body를 추적한다. model paper의 algorithm이 library inference/training code와 동일 revision인지 구분한다.

**GLM4-MoE 좌표.** `glm4_moe/modeling_glm4_moe.py:263` `Glm4MoeMLP`, `:381` `Glm4MoeMoE`를 본다. lite variant는 `glm4_moe_lite/modeling_glm4_moe_lite.py:360` MLP, `:478` MoE다. modular source가 DeepSeek V3 MLP를 재사용하는 부분과 lite override를 비교한다.

모델 이름보다 열을 채운다. experts 수, top-k, shared experts, expert intermediate width, router score function/dtype, normalization, group routing, capacity/drop, auxiliary loss, output combine, training gradient, kernel path를 기록한다. 빈칸은 family 유사성으로 채우지 않는다.

**Qwen eager MoE를 correctness oracle로 사용하는 범위**

**명료한 expert loop는 느려도 유용하다.** token마다 top-k experts를 고른 뒤 expert별 mask로 selected tokens를 모아 MLP를 실행하고 `index_add`로 weighted outputs를 합치는 Python/PyTorch 경로는 함수 의미를 읽기 쉽다. production grouped GEMM/DeepEP와 비교할 oracle이 된다.

oracle은 token order, expert indices, weights, output contributions를 모두 보존한다. 각 expert를 `f_e(x)=A_e x` 같은 구별 가능한 linear transform으로 바꾸면 expected output을 손으로 계산할 수 있다. duplicate assignment와 expert zero tokens, invalid sentinel을 포함한다.

하지만 eager oracle도 model source와 동일 bugs를 공유할 수 있다. 독립 hand computation과 conservation checks를 둔다. float accumulation order와 scatter `index_add` nondeterminism을 고려한다. exact bitwise보다 tolerance와 metadata equality를 본다.

optimized path가 capacity/drop 또는 weight normalization을 다르게 한다면 동일 함수가 아니다. oracle configuration을 production semantics에 맞추거나 deliberate difference로 표기한다. performance comparison은 같은 accepted assignments와 dtype에서 한다.

backward oracle은 input x, router weights/logits, expert parameters에 대한 gradients를 비교한다. top-k selection boundary에서 indices를 고정한 fixture를 쓴다. aux loss를 제외한 main path와 포함한 total path를 분리한다.

### shared expert·bias·controller state를 checkpoint에 보존한다

**global expert identity가 중심이다.** checkpoint key의 expert index, runtime placement, local grouped-GEMM slot, optimizer moment가 같은 global expert를 가리켜야 한다. EP size가 바뀌면 reshard mapping을 manifest로 만든다. shape가 같은 experts를 순서만 바꾸면 checksum without identity가 놓칠 수 있다.

shared expert는 routed experts와 다른 module path와 parameter group을 가질 수 있다. shared expert output scale/gate, routed sum과 residual merge 순서를 확인한다. checkpoint conversion에서 shared expert를 expert 0처럼 취급하지 않는다.

router correction bias나 load-balance controller state가 parameter/buffer/external trainer state 중 어디에 있는지 본다. optimizer gradient로 갱신되지 않는다면 update rule과 counter를 별도 저장한다. resume 뒤 bias가 0으로 돌아가면 routing과 expert moments의 data distribution이 달라진다.

auxiliary loss coefficients와 schedule은 checkpoint config/trainer state에 있다. model weights만 load해 evaluation하는 것은 가능해도 exact training resume은 아니다. router noise RNG와 expert dropout도 저장한다.

mHC가 결합된 GLM model이라면 residual streams와 mixing parameters, MoE experts, router state의 세 identity axes를 함께 보존한다. model conversion이 mHC를 단일 residual로 collapse할 수 있는지 논문/source에 근거가 없으면 임의 평균하지 않는다.

### combine에서 dispatch까지 backward 통신을 역재생한다

**expert output gradient를 분해한다.** combine output `y_t=Σ_s α_ts z_ts`에서 upstream `g_t`가 각 assignment output으로 `α_ts g_t`만큼 간다. weight gradient는 `g_t·z_ts`다. combine inverse mapping이 source token과 selected slot을 복원한다.

이 gradients를 expert owner ranks로 보내 expert MLP backward를 수행한다. expert input gradient를 다시 source ranks로 보내 같은 token의 top-k contributions를 합친다. forward dispatch/combine과 backward의 통신 방향·metadata lifetime을 표로 둔다.

router selected weights가 softmax에서 왔다면 weight gradients가 selected logits로 이어지고 normalization Jacobian이 적용된다. selection index에는 일반적으로 gradient가 없다. expert input x에는 expert MLP paths와 router linear path, shared expert, residual path gradients가 합쳐진다.

activation checkpointing으로 expert outputs z를 재계산하면 same expert weights snapshot과 routing metadata가 필요하다. optimizer step 전 backward가 완료되므로 weights는 보통 같지만 pipeline scheduling과 recompute state를 확인한다. controller bias를 forward/backward 사이 갱신하지 않는다.

backward communication volume과 buffers는 forward와 다를 수 있다. saved top-k weights, inverse map, expert offsets가 memory를 차지한다. OOM 분석에 forward receive payload만 세지 않는다. gradient accumulation window에서 buffers가 언제 해제되는지 본다.

### checkpoint reshard를 expert 집합·순열·slice로 검증한다

**네 experts, 두 ranks에서 시작한다.** global experts 0,1을 rank 0, 2,3을 rank 1이 소유한다고 하자. 각 weight를 expert ID로 구별 가능한 상수로 채우고 optimizer moments도 다른 상수로 둔다. router columns 역시 global expert 순서에 맞춘다.

world size 4로 reshard해 rank마다 한 expert를 소유하게 한다. load 후 each expert weight/moments/router column mapping을 확인한다. golden tokens가 각 expert를 선택하도록 logits를 구성하고 routed output과 first delta를 old topology dense reference와 비교한다.

expert tensor parallel이 있으면 한 expert 안의 matrix shards도 재분할한다. global expert axis와 matrix row/column axes를 혼동하지 않는다. shared experts와 replicated router parameters를 별도로 처리한다. padded experts가 있으면 selection mask를 보존한다.

world size를 줄일 때 multiple experts가 한 rank로 모이고 grouped GEMM ordering이 바뀔 수 있다. logical output은 유지되어야 하지만 floating accumulation order는 tolerance가 필요하다. dispatch handle과 runtime buffers는 durable checkpoint state가 아니라 new topology에서 재구성한다.

reshard failure injection으로 moments를 expert 1/2 사이 swap하고 router columns만 old order로 둔다. forward와 first update 중 어느 detector가 잡는지 본다. weights만 비교하는 load test가 왜 부족한지 보여 준다.

**MoE metrics를 control plane과 data plane으로 나눈다**

**control-plane metrics.** router logits/entropy, selected experts, offered/accepted/dropped counts, balance loss, correction bias, capacity와 placement를 본다. 이 값들은 routing decision의 원인과 정책을 설명한다.

**data-plane metrics.** send/receive tokens/bytes, dispatch/combine latency, peer wait, grouped GEMM shapes/latency, buffer high-water mark, overlap, retry/error를 본다. network와 compute 실행을 설명한다.

두 planes를 LayerID·UpdateID로 연결한다. router가 균등해도 placement 때문에 peer traffic이 skew될 수 있고, traffic이 균등해도 expert matrices/sequence lengths로 compute가 skew될 수 있다. 평균만으로 원인을 결론내리지 않는다.

Prometheus labels에는 모든 layer/expert/rank pair를 무제한 넣지 않는다. selected layers, quantile/buckets, max owner를 낮은 cardinality로 노출하고 상세 matrix는 trace artifact에 둔다. hot expert ID는 exemplar로 연결한다.

alerts에는 conservation mismatch, dropped-rate 지속, zero-token expert 장기화, rank count disagreement, dispatch timeout, combine inverse mismatch, mHC constraint residual을 둔다. throughput 저하는 data distribution transition annotation과 함께 본다.

**성능 최적화의 순서를 function parity 뒤에 둔다**

**첫째, dense/eager oracle을 봉인한다.** same assignments, weights, expert functions에서 output과 backward를 재현한다. router decision과 dispatch를 분리한다. accepted token ledger가 정확해야 한다.

**둘째, local permutation/grouped GEMM을 넣는다.** network 없이 expert-contiguous layout과 inverse combine을 검증한다. counts skew와 zero experts를 시험한다. eager experts와 output/gradients를 비교한다.

**셋째, intra-node dispatch를 넣는다.** global IDs와 rank mapping, NVLink path, stream events를 검증한다. topology size를 작은 fixture부터 늘린다. single-node PASS를 inter-node 증거로 쓰지 않는다.

**넷째, inter-node와 overlap을 넣는다.** RDMA/NIC path, peer heads, buffer reuse, straggler를 본다. synchronization-forced reference와 비교한다. failure/hang injection과 recovery boundary를 둔다.

**다섯째, precision/quantization을 바꾼다.** router, payload, expert GEMM, combine accumulator dtype을 한 축씩 바꾼다. scaling metadata와 communication representation을 확인한다. numerical drift와 bandwidth gain을 함께 측정한다.

여러 최적화를 한꺼번에 켜면 first difference를 찾기 어렵다. child configurations와 expected impact map을 만든다. 최종 production path가 빠르더라도 모든 중간 oracle을 보존해 future upgrade에 재사용한다.

### imbalance·hang·gradient drift를 서로 다른 실험으로 가른다

**loss spike와 expert 0 집중.** input norm과 router logits를 먼저 본다. config/group mask, correction bias restore, top-k tie, data domain transition을 확인한다. capacity drop이 시작된 UpdateID와 연결한다. balance coefficient만 올려 증상을 숨기지 않는다.

**tokens 보존식은 맞지만 output이 다르다.** permutation order, expert weight mapping, combine selected-slot weights를 본다. count는 payload identity를 보증하지 않는다. expert-specific canary transform과 tuple ledger로 first swap을 찾는다.

**single node 정상, multi-node hang.** rank별 last DeepEP call, dispatch/combine phase, counts, process group, topology mapping을 비교한다. 한 rank의 upstream OOM/exception과 network transport를 분리한다. zero-token rank가 conditional call을 건너뛰지 않았는지 본다.

**throughput이 간헐적으로 절반.** hot expert/peer skew, largest grouped GEMM, buffer contention, GC/checkpoint overlap, network tail을 trace한다. 평균 router entropy가 정상이어도 batch-local concentration이 있을 수 있다.

**resume 뒤 experts specialization이 바뀐다.** weight뿐 아니라 optimizer moments, router columns, correction bias, RNG, global expert placement를 비교한다. golden routing과 first delta를 old/new topology에서 확인한다.

**mHC에서만 NaN.** Sinkhorn input/log domain, FP32 cast, iteration, constraint residual, stream norm을 본다. MoE router sinkhorn과 mHC Sinkhorn을 같은 state로 혼동하지 않는다. source caller를 분리한다.

**10장으로 넘길 모델 해부 dossier**

model-specific dossier는 dense MLP equation과 projections, activation approximation, intermediate width, bias, tensor-parallel mapping을 가진다. MoE이면 router, top-k, expert/shared paths, capacity/drop, aux losses, global expert ownership을 추가한다.

source coordinates는 Transformers commit `550d7b...`의 Qwen3-MoE, DeepSeek V3, GLM4-MoE classes와 DeepEP commit `01dc3a...` dispatch stack, mHC Megatron snapshot `e7e1a1...` hyper-connection stack을 분리해 둔다. 하나의 project evidence로 다른 project function을 채우지 않는다.

tensor atlas에는 residual/norm input, router logits, selected indices/weights, accepted ledger, dispatched payload, expert input/output, combined output, residual/mHC output, backward projections가 있다. large runtime 없이 small fixtures로 확인한 범위와 미실행 topology를 구분한다.

10장은 특정 model의 config와 checkpoint를 이 dossier 표에 넣는다. model card 주장, library source, communication backend, actual recipe가 서로 맞는지 검토한다. 빈칸은 “일반적으로 그렇다”로 채우지 않고 next investigation으로 남긴다.

인계 전에는 dense token 한 개와 routed token 한 개를 input에서 loss/gradient까지 왕복한다. routed token은 source rank에서 expert owner와 다시 source로 돌아오는 tuple을 보존한다. mHC가 있으면 residual stream mapping과 constraint도 포함한다.

## 9.11 balancing objective와 controller state를 분리한다

load balance는 model gradient로 작동하는 auxiliary loss와 gradient 밖 correction controller로 구현될 수 있다. Sinkhorn, z-loss와 auxiliary-loss-free bias가 어느 통계로 언제 갱신되는지 분리한다.

**2×2 양수 행렬에서 시작한다.** `M=[[a,b],[c,d]]`의 rows를 합으로 나누고 columns를 다시 합으로 나누는 과정을 반복한다. 모든 entries가 양수이고 적절한 조건이면 row/column sums가 1인 doubly stochastic matrix에 가까워진다. finite iterations에서는 두 residuals가 정확히 0이 아닐 수 있다.

log-domain 또는 exp parameterization을 쓰는지 source에서 확인한다. 큰 logits를 직접 exp하면 overflow할 수 있다. max subtraction과 FP32 computation, epsilon clamp가 필요할 수 있다. `hyper_connection.py:109` 부근에 남은 FP32 정밀도 관련 미해결 주석도 현재 구현의 정밀도 상태를 과장하지 말아야 할 근거다.

iteration 수를 늘리면 constraint residual은 줄 수 있지만 compute와 backward graph가 늘어난다. tolerance-based early stop은 data-dependent control flow와 compile/distributed behavior를 만들 수 있다. fixed iteration인지 convergence test인지 exact function을 본다. config `mhc_sinkhorn_iterations`가 어디서 validation되고 checkpoint에 저장되는지 추적한다.

입력 matrix가 거의 permutation이면 stream routing이 한 경로에 집중될 수 있다. uniform이면 모든 streams를 섞는다. singular values와 entropy, row/column residual을 함께 본다. doubly stochastic이라는 사실이 좋은 diversity를 보장하지 않는다.

backward directional derivative는 entries가 positive이고 boundary에서 떨어진 fixture로 한다. normalization iteration을 detach하는 branch가 있는지 확인한다. legacy transformer의 router sinkhorn처럼 `route.detach()`가 쓰이는 code와 mHC gradient path를 혼동하지 않는다.

**residual depth를 동역학 관점으로 읽는다**

**표준 residual은 작은 증분의 합이다.** `x_{l+1}=x_l+F_l(x_l)`는 depth를 이산 시간으로 보면 identity에 update를 더하는 형태다. 이 직관은 scale 안정성을 이해하는 데 도움이 되지만 실제 blocks는 norm, attention/MLP, learned scales를 포함한다.

Jacobian은 `I+J_F`이고 gradient가 product를 거쳐 흐른다. identity term은 gradient path를 제공하지만 `J_F`의 spectrum과 nonnormality, norm placement가 깊은 stability를 결정한다. “residual이면 vanishing gradient가 해결된다”는 절대 문장을 피한다.

multiple streams와 learned mixing은 state dimension을 확장하고 depth마다 basis를 재결합할 수 있다. mHC constraint는 mixing scale 폭주를 막으려는 기하적 목적과 연결된다. 실제 효과는 equations, initialization, finite precision과 experiments로 확인한다.

layer별 residual stream norms, sublayer update/residual ratio, mixing singular values, gradient norms를 본다. 평균 stream만 보면 하나의 stream collapse나 cancellation을 놓친다. checkpoint resume 뒤 stream permutation이 바뀌지 않는지 canary를 둔다.

### data mixture 변화와 expert load 변화의 상관을 추적한다

**router specialization은 데이터 분포를 반영한다.** language, domain, format, task length에 따라 expert selection이 달라질 수 있다. curriculum stage나 mixture weight가 바뀌면 offered counts와 hot experts가 바뀐다. routing imbalance를 인프라 문제만으로 보지 않는다.

그러나 expert label을 의미 category로 즉시 해석하지 않는다. routing correlation과 causal specialization은 다르다. intervention으로 expert를 mask/swap하고 output 변화를 보거나 representation analysis를 한다. model behavior와 training data lineage를 연결하되 개인정보를 expert metrics label에 넣지 않는다.

curriculum 초기에 일부 experts가 거의 사용되지 않으면 optimizer moments와 capacity가 뒤늦은 domain stage에 준비되지 않을 수 있다. balance controller와 expert warmup, random routing을 사용할 수 있으나 actual recipe evidence를 요구한다. zero-token duration과 first activation 시점의 gradients를 본다.

data duplication이 특정 patterns를 과다 노출하면 router와 experts가 그 방향으로 편향될 수 있다. 4장의 dedup cluster와 mixture, 6장의 sample ordering을 UpdateID routing metrics와 연결한다. expert 문제를 LR 하나로만 고치지 않는다.

### auxiliary loss의 reduction과 global weight를 계산한다

**단위가 다른 losses를 분리한다.** language-model CE는 valid tokens, router balance는 token-expert statistics, z-loss는 router rows, mHC constraint는 layer/stream matrices를 단위로 할 수 있다. 각각 numerator와 denominator를 기록한 뒤 coefficients를 적용한다.

tensor parallel이나 expert parallel에서 statistics를 어느 group에서 aggregate하는지 본다. local balance loss를 각 rank에서 계산해 DDP 평균하면 global token distribution의 loss와 다를 수 있다. global probabilities/counts를 어떻게 합치는지 model/trainer source를 읽는다.

gradient accumulation에서 aux loss denominator도 전체 window와 맞춰야 한다. microbatch mean 평균은 variable tokens에서 weight를 바꾼다. main CE만 `num_items_in_batch`를 교정하고 aux는 local mean이면 relative coefficient가 batch마다 흔들린다.

loss logging은 `main`, `router_balance`, `router_z`, `mhc_constraint`, total을 각각 numerator/denominator/coefficient와 함께 둔다. coefficient schedule과 first/last UpdateID를 기록한다. total scalar가 같아도 components가 상쇄될 수 있다.

failure fixture는 aux를 두 번 더하거나 coefficient를 적용하지 않고 logging만 하는 경로, detached statistic을 넣는다. main loss gradient와 router/mHC parameter gradients를 독립 oracle과 비교한다.

**dispatch payload precision과 scale metadata**

**통신 dtype은 expert compute dtype과 같지 않을 수 있다.** hidden payload를 BF16, FP8 또는 quantized form으로 보내고 scales를 별도 전송할 수 있다. DeepEP dispatch signatures가 tensor 또는 `(tensor,scales)` tuple을 받는 경로를 source에서 확인한다. scale layout과 token permutation이 함께 이동해야 한다.

per-token, per-channel, block scale은 metadata shape와 dequantization 오차를 바꾼다. scale이 source order에 있고 payload가 expert order로 permute되면 동일 permutation을 적용해야 한다. recognizable scales fixture로 swap을 잡는다.

router top-k weights의 dtype과 combine accumulator를 확인한다. expert output low precision과 FP32 accumulation 조합이 있을 수 있다. top-k가 커지거나 outputs가 상쇄될 때 error가 증가할 수 있다. eager FP64→BF16→quantized dispatch 순으로 비교한다.

communication compression은 bandwidth를 줄이지만 overflow/underflow와 scale buffer memory가 있다. performance report에는 payload bytes와 scale bytes, quantization/dequantization latency를 포함한다. accuracy report는 expert output/combined output/gradients를 본다.

**network topology와 expert placement를 함께 설계한다**

**node 내부와 node 사이 비용이 다르다.** NVLink/NVSwitch peer와 RDMA/InfiniBand path의 bandwidth·latency가 다르므로 experts를 ranks에 배치할 때 frequent traffic을 고려한다. 실제 topology와 NIC binding, process rank mapping을 inventory로 둔다.

DeepEP normal inter-node path는 RDMA와 node-local forwarding을 조합할 수 있다. num RDMA ranks와 global ranks mapping을 source/runtime config에서 확인한다. 모든 ranks가 같은 logical expert placement와 topology revision을 가져야 한다.

expert replication은 hot expert compute/traffic을 분산하지만 weight와 optimizer state synchronization이 필요하다. dynamic placement는 checkpoint global identity와 routing mapping을 복잡하게 한다. 공개 implementation evidence 없이 지원한다고 추정하지 않는다.

placement optimization은 historical routing matrix를 쓸 수 있으나 curriculum 변화에서 stale해질 수 있다. migration cost와 stabilization window를 고려한다. quality function은 같은 weights/routing이면 유지되어야 하고 performance만 바뀌어야 한다. migration first-difference fixture를 둔다.

network 장애 시 in-flight dispatch의 정확히 한 번 처리를 보장하는지 묻는다. 일반 synchronous training은 partial update를 버리고 checkpoint boundary로 돌아갈 수 있다. payload retry를 application layer에서 중복 combine하지 않게 protocol을 본다.

**MoE 메모리 원장을 상태별로 작성한다**

parameters에는 routed expert matrices, shared experts, router, mHC mappings가 있다. optimizer states는 expert sharding에 따라 local이지만 replicated router와 shared parameters는 다를 수 있다. checkpoint staging을 별도 센다.

activations에는 residual input, router logits/probabilities, top-k indices/weights, dispatch permutation/handle, received expert inputs, gated intermediate, expert outputs, combine metadata가 있다. backward에 무엇을 저장하고 재계산하는지 source로 확인한다.

communication buffers는 max tokens, hidden, ranks/experts와 overlap buffer 수에서 정해질 수 있다. capacity/dropless, quantization scales, RDMA/NVL heads가 추가된다. actual high-water와 config hint를 비교한다. over-allocation과 overflow를 모두 본다.

grouped GEMM workspace와 compilation cache, allocator fragmentation도 peak에 포함한다. expert load skew가 특정 rank receive buffer와 activations를 키운다. 평균 memory가 아니라 rank max와 layer tail을 본다.

OOM mitigation마다 함수 영향을 표시한다. activation recompute는 compute/RNG, capacity reduction은 drop/objective, expert offload는 transfer, chunked dispatch는 ordering/overlap, lower precision은 numerical error를 바꾼다. 단순히 batch size를 줄이는 것과 구분한다.

## 9.12 GoldenRoutingRun으로 failure와 regression을 재현한다

작은 token·expert 행렬에서 routing score, accepted assignment, split vector, expert output과 combined gradient를 모두 손계산한다. collapse·drop·count mismatch와 wrong inverse permutation을 하나씩 주입해 detector가 최초 경계에서 울리는지 본다.

**입력을 설계한다.** tokens 4개, experts 3개, top-k 2, two logical ranks를 둔다. router logits를 tie 없이 명확한 값으로 설정한다. selected weights normalization을 손계산한다. one expert는 zero accepted, one token은 capacity 때문에 하나의 assignment가 drop되도록 별 fixture를 둔다.

각 expert는 `f_e(x)=(e+1)x+c_e` 같은 구별 가능한 transform을 쓴다. dispatch permutation이 틀리면 output이 즉시 달라진다. combined output과 selected-weight gradients를 FP64로 계산한다. shared expert가 있으면 별 상수 transform으로 분리한다.

tuple ledger에서 offered, accepted, dropped, send/recv, expert rows, returned contributions를 모두 맞춘다. source token order를 바꾸고 partition을 바꿔도 logical result가 같다. tie-breaking fixture는 별도로 deterministic policy를 확인한다.

DeepEP reference dispatch/combine과 optimized path가 available한 작은 환경에서 비교할 수 있도록 input schema를 둔다. 실행하지 않았다면 expected invariant와 `NOT_RUN`을 남긴다. 대규모 model runtime은 필요하지 않다.

backward에서는 router main gradient, expert parameters, input residual을 비교한다. aux loss를 켜고 끈 child fixture로 gradient decomposition을 확인한다. accumulation과 uneven rank counts를 추가해 global denominator를 본다.

checkpoint fixture는 experts와 router/mHC states를 ID-specific values로 저장하고 topology를 바꿔 load한다. first routing과 first parameter delta가 reference와 맞아야 한다. buffer handles는 새 runtime에서 재생성한다.

### top-k·capacity·normalization option을 state diff로 기록한다

**`num_experts`.** router output columns, expert parameter axis, checkpoint/optimizer state, average load와 capacity, placement를 바꾼다. 기존 checkpoint resize는 임의로 할 수 없고 migration을 정의한다.

**`top_k`.** offered assignments, selected weight normalization, compute/communication, combine sum과 router gradient를 바꾼다. throughput 옵션이 아니라 model function이다. k가 experts 수와 capacity에 맞는지 validation한다.

**`capacity_factor`·dropless.** accepted/dropped set과 buffers, straggler를 바꾼다. overload fixture에서 first difference가 acceptance bitmap이어야 한다. normal non-overload batch는 동일할 수 있다.

**router dtype·normalization.** selected indices와 weights, balance/z-loss 수치를 바꿀 수 있다. unselected logits까지 포함한 gradient를 본다. FP32 router option이 expert GEMM dtype을 자동으로 바꾸지 않는다.

**aux-loss coefficient.** forward expert output은 같고 total loss와 router gradient 이후가 달라져야 한다. controller bias 방식은 routing state 자체를 바꿀 수 있다. source owner를 분리한다.

**EP/TP size.** logical function을 유지하면서 placement, shards, collectives, buffer와 rounding order를 바꿔야 한다. global expert mapping과 dense reference가 기준이다. capacity를 local scope로 계산하면 function도 달라질 수 있다.

**DeepEP dispatch config.** channel/SM/buffer/low-latency path와 overlap을 바꿀 수 있다. accepted assignments와 logical output은 유지되어야 한다. 실제 selected config helper와 runtime path를 확인한다.

**mHC enable·stream count·Sinkhorn iterations.** residual state shape, mixing modules/parameters, compute/memory, checkpoint schema를 바꾼다. simple residual의 성능 flag가 아니다. construction tests와 numerical fixtures를 모두 요구한다.

### 독립 구현이 ledger에서 같은 결과를 재구성하게 한다

검토자는 dense MLP에서 gate/up branch를 swap하고 branch-order detector를 본다. MoE에서 expert indices 두 개를 swap하고 tuple ledger를 본다. dispatch counts만 유지한 채 payload order를 바꾸어 identity detector가 작동하는지 본다.

router weight를 detach하고 main-loss router gradient가 사라지는지, aux gradient만 남는지 확인한다. capacity를 낮춰 expected dropped IDs와 loss 변화가 맞는지 본다. zero-token rank와 expert가 collective/descriptor를 건너뛰지 않는지 본다.

mHC에서는 row/column normalization iteration을 하나 줄이고 constraint residual과 output first difference를 본다. checkpoint stream permutation과 missing mixing state를 load gate가 거부하는지 확인한다. PR source의 미검증 production 범위를 명시한다.

source evidence는 exact commits와 symbols, tests를 다시 resolve한다. line drift와 body fingerprint를 확인한다. model card 문장, paper equation, Transformers implementation, DeepEP communication, Megatron PR을 서로 다른 evidence classes로 유지한다.

감사 결과에는 PASS뿐 아니라 실패 injection의 expected/observed detector, 실행 환경, topology와 미실행 backend가 있다. one/two-rank toy가 multi-node recovery를 증명하지 않는다. 그러나 필요한 input, invariant, trace를 구체화해 다음 실행을 즉시 가능하게 한다.

최종 질문은 “MoE가 dense보다 좋은가”가 아니다. token이 어떤 router state로 experts를 선택하고, 어떤 identity와 buffer로 이동하며, 어떤 expert 함수와 residual mixing을 거쳐 돌아오고, backward와 checkpoint에서 그 경로가 어떻게 보존되는가다. 이 질문에 답해야 성능·품질 trade-off를 논할 근거가 생긴다.

**dense와 sparse의 공정한 비교를 설계한다**

**무엇을 고정하는지 먼저 정한다.** total parameters, active parameters/token, training FLOPs, wall-clock, tokens, memory, communication 중 무엇을 맞출지에 따라 비교가 달라진다. MoE는 total capacity를 늘리면서 token당 일부 experts만 활성화할 수 있지만 router와 communication 비용이 추가된다.

같은 nominal FLOPs라도 grouped GEMM utilization과 network wait, dense GEMM efficiency가 다르다. theoretical active FLOPs와 measured model FLOPs utilization, end-to-end tokens/sec를 함께 본다. padding/capacity drop으로 실제 executed rows도 계산한다.

quality 비교는 같은 tokenizer, corpus mixture, valid tokens, optimizer budget, evaluation harness를 가능한 한 맞춘다. MoE-specific aux losses와 expert initialization은 method의 일부로 기록한다. dense보다 많은 total parameters가 checkpoint/storage/serving에 주는 비용도 포함한다.

failure rate와 operational complexity를 결과에 넣는다. load skew, dispatch hang, reshard, expert state corruption이 recovery point objective와 engineering cost를 바꾼다. peak throughput만으로 system value를 결론내리지 않는다.

small-scale ablation이 full-scale routing dynamics를 완전히 대표하지 않을 수 있다. evidence scope를 model size와 experts, topology, token budget과 함께 표시한다. 논문 result와 local implementation의 차이를 구분한다.

**router와 expert의 학습률을 진단한다**

**gradient scale을 분리한다.** router는 모든 tokens의 selection/weights에서 gradient를 받고 각 expert는 routed subset에서 gradient를 받는다. expert별 effective token count가 달라 optimizer moment와 update noise가 다르다. shared expert는 훨씬 많은 tokens를 볼 수 있다.

router LR, expert LR, shared/dense LR를 parameter groups로 다르게 둘 수 있다. 실제 recipe source가 없으면 권장값을 지어내지 않는다. group mapping과 applied LR, update/weight ratio를 layer/expert bucket으로 본다.

rare expert는 gradient가 드물고 Adam moments가 오래 stale할 수 있다. zero-token step에서 optimizer가 `grad=None`인지 zero인지에 따라 weight decay와 step state가 달라진다. source/fixture로 확인한다. global step bias correction이 expert activation count와 다르다는 점도 기록한다.

gradient clipping을 global model norm으로 하면 hot dense/shared components가 coefficient를 결정해 rare experts도 같이 축소된다. expert/group별 clipping은 다른 알고리즘이다. raw norms와 coefficient owner를 본다. router-only spikes를 전체 clipping으로 숨기지 않는다.

balance controller update와 gradient optimizer update가 둘 다 router decisions에 영향을 주면 두 time scales를 기록한다. checkpoint resume에서 하나만 복원하면 routing trajectory가 달라진다.

**fine-tuning에서 MoE의 특별한 선택**

**모든 experts를 학습할지 결정한다.** full MoE fine-tuning은 expert weights와 router, shared experts를 업데이트한다. adapter-only는 각 expert에 LoRA를 붙일지 shared layers만 붙일지, router를 고정할지 선택한다. trainable parameter 수뿐 아니라 token별 active adapters와 optimizer state를 계산한다.

router를 고정하면 pretrained specialization을 유지하지만 new domain routing을 적응시키지 못한다. router만 학습하면 existing experts의 input distribution이 바뀌고 output capacity가 제한될 수 있다. experts 일부만 학습하면 load와 forgetting이 비대칭적이다. 실험 변수로 분리한다.

expert별 LoRA modules는 global expert identity와 checkpoint mapping을 따라야 한다. EP reshard에서 base expert와 adapter/moments를 함께 이동한다. adapter merge가 quantized/fused expert format과 호환되는지 본다. shared expert adapter와 routed adapters를 구분한다.

작은 fine-tuning dataset은 일부 experts만 자주 선택해 나머지 trainable parameters가 거의 업데이트되지 않을 수 있다. domain별 routing coverage와 expert effective samples를 측정한다. balance를 강제로 높이는 것이 task quality와 충돌할 수 있어 validation으로 결정한다.

preference/RL fine-tuning에서는 generated sequence와 policy changes로 routing distribution이 빠르게 변할 수 있다. old/reference policy와 current policy가 experts를 다르게 사용하면 KL과 reward, system throughput을 함께 본다. 19·20장의 rollout/training 분리와 연결한다.

**보안과 안정성 관점의 expert routing**

특정 trigger와 expert 선택의 correlation은 model behavior 분석 단서가 될 수 있지만 expert 하나가 유해 기능을 소유한다고 단정하지 않는다. red-team prompts의 routing distribution과 benign controls를 비교한다. privacy-sensitive labels를 metrics에 넣지 않는다.

adversarial input이 hot expert를 만들어 service/training straggler를 유발할 가능성을 capacity와 batch controls에서 검토한다. training corpus의 repeated pattern과 poisoning이 router를 왜곡하는지도 본다. 4장 provenance와 25장 red-team taxonomy를 연결한다.

expert dropout이나 routing perturbation을 robustness 목적으로 쓰면 objective와 RNG가 바뀐다. exact implementation과 tests를 요구한다. 장애로 expert가 unavailable할 때 다른 expert로 자동 reroute하는 것은 정상 model function과 다르며 별 degraded mode다.

checkpoint integrity에서는 expert shard substitution과 router-column mismatch를 checksum/identity manifest로 잡는다. 단일 expert corruption이 전체 weight average에서 작게 보일 수 있다. global expert별 selected digest를 sampling한다.

**독자가 실제 repository를 파는 순서**

먼저 config에서 hidden/intermediate sizes, expert 수, top-k, shared experts, router options, aux coefficients, residual/mHC fields를 찾는다. 다음 module construction에서 어떤 class와 weights가 만들어지는지 inventory를 만든다.

forward에서 dense/experts branch, router logits, selection, dispatch, expert compute, combine, residual merge 순으로 symbols를 적는다. generated modular code와 runtime kernel replacement를 구분한다. training loss가 aux components를 수집하는 caller까지 올라간다.

backward는 일반 autograd인지 custom Function인지 확인하고 saved metadata를 찾는다. source tests가 forward, gradients, distributed를 어디까지 다루는지 표로 둔다. communication backend가 별 repository라면 exact commit과 API adapter를 연결한다.

checkpoint에는 parameter keys, expert mapping, optimizer/controller state, config schema를 본다. reshard/converter scripts와 tests를 찾는다. model card의 architecture 주장을 source에서 확인할 수 없는 경우 가설로 표시한다.

마지막으로 small fixture를 설계한다. expert-specific transforms, known router logits, skew/zero/drop, two ranks, first delta를 포함한다. 실행하지 못한 hardware path도 expected invariants와 trace points를 적는다. 이 순서가 repository 규모가 커도 길을 잃지 않게 한다.

**최종 봉인과 다음 장의 출발점**

봉인된 dossier에는 dense/SwiGLU equations와 backward, model-specific router/experts, assignment conservation, DeepEP handle/dispatch/combine, mHC mappings/constraints, optimizer/checkpoint ownership이 있다. 각 항은 source revision과 fixture, 미검증 범위를 가진다.

Qwen, DeepSeek, GLM의 행은 실제 class 좌표로 채워지고 공통점과 차이가 분리된다. DeepEP는 communication evidence, Megatron snapshot은 mHC implementation evidence로만 사용한다. 논문과 model card의 주장은 별 sources로 연결한다.

독립 검토자는 token 하나의 tuple을 source hidden에서 selected experts, destination buffers, expert output, combine, residual과 loss gradient까지 왕복한다. expert ID와 stream ID, optimizer state가 checkpoint round trip에서도 보존되는지 본다.

성능 gate는 같은 logical assignments에서 correctness를 통과한 configurations만 비교한다. capacity와 top-k, residual mixing이 달라지면 model function 변경으로 새 RunRevision을 만든다. fast path의 미실행 topology를 숨기지 않는다.

이 봉인이 닫히면 10장은 특정 model 하나를 골라 attention과 MLP/MoE, residual/mHC가 layer 전체에서 결합되는 방식을 해부할 수 있다. 9장의 표와 fixtures가 model family 이름 대신 함수와 state를 비교하는 공통 언어가 된다.

**한 번의 변경을 끝까지 추적하는 예제**

top-k를 2에서 4로 바꾸는 변경을 생각하자. config 숫자 하나가 먼저 router selection output shape와 offered assignments를 바꾼다. selected weight normalization의 분모와 각 contribution scale도 달라진다. expert compute와 dispatch bytes, buffer high-water, combine additions가 늘어난다.

capacity formula가 `Nk/E`를 기준으로 하면 nominal capacity도 바뀔 수 있다. capacity를 고정하면 dropped rate가 늘 수 있다. router auxiliary statistics와 gradient도 바뀐다. 따라서 top-k 변경은 throughput tuning이 아니라 objective와 architecture state 변경이다.

first-difference map은 input residual과 router logits까지 같고 top-k indices/weights에서 갈라져야 한다. 그보다 앞서 다르면 batch, checkpoint, RNG가 통제되지 않았다. selected set 뒤 expert inputs와 outputs, combine, loss, gradients, optimizer delta가 예상대로 바뀌는지 본다.

checkpoint는 기존 expert weights를 그대로 load할 수 있어도 config와 optimizer trajectory가 새 RunRevision이다. DeepEP buffer hints와 grouped GEMM descriptors를 새 maximum assignment에 맞춘다. serving engine도 같은 top-k와 normalization을 지원하는지 별도 검증한다.

반대로 DeepEP dispatch config만 바꾸는 child run이라면 router logits, top-k, accepted ledger와 logical output은 같아야 한다. stream/channel/buffer와 latency만 달라지는 것이 목표다. logical assignment가 바뀌면 performance-only 주장이 실패한다.

mHC Sinkhorn iterations만 바꾸면 dense/MoE expert output 이전은 같고 mixing matrix와 residual output 이후가 달라져야 한다. constraint residual과 latency가 trade-off한다. option마다 이렇게 바뀌어야 할 첫 tensor와 유지되어야 할 tensors를 적는 것이 실제 의미 설명이다.

**최종 현장 체크리스트**

**모델 함수.** dense/gated MLP equation과 projection order가 source·checkpoint와 맞는가. activation approximation과 dtype, intermediate rounding을 아는가. shared expert와 routed output, residual/mHC merge 순서를 복원했는가.

**router.** score function, group mask, top-k, normalization, correction bias, noise와 dtype을 확인했는가. main loss와 aux loss gradient를 분리했는가. tie와 capacity boundary의 불연속을 test했는가.

**assignment.** offered, accepted, dropped의 보존식이 맞는가. global expert ID와 source token tuple이 dispatch/combine round trip에서 유지되는가. zero-token expert와 rank가 합법적으로 처리되는가.

**expert compute.** grouped GEMM offsets와 weight layout, gate/up order, quantization scales가 맞는가. eager oracle과 forward/dX/dW가 맞는가. skew와 workspace memory를 rank max로 보았는가.

**통신.** DeepEP commit과 Buffer path, layout/handle schema, intra/inter-node 및 low-latency 분기를 확인했는가. stream event와 buffer lifetime, send/receive counts, payload identity를 관측하는가. 실행하지 않은 topology를 표시했는가.

**residual·mHC.** state shape와 mapping equations, Sinkhorn iterations/dtype, constraint residual, block caller와 recompute를 확인했는가. PR source의 상태와 tests 범위를 명시했는가. checkpoint stream identity가 보존되는가.

**학습 상태.** expert별 optimizer moments, router/controller, aux schedules, RNG, placement가 checkpoint에 있는가. EP size 변경에서 weights와 moments, router columns가 같은 global experts로 이동하는가. 첫 routed delta를 재계산했는가.

**운영.** routing control-plane과 network/compute data-plane metrics가 UpdateID로 연결되는가. capacity drop, hot expert, combine mismatch, mHC constraint, hang detector가 있는가. high-cardinality details를 안전한 artifact로 분리했는가.

**반증.** branch swap, expert swap, payload permutation, router detach, capacity drop, zero-token rank, checkpoint identity swap, mHC iteration 변경을 주입했는가. 각 failure가 예상 최초 detector에서 잡히는가. 수정 후 정상과 failure suite를 모두 반복했는가.

이 체크리스트의 모든 답에는 source 좌표나 fixture, manifest가 필요하다. “프레임워크가 처리한다”는 답은 owner를 찾지 못했다는 뜻이다. 반대로 작은 fixture가 통과했다는 사실을 multi-node production 보장으로 확대하지 않는다.

마지막 dossier는 독자가 새로운 MoE model을 만났을 때 빈 표로 재사용할 수 있다. config와 functions, state, collectives, checkpoints를 채우고 근거 없는 칸을 남긴다. 이 습관이 빠르게 변하는 architecture 이름보다 오래 간다.

9장의 완성 조건은 expert 개념을 설명하는 것이 아니라 dense token과 routed token 각각의 전체 생애를 재현하는 것이다. source hidden에서 projections와 routing, network, expert, residual을 거쳐 loss로 가고, backward와 optimizer, checkpoint를 거쳐 다시 같은 identity로 돌아와야 한다.

**봉인된 증거를 다음 revision에서 재사용한다**

source가 바뀌면 먼저 Qwen·DeepSeek·GLM model classes와 DeepEP, mHC symbols의 body fingerprints와 callers를 diff한다. router default, expert layout, handle schema, mixing config 가운데 실제 semantic change를 찾는다. line 이동만으로 모든 fixture를 무효화하지 않는다.

model code만 바뀌면 eager routing과 gradients를, DeepEP만 바뀌면 동일 assignment의 dispatch/combine과 overlap을, mHC만 바뀌면 mixing matrix와 residual fixture를 우선 실행한다. shared integration GoldenBatch는 마지막에 전체 경계를 확인한다. 영향 범위에 맞춘 재검증이 속도와 깊이를 함께 지킨다.

golden output을 새 implementation 결과로 자동 갱신하지 않는다. diff가 의도한 equation인지 paper·source·migration note로 확인하고 독립 oracle을 수정한다. 성능 최적화가 numerical tolerance를 넓혀야 한다면 error distribution과 downstream growth를 근거로 승인한다.

checkpoint converter가 바뀌면 global expert IDs와 router columns, expert matrices, optimizer moments, mHC streams를 ID-specific canary로 검증한다. 값 합계나 전체 checksum만으로 순서 permutation을 잡지 못할 수 있다. first routed update까지 비교한다.

이 evidence lifecycle은 특정 release에서 끝나지 않는다. 새로운 MoE architecture와 communication backend를 같은 표에 추가하되, 공통 개념과 project-specific 사실을 분리한다. 독자는 source revision을 바꿔도 어느 함수와 state, test를 다시 파야 하는지 즉시 알 수 있다.

최종 서명은 tested model/config/topology/dtype, source commits, fixtures, PASS와 `NOT_RUN`, owner와 재검토 조건을 포함한다. 이 서명이 있어야 다음 장이 model 전체를 해부할 때 MoE 부분을 확인된 계약으로 사용할 수 있다.

서명 직전에는 source token 하나를 다시 골라 router logits와 selected weights를 손으로 검산한다. dispatch tuple과 destination buffer, expert-specific output, combine contribution을 차례로 확인하고 total output을 eager oracle과 비교한다. backward에서는 selected-weight, expert matrix, input residual gradient를 확인한다.

같은 token을 checkpoint round trip과 topology reshard 뒤에도 반복한다. global expert identity와 optimizer moment가 유지되고 first delta가 맞아야 한다. routing controller나 mHC state가 누락되면 weight parity만으로 승인하지 않는다.

마지막으로 성능 instrumentation을 제거한 configuration에서 logical digest가 유지되는지 본다. debug synchronization을 끄면서 race가 나타나지 않는지 stress한다. 실행하지 못한 multi-node·CUDA path는 정확한 입력과 invariant, trace points를 남긴다. 이 인계는 미지의 범위를 감추지 않으면서 다음 검증을 바로 시작하게 한다.

모든 artifact는 UpdateID·LayerID·ModelRevision으로 연결한다. 시간순 로그만으로 expert 생애를 추정하지 않는다. 이 세 식별자가 같을 때만 router, 통신, expert compute, residual과 gradient 기록을 하나의 사건으로 결합한다. 독립 검토자가 같은 tuple을 왕복해 재현하면 9장의 봉인을 승인한다.

**고정 소스 좌표에서 실행 계약을 복원한다**

소스 읽기의 출발점은 클래스 이름이 아니라 고정 revision의 함수 body다. 이 장의 Qwen 기준 좌표는 `sources/transformers-main-qwen4exp/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py`다. 그 파일에서 `Qwen3MoeMLP`는 193행 부근, packed expert 구현은 `Qwen3MoeExperts`가 시작하는 210행 부근, router는 `Qwen3MoeTopKRouter`가 시작하는 249행 부근, sparse block은 `Qwen3MoeSparseMoeBlock`이 시작하는 270행 부근이다. 줄 번호는 탐색 표지이며 최종 증거에는 revision과 함수 body fingerprint를 같이 남긴다. 줄이 이동해도 body가 같다면 의미 변경으로 오판하지 않고, 줄이 같아도 body가 바뀌면 새 ModelRevision으로 취급한다.

첫 통과에서는 constructor가 만드는 parameter와 config 참조만 적는다. MLP가 만드는 gate, up, down weight의 논리 shape, experts가 packed tensor를 어떤 축으로 쌓는지, router weight가 hidden에서 expert logit으로 가는 방향을 표로 옮긴다. 이때 checkpoint key를 아직 함수 의미로 해석하지 않는다. 저장 이름은 serialization 계약이고, transpose와 packing을 거친 뒤 실제 GEMM이 소비하는 layout은 실행 계약이기 때문이다. 같은 `gate_proj`라는 문자열도 `[E,F,C]`, `[E,C,F]`, 또는 두 projection이 합쳐진 `[E,2F,C]`를 가리킬 수 있다.

둘째 통과에서는 `forward` 호출 순서를 따라 logical tensor에 새 이름을 붙인다. 입력을 `x_source`, router 원출력을 `router_logits`, 선택 뒤를 `selected_ids`와 `selected_weights`, packed expert 입력을 `expert_in`, expert 결과를 `expert_out`, 원 token 순서로 복원한 값을 `routed_sum`이라 하자. sparse block의 반환값이 hidden만인지 router 부가 출력도 포함하는지 caller에서 확인한다. 함수가 tensor 하나를 반환한다고 router state가 사라진 것은 아니다. loss caller가 hook, module field, 별도 tuple을 통해 통계를 수집할 수 있으므로 호출자까지 올라간다.

셋째 통과에서는 tensor shape뿐 아니라 identity를 붙인다. `x_source[n]`의 `n`은 평탄화 index이면서 원래 `(batch, position)`으로 역변환할 수 있어야 한다. assignment에는 `(UpdateID, LayerID, source_rank, token_id, slot_id, global_expert_id)`를 붙인다. expert별 sort가 일어나면 새 buffer offset을 추가하되 source identity를 덮어쓰지 않는다. combine은 이 tuple에서 token과 slot을 사용하고 checkpoint는 global expert를 사용한다. 하나의 정수 index를 세 의미로 재사용하면 single-rank test는 통과해도 reshard나 top-k에서 깨진다.

넷째 통과에서는 autograd 경계를 찾는다. 일반 tensor 연산이면 저장되는 중간값을 framework가 결정하지만 activation checkpointing은 forward를 다시 실행한다. custom autograd 함수면 `save_for_backward` 또는 동등한 saved metadata가 계약이다. router의 top-k 결과, dispatch permutation, expert offsets, combine weight 중 무엇을 저장하고 무엇을 재계산하는지 확인한다. 재계산이 stochastic router noise나 비결정적 slot allocation을 다시 호출하면 같은 weight로도 다른 graph를 미분할 수 있다. forward assignment digest와 backward assignment digest가 같은지 검증하는 이유다.

다섯째 통과에서는 효과를 source에서 바로 추정하지 않고 fixture로 닫는다. packed expert가 빠를 것이라는 설명은 성능 가설이고, eager loop와 수학적으로 같다는 것은 정확성 명제다. 작은 비대칭 expert weight로 두 경로의 forward, input gradient, expert별 weight gradient를 비교한다. 그 뒤 token 수와 expert skew를 늘려 kernel 선택, workspace, latency를 잰다. parity와 speed를 한 실험에서 동시에 판정하면 오차가 성능 설정 때문인지 함수 불일치 때문인지 분리하기 어렵다.

반례에서 좌표 연결이 실제로 필요한 이유가 드러난다. router 열 두 개와 expert 두 개를 함께 교환하면 최종 출력은 같을 수 있다. output parity만 보면 checkpoint mapping 오류를 놓친다. router 열만 교환하면 선택 expert가 달라지고, expert만 교환하면 같은 ID가 다른 함수를 실행한다. 복구 검사는 출력뿐 아니라 선택 ID, expert parameter digest, optimizer moment digest를 함께 비교한다. 이 세 값이 일치할 때만 identity-preserving load라 부른다.

## 9.13 한 token의 source contract를 optimizer commit까지 잇는다

Golden fixture를 실제 model source에 꽂아 norm, dense 또는 router branch, dispatch, expert, combine과 residual을 순서대로 확인한다. 같은 TokenID가 optimizer update와 checkpoint artifact까지 이어져야 실행 계약이 닫힌다.

입력 `x`가 `[C]`인 token 하나라고 하자. source 좌표의 projection 순서를 따라 `g=xW_g`, `u=xW_u`, `s=SiLU(g)`, `m=s⊙u`, `y=mW_d`를 계산한다. 원장에는 각 tensor의 논리 shape, 저장 dtype, 누산 dtype, device, storage alias, producer 함수와 consumer 함수를 적는다. batch 전체 checksum만 저장하지 않고 선택한 canary token의 작은 slice와 고정된 FP64 oracle을 남긴다. checksum은 permutation을 잡는 데 유용하지만 어느 연산에서 차이가 시작했는지를 알려 주지 않는다.

forward에서 가장 먼저 확인할 failure는 projection 의미 교환이다. `W_g`와 `W_u`는 shape가 같을 수 있고 random input에서는 최종 분포도 비슷할 수 있다. 그러나 `g`에 큰 음수, 0 근처, 큰 양수를 만들도록 설계한 입력에서는 SiLU가 비대칭적으로 반응한다. gate와 up을 바꾸면 `s`, `m`, `y`가 순서대로 갈린다. 최초 차이가 GEMM 출력인지 activation인지 알아야 converter, kernel, activation 설정 중 올바른 owner에게 문제를 보낼 수 있다.

backward 원장은 upstream `dy`에서 시작한다. `dm=dyW_d^T`, `dW_d=m^Tdy`, `du=dm⊙s`, `ds=dm⊙u`, `dg=ds⊙SiLU'(g)`를 거쳐 `dW_u=x^Tdu`, `dW_g=x^Tdg`, `dx=duW_u^T+dgW_g^T`를 만든다. 실제 batch에서는 outer product가 token 축 reduction이 된다. gradient accumulation, data parallel reduction, optimizer update 전후를 구분하려면 local contribution, reduced gradient, clipped gradient, optimizer-consumed gradient를 서로 다른 state로 기록한다.

tensor parallel이 들어가면 고정 함수는 유지되지만 ownership이 달라진다. gate와 up을 intermediate 축으로 나누면 각 rank는 `F_local` 좌표만 계산한다. down weight도 대응하는 입력 축으로 나뉘고 rank별 `[N,C]` partial output이 합쳐진다. 검증은 local shard가 올바른 global intermediate range를 소유하는지부터 시작한다. gate의 0번 shard와 up의 1번 shard를 곱하는 오류는 shape와 collective를 모두 만족하면서 값만 조용히 망가뜨린다.

checkpoint에는 세 projection의 parameter와 optimizer state가 같은 shard convention으로 저장되어야 한다. fused packed weight를 저장하는 구현과 logical weight 세 개를 저장하는 구현 사이 converter는 split 축, 순서, transpose를 명시해야 한다. AdamW라면 parameter뿐 아니라 first moment와 second moment도 같은 변환을 받는다. FP32 master weight가 있는 low-precision recipe는 네 번째 representation이다. load 직후 weight parity만 확인하고 첫 update 뒤 갈라지는 사례는 대개 moment나 master weight 변환 누락이다.

MLP failure를 복구할 때는 마지막 정상 output부터 역추적하지 않는다. 같은 `x_source`를 고정하고 projection 출력부터 순서대로 비교한다. GEMM 직후 다르면 weight/shard/layout을, activation 직후 다르면 SiLU 식과 dtype을, product 직후 다르면 split과 alignment를, down 이후 다르면 reduction과 output layout을 본다. residual add 이후에만 다르면 branch 외부의 scale, dropout, dtype, alias가 범인이다. 이 first-difference 순서가 커널 교체와 checkpoint 변환을 같은 절차로 진단하게 한다.

성능 효과도 원장의 state 변화로 설명한다. gate와 up을 한 GEMM으로 합치면 launch 수와 input read가 줄 수 있지만 logical `g`와 `u`는 여전히 존재한다. activation과 product를 fuse하면 중간 materialization byte가 줄지만 backward를 위해 어떤 값을 저장하는지가 바뀐다. recomputation을 켜면 저장 메모리는 감소하고 추가 GEMM이 생긴다. “fusion으로 빨라졌다”가 아니라 launch, bytes, saved tensors, recompute FLOP, end-to-end step time을 각각 전후 비교한다.

### router decision boundary와 selected branch를 source에서 찾는다

router source 함수에서 `x_source`가 어떤 dtype으로 projection되는지부터 고정한다. hidden이 BF16이어도 router GEMM이나 softmax는 FP32로 승격될 수 있다. `router_logits`에 bias, correction bias, group mask, jitter가 어느 순서로 적용되는지 분리한다. 선택에 쓰는 score와 mixture weight에 쓰는 score가 같다는 가정을 버린다. aux-loss-free controller는 selection score만 보정하고 combine weight는 원래 probability에서 가져갈 수 있다. 두 tensor를 모두 보존해야 이 차이를 검증할 수 있다.

top-k 이전의 후보 집합을 `eligible`, top-k 직후를 `selected`, capacity 승인 뒤를 `accepted`라 부른다. group routing이 있으면 `eligible` 전에 group score와 group mask가 존재한다. 각 전이는 입력 cardinality, 출력 cardinality, tie 정책, dtype, owner 함수를 가진다. `selected` 수가 `N×k`라고 해서 `accepted`도 같지는 않다. drop, reroute, duplicated expert 제거 정책이 개입할 수 있기 때문이다. 원장은 전이마다 수를 세고 reason code를 남긴다.

결정 경계의 반례를 의도적으로 만든다. expert 0과 1의 logit 차이가 FP32에서는 양수지만 BF16 반올림 뒤 0이 되도록 값을 고른다. stable tie-break가 expert 0을 택하는지, backend top-k가 임의 순서를 내는지 본다. 다음에는 k번째와 k+1번째 score를 아주 가깝게 두고 finite difference를 계산한다. 선택 집합이 유지되는 구간에서는 selected weight gradient와 수치 미분이 맞아야 하고, 경계를 넘는 perturbation은 미분 불연속으로 별도 표시한다.

router state와 학습 objective도 이어야 한다. main loss는 accepted expert output과 selected mixture weight를 통해 router에 gradient를 보낸다. load-balancing loss는 평균 probability와 assignment fraction 같은 집계를 통해 추가 gradient를 만들 수 있다. z-loss는 logit scale에 작용한다. controller bias update는 autograd 바깥의 count 기반 state transition일 수 있다. `router_total_grad` 하나만 보면 네 경로를 분리할 수 없으므로 main, balance, z, controller delta를 독립적으로 계산하거나 ablation한다.

분산 환경에서 통계 domain을 확인한다. 각 data-parallel rank가 자기 microbatch만으로 balance loss를 계산하면 global batch의 expert fraction과 다를 수 있다. EP group 전체 count를 reduce하더라도 probability gradient가 어느 token에 돌아가는지 구현에 따라 달라진다. gradient accumulation 동안 microbatch별 auxiliary scalar를 단순 평균하면 nonpadding token 수가 다른 경우 분모가 틀어진다. numerator와 valid-token denominator를 누적한 global reference를 만들어 비교한다.

checkpoint에서 router weight만 저장하면 충분하지 않을 수 있다. correction bias, expert count moving average, controller step, temperature schedule, jitter RNG, auxiliary coefficient schedule이 다음 assignment를 바꾼다. 이 state가 parameter가 아니어서 일반 `state_dict`에서 빠지는지 찾는다. 저장 전과 load 후에 같은 hidden을 넣어 logits뿐 아니라 eligible, selected, accepted, weight를 비교한다. resume 첫 step에서만 비교하면 optimizer와 data order 변화가 함께 섞인다.

복구는 symptom별로 다르다. entropy가 급락했지만 logits와 selected가 의도대로라면 data distribution 변화일 수 있다. logits부터 갈리면 hidden 또는 router weight/state를 본다. selected는 같고 accepted만 갈리면 capacity domain, token order, slot allocation을 본다. accepted는 같고 output만 갈리면 dispatch나 expert compute 문제다. 이 분류는 “router collapse”라는 넓은 이름을 source 함수 단위의 조치로 바꾼다.

**Switch Transformers의 한 token을 router에서 combine까지 추적한다**

여기서는 추상적인 MoE 의사 코드가 아니라 Hugging Face Transformers 공식 저장소의 `SwitchTransformersTop1Router.forward`, `SwitchTransformersExperts.forward`, `SwitchTransformersSparseMLP.forward`를 한 경로로 읽는다. 이 구현의 top-k는 `k=1`이다. 따라서 현대 top-2 MoE의 일반형을 모두 대표하지는 않지만, **선택과 capacity 승인, dispatch, expert 실행, combine이 서로 다른 상태**라는 사실과 shape 하나가 그 계약을 어떻게 무력화하는지를 짧게 드러낸다. 기준 revision은 `550d7b3834670483a4df436541272c055dc364bf`, 핵심 좌표는 `modeling_switch_transformers.py` 90~107행과 164~191행이다.

핵심은 다음 두 짧은 조각이다. router 쪽은 probability에서 expert를 하나 고른 다음 token 순서대로 expert별 누적 순위를 매겨 capacity 밖의 assignment를 0으로 만든다.

```python
router_logits, expert_index = torch.max(router_probs, dim=-1, keepdim=True)
expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)
token_priority = torch.cumsum(expert_index, dim=-2)
expert_capacity_mask = token_priority <= self.expert_capacity
expert_index = expert_index * expert_capacity_mask
```

의도된 상태부터 풀면 router probability는 token별 `[T,E]`, top-1 slot을 보존한 selected mask는 `[T,K=1,E]`다. 같은 expert가 몇 번째 token을 받았는지 세려면 `T`축을 누적해야 한다. 그런데 Sparse MLP는 187행에서 `[B,S,C]`를 `[T,C]`로 편 뒤 router를 호출한다. `torch.max(..., keepdim=True)`의 index는 `[T,1]`, `one_hot`은 `[T,1,E]`이고 `cumsum(dim=-2)`는 token 축이 아니라 길이 1인 slot 축을 누적한다. capacity가 1 이상이면 모든 selected row의 priority가 늘 1이므로 overflow가 생기지 않는다. 이것은 문서의 의도나 논문 일반론이 아니라 해당 revision의 실제 shape 추론이다.

3차원 `[B,S,C]`를 router에 직접 넣어도 one-hot shape는 `[B,S,1,E]`이고 `-2`축은 다시 singleton slot이다. 반면 공식 `test_max_routing_capacity`는 router가 반환한 세 번째 tensor를 다시 `argmax`해 `[B,S,E]`를 만들고 그 sequence 축에 `cumsum(dim=-2)`을 적용한다. 이 test는 router가 실제 반환한 accepted mask와 reference를 비교하지 않는다. capacity 상한만 검사하므로 production path의 축 불일치를 잡지 못한다. 이처럼 test 이름이 기능과 같아 보여도 assertion이 어느 tensor를 관측하는지 끝까지 읽어야 한다.

Sparse MLP는 accepted mask를 expert별 boolean index로 바꾸고, 해당 token만 expert에 보낸다. 마지막 줄에서 선택 probability를 expert 출력에 곱한다.

```python
expert_mask = selected_experts.permute(2, 1, 0)
for expert_idx in expert_hit:
    idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
    current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
    current_hidden_states = self[f"expert_{expert_idx[0]}"](current_state) \
        * routing_weights[top_x, idx, None]
    final_hidden_states.index_add_(0, top_x, current_hidden_states)
```

`final_hidden_states`는 `[T,C]`의 zero buffer다. mask를 `[E,K,T]`로 바꾼 뒤 `where`가 slot index `idx`와 source token `top_x`를 복원한다. expert 입력과 출력은 `[M_e,C]`, routing weight는 `[M_e,1]`이다. `index_add_`는 contribution을 원래 token row에 더하므로 top-2로 일반화해도 combine reduction을 수행할 수 있다. 이 revision의 `K=1`에서는 사실상 weighted scatter다. 정말 drop된 token이라면 zero row가 남지만, 앞의 축 문제 때문에 capacity 1 이상에서는 그 상태에 도달하지 않는다.

작은 fixture로 모든 중간값을 닫아 보자. `B=1,S=3,C=1,E=2,capacity=1`이고 router probability가 차례로 `(0.8,0.2)`, `(0.75,0.25)`, `(0.25,0.75)`라고 하자. 이는 logit `(ln 4,0)`, `(ln 3,0)`, `(0,ln 3)`의 softmax다. hidden scalar는 `(1,2,3)`, expert 함수는 `f_0(x)=2x`, `f_1(x)=-x`로 둔다.

```text
selected expert       = [0, 0, 1]
selected mask          = [[1,0], [1,0], [0,1]]
intended priority      = [[1,0], [2,0], [2,1]]
actual priority        = [[1,0], [1,0], [0,1]]
intended accepted      = [[1,0], [0,0], [0,1]]
actual accepted        = [[1,0], [1,0], [0,1]]
expert contributions  = [2, 4, -3]
max router probability = [0.8, 0.75, 0.75]
actual combined output = [1.6, 3.0, -2.25]
```

두 번째 token에서 최초 불일치는 router score나 top-1 선택이 아니라 `token_priority`다. 의도한 semantic oracle은 expert 0의 두 번째 배정을 priority 2로 세지만 실제 `[T,1,E]` tensor는 slot 축을 누적해 다시 1을 낸다. 따라서 의도한 accepted load `(1,1)` 대신 실제 load `(2,1)`이 되고 drop rate는 0이다. probability, selected mask, priority, accepted mask를 따로 저장하지 않으면 이 차이는 “expert 0이 조금 더 바쁘다”는 통계로 묻힌다.

upstream gradient를 세 row 모두 1로 두면 실제 expert 입력 경로의 gradient는 `(1.6,1.5,-0.75)`다. semantic oracle에서 drop되어야 할 둘째 token의 값은 0이어야 한다. selected probability gradient는 실제 `(2,4,-3)`, oracle은 `(2,0,-3)`이다. softmax logit gradient도 둘째 token에서 실제 `(0.75,-0.75)`, oracle `(0,0)`으로 처음 갈린다. hard selection과 capacity 비교 자체에는 일반적인 autograd gradient가 없지만 잘못 살아남은 branch에는 정상적인 연속 gradient가 흘러 expert 0의 parameter와 optimizer moment까지 오염시킨다.

이 fixture의 first-divergence 검사는 네 단계로 둔다. 첫째 router logit/probability가 다르면 dtype, classifier weight, jitter를 본다. 둘째 probability는 같고 selected mask가 다르면 tie와 top-k를 본다. 셋째 selected는 같고 accepted가 다르면 token 순서, capacity 값과 capacity domain을 본다. 넷째 accepted까지 같고 출력이 다르면 boolean dispatch, expert weight, scatter 위치와 combine probability를 본다.

expert imbalance가 gradient imbalance로 번지는 최초 지점은 priority tensor다. 이 fixture에서는 expert 0의 두 번째 token이 의도와 달리 expert parameter gradient와 hidden gradient에 기여한다. selected count만 보고 capacity가 적용됐다고 믿거나 최종 loss만 보면 optimizer moment가 왜 reference와 갈리는지 설명할 수 없다.

공식 test의 증명 범위도 읽어야 한다. `test_max_routing_capacity` 917~931행은 총 assignment 상한만 확인하며 실제 router의 accepted mask와 reference parity를 검사하지 않는다. `test_token_dropping` 1032~1040행은 capacity 0에서 출력이 모두 0인 극단만 본다. singleton priority도 `1<=0`에서는 false이므로 이 test는 통과할 수 있다. 위의 세-token fixture는 기존 test를 대체하는 것이 아니라, capacity 1의 **부분 overflow identity와 gradient**라는 빈 구간을 채우는 회귀 oracle이다.

이 구현을 top-2나 expert-parallel 구현으로 옮길 때 보존할 것은 Python loop가 아니다. `selected=[B,S,K]`, `accepted=[B,S,K]`, `weight=[B,S,K]`, expert별 ragged row와 source token/slot, combine reduction이라는 상태 계약이다. capacity가 없는 dropless 구현에서는 selected와 accepted가 같고, capacity padding만 하는 구현에서는 logical accepted 수와 physical row 수가 다르다. 구현 이름이 모두 MoE여도 어느 상태가 사라지고 어느 상태가 추가되는지를 비교해야 같은 함수인지 판단할 수 있다.

### assignment ledger를 count·payload protocol로 승격한다

single-process scatter/gather에서 만든 assignment ledger를 EP 통신의 source of truth로 사용한다. 각 accepted row에는 source token, slot, global expert, owner rank, local expert, destination offset, combine weight가 있다. pack 함수는 이 row 순서를 activation buffer에 투영하고, count 함수는 destination별 row 수를 센다. metadata와 activation이 서로 다른 sort key를 쓰지 않았는지 digest로 확인한다. payload 값 합만 비교하면 같은 값의 token이 뒤바뀐 permutation 오류를 놓친다.

count exchange와 payload exchange는 하나의 프로토콜이다. rank `r`의 `send_count[r,d]`는 rank `d`의 `recv_count[d,r]`와 같아야 한다. 모든 pair를 합한 값은 remote accepted assignment 수와 같아야 하고 local assignment는 별도 경로로 세야 한다. zero count도 명시적 상태다. 어떤 rank가 empty라 collective를 건너뛰면 다른 rank가 기다릴 수 있으므로 호출 sequence number와 process group ID까지 flight recorder에 남긴다.

payload byte 원장은 activation만 세지 않는다. hidden width `C`, element byte `b`, remote assignment `A_remote`이면 기본 forward activation은 `A_remote×C×b`지만 routing weight, source offset, expert offset, quantization scale, padding과 alignment가 붙는다. backward에는 expert input gradient와 필요한 metadata가 역방향으로 흐른다. 구현이 metadata를 host에 복사하거나 count를 CPU에서 동기화하면 network byte에는 작아도 critical path에 큰 bubble을 만든다. profiler event와 원장을 같은 UpdateID로 연결한다.

owner rank에서 unpack한 뒤 expert별 offset은 grouped GEMM descriptor가 된다. `expert_offsets[e+1]-expert_offsets[e]`가 실제 accepted count와 같아야 한다. capacity padding을 쓰면 logical count와 physical rows를 분리하고 padding mask를 둔다. 빈 expert는 길이 0의 합법 구간이다. offset이 단조 증가하고 마지막 값이 packed row 수와 같은지 검사한다. off-by-one은 다음 expert weight로 token을 보내면서도 kernel 자체는 정상 종료할 수 있다.

reverse dispatch는 forward ledger를 거꾸로 재생한다. expert output row를 source rank와 source token/slot으로 보내고, source에서 slot별 weight를 곱해 token별 sum을 만든다. combine 전에는 accepted row마다 정확히 하나의 output이 있는지, combine 후에는 각 token의 accepted slot 수와 contribution 수가 같은지 검사한다. fully dropped token은 contribution 0이라는 명시적 state이며 residual-only가 정책인지 오류인지 reason code로 결정한다.

backward에서도 같은 ledger가 필요하다. token output gradient는 combine weight를 곱해 각 accepted expert output row로 분기되고 reverse 통신으로 owner에게 간다. expert backward가 만든 input gradient는 forward dispatch 반대 방향으로 source에 돌아오며 slot별 합이 router 이전 hidden gradient에 더해진다. selected weight gradient는 expert output과 token gradient의 내적으로 생긴다. forward ledger를 잃고 routing을 재계산하면 tie, noise, capacity 순서 때문에 다른 전문가를 미분할 위험이 있다.

장애 복구는 collective 한가운데의 부분 결과를 commit된 state로 보지 않는다. 어느 rank는 expert GEMM까지 끝냈고 다른 rank는 receive 중일 수 있다. 전체 update가 optimizer commit marker에 도달하지 않았다면 마지막 consistent checkpoint와 data cursor에서 group 전체를 재실행한다. exactly-once 재사용 프로토콜과 global transaction ID가 구현되어 있지 않은데 일부 buffer를 살리는 것은 중복 또는 누락 gradient를 만들 수 있다.

반례 fixture는 payload permutation이 가장 강하다. 두 token의 activation 합이 같도록 만들고 metadata 순서만 교환한다. aggregate checksum은 같아도 expert-specific output과 source identity digest는 달라져야 한다. 다음에는 destination count 하나를 1 줄여 hang 또는 bounds detector가 예상 지점에서 작동하는지 본다. 마지막에는 zero-token rank가 모든 collective에 참여하면서 빈 buffer를 안전하게 처리하는지 확인한다. 세 반례가 통과해야 평균 throughput 측정으로 넘어간다.

### grouped GEMM을 ragged expert batch로 실행한다

expert 함수는 각각 독립된 gated MLP지만 실행기는 여러 expert의 가변 크기 GEMM을 한 호출로 묶을 수 있다. source 좌표에서 grouped GEMM wrapper, descriptor 생성자, kernel dispatch 조건을 찾는다. 논리 입력은 expert별 `[M_e,C]`이고 gate/up weight는 expert별 `[C,F]`, down weight는 `[F,C]`다. 물리 buffer가 `[ΣM_e,C]`로 이어져 있어도 expert 경계는 offsets에 남아 있다. offsets가 곧 함수 identity의 일부다.

첫 reference는 Python 또는 eager loop로 expert마다 독립 계산한다. expert weight를 서로 구별되는 대각 또는 저차원 변환으로 만들고 grouped output과 row별 비교한다. forward parity 뒤 `dX`, `dW_gate`, `dW_up`, `dW_down`을 expert별로 비교한다. 전체 weight gradient norm만 비교하면 expert 2와 3의 gradient가 교환된 오류를 놓친다. zero-token expert의 gradient가 `None`, 명시적 zero, stale buffer 중 무엇인지도 optimizer 계약과 연결한다.

두 번째 reference는 packing layout을 검증한다. packed gate/up tensor가 `[E,2F,C]`라면 logical half의 순서와 stride를 확인한다. kernel이 `[E,C,2F]`를 기대하면 converter 또는 wrapper가 transpose해야 한다. quantized weight는 scale의 expert 축, block 축, gate/up 절반을 따라야 한다. weight data를 올바르게 permutation하고 scale을 그대로 두면 finite output이 나오지만 expert별 크기가 체계적으로 틀어진다. canary scale을 expert마다 다르게 두어 잡는다.

자원 상태는 `M_e` 분포에 좌우된다. 균등하면 비슷한 tile을 반복하지만 skew가 심하면 하나의 큰 GEMM과 많은 작은 GEMM이 섞인다. kernel launch 수가 줄어도 작은 expert의 tile waste와 큰 expert의 tail이 남는다. useful FLOP는 accepted 실제 row를 기준으로, executed FLOP는 padding과 tile rounding을 포함해 센다. 둘의 비율, HBM byte, workspace high-water, kernel duration을 expert histogram과 같이 기록한다.

capacity padding은 regular shape를 주지만 model policy와 실행 최적화를 혼동하기 쉽다. capacity를 넘은 real assignment를 drop하는 것은 함수 변경이고, capacity까지 dummy row를 채우는 것은 같은 accepted set을 실행하는 layout 선택일 수 있다. padding row가 loss, gradient, expert count에 들어가면 더 이상 단순 layout이 아니다. physical row mask가 forward output, backward reduction, auxiliary statistics에서 일관되게 적용되는지 검사한다.

overlap에서는 buffer lifetime이 중요하다. chunk 0의 expert input을 GEMM이 읽는 동안 communication stream이 같은 storage를 chunk 1로 덮어쓰면 드문 corruption이 생긴다. producer event, consumer wait, reuse event를 명시하고 debug synchronization을 제거한 stress test를 실행한다. deterministic routing plan을 고정한 채 overlap on/off logical digest를 비교한다. on에서만 값이 갈리면 router가 아니라 stream ownership 문제로 범위를 줄인다.

checkpoint reshard는 packed expert 축을 global ID 기준으로 재배열한다. EP size 2에서 rank마다 네 expert를 소유하다 EP size 4에서 두 expert씩 소유한다고 하자. 각 global expert의 gate/up/down weight, quantization scale, optimizer moments, master weight를 같은 destination으로 보낸다. local slot 번호만 복사하면 rank map이 바뀔 때 identity가 틀어진다. load 뒤 expert별 canary output과 첫 optimizer delta를 검사한다.

복구의 최소 단위는 kernel 재실행이 아니라 logical assignment다. OOM으로 grouped workspace를 줄여 다른 kernel을 선택해도 accepted rows와 expert mapping은 같아야 한다. capacity를 낮춰 OOM을 피하면 assignments와 objective가 바뀌므로 별 RunRevision이다. microbatch를 줄이면 router 통계와 capacity domain도 달라질 수 있다. 성능 또는 메모리 완화가 함수 보존인지 변경인지 ledger로 판정한다.

**residual 경로를 깊이 방향의 durable state로 본다**

residual은 layer 입력과 출력 사이의 단순 덧셈처럼 보이지만 모델 전체에서 가장 오래 살아남는 token state다. source 좌표에서는 norm 호출, attention branch, MLP 또는 MoE branch, residual add의 정확한 순서를 찾는다. pre-norm이면 branch가 normalized view를 받고 identity stream은 원래 값을 유지한다. post-norm이면 합산 뒤 norm이 다음 layer state를 만든다. parallel residual과 sequential residual도 호출 graph가 다르므로 식을 source와 일치시킨다.

원장에는 `h_layer_in`, `h_norm_for_attn`, `attn_out`, `h_after_attn`, `h_norm_for_mlp`, `mlp_or_moe_out`, `h_layer_out`을 둔다. parallel 구조라면 두 norm 입력이 같은 tensor인지 확인한다. in-place 연산은 logical 이름이 달라도 storage를 공유할 수 있어 version counter와 alias를 적는다. activation checkpointing이 이전 residual을 다시 필요로 하는데 storage가 덮였다면 forward는 정상이고 backward에서만 깨질 수 있다.

dtype 경계도 durable state의 일부다. branch GEMM은 BF16 또는 FP8을 쓰고 accumulator는 FP32일 수 있으며 residual add는 다시 BF16일 수 있다. fused add-norm이 FP32 residual buffer를 별도로 유지하는 구현도 있다. 한 layer 오차는 작아도 같은 block을 수십 번 반복하면 drift가 누적된다. 고정된 선형 branch를 여러 depth로 반복해 FP64 reference와 RMS, max error, 방향 cosine을 비교한다.

MoE에서는 fully dropped token의 branch output이 0일 수 있다. 이때 residual 덕분에 hidden이 살아남으므로 loss가 NaN이 되지 않고 routing failure가 숨을 수 있다. routed output zero ablation과 shared-expert-only ablation을 만들어 branch 기여를 측정한다. residual output이 finite라는 사실은 expert 경로가 정상이라는 증거가 아니다. accepted count와 routed norm을 함께 gate로 둔다.

backward에서 identity 경로는 upstream gradient를 직접 전달하고 branch Jacobian이 추가된다. residual scale이나 learned mixing이 있으면 이 단순 합이 바뀐다. branch를 0으로 만든 fixture에서 `dh_in`이 예상 identity 또는 scaling과 같은지 확인한다. attention을 0으로, MLP를 0으로, 둘 다 켠 세 fixture는 parallel과 sequential graph를 구분한다. 같은 output shape만으로 residual 구현을 승인하지 않는다.

mHC 같은 다중 stream mixing은 state 차원을 늘린다. 고정 소스가 있는 경우 config key에서 stream 수와 initialization을, module에서 pre-mix와 post-mix 함수를, kernel에서 constraint 적용 dtype과 iteration을, checkpoint에서 mixing parameter key를 찾는다. 로컬 runtime의 `vllm/model_executor/layers/mhc.py`와 kernel 구현은 실행 경로 증거이지 특정 학습 모델의 architecture 선언을 자동으로 증명하지 않는다. 모델 revision의 constructor와 caller가 실제로 이를 선택하는지 별도로 확인한다.

checkpoint에서 residual 자체는 보통 activation이라 저장하지 않지만, 그 진화를 결정하는 norm weight, branch scale, mixing matrix, controller state는 저장한다. pipeline checkpoint나 activation snapshot을 지원한다면 token position과 layer boundary를 명시해야 한다. layer 10 출력 activation을 layer 11 입력으로 복원하면서 norm 전후를 혼동하면 shape가 같아도 재시작 결과가 달라진다. durable checkpoint와 ephemeral activation checkpoint를 용어상 분리한다.

복구는 residual checksum을 경계 표지로 쓴다. layer 입력까지 같고 router logits부터 다르면 router 내부 문제다. MoE combined output까지 같고 layer output만 다르면 residual scale, norm, dtype, alias를 본다. layer output이 같고 다음 layer norm에서 갈리면 다음 caller 또는 fused kernel 경계다. 8장의 attention output checksum과 9장의 MLP/MoE checksum을 10장의 block atlas에서 같은 LayerID로 결합한다.

### optimizer moment와 update를 global ExpertID에 연결한다

forward와 backward가 맞아도 optimizer가 다른 parameter를 갱신하면 학습 함수는 틀린다. parameter inventory에서 dense MLP, router, routed expert, shared expert, residual mixing을 group별로 나눈다. 각 parameter에 global semantic ID, local tensor key, shard range, dtype, optimizer group, learning-rate schedule, weight-decay 정책을 붙인다. Python object identity나 local list position은 reshard 뒤 안정적인 ID가 아니다.

expert별 effective batch는 다르다. expert `e`가 받은 accepted token 수 `M_e`는 gradient noise와 magnitude에 영향을 준다. loss reduction이 global token mean이면 expert gradient는 자연스럽게 routed 빈도에 비례할 수 있다. expert 내부에서 다시 `M_e`로 평균하면 다른 objective가 된다. source에서 reduction 위치를 확인하고 synthetic duplicate-token fixture로 scaling을 검산한다. token을 두 배 복제했을 때 gradient가 유지되는지 두 배가 되는지는 denominator 계약에 달려 있다.

zero-token expert의 상태 전이는 특히 중요하다. `grad=None`이면 많은 optimizer가 update와 weight decay를 건너뛰지만 explicit zero tensor면 decoupled weight decay와 step counter가 진행할 수 있다. global optimizer step이 증가하면서 expert moment bias correction도 증가하는지, active step만 세는 controller가 있는지 확인한다. 두 정책 모두 가능하지만 checkpoint resume과 replica parity에서 일관되어야 한다.

gradient clipping은 raw gradient와 optimizer 사이의 별도 함수다. global norm clip은 hot shared path나 dense layers가 coefficient를 결정해 rare expert gradient도 축소한다. router 전용 clip, expert group별 clip, per-parameter clip은 다른 알고리즘이다. raw norm, reduction 후 norm, clip coefficient, clipped norm을 group과 expert별로 기록한다. “gradient가 작다”는 관측을 routing 희소성과 clipping 효과로 분해한다.

mixed precision에서는 overflow detector와 loss scale도 owner를 가진다. 한 expert에서 inf가 발생했을 때 전체 optimizer step을 skip하는지 해당 shard만 skip하는지 확인한다. 일부 rank만 skip하면 replicas와 collective sequence가 갈릴 수 있다. overflow flag가 DP 또는 전체 model group으로 reduce되는 source 좌표를 찾는다. skipped update에서도 controller bias나 scheduler가 진행한다면 다음 routing이 달라지므로 atomic step boundary를 정의한다.

checkpoint manifest는 optimizer class 이름보다 state mapping을 자세히 적는다. expert별 first/second moment, step, FP32 master, gradient scaler, scheduler position, router controller, balance moving statistics가 필요하다. EP reshard converter가 parameter만 처리하고 optimizer state를 새로 초기화한다면 warm resume가 아니라 weight-only restart다. 운영 문서와 실험 lineage에 그 차이를 표시한다.

one-step delta fixture는 mapping 오류를 가장 잘 잡는다. 각 expert weight에 고유 canary를 넣고 각 expert로 알려진 token을 routing한다. 저장·reshard·load 후 동일 batch로 forward/backward/update를 한 번 실행한다. global expert ID별 `weight_before`, `grad`, `moment_before`, `weight_after`를 reference와 비교한다. loss parity와 load 성공만으로는 optimizer 누락을 발견할 수 없다.

복구 시 router만 더 작은 학습률로 낮추는 임시 조치는 새 recipe다. collapse 증상을 줄일 수 있지만 원래 state를 복원한 것은 아니다. 먼저 checkpoint mapping, controller state, data cursor, capacity와 clip을 확인한다. 원인이 잘못된 expert moment인데 router LR을 낮추면 routing 변화만 느려지고 expert 손상은 남는다. 원인 수정 run과 완화 run을 별 revision으로 보존한다.

**체크포인트를 모델 함수의 직렬화된 증명으로 만든다**

좋은 checkpoint는 tensor 묶음이 아니라 다음 step의 함수를 재구성하는 manifest다. model config에는 hidden/intermediate 크기, expert 수, top-k, shared expert 수, routing normalization, capacity와 drop 정책, residual 구조가 있어야 한다. parameter에는 global semantic ID와 shape/layout이, optimizer에는 같은 ID의 moments가, runtime state에는 controller와 RNG가, 분산 manifest에는 rank group과 shard range가 있어야 한다.

저장 직전에는 모든 rank가 같은 committed UpdateID를 가리키는지 확인한다. collective 또는 gradient accumulation 도중 rank별 local state를 섞어 저장하면 파일은 모두 존재해도 논리 snapshot이 아니다. data cursor, accumulation microstep, optimizer step, scheduler step, controller step의 관계를 적는다. 저장 완료 marker는 모든 필수 shard와 manifest checksum이 durable해진 뒤 쓴다.

global expert identity는 checkpoint key 문자열보다 상위 계약이다. `experts.0`이 rank마다 local expert 0을 뜻한다면 global ID는 별 mapping에 있어야 한다. router weight 열 `e`, expert parameter `e`, optimizer state `e`, balance bias `e`가 같은 global expert를 가리켜야 한다. reshard converter는 이 묶음을 atomic record처럼 이동한다. 각 구성요소를 독립 sort하면 permutation이 생길 수 있다.

residual mixing이나 mHC state도 누락하기 쉽다. mixing parameter가 module parameter라면 일반 state dict에 들어가지만 normalization iteration counter, cached projection, controller 통계가 buffer 또는 runtime object일 수 있다. cache가 재생성 가능한지 trajectory에 영향을 주는 durable state인지 구분한다. 재생성 가능하더라도 source revision과 config가 같아야 한다. inference runtime의 precomputed kernel state를 학습 checkpoint 필수 state와 혼동하지 않는다.

로드 검증은 네 층으로 한다. 구조 검사는 key, shape, dtype, global ID를 본다. 값 검사는 expert별 digest와 selected slices를 본다. 함수 검사는 golden hidden에서 router, expert, residual 출력을 비교한다. 학습 검사는 한 step delta를 비교한다. 구조와 값이 같아도 backend tie-break와 reduction order 때문에 exact function이 달라질 수 있고, 함수가 같아도 optimizer state 누락 때문에 다음 delta가 달라질 수 있다.

부분 checkpoint failure를 주입한다. 한 expert shard를 이전 CheckpointID 파일로 바꾸고 manifest가 거부하는지 본다. router bias 파일 하나를 빼고 strict load가 실패하는지 본다. optimizer moment의 expert order만 permutation해 one-step delta gate가 잡는지 본다. 완료 marker 전에 process를 죽여 incomplete snapshot이 선택되지 않는지 본다. failure를 만들지 않으면 복구 경로가 정상 경로의 우연한 변형인지 알 수 없다.

EP 크기 변경은 portability 수준을 선언해야 한다. global expert weight와 mapping을 정확히 보존하는 것은 필수다. floating-point collective 순서가 바뀌어 bitwise equality가 깨질 수 있다. capacity domain이나 tie allocation이 rank partition에 의존하면 routing trajectory도 달라질 수 있다. exact, tolerance-bounded, statistical continuation 중 어느 계약인지 정하고 그에 맞는 golden gate를 둔다.

복구 승인 문서는 source revision, checkpoint ID, 원래와 새 topology, converter revision, missing/unexpected keys, expert별 digest, routing fixture, one-step delta, 미실행 hardware path를 포함한다. “로드 성공”은 API가 파일을 읽었다는 뜻일 뿐이다. 모델 함수와 학습 state의 복원이 증명되려면 네 층 검사가 모두 근거를 가져야 한다.

## 9.14 failure를 최초 tensor·state·collective로 분류한다

마지막 loss나 NCCL timeout에서 거꾸로 추측하지 않는다. residual, router, assignment, dispatch, expert output과 combine을 왼쪽부터 비교하고 최초로 깨진 invariant의 owner에게 incident를 돌린다.

NaN이 보이면 loss에서 거꾸로 막연히 추적하지 않고 고정 canary token의 최초 non-finite tensor를 찾는다. residual 입력, norm 출력, router logits, selected weights, expert input, gate/up preactivation, product, down output, weighted combine, residual output 순서로 검사한다. aux와 z-loss reduction은 main path와 별도다. 최초 NaN이 router z-loss인데 main output은 finite일 수 있고, 특정 rare expert만 NaN이면 batch 평균에서 늦게 드러날 수 있다.

router logits가 finite인데 selected weight가 NaN이면 softmax 안정화, mask의 `-inf`, all-masked 후보를 본다. selected까지 finite인데 expert input이 깨지면 dispatch permutation, quantization scale, buffer lifetime을 본다. 특정 expert activation에서 시작하면 weight, scale, outlier와 kernel을 eager reference에 비교한다. combine에서 처음 생기면 잘못된 offset, 중복 write, FP16 accumulation overflow를 의심한다. residual 이후라면 fused add-norm과 dtype을 본다.

loss plateau는 값이 finite라 더 어렵다. requested/accepted/dropped, fully dropped token, router entropy, expert별 effective tokens, gradient, optimizer delta를 한 window로 본다. routed gradient가 있는데 delta가 0이면 optimizer group 또는 overflow skip 문제다. router가 균등한데 validation이 정체하면 balance가 품질을 보장하지 않는 반례다. shared expert만 충분히 학습해 loss를 낮추고 routed experts가 죽어 있을 수도 있다.

throughput 하락은 control plane과 data plane을 나눈다. routing histogram과 accepted count가 바뀌면 workload 변화다. logical assignments가 같고 send matrix가 달라지면 placement 또는 dispatcher 변화다. 둘 다 같고 grouped GEMM 시간이 늘면 kernel, clock, workspace를 본다. compute도 같고 exposed collective tail이 늘면 network 또는 overlap event를 본다. tokens/sec 한 숫자는 이 네 원인을 구별하지 못한다.

hang에서는 마지막 완료 collective 이름보다 모든 rank의 다음 예정 sequence를 비교한다. count mismatch, process group membership, conditional empty-rank skip, TP와 EP collective order 교차, stream event 미기록을 본다. timeout이 process를 죽이기 전에 rank별 sequence number, send/recv counts, buffer readiness, current LayerID를 durable artifact로 남긴다. 한 rank stack trace만으로 global state machine을 복원할 수 없다.

checkpoint resume divergence는 load 직후 단계별로 좁힌다. 동일 hidden에 대한 router logits, selected IDs, accepted ledger, expert output, residual output이 같은지 본다. 모두 같으면 동일 batch의 gradients와 clip coefficient, overflow flag를 비교한다. 그것도 같으면 moments와 first delta를 본다. 첫 차이가 data input이면 sampler와 cursor 문제다. 이 순서는 random seed를 반복 변경하며 증상을 희석하는 일을 막는다.

반례와 복구는 한 쌍이어야 한다. expert permutation을 주입했다면 global-ID manifest로 거부하고 올바른 reshard 뒤 fixture가 회복되는지 확인한다. stale routing을 주입했다면 saved assignment digest detector가 실패하고 recompute 정책 수정 뒤 통과해야 한다. zero-token rank skip을 주입했다면 collective sequence detector가 잡고 unconditional participation 뒤 stress가 통과해야 한다. detector만 울리는 것은 복구 완료가 아니다.

독립 검토자는 failure 이름 대신 `first_bad_tensor`, `producer_function`, `state_transition`, `affected_identity`, `recovery_action`, `post_recovery_gate`를 요구한다. 이 여섯 칸이 채워지면 NaN, plateau, hang, corruption을 같은 언어로 비교할 수 있다. 근거가 없는 원인은 가설로 남기고 production incident처럼 서술하지 않는다.

**세 장을 관통하는 변경 영향표를 완성한다**

9장의 변경은 10장 모델 해부, 15장 병렬 ownership, 17장 checkpoint 복구로 이어진다. 예를 들어 `top_k` 변경은 9장에서 selected shape와 combine 함수를, 10장에서 block output과 loss 경로를, 15장에서 assignment 수와 all-to-all bytes를, 17장에서 config compatibility와 첫 update trajectory를 바꾼다. 한 장에서만 “성능 옵션”으로 기록하면 나머지 장의 invariant가 모순된다.

`intermediate_size` 변경은 dense와 expert projection shape, parameter 수, optimizer state, tensor parallel divisibility, kernel tile, checkpoint schema를 바꾼다. 기존 checkpoint를 부분 load하면 새 좌표의 초기화 정책이 필요하다. 단순히 더 큰 MLP가 품질을 높인다고 쓰지 않고 activated parameter와 total parameter, memory, FLOP, convergence를 새 실험으로 측정한다. 이전 run과 같은 RunRevision으로 묶지 않는다.

`capacity_factor` 변경은 capacity source 식의 output, accepted ledger, drop 분포, grouped GEMM rows, communication bytes, residual-only token 비율을 바꿀 수 있다. drop이 없는 구간에서는 logical output이 같을 수 있지만 buffer reservation과 padding FLOP는 달라진다. 경계가 있는 batch에서는 objective 자체가 달라진다. 두 구간을 따로 시험해 performance-only 주장 범위를 제한한다.

EP size 변경은 모델의 global expert 함수가 유지되어야 하지만 owner rank, local slot, send matrix, collective group, checkpoint shard가 바뀐다. capacity와 tie-break가 global invariant라면 routing ledger도 유지할 수 있다. local token order나 atomic allocation에 의존하면 exact assignment가 달라질 수 있다. 15장은 rank map을, 17장은 reshard mapping을, 9장은 golden expert identity와 output을 같은 manifest에서 읽는다.

fused kernel 변경은 config와 parameter가 같아도 saved tensor, compute dtype, approximation, workspace, stream event를 바꾼다. 9장의 FP64/eager oracle과 gradient gate를 먼저 통과하고 10장에서 full block drift를, 15장에서 shard와 collective 경계를, 17장에서 resume 후 kernel selection 재현성을 확인한다. checkpoint format이 같다는 이유로 실행 함수가 같다고 단정하지 않는다.

residual mixing 변경은 MLP/MoE 이전 tensor가 유지되고 branch output 이후 mixing부터 갈려야 한다. stream 수가 늘면 activation state와 checkpoint parameter, pipeline boundary payload도 변할 수 있다. mixing constraint iteration을 바꾸는 것은 numerical 함수 변경이며 latency와 안정성 trade-off를 새 revision에서 측정한다. ordinary residual로 fallback하면 복구 모드이지 동일 모델 복원이 아니다.

router controller 변경은 selection bias state, count reduction group, checkpoint buffer, routing trajectory를 바꾼다. auxiliary coefficient 변경은 loss scalar와 gradient를 바꾸지만 parameter shape는 유지할 수 있다. 둘을 같은 “balancing 조정”으로 묶지 않는다. 전자는 autograd 밖 state transition일 수 있고 후자는 objective 항이다. source 함수, mutable state, distributed reduction, recovery 요구가 서로 다르다.

최종 영향표의 각 행은 `config/source 좌표 → producer 함수 → 최초 변경 tensor/state → downstream collective/owner → checkpoint key/state → 기대 효과 → 반례 → 복구 gate` 순서를 가진다. 어느 칸도 “자동 처리”로 채우지 않는다. 모르는 칸은 `UNVERIFIED`로 남기고 owner와 필요한 fixture를 적는다. 이 표가 10·15·17장에서 동일 ModelRevision을 가리킬 때 장간 연결이 닫힌다.

**출판 전 재현 가능한 심화 실습**

실습 A는 dense SwiGLU다. `C=4`, `F=6`, token 두 개, 비대칭 gate/up/down weight를 FP64로 만든다. source projection 순서대로 forward와 analytic backward를 계산하고 finite difference로 세 weight와 input gradient를 확인한다. gate/up swap, split 축 swap, SiLU를 ReLU로 변경하는 세 failure를 주입한다. 각 failure의 최초 차이가 각각 projection 의미, split 결과, activation output인지 기록한다.

실습 B는 single-process MoE다. `N=6`, `E=3`, `k=2`로 두고 expert마다 식별 가능한 변환을 준다. router logits을 직접 정해 정상, tie, skew, fully dropped token을 만든다. selected, accepted, dropped 원장을 손으로 계산하고 dispatch, expert output, weighted combine을 reference와 비교한다. router 열과 expert를 함께 permutation했을 때 output은 유지되고 identity manifest는 바뀌는지 본다.

실습 C는 두 rank EP다. global expert 네 개를 rank마다 두 개씩 소유하게 하고 remote와 local assignment를 섞는다. send/recv matrix, packed offsets, reverse mapping을 계산한다. metadata 두 row permutation, count 하나 손상, rank 하나 zero-token의 세 failure를 주입한다. conservation detector, collective sequence detector, identity digest가 예상 최초 지점에서 각각 반응해야 한다.

실습 D는 grouped GEMM이다. expert token 수를 `[0,1,7,16]`으로 만들고 eager loop와 grouped kernel의 forward/backward를 비교한다. padding on/off에서 logical output이 같은지, useful와 executed FLOP가 어떻게 다른지 센다. overlap on/off에서 동일 digest를 확인하고 buffer 재사용 event 하나를 제거한 stress failure가 race detector 또는 parity gate에 잡히는지 본다.

실습 E는 residual depth다. 동일 branch를 여러 layer 반복해 pre-norm, post-norm, parallel, sequential의 식을 따로 구현한다. branch zero fixture로 identity gradient를 확인하고 BF16 residual과 FP32 residual의 깊이별 drift를 측정한다. fully dropped MoE output을 넣어 residual이 finite를 유지해도 routed contribution detector가 실패하는지 본다.

실습 F는 checkpoint reshard다. EP 2에서 global expert별 고유 weight와 Adam moments, router columns, balance bias를 저장하고 EP 4로 변환한다. 구조, 값, 함수, one-step delta 네 gate를 순서대로 실행한다. moment만 permutation한 failure는 앞 세 gate를 통과하고 마지막 gate에서 실패해야 한다. 이 결과로 각 gate가 필요한 이유를 확인한다.

실습 G는 통합 변경이다. 같은 golden batch에서 top-k 2→3, capacity factor 변경, dispatcher backend 변경, fused MLP 변경을 하나씩 적용한다. 변경마다 유지되어야 할 마지막 tensor와 달라져야 할 첫 tensor를 미리 쓴다. 실행 뒤 first-difference가 예상 경계와 다르면 효과 분석을 중단하고 configuration 또는 source 선택부터 재검토한다.

모든 실습 artifact는 source revision, config, seed, dtype, device, topology, input digest, expected invariant, actual result, `PASSED/FAILED/NOT_RUN`을 포함한다. GPU나 multi-node 환경이 없어 실행하지 못한 항목은 oracle, 입력, trace point를 남기되 통과로 세지 않는다. 독자가 다른 backend에서 같은 실습을 재실행할 수 있어야 한다.

**9장의 최종 계약을 다시 봉인한다**

최종 계약의 첫 축은 함수다. dense MLP는 gate/up/down projection과 activation, expert MLP는 같은 함수의 global-ID별 instance, router는 candidate에서 accepted assignment로 가는 상태 기계, residual은 branch output을 깊이 방향 state에 합치는 규칙이다. 각 함수는 고정 source 좌표, logical equation, 작은 oracle, backward gate를 가진다.

둘째 축은 tensor와 mutable state다. hidden, logits, selected IDs와 weights, assignment ledger, packed buffers, expert offsets, combined output, residual output, gradients, optimizer moments, controller bias를 producer와 consumer에 연결한다. ephemeral tensor와 durable checkpoint state를 나누고, 재계산 가능한 값도 RNG와 source revision 조건을 적는다.

셋째 축은 분산 ownership이다. token source rank, global expert owner, local expert slot, TP shard, DP replica, collective group을 한 rank map에 둔다. count와 payload, forward와 backward, dispatch와 combine이 같은 assignment identity를 사용한다. empty rank와 zero-token expert를 정상 state로 포함하고 conditional collective skip을 금지한다.

넷째 축은 checkpoint와 복구다. global expert weight, router column, optimizer moment, controller state가 같은 ID로 이동한다. 저장은 committed UpdateID에서 atomic하게 닫히고 load는 구조, 값, 함수, one-step delta 네 층으로 검증한다. incomplete 또는 mixed checkpoint는 완료 marker와 digest에서 거부한다.

다섯째 축은 효과와 failure다. 품질, 균형, useful FLOP, executed FLOP, network bytes, memory high-water, latency를 별도 지표로 둔다. 첫 bad tensor와 state transition으로 NaN, plateau, throughput, hang, divergence를 분류한다. 효과 주장은 반례와 recovery gate를 동반하고 실행하지 않은 경로는 `NOT_RUN`으로 남긴다.

9장의 독립 재현 시험은 token 하나를 두 방향으로 왕복한다. forward에서는 source residual에서 router, accepted expert, destination buffer, expert output, weighted combine, residual output까지 간다. backward에서는 loss gradient에서 combine weight, expert output gradient, expert parameter와 input gradient, router gradient, residual identity gradient까지 돌아온다. optimizer와 checkpoint를 거쳐 다음 step의 같은 global identity로 이어진다.

이 왕복이 닫히면 10장은 layer 전체에서 attention과 MLP/MoE가 결합되는 실제 호출 graph를 해부한다. 15장은 동일 ledger를 rank ownership과 collective 비용으로 확장한다. 17장은 동일 global ID를 checkpoint commit과 reshard 복구에 사용한다. 세 장의 manifest가 같은 ModelRevision, UpdateID, LayerID를 가리키지 않으면 어느 한 장의 부분 성공으로 전체 학습 stack을 승인하지 않는다.

최종 서명에는 고정 source body fingerprint, tested config와 dtype, topology, oracle 결과, failure injection 결과, checkpoint round trip, 미검증 범위와 재검토 조건을 적는다. 이 서명은 특정 architecture 이름을 외웠다는 증명이 아니다. 새 MLP, 새 router, 새 dispatcher가 와도 좌표에서 함수로, 함수에서 tensor/state로, state에서 분산과 checkpoint로, 다시 효과와 failure와 복구로 이어지는 추론 사슬을 재사용할 수 있다는 증명이다.

### dense·gated MLP의 shape·FLOPs·saved tensor를 비교한다

hidden input을 `X ∈ R^{B×S×H}`라 하고 token 축을 `T=B·S`로 펴면 일반 MLP는 `U=XW_up`, `A=φ(U)`, `Y=AW_down`이다. `W_up`은 `H×I`, `W_down`은 `I×H`다. bias를 빼고 multiply-add를 두 FLOP으로 세면 forward projection 비용은 약 `4THI`이며 activation 비용은 별도다. backward는 input gradient와 두 weight gradient 때문에 대략 forward projection의 두 배를 추가하지만 kernel fusion과 recompute 여부에 따라 실제 bytes와 latency는 달라진다.

SwiGLU는 `G=XW_gate`, `U=XW_up`, `A=silu(G)⊙U`, `Y=AW_down`이고 GeGLU는 gate activation이 GELU다. 두 입력 projection 때문에 같은 intermediate width에서 projection FLOPs와 parameter가 일반 MLP보다 늘어난다. 모델이 parameter budget을 맞추려고 gated intermediate width를 줄일 수 있으므로 architecture config의 실제 `intermediate_size`를 읽어야 한다. 이름만 보고 4H 또는 8H/3을 대입하지 않는다.

backward에 필요한 saved tensor도 다르다. SwiGLU는 `G`, `U` 또는 재계산 가능한 `X`와 weight가 필요하고 `dG`에는 SiLU derivative와 `U`가, `dU`에는 activated gate가 들어간다. activation checkpointing이 projection output을 저장하지 않으면 backward에서 gate/up GEMM을 재실행한다. memory 절감량과 recompute FLOPs를 profiler의 allocated bytes와 kernel trace로 확인한다.

shape ledger는 batch, sequence, hidden, intermediate, dtype, stride, contiguous 여부와 tensor-parallel shard를 각 함수 경계에 기록한다. fused kernel이 gate와 up을 하나의 `2I` 출력으로 만들면 split order가 `[gate, up]`인지 `[up, gate]`인지 source와 tiny fixture로 확인한다. 두 절반의 shape가 같아서 잘못 뒤집혀도 즉시 crash하지 않고 품질만 무너질 수 있다.

### residual·dropout·norm의 RNG와 backward 경계를 고정한다

pre-norm block의 MLP 경로는 보통 `R'=R+Dropout(MLP(Norm(R)))`이고 post-norm은 `R'=Norm(R+Dropout(MLP(R)))`다. parallel residual은 attention과 MLP가 같은 normalized input을 받아 함께 더해지고 sequential residual은 attention 결과가 MLP 입력에 영향을 준다. 수식의 괄호가 checkpoint 호환성과 gradient path를 결정하므로 config flag보다 실제 forward 순서를 기록한다.

dropout mask는 branch output shape, RNG stream, device와 UpdateID에 연결한다. tensor parallel에서 동일 mask가 필요한 구간과 rank별 독립 mask가 필요한 구간을 구분한다. activation checkpoint recompute가 forward와 다른 RNG state를 쓰면 gradient가 달라진다. dropout 0 fixture만으로 이 경로를 승인하지 않고 nonzero probability에서 uninterrupted/recompute parity를 확인한다.

residual stream을 FP32로 유지하는지 input dtype으로 되돌리는지, norm 통계가 어느 dtype에서 계산되는지도 경계 상태다. BF16 branch가 finite여도 깊은 layer에서 residual drift가 커질 수 있다. branch를 0으로 만든 fixture에서 pre-norm residual output과 identity gradient를 검사하고, norm scale을 1·bias를 0으로 둔 작은 vector를 손계산한다.

MoE branch에서 모든 assignment가 drop되거나 expert output이 0이어도 residual 때문에 output은 정상처럼 보일 수 있다. residual finite 검사를 routed contribution 검사로 대신하지 않는다. layer별 branch norm, residual-to-branch ratio, accepted token count와 gradient norm을 함께 관측한다.

**router softmax·top-k·capacity를 assignment ledger로 펼친다**

router logits는 `L=XW_r`이고 expert 수를 E라 하면 shape는 `T×E`다. softmax probability `P`에서 token별 top-k expert를 고르되 softmax를 전체 E에 먼저 적용하는지 top-k logits에 다시 normalization하는지 구현에 따라 combine weight가 달라진다. selected ID와 pre/post-normalization weight를 둘 다 기록한다. tie-breaking과 dtype도 경계 근처 assignment를 바꿀 수 있다.

capacity를 token-choice 방식에서 expert당 `C=ceil(capacity_factor·T·k/E)`로 정할 수 있지만 padding token, expert-parallel group 내부 token 수와 minimum capacity 적용 여부를 확인한다. 각 candidate는 TokenID, ExpertID, router weight, priority, source rank를 가진다. capacity sorting 뒤 accepted 또는 dropped 상태와 이유를 붙인다. accepted+dropped candidate 수가 valid token·k와 같아야 한다.

dropless routing은 capacity drop이 없다는 뜻이지 무한 자원이 있다는 뜻이 아니다. expert별 token count가 치우치면 packed buffer, grouped GEMM workspace와 all-to-all receive size가 커진다. admission 또는 dynamic buffer가 최대 count를 처리해야 한다. capacity routing에서는 drop policy가 낮은 probability 우선인지 position 우선인지 random인지가 품질과 bias를 바꾼다.

router z-loss는 보통 token별 `logsumexp(logits)^2`를 줄여 logit magnitude를 억제한다. load-balancing auxiliary loss는 mean routing probability와 assignment fraction의 곱을 사용하지만 top-1/top-k, token mask와 scaling convention이 구현마다 다르다. main loss에 더하는 coefficient와 global denominator를 명시하고 분산 rank의 local mean 평균 오류를 피한다.

**load balance를 auxiliary loss와 controller state로 나눈다**

auxiliary-loss balancing은 router gradient에 직접 압력을 주므로 task objective와 경쟁한다. coefficient sweep에서 expert utilization만 보지 않고 main loss, router entropy, assignment churn과 downstream 품질을 함께 본다. padding과 dropped token을 fraction denominator에 넣는지 tiny batch로 확인한다. 한 rank에 valid token이 없을 때도 전역 statistic이 올바르게 reduce되어야 한다.

auxiliary-loss-free 방식은 expert별 load에 따라 routing bias를 별도 controller state로 갱신할 수 있다. bias가 forward 선택에 들어가지만 gradient로 학습되지 않는다면 update rule, step, clipping, smoothing과 synchronization owner를 checkpoint해야 한다. 동일 weight를 load해도 balance bias가 초기화되면 다음 assignment가 달라진다.

shared expert가 있는 구조에서는 routed experts의 capacity와 shared path 비용을 분리한다. shared expert는 모든 token을 처리하는 dense-like path일 수 있고 routed output과 합쳐지는 scale 또는 gate가 있을 수 있다. shared path가 강해 routed experts collapse를 가리는지 routed contribution, shared contribution과 각각의 gradient를 본다.

collapse detector는 token count 한 시점만 보지 않는다. expert별 assignment fraction, probability mass, accepted fraction, output norm, gradient norm과 optimizer update norm을 window로 추적한다. 항상 선택되지만 output이 0인 expert와 선택되지 않는 expert는 다른 장애다. entropy가 높아도 capacity sorting 뒤 accepted load가 치우칠 수 있다.

**expert parallel all-to-all을 permutation 보존 문제로 검증한다**

dispatch 전에 local tokens의 accepted assignment를 destination rank와 local expert slot 기준으로 정렬한다. packed payload와 함께 source TokenID, assignment slot, combine weight와 reverse index를 보낸다. count exchange가 payload all-to-all보다 먼저 완료되어 receive offsets를 결정한다. zero-count rank도 collective sequence에 참여한다.

destination에서는 local expert별 offsets로 grouped GEMM을 실행하고 결과를 원 source rank로 되돌린다. combine 단계는 reverse index로 같은 TokenID의 k개 결과를 모아 router weight를 적용한다. dispatch와 return permutation이 서로 다른 metadata revision을 쓰면 shape는 맞아도 다른 token의 결과를 섞는다. payload digest보다 `(TokenID, ExpertID, slot)` identity digest가 필요하다.

accepted assignment 수가 A, hidden이 H, element bytes가 b이면 방향당 통신 payload는 대략 `A·H·b`이고 forward dispatch/return과 backward 두 방향이 반복된다. metadata, count exchange와 network padding을 별도로 센다. useful expert FLOPs가 같아도 imbalance가 심하면 max receive rank가 latency를 결정한다.

failure injection은 count 하나 감소, payload row swap, reverse index 중복, wrong process group, rank 하나의 conditional collective skip과 buffer reuse event 제거를 포함한다. count conservation, identity uniqueness, collective sequence, bounds와 end-to-end dense oracle가 각각 예상 단계에서 실패해야 한다. timeout만으로 metadata 오류를 분류하지 않는다.

**DeepSeek·Qwen·Mixtral 구현 차이를 함수 좌표로 해석한다**

architecture 이름은 실행 계약을 대신하지 않는다. DeepSeek 계열에서 routed/shared expert 결합, group-limited routing, correction bias 또는 top-k normalization이 어느 revision에 존재하는지 source body를 확인한다. Qwen MoE 계열의 shared expert와 gate, expert intermediate width, normalization flag를 config와 forward에서 연결한다. Mixtral 계열은 router logits, top-k weight normalization과 expert loop 또는 fused path의 실제 함수 좌표를 고정한다.

Transformers 구현에서는 model config, sparse block forward, router output shape, expert module list와 auxiliary-loss return 경로를 따라간다. `output_router_logits` 같은 flag가 학습 loss에 실제로 연결되는지 확인한다. load balancing 함수가 attention mask와 top-k를 어떻게 받는지, generation 또는 gradient checkpointing에서 반환 tuple 순서가 달라지는지도 시험한다.

Megatron 계열에서는 router, token dispatcher, all-to-all/all-gather 선택, grouped MLP, expert tensor parallel과 auxiliary loss tracker를 call graph로 잇는다. sequence parallel과 expert parallel group이 다른 collective group을 쓸 수 있다. config dump의 숫자만 보지 않고 rank map과 실제 communicator membership을 기록한다.

동일 checkpoint를 서로 다른 구현으로 옮길 때 expert ID, gate/up projection order, router column, shared expert, bias/controller, tensor-parallel shard와 activation convention을 mapping table로 만든다. key rename만 성공한 변환을 승인하지 않고 tiny tensor forward, router assignment, expert output, backward와 one-step optimizer delta를 비교한다.

**expert optimizer와 checkpoint를 global ExpertID로 봉인한다**

optimizer state는 local module index가 아니라 global ExpertID와 parameter ID에 연결한다. expert parallel degree가 바뀌면 weight, master weight, first/second moment와 step state가 새 owner로 함께 이동해야 한다. router column과 expert weight의 ID 관계도 유지한다. parameter group별 learning rate나 weight decay가 다르면 group mapping을 manifest에 넣는다.

checkpoint manifest는 layer, global ExpertID, parameter role, global shape, shard range, dtype, owner topology, checksum과 replica를 기록한다. shared expert와 routed expert namespace를 분리한다. balance controller bias, z-loss scheduler 또는 router temperature 같은 mutable state도 저장한다. incomplete generation에서 일부 expert만 최신인 상태를 commit marker와 UpdateID closure로 거부한다.

EP 2→4 또는 4→2 reshard는 global expert set coverage를 먼저 검사하고 각 expert 내부 TP shard를 다시 나눈다. expert 수 자체를 바꾸는 것은 reshard가 아니라 architecture migration이다. empty expert state와 optimizer slot 누락을 정상 0으로 채우지 않는다. load report는 exact, converted, reset과 missing을 field별로 낸다.

one-step parity는 동일 golden token이 같은 expert에 배정되고 forward output, expert/router gradient, optimizer update와 다음 router decision이 맞는지 본다. weight parity만으로 moment permutation을 찾을 수 없다. resume 중 world size가 달라지면 sampler와 RNG까지 연결해 같은 assignment 입력을 재현한다.

**expert quantization을 routing과 collective 경계에서 검증한다**

expert weight를 FP8 또는 INT8/INT4로 저장하거나 실행할 때 scale granularity가 tensor, channel, group 중 무엇인지 기록한다. gate/up/down projection이 서로 다른 scale을 가지며 fused layout에서 scale 배열 순서가 weight split과 맞아야 한다. router는 높은 정밀도로 남겨도 expert output 오차가 combine weight와 residual을 통해 누적된다.

expert별 calibration token 수가 imbalance 때문에 크게 다를 수 있다. 자주 선택된 expert만으로 scale을 정하면 rare expert의 activation range를 놓친다. global ExpertID별 calibration coverage, activation percentile와 saturation을 기록하고 synthetic routing으로 모든 expert를 강제 방문한다. shared expert와 routed expert를 같은 calibration 통계로 합치지 않는다.

quantized payload를 all-to-all 전후 어느 지점에서 dequantize하는지에 따라 network bytes와 kernel 계약이 달라진다. activation quantization scale을 token과 함께 보내야 한다면 permutation metadata가 scale row에도 동일하게 적용되어야 한다. row swap failure가 hidden payload와 scale의 identity digest에서 잡히는지 시험한다.

quantized checkpoint는 packed bytes뿐 아니라 quantization scheme, scale/zero-point, group size, original shape, padding과 kernel revision을 보존한다. 다른 backend가 scheme을 지원하지 않으면 silent dequantized load가 아니라 명시적 conversion artifact를 만든다. conversion 뒤 router assignment, expert output와 one-step delta 허용 오차를 따로 정한다.

### collapse·drop·dispatch failure matrix로 원인을 격리한다

장애 행렬의 행은 token이 MoE를 통과하는 시간 순서를 따른다. router logits와 top-k 선택에서 시작해 capacity sorting, dispatch plan, count 교환, payload all-to-all, expert 계산, 반환과 combine, residual로 이어지고, 뒤에는 backward·optimizer·checkpoint를 둔다. 열에는 `NaN`/`Inf`, 동점, 극단 logit, all-to-one collapse, token을 받지 못한 expert, capacity overflow와 drop, 오래된 permutation, count 불일치, rank 소실, moment 교환과 quantization saturation을 놓는다. 각 셀에는 최초 detector, 깨지는 invariant, 상태 owner와 recovery를 함께 기록해야 “MoE가 불안정하다”는 보고를 조사 가능한 한 경계로 좁힐 수 있다.

각 칸에는 최초 bad tensor, detector, expected action, recoverability와 artifact를 적는다. router NaN은 expert kernel 실패로 분류하지 않고 finite check에서 중단한다. capacity drop은 policy 안의 예상 상태일 수 있지만 drop rate SLO를 넘으면 release failure다. dispatch mismatch는 retry 전에 generation과 assignment ledger를 폐기해 stale buffer가 combine에 들어오지 않게 한다.

golden run은 dense MLP, single-expert MoE, 균등 top-k, 극단 imbalance, zero-token rank와 shared-expert 조합을 포함한다. eager reference와 fused/distributed path의 forward, backward와 one-step update를 비교한다. 성능 run은 useful/executed FLOPs, dispatch bytes, padding, buffer high-water, expert p50/p99 tokens와 end-to-end latency를 같은 trace에 둔다.

failure recovery 뒤에는 process 생존보다 다음 UpdateID의 의미를 확인한다. 선택된 checkpoint의 global expert coverage, router/controller state, sampler/RNG, optimizer moment를 검증하고 첫 assignment와 parameter delta를 uninterrupted oracle과 비교한다. degraded fallback으로 dense 또는 shared-only 경로를 사용했다면 새 branch와 명시적 품질 gate가 필요하다.

최종 certificate는 architecture/config digest, 함수 좌표, shape/FLOPs ledger, residual/RNG 경계, routing equation과 denominator, assignment permutation, process groups, expert optimizer/checkpoint, quantization state, 장애 행렬과 recovery oracle를 잇는다. 독립 reviewer가 동일 artifact로 token의 forward/backward 왕복과 다음 update를 재구성할 수 있을 때 MoE 운영 계약이 닫힌다.

### router·auxiliary·z-loss gradient를 작은 tensor로 검산한다

router 시험은 top-k ID만 비교하지 않고 selected weight에서 logits로 돌아가는 gradient를 계산한다. expert output을 서로 다른 상수 vector로 두면 combine output의 변화가 router weight에 어떻게 전달되는지 손으로 구할 수 있다. 전체 softmax 뒤 top-k를 고르는 구현과 selected logits만 재정규화하는 구현은 선택 ID가 같아도 선택되지 않은 expert logit의 gradient가 다르다. expected gradient를 convention별로 따로 적는다.

hard top-k 경계는 선택 ID 자체에 미분하지 않으므로 작은 logit perturbation이 경계를 넘을 때 불연속이 생긴다. tie와 거의 같은 logit fixture에서 dtype을 FP32, BF16로 바꾸어 assignment churn을 측정한다. deterministic tie-break가 필요한 재현 시험과 stochastic exploration을 허용하는 학습을 분리하고 RNG 또는 stable ordering 조건을 manifest에 남긴다.

main loss, load-balancing loss와 z-loss는 scalar 하나로 합치기 전 각각의 raw sum, denominator, coefficient와 weighted contribution을 기록한다. valid token 수가 다른 microbatch를 두 개 만들고 accumulation 전체의 global numerator/denominator로 계산한 결과를 single batch 결과와 비교한다. rank별 local mean 평균, padding 포함, top-k candidate를 token처럼 중복 계산하는 오류가 fixture에서 드러나야 한다.

z-loss만 켠 fixture에서는 expert output을 loss에서 끊고 router logit magnitude가 줄어드는 방향의 gradient를 확인한다. balance loss만 켠 fixture에서는 한 expert로 치우친 assignment가 완화되는 방향인지 보되 실제 top-k 불연속 때문에 한 step에서 count가 반드시 개선된다고 가정하지 않는다. controller bias 방식이면 gradient 대신 update rule의 sign, clipping과 전역 count synchronization을 검산한다.

최종 수치표는 logits, probability, selected IDs, pre/post-normalized weight, accepted mask, expert count, main/balance/z numerator와 router gradient를 한 TokenID로 연결한다. eager와 fused router, 단일 rank와 분산 reducer, 저장 전과 resume 후 첫 step이 같은 표를 재생성해야 한다. 이 표가 일치해야 높은 utilization이나 낮은 loss가 올바른 routing 수학에서 나온 결과라고 판정한다.

## 9.15 구현 심화에서 test·release·10장 handoff까지 닫는다

마지막 절은 framework와 kernel별 세부를 owner schema에 연결한다. eager oracle, EP fixture, checkpoint round trip과 production metric이 합의할 때만 모델 해부 dossier를 10장으로 넘긴다.

**같은 비선형 함수라는 말이 감추는 gradient 모양을 복원한다**

정규분포의 누적분포함수를 이용하는 정확한 GELU는 `GELU(x)=xΦ(x)`이고 미분은 `Φ(x)+xφ(x)`다. 여기서 `φ`는 표준정규분포의 밀도다. 큰 양수에서는 미분이 1에 가까워지고 큰 음수에서는 0에 가까워지지만, 0 부근에서는 단순 ReLU와 달리 음수 입력도 매끄럽게 통과시킨다. 실제 구현은 `tanh` 근사 `0.5x[1+tanh(√(2/π)(x+0.044715x³))]`를 사용할 수 있다. 정확식과 근사식은 이름이 같아도 forward와 derivative가 조금 다르므로 checkpoint 이식에서 activation 문자열만 맞추고 끝내지 않는다.

SiLU는 `silu(x)=xσ(x)`이고 미분은 `σ(x)+xσ(x)(1-σ(x))`다. 음수 구간에서 함수값과 미분이 모두 0으로 즉시 잘리지 않으며, 약한 음의 골짜기가 생긴다. SwiGLU의 gate gradient는 `dG=dA⊙U⊙silu'(G)`이므로 up branch `U`의 부호와 크기가 gate 학습을 직접 조절한다. 반대로 `dU=dA⊙silu(G)`다. gate가 강하게 음수로 포화되면 up projection은 존재해도 gradient가 거의 전달되지 않는다. 따라서 activation histogram만 아니라 gate derivative, `U`, 두 projection의 gradient norm을 함께 봐야 한다.

GeGLU에서는 `dG=dA⊙U⊙gelu'(G)`가 된다. GELU exact, tanh approximate, `gelu_new`, `quick_gelu` 같은 이름이 같은 derivative를 보장하지 않는다. source에서 activation registry가 어떤 callable을 반환하는지, fused kernel이 어느 approximation flag를 받는지 확인한다. PyTorch reference를 만들 때도 동일 approximation을 명시한다. 허용 오차를 넓혀 서로 다른 함수를 같은 것으로 판정하면 깊은 layer에서 누적되는 drift를 숨긴다.

수치 검산은 `{-20,-5,-1,0,1,5,20}`과 0 부근의 작은 간격을 쓴다. analytic derivative, FP64 finite difference, framework autograd와 fused backward를 비교한다. BF16 finite difference는 반올림 때문에 oracle로 부적절하다. 먼저 FP64 식을 기준으로 삼고 FP32 reference, BF16/FP16 kernel 순서로 허용 오차를 정한다. 극단 입력에서는 `exp`나 `tanh`가 포화되어도 NaN이 없어야 하며, signed zero와 subnormal 처리 차이도 기록한다.

**activation 선택을 품질 취향이 아니라 계산 계약으로 바꾼다.**

activation을 바꾸면 함수만 바뀌는 것이 아니다. gated MLP에서 intermediate width, parameter budget, 초기화 scale, fused kernel 지원, saved tensor와 backward recompute가 함께 달라질 수 있다. 같은 파라미터 수를 맞추려면 dense GELU의 `I`와 SwiGLU의 `I_g`가 다르다. bias를 제외한 dense MLP 파라미터는 약 `2HI`, gated MLP는 `3HI_g`이므로 단순한 예산에서는 `I_g≈2I/3`가 된다. 실제 모델은 정렬 배수와 성능 요구 때문에 이 값을 반올림한다.

모델 카드의 activation 이름, config의 `hidden_act`, module 생성자의 callable, forward의 gate/value 순서, compiler가 선택한 kernel을 한 줄로 잇는다. config를 바꿨는데 이미 생성된 module이나 compiled graph가 옛 callable을 유지하는 경우도 있다. 따라서 변경 영향표에는 config parse, module construction, graph capture, checkpoint metadata, export config와 serving loader를 모두 넣는다.

훈련 비교는 parameter count만 맞추거나 FLOPs만 맞추는 두 실험을 분리한다. gated activation은 elementwise multiply를 추가하고 두 입력 projection을 요구하지만 width를 줄이면 총 GEMM 비용이 달라진다. executed FLOPs, HBM bytes, saved activation bytes, tokens/s와 validation loss를 같은 step과 token budget에서 비교한다. kernel 미지원으로 작은 operation이 분절되면 수학적 이점과 시스템 손해가 동시에 나타날 수 있다.

### projection layout과 fused kernel의 실제 bytes를 검산한다

**논리 행렬과 저장 행렬의 전치를 구분한다**

수식에서 `X[T,H] @ W[H,I]`라고 써도 `torch.nn.Linear(H,I)`의 parameter는 보통 `[I,H]`로 저장되고 forward는 입력과 weight transpose를 곱한다. tensor parallel 라이브러리는 weight를 output 또는 input dimension으로 shard하고 fused gate-up은 `[2I,H]` 한 장으로 저장할 수 있다. checkpoint key의 shape만 보고 논리 축을 추정하면 gate/up split과 TP shard를 동시에 뒤집을 수 있다.

각 parameter에는 logical role, stored shape, logical transpose, stride, shard axis, global offset, alignment와 dtype을 기록한다. gate-up fusion에서 local tensor의 앞 절반이 gate인지 up인지, global `[gate_all,up_all]`을 shard한 것인지 rank별 `[gate_r,up_r]`을 이어 붙인 것인지 확인한다. 두 layout은 같은 local shape를 가질 수 있지만 reshard concatenate/split 순서가 다르다.

down projection은 보통 intermediate 축을 TP rank가 나누어 부분 output을 만든 뒤 reduce한다. column-parallel gate/up과 row-parallel down의 조합에서 gate와 up은 동일 partition을 가져야 elementwise 곱이 local하게 닫힌다. 잘못된 partition은 추가 collective를 요구하거나 같은 크기의 다른 channel을 곱한다. module의 partition helper, weight loader, forward collective와 checkpoint converter를 같은 축 표에 놓는다.

**fused kernel의 pointer와 stride 계약을 확인한다.**

fused activation kernel은 gate와 up이 contiguous half인지 interleaved인지, leading dimension과 vector alignment가 무엇인지 가정한다. tensor가 view인지 contiguous copy인지, padding channel이 output에 포함되는지 source에서 확인한다. `view`, `reshape`, `chunk`, `split`, `transpose`, `contiguous` 호출은 단순 문법이 아니라 bytes 해석을 바꾸는 경계다.

작은 fixture는 H=3, I=5처럼 정렬에 불편한 크기를 쓰고 각 weight row에 고유한 수를 넣는다. reference에서 gate, up, activation, product, down output을 인쇄 가능한 표로 만든다. fused path를 강제할 수 없는 작은 shape라면 실제 정렬 shape에서도 특정 channel에 one-hot pattern을 넣어 split 순서를 검증한다. all-one weight는 permutation 오류를 숨기므로 금지한다.

weight loader는 safetensors slice, quantized packed weight, TP rank와 expert local ID를 모두 소비한다. loader가 gate와 up을 별도 key에서 fused destination으로 복사한다면 destination offset 계산을 시험한다. 반대로 fused checkpoint를 분리 module로 읽을 때도 동일하다. copy 뒤 parameter checksum 하나만 비교하지 않고 role별 logical slice digest를 만든다.

**MLP FLOPs보다 HBM traffic이 먼저 병목이 되는 조건을 계산한다**

**arithmetic intensity를 token 수와 weight 재사용으로 해석한다**

GEMM의 peak FLOPs만으로 MLP 속도를 예측할 수 없다. 한 microbatch의 token 행렬 `X[T,H]`가 weight `W[H,I]`를 곱할 때 weight가 cache와 shared resource에서 얼마나 재사용되는지는 T에 달려 있다. T가 작으면 거대한 weight를 적은 token이 나누어 쓰므로 weight bytes가 지배하고, T가 커지면 산술 집약도가 올라간다. MoE에서는 전체 T가 커도 expert별 `T_e`가 작아 작은 GEMM 군으로 다시 분해된다.

간단한 roofline 원장은 연산량 `2THI`, 최소 input/output bytes, weight bytes와 중간 activation bytes를 별도로 센다. gate/up/down 세 GEMM과 activation product 사이에 `G`, `U`, `A`가 HBM으로 왕복하는지 fused epilogue나 recompute로 줄어드는지 확인한다. theoretical bytes와 profiler의 DRAM read/write가 다르면 cache reuse, temporary workspace, layout conversion과 padding을 추적한다.

MoE grouped GEMM은 expert별 작은 연산을 한 launch에 묶어 launch overhead와 scheduling을 줄이지만, 각 expert의 M/N/K와 pointer 배열을 준비해야 한다. token count의 p50이 0 또는 아주 작고 p99만 크면 단일 tile 정책이 비효율적일 수 있다. useful FLOPs는 실제 accepted token만, executed FLOPs는 padding과 tile waste를 포함해 계산한다.

**fusion의 이익과 관측 가능성의 손실을 함께 관리한다.**

gate-up projection fusion, activation fusion, down projection과 reduce-scatter overlap은 HBM traffic과 launch 수를 줄인다. 그러나 중간 tensor가 사라지면 gate saturation, split 오류와 first non-finite 위치를 찾기 어렵다. production kernel을 바꾸기 전에 debug mode에서 중간 checksum, sampled activation과 per-role norm을 노출하는 경로를 마련한다.

benchmark matrix는 T, H, I, dtype, TP degree, expert token 분포, alignment와 fusion flag를 축으로 둔다. warm-up, graph capture, allocator 상태를 통제하고 kernel 시간뿐 아니라 end-to-end layer 시간과 collective wait를 잰다. 작은 synthetic uniform 분포에서 빠른 kernel이 실제 long-tail routing에서는 느릴 수 있다.

최적화 승인은 수치 parity, backward parity, one-step update, peak memory, tokens/s와 tail latency를 모두 요구한다. profiler trace의 kernel 이름 하나가 줄었다고 승인하지 않는다. 15장의 collective 원장과 16장의 CUDA trace를 같은 UpdateID와 microbatch에 연결하면 compute 최적화가 network idle을 늘렸는지 판별할 수 있다.

**tensor parallel MLP의 forward와 backward collective를 대칭으로 푼다**

**column-parallel 입력 projection과 row-parallel 출력 projection을 연결한다**

TP 크기를 P라 하자. gate/up weight의 output channel을 P개로 나누면 각 rank는 `I/P` channel의 gate와 up을 계산한다. activation product도 local하게 닫힌다. down weight는 input channel을 같은 방식으로 나누어 각 rank가 `[T,H]` partial output을 만들고 all-reduce 또는 reduce-scatter로 합친다. sequence parallel 여부에 따라 최종 token 축 ownership이 달라진다.

forward ledger에는 input replicated/sharded 상태, local intermediate range, partial output, collective 종류와 결과 ownership을 적는다. backward에서는 down weight gradient가 local intermediate와 output gradient로 계산되고, intermediate gradient는 down weight shard를 거쳐 local하게 나온다. gate/up weight gradient는 입력이 replicated인지 all-gather되었는지에 따라 통신이 달라진다. input gradient partial은 합쳐져야 한다.

collective 이름을 암기하지 말고 각 tensor의 수학적 합과 원하는 owner로부터 도출한다. 같은 API가 async handle을 반환할 수 있고 downstream stream이 wait해야 한다. forward의 reduce가 끝나기 전에 residual add가 읽거나 backward의 all-reduce buffer가 재사용되면 간헐적 오염이 생긴다.

**sequence parallel과 activation checkpoint의 교차 효과를 검증한다.**

sequence parallel은 token 축을 rank에 나누어 norm과 dropout activation memory를 줄일 수 있다. MLP 입력 projection 전에 all-gather가 필요한지, row-parallel output을 reduce-scatter로 바로 sequence shard에 돌려주는지 구현을 확인한다. dropout RNG가 global token position과 연결되지 않으면 TP/SP degree 변경에서 mask가 달라진다.

activation checkpoint가 MLP 내부를 재계산할 때 forward와 동일한 all-gather/reduce 순서를 다시 실행한다. checkpoint wrapper가 async collective handle이나 sharded view를 저장하면 수명이 꼬일 수 있다. saved tensor hook과 trace로 무엇이 저장되고 무엇이 재계산되는지 확인한다.

parity matrix는 TP 1과 TP 2/4, SP on/off, checkpoint on/off를 교차한다. 동일 global input과 weight에서 forward, input/weight gradient와 one-step update를 비교한다. rank별 local checksum만 아니라 gather한 global logical tensor를 비교한다. 0 token shard가 가능한 packed sequence에서도 collective 순서가 유지되어야 한다.

**router precision은 softmax 정확도보다 결정 경계의 안정성 문제다**

**logit 계산, score 변환, top-k 선택의 dtype을 분리한다**

hidden과 router weight가 BF16이어도 GEMM accumulate, logits 저장, softmax, correction bias addition과 top-k 비교가 같은 dtype일 필요는 없다. 선택 경계에서 logit 차이가 BF16 한 ULP보다 작으면 FP32에서는 다른 순서를 유지하던 두 expert가 같아질 수 있다. stable tie-break가 rank나 kernel에 따라 다르면 재현성이 무너진다.

router pipeline의 각 단계에 input dtype, accumulation dtype, output dtype와 cast 위치를 적는다. softmax는 최대 logit을 빼서 안정화하지만 top-k를 logits에서 먼저 수행하면 전체 softmax overflow와 무관할 수 있다. sigmoid score, softmax score, group-limited selection과 bias-corrected score는 normalization 의미가 다르다.

precision audit은 선택된 ID 일치율만 보지 않는다. margin `score_k-score_{k+1}`, churn rate, combine weight 차이, expert load와 downstream output 오차를 margin bucket별로 본다. 작은 margin token은 본질적으로 민감하므로 절대 100% 일치 주장보다 허용 정책과 deterministic mode를 정의한다.

**분산 rank가 같은 router 결정을 보도록 입력 identity를 고정한다.**

TP rank가 router logits의 일부를 계산해 gather하거나 reduce한다면 collective reduction 순서와 dtype이 결과를 바꿀 수 있다. expert tensor parallel과 data parallel이 겹칠 때 어느 rank가 top-k를 수행하고 결과를 broadcast하는지 확인한다. 모든 rank가 독립 top-k를 수행하면 수치 미세 차이가 collective count 불일치로 증폭될 수 있다.

production에서는 margin p1/p50/p99, non-finite logits, expert별 score drift와 assignment churn을 표본 추적한다. precision 변경, compiler upgrade나 GPU architecture 변경 전후 같은 fixed token corpus로 비교한다. 낮은 margin 비율이 급증하면 router weight scale, norm, temperature와 quantization을 조사한다.

resume parity에서 router RNG jitter가 있다면 RNG counter를 TokenID와 UpdateID에 연결한다. jitter를 끄고 deterministic 수학 parity를 먼저 확인한 뒤 켠 상태에서 distribution parity를 검증한다. 무작위성을 핑계로 count conservation이나 checkpoint state 누락을 허용하지 않는다.

**top-k routing의 gradient를 선택 내부와 선택 경계로 나눈다**

**selected weight normalization의 Jacobian을 손으로 계산한다**

선택된 k개 logits만 softmax로 다시 정규화하면 `w_i=exp(l_i)/Σ_{j∈K}exp(l_j)`이고 Jacobian은 같은 selected set 안에서 `∂w_i/∂l_j=w_i(δ_ij-w_j)`다. expert output `E_i`의 weighted sum `y=Σw_iE_i`라면 `∂y/∂l_j=w_j(E_j-y)`가 된다. expert 출력이 모두 같으면 router weight gradient가 0이라는 중요한 oracle을 얻는다.

전체 E softmax의 probability를 top-k 후 그대로 쓰거나 선택 weight 합으로 다시 나누는 구현은 선택되지 않은 logit gradient가 다르다. loss 계산 코드가 router logits를 별도로 받아 auxiliary term을 더하면 main combine 경로와 aux 경로의 gradient를 분리해야 한다. hook으로 두 contribution을 측정하거나 각각 loss를 끈 fixture를 만든다.

hard selection의 set K는 거의 모든 지점에서 상수로 취급되고 경계에서 불연속이다. autograd graph가 top-k index에 gradient를 주지 않는다는 사실과 router가 학습되지 않는다는 주장은 다르다. 선택된 score의 연속 gradient와 auxiliary/controller 신호가 router를 움직인다. 이를 설명할 때 straight-through estimator를 실제 사용하지 않는 구현에 임의로 끼워 넣지 않는다.

**token drop이 gradient support를 어떻게 바꾸는지 기록한다.**

capacity에서 assignment가 drop되면 해당 expert output 경로의 gradient가 사라지거나 residual만 남는다. combine weight를 drop 뒤 재정규화하는지, 남은 weight 합이 1보다 작은 채로 두는지에 따라 output scale과 router gradient가 달라진다. token 전체 assignment가 drop되는 경우의 fallback을 확인한다.

accepted mask는 forward 임시값이지만 backward 의미를 결정한다. checkpoint recompute에서 capacity sorting tie가 달라져 mask가 바뀌면 잘못된 gradient를 계산한다. deterministic priority 또는 saved assignment metadata가 필요하다. microbatch 순서와 EP rank count가 capacity 결과에 영향을 주는지도 시험한다.

gradient 관측은 router weight norm 하나로 끝내지 않는다. expert column별 gradient, selected/unselected score contribution, dropped token 비율, margin bucket과 token domain을 연결한다. 특정 언어 또는 안전 데이터가 자주 drop되어 gradient support를 잃으면 전체 loss가 정상이어도 specialization이 왜곡된다. 이 지점은 6장의 mixture/curriculum 원장과 27장의 안전 평가를 잇는다.

**load-balancing loss의 분모를 microbatch와 rank를 넘어 보존한다**

**대표식을 구현의 reduction 순서로 펼친다**

top-k routing의 한 형태는 expert e의 평균 probability `p_e`와 assignment fraction `f_e`를 구해 `E·Σ_e p_e f_e`를 사용한다. 그러나 `p_e`가 전체 softmax인지 selected score인지, `f_e`가 candidate인지 accepted assignment인지, k로 나누는지, padding을 제외하는지에 따라 값이 달라진다. 논문 식과 구현 함수의 tensor shape/reduction 축을 나란히 적는다.

microbatch마다 local mean을 구해 단순 평균하면 valid token 수가 다른 경우 global mean과 달라진다. 정확한 accumulation은 numerator와 denominator를 보존해 전체 step에서 나눈다. data-parallel rank도 같은 원리다. rank별 token 수가 다른 packed batch, 마지막 batch와 sequence packing에서 특히 중요하다.

작은 fixture는 rank A에 token 1개, rank B에 3개를 두고 expert score를 비대칭으로 만든다. global concatenation reference와 distributed reduction을 비교한다. padding mask를 뒤집거나 k factor를 누락하면 예상 scalar와 gradient가 모두 실패하도록 수를 고른다.

**balance가 좋아졌다는 주장을 specialization 손실과 함께 평가한다.**

균등 utilization은 목적 그 자체가 아니다. 데이터가 이질적이면 유용한 specialization이 불균등 분포를 만들 수 있다. balance coefficient를 높여 count가 균등해져도 main loss나 rare-domain 성능이 나빠질 수 있다. expert별 domain mixture, token difficulty, output/gradient similarity와 load를 함께 본다.

load skew는 router 문제뿐 아니라 데이터 shard, padding, capacity policy, expert 속도, stale controller와 network placement의 결과일 수 있다. probability mass는 균등한데 accepted count만 치우치면 capacity/dispatch를, count는 균등한데 step tail이 길면 expert compute나 network를 조사한다.

coefficient scheduler가 있다면 scheduler state, global step 정의, resume 위치를 checkpoint한다. gradient accumulation의 microstep을 global step으로 잘못 세면 coefficient가 빨리 변한다. 실험 기록에는 raw loss, coefficient, weighted contribution, numerator/denominator와 router gradient contribution을 남긴다.

### auxiliary-loss-free bias를 측정·갱신·적용 단계로 나눈다

**correction bias의 측정·갱신·적용 시점을 분리한다**

aux-free 방식은 관측된 expert load와 목표 load의 오차로 selection bias를 갱신할 수 있다. 이를 단순한 학습 파라미터로 보면 안 된다. controller에는 측정 window, target, update rate, sign, clipping, smoothing, synchronization과 적용 지연이 있다. 너무 큰 update rate는 routing이 expert 사이를 왕복하는 oscillation을 만들고 너무 작으면 collapse를 늦게 교정한다.

한 step의 순서를 `score 계산→현재 bias 적용→top-k→accepted count 측정→전역 reduce→bias 갱신→다음 step 적용`처럼 고정한다. 구현이 microbatch마다 갱신하는지 optimizer step마다 갱신하는지 확인한다. gradient accumulation 중 bias가 변하면 같은 logical batch의 microbatch가 서로 다른 router를 경험한다.

controller state는 모델 parameter와 다른 synchronization 경로를 가질 수 있다. DP replica마다 local count로 갱신하면 같은 weight가 다른 bias를 갖는다. EP group만 reduce할지 전체 DP/EP world에서 reduce할지는 expert identity와 데이터 ownership에 따라 결정한다. process group을 manifest에 명시한다.

**제어 안정성을 step response와 복구 시험으로 검증한다.**

synthetic load를 한 expert로 갑자기 몰아 controller의 step response를 본다. overshoot, settling time, steady-state error와 bias saturation을 기록한다. 데이터 분포를 원래대로 돌렸을 때 bias가 정상 범위로 회복하는지도 본다. utilization 한 시점만으로 안정성을 판정하지 않는다.

checkpoint를 bias 포함/제외 두 방식으로 resume해 첫 수십 step의 assignment를 비교한다. bias 누락은 loss spike보다 expert reshuffle과 network buffer high-water로 먼저 나타날 수 있다. stale count accumulator, smoothing EMA와 update counter도 저장 대상인지 확인한다.

controller가 selection에만 쓰이고 combine weight에는 쓰이지 않는 구조와 둘 다 쓰이는 구조를 구분한다. correction을 score에 더해 ID를 고른 뒤 원래 probability로 combine할 수 있다. selection score, combine score와 training gradient score를 하나로 뭉개면 수학을 잘못 설명하게 된다.

**expert capacity와 dropless routing의 메모리 상한을 계산한다**

**capacity factor를 품질과 메모리의 공동 knob로 해석한다**

capacity routing에서 expert buffer는 대개 `E_local×C×H` 또는 packed equivalent를 예약한다. capacity factor를 높이면 drop은 줄지만 activation buffer, workspace와 통신 payload 상한이 커진다. 낮추면 메모리를 줄이면서 low-priority assignment를 제거해 objective가 변한다. 따라서 OOM 해결을 위해 값을 낮추는 행위는 순수 시스템 튜닝이 아니다.

valid token T, top-k k, global expert E와 EP degree를 이용한 nominal capacity 식을 시작점으로 삼되 구현의 rounding, minimum capacity, sequence padding과 local/global T를 확인한다. capacity alignment가 tile 또는 communication chunk에 맞춰 올라갈 수 있다. config 값에서 계산한 bytes와 allocator high-water가 다른 이유를 이 항목에서 찾는다.

dropless는 모든 assignment를 처리하므로 expert별 최대 token 수가 데이터와 router에 따라 달라진다. exact-size allocation은 count exchange 뒤 가능하지만 allocation overhead가 있고, reusable worst-case buffer는 메모리를 많이 쓴다. chunked dispatch는 상한을 낮추지만 collective 횟수와 latency를 늘릴 수 있다.

**OOM을 expert compute와 dispatch buffer로 분류한다.**

메모리 원장은 router logits/top-k metadata, send/receive counts, send/receive payload, sorted hidden, expert intermediates, grouped GEMM workspace, returned output, combine buffer와 backward saved state를 시간축에 놓는다. overlap이 켜지면 두 microbatch 또는 dispatch/compute buffer가 동시에 살아 peak가 커진다.

OOM 재현에서 expert count histogram과 allocation trace를 같은 step에 저장한다. 평균 load가 아니라 max local receive, max expert token과 in-flight generation 수가 핵심이다. rank 하나만 OOM이면 그 rank의 expert placement와 receive load를 본다. 모든 rank가 같은 지점에서 OOM이면 global capacity/workspace 설정을 조사한다.

복구 정책이 capacity를 즉석에서 낮추거나 token을 drop한다면 학습 의미가 바뀐다. 동일 UpdateID를 재시도할 때 이전 partial gradient와 controller count를 폐기해야 한다. 자동 완화는 새 config digest와 degraded flag를 남기고 품질 gate를 통과해야 한다.

**token permutation을 순열 대수로 검증한다**

**dispatch와 combine이 서로 역함수인지 증명한다**

accepted assignment A개의 dispatch permutation `π`가 source assignment order를 destination packed order로 바꾼다고 하자. expert compute가 순서를 보존한다면 return/combine은 `π^{-1}` 또는 그것과 동등한 reverse index를 사용해야 한다. top-k에서는 한 TokenID가 여러 assignment를 가지므로 key는 TokenID만 아니라 `(TokenID, slot 또는 ExpertID)`다.

순열 검증은 `sorted_indices`의 범위, 유일성, coverage를 확인한다. drop이 있으면 accepted subset의 순열이며 candidate 전체의 permutation이 아니다. padded capacity slot은 실제 assignment와 구분해야 한다. reverse map에서 duplicate나 missing이 있으면 합산 결과가 우연히 맞을 수 있는 all-one output 대신 고유 basis vector로 잡는다.

source rank를 넘는 all-to-all에서는 local TokenID가 충돌할 수 있다. global identity에 source DP/EP rank와 local sequence position 또는 stable sample/token ID를 넣는다. sequence packing과 microbatch 재정렬 뒤에도 같은 identity가 loss mask와 연결되어야 한다.

**backward에서 순열을 역순으로 재생한다.**

combine output gradient는 top-k slot으로 scatter되고 router weight가 곱해진 뒤 return permutation을 따라 destination expert output gradient가 된다. expert input gradient는 dispatch의 역경로로 source token에 모인다. forward metadata를 잘못 재계산하면 tie, capacity 또는 controller 변화로 다른 permutation을 만들 수 있다.

autograd custom function이 indices, splits와 weights를 `ctx`에 무엇으로 저장하는지 확인한다. activation checkpoint와 compile이 이 state를 재구성하는 방식도 본다. saved metadata dtype이 token 수를 담기에 충분한지, CPU/GPU 이동과 stream lifetime이 안전한지 검사한다.

finite-difference fixture는 작은 expert linear function을 두고 특정 input element를 흔든다. analytical input/router/expert gradient가 permutation reference와 맞는지 본다. expert output을 global assignment ID로 채우면 forward combine의 provenance를 한눈에 읽을 수 있다.

**grouped GEMM을 ragged batch scheduler로 해부한다**

**expert별 M 차이를 descriptor와 tile scheduling에 연결한다**

각 expert GEMM은 `M=T_e`, `K=H`, `N=I` 또는 그 반대 projection을 가진다. H와 I는 같아도 M이 expert별로 다르다. grouped GEMM은 pointer, leading dimension, M/N/K와 offsets 배열을 받아 여러 문제를 한 launch에서 처리한다. zero-token expert는 descriptor를 생략하거나 M=0으로 두는데 kernel 계약을 확인한다.

token sorting 뒤 expert offsets의 차이가 `T_e`다. offset의 마지막 값은 accepted assignment 수와 같아야 한다. int32 overflow 가능성, alignment padding과 actual/padded M을 분리한다. grouped GEMM descriptor가 host에서 만들어지면 graph capture와 dynamic token count가 충돌할 수 있고 device-side 생성이면 synchronization을 확인한다.

작은 M expert는 tile 일부만 사용해 waste가 크다. 여러 expert를 한 CTA가 처리하거나 persistent scheduling을 쓸 수 있지만 weight cache locality와 load balance가 달라진다. profiler에서는 achieved FLOPs만 아니라 active warps, tile utilization, DRAM bytes와 expert별 M 분포를 함께 본다.

**grouped path와 expert loop reference를 역할별로 비교한다.**

eager reference는 expert별 mask로 token을 골라 개별 MLP를 실행하고 index-add로 합친다. 느리지만 함수 oracle로 유용하다. grouped path는 permutation, descriptors와 fused activation을 추가한다. 두 경로의 비교를 router selection, packed input, projection intermediate, expert output와 combine 단계로 쪼갠다.

backward grouped GEMM은 input gradient, weight gradient에서 문제 shape와 reduction이 다르다. 같은 expert parameter에 여러 microbatch contribution이 누적될 때 beta 값 또는 zeroing 순서가 중요하다. zero-token expert의 gradient가 `None`, zero tensor 또는 이전 buffer 잔재인지 확인한다.

determinism mode에서는 grouped GEMM의 weight-gradient accumulation 순서가 달라 수치 오차가 생길 수 있다. deterministic 알고리즘 요구와 성능 모드를 분리하고 tolerance를 dtype/shape별로 정한다. non-determinism이 assignment ID나 count conservation 오류를 가리는 면허는 아니다.

**all-to-all 비용을 topology와 message histogram으로 예측한다**

**평균 bytes가 감추는 hot rank를 찾는다**

EP rank r에서 s로 보내는 assignment 수를 행렬 `C[r,s]`로 두면 payload bytes는 대략 `C[r,s]·H·b`다. 전체 합만 보면 모든 배치가 같아 보이지만 collective 완료 시간은 hot source/destination, link contention과 작은 message overhead에 좌우된다. count matrix와 physical topology를 함께 시각화한다.

intra-node NVLink/NVSwitch와 inter-node fabric을 넘는 expert placement는 비용이 다르다. group-limited routing이나 node-limited routing은 품질/선택 자유도를 일부 제한해 remote traffic을 줄일 수 있다. score 단계의 group selection, expert selection과 combine normalization을 분리해 수학을 적는다.

hierarchical all-to-all은 node 내부 pack, inter-node exchange, node 내부 scatter로 나뉠 수 있다. 각 단계의 buffer ownership, count와 stream/event를 기록한다. 한 단계만 빠르게 보이는 microbenchmark가 end-to-end dispatch를 대표하지 않는다.

**overlap을 숨은 serialization과 함께 측정한다.**

dispatch communication과 이전 microbatch expert compute를 겹칠 수 있지만 같은 HBM bandwidth, copy engine 또는 SM을 경쟁할 수 있다. async API 호출 시간과 실제 GPU timeline을 구분한다. downstream wait event가 너무 일찍 걸리면 overlap이 없고 너무 늦으면 stale data를 읽는다.

message histogram은 zero, tiny, medium, large bucket과 peer별 p99를 남긴다. router imbalance, batch shape와 network tail을 같은 trace에 연결한다. count exchange latency, payload latency, queueing과 combine return을 분리한다. NCCL error가 없어도 wrong group이나 peer mapping은 논리적 permutation 오류를 만들 수 있다.

15장의 topology manifest와 결합해 rank→GPU→node→NIC→expert global ID를 한 표로 만든다. placement 변경 실험은 동일 token/router decision을 고정해 network 효과만 비교한다. router가 함께 바뀌면 성능 차이를 placement 때문이라고 단정할 수 없다.

**DeepSeek 계열 MoE를 읽을 때 이름보다 상태 전이를 따른다**

**routed expert와 shared expert의 결합점을 찾는다**

DeepSeek 계열 구현을 조사할 때 먼저 config에서 hidden size, routed/shared expert 수, expert intermediate size, top-k, group 수, group당 선택 수, normalization과 routed scaling을 찾는다. 그다음 decoder layer가 dense MLP와 MoE block 중 무엇을 생성하는지 layer index 조건을 따른다. config 필드가 존재해도 해당 revision의 forward에서 사용되지 않을 수 있다.

MoE forward는 router/gate가 top-k indices와 weights를 만들고, training path가 token을 반복·정렬하거나 dispatcher로 넘기며, expert output을 원순서로 합친 뒤 shared expert path와 결합하는 흐름으로 읽는다. inference 전용 최적화 함수가 `no_grad` 또는 custom op를 쓰면 training backward를 설명하는 근거로 혼용하지 않는다.

group-limited selection에서는 expert score를 group으로 묶어 group score를 계산하고 일부 group만 남긴 뒤 expert top-k를 고를 수 있다. group score가 max, top-n 합 또는 bias-corrected 값인지 revision을 고정한다. correction bias가 selection에만 쓰이는지 combine weight에도 쓰이는지 확인한다.

**공개 구현과 논문 식 사이의 간극을 evidence로 남긴다.**

논문은 auxiliary-loss-free balancing, shared expert isolation, node-limited routing과 같은 개념을 제시하지만 공개 checkpoint/modeling code, training framework와 production kernel이 모두 같다고 가정할 수 없다. 각 주장은 논문 equation, 공개 config, Transformers modeling, Megatron/DeepSpeed 계열 구현 중 어느 근거에 기대는지 구분한다.

source 좌표는 repository, commit, file, symbol과 body fingerprint를 가진다. modeling file의 class 이름만 적지 않고 constructor가 만든 parameter, forward input/output와 helper 호출을 기록한다. 외부 custom kernel이 있으면 wrapper schema와 fallback reference까지 잇는다.

실습은 공개 config로 모듈 shape와 global expert mapping을 복원하되 대규모 모델을 실행하지 않는다. 작은 동형 module에 같은 routing convention을 적용해 top-k, group mask, normalization, shared 결합과 backward를 검산한다. 원본 대규모 수치를 임의 축소해도 유지되는 불변식과 유지되지 않는 성능 주장을 구분한다.

**Qwen MoE 계열에서 shared gate와 expert layout을 추적한다**

**config에서 forward까지 사용되는 필드만 승인한다**

Qwen 계열의 여러 세대와 dense/MoE 변형은 이름이 비슷해도 sparse layer 배치, routed/shared expert, activation, top-k와 normalization이 다를 수 있다. AutoConfig 결과만 보지 않고 실제 architecture class와 modeling revision을 고정한다. `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size` 같은 필드가 어느 module shape를 만드는지 따라간다.

shared expert output에 sigmoid gate를 곱하는 구조라면 gate input, projection shape, activation과 routed output의 합산 위치를 기록한다. shared gate가 token별 scalar인지 channel vector인지에 따라 표현력과 비용이 다르다. gate saturation은 shared path가 routed path를 가리는 장애를 만들 수 있으므로 gate distribution과 두 contribution norm을 따로 측정한다.

routed expert가 ModuleList loop로 구현된 eager path는 correctness oracle로 좋지만 empty expert, duplicate top-k와 dtype 처리 convention을 확인한다. optimized path가 custom op나 fused MoE로 대체될 때 입력 schema, sorted token contract와 output ordering을 비교한다.

**model conversion에서 projection role과 expert ID를 보존한다.**

Qwen checkpoint의 expert gate/up/down key를 다른 backend의 fused expert tensor로 옮길 때 expert dimension, projection role, TP axis와 quantization packing 순서를 mapping한다. shared expert와 shared gate는 global routed ExpertID namespace에 섞지 않는다. router weight의 expert column 순서가 expert tensor 첫 축과 맞아야 한다.

conversion fixture는 expert마다 다른 상수 weight와 router column을 넣어 한 token이 예상 global expert를 선택하는지 확인한다. top-k가 둘 이상이면 서로 다른 output basis를 combine해 weight까지 검산한다. load report key coverage 100%만으로 semantic mapping을 승인하지 않는다.

훈련/서빙 parity는 attention cache와 무관한 짧은 단일 step에서도 시험할 수 있다. teacher-forcing hidden을 같은 MoE module에 넣어 training eager, training fused와 serving kernel의 router ID, weight와 output을 비교한다. serving이 capacity나 batching을 다르게 처리하면 그 차이를 별도 정책으로 기록한다.

**Mixtral 계열 sparse block을 minimal reference로 사용한다**

**단순한 top-k 구조에서 핵심 불변식을 먼저 검증한다**

Mixtral 계열의 공개 modeling 구현에서는 router linear, softmax, top-k, selected weight normalization, expert별 MLP와 index-add 흐름이 비교적 직접 드러난다. 다만 라이브러리 revision마다 class와 반환값이 달라질 수 있으므로 고정 commit의 symbol과 forward body를 reference로 사용한다.

hidden `[B,S,H]`를 `[T,H]`로 펴고 router logits `[T,E]`를 만든 뒤 top-k weight/ID를 얻는다. expert mask가 `(expert,slot,token)` 순서를 어떻게 표현하고 token을 골라 expert MLP에 넣는지 추적한다. weighted output을 원 token row에 더할 때 dtype cast와 index accumulation 순서를 기록한다.

이 경로로 count conservation, duplicate token slot, zero-token expert, selected normalization과 router logits 반환을 검산한다. 그 뒤 grouped GEMM/distributed path에 같은 golden fixture를 적용한다. reference가 느리다는 이유로 수학적 oracle 가치가 사라지지 않는다.

**auxiliary loss API와 실제 global loss 연결을 확인한다.**

model output에 router logits가 포함되어도 trainer가 auxiliary loss를 main loss에 더하지 않으면 router balancing은 작동하지 않는다. causal LM wrapper가 label 존재 시 aux 함수를 호출하는지, coefficient를 어디서 곱하는지, tuple/structured output에서 loss 순서가 무엇인지 확인한다.

gradient checkpointing이나 `return_dict` flag에 따라 router logits 수집 방식이 달라질 수 있다. layer별 tuple을 concatenate할 때 token/attention mask 정렬이 유지되는지 본다. padding mask를 aux 함수에 전달하지 않는 revision은 packed/padded batch에서 denominator 의미가 달라질 수 있다.

훈련 recipe 문서의 `router_aux_loss_coef` 같은 옵션은 config field→wrapper loss→gradient contribution까지 닫혀야 한다. 값이 저장되지만 소비되지 않는 dead option, serving에서만 읽는 option과 이름 충돌을 source test로 구분한다.

**Transformers에서 MoE call graph를 자동 추적하지 않고 수동으로 복원한다**

**AutoModel의 동적 선택에서 실제 class까지 내려간다**

모델 카드의 architecture, config `model_type`과 `architectures`, AutoConfig mapping, AutoModel mapping을 따라 실제 causal LM wrapper와 base model class를 찾는다. remote code가 필요한 모델은 `trust_remote_code` 여부와 downloaded revision을 고정한다. 같은 모델 이름이 내장 modeling과 remote modeling 중 어느 것을 실행했는지 artifact에 남긴다.

causal LM forward에서 decoder layer, sparse block, router helper와 expert module로 내려가고 반환값이 다시 loss wrapper까지 올라오는 경로를 그린다. 각 함수의 positional/keyword arguments, tensor shape, optional flag와 mutable state를 적는다. grep 결과의 함수 이름 목록은 call graph가 아니다.

hook을 걸 위치는 router logits 전후, selected IDs/weights, dispatch 전 hidden, expert output, combined branch와 residual output이다. compile/custom op가 hook을 우회하면 debug fallback을 사용한다. 대규모 실행 없이도 작은 config로 class를 instantiate할 수 있다면 shape oracle만 수행하고, 실행 금지 조건에서는 source와 test fixture를 정적 분석해 `NOT_RUN`으로 표시한다.

**library test를 specification의 일부로 읽는다.**

Transformers의 model test는 config fixture, input preparation, output shape, gradient retention, save/load와 integration expectation을 담는다. MoE-specific test에서 router logits, auxiliary loss와 expert initialization을 찾는다. test가 없는 경로는 보장되지 않는다는 신호이지 곧 오류라는 뜻은 아니다.

generic tests가 sparse state를 실제로 검사하는지 확인한다. state dict round-trip이 expert semantic ID permutation을 잡지 못할 수 있다. 프로젝트 자체 fixture에 global ExpertID와 projection role pattern을 추가해야 한다. upstream test 이름과 local regression test를 evidence map에 연결한다.

라이브러리 upgrade는 modeling diff, config default diff, test diff와 checkpoint conversion diff를 함께 검토한다. class rename만 보고 behavior가 같다고 단정하지 않는다. activation approximation, top-k normalization, output tuple와 attention mask 처리처럼 수치와 API를 바꾸는 줄을 우선 분류한다.

### Transformers·Megatron 구현을 process group까지 추적한다

**모듈 경계가 만드는 tensor ownership을 기록한다**

Megatron 계열 구현에서는 transformer layer가 MoE layer를 호출하고 router가 logits와 routing map/weights를 만들며 token dispatcher가 permutation과 communication을 담당하고 experts가 grouped 또는 sequential MLP를 실행한다. 구체 class와 함수 이름은 revision별로 달라지므로 고정 source에서 constructor와 forward를 따라간다.

router의 load-balancing type, top-k, score function, pre-softmax 여부, z-loss, input jitter와 capacity/drop flag를 config에서 소비 지점까지 잇는다. dispatcher type이 allgather인지 alltoall인지에 따라 token duplication, sequence parallel 요구와 collective가 달라진다. option 조합의 validation assertion도 실행 계약이다.

expert tensor parallel을 사용하면 한 expert 내부 weight가 추가로 shard된다. expert parallel group은 expert 집합을 나누고 expert tensor-parallel group은 한 expert 계산을 나눈다. data/tensor/pipeline/context parallel group과 혼동하지 않도록 rank membership을 표로 만든다.

**token dispatcher의 preprocess와 postprocess를 대칭으로 읽는다.**

dispatcher preprocess는 token count, routing map과 destination split을 만들고 permutation/collective를 준비한다. token permutation 뒤 local expert offsets가 expert MLP의 input contract다. postprocess는 expert output에 weight를 적용하고 unpermute/collective를 거쳐 original hidden order로 돌린다.

async permutation, shared expert overlap 또는 flex dispatcher가 있다면 handle, stream, event와 buffer lifetime을 추적한다. config flag가 켜졌다고 실제 overlap이 생기는 것이 아니다. profiler timeline과 wait 위치를 확인한다.

loss tracker가 z-loss와 aux loss를 logging하는 경로와 실제 autograd loss에 주입하는 경로를 분리한다. metric이 보인다고 gradient가 흐른다는 뜻은 아니다. coefficient 0/nonzero fixture에서 router gradient 차이를 확인한다.

**expert collapse를 네 가지 서로 다른 현상으로 분해한다**

**선택 collapse, 계산 collapse, 학습 collapse, 의미 collapse를 구별한다**

선택 collapse는 대부분 token이 소수 expert로 향하는 현상이다. 계산 collapse는 count가 분산되어도 일부 expert kernel이나 rank가 병목이 되는 현상이다. 학습 collapse는 expert gradient/update가 0에 가까워 기능을 얻지 못하는 현상이다. 의미 collapse는 여러 expert가 비슷한 함수를 학습해 specialization 다양성이 사라지는 현상이다.

선택 collapse는 count, probability mass, margin, entropy와 accepted fraction으로 본다. 계산 collapse는 expert별 token당 kernel 시간, padding waste, rank receive bytes와 tail latency로 본다. 학습 collapse는 gradient/update norm, optimizer moments와 visit interval로 본다. 의미 collapse는 expert output similarity, parameter/gradient cosine, activation subspace와 domain별 기능 평가가 필요하다.

네 현상은 독립적일 수 있다. utilization이 균등해도 expert outputs가 모두 같으면 의미 collapse다. 한 expert에 token이 몰려도 그 expert가 빠른 placement에 있어 step time은 정상일 수 있다. shared expert가 강하면 routed expert output이 약해 학습 collapse를 가릴 수 있다.

**원인별 개입을 반증 가능하게 설계한다.**

router learning rate, balance coefficient, correction bias, capacity, temperature, jitter, data shuffle와 expert initialization은 서로 다른 원인에 작용한다. 한꺼번에 바꾸지 않고 각 개입의 예상 중간 지표를 적는다. balance coefficient를 올렸는데 probability mass는 변하지 않고 accepted count만 변했다면 capacity와 denominator를 의심한다.

dead expert 회생을 위해 weight를 복제·재초기화하거나 router bias를 조절하면 optimizer moment와 global ExpertID 계보가 바뀐다. intervention event, source expert, reset state와 UpdateID를 checkpoint manifest에 남긴다. 조용한 in-place 수정은 재현성을 파괴한다.

domain별 collapse 분석은 6장의 mixture provenance와 연결한다. 특정 domain이 드문 expert에만 의존한다면 load balancing이 그 specialization을 지울 수 있다. 평균 benchmark뿐 아니라 expert ablation, forced routing과 domain slice를 사용하되 forced routing 결과를 정상 policy 품질로 오해하지 않는다.

**expert initialization이 초기 routing과 gradient를 만드는 방식을 검증한다**

**router와 expert 대칭을 언제 어떻게 깨는지 확인한다**

모든 expert가 같은 weight로 시작하고 router도 대칭이면 expert output이 같아 main loss의 router gradient가 0이 될 수 있다. 실제 초기화의 서로 다른 random draw, router noise, data 순서와 auxiliary signal이 대칭을 깬다. expert cloning으로 dense checkpoint에서 시작하는 upcycling은 이 문제를 의도적으로 만든다.

upcycling에서는 dense MLP weight를 여러 expert로 복제하고 router를 새로 만든다. 처음에는 기능 parity가 좋지만 specialization을 위해 router/expert가 갈라져야 한다. 작은 expert-specific perturbation, router initialization과 balance schedule이 어떤 역할을 하는지 기록한다. perturbation scale이 너무 크면 원래 기능을 잃고 너무 작으면 수치적으로 대칭이 오래 유지된다.

projection별 initialization variance는 gated multiply의 분산을 결정한다. gate와 up이 독립인지, down projection scale이 residual depth에 맞춰 조정되는지 source initializer를 확인한다. config의 initializer range가 모든 parameter에 똑같이 적용된다고 가정하지 않는다.

**초기 수백 step을 별도 phase로 관측한다.**

초기에는 router entropy, margin, expert count, gate/output variance, gradient와 balance contribution을 짧은 간격으로 기록한다. 안정화 뒤 평균만 보면 첫 collapse와 회복을 놓친다. optimizer warmup, router 전용 learning rate와 controller 시작 시점을 같은 timeline에 둔다.

seed parity는 global ExpertID별 initializer RNG가 topology에 독립적인지 본다. EP degree가 바뀌면 local module 생성 순서가 달라져 expert weight가 달라질 수 있다. global ID 기반 seed 또는 checkpoint materialization으로 동일 초기 weight를 보장한다.

initialization test는 weight 통계만 아니라 작은 input의 expert별 output covariance와 router gradient를 측정한다. 동일 output expert fixture가 main router gradient 0을 만드는지 확인하면 autograd convention도 검증된다.

**MoE fine-tuning에서 어느 파라미터를 열 것인지 state 단위로 결정한다**

**router-only, expert-only, shared-only, adapter 조합의 경로를 구분한다**

router-only fine-tuning은 기존 expert 함수를 유지한 채 token 배치를 바꾼다. 적은 파라미터라도 routing distribution과 network load가 크게 변할 수 있다. expert-only는 router 선택을 고정하더라도 선택된 expert만 gradient를 받아 데이터 coverage가 불균등하다. shared-only는 모든 token이 통과하는 경로를 조절하지만 routed specialization과 상호작용한다.

LoRA를 router, gate/up/down, shared expert 또는 일부 global ExpertID에 붙일 수 있다. adapter parameter 이름, target module pattern, rank/scale/dropout과 base weight shard ownership을 기록한다. expert ModuleList regex가 일부 expert만 매칭하거나 fused gate-up module을 놓칠 수 있으므로 실제 trainable parameter manifest를 만든다.

freeze는 `requires_grad=False`만이 아니다. optimizer group 제외, weight decay, gradient buffer, distributed reduction과 checkpoint 저장이 함께 맞아야 한다. frozen router라도 aux loss를 계산하면 불필요한 메모리와 metric 혼동이 생길 수 있다. 반대로 router gradient를 원하면서 detached routing helper를 쓰면 학습되지 않는다.

**adapter merge와 serving export에서 routing 의미를 보존한다.**

expert별 adapter를 merge할 때 global ExpertID와 TP shard를 유지한다. shared adapter와 routed adapter namespace를 분리한다. quantized base에 adapter를 적용하는 순서와 merge dtype이 expert별 scale을 바꿀 수 있다.

serving backend가 expert adapter를 지원하지 않으면 dense/shared 경로만 export되거나 adapter가 silently ignored될 위험이 있다. exported state key coverage, one-token routing/output parity와 forced expert fixture를 검사한다. router-only adapter는 expert weight checksum이 같아도 output을 크게 바꾸므로 모델 weight diff 크기로 효과를 추정하지 않는다.

fine-tuning 데이터 mixture는 expert visitation을 결정한다. trainable expert가 거의 선택되지 않으면 effective update count가 작다. global step 대신 expert별 accepted token, gradient-bearing token과 update interval을 기록한다. curriculum 단계가 바뀔 때 visitation과 optimizer moment age를 본다.

**optimizer가 sparse expert를 업데이트할 때 step의 의미를 다시 정의한다**

**선택되지 않은 expert의 moment와 weight decay를 확인한다**

한 step에 token을 받지 않은 expert는 gradient가 `None`인지 zero tensor인지에 따라 optimizer 동작이 다를 수 있다. Adam 계열에서 zero gradient가 전달되면 moment가 decay하고 decoupled weight decay가 적용될 수 있지만 `None`이면 parameter update 전체를 건너뛸 수 있다. framework와 optimizer 구현을 source로 확인한다.

expert별 방문 간격이 길면 global optimizer step으로 bias correction을 계산하는 것과 local update count를 쓰는 것이 다르다. 대부분 구현은 global step을 공유하지만 sparse optimizer는 다른 convention을 가질 수 있다. checkpoint state의 `step`이 parameter별인지 group별인지 기록한다.

Muon처럼 행렬 구조를 이용하는 optimizer를 expert projection에 적용한다면 작은/불균등 expert gradient, TP shard와 orthogonalization communication의 의미를 별도로 검증해야 한다. optimizer 이름만 바꿔 dense layer와 같은 효과를 가정하지 않는다. router vector/matrix와 expert matrices를 서로 다른 optimizer group으로 나눌 수도 있다.

**expert별 update age를 관측 가능한 state로 만든다.**

각 global ExpertID에 last-visited UpdateID, gradient-bearing token count, last-update, update norm과 moment norm을 기록한다. 너무 오래 선택되지 않은 expert가 다시 호출될 때 stale moment와 weight가 큰 loss를 만들 수 있다. 평균 optimizer 통계는 이를 숨긴다.

gradient clipping이 global norm이면 hot expert가 전체 scale을 결정하고 cold expert gradient도 줄인다. expert별/group별 clipping은 다른 objective를 만든다. clipping 전후 router/shared/expert별 norm과 applied scale을 저장한다. distributed global norm reduction이 모든 expert shard를 포함하는지 확인한다.

resume 뒤 optimizer state가 누락된 expert는 weight가 맞아도 update trajectory가 달라진다. one-step parity와 몇 step의 visitation pattern을 함께 비교한다. state reset을 허용하면 exact resume가 아니라 warm restart로 분류한다.

**MoE checkpoint reshard를 집합·순열·slice 세 단계로 검증한다**

**global expert 집합과 owner mapping을 먼저 닫는다**

첫 단계는 checkpoint의 모든 `(LayerID,GlobalExpertID,Role)` 집합이 target architecture와 같은지 확인하는 것이다. expert 수, shared/routed 구분과 sparse layer 위치가 다르면 단순 reshard가 아니다. missing/duplicate ID를 발견한 뒤 tensor concatenate를 시도하지 않는다.

둘째 단계는 expert owner 순열이다. source EP rank/local slot에서 global ID로 복원하고 target EP rank/local slot으로 다시 배치한다. router weight의 expert output column, controller bias, count EMA와 optimizer state가 같은 순열을 사용해야 한다. 각 field가 독립 converter를 쓰면 하나만 빠질 수 있으므로 공통 mapping artifact를 사용한다.

셋째 단계는 한 expert 내부 TP slice다. gate/up/down의 logical shard axis와 fused layout을 복원해 global logical tensor를 만들거나 streaming slice 변환을 한다. 메모리 때문에 전체 materialization을 피하더라도 global offset coverage와 overlap을 검사한다.

**topology 변경 복구를 값과 함수 양쪽에서 확인한다.**

값 검사는 role별 logical checksum, shape, dtype와 optimizer slot을 비교한다. 함수 검사는 golden hidden이 같은 router ID/weight와 expert output을 만드는지 본다. backward/one-step 검사는 moment와 shard reduction 오류를 잡는다.

EP와 TP를 동시에 바꿀 때 순서를 명시한다. source local layout→global expert logical layout→target local layout의 두 단계가 이해하기 쉽지만 streaming converter는 동일 불변식을 유지해야 한다. conversion manifest에 source/target topology, mapping digest와 field coverage를 남긴다.

atomicity는 expert별 파일 성공이 아니라 한 UpdateID의 모든 weight/router/optimizer/controller/RNG가 닫혀야 한다. 부분 conversion artifact는 완료 marker 없이 격리한다. 재시도는 deterministic output path와 digest로 idempotent해야 한다.

**MoE 관측성을 원인-결과 그래프로 설계한다**

**router, dispatcher, expert, optimizer 지표를 한 TokenID에 잇는다**

router plane에는 logit norm, entropy, top-k margin, probability mass, candidate/accepted count, drop과 churn이 있다. dispatcher plane에는 peer count matrix, packed rows, bytes, permutation errors, buffer high-water와 collective latency가 있다. expert plane에는 token count, GEMM shape, executed/useful FLOPs, output/gradient norm과 kernel time이 있다. optimizer plane에는 update norm, moment, age와 clipping scale이 있다.

이 지표를 독립 dashboard로 나열하지 않고 causal edge를 만든다. margin 축소→assignment churn→peer count 변화→buffer 재할당→tail latency처럼 이어질 수 있다. 또는 데이터 curriculum 변화→domain routing 편향→cold expert age 증가→재방문 loss spike가 될 수 있다. 같은 UpdateID, layer, global ExpertID와 rank label로 join할 수 있어야 한다.

label cardinality를 통제한다. Prometheus metric에 TokenID나 sample ID를 직접 label로 넣으면 폭발한다. aggregate metric은 layer/expert/rank 정도로 제한하고 상세 assignment는 sampled trace 또는 artifact에 저장한다. exemplars나 trace ID로 두 계층을 잇는다.

**경보를 symptom이 아니라 검증 가능한 가설로 만든다.**

`expert_load_cv 높음` 하나는 경보다. runbook은 probability mass도 치우쳤는지, accepted 단계에서만 치우쳤는지, 특정 source rank/domain인지, controller state가 최신인지 순서대로 확인한다. 각 분기에서 필요한 query와 artifact를 적는다.

throughput 하락은 max receive count, grouped GEMM M histogram, network peer tail, workspace allocation과 overlap wait로 분해한다. loss spike는 first non-finite, routed/shared contribution, dropped token domain, router gradient와 optimizer reset을 본다. hang은 collective sequence와 counts를 먼저 확인한다.

경보 threshold는 hardware/topology와 batch shape별 baseline에서 정한다. expert 수가 다른 모델에 고정 count threshold를 복사하지 않는다. ratio도 denominator 0과 tiny batch를 처리한다. 배포 전 synthetic imbalance와 corrupted permutation을 넣어 경보가 예상 원인으로 연결되는지 시험한다.

**MoE 장애를 최초 잘못된 상태에서 역추적한다**

**loss NaN이 나타나기 전 router와 expert 경계를 검사한다.**

NaN triage는 최종 loss에서 시작하되 뒤로만 추측하지 않는다. layer residual output, routed/shared branch, combine, expert output, activation product, projection, packed hidden과 router logits에 finite check를 이분 배치한다. 최초 non-finite tensor의 producer와 input을 확보한다.

router logits가 finite인데 softmax가 NaN이면 stabilization/dtype을 본다. expert output 한 global ID에서만 NaN이면 해당 weight/optimizer state, activation range와 quantization scale을 본다. combine 뒤에만 NaN이면 weight normalization, duplicate accumulation 또는 uninitialized output을 조사한다.

residual이 NaN을 다음 layer로 퍼뜨리므로 발견 layer가 원인 layer와 다를 수 있다. sampled checksum과 anomaly hook의 overhead를 관리하면서 reproduction window를 좁힌다. 재현 seed, data provenance, topology와 source revision을 bundle로 남긴다.

**hang과 silent corruption을 서로 다른 증거로 다룬다.**

hang은 rank별 collective sequence number, process group, send/receive splits와 CUDA event wait graph를 비교한다. 한 rank가 zero token이라 collective를 건너뛰었는지, 예외 뒤 다른 rank가 계속 진입했는지 본다. watchdog timeout stack만으로 원인을 확정하지 않는다.

silent corruption은 count가 맞아도 permutation, scale metadata 또는 expert ID가 틀릴 수 있다. identity-coded payload, reverse map invariant와 end-to-end eager oracle가 필요하다. checksum은 같은 bytes의 잘못된 row 배치를 잡지 못하므로 identity와 함께 계산한다.

복구 후에는 buffer generation을 증가시키고 이전 async handle이 새 buffer를 쓰지 못하게 한다. failed microbatch의 partial gradient, controller count와 optimizer step을 폐기한다. checkpoint rollback은 UpdateID closure를 기준으로 한다.

**training-serving parity를 MLP 함수와 routing 정책으로 분리한다**

**같은 weight가 같은 함수를 실행하는지 먼저 본다.**

training backend와 serving backend의 MLP activation approximation, gate/up order, weight transpose, quantization, accumulation dtype와 residual scale을 비교한다. teacher-forcing hidden을 직접 넣을 수 있는 module fixture로 attention/cache 변수를 제거한다. dense, shared, 각 routed expert를 forced selection으로 한 번씩 방문한다.

weight converter는 key coverage 외에 logical role digest와 output oracle를 통과해야 한다. serving fused kernel이 gate-up packed layout을 요구하면 training checkpoint에서 conversion한 bytes와 scale metadata를 검증한다. activation 이름이 같아도 approximation flag를 확인한다.

**routing 정책 차이를 의도된 것과 오류로 구분한다.**

serving은 capacity drop 없이 dynamic batching을 처리하거나 expert parallel topology가 다를 수 있다. top-k score, normalization, correction bias, routed scale와 shared combination이 training과 같은지 본다. load-balancing loss는 serving에 필요 없지만 selection controller bias는 필요할 수 있다.

같은 hidden에서 router logits, selected global ExpertID, combine weight, expert output와 final branch를 단계별 비교한다. batch composition이 routing에 영향을 주는 capacity 정책이면 단일 token parity와 실제 batch parity를 분리한다. serving batch의 다른 요청 때문에 한 요청의 expert 선택이 달라진다면 정책적으로 허용되는지 명시한다.

KV cache가 개입하는 autoregressive serving에서도 MoE 입력 hidden을 capture해 training/reference module에 재생한다. 차이가 attention에서 시작했는지 MoE에서 시작했는지 분리한다. 1권의 serving 추적과 이 장의 학습 함수 원장은 여기서 만난다.

**dense·MoE 비교 실험을 parameter·FLOPs·token budget으로 정규화한다**

**무엇을 같게 했는지 제목에 적는다.**

dense와 MoE는 total parameter, active parameter, forward FLOPs, memory, communication과 학습 token 효율이 다르다. “같은 크기”라는 표현 대신 total parameter matched, active FLOPs matched, wall-clock matched 또는 token budget matched를 명시한다. expert parameter가 optimizer/checkpoint memory를 차지하지만 매 token 모두 실행되지는 않는다.

active FLOPs에는 selected experts와 shared expert, router, padding/grouped GEMM waste를 구분한다. theoretical useful FLOPs와 measured executed FLOPs가 다르다. network와 imbalance 때문에 같은 FLOPs에서도 wall-clock이 달라진다. hardware utilization만으로 모델 효율을 평가하지 않는다.

parameter matched 비교에서 MoE intermediate width나 expert 수를 줄일 수 있고, FLOPs matched 비교에서는 total capacity가 크게 늘 수 있다. optimizer state와 checkpoint I/O, fault blast radius까지 운영 비용에 넣는다. validation loss는 같은 seen token과 data order에서 비교한다.

**성능 곡선을 단일 종점이 아니라 학습 전 과정으로 본다.**

loss-versus-token, loss-versus-FLOPs, loss-versus-wall-clock과 downstream quality를 각각 그린다. routing balance, expert specialization과 communication overhead가 시간에 따라 변하므로 초기/중기/후기를 나눈다. early instability 때문에 최종 품질만 같아도 복구 비용이 다를 수 있다.

ablation은 shared expert, top-k, balance method, capacity, expert count와 placement를 한 축씩 바꾼다. 여러 값을 동시에 바꾸면 원인을 분리할 수 없다. 각 run의 config/source/data/topology digest와 seed를 고정한다.

통계는 여러 seed와 confidence interval을 사용하고 실패 run을 제외하지 않는다. OOM/hang/collapse도 시스템 결과다. 성공 run만 평균내면 sparse architecture의 안정성 비용을 숨긴다.

**CUDA kernel 최적화를 수학적 oracle 뒤에 배치한다**

**epilogue fusion이 어떤 tensor를 없애는지 적는다.**

gate/up GEMM epilogue에서 activation과 multiply를 융합하면 중간 `G`, `U`의 HBM 저장을 줄일 수 있다. 그러나 backward가 필요하면 필요한 값 또는 압축된 상태를 저장하거나 재계산해야 한다. training kernel과 inference kernel의 계약을 혼동하지 않는다.

down projection 뒤 TP reduction과 residual add를 융합하거나 overlap할 수 있다. reduction 결과의 ownership, accumulation dtype와 residual read 시점이 바뀐다. custom autograd가 backward collective를 정확히 정의하는지 본다.

grouped expert kernel은 pointer array, expert offsets, quantization scale와 workspace를 입력으로 받을 수 있다. wrapper에서 만든 descriptor의 device, lifetime와 graph capture compatibility를 검사한다. kernel 내부를 읽을 때 tile shape, split-K, accumulation, bounds/padding과 epilogue를 logical equation에 대응시킨다.

**kernel benchmark를 shape distribution과 연결한다.**

single GEMM peak shape가 아니라 실제 trace에서 얻은 expert M histogram을 replay한다. uniform, Zipf-like, all-to-one, many-zero 분포를 포함한다. forward gate/up/down과 backward dgrad/wgrad를 따로 측정한다. allocator와 descriptor 생성 시간도 end-to-end에 포함한다.

수치 검증은 FP32 reference, dtype별 tolerance, extreme activation, non-contiguous/padded layout과 zero M을 포함한다. gradient finite difference는 작은 shape에서 하고 production shape는 eager/autograd reference와 비교한다. race 검출을 위해 stream 변경과 buffer poisoning을 사용한다.

16장의 CUDA 도구와 연결해 kernel source→launch site→trace event→tensor role을 잇는다. kernel 이름이 동적으로 바뀌어도 NVTX range와 call-site fingerprint로 재현한다. 성능 regression은 source/config/topology 변화와 함께 bisect한다.

**residual branch scaling이 깊이와 sparse variance에 미치는 영향을 계산한다**

**identity path와 branch variance를 분리한다.**

residual update를 `r_{l+1}=r_l+α_l f_l(Norm(r_l))`로 쓰면 `α_l`과 branch 출력 분산이 깊이 방향 안정성을 좌우한다. 초기화나 architecture가 residual projection을 depth-dependent scale로 줄일 수 있다. 실제 initializer와 forward multiplier를 확인한다.

MoE branch는 token마다 다른 expert와 combine weight를 사용해 출력 분산이 routing에 의존한다. top-k weight 합, shared contribution과 expert별 output scale가 residual ratio를 바꾼다. dense MLP에서 안정한 scale이 sparse branch에서 자동으로 같지 않다.

layer별 `||branch||/||residual||`, cosine, channel variance와 token/domain slice를 본다. 평균 ratio가 정상이어도 특정 expert/domain에서 spike가 생길 수 있다. branch를 0으로 한 identity oracle, expert output을 상수로 한 combine oracle와 실제 distribution을 연결한다.

**norm 위치와 precision이 residual drift를 바꾸는 경로를 추적한다.**

pre-norm은 identity gradient path를 제공하지만 residual state 자체가 layer를 거치며 누적된다. residual을 FP32에 유지하거나 BF16로 저장하는 선택은 memory와 drift를 바꾼다. fused residual-norm kernel이 input/output dtype과 saved statistic을 어떻게 처리하는지 확인한다.

gradient에서는 identity contribution과 branch Jacobian contribution을 hook으로 분리할 수 있다. 깊이별 gradient norm만 보면 cancellation을 놓칠 수 있다. finite-difference로 작은 stacked block을 검산하고 activation checkpoint/fusion 경로와 비교한다.

mHC 같은 learned residual mixing 계열을 비교할 때는 mixing state, normalization/constraint, initialization과 checkpoint를 명시한다. “residual 개선”이라는 이름 대신 표준 identity update에서 어떤 행렬/계수가 추가되고 gradient와 compute가 어떻게 바뀌는지 수식과 source로 설명한다.

**mHC와 일반 residual을 동일한 상태 원장에 놓는다**

**hyper-connection state가 어디에서 생성되고 소비되는지 찾는다.**

일반 residual은 대개 한 hidden stream과 branch output을 더한다. hyper-connection 계열은 여러 stream 또는 확장된 residual state 사이의 mixing, branch input projection과 output redistribution을 학습할 수 있다. 정확한 mHC 정의는 해당 논문과 공개 구현 revision의 수식으로 고정한다.

source 분석은 configuration, connection module 생성, pre/post mapping, constraint 또는 normalization helper, forward에서 attention/MLP를 감싸는 호출 순서를 따른다. mixing parameter의 shape, 공유 범위, dtype, initializer와 optimizer group을 기록한다. 이름이 같은 비공식 구현을 근거로 섞지 않는다.

추가 stream dimension이 hidden activation, communication과 checkpoint에 미치는 비용을 계산한다. TP/SP에서 어느 축이 shard되는지, branch function은 합쳐진 hidden을 받는지 각 stream을 받는지 확인한다. residual dropout/RNG가 mapping 전후 어디에 적용되는지도 중요하다.

**기하학적 제약을 수치 불변식으로 바꾼다.**

mixing matrix에 doubly stochastic, nonnegative 또는 특정 합 제약이 있다면 normalization 뒤 row/column sum, 최소값과 condition을 검사한다. Sinkhorn 반복을 사용한다면 iteration 수, log-space 안정화, epsilon과 dtype을 기록한다. constraint가 근사적으로만 만족되는 허용 오차를 정한다.

identity initialization에서는 새 구조가 기준 residual과 같은 또는 가까운 함수를 만드는지 tiny stacked block으로 비교한다. forward만 아니라 input/branch/mixing gradient를 본다. checkpoint load에서 mixing state가 누락되면 default identity로 조용히 채우지 않고 migration으로 표시한다.

MoE와 함께 쓸 때 routed branch variance와 learned mixing이 서로 보상해 collapse를 감출 수 있다. expert utilization, branch contribution과 mixing coefficient를 동시에 관측한다. 7장의 norm/position state, 8장의 attention branch와 이 장 MLP branch가 같은 connection module을 공유하는지 확인한다.

**소스 코드 좌표를 함수 이름이 아니라 실행 가능한 증거 묶음으로 만든다**

**revision·symbol·body·caller를 함께 고정한다.**

`modeling_x.py의 MoE.forward`처럼 쓰면 upstream 변경 뒤 좌표가 흐려진다. repository URL, commit hash, file path, symbol, 시작 줄은 탐색용이고 body fingerprint와 relevant excerpt digest는 동일성 검사용이다. caller/callee와 constructor에서 주입된 config도 함께 기록한다.

한 함수의 의미는 wrapper와 helper에 흩어져 있다. router forward, top-k helper, load loss, dispatcher, fused op schema, checkpoint loader와 config validation을 evidence bundle로 묶는다. 논문 equation과 모델 카드 설명은 각각 별도 source이며 코드 behavior를 대신하지 않는다.

부분 인용은 핵심 조건과 state transition을 이해할 만큼만 사용하고 전후 맥락을 해설한다. 코드 줄을 길게 복사하는 대신 입력 shape, branch condition, mutable state, output과 exception을 한국어로 복원한다. 저자 해석과 source 사실을 구분한다.

**upgrade diff를 의미 단위로 분류한다.**

새 release에서 파일 전체 diff를 읽는 대신 config default, routing equation, dtype/cast, layout, process group, loss denominator, checkpoint key와 test expectation 변화로 분류한다. 각 변화가 forward, backward, distributed, resume와 serving parity 중 어디를 깨뜨릴 수 있는지 적는다.

line number가 바뀌어도 body fingerprint와 symbol graph로 같은 구현인지 찾는다. 함수가 custom op로 옮겨지면 Python wrapper뿐 아니라 schema, C++/CUDA dispatch와 fallback을 새로 고정한다. 삭제된 test나 validation assertion도 위험 신호다.

독자가 repository를 다시 파는 실습은 검색어만 주지 않는다. config class→module constructor→forward→helper→custom op→test→checkpoint converter의 순서와 각 단계의 질문을 제공한다. 발견 내용을 shape/state/ownership/effect/failure 표에 채우면 새로운 모델도 동일 방법으로 비교할 수 있다.

### function·routing·collective·checkpoint 회귀 시험으로 release한다

**수식 시험에서 클러스터 시험까지 실패 범위를 좁힌다.**

첫 층은 FP64 scalar/vector 식이다. GELU/SiLU derivative, top-k selected normalization, balance/z-loss와 residual update를 검산한다. 둘째 층은 단일 expert/dense module의 forward/backward와 one-step optimizer다. 셋째 층은 multi-expert eager routing과 permutation이다.

넷째 층은 fused/grouped kernel parity, 다섯째 층은 TP/EP collective와 zero-token rank, 여섯째 층은 checkpoint save/reshard/resume, 일곱째 층은 serving export parity다. 마지막은 실제 shape distribution의 성능/장애 주입이다. 아래 층이 실패하면 위층의 loss curve로 원인을 찾으려 하지 않는다.

각 test는 config/source digest, seed, tensor fixture, expected invariant와 tolerance를 가진다. golden output만 저장하면 implementation 변화에 취약하므로 중간 logical table도 저장한다. expected failure test는 corrupted count/permutation/state가 정확한 detector에서 거부되는지 본다.

**옵션 하나의 영향 범위를 regression matrix로 고정한다.**

예를 들어 `top_k` 변경은 router output shape, accepted assignments, buffer size, expert FLOPs, network bytes, balance denominator, checkpoint config와 serving kernel schema를 바꾼다. `capacity_factor`는 drop/memory/objective를, `router_dtype`은 margin/churn과 count agreement를 바꾼다.

옵션별 producer, changed tensor/state, downstream consumer, observable effect, failure와 rollback을 표로 만든다. config parser가 값을 받는다는 test와 실제 effect test를 구분한다. dead option은 effect가 없음을 찾아내야 한다.

CI는 작은 동형 fixture로 의미 회귀를 빠르게 잡고 nightly는 multiple topology와 real histogram replay를 수행한다. GPU가 없는 환경에서는 source/config/schema/fixture 생성을 검증하고 실행 증거를 `NOT_RUN`으로 남긴다. 실행하지 않은 kernel을 통과했다고 쓰지 않는다.

**9장의 지식을 다른 장과 왕복시키는 교차 탐색로**

**입력 표현에서 optimizer state까지 한 토큰을 추적한다.**

5장의 tokenizer와 6장의 packing/mixture가 만든 TokenID, mask와 domain provenance가 7장의 embedding/residual state를 거쳐 8장의 attention branch와 이 장 MLP/MoE에 들어온다. routing 분석에서 domain 편향을 말하려면 이 provenance가 끊기지 않아야 한다.

10장은 실제 모델 decoder layer가 attention, norm, MLP/MoE와 residual을 어떤 순서로 호출하는지 닫는다. 이 장의 standalone equation이 모델의 실제 괄호와 다르면 10장의 call graph가 우선한다. 11~14장의 loss/SFT/PEFT/RL은 router/expert에 어떤 gradient signal을 주는지 연결한다.

15장은 TP/EP/DP/PP process group과 collective 비용을, 16장은 CUDA kernel/precision을, 17장은 checkpoint와 장애 복구를 확장한다. 이 장의 assignment ledger와 global ExpertID는 세 장에서 같은 식별자를 써야 한다. 26장의 관측성은 metric/trace cardinality와 runbook을 운영 체계로 만든다.

**질문에서 관련 장으로 갔다가 다시 함수로 돌아온다.**

“특정 언어 성능이 떨어졌다”면 27장 평가 slice에서 6장 데이터 mixture, 이 장 domain별 drop/routing, optimizer visit age를 거쳐 실제 router/expert gradient로 돌아온다. “step time p99가 늘었다”면 26장 trace에서 15장 topology와 이 장 count matrix/grouped M histogram으로 내려온다.

“resume 후 loss가 튄다”면 17장 commit closure에서 이 장 router/controller/optimizer mapping과 첫 assignment를 확인한다. “serving 품질만 다르다”면 1권 serving path와 이 장 activation/layout/routing parity를 비교한다. 장간 링크는 참고 문헌 장식이 아니라 진단의 왕복 경로다.

각 교차 탐색은 입력 artifact, join key, 기대 불변식과 종료 판정을 가진다. 다른 장을 읽으라는 말로 끝내지 않고 돌아와 확인할 tensor와 state를 적는다. 그래야 독자가 어느 페이지에서 시작해도 실제 문제 해결로 수렴한다.

**한 개 token으로 dense·SwiGLU·MoE를 끝까지 손계산한다**

**작은 정수 weight로 forward provenance를 만든다.**

hidden `x=[1,-1]`과 작은 2×2 또는 2×3 weight를 정해 dense GELU, SwiGLU gate/up/down을 계산한다. transcendental activation은 충분한 정밀도의 표를 별도 제공하고 projection 결과, activated gate, product와 output을 단계별로 적는다. 각 channel에 서로 다른 값을 써 permutation을 드러낸다.

MoE는 expert 두 개, top-2 router로 만든다. router logits, softmax, selected weights와 각 expert MLP output을 계산해 weighted sum과 residual add를 구한다. capacity가 1인 두-token 변형에서는 candidate priority, accepted/drop과 재정규화 convention을 비교한다.

이 표의 모든 행은 TensorID, producer function과 consumer를 가진다. 수식 결과를 eager reference fixture의 expected 값으로 옮긴다. dtype을 FP64→FP32→BF16으로 바꾸며 어느 단계에서 반올림이 assignment를 바꾸는지 본다.

**backward와 optimizer update로 왕복을 닫는다.**

간단한 scalar loss `L=½||y-target||²`를 두고 output gradient부터 down, activation product, gate/up, expert weight, router weight와 input으로 chain rule을 적용한다. selected softmax Jacobian과 residual identity gradient를 분리한다. capacity drop assignment에는 어느 gradient가 사라지는지 표시한다.

SGD 한 step으로 먼저 손계산하고 Adam은 first/second moment, bias correction과 parameter update를 표로 만든다. 선택되지 않은 expert gradient가 None인지 zero인지 두 convention을 비교한다. checkpoint에 저장해야 다음 step을 재현할 state를 색인한다.

TP/EP 변형에서는 같은 logical 표를 rank별 shard/packed row로 나눈 뒤 collective로 복원한다. 숫자가 작아도 count, permutation, owner와 global ExpertID 불변식은 실제 클러스터와 같다. 이 손계산은 구현 전체의 가장 작은 semantic checksum이다.

**변경 전후를 판정하는 운영 인수 기준**

**정확성·학습성·성능·복구를 별도 gate로 둔다.**

정확성 gate는 forward/backward/one-step parity, count/permutation, finite와 identity를 본다. 학습성 gate는 router/expert gradient support, balance/controller 동작, collapse와 representative loss curve를 본다. 성능 gate는 useful/executed FLOPs, HBM, message histogram, tail과 peak memory를 본다. 복구 gate는 checkpoint closure, reshard와 failure injection 뒤 exact/warm resume를 본다.

한 gate의 성공으로 다른 gate를 대신하지 않는다. loss가 내려가도 expert ID permutation이 틀릴 수 있고, parity가 맞아도 network tail이 배포를 막을 수 있다. throughput이 좋아도 drop 정책이 데이터 domain을 편향시킬 수 있다.

승인 artifact에는 source/config/data/topology digest, model/update ID, dtype, kernel/collective revision, test matrix와 미검증 범위를 포함한다. benchmark 숫자는 trace와 연결하고 metric은 denominator를 적는다. reviewer가 raw artifact에서 결론을 재계산할 수 있어야 한다.

**미검증 범위를 미래의 장애 부채로 남기지 않는다.**

사용하지 않은 quantization, topology, capacity path와 serving backend는 `NOT_RUN`으로 명시한다. 지원하지 않는다고 선언할지 다음 gate로 넘길지 owner와 조건을 적는다. “문제없을 것으로 예상”은 증거가 아니다.

release 뒤에는 router margin/load, buffer high-water, expert kernel tail, loss slice와 checkpoint health를 관측한다. rollback은 weight 파일만 아니라 router/controller/optimizer/config와 code revision을 함께 되돌린다. incompatible assignment ledger가 남지 않게 generation을 바꾼다.

이 기준을 통과한 구현은 특정 모델 이름을 안다는 수준을 넘는다. dense/gated 함수의 미분과 bytes, sparse assignment의 제어와 통신, global expert state의 학습과 복구를 한 계약으로 설명하고 재현할 수 있다. 다음 모델에서 함수 이름이 달라져도 같은 질문과 fixture로 의미를 복원할 수 있다.

## Router에서 expert combine까지 tensor를 따라간다

Dense FFN은 `[N,D]→[N,H]→[N,D]`의 모든 parameter를 모든 token에 적용한다. GLU 계열은 gate와 value 두 projection을 만들고 `φ(gate)⊙value`를 down-project한다. GeGLU는 GELU, SwiGLU는 SiLU를 gate activation으로 쓴다. parameter와 activation byte가 늘어나는 것, nonlinear gating이 표현을 바꾸는 것, fused kernel이 실제로 빠른지는 서로 다른 주장이다.

Mixtral의 고정 구현은 hidden `[N,D]`와 router weight `[E,D]`로 logits `[N,E]`를 만든다. FP32 softmax 뒤 top-k index와 weight `[N,k]`를 고르고 k축 합이 1이 되도록 다시 정규화한다. expert별로 선택 token을 모아 gate/up projection, activation(gate)×up과 down projection을 수행하고 token index에 weighted `index_add_`한다. 직접 테스트는 router logits shape와 load-balancing aux loss가 masked padding 추가에 불변이고 mask를 빼면 달라지는지를 검사한다.

이 정상 경로만으로 capacity와 expert parallelism은 닫히지 않는다. all-token-one-expert collapse, exact router tie, NaN/Inf, zero-token expert와 capacity overflow를 작은 fixture에 넣는다. token conservation `ΣM_e=Nk`, combine weight sum, drop reason과 deterministic tie-break를 검사한다. no-drop은 손실 없음과 공짜가 같은 말이 아니다. padding·all-to-all tail과 expert memory가 늘 수 있다.

load-balance aux loss와 router z-loss는 main LM loss와 분모·계수가 다른 목적함수다. 각각 gradient-check하고 padding과 rank별 valid-token 불균형을 넣는다. shared expert는 모든 token이 통과하는 dense-like 경로와 routed 경로의 출력·parameter owner를 분리한다. EP에서는 rank별 split size, expert owner, dispatch permutation과 reverse combine을 reference와 맞춘다. DeepSeek·Qwen·OLMo·Megatron의 group routing, shared expert, no-drop과 precision 정책을 Mixtral 구현에서 자동으로 일반화하지 않는다.
