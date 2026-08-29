# 14장. 저정밀과 학습 커널

저정밀 학습은 dtype 이름을 바꾸는 일이 아니다. 더 정확히 말하면 **한 update가 어떤 표현으로 저장되고, 어떤 표현으로 읽혀 어느 커널에서 누산되며, 어떤 조건을 만족해야 optimizer state에 커밋되는지를 정하는 일**이다. 이 장의 중심 질문도 “BF16과 FP8 가운데 무엇이 빠른가”가 아니다. `TensorID` 하나를 골랐을 때 다음 상태 사슬을 끊김 없이 복원할 수 있는가가 중심 질문이다.

`FP32 master/storage → autocast가 선택한 compute input → kernel accumulator·epilogue → output storage → scaled backward → unscale → rank 간 finite 합의 → clip → optimizer moment·master update → 저정밀 model view 재생성 → FP8 scale·cache version 갱신 → UpdateID 커밋`

이 사슬에서 앞부분은 수치 표현, 가운데는 CUDA dispatch, 뒷부분은 학습 트랜잭션이다. 어느 한 부분만 관찰하면 원인을 잘못 붙인다. 예를 들어 loss가 유한하더라도 어떤 rank의 gradient는 Inf일 수 있다. FP8 GEMM이 정확하더라도 optimizer가 갱신한 master weight를 quantized weight cache가 보지 못하면 다음 forward는 오래된 값을 읽는다. 반대로 parameter가 움직이지 않았다고 모두 overflow인 것도 아니다. gradient accumulation 경계, scheduler clock, `zero_grad` 시점이 어긋났을 수 있다.

따라서 이 장은 모든 옵션을 `옵션 → 직접 바뀌는 상태 → 선택 가능한 커널 → 기대 효과 → 새 실패 모드 → 판별 fixture`의 여섯 칸으로 읽는다. 옵션 이름이나 설정 로그는 실행 증거가 아니다. 실제 dtype, scale version, kernel identity와 커밋 여부가 같은 `RunID·TensorID·UpdateID`로 결합되어야 한다.

## 14.1 dtype·scale·rounding이 만드는 수치 계약

저정밀 학습의 출발점은 dtype 이름이 아니라 표현 가능한 범위, 누적 형식, 반올림과 scale의 결합이다. FP32·TF32·BF16·FP16·FP8을 이 계약으로 비교한 뒤 loss scaling과 fused backward가 어느 상태를 보존하거나 잃는지 살핀다.

### 14.1.1 FP32·TF32·BF16·FP16·FP8

**범위와 정밀도의 교환**

FP16은 BF16보다 fraction bit가 많지만 exponent가 좁아 overflow에 취약하다. BF16은 FP32와 비슷한 exponent 범위를 가져 gradient scaling 부담을 줄인다. 여기서 “범위가 넓다”와 “작은 차이를 잘 보존한다”는 다른 성질이다. 같은 입력으로 큰 값 fixture와 서로 가까운 값 fixture를 따로 만들지 않으면 FP16의 overflow와 BF16의 거친 반올림을 한 문장으로 뭉개게 된다.

TF32는 FP32 tensor를 받되 Tensor Core matmul 내부 정밀도를 바꾸는 실행 모드다. parameter storage를 TF32로 바꾼다는 뜻이 아니다. FP8은 더 나아가 E4M3/E5M2 code만으로 실수값을 복원할 수 없고 scale이 필요하다. delayed scaling이라면 amax history와 history cursor까지 다음 cast의 의미를 결정한다. 그러므로 checkpoint의 최소 단위도 `tensor bytes`가 아니라 `bytes + scale + history + recipe + version`이다.

독자는 dtype을 다음 원장으로 기록한다.

| 위치 | 물어야 할 것 | 놓쳤을 때 생기는 오판 |
|---|---|---|
| master parameter | update가 적용되는 권위 있는 값의 dtype과 소유 rank | model view만 보고 작은 update가 보존된다고 판단한다 |
| storage/model view | forward가 읽는 parameter·buffer dtype | checkpoint 크기를 compute precision과 동일시한다 |
| compute input | autocast·quantizer 뒤 실제 operand dtype | 설정값만 보고 Tensor Core 경로를 탔다고 판단한다 |
| accumulator | GEMM·reduction·split-K가 합을 모으는 dtype과 순서 | 입력 dtype만으로 오차와 재현성을 설명한다 |
| output/epilogue | bias·activation·output cast의 순서 | fusion 전후 반올림 위치 차이를 놓친다 |
| gradient/collective | local gradient, 통신 payload, reduction accumulator | 한 rank의 finite 판정과 global 판정을 혼동한다 |
| optimizer state | master, moment, step counter와 model view 갱신 순서 | skip step에서 일부 state만 움직인 partial commit을 놓친다 |

**autocast의 경계**

PyTorch autocast는 model 전체를 한 dtype으로 복사하지 않고 op별 정책으로 입력을 cast한다. normalization·reduction처럼 범위가 필요한 연산은 더 높은 정밀도를 유지할 수 있다. `matmul_precision`, TF32 backend flag, autocast dtype은 서로 다른 switch다. 하나는 FP32 matmul의 내부 경로에, 하나는 backend 허용 정책에, 하나는 autocast 대상 op의 입력 선택에 영향을 줄 수 있으므로 한꺼번에 켜고 “BF16 효과”라고 이름 붙이지 않는다.

관찰도 op 경계에서 해야 한다. module parameter dtype만 출력하면 autocast가 만든 임시 operand를 놓친다. profiler의 실제 kernel, dispatch 전 op input·output hook, kernel trace를 결합한다. 단, hook이 graph break나 materialization을 일으켜 dispatch 자체를 바꿀 수 있으므로 관측을 켠 run과 성능 run을 분리한다. fixture는 같은 GoldenBatch에 대해 autocast off/on만 바꾸고 operator별 입력·출력 dtype, 선택 커널, 첫 번째 수치 차이를 기록한다.

### 14.1.2 loss scaling과 overflow

**scale의 수명과 optimizer commit**

loss에 `S`를 곱해 backward하면 local gradient는 `g_s=Sg`가 된다. `unscale_`은 이를 `g=g_s/S`로 되돌리는 연산인 동시에 non-finite 검사의 경계다. dynamic scaler는 finite 여부에 따라 scale과 growth tracker를 갱신한다. 중요한 것은 이것이 단순한 숫자 보정이 아니라 **optimizer commit을 허가하거나 거부하는 제어 상태**라는 점이다.

한 update를 다음 사건으로 기록한다.

| 사건 | 읽는 상태 | 쓰는 상태 | 통과 불변식 |
|---|---|---|---|
| scaled backward | loss, scale, RNG | scaled gradient | accumulation 중 scale generation이 바뀌지 않는다 |
| unscale·local check | scaled gradient, scale | unscaled gradient, local found-inf | optimizer별로 한 번만 unscale한다 |
| finite 합의 | rank별 found-inf | group found-inf | 참여 rank가 같은 commit 결정을 본다 |
| clip | unscaled global gradient | clip coefficient, clipped gradient | norm의 단위가 원래 gradient와 같다 |
| optimizer step | master, moment, clipped gradient, step | candidate master·moment | overflow이면 권위 state가 하나도 전진하지 않는다 |
| model view/cache 갱신 | committed master, old cache version | storage view, quantized cache, scale version | 소비할 parameter version과 cache version이 같다 |
| clock commit | group found-inf, update result | UpdateID, scheduler/scaler tracker | loop iteration이 아니라 성공한 update와 연결된다 |

분산 학습에서는 rank 0만 finite라고 step해서는 안 된다. 어느 rank에서든 non-finite가 발견되면 해당 optimizer와 process group이 정의한 정책에 따라 같은 커밋 결정을 내려야 한다. 이 합의가 없으면 parameter replica가 한 step 만에 갈라진다. 여러 optimizer가 있을 때는 found-inf가 optimizer별인지 전체 트랜잭션 공통인지 소스 계약을 확인한다. 둘을 임의로 섞으면 한 optimizer만 움직인 partial update가 생긴다.

overflow step에서는 parameter뿐 아니라 momentum, variance, optimizer step counter, EMA, scheduler, quantized weight cache와 `UpdateID`가 전진했는지 확인한다. scaler의 backoff는 실패한 update를 처리하기 위한 정당한 상태 변화지만 optimizer commit과는 구분한다. scaler state를 checkpoint에서 빼면 재개 직후 skip 패턴이 달라지고, growth interval 경계에서는 같은 GoldenBatch도 다른 scale generation을 읽을 수 있다.

**clipping 순서와 세 개의 음성 대조군**

scaled gradient를 먼저 clip하면 임계값 의미가 `S`배 달라진다. 올바른 관찰 지점은 unscale 뒤, optimizer step 전이다. FSDP에서는 전체 norm 계산에 collective가 필요할 수 있어 각 rank의 local norm을 전체 norm으로 오인하면 안 된다.

정상 fixture만으로 이 순서를 증명하지 않는다. 첫 음성 대조군은 unscale 전에 clip해 최종 clip coefficient와 parameter delta가 달라져야 한다. 둘째는 한 rank에만 Inf를 넣어 모든 rank의 parameter·moment hash가 그대로인지 확인한다. 셋째는 overflow인데 scheduler만 전진시키고 다음 성공 step의 learning rate와 delta가 uninterrupted control에서 갈라지는지 본다. 이 세 실험은 각각 단위 오류, finite 합의 오류, commit clock 오류를 분리한다.

NaN·overflow·underflow도 한 증상으로 묶지 않는다. overflow fixture는 큰 finite 입력을 점차 키워 최초 Inf가 생긴 operator를 찾는다. underflow fixture는 작은 gradient를 로그 스케일로 줄이며 zero 비율과 master delta를 본다. NaN fixture는 `Inf-Inf`, 잘못된 denominator, invalid normalization처럼 생성 원인이 다른 입력을 쓴다. FP32 eager→목표 dtype eager→fused→distributed 순으로 올라가 최초로 달라진 edge만 조사한다. seed를 고정했는데 실행마다 오차 부호가 달라지면 atomic·reduction tree·stochastic rounding RNG를 별 축으로 분리한다.

### 14.1.3 fused backward

**fusion이 저장 tensor와 반올림 위치를 바꾼다**

fused cross entropy는 full logits 확률 tensor를 남기지 않고 row-wise max와 denominator를 이용해 gradient를 만들 수 있다. FlashAttention backward는 attention matrix를 HBM에 저장하는 대신 output과 log-sum-exp를 바탕으로 tile을 재계산한다. fusion은 수학식만 같다고 충분하지 않다. mask, causal offset, dropout RNG, reduction dtype, stride를 reference와 비교한다.

**kernel 선택 옵션은 상태 전이표로 읽는다**

`torch.compile`, fused optimizer, SDPA backend 선택은 graph capture와 dispatch를 바꾼다. unsupported shape에서는 fallback될 수 있으므로 옵션이 켜졌다는 로그보다 실제 kernel name과 trace가 근거다. deterministic 옵션은 성능을 낮추거나 일부 kernel을 배제하며 모든 CUDA 연산의 bitwise 동일성을 약속하지 않는다.

각 옵션은 최소 네 종류로 분류한다. 수치 옵션은 input·accumulator·rounding을 바꾼다. dispatch 옵션은 같은 논리식을 다른 kernel로 보낸다. lifecycle 옵션은 capture, cache와 lazy allocation을 바꾼다. commit 옵션은 overflow skip과 optimizer state 갱신을 바꾼다. 한 옵션이 여러 열을 건드리면 checkpoint schema와 rollback 범위도 함께 바뀐다.

shape fixture는 fast path 하나가 아니라 guard 양쪽을 짝으로 둔다. 행렬 차원의 tile 배수 전후, aligned/offset pointer, contiguous/sliced stride, attention head dimension 경계, all-masked row를 사용한다. 각 행에는 expected backend와 fallback 허용 여부를 미리 쓴다. CUDA launch 성공, memory-safety, numerical parity, backward parity, actual dispatch, performance를 독립 판정한다. fallback으로 reference가 통과한 결과를 optimized kernel의 correctness 증거로 승격하지 않는다.

### 14.1.4 CUDA 12/13 compatibility

**네 층을 분리한다**

driver가 지원하는 CUDA runtime 범위, wheel이 포함한 runtime library, extension을 컴파일한 toolkit, GPU compute capability를 분리한다. PTX JIT 가능성과 실제 cubin 포함 여부도 다르다. CUDA major를 올렸다고 kernel이 자동으로 빨라지지 않으며 compiler·CUTLASS/Triton·PyTorch 조합이 같은지부터 확인한다.

**실험·디깅·handoff**

동일 batch에서 reference FP32와 저정밀 경로의 loss, max/mean logit error, gradient cosine, overflow count를 기록한다. 성능은 tokens/s와 peak memory뿐 아니라 compile time·fallback count를 함께 본다. `invalid device function`은 arch list/cubin, illegal memory access는 가장 먼저 실패한 kernel과 shape, NaN은 최초 non-finite tensor를 추적한다.

**FP8은 scale state를 학습 생명주기에 넣는다**

E4M3은 정밀도를, E5M2는 범위를 더 확보하는 방향으로 쓰인다. tensor를 FP8로 cast하려면 실수 범위를 representable range에 맞추는 scale이 필요하다. delayed scaling은 이전 amax history에서 다음 scale을 정하므로 history, recipe, update interval이 checkpoint state다. per-tensor와 block scaling은 metadata 수와 kernel layout이 다르다.

forward activation, weight, backward gradient에 같은 format을 강제하지 않는다. Transformer Engine류 recipe는 연산별 FP8 사용과 higher-precision accumulation을 조합한다. 첫/마지막 layer, normalization, reduction을 높은 정밀도로 유지하는 선택은 정확도와 kernel coverage를 바꾼다. FP8 가능 GPU라는 사실만으로 모든 op가 FP8 kernel을 탔다고 쓰지 않는다.

overflow·underflow 시험은 amax가 갑자기 100배 커지는 synthetic tensor, 매우 작은 gradient, history reset resume을 포함한다. FP8 run과 BF16 reference의 layer별 output·gradient error, scale, saturation/zero 비율을 기록한다. checkpoint load 뒤 첫 scale과 첫 output이 uninterrupted reference와 맞는지 본다.

**fused kernel의 수치 계약**

fused CE는 `[B,T,V]` logits 전체를 저장하지 않고 tile별 max와 exp sum으로 log-sum-exp를 만들 수 있다. test는 ignore index, all-masked row, 큰 logit, vocab shard를 포함한다. loss sum과 denominator를 reference FP32와 비교하고 backward gradient의 row sum invariant를 본다.

FlashAttention backward는 forward의 LSE와 RNG state를 사용해 tile을 재계산한다. causal·window·GQA head mapping·dropout seed·variable length가 reference와 같아야 한다. kernel 이름만 확인하지 말고 shape/dtype/layout과 실제 dispatch 이유를 기록한다. fallback은 correctness에는 성공해도 성능 실험을 무효로 만들 수 있다.

fused optimizer는 parameter·gradient·moment update를 한 kernel에 넣는다. overflow skip과 decay 순서, capturable step tensor, state dtype을 unfused reference와 한 step 비교한다. compiler fusion은 graph break 전후 kernel 수와 temporary memory를 trace한다.

**CUDA 12와 13을 비교하는 표**

비교 행은 CUDA major 하나가 아니라 driver, toolkit, wheel runtime, NVRTC/PTX, compiler, PyTorch/Triton/CUTLASS revision, compute capability다. CUDA 12로 빌드한 wheel을 새 driver에서 실행하는 compatibility와 CUDA 13 toolkit으로 extension을 다시 빌드하는 source compatibility를 분리한다. `torch.version.cuda`와 `nvcc --version`이 다른 것은 반드시 오류가 아니다.

CUDA 13 지원을 주장하려면 framework release의 공식 matrix와 extension build guard를 확인한다. DeepSpeed처럼 system toolkit과 PyTorch CUDA version을 비교하는 builder에서 skip-check 환경 변수는 검증이 아니라 위험한 우회다. Blackwell feature gate, arch list, PTX/cubin 포함을 artifact manifest에 둔다.

**실패 결정 트리**

import 실패는 shared library resolution과 ABI를, extension compile 실패는 compiler/toolkit/header/arch를, `no kernel image`는 cubin/PTX arch를 본다. illegal memory access는 첫 실패 kernel을 compute-sanitizer 또는 동기화된 최소 fixture로 좁힌다. NaN은 최초 non-finite layer, FP8 scale/saturation, loss scaler, reduction dtype 순서로 본다.

성능 regression은 실제 kernel dispatch, graph break, occupancy, HBM bytes, compilation cache를 비교한다. CUDA major가 원인이라고 결론내리기 전에 framework와 kernel revision을 고정한다. 같은 binary를 두 driver에서, 같은 source를 두 toolkit에서 빌드하는 실험을 분리한다.

## 14.2 오차 예산에서 migration failure까지

단일 연산의 오차가 작다는 사실만으로 한 step이 안전해지지는 않는다. Transformer Engine과 Torch가 소유하는 scale state, CUDA 세대 이행, failure injection을 한 오차 예산에 놓아 최초로 허용 범위를 벗어난 경계를 찾는다.

### 14.2.1 floating-point error budget을 분해한다

한 layer의 오차를 storage quantization, input cast, multiply, accumulator, output cast, collective reduction으로 나눈다. BF16 parameter를 FP32 accumulator GEMM에 쓰는 경로와 BF16 accumulator 경로는 이름이 같은 BF16 training이어도 다르다. reduction tree 순서가 바뀌면 FP32에서도 rounding 차이가 난다. bitwise와 numerical-equivalent를 구분한다.

unit roundoff만 나열하지 않고 실제 tensor scale을 본다. activation/gradient histogram, zero/subnormal/saturation, max/percentile를 layer별로 기록한다. relative error는 true value가 0에 가까울 때 폭발하므로 max absolute, RMS, cosine, task-level delta를 함께 쓴다.

**dynamic loss scaler state machine**

scaler state에는 current scale, growth tracker, growth/backoff factor, growth interval이 들어간다. `scale(loss)`가 graph에 multiplier를 넣고 `unscale_(optimizer)`가 gradient를 되돌리며 finite 검사를 한다. `step`은 overflow면 optimizer를 건너뛰고 `update`가 scale을 낮춘다. optimizer가 여러 개면 unscale/step 호출과 found-inf 결합 계약을 확인한다.

scale을 너무 크게 시작한 run은 여러 update를 skip할 수 있다. scheduler가 committed update가 아니라 loop iteration을 세면 lr가 소모된다. checkpoint에서 scaler를 빼면 resume 첫 skip pattern이 달라진다. growth interval 경계 직전 저장/복구를 test한다.

**FP8 scale recipe**

FP8 cast를 `q=clip(round(x·s),range)`, dequantize를 `q/s`로 생각하면 scale 선택이 오차와 saturation을 결정한다. current scaling은 현재 amax를 사용해 동기화가 필요할 수 있고 delayed scaling은 history로 latency를 줄이는 대신 stale scale 위험이 있다. amax history length, margin, update algorithm, per-tensor/block granularity를 recipe ID로 묶는다.

DP/TP rank가 서로 다른 scale을 쓰면 collective 또는 matmul 의미가 달라질 수 있다. scale/amax reduction owner를 확인한다. scale state checkpoint에서 rank shard인지 replicated인지, topology 변경 때 어떻게 합치는지 적는다. history 누락을 zero-init하면 resume 첫 layer output이 달라질 수 있다.

### 14.2.2 Transformer Engine과 Torch 경계

framework autocast가 FP8 recipe를 자동으로 의미하지 않는다. FP8-aware module 또는 context가 weight/activation cast와 amax update를 삽입하고 지원 kernel로 dispatch한다. unsupported op는 BF16/FP16으로 남을 수 있다. module conversion 전후 parameter class, extra scale state, forward output dtype를 dump한다.

upstream test는 supported shape, recipe, save/load를 확인할 수 있지만 자신의 model fusion·distributed topology까지 보장하지 않는다. golden layer를 BF16 reference와 비교하고 full model에서는 최초 divergence와 scale anomaly를 찾는다.

**compiler pipeline을 네 단계로 본다**

Python graph capture/IR 생성, high-level fusion, device code 생성(PTX 또는 intermediate), assembler/link와 cubin 생성, driver load/JIT를 분리한다. `torch.compile` graph break는 첫 단계, Triton lowering 오류는 codegen, `ptxas` 오류는 toolkit/arch, `no kernel image`는 binary coverage 문제다.

compile cache key에는 graph/shape guards, dtype, device capability, compiler/runtime revision이 영향을 준다. 다른 GPU에서 cache artifact를 무조건 재사용하지 않는다. dynamic shape recompilation 횟수와 cold/warm latency를 성능 보고서에 분리한다.

**CUDA compatibility의 네 관계**

첫째 driver와 runtime compatibility, 둘째 application/wheel이 포함한 CUDA user-space library, 셋째 extension을 만든 toolkit/compiler, 넷째 cubin/PTX가 지원하는 compute capability다. `nvidia-smi`가 보여주는 “CUDA Version”은 driver가 지원하는 상한 정보이지 현재 process가 그 toolkit으로 빌드됐다는 뜻이 아니다.

minor-version compatibility와 forward compatibility의 조건은 NVIDIA 공식 compatibility guide에서 확인한다. PTX JIT는 future architecture 가능성을 줄 수 있지만 포함된 PTX version을 driver가 이해해야 한다. SASS cubin만 있고 target SM이 없으면 새 GPU에서 실행되지 않는다. 반대로 toolkit이 새로워도 framework ABI가 지원하지 않으면 extension build가 실패한다.

### 14.2.3 CUDA 12→13 migration manifest

migration 전 baseline은 container digest, driver, toolkit, PyTorch/TE/Triton/CUTLASS, compiler, arch list, loaded shared libraries, kernel trace다. CUDA 13 environment에서는 한 번에 framework까지 모두 올리지 않고 가능한 supported 조합 단위로 바꾼다. 공식 지원 matrix 밖의 조합은 experimental로 표시한다.

extension은 clean build하고 compile command와 generated arch를 보존한다. unit test, numerical golden batch, distributed collective, checkpoint load, compile cache cold/warm, throughput을 순서대로 실행한다. 실행하지 않은 GPU architecture는 지원으로 쓰지 않는다.

**CUDA Graph와 capturable state**

CUDA Graph는 안정적인 memory address와 반복 가능한 control flow를 요구한다. optimizer `capturable`, static buffer, step tensor device placement이 필요할 수 있다. dynamic loss scale overflow branch, variable sequence shape, lazy state initialization은 capture를 깨뜨릴 수 있다. warmup에서 state와 workspace를 materialize한다.

graph replay에서 input/output buffer reuse를 stream event 없이 하면 이전 request와 race가 난다. eager와 graph replay의 logits/gradient/update를 비교한다. graph가 실제 replay됐는지 profiler marker와 launch pattern으로 확인한다.

**kernel compatibility fixture**

fixture는 matmul, norm, CE, attention forward/backward, fused optimizer를 대표 shape로 실행한다. 각 항목은 input shape/stride/alignment/dtype, selected kernel, output tolerance, backward tolerance, workspace를 기록한다. odd sequence, noncontiguous, GQA, all-masked, very large logit처럼 dispatch 경계 shape를 포함한다.

CUDA 12와 13 결과 비교에서 compiler가 다른 연산 순서를 선택할 수 있으므로 bitwise를 기본 요구하지 않는다. 허용오차를 BF16/FP8 경로별 사전 정의한다. kernel fallback은 correctness PASS와 performance FAIL을 분리한다.

### 14.2.4 failure injection

잘못된 arch list로 extension을 빌드해 load-time 실패가 명확한지 본다. PTX를 제거한 binary, 오래된 driver, missing shared library를 각각 분리한다. FP8 history를 삭제하고 resume output drift를, loss scaler counter를 rollback해 skip drift를 만든다. fused CE에 all-ignored labels를 넣어 denominator 0 처리를 확인한다.

attention test에서는 causal offset 하나, GQA mapping, dropout seed를 바꿔 reference가 실패하는지 확인한다. illegal access 실험은 격리된 test process와 최소 shape에서 수행하며 production GPU에 무제한 fault를 주지 않는다.

**성능 진단**

tokens/s가 떨어지면 compile/recompile 시간, CPU launch gap, kernel duration, occupancy, memory throughput, collective overlap으로 분해한다. 새 CUDA에서 kernel 이름이 달라졌는지, Tensor Core instruction을 쓰는지 profiler로 본다. GPU utilization 평균은 graph break와 작은 kernel gap을 숨길 수 있다.

FP8이 BF16보다 느리면 cast/amax overhead, unsupported fallback, 작은 shape, scale collective를 본다. memory는 줄었는데 throughput이 그대로면 병목이 network/data 또는 non-FP8 op일 수 있다. theoretical FLOPS만으로 결론내리지 않는다.

**재현·release 표**

각 binary artifact에 source revision, compiler/toolkit, compile flags, target SM/PTX, dependency ABI, content hash를 기록한다. runtime record에는 driver, loaded library path, device capability, actual kernel dispatch를 둔다. 이 둘을 연결해야 “어떤 소스로 어떤 GPU에서 무엇이 실행됐는가”를 답할 수 있다.

release gate는 build success, unit correctness, golden numerical tolerance, no unexpected fallback, distributed smoke, checkpoint resume, performance budget을 각각 판정한다. CUDA 12 build와 CUDA 13 build를 별 artifact로 보존하고 rollback 가능성을 확인한다.

**오차 경계별 결정 트리**

빌드 전 실패는 support matrix와 compiler/ABI, load 실패는 shared library와 binary arch, 첫 kernel 실패는 shape/alignment/device code, numerical divergence는 최초 layer와 scale/reduction, 성능 regression은 dispatch/trace를 본다. 원인을 확인하기 전에 여러 환경 변수를 동시에 바꾸지 않는다.

NaN이 FP8에서만 나면 amax/saturation/history를 BF16 reference와 비교한다. fused에서만 나면 unfused op별 checkpoint로 좁힌다. distributed에서만 나면 scale과 gradient reduction dtype/denominator를 본다. resume에서만 나면 scaler·FP8 history·compiler cache를 의심하기 전에 실제 state부터 비교한다.

**end-to-end numerical ladder**

같은 GoldenBatchID를 FP32 eager, BF16 eager, BF16 fused, FP8, compiled FP8 순서로 실행한다. 각 rung은 바로 이전 rung과 비교해 추가한 변화의 error budget을 계산한다. FP32→BF16은 dtype, BF16 eager→fused는 algorithm schedule, fused→FP8은 scaling, FP8→compiled는 compiler/fusion 차이다.

forward loss, layer summary, gradient cosine, parameter delta를 비교한다. 어느 rung에서 threshold가 깨졌는지 찾으면 여러 최적화를 한꺼번에 끄지 않아도 된다. performance도 같은 ladder에서 kernel 수, HBM peak, tokens/s를 기록한다.

**공식 compatibility 자료 읽는 법**

CUDA release note는 toolkit component와 제거·변경 사항, compatibility guide는 driver/runtime 관계, programming guide는 execution/memory semantics를 다룬다. framework support matrix는 별도다. 한 문서의 “지원”을 다른 계층으로 확장하지 않는다.

CUDA 12.x minor끼리도 bundled library와 compiler가 다를 수 있고 CUDA 13.x에서는 deprecated/removed API와 new architecture support가 변할 수 있다. 정확한 `12.x.y`, `13.x.y`와 component version을 적는다. 최신이라는 표현 대신 검증한 조합을 쓴다.

## 14.3 extension·ABI·shape 경계를 release 전에 검증한다

수치 fixture가 통과해도 extension이 다른 shared library를 읽거나 shape별 fallback이 달라지면 배포 결과는 갈린다. build provenance, ABI, kernel shape와 복구 종료 조건을 release gate 앞에서 함께 검증한다.

### 14.3.1 extension build audit

build log에서 include/library path, compiler, C++ ABI flag, `-gencode`, PTX/SASS target을 추출한다. wheel의 shared objects를 검사해 필요한 symbol과 arch가 들어 있는지 확인한다. runtime JIT가 발생하면 cache path와 cold latency를 기록한다.

source conditional compile이 CUDA version macro나 compute capability로 다른 kernel을 선택하는지 본다. 새 toolkit에서 compile됐지만 feature gate가 old path를 탔다면 CUDA 13 전용 성능을 검증한 것이 아니다.

**distributed precision.**

gradient/parameter collective dtype와 reduction accumulator를 기록한다. BF16 reduce와 FP32 reduce는 bytes와 error가 다르다. FP8 communication은 scale metadata와 collective 지원을 요구한다. rank별 scale이 다르면 quantized value를 그대로 합할 수 없다.

single-rank reference와 N-rank logical gradient를 비교하고 world size 증가에 따른 error를 본다. reduction order 차이를 tolerance에 포함하되 systematic scale 오류를 rounding으로 넘기지 않는다.

**checkpoint portability.**

BF16 model checkpoint는 CUDA toolkit과 독립적일 수 있지만 fused optimizer opaque state, FP8 amax history, compiled graph cache는 portability가 다르다. portable logical state와 rebuildable cache, hardware-specific binary를 분리 저장한다.

CUDA 12 환경 checkpoint를 CUDA 13 환경에서 load할 때는 eager BF16 reference부터 복원한다. 그다음 FP8 state, fused optimizer, compile을 단계적으로 켠다. 처음부터 full optimized path로 load하면 incompatibility 위치를 알기 어렵다.

### 14.3.2 release failure table

`build`, `load`, `dispatch`, `numerics`, `distributed`, `resume`, `performance` 일곱 gate를 둔다. build가 통과해도 unsupported fallback이면 dispatch fail, logits가 맞아도 target throughput 미달이면 performance fail이다. 각 gate의 log와 artifact hash를 연결한다.

rollback은 옛 container만 띄우는 것이 아니라 compatible driver와 model/checkpoint state를 확인한다. 새 run이 쓴 hardware-specific optimizer state를 옛 binary가 읽을 수 있는지 검증한다.

**FP8 block scaling 수치 예.**

한 block 값이 `[0.01,0.02,1,100]`이면 큰 outlier가 공통 scale을 지배해 작은 값의 유효 정밀도를 잃게 한다. block을 둘로 나누면 작은 값 block은 더 적합한 scale을 쓸 수 있지만 scale metadata와 kernel indexing이 늘어난다. per-tensor, block-128, block-32의 saturation·zero 비율을 비교한다.

amax history가 outlier를 여러 step 기억하면 이후 정상 batch에서도 scale이 보수적일 수 있다. history length와 margin을 바꾼 실험에서 즉시 saturation과 장기 under-utilization을 함께 본다. resume 때 history reset negative test가 민감한지 확인한다.

**PTX와 cubin inspection.**

binary inspection 도구로 extension의 target SM과 PTX section을 확인한다. build log의 `-gencode`와 실제 artifact가 맞는지 비교한다. target GPU용 cubin이 있으면 direct load, PTX만 있으면 driver JIT 가능성과 cache를 확인한다. 어느 것도 없으면 `no kernel image`가 정상 실패다.

새 architecture에서 JIT 성공만으로 performance support를 선언하지 않는다. architecture-specific instruction과 tuning이 없는 generic PTX일 수 있다. actual kernel trace, instruction mix, benchmark를 별 gate로 둔다.

### 14.3.3 ABI·shared-library 반례

같은 CUDA major라도 cuBLAS/cuDNN/NCCL와 C++ ABI 조합이 달라 symbol resolution이 실패할 수 있다. `ldd`/loader trace로 process가 실제 읽은 library path를 저장한다. host에 설치된 toolkit library와 wheel bundled library가 섞이지 않는지 본다.

import가 성공해도 lazy-loaded kernel library가 첫 op에서 실패할 수 있다. smoke test는 import, device init, 대표 op, backward, collective까지 단계화한다. 실패 단계별 error를 release report에 남긴다.

**CUDA 12 checkpoint를 13에서 복구한다.**

portable model tensor와 tokenizer부터 CUDA 13 eager BF16에서 load해 fixed-token logits를 비교한다. 그다음 optimizer logical state, GradScaler, FP8 history를 load한다. fused extension과 compiled graph cache는 재빌드한다. 각 단계마다 다음 update를 reference와 비교한다.

opaque backend state가 version migration을 지원하지 않으면 parameter-only branch로 시작하고 optimizer reset을 선언한다. 파일이 deserialize된다는 이유로 state semantics가 같다고 보지 않는다. rollback을 위해 CUDA 12 compatible checkpoint parent를 보존한다.

**compiler optimization 반례.**

fast-math, reduced-precision reduction, TF32 flag를 한 번에 켜지 않고 각각 비교한다. 작은 toy에서 algebraically equivalent expression이 rounding과 overflow 순서 때문에 달라질 수 있다. compiler가 re-association한 reduction은 bitwise reference와 다를 수 있으나 사전 tolerance와 invariant를 만족해야 한다.

graph fusion이 dropout RNG 소비 순서를 바꾸는지 seed fixture로 본다. checkpoint recompute와 fused dropout이 같은 mask 계약을 쓰지 않으면 backward가 달라진다. RNG offset을 kernel state로 관찰한다.

**kernel shape boundary test**

head dimension 64/128 같은 fast path뿐 아니라 80, sequence 1, very long sequence, non-multiple tile, GQA ratio, noncontiguous stride를 넣는다. supported guard가 fallback 또는 명확한 error를 내야 한다. 잘못된 specialized kernel로 들어가 silent corruption이 나면 가장 위험하다.

fused optimizer는 empty gradient group, sparse `None`, odd tensor count, large step counter를 시험한다. CE는 vocab shard 경계와 ignore index, attention은 zero-length K와 causal offset을 시험한다.

**multi-GPU precision recovery.**

rank별 scaler found-inf와 FP8 amax가 global policy에 맞게 합쳐지는지 확인한다. rank 하나의 scale history를 손상시키고 layer output hash가 달라지는지, publication 전에 fail-fast하는지 본다. collective dtype mismatch는 hang 또는 corruption 전에 metadata assertion으로 잡는다.

resume topology가 바뀌면 replicated amax history가 같은지, sharded weight scale이 새 shard와 맞는지 mapping report를 만든다. scale tensor shape가 맞는 것만으로 logical weight slice 대응을 증명하지 못한다.

**공식 문서 좌표를 보고서에 남긴다.**

compatibility guide의 표, toolkit release note의 component version, programming guide의 architecture/precision 절, framework 공식 install matrix를 각각 URL·문서 version·접근일로 기록한다. 블로그 benchmark를 support contract로 사용하지 않는다.

source claim은 고정 commit의 build guard와 dispatch 함수, test claim은 test name과 assertion을 쓴다. 실제 GPU execution은 별 실행 record다. 세 근거를 한 문장에 섞지 않으면 “코드에 branch가 있음”과 “이 환경에서 검증됨”을 구분할 수 있다.

**migration 복구 리허설**

rehearsal은 CUDA 12 baseline의 GoldenBatchID, checkpoint, kernel trace를 고정한다. CUDA 13 candidate를 build하고 BF16 eager→fused→FP8→compiled ladder를 순서대로 통과한다. 각 rung failure에서 이전 rung은 유지해 원인을 격리한다.

장애로 old driver, missing cubin, stale compile cache, reset FP8 history, partial optimizer state를 넣는다. release gate가 각각 build/load/dispatch/numerics/resume에서 정확히 닫히는지 본다. 모든 gate 통과 뒤에만 throughput을 비교한다.

**handoff의 byte 계산.**

15장에 parameter·gradient·optimizer·scale state의 dtype과 bytes를 넘긴다. collective payload가 BF16인지 FP32인지, FP8이면 scale metadata가 누가 소유하는지 명시한다. FSDP shard와 TP weight slice가 quantization block 경계를 자르는지도 검사한다.

17장 checkpoint에는 GradScaler, amax history, quantization metadata, portable tensor와 rebuildable binary/cache 구분을 넘긴다. 이 계약이 있어야 topology 또는 CUDA version 변경 후 어떤 state를 복원하고 무엇을 재생성할지 결정할 수 있다.

**승인용 허용오차 원장**

각 rung은 tensor 이름, reference/candidate dtype, max·RMS error, cosine, threshold, 판정을 남긴다. threshold는 FP32→BF16, BF16→FP8, eager→fused에 따로 둔다. 여러 변환의 총 오차만 보면 어느 단계가 예산을 초과했는지 알 수 없다.

성능 원장도 cold compile, warm replay, kernel, collective를 분리한다. numerical PASS와 performance PASS는 독립 gate다. 실제 dispatch와 binary hash가 없는 benchmark는 release 근거가 아니다.

**복구 종료 조건**

CUDA 13 candidate에서 portable checkpoint, scaler, FP8 history를 복원하고 첫 GoldenBatchID의 logits·gradient·delta를 비교한다. compiled cache는 재생성한다. 실패하면 마지막 통과 rung으로 rollback하고 원인을 기록한다.

이 결과와 dtype/byte manifest가 다음 장의 shard·collective 설계를 고정한다.

**고정 실행 좌표의 최소 단위.**

PyTorch `550d7b3834670483a4df436541272c055dc364bf`에서 autocast·GradScaler·SDPA dispatch의 실제 symbol과 test를 각각 고정한다. CUDA 공식 문서는 toolkit `12.x.y`와 `13.x.y` release note, compatibility guide revision, programming guide section을 기록한다. driver 상한 표와 toolkit build support를 같은 좌표로 인용하지 않는다.

extension은 source commit, build file의 version/arch guard, dispatch function, correctness test 네 좌표를 한 묶음으로 둔다. 실제 candidate GPU trace가 없으면 kernel이 실행됐다고 쓰지 않는다.

candidate 환경의 첫 실행은 actual loaded library와 kernel dispatch를 기록한다. manifest의 기대 binary와 다르면 numerical test 전에 실패한다. 이 검사가 host library 혼입을 막는다.

**사례 연구: CUDA 12 기준선을 CUDA 13 후보로 옮긴다.**

사례는 같은 GPU가 두 toolkit build를 모두 지원한다는 전제에서 시작한다. driver가 CUDA 13 runtime을 지원하는지, compiler가 target compute capability code를 생성하는지, extension과 framework wheel이 해당 toolkit ABI를 지원하는지 각각 확인한다. `nvidia-smi`가 보여주는 “CUDA Version” 한 줄을 installed toolkit compiler와 동일시하지 않는다.

baseline manifest에는 GPU name/SM, driver, toolkit/nvcc, runtime library path, framework build CUDA, cuDNN/NCCL, compiler, extension commit과 build flags를 기록한다. candidate도 같은 schema를 쓴다. 실제 loaded `.so`와 kernel binary hash를 trace한다. shell의 `nvcc --version`만 맞고 process가 old library를 load하는 혼입을 막는다.

## 14.4 CUDA 세대와 FP8 실행 상태를 함께 읽는다

CUDA 12.x와 13.x의 차이는 버전 번호 하나로 판정하지 않는다. compiler graph·cache key·FP8 metadata·실행 library를 구성요소별로 고정하고, 성능 incident와 공식 문서의 지원 문장을 같은 증거표에 연결한다.

### 14.4.1 CUDA 12.x와 13.x 차이를 읽는 순서

major toolkit 변경은 language/compiler, runtime/driver compatibility, library, architecture support와 deprecation을 나눠 읽는다. 공식 release note의 component version과 known issue, compatibility guide의 최소 driver, programming guide의 feature contract를 서로 바꾸어 인용하지 않는다. minor `12.x.y`, `13.x.y`까지 build manifest에 넣는다.

CUDA 13에서 오래된 architecture의 offline compilation support가 달라질 수 있다. nvcc `-gencode`가 성공하는지와 driver JIT PTX가 실행 가능한지는 다른 경로다. fatbin에 cubin/PTX가 무엇으로 들어갔는지 검사한다. target GPU에서 actual loaded image와 JIT event를 확인한다.

framework가 CUDA 13 wheel을 제공한다는 사실은 모든 third-party extension이 준비됐다는 뜻이 아니다. FlashAttention, Transformer Engine, bitsandbytes, custom fused optimizer의 build guard, included cubin과 test matrix를 각각 본다. unsupported extension fallback이 eager PyTorch로 조용히 바뀌면 correctness는 맞아도 성능 결과가 다른 backend다.

**precision ladder를 한 단계씩 오른다.**

첫 rung은 FP32 eager, 둘째 BF16 eager, 셋째 BF16 fused, 넷째 FP8 eager/fused, 다섯째 compiled graph다. 각 rung은 같은 GoldenBatch, parameter/checkpoint와 loss denominator를 쓴다. 바로 FP8 compiled candidate와 CUDA 12 baseline을 비교하면 precision, fusion, compiler와 toolkit 변화가 한꺼번에 섞인다.

각 edge는 logits, loss sum/count, layer activation 선택점, gradient와 first delta의 max/RMS error와 cosine을 기록한다. BF16→fused는 동일 dtype에서 kernel semantics를, fused→FP8은 representation과 scale을, eager→compiled는 graph lowering을 주로 격리한다. 예상 밖 edge에서 최초 divergence를 찾는다.

성능도 cold compile, warm replay, kernel critical path, end-to-end step으로 나눈다. compiled 첫 실행 시간을 warm throughput에 섞지 않는다. numerical gate가 통과한 rung만 성능 후보가 된다. fallback backend는 실제 dispatch 이름으로 별 행에 둔다.

**BF16·FP16의 수치 fixture.**

FP16은 BF16보다 fraction bit는 많지만 exponent range가 좁다. 같은 큰 activation/gradient에서 FP16 overflow와 BF16 finite를 작은 tensor로 재현한다. 반대로 작은 상대 차이에서는 BF16 rounding이 더 클 수 있다. “BF16이 항상 정확하다”거나 “FP16이 항상 낫다”고 쓰지 않는다.

GradScaler fixture는 정상 step, overflow step, 다음 회복 step을 가진다. overflow에서 optimizer parameter/state와 committed scheduler가 움직이지 않고 scale/backoff만 정책대로 바뀌는지 확인한다. checkpoint save/load 뒤 scale, growth tracker와 다음 skip 판정이 uninterrupted control과 같아야 한다.

unscale 이전 gradient norm과 이후 norm을 구분한다. clipping은 unscale 뒤 global gradient에 적용해야 한다. wrong-order negative control은 clipping coefficient와 final delta mismatch를 만들어야 한다. loss가 finite하다는 이유로 통과하지 않는다.

### 14.4.2 FP8 format과 scale state

FP8은 하나의 dtype이 아니다. E4M3와 E5M2는 exponent/fraction tradeoff와 사용 역할이 다르다. forward activation/weight와 backward gradient에 어떤 format을 쓰는지 recipe에 적는다. 저장 tensor, compute input, accumulator dtype도 분리한다. tensor core가 FP8 input을 받아도 accumulation과 output이 더 높은 dtype일 수 있다.

FP8 tensor는 값 code만으로 의미가 완전하지 않고 scale과 amax history가 동적 quantization state를 소유할 수 있다. per-tensor, per-channel/block scaling은 metadata shape와 kernel contract를 바꾼다. delayed scaling은 과거 amax window와 update interval이 checkpoint state다.

toy fixture는 작은 값, 정상 분포, outlier를 가진다. same codebook에서 scale을 바꿔 saturation count, zero/underflow, reconstruction error를 측정한다. amax history를 reset한 resume negative control이 first logits/gradient를 바꾸는지 본다. parameter만 restore한 checkpoint를 완전하다고 하지 않는다.

**FP8 source와 test 지도.**

선택한 Transformer Engine 또는 framework FP8 stack의 고정 commit에서 autocast/context, recipe config, amax/scale update, FP8 linear dispatch, checkpoint state를 잇는다. public API만 읽고 실제 kernel과 state owner를 추정하지 않는다. hardware capability guard와 fallback branch를 포함한다.

upstream test는 supported SM/dtype, forward/backward tolerance, scale update와 serialization 가운데 무엇을 assert하는지 표로 둔다. local GoldenBatch는 model의 mask/loss denominator와 resume를 검증한다. 대형 multi-GPU collective FP8을 실행하지 않았다면 proposed다.

실제 trace에서 kernel name, input/accumulator dtype, launch shape를 확인한다. config가 FP8이어도 unsupported shape가 BF16 fallback일 수 있다. fallback 비율을 layer/shape별로 보고한다. config 이름을 benchmark label로 쓰지 않는다.

**fused kernel의 negative control.**

eager reference와 fused candidate는 동일 mask, causal boundary, dropout seed와 reduction denominator를 써야 한다. attention에서는 all-masked row, odd length, non-contiguous input, large head dim과 causal offset을 fixture로 둔다. MLP는 gated activation과 bias 유무, tail shape를 넣는다.

wrong stride, mask orientation, sequence length 하나를 교란해 test가 실패하는지 본다. padding 값만 바꾸었는데 valid output이 변하면 mask/reading bug다. fused kernel이 output은 맞지만 backward gradient를 틀릴 수 있으므로 forward와 backward를 따로 판정한다.

atomic/reduction order 때문에 bitwise가 어려우면 dtype별 tolerance를 사전에 정한다. 평균 error만 보지 않고 max, RMS, cosine과 task-critical token을 본다. NaN/Inf count는 tolerance 밖 별 invariant다.

### 14.4.3 compiler graph와 cache key

`torch.compile`류 경로는 graph capture, decomposition, code generation, autotune/cache를 가진다. dynamic shape, control flow와 mutation이 graph break를 만들 수 있다. compile 성공 message보다 graph count, break reason, generated kernel과 fallback을 trace한다.

cache key에는 framework/compiler commit, toolkit, driver 영향 범위, GPU architecture, source/config, shape/stride/dtype와 flags가 들어가야 한다. old CUDA 12 compile cache를 CUDA 13 candidate에서 재사용하는 장애를 넣고 invalidation이 작동하는지 본다. cache hit가 correctness identity를 우회하지 않게 binary manifest를 검증한다.

cold compile과 warm replay를 분리하고 shape churn으로 recompilation이 발생하는지 본다. training sequence length bucket이 많으면 compile cost와 cache pressure가 throughput을 잠식할 수 있다. best static shape kernel 하나를 전체 workload 속도로 일반화하지 않는다.

**extension build와 binary inspection.**

extension artifact에는 source commit, submodule, compiler/nvcc flags, target arch, ABI macro와 linked library를 기록한다. build log와 resulting `.so` hash를 보존한다. environment에서 즉석 build한 파일을 source revision 없이 cache하지 않는다.

`-gencode`와 fatbin inspection으로 cubin/PTX target를 확인한다. missing cubin에서 PTX JIT가 가능한지, prohibited old architecture인지 candidate GPU에서 load test한다. build 성공과 kernel 실행은 별 gate다. wrong binary를 `LD_LIBRARY_PATH`로 먼저 load하는 negative control을 manifest mismatch가 잡아야 한다.

extension upgrade는 source API뿐 아니라 state/checkpoint schema와 kernel numerical test를 다시 수행한다. CUDA compiler가 바뀌면 same source에서도 binary와 numerical/performance evidence가 새 RunID다.

**CUDA 12→13 migration fault matrix.**

첫 fault는 old driver다. candidate process가 시작 전 compatibility gate에서 실패해야 한다. 둘째 missing target cubin/PTX로 load/dispatch error를 명확히 낸다. 셋째 stale compile cache로 binary key mismatch를 만든다. 넷째 old runtime library path 혼입을 loaded-library audit가 잡는다.

다섯째 FP8 amax history를 삭제한다. loader가 numerical-resume 등급을 거부하거나 명시한 fresh-calibration branch를 만든다. 여섯째 GradScaler만 한 step 오래된 state를 넣어 resume delta mismatch를 확인한다. 일곱째 fused kernel mask를 교란한다.

각 fault는 build, load, dispatch, numerical, resume 어느 gate에서 닫혀야 하는지 정한다. process가 fallback으로 살아난 것을 migration 성공으로 세지 않는다. fallback을 허용하면 backend identity와 performance result를 새 행으로 기록한다.

**incident/RCA: candidate만 느리다**

첫째 actual dispatch와 fallback 비율을 비교한다. extension build 실패나 unsupported shape로 eager path가 늘었을 수 있다. 둘째 compile cold/recompile과 warm kernel을 분리한다. 셋째 kernel은 빨라도 graph break와 host sync가 critical path를 늘렸는지 본다.

메모리 peak 증가가 batch를 줄였거나 allocator pressure를 만들었는지 본다. FP8 scale update/transpose workspace와 fused temporary를 phase별로 측정한다. toolkit library heuristic이 다른 algorithm을 골랐다면 workspace와 selected implementation을 trace한다.

수정은 한 edge씩 한다. compile cache만 비우고 warm replay, extension만 rebuild하고 eager/fused parity처럼 변수를 격리한다. CUDA 13이 느리다는 전체 결론 전에 최초 backend divergence를 찾는다.

**incident/RCA: resume 뒤 FP8만 발산한다**

GoldenBatch token과 BF16 path가 같으면 FP8 recipe, amax history, scale, update interval을 비교한다. first layer FP8 input/output과 scale을 uninterrupted control과 맞춘다. history window 순서와 owner rank가 checkpoint에서 바뀌지 않았는지 본다.

world size 변경이 per-rank scale 관측과 reduction을 바꿀 수 있다. scale state를 replicated/global로 정의했는지 source와 manifest에서 확인한다. portable tensor를 restore한 뒤 compiled kernel cache는 재생성하되 scale state는 재초기화하지 않는다.

optimizer/scaler/scheduler도 같은 cut인지 확인한다. FP8 failure처럼 보여도 GradScaler 또는 lr one-step offset일 수 있다. rung ladder에서 BF16 resume가 통과하고 FP8 edge에서 처음 갈리는지 확인한다.

**performance와 numerical 원장**

한 행은 environment, rung, actual backend, shape/dtype, cold/warm, loss/logit/gradient error, kernel/step time, peak bytes를 가진다. numerical PASS와 performance PASS는 독립이다. throughput은 같은 valid token workload에서 측정한다.

FP8 speedup은 FP8 kernel이 차지한 비율, fallback, scaling overhead를 함께 낸다. attention/MLP만 FP8이고 embedding/loss가 BF16이면 전체 speedup은 Amdahl 비율에 제한된다. theoretical tensor throughput을 end-to-end와 동일시하지 않는다.

CUDA 12와 13 결과에는 driver/toolkit/framework/extension binary를 완전히 표시한다. 하나의 숫자만 바뀐 비교인지 확인한다. candidate에서 다른 framework commit까지 올렸다면 compound migration으로 분류하고 ladder를 더 세분한다.

**observability와 운영 gate**

dashboard에는 loaded CUDA/library, actual kernel backend, compile/recompile, FP8 saturation/amax/scale, GradScaler skip, NaN/Inf, step latency와 peak를 표시한다. kernel name 같은 고카디널리티는 bounded mapping이나 trace에 둔다. RunID와 GoldenBatch로 상세 artifact를 찾는다.

alert는 unexpected fallback, binary hash mismatch, compile storm, FP8 saturation 증가, scaler skip burst, numerical rung threshold 초과를 잡는다. GPU utilization만으로 migration을 판정하지 않는다. correctness alert는 성능 SLO보다 우선한다.

rollback은 마지막 통과 rung과 portable checkpoint를 사용한다. compiled cache와 binary는 environment에 맞게 재생성한다. CUDA 12로 돌아갈 때 CUDA 13 전용 checkpoint state가 portable한지 manifest에서 판정한다. 변환기 없이는 자동 load하지 않는다.

**독자 인수 시험**

독자는 CUDA compatibility manifest를 채우고 process loaded library와 binary target를 검증한다. FP32→BF16→fused→FP8→compiled ladder에서 같은 GoldenBatch error ledger를 만든다. GradScaler overflow와 FP8 history reset negative test를 실행한다.

old driver, stale cache, wrong library, missing cubin, mask bug를 주입해 예상 gate가 실패하는지 본다. profiler에서 actual kernel과 fallback, cold/warm, temporary critical path를 찾는다. checkpoint resume first logits/gradient/delta를 uninterrupted control과 비교한다.

최종 승인서는 어떤 toolkit이 더 새롭다는 문장이 아니다. 지원 topology에서 compatibility, numerical, resume와 fault gate를 통과했고 같은 workload의 warm end-to-end 지표가 목표를 만족했는지 쓴다. 미실행 GPU/extension은 별 한계로 남긴다.

**현장 결정표와 최소 제출 파일**

build가 실패하면 toolkit 자체에 앞서 compiler path, target arch, extension guard와 ABI를 본다. load가 실패하면 linked/runtime library와 cubin/PTX를 본다. 실행되지만 느리면 actual dispatch, fallback, compile/cache와 critical path를 본다. numerical mismatch면 precision ladder에서 최초 edge를 찾는다. resume만 실패하면 scaler·FP8 history와 checkpoint cut을 본다.

이 분류는 “CUDA 13 문제”라는 큰 진단을 build/load/dispatch/numerics/state의 소유자로 좁힌다. 각 단계는 서로 다른 artifact와 담당자를 가진다. compiler flag를 바꿔 numerical bug를 숨기거나 tolerance를 넓혀 wrong dispatch를 통과시키지 않는다.

최소 제출 파일은 `cuda-environment`, `loaded-library`, `binary-manifest`, `precision-ladder`, `kernel-dispatch`, `fp8-state`, `checkpoint-resume`, `fault-matrix`, `performance-ledger`, `source-test-map`이다. GPU/driver/toolkit/framework/extension digest를 공유한다. result에서 actual binary와 kernel trace로 내려갈 수 있어야 한다.

`precision-ladder`는 edge별 logits/loss/gradient/delta error와 denominator를 가진다. `fp8-state`는 format, scale granularity, amax window와 saturation을 가진다. `fault-matrix`는 old driver, stale cache, missing cubin, wrong library, mask와 history reset의 expected gate를 가진다.

**CUDA 공식 문서 인용 카드**

호환성 claim은 compatibility guide의 toolkit/driver 조건과 revision을 쓴다. language/runtime behavior는 programming guide section을, component change와 known issue는 정확한 release note를 쓴다. 서로 다른 minor version 문서를 섞지 않는다. archive URL 또는 versioned document를 우선한다.

문서가 지원한다고 말하는 것과 framework/extension이 지원하는 것은 별 층이다. NVIDIA 문서, framework build matrix, extension source guard, 실제 candidate trace를 네 열에 둔다. 한 열의 PASS로 다른 열을 추론하지 않는다.

문서 update가 생기면 기존 run의 historical environment 해석을 바꾸지 않는다. 새 candidate에만 새 revision을 연결한다. hardware/driver/toolkit가 동일해도 extension binary가 달라지면 execution evidence는 새 RunID다.

**실행 경로 구두 검산**

인수자는 tensor 하나를 골라 storage, compute input, accumulator, gradient, optimizer state와 collective dtype을 설명한다. FP8이면 code와 scale/amax owner를 찾는다. 이어 profiler에서 실제 kernel과 fallback을 확인하고 해당 binary의 target arch와 source revision을 찾는다.

두 번째 질문은 CUDA 12 baseline에서 13 candidate로 무엇이 바뀌었는가다. driver/toolkit/compiler/library/extension 중 실제 diff를 나열하고 precision ladder에서 변수를 격리한다. candidate가 빨라졌다는 숫자만으로 원인을 설명하지 않는다.

세 번째 질문은 crash/resume다. portable checkpoint에는 scaler와 FP8 history가 있고 compiled cache/binary는 환경에 맞게 검증 또는 재생성한다. first GoldenBatch의 logits, gradient, delta가 선언 tolerance를 만족해야 한다. 미실행 GPU와 fault를 명확히 남긴다.

이 답들이 environment manifest, source/test 지도와 execution artifact에서 일치할 때만 migration을 승인한다. 최신 toolkit이라는 이유는 승인 조건이 아니며 compatibility, numerical, recovery와 workload 성능이 모두 독립 gate다.

**경계 회귀 표본**

빠른 CI는 BF16 eager reference와 한 fused kernel, FP8 toy scale, GradScaler overflow, binary manifest를 검사한다. release candidate job은 실제 GPU에서 전체 precision ladder, checkpoint resume, stale cache와 wrong library fault를 실행한다. 두 job은 같은 GoldenBatch와 environment schema를 쓴다.

CUDA minor, driver, framework 또는 extension 중 하나가 바뀌면 loaded library와 binary hash부터 다시 수집한다. numerical result가 같아도 dispatch가 달라졌다면 성능 evidence는 승계하지 않는다. 반대로 kernel 시간이 같아도 error threshold가 깨지면 release하지 않는다.

archive는 portable checkpoint와 재생성 가능한 compile/binary cache를 구분한다. scale·amax·scaler는 학습 state로 보존하고 old compiled graph는 candidate 환경에서 검증 없이 쓰지 않는다. 이 구분이 rollback과 forward migration을 안전하게 만든다.

운영 dashboard의 dtype 표본과 dispatch 통계는 sampling interval을 명시한다. 모든 kernel을 항상 trace해 성능을 훼손하지 않고 release·incident 구간에서 상세 trace를 켠다. 평상시에는 unexpected fallback, compile count, saturation과 scaler skip 같은 bounded metric으로 경보한다.

새 GPU architecture를 추가하면 기존 CUDA 13 PASS를 승계하지 않는다. target cubin/PTX, capability guard, FP8 format 지원, kernel shape와 numerical ladder를 다시 검증한다. 동일 toolkit도 architecture별 backend와 성능이 다를 수 있다.

모든 승인에는 실제 실행 증거가 필요하다.

**이 장이 넘기는 것.** parameter/state/compute/collective dtype 표, scaler state, kernel dispatch trace, CUDA compatibility manifest.

**다음 장에서 깨질 수 있는 것.** sharding과 collective가 dtype별 byte 수와 accumulator 소유권을 바꾼다.

**검증 체크포인트.** eager reference와 fused 경로의 mask·loss denominator·gradient 허용오차 및 실제 dispatch를 확인한다.

## 14.5 FP8 함수 경계에서 dispatch·resume·진단까지

이제 FP8 경로를 module 이름이 아니라 함수 호출과 mutable state로 해부한다. 입력 shape와 recipe가 kernel dispatch를 고르는 지점부터 checkpoint 복원과 rollback까지 따라가면, 지원된다는 문장과 실제 실행됐다는 증거를 구분할 수 있다.

### 14.5.1 FP8 실행 경로를 함수와 상태로 해부하기

저정밀 학습을 검토할 때 가장 먼저 버려야 할 그림은 `autocast가 텐서를 FP8로 바꾸고 GEMM을 호출한다`는 한 줄짜리 그림이다. 실제 경로에는 레시피 선택, 텐서별 양자화기 생성, amax 관측, scale 계산, 입력 변환, 행렬 배치 변환, 커널 선택, 누산, 출력 변환, 역전파용 상태 저장이 차례로 끼어든다. 이 단계 가운데 하나라도 BF16 fallback이면 실행 결과와 비용 구조가 달라진다. 그러므로 함수 지도는 사용자 옵션에서 시작해 실제 CUDA 커널과 저장 상태까지 닫혀야 한다.

NVIDIA Transformer Engine의 고정 태그 `v2.6.0`을 예로 들면 Python의 FP8 API 진입점은 [`transformer_engine/pytorch/fp8.py`](https://github.com/NVIDIA/TransformerEngine/blob/v2.6.0/transformer_engine/pytorch/fp8.py)에 놓인다.

양자화기와 텐서 표현은 [`transformer_engine/pytorch/tensor/float8_tensor.py`](https://github.com/NVIDIA/TransformerEngine/blob/v2.6.0/transformer_engine/pytorch/tensor/float8_tensor.py)에서 확인한다. 모듈 공통 상태는 [`transformer_engine/pytorch/module/base.py`](https://github.com/NVIDIA/TransformerEngine/blob/v2.6.0/transformer_engine/pytorch/module/base.py)가 소유한다.

CUDA 측 cast와 transpose 경로는 [`transformer_engine/common/cast/cast.cu`](https://github.com/NVIDIA/TransformerEngine/blob/v2.6.0/transformer_engine/common/cast/cast.cu), GEMM 경계는 [`transformer_engine/common/gemm/cublaslt_gemm.cu`](https://github.com/NVIDIA/TransformerEngine/blob/v2.6.0/transformer_engine/common/gemm/cublaslt_gemm.cu)에서 추적한다. 파일 이름을 나열하는 데서 끝내지 말고 `recipe → quantizer → tensor metadata → extension binding → device kernel` 호출 사슬을 같은 소스 태그 안에서 잇는다.

**상태 기계: 관측, 갱신, 소비**

FP8 상태를 세 종류로 나누면 재개 오류가 선명해진다. 첫째 설정 상태는 E4M3 또는 E5M2 형식, margin, history 길이, amax 계산 알고리즘, 블록 크기다. 둘째 관측 상태는 텐서별 amax history와 현재 history cursor다. 셋째 파생 상태는 scale과 inverse scale이다. 설정만 저장하고 관측을 버리면 같은 입력에서도 다음 scale이 달라진다. 파생값만 저장하고 recipe를 바꾸면 저장값의 의미가 달라진다.

한 step을 `observe → reduce → update → cast → consume`으로 표기한다. `observe`는 변환 전 실수 텐서의 절댓값 최댓값을 구한다. `reduce`는 필요한 parallel group에서 amax를 합의한다. `update`는 format의 representable maximum과 margin으로 scale을 정한다. `cast`는 clipping과 rounding을 수행하고 `consume`은 GEMM 또는 통신이 그 표현을 읽는다. delayed scaling에서는 step `t`의 cast가 이전 관측으로 만든 scale을 쓸 수 있으므로 event에 `observed_at`, `effective_at` 두 시점을 남긴다. 현재 scaling과 delayed scaling을 이름만 비교하면 이 시간 차이를 놓친다.

역전파는 forward의 거울이 아니다. activation gradient는 weight와 다른 분포이고 E5M2처럼 더 넓은 범위를 택할 수 있다. weight는 step 사이에 재사용되므로 cache된 transpose와 scale의 무효화 시점이 중요하다. optimizer가 parameter를 갱신했는데 FP8 weight cache를 갱신하지 않으면 다음 forward가 오래된 weight를 읽는다. 따라서 `ParameterVersion`, `QuantizedWeightVersion`, `ScaleVersion`을 연결하고 소비 직전에 동일성을 검사하는 실험이 필요하다.

**레시피 옵션은 어떤 상태와 커널을 바꾸는가**

format 옵션은 표현 가능한 지수 범위와 fraction 정밀도를 바꾸며 saturation과 rounding 오차를 동시에 바꾼다. margin은 amax에 대한 여유를 늘려 overflow를 줄이지만 작은 값에 쓸 유효 code를 줄인다. history 길이는 spike에 대한 기억과 회복 속도를 바꾼다. per-tensor scale은 metadata가 작지만 outlier channel이 전체 tensor의 해상도를 결정한다. block scaling은 지역 범위에 맞추는 대신 scale tensor의 shape, memory traffic, kernel tile 계약을 바꾼다.

`amax_compute_algo=max`와 최근 값 기반 방식은 같은 history를 다른 scale로 변환한다. update interval을 늘리면 scale 계산과 동기화 비용을 줄일 수 있지만 distribution shift를 늦게 반영한다. first/last layer를 높은 정밀도로 남기는 정책은 단순 안전장치가 아니다. embedding 바로 뒤와 vocab projection 앞은 분포와 task loss의 민감도가 달라 어느 경계를 제외했는지 실험 ID에 포함해야 한다.

옵션 표에는 `사용자 값`, `해석한 recipe 객체`, `생성된 quantizer 수`, `scale granularity`, `추가 저장 상태`, `선택 가능한 kernel`, `fallback 조건`을 둔다. 환경 변수 하나가 커널만 바꾸는지 수치 상태도 바꾸는지 구분한다. 예컨대 rowwise scaling을 켜는 옵션은 scale shape와 저장 schema까지 바꿀 수 있다. 반면 autotune algorithm 선택은 같은 논리 연산에서 구현과 workspace를 바꿀 수 있다. 두 종류를 같은 성능 스위치로 취급하면 rollback 호환성을 잘못 판단한다.

**누산과 epilogue를 별도로 감사한다**

입력이 FP8이라는 사실은 곱셈 결과를 어떤 dtype으로 누산하는지 말해 주지 않는다. Tensor Core instruction, cuBLASLt compute type, split-K reduction, epilogue output type를 따로 기록한다. bias와 activation이 GEMM epilogue에 fused되면 반올림 위치가 eager reference와 달라진다. atomic split-K는 reduction 순서가 실행마다 달라질 수 있다. deterministic 요구가 있으면 선택 알고리즘과 workspace 제약을 함께 검토한다.

검산용 작은 행렬에서는 고정 FP64 reference를 만든다. 그다음 입력만 FP8로 모사한 reference, 실제 FP8 GEMM, fused bias, fused activation을 한 단계씩 추가한다. 각 경계에서 max absolute error, RMS, cosine, saturation count를 기록한다. 최종 loss만 비교하면 입력 cast 오차와 epilogue 오류가 상쇄될 수 있다. 반례로 bias를 output cast 전과 후에 더해 두 결과가 구별되는지 확인한다.

### 14.5.2 CUDA 12.x와 13.x: 지원 문장을 실행 계약으로 번역하기

NVIDIA의 [CUDA Compatibility 문서](https://docs.nvidia.com/deploy/cuda-compatibility/)는 드라이버와 CUDA user-space 구성요소의 호환 관계를 설명한다. [CUDA 12.9.1 Release Notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)와 [CUDA 13.0 Release Notes](https://docs.nvidia.com/cuda/archive/13.0/cuda-toolkit-release-notes/index.html)는 각 toolkit의 구성요소, 지원 환경, 알려진 제한을 기록한다.

[CUDA C++ Programming Guide 12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html)와 [CUDA C++ Programming Guide 13.0](https://docs.nvidia.com/cuda/archive/13.0/cuda-c-programming-guide/index.html)는 실행·메모리·동기화 의미를 확인하는 기준이다. Compatibility 문서, Release Notes와 Programming Guide는 서로 대신 읽을 수 없다.

“CUDA 13 지원”을 다섯 문장으로 분해한다. 드라이버가 해당 runtime을 지원한다. framework wheel이 해당 CUDA 변형으로 배포되었다. C++/CUDA extension이 해당 toolkit header와 ABI로 빌드된다. binary가 대상 GPU의 cubin 또는 호환 PTX를 포함한다. 실제 model shape가 기대한 저정밀 kernel로 dispatch된다. 앞 네 문장이 참이어도 마지막은 fallback일 수 있다.

**환경 지문과 로딩 증거**

환경 지문은 `nvidia-smi` 한 줄이 아니다. driver version, device name와 compute capability, wheel tag, `torch.version.cuda`, NVRTC, cuBLAS/cuBLASLt, cuDNN, NCCL, toolkit compiler, host compiler, extension hash를 수집한다. process가 실제로 연 shared object 경로도 남긴다. container 안에 CUDA 13 toolkit이 있어도 CUDA 12 runtime을 묶은 wheel이 실행될 수 있으며, 그 자체는 모순이 아니다.

빌드 지문에는 완전한 compile command, include 순서, library search path, C++ ABI, `-gencode`, PTX 포함 여부가 들어간다. load 지문에는 resolved soname과 symbol version이 들어간다. dispatch 지문에는 op, shape, stride, dtype, selected backend와 fallback reason이 들어간다. 세 지문을 합쳐야 어느 층의 변경이 성능 또는 수치 차이를 만들었는지 설명할 수 있다.

CUDA 12→13 비교 실험은 같은 binary를 두 드라이버에서 비교하는 축과, 같은 source를 두 toolkit으로 빌드하는 축을 분리한다. framework까지 동시에 바꾸면 세 번째 축이다. 가능한 한 한 축씩 움직여 최초 차이를 찾는다. 지원 matrix 때문에 조합을 동시에 올려야 한다면 compound migration임을 선언하고 중간 rung을 framework의 공식 artifact로 만든다.

**CUDA 13 전환에서 자주 생기는 잘못된 추론**

첫 번째 오해는 toolkit major가 올라가면 모든 kernel이 새 codegen을 쓴다는 것이다. wheel에 포함된 cubin과 extension build 시점을 확인해야 한다. 두 번째 오해는 PTX가 있으면 모든 미래 GPU에서 실행된다는 것이다. driver가 PTX ISA를 해석해야 하며 application의 capability guard가 새 GPU를 거부할 수도 있다. 세 번째 오해는 import 성공이 ABI 호환을 증명한다는 것이다. 늦게 resolve되는 symbol과 특정 shape에서만 호출되는 extension이 남는다.

네 번째 오해는 numerical tolerance가 같으면 backend가 같다는 것이다. BF16 fallback이 우연히 더 정확할 수 있다. 다섯 번째 오해는 throughput 향상이 CUDA major 때문이라는 것이다. framework graph, kernel library revision, autotune cache, clock와 workload가 함께 고정되어야 한다. 여섯 번째 오해는 새로운 library가 이전 checkpoint의 opaque state를 읽는다는 것이다. portable tensor와 backend 전용 cache를 분리해야 한다.

### 14.5.3 커널 디스패치와 소스 좌표 감사

PyTorch의 autocast 정책과 scaler 계약은 고정 태그의 [`torch/amp/grad_scaler.py`](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/amp/grad_scaler.py) 및 [`aten/src/ATen/autocast_mode.cpp`](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/autocast_mode.cpp)에서 읽는다. scaled loss 생성, optimizer별 found-inf, skip, growth tracker update 순서를 실제 함수 호출과 맞춘다. 문서의 사용 예만으로 여러 optimizer와 accumulation의 state transition을 추론하지 않는다.

scaled gradient의 검사 시점이 중요한 이유는 clip과 weight decay가 단위를 소비하기 때문이다. `unscale_` 전에 norm을 재면 scale이 포함되고, overflow step에서 scheduler를 먼저 전진시키면 parameter update 수와 learning-rate clock이 갈라진다. test는 finite step, 한 optimizer만 overflow, accumulation 중 overflow, growth interval 직전 checkpoint를 포함한다. 각 case에서 optimizer step count와 scheduler step count가 기대값인지 본다.

**dispatch guard를 표로 바꾸기**

커널 후보마다 device capability, dtype, shape divisibility, alignment, stride, head dimension, dropout, causal flag, deterministic mode, workspace 한도를 적는다. source의 조건문을 그대로 복사하는 대신 각 조건이 왜 필요한지 설명한다. tile 배수는 out-of-bound predication 또는 tensor-core layout과 연결되고, alignment는 vectorized load와 연결된다. capability guard는 instruction availability와 연결된다.

boundary fixture는 guard 양쪽 값을 사용한다. head dimension 127과 128, contiguous와 sliced stride, aligned와 offset pointer, sequence tile 배수 전후를 비교한다. 지원 밖 입력은 명확한 fallback 또는 오류여야 한다. 지원 guard가 잘못되어 fast path로 들어가는 silent corruption이 가장 위험하다. 실제 dispatch identity를 trace하지 않으면 reference PASS가 fallback 덕분인지 알 수 없다.

source 좌표는 선언, dispatch, kernel, test 네 종류를 한 카드로 묶는다. 선언은 public option의 의미, dispatch는 조건, kernel은 수치 연산, test는 upstream이 보장한 범위다. upstream test가 forward만 검사하면 backward와 resume 보장은 비어 있다. 이 빈칸을 로컬 fixture의 목적과 연결한다.

**fused attention과 CE의 불변식**

attention은 확률 행렬을 저장하지 않고 tile별 online softmax를 사용할 수 있다. 각 query row의 max와 exp sum을 block 순서에 따라 갱신해도 안정 softmax와 동치여야 한다. causal mask, packed sequence 경계, GQA head 매핑, dropout RNG offset은 동치 조건이다. forward LSE를 backward recompute가 같은 좌표계로 읽어야 한다.

CE는 local 또는 vocab-sharded logits에서 global max, exp sum, target logit, valid count를 합성한다. all ignored batch의 denominator 처리, target shard boundary, 매우 큰 logit을 시험한다. gradient row sum이 0에 가까운지와 ignored row가 정확히 0인지 본다. 평균 loss 하나가 맞는 것으로 target offset 오류를 통과시키지 않는다.

### 14.5.4 재개와 롤백의 원자성

저정밀 checkpoint cut은 parameter, optimizer moment, scheduler, RNG, GradScaler, FP8 recipe, amax history, scale version을 같은 committed step으로 묶어야 한다. compiled graph와 autotune cache는 재생성 가능한 artifact로 분리한다. fused optimizer의 opaque state는 portable logical state로 변환 가능한지 명시한다. deserialize 성공은 next update의 의미가 같다는 증거가 아니다.

재개 검증은 저장 직전 control과 load 직후 candidate에 같은 GoldenBatch를 준다. forward logits, loss, unscaled gradient, clip coefficient, parameter delta, next scale을 순서대로 비교한다. first divergence가 scale이면 optimizer를 조사하기 전에 FP8 state를 본다. BF16 rung이 통과하고 FP8 rung만 실패하는지 확인해 데이터와 RNG 문제를 제외한다.

**topology 변경과 scale owner**

DP 또는 TP 크기가 바뀌면 amax를 어느 group에서 합쳤는지가 달라질 수 있다. per-rank history를 단순 concatenate하지 않는다. scale 의미가 global tensor인지 local shard인지 recipe에서 확인하고 reshard 함수를 정의한다. tensor scale이 global이면 old rank history를 max-reduce할 수 있지만, block scale이면 block의 global offset을 기준으로 재배치해야 한다.

topology 변경 checkpoint에는 old/new tensor slice와 scale slice를 함께 기록한다. parameter만 정확히 옮기고 scale block을 다른 shard에 붙이는 오류는 첫 logits에서 드러날 수 있다. optimizer moment, master weight, FP8 cache의 `ParameterVersion`도 같은 mapping을 따라야 한다. 지원하지 않는 조합은 조용히 history를 초기화하지 말고 numerical-resume 등급을 거부한다.

**장애 주입 계획**

장애 주입은 실제 대규모 학습을 실행하라는 뜻이 아니라 release fixture의 설계다. stale amax history, 한 rank의 scale 누락, GradScaler counter 한 step rollback, 오래된 quantized weight cache, 잘못된 cubin, wrong shared library, forced fallback을 각각 단일 변수로 만든다. 예상 gate와 최초 실패 artifact를 미리 쓴다.

wrong library는 load manifest에서, missing target는 dispatch 전에, stale history는 resume numerical gate에서 닫혀야 한다. 모든 장애가 최종 loss NaN으로만 검출된다면 관측 설계가 부족하다. 반대로 정상 fallback을 무조건 오류로 취급하지 않고 승인된 fallback 목록과 성능 등급을 둔다. 목록 밖 fallback은 correctness가 맞아도 release를 막는다.

**실전 디버깅 워크북**

증상이 `invalid device function`이면 binary target와 actual device capability, PTX/cubin 목록, loaded extension hash 순서로 본다. `undefined symbol`이면 import stack보다 resolved shared library와 build ABI를 본다. illegal memory access이면 비동기 peer error에 속지 않고 최소 shape, 동기화 경계, 최초 kernel을 찾는다. NaN이면 output에서 역으로 가지 않고 layer hook으로 최초 non-finite 입력 또는 gradient를 찾는다.

성능 저하는 cold compile, recompile, fallback, host gap, kernel duration, memory traffic, collective wait로 분해한다. FP8인데 느리면 FP8 coverage, cast/transpose/amax overhead, scale collective, 작은 GEMM을 본다. GPU utilization 평균은 짧은 gap을 숨기므로 critical path trace를 사용한다. CUDA 13 candidate만 느리면 binary와 algorithm 선택이 baseline과 같은지부터 비교한다.

**증상→관측→분기→원인→검증**

resume 직후 NaN이라는 증상에서 관측은 first diverging layer, scale, amax, saturation, scaler found-inf다. BF16도 갈리면 checkpoint cut 또는 RNG로 분기한다. BF16은 같고 FP8만 갈리면 recipe/history/cache owner로 분기한다. history를 control에서 복원해 차이가 사라지면 원인 가설이 강화된다. 이어 history reset negative control이 동일 증상을 만드는지 반증한다.

throughput 20% 하락이라는 증상에서 관측은 actual kernel coverage와 step critical path다. fallback이 늘면 dispatch guard와 build를 본다. kernel은 같고 launch gap이 늘면 graph break와 CPU를 본다. kernel duration만 늘면 clock, algorithm, workspace, memory path를 본다. 한 변수 수정 뒤 같은 workload와 warm 조건으로 재측정한다.

loss가 서서히 달라지는 증상에서는 최초 한 step delta를 high precision oracle과 비교한다. 한 step부터 일정 배수면 denominator 또는 scaling, 여러 step 뒤 누적이면 rounding/optimizer state, spike 뒤부터면 amax history를 의심한다. 이 분기는 원인을 보장하지 않지만 무관한 층을 동시에 바꾸지 않게 한다.

**리뷰어 체크리스트**

리뷰어는 첫째 public 옵션이 실제 recipe와 quantizer를 어떻게 만들었는지 확인한다. 둘째 dtype ledger에 storage, input, accumulator, output, gradient, optimizer와 collective를 모두 채운다. 셋째 kernel dispatch guard와 actual trace를 비교한다. 넷째 scale/amax/scaler가 checkpoint cut에 포함되었는지 본다.

다섯째 CUDA 환경의 driver, runtime, toolkit, extension, target architecture를 분리한다. 여섯째 eager→fused→FP8→compiled ladder에서 최초 error edge를 찾는다. 일곱째 fallback을 별 성능 결과로 분류한다. 여덟째 resume first update와 uninterrupted control을 비교한다. 아홉째 장애 주입이 예상 gate에서 실패하는지 확인한다.

최종 인계 묶음은 `PrecisionPolicy`, `KernelDispatchMap`, `CudaEnvironment`, `BinaryManifest`, `ScaleStateSchema`, `GoldenBatchID`, `ResumeComparison`, `FaultMatrix`를 포함한다. 각 값은 같은 source revision과 RunID를 가리킨다. 이 묶음이 있어야 다음 장에서 collective dtype과 shard별 scale owner가 바뀌더라도 저정밀 오차와 분산 소유권 오류를 구별할 수 있다.

## 14.6 옵션·통신·품질을 하나의 효과 사슬로 잇는다

옵션 하나는 local kernel만 바꾸지 않는다. collective dtype과 optimizer 경계, 처리량, 최종 품질의 오차 예산까지 이어지므로 request→applied state→runtime effect→model metric 순서로 추적한다.

### 14.6.1 옵션 하나를 끝까지 추적하는 읽기 실습

실제 소스 독해는 `FP8 사용`이라는 설정 이름에서 멈추지 않는다. 설정 parser가 문자열을 recipe enum으로 바꾸고, module construction이 quantizer를 어느 tensor slot에 붙이며, forward context가 첫 microbatch와 recompute를 어떻게 구별하는지 찾는다. 이어 extension binding이 scale pointer, amax pointer, transpose buffer, workspace와 stream을 어떤 순서로 kernel에 넘기는지 확인한다. 마지막으로 upstream test가 이 경로의 어느 상태를 assert하는지 읽는다.

이 방식이 필요한 이유는 같은 옵션이 실행 phase마다 다른 효과를 낼 수 있기 때문이다. training forward에서는 activation과 weight를 변환하지만 evaluation에서는 weight cache만 쓸 수 있다. activation checkpoint recompute에서는 amax history를 두 번 갱신하면 안 된다. microbatch accumulation에서 첫 microbatch가 weight cache를 만들고 다음 microbatch가 재사용할 수 있다. graph capture에서는 lazy allocation과 Python state mutation이 금지될 수 있다. 옵션의 의미는 이러한 상태 분기를 포함한다.

**함수 호출 카드**

호출 카드의 첫 행은 public symbol과 고정 source permalink다. 둘째 행은 입력 option과 default resolution이다. 셋째 행은 생성하거나 갱신한 state key다. 넷째 행은 extension 함수의 argument shape, dtype, pointer owner다. 다섯째 행은 kernel guard와 fallback이다. 여섯째 행은 upstream test와 비어 있는 case다. 일곱째 행은 checkpoint serialize/restore 함수다.

카드는 코드 줄 번호만 수집하는 목록이 아니다. `recipe.margin`이 scale update 식의 어느 항에 들어가며, scale이 cast와 GEMM 가운데 누가 소비하고, history가 어느 시점에 shift되는지를 문장으로 잇는다. source revision이 바뀌면 symbol 이동만 확인하지 않고 state transition diff를 낸다. default가 바뀌면 explicit config가 없는 기존 run도 새로운 의미를 가질 수 있다.

**테스트 층을 분리한다**

unit test는 scale 계산, cast saturation, history rotation을 작은 tensor로 검사한다. operator test는 GEMM, norm, attention, CE forward/backward를 높은 정밀도 reference와 비교한다. module test는 cache와 microbatch, activation recompute를 포함한다. distributed test는 amax와 gradient owner를 검사한다. resume test는 checkpoint 전후 next update를 비교한다. migration test는 binary와 CUDA 환경을 바꾼다.

한 층의 PASS를 다른 층으로 확대하지 않는다. scale 함수 unit test가 맞아도 잘못된 tensor slot에 scale을 붙일 수 있다. single-rank module이 맞아도 rank별 amax reduction이 틀릴 수 있다. uninterrupted training이 맞아도 history serialization이 빠질 수 있다. CUDA 12 artifact가 맞아도 CUDA 13 rebuild의 dispatch가 fallback일 수 있다.

테스트에는 negative control을 대응시킨다. scale unit에는 history order 교환, operator에는 mask 방향, module에는 stale weight cache, distributed에는 wrong reduction group, resume에는 missing scaler, migration에는 wrong shared library를 둔다. 정상 case만 통과하는 test는 오류를 발견하는 능력을 증명하지 못한다.

### 14.6.2 저정밀 통신과 optimizer 경계

저정밀 학습의 dtype 원장은 local GEMM에서 끝나지 않는다. gradient bucket이 BF16인지 FP32인지, reduce-scatter가 어떤 dtype을 전송하고 어떤 dtype으로 누산하는지 기록한다. FP8 통신은 quantized payload 외에 scale metadata가 필요하며 rank마다 서로 다른 scale로 만든 code를 그대로 더할 수 없다. scale을 합의한 뒤 quantize하는지, dequantize 뒤 높은 정밀도로 reduce하는지 구현을 읽는다.

통신 압축 hook은 reducer와 optimizer 사이에 들어가 gradient 의미를 바꾼다. residual error-feedback을 쓰면 residual buffer도 checkpoint state다. hook이 반환한 tensor가 average인지 sum인지, DDP가 추가 division을 하는지 확인한다. 압축 성능은 payload byte 감소와 quantize/dequantize kernel, scale collective, numerical error를 함께 측정한다.

**fused optimizer의 step 상태**

fused Adam류는 여러 parameter의 gradient, moment, parameter update를 multi-tensor kernel로 묶는다. step counter가 host integer인지 device tensor인지, bias correction이 어느 정밀도로 계산되는지, weight decay가 moment 전후 어디에 적용되는지 본다. `capturable` 옵션은 step과 found-inf 같은 control state의 device placement를 바꿀 수 있다. CUDA Graph replay가 주소와 control flow를 고정해야 하기 때문이다.

overflow skip에서 parameter뿐 아니라 moment, step counter, scheduler가 전진하지 않아야 하는지 policy를 명시한다. 여러 optimizer 가운데 하나만 found-inf인 case를 시험한다. global skip과 optimizer별 skip은 학습 의미가 다르다. checkpoint는 실제 committed update 수를 저장해야 하며 loop iteration으로 대신하지 않는다.

unfused FP64 또는 FP32 reference와 한 parameter group의 한 step을 비교한다. gradient, decay contribution, moment, normalized update, final delta를 중간별로 기록한다. final parameter가 우연히 비슷해도 moment가 다르면 다음 step에서 발산한다. empty gradient, sparse/unsupported tensor, mixed group dtype와 very large step counter를 경계 case로 둔다.

**graph capture와 메모리 주소**

CUDA Graph는 launch overhead를 줄일 수 있지만 static memory address, 지원되는 stream dependency와 반복 가능한 control path를 요구한다. lazy FP8 workspace 생성, dynamic amax buffer resize, overflow에 따른 Python branch, shape 변화가 capture를 깨뜨릴 수 있다. warmup에서 state와 workspace를 materialize하고 capture/replay가 실제 사용되었는지 trace한다.

input staging buffer에 다음 microbatch를 너무 일찍 덮으면 replay kernel이 잘못된 데이터를 읽는다. output과 gradient buffer도 consumer 완료 event 전에 재사용하면 race가 난다. eager와 replay를 같은 GoldenBatch로 비교하며 한 번이 아니라 여러 replay에서 buffer alias 오류를 찾는다. graph cache key에는 shape, stride, dtype, device capability, recipe와 binary identity를 포함한다.

### 14.6.3 수치 예산을 모델 품질과 연결하기

커널 오차가 작다는 주장에는 모델 수준의 허용 기준이 필요하다. layer output error가 residual stream을 따라 누적되거나 normalization에서 확대될 수 있다. 반대로 큰 absolute error가 activation scale에 비해 작을 수 있다. layer별 RMS와 cosine, logits KL, top-token margin, task metric을 단계적으로 연결한다. 어느 단계에서 허용 기준을 적용했는지 사전에 정한다.

gradient cosine만 높고 norm이 일정 배수로 틀리면 optimizer와 clipping 결과는 달라진다. loss가 같아도 rare token logit이 크게 다를 수 있다. 평균 metric은 saturation이 특정 expert나 layer에 몰린 현상을 숨긴다. scale, zero, saturation, error를 tensor 역할과 layer별로 보고 rank max도 확인한다.

**허용오차를 사후 조정하지 않는다**

BF16, FP8, fused reduction마다 reference와 기대 error budget을 pilot에서 정하고 release 전에 고정한다. failure를 본 뒤 tolerance를 넓히면 wrong mask와 정상 rounding을 구분할 수 없다. mathematical invariant인 finite count, ignored gradient zero, causal future contribution zero는 근사 tolerance와 별도로 둔다.

stochastic rounding 또는 nondeterministic reduction은 distributional test를 쓸 수 있다. seed 집합에서 bias, variance, worst quantile을 보고 control과 비교한다. 그러나 nondeterminism을 이유로 arbitrary drift를 허용하지 않는다. RNG algorithm, seed/counter owner와 sample 수를 기록한다.

**승인 가능한 실험 설계서**

실험 설계서는 baseline과 candidate의 정확한 source, driver, runtime, toolkit, binary와 workload를 고정한다. precision ladder의 각 rung에서 바뀌는 변수 하나, 수치 관측, dispatch 관측, 성능 관측을 적는다. cold compile과 warm steady-state, uninterrupted와 resume를 분리한다. 미지원 또는 미실행 GPU는 빈칸이 아니라 명시적 한계다.

결과가 실패하면 증상→관측→분기→최초 원인→수정→동일 fixture 재검증을 따른다. 여러 환경 변수를 동시에 바꾸지 않는다. 성공하면 negative control이 여전히 실패하는지 확인한다. 마지막으로 artifact hash와 source/test map을 인계해 다음 검토자가 숫자를 재해석하지 않고 실제 코드 경로까지 내려갈 수 있게 한다.

**승인 직전 교차 검산**

승인자는 설정 파일, 저장 상태, 실행 trace를 서로 독립적으로 읽은 뒤 같은 결론이 나오는지 확인한다. 설정에 FP8이 켜져 있어도 dispatch trace의 BF16 fallback 비율이 승인 범위를 넘으면 성능 결과를 거부한다. trace가 FP8이어도 scale history가 checkpoint schema에 없으면 exact resume를 거부한다. binary manifest가 대상 SM을 포함해도 GoldenBatch 수치 ladder가 실패하면 배포하지 않는다.

마지막 실험은 option 하나를 의도적으로 뒤집어 관측 지표가 예상 방향으로 변하는지 보는 반증이다. margin을 바꾸면 saturation과 작은 값의 해상도가 함께 움직여야 하고, forced fallback은 kernel identity와 처리량을 바꾸어야 한다. scaler history rollback은 resume 첫 update에서 드러나야 한다. 변화가 보이지 않으면 option이 실제 경로에 연결되지 않았거나 관측기가 잘못된 것이다.

이 교차 검산은 최신 CUDA나 높은 이론 FLOPS를 승인 근거로 삼지 않게 한다. 환경 호환성, 실제 함수와 커널 경로, 수치 불변식, 상태 재개, workload 성능이 각각 독립 gate를 통과해야 한다. 검증하지 않은 shape, GPU, topology는 명시적 미지원 범위로 남겨 다음 실험의 출발점으로 인계한다.

## 14.7 CUDA 12→13 migration을 구성요소 수명주기로 실행한다

migration은 새 toolkit에서 build가 끝났을 때 완료되지 않는다. driver·runtime·compiler·math library·extension·artifact의 수명주기를 나누고, 기준 환경과 후보 환경을 같은 수치·성능·복구 fixture로 검산한다.

### 14.7.1 CUDA 12.x와 13.x를 구성요소 수명주기로 비교한다

CUDA toolkit의 major version은 compiler 하나의 버전이 아니다. nvcc·ptxas, runtime, math library, profiler, header와 developer tool의 묶음이다. framework wheel은 이 묶음 가운데 일부 user-space library를 자체 포함할 수 있고, host의 toolkit은 custom extension build에만 쓰일 수 있다. kernel module과 user-space driver library는 host driver 배포에 속한다. 따라서 `nvidia-smi`, `nvcc --version`, `torch.version.cuda`가 서로 다른 숫자를 보이는 상황을 곧바로 혼입이라고 판정하지 않는다. process가 실제 load한 library와 extension build log가 관계를 증명한다.

12.x에서 13.x로 옮길 때 release note의 added·deprecated·removed 항목을 dependency graph에 투영한다. framework core, Transformer Engine, Triton, CUTLASS, FlashAttention, bitsandbytes, custom fused optimizer가 어떤 header, ABI, architecture guard를 쓰는지 찾는다. build가 성공하는 것과 대상 GPU에서 기대 cubin을 load하는 것은 다른 gate다. PTX가 포함되어 driver JIT로 실행되는 경로와 architecture-specific cubin 경로도 성능 결과를 분리한다.

호환성 실험은 직교 축으로 만든다. 같은 application binary에서 driver만 바꾸어 runtime compatibility를 본다. 같은 source·framework에서 toolkit만 바꾸어 extension code generation을 본다. 같은 binary를 GPU architecture별로 실행해 fatbin coverage를 본다. framework release까지 동시에 바꾸어야 한다면 compound migration이라고 표시하고 source·dispatcher 변화까지 diff한다. “CUDA 13 효과”라는 단일 원인으로 합치지 않는다.

artifact에는 host compiler, C++ ABI, `-gencode`, PTX ISA, linked library SONAME, RPATH와 binary hash를 둔다. container digest만으로는 host driver와 mounted library를 설명하지 못한다. cache key에는 compiler·toolkit·GPU·source·flags를 포함하고 old autotune/compile cache를 의도적으로 주입해 invalidation test를 한다. stale cache가 수치적으로 맞아도 다른 kernel을 선택할 수 있으므로 dispatch identity를 검증한다.

**AMP의 자동성은 dtype 정책의 자동 선택일 뿐이다**

autocast는 영역 안의 모든 연산을 같은 낮은 정밀도로 바꾸지 않는다. operator별 policy와 입력 dtype에 따라 낮은 정밀도, FP32, widest-input 등의 경로를 선택할 수 있다. parameter storage dtype, GEMM input dtype, accumulator dtype, output dtype와 gradient dtype을 따로 적는다. TF32는 FP32 tensor의 storage를 바꾸지 않고 특정 연산의 내부 계산 경로에 영향을 줄 수 있으므로 BF16/FP16 autocast와 같은 축으로 쓰지 않는다.

GradScaler는 작은 FP16 gradient가 underflow하는 위험을 줄이기 위해 loss와 gradient를 scale한다. optimizer 직전에는 unscale하고 모든 관련 gradient의 nonfinite를 판정한 뒤 update 전체를 실행하거나 건너뛴다. 여러 optimizer가 있거나 gradient accumulation을 쓸 때 unscale·check·step·update 호출 순서를 state machine으로 그린다. clipping은 unscale 뒤에 해야 원래 단위의 threshold를 뜻한다. overflow step에서 scheduler가 전진하는지는 13장의 clock 계약과 맞춘다.

BF16은 FP16보다 exponent 범위가 넓어 dynamic scaling 필요가 줄 수 있지만, 모든 연산이 안전하다는 뜻은 아니다. softmax, norm, reduction, optimizer state에는 더 높은 정밀도 누산이나 storage가 필요할 수 있다. FP8은 format뿐 아니라 tensor별 scale·amax history가 추가 state다. E4M3와 E5M2의 범위·정밀도 선택, delayed/current scaling, history length와 margin이 실제 tensor role과 연결되어야 한다.

precision ladder는 FP32 reference, BF16/FP16 eager, fused, FP8, compiled 순으로 한 edge씩 이동한다. 각 rung에서 forward output, loss, selected gradient, optimizer delta, nonfinite와 kernel dispatch를 비교한다. tolerance는 tensor role과 dtype별로 사전 등록하고 실패 뒤 넓히지 않는다. 실제 품질 실험 전 작은 pathological fixture로 overflow, underflow, saturation, cancellation을 의도적으로 만든다.

**fused kernel의 이득을 memory traffic 식으로 설명한다**

여러 elementwise 연산을 별 kernel로 실행하면 각 단계가 중간 tensor를 HBM에 쓰고 다음 kernel이 다시 읽는다. fusion은 producer의 값을 register나 shared memory에 두고 consumer까지 이어 HBM 왕복과 launch를 줄일 수 있다. 그러나 register pressure가 커져 occupancy가 낮아지거나, 큰 fused kernel의 한 부분이 shape에 맞지 않아 전체가 느려질 수 있다. fusion 자체가 목표가 아니라 end-to-end critical path와 live memory 감소가 목표다.

attention, norm, activation, cross entropy, optimizer fusion은 서로 다른 정확성 경계를 가진다. mask와 causal offset, reduction denominator, epsilon, stochastic RNG, saved tensor와 backward 재계산을 검사한다. forward parity만으로 backward를 승인하지 않는다. activation checkpointing과 결합하면 fused backward가 기대한 tensor를 저장하는지, recompute가 같은 RNG·scale state를 쓰는지 본다.

dispatcher는 device capability, dtype, shape, stride, alignment, dropout, mask와 deterministic setting을 읽을 수 있다. 옵션이 켜졌다는 사실보다 어떤 guard가 참이어서 어느 kernel이 선택되었는지를 trace한다. unsupported shape의 fallback은 correctness 결과에는 포함할 수 있지만 같은 성능 backend로 평균내지 않는다. workload shape histogram과 backend coverage를 함께 보고한다.

benchmark는 cold compile·autotune과 warm steady state를 나누고, GPU clock·power·concurrency를 기록한다. kernel microbenchmark가 빨라도 graph break나 layout transform, cast, amax reduction이 늘어 step이 느릴 수 있다. profiler에서 HBM byte, achieved occupancy, launch gap, temporary allocation과 collective wait를 함께 본다. 수치 error와 speedup은 같은 RunID의 별 gate다.

**CUDA Graph와 compiler의 상태 경계를 닫는다**

`torch.compile`류 compiler는 Python graph capture, decomposition, scheduling/fusion, device code generation과 cache를 가진다. graph break가 발생하면 eager island와 compiled region이 섞일 수 있다. compile 성공 여부 대신 graph 수, break reason, guard와 recompile count를 수집한다. dynamic sequence length가 매번 새 specialization을 만들면 warm benchmark가 production distribution을 대표하지 않는다.

CUDA Graph는 이미 선택된 kernel launch sequence를 재사용해 host launch overhead를 줄이는 별 층이다. static address와 반복 가능한 control flow가 필요하므로 optimizer step tensor, FP8 workspace, communication buffer를 warmup에서 materialize한다. allocator가 주소를 재사용하는지, replay 중 learning rate·scale이 최신 device state를 읽는지 확인한다. overflow나 shape 변화가 생길 때 graph를 선택적으로 쓰는 정책도 기록한다.

compiler cache와 graph executable은 portable checkpoint state가 아니다. checkpoint에는 parameter, optimizer, scaler, FP8 mathematical state를 두고 target environment에서 binary와 graph를 다시 만든다. CUDA 12 artifact를 13 environment로 옮겼을 때 old cache를 로드하지 않도록 environment digest를 key에 넣는다. rebuild 뒤 GoldenBatch parity와 actual dispatch를 다시 검사한다.

최종 migration 승인은 build, load, dispatch, numerics, state resume, performance 여섯 축이다. 한 축의 PASS를 다른 축으로 확대하지 않는다. 이 분해가 있어야 CUDA major 변경, framework 변경, kernel 변경 가운데 실제 원인을 찾을 수 있다. CUDA Graph는 같은 주소와 실행 구조를 재사용해 launch overhead를 줄인다. 그러나 포인터, shape, stream dependency와 allocator 상태가 capture 뒤 바뀌면 재생의 의미가 깨진다. optimizer와 scaler처럼 매 step 변하는 값은 고정 주소 tensor에서 갱신하거나 graph 밖의 제어로 둔다. graph가 빨랐다는 결과만 남기지 않고 capture 가능 조건과 fallback 경로를 manifest에 넣는다.

**CUDA 12.x와 13.x를 네 계약으로 분해한다**

“CUDA 버전을 올린다”는 말에는 서로 다른 네 변화가 겹친다. 첫째는 kernel을 만드는 compiler toolchain이다. `nvcc`, host compiler, PTX assembler와 device library가 여기에 속한다. 둘째는 실행 시 process가 로드하는 CUDA runtime과 math library다. 셋째는 kernel을 GPU에 제출하고 PTX를 JIT하는 driver다. 넷째는 PyTorch, Transformer Engine, NCCL과 custom extension이 기대하는 ABI와 wheel build metadata다. 하나의 `12.8` 또는 `13.0` 문자열로 네 층을 대표시키지 않는다.

extension을 CUDA 12.x에서 build하고 다른 host의 CUDA 13.x user-space와 섞었을 때 성공 여부는 단순 대소 비교가 아니다. binary가 cubin만 포함하는지 PTX도 포함하는지, target SM이 무엇인지, 링크한 library가 static인지 dynamic인지, 사용하는 symbol version과 driver가 지원하는 PTX ISA가 무엇인지에 달렸다. “driver는 backward compatible하다”는 문장을 모든 user-space library의 ABI 호환으로 확대하지 않는다.

환경 지문은 최소한 GPU model과 compute capability, driver version, toolkit compiler version, framework wheel의 build CUDA, runtime에서 로드된 `libcudart`, cuBLAS·cuDNN·NCCL, extension의 build flags와 architecture list를 담는다. shell의 `nvcc --version`은 framework가 실제 로드한 runtime을 증명하지 않는다. process map, library resolution과 framework API를 교차한다.

CUDA 13 계열로 갈 때는 지원 GPU·host compiler·OS가 달라질 수 있고 오래된 architecture 대상 binary 생성 정책도 변할 수 있다. 정확한 차이는 설치된 release notes와 compatibility 문서의 특정 버전에서 확인한다. 책은 “13은 항상 빠르다” 같은 결론을 내리지 않는다. compiler 최적화, library algorithm, framework dispatch와 hardware가 모두 같지 않으면 성능 차이의 원인을 분리해야 한다.

**compiler와 PTX·cubin의 수명**

소스 kernel은 front-end와 NVVM 계층을 거쳐 PTX 또는 device code가 되고, assembler가 특정 SM의 cubin을 만든다. fat binary는 여러 cubin과 PTX fallback을 담을 수 있다. 현재 GPU용 cubin이 있으면 그것을 로드하고, 없지만 호환 PTX가 있으면 driver JIT가 실행될 수 있다. 이때 최초 실행 latency와 cache가 개입한다.

benchmark에서 candidate만 첫 iteration이 느리다면 algorithm 자체가 아니라 PTX JIT 또는 compiler cache miss일 수 있다. cold process와 warm process를 분리하고 cache 위치, permission, image immutability를 기록한다. cache를 지운 결과와 유지한 결과를 섞지 않는다. production container에서 cache가 ephemeral이면 배포마다 cold cost가 반복될 수 있다.

architecture list가 현재 GPU를 누락하면 예상치 못한 PTX path 또는 “no kernel image” 오류가 난다. 반대로 너무 많은 architecture를 넣으면 build 시간과 binary 크기가 늘어난다. source revision, `-gencode` 또는 framework architecture 환경, resulting binary의 code object를 release artifact에 기록한다.

device link와 relocatable device code를 쓰는 extension은 object 간 toolchain 일관성이 중요하다. 일부 object를 이전 toolkit으로 만들고 나머지를 새 toolkit으로 만들면 link 또는 runtime symbol 문제가 생길 수 있다. clean rebuild와 incremental build를 비교하고 stale object가 재사용되지 않게 build cache key에 toolkit, host compiler, flags, headers hash를 넣는다.

**runtime·driver·library ABI**

CUDA runtime에는 static과 shared linking 경로가 있다. dynamic library search order가 host toolkit을 먼저 잡는지 wheel에 포함된 library를 잡는지 확인한다. `LD_LIBRARY_PATH` 하나가 예상과 다른 cuBLAS 또는 cuDNN을 로드할 수 있다. import 성공은 첫 kernel 실행, 특정 dtype 또는 algorithm 호출의 symbol resolution까지 보증하지 않는다.

driver API와 runtime API의 version reporting 의미도 구분한다. driver가 보고하는 최대 지원 수준과 process가 링크한 runtime version은 같은 값이 아니다. compatibility package를 쓴다면 어떤 library를 제공하고 어느 driver 범위를 전제하는지 공식 matrix로 확인한다. 설치 파일 이름이 있다고 지원을 추측하지 않는다.

ABI 검사는 library 존재 여부, required symbol, symbol version, dependency chain을 본다. `ldd` 또는 플랫폼 도구 결과를 artifact로 남기고 실행 중 실제 map을 확인한다. container 안과 host driver mount의 경계를 표시한다. 같은 image가 node pool마다 다르게 동작하면 host driver와 GPU firmware, mount된 library 차이를 먼저 의심한다.

release test는 import, allocation, 간단한 GEMM에서 끝나지 않는다. 사용하는 dtype과 shape, attention·normalization·optimizer kernel, distributed collective까지 대표 경로를 실행하는 소규모 fixture가 필요하다. 대규모 모델을 실행하지 않아도 dispatch와 symbol·numerical contract를 검증할 수 있다.

**AMP를 dtype 상태 기계로 읽는다**

automatic mixed precision은 모든 tensor를 낮은 dtype으로 바꾸는 기능이 아니다. operator별 autocast policy가 입력을 cast하거나 더 높은 정밀도를 유지하고, parameter master copy와 optimizer state는 별 dtype을 가질 수 있다. forward output dtype, backward accumulation dtype, reduction dtype, optimizer compute dtype을 분리한다.

FP16은 exponent 범위가 좁아 작은 gradient underflow와 큰 값 overflow에 취약하다. BF16은 FP32와 비슷한 exponent 범위를 갖지만 fraction bit가 적어 상대 정밀도가 낮다. TF32는 Ampere 이후 특정 FP32 matrix 연산의 내부 경로와 연관되며 tensor storage dtype을 TF32로 바꾸는 것이 아니다. FP8은 format 선택과 scale state가 명시적으로 필요하다.

autocast context는 thread-local 또는 실행 context의 상태일 수 있다. custom autograd function, compiled graph, worker thread에서 context가 전파되는지 확인한다. operator가 allowlist에 없거나 custom CUDA kernel이면 입력 dtype과 accumulation을 스스로 정의해야 한다. “AMP를 켰다”는 설정만으로 전체 경로를 설명하지 못한다.

**GradScaler의 정확한 사건 순서**

scaled loss (L_s=sL)를 backward하면 gradient도 (s)배다. optimizer update 전에 (s)로 나누어 원래 scale로 되돌린다. clipping은 보통 unscale 뒤 수행해야 원래 gradient norm의 threshold와 비교된다. nonfinite 검사는 parameter group과 device에 걸쳐 일관되게 합쳐야 한다.

정상 step에서는 optimizer update가 commit되고 growth tracker가 증가한다. 정해진 interval 동안 정상이라면 scale을 growth factor만큼 키울 수 있다. Inf/NaN이 발견되면 optimizer step을 skip하고 scale을 backoff하며 tracker를 초기화한다. 이 순서가 scheduler committed clock과 연결된다.

gradient accumulation에서는 매 microbatch마다 scale을 임의로 바꾸지 않는다. 한 logical update 동안 같은 scale을 사용하고 accumulation boundary에서 unscale·검사한다. 중간 microbatch에서 nonfinite를 조기에 찾더라도 모든 rank가 같은 collective/control flow에 도달하도록 설계한다. rank별로 한쪽만 optimizer를 skip하면 parameter가 즉시 갈라진다.

scaler state에는 current scale, growth tracker, growth/backoff factor와 interval이 있다. checkpoint에서 누락하면 resume 직후 overflow 양상이 달라진다. optimizer와 scaler commit generation을 묶는다. parameter는 update됐는데 scaler만 이전 state이거나 그 반대인 torn checkpoint를 거부한다.

**BF16에도 수치 검사가 필요한 이유**

BF16은 넓은 exponent 덕분에 loss scaling 없이 쓰는 경우가 많지만 안전하다는 뜻은 아니다. fraction이 짧아 작은 update가 parameter 표현에서 사라질 수 있고, reduction·normalization의 cancellation과 softmax overflow가 남는다. optimizer master weight를 FP32로 두는 이유와 실제 cast 경계를 확인한다.

BF16 activation, FP32 accumulation이라는 표기가 모든 kernel에서 지켜지는지 source와 profiler로 본다. fused kernel이 내부 accumulation dtype을 옵션에 따라 바꿀 수 있다. output만 BF16이라고 내부 수치 계약을 추측하지 않는다. 고정 입력에 대한 FP32 reference와 error distribution을 shape·magnitude별로 비교한다.

nonfinite rate가 0이어도 정확도 열화가 있을 수 있다. layer별 activation range, gradient cosine similarity, update-to-weight ratio, validation probe를 본다. tolerance는 kernel upgrade 결과를 본 뒤 정하지 말고 baseline과 모델 품질 요구에서 사전 정의한다.

**FP8은 tensor와 scale의 결합 자료형이다**

FP8 값만 저장해서는 원래 실수 범위를 알 수 없다. tensor 또는 block에 연결된 scale과 inverse scale, format, amax history, update recipe가 의미의 일부다. E4M3과 E5M2는 exponent와 fraction 배분이 달라 forward activation, gradient 등 서로 다른 역할에 쓰일 수 있다. hardware와 library가 지원하는 실제 조합을 확인한다.

일반적인 흐름은 현재 tensor의 절댓값 최대치 또는 통계량을 관측하고, representable range에 맞는 scale을 결정하고, quantize된 값을 kernel에 공급하는 것이다. delayed scaling은 과거 amax history를 사용해 현재 step의 scale을 결정할 수 있다. current scaling은 현재 값과 더 가깝지만 fusion·synchronization 비용과 causal ordering이 달라질 수 있다.

scale이 너무 작으면 saturation이 늘고 너무 크면 유효 fraction을 적게 사용한다. outlier 하나가 tensor 전체 scale을 지배할 수 있어 per-tensor, per-channel, block scaling의 trade-off가 생긴다. 더 세밀한 scale은 metadata와 연산 비용, kernel 제약을 늘린다.

**Transformer Engine의 상태 경로를 읽는다**

고정 checkout `sources/training-transformer-engine`에서 Python module이 FP8 autocast context와 recipe를 만들고, module forward가 metadata를 준비하며, extension binding이 C++/CUDA kernel로 dispatch하는 경로를 추적한다. 실제 symbol은 checkout에서 확인하고 revision `3691693263d2b66a68867e39b7449876844e06cf`와 함께 기록한다. 이름을 기억으로 쓰지 않는다.

읽을 때 입력 tensor만 보지 않는다. FP8 metadata에 어떤 scale, inverse scale, amax history index가 있고 forward와 backward가 그것을 누가 갱신하는지 본다. distributed amax reduction이 있다면 어느 group과 어느 stream에서 실행되는지 확인한다. graph capture에서 metadata 주소와 update ordering이 안정적인지도 본다.

recipe 옵션은 상태 공간을 바꾼다. amax history length를 바꾸면 checkpoint shape와 resume compatibility가 달라질 수 있다. scaling algorithm과 margin은 scale 수열을 바꾼다. FP8 output을 다음 module이 소비하는 경로와 중간에 higher precision으로 되돌리는 경로는 메모리 traffic과 오차가 다르다.

test를 읽을 때 단순 forward success와 numerical reference, checkpoint/resume, graph capture, distributed consistency를 분리한다. 특정 shape와 hardware에서 skip된 test는 지원 증거가 아니다. skip condition과 expected capability를 source card에 포함한다.

**FP8 checkpoint의 불변식**

model weight와 optimizer state뿐 아니라 FP8 scale metadata가 같은 logical step이어야 한다. amax history ring의 current index, history contents, scale과 inverse scale의 곱 관계를 검증한다. load 후 첫 forward가 history를 초기화하거나 한 번 advance하는지 boundary fixture로 잡는다.

topology가 바뀌면 amax reduction group이 달라질 수 있다. global scale을 유지할지 local shard scale을 허용할지 정책을 정한다. old group의 partial statistic을 new group state로 그대로 읽지 않는다. resize dry-run에서 같은 input에 대한 scale decision과 output error를 비교한다.

resume 뒤 FP8만 발산하면 먼저 scale state 누락, history index off-by-one, recipe default 변경, distributed reduction group을 본다. optimizer lr부터 낮추면 원인을 가릴 수 있다. BF16 reference path를 같은 checkpoint와 batch에서 실행해 divergence가 quantization 경계에 있는지 좁힌다.

**fused kernel을 memory traffic과 상태로 설명한다**

두 operator를 fusion하면 intermediate tensor를 global memory에 쓰고 다시 읽는 traffic을 줄일 수 있다. kernel launch도 줄어든다. 그러나 fusion은 단순 속도 합치기가 아니다. 중간 값을 어떤 dtype으로 유지하는지, backward를 위해 무엇을 저장하는지, reduction 순서와 RNG 소비가 바뀐다.

예를 들어 bias-add, activation, dropout fusion은 activation 전 값을 저장할지 재계산할지에 따라 backward memory와 compute가 달라진다. dropout mask 생성이 fused kernel 안으로 들어가면 RNG offset의 owner와 parallel reproducibility를 다시 검증해야 한다. shape fallback이 unfused path로 내려갈 때 RNG 수열까지 같을 필요가 있는지 정의한다.

cross entropy fusion은 logits 전체를 materialize하지 않거나 chunked reduction을 사용할 수 있다. label smoothing, ignore index, class weights, vocabulary parallel과 reduction denominator가 reference 의미와 같아야 한다. 빠른 loss scalar 하나가 맞아도 gradient가 틀릴 수 있으므로 forward와 backward를 별도 비교한다.

**traffic 식과 roofline**

unfused chain의 대략적인 byte는 각 intermediate의 write와 read를 합해 계산한다. fusion 뒤 register 또는 shared memory에 머무는 값은 HBM traffic에서 빠질 수 있다. 그러나 input·output과 saved tensor, workspace를 포함해야 한다. theoretical byte 감소가 같은 비율의 속도 향상을 보장하지 않는다. compute, occupancy, register pressure와 memory coalescing이 병목을 바꿀 수 있다.

arithmetic intensity를 계산하고 profiler의 DRAM byte, achieved bandwidth, FLOP, occupancy와 비교한다. fusion 뒤 register 사용이 늘어 active blocks가 줄면 예상보다 느릴 수 있다. small shape에서는 launch 감소가 크고, large shape에서는 library GEMM이 지배할 수 있다. shape bucket별로 결과를 낸다.

benchmark는 warmup, synchronization과 measurement stream을 명시한다. asynchronous launch 직후 host timer를 멈추면 실제 kernel 시간을 재지 못한다. compile/JIT와 autotune을 cold와 steady state로 분리한다. allocator와 cache 상태도 고정한다.

**dispatch guard와 fallback**

fused path는 dtype, alignment, stride, contiguous, shape divisibility, SM capability와 build flag를 guard로 가질 수 있다. 옵션 하나를 켰다고 모든 batch가 fused kernel을 쓰지 않는다. 실제 dispatch를 trace하거나 profiler kernel name으로 확인한다.

guard 표에는 predicate, fast path, fallback, numerical difference, expected test를 둔다. sequence tail이나 odd hidden size처럼 경계 shape를 넣는다. training 중 dynamic shape 분포에서 fast-path hit rate를 계측한다. microbenchmark의 대표 shape 하나만으로 전체 run 속도를 예측하지 않는다.

fallback이 correctness reference라고 가정하지 않는다. 둘 다 같은 잘못된 denominator를 공유할 수 있다. 독립 FP32 formulation과 finite difference 또는 작은 autograd reference를 사용한다. fused/unfused equivalence는 tolerance, dtype, magnitude regime를 명시한다.

**CUDA Graph capture의 실전 상태 기계**

capture 전에 memory pool과 static input/output buffer를 준비한다. warmup으로 lazy initialization, library handle, autotune과 JIT를 capture 밖에서 끝낸다. capture stream과 다른 stream의 dependency가 허용 규칙을 만족하는지 확인한다. capture 중 host synchronization, dynamic allocation, unsupported collective가 들어오면 실패하거나 의도치 않은 동작을 낼 수 있다.

replay 때 input data를 static buffer에 copy하고 graph를 launch한다. buffer 주소는 같아도 내용은 바뀐다. shape가 달라지면 다른 graph bucket을 선택하거나 eager fallback한다. optimizer scalar인 lr, loss scale, step counter가 host Python 값으로 capture되면 첫 값이 고정될 수 있다. device tensor에서 읽도록 만들거나 graph 밖에서 갱신한다.

**RNG와 collective**

dropout RNG가 replay마다 같은 mask를 만들면 학습 의미가 깨진다. graph-safe generator가 offset을 어떻게 advance하고 checkpoint에 무엇을 저장하는지 확인한다. multiple graph와 pipeline stage가 같은 generator를 공유하면 replay order가 seed 소비를 바꿀 수 있다.

NCCL collective capture는 library와 graph 지원 조건, communicator lifetime을 확인한다. rank 모두가 동일한 collective sequence와 graph replay count를 가져야 한다. 한 rank가 shape fallback으로 eager path에 내려가고 다른 rank는 graph를 replay하면 hang할 수 있다. dispatch decision을 group-wide로 합의하거나 bucket 조건을 보장한다.

graph capture checkpoint는 graph 실행 객체 자체보다 재구성 가능한 recipe와 static state를 저장하는 경우가 많다. load 뒤 graph를 재capture할 때 first-step numerical state가 변하지 않는지 본다. compiler cache와 graph cache invalidation key에 device, library, shape, dtype, recipe를 넣는다.

**실패 주입**

입력 주소 변경, odd shape, allocator pressure, loss scale 변경, skipped optimizer step, topology resize를 주입한다. 안전한 fallback인지 명시적 오류인지 기대를 정한다. 조용히 stale buffer를 읽는 것은 최악의 실패다. output checksum과 sentinel region으로 탐지한다.

capture 성공 뒤에도 replay 중 nonfinite가 발생할 수 있다. optimizer step skip을 graph 내부 조건으로 처리하는지 graph 밖에서 분기하는지 확인한다. rank별 predicate가 달라지지 않도록 nonfinite flag를 collective한다. scheduler committed clock도 같은 predicate를 소비한다.

### 14.7.2 CUDA 12→13 migration의 실제 절차

baseline artifact부터 동결한다. container digest, driver와 GPU inventory, framework·extension wheel, compiler flags, loaded library map, representative shape와 numerical output, profiler trace를 저장한다. candidate에서는 toolkit만 바꾸려 하되 dependency solver가 framework나 library를 함께 올렸다면 별 변화로 표시한다.

build 단계에서 clean build를 수행하고 compiler warning, generated architecture, linked dependencies와 binary size를 비교한다. custom extension마다 import test, representative kernel, backward, odd shape와 unsupported-path test를 수행한다. source compile 성공은 runtime compatibility의 충분조건이 아니다.

정확성 ladder는 FP32 reference, BF16/FP16 AMP, FP8, fused path, graph capture 순으로 한 층씩 올린다. 각 층에서 forward error, gradient error, optimizer delta와 nonfinite를 비교한다. 모든 기능을 한 번에 켜고 차이가 나면 원인 공간이 너무 크다.

성능 ladder는 cold start, steady kernel, end-to-end step을 분리한다. kernel time이 같아도 compile cache, graph capture, allocator, dataloader 때문에 step time이 달라질 수 있다. 반대로 step time이 같아도 특정 kernel regression이 다른 이득에 가려질 수 있다. trace와 counter로 원인을 분해한다.

**rollback 가능성**

candidate checkpoint가 baseline runtime에서 읽히는지 보장하지 않는다. FP8 metadata schema, optimizer fused state, serialization dtype가 바뀔 수 있다. migration 전 baseline-compatible checkpoint를 보존하고 candidate가 쓴 artifact를 별 generation에 둔다. rollback test를 실제로 수행한다.

node pool을 점진 전환할 때 한 distributed job 안에 서로 다른 driver/library 경로를 섞지 않는다. 공식 지원 범위 안이어도 collective와 kernel numerical path가 달라질 수 있다. homogeneous pool로 canary를 만들고 inventory admission control을 둔다.

승인 조건은 지원 matrix 문장, build 증거, numerical gate, performance gate, failure injection과 rollback rehearsal이 모두 통과하는 것이다. “unit test가 통과했다”는 한 줄은 dynamic loading, rare shape와 장시간 scale drift를 덮지 못한다.

**저정밀 장애를 최초 경계로 좁힌다**

loss가 NaN이면 data input, forward activation, loss reduction, scaled gradient, unscale, clipping, collective, optimizer update 순으로 finite sentinel을 배치한다. 모든 tensor를 dump하면 I/O와 개인정보 위험이 크므로 shape, dtype, min/max, finite count, norm과 sample hash를 기록한다. 최초 nonfinite tensor와 producer kernel을 찾는다.

baseline BF16은 정상이고 FP8만 실패하면 scale metadata와 quantization boundary를 우선 본다. eager는 정상이고 graph만 실패하면 captured scalar, address, RNG와 stream dependency를 본다. unfused는 정상이고 fused만 실패하면 dispatch guard, accumulation dtype, saved tensor와 backward를 본다. CUDA 12는 정상이고 13만 실패하면 실제 loaded library와 rebuild 여부부터 확인한다.

**silent error의 탐지**

NaN이 없어도 update 방향이 서서히 달라질 수 있다. 고정 probe batch에서 layer별 activation·gradient cosine, relative error, update checksum을 주기적으로 비교한다. 전체 모델 checksum은 최초 원인을 찾기 어렵기 때문에 stage별 digest를 둔다.

tolerance는 tensor magnitude와 dtype에 따라 정한다. 절대오차 하나로 0 근처와 큰 값을 모두 평가하지 않는다. absolute, relative, ULP 또는 cosine을 목적에 맞게 조합한다. reduction은 nondeterministic order 때문에 bitwise 비교가 부적절할 수 있지만 체계적인 bias는 허용하지 않는다.

성능 이상과 정확성 이상을 같은 incident timeline에 놓는다. fallback 때문에 느려진 동시에 더 높은 정밀도로 바뀌어 output이 달라질 수 있다. profiler의 kernel name, dispatch counter와 numerical probe를 event ID로 조인한다.

**인수 판정과 독자 실습**

독자는 scalar와 작은 matrix에서 FP16 loss scaling부터 손으로 계산한다. overflow window를 만들고 unscale 전후 clipping 결과가 왜 다른지 보인다. checkpoint 전후 scaler state와 optimizer delta를 비교한다. 그 다음 BF16 path에서 representable spacing 때문에 작은 update가 사라지는 예를 만든다.

FP8 실습은 고정 tensor의 range를 바꾸며 scale, saturation rate와 reconstruction error를 기록한다. history 기반 scale에서 outlier가 몇 step 동안 영향을 남기는지 본다. resume 때 history를 누락해 첫 output이 달라지는 negative test를 만든다. hardware 실행이 없으면 source와 unit fixture 수준에서 state transition oracle을 구축한다.

fusion 실습은 elementwise chain의 analytical HBM byte를 계산하고 profiler가 있을 때 measured byte와 비교한다. odd stride와 shape로 fallback을 유도하고 실제 dispatch를 확인한다. forward뿐 아니라 backward gradient와 saved tensor memory를 비교한다.

CUDA migration 실습은 두 toolkit을 대규모로 실행하는 대신 고정 환경 manifest와 extension build matrix를 만든다. compiler, runtime, driver, library를 별 열로 두고 어떤 조합을 공식 문서가 지원하는지 근거 좌표를 붙인다. clean build, loaded map, representative test와 rollback 순서를 rehearsal한다.

최종 제출물은 `PrecisionRecipeCard`, `CudaEnvironmentFingerprint`, `KernelDispatchTable`, `NumericalErrorBudget`, `GraphCaptureContract`, `MigrationCertificate`다. 각 카드에는 revision, 입력 shape·dtype, mutable state, expected fallback, source/test 좌표와 실패 증거가 있다.

이 장의 핵심은 낮은 bit 수가 빠르다는 구호가 아니다. dtype은 범위와 오차의 계약이고, FP8은 scale state까지 포함한 자료형이며, fusion은 memory traffic과 autograd 상태의 재설계다. CUDA 세대 전환은 compiler·runtime·driver·ABI 네 계약의 migration이다. 이 경계를 증명할 수 있을 때만 속도 향상을 학습 품질과 복구 가능성을 잃지 않고 사용할 수 있다.

**softmax와 normalization의 수치 경계를 해부한다**

attention score의 softmax를 그대로 (e^{x_i}/\sum_j e^{x_j})로 계산하면 큰 양수에서 overflow가 난다. 보통 row maximum을 빼 (e^{x_i-m})으로 계산한다. 이 변환은 정확한 실수에서는 같은 값이지만 finite precision에서는 범위를 안정화한다. causal mask와 padding mask가 들어간 row에서 모든 원소가 masked된 경우, maximum과 denominator가 어떻게 처리되는지도 명시해야 한다.

fused attention은 score matrix 전체를 HBM에 쓰지 않고 tile 단위로 online maximum과 normalization sum을 갱신할 수 있다. 이전 tile의 maximum (m_{old})와 새 maximum (m_{new})가 다르면 이전 누적합과 output accumulator를 (e^{m_{old}-m_{new}})로 rescale한다. 이 상태 전이가 틀리면 forward output뿐 아니라 backward gradient가 오염된다.

저정밀 입력을 사용해도 maximum, exponential, sum과 output accumulator를 더 높은 dtype에서 처리할 수 있다. 어떤 단계가 FP32인지 kernel source와 dispatch 옵션으로 확인한다. output dtype만 보고 내부 안정성을 판단하지 않는다. 긴 sequence, 큰 score range, mask-only row와 noncontiguous stride를 numerical fixture에 넣는다.

LayerNorm은 평균과 분산을 구하고 정규화한다. (E[x^2]-E[x]^2) 형태는 큰 평균과 작은 분산에서 cancellation에 취약할 수 있다. Welford류 online algorithm 또는 higher-precision accumulation 여부가 중요하다. RMSNorm은 mean subtraction이 없지만 square reduction과 reciprocal square root의 오차가 남는다.

epsilon이 어디에 더해지는지, epsilon dtype과 값, variance가 biased인지가 output을 바꾼다. fused residual-add+norm은 residual을 어느 dtype으로 더하고 저장하는지도 중요하다. residual stream을 FP32로 유지하는 recipe와 낮은 dtype으로 유지하는 recipe는 memory와 정확성 계약이 다르다.

**backward의 취약점**

forward relative error가 작아도 backward reduction은 더 민감할 수 있다. softmax probability가 0 또는 1에 가까우면 작은 차이가 gradient에 크게 영향을 줄 수 있고, norm backward는 여러 reduction 항이 상쇄된다. finite difference는 작은 tensor의 독립 확인에 유용하지만 step size와 dtype에 민감하므로 FP64 reference와 analytic formula를 함께 사용한다.

gradient test는 random normal 하나로 끝내지 않는다. constant row, near-constant row, large magnitude, subnormal 근처, outlier와 masked row를 포함한다. scale을 로그 간격으로 바꾸며 nonfinite와 error transition을 찾는다. 실제 모델 activation histogram에서 대표·극단 quantile을 뽑아 synthetic fixture와 연결한다.

distributed tensor parallel에서는 softmax 전 score 또는 이후 output이 shard될 수 있다. reduction maximum과 sum의 collective dtype, group과 순서를 확인한다. rank별 local maximum만 쓰면 global softmax가 틀린다. CP나 sequence partition과 결합할 때 mask와 online state ownership을 명시한다.

**optimizer kernel의 저정밀 계약**

fused optimizer는 여러 parameter tensor의 gradient, moment와 weight decay를 하나 또는 몇 kernel에서 처리한다. multi-tensor apply는 launch를 줄이고 memory traversal을 묶는다. 그러나 parameter, gradient, first/second moment, master weight와 step counter의 dtype이 서로 다를 수 있다. 각 buffer의 owner와 업데이트 순서를 표로 만든다.

FP16 parameter에 작은 delta를 직접 더하면 rounding으로 사라질 수 있어 FP32 master weight를 두는 방식이 쓰인다. optimizer는 master를 갱신하고 낮은 dtype parameter로 cast한다. checkpoint가 model parameter만 저장하면 optimizer resume에서 master를 잃는다. full state와 model-only export를 구분한다.

Adam류의 bias correction에서 step counter가 host scalar인지 device tensor인지 graph capture 가능성을 바꾼다. step이 FP32 tensor로 커질 때 정수 정밀도와 pow 계산도 본다. skipped update에서 counter와 beta power가 전진하면 다음 update가 달라진다. fused kernel의 nonfinite gate가 모든 parameter group에 동일하게 적용되는지 검사한다.

weight decay, clipping과 gradient scaling의 순서가 reference optimizer와 같아야 한다. fused path가 clip norm을 내부에서 계산하면 reduction group과 accumulation dtype을 확인한다. sharded optimizer에서는 local norm과 global norm을 혼동하지 않는다. empty shard와 sparse gradient 경계도 test한다.

**optimizer state compression**

moment를 8-bit 또는 blockwise quantized state로 저장하면 scale metadata가 추가된다. FP8 activation과 마찬가지로 state 값만으로 의미가 완결되지 않는다. block size, outlier 처리, scale update cadence와 error feedback을 기록한다. parameter group과 shard가 바뀔 때 block boundary가 달라지면 resume trajectory도 달라질 수 있다.

압축의 이득은 optimizer-state HBM byte와 communication·checkpoint byte에서 나오지만 dequantize compute와 metadata traffic이 든다. theoretical memory 절감과 peak live-set을 구분한다. update 순간 full-precision temporary가 생기면 peak가 예상보다 높을 수 있다.

정확성은 final metric만 보지 않는다. moment reconstruction error, update direction cosine, small-gradient preservation과 long-horizon drift를 본다. 같은 state를 quantize/dequantize 반복할 때 오차가 누적되는지, checkpoint serialization이 한 번 더 round trip을 만드는지 확인한다.

**compiler fusion과 graph break를 읽는 법**

framework compiler는 Python graph를 capture하고 IR로 낮춘 뒤 operator fusion과 kernel generation을 수행할 수 있다. Dynamo류 front-end graph, AOT autograd graph, compiler IR, generated kernel을 구분한다. Python 코드 한 줄이 kernel 하나와 대응하지 않는다. graph break가 생기면 eager와 compiled segment가 섞인다.

dynamic shape guard는 입력의 size, stride, dtype, device와 Python value에 조건을 건다. 새로운 길이가 guard를 벗어나면 recompile 또는 fallback이 발생한다. sequence-length curriculum에서 compile cache가 폭증할 수 있다. bucket policy와 observed graph count, compile time을 계측한다.

custom autograd와 side effect, data-dependent control flow는 capture를 막거나 의미를 고정할 수 있다. optimizer와 scheduler scalar를 Python에서 읽으면 specialization이 지나칠 수 있다. graph 안에 남길 state와 밖에서 업데이트할 state를 정하고 source-level explanation과 generated artifact를 연결한다.

**generated kernel을 감사한다**

generated source에서 load/store dtype, accumulator type, reduction tree, vector width와 boundary mask를 본다. source가 매 run 바뀔 수 있으므로 compiler/framework revision, config와 cache key를 함께 저장한다. generated code 전체를 책에 복제하지 않고 중요한 kernel signature와 contract를 원장 좌표로 남긴다.

컴파일러가 algebraic rewrite를 하면 exact floating-point 순서가 달라질 수 있다. fast-math, TF32와 reassociation 옵션은 성능과 오차를 바꾼다. baseline과 candidate의 option provenance를 기록한다. deterministic flag가 모든 custom kernel을 강제한다고 가정하지 않는다.

graph break가 사라져 빨라졌더라도 saved tensor와 mutation semantics가 바뀌지 않았는지 test한다. aliasing view를 copy로 오인하거나 in-place version counter를 놓치면 backward가 틀릴 수 있다. odd stride, shared storage와 in-place update를 negative fixture에 넣는다.

**stream과 event로 kernel 순서를 증명한다**

CUDA stream 안의 작업은 순서가 있지만 서로 다른 stream 사이에는 event 또는 명시적 dependency가 필요하다. H2D copy, forward compute, gradient communication, optimizer와 checkpoint copy가 여러 stream에 놓이면 host 코드 순서만으로 실행 순서를 추측하지 않는다.

producer stream이 tensor를 쓰고 consumer stream이 읽기 전에 event를 wait해야 한다. allocator도 tensor의 마지막 사용 stream을 알아야 memory를 너무 일찍 재사용하지 않는다. custom extension이 stream recording을 놓치면 드물게 stale data 또는 corruption이 생길 수 있다.

nonfinite flag를 별 reduction stream에서 계산하고 optimizer stream이 기다리지 않으면 stale flag로 update할 수 있다. FP8 amax reduction과 scale update도 같은 문제를 가진다. event graph에 tensor/state producer, consumer와 wait edge를 표시한다. profiler timeline으로 실제 순서를 표본 확인한다.

**overlap의 정확성**

communication-compute overlap은 dependency가 없는 구간만 겹칠 수 있다. gradient bucket all-reduce가 끝나기 전에 optimizer가 그 gradient를 읽으면 틀린다. stream priority가 dependency를 대체하지 않는다. event가 없으면 빠른 GPU에서는 우연히 맞고 혼잡한 GPU에서 실패할 수 있다.

benchmark에서 overlap 이득은 critical path로 측정한다. 각 kernel 시간을 더한 값만 비교하면 동시 실행을 반영하지 못한다. timeline의 gap, overlap과 end-to-end step을 본다. profiler 자체 overhead와 synchronization 영향을 별도로 측정한다.

failure injection은 communication stream을 인위적으로 지연하고 consumer가 기다리는지 본다. compute load를 바꿔 race 재현 확률을 높인다. 결과 checksum과 sanitizer류 도구를 조합하되 모든 race를 도구가 찾는다고 가정하지 않는다.

**메모리 allocator와 저정밀의 역설**

dtype byte가 절반이라고 peak memory도 정확히 절반이 되지 않는다. parameter master copy, optimizer state, saved activation, temporary workspace, fragmentation과 graph static pool이 함께 존재한다. FP8은 scale metadata와 higher-precision accumulation을 추가한다. fusion은 intermediate를 줄이지만 workspace 또는 saved tensor를 늘릴 수 있다.

메모리 timeline은 allocated, reserved, active와 peak live-set을 구분한다. allocator reserved가 높다고 모두 live tensor는 아니다. OOM 직전 snapshot에서 큰 block과 fragmentation을 본다. dynamic shape와 varying workspace가 pool을 조각낼 수 있다.

activation checkpointing은 저장 tensor를 줄이고 forward를 재계산한다. recompute가 autocast·FP8 scale과 같은 상태를 재현하는지 확인한다. backward 중 재계산에서 amax history를 두 번 갱신하면 원래 forward와 다른 state transition이 생길 수 있다. RNG와 autocast context도 보존한다.

**OOM 결정 트리**

첫 질문은 deterministic shape에서 재현되는가다. 특정 길이에서만 나면 dispatch workspace와 bucket을 본다. 여러 step 뒤에만 나면 graph/cache 누적, tensor reference leak, variable workspace와 fragmentation을 본다. resume 뒤에만 나면 restored state와 graph recapture의 이중 allocation을 의심한다.

precision을 낮춰 해결됐다고 끝내지 않는다. 어떤 buffer가 줄었고 numerical contract가 어떻게 변했는지 기록한다. batch를 줄이면 scheduler·gradient noise가 변하므로 동일 recipe가 아니다. allocator tuning은 fragmentation을 줄일 수 있지만 실제 live-set 초과를 숨길 수 없다.

memory gate에는 representative max shape, optimizer step과 checkpoint overlap 시점, graph capture pool까지 포함한다. steady forward peak만 재면 실제 운영 peak를 놓친다.

**CUDA 공식 문서를 근거로 바꾸는 방법**

공식 문서는 programming guide, runtime/driver API, compatibility guide, release notes, library별 문서로 나뉜다. 질문에 맞는 문서를 선택한다. kernel execution semantics를 release notes 한 줄로 설명하거나, toolkit compatibility를 programming guide의 일반 문장으로 대신하지 않는다.

근거 카드에는 문서 제목과 버전, section anchor, 접근 또는 vendoring 날짜, 뒷받침하는 claim을 적는다. latest 웹 문서는 시간이 지나 바뀔 수 있으므로 가능하면 versioned PDF 또는 archive와 checksum을 둔다. 문장 일부를 떼어 모든 환경에 일반화하지 않고 전제와 예외를 함께 요약한다.

공식 지원과 실제 검증을 분리한다. matrix가 조합을 지원해도 우리 extension과 shape의 correctness·performance는 별 시험이 필요하다. 반대로 우연히 실행된 비지원 조합을 production 지원으로 승격하지 않는다. `SupportedByVendor`, `ValidatedLocally`, `ObservedOnly` 상태를 구분한다.

**CUDA 12.x와 13.x 비교표의 유지보수**

minor release마다 compiler, supported host, libraries와 known issue가 달라질 수 있다. “12.x” 전체를 하나의 행으로 합치지 않고 실제 baseline minor와 candidate minor를 기록한다. driver branch와 GPU architecture도 열에 둔다.

책의 서술은 변하지 않는 분석 틀을 제공하고, 세부 표는 source registry와 evidence card에서 갱신 가능하게 한다. ABI, PTX, hardware deprecation처럼 시간이 민감한 주장은 고정 버전 좌표를 가진다. 최신이라는 단어만 쓰지 않는다.

**장애 대응 재생 훈련**

첫 scenario는 새 container가 import는 되지만 첫 FP8 attention에서 undefined symbol로 죽는 경우다. loaded library map과 extension dependency를 baseline과 diff하고, wheel build CUDA와 host mount를 확인한다. symlink를 임의 변경하기 전에 공식 compatibility와 symbol owner를 찾는다.

둘째는 정상 실행하지만 resume 뒤 loss가 천천히 벌어지는 경우다. FP8 metadata와 GradScaler, optimizer master state의 generation을 비교한다. 같은 checkpoint에서 BF16 probe와 FP8 probe를 실행해 경계를 좁힌다. lr 조정은 원인 확인 뒤 별 변경으로 한다.

셋째는 graph-enabled candidate만 간헐적으로 hang하는 경우다. rank별 graph bucket, collective sequence와 replay count를 비교한다. 한 rank의 shape fallback과 event dependency를 본다. timeout stack만으로 NCCL 자체를 원인이라고 단정하지 않는다.

넷째는 fused kernel이 특정 hidden size에서만 느린 경우다. dispatch guard와 chosen kernel, register/occupancy, workspace, alignment를 비교한다. fallback이 더 빠른 shape라면 dispatch heuristic 문제일 수 있다. correctness가 같다는 gate부터 통과한다.

**release decision**

release owner는 environment fingerprint와 build provenance가 완전한지 확인한다. numerical owner는 representative와 adversarial fixture, backward와 long-horizon probe를 승인한다. performance owner는 cold/steady/end-to-end와 shape distribution을 승인한다. training owner는 resume·overflow·checkpoint 의미를 승인한다.

조건부 승인은 제한 범위를 machine-readable하게 둔다. 예를 들어 특정 SM, dtype, shape와 eager path만 허용하고 나머지는 admission에서 거부한다. 문서에만 제한을 적고 runtime이 silent fallback 또는 unsupported path를 허용하지 않는다.

rollback trigger는 nonfinite rate, numerical probe drift, performance regression, graph mismatch와 unknown library map이다. rollback artifact와 baseline-compatible checkpoint가 실제로 존재하는지 확인한다. 장애 중 급히 이전 image를 찾지 않는다.

### 14.7.3 migration 증거를 교차 재생한다

저정밀 recipe를 설명할 때 독자는 storage, compute, accumulation, communication과 optimizer dtype을 각각 말할 수 있어야 한다. scaler가 언제 증가·감소하고 어느 사건에서 optimizer와 scheduler를 멈추는지 그릴 수 있어야 한다. FP8에서는 scale과 amax history의 owner·checkpoint를 추가해야 한다.

kernel 최적화를 설명할 때는 줄어든 HBM byte와 launch, 늘어난 register·workspace, 바뀐 saved tensor와 reduction 순서를 함께 말해야 한다. fast-path guard와 fallback hit rate를 모르면 실제 run이 무엇을 사용했는지 모르는 것이다.

CUDA migration을 설명할 때 compiler, runtime, driver, library ABI를 분리하고 실제 process의 loaded map으로 증명해야 한다. version 문자열 하나나 import 성공으로 승인하지 않는다. 공식 support evidence와 local validation을 별 상태로 유지한다.

graph와 compiler를 설명할 때 capture된 주소, mutable scalar, RNG, stream event와 dynamic-shape cache key를 말할 수 있어야 한다. replay 성공이 semantic correctness를 자동 보장하지 않는다. failure injection과 independent oracle이 필요하다.

최종적으로 모든 최적화는 같은 질문으로 돌아온다. 어떤 byte와 동기화를 줄였고, 그 대가로 어떤 상태와 전제를 추가했는가. 그 상태는 checkpoint와 topology 변경에서 복구되는가. 수치 오차는 사전에 정한 예산 안에 있는가. 이 질문에 코드·test·trace·수학으로 답할 때만 낮은 정밀도와 CUDA kernel은 위험한 마법이 아니라 관리 가능한 학습 시스템이 된다.

**한 training step을 dtype 원장으로 재생한다**

구체적인 decoder block 하나를 따라가자. embedding output이 BF16으로 block에 들어오고, RMSNorm reduction은 FP32 accumulator를 사용한다고 가정한다. normalized activation은 FP8로 quantize되어 GEMM에 들어가며, GEMM accumulator는 더 높은 정밀도를 쓰고 output은 BF16으로 돌아온다. residual add는 BF16 또는 FP32 중 recipe가 정한 경로를 따른다. 이 한 문장 안에도 storage buffer, kernel input, accumulator, output과 residual의 다섯 dtype이 있다.

attention에서는 Q·K·V projection, score 계산, softmax state와 value accumulation의 dtype을 각각 기록한다. causal mask의 sentinel이 낮은 dtype에서 representable한지 확인한다. fully masked row가 있다면 output 정책을 test한다. fused attention이 online softmax를 쓰면 running max와 sum의 dtype, tile boundary rescale을 source card에 넣는다.

MLP에서는 gate와 up projection output, activation function, elementwise product와 down projection을 본다. SwiGLU류 product가 두 낮은 정밀도 tensor를 곱할 때 underflow·rounding이 어떻게 누적되는지 probe한다. fusion이 intermediate를 register에 유지하면 profiler에서는 tensor가 보이지 않을 수 있으므로 generated kernel과 reference hook을 함께 쓴다.

loss에서는 logits projection과 cross entropy를 분리한다. vocabulary parallel이면 global maximum과 denominator를 어느 collective와 dtype으로 얻는지 본다. ignore token이 많은 batch에서 local valid count와 global loss scale을 검산한다. fused CE가 logits materialization을 피하더라도 label smoothing과 gradient가 reference와 같아야 한다.

backward에서는 각 saved tensor의 dtype과 recompute 여부를 기록한다. activation checkpointing이 forward를 다시 실행할 때 FP8 scale history와 RNG가 같은 의미를 유지하는지 본다. gradient가 BF16 또는 FP16 buffer에 쓰이고 bucket reduction이 FP32 또는 낮은 dtype으로 이루어지는지 확인한다. communication compression은 optimizer input의 오차 예산에 포함한다.

optimizer boundary에서 GradScaler가 unscale하고 global norm을 계산하며 clip한다. fused optimizer가 FP32 master weight와 moments를 갱신하고 model parameter로 cast한다. update 성공 여부가 scaler·optimizer·scheduler의 같은 commit generation에 기록된다. checkpoint는 이 모든 mutable state를 하나의 manifest로 묶는다.

**원장의 최소 열**

각 edge마다 producer symbol, consumer symbol, logical tensor, shape, storage dtype, compute·accumulate dtype, scale owner, lifetime, stream, checkpoint 여부와 reference test를 둔다. 같은 tensor 이름이 view 또는 alias로 바뀌면 storage identity도 기록한다. graph capture에서는 주소 안정성 열을 추가한다.

한 step의 원장을 작성하면 “BF16 training”이라는 표현이 얼마나 불완전한지 드러난다. 실제 recipe는 수십 개 경계의 조합이다. 성능 최적화는 그중 일부 경계를 fusion하거나 dtype을 바꾼다. candidate diff는 config 한 줄이 아니라 이 원장의 변경 행으로 검토한다.

**FP8 scale 지연을 수치로 이해한다**

간단히 FP8 최대 유한 magnitude를 (F), 관측 amax를 (a), margin을 (m)이라 하자. 이상화한 scale은 대략 (s=F/(a2^m)) 형태로 생각할 수 있다. 실제 library는 power-of-two rounding, epsilon, format과 recipe에 따라 다르므로 source 식을 확인한다. 여기서는 scale이 range와 precision을 어떻게 교환하는지 보기 위한 직관이다.

정상 step의 amax가 8인데 갑자기 128 outlier가 들어오면 history max 방식은 scale을 16분의 1 수준으로 줄일 수 있다. 다음 여러 step의 실제 amax가 다시 8이어도 history window에 outlier가 남아 있으면 작은 값이 FP8 grid의 좁은 부분을 사용한다. saturation은 줄지만 quantization error가 늘 수 있다. history length가 state와 품질에 영향을 주는 이유다.

반대로 현재 amax만 쓰면 outlier 다음 step에 scale이 즉시 돌아오지만 step별 scale 변화가 크다. distributed rank마다 amax가 다를 때 local scale을 쓰면 동일 logical operation의 shard마다 quantization이 달라진다. global max reduction은 일관성을 높이지만 communication과 synchronization을 추가한다. reduction group은 DP, TP 또는 module shard 의미에 맞아야 한다.

scale metadata를 checkpoint하지 않고 재개하면 첫 history window가 비어 default scale을 사용할 수 있다. output은 finite여도 baseline과 다르고, 새 history가 채워지는 동안 transient가 생긴다. resume fixture는 checkpoint 직전 outlier를 배치해 누락을 민감하게 만든다. history contents와 index, 다음 scale을 독립 oracle과 비교한다.

**scale telemetry**

모든 tensor의 amax를 label로 노출하면 cardinality가 폭발한다. module group과 tensor role별 histogram 또는 sampled trace를 사용하고, anomaly 때 원본 state를 좁혀 조회한다. saturation fraction, zero fraction, scale change ratio, history age를 함께 본다. amax 하나만으로 유효 precision을 알 수 없다.

scale 경보는 fixed threshold보다 baseline distribution과 recipe bound를 조합한다. saturation 급증, zero fraction 증가, scale oscillation과 rank divergence를 탐지한다. data mixture와 sequence-length event를 annotation해 정상적인 distribution shift를 장애로 오인하지 않는다.

**CUDA 세대 전환에서 재현 가능한 benchmark를 만든다**

baseline과 candidate는 같은 GPU clock/power policy, process placement, input shape와 data를 사용한다. thermal state와 다른 workload 간섭을 기록한다. 여러 node를 섞기 전에 한 GPU kernel, 한 node step, multi-node 순으로 범위를 넓힌다. 각 단계가 다른 원인을 분리한다.

kernel benchmark는 warmup 횟수와 측정 반복, stream, synchronization, cache 상태를 적는다. median뿐 아니라 tail과 분산을 본다. autotune이 algorithm을 선택한다면 선택 결과와 workspace를 저장한다. CUDA/library version이 heuristic을 바꾸면 동일 operator가 다른 kernel을 사용할 수 있다.

end-to-end step은 data와 checkpoint를 제외한 pure training step과 포함한 운영 step을 나눈다. compiler·graph cold cost를 별도로 측정한다. 긴 run에서 amortize되는지, elastic restart가 잦아 반복되는지 판단한다. 평균만 보고 deployment latency를 숨기지 않는다.

성능 차이를 발견하면 profiler trace를 kernel mapping으로 정렬한다. 이름이 바뀌었으면 operator call card와 shape로 대응시킨다. launch count, duration, DRAM byte, occupancy, communication overlap을 비교한다. 한 kernel의 개선이 다른 stream의 대기를 늘렸는지 critical path에서 본다.

**통계와 승인**

반복 측정은 process restart와 within-process iteration을 구분한다. confidence interval을 보고 noise보다 작은 차이를 승격하지 않는다. node 간 편차가 크면 candidate 효과와 hardware variability를 mixed effect로 분리하거나 paired test를 사용한다.

성능 승인은 correctness gate 뒤에 한다. candidate가 빠르지만 FP32 reference 오차 또는 resume invariant를 넘으면 기각한다. 반대로 작은 regression이 있지만 공식 지원과 안정성을 위해 migration해야 한다면 비용을 명시하고 별 최적화 계획을 둔다. 수치를 숨겨 “동등”이라 하지 않는다.

**source·test·trace를 한 삼각형으로 묶는다**

source는 의도와 가능한 경로를 보여 주지만 실제 dispatch를 증명하지 않는다. test는 특정 입력의 계약을 보여 주지만 production shape를 모두 대표하지 않는다. trace는 실제 kernel을 보여 주지만 왜 그 경로가 선택되었는지와 독립 correctness를 설명하지 않는다. 세 증거를 함께 사용한다.

source card에는 revision, path, symbol, guard와 mutable state가 있다. test card에는 입력 shape·dtype·hardware condition, expected output/state와 tolerance가 있다. trace card에는 RunID, environment fingerprint, operator-to-kernel mapping, stream과 timing이 있다. 공통 EvidenceID로 연결한다.

framework upgrade 때 source symbol이 이동하면 content hash와 semantic role로 새 좌표를 찾고 diff한다. test가 삭제되거나 skip 범위가 넓어지면 support 약화로 본다. production trace에서 fast-path hit rate가 줄면 dynamic shape나 guard 변화와 연결한다.

**짧은 코드 인용의 원칙**

책에는 guard 또는 state update를 이해하는 데 필요한 핵심 줄만 인용한다. 주변 전제와 caller를 말로 설명하고 고정 revision 좌표를 제공한다. 긴 implementation을 복제해 독자를 코드 숲에 버리지 않는다. 인용이 최신 main branch라고 주장하지 않고 분석한 checkout을 명시한다.

수학식은 코드와 양방향으로 연결한다. online softmax 식의 running max가 어느 local variable인지, GradScaler 상태 전이가 어느 branch인지, scale 식의 margin이 어느 recipe field인지 붙인다. 식만 정확하고 호출 경로가 틀린 설명을 피한다.

**production 전 migration 리허설**

리허설은 baseline checkpoint에서 candidate environment를 시작한다. 첫 step 전에 loaded library와 graph/compiler cache 상태를 기록한다. 고정 probe batch로 eager BF16 reference, FP8, fused와 graph path를 차례로 실행한다. 각 단계의 output·gradient·state digest를 비교한다.

그 뒤 overflow를 주입해 모든 rank가 update를 skip하고 scaler·scheduler가 합의하는지 본다. checkpoint를 만들고 새 process에서 scale history, optimizer master와 graph recapture를 검사한다. odd shape로 fallback을 유도하고 unsupported path가 명시적으로 드러나는지 확인한다.

다음은 node 하나를 제거하는 대신 소규모 communicator 재구성 fixture로 topology change를 흉내 낸다. FP8 amax group과 graph collective가 새 group을 참조하는지 static/source 수준과 test로 확인한다. 대규모 training runtime을 돌리지 않고도 state ownership 오류를 상당 부분 찾을 수 있다.

마지막으로 baseline으로 rollback해 preserved checkpoint를 읽고 같은 probe가 통과하는지 본다. candidate가 쓴 새 schema checkpoint만 남았다면 rollback 준비가 실패한 것이다. release manifest에는 baseline과 candidate 양쪽의 읽기 가능 범위를 쓴다.

승인 회의에서 각 owner는 하나의 질문에 답한다. build owner는 어떤 compiler와 library가 실제 binary에 들어갔는가, numerical owner는 오차 예산을 무엇으로 증명했는가, runtime owner는 graph·stream·allocator 상태를 어떻게 복구하는가, training owner는 skipped update와 checkpoint의 의미가 보존되는가를 답한다.

이 리허설이 완료되면 CUDA 12.x에서 13.x로의 전환은 더 이상 모호한 업그레이드가 아니다. 환경·binary·dtype·kernel·state의 변화가 추적 가능한 migration이 된다. 문제가 생겼을 때도 “CUDA가 이상하다”가 아니라 최초로 달라진 library, dispatch, scale 또는 stream edge까지 내려갈 수 있다.

**수치·binary·runtime 증거를 읽는 순서**

첫 파일은 environment fingerprint다. GPU·driver·toolkit·framework·library·extension과 build flag가 한 generation으로 묶여야 한다. 두 번째는 dtype ledger다. 모델의 주요 경로마다 storage, compute, accumulation, reduction과 optimizer dtype을 보여 준다. 세 번째는 dispatch table이며 production shape 분포에서 fast path와 fallback이 실제로 얼마나 선택되었는지 담는다.

네 번째는 numerical report다. FP32 또는 독립 reference에 대한 forward·backward·optimizer delta, adversarial input과 long-horizon probe를 포함한다. 다섯 번째는 state report다. GradScaler, FP8 metadata, optimizer master, RNG와 graph scalar가 checkpoint/resume에서 같은 generation인지 증명한다. 여섯 번째는 profiler와 memory timeline이다. 성능 이득의 원인이 launch, HBM traffic, overlap 또는 compiler cache 중 무엇인지 드러낸다.

이 순서를 지키면 빠른 trace만 보고 정확성을 합리화하지 않게 된다. 환경과 의미를 고정하고, 경로를 확인한 뒤, 오차와 상태를 검증하고, 마지막에 성능을 판단한다. 어느 파일이 누락되었으면 그 층의 결론은 아직 가설이다.

reviewer는 표본 tensor 하나를 골라 모든 파일을 관통해 추적한다. source의 dispatch guard가 trace의 kernel과 맞는지, dtype ledger의 accumulator가 generated code와 맞는지, numerical error가 예산 안인지, checkpoint에 scale state가 있는지 확인한다. 서로 다른 문서가 같은 옵션 이름만 공유하는 것이 아니라 동일 EvidenceID와 recipe generation으로 연결되어야 한다.

장애가 발생하면 같은 묶음을 역순으로 사용한다. 성능 trace에서 최초 이상 kernel을 찾고, state와 numerical report로 의미 차이를 좁힌 뒤, dispatch와 dtype ledger를 거쳐 environment/library 변경까지 올라간다. 무작정 toolkit을 재설치하거나 precision을 높이지 않는다. 가장 작은 반증 fixture를 추가하고 수정 뒤 전체 ladder를 다시 실행한다.

최종 release note에는 지원 범위와 미지원 범위를 똑같이 분명히 쓴다. 검증하지 않은 GPU, shape, dtype와 graph 조합은 “아마 동작”이 아니라 unvalidated다. fallback의 성능·정확성 의미도 적는다. 운영자는 incident 도중 source를 새로 해석하지 않고 이 계약으로 admission과 rollback을 결정할 수 있어야 한다.

이렇게 증거 묶음을 닫으면 CUDA 12.x와 13.x의 차이, AMP와 FP8, fusion과 capture가 서로 떨어진 기술 목록으로 남지 않는다. 모두 tensor가 어느 표현으로 어느 kernel과 stream을 지나며, 어떤 mutable state를 남기고, 다음 step과 checkpoint에 어떻게 이어지는가라는 하나의 실행 서사로 합쳐진다. 이 서사가 바로 저정밀 학습을 설명하고 디버깅할 수 있는 최소 단위다.

마지막 표본 검사는 서로 다른 두 node에서 같은 artifact를 로드해 수행한다. library map, kernel dispatch, first-step digest와 scale state가 같아야 한다. 차이가 나면 node image, driver와 GPU capability까지 evidence에 올린다. 동일 container라는 사실만으로 실행 환경이 같다고 판단하지 않는다.

baseline에는 의도적으로 지원하지 않는 shape와 dtype도 넣는다. 명시적 오류 또는 문서화된 fallback이 나와야 한다. silent coercion이나 우연한 kernel 선택은 release blocker다. 실패 경로까지 예측 가능해야 운영 중 새 입력이 들어왔을 때 정확성과 성능을 설명할 수 있다.

이 최종 검산을 통과한 recipe만 immutable ID로 배포한다. 이후 옵션, compiler flag, library 또는 hardware가 하나라도 달라지면 새 generation이며 핵심 ladder를 다시 수행한다. 검증 결과를 이전 이름 아래 덮어쓰지 않는다. 이 규율이 저정밀 최적화를 재현 가능한 공학으로 유지한다.

인수자는 마지막으로 원본 test의 skip 조건과 CI hardware를 읽는다. 통과한 badge가 실제 production GPU와 FP8·graph 경로를 실행했는지 확인한다. 실행되지 않은 test는 성공 증거가 아니다. coverage가 비어 있으면 소규모 고정 fixture를 직접 추가하고 결과와 환경 지문을 반드시 함께 끝까지 완전하게 보존한다.

**CUDA 12.x와 13.x를 지원 문장 대신 실행 계약으로 고정한다**

CUDA 세대를 비교할 때 가장 먼저 고정할 값은 `driver_version`, process가 실제 적재한 runtime과 수학 library, extension을 만든 toolkit, 생성된 device code의 target이다. 이 네 값은 서로 독립적이다. 새 driver에서 CUDA 12 계열 runtime을 포함한 wheel을 실행하는 경우와, CUDA 13 계열 toolkit으로 같은 extension source를 다시 컴파일하는 경우는 전혀 다른 시험이다. 전자는 binary 실행 호환성의 질문이고 후자는 source, host compiler, header, linker와 device-code 생성의 질문이다. 옵션은 `CUDA_HOME`, wheel 선택, `TORCH_CUDA_ARCH_LIST`, build isolation이다. 상태는 include와 library search path, ABI macro, cubin과 PTX section이다.

효과는 import 가능성, 최초 kernel load, JIT 발생 여부와 실제 instruction path다.

공식 compatibility 문서를 읽을 때도 이 층을 섞지 않는다. driver의 minimum version 표는 application에 포함된 모든 third-party extension의 ABI를 보장하지 않는다. toolkit release note가 새 compiler와 library를 제공한다는 사실은 사용 중인 framework wheel이 그 조합을 지원한다는 뜻이 아니다. framework의 설치 표가 특정 CUDA build를 제공한다는 사실도 local `nvcc`로 만든 임의 extension을 자동 보증하지 않는다. 따라서 환경 원장에는 주장마다 근거 종류를 붙인다. `driver-runtime`, `framework-distribution`, `extension-build`, `device-coverage` 네 칸 가운데 빈 칸이 있으면 전체 조합은 검증되지 않았다.

C++ extension에는 host ABI가 한 층 더 있다. `_GLIBCXX_USE_CXX11_ABI`, C++ 표준, host compiler major, pybind와 framework header가 기대하는 symbol, linker가 선택한 `libstdc++`를 기록한다. CUDA library ABI와 C++ ABI를 “CUDA 버전” 한 칸에 접지 않는다. undefined symbol이 뜨면 먼저 오류 symbol의 제공 library, `DT_NEEDED`, `RPATH`와 실제 loader map을 맞춘다. 환경 변수로 다른 library를 앞세워 import가 성공해도 lazy binding 때문에 첫 fused op에서 실패할 수 있다. import, device initialization, representative forward, backward, optimizer와 collective를 순서대로 실행하는 이유다.

다음은 build manifest의 한 예다. 값 자체보다 누가 만들고 누가 검증하는지가 중요하다.

```yaml
build_id: lp-kernel-cu13-sm100-r7
source_revision: "고정 커밋 해시"
framework:
  package_version: "검증한 정확한 버전"
  embedded_cuda_runtime: "검증한 정확한 버전"
host:
  compiler: "컴파일 로그에서 추출"
  cxx_standard: "c++17"
  cxx11_abi: 1
device_code:
  sass_targets: ["sm_90", "sm_100"]
  ptx_targets: ["compute_100"]
libraries:
  expected: ["cudart", "cublas", "nvrtc", "nccl"]
validation:
  gpu_models: ["실제로 시험한 장치 식별자"]
  cold_jit: true
  warm_jit: true
```

이 manifest는 build 성공 때 자동 생성하고 artifact 내부에도 넣는다. runtime은 자신의 device capability가 `sass_targets`에 있으면 직접 적재를 기대하고, PTX만 있으면 driver JIT 경로와 cache permission을 확인한다. target이 없으면 조용한 CPU 또는 느린 kernel fallback을 허용하지 않고 명확히 실패시킨다. PTX를 포함했다는 이유로 미래 장치의 성능을 약속하지 않는다. correctness 가능성과 architecture-specific tuning은 별 gate다.

CUDA 12.x에서 13.x로 이동하는 시험 행렬은 한 축만 바꾼 paired experiment를 포함한다. 같은 binary를 두 지원 driver에서 실행하면 driver/runtime 관계를 본다. 같은 source와 dependency revision을 두 toolkit으로 clean build하면 compiler/codegen 차이를 본다. 같은 candidate binary를 eager와 compile 경로에서 실행하면 framework compiler 차이를 본다. framework, Triton, Transformer Engine과 toolkit을 한꺼번에 올린 결과는 최종 통합 시험으로는 유용하지만 원인 규명 시험으로는 부족하다.

**ABI 실패를 재현 가능한 fixture로 만든다**

ABI 시험은 오류가 날 때까지 library path를 섞는 방식이 아니다. 정상 container를 복제한 격리 환경에서 extension이 요구하는 symbol 목록과 제공 library 목록을 비교한다. loader debug output은 절대 경로와 content hash를 남기되 시스템 전체 환경 변수와 credential은 수집하지 않는다. 예상하지 않은 host toolkit library가 wheel bundled library보다 먼저 적재되면 실패다. 우연히 symbol이 호환되어 실행되더라도 reproducibility gate에서 막는다.

negative fixture는 세 종류면 충분히 강하다. 첫째, 지원하지 않는 SM만 담은 작은 extension은 load 또는 첫 launch에서 의도한 오류를 내야 한다. 둘째, ABI macro가 다른 작은 C++ extension은 link/import 경계가 명확해야 한다. 셋째, 필요한 shared object 하나를 격리해 loader가 dependency 이름과 search path를 알려야 한다. production artifact를 손상시키지 않고 별 fixture에서 수행한다. 실패 메시지와 진단 명령이 runbook과 같은지 CI에서 확인한다.

**PyTorch AMP를 호출 순서가 있는 상태 기계로 시험한다**

AMP의 핵심 옵션은 autocast의 `device_type`, compute dtype, enable flag, GradScaler의 initial scale과 growth 정책이다. autocast 진입은 parameter storage를 영구 변환하지 않는다. dispatcher가 연산별 policy에 따라 입력을 변환하고 output dtype을 정한다. 따라서 “모델이 BF16이다”라는 한 문장 대신 parameter storage, matrix input, accumulator, reduction output, gradient storage를 표로 남긴다. 7장의 normalization과 11장의 optimizer master state를 같은 원장에 연결해야 forward가 끝난 뒤의 수치 상태를 설명할 수 있다.

다음 fixture는 autocast 정책과 backward 상태를 분리한다. 예시는 개념을 고정하기 위한 최소 코드이며 실제 지원 device에서 tolerance를 calibration해야 한다.

```python
def amp_step(model, batch, optimizer, scaler, amp_dtype):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(batch["input_ids"])
        loss_sum = token_loss_sum(logits, batch["labels"])
        valid = (batch["labels"] != -100).sum()
        loss = loss_sum / valid.clamp_min(1)
    scaled = scaler.scale(loss)
    scaled.backward()
    scaler.unscale_(optimizer)
    grad_snapshot = snapshot_grad_statistics(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return loss.detach(), grad_snapshot, old_scale, scaler.get_scale()
```

테스트는 네 상태를 관찰한다. backward 직후 gradient는 아직 scale이 적용된 상태다. `unscale_` 뒤에는 clipping threshold와 같은 단위를 갖는다. `step`은 finite 검사 결과에 따라 parameter commit을 수행하거나 건너뛴다. `update`는 scaler의 다음 상태를 정한다. 옵션을 BF16으로 바꾸었다고 scaler가 항상 불필요하다고 단정하지 않는다. BF16의 exponent 범위가 넓어도 upstream loss, custom FP16 kernel 또는 잘못된 reduction 때문에 non-finite가 생길 수 있다. BF16 경로에서 scaler를 끄는 결정은 overflow가 절대 없다는 선언이 아니라 scale-underflow 완화가 필요하지 않다는 실험 결과다.

overflow fixture에서는 특정 parameter hook으로 `inf` gradient 하나를 넣는다. 기대 효과는 모든 관련 optimizer parameter가 그대로이고 optimizer moment와 step counter도 commit되지 않는 것이다. scheduler가 loop iteration이 아니라 committed update를 기준으로 움직이는지 검사한다. 여러 optimizer가 있으면 found-inf가 어떻게 결합되는지 별 계약이 필요하다. 한 optimizer만 건너뛰고 다른 optimizer가 전진하는 정책이라면 checkpoint generation과 scheduler도 둘로 나누어야 한다. 의도하지 않은 partial update는 가장 위험한 silent failure다.

accumulation 중 overflow도 시험한다. microbatch 1의 finite gradient에 microbatch 2가 `inf`를 더하면 마지막 unscale에서 전체 update를 건너뛰어야 한다. overflow 뒤 gradient buffer를 지우지 않으면 다음 accumulation window가 오염된다. option은 accumulation length와 `zero_grad` 위치다. 상태는 scaled gradient buffer와 found-inf flag다. 효과는 commit 여부와 다음 window의 출발점이다. 이 연결을 13장의 scheduler step 정의와 15장의 global gradient denominator에 상호참조한다.

**autocast 경계의 고정 테스트**

operator별 dtype을 문서 기억에 의존하지 않고 runtime hook과 profiler로 수집한다. linear, normalization, softmax, loss reduction, custom extension 각각에 input/output dtype과 accumulator 근거를 붙인다. generated Triton 또는 SASS에서 accumulator를 확인할 수 없는 연산은 independent high-precision reference와 adversarial range test를 사용한다. 큰 양수와 음수가 섞인 reduction, 거의 같은 logits, 매우 작은 gradient가 경계를 드러낸다.

autocast nested disabled region도 시험한다. 특정 normalization을 FP32로 강제하면 입력을 명시적으로 FP32로 올렸는지, region을 나온 output이 다음 BF16 matmul로 어떻게 변환되는지 본다. option 하나는 cast kernel 수와 temporary allocation도 바꾼다. 정확도만 좋아지고 graph break와 bandwidth overhead가 커질 수 있으므로 dtype trace와 performance trace를 같은 RunID로 묶는다.

**Transformer Engine의 FP8와 NVFP4를 metadata 경로까지 검증한다**

FP8 module의 논리 값은 작은 형식의 payload만으로 복원되지 않는다. format, scale 또는 inverse scale, amax history, update index, recipe와 granularity가 함께 있어야 한다. E4M3와 E5M2는 exponent와 fraction의 배분이 다르므로 forward activation과 backward gradient에 서로 다른 형식을 쓰는 recipe가 가능하다. option은 recipe, history 길이, margin, per-tensor 또는 block granularity다. 상태는 payload와 metadata tensor다. 효과는 saturation, zero 비율, 통신과 cast overhead, checkpoint 크기와 resume trajectory다.

Transformer Engine을 쓸 때는 FP8 context가 활성화됐는지보다 실제 module과 kernel 경로를 확인한다. module conversion 뒤 extra state key와 parameter wrapper를 기록한다. forward마다 amax가 어느 stream에서 갱신되고 scale update가 현재 step 또는 다음 step에 적용되는지 확인한다. distributed amax reduction을 쓰면 process group도 state 계약의 일부다. 15장의 DP·TP group 소유권과 맞지 않는 group을 쓰면 rank마다 같은 이름의 module이 다른 scale generation을 가질 수 있다.

FP8 reference test는 세 층으로 나눈다. 첫 층은 quantize-dequantize oracle이다. 경계값, 0, 매우 작은 값, representable maximum 전후와 NaN/Inf 정책을 시험한다. 둘째는 한 linear layer의 forward/backward다. weight와 activation의 scale을 독립적으로 흔들고 FP32 accumulation reference에 대해 output RMS, maximum absolute error, gradient cosine을 본다. 셋째는 두 layer와 residual이다. 첫 layer의 quantization error가 두 번째 layer scale을 바꾸는 장기 상태 효과를 확인한다.

NVFP4처럼 더 작은 형식 또는 block-scaled 경로에서는 metadata granularity가 훨씬 중요하다. “4비트”라는 저장 폭만으로 실제 memory byte와 kernel 계약을 계산할 수 없다. block마다 scale이 있고 scale 자체의 표현, alignment, padding과 packing header가 있다. logical element 수가 `N`, payload가 원소당 4 bit, block 크기가 `G`, scale metadata가 block당 `M` byte라면 이상적 byte는 `N/2 + ceil(N/G)M`이고 실제 byte는 tile alignment와 workspace를 더한다. profiler allocation과 artifact byte를 이 식에 맞춘다.

NVFP4 option을 켰을 때 상태 변화는 weight block layout, activation block scale, kernel tile과 calibration state다. 효과는 HBM traffic 감소 가능성과 더 좁은 local dynamic range다. outlier가 block을 지배하는 fixture, block 경계에서 outlier 위치만 바꾸는 fixture, non-multiple tail block을 반드시 넣는다. tail padding이 amax에 들어가거나 scale index가 한 칸 밀리면 정상 random input에서 잘 보이지 않는 systematic error가 생긴다.

checkpoint round trip은 payload checksum만 비교하지 않는다. save 직전의 scale history, circular index와 다음 scale을 저장하고, 새 process에서 동일 probe를 한 번 실행해 output과 “실행 뒤 metadata”를 비교한다. uninterrupted path와 resumed path가 첫 output은 같지만 history index가 달라 두 번째 output부터 어긋나는 경우도 잡는다. topology를 바꾸면 replicated history와 sharded block scale을 어떻게 재배치하는지 명시한다. 지원되지 않는 재배치는 명확히 거절한다.

**format fallback은 상태 전이다**

지원하지 않는 shape에서 BF16으로 fallback하면 output dtype만 같아 보일 수 있다. 그러나 FP8 amax가 갱신되지 않았는지, 다음 supported shape가 stale history를 쓰는지 확인해야 한다. fallback은 kernel 선택만이 아니라 metadata state transition이다. dispatch table에는 `used_format`, `fallback_reason`, `amax_updated`, `scale_generation`을 기록한다. odd shape 다음 정상 shape의 결과까지 시험한다.

## 14.8 kernel oracle에서 운영 runbook까지

Triton이나 fused kernel의 정확성은 한 번의 출력 비교에서 장기 안정성으로 확장되어야 한다. source 좌표, 작은 oracle, boundary failure를 묶고 operator 승인 카드와 변경 diff를 거쳐 증상별 runbook으로 연결한다.

### 14.8.1 Triton kernel을 source, oracle, failure 세 묶음으로 구현한다

Triton kernel 검토는 `@triton.jit` 함수만 읽고 끝내지 않는다. Python wrapper가 shape guard, stride, device와 dtype을 검증하고 launch grid와 meta-parameter를 정한다. kernel은 pointer arithmetic, mask, accumulator와 store cast를 정한다. autotune은 candidate와 cache key를 정한다. 이 셋이 합쳐진 것이 실행 계약이다. option은 `BLOCK_SIZE`, warp 수, stage 수와 fast-math다. 상태는 compiled variant와 autotune cache다. 효과는 coverage, numerical order, occupancy와 cold latency다.

다음은 row reduction kernel의 테스트 골격이다. 실제 구현은 repository의 고정 symbol에 맞추되 oracle 구조는 유지한다.

```python
@pytest.mark.parametrize("n", [1, 31, 32, 33, 127, 128, 129, 4097])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_row_sum_kernel(n, dtype):
    x = adversarial_rows(rows=7, cols=n, dtype=dtype, device="cuda")
    x_ref = x.float().sum(dim=-1)
    out = triton_row_sum(x)
    torch.testing.assert_close(out.float(), x_ref, rtol=budget(dtype, n),
                               atol=absolute_budget(dtype, n))
    assert torch.isfinite(out).eq(torch.isfinite(x_ref)).all()
```

`31, 33, 129`는 tile mask 경계를 검증한다. `4097`은 여러 tile 또는 큰 block 선택을 자극한다. adversarial row는 같은 부호만이 아니라 큰 값과 작은 반대 부호, alternating values, zero와 subnormal을 포함한다. reference도 입력 dtype에서 먼저 sum하면 같은 오류를 복제하므로 FP32 또는 필요하면 FP64로 올린다. tolerance는 `n`에 따라 reduction error가 자랄 수 있음을 반영하되, 잘못된 mask 하나를 허용할 정도로 넓히지 않는다.

backward kernel은 finite difference만으로 승인하지 않는다. finite difference는 step 크기와 저정밀 rounding에 민감하다. PyTorch 고정 reference gradient, analytic invariant와 일부 FP64 difference를 조합한다. fused softmax-cross-entropy라면 각 valid row gradient 합이 거의 0인지, ignored row가 정확히 0인지, loss reduction denominator가 valid token 수인지 확인한다. vocabulary tile tail, 매우 큰 target logit, target이 마지막 원소인 경우가 필수다.

source failure fixture는 mask를 의도적으로 제거한 production 변형을 만들지 않는다. test-only buggy kernel 또는 wrapper monkeypatch로 tail read를 유도해 sanitizer와 oracle이 실패하는지 본다. wrong stride, noncontiguous input, misalignment는 wrapper가 거절하거나 올바른 generic path를 선택해야 한다. silent contiguous assumption을 피한다. error message에는 expected layout과 받은 stride를 포함한다.

autotune cache는 shape, dtype, capability와 compiler generation을 충분히 키에 넣어야 한다. CUDA 세대나 GPU가 바뀐 cache를 재사용하는 negative test를 둔다. 선택된 config가 correctness에 영향을 주면 본질적으로 kernel bug다. 모든 config가 같은 oracle을 만족한 뒤에만 timing으로 선택한다. 후보 하나가 illegal access를 내면 전체 process가 오염될 수 있으므로 autotune correctness screening을 격리 process에서 수행한다.

**fused forward와 backward의 저장 상태**

fused kernel마다 backward가 요구하는 saved tensor 목록을 작성한다. attention은 output, log-sum-exp, mask/sequence metadata와 dropout RNG 좌표가 필요할 수 있다. normalization은 mean과 inverse standard deviation 또는 재계산 입력이 필요하다. cross entropy는 row max와 denominator, target와 ignore mask가 필요할 수 있다. option으로 recompute를 켜면 저장 byte는 줄고 forward 수학이 backward 중 다시 실행된다. 상태는 saved tensor에서 RNG와 input lifetime으로 이동한다. 효과는 HBM과 compute뿐 아니라 mutation 위험이다.

in-place residual이 recompute 입력을 덮어쓰는 fixture를 만든다. eager unfused는 통과하지만 fused backward가 틀릴 수 있다. tensor version counter와 alias 검사, cloned reference로 잡는다. activation checkpointing과 fused dropout을 함께 켠 경우 2장의 chain rule과 9장의 residual 경계를 따라 동일 RNG coordinate가 재생되는지 확인한다. seed만 같고 kernel이 소비한 counter 수가 다르면 mask가 달라진다.

**PyTorch compile과 kernel dispatch를 증거로 남긴다**

`torch.compile` option은 boolean이 아니다. backend, mode, dynamic shape, full graph 요구, cache와 guard가 state를 바꾼다. graph capture 결과, guards, generated code, compiled artifact와 cache generation이 새 상태다. 효과는 fusion, launch 수, memory 계획, numerical order와 recompilation이다. 옵션→상태→효과를 기록하지 않으면 compiled run의 성능과 오류를 eager run에 잘못 귀속한다.

고정 test는 세 경로를 같은 입력으로 실행한다. eager FP32는 수치 oracle, eager target dtype은 precision baseline, compiled target dtype은 compiler delta다. forward, backward와 optimizer parameter delta를 각각 비교한다. target dtype eager부터 실패하면 compiler 문제로 분류하지 않는다. compiled에서만 실패하면 graph partition을 줄이고 한 operator 단위로 재현한다. full graph 실패를 피하려고 graph break를 허용했다면 break count와 eager island가 dispatch report에 남아야 한다.

dynamic shape test는 단순히 여러 길이가 실행되는지만 보지 않는다. 길이 sequence `128, 129, 128, 257, 129`를 보내 compile 횟수와 cache hit를 본다. 각 길이에서 fast path 또는 fallback, workspace와 output tolerance를 기록한다. recompile storm은 correctness pass이지만 production performance fail이다. shape guard가 지나치게 넓어 unsupported kernel에 들어가는 것은 correctness fail이다.

PyTorch SDPA backend 선택도 같은 방법으로 다룬다. option이 특정 backend를 허용하거나 금지한다. 상태는 dispatch predicate, chosen kernel과 saved backward state다. 효과는 attention matrix materialization, RNG 사용과 tolerance다. causal, arbitrary mask, GQA, variable length와 dropout 조합마다 실제 backend를 assert한다. 요청한 backend가 불가능할 때 명확히 실패할지 fallback할지는 제품 계약으로 정하고 test한다.

**kernel failure를 증상에서 최초 잘못된 상태로 역추적한다**

`illegal memory access`가 보고된 Python 줄은 실패 kernel과 다를 수 있다. 비동기 launch 때문에 다음 synchronization에서 드러나기 때문이다. 격리된 최소 fixture에서 동기화 지점을 추가하고 stream별 event를 기록한다. sanitizer를 사용하면 timeout과 memory overhead가 커지므로 작은 shape부터 시작한다. 실패 뒤 같은 process의 tensor 값을 근거로 계속 분석하지 않는다. CUDA context가 손상되었을 수 있으므로 artifact를 남기고 process를 재시작한다.

`misaligned address`는 pointer base만 보지 않는다. slice offset, element size, vectorized load 폭과 stride를 계산한다. wrapper guard가 alignment를 증명했는지 source에서 확인한다. contiguous flag가 true여도 storage offset 때문에 요구 alignment가 깨질 수 있다. test에는 aligned allocation의 offset view를 넣는다. specialized vector path가 generic scalar path로 fallback해야 하는 경우 성능 기대도 함께 고친다.

NaN은 마지막 loss에서 거슬러 올라가기보다 boundary probe를 이분 탐색한다. input finite, cast 뒤 saturation/zero, matmul accumulator, normalization statistic, softmax row max와 denominator, gradient unscale, optimizer moment를 본다. 최초 non-finite와 최초 reference tolerance 초과는 다를 수 있다. finite지만 큰 quantization drift가 몇 layer 뒤 NaN을 만들 수 있으므로 두 사건을 모두 보존한다.

silent corruption은 canary와 invariant가 필요하다. output padding 영역의 sentinel, row-sum invariant, untouched ignored gradient, replicated parameter checksum을 사용한다. kernel 뒤 allocator가 재사용되기 전에 검사한다. random loss 하나가 맞았다는 사실은 out-of-bounds write가 없다는 증거가 아니다. 26장의 모니터링 경보는 production에서 sampling된 invariant를 수집하고, 이 장의 fixture는 원인 재현을 담당한다.

**수치 허용오차를 사후에 넓히지 않는다**

tolerance는 reference 결과를 보기 전에 dtype, reduction 길이와 algorithm 차이에 따라 정한다. absolute와 relative tolerance를 함께 쓰고 0 근처 값에는 absolute 기준을 적용한다. gradient cosine이 높아도 특정 작은 parameter의 부호가 뒤집힐 수 있으므로 max error와 parameter update error를 함께 본다. FP8와 NVFP4는 saturation/zero fraction이라는 구조 지표도 필수다.

실패 뒤 tolerance를 넓히려면 독립 근거와 새 RecipeID가 필요하다. task metric이 유지됐다는 이유만으로 kernel invariant를 포기하지 않는다. 반대로 bitwise equality를 모든 CUDA 세대에 강제해 합법적 reduction order 차이를 장애로 만들지도 않는다. numerical-equivalent, trajectory-equivalent, metric-equivalent의 요구 수준을 실험 목적에 맞춰 구분한다.

**저정밀 release를 한 step의 commit protocol로 닫는다**

한 training step의 commit 전 상태에는 parameter, optimizer moment, scaler, FP8/NVFP4 metadata, RNG, scheduler와 data cursor가 있다. forward와 backward는 후보 상태를 만들고 finite 검사와 distributed 합의가 commit 여부를 정한다. overflow로 skip하면 parameter와 moment, scheduler의 committed-update counter가 움직이지 않아야 한다. 그러나 data cursor를 되돌릴지는 별 정책이다. replay한다면 RNG도 되돌려야 하고, 소비한다면 exact sample trajectory가 달라짐을 선언한다.

fused optimizer가 자체 overflow 검사를 하면 framework scaler와 역할이 겹칠 수 있다. found-inf owner와 reduction group을 하나로 고정한다. rank 하나만 overflow를 보고 다른 rank가 update하면 replica가 즉시 갈라진다. 15장의 DP process group에서 boolean 또는 count를 합의하고, update 뒤 replicated parameter digest를 표본 검사한다. TP/EP shard는 같은 global tensor range끼리 비교해야 한다.

checkpoint는 committed generation만 가리킨다. 비동기 save가 parameter generation `g`와 scale history `g+1`을 섞지 않도록 snapshot barrier 또는 copy-on-write 계약을 둔다. compiled graph의 executable과 autotune cache는 rebuildable artifact이며 logical state generation에 포함시키지 않을 수 있다. 그러나 graph가 가진 device scalar와 RNG buffer가 training state라면 logical checkpoint로 승격한다. 무엇이 cache이고 무엇이 state인지 이름으로 정하지 않는다.

release gate는 GoldenBatch 하나로 끝나지 않는다. 정상 범위, range outlier, tail shape, overflow, resume, fallback과 cold compile을 순서대로 실행한다. 각 fixture에는 expected dispatch, dtype ledger, state delta와 performance ceiling을 명시한다. CUDA 12 baseline과 13 candidate의 결과를 같은 schema로 저장한다. candidate가 새 kernel 이름을 쓰더라도 logical operator와 연결할 수 있어야 한다.

**14장의 인수 과제**

첫 과제는 작은 MLP와 attention block에 dtype ledger를 붙이는 것이다. FP32 eager, BF16 autocast, FP16과 scaler, FP8 또는 지원되는 block-scaled 경로를 순서대로 실행한다. parameter storage, op input, accumulator 근거, output, gradient와 optimizer state를 기록한다. 2장의 backward 식과 11장의 AdamW update를 실제 tensor delta로 맞춘다.

둘째 과제는 Triton reduction 또는 elementwise fusion 하나를 고른다. fixed source revision, wrapper guard, launch meta-parameter와 generated target을 기록한다. tile 경계, noncontiguous, alignment, extreme range와 empty 의미를 test한다. eager high-precision oracle, target-dtype unfused와 fused를 ladder로 비교한다. unsupported input이 명시적 오류 또는 계약된 fallback을 내는지 확인한다.

셋째 과제는 CUDA migration rehearsal이다. 두 environment의 driver, wheel runtime, toolkit, compiler, ABI, library map과 device-code target을 채운다. 같은 binary 실행 시험과 clean rebuild 시험을 분리한다. import부터 forward/backward, compile, collective와 resume까지 smoke ladder를 통과시킨다. 측정하지 않은 장치와 shape는 지원 목록에서 뺀다.

넷째 과제는 상태 실패 주입이다. GradScaler growth tracker, FP8 amax history, block scale index와 RNG counter 가운데 하나씩 누락하거나 한 generation 뒤의 값을 넣는다. loader가 거절해야 하는 오류와, load는 가능하지만 probe가 차이를 찾아야 하는 오류를 구분한다. 첫 output뿐 아니라 두 번째 step의 metadata delta를 비교한다.

다섯째 과제는 성능과 정확성의 공동 승인이다. kernel launch, HBM byte, temporary peak, compile cold/warm, tokens/s를 기록하고 같은 RunID의 numerical report에 연결한다. 빠르지만 fallback state가 틀린 후보, 정확하지만 recompile storm이 있는 후보를 각각 실패시킨다. 승인된 RecipeID는 옵션, artifact와 state schema를 immutable하게 묶는다.

최종 답안은 “AMP를 켰다”, “FP8을 썼다”, “CUDA 13에서 돈다”가 아니다. 어떤 옵션이 어떤 storage·scale·compiler·kernel 상태를 만들었고, 그 결과 어느 수치·memory·stream·checkpoint 효과가 생겼는지를 보여 주어야 한다. 이 형식은 15장의 병렬 소유권 원장으로 이어진다. 저정밀 tensor가 collective를 건너면 dtype과 scale metadata의 owner, reduction denominator와 commit 합의까지 함께 설명해야 비로소 한 training step이 닫힌다.

### 14.8.2 한 번의 정확한 step을 장기 안정성으로 확장한다

kernel의 한 step oracle이 통과해도 수천 update 뒤 trajectory가 갈라질 수 있다. 작은 bias가 optimizer moment에 누적되고, FP8 scale history가 data distribution의 주기를 따라 진동하며, loss scaler가 skip한 step과 scheduler가 어긋날 수 있기 때문이다. 장기 probe는 단순히 최종 loss만 비교하지 않는다. 매 `K` update마다 parameter delta norm, optimizer first/second moment, scale history, saturation과 zero 비율, skipped update 누계와 RNG counter를 snapshot한다. snapshot overhead가 크면 representative layer와 fixed parameter IDs를 표본으로 고정한다.

옵션은 precision recipe, fusion, compile과 checkpoint interval이다. 상태는 parameter 외에도 optimizer와 scale의 시계열이다. 효과는 convergence, skip cadence, scale oscillation과 재현성이다. 같은 최종 metric이라도 optimizer state가 크게 다르면 이후 data regime에서 결과가 갈릴 수 있다. 반대로 bitwise divergence가 있어도 사전 정의한 numerical·trajectory budget을 지키면 같은 recipe class로 승인할 수 있다. 요구 수준을 실험 뒤 정하지 않는다.

장기 fixture의 data는 안정 범위, outlier, 다시 안정 범위의 세 구간으로 나눈다. outlier는 activation amax와 gradient magnitude를 독립적으로 자극한다. FP8 delayed scaling이 outlier를 얼마나 오래 기억하는지, GradScaler가 몇 step을 skip하고 언제 회복하는지 본다. 복귀 구간에서 saturation만 줄고 zero fraction이 오래 높게 남는다면 stale scale의 비용이다. FP16 scaler가 회복했지만 optimizer moment가 outlier 직전과 달라졌다면 partial commit 여부를 조사한다.

checkpoint는 세 경계에서 만든다. scaler growth interval 직전, FP8 history circular buffer가 wrap되기 직전, outlier를 관측한 직후다. 각 checkpoint를 재개한 run과 uninterrupted run을 다음 `2K` update 동안 비교한다. 첫 logits만 비교하면 counter와 history index 오류를 놓친다. save/load가 dtype을 유지해도 endianness, shape와 state schema가 다르면 loader가 거절해야 한다. 누락 field를 default로 채우는 migration은 old schema별 명시 함수와 fixture가 있어야 한다.

**optimizer와 fused update의 장기 반례**

AdamW를 예로 들면 gradient가 맞아도 step counter, bias correction, weight decay 순서와 moment dtype이 다르면 parameter delta가 달라진다. fused optimizer를 unfused reference와 비교할 때 첫 step, counter가 커진 step, zero gradient, overflow skip과 weight decay만 남는 입력을 넣는다. overflow step에서 decay까지 적용하는지 여부는 반드시 명시한다. 일반적으로 “optimizer step을 건너뛴다”면 decay와 moment, counter를 모두 건너뛰는 계약을 기대하지만 실제 사용 구현을 고정 source와 test로 확인한다.

moment를 BF16에 저장하고 update accumulator만 FP32인 recipe와 moment 자체가 FP32인 recipe는 memory와 장기 error가 다르다. 작은 일정 gradient를 수천 번 넣으면 moment quantization이 드러난다. alternating gradient는 cancellation과 bias correction을 자극한다. parameter magnitude가 큰데 update가 매우 작으면 output cast에서 update가 사라질 수 있다. FP32 master parameter가 이를 막는지, optimizer가 low-precision parameter에 직접 쓰는지 dtype ledger에서 확인한다.

fused multi-tensor optimizer는 tensor list packing과 alignment도 state다. empty gradient, frozen parameter, sparse 또는 noncontiguous gradient가 섞이면 list index가 어긋날 수 있다. test는 parameter마다 서로 다른 상수 gradient를 넣어 잘못된 moment 연결을 식별한다. checkpoint round trip 뒤 list 순서가 바뀌어도 ParameterID로 올바른 state가 붙어야 한다. Python iteration order에 기대지 않는다.

**stochastic rounding과 결정성의 범위**

stochastic rounding을 사용하면 low-precision update의 작은 값을 확률적으로 보존할 수 있지만 RNG state와 kernel schedule이 수치 계약에 들어온다. option은 enable flag와 RNG seed/counter owner다. 상태는 각 device 또는 tensor stream의 random coordinate다. 효과는 bias 감소 가능성, run variance와 checkpoint 크기다. 동일 seed라도 launch partition과 fusion이 바뀌면 소비 순서가 달라질 수 있다.

test는 fixed seed exact replay와 distributional test를 나눈다. 같은 binary와 shape의 replay가 목표라면 checkpoint 뒤 bitwise sequence를 확인한다. CUDA/compiler 또는 topology migration에서는 counter mapping이 달라질 수 있으므로 mean error와 variance가 이론적·경험적 bound 안인지 본다. stochastic path를 deterministic reference와 한 번 비교해 차이를 모두 bug로 분류하지 않는다. 반대로 “확률적”이라는 말로 systematic drift와 scale 오류를 숨기지 않는다.

### 14.8.3 operator별 저정밀 승인 카드를 완성한다

matmul 승인 카드에는 `M,N,K`, batch, transpose, stride, alignment, input/storage dtype, accumulator와 output dtype이 있다. TF32 허용, reduced-precision reduction과 fast-math option을 별도 열로 둔다. selected library/kernel algorithm, workspace와 target SM을 기록한다. test는 작은 exact integer, cancellation, large-K, tail tile과 noncontiguous를 포함한다. forward뿐 아니라 두 input gradient를 reference에 맞춘다.

normalization 카드는 reduction axis와 길이, epsilon, mean/variance accumulator, affine parameter와 output dtype을 기록한다. 거의 상수인 큰 값은 `E[x²]-E[x]²` 형태의 불안정성을 드러낸다. zero variance, 매우 작은 epsilon, odd hidden size와 residual fusion을 시험한다. RMSNorm과 LayerNorm을 이름만 보고 같은 kernel로 간주하지 않는다. mean subtraction과 backward 식이 다르다.

softmax와 attention 카드에는 row domain, mask 의미, causal offset, scale, dropout, head mapping과 variable-length metadata를 기록한다. all-masked row의 output과 gradient 정책을 정한다. online softmax의 running max와 denominator accumulator dtype을 확인한다. 매우 큰 양·음 logits, 한 원소 row, 긴 row와 mask tail을 시험한다. attention backward는 dQ, dK와 dV 각각의 tolerance와 RNG replay를 검사한다.

cross entropy 카드에는 logits global/sharded shape, target mapping, ignore index, label smoothing, reduction과 denominator를 기록한다. vocabulary parallel이면 global max와 exp sum, target logit owner의 collective를 손으로 계산한다. target이 shard 경계와 마지막 tail에 있는 경우를 넣는다. all-ignored batch는 loss 값과 update skip 정책을 정한다. NaN을 0으로 바꾸어 진행하지 않는다.

embedding 카드는 index dtype, padding index, tied weight, sparse 여부와 vocab shard mapping을 기록한다. duplicated index의 gradient accumulation order가 dtype error를 키울 수 있다. out-of-range index는 명확히 거절한다. TP vocab shard에서 local index 변환과 mask가 맞는지 15장의 소유권 함수에 연결한다. tied output weight가 optimizer state를 하나만 갖는지도 확인한다.

optimizer 카드에는 parameter/master, gradient, moment, norm reduction과 scalar hyperparameter dtype을 기록한다. foreach, fused, capturable와 differentiable option이 kernel과 state를 어떻게 바꾸는지 적는다. step counter가 host integer인지 device tensor인지, graph replay에서 갱신되는지 본다. overflow, zero gradient, frozen parameter, weight decay와 checkpoint를 시험한다.

communication 카드에는 payload dtype, scale metadata, reduction accumulator, process group과 denominator를 기록한다. BF16 all-reduce와 FP32 all-reduce의 byte와 error를 비교한다. FP8 또는 block-scaled communication은 rank별 scale을 그대로 더할 수 없으므로 dequantize/requantize 또는 shared scale 계약이 필요하다. scale collective가 payload 절감 이득을 상쇄하는 작은 bucket도 측정한다.

**카드 사이의 경계 test**

개별 operator가 통과해도 경계에서 오류가 난다. normalization FP32 output이 fused residual에서 BF16으로 cast되는 위치, attention output과 projection matmul 사이 layout, cross entropy gradient와 loss scaler multiplier, reduce-scatter output과 fused optimizer input을 pairwise test한다. producer output dtype·stride·lifetime과 consumer 요구가 같은지 schema로 비교한다.

compiler fusion은 두 카드를 하나의 kernel로 합치므로 중간 tensor를 관찰하기 어렵다. debug build에서 fusion을 끈 reference, generated code의 cast/store와 end-to-end invariant를 조합한다. fusion 때문에 intermediate rounding이 사라지는 것은 합법적 algorithm 변화일 수 있지만 error budget에 별 항목으로 둔다. intermediate mask 또는 scale update가 사라지는 것은 의미 오류다.

stream 경계에서는 producer completion과 consumer wait를 test한다. autocast나 quantization kernel이 side stream에서 scale을 갱신하고 matmul이 default stream에서 stale scale을 읽지 않는지 본다. graph capture가 event를 고정해 replay마다 올바른 buffer generation을 가리키는지 확인한다. allocator가 같은 주소를 재사용한다고 같은 logical tensor generation은 아니다.

**최종 판정표**

정확성 PASS는 supported domain의 oracle tolerance, invariant와 negative test를 모두 만족한다는 뜻이다. dispatch PASS는 의도한 production shape가 검증한 kernel을 타고 fallback이 계약대로 기록된다는 뜻이다. state PASS는 scaler, FP8/NVFP4 metadata, optimizer, RNG와 graph scalar가 step과 resume에서 맞는다는 뜻이다. compatibility PASS는 driver/runtime/toolkit/ABI/device target의 검증 조합 안에 있다는 뜻이다. performance PASS는 cold와 warm, memory와 throughput budget을 만족한다는 뜻이다.

한 열의 실패를 다른 열의 성공으로 덮지 않는다. 정확하지만 미검증 fallback은 dispatch 실패다. 빠르지만 scale history가 복구되지 않으면 state 실패다. CUDA 13에서 source compile이 되어도 target GPU cubin/PTX와 runtime library가 검증되지 않으면 compatibility 실패다. 모든 열이 PASS인 정확한 artifact와 RecipeID만 배포한다.

incident 뒤에는 같은 판정표에서 최초로 깨진 열과 fixture를 찾는다. environment 변경이면 compatibility ladder, 특정 shape면 dispatch guard, finite drift면 numerical ladder, resume 전용이면 state round trip, 느려짐이면 profiler와 compile cache를 연다. 모든 최적화를 끄는 것은 임시 containment일 수 있지만 root cause 증거가 아니다. 최소 반례를 회귀 test에 추가한 뒤 전체 카드와 장기 probe를 다시 실행한다.

이 마지막 카드는 3장의 작은 GPT golden run을 device kernel 수준으로 확장하고, 28장의 single-GPU golden run을 CUDA 세대 전환의 기준점으로 만든다. 이어 15장의 분산 실행에서는 같은 카드에 placement와 collective 열을 추가한다. 한 GPU에서 정의하지 못한 dtype·state·commit 의미는 여러 GPU에서 통신을 붙인다고 명확해지지 않는다.

### 14.8.4 실전 변경 요청을 검증 가능한 diff로 바꾼다

실전에서는 “BF16에서 FP8로 바꿔 달라” 또는 “CUDA 13 image로 올려 달라”처럼 효과만 적힌 변경 요청이 들어온다. 구현자는 이를 option diff, state-schema diff, dispatch diff와 evidence diff로 풀어야 한다. option diff에는 autocast dtype, recipe, compiler flag와 architecture target이 있다. state-schema diff에는 scale, amax history, optimizer master와 checkpoint version이 있다. dispatch diff에는 새 kernel guard, fallback과 graph partition이 있다. evidence diff에는 추가해야 할 oracle, failure injection과 benchmark가 있다.

첫 질문은 변경하지 않는 층이다. tokenizer와 GoldenBatchID, model graph, optimizer hyperparameter, global denominator와 scheduler horizon을 고정한다. CUDA image 이동과 precision 변경을 동시에 요청받았으면 가능하면 environment-only와 precision-only 중간 후보를 만든다. 중간 후보가 불가능하면 결과 해석이 제한된다는 사실을 보고서에 쓴다. 원인을 안다고 꾸미지 않는다.

둘째 질문은 schema 호환성이다. 기존 checkpoint에 새 FP8 또는 block-scale field가 없을 때 eager BF16 warm start가 가능한지, optimized path가 어느 시점에 scale을 초기화하는지 정한다. 반대로 새 checkpoint를 old binary가 읽을 수 있는지 확인한다. forward-compatible reader가 unknown field를 무시해도 optimizer와 scale 의미가 보존된다는 뜻은 아니다. bidirectional load matrix를 만든다.

셋째 질문은 첫 실행 비용이다. 새 Triton/CUDA kernel compile, PTX JIT, autotune과 graph capture가 startup에 집중될 수 있다. production worker가 elastic restart할 때마다 이 비용을 내면 steady benchmark와 운영 throughput이 다르다. cold cache, process-local warm cache, node cache와 immutable prebuilt artifact를 분리한다. cache가 source, capability와 compiler hash를 검증하는지도 test한다.

넷째 질문은 fallback budget이다. production shape histogram에서 각 guard가 차지하는 비율을 계산한다. 평균 길이 하나로 fast path hit를 주장하지 않는다. tail shape, rare mask와 dtype 조합이 전체 step의 graph break를 만들 수 있다. fallback은 expected 목록과 최대 비율을 release contract에 넣는다. 새 data mixture로 비율이 넘으면 성능 incident이자 coverage 변경이다.

다섯째 질문은 rollback state다. candidate가 한 번이라도 optimizer와 FP8 history를 새 schema로 commit했다면 baseline binary가 읽을 수 있는 마지막 generation을 보존한다. rollback은 container tag 교체가 아니라 model, optimizer, scaler, scale, RNG와 data cursor의 compatible generation 선택이다. candidate가 일부 update를 수행한 뒤 baseline-format model-only save만 만들면 optimizer trajectory는 이어지지 않는다. warm restart로 명명한다.

**변경 전후 표본 tensor 추적**

대표 tensor는 embedding output, attention score 또는 projection, normalization statistic, logits와 한 optimizer moment에서 고른다. baseline과 candidate에서 source operator, storage/input/accumulator/output dtype, selected kernel, stream, scale metadata와 checkpoint key를 잇는다. tensor 하나가 forward에서 optimizer와 save까지 관통해야 서로 다른 보고서의 옵션 이름만 우연히 같은 상황을 피한다.

attention projection 표본에서는 BF16 input이 FP8 block으로 cast되는 위치, scale generation과 matmul accumulator, output cast와 residual add를 기록한다. backward에서는 output gradient format, dWeight accumulator와 gradient collective dtype을 기록한다. optimizer가 FP32 master에 update하고 BF16/FP8 cache를 언제 다시 만드는지도 본다. cache refresh가 skip step에도 실행되어 parameter generation과 어긋나지 않는지 확인한다.

logits 표본에서는 vocabulary tail, cross-entropy max/sum accumulator와 target owner를 기록한다. fused path가 full logits를 저장하지 않아도 loss sum과 denominator, backward row invariant를 비교할 수 있다. compiled fusion에서 intermediate가 사라졌으면 generated code의 reduction과 store를 확인한다. 디버그를 위해 logits 전체를 materialize한 경로는 성능 baseline이 아니라 numerical oracle이다.

optimizer moment 표본에서는 ParameterID, shard/global offset, state dtype, step counter와 checkpoint generation을 기록한다. fused list index와 save key가 같은 logical parameter를 가리키는지 확인한다. CUDA environment가 바뀌어 fused optimizer extension을 rebuild해도 logical moment는 portable tensor로 load되어야 한다. opaque state라면 지원 범위와 reset 비용을 명시한다.

**운영 중 admission과 자동 격리**

runtime admission은 device capability, driver/runtime 조합, artifact target, supported shape·stride·dtype와 recipe ID를 검사한다. 조건 밖 입력을 fast kernel에 넣지 않는다. 계약된 generic fallback이 있으면 telemetry와 함께 허용하고, 없으면 요청 또는 job을 명확히 거절한다. 검증하지 않은 환경 변수를 켜서 guard를 우회하지 않는다.

scale telemetry가 saturation, zero 또는 rank divergence threshold를 넘으면 즉시 precision을 전환할지 job을 멈출지 정책을 정한다. 자동 BF16 fallback이 parameter update 중간에 일어나면 같은 step에서 형식이 섞일 수 있다. 안전 경계는 보통 다음 uncommitted step 시작 또는 checkpoint recovery다. 모든 rank가 같은 recipe generation으로 전환하도록 합의한다.

kernel fault는 해당 process만 generic path로 계속 돌리지 않는다. illegal write 뒤 context와 tensor state를 신뢰할 수 없기 때문이다. 마지막 committed checkpoint로 새 process를 시작하고 offending shape와 artifact를 격리한다. numerical threshold 초과처럼 context가 건강한 사건은 fixed probe를 실행해 scale/fallback을 분류할 수 있지만 optimizer commit 전이어야 한다.

admission 결과와 자동 조치는 26장의 관측성 schema에 `RecipeID`, `ArtifactID`, `ScaleGeneration`, `DispatchReason`, `CommitGeneration`으로 남긴다. cardinality가 큰 tensor 원본은 incident 표본으로 제한한다. 운영 dashboard의 평균 FP8 사용률만으로 정확성을 판단하지 않고 saturation·fallback·skip과 resume 사건을 같은 시간축에서 본다.

**runbook: 증상 하나에 변경 하나만 대응한다**

import 오류면 loader map과 ABI를 먼저 고친다. kernel launch 전 오류면 target SM/PTX와 driver JIT를 본다. 특정 shape의 illegal access면 wrapper guard, stride, alignment와 tile mask를 본다. finite numerical drift면 eager target dtype, fused, compiled ladder로 최초 변화를 찾는다. NaN이면 최초 non-finite와 최초 tolerance 초과를 모두 찾는다. resume 전용 drift면 scaler, scale history, RNG와 optimizer generation을 비교한다. 느려짐이면 fallback, recompile, launch gap, HBM과 stream overlap을 차례로 본다.

각 단계에서 환경 변수 여러 개를 동시에 바꾸지 않는다. 동기 launch 옵션은 실패 위치를 좁히는 진단 조건이지 최종 성능 설정이 아니다. precision을 FP32로 올려 증상이 사라져도 root cause가 low-precision range인지 race인지 구분한다. fusion을 끄면 사라지는 오류도 intermediate state, alias와 RNG를 더 검사한다.

수정 뒤에는 최소 재현만 통과시키고 끝내지 않는다. operator 카드, 경계 test, long-horizon probe, checkpoint round trip, CUDA compatibility fixture와 performance budget을 실행한다. source patch가 guard를 추가해 generic fallback이 늘었다면 correctness 수정과 performance regression을 둘 다 보고한다. release note는 지원이 늘어난 domain과 여전히 거절하는 domain을 정확히 적는다.

이 runbook의 최종 산출물은 원인 문장, 반증 fixture, source diff, artifact hash와 재검증 결과다. “CUDA 업데이트 후 해결”, “AMP를 꺼서 해결” 같은 조치는 원인 문장이 아니다. 어떤 state edge가 틀렸고 어떤 invariant가 이제 이를 막는지 써야 다음 CUDA 12.x/13.x patch, framework와 kernel revision에서도 지식을 재사용할 수 있다.

운영자는 마지막으로 fixed probe의 원시 입력 hash와 expected tensor summary를 artifact 옆에 둔다. library patch가 바뀔 때마다 같은 probe를 실행하여 build, load, dispatch, 수치, state와 성능 gate의 순서로 비교한다. 이전 결과 파일을 덮어쓰지 않고 candidate generation을 새로 만든다. 지원 범위를 넓히는 결정은 새 shape와 장치의 negative fixture까지 통과한 뒤에만 한다. 이렇게 하면 작은 patch update도 근거가 남는 변경이 된다.

장기 학습에 투입한 뒤에는 사전 검증과 production telemetry의 분포를 주기적으로 대조한다. probe에 없던 saturation, fallback 또는 compile guard가 나타나면 입력 분포를 익명화한 최소 fixture로 환원한다. 임시 허용은 만료 시점과 rollback 조건을 가져야 한다. 발견된 경계를 operator 카드에 추가하고 다음 artifact의 admission 규칙으로 승격한다.

## 14.9 autocast·scaler·Transformer Engine의 state를 분리한다

autocast, loss scaler, FP8 recipe는 모두 ‘저정밀 설정’처럼 보이지만 서로 다른 state와 clock을 가진다. master weight·scale history·amax·distributed group을 분리해 어떤 옵션이 GEMM·attention·norm·optimizer의 dtype과 fallback을 바꾸는지 명시한다.

### 14.9.1 표현 형식·누적·반올림을 한 dtype contract로 쓴다

FP32, TF32, FP16, BF16과 FP8을 storage 이름만으로 비교하지 않는다. 각 tensor/op에 storage, input cast, multiply, accumulator, output cast, reduction와 collective dtype을 적는다. TF32는 FP32 tensor storage를 그대로 받으면서 지원 matmul 내부 multiply precision을 바꾸는 실행 mode다. BF16과 FP16은 exponent/fraction tradeoff가 달라 같은 scale에서 overflow/rounding이 다르다.

FP8은 E4M3/E5M2 payload에 scale과 amax history가 결합된다. actual representable range와 special-value policy는 사용하는 device/library recipe의 공식 문서와 함수에서 확인한다. format 이름만 보고 모든 kernel이 같은 NaN/Inf·subnormal behavior를 가진다고 쓰지 않는다.

**Rounding fixture**

halfway values, large+small cancellation, subnormal/zero 경계, representable maximum 전후와 long reduction을 넣는다. FP64/FP32 reference, target eager와 actual fused kernel을 비교한다. relative error만 쓰지 않고 absolute/RMS/cosine, zero/saturation와 invariant를 함께 본다.

### 14.9.2 master weights·loss scaling·autocast의 세 state를 분리한다

master weight는 low-precision model view와 별 FP32 optimizer-authoritative parameter일 수 있다. optimizer update가 어느 copy에 쓰이고 model view를 언제 refresh하는지 source/state_dict에서 확인한다. low-precision parameter에 직접 update하는 recipe와 memory·small-update 보존이 다르다.

GradScaler state는 current scale, growth tracker/interval/factors와 found-inf다. overflow면 parameter, master, moments, scheduler와 update clock이 함께 skip되어야 한다. multiple optimizers/ranks의 found-inf 합의와 partial update를 test한다. scaler checkpoint 누락은 resume skip cadence를 바꾼다.

autocast는 op별 input/output policy이며 parameter storage나 master state를 자동 정의하지 않는다. PyTorch autocast context, backend TF32/matmul precision flags와 custom op registration을 fixed source/function으로 기록한다. norm/reduction/loss가 실제 어느 dtype을 탔는지 hook/profiler로 확인한다.

**Order failure**

scaled gradient clipping, denominator 적용 전 clipping, optimizer step 뒤 scaler update 누락과 overflow에서 scheduler advance를 독립 주입한다. unscale→global denominator/collective→clip→commit의 recipe-defined order를 selected gradients와 moments로 검증한다.

### 14.9.3 GEMM·attention·norm·optimizer의 dtype/fallback 표

GEMM card에는 A/B storage, input format/scales, accumulator, epilogue/output, TF32/fast-math, shape/alignment와 selected library/kernel을 기록한다. attention은 Q/K/V, score/LSE/probability/PV accumulator와 backward dQ/dK/dV를 추가한다. norm은 statistic accumulator와 affine/output, optimizer는 gradient/moment/master/scalars를 기록한다.

fallback은 dtype contract와 state transition이다. unsupported FP8 shape가 BF16 GEMM으로 내려가면 amax/history가 갱신되는지, output dtype과 다음 scale generation을 확인한다. fused attention→math fallback은 saved state/RNG/memory를 바꾼다. norm/optimizer fallback도 state keys와 graph capture 가능성을 바꿀 수 있다.

**Dispatch boundary**

head/hidden/tile `n-1,n,n+1`, odd/tail, noncontiguous, misaligned, all-masked와 zero-sized logical shard를 넣는다. expected kernel/fallback/error와 numerical oracle, workspace ceiling을 test한다. silent contiguous copy나 FP32 promotion도 performance/effect에 기록한다.

### 14.9.4 Transformer Engine recipe를 module·metadata·distributed group에서 읽는다

Transformer Engine 사용 시 FP8 context/recipe, converted modules, cast/amax update, delayed/current scaling와 supported kernels을 fixed revision에서 고정한다. framework autocast가 FP8 recipe를 의미하지 않는다. actual module extra state keys, forward output와 kernel trace를 확인한다.

state card에는 format by tensor role, scale/inverse scale, amax history/index, margin/algorithm, granularity, update interval와 reduction group을 기록한다. parameter payload/checkpoint와 runtime derived caches를 분리한다. recipe/config checksum이 model checkpoint requirement다.

**Scale history failure**

outlier 직전/후, circular wrap, reset resume, rank 하나 stale history와 supported→fallback→supported shapes를 넣는다. first/second output과 metadata delta를 uninterrupted reference와 비교한다. payload checksum만 맞고 history/index가 다르면 실패다.

DP/TP amax group은 tensor ownership과 맞아야 한다. wrong same-size group이 silent scale divergence를 만들 수 있다. global TensorID/head/block pattern으로 검출한다. communication byte와 synchronization cost도 FP8 speed report에 넣는다.

**CUDA 12.x·13.x 차이를 검증 행렬로 제한한다**

CUDA 공식 compatibility/release/programming 문서가 각각 driver/runtime, component 변화와 execution semantics 중 무엇을 보장하는지 분리한다. framework/Transformer Engine/Triton/CUTLASS support matrix는 별 evidence다. CUDA major 하나로 third-party extension ABI와 kernel coverage를 보장하지 않는다.

environment 행은 exact driver, wheel-bundled runtime/libraries, system toolkit/nvcc, host compiler/C++ ABI, framework/extensions, device capability와 cubin/PTX targets다. `nvidia-smi` 상한, `torch.version.cuda`와 `nvcc --version`을 같은 값으로 기대하지 않는다.

**Paired toolchain experiment**

same binary를 supported drivers에서 실행해 runtime relation을, same source/dependency를 CUDA 12.x/13.x toolkit으로 clean build해 compiler/codegen relation을 본다. import→representative fwd/bwd→compile→collective→checkpoint와 actual dispatch를 비교한다. 한 번에 framework와 source까지 바꾼 결과는 통합 evidence이지 root-cause evidence가 아니다.

지원하지 않은 GPU/shape/library 조합은 `NOT_RUN`이다. PTX JIT 성공은 architecture-tuned performance support가 아니다. actual SM instruction/kernel, cold JIT/cache와 benchmark를 별 gate로 둔다.

**stochastic rounding과 amax를 RNG·history clock으로 검증한다**

stochastic rounding은 작은 update/quantization bias를 줄일 수 있지만 RNG seed/counter, tensor coordinate, launch partition와 checkpoint를 state로 추가한다. exact replay requirement와 topology/toolchain migration의 distributional requirement를 구분한다. “확률적”이라는 말로 systematic drift를 숨기지 않는다.

amax delayed scaling에는 history window와 update clock이 따른다. overflow-skipped update에서 history를 갱신할지 recipe를 명시한다. data microbatch clock, forward call와 committed update를 섞지 않는다. rank/module별 scale telemetry에 generation을 둔다.

**Stochastic property**

same binary/shape fixed seed replay, many-trial mean/variance, positive/negative halfway symmetry와 zero/saturation을 test한다. graph/fusion/topology가 RNG consumption을 바꾸면 distributional bounds와 logical coordinate mapping을 확인한다. checkpoint boundary에서 next sequence를 비교한다.

## 14.10 collective·checkpoint·optimizer의 저정밀 일관성

rank마다 finite 여부나 scale generation이 다르면 local kernel이 정확해도 replica는 갈라진다. collective의 accumulator dtype, portable checkpoint state와 binary 의존성, optimizer의 master·moment·cast 순서를 global numerical oracle로 검증한다.

### 14.10.1 distributed collective dtype와 global numerical oracle

gradient/parameter/activation collective마다 payload, reduction accumulator, pre/post scale와 process group을 기록한다. BF16 reduce와 FP32 reduce는 byte/error가 다르다. FP8/block-scaled payload는 shared/communicated scale와 metadata가 필요하며 rank별 arbitrary quantized values를 그대로 합하지 않는다.

DP valid-token denominator와 DDP average factor를 precision scale과 분리한다. TP partial output, CP attention와 EP token collectives도 logical global tensor reference를 만든다. reduction order 차이 tolerance와 systematic wrong scaling을 구분한다.

**Collective failure**

rank별 magnitude, wrong dtype/group, stale scale, uneven shards와 one-rank overflow를 넣는다. single-rank concatenated FP32 oracle와 global gradient/parameter delta를 비교한다. collective byte/stream completion과 optimizer read edge를 확인한다.

### 14.10.2 checkpoint·export compatibility를 portable state와 binary로 나눈다

portable logical state에는 parameter/master, optimizer moments, scaler, FP8 scale/history, stochastic RNG, scheduler와 config가 포함된다. fused opaque state, compiled graph, autotune/JIT cache와 cubin은 portability가 다르다. cache라고 이름 붙은 training-critical scalar는 logical state로 승격한다.

CUDA 12→13 load는 eager BF16 parameter/logits, optimizer logical state, scaler/FP8, fused/compiled path 순으로 켠다. hardware-specific binaries/caches는 rebuild하고 artifact manifest를 검증한다. old/new reader matrix와 rollback parent를 보존한다.

**Mixed-generation failure**

new payload/old scale, new master/old low-precision view, scaler one-step drift, old graph constants와 wrong cubin target을 넣는다. loader/admission 또는 next-two-update fixture가 optimizer commit 전에 실패해야 한다.

exported inference model도 required tokenizer/config, precision recipe/quantization scales와 target architecture를 선언한다. training FP8 state를 inference quantization으로 자동 재사용하지 않는다. serving cached path를 full reference와 비교한다.

### 14.10.3 numerical parity에서 failure ladder까지

FP32 eager→TF32/target AMP→fused→FP8/low format→compiled/graph→distributed 순서로 rung 하나씩 추가한다. forward layer summaries/LSE, loss numerator/denominator, dInputs/dParams, optimizer state/delta와 checkpoint resume를 비교한다. performance도 kernel count/HBM/compile/throughput을 같은 ladder에 둔다.

failure matrix는 representation boundary, accumulator, autocast policy, scale/history/RNG, dispatch fallback, ABI/SM, collective dtype/group와 mixed checkpoint다. 각 failure에는 one injection과 expected first gate를 지정한다. 여러 환경 변수를 동시에 바꾸지 않는다.

최종 dossier에는 official support evidence, exact environment/build/binary, dtype ledger, operator/dispatch cards, numerical report, state/checkpoint, distributed oracle와 performance/rollback을 담는다. 같은 RecipeID/RunID/UpdateID를 가리킨다.

독립 reviewer는 tensor 하나를 storage→cast/scale→kernel accumulator/output→collective→optimizer/master→checkpoint까지 추적한다. CUDA 12.x/13.x candidate에서 support 범위와 actual binary/kernel을 확인하고 one failure를 재생한다. 모든 state와 tolerance가 맞을 때만 low-precision recipe를 봉인한다.

**optimizer kernel의 master·moment·cast 순서를 손계산한다**

mixed-precision AdamW 표본에서 model parameter, FP32 master, gradient, first/second moment, step와 output model view를 분리한다. fused/foreach/single 경로가 same logical recurrence와 overflow policy를 만족해야 한다. optimizer가 master를 쓰지 않는 recipe는 별 oracle을 가진다.

2×2 parameter와 small update를 사용해 BF16 model view에서 사라지는 delta가 FP32 master에 누적되는지 본다. several steps 뒤 cast가 바뀌는 boundary를 계산한다. direct low-precision update와 차이를 장기 probe에서 확인한다.

**Optimizer failure**

master refresh 누락, moment BF16 unexpected, step/decay order, stochastic cast counter와 one parameter list permutation을 넣는다. first output이 같아도 second update/moment에서 divergence를 잡는다. checkpoint ParameterID와 fused list index를 맞춘다.

## 14.11 reduction·공식 근거·production canary

norm과 softmax는 큰 reduction 범위 때문에 GEMM과 다른 실패 양상을 보인다. dtype별 범위 실패를 재현하고 공식 근거와 runtime trace를 대조한 뒤, export·serving 계약까지 포함한 canary와 rollback으로 승격 여부를 정한다.

### 14.11.1 norm·softmax reduction의 범위 실패를 dtype별로 분리한다

norm과 softmax는 reduction 범위 때문에 matmul input precision만 올려도 안정성이 해결되지 않는다. mean/variance/RMS, row max, exp sum와 backward dot의 accumulator dtype을 source/generated kernel에서 확인한다. epsilon, mask와 output cast를 기록한다.

constant-large norm input, nearly tied logits, long row, all-masked, extreme signs와 odd tail을 FP32 oracle과 비교한다. FP16 overflow, BF16 precision loss와 FP8 saturation을 다른 symptom으로 분류한다. output finite만으로 통과시키지 않는다.

**Reduction fallback**

fused norm/attention이 unsupported shape에서 framework op로 fallback될 때 accumulator와 saved state가 달라지는지 본다. fallback 전후 graph/stream, temporary memory와 tolerance를 기록한다. production histogram의 hit rate를 support report에 둔다.

### 14.11.2 official evidence와 runtime evidence를 양방향 검증한다

공식 CUDA compatibility/release 문서는 가능한 driver/toolkit 관계와 component 변화의 근거다. framework/TE documentation와 source guards는 package-supported combinations의 근거다. runtime trace/binary inspection은 실제 process가 어느 library, SM/PTX와 kernel을 실행했는지의 근거다. 세 evidence를 서로 대체하지 않는다.

manifest claim 하나를 골라 문서 section, framework build/support, artifact targets와 runtime probe를 연결한다. version 상한/최소, exact tested patch와 device를 적는다. 검증하지 않은 CUDA 13.x minor나 future architecture를 포함하지 않는다.

**Evidence drift**

driver/library/container 또는 documentation/support matrix가 바뀌면 affected cells를 stale로 돌린다. source compile success만으로 load/dispatch/numerical/performance를 PASS하지 않는다. same fixed fixture를 candidate generation에서 다시 실행한다.

### 14.11.3 production canary와 rollback 판정

canary는 same parent checkpoint와 GoldenBatch에서 BF32/target baseline, candidate precision과 actual optimized path를 비교한다. normal, outlier, overflow, fallback, distributed collective와 checkpoint/resume를 짧은 sequence로 실행한다. cold/warm compile/JIT를 나눈다.

monitor는 first non-finite, saturation/zero, scale/history divergence, skip cadence, unexpected fallback, gradient/update error, kernel latency와 collective dtype를 RecipeID별로 본다. baseline distribution을 벗어나면 next commit 전 containment/rollback을 결정한다.

**Rollback rehearsal**

candidate artifact와 state를 한 step 진행한 뒤 preserved compatible parent로 돌아간다. old model/master/optimizer/scaler/FP8 state, data/RNG와 environment artifact를 복원한다. candidate-only schema를 old binary에 억지로 load하지 않는다. warm start와 exact rollback을 구분한다.

독립 인수자는 canary event 하나에서 official support, binary/library map, dtype/state ledger, selected kernel, numerical delta와 checkpoint root를 재생한다. option 하나가 달라지면 new state/effect와 invalidated evidence를 예측한다.

이 canary가 닫히면 FP32/TF32/FP16/BF16/FP8은 단순 성능 label이 아니다. 표현, accumulator, rounding, scaling/RNG, kernel/toolchain, collective와 durable state가 한 계약이 된다. 실제 검증한 CUDA 12.x/13.x 범위 안에서만 그 계약을 지원으로 선언한다.

### 14.11.4 export와 serving dtype contract를 training state에서 파생한다

training checkpoint의 BF16/FP8 tensors와 amax history가 serving quantization/export format을 자동 정의하지 않는다. exporter에는 source model/master view, target dtype/quantization, per-tensor/block scales, calibration data, fused operator와 target device/runtime를 명시한다. derived artifact는 parent CheckpointID와 ExportRecipeID를 선언한다.

training FP8 scale은 optimizer/activation distribution과 delayed history를 반영할 수 있고 inference calibration scale과 목적이 다르다. 재사용하려면 explicit compatibility와 fixed-probe를 요구한다. master weight 또는 canonical model weight 가운데 어느 copy를 export하는지 source에서 확인한다.

**Export failure**

stale low-precision view/new master, missing scale block, wrong layout/tied head, unsupported SM와 runtime library mismatch를 넣는다. loader/admission과 fixed-token logits/cached decode가 expected gate에서 실패해야 한다. shape/load 성공만으로 승인하지 않는다.

portable export에는 logical weights/scales/schema와 required tokenizer/config, binary-specific engine에는 compiler/toolkit/SM/library manifest를 둔다. CUDA 12에서 만든 engine을 CUDA 13 environment에 그대로 쓸지 backend support 범위와 actual load/dispatch로 판정한다. 필요하면 rebuild한다.

**numerical certificate를 구성하는 표본 세 개**

첫 표본은 GEMM weight row다. checkpoint storage/master에서 cast/scale, selected Tensor Core path, accumulator/epilogue, backward gradient와 optimizer delta까지 추적한다. 둘째는 attention/norm reduction row로 mask, LSE/statistic, output와 backward invariant를 본다. 셋째는 distributed gradient shard로 collective dtype/scale, denominator와 commit을 본다.

각 표본은 FP32 oracle, target eager, fused/compiled와 CUDA candidate paths를 같은 input에서 비교한다. actual kernel/library/SM, tolerance, saturation/zero와 state generation을 보존한다. unsupported path는 `NOT_RUN`이다.

**Certificate failure**

표본마다 rounding boundary, scale/history/RNG, fallback/ABI와 wrong collective를 하나씩 독립 주입한다. first expected detector가 실패해야 한다. tolerance나 support scope를 결과 뒤 바꾸면 새 certificate generation을 만든다.

certificate는 official evidence, build/runtime artifact, dtype/operator cards, numerical/state/checkpoint, performance와 rollback을 checksum으로 묶는다. independent reviewer가 source environment 없이도 artifact resolver와 fixed commands로 같은 pass/fail을 얻어야 한다.

이 세 표본은 모든 tensor를 저장하지 않고도 training step의 서로 다른 precision 위험을 관통한다. 새 dtype, GPU, toolkit, kernel 또는 topology가 들어오면 relevant 표본과 long-horizon probe를 다시 실행한다. 동일 recipe 이름 아래 evidence를 덮어쓰지 않는다.

**support matrix의 유지보수 계약**

행은 GPU architecture와 CUDA/toolkit/framework/TE/kernel artifact 조합, 열은 GEMM, attention forward/backward, norm, optimizer, graph, collective와 export다. 각 cell에는 supported shape/dtype/recipe, actual dispatch, numerical/state/checkpoint와 performance EvidenceID를 기록한다. 빈 cell은 `NOT_RUN`이다.

driver patch, framework wheel, extension rebuild, compiler flag, recipe 또는 scale granularity가 바뀌면 affected cells를 stale로 돌린다. source compatibility와 binary/runtime compatibility, correctness와 tuned performance를 separate status로 유지한다.

**유지보수 rehearsal**

old/new environment에서 same fixed artifacts와 GoldenBatch를 resolver로 실행한다. library map, SM/PTX/cubin, selected kernels, dtype ledger와 next update/checkpoint를 비교한다. stale compile/JIT/autotune cache를 주입해 key validation이 막는지 본다.

production telemetry의 shape·dtype·fallback·scale 분포가 verified matrix를 벗어나면 new support request를 만든다. generic fallback이 성공해도 latency/memory contract와 state history를 다시 검증한다. silent support 확대를 금지한다.

독립 reviewer는 matrix PASS 하나를 골라 official document scope, framework/source guard, binary target, runtime trace와 numerical certificate를 양방향 확인한다. 한 evidence가 없으면 해당 층의 claim을 낮춘다.

이 유지 규칙은 CUDA 12.x/13.x와 low-precision formats가 계속 바뀌어도 검증 범위를 정확히 보존한다. 빠른 kernel 하나가 아니라 표현·state·toolchain·분산·복구 전체가 같은 generation으로 닫힐 때만 지원을 계속 선언한다.

최종 승인 결과는 immutable RecipeID와 parent certificate로 보존한다. 다음 release가 default dtype, accumulator, rounding, scale update, fallback 또는 binary target을 하나라도 바꾸면 기존 PASS를 복사하지 않는다. 관련 operator와 long-horizon·resume·rollback fixture를 다시 실행해 독립 검토자가 같은 제한 결론을 재생하게 한다.

모든 지원 판정은 실제 실행한 장치와 정확한 software artifact 범위 안에서만 끝까지 유효하다.

## 14.12 표현 형식·stochastic rounding·GEMM dispatch

여기서는 dtype의 비트 배치에서 다시 시작해 subnormal과 stochastic rounding의 확률적 상태를 닫는다. 그 위에서 cuBLASLt heuristic과 autotune cache가 어느 Tensor Core 경로를 골랐는지 재현 가능한 dispatch 증거로 만든다.

### 14.12.1 표현 형식을 부호·지수·가수와 subnormal 정책으로 해부한다

**같은 bit 수에서도 범위와 간격이 달라진다.**

부동소수점은 대략 `(-1)^s·2^e·(1.f)`로 읽는다. 지수 bit가 많으면 동적 범위가 넓고 fraction bit가 많으면 한 지수 구간의 상대 간격이 촘촘하다. FP32, TF32, BF16, FP16, FP8 E4M3/E5M2와 FP4 계열은 지수·가수 배분과 special value 정책이 다르다.

FP16은 BF16보다 fraction 정밀도가 높지만 exponent 범위가 좁다. BF16은 FP32와 비슷한 exponent 폭을 유지해 overflow 위험을 줄이지만 unit roundoff가 더 크다. TF32는 storage dtype이 아니라 FP32 input을 Tensor Core matmul에서 축약된 mantissa로 처리하는 compute mode다.

FP8 E4M3와 E5M2는 같은 8bit라도 정밀도/범위 trade-off가 다르며 NVIDIA 구현의 exact finite/NaN encoding과 saturation 정책은 공식 format/Transformer Engine 문서를 따라야 한다. 이름만으로 IEEE binary format을 가정하지 않는다. FP4/NVFP4도 tensor 하나의 dtype이 아니라 block scale과 higher-level scale이 결합된 recipe일 수 있다.

**ULP와 상대 오차를 tensor role에 연결한다.**

값 x의 인접 representable 간격은 exponent에 따라 달라진다. 작은 parameter update가 BF16 model view에서 사라져도 FP32 master에는 누적될 수 있다. 반대로 FP8 cast는 scale로 범위를 이동시키지만 block 안의 outlier가 scale을 지배하면 작은 값은 zero가 된다.

fixture는 representable boundary 양쪽, 최대 finite 근처, minimum normal/subnormal, signed zero, NaN/Inf를 사용한다. hardware가 subnormal을 flush하는지 operator/backend별로 확인한다. format 표를 곧바로 training 안정성 결론으로 바꾸지 않는다.

**rounding error를 연산 길이와 condition으로 예산화한다**

**한 번의 cast와 긴 reduction을 구분한다.**

round-to-nearest의 한 번 오차와 K개 항의 dot/reduction 누적 오차는 다르다. 단순 상한 `γ_k≈ku/(1-ku)`는 이해의 출발점이지만 실제 Tensor Core blocked accumulate, FMA와 reduction tree는 연산 순서가 다르다. condition과 cancellation이 실제 오차를 결정한다.

GEMM은 input quantization, product, accumulator, split-K reduction, epilogue와 output cast의 경계가 있다. softmax/norm은 max·sum·variance reduction과 exp/rsqrt approximation이 있다. optimizer는 moment recurrence와 parameter cast가 여러 step 누적된다.

error ledger에는 각 edge의 input range, format, accumulator, rounding mode, operation count/condition proxy와 observed error를 기록한다. final loss 오차만으로 어느 edge가 budget을 썼는지 알 수 없다.

**절대·상대·구조 invariant를 함께 사용한다.**

zero 근처에서는 상대 오차가 무의미하고 큰 값에서는 절대 오차만으로 과도하게 엄격할 수 있다. max/RMS/relative/cosine과 saturation/zero count를 tensor role별로 쓴다. causal mask, ignored gradient zero와 count conservation은 tolerance가 아닌 exact 구조 invariant다.

tolerance는 FP64/FP32 oracle과 pilot distribution을 보고 release 전에 고정한다. 실패 뒤 넓히면 wrong layout/mask를 정상 rounding으로 오인한다.

**TF32를 autocast dtype과 혼동하지 않는다**

**FP32 storage에서 matmul compute mode가 달라지는 경로를 찾는다.**

TF32는 일반적으로 FP32 tensor의 GEMM/convolution이 NVIDIA Tensor Core 경로에서 낮은 input mantissa 정밀도와 FP32 계열 accumulator를 사용하는 모드다. output tensor는 FP32일 수 있다. `tensor.dtype`만 출력해 TF32 사용 여부를 알 수 없다.

framework의 matmul precision/allow-TF32 설정, cuBLAS compute type와 actual kernel/metric을 기록한다. 설정이 true여도 작은 shape, unsupported layout 또는 deterministic policy 때문에 다른 path를 선택할 수 있다. requested/effective를 나눈다.

FP32 strict oracle→TF32-enabled eager→compiled/fused 순으로 비교한다. condition 높은 GEMM, cancellation과 role-coded matrix를 사용한다. convolution/attention projection과 optimizer matrix transform은 각각 dispatch가 다를 수 있다.

**TF32 성능을 BF16/FP16과 같은 recipe로 비교하지 않는다.**

BF16/FP16 autocast는 tensor dtype/cast, saved activation와 communication bytes를 바꾸지만 TF32는 FP32 storage가 유지될 수 있다. memory 절감과 loss scaling 요구가 다르다. 속도 비교에는 input/output bytes와 selected algorithm을 포함한다.

checkpoint는 TF32 compute flag보다 parameter dtype을 저장하지만 exact resume에는 environment/recipe가 필요하다. candidate environment가 default matmul policy를 바꾸면 child numerical certificate를 만든다.

**FP16과 BF16의 overflow·underflow를 backward까지 추적한다**

**forward finite가 backward 안전을 뜻하지 않는다.**

FP16은 좁은 exponent 때문에 작은 gradient underflow와 큰 activation/gradient overflow에 민감하다. loss scaling은 loss와 backward gradient를 확대해 small gradient를 representable 범위로 옮긴 뒤 optimizer 전에 unscale한다. 이미 forward에서 overflow한 activation을 loss scale로 고칠 수는 없다.

BF16은 exponent 범위가 넓어 dynamic loss scaling이 불필요한 경우가 많지만 fraction이 거칠어 cancellation, small update와 reduction 정밀도 문제가 남는다. norm/softmax/optimizer moment를 높은 정밀도로 유지하는 이유다.

forward activation, loss, scaled gradient, unscaled gradient, clip과 optimizer input을 별 TensorID로 둔다. first non-finite가 어디인지 찾는다. scaler found-inf만 보고 forward 원인을 놓치지 않는다.

**pathological fixture로 범위 경계를 만든다.**

large logits/activation, tiny loss gradient, long cancellation reduction, mixed magnitude parameter를 사용한다. FP32 oracle, FP16 without/with scaling, BF16을 비교한다. zero 비율, Inf/NaN, gradient cosine과 update를 본다.

checkpoint resume 뒤 GradScaler scale/growth tracker가 빠지면 처음 몇 step의 skip/update가 달라진다. next-update parity로 검증한다.

**autocast를 operator policy table로 복원한다**

**context manager가 모든 tensor를 한 dtype으로 바꾸지 않는다.**

autocast는 operator별로 lower precision 실행, FP32 강제, widest input promotion 같은 policy를 적용한다. 정확한 allow/deny 목록과 behavior는 framework revision/device type에 고정한다. Python scope 안에 있다는 사실로 actual kernel dtype을 단정하지 않는다.

parameter storage, input cast, kernel compute/accumulate, output와 saved tensor dtype을 operator card에 적는다. custom autograd 함수와 custom op는 autocast registration/decorator가 필요할 수 있다. 미등록 op가 FP32 fallback 또는 잘못된 low dtype을 받을 수 있다.

nested autocast disabled region과 explicit cast가 policy를 override한다. compiler가 cast를 fusion/reorder할 수 있으므로 eager graph와 generated graph를 비교한다.

**policy upgrade를 source diff와 trace로 확인한다.**

framework upgrade에서 autocast list/default가 바뀌면 model source가 같아도 dtype graph가 달라진다. fixed operator fixture와 runtime trace를 재실행한다. documentation의 일반 설명을 현재 build behavior로 대신하지 않는다.

unknown/custom operator는 FP32 reference와 dtype assertion을 가진다. silent promotion/fallback을 성능 결과에 표시한다. backward는 forward에서 선택된 cast와 saved state에 영향을 받으므로 별로 검증한다.

**GradScaler를 성공한 optimizer commit의 제어기로 읽는다**

**scale·growth tracker·found-inf의 사건 순서를 닫는다.**

일반 순서는 scaled loss backward, optimizer별 unscale, non-finite 검사, 선택적 clipping, step 실행/skip, scaler update다. 여러 optimizer가 있으면 각 found-inf와 global/individual skip 정책을 확인한다. `update`를 너무 일찍 호출하면 state clock이 갈린다.

scale growth/backoff factor, growth interval와 tracker는 checkpoint state다. attempted loop가 아니라 successful unscaled finite step을 어떻게 세는지 source에서 확인한다. gradient accumulation에서는 마지막 microstep에만 unscale/step한다.

double unscale, clip-before-unscale와 scheduler advance-on-skip을 mutation으로 만든다. expected gradient magnitude, parameter/moment와 scaler state에서 실패해야 한다. loss가 finite하다는 사실은 충분하지 않다.

**fused optimizer와 graph capture의 제어 경계를 확인한다.**

fused optimizer가 found-inf/scale tensor를 device에서 소비할 수 있다. host Python branch 없이 parameter update를 conditional하게 하는 contract와 step counter를 확인한다. CUDA Graph replay에서 scale/lr/found-inf device address가 안정하고 값은 최신이어야 한다.

skip된 step에 optimizer moment, FP8 amax/history, scheduler와 checkpoint UpdateID가 전진하는지 policy를 통합한다. partial state advance를 거부한다.

**FP8을 encode·scale·history·recipe의 합성 상태로 정의한다**

**실수 tensor와 FP8 bytes 사이의 변환을 명시한다.**

실수 x를 scale s와 format range에 맞춰 `q=cast_fp8(x·s)`로 저장하고 소비 시 `x̂=q/s` 또는 inverse scale을 사용한다. exact convention은 library source를 따른다. saturation, rounding와 zero 생성이 변환 오차다.

scale을 current amax에서 즉시 만들거나 과거 history로 delayed하게 만들 수 있다. margin, history length, amax algorithm과 update interval이 scale trajectory를 정한다. per-tensor, per-channel/block scaling은 metadata granularity와 kernel layout이 다르다.

forward activation, weight와 backward gradient는 서로 다른 format/quantizer/history를 쓸 수 있다. “FP8 recipe” 한 행이 아니라 tensor slot별 state 표가 필요하다.

**scale version과 parameter version을 연결한다.**

weight update 뒤 cached FP8 weight/transpose가 stale하지 않아야 한다. ParameterVersion, quantized cache version, scale/history version과 stream event를 연결한다. recompute/checkpoint에서 amax를 두 번 관측하는지 확인한다.

checkpoint에는 recipe, history/cursor, scale/inverse, cache rebuild key와 distributed owner를 둔다. derived cache를 저장하지 않으면 deterministic rebuild fixture가 필요하다.

**E4M3·E5M2 선택을 forward/backward tensor role로 검증한다**

**범위와 정밀도 trade-off를 실제 분포에 대입한다.**

forward activation/weight는 정밀도가 중요한 E4M3류를, backward gradient는 넓은 범위의 E5M2류를 쓰는 recipe가 있을 수 있다. 이것은 보편 규칙이 아니라 fixed Transformer Engine recipe와 hardware 지원에 따른다.

tensor role별 amax, percentile, saturation/zero와 quantization RMS/cosine을 측정한다. outlier 한 개가 per-tensor scale을 지배하는 경우와 block scaling을 비교한다. format만 바꾸고 scale algorithm을 함께 바꾸지 않는다.

backward gradient가 E5M2여도 accumulation과 weight gradient output은 higher precision일 수 있다. GEMM input formats, accumulator와 output를 별로 기록한다. kernel trace/descriptor에서 actual compute type을 확인한다.

**format mismatch를 checkpoint와 distributed에서 잡는다.**

forward slot history를 backward slot에 load, E4M3 scale을 E5M2 bytes와 결합, rank별 recipe mismatch를 주입한다. schema/version validator가 kernel launch 전에 실패해야 한다.

one-step과 long-horizon에서 layer output, gradient, optimizer delta와 scale trajectory를 BF16 parent와 비교한다. 품질 차이를 format 하나로 과장하지 않는다.

**NVFP4·block scaling을 두 단계 scale로 읽는다**

**4bit 값과 metadata overhead를 함께 센다.**

FP4 계열 training path는 작은 block마다 local scale을 두고 상위 scale을 더 둘 수 있다. effective real value는 FP4 code와 block/global scale의 합성이다. exact block size, scale format와 layout은 Transformer Engine 고정 source/tag와 NVIDIA 공식 recipe 문서에서 확인한다.

4bit payload만 보고 메모리/통신을 절반이라고 단정하지 않는다. block scale, alignment/padding, transpose cache와 higher-precision master/saved tensor가 추가된다. small/odd shape에서 metadata 비율이 커질 수 있다.

block 안 outlier, block boundary permutation, tail padding과 zero block을 fixture로 만든다. block scale index가 matrix layout/transpose와 함께 이동하는지 확인한다. bytes checksum만으로 scale-row mismatch를 잡지 못하므로 logical block ID를 사용한다.

**NVFP4 지원을 GPU capability와 kernel dispatch로 제한한다.**

library가 recipe class를 제공해도 target GPU/SM, toolkit, Transformer Engine binary와 operator shape가 kernel을 지원해야 한다. unsupported path가 BF16/FP8 fallback하면 actual coverage를 기록한다.

학습 convergence와 품질은 공개 paper/card 조건과 target experiment를 분리한다. source 존재를 모든 모델의 안정성 증거로 확대하지 않는다. `NotRun` hardware는 지원으로 표시하지 않는다.

**amax history를 작은 제어 시스템으로 해부한다**

**관측 지연이 scale overshoot와 saturation을 만드는 과정을 본다.**

delayed scaling은 과거 amax window에서 다음 scale을 결정한다. activation 분포가 갑자기 커지면 한두 step 동안 old scale로 saturation할 수 있다. 반대로 outlier 뒤 값이 작아져도 conservative scale이 precision을 잃게 할 수 있다.

history length, max/most-recent algorithm, margin과 update interval을 제어 parameter로 본다. step response로 sudden ×100, decay, alternating outlier와 stable input을 넣어 scale, saturation/zero와 settling을 측정한다.

microbatch마다 amax를 기록하고 optimizer step마다 scale을 갱신하는지, activation checkpoint recompute가 history에 기여하는지 source/context를 확인한다. 동일 logical tensor를 두 번 관측하면 scale trajectory가 달라진다.

**분산 amax aggregation의 owner와 denominator를 기록한다.**

TP/DP replica가 같은 quantized weight/activation scale을 공유해야 하는지 tensor role별로 정한다. max-reduce group, timing과 dtype을 기록한다. wrong process group 또는 rank 하나 history reset을 mutation으로 만든다.

checkpoint resume는 cursor/history/scale와 next amax update를 uninterrupted branch와 비교한다. history 초기화는 exact resume가 아니다.

### 14.12.2 stochastic rounding을 RNG state가 있는 cast로 취급한다

**기대값과 한 번의 결정론을 구분한다.**

stochastic rounding은 인접 representable 값 사이 확률을 사용해 양자화 오차의 기대 편향을 줄일 수 있다. 한 번의 출력은 deterministic nearest와 다르며 bitwise parity가 아니라 RNG 고정 또는 distributional property가 필요하다.

사용되는 tensor role, rounding mode, RNG algorithm/counter, seed와 device/rank ownership을 source/kernel descriptor에서 확인한다. “저정밀이면 stochastic rounding”이라고 가정하지 않는다. hardware instruction, library emulation과 output cast가 다를 수 있다.

tiny update accumulation에서 deterministic rounding이 계속 0이 되는 사례와 stochastic rounding 분포를 비교한다. 평균, variance, bias와 parameter trajectory를 여러 seed에서 측정한다. 단일 운 좋은 run으로 개선을 주장하지 않는다.

**checkpoint·graph·분산에서 RNG를 보존한다.**

cast RNG counter가 parameter/update order와 연결되면 fused list permutation이 결과를 바꾼다. stable ParameterID와 counter allocation을 기록한다. CUDA Graph replay에서 매번 같은 random bits를 재사용하지 않는지 확인한다.

resume 뒤 RNG/counter를 복원해 exact stream 또는 predeclared distributional parity를 검증한다. rank별 RNG가 필요한지 shared rounding이 필요한지 tensor ownership에 따라 정한다.

**Transformer Engine의 module·quantizer·extension call graph를 닫는다**

**Python context에서 device pointer까지 내려간다.**

Transformer Engine v2.6.0 source card를 기준으로 recipe/context 진입, module base의 FP8 state, Float8Tensor/quantizer, extension binding, cast/transpose와 cuBLASLt GEMM으로 이어지는 호출을 그린다. 각 edge의 tensor slot, scale/amax pointer, workspace와 stream을 적는다.

linear module forward는 input/weight quantization, possible cache/transpose, GEMM, bias/activation와 output를 만든다. backward에는 grad output quantization, dgrad/wgrad GEMM과 amax/history update가 포함된다. forward의 거울이라고 축약하지 않는다.

TE version, CUDA/cuBLASLt, PyTorch extension ABI와 GPU capability가 같은 binary manifest에 있어야 한다. Python package version만으로 loaded shared object를 증명하지 않는다.

**upstream test의 범위와 local integration을 분리한다.**

TE unit test가 quantizer/serialization 또는 module parity를 확인해도 target model의 TP group, checkpoint wrapper와 compiler integration을 모두 증명하지 않는다. test node, config/device와 asserts를 기록한다.

local GoldenLayer는 forward/backward/one-step, history resume와 stale cache mutation을 포함한다. actual kernel trace와 fallback coverage를 확인한다. binary 없는 환경에서는 source evidence와 `NotRun`을 구분한다.

**cuBLASLt GEMM descriptor를 수치 계약으로 읽는다**

**A·B·C·D dtype과 compute type을 따로 적는다.**

cuBLASLt matmul은 input/output data types, compute type, transpose/layout, scale type, epilogue, alpha/beta, pointer alignment와 workspace를 descriptor/preference로 지정한다. FP8 input이라는 한 값만으로 accumulator/output을 알 수 없다.

algorithm heuristic은 shape, alignment, workspace와 toolkit/library version에 따라 다른 kernel을 고를 수 있다. deterministic/split-K, epilogue support와 actual algorithm ID를 기록한다. heuristic success가 수치 parity를 보장하지 않는다.

bias/GELU/ReLU 등 epilogue fusion은 output cast와 activation 순서를 바꾼다. eager reference에서 bias before/after cast를 구분한다. beta nonzero와 accumulated output도 fixture에 넣는다.

**descriptor mutation을 first gate에서 잡는다.**

transpose/leading dimension, scale pointer, compute type, epilogue, alpha/beta와 workspace size를 하나씩 잘못 넣는다. role-coded matrices가 layout 오류를 드러내야 한다. unsupported combination의 explicit error/fallback을 확인한다.

CUDA 12.x/13.x 비교에서는 cuBLASLt component exact version과 selected algorithm을 함께 diff한다. toolkit major만 원인으로 단정하지 않는다.

**CUTLASS template와 generated kernel의 specialization을 추적한다**

**template parameter가 실제 instruction·layout을 결정한다.**

CUTLASS kernel은 architecture tag, operator class, element/accumulator types, layout, tile/cluster shape, stages와 epilogue를 template/config로 고정한다. source가 compile된 binary specialization과 같은지 build manifest와 symbol/fatbin으로 확인한다.

CUDA toolkit과 CUTLASS revision, compiler flag, target arch가 generated code를 바꾼다. header-only라는 이유로 runtime artifact provenance를 생략하지 않는다. vendored CUTLASS와 system checkout을 구분한다.

kernel selection wrapper는 shape/alignment/dtype/SM guard를 갖는다. guard를 통과하지 못하면 다른 specialization 또는 framework fallback이다. production shape histogram에서 hit rate를 계산한다.

**reference와 specialization의 수치·성능을 분리한다.**

host/device reference는 correctness oracle, target kernel은 actual path다. odd/tail, misalignment, noncontiguous, split-K와 epilogue fixture를 포함한다. max/RMS/cosine과 exact structural invariant를 본다.

performance는 tile utilization, executed FLOPs, HBM, workspace와 critical path를 기록한다. 한 architecture의 최적 tile을 다른 SM에 그대로 적용하지 않는다.

**Triton kernel을 Python source와 JIT artifact의 두 revision으로 고정한다**

**source code만으로 실제 kernel을 식별하지 않는다.**

Triton JIT는 Python kernel source, constexpr/meta parameters, compiler/Triton version, target GPU와 launch config로 generated IR/PTX/cubin을 만든다. autotuner가 config를 선택하면 key와 cache artifact가 실행 의미/성능의 일부다.

framework compile이 Triton source를 동적으로 생성할 수 있다. generated source/IR hash와 call site, input guards를 저장한다. source revision이 같아도 compiler upgrade로 kernel이 바뀔 수 있다.

masked load/store, boundary, reduction, accumulator와 `tl.dot` input precision option을 읽는다. program ID mapping과 stride가 logical tensor layout과 맞는지 role-coded fixture로 검증한다.

**stale cache와 wrong target을 실패 주입한다.**

다른 SM/toolchain에서 만든 cache, meta key 누락, shape collision과 corrupted artifact를 넣는다. cache resolver가 재compile하거나 거부해야 한다. 우연히 load되어 실행되면 binary target/guard가 잡아야 한다.

eager/reference→Triton→compiled graph를 forward/backward와 performance로 비교한다. JIT warm/cold를 나누고 compile time을 배포 비용에 포함한다.

**compiler fusion이 rounding edge를 이동시키는 순간을 찾는다**

**연산 graph 동일성과 수치 동일성을 구분한다.**

eager에서 GEMM output을 BF16로 저장한 뒤 bias/activation을 수행하던 경로가 fusion되면 accumulator/epilogue에서 bias/activation 후 한 번만 cast할 수 있다. 수학적 실수 연산은 같아도 rounding edge가 달라진다.

norm+residual, softmax attention, cross entropy와 optimizer fusion도 reduction/order/saved state를 바꿀 수 있다. compiler graph/IR에서 casts, reductions와 custom calls를 확인한다. 최종 kernel count 감소만 보지 않는다.

precision ladder는 eager exact policy→compiled without fusion 가능 경로→fused generated kernel을 비교한다. expected difference budget과 invariants를 사전에 둔다. fusion이 품질을 개선/악화한다고 일반화하지 않는다.

**graph break와 fallback을 별 실행으로 분류한다.**

dynamic shape, unsupported custom op, hook와 data-dependent branch가 graph를 끊을 수 있다. partial compile의 경계와 eager fallback dtype policy를 기록한다. benchmark에서 compiled 비율/graph count와 recompilation을 표시한다.

framework/compiler upgrade는 generated kernel과 guards를 stale로 만든다. checkpoint는 portable mathematical state만 보존하고 graph를 target environment에서 재생성한다.

**CUDA Graph replay에서 scale·lr·RNG가 살아 움직이는지 확인한다**

**주소 안정성과 값 갱신을 분리한다.**

CUDA Graph는 capture한 launch와 pointer를 replay한다. tensor address가 같아도 learning rate, GradScaler scale, FP8 scale/history와 RNG counter가 최신 device value를 읽어야 한다. capture 때 Python scalar가 상수화되지 않았는지 본다.

lazy FP8 quantizer/workspace, optimizer state와 collective buffer를 warm-up에서 생성한다. allocator가 capture pool 주소를 유지하고 model/adapter 변경 뒤 old graph를 재사용하지 않게 generation key를 둔다.

overflow/skip, variable shape, optional modality/expert와 checkpoint recompute 같은 control path를 graph 안/밖 정책으로 정한다. graph를 강제로 replay해 잘못된 branch를 실행하지 않는다.

**graph fixture를 여러 replay로 구성한다.**

lr/scale 변경, normal→overflow→normal, stochastic rounding counter, two sequence shapes와 checkpoint reload를 순서대로 실행한다. eager control과 parameter/state/next scale을 비교한다. 첫 replay만 맞는 시험은 부족하다.

distributed capture는 NCCL graph support/collective order와 group을 확인한다. rank 하나가 다른 graph generation을 쓰면 admission에서 막는다.

**CUDA 12.x에서 13.x로 이동할 때 compiler 축만 먼저 격리한다**

**같은 source에서 toolkit build artifact를 두 개 만든다.**

extension source, framework headers, host compiler와 target arch list를 고정하고 CUDA toolkit/nvcc·ptxas/device library만 바꾼 build를 만든다. 가능한 조합인지 official release note/support matrix를 먼저 확인한다. 불가능하면 compound migration으로 표시한다.

compile command, macros, include/library path, PTX ISA, cubin/fatbin targets와 binary digest를 비교한다. build 성공 뒤 target GPU load, actual dispatch와 numerical ladder를 실행한다. source compile success는 runtime support가 아니다.

compiler diagnostic/deprecation을 무시해 오래된 API가 다른 fallback을 타지 않는지 본다. generated SASS를 책에서 과잉 해석하지 않고 instruction class/target 확인과 profiler evidence로 제한한다.

**driver/runtime/library 축은 별 child 실험으로 둔다.**

같은 binary를 compatible driver 후보에서 실행해 driver/PTX JIT 축을 본다. user-space cuBLASLt/NCCL/TE를 바꾸는 실험은 library algorithm/ABI 축이다. framework까지 올리면 dispatch/autocast/compiler source가 바뀐다.

성능 차이를 “CUDA 13 효과”로 묶지 않고 changed component와 actual kernel/algorithm에 귀속한다. 각 child는 rollback parent를 가진다.

**CUDA 13 전환에서 제거·지원 범위 변화를 dependency graph로 반영한다**

**공식 release notes의 added·changed·deprecated·removed를 소비자에 연결한다.**

CUDA Toolkit 13.0 release notes와 compatibility/programming guide의 정확한 section을 source card에 둔다. host compiler/OS/GPU architecture support, library component version과 deprecated/removed 기능을 framework, TE, Triton, CUTLASS, custom extension dependency에 투영한다.

책의 고정 비교는 12.9.1과 13.0 문서를 기준으로 하며, 이후 13.x minor의 내용을 자동 포함하지 않는다. 설치된 13.x patch가 다르면 해당 release notes와 component table로 child card를 만든다.

오래된 SM 대상 binary 생성 정책 변화는 새 GPU가 빠르다는 문제와 다르다. target arch가 더 이상 toolkit에서 생성되지 않으면 old binary/runtime 또는 별 build strategy가 필요할 수 있다. 정확한 지원 여부는 문서와 build probe로 판정한다.

**API/ABI 영향과 numerical effect를 별 gate로 둔다.**

header/API compile, dynamic symbol load, binary target, kernel dispatch, numerical parity와 performance를 순서대로 본다. build가 성공해도 library heuristic이 달라 output/performance가 바뀔 수 있다. ABI가 맞아도 unsupported dtype/SM fallback일 수 있다.

미실행 조합은 official support 문장을 runtime PASS로 올리지 않는다. 문서 scope와 actual evidence를 나란히 둔다.

**fatbin·cubin·PTX와 SM capability를 실행 artifact로 읽는다**

**binary에 무엇이 들어 있는지 확인한다.**

fat binary는 여러 architecture의 cubin과 PTX를 포함할 수 있다. target GPU와 정확히 맞는 cubin, compatible cubin 또는 PTX JIT 중 어느 경로인지 확인한다. `TORCH_CUDA_ARCH_LIST` 같은 build option의 요청값과 binary inspection을 비교한다.

PTX 포함은 미래 GPU 호환 가능성을 줄 수 있지만 driver가 PTX ISA를 JIT할 수 있어야 하고 architecture-specific 최적화/성능은 다를 수 있다. PTX가 있다는 사실을 모든 future SM 지원으로 확대하지 않는다.

`no kernel image`는 package import가 아니라 특정 extension kernel의 target 누락일 수 있다. first failing symbol/kernel과 binary path를 찾는다. 여러 extension이 서로 다른 arch list를 가질 수 있다.

**SM feature guard와 actual operator coverage를 분리한다.**

GPU가 FP8/FP4 instruction을 지원해도 framework/library/kernel이 해당 shape/dtype으로 dispatch해야 한다. device capability check, compiled specialization와 wrapper guard를 모두 본다.

support matrix cell에는 GPU model/SM, binary target, toolkit/compiler, loaded library, operator shape와 actual kernel EvidenceID를 기록한다. 새 GPU에서는 old PASS를 복사하지 않는다.

### 14.12.3 cuBLASLt heuristic과 autotune cache를 재현 상태로 관리한다

**알고리즘 선택 입력을 보존한다.**

GEMM shape/strides/dtypes, compute type, epilogue, alignment, workspace limit, determinism와 library version이 heuristic 후보를 결정한다. 선택 algorithm ID와 required workspace, timing을 기록한다. 같은 CUDA major에서도 component patch가 다르면 후보가 달라질 수 있다.

autotune은 warm-up state, clock/noise와 input pointer alignment의 영향을 받을 수 있다. benchmark iteration, chosen result와 cache key를 artifact로 둔다. cache를 portable mathematical checkpoint로 취급하지 않는다.

workspace limit 변경은 algorithm과 peak memory/latency/numerics를 바꿀 수 있다. OOM 완화로 limit을 줄이면 child performance/numerical evidence가 필요하다.

**stale/poisoned algorithm을 negative control로 넣는다.**

다른 shape/SM/library에서 만든 cache entry, insufficient workspace와 unsupported epilogue를 주입한다. resolver가 reselect/error해야 한다. wrong algorithm을 실행한 결과가 finite여도 descriptor/guard가 막아야 한다.

CUDA 12/13 비교는 same algorithm forced 가능 여부와 default heuristic을 별로 측정한다. default 성능 차이와 compiler/library kernel 자체 차이를 분리한다.

**attention dtype contract를 QK·softmax·PV·backward로 나눈다**

**QK GEMM과 softmax reduction의 정밀도를 분리한다.**

Q/K storage와 input cast, dot accumulator, scale, mask, row max/exp/sum accumulator, probability/dropout와 V GEMM output dtype을 적는다. FlashAttention류 fused kernel은 score matrix를 materialize하지 않아도 같은 logical stages/LSE state를 유지한다.

FP8 Q/K/V를 써도 softmax statistic/LSE는 higher precision일 수 있다. causal/sliding/padding mask는 exact support invariant다. all-masked row, long context, near-tied scores와 extreme logits를 fixture로 둔다.

backward에는 saved/recomputed LSE, dropout RNG, dQ/dK/dV accumulation와 reduction order가 포함된다. forward parity만으로 training kernel을 승인하지 않는다. activation checkpoint recompute가 scale/history/RNG를 이중 갱신하지 않아야 한다.

**backend selection과 fallback을 shape histogram으로 기록한다.**

head dimension, sequence, mask, dropout, output-attention request, dtype와 SM guard가 eager/SDPA/Flash/custom 경로를 정한다. requested backend와 actual kernel을 trace한다.

fallback은 accumulator/memory/latency가 다를 수 있다. production hit rate와 tail shapes를 support matrix에 넣는다. 8장의 attention 수학과 이 장 dtype/dispatch를 같은 TensorID로 잇는다.

**norm과 softmax를 reduction tree로 시험한다**

**평균·분산·RMS·log-sum-exp의 accumulator를 확인한다.**

RMSNorm/LayerNorm은 squared values의 reduction, epsilon, rsqrt와 affine로 구성된다. softmax는 max subtraction, exp와 sum이다. input/output dtype만 보고 내부 accumulator를 알 수 없다. generated/fused kernel source와 trace를 확인한다.

reduction tree, vector width와 tail 처리로 결과가 달라질 수 있다. long row, odd hidden, large+small cancellation, constant/zero input와 all-masked logits를 쓴다. FP32 reference와 forward/backward를 비교한다.

epsilon은 low precision cast 전/후 적용 위치가 중요하다. saved mean/rstd/LSE dtype과 recompute path를 기록한다. fusion이 residual add를 norm accumulator에 포함하는지 본다.

**분산 norm과 sequence parallel 경계를 확인한다.**

hidden dimension이 shard되면 statistic을 global reduce해야 할 수 있다. token/sequence shard는 local statistic일 수 있다. process group와 collective dtype을 model parallel layout에서 도출한다.

wrong group/local-only statistic을 mutation으로 넣는다. shape는 맞고 output finite여도 FP32 global oracle와 달라야 한다. 15장의 ownership과 연결한다.

**optimizer dtype contract를 master·moment·delta·view로 닫는다**

**fused AdamW의 opaque kernel을 logical recurrence로 복원한다.**

gradient unscale/clip, FP32 master 존재, first/second moment dtype, bias correction, epsilon, decay, parameter update와 model view cast 순서를 적는다. fused/foreach/single path가 같은 RecipeID의 recurrence를 만족하는지 gradient tape로 확인한다.

FP32 master 없이 BF16 parameter를 직접 갱신하면 small delta가 사라질 수 있다. 2×2 parameter에 ULP보다 작은 반복 delta를 넣어 master 누적과 direct update를 비교한다. stochastic rounding이 있다면 RNG state를 포함한다.

FP8 model weight는 보통 optimizer canonical/master와 quantized cache를 구분한다. parameter update 뒤 cache version/scale을 갱신한다. stale cache는 next forward fixture가 잡아야 한다.

**overflow와 multi-optimizer transaction을 검증한다.**

AdamW, Muon/other group, FP8 state가 한 model step에 있다면 found-inf에서 모두 skip하거나 explicit policy를 가진다. one group moment만 전진하지 않는다. scheduler/scaler와 checkpoint UpdateID를 같이 본다.

checkpoint는 master/moments/step, model view, scale/history와 RNG를 same commit에 둔다. next-two-update parity로 load를 검증한다.

**collective dtype와 compression을 global 수학에서 도출한다**

**전송 dtype과 reduction accumulator를 나눈다.**

gradient bucket이 BF16/FP16/FP32인지, collective가 어떤 dtype으로 합하고 output을 어디에 저장하는지 기록한다. NCCL API payload dtype만으로 internal protocol/reduction order와 error를 모두 알 수 없지만 logical expectation과 trace를 고정할 수 있다.

FP8/FP4 communication compression은 payload 외 scale metadata를 요구한다. rank-local scale로 quantize한 값을 그대로 add할 수 없으므로 shared scale, dequantize-high-precision reduce 또는 error-feedback 같은 exact algorithm을 source에서 확인한다.

residual/error-feedback state가 있으면 checkpoint와 owner가 필요하다. 통신 압축 옵션이 model compute FP8 recipe와 같은 scale state라고 가정하지 않는다.

**global FP32 oracle와 rank-skew fixture를 사용한다.**

rank마다 magnitude/domain이 다른 gradient와 zero-valid-token rank를 만든다. concatenate/global denominator FP32 reference, uncompressed collective와 compressed path를 비교한다. sum/mean convention, scale와 saturation을 측정한다.

world size/topology 변경에서 scale/error state migration을 검증한다. bandwidth 절감과 collective tail, extra quantize/dequantize kernel을 함께 센다.

**low-precision checkpoint를 portable state와 rebuildable cache로 분리한다**

**다음 수학적 update를 결정하는 state를 저장한다.**

canonical/model/master weights, optimizer moments/step, scaler, FP8/FP4 recipe, amax history/cursor, scale, stochastic RNG, scheduler와 data cursor가 portable state 후보다. tensor-parallel scale slice와 ParameterID mapping을 포함한다.

quantized weight transpose, cuBLASLt algorithm, Triton/compile cache와 CUDA Graph executable은 environment-specific derived artifact일 수 있다. 저장하더라도 target digest가 다르면 폐기/rebuild한다. binary deserialize 성공을 portable resume로 쓰지 않는다.

checkpoint root는 모든 component가 같은 committed UpdateID와 RecipeID를 가리키는지 확인한다. scale history만 stale하거나 master/model view가 다른 mixed generation을 거부한다.

**rebuild와 exact resume를 별로 검증한다.**

portable state load→cache/graph rebuild→GoldenBatch forward/backward/next update를 uninterrupted branch와 비교한다. rebuild가 autotune algorithm을 바꿔도 numerical tolerance/성능 support를 다시 판정한다.

CUDA 12→13 migration에서 portable schema compatibility와 binary/runtime compatibility를 분리한다. unsupported new state는 silent reset하지 않고 warm migration으로 표시한다.

**topology 변경 시 scale block과 parameter shard를 함께 이동한다**

**global logical tensor에서 old/new slice를 계산한다.**

TP/DP/EP degree가 바뀌면 parameter/master/moment뿐 아니라 per-channel/block scale, amax history와 quantized cache metadata가 같은 logical offset mapping을 따라야 한다. scale first dimension이 packed weight layout과 다를 수 있어 schema를 명시한다.

global ParameterID, quantization block ID, logical range와 owner rank를 manifest에 둔다. coverage/overlap과 tail padding을 검사한다. bytes 수가 맞아도 scale block permutation은 output을 오염시킨다.

activation scale가 runtime/replica-local인지 durable global state인지 tensor role별로 정한다. DP replica history를 합칠지 각각 복원할지 recipe가 결정한다. controller group가 바뀌면 exact mapping을 지원하는지 본다.

**reshard fixture를 role-coded scale로 만든다.**

각 block weight/scale에 고유 pattern을 넣어 old→new→global reconstruction과 one-step output을 비교한다. wrong scale swap, missing tail과 rank-local history reset을 주입한다.

unsupported reshard는 canonical high-precision weight에서 scales를 recalibrate하는 warm migration일 수 있다. calibration data/recipe와 quality gate를 선언한다. exact resume라고 쓰지 않는다.

**low-precision 장애를 non-finite 이전의 신호로 찾는다**

**saturation·zero·scale lag·cosine drift를 관측한다.**

NaN/Inf는 늦은 증상이다. tensor slot별 amax, scale, saturation count, zero/subnormal, quantization error RMS/cosine와 history age를 기록한다. layer/role와 UpdateID로 join한다.

GEMM output이 finite여도 small singular/value direction이 사라질 수 있다. FP32/BF16 sampled oracle와 gradient/update cosine을 본다. norm/softmax는 LSE/rstd와 backward invariant를 표본화한다.

unexpected fallback, algorithm change, graph break와 recompile은 numerical/performance 변화를 만든다. kernel/descriptor EvidenceID와 tensor error를 같은 trace에 둔다.

**경보를 원인 분기로 바꾼다.**

saturation 급증이면 input distribution, scale/history lag, wrong tensor slot와 format을 본다. zero 급증이면 scale/outlier, underflow/rounding과 cast 위치를 본다. resume 직후 scale drift면 checkpoint history/owner를 본다.

loss spike와 모든 low-precision metric이 정상이라면 data/loss/optimizer clock을 조사한다. “FP8 문제”라는 넓은 분류로 끝내지 않는다.

**performance 주장을 cast·metadata·kernel·communication으로 분해한다**

**저정밀 payload 절감이 end-to-end speedup으로 가는 조건을 계산한다.**

GEMM Tensor Core 처리량과 weight/activation bytes 감소가 이익을 만든다. 반면 cast, transpose, amax reduction, scale update, metadata, workspace, graph compile와 fallback이 비용이다. small GEMM이나 low coverage에서는 overhead가 이익을 넘을 수 있다.

operator별 FP8/FP4 coverage, cast/amax kernel time, GEMM, collective, host gap와 peak memory를 trace한다. theoretical peak FLOPs를 tokens/s로 동일시하지 않는다. fused kernel은 eliminated HBM bytes를 계산한다.

CUDA 12/13 비교는 actual cuBLASLt/CUTLASS/Triton algorithm과 compile/JIT cache를 맞추거나 차이를 명시한다. framework/TE upgrade가 함께면 compound effect다.

**공정 benchmark의 warm/cold와 quality 조건을 고정한다.**

cold compile/autotune, warm steady, checkpoint/restart와 long-horizon useful updates/hour를 별로 측정한다. 같은 batch/sequence/token/quality tolerance를 사용한다. memory 절감으로 batch를 키웠다면 별 recipe 효과다.

수치/quality gate를 실패한 빠른 kernel은 성능 후보가 아니다. fallback run을 target precision throughput에 포함하지 않는다.

**CUDA 12.x·13.x support matrix를 patch 단위 artifact로 만든다**

**major 번호보다 exact component tuple을 key로 쓴다.**

행 key는 GPU/SM, driver, toolkit compiler, framework wheel build/runtime, cuBLASLt, NCCL, TE, Triton/CUTLASS/custom extension와 binary digest다. `12.x` 또는 `13.x`는 요약 label일 뿐 support identity가 아니다.

열은 build, load, binary target, GEMM/attention/norm/optimizer/collective/graph/export의 dtype/shape다. 각 cell에 official doc scope, source guard, actual dispatch, numerical/checkpoint/performance EvidenceID와 status를 둔다.

CUDA 12.9.1→13.0 고정 비교를 기준 child로 보존하고, 13.1 등 이후 release는 해당 release note와 새 artifact로 추가한다. future behavior를 소급하지 않는다.

**status를 supported·correct·tuned로 분리한다.**

official/framework supported, build/load successful, numerical correct, checkpoint-resumable와 performance-tuned는 다른 상태다. generic fallback이 correct해도 FP8 tuned cell은 아니다.

환경 변경은 affected cells만 stale로 만들되 ABI/dispatch common dependency가 바뀌면 여러 cell을 갱신한다. 독립 reviewer가 tuple에서 raw evidence까지 이동할 수 있어야 한다.

## 14.13 공식 문서에서 Tensor Core와 precision migration 승인까지

마지막 대절은 문서의 지원 범위를 코드와 실행으로 좁히는 독해 절차다. GEMM shape에서 dispatch 조건을 역산하고, precision 변경이 objective·optimizer·scheduler의 의미를 보존했다는 교차 증거가 있을 때만 migration을 승인한다.

### 14.13.1 official CUDA 문서를 코드·실행과 연결하는 독해 절차

**문서 종류별 질문을 구분한다.**

release notes는 component/version, added/changed/deprecated/removed와 known issue를 찾는다. compatibility 문서는 driver와 toolkit/user-space 관계를, programming guide는 execution/memory/synchronization 의미를, library docs는 API/data/compute type와 algorithm contract를 제공한다.

문서의 “지원”은 application build/dispatch/numerical result를 자동 증명하지 않는다. 문장마다 section/version, claim와 소비 component를 evidence card에 둔다. archive URL로 exact version을 고정한다.

source에서는 framework guard, extension build condition와 descriptor를 찾고, binary에서는 target/linked library를, runtime에서는 loaded library/selected kernel을 확인한다. 네 층이 같은 support claim을 향해야 한다.

**known issue와 workaround를 recipe state로 남긴다.**

특정 toolkit/library/SM issue가 있다면 affected operator/shape, workaround flag, numerical/performance 영향과 removal condition을 기록한다. workaround를 영구 default로 숨기지 않는다.

새 patch에서 issue가 고쳐졌다는 release note만으로 local workaround를 제거하지 않고 fixed fixture를 실행한다. 반대로 문서에 없는 local failure도 runtime evidence로 남긴다.

**error budget을 한 step에서 긴 학습으로 확장한다**

**local error와 trajectory divergence를 구분한다.**

한 operator의 small rounding 차이는 chaotic/nonlinear training trajectory에서 시간이 지나며 커질 수 있다. bitwise 동등을 요구하지 않더라도 finite, loss distribution, gradient/update 통계와 downstream 품질 예산이 필요하다.

GoldenStep은 operator/first update correctness를, short deterministic cycle은 scaler/scale/history/moment와 resume drift를, long paired run은 stability/quality를 담당한다. 한 rung의 PASS를 다른 rung으로 확대하지 않는다.

selected layer output/gradient/update error와 scale/saturation trajectory를 UpdateID별로 기록한다. first threshold crossing을 binary search해 operator ladder로 돌아간다. 최종 loss만 보고 tolerance를 조절하지 않는다.

**여러 seed와 실패 run을 포함한다.**

stochastic rounding, data order와 kernel nondeterminism이 있으면 multiple seed와 confidence interval을 사용한다. NaN/OOM/hang과 recovery 실패를 제외하지 않는다. baseline에도 같은 tuning/resource budget을 준다.

quality slice와 safety/evaluation을 포함한다. 저정밀 최적화가 rare token/domain 또는 multimodal outlier에 다른 영향을 줄 수 있다.

**failure fixture를 representation부터 commit까지 층별로 배치한다**

**표현 층을 깨뜨린다.**

maximum finite 초과, tiny/subnormal, outlier block, E4M3/E5M2 mismatch, scale block permutation와 stochastic RNG reuse를 주입한다. cast/quantizer detector가 잡아야 한다.

**연산·dispatch 층을 깨뜨린다.**

wrong accumulator/epilogue, transpose/stride, unsupported SM, stale Triton/compile cache, cuBLASLt algorithm/workspace와 eager fallback을 넣는다. source guard/binary/runtime/numerical 중 expected first gate를 지정한다.

**분산·상태 층을 깨뜨린다.**

rank-local scale mismatch, wrong amax group, collective dtype/denominator, history cursor reset, master/cache version mismatch와 graph stale pointer를 넣는다. count/shape만 아니라 logical ID를 검사한다.

**commit 층을 깨뜨린다.**

GradScaler one-step-ahead, partial optimizer update, checkpoint missing history/scale, topology reshard scale swap와 candidate binary rollback을 시험한다. old/new complete terminal 외 상태를 거부한다.

**관측 지표의 cardinality와 표본 비용을 설계한다**

**상시 metric과 상세 trace를 나눈다.**

상시 metric은 recipe/backend별 non-finite, saturation/zero, scale range/history age, scaler skip, fallback, kernel/collective latency와 checkpoint health를 layer/role 수준으로 요약한다. TokenID/tensor 개별 값은 label로 넣지 않는다.

상세 artifact는 sampled TensorID의 amax/scale/error, kernel descriptor, RNG/version와 gradient/update를 보존한다. trace exemplar/RunID로 dashboard에서 연결한다. 민감한 raw activation은 통계/보안 정책을 따른다.

비싼 FP32 oracle/SVD는 canary/offline sample에서 실행한다. production probe가 graph break나 synchronization으로 성능을 크게 바꾸지 않는지 측정한다. debug mode와 performance result를 분리한다.

**metric 자체의 denominator와 clock을 검증한다.**

saturation rate의 element denominator, scale p99의 tensor slot 집합, skip rate의 attempted/committed step을 명시한다. checkpoint resume에서 counter reset이 가짜 개선을 만들지 않게 한다.

alert fixture로 intentional saturation, fallback와 stale history를 넣어 expected dashboard/runbook으로 이어지는지 확인한다.

**low-precision incident를 최초 달라진 edge로 좁힌다**

**NaN incident의 precision ladder를 재생한다.**

same input/checkpoint에서 FP32 strict, TF32, BF16/FP16 eager, fused, FP8/FP4, compiled/graph, distributed를 순서대로 실행한다. 마지막 통과 rung과 첫 실패 edge의 tensor/state/kernel을 비교한다.

FP32/BF16부터 다르면 data/checkpoint/model 함수 가능성이 크다. FP8에서만 다르면 quantizer/scale/history/cache, compiled에서만이면 fusion/guard/cache, distributed에서만이면 collective/owner를 본다.

**느림 incident는 actual coverage부터 본다.**

FP8 flag가 켜져도 GEMM coverage, cast/amax overhead, small shape, fallback/compile와 collective tail을 확인한다. GPU utilization 평균 대신 timeline/roofline을 본다. CUDA candidate의 selected algorithm과 binary를 baseline과 비교한다.

**resume drift는 portable state를 먼저 본다.**

master/model view, optimizer/scaler, recipe/history/scale/RNG, data cursor와 graph rebuild를 확인한다. first next update intermediate를 uninterrupted control과 비교한다. tolerance를 넓혀 mixed generation을 통과시키지 않는다.

**training에서 export·serving으로 precision lineage를 넘긴다**

**canonical weight와 training cache를 구분한다.**

exporter가 FP32 master, BF16 model view 또는 FP8 cached weight 중 무엇을 읽는지 명시한다. training delayed scale/amax가 inference calibration과 목적/분포가 다를 수 있다. 그대로 재사용하려면 compatibility fixture가 필요하다.

target serving format의 per-tensor/channel/block scale, packing/layout, fused operator와 target SM/runtime를 ExportRecipeID로 고정한다. parent training CheckpointID와 tokenizer/config를 연결한다.

serving engine/cubin/compile cache는 environment-specific artifact다. CUDA 12 build를 13 runtime에 옮길 수 있는지는 backend 공식 지원과 actual load/dispatch/parity로 판정한다. 필요하면 canonical weight에서 rebuild한다.

**train/serve parity를 layer와 token에서 확인한다.**

known hidden/input IDs로 embedding, selected GEMM/attention, logits와 autoregressive decode를 비교한다. training FP8 stochastic/dropout state와 eval serving policy 차이를 명시한다. cache/KV quantization은 별 state다.

wrong scale/layout/tied head, unsupported SM와 stale engine을 mutation으로 넣는다. finite logits는 semantic parity 증거가 아니다.

**CUDA 환경 변경의 rollback을 binary와 state 두 축으로 연습한다**

**portable parent를 보존한다.**

candidate CUDA 13/toolchain에서 한 update를 진행하기 전 CUDA 12 parent의 model/master/optimizer/scaler/FP8 state, data/RNG와 source recipe를 durable하게 보존한다. candidate-only binary/cache는 별 generation이다.

candidate state schema가 parent binary에서 읽힐 수 있는지 converter로 확인한다. incompatible history/FP4 metadata를 old code에 강제로 load하지 않는다. exact rollback이 불가능하면 candidate update 전 admission을 막거나 warm conversion 계획을 세운다.

rollback은 package downgrade만이 아니라 process loaded libraries, extension binaries, Triton/compile/autotune cache와 CUDA Graph를 parent environment로 복원한다. environment digest가 cache key를 분리해야 한다.

**rollback 뒤 next updates를 재생한다.**

parent checkpoint에서 same GoldenBatch와 data cursor로 forward/backward/optimizer/scale trajectory를 확인한다. candidate process의 stale shared memory/cache가 섞이지 않게 새 process/container를 사용한다.

release rehearsal은 failure trigger, containment, artifact resolver와 evidence update 시간을 측정한다. rollback 가능성이 수치 성능만큼 승인 기준이다.

**low-precision 옵션을 state 변화표로 번역한다**

**사용자 flag에서 실제 tensor slot까지 이동한다.**

`autocast dtype`, TF32 policy, GradScaler enable/growth, TE recipe/format/margin/history, FP4 block, stochastic rounding, attention backend, compile/graph, collective compression과 workspace flag를 행으로 둔다.

각 행에는 parser/default, source consumer, created state/buffer, dtype/scale/RNG owner, dispatch guard, checkpoint field, expected numerical/performance effect와 failure fixture가 있다. flag가 존재하지만 selected module/backend가 소비하지 않으면 `Unused`다.

옵션 조합 validation도 계약이다. FP8 recipe가 unsupported GPU에서 explicit error인지 fallback인지, graph와 dynamic scaling이 어떻게 조합되는지 source/test로 확인한다. documentation table만 믿지 않는다.

**한 옵션씩 paired child를 만든다.**

same parent/GoldenBatch에서 changed first edge, kernel/state, error와 resource를 기록한다. 여러 옵션을 동시에 켜면 원인을 분리할 수 없다. support matrix 제약 때문에 compound migration이면 그대로 표시한다.

option default가 framework upgrade로 바뀌면 effective config digest가 달라져야 한다. serialized user config가 같다는 이유로 evidence를 승계하지 않는다.

**14장의 독립 재현 실습을 한 GPU 없이도 준비한다**

**정적 evidence와 실행 evidence를 구분한다.**

GPU가 없는 환경에서도 official archive docs, source commit/tag, build config, operator dtype/scale schema, checkpoint fields와 expected fixture를 준비할 수 있다. 실제 dispatch/numerical/performance는 `NotRun`이다.

FP64 software emulation으로 FP16/BF16/FP8/FP4 quantization boundary와 small GEMM/error ledger를 만들 수 있지만 hardware rounding/kernel과 동일하다고 주장하지 않는다. GPU job이 이 fixture를 소비하게 한다.

environment resolver는 GPU/SM/driver/toolkit/wheel/library/binary를 수집하고 support matrix cell을 선택한다. mismatched tuple이면 실행 전에 reject한다.

**GPU 인수 job의 순서를 고정한다.**

binary/load probe→FP32 strict→target eager→fused→FP8/FP4→compile/graph→distributed→checkpoint/resume→performance 순이다. 앞 rung이 실패하면 뒤 결과를 승인하지 않는다.

각 command/result는 artifact digest와 actual kernel trace를 남긴다. 책의 예시 출력은 실제 실행된 환경만 사용하고 미실행 숫자를 채우지 않는다.

**14장의 수치·시스템 인수 기준**

**표현과 연산 계약을 증명한다.**

FP32/TF32/BF16/FP16/FP8/FP4의 format·scale·rounding·accumulator가 operator별로 고정됐는가. GEMM/attention/norm/optimizer/collective의 forward/backward와 structural invariant가 FP32 oracle/error budget을 통과하는가.

**state와 복구 계약을 증명한다.**

GradScaler, master/moment, recipe, amax history/scale, stochastic RNG와 cache version이 UpdateID에 연결되는가. checkpoint round-trip, topology reshard, graph/cache rebuild와 CUDA 12/13 rollback에서 next updates가 검증됐는가.

**toolchain과 dispatch 계약을 증명한다.**

official CUDA exact-version evidence, framework/TE/CUTLASS/Triton source, build binary target, loaded library, actual kernel/algorithm과 SM guard가 같은 support cell을 가리키는가. fallback/compound migration과 `NotRun`을 분리했는가.

**성능과 운영 계약을 증명한다.**

cast/amax/metadata/kernel/collective/compile/checkpoint 비용과 quality tolerance를 포함한 공정 benchmark가 있는가. 관측성, failure injection, canary와 rollback이 first bad edge로 수렴하는가.

모든 필수 evidence가 닫힐 때만 정확한 GPU·toolkit·framework·recipe 범위에서 저정밀 학습을 승인한다. 이 판정은 CUDA 13이 CUDA 12보다 항상 빠르거나 FP8/FP4가 모든 모델에서 안전하다는 주장이 아니다. 어떤 조건에서 무엇을 직접 확인했고 다음 변경이 어느 cell을 무효화하는지 재현할 수 있다는 증명이다.

### 14.13.2 GEMM shape에서 Tensor Core dispatch 조건을 역산한다

**논리 shape와 physical leading dimension을 함께 본다.**

Transformer linear의 논리 `M=T, K=H, N=I`가 같아도 stored transpose, stride, batch, alignment와 padding이 kernel eligibility를 바꾼다. TP shard와 MoE expert별 token M은 작은/불균등 shape를 만든다. descriptor의 actual M/N/K, lda/ldb/ldc, transpose와 pointer alignment를 기록한다.

Tensor Core 지원 dtype/architecture라도 dimension/alignment/epilogue/workspace guard에서 SIMT 또는 fallback을 선택할 수 있다. exact 조건은 selected cuBLASLt/CUTLASS/Triton source와 runtime algorithm에서 확인한다. 일반적인 배수 규칙을 모든 library version에 고정하지 않는다.

production trace에서 operator role별 shape histogram, target kernel hit/fallback와 padding waste를 계산한다. 평균 shape 하나로 coverage를 예측하지 않는다. sequence packing, expert routing와 microbatch가 M 분포를 바꾼다.

**tail과 zero-size 경계를 fixture로 만든다.**

정렬 배수 바로 전/후, M=0/1, odd K/N, noncontiguous view와 misaligned slice를 넣는다. output/gradient parity와 actual dispatch, workspace/latency를 기록한다. padding path는 padded channel이 output/gradient에 새지 않아야 한다.

모델 config/parallel degree 변경으로 shape가 달라지면 performance support cell을 stale로 만든다. 같은 dtype flag의 과거 benchmark를 승계하지 않는다.

**split-K와 atomic reduction의 수치·재현성 비용을 계산한다**

**K 축 분할이 accumulator 합 순서를 바꾼다.**

split-K는 K dimension 부분 곱을 여러 CTA/partition에서 계산한 뒤 합친다. workspace reduction 또는 atomic accumulation을 쓸 수 있다. 병렬성은 늘지만 합 순서와 temporary bytes가 달라진다.

FP8/FP16 input에서도 partial accumulator가 FP32일 수 있으나 final reduction 순서 때문에 bitwise 결과가 달라질 수 있다. deterministic 요구가 있으면 algorithm 선택과 workspace constraint를 고정한다. nondeterminism을 layout 오류 허용으로 사용하지 않는다.

cancellation이 큰 dot, K 길이 sweep과 repeated run으로 max/RMS variance를 측정한다. forward뿐 아니라 weight-gradient GEMM은 K 역할이 token dimension이 될 수 있어 batch/sequence에 따라 민감하다.

**algorithm ID와 determinism을 checkpoint 밖 환경 state로 둔다.**

mathematical checkpoint가 algorithm ID를 반드시 저장할 필요는 없지만 exact reproducibility certificate에는 library/version/algorithm과 workspace가 필요하다. restart에서 heuristic이 다른 split-K를 고르면 metric tolerance 요구로 분류한다.

CUDA 12/13 library 비교는 same algorithm 가능 여부와 default selection을 분리한다. 빠른 atomic path가 numerical budget을 넘으면 승인하지 않는다.

**attention dropout과 low-precision RNG를 backward까지 보존한다**

**mask RNG와 stochastic rounding RNG를 별 stream으로 구분한다.**

attention dropout mask는 probability tensor/implicit fused path에서 생성되고 backward가 같은 mask를 재사용해야 한다. stochastic rounding RNG는 casts/updates에 쓰일 수 있다. seed가 같아도 counter allocation과 operator order가 달라 서로 간섭할 수 있다.

Flash/fused attention은 dropout mask를 저장하지 않고 seed/offset에서 재생성할 수 있다. activation checkpoint recompute와 CUDA Graph replay에서 logical forward 한 번당 RNG 소비가 맞는지 확인한다. amax 관측도 recompute에서 중복되지 않아야 한다.

fixture는 dropout 0과 nonzero, checkpoint on/off, eager/fused/compiled와 resume를 교차한다. forward output, dQ/dK/dV, RNG counter와 next random op를 비교한다. dropout 0 fixture만으로 RNG contract를 승인하지 않는다.

**분산 rank의 RNG ownership을 global token과 연결한다.**

TP/SP에서 동일 global attention element가 어느 rank에 있는지에 따라 mask stream을 설계한다. degree 변경 exact parity를 요구하는지 recipe에 명시한다. duplicate/shared tensor에 서로 다른 mask가 적용되지 않게 한다.

checkpoint는 framework/CUDA RNG와 library-specific state를 보존한다. graph executable은 portable RNG state가 아니다.

**FP4 block metadata를 checkpoint byte layout에서 검산한다**

**logical block과 packed offset의 양방향 map을 만든다.**

weight logical shape, storage transpose, quantization block shape, tail padding과 code packing 순서를 schema로 쓴다. 각 block의 local scale, optional global scale와 packed code byte range를 ParameterID에 연결한다.

checkpoint converter는 canonical weight→blocks→packed artifact와 reverse/dequantized validation을 제공한다. scale tensor shape가 맞아도 block order가 틀릴 수 있다. block마다 고유 scale/pattern을 넣은 role-coded fixture를 쓴다.

TP shard가 block boundary를 자르면 source/global layout에서 먼저 block을 정의할지 shard-local block을 정의할지 recipe가 결정한다. world-size 변경에서 scale block을 분할/합치는 방식이 필요하다. unsupported면 canonical weight 재양자화 migration이다.

**metadata corruption을 output first-difference와 연결한다.**

scale one-block swap, nibble order, tail mask, global scale와 format tag를 깨뜨린다. loader schema/checksum과 dequantized logical checksum, GEMM output 중 expected first detector를 확인한다.

serialized bytes 절감과 loaded workspace/cache/master bytes를 분리한다. 작은 tensor에서는 metadata/alignment가 이익을 줄인다.

**quantized training의 fake-quant와 native kernel을 분리한다**

**QAT graph의 forward approximation과 backward estimator를 적는다.**

fake quantization은 higher-precision tensor를 quantize-dequantize해 forward 오차를 모사하고 backward에 straight-through 또는 custom gradient를 사용할 수 있다. native FP8/FP4 Tensor Core kernel은 packed/quantized input을 실제 instruction으로 계산한다. 두 graph는 performance와 수치가 다르다.

observer/amax/scale update, fake quant op, STE mask/clipping과 master parameter를 source call graph로 잇는다. scale이 trainable parameter인지 derived controller state인지 구분한다. optimizer group/checkpoint owner가 달라진다.

fake-quant reference는 native kernel의 input dequantized equation oracle로 유용하지만 accumulator/epilogue/rounding과 hardware saturation을 모두 재현하지 않을 수 있다. 단계별 비교한다.

**calibration·QAT·native training phase를 state transition으로 둔다.**

observer warmup, scale freeze/learn, native enable와 export를 phase clock으로 기록한다. resume가 phase/counter를 복원해야 한다. observer reset은 exact resume가 아니다.

phase boundary 직전/후 checkpoint와 distribution shift fixture를 사용한다. quality 개선을 native format 자체 효과로 과장하지 않고 recipe 전체를 비교한다.

**CUDA stream·event 오류를 저정밀 cache version 문제로 찾는다**

**producer와 consumer의 stream 관계를 tensor state에 붙인다.**

cast/transpose/amax, GEMM, scale update, optimizer와 cached weight refresh가 서로 다른 streams에서 실행될 수 있다. event/wait가 빠지면 consumer가 stale scale/weight 또는 미완성 transpose를 읽는다. shape/dtype는 정상이고 오류가 간헐적일 수 있다.

TensorID와 ScaleVersion/ParameterVersion에 producer stream, ready event와 consumer wait를 기록한다. default-stream implicit ordering에 기대지 않는다. compiler/graph가 stream schedule을 바꾸는지도 본다.

buffer 재사용에는 last-consumer event가 필요하다. double buffering generation을 둔다. asynchronous checkpoint가 state를 복사하는 동안 optimizer가 갱신하지 않게 snapshot event를 사용한다.

**event 제거와 delay를 장애 주입한다.**

cast-ready, amax-reduce, weight-cache refresh와 collective-complete wait를 하나씩 제거하거나 producer를 지연한다. version assertion, sampled checksum 또는 numerical parity가 first gate에서 실패해야 한다.

illegal memory access 뒤 첫 failing kernel보다 앞선 async producer를 조사한다. `CUDA_LAUNCH_BLOCKING`은 진단 도구이지 production fix가 아니다.

**allocator fragmentation과 graph pool을 precision recipe에 포함한다**

**theoretical byte와 reserved/peak를 분리한다.**

FP8/FP4 payload가 작아져도 scale/history, transpose cache, workspace와 simultaneous higher-precision master가 있다. allocator size classes, fragmentation와 graph private pool 때문에 reserved bytes가 예상만큼 줄지 않을 수 있다.

allocation event를 tensor role/lifetime에 연결하고 allocated, active, reserved와 non-releasable을 구분한다. precision별 동일 batch에서 peak timeline을 비교한다. cold compile/autotune workspace도 별로 측정한다.

CUDA Graph capture pool은 안정 address를 위해 memory를 붙잡을 수 있다. 여러 shape/recipe graph가 각각 pool을 만들면 증가한다. graph cache eviction과 checkpoint/reload 정책을 기록한다.

**OOM 완화가 execution graph를 바꾸는지 검증한다.**

workspace limit, graph disable, batch/sequence 감소, activation checkpoint와 cache policy를 한 축씩 바꾼다. algorithm/fallback/dtype와 quality effect를 재검증한다. allocator config 숫자만 바꿔 근본 수명 중첩을 숨기지 않는다.

OOM 주입 뒤 partial optimizer/scale commit을 거부하고 parent UpdateID에서 재시도한다.

**multi-node 저정밀 수치 재현을 topology와 protocol에 묶는다**

**같은 NCCL dtype이라도 reduction tree가 달라질 수 있다.**

node/rank topology, channel/protocol와 collective algorithm이 floating reduction 순서를 바꿀 수 있다. exact bitwise 요구와 tolerance-based global oracle를 구분한다. world size 변경은 denominator/gradient batch도 바꿀 수 있어 같은 logical global batch를 구성한다.

FP8/FP4 compression이 있으면 quantize scale owner, payload, dequantize/reduce와 error feedback을 exact source에서 확인한다. scale metadata와 data collective order를 기록한다.

TP activation, DP gradient, EP expert payload와 optimizer state는 다른 process groups/dtypes를 쓸 수 있다. 하나의 “NCCL precision”으로 합치지 않는다. 15장의 group map을 소비한다.

**topology fixture와 rank failure를 결합한다.**

single node→two node, rank reorder와 zero-data rank에서 FP32 global oracle, output/gradient/update를 비교한다. one rank scale/history mismatch와 collective skip을 주입한다.

hang과 numerical drift를 별 terminal로 분류한다. partial collective 뒤 scale/optimizer state를 commit하지 않는다. resume checkpoint의 group/topology mapping을 검증한다.

**low-precision benchmark에서 energy·cost를 과장 없이 다룬다**

**측정 가능한 경계만 보고한다.**

tokens/s, GPU time, power telemetry와 wall-clock을 측정할 수 있지만 data-center total energy/탄소는 cooling, host/network와 전력 mix가 필요하다. GPU power 한 값으로 전체 환경 효과를 단정하지 않는다.

precision recipe가 batch를 키우거나 training token/step 수를 바꾸면 fixed token/quality에서 비교한다. 빠른 step이 convergence에 더 많은 step을 요구할 수 있다. failed run과 checkpoint/recovery 비용도 운영 cost에 넣는다.

power sampling interval, device set, idle baseline와 job boundary를 기록한다. compiler/autotune warmup과 steady phase를 나눈다. DVFS/thermal과 concurrent workload를 통제한다.

**비용 절감 주장을 quality·support 범위와 묶는다.**

수치/quality gate를 통과한 run만 동일 task 결과 비용으로 비교한다. fallback coverage와 unsupported shapes를 숨기지 않는다. 특정 GPU/CUDA/recipe 결과를 다른 architecture로 확대하지 않는다.

이 자료는 구매/운영 판단의 한 열이지 수학적 correctness 증거가 아니다. certificate에 측정 scope와 uncertainty를 붙인다.

**operator별 backward coverage를 release gate로 만든다**

**forward-only kernel과 training kernel을 구분한다.**

low-precision GEMM/attention/norm op가 forward를 지원해도 backward dgrad/wgrad, double backward, checkpoint recompute와 distributed gradient가 같은 format/backend를 지원하지 않을 수 있다. serving benchmark를 training support로 쓰지 않는다.

operator card에는 forward, dInput, dWeight, dBias, saved tensor/recompute, dtype/accumulator와 deterministic status를 둔다. unsupported backward가 eager BF16 fallback이면 coverage/performance를 별로 표시한다.

finite difference는 작은 smooth fixture에서 사용하고 production shape는 FP32 autograd reference와 비교한다. non-differentiable quantizer/STE는 정의된 surrogate gradient를 oracle로 한다.

**second-order·gradient penalty 경로를 별 지원으로 둔다.**

RL/regularization 또는 optimizer가 higher-order gradient를 요구하면 custom autograd가 지원하는지 확인한다. `once_differentiable`/detached path와 graph retention을 source/test로 본다. 일반 SFT backward PASS를 확대하지 않는다.

gradient checkpoint, compile와 graph 조합도 별 cell이다. upstream test coverage와 target integration test를 분리한다.

### 14.13.3 precision migration을 objective·optimizer·scheduler와 함께 승인한다

**dtype만 바뀌어도 effective training dynamics가 달라진다.**

gradient zero/saturation, GradScaler skip, FP8 scale lag와 fused rounding이 update trajectory를 바꾼다. 같은 lr/scheduler를 유지하는 것이 공정 control일 수 있지만 최적 recipe라고 단정할 수 없다. baseline과 candidate tuning 예산을 정한다.

loss numerator/denominator, gradient clipping, optimizer master/moment와 scheduler committed clock을 same UpdateID에서 비교한다. precision candidate에서 skip이 많으면 wall-clock뿐 아니라 seen token per update가 달라진다.

parent checkpoint에서 precision child를 만들 때 scaler/FP8 state initialization, master/model view와 compile graph를 명시한다. warmup/calibration phase를 추가하면 별 RecipeID다.

**공통 parameter/tensor의 control을 유지한다.**

FP32/BF16 fallback operator와 excluded layers의 output/gradient가 expected relation을 갖는지 본다. candidate가 tokenizer/data/model option도 바꾸지 않았음을 digest로 확인한다.

quality/throughput 비교는 fixed token, compute와 wall-clock 추정값을 구분한다. 13장의 schedule과 24장의 평가에 precision lineage를 넘긴다.

**독립 reviewer가 재생하는 low-precision certificate**

**첫 reviewer는 environment와 binary만 읽는다.**

GPU/SM, driver, toolkit, wheel/runtime libraries, TE/Triton/CUTLASS/extension source/build와 fatbin target을 재구성한다. support matrix cell과 official exact-version documents가 맞는지 확인한다.

**둘째 reviewer는 tensor/state ledger만 읽는다.**

selected GEMM, attention/norm와 optimizer shard의 storage/input/accumulator/output, scale/history/RNG/master/moment와 collective를 추론한다. checkpoint에서 next update state를 복원한다.

**두 reviewer는 GoldenStep에서 만난다.**

FP32→target eager→fused/FP8/FP4→compiled/graph→distributed ladder의 actual kernels, error와 state transition을 비교한다. 한 failure mutation을 재생해 expected first detector와 rollback parent를 확인한다.

certificate에는 unsupported/NotRun cell, tolerance와 invalidation keys가 있다. 문서 support, source build, runtime dispatch, numerical correctness와 tuned performance를 서로 다른 status로 유지한다.

**CUDA 12.x·13.x 비교를 독자의 조사 질문으로 압축한다**

첫 질문은 “설치한 CUDA가 몇인가”가 아니라 process가 어떤 driver/runtime/math library와 extension binary를 실제로 읽고 target GPU에서 어느 kernel을 실행하는가다. `nvcc`, wheel build tag와 loaded library를 같은 값으로 가정하지 않는다.

둘째 질문은 baseline과 candidate에서 무엇이 바뀌었는가다. driver-only, toolkit compiler, library, framework/TE/compiler와 GPU architecture를 직교 축으로 분리한다. 동시에 바뀌면 compound migration이다.

셋째 질문은 그 변화가 build, load, dispatch, numerics/state, performance와 checkpoint/rollback 중 어디에서 처음 나타나는가다. 한 층의 PASS를 다음 층으로 확대하지 않는다.

넷째 질문은 공식 문서의 exact version scope와 local evidence가 같은 조합을 가리키는가다. CUDA 13.0 문장으로 모든 13.x patch와 extension을 지원한다고 쓰지 않는다.

이 질문에 artifact로 답할 수 있을 때 버전 논쟁이 재현 가능한 engineering decision이 된다. 답할 수 없는 조합은 unsupported가 아니라 우선 `NotRun/Unknown`으로 남겨 다음 fixture를 정한다.

**14장을 다른 장과 왕복시키는 진단 경로**

7~10장의 embedding/norm/attention/MLP/model graph는 이 장에 TensorID와 operator shape를 준다. 이 장은 actual dtype, scale/history, kernel/accumulator와 error를 되돌린다. model output 이상은 source 함수와 precision edge를 양방향 비교한다.

11~13장의 optimizer/scheduler는 master/moment/scaler와 committed clock을 준다. 이 장의 overflow/stochastic cast/FP8 cache가 update를 바꾸면 같은 UpdateID로 돌아간다. 15~17장은 process group, collective, checkpoint/reshard와 복구를 소비한다.

21장의 multimodal outlier, 20장의 rollout/generation과 24~26장의 evaluation/monitoring은 precision sensitivity와 운영 지표를 연결한다. 28~30장의 GoldenRun은 precision ladder와 export/serving parity를 실행한다.

장간 링크는 “참조하라”로 끝나지 않는다. 입력 artifact, join key, expected invariant와 되돌아와 확인할 state를 적는다. 예를 들어 resume NaN은 17장 checkpoint root→이 장 history/scale→10장 first layer output→11장 optimizer delta로 왕복한다.

**변경 요청서를 option→state→effect로 작성한다**

변경 요청 첫 줄은 parent environment/RecipeID와 exact diff다. 예를 들어 CUDA 12.9.1→13.0 toolkit rebuild, TE v2.6.0 recipe E4M3/E5M2→NVFP4, BF16 attention→FP8 또는 graph enable을 구분한다.

둘째는 source/build/binary/dispatch 변화다. 변경된 documents, commits/tags, compiler flags, fatbin targets, loaded library와 expected kernel을 적는다. 셋째는 tensor state 변화로 dtype, scale/history/RNG/cache/master/checkpoint schema를 적는다.

넷째는 기대 효과와 비용이다. FLOPs/HBM/metadata/cast/collective/workspace, numerical error와 quality, compile/checkpoint/rollback을 포함한다. 다섯째는 precision ladder, failure mutation와 stop condition이다.

승인 후 actual first difference가 요청서와 맞는지 비교한다. 예상하지 못한 fallback/state가 있으면 새 evidence 없이 범위를 확대하지 않는다. 결과 파일은 parent를 덮지 않고 child certificate로 보존한다.

**14장의 독립 승인 판정**

독자는 임의 tensor 하나를 checkpoint storage에서 cast/scale과 kernel input, accumulator/epilogue, backward, collective, optimizer master/moment와 다음 checkpoint까지 추적할 수 있어야 한다. FP8/FP4이면 amax/history/block/RNG owner도 포함한다.

임의 CUDA environment 하나에서는 official exact-version support, framework/source guard, build binary target, process-loaded libraries, actual kernel/algorithm과 SM capability를 재구성할 수 있어야 한다. CUDA major 문자열만으로 빈칸을 채우지 않는다.

임의 장애 하나에서는 FP32→target→fused→quantized→compiled/graph→distributed ladder의 최초 실패 edge, 관련 state generation과 rollback parent를 찾을 수 있어야 한다. tolerance 확대나 generic fallback으로 의미 오류를 숨기지 않는다.

이 세 방향이 같은 RecipeID·TensorID·UpdateID에서 만나고 전수 failure/checkpoint/performance gate가 검증된 범위만 support matrix에 PASS로 남는다. 그때 저정밀 학습은 “bit를 줄여 빠르게 한다”는 문구가 아니라, 수치 표현·CUDA toolchain·분산 state·복구와 품질을 함께 설계하는 공학이 된다.

**정적 scale과 동적 scale을 실험에서 분리한다**

정적 scale은 calibration 또는 사전 결정값을 고정해 실행 중 같은 mapping을 사용한다. 동적/current/delayed scale은 tensor 분포를 관측해 step 또는 microbatch에 따라 바뀐다. 같은 FP8/FP4 format이라도 state, kernel overhead와 resume 의미가 다르다.

정적 scale 실험은 calibration corpus, percentile/max algorithm, block granularity와 freeze 시점을 보존한다. training data 분포가 calibration 범위를 벗어나면 saturation/zero가 증가할 수 있다. 동적 scale 실험은 amax/history/cursor와 update clock을 보존한다.

두 recipe를 비교할 때 format, operator coverage와 accumulation을 같게 두고 scale policy만 바꾼 child를 만든다. 정상 분포, outlier burst, curriculum 전환과 multimodal rare batch에서 error/scale/throughput을 본다. 동적 정책의 amax kernel/reduction 비용도 포함한다.

checkpoint fixture는 static scale 누락, dynamic history reset과 phase freeze flag rollback을 구분한다. loader가 default scale로 조용히 채우지 않는다. next several updates에서 output, saturation와 scale trajectory를 uninterrupted control과 비교한다.

**kernel launch 성공과 메모리 안전을 수치 정확성과 분리한다**

illegal memory access가 없고 output이 finite여도 out-of-bounds masked load, wrong stride와 stale buffer가 다른 유효 메모리를 읽어 silent corruption을 만들 수 있다. kernel contract에는 bounds, alignment, alias, stream lifetime와 numerical equation을 함께 명시한다.

guard page/sanitizer가 가능한 debug build, red-zone/canary buffer, tail poisoning와 role-coded input을 사용한다. output padding과 unused workspace가 예상대로 유지되는지 확인한다. optimized build의 성능 결과는 sanitizer run과 분리한다.

in-place/alias 지원 여부를 source schema에서 확인한다. residual fusion이나 optimizer list에서 input/output storage가 겹치면 read-before-write 순서가 중요하다. unsupported alias는 wrapper에서 거부한다.

CUDA 12/13 rebuild에서 compiler가 undefined behavior를 다르게 노출할 수 있다. candidate에서 처음 보인 오류를 toolkit 탓으로 끝내지 않고 source bounds/alias invariant를 확인한다. memory-safe PASS와 numerical PASS, performance PASS를 별 status로 둔다.

**negative control이 지원 선언의 범위를 지킨다**

release 직전에는 FP8 history 한 slot swap, FP4 block scale permutation, GradScaler tracker rollback, stale quantized weight cache, wrong SM binary, Triton cache collision와 collective dtype mismatch를 다시 주입한다. 모든 mutation은 정상 경로와 동일 shape를 유지해 단순 shape 검사만 통과하도록 만든다.

expected first detector는 schema/version, binary guard, scale identity, numerical ladder, collective oracle와 commit closure 중 하나다. detector가 최종 loss spike까지 기다리면 관측 위치를 앞당긴다. mutation이 아무 효과가 없다면 해당 branch가 fixture에서 실행됐는지 trace한다.

독립 reviewer는 mutation ID 하나를 골라 official/source claim, affected TensorID/state, kernel event와 checkpoint parent까지 왕복한다. 수정 후 정상 fixture와 주변 dtype/backend cell을 다시 실행한다. 단순히 mutation test를 삭제해 green 상태를 만들지 않는다.

이 negative control가 계속 민감할 때 support matrix의 PASS는 실제 의미를 유지한다. 새 CUDA patch, GPU, Transformer Engine, compiler 또는 recipe가 들어오면 동일 mutation suite를 child certificate에서 재실행한다. 확인하지 않은 조합은 성공을 기대하더라도 `NotRun`으로 남긴다.

**저정밀 학습을 시작하기 전의 열두 문장**

첫째, parameter storage dtype과 GEMM input dtype은 다를 수 있다. 둘째, GEMM input dtype과 accumulator dtype도 다를 수 있다. 셋째, output dtype이 같아도 epilogue fusion이 rounding 위치를 바꿀 수 있다. 넷째, autocast는 전역 cast가 아니라 operator 정책이다.

다섯째, FP16 loss scaling은 backward underflow를 완화하지만 forward overflow를 고치지 않는다. 여섯째, BF16은 범위가 넓어도 reduction과 작은 update 오차가 남는다. 일곱째, FP8/FP4 tensor는 scale·history·block metadata와 떨어져서는 의미가 없다. 여덟째, stochastic rounding은 RNG와 checkpoint state를 요구한다.

아홉째, CUDA toolkit, framework wheel runtime, driver, loaded library와 extension binary는 서로 다른 artifact다. 열째, GPU가 어떤 format을 지원해도 model shape가 실제 kernel로 dispatch된다는 보장은 없다. 열한째, 빠른 forward kernel은 backward·checkpoint·분산 학습 지원을 자동 보장하지 않는다. 열두째, finite loss는 올바른 scale, layout와 commit을 증명하지 않는다.

독자는 이 열두 문장을 환경 manifest, dtype 원장, operator card, GoldenStep과 failure suite의 열로 바꾼다. 어느 문장이 직접 증거를 갖지 못하면 그 지점이 다음 조사 대상이다. 반대로 모든 열이 닫히면 특정 CUDA·GPU·recipe에서 왜 그 dtype을 선택했고 어떤 실패를 어디에서 감지하며 어떻게 복구하는지 설명할 수 있다.

이 준비표는 precision 선택을 보수적으로 만들기 위한 것이 아니다. 불확실성을 앞 단계에서 좁혀 실제로 안전한 최적화를 더 빠르게 승인하기 위한 것이다. 성능 수치는 이 계약 위에서만 의미가 있고, 새 버전에서는 영향을 받은 열만 정직하게 재검증한다.

## 14.14 MXFP8을 숫자 형식이 아니라 블록 계약으로 읽는다

**서른두 값이 하나의 자를 공유한다**

MXFP8의 핵심은 “FP8을 쓴다”가 아니라 **연속한 32개 값마다 서로 다른 2의 거듭제곱 자를 댄다**는 데 있다. 고정된 블록 \(B\)에 대한 개념적 양자화는 다음처럼 쓸 수 있다.

\[
a_B=\max_{i\in B}|x_i|,\quad e_B=\operatorname{clip}\!\left(\left\lceil\log_2\frac{a_B}{q_{\max}}\right\rceil,e_{\min},e_{\max}\right),\quad s_B=2^{e_B},
\]

\[
q_i=\operatorname{round}_{F8}(x_i/s_B),\qquad \widehat{x}_i=s_Bq_i.
\]

여기서 \(q_i\)는 E4M3 또는 E5M2 코드이고, \(s_B\)는 E8M0으로 표현하는 공유 scale이다. 구현에 따라 `scale` 대신 역수인 `scale_inv`를 저장하므로 변수 이름만 보고 곱셈 방향을 단정하면 안 된다. 중요한 불변량은 코드와 그 코드를 해석할 블록 scale이 같은 분할, 순서, layout으로 이동한다는 것이다.

기하학적으로 tensor 전체를 축소하는 per-tensor FP8은 모든 좌표에 하나의 격자를 씌운다. MXFP8은 32차원 조각마다 격자 간격을 바꾼다. outlier가 있는 조각만 거친 격자를 쓰므로 다른 조각의 작은 값까지 함께 뭉개지지 않는다. 대신 scale metadata와 블록 경계가 새 상태가 되고, tensor를 전치하거나 shard하면 어느 값들이 한 자를 공유하는지가 달라진다. 따라서 MXFP8은 E4M3의 변종이 아니라 **데이터 코드, 공유 지수, 블록 축, padding, swizzle을 묶은 저장 계약**이다.

E4M3과 E5M2의 선택도 목적이 다르다. 보존된 Transformer Engine 소스의 `Format`은 E4M3 최대 크기를 448, E5M2를 57,344로 두며 `HYBRID`는 forward에 E4M3, backward에 E5M2를 배정한다. E4M3은 가수 비트가 하나 더 있어 같은 scale 안에서 더 촘촘하지만 범위가 좁다. E5M2는 gradient spike를 담기 쉬운 대신 간격이 더 거칠다. UE8M0 또는 E8M0 공유 scale은 가수가 없어 정확히 2의 거듭제곱만 나타낸다. 이 제약은 임의 실수 scale보다 오차가 작다는 뜻이 아니다. 곱셈과 metadata 해석을 단순화하고 표준화하는 대가로 scale 자체의 반올림 오차를 받아들이는 선택이다.

amax는 장기 상태가 아니다. `MXFP8Quantizer`는 현재 입력을 32개씩 보고 scale을 즉시 만들며 `calibrate()`도 상태를 갱신하지 않는다. delayed scaling의 `amax_history_len`, `amax_compute_algo`, 분산 `reduce_amax`를 MXFP8에 그대로 투영하면 틀린 설명이 된다. `MXFP8BlockScalingRecipeState`가 “state가 필요 없다”고 말하는 뜻도 checkpoint에 아무것도 없다는 뜻은 아니다. 과거 amax window가 없다는 뜻이고, 양자화한 parameter를 저장한다면 FP8 코드와 scale metadata는 여전히 함께 보존해야 한다.

**옵션 하나가 consumer와 상태를 어떻게 바꾸는가**

`MXFP8BlockScaling(fp8_format=..., backward_override=..., enable_2d_quantization=...)`을 다음 효과 사슬로 읽자.

| 입력 옵션 | 최초 consumer | 바뀌는 상태·layout | 기대 효과 | 대표 실패 |
|---|---|---|---|---|
| `fp8_format=E4M3` | recipe state의 dtype 선택 | forward와 backward quantizer 모두 E4M3 | 더 촘촘한 유효숫자 | 큰 gradient saturation |
| `fp8_format=HYBRID` | `get_fp8_te_dtype(recipe, mode)` | forward E4M3, backward E5M2 | backward 범위 확대 | pass별 dtype을 로그가 숨김 |
| 순수 `E5M2` | recipe `__post_init__` | 상태 생성 전 assertion | 미지원 조합 조기 차단 | “FP8 GPU”라서 통과할 것이라 오판 |
| `enable_2d_quantization=True` | `make_quantizers()` | forward linear/grouped-linear의 **weight role만** 2D quantizer | weight 두 축의 outlier 국소화 | activation·backward에도 적용됐다고 오해 |
| `backward_override=high_precision` | autograd 저장 경로 | 원래 고정밀 operand의 수명 연장 | backward 기준 정확도 강화 | activation peak 상승 |
| `backward_override=dequantized` | 저장 tensor 복원 경로 | MXFP8 operand를 compute dtype으로 복원 | 고정밀 원본 저장보다 메모리 절충 | 이미 난 양자화 오차는 복구되지 않음 |
| `rowwise/columnwise` usage | quantizer allocation·cast | data와 `scale_inv` 사본의 방향 | fprop/dgrad/wgrad operand 준비 | 전치를 단순 view로 가정해 이중 양자화 |

마지막 행이 특히 중요하다. rowwise 양자화 tensor의 전치는 columnwise로 원본을 직접 양자화한 결과와 일반적으로 같지 않다. 전치 뒤의 32개 묶음이 원래 묶음과 달라지기 때문이다. Transformer Engine은 forward와 backward에서 둘 다 필요할 때 고정밀 입력에서 두 표현을 각각 만든다. `transpose(fp8(x))`로 두 번째 표현을 대신 만들면 double quantization뿐 아니라 블록 구성 자체가 바뀐다.

shape도 API 부속 조건이 아니다. 현재 `MXFP8Quantizer.is_quantizable()`은 입력이 2차원 이상이고 마지막 축과 그 앞축들의 곱이 모두 32로 나누어지는지 검사한다. scale tensor는 compact 논리 shape 그대로 저장되지 않을 수 있다. rowwise scale은 `[M, K/32]`를 바탕으로 행을 128의 배수, 열을 4의 배수로 padding하고, columnwise scale은 `[M/32, K]`를 바탕으로 각각 4와 128의 배수로 맞춘다. GEMM용 swizzle까지 적용되면 같은 수치 metadata도 byte layout은 다르다. 따라서 저장 크기를 `numel/32`로만 추정하거나 scale tensor를 평범한 `[M,K/32]` 배열로 잘라 붙이는 checkpoint 코드는 경계 shape에서 깨진다.

**forward·backward·optimizer의 경계를 따로 그린다**

학습 step에서 MXFP8이 모든 상태를 8비트로 만드는 것은 아니다. 가장 안전한 mental model은 네 층이다.

1. **parameter의 권위 상태**: BF16/FP16/FP32 master parameter 또는 명시적으로 지원되는 primary FP8 parameter다.
2. **GEMM operand 표현**: activation과 weight를 호출에 맞춰 MXFP8 data+scale로 만든다. 필요하면 rowwise와 columnwise 표현을 함께 유지한다.
3. **누적과 출력**: Tensor Core 입력이 FP8이어도 accumulator와 GEMM output dtype은 별도 계약이다. “입력 8비트”를 “합도 8비트”로 읽지 않는다.
4. **optimizer 상태**: Adam moment와 master update dtype은 optimizer 구현이 결정한다. MXFP8 autocast만 바꿨다고 `m`, `v`, parameter update가 자동으로 MXFP8이 되지 않는다.

forward linear에서는 \(X\)와 \(W\)가 각자 블록 양자화되어 GEMM에 들어가고, backward에서는 \(dY\)와 저장 operand가 dgrad와 wgrad의 서로 다른 전치 방향을 요구한다. `backward_override`는 gradient 결과 dtype 하나를 바꾸는 스위치가 아니다. backward가 참조할 operand를 어느 표현으로 저장하는지를 바꾸고 activation lifetime, peak memory, 재양자화 오차를 함께 움직인다. `high_precision`과 `dequantized`를 같은 “고정밀 backward”로 묶으면 전자는 양자화 전 원본, 후자는 양자화 후 복원본이라는 차이를 잃는다.

Blackwell 지원도 제품명으로 승인해서는 안 된다. 보존된 Transformer Engine 리비전의 capability gate는 compute capability 10.x에서 MXFP8을 허용하지만 12.0 이상에는 아직 모든 GEMM layout이 지원되지 않는다는 이유로 거부한다. 기본 recipe도 이 판정에 따라 MXFP8, current scaling, delayed scaling 사이에서 달라진다. 실행 manifest에는 GPU 이름만 아니라 compute capability, CUDA toolkit/runtime, cuBLASLt, PyTorch build, Transformer Engine commit과 실제 선택 recipe를 기록해야 한다. “Blackwell이면 MXFP8”은 코드가 가진 음의 capability branch를 지워 버린 문장이다.

TorchAO의 일반 Float8 training과도 구분해야 한다. TorchAO의 `Float8TrainingTensor`와 rowwise/tensorwise scaling은 유용한 비교 대상이지만, 그것만으로 OCP MX의 32-value E8M0 block layout을 구현했다고 말할 수 없다. 같은 E4M3 payload라도 scale granularity와 layout, kernel ABI가 다르면 교환 가능한 artifact가 아니다. 통합 테스트는 “둘 다 FP8”이 아니라 producer가 만든 data·scale·layout을 consumer가 그대로 해석하는지 검사해야 한다.

**실패 fixture가 설명을 완성한다**

MXFP8 검증은 평균 loss가 비슷하다는 한 줄로 끝낼 수 없다.

- **블록 경계 oracle**: 31, 32, 33번째 원소에 크기가 다른 outlier를 놓아 scale 공유 범위를 확인한다.
- **포화·underflow oracle**: 한 블록에 \(q_{\max}s\) 바로 아래·위 값과 최소 subnormal 주변 값을 놓고 code, saturation count, zero count를 비교한다.
- **방향 oracle**: 같은 고정밀 행렬에서 rowwise 결과의 전치와 직접 columnwise 양자화를 비교해 둘이 같지 않은 반례를 고정한다.
- **shape oracle**: `M`, `K`의 32 정렬과 scale padding의 128×4 또는 4×128 경계를 교차한다. skip과 실패를 구별한다.
- **resume oracle**: data만, scale만, swizzle flag만 손상한 checkpoint를 각각 읽혀 fail-fast 여부를 확인한다. 조용한 dequantization이 가장 위험하다.
- **backward oracle**: `None`, `high_precision`, `dequantized`에서 gradient reference와 peak activation memory를 함께 기록한다.
- **capability oracle**: 지원 GPU가 없는 CI에서는 `is_mxfp8_available(return_reason=True)`의 거부 이유를 assertion한다. skip도 지원 경계의 실행 가능한 사양이다.

운영 지표도 block scale의 결과를 겨냥해야 한다. tensor별 amax 하나보다 block-scale exponent histogram, saturated-code 비율, zero/underflow 비율, rowwise·columnwise scale 분포, fallback GEMM 수, requantization 수를 layer와 operand role별로 본다. loss spike가 보이면 어느 layer의 어느 operand에서 scale 지수가 상한에 붙었는지, 그 직전에 shape fallback이나 recipe 변경이 있었는지 역추적한다. 그래야 “MXFP8이 불안정하다” 대신 “wgrad E4M3 block의 0.07%가 포화됐고 `HYBRID` 전환 뒤 사라졌다”처럼 조치 가능한 진단이 된다.

마지막 승인 질문은 간단하다. **누가 scale을 만들었고, 어느 32개가 공유하며, 어떤 방향과 padding으로 저장했고, 어느 GEMM이 소비하며, backward와 optimizer에는 무엇이 남는가?** 이 다섯 질문에 소스 좌표와 failure fixture로 답하지 못하면 MXFP8은 아직 설정된 것이 아니라 이름만 적힌 것이다.

## 14.15 FP8 cast를 값과 layout의 두 oracle로 분해한다

DeepGEMM의 `per_token_cast_to_fp8`을 읽을 때 흔히 하는 실수는 이 함수가 테스트에서 호출된다는 사실만 보고 “FP8 변환 정확도가 검증됐다”고 쓰는 것이다. 고정 리비전 `559d79fb6994a58b8a15b4b93bf13ccc16edf247`의 `deep_gemm/utils/math.py:26-49`에서 함수는 입력의 마지막 축을 `gran_k` 묶음으로 보고 블록별 절댓값 최대치를 만든다. 이어 scale을 계산하고 FP8 payload와 scale tensor를 돌려준다. 이때 반환값은 하나의 배열이 아니라 서로 다른 소비자를 가진 두 상태다. payload는 GEMM operand이고 scale은 kernel이 payload를 복원하는 ABI다.

핵심 부분을 의미가 드러나는 만큼만 줄이면 다음과 같다.

```python
grouped = x.view(*x.shape[:-1], -1, gran_k)
amax = grouped.abs().float().amax(dim=-1)
scale = amax / fp8_max
fp8 = (grouped / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
return fp8.view_as(x), scale
```

실제 구현의 UE8M0·padding 분기는 이보다 복잡하지만, 디버깅 좌표는 선명하다. 입력이 `[M,K]`이고 `gran_k=128`이면 논리 scale shape는 `[M,K/128]`이다. 여기서 `K`의 나눗셈 조건, zero block의 scale, outlier가 있는 블록의 포화, scale 표현 방식이 payload 값을 정한다. 그 다음 TMA layout 변환은 같은 scale 값을 kernel이 요구하는 stride와 padding으로 재배치한다. 따라서 첫 oracle은 `x→amax→scale→fp8→dequantized x`의 수치 oracle이고, 둘째는 `scale logical layout→TMA layout`의 값·shape·stride oracle이다.

공개 `tests/test_layout.py:45-80`은 둘째만 직접 닫는다. 테스트는 cast가 만든 `fp32_sf`를 받아 TMA-aligned 구현과 PyTorch reference를 비교하고 값, shape, stride가 같은지 검사한다. 이 범위를 축약하면 다음과 같다.

```python
x, fp32_sf = per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k)
packed_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf)
ref_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(fp32_sf)
assert torch.equal(packed_sf, ref_sf)
assert packed_sf.shape == ref_sf.shape
assert packed_sf.stride() == ref_sf.stride()
```

이 테스트가 실패하면 최초 불일치는 세 층으로 좁힌다. 값이 먼저 다르면 UE8M0 packing이나 transpose index를 본다. 값은 같고 shape가 다르면 padding과 group cardinality를 본다. 둘이 같고 stride만 다르면 contiguous copy, view, TMA consumer의 기대 stride를 확인한다. 반대로 이 테스트가 통과해도 FP8 payload의 rounding·saturation·NaN 처리가 맞다는 결론은 나오지 않는다. 수치 oracle이 별도로 필요하다.

변형 실험은 한 축씩만 흔든다. `gran_k`를 32·64·128로 바꾸고 outlier를 블록 경계의 앞·뒤에 놓는다. `use_ue8m0`만 바꾸어 scale 표현과 packing을 분리한다. 모든 값이 0인 block, `fp8_max` 바로 아래·위, `NaN`과 `Inf`가 섞인 block을 넣고 정책을 기록한다. 수치 oracle은 독립 FP32 reference로 `amax`와 scale을 계산하고, payload를 dequantize한 뒤 원소별 오차·포화 수·zero 수를 비교해야 한다. layout oracle의 통과 여부와 같은 표에 넣되 한 열로 합치지 않는다.

후속 디깅도 이 분리를 따른다. GEMM 결과가 틀렸는데 scale layout test부터 깨졌다면 kernel 전에 생산자-소비자 ABI를 고친다. layout은 맞고 dequantized operand가 틀리면 cast의 scale·rounding·특수값 경계를 본다. 두 oracle이 맞는데 GEMM만 다르면 selected kernel, accumulator dtype, epilogue, stream dependency로 내려간다. 이 순서가 없으면 layout 버그를 “FP8 정밀도 한계”로, rounding 차이를 “TMA stride 버그”로 잘못 분류한다.

이 질문에 답하면 dtype 표는 실행 계약이 된다. 다음 15장에서는 같은 `TensorID`에 owner·replica·global slice를 붙여, 저정밀 payload와 scale metadata가 DP·TP·FSDP·ZeRO 경계에서 누구에게 이동하고 어디서 다시 합쳐지는지 추적한다.

## 14.16 숫자 형식에서 skipped optimizer step까지 하나의 dtype graph로 읽는다

FP16은 부호 1·지수 5·가수 10비트, BF16은 1·8·7비트다. BF16은 FP32와 비슷한 지수 범위를 얻는 대신 한 binade 안의 간격이 더 크다. TF32는 저장 dtype이 아니라 지원 GPU에서 FP32 행렬곱 입력의 유효 가수 정밀도를 낮추고 FP32 누산을 사용하는 실행 모드다. FP8 E4M3은 더 촘촘한 유효숫자와 좁은 범위, E5M2는 더 넓은 범위와 거친 간격을 택한다. 형식 이름만으로 subnormal, NaN/Inf encoding과 saturation 규칙을 추정하지 말고 실제 dtype·kernel 계약을 확인한다.

Transformer Engine의 delayed scaling은 과거 amax history와 축약 함수를 state로 가진다. current scaling은 현재 tensor의 amax에서 곧바로 scale을 계산한다. 같은 E4M3 payload라도 scale의 시간 좌표가 다르므로 같은 알고리즘이 아니다. MXFP8은 작은 block마다 E8M0 scale을 두며 rowwise와 columnwise가 다른 layout을 요구한다. NVFP4는 더 작은 payload 위에 block scale과 tensor-level scale을 겹치는 계층형 계약을 쓴다.

| 경로 | 계산/저장 상태 | checkpoint에서 빠지면 생기는 일 |
|---|---|---|
| FP16 AMP | FP32 master weight, GradScaler scale·growth tracker | 재시작 직후 overflow/skip 궤적 변화 |
| BF16 | BF16 activation/gradient와 대개 FP32 optimizer state | master/optimizer cast 차이로 delta 변화 |
| FP8 delayed | payload, scale·inverse, amax history·cursor | 이전 window와 다른 scale 선택 |
| FP8 current | 현재 amax와 scale | accumulation/reduction 시점 차이 |
| MXFP8 | block payload와 row/column E8M0 scale layout | transpose GEMM의 scale 소유권 오류 |
| NVFP4 | FP4 payload, block/tensor scale, RHT·rounding state | outlier와 backward 분포 변화 |

GradScaler는 scaled loss를 backward한 뒤 gradient를 unscale하고 inf/NaN을 모은다. overflow가 있으면 optimizer step을 실행하지 않는다. 이때 parameter는 그대로지만 scale과 growth tracker는 전진할 수 있다. gradient accumulation 도중 overflow가 난 microbatch, distributed found-inf reduction, scheduler가 optimizer skip에도 전진하는지까지 하나의 update transaction으로 검사해야 한다.

경계 fixture는 0, 최소 subnormal 주변, 최소 normal, 1 ULP의 양옆, tie, 최대 finite와 overflow, 한 block의 outlier를 포함한다. payload bit pattern, scale·inverse, saturation bitmap, forward 출력, Q/K/V나 linear weight gradient, FP32 master-weight delta를 비교한다. 중간 microbatch에 inf를 넣어 모든 rank가 같은 step을 건너뛰는지 보고, checkpoint roundtrip 뒤 amax history와 scaler state가 같은지 확인한다.

정적 지원표는 toolkit과 architecture를 분리한다. CUDA 12.x와 13.x라는 버전만으로 FP8·MXFP8·NVFP4 training 가능 여부를 판정할 수 없다. compiler가 dtype intrinsic을 아는지, 대상 compute capability에 해당 Tensor Core 명령이 있는지, driver와 framework build가 그 toolkit을 지원하는지, Transformer Engine/torchao가 해당 recipe와 backward kernel을 노출하는지를 모두 맞춘다. 공식 문서와 고정 소스에서 확인한 지원 조건은 compatibility 후보일 뿐, 특정 GPU의 정확도와 처리량을 입증하지 않는다. 이 장의 하드웨어 수치는 실제 실행 전까지 `NOT_VERIFIED`다.

## 14.17 fused kernel은 저장한 값과 다시 계산한 값까지 같아야 한다

fused cross entropy는 최대 logits를 뺀 logsumexp, target logit, ignore mask와 reduction denominator를 한 kernel에서 처리한다. 극단 logits `[10000,0,-10000]`, vocabulary tail의 odd shape, non-contiguous row와 모든 label이 ignore인 batch를 넣는다. 출력 loss뿐 아니라 logits gradient와 zero-valid 처리 정책을 eager reference와 비교한다.

FlashAttention의 online softmax는 row max와 정규화 합을 tile마다 갱신한다. backward는 저장된 LSE와 output, dropout RNG state를 사용하거나 일부 값을 재계산한다. causal·local·padding mask가 같은 token 집합을 선택하는지, dropout 0과 고정 seed에서 Q/K/V gradient가 reference와 맞는지 본다. all-masked row는 유한성만 확인하지 말고 output과 gradient의 명시된 값을 검사한다.

RMSNorm·MLP fusion에서는 accumulator dtype과 saved tensor가 핵심이다. RMSNorm backward는 `rstd`, 입력과 weight를 통해 입력·gamma gradient를 만들며 residual/dropout을 합치면 RNG mask와 residual dtype도 계약이 된다. fused MLP는 첫 GEMM, activation, 두 번째 GEMM을 잇는다. activation 입력을 저장하는지 재계산하는지에 따라 checkpoint와 메모리 절약 경로의 오차·RNG 요구가 달라진다.

fused Adam은 parameter, gradient, 1·2차 moment, step, master weight를 한 update에서 바꾼다. 여러 block이 같은 accumulation 대상에 기여할 때 atomic ordering은 부동소수점 덧셈 순서를 바꿀 수 있다. 동일 seed 반복, 여러 stream, odd shard와 accumulation 횟수를 바꾸어 deterministic 모드의 bitwise 요구와 일반 모드의 tolerance 요구를 분리한다. race-condition test가 없으면 한 번의 parity 성공을 ordering 안전성 증거로 쓰지 않는다.

release fixture는 contiguous와 transposed view, 정렬되지 않은 차원, extreme logits, zero-valid, fully masked, dropout seed 두 개를 조합한다. 각 연산은 shape·stride·dtype·accumulator, mask와 reduction, saved/recomputed tensor, forward와 모든 입력/parameter gradient를 내보낸다. eager와의 첫 차이를 찾은 다음에만 성능을 잰다. 소스의 tile 크기나 fused 호출 존재는 occupancy, register pressure, 처리량을 증명하지 않으므로 GPU 성능 주장은 실행 전까지 `NOT_VERIFIED`다.

## NF4 저장 상태와 dequant 계산을 분리한다

NF4는 sign·exponent·mantissa를 가진 IEEE 4-bit float가 아니라 정규분포 quantile을 바탕으로 정한 16-value lookup codebook이다. bitsandbytes 경로는 block별 absmax로 값을 정규화하고 두 4-bit code를 storage byte 하나에 담는다. 복원에는 packed data만 아니라 shape, 원 dtype, blocksize, quant type과 absmax가 필요하다. double quant 또는 compressed statistics는 absmax를 다시 양자화하므로 nested absmax·offset과 두 번째 state까지 checkpoint schema의 일부다.

직접 upstream 테스트는 여러 dtype·blocksize에서 FP4/NF4 quantize→dequantize 오차 bound와 compressed-statistics 경로를 검사한다. 이것은 model accuracy나 QLoRA gradient를 증명하지 않는다. edge fixture에는 all-zero block, code threshold tie, saturation 극값, NaN/Inf와 홀수 element 수를 넣어 scale-zero 처리, nibble order, code index와 dequant dtype을 golden reference에 비교한다.

QAT의 fake quant는 observer가 min/max에서 scale·zero-point를 만들고 clamp·round를 흉내 내되 backward에서는 STE를 사용한다. per-tensor와 per-channel axis, observer freeze 전후를 분리한다. QLoRA는 frozen packed base를 compute dtype으로 dequant해 matmul하고 gradient가 activation과 LoRA A/B로 흐르는지, packed base에는 optimizer gradient가 생기지 않는지 autograd graph로 확인한다. 둘 다 “4-bit 학습”이라 불러도 학습되는 상태가 다르다.

adapter merge는 dequantized base에 ΔW를 더한 뒤 다시 양자화할 수 있어 unmerged forward와 정확히 같지 않다. merge→requant를 반복하며 drift를 측정하고, bitsandbytes·torchao·safetensors·Transformers export 사이 quant type, blocksize, nibble order, absmax/nested metadata, compute dtype과 tied-weight identity를 manifest로 검증한다. 모르는 schema를 추정해 읽지 않는다. accuracy·peak memory·kernel throughput은 별 RuntimeUnverified 실험이다.

## 14.18 CPU의 한 batch가 HBM의 한 cache line이 되기까지

**대역폭 하나가 아니라 소유권이 바뀌는 경로로 읽는다**

학습이 `data wait`에 막혔다고 해서 DataLoader worker 수부터 늘리면 안 된다. 한 batch는 대개 storage와 OS page cache를 거쳐 pageable user buffer에 놓이고, 필요하면 page-locked host buffer로 복사된 뒤 DMA로 PCIe root complex 또는 CPU–GPU 연결을 지나 HBM에 도착한다. kernel은 그 데이터를 L2, SM의 L1/shared memory와 register로 가져간다. 이 경로에는 하나의 “메모리 속도”가 없다. 각 경계마다 생산자, 소비자, 주소 공간, 전송 엔진과 완료 조건이 달라진다.

이를 latency 합으로 쓰면 다음과 같다.

\[
T_{input}=T_{read}+T_{decode}+T_{pageable\rightarrow pinned}
+T_{queue}+T_{H2D}+T_{consumer\ wait}.
\]

겹침이 전혀 없을 때는 합이 맞지만, 실제 step의 노출 시간은 대략 `max(compute, overlapped transfer)`와 겹치지 못한 꼬리의 합이다. 따라서 H2D 시간이 8 ms라는 수치만으로 step이 8 ms 느려졌다고 말할 수 없다. copy stream이 compute와 겹쳤는지, 다음 batch가 소비 시점 전에 도착했는지, producer buffer가 안전하게 살아 있었는지를 함께 봐야 한다.

PyTorch의 고정 소스에서 `DataLoader(pin_memory=True)`는 GPU copy를 실행하지 않는다. `_pin_memory_loop`가 worker output queue를 읽고 `pin_memory()`를 재귀 호출한다. tensor뿐 아니라 mapping, named tuple, 일반 sequence 안의 tensor도 새 page-locked host storage로 옮긴 뒤 main process queue에 넣는다. 즉 옵션이 바꾸는 상태는 `CPU tensor → GPU tensor`가 아니라 `pageable host object → DMA 가능한 host staging object`다. 중첩 dictionary가 pinned되는 직접 시험과 `pin_memory_device`만 쓰면 pinned되지 않는 음성 시험이 이 경계를 고정한다.

여기서 첫 번째 디깅 질문은 “pinning이 켜졌는가?”가 아니라 다음 네 가지다.

1. collate 뒤 어느 객체가 실제로 pinned됐는가.
2. worker output과 pin thread 사이 queue가 어디서 찼는가.
3. pinned allocation·복사가 소비자보다 먼저 끝났는가.
4. batch tensor가 `.to(device, non_blocking=True)`로 전달되고 별도 stream dependency가 닫혔는가.

`num_workers`, `prefetch_factor`, batch size와 pinning을 한꺼번에 바꾸면 어느 queue가 병목이었는지 알 수 없다. fixture는 같은 sample ID·shape를 유지한 채 worker 0/1/N, pinning on/off, blocking/non-blocking copy를 한 축씩 교차한다. trace에는 read, decode, collate, pin, enqueue, H2D enqueue, H2D complete와 first consumer kernel을 별도 range로 남긴다.

**pinned memory는 빠른 RAM이 아니라 DMA 수명 계약이다**

운영체제는 pageable memory의 물리 page를 옮기거나 swap할 수 있다. 장치가 host memory를 DMA로 읽는 동안 물리 주소가 바뀌면 안 되므로 CUDA 전송 경로는 page-locked memory를 사용하거나 내부 staging을 거친다. pinning의 이득은 “RAM 자체가 빨라진다”가 아니라 DMA가 안정된 물리 page를 비동기로 참조할 수 있다는 데 있다. 대가는 memlock, pin/unpin 비용, host reclaim 압력과 추가 CPU copy다.

PyTorch의 `CUDACachingHostAllocator`는 `cudaHostAlloc` 또는 `cudaHostRegister` 비용을 매 batch마다 내지 않도록 block을 캐시한다. 이 때문에 Python tensor 참조가 사라진 시각과 block을 다시 빌려도 되는 시각은 다르다. `Copy.cu`의 non-blocking CPU↔GPU branch는 `cudaMemcpyAsync`를 발행한 뒤 storage context와 data pointer를 key로 host allocator에 stream event를 기록한다. slice처럼 원 allocation의 base pointer와 다른 view도 이 두 key로 원 소유권을 찾는다. event가 끝나기 전에 allocator가 같은 block을 다음 batch에 재사용하면 DMA가 읽는 동안 내용이 덮인다.

이를 작은 상태 기계로 쓰면 이해가 쉽다.

```text
Pageable --pin/copy--> PinnedAvailable --enqueue H2D--> InFlight(stream,event)
    ^                                                        |
    +---------------- cache/reuse <---- event complete ------+
```

`non_blocking=False`는 같은 상태를 더 빨리 만드는 옵션이 아니라 `memcpy_and_sync`로 호출자가 완료를 기다리는 다른 전이이다. `True`에서는 host 호출이 돌아왔다는 사실과 device consumer가 읽어도 된다는 사실을 구분해야 한다. 별도 copy stream을 쓰면 consumer compute stream이 event를 기다려야 하고, host buffer는 copy event가 끝날 때까지 재사용할 수 없다.

pinned pool 크기도 throughput 숫자 하나로 결정하지 않는다. batch당 pinned byte를 (B_p), 동시에 살아 있는 prefetch·in-flight batch 수를 (Q), allocator rounding과 fragmentation 계수를 \(\rho\ge1\)라 하면 최소 working set은 대략

\[
M_{pin}\approx B_pQ\rho
\]

이다. 여기에 checkpoint staging과 optimizer offload buffer가 같은 host memlock 예산을 경쟁한다. DataLoader만 보고 `Q`를 세면 async checkpoint가 시작되는 순간 host OOM이나 paging storm을 놓친다. `VmLck`, allocator의 current/cached/peak byte, queue depth와 checkpoint stage byte를 같은 timeline에 놓는다.

**PCIe·NVLink·HBM·cache를 서로 다른 병목으로 분해한다**

PCIe 또는 CPU–GPU NVLink는 host와 device 주소 공간 사이 전송의 상한을 정하고, HBM은 device-resident operand가 GPU memory controller를 드나드는 상한을 정한다. node 안 GPU peer traffic은 topology에 따라 직접 NVLink/NVSwitch, PCIe peer path 또는 host staging을 탈 수 있다. 이 셋의 theoretical bandwidth를 한 표에 놓는 것은 가능하지만, 숫자가 크다는 이유로 같은 transaction이라고 보면 안 된다.

복수 GPU copy에 대한 PyTorch 소스는 이 차이를 코드 분기로 드러낸다. contiguous·same-dtype이면 allocator의 async memcpy 경로를 사용할 수 있다. P2P capability가 없거나 dtype·layout이 다르면 peer async, 임시 contiguous tensor 또는 copy kernel이 필요할 수 있다. source와 destination current stream에는 양방향 event barrier가 생긴다. 따라서 `cuda:0→cuda:1` 한 줄은 실제 link를 말하지 않는다. GPU UUID, PCI bus ID, NVLink peer, root complex, P2P enablement, selected copy path와 stream event를 한 행으로 기록해야 한다.

HBM에 도착한 뒤에는 warp가 요청한 유효 byte와 memory fabric이 실제로 운반한 byte가 갈라진다. warp lane이 연속·정렬된 주소를 읽으면 요청이 적은 수의 sector transaction으로 합쳐진다. stride가 크거나 경계가 어긋나면 같은 FLOP에도 더 많은 sector를 가져온다. 유효 대역폭은

\[
BW_{useful}=\frac{bytes\ actually\ used}{time},\qquad
\eta_{transaction}=\frac{requested\ useful\ bytes}{transferred\ sector\ bytes}
\]

로 나눠 본다. `L2 hit rate`가 높다는 사실만으로 좋다고 판단할 수 없다. 재사용할 가치가 없는 over-fetched line을 반복해서 맞히는 경우도 있고, streaming workload는 hit rate가 낮아도 transaction efficiency와 HBM throughput이 충분할 수 있다. 반대로 fusion은 intermediate HBM write/read를 줄이지만 register pressure와 shared-memory footprint를 키워 occupancy를 낮추거나 spill을 global memory traffic으로 되돌릴 수 있다.

kernel 디깅은 다음 순서가 효율적이다. 먼저 tensor의 shape·stride·alignment와 실제 selected kernel을 고정한다. 그 다음 requested global-load/store byte와 sector transaction, L2 downstream byte, DRAM byte를 본다. 이어 register/thread, shared memory/block, active warp와 spill을 확인한다. 마지막에만 “memory bound”라고 이름 붙인다. arithmetic intensity \(I=FLOP/byte\)와 장비의 compute/bandwidth roof를 비교하되, byte는 tensor 논리 크기가 아니라 실제 계측 traffic을 사용한다.

**NUMA·CPU optimizer·checkpoint가 같은 host fabric을 경쟁한다**

두 socket host에서는 DataLoader worker가 만든 page, pinned allocator를 호출한 thread, GPU가 붙은 PCIe root와 CPU optimizer thread가 서로 다른 NUMA node에 놓일 수 있다. first-touch가 remote node에 page를 배치하면 pinning에 성공해도 H2D DMA가 socket interconnect를 한 번 더 지난다. GPU utilization 저하와 CPU memory bandwidth 상승이 함께 나타날 수 있으며, worker 수를 늘리면 remote traffic만 키울 수도 있다.

DeepSpeed의 ZenFlow CPU Adam 고정 소스는 optimizer를 전용 process에 두고 고정 core thread pool이 contiguous element slice를 처리하도록 한다. 주석은 Adam state가 그 process와 pool에 NUMA-local하다고 명시한다. 공개 시험은 subgroup update, `exp_avg`, `exp_avg_sq`와 step 증가를 검증하지만 NUMA locality나 성능은 측정하지 않는다. 따라서 코드 구조에서 “locality를 의도했다”는 결론은 가능해도 “원격 접근이 없다”거나 특정 배속을 얻는다는 결론은 불가능하다.

CPU offload를 승인할 때는 다음 좌표를 보존한다.

- optimizer process PID와 bound CPU set, NUMA memory policy
- parameter·gradient·moment buffer의 allocation node와 pinned 여부
- GPU와 PCIe root, NIC·NVMe controller의 NUMA distance
- D2H/H2D byte, CPU update byte와 memory-channel bandwidth
- copy stream, optimizer future, next forward가 기다리는 commit event

checkpoint도 같은 문제를 만든다. TorchTitan의 async 경로는 device state를 pinned CPU staging으로 옮기는 future와 storage save future를 분리한다. staging이 끝났다는 것은 GPU buffer를 재사용할 수 있는 host snapshot 경계이지, 원격 object store에 durable generation이 commit됐다는 뜻이 아니다. 연속 save 시험은 이전 saving future를 기다리는 계약을 고정한다. 운영 dashboard가 `checkpoint complete`를 staging future에서 찍으면 장애 복구 시 존재하지 않는 generation을 성공으로 센다.

세 workload가 겹치는 반례를 반드시 둔다. 큰 batch H2D, CPU Adam update, async checkpoint D2H·write를 같은 step에 시작하고 각각 단독 control과 비교한다. host memory controller, UPI/Infinity Fabric, PCIe link, pinned pool과 storage queue를 계층별로 기록한다. checkpoint를 끄자 step이 빨라졌다는 관측만으로 storage가 원인이라고 결론내지 않는다. D2H가 PCIe를, staging이 NUMA bandwidth를, writer가 page cache를 경쟁했을 수 있다.

**UVM page migration은 평균보다 fault tail을 먼저 본다**

Unified Virtual Memory는 CPU와 GPU가 같은 pointer를 쓴다는 프로그래밍 모델이지 모든 장비에서 동일한 물리 cache가 된다는 뜻이 아니다. platform과 allocation·advice에 따라 page migration, remote access 또는 hardware coherence가 개입한다. working set이 HBM을 넘으면 access 순서가 page residency를 흔들고 fault·migration이 step tail에 군집할 수 있다.

UVM을 사용할 때는 allocation byte만 기록하지 말고 page residency, fault source, migrated byte와 prefetch/advice 호출을 epoch·step 구간에 맞춘다. sequential streaming, 반복 재사용, CPU–GPU ping-pong, working-set oversubscription을 별 fixture로 둔다. 평균 step만 보면 수백 step에 한 번 생기는 eviction storm을 숨긴다. p50, p95, p99와 max를 보고 최초 page-fault burst가 DataLoader wait, kernel stall, collective late arrival 중 어디보다 앞섰는지 확인한다.

이번 소스 감사는 UVM migration을 실행하지 않았다. 따라서 page 크기, fault latency, oversubscription 처리량과 Grace 계열 coherent link의 효과는 모두 장비별 `NOT_VERIFIED`다. 공식 문서의 가능 경로와 profiler counter 이름은 실행 계획을 만드는 근거이지 측정 결과가 아니다.

**한 장의 증거 묶음으로 원인을 닫는다**

메모리 계층 사건의 최소 증거 묶음은 다음과 같다.

| 층 | 상태·좌표 | 직접 반증 질문 |
|---|---|---|
| 데이터 | sample/batch ID, decode·collate 시간, queue depth | GPU가 기다린 최초 시각은 pinning 이전인가 |
| host memory | pageable/pinned byte, allocator cached/in-flight, NUMA node | memlock·remote page·reuse event가 원인인가 |
| 전송 | H2D/D2H/P2P byte, copy engine, stream/event | enqueue와 complete 중 어디가 늦었는가 |
| topology | GPU UUID, PCIe root/switch, NVLink peer, CPU node | 의도한 direct path가 실제 선택됐는가 |
| device memory | allocated/reserved, HBM read/write, L2 downstream | capacity, transaction waste, bandwidth 중 무엇인가 |
| SM | selected kernel, sector efficiency, register/shared/spill | fusion이 traffic을 줄이고도 occupancy를 잃었는가 |
| 비동기 저장 | staging future, writer future, durable manifest | host snapshot을 durable commit으로 오인했는가 |

실용적인 failure ladder는 `입력 생산 → pin queue → DMA enqueue → DMA complete → first consumer → HBM/L2 transaction → optimizer/checkpoint 경쟁` 순으로 최초 불변식 위반을 찾는다. 각 단계에 monotonic timestamp와 tensor/batch identity를 붙이고, 같은 shape의 clean control을 둔다. metric 상관만으로 인과를 확정하지 않고 stream event, allocator state와 test fixture로 반증한다.

마지막으로 옵션을 효과가 아니라 상태 전이로 기록한다. `pin_memory=True`는 pinned staging을 만든다. `non_blocking=True`는 async copy와 event-tracked lifetime을 만든다. worker·prefetch 증가는 동시에 살아 있는 host object와 queue depth를 늘린다. CPU optimizer offload는 HBM capacity를 줄이는 대신 host NUMA 계산과 PCIe/DMA 경계를 추가한다. async checkpoint는 pause를 숨길 수 있지만 pinned pool·link·storage의 동시 경쟁과 두 단계 commit을 만든다. 이 인과 사슬까지 설명해야 “GPU가 놀고 있으니 DataLoader를 늘린다”는 추측이 재현 가능한 진단으로 바뀐다.
