# 22장 diffusion과 flow: 노이즈에서 경로를 학습한다

autoregressive 모델이 다음 token의 조건부 분포를 맞힌다면 diffusion은 오염된 상태에서 깨끗한 방향을 맞힌다. 그러나 “노이즈를 지운다”는 한 문장만으로는 학습 코드를 읽을 수 없다. 실제 한 microstep은 **원자료와 조건 → latent → noise와 time 표본 → forward corruption 또는 interpolation → denoiser conditioning → prediction target → 가중된 합과 분모 → gradient → optimizer와 EMA → checkpoint**라는 상태 사슬이다. 생성은 이 사슬을 거꾸로 재생하는 일이 아니다. 학습된 score·noise·velocity field를 **별도의 수치 solver 계약**으로 적분하는 과정이다.

따라서 이 장의 기준 질문은 “어떤 diffusion인가”가 아니라 “어느 좌표의 상태를, 어느 시간 규약으로 만들고, 모델 출력의 어떤 의미에 손실을 주며, 어느 solver가 그 출력을 어떻게 해석하는가”이다. 이 네 질문 중 하나라도 비어 있으면 shape가 모두 맞고 loss도 내려가면서 잘못된 모델을 만들 수 있다. 특히 `prediction_type`, latent scale, time 방향과 loss denominator는 문자열이나 사소한 상수가 아니라 학습할 벡터장을 바꾸는 상태다.

## 22.1 확산 학습을 corruption·target·denoising 상태 사슬로 읽는다

확산 모델을 노이즈 제거 그림으로 축약하지 않고 clean sample, time/noise draw, corrupted state, target, prediction과 loss의 소유자로 나눈다.

### 독자가 먼저 붙잡아야 할 상태 원장

| 경계 | 대표 상태와 shape | 이 경계가 바꾸는 의미 | 반드시 남길 증거 |
|---|---|---|---|
| 원자료·조건 | image/video/audio, text token `[B,L]`, mask | 학습 모집단과 조건 분포 | `sample_id`, 전처리·tokenizer revision, condition checksum |
| 표현 좌표 | pixel 또는 VAE latent `[B,C,H,W]` | 거리·분산·SNR의 단위 | VAE revision, posterior sample/mode, scale·shift, latent norm |
| 시간·무작위성 | `t` 또는 `sigma` `[B]`, noise와 dropout mask | 어떤 난이도를 얼마나 자주 보는가 | proposal density, RNG owner/state, time histogram |
| 오염·보간 | `x_t` `[B,C,H,W]` 또는 corrupted token `[B,L]` | forward conditional path | schedule digest, `x_0/noise/x_t` checksum |
| 조건부 denoiser | timestep embedding, cross-attention state, model output | 조건이 벡터장에 들어오는 위치 | tensor shape·dtype, CFG-drop mask, first block trace |
| target·loss | `epsilon`, `x_0`, `v`, flow velocity, score 또는 categorical posterior | 모델 출력의 좌표계와 gradient budget | unreduced error, time weight, numerator와 valid count |
| update | gradient, optimizer/scaler, EMA | 실제로 바뀐 parameter trajectory | overflow·skip 여부, parameter delta, EMA update count |
| 저장·재개 | weights, optimizer, RNG, sampler, EMA, config | 다음 step의 동일성 | artifact digest, first-resumed-batch golden trace |
| 생성·평가 | initial state, time grid, solver history, decoder | field를 어떤 경로와 비용으로 적분하는가 | step별 state, NFE, failed-row 포함 평가 분모 |

이 표는 추상적인 목차가 아니라 장애를 자르는 순서다. 두 run의 최종 이미지가 다르면 먼저 같은 `sample_id`가 같은 latent가 되었는지, 같은 `(t, noise)`가 같은 `x_t`를 만들었는지, 같은 condition에서 첫 model output이 같은지, 그다음 solver update가 같은지를 비교한다. 최초로 달라진 행보다 아래쪽 component를 먼저 의심하면 원인과 결과를 뒤집게 된다.

### 이름이 아니라 입력·목적함수·solver 계약으로 나눈다

| 계열 | 학습 입력을 만드는 법 | 모델이 맞히는 대표량 | 생성 계약 | 자주 생기는 범주 오류 |
|---|---|---|---|---|
| DDPM | `x_t=alpha_t x_0+sigma_t epsilon` | `epsilon`, `x_0`, `v` 또는 이에 대응하는 score | reverse Markov transition, 분산을 포함할 수 있음 | epsilon checkpoint를 v scheduler로 해석 |
| DDIM·probability-flow ODE | DDPM과 같은 marginal을 쓸 수 있음 | 같은 denoiser parameterization | 선택한 deterministic/stochastic coupling과 ODE형 update | “같은 marginal”을 “같은 sample path”로 오해 |
| score SDE | SDE의 perturbation kernel로 `x_t`를 sample | 시간별 score `nabla_x log p_t(x)` | reverse-time SDE 또는 probability-flow ODE | score와 epsilon 사이의 `sigma_t` scale 누락 |
| flow matching | 선택한 endpoint coupling과 path로 `x_t` 보간 | conditional velocity 또는 marginal velocity | `dx/dt=u_theta(x,t,c)`의 ODE 적분 | diffusion의 `v_prediction`과 flow velocity를 같은 `v`로 취급 |
| discrete diffusion LM | mask/범주 transition으로 token state를 오염 | clean-token logits, posterior 또는 score-like quantity | categorical transition, confidence commit·remask 정책 | Gaussian 식과 continuous DDIM 옵션을 그대로 이식 |

DDPM과 DDIM의 차이는 단순히 “확률적 대 결정적”이라는 옵션 차이가 아니다. forward marginal을 공유할 수 있어도 reverse path와 stochastic state가 다르다. score SDE는 시간별 log-density gradient를 목표로 삼고 reverse SDE와 probability-flow ODE라는 두 생성 해석을 구분한다. flow matching은 미리 고른 확률 경로의 속도 회귀에서 출발하며, discrete diffusion은 애초에 상태공간과 transition measure가 다르다. 비교 표에서는 반드시 입력 상태, target 식, time 방향, solver state와 NFE를 한 행에 둔다.

### DDPM과 score matching

먼저 forward process와 목표를 고정한다. `x_t=√ᾱ_t x_0+√(1-ᾱ_t)ε`, `ε~N(0,I)`로 표본을 오염시킨다. 모델이 `ε`, `x_0`, velocity `v` 중 무엇을 예측하는지에 따라 target scale과 SNR weighting이 달라진다. 입력 latent `[B,C,H,W]`, timestep `[B]`, condition embedding `[B,L,D]`의 dtype과 broadcast 위치를 고정한다. prediction type을 바꾸고 checkpoint를 그대로 재개하면 loss 숫자는 나와도 다른 목적함수를 최적화한다.

왜 timestep sampling이 데이터 sampling인가? uniform timestep은 각 정수 step을 같은 확률로 뽑지만 각 step의 SNR과 gradient 난이도까지 같게 만들지는 않는다. 특정 구간에서 gradient 분산이 커지면 loss-aware sampler가 그 구간의 표본 빈도를 바꾸고, 그 결과 모델이 보는 오염 난이도의 분포도 달라진다. 이 때문에 sampler histogram과 RNG는 부수적인 로그가 아니라 run state다. plateau가 나타나면 batch별 timestep histogram과 SNR bucket loss를 함께 대조해야 데이터 문제와 noise curriculum 문제를 나눌 수 있다.

다음 경계는 score와 epsilon parameterization이다. Gaussian perturbation에서 score `∇_{x_t} log p_t(x_t)`는 noise prediction과 scale 관계를 갖는다. epsilon target을 score로 해석할 때 `σ_t`를 빠뜨리면 timestep별 크기가 달라진다. 구현의 `prediction_type`과 scheduler가 model output을 어떤 식으로 `x_0` 또는 derivative로 바꾸는지 함께 읽는다.

이 흐름에서 SNR weighting도 빠뜨릴 수 없다. 단순 MSE는 timestep별 정보량 차이를 그대로 둔다. min-SNR류 weighting은 특정 SNR 구간이 gradient를 독점하지 않게 한다. metric은 weighted loss만 내지 말고 raw MSE, weight, weighted contribution을 SNR bucket별로 낸다. weight 변경 뒤 평균 loss 숫자를 이전 run과 직접 비교하지 않는다.

이제 살펴볼 것은 noise 생성에서 backward까지 이어지는 소유권이다. DDPM 논문의 forward Markov chain은 닫힌 형태의 `q(x_t|x_0)` 덕분에 중간 step을 순서대로 만들지 않고 임의 `t`를 직접 sample할 수 있다. 구현의 training loop는 clean batch, timestep, Gaussian noise를 뽑고 scheduler의 `add_noise` 또는 동등 식으로 `x_t`를 만든다. denoiser 입력과 target을 만든 뒤 weighted MSE를 reduce한다.

latent diffusion에서는 `x_0`가 pixel이 아니라 VAE latent다. encoder posterior에서 sample하는지 mode를 쓰는지, latent scaling factor를 어디서 곱하는지가 state 계약이다. VAE를 freeze해도 stochastic posterior sample RNG가 재현성에 들어간다. text condition에는 tokenizer/text encoder revision과 dropout 정책이 따른다.

batch atlas에는 `sample_id, x0 checksum, latent checksum, t, noise checksum, xt checksum, condition checksum, target type, loss weight`를 담는다. 같은 `x_t`를 재생성할 수 없다면 denoiser regression을 분리할 수 없다.

backward의 시간 조건부터 살펴보자. UNet/DiT는 timestep embedding을 여러 block에 주입한다. timestep tensor dtype과 normalization이 달라지면 같은 integer `t`도 다른 conditioning이 된다. classifier-free training은 일정 확률로 condition을 비우므로 condition-drop mask와 RNG가 batch state다.

backward에서는 noisy latent, timestep embedding, condition KV, block activation이 저장되거나 recompute된다. activation checkpointing과 attention backend는 memory/order를 바꾸지만 objective는 같아야 한다. first-divergence test는 같은 `x_t`와 condition에서 block별 output/gradient를 비교한다.

VAE/text encoder를 freeze했는지, LoRA가 denoiser의 어떤 projection에 들어갔는지 optimizer manifest로 확인한다. frozen parameter delta=0, adapter gradient nonzero를 작은 overfit fixture에서 검증한다.

Diffusers 학습 loop를 tensor 경계에서 멈춰 읽는다. Diffusers 고정 revision `d57cecde92a6d396845ab35425aa27469dff8173`의 [`examples/text_to_image/train_text_to_image.py`](https://github.com/huggingface/diffusers/blob/d57cecde92a6d396845ab35425aa27469dff8173/examples/text_to_image/train_text_to_image.py)는 이 상태 사슬을 한곳에서 보여 준다.

`vae.encode(...).latent_dist.sample()`과 scaling 뒤의 latent, `torch.randn_like(latents)`의 noise, batch마다 뽑은 integer timestep, `noise_scheduler.add_noise`, UNet forward, `prediction_type`별 target, unreduced MSE와 SNR weight, `accelerator.backward`, optimizer와 EMA update를 순서대로 읽는다.

로컬 보존 좌표는 같은 파일 `968`, `972`, `982`, `988–990`, `1000–1046`, `1058`행이다. 이 좌표는 revision이 바뀌면 다시 확인해야 하며, 예제 하나의 선택을 모든 pipeline의 보편적 기본값으로 확대해서는 안 된다.

전형적인 image latent에서 `pixel_values [B,3,H,W]`는 VAE를 지나 `latents [B,C,h,w]`가 되고, noise도 정확히 그 shape를 가진다. timestep은 `[B]`지만 scheduler의 `add_noise` 내부에서는 누적 alpha를 sample rank까지 `unsqueeze`해 broadcast한다. condition hidden state는 보통 `[B,L,D]`이며 denoiser 출력은 fixed-variance noise head라면 latent와 같은 `[B,C,h,w]`다. learned variance를 함께 내는 scheduler 조합은 channel 분할 계약이 추가되므로 “항상 같은 shape”라고 일반화하지 않는다. shape 표에는 axis의 의미와 함께 owner, dtype, valid mask를 적는다.

여기서 중요한 경계는 세 개다. 첫째, `add_noise(x_0, epsilon, t)`는 학습 입력을 만들 뿐 inference `step`이 아니다.

둘째, target branch는 model output의 의미를 정한다. [`DDPMScheduler.get_velocity`](https://github.com/huggingface/diffusers/blob/d57cecde92a6d396845ab35425aa27469dff8173/src/diffusers/schedulers/scheduling_ddpm.py#L611)는 sample·noise·timestep으로 velocity target을 만들고, 같은 파일의 [`step`](https://github.com/huggingface/diffusers/blob/d57cecde92a6d396845ab35425aa27469dff8173/src/diffusers/schedulers/scheduling_ddpm.py#L445)는 `prediction_type`에 따라 출력 해석을 바꾼다.

셋째, `reduction="none"` 뒤 공간·channel 평균과 batch weighted mean의 순서가 실제 gradient 분모를 정한다. 함수 이름보다 이 세 경계의 수치 fixture가 강한 증거다.

가중치와 분모를 하나의 식으로 닫는다. sample `i`의 유효 원소 집합을 `M_i`, time weight를 `w_i`, 원소 오차를 `e_ij`라 하면 안전한 기록 단위는 다음 두 값이다.

`N = sum_i w_i sum_(j in M_i) e_ij`, `D = sum_i w_i |M_i|`, `L = N/D`.

실제 recipe가 sample 내부 평균 뒤 `mean_i(w_i mean_j e_ij)`를 쓰면 그 식 자체가 계약이다. 모든 sample의 유효 원소 수가 같을 때만 위 global element mean과 같아질 수 있다. variable resolution, inpainting mask, padded video frame, discrete-token corruption에서는 같지 않다. 분산 학습에서 rank-local scalar loss를 평균하면 더 위험하다. 각 rank의 `(N_r,D_r)`를 합쳐 `sum_r N_r / sum_r D_r`를 만들지, 아니면 의도적으로 rank/sample 균등 가중을 쓸지 명시한다. DDP가 gradient를 평균하는 사실이 잘못된 local denominator를 자동으로 고쳐 주지는 않는다.

네 개의 분리 실험. scheduler mismatch는 같은 `x_t`와 model output을 epsilon·v branch에 각각 넣어 첫 `pred_original_sample`부터 비교한다. time mismatch는 같은 scalar `t`가 training의 integer index인지 inference sigma인지 바꾸고 `x_t`와 첫 update를 비교한다. latent-scale mismatch는 VAE 출력 직후와 scale 적용 뒤 norm만 바꾸고 noise·condition을 고정한다. denominator mismatch는 두 rank에 유효 원소 수가 1과 3인 손계산 batch를 주어 local-mean average와 global `(N,D)`의 차이를 노출한다. 네 실험 모두 최종 이미지가 아니라 **최초 달라져야 하는 tensor**를 assertion으로 삼는다.

## 22.2 sampler를 학습된 score·velocity의 수치 적분기로 읽는다

학습 objective와 inference solver를 구분하고 step size, schedule과 prediction parameterization이 trajectory를 어떻게 바꾸는지 본다.

### DDIM·DPM-Solver

DDIM의 deterministic 경로와 stochastic DDPM 경로는 같은 denoiser를 써도 trajectory가 다르다. DPM-Solver류는 더 큰 step으로 ODE를 적분하며 model evaluation 수를 줄인다. “20 steps” 비교는 scheduler, sigma sequence, prediction type, guidance scale가 같을 때만 의미가 있다. 각 step의 latent checksum과 norm을 기록하면 최초 발산 지점을 찾을 수 있다.

### flow matching의 직관

flow matching은 두 분포 사이의 확률 경로와 그 속도장을 학습한다. 단순 선형 interpolation에서는 `x_t=(1-t)x_0+t x_1`, target velocity는 `x_1-x_0`다. 실제 모델은 coupling과 time parameterization에 따라 난이도가 달라진다. “직선이라 쉽다”는 설명은 선택한 coupling이 실제 데이터 manifold에서 짧은 경로를 만든다는 보장이 없음을 함께 말해야 한다.

### local truncation과 global trajectory error

solver의 한 step 오차가 작아도 여러 step에서 누적된다. step `k`의 latent error와 최종 perceptual/task error를 함께 본다. model evaluation 수를 줄일 때 sigma grid가 고주파 구간을 건너뛰는지 확인한다. 같은 20-step이라도 spacing이 다르면 다른 실험이다.

guidance가 벡터장을 바꾼다. classifier-free guidance는 conditional과 unconditional prediction을 `u+w(c-u)`로 조합한다. `w`가 크면 condition adherence가 좋아질 수 있지만 norm이 커지고 saturation/artifact가 생길 수 있다. 두 prediction norm과 조합 후 norm을 step별로 기록한다. guidance rescale 구현은 별도 옵션 계약이다.

이제 살펴볼 것은 scheduler 구현의 상태 머신이다. 대표 scheduler는 training noise schedule과 inference timestep schedule을 구분한다. `set_timesteps(N)`이 inference grid와 내부 index를 만들고 `scale_model_input`, model call, `step`이 반복된다. multistep solver는 이전 model output history를 소유할 수 있다. 중간 resume에는 latent와 loop index만이 아니라 이 history가 필요하다.

config에는 beta/sigma schedule, prediction type, timestep spacing, clipping/dynamic threshold, solver order와 algorithm type이 들어간다. class 이름이 같아도 config가 다르면 다른 trajectory다. serialized scheduler config digest와 실제 timestep/sigma array checksum을 저장한다.

`step` return에 previous sample뿐 아니라 predicted original sample이 있으면 둘을 기록한다. divergence가 model prediction인지 solver transform인지 분리하는 관측점이다.

ODE·SDE 경로의 차이부터 살펴보자. probability flow ODE와 reverse SDE는 같은 marginal을 목표로 할 수 있지만 sample path와 stochastic state가 다르다. deterministic ODE solver는 initial noise와 numerical path로 정해지고, ancestral sampler는 step별 추가 noise를 뽑을 수 있다. seed 하나만 기록하지 말고 generator state와 noise draw 위치를 기록한다.

solver order를 높이면 smooth vector field에서 evaluation당 정확도가 좋아질 수 있으나 model error와 discontinuous guidance가 지배하면 이득이 제한된다. 공개 benchmark 수치를 우리 model/latent/guidance에 그대로 적용하지 않는다. 같은 initial noise의 step별 error와 최종 paired metric으로 비교한다.

## 22.3 discrete text diffusion의 categorical state와 transition을 정의한다

연속 Gaussian noise와 mask·token transition은 다른 확률공간이다. transition matrix, posterior와 likelihood 경계를 별도로 유도한다.

### mask corruption과 transition matrix

연속 Gaussian 대신 token을 mask로 바꾸거나 categorical transition을 적용한다. 상태는 `[B,L]` integer이고 timestep별 corruption probability가 attention mask와 별개다. padding, original mask token, corrupted mask token을 같은 ID로 처리하면 loss-bearing 위치를 잃는다. 평가에서는 denoising step마다 변경된 token 수와 확정된 token의 재오염 허용 여부를 기록한다.

### autoregressive와 공정하게 비교하기

한 diffusion step은 sequence 전체 logits를 계산하므로 step 수만으로 AR token step과 비교할 수 없다. model FLOP, KV 재사용 여부, 실제 wall time, 완성 품질을 같은 hardware에서 측정한다. 실행하지 않은 속도 비교는 복잡도 예상으로만 쓴다.

### absorbing mask와 categorical transition

absorbing mask에서는 한 번 mask가 된 token이 forward process에서 돌아오지 않는다. uniform categorical corruption은 다른 vocabulary token으로 이동할 수 있다. transition matrix가 다른데 같은 discrete diffusion이라 부르면 posterior와 loss가 달라진다. padding과 special token을 transition 대상에서 제외하는지 확인한다.

다음 경계는 denoising update 정책이다. 모든 위치를 매 step 다시 뽑는지, confidence가 높은 token을 확정하는지, 일부를 remask하는지가 generation state다. 고정 seed 재현에는 token state뿐 아니라 확정 mask, confidence, scheduler step이 필요하다. partial resume가 이 셋을 복원하지 않으면 같은 trajectory가 아니다.

이 흐름에서 discrete loss의 tensor 계약도 빠뜨릴 수 없다. categorical model output은 `[B,L,V]` logits다. target이 original token인지 reverse transition posterior인지에 따라 CE/KL이 달라진다. corrupted/padding/condition 위치 mask와 denominator를 기록한다. mask-only loss라면 timestep에 따라 target 수가 달라지므로 loss 합과 유효 위치 수를 별도로 reduce한다.

vocabulary 전체 softmax는 큰 비용이다. factorization/absorbing state와 sampled objective가 들어가면 논문 식과 구현 approximation을 구분한다. token transition에서 special/control token을 보호하는 mask가 train과 generation에 같은지 본다.

이제 살펴볼 것은 text diffusion 실패 실험이다. all-mask, no-mask, padding-only, 하나의 corrupted token fixture를 만든다. all-mask에서 target count와 logits finite를, no-mask에서 loss 정책을, padding-only가 분모에 들어가지 않는지, single corruption의 posterior를 손으로 검산한다. generation resume에서 confirmed mask를 일부 지워 결과가 달라지는지 확인한다.

AR 비교는 같은 tokenizer, prompt set, output length, hardware와 품질 관문를 사용한다. wall time에 model evaluation 수와 processed logits elements를 같이 기록한다.

## 22.4 trajectory error와 checkpoint resume를 시간축으로 추적한다

parameter뿐 아니라 timestep sampler, noise RNG, EMA, scheduler와 data state가 다음 trajectory를 결정한다.

### scheduler state 계약

재현 가능한 sample은 model digest, VAE/tokenizer digest, scheduler class/config, timestep/sigma array, seed, RNG backend, condition checksum을 요구한다. 중간 latent만 저장하고 solver history가 필요한 multistep scheduler state를 버리면 정확히 재개할 수 없다.

### 분기 실험

같은 `x_t`에서 두 scheduler가 내는 다음 latent의 cosine distance와 max error를 잰다. 첫 step부터 다르면 config/prediction type, 뒤에서 누적되면 numerical precision/solver error를 의심한다. NaN은 latent norm, attention score, guidance 전후 prediction norm 순서로 이분한다.

### 소스/시험 경계

Diffusers 고정 checkout에서 scheduler의 `set_timesteps`, scale/model-input, `step`과 config serialization을 추적한다. upstream unit test는 수식 fixture와 shape를 증명할 수 있지만 특정 checkpoint의 품질·속도를 증명하지 않는다. 논문의 solver order와 library의 clipping/dynamic threshold/default spacing도 분리한다.

trajectory 결정 트리부터 살펴보자. initial noise가 다르면 seed/RNG/device, 첫 model output이 다르면 condition/tokenizer/model, model output은 같은데 next latent가 다르면 scheduler config/dtype, 중간부터 다르면 solver history/nondeterminism을 본다. 최종 image metric만으로는 이 순서를 복원할 수 없다.

다음 경계는 checkpoint inventory이다. training checkpoint는 denoiser, trainable VAE/text encoder/adapter, optimizer, scaler, scheduler LR clock, data sampler와 RNG를 포함한다. noise scheduler config는 model weight가 아니지만 run artifact다. timestep sampler가 loss-aware라면 bucket statistics도 state다. EMA model을 평가에 쓰면 EMA accumulator와 update count를 저장한다.

inference trajectory checkpoint는 model/VAE/text/tokenizer/scheduler digest, condition, initial noise, current latent, timestep index, solver history와 generator state를 가진다. training checkpoint와 sample-resume checkpoint를 같은 것으로 부르지 않는다.

이 흐름에서 실패 주입도 빠뜨릴 수 없다. 첫째 prediction type을 epsilon에서 v로 바꾸고 old checkpoint를 load해 config mismatch가 fail-fast하는지 본다. 둘째 multistep history를 버리고 중간 resume해 step별 divergence가 검출되는지 본다. 셋째 condition dropout RNG를 복원하지 않은 training resume에서 첫 batch loss/gradient가 달라지는지 본다. 넷째 partial artifact에서 VAE scale만 바꿔 latent norm alert가 잡는지 본다.

각 실험은 final sample만 보지 않는다. initial/condition/model-output/solver-output checksum을 비교해 최초 경계를 찾는다. 예상과 다른 경계가 먼저 갈라지면 원 가설을 기각한다.

## 22.5 한 trajectory에서 DDPM·flow·discrete objective를 검산한다

작은 scalar·categorical fixture로 corruption, target, loss와 reverse step을 손 계산해 서로 다른 parameterization을 비교한다.

`[1,4,64,64]` BF16 latent에 20-step solver를 적용한다고 하자. manifest는 16,384 element, initial norm, sigma array를 가진다. 매 step conditional/unconditional prediction norm, guided prediction, predicted x0, next latent와 duration을 기록한다. full tensor 보존은 첫/문제 step으로 제한하고 나머지는 checksum/statistic을 쓴다.

step 0부터 model output이 다르면 model/condition path다. model output은 같고 next latent만 다르면 scheduler다. step 7까지 같다가 달라지면 multistep history, nondeterministic kernel 또는 precision을 본다. 이 trace가 solver 이름보다 더 정확한 실험 identity다.

### 논문·구현·실행 판정

논문에는 objective와 solver의 가정/수렴 성질이 정리돼 있다. 대표 구현에서는 config default, clipping, dtype, history와 API state를 확인한다. upstream test는 작은 수식 fixture와 serialization을 확인할 수 있다. 실제 checkpoint의 품질·속도·memory는 우리 hardware 실행이 필요하다.

따라서 “DPM-Solver가 빠르다” 대신 같은 model/guidance/quality에서 필요한 model evaluation과 wall time을 측정해야 한다고 쓴다. 실행 전에는 예상 복잡도다. 공개 수치를 인용하면 workload와 revision을 함께 둔다.

### DDPM 손계산 fixture

2차원 `x_0=[1,-1]`, `ᾱ_t=0.64`, noise `ε=[0.5,-0.25]`를 쓰면 `x_t=0.8x_0+0.6ε=[1.1,-0.95]`다. epsilon prediction model이 `[0.4,-0.3]`을 내면 element MSE는 `[0.01,0.0025]`, mean은 `0.00625`다. batch/time weight를 곱하기 전 raw 값을 저장한다.

이 작은 fixture로 scheduler `add_noise`, target construction, reduction을 검산한다. framework 결과가 다르면 sqrt alpha, broadcast dtype, variance schedule index와 mean/sum을 본다. training loop 전체를 실행하기 전에 CPU 고정 tensor로 닫는다.

같은 fixture에서 x0 prediction과 v prediction target도 계산해 config switch가 target을 실제로 바꾸는지 본다. checkpoint의 output head shape는 같아 load가 성공할 수 있으므로 semantic mismatch를 config guard로 막아야 한다.

### latent scaling 실패

latent diffusion의 VAE scaling factor가 빠지면 `x_0` norm과 SNR 해석이 달라진다. denoiser가 학습 때 본 분포와 inference 분포가 달라져 과포화/무의미한 sample이 생길 수 있다. encoder 직후 latent norm, scale 적용 뒤 norm, initial noisy latent norm을 metric으로 둔다.

실패 실험에서는 동일 image를 encode하고 scaling factor를 1과 declared value로 바꾼다. model input checksum과 첫 prediction norm이 예상 비율로 변하는지 본다. final image만 비교하면 VAE decode clipping과 guidance가 오류를 가릴 수 있다.

export package는 VAE config와 scaling factor를 포함하고 serving pipeline이 이를 중복 적용하지 않는지 확인한다. base model과 VAE를 별도 repo에서 가져왔다면 두 digest의 compatibility를 manifest에 둔다.

이제 살펴볼 것은 condition 경로 검산이다. text prompt는 tokenizer→text encoder→attention mask→condition embedding을 지난다. negative prompt와 unconditional branch가 빈 문자열인지 learned null embedding인지 구현을 읽는다. prompt truncation과 padding이 condition checksum을 바꾼다. 같은 image seed인데 prompt token이 다르면 scheduler를 조사할 이유가 없다.

cross-attention layer별 condition KV shape와 mask를 표본 하나에서 기록한다. text encoder freeze/LoRA policy와 optimizer parameter set을 비교한다. condition dropout에서는 dropped sample의 embedding identity와 mask를 확인한다.

실패 fixture로 빈 prompt, max-length 경계, 서로 같은 prefix를 가진 두 prompt를 쓴다. tokenizer divergence, encoder divergence, denoiser cross-attention divergence를 순서대로 찾는다.

distributed training의 denominator부터 살펴보자. rank마다 서로 다른 timestep/SNR bucket을 뽑는다. local mean을 rank 평균하면 각 rank의 element 수가 같을 때는 같지만 variable resolution/mask가 있으면 global element mean과 달라진다. numerator, valid latent/mask element와 weight sum을 분리해 reduce한다.

gradient accumulation 중 timestep histogram이 skew될 수 있다. optimizer step 단위 histogram과 weighted contribution을 낸다. loss-aware sampler 통계가 rank별인지 global인지, update에 collective가 있는지 source에서 확인한다. rank-local sampler가 각자 다른 난이도 분포로 drift할 수 있다.

FSDP/TP로 denoiser를 shard할 때 noise/condition batch owner와 dropout RNG를 기록한다. activation checkpoint recompute가 condition dropout을 다시 뽑지 않도록 RNG 보존을 확인한다.

다음 경계는 EMA와 평가 artifact이다. diffusion training은 raw model과 EMA model을 함께 가질 수 있다. EMA `θ_ema←μθ_ema+(1-μ)θ`에서 decay, warmup, update count가 결과를 결정한다. optimizer step이 skip됐을 때 EMA도 skip하는지 구현을 읽는다. resume에 EMA state가 없으면 평가 모델 lineage가 끊긴다.

model card 점수가 EMA인지 raw인지, sampler/guidance가 무엇인지 확인한다. checkpoint 파일명 `ema`만 믿지 않고 parameter checksum과 update count를 manifest에 둔다. export가 어느 branch를 선택했는지도 derivation DAG에 남긴다.

실험은 같은 initial noise에서 raw/EMA를 paired 평가한다. 품질 평균뿐 아니라 step별 prediction norm과 sample diversity를 본다. 실행하지 않은 일반적 EMA 우월성을 결과로 쓰지 않는다.

이 흐름에서 quantization·저정밀 실행도 빠뜨릴 수 없다. denoiser weight를 FP8/INT8/INT4로 바꾸면 model output error가 solver trajectory에서 누적될 수 있다. 단일 forward cosine이 높아도 여러 step 뒤 sample이 갈라질 수 있다. layer output error, step model-output error, next-latent error, final metric을 계층적으로 본다.

time embedding, normalization, output projection처럼 민감한 layer를 higher precision으로 남길 수 있다. 실제 backend가 quant kernel을 선택했는지 fallback log와 profiler로 확인한다. memory/latency 수치는 같은 scheduler evaluation 수와 resolution에서 비교한다.

quantized checkpoint는 scale/group/calibration digest와 denoiser config를 가진다. VAE/text encoder dtype도 별도로 기록한다. “전체 pipeline 8비트”라는 표현으로 서로 다른 component dtype을 숨기지 않는다.

이제 살펴볼 것은 성능 측정 계약이다. latency는 prompt encode, VAE encode/decode, denoiser model evaluations, scheduler CPU/GPU step와 transfer를 나눈다. warmup, CUDA synchronization, compilation/graph capture를 구분한다. batch/resolution/steps/guidance가 다르면 비교하지 않는다.

throughput을 images/s로만 내면 resolution과 step 수를 숨긴다. latent elements×model evaluations/s와 end-to-end images/s를 같이 낸다. peak memory는 denoiser step, attention, VAE decode 중 어느 시점인지 표시한다.

Nsight trace가 없으면 kernel 원인을 단정하지 않는다. PyTorch profiler의 operator time과 memory는 후보를 좁히고, kernel counter가 필요하면 문제 step/layer만 NCU로 본다.

최종 decision gate부터 살펴보자. representation checksum, condition checksum, noise/timestep, prediction type, scheduler trajectory, checkpoint/EMA, eval protocol이 모두 고정돼야 모델 비교를 승인한다. 한 항목이 다르면 별도 experiment ID다. quality 점수 하나로 config drift를 승인하지 않는다.

solver 교체는 같은 model output fixture에서 next latent hand-check를 먼저 통과한다. model 교체는 같은 scheduler/initial noise에서 first prediction을 비교한다. quantization은 full-precision trajectory와 error budget을 비교한다. resume는 same trajectory state와 solver history를 비교한다.

장 끝의 `NoiseTrajectoryID`에는 이 모든 digest와 실행 여부가 들어간다. 23장의 unlearning이 diffusion model에 적용되면 어떤 representation/denoiser/EMA/quantized descendant가 무효화되는지 추적할 수 있다.

flow와 discrete diffusion을 나란히 검산한다. 2차원 source `x_0=[-1,0]`, target `x_1=[1,2]`와 선형 경로를 쓰자. `t=0.25`에서 `x_t=[-0.5,0.5]`, target velocity는 `[2,2]`다. model prediction이 `[1.8,2.1]`이면 element squared error는 `[0.04,0.01]`, mean은 `0.025`다. time weighting과 batch reduction 전 raw 값이다.

이 fixture로 interpolation 방향, time broadcast, target sign을 검산한다. source/target 이름을 반대로 쓰면 velocity sign이 바뀌지만 loss는 계속 내려갈 수 있다. generation ODE가 training orientation과 같은지 one-step integration으로 확인한다.

nonlinear probability path나 optimal-transport coupling을 쓰면 target velocity도 달라진다. 논문의 선택과 implementation sampler를 같은 이름만 보고 연결하지 않는다. coupling batch permutation과 RNG도 training state다.

이 흐름에서 discrete diffusion 손계산 fixture도 빠뜨릴 수 없다. vocabulary `{A,B,MASK,PAD}`에서 `A`를 확률 `q_t`로 MASK로 바꾸는 absorbing process를 생각하자. padding은 transition 대상이 아니다. 입력 `[A,PAD]`가 `[MASK,PAD]`가 됐다면 loss-bearing 위치는 첫 token 하나다. logits CE를 sequence 길이 2로 나누지 않고 target count 1로 나눈다.

single-position posterior를 작은 transition matrix로 직접 계산해 library 결과와 비교한다. mask token ID와 padding ID가 같거나 attention mask가 corruption mask로 재사용되면 fixture가 실패해야 한다. no-corruption timestep의 loss 정책도 명시한다.

generation fixture는 `[MASK,MASK]`와 confirmed mask를 저장하고 한 step 뒤 token/state를 비교한다. confidence tie의 sampling order와 RNG를 고정한다. resume에서 confirmed mask를 잃으면 이미 확정한 token을 다시 바꿀 수 있다.

재시작 뒤 첫 분기를 진단한다. resume 검증은 final sample 생성보다 training의 첫 batch에서 시작한다. sample ID, latent/condition, timestep, noise, condition-drop mask, loss weight와 LR을 uninterrupted run과 비교한다. 하나라도 다르면 parameter parity를 기대하지 않는다.

async data prefetch가 noise를 worker에서 뽑는지 trainer에서 뽑는지 확인한다. worker RNG를 저장하지 않으면 같은 sample에도 다른 `x_t`가 생긴다. sample-exact가 요구되지 않는다면 이 차이를 statistical resume로 명시한다.

EMA update count와 optimizer skipped step도 비교한다. AMP overflow로 optimizer가 skip됐는데 EMA와 scheduler만 전진하면 raw/EMA lineage와 time clock이 어긋난다.

NaN 결정 트리부터 살펴보자. clean/latent가 finite인지, `x_t`가 finite인지, first denoiser block, attention, model output, guided output, scheduler next latent 순으로 본다. training NaN이면 raw/weighted loss와 gradient unscale/clip/step을 잇는다. 첫 nonfinite 경계 이전만 조사한다.

특정 low-noise/high-SNR bucket에서만 NaN이면 target/weight scale과 normalization을 본다. guidance에서만 생기면 conditional-unconditional difference와 scale을 본다. scheduler step에서만 생기면 sigma division, clipping과 dtype을 본다.

같은 fixture를 FP32, guidance 1, deterministic one-step으로 축소해 반증한다. failing sample/timestep/noise를 버리는 것은 해결이 아니다.

다음 경계는 plateau 결정 트리이다. weighted 평균만 평평하면 SNR bucket별 raw loss와 contribution을 본다. 모든 bucket gradient가 0이면 detach/frozen/AMP skip, gradient와 delta는 있는데 loss가 고정이면 LR/target/data를 본다. small latent batch를 overfit해 capacity와 pipeline을 분리한다.

condition을 무시하면 prompt shuffle 전후 loss/outputs가 거의 같을 수 있다. cross-attention gradient와 conditional-unconditional prediction distance를 본다. VAE reconstruction ceiling이 낮으면 denoiser를 개선해도 pixel metric이 제한된다.

solver sample 품질 plateau는 training plateau와 다르다. evaluation scheduler/guidance/EMA drift와 fixed initial-noise paired sample을 먼저 확인한다.

이 흐름에서 다음 장으로 넘기는 artifact도 빠뜨릴 수 없다. diffusion model에 데이터 삭제나 concept edit를 적용할 때 base denoiser뿐 아니라 VAE, text encoder adapter, EMA, quantized/exported copy와 cached latent가 descendant다. `NoiseTrajectoryID`가 어느 model/config digest를 썼는지 알아야 retest 대상을 찾는다.

23장은 특정 concept을 지웠다는 behavioral score와 산출물 무효화을 분리한다. 동일 initial noise/condition의 paired trajectory로 edit 전후 최초 model-output divergence와 neighborhood prompt 영향을 본다. scheduler가 다르면 edit 효과를 비교하지 않는다.

이 장의 최종 handoff는 representation, condition, model, scheduler, trajectory와 evaluation의 digest 묶음이다. 일부만 있으면 재생성 가능한 sample artifact가 아니다.

이제 살펴볼 것은 독자가 마지막으로 답해야 할 질문이다. 첫째, model output은 epsilon, score, x0, velocity 중 무엇이며 scheduler는 그것을 어떻게 해석하는가. 둘째, `x_0`는 pixel, VAE latent, discrete token 중 무엇이고 normalization/scale은 어디서 적용되는가. 셋째, timestep/noise/condition-drop을 누가 뽑고 resume 때 어떤 RNG가 복원되는가. 넷째, loss의 numerator와 denominator, SNR weight는 무엇인가.

다섯째, inference scheduler가 소유하는 timestep, sigma, model-output history는 무엇인가. 여섯째, guidance가 어느 dtype에서 어떤 norm으로 결합되는가. 일곱째, raw/EMA/quantized 중 어느 model이 평가와 export에 쓰였는가. 여덟째, 공개 test가 수식 fixture·serialization·shape 중 무엇을 실제로 검증했는가.

답이 문서의 이름이나 pipeline class 하나뿐이면 부족하다. tensor checksum과 state owner, revision을 답할 수 있어야 한다.

검증하지 못한 경계부터 살펴보자. 작은 fixture의 수식 일치는 대형 model의 perceptual quality를 증명하지 않는다. deterministic scheduler도 GPU kernel과 VAE decode까지 bitwise deterministic하다는 뜻이 아니다. 공개 benchmark의 step/latency는 다른 hardware, compiler, attention backend와 guidance에서 달라질 수 있다.

production image/video generation은 safety filter, watermark, postprocessing과 storage까지 포함할 수 있다. 이 장은 representation→denoiser→scheduler trajectory를 닫지만 외부 서비스 전체를 보장하지 않는다. 실행하지 않은 항목은 trace template과 예상 판정만 남긴다.

이 한계를 명시한 뒤에도 디버깅은 가능하다. 최초 divergence를 representation, condition, model output, scheduler output, decode로 나누면 최종 품질 점수만 볼 때보다 훨씬 좁은 가설을 세울 수 있다.

마지막 검사는 저장된 trajectory를 독립 reader로 다시 읽는 것이다. config class를 import하는 데 그치지 않고 timestep/sigma 배열, prediction type, initial/condition/model digest, step별 checksum과 solver history가 서로 일관적인지 확인한다. reader가 원 package의 mutable default를 다시 적용하면 실패다. 이 독립 검증을 통과한 artifact만 다음 장의 edit·unlearning paired 비교에 사용한다.

검증 보고서는 성공한 checksum뿐 아니라 누락된 solver state, 재현하지 못한 GPU kernel, 미실행 품질·성능 비교를 적는다. 독자는 이 경계를 보고 어느 결과를 다시 실행해야 하는지 판단한다. 숨기지 않은 구체적 한계도 충분히 독립적으로 재현 가능한 trajectory artifact의 일부다.

## 22.6 parameterization·solver·Diffusers 함수 경계를 고정한다

epsilon, sample, velocity와 flow target이 scheduler config와 model output을 어떻게 해석하는지 실제 함수 호출에 연결한다.

한 차원의 clean latent `x0=2`, noise `ε=-1`, cumulative alpha가 `0.64`라면 noisy input은 `sqrt(0.64)*2 + sqrt(0.36)*(-1)=1.0`이다. epsilon-prediction model의 target은 -1이며 prediction이 -0.8이면 squared error는 0.04다. batch loss는 valid element와 weight의 numerator/denominator를 명시한다.

같은 `x_t`에서 x0 prediction과 velocity prediction은 scheduler formula로 변환 가능하지만 parameterization과 numerical scale이 다르다. config의 `prediction_type`이 model target 생성과 inference scheduler 양쪽에서 일치해야 한다. 이름만 바꾸어 checkpoint를 재사용하지 않는다.

negative control은 target은 epsilon인데 scheduler가 velocity로 해석하도록 바꾼다. shape와 loss 감소는 가능해도 known scalar fixture와 first denoise step이 실패해야 한다. 수식은 framework source의 target branch와 step symbol에 연결한다.

### velocity와 x0 변환

일반적으로 `v = sqrt(alpha)*ε - sqrt(1-alpha)*x0` 형태를 쓰는 convention이 있다. 위 숫자에서는 `v=0.8*(-1)-0.6*2=-2.0`이다. model이 v=-1.9를 내면 scheduler가 x0와 epsilon을 어떤 식으로 복원하는지 손계산한다. alpha가 timestep 또는 sigma 정의와 맞는지 확인한다.

library마다 alpha/sigma 배열의 indexing, terminal value와 zero-SNR rescale가 다를 수 있다. 공식 문서의 식과 설치 source의 tensor dtype/device branch를 함께 읽는다. small float64 fixture와 production dtype tolerance를 분리한다.

negative fixture는 timestep index를 하나 이동시킨다. noise checksum은 같아도 alpha가 달라 target과 reconstruction이 변해야 한다. `NoiseTrajectoryID`가 timestep/sigma array digest를 포함해야 하는 이유다.

### SNR weighting 수치

두 timestep의 raw MSE가 각각 0.2와 0.8이고 weight가 4와 0.5라면 weighted numerator는 1.2다. denominator를 weight 합 4.5로 둘지 valid element 2로 둘지 objective 계약에 따라 loss는 0.267 또는 0.6이 된다. “weighted MSE”라는 이름만으로 결정할 수 없다.

min-SNR, p2 weighting과 timestep sampling은 gradient contribution을 함께 바꾼다. timestep histogram, raw loss sum, weight sum과 final contribution을 기록한다. rank별 timestep 분포가 다르면 global numerator/denominator를 reduce한다.

negative control은 weight를 적용하되 denominator는 old value로 남기는 구현이다. 작은 two-timestep fixture가 framework output과 맞지 않아야 한다. W&B scalar 하나만으로 이 오류를 찾기 어렵다.

### flow matching 직선 경로

simple flow matching에서 `x_t=(1-t)x_0+t x_1`이면 target velocity는 `x_1-x_0`로 일정하다. `x0=2`, noise endpoint `x1=-1`, `t=0.25`이면 `x_t=1.25`, target은 -3이다. model prediction -2.5의 squared error는 0.25다.

어느 endpoint가 data/noise인지, time이 0→1 또는 1→0인지 library convention을 확인한다. sampler ODE가 학습 방향의 역방향으로 integrate할 수 있다. sign 오류는 output이 생성되면서도 trajectory가 틀릴 수 있다.

negative control은 endpoint를 swap하고 target sign을 바꾸지 않는다. scalar fixture와 one-step Euler가 실패해야 한다. 소스 원장에는 path sampler, target builder와 solver step symbol을 둔다.

다음 경계는 Euler와 Heun step이다. ODE `dx/dt=f(x,t)`에서 Euler는 `x_{n+1}=x_n+h f_n`이다. `x=1`, `h=-0.1`, `f=2`이면 다음 값은 0.8이다. Heun은 predictor의 `f`를 다시 평가해 두 slope 평균을 사용한다. model evaluation 수와 solver history가 artifact에 들어간다.

같은 “20 step”도 Euler 20 NFE와 predictor-corrector 40 NFE일 수 있다. latency/quality 비교에는 NFE, timestep grid와 guidance batch를 분모로 둔다. adaptive solver는 acceptance/rejection state와 tolerance를 저장해야 trajectory resume가 가능하다.

negative fixture는 solver history 없이 중간 checkpoint를 resume한다. multistep solver라면 동일 latent만으로 next step이 같지 않아야 하며 loader가 불완전 state를 거부해야 한다.

이 흐름에서 classifier-free guidance 수치도 빠뜨릴 수 없다. conditional prediction 1.2, unconditional 0.4, guidance scale 5라면 흔한 식 `u+s(c-u)`는 4.4다. 다른 rescale 또는 convention이 있으면 값이 달라진다. model output dtype과 결합 dtype, norm/rescale를 기록한다.

training condition-drop 확률과 inference guidance는 연결되지만 같은 knob이 아니다. dropout mask RNG와 actual dropped count를 보존한다. text encoder output cache key에는 prompt/tokenizer/encoder/clip-skip와 dtype를 포함한다.

negative control은 cond/uncond batch order를 바꾼다. shape는 같지만 guidance scalar fixture가 반대 방향으로 커져야 한다. high scale의 saturation과 dynamic threshold를 별도 소스 분기로 확인한다.

이제 살펴볼 것은 VAE scale과 decode이다. pixel을 VAE latent로 바꿀 때 latent scaling factor, mean/std와 stochastic posterior sampling을 고정한다. 같은 image라도 posterior mode와 sample은 다른 `x0`다. encoder RNG와 latent digest를 `RepresentationID`에서 받는다.

latent scale을 0.18215로 기대하는 checkpoint에 1.0을 쓰면 denoiser input 분포가 달라진다. output image가 나오더라도 quality가 무너질 수 있다. known synthetic latent의 encode/decode와 scale branch를 소스 좌표로 검산한다.

negative control은 scale metadata만 바꾼다. initial noise는 같지만 first model input/output norm과 decoded image digest가 달라야 한다. denoiser 문제로 오인하지 않도록 representation edge에서 실패시킨다.

구현 좌표와 운영 사건을 연결한다. ledger는 noise addition, velocity/target conversion, timestep sampling, scheduler `set_timesteps`, `scale_model_input`, `step`과 solver state serialization을 `repository@commit:path:symbol`로 둔다. pipeline class 하나가 아니라 실제 prediction-type branch와 formula까지 내려간다.

Diffusers 계열이면 scheduler config serialization과 from-config, pipeline의 cond/uncond batching, VAE scale과 EMA selection을 따라간다. upstream test가 scalar formula, shape, save/load 중 무엇을 assert하는지 기록한다.

upgrade 시 default timestep spacing, terminal sigma와 clipping branch diff를 본다. mutable scheduler config를 trajectory reader가 다시 해석하지 않고 saved arrays를 검증하게 한다. 미실행 scheduler는 호환이라 단정하지 않는다.

다음 경계는 NaN과 OOM playbook 연결이다. NaN은 input latent/noise, model output, weighted numerator, guidance 결합과 solver state 경계에서 찾는다. [NaN playbook](../playbooks/01-nan.md)에 timestep, sigma, SNR weight, first non-finite layer와 offending RepresentationID를 넘긴다. high guidance와 half precision overflow를 분리한다.

OOM은 latent shape, cond/uncond batch doubling, attention, VAE decode와 solver history를 분해한다. [OOM playbook](../playbooks/05-oom.md)으로 peak range와 allocator snapshot을 넘긴다. resolution/batch를 줄이면 objective와 throughput denominator가 달라졌음을 기록한다.

negative tests는 extreme sigma와 guidance를 작은 synthetic model에 넣어 finite guard가 어느 경계에서 실패하는지 본다. production prompt/image를 사용하지 않는다. fix 뒤 same trajectory checksum과 quality sentinel을 재실행한다.

이 흐름에서 trajectory 완료 패키지도 빠뜨릴 수 없다. 패키지는 RepresentationID, model/condition, initial noise/RNG, prediction type, timestep/sigma/weight, step별 model input/output와 solver latent digest를 가진다. raw/EMA, guidance, VAE와 scheduler/실행 환경 리비전을 포함한다.

독립 검토자는 epsilon/v/flow scalar, weighted denominator, Euler/Heun과 guidance를 손계산한다. cond order, timestep shift, VAE scale과 missing solver history negative control이 실패해야 한다. step별 checksum만 있고 formula/config가 없으면 불완전하다.

23장에는 edit 전후 paired trajectory가 같은 initial/condition/scheduler를 쓰도록 이 package를 넘긴다. quality 차이가 representation, model edit 또는 solver drift 중 어디서 처음 생겼는지 찾을 수 있다.

이제 살펴볼 것은 timestep sampling 분포이다. uniform integer timestep과 logit-normal/continuous time sampling은 학습 기여가 다르다. expected distribution과 실제 sampled histogram, RNG state를 기록한다. bucket별 raw MSE, weight와 weighted contribution을 보면 특정 시간대가 objective를 지배하는지 알 수 있다.

1,000 timestep에서 10k sample이면 bucket noise가 있으므로 exact 균등을 요구하지 않고 interval/seed를 본다. 그러나 rank 하나가 같은 seed로 동일 timestep sequence를 복제하면 global diversity가 줄어든다. rank-specific generator와 resume state를 확인한다.

negative control은 sampler config만 바꾸고 RunID를 재사용한다. manifest diff가 새 trajectory/objective로 분기해야 한다. W&B 평균 loss가 비슷하다고 같은 run으로 합치지 않는다.

EMA 선택 사건부터 살펴보자. training raw model은 좋아지지만 inference sample이 달라졌다면 pipeline이 EMA와 raw 중 무엇을 load했는지 digest를 본다. EMA decay, update frequency, skipped optimizer step과 checkpoint state를 기록한다. weight 파일 이름 `ema`만 믿지 않는다.

one-parameter toy에서 raw가 1→3, decay 0.9이면 EMA update convention에 따른 값을 손계산한다. bias correction/warm-up과 update order가 구현마다 다를 수 있다. 소스 심볼과 fixture를 연결한다.

negative control은 raw model과 EMA config를 mismatch해 load한다. startup manifest가 거부하고 evaluation table이 model variant를 별도 행으로 만들어야 한다. merge/quant/export에도 exact parent를 넘긴다.

다음 경계는 condition cache 사건이다. prompt text는 같지만 tokenizer, text encoder, clip-skip 또는 negative prompt가 바뀌면 condition embedding이 달라진다. cache key에 prompt bytes, template/token IDs, encoder digest, layer selection, dtype와 attention mask를 넣는다.

stale cache로 cond는 old, denoiser는 new일 때 output은 생성되지만 experiment가 혼합된다. condition checksum을 step 0 trajectory에 기록한다. cache-off subset과 hit result를 비교해 key completeness를 검증한다.

[tokenizer mismatch playbook](../playbooks/04-tokenizer-mismatch.md)으로 raw/token/embedding first divergence를 찾는다. negative control은 same prompt alias 아래 encoder revision만 바꾼다. cache hit가 나면 실패다.

이 흐름에서 training resume trace도 빠뜨릴 수 없다. 연속 100 step과 50 step checkpoint-resume를 비교한다. model/optimizer/EMA, timestep/noise/condition-drop RNG, data cursor와 accumulation 위치를 저장한다. resume 첫 batch의 RepresentationID, timestep, noise checksum과 loss contribution이 같아야 한다.

noise만 달라도 final loss가 비슷할 수 있다. sample-exact 요구에서는 trajectory ledger가 first divergence를 잡는다. 비결정적 kernel은 tensor tolerance와 RNG equality를 분리한다. checkpoint marker 없는 generation을 load하지 않는다.

[partial checkpoint playbook](../playbooks/09-partial-checkpoint.md)으로 missing EMA/scheduler/RNG를 주입한다. weights-only resume는 child experiment이며 동일 RunID를 이어 쓰지 않는다.

## 22.7 evaluation과 prediction target의 공동 인수 조건을 세운다

생성 품질, condition 충실도, trajectory 안정성, 안전과 성능을 분리하고 target/scheduler mismatch를 먼저 차단한다.

샘플이 그럴듯하다는 관찰만으로 prediction target, scheduler와 resume 경로를 인수할 수 없다. 어떤 basis를 예측했는지, timestep별 gradient 기여가 무엇인지, sampler가 그 출력을 어떤 식으로 해석했는지를 함께 검증해야 한다. 다음 절부터 품질 점수를 내기 전에 닫아야 할 수치 계약을 prediction target과 SNR weighting에서 시작해 추적한다.

### prediction target을 scheduler와 함께 고정한다

### epsilon·sample·velocity는 같은 출력 이름이 아니다

Gaussian forward process를 (x_t=\alpha_t x_0+\sigma_t\epsilon)로 쓰자. 모델이 (epsilon)을 예측하면 (hat x_0=(x_t-\sigma_t\hat\epsilon)/\alpha_t)로 복원한다. sample prediction은 (hat x_0)를 직접 내고, velocity는 convention에 따라 (v=\alpha_t\epsilon-\sigma_t x_0) 같은 회전 좌표를 낸다. 세 target은 정보상 변환 가능해도 loss weighting과 수치 조건이 다르다.

Diffusers 고정 revision `d57cecde92a6d396845ab35425aa27469dff8173`의 [`DDPMScheduler.step`](https://github.com/huggingface/diffusers/blob/d57cecde92a6d396845ab35425aa27469dff8173/src/diffusers/schedulers/scheduling_ddpm.py)은 `prediction_type`에 따라 model output을 (x_0)로 변환하고 posterior mean과 variance branch를 계산한다. 같은 파일의 `add_noise`는 학습 시 (x_t)를 만든다. 학습 target과 sampling converter가 같은 convention인지 반드시 맞춘다.

checkpoint가 epsilon으로 학습되었는데 scheduler만 v-prediction으로 바꾸면 코드가 실행될 수 있다. shape도 맞는다. 그러나 model output을 잘못된 basis로 해석해 trajectory가 즉시 틀어진다. config validation은 `prediction_type`, alpha/sigma schedule, timestep scaling, latent scaling을 checkpoint metadata와 비교해 fail closed해야 한다.

### SNR weighting은 어느 timestep의 gradient를 키울지 정한다

단순 MSE (\|\epsilon-\hat\epsilon\|^2)도 timestep sampling 분포 때문에 특정 SNR 영역을 더 자주 본다. min-SNR weighting이나 p2 weighting은 쉬운 고-SNR step이 gradient를 독점하는 것을 줄이려는 정책이다. lambda 이름만 기록하지 말고 timestep bin별 raw loss, weighted loss, gradient norm을 본다.

작은 fixture에서 동일 (x_0,\epsilon)을 두 timestep에 놓고 weight 전후 기여를 계산한다. batch mean 전에 spatial/channel 평균이 어디서 이루어지는지도 확인한다. loss reduction 순서가 다르면 같은 weight 공식이 다른 분모를 갖는다.

classifier-free guidance는 학습 dropout과 추론 결합의 계약이다. 조건 (c)를 일정 확률로 null condition으로 바꾸어 unconditional branch를 학습한다. 추론에서는 (hat y=\hat y_u+s(\hat y_c-\hat y_u))로 결합한다. guidance scale을 키우면 조건 적합도가 올라갈 수 있지만 vector field를 학습 분포 밖으로 외삽해 saturation과 artifact가 생긴다.

condition dropout probability, null embedding, negative prompt encoding이 checkpoint 계약이다. inference 코드만 두 번 forward한다고 unconditional model이 생기지 않는다. conditional/unconditional batch를 concat할 때 sample 순서와 timestep 복제가 맞는지 unit test한다.

여기에는 서로 다른 세 상태를 섞지 않는다. 학습의 **condition dropout**은 표본 `i`의 조건을 null representation으로 대체해 unconditional 구간에도 gradient를 주는 Bernoulli 사건이다. 추론의 **null branch**는 그 학습된 입력 규약을 재현한다. **negative prompt branch**는 의미를 가진 별도 조건이며 null과 같지 않다. 빈 문자열도 tokenizer special token과 attention mask를 지나므로 “zero tensor”라고 가정하지 않는다. dropout probability만 저장하지 말고 microstep별 realized count, drop mask checksum, null token·embedding digest를 남긴다.

분리 실험은 네 행이면 된다. `p_drop=0`에서는 unconditional branch가 학습되었다고 주장하지 않는다. `p_drop=1`에서는 서로 다른 caption이 동일 null condition tensor가 되는지 보되 데이터 augmentation과 timestep은 고정한다. inference에서 null과 negative prompt embedding을 교환해 첫 conditional difference가 condition encoder 출력에서 시작하는지 확인한다. CFG concat 경로에서는 `[uncond_0..B-1, cond_0..B-1]`와 duplicated timestep·latent의 행 대응을 의도적으로 뒤섞어 golden assertion이 잡는지 본다. 마지막 image의 선호도보다 첫 denoiser input/output pair가 먼저다.

solver를 수치해석기로 읽는다. Euler의 한 step에는 상태 convention이 숨어 있다. Euler update는 (x_{i+1}=x_i+h_if(x_i,t_i))다. 쉬워 보이지만 Diffusers의 sigma scheduler에서는 model input을 (x/\sqrt{\sigma^2+1})로 scale한 뒤 denoised estimate에서 derivative를 만든다. [`EulerDiscreteScheduler.scale_model_input`](https://github.com/huggingface/diffusers/blob/d57cecde92a6d396845ab35425aa27469dff8173/src/diffusers/schedulers/scheduling_euler_discrete.py)과 같은 파일의 `step`을 떨어뜨려 읽으면 이 계약을 놓친다.

step index, sigma_hat, churn noise, predicted (x_0), derivative, next sigma를 trace로 남긴다. `scale_model_input` 호출을 누락한 고장 주입은 실행되지만 품질이 무너지는 대표 사례다. scheduler가 `is_scale_input_called` 같은 guard를 제공해도 custom loop가 경고를 무시하지 않게 테스트한다.

Heun은 함수 평가를 더 써서 국소 오차를 줄인다. Euler predictor로 (	ilde x=x_i+h f(x_i,t_i))를 만들고 다음 위치의 vector field를 평가한 뒤 (x_{i+1}=x_i+\frac h2[f(x_i,t_i)+f(\tilde x,t_{i+1})])로 보정한다. [`HeunDiscreteScheduler.step`](https://github.com/huggingface/diffusers/blob/d57cecde92a6d396845ab35425aa27469dff8173/src/diffusers/schedulers/scheduling_heun_discrete.py)은 이전 derivative와 dt를 상태로 저장하며 predictor/corrector 호출이 번갈아 일어난다.

따라서 resume checkpoint를 solver interval 중간에 찍으면 `prev_derivative`, `dt`, step index까지 저장해야 한다. 단순 timestep만 복원하면 corrector가 predictor로 바뀐다. 함수 평가 수(NFE)를 step 수와 구분해 latency를 비교한다.

DPM-Solver multistep은 history가 모델 상태다. 고차 multistep solver는 이전 model output history를 사용한다. [`DPMSolverMultistepScheduler.step`](https://github.com/huggingface/diffusers/blob/d57cecde92a6d396845ab35425aa27469dff8173/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py)은 history를 밀고 1·2·3차 update와 마지막 lower-order fallback을 선택한다. scheduler object를 새로 만들고 현재 timestep만 맞추는 resume는 동일 trajectory가 아니다.

짧은 step에서는 높은 차수가 항상 낫지 않다. vector field가 guidance 때문에 거칠거나 schedule 간격이 불균일하면 history error가 증폭될 수 있다. 동일 NFE와 동일 seed로 solver를 비교하고 terminal behavior와 thresholding을 함께 고정한다.

## 22.8 flow·discrete diffusion·training loop를 state transition으로 비교한다

연속 velocity field와 categorical Markov transition이 batch, target과 backward에서 요구하는 state를 공통 표에 놓는다.

### 직선 conditional path에서 target을 손으로 만든다

noise (x_0\sim p_0), data (x_1\sim p_1)를 뽑고 (x_t=(1-t)x_0+tx_1)로 잇는다면 conditional velocity는 (u_t=x_1-x_0)다. 모델은 (v_\theta(x_t,t,c))가 이 velocity의 조건부 평균을 근사하도록 MSE를 학습한다. 개별 점 쌍은 직선이지만 marginal density의 전체 흐름은 단순 점대점 매칭이 아니다.

기하학적으로 모델은 위치 (x_t)와 시간 (t)에서 접벡터를 낸다. sampling은 (dx/dt=v_\theta(x,t,c))를 적분한다. DDPM posterior 공식을 그대로 적용하지 않는 이유다. endpoint convention이 noise→data인지 data→noise인지 repository마다 확인한다.

[`FlowMatchEulerDiscreteScheduler.set_timesteps`](https://github.com/huggingface/diffusers/blob/d57cecde92a6d396845ab35425aa27469dff8173/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)는 sigma/timestep과 shift, terminal을 구성하고 `step`은 velocity에 (dt)를 곱한다. dynamic shift, Karras/exponential/beta schedule은 같은 vector field를 어디서 촘촘히 평가할지 바꾼다.

### rectification과 coupling이 path 난이도를 바꾼다

독립 coupling은 noise와 data를 무작위로 짝짓는다. optimal-transport에 가까운 coupling은 평균 path 길이와 교차를 줄일 수 있다. rectified flow는 학습된 trajectory에서 새 pair를 만들며 더 곧은 흐름을 노린다. “flow matching은 직선”이라는 설명은 conditional interpolation과 실제 learned marginal path를 구분해야 한다.

평가에는 endpoint 품질뿐 아니라 path curvature, velocity norm, solver 간 민감도, NFE-품질 곡선을 넣는다. curvature가 큰 구간에서 adaptive step 또는 더 촘촘한 schedule이 이득일 수 있다.

### 실제 Transformers 모델에서 출력 공간을 확인한다

Transformers commit `36deb0b53ed0863f4b4dfdea23dcaec7f3df3701`의 Qwen2.5-Omni [`modeling_qwen2_5_omni.py`](https://github.com/huggingface/transformers/blob/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py)는 Token2Wav DiT가 noised mel과 timestep, speech code, speaker/reference 조건으로 velocity를 예측하고 RK4로 mel trajectory를 적분하는 경로를 갖는다. 그 뒤 BigVGAN은 waveform을 내는 deterministic vocoder다. 전체 음성 생성기를 diffusion이라고 한 덩어리로 부르면 경계가 흐려진다.

PI0 계열은 이미지가 아니라 action trajectory에 flow matching을 적용할 수 있다. 중요한 질문은 “diffusion 모델인가”가 아니라 상태 공간, condition, time convention, prediction target, integrator가 무엇인가다.

discrete diffusion의 확률공간을 분리한다. mask absorbing process와 uniform replacement는 다르다. mask diffusion에서는 token이 시간에 따라 `[MASK]`로 이동하고 reverse model이 원 token 분포를 예측한다. uniform corruption에서는 vocabulary의 임의 token으로 바뀐다. 전자는 corrupted 위치가 보이고, 후자는 입력 token이 진짜인지 noise인지 직접 알 수 없다. transition matrix와 loss weighting이 다르다.

[`DiscreteDDIMScheduler.step`](https://github.com/huggingface/diffusers/blob/d57cecde92a6d396845ab35425aa27469dff8173/src/diffusers/schedulers/scheduling_discrete_ddim.py)은 categorical (x_0)와 clean/stay/uniform route를 sampling한다. 같은 파일의 `step_correct`는 leave-one-out logits와 선택된 좌표의 Gibbs식 resampling을 수행한다. continuous DDIM의 eta를 그대로 이 코드에 대입해서는 안 된다.

confidence-ranked refinement는 token별 clock을 만든다. [`BlockRefinementScheduler.step`](https://github.com/huggingface/diffusers/blob/d57cecde92a6d396845ab35425aa27469dff8173/src/diffusers/schedulers/scheduling_block_refinement.py)은 confidence 순으로 이번 step에 확정할 위치 수를 정하고 나머지를 mask 또는 uniform 상태로 둔다. 모든 위치가 같은 global timestep에 있어도 실질적으로는 확정 시각이 다르다.

confidence가 calibration되지 않으면 쉬워 보이는 오답을 먼저 고정해 뒤 step이 복구하지 못한다. entropy, margin, acceptance age를 token별로 trace한다. 마지막 step에서 모두 commit되는지 terminal invariant를 검사한다.

DiffusionGemma는 바깥 AR와 안쪽 denoising을 결합한다. 같은 Transformers revision의 [`generation_diffusion_gemma.py`](https://github.com/huggingface/transformers/blob/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py)는 canvas block을 바깥에서 autoregressive하게 늘리고, 각 block 내부를 반복 denoising한다. entropy-bound sampler는 낮은 entropy prefix를 받아들이고 나머지를 uniform token으로 다시 noise화한다.

[`modeling_diffusion_gemma.py`](https://github.com/huggingface/transformers/blob/36deb0b53ed0863f4b4dfdea23dcaec7f3df3701/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py)의 현재 forward는 logits와 self-conditioning graph를 제공하지만 labels와 training loss가 없다. inference 구현을 보았다고 pretraining corruption과 loss가 공개되었다고 확대하지 않는다. 이 negative boundary가 코드 독해의 일부다.

학습 loop를 함수와 state로 고정한다. 다음 경계는 latent diffusion의 한 microstep이다. 전형적인 text-to-image 학습은 image를 VAE latent로 encode하고 scaling factor를 곱한다. timestep과 noise를 뽑아 scheduler `add_noise`로 (x_t)를 만든다. text encoder hidden state와 (x_t,t)를 denoiser에 넣고 prediction type에 맞는 target과 MSE를 계산한다. accelerator accumulation 경계에서 backward, clip, optimizer, scheduler, zero_grad를 수행한다.

이 호출 사슬에서 VAE posterior sampling seed, latent scaling, text dropout, noise, timestep이 모두 RNG state다. resume first-divergence trace에 각각의 checksum을 넣는다. 한 값이라도 다르면 최종 image 비교는 원인을 알려 주지 않는다.

trainable scope가 objective를 바꾼다. UNet/DiT 전체, LoRA adapter, text encoder, VAE를 어느 조합으로 학습하는지에 따라 gradient 경로가 다르다. VAE를 동결해도 posterior sample은 stochastic할 수 있다. text encoder를 동결해도 tokenizer와 prompt dropout은 condition 분포를 바꾼다.

optimizer parameter group을 이름 glob으로 만들 때 새 module이 누락되거나 frozen parameter가 들어갈 수 있다. 첫 step 전에 trainable parameter 이름·shape·dtype·group을 manifest로 출력하고 expected set과 비교한다. LoRA merge/unmerge 상태도 checkpoint와 evaluation에서 고정한다.

이제 살펴볼 것은 distributed timestep sampling의 함정이다. 각 rank가 독립 timestep을 뽑으면 전체 batch 분포는 기대상 맞지만 작은 batch에서는 rank별 SNR 편차가 크다. timestep bin별 loss와 gradient를 all-reduce 전에 기록한다. loss-aware sampler를 쓰면 sampler histogram과 update state도 checkpoint해야 한다.

고장 주입은 한 rank의 timestep dtype이나 scaling을 바꾸는 것이다. shape가 같아 collective는 성공하지만 gradient가 서로 다른 objective의 평균이 된다. golden batch에서 rank별 timestep과 target checksum을 수집한다.

## 22.9 옵션·precision·quantization 오류를 trajectory 최초 차이로 줄인다

최종 이미지나 문장 차이가 아니라 time draw, corrupted input, target, block output와 solver state의 첫 불일치를 찾는다.

NaN은 model output 이전과 이후로 나눈다. 먼저 input latent와 scaled input이 finite인지, model output이 finite인지, (x_0) 변환과 thresholding 뒤가 finite인지, variance와 noise 뒤가 finite인지 순서대로 검사한다. fp16 overflow, sigma=0 division, guidance scale, VAE scaling을 각각 분리한다. 최종 black image만 보고 원인을 추측하지 않는다.

품질 회귀는 동일 noise에서 비교한다. baseline과 candidate가 동일 initial noise, condition embedding, timestep grid를 쓰게 한다. step별 latent cosine/L2, predicted (x_0), velocity norm을 비교해 첫 divergence를 찾는다. scheduler 변경이라면 같은 timestep index가 같은 sigma를 뜻하지 않으므로 sigma/time 값으로 정렬한다.

이 흐름에서 23장으로 넘기는 artifact도 빠뜨릴 수 없다. 23장의 concept editing과 unlearning은 특정 개념을 지운 뒤 생성 trajectory가 어떻게 변했는지 평가해야 한다. 이 장은 `TrajectoryID`, initial noise, condition, model/VAE/text revisions, prediction target, scheduler config, timestep grid, step별 checksum, terminal output을 넘긴다. seed 하나의 결과가 아니라 여러 seed에서 target concept와 locality concept의 분포를 넘긴다.

최종 인수 조건은 checkpoint metadata와 scheduler 계약 일치, solver history를 포함한 resume, 동일 NFE 비교, timestep-bin loss 계측, discrete/continuous state 구분, 최초 trajectory divergence 출력이다. 이 여섯 조건이 닫혀야 sampler 옵션을 “품질 knob”가 아니라 검증 가능한 수치 적분 선택으로 다룰 수 있다.

### sampler 옵션을 상태 변화로 번역한다

`num_inference_steps`는 단순 품질 숫자가 아니다. step 수를 늘리면 vector field를 더 자주 평가하지만 schedule 생성 규칙과 solver order가 함께 작동한다. 20-step Euler와 20-step Heun은 NFE가 다를 수 있고, DPM multistep은 초기 history 구축과 마지막 fallback 때문에 유효 차수가 구간마다 다르다. latency 비교는 step이 아니라 실제 denoiser forward 횟수로 한다.

step 수를 바꿀 때 sigma/timestep grid 전체를 저장한다. 같은 scheduler 이름도 `timestep_spacing`, terminal sigma, shift 옵션에 따라 grid가 달라진다. 품질 차이가 solver인지 grid인지 분리하려면 한 요소씩 고정한다.

`eta`, churn, ancestral noise는 RNG 위치를 바꾼다. DDIM의 eta=0은 주어진 initial noise에서 결정적 경로를 만들고 eta>0은 posterior variance를 주입한다. Euler ancestral이나 churn은 다른 위치에서 noise를 넣는다. “seed를 같게 했다”는 문장만으로 동일 stochastic process가 아니다. 어느 step에서 어떤 shape의 random tensor를 뽑았는지 trace한다.

resume는 RNG state와 solver state를 함께 복원해야 한다. noise draw 직전/직후 중 어디서 checkpoint했는지 명시한다. 중간 step 저장이 필요 없다면 지원하지 않는다고 fail closed하고 항상 interval boundary에서만 저장한다.

guidance와 rescale은 vector field 크기를 바꾼다. guidance scale이 클수록 conditional-unconditional 차이가 커진다. rescale이나 dynamic thresholding은 폭주를 줄이지만 원 vector field를 변형한다. step별 conditional norm, unconditional norm, guided norm, predicted (x_0) range를 기록한다. 이미지 saturation만 보고 마지막 decoder를 탓하지 않는다.

negative prompt는 null condition과 다르다. tokenizer truncation과 text encoder pooling까지 포함한 embedding checksum을 저장한다. prompt weighting extension을 쓴다면 공식 pipeline 밖의 변환임을 명시한다.

### quantization과 low precision이 trajectory에 누적되는 방식

denoiser 오차는 step마다 다시 입력된다. autoregressive token generation처럼 이전 discrete choice만 돌아오는 것이 아니라 diffusion에서는 작은 denoiser 오차가 다음 continuous state 전체에 들어간다. (e_i)가 update에 (h_ie_i)로 주입되고 이후 Jacobian에 의해 증폭될 수 있다. timestep별 calibration이 필요한 이유다.

quantization calibration sample을 최종 clean latent만으로 만들면 noisy high-sigma activation range를 놓친다. timestep stratified calibration을 하고 block별 activation range와 output error를 측정한다. Q-Diffusion 고정 snapshot `715783da70baa267321d6700ceb8941400c309d1`의 [repository](https://github.com/Xiuyu-Li/q-diffusion/tree/715783da70baa267321d6700ceb8941400c309d1)는 timestep-aware calibration 계보를 읽는 출발점이다.

text encoder·VAE·denoiser를 따로 양자화한다. condition embedding 오차, vector field 오차, latent encode/decode 오차는 영향 경로가 다르다. component를 하나씩 양자화해 동일 initial noise에서 first-divergence를 측정한다. 최종 FID 하나로 어느 component가 실패했는지 알 수 없다.

mixed precision에서는 scheduler arithmetic을 fp32로 유지할지 확인한다. alpha cumulative product, sigma difference, variance가 fp16에서 underflow할 수 있다. model output은 low precision이어도 state update와 threshold 통계를 fp32로 올리는 구현이 많다.

QAT·LoRA distillation의 target을 분리한다. EfficientDM snapshot `293e212b8c62c356a6da44d6550cd296c87b9f63`의 [repository](https://github.com/ThisisBillhe/EfficientDM/tree/293e212b8c62c356a6da44d6550cd296c87b9f63)에는 diffusion quantization-aware LoRA/distillation 계보가 정리돼 있다. teacher의 noise prediction, feature, 최종 sample 가운데 어느 것을 맞추는지 확인한다. teacher scheduler와 student scheduler가 다르면 같은 timestep index의 target을 비교해서는 안 된다.

## 22.10 종단 실험·고장 주입·운영 인수를 하나의 trajectory로 닫는다

quality metric, distribution shift, condition, compute와 safety를 golden trajectory와 failure injection에 묶는다.

품질·다양성·조건 충실도는 서로 다른 축이다. FID는 feature distribution의 평균·공분산 차이를 요약하지만 prompt별 충실도를 직접 보지 않는다. CLIP 계열 score는 text-image alignment proxy지만 rendering artifact와 세부 count에 약할 수 있다. precision/recall, human preference, task-specific detector를 함께 본다.

seed를 고정한 paired 비교와 seed distribution 비교를 모두 수행한다. paired 비교는 trajectory 차이를 찾고, distribution 비교는 sampling 품질을 본다. sample 수와 confidence interval을 보고한다. 좋은 이미지 몇 장을 고르는 것은 평가가 아니다.

discrete text diffusion은 compute budget을 맞춘다. AR 모델은 생성 token마다 forward하고 diffusion model은 canvas 전체를 여러 step 갱신한다. latency만 비교하면 batch/compile 차이가 섞인다. FLOPs, NFE, 생성 token, wall time, peak memory를 함께 둔다. adaptive stopping은 쉬운 행에 계산을 덜 쓰므로 품질-비용 곡선을 행 난이도별로 본다.

정답률뿐 아니라 확정 순서와 entropy calibration을 관찰한다. 초기 오답이 높은 confidence로 고정되는지, renoise가 복구하는지 trace한다. 마지막 output만 보면 refinement mechanism을 평가할 수 없다.

안전과 권리 검사는 condition family를 분리한다. 금지 concept, 실제 인물, 스타일 모방, 개인정보, 폭력 등 category별 prompt family를 만든다. concept erasure 뒤 target 감소와 인접 정상 개념 손실을 함께 본다. prompt exact match만 쓰면 paraphrase generalization을 놓친다.

### 종단 실험과 고장 주입

2차원 Gaussian 손실험부터 살펴보자. 2차원 noise와 네 점 data mixture로 작은 velocity model을 학습하거나 analytic vector field를 만든다. Euler와 Heun을 같은 grid에서 적분해 trajectory와 global error를 그린다. step을 절반으로 줄였을 때 Euler error가 대략 1차, Heun이 2차로 줄어드는 구간을 확인한다. stiffness가 큰 field에서는 이 단순 관계가 깨지는 것도 본다.

다음 경계는 pipeline golden trace이다. 작은 tensor, 두 timestep, 고정 noise를 사용한다. `add_noise`, model input scale, target, model output converter, solver update의 기대값을 손으로 계산한다. epsilon/v/sample 세 parameterization fixture를 별도로 둔다. scheduler 이름만 바꾼 채 checkpoint target을 유지하는 실패를 test가 잡아야 한다.

이 흐름에서 운영 결정 트리도 빠뜨릴 수 없다. 검은 출력이면 VAE scaling→initial noise→condition embedding→model output range→scheduler update→decoder 순서로 본다. seed 재현이 안 되면 generator owner→noise draw 위치→distributed rank→solver history 순서다. 품질은 같고 느려졌다면 NFE→compile shape→CFG concat→offload/H2D 순서다. concept erasure side effect면 target prompt family→neighbor family→text encoder 변화→denoiser layer delta 순서다.

이제 살펴볼 것은 인계 가능한 완료 패키지이다. model/condition/VAE revision, prediction target, schedule, solver, RNG draw ledger, timestep별 state checksum, NFE와 runtime, quality/safety row contribution을 묶는다. 23장은 이 패키지의 model delta를 바꾸고 같은 trajectory fixture를 재실행한다. 24장은 metric 분모와 uncertainty를 검증한다. 이 연결이 닫혀야 diffusion fine-tuning 결과를 이미지 모음이 아니라 재현 가능한 실험으로 다룰 수 있다.

최종 image quality metric은 scheduler/seed/decoder가 섞인 결과다. 같은 initial noise와 condition에서 old/new model trajectory를 pair하고 최초 model-output divergence를 기록한다. 다른 seed 평균만 비교하면 variance가 커진다.

VAE decode를 바꾸고 denoiser를 고정한 control, scheduler를 바꾸고 model을 고정한 control을 둔다. metric이 어느 변화에 민감한지 calibration한다. prompt family와 sample count, failed generation denominator를 보존한다.

perceptual metric이 좋아도 prompt alignment나 safety가 악화될 수 있다. component와 human audit를 분리한다. 공개 metric implementation과 encoder revision을 소스 원장에 둔다.

### 운영 인수와 최종 trajectory

### 지원 scheduler matrix

행에는 prediction type, scheduler/solver, timestep spacing, guidance, VAE, raw/EMA와 dtype/runtime를 둔다. 열에는 scalar fixture, save/load, trajectory replay, quality/performance와 negative test를 둔다. 이름이 같은 scheduler도 config가 다르면 다른 행이다.

epsilon Euler에서 검증한 결과를 velocity multistep에 상속하지 않는다. multistep은 history, adaptive는 tolerance/acceptance state를 추가한다. video diffusion은 temporal latent/noise correlation과 frame condition을 추가한다.

독립 검토자는 표에서 임의 행의 formula와 first step을 손계산하고 artifact reader로 arrays/digest를 확인한다. 미실행 조합은 unsupported/unverified로 표시한다.

독자의 최소 실습부터 살펴보자. 1D scalar x0/noise로 epsilon, velocity와 flow target을 계산하고 framework output과 비교한다. 4-step Euler와 Heun trajectory를 저장해 독립 reader로 재계산한다. timestep shift, cond order, VAE scale와 missing history를 하나씩 깨뜨린다.

작은 denoiser에서 continuous/resume noise checksum을 비교하고 EMA raw variant를 분리한다. GPU가 없어도 float64 CPU fixture로 수식과 state owner를 검증할 수 있다. perceptual quality는 미실행으로 남긴다.

보고에는 소스 좌표, numeric table, negative assertion, trajectory digest와 지원 matrix를 둔다. 이 패키지가 통과해야 production model의 느리고 비싼 sample을 해석할 기준선이 생긴다.

다음 경계는 latent trajectory first divergence이다. old/new run의 final image만 다르면 step 0 representation/condition/noise부터 비교한다. 셋이 같다면 각 step의 model input, prediction과 scheduler output digest를 순서대로 본다. step 7 prediction부터 다르면 model/weight/kernel, prediction은 같고 scheduler output만 다르면 solver/config 가설이다.

selected tensor slice와 norm, dtype를 저장해 digest mismatch를 수치로 해석한다. chaotic amplification 때문에 final 차이가 커져도 최초 작은 divergence를 찾는다. stochastic sampler는 RNG counter와 injected noise를 step별로 둔다.

negative control은 final decoded image를 같게 만들도록 postprocess를 고정하되 중간 trajectory 하나를 바꾼다. final-only test가 놓치고 trajectory ledger가 잡아야 한다. 반대로 harmless serialization metadata 차이는 tensor digest와 구분한다.

이 흐름에서 performance denominator도 빠뜨릴 수 없다. generation latency는 prompt encode, denoiser NFE, VAE decode와 safety/postprocess를 분리한다. image/s, denoise step/s와 model evaluation/s는 guidance batch와 solver에 따라 다르다. 같은 “20 steps”라도 NFE가 다르면 직접 비교하지 않는다.

resolution, batch, dtype, attention backend와 GPU clock을 고정하고 warm-up/compile을 제외한 steady 반복을 낸다. quality와 latency를 같은 seed/prompt set에서 pair한다. OOM/failed generation을 처리량 분모에서 숨기지 않는다.

negative control은 solver step 수는 같지만 Heun NFE를 두 배로 만든다. step/s만 보면 같은 듯 보여도 model-call ledger와 GPU time이 차이를 보여야 한다. profiler overhead run은 발표 성능과 분리한다.

이제 살펴볼 것은 diffusion safety와 provenance이다. condition prompt, source image와 generated artifact는 safety/privacy 이슈를 가질 수 있다. training/evaluation media lineage와 license, private prompt 접근을 통제한다. generated image 자체뿐 아니라 latent/embedding artifact도 민감할 수 있다.

safety filter와 watermark는 denoiser 밖의 postprocess edge다. on/off에 따라 output score가 달라지므로 model quality와 서비스 policy를 분리한다. filter-only change를 unlearning이나 model safety 개선으로 쓰지 않는다.

artifact에는 denoiser trajectory와 postprocess digest를 함께 두되 독립 edge로 연결한다. RevocationID가 VAE/condition encoder/denoiser/export cache descendant에 전파되는지 23·27장에서 확인한다.

5,000어절 중간 인수 조건부터 살펴보자. epsilon/v/x0/flow, SNR weighting, guidance와 Euler/Heun의 scalar fixture를 독립적으로 계산할 수 있어야 한다. scheduler 소스 좌표와 saved timestep/sigma/history가 일치해야 한다.

timestep shift, cond swap/cache, VAE scale, wrong EMA, missing solver state와 resume RNG negative control이 실패해야 한다. representation/model/scheduler/decode의 first divergence를 찾을 수 있어야 한다.

지원 matrix, performance denominator, safety/postprocess 경계와 미실행 quality/kernel을 기록한다. 이 조건을 통과한 NoiseTrajectoryID만 23장의 paired edit 비교에 사용한다.

다음 경계는 마지막 종합 trajectory이다. scalar latent `x0=2`, fixed noise -1과 four-step schedule을 시작으로 epsilon model과 Euler trajectory를 손계산한다. 같은 initial/condition에서 prediction type만 velocity로 바꾼 wrong run, timestep index를 하나 민 run과 condition order를 바꾼 run을 만든다. 모두 shape는 같지만 step 0 또는 첫 scheduler output에서 divergence해야 한다.

작은 tensor model에서는 raw와 EMA, checkpoint 2-step resume와 continuous 4-step을 비교한다. resume artifact가 model/optimizer/EMA, timestep/noise RNG와 solver state를 모두 복원해야 한다. missing history negative run은 loader 또는 next-step digest에서 실패한다.

마지막으로 동일 trajectory를 VAE scale 두 개로 decode하고 quality metric 변화가 denoiser edit가 아님을 확인한다. latency 보고는 NFE, denoiser/condition/VAE와 failed generation denominator를 분리한다. postprocess filter on/off도 별도 edge로 둔다.

독립 검토자는 saved arrays와 source formula를 사용해 trajectory를 package import 없이 다시 계산한다. mutable scheduler default를 적용하지 않는다. 합성 fixture 통과와 production GPU quality 미실행을 구분해 기록한다.

검토 결과는 representation, condition, model, scheduler, decode와 postprocess의 owner별로 first divergence를 표시한다. initial latent가 다르면 21장 handoff로 돌아가고, model input은 같지만 prediction이 다르면 weight/kernel, prediction은 같고 next latent가 다르면 solver를 본다. final image metric만 다른 경우 VAE와 postprocess를 조사한다.

fix 뒤에는 wrong prediction type, timestep shift, stale condition, scale mismatch와 missing history가 여전히 각각의 guard에서 실패하는지 재실행한다. golden trajectory가 통과했다는 사실만으로 negative detector를 면제하지 않는다. 승인 record에 scheduler arrays, formula source, model/VAE digest와 독립 계산 결과를 서명한다. 이 record가 paired edit의 parent가 된다.

독립 환경에서는 scheduler package cache와 pipeline alias를 사용하지 않고 manifest의 exact arrays를 읽는다. step별 model-call 수, guidance cond order와 dtype도 비교한다. GPU kernel이 달라 tolerance가 필요하면 첫 divergence와 반복 분포를 기록하고 수식·state mismatch를 tolerance로 숨기지 않는다. 최종 record의 reader version과 checksum도 보존해 검증기 변경을 trajectory 변경과 구분한다.

판정자와 재검토 시각, 지원 scheduler 행도 같은 manifest에 고정해 다음 실험이 정확한 증거 범위를 알게 한다.

이 장이 넘기는 것. `NoiseTrajectoryID`, initial noise checksum, timestep/sigma ledger, condition checksum, step별 latent digest를 23장과 평가 장에 넘긴다.

## 22.11 새 구현과 현장 incident를 source 질문으로 감사한다

repository에서 corruption, target, model, reduction, EMA와 scheduler owner를 찾고 실제 장애 네 건에 적용한다.

상태 공간과 endpoint. 첫 질문은 모델 이름이 아니라 (x_t)가 무엇인지다. pixel, VAE latent, Mel, action, atom coordinate, token ID 가운데 하나를 고른다. (t=0)과 (t=1) 중 어느 쪽이 data인지, terminal에서 어떤 invariant가 성립하는지 확인한다. 코드 변수 `sample`, `noise`, `latents`의 이름을 수식에 그대로 대입하지 않는다.

corruption 생성자. 학습 입력을 만드는 함수에서 data, noise, timestep, transition matrix를 찾는다. continuous라면 alpha/sigma와 broadcasting axis, discrete라면 mask/uniform transition과 special token을 본다. noise가 dataloader, trainer, scheduler 중 누가 생성하는지와 generator owner를 기록한다.

prediction target. forward 출력의 shape만 보지 말고 target을 만드는 branch를 찾는다. epsilon, (x_0), velocity, score, clean-token logits는 서로 다른 계약이다. 논문 수식과 코드 sign·scale·time convention을 작은 scalar fixture로 대조한다.

loss reduction. unreduced error의 channel/spatial/token 축을 어디서 평균내고 timestep weight를 언제 곱하는지 본다. mask된 token만 학습하는지 전체 vocabulary CE인지 확인한다. distributed mean이 rank별 유효 unit 차이를 올바르게 반영하는지도 계산한다.

condition 경로. text/image/audio/action condition이 encoder에서 어떤 dtype과 mask로 나오고 denoiser의 cross-attention·AdaLN·concat 중 어디에 들어가는지 찾는다. classifier-free dropout이 학습에 실제 존재하는지, null condition이 무엇인지 확인한다.

solver 변환. model output을 (x_0), derivative, posterior logits로 바꾸는 함수를 찾는다. model input scaling과 output conversion을 한 쌍으로 읽는다. custom loop가 pipeline helper를 우회할 때 빠지는 전처리가 없는지 negative test를 만든다.

solver history. multistep derivative, predictor/corrector phase, step index, lower-order flag를 목록화한다. serialization이 이 state를 지원하지 않으면 중간 resume를 지원한다고 쓰지 않는다. boundary-only checkpoint 정책을 명시한다.

RNG ledger. initial noise, posterior noise, churn, categorical sample, Gumbel selection, condition dropout을 서로 다른 stream으로 구분한다. 같은 seed가 아니라 같은 draw sequence와 tensor shape를 재현해야 한다. rank 수가 바뀌면 어떤 보장을 포기하는지 쓴다.

component revision. denoiser뿐 아니라 tokenizer/text encoder, VAE/codec, scheduler config, safety/postprocess를 고정한다. mutable model alias에서 받은 config를 checkpoint 내부 config보다 우선하지 않는다. load 시 hash mismatch를 경고로 넘기지 않는다.

trainable scope. full fine-tune, LoRA, control module, text encoder 가운데 실제 `requires_grad`와 optimizer membership을 덤프한다. EMA가 어떤 parameter를 추적하고 update가 optimizer step 뒤 몇 번 일어나는지 확인한다. gradient accumulation microstep에서 EMA가 갱신되면 decay 의미가 달라진다.

평가 분모. 품질 표본 수, seed, failed generation, NFE, latency stage를 보고한다. scheduler 20 step이라는 문자열로 compute를 맞췄다고 주장하지 않는다. guidance batch doubling과 adaptive stopping도 실제 model call 원장에 넣는다.

negative evidence. 현재 revision에 training loss가 없거나 특정 scheduler test가 없다면 명시한다. inference graph에서 관찰한 사실을 학습 recipe로 확대하지 않는다. model card, 논문, library 구현, 직접 실험은 서로 다른 근거 층이다.

### 현장 장애 네 건을 끝까지 추적한다

resume 뒤 색감이 달라졌다. 먼저 initial latent, prompt embedding, timestep array를 비교한다. 같다면 첫 model output과 scheduler output을 본다. 둘도 같고 final decode만 다르면 VAE·postprocess revision이다. EMA/raw weight 선택과 autocast dtype도 확인한다. 최종 이미지 hash부터 원인을 추측하지 않는다.

loss는 정상인데 sample이 무너진다. 학습 target과 sampler prediction type 불일치를 가장 먼저 검사한다. timestep normalization, latent scaling, condition dropout과 null embedding도 본다. training loss가 감소하는 것은 sampler가 그 출력을 올바르게 해석한다는 증거가 아니다.

step을 늘렸는데 품질이 나빠졌다. 새 grid가 학습 time distribution과 맞는지, solver가 terminal에서 overshoot하는지, guidance가 고차 history를 불안정하게 만드는지 본다. 동일 NFE와 동일 sigma range 대조군을 만든다. 단순히 더 많은 step이 항상 작은 오차를 뜻하지 않는다.

일부 rank만 NaN이다. rank별 timestep/SNR, input latent range, condition length, loss scale, found-inf를 비교한다. 한 rank의 극단 high-resolution sample이나 잘못된 mask가 원인일 수 있다. global reduced loss만 보면 NaN 발생 위치가 사라진다. offending `NoiseTrajectoryID`를 작은 단일 GPU fixture로 재생한다.

이 네 사건을 해결한 뒤에는 반드시 동일 fault를 다시 주입한다. prediction type mismatch, stale VAE, missing solver history, rank별 timestep scaling이 각각 의도한 guard에서 실패해야 한다. 수정 뒤 golden path 성공만 확인하면 detector가 아니라 우연히 증상을 피한 것일 수 있다.

### 이 장의 최종 판정

독자는 scheduler 이름을 보지 않고도 state, target, update equation, stochastic draw, terminal condition을 설명할 수 있어야 한다. 임의 생성물에서 step별 latent와 condition을 원 checkpoint까지 역추적하고, 같은 initial state에서 baseline과 candidate의 최초 불일치를 찾을 수 있어야 한다. solver 비용은 NFE로, 학습 기여는 timestep-bin loss와 gradient로, 품질은 seed distribution과 paired trace로 보고해야 한다.

이 판정이 닫히면 diffusion과 flow는 “노이즈를 지우는 모델”이라는 비유를 넘어선다. 학습된 vector field 또는 categorical reverse kernel, 이를 읽는 수치 적분기, condition과 decoder 공급망이 결합된 시스템으로 보인다. 23장은 이 시스템의 일부 weight나 concept를 바꿨을 때 어느 trajectory가 변하고 어느 이웃이 보존되는지를 같은 원장으로 검증한다.

승인 reviewer는 마지막으로 두 개의 독립 계산을 수행한다. continuous fixture에서는 저장된 alpha·sigma 또는 time grid만으로 (x_t), prediction 변환, next state를 package 없이 다시 계산한다. discrete fixture에서는 transition probability를 합산해 1이 되는지, terminal에서 mask나 미확정 위치가 남지 않는지 확인한다. 구현 결과와 수식 결과가 tolerance 밖이면 GPU kernel 오차라는 설명부터 받아들이지 않는다. convention과 broadcasting을 먼저 검사한다.

운영 metric에는 timestep별 denoiser norm과 latency, solver step index, guidance norm ratio, VAE range, failed sample을 넣는다. 평균 latency가 안정돼도 특정 sigma 구간의 kernel fallback이나 dynamic shape compile이 tail을 만들 수 있다. 장애 시 `NoiseTrajectoryID` 하나를 선택해 condition encode, initial noise, 모든 model call, scheduler state, decode를 재생한다.

지원 범위는 checkpoint마다 별도다. epsilon checkpoint에서 검증한 scheduler matrix를 flow checkpoint에 상속하지 않는다. 이미지 latent에서 통과한 precision 설정을 audio Mel이나 action trajectory에 일반화하지 않는다. state 공간과 magnitude가 달라 quantization·clipping·solver 안정성도 달라진다.

마지막 인계에는 성공 trace뿐 아니라 negative trace를 포함한다. wrong prediction type은 load gate, 누락된 input scaling은 첫 model call, stale condition은 condition checksum, missing history는 resume boundary, VAE scale mismatch는 decode boundary에서 각각 실패해야 한다. 이 위치가 고정되어야 23장의 편집 전후 차이가 학습된 지식 변화인지 실행 계약 drift인지 판별할 수 있다.

검증자는 각 trace의 첫 불일치 행에 state owner와 재현 명령을 붙인다. 재현 명령은 mutable pipeline alias 대신 저장된 config와 배열을 읽어야 하며, package default가 바뀌어도 같은 계산을 복원해야 한다. GPU가 달라 bitwise 비교가 불가능하면 허용 오차의 근거와 반복 분포를 먼저 고정한다. tolerance는 prediction type, timestep, condition 순서 같은 계약 오류를 덮는 장치가 아니다. 최종 보고서는 어떤 scheduler와 dtype, state 공간, guidance 범위를 실제 검증했는지와 어떤 경로를 실행하지 않았는지를 나란히 적는다. 범위 밖의 성공 주장은 새 golden trajectory 없이 승인하지 않는다.

승인 시각, reviewer, 모든 digest와 미실행 경로를 immutable record에 서명한다. 다음 실험은 이 record를 parent로 삼고 단 하나의 가설만 바꾼다. 그래야 품질 차이를 설정 drift가 아니라 의도한 학습 변화로 귀속할 수 있다.

## 22.12 forward process·DiT·training loop를 tensor graph로 해부한다

pixel/latent patch, condition, time embedding과 denoiser output을 transition 및 loss state에 연결한다.

확산 학습의 출발점은 깨끗한 데이터 `x_0`에서 무작위 시간 `t`의 오염된 상태 `x_t`를 만드는 일이다. Gaussian diffusion에서는 흔히 `x_t=α_t x_0+σ_t ε`, `ε~N(0,I)`로 쓴다. 이 식에서 schedule은 단순한 noise 양 목록이 아니다. 데이터 manifold를 어느 속도로 흐리게 관측할지 정하는 측정 장치다. `α_t²+σ_t²=1` convention인지, variance exploding인지, log-SNR을 어떤 방향으로 매기는지에 따라 같은 `t` 숫자의 의미가 달라진다.

훈련 batch에서 반드시 보존할 최소 난수 상태는 sample ID, timestep 또는 연속 시간, noise seed와 실제 noise tensor의 checksum이다. augmentation과 latent encoder에도 난수가 있으면 별도 stream을 쓴다. 하나의 global RNG에 의존하면 worker 수나 연산 순서가 바뀔 때 noise까지 달라져 두 실행의 first divergence가 데이터인지 확산인지 구분되지 않는다.

시간 sampling 분포 `p(t)`와 loss weight `w(t)`는 함께 목적함수를 만든다. 구현된 추정량은 `E_{t~p(t),ε}[w(t)ℓ_t]`다. uniform timestep에서 다른 sampler로 바꾸고 weight를 그대로 두면 “더 중요한 시간을 자주 본다”가 아니라 최적화하는 적분의 measure 자체가 바뀐다. importance correction을 하는지, 의도적으로 objective를 재가중하는지 명시한다.

log-SNR을 공통 좌표로 사용한다.

`λ(t)=log(α_t²/σ_t²)`는 signal과 noise의 상대 크기를 나타낸다. 서로 다른 schedule을 timestep index만으로 비교하면 같은 index가 전혀 다른 난이도일 수 있다. log-SNR 분위수에서 denoising error, gradient norm, sample 수를 비교하면 schedule 간 대응이 쉬워진다. `λ`가 큰 깨끗한 쪽과 작은 noise 쪽에서 수치 범위도 다르므로 dtype 오류를 찾는 축이 된다.

signal prediction, epsilon prediction, velocity prediction은 같은 조건에서 변환 가능하지만 loss weighting까지 자동으로 같아지지는 않는다. 예를 들어 epsilon과 `x_0` 사이 변환에는 `α_t`, `σ_t`로 나누는 항이 들어가 끝점에서 오차가 증폭될 수 있다. 모델 output을 바꾸고 scheduler config만 맞췄다는 이유로 학습 objective가 같다고 주장하지 않는다.

latent diffusion의 `x_0`는 pixel이 아니다. VAE encoder가 만든 latent를 scale factor로 조정한 값이 diffusion state다. pixel normalization, encoder posterior sampling 여부, latent scaling과 shift가 하나라도 달라지면 같은 이미지에서 다른 `x_0`를 얻는다. VAE를 frozen해도 processor와 encoder stochasticity는 입력 계약으로 남는다. 원 이미지 hash, normalized pixel 통계, posterior mean·sample, scale 뒤 latent checksum을 golden artifact로 둔다.

decoder가 출력 품질을 결정하므로 denoiser loss가 좋아져도 pixel metric이 나빠질 수 있다. VAE reconstruction floor를 먼저 측정하고 diffusion error와 decode error를 분리한다. 새로운 VAE로 교체하면 latent distribution과 denoiser target이 바뀌므로 단순한 export 최적화가 아니다.

### prediction target을 코드 분기까지 추적한다

Diffusers의 `sources/training-diffusers/src/diffusers/schedulers/scheduling_ddpm.py:461` 이후 `DDPMScheduler.step`은 `prediction_type`에 따라 model output을 `pred_original_sample`로 바꾼다. `epsilon`이면 `(x_t-√β̄_t ε̂)/√ᾱ_t`, `sample`이면 output 자체, `v_prediction`이면 `√ᾱ_t x_t-√β̄_t v̂`를 쓴다. 체크포인트가 학습한 target과 이 분기가 다르면 shape는 맞고 이미지도 나올 수 있지만 trajectory 의미는 틀린다.

같은 함수는 learned variance일 때 channel을 둘로 split하고, `thresholding`이나 clipping을 적용한 뒤 posterior mean을 계산한다. 따라서 output channel 수, variance type, prediction type, clipping은 독립 옵션이 아니다. 모델 head shape와 scheduler config, checkpoint metadata를 load 전에 교차 검증해야 한다.

이제 살펴볼 것은 세 target의 손계산 fixture이다. 1차원 `x_0=2`, noise `ε=-1`, `α=0.8`, `σ=0.6`을 두면 `x_t=1.0`이다. 정확한 epsilon 예측 `-1`에서 `x_0=(1-0.6(-1))/0.8=2`가 복원된다. velocity를 `v=αε-σx_0=-2.0`으로 정의하면 `αx_t-σv=2`다. 이 작은 수를 scheduler adapter와 training target 함수 양쪽의 test로 사용한다.

부호 convention이 다른 문헌이나 구현을 섞으면 velocity 값이 반대가 될 수 있다. 함수 이름보다 실제 변환식을 기록하고, exact target을 넣었을 때 `x_0`가 복원되는 round-trip을 요구한다. 끝점 가까이에서 작은 `α` 또는 `σ`로 나누는 경로는 fp32로 검산한다.

loss weighting을 gradient 크기로 확인한다. min-SNR weighting, p2 weighting, flow weighting은 timestep별 loss 기여를 바꾼다. 설정값만 기록하지 말고 log-SNR bucket별 sample count, unweighted error, weight, weighted loss, gradient norm을 기록한다. weight clamp가 적용되는 경계에서 broadcasting이 batch·channel·공간 중 어느 축인지 확인한다.

per-sample timestep인데 scalar loss로 먼저 평균한 뒤 weight를 곱하면 의도한 재가중이 되지 않는다. unreduced loss를 sample별로 공간·채널 평균하고 sample weight를 적용한 뒤 batch reduction하는 순서를 test한다. distributed 평균과 gradient accumulation denominator까지 식에 포함한다.

### flow matching을 경로와 속도장의 문제로 읽는다

flow matching에서는 기준 분포 `x_0`와 데이터 `x_1` 사이 conditional probability path `x_t`를 정하고 그 path의 속도 `u_t(x_t|x_0,x_1)`를 모델이 예측한다. 가장 단순한 직선 보간 `x_t=(1-t)x_0+t x_1`의 target velocity는 `x_1-x_0`다. 하지만 coupling이 독립 pairing인지 optimal transport 근사인지에 따라 같은 끝점 분포 사이에서도 개별 path의 교차와 곡률이 달라진다.

학습은 무작위 path 지점에서 local velocity를 맞추고 추론은 ODE solver로 그 field를 적분한다. 따라서 낮은 training MSE가 적은 step에서 좋은 trajectory를 보장하지 않는다. vector field의 Lipschitz 성질, solver step 크기, off-path error가 누적된다. training time bucket error와 inference local truncation error를 분리한다.

Euler step을 구현 상태와 연결한다. Diffusers의 `sources/training-diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py:423`의 `step`은 integer index 전달을 거부하고 실제 scheduler timestep을 요구한다. `_init_step_index`는 schedule에서 시작 위치를 잡고, sample을 fp32로 올려 update의 정밀도를 확보한다. 열거문의 `i`와 `timestep`을 혼동하면 첫 step부터 잘못된 sigma를 쓴다.

일반 경로에서 `sigma`, `sigma_next`, `dt`를 얻고 model output으로 sample을 갱신한다. per-token timestep 경로는 token마다 current·next sigma를 찾아 `dt`가 tensor가 된다. 이는 단지 빠른 sampling 옵션이 아니라 sequence 위치마다 다른 시간에 있는 상태를 적분하는 모델이다. mask와 broadcasting shape를 별도 fixture로 검증해야 한다.

time shift는 학습 난이도와 solver grid를 바꾼다. resolution이나 sequence length에 따라 time distribution을 shift하는 구현이 있다. shift는 같은 uniform random `u`를 다른 `t`로 매핑해 특정 noise 구간을 더 자주 보거나 solver step을 재배치한다. config의 shift 값만 비교하지 말고 실제 sampled `t`, sigma, log-SNR histogram을 본다.

학습 sampler와 추론 scheduler의 shift convention이 다르면 모델이 적게 본 구간을 solver가 크게 건널 수 있다. model card의 권장 shift와 코드 default, pipeline이 runtime에서 재계산한 값이 같은지 resolved artifact로 고정한다.

### discrete diffusion을 Markov transition으로 검산한다

이산 token에서는 Gaussian 덧셈 대신 transition matrix `Q_t`로 `q(x_t|x_{t-1})`를 정의한다. mask absorbing은 token을 특별한 mask state로 보내고 다시 원 token으로 돌아오지 않는 forward process를 쓸 수 있다. uniform replacement는 vocabulary 전체로 퍼뜨린다. 둘은 같은 “token noise”가 아니며 posterior와 loss target이 다르다.

행렬의 각 행이 1로 합쳐지고 모든 원소가 음수가 아닌지 먼저 검사한다. cumulative transition `Q̄_t=Q_1…Q_t`의 orientation이 row vector convention인지 column vector convention인지 손계산한다. broadcasting으로 batch·sequence마다 timestep이 다를 때 올바른 행렬이 선택되는지 test한다.

이제 살펴볼 것은 absorbing mask의 terminal 조건이다. 충분히 큰 `t`에서 거의 모든 유효 token이 mask가 되는 schedule이라면 padding, BOS, condition prefix는 transition 대상에서 제외해야 한다. attention mask와 diffusion mask를 같은 이름으로 쓰면 padding까지 복원하거나 조건 token을 오염시킬 수 있다. `is_diffused`, `is_valid`, `is_condition` 세 mask를 논리적으로 분리한다.

reverse model이 `p(x_0|x_t)`를 예측하는지, transition posterior를 예측하는지, score-like quantity를 예측하는지에 따라 loss와 sampler가 다르다. 논문의 기호를 production tensor 이름에 대응시키고 exact posterior가 알려진 작은 vocabulary fixture로 probability 합과 step 결과를 검산한다.

confidence refinement는 token별 clock이다. 한 번에 모든 mask를 채우고 confidence가 낮은 위치를 다시 mask하는 방식에서는 token별로 확정 시점이 다르다. confidence score, remask count, tie-breaking, temperature, top-k가 trajectory state다. 동일 seed라도 sorting 안정성이나 동점 처리 변화가 sequence를 바꿀 수 있다.

매 step에 확정·미확정 위치 mask, 선택 token probability, remask 이유를 보존하면 첫 divergence를 찾을 수 있다. 최종 text만 비교하면 초기의 한 confidence 순서 차이가 뒤 전체 문장 변화로 증폭된 원인을 알 수 없다.

DiT를 patch sequence와 condition graph로 해부한다. Diffusion Transformer는 latent image나 video를 patch token으로 만들고 timestep·text 조건과 함께 transformer block에 넣는다. 이름이 transformer라고 causal LM과 같은 mask를 쓰는 것은 아니다. image latent token은 보통 양방향 attention을 하고, text condition은 cross-attention이나 joint attention, adaptive normalization을 통해 주입된다.

patchify에서 latent `[B,C,H,W]`가 `[B,N,D]`로 바뀌는 reshape·permute 순서를 확인한다. unpatchify가 정확한 역함수인지 sequential numbers fixture로 검증한다. height와 width가 뒤집혀도 shape는 같을 수 있다. video에서는 time까지 포함해 spatial patch와 temporal patch의 순서가 position encoding convention과 맞아야 한다.

이 흐름에서 timestep embedding이 block을 조절하는 방식도 빠뜨릴 수 없다. sinusoidal 또는 learned embedding을 MLP에 통과시켜 adaptive LayerNorm의 scale, shift, gate를 만든다. 이때 zero initialization은 학습 초기 block을 identity에 가깝게 유지할 수 있다. gate가 포화하거나 time embedding norm이 특정 구간에서 붕괴하면 noise level에 맞는 연산을 못 한다.

log-SNR bucket별 modulation scale·shift·gate의 norm과 분포를 본다. timestep dtype이 integer에서 float로, 정규화 범위가 `[0,T]`에서 `[0,1]`로 바뀌는 오류는 shape test로 잡히지 않는다. 알려진 세 시간의 embedding checksum과 block output을 golden fixture로 둔다.

condition dropout과 classifier-free guidance를 연결한다. 훈련에서 확률 `p_uncond`로 text condition을 비우면 한 모델이 conditional과 unconditional prediction을 모두 학습한다. 추론에서는 `f_cfg=f_u+s(f_c-f_u)`처럼 guidance scale `s`로 차이를 증폭한다. null condition의 token·mask가 무엇인지, dropout이 batch 단위인지 sample 단위인지, 다른 modality condition도 함께 빠지는지 기록한다.

큰 guidance는 조건 일치도를 높일 수 있지만 vector field를 학습 분포 밖으로 밀고 포화·artifact를 만든다. conditional-unconditional difference norm과 base prediction norm의 비율을 timestep별로 본다. guidance rescale이나 dynamic threshold가 이 비율을 다시 바꾸므로 옵션을 독립 knob로 해석하지 않는다.

training loop를 state transition 표로 만든다. 한 step은 data sample, latent encode, timestep·noise sample, noisy state 구성, condition dropout, model prediction, target 구성, weighted reduction, backward, optimizer, EMA update 순으로 진행된다. 각 단계의 입력 hash와 RNG stream, dtype, owner를 표로 만든다. gradient accumulation에서는 optimizer·EMA·scheduler가 microstep마다 움직이는지 update step마다 움직이는지 구분한다.

EMA는 단순한 모델 복사본이 아니다. decay schedule, warm-up, update counter, 어떤 parameter와 buffer를 포함하는지가 상태다. checkpoint에 EMA가 빠지면 sampling 품질이 달라질 수 있고, resume에서 counter가 0으로 돌아가면 effective averaging window가 바뀐다.

SEDD와 MDLM에서 실제 loss 경계를 찾는다.

`sources/training-diffusion-sedd/losses.py:83`의 `step_fn`은 state, batch와 조건을 받아 학습 step의 중심 경계를 이룬다. `sources/training-diffusion-mdlm/diffusion.py:360-413`의 `_compute_loss`와 training·validation 호출에서 prefix별 metric과 공통 objective의 경계가 드러난다. 함수 이름만 인용하지 말고 noise schedule, model output, mask, reduction이 어느 호출에서 결합되는지 따라간다.

production 코드와 test가 없으면 작은 vocabulary와 deterministic noise로 reference loss를 만든다. train과 validation이 dropout·EMA·sampling mode 외에 objective까지 다르게 쓰지 않는지 비교한다. metric prefix 차이가 실제 계산 차이를 숨기지 않게 한다.

gradient accumulation의 timestep 표본을 감사한다. microbatch마다 timestep을 독립 표본화하면 한 optimizer step의 time histogram이 작을 때 크게 흔들린다. distributed rank까지 합친 실현 분포를 기록한다. loss weight가 극단적인 timestep 하나에 의해 gradient norm이 지배되는지 확인한다.

accumulation 중 overflow가 나면 해당 microbatch만 버리는지 전체 optimizer step을 버리는지 정책을 고정한다. 부분 gradient가 남은 채 step하면 intended estimator가 아니다. scaler와 gradient buffer clear, scheduler·EMA counter가 원자적으로 움직이는지 실패 주입으로 검증한다.

precision과 quantization을 trajectory error로 해석한다. denoising은 model call을 여러 번 이어 가므로 각 call의 오차가 다음 입력이 된다. 한 step의 logit이나 velocity 오차가 작아도 unstable 구간에서 누적될 수 있다. fp32 reference trajectory와 저정밀 trajectory를 같은 initial state·condition·time grid에서 비교하고 step별 state relative error, direction cosine, decoded metric을 기록한다.

flow Euler 구현이 sample을 fp32로 upcast하는 이유는 update 누적의 정밀도를 지키기 위해서다. denoiser는 bf16이어도 scheduler state update를 fp32로 할 수 있다. pipeline 전체 dtype 하나만 보고 판단하지 말고 model input/output, scheduler arithmetic, latent storage, VAE decode 경계를 각각 확인한다.

timestep별 calibration으로 양자화한다. activation 범위는 noise level에 따라 달라진다. 깨끗한 쪽 calibration만으로 scale을 정하면 높은 noise 구간에서 saturation이 생기고, 반대는 정밀도를 낭비할 수 있다. timestep 또는 log-SNR stratified calibration set을 만들고 layer별 clipping과 error를 비교한다.

text condition 길이, image resolution, guidance batch doubling도 activation 분포를 바꾼다. calibration manifest에 조건 분포를 포함한다. weight-only, activation quantization, KV나 cache quantization을 구분해 어느 경계에서 trajectory가 처음 달라지는지 본다.

양자화 오류와 solver 오류를 직교 실험으로 분리한다. 동일 solver grid에서 precision만 바꾸고, 동일 fp32 model에서 solver step 수만 바꾼다. 그다음 조합을 비교해 interaction을 본다. 낮은 precision에서 step 수를 늘리면 truncation error는 줄어도 quantization error를 더 자주 주입할 수 있다. “step이 많을수록 항상 좋다”는 직관이 깨지는 지점이다.

## 22.13 distributed sampling·solver·모델 계열을 공정하게 비교한다

rank별 timestep·condition 불균형과 reduction 분모를 고정하고 SEDD·MDLM·LLaDA와 multimodal condition을 같은 질문으로 비교한다.

이미지·영상 latent의 shape와 text condition 길이가 rank별 compute를 바꾼다. timestep 자체도 일부 architecture나 loss weighting에서 kernel path와 gradient magnitude를 달리할 수 있다. rank별 latent elements, condition tokens, sampled time histogram, forward·backward·collective 시간을 같은 step에서 본다.

resolution bucketing은 padding을 줄이지만 bucket별 data distribution과 timestep 분포가 우연히 결합될 수 있다. sampler RNG를 분리하고 각 bucket에서 time histogram을 검증한다. 영상 duration tail이 특정 rank에 몰리면 NCCL wait가 늘어도 네트워크 문제라고 단정하지 않는다.

다음 경계는 FSDP 경계와 EMA ownership이다. sharded parameter에서 EMA를 full replica로 유지하면 메모리 이점이 줄고, shard별로 유지하면 저장·평가 때 gather가 필요하다. EMA update가 sharded local parameter의 올바른 값을 쓰는지, mixed precision master weight를 쓰는지 정한다. rank별 EMA checksum의 결합이 full reference와 맞는 작은 모델 test가 필요하다.

checkpoint는 denoiser, condition encoder, VAE, optimizer shard, scheduler, EMA, RNG와 sampler를 원자적 manifest에 묶는다. 일부 rank만 성공한 checkpoint를 valid로 노출하지 않는다. object store에서는 임시 prefix에 쓰고 모든 shard digest 검증 뒤 commit marker를 만든다.

이 흐름에서 condition cache와 trainable encoder의 모순도 빠뜨릴 수 없다. frozen text encoder embedding을 cache하면 처리량이 늘지만 tokenizer, prompt template, encoder revision, attention mask가 key에 있어야 한다. encoder를 fine-tune하거나 condition dropout을 embedding 생성 전에 적용하는 stage에서는 cache 의미가 달라진다. trainable encoder인데 detached cache를 쓰면 gradient가 조용히 끊긴다.

21장의 `ConditionArtifact`를 받아 원문·processor·encoder digest와 dropout 상태를 이어 쓴다. diffusion trajectory ID는 condition artifact ID를 참조한다. 그래야 생성 실패에서 condition이 달라진 것과 denoising이 달라진 것을 분리한다.

### sampler를 수치해석 알고리즘으로 검증한다

Euler, Heun, multistep solver는 단순한 품질 preset이 아니다. Euler는 한 점의 vector field로 전진하고, Heun은 predictor와 corrector 두 평가로 local error를 줄인다. multistep은 이전 model output history를 사용하므로 중간 resume에 history가 필요하다. scheduler 객체의 `_step_index`, 이전 output buffer, begin index가 숨은 상태다.

Diffusers scheduler들의 `step` signature와 state field를 비교하면 같은 pipeline 교체가 왜 항상 안전하지 않은지 보인다. model input scaling, timestep convention, prediction conversion, variance/noise injection, history 길이가 다르다. `from_config`가 받아들이는 필드와 무시하는 필드를 resolved diff로 확인한다.

analytic vector field로 solver order를 확인한다. 해를 아는 ODE `dx/dt=ax`나 상수 velocity field를 작은 tensor로 구현한다. step size를 절반으로 줄였을 때 global error가 기대 차수로 감소하는지 본다. 이 test는 생성 품질과 무관하게 timestep 순서, sign, `dt`, history 사용 오류를 잡는다.

reverse time에서 `dt` 부호를 잘못 쓰면 모델이 좋아도 반대 방향으로 간다. scheduler timesteps가 내림차순인지 sigma가 어느 방향인지 배열을 출력하고 첫 두 step을 손계산한다. integer index와 실제 timestep 혼동도 이 fixture에서 드러난다.

stochastic sampler의 난수 소비를 고정한다. ancestral 또는 churn 옵션은 step 중 새 noise를 넣는다. generator를 어디에 전달하는지, batch 전체와 sample별 stream을 어떻게 쓰는지 기록한다. solver step 수나 condition batch ordering이 바뀌면 난수 소비 순서도 바뀔 수 있다. 비교 실험에서는 step별 noise checksum을 보존한다.

결정론 solver 결과와 stochastic 다양성을 같은 재현 기준으로 판단하지 않는다. 전자는 허용오차 내 trajectory 동일성을, 후자는 seed 고정 동일성과 seed 집합의 분포 품질을 각각 본다.

### discrete text diffusion의 likelihood와 평가를 분리한다

autoregressive 모델은 정해진 token 순서로 conditional probability를 곱하지만 discrete diffusion은 여러 위치를 동시에 오염하고 복원한다. training loss가 variational bound의 어느 항인지, denoising cross-entropy의 surrogate인지, score entropy인지에 따라 reported perplexity의 의미가 다르다. AR perplexity와 이름이 같아도 동일한 확률 정규화와 decoding factorization을 뜻하지 않을 수 있다.

`sources/training-diffusion-sedd/losses.py`의 loss factory와 step 경계, `sources/training-diffusion-mdlm/diffusion.py`의 `_compute_loss`를 읽을 때 모델 output parameterization, noise schedule weight, mask 대상, token reduction을 표로 대응시킨다. validation loss가 generation 품질과 어긋나면 sampler step·remasking·temperature의 inference gap을 별도 축으로 본다.

padding과 condition prefix는 확산 상태가 아니다. 문장 길이가 다른 batch에서 padding token을 noise transition에 포함하면 모델은 길이 분포와 pad 복원을 학습한다. attention validity mask, loss mask, diffusion eligibility mask를 분리하고 각 합계를 기록한다. prefix-conditioned generation에서는 prompt token이 고정 boundary condition인지 일부 noise를 넣는지 명시한다.

inpainting은 알려진 token을 매 step 다시 고정하는 projection으로 볼 수 있다. sampler가 한 번만 고정하고 뒤 step에서 덮어쓰는지, model input과 output 양쪽에 mask를 적용하는지 작은 sequence로 확인한다. 고정 token probability가 변하지 않는다는 불변식을 둔다.

길이 생성과 내용 생성을 분리한다. 고정 최대 길이의 mask 열에서 EOS를 복원하는 방식, 길이를 별도 모델로 예측하는 방식, insert-delete transition을 쓰는 방식은 state space가 다르다. EOS가 일찍 확정된 뒤 뒤 token을 어떻게 처리하는지, remask에서 EOS가 다시 바뀔 수 있는지 설정한다. 길이 오류가 내용 metric에 섞이지 않도록 length calibration과 conditional quality를 따로 본다.

D3PM의 insert-delete 구현은 `sources/training-diffusion-d3pm/d3pm/insertdelete/forward_process.py`와 `dynamic_programs.py`에 정렬 상태와 동적 계획 경계를 둔다. 단순 substitution transition보다 state가 크므로 pointer와 alignment table이 checkpoint·test의 일부다. 작은 문자열에서 모든 가능한 경로 확률 합을 열거해 reference와 비교한다.

### SEDD·MDLM·LLaDA를 같은 질문으로 비교한다

모델 패밀리를 이름으로 나열하지 않고 여섯 질문으로 비교한다. forward corruption은 무엇인가, model output은 무엇을 근사하는가, loss는 어떤 measure에서 평균되는가, sampler state는 무엇인가, 조건 token은 어떻게 고정되는가, terminal에 미해결 상태가 남으면 어떻게 처리하는가다. 같은 transformer backbone이어도 이 답이 다르면 checkpoint와 sampler는 호환되지 않는다.

SEDD의 score entropy는 연속 Gaussian score matching을 이산 상태에 그대로 복사한 것이 아니다. 가능한 token 전이 비율을 다루며 noise graph와 weighting이 objective에 들어간다. MDLM의 masked diffusion은 absorbing mask 구조를 이용해 loss와 sampler를 단순화한다. LLaDA 계열의 mask prediction·confidence remasking은 parallel decoding의 schedule을 드러낸다.

이제 살펴볼 것은 공정한 비교의 compute 장부이다. AR 모델의 한 token당 한 forward와 diffusion의 여러 full-sequence forward를 단순 step 수로 비교하지 않는다. 생성 token 수, model calls, call당 active sequence 길이, FLOPs, wall time, peak memory와 kernel utilization을 보고한다. confidence refinement에서 확정 token도 매 call 계산하는지 sparse하게 제외하는지 확인한다.

품질은 같은 tokenizer와 데이터, parameter budget, training tokens, decoding compute에서 비교하는 축과 각 방법의 최적 조건에서 비교하는 축을 분리한다. 한 축만으로 “더 효율적”이라 일반화하지 않는다.

semi-autoregressive와 block diffusion의 경계부터 살펴보자. 문장을 block으로 나누고 block 내부는 diffusion, block 사이는 autoregressive하게 진행하면 두 clock이 생긴다. outer block index와 inner denoising step, KV cache와 block state를 함께 관리한다. block size 변경은 latency뿐 아니라 conditional independence와 error propagation을 바꾼다.

고정 prefix KV를 재사용할 때 inner refinement가 바꾼 token의 KV를 언제 갱신하는지 확인한다. stale cache를 쓰면 최종 token 열은 그럴듯해도 모델이 실제로 조건화한 hidden state와 어긋날 수 있다. block 하나의 모든 refinement에서 cache checksum을 추적한다.

multimodal diffusion의 condition을 세 경로로 나눈다. text condition은 cross-attention key/value, pooled embedding, adaptive normalization vector로 들어갈 수 있다. image condition은 concat channel, ControlNet residual, adapter token으로 들어갈 수 있다. audio·video condition은 시간 정렬 mask와 함께 들어갈 수 있다. “condition encoder를 쓴다”가 아니라 어느 block의 어느 연산에 어떤 shape로 결합되는지 적는다.

condition cache는 encoder output만 저장할 수도 있고 pooled·negative prompt embedding까지 저장할 수도 있다. classifier-free guidance에서 conditional과 unconditional batch를 concat하는 순서가 바뀌면 output split도 바뀐다. batch 2의 고유 scalar condition으로 순서를 검증한다.

ControlNet류 residual의 scale을 추적한다. 조건 branch가 여러 down block과 mid block에 residual을 더하면 scale 목록의 길이와 block 순서가 계약이다. 해상도별 feature shape가 맞아도 residual이 다른 block에 들어가면 의미가 달라진다. block ID, tensor shape, norm, condition scale을 trace에 남기고 scale 0에서 base model과 같아지는지 확인한다.

guess mode, condition dropout, multi-control 합성은 residual 분포를 바꾼다. 여러 조건을 단순 합할 때 norm이 조건 수에 따라 커지는지, normalize나 learned gating이 있는지 본다. 조건 하나씩과 조합의 interaction을 평가한다.

이제 살펴볼 것은 video diffusion의 시간 조건과 메모리이다. video latent는 공간뿐 아니라 시간 attention과 convolution을 쓴다. frame 수가 늘 때 full spatiotemporal attention인지 factorized attention인지에 따라 비용이 다르다. temporal chunking이 경계 artifact를 만들 수 있으므로 overlap과 position continuity를 기록한다.

첫 frame이나 keyframe conditioning에서 조건 frame을 noise에서 제외하는지, latent concat으로 넣는지, 별도 encoder를 쓰는지 확인한다. motion bucket, FPS, duration embedding은 시각 내용과 별도의 condition이다. inference에서 기본값이 training 분포와 다르면 motion speed가 달라질 수 있다.

데이터 target을 생성하는 전처리를 학습 코드로 본다. diffusion 학습 target은 raw image가 아니다. crop·resize·color transform, VAE encode, latent sample, noise와 timestep을 거친다. 이 중 일부는 dataset worker, 일부는 GPU training step에서 실행된다. 경계를 옮기면 cache 가능성과 stochasticity, dtype, 재현성이 바뀐다.

aspect-ratio bucketing에서 crop 위치가 caption의 주체를 지우지 않는지 21장의 task validity 검사를 이어 쓴다. aesthetic filtering과 watermark 제거는 데이터 분포를 바꾼다. generated caption이나 recaptioning model의 provenance도 condition artifact에 포함한다.

다음 경계는 latent cache의 posterior 선택이다. VAE posterior mean을 cache하는 방식과 매 epoch sample을 뽑는 방식은 서로 다른 training distribution이다. sample을 cache하면 posterior noise가 한 번 고정되고, mean은 그 변동을 제거한다. 어느 쪽도 단순 구현 최적화가 아니다. non-cached reference와 latent norm·reconstruction·training 결과를 비교한다.

cache key에는 VAE revision, scaling과 shift, image transform, crop coordinates, dtype이 들어간다. VAE를 바꾸거나 augmentation policy를 바꾸면 전체 cache를 무효화한다. 파일 존재 여부만으로 hit를 인정하지 않고 header manifest와 checksum을 검증한다.

이 흐름에서 caption dropout과 negative prompt의 데이터 계보도 빠뜨릴 수 없다. empty condition은 실제 빈 caption, 의도적 CFG dropout, 필터 실패를 구분해야 한다. null embedding이 tokenizer의 빈 문자열인지 학습된 token인지도 checkpoint 계약이다. negative prompt는 보통 inference 입력이지만 일부 학습은 preference나 quality condition을 사용한다. 어떤 문자열과 weight가 어느 condition branch로 갔는지 보존한다.

장애를 trajectory의 최초 다른 행으로 축소한다. 한 생성의 trajectory table에는 step index, model timestep, sigma·alpha, latent checksum과 norm, condition checksum, prediction norm, guidance difference, scheduler history digest, next latent, RNG digest를 행마다 기록한다. 두 실행의 최초 다른 행과 열을 찾으면 model, condition, scheduler, random noise 중 어느 owner를 먼저 볼지 정할 수 있다.

최종 image가 다르다는 사실은 정보가 너무 적다. 첫 latent가 다르면 seed·shape·device generator, 첫 model output이 다르면 condition·weight·kernel, model output은 같은데 next latent가 다르면 scheduler config·dtype, latent는 같은데 pixel이 다르면 VAE·postprocess를 본다.

NaN을 시간 구간까지 좁힌다. NaN 발생 시 전체 run을 버리기 전에 최초 timestep, module, tensor를 찾는다. forward hook을 항상 켜면 비용이 크므로 finite check를 block 경계와 time bucket sample에 둔다. overflow 직전 norm, guidance ratio, activation dtype과 condition 길이를 기록한다.

NaN latent를 clamp해 계속 생성하는 것은 복구가 아니라 증상 은폐다. 실패 sample을 격리하고 동일 artifact를 fp32·guidance 1·작은 step으로 재생해 원인을 줄인다. 정상 경로에 clamp가 설계돼 있다면 수학적 위치와 범위를 config에 고정한다.

resume history 누락을 검출한다. multistep solver 중간에서 latent와 step index만 저장하고 이전 model outputs를 잃으면 다음 step이 다른 알고리즘이 된다. history buffer의 길이·순서·dtype·checksum을 저장한다. resume 직전 다음 update를 미리 계산해 expected digest를 남기고 복원 후 비교한다.

training resume도 EMA, optimizer, data cursor, time sampler RNG가 필요하다. generation resume와 training resume를 같은 “checkpoint” 용어로 뭉개지 않는다. 각각의 state schema와 허용 범위를 둔다.

품질 평가를 분포·조건·trajectory 세 층으로 나눈다. FID 같은 분포 지표는 sample 집합과 feature extractor, preprocessing, reference statistics에 의존한다. prompt adherence judge는 condition 해석과 judge bias에 의존한다. reconstruction·denoising error는 trajectory 내부를 보지만 perceptual quality와 일치하지 않을 수 있다. 세 층을 함께 보고 서로 대신하지 않는다.

seed를 paired design으로 고정하면 두 모델의 prompt별 차이 분산을 줄일 수 있다. 그러나 한 seed의 paired 결과를 다양성 전체로 일반화하지 않는다. 여러 seed와 prompt strata에서 effect size와 신뢰구간을 보고, prompt와 seed를 random effect로 다룰 수 있다.

이제 살펴볼 것은 solver별 Pareto frontier이다. step 수, model calls, wall time, energy, 품질, prompt adherence, safety를 함께 놓는다. Heun 한 step은 두 model evaluation일 수 있으므로 nominal step보다 NFE를 보고한다. compile warm-up과 VAE decode를 포함한 end-to-end latency와 denoiser-only latency를 분리한다.

동일 checkpoint에서 scheduler를 바꿀 때 권장 prediction type과 input scaling 호환성을 먼저 통과시킨다. 실패한 조합의 낮은 품질을 solver 자체의 열등함으로 보고하지 않는다.

discrete generation의 terminal 품질부터 살펴보자. 최종 mask 잔존율, token 변경 횟수, early 확정의 오류율, 길이 calibration을 본다. fluency 점수만 높아도 사실 token이 초기 step에서 잘못 고정돼 수정되지 않을 수 있다. step별 oracle correctness를 분석해 schedule이 수정 기회를 어디서 잃는지 본다.

안전과 provenance를 생성 경로에 포함한다. diffusion model은 학습 이미지 재현, 얼굴·스타일 모방, 유해 이미지 생성 위험이 있다. safety checker가 pipeline 끝에 있어도 denoiser와 data의 위험이 사라지는 것은 아니다. prompt, condition asset, seed, checkpoint·adapter, scheduler, safety policy revision을 생성 lineage에 묶는다.

워터마크나 provenance metadata는 resize·reencode에서 사라질 수 있다. 생성물 표시와 학습 데이터 provenance는 다른 문제다. 모델이 특정 training sample을 근접 재현하는지 nearest-neighbor와 membership 관점에서 평가하고, decoder·VAE artifact를 실제 복제로 오인하지 않게 한다.

concept erasure를 편집 문제로 연결한다. 특정 개념을 지우는 fine-tuning은 prompt에서 그 개념 생성만 줄이고 관련 개념과 일반 품질을 손상할 수 있다. direct prompt, paraphrase, image condition, adapter 결합에서 suppression과 collateral damage를 본다. 23장의 unlearning과 연결하되 법적 삭제, 행동 억제, representation 편집을 같은 주장으로 부르지 않는다.

편집 전후에는 같은 noise trajectory를 사용해 최초 model-output 차이를 찾는다. 너무 이른 모든 prompt에서 큰 차이가 나면 국소 편집이 아닐 수 있다. target·neighbor·unrelated prompt slice와 seed paired 효과를 보고한다.

red-team 변환이 condition encoder를 겨냥한다. 텍스트 철자 변형, 이미지 안의 텍스트, control image perturbation, audio condition의 초음파·시간 이동은 condition encoder와 connector를 공격한다. 안전 평가가 prompt 문자열만 저장하면 실제 condition tensor를 재현하지 못한다. 21장의 asset·processor artifact를 그대로 참조한다.

옵션을 실제 상태 변화로 번역하는 표

`prediction_type`은 model output을 `x_0`나 update로 해석하는 식을 바꾼다. `variance_type`은 head channel과 reverse transition noise를 바꾼다. `num_train_timesteps`와 beta schedule은 forward measure를 바꾼다. inference step 수와 spacing은 solver grid를 바꾼다. guidance scale은 conditional-unconditional field의 선형 결합을 바꾼다.

`clip_sample`, dynamic threshold는 predicted original sample의 범위를 바꾼다. flow의 shift는 time·sigma grid를 바꾼다. churn과 ancestral noise는 RNG 소비와 stochastic transition을 바꾼다. 각각 checkpoint 호환성, 메모리·NFE, 품질 효과, 확인 metric, 실패 증상을 기록한다.

default를 신뢰하지 않고 resolved schedule을 저장한다. 라이브러리 버전에 따라 config default와 timestep spacing이 바뀔 수 있다. 실행 시 최종 `timesteps`, `sigmas`, `alphas_cumprod`, prediction·variance type을 artifact로 저장한다. 배열 hash와 첫·마지막 몇 값을 기록하면 mutable package 없이도 trajectory를 복원할 수 있다.

pipeline이 scheduler config를 복사하면서 지원하지 않는 field를 무시하거나 새 default를 채울 수 있다. 원 config와 instantiated object의 resolved config를 diff하고 warning을 실패로 승격할 중요 필드를 정한다.

옵션 상호작용을 작은 grid로 검증한다. prediction type×scheduler×guidance×dtype의 모든 조합을 대규모 평가할 필요는 없지만 load gate와 one-step analytic fixture는 전부 돌릴 수 있다. 호환 조합만 작은 prompt·seed grid로 trajectory를 생성하고 unsupported 조합은 명시적으로 거부한다. 성공적으로 실행된다는 사실을 호환 증거로 쓰지 않는다.

한 step을 코드 없이도 다시 계산한다. 검증용 trajectory는 거대한 latent가 필요하지 않다. 두 원소 sample `x_t=[1,-2]`, model output, `ᾱ_t`, `ᾱ_{t-1}`를 작은 유리수에 가깝게 정한다. DDPM의 epsilon 분기에서 `x̂_0`, posterior coefficient, mean, variance를 차례로 계산한다. noise를 0으로 두는 마지막 step과 고정 noise를 넣는 중간 step을 모두 만든다. 구현 출력과 각 중간값을 비교하면 최종 sample mismatch보다 정확한 오류 위치를 얻는다.

`DDPMScheduler.step`은 learned variance에서 channel을 둘로 나누고, prediction type 변환 뒤 threshold 또는 clip을 적용한다. fixture 하나는 channel 두 배와 learned-range를 사용하고, 하나는 잘못된 channel 수가 load gate에서 거부되는지 확인한다. clipping 전후의 `x̂_0`를 저장해야 solver 차이와 range policy 차이를 구분할 수 있다.

flow fixture는 상수 velocity `v=[2,-1]`, `sigma=1`, `sigma_next=0.75`를 둔다. 구현이 사용하는 `dt` 부호에 맞춰 다음 sample을 손으로 계산한다. 이어 per-token time에서 첫 token과 둘째 token의 `dt`를 다르게 둔다. scalar broadcasting이 우연히 shape를 통과해 두 token에 같은 update를 주는 오류를 잡는다.

이제 살펴볼 것은 prediction adapter의 round-trip 표이다. 동일한 `x_0`, noise, `α`, `σ`에서 epsilon, sample, velocity target을 모두 만들고 각 adapter로 다시 `x_0`와 epsilon을 복원한다. 표의 열은 target type, model output, reconstructed `x_0`, reconstructed noise, absolute error다. 끝점에 가까운 시간도 포함해 division stability를 본다.

training target 함수와 inference scheduler가 같은 convention을 쓰는지 이 표 하나로 연결한다. 논문의 기호가 `t=0`을 data로 두고 코드가 noise로 두는 경우 time reversal까지 명시한다. 변수 이름이 같다는 사실보다 수치 round-trip이 강한 증거다.

scheduler state를 serialization한다.

`step_index`, begin index, timestep·sigma 배열, multistep history, RNG state를 JSON metadata와 tensor artifact로 나눈다. float 배열을 JSON decimal로만 저장하면 round-trip이 달라질 수 있으므로 원 dtype byte hash도 둔다. restore 뒤 다음 한 step의 중간값과 결과가 연속 실행과 맞아야 한다.

학습 objective와 sampling algorithm의 틈을 지도화한다. denoising model은 training sampler가 만든 state 분포에서 local target을 배운다. inference solver가 자신의 이전 오차로 만든 state는 그 분포 밖으로 나갈 수 있다. 이를 exposure gap으로 볼 수 있다. step 수를 크게 줄이거나 guidance를 키울수록 off-path state가 커질 수 있다. training validation loss만으로 적은-step 품질을 예측하기 어려운 이유다.

고정 data·noise 쌍에서 true forward path 주변에 작은 perturbation을 주고 vector field error를 측정한다. path 위 error와 수직 방향 perturbation의 error 증가율을 비교하면 local robustness를 볼 수 있다. 이는 전체 생성 품질을 대체하지 않지만 solver가 틀어졌을 때 되돌리는 field인지 더 밀어내는 field인지 설명한다.

consistency·distillation의 교사 상태를 고정한다. 적은 step 모델을 distill할 때 teacher checkpoint, scheduler, guidance, solver와 sampled trajectory가 label generator다. teacher model 이름만 기록하면 target을 복원할 수 없다. teacher output 또는 trajectory checksum, time pair sampling, boundary condition과 student parameterization을 저장한다.

teacher가 EMA인지 raw weight인지, condition dropout과 guidance가 teacher target에 어떻게 들어가는지 확인한다. student의 낮은 step 품질 개선이 teacher 분포 모방인지 새 objective의 효과인지 ablation으로 분리한다.

rectification의 coupling 비용을 측정한다. rectified flow는 현재 모델이나 transport로 endpoint pairing을 다시 만들어 path를 곧게 할 수 있다. 이 재결합은 추가 데이터 생성 단계이며 출발 모델과 seed, pairing algorithm이 provenance다. curvature proxy와 필요한 NFE가 줄었는지 보되 endpoint diversity가 줄지 않았는지 확인한다.

컴파일과 커널 최적화를 수치 계약 아래 둔다. DiT의 attention과 MLP, convolution·normalization은 CUDA kernel 선택과 compile graph에 영향을 받는다. dynamic resolution, guidance batch 크기, timestep embedding shape가 graph specialization을 늘릴 수 있다. compile cache hit, graph count, 첫 실행 비용과 steady-state latency를 구분한다.

fused attention이나 normalization으로 바꿀 때 같은 latent·condition·time에서 block별 출력과 gradient를 reference에 비교한다. 최종 image의 시각적 유사성은 작은 systematic bias를 놓친다. 여러 log-SNR bucket과 condition length, dtype에서 tolerance를 정한다.

CUDA graph와 RNG의 관계부터 살펴보자. CUDA graph capture는 고정 shape와 memory address를 선호한다. stochastic noise 생성과 dynamic condition을 capture 안팎 어디에 둘지 정한다. replay마다 같은 noise를 재사용하는 실수를 막기 위해 generator state와 output checksum을 확인한다. 반대로 결정론 fixture에서는 의도한 동일성이 유지돼야 한다.

variable shape를 padding해 graph를 재사용하면 padding mask와 compute 낭비가 늘어난다. bucket별 graph 수, padding 비율, capture memory, latency를 함께 비교한다. 최적화가 objective의 loss denominator를 바꾸지 않았는지 검사한다.

attention backend가 condition mask를 보존하는지 확인한다. SDPA, flash 계열, eager attention은 mask shape와 all-masked row 처리, dropout RNG가 다를 수 있다. cross-attention에서 text padding이 완전히 차단되는지, joint attention에서 modality segment가 의도한 연결을 갖는지 작은 identity-value fixture로 확인한다. backend fallback이 특정 head dimension이나 sequence length에서 발생하는지도 metric에 남긴다.

이 흐름에서 저장소를 읽는 디깅 동선도 빠뜨릴 수 없다. 첫째, training entry point에서 noise scheduler 생성과 config를 찾는다. 둘째, dataset에서 pixel·latent와 condition을 만드는 경로를 찾는다. 셋째, timestep과 noise를 표본화하는 함수, noisy state 식, model call의 인자, target·weight·reduction을 찾는다. 넷째, optimizer·EMA·checkpoint owner를 찾는다. 마지막으로 validation pipeline이 raw model인지 EMA인지, 같은 scheduler와 processor를 쓰는지 확인한다.

검색어는 `prediction_type`, `add_noise`, `get_velocity`, `sigmas`, `alphas_cumprod`, `compute_loss`, `snr`, `weighting`, `ema`, `register_to_config`처럼 상태를 드러내는 이름을 쓴다. 검색 결과를 바로 설명으로 옮기지 않고 caller와 test를 따라 실제 사용 여부를 확인한다.

이제 살펴볼 것은 Diffusers scheduler를 읽는 순서이다. constructor에서 schedule 배열과 config normalization을 보고, `set_timesteps`에서 inference grid 생성, `scale_model_input`에서 model 전처리, `step`에서 prediction conversion과 update, `add_noise`에서 forward process를 본다. state property와 begin-index 처리를 확인한다. pipeline이 이 메서드들을 어떤 순서로 부르는지 example과 test로 연결한다.

`scheduling_flow_match_euler_discrete.py`가 integer index를 거부하는 검사는 API 의미를 보호한다. 사용자는 `enumerate(timesteps)`의 index가 아니라 배열의 timestep 값을 전달해야 한다. 이런 방어 코드에는 과거에 흔했던 오류 양식이 담겨 있다.

논문 식과 구현 tensor를 잇는 표부터 살펴보자. 각 기호에 code symbol, shape, dtype, sampling source, owner를 대응시킨다. `x_0`가 pixel인지 scaled latent인지, `t`가 integer index인지 continuous float인지, `ε`가 sample별 tensor인지 channel-shared인지, `c`가 token·pooled·control residual인지 적는다. 식의 기대값이 코드에서 어느 reduction과 distributed 평균으로 구현되는지도 쓴다.

실패 사례 여섯 개를 역추적한다. 첫 사례는 모든 이미지가 과포화되는 경우다. condition과 denoiser output은 reference와 같지만 `pred_original_sample`에서 달라지면 threshold·clip config를 본다. VAE decode 직전 latent는 같고 pixel만 다르면 VAE scale과 postprocess를 본다. guidance를 낮춰 숨기는 것은 원인 수정이 아니다.

둘째는 seed 고정에도 결과가 달라지는 경우다. initial noise, stochastic scheduler step의 noise, condition dropout, VAE posterior RNG를 순서대로 hash한다. batch ordering과 generator device도 확인한다. deterministic algorithm 설정만 켜고 RNG 소비 순서를 기록하지 않으면 원인을 못 찾는다.

셋째는 resume 직후 품질이 한동안 흔들리는 경우다. raw weight는 같지만 EMA counter와 weight, optimizer moment, time sampler state가 누락됐을 수 있다. resume 첫 step의 parameter update와 EMA update를 uninterrupted reference와 비교한다.

넷째는 GPU를 늘렸더니 loss curve가 바뀌는 경우다. global batch와 learning rate 외에 timestep negative 또는 condition dropout 표본 분포, loss denominator, all-reduce scaling, gradient accumulation을 본다. contrastive condition loss가 있다면 negative pool도 world size에 따라 달라질 수 있다.

다섯째는 inference step을 줄였더니 특정 prompt만 붕괴하는 경우다. prompt별 condition norm과 trajectory curvature, guidance ratio를 time grid에 그린다. solver가 큰 step으로 불안정 구간을 건너는지, off-path model error가 커지는지 확인한다. 전체 평균이 아니라 prompt slice와 first divergence를 본다.

여섯째는 discrete sampler가 mask를 남기거나 반복 문장을 만드는 경우다. terminal unresolved count, confidence tie, remask schedule, EOS·padding mask, temperature를 본다. 마지막에 mask를 임의 token으로 치환하는 fallback은 실패를 숨기므로 metric과 명시적 policy로 처리한다.

이 흐름에서 각 사례의 최소 반증 실험도 빠뜨릴 수 없다. threshold 문제는 clip을 끈 one-step fixture, RNG 문제는 noise tensor를 외부 주입한 실행, EMA 문제는 raw와 EMA 동시 평가, world-size 문제는 같은 global sample·time 배열 강제, solver 문제는 analytic field, discrete 문제는 작은 vocabulary exhaustive transition으로 반증한다. 큰 benchmark를 다시 돌리기 전에 가장 작은 경계를 검사한다.

수정 뒤 negative test를 보존한다. 오류가 사라졌다는 성공 이미지와 함께 원래 잘못된 config가 load gate에서 실패하는지 확인한다. 잘못된 prediction type, stale condition cache, 누락 history, VAE scale mismatch를 fixture로 남긴다. 회귀 테스트가 없으면 같은 증상이 다른 모델에서 되풀이된다.

모니터링 지표를 시간축으로 조직한다. 학습 지표에는 log-SNR bucket별 unweighted error, weight, weighted loss, gradient norm, sample 수를 둔다. condition dropout률과 실현률, latent norm·finite, optimizer overflow, EMA distance를 함께 본다. 전체 평균은 특정 time 구간의 붕괴를 숨긴다.

시스템 지표에는 resolution·duration bucket별 dataloader·VAE·denoiser·backward·collective 시간, HBM peak, compile graph와 fallback을 둔다. step ID와 sampled time histogram을 연결하면 느린 batch가 입력 shape 때문인지 loss overflow 재시도 때문인지 알 수 있다.

inference 지표에는 NFE, timestep별 model latency, guidance norm ratio, latent norm, scheduler update norm, safety intervention, VAE decode를 둔다. production에서 모든 latent를 저장하지 않고 이상 trace의 checksum과 통계, 접근 통제된 exemplar만 보존한다.

알람은 schedule 변화에 민감해야 한다. resolved timestep·sigma 배열 hash가 승인 artifact와 다르면 생성 시작 전에 알린다. package upgrade 뒤 default가 바뀌는 것을 품질 drift로 발견해서는 늦다. training에서는 time sampler histogram이 예상 분포의 허용 구간을 벗어나면 data·RNG owner에게 알린다.

이 흐름에서 평균 품질 이전의 선행 신호도 빠뜨릴 수 없다. 특정 time bucket gradient 소실, condition-uncondition difference 폭증, EMA distance 급변, terminal mask 잔존, scheduler history 누락이 선행 신호다. 이들은 최종 FID나 사람 평가보다 원인에 가깝고 빠르다. benchmark regression과 같은 incident ID로 연결해 전파 경로를 보인다.

이제 살펴볼 것은 공개 모델을 검토하는 공통 양식이다. 모델 카드에서 base architecture, state space, prediction target, training noise schedule, condition encoder, VAE·tokenizer, 권장 scheduler와 guidance, resolution·duration 범위를 추출한다. 공개 code에서 각각을 읽는 함수와 config를 찾는다. 논문에만 있고 code에서 확인하지 못한 항목은 추정으로 표시한다.

checkpoint 파일의 config와 model card 예제를 비교한다. 예제가 runtime에서 scheduler를 교체하거나 shift를 계산하면 최종 resolved 값을 기록한다. pipeline class가 자동으로 safety checker, VAE slicing, CPU offload를 켜는지도 실행 의미에 포함한다.

이미지·영상·audio·text를 같은 표에 억지로 맞추지 않는다. 공통 열은 state space, time, target, condition, solver, loss reduction, checkpoint state다. modality별 열에는 pixel/latent scale, frame·patch·token eligibility, decoder와 terminal validity를 둔다. 공통성은 디버깅 동선을 제공하고 차이는 실제 측정 계약을 보존한다.

다음 경계는 지원 주장의 증거 등급이다. 공개 training code와 고정 test로 확인한 사실, inference code만 확인한 사실, 논문·모델 카드 서술, 제3자 재현을 구분한다. inference scheduler 코드가 공개됐다고 training objective를 역으로 단정하지 않는다. 미공개 mixture나 optimizer는 빈칸으로 남기는 것이 정확하다.

종단 golden trajectory를 만든다. 연속 fixture는 작은 latent, 고정 condition, 두 time step과 exact model output을 artifact로 둔다. model을 실제 neural network 대신 analytic function으로 바꾼 scheduler-only test와, 실제 작은 model을 쓰는 integration test를 분리한다. 전자는 수식 계약, 후자는 shape·dtype·호출 계약을 검증한다.

이산 fixture는 vocabulary 4개와 mask·padding을 두고 transition과 posterior를 열거한다. 동일 confidence 동점, EOS, condition prefix, terminal mask를 포함한다. step별 probability 합, selected token, remask, RNG를 기록한다.

multimodal fixture는 21장의 condition artifact를 참조해 text·image condition 순서를 바꾸는 negative case를 둔다. condition checksum이 load boundary에서 다르고 trajectory 첫 model call에서 실패해야 한다. scheduler 오류와 condition 오류를 같은 최종 결과로만 보지 않는다.

이제 살펴볼 것은 산출물 schema이다. run·trajectory ID, 소스 리비전, model·VAE·condition encoder hash, resolved config, initial state, condition IDs, time grid, RNG digest, step rows, decoded output와 metric을 포함한다. 큰 tensor 원문은 content-addressed storage에 두고 문서에는 hash와 shape·dtype을 둔다. 삭제·접근 정책도 함께 지정한다.

독립 재계산부터 살펴보자. 검토자는 라이브러리 scheduler를 호출하지 않고 저장된 배열과 식으로 최소 두 step을 다시 계산한다. 구현과 같은 helper를 재사용하면 같은 오류를 공유한다. continuous와 discrete 각각 작은 reference를 별도 코드로 유지한다. GPU 없이 CPU fp64로 계산 가능한 크기로 둔다.

다음 경계는 이 장의 인수 판정이다. 확산 모델을 이해했다는 말은 “noise를 지운다”는 비유를 아는 데서 끝나지 않는다. forward measure, prediction target, weight, condition, vector field 또는 transition, solver state, precision, checkpoint와 평가가 하나의 trajectory로 이어져야 한다. 각 연결은 수식과 고정 revision 함수, 작은 numeric fixture로 검증한다.

학습 개선 주장은 어느 time·condition 분포에서 어떤 gradient estimator를 바꿨는지 말해야 한다. sampling 개선 주장은 동일 checkpoint와 condition에서 solver grid·NFE·stochasticity를 어떻게 바꿨는지 말해야 한다. 둘을 최종 품질 하나로 섞으면 원인을 잃는다.

운영 개선은 first divergence를 더 빨리 찾게 해야 한다. condition artifact, initial noise, model prediction, scheduler update, decode 경계를 순서대로 비교하고 owner를 지정한다. 재현 불가능한 성공 sample보다 재생 가능한 실패 trajectory가 시스템을 더 강하게 만든다.

다음 장의 knowledge editing과 unlearning은 이 golden trajectory를 baseline으로 사용한다. 편집 전후 같은 condition·noise·schedule에서 최초로 model output이 달라지는 위치와 범위를 측정한다. 그래야 실행 환경 drift를 지식 변화로 오인하지 않고, target suppression과 collateral damage를 trajectory 수준에서 설명할 수 있다.

## 22.14 score·probability flow·inverse problem의 기하를 구분한다

score field, probability flow ODE, posterior sampling과 pipeline optimization이 각각 바꾸는 수학적 대상을 분리한다.

데이터 분포가 고차원 공간의 얇은 영역에 놓여 있다고 생각하자. Gaussian forward process는 각 데이터 점 주변에 점점 넓은 구름을 만들고, 큰 noise에서는 여러 구름이 겹쳐 단순한 기준 분포에 가까워진다. score는 각 noise scale에서 log density가 가장 빠르게 증가하는 방향이고, flow velocity는 선택한 probability path를 따라 mass가 움직일 방향이다. 둘 다 “좋아 보이는 그림 쪽”을 직접 가리키는 벡터가 아니라 분포와 path에서 정의된다.

저차원 장난감으로 두 개의 Gaussian mixture를 그리면 intuition을 검증할 수 있다. noise scale이 작을 때 score는 가까운 mode의 세부 구조를 가리키고, scale이 크면 전체 mass 중심과 넓은 구조를 가리킨다. schedule이 특정 scale을 적게 표본화하면 그 해상도의 vector field가 약해질 수 있다. 이미지의 큰 구도와 세부 질감이 서로 다른 time 구간에서 형성된다는 경험적 관찰을 이런 다중 scale geometry와 연결할 수 있지만, layer별 기능으로 곧바로 단정하지는 않는다.

이 흐름에서 path curvature와 solver 난이도도 빠뜨릴 수 없다. 직선 conditional path라도 marginal velocity field는 여러 endpoint pairing이 겹치며 복잡해질 수 있다. trajectory의 tangent 변화량이나 연속 step velocity cosine으로 curvature proxy를 만든다. 곡률이 큰 구간에 더 촘촘한 solver grid를 배치하면 같은 NFE에서 error를 줄일 가능성이 있다. 그러나 모델 오차가 큰 구간과 geometric curvature가 큰 구간은 다를 수 있으므로 둘을 따로 측정한다.

adaptive solver는 local error estimate에 따라 step을 바꾸지만 neural field evaluation 비용과 batch divergence를 만든다. sample마다 다른 step 수를 쓰면 GPU batching이 어려워진다. 품질 이득과 tail latency, 재현 state를 함께 평가한다.

이제 살펴볼 것은 discrete simplex의 geometry이다. token probability는 vocabulary simplex 위에 있다. mask diffusion은 관측 token을 mask 꼭짓점 쪽으로 보내고 reverse model은 다시 여러 token 방향의 확률을 배분한다. argmax refinement는 simplex 내부 분포를 꼭짓점으로 투영하며 작은 logit 차이를 불연속적인 token 선택으로 바꾼다. confidence tie와 temperature가 trajectory를 크게 바꾸는 이유다.

KL, total variation, logit margin은 서로 다른 차이를 본다. step별 probability distribution을 보존한 작은 fixture에서 선택 token만이 아니라 이 거리와 entropy를 비교한다. 최종 문자열 동일성은 내부 불안정성을 숨길 수 있고, 문자열 차이는 확률분포상 아주 작은 동점 차이일 수 있다.

### 디버깅 체크리스트를 질문 순서로 압축한다

첫 질문은 “무엇이 state인가”이다. pixel, VAE latent, token, mask, action trajectory 가운데 무엇이 확산되는가. 둘째는 “시간의 방향과 단위는 무엇인가”이다. index, normalized time, sigma, log-SNR을 구분한다. 셋째는 “모델 output을 무엇으로 해석하는가”이다. epsilon, sample, velocity, score, posterior 중 하나를 식과 코드로 고정한다.

넷째는 “어느 분포에서 loss를 평균하는가”이다. timestep sampler, data mixture, condition dropout, loss weight와 denominator를 쓴다. 다섯째는 “추론 state가 무엇인가”이다. time grid, step index, history, RNG, guidance branch를 쓴다. 여섯째는 “checkpoint가 이 state를 모두 복원하는가”이다. weight만 복원되는 경로와 exact resume를 구분한다.

일곱째는 “최초 불일치가 어디인가”이다. condition, initial state, first model output, scheduler update, decode를 순서대로 본다. 여덟째는 “비교의 분모가 같은가”이다. NFE, wall time, seed·prompt set, precision, VAE와 safety postprocess를 맞춘다. 아홉째는 “지원하지 않은 조합을 거부하는가”이다. silent fallback과 default drift를 막는다.

코드 리뷰에서 남길 열두 좌표부터 살펴보자. noise 생성 함수, time sampler, state mixing 식, model forward, condition injection, target builder, loss reduction, optimizer·EMA update, scheduler constructor, input scaling, step update, checkpoint save/load 좌표를 고정 revision에 묶는다. 각 좌표에 caller와 test 또는 numeric fixture를 붙인다. symbol만 있고 호출되지 않는 dead code를 근거로 쓰지 않는다.

다음 경계는 실험 보고서에서 남길 열두 수치이다. data와 condition 수, 학습 state elements, time bucket 표본 수, unweighted·weighted loss, gradient norm, overflow·skipped step, EMA distance, inference NFE, step latency, trajectory error, 조건 품질, 안전·재현 실패율을 남긴다. 평균과 함께 분모·분포·허용오차를 제시한다.

### 종합 사고 실험: 좋은 loss와 나쁜 sampler

validation denoising loss가 이전보다 낮지만 생성 품질이 나빠졌다고 하자. 먼저 loss가 같은 time measure와 weight에서 계산됐는지 본다. sampler time grid가 training distribution의 약한 구간을 크게 건너는지 본다. prediction conversion과 VAE scale이 같은지, guidance가 field를 분포 밖으로 미는지, EMA weight를 실제로 썼는지 확인한다.

반대 상황도 가능하다. loss는 조금 높지만 더 나은 solver와 condition encoder, decoder가 최종 품질을 높일 수 있다. 이때 모델 학습이 좋아졌다고 말하지 않고 system 결과가 좋아졌다고 말한다. 구성 요소별 paired trajectory와 ablation이 귀속을 정한다.

discrete 모델에서는 token cross-entropy가 좋아져도 confidence ranking이 나빠져 early wrong fixation이 늘 수 있다. step별 oracle correctness와 remask recovery를 보면 objective와 sampler의 틈을 찾는다. continuous 모델에서는 local velocity MSE와 integration stability의 틈을 본다.

이 사고 실험을 해결할 수 있다면 독자는 확산을 한 문장의 은유가 아니라 검증 가능한 시스템으로 읽는다. 수식은 target과 update를 제한하고, 코드는 그 제한을 tensor와 state로 구현하며, 관측성은 어느 제한이 처음 깨졌는지 알려 준다. 이 세 층이 맞물릴 때 비로소 새로운 architecture나 scheduler를 안전하게 검토할 수 있다.

### 23장으로 넘기는 불변식

편집 또는 unlearning 전 baseline에는 condition artifact, initial noise·token state, resolved time grid, prediction convention, model·EMA weight hash, scheduler history와 VAE·tokenizer revision이 고정돼야 한다. 편집 후 동일 artifact를 재생해 target prompt와 neighbor·unrelated prompt에서 step별 prediction 차이를 얻는다.

target 출력만 사라졌다는 사실은 삭제의 충분조건이 아니다. 다른 seed, paraphrase, image condition, adapter 결합에서 회복될 수 있다. 반대로 모든 trajectory가 크게 달라지면 collateral damage다. 23장은 이 차이를 지식의 위치와 변경 범위, 데이터 lineage로 연결한다.

따라서 이 장의 최종 산출물은 sample 이미지 모음이 아니라 재생 가능한 trajectory 묶음이다. 성공과 실패, continuous와 discrete, conditional과 unconditional, raw와 EMA를 포함한다. 각 묶음은 첫 불일치 owner와 미검증 범위를 가진다. 이것이 다음 장의 변경 주장을 환경 drift와 분리하는 기준선이다.

이 흐름에서 독자가 제출할 종합 실습 패키지도 빠뜨릴 수 없다. 첫 파일은 수식 장부다. forward state 식, target 변환, loss weight와 reduction, reverse update를 한 페이지에 쓰고 각 기호를 실제 tensor 이름·shape·dtype에 대응시킨다. epsilon·sample·velocity 중 사용하지 않는 표현도 round-trip 표에는 포함해 convention 오류를 반증한다. discrete 모델이면 transition matrix와 posterior, terminal rule을 같은 형식으로 쓴다.

둘째 파일은 source map이다. 고정 commit에서 time sampler, noise 또는 transition 생성, model call, target, reduction, scheduler step, EMA, checkpoint 함수와 test 좌표를 기록한다. 논문만 공개되고 학습 코드가 없는 경계는 비어 있다고 명시한다. 추론 코드에서 학습 objective를 추정해 채우지 않는다.

셋째는 golden trajectory다. 연속 두 step 또는 이산 세 step이면 충분하지만 모든 중간 상태와 checksum이 있어야 한다. condition과 RNG를 고정한 positive trace, prediction type이나 transition convention을 의도적으로 틀린 negative trace를 함께 둔다. 실패는 load gate나 최초 잘못된 update에서 검출돼야 한다.

넷째는 상태 복원 보고서다. training checkpoint와 generation resume를 구분해 weight, EMA, optimizer, scaler, data cursor, time sampler RNG, solver history 가운데 무엇을 복원했는지 표로 만든다. uninterrupted 실행과 resume 실행의 다음 step을 비교하고 허용오차 근거를 적는다.

다섯째는 평가 카드다. 학습 loss, trajectory error, 조건 일치, 분포 품질, 다양성, latency·NFE·메모리, 안전과 재현성을 분리한다. prompt·seed·resolution·duration·time bucket별 분모와 신뢰구간을 쓴다. 종합 평균 하나로 실패 slice를 감추지 않는다.

여섯째는 장애 주입 기록이다. stale condition cache, 잘못된 VAE scale, 누락된 multistep history, integer timestep 전달, NaN prediction, terminal mask 잔존을 차례로 주입한다. 어느 assertion·metric·owner가 반응했는지와 수정 뒤 negative test가 유지되는지 남긴다.

이 여섯 파일을 다른 검토자가 package default에 의존하지 않고 재계산할 수 있으면 실습은 통과한다. 생성물이 보기 좋다는 평가는 마지막에 붙는다. 먼저 trajectory의 수학과 상태가 맞아야 품질의 원인을 설명할 수 있다.

마지막 질문은 간단하다. 첫 model call이 달라졌을 때 condition과 weight 중 어느 쪽인지, model call은 같은데 next state가 달라졌을 때 어느 scheduler 상태인지, latent는 같은데 output이 달라졌을 때 어느 decoder 계약인지 답할 수 있는가. 답이 좌표와 fixture로 이어지면 이 장은 끝난다. 답이 “라이브러리가 알아서 한다”에 머물면 아직 시작하지 않은 것이다.

실무 인수 회의에서는 임의의 trajectory 하나를 골라 거꾸로 읽는다. 최종 output에서 decoder 입력 latent를 찾고, 직전 scheduler update의 sample과 prediction을 찾으며, 그 prediction을 만든 model weight·condition·time을 찾는다. 이어 initial state와 data artifact까지 도달해야 한다. 중간 연결 하나가 mutable alias나 기억에 의존하면 인수를 보류한다.

검토자는 설정 하나를 임의로 바꾸고 영향 범위를 예측한다. guidance scale이면 conditional·unconditional 결합 이후부터, VAE scaling이면 training latent 생성과 decode 경계부터, prediction type이면 target과 scheduler conversion부터 달라져야 한다. 예상보다 앞에서 차이가 나면 숨은 coupling이고, 예상 지점에서 아무 차이가 없으면 option이 읽히지 않거나 cache가 stale할 수 있다.

또한 두 실행의 품질 차이에 반드시 귀속 수준을 붙인다. model weight 변화, training objective 변화, sampler 변화, condition processor 변화, decoder·postprocess 변화 가운데 무엇을 직접 통제했는지 쓴다. 여러 축이 함께 바뀌었으면 시스템 비교라고 말하고 단일 기법의 효과로 포장하지 않는다.

이 규율은 확산 연구의 속도를 늦추지 않는다. 오히려 실패한 대규모 실행을 작은 numeric fixture로 환원하고, 검증된 state를 다음 실험에서 재사용하게 한다. 새로운 solver와 architecture가 나와도 state·time·target·condition·update라는 질문은 그대로 남는다. 그 질문을 정확히 답하는 능력이 특정 라이브러리 사용법보다 오래가는 기술이다.

최종 서명에는 검증한 scheduler와 prediction type, state 공간, dtype, condition 범위, solver step 수를 적는다. 검증하지 않은 video 길이, audio 표현, discrete vocabulary, quantized backend는 별도 열에 남긴다. 한 성공 경로를 전체 모델 지원으로 확대하지 않는다.

이제 독자는 결과 이미지가 아니라 계산의 계보를 본다. 어느 데이터가 어떤 상태가 되었고, 어느 시간에서 무엇을 예측했으며, 어떤 update가 다음 상태를 만들었는지를 재현한다. 바로 그 계보가 23장에서 지식을 편집하거나 지울 때 비교할 기준이다.

최종 인수 후에도 새 라이브러리 revision은 같은 golden trajectory를 다시 통과해야 한다. default 배열과 kernel이 바뀌면 새 lineage로 분기하고 차이를 기록한다. 성능이 좋아졌다는 이유로 계약 변화가 사라지지는 않는다. 재현 가능한 변화만 다음 실험의 토대가 된다.

score와 probability flow ODE를 같은 장에서 구분한다. forward SDE를 `dx=f(x,t)dt+g(t)dW`로 쓰면 reverse-time SDE에는 score `∇_x log p_t(x)`가 들어간다. probability flow ODE는 같은 marginal density를 갖도록 stochastic diffusion을 deterministic vector field로 바꿀 수 있다. 두 경로가 같은 marginal을 갖는다는 말은 동일 seed의 sample trajectory가 같다는 뜻이 아니다.

score model output이 epsilon prediction으로 parameterize될 때 `s_θ(x_t,t)`로 변환하는 scale과 부호를 schedule에서 유도한다. 구현 helper를 믿기 전에 작은 Gaussian에서 analytic score `-(x-μ_t)/σ_t²`와 비교한다. log-SNR 끝점에서 division이 커질 수 있으므로 fp32·fp64 reference를 둔다.

SDE와 ODE solver state부터 살펴보자. reverse SDE는 step마다 새 noise와 generator state가 필요하고 probability flow ODE는 결정론 solver history가 필요하다. 같은 scheduler라는 이름 아래 RNG·history ownership이 다르다. checkpoint schema와 재현 허용오차를 분리한다.

다음 경계는 likelihood 계산의 divergence이다. probability flow ODE로 exact 또는 근사 likelihood를 구할 때 vector field divergence와 Hutchinson trace estimator가 필요할 수 있다. probe noise와 solver tolerance가 state다. 생성 FID와 likelihood가 같은 model ranking을 만든다고 가정하지 않는다.

flow matching의 continuity equation을 직관으로 읽는다. density `p_t(x)`와 velocity `v_t(x)`는 `∂_t p_t+∇·(p_t v_t)=0`을 만족한다. 이는 점 하나가 어디로 가는지뿐 아니라 확률 질량이 압축·팽창하며 이동하는 방식을 제한한다. vector field가 국소적으로 발산하면 density가 줄고 수렴하면 늘어난다. sample quality와 mode coverage를 field geometry로 연결하는 출발점이다.

conditional flow target을 endpoint pair에서 쉽게 만들 수 있어도 marginal velocity는 같은 `x_t`를 지나는 여러 path의 conditional expectation이다. independent coupling이 path crossing을 많이 만들면 network가 평균 velocity를 학습하며 곡률과 ambiguity가 커질 수 있다. optimal transport나 rectification이 pairing을 바꾸는 이유다.

이제 살펴볼 것은 field의 Lipschitz와 step 크기이다. 인접 state에서 velocity가 급격히 달라지는 구간은 Euler local error가 크다. Jacobian-vector product나 finite difference로 field sensitivity를 sample한다. log-SNR·condition slice별로 curvature와 error를 보고 solver grid를 배치한다. 14장의 low-precision 오차와 interaction도 본다.

divergence와 collapse부터 살펴보자. field가 많은 initial noise를 같은 좁은 영역으로 과도하게 모으면 diversity가 줄어들 수 있다. endpoint pair distance, sample covariance와 nearest-neighbor를 본다. 단일 perceptual score로 mode collapse를 숨기지 않는다.

다음 경계는 diffusion transformer의 normalization과 conditioning gate이다. DiT류 block은 timestep·condition embedding에서 adaptive LayerNorm의 scale·shift와 residual gate를 만든다. zero-initialized gate는 초기 network를 identity에 가깝게 두어 큰 transformer의 학습을 안정화할 수 있다. 그러나 gate가 계속 0 근처면 block이 학습되지 않고, 특정 time에서만 포화하면 schedule 일부가 병목이 된다.

block별 gate norm, gradient와 log-SNR bucket을 기록한다. condition dropout에서 null condition과 실제 condition의 gate 차이를 본다. text encoder가 frozen이어도 modulation MLP가 condition을 무시할 수 있다.

이 흐름에서 joint attention의 modality 균형도 빠뜨릴 수 없다. text와 latent token을 한 attention에 넣으면 token 수가 많은 image·video가 softmax denominator를 지배할 수 있다. modality별 attention mass, entropy와 output norm을 본다. token count normalization이나 separate QKV가 어떤 tensor를 바꾸는지 source에서 확인한다.

이제 살펴볼 것은 QK norm과 안정성이다. QK normalization은 attention logit scale을 제한할 수 있지만 dtype·epsilon과 rotary 적용 순서가 중요하다. fused kernel 전후의 block output을 time·sequence bucket에서 비교한다. 8장의 attention 구현 검산을 diffusion sequence에 적용한다.

discrete diffusion의 posterior를 작은 vocabulary로 유도한다. forward transition `Q_t[i,j]=q(x_t=j|x_{t-1}=i)`와 cumulative `Q̄_t`가 있을 때 posterior `q(x_{t-1}|x_t,x_0)`는 Bayes rule로 계산한다. vocabulary 세 개와 mask state 하나를 두고 모든 조합을 열거한다. row·column orientation, batch gather와 normalization을 production tensor와 비교한다.

mask absorbing에서 mask로 간 token은 forward에서 돌아오지 않지만 reverse model은 원 token 분포를 예측한다. padding·condition prefix는 transition 대상이 아니어야 한다. EOS가 diffusion state인지 boundary인지도 길이 모델과 연결된다.

다음 경계는 score entropy와 cross-entropy이다. 이산 score 계열 objective와 masked-token cross-entropy는 모델 output과 weight가 다르다. 동일 transformer라도 sampler 호환성을 가정하지 않는다. `sources/training-diffusion-sedd/losses.py`의 step factory와 `training-diffusion-mdlm/diffusion.py:_compute_loss`에서 schedule·mask·reduction이 결합되는 위치를 비교한다.

이 흐름에서 transition sparsity의 구현도 빠뜨릴 수 없다. 큰 vocabulary의 dense `V×V` 행렬은 비현실적이므로 absorbing·uniform 구조를 analytic하게 계산하거나 sparse 연산을 쓴다. 수학의 matrix notation과 실제 gather·scalar coefficient를 대응시킨다. dense toy reference로 optimized path를 검증한다.

timestep sampling을 분산 Monte Carlo로 해석한다. 한 optimizer step의 gradient는 data, timestep, noise와 condition dropout에 대한 Monte Carlo estimate다. rank와 microbatch가 독립 RNG stream을 가지면 global sample 수가 늘지만, seed collision이나 sampler broadcast는 다양성을 줄인다. sample ID·rank·microstep에서 RNG를 결정적으로 유도하고 실현 time histogram을 기록한다.

importance sampler가 과거 loss 통계로 time probability를 바꾸면 sampler state가 학습과 함께 진화한다. 통계의 EMA, warm-up과 distributed aggregation을 checkpoint에 넣는다. resume에서 histogram이 초기화되면 objective estimator가 달라진다.

variance reduction과 bias부터 살펴보자. loss-aware sampling은 high-loss time을 더 자주 표본화해 gradient variance를 줄일 수 있지만 correction weight가 없으면 objective를 바꾼다. correction이 있어도 extreme inverse probability가 variance를 키울 수 있어 clipping을 쓸 수 있다. clipping이 만드는 bias를 명시한다.

다음 경계는 noise coupling 실험이다. 두 checkpoint를 비교할 때 동일 data·timestep·noise를 쓰면 paired variance가 줄어든다. 하지만 학습 자체에서 noise를 고정하면 overfit할 수 있다. 평가용 coupled noise와 학습 RNG를 분리한다.

classifier-free guidance를 vector geometry로 읽는다. conditional prediction `v_c`와 unconditional `v_u`의 차이 `d=v_c-v_u`는 condition이 vector field를 바꾼 방향이다. guidance는 `v_u+s d`로 이 방향을 연장한다. 큰 `s`는 단순 condition 강화가 아니라 training에서 본 두 field 사이 segment를 넘어 extrapolation한다.

`||d||/||v_u||`, cosine과 timestep별 distribution을 본다. condition 종류와 prompt 길이, negative prompt에 따라 direction이 다르다. guidance rescale은 norm을 조절하지만 방향 오류를 고치지 못한다.

이제 살펴볼 것은 parallel CFG와 batch ordering이다. conditional·unconditional 입력을 batch concat해 한 forward로 계산할 때 output split 순서와 media condition replication을 검증한다. dynamic batch scheduler가 순서를 바꾸면 silent swap이 생길 수 있다. 고유 scalar condition fixture를 쓴다.

guidance distillation부터 살펴보자. teacher guidance를 student 한 번의 forward에 distill하면 teacher checkpoint·scheduler·scale과 prompt distribution이 target generator다. 다양한 scale을 조건으로 주는지 고정 scale인지 기록한다. student quality를 teacher compute와 함께 비교한다.

diffusion data curriculum을 해상도·시간·noise로 분해한다. 낮은 해상도에서 높은 해상도로 가는 curriculum은 latent token 수와 data crop을 동시에 바꾼다. 짧은 video에서 긴 video로 가면 temporal position과 memory, data distribution이 바뀐다. noise curriculum은 time sampling measure를 바꾼다. 세 축을 한 stage 이름으로 뭉개지 않는다.

stage 전이에서 VAE·condition cache, compile graph, optimizer moment와 EMA가 계속 유효한지 본다. resolution 변경으로 latent scale 통계가 바뀌면 normalization과 loss weight를 재검증한다. 6장의 data mixture와 13장의 scheduler state를 함께 인계한다.

이 흐름에서 progressive resizing의 반증도 빠뜨릴 수 없다. 같은 total compute에서 처음부터 mixed resolution인 대조군과 비교한다. 개선이 curriculum 순서인지 더 많은 low-resolution samples 때문인지 분리한다. high-resolution slice와 small-text·fine-detail을 별도 평가한다.

이제 살펴볼 것은 time curriculum의 끝점이다. 깨끗한 쪽이나 noise 쪽을 늦게 도입하면 해당 구간 target이 급변한다. time bucket별 gradient와 optimizer overflow를 본다. 모든 time을 포함한 golden validation을 stage 내내 유지한다.

distributed DiT의 sequence와 context ownership부터 살펴보자. 큰 image·video latent는 spatial·temporal token을 sequence parallel로 나눌 수 있다. attention all-to-all과 position shard, condition replication의 ownership을 명시한다. rank별 token 수가 달라 padding·collective 참여가 틀어질 수 있다.

condition text는 작아 replicate할 수 있지만 gradient가 필요한 trainable encoder라면 replicated parameter reduction이 필요하다. cached frozen condition과 trainable encoder를 stage gate로 구분한다. FSDP가 DiT와 encoder·VAE를 어떻게 wrap하는지 parameter map을 저장한다.

다음 경계는 timestep의 broadcast이다. sequence shard들은 같은 sample의 동일 timestep·noise convention을 써야 한다. rank별로 독립 timestep을 뽑아 한 sample shard에 다른 time을 주는 오류를 작은 distributed fixture로 잡는다. per-token time 모델이라면 의도한 tensor와 별개다.

이 흐름에서 EMA shard의 검산도 빠뜨릴 수 없다. local shard EMA를 gather해 full fp32 reference와 비교한다. update counter와 skipped optimizer step을 일치시킨다. 17장의 atomic checkpoint와 29장의 rank-loss injection을 적용한다.

trajectory observability를 계층화한다. 입력에는 data·condition artifact와 latent scale, 학습에는 sampled time·noise·target·weight·loss·gradient, inference에는 initial state·time grid·prediction·guidance·next state, 시스템에는 kernel·dtype·latency·memory가 있다. 같은 `NoiseTrajectoryID`로 계층을 연결한다.

모든 latent 원문을 상시 저장하지 않고 fixed golden과 anomaly exemplar에서만 content-addressed tensor를 둔다. 일반 run은 norm, finite, checksum과 small projection을 기록한다. 개인정보·저작권 asset의 접근 통제를 유지한다.

first-divergence 자동화부터 살펴보자. 두 trajectory의 condition, initial state, step arrays를 검증하고 각 row의 prediction·scheduler update를 비교한다. 최초 다른 column에 model·scheduler·VAE owner를 붙인다. tolerance를 time bucket·dtype별 reference distribution에서 정한다.

다음 경계는 26장과의 연결이다. Prometheus에는 time bucket loss·overflow, NFE·step latency, compile fallback과 failure count 같은 bounded metric을 둔다. trajectory ID는 exemplar·trace로 연결한다. high-cardinality seed·prompt를 metric label에 넣지 않는다.

이 흐름에서 22장 심화 wave의 실험 계약도 빠뜨릴 수 없다. 독자는 analytic Gaussian score, constant velocity ODE, 4-state discrete posterior를 손으로 계산한다. 이어 Diffusers `DDPMScheduler.step`과 `FlowMatchEulerDiscreteScheduler.step`, SEDD·MDLM loss 경계를 고정 revision에서 대응시킨다. exact target round-trip과 wrong convention negative test를 둔다.

학습 실험은 time sampler, target·weight, curriculum과 precision을 분리하고 log-SNR bucket별 gradient를 본다. inference 실험은 checkpoint를 고정한 채 solver grid·NFE·guidance를 바꾸며 trajectory error와 quality·latency를 본다. training 개선과 sampler 개선을 같은 주장으로 섞지 않는다.

21장의 condition artifact, 14·15장의 precision·parallel, 17장의 checkpoint, 23장의 editing baseline, 24장의 statistics와 28~30장의 재현 실습이 이 장의 trajectory에 접속한다. 확산을 이해한다는 것은 noise metaphor가 아니라 state·time·target·condition·update의 모든 전이를 재계산할 수 있다는 뜻이다.

inverse problem과 posterior sampling을 분리한다. super-resolution, inpainting, deblurring은 관측 `y=A(x)+η`와 prior를 결합해 posterior를 표본화하는 문제로 볼 수 있다. 단순 text-to-image condition과 달리 observation consistency가 명시적이다. sampler가 매 step 관측 영역을 projection하는지, gradient guidance를 쓰는지, 별도 conditional model인지 구분한다.

inpainting mask는 pixel·latent·token 가운데 어느 공간에 있고 blur·downsampling operator와 shape가 무엇인지 기록한다. mask 경계에서 VAE receptive field가 관측 정보를 섞을 수 있다. known region exactness와 unknown region quality, boundary artifact를 따로 본다.

data consistency와 prior의 균형부터 살펴보자. 관측 likelihood weight가 크면 noise까지 맞추고, 작으면 plausible하지만 관측과 다른 sample이 된다. measurement residual과 perceptual quality를 Pareto로 본다. guidance scale을 model quality knob 하나로 부르지 않는다.

다음 경계는 inverse fixture이다. 2차원 Gaussian prior와 선형 observation에서 analytic posterior mean·covariance를 구해 sampler 통계를 비교한다. image metric 전에 posterior mean과 variance, observation residual을 검산한다.

consistency model과 적은-step 생성을 이해한다. consistency model은 같은 probability flow trajectory 위 서로 다른 time의 state가 동일한 endpoint로 매핑되도록 학습할 수 있다. boundary condition과 teacher·EMA target, time pair sampling이 핵심이다. one-step 생성 성능만 보고 일반 diffusion loss와 같다고 생각하지 않는다.

teacher-free 또는 distillation 설정에 따라 target network와 stop-gradient, EMA update가 달라진다. source에서 online·target model의 owner와 checkpoint를 추적한다. time pair가 adjacent인지 skip인지, solver가 target 생성에 들어가는지 기록한다.

consistency의 transitivity test.

`f(x_t,t)`와 중간 state를 거친 `f(x_s,s)`가 같은 endpoint를 내는지 작은 analytic path에서 확인한다. pair loss가 낮아도 긴 time skip에서 error가 누적될 수 있다. 여러 간격의 consistency matrix를 본다.

이제 살펴볼 것은 few-step Pareto이다. 1·2·4·8 step에서 NFE, latency, diversity와 condition adherence를 비교한다. compile·VAE 시간을 포함한다. teacher quality와 student compute를 함께 보고 distillation data 생성 비용을 숨기지 않는다.

discrete diffusion의 병렬 decoding 실패를 해부한다. 여러 mask를 동시에 채우면 서로 의존하는 token을 같은 step에 독립적으로 결정할 수 있다. subject-verb agreement, code identifier와 closing delimiter에서 consistency가 깨질 수 있다. block size와 remask policy가 dependency를 얼마나 회복하는지 syntax·semantic slice로 본다.

confidence는 model probability가 높다는 뜻이지 token이 옳다는 뜻은 아니다. calibration이 나쁘면 잘못된 token을 일찍 고정한다. step별 confidence-correctness reliability와 remask recovery rate를 측정한다.

다음 경계는 entropy와 margin 선택이다. max probability, top1-top2 margin, entropy는 다른 uncertainty를 본다. temperature가 selection 순서를 바꾼다. 동일 logits fixture에서 각 rule의 확정 mask를 손계산한다. tie-breaking과 distributed sort의 안정성을 고정한다.

이 흐름에서 KV·hidden reuse도 빠뜨릴 수 없다. 확정 token의 computation을 cache해 속도를 높일 때 다른 token 변화가 그 hidden state를 무효화할 수 있다. dependency와 cache invalidation을 명시한다. output 일치와 speed를 block fixture에서 검증한다.

확산 모델의 knowledge editing을 trajectory로 측정한다. 23장의 concept editing·unlearning은 동일 prompt·condition·initial noise·schedule에서 edit 전후 model prediction을 비교한다. target concept에서 어느 time·block부터 차이가 생기는지, neighbor·unrelated prompt에도 같은 차이가 퍼지는지 본다. 최종 image suppression만으로 locality를 판단하지 않는다.

text encoder edit, cross-attention edit, denoiser weight와 adapter는 최초 차이 위치가 다르다. condition embedding부터 다르면 encoder, condition은 같고 cross-attention output부터 다르면 denoiser conditioning, scheduler만 다르면 편집 주장이 아니다.

seed-paired effect부터 살펴보자. 같은 seed 쌍에서 target score와 collateral quality 차이를 구해 variance를 줄인다. 여러 seed·prompt family에서 cluster interval을 낸다. cherry-picked 성공 image를 근거로 쓰지 않는다.

다음 경계는 재생 공격이다. paraphrase, negative prompt, image condition, adapter composition과 guidance scale을 바꿔 concept가 회복되는지 본다. suppression 범위와 attack budget을 명시한다. 법적 data 삭제와 behavior suppression을 구분한다.

이 흐름에서 확산 학습 장애의 현장 runbook도 빠뜨릴 수 없다. loss가 갑자기 폭증하면 data·VAE latent norm, sampled time과 weight, model prediction finite, gradient·scaler를 순서대로 본다. 특정 resolution·time bucket에 몰리면 data curriculum과 precision을 조사한다. 모든 rank에서 같은 step인지 비교해 local corrupt batch와 global optimizer state를 나눈다.

sample quality만 서서히 나빠지면 EMA distance와 validation trajectory, condition dropout 실현률, data mixture drift를 본다. training loss가 안정돼도 EMA update 누락이나 VAE·text encoder cache drift가 있을 수 있다.

이제 살펴볼 것은 OOM과 straggler이다. latent elements·condition tokens·attention backend와 saved activation을 memory ledger에 대조한다. rank별 resolution·video duration tail이 collective wait를 만드는지 본다. 16장의 scheduler와 29장의 failure injection을 사용한다.

resume divergence부터 살펴보자. next data IDs, time·noise RNG, optimizer·scaler·EMA와 sampler statistics를 uninterrupted reference와 비교한다. 첫 update 전부터 다른지 뒤인지로 owner를 줄인다. weight checksum만 같아도 exact resume는 아니다.

다음 경계는 source audit에서 놓치기 쉬운 함수이다. training script의 model call만 읽지 않고 scheduler constructor·`set_timesteps`, `add_noise`, target converter, VAE encode, condition dropout, EMA와 save/load를 찾는다. inference pipeline에서는 `scale_model_input`, scheduler `step`, guidance concat·split과 VAE decode를 찾는다.

Diffusers `scheduling_ddpm.py:461`의 `step`과 `scheduling_flow_match_euler_discrete.py:423`의 `step`은 output 해석과 state update의 기준 좌표다. tests에서 prediction type, custom timesteps, begin index, per-token time와 dtype case를 찾아 production 분기와 연결한다.

이 흐름에서 고정 revision 독자 표기도 빠뜨릴 수 없다. repository commit, relative path, symbol, line span과 content hash를 함께 둔다. 논문 식과 실제 tensor symbol을 표로 잇는다. upstream 변경으로 line이 움직이면 hash·symbol 검증을 다시 수행한다.

이제 살펴볼 것은 심화 인수 판정이다. 독자는 continuous SDE·ODE와 flow matching, discrete Markov transition을 서로 바꾸어 말하지 않는다. 각각의 state, target, time measure와 sampler history를 구분한다. analytic fixture에서 변환과 update를 재계산하고 production source의 분기와 대응시킨다.

또한 training loss와 inference quality, model과 solver, condition과 trajectory, precision과 numerical integration을 분리해 실험한다. 첫 divergence와 paired trajectory로 원인을 찾는다. 최종 생성물만 보고 scheduler·VAE·model 변화의 귀속을 추측하지 않는다.

18,000단어 인수는 분량이 아니라 이 연결이 실제 행동으로 이어지는가로 판단한다. 새 diffusion architecture나 scheduler를 만났을 때 state·time·target·condition·update·checkpoint를 찾고, 수식과 코드, 장애 trace와 평가를 한 경로로 설명할 수 있어야 한다.

noise schedule을 정보 손실 곡선으로 해석한다

`x_t`와 `x_0` 사이 mutual information은 noise가 커질수록 줄어든다. log-SNR은 이 정보 손실의 유용한 좌표지만 실제 data distribution과 VAE latent anisotropy 때문에 모든 방향이 같은 속도로 지워지지는 않는다. latent covariance의 principal direction별 signal·noise 비를 probe한다.

schedule이 어떤 log-SNR 구간에 step을 많이 배치하는지와 data의 coarse·fine structure가 어느 구간에서 사라지는지 연결한다. 이는 “초기에는 구도, 후기에 세부”라는 직관을 검증할 출발점이지 고정 법칙이 아니다. frequency·semantic probe와 timestep intervention으로 확인한다.

schedule 비교의 공통 축부터 살펴보자. linear beta, cosine, Karras sigma와 flow time을 raw index가 아니라 log-SNR·sigma 분위수에서 비교한다. train sample density와 inference step density를 같은 plot에 둔다. model이 적게 본 구간에 solver가 큰 update를 두는지 찾는다.

조건 정보가 trajectory에 들어오는 시점을 측정한다. 같은 initial noise에서 condition만 바꾸고 step별 prediction difference를 본다. text noun·style, image condition과 control signal이 어느 time에서 field를 크게 바꾸는지 분해한다. condition 차이가 초기에만 크거나 후기에만 크다는 관찰을 prompt family와 architecture 전체로 일반화하지 않는다.

cross-attention output과 adaptive norm gate를 block별로 patch해 condition effect를 찾는다. 21장의 multimodal connector counterfactual과 같은 방법이다. condition embedding이 달라져도 model이 이를 사용하지 않으면 prediction difference는 작다.

이 흐름에서 condition leakage도 빠뜨릴 수 없다. training caption에 filename·watermark나 split marker가 있으면 model이 shortcut을 쓸 수 있다. condition shuffle, masked keyword와 source-family split에서 효과를 본다. 4·24장의 corpus contamination 규율을 적용한다.

학습 loss의 reduction을 tensor로 검산한다. pixel·latent·token 차원의 unreduced error에서 valid mask와 timestep weight를 적용하고 sample·batch를 줄이는 순서를 적는다. video frame와 discrete token padding을 제외한다. sample 평균 뒤 batch 평균과 모든 elements 합 뒤 나누는 방식은 shape 분포가 다를 때 같지 않다.

distributed rank마다 valid elements가 다르면 local mean all-reduce는 global element mean이 아니다. numerator와 denominator를 따로 reduce하거나 의도한 sample estimand를 명시한다. gradient accumulation과 DDP average 계수까지 2장의 한 step 수식으로 돌아가 검산한다.

loss log의 충분조건부터 살펴보자. weighted scalar 외에 unweighted numerator, valid count, weight distribution과 time·resolution bucket을 남긴다. 그래야 batch mixture 변경 뒤 curve를 비교할 수 있다. NaN·empty mask를 0으로 기록하지 않는다.

다음 경계는 pipeline optimization과 모델 변경의 경계이다. VAE tiling, attention slicing, CPU offload, compile, fused kernel과 quantization은 같은 checkpoint의 실행 비용을 바꿀 수 있지만 수치도 달라질 수 있다. 최적화 전후 condition·initial state와 time grid를 고정하고 first divergence를 찾는다. 최종 image만 비교하지 않는다.

batching과 prompt ordering이 seed assignment를 바꾸면 최적화와 stochastic sample 변화가 섞인다. sample별 generator와 trajectory ID를 사용한다. dynamic batching에서 condition/uncondition pair가 분리되지 않는지 검사한다.

이 흐름에서 성능 보고도 빠뜨릴 수 없다. denoiser NFE·latency, VAE·condition encode와 end-to-end를 분리한다. cold compile과 steady state, p50·tail, HBM과 energy를 함께 본다. 24장의 paired quality와 같은 bundle manifest를 쓴다.

이제 살펴볼 것은 독립 검토자의 종단 재계산이다. 검토자는 저장된 alpha·sigma 또는 transition 배열만으로 forward state와 exact target을 만들고, model output을 `x_0`나 score·velocity로 변환한다. 이어 solver의 다음 state를 package 없이 계산한다. discrete fixture는 posterior probability 합과 terminal unresolved mask를 확인한다.

두 번째로 checkpoint resume의 다음 batch·time·noise와 optimizer·EMA update를 uninterrupted reference와 비교한다. 세 번째로 inference trajectory 첫 차이의 owner를 condition, model, scheduler, VAE 중 하나로 분류한다. 범위를 확인하지 못하면 결론을 그 경계 앞에서 멈춘다.

이 재계산이 통과해야 새 scheduler default나 kernel optimization, 편집 delta를 안전하게 받아들일 수 있다. 22장의 지식은 특정 sampler recipe가 아니라 확률 경로와 실행 state의 불변식을 찾아 검증하는 방법이다.

forward noising을 재매개화 가능한 확률 경로로 읽는다. continuous data에서 forward state를 `x_t=alpha_t x_0+sigma_t epsilon`으로 쓸 때 `alpha_t`, `sigma_t`가 어떤 convention을 따르는지 먼저 고정한다. variance-preserving, variance-exploding과 flow interpolation은 같은 기호를 다르게 쓸 수 있다. scheduler object의 stored arrays와 paper 식을 직접 매핑한다.

주어진 `x_0`, `epsilon`, `t`에서 `x_t`를 package 없이 계산하는 scalar·tiny tensor fixture를 만든다. batch별 timestep broadcast shape, sample dtype와 device를 확인한다. noise generator가 sample별인지 global인지, resume에서 RNG가 복원되는지도 기록한다.

SNR `alpha_t^2/sigma_t^2`와 log-SNR은 time index보다 schedule 비교에 유용하지만 `alpha^2+sigma^2=1`이 아닌 path에서는 해석이 달라진다. endpoint와 clipping, zero sigma division을 처리한다. schedule plot만 보고 training density를 추정하지 않고 actual sampled time ledger를 본다.

VAE latent에서 `x_0`는 pixel이 아니라 scaled latent다. encoder sample·mode, scaling factor와 latent statistics가 path의 시작점을 정한다. VAE revision이나 scale이 바뀌면 같은 noise seed라도 model input이 달라진다.

video·audio는 frame·time mask를 forward noising과 loss에 반영한다. padded elements를 noise로 채워도 loss count에서 제외하는지 확인한다. modality별 covariance가 달라 isotropic noise 직관이 얼마나 맞는지 probe한다.

fixture 22-FN. scalar `x_0`, fixed epsilon와 세 timesteps에서 forward state, conditional mean·variance를 계산한다. alpha/sigma swap, VAE scale 누락, per-sample broadcast 오류를 각각 주입한다.

prediction parameterization을 선형 변환 표로 닫는다. model이 epsilon, clean sample `x_0`, score 또는 velocity `v`를 예측할 수 있다. 같은 `x_t`에서 이 표현들은 alpha·sigma에 의존하는 선형 변환으로 연결되지만 endpoint에서 수치 조건이 나빠질 수 있다. training target과 inference scheduler가 기대하는 prediction type을 별 state로 둔다.

velocity convention은 repository마다 부호·계수가 다를 수 있다. 이름 `v_prediction`만으로 식을 채우지 않고 target construction 함수와 scheduler conversion을 고정 revision에서 읽는다. tiny fixture로 epsilon↔x0↔v 왕복을 검산한다.

loss weighting은 parameterization에 따라 timestep별 의미를 바꾼다. uniform MSE라도 epsilon과 x0 prediction은 SNR 구간별 gradient scale이 다르다. min-SNR, p2와 custom weights는 numerator·denominator와 coefficient schedule을 기록한다.

inference에서 wrong prediction type을 쓰면 shape와 finite 값은 정상인데 trajectory가 틀어진다. load admission에서 model config, scheduler config와 training manifest를 교차 검증한다. default fallback으로 추정하지 않는다.

learned variance가 model output channel에 concat되면 mean prediction과 variance slice, loss terms를 분리한다. split axis·range와 detach policy를 source에서 확인한다. serving export가 variance head를 버리지 않는지 본다.

반증 실험 22-PT. same checkpoint에 prediction type만 바꾸고 first model output은 같지만 first conversion·next state가 갈리는지 확인한다. 차이가 model call 전에 나타나면 input scaling도 바뀐 것이다.

time sampling을 Monte Carlo estimator로 검증한다. training objective가 time distribution `q(t)` 아래 expectation이면 sampler가 non-uniform할 때 importance weight가 필요할 수 있다. 목표 measure와 실제 draw distribution, loss weight를 한 식에 둔다. 단순히 특정 timesteps를 많이 뽑고 같은 평균을 쓰면 objective가 바뀐다.

discrete index sampler는 number of train timesteps, endpoint inclusion과 index→continuous time mapping을 가진다. continuous sampler는 uniform `t`, log-SNR·sigma density 등일 수 있다. random variate와 transformed time을 둘 다 ledger에 둔다.

loss-aware sampler가 recent loss로 q를 바꾸면 controller state, smoothing window와 distributed aggregation을 checkpoint한다. worker prefetch에 old sampler revision이 남는지 publication boundary를 기록한다. resume 뒤 q가 초기화되면 trajectory가 달라진다.

stratified·antithetic sampling은 variance를 줄일 수 있지만 batch 내 correlation과 rank partition을 만든다. rank-local strata가 global distribution을 중복하지 않는지 본다. world-size 변경에서 보장 등급을 새로 정한다.

time bucket metric은 q에 따른 observed mean과 target objective estimate를 구분한다. rare bucket의 high loss가 sample count 부족인지 model failure인지 confidence interval을 포함한다. empty bucket을 0으로 기록하지 않는다.

수치 실험 22-TS. 세 timesteps, 알려진 per-time loss와 non-uniform q로 unbiased weighted estimate와 biased naive mean을 손으로 계산한다. accumulation·DDP global count까지 selected parameter delta로 비교한다.

solver를 state transition과 history 소유권으로 읽는다. inference scheduler의 `step`은 model output, current sample, time와 internal history를 받아 next sample과 predicted clean sample을 반환할 수 있다. Euler-like one-step, multistep와 stochastic solver를 같은 이름으로 묶지 않는다. 함수 signature보다 instantiated class와 state를 기록한다.

multistep method는 이전 model outputs·derivatives와 order counter를 보존한다. prompt batch 변경이나 trajectory 재사용 사이 history가 섞이지 않도록 TrajectoryID에 귀속한다. resume·pause를 지원하면 history와 current time index를 저장해야 한다.

stochastic step은 추가 noise와 generator state를 소비한다. model prediction이 같아도 RNG가 다르면 next state가 다르다. deterministic solver와 같은 exact fixture를 요구하지 않되 seed·draw identity를 고정한다.

input scaling 함수가 model call 전에 sample을 바꿀 수 있다. 새 scheduler에서 model output부터 달라졌다면 solver step만 바뀐 것이 아니다. scaled input checksum과 time embedding을 비교한다.

integer timestep, sigma lookup과 floating comparison은 device·dtype에 민감하다. scheduler arrays가 CPU FP64인데 model input은 GPU dtype일 수 있다. index rounding과 duplicate timestep을 negative fixture로 넣는다.

NFE는 loop step 수와 다를 수 있다. guidance의 conditional/unconditional batch, predictor-corrector와 higher-order evaluation을 센다. solver quality·latency 비교는 model calls와 VAE·condition cost를 분리한다.

flow matching의 vector field와 ODE 적분을 연결한다. flow matching은 선택한 probability path `x_t`의 conditional velocity target을 학습한다. linear interpolation이면 target이 단순해 보이지만 coupling, time orientation과 data/noise endpoint convention을 확인해야 한다. `t=0`이 data인지 noise인지 repository마다 다를 수 있다.

model output field `u_theta(x_t,t,c)`와 training target을 tiny pair에서 계산한다. time sampling density와 weight를 포함한다. diffusion velocity와 flow velocity를 같은 `v` 기호 때문에 혼동하지 않는다.

inference는 learned field를 ODE solver로 적분한다. Euler·Heun 등에서 step size, direction과 model evaluation points를 손으로 전개한다. model·field가 같아도 solver와 grid가 quality·cost를 바꾼다.

continuity equation은 density가 vector field를 따라 보존되는 조건을 설명하지만 finite neural approximation과 numerical solver가 정확한 density를 보장한다는 뜻은 아니다. trajectory divergence와 endpoint quality를 관측한다.

classifier-free guidance를 field에 적용하면 conditional·unconditional vector의 affine combination이 된다. 큰 guidance는 training field 밖으로 나갈 수 있다. field norm, angle과 solver stability를 함께 본다.

fixture 22-FM. 1차원 Gaussian endpoints와 analytic linear field에서 training target과 Euler·Heun next state를 계산한다. time orientation, step sign과 guidance formula 오류를 주입한다.

DiT block을 token·condition·gate 상태로 해부한다. Diffusion Transformer는 image·video latent를 patch/token sequence로 바꾸고 time·condition을 block에 주입한다. patchify shape, spatial·temporal position와 unpatchify inverse를 먼저 닫는다. padding·crop 때문에 latent extent가 달라지는지 본다.

time embedding은 scalar time을 sinusoidal·MLP representation으로 바꿀 수 있다. sigma·log-SNR이나 integer index 중 실제 input을 확인한다. scheduler time과 model time convention이 다르면 finite하지만 틀린 conditioning이 된다.

adaptive norm은 condition에서 shift·scale·gate를 만들고 attention·MLP branch를 조절한다. gate initialization이 near-zero인지, branch order와 residual equation을 source에서 복원한다. classifier-free condition drop이 어느 embedding을 대체하는지 본다.

cross-attention condition tokens는 text encoder revision, mask와 projection을 가진다. cached embeddings의 tokenizer·encoder·dtype key를 고정한다. prompt ordering과 negative condition batch pairing이 guidance에서 유지되는지 검사한다.

sequence parallel은 latent tokens를 나누고 attention collective를 추가한다. global spatial·temporal positions와 mask, condition replication을 보존해야 한다. distributed output을 dense tiny DiT와 비교한다.

checkpoint에는 denoiser, condition encoder freeze 상태, VAE, EMA와 config를 구분한다. serving pipeline이 다른 VAE·text encoder를 조용히 결합하지 않도록 bundle admission을 둔다.

EMA를 별도 parameter trajectory로 관리한다. EMA weight `theta_ema`는 optimizer-updated `theta`와 다른 state다. decay convention, update frequency, warmup과 skipped optimizer step에서 갱신 여부를 source로 확인한다. evaluation·serving이 raw인지 EMA인지 ModelVariantID를 둔다.

gradient accumulation 중 microstep마다 EMA를 갱신하면 optimizer effect마다 갱신하는 것과 다르다. AMP overflow로 parameter update가 없는데 EMA counter만 전진하는지 본다. scheduler step과 같은 UpdateID에 연결한다.

distributed sharding에서 EMA도 parameter와 같은 global identity·shard를 가져야 한다. full EMA를 한 rank에 모으는 방식은 메모리 peak와 checkpoint cost를 만든다. reshard 후 row·tensor canary를 비교한다.

checkpoint save에서 raw weights는 update k, EMA는 k-1인 혼합을 막는다. snapshot boundary와 manifest에 두 variant의 UpdateID를 둔다. load 뒤 next EMA update를 uninterrupted control과 비교한다.

EMA decay 변경이나 누락은 model loss curve에 즉시 나타나지 않고 sample quality에서만 보일 수 있다. selected parameter의 raw·EMA delta와 effective averaging horizon을 기록한다.

실패 주입 22-EM. EMA field 누락, raw/EMA key swap, skip step에서 잘못된 update와 sharding permutation을 넣는다. bundle validator와 first next EMA delta가 잡아야 한다.

diffusion checkpoint를 pipeline bundle로 봉인한다. 학습 재개 checkpoint는 denoiser raw·EMA, optimizer·scheduler·scaler, RNG, time sampler/controller, data cursor와 accumulation state를 가진다. inference bundle은 선택 weight variant, VAE, condition encoders, tokenizer·processor, scheduler와 generation defaults를 가진다. 두 artifact role을 구분한다.

VAE scaling, latent channels, patch size와 denoiser config를 교차 검증한다. shape가 맞아도 scaling factor가 다르면 trajectory가 처음부터 달라진다. condition encoder hidden과 projection, tokenizer revision도 admission matrix에 둔다.

resume fixture는 next raw batch, sampled time·noise, model loss, optimizer와 EMA delta를 uninterrupted control과 비교한다. data·noise RNG owner를 나누고 prefetch를 고려한다. load success나 비슷한 loss만으로 통과하지 않는다.

inference golden trajectory는 condition bytes·embeddings, initial noise, time grid, scaled inputs, model predictions, scheduler states와 decoded output을 선택 step에서 저장한다. full image hash만 사용하지 않고 first divergence를 찾는다.

component upgrade는 한 번에 하나씩 한다. scheduler만 바꾸면 model scaled input이 같아야 하는지 convention을 확인하고, VAE만 바꾸면 latent부터 의도적으로 달라진다. bundle 전체를 바꾸고 품질만 비교하지 않는다.

partial checkpoint·object store commit은 17장의 durable generation protocol을 따른다. EMA·optimizer shard 누락을 model weight 존재로 숨기지 않는다. serving export는 새로운 ArtifactID와 provenance를 만든다.

## 22.15 discrete 구현·대규모 복구·관측·certificate를 닫는다

categorical corruption에서 대규모 ownership, Diffusers scheduler state, time/sigma 관측과 최종 source artifact까지 인수한다.

discrete token diffusion은 vocabulary state에 forward transition을 적용한다. uniform replacement, absorbing mask와 structured corruption은 서로 다른 Markov chain이다. transition matrix 또는 closed-form marginal, special token 처리와 terminal distribution을 고정한다.

small vocabulary 3과 mask token 하나로 `q(x_t|x_0)`와 posterior `q(x_{t-1}|x_t,x_0)`를 손으로 계산한다. probability 합, impossible transition과 endpoint를 확인한다. padding·BOS·EOS가 corrupt 가능한지 loss mask와 분리한다.

model은 clean token, previous state distribution, score-like logits나 transition parameter를 예측할 수 있다. training target과 sampler interpretation을 맞춘다. continuous diffusion의 epsilon prediction 용어를 그대로 옮기지 않는다.

parallel decoding은 여러 masked positions를 confidence로 선택해 갱신할 수 있다. confidence 계산, tie, reveal schedule와 re-mask policy가 sampler state다. 같은 model logits에서도 decision rule이 달라 output과 NFE가 바뀐다.

tokenizer vocabulary 변경은 transition width와 model head, mask ID를 모두 바꾼다. added token migration과 physical padded rows를 확인한다. old transition cache를 재사용하지 않는다.

distributed sequence shard는 global unresolved mask와 reveal count를 일관되게 계산해야 한다. rank-local top-k로 선택하면 global schedule과 다르다. dense tiny oracle과 비교한다.

failure suite 22-DD. mask ID swap, posterior normalization 누락, padding corruption, confidence tie와 terminal unresolved token을 주입한다. 각 오류가 sampler 종료 전 잡혀야 한다.

### consistency·distillation을 teacher trajectory 계약으로 읽는다

적은-step 생성은 단순히 inference step을 건너뛰는 것이 아니다. consistency training·distillation은 teacher field나 trajectory, boundary condition과 student mapping을 사용한다. teacher checkpoint, solver, time-pair sampling과 target construction을 immutable parent로 둔다.

같은 noisy state에서 teacher가 어느 두 times를 연결하는지, stop-gradient가 어디에 있는지 source를 읽는다. target network·EMA가 있으면 student optimizer와 별 state다. target update frequency와 checkpoint를 기록한다.

teacher trajectory cache는 VAE, condition, scheduler·solver, prediction type와 dtype을 key에 포함한다. stale cache가 존재하면 student loss는 정상적으로 내려가도 다른 teacher function을 학습한다. live recompute fixture와 비교한다.

distillation loss denominator는 time pair, sample와 valid elements 중 무엇인지 확인한다. teacher uncertainty·guidance와 weighting을 따로 기록한다. distributed local mean 오류를 2장의 numerator/count 방식으로 검산한다.

student step 수가 줄어도 per-step model size·guidance batch와 VAE cost가 같지 않을 수 있다. NFE와 end-to-end latency, memory·energy를 품질과 함께 비교한다. teacher와 student의 compute budget을 명시한다.

평가는 paired condition·initial state에서 teacher, base solver와 student를 비교한다. final metric뿐 아니라 selected intermediate clean prediction과 failure slice를 본다. teacher artifact를 바꾸면 새 distillation lineage다.

### diffusion data·condition pipeline을 학습 target과 연결한다

image·audio·video decode, resize·crop·resample·frame sampling은 `x_0` distribution을 만든다. transform revision과 RNG, selected crop·timestamps를 SampleID에 붙인다. resume에서 동일 transform을 요구하는지 등급을 정한다.

caption·condition은 tokenizer·text encoder·template를 거쳐 embeddings가 된다. classifier-free condition drop은 raw text 삭제, learned null embedding 또는 attention mask 변경일 수 있다. drop RNG와 unconditional representation을 저장한다.

resolution·aspect bucket은 latent shape, patch sequence와 batch memory를 바꾼다. bucket sampler의 target·realized mass, padding·crop loss와 rank tail을 본다. 특정 aspect를 반복해 throughput을 맞추면 data objective가 달라진다.

video clip curriculum은 duration·frame rate와 motion distribution을 함께 바꾼다. 같은 frame count라도 timestamp span이 다를 수 있다. temporal mask와 loss valid count, condition alignment를 확인한다.

caption filtering·aesthetic weighting은 source bias와 duplication을 만들 수 있다. detector score, policy threshold와 loss weight를 분리한다. watermark·filename shortcut과 benchmark contamination을 counterfactual로 검사한다.

packed multimodal batch에서 sample별 time·noise와 loss elements를 추적한다. variable shape를 sample mean으로 줄일지 element mean으로 줄일지 objective를 명시한다. data pipeline ledger가 22.80 time sampler와 22.75 denominator에 연결돼야 한다.

### diffusion 평가에서 model·solver·decoder를 분리한다

FID·CLIP score·human preference 같은 aggregate metric은 component 귀속을 직접 주지 않는다. 같은 condition·initial noise에서 model field, scheduler trajectory와 VAE decode를 단계별로 고정하는 2×2 또는 component swap을 설계한다.

paired evaluation은 prompt·seed·condition과 output pairing을 보존한다. batch ordering이 generator assignment를 바꾸지 않게 sample별 RNG를 쓴다. invalid·filtered output과 retry를 score에서 조용히 제외하지 않는다.

metric encoder revision, preprocessing와 sample count를 EvalID에 둔다. FID reference statistics가 dataset·resolution과 맞는지 확인한다. small sample uncertainty와 multiple comparisons를 보고한다.

human evaluation은 side randomization, judge rubric, annotator·model revision과 tie를 기록한다. aesthetic preference가 text faithfulness·safety를 가리지 않게 축을 나눈다. 자동 judge와 생성 model의 shared bias를 counterfactual로 본다.

video·audio는 temporal consistency, motion·sync와 per-frame quality를 분리한다. 이미지 metric을 frame 평균해 전체 품질로 대체하지 않는다. duration·aspect·language·condition slice를 둔다.

solver comparison은 NFE, condition guidance evaluations, VAE·encoder와 compile warmup을 보고한다. latency와 quality Pareto를 제시하며 빠른 설정이 seed·resolution을 바꿔 얻은 이득을 금지한다.

출시 관문는 quality, condition adherence, diversity, safety, memorization·privacy와 robustness를 분리한다. 평균 향상이 중요한 slice 하락을 덮지 못하게 hard floor를 둔다.

diffusion 장애를 first-divergence runbook으로 운영한다. loss가 즉시 NaN이면 decoded·latent `x_0`, alpha·sigma, sampled time, target와 model output을 순서대로 본다. VAE scale·dtype, endpoint division과 weighted denominator를 확인한다. optimizer epsilon이나 gradient clip을 먼저 바꾸지 않는다.

loss는 내려가지만 sample이 무의미하면 training prediction type과 inference scheduler conversion, VAE·condition bundle을 교차 검증한다. fixed noisy state의 target·model prediction과 first solver state를 손으로 계산한다.

resume 뒤만 갈리면 next data, time·noise RNG, optimizer·EMA, sampler controller와 accumulation을 비교한다. raw weight와 EMA variant가 바뀌지 않았는지 본다. first optimizer·EMA delta가 종료 조건이다.

특정 resolution·duration에서만 OOM이면 latent tokens, attention backend, activation checkpoint, VAE와 condition memory를 phase별로 센다. batch를 drop하지 않고 token budget·shape bucket을 명시적 child recipe로 바꾼다.

multi-rank hang은 variable token count, sequence/context collective와 one-rank OOM을 시간축에서 찾는다. rank별 time bucket·shape와 collective sequence를 저장한다. timeout 증가로 sequence divergence를 숨기지 않는다.

품질이 scheduler upgrade 뒤만 변하면 scaled model input, time grid, prediction conversion, history와 RNG를 한 경계씩 비교한다. model output 전에 갈리면 solver-only 변경이 아니다.

IncidentID에는 PipelineBundleID, CheckpointID, condition·initial state, first different tensor, 소스 좌표, recovery와 negative fixture를 둔다. 복구 뒤 clean control과 failure suite를 모두 반복한다.

diffusion repository를 함수 순서로 읽는다. config에서 state space, VAE scale·channels, prediction type, train timesteps·schedule, time sampler, loss weight, condition drop, model architecture, EMA와 precision을 추출한다. serialized config와 library defaults·runtime override를 합쳐 effective manifest를 만든다.

dataset path에서는 decode·transform, condition encoding, cache와 collator를 따라간다. sample ID, resolution·duration, condition bytes와 transform RNG가 batch까지 보존되는지 본다. variable shape의 mask와 valid loss count를 찾는다.

training step은 raw sample→latent `x_0`→time·noise→`x_t`→model input scaling→prediction→target conversion→unreduced loss→weight·reduction 순으로 추적한다. 각 함수의 shape·dtype, RNG와 state mutation을 표로 둔다. training script의 log scalar만 보지 않는다.

model source에서는 patchify, time·condition embedding, DiT/UNet blocks, attention·norm·gate와 output head를 읽는다. modular/generated source와 runtime fused path를 구분한다. checkpoint key가 있다는 사실을 caller evidence로 쓰지 않는다.

scheduler source는 time grid construction, input scaling, prediction conversion, variance와 `step` history를 읽는다. pipeline loop가 scheduler output을 어떻게 소비하고 VAE로 넘기는지 위아래 caller를 연결한다. solver class 이름보다 state transition을 식으로 옮긴다.

optimizer 경계에서는 accumulation, distributed denominator, scaler, clipping, raw step, EMA와 scheduler counter의 순서를 확인한다. checkpoint save field와 load 순서를 대칭 표로 만든다. time-sampler controller와 data cursor 누락을 찾는다.

tests는 formula·scheduler·pipeline·training resume·distributed·component compatibility로 분류한다. upstream inference test가 training backward나 current VAE bundle을 증명한다고 과장하지 않는다. 미검증 조합을 matrix에 남긴다.

최종 SourceCard에는 commit, symbol·caller, equation, tensor/state, option, test, checkpoint와 first-failure owner가 있다. revision diff에서는 prediction conversion·default schedule과 bundle admission을 우선 재검증한다.

저정밀·quantization을 trajectory 오차로 평가한다. training autocast는 denoiser matmul, attention, norm, target·loss와 optimizer state에 서로 다른 dtype을 적용할 수 있다. `x_t`와 alpha·sigma 계산이 낮은 precision에서 endpoint 정보를 잃지 않는지 본다. loss numerator와 sensitive conversion은 FP32 reference에 대조한다.

FP8·quantized training은 activation·weight scale, amax history와 master state를 추가한다. time·condition에 따라 activation distribution이 달라져 하나의 평균 scale이 특정 sigma에서 saturation을 만들 수 있다. time bucket별 overflow·quantization error를 본다.

inference quantization은 model field를 바꾸므로 같은 solver에서도 trajectory 오차가 누적된다. paired condition·initial state에서 step별 model prediction, clean estimate와 state distance를 기록한다. final image metric만으로 first divergence를 찾을 수 없다.

VAE·text encoder와 denoiser를 각각 quantize할 수 있다. VAE 변경은 state space encode/decode, condition encoder는 conditioning, denoiser는 field를 바꾼다. component별 2×2 swap으로 귀속한다.

attention fused kernel은 sequence·head dim·mask와 dtype에 따라 fallback한다. particular resolution·video length에서만 path가 달라질 수 있다. selected backend, memory, latency와 eager numerical probe를 같은 TrajectoryID에 둔다.

solver arrays와 multistep history의 dtype도 중요하다. model output만 FP32로 올려도 sigma subtraction이나 history combination이 낮은 precision이면 오차가 난다. analytic field fixture로 integration error를 측정한다.

quantized export는 training resume checkpoint가 아니다. scale metadata, packed layout, VAE·condition·scheduler bundle과 calibration provenance를 새 ArtifactID로 만든다. checkpoint converter round trip과 selected output canary를 검사한다.

성능 승인은 correctness를 통과한 동일 condition·seed·NFE에서 한다. memory·latency 개선, trajectory error와 paired quality의 허용 budget을 사전에 둔다. tolerance를 결과 뒤 넓히지 않는다.

이제 살펴볼 것은 독립 검토 체크리스트이다. **state path.** raw sample과 condition에서 latent·time·noise·`x_t`를 재계산했는가. VAE scale, mask와 RNG owner가 고정됐는가. forward path equation과 stored scheduler arrays가 맞는가.

target. epsilon·x0·score·velocity convention, variance slice와 timestep weight를 source target 함수에서 확인했는가. numerator·valid count와 DDP·accumulation scale을 재구성했는가.

model. patch/token positions, time·condition embedding, norm·gate·attention과 output head의 shape·dtype를 추적했는가. selected fused/compiled branch와 backward를 eager oracle에 맞췄는가.

solver. time grid, input scaling, prediction conversion, history, stochastic RNG와 NFE를 기록했는가. first next state를 package 없이 계산했는가. model과 solver 변경을 구분했는가.

bundle. raw·EMA weight, VAE, condition encoders, tokenizer·processor, scheduler와 generation defaults가 compatible revision인가. mutable component를 조용히 섞지 않는 admission gate가 있는가.

resume. next data·time·noise, optimizer·scheduler·scaler, EMA, sampler controller와 cursor가 복원되는가. first raw·EMA delta를 uninterrupted control과 비교했는가.

분산. variable valid elements의 global denominator, sequence/context ownership과 global position을 dense reference에 맞췄는가. zero-valid rank·uneven shape와 collective failure를 시험했는가.

평가. paired condition·seed, NFE·latency, model·solver·decoder component와 metric revision을 분리했는가. quality·diversity·safety·memorization의 hard floors와 uncertainty가 있는가.

반증. wrong prediction type, VAE scale, stale condition, missing solver history, EMA swap와 precision overflow가 expected first detector에서 잡히는가. failure 뒤 clean control을 반복했는가.

모든 답은 소스 좌표, tensor artifact나 executable fixture를 가리킨다. 미실행 modality·resolution·solver는 필요한 input·명령·expected invariant와 함께 `NotExecuted`로 남긴다.

관측성을 data·field·solver·decode 네 plane으로 나눈다. data plane은 sample·condition revision, resolution·duration, latent statistics, time·noise distribution과 valid elements를 본다. raw prompt·media를 metric label로 내보내지 않고 bucket과 bounded canary를 쓴다. transform failure와 cache revision을 연결한다.

field plane은 time bucket별 weighted·unweighted loss, model prediction·target norm, gradient, condition/uncondition difference와 nonfinite를 본다. 평균 loss는 endpoint나 rare resolution 실패를 숨기므로 sigma·shape·condition slice의 tail을 둔다.

solver plane은 scaled input, prediction conversion, state norm, history order, NFE와 step latency를 본다. stochastic noise draw와 TrajectoryID를 연결한다. scheduler 이름만 label로 두지 않고 effective config digest를 사용한다.

decode plane은 VAE input·output range, tiling·precision, decode latency와 invalid outputs를 본다. field가 정상인데 image만 깨지면 VAE·postprocess로 범위를 좁힌다. safety filter가 output을 제거한 경우 generation failure와 분리한다.

각 plane은 같은 RunID·CheckpointID·PipelineBundleID와 selected condition·seed를 공유한다. wall-clock 순서만으로 다른 trajectory의 metric을 결합하지 않는다. high-cardinality step trace는 bounded forensic artifact에 둔다.

alert는 first detector와 effect barrier를 가진다. nonfinite target은 model backward 전에, wrong bundle은 generation 전에, stale solver history는 next state 전에 실패해야 한다. 늦은 quality metric만 경보로 쓰지 않는다.

monitoring outage는 성공 값 0이 아니라 unknown이다. telemetry 누락 중 자동 promotion을 막고 raw training effect와 checkpoint는 별도 ledger로 보존한다. 관측 복구 뒤 gap 범위와 release 영향 평가를 남긴다.

장기 baseline은 model·data·schedule revision과 hardware·kernel에 조건부다. curriculum·resolution phase가 바뀌면 예상 범위를 새로 승인한다. threshold를 incident 결과에 맞춰 조용히 넓히지 않는다.

사전학습·조건 미세조정·LoRA의 state 차이를 구분한다. diffusion pretraining은 denoiser 전체, condition encoder·VAE의 freeze 정책과 data·time objective를 가진다. fine-tuning은 full denoiser, attention·norm 일부, adapter나 textual embedding만 업데이트할 수 있다. trainable count가 아니라 parameter identity와 optimizer group을 기록한다.

LoRA target은 attention Q/K/V/O, MLP나 convolution에 붙을 수 있다. fused projection에서 logical slice와 physical parameter가 다르므로 target matcher와 rank·scale을 source에서 확인한다. adapter dropout·dtype와 base freeze를 검사한다.

textual inversion류는 tokenizer added token, text encoder embedding row와 optimizer state를 바꾼다. token→ID, row migration과 serving tokenizer bundle을 함께 배포한다. decoded string이 같다는 이유로 ID mismatch를 허용하지 않는다.

DreamBooth·subject fine-tuning은 repeated subject data, prior-preservation data와 loss weight를 가진다. 두 numerator/count, batch composition과 image·caption duplication을 분리한다. class prior가 stale generator에서 왔으면 lineage를 남긴다.

ControlNet·adapter 계열은 condition branch와 zero-initialized connector, base freeze와 merge·scale를 가진다. training checkpoint와 inference pipeline의 branch revision을 맞춘다. connector를 0으로 했을 때 base output parity가 성립하는지 counterfactual로 본다.

adapter merge는 serving artifact이고 optimizer resume state를 대체하지 않는다. merge 전후 field output을 selected times·conditions에서 비교한다. quantized base나 fused layout에서 merge rounding과 rollback을 검증한다.

preference·reward fine-tuning을 image/video에 적용하면 paired data, reward model·judge와 generation solver가 추가된다. reward 상승과 text adherence·diversity·safety를 분리하며 stale rollout과 policy version 원칙을 19·20장에서 가져온다.

각 fine-tuning recipe의 checkpoint에는 base, adapter/connector, tokenizer·condition encoder, VAE, optimizer·EMA와 data objective를 연결한다. 이름이 같은 base repository의 mutable revision을 허용하지 않는다.

대규모 diffusion 학습의 병렬 소유권과 복구를 검증한다. data parallel은 서로 다른 samples·times를 처리하고 gradient를 reduce한다. variable resolution·duration과 valid elements가 rank마다 다르면 local mean 평균이 target global objective와 다를 수 있다. numerator와 denominator, DDP averaging factor를 손으로 전개한다.

tensor parallel은 attention·MLP와 patch projection을 shard하고, sequence/context parallel은 latent tokens를 나눈다. spatial·temporal global position, condition tokens와 mask가 같은 logical sequence를 나타내야 한다. dense tiny DiT를 분할했다가 복원해 forward·gradient를 비교한다.

pipeline parallel은 blocks를 stage로 나누며 activation shape가 resolution·duration에 따라 변한다. microbatch schedule, stage boundary tensor와 bubble을 기록한다. condition encoder·VAE를 별 stage·service로 두면 version과 failure domain을 분리한다.

resolution bucket을 rank마다 독립적으로 고르면 shape collective와 load가 어긋날 수 있다. global batch plan과 rank assignment, maximum token budget을 만든다. 한 rank OOM 뒤 다른 rank hang을 별 사건으로 보지 않고 최초 shape·memory를 찾는다.

EMA·optimizer sharding은 raw parameter와 같은 global identity를 보존한다. checkpoint reshard에서 model만 옮기고 EMA·moments를 local order로 붙이지 않는다. tensor-specific canary와 first delta를 비교한다.

multi-cluster training은 dataset·VAE latent cache와 checkpoint staging, network topology를 고려한다. stale cache·component bundle이 cluster별로 다르지 않도록 root manifest digest handshake를 둔다. membership generation이 바뀌면 old rank의 effect를 fencing한다.

failure injection에는 rank kill, slow storage, VAE service mismatch, sequence-tail OOM, collective stall과 partial EMA shard를 넣는다. 전 rank가 같은 failure generation으로 중단하고 마지막 durable checkpoint·data/time cursor로 돌아가야 한다.

resume 뒤 first batch의 SampleIDs, transform, time·noise, condition embeddings, loss numerator/count, raw·EMA delta를 uninterrupted control과 비교한다. topology가 달라 exact numerical trajectory를 보장하지 않으면 logical state와 statistical grade를 명시한다.

성능은 denoiser compute, attention collectives, VAE·condition service, data staging와 checkpoint를 분리한다. 평균 GPU utilization보다 slowest rank와 exposed communication tail을 본다. correctness parity를 통과한 layout만 비교한다.

운영 manifest에는 DP·TP·PP·CP mesh, parameter·activation·EMA ownership, process groups, bucket plan, checkpoint schema와 tested topology가 있다. 미실행 cluster 조합은 필요한 resource, command와 expected invariant를 남긴다.

변경 승인표를 component별 first difference로 작성한다. scheduler 변경은 time grid, input scaling, prediction conversion, history와 next state를 바꿀 수 있다. model checkpoint와 condition·initial noise가 같다면 어느 model input까지 같아야 하는지 먼저 선언한다. model output 전 차이는 solver-only 주장과 모순된다.

VAE 변경은 pixel/media↔latent mapping, scale와 decode를 바꾼다. denoiser가 같아도 `x_0`와 initial latent부터 달라진다. VAE 개선을 denoiser 학습 개선으로 쓰지 않는다. latent distribution과 paired decode를 별도로 평가한다.

condition encoder·tokenizer 변경은 embeddings와 cross-attention input을 바꾼다. prompt bytes·IDs·mask와 selected condition vectors를 비교한다. cached old embeddings를 새 encoder revision에 재사용하지 않는다.

prediction type·loss weight 변경은 training target과 gradient를 바꾼다. same noisy input의 target, unreduced error, numerator/count와 first parameter delta에서 차이가 시작돼야 한다. scheduler default도 compatibility matrix에 맞춘다.

attention·norm kernel upgrade는 logical model function 보존을 목표로 할 수 있다. eager reference와 selected times·shapes의 forward/backward, trajectory error를 통과한 뒤 memory·latency를 비교한다. 특정 resolution fallback을 숨기지 않는다.

precision·quantization은 rounding과 scale state를 추가한다. expected first numerical difference, time-bucket error budget과 quality floor를 사전에 둔다. final metric이 비슷하다는 이유로 internal saturation을 허용하지 않는다.

data·curriculum 변경은 sample·condition distribution, resolution·time sampling과 loss mass를 바꾼다. realized ledger와 checkpoint descendants를 연결한다. solver를 동시에 바꾸면 2×2 조합으로 interaction을 분리한다.

EMA decay·variant 변경은 raw training parameter가 아니라 evaluation/serving trajectory를 바꿀 수 있다. raw·EMA IDs와 selected delta를 분리한다. serving bundle이 어느 variant를 읽었는지 admission log로 증명한다.

승인표의 각 행에는 old/new artifact, first changed tensor/state, 유지 invariant, 소스 심볼, fixture, metric, rollback과 reviewer가 있다. `Unknown`과 `NotExecuted`를 PASS로 승격하지 않는다.

마지막 reviewer는 한 행을 선택해 condition·state·target·model·solver·decode를 왕복한다. 변경 작성자의 설명이 아니라 stored arrays와 tensor artifact에서 최초 차이를 다시 계산한다. 이 검토가 통과한 component 조합만 새 PipelineBundleID로 배포한다.

승인 뒤에도 표는 immutable evidence로 남긴다. model·scheduler·VAE·encoder·CUDA 또는 compiler revision 중 하나가 바뀌면 body fingerprint와 caller diff로 affected rows를 stale 처리한다. line 이동만으로 전체 결과를 폐기하지 않지만 semantic change를 과거 PASS로 상속하지 않는다.

운영 canary는 전체 media를 저장하지 않고 approved synthetic condition·seed의 selected trajectory digests를 사용한다. condition encode, initial state, model prediction, next state와 decode summary를 단계별로 확인한다. canary가 실패하면 component bundle publication을 중단하고 old pointer를 유지한다.

rollback은 파일 하나를 되돌리는 것이 아니다. weight variant, VAE, condition encoders, tokenizer·processor, scheduler, precision·kernel options와 cache namespace를 같은 bundle generation으로 복원한다. mixed revision replica를 traffic에 남기지 않는다.

사건 뒤 threshold를 넓혀 canary를 통과시키지 않는다. first difference의 equation·dtype·state owner를 찾고, 수정한 component와 downstream trajectory를 재검증한다. failure trace와 clean control을 모두 새 regression suite에 보존한다.

이 표의 목적은 변화 자체를 막는 것이 아니라 모델 학습, numerical solver와 media decode의 개선을 정확히 귀속하는 것이다. 귀속이 가능해야 새 기법의 효과를 재현하고 실패 시 안전하게 이전 상태로 돌아갈 수 있다.

최종 서명에는 tested modality·resolution·duration, prediction type, time sampler, solver·NFE, raw·EMA variant, VAE·condition bundle, dtype·kernel과 topology를 적는다. 지원하지 않은 discrete vocabulary, stochastic solver나 cluster layout은 필요한 입력·명령·expected invariant와 함께 남긴다.

독립 재현자는 서명만 읽지 않고 같은 artifact root에서 analytic forward fixture, one-step solver, resume delta와 paired evaluation sample을 다시 계산한다. 결과가 다르면 환경·source·bundle digest부터 비교하고, 미확인 차이를 통계적 변동으로 넘기지 않는다.

이 마지막 절차가 수학적 path, training target, implementation branch, distributed state와 생성 품질을 하나의 검증 가능한 계보로 묶는다. 그 계보가 닫혀야 22장을 새로운 diffusion·flow 구현을 조사하는 기준서로 사용할 수 있다.

검토 결과에는 성공 trajectory뿐 아니라 prediction type 오류, stale condition, VAE scale 누락과 solver history 손실이 예상 경계에서 차단된 negative trace를 함께 둔다. 다음 revision은 이 반례를 먼저 통과해야 하며, 새 output으로 기준값을 자동 교체하지 않는다. 이 규칙은 구현 이름이 바뀌어도 검증 의미를 보존한다.

마지막 통합 사고 실험부터 살펴보자. 같은 checkpoint에서 새 scheduler가 FID와 latency를 모두 개선했다고 하자. 먼저 prediction type, input scaling, time·sigma 배열과 VAE가 같은지 확인한다. 동일 condition·initial noise의 model output이 같고 next state부터 달라져야 solver 개선으로 귀속할 수 있다. model call부터 다르면 pipeline이 input을 바꾼 것이다.

step 수가 줄어 model calls가 감소했지만 guidance가 두 배 batch를 만들거나 higher-order solver가 step당 두 evaluation을 한다면 nominal step 대신 NFE와 end-to-end 비용을 본다. compile warm-up과 dynamic shape fallback도 분리한다. 품질 비교는 paired prompt·seed와 여러 condition slice에서 interval을 제시한다.

학습 loss는 그대로인데 새 scheduler 품질이 좋아졌다면 model training이 개선됐다고 쓰지 않는다. 반대로 training objective를 바꾸고 old scheduler로도 좋아졌다면 field가 개선된 evidence다. 두 변화가 함께라면 2×2 조합으로 interaction을 본다.

discrete 모델에서도 같은 질문을 적용한다. confidence rule을 바꿔 품질이 오르면 model probability와 sampler decision을 분리하고, token cross-entropy가 좋아졌다면 고정 sampler에서 비교한다. terminal mask·길이와 compute budget을 맞춘다.

이 사고 실험의 답은 숫자 하나가 아니라 귀속 가능한 lineage다. data·condition, weight, prediction convention, scheduler와 decoder 중 무엇이 바뀌었는지와 첫 trajectory 차이를 남긴다. 변경하지 않은 경계를 golden artifact로 증명한다.

22장의 모든 수학과 코드 설명은 이 귀속을 가능하게 하기 위해 존재한다. probability path와 vector field, transition과 posterior, solver state를 연결하면 새로운 기법의 홍보 문장을 실제 함수·tensor·실험 질문으로 번역할 수 있다.

검토자는 마지막으로 지원 matrix의 조합 하나를 임의 선택한다. model·VAE·condition encoder revision, state 공간, prediction type, time grid, dtype와 attention backend를 확인하고 golden trajectory를 재생한다. config 파일에 값이 있다는 사실이 아니라 instantiated object와 저장 배열이 같은 값을 쓰는지 본다.

negative run에서는 prediction type을 의도적으로 바꾸고 VAE scale을 누락하며 multistep history 하나를 지운다. 각각 load gate, first conversion, decode 또는 resume boundary에서 실패해야 한다. 최종 품질이 나빠진 뒤에만 발견된다면 검증 경계가 부족하다.

운영 metric의 time bucket loss와 trajectory norm, scheduler step latency와 compile fallback을 same run ID로 잇는다. 특정 sigma에서 latency와 수치 오차가 동시에 증가하면 kernel·dtype path를 재현한다. 평균 latency와 평균 loss는 이 tail을 숨길 수 있다.

검증하지 않은 resolution·duration, discrete vocabulary와 stochastic solver는 미지원으로 남긴다. 새 경로를 열 때 analytic fixture, source 좌표, trajectory와 paired evaluation을 추가한다. 이 절차가 유지될 때 diffusion stack은 빠르게 변해도 책의 디깅 방법은 낡지 않는다.

독자는 source audit 결과를 실험 전에 사용한다. scheduler의 integer-index 방어, sample fp32 upcast, prediction conversion과 variance split 같은 분기는 흔한 실패를 미리 알려 준다. tests가 어느 boundary case를 덮는지 읽고 자신의 model·dtype·condition 조합에 빠진 case를 추가한다.

실험 뒤에는 성공 trajectory만 보존하지 않는다. wrong convention, stale condition, missing history와 precision overflow가 각각 기대 gate에서 멈춘 negative trace를 함께 둔다. 이 반례들이 새 revision의 regression test가 된다.

마지막 인수 문장은 범위를 포함한다. “이 state와 target, schedule·solver·condition 조합에서 수식과 구현, 재현 trajectory와 품질·비용을 확인했다.” 그보다 넓은 architecture·modality 주장은 새 evidence를 필요로 한다. 정확한 기술적 범위가 장기적인 실험 재현성과 독립적인 결과 검증 가능성을 함께 지킨다. 다음 새로운 구현 변경과 runtime 최적화도 동일한 엄격한 기준과 golden trajectory에서 독립적으로 다시 평가한다. 그 차이와 모든 미검증 경계를 공식 기록으로 함께 남긴다. 영구 보존한다.

forward diffusion을 조건부 Gaussian의 기하로 다시 유도한다. DDPM의 한 step은 \(q(x_t|x_{t-1})=\mathcal N(\sqrt{1-β_t}x_{t-1},β_tI)\)다. \(α_t=1-β_t\), \(\bar α_t=\prod_{s=1}^tα_s\)라 두면 여러 step을 접어 \(x_t=\sqrt{\bar α_t}x_0+\sqrt{1-\bar α_t}\epsilon\)로 직접 sampling할 수 있다. training loop가 매번 t번 noising하지 않는 이유가 이 Gaussian closure다.

기하적으로 x_t는 clean point x_0와 isotropic noise ε의 직교 좌표 조합이다. 계수 제곱의 합이 1인 variance-preserving convention에서는 t가 증가할수록 data 방향 성분은 줄고 noise 방향 성분은 커진다. 하지만 모든 scheduler가 같은 time convention을 쓰지는 않는다. sigma space, variance-exploding path와 flow interpolation을 ᾱ 식에 억지로 대입하지 않는다.

posterior \(q(x_{t-1}|x_t,x_0)\)도 Gaussian이며 mean은 x_t와 x_0의 schedule-dependent 선형 결합이다. model이 ε를 예측하더라도 sampler는 이를 x_0 estimate로 변환해 posterior mean을 만들 수 있다. Diffusers `DDPMScheduler.step`의 prediction branch와 posterior coefficient가 바로 이 연결이다. training target과 step converter가 다른 convention이면 tensor shape는 맞고 결과도 그럴듯해 failure가 늦게 드러난다.

작은 fixture는 scalar x_0=2, ε=-1과 세 개의 ᾱ를 고정한다. add_noise 결과, ε→x_0 역변환, posterior mean과 variance를 FP64로 계산한다. boundary t=0, clipping과 thresholding branch도 포함한다. library output이 수식과 다르면 schedule array index, dtype와 prediction_type부터 본다.

forward process는 data를 파괴하는 과정이 아니라 training target을 여러 noise scale에서 관측하게 만드는 실험 설계다. 어느 t를 얼마나 자주 sample하고 loss를 어떻게 weight하는지가 model이 어느 SNR 구간을 잘 배우는지 결정한다. schedule를 sampling-only option으로 취급하지 않는 이유다.

score, epsilon과 denoising 방향을 같은 좌표에서 연결한다. score는 noisy marginal의 log density gradient \(s_t(x)=\nabla_x\log p_t(x)\)다. Gaussian corruption에서 conditional score는 \(-(x_t-\sqrt{\bar α_t}x_0)/(1-\bar α_t)=-\epsilon/\sqrt{1-\bar α_t}\)다. ε-prediction과 score prediction은 scale 변환으로 연결되지만 loss weighting과 numerical range는 달라진다.

score vector는 현재 noisy point에서 density가 빠르게 증가하는 방향을 가리킨다. 단순히 nearest clean image로 향하는 화살표는 아니다. p_t는 모든 data sample을 convolution한 marginal이므로 여러 mode의 영향이 섞인다. 고 noise에서는 global structure, 저 noise에서는 local detail에 가까운 field가 나타나는 직관은 유용하지만 각 coordinate의 exact behavior를 보장하지 않는다.

denoising score matching은 true marginal score를 직접 알 필요 없이 sampled x_0와 ε에서 conditional target을 만든다. conditional score의 expectation이 marginal score가 되는 관계를 쓴다. training code에서는 target tensor가 ε인지 scaled score인지 확인하고 loss weight가 이 scale 차이를 상쇄하는지 본다. 논문 표기 sθ와 code의 `model_pred`를 이름만으로 연결하지 않는다.

reverse-time SDE와 probability-flow ODE는 같은 marginal을 가질 수 있지만 sample path와 solver가 다르다. reverse SDE에는 stochastic noise가 남고 ODE는 deterministic vector field를 적분한다. 동일 score model이라도 seed ownership, NFE, solver error와 likelihood 계산 의미가 달라진다. sampler 이름만 바꿔 같은 trajectory를 기대하지 않는다.

score fixture는 1D Gaussian mixture에서 수치 density derivative와 analytic conditional expectation을 비교한다. 여러 t에서 scale을 확인하고 ε target을 score로 변환한 결과가 맞는지 본다. float16에서 작은 sigma division이 overflow하는 case를 포함해 model output보다 converter의 precision을 별도 검증한다.

epsilon·x0·v parameterization을 회전 변환으로 읽는다. variance-preserving path에서 \(x_t=a_tx_0+b_tε\), \(a_t=\sqrt{\bar α_t}\), \(b_t=\sqrt{1-\bar α_t}\)라 두자. velocity target을 \(v_t=a_tε-b_tx_0\)로 정의하면 `(x_t,v_t)`는 `(x_0,ε)`의 직교 회전과 같다. 역변환은 \(x_0=a_tx_t-b_tv_t\), \(ε=b_tx_t+a_tv_t\)다.

ε-prediction은 high noise에서 target scale이 안정적이지만 x_0 복원은 작은 a_t로 나눌 수 있다. x0-prediction은 low noise에서 clean target이 직접적이지만 high noise input에서 data를 추정해야 한다. v-prediction은 time에 따라 두 역할을 회전해 SNR 균형을 다르게 만든다. 어느 것이 우월하다는 보편 명제가 아니라 schedule, weighting과 model에 따른 선택이다.

Diffusers scheduler config의 `prediction_type`은 단순 label이 아니다. `epsilon`, `sample`, `v_prediction` branch가 model output을 pred_original_sample과 ε로 변환한다. checkpoint metadata와 training target generator가 같은 값을 써야 한다. export나 pipeline에서 scheduler만 교체할 때 config default가 달라지지 않는지 admission gate가 검사한다.

golden test는 같은 x_0, ε, t에서 세 target을 만들고 각 target에서 x_0와 ε를 역복원한다. a_t가 0 또는 1에 가까운 boundary, clipping이 켜진 path와 rescale branch를 구분한다. clipping 뒤 ε를 다시 계산하는 option은 trajectory를 바꾸므로 expected output을 별도로 둔다.

loss 비교에서도 raw MSE 숫자를 직접 비교하지 않는다. 선형 target 변환은 time별 scale과 gradient 방향을 바꾼다. 같은 physical reconstruction error를 x0·ε·v 좌표로 변환해 보고, SNR bucket별 numerator·denominator와 gradient norm을 낸다. parameterization 교체는 checkpoint-compatible sampling option이 아니라 새 학습 objective다.

SNR weighting을 time별 gradient budget으로 해석한다. SNR은 variance-preserving path에서 \(\bar α_t/(1-\bar α_t)\)다. 작은 t는 clean signal이 강하고 큰 t는 noise가 강하다. timestep을 균등하게 뽑아도 target parameterization과 loss scale 때문에 실제 gradient contribution은 균등하지 않다. sample count, loss mass와 gradient norm을 time bucket별로 따로 본다.

Min-SNR류 weighting은 높은 SNR 구간의 과도한 영향에 cap을 두어 time task 사이 conflict를 완화하려는 설계다. 정확한 weight는 ε, x0, v prediction에 따라 변환될 수 있다. `min(snr,γ)`만 외워 모든 target에 적용하지 않는다. code에서 target branch, weighting branch와 final reduction 순서를 함께 읽는다.

P2 weighting, EDM preconditioning과 flow weighting도 같은 이름의 “SNR 보정”이 아니다. 어떤 physical error를 어느 time distribution 아래 근사하는지 적는다. weighting을 바꾸면 optimal predictor가 같은지, 단지 estimator variance만 바뀌는지 구분한다. data distribution과 finite model에서는 둘 다 behavior를 바꿀 수 있다.

distributed training에서는 rank마다 timestep histogram이 달라 local mean이 time mixture를 흔든다. global loss numerator와 valid element 수뿐 아니라 target weighting 합을 추적한다. importance sampling으로 time density p(t)를 바꾸면 unbiased estimator를 원할 경우 objective measure와 proposal ratio를 포함한다. 실제 구현이 biased curriculum을 의도했다면 그 사실을 숨기지 않는다.

fixture는 동일 error magnitude를 가진 네 time bucket에서 unweighted, Min-SNR과 chosen flow weighting의 expected contribution을 손계산한다. batch partition과 accumulation order를 바꿔 global gradient가 같은지 본다. metric에는 sampled count, weight sum, weighted loss, unweighted diagnostic와 gradient norm을 함께 낸다.

DDIM을 같은 marginal과 다른 coupling으로 이해한다. DDIM은 DDPM과 같은 training objective를 사용할 수 있지만 reverse transition의 stochasticity를 조절하는 non-Markovian construction을 사용한다. eta가 0이면 주어진 initial noise와 model 아래 deterministic path가 되고, eta가 커지면 추가 noise가 들어간다. “step을 건너뛴 DDPM”이라는 설명만으로는 coupling과 reproducibility 차이를 놓친다.

DDIM step은 predicted x_0와 ε direction을 이용해 다음 ᾱ 위치로 이동하고 variance term을 선택한다. timestep spacing이 바뀌면 같은 nominal step 수에서도 방문하는 ᾱ가 다르다. `set_timesteps`, offset, spacing과 terminal convention이 trajectory state다. inference loop의 integer index와 scheduler timestep을 구분한다.

deterministic inversion은 완전한 image recovery를 자동 보장하지 않는다. model error, VAE loss, CFG와 discretization이 누적된다. forward inversion에서 사용한 prompt·guidance·scheduler·precision을 reverse와 같게 고정하고 latent trajectory의 first divergence를 본다. pixel similarity만 보면 VAE decode error와 field error를 섞는다.

DDPM과 DDIM 비교는 같은 model, prediction type, initial latent, condition과 NFE를 맞춘다. stochastic sampler는 seed별 distribution과 diversity를 보고 deterministic path는 paired reconstruction을 본다. nominal step 수가 같아도 guidance가 conditional/unconditional 두 forward를 쓰는 비용을 NFE와 token·pixel 처리량으로 보고한다.

negative fixture는 eta만 바꾸고 같은 path를 기대하는 test, timestep 배열을 reverse하지 않은 test, x0 clipping 후 ε를 갱신하지 않은 test를 포함한다. 각각 noise injection, 첫 step direction과 boundary에서 검출돼야 한다. final image가 나왔다는 사실은 solver contract 검증이 아니다.

EDM preconditioning을 단위와 dynamic range의 설계로 읽는다. EDM 계열은 data standard deviation σ_data와 noise level σ를 사용해 input, skip, output과 noise embedding coefficient를 구성한다. model이 직접 x0를 내는 것처럼 보여도 실제 denoised estimate는 scaled noisy input의 skip connection과 network output의 결합이다. 이 preconditioning은 time별 dynamic range와 optimization을 바꾼다.

sigma distribution은 training에서 어떤 noise scale을 얼마나 보는지 정하고 sampler의 sigma schedule은 inference evaluation grid를 정한다. 두 distribution을 같은 배열로 가정하지 않는다. log-normal training sampling, Karras-style inference spacing과 terminal zero 처리의 목적을 분리한다. checkpoint에는 preconditioning config와 σ_data가 필요하다.

EDM Euler scheduler는 model input scaling, model output preconditioning과 derivative 계산을 한 step 안에서 연결한다. Diffusers 고정 revision `d57cecde92a6d396845ab35425aa27469dff8173`의 `scheduling_edm_euler.py`에서 `scale_model_input`, `precondition_outputs`, `step`을 같은 call graph로 읽는다. 함수 하나만 복사하면 계수 계약을 놓친다.

churn은 특정 sigma 범위에서 noise를 추가해 stochastic exploration을 늘리는 inference option이다. training noise와 동일한 개념으로 쓰지 않는다. churn strength, min/max sigma, random generator와 per-step seed를 trajectory manifest에 둔다. deterministic canary에서는 꺼 두거나 exact random state를 보존한다.

analytic fixture는 σ=0에 가까운 값, σ=σ_data와 큰 σ에서 c_in, c_skip, c_out을 계산한다. network output 0일 때 denoised estimate가 어떤 skip을 따르는지 본다. dtype cast와 σ broadcasting shape를 확인한다. preconditioning 누락은 첫 model input에서 잡혀야 한다.

rectified flow를 straight line과 straight trajectory로 구분한다. 가장 단순한 flow matching path를 \(x_t=(1-t)x_0+tε\)라 두면 conditional velocity는 \(u_t=ε-x_0\)로 t에 무관하다. 그러나 model이 보는 것은 x_t와 condition이며 marginal conditional expectation을 학습한다. 개별 pair의 직선 target이 학습된 모든 sample trajectory가 완벽한 직선이라는 뜻은 아니다.

continuity equation \(∂_tp_t+∇·(p_tv_t)=0\)은 vector field가 probability mass를 운반하는 조건이다. flow matching은 path가 유도하는 conditional velocity를 회귀해 marginal field를 얻는다. source와 target coupling을 어떻게 정하느냐가 경로와 난이도를 바꾼다. independent noise-data coupling과 reflow coupling을 구분한다.

rectification 또는 reflow는 기존 field가 만든 coupling을 이용해 trajectory를 더 곧게 만들려는 단계가 될 수 있다. teacher-generated pair, solver와 distillation data가 새 lineage다. 단순히 같은 model을 더 학습한 것으로 쓰지 않는다. straightness metric도 Euclidean path length, curvature와 perceptual geometry 중 무엇인지 명시한다.

sampling은 learned ODE `dx/dt=vθ(x,t,c)`를 적분한다. Euler 한 step error는 field curvature와 step size에 좌우된다. time shift는 grid를 바꿔 특정 구간에 evaluation을 집중한다. Diffusers `FlowMatchEulerDiscreteScheduler`의 sigma, shift와 timestep state를 training t와 같은 변수라고 가정하지 않고 변환을 기록한다.

fixture는 2D Gaussian-to-mixture toy field에서 analytic conditional velocity, one-step Euler와 fine RK solution을 비교한다. target sign을 뒤집거나 t direction을 바꾼 negative case는 첫 delta에서 실패해야 한다. 이미지 final metric보다 vector field error와 trajectory curvature를 먼저 본다.

Diffusers scheduler를 config가 있는 state machine으로 읽는다. Diffusers scheduler는 stateless formula collection이 아니다. config, training noise arrays, inference timesteps·sigmas, current step index, model-output history와 random generator를 가질 수 있다. `set_timesteps` 호출 전후 state가 다르고 multistep solver는 이전 output과 lower-order counter를 저장한다. scheduler resume는 current sample과 integer step만 복원해서는 부족할 수 있다.

고정 commit `d57cecde92a6d396845ab35425aa27469dff8173`에서 `DDPMScheduler.add_noise`는 training corruption을, `step`은 reverse update를 맡는다. `EulerDiscreteScheduler.scale_model_input`과 `step`은 sigma preconditioning과 derivative를 나눈다. `HeunDiscreteScheduler.step`은 predictor/corrector phase state를, `DPMSolverMultistepScheduler.step`은 output history를 소비한다.

pipeline이 scheduler를 config에서 다시 만들 때 `from_config`가 모든 실행 상태를 복원하지는 않는다. 새 inference run에는 괜찮지만 중간 trajectory resume에는 history, step_index와 RNG가 필요하다. scheduler class 교체는 config-compatible하더라도 prediction type, input scaling과 accepted timestep representation이 다를 수 있다.

scheduler source audit 표에는 `set_timesteps`, `scale_model_input`, model prediction conversion, variance/noise injection, history mutation, returned state와 generator consumption을 행으로 둔다. option마다 어느 행이 바뀌는지 표시한다. timestep spacing, Karras sigmas, dynamic shift와 final sigma를 “품질 옵션” 하나로 묶지 않는다.

golden trajectory는 initial sample, condition, exact timestep/sigma arrays, per-step model input, output, converted x0/ε/v, derivative, noise draw와 next sample을 저장한다. 새 revision은 final image보다 각 step digest를 먼저 비교한다. expected semantic change가 있으면 최초 다른 행과 이유를 승인표에 적는다.

SD3 training example에서 flow target과 weighting을 추적한다. Diffusers 같은 commit의 `examples/dreambooth/train_dreambooth_sd3.py`와 `examples/controlnet/train_controlnet_sd3.py`는 SD3 계열 flow training을 실제 tensor로 추적할 좌표다. data transform, VAE latent, text encoder, timestep density, noisy latent, transformer prediction, weighting과 loss reduction을 call graph로 잇는다. example이라는 이유로 production contract를 자동 대표한다고 쓰지 않고 실행한 revision과 option만 설명한다.

`compute_density_for_timestep_sampling`은 uniform, logit-normal 등 선택에 따라 u를 뽑고 schedule index로 바꾼다. 이는 단순 dataloader random이 아니라 Monte Carlo proposal이다. `compute_loss_weighting_for_sd3`는 sigma에 따른 weighting scheme을 적용한다. density와 loss weight를 동시에 바꾸면 realized time contribution이 어떻게 변하는지 histogram과 gradient로 본다.

flow noisy latent가 `(1-σ)x0+σε`인지 다른 orientation인지 code 식을 기준으로 확인한다. target이 noise-latent 또는 latent-noise인지 sign을 고정한다. scheduler inference의 dt direction과 training target sign이 맞아야 한다. shape가 같은 sign error는 loss가 감소할 수 있으므로 1D analytic fixture가 필수다.

SD3의 transformer는 latent patch token과 여러 text condition을 결합한다. tokenizer별 max length, pooled embedding, sequence embedding과 mask의 ownership을 기록한다. cached text embedding은 encoder·tokenizer·prompt dropout revision을 key로 한다. classifier-free dropout이 있으면 empty condition embedding 생성 경로도 state다.

DreamBooth prior preservation을 함께 쓰면 instance와 class loss의 분모·weight가 분리된다. batch concatenate 순서와 class image provenance를 추적한다. flow weighting과 prior weight가 곱해지는 위치를 확인한다. total scalar만 log하면 time mixture와 instance/class mixture를 분리할 수 없다.

FLUX fine-tuning을 latent packing과 guidance condition으로 해부한다. FLUX 계열 training example에서는 VAE latent를 transformer가 기대하는 packed image sequence로 바꾸는 경계가 중요하다. latent channel·height·width를 patch-like token과 positional ID로 매핑하고 다시 unpack하는 함수의 shape와 ordering을 기록한다. resolution bucket이 바뀌면 token length와 position ID가 함께 바뀐다.

text condition은 T5 계열 sequence와 CLIP pooled embedding처럼 서로 다른 encoder output을 사용할 수 있다. prompt 하나가 두 tokenizer에서 다른 truncation을 겪는다. encoder를 freeze하고 embedding cache를 쓰더라도 exact prompt bytes, tokenizer, max length, dtype와 encoder digest가 cache key다. null condition과 prompt dropout도 별 entry다.

guidance-distilled model은 guidance scalar를 model condition으로 받을 수 있다. 이것은 sampling에서 conditional/unconditional prediction을 선형 결합하는 CFG와 동일하지 않다. training example이 guidance tensor를 어떻게 만들고 transformer forward에 넣는지 확인한다. serving pipeline이 이를 누락하면 weight는 맞아도 learned function 입력이 다르다.

flow time shift는 image token sequence length나 configured scheme에 따라 달라질 수 있다. training timestep density와 inference dynamic shift를 구분한다. resolution별 sigma histogram, loss와 gradient를 낸다. 같은 batch size라도 고해상도 token이 denominator와 compute를 지배하지 않는지 본다.

LoRA training에서는 target module, rank, alpha, dropout과 adapter dtype을 기록한다. dual text encoders까지 학습하는지 transformer만 학습하는지 parameter inventory로 검증한다. save/load 뒤 adapter scale, merge와 PipelineBundleID가 같은 golden latent prediction을 내야 한다.

DiT block을 adaLN gate와 residual 경로로 검산한다. DiT는 spatial latent를 patchify해 sequence로 만들고 transformer block을 적용한다. token 수는 latent height×width를 patch area로 나눈 값이며 attention 비용은 제곱으로 증가한다. VAE downsample factor와 patch size가 compute·detail bottleneck을 함께 결정한다. pixel resolution만 보고 transformer sequence를 추정하지 않는다.

timestep과 class/text condition은 adaptive layer normalization의 scale·shift·gate를 만들 수 있다. zero-initialized gate는 training 초기에 block residual을 작게 시작시키는 설계다. condition embedding이 norm 전후 어느 지점에 들어가는지, attention과 MLP gate가 별도인지 source에서 확인한다. “condition을 더한다”는 요약으로는 initialization dynamics를 설명하지 못한다.

joint attention 구조에서는 image token과 text token이 같은 attention space에 들어가거나 modality별 projection·norm을 거친다. sequence concatenate 순서, mask와 positional encoding이 state다. text length 변화가 image attention compute와 kernel shape를 바꾼다. context parallel partition에서 modality boundary가 rank 사이에 어떻게 놓이는지 기록한다.

DiT golden block은 B=1, 작은 image/text token과 fixed condition으로 eager FP32 output을 저장한다. adaLN scale·shift·gate, QKV, attention output, MLP와 residual을 단계별로 비교한다. fused attention·FP8·compile path는 이 reference와 forward/backward tolerance를 통과한 뒤 성능을 측정한다.

checkpoint에는 patch embed, positional parameters, condition embedder, transformer, output projection과 EMA를 logical identity로 묶는다. resolution interpolation이나 positional scheme 변경은 단순 load option이 아니라 model function migration이다. unknown key를 조용히 버리고 생성 결과만 보는 방식을 금지한다.

U-Net과 DiT의 inductive bias를 실행 경로로 비교한다. U-Net은 downsample·bottleneck·upsample과 skip connection으로 multi-scale spatial feature를 명시한다. convolution receptive field, resolution별 channel과 attention block이 구조에 박혀 있다. DiT는 patch sequence와 global attention으로 scale interaction을 학습하지만 token cost가 크다. 어느 architecture가 우월하다는 단정 대신 resolution, data와 compute regime를 적는다.

U-Net의 time embedding은 residual block에 scale-shift로 들어가고 text cross-attention이 특정 resolution에 배치될 수 있다. skip tensor의 shape와 order는 checkpoint 밖 실행 상태지만 gradient memory의 큰 부분이다. gradient checkpointing이 어느 block을 재계산하는지와 stochastic op를 확인한다.

DiT의 sequence parallel·FlashAttention과 U-Net의 spatial convolution kernel은 병렬 bottleneck이 다르다. U-Net은 high-resolution activation과 skip communication, DiT는 attention all-to-all·all-gather와 long sequence가 문제다. 같은 parameter 수나 FLOPs만으로 cluster efficiency를 비교하지 않는다.

두 model을 동일 VAE latent와 condition에 학습하더라도 target parameterization, time sampling과 normalization을 맞춰야 architecture ablation이 된다. loss curve 외에 time-bucket error, spatial frequency, condition alignment와 sampler robustness를 본다. solver는 고정하거나 2×2로 분리한다.

implementation review는 model input/output shape, timestep dtype, condition mask, residual scaling과 prediction head initialization을 공통 표에 놓는다. architecture name보다 tensor contract가 fine-tuning과 checkpoint compatibility를 결정한다.

VAE를 단순 전처리가 아닌 생성 좌표계로 다룬다. latent diffusion은 pixel x를 VAE encoder의 distribution으로 보내 latent z를 얻고 scale·shift를 적용한 좌표에서 denoising한다. posterior sample을 쓰는지 mode를 쓰는지, scaling factor와 shift factor가 무엇인지가 training target을 바꾼다. inference decode가 역변환을 정확히 적용해야 한다.

VAE가 lossy하면 denoiser가 완벽한 latent를 생성해도 pixel detail은 복원되지 않는다. reconstruction error, denoiser error와 solver error를 분리한다. paired x→encode→decode baseline을 모든 evaluation에 포함한다. VAE revision을 바꾸고 FID가 좋아졌다면 denoiser 개선으로 귀속하지 않는다.

latent cache는 비용을 줄이지만 random crop·flip, posterior sampling과 VAE revision을 freeze한다. cache key에는 source media digest, transform parameters, encoder digest, scale, dtype와 sampling seed를 넣는다. augmentation을 epoch마다 바꾸려면 precompute가 의미를 바꿀 수 있다. stale cache injection을 load gate에서 막는다.

video·audio VAE는 temporal compression과 causal boundary가 추가된다. chunk overlap과 first-frame handling이 latent sequence를 바꾼다. image VAE의 scaling 설명을 그대로 복사하지 않는다. decode tiling은 seam과 memory를 바꿀 수 있어 full decode reference와 비교한다.

VAE precision은 작은 latent scale과 decoder saturation에 민감할 수 있다. encode distribution parameter, sampled latent, normalized latent와 decode output을 dtype별로 비교한다. serving bundle은 model weight, VAE, processor와 scale config를 원자적으로 publish한다.

text conditioner를 prompt bytes에서 attention tensor까지 추적한다. prompt는 Unicode normalization, tokenizer, special token, padding·truncation을 거쳐 IDs와 mask가 된다. CLIP pooled output, T5 sequence output과 multiple encoder를 함께 쓰는 pipeline에서는 각각 max length와 dtype이 다르다. prompt string hash만 cache key로 쓰지 않는다.

classifier-free guidance training은 일정 확률로 condition을 drop한다. null condition이 empty string encoding인지 learned embedding인지, pooled와 sequence를 모두 drop하는지 확인한다. dropout RNG와 realized rate를 time·language bucket별로 관측한다. checkpoint resume에서 RNG가 달라지면 next update condition이 바뀐다.

sampling CFG는 conditional prediction \(f_c\)와 unconditional prediction \(f_u\)를 `f_u+w(f_c-f_u)`처럼 결합한다. guidance scale은 model field를 바꾸고 큰 값은 off-manifold 방향과 saturation을 만들 수 있다. guidance rescale이나 dynamic guidance는 별 algorithm이다. training의 condition dropout 확률과 inference guidance scale을 같은 knob로 설명하지 않는다.

negative prompt는 unconditional branch를 empty가 아닌 별 text condition으로 바꾼다. cached embedding, tokenizer와 mask가 positive branch와 독립 identity를 갖는다. prompt weighting parser나 textual inversion token이 있으면 rendered token sequence와 embedding injection을 manifest에 둔다.

condition audit는 prompt bytes, encoder별 IDs·mask, selected hidden statistics, pooled vector, null branch와 cross-attention output을 golden artifact로 만든다. model trajectory가 다를 때 initial latent가 같다면 condition edge부터 최초 차이를 찾는다.

ControlNet을 frozen backbone에 붙는 residual control로 이해한다. ControlNet류는 pretrained denoiser의 feature path를 복제하거나 별 control network를 두고 conditioning image에서 만든 residual을 backbone block에 주입한다. zero convolution은 초기 residual을 0에 가깝게 만들어 base behavior를 보존하려는 장치다. control image를 단순 추가 channel로 넣는 모든 방법을 ControlNet이라 부르지 않는다.

training state에는 base denoiser frozen 여부, control network, condition encoder, injection block map과 conditioning scale이 있다. base가 실제로 freeze됐는지 trainable parameter inventory와 gradient canary로 확인한다. optimizer에 frozen parameter가 들어가 memory를 쓰거나 EMA가 잘못 포함하지 않는지 본다.

conditioning data는 source image와 edge/depth/pose map의 exact transform alignment가 핵심이다. crop·resize·flip을 image와 control에 동일하게 적용하고 interpolation mode를 기록한다. preprocessor revision과 threshold가 새 DatasetRevision이다. misalignment는 loss가 내려가도 control fidelity를 약화한다.

SD3·FLUX용 control example은 U-Net ControlNet과 model input·residual injection contract가 다를 수 있다. 고정 Diffusers revision의 `examples/controlnet/train_controlnet_sd3.py`와 `examples/flux-control`을 각 model family의 실제 path로 읽는다. class 이름 유사성으로 tensor contract를 추정하지 않는다.

평가는 base quality, prompt alignment와 control adherence를 삼각형으로 본다. control scale sweep, missing·corrupt condition과 contradictory prompt를 시험한다. safety에서는 pose·depth나 identity-derived control의 privacy와 misuse를 data lineage에 포함한다.

DreamBooth와 LoRA를 identity binding의 두 state 층으로 나눈다. DreamBooth는 소수 instance image와 unique identifier를 text-to-image prior에 결합한다. instance reconstruction만 밀면 language drift와 overfitting이 생길 수 있어 class prior preservation을 사용할 수 있다. instance prompt, class prompt, class image generator revision과 prior loss weight가 objective state다.

class image를 현재 model로 생성하면 seed, scheduler, guidance와 model revision을 보존한다. 부족한 class image를 resume 때 새로 만들면 dataset이 바뀐다. generated 산출물 digest와 selection filter를 manifest에 둔다. instance/class batch concatenate 뒤 각 loss의 numerator와 denominator를 분리한다.

LoRA는 weight update를 low-rank A·B로 parameterize하지만 어느 module에 붙이는지가 function을 정한다. attention Q/K/V/out, MLP, text encoder와 convolution target을 구분한다. rank, alpha, dropout, initialization과 trainable dtype을 기록한다. adapter parameter 수만으로 expressivity를 비교하지 않는다.

DreamBooth full fine-tuning과 DreamBooth LoRA는 같은 data objective를 쓸 수 있지만 optimizer state, checkpoint 크기와 base dependency가 다르다. LoRA export는 base ModelID가 없으면 불완전하다. merge는 새 artifact이고 merge 전후 golden prediction을 비교한다. 여러 adapter scale의 composition은 training checkpoint와 다른 serving function이다.

overfitting 평가는 training image reconstruction만 보지 않는다. identifier fidelity, pose·background 다양성, class prior retention, prompt compositionality와 unrelated prompt regression을 본다. 얼굴·개인 identity data는 consent, deletion lineage와 memorization red-team을 포함한다.

DDPO를 diffusion sampler와 policy gradient 사이에서 분리한다. DDPO류는 denoising trajectory를 stochastic policy로 보고 final image reward로 update할 수 있다. state는 prompt, initial noise, 각 timestep sample, transition log probability, reward model과 baseline을 포함한다. supervised diffusion loss나 direct preference objective와 동일하지 않다.

sampler transition이 deterministic DDIM eta=0이면 naive action log probability가 퇴화할 수 있다. algorithm이 어떤 stochastic transition과 likelihood를 사용하는지 확인한다. scheduler를 임의로 교체하면 policy gradient estimator 의미가 달라진다. model prediction과 transition mean·variance, sampled noise를 RolloutID에 기록한다.

reward는 image aesthetic, alignment, compressibility 또는 task verifier일 수 있다. reward revision, preprocessing과 normalization을 고정한다. proxy exploitation, prompt-specific baseline과 clipping을 관측한다. 여러 reward를 합치면 component별 scalar와 coefficient를 남긴다.

distributed rollout과 training이 분리되면 behavior policy version과 learner policy version, old logp와 trajectory scheduler가 필요하다. stale rollout을 얼마나 허용하는지 20장의 policy-version fence와 연결한다. image만 저장하고 latent transition을 재구성하려 하면 exact likelihood를 잃을 수 있다.

evaluation은 reward 상승 외에 independent human/judge, diversity, prompt fidelity, artifact와 safety를 본다. reward model을 학습·selection·평가에 모두 쓰지 않는다. DDPO checkpoint에는 denoiser, optimizer, baseline, reward/reference IDs, sampler config와 rollout cursor가 함께 있어야 한다.

discrete text diffusion을 categorical corruption으로 유도한다. continuous Gaussian noise 대신 token state에 transition matrix Q_t를 적용할 수 있다. uniform replacement는 다른 vocabulary token으로 섞고 absorbing-mask process는 token을 특별 mask state로 보낸다. forward marginal은 matrix product로 계산하며 posterior는 x_t와 x_0에 조건부인 categorical distribution이다.

mask diffusion에서 t가 커질수록 mask 비율이 높아지고 model은 masked positions의 clean token distribution을 예측할 수 있다. loss가 모든 token인지 corrupted token만인지, time weight와 normalization이 무엇인지 확인한다. unchanged token을 포함하면 쉬운 copy signal이 objective를 지배할 수 있다.

reverse sampling은 한 번에 여러 token을 제안하고 confidence에 따라 일부를 commit할 수 있다. 병렬성은 모든 token이 독립이라는 뜻이 아니다. 잘못 확정된 token이 context가 되어 뒤 prediction을 왜곡한다. remasking 허용, acceptance schedule와 terminal all-unmasked invariant가 sampler state다.

text likelihood와 generation 품질은 별개로 평가한다. variational bound, token recovery accuracy, perplexity-like metric의 정의를 명시하고 AR perplexity와 직접 비교하지 않는다. generation은 exact length, EOS, repetition, reasoning와 compute/NFE를 본다. confidence calibration과 token commit age를 trace한다.

tiny vocabulary fixture는 `{A,B,[MASK]}` transition matrix를 만들고 two-step marginal과 posterior를 손계산한다. scheduler output, corrupted-token loss mask와 reverse sample probability를 비교한다. row stochasticity, terminal mask와 zero probability를 assert한다.

DiffusionGemma의 바깥 sequence와 안쪽 denoising 경계를 추적한다. DiffusionGemma류 hybrid를 읽을 때 전체를 순수 diffusion language model이라고 뭉개지 않는다. outer autoregressive context가 block 또는 segment를 선택하고 inner denoising이 block token을 병렬 refinement할 수 있다. 어느 token이 fixed context, noisy state와 newly committed output인지 mask로 표현한다.

training example은 outer position, block boundary, corruption time, clean token, noise/mask state와 loss mask를 가진다. inner step에서 bidirectional attention이 허용되는 범위와 future outer block 차단을 확인한다. attention mask 하나가 causal과 denoising relation을 동시에 소유할 수 있어 tiny matrix visualization이 유용하다.

sampling state는 outer cursor, inner timestep, current block token, confidence·commit mask와 RNG다. checkpoint나 streaming resume에서 outer tokens만 보존하면 inner trajectory가 달라진다. token latency는 first token, block commit과 final sequence를 구분한다. nominal denoising steps와 total model forward 수를 함께 낸다.

평가는 AR baseline과 같은 token budget, context와 generation length를 맞춘다. parallel block이 wall-clock을 줄여도 quality, correction ability와 memory를 본다. reasoning benchmark에서는 intermediate token을 너무 일찍 commit하는 failure와 length termination을 분석한다. likelihood 정의가 다르면 raw perplexity를 같은 축에 두지 않는다.

source가 공개한 inference code만 있고 training objective 세부가 없으면 그 경계를 명시한다. model card 서술에서 target·weight를 추정해 확정하지 않는다. 공개 function의 tensor contract, config와 test만 evidence로 사용하고 미공개 training path는 `NotVerified`로 둔다.

Qwen2.5-Omni Token2Wav를 mel ODE와 vocoder로 분해한다. Transformers commit `36deb0b53ed0863f4b4dfdea23dcaec7f3df3701`의 `modeling_qwen2_5_omni.py`에서 Token2Wav 경로는 speech code와 speaker/reference condition을 받아 noised mel state의 velocity를 DiT로 예측하고 RK4로 적분한다. 이후 BigVGAN류 vocoder가 mel을 waveform으로 변환한다. ODE field와 waveform decoder를 다른 component로 평가한다.

condition에는 text/speech token sequence, speaker embedding, reference mel과 attention mask가 포함될 수 있다. 각 length와 alignment, chunk boundary를 기록한다. streaming generation에서는 이전 mel context와 overlap이 state가 된다. text token cursor만으로 audio resume를 재현할 수 없다.

RK4 한 macro step은 네 번 field를 평가한다. nominal step 수보다 NFE가 네 배다. 각 stage의 time, temporary mel과 k1…k4를 golden trace로 저장한다. Euler와 비교할 때 NFE를 맞추고 field evaluation batching을 포함한 latency를 본다. dtype와 accumulation error가 waveform phase에 어떻게 증폭되는지도 본다.

mel trajectory error와 vocoder error를 분리하기 위해 ground-truth mel decode, predicted mel with fixed vocoder와 waveform metric을 단계별로 평가한다. vocoder revision을 바꾸고 audio 품질이 좋아졌다면 Token2Wav field 개선으로 쓰지 않는다. loudness normalization과 sample rate도 artifact다.

safety에는 voice identity consent, impersonation, watermark/provenance와 prompt leakage가 들어간다. speaker condition을 제거·교환한 counterfactual과 unauthorized reference를 시험한다. serving bundle은 DiT, condition processors, integrator config, vocoder와 audio postprocess를 원자적으로 고정한다.

PI0의 action flow를 로봇 제어 state로 해석한다. PI0 계열에서 flow matching의 state는 image latent가 아니라 horizon 길이의 continuous action chunk일 수 있다. source distribution은 noise action, target은 demonstration action이며 observation·language·proprioception이 condition이다. action normalization과 robot embodiment별 dimension이 좌표계를 정의한다.

training은 time t에서 noisy/interpolated action과 target velocity를 만든다. observation encoder와 action expert가 어느 token에서 상호작용하는지, horizon mask와 padding이 loss에 들어가는지 확인한다. invalid action dimension을 0으로 채우고 denominator에 포함하면 embodiment mixture가 왜곡된다.

inference ODE가 action chunk를 생성한 뒤 environment에는 일부 prefix만 실행하고 다음 observation에서 재계획할 수 있다. sampler trajectory와 physical control trajectory를 구분한다. integration step, action clipping, smoothing과 control frequency가 safety-critical state다. 같은 action latent라도 postprocess가 다르면 robot effect가 달라진다.

evaluation은 offline action MSE만 보지 않는다. closed-loop success, intervention, constraint violation, latency와 distribution shift를 본다. noise seed별 action 다양성이 exploration인지 unsafe jitter인지 구분한다. simulation과 real robot 결과의 domain gap을 명시한다.

checkpoint에는 model·EMA, optimizer, observation processors, action normalization statistics, embodiment schema, flow time convention과 data cursor를 넣는다. 21장의 multimodal conditioner와 25장의 red-team, 29장의 failure injection을 연결한다. emergency stop과 action bound는 learned model 밖 독립 safety layer로 유지한다.

ESMFold2류 구조 생성 주장을 state 공간부터 검증한다. 단백질 구조 model에 diffusion 또는 flow가 쓰인다는 주장에서는 무엇을 noise화하는지 먼저 묻는다. amino-acid sequence, residue frame, 3D coordinate, distance/orientation 또는 latent 중 어느 state인지가 핵심이다. 회전·병진 대칭을 어떻게 처리하는지와 물리 constraint를 확인한다. 이름만으로 image diffusion 수식을 복사하지 않는다.

3D coordinate flow는 global translation과 rotation에 equivariant해야 의미 없는 pose를 학습하지 않는다. residue frame을 쓰면 SO(3) rotation의 interpolation과 tangent vector가 Euclidean coordinate와 다르다. torsion angle은 periodic state다. source가 공개하지 않은 architecture를 추측해 확정하지 않고 paper equation과 code symbol이 연결되는 부분만 기술한다.

condition에는 protein sequence embedding, template, MSA와 pair representation이 있을 수 있다. 각 data 소스 리비전과 mask, chain boundary를 기록한다. crop과 chain permutation이 target geometry를 바꾸지 않는지 본다. length bucket별 compute와 loss denominator를 분리한다.

평가는 RMSD 하나가 아니라 alignment 방식, lDDT, clash, bond geometry, confidence calibration과 sequence family split을 본다. train homolog leakage를 막고 orphan·multimer·long protein slice를 나눈다. sampler step 감소가 physical validity를 해치지 않는지 trajectory 중간 constraint를 관측한다.

공개 명칭이나 차기 version의 세부가 검증되지 않았다면 `ESMFold2` 구현을 존재한다고 단정하지 않는다. 확인 가능한 repository, model card와 commit이 생길 때 SourceCard를 갱신한다. 이 경계 표시는 내용 부족이 아니라 과장을 막는 기술적 정확성이다.

이제 살펴볼 것은 CUDA kernel과 diffusion trajectory의 수치 계약이다. diffusion training의 큰 kernel은 attention, convolution, GEMM, normalization, VAE와 optimizer다. solver step 자체는 elementwise가 많지만 수십 번 반복되어 launch와 memory bandwidth가 중요하다. kernel fusion이 model prediction conversion이나 scheduler state까지 섞으면 reference 경계가 흐려진다. 먼저 eager tensor contract를 고정한다.

FlashAttention은 QK score를 materialize하지 않아 memory를 줄이지만 mask, dropout RNG와 accumulation order가 달라질 수 있다. DiT의 variable image/text sequence, context parallel과 padding mask를 representative shape로 시험한다. unsupported shape가 eager fallback하는 비율을 metric으로 낸다. 평균 latency만 보면 tail batch의 fallback을 놓친다.

FP8은 scale·amax history와 cast boundary가 추가된다. time bucket에 따라 activation magnitude가 달라 하나의 scale policy가 특정 sigma에서 포화할 수 있다. sigma별 saturation, amax, model prediction error와 final trajectory error를 본다. scale state는 checkpoint resume에 포함한다.

CUDA graph는 static shape·address·control flow를 요구한다. variable resolution, CFG batch doubling과 multistep phase가 graph variant를 늘린다. graph capture를 위해 random noise와 timestep을 잘못 고정하지 않는다. RNG input buffer와 scheduler history ownership을 명시한다. cache miss와 recapture latency를 관측한다.

kernel upgrade gate는 selected time·resolution·condition의 forward, backward, optimizer update와 K-step trajectory를 eager FP32 기준과 비교한다. tolerance를 final FID로 정하지 않는다. NaN, saturation과 first divergence를 kernel·dtype·shape에 연결한다. correctness를 통과한 variant만 throughput 표에 올린다.

분산 DiT에서 sequence·condition·parameter ownership을 고정한다. data parallel은 sample과 timestep을 나누고 gradient를 reduce한다. tensor parallel은 attention·MLP projection을 shard하며 global vocabulary가 없는 image model에서도 head/channel axis를 나눈다. context parallel은 image/text token sequence를 나누고 attention 통신을 만든다. pipeline parallel은 block stage와 activation을 나눈다.

condition encoder를 각 DP replica에 복제할지 별 service·stage에 둘지 선택한다. cached embedding을 쓰면 compute는 줄지만 storage와 revision consistency가 생긴다. text token sequence가 context-parallel shard에 어떻게 배치되고 image token과 joint attention하는지 logical index를 기록한다. rank-local order를 identity로 쓰지 않는다.

timestep sampling은 DP rank별 독립 RNG를 쓰되 global distribution을 관측한다. loss의 pixel/token count, time weight와 sample denominator를 전역에서 정확히 합친다. variable resolution batch는 rank compute가 크게 달라 straggler를 만든다. token-budget batching과 gradient scaling의 분모를 일치시킨다.

sequence parallel collective와 gradient reduce가 겹칠 때 NCCL stream·process group을 분리할 수 있다. overlap이 수치 의미를 바꾸지 않게 dependency event를 검증한다. collective timeout은 어느 rank·shape·time bucket에서 시작했는지 trace한다. deadlock failure fixture는 empty condition, tail microbatch와 checkpoint 동시 실행을 포함한다.

parallel layout 변경 resume는 raw model뿐 아니라 EMA, optimizer, FP8 scale과 sampler cursor를 reshard한다. first batch SampleID, latent/noise/time, condition tensor, loss numerator와 raw·EMA update를 비교한다. 15·16·17장의 ownership·scheduling·durable generation 계약을 그대로 적용한다.

diffusion checkpoint를 trajectory 재개와 학습 재개로 나눈다. 학습 checkpoint는 denoiser, optimizer, scheduler, scaler, raw·EMA parameter, RNG, data transform, timestep sampler, VAE·condition dependency와 accumulation state를 저장한다. inference trajectory checkpoint는 current sample, scheduler timesteps/sigmas, step index, multistep history, generator state, condition와 component bundle을 저장한다. 두 schema를 같은 파일 이름으로 혼동하지 않는다.

training resume는 next update equality를 검증한다. 같은 SampleID와 crop, VAE latent, ε, t, condition dropout, target, weight, loss denominator, gradient와 raw·EMA delta를 비교한다. VAE latent cache와 text cache manifest가 dependency closure에 포함된다. cache miss로 live recompute할 때 stochastic posterior가 같은지 확인한다.

trajectory resume는 current latent만 같아서는 부족하다. Heun predictor/corrector phase, DPM solver history, stochastic noise generator와 current sigma index를 복원한다. 새 scheduler object에 current timestep만 주입한 negative fixture가 first next state에서 실패해야 한다. solver class·config digest도 필수다.

distributed checkpoint는 model/EMA/optimizer shard를 global identity로 묶고 partial generation을 commit하지 않는다. object store의 manifest, checksum과 conditional commit은 17장 protocol을 따른다. VAE나 text encoder URI가 사라지면 weight checkpoint가 있어도 완전한 pipeline recovery가 아니다.

복구 후에는 새 durable generation과 golden sample을 만든다. same topology bitwise, topology 변경 numerical과 statistical recovery grade를 구분한다. final media만 비슷하다는 이유로 sample/noise cursor 누락을 허용하지 않는다.

평가를 field·solver·decoder·condition의 요인 실험으로 만든다. FID와 CLIP score 하나는 component 귀속을 하지 못한다. field checkpoint, scheduler, NFE, VAE, condition encoder, guidance와 precision을 factor로 기록한다. 하나를 바꿀 때 나머지를 고정하고 paired prompt·seed로 비교한다. 두 요인이 함께 바뀌면 최소 2×2 조합으로 interaction을 본다.

field 평가는 held-out noisy state에서 ε/x0/v/velocity error를 time bucket별로 본다. solver 평가는 analytic toy와 same field의 trajectory truncation error를 본다. decoder 평가는 ground-truth latent reconstruction을 본다. condition 평가는 prompt swap·drop과 alignment를 본다. final quality는 이 네 층의 합성 결과다.

image는 distribution, diversity, prompt adherence, anatomy·text rendering과 safety slice를 본다. audio는 mel·waveform, intelligibility, speaker similarity와 prosody, action은 closed-loop success와 constraint, protein은 geometry·confidence를 쓴다. modality별 metric을 같은 “quality” 숫자로 합치지 않는다.

paired sampling은 variance를 줄이지만 stochastic sampler의 diversity를 숨길 수 있다. fixed seed pair와 seed distribution을 함께 사용한다. prompt cluster 단위 confidence interval과 multiple-comparison selection을 보고한다. metric model revision과 preprocessing도 artifact다.

evaluation cache는 PipelineBundleID, prompt/media condition, seed, scheduler config, NFE, guidance, VAE·decoder와 metric revision을 key로 한다. mixed bundle output을 같은 run에 넣지 않는다. failure sample에서 trajectory trace로 내려갈 수 있게 SampleID를 보존한다.

생성 안전을 data에서 decoder output까지 추적한다. training data에는 consent, license, personal identity, explicit content와 protected material provenance가 필요하다. caption·image pair의 filter가 어느 revision에서 어떤 reason으로 제외했는지 남긴다. latent cache를 삭제해도 원 media와 descendant checkpoint lineage를 추적할 수 있어야 한다. 삭제 가능성을 과장하지 않고 재학습 범위를 명시한다.

condition safety는 prompt filter만이 아니다. image reference, ControlNet pose·depth, speaker reference와 robot observation이 공격 surface다. processor가 hidden payload나 malformed shape를 어떻게 처리하는지 fuzz한다. text encoder와 denoiser 사이 embedding attack도 counterfactual로 본다.

generation red-team은 memorization, identity imitation, unsafe composition, watermark removal, typography abuse와 model-specific shortcut을 다룬다. fixed prompt list만 반복하면 tuning leakage가 생기므로 dynamic generator와 sealed set을 둔다. refusal·block rate와 benign overblocking을 함께 측정한다.

decoder와 postprocess도 안전 boundary다. VAE·vocoder가 watermark/provenance를 삽입하거나 제거할 수 있고 resizing·compression이 detector 성능을 바꾼다. raw latent, decoded media와 published output의 ArtifactID를 구분한다. provenance metadata가 파일 변환에서 유지되는지 시험한다.

PI0 같은 action generation은 content filter가 아니라 physical constraint, collision, speed·force bound와 emergency stop이 필요하다. learned reward나 flow field 밖의 hard safety layer를 유지한다. safety event를 trajectory time과 action chunk coordinate에 연결해 최초 unsafe decision을 찾는다.

관측성을 time·sigma와 component revision에 맞춘다. training metric은 sampled time/sigma histogram, SNR, target norm, prediction norm, weighted·unweighted loss, weight sum과 gradient norm을 bucket별로 낸다. global 평균 하나는 high-noise collapse와 low-noise detail failure를 상쇄한다. parameterization과 weighting config digest를 dashboard에 표시한다.

data plane은 media resolution·duration, transform, VAE latent mean/std, cache hit, condition length와 dropout을 본다. model plane은 attention entropy 표본, adaLN gate, activation·gradient, FP8 scale와 saturation을 본다. solver plane은 sigma grid, model input/output norm, x0 estimate, derivative, step delta와 history 상태를 본다. decode plane은 saturation, clipping과 postprocess를 본다.

performance metric은 data/VAE, condition encoder, denoiser forward/backward, collective, optimizer, EMA, checkpoint와 sampling solver를 분리한다. resolution·token length·time bucket별 latency와 memory를 본다. compile graph count, fallback, CUDA kernel과 NCCL tail을 같은 trace에 연결한다.

NaN alert는 최초 nonfinite tensor를 target, model input, attention, prediction, conversion, loss와 optimizer 순서로 찾는다. trajectory explosion은 sigma, derivative norm과 dt를 본다. 품질 drift는 canary bundle에서 condition, initial state, selected step digest와 decode summary를 비교한다.

high-cardinality prompt·sample은 secure trace로 보내고 metric label에는 cohort와 bundle revision을 쓴다. incident evidence에는 PipelineBundleID, SampleID, source commit, topology, dtype, scheduler state와 first divergence가 있어야 한다. “FID가 떨어졌다”는 alert만으로는 행동할 수 없다.

failure fixture를 수식 경계마다 배치한다. data fixture는 wrong crop alignment, stale VAE cache, stale text embedding, missing scale factor와 corrupted media를 넣는다. forward fixture는 timestep broadcast 오류, noise seed collision, target sign flip, x0/ε/v mismatch와 loss denominator drift를 넣는다. 각 오류가 target 생성 직후 검출되어야 한다.

model fixture는 padding mask leakage, condition branch swap, adaLN gate omission, LoRA target 누락과 frozen base gradient를 넣는다. fused attention이 특정 resolution에서 다른 output을 내는 case와 FP8 saturation을 포함한다. final image까지 기다리지 않고 selected activation에서 first difference를 찾는다.

solver fixture는 integer index/timestep 혼동, sigma 배열 reverse 오류, input scaling 누락, Heun phase loss, multistep history 삭제, stochastic generator reset과 prediction_type mismatch를 넣는다. analytic scalar 또는 tiny tensor one-step expected value로 막는다. scheduler가 예외를 내는 것과 조용히 다른 trajectory를 만드는 것을 구분한다.

distributed fixture는 uneven resolution, zero-valid condition, rank kill, collective timeout, partial EMA shard, accumulation 중 checkpoint와 world-size restore를 다룬다. expected result는 동일 next update 또는 명시적 rejection이다. rank 일부만 다음 UpdateID로 넘어가면 실패다.

serving fixture는 mixed VAE/model/scheduler revision, stale compiled graph, guidance input 누락, cache collision과 decoder hot-swap을 넣는다. bundle admission이 traffic 전에 막아야 한다. 복구 뒤 old bundle rollback과 golden canary가 성공해야 incident가 닫힌다.

이 흐름에서 독립 검토자가 실행하는 diffusion golden suite도 빠뜨릴 수 없다. suite의 첫 묶음은 analytic path다. scalar DDPM add_noise/posterior, x0·ε·v 회전, SNR weight, rectified flow target, Euler·Heun·RK4와 categorical transition을 FP64로 계산한다. library function과 element별로 비교한다. schedule boundary와 invalid input negative case를 포함한다.

둘째는 component path다. fixed media와 prompt에서 transform, VAE latent, condition embeddings, noisy state, target, model prediction과 loss를 저장한다. U-Net/DiT, ControlNet, LoRA와 selected model family의 trainable parameter inventory를 확인한다. eager FP32가 reference다.

셋째는 distributed update다. variable resolution·time bucket을 uneven rank에 배치하고 global numerator·denominator, gradient, optimizer와 EMA를 비교한다. accumulation과 precision path, checkpoint cold resume를 실행한다. source topology와 다른 target에서도 지원 grade를 검증한다.

넷째는 solver trajectory다. exact initial state와 condition으로 모든 step의 model input, prediction, conversion, derivative, RNG와 next state를 기록한다. scheduler revision·NFE·guidance를 고정한다. multistep 중간 resume와 wrong-history negative fixture를 실행한다.

다섯째는 paired media evaluation과 안전이다. field·solver·VAE 요인 조합, condition counterfactual, memorization·identity·unsafe prompt와 modality-specific metric을 실행한다. 결과 certificate에는 source commit, PipelineBundleID, config, topology, dtype, tested range와 NotExecuted를 남긴다.

장간 연결을 state와 failure owner로 명시한다. 7·8·9장에서는 embedding, attention, MLP와 residual이 DiT·condition encoder 안에서 어떻게 계산되는지 다룬다. 14장은 mixed precision과 CUDA kernel의 오차 경계를, 15장은 DP·TP·PP·CP ownership을 설명한다. 22장은 이를 time-conditioned vector field와 trajectory error에 연결한다.

16장은 cluster scheduling, straggler와 failure domain을 맡고 17장은 model·EMA·optimizer·RNG·data cursor의 durable checkpoint protocol을 맡는다. diffusion-specific state는 VAE/text cache, timestep sampler와 solver history다. 공통 복구 계약 위에 이 추가 state를 얹는다.

18장은 LoRA·DreamBooth에서 base와 adapter provenance, data lineage를 잇는다. 19·20장은 preference와 online reward를 DDPO·reward fine-tuning에 연결하지만 objective와 rollout unit이 image trajectory인지 text token인지 구분한다. 이름이 RL이라고 같은 logprob·denominator를 쓰지 않는다.

21장에서는 multimodal processor, vision/audio token과 conditioner를 다룬다. Token2Wav, video diffusion와 action flow는 modality encoder·decoder 경계를 공유한다. 23장은 editing·unlearning이 어느 time·condition trajectory를 바꾸는지, 24·25장은 평가·red-team의 독립성을 맡는다.

26장은 time bucket·trajectory 관측성과 incident runbook을, 27장은 source·model·dataset 공급망을, 29장은 multi-node failure injection을 다룬다. crosslink는 관련 장 번호만 나열하는 것이 아니라 넘기는 ArtifactID, tensor/state, invariant와 failure owner를 적어야 작동한다.

최종 인수표를 수학·코드·trajectory로 닫는다. 수학 인수는 state 공간, forward path, time convention, prediction target, weighting, vector/score field와 reverse solver를 exact 식으로 적는다. DDPM·DDIM·EDM·rectified flow·categorical diffusion을 공통 단어로 뭉개지 않는다. parameterization 변환과 boundary fixture가 FP64로 맞아야 한다.

코드 인수는 Diffusers commit `d57cecde92a6d396845ab35425aa27469dff8173`의 selected scheduler와 training example symbol, Transformers Qwen2.5-Omni commit, local patch와 effective config를 포함한다. 공개 source가 없는 PI0·DiffusionGemma·ESMFold2 세부는 검증 범위를 명시하고 추측을 구현 사실로 쓰지 않는다.

component 인수는 denoiser, VAE, text/media conditioner, ControlNet·adapter, scheduler와 decoder를 독립 ArtifactID와 bundle로 묶는다. trainable parameter, cache, EMA와 checkpoint closure를 확인한다. mixed revision replica가 admission을 통과하지 못해야 한다.

실행 인수는 data→latent→condition→time/noise→target→prediction→loss→update와 initial state→solver steps→decode를 재생한다. 분산·precision·CUDA path가 golden reference와 허용 범위에서 맞고 resume 뒤 next update와 next solver state가 같아야 한다. NFE와 end-to-end 비용을 함께 낸다.

품질·운영 인수는 field·solver·decoder 요인 평가, modality별 metric, 안전·provenance, time-bucket observability, failure fixture, rollback과 독립 certificate를 포함한다. 어떤 chapter page를 열어도 method 이름이 아니라 왜 그 state가 필요하고 어느 함수·tensor에서 검증하는지 답할 수 있어야 한다.

SD3 time sampler option을 proposal distribution으로 검산한다. Diffusers 고정 revision의 `src/diffusers/training_utils.py`에 있는 `compute_density_for_timestep_sampling`과 `compute_loss_weighting_for_sd3`는 여러 SD3·FLUX 계열 example이 공유한다. 함수 이름은 편의 utility지만 반환한 u, sigma와 weight가 Monte Carlo estimator의 표본과 질량을 정한다. 호출자의 argument, scheduler sigma lookup과 target 식까지 이어 읽는다.

uniform은 u를 균등하게 뽑지만 logit-normal은 normal sample을 sigmoid로 보내 중앙 또는 설정된 구간에 질량을 모을 수 있다. mode, cosine-map류 선택도 같은 분포가 아니다. option의 평균·표준편차나 scale parameter가 실제 u histogram을 어떻게 바꾸는지 백만 sample 이론보다 먼저 고정 seed 표본과 quantile로 확인한다. boundary index clipping과 integer conversion을 본다.

weighting scheme은 sigma별 loss 중요도를 바꾼다. proposal density와 weighting을 함께 고려하지 않고 하나만 “SNR 균형”이라고 설명하지 않는다. target parameterization, latent/noise interpolation orientation과 reduction을 식에 넣는다. training objective measure를 바꾸려는 것인지 estimator variance를 줄이려는 것인지 source·논문 근거가 없으면 구분해 미확정으로 둔다.

distributed sampler는 rank별 generator가 같은 u를 반복하지 않게 global seed, DP rank와 UpdateID로 stream을 만든다. 동시에 전체 histogram이 target distribution을 따르는지 관측한다. world-size resume에서는 exact next u 또는 statistical resume grade를 명시한다. timestep cursor가 없는 checkpoint를 exact라고 부르지 않는다.

golden test는 각 weighting scheme에 sigma `[0, 0.1, 0.5, 0.9, 1]`을 넣고 FP64 expected를 저장한다. invalid scheme, extreme parameter와 zero/one boundary를 시험한다. 새 Diffusers revision에서 utility body나 default가 바뀌면 training example의 golden target과 gradient를 함께 stale 처리한다.

solver history와 RNG를 serializable transition state로 만든다. Euler는 현재 x, t와 field만 있으면 다음 state를 계산할 수 있지만 Heun predictor/corrector는 중간 derivative와 phase를 가진다. multistep DPM solver는 이전 model outputs, lower-order count와 timestep history가 필요하다. stochastic ancestral sampler는 generator state와 step별 noise도 필요하다. scheduler class에 따라 minimal state가 다르다.

trajectory checkpoint schema는 PipelineBundleID, sample tensor, inference timestep/sigma arrays, step index, prediction type, scale-input convention, history tensor, phase, RNG algorithm/state와 condition digest를 가진다. config JSON은 runtime arrays와 history를 대체하지 못한다. array를 재생성할 때 floating-point와 spacing revision이 같은지도 확인한다.

resume fixture는 K-step trajectory를 uninterrupted control로 만들고 각 step 직후 저장한다. 새 process에서 scheduler를 구성해 serialized 실행 상태를 주입하고 다음 model input과 next sample을 비교한다. history 하나 삭제, phase 반전, generator reset과 timestep nearest-match를 negative case로 둔다. failure가 final image가 아니라 첫 next state에서 나타나야 한다.

batch 중 일부 sample만 완료된 service에서는 scheduler state를 sample별로 소유할지 batch cohort로 소유할지 정한다. dynamic batching이 서로 다른 step index·history를 한 object에 섞지 않게 한다. request cancellation 뒤 buffer와 RNG를 다른 request가 재사용하지 않는다. SampleID와 trajectory attempt가 key다.

긴 video·audio trajectory는 state 저장 비용이 크다. 모든 step을 checkpoint하지 않고 preemption boundary를 정할 수 있지만 RPO는 model step 수와 NFE로 표현한다. deterministic replay가 가능하면 initial state와 exact condition·bundle에서 재계산할 수 있고, stochastic path라면 RNG와 external component를 고정한다.

discrete diffusion의 length·EOS·commit invariant를 분리한다. continuous image state는 shape가 고정되는 경우가 많지만 text diffusion은 sequence length와 EOS가 생성 과정의 일부다. fixed-length mask canvas에서 EOS 뒤 token을 어떻게 처리하는지, variable-length model이 length distribution을 별도로 예측하는지 확인한다. 마지막에 mask가 모두 사라졌다고 valid sentence가 완성된 것은 아니다.

commit scheduler가 confidence 높은 token을 먼저 확정할 때 EOS를 너무 일찍 고정하면 뒤 위치가 모두 잘릴 수 있다. EOS confidence에 별 threshold나 remasking을 적용하는지 source에서 본다. token confidence, entropy, commit age와 EOS position을 step별로 trace한다. exact terminal invariant는 no-mask, legal special-token pattern과 length bound를 포함한다.

block refinement에서는 block 사이 causal boundary와 block 안 bidirectional context가 공존할 수 있다. outer block을 commit한 뒤 inner token을 수정할 수 있는지 명시한다. cache가 committed token hidden을 저장한다면 remask가 cache invalidation을 요구한다. AR KV cache 규칙을 그대로 적용하지 않는다.

evaluation denominator는 prompt, generated sequence, non-special token과 model forward 중 무엇인지 분리한다. throughput은 tokens committed per second, NFE, padded canvas computation과 first-block latency를 함께 본다. AR tokens/s와 nominal output token만으로 비교하면 masked 재평가 비용을 숨긴다.

failure fixture는 terminal mask 잔존, duplicate EOS, early EOS, confidence NaN, no-progress step과 all-token remask cycle을 넣는다. scheduler는 maximum step과 progress invariant를 가져야 한다. 무한 refinement나 silent mask-to-pad 변환을 성공으로 처리하지 않는다.

modality별 loss mask와 denominator를 코드 state로 둔다. image latent MSE는 `[B,C,H,W]` 모든 element 평균처럼 보이지만 variable resolution padding, inpainting mask와 control region weight가 있으면 valid element가 다르다. sample별 mean을 다시 평균할지 global valid element mean을 쓸지 objective가 달라진다. resolution curriculum에서 큰 image가 gradient를 더 많이 내는지 명시한다.

video는 frame, spatial pixel과 latent channel 외에 valid duration mask가 있다. padded frame을 loss에 넣지 않고 temporal boundary·first frame weight를 확인한다. audio mel은 time frame과 frequency bin, silence mask와 duration이 있다. 긴 utterance가 objective를 지배하는지 utterance mean과 element mean을 구분한다.

action flow는 horizon와 action dimension mask가 있다. embodiment마다 dimension이 다르면 padded action을 제외하고 normalization statistic도 dimension별 lineage를 가진다. protein은 residue·pair·atom과 chain mask가 있다. modality 이름만 바꿔 image MSE reducer를 재사용하지 않는다.

distributed reduction은 각 rank가 weighted error sum과 weight/valid count를 내고 global objective 계약대로 합친다. variable shape microbatch에서 local mean 평균을 피한다. accumulation window의 denominator가 optimizer step 전체를 대표하도록 engine scaling과 맞춘다. empty-valid rank도 collective에 참여하되 NaN을 만들지 않는다.

metric은 sample count, latent/token element count, weight sum과 modality-specific valid unit을 함께 낸다. loss 변화가 resolution·duration mixture 때문인지 field 개선인지 분리한다. checkpoint에는 sampler/curriculum과 incomplete accumulation denominator가 들어간다.

diffusion incident를 first changed plane으로 분류한다. 생성 품질이 갑자기 나빠지면 먼저 PipelineBundleID와 canary input을 고정한다. data/condition plane에서 prompt IDs·embedding, media processor와 initial noise를 비교한다. field plane에서 scale_model_input, prediction과 x0/velocity conversion을 본다. solver plane에서 sigma·history·RNG와 next state를, decode plane에서 VAE·vocoder와 postprocess를 본다.

loss spike는 sample IDs, VAE cache, time histogram, target norm, loss weight와 denominator 순서로 본다. 특정 sigma bucket만 나쁘면 parameterization conversion, precision saturation이나 curriculum drift를 조사한다. 모든 bucket이 동시에 바뀌면 data scale, optimizer·EMA와 model revision을 먼저 본다. 평균 loss 하나로 root cause를 찾지 않는다.

NaN이 나오면 최초 nonfinite tensor를 추적한다. latent encode, noise interpolation, time embedding, attention, prediction, weighted error, gradient와 optimizer state를 순서대로 검사한다. mixed precision scale와 FP8 amax history, sigma division과 clipping을 확인한다. NaN batch를 skip한 뒤 sample cursor만 전진시키면 data semantics가 달라지므로 policy를 기록한다.

latency regression은 resolution/token cohort, compile variant, kernel fallback, guidance batch, NFE와 collective tail로 분해한다. scheduler step 수가 같아도 RK4·CFG는 model forward 수가 다르다. VAE decode나 condition service가 병목이면 denoiser kernel 최적화로 해결되지 않는다.

incident 종료에는 old bundle rollback, golden trajectory, affected sample 범위, actual RPO/RTO와 새 durable checkpoint가 포함된다. threshold를 넓혀 canary를 통과시키지 않는다. 최초 차이, 수정 source와 negative regression fixture를 연결해 다음 revision이 같은 오류를 다시 내지 않게 한다.

이 흐름에서 최종 source·artifact certificate의 최소 필드도 빠뜨릴 수 없다. source 부분은 repository, commit, path, symbol, body digest, caller와 selected option branch를 담는다. Diffusers scheduler와 training example, Transformers Token2Wav, model-specific 공개 code가 각각 별 record다. 논문 equation과 tensor symbol을 연결하고 공개되지 않은 training detail은 `NotVerified`로 남긴다.

모델 산출물에는 denoiser architecture·weight·EMA, VAE, conditioner/tokenizer/processor, adapter/ControlNet, prediction type, preconditioning과 precision을 넣는다. solver artifact에는 class, config, timestep/sigma arrays, input scaling, NFE, history schema와 RNG를 넣는다. 모두 PipelineBundleID 아래 immutable digest로 묶는다.

training evidence는 DatasetRevision, latent/text cache, SampleID, transform, time/noise RNG, target, weighting, numerator·denominator, optimizer·scheduler·scaler, topology와 checkpoint generation을 포함한다. next-update certificate는 raw·EMA delta까지 비교한다. serving evidence는 trajectory trace와 decoded output provenance를 가진다.

evaluation은 modality·resolution·duration, paired prompt/seed, field·solver·decoder factor, quality·cost·safety metric, evaluator revision과 confidence interval을 기록한다. known failure와 unsupported combination을 숨기지 않는다. best checkpoint를 고른 selection rule과 비교 횟수도 넣는다.

독립 verifier는 certificate에서 analytic fixture, one-update resume, K-step trajectory와 paired decode를 다시 실행한다. dependency가 없거나 digest가 다르면 결과를 추정하지 않고 실패한다. 이 certificate가 있어야 새 model, scheduler나 CUDA optimization을 이전 결과와 정확히 비교하고 안전하게 rollback할 수 있다.

sampler 비용을 step이 아니라 field evaluation으로 회계한다. sampler 이름 옆의 20 steps는 compute를 충분히 설명하지 않는다. Euler는 보통 step당 한 번 field를 부르지만 Heun predictor/corrector, RK4와 일부 solver는 더 많은 evaluation을 요구한다. CFG가 conditional·unconditional을 한 batch로 묶더라도 계산량과 memory는 증가한다. nominal step, NFE, effective batch token/pixel과 wall-clock을 모두 기록한다.

multistep solver는 이전 output을 재사용해 높은 order를 얻을 수 있지만 warm-up과 마지막 fallback에서 order가 낮아진다. adaptive 또는 duplicate timestep path는 실제 call 수가 config와 다를 수 있다. model forward trace에서 NFE를 세고 scheduler internal counter와 대조한다. skipped step과 failed retry도 비용에 포함한다.

품질 비교는 같은 NFE, 같은 wall-clock과 각 method의 권장 설정을 별 표로 낸다. NFE 일치 비교는 field call 효율을, wall-clock 일치는 system 효율을, 권장 설정 비교는 실제 사용점을 나타낸다. 한 표만으로 보편적 우월성을 주장하지 않는다. VAE decode, condition encoding과 compile warm-up 포함 여부를 명시한다.

variable resolution과 text length는 한 field evaluation 비용을 바꾼다. average NFE만 보고 cohort mixture가 다른 run을 비교하지 않는다. pixel/latent token당 시간, attention FLOP 추정, memory peak와 kernel fallback을 함께 본다. distributed run은 slowest rank와 collective exposed time을 사용한다.

stochastic sampler는 동일 NFE에서도 seed variance가 크므로 paired seed와 distribution 평가를 함께 한다. solver error가 field error보다 작은 구간에서는 step을 더 늘려도 이득이 작다. analytic toy와 held-out field residual을 이용해 compute를 field 개선과 solver refinement 중 어디에 쓸지 판단한다.

adapter·ControlNet bundle의 병합과 scale을 검증한다. LoRA adapter를 base weight에 merge하면 \(W'=W+s(α/r)BA\) 같은 effective update가 생긴다. library convention에 따라 scale 위치가 다를 수 있으므로 실제 merge 함수와 forward injection을 고정 source에서 비교한다. unmerged scale 1과 merged output이 golden input에서 같아야 한다. dtype cast와 repeated merge를 negative fixture로 둔다.

여러 adapter를 동시에 쓰면 단순 합이 attention·normalization의 비선형 network 안에서 기대한 style interpolation을 보장하지 않는다. adapter ID, target module, scale와 load order를 PipelineBundleID에 넣는다. 이름 collision과 서로 다른 base digest의 adapter를 admission에서 막는다. scale sweep은 quality·identity leakage와 base regression을 함께 본다.

ControlNet conditioning scale은 block residual의 크기를 바꾼다. guess mode, per-block scale와 guidance schedule이 있으면 하나의 scalar로 축약하지 않는다. missing condition에서 zero residual이 base output을 보존하는지 시험한다. preprocessing alignment가 다른 control artifact를 cache hit로 사용하지 않는다.

adapter와 ControlNet을 함께 쓰면 base field, identity update와 spatial control 세 효과가 상호작용한다. base only, adapter only, control only, 둘 다의 2×2 평가로 귀속한다. 동일 initial latent, prompt, scheduler와 VAE를 쓴다. control adherence 향상이 identity memorization이나 prompt fidelity 저하와 교환되지 않는지 본다.

checkpoint와 export에는 trainable state만 아니라 required base, processor, adapter config, control model과 scale policy가 들어간다. merged export는 새 artifact이며 원 adapter를 retention한다. rollback은 component 한 파일이 아니라 승인된 bundle 전체를 복원한다.

release matrix를 지원 조합의 실증 표로 만든다. release matrix의 축은 modality, model family, VAE/decoder, conditioner, prediction type, scheduler, NFE, guidance, resolution·duration, dtype·kernel과 topology다. 모든 Cartesian product를 시험할 수 없으므로 지원할 조합과 representative boundary를 먼저 고른다. 시험하지 않은 조합은 compatible 추정이 아니라 `NotExecuted`다.

각 지원 row는 analytic target, model forward/backward, one update, checkpoint resume, K-step trajectory, decode, quality·cost와 safety 결과를 가진다. source·bundle digest와 metric revision을 연결한다. row 사이에서 공유 가능한 evidence와 scheduler·modality 때문에 재검증해야 하는 evidence를 구분한다.

boundary row에는 최소·최대 resolution, odd aspect ratio, longest text, empty/negative condition, extreme sigma, stochastic seed, multistep resume와 mixed precision을 넣는다. distributed boundary는 uneven token, tail microbatch, rank reorder와 storage failure다. happy path 중앙값만 시험하지 않는다.

새 scheduler를 추가하면 model forward가 같다는 fixture, prediction conversion과 full trajectory를 추가한다. 새 VAE는 encode/decode baseline과 denoiser latent scale을 다시 본다. 새 kernel은 selected shapes의 eager parity와 trajectory를 본다. affected row만 stale 처리하되 dependency graph가 누락되지 않게 한다.

publication gate는 matrix row의 PASS와 certificate signature를 읽어 immutable bundle pointer를 이동한다. canary failure나 mixed revision replica가 있으면 old pointer를 유지한다. rollout 뒤 field·solver·decode metric이 승인 범위에 있는지 확인하고 rollback rehearsal을 실행한다. 이 표가 문서와 배포 admission의 공통 진실이어야 한다.

이 흐름에서 새 diffusion 논문을 읽을 때 적용하는 질문 순서도 빠뜨릴 수 없다. 첫 질문은 생성 state가 무엇인가다. pixel, latent, token, mel, action과 3D frame 중 무엇을 움직이는지 확인한다. 둘째는 source와 target distribution, forward probability path와 coupling이다. 셋째는 time convention, model input, prediction target과 loss weighting이다. 이 세 질문 없이 “diffusion” 또는 “flow”라는 이름만 비교하지 않는다.

넷째는 architecture와 condition 경계다. U-Net·DiT·hybrid, VAE·decoder, text/media encoder와 control input이 어느 tensor에서 만나는지 그린다. 다섯째는 sampling transition이다. ODE/SDE/categorical update, prediction conversion, input scaling, solver history, stochastic noise와 NFE를 확인한다. 학습 objective와 sampler가 어떤 식으로 이어지는지 적는다.

여섯째는 공개 evidence 범위다. 논문 equation, repository commit, model card, training example, inference scheduler와 test가 각각 무엇을 증명하는지 구분한다. inference code만 보고 비공개 training target을 확정하지 않는다. 소스 심볼과 tensor shape를 식의 항에 연결하고 default option을 effective config에서 확인한다.

일곱째는 data와 estimator다. media transform, cache, timestep proposal, loss denominator, condition dropout과 distributed reduction을 본다. 여덟째는 state와 failure다. optimizer·EMA·RNG·sampler·solver history·component bundle이 checkpoint와 service에서 어떻게 복원되는지 묻는다. negative fixture가 최초 경계에서 실패해야 한다.

마지막은 주장 수준이다. analytic fixture, one-step update, trajectory, paired quality, compute와 safety 중 무엇을 실제로 실행했는지 표시한다. model 개선, solver 개선, decoder 개선과 evaluator 변경을 분리한다. 이 순서를 지키면 새로운 이름이 등장해도 수식, code, state와 evidence의 빈칸을 빠르게 찾아낼 수 있다.

검토 결과의 최소 산출물은 EquationCard, SourceCard, ComponentBundle, GoldenTrajectory, FailureTrace와 EvaluationCertificate다. EquationCard는 state·time·target·weight를, SourceCard는 fixed commit과 selected branch를, ComponentBundle은 weight·VAE·conditioner·scheduler·decoder를 고정한다. GoldenTrajectory는 최초 state부터 decode까지의 선택 tensor를, FailureTrace는 잘못된 convention과 missing history가 막힌 지점을, EvaluationCertificate는 paired 품질·비용·안전과 미지원 범위를 담는다.

이 여섯 문서가 동일 ArtifactID와 digest를 사용해야 한다. 수식은 epsilon인데 config는 v, model은 새 VAE인데 evaluation cache는 이전 decoder인 혼합을 자동 검출한다. 독립 검토자는 임의의 row를 골라 source에서 tensor 식을 찾고 golden step을 다시 계산한다. 재계산할 수 없는 주장은 삭제하거나 `NotVerified`로 낮춘다. 이러한 엄격함이 새로운 diffusion architecture를 빠르게 받아들이면서도 설명의 신뢰성을 유지하는 방법이다.

release 서명에는 tested seed, resolution, duration, condition length, sigma range, NFE, dtype, topology와 hardware도 포함한다. 같은 source와 weight라도 kernel, collective order와 decoder backend가 달라 trajectory가 바뀔 수 있다. 허용 오차는 결과를 본 뒤 넓히지 않고 component별 first-difference budget으로 사전 고정한다. 새 runtime은 이 matrix를 다시 통과한 뒤에만 기존 capability를 상속한다.

## 22.16 코드 워크스루: FlowMatch Euler `step`에서 시간 방향을 증명한다

flow matching을 학습할 때 `u_theta(x_sigma,sigma)`가 잘 맞아도 sampler가 반대 방향으로 적분하면 결과는 무너진다. Diffusers 고정 revision `d57cecde`의 `FlowMatchEulerDiscreteScheduler.step`은 model output, sigma schedule, mutable step index와 stochastic branch가 만나는 가장 짧은 production 경계다. 이 함수는 training loss가 아니라 **학습한 velocity를 한 번 소비하는 해석기**이므로, 논문 objective와 inference parameterization을 연결할 때 둘을 같은 함수로 오인하지 않는다.

### 짧은 원문을 ODE 한 스텝으로 번역한다

핵심 경로는 `scheduling_flow_match_euler_discrete.py:483-511` 가운데 다음 부분이다.

```python
sample = sample.to(torch.float32)
sigma = self.sigmas[self.step_index]
sigma_next = self.sigmas[self.step_index + 1]
current_sigma = sigma
next_sigma = sigma_next
dt = sigma_next - sigma
if self.config.stochastic_sampling:
    x0 = sample - current_sigma * model_output
    noise = randn_tensor(sample.shape, generator=generator, ...)
    prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
else:
    prev_sample = sample + dt * model_output
```

`sample`과 `model_output`은 이미지 latent라면 `[B,C,H,W]`, video latent라면 구현에 따라 `[B,C,T,H,W]`, token field라면 `[B,L,D]`처럼 같은 shape다. `sigma`, `sigma_next`, `dt`는 scalar이며 전체 state에 broadcast된다. 함수는 update 전에 sample을 FP32로 올려 작은 `dt*u`가 저정밀 덧셈에서 사라지는 위험을 줄이고, deterministic 경로가 끝나면 output을 model dtype으로 되돌린다. scheduler의 mutable `_step_index`가 어느 sigma pair를 읽을지 소유한다.

선형 conditional path를 `x_sigma=(1-sigma)x_0+sigma epsilon`으로 놓으면 sigma가 noise 쪽 좌표이고 `dx/dsigma=epsilon-x_0`다. model이 이 velocity `u`를 예측할 때 Euler 식은

\[
x_{\sigma_{k+1}}=x_{\sigma_k}+(\sigma_{k+1}-\sigma_k)u_\theta(x_{\sigma_k},\sigma_k)
\]

다. 코드의 `dt`와 마지막 줄이 정확히 이 식이다. denoising schedule은 보통 sigma가 감소하므로 `dt<0`이다. “이전 sample”이라는 이름 때문에 `sigma-sigma_next`를 쓰면 부호가 두 번 뒤집히지 않는다. 실제로는 velocity와 반대 부호로 이동해 data endpoint에서 멀어진다.

stochastic branch는 단순히 Euler 결과에 noise를 더하지 않는다. 먼저 `x0=sample-sigma*model_output`으로 clean endpoint를 추정하고, 다음 sigma에서 `(1-sigma_next)x0+sigma_next*noise`로 다시 섞는다. 따라서 `stochastic_sampling=True`는 RNG state뿐 아니라 transition 식 자체를 바꾼다. `s_churn` 인자가 signature에 있어도 이 고정 함수 본문의 선택 경로에는 사용되지 않는다. 다른 Euler scheduler에서 본 churn 의미를 이 클래스에 자동 이식하면 안 된다.

### 세 숫자로 shape·state·최초 분기를 검산한다

scalar fixture를 `x0=2`, `epsilon=-1`, `sigma=0.75`, `sigma_next=0.25`로 잡자. forward state는 `x_sigma=0.25·2+0.75·(-1)=-0.25`, 정확한 velocity는 `u=epsilon-x0=-3`, `dt=-0.5`다. deterministic step은 `-0.25+(-0.5)(-3)=1.25`이고, 이는 직접 계산한 `x_0.25=0.75·2+0.25·(-1)=1.25`와 같다. 이 등식은 model 품질과 무관하게 scheduler convention을 검사한다.

첫 변형에서는 `dt`를 양수 `sigma-sigma_next`로 바꾼다. 최초 차이는 model output이 아니라 `dt`이며 결과가 `-1.75`로 data에서 멀어진다. 둘째는 `model_output=+3`으로 target sign만 뒤집는다. 최초 차이는 model output checksum이고 같은 `dt`에서 역시 `-1.75`가 된다. 셋째는 BF16 sample `4096`과 작은 update를 사용한다. 함수가 FP32로 올리지 않은 복제본과 비교하면 최초 차이가 update 덧셈에서 나타날 수 있다. 넷째는 동일 generator state로 stochastic branch를 두 번 실행해 noise checksum과 output을 맞추고, generator를 복원하지 않은 run은 `randn_tensor`에서 처음 갈라져야 한다.

per-token time을 쓰면 함수는 `[B,L]` sigma마다 schedule에서 바로 아래 값을 찾아 `dt`를 `[B,L,1]`로 만든다. 이때 global `_step_index`의 scalar sigma pair와 동일하다고 가정하지 않는다. token별 sigma가 같을 때만 scalar 경로와 비교하고, 서로 다른 sigma fixture에서는 각 token이 자기 `dt`로 움직이는지 본다. 함수가 어느 경우에도 마지막에 `_step_index`를 하나 올린다는 사실도 resume state에 포함한다.

고정 revision에는 이 scheduler의 독립 unit test 파일이 확인되지 않는다. `tests/pipelines/stable_audio_3/test_stable_audio_3.py:233-247`은 stochastic scheduler 설정이 pipeline의 8-step 기본값을 선택한다는 통합 계약만 검증한다. 위 Euler 수치, integer timestep 거부, scalar/per-token parity, dtype 복원과 RNG replay는 별도 golden fixture로 채워야 한다. pipeline test를 step 식의 직접 증거로 과장하지 않는 것이 중요하다.

trajectory 비교 순서는 `initial sample → passed timestep 값 → step_index → current/next sigma → dt → model_output → x0 또는 noise draw → FP32 prev_sample → output dtype → incremented step_index`다. 마지막 waveform이나 image만 비교하지 않는다. 학습 target sign 오류는 model output에서, schedule 역전은 sigma/dt에서, resume 오류는 step index에서, stochastic 재현 오류는 noise draw에서 처음 나타난다. 이렇게 first divergence를 고정하면 flow 논문의 경로 미분, model parameterization과 production scheduler의 mutable state가 하나의 검증 가능한 설명으로 닫힌다.

운영 조치는 최초로 갈라진 주체에만 적용한다. `x_t`가 먼저 다르면 data·VAE·noise/time sampler를, model output이 먼저 다르면 target parameterization과 condition path를, `dt` 이후만 다르면 scheduler·solver state를 고친다. 마지막 sample의 품질 지표가 나빠졌다는 이유만으로 세 층을 동시에 바꾸면 원인을 지우고 새 trajectory를 만들 뿐이다. 수정 뒤에는 같은 scalar fixture와 실제 latent의 짧은 trajectory를 함께 재생해 수식 계약과 production 경로가 모두 닫혔는지 확인한다.

## 22.17 BD3와 MDLM의 checkpoint는 model weight보다 넓다

저장·복구 경로를 다음 corruption의 입력으로 읽는다. 두 구현의 `on_save_checkpoint`는 EMA가 있으면 `checkpoint['ema']`를 저장하고, optimizer step에 `accumulate_grad_batches`를 곱해 Lightning batch progress를 교정한다. MDLM은 sampler의 `random_state`도 보존한다. BD3는 여기에 `sampling_eps_min/max`를 넣고, load 때 `torch.compile`이 붙인 `_orig_mod.` prefix를 제거한다. 즉 상태 전이는 `optimizer progress → batch cursor`, `sampler RNG → 다음 표본`, `sampling epsilon → 다음 time/noise 좌표`, `EMA → 평가·sampling weight` 네 갈래다.

짧은 원문인 `completed = optimizer_steps * accumulate_grad_batches`가 필요한 이유는 framework의 기본 batch counter가 한 iteration 뒤처질 수 있기 때문이다. 그러나 accumulation 설정을 resume 때 바꾸면 같은 optimizer step에서도 다른 batch cursor가 만들어진다. `_orig_mod.` 제거 역시 key 호환을 복구할 뿐 tensor 값·optimizer slot·새 compile graph의 동등성을 증명하지 않는다.

최초 불일치와 장애 주입을 고정한다. 복구 비교 순서는 `global_step → optimizer progress → derived batch progress → sampler random_state → next SampleID → sampled t/epsilon → corruption checksum → EMA checksum`이다. BD3에서 `sampling_eps_min/max`를 누락하면 최초 차이는 time sampler에, MDLM에서 sampler state를 누락하면 SampleID 또는 corruption RNG에 생긴다. EMA만 빠지면 training loss가 같아도 첫 evaluation output이 갈라질 수 있다.

변형 실험은 checkpoint를 저장한 뒤 accumulation을 4에서 8로 바꾸는 것이다. load 성공을 복구 성공으로 세지 말고, derived batch cursor와 다음 SampleID가 기준 run과 달라지는 순간 거부해야 한다. 진단 체크리스트는 compile prefix 집합, EMA key·shape, optimizer step, 저장 당시 accumulation, epoch/batch progress, sampler state, epsilon 범위, 첫 corruption checksum을 포함한다. 이 가운데 하나라도 비교할 수 없으면 checkpoint는 읽혔을 뿐 trajectory가 복구된 것은 아니다.

## 22.18 causal video token과 diffusion latent를 같은 표현으로 오인하지 않는다

continuous AutoencoderKL latent는 mean/log-variance posterior에서 sample한 실수 tensor이고 discrete video tokenizer는 index stream을 낸다. 이름에 MAGVIT이 들어간 Diffusers autoencoder가 MAGVIT-v2의 LFQ token을 구현한다고 단정할 수 없다. downstream diffusion이 기대하는 scaled latent와 autoregressive LM이 기대하는 integer vocabulary는 저장 schema와 loss가 다르다.

temporal causality는 문서 문구가 아니라 prefix invariance로 시험한다. 같은 앞부분과 서로 다른 미래 frame을 넣었을 때 앞 token이 같아야 한다. full clip과 chunk encode의 index, overlap state, first-frame policy와 seam도 비교한다. tiled decode의 영상 seam test는 temporal causal encode를 자동 증명하지 않는다.

공개 shape·chunk 시험은 특정 구현의 tensor 계약을 고정할 뿐 논문의 reconstruction FID·LPIPS, sampling 품질, token rate와 GPU throughput을 재현하지 않는다. training code·dataset mixture·distributed codebook state가 공개되지 않은 기법은 해당 빈칸을 `NegativeEvidence`로 유지한다.
