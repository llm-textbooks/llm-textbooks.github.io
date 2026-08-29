# 21장 이미지·음성·영상이 학습 신호가 되는 순간

텍스트 모델에 이미지를 붙인다고 입력이 “조금 더 길어지는” 것은 아니다. JPEG의 가변 길이 바이트, 음성의 시간축, 영상의 공간·시간축을 Transformer가 소비할 수 있는 일정한 tensor로 바꾸는 순간부터 표본의 의미와 손실 분모가 달라진다. 이 장은 `MediaSampleID` 하나가 decoder의 loss-bearing token이 될 때까지를 추적한다.

먼저 한 줄의 상태 사슬을 세운다. 이 장의 모든 구현과 장애는 다음 사슬의 어느 화살표가 의미를 바꾸었는지 찾는 문제다.

```text
raw bytes·sample·PTS
  → decoder와 processor가 만든 pixel·waveform·frame
  → patch·tubelet·log-Mel frame 또는 codec symbol
  → vision/audio encoder의 연속 feature 또는 이산 code
  → projector·merger·resampler가 만든 language-width rows
  → placeholder 치환 또는 cross-attention memory
  → attention mask·position·labels
  → modality별 loss sum/count
  → encoder·connector·decoder별 gradient와 optimizer update
  → checkpoint·processor bundle·feature cache
  → grounding·ASR·temporal·counterfactual evaluation
```

이 사슬에는 서로 바꿔 쓸 수 없는 네 종류의 ‘token’이 등장한다. JPEG byte는 token이 아니다. ViT patch row는 연속 벡터이고 vocabulary ID가 아니다. RVQ code는 이산 ID지만 text vocabulary와 별도 codebook의 좌표다. `<image>`는 media 자체가 아니라 feature rows가 들어갈 자리를 표시하는 제어 기호다. 어느 종류인지 생략한 채 “이미지가 576 token이 된다”고 쓰면 cache key, embedding lookup, loss target과 메모리 계산 가운데 적어도 하나를 잘못 설명하게 된다.

따라서 이 장의 독해 순서도 구현의 순서를 따른다. 먼저 각 modality의 시계와 좌표를 고정하고, 그 좌표가 tensor shape로 어떻게 접히는지 확인한다. 다음으로 그 tensor의 어느 원소가 loss의 분자와 분모에 들어가는지 추적한다. 마지막으로 값·shape·소유권이 처음 갈라지는 경계에 실패를 주입하고, 그 경계를 재현하는 관측값으로 디버깅한다. 이 순서를 거꾸로 밟아 평균 loss부터 해석하면 전처리 결함을 optimizer 문제로 오인하기 쉽다.

한 표본의 장부는 값보다 변환을 보존한다. `MediaSampleID`에서 시작해 각 경계마다 `입력 ID → 함수와 고정 revision → option → 출력 TensorID(shape·dtype·checksum) → 좌표 map → state owner`를 한 행으로 남긴다. 예컨대 crop 뒤 pixel checksum은 같아도 `image_grid_thw`가 다르면 위치 의미가 같지 않다. feature checksum은 같아도 두 이미지의 placeholder owner가 뒤바뀌면 조건이 달라진다. logits가 같아도 유효 label 수가 달라지면 update의 추정량이 다르다. 그러므로 “같다”는 판정은 bytes, 좌표, 순서, 분모, parameter owner 중 어느 층까지 같은지를 붙여 말한다.

네 모델 패밀리를 같은 계약으로 읽는다.

모델 이름은 호출 사슬을 대신하지 않는다. 아래 표는 공개 구현에서 확인할 수 있는 경계를 같은 질문으로 맞춘 것이다. 학습 mixture처럼 공개 코드가 증명하지 않는 항목은 이 표에서 추론하지 않는다.

| 패밀리 | processor가 만드는 계약 | 결합 경계와 shape 질문 | 주 손실과 gradient 질문 | 고정 source에서 확인할 좌표 |
|---|---|---|---|---|
| LLaVA형 | 대화 token과 `IMAGE_TOKEN_INDEX`, pixel batch, 이미지 순서 | vision feature `[N_i,D_v]`를 `mm_projector`로 `[N_i,D]`로 만든 뒤 placeholder 하나를 가변 길이 열로 확장한다. 확장 뒤 labels와 attention mask가 같은 offset으로 재작성되는가 | 보통 assistant text CE가 connector와, 해제했다면 vision tower까지 간접적으로 학습한다. projector 전용 LR·저장 분기가 실제 parameter set과 일치하는가 | `sources/training-multimodal-llava/llava/model/llava_arch.py:131-270`의 `encode_images`, `prepare_inputs_labels_for_multimodal`; `llava/train/llava_trainer.py:165-192,239-247`의 optimizer·저장 분기 |
| PaliGemma형 | processor가 이미지마다 고정된 수의 `<image>` occurrence를 text에 미리 넣고 pixel tensor를 만든다 | vision output을 `PaliGemmaMultiModalProjector`로 투영한 뒤, image-token mask가 가리키는 embedding 원소에 scatter한다. occurrence 수 × hidden width와 feature 원소 수가 정확히 같은가 | conditional generation labels의 CE가 language model과 projector, freeze 정책에 따른 vision tower로 흐른다. placeholder는 supervision 대상인지 context인지 label mask로 확인한다 | `sources/transformers-main-qwen4exp/src/transformers/models/paligemma/processing_paligemma.py:200-218`의 `replace_image_token`; `modeling_paligemma.py:90-177`의 projector·`get_image_features`·cardinality 검사 |
| Qwen2-VL형 | flatten된 patch와 `image_grid_thw`; video이면 시간축을 포함한 grid | vision patch embedding→blocks→merger를 지난 열이 text placeholder와 결합된다. `grid_thw`가 feature 개수뿐 아니라 3축 rotary 좌표 생성까지 살아 있는가 | language CE의 mask 외에 dynamic resolution이 유효 target당 계산량을 어떻게 바꾸는가. vision merger와 language parameter의 trainable owner는 누구인가 | `sources/transformers-v5.15.1/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`의 patch embed·rotary·merger·model `forward`; 같은 revision processor의 grid 생성 경로 |
| Whisper형 | waveform을 resample·pad/trim한 뒤 연속 log-Mel `[B,F,L]`과 길이/mask를 만든다 | convolution stride와 encoder가 시간 열을 줄이고 decoder가 cross-attention memory로 읽는다. text sequence 안의 placeholder 치환은 없다 | decoder token CE 또는 별도 head의 목적함수다. audio frame은 대개 context이지 vocabulary target이 아니다. padding·실제 무음·truncation이 encoder mask에서 구별되는가 | `sources/transformers-v5.15.1/src/transformers/models/whisper/feature_extraction_whisper.py`; `modeling_whisper.py:540-648`의 encoder와 decoder cross-attention 경로 |

PaliGemma와 LLaVA는 모두 image feature를 language width에 맞추지만 splice 계약은 같지 않다. LLaVA의 `prepare_inputs_labels_for_multimodal`은 한 placeholder 주변의 text 구간을 나누고 feature 열을 삽입하며 새 label 열을 만든다. PaliGemma processor는 `image_seq_length`만큼 image token을 먼저 반복하고 model은 개수 검사를 거쳐 `masked_scatter`한다. 따라서 LLaVA fixture의 “placeholder 하나”를 PaliGemma에 그대로 적용하거나, PaliGemma의 exact occurrence 검사를 LLaVA의 가변 expansion으로 일반화하면 안 된다.

Qwen2-VL은 길이만 맞추는 검사로 충분하지 않다. 같은 원소 수 `T×H×W`를 가진 `grid_thw` 두 개라도 시간과 공간의 배치가 다르면 rotary phase와 merger 이웃이 달라진다. Whisper는 더 근본적으로 text embedding splice가 아니라 encoder-decoder 조건화다. 네 패밀리를 모두 “projector가 media token을 text에 넣는다”로 압축하면, Whisper에는 존재하지 않는 placeholder를 만들고 Qwen의 3축 좌표를 지우게 된다.

고정 revision을 한 번 끝까지 걸어 본다. LLaVA의 내부 `forward`에서는 `get_image_features`의 결과를 이어 붙인 뒤 placeholder mask의 원소 수와 feature 원소 수를 먼저 맞추고, 통과한 값만 `masked_scatter`한다. 따라서 이 경로에서 첫 질문은 “projector가 학습되는가”가 아니라 “processor가 만든 image token 자리와 vision tower가 낸 행이 정확히 같은가”다. projector 자체는 `Linear→activation→Linear`일 뿐이다. base vision tower 동결이나 optimizer 포함 여부는 이 클래스만 읽고 추론할 수 없으며, training script의 `requires_grad`와 parameter group을 따로 확인해야 한다.

PaliGemma는 비슷해 보이지만 projector 뒤에 hidden size의 제곱근으로 나누는 값 변환이 있다. 이 식이 존재한다는 사실과 “두 modality의 분산을 이상적으로 맞춘다”는 효과 주장은 구분한다. conditional-generation wrapper는 labels를 내부 모델에 전달하고 LM head 뒤 loss 함수에 다시 넘기지만, 어느 image·prompt 위치가 `-100`인지는 이 wrapper가 만들지 않는다. loss 이상을 조사할 때 model forward만 보고 마스크가 옳다고 결론 내리지 말고 processor·collator가 만든 label atlas를 함께 펼쳐야 한다.

Qwen2-VL의 이미지와 영상 경로는 같은 visual module을 호출하더라도 각각 `image_grid_thw`와 `video_grid_thw`를 보존한다. 영상의 `T`를 이미지 batch 수처럼 취급해 곱만 맞추면 cardinality 검사는 통과할 수 있어도 시간 rotary 좌표는 달라진다. 최소 fixture는 원소 수가 같고 `T,H,W` 분해만 다른 두 grid를 만들어 position과 merger 출력이 달라지는지 보는 것이다. 반면 Whisper는 labels가 주어지고 decoder 입력이 없을 때 labels를 start/pad 규칙으로 오른쪽 이동한다. audio frame은 encoder memory이며 decoder token label과 같은 축이 아니다. 이 네 경로를 같은 표에 놓는 이유는 API 이름을 통일하려는 것이 아니라, placeholder·grid·cross-attention 가운데 어느 계약을 시험해야 하는지 갈라내기 위해서다.

확산 학습도 같은 방식으로 식과 호출자를 나눈다. `DDPMScheduler.add_noise`가 보장하는 것은 주어진 timestep에서 `sqrt(alpha_bar_t)·x0 + sqrt(1-alpha_bar_t)·epsilon`을 broadcast 계산한다는 사실이다. timestep을 균등하게 뽑는 것은 text-to-image training loop이며 scheduler 함수의 보장이 아니다.

같은 loop는 VAE latent에 scaling factor를 적용하고 noise와 timestep을 만든 뒤, `prediction_type`이 epsilon이면 noise를, v-prediction이면 velocity를 target으로 고른다. scheduler 설정과 target branch가 어긋나면 tensor shape는 그대로여서 학습이 돌아가지만 model output의 의미가 바뀐다. 디버깅 장부에는 `latent checksum→timestep→prediction_type→target checksum→loss numerator/count`를 한 행으로 남긴다.

SDXL DreamBooth LoRA 예제에서는 VAE·두 text encoder·UNet base를 먼저 동결하고, UNet의 q/k/v/out projection에 adapter를 붙인다. `train_text_encoder`가 켜진 경우에만 두 text encoder에도 adapter가 늘어난다. prior preservation은 batch의 prediction과 target을 둘로 나눠 두 번째 절반에서 prior MSE를 계산하고 `instance_loss + prior_loss_weight·prior_loss`로 합친다.

그러므로 옵션 하나는 데이터 collate 순서, trainable parameter set, loss graph를 동시에 바꾼다. 점검할 때에는 instance/class ID 배열, chunk 경계, 두 loss의 sum/count, 실제 optimizer parameter ID를 함께 저장한다. class image가 있다는 사실만으로 prior 항이 활성화됐다고 판단해서는 안 된다.

마지막으로 증거의 부정 경계도 기록한다. Diffusers의 layerwise casting은 `storage_dtype`, `compute_dtype`, skip pattern과 class를 hook 설치 함수에 넘기지만, 그 진입점만으로 optimizer state와 serialization 메모리가 얼마나 줄었다고 말할 수는 없다. ASR evaluator는 audio·label column과 generation option을 상위 evaluator에 전달하지만 WER 정규화 규칙을 그 절에서 정하지 않는다. `EvaluationModule.compute`는 feature schema를 검사하고 finalize한 뒤 rank 0에서만 metric을 계산한다. 이 세 사례처럼 “코드가 하는 일”과 “그 결과로 기대하는 효과” 사이에 별도 test·measurement 칸을 두면, 정적 소스 인용이 성능 실험인 것처럼 부풀려지는 일을 막을 수 있다.

최초 불일치가 담당자를 정한다.

최종 정답 하나만 비교하면 서로 다른 결함이 모두 ‘멀티모달 성능 저하’로 보인다. 같은 `MediaSampleID`의 두 실행을 아래 순서로 나란히 놓고 최초로 달라진 행에서 조사를 시작한다.

| 관측된 최초 불일치 | 먼저 고정할 상태 | 최소 분리 실험 | 성급하게 건드리지 않을 것 |
|---|---|---|---|
| decoded pixel·waveform·frame | codec build, color/channel, sample rate, orientation, PTS·seek | 같은 bytes를 CPU reference backend와 문제 node에서 decode해 transform 전 checksum을 비교 | projector·learning rate |
| crop·patch·Mel·tubelet | processor revision, resize/crop, window/hop, frame sampler, RNG | 의미를 아는 checkerboard·click·timestamp 영상으로 좌표 map을 손계산 | language loss weight |
| encoder feature | loaded tower/checkpoint, dtype, train/eval, selected layer | 동일 processor tensor를 tower에 직접 넣고 layer별 최초 divergence를 찾음 | dataset caption 정제 |
| projector·resampler output | feature selection, connector weight, cast·norm, cache generation | frozen synthetic feature를 connector에 넣어 output과 gradient를 finite-difference 또는 golden 값과 비교 | media decoder |
| embedding의 media span | placeholder occurrence, flatten/unflatten 순서, owner offset | 두 단색 이미지를 교환하되 shape를 유지해 span checksum이 함께 교환되는지 확인 | vision tower 재학습 |
| position·attention·labels | grid, padding side, causal/cross mask, ignore index, truncation | 2×2 patch와 짧은 답을 손으로 펼쳐 위치별 source·target·visibility atlas를 비교 | optimizer 종류 |
| loss sum/count만 | shift, 유효 target, modality weight, DP reduction | 같은 logits·labels에서 numerator와 denominator를 독립 재계산하고 world size 1·2를 비교 | model architecture |
| gradient·parameter delta | freeze map, detach, optimizer group, accumulation·scaler | modality 하나씩 backward해 `grad is None`, norm, update/weight ratio를 parameter ID별 비교 | processor cache 삭제 |
| resume 이후 표본부터 | sampler cursor, prefetch commit, transform RNG, curriculum phase | 연속 run과 save/resume run의 다음 64개 ID·PTS·crop을 먼저 비교 | CUDA kernel 결정성 |
| export/runtime에서만 | processor bundle, projector·adapter key, quantization/cast, loader dispatch | 고정 media의 processor output→projector output→첫 logits를 export 전후 비교 | 학습 데이터 재수집 |

이 표의 순서는 인과 순서다. 예를 들어 서로 다른 rank에서 loss가 갈렸더라도 decoded checksum이 먼저 다르면 NCCL 문제가 아니다. 반대로 processor와 logits까지 같은데 loss count만 다르면 codec을 바꿀 이유가 없다. 평균 metric이 회복돼도 최초 불일치를 제거하지 못했다면 우연한 상쇄일 수 있으므로 승인을 보류한다.

## 21.1 raw image·audio·video를 표현 단위로 자른다

### 21.1.1 patch와 tubelet이 보존·폐기하는 좌표

높이 `H`, 너비 `W`, 채널 `C`인 이미지를 `P×P` patch로 자르면 visual sequence 길이는 대략 `N=(H/P)(W/P)`다. patch projection의 입력은 `[B,N,P²C]`, 출력은 `[B,N,D_v]`다. `P`를 두 배로 키우면 token 수는 1/4이 되지만 작은 문자와 경계 정보가 한 patch 안에서 섞인다. 영상 tubelet은 여기에 시간 폭 `T_p`를 더해 `N=(T/T_p)(H/P)(W/P)`가 된다. frame 수만 기록하고 sampling FPS와 시작 PTS를 버리면 resume 뒤 같은 tensor를 재생성할 수 없다.

왜 이 좌표를 모두 남겨야 할까. decode library나 crop 위치가 달라지면 같은 파일도 다른 픽셀을 내고, 그 차이는 patch projection을 지난 뒤 원본과 대조하기 어려운 표현 차이로 바뀐다. 따라서 첫 표본에는 원본 SHA와 decode library revision, resize·crop 좌표, sampled frame PTS, patch grid를 함께 저장한다. 학습이 갈라졌을 때 이 기록을 거꾸로 따라가면 파일 선택 오류인지 전처리 오류인지부터 가를 수 있다. shape가 같다는 사실만으로는 같은 표본임을 증명할 수 없다.

### 21.1.2 log-Mel과 RVQ의 rate·distortion 계약

음성은 waveform `[B,T]`를 STFT와 Mel filterbank로 `[B,F,L]`로 바꿀 수 있다. window, hop, padding, center 옵션이 `L`과 timestamp 대응을 바꾼다. EnCodec류 RVQ는 encoder latent를 여러 codebook의 이산 index `[B,Q,L]`로 바꾼다. bitrate를 올리면 `Q`가 늘고 재구성은 좋아지지만 decoder가 처리할 symbol rate와 embedding table이 커진다. 사용되지 않는 codebook entry는 gradient가 0인 채 죽을 수 있으므로 perplexity와 usage histogram을 평균 하나로 뭉개지 않는다.

### 21.1.3 visual tokenizer와 feature encoder를 구분한다

ViT feature는 대개 연속 벡터다. VQ-VAE·VQGAN·MAGVIT류 tokenizer는 codebook index라는 이산 symbol을 만든다. 둘 다 관습적으로 visual token이라 부르지만 학습 계약은 다르다. 연속 feature는 projector와 decoder loss를 통해 gradient를 받고, 이산 token은 codebook commitment·reconstruction·perceptual/adversarial loss와 token prediction objective를 가질 수 있다.

nearest code `k*=argmin_k ||z_e-e_k||²`는 미분 불가능하므로 straight-through estimator나 EMA codebook update가 들어간다. commitment coefficient를 높이면 encoder output이 codebook에 붙지만 reconstruction 자유도가 줄 수 있다. usage entropy, dead entry, reconstruction metric을 language loss와 함께 본다.

timestamp를 tensor index로 바꾸는 식.

audio sample rate가 `f_s`, STFT hop이 `h`라면 frame `j`의 기준 시간은 대략 `jh/f_s`다. video frame PTS와 가장 가까운 audio frame을 매핑할 때 container time base와 resampling을 반영한다. 단순 array index로 sync를 맞추면 variable frame rate 영상에서 드리프트한다. manifest에는 원 PTS와 최종 feature index를 모두 기록한다.

논문의 표현 단위가 구현에서 바뀌는 지점.

ViT 논문은 이미지를 patch sequence로 바꾸는 핵심 아이디어를 제시하지만 production processor는 resize, shortest-edge policy, crop, channel normalization과 padding을 먼저 수행한다. 논문의 `224×224` 입력을 읽고 원본 사진을 곧바로 `14×14` patch로 나눈다고 상상하면 실제 code path를 놓친다. processor config가 `H'×W'`를 결정하고 vision embedding이 kernel/stride로 patch를 만든 뒤 class token 또는 register token을 더할 수 있다. 따라서 실제 길이는 `(H'/P)(W'/P)+N_special`이다.

음성도 Whisper 논문의 log-Mel 설정과 library feature extractor default가 같은지 확인해야 한다. `center=True` padding, waveform normalization, chunk length와 attention mask 반환 여부가 frame 수를 바꾼다. 영상 모델은 uniform frame sampling, scene-aware sampling, fixed FPS 중 무엇을 쓰는지에 따라 동일 파일에서도 다른 표본을 만든다. 논문은 모델 아이디어를 설명하고, processor source와 config가 우리 run의 tensor를 결정한다.

대표 구현을 읽을 때에는 `processor.__call__→image/audio/video preprocess→batch feature→model forward`를 따라간다. 각 경계에서 Python list가 tensor로 바뀌는 시점, batch padding의 owner, device/dtype 이동의 owner를 적는다. data worker가 GPU tensor를 만들거나 model forward가 다시 normalize하면 복사와 중복 전처리가 생길 수 있다.

tokenizer backward의 소유권.

이산 tokenizer를 end-to-end로 함께 학습하는지, 사전학습 후 freeze하는지에 따라 backward graph가 달라진다. frozen VAE/RVQ라면 code index 또는 latent가 dataset artifact처럼 취급되고 tokenizer optimizer state가 없다. 공동 학습이면 reconstruction/adversarial/commitment loss와 language loss가 tokenizer encoder·decoder·codebook에 어떤 비율로 흘러가는지 기록해야 한다.

straight-through quantization에서는 forward가 선택된 code vector를 쓰지만 backward는 encoder output으로 identity gradient를 흘리는 식의 근사를 쓴다. “argmin을 미분했다”고 설명하면 틀린다. codebook을 gradient로 업데이트하는지 EMA count/sum으로 업데이트하는지도 checkpoint state를 바꾼다. EMA 방식은 entry별 count와 accumulator를 저장하지 않으면 resume 뒤 codebook trajectory가 달라진다.

vision tower를 freeze해도 projector 입력 activation은 projector gradient 계산에 필요하다. activation checkpointing 경계를 vision tower 안/밖 어디에 두는지에 따라 memory와 recompute 비용이 달라진다. freeze는 parameter gradient를 없애지만 모든 activation 저장을 자동으로 없애지는 않는다.

## 21.2 encoder·projector·resampler가 decoder 좌표를 만든다

### 21.2.1 연속 feature와 token ID를 혼동하지 않는다

텍스트 ID를 embedding lookup해 얻은 `[B,L_t,D]`와 vision tower의 `[B,L_v,D_v]`는 projector를 거쳐 같은 hidden width `D`가 된다. placeholder ID의 위치에 feature를 splice할 때 `L_v`가 placeholder 수와 다르면 이후 label, attention mask, position ID가 모두 밀린다. 올바른 검사는 “forward가 성공했다”가 아니라 splice 전후 각 text token의 byte offset과 최종 position을 비교하는 것이다.

### 21.2.2 placeholder·mask·loss denominator를 함께 바꾼다

vision feature가 context만 제공한다면 그 위치의 label은 ignore index여야 한다. 유효 label 수 `M=Σ1[label≠ignore]`로 loss를 나누지 않고 전체 sequence 길이로 나누면 이미지 해상도가 높을수록 같은 답변의 gradient가 작아진다. 반대로 discrete image token을 예측하는 모델은 modality별 loss weight와 vocabulary partition을 명시해야 한다. `GoldenBatchID`에는 modality별 유효 target 수를 따로 기록한다.

### 21.2.3 projector의 shape·값·gradient를 검산한다

첫째, placeholder span과 feature length를 확인한다. 둘째, splice 뒤 원 text token의 상대 순서와 label이 보존되는지 확인한다. 셋째, projector parameter에 gradient가 도달하고 frozen vision tower에는 정책대로 도달하지 않는지 본다. forward 성공은 셋 중 아무것도 보장하지 않는다.

```python
assert image_features.shape[-1] == text_embeds.shape[-1]
assert inserted_positions.numel() == image_features.shape[-2]
assert labels[inserted_positions].eq(ignore_index).all()
```

이 코드는 라이브러리 소개이 아니라 golden invariant다. 실제 Transformers 고정 revision의 multimodal model별 merge 함수는 placeholder 확장 방식과 position 계산이 다르므로 소스 기록의 revision에서 확인한다.

position과 attention 좌표.

2D/3D RoPE를 쓰는 vision tower와 1D text position을 쓰는 decoder는 좌표계를 공유하지 않을 수 있다. projector 뒤 feature가 decoder position 17~272를 차지한다면 이후 text position이 밀린다. 일부 모델은 modality별 position tuple을 쓴다. cache·packing·resume에서 position tensor shape와 생성 규칙을 artifact 계약으로 취급한다.

실제 splice 구현을 읽는 법.

대표적인 구현은 세 부류다. 첫째, 하나의 image placeholder를 `L_v`개 embedding으로 확장한다. 둘째, 전처리 단계에서 placeholder ID를 이미 `L_v`개 넣고 model이 정확히 치환한다. 셋째, text sequence와 vision sequence를 별도 stream으로 유지하며 cross-attention한다. 세 방식은 label shift, sequence length, KV cache와 packing이 다르다.

확장 방식에서는 원 text index `i` 뒤의 모든 token 위치가 `L_v-1`만큼 밀린다. position과 labels를 함께 이동해야 한다. 치환 방식에서는 placeholder count와 feature count가 exact match여야 한다. cross-attention 방식에서는 decoder self-attention length는 text만 포함하지만 별도 encoder mask와 KV가 생긴다. “multimodal token 256개”라는 숫자만으로 memory를 비교할 수 없다.

Transformers의 고정 revision에서 model별 `get_image_features`, feature selection, masked scatter/merge 경로를 읽는다. config의 vision feature layer와 selection strategy가 class token을 포함하는지 확인한다. upstream test가 image placeholder count mismatch를 예외로 만드는지, batch별 image 수가 다를 때 padding을 검증하는지 본다. test가 한 image 한 prompt만 다루면 multi-image packing까지 증명한다고 쓰지 않는다.

projector의 backward와 병목.

선형 projector라면 `Y=XWᵀ+b`에서 `∂L/∂W=(∂L/∂Y)ᵀX`다. MLP projector는 activation과 normalization이 추가된다. projector 입력 norm이 modality/해상도에 따라 크게 달라지면 decoder residual scale과 충돌할 수 있다. feature norm, projector output norm, 첫 decoder block 뒤 norm을 비교한다.

vision tower와 decoder를 모두 학습하면 서로 다른 LR/weight decay parameter group을 쓰는 경우가 많다. group manifest는 module prefix 추측이 아니라 실제 parameter identity와 trainable flag를 나열한다. 누락된 projector가 default optimizer group으로 들어가거나 frozen tower가 optimizer state를 차지하는지 검사한다.

backward hook에서 tower 마지막 layer, projector, decoder 첫 layer의 gradient norm을 기록한다. projector gradient는 있는데 tower가 0이면 freeze policy와 맞는지 확인한다. decoder는 학습되는데 projector가 0이면 feature detach, label mask, modality branch를 의심한다. gradient norm 하나가 작다는 이유로 고장이라 하지 않고 작은 overfit fixture에서 loss와 grounding이 실제 변하는지 본다.

## 21.3 multimodal loss·mixture·resume를 같은 상태로 관리한다

### 21.3.1 sample·token·loss 질량을 분리한다

이미지-텍스트 50%, 순수 텍스트 50%로 표본을 뽑아도 이미지 표본의 visual sequence가 길면 실제 attention FLOP과 token 비율은 50:50이 아니다. manifest에는 configured sample weight, realized text/vision/audio token, 유효 label, GPU seconds를 함께 남긴다. curriculum이 resolution이나 frame 수를 올리는 step은 scheduler state이므로 checkpoint에 포함한다.

분산 batching과 padding 낭비.

길이와 해상도로 bucket을 나누면 padding은 줄지만 shard마다 modality 분포가 달라질 수 있다. 모든 rank가 같은 수의 optimizer contribution을 만드는지, loss denominator를 global reduce하는지 확인한다. 한 rank의 빈 audio batch를 0 loss로 처리하면서 분모에는 넣으면 조용한 bias가 생긴다.

### 21.3.2 curriculum을 phase 상태 기계로 만든다

curriculum을 `phase0:text`, `phase1:low-res image`, `phase2:video/audio`로 나누면 phase ID, 전환 token 수, resolution/frame/audio-rate config가 checkpoint state다. resume 시 global step만 보고 phase를 재계산하면 skipped sample이나 world-size 변경으로 realized token이 달라져 drift할 수 있다. 전환 조건이 optimizer step인지 consumed token인지 명시한다.

mixture 결정 트리.

text 품질만 떨어지면 realized modality token과 gradient norm을 본다. vision grounding만 정체면 projector gradient와 image-text alignment를 본다. 모든 modality가 느려지면 padding/FLOP 증가와 LR schedule의 token clock을 본다. sample weight를 조절하기 전에 어떤 분모가 변했는지 확인한다.

modality별 계산량을 회계한다.

self-attention 비용은 단순화하면 sequence 길이 제곱에 비례한다. text `L_t`와 visual `L_v`를 concat하면 attention score는 `(L_t+L_v)²` 규모다. image resolution을 두 배로 하면 patch 수는 네 배, 해당 부분의 score 요소는 열여섯 배까지 늘 수 있다. FlashAttention이 score matrix를 materialize하지 않아도 FLOP이 사라지는 것은 아니다.

cross-attention 구조의 계산량은 text self-attention의 `L_t²`와 text-query/vision-key의 `L_tL_v`로 나뉩다. 어느 쪽이 싼지는 layer 수와 cross-attention 빈도에 달렸다. curriculum의 resolution/frame 변경 전에 예상 FLOP, activation bytes, valid target을 계산하고 실제 profiler와 비교한다.

GPU seconds당 loss-bearing token과 modality sample을 함께 본다. 고해상도 표본이 batch를 지배하면 configured mixture는 같아도 update frequency와 queue wait가 바뀐다. dynamic batching이 shape가 비슷한 표본을 묶을 때 domain order까지 상관되는지 확인한다.

분산 resume에서의 sample identity.

media decode는 worker 수와 prefetch에 민감하다. checkpoint가 dataset index만 저장하고 worker queue 안의 이미 decode된 표본을 저장하지 않으면 resume 뒤 일부 표본이 반복되거나 건너뛸 수 있다. exact resume가 필요하면 global sample order와 consumed boundary를 durable하게 만들고 prefetch 결과를 commit 전에 소비된 것으로 세지 않는다.

video decoder가 seek 후 keyframe부터 복호화해 target PTS에 도달하는 과정도 library/version에 따라 달라질 수 있다. deterministic resume test는 `MediaSampleID`뿐 아니라 sampled frame PTS와 pixel/feature checksum을 비교한다. JPEG/audio decode의 하위 bit 차이를 허용한다면 어느 단계부터 tolerance를 적용할지 사전에 정한다.

DP rank별 modality count와 global denominator를 비교한다. 한 rank가 corrupt media를 skip하고 다른 rank는 정상 batch를 처리하면 collective shape 또는 contribution 수가 달라질 수 있다. skip을 local control flow로 두지 말고 모든 rank가 공유하는 replacement/invalid-sample accounting을 사용한다.

resume와 고장 진단.

### 21.3.3 AV sync와 deterministic resume를 검증한다

영상-음성 표본은 container timestamp, decoder seek, augmentation seed가 함께 복원돼야 한다. checkpoint 뒤 첫 32개 `MediaSampleID`, sampled PTS, crop, token checksum을 uninterrupted run과 비교한다. 다르면 model RNG보다 sampler/decoder cursor를 먼저 의심한다.

현장 체크.

loss가 내려가는데 grounding이 나쁘면 projector gradient norm, placeholder-feature count, modality별 label count를 본다. audio가 반복되면 RVQ usage와 frame hop을, 영상 lip-sync가 깨지면 PTS 차이 분포를 본다. 공개 코드에서 확인한 계약과 실제 장비 실행을 구분한다. 이 책의 golden lab에서 로컬 실행 전 항목은 `실행 예정`으로 표시한다.

upstream test와 독자 실험.

고정 source의 processor/model test가 검증하는 것은 주로 shape, placeholder 수, forward output이다. 동일 media bytes가 decoder backend·worker 수·resume 뒤 같은 feature checksum을 만드는지는 별도 실험이다. 독자는 3초 synthetic audio와 timestamp가 그려진 8-frame video를 만들어 crop/PTS/token atlas를 손으로 검산한다.

실패 판정.

재생성 feature checksum이 다르면 먼저 decoder/library nondeterminism과 augmentation RNG를 분리한다. checksum은 같은데 logits가 다르면 projector/model dtype과 kernel, logits는 같은데 text가 다르면 sampling/stop을 본다. 최초 divergence가 없는 상태에서 최종 caption만 비교하지 않는다.

네 가지 실패 주입 실험.

첫 실험은 placeholder mismatch다. image 두 장을 주고 placeholder를 하나만 넣어 silent reuse가 아니라 명시적 실패가 나는지 본다. 두 번째는 corrupt frame/audio다. local worker만 skip할 때 rank들이 같은 batch cardinality를 유지하는지 본다. 세 번째는 resume이다. prefetch가 가득 찬 시점에 checkpoint하고 재개해 다음 64개 media/PTS ID를 비교한다. 네 번째는 dead codebook이다. 일부 entry가 선택되지 않도록 편향된 작은 dataset을 주고 usage alert와 recovery policy가 작동하는지 본다.

각 실험에는 예상 metric과 기각 조건을 명시한다. placeholder 실험에서 예외가 없어도 label/position atlas가 정확하면 설계상 허용일 수 있다. corrupt media가 대체 표본을 쓰면 replacement ID가 기록돼야 한다. resume checksum 차이가 augmentation seed 변경으로 설명되면 sample-exact는 실패지만 statistical resume라는 더 낮은 등급은 가능하다.

checkpoint inventory.

공동 학습 시 vision/audio tokenizer, projector, decoder의 weight와 각 optimizer state를 저장한다. curriculum phase, modality sampler, media augmentation RNG, codebook EMA, loss scaler와 scheduler clock이 필요하다. processor config와 external codec/library는 checkpoint tensor가 아니지만 artifact manifest에 고정한다.

adapter만 학습한 multimodal model은 projector/vision adapter가 export에 포함되는지 확인한다. language adapter만 저장하고 새 projector를 누락하면 load는 성공해도 grounding이 깨질 수 있다. merged artifact와 서빙 실행 환경에서 같은 media bytes→feature→first logits를 비교한다.

독자용 관측 trace.

한 표본의 trace는 `decode 12ms→resize/crop 3ms→vision 18ms→projector 1ms→decoder 31ms→backward 72ms`처럼 경계를 나눈다. 시간 수치 자체보다 tensor identity를 잇는 것이 목적이다. 각 span에 media ID, feature shape, valid label, device/stream을 붙인다. dataloader prefetch와 GPU compute overlap이 보이도록 host와 device timestamp를 같이 둔다.

vision 시간이 길어졌을 때 input resolution/patch count가 같은지 먼저 본다. 같다면 kernel/dtype/clock, 다르면 workload drift다. decoder 시간이 길어졌다면 splice 후 sequence length와 padding을 본다. end-to-end 평균만 보면 둘을 구분할 수 없다.

## 21.4 한 표본을 processor에서 loss까지 종단 추적한다

가상의 336×336 RGB 이미지와 “표지판의 숫자는?”이라는 질문을 사용하자. processor가 336을 유지하고 patch 14를 쓰면 24×24=576 patch가 생긴다. class token을 버리는 feature selection이라면 projector 입력은 `[1,576,D_v]`다. 질문 template 안 image placeholder 하나가 576개 embedding으로 확장되고 text가 12 token이라면 decoder 입력 길이는 대략 588이다. 실제 special token을 포함한 값은 atlas에서 확인한다.

label은 assistant 답 “42”와 EOS 위치만 유효하다고 하자. visual 576개와 prompt token은 attention context이지만 CE 분모에는 들어가지 않는다. `M=3`인데 전체 길이 591로 loss를 나누면 gradient scale이 약 197배 달라진다. framework가 token mean을 이미 반환한 뒤 accumulation에서 다시 sequence length로 나누는 이중 reduction도 검사한다.

forward atlas는 pixel checksum, vision patch output, selected feature, projector output, merged embedding, first decoder block, logits를 잇는다. backward atlas는 LM head, decoder first/last block, projector, vision last block의 gradient를 잇는다. vision freeze라면 마지막 값이 없어야 하고 projector는 있어야 한다. 이 한 사례가 processor와 loss를 동시에 검산한다.

### 21.4.1 음성 표본을 waveform에서 decoder loss까지 추적한다

16kHz 3초 waveform은 48,000 sample이다. 25ms window와 10ms hop을 쓰면 boundary/padding 정책에 따라 약 300 frame이 된다. log-Mel `[80,L]`을 encoder가 subsample하면 decoder가 보는 audio token은 더 짧다. “3초니까 300 token”이라고 단정하지 않고 feature extractor와 encoder stride를 각각 계산한다.

transcription target이 8 token이면 audio frame은 context이고 8개 target이 loss 분모다. codec language model처럼 RVQ code를 예측한다면 `[Q,L]`의 어느 축을 flatten하고 codebook별 vocabulary offset을 쓰는지가 label contract다. codebook 0만 학습되고 뒤 codebook gradient가 0인지 histogram으로 확인한다.

시간 이동 augmentation 후 transcript alignment가 유지되는지 synthetic click와 timestamp token으로 시험한다. resume 전후 waveform chunk offset, augmentation seed, feature checksum을 비교한다. final WER만 같아도 sample window가 다르면 sample-exact는 아니다.

### 21.4.2 영상 표본을 timestamp에서 token까지 추적한다

10초 29.97fps 영상에서 8 frame을 uniform sample한다고 하자. frame index를 `round(linspace(0,N-1,8))`로 고르는 구현과 timestamp로 고르는 구현은 variable frame rate에서 다르다. 실제 PTS와 decoded pixel hash를 manifest에 둔다. tubelet이 2 frame씩 묶이면 temporal token 수는 4이고 spatial patch 수와 곱해진다.

audio-video 모델은 각 video token이 가리키는 시간 구간과 audio frame 구간을 연결한다. padding된 마지막 tubelet과 audio tail을 mask한다. cross-modal contrastive loss가 있으면 positive pair ID와 distributed negative pool의 owner를 저장한다. all-gather된 embedding으로 negative를 만들 때 다른 rank의 gradient가 통과하는지 detach 여부를 읽는다.

network decode stall이 GPU idle로 나타나는지, 긴 video가 batch padding을 키우는지 trace한다. input queue depth, decode latency, H2D bytes, vision/video kernel과 decoder 시간을 함께 본다.

### 21.4.3 test pyramid를 raw asset부터 update까지 쌓는다

가장 아래에는 processor pure-function test가 있다. 고정 bytes가 예상 crop/PTS/token shape를 내는지 검사한다. 다음은 model unit test로 placeholder와 feature count, mask, logits shape를 검사한다. 그 위에는 backward test로 projector/tower/decoder gradient 정책을 검사한다. 그 위에는 checkpoint roundtrip과 worker/rank resume test가 있다. 마지막은 작은 overfit과 modality-specific eval이다.

아래 test가 통과하지 않으면 위 점수는 해석하지 않는다. 반대로 위 품질 test 하나가 processor의 모든 edge case를 증명하지 않는다. test 이름을 나열하지 말고 어떤 invariant와 failure branch를 검증하는지 표에 기록한다.

corrupt media, zero-length audio, multiple images, odd resolution, missing placeholder, all-ignore label, dead codebook, variable frame rate를 최소 fixture로 둔다. production codec 전체를 재현하지 못하면 coverage 한계로 남긴다.

옵션 변경의 상태 diff.

`image_size`는 patch count, FLOP, merged sequence와 memory를 바꾼다. `vision_feature_layer`는 projector 입력 의미와 shape를 바꿀 수 있다. `freeze_vision_tower`는 parameter owner와 optimizer/checkpoint state를 바꾼다. `num_frames/FPS`는 PTS manifest와 token 수를 바꾼다. `audio_hop_length`는 frame rate와 alignment를 바꾼다. `codebook_count`는 bitrate, target shape와 embedding table을 바꾼다.

옵션마다 config before/after만 저장하지 않고 실제 tensor/state diff를 만든다. 예상 patch count와 관측 count, optimizer parameter set, checkpoint key set, valid target 수, peak memory를 비교한다. 효과가 없으면 unsupported/no-op branch 또는 cached preprocessing을 의심한다.

장의 최종 판정.

이 장이 닫혔다고 말하려면 같은 media bytes가 고정 processor에서 같은 feature/token identity를 만들고, placeholder/position/label이 맞으며, backward owner가 정책과 같고, resume 뒤 sample/PTS stream이 요구 등급으로 복구돼야 한다. modality eval이 좋다는 사실만으로 앞 계약을 대신하지 않는다.

실제 대형 model의 품질과 throughput은 공개 보고 수치와 로컬 실행을 구분한다. 이 장의 tensor 계산은 독자 fixture로 실행할 수 있지만 production cluster의 decoder nondeterminism과 codec version 차이는 별도 검증 대상이다.

잘못된 설명을 걸러내는 질문.

“이미지를 token으로 바꾼다”는 문장을 만나면 연속 feature인지 discrete code인지 묻는다. “해상도를 높였다”면 resize/crop 후 grid와 attention FLOP을 묻는다. “audio token”이면 log-Mel frame, encoder latent, RVQ index 중 무엇인지 묻는다. “영상 32 frame”이면 원 PTS와 sampling policy를 묻는다. 용어보다 tensor identity가 먼저다.

“vision tower를 freeze해 memory가 줄었다”면 parameter gradient, optimizer state, activation 중 무엇이 줄었는지 묻는다. “multimodal loss”면 target 위치와 denominator, modality별 weight를 묻는다. “deterministic”이면 media decode, augmentation, model kernel, sampling 중 어디까지인지 묻는다. 답이 없으면 주장을 좁힌다.

운영에서 자주 생기는 조용한 오류.

processor cache key에 crop/codec revision이 빠져 낡은 feature가 재사용될 수 있다. placeholder count는 맞지만 image order가 뒤바뀔 수 있다. multi-image prompt에서 batch flatten/unflatten index가 달라질 수 있다. all-ignore sample이 분모만 늘릴 수 있다. corrupt media replacement가 다른 rank에서만 일어날 수 있다. checkpoint에는 model이 있지만 projector가 export에서 누락될 수 있다.

이 오류들은 exception보다 위험하다. output이 나오고 평균 loss도 내려갈 수 있기 때문이다. identity, order, denominator, owner, descendant artifact를 각각 assert하는 이유다.

다음 장과의 접점.

22장은 이미지 pixel 자체가 아니라 21장이 만든 latent 또는 discrete code를 `x_0`로 받는다. 따라서 VAE/tokenizer normalization, latent scale, shape, codebook vocabulary와 mask가 noise process의 입력 계약이다. 이 값이 달라지면 같은 diffusion scheduler 이름도 다른 확률 경로를 학습한다.

`MediaSampleID→RepresentationID→NoiseTrajectoryID`를 한 edge로 묶는다. 원 media와 reconstruction fixture를 보존해 diffusion 결과의 문제가 representation 손실인지 denoiser/solver 문제인지 분리한다. 이 handoff가 없으면 22장의 품질 평가가 tokenizer 오류를 model 오류로 흡수한다.

마지막으로 handoff manifest를 실제로 읽어 검산한다. image에는 original/pixel/patch/projector checksum, audio에는 waveform/log-Mel 또는 RVQ checksum, video에는 PTS/frame/tubelet checksum을 기록한다. 모든 modality에는 normalization, valid mask, position, target count와 processor revision을 기록한다. shape만 있고 bytes identity가 없거나, checksum만 있고 생성 config가 없으면 재생성 계약은 미완성이다.

독자는 장을 마치기 전에 synthetic 표본 하나를 두 번 처리하고 worker 수를 바꿔 세 번째 처리한다. deterministic으로 선언한 경계까지 checksum이 같은지 확인하고, 다른 경계는 tolerance와 원인을 기록한다. projector 한 step 뒤 gradient owner와 checkpoint key를 대조한다. 이 세 검사가 통과해야 다음 장의 noise trajectory가 안정된 입력 위에서 출발한다.

검사 결과에는 성공만 쓰지 않는다. 미지원 media 형식, 비결정적 decoder, 검증하지 못한 multi-node preprocessing과 production codec 차이를 구체적인 제한 사항과 후속 실행 과제로 남긴다. 이 목록은 실패가 아니라 주장의 경계다. 경계를 숨긴 높은 점수보다 재현 가능한 작은 fixture가 이후 디버깅에 더 큰 가치가 있다.

## 21.5 patch·frame·projector gradient를 손으로 검산한다

입력이 336×336이고 patch가 14×14라면 padding/crop 뒤 patch grid는 24×24, 즉 576개다. CLS token을 쓰면 577개가 될 수 있고, projector가 2×2 spatial merge를 하면 144개로 줄 수 있다. “image token 576”이라는 설명은 processor crop, vision tower 출력과 merge branch를 함께 고정해야 한다.

두 이미지 prompt에서는 placeholder 하나가 144 token span 두 개로 확장되는지, image order가 batch flatten/unflatten 뒤 유지되는지 확인한다. text token 80과 image token 288이면 attention sequence는 architecture의 insertion rule에 따라 368 부근이지만 BOS·separator와 newline token이 추가될 수 있다. 실제 token/feature ledger가 기준이다.

negative control은 두 이미지 feature 순서를 바꾸되 shape와 count는 그대로 유지한다. shape assertion만 있다면 통과하지만 image-ID→feature-span checksum은 실패해야 한다. 이 fixture가 multi-image grounding의 조용한 순서 오류를 잡는다.

### 21.5.1 audio frame과 mask를 수치로 맞춘다

16kHz 음성 10초는 160,000 sample이다. 25ms window와 10ms hop이면 경계 처리 방식에 따라 약 998~1,001 frame이 나온다. center padding, resample과 trim이 frame 수를 바꾸므로 공식 processor source의 feature extractor symbol과 config를 고정한다.

두 utterance를 1,000 frame으로 padding했지만 유효 frame이 700과 1,000이면 loss denominator는 target 설계에 따라 1,700이어야 한다. sample별 mean을 평균하면 짧은 음성이 과가중될 수 있다. CTC, autoregressive codec와 contrastive objective는 target 축이 다르므로 이름을 분리한다.

negative control은 padding frame에 nonzero target을 넣고 loss가 변하는지 본다. mask가 맞으면 변하지 않아야 한다. resampler revision을 바꾸어 waveform checksum과 feature checksum guard가 새 RepresentationID를 만드는지도 확인한다.

### 21.5.2 video timestamp trace를 재계산한다

30fps 12초 영상에서 8 frame을 균등 sampling한다면 frame index만 저장하지 않고 원본 PTS와 decode order를 저장한다. variable-frame-rate 영상은 `index/fps`가 실제 시각이 아니다. seek가 keyframe에서 시작해 decoder revision에 따라 선택 frame이 달라질 수 있다.

각 sampled frame에는 original media digest, PTS, decode pixel checksum, crop/resize와 final tensor digest를 기록한다. tubelet이 2 frame을 묶으면 temporal token count와 padding mask를 기록한다. frame 8개가 들어왔다는 사실만으로 시간 정렬이 맞는 것은 아니다.

negative fixture는 동일 pixel을 다른 PTS에 배치한다. appearance-only checksum은 같지만 temporal manifest가 달라야 한다. 질문이 “언제 문이 열렸는가”라면 PTS 오류가 label alignment를 깨므로 final answer score 이전에 잡는다.

### 21.5.3 projector gradient의 소유권을 증명한다

vision tower를 freeze하고 projector와 language adapter만 학습한다고 하자. startup ownership 표는 vision parameter gradient 없음, projector/adapter gradient 있음, frozen base delta 0을 기대한다. 한 golden update 뒤 parameter group별 gradient norm과 delta checksum을 비교한다.

projector output 144×hidden에서 activation은 language dtype, vision output은 다른 dtype일 수 있다. cast 위치와 layer norm을 소스 좌표로 기록한다. projector gradient가 0이면 placeholder span이 attention/loss 경로에 연결됐는지, detach와 mask를 본다.

finite-difference는 작은 float64 projector와 synthetic vision feature에서 selected scalar를 검사한다. production fused tower 전체를 수치 미분하지 않는다. checkpoint save/reload 뒤 projector key가 missing/unexpected 없이 복원되고 logits가 tolerance 안인지 본다.

contrastive loss 분모.

batch에 image-text pair 4개가 있으면 4×4 similarity matrix와 대각 positive를 만든다. distributed gather를 쓰면 global batch와 duplicate sample, temperature가 objective를 바꾼다. rank별 local mean을 평균하지 않고 global row/column loss의 numerator와 denominator를 기록한다.

같은 image의 두 caption이 batch에 들어오면 하나를 false negative로 취급할 수 있다. multi-positive mask와 family ID가 필요한지 objective contract에서 결정한다. padding/invalid media pair를 matrix에서 빼면서 index mapping이 흔들리지 않는지 fixture를 둔다.

negative control은 pair order를 text 쪽만 바꿔 diagonal target이 잘못되는지 본다. 평균 similarity가 그럴듯해도 known 2×2 손계산 loss가 실패해야 한다. temperature가 learnable이면 parameter ownership과 checkpoint state를 포함한다.

reward hacking이 red-team 회귀로 이어지는 경로.

text-only row와 image row를 같은 block에 pack할 때 image feature span이 document boundary를 넘어 attention하지 않는지 architecture 계약을 확인한다. block-diagonal mask, position reset과 loss mask를 token/feature index 표로 만든다. 단순 causal mask는 앞 문서의 image를 다음 text가 볼 수 있게 할 수 있다.

contributing language token, modality token과 auxiliary loss denominator를 따로 기록한다. image token이 많다는 이유로 language loss의 sample weight가 달라지는지 확인한다. packing efficiency와 학습 의미를 함께 본다.

[sample-repeat playbook](../playbooks/03-sample-repeat.md)으로 packed row lineage와 duplicate family를 확인한다. negative fixture는 image span owner를 이웃 row로 바꾸되 block shape는 유지한다. owner assertion과 known answer가 실패해야 한다.

구현 근거와 실패 경계를 고정한다.

processor의 decode/resize/crop/normalize, model의 placeholder expansion, vision/audio encoder, projector와 loss mask를 `repository@commit:path:symbol`로 기록한다. config field가 소비되는 branch, input/output shape·dtype와 side effect를 적는다. model card의 processor 이름만으로 구현을 고정하지 않는다.

Transformers 계열이면 `AutoProcessor` 진입에서 실제 architecture-specific processor/model forward로 내려간다. remote code를 쓴다면 exact revision과 custom Python digest를 포함한다. upstream test는 shape/serialization 중 무엇을 보장하는지 구분한다.

line number만 인용하지 않고 symbol과 semantic anchor를 둔다. upgrade 시 crop default, image token expansion과 loss mask branch diff를 보고 영향 fixture를 선택한다. 실행하지 않은 codec/GPU path는 미검증으로 남긴다.

tokenizer mismatch와 modality placeholder.

special token `<image>`가 tokenizer에서 하나의 ID인지 여러 text token으로 쪼개지는지 exact fixture로 확인한다. vocabulary resize 뒤 embedding/head가 checkpoint와 호환되는지 본다. chat template가 placeholder 앞뒤 newline을 추가하면 position과 expansion 위치가 달라진다.

[tokenizer mismatch playbook](../playbooks/04-tokenizer-mismatch.md)은 raw prompt bytes, template, special-token mapping과 token IDs를 요구한다. weight 또는 projector를 의심하기 전에 이 경계를 검증한다. processor와 tokenizer revision을 bundle로 서명한다.

negative control은 동일 문자열의 tokenizer JSON만 바꾸어 special ID를 이동시킨다. startup guard가 산출물 불일치를 거부해야 한다. output이 우연히 그럴듯하다는 이유로 자동 remap하지 않는다.

corruption과 replacement 정책.

decode 실패 media를 black image나 zero audio로 대체하면 training은 계속되지만 label과 입력이 불일치한다. `drop`, `quarantine`, `deterministic replacement` 중 정책을 선언하고 count·family와 loss denominator를 기록한다. rank마다 decoder 결과가 달라지는 것을 막는다.

corrupt fixture, truncated file와 unsupported codec을 넣어 processor가 같은 failure code를 내는지 본다. retry가 mutable remote URL에서 다른 bytes를 받지 않게 original digest를 고정한다. skip으로 특정 source가 과소표집되는지도 draw ledger에서 본다.

실제 원문은 접근 제한 quarantine에 두고 공개 regression에는 synthetic corrupt media를 사용한다. processor update 뒤 과거 quarantine을 재처리하면 새 RepresentationID와 dataset revision을 만든다.

multimodal 재현 package의 필드.

패키지는 media manifest, decode/crop/timestamp, representation tensor schema/checksum, placeholder/token span, valid/loss mask, projector ownership과 golden update를 포함한다. image/audio/video numeric fixtures와 negative order/padding/corruption controls를 둔다.

독립 검토자는 336 image의 patch count, 10초 audio frame, video PTS와 multi-image order를 재계산한다. 소스 좌표와 loaded processor/model revision이 일치하는지 확인한다. checksum만 있고 생성 config가 없으면 실패다.

22장에는 verified RepresentationID와 normalization/scale, mask·position과 미지원 codec/modality를 넘긴다. 이 패키지가 있어야 diffusion/flow의 noise objective가 잘못된 representation을 학습하는 문제와 분리된다.

정렬·curriculum 장애를 사례로 읽는다.

training loss는 내려가지만 “두 번째 이미지의 색” 질문에서 첫 이미지를 답한다고 하자. multi-image count와 tensor shape는 정상이다. trace에서 placeholder span owner, flattened feature order와 batch unflatten index를 비교한다. image 0/1 checksum이 prompt placeholder 1/0에 연결됐다면 순서 오류다.

단일 이미지 evaluation이 높았던 이유는 order dimension이 없었기 때문이다. 두 이미지의 pixel을 교환한 paired fixture, 같은 pixel과 다른 caption의 negative control을 추가한다. projector weight를 다시 학습하기 전에 preprocessing/collator 소스 분기를 고친다.

fix 뒤 golden token/feature checksum, answer와 gradient를 재검증하고 cache key에 image order manifest가 들어가는지 본다. stale feature cache를 지우는 것만으로 끝내지 않고 key regression을 만든다. 기존 checkpoint가 잘못된 alignment를 학습했으므로 data-only fix 뒤 retrain 또는 영향 평가가 필요하다.

audio-text alignment 사건.

ASR transcript는 맞지만 audio question answering이 긴 clip 끝부분을 무시한다고 하자. valid frame mask, feature truncation과 label span을 본다. processor max frames에서 tail이 잘렸는데 transcript는 full text를 사용했다면 input-target 불일치다.

10초/30초/경계 길이 fixture에서 waveform duration, frame count, truncation side와 answer event timestamp를 기록한다. tail event가 잘린 row는 학습에서 거부하거나 windowing/segment label로 바꾼다. padding frame과 잘린 frame을 같은 ignore reason으로 합치지 않는다.

fix 뒤 frame denominator, projector/encoder gradient와 length-bucket performance를 본다. 짧은 clip 평균만 좋아졌다는 이유로 승인하지 않는다. resample/codec path와 실제 production audio가 같은지 지원 표에 둔다.

modality dropout과 curriculum.

text-only, image, audio row를 섞을 때 batch마다 modality contribution을 기록한다. curriculum이 image 70%에서 audio 30%로 바뀌면 global loss 변화가 model 회귀인지 mixture 변화인지 분리한다. modality별 numerator/denominator와 sampled family ledger가 필요하다.

modality dropout은 missing media robustness를 위한 augmentation일 수 있지만 placeholder, mask와 target을 일관되게 바꿔야 한다. image를 drop하고 질문/answer는 image 의존인 채 두면 contradictory row다. drop RNG와 실제 count를 checkpoint한다.

negative control은 media feature만 zero로 만들고 placeholder/target을 유지한다. validation이 이를 정책상 dropout 또는 invalid row로 명확히 분류해야 한다. silent zero feature를 정상 image로 세지 않는다.

multimodal OOM 수치 분석.

text sequence 512와 image token 576을 결합하면 attention length가 약 1,088이 되어 quadratic attention element가 text-only 대비 대략 `(1088/512)^2≈4.5`배다. 실제 memory는 head, kernel, checkpoint와 fused path에 따라 다르지만 증가 방향을 예측할 수 있다.

두 image면 token merge가 없다면 1,664로 더 커진다. OOM에서 batch만 줄이기 전에 crop/patch/merge와 placeholder count를 확인한다. [OOM playbook](../playbooks/05-oom.md)에 modality token, sequence bucket과 peak range를 넘긴다.

token merge를 켜 memory가 줄면 representation과 quality가 바뀐 새 experiment다. same checkpoint에서 cache/token schema가 호환되는지 확인하고 image detail/grounding evaluation을 다시 실행한다.

지원 범위와 negative matrix.

행에는 image codec/size, audio rate/duration, video codec/fps, processor/model revision과 single/multi-image/modal combination을 둔다. 열에는 decode checksum, token/frame count, order/mask, gradient, checkpoint/export와 production parity를 둔다.

negative matrix는 image swap, audio padded target, video PTS shift, wrong placeholder token, corrupt media, cache-key omission과 missing projector를 포함한다. 각 fault의 expected first assertion을 적는다. 최종 task score만 실패 조건으로 두지 않는다.

독립 검토자가 image/audio/video 손계산과 negative fixture를 재실행하고 미지원 codec·remote decoder를 확인한다. 지원 표 밖의 media는 자동 fallback하지 않고 명시적 error/quarantine 정책을 적용한다.

독자의 최소 실습.

첫째 synthetic 28×28 image와 patch 14로 4 patch를 만들고 exact tensor 순서를 검산한다. 둘째 짧은 sine waveform으로 frame 수와 padding mask를 손계산한다. 셋째 timestamp가 명시된 color frame video로 sampling PTS를 확인한다.

세 representation을 text placeholder에 넣고 projector 한 update의 gradient owner와 checkpoint reload를 본다. image order, audio pad target과 video PTS를 하나씩 깨 negative assertion을 확인한다. 실제 대형 model이 없어도 processor/collator 계약은 검증할 수 있다.

보고에는 소스 리비전, manifest, expected/actual checksum과 미실행 GPU/model path를 적는다. 이 실습을 통과한 뒤에만 실제 private media와 장기 training을 사용한다.

배포·분산 경계를 검증한다.

processor와 projector가 한 장비에서 맞았다고 배포 경계까지 닫히지는 않는다. decoder backend, 시간축, modality별 padding과 shard 배치가 달라지면 같은 원본도 다른 token sequence가 된다. 이제 단일 표본의 표현 검증에서 나아가 오디오·영상·자막의 시간 좌표와 분산 owner를 함께 고정한다.

## 21.6 세 modality의 공간·시간 좌표를 processor 계약으로 묶는다

앞 절의 손계산은 modality마다 따로 보였지만 실제 collator에서는 한 batch 계약으로 만난다. 여기서는 “언제·어디를 관측했는가”라는 시공간 좌표를 먼저 고정한 뒤, 그 좌표가 processor shape와 placeholder 수를 결정하고, 다시 유효 loss 수와 실패 위치를 결정하는 인과 사슬로 묶는다.

### 21.6.1 frame index보다 presentation timestamp를 쓴다

영상의 100번째 frame은 고정된 시간이 아니다. variable-frame-rate 영상에서는 container time base와 PTS를 읽어야 한다. 음성은 sample index를 sample rate로 나눈 시간축을 갖는다. 자막은 별도의 시작·종료 시각을 갖는다. 세 축을 단순 index로 묶으면 decode backend를 바꾸는 순간 정렬이 달라진다.

TorchCodec 고정 commit `60db9be740ea8205a92068bce03bae9d985806c0`의 [`VideoDecoder`](https://github.com/pytorch/torchcodec/blob/60db9be740ea8205a92068bce03bae9d985806c0/src/torchcodec/decoders/_video_decoder.py)는 index 접근과 presentation-time 접근을 별도 API로 둔다. Decord commit `d2e56190286ae394032a8141885f76d5372bd44b`의 [`VideoReader`](https://github.com/dmlc/decord/blob/d2e56190286ae394032a8141885f76d5372bd44b/python/decord/video_reader.py)는 batch index와 fast/accurate seek 경로를 구분한다.

API 차이는 사소한 편의가 아니라 “어느 frame을 보았는가”라는 데이터 계보 차이다.

manifest에는 container 시작 시각, stream time base, 원 PTS, decode 뒤 frame PTS, audio sample range, subtitle interval을 남긴다. augmentation이 2.0초 clip을 뽑았다면 seed뿐 아니라 선택한 시간 구간을 기록한다. resume 때 RNG만 복원하고 decoder가 다른 keyframe에서 seek하면 같은 표본이 아니다.

### 21.6.2 AV sync를 숫자로 검산한다

48 kHz 음성에서 sample 96,000은 2초다. 29.97 fps 영상의 nominal frame 60은 정확히 2초가 아닐 수 있다. frame PTS가 60,060이고 time base가 1/30,000이면 2.002초다. 허용 오차를 한 frame이라고 선언할지 20 ms라고 선언할지는 task에 따라 다르다. lip-reading은 더 엄격하고 장면 설명은 느슨하다.

golden fixture에는 clap이나 입술 닫힘처럼 관측 가능한 sync event를 넣는다. decode→crop→resample→batch collate 뒤에도 audio peak와 video event의 offset이 범위 안인지 검사한다. 평균 offset만 보면 일부 표본의 큰 drift가 상쇄되므로 p50/p95/max와 duration별 slope를 본다.

### 21.6.3 worker별 media backend를 동일하게 고정한다

node image에 설치된 ffmpeg codec, hardware decode 여부, color conversion이 다르면 같은 asset hash에서도 tensor가 달라질 수 있다. decode library revision, codec build, pixel format, resample filter를 run manifest에 넣는다. golden asset의 decoded tensor hash는 완전 일치 또는 명시한 tolerance로 비교한다.

고장 주입은 한 worker만 다른 ffmpeg build를 쓰게 하는 것이다. checksum gate가 admission 전에 잡아야 한다. 학습 loss가 이상해질 때까지 기다리면 원인을 projector나 optimizer에서 찾게 된다.

## 21.7 vision patch·grid·placeholder 예산을 역산한다

### 21.7.1 patch와 merge가 sequence length를 결정한다

높이 (H), 너비 (W), patch 크기 (P)라면 기본 patch 수는 (lceil H/P\rceil\lceil W/P\rceil)다. spatial merge가 (m\times m)이면 decoder에 들어갈 시각 token은 대략 그 수를 (m^2)로 나눈다. 영상은 여기에 temporal patch 또는 sampled frame 수가 곱해진다. 이미지 장수만으로 batch 비용을 예측할 수 없는 이유다.

Transformers commit `550d7b3834670483a4df436541272c055dc364bf`의 [`Qwen2VLImageProcessor`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)는 flattened patch와 `image_grid_thw`를 함께 만든다.

같은 revision의 [`Qwen2.5-Omni processor`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py)는 grid와 FPS를 이용해 modality placeholder 길이와 temporal index를 만든다. grid metadata는 로깅 장식이 아니라 embedding insertion의 cardinality 계약이다.

예를 들어 448×448, patch 14, merge 2이면 32×32 patch가 256 token으로 줄어든다. 같은 이미지를 896×896으로 올리면 1,024 token이다. pixel 수 네 배가 attention sequence도 네 배로 만들고, self-attention 일부 비용은 제곱으로 뛴다. curriculum에서 해상도를 올릴 때 learning rate만 아니라 token budget과 batch composition을 다시 계산해야 한다.

### 21.7.2 dynamic resolution은 sampling 정책이다

긴 변을 고정 resize하면 작은 글자를 잃고, 원 해상도를 유지하면 문서 이미지가 batch를 독점한다. min/max pixels, aspect-ratio bucket, tile 수를 정책으로 둔다. “이미지 30%”라는 mixture weight보다 실제 vision token 비율을 계측한다.

표본마다 `raw_pixels`, `resized_pixels`, `grid_thw`, `vision_tokens`, `text_tokens`, `truncated`를 남긴다. loss를 modality별로 집계할 때 sample 평균과 supervised-token 평균을 모두 본다. 고해상도 한 장이 수백 저해상도 표본만큼 gradient unit을 가질 수 있다.

### 21.7.3 placeholder와 feature cardinality를 fail closed로 검사한다

LLaVA commit `c121f0432da27facab705978f83c4ada465e46fd`의 [`llava_arch.py`](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/model/llava_arch.py)는 vision feature를 image placeholder 위치에서 text embedding 사이에 넣고 label mask를 재작성한다.

Qwen2.5-Omni의 [`modular_qwen2_5_omni.py`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py)는 modality feature를 mask 위치에 `masked_scatter`한다.

feature 수와 placeholder 수가 다르면 truncate나 broadcast로 넘어가면 안 된다. sample ID, grid, placeholder span, feature shape를 출력하고 그 batch를 격리한다. shape만 맞고 순서가 틀린 경우를 잡기 위해 각 이미지에 알려진 색상/패턴을 넣은 two-image fixture로 insertion 순서를 검사한다.

## 21.8 log-Mel·RVQ·audio loss의 서로 다른 시계를 읽는다

### 21.8.1 log-Mel은 연속 feature다

Whisper commit `5f86d1d86363843179951550570367b37c5d6f78`의 [`audio.py`](https://github.com/openai/whisper/blob/5f86d1d86363843179951550570367b37c5d6f78/whisper/audio.py)는 ffmpeg decode, mono PCM resample, pad/trim, STFT power, Mel projection, log clamp와 normalization을 순서대로 수행한다. 여기서 나온 값은 연속 feature tensor다. vocabulary lookup을 거치지 않고 audio encoder로 들어간다.

window, hop length, Mel bin, clamp floor가 바뀌면 time resolution과 dynamic range가 바뀐다. 같은 waveform이라도 다른 학습 입력이다. preprocessing을 “Whisper 방식”이라는 이름으로만 저장하지 말고 parameter와 code revision을 저장한다.

CTC와 encoder-decoder loss도 다르다. CTC는 가능한 alignment 경로를 주변화하고 input length와 target length 계약이 엄격하다. seq2seq는 decoder token 위치에서 CE를 계산한다. 같은 음성 데이터라도 blank token, timestamp token, forced prefix, padding mask가 loss-bearing unit을 바꾼다.

### 21.8.2 RVQ code에는 시간·codebook 좌표가 있다

EnCodec commit `0e2d0aed29362c8e8f52494baf3e6f99056b214f`의 [`model.py`](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py)는 encoder ratio로 frame rate를 구하고 bandwidth에 맞춰 RVQ code 수를 선택한다. code tensor `[B,K,T]`에서 (K)는 시간축이 아니라 residual codebook 축이다.

첫 codebook은 큰 구조를, 뒤 codebook은 앞의 양자화 잔차를 근사한다. (z\approx e_{1,k_1}+\cdots+e_{K,k_K})이므로 codebook을 독립 categorical channel처럼 취급해도 decode에서는 합성된다. bandwidth dropout은 단순 augmentation이 아니라 뒤 codebook supervision 빈도를 바꾼다.

같은 repository의 [`core_vq.py`](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/quantization/core_vq.py)는 EMA cluster size가 작은 entry를 sampled vector로 만료·교체한다. dead-code replacement가 rank마다 독립적으로 일어나면 code ID 의미가 갈라질 수 있다. codebook state와 EMA count도 checkpoint 대상이다.

### 21.8.3 음성 objective의 분모를 분리한다

waveform reconstruction, multi-scale STFT, adversarial, feature matching, code prediction CE가 함께 쓰일 수 있다. 각 loss는 단위와 분산이 다르다. config의 lambda만 기록하지 말고 batch별 raw loss, weighted loss, shared parameter gradient norm을 기록한다. adversarial discriminator update와 generator update의 step 비율도 상태다.

고장 주입으로 한 rank의 audio length mask를 한 frame 늘린다. loss가 거의 같아도 collective 전에 gradient shape는 같으므로 조용히 통과할 수 있다. golden batch에서 유효 frame 수와 supervised unit 수를 rank별 assert해야 한다.

## 21.9 connector·interleave·cross-attention의 gradient 길을 그린다

### 21.9.1 projector는 압축과 basis 변환을 함께 한다

vision encoder feature (v\in\mathbb R^{d_v})를 language hidden (h\in\mathbb R^{d_l})로 보내는 선형 projector (h=Wv+b)는 차원만 맞추는 adapter가 아니다. decoder가 이미 학습한 embedding manifold에서 의미 있는 방향으로 feature를 회전·스케일한다. (d_v>d_l)이면 null space로 정보가 사라지고, (d_v<d_l)이면 image feature는 초기에는 낮은 차원 부분공간에 놓인다.

projector-only warmup은 decoder 좌표계를 고정한 채 이 사상을 먼저 맞추려는 선택이다. 그러나 encoder와 decoder를 완전히 동결하면 connector 용량이 부족할 수 있다. projector→상위 vision block→language block 순으로 unfreeze하는 curriculum을 비교한다. 각 phase의 trainable parameter manifest와 optimizer state를 분리한다.

### 21.9.2 interleave가 position geometry를 바꾼다

이미지 token을 text 중간에 삽입하면 뒤 text token의 position이 밀린다. multimodal RoPE는 시간·높이·너비 축을 별도로 부호화할 수 있다. placeholder 제거 후 feature insertion 순서와 position ID 생성 순서가 엇갈리면 shape는 맞지만 attention geometry가 틀어진다.

검산 fixture는 image 앞·중간·뒤 세 위치에 같은 질문을 놓고 position ID와 causal mask를 덤프한다. label shift 뒤 image 위치와 padding 위치가 `-100`인지, 첫 answer token이 supervision에 들어가는지 확인한다. Qwen2.5-Omni의 같은 `modular_qwen2_5_omni.py`에서 embedding scatter 뒤 language logits와 labels로 loss가 계산되는 구간을 한 호출 사슬로 읽는다.

### 21.9.3 collapse를 feature·gradient에서 조기에 찾는다

connector가 평균적인 language embedding 하나만 내면 loss가 잠시 내려갈 수 있다. modality token의 covariance spectrum, sample 간 cosine distance, gradient norm을 본다. image를 shuffle했는데 loss가 거의 같다면 decoder가 이미 text shortcut만 쓰는 중이다.

반증 실험은 동일 text에 다른 image를 붙인 contrast batch와 동일 image에 다른 question을 붙인 batch다. logits sensitivity와 answer flip을 측정한다. representation probing은 인과 증거가 아니므로 image ablation과 feature permutation을 함께 수행한다.

mixture·curriculum·resume를 하나의 상태로 저장한다.

설정 비율과 실현 비율을 구분한다.

dataset sampler가 image 40%, audio 30%, video 30%를 뽑아도 decode 실패, length filter, gradient accumulation 뒤 실제 loss unit 비율은 다르다. sample count, modality token, supervised token, optimizer step 기여를 각각 집계한다. distributed sampler에서는 rank별 편차도 본다.

curriculum state는 phase 이름만이 아니다. dataset weight, resolution/FPS/duration cap, augmentation strength, trainable module, loss lambda, optimizer group, RNG cursor를 포함한다. phase 전환 직전과 직후 golden batch를 실행해 token count와 gradient owner 변화가 의도와 맞는지 본다.

exact resume의 최소 상태.

sampler cursor, worker seed, asset decode parameters, selected time interval, crop, masking, codebook dropout, dynamic batching queue, accumulation microstep, optimizer/scheduler/scaler, trainable scope가 필요하다. data loader prefetch queue까지 byte-exact resume하기 어렵다면 sample-exact 보장 범위를 명시하고 first-divergence를 측정한다.

resume 테스트는 N step 연속 run과 K step 저장→재시작→N-K step run을 비교한다. sample ID, transform trace, loss-unit count, logits checksum, gradient norm, parameter checksum 순서로 최초 불일치를 찾는다. 최종 checkpoint만 비교하면 원인 위치를 잃는다.

22장으로 넘기는 계약.

이 장이 넘기는 것은 단순 embedding tensor가 아니다. modality별 원 시간/공간 좌표, tokenizer/encoder revision, connector와 interleave 위치, mask, loss 분모, mixture 실현값을 포함한 `MultimodalBatchID`다. 22장의 diffusion은 이 표현을 condition 또는 생성 상태로 사용한다. condition이 무엇인지 고정하지 않으면 scheduler 비교도 의미가 없다.

최종 인수 조건은 여섯 가지다. 원 asset에서 loss-bearing unit까지 역추적할 수 있어야 한다. placeholder와 feature cardinality가 fail closed여야 한다. audio/video sync fixture가 decode backend 변경을 잡아야 한다. modality shuffle이 성능을 떨어뜨려야 한다. resume 첫 step의 표본과 transform이 같아야 한다. config weight와 실현 gradient mixture를 모두 보고해야 한다.

멀티모달 loss를 한 개의 숫자로 압축하지 않는다.

supervision topology를 먼저 그린다.

멀티모달 모델이라고 모든 modality가 직접 loss를 받는 것은 아니다. image encoder와 projector가 language CE를 통해 간접 gradient만 받을 수 있고, contrastive loss나 reconstruction loss가 별도로 붙을 수 있다. speech model은 CTC와 decoder CE, codec reconstruction을 동시에 가질 수 있다. 각 loss에서 어느 parameter까지 gradient가 흐르는지 표로 만든다.

```text
loss                 direct target        gradient owner
language CE          answer token         LM + connector + encoder(해제 시)
contrastive          paired sample        projection heads + encoders
codec reconstruction waveform/spectrum    decoder + quantizer + encoder
commitment           latent/codebook      encoder 또는 codebook
```

loss 이름이 같아도 mask 분모가 다르면 mixture가 달라진다. language CE는 answer token 수, contrastive loss는 valid pair 수, reconstruction은 audio sample이나 STFT bin 수로 정규화될 수 있다. batch별 `raw_sum`, `denominator`, `reduced_loss`, `lambda`, `weighted_loss`를 기록한다.

gradient route를 hook으로 확인한다.

작은 golden batch에서 encoder, connector, 첫 language block, LM head의 gradient norm을 저장한다. image를 제거했을 때 connector gradient가 0이 되는지, text-only row가 vision encoder를 건드리지 않는지 확인한다. gradient checkpointing이나 frozen module 때문에 `grad is None`인 경우와 실제 0 tensor를 구분한다.

모달리티별 microbatch를 따로 backward해 shared parameter gradient cosine을 구한다. 음성과 영상 gradient가 계속 음수라면 sampling weight만 조절해서 해결되지 않을 수 있다. adapter 분리, alternating update, gradient projection을 실험하되 throughput과 generalization trade-off를 같이 본다.

contrastive batch의 global negatives를 검산한다.

분산 contrastive 학습은 다른 rank embedding을 gather해 negative로 쓸 수 있다. gather가 autograd를 보존하는지, local positive index가 global offset과 맞는지 확인한다. 마지막 uneven batch나 filtered sample 때문에 rank별 batch size가 다르면 positive label이 틀어질 수 있다.

두 rank, 각 두 sample의 작은 fixture에서 similarity matrix 4×4와 target `[0,1,2,3]`를 손으로 만든다. rank 1의 target이 `[0,1]`로 남는 고장 주입을 하고 test가 잡는지 본다. temperature parameter가 learnable이면 optimizer group과 clamp 범위를 기록한다.

데이터 품질을 modality별이 아니라 사건별로 본다.

같은 사건의 중복은 cross-modal leakage다.

동일 강연의 영상, 추출 음성, 자막, 요약문이 서로 다른 데이터셋에 들어갈 수 있다. byte hash는 모두 다르다. event ID, speaker/session, source URL, 시간 구간을 연결해 split한다. train에 영상이 있고 eval에 자막만 있어도 내용 누출이다.

perceptual image hash, audio fingerprint, transcript n-gram, embedding similarity를 cascade로 쓴다. 후보를 자동 삭제하기보다 출처 계열로 묶고 사람이 경계를 검토한다. synthetic caption이 원 이미지에서 생성되었으면 caption provenance도 같은 family다.

caption 품질을 유창성으로 판정하지 않는다.

유창하지만 이미지에 없는 객체를 말하는 caption은 학습에 더 해롭다. object/attribute/relation grounding, OCR fidelity, temporal ordering을 별도 label로 둔다. captioner model과 prompt revision을 보존하고, 같은 생성기가 만든 평가 질문을 학습 caption으로 쓰지 않게 한다.

hard negative를 만들 때 한 객체만 바꾼 caption은 유용하지만 실제 이미지에 바뀐 객체도 존재할 수 있다. detector 자동 라벨만 믿지 말고 샘플 감사와 uncertainty를 둔다. relation swap과 count swap은 모델이 텍스트 shortcut을 쓰는지 드러낸다.

음성·영상 filtering은 demographic shift를 만들 수 있다.

SNR, accent confidence, face detection, motion score로 필터링하면 “깨끗한” 데이터가 특정 화자·환경에 치우친다. reject rate를 언어, 지역, 장치, duration별로 본다. 품질 threshold 전후 downstream slice 성능을 비교한다.

discarded manifest를 보존해야 threshold를 다시 평가할 수 있다. 개인정보나 권리 문제로 원 asset을 보존할 수 없다면 최소한 reject reason과 aggregate를 남긴다. 데이터 품질과 대표성은 같은 축이 아니다.

메모리와 처리량을 실제 token으로 예측한다.

collator가 OOM 분포를 만든다.

text length만 보고 batch를 만들면 큰 image grid나 긴 audio가 한 batch에 몰린다. 예상 비용을 `text_tokens + a*vision_tokens + b*audio_frames`로 근사해 bucket한다. 계수 (a,b)는 이론값보다 profiler로 fit한다. video decoder CPU memory와 pinned-memory queue도 GPU token 비용 밖에서 병목이 된다.

batch마다 max/total modality length, padding ratio, decode time, H2D time, forward/backward peak memory를 남긴다. OOM 직전 batch manifest를 저장해 exact replay한다. 자동 batch shrink가 표본을 버리거나 gradient accumulation을 바꾸지 않는지 확인한다.

variable shape와 compile cache를 연결한다.

dynamic resolution은 kernel shape와 graph compile cache를 폭발시킬 수 있다. grid bucket을 제한하면 padding은 늘지만 compile 재사용과 collective shape 안정성이 좋아진다. unique shape 수, compile time, cache eviction을 데이터 metric과 함께 본다.

한 node만 다른 bucket policy를 쓰면 collective tensor shape가 맞지 않거나 padding 분모가 달라질 수 있다. policy checksum을 rank 0에서 broadcast하고 모든 rank가 assert한다.

activation checkpoint와 frozen encoder의 trade-off.

frozen vision encoder output을 offline cache하면 GPU 계산을 줄이지만 augmentation을 고정하고 encoder fine-tuning을 막는다. cache key에 asset, crop/resize, encoder revision, dtype을 넣는다. stochastic augmentation을 cache 뒤에 적용할 수 있는지 구분한다.

encoder를 online으로 돌릴 때 activation checkpointing은 memory를 줄이고 재계산을 늘린다. frozen module은 no-grad로 activation을 저장하지 않을 수 있으므로 blanket checkpointing보다 trainable boundary에 맞춘다. profiler에서 decode·encoder·connector·LM을 stage별로 분해한다.

독자가 직접 수행할 종단 실험.

세 표본 golden batch.

첫 표본은 한 이미지와 짧은 질문, 둘째는 1초 audio와 transcript, 셋째는 두 frame video와 시간 질문으로 만든다. 작은 synthetic asset을 써서 기대 grid, frame, token, label을 손으로 알 수 있게 한다. processor output부터 loss까지 모든 shape와 mask를 snapshot한다.

실험 A는 image 두 장의 순서를 바꾸고 answer가 바뀌는지 본다. 실험 B는 audio waveform을 time shift해 sync test가 실패하는지 본다. 실험 C는 video PTS를 유지한 채 frame index만 바꿔 timestamp 기반 접근의 차이를 본다. 실험 D는 modality placeholder 하나를 제거해 fail-closed 오류를 확인한다.

학습 전 반증 질문.

이미지를 shuffle해도 metric이 유지되는가? transcript를 제거해도 audio task가 풀리는가? video의 마지막 frame만 주어도 같은 답을 내는가? 고해상도 token이 loss를 독점하는가? projector만 train할 때 representation covariance가 붕괴하는가? 이러한 반증을 통과하지 못하면 더 큰 cluster를 투입하지 않는다.

디버깅 결정 트리.

loss가 NaN이면 raw asset finite→decode tensor finite→normalization range→connector output→logits→각 loss term 순서로 간다. loss가 내려가지만 modality를 무시하면 label mask→placeholder/feature order→gradient owner→data shortcut 순서다. OOM이면 raw 해상도보다 실제 grid/token→padding→activation owner→compile shape 순서다. resume가 갈리면 sample ID→time/crop transform→feature checksum→logits→gradient로 내려간다.

최종 결과물은 golden assets, expected processor outputs, batch manifest, loss ledger, gradient route, resume trace, failure injection report다. 이 패키지가 있어야 멀티모달 학습을 “이미지를 넣었다”가 아니라 재현 가능한 계산 그래프로 설명할 수 있다.

## 21.10 LLaVA·PaliGemma·Qwen2-VL·Whisper를 같은 질문으로 비교한다

공통 tensor 계약을 세운 뒤에야 모델 이름의 차이를 읽을 수 있다. 네 계열을 processor output, media representation, fusion 위치, position·mask, loss와 trainable owner라는 같은 열에 놓고 비교한다. projector라는 이름이 같아도 placeholder splice, masked scatter, 3축 grid와 encoder-decoder cross-attention은 서로 다른 프로그램이다.

### 21.10.1 LLaVA splice와 native interleave를 구분한다

LLaVA형은 pretrained vision tower, projector, causal LM의 경계가 비교적 선명하다. connector pretraining과 instruction tuning phase를 나누어 관찰하기 쉽다. 반면 Qwen2-VL/Omni처럼 processor부터 3D position과 여러 modality가 통합된 계열은 token budget과 position geometry가 model-specific하다. “projector를 붙였다”는 공통 설명만으로는 부족하다.

두 계열을 비교할 때 vision benchmark 점수부터 보지 않는다. raw asset→processor grid→feature sequence→placeholder/interleave→position→label mask→loss의 일곱 경계를 같은 표에 넣는다. 공개 코드가 inference-only인지 labels/loss까지 있는지도 표시한다. model card의 학습 혼합 비율이 비공개라면 추정값을 사실처럼 채우지 않는다.

dual encoder와 generative decoder.

### 21.10.2 dual encoder와 generative decoder를 구분한다

dual encoder checkpoint를 vision tower로 가져오면 contrastive space가 language hidden space와 같아지는 것이 아니다. projector와 instruction data가 새 좌표계를 만든다. encoder를 너무 일찍 full fine-tune하면 retrieval representation이 무너질 수 있고, 너무 오래 동결하면 grounding이 task에 맞지 않을 수 있다. phase별 retrieval probe와 generative probe를 같이 본다.

unified discrete tokenizer의 장점과 비용.

Chameleon류처럼 image를 discrete code로 바꾸어 text token과 한 vocabulary sequence에서 학습하면 architecture가 통합된다. 하지만 image codebook 품질이 생성 상한을 정하고 vocabulary/sequence가 커진다. code ID에는 text subword와 같은 의미가 없으며 decoder가 code sequence를 image로 복원해야 한다.

### 21.10.3 discrete visual tokenizer의 장점과 비용

모델 비교 뒤 확인할 독해 체크리스트.

코드를 열었을 때 묻는 순서.

processor가 반환하는 key와 shape는 무엇인가. modality length를 어느 metadata에서 계산하는가. encoder 출력이 projection되기 전후 dtype은 무엇인가. placeholder 수와 feature 수를 누가 검사하는가. position ID와 causal mask는 insertion 전후 어느 시점에 만들어지는가. labels의 `-100` 위치와 loss denominator는 무엇인가. 어느 module이 trainable하며 optimizer group에 실제 들어갔는가. 이 질문의 답을 한 호출 사슬로 연결한다.

데이터 manifest에서 묻는 순서.

원 asset의 권리·해시·시간축은 있는가. decode와 transform revision은 있는가. event/출처 계열 split인가. synthetic caption의 생성 계보가 있는가. reject와 decode-failure가 mixture를 어떻게 바꾸었는가. sample/token/loss-unit 비율이 모두 있는가. 삭제 요청이 derived feature cache와 checkpoint까지 전파되는가를 확인한다.

실험 보고서에서 묻는 순서.

모달리티를 제거·shuffle한 반증이 있는가. shortcut slice가 있는가. trainable scope와 gradient route가 있는가. 해상도·duration·FPS 변화가 실제 token budget으로 보고되었는가. resume 보장 범위가 sample/step/metric 중 어디까지인가. 가장 큰 실패 tail을 평균과 별도로 보여 주는가를 본다.

공통 계약의 판정 기준.

멀티모달 학습은 tensor shape가 맞는다고 완성되지 않는다. 원 신호의 시간·공간 좌표가 feature와 token 위치에 보존되고, 그 위치의 label만 loss를 내며, 의도한 parameter가 gradient를 받고, 데이터 mixture가 실제 optimizer 기여로 확인되어야 한다. 여기에 분산 decode 일치, exact 또는 명시적 resume, leakage family 차단까지 닫혀야 한다.

이 장의 핵심은 modality마다 별도 마법을 외우는 것이 아니다. 모든 modality를 `원 신호의 좌표 → 표현 단위 → 결합 좌표 → supervision 단위 → parameter update`로 펼치는 습관이다. 이 습관은 22장의 noise trajectory, 23장의 parameter edit, 24장의 metric contribution을 읽는 동일한 방법이 된다.

export와 multi-node 진입 조건.

training checkpoint에는 vision tower, projector와 language model key가 있지만 export script가 language model state만 썼다고 하자. target runtime이 projector를 random initialize하거나 missing warning으로 넘기면 text output은 되지만 image grounding이 무너진다. tensor schema의 expected key set과 derivation manifest로 load 전에 실패시킨다.

base text golden만으로는 이 오류를 못 잡는다. 고정 synthetic image와 image-dependent answer, selected projector output/logits를 export 전후 비교한다. quantization이 projector를 지원하는지도 dtype/shape와 runtime loader source에서 확인한다.

negative control은 projector shard와 processor config를 각각 하나씩 누락한다. schema와 bundle gate가 서로 다른 최초 지점에서 거부해야 한다. export artifact는 model weight뿐 아니라 processor/tokenizer/template와 modality config를 하나의 bundle로 묶는다.

multi-node preprocessing 일치.

두 rank가 같은 media를 중복 처리하는 fixture에서 decode pixel/waveform, crop/sample RNG와 feature checksum이 deterministic 경계까지 같아야 한다. node별 codec library, locale와 hardware decoder가 다르면 차이가 날 수 있다. loaded decoder/library digest를 topology manifest에 둔다.

augmentation seed는 sample ID, epoch와 transform revision에서 유도하고 rank/worker 재배치 뒤 policy상 같아야 하는지 선언한다. 다른 augmentation이 허용돼도 label/manifest와 RNG ledger가 필요하다. corrupt media skip이 rank마다 달라 global batch denominator가 어긋나지 않게 한다.

negative control은 한 node의 codec revision을 바꾼다. startup compatibility 또는 feature checksum sampling audit가 잡아야 한다. 검증하지 않은 hardware decode path는 CPU reference 결과를 상속하지 않는다.

독립 검토의 중간 gate.

독자는 image patch/token, audio frame, video PTS와 modality loss denominator를 손계산할 수 있어야 한다. processor→encoder→projector→language loss의 소스 좌표와 state owner를 따라갈 수 있어야 한다.

image order, audio padding, video timestamp, placeholder/tokenizer, corruption/cache, projector export와 node codec negative control이 각자 예상 assertion에서 실패해야 한다. task 평균만으로 이 조건을 대신하지 않는다.

완료 패키지와 상대 playbook 링크, 지원 범위와 미실행 codec/model path를 독립 검토자가 확인한다. RepresentationID의 bytes/config가 닫힐 때 22장 noise trajectory에 안전하게 넘긴다.

export·multi-node 종합 fixture.

한 conversation에 image 두 장, 4초 audio와 text 질문이 있다고 하자. processor manifest는 image patch 144개씩, audio valid frame 398개, text 72 token과 각 placeholder span을 기록한다. collator가 image 순서를 유지하고 audio padding을 제외하며 assistant 40 token만 language loss에 넣는지 손으로 표를 만든다.

projector loss와 language loss가 있다면 raw numerator, denominator와 weight를 분리한다. 전체 scalar가 같아도 image owner와 audio mask가 틀릴 수 있다. 한 optimizer update에서 vision/audio freeze 정책, projector/adapter gradient와 checkpoint key를 검산한다.

negative run은 두 image order를 바꾸고 audio tail을 잘라 shape를 padding으로 복원한다. count와 batch shape는 같지만 identity/PTS·valid-mask assertion, known answer가 실패해야 한다. export에서는 audio projector 하나를 누락해 schema가 거부하는지 본다.

독립 검토자는 소스 좌표의 processor/model branch와 실제 loaded revision을 대조한다. 이 종합 fixture가 통과해도 검증하지 않은 production codec과 multi-node hardware decoder는 지원 표에 남긴다. 작은 exact 증거를 대규모 품질 일반화로 확대하지 않는다.

검토 결과에는 각 modality의 최초 실패 경계와 owner를 적는다. decode checksum이면 data/processor, feature는 맞고 projector가 다르면 model bridge, token·mask가 다르면 collator/template, export에서만 다르면 bundle/loader가 담당한다. 한 팀에 “multimodal 문제”로 넘기지 않는다. fix 뒤 same negative fault가 다시 실패하고 golden path가 통과하는지 확인한다. 승인 시각과 source/processor/model digest를 서명해 다음 장이 mutable alias를 다시 해석하지 못하게 한다.

독립 환경에서 cache를 비우고 fixture를 다시 생성해 hidden local artifact 의존성도 확인한다. 재생성 결과와 manifest가 같을 때 승인한다.

실패하면 최초로 달라진 bytes, transform revision과 담당 owner를 review record에 추가하고 반드시 완전히 독립적으로 다시 검증한다.

이 장이 넘기는 것. `MediaSampleID`, 원본 SHA, timestamp/crop manifest, feature shape, modality별 target count를 22장에 넘긴다.

마지막으로 reviewer는 수치 세 개를 임의로 골라 역산한다. 첫째, 한 영상의 `vision_tokens`를 원 해상도·sampled frame·patch·merge 설정에서 다시 계산한다. 둘째, 한 음성의 valid frame과 label token을 waveform duration·hop length·padding mask에서 다시 구한다. 셋째, language loss의 numerator와 denominator를 assistant label mask에서 재합산한다. dashboard의 값과 손계산이 다르면 평균 metric을 승인하지 않는다.

또한 feature cache를 비운 run과 채운 run을 같은 golden batch로 비교한다. cache hit가 결과를 바꾼다면 key에 transform이나 encoder revision이 빠진 것이다. 서로 다른 rank 수로 재실행할 때 sample order가 달라도 global mixture와 gradient accumulation 의미가 유지되는지 확인한다. rank 수 변화 뒤 scheduler step이나 loss denominator가 달라지면 재현 범위를 별도 run lineage로 분기한다.

지원하지 않는 경로도 결과물이다. 검증하지 않은 codec, hardware decoder, variable-frame-rate container, 긴 audio, 다중 image 수, dynamic resolution 범위를 표에 남긴다. smoke test 한 건을 모든 media로 일반화하지 않는다. 새 경로를 열 때는 해당 decoder·processor·interleave·loss fixture를 추가하고 `MediaSampleID` schema version을 올린다.

이 절차의 목적은 문서를 무겁게 만드는 것이 아니라 문제의 최초 owner를 빨리 찾는 것이다. raw bytes가 다르면 ingestion, decoded tensor가 다르면 media backend, feature가 다르면 encoder/transform, embedding sequence가 다르면 connector/collator, logits가 다르면 model, loss만 다르면 labels/reduction을 본다. 이 분기가 자동화되어 있으면 대규모 학습에서 발견한 이상을 작은 golden batch로 축소할 수 있다.

최종 승인자는 “멀티모달 성능이 좋아졌다”는 결론 옆에 무엇을 보지 못했는지도 쓴다. 공개 inference 코드에서 확인한 사실과 실제 학습 loop에서 검증한 사실, model card에서만 알려진 사실을 구분한다. 이 정직한 경계가 다음 장의 diffusion condition과 그다음 장의 knowledge change 평가를 오염시키지 않는다.

승인 뒤에도 drift 감시는 계속한다. modality별 decode 실패율, token 길이 분위수, padding 비율, loss 분모, gradient norm, cache hit, first-divergence 표본을 시계열로 남긴다. 평균이 안정돼도 특정 언어·codec·해상도 bucket의 tail이 움직이면 새 데이터 generation을 격리한다. 알람은 모델 품질 점수보다 먼저 입력 계약 변화를 가리켜야 한다.

독자는 이제 새 모델을 만났을 때 이름을 외우는 대신 다섯 경계를 찾을 수 있다. 신호를 자르는 함수, 표현을 만드는 함수, modality를 결합하는 함수, label을 배치하는 함수, loss를 줄이는 함수다. 이 다섯 좌표와 state owner를 고정하면 architecture가 달라져도 같은 방법으로 학습 경로를 검증할 수 있다.

검증 체크포인트. 같은 manifest로 두 번 전처리했을 때 token/feature checksum과 유효 label 수가 같아야 한다. 검증 범위도 함께 기록한다. 미검증 경계와 담당 owner도 다음 장에 명시적으로 인계한다.

## 21.11 processor→model→loss 함수 호출 사다리를 읽는다

이제 개념 계약을 실제 함수 호출에 겹쳐 놓는다. 호출 이름을 나열하는 것이 목적이 아니다. 각 함수가 modality clock을 어느 단위로 바꾸고, 반환 shape의 소유권을 누구에게 넘기며, 그 결과가 loss mask와 denominator에 언제 반영되는지를 확인해야 한다. 장애가 나면 이 사다리의 반환값을 위에서 아래로 대조해 최초 불일치를 찾는다.

비교표의 설명을 실제 코드로 내린다. processor가 만든 key와 shape에서 시작해 tower `forward`, projector·merger·resampler, placeholder 또는 cross-attention mask, loss reducer와 optimizer group까지 호출자와 피호출자를 잇는다. 이름이 비슷한 utility보다 실제 trainer가 통과한 branch를 권위 경로로 삼는다.

멀티모달 학습을 이해하는 가장 빠른 길은 모델 이름을 외우는 일이 아니라 샘플 하나가 어느 함수에서 어떤 텐서로 바뀌는지 추적하는 일이다. 원본 샘플에는 대화 메시지, 이미지·음성·영상의 URI, 시간 구간, 해상도 같은 메타데이터가 있다. 데이터셋의 `__getitem__`은 이들을 읽지만, 반환 객체가 곧 모델 입력은 아니다. processor가 텍스트 토큰화와 매체 전처리를 수행하고, collator가 길이를 맞추며, 입력 준비 함수가 placeholder를 실제 feature 열로 치환한다. 이 네 경계의 계약이 어긋나면 손실은 계산되면서도 잘못된 위치를 학습한다.

LLaVA 계열에서는 `sources/training-multimodal-llava/llava/train/train.py`의 데이터 준비 경로와 `llava_trainer.py`의 optimizer 구성이 서로 다른 질문에 답한다. 전자는 이미지 placeholder와 대화 label을 배치하고, 후자는 `mm_projector`를 별도 learning-rate group으로 분리한다. `llava_trainer.py:165-192`가 decay 여부와 projector 여부를 교차해 네 그룹을 만드는 까닭은 이미 언어 공간을 형성한 decoder와 아직 두 공간을 접착해야 하는 connector의 적정 이동 크기가 다르기 때문이다. 같은 학습률은 connector를 너무 느리게 정렬하거나 decoder를 먼저 훼손할 수 있다.

Qwen2-VL의 경로는 connector 하나보다 복잡하다. `sources/transformers-v5.15.1/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`에는 patch embedding, rotary position 계산, vision block, merger, 언어 모델 호출이 분리돼 있다. 각 `forward`에서 `[batch, channel, time, height, width]`, patch 열, merged token 열 중 무엇을 받는지, `grid_thw`가 어디까지 살아 있는지 적어야 한다. grid를 잃은 뒤에는 같은 길이의 열이어도 원래 시간·공간 이웃을 복구할 수 없다.

collator는 값뿐 아니라 좌표계를 배치한다.

텍스트 collator는 padding과 label의 `-100` 마스킹을 주로 책임진다. 멀티모달 collator에는 매체 개수, 매체별 token 수, 공간·시간 grid, placeholder와 asset의 대응이 더해진다. 따라서 불변식은 `len(pixel_values)` 같은 단일 길이가 아니라 매체 인덱스, placeholder 순번, feature offset의 결합이다.

안전한 테스트는 이미지가 없는 샘플, 이미지 한 장, 해상도가 다른 이미지 두 장, 프레임 수가 다른 영상, 비어 있는 음성을 한 배치에 일부러 섞는다. 배치 이후 각 샘플 offset으로 feature를 다시 잘라 원본과 대응시킨다. 개수만 같고 순서가 바뀌는 오류를 잡으려면 테스트용 단색·체커보드·시간 ramp를 써서 connector 직전 통계를 확인한다. 의미를 아는 합성 입력이 shape test를 alignment test로 바꾼다.

이미지 feature가 언어 embedding 사이에 끼어들면 label도 같은 길이로 확장해야 한다. 이미지 구간에는 대개 다음 텍스트 token 정답을 주지 않으므로 ignore index를 채운다. 그러나 BOS, 이미지 시작·끝, newline 처리 convention이 processor와 모델에서 다르면 label이 한 칸씩 밀린다. `[BOS, 사용자, IMG, 질문, 답변]`처럼 짧은 fixture에서 IMG가 세 feature token으로 전개된 뒤의 embedding과 label을 손으로 적고, 각 위치가 무엇을 예측하는지 확인한다.

optimizer group을 파라미터 이름까지 감사한다.

이름 기반 group 분리는 module rename에 취약하다. 기대 파라미터가 정확히 한 group에 속하는지, frozen parameter가 어느 group에도 없는지, group별 parameter 수·dtype·weight decay·학습률이 resume 전후 같은지 검사한다. `requires_grad=False`와 learning rate 0도 같지 않다. 전자는 activation 저장, gradient 통신, optimizer state 할당을 줄일 수 있지만 후자는 계산과 state를 남길 수 있다.

단계적으로 tower를 해제할 때는 새 파라미터를 optimizer에 언제 넣는지, Adam moment를 0에서 시작하는지, scheduler의 현재 step에 어떤 학습률이 되는지를 상태로 저장한다. 이 정보를 복원하지 않으면 weight가 같은 checkpoint에서 출발해도 첫 update가 달라진다. 고정 probe batch에서 connector, vision 마지막 block, language embedding의 gradient norm을 비교하면 누가 정렬 비용을 떠안는지 보인다.

### 21.11.1 vision tower를 해상도와 token 예산으로 읽는다

patch 크기를 `p`, 높이와 너비를 `H,W`라 하면 patch 수는 대략 `N=(H/p)(W/p)`다. vision self-attention의 score 행렬은 `N²`에 비례하므로 각 변을 두 배로 키우면 patch 수가 네 배, attention 항은 열여섯 배가 된다. connector가 열을 줄이기 전에 tower는 이 비용을 이미 지불한다. 동일 batch size라도 동적 해상도의 `sum(T_i H_i W_i)`가 요동하므로 OOM과 step time을 샘플 수로 예측하면 안 된다.

인접 patch 병합은 공짜 압축이 아니다. 단순 평균은 저주파를 남기고 작은 글자·얇은 경계·짧은 시간 변화를 지운다. 학습된 merger도 학습 분포에서 loss가 요구한 방향만 보존한다. 따라서 token 절감률과 함께 OCR의 작은 글자, chart의 얇은 선, 영상의 짧은 사건 slice를 평가한다. 병합 전후 covariance의 고윳값, retrieval alignment, conditional loss를 함께 보면 압축과 정렬을 구분할 수 있다.

tile과 crop은 위치 계약을 요구한다.

고해상도 이미지를 tile로 나누면 tile 내부 위치와 전체 이미지 위치가 모두 필요하다. 단순 연결은 서로 먼 경계 token을 이웃으로 만들 수 있다. global thumbnail, local tiles, separator, 2D rotary 좌표 중 무엇을 쓰는지 config와 processor 양쪽에서 확인한다. 질문이 가리키는 물체를 crop이 지우면 정답은 더 이상 입력에서 도출되지 않는다. 이는 보통 label noise가 아니라 조건부 정보 삭제이므로 질문·bounding box·OCR 영역과 augmentation을 연동한다.

rank별 workload도 patch budget으로 맞춘다. 긴 영상이나 고해상도 이미지가 한 rank에 몰리면 다른 rank는 collective에서 기다리고 profiler에는 NCCL 시간이 길게 보인다. rank별 patch 수, 전처리 시간, vision forward, language forward, collective wait를 같은 step ID로 기록해야 통신 병목과 입력 불균형을 구분한다.

위치 부호의 단위를 검증한다.

2D 또는 3D rotary position을 쓰는 모델은 좌표 값뿐 아니라 좌표 단위가 계약이다. frame index와 실제 timestamp, patch index와 원본 pixel 위치는 같지 않다. frame drop이나 tile reorder 뒤 position ID가 단조 증가해도 사건과 어긋날 수 있다. 작은 격자에서 좌표를 출력해 손으로 계산한 sin·cos phase와 비교하고, processor가 만든 grid와 모델이 소비한 grid를 같은 fixture에 보존한다.

### 21.11.2 audio tower를 waveform·frame·decoder 시계로 읽는다

음성 학습에는 waveform sample, spectrogram frame, 출력 token의 서로 다른 clock이 있다. sample rate가 16 kHz라고 encoder가 초당 16,000개 token을 보는 것은 아니다. window와 hop으로 log-Mel frame을 만들고 convolution stride가 시간을 다시 줄인다. Whisper의 `sources/transformers-v5.15.1/src/transformers/models/whisper/feature_extraction_whisper.py`와 `modeling_whisper.py:540-648`을 함께 읽어야 duration에서 encoder 열 길이까지 연결된다.

Whisper encoder의 `forward`는 feature length, convolution, positional embedding, encoder layer를 잇고 decoder의 cross-attention이 그 열을 조건으로 쓴다. 같은 파일 `modeling_whisper.py:1241-1333`의 audio-classification `projector`는 이름은 같아도 시각-언어 connector와 역할이 다르다. 여기서는 encoder hidden size를 분류 차원으로 투영한다. 저장소 검색에서 같은 명칭을 찾았다는 이유로 같은 개념이라 묶으면 안 된다.

padding과 silence를 구별한다.

고정 길이 log-Mel에서 무음과 padding은 값이 비슷할 수 있지만 무음은 관측된 사건이고 padding은 관측하지 않은 영역이다. mask가 이 차이를 잃으면 모델은 padding 패턴을 문장 끝 단서로 학습한다. 유효 음성은 같고 padding 길이만 다른 쌍에서 encoder의 유효 구간 출력과 loss가 유지되는지 검사한다. 잘린 마지막 phoneme, 실제 무음, decode 실패로 생긴 0도 별도 fixture로 둔다.

가변 길이 음성은 processor가 산출한 유효 frame 수로 bucketing한다. 원본 duration만으로는 resample과 feature extraction 뒤 길이를 정확히 예측하지 못할 수 있다. augmentation seed는 sample ID와 epoch에서 결정적으로 유도하고, manifest에는 원본 hash, sample rate, resampler revision, 유효 frame, crop offset을 남긴다.

codec token의 손실 분모를 분리한다.

neural codec의 RVQ는 여러 codebook이 residual을 순서대로 설명한다. codebook `k`마다 분포와 난이도가 다르므로 `L=sum_k lambda_k L_k`의 각 `L_k`, 활성 token 수, perplexity, code usage entropy를 따로 기록한다. 앞쪽 codebook의 큰 신호가 뒤쪽 residual codebook의 붕괴를 가릴 수 있다.

전체 reconstruction loss가 줄어도 일부 code만 반복 사용하는 collapse가 일어난다. 흔한 소리는 유지되지만 희귀 음색과 자음이 사라질 수 있다. dead-code 비율, ASR intelligibility, speaker similarity를 분리해 tokenizer 오차와 생성 모델 오차를 같은 숫자에 합치지 않는다.

### 21.11.3 video sampling을 시간 사건으로 읽는다

영상의 결정은 몇 장을 보느냐보다 어느 시간에서 무엇을 보존하느냐다. 균일 sampling은 긴 정적 구간을 대표하지만 짧은 사건을 놓친다. shot-aware 방식은 장면 전환에, motion-aware 방식은 움직임에 유리하지만 정적인 자막이나 도표를 놓칠 수 있다. manifest에 원본 FPS, duration, 선택 timestamp, sampling 정책과 seed를 저장한다.

시간 위치는 frame index보다 실제 timestamp와 연결해야 한다. 서로 다른 FPS 영상을 같은 frame index로 취급하면 같은 숫자가 다른 시간을 뜻한다. temporal RoPE의 scale factor를 processor와 model config 양쪽에서 확인하고, clip concat과 frame drop 뒤 timestamp가 원본 사건을 계속 가리키는지 검증한다.

straggler를 데이터 경로부터 찾는다.

영상은 storage, CPU decoder, host memory, GPU를 가로지른다. rank별 step time 차이가 나면 `read`, `decode`, `sample`, `resize`, `host-to-device`, `vision-forward`, `language-forward`, `collective`를 각각 계측한다. 한 rank의 decode tail latency가 collective wait로 보일 수 있기 때문이다. GPU timeline의 빈 구간과 dataloader queue depth를 함께 본다.

decode 실패를 rank별로 조용히 건너뛰면 global sample order와 gradient denominator가 달라진다. 모든 rank가 같은 batch를 폐기하거나, 사전에 검증한 대체 ID를 결정적으로 고르거나, zero-weight placeholder로 collective shape를 유지하는 정책 중 하나를 고정한다. 실패 횟수, asset hash, decoder 오류를 checkpoint 외부 감사 로그에도 남긴다.

실제 정보량으로 global batch를 말한다.

`batch_size × world_size × accumulation`은 멀티모달 global batch를 충분히 설명하지 못한다. optimizer step당 text token, image patch, audio frame, video frame의 합과 분포를 함께 보고한다. curriculum에서 영상 비중이 늘면 sample 수가 같아도 compute와 gradient 분포가 달라진다.

microbatch마다 modality 구성이 다를 때 각 microbatch loss를 먼저 평균한 뒤 합치는 식과 모든 유효 token loss를 합쳐 한 번 나누는 식은 같지 않다. 원하는 추정량을 먼저 쓰고 DDP의 자동 평균 계수까지 넣어 실제 구현식을 검산한다. 손실 값만 로그하지 말고 분자와 modality별 분모를 남긴다.

objective를 gradient 충돌로 읽는다.

`L=lambda_text L_text + lambda_align L_align + lambda_recon L_recon`에서 가중치만 보아서는 실제 영향력을 모른다. 각 항의 규모, 공유 파라미터에 대한 gradient norm, gradient cosine이 중요하다. 두 gradient의 cosine이 음수면 한 항을 개선하는 update가 다른 항을 악화하는 방향이다. weight 변경은 단순 강조가 아니라 공유 표현의 이동 방향을 바꾼다.

모든 step의 gradient를 저장할 필요는 없다. 고정 probe batch에서 connector, tower 마지막 block, language embedding과 decoder 중간 block의 norm과 선택한 손실 쌍 cosine을 주기적으로 측정한다. align loss가 줄면서 language gradient가 폭증하면 connector가 해결할 변환을 decoder가 대신 떠안는 중일 수 있다.

mixed precision 장애를 목적함수 변화와 구분한다.

dynamic loss scale은 underflow를 막기 위한 수치 장치이고 modality loss weight는 목적함수다. 로그에서 둘을 같은 `scale`로 부르면 분석이 꼬인다. overflow로 step이 skip된 시점에는 batch 구성, unscaled gradient norm, scaler 값, optimizer와 scheduler step counter를 함께 남긴다. scheduler가 skip된 optimizer step에도 증가했는지 확인한다.

tower별 dtype이 다르면 connector 경계의 cast가 학습 그래프 일부가 된다. fp32 normalization 뒤 bf16 projection인지, fp16 tower 출력 자체를 저장하는지에 따라 작은 feature가 사라지는 위치가 다르다. 고정 fixture로 tower 출력, cast 직후, connector 출력의 finite 비율과 오차를 비교한다.

조건부 경로의 unused parameter를 설계한다.

텍스트 전용 batch에서는 vision tower가 호출되지 않을 수 있다. DDP가 모든 parameter의 gradient를 기다리면 hang이나 오류가 난다. `find_unused_parameters=True`는 동작을 돕지만 graph traversal 비용과 bucket 준비가 달라진다. modality를 항상 섞을지, tower별 batch와 조건부 collective를 설계할지, zero-valued 연결을 둘지 선택해야 한다.

world size 2의 정적 fixture에서 rank별 modality를 다르게 주고 gradient가 생긴 파라미터 이름을 비교한다. 실제 대규모 실행 대신 테스트 설계와 기대 불변식을 코드 옆에 남겨 두면 첫 장비 실행에서 원인을 빠르게 줄일 수 있다.

체크포인트를 재현 상태로 정의한다.

멀티모달 checkpoint에는 weight 외에도 processor config, tokenizer added token, image normalization, resize/crop, audio sample rate와 Mel 설정, video sampling이 포함돼야 한다. `llava_trainer.py:239-247`처럼 projector만 저장하는 경로는 단계 학습에 유용하지만 그 파일만으로 모델을 복원할 수 있다는 뜻은 아니다. base LLM, vision encoder revision, projector type, special-token convention이 맞아야 한다.

`sources/training-multimodal-llava/docs/MODEL_ZOO.md:51-57`이 projector와 base LLM·vision encoder 호환성을 강조하는 이유도 이 계약 때문이다. shape가 같다는 것은 좌표계가 같다는 증거가 아니다. export manifest에는 저장소 commit, config와 processor hash, 파라미터 이름·shape, freeze map을 함께 기록한다.

partial load를 실패 목록으로 검증한다.

`strict=False`는 호환성 정책이 아니다. missing, unexpected, shape-mismatch key를 분류해 기대 목록과 대조한다. projector-only load라면 언어·vision key 누락은 기대되지만 projector key 하나의 누락은 치명적이다. load 직후 고정 입력의 connector 출력까지 허용오차로 비교한다.

adapter에는 base revision, target module, merge dtype과 순서가 더해진다. vision linear까지 LoRA를 붙였는지 language attention만 대상으로 했는지에 따라 의미가 달라진다. 양자화 base에서 학습한 adapter를 full-precision base에 합칠 때 동일성을 가정하지 말고 probe를 통과시킨다.

exact resume를 샘플 순서까지 검증한다.

checkpoint 직전과 직후의 global sample ID, augmentation seed, modality token count를 작은 ring buffer로 남긴다. streaming dataset에는 shard permutation, worker별 cursor와 byte offset도 필요하다. 영상 decoder 버전이 바뀌면 같은 timestamp에서 다른 frame을 고를 수 있으므로 container와 codec revision까지 환경 manifest에 묶는다.

resume 뒤 첫 batch ID, optimizer group, moment, scaler, scheduler step을 이전 실행의 예상 상태와 비교한다. weight checksum만 같은 checkpoint는 exact resume의 충분조건이 아니다.

실패를 최초 불일치 표로 좁힌다.

“이미지 질문 성능이 떨어졌다”는 진단 가능한 문장이 아니다. asset hash, decode 결과, normalized tensor, patch 수, tower 출력, connector 출력, 삽입 embedding, mask, label, loss 항, gradient, update 순으로 두 실행을 비교한다. 최초로 달라진 단계가 원인 후보의 상한을 정하고 뒤 차이는 대부분 전파 결과다.

| 증상 | 먼저 고정할 경계 | 반증 실험 | 흔한 원인 |
|---|---|---|---|
| OCR만 급락 | crop·resize·grid | 고정 원본 크기와 crop | 작은 글자 소실, tile 순서 |
| 음성 끝 누락 | duration·feature mask | padding 길이만 바꾼 쌍 | 무음과 padding 혼동 |
| 영상 관계 급락 | timestamp·frame ID | 동일 frame 순서 fixture | 시간 좌표 재설정 |
| 텍스트 능력 급락 | language gradient·LR group | connector만 학습한 대조군 | decoder 과이동 |
| rank별 loss 차이 | modality budget | 같은 manifest slice 강제 | 불균형·skip 정책 |
| resume 후 발산 | cursor·optimizer state | 다음 한 step 비교 | sampler·moment 불일치 |

지표를 교차 slice로 자른다.

이미지·음성·영상을 각각 평가하는 데서 멈추지 않는다. 해상도, 길이, 언어, 화자, 배경 소음, frame rate, 질문 유형과 답변 길이의 교차 slice가 필요하다. slice가 많으면 표본이 줄므로 분모와 신뢰구간을 붙이고 핵심 slice를 사전에 정한다.

connector drift는 cosine 하나로 판단하지 않는다. norm, covariance spectrum, retrieval recall, conditional loss를 함께 본다. cosine이 같아도 norm 변화가 attention logit scale을 바꾸고, 평균 norm이 같아도 특정 방향이 붕괴할 수 있다.

capability와 안전을 같은 lineage에 묶는다.

텍스트 안전 필터는 이미지나 음성으로 우회될 수 있다. 출시 패킷에는 정상 품질, modality별 공격 성공률, 도구 호출 위험, 개인정보 재현, 변환 공격 강건성을 함께 넣는다. 25장의 red-team asset hash와 prompt family ID를 공유하면 실패를 학습 데이터와 checkpoint까지 역추적할 수 있다.

설명을 구현 작업으로 바꾸는 작업 계약.

새 저장소에서는 processor, collator, 입력 준비, tower, connector, decoder, loss, trainer, checkpoint 경계를 찾는다. 각 경계의 입력·출력 shape와 dtype, mask, gradient, 저장 상태와 고정 revision 좌표를 기록한다. 합성 asset 하나로 전체 경로를 손으로 재구성해 코드 결과와 비교하기 전에는 대규모 설정을 논하지 않는다.

최소 인수 패키지는 원본·전처리 hash, token·patch·frame budget, placeholder-feature bijection, label mask, loss 분자·분모, optimizer membership, rank별 workload, checkpoint lineage, resume 다음 batch, modality별 평가 slice를 포함한다. 이 패키지가 있어야 성능 변화에서 깨진 계약을 질문할 수 있다.

이미지는 공간 표본, 음성은 연속 시간 신호, 영상은 사건의 부분 관측, 텍스트는 tokenizer가 만든 이산 열이다. 공통 transformer에 넣어도 측정 과정과 손실 분모는 같아지지 않는다. 좋은 구현은 차이를 명시적인 processor·mask·position·objective 계약으로 연결하고, 좋은 디버깅은 그 계약을 역순으로 풀어 최초 불일치를 찾는다.

connector를 선형대수와 최적화 문제로 해부한다.

vision encoder의 출력 한 행을 `v∈R^{d_v}`, language model이 기대하는 embedding을 `e∈R^{d_l}`라 하자. 가장 단순한 connector는 `e=Wv+b`다. 이 식은 차원만 맞추는 배관처럼 보이지만 실제로는 두 표현 공간의 basis와 scale을 함께 바꾼다. `W`의 singular value가 몇 축에 몰리면 vision feature의 많은 방향이 같은 language 방향으로 눌린다. 반대로 큰 singular value가 있으면 작은 vision perturbation이 decoder 입력에서 크게 증폭된다.

MLP connector는 `e=W_2 φ(W_1v+b_1)+b_2`로 비선형 경계를 만들 수 있다. 표현력은 늘지만 데이터가 부족할 때 특정 caption 패턴을 외우기 쉬워진다. connector만 pretrain하는 단계와 full instruction tuning 단계를 나누는 이유는 처음에는 두 공간의 거친 정렬을 안정적으로 만들고, 그 뒤 task-conditioned interaction을 학습하기 위해서다. 이 단계 구분은 `freeze` 설정, optimizer group, checkpoint 파일과 함께 재현해야 한다.

선형 connector의 gradient는 한 샘플에서 `∂L/∂W=(∂L/∂e)v^T`인 outer product다. 배치 전체 update의 rank와 방향은 vision feature의 span과 decoder가 돌려보낸 error vector에 의해 정해진다. 데이터에 특정 시각 개념이나 언어 표현이 부족하면 그 방향은 충분히 갱신되지 않는다. “더 많은 image-text pair”보다 feature covariance와 error covariance가 어떤 방향을 덮는지 보는 편이 정렬 실패를 더 정확히 설명한다.

scale mismatch를 attention logit까지 추적한다.

connector 출력 norm이 text embedding보다 두 배 크다고 하자. layer normalization이 뒤에 있으면 괜찮다고 단정하기 쉽지만 normalization 위치와 residual 경로에 따라 첫 attention의 query·key scale과 residual 비율이 달라질 수 있다. attention logit은 대략 `q·k/√d`이므로 norm 변화가 softmax 포화로 이어질 수 있다. connector 출력과 같은 위치의 text embedding norm, 첫 block normalization 전후, attention entropy를 함께 기록한다.

정렬 단계에서는 modality token의 평균·표준편차만 맞추는 것으로 부족하다. 방향별 분산, token 간 상관, position에 따른 norm도 본다. 모든 이미지 token이 비슷한 방향이면 평균과 표준편차는 정상이어도 attention이 개별 patch를 구분하지 못한다. effective rank와 pairwise cosine 분포가 collapse의 조기 신호가 된다.

connector 용량과 token compression을 분리한다.

MLP 층을 늘리는 일과 patch 수를 줄이는 일은 서로 다른 축이다. 전자는 feature별 변환의 표현력을 바꾸고, 후자는 sequence의 정보 병목과 decoder 비용을 바꾼다. resampler나 query-former는 학습 가능한 query로 variable-length feature를 고정 개수 token으로 압축한다. 이때 query 수는 단순한 성능 knob가 아니라 decoder에 전달할 수 있는 독립 관측 수의 상한이다.

비교 실험은 connector 파라미터 수, 출력 token 수, decoder FLOPs를 따로 맞춘다. MLP와 resampler를 비교하면서 token 수까지 달라지면 정확도 변화가 변환 방식 때문인지 계산 예산 때문인지 알 수 없다. 고정 compute와 고정 token 두 조건을 모두 보고 Pareto frontier를 제시한다.

attention mask를 블록 행렬로 그린다.

텍스트와 이미지 token을 한 열에 섞으면 attention mask는 누가 누구를 볼 수 있는지를 정의하는 블록 행렬이 된다. decoder-only 모델에서 텍스트는 보통 causal하지만 image token 내부를 양방향으로 볼지, 앞선 텍스트를 볼지, 뒤 답변 token이 모든 image token을 볼지는 구현에 따라 다르다. 단순한 삼각 mask로 가정하면 interleave의 의미를 놓친다.

열을 `[system, user text, image, question, answer]` 블록으로 나누고 허용 연결을 0/1 표로 그린다. image block이 vision tower에서 이미 양방향 encoding을 마쳤다면 decoder 안에서는 causal 위치에 놓여도 각 image embedding 자체가 전체 이미지 문맥을 담을 수 있다. 그러나 여러 image를 interleave할 때 뒤 image가 앞 질문을 조건으로 encode되는 것은 아니다. tower 호출 시점과 decoder attention을 구분한다.

padding mask와 causal mask의 합성을 확인한다.

padding mask는 유효하지 않은 key를 숨기고 causal mask는 미래 key를 숨긴다. 구현에서 bool, additive `-inf`, 2D, 4D mask가 오가며 dtype 변환이 일어난다. fp16에서 충분히 작은 유한값을 쓰는 경우 softmax 결과가 정확히 0인지 확인한다. all-masked row는 NaN을 만들 수 있으므로 빈 modality나 완전히 ignore된 샘플 fixture가 필요하다.

left padding은 position ID와 cache 위치를 바꾼다. 학습은 right padding, 추론은 left padding을 쓰면 같은 대화의 token 위치가 달라질 수 있다. 멀티모달 placeholder 전개 후 position을 다시 계산하는지, 원래 placeholder 한 칸을 여러 feature가 공유하는지 모델별로 고정한다.

sequence packing에서 샘플 경계를 봉쇄한다.

여러 대화를 한 열에 pack하면 서로 다른 샘플이 attention으로 섞이지 않도록 block-diagonal mask가 필요하다. 이미지 feature의 variable expansion 때문에 원본 text offset만으로 block을 만들면 경계가 틀어진다. expansion 뒤 최종 offset에서 mask와 position ID를 생성하고 label도 같은 경계에서 reset한다.

packing 효율은 총 유효 token 비율만으로 보지 않는다. vision patch와 text token의 연산 비용이 다르고 tower는 decoder packing 전에 별도로 실행될 수 있다. `decoder tokens`, `tower patches`, `padding`, `samples per pack`을 분리해서 측정한다. 이미지가 큰 샘플 하나 때문에 text padding이 늘어나는 경우 modality-aware bin packing이 유리할 수 있다.

loss를 확률모형의 조건으로 다시 쓴다.

멀티모달 causal language modeling은 보통 `-log p(y_t | y_<t, x_text, x_media)`를 최소화한다. 여기서 media token 위치의 label을 ignore해도 media가 학습되지 않는 것은 아니다. 답변 token의 log probability가 media feature에 의존하므로 chain rule을 통해 tower와 connector로 gradient가 흐른다. 다만 답변이 이미 텍스트만으로 예측 가능하면 모델은 media를 무시하는 지름길을 택할 수 있다.

이 지름길을 찾으려면 같은 질문에서 media를 제거·교체·shuffle한 counterfactual을 평가한다. 정답 log probability가 거의 변하지 않으면 높은 정확도도 시각 grounding의 증거가 아니다. positive media와 hard-negative media의 log-likelihood 차이, attention attribution 변화, answer consistency를 함께 본다.

contrastive loss의 배치 의미를 해부한다.

image-text contrastive loss에서 배치의 다른 샘플은 negative가 된다. 분산 학습에서 all-gather로 global negatives를 쓰는지 local rank negatives만 쓰는지에 따라 목적함수가 바뀐다. gather된 feature에 gradient가 흐르는지 detach되는지도 다르다. world size 변경은 단지 처리량 변화가 아니라 negative pool 크기와 gradient estimator 변화가 될 수 있다.

temperature `τ`는 `sim/τ`로 logit 간격을 조절한다. 작은 `τ`는 hard negative에 큰 gradient를 주지만 잘못 짝지어진 데이터에 민감하다. 학습된 temperature가 clamp되는지, rank별 batch 크기가 다를 때 denominator가 같은지 확인한다. positive pair가 다른 rank에 중복되면 false negative가 생길 수 있으므로 asset와 caption identity를 global batch에서 검사한다.

auxiliary reconstruction의 누출을 막는다.

masked patch나 codec reconstruction loss는 표현이 입력 세부를 보존하도록 돕지만 downstream answer에 불필요한 픽셀·음향 정보를 과도하게 유지할 수 있다. 개인정보나 워터마크 같은 nuisance까지 보존할 수 있으므로 reconstruction 품질과 task relevance, privacy leakage를 함께 평가한다. auxiliary head가 export에는 빠지더라도 그것이 만든 representation bias는 본체에 남는다.

loss weight schedule도 checkpoint 상태다. 초기에 alignment를 크게 두고 후기에 instruction loss를 키우는 curriculum은 동일 최종 weight만으로 재현되지 않는다. step 기준인지 consumed-token 기준인지, modality mix 변경과 동시에 일어나는지 기록한다.

데이터 혼합을 확률표본추출로 해석한다.

데이터셋 `D_m`을 확률 `p_m`으로 선택하고 내부 샘플을 뽑는다면 관측된 modality 비율은 유한 실행에서 설정값과 다르다. 분산 sampler와 worker prefetch, decode 실패, 길이 bucketing이 실현 확률을 더 바꾼다. 로그에는 설정 `p_m`과 실제 선택 수, 성공적으로 loss에 들어간 수를 모두 남긴다.

한 샘플의 loss가 token 평균이면 긴 답변과 짧은 답변의 샘플 가중치가 달라진다. dataset sampling weight, sample weight, modality loss weight, token reduction이 곱해져 최종 gradient 기여도를 만든다. 이 네 값을 분리하지 않으면 “이미지 데이터를 30% 넣었다”는 문장은 학습 영향력을 설명하지 못한다.

curriculum의 전이를 상태기계로 만든다.

`connector warm-up → tower 일부 해제 → full instruction tuning → preference` 같은 단계는 날짜가 아니라 전이 조건을 가져야 한다. consumed modality tokens, validation grounding, connector gradient 안정성, collapse 지표 같은 조건과 최대 step을 함께 둔다. 자동 전이가 일어날 때 freeze map, optimizer group, loss weight, data mixture를 원자적으로 바꾸고 event를 checkpoint manifest에 기록한다.

resume가 전이 경계 근처에서 일어나면 이전 stage optimizer를 복원한 뒤 다시 전이하는지, 이미 전이한 상태를 불러오는지 명확해야 한다. stage ID와 transition event ID를 저장하고 동일 event를 두 번 적용하지 않는 idempotence test를 둔다.

dedup을 매체와 의미 두 층에서 수행한다.

이미지 byte hash만으로는 resize·reencode·crop 복제를 못 잡고 perceptual hash만으로는 같은 배경의 다른 텍스트를 합칠 수 있다. audio도 waveform hash, acoustic fingerprint, transcript similarity가 서로 다른 복제를 찾는다. video는 frame-level 근접성과 clip temporal overlap을 함께 봐야 한다.

평가 오염을 막을 때 caption text만 비교하면 benchmark 이미지가 다른 질문으로 학습에 들어온 경우를 놓친다. asset identity, near-duplicate media, OCR text, transcript, semantic embedding을 계층적으로 비교하고 어느 규칙으로 제외했는지 lineage를 남긴다. false positive로 제거된 데이터가 특정 언어나 문화권에 편향되지 않았는지도 표본 감사한다.

모델 패밀리 차이를 변경 가능한 계약으로 정리한다.

LLaVA형은 외부 vision tower와 비교적 명시적인 projector, decoder를 연결해 경계 추적이 쉽다. Qwen2-VL형은 동적 해상도와 grid-aware position, merger가 깊이 결합된다. Qwen2-Audio의 `sources/transformers-v5.15.1/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py:414-429`는 audio projector를 구성하고, `:648-714` 부근 forward 경로가 선택된 audio feature를 projector에 통과시킨다. Whisper는 encoder-decoder ASR 조건화가 중심이다.

이들을 “모두 projector가 있다”로 묶지 않고 입력 단위, tower 출력 단위, 압축 방식, position 계약, decoder 조건화, loss 대상, freeze·export 단위를 표로 비교한다. 같은 config 이름도 효과가 다르고 같은 효과도 다른 이름으로 구현된다. 옵션 설명은 이름이 아니라 어떤 텐서 shape, 파라미터 집합, gradient와 저장 artifact를 바꾸는지까지 내려간다.

구현 좌표를 고정 revision과 테스트에 묶는다.

파일과 줄 번호는 읽기 시작점이지 영구 식별자가 아니다. source registry의 commit과 repo-relative path, symbol, line span, content hash를 함께 쓴다. upstream 변경 뒤 line이 움직여도 symbol과 hash로 재탐색할 수 있다. 한 구현 주장을 뒷받침하려면 가능하면 production 함수와 그것을 호출하거나 검증하는 test·example 좌표를 쌍으로 둔다.

LLaVA의 projector 전용 저장은 trainer의 저장 분기와 model zoo의 호환성 경고를 함께 읽는다. 전자는 실제 동작, 후자는 사용자에게 요구되는 조립 계약을 보여 준다. Qwen2-VL은 processor test에서 `grid_thw`와 placeholder expansion 기대값을 찾아 modeling forward와 연결한다. Whisper는 feature extractor의 frame 계약과 encoder 입력 검사를 연결한다.

모델 카드 주장을 검증 가능한 문장으로 바꾼다.

“dynamic resolution을 지원한다”는 문장은 최소·최대 pixel, patch·merge 규칙, batch의 허용 grid, 학습에서 본 범위, 추론 backend 제한으로 분해한다. “긴 audio를 지원한다”도 feature window, chunk overlap, timestamp token, decoder context, 평가 duration 분포로 나눈다. 지원 범위를 벗어난 입력은 성공적으로 실행돼도 품질 보장은 없다.

관측성을 텐서 계약 위에 세운다.

대시보드의 첫 층은 입력이다. modality별 성공·실패 수, 원본 길이·해상도, decode latency, 유효 frame·patch·token, padding 비율, cache hit를 본다. 둘째 층은 표현이다. tower·connector 출력 norm, finite 비율, effective rank, attention entropy를 본다. 셋째 층은 학습이다. loss 분자·분모, gradient norm, overflow, optimizer step을 본다. 넷째 층은 시스템이다. rank별 stage time, H2D 대역폭, collective wait, memory peak를 본다.

모든 metric에는 run, global step, consumed modality units, data snapshot, model revision을 label로 붙이되 sample ID처럼 cardinality가 큰 값은 trace나 exemplar에 둔다. 평균만 전송하지 말고 분위수와 오류 reason count를 유지한다. 한 modality가 완전히 빠지면 그 loss가 0으로 보이는지 missing으로 보이는지도 경보 의미를 바꾼다.

알람을 원인에 가까운 순서로 설계한다.

최종 benchmark 하락은 늦고 비싸다. decode failure 증가, token budget drift, placeholder mismatch, connector norm 변화, modality별 gradient 소실이 더 빠른 신호다. 알람은 이 순서로 계층화하고 upstream 알람이 울리면 downstream score 알람을 원인으로 오해하지 않도록 같은 incident에 묶는다.

정상 범위는 전체 평균이 아니라 modality와 주요 slice별 baseline에서 만든다. 새로운 dataset shard가 들어오는 시점에는 expected change window를 선언하지만 알람을 끄지는 않는다. 변경 승인 문서에 예상 방향과 허용 폭을 적고 실제 관측과 비교한다.

trace 하나로 rank와 sample을 연결한다.

느린 step의 trace에는 global batch ID, rank, sample들의 asset hash prefix, modality budget, 각 pipeline stage 시간을 남긴다. 개인정보 원문은 넣지 않고 별도 접근 통제 저장소의 ID만 참조한다. 동일 trace ID로 data loader 로그, GPU profiler range, NCCL event, loss exemplar를 연결하면 “통신이 느렸다”를 어느 입력이 어느 rank를 늦췄는지까지 줄일 수 있다.

작은 실험으로 큰 실패를 반증한다.

첫 실험은 한 샘플 overfit이다. 목적은 일반화가 아니라 label과 gradient 경로가 존재하는지 보는 것이다. 텍스트만으로 답할 수 없는 합성 이미지·질문을 사용하고 media shuffle에서 정답 확률이 떨어져야 한다. connector만 학습, connector+tower 마지막 block, full model 세 조건에서 gradient와 수렴 속도를 비교한다.

둘째는 두 샘플 permutation test다. 서로 다른 고유 패턴의 asset과 답을 만든 뒤 매체 순서, placeholder 순서, 배치 순서를 각각 바꾼다. 올바른 구현은 의미에 맞춰 답이 바뀌고 배치 순서 자체에는 불변이어야 한다. 이 실험은 silent cross-sample leakage를 잘 잡는다.

셋째는 resume one-step equivalence다. 같은 상태에서 연속 두 step을 간 실행과 한 step 뒤 저장·복원해 다음 step을 간 실행의 sample IDs, logits 일부, loss 분자·분모, gradient, updated parameter를 비교한다. stochastic augmentation과 dropout RNG까지 복원 범위에 포함할지 명시한다.

수치 허용오차를 단계마다 다르게 둔다.

decode와 token ID는 exact equality를 요구한다. fp32 processor 결과는 매우 작은 tolerance, bf16 tower와 분산 reduction은 더 넓은 tolerance가 필요할 수 있다. 최종 logit 허용오차 하나만 두면 앞단의 작은 불일치를 놓친다. 경계별 dtype과 연산 특성에 맞춰 absolute·relative tolerance를 정한다.

동일성이 보장되지 않는 kernel이나 hardware decoder는 통계적 등가 범위를 별도로 검증한다. 그러나 비결정성을 이유로 큰 차이를 허용하지 않는다. repeated run의 자연 변동 분포를 먼저 측정하고 변경 전후 차이가 그 범위 안인지 본다.

실패 주입을 복구 절차의 일부로 둔다.

깨진 이미지, 잘못된 sample rate, placeholder 하나 누락, NaN feature, rank 한 곳의 decode 지연, projector key 누락을 의도적으로 주입한다. 각 실패가 조용한 품질 저하가 아니라 예상한 assertion·metric·quarantine으로 이어져야 한다. 복구 후 동일 asset이 어떻게 재처리되고 lineage가 어떻게 분기되는지도 확인한다.

실무 증명표: 누가 무엇을 검증하는가.

데이터 담당자는 asset 권리·hash·decode·dedup·split 오염과 modality metadata를 증명한다. 모델 담당자는 processor-to-tower, connector, insertion, mask, loss, gradient 경계를 증명한다. 분산 담당자는 workload 균형, collective participation, optimizer ownership, checkpoint atomicity를 증명한다. 평가·안전 담당자는 counterfactual grounding, slice uncertainty, 공격 변환을 증명한다.

각 증명에는 주장, 고정 입력, 실행 또는 정적 분석 절차, 기대 결과, 실제 결과, source 좌표, 미검증 범위, owner와 expiry가 있다. “테스트 통과”만 적지 않고 어떤 불변식을 어떤 fixture가 덮는지 적는다. 테스트 파일이 존재하는 사실과 필요한 경계를 실제로 검증한다는 사실은 다르다.

출시를 막아야 하는 조건.

placeholder-feature bijection을 재현하지 못하거나 label 분모를 설명하지 못하면 출시를 막는다. base·tower·processor revision이 없는 projector나 adapter도 막는다. rank별 decode skip이 sample order를 바꾸는데 lineage가 없거나, resume 뒤 mixture가 달라지거나, media counterfactual에 모델이 반응하지 않는데 grounding을 주장하는 경우도 막는다.

성능 평균이 좋아도 주요 안전 slice의 공격 성공률이 나빠지고 비용 행렬상 허용할 수 없으면 막는다. 반대로 모든 미검증 조합을 무조건 blocker로 만들지는 않는다. 지원 범위를 명시적으로 좁히고 런타임 validation으로 거부할 수 있다면 제한 출시가 가능하다.

다음 장에 넘길 조건 표현.

22장의 diffusion·flow 모델에서도 이미지·텍스트·audio 조건은 processor, encoder, connector를 거쳐 denoiser에 들어간다. 따라서 이 장은 `ConditionArtifact`의 원본 hash, encoder revision, shape·dtype, mask, dropout 여부, cache key를 넘긴다. 22장은 이를 어느 timestep의 어떤 prediction target과 결합했는지 추가한다.

이 인계가 있으면 생성 실패를 condition corruption과 trajectory 오류로 나눌 수 있다. 조건 tensor가 처음부터 다르면 21장의 경계를 보고, 조건은 같은데 첫 denoising prediction이 다르면 22장의 model·scheduler 경계를 본다. 두 장의 공통 ID가 최초 불일치 탐색을 끊김 없이 만든다.

메모리 장부를 텐서 수명으로 작성한다.

멀티모달 학습의 메모리는 파라미터 수에 activation을 대충 더해서 예측하기 어렵다. tower가 만든 patch·frame feature가 connector와 decoder backward까지 살아 있는지, gradient checkpointing으로 어느 구간을 재계산하는지, variable-length sample을 padding하는지 packing하는지에 따라 peak 위치가 달라진다. 메모리 장부에는 tensor 이름, shape, dtype, 생성 함수, 마지막 소비 함수, backward 보존 여부, shard·replication 상태를 적는다.

vision tower activation은 대략 layer 수와 patch 수, hidden size에 비례하고 attention score를 materialize하면 patch 수의 제곱 항이 생긴다. flash attention 계열 kernel은 전체 score 행렬을 HBM에 저장하지 않아 상수를 크게 줄이지만, patch 열 자체와 QKV·MLP activation은 남는다. “flash attention을 켰으니 고해상도가 안전하다”가 아니라 실제 구현의 saved tensor와 peak allocation을 확인한다.

audio와 video의 길이 tail은 평균보다 peak를 결정한다. allocator reserved memory와 실제 allocated memory를 구분하고, step별 최대 modality budget과 peak를 연결한다. OOM 직전 샘플을 잃지 않도록 asset ID가 아닌 비식별 hash와 shape·budget만 emergency buffer에 남긴다. 재현 fixture는 같은 shape의 합성 tensor로 만들 수 있다.

feature cache의 소유권과 gradient 경계를 구분한다.

frozen tower 출력은 미리 계산해 cache할 수 있다. 이때 cache key에는 원본 asset hash, decode backend와 설정, augmentation, processor·tower revision, dtype, layer 선택, patch merge가 포함돼야 한다. 하나라도 빠지면 오래된 feature를 정상 입력처럼 재사용한다. random crop이나 SpecAugment를 cache 앞에 두면 augmentation 다양성이 고정될 수 있다.

tower를 해제하는 순간 cached detached feature는 gradient 경로를 끊는다. stage 전이와 cache invalidation을 원자적으로 묶고, trainable tower인데 cache hit가 발생하면 실패시키는 assertion을 둔다. connector만 학습하는 stage에서는 cache가 정확한 최적화지만 full fine-tuning에서는 목적함수를 바꾸는 오류다.

cache 저장 dtype도 학습 입력의 일부다. fp16으로 저장한 feature를 bf16 학습에 쓰면 단순 cast로 원래 정보를 되찾지 못한다. cache 생성 dtype과 consumer dtype, compression을 manifest에 기록하고 non-cached reference와 오차 분포를 측정한다.

activation checkpoint 경계를 tower와 connector에 맞춘다.

checkpointing은 activation을 버리고 backward에서 forward를 재실행한다. stochastic augmentation이나 dropout이 재계산 구간 안에 있으면 RNG 보존이 필요하다. media decode처럼 비결정적이거나 비싼 CPU 작업을 checkpoint 함수 안에 넣어서는 안 된다. 재계산 가능한 순수 tensor 연산 경계를 tower block과 decoder block에 둔다.

connector가 작다고 checkpoint 경계를 무조건 tower 뒤에 두면 큰 tower 출력 전체를 보존해야 할 수 있다. 어떤 tensor가 저장되는지 profiler와 autograd saved-tensor hook으로 확인한다. 선택은 FLOPs와 HBM뿐 아니라 host decode 재실행 가능성, collective 위치에도 영향을 준다.

CUDA 실행을 modality pipeline과 연결한다.

CPU 전처리가 끝난 tensor는 pinned host memory에서 비동기 H2D copy를 거쳐야 overlap이 가능하다. 하지만 variable shape를 collate하면서 매번 작은 allocation과 copy를 만들면 copy engine과 launch overhead가 병목이 된다. modality별 contiguous layout, batch tensor 조립 위치, `non_blocking`의 실제 전제인 pinned memory 여부를 확인한다.

영상의 `[B,T,C,H,W]`와 vision tower가 기대하는 flatten 순서가 다르면 transpose 뒤 비연속 tensor가 생긴다. `.contiguous()`는 단순 문법이 아니라 전체 복사를 발생시킨다. profiler에서 transpose, contiguous copy, dtype cast가 tower 시작 전에 얼마나 차지하는지 보고 processor·collator 단계에서 원하는 layout을 직접 만들 수 있는지 검토한다.

stream overlap을 정확성 계약과 함께 본다.

별도 CUDA stream에서 media feature를 계산하고 language 준비와 겹칠 수 있지만 consumer stream이 event를 기다리지 않으면 race가 난다. 기본 stream의 암묵적 동기화에 기대면 코드 변경이나 CUDA graph에서 깨질 수 있다. producer event, consumer wait, tensor 수명을 명시하고 고정 입력의 반복 실행으로 비결정적 mismatch를 검사한다.

여러 image를 개별 kernel로 처리하면 launch가 많아지고, 하나로 합치면 padding과 최대 크기 비용이 늘어난다. 실제 patch bucket별 kernel duration과 occupancy를 측정해 batching 경계를 정한다. GPU utilization 하나는 copy stall, launch gap, memory-bound kernel을 구분하지 못한다.

dtype은 tower별 독립 선택이지만 경계 cast는 공유 비용이다.

vision tower가 fp16, language model이 bf16, connector weight가 fp32라면 forward와 backward의 cast 경로를 그린다. fp16은 범위, bf16은 정밀도, fp32 connector는 bandwidth 비용의 trade-off가 있다. autocast 영역이 함수 경계에서 중첩될 때 실제 output dtype을 test로 고정한다.

gradient scaler가 fp16 tower만 보호하고 bf16 decoder에는 필요하지 않을 수 있다. 하나의 global overflow가 모든 optimizer group step을 skip하는지, group별 처리가 가능한지 확인한다. tower에서만 overflow가 반복되면 batch의 특정 해상도·feature norm과 연동해 원인을 찾는다.

modality dropout과 missingness를 인과적으로 구분한다.

학습 중 media를 확률적으로 제거하는 modality dropout은 모델이 일부 입력이 없을 때 견디게 할 수 있다. 그러나 실제 데이터의 missing media와 의도적 dropout, decode 실패를 같은 special token으로 표현하면 모델은 원인을 구분할 수 없다. 각 상태를 manifest와 mask에 분리하고 loss 포함 정책을 명시한다.

dropout 확률 `p`는 조건 정보량과 gradient 경로를 바꾼다. media가 빠진 batch에서는 tower가 unused가 되고 text shortcut이 강화될 수 있다. 반대로 너무 낮으면 배포의 결측 입력에 취약하다. 정상 media, 의도적 dropout, 자연 결측의 평가 slice를 따로 만들고 calibration까지 비교한다.

missing-not-at-random을 데이터 품질로 본다.

음성이 손상된 샘플이 특정 언어·지역·장비에 몰리면 decode 성공 샘플만 남기는 순간 분포가 바뀐다. 이미지 OCR 실패도 저해상도·특정 문자체계에 편향될 수 있다. 실패율을 전체 하나로 보지 말고 가능한 metadata slice별로 보고, quarantine가 학습 모집단을 어떻게 바꿨는지 기록한다.

결측을 대체하는 zero tensor는 실제 검은 이미지나 무음과 겹칠 수 있다. 별도 validity mask와 missing reason embedding을 쓸지, 샘플을 제외할지 위협·품질 모델에 따라 결정한다. 어떤 선택이든 loss denominator와 sampler probability에 반영된 실제 효과를 측정한다.

counterfactual 평가의 최소 쌍을 만든다.

같은 질문에 올바른 media, 무관 media, 부분적으로 가린 media, media 없음의 네 조건을 만든다. 정답 가능성이 media에 의존하는 항목만 골라 log probability와 답변을 비교한다. 올바른 media와 무관 media 차이가 작으면 grounding이 약하고, media 없음에서만 급격히 거절하면 missing token을 과도하게 학습했을 수 있다.

영상에서는 시간 순서를 뒤집고, 음성에서는 구간 순서를 바꾸며, 이미지에서는 영역을 교환한다. 내용 집합은 비슷하지만 관계가 깨지는 변환이 모델이 구조를 사용하는지 보여 준다. 단순 noise 강건성과 관계 이해를 구분한다.

데이터 권리·개인정보와 학습 artifact를 잇는다.

멀티모달 원본은 텍스트보다 개인정보 밀도가 높을 수 있다. 얼굴, 음성 biometrics, 위치 metadata, 화면 속 계정 정보가 함께 들어온다. 수집 시점의 동의와 사용 범위, 파생 feature·caption·transcript의 보존 정책을 asset lineage에 연결한다. 원본을 삭제해도 feature cache와 checkpoint에서 영향이 남을 수 있다.

삭제 요청은 asset hash로 dataset snapshot, derived crop, audio segment, cached embedding, synthetic caption, 학습 run을 역추적해야 한다. checkpoint 자체에서 기여를 제거할 수 있는지는 23장의 unlearning 문제지만, 먼저 어떤 실행이 해당 asset을 소비했는지 알아야 한다. sample-level consumption log를 무기한 원문과 함께 저장하지 않고 privacy-preserving ID와 접근 통제를 설계한다.

얼굴·음성 redaction을 전처리 함수로 고정한다.

redaction 모델 revision, threshold, 검출 box나 time span, 실패 정책을 기록한다. blur나 beep 뒤에도 OCR·speaker 특징이 남을 수 있으므로 변환 성공을 시각 품질만으로 판단하지 않는다. 원본 접근 권한과 redacted derivative의 권한을 분리하고 cache key가 두 버전을 혼동하지 않게 한다.

redaction은 label 의미를 깨뜨릴 수 있다. 얼굴 표정 질문에서 얼굴을 가리거나 발화자 식별 과제에서 음성을 변조하면 정답이 성립하지 않는다. privacy 변환 뒤 task validity를 다시 판정하고 해당 샘플을 다른 objective로 보내거나 제외한다.

synthetic caption의 provenance를 보존한다.

모델이 생성한 caption은 원본 관측이 아니라 추론된 label이다. 생성 모델·prompt·sampling 설정·후처리·검수 상태를 저장한다. 같은 생성 모델로 학생을 평가하면 오류가 상관돼 점수가 부풀 수 있다. human caption, synthetic caption, OCR transcript를 source type별로 slice한다.

caption hallucination은 media와 모순되는 강한 supervision이 된다. entailment filter 하나에 맡기지 말고 asset 교체 counterfactual, OCR 일치, object grounding 표본 검수를 결합한다. filter 자체의 언어·문화 편향도 추적한다.

모델이 media를 사용하는지 회로 수준에서 확인한다.

attention weight가 image token에 높다는 사실만으로 인과적 사용을 증명할 수 없다. attention output은 value와 이후 residual에 의해 달라지고, 여러 head가 보상할 수 있다. activation patching은 올바른 media 실행의 특정 layer·token activation을 잘못된 media 실행에 옮겨 정답 logit이 회복되는지 본다. 이는 어느 경계가 필요한 정보를 운반하는지 더 직접적으로 묻는다.

connector 출력 전체, 특정 patch, 특정 decoder layer의 residual을 차례로 patch해 최초로 행동이 회복되는 지점을 찾는다. 다만 patching 결과는 선택한 입력 쌍과 개입 방식에 의존한다. 여러 관계 유형과 negative control에서 반복하고, 인과적 충분성과 일반적 설명을 구분한다.

logit lens를 modality 조건 차이로 사용한다.

각 decoder layer residual을 vocabulary head에 투영해 정답 token logit이 언제 상승하는지 비교할 수 있다. 올바른 media와 shuffle media의 logit 차이를 layer별로 그리면 조건 정보가 언제 언어 결정으로 변환되는지 보인다. normalization과 tied embedding을 실제 모델과 동일하게 적용해야 한다.

이 분석은 학습 모니터링의 매 step 지표가 아니라 고정 probe의 진단 도구다. connector 변경 전후에 정보가 더 일찍 들어오는지, 특정 layer에 병목이 생기는지 확인한다. 높은 최종 점수가 같은 내부 경로를 뜻하지는 않는다.

gradient attribution의 단위를 보존한다.

정답 logit을 patch feature에 미분하면 sensitivity를 얻지만 feature scale과 saturation에 영향을 받는다. raw gradient, gradient×input, integrated gradients가 서로 다른 질문에 답한다. attribution map을 원본 pixel로 되돌릴 때 patch merge, crop, tile offset을 역변환해야 한다. 그렇지 않으면 예쁜 heatmap이 잘못된 영역을 가리킨다.

실전 설정 변경을 상태 변화로 번역한다.

`mm_projector_lr`을 올리면 connector update의 크기와 Adam moment 축적이 바뀐다. `vision_tower_lr`을 0에서 양수로 바꾸면 gradient graph, communication, optimizer state와 checkpoint 크기가 바뀐다. `image_aspect_ratio`, 최대 pixel, patch merge는 processor 출력 shape와 decoder token 예산을 바꾼다. `freeze_*`, `tune_mm_mlp_adapter`, LoRA target은 trainable parameter 집합을 바꾼다.

각 옵션 설명에는 기본값만 쓰지 않는다. 읽는 함수, 변경되는 객체, 텐서·파라미터·state, 성능·메모리·처리량 효과, 상호작용, 재시작 호환성, 확인 metric과 실패 증상을 쓴다. CLI flag가 config에 저장됐지만 실제 분기에서 읽히지 않는 dead option도 test로 잡는다.

멀티모달 processor parity를 개수와 가시성의 두 계약으로 닫는다.

멀티모달 입력의 parity를 최종 logits 하나로만 검사하면 서로 다른 두 오류가 상쇄될 수 있다. 첫째는 **무엇을 몇 칸 넣었는가**라는 cardinality 오류이고, 둘째는 **넣은 칸을 attention이 볼 수 있는가**라는 visibility 오류다. PaliGemma의 고정 Transformers revision `550d7b38…`은 이 둘을 서로 다른 함수와 테스트로 드러낸다.

Feature–placeholder 보존은 배치 합계까지 검사한다.

`PaliGemmaForConditionalGeneration.get_placeholder_mask`(`modeling_paligemma.py:151-173`)는 `input_ids`가 있으면 `image_token_id`의 위치를 찾고, embedding 직접 입력이면 image-token embedding과 원소별로 같은 행을 찾는다. 이어서 image-token 수에 hidden width를 곱한 값과 `image_features.numel()`을 비교한다. 이는 단순 shape 검사보다 강하다. `[B,N_v,D]`의 feature를 text embedding에 scatter하려면 batch 전체에서 placeholder scalar 수가 `B×N_v`와 같아야 하기 때문이다.

직접 oracle은 `test_mismatching_num_image_tokens`(`test_modeling_paligemma.py:203-233`)이다. 정상 forward를 먼저 통과시킨 뒤 image 하나를 제거하고 text placeholder는 남겨 명시적 `ValueError`를 기대한다. 이어 한 image/두 placeholder는 실패하고 두 image/두 placeholder는 통과하는 multi-image 대조를 만든다. 따라서 최소 fixture는 `(feature rows, placeholder rows)=(1,1),(1,2),(2,2)` 세 상태로 충분하다. 기대 결과는 각각 `pass, reject, pass`이며, 오류 문자열뿐 아니라 실제 token/feature count도 failure ledger에 남긴다.

그러나 이 검사는 **배치 전체 cardinality**를 닫을 뿐 sample별 bijection을 완전히 증명하지 않는다. 두 표본의 placeholder 수가 `[0,2]`, feature 수가 `[1,1]`인데 flatten 합계만 2라면 전체 수는 맞아도 owner가 틀릴 수 있다. 실무 fixture에는 `sample_id → media_index → placeholder ordinal → feature [start,end)`를 보존하고, 표본 경계에서 prefix sum이 일치하는지 추가한다. 이미지 순서를 바꾼 대조군에서 count는 같고 feature checksum–owner map만 달라져야 이 빈틈을 잡는다.

Padding key 차단은 token type과 독립이어야 한다.

같은 revision의 `test_attention_mask_with_token_types`(`test_modeling_paligemma.py:291-331`)는 eager attention을 token-type ID가 있는 경로와 없는 경로로 각각 실행한다. 그런 뒤 `attention_mask == 0`인 모든 key 위치에 대해 모든 batch·head·query의 attention weight가 정확히 0인지 검사한다. 핵심은 최종 문장 일치가 아니라 attention tensor의 축 계약이다. 검사식 `layer_attn[batch, :, :, pad_key] == 0`은 padding을 **query가 아니라 key 축에서** 차단했는지 고정한다.

golden batch는 길이 `[5,3]`인 두 표본, 오른쪽 padding 두 칸, token-type on/off 두 실행으로 만든다. 각 layer에서 `(B,H,Q,K)`를 기록하고 pad key 열의 최대 절댓값, 유효 key 열의 합, all-masked query의 finite 여부를 비교한다. token-type을 제거해도 pad key 최대값은 0이어야 하지만 유효 위치의 attention 전체가 동일할 필요는 없다. token type 자체가 의미 있는 visibility 규칙을 바꿀 수 있기 때문이다.

이 upstream test는 eager backend만 직접 닫는다. 같은 파일의 FlashAttention2·SDPA padding-free parity tests는 해당 architecture 준비 문제로 skip되어 있다. 그러므로 eager에서 통과했다는 이유로 FA2·SDPA, left padding, packed sequence, export runtime까지 승인하지 않는다. backend별 fixture는 같은 token–media atlas로 별도 실행하고, skip은 ‘성공’이 아니라 명시적인 미검증 상태로 release ledger에 남긴다.

두 폐루프를 합치면 디버깅 순서가 선명해진다. placeholder–feature count가 다르면 attention을 보기 전에 processor·collator·merge를 고친다. count와 owner map이 같고 pad key가 보이면 mask 조립과 backend를 본다. 둘 다 같고 logits만 다르면 tower/projector weight, dtype, position과 kernel 순서로 내려간다. 이렇게 최초 불일치의 층을 고정해야 processor parity가 ‘대충 같은 답을 냈다’는 인상비평을 벗어난다.

변경 전후 diff를 기계적으로 만든다.

실행 시작 시 resolved config, trainable parameter names와 수, optimizer group summary, processor output schema, sample batch shape를 출력한다. 두 run 비교 도구는 raw CLI가 아니라 이 resolved artifact를 diff한다. 환경변수, default 변경, auto-detection이 최종 값을 바꿀 수 있기 때문이다.

학습률 변경은 예상 update-to-weight ratio와 함께 본다. `||ΔW||/||W||`가 connector와 tower·decoder에서 어떻게 달라졌는지 probe step으로 확인한다. loss가 안정적이어도 특정 작은 module의 상대 update가 지나치게 클 수 있다.

옵션 조합의 불법 상태를 조기에 거부한다.

trainable tower와 stale feature cache, projector-only save와 full-model resume 기대, dynamic resolution과 고정 grid position, audio resample off와 고정 sample-rate feature extractor 같은 조합은 실행 전에 거부해야 한다. validation은 단일 옵션 범위보다 옵션 간 제약을 검사한다.

지원 matrix에는 model revision, processor revision, dtype, attention backend, modality 범위, distributed strategy를 행으로 둔다. 한 조합의 smoke test를 유사 조합 전체에 일반화하지 않는다.

독자가 직접 작성할 최소 참조 구현.

참조 구현은 거대한 모델을 학습하지 않는다. 작은 text embedding, 고정된 2×2 patch encoder, 선형 connector, 한 층 causal decoder를 만든다. 이미지 placeholder 하나를 네 feature로 전개하고 답변 두 token만 label로 둔다. forward에서 shape와 mask를 출력하고 한 step 뒤 connector gradient를 손계산의 outer-product 방향과 비교한다.

두 번째 버전은 이미지 두 장을 interleave하고 각 이미지에 고유 basis vector를 준다. 순서를 바꾸면 정답이 바뀌도록 데이터를 만든다. collator가 asset-feature 대응을 보존하는지, block mask가 다른 샘플을 차단하는지, packing 뒤 label이 맞는지 test한다.

세 번째 버전은 tower freeze/cache와 exact resume를 넣는다. cache key 하나를 의도적으로 빼 stale hit를 재현하고, stage 전이 후 cache 사용을 assertion으로 막는다. 한 step 저장·복원 결과를 연속 실행과 비교한다. 이 작은 코드는 대규모 모델의 성능을 흉내 내지 않지만 상태 소유권과 오류 양식을 선명하게 보여 준다.

참조 구현에서 반드시 실패시킬 테스트.

placeholder 수 불일치, feature 순서 교환, label 한 칸 이동, all-masked row, NaN connector 출력, frozen parameter의 optimizer 편입, partial checkpoint missing key, resume sampler cursor 차이를 각각 주입한다. 테스트가 실패를 검출하지 못하면 구현이 성공하는 예제보다 먼저 고친다.

negative test의 오류 메시지는 기대값과 실제값, sample ID, 함수 경계를 포함한다. “shape mismatch”보다 “sample A의 두 번째 image placeholder는 4 patch를 기대했으나 asset B의 6 patch offset을 받았다”가 복구 가능하다.

대규모 실행으로 넘어가는 조건.

작은 fixture에서 forward·loss·gradient·checkpoint·resume가 통과하고, 실제 processor의 소규모 frozen batch에서 두 번 같은 artifact를 만들며, world-size 변화에 따른 목적함수 차이를 설명할 수 있어야 한다. 메모리 장부와 rank workload 예측이 profiler의 작은 실행과 허용오차 안에서 맞아야 한다.

그 뒤에도 대규모 실행은 새로운 증거다. hardware decoder, filesystem tail latency, collective topology, rare long sample은 작은 실험에 없었다. 최초 몇 step에 상세 trace를 켜고 안정화 뒤 sampling rate를 낮춘다. 새로운 실패는 다시 작은 golden fixture로 축소해 회귀 테스트에 편입한다.

성능 측정 전에 닫을 tensor 계약.

멀티모달 모델을 잘 학습했다는 판정은 benchmark 평균 하나가 아니다. 원 신호에서 token·feature로 가는 측정 과정, 서로 다른 공간을 잇는 connector, 위치와 mask, loss의 조건과 분모, optimizer group, 분산 workload, checkpoint lineage, counterfactual grounding이 하나의 설명으로 연결돼야 한다.

구현을 읽을 때는 함수 이름보다 상태의 이동을 따라간다. processor가 만든 grid와 mask를 누가 소비하는가, placeholder가 어디서 확장되는가, 어느 label이 살아남는가, 어떤 loss가 어느 파라미터에 gradient를 보내는가, 어떤 state가 저장되고 resume에서 복원되는가를 묻는다. 각 답을 고정 revision의 함수와 test에 묶는다.

운영에서는 first divergence를 가장 앞 경계부터 찾는다. 원본, decode, transform, tower, connector, sequence, logit, loss, gradient, update, export 순서다. 이 순서를 지키면 “멀티모달이라 복잡하다”는 말이 디버깅을 포기하는 핑계가 되지 않는다.

이 장의 완료 조건은 독자가 처음 보는 모델에서도 동일한 감사를 수행할 수 있는가이다. 아키텍처 이름이 달라도 신호의 표본화, 표현 공간의 정렬, 조건부 확률, 분산 상태의 소유권은 남는다. 그 공통 축과 modality 고유의 차이를 동시에 보존할 때 설명은 실제 코드를 고치고 학습 실패를 복구하는 지식이 된다.

현장 리뷰를 한 시간 안에 시작하는 순서.

처음 10분에는 모델 카드보다 저장소 revision과 processor·model config를 고정한다. special token, image patch, audio feature, video sampling 관련 설정을 resolved form으로 뽑고 source registry의 commit과 일치하는지 확인한다. 이 단계에서 mutable branch나 원격 processor 코드를 섞으면 이후의 줄 좌표와 실행 의미가 모두 흔들린다.

다음 10분에는 training entry point에서 dataset, collator, model forward, loss 호출까지 call chain을 그린다. 함수마다 파일·symbol과 입력 key를 적고, `pixel_values`, `image_grid_thw`, `input_features`, `attention_mask`, `labels`가 생성되고 소비되는 지점을 표시한다. 사용되지 않는 config와 암묵적으로 생성되는 mask도 찾는다.

세 번째 10분에는 한 batch를 수치로 고정한다. 샘플별 원본 길이, patch·frame·text token 수, placeholder 수, 최종 sequence 길이, 유효 label 수를 손계산한다. 코드 로그와 하나라도 다르면 optimizer로 넘어가지 않는다. 평균 loss보다 이 정수들이 입력 계약을 더 직접적으로 증명한다.

네 번째 10분에는 trainable parameter와 optimizer state를 감사한다. tower·connector·decoder별 trainable count, group learning rate와 decay, LoRA target, dtype을 출력한다. 예상 parameter 하나를 골라 loss에서 gradient까지 autograd 경로가 있는지 확인하고, freeze한 parameter의 gradient와 state가 실제로 없는지 확인한다.

다섯 번째 10분에는 저장과 복원을 읽는다. full weight, adapter, projector-only 파일이 각각 무엇을 담고 무엇을 외부 base에 의존하는지 목록화한다. sampler cursor, processor hash, optimizer·scheduler·scaler가 누락되면 exact resume가 아니라 warm restart라고 명명한다. 이름을 정확히 붙이는 것이 과장된 재현성 주장을 막는다.

마지막 10분에는 실패 하나를 주입한다. placeholder 순서를 바꾸거나 processor revision을 바꾸어 어느 assertion과 metric이 먼저 반응하는지 본다. 아무 경보 없이 loss만 달라진다면 관측 경계가 부족하다. 발견한 최초 불일치를 regression fixture로 남기고 owner와 복구 절차를 붙인다.

리뷰 산출물을 코드 변경과 함께 유지한다.

호출 그래프와 텐서 장부를 별도 문서로만 두면 코드가 바뀔 때 낡는다. 핵심 불변식은 unit test와 schema validation으로 옮기고, source span은 commit마다 재검증한다. processor 출력 key나 symbol이 바뀌면 문서 좌표 검사도 실패해야 한다. 설명과 구현이 같은 변경에서 갱신되는 구조가 필요하다.

리뷰에서 확인하지 못한 학습 전용 경로는 추론 코드로 추정하지 않는다. 공개 저장소에 forward만 있고 data mixture나 loss reduction이 없다면 그 경계를 미검증으로 남긴다. 모델 카드의 서술, 논문의 식, 공개 코드에서 직접 확인한 동작을 서로 다른 근거 등급으로 표시한다.

좋은 질문은 숫자와 owner를 요구한다.

“이미지 token은 많습니까?” 대신 “이 batch의 merge 뒤 patch 수 합과 rank별 최댓값은 얼마이며 어느 함수가 계산합니까?”라고 묻는다. “audio padding은 처리합니까?” 대신 “유효 frame mask가 어디서 만들어져 어느 attention과 loss 분모에 들어갑니까?”라고 묻는다. “resume가 됩니까?” 대신 “다음 global sample ID와 augmentation seed, optimizer moment가 연속 실행과 같습니까?”라고 묻는다.

이런 질문은 설명을 어렵게 만드는 것이 아니라 모호함을 제거한다. 답을 찾는 과정에서 데이터, 모델, 분산, 평가 팀의 책임 경계가 드러난다. 답이 없는 항목은 새로운 실험이나 source audit의 작업 목록이 된다.

종합 독해 실습: 한 오류를 세 관점에서 설명한다.

동영상 질문 정확도가 내려갔고 GPU utilization도 낮아졌다고 가정하자. 데이터 관점에서는 새 decoder가 variable-frame-rate timestamp를 다르게 해석해 중요한 짧은 사건을 빼고 긴 clip을 더 많이 decode했을 수 있다. 모델 관점에서는 frame 수 증가가 position scale 범위를 벗어나고 attention이 희석됐을 수 있다. 시스템 관점에서는 rank별 frame budget tail이 collective wait를 늘렸을 수 있다.

세 설명은 경쟁하는 추측이 아니라 같은 사건의 연결된 가설이다. 이전·현재 실행에서 asset hash와 선택 timestamp를 비교하고, 같은 frame tensor를 강제로 넣어 model output을 비교하며, 같은 budget을 rank에 배치해 step timeline을 비교한다. 최초 불일치가 timestamp라면 뒤의 model·system 변화가 모두 설명된다. timestamp는 같지만 output이 다르면 model revision과 position 경계를 본다.

복구는 decoder를 되돌리는 데서 끝나지 않는다. 변환 revision이 cache key에 포함됐는지, 영향받은 feature cache를 격리했는지, 해당 run과 checkpoint lineage를 표시했는지, benchmark와 red-team 영상 slice를 재평가했는지 확인한다. negative fixture에 variable-frame-rate clip을 추가해 같은 회귀를 막는다.

이 실습의 핵심은 원인을 한 층에 가두지 않는 것이다. 입력의 작은 상태 변화가 표현, scheduler workload, 통신 wait, 최종 품질로 전파된다. 반대로 최종 증상에서 각 경계를 역추적하면 거대한 멀티모달 학습도 유한한 계약들의 연쇄로 축소된다. 그것이 이 장 전체에서 사용한 디깅 방법이며 다음 장의 diffusion trajectory에도 그대로 이어진다.

검토자는 마지막으로 세 종류의 동일성을 구분한다. byte 동일성은 원본·token·checkpoint처럼 정확히 같아야 하는 artifact에 쓴다. 수치 동일성은 dtype과 병렬 reduction 때문에 허용오차가 필요한 feature·logit에 쓴다. 의미 동일성은 서로 다른 decode나 kernel이 허용되지만 task 결과와 안전 성질이 유지돼야 할 때 쓴다. 의미 동일성으로 byte 차이를 얼버무리거나, byte 동일성을 보지 못한 채 재현을 주장하지 않는다.

최종 인수 기록에는 각 경계가 어느 동일성 수준을 요구하는지 적는다. 같은 asset에서 processor token이 달라졌다면 의미 점수가 비슷해도 새 lineage다. 동일 token에서 bf16 logit이 작은 허용오차 안에 있고 판정과 calibration이 유지되면 지원 범위 안의 수치 등가일 수 있다. 이 구분이 있어야 최적화와 회귀, 호환 변경과 목적함수 변경을 정직하게 가른다.

독자가 남겨야 할 마지막 한 줄은 모델의 홍보 문구가 아니다. “이 revision에서 이 processor와 이 입력 범위, 이 mask·loss·optimizer·checkpoint 계약을 검증했고, 다음 경계는 아직 검증하지 않았다”라는 문장이다. 그 문장은 짧지만 코드 좌표, fixture, metric, 실패 주입, 담당 owner가 뒤에서 지탱한다. 이 정도의 근거가 마련됐을 때 비로소 멀티모달 학습을 이해했다고 말할 수 있다.

## 21.12 contrastive geometry·resampler·position을 직관으로 연결한다

이미지 embedding `v_i`와 텍스트 embedding `t_i`를 정규화하면 cosine similarity는 단위구면 위 각도의 cosine이다. InfoNCE류 loss는 positive pair의 각도를 줄이고 같은 배치의 negative들과 각도를 벌린다. temperature `τ`는 단순 확률 보정이 아니라 각도 차이를 logit 차이로 확대하는 배율이다. `τ`가 작을수록 가까운 hard negative가 gradient를 지배하고 잘못 짝지어진 caption의 피해도 커진다.

한 방향 loss `L_i=-log exp(v_i·t_i/τ)/Σ_j exp(v_i·t_j/τ)`와 text-to-image 반대 방향을 구분한다. symmetric loss는 두 retrieval 방향을 함께 학습하지만 sample weighting과 distributed gather가 동일해야 한다. batch 안에 같은 이미지의 여러 올바른 caption이 있으면 다른 positive를 false negative로 밀 수 있다. asset family와 semantic-equivalence ID를 global negative mask에 반영할지 결정한다.

global batch가 목적함수를 바꾸는 지점.

rank별 feature를 all-gather하면 negative pool이 world size만큼 커진다. gather가 autograd를 보존하는지, remote feature는 detach되는지에 따라 gradient가 다르다. rank별 local loss를 평균할 때 positive index offset과 uneven last batch를 검산한다. 15장의 collective·gradient ownership과 연결해 world size 변경을 단순 throughput 변화로 기록하지 않는다.

구면 collapse의 관측값.

모든 embedding이 비슷한 방향에 몰리면 positive similarity도 높아 보일 수 있다. uniformity, covariance effective rank, pairwise cosine histogram과 retrieval recall을 함께 본다. modality별 norm은 normalization 전에도 기록한다. connector가 norm을 줄이고 normalization이 이를 숨길 수 있기 때문이다.

### 21.12.1 resampler query를 정보 병목으로 본다

Q-Former나 learned resampler는 많은 media feature에서 고정 개수 query output을 만든다. query 수 `M`은 decoder가 받는 관측 슬롯의 상한이고 cross-attention은 각 query가 media의 어떤 조합을 읽을지 학습한다. `M`을 줄이면 decoder token 비용은 줄지만 작은 객체·짧은 음향 사건이 같은 slot에 겹칠 가능성이 커진다.

attention map을 예쁘게 그리는 데서 멈추지 않는다. query별 entropy, 서로 다른 query attention의 overlap, output covariance rank와 특정 region을 제거한 counterfactual effect를 본다. 여러 query가 동일한 global feature만 읽으면 nominal slot 수와 유효 정보량이 다르다.

고정 query와 동적 token merge의 차이.

고정 query는 입력 크기와 무관하게 출력 수를 일정하게 만들고, 동적 merge는 입력 구조에 따라 token을 묶는다. 전자는 batching과 decoder budget이 예측 가능하지만 고해상도 정보가 같은 병목을 통과한다. 후자는 정보량에 적응할 수 있지만 rank별 길이와 kernel shape가 요동한다. 16장의 scheduler는 sample 수가 아니라 최종 media token budget으로 workload를 배분해야 한다.

resampler pretraining의 target.

caption loss만 쓰면 질문과 무관한 세부가 사라질 수 있다. contrastive, matching, reconstruction과 instruction loss가 query에 서로 다른 gradient를 보낸다. 7장의 embedding geometry와 20장의 multi-objective RL 관점처럼 gradient cosine과 effective rank를 함께 본다.

### 21.12.2 modality position을 공통 attention에 접속한다

텍스트에는 1차원 순서가, 이미지에는 2차원 격자가, 영상에는 시간을 더한 3차원 좌표가, 음성에는 시간과 주파수 축이 있다. 이들을 하나의 decoder sequence에 넣으면 내부 좌표와 interleave 순서라는 두 position이 생긴다. position ID 하나로 둘을 모두 표현하려는 구현은 어느 관계를 보존하고 어느 관계를 잃는지 밝혀야 한다.

Qwen2-VL의 `modeling_qwen2_vl.py`에서 `grid_thw`와 rotary position 계산 경로를 따라가면 processor가 만든 시간·높이·너비 격자가 model attention phase로 변환되는 지점을 찾을 수 있다. 호출자는 placeholder expansion 뒤 text position과 vision grid를 결합한다. `grid_thw`의 원소 곱과 실제 vision token 수, merge factor 뒤 열 길이를 assertion한다.

M-RoPE의 손계산 fixture.

작은 1×2×2 grid에 시간·높이·너비 좌표를 붙이고 각 축에 배정된 rotary dimension을 표시한다. 동일 patch를 tile 순서만 바꾼 입력과 좌표까지 바꾼 입력을 비교한다. token order와 geometric coordinate 중 하나만 바뀌었을 때 attention phase가 예상대로 움직이는지 확인한다.

위치 extrapolation의 범위.

더 긴 video나 더 큰 grid가 실행된다는 사실은 학습 범위 밖 좌표가 의미 있게 일반화한다는 증거가 아니다. phase wrap, scaling과 interpolation을 config에서 확인하고 length·resolution bucket별 성능과 attention entropy를 본다. 10장의 실제 모델 해부에서 position config와 processor limit를 같은 표로 읽는다.

### 21.12.3 tokenizer·chat template가 modality loss를 바꾼다

5장에서 배운 tokenizer·template 계약은 멀티모달에서 placeholder cardinality와 label 위치까지 지배한다. `<image>` 문자열이 단일 special token인지 일반 subtoken 여러 개인지, processor가 어느 단계에서 확장하는지 확인한다. template가 system·user·assistant delimiter를 추가하면 이미지 앞뒤 position과 answer mask가 달라진다.

한 conversation 안에 여러 media가 있으면 메시지 content list의 순서와 rendered placeholder 순서, asset array 순서가 bijection이어야 한다. placeholder 수만 같아도 순서가 바뀔 수 있다. 각 asset에 test feature ID를 넣어 model input embedding에서 round-trip한다.

assistant-only loss의 경계.

template가 generation marker를 잘못 인식하면 user의 위험·정답 text를 label로 학습하거나 assistant answer를 ignore할 수 있다. raw message span에서 rendered character span, token span, feature expansion 뒤 final label span으로 이어지는 mapping을 fixture에 저장한다. 18장의 SFT loss mask 검사와 동일한 형식을 쓴다.

special token resize와 checkpoint.

새 modality token을 추가하면 embedding·LM head resize와 initialization, tied weight가 바뀐다. adapter-only checkpoint가 새 row를 담는지, base tokenizer와 조립할 때 ID가 같은지 확인한다. token ID mismatch는 shape가 맞아도 다른 의미를 주입한다.

멀티모달 mixture를 최적화 변수로 다룬다.

6장의 mixture 설계를 확장해 dataset 선택 확률, modality token 비용, task loss와 data quality를 함께 본다. 이미지 caption 1개와 30초 audio instruction 1개는 sample 하나라는 점만 같고 compute·gradient 정보량은 다르다. optimizer step당 consumed text tokens, patches, audio seconds와 video frames를 상태로 기록한다.

mixture weight를 바꾸면 tower별 gradient 빈도와 optimizer moment가 바뀐다. 드문 audio batch가 긴 간격으로 들어오면 audio connector moment가 오래된 상태에서 큰 update를 받을 수 있다. modality-interleaved microbatch와 modality-homogeneous step을 비교하고 gradient norm·collective idle을 본다.

quality-aware sampling의 편향.

caption-model score나 audio clarity로 high-quality data를 더 뽑으면 쉬운·주류 언어가 과대표집될 수 있다. quality score의 출발 모델과 언어·domain별 calibration을 감사한다. tail coverage와 human-audited random baseline을 유지한다.

curriculum transition의 반증.

connector warm-up 뒤 tower를 해제했을 때 성능 향상이 데이터 stage 변화 때문인지 trainable parameter 변화 때문인지 2×2 ablation을 한다. transition 직전·직후 같은 golden batch의 loss component, gradient와 update를 비교한다. 17장의 exact resume state에 stage event를 포함한다.

tensor parallel과 sequence parallel에서 media token을 나눈다.

language decoder tensor parallel은 projection weight와 activation을 hidden·head 축으로 나눌 수 있다. media token 길이가 커지면 sequence/context parallel도 필요할 수 있다. 이때 image grid와 sample 경계가 rank 사이에 잘려도 attention·position과 gradient가 원래 연산과 같아야 한다.

variable media length에서 rank별 sequence shard가 uneven할 수 있다. padding으로 맞추는지 all-to-all metadata를 쓰는지, position IDs와 attention mask가 어떻게 shard되는지 본다. collective count mismatch는 hang으로 나타나고, 잘못된 offset은 silent quality 오류가 된다.

connector의 parallel ownership.

작은 connector를 replicate하고 output sequence를 shard할지, connector weight도 shard할지 선택한다. replicated connector gradient all-reduce와 decoder sharding 경계를 trace한다. FSDP flat parameter에 tower·connector가 함께 묶이면 별도 learning rate·저장이 깨지지 않는지 15장의 parameter ownership 검사로 확인한다.

rank workload의 예측식.

rank `r`의 비용을 text token, vision patch, audio frame, video frame의 가중합으로 근사하고 profiler로 coefficient를 fit한다. 평균 오차와 tail sample을 본다. scheduler가 이 예측으로 bucket을 만들었을 때 step tail과 padding 비용이 실제로 줄었는지 측정한다.

failure injection을 데이터·수치·분산 세 층으로 확장한다.

데이터 층에서는 corrupt JPEG, 잘못된 sample rate, variable-frame timestamp 역전, placeholder-asset 순서 교환을 넣는다. 수치 층에서는 connector NaN, extreme feature norm, fp16 overflow와 all-masked attention row를 넣는다. 분산 층에서는 rank 한 곳의 decode 지연, 조건부 unused tower, checkpoint shard 누락을 넣는다.

각 실패에는 예상 최초 gate가 있다. corrupt asset은 decode checksum, 순서 교환은 bijection, norm 폭증은 connector finite·norm, unused path는 DDP participation, shard 누락은 commit manifest에서 잡혀야 한다. 최종 benchmark 하락까지 기다리는 test는 너무 늦다.

28·29장 실습과 연결한다.

28장의 single-GPU golden run에는 한 modality씩과 혼합 batch의 exact artifact를 넣는다. 29장의 multi-node injection에서는 동일 manifest의 rank 배분, straggler와 collective 참여를 검증한다. 작은 fixture에서 발견한 failure signature를 cluster runbook의 alert와 owner에 연결한다.

26장 관측성 인계.

metric은 modality budget, decode failure, feature·gradient norm, placeholder mismatch, cache hit·invalid, rank stage time과 checkpoint state를 포함한다. trace exemplar에는 원문 대신 asset ID와 shape·hash를 기록한다. cardinality가 큰 label을 Prometheus metric에 직접 넣지 않는다.

모델 카드의 멀티모달 주장을 실험 계약으로 바꾼다.

“native dynamic resolution”, “audio understanding”, “long video” 같은 문장은 processor limit, 학습 mixture, position range, connector·tower와 평가 protocol로 분해한다. 공개 Transformers implementation의 config·processor·model forward와 model card example을 같은 revision에서 대조한다. example 실행이 training behavior를 증명하지 않는 경계도 명시한다.

Qwen2-Audio의 `modeling_qwen2_audio.py:414-429`의 projector 구성과 `:648-714`의 audio feature 선택·투영 경로, Whisper `modeling_whisper.py:540-648` encoder, LLaVA `llava_trainer.py:165-192` optimizer groups를 독자용 source 좌표로 둔다. 서로 다른 projector가 같은 역할이라고 뭉개지 않는다.

독자의 비교표.

행은 input unit, processor output, tower, merge·resampler, position, decoder conditioning, objective, trainable stages, checkpoint와 verified limits다. 열은 모델 family다. 셀에는 이름보다 tensor·state와 소스 심볼을 쓴다. 새 model이 나와도 같은 질문으로 빈칸을 채울 수 있다.

심화 구현의 검증 기준.

추가된 설명은 기존 개념을 반복하지 않고 대조학습 기하, resampler 병목, multi-axis position, mixture·parallel state와 failure injection을 실제 함수 경계에 연결한다. 각 주제는 독자가 손계산하거나 작은 fixture로 반증할 수 있어야 한다. 설명이 model card의 홍보 문장을 그대로 옮기는 데 머물지 않는다.

이 장을 다시 읽은 독자는 media signal이 token이 되는 전처리, 표현 공간을 바꾸는 connector, 위치와 attention, loss·optimizer, 분산 ownership과 운영 trace를 한 경로로 그릴 수 있어야 한다. 5·6·7·15~18·26·28~30장의 계약을 교차해 어느 층의 변경이 멀티모달 behavior를 바꾸는지 설명할 수 있어야 한다.

음성과 영상 tokenizer의 rate-distortion을 비교한다.

codec tokenizer는 연속 신호를 bitrate가 제한된 이산 index로 바꾼다. 초당 frame 수 `f`, codebook 수 `K`, codebook vocabulary `V_k`라면 이상적 bitrate는 대략 `fΣ_k log2 V_k`다. bitrate를 낮추면 sequence는 짧아지지만 reconstruction distortion과 semantic loss가 커진다. waveform MSE, perceptual audio quality, phoneme intelligibility와 speaker identity가 같은 방향으로 움직인다고 가정하지 않는다.

영상 tokenizer도 spatial·temporal downsampling과 codebook을 통해 bitrate를 정한다. 동일 token/s라도 짧은 motion과 작은 text를 보존하는 정도가 다르다. reconstruction metric과 downstream question answering, generation consistency를 함께 본다. tokenizer 자체의 distortion floor를 backbone 학습 실패로 오인하지 않는다.

codebook utilization의 기하.

embedding table의 사용 확률 entropy와 pairwise distance, residual stage별 explained energy를 본다. dead code가 많으면 nominal vocabulary보다 유효 capacity가 작다. 반대로 모든 code를 균등하게 쓰도록 강제하면 의미 없는 noise까지 분산할 수 있다. rate, distortion과 downstream utility의 Pareto curve를 그린다.

tokenizer revision의 공급망.

codec weight·sample rate·normalization·chunk overlap과 codebook ordering을 checkpoint에 묶는다. tokenizer가 달라지면 동일 ID가 다른 음향·영상 prototype을 뜻한다. text tokenizer보다 더 큰 binary artifact와 decoder가 필요하므로 source·hash와 license를 함께 고정한다.

cross-attention과 early fusion의 gradient 경로.

early fusion은 projected media token을 text sequence에 넣어 self-attention을 공유한다. cross-attention은 language hidden이 별도 media key/value를 읽는다. early fusion은 media-media·text-media interaction이 깊게 섞이지만 sequence 비용과 mask 복잡도가 크다. cross-attention은 조건 경계와 cache가 명시적이지만 어느 layer에 삽입하느냐가 정보 흐름을 제한한다.

gradient 관점에서 early fusion의 media token은 모든 후속 self-attention과 residual을 지나고, cross-attention은 삽입된 block의 query와 media projection을 통해 흐른다. frozen tower·connector·cross-attention adapter 조합별 trainable path를 autograd graph에서 확인한다. 출력 loss가 tower parameter까지 실제로 도달하는지 zero가 아닌 gradient만 보지 않고 norm과 finite difference를 본다.

KV cache와 media condition.

inference에서 media key/value를 한 번 계산해 cache할 수 있지만 training에서는 dropout과 trainable projection 때문에 stale cache가 objective를 바꾼다. cache key와 trainability gate를 17장의 checkpoint·resume 규율에 연결한다. multi-turn 대화에서 새 image가 추가될 때 이전 cache를 부분 재사용하는 규칙도 명시한다.

attention attribution의 반례.

media attention weight가 높아도 value가 거의 상수거나 output projection이 지우면 영향이 작다. media value를 patch하거나 제거한 causal effect와 attention map을 비교한다. head별 결과를 평균 heatmap으로만 보여 주지 않는다.

멀티모달 데이터의 정답 가능성을 검증한다.

질문과 답이 media에서 실제로 도출되는지 검사한다. caption leakage로 text에 답이 그대로 있거나, image가 없어도 상식만으로 답하면 grounding 학습 신호가 약하다. 반대로 crop·audio cut·frame sampling이 evidence를 제거하면 불가능한 label이 된다.

원 media, text-only, shuffled media, evidence-masked media 조건에서 answer probability를 비교한다. human annotator도 media만으로 답할 수 있는지와 ambiguity를 판정한다. data curation score가 model shortcut에 과적합하지 않도록 여러 baseline과 사람 표본을 쓴다.

hard negative 생성.

같은 객체지만 attribute가 다른 image, 같은 화자지만 다른 발화, 같은 장면에서 사건 순서만 다른 video를 고른다. 너무 쉬운 random negative는 connector가 global domain만 구분해도 성공한다. negative generator revision과 false-negative review를 lineage에 둔다.

synthetic instruction의 오류 전파.

caption·QA generator가 hallucination하면 강한 false supervision이 된다. 출발 모델과 prompt, confidence·filter와 human review를 기록한다. 생성 답을 다시 같은 family judge로 검증하면 공유 오류가 남을 수 있다. 24장의 judge calibration과 contamination 규율을 적용한다.

학습 중 modality collapse를 조기에 찾는다.

모델이 media를 무시하고 language prior만 쓰는 collapse는 전체 loss가 감소하는 동안 생길 수 있다. fixed probe에서 correct·shuffled·zero media의 answer log probability gap을 시계열로 본다. gap이 0에 가까워지면 connector gradient, data shortcut과 modality dropout을 조사한다.

tower 출력은 다양하지만 connector 뒤 effective rank가 줄 수 있고, connector는 정상이지만 decoder attention gate가 닫힐 수 있다. tower, connector, 첫 fusion layer, 마지막 residual에서 counterfactual activation distance를 본다. 최초로 차이가 사라지는 경계가 owner다.

modality dominance.

반대로 media가 text instruction을 압도하면 image 안의 우연한 글자나 audio background가 답을 지배한다. text instruction을 바꾼 counterfactual과 media를 유지해 conditional sensitivity를 측정한다. cross-modal conflict set에서 precedence가 policy와 맞는지 본다.

gradient starvation.

loss scale과 data frequency 때문에 작은 modality의 gradient가 optimizer noise 아래 묻힐 수 있다. modality별 isolated gradient와 combined update의 cosine·norm, optimizer moment를 측정한다. 단순 loss weight 증가가 다른 objective를 훼손하지 않는지 13장의 scheduler·optimizer 관점으로 본다.

export에서 processor와 model을 하나의 bundle로 검증한다.

multimodal export는 weight 파일만 옮기는 작업이 아니다. tokenizer·chat template, image processor, audio feature extractor, video sampler, special token과 model config가 같은 bundle이어야 한다. processor가 remote code에 의존하면 revision과 허용 정책을 manifest에 둔다.

export 전후 golden assets에서 decode tensor, feature shape, placeholder expansion, logits와 generation을 단계별 비교한다. backend가 tower를 별도 graph로 compile하거나 projector를 fuse해도 first divergence를 찾을 중간 probe를 남긴다. 최종 text 일치만으로 media path를 검증하지 않는다.

양자화 경계.

language decoder만 4-bit, tower·connector는 bf16일 수 있고 전부 다른 scale을 쓸 수 있다. 어떤 module이 양자화 대상에서 제외됐는지 resolved module list를 저장한다. connector 작은 weight의 양자화 오차가 representation basis를 크게 바꿀 수 있어 modality slice로 평가한다.

runtime의 dynamic shape.

지원 해상도·frame·audio length의 최소·최대와 bucket을 test한다. compile fallback, padding과 OOM을 monitoring에 연결한다. 실행 성공 범위와 품질 검증 범위를 따로 문서화한다.

관측 패널에서 원인과 결과를 분리한다.

입력 패널에는 modality counts·length·decode·cache, 표현 패널에는 norm·rank·counterfactual gap, 학습 패널에는 loss numerator·denominator와 gradient, 시스템 패널에는 rank stage time·memory·collective, 결과 패널에는 grounding·utility·safety를 표시한다. 한 패널의 변화가 다음 패널로 어떻게 전파됐는지 trace ID로 잇는다.

alarm은 원인에 가까운 input schema·placeholder mismatch와 stale cache를 먼저 울리고 benchmark regression은 뒤 확인으로 쓴다. 평균이 안정돼도 tail resolution·codec·language bucket을 본다. 26장의 Prometheus metric에는 bounded labels만 두고 exemplar trace로 sample 세부를 연결한다.

운영 drift의 기준선.

processor·decoder library와 data source가 바뀌면 input distribution 기준선을 새 lineage로 만든다. 기존 threshold를 그대로 적용하기 전에 expected change와 canary를 본다. drift를 model adaptation의 학습 data로 되돌릴 때 production privacy와 test firewall을 지킨다.

종합 실험 설계: 한 모델을 네 축으로 분해한다.

독자는 같은 base에서 connector type, tower freeze, token budget, data mixture를 각각 두 수준으로 둔 fractional factorial 실험을 설계한다. 모든 조합을 대규모 학습하지 않고 작은 golden run과 선택 조합으로 main effect와 큰 interaction을 찾는다. seed와 compute budget, checkpoint selection을 고정한다.

결과는 전체 score 하나가 아니라 grounding counterfactual, modality별 quality, language utility, gradient·rank와 throughput·memory를 함께 본다. connector 변경이 quality를 높였지만 token budget도 늘었다면 compute-matched 비교를 추가한다. tower 해제 효과가 특정 data mixture에서만 나타나면 interaction으로 보고한다.

실험 종료 뒤 인계.

winning config만 남기지 않고 failed combinations의 first divergence와 unsupported range를 보존한다. 28장의 single-GPU exact run, 29장의 multi-node failure injection, 30장의 end-to-end recipe에 같은 config·artifact ID를 넘긴다. 독자는 recipe 숫자를 복사하기 전에 자신의 model·processor 계약과 비교한다.

이 심화 wave가 더한 핵심은 multimodal을 단일 architecture category가 아니라 측정·압축·정렬·조건화·분산의 결합으로 읽는 법이다. 이미지·음성·영상의 고유 구조를 보존하면서도 tokenizer, optimizer, checkpoint와 evaluation의 공통 질문을 적용한다. 어느 페이지에서 시작해도 함수와 tensor, 수학과 운영 증거로 다음 디깅 지점을 찾을 수 있어야 한다.

수학적 직관을 디버깅 질문으로 바꾼다.

구면 위 대조학습에서는 positive angle과 negative density가 핵심이므로 retrieval 실패에서 temperature, false negative와 embedding collapse를 본다. 선형 connector에서는 `ΔW h`가 실제 activation distribution에서 얼마나 큰지 보므로 weight norm보다 output subspace를 본다. resampler에서는 query 수와 attention overlap이 병목이므로 token 절감률만 보지 않는다.

position phase에서는 좌표의 단위와 axis assignment를 보며, codec에서는 bitrate·distortion·downstream utility를 함께 본다. gradient conflict에서는 loss 값보다 공유 parameter의 방향을 본다. 각각의 수학은 장식이 아니라 어느 metric과 반증 실험을 먼저 선택할지 알려 준다.

손계산의 최소 단위.

2×2 image grid의 rotary 좌표, 두 positive pair의 InfoNCE, 한 linear connector outer-product gradient, 두 codebook loss denominator를 종이에 계산한다. 이 작은 계산이 production tensor와 일치하면 복잡한 architecture도 같은 불변식으로 읽을 수 있다.

실무자가 남기는 모델 해부 노트.

첫 페이지에는 소스 리비전과 processor-to-loss call graph를 둔다. 둘째에는 modality별 tensor shape·dtype·mask와 위치를 둔다. 셋째에는 trainable parameter group, cache·checkpoint state를 둔다. 넷째에는 golden assets와 counterfactual·failure injection 결과를 둔다. 다섯째에는 지원 matrix와 미검증 범위를 둔다.

노트의 모든 주장은 함수 또는 config 좌표와 test·fixture를 참조한다. line number만 남기지 않고 symbol과 content hash를 쓴다. 논문 설명과 공개 inference code, 실제 training path에서 직접 확인한 사실을 구분한다.

다른 장으로 가는 독해 경로.

tokenizer·template 오류는 5장, mixture·packing은 6장, embedding geometry는 7장, attention·MoE는 8·9장, 실제 model config는 10장으로 간다. optimizer·precision·parallel·cluster·checkpoint는 11~17장, SFT·preference·RL은 18~20장으로 간다. evaluation·safety·monitoring과 재현 run은 24~30장으로 이어진다.

심화 실험의 승인 조건.

18,000단어라는 숫자는 내용의 증거가 아니다. 이 장의 인수는 독자가 임의의 multimodal model에서 media의 물리 단위가 processor tensor와 token으로 바뀌는 경로, tower·connector의 표현 기하, position·mask, loss와 gradient, distributed ownership과 export를 함수 단위로 추적할 수 있는지로 판단한다.

또한 성능 변화에서 첫 원인을 찾을 수 있어야 한다. asset·decode, feature, connector, embedding sequence, logits, loss, update, serving bundle을 순서대로 비교한다. counterfactual로 model이 media를 실제 사용하는지 반증하고, workload와 collective wait를 구분하며, resume와 cache의 state를 검산한다.

마지막 결과는 architecture 이름을 외운 목록이 아니다. 새 model card의 주장을 tensor 계약과 실험으로 번역하고, 확인하지 못한 training·hardware·modality 범위를 정직하게 남기는 능력이다. 이 능력이 갖춰졌을 때 멀티모달 학습은 신비한 결합이 아니라 수정하고 검증할 수 있는 시스템이 된다.

독립 검토자의 cold replay.

인수자는 저장된 image·audio·video fixture 가운데 하나를 임의로 골라 raw asset hash에서 processor output, tower·connector feature, final embedding과 label mask까지 재생한다. 다른 구현 helper를 재사용하지 않고 token·patch·frame 수와 loss denominator를 다시 센다. 이어 media shuffle과 padding-only 변형이 예상 gate에서 실패하는지 확인한다.

두 번째 검산은 optimizer다. projector, tower와 decoder parameter group에서 이름 하나씩을 골라 learning rate·decay·trainable state, gradient와 update-to-weight ratio를 확인한다. checkpoint를 저장·복원한 다음 같은 global sample과 augmentation seed에서 다음 update가 허용오차 안에 있는지 본다.

세 번째 검산은 분산·운영이다. rank별 modality budget과 stage time으로 collective wait의 원인을 설명하고, export bundle의 processor·tokenizer·tower revision을 attestation한다. cache를 비운 실행과 채운 실행이 같은 feature를 만들며, stale revision을 넣었을 때 load gate가 거부해야 한다.

이 세 검산은 큰 benchmark를 대체하지 않지만 benchmark 변화의 원인을 찾을 기준선을 만든다. 작은 exact evidence와 넓은 통계 평가를 함께 쓸 때만 재현성과 일반화를 동시에 주장할 수 있다.

검토 결과에는 성공한 경로와 함께 실행하지 못한 codec, 해상도·길이, 언어, 분산 전략과 backend를 기록한다. 지원 범위 밖 입력이 우연히 통과한 사실을 품질 보장으로 바꾸지 않는다. 새 조합을 지원할 때 processor·mask·loss·checkpoint의 golden fixture부터 추가한다.

모든 artifact는 model revision 하나가 아니라 data snapshot, processor·tokenizer, model·adapter와 runtime 조합을 parent로 참조한다. 이 조합이 달라지면 새 lineage로 평가한다. 부분 호환성을 주장하려면 변경되지 않은 경계와 다시 검증한 경계를 구체적으로 적는다.

독자는 이 인수 기록만으로 어떤 파일과 함수부터 열고 어떤 숫자를 손계산하며 어느 metric과 trace를 확인할지 결정할 수 있어야 한다. 책의 설명이 실제 디깅 행동으로 이어지는가가 마지막 품질 기준이다.

마지막으로 같은 fixture를 single GPU와 지원되는 distributed 경로에서 비교한다. token·feature·label checksum, loss 분자·분모와 global gradient가 정의한 허용오차 안에 있어야 한다. 차이가 나면 rank sample order, reduction 계수, 조건부 unused parameter와 dtype을 순서대로 본다. 이 엄격한 검증이 29장의 실제 multi-node 장애 훈련과 30장의 종단 recipe 검산으로 직접 이어진다. 모든 검증 결과는 다음 새로운 구현 변경 전의 엄격하고 독립적인 비교 기준선으로 안전하게 장기간 영구 보존한다. 검토자가 다시 확인한다.

## 21.13 sample·fusion·loss·distributed ownership 계약을 고정한다

단일 표본에서 맞았던 계산도 분산 학습에서는 자동으로 보존되지 않는다. sampler가 정한 표본, fusion이 만든 좌표, loss가 센 유효 target, collective가 합친 분자·분모가 서로 같은 사건을 가리켜야 한다. 이 절은 그 네 소유권을 하나의 update 장부로 연결하고, resume나 rank 변경 뒤 무엇부터 비교해야 하는지 정한다.

수학적 직관은 여기서 배치 계약으로 바뀐다. message와 media asset, processor revision, feature row와 target span을 하나의 SampleID 아래 묶고, fusion 방식별 attention 가시성과 loss denominator를 기록한다. 분산 실행에서는 decode·encoder·language stage의 tensor 소유자와 checkpoint writer를 따로 적는다.

멀티모달 학습 sample은 text와 image file의 tuple이 아니다. structured messages, media locators와 immutable checksums, 시간·공간 spans, task target, loss policy와 data lineage로 구성된다. processor가 media를 읽기 전에 message part의 순서와 media reference cardinality를 검증한다. 같은 image가 여러 turn에서 참조되면 reuse인지 독립 occurrence인지 SampleID가 설명한다.

option은 maximum media count, decode policy, resize/clip, placeholder expansion, truncation과 modality dropout이다. 상태는 decoded tensor/codes, coordinate map, feature cardinality, text placeholders와 labels mask다. 효과는 sequence length, encoder compute, objective denominator와 checkpoint/feature cache다. input 검증에서 이 인과를 표로 남긴다.

Golden fixture는 text-only, image 하나/여러 장, audio, video, interleaved media와 corrupt/missing reference를 포함한다. media마다 고유 상수·impulse·frame pattern을 넣어 permutation을 검출한다. random media는 swap되어도 통계가 비슷해 오류를 숨긴다. raw→processor→model input까지 checksum과 offsets를 보존한다.

alignment data failure

caption이 다른 image와 한 칸 밀린 sample, timestamp가 transcript 밖, video answer가 clip 뒤 사건을 참조하는 sample을 넣는다. schema/temporal validator와 contrastive hard-negative가 failure를 구분해야 한다. model loss가 내려간다는 사실은 alignment가 맞다는 증거가 아니다.

duplicate media·caption이 train/eval에 걸치거나 같은 source의 near duplicate가 benchmark에 있는지 4장의 lineage/dedup graph로 확인한다. text와 media 각각의 checksum만이 아니라 pair/segment relation identity를 둔다.

vision tokenizer·encoder를 pixel 좌표에서 feature sequence까지 추적한다.

vision processor는 decode, orientation/color, resize/crop, normalization과 batch pad를 수행한다. patch embedding은 pixel grid를 patches/tokens로 바꾸고 encoder는 position·attention·MLP를 거쳐 feature sequence를 만든다. source card는 processor class/call, vision model `forward`, patch embed, encoder layer와 output selection을 fixed revision으로 잇는다.

image `[B,3,H,W]`, patch `P`이면 단순 grid는 `H/P × W/P`지만 padding, tiling, dynamic resolution과 special tokens가 cardinality를 바꾼다. actual `grid_thw` 또는 image sizes metadata와 feature rows를 맞춘다. placeholder count가 feature count와 같거나 resampler-defined contract를 만족해야 한다.

option은 crop/resize/interpolation, patch, selected layer, CLS 제거, feature strategy, freeze와 gradient checkpoint다. 상태는 pixel coordinates, grid, selected hidden, trainable graph와 saved activations다. 효과는 spatial coverage, feature/placeholder length, memory와 gradient owner다.

vision geometry fixture

corner마다 다른 색, 중앙 impulse, odd size, extreme aspect와 EXIF rotation을 넣는다. crop coordinates와 patch positions를 독립 계산한다. dynamic tiling threshold의 `n-1,n,n+1`에서 grid transition을 확인한다. same output shape라고 crop 의미가 같다고 보지 않는다.

frozen encoder는 parameter gradient가 없어도 projector input gradient 또는 activation 저장 정책이 다를 수 있다. `requires_grad`, train/eval mode, dropout/norm buffer와 optimizer group을 확인한다. adapter/LoRA가 붙으면 exact target modules와 trainable count를 검증한다.

audio·video 경로를 시간 좌표와 rate로 해부한다.

audio processor는 decode, sample-rate/channel, normalization, window/hop 또는 codec tokenization을 수행한다. encoder output frame은 raw time span을 가져야 한다. source card는 feature extractor, codec/encoder `forward`, attention/convolution stack와 projector를 잇는다. declared sample rate와 actual waveform rate가 다르면 시간축 전체가 틀린다.

video 처리는 container decode, actual frame timestamps, sampling, per-frame vision transform와 temporal fusion으로 구성된다. nominal FPS만으로 selected frames를 재현하지 않는다. variable frame rate, duplicate/missing timestamp, clip start/end와 decode backend를 fixture에 넣는다. frame count와 placeholder/video feature span을 맞춘다.

rate-distortion과 objective

audio codec bitrate/frame rate, video sampling FPS와 spatial resolution은 token rate와 정보 손실을 바꾼다. option→codes/features state→sequence/compute·quality effect를 기록한다. bitrate 하나를 바꿔도 tokenizer vocabulary, codebook state와 decoder target이 바뀔 수 있다.

speech transcription loss가 text decoder에만 걸리는지 codec reconstruction/CTC/contrastive loss가 함께 있는지 numerator/denominator를 분리한다. silence, very short clip와 all-masked frames에서 0 denominator 정책을 정한다. timestamp alignment를 한 hop 이동해 detector가 실패하는지 본다.

### 21.13.1 projector·resampler를 좌표 bridge로 검증한다

projector는 encoder width `Cv`를 language hidden `C`로 매핑한다. linear/MLP/gated projector, resampler/query transformer와 pooling은 state와 token cardinality가 다르다. model 이름만으로 구조를 추측하지 않고 config→module construction→checkpoint shapes→forward symbol을 잇는다.

linear projector input `[N,Cv]`와 weight `[C,Cv]`는 output `[N,C]`다. MLP면 activation/intermediate, resampler면 learned queries와 cross-attention state를 추가한다. query count option은 language sequence length와 bottleneck을 바꾼다. projector dtype/autocast와 norm 위치도 기록한다.

projector backward fixture

encoder feature rows마다 서로 다른 basis vector, upstream language gradient마다 다른 constant를 넣어 dInput/dWeight를 손계산한다. wrong media ordering과 transpose를 찾는다. frozen encoder에서는 dInput 계산이 생략될 수 있어도 projector gradient가 맞아야 한다.

resampler는 mask된/padded feature를 attend하지 않고 query output 수를 고정 또는 policy대로 만든다. all-masked media, variable grid와 query count를 test한다. learned query/checkpoint 누락, norm epsilon와 positional mapping mismatch를 failure로 주입한다.

fusion을 early replacement·cross-attention·late objective로 구분한다.

early fusion은 text placeholder span을 projected media features로 대체/삽입해 공동 self-attention sequence를 만든다. cross-attention은 language query가 별 media memory를 읽고 gate/layer pattern을 가질 수 있다. late fusion/contrastive는 separate encoders의 pooled representations와 objective에서 만난다. 세 방식의 mask·state·backward가 다르다.

replacement contract에는 placeholder indices, media feature offsets, order, position IDs와 labels mask를 명시한다. feature count mismatch를 pad/truncate로 조용히 해결하지 않는다. cross-attention은 media attention mask, layer selection, gate와 cached memory owner를 기록한다. late objective는 pair IDs와 negative set을 기록한다.

fusion failure suite

두 images의 feature order swap, placeholder one-off, media padding unmask, cross-attention gate stale/zero와 negative pair duplicate를 각각 주입한다. text-only output이 맞아도 modality path가 틀릴 수 있다. selected media token에 대한 output sensitivity와 projector/encoder gradient를 확인한다.

modality dropout은 intentional random missingness와 real decode failure를 분리한다. dropout mask/RNG와 fallback representation을 state로 저장한다. decode failure를 dropout으로 위장하면 data 품질과 objective distribution을 알 수 없다.

### 21.13.2 loss mask와 multi-objective denominator를 전수한다

causal language loss는 prompt, media placeholders, padding과 assistant target을 mask한다. contrastive, matching, reconstruction, temporal alignment와 router/auxiliary loss가 추가될 수 있다. 각 loss에는 numerator, denominator, weight와 participating samples/ranks를 기록한다. scalar total만 저장하지 않는다.

option은 assistant-only, media reconstruction weight, contrastive temperature, hard negatives와 missing modality policy다. 상태는 labels/masks, pair matrix, negative queue와 running statistics다. 효과는 gradient owner와 committed objective mass다. schedule로 weights가 바뀌면 controller state를 checkpoint한다.

uneven modality batch

rank 0에는 images 2장과 valid labels 10, rank 1에는 text-only labels 2를 넣는다. global token mean과 modality loss denominator를 single-process concatenated oracle로 비교한다. local means rank average는 틀릴 수 있다. zero-participant rank도 collectives에 같은 ordinal로 참여한다.

contrastive all-gather에서 positive index는 global batch offset과 맞아야 한다. variable local batch, dropped corrupt sample와 elastic world size를 넣는다. gather 순서가 pair identity와 다르면 loss는 finite하게 틀린다. stable PairID로 검증한다.

variable-shape batching을 buckets·ragged metadata·kernel로 연결한다.

image grids, audio frames, video frames와 text lengths가 다르면 batch는 padding, bucketing, flatten+offset 또는 nested/ragged representation을 쓴다. batch sampler option은 data order·mixture와 padding waste를 바꾼다. collator는 each sample media offsets, grids/timestamps, sequence placement와 labels를 반환한다.

shape bucket state에는 pending samples, boundaries, max wait와 deterministic tie-break가 있다. checkpoint/resume에서 queue를 빼면 sample ordering과 realized modality mixture가 바뀐다. 6장의 DrawID/PackedSampleID ledger에 media shape class를 추가한다.

batch permutation parity

sample을 개별 처리한 결과와 batch 처리의 non-pad features/tokens/labels를 비교한다. batch 순서를 permutation하고 SampleID로 복원한다. longest neighbor 때문에 다른 sample crop/truncate가 달라지지 않아야 한다. feature flatten offsets와 placeholder offsets를 고유 pattern으로 검증한다.

kernel fast path가 fixed grid/head/length만 지원하면 dispatch/fallback을 shape histogram에서 기록한다. padding으로 fast path를 강제할 때 added compute와 mask correctness를 본다. rare large sample 하나가 batch OOM을 만들면 admission/rebucket policy를 적용하고 조용히 drop하지 않는다.

### 21.13.3 distributed ownership을 pipeline phase별로 쓴다

data parallel은 complete sample replica owner, TP는 language/vision projections와 heads, sequence/context parallel은 joint/media sequence, pipeline parallel은 encoders/projector/language layers, expert parallel은 modality router/experts를 나눌 수 있다. logical tensor마다 global shape, local slice, replica axes, phase와 process group을 둔다.

vision encoder와 language model이 다른 pipeline stages에 있으면 projected features send/recv schema와 microbatch ID를 기록한다. variable shapes는 metadata message와 payload ordinal을 맞춘다. same shape media swap을 SampleID checksum으로 잡는다. encoder freeze가 stage backward communication을 어떻게 줄이는지도 확인한다.

distributed failure

wrong TP/CP group, empty-media rank, unequal contrastive batch, EP zero-token, PP metadata ordinal과 delayed feature transfer를 독립 주입한다. single-GPU reference의 output/loss/selected gradient와 global reconstruction을 비교한다. group size만 같아 silent wrong-axis가 가능하다.

collective byte는 media features, text hidden, contrastive embeddings와 gradients별 numel·dtype으로 계산한다. overlap은 producer/consumer event DAG로 증명한다. asynchronous media transfer 완료 전 fusion buffer를 재사용하지 않는다.

checkpoint와 processor bundle을 하나의 generation으로 묶는다.

checkpoint root는 language, modality encoders/tokenizers, projector/resampler, adapters, optimizer/scheduler/scaler, RNG/data cursor와 processor BundleID를 묶는다. frozen encoder도 exact revision과 config가 필요하다. external registry alias를 load 시 다시 resolve하지 않는다.

processor bundle에는 text/chat template, image/audio/video decode/transform, special placeholders와 feature cache schema가 포함된다. model checkpoint가 required digest를 선언한다. same file names와 shape는 호환성 근거가 아니다. 5장의 bundle migration matrix를 사용한다.

mixed-generation failure

new image processor/old projector, new codec/old embedding, new template/old placeholder map, old feature cache/new model과 optimizer component stale를 각각 섞는다. loader/admission 또는 Golden fixture에서 optimizer commit 전에 실패해야 한다.

topology 변경 checkpoint는 global parameter/optimizer offsets, data sampler와 modality batch queues를 재배치한다. cached media features는 processor generation이 맞으면 portable할 수 있지만 device/compiler derived cache는 rebuild한다. first multimodal update를 reference와 비교한다.

multimodal failure·evaluation evidence dossier.

failure 분기는 data alignment/decode, processor geometry/time, encoder, projector, fusion/mask, objective denominator, distributed owner와 checkpoint generation 순서다. final loss에서만 거슬러 가지 않고 boundary fixture로 first difference를 찾는다. text-only와 modality-required examples를 함께 사용한다.

evaluation은 text prior만으로 맞힐 수 있는 sample, modality counterfactual, corrupted/missing media와 temporal/spatial grounding을 나눈다. image/audio/video를 바꾸고 answer가 변해야 하는 causal fixture를 둔다. benchmark score만으로 media usage를 주장하지 않는다.

dossier에 포함할 증거 파일.

첫 파일은 data/pair lineage, 둘째는 processor coordinates와 feature cardinality, 셋째는 encoder/projector/fusion tensor atlas, 넷째는 loss numerator/denominator와 backward, 다섯째는 distributed ownership/byte, 여섯째는 checkpoint/bundle과 evaluation/failure matrix다. 같은 RunID·generation을 가리킨다.

검토자는 media 하나를 raw byte/timestamp에서 encoder feature, placeholder/fusion, loss와 gradient, checkpoint까지 추적한다. 이어 missing/counterfactual media에서 expected output sensitivity와 failure policy를 확인한다. source function, tensor oracle와 runtime trace가 맞아야 한다.

이 dossier가 닫히면 멀티모달 학습은 “이미지와 text를 함께 넣었다”는 설명을 넘는다. vision/audio/video 좌표가 processor와 encoder를 지나 language objective에 어떤 state로 들어가고, 분산·복구 뒤에도 같은 의미를 유지하는지 증명한다. 새 modality와 backend도 같은 열을 채워 승인한다.

실제 source를 processor→model→loss 호출 사다리로 읽는다.

Transformers 계열 구현을 해부할 때 processor의 `__call__`, image/audio/video helper, multimodal conditional-generation class `forward`, modality encoder, projector/resampler, merge helper와 loss helper를 fixed revision으로 잇는다. class 이름은 선택한 checkpoint config와 Auto class가 실제 load한 symbol에서 가져온다. 다른 model family의 helper를 이름이 비슷하다는 이유로 투영하지 않는다.

source card에는 path/symbol/caller, config guards, input/output shapes, read/write state, fallback와 upstream tests를 기록한다. processor output key가 model `forward` argument와 어떻게 매핑되는지 확인한다. unused key, silently ignored metadata와 default generation은 별 상태다. shape validation이 wrapper, model 또는 kernel 어디서 일어나는지도 기록한다.

function boundary fixture

processor-only output, model embedding 직전, encoder output, projected media, merged hidden과 logits/loss에 hooks 또는 trace를 붙인다. hook가 compile graph를 바꿀 수 있으므로 eager numerical trace와 optimized dispatch trace를 분리한다. same Golden fixture에서 output/gradient parity를 먼저 확인한다.

config의 selected vision layer, feature strategy, projector type, placeholder ID와 loss policy를 하나씩 바꿔 expected 소스 분기가 실행되는지 본다. field가 accepted됐지만 active class가 소비하지 않으면 `unused`로 실패시킨다. source diff와 runtime branch coverage가 같은 답을 내야 한다.

alignment curriculum을 난이도·modality·objective 질량으로 정산한다.

alignment 학습은 caption/pair, instruction/dialogue, grounding, OCR, speech와 video temporal tasks를 단계 또는 mixture로 구성할 수 있다. planned sample weight와 실제 valid-target/objective mass를 구분한다. media decode 실패, variable length와 masks가 realized distribution을 바꾼다.

curriculum option은 stage boundary, source/modality weights, maximum resolution/duration, freeze/unfreeze와 loss weights다. 상태는 schedule horizon, sampler queues, trainable graph와 optimizer groups다. 효과는 gradient owner, compute와 forgetting/alignment다. checkpoint에 controller/sampler와 unfreeze generation을 저장한다.

curriculum knot test

image-only warmup에서 interleaved video로 넘어가는 직전 checkpoint를 만든다. uninterrupted/resumed next SampleIDs, processor limits, trainable modules, optimizer state와 loss weights를 비교한다. newly unfrozen encoder에 moment가 필요한지, scheduler group lr가 무엇인지 확인한다.

modality source 하나가 unavailable하면 automatic reweight가 objective를 바꾼다. planned/available/drawn/valid/committed mass를 기록하고 pause/renormalize 정책을 적용한다. text-only fallback으로 throughput을 유지해 modality 비율 변화를 숨기지 않는다.

contrastive·generative·reconstruction gradient 경로를 분리한다.

contrastive objective는 pooled text/media embeddings와 global negatives, temperature를 사용한다. generative objective는 language target tokens, reconstruction은 pixel/audio/code targets를 사용할 수 있다. total loss scalar 아래 parameter별 gradient 기여가 다르다. loss component별 numerator, denominator와 weight를 저장한다.

parameter group은 vision/audio/video encoder, projector, language decoder, temperature와 modality-specific heads로 나눈다. component 하나씩 backward하거나 autograd hook로 selected ParameterID gradient 기여를 비교한다. total gradient가 weighted sum과 맞는지 확인한다. multiple backward의 graph/RNG 효과를 조심한다.

gradient routing failure

media feature를 detach해 projector만 또는 language만 학습되는 오류, contrastive all-gather에서 remote negatives detach 정책 mismatch, frozen encoder에 accidental gradient와 loss weight double application을 주입한다. final total loss가 finite해도 gradient coverage validator가 실패해야 한다.

contrastive temperature가 learnable이면 scalar parameter, clamp/log parameterization과 optimizer group을 checkpoint한다. global batch size 변화가 negative set과 objective scale을 바꾸므로 DP topology migration에서 one-update oracle을 다시 실행한다.

feature/token cache를 model state와 분리해 검증한다.

offline image/audio/video features 또는 codec tokens는 compute를 줄이지만 processor/encoder artifact와 좌표를 고정한다. cache key는 media checksum, decoder/processor, encoder checkpoint, selected layer/strategy, dtype/layout과 codebook을 포함한다. path와 resize 값만으로는 부족하다.

cached feature를 사용하면 encoder augmentation/dropout과 trainability가 사라질 수 있다. encoder fine-tuning recipe와 호환되지 않는다. option `use_cached_features`는 trainable graph, augmentation distribution, data loader byte와 checkpoint requirement를 바꾼다. effect table에 둔다.

cache failure suite

partial write, checksum mismatch, old processor/new encoder, selected layer mismatch, row order swap와 stale deletion revision을 넣는다. loader는 feature shape만 보지 않고 parent generation과 coordinate map을 검증한다. cache miss 재계산이 가능해도 source media rights/access를 확인한다.

fresh online feature와 cache path를 canonical media에서 비교한다. deterministic processor/encoder면 exact 또는 numerical tolerance, stochastic augmentation이면 cache가 어떤 fixed view를 나타내는지 선언한다. cache hit/miss가 objective를 조용히 두 distribution으로 나누지 않게 telemetry를 둔다.

modality별 numerical ladder와 kernel coverage.

vision patch/projection, audio convolution/attention, video temporal, projector와 language fusion을 FP32/eager target dtype/fused/compiled ladder로 비교한다. 각 rung의 feature RMS/max, cosine, loss와 selected gradients를 기록한다. 한 final loss tolerance에 모든 modality error를 묻지 않는다.

variable grid/frame/sequence에서 actual kernel과 fallback, accumulator, workspace를 trace한다. image grid tail, audio frame tail와 video temporal mask가 tile guard를 자극한다. CUDA 12/13 migration은 14장의 binary/dispatch fixture를 재사용한다. same processor output을 두 environment에 제공한다.

non-finite 진단

pixel/audio normalization, encoder norm/attention, projector, fusion LSE, loss와 backward에서 first non-finite를 찾는다. all-masked media row와 zero-energy audio가 대표 boundary다. FP8를 쓰면 encoder/projector별 amax/scale과 saturation을 본다.

illegal memory access는 next synchronization이 아니라 first kernel을 최소 shape에서 찾는다. odd dynamic image, empty modality, ragged offset과 noncontiguous feature를 격리 process에 넣는다. failure 뒤 context state를 신뢰해 계속 학습하지 않는다.

evaluation을 perception·grounding·reasoning·protocol로 분해한다.

perception은 content 식별, grounding은 spatial/temporal reference, reasoning은 media+text composition, protocol은 chat/tool/output format을 평가한다. 한 benchmark score가 어느 층을 측정하는지 명시한다. text-only prior로 풀 수 있는 item을 modality usage 증거로 쓰지 않는다.

counterfactual fixture는 text를 고정하고 media를 바꾸거나 media를 고정하고 질문/target을 바꾼다. relevant region/frame/audio span을 mask/replace해 output sensitivity를 본다. random corruption과 semantic counterfactual을 구분한다. model이 바뀐 media에 무반응이면 alignment/fusion collapse를 의심한다.

processor-eval parity

evaluation runner가 training과 같은 processor bundle, template, placeholders와 decode/stop protocol을 쓰는지 확인한다. resize/crop, video sampling 또는 audio rate가 다르면 architecture 품질과 input 차이를 분리할 수 없다. EvalRunID에 BundleID와 feature summaries를 둔다.

metric aggregation도 valid examples, missing/corrupt media, language/task strata와 denominator를 보고한다. failure samples를 0점/제외하는 정책이 score를 바꾼다. benchmark contamination과 pair duplication을 4장의 provenance로 검사한다.

production admission과 rollback을 multimodal generation으로 닫는다.

admission은 model/checkpoint, processor/tokenizer, modality encoders/codecs, feature cache, supported media shapes/durations, dtype/backend와 mesh를 검증한다. one component alias를 `latest`로 다시 resolve하지 않는다. workers/ranks가 같은 generation에 합의한 뒤 sample을 받는다.

startup dry-run은 text-only와 각 modality-required fixture를 optimizer commit 없이 실행한다. feature/placeholder parity, selected encoder/projector/fusion, loss denominators, trainable gradients와 distributed ordinals를 확인한다. dry-run data/RNG cursor를 복원한다.

rollback rehearsal

new processor/model bundle로 first update와 checkpoint를 만든 뒤 old parent로 rollback한다. old feature/token cache와 compatible data cursor를 선택한다. new vocabulary/codec/projector state를 old model에 붙이지 않는다. exact rollback이 불가능하면 weights-only warm start와 lost optimizer trajectory를 선언한다.

incident는 media decode/geometry, encoder, projector/cardinality, fusion/mask, loss denominator, distributed owner와 generation의 decision tree로 진단한다. 모든 modality를 text-only로 강등해 증상을 없애는 것은 containment이지 root cause가 아니다. 최소 modality fixture를 추가한다.

최종 운영 서명은 sample 하나를 raw media/text에서 committed UpdateID와 checkpoint까지, checkpoint에서 serving/eval processor까지 왕복한다. option 하나의 state/effect와 failure detector를 답할 수 있어야 한다. 이 왕복이 멀티모달 generation의 완결된 운영 계약이다.

interleaved long-context에서 media 좌표와 eviction을 검증한다.

여러 turn과 여러 media가 긴 context에 섞이면 단순 placeholder count 외에 순서, position, truncation과 cache eviction이 중요하다. message segment마다 text/media span, absolute/logical position, attention domain과 target role을 둔다. context window를 넘을 때 어느 turn/media를 제거하는지 policy를 기록한다.

option은 maximum context, image/video token budget, turn truncation, media pooling과 cache eviction이다. 상태는 retained message tree, media feature offsets, position IDs, KV/media cache와 lost spans다. 효과는 answer evidence, compute와 serving-training parity다. tail truncation이 projector feature 중간을 자르지 않게 admission한다.

long-context fixture

turn 1 image A, turn 2 audio B, turn 3 image C를 넣고 마지막 질문이 A/C를 각각 참조하게 만든다. budget을 경계 전후로 바꿔 retained spans와 placeholder-feature parity를 확인한다. text marker만 남고 media feature가 eviction되는 ghost reference를 금지한다.

training packed/interleaved mask와 serving cache path의 allowed keys를 비교한다. CP가 sequence를 shard하면 media spans의 global position과 owner가 맞아야 한다. cache page reorder에서 media A/C가 swap되지 않도록 stable MediaOccurrenceID를 사용한다.

multimodal checkpoint size와 저장 계층을 정산한다.

checkpoint byte를 language model, each encoder/tokenizer/codebook, projector/resampler, adapters, optimizer, runtime scale/router state와 processor manifest로 나눈다. frozen external encoder를 checkpoint에 복제하지 않으면 immutable required artifact와 resolver가 필요하다. registry 이름만 남기지 않는다.

optimizer state는 trainable graph와 일치해야 한다. frozen vision encoder moment가 남거나 newly unfrozen audio encoder moment가 누락되면 schema drift다. ParameterID별 expected moment count와 dtype을 검증한다. tied/shared modality embedding은 alias와 하나의 state owner를 확인한다.

sharded save failure

PP stages에 encoder/projector/language가 분리되고 TP/EP shards가 있을 때 global keys/offset coverage를 만든다. shard 하나 truncate, wrong stage file, stale processor manifest와 mixed optimizer generation을 주입한다. root manifest publish 전에 모두 거절한다.

topology migration은 global parameter offsets뿐 아니라 sampler/bucket queues와 curriculum state를 옮긴다. external feature cache는 processor/encoder generation이 같을 때만 재사용한다. target topology first multimodal update와 checkpoint round trip을 검증한다.

freeze·unfreeze·adapter 전환을 optimizer 사건으로 기록한다.

vision/audio/video encoder freeze는 `requires_grad`뿐 아니라 module train/eval, dropout/norm buffers, activation graph와 optimizer groups를 바꾼다. projector-only, last encoder blocks, adapters와 full unfreeze를 trainable graph diff로 표현한다. target module patterns의 match count/shape를 startup에서 검증한다.

unfreeze schedule knot에서 new parameter moments, lr/weight decay group과 scheduler horizon을 만든다. 기존 optimizer에 parameter를 append할지 rebuild할지 source/API 계약을 확인한다. checkpoint resume가 knot 전/후 state를 구분해야 한다.

trainability failure

frozen encoder에 gradient, adapter가 wrong modality layer에 match, projector 누락, newly unfrozen state가 optimizer에 없음과 eval-mode dropout inconsistency를 각각 주입한다. selected parameter gradient와 one-step delta, frozen checksum을 검사한다.

quantized/frozen base와 trainable adapter는 dtype/master state와 merge/export를 추가한다. merged checkpoint가 original base+adapter logits/feature와 맞는지, processor requirement가 보존되는지 본다. adapter-only artifact는 base ModelGeneration과 BundleID를 선언한다.

독립 검토자의 최종 네 경로.

첫 경로는 image raw pixels에서 crop/grid, encoder feature, projector, placeholder와 assistant loss까지다. 둘째는 audio/video timestamp에서 frame/code, temporal position, fusion과 gradient까지다. 셋째는 SampleID에서 distributed rank/stream, denominator와 committed UpdateID까지다. 넷째는 checkpoint root에서 model·processor·cache와 eval/serving input까지다.

각 edge에는 source function, option/config, TensorID·shape/dtype, mutable state, artifact generation과 test가 있다. payload를 볼 권한이 없으면 authorized checksum/coordinate verifier를 사용한다. 빈 query를 no effect로 해석하지 않는다.

option blind test

reviewer에게 crop, audio hop, video FPS, query count, placeholder policy, loss mask, freeze 또는 mesh option 하나를 바꾼 child config를 준다. 예상 state diff, sequence/compute/gradient/checkpoint effect와 required invalidation을 작성하게 한다. runtime trace와 negative fixture가 답을 확인한다.

model family 설명이나 benchmark score를 답으로 대신하지 않는다. actual loaded processor/model classes와 checkpoint keys를 확인한다. 실행하지 않은 modality/backend/shape/topology는 `NOT_RUN`이며 필요한 artifact와 invariant를 남긴다.

네 경로와 blind test가 맞으면 raw alignment data에서 durable update와 evaluation까지 lineage가 닫힌다. 이 결과를 22장의 생성 모델 비교, 29장의 multi-node failure와 30장의 end-to-end recipe에 동일 IDs로 넘긴다. downstream이 bundle이나 fusion policy를 바꾸면 affected 경로를 다시 실행한다.

Transformers와 MLX 구현 좌표를 같은 질문으로 읽는다.

Transformers 구현에서는 선택 checkpoint의 processor class, multimodal model `forward`, vision/audio tower, projector/merger, placeholder merge helper와 causal loss helper를 fixed revision에서 고정한다. `processing_*`, `modeling_*` 파일 이름만 적지 않고 실제 loaded class와 symbol, caller와 config guard를 기록한다. Auto class mapping과 custom-code authority도 manifest에 둔다.

MLX 계열 구현에서는 Python model construction, processor/tokenizer adapter, vision encoder, projection/merge와 loss/train step의 실제 symbol을 같은 schema로 기록한다. eager array API가 fused Metal kernel로 내려가는 부분은 input layout, dtype와 selected compiled function을 구분한다. PyTorch state_dict 이름을 MLX parameter tree에 그대로 투영하지 않는다.

공통 질문은 같다. media processor output key/shape는 무엇인가, encoder feature rows는 어떻게 선택되는가, projector가 language hidden에 어떻게 맞추는가, placeholder/attention mask는 어디서 합쳐지는가, labels와 loss denominator는 누가 만드는가, cache/checkpoint에는 어떤 state가 남는가다. backend마다 답이 다른 것을 숨기지 않는다.

cross-framework fixture

동일 immutable text/media와 matched checkpoint conversion이 존재할 때 processor IDs/pixels/features, projector output, merged embeddings, logits/loss를 단계별로 비교한다. exact conversion이 없으면 processor-only 또는 isolated module 범위만 비교하고 full parity를 주장하지 않는다. dtype/kernel tolerance를 사전 정의한다.

parameter conversion에는 global logical ID, transpose/layout, norm and bias, tied alias와 quantization metadata를 기록한다. value checksum만 아니라 one-step selected gradient/delta까지 가능한 범위를 검증한다. framework 중 하나가 inference-only이면 backward cell은 `NOT_RUN`이다.

self-attention·cross-attention mask를 packing과 media validity에서 재구성한다.

early-fusion self-attention predicate는 causal text order, packed document/message boundary, media feature visibility와 padding을 합성한다. media tokens가 prompt prefix처럼 모든 later text에 보이는지, media tokens끼리 bidirectional인지 model contract를 따른다. dense predicate를 작은 fixture에서 직접 그린다.

cross-attention은 language query validity와 media key validity, layer/gate pattern을 가진다. padded image patches, audio frames와 dropped video frames가 key domain에서 빠져야 한다. query가 padding/prompt인지 assistant target인지 attention과 loss mask는 독립이다. loss ignored query도 hidden computation에는 참여할 수 있다.

mask/packing 반례

document A image와 document B question을 같은 packed sequence에 두고 B가 A media를 보지 못하는 expected predicate를 만든다. cumulative lengths, media offsets와 kernel varlen metadata를 dense reference와 비교한다. 한 offset을 이동해 cross-document leakage detector가 실패하는지 본다.

cross-attention gate가 0이면 media effect와 encoder/projector gradient가 어떻게 되는지 확인한다. all-invalid media row의 output/LSE policy를 정한다. fused kernel fallback도 같은 logical mask를 만족해야 한다. forbidden media feature에 큰 sentinel을 넣어 leakage를 선명하게 만든다.

distributed shape contract와 loss routing의 최종 수치 시험.

rank별 batch는 text length, image grids, audio/video frames와 media count가 다를 수 있다. process group collective가 fixed numel을 요구하면 padding+mask, split vectors 또는 ragged protocol을 명시한다. metadata와 payload가 같은 SampleID·ordinal을 가져야 한다. rank-local maximum으로 global buffer를 추측하지 않는다.

TP projector/output, CP joint sequence, PP encoder→language transfer와 DP contrastive negatives 각각에 global/local shape 식을 둔다. 예를 들어 projected features global `[Nmedia,C]`가 TP hidden shard `[Nmedia,C/tp]`인지 replicated인지 source에서 확인한다. collective byte와 backward adjoint를 계산한다.

수치 fixture

DP 2에서 rank 0은 image features 6·labels 8, rank 1은 audio features 3·labels 2를 갖게 한다. language loss sum/count, modality loss sum/count와 weights를 independent concatenated oracle로 계산한다. local means를 평균하지 않는다. zero-modality participant도 expected collectives에 참여한다.

PP metadata ordinal, CP media offset, TP hidden slice와 contrastive positive global index를 하나씩 틀린 negative fixture를 둔다. group size와 tensor shape가 같아도 stable SampleID/MediaOccurrenceID와 global reconstruction이 실패해야 한다. optimizer commit 전에 차단한다.

최종 trace는 planned SampleID에서 processor shapes, rank placements, forward masks, component losses, gradient reductions와 checkpoint offsets를 한 timeline에 둔다. 이 shape·loss contract가 single GPU와 distributed 경로에서 맞고 rank failure 뒤 target topology에서 재생되면 21장의 멀티모달 학습 의미가 완전히 닫힌다.

processor·model bundle의 cold release rehearsal.

rehearsal은 parent checkpoint와 immutable processor bundle에서 시작한다. text-only, image, audio, video와 interleaved fixtures를 같은 model generation으로 처리한다. source functions와 actual loaded classes, processor output keys, feature cardinality, fusion masks, loss components와 selected gradients를 기록한다. one modality가 fallback되면 reason과 changed objective mass를 명시한다.

첫 장애는 image dynamic-grid 경계, audio zero-energy, video variable frame rate와 malformed chat/media relation이다. processor admission 또는 modality-specific invariant가 예상 위치에서 실패해야 한다. 둘째는 feature/placeholder one-off, media order swap, packed cross-document mask와 all-invalid cross-attention이다. fusion 이전 또는 dense-mask oracle에서 차단한다.

셋째는 rank별 variable shapes, empty modality participant, wrong group와 delayed PP feature transfer다. collective ordinal, payload byte, event dependency와 global denominator를 확인한다. optimizer가 읽기 전에 모든 component gradients가 정의한 group에서 완료되어야 한다. rank 하나만 update하는 partial commit은 허용하지 않는다.

넷째는 checkpoint 중 process 종료와 mixed processor/encoder/projector generation이다. incomplete root는 보이지 않아야 하고 loader는 stale feature cache와 BundleID mismatch를 거절해야 한다. target topology에서 first update를 dry-run한 뒤 new root round trip을 수행한다. parent는 commit 확인 전 보존한다.

release 결과표

결과표 열은 modality/task, raw shape/duration, processed shape, tokens/features, selected encoder/projector/fusion, loss numerator/denominator, gradient owner, distributed placement, checkpoint keys, eval counterfactual과 support status다. 각 셀에는 RunID와 EvidenceID를 기록한다. `NOT_RUN`과 `해당 없음`을 0 또는 PASS로 표시하지 않는다.

성능 열은 decode/processor, encoder, projector/fusion, language forward/backward, communication과 checkpoint를 분리한다. cold cache와 warm cache, padding waste와 fallback을 함께 보고한다. 빠른 candidate가 alignment, mask 또는 resume invariant를 깨면 기각한다.

최종 승인자는 임의 media occurrence를 raw coordinate에서 committed update까지, checkpoint에서 evaluation counterfactual까지 두 번 추적한다. 이어 option 하나를 바꾸어 invalidated caches, shape/mask/gradient와 migration 비용을 예측한다. artifact와 trace가 이 답을 재현하면 release를 봉인한다.

release 이후 processor·data drift를 감시한다.

production에서는 modality별 input shape, decode failure, feature·placeholder cardinality, valid loss mass, gradient coverage와 fallback 비율을 release baseline과 비교한다. 전체 평균만 보지 않고 processor/model generation, language, media type와 shape bucket으로 나눈다. 새로운 shape class는 우연히 실행됐다는 이유로 지원 범위에 넣지 않는다.

drift가 감지되면 source data 변화, processor artifact, cache, model dispatch와 objective policy 순서로 paired samples를 만든다. 같은 raw checksum에서 processor output이 달라졌는지 먼저 본다. raw 자체가 다르면 4장 corpus generation 문제로 분리한다. fixed media인데 model path만 달라졌으면 loaded class와 backend를 확인한다.

임시 quarantine 또는 text-only fallback에는 owner, 범위, 만료 시점과 lost objective mass를 명시한다. fallback sample을 정상 multimodal consumption으로 세지 않는다. 수정 뒤 incident 구조를 보존한 비식별 fixture와 기존 modality suite를 함께 실행한다.

새 processor, encoder, codec, CUDA나 mesh가 배포되면 parent 결과를 덮지 않고 child evidence를 만든다. cache invalidation, checkpoint compatibility, distributed shape와 counterfactual evaluation을 재검증한다. 이 유지 절차가 멀티모달 의미를 초기 demo가 아니라 장기 학습 전체에서 보존한다.

모든 재검증 결과는 동일한 SampleID, BundleID, ModelGeneration과 UpdateID로 연결해 독립 검토자가 다시 재생할 수 있게 안전하게 보존한다.

한 장의 이미지가 실제 학습 배치가 되는 호출 사다리.

멀티모달 코드를 읽을 때 가장 흔한 실수는 `model.forward`부터 여는 것이다. 그 시점에는 원본 이미지의 크기, EXIF 방향, 색 공간, resize와 crop 선택이 이미 사라졌다. 먼저 dataset row가 반환하는 `PIL.Image`·경로·바이트 중 무엇이 신뢰 경계인지 확인하고, processor가 만드는 `pixel_values`, `image_grid_thw`, `aspect_ratio_ids` 같은 필드를 실제 loaded class 기준으로 적는다. 같은 `AutoProcessor` 호출이라도 모델 계열에 따라 key와 의미가 다르다.

호출 사다리는 `dataset.__getitem__ → collator → processor.__call__ → image processor preprocess → tensor batch → multimodal model.forward → vision tower → feature selector → projector/merger → language model → loss helper`로 쓴다. 각 화살표마다 입력·출력 shape, dtype, device, padding owner와 예외를 기록한다. 함수 이름이 release 사이에서 바뀔 수 있으므로 저장소 revision, 경로, symbol과 caller를 함께 고정한다. 이 기록은 문서 장식이 아니라 데이터 변환을 역재생하기 위한 실행 지도다.

processor가 dynamic resolution을 쓴다면 `H×W`만으로 feature 수를 예상하지 않는다. resize 뒤 grid, patch merge factor, temporal patch와 special token을 반영해 기대 cardinality를 계산하고 실제 tensor와 비교한다. 예를 들어 raw image 두 장이 한 conversation에 들어가면 각 occurrence의 시작 offset과 feature count를 별도로 보존한다. 합계만 맞으면 두 이미지의 순서가 뒤바뀐 결함을 놓친다.

collator는 list를 단순히 stack하는 도구가 아니다. 대화 template에 media placeholder를 삽입하고, text token을 padding하며, 가변 크기 media를 flatten하거나 묶고, label mask를 만든다. 따라서 collator revision이 바뀌면 모델 weight가 그대로여도 학습 목적함수가 바뀔 수 있다. golden sample은 collator 전 row, processor 출력, 최종 labels를 모두 저장해야 한다.

최소 코드 검산. 첫 표본에서 placeholder occurrence마다 `(MediaOccurrenceID, raw_sha, processed_shape, feature_count, merged_span)`을 출력한다. 이어 merged embedding의 해당 span을 0으로 바꾼 반사실 forward를 수행해 정답 token logits의 변화를 본다. 변화가 전혀 없다면 이미지가 사용되지 않았거나 gate가 닫혔을 가능성이 있다. 변화가 있다는 사실만으로 올바른 grounding을 증명하지는 않으므로 위치를 바꾼 이미지, 무관한 이미지와 원본 이미지를 함께 비교한다.

이미지 해상도 정책은 곧 compute와 의미의 정책이다.

고정 정사각 resize는 batch가 단순하고 kernel shape가 안정적이지만 긴 문서와 작은 글자를 찌그러뜨릴 수 있다. shortest-edge resize와 center crop은 자연 이미지에는 편리해도 문서 가장자리 표를 잘라낼 수 있다. dynamic tiling은 세부를 보존하지만 한 표본이 차지하는 visual tokens와 GPU 시간이 급격히 달라진다. 어느 방식이 우월한 것이 아니라 데이터의 의미 단위와 자원 예산을 함께 맞춰야 한다.

해상도 curriculum을 올릴 때 sample 수만 같게 두면 비교가 공정하지 않다. attention 비용은 sequence 길이에 대해 대략 제곱으로 증가하고, vision encoder와 language decoder가 보는 길이도 서로 다르다. update마다 raw pixels, visual features, language targets, padding tokens와 GPU seconds를 기록한다. loss가 좋아져도 같은 wall-clock과 같은 유효 target 예산에서 좋아졌는지 분리한다.

문서·차트 데이터는 crop provenance가 특히 중요하다. OCR text를 supervision으로 쓸 때 OCR engine revision, bounding box 좌표계, page rotation과 confidence threshold를 남긴다. crop 뒤 좌표를 원본 좌표처럼 저장하면 grounding 평가는 조용히 틀어진다. geometric transform은 affine matrix나 동등한 역변환 정보로 보존해 predicted box를 원본 공간으로 되돌릴 수 있어야 한다.

경계 fixture. 폭과 높이가 patch size의 배수인 이미지, 한 픽셀 모자란 이미지, EXIF 회전 이미지, 투명 채널 이미지, 아주 긴 문서와 빈 이미지를 준비한다. 예상 grid와 mask를 손으로 계산하고, CPU processor와 fast processor가 같은 계약을 만족하는지 비교한다. 구현이 다른 interpolation을 사용하면 pixel equality 대신 허용된 feature·metric 범위를 사전에 정한다.

음성에서는 padding보다 시간이 먼저다.

음성 배치는 길이가 같은 tensor로 보이더라도 실제 시간축이 다를 수 있다. resampling 전 sample rate, channel mix, 시작·종료 timestamp, resampler와 rounding 정책이 feature frame을 결정한다. waveform 16,000개가 항상 정확히 1초라는 가정은 입력 sample rate가 이미 16 kHz로 검증됐을 때만 맞다. container metadata와 decoded sample count가 충돌하면 어느 쪽을 권위로 삼았는지 기록한다.

STFT frame 수는 window, hop, padding과 center 옵션의 함수다. attention mask를 waveform 길이에서 단순 비례 계산하면 마지막 부분의 유효 frame이 하나 어긋날 수 있다. feature extractor가 반환하는 mask 또는 동일 공식을 사용하고, 한 샘플씩 독립 처리한 결과와 padded batch의 유효 구간을 비교한다. 음성 끝의 silence 제거도 label timestamp와 함께 이동해야 한다.

ASR, audio captioning, speech-to-speech는 objective가 다르다. ASR은 transcript token의 conditional likelihood를 주로 최적화하고, captioning은 사건·환경·화자 정보의 선택 문제가 들어간다. neural codec token을 생성하는 speech model은 여러 codebook과 시간 step을 어떤 순서로 예측하는지 명시해야 한다. `[time, codebook]`을 flatten하는 순서가 바뀌면 같은 정수열도 다른 음향을 뜻한다.

멀티스피커와 긴 음성에서는 segment 경계가 의미를 바꾼다. 무작위 chunk가 질문과 답을 갈라놓거나 화자 turn을 자르면 학습 신호가 모순된다. VAD·diarization·forced alignment의 revision과 confidence를 dataset lineage에 넣고, 낮은 confidence 구간을 버렸을 때 언어·억양·환경별 제거율을 본다. 정제는 잡음만 없애는 작업이 아니라 분포를 재작성하는 작업이다.

시간 반사실. 동일 waveform을 유지한 채 transcript timestamp를 한 hop 이동하고 alignment loss가 악화되는지 본다. audio를 뒤집거나 다른 sample의 transcript와 짝지어 모델이 modality를 실제 사용하는지 확인한다. silence·clipping·DC offset·resampling aliasing fixture는 processor 단계에서 탐지되는지, 모델까지 허용된다면 어떤 support status인지 명시한다.

영상은 frame 묶음이 아니라 시간적 표본추출이다.

영상 학습의 입력은 파일이 아니라 sampling policy가 만든 관측이다. uniform `N` frames, fixed FPS, timestamp list, keyframe 또는 scene-aware sampling은 서로 다른 사건을 보존한다. 원본 duration과 frame count만 저장하지 말고 container time base, 선택 PTS, decode backend와 seek policy를 저장한다. variable-frame-rate 파일에서 ordinal index만으로 frame을 재현할 수 없다.

clip이 길어질수록 모든 frame을 쓰기 어렵기 때문에 temporal pooling이나 tubelet이 들어간다. tubelet 크기를 키우면 feature 수는 줄지만 빠른 동작이 평균화된다. spatial merge와 temporal merge를 동시에 적용하면 작은 물체의 짧은 사건이 두 축에서 사라질 수 있다. 모델이 틀린 원인이 reasoning인지 sampling인지 구분하려면 원 사건이 sampled frames에 실제 포함됐는지 먼저 확인한다.

video-text 데이터의 자막은 흔히 완전한 정답이 아니다. 자막 시점이 늦거나 화면 밖 발화를 포함하고, 자동 번역이 고유명사를 바꾼다. clip-caption pair를 만들 때 transcript overlap, shot boundary와 temporal IoU를 보존한다. quality filter가 긴 침묵이나 비영어 발화를 과도하게 제거하는지도 언어·장르별로 감사한다.

batching에서는 frame 수뿐 아니라 각 frame의 grid가 compute를 결정한다. `Σ_i T_i H_i W_i`에 가까운 feature량과 attention length를 함께 bucket한다. worker가 video decode를 오래 잡고 있으면 GPU utilization 저하가 모델 병목처럼 보일 수 있으므로 decode queue time, host memory, H2D bytes와 GPU kernel time을 분리한다. cached frames는 decode 비용을 숨기므로 cold·warm 결과를 따로 보고한다.

시간 순서 fixture. 원본, frame 역순, 중간 사건 제거, 같은 정지 frame 반복과 무관 영상의 다섯 입력을 같은 질문에 평가한다. 원본과 역순의 답이 같아야 하는 질문과 달라야 하는 질문을 분리한다. 이 시험은 단순 이미지 인식만으로 video benchmark를 푸는 shortcut을 드러낸다.

fusion 구조가 gradient가 갈 수 있는 길을 정한다.

early fusion은 media features를 language embeddings 사이에 삽입해 decoder self-attention으로 함께 처리한다. 구현은 단순해 보이지만 sequence 길이와 causal mask, position, KV memory가 직접 늘어난다. cross-attention 구조는 language stream과 media stream을 분리하고 선택 layer에서 연결한다. language self-attention 길이는 유지되지만 media KV와 gate, 별도 mask가 생긴다. late fusion이나 contrastive dual encoder는 생성 decoder가 media token을 직접 보지 않을 수도 있다.

구조를 비교할 때 parameter 수나 benchmark score만 나열하지 않는다. media encoder까지 gradient가 도달하는 경로, projector·resampler가 압축하는 정보, modality feature가 어느 layer에서 residual에 더해지는지 그린다. gate가 sigmoid 또는 tanh로 초기화되면 초기값이 gradient scale을 크게 바꾼다. gate가 정확히 0이고 곱셈 바깥의 경로가 없다면 encoder 쪽 gradient가 차단될 수 있으므로 수식을 실제 코드 순서대로 미분한다.

Perceiver-style resampler의 learned queries가 `M`개라면 가변 media feature `N`개를 고정 `M`개로 압축한다. 이는 batching을 안정시키지만 `M`이 정보 병목이다. attention map 하나를 해석으로 단정하지 말고, query 수를 바꾼 ablation과 특정 영역 삭제 반사실을 함께 본다. projector output norm과 decoder residual norm 차이가 크면 normalization 또는 scale parameter가 사실상 gate 역할을 할 수 있다.

모델을 freeze하는 단계별 curriculum도 fusion 구조와 결합된다. encoder·decoder를 얼리고 projector만 학습하는 alignment 단계, decoder 일부와 adapter를 여는 instruction 단계, 전체 또는 일부 tower를 미세조정하는 단계는 optimizer state와 gradient graph가 다르다. 단계 전환 때 단순히 `requires_grad`를 바꾸면 optimizer group에 새 parameter가 없거나 scheduler age가 잘못될 수 있다. 새 optimizer를 만들지 state를 이관할지 정책을 명시하고 checkpoint boundary에서 검증한다.

멀티모달 손실은 단일 숫자로 평균내기 전에 분해한다.

language generation, contrastive alignment, image reconstruction, codec prediction, bounding-box regression과 temporal localization은 단위와 분모가 다르다. 총손실 `L=Σ_m λ_m S_m/C_m`에서 각 `S_m`은 유효 항의 합, `C_m`은 전역 유효 개수다. rank-local mean을 다시 평균하면 modality가 없는 rank와 긴 표본이 잘못 가중된다. 먼저 sum과 count를 정의한 process group에서 reduce한 뒤 weight를 적용한다.

contrastive loss는 global negatives를 쓰는지 local negatives만 쓰는지에 따라 목적함수가 달라진다. all-gather한 embeddings에 gradient가 흐르는지, remote features를 detach하는지 source에서 확인한다. batch size를 늘렸다는 말은 negative pool이 실제로 늘었을 때만 성립한다. duplicate caption과 동일 이미지의 여러 crop을 서로 negative로 두면 false negative가 생기므로 group ID를 mask에 반영한다.

bounding box와 segmentation은 좌표 normalization, invalid region과 empty target 처리가 핵심이다. augmentation 뒤 box를 변환하지 않거나 crop 밖 객체를 남기면 손실이 잘못된다. empty target batch가 NaN을 만들지 않더라도 분모 1로 위장될 수 있다. modality·task별 valid count를 관측하고 zero-count는 정확히 0 contribution이되 collective 순서는 유지한다.

weight `λ_m`을 자동 조절하는 기법을 쓰더라도 문제를 없애는 것은 아니다. gradient norm balancing, uncertainty weighting이나 schedule은 새 state와 hyperparameter를 만든다. task별 selected parameter gradient의 cosine과 norm, loss scale, 유효 count를 함께 기록한다. 한 task loss가 내려가면서 다른 task의 grounding이 붕괴하면 총손실 평균은 이를 숨길 수 있다.

멀티모달 데이터 오염과 안전은 픽셀·시간축까지 내려간다.

텍스트 중복 검사만으로 이미지·영상 contamination을 찾을 수 없다. exact file hash는 재인코딩과 crop을 놓치고, perceptual hash는 threshold에 따라 unrelated image를 묶을 수 있다. OCR text, image embedding, temporal fingerprints와 metadata를 단계적으로 사용하되 각 detector의 false-positive 표본을 사람이 감사한다. benchmark image가 caption만 바뀌어 train에 들어간 경우도 같은 evaluation family로 연결한다.

개인정보는 EXIF, 얼굴, 번호판, 화면 속 문서, 음성의 화자 정보와 transcript에 동시에 존재한다. 한 modality만 비식별화하면 다른 modality가 원정보를 복원할 수 있다. redaction mask와 audio mute interval을 원 좌표계에 저장하고, caption·OCR·ASR 파생물도 함께 갱신한다. 삭제 요청은 raw asset뿐 아니라 crop, feature cache, embeddings, packed shard와 checkpoint 영향 범위를 추적해야 한다.

안전 필터가 학습 전에 특정 집단이나 환경의 표본을 더 많이 제거하면 모델 성능 격차가 커질 수 있다. filter score 분포와 disposition을 언어, 피부색·억양 같은 적법하고 승인된 감사 slice, 해상도와 source별로 본다. 민감 속성을 무분별하게 새로 추론하지 않고, 정책상 허용된 annotation과 aggregate만 사용한다.

멀티모달 red team은 prompt injection이 이미지 속 글자나 음성, 영상 frame에 숨는 경우를 포함한다. 모델이 media text를 instruction으로 취급하는 경계, system message와 충돌할 때 우선순위, tool call로 이어지는지를 평가한다. 공격 표본은 실제 유해 payload를 불필요하게 보존하지 않고 최소 재현 fixture와 접근 통제를 둔다.

무엇을 모니터링해야 원인을 찾을 수 있는가.

운영 대시보드의 첫 줄은 modality별 admission count, decode failure, raw size·duration, processed shape와 feature count 분포다. 둘째 줄은 processor·encoder·projector·fusion·decoder별 latency, GPU memory, kernel fallback과 padding waste다. 셋째 줄은 task별 loss sum/count, selected gradient norm, gate와 feature norm이다. 넷째 줄은 grounding·temporal·speech·text-retention 평가와 safety slices다.

이 지표를 모두 평균 하나로 만들지 않는다. ModelGeneration, ProcessorBundleID, dataset generation, modality, shape bucket와 rank를 공통 label로 갖되 고카디널리티 SampleID는 metric label이 아니라 trace exemplar에 둔다. Prometheus에는 집계 가능한 histogram과 counter를, object store에는 sample trace와 artifact를 저장한다. metric에서 trace로 이동할 exemplar key를 남긴다.

증상에서 원인으로 가는 순서도 정한다. GPU idle이 늘면 먼저 decode queue와 H2D를 보고, visual token이 늘었으면 resolution curriculum과 sampler를 확인한다. loss spike가 특정 modality에만 있으면 valid denominator, processor drift, corrupted media와 gradient scale을 본다. 모든 modality가 동시에 악화되면 optimizer·checkpoint·distributed state 같은 공통 경계를 먼저 의심한다.

배포 판정. 새로운 processor나 codec은 평균 benchmark 상승만으로 승인하지 않는다. golden media의 exact 또는 bounded transform, variable-shape batch, modality-absent rank, checkpoint resume, stale-cache rejection과 counterfactual grounding을 통과해야 한다. 실패가 한 번이라도 관측된 shape·backend는 수정과 재검증 전까지 support matrix에서 제외한다.

## 21.14 공개 모델 구현을 source 좌표에서 해부한다

이 절은 모델 카드를 반복하지 않는다. 고정 revision의 processor와 modeling source에서 cardinality assertion, grid·mask 생성, cross-attention 가시성, resampler 출력과 labels 전달 위치를 찾아 앞 절의 계약에 꽂는다. 공개 코드가 추론 경로만 제공하면 학습 loss와 backward는 `SOURCE_ONLY`로 확대하지 않는다.

하나의 공개 checkpoint와 고정 Transformers revision을 고른다. 실행이 어려우면 source와 upstream tests만으로 정적 dossier를 만들되 실행 결과처럼 쓰지 않는다. processor class의 `__call__`에서 시작해 image 또는 audio preprocess, model `forward`, feature extraction, projector·merge와 loss까지 실제 symbol을 연결한다. config가 선택하는 branch와 test가 덮는 shape를 옆에 적는다.

첫 계산은 feature cardinality다. raw 크기와 processor 설정에서 grid 또는 frame 수를 손으로 구하고 special·merge 요소를 더한다. 둘째는 placeholder 뒤 text token 위치와 labels를 계산한다. 셋째는 유효 language target, modality target의 loss sum/count를 계산한다. 넷째는 tower·projector·decoder 중 어느 parameter가 gradient를 받아야 하는지 예상한다.

그다음 네 개의 결함을 심는다. media 순서를 바꾸고, placeholder를 하나 줄이고, padding mask를 한 칸 늘리고, processor bundle ID를 stale cache와 섞는다. 각 결함이 어느 함수·invariant에서 처음 막혀야 하는지 적고 upstream test가 실제로 보장하는 범위를 확인한다. 단순 exception이 아니라 잘못된 표본이 optimizer update에 들어가기 전에 차단되는지가 판정 기준이다.

마지막에는 같은 SampleID를 dataset manifest, processor trace, model tensors, loss contribution, distributed placement와 checkpoint lineage에서 찾는다. 어느 링크도 파일명이나 자연어 추측에 의존하지 않아야 한다. 이 해부를 이미지 한 건, 음성 한 건, 영상 한 건과 interleaved 한 건에 반복하면 멀티모달 학습을 데모가 아니라 검증 가능한 시스템으로 이해하게 된다.

### 21.14.1 cross-attention 가시성과 placeholder 치환을 구분한다

Transformers 고정 revision `550d7b3`에서 Mllama 계열을 읽으면 vision embedding을 text embedding 사이에 단순 삽입하는 그림이 맞지 않는다. image tower가 만든 tile별 feature는 별도 media sequence로 남고, language layer의 cross-attention이 이를 본다. 따라서 text token 수와 vision token 수를 더한 하나의 causal sequence만 그리면 mask와 memory를 잘못 계산한다.

processor와 model 경계에서 중요한 값은 image 수, tile 수, tile별 aspect 정보와 text token별 media 가시성이다. cross-attention mask는 단순 `[B,L_text]`가 아니라 어느 text 위치가 어느 image·tile을 볼 수 있는지를 표현한 뒤 실제 vision-token 축으로 확장된다. text prompt의 이미지 구간과 tile ordinal이 어긋나면 tensor shape가 맞아도 질문이 다른 tile을 보게 된다.

이 구조에서 label mask와 cross-attention mask는 독립이다. user prompt token은 loss에서 제외되어도 image feature를 읽고 assistant response의 hidden state에 영향을 줄 수 있다. 반대로 assistant target이 유효해도 해당 위치의 cross-attention 가시성이 닫혀 있으면 답은 사실상 text-only로 학습된다. golden fixture는 두 mask를 나란히 펼쳐야 한다.

gradient 소유권도 분리한다. language loss에서 cross-attention projection과 vision features까지 이어지는 경로, frozen vision tower에서 멈추는 경로, adapter가 붙은 language projection의 경로를 실제 `requires_grad`와 optimizer parameter identity로 확인한다. “vision을 freeze했다”는 설정 문자열만으로 cross-attention 모듈까지 얼었다고 추정하지 않는다.

Llama 4: feature 수를 줄이는 reshape는 정보 배치를 바꾼다.

같은 고정 소스의 Llama 4 vision 경로에는 공간 feature를 재배열해 token 수를 줄이고 channel 폭을 늘리는 pixel-shuffle 계열 변환이 있다. 이는 단순 pooling과 다르다. 이웃한 공간 위치의 값이 channel 축의 서로 다른 slice로 이동하므로 이후 projector의 weight가 공간 sub-position을 구분할 수 있다. 압축률 `r`이면 대략 token 수는 `r²`배 줄고 입력 channel은 `r²`배 늘어난다.

손으로 작은 `[H=4,W=4,C=1]` grid를 만들고 각 위치에 고유 번호를 넣은 뒤 reshape 결과의 token·channel 배열을 적어 보면 layout을 가장 빨리 이해할 수 있다. `reshape→permute→reshape` 순서 중 permute 하나가 바뀌면 shape는 그대로지만 공간 의미가 섞인다. upstream test가 값 배치까지 확인하는지, 아니면 출력 shape만 확인하는지 구분한다.

processor가 확장하는 placeholder 수는 압축 뒤 feature 수와 일치해야 한다. raw patch 수를 기준으로 placeholder를 만들면 mismatch가 발생하거나 일부 feature가 버려질 수 있다. dynamic resolution, class/register token 제거와 newline·separator token이 있으면 식에 포함한다. 각 단계의 cardinality를 별도 assertion으로 두면 어느 변환에서 한 칸이 틀렸는지 찾을 수 있다.

vision 공동학습에서는 이 재배열 자체는 parameter가 없더라도 backward의 layout을 결정한다. projector gradient를 inverse layout으로 되돌렸을 때 원 grid의 어느 patch가 큰 신호를 받는지 시각화할 수 있다. 이를 attention heatmap과 동일시하지는 않되, crop이나 small-object 표본이 실제 spatial region에 gradient를 보내는지 확인하는 진단으로 쓴다.

### 21.14.2 Qwen 계열의 grid·3축 position을 보존한다

Qwen2.5-VL 계열은 image와 video를 동일한 정적 token 수로 만들지 않는다. processor가 반환하는 grid의 temporal·height·width 축은 feature cardinality뿐 아니라 multimodal rotary position을 만드는 입력이다. 따라서 `pixel_values`만 cache하고 grid metadata를 잃으면 같은 feature rows를 text sequence의 올바른 위치 좌표에 놓을 수 없다.

3축 M-RoPE를 직관적으로 보면 text token은 진행 방향이 하나지만 media patch에는 시간·행·열 세 좌표가 있다. 세 좌표를 head 차원의 서로 다른 부분에 회전시켜 상대 위치 신호를 준다. 구현에서는 각 축의 position IDs가 어떤 순서와 channel partition으로 들어가는지 확인한다. 논문의 이름만 보고 균등 3분할이라고 추정하면 안 된다.

image와 video가 한 prompt에 섞이면 occurrence별 `grid_thw`, merge size, feature rows와 placeholder span이 같은 순서를 가져야 한다. 합계가 같아도 image A와 video B의 grid를 교환하면 3축 position은 틀리지만 matrix multiplication은 실행될 수 있다. ordinal과 media type을 포함한 stable identity를 merge 직전에 검증한다.

video에서는 temporal patch와 frame sampling이 겹친다. 원 PTS에서 frame을 고르고 processor가 temporal merge를 한 뒤 M-RoPE time coordinate를 만든다. frame index만 남기면 variable FPS의 실제 시간 간격이 사라지고, position이 물리적 시간을 표현한다고 과장할 수 있다. 코드가 사용하는 좌표가 ordinal인지 timestamp-derived인지 정확히 설명한다.

Qwen3-VL deepstack: vision signal이 한 번만 들어간다는 가정을 버린다.

Qwen3-VL 구현을 읽을 때 projector 출력이 language embedding에 한 번 주입되고 끝난다고 가정하면 deepstack 경로를 놓친다. vision tower의 여러 깊이에서 선택한 feature가 language model의 지정 layer에 추가로 들어갈 수 있다. 어느 vision layer가 어느 language layer에 연결되는지는 config와 실제 forward branch가 정한다.

이 구조는 activation memory와 gradient 경로를 바꾼다. 여러 vision intermediate를 backward까지 보존하거나 재계산해야 하며, language layer별 injected tensor의 dtype·shape가 맞아야 한다. base embedding 주입만 검사하면 deepstack feature 순서가 바뀌거나 한 layer가 누락된 결함을 발견하지 못한다.

fixture는 각 selected vision depth에 서로 다른 sentinel scale을 넣고 지정 language layer에서만 그 신호가 나타나는지 확인한다. 실제 model weight를 바꾸기 어렵다면 hook 또는 작은 mock module로 routing을 검증한다. layer index가 0-based인지, gradient checkpointing wrapper 뒤 module identity가 유지되는지도 본다.

MoE language backbone과 결합되면 vision injection은 router 입력 분포에도 영향을 줄 수 있다. image·text-only 표본별 expert selection, load balance와 dropped token을 비교하되 인과를 곧바로 확정하지 않는다. 같은 text와 media ablation의 paired trace에서 injection 전후 hidden norm과 router logits를 연결한다.

### 21.14.3 Perceiver resampler가 고정 media budget을 만든다

Idefics2 계열은 가변 patch feature를 learned latent queries로 읽어 고정 개수의 media latents로 압축한다. cross-attention에서 query 수 `M`은 출력 sequence 길이를 결정하고, raw patch 수 `N`은 key/value 길이를 결정한다. decoder 비용은 안정되지만 resampler 자체 비용과 정보 병목은 원 해상도에 따라 달라진다.

같은 이미지에서 해상도를 올려도 출력 latent 수가 같다면 “visual token 수가 같으므로 compute가 같다”고 말할 수 없다. vision encoder와 resampler의 `M×N` attention은 커진다. 반면 language decoder의 media span은 고정될 수 있다. 단계별 FLOP·memory를 나눠야 이 구조가 왜 긴 이미지 배치를 안정시키는지, 어디에서 비용을 지불하는지 보인다.

learned queries는 이미지마다 동일 parameter에서 시작하지만 key/value에 따라 다른 요약을 만든다. query ordinal에 고정 의미가 있다고 단정하지 않는다. crop 삭제와 patch permutation, query 수 변경에서 downstream grounding이 어떻게 변하는지 본다. resampler attention weight만으로 객체 슬롯이라는 해석을 확정하지 않는다.

checkpoint에는 resampler query와 projection·normalization state가 반드시 포함돼야 한다. vision tower와 language model만 이관하고 connector를 새로 초기화하면 shape는 맞아도 정렬이 사라진다. conversion manifest는 connector keys와 transpose, dtype, 누락 정책을 별도 그룹으로 둔다.

Idefics3: pixel mask가 position과 유효 patch를 함께 정한다.

Idefics3의 variable-resolution 경로에서는 padded pixel tensor만으로 실제 이미지 경계를 알 수 없다. pixel mask가 어느 patch가 유효한지 정하고, patch position을 만드는 데도 관여한다. 두 이미지가 같은 padded `H_max×W_max` tensor여도 유효 영역이 다르면 feature sequence와 위치 의미가 다르다.

collator에서 pixel padding을 한 뒤 mask를 잃거나 resize 전 크기로 mask를 만들면 padded 검은 영역이 실제 patch처럼 encoder에 들어간다. 검은 배경 이미지에서는 값만 보고 오류를 찾기 어렵다. padding 영역에 큰 sentinel을 넣되 mask가 있을 때 출력의 유효 feature에 영향을 주지 않는지 검사한다.

connector가 유효 features를 language placeholder에 넣을 때 batch별 feature count가 다를 수 있다. flattened batch에서는 sample offsets와 counts가 정확해야 한다. 첫 sample의 마지막 patch가 둘째 sample placeholder로 넘어가는 one-off 결함은 전체 합계 assertion을 통과한다. sample별 prefix sum과 occurrence identity를 비교한다.

position을 2D grid에서 만드는 구현은 원본 aspect와 padded grid의 관계를 보존해야 한다. row-major flatten 순서, invalid patch 제거 전후 position 배열을 작은 비정사각 fixture에서 손으로 그린다. 정사각 한 장만 있는 upstream test는 이 경계를 증명하지 못한다.

MLX-VLM: 추론 포팅과 실제 학습 경로를 구분한다.

MLX-VLM 고정 revision `2b31570`에는 단순 inference wrapper를 넘어 SFT, ORPO, LoRA와 parameter 선택 경로가 존재한다. 따라서 “MLX는 추론만 한다”거나 반대로 “PyTorch recipe와 동일하다”는 두 일반화 모두 피한다. 실제 trainer가 만드는 batch, completion mask, `value_and_grad`, accumulation과 optimizer update 순서를 읽어야 한다.

MLX의 parameter tree와 PyTorch `state_dict`는 이름·layout·tied alias 표현이 다를 수 있다. conversion에서 key 문자열만 맞추지 말고 linear weight transpose, normalization, vision projector와 adapter insertion 위치를 기록한다. converted base와 adapter를 합친 뒤 동일 fixture의 processor output·selected activations·logits를 단계별로 비교한다.

full fine-tune, language LoRA, vision/projector unfreeze는 서로 다른 trainable tree를 만든다. tree flatten 결과에서 parameter 이름, shape, element 수와 optimizer state allocation을 계산한다. projector를 연다고 했지만 selection predicate가 language prefix만 고르면 실제로는 얼어 있을 수 있다. 반대로 vision tower가 실수로 포함되면 unified memory와 update 비용이 급증한다.

ORPO 같은 preference 학습에서는 chosen/rejected가 동일 media를 가리키는지 확인한다. 이미지가 달라지면 preference는 답변 차이와 media 차이를 동시에 학습한다. completion log-prob의 token mask와 reduction, odds-ratio term과 beta를 실제 코드 수식으로 재계산한다. 이름이 같은 알고리즘도 framework마다 averaging과 sign이 다를 수 있다.

contrastive pretraining에서 생성형 instruction tuning까지.

CLIP류 dual encoder는 paired image·text embedding의 유사도를 batch 내 분류 문제로 만든다. 정규화된 embedding `u_i`, `v_j`와 temperature `τ`에서 logit은 `u_i·v_j/τ`다. image→text와 text→image 두 cross-entropy를 평균할 수 있다. 여기서 batch와 all-gather가 negative set이므로 분산 topology가 목적함수의 일부다.

SigLIP류 pairwise sigmoid objective는 모든 pair를 하나의 softmax 분모로 묶지 않고 positive·negative pair를 독립 logistic 항으로 본다. 두 objective를 “대조학습” 한 단어로 합치지 않는다. temperature/bias, global negatives, duplicate mask와 reduction을 수식과 코드에서 각각 확인한다. 같은 global batch라도 communication 방식과 gradient가 다르다.

caption pretraining은 visual condition에서 다음 text token을 예측한다. 짧고 일반적인 caption만 많으면 객체 존재는 배우지만 OCR, 공간 관계와 긴 reasoning은 약할 수 있다. interleaved document와 instruction data는 여러 media occurrence, conversation role과 response-only loss mask를 추가한다. 데이터 단계가 바뀔 때 decoder target mass와 vision gradient coverage도 같이 바뀐다.

alignment projector만 먼저 학습하는 단계는 이미 학습된 두 표현 공간 사이에 작은 bridge를 만든다. 그다음 instruction tuning에서 language layers나 tower 일부를 열면 bridge가 고정 좌표계를 연결하는 문제가 아니라 세 좌표계가 함께 움직이는 문제가 된다. learning rate와 freeze schedule을 분리하는 이유다. 하지만 무조건 projector warm-up이 필요한 것은 아니며 모델·초기화·데이터의 실제 ablation으로 판단한다.

멀티모달 평가를 정답률 하나에서 해방한다.

VQA 정답률이 높아도 모델이 이미지만 보고 답했는지, 질문의 언어 shortcut을 썼는지 알 수 없다. 원 이미지, 의미 보존 변형, 관련 영역 삭제, 무관 이미지, text-only의 paired set을 만든다. 정답 변화가 기대되는 쌍과 불변이어야 하는 쌍을 사전에 정의한다. 이를 grounding sensitivity와 invariance로 분리한다.

OCR·문서 평가는 processor resolution과 crop이 점수의 큰 부분을 결정할 수 있다. benchmark가 제공한 이미지의 resize·page split, answer normalization과 exact-match 규칙을 보존한다. 학습과 평가 processor가 다르면 모델 비교가 아니라 입력 정책 비교가 섞인다. 작은 글자 size bucket과 page aspect별 점수를 함께 본다.

영상 평가는 시간 순서를 요구하는 질문과 단일 frame으로 풀리는 질문을 분리한다. frame count를 늘린 결과가 좋아져도 사건이 포함될 확률이 오른 것인지 temporal reasoning이 좋아진 것인지 구분한다. oracle event frames와 uniform samples를 비교하면 sampling ceiling을 볼 수 있다.

음성 평가는 WER만으로 충분하지 않다. punctuation·normalization 규칙을 고정하고, 억양·잡음·화자 overlap·긴 문맥과 timestamp localization을 나눈다. speech generation이면 intelligibility, speaker similarity, prosody와 안전성을 별도 측정한다. codec reconstruction metric과 최종 언어 의미도 구분한다.

마지막 보고서는 model score 옆에 processor bundle, sampling, prompt template, decode parameters, evaluation code revision과 exclusion을 둔다. 지원하지 않는 modality나 실패한 decode를 0점으로 넣었는지 제외했는지 명시한다. 그래야 점수가 모델의 어느 능력과 시스템의 어느 정책을 함께 반영하는지 독자가 판단할 수 있다.

## 21.15 checkpoint·evaluation·failure evidence로 release를 닫는다

release는 checkpoint 파일이 열리는 순간이 아니라, 저장된 processor와 modality 상태로 같은 좌표·shape·loss 판정을 재생할 수 있을 때 닫힌다. 따라서 여기서는 평균 점수보다 먼저 재현 가능한 failure evidence를 요구한다. 원시 매체에서 평가 결과까지 최초로 달라진 경계를 찾고, 그 경계의 owner와 rollback 단위를 함께 기록한다.

모델 weight만 저장해서는 같은 멀티모달 학습을 재개할 수 없다. processor·sampling·feature-cache identity, trainable ownership, optimizer state와 modality별 loss clock을 같은 generation에 묶고, perception·grounding·reasoning 평가와 failure injection을 그 subject에 연결한다. release는 text-only smoke test가 아니라 media-dependent negative control까지 통과해야 한다.

데이터 설정에 `image_caption: 0.4`, `video_qa: 0.1`, `text: 0.5`라고 썼다고 해서 학습의 절반이 text라는 뜻은 아니다. sampler weight는 표본 선택 확률이고, 한 표본이 만드는 visual features, language targets와 GPU 시간이 서로 다르다. 선택된 표본 수, raw media seconds·pixels, encoder features, decoder input tokens, loss-bearing targets와 accelerator seconds를 모두 누적한다.

mixture의 실현값은 worker failure와 filter에도 영향을 받는다. 영상 decode 실패가 많아 재시도 뒤 text fallback을 넣으면 configured video 비율은 유지된 것처럼 보이면서 실제 gradient는 text로 기운다. admission, sampled, decoded, collated, forward-valid와 update-contributed의 funnel을 modality별로 기록한다. fallback은 원 modality 성공으로 세지 않는다.

epoch 개념도 조심한다. 거대한 image-text corpus와 작은 고품질 instruction set을 섞으면 작은 set이 여러 번 반복된다. global sample counter만으로는 각 source의 exposure를 알 수 없다. source generation별 unique count, repeats 분포, 마지막 노출 step과 유효 target 누적량을 보존한다. caption 하나가 여러 crop으로 증강돼도 원 asset family의 반복으로 연결한다.

curriculum은 시간에 따라 mixture와 해상도, duration, task weight를 바꾼다. schedule 함수의 입력이 optimizer step인지 consumed tokens인지 wall-clock인지 확인한다. skipped update, gradient accumulation과 resume가 있어도 같은 지점에서 같은 policy가 선택되어야 한다. curriculum state를 config만으로 재계산할 수 없다면 checkpoint에 명시적으로 넣는다.

회계 fixture. image 두 표본은 각각 features 256, language targets 20이고 text 두 표본은 targets 200이라고 하자. 표본 비율은 50:50이지만 language loss mass는 40:400이다. sample mean, token mean과 modality-balanced mean은 서로 다른 목적함수다. 원하는 식을 먼저 쓰고 collator·loss가 그 식을 구현하는지 손으로 계산한다.

가변 형상 batch sampler는 자원 스케줄러다.

텍스트 길이만으로 batch를 만들면 visual feature가 많은 표본이 한 microbatch에 몰려 OOM이 난다. 비용 추정량은 모델에 따라 `aL_text + bL_media + cL_text² + dL_textL_media` 같은 형태가 될 수 있다. 정확한 FLOP 식이 아니어도 실제 peak memory·latency를 설명하는 monotonic proxy를 calibration하면 token-count보다 안전한 budget sampler가 된다.

bucket 경계가 너무 촘촘하면 padding은 줄지만 shuffle entropy가 떨어지고 같은 해상도·modality가 연속된다. 너무 거칠면 tail 표본 하나 때문에 전체 batch가 커진다. bucket별 표본 수, padding ratio, queue age와 source diversity를 함께 본다. curriculum이 resolution을 올리면 기존 cost calibration을 다시 맞춘다.

분산에서는 rank마다 local cost가 비슷해야 step straggler가 줄어든다. 단순 round-robin은 긴 영상이 한 rank에 몰릴 수 있다. global batch planner가 estimated costs를 균형 배치하거나, rank별 batch가 달라도 collective 순서를 맞추는 ragged protocol이 필요하다. data loader가 local OOM 뒤 batch를 조용히 축소하면 global denominator와 optimizer 의미가 바뀐다.

dynamic batching은 재현성에도 영향을 준다. worker completion 순서대로 표본을 묶으면 decode 속도 차이가 batch composition을 바꾼다. sample order와 batch plan을 deterministic artifact로 저장하거나, 비결정성을 허용한다면 seed만으로 exact replay가 불가능함을 명시한다. 실패 재현에는 당시 batch member와 processed shape가 필요하다.

CUDA 관점에서 processor와 model 사이의 숨은 비용을 찾는다.

이미지·음성 decode와 resize가 CPU에서 이루어지면 pinned host buffer와 asynchronous H2D가 GPU compute와 겹칠 수 있다. 그러나 worker가 pageable memory를 반환하거나 main thread가 `.to(device)` 직후 동기화를 일으키면 overlap이 사라진다. profiler에서 decode, collate, pin, H2D와 첫 vision kernel의 stream dependency를 본다.

가변 형상을 매 step 새로 만나면 kernel compilation, allocator fragmentation과 CUDA graph miss가 늘 수 있다. shape bucket은 padding 절감뿐 아니라 compile·graph 재사용의 단위다. processor가 반환하는 grid class별 compile count, graph capture status와 allocator reserved/active bytes를 관측한다. warm benchmark 하나만으로 steady state를 대표하지 않는다.

vision attention이 FlashAttention 또는 fused kernel로 내려갈 때 지원되는 dtype, head dimension, mask와 variable-length metadata를 확인한다. pixel mask에서 만든 arbitrary 2D validity가 kernel의 단순 padding mask로 정확히 표현되는지 검산한다. 지원하지 않는 mask가 dense fallback으로 바뀌면 수치 의미는 같아도 memory와 latency가 급증할 수 있다.

mixed precision에서 vision tower, projector와 language model이 서로 다른 dtype을 쓸 수 있다. projector 앞의 implicit cast, normalization accumulator와 loss scaling을 추적한다. FP8 vision linear를 쓰더라도 softmax, norm과 selected reduction이 더 높은 precision일 수 있다. “FP8 모델”이라는 이름 대신 operator별 input·weight·accumulator·output dtype을 표로 만든다.

audio와 video는 host decode가 병목이 되기 쉬우므로 GPU utilization만 보고 batch를 키우지 않는다. decode queue가 비어 있는지, H2D가 compute stream을 막는지, vision encoder가 실제 병목인지 구분한다. cached features로 모델 최대 throughput을 재고 raw-media end-to-end throughput과 비교하면 전처리 ceiling을 분리할 수 있다.

tensor parallel과 pipeline parallel에서 media feature는 누구의 것인가.

vision tower가 한 rank에만 있고 language model은 TP로 나뉘어 있다면 projector 출력 hidden을 shard하거나 replicate해야 한다. global `[N_media,D]` feature를 각 TP rank가 `[N_media,D/tp]`로 갖는지, full `D`를 복제하는지는 다음 linear와 collective 구현에 달려 있다. source에서 실제 layout을 확인하고 H2D·interconnect bytes를 계산한다.

pipeline parallel에서는 vision stage와 language stage 사이에 feature뿐 아니라 mask, occurrence offsets, grid·position metadata가 이동한다. payload tensor만 보내고 ordinal을 local batch에서 재생성하면 variable media count에서 어긋날 수 있다. microbatch ID와 SampleID, media occurrence, payload shape를 하나의 transfer envelope로 묶는다.

context parallel이 joint sequence를 나누면 media span이 CP boundary를 가를 수 있다. position과 causal/cross-document mask가 global coordinate를 유지해야 한다. media를 한 CP rank에 몰아두는 최적화는 다른 rank가 필요한 KV를 어떻게 얻는지와 backward adjoint를 명시한다. 통신을 줄였다는 설명만으로 correctness를 증명하지 않는다.

encoder를 별도 GPU pool에 두는 비동기 구조에서는 feature가 어느 model generation에서 만들어졌는지가 중요하다. tower가 업데이트되는데 오래된 feature cache를 소비하면 한 optimizer batch 안에 서로 다른 encoder version이 섞인다. publication fence를 두거나 encoder freeze 구간에서만 cache를 허용한다. feature message에는 encoder·processor bundle과 raw asset checksum을 넣는다.

### 21.15.1 checkpoint에 processor·feature identity를 포함한다

멀티모달 checkpoint에는 vision/audio/video tower, projector·resampler, language model, adapters와 optimizer state가 들어간다. 여기에 processor config, tokenizer/chat template, special media IDs, feature selection layer, merge factor와 curriculum state가 함께 고정돼야 한다. weight만 복구하고 processor를 최신 default로 읽으면 첫 forward가 성공해도 다른 입력 공간에서 학습을 이어간다.

tied parameters와 frozen modules는 저장 정책을 복잡하게 한다. frozen tower를 parent checkpoint에서 참조하고 child에는 adapter만 둘 수 있지만 parent가 삭제되면 복구할 수 없다. manifest가 content-addressed dependency를 명시하고 garbage collector가 reachability를 계산해야 한다. 외부 Hub revision을 mutable branch 이름으로만 참조하지 않는다.

optimizer group은 단계별 freeze policy와 맞아야 한다. projector만 학습하던 checkpoint에서 tower 일부를 여는 단계로 resume할 때 새 parameter의 moment를 0으로 시작할지 별도 warm-start할지 정한다. scheduler는 global step을 이어가도 새 group의 effective age가 다를 수 있다. group별 creation step, LR multiplier와 state initialization을 기록한다.

sharded checkpoint에서 modality tower가 특정 ranks에만 존재하면 save 참여와 expected keys가 비대칭이다. rank-local 파일 수가 모두 같아야 한다는 가정을 버리고 logical parameter inventory를 global manifest와 비교한다. incomplete save가 일부 modality keys만 빠뜨렸는데 language-only smoke test가 통과하는 상황을 negative fixture로 둔다.

resume 첫 단계에서는 update하지 않고 processor부터 loss까지 dry-run한다. parent run의 golden media가 같은 processed shapes, logits·loss 허용오차와 gradient ownership을 재현하는지 본다. 이후 실제 한 update를 수행해 optimizer group별 delta와 global denominator를 비교한다. 이 두 단계가 통과하기 전 대규모 data stream을 열지 않는다.

feature cache가 유효한 정확한 조건.

frozen media encoder의 output을 cache하면 비싼 decode·encoder forward를 줄일 수 있다. 하지만 cache key가 raw path뿐이면 파일 교체, processor 변경, crop randomness, encoder revision과 feature selection을 구분하지 못한다. key에는 raw content hash, decode/processor bundle, deterministic transform parameters, encoder weights, selected layer·strategy, output dtype와 schema가 필요하다.

random crop·color augmentation 뒤 feature를 cache하면 첫 augmentation이 영구 고정된다. 다양성을 원한다면 augmentation seed까지 key에 넣어 여러 variant를 저장하거나, deterministic resize까지만 cache하고 stochastic transform 이후는 매번 계산한다. 어느 지점을 경계로 선택했는지 curriculum의 의도와 비용을 함께 설명한다.

encoder를 unfreeze하는 순간 기존 feature cache는 학습 graph와 의미가 모두 틀린다. cached tensor에는 encoder backward 경로가 없고, weight update 뒤 stale representation이다. projector-only stage에서는 유효하던 cache를 joint tuning stage에서 자동으로 차단해야 한다. 경고 로그에 그치지 않고 generation mismatch를 hard error로 만든다.

cache corruption은 value checksum과 shape만으로 충분하지 않다. sample A의 정상 feature가 sample B key 아래 저장되면 둘 다 유효한 tensor다. raw identity와 occurrence ordinal을 value metadata에도 넣고 읽을 때 key와 대조한다. 작은 sentinel media suite를 주기적으로 재계산해 cache hit와 fresh path의 feature·downstream loss를 비교한다.

### 21.15.2 최초 잘못된 경계에서 failure를 진단한다

증상 1은 language loss는 정상인데 image benchmark만 급락하는 경우다. 먼저 evaluation processor와 prompt template drift를 확인한다. 다음으로 admission·decode와 feature cardinality, projector gradient, cross-attention gate를 본다. optimizer 전체를 되돌리기 전에 image path의 최초 divergence를 paired golden sample에서 찾는다.

증상 2는 특정 긴 영상에서만 OOM이 나는 경우다. raw duration이 아니라 sampled frames, grid, visual features와 merged sequence를 확인한다. sampler cap이 적용되지 않았는지, variable FPS가 예상보다 많은 frames를 만들었는지, dense mask fallback이나 compile class가 달라졌는지 본다. batch size를 전역으로 낮추는 것은 원인 규명 뒤의 임시 완화책이다.

증상 3은 resume 뒤 첫 수백 step에서 loss가 튀는 경우다. processor bundle, curriculum phase, optimizer groups와 feature cache generation을 parent와 비교한다. 새로 열린 tower parameter가 full LR로 시작하거나 moment가 누락됐는지 본다. 동일 golden batch의 update delta가 parent preemption 직전과 어디서 처음 달라지는지 추적한다.

증상 4는 분산 규모를 늘렸을 때 contrastive loss만 달라지는 경우다. global negative pool과 temperature, duplicate mask, all-gather gradient와 reduction denominator를 확인한다. global batch가 커졌다면 목적함수 자체가 바뀔 수 있으므로 동일한 결과를 기대하기 전에 수식을 맞춘다. parity test는 같은 effective negative set을 구성해야 한다.

증상 5는 모든 score가 좋아졌지만 text-only 능력이 떨어지는 경우다. mixture의 realized language targets, language parameter gradient와 replay 비율을 본다. visual sequence가 길어져 같은 sample 비율에서 text target mass가 줄었을 수 있다. multimodal 평균만 보지 말고 parent text suite와 forgetting slice를 출시 관문에 둔다.

각 incident 기록에는 symptom, 최초 divergent artifact, 배제한 가설, root cause, 임시 완화, 영구 수정과 regression fixture를 담는다. “데이터 문제”나 “CUDA 불안정”처럼 층을 건너뛴 결론은 허용하지 않는다. 동일 결함이 다음 release에서 어느 assertion과 dashboard로 더 일찍 잡히는지가 종료 조건이다.

한 표본으로 끝까지 계산하는 종합 연습.

비정사각 문서 이미지 한 장과 질문·답변 대화를 고른다. 원본 checksum, width·height, EXIF와 OCR boxes를 기록한다. processor의 resize/crop 뒤 크기와 patch grid를 손으로 계산하고, merge 또는 resampler 뒤 feature count를 구한다. chat template가 만든 token IDs에서 media placeholder와 assistant target byte span을 표시한다.

model 구조가 early fusion이면 placeholder 확장 뒤 최종 text position과 causal/document mask를 그린다. cross-attention이면 text query별 media visibility와 valid media key를 그린다. 2D/3D position이 있으면 각 feature row의 좌표를 작은 표로 만든다. projector input·output과 첫 language residual의 shape와 norm을 예상한다.

정답 token 세 개만 골라 shifted logits와 labels, ignore mask를 확인한다. cross-entropy numerator를 직접 계산하고 전체 valid target count로 나눈다. auxiliary contrastive나 grounding loss가 있으면 각각 sum/count와 weight를 따로 계산한 뒤 total을 합친다. framework 반환 loss와 허용오차 안에서 맞아야 한다.

backward에서는 tower, projector, fusion과 language selected parameter의 gradient 유무와 shape를 예상한다. freeze policy와 다른 gradient가 나타나면 즉시 중단한다. DP·TP가 있다면 local tensor와 global logical tensor를 재구성하고 collective 뒤 denominator와 gradient가 single-device oracle와 맞는지 본다.

마지막으로 checkpoint를 저장한 뒤 새 process에서 같은 processor bundle로 불러온다. feature cache를 일부러 이전 encoder generation으로 바꿔 loader가 거절하는지 확인한다. 정상 경로의 processed tensor, loss와 selected delta가 재현되면 이 표본은 이후 코드·데이터·커널 변경을 감시하는 영구 회귀 fixture가 된다.

visual tokenizer는 압축률과 생성 어휘를 함께 설계한다.

VQ 계열 visual tokenizer는 encoder latent `z_e`를 가장 가까운 codebook vector `e_k`로 치환한다. 이미지 한 장은 연속 pixel 배열에서 이산 index grid로 바뀌고, 생성 모델은 그 index를 vocabulary처럼 예측할 수 있다. 하지만 text tokenizer와 달리 code의 의미는 학습과 함께 움직이며, 인접 index가 비슷한 시각 개념이라는 보장도 없다.

reconstruction loss만 줄이면 perceptual quality나 downstream semantics가 충분하다고 할 수 없다. pixel loss는 작은 위치 차이에 민감하고, perceptual/adversarial loss는 세부 색이나 글자를 희생할 수 있다. OCR·얼굴·도표 같은 목적 slice에서 reconstruction, perceptual metric과 downstream task를 함께 본다. tokenizer의 손실과 최종 생성 모델의 손실은 같은 목표가 아니다.

codebook collapse는 평균 reconstruction loss에 숨는다. entry usage count, entropy, dead-code age와 assignment distance를 추적한다. EMA codebook이면 count/sum accumulator와 decay가 학습 state이고, gradient 방식이면 embedding optimizer state가 필요하다. distributed training에서 rank-local EMA를 독립 적용하면 codebook이 rank마다 갈라지므로 reduce 순서와 분모를 확인한다.

Residual VQ는 여러 codebook이 앞 단계의 잔차를 순차적으로 근사한다. `[B,Q,T]` index에서 `Q`가 늘면 bitrate와 fidelity가 높아지지만 생성 sequence·vocabulary routing이 복잡해진다. coarse code만 먼저 예측하고 fine code를 조건부 생성하는지, time-major로 interleave하는지에 따라 causal mask와 loss shift가 다르다.

tokenizer를 freeze한 뒤 생성 모델을 학습하면 code semantics는 안정적이지만 tokenizer의 결함도 고정된다. 공동 학습하면 생성 objective가 codebook을 움직여 이미 저장한 token dataset을 stale하게 만들 수 있다. offline token shard는 tokenizer generation을 key로 갖고, tokenizer update가 시작되면 자동 무효화해야 한다.

음성 codec token은 시간과 계층, 두 축에 놓인다.

neural audio codec은 waveform을 낮은 frame rate의 latent로 만들고 RVQ codebooks로 양자화한다. 하나의 시간 frame에 여러 code가 있으므로 language token처럼 1차원으로 펼칠 규칙이 필요하다. time-major는 같은 시점의 codebooks를 연속 배치하고, codebook-major는 한 계층의 전 시간을 먼저 배치할 수 있다. attention locality와 streaming latency가 달라진다.

codebook 0이 대략적인 음향을, 뒤 codebook이 잔차 세부를 담는 구조라면 계층별 loss weight와 dropout을 사용할 수 있다. 일부 fine codebooks를 무작위로 생략하면 variable bitrate를 학습할 수 있지만 decoder가 어떤 codebooks가 존재하는지 mask로 알아야 한다. 빠진 code를 0 index로 위장하면 실제 code 0과 충돌한다.

speech-language joint vocabulary에서는 text IDs와 codec IDs의 namespace를 분리하거나 offset으로 합친다. special token, speaker token과 codec range가 겹치지 않는지 config와 embedding table 크기를 확인한다. checkpoint conversion에서 vocabulary resize를 놓치면 일부 rows가 임의 초기화되거나 tied output head와 어긋난다.

streaming 생성은 frame을 다 만든 뒤 vocoder를 돌리는 offline 경로와 다르다. lookahead, chunk overlap과 codec receptive field가 첫 소리 지연과 경계 artifact를 정한다. 학습 chunking이 추론 chunking과 다르면 boundary distribution이 바뀐다. training fixture에 chunk 시작·끝과 cache reset을 넣는다.

멀티모달 LoRA는 어느 선형층에 붙였는지가 전부다.

language attention의 `q_proj`에 LoRA를 붙이는 것과 vision tower attention, projector, cross-attention에 붙이는 것은 학습 가능한 함수가 다르다. target module 문자열에 `q_proj`라고만 쓰면 vision과 language 양쪽의 동명 module이 선택될 수도 있다. 실제 matched parameter의 qualified name, shape와 총 element를 저장한다.

projector-only LoRA는 작은 비용으로 modality 좌표 bridge를 바꾸지만 tower 표현과 language reasoning 자체는 고정된다. cross-attention LoRA는 media를 읽는 방식에 직접 개입한다. language MLP LoRA는 일반 instruction following을 바꾸지만 grounding 병목을 못 풀 수 있다. adapter 위치는 benchmark 이름이 아니라 실패 가설에서 선택한다.

vision tower의 convolution·patch embedding이나 3D tubelet projection은 일반 linear-target 규칙에서 빠질 수 있다. PEFT library가 지원하는 module type과 weight layout을 확인한다. adapter가 생성됐다는 로그보다 selected parameter에 실제 nonzero gradient와 update가 있는지 본다.

여러 modality adapter를 분리하면 image·audio·video별 교체가 쉽지만 shared language layers에서 상호작용을 놓칠 수 있다. 하나로 합치면 transfer가 가능하지만 한 modality의 gradient가 다른 modality를 방해할 수 있다. adapter별 gradient cosine, paired evaluation과 routing state를 관측한다. 단순 parameter 수 비교로 결정하지 않는다.

merge 뒤에는 base plus low-rank delta의 dtype과 scale을 확인한다. quantized base에 adapter를 적용했다가 FP16으로 merge하고 다시 양자화하면 원본 QLoRA runtime과 다른 함수가 된다. processor·adapter·base·merge·quantization generation을 하나의 배포 manifest로 묶고 golden media logits를 비교한다.

선택적 unfreeze는 최적화 문제를 갑자기 바꾼다.

vision tower 마지막 `k` blocks만 여는 전략은 저수준 feature를 보존하면서 고수준 정렬을 바꾸려는 선택이다. 그러나 block 번호만으로 역할을 단정하지 않는다. normalization, positional embedding과 patch projection을 함께 열지 여부가 중요하다. parameter group inventory가 정책을 정확히 반영해야 한다.

오랫동안 frozen이던 layer를 열면 그 parameter에는 optimizer moment가 없고 gradient scale도 projector와 다를 수 있다. 같은 global LR을 적용하면 첫 update가 과도할 수 있다. 별도 LR, warmup 또는 gradient clipping을 고려하되 작은 paired experiment로 결정한다. group별 update/weight norm을 기록한다.

BatchNorm running state가 있는 encoder라면 `requires_grad=False`와 eval mode는 다르다. parameter gradient를 막아도 train mode의 running statistics는 변할 수 있다. ViT의 LayerNorm 중심 구조에서도 dropout과 stochastic depth가 train/eval에 따라 달라진다. frozen tower가 deterministic이어야 feature cache가 유효하다는 전제와 연결한다.

단계 전환 checkpoint는 old optimizer를 그대로 load한 뒤 새 group을 추가하는 코드 경로를 테스트해야 한다. scheduler가 group 수를 가정하거나 state dict ordering에 의존하면 잘못된 LR이 매핑될 수 있다. qualified parameter IDs로 group을 검증하고 첫 update 전에 expected LR·moment presence를 출력한다.

합성 멀티모달 데이터의 교사 오류를 추적한다.

caption이나 visual question을 강한 모델로 합성하면 규모를 빠르게 늘릴 수 있지만 teacher의 hallucination과 style이 그대로 들어온다. raw media에서 검증 가능한 객체·문자·시간 사건과 teacher 문장을 분리한다. OCR·detector·사람의 독립 신호가 모두 정답은 아니므로 agreement와 disagreement를 보존한다.

합성 파이프라인은 teacher model, prompt, sampling, image preprocessing과 refusal policy를 고정한다. 같은 teacher라도 crop이나 prompt가 바뀌면 다른 dataset generation이다. 실패·거절 표본을 조용히 버리면 쉬운 이미지로 분포가 기운다. source·language·complexity별 acceptance funnel을 기록한다.

질문 생성 뒤 같은 teacher로 답을 만들고 다시 같은 계열 judge로 검증하면 오류가 상관될 수 있다. 독립 rule, 다른 model family와 사람 spot audit를 섞는다. judge score threshold 주변 표본과 높은 confidence 오답을 우선 조사한다. 자동 합의는 사실성 증명이 아니다.

합성 데이터를 curriculum 후반에 많이 넣으면 모델이 teacher 표현을 과도하게 모방할 수 있다. human·organic caption과 synthetic의 realized target mass, style classifier와 evaluation diversity를 본다. synthetic flag를 provenance에 남겨 삭제·재가중·ablation이 가능하게 한다.

modality dropout은 견고성을 만들 수도, shortcut을 가르칠 수도 있다.

학습 중 image를 일정 확률로 제거하면 text-only에서도 동작하도록 만들 수 있다. 하지만 media placeholder만 남기고 zero feature를 넣는지, placeholder와 관련 문장을 함께 제거하는지에 따라 다른 objective다. 정답이 질문 text만으로 추론 가능하면 모델은 media를 무시하는 shortcut을 배울 수 있다.

dropout 정책은 표본마다 어떤 modality가 원래 있었고 무엇이 제거됐는지 기록한다. loss를 동일하게 유지할지, missing-modality special response를 가르칠지 정한다. 무작위 제거와 실제 decode failure fallback을 같은 사건으로 세지 않는다. 전자는 설계된 augmentation이고 후자는 인프라 결함이다.

paired batch에서 원본, image dropout, text hint dropout과 both-present를 비교한다. tower·projector gradient, 답변 정확도와 calibration 변화로 모델이 어느 신호에 의존하는지 본다. 단일 attention visualization보다 이 반사실이 강한 근거다.

classifier-free guidance를 위한 condition dropout은 생성 diffusion의 unconditioned branch를 학습하는 목적이다. VLM의 robustness dropout과 이름이 비슷해도 수식과 사용처가 다르다. 22장에서 noise state와 condition dropout이 어떻게 guidance 식으로 연결되는지 다시 구분한다.

production admission으로 넘어갈 조건.

멀티모달 장을 읽었다는 것은 모델 이름을 구별하는 데서 끝나지 않는다. 독자는 raw media 하나가 어떤 processor state로 tensor가 되고, 어느 feature rows와 positions로 변환되며, 어떤 mask·fusion을 거쳐 어느 target loss와 parameter update에 기여했는지 재구성할 수 있어야 한다.

출구 시험은 네 부분이다. 첫째 image·audio·video·interleaved 표본의 cardinality와 timestamp·position을 손으로 계산한다. 둘째 early fusion, cross-attention, resampler와 deepstack의 gradient 경로를 그린다. 셋째 single device와 distributed loss sum/count, checkpoint resume를 맞춘다. 넷째 media swap·mask one-off·stale cache·processor drift를 최초 경계에서 차단한다.

source 근거는 고정 revision의 실제 processor, model, loss와 tests를 가리켜야 한다. 모델 카드가 전체 training recipe를 공개하지 않았다면 알려진 구현 경계까지만 말한다. inference-only sampler를 training loss의 존재로 확대하지 않는다. gated 또는 접근하지 못한 자료는 추정으로 채우지 않는다.

이 조건을 통과하면 21장은 22장의 diffusion·flow 생성, 24장의 평가, 25장의 red team, 29장의 분산 장애와 30장의 end-to-end recipe에 같은 SampleID·BundleID·UpdateID를 넘긴다. downstream에서 media 정책이 바뀌면 영향받은 변환·mask·loss·cache·평가를 다시 실행한다.

image-text interleave에서 인과 경계를 그린다.

한 문서에 이미지와 문단이 번갈아 나오는 표본은 단일 image-caption pair보다 훨씬 많은 결정을 요구한다. 각 문장이 앞의 이미지, 뒤의 이미지 또는 문서 전체를 설명하는지 관계가 필요하다. 단순히 파일 순서대로 placeholder를 끼우면 페이지 layout과 DOM reading order가 어긋나는 자료에서 잘못된 정렬을 학습한다.

relation은 `(text span, media occurrence, relation type, confidence, source)`로 저장한다. caption, alt text, 근처 문단, OCR 영역과 질문의 관계를 구분한다. 한 text span이 여러 media를 참조하거나 한 image가 여러 문단에 걸쳐 설명될 수 있으므로 일대일 외래키로 축소하지 않는다. 낮은 confidence 관계를 버린 비율과 source별 편향도 기록한다.

packing할 때 두 문서가 같은 sequence에 들어가면 document boundary가 media attention에도 적용돼야 한다. text causal mask만 끊고 cross-attention media pool을 공유하면 둘째 문서가 첫째 이미지에 접근한다. 작은 dense predicate에서 `(query document == media document)` 조건을 확인하고 fused varlen metadata가 이를 보존하는지 비교한다.

long-context curriculum에서는 media span이 context window 밖으로 밀릴 수 있다. 질문은 남았는데 관련 image가 잘렸다면 답변 target을 그대로 학습시키지 않는다. relation-aware truncation으로 질문과 필요한 media를 함께 보존하거나 표본을 제외하고, lost target mass를 기록한다. 단순 left truncation은 multimodal 정렬을 조용히 깨뜨린다.

grounding supervision은 좌표계 변환의 연쇄다.

bounding box는 원본 pixel, resize된 image, crop, normalized `[0,1]`, patch grid와 model token 좌표 중 하나에 존재한다. 어느 좌표인지 생략하면 숫자 네 개가 같은 범위여도 의미가 없다. 원본에서 model 입력까지 모든 affine·crop 변환과 clipping을 합성하고 역변환 가능성을 검사한다.

box regression을 discrete location tokens로 바꾸면 tokenizer bin 수와 rounding policy가 새로운 양자화 오차를 만든다. `x/W`를 `floor`, `round` 중 무엇으로 binning하는지, maximum coordinate가 마지막 bin을 넘지 않는지 확인한다. decode 뒤 원 pixel로 복원해 IoU를 계산한다. token exact match만 보면 서로 가까운 box도 완전 오답이 된다.

segmentation mask는 resize interpolation을 잘못 선택하면 class label이 섞인다. image에는 bilinear를 쓰더라도 categorical mask에는 nearest-neighbor가 필요할 수 있다. ignore region과 padding mask를 구분하고, augmentation 뒤 area가 0이 된 객체의 target을 제거한다. empty target의 loss denominator를 별도로 처리한다.

point·box·text grounding을 함께 학습하면 task token과 output grammar가 충돌할 수 있다. constrained decoding을 평가에만 쓰는지 학습 target에도 canonical serialization을 쓰는지 명시한다. malformed coordinate sequence의 처리, special token collision과 tokenizer round trip을 fixture로 둔다.

text 능력 보존은 별도 데이터와 평가가 필요하다.

멀티모달 instruction tuning은 language-only 표본을 섞더라도 text 능력을 자동 보존하지 않는다. visual tokens가 긴 표본은 같은 sequence budget에서 language targets를 적게 제공하고, optimizer update는 projector·vision branch의 큰 gradient에 영향을 받는다. realized language target mass와 language parameter update를 parent run과 비교한다.

text replay의 비율은 sample 비율이 아니라 유효 target과 gradient 관점에서 정한다. 짧은 text 대화가 긴 multimodal batch에 묻히면 configured 비율이 충분해 보여도 기여가 작다. task별 loss sum/count와 selected language layer gradient norm·cosine을 기록한다. conflict가 크면 batch alternation, loss weight 또는 adapter 분리를 시험한다.

parent text suite는 일반 지식, reasoning, code, multilingual, safety와 chat-template compliance를 포함한다. multimodal 평가가 좋아지는 동안 어느 slice가 떨어졌는지 본다. text-only 입력에서 media branch가 완전히 비어 있을 때 mask, collective와 gate가 정상인지도 테스트한다. dummy image를 강제로 넣는 runtime은 배포 계약과 다를 수 있다.

forgetting이 보이면 곧바로 더 많은 text를 넣기 전에 원인을 분리한다. tokenizer/chat template 변화, optimizer LR, language parameter unfreeze, loss denominator와 data mixture를 차례로 본다. processor가 text-only prompt에도 media special token을 넣는 결함은 데이터 비율로 해결되지 않는다.

멀티모달 preference와 RL은 media identity부터 고정한다.

chosen과 rejected 답변을 비교할 때 두 응답이 반드시 같은 raw media와 같은 processor transform을 조건으로 해야 답변 선호를 학습한다. 한쪽이 다른 crop이나 frame sample을 보면 preference label은 응답과 관측 차이를 섞는다. pair row는 하나의 MediaBundleID를 공유하고 두 branch의 processed checksum을 대조한다.

평가자는 이미지의 사실을 보지 못하고 text 답변만 판단할 수 있다. annotation UI가 실제 media, 확대·재생·음량과 timestamp를 어떻게 제공했는지 기록한다. 접근성 실패나 decode 오류가 있는 pair는 tie·invalid로 분리한다. labeler가 media를 확인했다는 사실을 inference로 채우지 않는다.

멀티모달 reward model은 media encoder를 공유하거나 별도로 둘 수 있다. policy와 reward의 processor가 다르면 동일 raw asset도 다른 관측을 갖는다. reward score를 해석할 때 reward processor generation을 trajectory에 넣는다. image swap, 관련 영역 삭제와 text-only 반사실에서 score가 사실성에 반응하는지 검증한다.

online rollout에서 video sampling이나 random crop이 비결정적이면 trajectory replay가 불가능해진다. 선택 PTS, crop seed와 processed artifact를 보존한다. policy update 뒤 같은 prompt를 다시 생성하는 것은 원 trajectory log-prob 검산을 대신하지 않는다. 19·20장의 policy/reference/reward identity에 MediaBundleID를 추가한다.

멀티모달 보안 경계는 decoder 이전에 시작된다.

외부 URL에서 media를 가져오는 loader는 네트워크, 파일 parser와 model을 잇는 공격면이다. 허용 scheme·host, redirect, 크기·duration, decompression ratio와 decode timeout을 admission에서 제한한다. MIME header만 믿지 않고 parser가 확인한 format을 사용한다. 실패 파일은 worker 전체를 죽이지 않되 정상 sample로 위장하지 않는다.

image metadata와 container에는 예상 밖의 큰 profile, nested chunk나 malformed timestamp가 있을 수 있다. decode library revision과 sandbox 경계를 고정하고, parser crash·OOM을 fixture로 둔다. 학습 corpus는 한 번 정제됐다는 이유로 안전하다고 가정하지 않는다. 재처리 pipeline과 새 decoder가 다른 취약점을 열 수 있다.

prompt injection은 OCR된 문자열뿐 아니라 작은 글자, subtitle와 음성으로 들어온다. system instruction과 media content를 같은 trust level로 합치지 않는다. 학습에서는 media 속 명령을 무조건 따르는 target을 제거하거나 context로 인용하도록 정책을 설계한다. tool-use 평가에서는 media가 요구한 외부 action을 실행하기 전에 user intent와 authorization을 확인한다.

feature cache와 checkpoint도 공급망 경계다. 공격자가 정상 raw key에 조작 feature를 넣으면 decode·검열을 우회할 수 있다. content signature, producer identity와 bundle generation을 검증한다. adapter나 projector만 배포해도 어느 base·processor와 결합해야 하는지 서명된 manifest를 요구한다.

### 21.15.3 evaluation·rollback·release 판정표를 만든다

운영 판정표의 행은 image, document, audio, video, interleaved, text-only와 각 failure class다. 열은 admission, processor, feature cardinality, position·mask, fusion, loss denominator, gradient owner, distributed placement, checkpoint, evaluation, safety와 rollback이다. 각 셀에는 PASS·FAIL·NOT_RUN과 재생 가능한 근거를 기록한다.

PASS는 forward가 한 번 끝났다는 뜻이 아니다. 정상 fixture와 하나 이상의 의미 있는 negative fixture가 기대 경계에서 판별되고, update 전 차단 또는 올바른 loss contribution이 확인돼야 한다. NOT_RUN을 0이나 빈칸으로 두지 않는다. 지원하지 않는 shape·backend·topology는 명시적으로 제한한다.

release reviewer는 임의 행 하나를 골라 raw bytes부터 processor tensor, encoder feature, fusion mask, selected logits·loss, gradient와 checkpoint delta까지 추적한다. 이어 processor option 하나를 바꾸어 어떤 cache와 결과가 무효화되는지 예측한다. manifest와 실제 loader가 같은 답을 내야 한다.

마지막으로 parent text suite와 modality별 counterfactual, resume와 rollback을 실행한다. 성능 개선이 있어도 media identity, privacy deletion, distributed denominator나 parent 복구가 끊기면 승인하지 않는다. 멀티모달 학습의 완성도는 화려한 demo가 아니라 입력의 의미가 update와 배포 뒤에도 보존되는 정도로 판단한다.

한 장짜리 수치 검산표.

검산표 첫 칸에는 raw media의 크기·시간과 checksum을 적는다. 둘째 칸에는 resize, crop, frame PTS, STFT·codec 설정과 processor 출력 shape를 적는다. 셋째 칸에는 patch·tubelet·audio frame, merge·resampler 뒤 feature 수와 placeholder occurrence를 적는다. 숫자는 설명문이 아니라 실제 tensor와 대조한다.

넷째 칸에는 text IDs, media span, position과 attention predicate를 적는다. 다섯째 칸에는 modality별 loss sum·count·weight, total loss와 selected gradient를 적는다. 여섯째 칸에는 DP·TP·PP placement, collective bytes와 global reconstruction을 적는다. 마지막 칸에는 checkpoint keys와 evaluation counterfactual을 둔다.

각 칸의 값이 앞 칸에서 계산되지 않으면 숨은 default가 있다는 뜻이다. 예를 들어 feature 수를 processor 출력에서 읽기만 하고 raw grid로 예측할 수 없다면 merge 정책을 아직 이해하지 못한 것이다. loss를 재계산할 수 없다면 shift·mask·reduction 중 하나가 빠졌다. checkpoint에서 processor generation을 찾지 못하면 resume 의미가 열려 있다.

독자가 피해야 할 여덟 문장.

“이미지는 256 tokens가 된다”는 해상도·processor·merge와 special token을 생략한다. “vision encoder를 freeze했다”는 train mode, cache와 connector gradient를 설명하지 않는다. “multimodal loss를 평균했다”는 분모와 task weight를 감춘다. “같은 image를 썼다”는 crop·decode·processor generation을 보장하지 않는다.

“FlashAttention을 써서 빨라졌다”는 실제 selected kernel, mask 지원과 fallback을 말하지 않는다. “global batch를 키웠다”는 contrastive negative set과 objective 변화를 생략한다. “checkpoint가 복구됐다”는 processor·curriculum·cache와 first update를 검증하지 않는다. “benchmark가 올랐다”는 input policy, contamination과 counterfactual grounding을 말하지 않는다.

좋은 설명은 이 문장을 금지하는 데 그치지 않고 빠진 주어와 상태를 채운다. 어느 revision의 어느 함수가 어떤 config에서 어느 shape·dtype·mask를 만들고, 그 결과 어떤 loss와 gradient가 변했는지 말한다. 실행하지 않은 조합은 명확히 `NOT_RUN`으로 남긴다.

장을 닫으며.

멀티모달 학습은 여러 종류의 파일을 Transformer에 넣는 기술이 아니다. 서로 다른 물리 좌표와 시간축을 유한한 representation으로 바꾸고, 그 representation이 어떤 질문과 target에 인과적으로 연결되는지 보존하는 작업이다. processor, model과 trainer는 이 의미를 나누어 소유한다.

따라서 오류를 찾는 기본 동작은 늘 같다. raw identity에서 시작해 변환 좌표, feature cardinality, position·mask, loss denominator, gradient owner와 durable checkpoint를 순서대로 확인한다. 평균 score나 tensor shape가 맞는다는 사실은 이 사슬을 대신하지 않는다.

독자가 이 사슬을 한 표본에서 재생하고, 결함 하나를 심어 최초 divergence를 찾아내며, resume 뒤 동일 update 의미를 증명할 수 있다면 멀티모달 모델은 더 이상 불투명한 조립품이 아니다. 다음 장에서는 같은 엄격함으로 noise와 data 사이의 경로를 학습하는 diffusion·flow 모델을 해부한다.

독립 자가 점검. 임의의 image-video conversation 하나를 고른다. 선택 frame의 PTS와 image crop, processor bundle, 각 media의 feature rows, placeholder 또는 cross-attention span을 설명하지 못하면 입력 경계를 다시 읽는다. assistant target의 유효 token 수와 modality auxiliary loss의 전역 분모를 손으로 계산하지 못하면 trainer 경계를 다시 읽는다. tower·projector·language parameter 중 누가 gradient를 받고 어느 optimizer group이 update하는지 찾지 못하면 backward와 checkpoint 경계를 다시 읽는다.

그다음 media 두 개의 순서를 바꾸고, mask 한 칸을 열고, feature cache generation을 과거 것으로 만든다. 세 결함이 모두 optimizer commit 전에 서로 다른 명확한 invariant에서 차단돼야 한다. 단지 답변 품질이 나빠지는 것으로 발견해서는 늦다. 마지막으로 rank 하나에 media가 없는 batch와 다른 rank에 긴 video가 있는 batch를 구성해 collective 순서와 global loss가 single-process oracle과 일치하는지 확인한다.

이 시험 결과는 model 이름이나 benchmark 표가 아니라 실행 계약의 성적표다. 새 encoder, processor, adapter, dtype, CUDA kernel 또는 topology를 도입할 때 같은 표를 새 generation으로 다시 만든다. 이전 PASS를 새 조합에 자동 상속하지 않는다. 변화가 계산 비용만 건드리는지, feature 의미·mask·gradient·resume까지 건드리는지를 이 표가 드러내야 한다.

독립 검토자는 성공 표본만 받지 않는다. decode 실패, 빈 audio, 손상된 image, variable-frame-rate video, media 없는 rank, 잘못된 placeholder와 중단된 checkpoint를 함께 받는다. 각 사례에서 어떤 상태가 생성됐고 어디까지 durable해졌으며 무엇이 폐기됐는지를 추적한다. 실패 표본이 queue 재시도 뒤 다른 SampleID로 중복 소비되거나 fallback text로 정상 집계되지 않아야 한다. 수정 뒤에는 incident를 비식별 회귀 fixture로 고정하고 parent suite까지 다시 실행한다. 이때 처리량 회복과 의미 보존을 별개의 열로 판정한다. 그래야 성능 최적화가 정렬·안전·복구 계약을 침식하지 않는다.

결국 독자가 보존해야 할 것은 media 파일 자체만이 아니라 그 파일이 학습 신호가 된 정확한 과정이다. 그 과정이 재생 가능할 때에만 모델의 변화도 설명할 수 있다. 데이터, 수식, 코드, 분산 event와 평가가 같은 표본을 가리키는 상태가 이 장의 진짜 완료선이다.

이 완료선은 새 modality가 추가돼도 변하지 않는 검증의 중심축이 된다. 독자는 지원하려는 modality마다 정상 표본 하나와 의미가 깨진 표본 하나를 골라, 두 표본의 최초 divergence가 예상한 processor·fusion·loss 경계에서 검출되는지 직접 증명해야 한다. 그 결과가 있어야 위의 원장이 설명문을 넘어 다음 release에서 재사용할 수 있는 회귀 계약이 된다.

## 21.16 Janus가 이미지 자리를 언어 embedding으로 치환하는 순간

`prepare_inputs_embeds`를 shape와 상태 전이로 읽는다.

Janus의 `prepare_inputs_embeds`는 `input_ids [B,T]`, `pixel_values [B,N,3,H,W]`, 두 boolean mask를 받아 `[B,T,D]`를 돌려준다. 핵심은 `images = rearrange(..., "b n c h w -> (b n) c h w")`, `images_embeds = self.aligner(self.vision_model(images))`, `inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]` 세 줄이다. vision tower와 aligner가 만든 `[B,N·T₂,D]` 행을 언어 embedding의 이미지 placeholder 위치에 대입한다. 음수 placeholder ID를 0으로 바꾸는 것은 lookup 오류를 피하기 위한 임시 상태일 뿐, 최종 의미는 mask 치환이 소유한다.

이 설계는 decoder가 별도 image branch를 알지 않아도 하나의 embedding sequence를 받게 한다. 대신 `sum(images_seq_mask)==sum(images_emb_mask)`가 깨지면 assignment cardinality가 처음 실패한다. 두 합은 같지만 media 순서가 다르면 shape 검사는 통과하고 의미가 뒤바뀐다. 진단 순서는 `(B,N,H,W) → tower T₂ → aligner D → flattened media order → 두 mask의 true index → 치환 직후 checksum`이다. 최초 불일치가 tower 출력 전이면 processor·vision 경계, T₂/D에서면 aligner, true index에서면 placeholder compiler의 책임이다.

사고실험으로 같은 크기의 이미지 두 장을 서로 바꾸되 text placeholder 순서는 고정해 보자. cardinality와 output shape는 그대로지만 치환 행 checksum이 첫 placeholder에서 달라져야 한다. 달라지지 않으면 media identity를 잃었고, 마지막 답변만 비교한다면 오류를 너무 늦게 발견한다. 체크리스트에는 mask dtype, 음수 ID 원본 보존, media 순서, T₂ 예측값, aligner gradient, 치환 전후 checksum과 resume 뒤 processor generation을 남긴다.

## 21.17 Llama 4의 builder 한 줄에서 expert gradient까지 내려간다

구조 선언과 token별 실행을 분리해 읽는다.

torchtune의 고정 revision `bd2a0fc…a1`에서 Scout builder의 68~84행은 decoder를 48개 층, query head 40개, KV head 8개, expert 16개와 shared expert 하나로 만든다. `attention_chunk_size=8192`와 `skip_rope_interval=4`도 같은 호출에서 굳어진다. 이 수치는 모델 카드의 장식이 아니다. GQA의 KV cache·projection shape, attention mask의 지역성, expert checkpoint key와 어느 parameter에 adapter를 붙일 수 있는지를 동시에 제한한다. 바로 앞 56~66행의 vision encoder는 14픽셀 patch, 최대 16개 tile과 4096차원 projection을 만들고, 85~95행의 `EarlyFusionModel`이 `<|patch|>` token과 encoder trainability를 연결한다.

하지만 `num_experts=16`만 보고 “각 token이 16개 FFN을 모두 지난다”고 쓰면 틀린다. Meta reference revision `0e0b8c…301`의 `MoE.forward()` 181~191행에서는 먼저 `router_scores = x @ W_router`를 계산하고 token별 top-k index만 고른다. 선택되지 않은 칸을 음의 무한대로 덮은 뒤 sigmoid를 적용한다. 이어 입력을 expert 축으로 복제·정렬하고 선택 weight를 곱해 local expert에 보낸다. 따라서 한 token의 routed loss가 만드는 gradient는 적어도 router weight, 선택된 expert와 shared expert 경로로 갈라진다. 선택되지 않은 expert가 그 token에서 0 gradient인 것은 optimizer 고장이 아니라 dispatch 결과일 수 있다.

이 지점에서 LoRA 옵션의 의미도 달라진다. Scout LoRA builder는 decoder, encoder, fusion에 각각 `full`, `lora`, `frozen` 상태를 받고 `apply_lora_to_mlp`가 켜지면 text decoder의 MoE 경로까지 대상으로 삼는다. `target_modules=[q_proj,v_proj]`만 기록해서는 expert 쪽 학습 용량을 재현할 수 없다. release manifest에는 세 하위 그래프의 resolved mode, 실제 adapter parameter key, expert별 gradient 유무, router와 shared-expert의 trainability를 펼쳐야 한다.

가장 작은 정적 walkthrough는 다음 인과열을 보존한다. `YAML component → llama4_scout_17b_16e() → llama4_decoder(num_experts=16) → MoE.forward()의 router matmul → top-k index → sigmoid weight → 선택 expert output → loss → adapter/base gradient`다. 각 화살표에는 config 소비 행, tensor shape와 checkpoint key를 붙인다. builder source는 구조를 입증하고 reference forward는 dispatch 수식을 입증하지만, 두 저장소가 동일 checkpoint 변환과 분산 expert placement를 종단 보장한다고 확대하지 않는다.

실행 가능한 소형 fixture를 설계할 때는 expert 셋, token 넷과 `top_k=2`로 줄인다. router logit을 손으로 정해 선택 index와 sigmoid weight를 계산하고, 선택되지 않은 expert의 해당-token gradient가 0인지 확인한다. expert 두 개의 점수가 동률인 사례에는 tie-breaking과 dtype을 기록한다. 한 expert weight를 손상시키는 negative fixture는 그 expert가 선택된 token에서만 최초 divergence가 나야 하며, shared expert가 모든 차이를 가리는지도 따로 본다. 이 책의 자료 수집에서는 대규모 Llama 4 학습을 실행하지 않았으므로 실제 throughput·expert load balance 수치는 장비 실행 전까지 미검증으로 남긴다.

## 21.18 VQ·LFQ·FSQ의 gradient와 code ownership을 구분한다

VQ는 encoder latent에서 최근접 code vector를 고르고 codebook loss와 commitment loss의 stop-gradient 방향을 반대로 둔다. `z + (z_q-z).detach()` 같은 straight-through estimator는 이산 선택을 미분한 것이 아니라 encoder에 대리 gradient를 흘린다. EMA codebook이면 cluster count·embedding average·dead-code replacement와 rank 간 동기화가 checkpoint state다.

LFQ는 explicit lookup table 없이 factorized binary/sign-like code를 만들 수 있고 entropy·utilization 항이 collapse를 제어한다. FSQ는 각 scalar를 유한 level로 양자화해 Cartesian index를 만든다. 따라서 FSQ에 VQ의 EMA dead-code lifecycle을 그대로 요구하거나 LFQ의 utilization loss를 commitment loss와 같은 것으로 부르면 안 된다.

TiTok처럼 2D token sequence로 압축하는 방식, MAGVIT-v2류 LFQ, Cosmos와 TokenFlow의 video token stream은 같은 질문으로 비교한다. latent와 index shape, causal 여부, chunk state, reconstruction objective, LM handoff, checkpoint state와 공개 training code 범위를 채운다. 빈 칸은 모델 이름으로 추정하지 않는다.

## 21.19 오디오·비디오의 서로 다른 시계를 하나의 학습 계약으로 묶는다

오디오나 비디오를 “토큰으로 바꾼다”는 문장은 변환 결과만 말하고 변환의 책임을 감춘다. 한 음성 표본에는 원본 sample index, STFT frame, encoder frame, codec의 `(codebook,time)`, 언어열의 placeholder와 supervised target이라는 서로 다른 좌표가 있다. 영상에는 container timestamp, presentation timestamp, sampled frame, patch·tubelet, pooling 뒤 feature와 언어 위치가 더해진다. 이 좌표들을 모두 `token position`으로 뭉개면 padding이나 frame sampling 하나가 바뀌었을 때 무엇이 학습 신호를 바꿨는지 역추적할 수 없다.

따라서 표본 원장에는 raw asset digest와 sample rate·time base, decoder revision, resample·crop 정책, frontend 설정, 유효 길이, feature shape, placeholder span, label mask와 update ID를 별 열로 둔다. 각 경계에는 정수 길이식을 둔다. 길이식이 예측한 feature 수와 실제 tensor row, placeholder occurrence가 같아야 한다. 값이 우연히 같아도 media 순서가 다를 수 있으므로 각 segment의 identity와 순서 checksum도 확인한다.

Qwen2-Audio의 placeholder를 convolution 길이식에서 재구성한다.

Transformers revision `35acfa…57a`의 `Qwen2AudioProcessor.__call__`은 prompt 안 `<|AUDIO|>` 개수와 전달된 audio 개수를 먼저 비교한다. feature extractor에는 attention mask와 최대 길이 padding을 강제하고, 유효 Mel 길이 `L`을 읽는다. 이어 `L₁=floor((L-1)/2)+1`, `L₂=floor((L₁-2)/2)+1`을 계산해 단일 audio marker를 `L₂`개로 확장한다. 이 산술은 문자열 편집 편의가 아니다. encoder가 두 단계로 줄인 유효 시간축과 language embedding의 빈자리를 같은 수로 만드는 컴파일 단계다.

`Qwen2AudioEncoder.forward`는 입력 Mel 길이가 `max_source_positions × stride₁ × stride₂`인지 fail closed로 검사한다. convolution 두 번, encoder layer, 평균 pooling을 지난 feature를 projector가 text hidden 차원으로 옮긴다. model 쪽에서는 유효 길이 mask로 padding feature를 버린 뒤 audio marker 수와 feature row 수가 다르면 예외를 낸다. 마지막 `masked_scatter`가 audio vector를 text embedding 자리에 삽입한다. 따라서 `feature_attention_mask`는 단순 padding 보조 정보가 아니라 문자열 확장, feature 선택과 LM position을 잇는 상태다.

손으로 검산할 때는 길이가 서로 다른 audio 두 개를 한 배치에 둔다. 각 `L→L₁→L₂`를 계산하고, prompt별 marker run, 유효 encoder row와 scatter 대상 true count를 비교한다. audio 순서를 바꾸고 text 순서를 고정한 음성 fixture에서는 shape가 같아도 첫 feature checksum이 달라져야 한다. `labels=-100`을 audio marker와 user prompt에 적용했는지, assistant target만 남았는지도 확인한다. 이 검사는 audio understanding 품질을 증명하지 않지만 잘못된 음성과 정답이 조용히 결합되는 오류를 optimizer 전에 막는다.

학습 단계 정책도 명시한다. audio tower를 고정하면 projector와 LM에만 gradient가 흐르는지 parameter별 norm으로 확인한다. tower를 풀면 frontend padding과 layer-drop RNG까지 checkpoint·resume 상태에 들어간다. 음성 길이가 긴 batch가 더 많은 LM 위치를 차지하므로 sample 평균과 유효 target 평균은 다른 objective다. audio frame 수, assistant target 수와 sample 수를 모두 보고해 어느 분모를 사용했는지 숨기지 않는다.

MusicGen의 RVQ stream을 인과 순서와 loss 분모로 읽는다.

neural codec의 `K`개 RVQ codebook은 같은 물리 시간 `t`에 `K`개의 정수를 낸다. 이를 단순히 `[tK,(tK+1),…]`로 평탄화하면 앞 codebook이 뒤 codebook을 보는 순서와 생성 지연이 임의로 결정된다. MusicGen의 delay pattern은 codebook `k`를 시간축에서 지연시켜 한 transformer가 여러 stream을 생성하되 미래 codec symbol을 보지 않도록 한다. `build_delay_pattern_mask`가 만든 BOS·PAD 영역과 생성 가능 영역은 곧 attention 이전의 인과 그래프다.

구현의 `MusicgenForCausalLM.forward`는 `[B,T,K]` labels를 codebook별로 꺼내고 대응 logits를 `[B·T,C]`로 평탄화해 cross entropy를 더한다. pad ID는 `-100`으로 바뀐다. 여기에서 세 가지를 구분해야 한다. 첫째, codebook별 CE 합은 평균과 다르다. 둘째, 각 stream의 유효 target 수가 다르면 같은 샘플도 loss 기여가 달라진다. 셋째, delay mask의 BOS·PAD 위치와 label ignore 위치가 어긋나면 실행은 되면서 잘못된 시각을 예측한다.

최소 oracle은 `K=3`, 물리 frame 네 개로 만든다. 원 codec matrix에 서로 다른 숫자를 넣고 delay pattern 뒤 각 autoregressive 위치의 정답을 손으로 쓴다. 모든 유효 칸에서 `sum(loss_k)`, `sum(valid_k)`와 global mean을 재계산한다. 한 stream의 마지막 frame을 pad로 바꿨을 때 해당 분자·분모만 줄어야 한다. stereo라면 좌우 codebook이 interleave되는 순서도 별 좌표로 보존한다. 생성 waveform의 청감만 비교하면 stream swap이나 한-frame shift를 너무 늦게 발견한다.

평가는 codec와 LM을 분리한다. codec에는 bitrate, reconstruction spectral distance, intelligibility·speaker 보존과 streaming boundary를 둔다. LM에는 token NLL·codebook별 accuracy, 조건 일치와 장기 구조를 둔다. 최종 audio에는 자동 metric과 blind human protocol을 함께 둔다. FAD나 CLAP 계열 점수 하나로 발음, 화자, 음질과 prompt 일치를 모두 설명하지 않는다. decoder·codec checkpoint가 달라지면 같은 integer code가 다른 waveform을 뜻하므로 LM checkpoint만으로 release를 복구할 수 없다.

비디오 frame budget을 학습 데이터와 평가의 일부로 취급한다.

Transformers의 `LlavaNextVideoProcessor.__call__`은 한 frame의 image token 수를 구하고 평균 pooling 비율에 맞춰 4로 나눈 뒤 frame 수를 곱해 `<video>` marker를 확장한다. 즉 `N_video = N_patch,pool × F`다. 여기서 `F`는 원본의 총 frame 수가 아니라 decode와 sampling을 거쳐 실제로 선택된 frame 수다. FPS, 균등 sampling, scene-aware sampling 또는 최대 frame cap을 바꾸면 같은 파일도 다른 LM sequence와 gradient budget을 만든다.

영상 fixture에는 variable-frame-rate와 scene cut을 반드시 넣는다. frame index만 저장하지 말고 container time base와 presentation timestamp를 보존한다. 목표 timestamp에서 decoder가 반환한 frame, resize·crop 좌표, vision layer와 pooling 뒤 row 범위를 기록한다. 같은 `F`라도 선택 시각이 다르면 의미가 다르다. audio가 함께 있다면 sampled video PTS와 audio sample 구간의 오차를 밀리초로 계산하고 허용 범위를 넘으면 AV supervision에서 제외하거나 별 상태로 보낸다.

학습 비용은 `video 개수`가 아니라 `decode pixels + vision patches + LM positions + valid targets`로 계측한다. 긴 영상이 frame cap에 자주 걸리면 corpus의 시간 분포가 잘리고, 짧은 clip을 반복 pad하면 특정 장면이 과대표집된다. realized ledger에는 source duration bin별 선택 frame 수, drop·decode failure, patch와 target token을 남긴다. rank마다 media 길이가 달라 생기는 straggler와 empty-media rank도 관측한다.

평가는 질문 유형을 시간 민감도별로 나눈다. 정적 물체 인식, 사건 순서, 짧은 동작, 장기 변화, audio-visual 동기와 자막 의존을 같은 평균에 숨기지 않는다. frame order를 뒤집기, 핵심 frame 제거, audio shift, caption 제거와 distractor frame 삽입 같은 counterfactual을 둔다. 원본과 교란본 답이 같다면 모델이 실제 시간 증거를 사용하지 않았을 수 있다. benchmark score만으로 grounding을 선언하지 않고 attention·feature attribution은 보조 증거로만 사용한다.

마지막 출시 관문는 한 audio-video 표본을 raw bytes에서 update까지 왕복한다. `sample/PTS → frontend frame → codec 또는 patch 좌표 → projector row → LM position → label mask → loss numerator/count → gradient owner → optimizer update`의 모든 화살표가 고정 revision의 함수와 manifest를 가리켜야 한다. source test가 통과한다는 사실은 이 연결의 구현 근거이지 실제 데이터 mixture의 품질이나 GPU 처리량을 증명하지 않는다. 그런 수치는 동일 recipe·dataset·hardware에서 실행하기 전까지 `NOT_RUN`으로 남긴다.
