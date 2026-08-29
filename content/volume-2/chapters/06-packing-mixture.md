# 6장. packing·mixture·curriculum

4장에서 정리한 문서 계보와 5장에서 확정한 토큰 열은 아직 학습 데이터가 아니다. 문서를 뽑고, 길이를 자르고, 여러 조각을 한 행에 채우고, 정답으로 인정할 위치를 고른 뒤, 그 손실이 실제 optimizer update에 반영되어야 비로소 “이 데이터를 학습했다”고 말할 수 있다. 이 장은 그 사이를 하나의 상태 전이로 잇는다.

> `DocumentRevision → TokenSpan → DrawID → PackPlanID → PackedSampleID → LossMask → ContributionID → UpdateID`

이 화살표는 단순한 처리 순서가 아니다. 왼쪽의 데이터가 오른쪽의 gradient 질량으로 바뀌는 동안 무엇이 삭제·반복·가려졌는지를 추적하는 책임 경계다. 처리량이 올랐는데 품질이 나빠졌거나, 설정한 domain 비율과 실제 학습 비율이 다르거나, 재개 직후 loss가 튄다면 이 경계 중 하나에서 최초 차이가 생긴다.

이 장을 읽을 때는 다음 표를 지도처럼 사용한다.

| 경계 | 입력 상태와 shape | 새로 결정되는 것 | 가장 먼저 볼 실패 징후 |
|---|---|---|---|
| 문서→토큰 | `DocumentID`, byte span → `token_ids[L]` | 길이·특수 토큰·원문 offset | 예상보다 긴/짧은 길이, source별 token 질량 변화 |
| 토큰→draw | source별 token span → ordered `DrawID` | source 선택, 중복, exhaustion | configured share와 draw share의 차이 |
| draw→pack | 여러 `token_ids[L_i]` → row `[T]` 또는 flat `[ΣL_i]` | 절단·overlap·배치 순서·빈칸 | tail 유실, 중복 증가, 처리량만 상승 |
| pack→objective | segment map → attention/position/label mask | 문서 간 context와 유효 target | packed/unpacked logits·gradient 불일치 |
| objective→reduction | `loss_token[B,T]`, `valid[B,T]` → `(loss_sum, valid_count)` | 실제 gradient 분모 | rank별 길이가 다를 때 loss scale 변화 |
| reduction→commit | microbatch 합 → `UpdateID` | 소비 확정, scheduler·selector 시계 | resume 뒤 next ID 또는 mixture version 차이 |

핵심 질문은 “packing을 켰는가”가 아니라 “어느 토큰이 어떤 문맥을 보고, 어느 target이 얼마의 가중치로 전역 분모에 들어갔는가”다. 따라서 이 장에서는 공간 효율, 데이터 보존, 목적함수 보존을 서로 다른 지표로 다룬다. 13장의 scheduler는 여기서 만든 valid-token clock을 받아야 하고, 16장의 분산 재개는 sampler cursor와 열린 packing buffer까지 같은 checkpoint cut으로 묶어야 한다.

## 6.1 원문에서 유효 토큰 손실까지: 데이터 경로의 불변식

### 6.1.1 문서 경계와 packing 손실

packing은 여러 짧은 예제를 고정 길이 sequence에 넣어 padding을 줄인다. 이 설명만 들으면 빈칸을 없애는 무해한 최적화처럼 보인다. 실제로는 packer가 문서의 tail을 버릴지, 긴 문서를 겹쳐 자를지, 서로 다른 문서가 attention으로 연결될지까지 결정한다. 즉 packer는 메모리 배치기이면서 목적함수 컴파일러다.

예를 들어 길이 `T=8`인 행에 길이 5인 A와 길이 3인 B를 넣으면 저장 공간은 완벽하게 찬다. 그러나 A의 마지막 token이 B의 첫 token을 예측하게 두면 독립된 두 문서를 학습한 것이 아니다. 반대로 경계 target만 가리고 B가 A를 계속 보게 두면, loss 표면에는 직접 드러나지 않는 문맥 누수가 남는다. “batch utilization 100%”는 corpus 보존이나 목적함수 보존을 증명하지 못한다.

그래서 `PackedSampleID`는 token bytes의 checksum만 가져서는 부족하다. 각 출력 구간이 어느 `DocumentID`의 어느 token offset에서 왔는지, 절단과 overlap은 무엇이었는지, attention·position·label 정책 revision은 무엇인지 담은 segment map을 가져야 한다.

경계를 가로질러 causal attention을 허용할지 block-diagonal mask로 막을지는 목적함수를 바꾼다. label mask만 막고 attention을 열어 두면 다음 문서가 이전 문서를 context로 본다. 이 차이는 tensor shape가 같아도 생기므로 shape test만으로는 잡히지 않는다.

**왜 packing하는가.** 고정 T row에 길이가 다른 sample 하나씩 넣으면 padding도 attention·MLP를 통과한다. 여러 segment를 한 row에 놓으면 계산 낭비를 줄이지만 이용률은 데이터 보존과 다르다.

| artifact | 형식 | 의미 | resume state |
|---|---|---|---|
| segment | DocumentID/start/length | 원문 token 구간 | source cursor |
| buffer | ordered segments | 미배치 tail | 전체 내용 |
| packed IDs | `[T]` | model input | checksum |
| segment map | ranges→DocumentID | 역추적 | checksum |
| attention group | `[T]`/block mask | context 경계 | policy |
| labels | `[T]` | supervised target | valid count |

row utilization은 `Σu_j/(rows·T)`, supervised utilization은 `ΣN_j/(rows·T)`, corpus preservation은 emitted unique/eligible token이다. 셋을 한 지표로 부르지 않는다.

concatenation은 순서대로 stream을 자르고, first/best-fit은 buffer에서 빈칸에 맞는 segment를 찾는다. sequence packing은 segment-aware mask와 position을 만든다. tail drop/overlap, buffer ordering과 tie-break가 결과를 바꾼다.

A 끝과 B 시작을 붙이면 A 마지막 위치가 B 첫 token을 예측할 수 있다. boundary label을 ignore해야 독립 objective다. B가 A를 context로 보지 않게 하려면 query/key segment가 같은 causal edge만 허용한다. label mask 하나로 attention leakage를 막지 못한다.

position을 segment마다 reset하면 standalone과 맞지만 row에 중복 좌표가 생긴다. continuous position은 구현이 쉽지만 짧은 sample이 높은 위치를 경험한다. learned position과 RoPE에서 효과가 다르므로 logits parity로 선택한다.

긴 document를 T로 자를 때 overlap은 context를 보존하지만 token을 반복하고, tail drop은 utilization을 높여도 coverage를 낮춘다. `dropped_tail,repeated_overlap,emitted_unique`를 source별로 센다.

nanochat 공개 경로의 best-fit packing은 긴 document tail이나 결합되지 않은 token을 버릴 수 있고 cursor가 packing buffer·prefetch·RNG 전체를 보존하지 않는다. “100% batch utilization”을 corpus preservation으로 승격하지 않는다.

**반례 1.** 채우기 쉬운 segment만 사용하면 row utilization 100%와 corpus loss 10%가 동시에 가능하다.

**반례 2.** boundary label은 막았지만 attention을 열어 두면 B loss가 A content에 의존한다.

**실험 6-A.** standalone sample과 packed segment의 logits·gradient를 비교한다. dropout을 끄고 position/mask를 맞춘다.

**실패 주입 6-B.** boundary target 하나를 남기고 contribution ledger가 wrong-document target을 탐지하는지 본다.

**설정 비율과 소비 비율은 다르다.**

### 6.1.2 설정 가중치와 실제 gradient 질량

설정 파일에 `web=0.5, code=0.3, math=0.2`라고 적혀 있어도 optimizer가 그 비율로 gradient를 받는 것은 아니다. 문서 길이가 다르고, source마다 reject·truncation 비율이 다르며, SFT라면 prompt mask 비율도 다르기 때문이다. 따라서 draw, accepted, emitted, supervised token은 서로 다른 분모로 세야 한다.

source `i`의 configured probability를 `q_i`, 뽑은 문서 수를 `D_i`, packer가 쓴 token 수를 `E_i`, loss mask를 통과한 target 수를 `N_i`라 하자. 문서 sampling 비율은 `D_i/ΣD`, token 처리 비율은 `E_i/ΣE`, 실제 목적함수에 가까운 비율은 `N_i/ΣN`이다. 여기에 sample/token weight가 있다면 최종 gradient 질량은 다시 달라진다. 이 단계들을 합쳐 “data mixture” 하나로 부르면 drift의 owner를 찾을 수 없다.

평균 supervised length를 `μ_i`라 하면 document sampling이 만드는 기대 target share는 대략

`r_i = q_i μ_i / Σ_j(q_j μ_j)`

다. 이 식은 왜 document 비율이 맞는데 token 비율은 틀릴 수 있는지 보여 주는 첫 근사다. 실제 운영에서는 길이의 꼬리, source 고갈, packer 절단, mask와 importance weight가 더해지므로 누적 counter로 다시 정산한다. checkpoint에는 설정 weight뿐 아니라 이 counter, source iterator, sampler RNG와 이미 뽑아 prefetch한 항목까지 저장해야 한다.

| counter | 증가 시점 | owner | durable link |
|---|---|---|---|
| requested | sampler decision | sampler | draw journal |
| accepted | loader 성공 | loader | DocumentID |
| raw token | tokenizer output | shard | manifest |
| emitted | packer write | packer | PackedSampleID |
| supervised | mask 완성 | collator/loss | denominator |
| consumed | optimizer commit | trainer | UpdateID |

이 표는 구현의 함수 경계와 그대로 맞아야 한다. 아래 코드는 특정 framework의 원문이 아니라, 실제 구현을 읽을 때 찾아야 할 상태 전이를 축약한 의사 코드다.

```python
draw = sampler.next(source_weights, rng_state)          # DrawID, SampleID
span = reader.read(draw.sample_id)                      # token_ids[L], source_id
plan = packer.place(span, open_bins, packing_policy)    # PackPlanID, ranges
batch = collator.materialize(plan)                      # input_ids[B,T]
labels, valid, weight = objective_masks(batch)          # 모두 [B,T]
loss_token = cross_entropy(logits[:, :-1], labels[:, 1:], reduction="none")
loss_sum = (loss_token * valid[:, 1:] * weight[:, 1:]).sum()
denom = (valid[:, 1:] * weight[:, 1:]).sum()
commit = optimizer_reduce_and_step(loss_sum, denom)     # UpdateID
ledger.link(draw, plan, denom, commit)
```

여기서 가장 위험한 오류는 함수가 예외를 내는 오류가 아니다. `labels`와 `valid`가 한 칸 어긋나거나, source weight가 row마다 한 번 적용되어 길이가 긴 sample의 token마다 적용되지 않거나, `denom`이 rank-local mean의 평균으로 축소되어도 tensor shape는 모두 정상이다. 그래서 각 경계는 shape뿐 아니라 stable ID, 합계와 예상 값을 함께 검사해야 한다.

token별 손실을 `ℓ_{rbt}`, 유효 mask를 `m_{rbt}∈{0,1}`, 가중치를 `w_{rbt}`라 하면 세계 전체에서 의도한 weighted-token mean은

`L = [Σ_r Σ_b Σ_t w_{rbt} m_{rbt} ℓ_{rbt}] / [Σ_r Σ_b Σ_t w_{rbt} m_{rbt}] = S / N`

이다. rank마다 먼저 `S_r/N_r`를 만든 뒤 이를 평균하면 `N_r`가 서로 다를 때 이 식과 달라진다. 올바른 구현은 rank별 `S_r`와 `N_r`를 따로 합치거나, DDP가 gradient를 rank 평균한다는 사실까지 포함해 local loss scale을 `world_size/N`에 맞춘다. gradient accumulation에서도 microbatch mean을 다시 평균하지 말고 numerator와 denominator를 update 경계까지 보존한다.

source capacity가 목표보다 작으면 반복 또는 재분배한다. OLMo-core 계열은 capacity와 repetition 상한을 확인하고 sequence-length sample 수로 바꾼 뒤 Hamilton largest remainder로 정수 allocation한다. floor quota 뒤 fractional remainder가 큰 source에 남은 sample을 준다. tie-break와 source order를 고정한다.

rank가 독립 sampler를 실행하면 world size와 retry가 realized share를 바꿀 수 있다. global sample index와 counter-based RNG는 topology 의존성을 줄이지만 packing buffer와 capacity도 deterministic해야 한다.

rank별 source counters는 sum한다. packed row 하나를 한 source로 세지 않고 segment token 기여를 센다. row 비율, emitted token, supervised token을 구분한다.

**반례 3.** 50/30/20 documents에서 평균 길이가 100/1000/10이면 token mixture는 크게 다르다.

**반례 4.** emitted share가 같아도 prompt mask 비율이 다르면 gradient objective share가 다르다.

**실험 6-C.** 길이·reject·mask 비율이 다른 세 source로 q→draw→accepted→emitted→supervised cascade를 계산한다.

**실패 주입 6-D.** rank 하나의 counter를 누락해 source 합과 global valid count 불변식이 실패하는지 본다.

**curriculum은 다음 batch를 바꾸는 feedback controller다.**

### 6.1.3 curriculum의 feedback 상태

curriculum은 “쉬운 것부터 어려운 것으로 간다”는 표어가 아니다. 관측한 신호를 이용해 다음 batch의 길이·domain·example 확률을 바꾸는 제어계다. 따라서 무엇을 관측했는지, 그 관측이 어느 model revision에서 나왔는지, 언제 sampling에 적용됐는지, 재개할 때 어떤 내부 상태를 복원하는지가 알고리즘의 일부다.

DoReMi는 reference 대비 excess loss로 domain weight를 갱신하고, RegMix는 작은 proxy run에서 mixture와 성능 관계를 회귀한다. DSIR은 target/source n-gram density ratio로 corpus를 재표본화한다. RHO-Loss는 reducible loss가 큰 예를 고른다. Skill-It과 DoGE는 validation skill loss 또는 domain gradient alignment로 다음 window의 확률을 바꾼다. 이름을 한 표에 놓을 수는 있지만 같은 문제의 interchangeable option은 아니다. 차이는 score의 의미, online/offline 여부, importance correction과 mutable state다.

**DoReMi**는 reference loss 대비 token excess loss→domain score→exponentiated-gradient weight→importance-corrected proxy loss→평균 weight export를 잇는다. reference quality와 domain 정의에 조건부이며 공개 PyTorch 구현과 논문 당시 production 구현의 차이를 보존한다.

**RegMix**는 prior→Dirichlet 후보→proxy config/run/metric→regression→large-run 후보 순서다. 후보 밖 extrapolation, seed variance와 regression family가 uncertainty다. config/CSV code와 notebook regression, 미완 test를 구분한다.

**DSIR**은 raw/target hashed n-gram density ratio→document weight→Gumbel top-k without replacement→offline selected corpus다. selector save/load는 output write cursor와 RNG를 원자 복구하는 trainer checkpoint가 아니다.

**RHO-Loss**는 current per-example CE에서 holdout irreducible loss를 빼 reducible loss가 큰 candidate를 고른다. noisy high-loss를 차감하려는 hard-example mining 변형이다. candidate cursor와 proxy identity가 state다.

**Skill-It**은 validation skill loss와 dependency graph로 다음 window mixture를 갱신한다. 공개되지 않은 graph-estimation code가 있다면 graph는 외부 input artifact다.

**DoGE**는 domain별 gradient와 target gradient inner product로 mutable sampling probability를 바꾼다. domain backward/gather 비용과 model·weight file atomicity가 추가된다.

| 방법 | 선택 단위 | 시점 | resume 상태 |
|---|---|---|---|
| DoReMi | domain | online proxy | reference·EG/avg weight |
| RegMix | mixture | offline runs | run/regression lineage |
| DSIR | document | pretraining 전 | counts·RNG·output cursor |
| RHO | candidate example | update | cursor·proxy·loss table |
| Skill-It | skill window | periodic | graph·weight·window |
| DoGE | domain draw | online | gradients·weight·sampler |

sampling을 q에서 p로 바꾸고 원 q objective를 유지하려면 `q(x)/p(x)` importance weight가 필요하다. p가 작은 영역은 variance가 커지고 clipping은 bias를 만든다. objective 변경인지 unbiased efficiency인지 명시한다.

**반례 5.** selector score 개선은 proxy mismatch 때문에 large model 개선을 보장하지 않는다.

**반례 6.** selector weight만 복구해도 cursor/RNG가 다르면 다른 batch를 뽑는다.

**실험 6-E.** toy domains에서 weight update, 확률 합, lower bound, importance corrected loss를 손계산과 비교한다.

**세 상태를 함께 멈춘다.**

**selector·sampler·model의 consistent cut**

checkpoint가 parameter만 보존하면 계산은 다시 시작할 수 있어도 같은 학습은 이어지지 않는다. model step만 저장하고 selector weight가 뒤처지면 재개 이후 다른 data distribution을 학습한다. reader cursor만 저장하고 열린 packing bin을 버리면 이미 읽은 tail이 사라진다. RNG만 복원하고 prefetch queue를 빼면 같은 draw가 중복될 수 있다.

따라서 `CheckpointID`는 model/optimizer뿐 아니라 selector revision, domain weight, sampler cursor, 열린 packing buffer, mixture counters를 하나의 consistent cut으로 묶어야 한다. 여기서 atomic하다는 말은 모든 파일을 같은 순간에 썼다는 뜻이 아니라, manifest가 가리키는 component들이 같은 마지막 committed `UpdateID`를 설명한다는 뜻이다.

update u의 consumed PackedSampleIDs, model/optimizer u, 그 loss로 갱신한 selector, 다음 sampler cursor가 같은 commit ID를 가져야 한다. old model과 future selector를 섞으면 trajectory가 모호하다.

| durable state | identity | 의미 |
|---|---|---|
| model/optimizer | UpdateID | parameter commit |
| cursor | global segment/sample | next draw |
| packing buffer | ordered tails | token preservation |
| RNG | algorithm/key/counter | draw 재현 |
| counters | source×stage | realized share |
| selector | revision/weights | next probability |
| curriculum | schedule/window | next length/domain |

권장 순서는 draw journal→packed batch commit→forward/backward→optimizer commit→selector update→next sampler state→checkpoint manifest publish다. crash는 마지막 consistent commit으로 돌아가고 in-flight draw는 lease ID로 dedup한다.

VSL curriculum은 sequence length와 FLOP, batch shape와 sample order를 함께 바꾼다. 같은 optimizer step이 같은 token progress가 아니다. OLMo-core의 curriculum identity mismatch 거부는 잘못된 schedule resume을 막지만 모든 sample-exact를 증명하지 않는다.

world-size 변경에서 load 가능, topology-portable, sample-exact, numerical equivalence를 분리한다. rank-local buffer/prefetch가 있으면 global permutation만으로 충분하지 않다.

**반례 7.** loss curve가 부드러워도 next PackedSampleID는 달라질 수 있다.

**반례 8.** selector probability가 같아도 RNG/cursor가 다르면 draw가 다르다.

**실패 주입 6-F.** packing buffer를 checkpoint에서 빼 tail coverage divergence를 찾는다.

**실패 주입 6-G.** model u와 selector u+1을 섞어 manifest version gate가 거부하는지 본다.

**실패 주입 6-H.** world size 8→16에서 다음 1000 segment IDs와 counters를 비교한다.

이제 장애를 관측값에서 거꾸로 좁힐 수 있다.

| 관측 | 첫 번째 의심 경계 | 바로 비교할 상태 | 원인을 가르는 실험 |
|---|---|---|---|
| 처리량은 늘었는데 validation이 악화됨 | pack→objective | segment mask·position·boundary label | 같은 child를 unpacked/packed로 실행해 logits·gradient 비교 |
| configured share는 맞지만 code loss 기여가 급감함 | draw→pack→mask | source별 draw/emitted/valid/weighted mass | packer와 mask를 차례로 우회해 최초로 share가 달라지는 지점 확인 |
| world size를 바꾼 뒤 loss scale만 달라짐 | reduction | rank별 `S_r`, `N_r`, DDP 평균 factor | 불균등 valid-count fixture와 단일-process oracle 비교 |
| resume 직후 sample이 반복되거나 사라짐 | cursor→buffer→prefetch | committed/leased/prefetched ID와 열린 bin | checkpoint 직전·직후 next-K DrawID/PackID 비교 |
| selector chart는 바뀌었지만 mixture가 늦게 반응함 | controller→queue | observation/apply step, MixtureVersion, queue depth | queue drain 전후 consumed version histogram 비교 |
| 특정 긴 문서만 거의 학습되지 않음 | chunk→pack | tail drop·truncation·overlap counter | 길이 bucket별 input/emitted/valid 보존율 비교 |

이 순서의 장점은 model kernel을 섣불리 의심하지 않는 데 있다. `DrawID`가 처음부터 다르면 sampler 문제이고, draw는 같은데 `PackPlanID`가 다르면 buffer·tie-break 문제다. pack까지 같은데 loss가 다르면 mask·position·denominator로 넘어간다. 최초 차이가 생긴 owner에게 조사를 멈추면 원인과 결과를 뒤섞지 않는다.

**조사 체크리스트.** tail/overlap/boundary mask/position을 찾는다. 네 mixture counter와 capacity/repetition을 확인한다. selector score·importance correction·mutable state를 적는다. checkpoint buffer/RNG/cursor/counter/selector/curriculum을 본다. topology 변경 계약과 test 범위를 확인한다.

**결정 트리.** coverage가 다르면 cursor→buffer/tail을 본다. packed logits만 다르면 mask/position이다. source share가 다르면 length→reject→packing→mask 분모다. resume만 다르면 buffer→RNG→cursor→selector version이다.

DSIR toy test는 fit/resample/save-load를 확인하지만 trainer atomic resume를 증명하지 않는다. RHO smoke test도 selection invariant 전체가 아니다. Skill-It/DoGE integration test가 공개되지 않은 경계는 미확인으로 남긴다.

**실습 6-I.** 4장의 세 source fixture를 pack해 네 분모를 계산한다. checkpoint에서 buffer/sampler를 복원해 다음 100 IDs를 비교하고 state 하나씩 빼 first divergence를 기록한다.

**실제 인계.** 7장에는 packed IDs `[B,T]`, segment-aware mask, position IDs, labels/valid count를 넘긴다. 13장에는 token/curriculum progress, 15·17장에는 rank ownership/atomic state, 24장에는 consumed IDs와 exposure를 넘긴다.

**Packer 호출 흐름.** source reader가 `(DocumentID,token_ids,offsets,source_id)`를 내고, chunker가 max length와 overlap policy로 segment를 만든다. bin selector가 현재 free space에 segment를 배치하고, collator가 BOS/EOS·boundary label·attention group·position을 만든다. batch assembler가 B rows를 묶고 device transfer한다. 각 함수 사이 payload checksum을 남기면 tail 손실이 어느 단계에서 생겼는지 찾을 수 있다.

**Buffer ordering.** best-fit은 남은 공간과 segment length를 비교하므로 future segment를 일정 window만큼 미리 봐야 할 수 있다. window 크기는 packing quality와 memory/latency, source order를 바꾼다. tie에서 먼저 도착한 segment를 고르면 worker scheduling이 output을 바꾼다. stable `(length,DocumentID,segment_index)` ordering을 test한다.

**Prefetch와 durable cursor.** reader cursor가 segment를 buffer에 넘긴 순간 전진하고 buffer가 memory에만 있으면 checkpoint 뒤 그 segment가 사라진다. cursor를 buffer 소비 뒤 commit하거나 buffer payload를 checkpoint해야 한다. async prefetch queue도 같은 문제다. queue head/tail sequence와 lease status를 저장한다.

**Packed sample identity.** `PackedSampleID`는 row bytes checksum만으로 만들면 같은 token IDs라도 다른 segment lineage를 구분하지 못한다. segment IDs·ranges, mask/position policy revision, token IDs/labels checksum을 canonical serialization해 만든다. source deletion은 segment map으로 역추적한다.

**Loss contribution identity.** packed row 안의 position마다 `ContributionID=(PackedSampleID,position,target_id)`를 둘 수 있다. boundary/ignored position은 reason을 갖고 contribution set에 들어가지 않는다. UpdateID는 consumed ContributionID range와 loss sum/count를 가리킨다.

**Varlen kernel 계약.** cumulative sequence lengths `cu_seqlens=[0,l_1,l_1+l_2,…]`는 flattened token의 segment 경계를 나타낸다. 마지막 값은 total tokens, 차이는 각 length, 단조 증가와 max length가 불변식이다. ID/label flatten order와 같은 순서여야 한다. 잘못된 offset은 shape가 맞아도 sample 간 attention을 섞는다.

**Padding-free와 position.** flattened batch는 `[total_tokens,C]`로 계산할 수 있지만 loss를 원 `[B,T]` dashboard와 연결하려면 row/segment mapping이 필요하다. rotary position은 segment-local 또는 document-continuous policy를 명시한다. sequence parallel shard가 segment를 나누면 position/causal boundary를 rank가 공유해야 한다.

**Packing과 flash attention.** block-diagonal dense mask를 만들지 않고 varlen attention을 쓰면 memory를 줄일 수 있다. 그러나 kernel이 dropout RNG를 flattened index에 매핑하는 방식, backward offset, head layout을 reference와 비교한다. dense masked reference와 output/dQ/dK/dV parity test를 둔다.

**Curriculum의 세 축.** length curriculum은 T, difficulty curriculum은 example score, domain curriculum은 mixture weight를 시간에 따라 바꾼다. 여러 축을 동시에 바꾸면 어느 요인이 효과를 냈는지 알기 어렵다. schedule state는 current phase, transition rule, observed metric, counter unit(step/token/sample)을 가진다.

**VSL token 회계.** B를 고정하고 T가 커지면 step token이 늘고 attention FLOP은 더 빠르게 증가한다. global token batch를 고정하려 B를 줄이면 gradient noise와 kernel utilization이 변한다. “같은 step” 비교 대신 processed/supervised token과 estimated/measured FLOP을 쓴다.

**Schedule transition.** fixed step, token threshold, validation trigger는 다른 state machine이다. validation-trigger는 evaluation noise와 frequency가 phase를 바꿀 수 있다. checkpoint에는 마지막 metric, patience/cooldown과 next transition eligibility가 필요하다. config schedule만 저장하면 부족하다.

**Feedback delay.** online selector가 window u의 loss를 보고 u+1 이후 probability를 바꾼다. rollout/prefetch queue가 이미 old weight로 sample을 뽑았다면 effective delay가 늘어난다. each draw에 `MixtureVersion`을 붙여 실제 적용 시점을 측정한다.

**Selector stability.** exponentiated update는 score scale과 learning rate에 민감하다. 작은 probability가 0에 가까워지면 importance weight variance가 폭발한다. floor, entropy regularization, smoothing은 collapse를 막지만 target mixture와 bias를 바꾼다. option별 probability min/max/entropy를 기록한다.

**DoGE gradient 비용.** domain별 gradient vector를 직접 만들면 domain 수만큼 backward 또는 vectorized gradient가 필요할 수 있다. flat gradient inner product는 parameter group, dtype, distributed reduction에 의존한다. norm이 큰 domain이 alignment를 지배할 수 있어 normalization/clipping 식을 확인한다.

**RHO candidate bias.** oversized candidate batch 자체가 source sampler에서 편향돼 있으면 top-k selector가 보지 못한 domain을 고를 수 없다. irreducible loss table의 dataset index가 current example과 정확히 맞아야 한다. shuffle/reindex 뒤 stale table을 쓰는 failure test를 둔다.

**DSIR density ratio.** target/source feature distribution `p_t(z),p_s(z)`에서 importance weight는 `p_t(z)/p_s(z)`다. hashed n-gram collision과 smoothing은 zero count와 variance를 제어한다. Gumbel top-k는 without-replacement subset을 만들지만 RNG seed와 shard virtual ordering이 materialized corpus identity를 바꾼다.

**RegMix experiment lineage.** candidate mixture ID가 proxy config, source commit, seed, checkpoint, metric row, regression training row와 이어져야 한다. failed/missing runs를 조용히 제외하면 selection bias가 생긴다. prediction confidence와 observed large-run residual을 남긴다.

**DoReMi reference leakage.** reference model이 target validation/domain corpus를 어떻게 학습했는지에 따라 excess loss 의미가 달라진다. reference checkpoint·tokenizer·domain mapping을 고정한다. proxy와 reference loss denominator가 같아야 subtraction이 의미 있다.

**실패 주입 6-J—stale irreducible index.** dataset shuffle 뒤 RHO loss table index를 갱신하지 않는다. selected examples와 score correlation이 깨지는지 example stable ID assertion으로 잡는다.

**실패 주입 6-K—mixture version lag.** sampler queue를 두 window 미리 채워 selector update 후에도 old version sample이 소비되게 한다. UpdateID별 drawn/consumed MixtureVersion histogram으로 lag를 측정한다.

**실패 주입 6-L—curriculum counter 단위.** step counter를 token counter로 잘못 복원해 length phase가 조기 전환되게 한다. checkpoint schedule schema가 unit mismatch를 거부해야 한다.

**실패 주입 6-M—Hamilton tie nondeterminism.** 같은 fractional remainder 두 source의 입력 order를 바꾸어 allocation이 달라지는지 본다. stable SourceID tie-break로 고친다.

**실험 6-N—packer 비교.** 동일 ordered segments에서 concatenation, first-fit, best-fit, varlen을 실행한다. row/supervised utilization, coverage, buffer peak, build time, standalone logits parity를 비교한다. throughput만으로 선택하지 않는다.

**실험 6-O—world-size fixture.** global deterministic segment sequence를 1/2/4 ranks에 배정하고 consumed global order, source counters, packing decision을 비교한다. exact를 요구하지 않는 design이면 어떤 property만 유지되는지 선언한다.

**실험 6-P—selector resume.** update u 전후 각 crash point에서 model, selector, sampler, packing buffer를 복원한다. uninterrupted reference의 next 100 IDs, weights, loss denominator, parameter delta와 비교한다.

**Upstream test와 제안 fixture.** OLMo의 allocation/curriculum identity test는 해당 integer allocation과 mismatch fail-fast를 지지한다. DSIR의 fit/resample/save-load는 selector object 동작을 지지한다. 그러나 packer→trainer→checkpoint atomicity 전체는 이 test들의 합으로 증명되지 않는다. 책의 fixture를 별도 integration test로 둔다.

**Metric dashboard.** configured/drawn/accepted/emitted/supervised/consumed share, packing row/supervised utilization, dropped/repeated tokens, buffer age/size, selector entropy/min probability, MixtureVersion lag, source별 loss·gradient norm을 UpdateID window로 본다. 평균뿐 아니라 rank min/max를 둔다.

**장애 조사 dossier.** failing CheckpointID와 parent, next expected/observed segment, packing buffer diff, RNG/cursor, source counters, selector weights/version, curriculum state, rank topology를 한 묶음으로 export한다. loss curve screenshot은 원인 증거가 아니다.

**Release 삭제와 mixture.** 4장의 RevocationID로 source capacity가 줄면 allocation과 repetition이 바뀐다. tombstoned segment가 buffer/cache에 남지 않는지 확인하고 mixture manifest를 새 revision으로 발행한다. 기존 checkpoint의 consumed ledger는 지우지 않고 revoked exposure로 표시한다.

**종료 조건.** 독자는 document weight가 supervised-token objective share로 바뀌는 모든 분모를 계산해야 한다. packed row를 standalone reference와 비교하고, selector/curriculum이 next draw를 바꾸는 상태를 열거해야 한다. crash 뒤 같은 next sample을 요구하는지, topology-portable만 요구하는지 보장 수준을 구분해야 한다.

**Code review 순서.** dataset `__iter__/__getitem__`에서 stable ID와 cursor를 찾는다. tokenizer shard reader의 boundary/EOS를 확인한다. packer의 buffer 자료구조와 tail/overlap policy를 읽는다. collator의 labels, attention mask, position IDs를 본다. sampler probability와 capacity allocation을 찾는다. selector가 score를 언제 갱신하고 sampler가 언제 읽는지 추적한다. checkpoint save/load field와 commit 순서를 대조한다. tests가 어느 state를 assertion하는지 표시한다.

**Mask 조사.** packed IDs가 같아도 attention backend별 mask 표현이 다를 수 있다. boolean은 True가 허용인지 차단인지 API마다 다르고 additive mask는 0/`-inf` convention을 쓴다. varlen은 mask 대신 offsets를 쓴다. 작은 두 segment fixture에서 probability forbidden block이 정확히 0인지 본다. BF16의 finite minimum을 `-inf` 대용으로 쓸 때 leakage tolerance를 확인한다.

**Gradient attribution.** source별 supervised token count만으로 source별 gradient contribution을 정확히 분해할 수 없다. 같은 packed forward의 per-source loss를 별도 backward하면 비용이 늘고 shared context가 있으면 attribution 정의가 모호하다. diagnostic subset에서 source loss sum과 gradient norm/cosine을 측정하고 일상 telemetry의 token share와 구분한다.

**Mixture와 scheduler.** source mixture가 평균 sequence length와 valid count를 바꾸면 update당 token이 변한다. step 기반 scheduler는 같은 step에서 서로 다른 token progress를 가진다. Checkpoint에는 scheduler counter뿐 아니라 cumulative emitted/supervised token을 둔다. mixture experiment는 same step이 아니라 same token/FLOP budget 비교도 제공한다.

**Repetition ledger.** source capacity가 작아 epoch가 반복될 때 `(DocumentID,segment_id,occurrence_index)`를 구분한다. unique segment coverage, average/max repeat와 최근 repeat distance를 센다. curriculum이 쉬운 source를 초기에 반복하고 후반에 줄이는 경우 total share만으로 time-local overfitting을 놓칠 수 있다.

**Data loss spike 결정 트리.** source share가 갑자기 변했는지 본다. changed source의 reject/truncation/valid count를 확인한다. packer dropped tail과 buffer age를 본다. selector weight·entropy·version lag를 확인한다. curriculum phase와 T를 본다. 모두 같으면 model/optimizer로 넘어간다. 먼저 LR을 낮추지 않는다.

**OOM 결정 트리.** nominal T/B와 actual packed/varlen token을 확인한다. longest segment와 buffer memory를 본다. dense block mask가 materialize됐는지 확인한다. attention backend fallback과 saved tensor를 본다. curriculum transition 직후면 batch/accumulation policy가 T 변화에 맞게 조정됐는지 본다.

**Sample repeat 결정 트리.** exact recent PackedSampleID ring을 확인한다. repeated segment가 intentional overlap/repetition인지 sampler retry인지 reason을 본다. worker lease와 commit journal을 확인한다. cursor rollback, prefetch replay, checkpoint parent를 본다. Bloom 경보만으로 확정하지 않는다.

**권장 manifest 구성.** 먼저 실행의 신원을 정하는 `RunID`, `UpdateID`, `MixtureVersion`, `CurriculumVersion`, `world_size`를 기록한다. 그다음 source별 여섯 counter와 draw lease, 정렬된 segment ID, PackedSampleID와 segment map digest를 붙인다. 마지막으로 labels·mask·position, buffer, sampler RNG·cursor, selector 상태의 digest와 누적 token·FLOP를 더한다. 이 값들은 서로 다른 시점의 상태를 이어 붙이지 말고 하나의 consistent cut에서 채취해야 한다.

**독자 확인 문제.** source A/B document probability가 0.5/0.5이고 평균 supervised length가 100/400이면 기대 token share를 구한다. A reject 20%, B truncation 50%를 더해 근사를 갱신한다. 각 microbatch mean을 평균했을 때 source weighting이 어떻게 다시 변하는지 설명한다.

**마지막 인계 gate.** 7장이 읽을 packed IDs, position, mask, labels의 checksum이 6장 manifest와 같아야 한다. 17장이 읽을 buffer/RNG/cursor/selector digest가 CheckpointID와 같은 commit이어야 한다. 24장이 읽을 consumption/exposure ledger의 supervised-token 합은 training loss denominator 누계와 reconciliation돼야 한다.

**최소 invariant 모음.** 모든 emitted token은 하나의 segment/offset ancestor를 가진다. overlap을 제외한 unique emitted와 dropped tail의 합은 eligible token과 맞는다. segment boundary 밖 attention probability는 정책상 0이다. loss sum의 valid count는 source별 supervised counter 합과 같다. allocation sample 합은 requested total과 같다. sampler probability 합은 1이고 lower bound를 지킨다. resume next ID와 mixture/curriculum version은 uninterrupted reference와 요구 수준에 맞는다.

**성능 주장 규칙.** packing으로 tokens/sec가 올랐다고 할 때 nominal·emitted·supervised tokens/sec를 모두 쓴다. varlen kernel과 dense padding 경로는 같은 model/dtype/hardware/revision, 같은 sample set에서 비교한다. selector가 quality를 높였다는 주장은 같은 token/FLOP budget과 seed distribution, baseline 및 confidence를 요구한다. proxy 결과를 large run 결과처럼 쓰지 않는다.

**공개되지 않은 경계.** production run 전체의 source별 requested→consumed 시계열, selector/model/sampler atomic checkpoint, world-size 변경 sample-exact 결과가 공식 자료에 없으면 결손으로 둔다. 설계 가능한 것과 공개 구현이 증명한 것을 구분한다.

**6장 완료 판정.** packing을 padding 제거 알고리즘으로만 설명하지 않고 objective·lineage·resume 변화로 설명한다. configured mixture에서 supervised objective까지 네 분모 이상을 재구성한다. curriculum/selector가 next batch를 바꾸는 feedback state를 찾는다. crash injection 결과를 기준 run과 ID·checksum으로 비교할 수 있을 때 다음 장으로 넘어간다.

검사 결과에는 성공한 sample뿐 아니라 drop·retry·stale version·미복구 buffer도 포함한다. 예외를 분모 밖으로 숨기면 utilization과 재현성 수치가 동시에 과장된다. 각 예외는 reason과 영향 token 범위를 가진다.

이 예외 원장까지 다음 장과 checkpoint 장에 함께 넘긴다.

## 6.2 packing은 공간 절약이 아니라 목적함수 변환이다

### 6.2.1 경계 정책을 식과 tensor로 검산한다

길이가 각각 `l_1,…,l_m`인 문서 조각을 한 row에 놓았다고 하자. 단순 causal mask는 위치 `i`가 `j≤i`인 모든 key를 보게 한다. 독립 문서 목적함수를 보존하려면 여기에 `segment(i)=segment(j)` 조건을 곱해야 한다. 따라서 허용 행렬은 `A_ij=1[j≤i]1[s_i=s_j]`다. target도 같은 경계를 따라야 한다. 위치 `i`의 다음 token이 다른 segment라면 그 label을 무시하거나, 명시적으로 삽입한 EOS를 예측하게 해야 한다. 두 선택은 같지 않다. 첫 선택은 경계 전이를 손실에서 지우고, 둘째는 문서 종료 확률을 학습한다. 어느 쪽이 옳은지는 corpus 생성 규칙과 추론 시 EOS 의미에 달려 있다.

이 차이를 코드에서 찾을 때 `input_ids`만 읽어서는 안 된다. collator가 만든 `labels`, `loss_mask`, `position_ids`, `cu_seqlens`, `max_seqlen`, segment identifier를 한 묶음으로 추적한다. Hugging Face 계열 collator는 종종 label의 일부를 `-100`으로 바꾸고 일반 causal mask를 그대로 사용한다. padding-free trainer는 여러 sample을 평탄화한 뒤 `position_ids`가 0으로 돌아가는 지점을 FlashAttention varlen 경계로 이용하기도 한다. 같은 이름의 `packing=True`라도 전자는 label 경계만, 후자는 attention 경계까지 표현할 수 있으므로 옵션 이름이 아니라 생성 tensor를 비교해야 한다.

작은 oracle은 네 문장만 있으면 된다. tokenized length가 3, 5, 6, 9이고 `T=8`인 fixture를 만든다. 각 token에 원문 좌표를 나타내는 서로 다른 정수를 심는다. packer 출력에서 좌표가 중복·소실됐는지 먼저 확인한다. 이어 dense block-diagonal FP64 attention과 실제 backend의 output을 비교한다. 마지막으로 각 segment를 단독 실행했을 때의 supervised position별 logits와 gradient를 packed 결과에서 역색인해 비교한다. dropout, stochastic layer, sequence-dependent normalization은 끄거나 같은 RNG ledger를 사용한다. 단순 loss scalar 일치는 오류가 서로 상쇄될 수 있으므로 충분하지 않다.

position reset은 RoPE에서 특히 조심한다. segment B가 row의 물리 위치 5에서 시작해도 독립 문서와 같은 함수를 원하면 position 0부터 회전해야 한다. 그러나 kernel이 `cu_seqlens`만 받고 position을 물리 offset으로 유도한다면 attention 경계는 맞고 회전 각도는 틀릴 수 있다. 반대로 continuous position을 의도한 장문 stream이라면 reset이 오류다. 문서 조각이 원문에서 offset 4096부터 왔을 때 segment-local, document-continuous, packed-row-continuous 가운데 무엇을 쓰는지 `PositionPolicyID`로 남긴다. 이름 없는 boolean 하나로는 재현할 수 없다.

EOS 삽입도 token 회계를 바꾼다. 원문 token N개에서 문서마다 EOS 한 개를 추가했다면 emitted token은 원문 coverage보다 커질 수 있다. 따라서 `raw_token`, `synthetic_boundary_token`, `repeated_overlap`, `supervised_token`을 별도 열로 센다. EOS가 이미 원문 끝에 있었는데 loader와 packer가 각각 붙이면 중복 EOS가 생긴다. tokenizer template, dataset preprocessing, collator 가운데 누가 경계 token의 owner인지 한 곳으로 정한다. owner가 둘이면 문서 수가 큰 corpus에서 작은 오류가 막대한 학습 질량으로 누적된다.

### 6.2.2 bin packing 품질과 재현성을 분리한다

first-fit decreasing은 segment를 길이순으로 정렬한 뒤 처음 맞는 bin에 넣어 빈칸을 줄인다. 하지만 global 정렬은 corpus 순서를 바꾸고, 거대한 buffer를 요구하며, 분산 streaming과 잘 맞지 않는다. online best-fit은 제한된 lookahead에서 남는 공간이 가장 작은 bin을 고른다. 이 방법의 결과는 lookahead 크기, 동률 규칙, worker 도착 순서에 의존한다. packing ratio만 보고 선택하면 재현성·source mixing·curriculum 순서를 잃을 수 있다.

packing 방식은 계산 효율, 통계적 의미, 운영 가능성의 세 축에서 평가한다. 계산 효율은 physical slot 가운데 실제 token이 차지하는 비율로 잰다. 통계적 의미를 판단할 때는 원래 sample 순서와 source/window mixture가 얼마나 보존되는지 본다. 운영 측면에서는 buffer를 checkpoint할 수 있는지, topology가 바뀐 뒤 어느 수준까지 재생할 수 있는지가 중요하다. offline packing을 완벽하게 최적화하면 slot 활용률은 높아지지만 epoch마다 거대한 materialization artifact가 생길 수 있다. streaming concatenation은 cursor가 명확하고 구현도 단순한 대신, 문서 경계를 넘는 objective를 허용할지 정확히 차단할지 정해야 한다. 따라서 어느 방식이든 한 축의 장점만으로 우월하다고 결론낼 수 없다.

분산 환경에서는 global pack과 local pack을 구분한다. global sample stream을 rank에 나눈 뒤 rank별 packer가 독립적으로 빈칸을 채우면 world size가 바뀔 때 segment 조합도 바뀐다. 같은 token을 소비해도 packed context와 dropout indexing이 달라져 trajectory가 달라진다. topology-portable만 요구한다면 허용할 수 있으나 sample-exact라고 부르면 안 된다. exact를 원하면 pack decision 자체를 global deterministic artifact로 만들거나, virtual packer shard와 counter-based RNG를 physical rank와 분리해야 한다.

packer state의 최소 단위는 source cursor 하나가 아니다. 아직 bin에 들어가지 않은 ordered segment, 부분적으로 채운 bin, lookahead queue, 이미 읽었으나 아직 commit되지 않은 lease, synthetic EOS 여부, overlap의 다음 offset을 저장해야 한다. serialization 뒤 복원한 state에서 다음 K개의 `(PackedSampleID, segment ranges, labels checksum)`가 uninterrupted reference와 같은지 검사한다. 메모리 객체를 pickle했다는 사실보다 canonical state와 다음 출력의 동등성이 중요하다.

실패는 crash 시점별로 주입한다. source를 읽은 직후, segment를 buffer에 넣은 직후, row를 만들었지만 trainer에 넘기기 전, microbatch forward 뒤 optimizer commit 전, optimizer commit 뒤 cursor publish 전에 프로세스를 중단한다. 각 시점에서 at-most-once와 at-least-once 가운데 어떤 계약을 택했는지 확인한다. 실전에서는 draw journal과 update commit을 연결해 중복 소비를 탐지할 수 있다. 단순히 loss curve가 이어지는지는 데이터 손실과 중복을 거의 잡지 못한다.

**mixture를 확률표가 아니라 질량 보존 문제로 본다.**

### 6.2.3 loss denominator까지 질량을 보존한다

source `i`의 설정 확률을 `q_i`라 하자. document draw 이후 filter acceptance `a_i`, 평균 emitted length `e_i`, supervised fraction `m_i`, per-token loss weight `w_i`가 있으면 기대 objective 질량은 대략 `q_i a_i e_i m_i w_i`에 비례한다. 실제 값은 길이와 acceptance, mask가 상관될 수 있어 단순 곱으로 정확히 복원되지 않는다. 그래서 document 단위 평균만 저장하지 않고 각 stage의 합과 교차 통계를 남긴다. 특히 품질 filter가 긴 문서를 더 자주 거부하거나 instruction source의 prompt 부분이 길면 설정 mixture와 gradient mixture가 크게 벌어진다.

loss reduction도 mixture다. microbatch마다 `loss_sum/valid_count`를 계산한 뒤 microbatch 평균을 같은 비중으로 더하면 valid token이 적은 batch가 과대표현된다. gradient accumulation에서 올바른 global token mean을 원한다면 모든 rank와 microbatch의 loss sum을 합하고 valid count 합으로 나누어야 한다. 구현상 backward 전에 분모를 모를 수 있으므로 각 microbatch loss sum을 적절히 scale하거나 accumulated gradient를 최종 global count로 보정한다. DDP가 gradient를 world-size로 평균하는지 합하는지도 식에 포함한다.

예를 들어 rank 0의 source A가 valid token 100개, rank 1의 source B가 400개이고 각 token 평균 loss가 2와 1이라 하자. rank별 mean을 평균하면 1.5지만 global token mean은 `(200+400)/500=1.2`다. 이 차이는 metric 표기 문제가 아니라 gradient coefficient의 차이다. packed row 수나 sample 수를 분모로 쓴 loss가 어떤 objective를 구현하는지 명시한다. aux loss가 sample mean이고 main loss가 token mean이면 두 coefficient의 상대 크기도 batch shape에 따라 바뀔 수 있다.

Hamilton allocation은 정수 batch quota를 만들 때 유용하다. 총 K draw에서 `Kq_i`를 계산하고 floor를 배정한 뒤 남은 수를 fractional remainder 순으로 준다. 하지만 작은 K에서는 한 source가 오랫동안 0개일 수 있다. window마다 독립 rounding하면 주기적 편향이 생긴다. 누적 deficit, stochastic rounding, alias sampling은 서로 다른 variance와 resume state를 가진다. 결과를 평가할 때 장기 realized share뿐 아니라 최대 starvation interval과 time-local share를 본다.

capacity가 부족한 source를 처리하는 정책도 objective 일부다. exhaustion에서 stop, repeat, replacement sampling, 다른 source로 renormalize 가운데 무엇을 하는지 명시한다. renormalize는 후반 mixture를 바꾸고, repeat는 exposure를 늘리며, stop은 token budget을 줄인다. `max_epochs`나 repetition cap을 두면 어느 시점에 어떤 source가 포화되는지 예측할 수 있다. checkpoint는 source별 occurrence counter와 exhaustion state를 보존해야 한다.

**adaptive curriculum의 제어 루프를 안정화한다.**

adaptive selector는 관측기, score 함수, controller, actuator, plant로 나누어 읽는다. validation 또는 training loss가 관측이고, excess loss·gradient alignment·reducible loss가 score다. exponentiated update나 optimizer가 controller이며 sampler probability가 actuator다. 실제 model 학습이 plant다. 이 분해를 하면 delay, noisy observation, saturation, stale actuation을 찾기 쉽다.

DoReMi는 어려운 domain을 무조건 더 많이 뽑는 방식이 아니다. proxy model의 loss가 reference model보다 특히 큰 domain을 강조하되, importance correction으로 원래 target objective와의 관계를 유지한다. reference가 지나치게 약하거나 domain마다 tokenization이 다르면 excess loss를 같은 척도로 비교할 수 없어 score의 의미가 흐려진다. 따라서 domain loss는 valid token의 합으로 계산하고 distributed reduction을 마친 뒤 weight를 갱신해야 한다. rank마다 서로 다른 weight를 잠시라도 사용하면 같은 update 안에서 실제 mixture가 갈라진다.

DoGE류 gradient alignment는 domain gradient `g_i`와 target gradient `g_t`의 내적을 본다. 양의 내적은 그 domain step이 target loss를 줄일 국소 가능성을 뜻하지만, curvature와 finite step 때문에 장기 개선을 보장하지 않는다. parameter subset, normalization, mixed precision scaling, distributed reduction 순서가 score를 바꾼다. 모든 parameter gradient를 materialize하는 비용을 줄이려고 projection하거나 일부 layer만 쓰면 그 근사도 configuration identity에 넣는다.

RHO-Loss는 현재 loss에서 irreducible loss estimate를 빼 noisy-but-hard와 learnable-hard를 구분하려 한다. stable ExampleID가 어긋나면 전혀 다른 example의 irreducible 값을 빼게 된다. 후보 pool이 source mixture를 이미 왜곡했다면 selector는 pool 밖을 복구하지 못한다. selection rate가 낮을수록 compute 절감 가능성과 selector bias가 함께 커진다. 선택되지 않은 example도 exposure ledger에 candidate-but-rejected로 남겨야 사후 감사를 할 수 있다.

feedback controller의 안전장치는 probability floor, maximum ratio, entropy regularization, EMA, cooldown, delayed validation, rollback이다. floor는 coverage를 지키지만 비효율 domain을 완전히 끌 수 없고, EMA는 noise를 줄이지만 반응을 늦춘다. clip은 importance weight variance를 줄이는 대신 bias를 만든다. 각 장치를 안정성 향상이라는 말로 뭉뚱그리지 않고 무엇을 제한하며 어떤 bias를 도입하는지 식과 metric으로 적는다.

**현장에서 재현 가능한 실험으로 닫는다.**

**ablation, 관측, 복구를 하나의 실행표로 묶는다.**

첫 실험군은 packing policy만 바꾼다. 동일한 ordered segment artifact, model checkpoint, optimizer state를 사용해 unpacked padding, concatenation, segment-masked dense, varlen 네 경로를 비교한다. 비교 열은 raw·emitted·unique·supervised token, padding slot, synthetic boundary, peak host/device memory, collator time, forward/backward time, output·gradient parity다. throughput이 빨라도 coverage나 objective가 달라진 경로는 동등 최적화가 아니라 별도 실험으로 분류한다.

두 번째 실험에서는 mixture의 분모가 pipeline을 통과하며 어떻게 달라지는지 검증한다. 이를 드러내려고 source마다 길이, reject rate, prompt mask 비율을 의도적으로 다르게 둔다. configured probability에서 출발해 requested document, accepted document, raw token, emitted token, supervised token, weighted loss 질량으로 이어지는 waterfall을 그린다. 예상식과 실제 counter의 차이는 residual로 남긴다. residual이 크다면 stage 사이의 상관이나 누락된 event를 조사한다. dashboard에도 비율만 띄우지 말고 분자·분모와 집계 window를 함께 저장한다.

세 번째 실험에서는 selector를 open-loop와 closed-loop로 나누어 검증한다. 먼저 고정한 weight schedule을 replay하면 model의 변동을 섞지 않고 sampler와 checkpoint만 확인할 수 있다. 그다음 synthetic score를 넣어 controller update 식, floor, normalization, version publish를 검사한다. 이 두 단계가 통과한 뒤 실제 validation/loss observer를 연결한다. 처음부터 실제 observer를 붙이면 model noise와 controller bug가 한꺼번에 나타나 원인을 구분하기 어렵다.

분산 검증은 physical topology를 바꾸기 전에 virtual global order로 시작한다. world size 1, 2, 4가 동일 virtual shard와 pack decision을 읽게 하고 global consumed ledger를 정렬해 비교한다. exact가 필요한 경우 순서와 packed grouping까지 같아야 한다. topology-portable 계약이면 unique coverage, mixture window, token budget만 같고 grouping은 달라도 되는지 명시한다. 숫자 하나로 두 계약을 섞지 않는다.

관측 항목은 `packing_utilization`, `supervised_utilization`, `unique_coverage`, `tail_drop_tokens`, `overlap_repeat_tokens`, `boundary_targets`, source별 six counters, `mixture_kl`, selector entropy/min/max, `mixture_version_lag`, buffer depth/age, lease retry, duplicate PackedSampleID, checkpoint replay divergence다. label에는 RunID, DataRevision, TokenizerRevision, PackingPolicyID, MixtureVersion, CurriculumVersion을 붙인다. source 이름을 고카디널리티 label로 직접 쓰기 어렵다면 stable ID와 외부 dimension table을 사용한다.

디버깅은 처음 갈라진 지점을 찾는 데서 시작한다. 원문 manifest가 같고 tokenizer shard checksum부터 다르면 데이터/토크나이저 문제다. segments까지 같고 packed rows가 다르면 buffer ordering과 tie-break다. IDs가 같고 labels/mask/position이 다르면 collator policy다. batch가 같고 loss가 다르면 model/dtype/RNG다. uninterrupted와 resumed run이 checkpoint 직후부터 갈라지면 buffer, cursor, RNG, selector version을 차례로 비교한다. 수천 step 뒤 validation 차이만 보는 것보다 훨씬 싸고 설명하기도 쉽다.

복구 승인에는 세 수준을 둔다. loadable은 파일이 읽힌다는 뜻이다. data-safe는 eligible token이 조용히 소실·중복되지 않는다는 뜻이다. trajectory-exact는 다음 batch, RNG, model/optimizer/selector update가 기준 run과 허용 오차 안에서 같다는 뜻이다. world-size 변경은 대개 세 번째를 보장하기 어렵다. 문서에는 실제로 검증한 수준만 표시한다.

마지막으로 7장에 넘기는 tensor 계약을 표로 고정한다. `input_ids[B,T]`에는 tokenizer와 packed-row checksum이, `labels[B,T]`에는 ignore reason과 valid count가, attention 표현에는 segment boundary와 backend convention이, `position_ids`에는 reset/continuous 정책이, varlen 경로에는 `cu_seqlens`와 max length가 붙는다. 11장의 optimizer에는 cumulative supervised token과 loss denominator를, 13장의 scheduler에는 step/token/FLOP progress를, 17장의 복구에는 buffer·cursor·RNG·version consistent cut을 넘긴다. 이 인계가 닫혀야 packing과 curriculum을 단순 전처리로 오해하지 않는다.

**구현을 읽는 순서는 데이터 흐름의 반대 방향이다.**

**loss에서 원문까지 역추적한다**

거대한 학습 저장소에서 packer 이름부터 검색하면 실제로 소비되는 경로가 아닌 utility와 오래된 구현을 읽기 쉽다. 더 확실한 출발점은 trainer가 호출하는 loss 함수다. loss가 받은 logits와 labels의 shape, ignore index, reduction을 확인하고 labels를 만든 collator 호출자로 올라간다. collator가 받은 sample schema를 찾고 dataset iterator, sampler, mixture builder, source reader까지 거슬러 올라간다. 이렇게 하면 configuration이 실제 객체 factory에 도달했는지, 중간 wrapper가 옵션을 덮어썼는지 확인할 수 있다.

각 함수 경계에는 질문이 하나씩 있다. loss는 어느 position을 분모에 넣는가. collator는 경계 target과 padding을 어떻게 구분하는가. packer는 segment lineage와 position을 보존하는가. sampler는 document와 token 가운데 무엇을 확률 단위로 삼는가. dataset worker는 retry에서 cursor를 언제 commit하는가. source reader는 압축 shard와 record offset을 어떻게 stable ID로 만드는가. 이 질문의 답을 한 호출 그래프에 적으면 설정 파일의 `packing`, `shuffle`, `weights` 같은 단어가 실제 상태 변경으로 번역된다.

PyTorch `DataLoader`를 쓴다면 main process sampler, worker process dataset iterator, prefetch queue의 소유권을 구분한다. map-style dataset의 index sampler와 iterable dataset의 worker sharding은 재개 방식이 다르다. persistent workers는 epoch 경계에도 내부 state가 남을 수 있고, `prefetch_factor`는 durable cursor보다 앞서 읽힌 sample 수를 늘린다. worker가 예외 뒤 재시작될 때 lease 없이 source cursor를 다시 읽으면 중복이 생긴다. framework가 iterator state를 저장한다고 가정하지 말고 `state_dict`가 실제로 무엇을 담는지 읽는다.

shuffle에는 적어도 shard permutation, record permutation, buffer shuffle, packed-row shuffle가 있다. seed 하나가 네 층을 모두 설명하지 못한다. epoch seed가 `base_seed+epoch`인지, global sample counter에서 파생되는지, rank와 worker ID가 섞이는지 확인한다. worker 수를 바꿨을 때 순서가 달라지는 것은 허용 가능한가를 계약으로 정한다. 재현성 보고서에는 seed 값뿐 아니라 RNG 알고리즘, key derivation, counter 위치와 shuffle buffer contents를 남긴다.

streaming dataset은 전체 길이를 모를 수 있다. scheduler가 `len(dataloader)`로 total step을 계산하거나 mixture allocation이 source length를 요구하면 추정치와 실제 capacity가 갈린다. compressed shard의 byte 크기를 token 수로 착각하지 않는다. source manifest에 record 수, tokenizer revision별 token 수, eligible/supervised 예상량과 산출 방법을 기록한다. 추정치를 사용했다면 error bar와 exhaustion 관측으로 보정한다.

데이터 병렬 rank마다 같은 packed batch가 복제되면 effective batch가 늘지 않는다. 반대로 각 rank의 sampler seed만 다르게 했는데 shard owner가 겹치면 부분 중복이 생긴다. global index modulo world size, distributed sampler padding, drop-last가 마지막 epoch에 어떤 sample을 반복·제거하는지 계산한다. `DistributedSampler`가 dataset length를 world size의 배수로 맞추려고 index를 추가할 수 있다는 일반 규칙을 실제 사용 경로와 대조한다. 무한 iterable에는 같은 가정이 적용되지 않는다.

**option 하나가 바꾸는 상태를 change sheet로 만든다.**

`max_seq_length`를 tensor shape만 바꾸는 옵션으로 보면 안 된다. 이 값이 달라지면 tokenizer truncation, document chunking, pack bin capacity와 position 범위가 함께 바뀐다. 그 결과 attention FLOP와 activation memory뿐 아니라 batch size, gradient accumulation, scheduler의 step당 token 같은 runtime state도 영향을 받는다. 예를 들어 2048을 8192로 늘리면서 global batch row 수를 유지하면 token batch는 네 배가 되고 attention 비용은 그보다 더 크게 증가한다. 동일한 token budget으로 비교하려면 batch와 accumulation을 조정하고 optimizer hyperparameter를 함께 scaling했는지 명시해야 한다.

`packing=True`는 어떤 packer class를 선택하는지, padding-free representation을 활성화하는지, position reset과 boundary mask를 만드는지, loss denominator를 바꾸는지 확인한다. 라이브러리 버전에 따라 의미가 달라질 수 있으므로 command line 설명을 그대로 책의 사실로 옮기지 않는다. factory branch, constructed object, golden batch dump 세 근거를 연결한다.

`drop_last=True`는 shape 안정성을 주지만 epoch마다 마지막 samples를 버린다. source별로 섞은 뒤 drop하면 작은 source가 반복적으로 tail에 놓이는지 확인한다. distributed sampler와 dataloader가 각각 drop-last를 적용하면 두 번 잘릴 수 있다. curriculum phase가 짧을 때 마지막 batch 손실 비중은 커진다. `dropped_by_reason` counter 없이는 filter reject와 batch tail drop을 구분할 수 없다.

`shuffle_buffer_size`를 키우면 local mixing은 좋아지지만 memory, startup latency, checkpoint payload, resume exactness 비용이 증가한다. 작은 buffer는 source ordering artifact를 남긴다. buffer 알고리즘이 replacement인지 reservoir인지 sliding window인지에 따라 sample probability가 다르다. 동일 seed의 output prefix, position별 source autocorrelation, resume next IDs로 검증한다.

`num_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory`는 model 수학을 바꾸지 않는 성능 옵션처럼 보인다. 그러나 stateful iterable과 nondeterministic arrival-order packer에서는 output 순서를 바꿀 수 있다. correctness gate에서는 worker 0/1/N의 global ledger를 비교하고, 성능 gate에서는 pinned transfer와 CPU tokenizer/packing 시간을 측정한다. 성능 설정을 먼저 켜고 divergence가 생기면 원인 범위가 지나치게 넓어진다.

mixture의 `temperature`는 원 weight에 거듭제곱을 적용해 평탄화하거나 첨예화할 수 있다. 구현마다 지수와 역온도 convention이 반대일 수 있다. 예를 들어 `p_i∝q_i^(1/T)`이면 T가 커질수록 평탄해진다. 작은 세 source fixture로 실제 확률을 출력하고 합, monotonicity, zero 처리, minimum floor를 검사한다. option 이름만 보고 방향을 서술하지 않는다.

curriculum의 `warmup_steps`도 optimizer warmup과 혼동하기 쉽다. mixture controller warmup인지, sequence length transition인지, selector EMA bootstrap인지 owner를 적는다. counter unit이 optimizer update인지 microbatch인지 token인지 validation event인지 확인한다. gradient accumulation을 바꿨을 때 같은 schedule을 뜻하는지 계산한다.

**데이터 선택의 효과를 인과 주장으로 과장하지 않는다.**

**인과 주장을 ablation으로 제한한다**

packing 실험에서 wall-clock budget을 고정하면 빠른 경로가 더 많은 token을 보고, token budget을 고정하면 시간 차이를 비교할 수 있다. 둘은 서로 다른 질문이다. quality-per-token, quality-per-FLOP, quality-per-hour를 분리한다. mixture나 curriculum은 평균 sequence length와 kernel efficiency도 바꾸므로 nominal token만 같아도 FLOP와 wall time이 다를 수 있다. 적어도 processed token, supervised token, estimated training FLOP, measured accelerator time을 함께 보고한다.

selector가 validation score를 사용하면 같은 validation set으로 최종 성능을 평가할 때 adaptive overfitting이 생길 수 있다. controller observation set, model selection set, 최종 holdout을 분리하거나 반복 관측 횟수를 고려한다. domain weight를 여러 번 튜닝한 뒤 가장 좋은 run만 보고하면 탐색 비용과 실패 run을 숨긴다. 모든 candidate mixture, seed, 중단 이유, missing metric을 run ledger에 남긴다.

proxy model에서 찾은 mixture가 큰 model에도 유효하다는 것은 가정이다. scaling에 따라 domain의 reducibility, capacity, interference가 달라질 수 있다. 여러 proxy 크기에서 weight 안정성과 rank correlation을 보고, large run에서는 최소한 baseline과 선택 mixture를 복수 seed로 비교한다. 하나의 checkpoint 개선을 보편적 법칙으로 쓰지 않는다.

curriculum은 같은 최종 데이터 양이라도 순서를 바꾼다. 쉬운 것부터 어려운 것으로 가는 직관은 loss landscape, optimizer state, catastrophic forgetting 때문에 항상 성립하지 않는다. reverse curriculum, shuffled schedule, static mixture를 포함한 ablation이 필요하다. phase 전환 직후 loss spike가 있으면 데이터 난이도 변화, length로 인한 token batch 변화, scheduler counter 변화, kernel backend 변경을 분해한다.

data quality score와 model loss는 종종 길이, 언어, formatting 같은 nuisance variable에 의존한다. selector가 실제 품질 대신 짧은 문서나 익숙한 tokenizer pattern을 선호할 수 있다. score를 length/language/source별로 stratify하고 selection odds를 본다. 높은 score sample의 원문 사례를 blind review하되 사례 몇 개를 전체 통계의 대체물로 쓰지 않는다.

importance weighting은 기대 objective를 맞추더라도 finite batch variance를 키운다. effective sample size `ESS=(Σw)^2/Σw²`를 window마다 계산하고 max weight, clipped mass, source별 gradient norm을 본다. ESS가 급락하면 unbiased라는 형식적 장점이 실제 최적화 불안정으로 이어질 수 있다. clipping threshold를 바꾼 sweep에서 bias와 variance의 교환을 보여 준다.

중복 데이터는 source mixture와 별개로 exposure를 왜곡한다. 같은 문서가 여러 source에 들어가면 source counter는 정상이어도 semantic content가 반복된다. exact hash, normalized hash, near-duplicate cluster ID를 segment lineage에 연결한다. train/validation contamination cluster가 selector score를 부풀릴 수 있으므로 평가 장의 contamination ledger와 이어 준다.

**독자가 직접 판정할 수 있는 종합 사례.**

세 source를 가정한다. A는 짧은 고품질 교과서, B는 긴 웹 문서, C는 instruction 대화다. document probability는 0.4, 0.4, 0.2이고 acceptance는 1.0, 0.5, 0.9다. 평균 raw length는 256, 2048, 768이며 C의 prompt 절반은 loss에서 제외된다. 독자는 먼저 기대 supervised 질량을 계산한다. 그다음 B가 `T=1024`에서 tail drop되는 경우와 chunk+overlap되는 경우를 나눈다. A가 packing 빈칸을 잘 채운다는 이유로 best-fit lookahead에서 먼저 선택되는 arrival-order bias도 추가한다.

이 사례에서 configured 40/40/20은 어떤 stage에서도 그대로 유지될 이유가 없다. B는 reject와 truncation으로 줄고 C는 prompt mask로 objective 질량이 줄며 A는 packer가 선호할 수 있다. 반대로 microbatch mean을 같은 비중으로 평균하면 짧은 A row가 과대표현될 수 있다. 한 숫자의 mixture가 아니라 waterfall과 loss coefficient로 답해야 한다.

이제 curriculum이 B의 validation excess loss를 보고 B probability를 올렸다고 하자. prefetch queue에 old mixture batch가 세 update 남아 있으면 actuation delay가 생긴다. B probability floor/ceiling, importance weight, ESS, buffer source composition을 계산한다. 바로 뒤 crash에서 model은 update u, selector는 u+1, sampler queue는 u-2 상태라면 어떤 consistent cut으로 돌아갈지도 정한다.

마지막으로 world size를 8에서 16으로 바꾼다. rank-local packer라면 같은 source counters를 맞춰도 segment 조합과 position이 달라질 수 있다. loadable, data-safe, topology-portable, trajectory-exact 가운데 실제 보장 수준을 고른다. 답에는 필요한 durable state, 비교할 ID와 checksum, 허용할 차이와 금지할 차이가 포함돼야 한다.

이 종합 사례를 통과했다는 뜻은 용어를 외웠다는 뜻이 아니다. option에서 함수 branch로, 함수에서 tensor와 state로, state에서 metric과 failure injection으로, 관측에서 복구 판정으로 연결할 수 있다는 뜻이다. 이 연결이 6장의 실질적 완료 조건이다.

## 6.3 mixture와 curriculum은 소비 질량을 제어한다

### 6.3.1 길이·난이도·영역 curriculum을 결합한다

길이 curriculum이 `T=1024→2048→4096`로 늘어난다고 하자. row batch B를 고정하면 step당 physical token이 길이에 비례해 늘고 self-attention의 score 연산은 대략 T에 더 민감하게 증가한다. activation memory와 step time도 달라진다. optimizer update 수만 맞춘 비교에서는 후반 phase가 더 많은 token과 FLOP를 소비한다. 반대로 global token batch를 고정하려 B를 줄이면 microbatch 수, gradient noise, GEMM shape와 pipeline bubble이 바뀐다.

따라서 schedule은 phase마다 sequence length, microbatch rows, accumulation, global physical token, expected supervised token, attention backend, activation checkpoint 정책을 함께 가진다. transition 전에 다음 shape가 divisibility와 memory budget을 만족하는지 preflight한다. transition 직후 OOM이 나면 단순히 batch를 줄이기 전에 optimizer/scheduler의 token progress와 gradient scaling도 다시 계산한다.

variable sequence length를 batch마다 뽑는 방식은 discrete phase와 다르다. length distribution `p(T)`가 있고 각 T에서 sample/document 절단과 packing efficiency가 다르다. 긴 sample이 긴 T batch에만 들어가면 data curriculum도 동시에 생긴다. length별 source share, unique coverage, valid fraction과 loss를 기록한다. batch T만 histogram으로 보면 어떤 content가 어느 길이에 배정됐는지 모른다.

RoPE와 learned position이 length transition에서 같은 방식으로 반응하지 않는다. learned table은 아직 충분히 update되지 않은 높은 position row를 갑자기 사용하고, RoPE는 parameter가 없어도 train distribution 밖 frequency 조합을 사용한다. length phase가 바뀌면 7장의 position range와 8장의 attention score/entropy 관측을 연결한다. 품질 spike를 data difficulty만으로 설명하지 않는다.

### 6.3.2 stale score와 관측 지연을 관리한다

example loss는 model이 학습하면서 변한다. 초기 checkpoint에서 계산한 loss ranking을 끝까지 쓰면 이미 학습된 example과 아직 어려운 example을 구분하지 못한다. score refresh 주기, scorer checkpoint, tokenizer/data revision을 `ScoreRevision`으로 고정한다. refresh 비용을 줄이려고 부분 sample이나 proxy model을 쓰면 coverage와 correlation을 측정한다.

loss가 높은 example을 어려운 것으로 보는 방법은 label noise, 깨진 encoding, 희귀 언어, 긴 sequence와 혼동된다. irreducible loss, disagreement, influence, gradient norm 같은 대안을 사용해도 완벽하지 않다. score를 여러 feature로 분해하고 high-score tail을 source/language/length/quality flag별로 검토한다. selector가 오류 데이터를 집중 학습하는 positive feedback을 막기 위한 quarantine와 ceiling을 둔다.

competence-based curriculum은 현재 progress에 따라 허용 데이터 범위를 넓힌다. progress를 step, token, validation metric 가운데 무엇으로 정의하는지 중요하다. token 기반이 topology와 accumulation 변화에 더 안정적일 수 있지만 supervised fraction이 바뀌면 physical token과 objective progress가 다르다. `cumulative_supervised_token`을 durable counter로 두고 overflow나 double-count를 검사한다.

self-paced learning에서 현재 model loss로 sample weight를 정하면 weight도 autograd graph에 포함할지 stop-gradient할지 결정해야 한다. 대부분 selection/controller는 model parameter gradient와 분리하지만 구현을 확인한다. score computation이 같은 forward activation을 재사용하는지 별도 proxy인지에 따라 memory와 stale delay가 달라진다. selected subset에 대한 loss denominator도 다시 계산해야 한다.

### 6.3.3 domain curriculum을 governance와 연결한다

domain 정의는 고정된 자연 법칙이 아니라 mapping artifact다. 하나의 문서가 code, math, multilingual처럼 여러 domain에 걸칠 수 있다. hard assignment인지 soft membership인지, classifier revision과 confidence, unknown bucket을 기록한다. mapping이 바뀌면 같은 domain weight vector도 다른 문서 집합을 뜻한다. selector checkpoint에는 weight뿐 아니라 DomainMapRevision이 필요하다.

source 이름과 domain을 혼동하지 않는다. source는 저장·라이선스·수집 lineage 단위이고 domain은 학습 목적의 의미 분류일 수 있다. 여러 source가 한 domain에 속하고 한 source가 여러 domain으로 갈릴 수 있다. metric은 source stage counters와 domain objective counters를 교차 집계한다. 문제가 생겼을 때 governance owner는 source를, curriculum controller는 domain을 찾아야 한다.

데이터 삭제나 라이선스 변경은 curriculum state에도 영향을 준다. 제거된 문서가 selector score table, packing buffer, materialized selected corpus, checkpoint exposure ledger에 남아 있을 수 있다. DataRevision을 올리고 stale score와 pack artifact를 무효화한다. 이미 소비한 ContributionID 범위는 provenance를 위해 남기되 재학습/언러닝 정책과 연결한다.

domain upweight가 개인정보나 unsafe content exposure를 늘릴 수 있다. quality/skill 목표와 safety filter가 충돌할 때 어느 gate가 우선하는지 정책으로 정한다. selector가 filter 이전 raw pool을 보아 score를 만들고 filter 이후 sampler가 뽑으면 기대 weight와 실제 mixture가 다르다. accepted pool에서 다시 normalize하고 rejected mass를 기록한다.

**production 장애를 데이터 상태에서 진단한다.**

**처리량 저하를 GPU 문제로 오인하지 않는다.**

GPU utilization이 주기적으로 떨어지면 dataloader queue depth와 batch ready latency를 먼저 timeline에 겹친다. source shard read, decompression, tokenization, filter, pack search, host allocation, H2D transfer 시간을 stage별로 잰다. best-fit lookahead가 buffer를 스캔하거나 긴 document tokenization이 head-of-line blocking을 만들 수 있다. 평균 batch build time보다 p95/p99와 source/length 조건이 중요하다.

worker를 늘려 처리량이 나아지지 않으면 shared filesystem metadata, decompression CPU, Python GIL, memory bandwidth, pinned-memory 한도를 분해한다. arrival-order packer에서 worker를 늘리면 output order도 바뀔 수 있으므로 performance 실험의 data identity를 확인한다. 동일한 materialized packed batches를 replay한 model-only baseline과 online pipeline을 비교하면 input 병목을 분리할 수 있다.

packing utilization이 갑자기 떨어지면 length distribution 변화, buffer size, curriculum T, source mixture, max document chunk, packer fallback을 본다. tokenizer revision 변경으로 같은 문서 token length가 늘었을 수도 있다. utilization과 raw length quantile, buffer occupancy, no-fit reason을 연결한다. 단순히 buffer를 키우면 memory와 resume payload가 커지므로 원인 없이 조정하지 않는다.

**loss 이상을 데이터 contribution으로 역추적한다.**

loss spike update의 PackedSampleID를 찾아 segment map으로 DocumentID와 offset, source, score, mixture version을 복원한다. position별 loss 상위 contribution을 표시하되 raw content 접근 권한과 개인정보를 지킨다. boundary target, encoding corruption, 반복 boilerplate, label mask 누락, stale score를 분류한다. 한 example이 원인이어도 왜 filter와 test를 통과했는지 upstream gate를 고친다.

source별 mean loss는 길이와 mask 차이 때문에 오해를 낳는다. token loss distribution, valid count, position bucket, language/domain을 stratify한다. mixture가 바뀐 직후 global loss가 오르는 것은 더 어려운 source를 많이 본 결과일 수 있다. 같은 source/length slice의 loss와 validation을 함께 본다. controller 목표가 global training loss 최소인지 target validation 개선인지 구분한다.

NaN batch는 token ID range, attention mask row가 전부 차단됐는지, valid count가 0인지, extreme length/position을 먼저 본다. all-ignore microbatch에서 `loss_sum/0`이 NaN이 될 수 있다. distributed global valid count가 0이면 optimizer step을 skip하고 scheduler/selector/cursor를 어떻게 처리할지 정한다. rank 일부만 skip하면 collective 순서가 갈린다.

sample repeat 경보는 exact duplicate와 intentional overlap을 구분한다. overlap segment는 같은 raw token을 반복하지만 다른 context와 label mask를 가질 수 있다. occurrence ID와 reason을 둔다. checkpoint rollback, worker retry, source-level duplication, small source curriculum repetition을 분리한다. Bloom filter는 memory 효율적이지만 false positive가 있으므로 exact recent ring과 provenance 조회로 확인한다.

**checkpoint 장애를 consistent-cut 검증으로 닫는다.**

manifest publish 전에 model shard는 모두 저장됐지만 sampler state가 실패할 수 있다. 완전한 CheckpointID는 모든 required component checksum과 parent ID가 등록된 뒤 원자적으로 publish한다. incomplete directory를 latest로 선택하지 않는다. object storage에서는 rename이 원자적이지 않을 수 있어 immutable objects와 small commit marker를 사용한다.

restore는 schema version, DataRevision, TokenizerRevision, PackingPolicyID, Mixture/Curriculum controller class를 검증한다. 호환되지 않는 변경을 permissive default로 채우지 않는다. migration이 가능하면 old→new state 변환과 golden next-batch test를 제공한다. packing buffer format을 바꿨다면 buffer를 버리고 cursor만 복구하는 것은 data-safe인지 계산한다.

crash 직전 prefetch queue에 있던 batch가 optimizer에 commit됐는지는 UpdateID ledger로 판정한다. consumed token counter는 forward 시작이 아니라 optimizer commit과 연결한다. overflow로 optimizer step이 skip됐다면 data를 소비한 것으로 셀지 replay할지 정책이 필요하다. 두 선택 모두 학습 trajectory와 exposure가 다르다. scaler overflow, scheduler step, selector update, cursor commit 순서를 명시한다.

resume 검증은 첫 batch 하나만으로 부족하다. buffer와 RNG 오류는 buffer가 비거나 refill될 때 나타날 수 있다. 최소한 buffer capacity와 prefetch depth를 넘는 K개의 PackedSampleID, segment map, counter, mixture version을 비교한다. model forward 없이 데이터 pipeline만 dry inspection할 수 있지만, 이 목표에서는 실제 runtime을 실행하지 않으므로 필요한 fixture와 expected invariant를 문서화하고 static test source를 감사한다.

**출판 가능한 주장과 미확인 경계를 구분한다.**

**논문 알고리즘, 공개 코드, production recipe를 세 층으로 쓴다.**

논문은 목적함수와 실험 조건을 제공하지만 production의 모든 checkpoint·retry·분산 세부를 공개하지 않을 수 있다. 공개 저장소는 특정 revision의 구현을 보여 주지만 논문 결과에 사용된 내부 code와 다를 수 있다. model card나 blog는 큰 run의 recipe를 요약하지만 loss denominator와 state machine을 생략할 수 있다. 세 출처가 합의하는 부분만 강한 사실로 쓰고 차이는 명시한다.

DoReMi, DoGE, Skill-It, RHO-Loss, DSIR, RegMix를 하나의 우열 표로 만들지 않는다. 선택 단위, online/offline, 필요한 reference/proxy, score, objective correction, compute overhead, mutable state, 공개 test 범위가 다르다. 독자는 자기 병목이 corpus selection인지 online mixture인지 example filtering인지 먼저 판정해야 한다. 방법 이름보다 문제와 계약을 연결한다.

공식 구현의 함수·line 좌표를 인용할 때 commit을 고정한다. main branch line number는 변한다. source note에는 repository, commit, path, symbol, line range, 왜 중요한지를 적는다. 본문에서는 필요한 짧은 코드 조각만 설명하고 전체 원문을 복제하지 않는다. test가 존재하면 무엇을 assertion하는지 읽고, 없는 state를 검증했다고 과장하지 않는다.

실행하지 않은 대규모 distributed 조합은 `NotExecuted`다. static code inspection으로 collective group과 state field를 찾을 수 있지만 topology별 성능, failure recovery 성공, numerical parity를 증명할 수 없다. 예상 결과와 관측 결과를 다른 열에 둔다. 독자가 실제 환경에서 실행할 command, fixture, metric, pass/fail criterion을 재현 가능한 형태로 제공한다.

**6장 semantic gate.**

독자는 다음 질문에 artifact로 답해야 한다. 원문 token 하나가 어느 packed row와 loss contribution에 들어갔는가. 독립 문서 objective가 attention과 label 양쪽에서 보존되는가. 설정 source weight가 실제 supervised gradient 질량으로 어떻게 변했는가. selector가 어느 관측으로 어느 version의 next draw를 바꾸었는가. crash 뒤 buffer, cursor, RNG, controller와 model이 같은 commit으로 복구되는가.

정답은 설명 문장만이 아니다. segment map, six-stage counter, contribution ledger, controller version timeline, checkpoint manifest, golden next-ID 비교가 있어야 한다. 이 가운데 하나라도 없으면 빠른 training run을 만들 수는 있어도 데이터가 무엇을 학습시켰는지 증명할 수 없다.

6장의 최종 회귀 묶음은 boundary logits/gradient parity, unique coverage conservation, mixture waterfall reconciliation, global valid-token loss reduction, controller probability invariant, stale-version delay, crash-point replay, world-size contract test다. 각 결과에는 source/data/tokenizer/packing/controller revision과 실행 여부가 붙는다. 실패는 first divergent artifact와 owner에게 연결한다.

이 gate가 닫히면 7장은 입력 ID와 position을 신뢰하고 embedding·norm을 해부할 수 있다. 13장은 step 대신 실제 token/FLOP progress를 받을 수 있고, 17장은 data state를 model state와 함께 복구할 수 있다. 24장은 어떤 평가 sample이 training contribution과 겹쳤는지 exposure ledger로 확인할 수 있다. packing·mixture·curriculum은 이렇게 책 전체의 데이터 혈관이 된다.

**최소 재현 실험 명세.**

재현 패키지는 임의 text 대신 고정 token ID fixture를 사용한다. source A, B, C에 서로 겹치지 않는 ID 범위를 주고 document length, EOS 유무, reject flag, supervised mask를 manifest에 적는다. tokenizer 변수를 제거한 첫 실험 뒤 실제 tokenizer fixture를 추가한다. expected segment, packed row, label, position, cumulative length를 사람이 읽을 수 있는 표와 machine-readable checksum으로 함께 둔다.

packer test는 순수 함수 층과 stateful iterator 층을 분리한다. 순수 함수는 주어진 ordered segments와 policy에서 같은 bins를 내야 한다. stateful test는 buffer, cursor, RNG를 serialize한 뒤 다음 output prefix를 비교한다. worker와 rank test는 virtual global index를 physical owner에 매핑하고 global ledger를 합쳐 중복과 누락을 검사한다. 한 test가 모든 층을 동시에 다루면 실패 위치가 모호하다.

mixture fixture는 document 수가 아니라 길이와 mask가 다른 source를 쓴다. 설정 1/3씩에서 document draw가 비슷해도 supervised token share가 달라지는 expected 값을 손계산한다. filter, truncation, overlap, boundary EOS를 차례로 켜 waterfall counter가 정확히 변하는지 본다. global loss reduction은 rank와 microbatch별 valid count를 불균등하게 해 단순 mean-of-means 오류를 노출한다.

controller fixture는 model 대신 결정적인 synthetic score sequence를 공급한다. weight normalization, floor, ceiling, entropy, version increment, publish/read timing을 확인한다. importance correction이 있다면 unclipped expectation과 clipped bias를 작은 유한 sample space에서 전수 계산한다. queue delay test는 old/new version batch를 의도적으로 섞고 consumed version histogram이 실제 지연을 반영하는지 본다.

crash matrix는 component와 시점을 행렬로 만든다. reader cursor 전후, buffer insert 전후, row publish 전후, optimizer commit 전후, selector publish 전후에 중단한다. 복구 뒤 next IDs, coverage, counter, model UpdateID, MixtureVersion을 비교한다. 요구 계약이 data-safe라면 순서 차이는 허용하되 중복·누락 한도를 정의하고, trajectory-exact라면 packed grouping과 RNG까지 같아야 한다.

성능 fixture는 correctness artifact를 그대로 replay한다. padding, packed dense, varlen의 physical·supervised tokens/sec, peak memory, collator CPU, transfer, kernel 시간을 같은 model/dtype/shape 조건에서 센다. online packer의 CPU 병목과 GPU kernel 개선을 한 숫자로 합치지 않는다. warm-up, measurement window, profiler overhead를 기록한다.

실험 결과 표에는 `Passed`, `Failed`, `NotExecuted`, `NotApplicable`만 사용한다. 예상상 통과할 것이라는 문장을 Passed로 바꾸지 않는다. failure에는 first divergent ID/tensor/counter, 관련 source symbol, 재현 fixture, owner와 다음 검증을 붙인다. 이 규칙이 있어야 책의 독자가 설명과 증거를 구분할 수 있다.

**90분 안에 최초 차이를 찾는 디깅 순서.**

## 6.4 분산 sampler를 결정적 상태 기계로 설계한다

### 6.4.1 worker·rank·epoch의 순서를 분리한다

분산 학습에서 “데이터를 섞었다”는 말은 충분하지 않다. 전역 sample 순서, rank별 shard 순서, worker별 prefetch 순서가 따로 있다. map-style dataset에서 distributed sampler는 보통 epoch와 seed로 전역 index permutation을 만든 뒤 rank가 일정 간격 또는 연속 구간을 가져간다. DataLoader worker는 rank-local index를 다시 queue로 받아 `__getitem__`과 collate를 실행한다. iterable dataset은 dataset 자체가 rank와 worker 정보를 읽어 stream을 나눌 수 있다. 두 방식을 섞으면 sampler와 dataset이 모두 sharding하여 일부 sample이 사라질 수 있다.

`set_epoch(e)` 호출은 장식이 아니다. sampler가 epoch를 seed derivation에 넣는 구현에서는 이를 빼면 매 epoch 같은 permutation이 반복된다. 반대로 exact resume를 원하면 epoch뿐 아니라 그 permutation 안의 cursor와 이미 prefetched된 index를 알아야 한다. optimizer step만 저장하고 epoch 처음부터 iterator를 다시 만든 뒤 이미 소비한 batch 수만큼 건너뛰는 방식은 비용이 크고, stochastic transform과 worker RNG가 같은 경로를 재현하지 못할 수 있다.

worker seed는 base seed, epoch, rank, worker ID를 충돌 없이 결합해야 한다. Python, NumPy, framework RNG를 각각 초기화한다. sample 단위 증강을 worker 실행 순서와 독립적으로 만들려면 `SampleID`와 transform revision에서 counter-based seed를 파생할 수 있다. 그렇지 않으면 prefetch scheduling이 달라졌다는 이유만으로 같은 sample의 crop이나 masking이 바뀐다. augmentation diversity와 재현 요구를 분리해 지원 수준을 적는다.

prefetch queue에 들어간 item은 아직 optimizer update에 기여하지 않았지만 RNG와 I/O cursor는 이미 소비했다. checkpoint가 queue를 저장하지 않으면 resume 뒤 그 item을 다시 만들거나 건너뛰게 된다. 이를 구분하려면 `issued`, `materialized`, `delivered`, `committed` 상태를 따로 기록해야 한다. batch가 training loop에 전달된 뒤 overflow로 optimizer step이 skip된 경우에는 sample을 committed로 셀지 명시적인 정책이 필요하다. 특히 data curriculum이 성공한 update를 feedback으로 사용한다면, skip된 step이 selector 상태까지 갱신하지 않도록 해야 할 수 있다.

마지막 rank를 맞추기 위한 padding index도 질량 회계에 들어간다. dataset 크기가 world size와 batch 크기로 나누어지지 않을 때 sampler는 sample을 반복하거나 drop할 수 있다. `drop_last=True`가 DataLoader batch에서 적용되는지 distributed sampler에서 적용되는지에 따라 버리는 수가 다르다. 반복 sample에 loss weight 1을 주면 configured mixture와 realized unique mixture가 달라진다. 각 epoch의 unique SampleID, duplicate count, dropped count를 보고한다.

elastic world-size 변경은 더 어렵다. rank 수가 달라지면 단순 stride partition의 다음 sample owner가 바뀐다. global consumption ledger와 deterministic global permutation이 있으면 아직 committed되지 않은 suffix를 새 rank에 나눌 수 있다. rank-local cursor만 저장한 checkpoint는 topology-portable하지 않다. data state를 model shard와 같은 rank 파일에만 묶으면 missing rank 복구도 어렵다. global manifest가 rank state와 commit step을 consistent cut으로 가리키게 한다.

### 6.4.2 token budget batcher의 합을 제어한다

가변 길이 학습에서 examples-per-batch는 compute와 objective 질량을 안정적으로 나타내지 못한다. token budget batcher는 sequence 길이 합 또는 padded token 수가 한도를 넘기기 전까지 sample을 묶는다. 길이 정렬은 padding을 줄이지만 비슷한 길이와 domain 또는 난이도가 상관되어 있으면 batch 분포를 바꿀 수 있다. shuffle window 안에서만 bucket하는 이유와 window 크기 trade-off를 기록한다.

packed token 수, padding 포함 token 수, loss-valid token 수는 다르다. attention compute는 padding과 block layout의 영향을 받고 LM loss는 mask가 유효한 label만 센다. gradient accumulation target이 “전역 2M tokens”라면 어느 token 정의인지 고정한다. assistant-only SFT에서 input token이 attention compute에는 들어가지만 loss denominator에는 빠진다. compute budget과 optimization mass를 두 열로 둔다.

rank마다 valid token 수가 다르면 rank-local mean loss를 DDP 평균하는 방식은 전역 token mean과 다르다. 각 rank의 loss sum과 valid count를 all-reduce하고 전역 denominator에 맞게 gradient scale을 정해야 한다. framework가 gradient를 rank 평균하는 factor까지 포함해 유도한다. 마지막 uneven batch나 variable packing에서만 나타나는 오류라 정규 step의 loss curve로 찾기 어렵다.

**mixture와 curriculum을 재현 가능한 제어계로 만든다.**

### 6.4.3 selector update와 실현 질량을 기록한다

adaptive mixture는 최근 loss나 gradient utility를 관측해 다음 sampling 확률을 바꾼다. 이때 metric이 어느 model step의 어느 checkpoint에서 계산됐고 selector가 언제 반영했는지 적어야 한다. 비동기 평가가 늦게 도착하면 현재 model보다 오래된 score가 mixture를 움직인다. observation step, arrival step, apply step을 분리하고 최대 staleness를 정책으로 둔다.

확률 update가 너무 빠르면 batch noise를 따라 domain 비율이 진동한다. exponential smoothing, minimum mass, trust region, entropy regularization, update interval이 안정화 장치다. 각 장치는 단순 option이 아니라 selector state를 추가한다. smoothing을 쓰면 이전 estimate를 checkpoint해야 하고, minimum mass는 희귀 domain이 완전히 사라지지 않게 한다. trust region은 한 update의 분포 이동을 제한한다. 이 값을 바꾼 effect를 realized token share와 validation utility로 본다.

domain score의 scale도 맞춰야 한다. token mean loss가 vocabulary와 난이도가 다른 domain 사이에서 직접 비교 가능한지 검토한다. raw loss, reference model 대비 excess loss, 학습 진전, gradient alignment는 서로 다른 signal이다. selector 논문의 목적함수와 production 구현이 같은 signal을 쓰는지 source에서 확인한다. 이름만 같은 DoReMi 또는 DoGE recipe라도 proxy model, group definition, smoothing과 temperature가 다르면 다른 제어계다.

curriculum은 sample property와 schedule을 잇는 함수다. 길이 curriculum은 max sequence length뿐 아니라 batch size, accumulation, attention kernel, position distribution을 함께 바꾼다. 난이도 curriculum은 score 생성 model과 revision, score 시점, tie policy를 가진다. domain curriculum은 governance constraint와 혼합될 수 있다. 여러 축이 동시에 변하면 효과를 분리할 수 없으므로 단계적 intervention과 교차 실험을 설계한다.

selector checkpoint에는 group vocabulary와 stable group ID, probability vector, score estimate, optimizer 또는 dual state, observation watermark, RNG, consumed mass가 들어간다. dataset revision에서 group이 추가·삭제되면 load migration 규칙이 필요하다. index 위치만 저장하면 domain ordering 변화로 확률이 다른 group에 붙는다. group ID와 checksum으로 정렬하고 unknown group은 fail-fast 또는 명시 initialization을 거친다.

**네 가지 질량 보존식으로 운영한다**

입력 질량은 raw bytes와 documents다. tokenization 뒤 token 질량으로 바뀌며 filtering과 dedup에서 삭제된 양을 ledger에 남긴다. sampling 질량은 configured probability와 draw count다. packing 질량은 emitted token, padding, boundary token, truncated token이다. optimization 질량은 loss-valid token과 gradient contribution이다. 네 단계 합계가 서로 다른 것은 정상이나 차이를 설명하지 못하면 장애다.

domain `d`에 대해 raw available token `A_d`, sampled token `S_d`, packed token `P_d`, valid target `N_d`, weighted loss contribution `L_d`를 기록한다. `S_d/A_d`는 oversampling 정도, `P_d/S_d`는 truncation과 packing 효과, `N_d/P_d`는 mask 효과, `L_d/N_d`는 평균 loss scale을 보여준다. 최종 aggregate loss만 보면 어떤 domain이 사라졌는지 알 수 없다.

mixture dashboard에는 configured share와 realized share를 나란히 둔다. realized share는 batch, token, valid target, weighted loss 네 denominator로 보여준다. KL divergence나 최대 편차는 경보 후보지만 작은 domain의 absolute count도 함께 본다. rare safety data가 0이 된 상황은 전체 KL이 작을 수 있다. window 크기와 alert persistence를 정해 일시 변동과 지속 drift를 구분한다.

## 6.5 packer를 tensor와 커널 경계까지 구현한다

### 6.5.1 position·attention·label 경계를 일치시킨다

여러 문서를 한 sequence에 이어 붙일 때 최소 세 경계 표현이 필요하다. attention mask 또는 segment ID는 다른 문서를 보지 못하게 한다. position ID는 문서마다 0으로 reset할지 전체 pack에서 증가할지 모델 계약에 맞춘다. labels mask는 boundary를 가로질러 다음 문서 첫 token을 예측하지 않게 한다. 세 tensor 가운데 하나만 빠져도 silent leakage 또는 쓸모없는 target이 생긴다.

예를 들어 문서 A token `[a0,a1,eos]`와 B `[b0,b1,eos]`를 pack한다. 일반 next-token shift는 `eos -> b0` target을 만든다. 문서 독립 objective라면 이 위치 label을 ignore해야 한다. EOS 뒤 다음 문서를 볼 수 없도록 block diagonal causal mask를 만든다. position reset 여부는 RoPE와 학습 recipe에 따라 선택하되 train과 evaluation에서 일관되어야 한다. scalar fixture는 A token을 바꾸었을 때 B logits가 변하지 않는지 확인한다.

varlen FlashAttention 계열 interface는 packed QKV와 cumulative sequence lengths를 받을 수 있다. 이때 `cu_seqlens`가 attention 경계를 표현하고 dense block mask를 물질화하지 않는다. 값은 0에서 시작하는 prefix sum이며 마지막 값이 total tokens와 같아야 한다. sequence 길이 0, 마지막 partial pack, dtype과 device를 검증한다. attention 경계가 맞아도 labels boundary는 collator가 별도로 책임진다.

FSDP 또는 sequence parallel 전에 packing tensor가 어떻게 shard되는지 본다. token sequence를 rank에 나누면 segment boundary가 rank 경계를 지날 수 있다. local rank가 이전 token의 segment ID를 모르면 첫 query mask나 label을 잘못 만들 수 있다. global metadata 또는 halo 정보를 전달한다. packed batch를 rank별 독립 생성하는 data parallel과 한 sequence를 rank across split하는 context parallel을 구분한다.

best-fit decreasing 같은 offline bin packing은 효율이 높지만 전체 length 목록을 알아야 하고 ordering을 바꾼다. streaming packer는 bounded buffer에서 선택하므로 memory와 randomness trade-off가 있다. deterministic tie-breaker를 stable SampleID로 정한다. 동일 길이 sample이 hash-map iteration 순서에 따라 달라지면 resume와 audit가 깨진다.

packing 효율은 `valid_or_used_tokens / allocated_tokens`로 정의하되 분자를 명시한다. attention compute 관점에서는 non-padding tokens, optimization 관점에서는 valid labels가 중요하다. 높은 packing efficiency가 더 많은 cross-document boundary와 짧은 segment를 만들어 kernel utilization 또는 learning dynamics를 바꿀 수 있다. 처리량, 유효 target throughput, validation을 함께 본다.

최종 test bundle은 unpacked 독립 실행과 packed 실행의 per-token logits·loss·gradient parity를 비교한다. dropout을 끄고 동일 position policy를 사용한다. model이 segment-aware mask를 지원하지 않으면 packing을 금지하거나 cross-document attention을 recipe의 명시된 선택으로 둔다. “다들 packing을 쓴다”는 이유로 목적함수 변경을 숨기지 않는다.

이 보강 절의 결론은 간단하다. DataLoader, sampler, packer, mixture selector는 model 바깥의 보조 코드가 아니다. 다음 gradient에 어떤 token이 얼마의 질량으로 들어갈지 결정하는 목적함수의 일부다. 따라서 model과 같은 수준으로 revision, state, checkpoint, negative test와 관측성을 가져야 한다.

**packing 알고리즘을 자료구조와 사건으로 구현한다.**

### 6.5.2 greedy streaming packer의 열린 상태

streaming packer는 아직 닫히지 않은 bin 목록, 각 bin의 남은 capacity, 들어간 SampleID와 token span, input iterator cursor와 tie-break RNG를 가진다. 다음 example 길이 `l`이 들어오면 적합한 bin을 고르거나 새 bin을 연다. exact fit, overflow와 oversized sample 정책을 둔다.

first-fit은 열린 bin을 순서대로 보고 처음 맞는 곳을 고른다. best-fit은 넣은 뒤 남는 공간이 가장 작은 bin을 고른다. bounded buffer에서 length sorting을 할 수 있다. 알고리즘 선택은 padding/unused capacity와 sample reorder, CPU cost를 바꾼다. pack efficiency만 보고 재현성과 mixture를 놓치지 않는다.

tie-break는 stable bin ID와 SampleID를 사용한다. hash-map iteration이나 thread completion 순서에 의존하면 같은 seed에서도 pack composition이 달라진다. pack manifest는 ordered child SampleID, source token interval, output interval과 boundary policy를 가진다.

oversized sample은 truncate, split 또는 reject한다. truncate는 삭제된 target 질량을 기록한다. split은 overlap, continuation marker와 position/mask 정책을 가진다. reject는 domain mixture를 바꾼다. 모델 context보다 긴 sample을 packer가 조용히 자르지 않는다.

bin이 닫히는 조건은 full, buffer flush, epoch/shard 끝 또는 latency limit일 수 있다. distributed rank마다 마지막 partial bin 수가 다르면 batch shape와 valid mass가 달라진다. drop/pad/emit 정책을 global objective와 연결한다. epoch 경계에서 sample을 다음 epoch bin으로 넘기는지 결정한다.

### 6.5.3 pack tensor materialization과 varlen kernel

child token arrays를 concat해 input IDs를 만든다. labels는 각 child 내부 next-token shift를 따르고 boundary crossing target을 ignore한다. segment ID 또는 cumulative lengths가 attention boundary를 표현한다. position ID는 child별 reset 또는 continuous policy를 따른다. loss mask는 original assistant/domain weight와 truncation을 반영한다.

copy를 줄이기 위해 preallocated buffer에 slice write할 수 있다. write cursor와 expected total length를 assertion한다. pinned memory, worker process와 tensor sharing은 lifetime을 추가한다. optimization 전에 canonical Python concat reference와 exact equality를 확인한다.

varlen interface를 쓰면 flattened token `[N]`과 cumulative sequence `[S+1]`를 만든다. `cu[0]=0`, monotonic, `cu[-1]=N`, 각 차이가 child length다. max sequence length metadata도 맞아야 한다. labels/weights는 flat coordinate와 정확히 정렬한다.

**sample mixture를 probability가 아닌 draw process로 읽는다.**

**draw process와 exhaustion을 상태로 만든다.**

domain weight `p_d`가 있어도 draw 구현은 여러 가지다. 매 sample 독립 categorical draw는 with-replacement다. domain을 먼저 고르고 해당 iterator에서 다음 item을 읽을 수 있다. finite domain shard를 without-replacement로 순회하고 epoch마다 비율을 맞출 수도 있다. 작은 domain exhaustion 처리에 따라 realized mixture가 다르다.

with-replacement는 같은 sample 반복 가능성이 있고 epoch 정의가 약하다. without-replacement는 available size와 weight가 충돌한다. oversampling은 iterator를 restart하거나 sample을 복제한다. undersampling은 일부 available data를 보지 않는다. SampleID duplicate와 coverage를 보고한다.

weighted interleave는 stopping strategy를 가진다. 첫 source가 끝나면 전체를 멈추거나, 모든 source가 끝날 때까지 재시작하거나, 확률을 남은 source에 재정규화할 수 있다. 설정 weight만으로 동작을 알 수 없다. source exhaustion fixture를 만든다.

temperature sampling은 base mass `n_d`에 지수 alpha를 적용해 `p_d∝n_d^alpha`처럼 만들 수 있다. alpha 1은 size 비례, 0은 domain 균등에 가깝다. 실제 formula와 minimum/maximum cap을 확인한다. document mass와 token mass 가운데 `n_d` 정의를 명시한다.

random draw를 재현하려면 mixture RNG와 domain-local iterator state를 모두 저장한다. domain draw sequence만 같아도 각 iterator shuffle/cursor가 다르면 SampleID가 달라진다. prefetch된 domain/sample event를 queue state에 포함하거나 replay policy를 둔다.

**확률 오차와 drift를 계산한다**

유한 window realized share는 configured p 주변에서 변동한다. 단순 편차 threshold는 rare domain에서 오경보를 낼 수 있다. multinomial 기대 분산 또는 confidence interval을 참고하되 sample이 packing/token weight로 독립이 아닐 수 있음을 적는다.

document share와 token share 차이는 length distribution으로 예상할 수 있다. domain d의 평균 token length가 `mu_d`면 document draw p가 만드는 장기 token mass는 대략 `p_d mu_d / sum p_j mu_j`다. truncation과 packing mask가 valid target mass를 다시 바꾼다. 실제 ledger로 교정한다.

drift 경보는 mixture가 변하는 세 경계에 따로 둔다. 먼저 설정 확률과 실제 draw 비율을 비교해 selector가 의도대로 작동했는지 본다. 다음으로 draw와 emitted 비율을 비교하면 source 고갈, filter, packing tail 때문에 사라진 sample을 찾을 수 있다. 마지막으로 emitted 비율과 valid-target 비율을 비교하면 길이 분포, truncation, label mask가 학습 기여도를 어떻게 바꾸었는지 드러난다. 설정과 draw가 일치하는데 valid share만 달라졌다면 sampling을 고칠 일이 아니라 source별 drop·truncation reason과 mask 정책을 조사해야 한다. 전체를 합친 KL 값 하나로는 이 책임 경계를 찾을 수 없다.

**curriculum을 schedule 함수로 구현한다.**

**schedule을 UpdateID 함수로 고정한다.**

max length를 2k에서 8k로 올리면 같은 token budget을 유지하기 위해 microbatch를 줄일 수 있다. accumulation, padding/packing 효율, attention kernel와 activation checkpoint policy가 달라진다. position distribution과 document fragmentation도 바뀐다. length 하나만 intervention했다고 쓰지 않는다.

schedule은 update, consumed token, wall time 또는 validation event를 독립변수로 쓸 수 있다. resume와 elastic batch에서 의미가 다르다. schedule function, boundary와 current phase를 checkpoint한다. config에서 재구성한다면 consumed mass ledger와 일치하는지 본다.

phase transition에서 optimizer LR, batch와 data mixture를 동시에 바꾸면 loss discontinuity 원인을 분리하기 어렵다. staggered transition 또는 factorial control을 고려한다. transition 전후 같은 evaluation slice와 gradient statistics를 둔다.

short-to-long curriculum은 long document가 초기에 truncate되어 보지 못한 target을 후반에 처음 노출한다. data order와 topic/domain이 length와 상관될 수 있다. phase별 unique DocumentID와 raw byte coverage를 보고한다. 단순 compute 절감과 curriculum effect를 분리한다.

**score의 생성과 유효기간을 관리한다**

difficulty는 reference model loss, target model excess loss, heuristic, reward, gradient norm 또는 learnability score일 수 있다. score producer model/checkpoint, tokenizer, context와 reduction을 기록한다. per-token과 per-document score를 구분한다.

target model이 학습하면서 difficulty 순서는 바뀔 수 있다. 오래된 score를 고정하면 curriculum이 stale해진다. 재계산 주기와 비용, asynchronous staleness를 둔다. score update가 sample selection과 같은 model checkpoint를 가리키는지 watermark를 사용한다.

loss가 높은 sample은 어렵거나 corrupt/다른 language일 수 있다. 무조건 hard mining하면 noise를 과대표집할 수 있다. quality/filter signal과 difficulty를 분리한다. high-loss tail을 수동/자동 audit하고 cap과 quarantine을 둔다.

RHO-Loss류 선택은 reference 또는 irreducible loss와 current excess를 사용해 learnable sample을 고를 수 있다. 구현 formula, batch candidate pool과 selection fraction을 source에서 확인한다. paper 이름만 붙이고 단순 top-loss sampler를 같은 algorithm으로 부르지 않는다.

**skill·domain curriculum을 graph로 표현한다.**

skill 간 prerequisite를 DAG로 두고 mastery estimate가 다음 sampling을 바꿀 수 있다. skill label source와 multi-label sample 처리를 정한다. graph가 사람이 만든 taxonomy인지 learned relation인지 구분한다. cycle와 disconnected skill을 검증한다.

mastery metric은 held-out evaluation 또는 training response다. training loss만 쓰면 exposure가 많은 skill이 좋아 보일 수 있다. uncertainty와 minimum exploration mass를 둔다. adaptive controller가 특정 skill을 영구 배제하지 않게 한다.

domain과 skill, length가 상관되면 marginal schedule을 독립 적용해 joint mixture가 예상과 달라진다. joint cell mass와 sparse cell을 본다. 원하는 constraint를 iterative proportional fitting이나 explicit joint sampler로 구현할 수 있지만 실제 code와 convergence를 검증한다.

**data weight가 loss까지 전달되는 경로를 추적한다.**

**sample weight를 optimizer contribution까지 추적한다.**

sample weight `w_i`를 sample mean loss에 곱하면 긴/짧은 sample 기여가 동일할 수 있다. token마다 `w_it`를 곱해 전체 weighted token sum으로 나누면 다른 objective다. domain weight, quality weight와 importance correction을 어디에 적용하는지 식으로 쓴다.

weighted loss `L=sum w_it l_it / sum w_it`에서 numerator와 denominator를 함께 aggregate한다. mean loss에 weight를 곱한 뒤 batch size로 나누는 구현과 비교한다. mask된 token은 denominator에서 빠진다. zero/negative weight 정책을 정한다.

sampling probability를 바꾸면서 importance weight `target_p/sample_p`를 적용하면 목표 분포의 unbiased estimate를 의도할 수 있지만 variance가 커진다. clipping/self-normalization이 bias를 추가한다. adaptive curriculum이 optimization target 자체를 바꾸는지 estimator만 바꾸는지 구분한다.

distributed rank별 weight mass가 다르면 DDP scaling을 전역 denominator에 맞춘다. count가 아니라 weight sum을 all-reduce한다. mixed precision에서 weight와 sum accumulation dtype을 본다. 작은 weight underflow를 확인한다.

**contribution ledger를 optimizer update에 연결한다.**

각 microbatch는 SampleID/token span, weight sum, loss sum과 UpdateID를 가진다. accumulation window의 global numerator/denominator가 scalar loss와 gradient를 만든다. overflow skip이면 contribution이 parameter update에 적용되지 않았음을 표시한다.

retry 또는 resume에서 같은 ContributionID가 두 update에 들어가지 않는지 본다. sample을 다시 계산했지만 첫 attempt update가 skip/rollback됐다면 duplicate data processing과 duplicate parameter contribution을 구분한다. durable update manifest가 contribution set을 가리킨다.

privacy deletion이나 data incident에서는 SampleID가 들어간 UpdateID와 descendant CheckpointID를 찾는다. exact influence 제거는 어렵지만 lineage가 영향 범위를 제공한다. offset 없는 packed tensor만 저장하면 원 sample을 찾기 어렵다.

## 6.6 처리량·prefetch·resume을 같은 상태 공간에서 본다

### 6.6.1 CPU pipeline과 GPU idle을 함께 측정한다

storage read, decompress, parse, normalize/tokenize 또는 token file read, filter, augmentation, packing, collate, pin과 H2D queue를 나눈다. batch ready timestamp와 GPU consume timestamp를 UpdateID로 연결한다. total data time만 보면 병목 owner를 알 수 없다.

worker 수를 늘리면 I/O와 CPU parallelism이 늘지만 memory, file descriptor와 contention도 늘어난다. tokenizer 내부 thread pool과 DataLoader process가 oversubscription을 만들 수 있다. CPU core pinning, NUMA와 storage locality를 본다. worker count sweep은 same SampleID/order와 performance를 별도 판정한다.

persistent worker는 epoch 사이 startup을 줄이지만 dataset/config hot change와 RNG reset semantics를 바꾼다. worker object가 old tokenizer/mixture를 계속 들고 있을 수 있다. revision heartbeat와 explicit restart를 둔다. memory leak과 open file을 관측한다.

prefetch factor는 queue depth와 host memory를 늘린다. compute overlap을 높일 수 있지만 checkpoint exactness와 stale curriculum response를 악화시킨다. adaptive selector update 뒤 이미 prefetched old-distribution batch가 몇 개 남는지 기록한다. control latency와 throughput trade-off다.

### 6.6.2 H2D overlap을 timeline으로 확인한다

pinned host tensor와 non-blocking copy는 overlap의 필요조건일 수 있지만 충분조건이 아니다. copy stream, event와 consumer stream wait를 확인한다. pageable source가 내부 staging copy를 만들 수 있다. profiler에서 H2D와 compute 구간이 겹치는지 본다.

packing을 GPU에서 하거나 tensor를 device에서 concat하면 CPU bottleneck을 줄일 수 있지만 additional kernel, memory와 graph dynamic shape가 생긴다. semantic parity를 CPU reference와 확인한다. 작은 copy를 많이 launch하는 것과 큰 packed copy를 비교한다.

GPU idle이 data 때문인지 distributed straggler 때문인지 rank별 batch ready와 collective wait를 함께 본다. 한 rank의 slow shard/I/O가 모든 rank를 멈출 수 있다. max/min/percentile과 offending DatasetShardID를 기록한다.

**분산 sampler를 topology 변경까지 복구한다.**

### 6.6.3 topology 변경에서도 global draw를 복구한다

전역 permutation `P(epoch,seed)`을 먼저 정의하고 rank r이 index position을 partition한다고 생각하면 topology 변경을 설명하기 쉽다. committed global positions 집합 또는 watermark를 저장하고 새 world size가 남은 suffix를 나눈다. rank-local RNG만 저장하는 방식보다 portable하다.

그러나 data filtering이나 iterable source가 동적이면 global random access permutation이 없을 수 있다. shard ID와 within-shard cursor, shuffle buffer content를 저장한다. external stream offset과 message acknowledgement도 durable state가 된다. 지원 가능한 resume 등급을 정직하게 낮춘다.

elastic scale-down에서 사라진 rank의 prefetched-but-uncommitted item을 재배정한다. committed contribution manifest에 없는 item은 다시 처리할 수 있다. scale-up에서도 이미 committed position을 새 rank가 다시 읽지 않게 한다. topology transition ID를 event log에 둔다.

**shard imbalance와 hotspot을 관측한다.**

shard별 byte, document/token, compression ratio, average processing time를 측정한다. 동일 파일 수 partition은 균형이 아닐 수 있다. cost-aware assignment를 하되 deterministic mapping과 checkpoint를 유지한다.

remote storage request와 local cache hit를 rank/shard별로 본다. popular shard를 모든 rank가 동시에 읽는 start-of-epoch thundering herd를 피한다. stagger/prefetch와 cache placement를 검토한다. data locality 때문에 sample order나 mixture가 바뀌면 기록한다.

corrupt shard를 한 rank만 skip하면 global sample set과 step 수가 달라지고 collective hang 가능성이 있다. error를 coordinator에 올리고 all-rank fail/replace policy를 적용한다. replacement sample과 mixture mass를 ledger에 남긴다.

**mixture 최적화 논문을 구현으로 검증하는 법.**

**objective·proxy·production sampler를 세 층으로 나눈다.**

논문 objective는 target validation utility와 domain weight 최적화를 정의할 수 있다. 실제 algorithm은 proxy model, gradient/loss statistics와 iterative update를 쓴다. production은 계산된 static weight만 가져오거나 online controller를 구현할 수 있다. 세 층을 같은 이름으로 합치지 않는다.

DoReMi류는 reference와 proxy training에서 domain excess loss와 robust weight update를 다룰 수 있다. 어떤 checkpoint/loss normalization과 smoothing을 쓰는지 본다. final weights를 큰 model data mixture로 옮길 때 domain mapping과 token budget을 확인한다.

RegMix류는 여러 mixture에서 proxy를 학습하고 downstream 관계를 회귀해 candidate를 고를 수 있다. proxy sample design, regression feature/target와 extrapolation 범위를 본다. code가 공개된 경우 config와 generated weight artifact를 고정한다. final model 결과를 regression의 확정 인과로 과장하지 않는다.

DSIR류 resampling은 target/source distribution density ratio를 feature space에서 추정할 수 있다. feature/tokenization, n-gram 또는 hash, smoothing과 clipping이 weight를 결정한다. duplicate와 rare feature의 ratio 폭주를 본다. resampled SampleID provenance를 보존한다.

DoGE나 gradient alignment 계열은 domain gradient와 target gradient relation을 사용할 수 있다. 어떤 parameter subset/proxy가 gradient를 대표하는지, distributed aggregation과 비용을 확인한다. gradient score staleness와 noise를 controller state에 넣는다.

Skill-It류는 skill relation과 online performance로 sampling을 조정할 수 있다. skill tag와 graph construction, update formula를 source에서 확인한다. 이름만 사용해 prerequisite curriculum 일반론을 구현됐다고 쓰지 않는다.

**공정한 ablation을 만든다.**

baseline과 adaptive run은 raw corpus, tokenizer, total valid token, model initialization, optimizer와 evaluation을 맞춘다. controller가 compute를 추가하면 total wall/compute budget 비교도 둔다. proxy training 비용을 숨기지 않는다.

static best weight를 결과를 본 뒤 고른 oracle과 비교할 때 정보 누출을 표시한다. validation tuning과 final test를 분리한다. domain별 metric과 aggregate weighting을 공개한다. seed와 uncertainty를 둔다.

controller option은 update interval, smoothing, temperature, min mass와 score source를 한 축씩 또는 interaction design으로 바꾼다. realized mixture와 quality, stability를 함께 본다. final metric만으로 mechanism을 설명하지 않는다.

**packing·mixture·curriculum 장애의 최초 차이.**

**최초 차이를 data graph에서 좁힌다**

spike UpdateID의 BatchDrawID와 PackID를 찾는다. child SampleID, domain/skill, length, truncation, valid count와 weight를 정상 window와 비교한다. input IDs와 boundary mask를 확인한다. model activation으로 넘어가기 전에 data mutation을 닫는다.

특정 domain realized share가 급증하면 configured selector state, draw event, emitted/valid mass를 단계별로 본다. configured가 변했으면 controller observation/apply event, draw만 변했으면 RNG/exhaustion, valid만 변했으면 length/mask/truncation이다.

throughput 저하는 rank별 batch-ready와 pack efficiency, sequence shape를 본다. curriculum phase change가 max length와 kernel을 바꾸었는지 확인한다. storage shard/worker와 H2D timeline을 본다. GPU kernel 문제로 바로 넘기지 않는다.

**resume 뒤 소비 순서를 검산한다**

checkpoint 전후 next domain draw, SampleID, augmentation, PackID를 비교한다. domain이 다르면 mixture RNG/state, sample이 다르면 domain iterator/sampler, augmentation만 다르면 worker RNG, pack만 다르면 open bins/tie-break state다. first difference owner가 선명하다.

sample sequence는 같지만 loss가 다르면 boundary mask, position, labels/weight와 model state로 넘어간다. prefetch queue replay가 tensor content까지 같았는지 본다. DatasetRevision과 tokenizer digest도 확인한다.

topology 변경 뒤만 다르면 global-to-rank partition과 committed watermark를 본다. sample-exact를 지원하지 않는 checkpoint라면 결과를 failure가 아니라 contract 밖으로 표시하되 duplicate/skip mass를 보고한다.

**Silent leakage를 잡는다.**

packed child B의 token을 고정하고 child A만 perturb한다. B logits/gradient가 boundary policy상 독립이어야 한다면 불변인지 본다. attention boundary, recurrent/convolution reset과 label crossing을 각각 깨뜨린 negative fixture를 둔다.

train/validation DocumentID 또는 dedup cluster overlap을 찾는다. curriculum score가 validation label/metric을 사용해 selection에 leakage하지 않는지 본다. adaptive controller의 observation source를 lineage로 확인한다.

**재현 패키지와 다음 소비자에게 넘길 계약.**

**Data graph를 machine-readable하게 제출한다.**

DatasetRevision은 raw shard와 filter/dedup/tokenizer를 가리킨다. SampleID는 raw DocumentID/byte span과 token span, domain/skill/quality를 가진다. BatchDrawID는 sampler/mixture event, PackID는 child layout과 masks, ContributionID는 loss weight와 UpdateID를 가진다.

selector state에는 configured weight, score, smoothing/dual state, observation/apply watermark와 RNG가 있다. sampler state에는 global/within-shard cursor, epoch와 worker state가 있다. packer state에는 open bins와 tie-break가 있다. root checkpoint가 consistent cut으로 이들을 가리킨다.

**실행 matrix를 제출한다.**

unpacked/packed parity, fixed/adaptive mixture, curriculum phase boundary, worker/prefetch, single/distributed와 resume/topology 변경을 행으로 둔다. 각 row는 config/source, input digest, expected invariant, status와 raw report를 가진다.

negative control은 boundary label/mask, duplicate/drop, stale score, missing selector state, lost prefetch, changed world size와 corrupt shard를 포함한다. failure가 optimizer effect 전에 감지되는지 본다.

performance report는 raw bytes/s, emitted/valid token/s, pack efficiency, batch-ready wait, H2D overlap과 GPU step을 잇는다. quality report는 domain/skill metric과 realized mass를 가진다. 둘을 한 aggregate throughput이나 loss로 축소하지 않는다.

**다음 장들에 넘긴다.**

7–10장은 정확한 packed input, segment/position/mask와 contribution denominator를 받는다. 15–17장은 sampler/packer/selector의 rank owner와 checkpoint schema를 받는다. 24장은 evaluation sample independence, 26장은 mass/latency metric, 27장은 dataset artifact digest를 받는다.

18–20장의 SFT/RL은 assistant/reward mask와 rollout data mixture를 같은 ledger로 확장한다. 21장의 multimodal은 modality token/frame budget을 추가한다. 25장의 red-team data도 target mixture와 leakage boundary를 가진다.

이 인계를 받은 consumer가 다른 tokenizer, sample set이나 denominator를 쓰면 explicit derivation을 만든다. 같은 이름의 dataset만으로 동일성을 주장하지 않는다. PackID와 ContributionID checksum이 장간 연결을 실제 artifact로 만든다.

**장 완료 문장.**

packing은 공간 절약이 아니라 attention/position/label 경계를 함께 보존하는 objective compiler다. mixture는 config 확률이 아니라 draw, emitted, valid와 weighted contribution 질량이다. curriculum은 observation과 state를 가진 feedback controller다.

독자는 option을 바꿀 때 어떤 iterator, RNG, queue, bin, mask, denominator와 checkpoint가 달라지는지 설명한다. 기대 throughput이나 quality가 나오지 않으면 data graph의 최초 차이를 찾는다. source, test와 실행 범위를 구분한다.

정상과 negative fixture가 모두 통과하고 resume 뒤 지원하는 등급에서 같은 contribution sequence가 복구될 때 6장을 승인한다. 미실행 multi-cluster/topology cell은 dependency와 command를 가진다. 이 기준이 데이터 선택을 모델 밖의 우연이 아니라 검증 가능한 training mechanism으로 만든다.

**작은 corpus로 packing을 손계산한다.**

문서 A 길이 5, B 길이 3, C 길이 7, D 길이 2이고 context capacity가 8이라고 하자. input order first-fit은 A 뒤 B를 넣어 `[A5,B3]`, C 뒤에는 남은 1에 들어갈 것이 없어 `[C7]`, D는 `[D2]`가 된다. allocated 24칸 가운데 token 17개를 써 효율은 70.8%다.

best-fit decreasing은 C7, A5, B3, D2 순으로 볼 수 있다. C와 남은 1, A+B exact, D가 남아 결과 allocated bin 수가 세 개로 같을 수 있다. 다른 length 집합에서는 줄어든다. 작은 사례 하나의 우위를 일반화하지 않고 distribution simulation을 한다.

각 pack의 actual input/labels를 쓴다. A token이 `[a0..a4]`, B가 `[b0..b2]`라면 concat input은 그대로지만 label에서 `a4→b0` 위치를 ignore한다. A와 B의 segment ID가 다르고 block-diagonal causal mask가 이를 차단한다. position reset 정책이면 B position은 0에서 다시 시작한다.

A token 하나를 바꾸어 B output과 gradient가 달라지는지 확인한다. boundary mask가 맞으면 B가 A를 보지 않는다. cross-document attention을 recipe로 의도했다면 fixture expected를 다르게 정의하고 이를 objective change로 선언한다. 어느 쪽이든 우연한 default로 남기지 않는다.

assistant-only sample이면 각 child의 assistant mask를 concat하고 boundary target를 제거한다. token 17개가 모두 valid target이 아닐 수 있다. pack efficiency 70.8%와 optimization efficiency `valid_labels/24`를 별도로 계산한다.

truncation으로 C를 6으로 자르면 bin 효율은 높아질 수 있지만 삭제 target이 생긴다. 원 byte/token span, truncated count와 domain mass를 ledger에 남긴다. efficiency 개선을 데이터 보존으로 착각하지 않는다.

**mixture draw를 숫자로 정산한다.**

domain X,Y,Z의 configured document draw 확률이 0.5,0.3,0.2이고 평균 token length가 100,500,50이라고 하자. 장기 예상 token mass는 50:150:10이므로 약 23.8%,71.4%,4.8%다. document 확률과 token contribution이 크게 다르다.

Y가 assistant-only valid ratio 0.2, X가 0.8, Z가 0.5라면 예상 valid target mass는 40:30:5가 되어 53.3%,40%,6.7%다. configured document, emitted token과 valid target 세 표가 서로 다르다. 어떤 분포를 목표로 하는지 먼저 정한다.

domain weight를 token target으로 맞추려면 document sampling을 평균 length와 mask ratio로 보정할 수 있으나 distribution tail과 truncation 때문에 실제 ledger feedback이 필요하다. online 재조정은 controller가 되며 smoothing, minimum mass와 checkpoint state를 가진다.

100 draws에서 observed 60,25,15가 나왔다고 즉시 drift로 판정하지 않는다. expected stochastic variation과 window를 본다. 하지만 Z가 연속 여러 window 0이면 minimum mass/iterator exhaustion을 확인한다. rare safety domain은 aggregate KL이 작아도 별도 minimum-count alert를 둔다.

Y source가 먼저 끝나 stopping strategy가 전체 stop이면 remaining data를 보지 못한다. Y를 restart하면 duplicate가 늘어난다. Y를 제외하고 재정규화하면 후반 mixture가 X/Z로 바뀐다. 세 정책을 fixture로 만들고 chosen behavior를 config보다 actual source에서 확인한다.

**curriculum controller를 수치 상태기계로 만든다.**

세 domain의 score estimate `s=[0.2,0.5,0.3]`, 이전 weight `p=[0.4,0.4,0.2]`를 두자. controller가 score에 temperature softmax를 적용하고 EMA로 새 weight를 만들 수 있다. exact formula, temperature와 EMA coefficient를 artifact에 적는다.

minimum mass 0.05를 적용한 뒤 renormalize하는 순서와 softmax 전에 logit floor를 적용하는 것은 다르다. 손계산 fixture로 expected p를 만든다. floating tolerance와 sum-to-one, nonnegative invariant를 확인한다.

observation step 100의 score가 asynchronous하게 step 120에 도착하고 apply가 128이면 세 timestamp를 저장한다. resume가 124에서 일어나면 pending observation queue를 복구할지 다시 계산할지 정책이 필요하다. apply event가 두 번 일어나지 않게 ObservationID를 idempotency key로 쓴다.

controller update 뒤 prefetch queue에 old distribution batch가 16개 남아 있다면 realized response가 지연된다. queue drain, invalidation 또는 accepted control latency를 결정한다. selector weight graph만 보고 즉시 mixture가 바뀌었다고 기대하지 않는다.

score가 NaN, domain 누락 또는 sample 수 부족이면 fail/hold/fallback policy가 있다. last-known-good를 쓰면 stale age를 metric으로 낸다. uniform fallback이 안전한지 domain governance constraint를 확인한다. 조용히 NaN을 0으로 바꾸지 않는다.

checkpoint에는 current p, EMA score, update count, pending observation, RNG와 applied ID set/watermark가 들어간다. load 뒤 next p와 domain draw가 uninterrupted controller와 같은지 비교한다.

**분산 data failure를 사건표로 푼다.**

32 rank 중 rank 17만 batch ready가 2초 늦고 나머지는 collective에서 기다린다고 하자. GPU profiler에는 all-reduce 시간이 커 보일 수 있지만 first cause는 data straggler다. rank별 `BatchRequested`, `BatchReady`, `H2DCompleted`, `ForwardStarted`, `CollectiveEntered`를 같은 UpdateID로 정렬한다.

rank 17의 worker, DatasetShardID와 storage request를 본다. 압축 block이 크거나 corrupt retry, cache miss, NUMA remote memory일 수 있다. 같은 shard를 다른 rank/worker에서 읽는 2×2 통제로 data와 host 원인을 분리한다. model kernel을 먼저 바꾸지 않는다.

corrupt sample을 rank 17만 skip하면 그 rank batch valid count와 sample set이 달라진다. DDP는 실행될 수 있지만 global objective가 바뀌고 iterable 길이 차이로 다음 step hang 가능성이 있다. coordinator가 replacement 또는 all-rank abort를 결정한다.

replacement는 같은 domain/length bucket에서 deterministic SampleID를 선택하고 original/replacement를 ledger에 남긴다. corrupt artifact는 quarantine하고 retry limit을 둔다. 무한 retry가 cluster 전체를 멈추지 않게 한다. 그러나 silent skip으로 availability를 사지 않는다.

node failure로 world size가 32에서 31 또는 새 32 topology로 바뀌면 last committed global contribution을 기준으로 repartition한다. rank-local prefetched item은 committed가 아니면 재배정한다. optimizer/global batch policy와 LR도 topology 변화에 맞춰야 한다. data recovery만으로 recipe 동일성이 보장되지 않는다.

**option 하나가 바꾸는 객체를 표로 설명한다.**

`packing=True`는 packer state, segment/mask/position/label tensor, batch length distribution과 attention backend를 바꾼다. 기대 효과는 padding 감소다. 실패 관측은 cross-document leakage, valid mass, kernel fallback과 pack CPU time이다. rollback은 unpacked collator와 cache invalidation이다.

`max_sequence_length`는 truncation, bin capacity, position distribution, attention compute와 batch size를 바꾼다. curriculum schedule의 phase state일 수 있다. 길이만 늘리고 GPU OOM이면 activation/microbatch/checkpointing을 본다. quality 변화는 raw byte coverage와 long evaluation을 함께 본다.

`shuffle_buffer_size`는 randomization quality, memory, cursor/checkpoint 크기와 prefetch를 바꾼다. 값이 1이면 거의 source order, 매우 크면 더 섞이지만 resume state가 크다. SampleID autocorrelation과 exact resume를 측정한다.

`num_workers`와 `prefetch_factor`는 process/RNG/queue, host memory와 control latency를 바꾼다. throughput이 늘지 않으면 CPU oversubscription, storage contention과 GPU bottleneck을 본다. exact output order와 augmentation parity를 별도 test한다.

`drop_last`는 discarded sample/token/valid mass와 rank step 정렬을 바꾼다. DataLoader와 sampler 중 적용 위치를 확인한다. rare domain tail이 반복적으로 버려지지 않는지 본다. rollback은 명시 padding/remainder policy다.

mixture `temperature`, `min_weight`, `update_interval`, `smoothing`은 selector state와 realized distribution response를 바꾼다. quality 기대와 exploration/stability 대가를 함께 쓴다. checkpoint 누락과 stale score negative fixture가 필요하다.

`curriculum_phase`는 data subset, length/difficulty, batch와 schedule clock을 바꾼다. manual override는 새 RunID/event를 만든다. resume가 phase boundary를 두 번 적용하지 않게 한다. phase 전후 evaluation과 mass ledger를 비교한다.

`seed`는 한 숫자가 아니라 mixture, sampler, worker, augmentation와 pack tie-break stream을 파생한다. base seed 변경은 전체 sample/pack trajectory를 바꾼다. component seed를 독립 manifest로 둔다. 같은 seed를 reproducible correctness의 충분조건으로 쓰지 않는다.

## 6.7 작은 corpus로 종단 감사를 수행한다

첫 15분에는 DatasetRevision, tokenizer, columns, sample/domain/quality schema와 raw shard를 고정한다. map/iterable, pretokenized 여부와 offset을 본다. split과 dedup cluster leakage를 확인한다. file 수가 아니라 byte/token/domain mass를 계산한다.

다음 15분에는 sampler와 mixture call path를 그린다. configured weight, actual stopping strategy, RNG와 domain iterator state를 찾는다. 열 개가 아니라 충분한 draw fixture로 SampleID sequence를 만든다. source exhaustion과 rare domain을 주입한다.

다음 15분에는 collator/packer를 본다. raw example에서 IDs, masks/weights, packed children와 labels를 출력한다. unpacked reference와 logits/loss parity를 위한 fixture를 만든다. truncation과 all-ignore sample을 넣는다.

다음 15분에는 DataLoader worker와 distributed partition을 본다. rank/worker sharding이 중복 적용되지 않는지, set-epoch/seed와 last remainder를 확인한다. batch-ready timeline과 prefetch queue를 본다. rank별 SampleID coverage를 합친다.

다음 15분에는 checkpoint schema를 본다. model checkpoint에 sampler/selector/packer/RNG가 어떻게 묶이는지 표로 만든다. step K에서 save/load하고 next draw, pack와 contribution을 비교한다. payload 또는 worker state 하나를 빼 negative fixture를 만든다.

마지막 15분에는 throughput과 objective mass를 정산한다. storage/CPU/packing/H2D/GPU time, configured/draw/emitted/valid/weighted domain share를 한 UpdateID window에 놓는다. 가장 큰 gap과 owner를 고른다. 미실행 topology는 command를 남긴다.

감사 결과는 “데이터 로더가 느리다”가 아니다. 예를 들어 “rank 17, shard S42 gzip block p99가 batch-ready를 1.8초 지연해 all-reduce wait로 관측됨”처럼 좁힌다. mixture 문제도 어느 mass 단계에서 처음 달라졌는지 쓴다.

수정은 semantic parity와 performance를 함께 검증한다. worker 수를 늘린 뒤 SampleID/augmentation/PackID가 정책상 같거나 의도된 차이를 가진다. pack algorithm을 바꾼 뒤 boundary objective와 realized mixture를 확인한다. control update를 빠르게 한 뒤 stability와 checkpoint를 본다.

최종 제출물은 source-to-state 표, mass ledger, boundary fixture, resume diff, distributed timeline와 change sheet다. 정상과 negative fixture가 같은 command suite에 있다. 독립 검토자는 option 하나를 바꾸어 예상 state와 first difference를 재현한다.

이 실전 시험을 통과한 pipeline만 긴 training run에 들어간다. model이 아무리 정확해도 잘못된 sample, mask, weight나 재개 순서를 받으면 원하는 objective를 학습할 수 없다. data path를 먼저 닫는 것이 계산 자원을 지키는 가장 싼 검증이다.

### 6.7.1 framework 추상화 아래의 실제 상태

map-style dataset은 index에서 sample을 얻는 계약이 있어 distributed permutation과 random access resume를 설계하기 쉽다. 그러나 `__getitem__` 안의 random augmentation, remote read와 dynamic filter는 index만 같아도 결과를 바꿀 수 있다. SampleRevision과 RNG를 함께 고정한다.

iterable dataset은 stream, generator 또는 shard iterator를 제공한다. epoch length가 알려지지 않거나 external source가 계속 들어올 수 있다. rank/worker sharding, shuffle buffer와 stopping은 dataset code가 소유할 수 있다. external sampler를 또 붙이면 double shard가 된다.

dataset transform은 eager materialization, lazy map 또는 batched map일 수 있다. fingerprint/cache가 transform function, arguments와 input revision을 key로 쓰는지 본다. closure나 external file/env를 hash하지 못하면 stale cache가 생긴다. effective transformation digest를 별도 manifest로 둔다.

cache file은 speed artifact이지 source of truth가 아니다. schema, row order, tokenizer와 processor revision을 검증한다. partial write와 concurrent writer를 atomic commit으로 처리한다. corrupt cache를 raw에서 재생성할 수 있어야 한다.

streaming shuffle은 buffer에서 random item을 뽑고 새 item으로 채우는 방식일 수 있다. buffer보다 먼 순서는 완전 permutation과 다르다. shard order randomization과 example buffer randomization을 분리한다. buffer content와 upstream cursor 없이는 exact resume가 안 된다.

framework의 `interleave`, `concatenate`, `select`, `filter` 같은 연산은 SampleID lineage를 보존해야 한다. row index는 변환 뒤 달라지므로 stable source ID를 유지한다. filter가 non-deterministic function이나 external service를 쓰지 않게 한다. 제거 이유와 version을 남긴다.

column casting과 schema inference도 data 의미를 바꿀 수 있다. integer label width, null/default, list flatten과 media decoder를 확인한다. JSON string에 담긴 structured conversation을 단순 text로 처리하지 않는다. schema validator가 training 전에 전수 또는 충분한 boundary 검사를 한다.

framework upgrade는 default multiprocessing, fingerprint, decoding과 formatting을 바꿀 수 있다. fixed revision source/test와 canonical dataset slice diff를 실행한다. file count와 row count가 같아도 field bytes와 SampleID order를 비교한다.

### 6.7.2 품질 feedback과 curriculum을 분리한다

quality filter는 corrupt, spam, duplicate와 policy violation을 줄이려는 data governance 단계다. curriculum difficulty는 학습 순서와 비율을 바꾸는 optimization 단계다. high loss를 quality bad로, low quality score를 difficulty hard로 자동 치환하지 않는다.

quality model도 checkpoint, tokenizer와 threshold를 가진다. score uncertainty, language/domain calibration과 false positive를 본다. threshold 변경은 DatasetRevision을 만든다. deleted/quarantined sample과 retained mass를 domain별로 보고한다.

curriculum controller가 quality score를 feature로 쓸 수 있지만 hard exclusion과 low sampling weight를 구분한다. minimum exploration을 두더라도 legal/privacy exclusion sample은 0이어야 한다. optimization controller가 governance constraint를 덮지 못하게 constraint layer를 둔다.

feedback loop에서 model failure sample을 더 뽑으면 data distribution이 바뀐다. online hard-example mining, red-team data와 user feedback은 collection policy와 consent, dedup와 contamination을 가진다. SampleRevision과 observation model을 연결한다.

evaluation failure를 training selector에 넣으면 그 evaluation set은 더 이상 untouched test가 아니다. curriculum tuning validation과 final holdout을 분리한다. benchmark contamination lineage를 기록한다. public benchmark exact item을 반복 학습해 metric을 올리지 않는다.

quality와 difficulty를 2축 scatter로 보고 high-quality-hard, high-quality-easy, low-quality-hard, low-quality-easy를 분리한다. 각 cell의 domain/length와 sample을 감사한다. single scalar로 합치면 서로 다른 action을 잃는다.

controller objective는 quality constraint 아래 utility를 최적화할 수 있다. minimum domain, maximum duplicate, safety coverage와 compute budget을 explicit constraint로 둔다. infeasible constraint가 나오면 조용히 normalize하지 않고 error/relaxation report를 낸다.

### 6.7.3 장 전체 causal graph를 재계산한다

raw dataset artifact가 DocumentID와 bytes를 만든다. filter/dedup/quality가 eligible SampleID와 attribute를 만든다. tokenizer가 token sequence와 offsets를 만든다. mixture와 curriculum selector가 next domain/sample distribution을 만든다. sampler가 BatchDrawID를 만든다.

worker transform과 augmentation이 SampleRevision을 만들고 packer가 PackID, segment, position, mask와 labels를 만든다. collator가 batch tensor와 valid/weight mass를 만든다. model loss가 ContributionID를 만들고 optimizer가 UpdateID에 적용한다. checkpoint가 model과 data controller state를 consistent cut으로 보존한다.

edge마다 producer, input revision, function/options, output digest와 event time이 있다. data graph와 training event graph가 UpdateID에서 만난다. sample이 어느 update에 기여했는지, update가 어느 checkpoint/evaluation으로 갔는지 양방향 탐색할 수 있다.

이 graph에서 packing option은 PackID 이후 mask/position/labels를 바꾸고, mixture option은 draw distribution과 후속 mass를 바꾸며, curriculum은 observation에서 selector state로 feedback edge를 추가한다. worker/prefetch는 item lifetime과 checkpoint cut을 바꾼다.

성능 graph도 같은 ID를 쓴다. shard read, worker ready, pack, H2D, forward와 collective timestamp를 연결한다. GPU all-reduce wait의 upstream data straggler를 찾는다. throughput 최적화가 sample order와 state 의미를 바꾸었는지 동시에 본다.

장애 graph는 last-good/first-bad edge를 찾는다. configured weight, domain draw, SampleID, transformed tensor, PackID, contribution과 update 순으로 비교한다. model loss에서 시작해 raw data까지 무작정 검색하지 않는다.

복구 graph는 last committed CheckpointID와 child data state를 load하고 next BatchDrawID/PackID/ContributionID를 비교한다. sample-exact가 지원되지 않으면 duplicate/skip set과 numerical impact를 보고한다. 지원 범위를 과장하지 않는다.

독립 검토자는 SampleID 하나를 골라 raw byte에서 optimizer update까지 추적한다. 이어 UpdateID 하나를 골라 모든 contributing SampleID와 pack boundary, mixture weight를 복원한다. selector weight 하나를 골라 observation, apply와 realized response를 찾는다.

negative control은 graph edge 하나를 끊는다. tokenizer digest mismatch, sampler cursor rollback, open bin 누락, stale score, label boundary와 rank shard duplication을 주입한다. validator가 해당 edge에서 실패하고 downstream update가 발생하지 않아야 한다.

최종 지원 matrix는 dataset type, packing/mixture/curriculum, workers, world size와 resume 조합을 가진다. 각 cell은 source/test/observed/unresolved를 가리킨다. 단일 GPU map dataset 통과를 streaming multi-cluster에 복사하지 않는다.

문서의 모든 “왜”도 graph에 답을 가진다. packing은 allocated token을 줄이려 들어갔지만 objective boundary를 보존해야 한다. mixture는 제한된 compute를 domain에 배분하려 들어갔지만 realized mass를 검증해야 한다. curriculum은 학습 시점별 utility를 높이려 들어갔지만 feedback 안정성과 복구 state를 가져야 한다.

이 causal graph와 수치 ledger, negative fixture가 함께 있을 때 독자는 새 data stack을 외운 API가 아니라 상태기계로 해부한다. 긴 학습을 시작하기 전에 잘못된 objective와 silent data drift를 작은 fixture에서 발견한다. 6장의 실무적 가치는 바로 그 조기 판정 능력이다.

최종 인수 문장은 artifact로 재생성된다. 고정 DatasetRevision에서 지원하는 topology와 option 조합은 예상 SampleID/PackID/contribution 질량을 만들고, 중단 뒤 선언한 등급으로 복구되며, 성능 최적화는 semantic parity를 유지한다. 미실행 조합은 명시돼 있다.

**source-to-test 추적표.**

dataset constructor 행에는 input shard, schema와 revision을 넣는다. map/iterable 선택과 sharding owner를 적는다. fixture는 row count보다 SampleID, raw bytes와 field type을 확인한다. null, corrupt media와 oversized document를 negative로 둔다.

filter/dedup 행에는 function/model revision, threshold와 removal reason을 넣는다. fixture는 exact duplicate, near duplicate, protected rare language와 false-positive boundary를 가진다. deterministic transform과 cache invalidation을 확인한다.

tokenized dataset 행에는 tokenizer digest, source offsets와 storage dtype을 넣는다. maximum ID와 vocabulary, length와 raw mapping을 확인한다. stale cache, uint overflow와 wrong tokenizer를 negative로 둔다.

mixture 행에는 source list, configured weight, stopping strategy, RNG와 iterator를 넣는다. fixture는 finite unequal source, exhaustion, minimum rare mass와 resume다. draw sequence, SampleID duplicate/coverage와 realized document/token share를 본다.

curriculum 행에는 score producer, schedule clock, selector update와 state를 넣는다. fixture는 phase boundary, stale observation, NaN score, missing domain과 resume pending update다. apply idempotency와 response latency를 본다.

sampler 행에는 global permutation 또는 shard/cursor, epoch, rank/worker projection과 remainder 정책을 넣는다. fixture는 dataset size가 world/batch로 나뉘지 않는 경우, set-epoch 누락, topology change와 corrupt shard다. global coverage/overlap을 검사한다.

packer 행에는 capacity, algorithm, buffer, tie-break와 open bins를 넣는다. fixture는 exact fit, oversized, final partial, equal-length tie와 crash with open bin이다. child spans, mask/position/labels와 unpacked parity를 본다.

collator 행에는 padding, truncation, label/weight와 tensor dtype/layout을 넣는다. fixture는 variable valid count, all-ignore, assistant mask, modality와 boundary다. loss numerator/denominator를 손계산한다.

DataLoader 행에는 worker init, seed, prefetch/persistent, pin과 queue state를 넣는다. fixture는 worker 수 변경, stochastic transform, evaluation insertion와 checkpoint다. SampleRevision과 next BatchDrawID, batch-ready timing을 본다.

distributed 행에는 data/model process group, rank batch, denominator collective와 elastic owner를 넣는다. fixture는 rank별 unequal valid mass, slow/corrupt rank, scale down/up다. single-process global batch gradient와 contribution set을 reference로 둔다.

checkpoint 행에는 model/optimizer보다 data-specific selector/sampler/packer/worker/RNG child를 적는다. fixture는 child missing, step mismatch, partial publish와 previous fallback이다. latest filename이 아니라 committed root를 선택한다.

monitoring 행에는 configured/draw/emitted/valid/weighted mass, unique/duplicate/drop, pack efficiency, queue depth, batch-ready와 H2D를 넣는다. metric label cardinality를 통제하면서 raw incident에는 SampleID/ShardID pointer를 남긴다.

각 source row는 repository, commit, path, symbol과 selected branch를 가진다. documentation option 설명만으로 implementation state를 확정하지 않는다. upstream test는 assertion, parameterized cases와 skip을 적는다. local fixture와 executed environment를 분리한다.

정적 review만 끝난 row는 `StaticReviewed`다. test가 있지만 실행하지 않았으면 `TestAvailable`이다. command와 raw report가 있으면 `Observed`다. reference나 tolerance가 없어 판정 못 하면 `Inconclusive`다. 성공처럼 보이는 빈칸을 허용하지 않는다.

최종 30분 재감사는 checksum과 링크부터 시작한다. DatasetRevision/tokenizer/config가 장 전체에서 일치하는지 본다. numeric 표의 denominator와 단위를 확인한다. code 조각과 설명의 stopping, mask와 state field가 같은지 본다.

다음으로 정상 fixture 한 개와 negative fixture 한 개를 재생한다. 정상 sample의 raw→token→pack→contribution을 추적하고, boundary mask 또는 sampler cursor를 깨뜨려 expected gate를 확인한다. 수정 이후 회귀 command가 dossier에 남아야 한다.

마지막으로 미실행 matrix를 읽는다. multi-node storage, elastic topology, streaming source와 adaptive controller의 production-scale cell은 필요한 cluster, command와 invariant를 가진다. 작은 static fixture의 통과를 실제 대규모 처리량이나 failure tolerance로 승격하지 않는다.

이 감사표가 닫히면 6장은 분량을 채운 설명이 아니라 재사용 가능한 조사 절차가 된다. 새로운 framework의 API가 달라도 dataset, selector, sampler, packer, contribution과 checkpoint owner를 찾아 같은 표에 배치할 수 있다.

마지막으로 질량과 상태를 함께 보존해야 한다. 먼저 어떤 data가 얼마나 draw되어 token·target·weight로 바뀌었는지 추적한다. 이어 다음 선택을 결정하는 RNG, cursor, queue, bin과 controller state를 저장한다. 질량 원장만 있으면 다음 sample을 재현할 수 없고, 상태만 있으면 실제 objective에 어느 data가 얼마나 기여했는지 알 수 없다. 어느 한쪽이라도 빠지면 학습 objective와 resume를 증명할 수 없다.

**독립 검토 질문과 반증 기준**

데이터 엔지니어는 raw shard에서 SampleID와 token cache까지 설명한다. filtering, dedup, tokenizer와 schema 변경이 어떤 artifact를 invalidate하는지 말한다. cache hit가 stale data를 숨기지 않는지 canonical slice로 확인한다.

학습 엔지니어는 BatchDrawID와 PackID에서 loss numerator/denominator까지 설명한다. padding과 valid target, domain/sample weight를 구분한다. accumulation과 DDP가 전역 weighted objective를 만드는 식을 제출한다.

분산 엔지니어는 rank/worker shard, prefetch와 elastic resume를 설명한다. global coverage와 overlap, corrupt rank policy와 last committed contribution을 보여준다. topology 변경 뒤 sample-exact 지원 여부를 명시한다.

연구자는 mixture/curriculum algorithm의 paper objective, proxy와 production implementation을 분리한다. score producer, staleness, controller state와 ablation을 제시한다. final metric만으로 mechanism을 증명하지 않는다.

운영자는 configured/draw/emitted/valid/weighted mass와 batch-ready timeline을 같은 UpdateID에서 읽는다. domain drift와 GPU idle의 첫 owner를 찾는다. automatic retry/skip이 data loss를 만들지 않는지 event를 본다.

독립 검토자는 PackID 하나를 골라 child sample, offsets, segment, position, labels와 weights를 재구성한다. unpacked reference와 objective parity를 확인한다. boundary token을 바꾸고 다른 child의 output이 정책대로 불변인지 본다.

이어 CheckpointID 하나를 골라 selector, sampler, open bin, worker/RNG와 prefetch state를 확인한다. new process에서 next draw/pack/contribution을 비교한다. 누락 field 하나를 제거해 validator가 지원 등급을 낮추거나 load를 거부하는지 본다.

마지막으로 option 하나를 바꾼다. max length, shuffle buffer, worker/prefetch, mixture temperature 또는 curriculum phase가 바꾸는 객체와 expected effect를 실행 전에 적는다. 실제 first difference와 throughput/quality trade-off를 보고한다.

모든 질문의 답은 문장만이 아니라 source card, artifact digest, numeric ledger와 raw report를 가리킨다. 미실행 multi-cluster cell은 그대로 남는다. 실행 권한이 없다는 이유로 expected observation을 실제 결과처럼 쓰지 않는다.

책의 다른 장도 이 계약을 소비한다. tokenizer와 model이 같아도 data contribution sequence가 다르면 다른 RunID다. checkpoint weight가 같아도 next sample과 selector state가 다르면 training resume가 아니다. 반대로 pack layout이 달라도 objective-equivalent임을 수치로 증명하면 명시 derivation을 만들 수 있다.

최종 승인자는 root manifest를 서명한다. dataset, tokenizer, sampler/selector/packer code, effective options와 checkpoint schema가 root 아래 있다. canonical 정상/실패 fixture의 report도 child다. 설명과 artifact가 다르면 release를 멈춘다.

이 상호검토가 끝나면 packing·mixture·curriculum은 경험적 레시피 목록이 아니다. 데이터가 gradient로 변하는 질량 보존 과정이며, 중단과 분산에서도 복구해야 하는 상태기계이고, 효과와 비용을 반례로 검증할 수 있는 설계가 된다.

승인 뒤에도 mass dashboard와 state heartbeat를 유지한다. configured probability가 그대로여도 source exhaustion, length drift, mask 변화와 corrupt retry가 realized objective를 바꿀 수 있다. DatasetRevision이나 code가 바뀌면 baseline window를 새로 만든다.

운영 중 발견한 장애는 같은 SampleID/PackID fixture로 축소한다. 원본 민감 data를 그대로 복제하지 않고 구조와 boundary를 보존한 synthetic minimal case를 만든다. fix와 함께 test suite에 넣고 영향 checkpoint를 lineage로 표시한다.

성능 최적화도 재승인을 받는다. new packer, worker 수, storage cache, GPU collator 또는 async selector가 들어오면 semantic parity, exact 또는 declared resume, realized mass와 throughput을 다시 비교한다. 빠르지만 다른 objective를 만드는 변경은 별도 실험이지 drop-in 최적화가 아니다.

장기적으로 dataset이 커지고 cluster가 바뀌어도 root 질문은 유지된다. 어느 원본이 선택됐고 어떤 경계와 weight로 loss에 들어갔으며, 다음 선택을 결정하는 상태가 어디에 보존되는가. 세 답이 artifact와 사건으로 이어져야 한다.

최종 보고서는 성공 범위와 실패·미실행 범위를 같은 크기로 다룬다. 지원하지 않는 streaming source, topology migration이나 adaptive controller 조합을 숨기지 않는다. 그 정직한 경계가 다음 실험이 무엇을 검증해야 하는지 알려준다.

이제 6장은 독자가 새로운 데이터 framework를 만나도 API 문서에 머무르지 않게 한다. source에서 실제 draw와 state owner를 찾고, 수치 ledger로 목적함수를 검산하며, 장애를 최초 질량 또는 상태 차이로 좁힐 수 있게 한다.

각 결론은 고정 revision, 재실행 명령, 입력 digest와 관측 report를 가리키며 추정값을 실행 결과로 표기하지 않는다.

새 학습 recipe를 받으면 먼저 DataRevision과 tokenizer를 고정한다. 다음으로 raw document에서 segments까지 unique coverage를 계산한다. packer의 tail, overlap, EOS, boundary mask와 position 정책을 찾는다. configured mixture를 six-stage counter로 내리고 loss denominator까지 연결한다. curriculum controller의 observation, score, update, publish, queue delay와 durable state를 적는다.

그 뒤에만 throughput을 본다. utilization 상승이 unique coverage 손실, objective 변경, source share 변화, longer token batch로 산 것인지 확인한다. uninterrupted/resumed, worker 수, world size를 바꿔 요구 계약의 invariants를 검사한다. 마지막으로 논문 결과, 공개 구현, 현재 recipe의 동일점과 차이를 표로 남긴다.

이 순서를 따르면 “packing을 켰더니 빨라졌다”, “curriculum을 쓰니 좋아졌다” 같은 결과를 재현 가능한 기술 주장으로 바꿀 수 있다. 어떤 token이 왜 선택됐고 어느 context와 weight로 gradient에 들어갔으며 장애 뒤 어떻게 이어졌는지를 설명할 수 있기 때문이다.

최종 승인자는 빠른 평균값보다 경계 사례를 먼저 확인한다. 길이 0에 가까운 sample, 정확히 T인 document, T보다 하나 긴 document, all-ignore label, source exhaustion, 같은 remainder의 allocation, empty rank, selector probability floor, checkpoint 직전 가득 찬 buffer를 포함한다. 정상 경로만 통과한 test는 운영 계약의 절반만 증명한다. 각 경계 사례가 동일한 lineage와 counter 체계를 사용하고 실패 이유를 숨기지 않을 때 6장은 닫힌다.

**고정 근거 좌표.** OLMo-core의 VSL bucket 생성은 [고정 소스 `numpy_dataset.py:1709-1846`](https://github.com/allenai/OLMo-core/blob/b7e9671d7ea48af94838c4f124703c3ae36f0c70/src/olmo_core/data/numpy_dataset.py#L1709-L1846), Grow-P2는 [`numpy_dataset.py:1850-1872`](https://github.com/allenai/OLMo-core/blob/b7e9671d7ea48af94838c4f124703c3ae36f0c70/src/olmo_core/data/numpy_dataset.py#L1850-L1872)에 고정한다.

linear 변형은 [`numpy_dataset.py:1876-1898`](https://github.com/allenai/OLMo-core/blob/b7e9671d7ea48af94838c4f124703c3ae36f0c70/src/olmo_core/data/numpy_dataset.py#L1876-L1898), curriculum identity를 검사하는 resume 경계는 [`data_loader.py:995-1018`](https://github.com/allenai/OLMo-core/blob/b7e9671d7ea48af94838c4f124703c3ae36f0c70/src/olmo_core/data/data_loader.py#L995-L1018)이다.

RegMix의 최종 1B mixture 예시는 [고정 config](https://github.com/sail-sg/regmix/blob/dd9d1c3b2d7c1756b1a90f0ad7603068e9856cc6/mixture_config/config_1b/regmix.yaml#L1-L21), 비교용 DoReMi weight는 [같은 revision의 config](https://github.com/sail-sg/regmix/blob/dd9d1c3b2d7c1756b1a90f0ad7603068e9856cc6/mixture_config/config_1b/doremi.yaml#L1-L21)다. 이 좌표가 증명하는 공개 동작과 production 전체 atomic resume를 구분한다.

누락 없이 전달한다.

**이 장이 넘기는 것.** `PackedSampleID`, segment map, realized mixture counters, selector/sampler state를 7장과 17장에 넘긴다.

**다음 장에서 깨질 수 있는 것.** padding과 segment mask가 position·normalization·attention layout에 맞지 않을 수 있다.

**검증 체크포인트.** emitted token 합, supervised token 합, segment 역색인, resume 직후 다음 PackedSampleID와 mixture counter를 비교한다.

## 6.8 mixture 최적화를 추정과 의사결정으로 분리한다

**설정 확률은 목표일 뿐이다.** source d의 configured probability를 `p_d`, draw t의 source indicator를 `I_{t,d}`라 하면 draw 수 기준 realized share는 `\hat p_d=N^{-1}Σ_t I_{t,d}`다. 그러나 documents 길이와 supervised ratio가 다르면 token 및 loss contribution share는 다르다.

source별 input tokens `T_d`, valid targets `V_d`, loss weight sum `W_d`를 별도로 누적한다. optimizer가 실제 본 목적함수 질량은 대개 `W_d/ΣW`에 가깝다. document share가 20%라도 긴 documents와 높은 supervised ratio로 gradient mass가 더 클 수 있다.

sampling with replacement에서는 source가 고갈되지 않지만 작은 source가 여러 epochs 반복된다. without replacement에서는 exhaustion 뒤 renormalization 또는 stop policy가 필요하다. `max_repetition_ratio`, `max_source_ratio` 같은 제한은 configured p를 그대로 실현하지 못하게 한다. planned와 feasible, realized distributions를 나눈다.

finite window 편차는 multinomial variance를 기준점으로 볼 수 있으나 data-loader buffering과 length bucketing이 draws를 독립적으로 만들지 않을 수 있다. 단일 confidence band로 이상을 단정하지 않는다. source selection, emitted samples, packed tokens, valid targets의 six-stage counters를 실제 pipeline에서 비교한다.

**OLMo-core source mixture를 고정한다.** 로컬 commit `b7e9671d7ea48af94838c4f124703c3ae36f0c70`의 `src/olmo_core/data/source_mixture.py:32` 부근 source config는 target ratio와 repetition/fraction 제한을 읽는 출발점이다. `:205` 이후 fractionalized mixture container와 `:241` 이후 builder config는 requested dataset size와 constraints를 조합한다.

이 source가 만들어 낸 mixture artifact와 runtime sampler를 분리한다. offline에서 source indices를 구성하는 것과 매 step adaptive draw를 하는 것은 상태가 다르다. render table은 계획 결과의 설명이지 실제 training consumption 증거가 아니다. loader counters와 이어야 한다.

### 6.8.1 temperature가 반복률을 바꾸는 방식

원 source mass `q_d`에 temperature exponent α를 적용해 `p_d=q_d^α/Σ_jq_j^α`처럼 만들 수 있다. `α<1`이면 작은 source를 상대적으로 올리고 `α>1`이면 큰 source를 더 강조한다. library마다 temperature를 `1/T`로 쓰므로 config 이름보다 식을 확인한다.

source가 가진 unique tokens를 `U_d`, 전체 planned draws를 N이라 하면 expected sampled tokens `Np_d`와 `U_d`의 비율이 반복 exposure의 거친 기준이다. documents 길이와 replacement policy를 고려해야 한다. 희소 language를 올렸다는 보고에는 unique coverage와 duplicates, expected epochs를 함께 둔다.

probability floor는 tiny sources가 완전히 사라지는 것을 막지만 sources 수가 많으면 floor 질량 합이 커진다. cap과 floor를 적용한 뒤 다시 normalize하는 순서를 명시한다. source 추가가 기존 sources probabilities를 바꾼다는 점도 RunRevision에 남긴다.

distributed sampler가 rank마다 독립 categorical draws를 하면 global share는 맞을 수 있어도 duplicate collisions와 source-local cursor가 달라진다. global draw stream을 partition하는지 rank-keyed counter RNG를 쓰는지 확인한다. world-size 변경의 exact resume 조건을 정의한다.

### 6.8.2 RegMix의 proxy와 regression 선택

**고정 repository를 읽는다.** 로컬 RegMix snapshot은 commit `dd9d1c3b2d7c1756b1a90f0ad7603068e9856cc6`다. `mixture_config/synthesize_mixture.py:116` `generate_weights_dirichlet`는 mixture 후보를 생성하고, `:253` `sort_and_deduplicate`는 가까운 후보를 정리하는 좌표다. `:39`의 token distribution과 `:86` train group 생성도 함께 읽는다.

`regression_fitting/collect_loss_data.py:29`는 W&B runs의 loss data를 수집하는 경계이고 `collect_mixture_data.py:7`, `:27`은 configs에서 mixture 정보를 모은다. notebook의 회귀 분석은 code/data artifact와 정확히 연결한다. notebook output만 복사해 production rule로 만들지 않는다.

RegMix의 핵심은 여러 candidate mixtures로 작은 proxy models/runs를 학습하고 domain losses와 mixture weights 관계를 회귀해 larger run의 mixture를 선택하는 접근이다. proxy model size, token budget, seeds, domains와 target evaluation을 고정한다. 회귀가 interpolation인지 extrapolation인지 본다.

candidate Dirichlet prior가 탐색 공간을 결정한다. 작은 probability 영역과 simplex corners의 coverage, dedup threshold가 회귀 식별성에 영향을 준다. proxy failures나 missing W&B rows가 특정 mixtures에 편향되어 있지 않은지 본다.

최종 config `mixture_config/config_1b/regmix.yaml`과 비교용 `doremi.yaml`은 weights artifact다. 이 weights가 실제 loader에서 document/token/valid-target 질량으로 어떻게 실현되는지 별도 evidence가 필요하다. config 존재와 training success를 합치지 않는다.

### 6.8.3 DoReMi류 reweighting 신호

domain weights를 reference model 대비 excess loss 또는 robust objective로 갱신하는 방법에서는 score 산출, exponentiated update, normalization과 smoothing이 상태기계를 이룬다. exact paper equation과 공개 implementation revision을 고정한다. 이름만으로 모든 adaptive mixture를 같은 식으로 설명하지 않는다.

domain loss는 token denominator와 batch composition에 의존한다. source마다 tokenizer fertility와 sequence length가 다르면 raw mean 비교가 bias를 가질 수 있다. 동일 metric unit과 validation streams를 설계한다. noisy 작은 domain의 score variance를 smoothing·floor가 어떻게 다루는지 본다.

online controller라면 observation UpdateID, weight publish ID, loader queue에 반영되는 지연을 기록한다. weight를 계산한 model state와 실제 해당 weight로 학습된 samples 사이 lag가 있다. dashboard에서 동시 시각만 보고 인과를 붙이지 않는다.

distributed ranks가 서로 다른 weights snapshot을 읽으면 global objective가 갈라진다. immutable WeightRevision과 barrier/atomic publish를 사용한다. checkpoint에는 current weights, optimizer/controller state, next publish schedule과 pending observations를 포함한다.

**packing을 bin-packing 문제와 학습 함수로 동시에 본다.**

길이 capacity T에 document segments를 넣는 것은 bin packing과 닮았지만 순서, EOS, truncation, cross-document attention과 labels 때문에 단순 낭비 최소화가 아니다. utilization `used/T`가 같아도 semantic boundaries가 다르면 model function이 다르다.

first-fit, best-fit, sort-by-length, online buffer 방식은 packing efficiency와 ordering bias, state 크기를 바꾼다. global sort는 length curriculum을 만들 수 있고 streaming buffer는 제한된 horizon의 근사다. algorithm name과 exact tie-break, buffer size, seed를 고정한다.

OLMo-core commit의 `src/olmo_core/data/utils.py:825` `_pack_document`, `:857` `pack_documents`, `:885` `pack_documents_into_instances`는 document length를 instances에 배치하는 함수 경계다. `numpy_dataset.py:1263` source별 packing과 `:1301` 전체 packing도 artifact construction을 읽는 좌표다.

packer output에는 child document/segment ID, source offsets, packed start/end, EOS insertion, truncation, attention segment, position, label-valid bitmap을 둔다. bin utilization만 저장하면 objective equivalence와 deletion reverse lookup을 검증할 수 없다.

tail handling은 drop, pad, carry, wrap, duplicate fill로 나뉠 수 있다. 각각 unique token coverage와 resume state를 바꾼다. `drop_last` 한 option이 loader batch와 document pack tail 중 어느 층인지 구분한다.

**cross-document 경계의 세 mask를 독립 검증한다.**

attention boundary는 token이 다른 document의 context를 보는지 결정한다. label boundary는 이전 document 마지막 위치가 다음 document 첫 token을 예측하는지 결정한다. position boundary는 RoPE position을 reset하는지 결정한다. 하나를 다른 둘의 proxy로 쓰지 않는다.

EOS를 문서 끝에 삽입하고 cross-document attention을 허용하는 continuous packing은 model이 EOS 뒤 새 문서가 온다는 패턴을 학습한다. block-diagonal attention은 documents를 독립 sequence처럼 만들 수 있지만 kernel metadata와 position IDs가 필요하다. 어느 정책이 training/serving과 맞는지 명시한다.

boundary fixture는 A와 B 두 짧은 documents를 pack한다. B의 content를 바꿨을 때 A positions의 output이 block-diagonal 정책에서 불변인지 본다. A의 마지막 target이 B 첫 token으로 향하지 않는지 label 표를 본다. B position reset을 cos/sin slice로 확인한다.

varlen kernel의 cu-seqlens와 loss flatten indices가 같은 token order를 쓰는지 본다. padded dense reference와 packed path의 logits/gradients를 비교한다. model별 attention mask API가 boolean/additive와 shape convention을 다르게 쓸 수 있어 8장 source로 연결한다.

**seed보다 넓은 sampler 상태**

exact resume state에는 epoch, permutation/cursor, RNG/counter, source cursors, replacement counts, shuffle buffer contents/order, open pack bins, partial batch, worker assignments와 prefetch가 포함될 수 있다. 구현이 update boundary에서 일부 state를 비우는지 확인한다.

OLMo-core data loader commit의 `data_loader.py:208`, `:441`, `:749` 부근 `state_dict`/`load_state_dict`는 loader 종류별 durable fields를 비교할 좌표다. VSL loader는 `:986` state와 `:995` load 경계에서 curriculum identity도 확인한다. 실제 state dict keys와 validation을 source에서 추출한다.

worker 수 변경이 exact draw order를 바꾸는지 test한다. per-worker RNG만 저장하면 new worker topology에서 mapping이 달라질 수 있다. sample-keyed transforms와 global counter-based RNG는 elastic reproducibility에 유리할 수 있지만 actual implementation evidence를 요구한다.

prefetch queue를 checkpoint하지 않으면 restart가 already drawn but unconsumed samples를 반복하거나 건너뛸 수 있다. loader가 committed consumption cursor를 optimizer UpdateID와 묶는지 본다. batch yield와 optimizer commit 사이 crash semantics를 정의한다.

**VSL curriculum을 길이별 compute와 sample ordering으로 읽는다.**

variable sequence length curriculum은 shorter sequences를 먼저/더 자주 사용해 early compute를 줄이거나 length distribution을 단계적으로 바꿀 수 있다. sequence length T에서 standard attention 비용이 대략 T²에 비례하지만 tokens/batch와 model FLOPs, kernel efficiency도 함께 변한다.

OLMo-core `numpy_dataset.py:1620` `VSLCurriculum`, `:1672` natural curriculum과 기존에 고정한 `:1709-1846` bucket 생성, `:1850-1872` Grow-P2, `:1876-1898` linear 변형을 같은 commit에서 읽는다. equations와 bucket counts, step mapping을 source에서 표로 만든다.

`data_loader.py:784` 이후는 VSL loader 설명과 construction, `:817` batches per bucket, `:911` batch indices 선택 경계를 보여 준다. `shuffle=False`에서 curriculum이 무시되는 branch `:799`도 옵션이 존재하지만 효과가 없는 반례다.

length curriculum은 같은 document의 truncated prefix를 반복하는지 length bucket별 distinct examples를 쓰는지에 따라 data exposure가 다르다. short phase에서 뒤쪽 tokens가 보이지 않는 편향을 측정한다. position distribution과 domain/document length correlation을 본다.

resume에서 curriculum string/identity가 다르면 source가 load를 거부하는 좌표는 semantic safety gate다. string equality가 모든 hidden state equivalence를 보장하는지는 별도 확인한다. bucket cursor와 current phase, batches consumed를 비교한다.

**compute 절감과 품질 효과를 분리한다**

baseline과 curriculum을 같은 optimizer updates로 비교하면 tokens와 FLOPs가 다를 수 있다. 같은 wall-clock, total tokens, estimated FLOPs, full-length token exposure 가운데 비교 축을 명시한다. early short sequence가 batch size를 늘리면 gradient noise도 달라진다.

quality는 final aggregate만 아니라 length/domain buckets와 short-context regression을 본다. curriculum이 긴 document tail이나 low-resource source를 덜 보여 주면 특정 tasks가 손해를 볼 수 있다. actual consumed spans를 4장 lineage로 역집계한다.

optimizer/scheduler time이 update 기준이면 variable tokens per update에서 warmup token budget이 달라진다. token-based schedule 또는 explicit accounting을 고려한다. 2장의 UpdateID와 ValidTargetCount에 curriculum phase를 연결한다.

ablation은 order만 바꾸고 multiset을 같게 한 경우, length와 batch/compute도 바꾼 경우를 나눈다. 논문의 speedup을 현재 hardware/kernel과 동일하다고 가정하지 않는다. source implementation과 actual measurement를 구분한다.

**분산 재현을 global draw ledger로 검증한다.**

logical global sequence `DrawID→SourceID→SampleID→PackID→UpdateID`를 둔다. ranks/workers는 이 stream의 partitions를 소유한다. actual implementation이 global stream을 materialize하지 않아도 audit fixture에서 대응을 복원한다.

world size 1의 concatenated reference와 world size 2/4의 global sample multiset, order policy, packed contributions를 비교한다. exact order equivalence를 지원하지 않으면 distribution/coverage contract를 명시한다. 두 기준을 섞지 않는다.

empty rank, source exhaustion, corrupt sample retry, unequal pack counts에서 모든 ranks가 동일 optimizer update count와 collective order를 유지하는지 본다. local loader 예외가 다른 ranks를 hang시키지 않도록 coordinated failure를 둔다.

elastic world-size resume은 source cursor와 sampler partition, open bins를 new topology로 remap해야 한다. 지원하지 않으면 checkpoint load에서 거부한다. “weights load 가능”과 “data trajectory resume 가능”을 분리한다.

global ledger의 count는 configured probability가 아니라 realized objective를 증명한다. source별 draws, unique samples, repetitions, input/valid tokens, loss weights를 UpdateID window로 집계한다. dashboard와 offline exact report를 연결한다.

**관측과 장애 playbook을 질량 보존식에서 시작한다.**

loss 변화가 mixture 전환과 함께 나타나면 source weights snapshot, realized valid-target share, average length/supervised ratio를 본다. configured p만 보지 않는다. score controller lag와 loader prefetch로 전환이 늦게 나타날 수 있다.

throughput 상승 시 padding waste, packed utilization, unique coverage, attention policy가 그대로인지 본다. document overlap이나 tail drop으로 빨라진 경우를 kernel 개선과 구분한다. tokens/sec의 token 정의도 input/valid로 나눈다.

resume 뒤 loss가 튀면 next SampleIDs, open bins, shuffle buffer, curriculum phase, source cursors를 비교한다. model/optimizer state가 같아도 data state가 다를 수 있다. first mismatched DrawID를 찾는다.

source 하나가 사라지면 fail, renormalize, fallback 중 정책을 실행한다. fallback이 objective를 바꾸므로 new WeightRevision과 alert를 만든다. stale cache로 조용히 old data를 쓰지 않는다. rights/deletion manifest와 loader gate를 연결한다.

metrics는 configured/realized source shares, pack utilization, segments/pack, boundary targets, truncation/tail drop, unique/repeat, buffer age, phase와 state heartbeat를 포함한다. source names 수가 많으면 stable family labels와 offline details로 cardinality를 제한한다.

**planned mass와 realized mass의 중간 검산.**

독립 검토자는 OLMo-core와 RegMix commits, exact symbols와 configs를 local checkout에서 다시 확인한다. source가 보여 주는 public behavior와 현재 recipe integration을 구분한다. 논문 equation과 code default가 다르면 둘을 나란히 적는다.

mixture fixture는 three sources with unequal document lengths and supervised ratios를 사용한다. configured draw share, realized documents, input tokens, valid targets, loss weight shares를 손계산한다. source exhaustion과 repetition cap을 주입한다.

packing fixture는 exact-fit, one-over, all-ignore, two-document boundary와 open-bin checkpoint를 포함한다. unpacked objective reference와 packed logits/gradients를 비교한다. deletion reverse lookup이 child spans를 찾는지 본다.

curriculum fixture는 natural, Grow-P2, linear schedule의 bucket sequence와 resume state를 확인한다. shuffle-disabled no-effect branch와 curriculum mismatch load rejection을 test한다. compute와 quality 주장을 실행하지 않았다면 `NOT_RUN`으로 둔다.

마지막으로 world size와 workers를 바꾸고 지원하는 재현 등급을 검증한다. exact next DrawID를 보장하지 않으면 statistical contract를 명확히 한다. 숨은 fallback 없이 validator가 범위를 표현해야 한다.

이 봉인 뒤 7장은 PackedSampleID의 token/document/position map을 받아 embedding·RoPE·norm으로 연결한다. 17장은 source/role별 supervised mass를 SFT loss에 연결한다. 24장은 mixture와 benchmark contamination strata를 evaluation에 연결한다.

**RegMix의 packed dataset을 iterator state로 읽는다.**

RegMix snapshot `dd9d1c...`의 `model_training/lit_gpt/packed_dataset.py:27` `PackedDataset`, `:59` `PackedDatasetBuilder`, `:121` `PackedDatasetIterator`는 mixture config와 실제 binary chunks 소비 사이의 구현 좌표다. builder와 iterator를 같은 객체로 보지 않는다.

builder는 output chunk size, separator token, dtype, vocabulary size와 document-end separator 정책을 가진다. `:106` `add_array`, `:117` remainder write가 tail을 어떻게 처리하는지 읽는다. exact-fit과 one-over arrays, empty remainder를 fixture로 둔다.

iterator는 filenames, number of chunks, block size, seed, shuffle, wrap를 읽는다. `:150` header read와 `:165` chunk loading은 binary format 및 mmap/chunk state를 검토할 좌표다. file ordering과 chunk refill이 sample order를 만든다.

`wrap`이 source exhaustion에서 처음으로 돌아가는지, shuffle buffer가 chunk 단위인지 block 단위인지 source에서 확인한다. world/rank worker partition이 wrapper에서 추가되는지도 call sites를 추적한다. iterator 자체의 seed만으로 distributed global order를 설명하지 않는다.

binary header에는 magic/version, dtype, chunk size 같은 schema evidence가 있어야 한다. corrupt/truncated file과 wrong dtype/vocab을 fail-closed하는지 test한다. memory map open/close lifetime과 worker fork behavior도 운영 stability에 영향을 준다.

이 구현 좌표는 RegMix 최종 weights가 어떤 tokens로 실현되는지 조사하는 출발점이다. 그러나 experiment script의 loader 조합, rank partition, checkpoint integration까지 따로 확인해야 한다. source class가 있다는 사실을 exact resume 증거로 사용하지 않는다.

**불확실성을 의사결정에 남긴다**

proxy runs의 domain losses에는 initialization, data ordering, optimizer와 measurement noise가 있다. regression prediction point estimate만으로 최종 mixture를 고르지 않고 residuals, cross-validation, seed variation과 candidate coverage를 본다. simplex boundary에서 불확실성이 커질 수 있다.

domain metrics가 많고 proxy runs가 적으면 회귀가 underdetermined하거나 regularization에 민감하다. features가 mixture weights 합 제약으로 공선성을 갖는다. 어떤 transform과 model class를 사용했는지 notebook/source에서 확인한다. notebook visualization을 독립 근거로 과장하지 않는다.

선택된 mixture가 second-best보다 prediction 차이가 uncertainty보다 작은지 본다. 여러 near-optimal candidates를 longer proxy로 재검증할 수 있다. 최종 run에서 planned weights와 actual consumption, validation trajectory를 비교해 proxy transfer를 사후 평가한다.

실패한 proxy runs를 제외할 때 missingness가 random인지 본다. 특정 extreme mixture가 OOM/instability로 빠지면 feasible region 자체의 정보다. 빈 row 처리와 W&B query filters를 `collect_loss_data.py` source로 확인한다.

regression artifact에는 input run IDs, configs, losses, feature transform, fit code commit, coefficients/model, prediction table과 selection reason을 둔다. 최종 mixture YAML만 보존하면 왜 그 weights인지 재현하지 못한다.

**packing 최적화의 계산 비용을 분해한다.**

padding waste는 batch의 allocated token slots와 actual input/valid tokens 차이로 센다. block-diagonal packed attention은 dense T×T 계산을 그대로 할 수도 있고 varlen kernel로 segment별 제곱합 `Σ_iL_i²`에 가까운 계산을 할 수도 있다. packing utilization만으로 attention FLOPs를 추정하지 않는다.

짧은 segments를 하나의 long dense causal sequence로 이어 cross-document attention을 허용하면 compute utilization은 높지만 문서 사이 불필요/의도된 context가 생긴다. block mask를 dense로 materialize하면 memory가 커질 수 있다. kernel support와 semantic policy를 함께 본다.

packer CPU cost, storage locality, decompression, tokenizer, collator-to-GPU transfer도 step pipeline에 들어간다. GPU utilization이 낮으면 pack algorithm보다 I/O/worker skew가 원인일 수 있다. queue depth와 stage latency를 trace한다.

best-fit search를 정교하게 하면 utilization은 좋아져도 CPU와 buffer memory, reorder horizon이 늘어난다. latency와 reproducibility, data freshness를 trade-off한다. online streaming에서 무한 lookahead의 offline optimum을 비교 기준으로 쓰지 않는다.

성능 A/B는 같은 logical SampleIDs와 boundary policy의 parity configuration에서 시작한다. faster packer가 tail documents를 drop하거나 repeats를 늘려 얻은 이득을 허용하지 않는다. first changed artifact가 layout뿐인지 objective까지인지 표시한다.

**운영 승인 조건과 보류 조건.**

source 행에는 configured probability, feasible mass, unique size, repetition/cap, realized draws/input/valid/loss mass가 있다. packing 행에는 capacity, algorithm/tie-break, buffer, utilization, tail, boundaries와 span reverse index가 있다. curriculum 행에는 schedule equation, bucket/state, compute/token exposure가 있다.

state 행에는 RNG/counter, source cursors, permutations, shuffle/open bins, prefetch, current weights/phase와 committed consumption cursor가 있다. distributed 행에는 rank/worker ownership, world-size contract, empty/exhaustion failure와 collective progress가 있다.

source evidence 행에는 OLMo-core `b7e967...`, RegMix `dd9d1c...`, exact functions/configs와 tests가 있다. 논문 equation과 repository behavior, current recipe observation을 별도 columns에 둔다. 실행하지 않은 result를 source reading으로 채우지 않는다.

고장 주입에는 source outage, probability floor conflict, corrupt chunk, tail one-over, open-bin crash, worker count change, curriculum mismatch, delayed adaptive weights가 있다. 각각 expected first detector와 rollback/fallback policy가 있다.

최종 승인은 임의 UpdateID의 valid targets를 source documents와 configured/realized mixture로 역추적하고, checkpoint에서 다음 draws와 packs를 재현하는 두 왕복으로 결정한다. 둘 중 하나가 끊기면 data objective의 계보가 닫히지 않았다.

**독립 검토자가 수행하는 마지막 반증.**

검토자는 configured weights가 같은 두 recipes를 만든다. 하나는 documents 길이가 같은 sources, 다른 하나는 길이가 크게 다른 sources를 쓴다. draw share는 같지만 input·valid-token share가 달라지는지 ledger가 보여야 한다. configured probability만 표시하는 dashboard는 이 반증에 실패한다.

다음으로 packer worker 수와 task completion order를 바꾼다. 지원하는 contract가 logical sample multiset인지 exact order인지 미리 선언한다. deterministic tie-break와 counter RNG를 썼다면 expected invariants를 확인하고, streaming buffer로 order가 달라지면 새 RunID와 실제 realized mass를 기록한다.

checkpoint 직전 open bin에 두 segments를 남기고 중단한다. exact resume을 주장하면 bin contents와 next document cursor가 복원되어 같은 PackID가 나와야 한다. update-boundary-only resume이면 save 시점에 bin을 비우거나 checkpoint를 거부하는 정책이 명시되어야 한다. 조용한 tail drop은 허용하지 않는다.

curriculum phase boundary 바로 전후에 저장한다. load 뒤 동일 bucket과 next batch indices를 확인한다. config string만 같고 consumed bucket counters가 다르면 trajectory가 아니다. schedule option을 바꾼 warm-start는 별 child run으로 만든다.

adaptive mixture에서는 old WeightRevision이 worker prefetch에 남은 상태를 주입한다. publish barrier와 queue drain 정책에 따라 어느 DrawID부터 new weights가 적용되는지 evidence가 있어야 한다. 모든 ranks가 같은 revision을 읽는지 handshake한다.

마지막으로 source shard 하나를 삭제 또는 corrupt한다. loader가 silent renormalization이나 stale cache로 진행하지 않고 정의된 fail/fallback transition을 실행하는지 본다. fallback이면 realized objective 변경을 alert하고 checkpoint/RunRevision에 남긴다.

이 여섯 반증이 예상 detector에서 잡히고 정상 fixture가 다시 통과할 때 6장의 추가 확장을 봉인한다. 판정은 단어 수가 아니라 source에서 gradient contribution까지 이어지는 질량과 state의 보존으로 내린다.

봉인 artifact는 7장에 token/document/position map을, 17장에 supervised contribution weights를, 16장에 rank/worker state contract를 넘긴다. downstream이 다른 schema를 요구하면 adapter가 span, IDs와 counters를 보존하는지 별 fixture로 확인한다.

**mixture 확률을 네 종류의 질량으로 전개한다.**

source 선택 확률 `p_s`는 document draw의 분포일 뿐 gradient 기여를 직접 뜻하지 않는다. source마다 문서 길이, tokenizer fertility, truncation, packing waste와 loss mask 비율이 다르다. 따라서 draw mass, input-token mass, valid-target mass와 weighted-loss mass를 나눠 기록한다. 마지막 항이 실제 목적함수에 가장 가깝다.

Update window `W`에서 source `s`의 input-token mass를 `I_s`, valid target을 `V_s`, sample weight를 곱한 target을 `G_s`라 두자. dashboard는 `p_s`, `I_s/sum I`, `V_s/sum V`, `G_s/sum G`를 나란히 보여야 한다. 같은 `p_s`인데 한 source 문서가 길면 token mass가 커지고, prompt mask가 길면 valid mass가 작아진다.

packing은 source 선택 뒤의 길이 편향을 바꾼다. 긴 문서가 여러 pack으로 잘리고 짧은 문서가 여러 개 합쳐지면 document count와 pack count의 관계가 달라진다. tail handling이 짧은 fragment를 drop하면 source별 손실률도 다를 수 있다. dropped span과 이유를 질량 원장에 넣는다.

loss weight나 curriculum coefficient가 token별로 다르면 valid count만으로도 부족하다. 각 token의 effective coefficient와 objective numerator를 source·domain·length bucket으로 집계한다. label smoothing이나 multi-task auxiliary loss는 별 numerator/count를 가진다. 모든 scalar를 하나의 “tokens”로 합치지 않는다.

분산 집계는 rank-local ratio를 평균하지 않는다. source별 global numerator와 denominator를 all-reduce하거나 durable ledger에서 합친 뒤 ratio를 계산한다. zero-valid rank와 source absent window를 합법 상태로 처리하되 global count 0인 metric은 명시적 undefined로 둔다.

**반증 실험 6-MA.** document 수는 같지만 길이가 10배 다른 두 source, 길이는 같지만 assistant loss mask 비율이 다른 두 source를 만든다. configured draw 50:50에서 네 질량이 어떻게 갈리는지 손으로 계산하고 loader ledger와 맞춘다.

**sampler를 결정적 상태 기계로 기술한다.**

sampler state는 seed 하나가 아니다. dataset revision, global draw counter, source weight revision, source별 shard permutation과 cursor, replacement policy, exhaustion state와 RNG counter를 가진다. 입력은 committed consumption cursor와 control event이고 출력은 다음 DocumentID·span이다. 각 transition이 어떤 state를 바꾸는지 표로 만든다.

counter-based RNG를 쓰면 DrawID에서 random variate를 직접 파생할 수 있어 worker scheduling 변화와 분리하기 쉽다. stateful RNG를 쓰면 호출 횟수와 순서가 state다. rejection sampling, empty source retry나 filter branch가 RNG를 추가로 소비하는지 확인한다. 같은 seed 로그만으로 같은 draw를 주장하지 않는다.

source exhaustion 정책은 stop, replacement, renormalization, fallback과 error로 나뉜다. 조용한 renormalization은 objective를 바꾸므로 WeightRevision과 alert가 필요하다. finite epoch와 infinite streaming이 같은 iterator 이름을 써도 의미는 다르다. `__len__`을 가짜 큰 수로 두는 구현에서 scheduler total step이 잘못 계산되지 않는지 본다.

distributed partition은 global draw 뒤 rank에 나눌 수도 있고 rank가 독립적으로 draw할 수도 있다. 후자는 collision과 실현 mixture variance가 다르다. global unique draw를 요구한다면 stable DrawID와 ownership 함수를 둔다. elastic membership에서 old generation rank가 draw를 계속 commit하지 못하게 fencing한다.

checkpoint는 sampler snapshot과 model update boundary를 원자적으로 연결한다. prefetch로 생성됐지만 아직 loss에 사용되지 않은 DrawID와 committed DrawID를 구분한다. exact resume은 queue를 복원하거나 committed boundary부터 같은 draws를 재생해야 한다.

**상태 전이 시험 6-SM.** source exhaustion, weight publication, worker restart, rank membership 변경과 checkpoint를 각 transition 직전·직후에 주입한다. next 32 DrawID와 source mass가 약속한 resume 등급에 맞아야 한다.

**packer를 열린 bin과 span 소유권으로 읽는다.**

online packer는 capacity `L`의 열린 bin 집합, 다음 span cursor, placement policy와 tie-break를 상태로 가진다. first-fit, best-fit와 length bucket은 같은 spans를 다른 PackID와 position에 배치한다. cross-document attention이 차단돼도 position·kernel shape와 sample ordering이 달라질 수 있다.

각 placed span은 SourceDocumentID, raw·token offsets, transform revision, PackID, position range, boundary flags와 loss mask를 가진다. separator와 EOS가 어느 document에 귀속되는지 정한다. label shift 뒤 첫 token의 target이 이전 document를 가리키지 않도록 boundary-specific labels를 검사한다.

긴 document를 자르는 policy는 stride overlap, hard cut, sentence boundary와 carry state가 있다. overlap은 token exposure를 반복하고 hard cut은 context를 잃는다. target mask가 overlap을 중복 학습하는지 기록한다. 단순히 pack utilization이 높다는 이유로 정책을 승인하지 않는다.

open bin을 checkpoint하지 않으면 resume 뒤 tail spans의 조합이 바뀐다. bin contents와 insertion order, remaining capacity, tie-break state를 저장하거나 update boundary에서 bin을 결정적으로 flush한다. flush가 padding waste와 source mass를 바꾸므로 빈도와 cost를 측정한다.

parallel packer는 worker completion order가 nondeterministic할 수 있다. exact order를 요구하면 stable sequence 번호로 결과를 재정렬한다. multiset 등급만 보장한다면 어떤 downstream 효과가 달라질 수 있는지 밝힌다. 같은 spans라도 attention predecessor와 optimizer batch grouping이 달라진다.

**identity fixture 6-PK.** 길이 3·5·6·2의 네 spans와 capacity 8을 사용해 예상 bins를 손으로 만든다. 동률 tie, oversize span, empty span과 checkpoint 직전 open bin을 추가한다. PackID와 reverse index, masks를 비교한다.

**boundary mask를 attention·position·loss로 분리한다.**

packed sequence에서 문서 경계는 세 함수에 영향을 준다. attention mask는 이전 document를 볼 수 있는지 정하고, position policy는 index를 reset할지 이어갈지 정하며, loss mask는 경계를 넘는 target을 학습할지 정한다. 하나의 `document_mask` option이 세 동작을 모두 보장한다고 가정하지 않는다.

block-diagonal attention은 segment ID나 cumulative sequence lengths로 kernel에 전달될 수 있다. dense reference mask와 selected CUDA backend의 의미를 작은 fixture에서 비교한다. backend가 지원하지 않아 dense causal fallback을 타면 cross-document leakage가 생길 수 있다. runtime selected branch를 기록한다.

position reset은 learned absolute position, RoPE와 relative bias에서 효과가 다르다. position ID가 reset돼도 RoPE cache와 attention segment가 올바르게 소비하는지 본다. serving prefill과 training packing의 position policy를 같은 이름으로 섞지 않는다.

loss boundary는 shifted labels에서 확인한다. document A의 마지막 input이 document B 첫 token을 target으로 갖지 않아야 하는 정책이면 해당 label을 ignore하거나 EOS target으로 만든다. collator 전 mask와 shift 후 final valid bitmap을 모두 저장한다. count가 예상한 합과 맞아야 한다.

cross-document context를 의도적으로 허용하는 recipe도 가능하지만 그것은 compute 최적화가 아니라 objective 선택이다. document order와 separator, privacy·contamination 영향을 평가한다. 무관 문서의 연결이 rare factual association을 만들 수 있다.

**반증 실험 6-BD.** 각 document가 고유 token alphabet을 쓰게 하고, 경계 앞뒤 attention probability와 target ID를 검사한다. attention만 차단, loss만 차단, position만 reset한 세 잘못된 변형을 validator가 구분해야 한다.

**curriculum controller를 계획과 실현으로 나눈다.**

curriculum config는 phase별 source·length·quality weight와 transition 조건을 정의한다. 실제 loader는 source exhaustion, filter yield, packing과 worker lag 때문에 다른 분포를 소비할 수 있다. planned weight revision과 realized mass를 UpdateID window별로 비교한다.

step 기반 transition은 microbatch나 optimizer effect 중 무엇을 세는지 명시한다. token 기반 transition은 valid·input·weighted token 가운데 어느 count인지 정한다. AMP skip이나 rollback이 clock을 전진시키는지도 확인한다. controller counter는 checkpoint state다.

adaptive curriculum은 validation loss나 gradient signal을 입력으로 받을 수 있다. 측정 window, smoothing, delay와 distributed aggregation을 저장한다. worker prefetch가 old weights로 만든 draws를 queue에 남길 수 있어 publication barrier와 effective DrawID 경계가 필요하다.

length curriculum은 메모리와 compute를 바꾸므로 batch·accumulation도 함께 바뀔 수 있다. global valid-token batch와 LR schedule이 어떻게 조정되는지 본다. 긴 sequence phase의 품질 향상을 단순히 더 많은 FLOP 결과와 구분한다.

quality curriculum은 detector score가 특정 언어·도메인에 편향됐을 수 있다. threshold별 retention, unique mass와 downstream slice를 본다. score policy 변경은 dataset revision이고 old tokenized cache를 그대로 재사용하지 않는다.

**전환 시험 6-CR.** phase boundary 직전 checkpoint, delayed weight publication, validation metric 누락과 source exhaustion을 주입한다. current phase, next DrawIDs와 realized mass가 예상 transition을 따르거나 명시적으로 fail해야 한다.

**distributed consumer의 exactly-once 경계를 정의한다.**

training data queue가 at-least-once delivery를 제공하면 worker·rank restart에서 sample이 재전달될 수 있다. DrawID·PackedSampleID와 optimizer UpdateID의 commit ledger로 duplicate를 식별한다. dequeue acknowledgment를 optimizer effect 전에 보내면 장애 때 누락되고, 뒤에 보내면 재전달 가능성이 생긴다.

gradient accumulation은 여러 packs를 하나의 effect에 묶는다. accumulation 중간 장애에서 일부 packs만 다시 쓰거나 버리지 않는다. exact contract는 boundary 이전으로 rollback해 같은 ordered packs를 재생하거나 gradient buffer와 cursor를 모두 저장해야 한다. 지원하지 않는 방식을 성공처럼 표시하지 않는다.

rank-local sample skip은 collective 순서와 global objective를 깨뜨릴 수 있다. corrupt sample을 한 rank에서만 drop하지 않고 global policy로 quarantine·replacement를 결정한다. replacement는 새 DrawID와 reason을 갖고 realized mass에 반영된다.

consumer lag metric은 prefetch draw와 committed draw의 거리, queue token·byte와 oldest age를 본다. 평균 queue depth는 한 partition의 starvation을 숨길 수 있다. source·rank bucket의 tail을 보되 ID를 metric label로 노출하지 않는다.

resume 뒤 ledger는 last committed UpdateID, consumed draws와 next uncommitted set을 일관되게 복원한다. model checkpoint와 data commit을 다른 generation에서 섞지 않는다. first two updates의 ordered SampleIDs와 delta를 uninterrupted control과 비교한다.

**fault matrix 6-DC.** worker kill, rank kill, coordinator kill, acknowledgment 유실, duplicate delivery와 partial checkpoint를 각각 주입한다. duplicate·missing 0 또는 선언한 distributional budget을 검증한다.

**data pipeline 성능을 objective 보존 뒤에 최적화한다.**

GPU idle이 높다고 packer부터 바꾸지 않는다. source storage read, decompress, parse, tokenize, filter, pack, collate, host-to-device와 compute의 stage latency·queue를 추적한다. 최초 starvation stage와 backpressure 방향을 찾는다. 한 stage의 처리량 평균보다 tail과 burst를 본다.

worker 수를 늘리면 CPU contention, memory, open files와 nondeterministic completion order가 커질 수 있다. throughput과 함께 ordered stream 또는 multiset invariant, memory peak와 restart behavior를 비교한다. 빠른 설정이 sample을 drop해 얻은 이득을 금지한다.

tokenized cache는 CPU 비용을 줄이지만 tokenizer·template·filter revision을 key에 포함해야 한다. cache hit가 높아도 stale artifact면 objective가 틀린다. canonical raw fixture를 live recompute해 cache value와 비교한다.

memory mapping과 sequential read는 storage locality에 유리하지만 shard partition이 rank마다 같은 파일 영역을 읽게 만들 수 있다. host page cache, network filesystem와 local staging을 구분한다. staging copy의 checksum·generation과 cleanup을 관리한다.

packing lookahead를 늘리면 utilization이 좋아져도 latency, buffer memory와 reorder horizon이 증가한다. source freshness와 exact resume cost가 바뀐다. 실제 training step time의 exposed data wait와 valid token/s를 비교한다.

성능 승인은 같은 logical draw set과 boundary policy에서 시작한다. objective mass, duplicate·missing, first update delta를 통과한 후보만 p50·p99와 resource cost를 비교한다.

## 6.9 거버넌스·오염·멀티모달 제약을 소비 계보에 묶는다

proxy run이 고르는 최적 mixture는 model size, tokenizer, token budget, optimizer와 evaluation distribution에 조건부다. 작은 모델의 domain loss를 큰 모델 품질로 옮길 때 무엇이 유지된다고 가정하는지 적는다. regression score가 낮다는 사실을 보편적 최적성으로 확대하지 않는다.

후보 mixture를 만들 때 simplex 전체를 고르게 덮는지, 특정 baseline 근처만 탐색하는지 본다. source 수가 많고 run 수가 적으면 interaction을 식별하기 어렵다. coefficient 하나를 domain의 본질적 가치로 해석하지 않고, 선택 모델과 sampled region의 국소 추정치로 둔다.

각 proxy run은 configured weights만 아니라 realized valid-token·weighted-loss mass를 feature로 가져야 한다. source exhaustion과 packing 때문에 실현값이 다르면 config를 회귀 입력으로 쓰는 것이 measurement error를 만든다. run 실패나 조기 종료도 selection bias를 낳으므로 missing reason을 기록한다.

evaluation metric은 domain별 sample 수와 uncertainty를 가진다. 여러 metric을 임의 가중합하면 choice가 weight에 민감하다. Pareto 후보와 중요 slice의 hard floor를 함께 본다. regression residual, leave-one-out와 seed 반복으로 안정성을 측정한다.

선정된 mixture와 근접 후보의 예측 차이가 uncertainty보다 작으면 단일 승자를 선언하지 않는다. longer proxy나 중간 규모 run으로 재검증한다. 최종 대형 run에서 realized mass와 validation trajectory를 되돌려 proxy transfer를 평가한다.

모든 실험 artifact에는 input run IDs, source revisions, feature transform, fit code, hyperparameters, coefficients·prediction과 선택 이유를 둔다. notebook 그림이나 최종 YAML만 보존하지 않는다.

### 6.9.1 삭제와 정책 변경을 소비 계보에 전파한다

문서 삭제 요청이나 라이선스 변경이 생기면 raw corpus에서 파일만 지우지 않는다. normalized document, token cache, dedup representative, packed sample, consumed UpdateID, checkpoint와 downstream adapter·synthetic data를 reverse index로 찾는다. 어느 단계까지 실제 영향 제거가 필요한지 정책과 근거를 기록한다.

dedup component의 대표 문서가 삭제되면 다른 member를 자동 승격하는 정책은 dataset content를 바꾼다. component membership과 winner rule을 다시 실행하고 새 DatasetRevision을 만든다. 이전 tokenized shard를 같은 이름으로 덮어쓰지 않는다.

이미 소비된 span은 checkpoint parameter에 영향을 줬을 수 있다. corpus deletion만으로 model 영향이 사라지지 않는다. 재학습, 검증된 unlearning, release 제한이나 평가 주장 축소 중 어떤 조치를 택했는지 descendant 계보에 남긴다.

loader는 tombstone DocumentID를 fail-closed해야 한다. stale local cache나 prefetch queue가 삭제 span을 계속 제공하지 않는지 canary를 둔다. 삭제 revision publication 시 모든 worker가 같은 generation을 확인하고 old queue를 drain·격리한다.

정책 threshold 변경도 삭제와 비슷한 migration이다. 새 quality·safety filter로 제외된 spans와 새로 포함된 spans, source mass·length·language drift를 계산한다. curriculum weight를 그대로 두면 feasible source mass 변화로 실현 objective가 달라질 수 있다.

**삭제 fixture 6-DL.** packed sample 중간 span과 dedup winner를 tombstone하고 reverse lookup, cache invalidation, next DrawID와 descendant report를 검사한다. 삭제된 span이 조용히 fallback source에서 다시 유입되지 않아야 한다.

### 6.9.2 contamination gate를 pack 전후에 둔다

benchmark contamination은 raw exact match뿐 아니라 normalized·tokenized n-gram, answer-bearing span, paraphrase와 synthetic lineage를 포함한다. cheap exact detector로 후보를 만들고 semantic detector와 수동 adjudication으로 마감한다. detector score와 정책 판정을 분리한다.

dedup 전에 scan하면 모든 raw occurrence를 찾기 쉽고, dedup 뒤 scan하면 실제 retained representative를 확인할 수 있다. tokenized·packed 단계에서는 어떤 span이 실제 학습 sequence에 들어갔는지 검증한다. 한 단계의 clean 결과를 다른 단계에 자동 상속하지 않는다.

packing은 benchmark prompt와 answer가 서로 다른 source fragment에서 우연히 인접하게 만들 수 있다. cross-document attention을 허용한다면 조합 contamination도 검토한다. separator와 loss mask가 정보를 차단하는지 boundary fixture를 쓴다.

mixture oversampling은 오염 span의 effective exposure를 늘린다. 존재 여부뿐 아니라 draw·valid-token·weighted-loss mass와 UpdateID를 센다. high-quality source라는 이유로 반복된 benchmark 해설은 작은 raw 비율보다 큰 영향력을 가질 수 있다.

오염 match가 발견되면 detector threshold를 보고 조정해 지우지 않는다. calibration positive·hard negative와 policy version을 고정하고, 영향 checkpoint와 evaluation을 격리한다. contaminated rows를 평가에서 빼 점수만 다시 내는 것은 model artifact를 복구하지 않는다.

회귀 gate에는 Unicode·format 변형, 보기 순서, code comment, 번역·paraphrase와 흔한 boilerplate를 넣는다. false positive와 false negative를 언어·길이·task slice로 보고한다.

### 6.9.3 멀티모달 sample을 budget vector로 pack한다

image patch, audio frame과 video token은 text token과 비용이 다르다. processor가 resize·crop·resample·frame select한 뒤 실현 media length를 기록한다. raw 파일 크기나 설정의 최대 frame 수로 GPU compute를 추정하지 않는다.

multimodal pack은 placeholder token 수와 encoder feature 수가 맞아야 한다. 여러 image가 한 sequence에 있을 때 각 placeholder range와 media asset ID를 연결한다. processor revision과 orientation, sample rate, frame timestamps를 cache key에 넣는다.

audio·video는 시간축 정렬이 supervision 의미를 정한다. segment crop과 transcript·label span이 같은 좌표계인지 확인한다. stochastic crop은 seed와 선택 구간을 저장하고 resume에서 요구한 등급에 맞게 재생한다. evaluation transform은 기본적으로 deterministic이다.

token-budget batcher는 text·vision·audio 비용을 하나의 scalar로 근사할 수 있지만 계수는 selected architecture와 kernel에 조건부다. modality별 memory·time을 관측해 budget model을 calibration한다. 긴 video 하나가 rank tail을 만들면 평균 budget이 정상이어도 step이 느려진다.

packing으로 서로 다른 media sample을 합칠 때 attention boundary와 position, loss mask를 명시한다. vision feature가 다른 document text에 보이지 않게 segment map을 검사한다. serving processor와 training collator의 placeholder policy도 비교한다.

corrupt media를 한 worker에서 skip하면 rank stream과 mixture가 바뀐다. global quarantine policy와 replacement DrawID를 쓰고 error source·revision을 남긴다. decoder library 변경은 DatasetRevision이다.

**대시보드를 원인별 네 층으로 나눈다**

control plane에는 configured source weights, WeightRevision, curriculum phase와 publication lag를 둔다. data plane에는 draw·input·valid·weighted mass, exhaustion과 drop을 둔다. packing plane에는 utilization, open bins, tail·boundary와 duplicate를 둔다. system plane에는 stage latency, queue depth, worker restart와 GPU starvation을 둔다.

한 층의 metric으로 다른 층 원인을 단정하지 않는다. configured weight가 정상인데 realized mass가 틀리면 length·mask·exhaustion을 본다. mass는 정상인데 GPU가 idle하면 storage·tokenizer·packer·copy stage를 본다. utilization이 높아도 cross-document leakage면 실패다.

window는 UpdateID와 DatasetRevision에 맞춘다. wall-clock window에 서로 다른 curriculum phase를 섞지 않는다. rank min·max와 source slice를 보되 DocumentID 같은 고카디널리티 값은 metric label이 아니라 incident artifact에 둔다.

alert에는 expected detector, owner와 자동 조치 한계를 적는다. source outage에서 silent renormalization하지 않고 fail 또는 승인된 fallback revision을 만든다. duplicate 경보에서 cursor를 임의로 앞으로 밀지 않는다. 원인 증거를 보존한 뒤 rollback한다.

장기 drift는 source length, tokenizer fertility, valid-label ratio, dedup yield와 processing error를 추적한다. upstream publisher가 같은 URL의 내용을 바꿔도 content digest와 revision이 잡아야 한다. baseline 범위는 계절·언어 변화가 있으면 재승인한다.

대시보드의 모든 비율은 numerator와 denominator를 재구성할 수 있어야 한다. 빈 source, zero-valid window와 missing telemetry를 0으로 표시하지 않는다. `Unknown`은 PASS가 아니다.

**데이터 objective 변경을 평가 설계와 연결한다.**

mixture나 curriculum 변경의 효과를 aggregate validation loss 하나로 판정하지 않는다. source별 validation, target task, language·length·quality slice와 safety를 나눈다. 훈련 source와 평가 domain 이름이 같아도 데이터 생성 계보가 겹치면 contamination을 확인한다.

비교 run은 consumed valid-token budget, optimizer effect, model·tokenizer와 scheduler를 맞춘다. configured document draws만 같고 valid-token mass가 다르면 compute와 objective가 모두 다르다. 긴 sequence mixture는 FLOP도 달라질 수 있어 token과 wall-clock·FLOP budget을 함께 보고한다.

curriculum은 순서 효과를 가지므로 최종 누적 mass가 같아도 trajectory가 다르다. A→B와 B→A, static mixture 대조군을 두고 phase boundary checkpoint를 비교한다. 단일 seed 승리를 일반화하지 않고 seed·data order variation과 uncertainty를 본다.

source weight를 높였을 때 해당 domain 성능과 다른 domain forgetting, calibration과 style 변화가 함께 움직일 수 있다. Pareto table과 중요 hard floor를 둔다. 한 평균이 rare but critical regression을 가리지 못하게 한다.

packing policy A/B는 같은 span multiset과 loss boundary를 우선 맞춘다. cross-document context, position reset이나 tail drop이 다르면 단순 성능 최적화가 아니라 objective A/B다. model quality와 throughput 결과를 함께 해석한다.

평가 artifact에는 DatasetRevision, WeightRevision, realized mass ledger, CheckpointID, EvalID와 confidence interval을 연결한다. 좋은 점수만 골라 mixture 결정을 설명하지 않고 실패·중단 run과 selection process를 보존한다.

**serialized sample의 공급망을 검증한다**

원격 corpus는 content digest, transport, publisher identity와 license snapshot을 가진다. mutable URL의 최신 내용을 신뢰하지 않고 fetch response와 raw bytes를 immutable staging에 둔다. archive extraction은 path traversal, symlink와 decompression bomb 한계를 검사한다.

parser·tokenizer·media decoder는 비신뢰 입력을 처리한다. credential과 production network가 없는 sandbox, CPU·memory·time limit와 file type allowlist를 사용한다. crash sample과 decoder version을 보존하되 원문 접근 권한을 제한한다.

dataset script나 remote code가 실행되는 loader는 dependency와 code revision을 supply-chain manifest에 넣는다. `trust_remote_code` 같은 boolean만 저장하지 않고 실제 resolved module path와 source digest를 기록한다. native implementation 추가로 selected class가 바뀌면 semantic fixture를 다시 실행한다.

cache poisoning을 막으려면 key에 content digest뿐 아니라 normalizer, tokenizer·template, processor, filter와 schema revision을 넣는다. write는 temporary object와 hash 검증 뒤 atomic publication을 쓴다. 다른 worker가 partial cache를 읽지 못하게 한다.

개인정보·민감 데이터 탐지는 raw와 normalized, packed artifact 전반의 lineage를 유지한다. metric label에 원문·DocumentID를 노출하지 않고 상세 span은 접근 제어된 adjudication queue에 둔다. 삭제가 필요하면 6.61의 descendant 조회를 실행한다.

failure injection에는 corrupt archive, parser hang, stale cache, remote code 변경, license metadata 누락과 unauthorized source를 넣는다. 안전 gate가 training worker와 GPU job 시작 전에 실패해야 한다.

**제3자가 수행하는 종단 재계산.**

검토자는 임의 UpdateID 하나를 고르고 loss numerator에서 시작해 반대로 내려간다. valid target을 PackedSampleID와 position에 매핑하고, child span과 DocumentID, source draw, WeightRevision과 curriculum phase를 찾는다. sample weight를 포함한 source별 contribution 합이 update ledger와 맞아야 한다.

다음에는 CheckpointID에서 정방향으로 진행한다. sampler·source cursor, open bins, prefetch와 committed DrawID를 복원해 다음 두 updates의 ordered packs를 만든다. exact resume 등급이면 ID·mask·mass가 같고 selected parameter delta가 tolerance 안에 있어야 한다.

세 번째 검사는 장애 경로다. source shard를 unavailable로 만들고 worker를 open-bin 상태에서 종료한다. loader가 silent renormalization·tail drop을 하지 않고 정의한 failure 또는 fallback revision을 만든다. 정상 복구 뒤 clean control을 반복한다.

네 번째는 변경 영향이다. tokenizer revision, worker 수, mixture weight 또는 pack capacity 하나를 바꾼다. expected first changed artifact와 unchanged invariants를 사전에 적고 실제 trace와 비교한다. 차이가 더 일찍 나타나면 숨은 state owner를 찾는다.

인수 보고서는 source coordinate, config, runtime selected branch, ledger query, failure artifact와 미실행 범위를 분리한다. source를 읽었다는 사실을 current cluster execution으로 과장하지 않는다.

마지막 통과 조건은 “데이터가 충분히 많다”가 아니다. 어떤 token이 어떤 정책과 상태 전이로 선택·배치되어 얼마의 gradient 질량으로 어느 update에 들어갔는지, 장애와 resume 뒤에도 재구성할 수 있어야 한다.

**option을 상태 변화표로 번역한다.**

`packing=true`는 충분한 설명이 아니다. pack capacity, placement algorithm, lookahead buffer, tie-break, tail·oversize policy, cross-document attention, position reset, separator와 loss boundary가 각각 state와 함수 출력을 바꾼다. option별 최초 변경 artifact를 적는다.

source weight 변경은 sampler threshold와 realized draw를 바꾸지만 이미 prefetch된 queue에는 old WeightRevision이 남을 수 있다. publication barrier, queue drain과 effective DrawID가 필요하다. config echo만으로 모든 worker가 새 값을 적용했다고 보지 않는다.

temperature sampling의 temperature는 raw dataset size를 transformed probability로 바꾼다. 0 또는 작은 source의 처리, probability floor·cap과 normalization 순서를 source에서 확인한다. temperature 변경은 rare source 반복률과 dedup exposure까지 계산한다.

shuffle buffer size는 randomness뿐 아니라 memory, reorder horizon과 checkpoint state를 바꾼다. worker 수·prefetch factor는 처리량과 completion order, cursor gap을 바꾼다. exact ordering을 요구하는지 multiset만 요구하는지 option table에 둔다.

`drop_last`는 마지막 batch 크기만 바꾸는 것이 아니다. epoch별 valid-token mass, scheduler update 수와 source tail을 바꾼다. streaming에서는 epoch 정의 자체가 인위적일 수 있다. partial batch flush 정책과 global denominator를 검사한다.

sequence length와 VSL schedule은 pack capacity, attention compute, batch token budget, position와 curriculum state를 함께 바꾼다. old checkpoint에서 warm-start할 때 position·optimizer state migration과 next bucket을 확인한다.

cache·mmap·local staging option은 model 함수가 아니라 artifact selection과 I/O state를 바꾼다. cache key·digest와 selected path를 기록하고 live recompute fixture로 의미 parity를 확인한다. 성능-only 주장이라면 DrawIDs와 batch tensors가 같아야 한다.

모든 option에는 default owner, allowed range, changed state, expected effect, failure shape, metric, checkpoint field와 rollback이 있다. 이 표가 비어 있으면 tuning을 시작하지 않는다.

**test pyramid를 scalar에서 cluster까지 쌓는다.**

가장 아래에는 probability normalization, temperature transform, valid-token mass와 bin placement의 scalar test가 있다. 손으로 계산 가능한 두 source·네 span fixture를 쓴다. tie, zero mass와 capacity boundary를 포함한다.

그 위에는 sampler state machine과 packer unit test가 있다. next DrawID, exhaustion, open bin, separator·attention·position·loss masks를 검사한다. negative fixture가 expected reason code로 실패해야 한다.

single-process integration은 raw DocumentID에서 PackedSampleID와 loss contribution까지 왕복한다. tokenizer·filter·cache revision을 바꾸어 stale artifact가 거부되는지 본다. duplicate·missing과 mass conservation을 확인한다.

multi-worker test는 scheduling 순서, prefetch, worker kill과 checkpoint resume을 다룬다. 지원하는 exact order 또는 multiset invariant를 명시한다. worker completion timing에 따라 silent tail drop이 생기지 않아야 한다.

multi-rank test는 global draw ownership, unequal valid counts, zero-valid rank와 membership generation을 다룬다. world size 1 reference와 loss·first delta를 비교한다. rank-local ratio 평균 오류를 주입한다.

fault test는 source outage, corrupt shard, stale cache, weight publication 지연, partial checkpoint와 duplicate delivery를 포함한다. fail-fast와 approved fallback을 구분하고 effect 전에 fencing되는지 본다.

마지막 cluster soak는 실제 storage topology와 worker 수에서 queue tail, realized mass, duplicate·missing, restart와 throughput을 장시간 본다. 작은 test의 PASS를 cluster performance 보장으로 과장하지 않고, 아래 단계 실패를 soak로 덮지 않는다.

각 test에는 source revision, fixture digest, environment, tolerance, 실행 상태와 artifact가 있다. 미실행 topology는 `NotExecuted`와 필요한 input·expected invariant를 남긴다. golden output을 새 구현 결과로 자동 갱신하지 않는다.

**실제 repository를 파는 순서를 고정한다.**

먼저 recipe config에서 dataset manifests, source weights, temperature·cap, sequence length, packing, shuffle·worker·prefetch, curriculum과 resume options를 추출한다. default가 schema, dataclass, CLI와 example 중 어디서 결정되는지 찾는다. 출력된 config만 보고 실제 소비 여부를 단정하지 않는다.

다음으로 dataset constructor와 loader factory를 따라간다. raw source가 어떤 class로 열리고 filter·tokenizer·cache·sampler·packer·collator가 어떤 순서로 결합되는지 호출 그래프를 만든다. generic framework와 project-specific override를 구분한다. lazy iterator라면 실제 branch는 첫 `next()`까지 추적한다.

sampler에서는 source choice, RNG, replacement·exhaustion, distributed partition과 state dict symbol을 찾는다. 반환 object에 DocumentID·span identity가 남는지 본다. ID가 일찍 사라지면 packed sample과 UpdateID 역추적을 위해 schema를 보강한다.

packer에서는 open-bin state, placement, tie, oversize·tail과 boundary masks를 읽는다. C++·CUDA extension이나 data service로 내려가면 Python wrapper의 shape assertion과 native entry를 연결한다. 빠른 path와 reference path가 같은 fixture를 소비하도록 한다.

trainer 경계에서는 batch가 어느 UpdateID에 포함되고 loss numerator/count가 어떻게 집계되는지 본다. data cursor 저장 시점과 optimizer commit 시점, prefetch acknowledgment를 연결한다. loader checkpoint와 model checkpoint가 같은 generation인지 확인한다.

tests는 unit·integration·resume·distributed·fault로 분류한다. upstream test가 단일 worker order를 증명해도 elastic resume을 증명하지 않는다. source coordinate마다 확인된 contract와 미검증 범위를 적는다.

마지막으로 작은 synthetic repository fixture를 만든다. source별 고유 token, 길이·mask 차이, dedup cluster, corrupt shard와 canary를 포함한다. configured probability에서 realized loss mass와 next draw, pack reverse index를 손으로 계산한다.

조사 결과는 `DataPathCard`로 남긴다. source revision, selected classes/functions, state fields, option→state→effect, checkpoint owner, metrics, tests와 failure recovery가 열이다. 빈칸을 framework 이름이나 문서 추측으로 채우지 않는다.

새 revision에서는 body fingerprint와 caller diff로 영향 범위를 정한다. sampler만 바뀌면 draw·resume fixture, packer만 바뀌면 layout·mask, tokenizer가 바뀌면 cache부터 loss contribution까지 재검증한다. 전체 golden을 무조건 덮어쓰지 않는다.

문서와 example은 의도를 설명하지만 현재 실행 branch의 증거는 아니다. example YAML이 지정한 option이 최신 schema에서 deprecated·renamed되었는지, CLI merge 뒤 실제 object에 어떤 값이 남는지 본다. 경고만 내고 무시하는 option은 configuration drift로 실패시킨다.

source에서 `state_dict`가 보인다고 모든 iterator state가 저장되는 것도 아니다. nested sampler, worker queue, open bins와 controller가 상위 checkpoint에 실제로 포함되는 caller를 추적한다. save와 load field를 대칭 표로 만들고 default initialization을 표시한다. load 성공 뒤 next draws를 비교한다.

테스트용 source는 크기와 길이를 의도적으로 비대칭으로 만든다. 그래야 document ratio를 token ratio로 오해하거나 local mean을 global mean으로 계산하는 버그가 드러난다. 모든 source가 같은 길이인 fixture만 쓰면 핵심 오류가 가려진다.

장애 fixture의 종료 조건은 exception 발생이 아니다. partial data effect가 optimizer에 들어가기 전에 전 rank가 같은 failure generation을 인식하고, 마지막 durable checkpoint와 data cursor로 복구해야 한다. fallback을 허용하면 새 WeightRevision과 realized objective를 남긴다.

성능 PR을 검토할 때는 old/new path가 같은 DrawIDs와 PackIDs를 생성하는 parity mode를 먼저 실행한다. 더 빠른 path가 reorder·drop·duplicate나 mask 변화를 만든다면 performance-only가 아니다. 의미 차이를 선언하고 품질·복구를 다시 승인한다.

최종 source map은 한 token의 owner를 함수 수준으로 왕복할 수 있어야 한다. 선택 함수, pack placement, collator mask, trainer denominator와 checkpoint field가 끊김 없이 이어질 때 repository 분석이 실제 학습 설명이 된다.

인수자는 마지막으로 동일 source revision을 두 환경에서 연다. worker 수와 storage path만 바꾼 parity run에서 지원 계약이 exact order라면 DrawID·PackID가 같아야 하고, multiset 계약이면 구성 span과 mass가 같아야 한다. 성능 환경 차이 때문에 dataset meaning까지 달라지는 것을 허용하지 않는다.

반대로 tokenizer·filter·mixture처럼 의미를 바꾸는 revision은 새 DatasetRevision을 요구한다. cache namespace, sampler state와 open bins를 이전 revision에서 그대로 읽지 않는다. migration tool이 있다면 old/new raw span, token IDs, masks와 next draws를 작은 fixture로 대조한다.

모든 미검증 칸에는 필요한 hardware·topology보다 먼저 필요한 input과 invariant를 적는다. 실행 자원이 없다는 이유로 예상 결과를 PASS로 채우지 않는다. source coordinate, 재현 명령, timeout, 안전 한계와 artifact 위치가 있으면 후속 검토가 즉시 이어질 수 있다.

이 source map과 질량 원장이 함께 있을 때 독자는 설정의 의도, runtime의 실제 선택, gradient에 들어간 결과와 장애 후 복구를 하나의 사건열로 읽는다. 그보다 적은 증거로 mixture와 packing이 재현 가능하다고 주장하지 않는다.

최종 승인자는 임의의 source 하나와 update 하나를 선택해 configured probability, realized valid mass, packed positions, loss contribution과 다음 checkpoint cursor를 독립적으로 다시 계산한다. 값이 맞지 않으면 평균 dashboard가 정상이어도 봉인을 거부한다. 이 재계산은 문서 작성자가 아닌 검토자가 수행하고 사용한 query와 artifact digest를 남긴다.

**마지막 인계 질의.**

인계자는 `PackedSampleID` 하나를 입력해 child documents, raw spans, source mixture, EOS와 boundaries, position과 label-valid bitmap을 반환하는 query를 실행한다. 합친 valid count가 1장의 loss denominator 기여와 맞아야 한다. 삭제된 source span이 있으면 current release에서는 resolver가 fail-closed한다.

다음으로 `UpdateID`를 입력해 모든 packs와 source별 input/valid/weight mass를 재구성한다. configured weights와 차이는 finite sampling, length, supervised ratio, cap/exhaustion, prefetch lag 가운데 어느 항으로 설명되는지 표시한다. unexplained mass는 0이어야 한다.

마지막으로 `CheckpointID`를 입력해 next DrawID와 WeightRevision, curriculum bucket, sampler cursor, open bins와 prefetch policy를 반환한다. exact resume 등급에서는 다음 packs가 같아야 하고 distributional 등급에서는 허용한 차이와 검증 window가 명시되어야 한다.

세 query가 동일 DatasetRevision과 tokenizer, code commits를 가리키는지 확인한다. 사람이 여러 로그에서 최신값을 짐작해야 하면 durable control plane이 아니다. resolver는 checksum과 schema mismatch에서 오래된 state로 조용히 fallback하지 않는다.

이 최종 질의는 mixture·packing·curriculum을 한 artifact graph로 묶는다. 논문 수식은 선택 규칙을 설명하고 공개 source는 구현 owner를 보여 주며 ledger는 실제 소비를 증명한다. 세 층이 일치할 때 data optimization이 빠르면서도 재현 가능한 학습 입력이 된다.

검토 기록에는 OLMo-core와 RegMix의 commit, 정확한 함수·config 좌표, fixture checksum과 실행 범위를 남긴다. source가 갱신되면 sampler state schema, curriculum identity, packer tail과 mixture constraint의 semantic diff를 먼저 만든다. 새 결과를 golden으로 자동 덮어쓰지 않는다.

운영 중 realized mass가 baseline을 벗어나면 source outage와 exhaustion, length drift, label mask, adaptive weight lag 순으로 조사한다. throughput 저하보다 objective drift를 먼저 배제한다. 복구 뒤 동일 CheckpointID child run에서 다음 draws와 contribution ledger를 다시 비교한다.

이 원칙이 지켜지면 데이터 양이 커지고 workers와 clusters가 바뀌어도 독자는 “어떤 token이 왜 이 update에 들어왔는가”에 답할 수 있다. 그 답이 6장의 최종 완료 조건이다.

최종 서명은 tested world size와 worker 수, exact 또는 distributional resume 등급, 지원하지 않은 streaming·elastic 조합을 함께 적는다. 미실행 셀에는 필요한 input, expected DrawID·mass invariant와 실패 detector를 남긴다. 다음 검토자는 모호한 재조사 없이 같은 source와 artifact에서 즉시 시작한다. 이 투명한 경계가 데이터 파이프라인의 성능 주장과 학습 재현성을 동시에 지킨다.

승인된 질량 원장은 다음 학습 revision의 비교 기준선으로 보존한다.

## 6.10 stable identity로 sample→pack→update를 재현한다

DocumentID에서 추출한 training sample은 SampleID, tokenizer가 만든 token sequence는 TokenizedSampleID, 여러 sequence를 담은 결과는 PackedSampleID를 가진다. SampleID는 source document와 selected span, template/format revision을 묶고 TokenizedSampleID는 tokenizer digest, special-token policy와 truncation decision을 더한다. PackedSampleID는 ordered child TokenizedSampleID, token ranges, boundary policy, target length와 packer revision을 묶는다.

mapping row는 child ordinal, source character/byte span, original token range, packed token range, inserted BOS/EOS/separator, padding, position range, attention segment와 loss-valid bitmap을 가진다. 하나의 source token이 overlap 때문에 여러 sample에 들어가면 duplication relation를 표시한다. truncation으로 사라진 suffix도 dropped span과 reason을 남긴다.

training loader는 packed tensor만 반환하더라도 sidecar ledger에서 각 token을 source로 역질의할 수 있어야 한다. 개인정보 삭제, contamination hit와 loss 이상을 PackedSampleID→TokenizedSampleID→DocumentID로 추적한다. row order나 shard offset만 identity로 쓰면 repack과 compaction 뒤 relation이 깨진다.

보존식은 pack의 non-padding token 수가 child retained token+inserted boundary 합과 같고, label-valid count가 loss bitmap 합과 같으며, source별 packed contribution이 ledger 합과 같다는 것이다. batch collator가 추가 padding을 넣으면 pack padding과 batch padding을 구분한다.

### 6.10.1 attention boundary를 mask 함수로 검증한다

문서를 붙였다고 자동으로 document attention이 차단되는 것은 아니다. causal mask만 쓰면 뒤 문서 token은 앞 문서를 볼 수 있다. 독립 문서 attention을 원하면 block-diagonal causal mask, sequence ID 또는 cumulative sequence length가 kernel에 실제 전달되어야 한다. model forward의 argument와 attention kernel 소비 지점을 source에서 확인한다.

packed sequence의 token별 SegmentID를 두고 attention 허용 조건을 `j≤i`와 `segment_j=segment_i`의 conjunction으로 정의한다. prefix-LM이나 multi-turn objective는 일부 cross-segment relation을 허용할 수 있으므로 단순 equality가 아니라 relation table을 쓴다. flash attention varlen path에서는 `cu_seqlens`, max sequence length와 token permutation이 같은 boundary를 표현해야 한다.

tiny fixture는 서로 다른 두 문서에 고유 token을 넣고 둘째 문서 첫 token의 hidden state가 첫 문서 content 변경에 영향받는지 비교한다. 차단 계약이면 동일해야 한다. mask visualization만 보지 않고 forward와 gradient oracle을 실행한다. padding, empty segment, one-token document와 maximum length 경계도 포함한다.

attention boundary는 position reset과 독립이다. position이 0으로 돌아가도 causal attention이 앞 문서를 볼 수 있고, position이 계속 증가해도 block mask로 차단할 수 있다. 두 option을 별 config와 ledger field로 보존한다.

### 6.10.2 position·loss·특수 토큰 경계를 계산한다

position policy는 pack 전체 0..L-1 연속, document별 reset, modality별 coordinate 또는 model-specific position ID로 나뉜다. rope scaling과 maximum position을 고려해 document reset이 architecture contract와 맞는지 확인한다. position ID dtype, padding value와 left/right padding도 기록한다.

causal LM label은 보통 input token을 한 칸 shift해 다음 token을 예측한다. 문서 A의 EOS 뒤 문서 B의 BOS를 예측하게 할지, boundary crossing label을 mask할지 정책이 필요하다. token별 `label_source_position`과 valid bitmap을 만들면 off-by-one을 검산할 수 있다. 첫 token, EOS, separator, padding와 truncated end의 label policy를 명시한다.

SFT prompt/completion sample에서는 prompt token label을 ignore하고 completion과 선택적 EOS만 valid로 둔다. chat template가 assistant header를 삽입하면 header를 학습할지 mask할지 결정한다. 여러 turn의 assistant response를 모두 학습하는지 마지막 response만 학습하는지도 metadata에 둔다.

loss denominator는 batch element 수나 packed length가 아니라 valid label token의 weight 합이다. rank별 valid count가 다르면 local mean 평균이 틀린다. 각 rank의 loss sum과 valid-weight sum을 전역 reduce해 나눈다. zero-valid pack과 rank를 포함한 fixture로 NaN, silent skip와 gradient scale을 확인한다.

### 6.10.3 fixed-length와 variable-length packing

fixed-length packer는 token stream을 target length L로 자르는 concat 방식과 문서 단위를 유지하며 bin에 넣는 방식으로 나뉜다. concat은 utilization이 높지만 document truncation/continuation, cross-boundary attention와 deletion relation가 까다롭다. bin 방식은 boundary를 보존하지만 tail padding과 fragmentation이 생긴다.

greedy streaming packer는 open buffer, current child list, remaining capacity, source/curriculum generation와 next DrawID를 mutable state로 가진다. sample이 남은 capacity보다 길 때 split, truncate, flush 후 새 pack 또는 oversize 전용 path 중 하나를 선택한다. 이 결정은 sample length와 option에 의해 deterministic해야 한다.

epoch 또는 shard 끝의 tail을 drop, pad, carry-over 또는 cross-source fill할지 명시한다. drop은 짧은 문서와 low-volume source mass를 체계적으로 줄일 수 있고 carry-over는 checkpoint/resume state를 늘린다. tail token, source와 valid count를 ledger에 남겨 realized mass를 정산한다.

packer checkpoint는 open buffer token IDs만이 아니라 child mapping, loss/attention/position state, next source draw와 tokenizer/template revision을 저장한다. resume가 open bin을 버리면 duplicate/skip과 mixture drift가 생긴다. update boundary checkpoint와 arbitrary-time checkpoint의 recovery 등급을 구분한다.

**variable-length packing을 token budget batch로 운영한다.**

variable-length batch는 sequence 수가 아니라 총 tokens, maximum sequence, attention quadratic cost 또는 multimodal patches의 budget으로 묶는다. batch planner는 candidate lengths와 resource model을 입력으로 sequence list를 출력한다. GPU memory와 kernel efficiency를 위해 length bucket을 쓰면 sampling order와 source correlation에 영향을 준다.

length sorting window가 크면 padding은 줄지만 가까운 길이끼리 모여 gradient update마다 domain/source가 치우칠 수 있다. window 내 randomization, source strata와 maximum delay를 둔다. realized update별 source/token mass와 length distribution을 관측한다. throughput gain만으로 objective 변화를 승인하지 않는다.

dynamic batch에서 gradient accumulation은 microbatch valid tokens가 다르다. target tokens per update를 기준으로 accumulation을 끝낼지 fixed microbatch count를 쓸지 명시한다. scheduler step, loss normalization와 learning-rate scaling이 같은 UpdateID boundary를 사용해야 한다.

OOM fallback이 긴 sequence를 drop하거나 batch를 재분할하면 exact consumption이 달라진다. retry는 같은 SampleIDs를 smaller microbatch로 처리하고 final commit에 한 번만 기록한다. unsupported oversize sample은 terminal reason과 source span을 남긴다.

**bin packing 효율을 waste와 selection bias로 동시에 본다.**

bin packing은 item length l_i를 capacity L의 bins에 배치한다. utilization은 `Σl_i/(number_of_bins·L)`이고 waste는 denominator의 나머지다. first-fit, best-fit, first-fit decreasing과 online greedy는 utilization과 order sensitivity가 다르다. offline sort는 미래 sample을 보고 순서를 바꾸므로 streaming/resume 계약과 비교한다.

first-fit decreasing은 높은 utilization을 주지만 length로 전체 buffer를 정렬해 source와 time order를 바꾼다. max lookahead window를 두고 delay, memory와 bias를 측정한다. 동일 length tie의 stable/random order와 seed를 기록한다. deterministic tie가 특정 source ID lexical order를 선호하지 않는지 본다.

효율 보고는 padding ratio만이 아니라 valid loss ratio, inserted boundary ratio, dropped/truncated tokens, open-tail carry와 kernel executed tokens를 포함한다. prompt mask가 큰 SFT pack은 non-padding utilization이 높아도 supervised valid ratio가 낮을 수 있다.

adversarial lengths `[L/2+1]` 반복, 많은 tiny sample, exactly L, L+1과 heavy-tail distribution으로 algorithm을 비교한다. conservation과 maximum wait를 검사한다. packer 변경 전후 동일 draws를 입력해 어떤 sample pair가 같은 pack에 묶였는지 component diff를 만든다.

**packing이 만드는 co-occurrence와 gradient bias를 측정한다.**

document attention을 완전히 차단해도 한 optimizer update에 함께 들어가는 sample 조합은 gradient covariance를 바꾼다. length bucket, source-aware fill와 greedy tail은 특정 domain을 같은 pack/update에 묶는다. update별 source vector, length, valid ratio와 pair co-occurrence matrix를 측정한다.

cross-document attention을 허용하면 co-occurrence는 model input 의미까지 바꾼다. unrelated 문서가 앞 context로 작동하고 boundary token을 학습한다. baseline unpacked run, blocked-attention packed run과 continuous-attention packed run을 작은 model에서 비교해 loss와 retrieval-like leakage를 본다.

short document가 남은 공간을 자주 채우면 document inclusion count는 늘어도 token mass는 작다. document-weighted metric과 token-weighted metric을 함께 보고 source별 effective gradient contribution을 valid tokens와 sample weights로 계산한다.

**sequence·prompt·completion truncation을 별 state로 분리한다.**

일반 pretraining document는 head, tail, random window, sentence-aware chunk와 overlapping sliding window 중 하나로 자른다. 각 chunk는 source span, overlap, truncation side와 selection seed를 가진다. head-only는 문서 서론을 과대표집하고 tail-only는 context를 잃는다. random window는 재현 가능한 RNG와 epoch policy가 필요하다.

SFT에서는 max prompt length와 max completion length를 별도로 둔다. 전체 max length에서 단순 right truncation하면 completion label을 잃을 수 있다. keep-end prompt, keep-start prompt, completion priority와 minimum supervised token 규칙을 명시한다. truncation 뒤 assistant response가 비면 reject하거나 zero-valid로 분류한다.

chat template special token이 truncation 경계에서 잘리면 문법이 깨질 수 있다. turn 단위 제거, system prompt 보존, last user turn과 assistant completion closure를 검사한다. token-level 자르기와 turn-aware 자르기의 retained content/valid token 차이를 audit한다.

prompt/completion boundary는 character substring 검색으로 찾지 않고 formatting 함수가 반환한 span 또는 separate tokenization alignment로 만든다. tokenizer가 prefix context에 따라 token boundary를 바꿀 수 있으므로 prompt tokens+completion tokens 단순 concatenation이 full text tokenization과 같은지 fixture로 확인한다.

**sample·token·loss mass를 구분한다**

source s의 configured weight `w_s`가 source draw 확률이면 한 draw의 기대 token 길이에 따라 token mass가 달라진다. expected valid length μ_s가 있을 때 token share는 대략 `w_s μ_s / Σ_j w_j μ_j`다. prompt mask와 sample weight까지 포함하면 loss mass는 또 다르다. planner는 원하는 target이 어느 mass인지 명시한다.

token-target mixture는 source draw weight를 length estimate의 역으로 조정할 수 있지만 heavy-tail, truncation와 packing 때문에 realized share가 달라진다. online estimator의 window, smoothing와 bounds를 checkpoint한다. estimator feedback이 weight oscillation을 만들지 않게 cooldown을 둔다.

temperature sampling은 raw source mass n_s에 대해 `p_s ∝ n_s^α` 형태를 쓸 수 있다. α=1은 raw 비율, α=0은 source 균등에 가깝다. source 정의가 domain, language 또는 dataset인지에 따라 결과가 다르다. nested hierarchy에서는 각 level normalization 순서를 기록한다.

weight table은 DatasetRevision, eligibility/deletion root, effective UpdateID, source inventory와 normalization digest를 가진다. 합이 1이라는 검사 외에 zero/negative/NaN, empty source, minimum floor, maximum cap와 rounding을 확인한다.

**quota·cap·exhaustion을 sampler state machine으로 만든다.**

quota는 per phase token, sample, unique document 또는 maximum repeat count로 정의한다. quota consumed는 requested가 아니라 committed valid mass 기준인지 명시한다. prefetch되었지만 abort된 sample과 gradient accumulation replay를 일관되게 처리한다.

source가 quota에 도달하거나 exhausted되면 stop, renormalize, replacement recycle 또는 fallback pool 전환 중 하나가 일어난다. transition event는 old/new eligible source set, weights, reason와 DrawID를 가진다. 각 rank가 독립적으로 exhaustion을 판단하면 mixture가 갈라지므로 coordinator generation에 합의한다.

replacement sampling은 작은 source를 반복 노출할 수 있다. DocumentID별 epoch/repeat count와 unique token coverage를 관측한다. cap은 과도한 repetition을 막지만 cap 이후 mass 이동이 다른 source를 과대표집한다. planned phase와 realized exhaustion time을 보고한다.

quota failure fixture는 inventory overestimate, deleted shard, empty document, corrupt loader, rank별 stale count와 transition 중 kill을 포함한다. recovery 뒤 next DrawID와 eligible set이 exact 또는 선언한 distributional 등급에 맞아야 한다.

**curriculum과 length ramp를 UpdateID 함수로 고정한다.**

length ramp는 target sequence length, maximum document span, batch token budget와 attention cost를 시간에 따라 바꾼다. phase boundary를 UpdateID 또는 consumed token으로 정의하고 wall-clock에 의존하지 않는다. sequence length가 늘면 batch size/accumulation, learning rate와 throughput도 영향을 받으므로 13장의 scheduler/scaling contract와 연결한다.

domain curriculum은 source weight vector와 quality/complexity bucket을 단계적으로 바꾼다. easy-to-hard라는 이름 대신 bucket 정의, score artifact, transition 조건과 expected mass를 기록한다. quality scorer가 target benchmark를 사용하지 않았는지 4장의 lineage gate를 거친다.

transition 직전 prefetch와 open bins를 drain, carry 또는 discard할지 명시한다. old phase sample이 new phase UpdateID에 들어가면 realized mass가 지연된다. event ledger에서 phase requested/packed/committed generation을 별도로 둔다.

adaptive transition은 metric source, evaluation lag, smoothing, threshold, hysteresis와 fallback을 controller state로 저장한다. metric outage에서 phase가 자동 전진하지 않게 한다. controller 변경은 resume state와 experiment branch를 새로 만든다.

**4장 corpus와 5장 tokenizer 인계를 불변식으로 받는다.**

4장 release에서 DocumentID, selected text span, source/language/quality bins, deletion root, license/privacy decision와 dedup component를 받는다. loader는 current deletion resolver를 통과한 document만 sample로 만든다. old local cache가 tombstone을 우회하지 않게 release+deletion generation을 cache key에 둔다.

5장 tokenizer 인계는 tokenizer files digest, normalization, special token IDs, chat template, max length와 supported padding/truncation mode를 포함한다. corpus token inventory estimate가 다른 tokenizer revision의 값이면 mixture planning에 사용하지 않는다. upgrade는 TokenizedSampleID와 packed shards를 새로 만든다.

golden boundary fixture는 Unicode combining, long code, multilingual, empty/short document, masked span, deletion target와 duplicate winner change를 포함한다. DocumentID에서 exact token IDs와 spans, pack mapping까지 재현한다. source text가 같아도 template revision이 다르면 새 SampleID다.

**valid-token denominator를 optimizer update까지 추적한다.**

collator는 input IDs, labels, attention/position state와 `loss_weight` 또는 valid bitmap을 반환한다. model loss가 internal mean을 쓰면 분산 전역 denominator와 gradient accumulation 계약이 깨질 수 있다. 가능하면 unreduced token loss 또는 local loss sum과 valid weight sum을 얻는다.

data parallel rank r의 numerator `N_r=Σ_i w_i ℓ_i`, denominator `D=Σ_rΣ_i w_i`를 사용해 global objective `Σ_rN_r/D`를 만든다. DDP gradient averaging factor를 고려해 local backward scalar를 조정한다. world size, uneven valid tokens와 zero-valid rank에서 single-process concatenated oracle와 parameter delta를 비교한다.

sequence mean 후 batch mean, token mean와 source-weighted mean은 서로 다른 objective다. prompt 길이가 긴 SFT에서 sequence mean은 짧은 completion을 과대할 수 있다. objective를 이름이 아니라 reducer equation, mask와 denominator fields로 manifest에 둔다.

logging loss도 training scalar와 같은 numerator/denominator를 사용한다. microbatch mean 평균이나 rank mean 평균을 dashboard에 올리면 checkpoint selection이 잘못될 수 있다. aborted update의 numerator를 committed metric에 포함하지 않는다.

**DrawID와 PackPlanID로 replay한다**

DrawID는 mixture generation, RNG counter와 selected source/sample을 식별한다. PackPlanID는 ordered DrawIDs, truncation, placement와 boundary policy를 식별한다. retry가 동일 DrawID를 새 sample로 바꾸지 않으며 OOM 재분할은 child PackPlanID relation를 만든다.

replay bundle은 dataset/deletion root, tokenizer/template, sampler and packer code/config, RNG state, source inventory와 open-bin checkpoint를 가진다. cold process에서 next N draws와 packs를 생성해 IDs, token tensors, masks와 valid counts를 비교한다. loader cache 없이도 가능해야 한다.

exact replay가 불가능한 streaming source나 elastic world-size에서는 distributional contract를 선언한다. duplicate/skip upper bound, source/token mass window, uniqueness와 contamination/deletion closure를 검증한다. exact라는 이름으로 다른 sample을 허용하지 않는다.

**packing leakage와 boundary failure를 negative fixture로 잡는다.**

문서 A에 unique secret marker, 문서 B에 prediction target을 두고 attention 차단 전후 B hidden/logit이 A 변경에 반응하는지 본다. block attention 계약이면 반응하지 않아야 한다. loss boundary fixture는 A 마지막 token label이 B 첫 token을 target으로 삼지 않는지 확인한다.

prompt leakage는 completion text가 prompt field나 preprocessing metadata에 복사되는 경우, rejected response가 chosen input에 섞이는 경우와 evaluation answer가 pack filler로 들어가는 경우를 포함한다. SampleID lineage와 token spans로 source를 찾는다. text equality detector만으로 template injection을 놓치지 않는다.

position failure는 reset omission, duplicate position, padding position overflow와 varlen cumulative length mismatch를 주입한다. eager reference attention과 fused kernel output/gradient를 비교한다. shape가 맞아도 boundary가 틀린 fixture가 필요하다.

pack mapping row swap, loss bitmap shift, stale deletion cache, duplicated DrawID와 tail buffer loss를 각각 주입한다. conservation, resolver, replay와 one-step parameter oracle가 expected first stage에서 실패해야 한다.

**distributed sampler를 global draw와 rank assignment로 분리한다.**

global sampler가 DrawID 순서와 SampleID를 결정하고 rank assigner가 world size, data-parallel group와 local batch shape에 맞춰 배분한다. 두 단계를 분리하면 world-size 변경에서 동일 global order를 유지하거나 명시적으로 재배치할 수 있다. rank가 각자 source RNG를 뽑으면 worker 수와 timing에 따라 mixture가 달라진다.

assignment ledger는 DrawID, destination DP rank, microbatch, accumulation slot, pack plan과 committed UpdateID를 가진다. padding replica나 duplicated sample은 synthetic flag와 loss weight 0을 가진다. global draws=assigned+pending+terminal rejected 보존식을 검사한다.

strided와 contiguous sharding은 length/source correlation에 다른 영향을 준다. precomputed pack list를 rank마다 stride로 나눌 때 pack ordering이 source block이면 rank별 mixture가 치우친다. global shuffle와 per-rank distribution을 함께 검산한다.

elastic membership은 membership generation과 draw lease를 연결한다. 떠난 rank의 uncommitted draws를 reclaim하고 이미 committed update의 draws는 재할당하지 않는다. stale rank가 늦게 consumption commit을 쓰지 못하도록 generation fence를 둔다.

**RNG를 source draw·document draw·packing tie로 분리한다.**

하나의 global PRNG를 모든 단계가 공유하면 logging sample이나 새로운 augmentation이 이후 mixture를 바꾼다. source selection, within-source document, truncation window, pack tie-break, augmentation와 worker transform에 counter-based 독립 stream을 둔다. key는 run seed, purpose tag, DrawID와 optional epoch다.

counter-based RNG는 worker scheduling과 batch grouping이 달라도 같은 DrawID 결과를 만들기 쉽다. library generator state를 저장하는 방식이면 각 consumer call count와 state schema를 checkpoint한다. Python, NumPy, framework CPU/GPU generator를 혼동하지 않는다.

resume fixture는 checkpoint 직전/직후, open bin, partial prefetch와 accumulation 중단에서 next 128 DrawIDs를 비교한다. diagnostic logging을 켜도 draw가 달라지지 않아야 한다. seed 0, large counter, world-size 변경과 worker restart를 포함한다.

**checkpoint에 leased·prefetched 상태를 포함한다**

sampler state는 next global DrawID, source inventories/cursors, weight/curriculum generation, quota counters, RNG counters, open bins, assigned leases, prefetch queues와 last committed UpdateID를 가진다. 단순 epoch와 batch index는 streaming, variable batch와 filtering을 복원하지 못한다.

checkpoint snapshot 중 workers가 prefetch를 계속하면 state가 섞일 수 있다. coordinator가 cut UpdateID/DrawID를 정하고 그 이전 committed, 이후 pending을 분리한다. open pack과 lease를 drain하거나 immutable snapshot으로 복사한다. checkpoint manifest가 모든 worker fragment를 포함한 뒤 commit한다.

load는 dataset/deletion/tokenizer/code revision compatibility를 확인한다. current deletion 때문에 sample이 invalid하면 exact replay를 거부하거나 approved replacement relation를 만든다. stale local cache에서 과거 sample을 그대로 읽지 않는다.

one-step oracle는 next draws, packs, valid denominator, loss와 parameter delta를 uninterrupted run과 비교한다. distributional resume라도 deleted/contaminated sample 0, duplicate bound와 mass window를 요구한다.

**prefetch를 성능 buffer가 아닌 observable state로 운영한다.**

prefetch queue는 requested DrawID, loading/ready/error 상태, memory bytes, enqueue time와 curriculum/deletion generation을 가진다. queue depth를 늘리면 I/O overlap은 좋아지지만 phase transition과 deletion 반응이 늦어진다. oldest age와 stale-generation count를 관측한다.

worker가 document를 decode/tokenize한 뒤 main process가 죽으면 ready item을 replay할지 폐기할지 checkpoint policy가 필요하다. item은 immutable cache key와 checksum을 가져 재사용 가능성을 판단한다. partial tensor나 missing sidecar는 quarantine한다.

backpressure는 packer open bins, tokenizer workers, storage reader와 trainer consumption rate를 연결한다. unbounded prefetch가 host memory를 채우거나 slow source가 전체 ordered queue를 막지 않게 maximum bytes와 head-of-line policy를 둔다. reorder가 DrawID semantics를 바꾸면 event를 남긴다.

**framework와 collator가 만드는 학습 입력을 고정한다.**

**Hugging Face Datasets mapping과 iterable state를 고정한다.**

map-style Dataset에서는 `shuffle(seed)`, `select`, `shard`, `map`, `filter`와 format transform의 fingerprint, indices mapping와 cache files를 기록한다. `set_transform`처럼 access time에 적용되는 함수도 revision과 randomness contract를 manifest에 넣는다. dataset row index를 stable SampleID로 쓰지 않는다.

IterableDataset에서는 source shard order, worker/rank split, streaming shuffle buffer, seed, epoch와 current cursor가 상태다. `set_epoch`가 seed에 어떻게 결합되는지 source에서 확인한다. skip/take로 resume하면 upstream network나 filter 변화에 따라 같은 sample을 보장하지 못할 수 있다.

batched map/tokenize는 input rows와 output chunks의 one-to-many relation을 sidecar로 출력한다. num_proc와 batch size 변경에도 TokenizedSampleID가 같아야 한다. cache fingerprint가 외부 tokenizer/template artifact를 놓치면 explicit digest로 namespace를 바꾼다.

DataLoader의 sampler, batch_sampler, collate_fn, num_workers, persistent_workers, prefetch_factor와 drop_last를 effective config로 기록한다. worker init seed와 exception propagation을 시험한다. worker crash가 batch skip으로 조용히 바뀌지 않아야 한다.

**TRL collator에서 prompt/completion mask를 함수 경계로 검증한다.**

TRL류 SFT pipeline에서는 formatting function 또는 messages field, chat template 적용, tokenization, truncation와 data collator가 labels를 만드는 call graph를 고정한다. assistant-only 또는 completion-only loss option이 template의 generation mask와 실제 labels에 연결되는지 확인한다.

chosen/rejected preference data는 19장으로 가지만 SFT packing 단계에서도 두 response가 섞이지 않게 separate SampleID를 둔다. prompt tokens 공유 optimization이 mask와 truncation을 바꾸지 않는지 concatenated/non-concatenated fixture로 비교한다.

padding-free 또는 packing option이 position IDs, sequence boundaries와 attention backend 요구를 충족하는지 본다. collator output keys가 model forward에서 실제 소비되는지 trace한다. boundary metadata를 만들고도 kernel이 무시하면 독립 document 계약은 실패다.

golden chat fixture는 system/user/assistant multi-turn, tool message, empty completion, long prompt, multiple EOS와 Unicode를 포함한다. expected token IDs, assistant spans, labels와 valid count를 hand-computed table로 보존한다.

**NeMo data pipeline에서 indexed dataset과 sample mapping을 추적한다.**

NeMo/Megatron 계열 indexed dataset은 binary token storage와 index file의 document boundaries, sizes, offsets와 dtype을 사용한다. builder revision, tokenizer/eod ID, index digest와 document count를 manifest에 둔다. bin과 idx가 다른 generation이면 loader admission에서 거부한다.

sample index, shuffle index와 document index가 sequence를 만드는 과정을 작은 corpus에서 재계산한다. 한 sample이 여러 document를 이어 붙이는 경우 시작 document/offset, end document/offset과 used token range를 ledger로 변환한다. extra token for labels와 sequence length off-by-one을 확인한다.

blendable dataset의 weights, sizes, random seed와 index build algorithm을 고정한다. requested sample count를 채우기 위해 source를 반복하는 방식과 epoch가 source별 exposure를 어떻게 만드는지 검산한다. index cache reuse key에 dataset digest와 sequence length가 포함되는지 본다.

**Megatron data loader를 consumed samples와 ramp-up state로 복원한다.**

Megatron류 training에서 consumed samples, train iterations, global batch size와 ramp-up schedule이 sampler offset을 결정할 수 있다. variable valid tokens와 packing을 쓰면 sample count가 objective mass를 정확히 나타내지 않을 수 있으므로 consumed tokens ledger를 추가한다.

cyclic/single-pass sampler, random seed, data-parallel rank/size와 drop-last semantics를 source 좌표에서 확인한다. data parallel size 변경 resume가 global order를 어떻게 재배치하는지 지원 등급을 명시한다. padding samples의 loss mask가 0인지 검사한다.

sequence parallel이나 context parallel은 data sample 자체보다 tensor ownership을 바꾸지만 attention boundary metadata가 분할 후에도 유지되어야 한다. packed sequence의 segment/position/loss state가 scatter/gather에서 같은 token identity를 유지하는지 15장의 rank map과 연결한다.

checkpoint에는 data state revision, consumed global DrawID high-water, blend/curriculum state와 index artifact digest를 포함한다. trainer iteration만으로 loader를 추정하지 않는다.

**collator를 pure mapping과 batch-local padding으로 분리한다.**

sample transform은 TokenizedSampleID에서 token IDs, labels, positions, segment와 weights를 만드는 deterministic 함수다. batch collator는 여러 transformed samples를 stack/pad하거나 variable-length metadata로 합친다. randomness와 truncation을 collator 안에 숨기면 batch composition에 따라 sample 내용이 달라진다.

left/right padding은 causal positions와 label shift에 영향을 준다. pad token이 EOS와 같더라도 attention/loss mask로 구분한다. batch-local max length padding, multiple-of alignment와 fixed target padding의 executed tokens를 기록한다.

collator fixture는 batch 순서 permutation 전후 각 SampleID의 logical tensors가 같은지 확인한다. padding-only 차이는 packed mapping에서 제외하고 batch sidecar에 둔다. empty/zero-valid sample은 명시적으로 reject하거나 valid denominator 0으로 처리한다.

**fused attention·varlen kernel에 boundary state를 전달한다.**

FlashAttention varlen류 kernel은 concatenated tokens, cumulative lengths, maximum sequence length를 받는다. pack 안 document별 attention isolation을 원하면 cumulative lengths가 batch item이 아니라 document segment 경계를 표현해야 한다. 그렇지 않으면 pack 단위로 서로 attention한다.

kernel마다 position ID, causal flag, sliding window와 attention bias 지원이 다르다. eager mask에서 가능한 arbitrary block relation이 fused path에서는 지원되지 않을 수 있다. unsupported contract를 성능 이유로 조용히 약화하지 않고 admission에서 다른 kernel이나 packing을 선택한다.

cu_seqlens는 0에서 시작해 total tokens로 끝나며 monotonically increase해야 한다. zero-length segment, int overflow, CPU/GPU device와 dtype을 검사한다. token permutation이 있으면 inverse mapping이 labels/positions와 함께 움직인다.

eager/fused differential은 random뿐 아니라 boundary marker fixture에서 output과 gradient를 비교한다. dropout이 있으면 RNG contract를 맞춘다. performance report는 useful valid tokens, executed attention tokens와 metadata build cost를 함께 본다.

## 6.11 소비 장과 framework에 데이터 계약을 인계한다

### 6.11.1 scheduler·SFT·RL 소비자에 계약을 넘긴다

sequence length와 valid-token distribution은 memory, step time와 communication을 결정한다. 13장에는 phase별 max/mean/p99 packed length, batch tokens, valid ratio, modality cost와 expected transition UpdateID를 넘긴다. scheduler가 sample count만 보고 warmup/decay를 계산하지 않게 token clock을 제공한다.

length ramp에서 OOM margin, activation checkpointing, microbatch와 accumulation plan이 함께 바뀔 수 있다. phase transition certificate는 data generation과 training config generation을 묶는다. rank 일부가 old length를 prefetch한 상태에서 collective shape가 달라지지 않게 drain한다.

throughput SLO는 raw tokens/s와 valid loss tokens/s를 분리한다. packing utilization이 좋아져 raw throughput이 높아도 prompt mask나 padding 때문에 useful tokens가 낮을 수 있다. source/domain mass SLO와 함께 본다.

**18장 SFT에 supervised span과 template closure를 인계한다.**

18장은 SampleID별 prompt, completion, messages roles, template revision, truncation spans, assistant loss bitmap과 sample weight를 받는다. completion이 비거나 EOS만 남은 sample은 training success로 세지 않는다. packed mapping에서 어느 assistant turn이 loss에 기여했는지 역질의한다.

LoRA/QLoRA 자체는 data contract를 바꾸지 않지만 memory 제약 때문에 max length, packing과 accumulation이 달라질 수 있다. adapter 실험 비교에서 data DrawIDs와 valid denominator를 고정하거나 차이를 명시한다. model option의 효과를 packing 변경과 혼동하지 않는다.

SFT checkpoint resume는 sampler/packer state와 adapter optimizer state를 같은 UpdateID로 commit한다. next pack, loss mask와 parameter delta parity를 확인한다. template나 tokenizer upgrade는 old checkpoint의 exact data replay를 거부한다.

**20장 online RL에 prompt distribution과 replay boundary를 인계한다.**

online RL prompt sampler는 pretraining mixture와 다른 unit을 가질 수 있지만 DatasetRevision, PromptID, source/policy, weight, curriculum와 deletion relation을 받아야 한다. rollout response는 새 child identity이며 prompt와 섞어 원 corpus로 재표지하지 않는다.

prompt packing을 지원하면 rollout group, generation length와 reward mask가 document boundary와 맞아야 한다. 여러 prompt가 fused generation batch에 있어도 KV cache나 position이 교차하지 않는지 확인한다. prompt truncation이 task instruction/answer key leakage를 만들지 않는다.

replay buffer는 PromptID, PolicyID, response tokens, reward/judge revision, consumed RL UpdateID를 가진다. source prompt 삭제나 contamination 판정이 buffer와 checkpoint에 전파된다. stale rollout을 새 policy generation의 on-policy sample로 세지 않는다.

**sampler·packer failure matrix를 state transition으로 실행한다.**

장애 행렬은 데이터를 뽑는 순간부터 checkpoint를 남길 때까지의 시간 순서로 읽는다. 행에는 source 추출, 문서 적재, tokenization, 잘라내기, pack 배치, 경계 생성, prefetch, rank 할당, collate, 순전파 분모 계산, 갱신 확정과 checkpoint 저장을 차례로 놓는다. 열에는 shard 누락, 삭제 경합, RNG 이탈, DrawID 중복, cache 손상, 열린 bin 유실, mask 이동, rank 손실, OOM 재시도, curriculum 혼합과 오래된 tokenizer를 놓는다.

각 칸은 injection point, first invalid state, detector, retry/fallback, duplicate/skip bound와 recovery oracle를 가진다. loader exception을 empty sample로 바꾸거나 zero-valid batch를 정상 update로 commit하지 않는다. failure 뒤 same DrawID가 다른 source content를 가리키지 않아야 한다.

kill matrix는 tokenization 중, pack flush 전후, prefetch ready 후, accumulation 중, sampler checkpoint와 training checkpoint commit 사이를 포함한다. exact 등급은 next tensor와 update parity, distributional 등급은 명시 mass/uniqueness window를 요구한다.

**leakage audit를 source span에서 parameter update까지 수행한다.**

contamination or deletion target span을 입력하면 affected TokenizedSampleID, overlapping chunks, packs, batches와 committed UpdateID를 찾는다. attention mask 때문에 loss-valid가 아니어도 context로 보인 token은 exposure로 센다. prompt token mask 0은 model이 prompt를 보지 않았다는 뜻이 아니다.

pack filler로 들어간 benchmark answer, rejected response, private metadata와 adjacent document를 찾는 fixture를 둔다. source text가 normalization/truncation으로 변해도 span relation와 keyed digest로 추적한다. duplicate component loser가 winner를 통해 노출되는 관계도 포함한다.

제거 후 새 deletion root를 loader에 적용하고 old cache, open bin, prefetched tensor, packed shard와 replay buffer에서 target이 반환되지 않는지 probe한다. already committed exposure는 affected training branch에 기록한다.

**end-to-end one-update oracle로 data objective를 검증한다.**

작은 corpus에 source별 고유 token, 길이, prompt mask, deletion과 duplicate를 넣는다. 고정 RNG로 DrawIDs를 뽑고 truncation, pack, positions, attention, labels, rank assignment와 collate를 손으로 예상한다. single rank와 distributed path에서 global loss numerator/denominator를 비교한다.

같은 initial model로 unpacked reference, fixed pack, variable pack와 fused-varlen path를 실행한다. attention/loss 계약이 같으면 허용 오차 안에서 parameter delta가 맞아야 한다. 계약이 다른 실험은 expected first divergence를 미리 쓴다.

checkpoint 뒤 cold resume에서 next draws, open bins, prefetch, curriculum, batch tensors와 update delta를 재현한다. world-size 변경은 지원 등급에 맞는 oracle을 쓴다. 새 durable checkpoint의 data ledger까지 readback한다.

인수 bundle은 4장 DatasetRevision, 5장 tokenizer, packer/sampler source, 13장 token clock, 18장 supervised spans, 20장 prompt distribution와 UpdateID를 연결한다. 독립 reviewer가 PackedSampleID와 UpdateID에서 양방향으로 같은 결과를 얻을 때 data objective가 승인된다.

**packing·weight·curriculum의 수학적 경계를 검산한다.**

### 6.11.2 packing의 lower bound와 online regret

총 item length가 S이고 capacity가 L이면 필요한 bin 수는 최소 `ceil(S/L)`이다. largest item과 incompatibility constraint 때문에 이 bound를 달성하지 못할 수 있다. 실제 bin 수와 lower bound 차이, padding tokens와 maximum open-bin count를 기록한다. utilization 하나로 algorithm을 비교하지 않는다.

online packer는 미래 item을 모르므로 offline best-fit decreasing 결과와 차이를 regret-like 지표로 볼 수 있다. 같은 draw stream에서 lookahead 0, 32, 256과 full sort를 비교해 waste, latency, memory, source co-occurrence와 replay state 크기를 측정한다. full sort가 training order를 바꾸는 비용도 포함한다.

**compatibility-constrained packing을 graph coloring으로 이해한다.**

모든 sample을 같은 bin에 넣을 수 있는 것은 아니다. license/privacy class, attention mode, modality, position scheme, loss objective, curriculum generation와 sequence format이 다르면 incompatible할 수 있다. SampleID를 vertex, 함께 pack 가능한 관계를 compatibility predicate로 두고 bin placement 전에 검사한다.

predicate revision과 rejection reason을 ledger에 남긴다. compatibility bin이 너무 세분되면 tail waste가 커지므로 throughput을 위해 boundary 계약을 약화하지 않는다. class별 inventory와 tail을 capacity plan에 반영한다. mixed-class fixture가 collator가 아니라 packer admission에서 실패해야 한다.

**document continuation을 chunk chain으로 복원한다.**

긴 문서를 여러 chunk로 나누면 ParentDocumentID, ChunkID, ordinal, source span, overlap와 predecessor/successor relation를 가진다. continuous attention을 허용할지 각 chunk를 독립 sample로 볼지 정한다. chunk order를 randomize해도 lineage는 유지한다.

overlap token은 unique corpus mass와 exposure mass에 다르게 센다. loss mask로 overlap prefix를 context-only로 둘 수도 있다. continuation marker와 position reset이 model contract에 맞는지 fixture로 검증한다. deletion은 chain 모든 affected span과 packs에 전파된다.

**EOS·BOS·separator 정책을 token budget과 objective에 포함한다.**

문서마다 BOS/EOS를 넣는지 pack 처음/끝에만 넣는지에 따라 token mass와 labels가 달라진다. tokenizer가 EOS와 PAD를 같은 ID로 써도 semantic role을 bitmap으로 구분한다. consecutive EOS, empty document와 already-terminated text의 중복 insertion rule을 둔다.

inserted tokens는 source token이 아니지만 training loss와 denominator에 들어갈 수 있다. source별 mass 보고에서 content와 boundary tokens를 분리한다. separator 변경은 TokenizedSampleID와 pack generation을 바꾸며 old cache를 재사용하지 않는다.

**sample weight를 token weight로 펼치는 규칙을 명시한다.**

sample weight a를 모든 valid token에 그대로 적용하면 긴 completion이 더 큰 총 mass를 가진다. sequence-normalized objective는 각 valid token에 `a/n_valid`를 줄 수 있다. source weight, example confidence와 curriculum weight가 곱해질 때 clipping과 normalization 순서를 기록한다.

weight tensor shape는 scalar, token vector 또는 segment vector일 수 있다. collator와 model reducer가 실제 weight를 소비하는지 trace한다. weighted numerator와 denominator를 rank 전역으로 합치며 negative, NaN와 zero-total weight를 거부한다.

**temperature·quota 조합의 stationary distribution을 시뮬레이션한다.**

temperature weight가 고정이어도 replacement 없는 quota와 exhaustion이 있으면 draw distribution은 시간에 따라 변한다. source inventory, weights, caps와 sample lengths를 넣은 simulator로 phase별 expected sample/token/loss mass를 만든다. analytic 초기 probability만 release 목표로 쓰지 않는다.

실제 ledger와 simulator 차이를 finite randomness, load failure, truncation, pack tail와 sampler defect로 분해한다. seed ensemble로 confidence band를 만들고 rare source minimum coverage를 검사한다. quota 변경은 새 WeightRevision이다.

**domain transition에서 catastrophic order effect를 시험한다.**

domain A 뒤 B를 학습하는 curriculum은 동일 mixture를 섞어 학습하는 것과 다른 parameter trajectory를 만든다. total token mass만 같다고 동등하지 않다. A→B, B→A, interleaved와 gradual transition을 같은 initial model과 token budget에서 비교한다.

transition 전후 domain loss, general retention, gradient cosine와 optimizer state 변화를 본다. packer가 transition 경계를 넘어 old/new domain을 한 pack에 섞는지 확인한다. intended gradual overlap과 stale prefetch를 구분한다.

**저장·streaming·멀티모달 소비의 운영 비용을 닫는다.**

### 6.11.3 shard·cache·streaming 비용

shard size는 request overhead, sequential throughput, shuffle granularity, failure retry와 selective deletion rebuild 사이 tradeoff다. shard manifest는 contained SampleID range/list digest, rows, tokens, bytes, schema와 checksum을 가진다. filename order를 sample identity로 쓰지 않는다.

작은 shard는 metadata와 open-file 비용을 키우고 큰 shard는 한 corrupt row의 blast radius와 tombstone compaction 비용을 키운다. source/language별 clustering은 compression을 높일 수 있지만 streaming order bias를 만든다. shuffled manifest와 ledger로 완화한다.

**cache를 tokenizer·policy·deletion generation에 결박한다.**

token cache key는 DocumentID만으로 부족하다. selected span, text digest, tokenizer/template, truncation, special token와 loss-mask revision을 포함한다. pack cache는 child token IDs, packer/boundary revision과 target length를 포함한다.

deletion root가 바뀔 때 cached payload를 다시 반환하지 않도록 resolver check를 cache hit보다 먼저 하거나 key에 generation을 둔다. negative cache에도 expiry가 필요하다. checksum mismatch와 partial write는 miss가 아니라 corruption event로 처리한다.

**streaming source outage를 availability-aware decision으로 처리한다.**

remote streaming shard가 unavailable일 때 무한 retry, source skip, fallback 또는 pause 중 정책을 정한다. source weight를 유지한 채 다른 source를 뽑으면 realized mixture가 바뀐다. outage generation, affected draws와 renormalization event를 기록한다.

ordered stream의 한 shard가 느려 head-of-line blocking을 만들 수 있다. bounded reorder를 허용하면 maximum displacement와 exact replay 등급을 낮춘다. outage recovery 뒤 missed draws를 보충할지 future probability만 복원할지 명시한다.

**worker-local transform의 determinism을 process 격리로 검증한다.**

tokenizer, parser와 augmentation이 global mutable cache, locale, thread count 또는 unordered map에 의존하면 worker 수에 따라 결과가 달라질 수 있다. 같은 SampleID를 worker 0/1, fresh process와 different batch position에서 실행해 digest를 비교한다.

native tokenizer parallelism과 fork 이후 thread state, random augmentation와 exception handler를 effective environment에 둔다. worker restart가 model/tokenizer artifact를 다른 revision으로 받지 않게 immutable local cache를 검증한다.

**data loader 성능을 stall 원인별로 분해한다.**

step data wait은 storage read, decompression, deserialize, tokenize, pack plan, collate, host-to-device와 synchronization으로 나눈다. queue wait와 compute를 trace에 연결하고 p50/p99를 sample length/source와 함께 본다. GPU utilization 하나로 loader 원인을 추정하지 않는다.

prefetch 증가, more workers, larger shards와 pinned memory 변경은 throughput, host memory, tail, exact resume state와 deletion lag를 함께 평가한다. optimization 전후 same DrawIDs에서 tensor digest와 denominator가 같은지 확인한다.

**modality clock과 preference pair batching**

text tokens, image patches, audio frames와 video segments는 단일 length로 비용을 표현하기 어렵다. sample은 budget vector와 model-specific sequence serialization을 가진다. packer는 GPU memory/compute bound와 modality compatibility를 만족하는 bins를 만든다.

image placeholder token과 patch tensor relation, audio timestamp와 transcript span, modality position/attention mask를 sidecar에 둔다. asset decode failure가 text-only fallback으로 조용히 변하지 않게 modality-valid mask와 reason을 기록한다.

**preference pair batching과 19장 reducer 인계를 준비한다.**

preference pair는 chosen/rejected가 같은 prompt와 PairID에 결박되어야 한다. length bucket이 두 response를 다른 update나 rank로 갈라 objective margin을 잃지 않게 pair를 batch atomic unit으로 둔다. prompt sharing optimization도 separate response mask를 보존한다.

19장에는 PairID, token IDs, prompt/response spans, truncation, valid token counts, sample weight와 reference-cache key를 넘긴다. sum/mean reduction을 collator가 결정하지 않고 reducer가 sufficient statistics에서 선택하게 한다.

## 6.12 장애·복구·release를 하나의 사건 원장으로 닫는다

### 6.12.1 rollout과 contamination 사건을 식별한다

online rollout prompt batch는 PolicyID, PromptID, sampling seed와 GenerationID를 가진다. fused generation에서 variable prompt length, KV cache slots와 position state가 다른 prompt 사이에서 섞이지 않게 mapping을 검증한다.

response가 완료되거나 timeout/truncate된 reason과 token span을 남긴다. RL training pack은 rollout group, old logprob, reward mask와 policy generation을 보존한다. stale response를 새 policy의 on-policy data로 재표지하지 않는다.

**data contamination을 pack context와 label exposure로 등급화한다.**

오염 token이 label-valid이면 direct target exposure, mask 0이지만 attention-visible이면 context exposure, attention 차단 segment이면 co-batch exposure로 구분한다. packed shard 존재만으로 모두 같은 위험으로 세지 않되 policy 삭제는 각 descendant를 처리한다.

BenchmarkItemID에서 matched source span, chunks, packs, attention relation, labels와 UpdateIDs를 반환한다. removal rebuild 뒤 direct/context exposure가 0인지 probe한다. co-batch gradient correlation은 separate sensitivity experiment로 평가한다.

**pack corruption을 structural·semantic checksum으로 검출한다.**

file checksum이 맞아도 mapping sidecar가 다른 pack과 섞이면 semantic corruption이다. payload digest와 함께 ordered child IDs, ranges, segment, positions, labels와 valid count의 structural digest를 둔다. loader가 둘의 generation을 일치시킨다.

range overlap/gap, nonmonotonic positions, attention segment out-of-bounds, label token mismatch와 duplicate DrawID validator를 publish 전에 실행한다. random cold decode로 tensor와 sidecar를 재계산한다.

### 6.12.2 exact resume와 distributional resume

행은 map dataset/streaming, fixed/variable pack, single/elastic world size, open-bin checkpoint, curriculum transition와 deletion change다. 열은 next DrawID, next SampleID, exact tensor, mass distribution, duplicate/skip bound와 first update parity다.

지원하지 않는 칸은 `NOT_RUN`이 아니라 unsupported reason과 fallback을 가진다. distributional resume를 exact 성공률에 포함하지 않는다. framework upgrade마다 support cells를 golden checkpoint로 다시 실행한다.

### 6.12.3 freeze→trace→rebuild→replay

wrong mask, stale tokenizer, deleted sample, source weight drift나 duplicate draw가 발견되면 affected data generation의 새 admission을 freeze한다. active training을 stop/drain할지는 exposure와 위험에 따라 결정하고 UpdateID를 기록한다.

trace는 first invalid SampleID/DrawID에서 source, cache, pack, rank, batch와 checkpoint를 양방향으로 찾는다. 수정은 새 code/config/data root로 child generation을 만들며 packed file을 수동 patch하지 않는다.

rebuild 뒤 incident fixture, random unaffected samples, mixture mass와 cold resume를 실행한다. affected checkpoint branch의 재학습/평가 범위를 명시하고 old artifact를 새 결과로 재표지하지 않는다.

**capacity planning을 valid tokens per second와 tail state로 계산한다.**

source별 read/decode/tokenize rate, pack utilization, valid ratio, batch cost와 step consumption을 이용해 worker, cache, network와 storage를 계획한다. raw rows/s가 높아도 supervised valid tokens/s가 낮을 수 있다.

capacity model은 p99 long sample, giant shard, source outage, pack tail와 phase length ramp를 포함한다. prefetch memory와 checkpoint snapshot size도 계산한다. canary의 평균만으로 full curriculum peak를 승인하지 않는다.

**blind data audit로 구현자의 가정을 반증한다.**

첫 reviewer는 PackedSampleID만 받아 source spans, tokenizer/template, draws, truncation, boundaries, valid denominator와 assigned UpdateID를 재구성한다. 둘째 reviewer는 UpdateID에서 source/token/loss mass와 curriculum, quotas, aborted/replayed draws를 정산한다.

세 번째 reviewer는 CheckpointID에서 next draws, RNG counters, source cursors, open bins, prefetch와 membership generation을 복원해 cold next-update oracle을 실행한다. 네 번째 reviewer는 deletion/contamination span에서 현재 loader와 과거 exposure를 역질의한다.

최종 certificate는 mapping conservation, attention/position/loss boundaries, packing efficiency와 bias, truncation, mixture/curriculum, distributed resume, framework call graph, failure matrix, chapter handoffs와 cold replay를 연결한다. 네 reviewer가 같은 artifact로 동일 결론을 얻을 때 packing에서 실제 update까지의 운영 계약이 닫힌다.

**수치 비교·elastic resume·upgrade를 반증 실험으로 승인한다.**

**동일 trace의 수치 반증 실험**

target length 16에 item lengths `[10, 9, 7, 6, 6, 5, 4, 3, 2]`를 넣고 online first-fit, best-fit, decreasing와 streaming concat을 손으로 계산한다. 각 bin의 ordered SampleIDs, remaining capacity, padding, waiting draws와 flush reason을 표로 남긴다. total length lower bound와 실제 bins 차이를 계산한다.

length만 맞춘 비교 뒤 source를 A/B, loss-valid ratio를 서로 다르게 부여한다. 같은 utilization이어도 update별 source mass와 valid denominator가 달라지는지 본다. decreasing sort가 긴 A를 앞 update에 몰고 tiny B를 tail filler로 만드는 order effect를 확인한다.

open-bin limit를 1, 4, 32로 바꾸어 utilization, maximum sample wait, memory와 resume state를 측정한다. timeout flush를 넣으면 worker 속도와 wall-clock이 pack 결과를 바꿀 수 있다. 재현이 필요하면 timeout 대신 draw-count boundary를 쓰거나 distributional 등급으로 낮춘다.

**truncation 정책을 retained information과 group fairness로 평가한다.**

head/tail/random/turn-aware truncation을 source language, domain, document type, prompt length와 completion length별로 비교한다. retained token 비율뿐 아니라 title, instruction, answer, code closing delimiter, citation와 safety qualifier가 남는지 span role별 recall을 계산한다.

긴 형태를 쓰는 language나 domain이 max length 때문에 더 많이 제거될 수 있다. document acceptance는 같아도 supervised completion valid mass가 줄어든다. cohort별 original/retained length, zero-valid rate와 loss contribution을 보고 threshold를 정한다.

random window는 여러 epoch에서 coverage를 높일 수 있지만 repeat와 overlap을 만든다. DocumentID별 union source-span coverage, token exposure count와 seed 안정성을 측정한다. evaluation answer나 private suffix가 우연히 선택되는 경우는 upstream gate로 차단한다.

**mixture 추정량을 Horvitz-style inclusion과 realized ledger로 검산한다.**

source draw 확률 p_s와 within-source sample 확률 q_i가 알려지면 sample inclusion weight를 해석할 수 있지만 replacement, quota와 rejection 때문에 실제 확률은 state-dependent다. 각 DrawID 시점의 eligible inventory와 normalized weight를 event에 남겨 selected probability를 재구성한다.

loss를 target population으로 reweight할 경우 importance weight clipping, normalization와 variance를 기록한다. 매우 작은 probability의 긴 sample이 gradient를 지배하지 않게 effective sample size와 maximum weight를 본다. reweighting을 하지 않는다면 optimized training distribution을 명시한다.

offline planner probability와 online realized probability를 source, document, token, valid token 네 unit에서 비교한다. deviation의 confidence interval은 repeated draws correlation과 quota transition을 simulation으로 반영한다. 단순 multinomial p-value 하나로 sampler를 승인하지 않는다.

**valid denominator의 DDP scaling을 one-parameter model로 증명한다.**

scalar parameter 하나와 token별 서로 다른 input/target을 가진 model로 analytical gradient를 계산한다. rank 0에 valid token 1개, rank 1에 7개, rank 2에 zero-valid pack을 배치한다. concatenated global objective gradient와 DDP backward 결과가 같은지 확인한다.

DDP가 rank gradient를 평균내면 local numerator를 global denominator에 맞춰 world-size factor로 scale해야 할 수 있다. framework loss reduction과 gradient accumulation/no_sync 동작을 source에서 확인한다. local mean을 backward한 뒤 metric만 global mean으로 고치는 것은 충분하지 않다.

sample weights, token weights, label smoothing와 mixed precision scaler를 차례로 추가한다. overflow로 update가 skip되면 denominator consumption commit도 미루거나 replay policy를 적용한다. logged loss, gradient와 consumed ledger가 같은 UpdateID를 가리킨다.

**elastic world-size와 deletion race**

membership generation g에서 rank별 leased DrawIDs와 committed high-water를 snapshot한다. world size가 바뀌면 committed draws는 고정하고 pending lease를 회수해 새 rank assigner로 배치한다. pack이 rank-local open bin에 있으면 complete pack 단위 이동, serialized bin 이동 또는 discard/replay 중 하나를 정한다.

global batch token target을 유지할 때 local microbatch와 accumulation 수를 재계산한다. curriculum/scheduler token clock과 UpdateID boundary가 같아야 한다. world size 변경 순간 partial accumulation gradient가 있으면 checkpoint로 완전 복원하거나 이전 update boundary로 rollback한다.

elastic fixture는 4→3, 3→5, zero-valid rank, long pack owner loss와 membership change 중 deletion update를 포함한다. duplicate/skip, source mass, pack digest와 first parameter update를 검사한다. unsupported exact path는 allocation 전에 거부한다.

**framework upgrade를 collator output과 next-update parity로 검증한다.**

old version에서 golden dataset, sampler checkpoint와 packed sidecar를 만든다. new version의 loader/collator로 cold restore해 next SampleIDs, token IDs, masks, positions, cumulative lengths, valid denominator와 batch keys를 비교한다. warning 없이 무시된 field가 없는지 model forward trace를 본다.

반대 방향 new writer→old reader 지원 여부도 시험한다. schema가 호환되지 않으면 명확히 실패해야 한다. default padding side, chat template, truncation, `drop_last`, worker seed, shuffle와 packing option의 effective diff를 출력한다.

performance kernel upgrade는 same logical tensor와 eager/fused gradient를 비교한다. tolerance를 dtype과 operation별로 정하고 boundary marker의 cross-segment 영향은 tolerance가 아니라 exact isolation으로 검증한다. upgrade certificate에 source body fingerprint와 fixture를 둔다.

**deletion과 prefetch race를 generation fence로 차단한다.**

worker가 deletion root d에서 sample을 resolve한 뒤 d+1 tombstone이 commit되고 trainer가 tensor를 소비할 수 있다. 허용 policy는 resolve-only, pack-time recheck, batch-time recheck 또는 update-commit recheck 중 freshness boundary를 정한다. high-risk deletion은 active prefetch invalidation signal을 보낸다.

prefetched item과 pack은 resolved deletion generation과 child DocumentIDs를 가진다. trainer admission이 minimum generation보다 오래된 item을 거부한다. open bin에서 target child만 제거하면 positions/masks가 달라지므로 pack 전체를 rebuild하고 새 PackPlanID를 만든다.

race fixture는 tokenization 중, pack committed 후, device transfer 후와 accumulation 중 tombstone을 넣는다. expected stop/replay/commit policy와 exposure ledger를 검증한다. old cache key와 disconnected worker가 target을 되살리지 못해야 한다.

**curriculum 효과를 data order와 model capacity로 분리한다.**

curriculum run과 baseline은 total source/token/loss mass, model init, optimizer와 scheduler를 맞추고 order만 바꾸는 controlled 비교를 둔다. 별도로 curriculum이 quota나 length를 바꾸는 production comparison을 한다. 두 효과를 하나의 품질 delta로 합치지 않는다.

학습 중 source별 held-out loss, gradient norm/cosine, forgetting, calibration와 downstream slice를 UpdateID/token clock에 맞춰 본다. early benefit가 final regression으로 바뀌는지 transition 이후 충분한 window를 둔다. checkpoint selection이 curriculum 평가 set에 과적합되지 않게 independent set을 사용한다.

adaptive controller는 metric lag 때문에 이미 바뀐 model에 오래된 signal을 적용할 수 있다. EvaluationID의 PolicyID/UpdateID와 controller decision time을 기록하고 maximum staleness를 둔다. delayed/reordered metric failure를 주입한다.

**pack-level observability를 cardinality 제한 event로 만든다.**

모든 PackedSampleID를 metric label로 쓰면 cardinality가 폭발한다. aggregate metric은 source, length bin, valid ratio, phase와 failure reason을 사용하고 exact IDs는 sampled trace와 durable ledger에 둔다. sampling probability와 trigger-based capture를 기록한다.

dashboard는 draw availability, tokenizer/cache, pack utilization/boundary, loader queue, batch valid denominator와 committed mass를 층으로 나눈다. throughput 저하가 storage인지 long sample인지 zero-valid인지 first divergence를 찾는다. 평균과 p99를 함께 본다.

alert에는 runbook query, affected generation와 freshness를 포함한다. stale dashboard가 green이어도 ledger high-water가 뒤처지면 publish/continue gate를 닫는다. metric failure가 training objective failure를 숨기지 않게 independent conservation auditor를 둔다.

**security boundary를 serialized sample과 worker 권한에 적용한다.**

dataset row의 pickle 또는 arbitrary object를 신뢰하지 않는다. Arrow/JSON/schema-validated primitive를 사용하고 custom transform code와 model artifact를 서명된 build로 고정한다. worker는 필요한 shard read와 output cache write만 허용하며 credential을 sample metadata에 전달하지 않는다.

malicious oversized length, nested list, invalid UTF/token ID, negative position와 NaN weight를 collator 전에 검증한다. decompression/resource bound를 둔다. exception log에 raw private prompt나 secret가 출력되지 않도록 DocumentID와 bounded reason을 쓴다.

cache와 checkpoint artifact의 checksum, schema와 producer revision을 load 전에 검사한다. remote code option이나 floating dataset script를 production에서 허용할 때 별 review와 pin이 필요하다. supply-chain 변경은 golden replay를 요구한다.

**mixture configuration 변경을 semantic diff로 승인한다.**

두 weight config의 key/value diff뿐 아니라 normalized source probability, expected token/loss mass, quota exhaustion time, curriculum phase와 empty-source fallback을 계산한다. source rename이 실제 dataset root 변경인지 alias인지 lineage로 확인한다.

candidate config를 frozen inventory simulator와 canary sampler에 적용해 DrawID distribution, unique coverage, repetition, pack utilization와 denominator를 비교한다. random variance를 seed ensemble로 분리한다. high-risk source weight 증가에는 privacy/contamination gate evidence를 연결한다.

promotion은 WeightRevision과 effective UpdateID를 원자적으로 활성화한다. active ranks가 digest에 합의하고 stale prefetch 처리 후 first committed mass를 확인한다. rollback은 최신 deletion root와 incident blocklist를 유지한다.

**pack format migration을 dual reader와 round trip으로 시험한다.**

sidecar schema나 tensor serialization이 바뀌면 old pack을 새 reader로 읽고 canonical logical representation을 비교한다. migration은 원 PackPlanID parent와 새 artifact digest를 가진다. source mapping이나 loss bitmap을 추정해 채우지 않는다.

dual reader canary는 representative fixed/variable, SFT, multimodal, zero-valid와 maximum-length packs를 읽는다. byte order, integer width, compressed bitmap와 cumulative length overflow를 검사한다. new writer output의 cold read와 old reader failure message도 지원 정책에 따라 본다.

**update commit을 data consumption의 durable effect로 정의한다.**

batch가 device에 올라가거나 forward를 끝냈다고 consumed로 세지 않는다. optimizer update가 성공하고 scaler overflow skip이 아니며 training checkpoint/event가 UpdateID를 commit할 때 DrawIDs와 valid mass를 durable consumption으로 기록한다. failed attempt는 별 AttemptID다.

gradient accumulation 중 여러 packs가 하나 update에 들어간다. commit record는 all PackPlanIDs, numerator/denominator, source mass, curriculum/weight/deletion generation와 model revision을 가진다. partial event는 selector가 정상 소비로 읽지 않는다.

**sample replay 비용과 retention을 recovery SLO에 포함한다.**

exact replay를 위해 raw documents, token cache, pack sidecar, sampler state와 code를 얼마나 오래 보존할지 정한다. 모든 packed tensor를 영구 보관하는 비용과 재생성 시간, 삭제 의무를 함께 본다. high-value checkpoint window에는 필요한 parent artifact를 보호한다.

RTO는 dataset mount, cache warm, index load, sampler restore, open-bin rebuild, prefetch fill와 first update로 분해한다. RPO는 lost committed DrawID가 아니라 rollback되는 UpdateID와 replay token을 센다. cold rehearsal로 p95를 측정한다.

**failure injection을 synthetic corpus와 live canary로 계층화한다.**

unit fixture는 off-by-one label, wrong EOS, range overlap와 denominator를 잡는다. process fixture는 worker kill, cache corruption와 open-bin checkpoint를 잡는다. distributed fixture는 rank loss, uneven valid tokens, membership change와 mixed curriculum을 잡는다.

live canary는 실제 storage/shard/tokenizer artifact의 representative sample을 읽되 민감 content를 승인 환경 밖으로 복사하지 않는다. deletion and contamination probes, source outage와 slow shard를 포함한다. production root는 변경하지 않고 child canary generation을 사용한다.

**final cold rehearsal을 SFT와 online RL consumer까지 확장한다.**

fresh environment에서 4장 release와 5장 tokenizer를 받아 pretraining pack, SFT supervised pack와 RL prompt batch를 각각 생성한다. consumer가 기대하는 tensors, spans, masks, IDs와 denominator를 검증한다. 같은 SampleID가 objective에 따라 다른 labels를 가질 때 child identity를 분리한다.

13장 token scheduler에는 committed valid token clock, 18장 trainer에는 assistant spans/weights, 20장 rollout에는 prompt/Policy generation을 전달한다. 각 consumer의 first update 또는 first rollout event가 source ledger와 연결되는지 확인한다.

**release certificate와 독립 재계산**

최종 판정은 높은 padding efficiency 하나가 아니다. source에서 pack까지 identity closure, attention/position/loss boundary, truncation utility, mixture planned/realized mass, curriculum transition, distributed exactness, framework call graph, valid-token gradient와 deletion/contamination 차단이 모두 필요하다.

감사자는 `DocumentID→PackedSampleID→UpdateID`와 `UpdateID→DrawID→source span`을 양방향으로 실행한다. checkpoint에서 cold next update를 재현하고 incident target이 current loader에서 0 exposure인지 확인한다. 성능 결과는 useful valid tokens/s와 objective parity를 함께 가진다.

미실행 topology, streaming backend, elastic 조합은 `NOT_RUN`과 필요한 fixture/oracle로 남긴다. 같은 evidence로 독립 운영자가 동일 resume, rollback와 release 결론을 얻을 때 운영 인수를 완료한다.

**valid-token 감사표를 batch에서 checkpoint까지 유지한다.**

각 microbatch는 input tokens, non-padding tokens, attention-visible tokens, label-valid tokens, weighted denominator와 source별 기여를 낸다. accumulation update는 microbatch numerator/denominator 합, DDP global 합과 committed UpdateID를 가진다. dashboard scalar와 checkpoint selection metric도 같은 합에서 계산한다.

audit fixture는 prompt만 있고 completion이 없는 row, padding replica, sample weight 0, rank 하나 전체 invalid, uneven sequence와 overflow-skip update를 포함한다. 각 경우 expected gradient와 consumption commit을 손으로 적는다. denominator 0이면 optimizer/scheduler가 전진하지 않아야 한다.

**replay divergence를 draw·token·pack·batch 네 경계로 좁힌다.**

resume 결과가 다르면 먼저 next DrawID와 SampleID를 비교한다. 같으면 source text/deletion root와 token IDs, 다음으로 truncation/pack placement와 boundary tensors, 마지막으로 rank assignment/collate를 비교한다. final loss만 비교해 first cause를 놓치지 않는다.

각 경계는 input/output digest, producer revision, RNG counter와 decision reason을 가진다. first mismatch에서 downstream 비교를 중단하고 source/config/state를 조사한다. expected distributional 차이는 allow rule과 mass window로 별 표시한다.

**mixture phase 인계를 two-phase activation으로 수행한다.**

coordinator는 candidate WeightRevision과 CurriculumGeneration을 준비하고 모든 ranks가 source inventory, deletion root와 digest를 검증하게 한다. prepare가 끝난 뒤 effective UpdateID에 commit한다. 일부 rank가 실패하면 old generation을 유지하고 candidate prefetch를 폐기한다.

activation 첫 window에서 requested/realized source, token, loss mass, quota와 exhaustion을 비교한다. old-generation pack이 남았다면 explicit carry relation로 보이거나 gate가 실패해야 한다. rollback token은 old weights를 복원하되 최신 deletion을 되돌리지 않는다.

**packing bias를 update co-occurrence ablation으로 승인한다.**

동일 DrawID stream을 random pack, length-sorted pack, source-stratified pack과 unpacked microbatch에 배치한다. total tokens와 optimizer schedule을 맞추고 update별 length/source co-occurrence, gradient variance, loss와 downstream slice를 비교한다.

utilization 이득과 behavior delta를 함께 보고한다. 특정 algorithm이 짧은 source를 tail filler로 반복하거나 긴 source를 초기 update에 몰면 realized ledger에서 확인한다. bias가 의도라면 curriculum contract, 아니면 randomization/window를 수정한다.

**distributed deletion replay를 world-size matrix에서 시험한다.**

DP 1, 2, 3과 elastic 4→2에서 동일 deletion target을 prefetch/open-bin/committed 상태에 각각 둔다. tombstone 뒤 target DrawID가 reclaim, rebuild 또는 exposure report 중 policy에 맞게 처리되는지 확인한다. rank-local cache와 restored checkpoint가 target을 되살리지 않아야 한다.

matrix는 next valid draws, duplicate/skip, source mass, first update denominator와 parameter delta를 기록한다. exact 지원 셀은 tensor parity, distributional 셀은 declared bound를 요구한다. topology 하나의 성공을 전체 지원으로 일반화하지 않는다.

**release certificate를 소비 증거로 완성한다.**

certificate는 dataset/deletion/tokenizer roots, sampler/packer/collator revisions, weight/curriculum generations, tested world sizes, valid-token reducer, cold resume와 failure fixtures를 담는다. representative PackedSampleIDs와 UpdateIDs는 원 source span까지 resolve된다.

승인자는 planned mixture가 valid loss mass로 실현됐는지, boundary가 kernel에서 적용됐는지, resume가 요구 등급을 만족하는지, 삭제·오염 target이 모든 active state에서 차단됐는지 답한다. 네 답이 raw event와 digest로 재현될 때 6장의 데이터 입력 계약이 최종 봉인된다.

**update별 mixture drift를 sequential test로 감시한다.**

고정된 긴 window가 끝날 때까지 기다리면 source outage나 stale weight를 늦게 발견한다. UpdateID마다 planned source/token/loss probability와 committed mass를 누적하고 minimum expected count를 넘긴 cohort에 sequential bound를 적용한다. 다만 pack과 quota가 draw를 상관시키므로 iid 가정을 그대로 쓰지 않고 simulator에서 phase별 alert band를 만든다.

alert가 나면 requested draws, eligibility failures, truncation, valid ratio, pack tail, aborted updates와 rank generation을 순서대로 분해한다. source draw는 정상인데 loss mass만 부족하면 긴 prompt mask나 zero-valid completion을 의심한다. 모든 rank의 draw가 다르면 RNG/membership, 특정 rank만 다르면 assignment/prefetch state를 본다.

drift를 수정하기 위해 즉시 weight를 반대로 조정하지 않는다. availability 또는 denominator bug를 고친 뒤 original planned distribution에서 canary를 재실행한다. intentional compensation은 새 WeightRevision, duration과 종료 조건을 가진다. 일시적 조정이 permanent curriculum로 굳지 않게 expiry를 둔다.

**replay catalog를 checkpoint support matrix와 결합한다.**

각 golden CheckpointID는 dataset type, pack algorithm, world size, worker count, curriculum phase, deletion generation, open-bin/prefetch 상태와 expected next 64 packs를 가진다. framework 또는 pipeline revision마다 relevant checkpoints를 cold load해 DrawID, tensors, denominator와 first update를 비교한다.

catalog는 쉬운 update-boundary checkpoint만 모으지 않는다. tail bin, source exhaustion 직전, phase transition, zero-valid rank, long sequence OOM 재분할과 deletion race 직전 상태를 포함한다. failure fixture와 valid recovery artifact를 서로 다른 retention class로 보호한다.

checkpoint가 오래된 dataset root를 요구해도 current deletion policy를 우회해서는 안 된다. exact parity가 policy와 충돌하면 blocked exact recovery로 기록하고 approved sanitized child replay를 별 등급으로 실행한다. support matrix는 그 차이를 명시한다.

**교대 복구 훈련으로 sample 원장을 사람에게서 독립시킨다.**

주 담당자는 synthetic mixture drift와 boundary corruption 하나를 선택해 artifact만 남긴다. 다음 교대자는 dashboard aggregate에서 affected UpdateID를 찾고 ledger로 DrawID, PackPlanID, SampleID와 source span까지 좁힌다. 구두 힌트나 임시 notebook에 의존하지 않는다.

복구자는 freeze 범위, safe checkpoint, replay 등급, rebuild partitions와 promotion gate를 결정한다. 다른 운영자가 같은 evidence로 다른 결론을 내면 resolver output, state transition 또는 runbook 조건이 모호한 것이다. 문장을 늘리기보다 machine-readable decision reason을 보강한다.

훈련 종료는 process 재시작이 아니다. next pack과 denominator가 oracle에 맞고, planned/realized mass가 정상 window로 돌아오며, incident target이 active cache와 prefetch에 없고, 새 checkpoint가 cold read될 때 끝난다. 이 교대 가능성이 6장의 실전 재현성을 완성한다.

**독립 계산으로 valid-token 질량을 다시 구한다.**

최종 auditor는 trainer가 출력한 합계를 그대로 복사하지 않는다. committed update event에서 PackPlanIDs를 모으고 각 sidecar의 label-valid bitmap과 token weight를 합해 denominator를 다시 계산한다. source, domain, curriculum phase별 numerator와 denominator를 분해하고 training metric의 값과 일치하는지 확인한다.

같은 packs를 single-process reference reducer에 넣어 distributed gradient와 parameter delta를 비교한다. padding replica, zero-valid rank와 uneven microbatch가 있어도 global objective가 같아야 한다. loss만 일치하고 parameter update가 다르면 DDP scaling, accumulation 또는 scaler state를 조사한다.

sampler 쪽에서는 weight table과 inventory에서 각 DrawID의 선택 probability를 재계산하고 quota/exhaustion transition을 적용한다. observed valid mass와 simulator band가 맞는지 확인한다. packer 쪽에서는 total retained+inserted+padding token과 source mapping의 gap/overlap을 검산한다.

삭제와 contamination probe를 마지막에 다시 실행한다. current deletion root보다 오래된 cache, checkpoint와 prefetch를 섞어도 target이 loader output에 나타나지 않아야 한다. 발견되면 단어 수나 다른 검사 성공과 무관하게 release를 거부한다.

이 독립 계산 결과는 DatasetRevision, WeightRevision, CurriculumGeneration, sampler checkpoint, tokenizer와 model UpdateID를 한 certificate에 묶는다. 모든 값이 원 artifact에서 재생성되고 미검증 셀이 명시될 때 설명과 실제 운영 증거의 간극이 닫힌다.

마지막 negative run에서는 source weight 하나, loss bitmap 한 칸, RNG counter, open-bin child order와 deletion generation을 각각 바꾼 복사본을 만든다. auditor는 aggregate throughput이 정상이어도 digest agreement, denominator 재계산, replay first difference와 resolver freshness에서 변조를 찾아야 한다. 각 복사본이 서로 다른 최초 gate에서 실패하는지 기록하면 검사들이 같은 증상만 중복 감시하지 않는다는 사실도 확인할 수 있다. 수정하지 않은 production artifact는 read-only로 유지하고 실패 복사본은 별 FixtureID와 만료 없는 regression retention을 가진다.

다음 loader, collator, framework 또는 topology revision은 이 복사본들을 다시 실행해 sample 선택부터 optimizer effect까지의 인과 사슬이 계속 닫혀 있는지 증명한다.

### 6.12.4 weight에서 outcome까지 폐루프를 닫는다

mixture 실험의 입력은 보통 `{"code": 0.2, "web": 0.8}`처럼 간단하다. 그러나 이 숫자와 최종 품질 사이에는 적어도 다섯 개의 상태 변환이 있다. 설정 weight \(w_d\)를 sampler가 draw 확률 \(q_d\)로 정규화하고, rank \(r\)의 iterator가 문서열을 뽑고, tokenizer·filter·packer가 소비 가능한 target token을 만들고, checkpoint가 그 진행 상태를 잘라 저장하며, 평가기가 outcome \(y\)를 낸다. 따라서 한 실험의 최소 계보는 다음 식으로 쓴다.

\[
W^{(v)} \rightarrow Q_t \rightarrow
N^{\mathrm{draw}}_{d,r,t} \rightarrow
N^{\mathrm{emit}}_{d,r,t} \rightarrow
N^{\mathrm{valid}}_{d,r,t} \rightarrow
C_t \rightarrow Y_{s,k}.
\]

\(W^{(v)}\)는 버전이 붙은 설정, \(Q_t\)는 시점 \(t\)에 sampler가 실제 사용한 분포다. 세 개의 \(N\)은 각각 뽑힌 문서, packer가 내보낸 token, loss mask가 1인 token의 rank별 누적값이다. \(C_t\)는 재개 가능한 제어 상태, \(Y_{s,k}\)는 평가 slice \(s\)와 checkpoint \(k\)의 결과다. 이 중 하나를 생략하면 “weight를 바꿔 성능이 올랐다”는 문장은 기제 설명이 아니라 상관관계 메모에 머문다.

**Skill-It의 구현을 이 폐루프 위에 올려 본다.** 공개 코드의 `MWTrainer.train`은 초기 prior와 skill dependency graph에서 exponentiated weight를 만들고, `train_data.set_proportions`에 넘긴다(`trainer/mw_trainer.py:70-115`). 일정 update 구간을 학습한 뒤 validation skill loss를 모아 graph와 곱하고 다시 지수화한다(`:139-221`). 개념적으로는

\[
\widetilde w_{t+1,i}=w_{0,i}\exp\!\left(\eta_t\sum_j A_{ij}\,\widehat L_{t,j}\right),
\qquad
q_{t+1,i}=\frac{\widetilde w_{t+1,i}}{\sum_k\widetilde w_{t+1,k}}
\]

이다. \(A_{ij}\)는 skill \(i\)의 데이터가 target skill \(j\) 학습에 도움을 준다는 관계, \(\widehat L\)은 선택한 window의 validation loss 신호다. 구현에는 loss normalization, window, 동적 off-diagonal 감쇠와 lone-node 질량 보존이라는 분기가 있으므로 논문의 “skill graph curriculum” 한 문장으로 실행 recipe를 대신할 수 없다.

그 다음 경계도 놓치기 쉽다. `LegoDataset.set_proportions`는 segment별 weight를 행 정규화하고, `_get_tokenized_train`은 먼저 segment 길이를 정수로 만든 뒤 마지막 segment가 나머지를 받게 한다(`dataset/lego_dataset.py:77-139`). 새 dataset은 update마다 다시 만들어지고 `SequentialSampler`, `num_workers=0`인 DataLoader로 소비된다(`trainer/utils.py:83-102`). 즉 이 고정 구현에서 기록되는 `weights/sum(weights)`는 sampler가 목표로 삼은 sample 질량이지 token 질량이 아니다. skill별 sequence 길이와 padding·mask가 다르면 \(q_i\)와 \(N^{\mathrm{valid}}_i/\sum_jN^{\mathrm{valid}}_j\)는 달라진다.

더 중요한 경계는 resume이다. 해당 학습 loop의 공개 구간에는 `counter`, `all_losses`, 현재 `weights`, graph 감쇠 상태, 방금 재생성한 dataset의 cursor를 하나의 `state_dict`로 저장하고 복원하는 경로가 보이지 않는다. 이것은 “Skill-It은 재개할 수 없다”는 보편 명제가 아니다. 이 고정 revision의 이 loop만으로는 update 중간 crash 뒤 sample-exact trajectory를 증명할 수 없다는 부재 근거다. model·optimizer checkpoint만 읽으면 같은 step에서 시작하더라도 selector의 관측 window와 다음 draw가 달라질 수 있다.

**대표 기법은 서로 다른 의사결정 문제에 답한다.** DoReMi는 reference 대비 token excess loss를 domain별로 모아 exponentiated-gradient weight를 갱신하고, proxy loss에는 현재 sampling probability의 역수를 이용한 importance correction을 넣는다. Skill-It은 skill dependency graph와 target skill loss를 이용해 다음 학습 구간의 sample 분포를 고른다.

RegMix는 많은 후보 mixture로 proxy model을 학습하고 downstream metric의 회귀 관계로 대형 run 후보를 선택한다. DataComp-LM은 mixture controller가 아니라 고정된 학습·평가 규약 아래 데이터 filtering과 구성의 효과를 비교하려는 benchmark다. 따라서 네 이름을 “adaptive sampling 성능 순위” 한 열에 놓는 것은 범주 오류다.

| 접근 | 선택 시점과 단위 | 관측 신호 | 공개 구현에서 직접 확인할 상태 | 과장하면 안 되는 결론 |
|---|---|---|---|---|
| DoReMi | proxy 학습 중 domain weight | reference 대비 token excess loss | domain weight, score buffer, sampling weight | proxy weight가 모든 규모의 최적 mixture라는 결론 |
| Skill-It | update 구간마다 skill sample weight | skill graph와 validation loss | loss window, graph, weight, dataset 재생성 | graph edge가 일반적인 인과 전이를 증명한다는 결론 |
| RegMix | 대형 run 전 candidate mixture | 여러 proxy run의 downstream metric | candidate config, run row, regression·선택 artifact | 후보 분포 밖에서도 회귀가 맞는다는 결론 |
| DataComp-LM | benchmark 제출 전 dataset 구성 | 고정 protocol의 model evaluation | dataset artifact, training config, evaluation row | 다른 tokenizer·예산에서도 순위가 보존된다는 결론 |

**여섯 문서짜리 fixture로 닫힌 고리를 검증한다.** A skill에는 길이 2의 sample 세 개, B에는 길이 6의 sample 세 개를 두고 configured sample weight를 각각 0.75와 0.25로 둔다. 두 rank가 여덟 번 draw하도록 고정 RNG event list를 만든다. 각 event에는 `MixtureVersion, SelectorUpdateID, GlobalDrawIndex, Rank, SampleID, RawTokens, EmittedTokens, ValidTokens`를 남긴다. A에서 여섯 번, B에서 두 번 뽑혔다면 sample share는 설정과 일치하지만 valid token이 각각 12와 12라면 gradient denominator share는 0.5와 0.5다. 이 차이는 sampler 실패가 아니라 길이와 mask를 통과하며 생긴 질량 변환이다.

네 번째 draw 직후 crash를 주입하고 다음 네 draw를 비교한다. checkpoint에는 model·optimizer뿐 아니라 mixture version, selector의 loss window, graph revision, mixture RNG, domain-local cursor, global draw index, prefetch queue 또는 명시적 replay cut이 들어가야 한다. 재개 뒤 최초 차이가 domain이면 selector/RNG, domain은 같고 SampleID가 다르면 local cursor, SampleID는 같고 valid token만 다르면 tokenizer·packing·mask revision을 조사한다. world size가 2에서 4로 바뀌면 rank-local cursor 동등성을 요구하지 않고 global draw stream을 새 rank에 재분배하는 별도 계약을 적용한다.

마지막으로 outcome을 mixture와 다시 연결한다. checkpoint \(k\)의 평가값만 보존하지 말고 그 시점까지의 누적 valid-token 벡터 \(\mathbf n_k\), mixture version별 체류 구간, selector observation/apply 지연을 같이 고정한다. 두 run의 final score가 같아도 \(\mathbf n_k\)가 다르면 같은 데이터 실험이 아니다. 반대로 configured weight가 달라도 실현 \(\mathbf n_k\)가 같다면 sampler·길이·mask가 차이를 상쇄했을 수 있다. 13장의 schedule 비교는 이 누적 token clock을 받아야 하며, `optimizer_step`만 맞춘 비교를 데이터 통제 실험이라고 부르지 않는다.

release gate는 세 질문으로 끝난다. 설정 weight가 어느 함수에서 실제 draw 확률이 되었는가. 모든 rank가 실제로 소비한 valid token을 checkpoint 전후에 재정산할 수 있는가. 그 누적 질량을 평가 row와 exact checkpoint로 결합했는가. 셋 중 하나라도 답할 수 없다면 mixture는 아직 닫힌 제어계가 아니다.

## 6.13 학습 목적을 바꾸는 네 가지 corruption을 같은 데이터 계약으로 읽는다

다음 토큰 예측만 알면 현대 언어 모델의 데이터 파이프라인을 절반만 본 셈이다. 같은 원문 토큰열이라도 어느 위치를 입력에서 감추고, 어느 위치를 label로 남기며, decoder가 무엇을 볼 수 있게 하느냐에 따라 전혀 다른 조건부 분포를 학습한다. masked LM, span corruption, FIM, UL2는 모두 ‘텍스트 일부를 변형한다’는 표면을 공유하지만, 예측 단위와 attention 방향, loss 분모, 추론 인터페이스가 다르다. 따라서 `objective_name` 하나로 묶지 말고 corruption state와 tensor 계약을 기록해야 한다.

원문을 \(x=(x_1,\ldots,x_T)\), 예측 위치 집합을 \(M\)이라 하자. decoder-only 다음 토큰 목적은

\[
\mathcal L_{\mathrm{AR}}=-\sum_{t=1}^{T-1}\log p_\theta(x_{t+1}\mid x_{\le t})
\]

이다. 모든 유효 위치가 왼쪽 문맥만 조건으로 갖는다. 반면 masked LM은 변형 함수 \(c_M(x)=\tilde x\)를 먼저 만들고

\[
\mathcal L_{\mathrm{MLM}}=-\sum_{t\in M}\log p_\theta(x_t\mid \tilde x)
\]

를 최소화한다. encoder의 양방향 attention 때문에 예측 위치는 왼쪽과 오른쪽의 관측 토큰을 함께 본다. 이 차이는 단순한 mask 비율이 아니다. AR은 정규화된 joint likelihood를 chain rule로 직접 정의하지만, 고전 MLM의 여러 조건부 예측을 그대로 곱한 값은 일반적으로 같은 joint likelihood가 아니다. 그래서 MLM loss와 AR perplexity를 같은 숫자로 비교하면 안 된다.

### 6.13.1 masked LM의 선택 mask와 입력 변형 mask를 분리한다

Transformers의 고정 소스 `DataCollatorForLanguageModeling.torch_mask_tokens`를 따라가면 상태 전이가 선명하다. 먼저 `labels = inputs.clone()`으로 정답을 보존한다. `mlm_probability`로 Bernoulli 선택 mask를 뽑되 special-token 위치의 확률은 0으로 만든다. 이어 `labels[~masked_indices] = -100`으로 loss가 흐를 위치를 확정한다. 그 뒤에야 선택 위치의 입력을 80% `[MASK]`, 10% 무작위 vocabulary token, 10% 원래 token으로 남긴다.

```python
labels = inputs.clone()
probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
masked_indices = torch.bernoulli(probability_matrix).bool()
labels[~masked_indices] = -100

indices_replaced = bernoulli_08 & masked_indices
inputs[indices_replaced] = mask_token_id
indices_random = bernoulli_05 & masked_indices & ~indices_replaced
inputs[indices_random] = random_words[indices_random]
```

두 번째 Bernoulli가 0.5인 이유는 전체 선택 위치의 절반이 아니다. 이미 80% replacement에서 제외된 20% 안에서 절반을 뽑으므로 무작위 치환은 전체 선택 위치의 10%가 된다. 나머지 10%는 원래 token을 입력에서 보지만 label은 여전히 유효하다. 이는 fine-tuning 때 `[MASK]`가 없는 입력과의 불일치를 줄이면서도, hidden state가 관측 token을 그대로 복사하는 데만 의존하지 않도록 만든다.

배치 fixture는 선택 mask \(M\), replacement mask \(R\), random mask \(Q\), special-token mask \(S\), labels를 모두 보존해야 한다. 최소 불변식은 \(R\subseteq M\), \(Q\subseteq M\setminus R\), \(M\cap S=\varnothing\), `labels[t] == -100` iff \(t\notin M\)이다. `attention_mask`는 padding과 attention 가시성을 제어하고 `labels == -100`은 loss 가시성을 제어하므로 둘을 같은 bitmap으로 취급해서는 안 된다.

실패 fixture도 구체적이어야 한다. special token이 선택되면 tokenizer protocol 자체를 예측하게 되고, padding label이 살아 있으면 짧은 문장이 과소 가중된다. `inputs`를 clone하지 않은 채 cache된 token array를 제자리 변형하면 다음 epoch의 원문이 오염된다. 분산 rank마다 RNG seed만 같고 sample shape가 다르면 random draw 소비량이 어긋나 exact resume이 깨진다. 따라서 sample identity에서 파생한 stateless corruption seed, 혹은 collator RNG counter를 checkpoint에 넣는다.

### 6.13.2 span corruption은 삭제 구간의 길이와 순서를 sentinel로 보존한다

T5식 span corruption은 독립 token masking보다 구조적인 복원 문제를 만든다. 여러 연속 token으로 이루어진 noise span마다 하나의 sentinel \(s_k\)를 부여한다. encoder 입력은 감춘 span을 sentinel 하나로 압축하고, decoder target은 각 sentinel 다음에 그 span의 원래 token을 순서대로 둔다. 예를 들어

```text
원문:    A B C D E F G H
입력:    A B <extra_id_0> F <extra_id_1> H
target:  <extra_id_0> C D E <extra_id_1> G </s>
```

가 된다. 목적함수는 decoder target \(y\)에 대한 teacher-forced sequence likelihood

\[
\mathcal L_{\mathrm{span}}=-\sum_{j=1}^{|y|}\log p_\theta(y_j\mid y_{<j},\,c_M(x))
\]

이다. 입력에서 삭제된 token 수 \(N_m\), noise span 수 \(K\)라면 EOS를 포함한 target 길이는 대략 \(N_m+K+1\), encoder 입력 길이는 \(T-N_m+K+1\)이다. 따라서 raw sequence length를 곧바로 `max_seq_length`로 자르면 padding이나 shape mismatch가 생긴다. 먼저 noise density와 평균 span 길이로 expanded raw length를 역산해야 한다.

고정된 Transformers Flax 예제의 `FlaxDataCollatorForT5MLM`은 이 계약을 함수로 드러낸다. `random_spans_noise_mask`가 noise token 수와 span 수를 정하고, `create_sentinel_ids`가 각 span 시작에 vocabulary 끝쪽 sentinel ID를 놓으며 span 내부에는 삭제 표지 `-1`을 둔다. `filter_input_ids`는 연속 mask token을 sentinel 하나로 압축하고 EOS를 붙인다. 마지막으로 `shift_tokens_right(labels, pad_token_id, decoder_start_token_id)`가 decoder 입력을 만든다. 즉 `mask→sentinel→compact→shape assert→decoder shift`가 끊을 수 없는 한 transaction이다.

검증 fixture는 길이가 2인 최소열, 모든 token이 mask될 뻔한 열, noise span 하나, 인접 span 경계, sentinel 수가 tokenizer의 `extra_ids`를 넘는 열을 포함한다. 각 fixture에서 noise token 보존, sentinel의 단조로운 순서, 입력과 target을 합쳐 원문을 유일하게 복원할 수 있는지, 계산된 input/target shape를 확인한다. sentinel ID를 일반 vocabulary와 충돌시키거나 target에서 sentinel을 한 칸 늦게 놓으면 loss는 유한해도 ‘어느 빈칸의 답인가’라는 주소가 바뀐다.

### 6.13.3 FIM은 decoder-only 모델의 순서를 재배열해 양쪽 문맥을 조건으로 만든다

FIM(fill-in-the-middle)은 encoder-decoder를 추가하지 않고 decoder-only causal loss를 그대로 사용한다. 원문을 prefix \(P\), middle \(M\), suffix \(S\)로 자르고 special token을 이용해 직렬화한다. 대표적인 PSM(prefix–suffix–middle) 변환은

\[
z=[\texttt{FIM\_PRE}],P,[\texttt{FIM\_SUF}],S,[\texttt{FIM\_MID}],M
\]

이며 학습은 여전히 \(-\sum_t\log p(z_{t+1}\mid z_{\le t})\)이다. middle token이 등장할 때 causal prefix 안에 원래의 왼쪽과 오른쪽 문맥이 모두 들어 있으므로 infilling을 배운다. SPM(suffix–prefix–middle)은 앞부분 순서를 바꿔 position bias와 긴 prefix 편향을 달리한다. [FIM 논문](https://arxiv.org/abs/2207.14255)은 이 단순한 데이터 변환을 높은 비율로 섞어도 실험 범위에서 원래 left-to-right 능력을 크게 해치지 않았음을 보고하지만, 이를 모든 tokenizer·모델·도메인에 대한 무조건적 보장으로 읽으면 안 된다.

FIM packer는 먼저 document 하나 안에서 character가 아닌 token 경계로 두 cut을 뽑는다. 빈 middle 허용 여부, middle 길이 분포, FIM 적용 확률, PSM/SPM 혼합률을 `TransformDecision`에 기록한다. 그 뒤 special token을 넣고 length budget을 계산한다. 변환 후 잘라 버리면 `[FIM_MID]`는 남고 정답 middle이 사라지는 zero-useful sample이 생길 수 있으므로, truncation은 세 segment와 marker를 원자 단위로 보존하거나 샘플을 명시적으로 거부해야 한다.

FIM과 일반 AR을 같은 batch에 pack할 때 문서 경계 attention이 열리면 suffix가 이웃 문서의 middle을 조건으로 보게 된다. `segment_ids`, `position_ids`, causal block mask, labels가 함께 변해야 한다. middle-only loss를 선택한다면 `[FIM_MID]` 이전 labels를 `-100`으로 만들고 denominator를 middle token 수로 센다. full-sequence loss라면 marker와 재배열된 prefix/suffix도 학습 질량을 갖는다. 두 정책은 같은 ‘FIM 적용률’ 아래에서도 gradient 분포가 다르다.

작은 반증 fixture는 `P=[a,b]`, `M=[c,d]`, `S=[e,f]`를 고정하고 변환된 ID 열, marker 위치, shift된 label, middle 시작에서의 visible prefix를 손으로 적는다. cut이 Unicode byte 중간에 걸리는 오류, special token이 tokenizer에서 둘로 쪼개지는 오류, EOS가 suffix와 middle 사이에 들어가 generation이 종료되는 오류, PSM/SPM 이름과 실제 배열이 뒤집힌 오류를 각각 독립 시험한다. serving 쪽 infill prompt builder도 같은 tokenizer revision과 marker 순서를 써야 한다.

### 6.13.4 UL2는 denoiser 하나가 아니라 denoiser 분포를 학습한다

[UL2 논문](https://arxiv.org/abs/2205.05131)은 architecture와 pre-training objective를 분리하고, Mixture-of-Denoisers(MoD)로 서로 다른 corruption regime을 섞는다. 직관적으로 R-denoiser는 비교적 짧고 규칙적인 span corruption으로 local 복원을 강조하고, S-denoiser는 prefix language modeling처럼 sequential generation에 가까운 조건을 만들며, X-denoiser는 더 높은 noise density와 긴 span으로 극단적인 복원을 요구한다. 구현에서 중요한 것은 글자 R/S/X가 아니라 각 mode가 가진 `noise_density`, `mean_noise_span_length`, prefix split 정책과 mixture probability다.

mode를 \(d\sim\pi\), 그 mode의 corruption을 \(c_d\), target 변환을 \(g_d\)라 쓰면 목적은

\[
\mathcal L_{\mathrm{MoD}}(\theta)=
\mathbb E_{x\sim D}\mathbb E_{d\sim\pi}
\left[-\sum_j\log p_\theta(g_d(x)_j\mid g_d(x)_{<j},c_d(x))\right]
\]

이다. 설정의 mixture probability \(\pi_d\)가 실제 gradient 질량은 아니다. mode마다 target 길이가 다르므로 token-mean reduction이면 실현 질량은 대략 \(\pi_d\mathbb E[|y_d|]\)에 비례한다. 긴 X target이 같은 sample 확률로 더 큰 gradient 질량을 차지할 수 있다. mode별 sample count뿐 아니라 valid target tokens, loss numerator, denominator, truncation/rejection 질량을 기록해야 하는 이유다.

UL2의 mode token은 ‘모델에게 풀이 규칙을 알려 주는 control plane’이다. mode token이 tokenizer에서 원자 token인지, encoder 입력의 어느 위치에 붙는지, packing 후 보존되는지, fine-tuning과 serving에서 어떤 mode를 선택하는지까지 artifact 계약에 포함한다. mode token을 잃으면 모델은 동일한 관측 패턴에서 서로 다른 복원 규칙을 추론해야 하고, 잘못 붙이면 objective label과 조건 신호가 모순된다.

운영 fixture는 같은 원문과 같은 RNG key로 모든 denoiser를 생성해 `(mode, input_ids, labels, decoder_input_ids, valid_count)` golden tuple로 저장한다. mode weight 하나를 0으로 둔 경우 해당 corruption이 실제로 사라지는지, checkpoint resume 뒤 다음 mode draw와 span cut이 일치하는지, rank별 mode 질량이 global 합과 맞는지 확인한다. mode별 loss를 전체 평균 하나로만 보면 쉬운 R mode의 개선이 X mode 붕괴를 가릴 수 있으므로 validation도 동일한 denoiser별 slice를 유지한다.

### 6.13.5 objective mixture의 최초 차이를 corruption 이전부터 찾는다

네 목적을 한 trainer에서 다룰 때 공통 상태 사슬은 다음과 같다.

```text
DocumentID + token IDs
  → ObjectiveDraw(mode, RNG key)
  → cut/span/mask decision
  → transformed input + target + special tokens
  → attention/position/loss masks
  → pack placement
  → valid-token numerator/denominator
  → gradient contribution
```

loss가 달라졌다고 즉시 model kernel을 의심하지 않는다. 같은 `DocumentID`의 raw token IDs가 같은지, `ObjectiveDraw`와 RNG counter가 같은지, cut/span bitmap이 같은지, marker/sentinel ID가 같은지, transformed tensor와 pack boundary가 같은지 차례로 비교한다. 최초 차이가 corruption 이전이면 tokenizer/data revision 문제이고, corruption decision이면 seed·mode sampler 문제이며, tensor가 같은데 loss만 다르면 attention mask·shift·reduction 또는 model 경로 문제다.

release certificate에는 objective별 paper/config revision, transform symbol과 source span, tokenizer special-token mapping, RNG derivation, golden fixture digest, planned/realized sample 및 valid-token 질량을 넣는다. AR validation perplexity, MLM masked-token accuracy, span exact reconstruction, FIM middle likelihood는 서로 다른 질문에 답한다. 하나의 aggregate loss로 네 목적의 품질을 대신 증명하지 않는다. 이 구분을 지켜야 objective mixture가 ‘데이터 증강 옵션’이 아니라 재현 가능한 확률모형 계약이 된다.

packing·mixture·curriculum을 관통하는 최종 질문은 설정 파일의 비율이 아니라 **어느 원문 토큰이 어떤 문맥과 정답 규칙을 거쳐 어느 update에 얼마만큼 기여했는가**다. `DocumentRevision`에서 `UpdateID`까지의 사슬을 정방향으로 재생하고, 예상 밖 loss에서 최초로 다른 draw·pack·mask·분모를 역추적할 수 있어야 한다. 이 소비 원장은 13장의 학습 시계와 16장의 분산 재개가 데이터 의미를 잃지 않게 하는 입력이 된다.

### 6.13.6 DoReMi의 비율은 설정값이 아니라 갱신되는 상태다

DoReMi 공개 구현의 `DoReMiTrainer.training_step`(`doremi/trainer.py:272-388`, commit `7cde52d`)은 현재 모델의 token loss에서 reference model의 token loss를 빼 `excess_loss`를 만든다. 이어 domain ID와 token mask를 함께 모으고, 현재 sampling probability로 나눈 importance correction을 proxy loss에 적용한다. 여기서 분모는 장식이 아니다. 특정 domain을 적게 뽑았다는 이유만으로 그 domain의 관측 gradient 질량이 작아지는 편향을 보정한다. 반대로 probability가 지나치게 작으면 분산이 커지므로 sampling floor와 관측 창이 함께 필요하다.

`update_domain_weights`(`:239-270`)는 domain별로 유효 token의 nonnegative score 평균을 만들고 `log(w_d) + eta * score_d`를 정규화한 뒤 epsilon uniform floor를 섞는다. 따라서 로그에 찍힌 `train_domain_weights`는 원 corpus의 고정 비율도, 최종 대형 모델의 최적 비율도 아니다. reference와 proxy model, 현재 window, `eta`, `epsilon`, mask와 distributed gather가 만든 동적 제어 상태다. 이 값을 다음 sampler가 실제로 읽은 시점까지 연결해야 ‘weight를 계산했다’와 ‘그 비율로 소비했다’를 구분할 수 있다.

디버깅 fixture는 두 domain과 손으로 계산 가능한 네 token이면 충분하다. reference와 proxy loss를 고정하고 한 domain의 mask를 모두 false로 만든다. 먼저 excess loss·importance-corrected loss를 계산한 뒤, 빈 domain이 직전 score를 유지하는지, exponentiated update 뒤 합이 1인지, epsilon floor 아래로 내려가지 않는지 확인한다. 마지막으로 다음 batch의 realized domain count를 별도로 센다. 이 검사는 공개 Python 경로의 상태 전이를 고정하지만 production sampler의 장기 수렴이나 대규모 품질 향상을 증명하지는 않는다.

### 6.13.7 `weight`라는 이름을 확률로 읽기 전에 selector의 동사를 읽는다

같은 `weight` 필드도 구현에 따라 전혀 다른 상태를 바꾼다. Hugging Face Alignment Handbook의 `get_dataset`(`src/alignment/data.py:26-76`, commit `1de1fc9`)은 각 데이터셋을 읽은 뒤 `shuffle(seed).select(range(int(len(ds) * weight)))`를 수행한다. 여기서 weight는 매 draw의 source 확률이 아니라 **원본 행 가운데 한 번 보존할 비율**이다. 선택된 유한 데이터셋들을 이어 붙인 뒤 다시 한 번 shuffle한다. 따라서 (N_d)개 행을 가진 source (d)의 설정값이 (w_d)이면 구성 직후 행 수는

\[
n_d=\lfloor N_d w_d\rfloor,\qquad
q_d^{\mathrm{row}}=\frac{n_d}{\sum_j n_j}
\]

가 된다. (w_d)의 합이 1이라는 사실만으로 (q_d^{\mathrm{row}}=w_d)가 되지 않는다. source 크기가 다르면 더 크게 어긋난다. replacement sampling도 아니므로 작은 source를 여러 epoch 반복 노출하는 정책과도 다르다. 공개 test `test_loading_dataset_mixture`와 `test_loading_with_fractional_weights`(`tests/test_data.py:25-105`)는 100행 source에서 0.5·0.3·0.2가 50·30·20행으로, 0.7·0.4가 70·40행으로 물질화되고 그 뒤 train/test split이 적용됨을 고정한다. 이 test가 증명하는 것은 행 수 계약이지 token 비율·supervised-token 비율·장기 sampling 분포가 아니다.

이 경로를 다섯 분모로 풀면 설정 오류가 빨리 보인다. `configured`는 입력 (w_d), `selected`는 floor 뒤 (n_d), `emitted`는 tokenizer·truncation·packer를 통과한 token, `supervised`는 label mask 뒤 loss-bearing token, `consumed`는 실제 optimizer update에 commit된 token이다. 예컨대 instruction source A와 장문 code source B가 같은 행 수로 선택돼도 B의 emitted token 질량이 훨씬 클 수 있다. response-only loss라면 prompt가 긴 A의 supervised 질량은 다시 작아진다. 그러므로 source별 계기판은 `selected_rows`, `emitted_tokens`, `valid_target_tokens`, `committed_update_tokens`를 나란히 보여야 한다.

재개 경계도 엄격히 나눈다. 이 함수의 두 shuffle은 seed가 같고 입력 dataset revision과 행 순서가 같을 때 materialized 결과를 재구성하는 장치다. trainer가 어느 행까지 소비했는지, packer의 남은 조각이 무엇인지, prefetch 중 어느 batch가 optimizer step에 commit됐는지는 저장하지 않는다. `get_dataset` test를 sample-exact resume의 근거로 승격해서는 안 된다. 최소 fixture는 크기가 다른 두 source, 길이가 다른 행, response mask를 가진다. 구성 전후 행 ID와 순서를 고정하고 tokenize→pack→mask 뒤 네 분모를 계산한 다음 중간 update에서 중단한다. 복원 후 다음 `SampleID`, pack residual, RNG counter와 committed token ledger가 모두 같을 때만 exact resume다.

마지막으로 이 selector는 권리·삭제 상태를 해석하지 않는다. 삭제된 `DocumentRevision`이 기존 Hub dataset revision 안에 남아 있으면 같은 seed는 그 행을 결정론적으로 다시 고를 뿐이다. 허용 corpus generation과 tombstone journal을 `load_dataset` 이전에 검증하고, 선택 manifest에 원 source row fingerprint를 보존해야 한다.

quality classifier의 calibration/evaluation 문서가 mixture 후보에 섞이면 filter가 자기 평가 자료를 학습하는 계보 누수가 생기므로 `ClassifierTrainingSetID`, `CalibrationSplitID`, `BenchmarkItemID`와의 교차 hit도 selector gate에서 검사한다. 공개 함수와 test에는 이 production 보장이 없다는 부정 근거를 명시해, 결정적 shuffle을 거버넌스 폐루프로 오독하지 않는다.

### 6.13.8 streaming resume은 연산자별 증명으로 조립한다

HF Datasets의 iterable filter에는 강한 회귀 테스트가 있다. 2,000행에서 1·500·1,500행을 소비한 상태를 저장하고, 새 dataset graph에 state를 읽힌 뒤 `seen + rest == range(2000)`을 batched/non-batched 양쪽에서 확인한다. 이 테스트는 filter formatter가 한 example을 미리 읽더라도 state가 emitted 위치를 가리켜 남은 행을 한 번씩 재생한다는 계약을 고정한다.

하지만 shuffle은 다른 계약이다. `BufferShuffledExamplesIterable.load_state_dict`는 upstream iterable state를 복원해도 메모리 buffer payload를 저장하지 않았다고 경고하고 buffer를 새 입력으로 채운다. 따라서 filter 테스트가 통과했다는 사실을 shuffle 이후 sample-exact resume으로 확대할 수 없다. graph가 `read → filter → interleave → shuffle → tokenize → pack`이라면 각 노드의 state와 노드 사이 prefetch queue까지 닫혀야 전체 suffix가 같다.

재개 인증은 기준 실행의 다음 (K)개 `SampleID`·source·token span·pack placement를 golden suffix로 저장한다. yield 직후, buffer 교체 직후, shard 경계, worker prefetch 중, accumulation 중간에서 중단한다. 동일 worker/world size뿐 아니라 변경 시나리오도 분리한다. buffer content, RNG bit-generator state, upstream cursor, selected source state, worker assignment 또는 pack residual이 artifact에 없으면 `exact`가 아니라 `best-effort`라고 표시한다.

OLMo VSL의 공개 테스트는 curriculum의 또 다른 조각을 닫는다. 두 mmap의 문서를 길이별 instance로 만들고 world size 2에서 rank별로 읽어 global batch 원소 수와 `total_batches`, padding 제외 token 유일성을 검사한다. 이는 cardinality와 중복 방지의 직접 근거다. 그러나 source별 realized valid-token 비율이나 임의 crash 지점의 suffix parity까지 증명하지 않는다. 테스트 하나가 고정하는 불변식과 아직 열린 불변식을 함께 적어야 한다.

**mixture scheduler를 확률·관측·제어의 세 층으로 읽는다.**

시간 (t)의 configured distribution을 (p_d(t)), 실제 draw 수를 (N_d^{draw}), 유효 정답 token을 (T_d^{valid}), 성공 update에 반영된 질량을 (T_d^{commit})이라 하자. 운영자가 알고 싶은 realized 비율은 목적에 따라 달라진다.

\[
q_d^{draw}=\frac{N_d^{draw}}{\sum_jN_j^{draw}},\qquad
q_d^{valid}=\frac{T_d^{valid}}{\sum_jT_j^{valid}},\qquad
q_d^{commit}=\frac{T_d^{commit}}{\sum_jT_j^{commit}}.
\]

세 값이 다르다는 사실은 sampler 오류가 아니다. 문서 길이, filter pass rate, truncation, packing tail, response mask와 failed update가 만든 예상 가능한 변환일 수 있다. 오류는 이 차이를 계측하지 않거나, configured (p_d)만 저장하고 실제 gradient 질량으로 보고하는 것이다.

DoReMi의 exponentiated-gradient 갱신처럼 scheduler가 feedback을 받는다면 checkpoint에는 weight만 넣어서는 부족하다. reference/proxy revision, domain loss numerator·valid-token denominator, observation window, step, learning rate (eta), floor (epsilon), sampler RNG와 새 weight의 effective draw boundary를 저장한다. 재개 직전 계산한 새 weight가 이미 sampler에 반영됐는지 모르면 한 window를 중복 또는 누락한다. RegMix처럼 offline proxy 결과에서 비율을 고르는 경우에는 후보 vector, proxy run/seed/metric, regression artifact와 선택 규칙이 상태다.

실전 대시보드는 source별 `planned`, `drawn_documents`, `emitted`, `valid`, `committed`와 누적·최근 window를 함께 그린다. `q^{commit}-p`가 커지면 source exhaustion, 평균 길이, filter/quarantine, tail drop, mask, rank skew, failed step 순으로 분해한다. curriculum 단계가 바뀐 시점에는 weight 변화선뿐 아니라 이 다섯 분모의 변화선을 겹쳐야 loss 변화의 원인을 모델과 데이터 중 어디서 찾을지 결정할 수 있다.

## Configured mixture가 실제 학습 질량은 아니다

source 확률은 draw 의도일 뿐이다. 각 source에 대해 configured probability와 realized document, input token, valid target, committed weighted-loss 질량을 별도 열로 둔다. 긴 문서, truncation, supervised mask, source exhaustion, corrupt retry와 update skip 때문에 네 비율은 달라진다. packed tensor의 token마다 source UUID와 원문 offset을 되짚을 수 있어야 최종 denominator를 검산할 수 있다.

DoGE의 고정 구현은 seed로 NumPy generator를 만들고 probabilities, updatable probability handle과 stopping strategy를 cycling multi-source iterator에 전달한다. 그러나 seed 하나만 다시 주는 것은 resume가 아니다. bit-generator state, 각 child iterator cursor, exhaustion·renormalization 상태, controller weight generation과 prefetch queue가 어느 generation을 받아들였는지를 함께 복원해야 다음 UUID와 PackID가 같다.

작은 fixture는 길이와 supervised ratio가 다른 세 source, duplicate cluster 하나, threshold 경계 score, corrupt row 하나를 쓴다. K draw 뒤 checkpoint하고 새 process에서 이어, 다음 UUID 순서와 filter reason, PackID segment map, source별 valid-target·committed-loss 질량을 uninterrupted reference와 비교한다. configured weight만 같은 결과나 평균 survivor 수만 맞는 결과는 합격이 아니다.

## 6.14 SFT pack에서 optimizer commit까지 한 원장으로 닫는다

packing은 token을 붙이는 작업과 attention을 격리하는 작업을 함께 해야 한다. packed row에는 `SegmentID`, 각 segment의 position reset, label bitmap과 causal visibility를 둔다. 다음 example의 token이 이전 example을 보거나 이전 example의 마지막 logits가 다음 example 첫 token을 예측하면 cross-example leakage다. labels가 올바르게 운반됐다는 시험만으로 attention 격리까지 증명되지는 않는다.

causal LM loss는 logits와 labels를 한 칸 어긋나게 맞춘다. 따라서 shift 뒤의 valid prediction 수가 실제 denominator다. padding 전 label 수나 shift 전 `labels != -100`을 세면 각 sequence 첫 위치 처리 때문에 오차가 난다. rank 0에 유효 target 2개, rank 1에 8개를 두고 local numerator, global valid count, DP 평균 뒤 gradient가 단일 10-token reference와 같은지 검산한다. microbatch마다 local mean을 평균내지 않는다.

종단 fixture는 raw example hash에서 시작해 template digest, token ids, labels, PackID와 segment map, shifted numerator/denominator, accumulated gradient, optimizer parent/child digest를 잇는다. update가 commit된 뒤에만 sample cursor를 소비 상태로 확정한다. backward 뒤 crash와 optimizer step 뒤 checkpoint 전 crash를 각각 넣어, 재시작 후 batch order·RNG·gradient accumulation window와 parameter delta가 uninterrupted run과 같은지 확인한다.

Transformers의 causal-loss와 Trainer 시험은 `num_items_in_batch`, accumulation, resume batch order를 각각 직접 고정한다. TRL 시험은 assistant/completion label, zero-valid 제거, truncation과 packing label 운반을 고정한다. torchtune의 공개 recipe는 이 상태들이 한 loop에서 만나는 좌표를 제공하지만, TRL·LLaMA-Factory·Unsloth와 intermediate tensor 및 resume cursor가 완전히 같다는 공통 canonical test는 없다. 그 parity는 실행 전까지 검증되지 않은 요구조건이다.

## 6.15 도메인 혼합의 확률과 실현 질량을 코드에서 분리해 읽는다

Hugging Face Datasets의 map-style interleave 구현은 source 길이와 offset을 계산한 뒤 stopping strategy에 따라 전혀 다른 index 열을 만든다. `first_exhausted`는 어느 source 하나가 끝나면 멈추고, `all_exhausted`는 짧은 source의 cursor를 되감아 모든 source가 한 번 이상 소진될 때까지 계속한다. 확률이 주어지면 NumPy generator가 source index를 뽑는다. 따라서 probability vector는 최종 row 비율을 약속하지 않는다.

**같은 확률도 종료 정책이 바뀌면 다른 objective가 된다.**

고정 canonical fixture에서 `[0.3, 0.5, 0.2]`, seed 42와 `first_exhausted`는 길이 7의 특정 값 열을 만들고 fingerprint 재현성도 확인한다. 같은 확률에 `all_exhausted`를 쓰면 길이가 16이 되고 짧은 source row가 반복된다. 확률 없는 `all_exhausted`도 세 source를 최장 길이에 맞추어 순환한다. 옵션 하나가 단순 성능 knob가 아니라 어떤 예제를 몇 번 학습할지 바꾸는 objective knob인 이유다.

실패 fixture는 길이 2·5·11인 언어, 코드, 수학 source에 서로 다른 token 길이와 supervised ratio를 준다. 출력에서 document count뿐 아니라 input token, shift 뒤 valid-target token, optimizer가 실제 commit한 weighted-loss denominator를 센다. 기대값은 sampler index oracle에서 독립적으로 계산하고, duplicate 제거·filter·packing을 한 단계씩 켜면서 어느 변환이 질량을 바꿨는지 보존한다.

**classifier split과 corpus mixture의 RNG는 별도 상태다.**

classifier의 stratified train/test split seed와 corpus interleave seed를 같은 값으로 적어도 같은 RNG stream이 아니다. library revision, bit-generator 종류와 state, 호출 횟수, source cursor가 있어야 재현된다. split cache fingerprint가 같다는 사실도 training sampler의 다음 UUID를 증명하지 않는다. checkpoint에는 selector artifact digest와 split component ledger를 immutable parent로 연결하고, sampler RNG·cursor·exhaustion과 prefetch acceptance generation을 mutable state로 저장한다.

재개 시험은 K번째 draw 직후 process를 끊고 다음 source index, UUID, tokenizer output, PackID와 source별 누적 네 분모를 uninterrupted oracle과 비교한다. `all_exhausted`에서 끝난 source를 되감는 시점, corrupt row skip, quarantine 증가와 curriculum weight update가 같은 boundary에서 적용되는지도 확인한다. 최종 비율만 비슷한 것은 sequence parity가 아니며, seed만 같은 것은 state parity가 아니다.

[SourceRow에서 committed UpdateID까지](../labs/06-source-to-commit-golden-lab.md)에서는 두 segment의 첫 target mask를 0으로 두고 pack별 denominator 4, global denominator 8을 독립 계산한다. segment 경계를 하나만 바꿨을 때 pack→rank→update digest가 연쇄 변경되는 모습을 보면 configured sample count와 committed target 질량을 같은 값으로 보고할 수 없는 이유가 선명해진다.

[설정한 mixture가 실제 손실 질량이 되기까지](../labs/06-mixture-realized-mass-lab.md)에서는 같은 문제를 source 축으로 펼친다. 세 source가 각 두 문서를 내더라도 입력 토큰은 12/18/14, 유효 타깃은 12/12/3, 커밋 손실 질량은 8/12/0이 된다. source 소진 뒤 확률 재정규화와 curriculum 경계의 exact resume까지 정적 oracle로 검산한다.
