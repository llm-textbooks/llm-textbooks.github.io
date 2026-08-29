# 4장. 웹 코퍼스 제조

이 장의 출발점은 웹 주소가 아니라 원본 객체다. 원본 객체에서 문서를 추출하고, 필터와 중복 제거를 거쳐 보존 문서를 정한 뒤, 혼합 비율을 붙여 token shard와 packed sample을 만든다. 학습기가 그 sample을 실제로 뽑아 갱신에 사용해야 비로소 checkpoint에 영향을 준다. 따라서 이 장의 핵심 질문은 “무엇을 모았는가”보다 “어느 변환을 거친 무엇이 실제 손실에 기여했는가”다.

이 흐름을 한 줄로 쓰면 `raw object → extracted document → policy decision → dedup survivor → mixture eligibility → token/shard span → packed sample → UpdateID → CheckpointID`다. 앞 단계의 출력은 다음 단계의 입력이면서 동시에 감사 좌표다. 5장은 `DocumentID`와 byte span을 token offset으로 바꾸고, 6장은 mixture와 packing이 만든 계획 확률을 실제 소비 질량으로 바꾼다. 24장은 benchmark contamination을 발견했을 때 이 사슬을 역으로 타고 영향받은 `CheckpointID`를 찾는다.

독자는 이후의 모든 절을 같은 네 질문으로 읽으면 된다. 첫째, 관찰된 값은 문서 수·byte·token·sample·loss-bearing token 가운데 무엇을 셌는가. 둘째, 그 값이 달라질 수 있는 최초의 상태 전이는 어디인가. 셋째, 원인을 하나씩 떼어 내는 paired replay나 고장 주입은 무엇인가. 넷째, 어느 manifest·checksum·decision record가 판정을 닫는가. “품질이 좋아졌다”, “중복이 줄었다”, “삭제했다”는 문장은 이 네 질문에 답하기 전에는 결과가 아니라 가설이다.

## 4.1 acquisition에서 학습 후보 문서까지 경계를 세운다

### 4.1.1 수집·추출·언어 판별·품질 gate를 분리한다

웹 응답에는 메뉴, 광고, 반복 footer, 깨진 encoding이 섞인다. `DocumentID`는 URL만으로 만들면 안 된다. fetch 시각, content bytes checksum, extractor revision을 포함해야 같은 URL의 다른 내용을 구분한다. 추출 뒤에도 원본 byte offset과 정제 text offset의 대응을 보존해야 삭제와 오류 추적이 가능하다.

언어·품질 점수는 정답표가 아니라 필터가 참조하는 관찰값이다. threshold를 바꾸면 보존 집합이 달라지고, 그 결과 domain mixture와 tokenizer가 보게 될 token 질량도 달라진다. 그러므로 “점수가 낮다”와 “학습 가치가 없다”를 같은 말로 쓰지 않는다. 높은 threshold는 노이즈를 덜 남길 수 있지만, 분류기가 익숙하지 않은 소수 언어와 코드·수식 문서를 체계적으로 버릴 수도 있다. 어느 문서가 사라졌고 그 문서들이 차지하던 질량이 어떻게 이동했는지는 parent/child manifest diff로 설명한다.

**두 파이프라인의 경계.** 코퍼스 제조는 crawl manifest→raw record→text extraction→tagging→dedup/filter/decontamination→immutable document shard까지다. 학습 소비는 tokenizer→token shard→packing→mixture sampler→rank assignment→optimizer step으로 이어진다. 첫 파이프라인이 “학습 가능한 후보”를 만들고, 둘째가 그 후보 가운데 실제 gradient에 들어간 표본을 정한다. 공개 제조 script를 똑같이 실행할 수 있다는 사실만으로 trainer의 sample-exact resume가 증명되지는 않는다. 이 경계를 잇는 최소 열쇠가 `DocumentID`, content checksum, tokenizer revision, token offset과 소비 기록이다.

**Raw identity.** URL은 시간이 지나며 내용이 바뀌고 같은 content가 여러 URL에 나타난다. identity에는 crawl snapshot, WARC/WET locator, fetch timestamp, response/body checksum을 포함한다. canonical URL은 검색 key이지 유일한 content identity가 아니다. redirect와 query normalization 정책도 revision으로 둔다.

| 단계 | 입력 | 출력 | 필수 provenance | silent failure |
|---|---|---|---|---|
| collection | crawl manifest | raw records | snapshot·offset·SHA | URL 재사용 |
| extraction | HTML/WET | main text | extractor·offset map | 메뉴/본문 혼합 |
| language | text | score/tag | model·threshold | code·혼합언어 탈락 |
| quality | features | score/reason | feature/model revision | domain 편향 |
| PII/safety | spans/document | tag/redaction/drop | rule·span | 과잉 삭제·잔존 |
| materialize | accepted docs | shards | ordered IDs·SHA | 중복·누락 |

**Extraction도 목적함수를 바꾼다.** boilerplate를 남기면 반복 menu가 token mass를 차지하고, 과도한 extractor는 표·code·caption을 잃는다. text만 저장하면 원 HTML 어느 구간이었는지 삭제와 오류 조사에 답하기 어렵다. raw byte→extracted character interval map, extractor config digest, reject reason을 sidecar로 둔다.

**Threshold의 수학.** score `s(x)`와 threshold `t`에서 보존 집합은 `K_t={x:s(x)≥t}`다. `t`가 오르면 분류기 기준의 low-score 문서는 단조롭게 줄지만, 사람이 원하는 품질이나 downstream 성능까지 단조롭게 좋아진다는 보장은 없다. 여기서 false positive는 정책상 버려야 할 문서를 보존한 경우인지, detector가 위험하다고 잘못 판정해 깨끗한 문서를 버린 경우인지 먼저 양성 클래스부터 정의해야 한다. 이 장에서는 혼동을 피하려고 `clean-but-rejected`와 `harmful-but-retained`처럼 실제 결과를 함께 적는다. domain별 score 분포와 두 오류를 사람이 층화 감사한다. score를 먼저 materialize해야 같은 document universe에서 threshold만 sweep할 수 있다.

DCLM은 quality classifier accuracy만 보지 않고 고정 training/evaluation protocol의 downstream model 성능으로 filter와 threshold를 선택한다. 그렇다고 특정 threshold가 다른 crawl, language, tokenizer에 보편적으로 이식되는 것은 아니다. raw pool과 compute/eval revision을 함께 고정한다.

**PII와 toxicity.** email·전화번호 span redaction, 유해 표현 score, bad-word document drop은 목적이 다르다. blocklist는 정체성 표현과 인용 문맥을 과잉 제거할 수 있다. placeholder 치환은 문서를 보존하지만 원 offset과 접근 정책이 필요하다. toolkit이 tagger를 제공한다고 특정 release가 모든 PII를 제거했다고 쓰지 않는다.

**반례 1.** 문서 수가 줄었다고 학습 질량도 같은 비율로 줄었다고 볼 수 없다. 짧은 문서를 대거 버리고 긴 문서를 남기면 retained-document 비율은 낮아져도 retained-token 비율은 더 높을 수 있다. 따라서 document, raw byte, tokenizer-specific token을 stage별로 따로 세고, 학습 영향은 마지막에 실제 소비된 loss-bearing token으로 확인한다.

**반례 2.** 영어 score가 낮은 문서는 다른 언어뿐 아니라 code·math·표일 수 있다. language threshold를 quality 진실값으로 쓰지 않는다.

**실패 주입 4-A.** 같은 raw record를 extractor 두 revision으로 처리해 checksum, length, source offset coverage, filter score를 비교한다. 첫 차이를 extraction stage에 고정한다.

**중복은 확률 문제다.**

### 4.1.2 최초 수집부터 exact/fuzzy identity를 준비한다

exact hash는 동일 bytes만 잡는다. normalization 뒤 hash는 공백과 boilerplate 변형을 더 잡지만 normalization revision이 바뀌면 집합도 달라진다. MinHash LSH의 후보 확률은 similarity `s`, band 수 `b`, band당 row `r`에서 `1-(1-s^r)^b`다. 이는 hard threshold가 아니다. 후보 생성 뒤 실제 similarity 판정과 survivor 선택 규칙이 필요하다.

Bloom filter는 compact membership test지만 false positive가 있다. 데이터 오염 제거에 쓰면 깨끗한 문서도 탈락할 수 있다. bit 수, hash 수, 삽입 수와 예상 false-positive rate를 기록한다.

**Exact key를 명시한다.** raw bytes SHA, Unicode-normalized text SHA, whitespace-canonical SHA, URL key는 다른 중복 정의다. raw exact는 banner 차이를 놓치고 강한 normalization은 code indentation이나 숫자를 합칠 수 있다. key function revision과 survivor policy를 분리한다.

**MinHash 유도.** shingle 집합 A/B의 Jaccard는 `J=|A∩B|/|A∪B|`다. random permutation 아래 두 집합의 최소 원소가 같을 확률은 합집합 최소가 교집합에 속할 확률과 같아 J다. r개 hash가 모두 같은 band 확률은 `s^r`, b개 band 중 하나라도 같은 후보 확률은 `1-(1-s^r)^b`다.

FineWeb의 `b=14,r=8`은 hard cutoff가 아니다. candidate 생성 뒤 pair 판정, connected component, survivor 선택이 남는다. A~B와 B~C가 연결되면 A/C similarity가 낮아도 같은 cluster가 될 수 있다.

| artifact | 역할 | 필수 필드 |
|---|---|---|
| normalized key | exact 비교 | function revision·SHA |
| signature | fuzzy candidate | n-gram·seed·b/r |
| edge | pair 판정 | IDs·similarity |
| cluster | component | cluster revision |
| survivor | 보존 정책 | survivor·reason·score |
| reject sidecar | 역추적 | removed→survivor |

**Survivor는 분포 정책이다.** 최초/최신 crawl, 가장 긴 문서, 최고 quality 중 무엇을 남기느냐에 따라 time/domain mass가 달라진다. FineWeb에서 global dedup이 더 많이 제거해도 dump-local MinHash보다 downstream 결과가 좋지 않았다는 관찰은 “더 강한 dedup이 항상 낫다”를 깨뜨린다.

**Bloom 식.** m bits, n insertions, k hashes에서 FP는 근사적으로 `(1-e^{-kn/m})^k`, 최적 k는 `(m/n)ln2`다. 실제 n이 capacity를 넘으면 FP가 증가한다. 이론상 false negative가 없지만 normalization/segmentation이 바뀌면 같은 원문이 다른 key가 되는 외부 false negative가 생긴다. binary digest와 inserted manifest를 저장한다.

**반례 3.** URL dedup은 mirror content를 놓치고 갱신된 같은 URL을 합칠 수 있다. URL과 content cluster를 별도로 둔다.

**반례 4.** line dedup은 순서가 바뀐 near duplicate를 놓치고 공통 license header를 지우며 code 문맥을 훼손할 수 있다.

**실험 4-B.** 동일 raw slice에서 exact, URL, dump-local MinHash, global MinHash를 적용하고 retained docs/tokens/domain, cluster size와 fixed-token downstream loss를 비교한다.

**실험 4-C.** known inserted/non-inserted key로 Bloom empirical FP/FN을 재고 capacity 초과, seed 변경, normalization 변경을 분리한다.

## 4.2 license·provenance·삭제 책임을 원장에 고정한다

### 4.2.1 decontamination과 삭제 lineage

benchmark 문자열과 비슷한 문서를 제거했다는 것만으로 contamination이 사라졌다고 말할 수 없다. match normalization, n-gram 길이, threshold, split 공개 시점, 검사한 corpus revision이 필요하다. 더 나아가 raw object→extracted document→dedup survivor→token shard offset→packed sample→optimizer step→checkpoint의 역색인이 있어야 삭제 요청의 영향을 판정할 수 있다.

**Detector 단위.** exact string, token n-gram, paragraph Bloom, MinHash, semantic embedding은 recall/precision과 비용이 다르다. benchmark answer만 제거할지 prompt까지 제거할지, train/test split 공개 시점을 정한다. text n-gram과 token-ID n-gram은 normalization과 tokenizer에 따라 다르다.

**Hit와 소비는 다르다.** corpus에 benchmark span이 있어도 filter/mixture에서 선택되지 않았거나 truncation·mask 뒤 loss에 기여하지 않았을 수 있다. 반대로 near duplicate가 detector를 피하고 소비될 수 있다. hit ledger를 `ConsumptionID`와 연결해 loss-bearing exposure를 판정한다.

Canonical chain은 `RawRecordID→ExtractedDocumentID→NormalizedDocumentID→DedupCluster/Survivor→TokenSegment(shard,offset,length)→PackedSampleID→BatchDrawID→UpdateID→CheckpointID`다. 각 edge는 transform revision과 input/output checksum을 갖는다.

**Tombstone.** 삭제 요청을 검증하면 먼저 `RevocationID`를 만들고, 대상 identity를 “다음 세대에서 소비해서는 안 되는 상태”로 바꾼다. tombstone은 payload를 대신하는 빈 파일도, 이미 일어난 학습을 취소하는 마법도 아니다. raw/normalized ID를 resolve하고 dedup loser→survivor와 mirror 관계를 따라가며, 다음 corpus release에서 제외한다. 이어 token/packed shard와 mixture capacity를 새 generation으로 만들고, 영향받은 checkpoint·adapter·quantized descendant를 표시한다. 이때 storage 삭제, 향후 loader 차단, 기존 model 대응은 서로 다른 완료 상태다. parameter에 흡수된 정보를 파일 하나를 지웠다는 이유로 제거했다고 주장하지 않는다.

| 대상 | 즉시 조치 | 재생성 | 보장하지 못함 |
|---|---|---|---|
| raw doc | future manifest 제외 | extracted/token shard | 기존 model forgetting |
| survivor | cluster 재선택 | downstream shard | 같은 distribution |
| token segment | loader 차단 | packed sample/index | 기존 optimizer 영향 제거 |
| checkpoint | quarantine/표기 | retrain/unlearning 검토 | exact subtraction |

문서가 update `u`에서 소비됐다면 그 뒤의 descendant checkpoint는 직접 또는 간접 영향을 받았을 수 있다. 영향은 parameter뿐 아니라 optimizer moment, loss scale, curriculum cursor와 뒤따른 sample 순서에도 번진다. 학습 trajectory는 비선형이므로 한 sample의 gradient를 사후에 빼서 원래 경로를 복원할 수 없다. 그래서 계보가 답하는 것은 우선 “어디까지 영향 가능성이 전파됐는가”이며, 실제 대응은 격리·재학습·선택적 unlearning·serving policy 가운데 별도로 판정한다.

**반례 5.** 요청 문서가 dedup reject여도 survivor가 같은 내용을 담을 수 있다. reject→survivor edge를 따른다.

**반례 6.** source token shard를 지워도 materialized packed cache와 object-store replica가 남을 수 있다.

**실패 주입 4-D.** URL 하나를 tombstone하고 raw record, mirror cluster, token offsets, packed samples, consumption과 checkpoint까지 역추적한다. 어느 지점에서 조회가 끊기면 삭제 성공으로 세지 않고 `UNRESOLVED_EDGE`로 기록한다. 이어 새 release loader에서 대상이 거부되는지, old release alias가 production selector에서 빠졌는지, backup 복원 뒤 tombstone journal이 다시 적용되는지를 각각 시험한다.

**실험 4-E.** exact, 8/13-gram, MinHash, semantic detector hit와 human precision을 비교하고 clean eval 변화와 removed-domain mass를 분리한다.

**파이프라인 이름보다 경계를 비교한다.**

### 4.2.2 공개 corpus의 권리·공정 경계를 비교한다

FineWeb/DataTrove와 Dolma는 공개 curation code가 강하고, DCLM은 데이터 선택을 downstream model 성능과 연결하는 계약이 강하다. C4는 중요한 baseline이지만 blocklist bias와 lineage 한계가 있다. 이 비교는 “어느 데이터셋이 최고인가”가 아니라 어느 lifecycle 단계가 공개되고 재현 가능한가를 묻는다.

**C4.** 2019-04 Common Crawl WET와 terminal punctuation, word count, boilerplate, bad-word rule의 역사적 baseline이다. TFDS commit `00a6c1cbe049634e1cfb823a910b83d6cb358ac2`의 text builder와 T5 paper를 함께 읽는다. 간결하지만 blocklist bias, full provenance와 deletion 전파가 닫히지 않는다.

**FineWeb.** 96개 dumps, fastText 영어 threshold, Gopher/C4/custom filter ablation, dump별 5-gram MinHash를 공개한다. DataTrove `a649de79c14a550dc90f48a15c025f2dd3fd3b57`의 `examples/fineweb.py`가 고정 진입점이다. `skip_completed`는 task resume이지 trainer sample cursor가 아니다.

**Dolma.** document/span attribute sidecar, tagger와 mixer, exact/paragraph Bloom dedup·decontam config가 강점이다. commit `669f534823b08d266a8fff01f8a1c916a5a56576`에 고정한다. Bloom FP와 release rights, trainer resume는 toolkit만으로 보장되지 않는다.

**DCLM.** Common Crawl pool, mapper/filter, BFF fuzzy dedup과 고정 compute/eval protocol을 연결한다. commit `361714bdd60bb9b7f4b2d8354cebbf0dec0c329e`를 쓴다. Ray exact-dedup 코드가 있어도 baseline recipe가 사용했다고 혼합하지 않는다.

| 질문 | C4 | FineWeb | Dolma | DCLM |
|---|---|---|---|---|
| curation code | builder | 강함 | 강함 | 강함 |
| sidecar | 제한 | stage 일부 | 강함 | intermediate |
| model selection protocol | baseline | ablation | 별도 | 핵심 |
| sample-exact resume | 미폐쇄 | 미폐쇄 | 미폐쇄 | 미폐쇄 |
| deletion→checkpoint | 미폐쇄 | 미폐쇄 | 미폐쇄 | 미폐쇄 |

옵션은 state diff로 읽는다. language/quality threshold는 retained set, MinHash n/b/r은 candidate graph, Bloom FPR은 bit array/hash 수, scope는 cluster universe, shard count는 task assignment를 바꾼다.

**Upstream test 경계.** transform unit test는 함수 invariant를 확인하지만 수십 TB release의 shard completeness와 rights를 증명하지 않는다. smoke 성공은 model quality 근거가 아니다. DCLM result는 고정 protocol 안의 data effect이며 다른 optimizer/model로 자동 이전하지 않는다.

**조사 체크리스트.** raw snapshot·rights를 고정한다. stage DAG와 ID를 그린다. filter score/reason/threshold를 찾는다. dedup key/signature/cluster/survivor를 확인한다. benchmark manifest를 고정한다. shard checksum/count를 검증한다. tokenizer/packing/consumption ID 연결과 deletion SLA를 본다.

**결정 트리.** 결과가 다를 때는 가장 뒤의 metric부터 추측하지 말고 최초로 갈라진 artifact를 찾는다. document count가 다르면 raw manifest→logical task completion→accepted/rejected/error 보존식을 본다. count는 같고 bytes가 다르면 extractor·normalizer·ordering을 고정해 paired replay한다. dedup 결과만 다르면 key와 seed, candidate edge, component, survivor tie-break를 차례로 비교한다.

token만 다르면 동일 normalized checksum에 tokenizer revision을 교차 적용한다. mixture만 다르면 eligible inventory, logical length, packing과 sampler event를 나눈다. checkpoint 역추적이 안 되면 “영향 없음”이 아니라 끊긴 lineage edge를 명시한다.

| 관찰값 | 가능한 최초 원인 | 분리 실험 | 판정 증거 |
|---|---|---|---|
| raw count부터 다름 | crawl 범위·task 누락·retry 중복 | 동일 crawl manifest로 task set 재계산 | expected/chosen task manifest |
| count는 같고 text SHA가 다름 | extractor·normalizer·순서 | 같은 raw IDs의 old/new paired replay | first-divergence span map |
| survivor만 다름 | signature·candidate·component·tie-break | edge set과 ranking tuple 독립 비교 | loser→winner decision record |
| token 수만 다름 | tokenizer·special token·normalization | 같은 text SHA에 tokenizer만 교차 적용 | tokenizer digest·token interval |
| planned mixture는 같고 소비 질량이 다름 | availability·packing·sampler·resume | event ledger로 요청/선택/실패를 재생 | UpdateID별 valid-token mass |
| tombstone 뒤에도 sample이 나옴 | stale manifest·cache·replica·reverse index 누락 | cold loader와 backup restore probe | tombstone root·resolver denial |

**실제 인계.** 5장에 corpus/document bytes와 offsets, 6장에 accepted manifest·source/domain·length/capacity, 24장에 contamination hit·benchmark revision, 23·27장에 RevocationID와 descendant index를 넘긴다.

**상태 기계를 파일로 내린다.** 권장 전이는 `RAW_MANIFESTED→EXTRACTED→TAGGED→DEDUP_INDEXED→FILTERED→DECONTAMINATED→TOKENIZED→PACKED→MIXED→CONSUMED`다. 각 stage manifest에는 input manifest digest, code/config revision, output shard digest, accepted/rejected count, bytes와 token count를 기록한다. output 몇 개가 존재한다는 이유로 stage complete를 선언하지 않는다. 모든 expected task의 commit marker와 aggregate count가 맞아야 한다.

**Commit과 retry.** worker가 임시 shard를 쓰고 checksum을 계산한 뒤 immutable final key와 task marker를 publish한다. retry는 같은 task input에서 같은 ordered DocumentID set을 내거나 기존 valid output을 재사용해야 한다. 두 worker가 같은 task를 완료하는 speculative execution에서는 manifest compare-and-swap 또는 deterministic output identity가 필요하다. append-only output에 재시도하면 duplicate document가 생긴다.

**분산 partition.** `hash(DocumentID) mod N`은 worker 수 N이 바뀌면 assignment가 대부분 달라진다. 제조 결과 set이 같아도 shard layout/order가 달라질 수 있다. release identity가 ordered shards까지 포함하는지 document set만 포함하는지 정한다. trainer sample-exact 소비가 shard order에 의존하면 제조 reshard가 학습 trajectory를 바꾼다.

**Count reconciliation.** 단일 입력을 terminal state 하나로 보내는 stage라면 `input=accepted+rejected+error`가 성립해야 한다. split/merge transform에는 이 식을 그대로 쓰지 않고 parent-child edge의 cardinality와 orphan 수를 검산한다. byte와 token은 손실 있는 변환 때문에 보존되지 않을 수 있으므로, 차이를 오류로 몰기 전에 어느 규칙이 얼마를 더하거나 뺐는지 reason별로 설명한다. 반대로 error를 reject에 숨겨 식만 맞추면 안 된다. transient fetch failure는 재시도할 수 있지만 quality reject는 같은 policy 아래 의도된 terminal state이기 때문이다.

| counter | 의미 | 흔한 오독 |
|---|---|---|
| raw records | crawl 입력 | unique URL/doc로 오독 |
| extracted docs | extractor output | content unique로 오독 |
| survivor docs | dedup 보존 | 원 정보량으로 오독 |
| normalized bytes | 정제 text 크기 | 원 crawl bytes로 오독 |
| tokenizer tokens | 특정 revision | tokenizer 독립 크기로 오독 |
| supervised tokens | mask 뒤 target | attention compute token과 혼동 |

**Quality ablation의 공정성.** filter A/B 비교에서 raw universe, tokenizer, model config, optimizer, token budget, evaluation을 고정한다. retained corpus 크기가 다르면 같은 epoch가 아니라 같은 consumed token budget을 사용해야 data quality와 양을 분리한다. 작은 corpus를 더 반복하면 repetition 효과가 들어오므로 unique token과 repeat count를 기록한다.

**Domain drift.** global threshold 하나는 news, forum, code, academic text에 다른 acceptance rate를 만든다. overall quality score가 올라가도 특정 domain이 사라질 수 있다. domain classifier 자체도 불완전하므로 raw source, URL host, content classifier의 여러 view를 함께 본다. mixture가 후속 sampler에서 보정될 수 있어도 제거된 document는 되살릴 수 없다.

**Temporal dedup trade-off.** 여러 crawl dump의 같은 URL·기사에는 수정·업데이트가 있을 수 있다. global dedup은 유용한 변화와 time coverage를 줄이고, dump-local dedup은 반복 exposure를 남긴다. 수정 timestamp와 diff를 보존해 exact duplicate, near duplicate, updated version을 구분한다. survivor policy가 최신성을 의도하는지 diversity를 의도하는지 쓴다.

**MinHash seed와 reproducibility.** hash family와 seed가 바뀌면 signature와 candidate edge가 바뀐다. 병렬 처리 순서가 survivor tie-break에 들어가면 같은 seed에서도 result가 달라질 수 있다. candidate edge를 정렬하고 stable DocumentID 기준으로 cluster representative를 선택한다. graph library component ordering을 release identity로 쓰지 않는다.

**LSH parameter sweep.** b를 늘리면 candidate recall과 비용이 대체로 늘고, r을 늘리면 높은 similarity를 더 요구한다. n-gram 길이가 짧으면 boilerplate 공통성이 similarity를 올리고, 길면 작은 편집에도 overlap이 급감할 수 있다. sampled exact Jaccard와 human duplicate label로 S-curve를 calibration한다.

**Bloom deletion 문제.** 일반 Bloom filter는 원소를 안전하게 삭제하기 어렵다. bit 하나를 지우면 다른 원소 membership까지 깨진다. counting Bloom이나 index 재빌드가 필요하다. deletion request에서 corpus document만 지우고 dedup/decontam filter binary를 그대로 두면 future processing의 설명 가능성이 떨어진다. filter generation을 descendant로 등록한다.

**Benchmark versioning.** 평가 benchmark도 수정되고 contamination detection용 normalization도 바뀐다. dataset name만 저장하지 말고 split row IDs, raw prompt/answer checksum, release date, detector config를 고정한다. 공개 이전 corpus와 공개 이후 corpus를 시간상 구분하되 web page의 사전 유출과 mirror 가능성을 별도로 본다.

**Semantic contamination의 함정.** embedding similarity는 paraphrase를 잡지만 같은 주제의 깨끗한 교육 문서를 과잉 제거할 수 있다. threshold를 인간 label과 calibration하고 exact/n-gram detector와 교집합·차집합을 본다. semantic detector model이 평가 benchmark를 학습했을 가능성도 provenance에 포함한다.

**라이선스와 접근권.** curation code license, dataset distribution 조건, 원 crawl content의 권리는 같은 것이 아니다. 공개 script를 실행할 수 있다는 사실이 결과 document의 재배포 권리를 만들지 않는다. 책의 기술 분석에서는 각 artifact의 license locator와 접근 시점을 source note에 두고, 제거·opt-out 경로를 lineage와 연결한다.

**보안 경계.** raw HTML과 compressed archive는 parser bug, decompression bomb, path traversal, malicious payload를 포함할 수 있다. extraction sandbox, size/time limit, content type 검증을 둔다. text 안의 prompt injection은 pretraining data로서 별도 safety 문제다. curation worker의 secret과 network access를 제한한다.

**실패 주입 4-F—partial shard.** output payload 절반만 쓰고 worker를 종료한다. restart가 file existence만 보고 skip하는지 checksum/commit marker로 재실행하는지 본다. aggregate manifest가 partial을 release에 포함하지 않아야 한다.

**실패 주입 4-G—duplicate retry.** task 완료 응답 직전에 worker를 죽여 scheduler가 retry하게 한다. 같은 DocumentID가 두 output shard에 나타나지 않는지 global reconciliation으로 검사한다.

**실패 주입 4-H—world-size 변경.** worker 8에서 16으로 resume해 final survivor set과 ordered manifest를 비교한다. set equality와 shard/layout equality를 따로 판정한다. 후속 trainer fixture에서 다음 sample IDs가 유지되는지도 별도다.

**실패 주입 4-I—threshold config만 교체.** score artifact는 고정하고 threshold만 바꾼다. extractor/tagger가 재실행되지 않아야 하며 before/after reject reason과 domain/token delta가 재현돼야 한다. score와 filter를 한 transform에 묶으면 이 실험이 불가능하다.

**Test pyramid.** unit test는 normalizer, feature, filter predicate, MinHash/Bloom 수학을 검사한다. property test는 determinism, count reconciliation, no-future ID collision을 본다. integration test는 작은 raw fixture의 전체 stage와 retry를 실행한다. release audit는 모든 shard checksum과 aggregate counts를 검증한다. downstream ablation은 data effect를 본다. 아래 test 통과가 위의 quality 결론을 대신하지 않는다.

**Fixture corpus.** exact duplicate, whitespace variant, reordered paragraph, mirror URL, updated article, multilingual/code/math, PII span, benchmark exact/paraphrase, malformed HTML, very long document를 포함한다. 각 문서의 expected stage result와 survivor/offset lineage를 사람이 검토해 golden JSONL로 둔다.

**성능 측정.** docs/sec만 보고 correctness를 희생하지 않는다. bytes read/written, parser time, tagger batch utilization, candidate edges, cluster memory, object-store requests를 stage별로 본다. skipped completed task는 처리량 분모에서 제외한다. cache hit와 cold run을 구분한다.

**오류 조사 기록철.** 문제 DocumentID의 raw locator/checksum, 각 transform input/output slice, score/reasons, dedup edges/cluster/survivor, shard/offset, 소비 history를 한 묶음으로 export한다. 전체 corpus payload를 복사하지 않고 필요한 span과 checksum을 남긴다. 원문 접근 제한도 그대로 보존한다.

**현실적인 종료 조건.** 공개 자료만으로 다섯 corpus의 deletion→checkpoint 전체 폐루프와 trainer sample-exact resume는 닫히지 않는다. 이를 숨기지 않은 상태에서도 corpus 제조 장은 완성될 수 있다. 독자가 공개된 transform을 재현하고, 공개되지 않은 production 보장을 별도 결손으로 판정하며, 자기 pipeline에 필요한 ID와 test를 설계할 수 있으면 된다.

**릴리스 판정 기준.** expected task 전부가 committed됐는지 확인한다. raw input manifest와 output manifest digest를 고정한다. stage count reconciliation과 error budget을 검토한다. sampled raw→final offset을 역추적한다. dedup cluster/survivor 결정성을 재실행한다. benchmark hit와 removal ledger를 검토한다. license·PII·opt-out 상태를 확인한다. tokenizer 전환 전 accepted DocumentID 목록을 immutable하게 publish한다.

**학습 실행 연결 기준.** tokenizer shard가 어느 corpus release를 읽었는지 확인한다. token segment마다 DocumentID/offset index가 있는지 본다. packer가 segment map을 보존하는지 확인한다. sampler가 출처/도메인과 cursor를 기록하는지 본다. optimizer UpdateID가 consumed sample IDs를 가리키는지 확인한다. CheckpointID가 consumption high-water mark와 atomic하게 commit되는지 본다.

**삭제 조사 결정 트리.** 요청 identifier가 URL뿐이면 crawl manifest에서 모든 raw records를 resolve한다. content checksum으로 mirror를 찾는다. normalized/dedup graph에서 reject와 survivor를 찾는다. token/packed reverse index를 조회한다. consumption이 없으면 future artifact만 revoke한다. consumption이 있으면 영향 RunID/CheckpointID를 표시한다. 후손 adapter·merge·quantization을 따라 release gate를 막는다. unlearning/retrain 결과는 별도 EvalID로 검증한다.

**Quality regression 결정 트리.** downstream loss가 나빠졌다면 retained token budget과 repetition부터 맞춘다. domain/language mixture를 비교한다. removed document score·reason을 sample audit한다. dedup cluster size와 survivor policy를 본다. tokenizer fertility와 truncation을 확인한다. contamination 변화와 clean validation 변화를 분리한다. filter 하나씩 되돌리는 ablation으로 첫 원인을 찾는다.

**책에서 수치를 인용하는 규칙.** FineWeb·DCLM 논문의 token 수와 성능은 해당 tokenizer, model, budget, evaluation 조건에 붙인다. 서로 다른 paper의 숫자를 같은 표에서 우열로 정렬하지 않는다. 공개 code가 paper production run과 다르다고 명시하면 구현 설명과 실험 결과를 두 provenance로 나눈다. 현재 checkout에서 재실행하지 않은 값은 원문 보고값으로 표시한다.

**독자 실습 4-J—작은 제조 pipeline.** fixture 20개를 raw manifest로 만들고 extraction, language/quality tag, exact/MinHash dedup, benchmark Bloom, materialization을 단계별 실행한다. 각 stage JSONL과 reject sidecar를 hash한다. 한 threshold와 dedup seed만 바꾼 child run을 만들어 manifest diff를 낸다. URL 하나를 tombstone하고 모든 descendant status를 확인한다.

**독자 실습 4-K—두 분모.** document 50/30/20 source mixture를 만들되 평균 길이를 다르게 한다. document 비율, emitted token 비율, supervised token 비율을 계산한다. 이 결과를 6장 sampler의 입력으로 넘긴다. corpus 제조 단계에서 “50% 데이터”라는 표현이 어느 분모인지 쓰지 않으면 실험을 실패 처리한다.

**4장 최종 handoff manifest.** `CorpusRevision`, ordered shard list/checksum, DocumentID generation revision, raw/extracted/normalized checksum과 interval map, source/domain tags, quality score/reason, dedup cluster/survivor, contamination hit, RevocationID status, logical length/capacity를 포함한다. 5장은 bytes와 offsets를 tokenizer 좌표로 바꾸고, 6장은 source와 length를 실제 packed/supervised mixture로 바꾼다.

**확인 문제.** MinHash `b,r`를 바꿨을 때 후보 recall과 cluster graph가 왜 함께 변하는지 식으로 설명한다. Bloom FP가 deletion lineage에 어떤 잘못된 reject를 만드는지 적는다. dump-local dedup과 global dedup을 같은 token budget으로 비교해야 하는 이유를 설명한다. task resume와 trainer resume가 저장해야 하는 상태를 각각 열거한다.

답에는 “중복 제거율” 하나만 쓰지 않는다. 제거된 document·token·domain mass, survivor rule, downstream repetition, detector FP/FN을 함께 쓴다. 삭제 요청에는 raw file 삭제 외에 filter index, packed cache, checkpoint와 후손 artifact의 상태를 적는다.

**입문 경로에서 먼저 확인할 보존식.** 모든 accepted document는 하나의 raw ancestor와 transform revision을 가져야 한다. 모든 reject는 reason 또는 survivor를 가져야 한다. release manifest count는 committed shard 합과 맞아야 한다. tombstoned ID는 다음 release와 loader에서 소비되지 않아야 한다. 공개 evidence가 확인하지 못한 기존 checkpoint forgetting은 통과로 표시하지 않는다.

**이 장이 넘기는 것.** `CorpusRevision`, `DocumentID`, raw/normalized checksum, offset map, dedup survivor ID를 5장에 넘긴다.

**다음 장에서 깨질 수 있는 것.** tokenizer normalization이 corpus normalization을 다시 수행하면 offset과 삭제 lineage가 어긋난다.

**검증 체크포인트.** 같은 input과 revision에서 survivor set이 결정적이며, 삭제한 DocumentID가 token shard 역색인에서 발견되는지 확인한다.

## 4.3 공개 pipeline을 parse·normalize stage 계약으로 읽는다

### 4.3.1 예제 파일을 여러 job의 계약으로 복원한다

기준 DataTrove checkout은 commit `a649de79c14a550dc90f48a15c025f2dd3fd3b57`이다. `examples/fineweb.py:1-33`은 reader, filter, token counter, writer와 MinHash 구성 요소를 import하고 저장 위치와 dump를 정한다. 이 파일을 “FineWeb 구현” 한 덩어리로 읽지 않는다. extraction/filtering job, signature job, bucket job, cluster job, remove-ID를 적용하는 filter job이 서로 다른 materialized artifact를 주고받는다.

`examples/fineweb.py:80-91`의 `MinhashConfig`는 hash 수, bucket 수, n-gram 크기를 고정한다. `107-118`의 `MinhashDedupSignature`는 document에서 signature를 만들고, `123-133`의 buckets 단계는 같은 bucket 후보를 모은다. `144-154`의 cluster 단계는 candidate graph를 component로 만들며, `166`의 `MinhashDedupFilter`가 remove IDs를 실제 corpus에 적용한다. “MinHash를 켰다”는 옵션 하나가 아니라 signature→candidate→cluster→survivor→filter의 다섯 상태다.

각 경계 manifest에는 input DocumentID와 normalized-text checksum, n-gram/hash config, signature locator, bucket keys, candidate edges, component ID, survivor rule, remove reason을 둔다. 최종 survivor만 보존하면 왜 제거됐는지, config 변경 때 어느 단계부터 재사용할지 알 수 없다. signature config가 같고 survivor 정책만 바뀌었다면 extraction부터 다시 실행할 이유는 없다.

`src/datatrove/data.py:38-57`의 `Document` dataclass는 text, id, media, metadata를 pipeline의 공통 운반 단위로 만든다. 그러나 이 Python object가 영속 계보를 자동 보장하지 않는다. writer가 어느 필드를 materialize하는지, ID가 stage에서 유지되는지, metadata가 serialization 뒤 살아 있는지 확인한다.

### 4.3.2 skip_completed가 보장하는 범위를 확인한다

`src/datatrove/executor/base.py:54-59`는 executor가 `skip_completed`를 상태로 가지며, `162-165`의 `is_task_complete`는 `completions/{rank:05d}` marker 존재를 확인한다. 이는 data 제조 task resume다. trainer의 sampler cursor나 optimizer update resume가 아니다. 같은 `resume`라는 단어를 사용해도 저장하는 state가 완전히 다르다.

marker가 올바름을 증명하려면 payload checksum과 task input/config digest에 묶여야 한다. 예전 config의 marker가 새 run을 가리키거나 payload 일부만 쓴 뒤 marker가 남으면 task가 조용히 skip될 수 있다. `jobs.py:316`의 주석처럼 이전 marker가 동작을 가릴 수 있는 경로를 읽고, run namespace와 immutable manifest를 둔다.

task 완료 조건은 output file 존재가 아니다. expected input partition 전부 소비, accepted/rejected/error count reconciliation, payload close와 checksum, marker publish까지다. worker 종료 실패를 payload 쓰기 전, 중간, close 뒤, marker 전후에 주입한다. restart가 incomplete task만 재실행하고 duplicate output을 만들지 않는지 본다.

## 4.4 exact·near dedup을 확률과 graph로 검산한다

### 4.4.1 Jaccard 추정과 LSH 후보 확률

두 document의 n-gram 집합을 A,B라 하면 Jaccard는 `J=|A∩B|/|A∪B|`다. minwise hash에서 한 permutation의 최소 원소가 같은 확률은 J다. k개 hash의 일치 비율은 J의 추정량이며 variance는 대략 `J(1-J)/k`다. k를 늘리면 추정 노이즈가 줄지만 signature 저장과 계산 비용이 증가한다.

signature를 b개 band, band마다 r개 row로 묶으면 similarity s인 두 문서가 적어도 한 band에서 후보가 될 확률은 `1-(1-s^r)^b`다. threshold가 딱 하나로 잘리는 것이 아니라 S자 확률 곡선이다. b와 r은 recall, candidate 수, shuffle I/O, cluster memory를 동시에 바꾼다. production 값은 paper의 관례가 아니라 sample label에서 false merge와 miss 비용을 calibration해 정한다.

n-gram 길이도 별도 축이다. 짧으면 공통 boilerplate 때문에 관련 없는 문서가 닮아 보이고, 길면 작은 편집이나 문단 재배치에 overlap이 급감한다. word와 character n-gram은 언어·tokenization 민감도가 다르다. normalizer가 whitespace와 punctuation을 어떻게 바꾸는지도 signature 의미에 들어간다.

### 4.4.2 candidate edge에서 survivor를 선택한다

LSH는 duplicate 판정을 확정하지 않고 비교할 후보를 만든다. candidate graph의 connected component를 cluster로 삼으면 A≈B, B≈C이나 A와 C가 threshold 아래여도 셋이 묶이는 transitive chaining이 생긴다. cluster 크기가 큰 boilerplate hub에서는 이 효과가 커진다. component 내부 pair score 분포와 대표까지의 similarity를 audit한다.

survivor를 “첫 문서”로 고르면 병렬 순서가 release 결과를 바꿀 수 있다. stable DocumentID, quality score, timestamp, source priority를 명시적으로 tie-break한다. 최신 문서를 고르면 수정본 보존에 유리하지만 historical diversity가 줄어들 수 있다. 최고 quality score는 classifier bias를 증폭할 수 있다. 정책은 제거율이 아니라 후속 distribution을 바꾼다.

graph artifact에는 edge 근거와 config를 남긴다. cluster ID만 있으면 false merge를 분해하기 어렵다. 삭제 요청이 reject 문서를 가리켜도 survivor가 같은 내용을 보존할 수 있으므로 reject→survivor edge가 필요하다.

### 4.4.3 exact와 fuzzy의 적용 순서를 검증한다

exact content hash는 싸고 설명 가능하다. 먼저 exact duplicate를 줄이면 MinHash candidate universe와 비용이 작아진다. 다만 normalization 뒤 hash인지 raw byte hash인지 구분한다. raw byte가 다른 whitespace variant는 남고, 강한 normalization은 의미 있는 code/수식 차이를 지울 수 있다.

URL dedup은 content dedup이 아니다. 하나의 URL이 시간에 따라 바뀌고 같은 content가 mirror URL에 나타난다. URL은 source locator, content checksum은 payload identity, normalized checksum은 transform identity로 따로 둔다. 이 셋을 한 `id` 열에 덮어쓰지 않는다.

## 4.5 contamination·PII·safety gate의 손실을 구분한다

### 4.5.1 Bloom false positive를 제거 질량으로 센다

bit 배열 크기가 `m`, hash 함수 수가 `k`, 삽입 원소 수가 `n`인 Bloom filter의 false-positive 확률은 근사적으로 `(1-e^{-kn/m})^k`다. 정상 구현의 membership query에는 자료구조 내부 false negative가 없지만 false positive는 있다. 즉 삽입하지 않은 깨끗한 n-gram도 “이미 있음”으로 답할 수 있다. decontamination 정책이 이 응답을 제거 명령으로 바꾸면 자료구조의 false positive가 곧 `clean-but-rejected` 학습 질량이 된다. FPR 하나만 보고 안전하다고 판정하지 않고, 실제 query 수를 곱한 예상 오제거 수, 길이별 query 수, domain별 empirical negative fixture를 함께 본다.

Dolma 기준 checkout은 `669f534823b08d266a8fff01f8a1c916a5a56576`이다. `python/dolma/cli/deduper.py:58-96`은 Bloom config와 document/paragraph dedup configuration을 CLI schema로 드러낸다. Python 진입점 `python/dolma/__init__.py:26-40`은 config를 Rust pipeline에 넘긴다. 실제 hashing과 storage semantics는 Rust 구현까지 따라가야 하며 CLI field만으로 알고리즘을 추정하지 않는다.

일반 Bloom filter는 안전한 삭제를 지원하지 않는다. 공유 bit를 지우면 다른 원소까지 false negative가 된다. benchmark 수정이나 opt-out에서 counting Bloom 또는 generation 재빌드가 필요하다. filter binary도 dataset release의 descendant artifact로 등록한다.

### 4.5.2 contamination을 네 evidence 수준으로 나눈다

exact answer 문자열 일치는 가장 설명하기 쉽지만 paraphrase와 번역된 유출을 놓친다. n-gram overlap은 부분 복사를 잡는 대신 흔한 지시문과 boilerplate를 과잉 검출한다. semantic similarity는 paraphrase recall을 높일 수 있으나 같은 주제를 다룬 정당한 교육 문서를 제거한다. 마지막으로 model-behavior contamination은 prompt perturbation, answer likelihood, reasoning trace 같은 별도 평가가 필요하다. 앞의 세 단계는 corpus 안의 흔적을 찾고, 마지막 단계는 그 흔적이 model behavior로 나타나는지를 묻는다. 둘을 같은 검출률로 합치지 않는다.

이 책에서 leakage는 평가에 쓰지 말아야 할 정보가 데이터·feature·prompt·운영 경계를 넘어간 사건을 넓게 가리킨다. contamination은 그 가운데 benchmark item 또는 의미상 가까운 내용이 학습·선택·튜닝 과정에 섞인 상태를 가리킨다. benchmark 원문이 corpus에 있었다는 hit, 실제 loss-bearing token으로 소비됐다는 exposure, 점수에 인과적으로 영향을 줬다는 behavior evidence는 서로 다른 주장 수준이다.

detector별 교집합과 차집합을 sample audit한다. “오염률 2%” 하나로 합치지 않는다. document hit, token hit, benchmark item coverage, consumed exposure를 구분한다. hit 문서가 dedup에서 이미 reject됐는지, survivor가 남았는지, packer가 실제 소비했는지 계보를 따른다.

benchmark에도 revision이 있다. split row IDs, prompt/answer checksum, release date, normalization, n-gram config를 고정한다. corpus crawl date가 benchmark 공개 전이라고 무조건 clean한 것은 아니다. pre-release web mirror와 원 출처가 있을 수 있다. 반대로 같은 주제 문서라고 정답 유출은 아니다.

## 4.6 quality filter를 데이터 목적함수로 읽는다

quality classifier score threshold를 올리면 평균 score가 높아질 수 있지만 retained tokens, domain, language, style, 길이가 함께 바뀐다. downstream 성능 변화가 quality 때문인지 양과 mixture 때문인지 분리하려면 같은 consumed token budget, tokenizer, model, optimizer, evaluation을 고정한다.

retained corpus가 작아 같은 token budget 동안 더 반복되면 repetition 효과가 들어간다. unique documents, unique tokens, average exposure를 기록한다. epoch 수만 맞추면 서로 다른 token budget이다. 같은 steps만 맞춰도 sequence length와 valid mask가 다르면 supervised targets가 다르다.

threshold sweep은 score histogram 전체와 domain별 acceptance curve를 본다. global threshold가 forum, code, academic, multilingual text에 다른 recall을 만들 수 있다. classifier domain tag 자체가 틀릴 수 있으므로 URL host, source collection, content classifier의 여러 view를 함께 둔다.

filter A/B의 removed sample을 human audit할 때는 score 경계 부근과 극단, 주요 domain을 층화 추출한다. precision만 아니라 어떤 가치 있는 mode가 사라지는지 taxonomy를 만든다. 결과는 threshold 선택뿐 아니라 feature와 training data 개선으로 되먹임한다.

### 4.6.1 PII·safety와 quality의 목표를 분리한다

quality, PII, toxicity, malware, license filter는 실패 비용이 다르다. 하나의 총점으로 합치면 왜 reject됐는지 설명하기 어렵다. detector score와 reason을 sidecar로 보존하고 materialization policy에서 결합한다. PII span redaction은 document reject와 다르며 offset map을 갱신해야 한다.

redaction placeholder가 tokenizer에서 희귀 pattern을 만들거나 원 주변 context를 왜곡할 수 있다. 원문 접근권은 제한하되 redaction transform revision, span type, old/new interval을 남긴다. 삭제 요청과 security incident에는 raw payload를 아무 dossier에나 복사하지 않고 권한 있는 resolver를 둔다.

**분산 제조의 정확성은 throughput과 별개다.**

### 4.6.2 partition·retry·speculation의 선택 편향을 찾는다

`hash(DocumentID) mod N`은 worker 수 N이 바뀌면 대부분의 assignment가 바뀐다. final document set이 같아도 shard layout과 order가 달라진다. release identity가 set만 포함하는지 ordered shards까지 포함하는지 명시한다. trainer sample order가 shard order에 의존하면 reshard는 trajectory를 바꾼다.

worker는 임시 shard를 쓰고 checksum을 계산한 뒤 immutable final key와 marker를 publish한다. 완료 응답 직전 죽어 scheduler가 retry해도 같은 DocumentID가 두 shard에 나타나지 않아야 한다. deterministic output ID와 compare-and-swap manifest가 필요하다. append-only writer에 무심코 retry하면 duplicate exposure가 생긴다.

speculative worker 둘이 같은 partition을 처리할 때 먼저 끝난 payload만 commit하고 다른 결과가 checksum까지 같은지 확인한다. 다르면 nondeterministic transform이다. 먼저 끝났다는 이유로 임의 결과를 채택하면 release 재현성이 없다.

### 4.6.3 stage별 count를 reconciliation한다

단순 transform은 `input=accepted+rejected+error`를 만족해야 한다. split/merge extractor는 lineage edge cardinality를 쓴다. error를 reject에 숨기지 않는다. transient fetch failure와 quality reject는 retry 정책이 다르다. document count, bytes, tokens는 stage마다 보존되지 않으므로 변화 이유를 적는다.

| 수량 | 정확한 뜻 | 흔한 오독 |
|---|---|---|
| raw records | crawl 입력 행 | unique URL로 오독 |
| extracted docs | parser 산출물 | content unique로 오독 |
| survivors | dedup 대표 | 원 정보량으로 오독 |
| normalized bytes | transform 후 bytes | crawl bytes로 오독 |
| tokenizer tokens | 특정 revision 출력 | tokenizer 독립 크기 |
| supervised targets | mask 뒤 loss 위치 | compute tokens와 혼동 |

release audit는 모든 expected task marker, shard checksum, aggregate counts, sampled raw→final lineage를 검사한다. random sample만으로 missing shard를 잡을 수 없고 aggregate count만으로 duplicate와 omission이 상쇄된 경우를 잡을 수 없다. manifest set reconciliation이 필요하다.

## 4.7 token contribution과 다음 소비자 인계를 보존한다

accepted text만 넘기면 5장에서 normalization이 한 번 더 일어났을 때 원 offset을 잃는다. `CorpusRevision`, DocumentID, raw/extracted/normalized checksum, interval map, language/domain/quality attributes를 넘긴다. tokenizer는 자신의 normalizer가 corpus text를 어떻게 바꿨는지 child interval map을 만든다.

token shard에는 `(DocumentID, normalized-char/byte interval, token start,length)` 역색인이 필요하다. packed sample은 여러 document segment를 합치므로 segment별 source ID와 boundaries를 보존한다. deletion은 raw→survivor→token segment→packed sample→consumption을 거슬러야 한다.

logical length와 physical capacity를 구분한다. memory-mapped shard 끝의 padding이나 preallocated bytes를 valid tokens로 세지 않는다. shard index와 checksum이 logical range를 정의한다. partial write가 capacity만 정상으로 보이는 실패를 주입한다.

### 4.7.1 tokenizer·packer까지 source 좌표를 넘긴다

요청이 URL이면 모든 crawl snapshot의 raw records를 resolve한다. content checksum으로 mirror를 찾고 normalized/dedup graph에서 reject와 survivor를 찾는다. token과 packed reverse index, sampler consumption을 조회한다. 아직 소비되지 않았으면 future artifacts를 revoke한다. 소비됐다면 영향 RunID와 descendant checkpoints/adapters를 표시한다.

파일 삭제가 model forgetting은 아니다. nonlinear training trajectory와 optimizer moments 때문에 sample gradient를 단순히 빼서 원상 복구할 수 없다. retrain, unlearning, evaluation을 별도 artifact로 다루고 보장 수준을 쓴다. “삭제 완료”를 storage deletion, future exclusion, released-model mitigation으로 나눈다.

dedup reject였던 문서도 survivor가 내용을 담고 있을 수 있다. Bloom filter와 packed cache, object-store replica도 후손이다. revoke ledger는 각 artifact의 status, 처리 시각, 처리 근거, 남은 불가능 경계를 기록한다.

실패 주입은 fixture URL 하나를 tombstone하고 raw, normalized, cluster, token offsets, packed samples, consumption, checkpoints를 역추적한다. 끊긴 edge가 나오면 조용히 성공 처리하지 않고 provenance gap으로 남긴다.

### 4.7.2 코드·수식·다국어의 contribution을 별도 감사한다

자연어용 quality rule이 terminal punctuation이나 평균 단어 길이를 강하게 보면 code, 표, 수식, 목록을 제거할 수 있다. language detector는 짧은 code와 고유명사를 잘못 분류한다. HTML extractor가 `<pre>`, MathML, alt text를 어떻게 처리하는지 fixture로 확인한다.

Unicode normalization은 수식 기호와 결합 문자, 전각 문자에 영향을 준다. source code에서는 whitespace와 quote가 의미를 가진다. 하나의 aggressive normalizer를 모든 domain에 적용하지 않는다. transform policy가 domain classifier 결과에 의존하면 classifier 오류가 비가역 삭제로 이어진다.

다국어 threshold는 언어별 calibration data와 script coverage를 본다. mixed-language 문서를 한 label로 버리지 말고 span 또는 document policy를 명시한다. tokenizer fertility와 연결해 retained bytes뿐 아니라 tokens per character와 truncation을 본다.

**보안과 권리도 pipeline state다.**

compressed archive와 HTML은 decompression bomb, path traversal, parser exploit를 포함할 수 있다. extraction worker에 size/time limit, sandbox, content-type 검증, 제한된 network/secret 권한을 둔다. corpus text의 prompt injection은 pretraining safety 문제이고 worker command injection과는 다른 보안 경계다.

curation code license, dataset distribution terms, 원 web content의 권리는 서로 다르다. 공개 script를 실행할 수 있다는 사실이 결과 문서 재배포 권리를 만들지 않는다. source locator, access date, license/terms revision, opt-out 경로를 manifest에 연결한다.

권리 상태가 바뀌면 future release filter에 반영하고 이미 materialized shard와 descendants를 찾는다. legal status와 technical deletion status를 같은 boolean로 압축하지 않는다. 공개 가능한 metadata와 제한된 payload를 분리한다.

### 4.7.3 관측성과 비용 원장을 같은 분모에 둔다

stage별 `docs/sec` 하나만 보면 빠르게 실패한 worker를 높은 처리량으로 오해하기 쉽다. 먼저 처리량의 분자를 accepted document로 셌는지 input attempt로 셌는지 확인한다. 읽고 쓴 byte와 parser latency를 함께 보면 입출력 병목과 계산 병목을 나눌 수 있고, reject reason·error·skip count를 보면 속도가 실제 산출물로 이어졌는지 확인할 수 있다. tagger는 batch utilization을, MinHash 단계는 signature 생성률·candidate edge 수·component 크기를 따로 본다. object-store request와 retry amplification은 저장 계층의 비용을 드러낸다. cache hit가 많은 재실행과 cold run은 별도로 비교하고, 실행하지 않은 task는 처리량 분모에 넣지 않는다.

candidate edge 폭증을 관찰했다면 곧바로 threshold를 올리지 않는다. 가능한 원인은 similarity threshold 완화, 공통 boilerplate 증가, giant bucket cap 변화, partition skew다. 같은 input sample에서 normalization view와 threshold를 고정한 채 bucket·edge·component 수를 단계별로 비교하면 최초 분기점을 찾을 수 있다. component size histogram과 top hub의 실제 span을 읽어 boilerplate bridge인지도 판정한다. 특정 worker만 느리면 input compressed size, document length, malformed rate, storage locality를 교차 비교한다. 평균은 straggler를 숨기므로 p50/p95/p99와 maximum을 함께 둔다.

metric label에 URL이나 DocumentID를 직접 넣어 cardinality를 폭발시키지 않는다. detailed dossier는 log/object artifact로, metric은 stage/reason/domain의 제한된 label로 둔다. sample trace가 aggregate metric의 어느 bucket에 속하는지 correlation ID로 잇는다.

**release 전 통제 실험.**

fixture corpus에는 exact duplicate, whitespace 변형, 문단 재배치, mirror URL, 업데이트 기사, code/math/multilingual, PII span, benchmark exact/paraphrase, malformed HTML, 매우 긴 문서를 넣는다. 각 문서의 expected stage result, cluster, survivor, offsets를 사람이 검토한다.

determinism test는 같은 input/config를 worker 수와 scheduling 순서를 바꿔 두 번 처리한다. survivor set, reason, checksum을 비교한다. set은 같고 shard order만 다르면 release contract가 어느 수준을 요구하는지 판정한다. config 한 항목만 바꾼 child run은 영향 stage 이후만 달라져야 한다.

quality ablation은 raw universe, consumed token budget, tokenizer, model config, optimizer, evaluation을 고정한다. retained size가 작아 반복이 늘면 unique exposure를 보고한다. clean validation과 contamination-sensitive benchmark를 분리한다. 수치는 해당 조건에만 붙인다.

deletion drill은 이미 설명한 reverse lineage를 시간 제한 안에 수행한다. partial shard와 duplicate retry, world-size 변경, stale completion marker도 주입한다. 실패가 예상 detector에서 차단되고 release manifest가 불완전 artifact를 선택하지 않아야 한다.

**조사 dossier와 결정 트리.**

문서 하나의 dossier에는 raw locator/checksum, extractor output과 interval, tag scores/reasons, dedup edges/cluster/survivor, contamination hits, token/packed offsets, consumption history를 둔다. payload 접근권을 보존하며 필요한 span만 resolver로 본다. 전체 private corpus를 bug report에 복사하지 않는다.

stage count가 다르면 raw manifest, task markers, input/accepted/rejected/error reconciliation을 본다. count가 같고 bytes가 다르면 extractor와 ordering이다. survivors만 다르면 seed, signature, candidate, component, tie-break를 본다. token만 다르면 tokenizer revision과 normalization이다. mixture만 다르면 logical length, sampler와 packing이다.

downstream loss가 나빠졌다면 곧바로 “필터가 나쁘다”고 결론내리지 않는다. 먼저 두 run의 consumed valid-token budget과 repetition을 맞춘다. 그다음 domain/language mixture, removed score/reason sample, cluster size와 survivor, tokenizer fertility, truncation을 비교한다. parent corpus에서 filter 하나만 되돌린 child를 만들어 paired ablation하면 데이터 양, 혼합, tokenization과 정책 변화 가운데 최초 원인을 줄일 수 있다. loss 회복만으로 정책상 제거해야 할 content를 되살려서는 안 되므로 품질 목적과 권리·안전 gate도 분리해 판정한다.

삭제가 안 되면 identifier resolve, mirror checksum, reject→survivor, token reverse index, packed cache, consumption, checkpoints 순서로 끊긴 edge를 찾는다. “찾지 못함”은 삭제 성공이 아니라 계보 불완전이다.

**5장과 6장으로 넘기는 소비 계약.**

최종 manifest는 `CorpusRevision`, ordered shards와 checksums, DocumentID 생성 revision, raw/extracted/normalized checksums와 interval maps, source/domain tags, quality scores/reasons, dedup cluster/survivor, contamination hits, RevocationID, logical length/capacity를 포함한다.

5장은 normalized bytes와 offsets를 tokenizer IDs와 token intervals로 바꾼다. 6장은 source/domain/length를 packed sample과 supervised-token mixture로 바꾼다. 따라서 “데이터 50%”라는 표현에는 document, byte, emitted token, supervised target 중 분모가 반드시 붙는다.

공개 code가 production run의 deletion→checkpoint 폐루프를 증명하지 않으면 미확인으로 남긴다. 공개 transform을 재현할 수 있다는 것과 조직의 전체 governance가 닫혔다는 것은 다르다. 이 경계를 정직하게 표시해야 독자는 자기 stack에서 무엇을 더 만들어야 하는지 안다.

장 종료 판정은 세 문장으로 요약된다. 모든 accepted document에는 raw ancestor와 transform revision이 있다. 모든 reject에는 reason 또는 survivor가 있다. 모든 release shard는 committed manifest와 count가 맞고 tombstoned ID는 다음 loader에서 소비되지 않는다. 이 셋을 evidence로 증명하지 못하면 corpus는 크더라도 학습 가능한 제품으로 완성되지 않았다.

## 4.8 immutable raw snapshot과 streaming generation을 나눈다

앞 절까지는 한 문서가 학습 기여로 바뀌는 논리 경계를 세웠다. 이제 그 경계를 다시 실행할 수 있도록 crawl index, request·response와 WARC payload를 immutable generation으로 묶는다. streaming은 원본을 덜 보존하는 방식이 아니라 어느 시점의 어떤 record를 읽었는지 더 엄격히 기록해야 하는 실행 방식이다.

### 4.8.1 crawl index에서 WARC record까지 보존한다

웹 수집의 첫 입력은 URL 목록이 아니라 crawl revision과 record locator다. Common Crawl 같은 공개 snapshot은 crawl ID, WARC filename, byte offset과 length로 개별 response를 가리킬 수 있다. 원본을 내부 object store로 복사하는 경우 source locator, response payload checksum, fetch metadata, ingest 시각을 분리한다. URL은 내용 식별자가 아니다. 같은 URL이 시간에 따라 바뀌고 서로 다른 URL이 같은 payload를 가질 수 있다.

HTTP status, MIME header, detected MIME, charset, transfer/content encoding은 extraction 앞의 state다. header가 HTML이라도 실제 payload가 PDF나 binary일 수 있다. decompression bomb, 비정상적으로 큰 record, malformed chunk를 resource limit 안에서 격리한다. parser가 crash한 record를 조용히 drop하지 않고 `ERROR` reason과 raw locator를 남긴다.

robots, crawl policy, 접근 조건은 획득 시점의 정책 기록과 연결한다. 공개 접근 가능성은 자유로운 재배포나 모든 학습 용도를 자동 보장하지 않는다. 도메인별 terms snapshot, 라이선스 표식, jurisdiction, opt-out 신호를 policy tag로 보존하고 법적 판단과 기술 filter를 분리한다. policy가 바뀌면 어느 derived revision이 영향을 받는지 reverse lineage로 찾는다.

immutable raw zone은 재처리를 가능하게 하지만 무기한 보관 권리를 뜻하지 않는다. retention과 deletion 정책을 별도 state로 둔다. raw payload를 삭제해야 해도 checksum, 삭제 사유, tombstone, 변환 계보를 허용 범위에서 보존한다. checksum이 개인 정보 자체를 누설할 위험과 재식별 가능성도 접근 정책에서 검토한다.

### 4.8.2 incremental crawl의 중복과 갱신을 구분한다

새 crawl의 payload checksum이 기존과 같으면 exact duplicate일 수 있다. 일부 문단만 바뀌면 갱신 기사나 템플릿 변경일 수 있다. 오래된 문서를 무조건 버리거나 새 문서를 무조건 survivor로 택하지 않는다. publish time 신뢰도, content completeness, canonical link, extraction 품질, source priority를 deterministic tie-break로 만든다.

temporal dataset에서는 미래 snapshot이 과거 benchmark 시점에 섞이지 않도록 crawl time과 content time을 구분한다. 서버가 잘못된 날짜를 보낼 수 있어 관측 값과 파싱 신뢰도를 함께 둔다. time cutoff는 URL 단위가 아니라 payload와 derived span까지 전파되어야 한다.

**extraction은 text 생성이 아니라 좌표 변환이다.**

### 4.8.3 DOM node를 normalized byte span으로 변환한다

HTML extractor는 navigation, cookie banner, script/style, 반복 footer를 제거하고 본문 순서를 선택한다. 출력 문자열만 저장하면 왜 문장이 사라졌는지 알 수 없다. 원 payload byte interval, DOM path, extracted interval, normalization interval의 mapping을 보존한다. 완전한 문자별 map이 비싸면 block 단위와 exception span을 결합하되 정확도 범위를 표시한다.

Unicode normalization, whitespace folding, hyphen 결합, entity decode는 exact dedup hash와 tokenizer IDs를 바꾼다. 각 transform을 revisioned stage로 두고 before/after checksum, removed/inserted interval을 남긴다. language classifier를 normalization 전후 어느 text에 적용했는지도 중요하다. mojibake가 언어 오분류와 quality reject로 이어질 수 있다.

표, code block, 수식, alt text는 prose extractor의 품질 규칙으로 평가하면 손실된다. structure tag를 보존하거나 별도 extractor route를 둔다. markdown으로 변환할 때 fence와 indentation이 유지되는지 fixture로 본다. DOM 순서와 시각 순서가 다른 page에서는 accessibility tree나 layout 정보가 필요한지 결정한다.

**parser upgrade의 영향 반경을 측정한다.**

extractor revision을 바꾸면 전체 crawl을 즉시 교체하지 않는다. 고정 stratified fixture와 실제 sample에서 text length, retained nodes, language, quality score, dedup cluster, token fertility가 어떻게 이동하는지 본다. downstream model proxy를 돌리기 전 stage-level 차이를 설명해야 한다.

old/new extraction의 document alignment에는 raw record ID를 쓴다. accepted set의 Jaccard, content checksum 변화율, paragraph addition/removal, domain별 이동을 보고한다. aggregate length가 같아도 서로 다른 문장이 교체될 수 있다. high-impact diff를 사람이 읽고 regression category를 taxonomy에 추가한다.

**DataTrove를 실행기와 block 계약으로 읽는다.**

**pipeline block의 입력·출력·side effect.**

DataTrove 계열의 local pipeline executor와 distributed executor는 block 목록을 같은 추상 단계로 실행하지만 scheduling, task state, logging, retry가 다르다. source anchor는 저장소 고정 revision에서 executor class, pipeline block interface, reader/writer, filter, dedup component와 tests로 잡는다. README 예제만으로 production semantics를 추정하지 않는다.

reader는 `Document`에 text와 ID, metadata를 넣고 modifier는 text 또는 metadata를 바꾸며 filter는 keep/reject와 reason을 정한다. writer는 shard serialization과 경로를 소유한다. 실제 revision의 field와 method signature를 확인한다. custom block이 in-place로 metadata를 바꾸는지 새 document를 내는지에 따라 lineage capture 위치가 달라진다.

executor의 task ID와 input shard mapping을 manifest에 둔다. `skip_completed`는 completion marker를 신뢰해 작업을 생략한다는 뜻이지 output correctness를 다시 증명한다는 뜻이 아니다. marker가 어느 파일 write 뒤 생성되는지, partial output이 남을 때 retry가 overwrite/append하는지, config revision이 marker namespace에 포함되는지 확인한다.

**stats와 exclusion writer를 감사 증거로 바꾼다.**

filter가 reject count만 올리면 삭제된 문서의 이유를 표본 이상 추적하기 어렵다. exclusion writer가 rejected document와 reason을 보존할 수 있지만 payload 보관 권한과 비용을 고려해야 한다. 최소한 DocumentID, input/output checksums, filter symbol/revision, score, threshold, reason을 남긴다.

pipeline stats는 stage-local counter다. 이전 stage accepted와 다음 stage input, retry duplicate, error quarantine, written documents를 reconcile한다. 여러 worker counter의 단순 합이 speculative duplicate를 포함하는지 확인한다. task attempt와 logical task를 구분해야 정확한 분모가 나온다.

DataTrove 예제를 채택해도 organization-level deletion과 checkpoint consumption 계보가 자동으로 생기지는 않는다. 공개 library가 제공하는 변환·실행·통계 기능과 별도 control plane이 제공해야 할 revision registry, tombstone, reverse index, atomic release를 경계표로 나눈다.

**Dolma를 tagger·mixer·artifact 계약으로 읽는다.**

**annotation은 bool보다 풍부한 좌표다.**

Dolma 계열 파이프라인은 문서나 span에 attribute를 붙이고 이를 기반으로 filter/mix할 수 있다. 검토할 때 저장소 고정 revision의 tagger interface, attribute serialization, processor, mixer config, tests를 연결한다. tag 이름만 보지 않고 값의 범위, span 좌표 기준, 겹침 처리, absence 의미를 읽는다.

PII나 toxicity detector가 span을 반환하면 extraction/normalization 이후 좌표인지 확인한다. 이후 text modifier가 앞부분을 삭제하면 offset을 갱신해야 한다. detector score threshold를 바꾸면 reject set만 아니라 redact된 output checksum과 downstream dedup signature가 바뀐다. stage order를 config list로 보존한다.

mixer는 source별 weight와 target size를 적용하지만 logical documents, bytes, tokens 중 어느 단위인지 확인한다. oversampling은 같은 document 노출을 늘리고 without-replacement selection은 unique mass를 줄인다. mixture manifest에는 source revision, eligible count, sampling seed, emitted sequence와 repetition histogram을 둔다.

**Dolma release를 그대로 복제했다고 말할 조건.**

공개 config 이름이 같아도 source snapshot, tool revision, classifier artifact, model checksum, external dependency가 다르면 같은 corpus가 아니다. 각 stage의 frozen artifact와 ordered inputs가 있어야 한다. 일부 비공개 또는 사라진 artifact가 있으면 재현 범위를 명시한다.

release card의 aggregate 통계는 provenance proof를 대신하지 않는다. 우리가 만든 shard의 checksums, counts, domain/language/quality distribution을 공개 reference와 비교하고 차이를 설명한다. 결과가 다를 때 threshold를 임의 조정해 숫자만 맞추지 않는다.

**DCLM을 data benchmark로 읽는다.**

**pool·processing·training·evaluation을 분리한다.**

DCLM 저장소의 high-level workflow는 raw source selection, processing, tokenization/shuffling, training, evaluation을 구분한다. leaderboard 비교에서는 processing 전략만 자유롭고 나머지 recipe를 표준화하는 범위가 있다. 따라서 높은 score는 filter 하나의 보편적 우월성이 아니라 고정된 pool·compute scale·training/evaluation 조건 아래 결과다.

DCLM-Pool, RefinedWeb, Baseline은 같은 이름의 stage가 아니다. README의 고정 revision에서 각 artifact의 전처리 범위와 알려진 비정합을 읽는다. competition subset과 refined subset이 정확히 대응하지 않는다는 경고가 있다면 같은 leaderboard population으로 합치지 않는다. artifact locator와 발표 시점, correction note를 source dossier에 둔다.

Ray 기반 exact content/URL dedup과 baseline에서 사용한 Rust 기반 fuzzy substring 도구의 관계도 구분한다. 공개 README는 Rust 구현이 Ray YAML pipeline에 바로 통합되지 않는 경계를 설명한다. “DCLM pipeline을 실행했다”는 문장에는 어느 dedup 도구를 별도 단계로 실행했는지 포함해야 한다.

**BFF의 Bloom filter를 손실 함수처럼 감사한다.**

Rust BFF 구현에는 expected n-gram count, filter size, hash 수, false-positive 목표와 remove type을 설정한다. Bloom false positive는 중복이 아닌 n-gram을 이미 본 것으로 판단해 text를 제거하는 데이터 손실이다. 예상 count가 실제 삽입량보다 작으면 occupancy와 false-positive rate가 올라간다. 실행 뒤 bit occupancy와 empirical negative fixture를 측정한다.

document-level, paragraph-level, substring removal은 survivor text를 다르게 만든다. `old-both` 같은 option을 이름만 복사하지 않고 `process_line_*` 계열 함수와 tests에서 정확한 removal 경계를 읽는다. output annotation, no-update mode, filter persistence가 두-pass 실행의 state에 어떤 영향을 주는지 본다.

Bloom filter 파일은 corpus state다. build inputs, insertion order가 결과에 영향을 주는지, hash seed/algorithm, serialized checksum을 manifest에 둔다. retry가 같은 n-gram을 다시 삽입하는 것은 membership에는 같을 수 있어도 count metric을 왜곡한다. shared filter의 concurrency와 deterministic output도 fixture로 확인한다.

## 4.9 quality·dedup·split 정책의 선택 편향을 측정한다

### 4.9.1 threshold는 품질의 자연 법칙이 아니다

fastText나 다른 classifier가 reference-positive와 web-negative를 구분하도록 학습되면 score는 label construction을 반영한다. 높은 threshold는 reference와 문체가 다른 유용한 domain, dialect, 짧은 답변, code/math를 제거할 수 있다. ROC 숫자 하나보다 domain/language/length별 retained rate와 사람이 읽은 false reject를 본다.

classifier model checksum, tokenizer/feature revision, training datasets와 cutoff를 고정한다. benchmark 문서가 positive training set에 들어가면 contamination filter 이전에 selection bias가 생길 수 있다. model card의 의도와 실제 inference wrapper의 normalization, truncation, batch 처리도 확인한다.

global threshold와 percentile selection은 다르다. score distribution이 crawl마다 이동하면 같은 threshold의 retained mass가 바뀐다. percentile은 mass를 고정하지만 절대 품질 기준을 바꾼다. child corpus에서 threshold sweep을 만들고 retained unique tokens, domain mixture, repetition, proxy loss와 benchmark를 함께 본다.

**heuristic을 classifier 앞뒤 어디에 놓는가.**

짧은 문서, 반복 기호, stop-word ratio 같은 heuristic을 먼저 적용하면 classifier가 보는 population이 달라진다. classifier 뒤에 적용하면 비용은 늘고 reject reason overlap이 달라진다. 각 filter의 marginal removal과 cumulative removal을 둘 다 기록한다. 첫 reject reason만 저장하면 뒤 filter가 제거했을 문서를 알 수 없어 ablation을 재구성하기 어렵다.

shadow mode에서 모든 score를 계산하고 실제 reject는 하지 않는 sample run을 둔다. 이 artifact로 순서 변경과 threshold ablation을 재생한다. 전체 corpus에 모든 detector를 적용하는 비용이 크면 stratified sample과 confidence interval을 사용하며 추정임을 명시한다.

**dedup을 connected-component 정책으로 완성한다.**

### 4.9.2 edge 검출과 survivor 정책을 분리한다

MinHash/LSH는 유사할 가능성이 있는 candidate를 줄이고 exact similarity 검증이 edge를 확정한다. A와 B, B와 C가 threshold를 넘지만 A와 C는 넘지 않아도 connected component는 셋을 묶을 수 있다. pair removal과 component survivor 정책은 결과가 다르다. component size와 diameter, hub 문서를 검사한다.

survivor는 earliest crawl, 높은 quality, 긴 content, preferred domain 같은 ranking tuple로 결정한다. tie-break가 deterministic하지 않으면 worker scheduling에 따라 corpus가 바뀐다. ranking feature가 future information을 쓰면 temporal cutoff를 깨뜨릴 수 있다. survivor decision record에 후보 전체 ID와 score, 선택 규칙 revision을 둔다.

paragraph dedup은 문서 일부를 제거해 새로운 text checksum을 만든다. 그 뒤 document fuzzy dedup을 다시 계산할지 stage order를 정한다. boilerplate removal 전 signature는 공통 footer 때문에 거대한 component를 만들 수 있다. order A/B에서 candidate edges, components, retained tokens, downstream quality를 비교한다.

### 4.9.3 benchmark contamination에 별도 namespace를 둔다

benchmark exact text, normalized text, n-gram overlap, paraphrase/semantic overlap을 서로 다른 detector로 둔다. false positive와 false negative 비용이 다르므로 threshold와 action도 분리한다. exact hit는 span을 제거할 수 있지만 semantic hit는 review tag만 붙일 수 있다. test question만 아니라 answer/explanation과 benchmark를 소개하는 문서도 정책 범위를 정한다.

benchmark version과 prompt formatting을 고정한다. 나중에 추가된 benchmark를 과거 corpus에 적용한 retrospective audit와 당시 사용한 pretraining filter를 구분한다. 보고서에는 detector coverage와 미검출 가능성을 남긴다. “오염 없음” 대신 검사한 namespace·revision·threshold 아래 hit 수라고 표현한다.

## 4.10 deletion과 provenance를 학습 소비까지 닫는다

### 4.10.1 DocumentID에서 packed sample·checkpoint까지 전파한다

삭제 요청을 canonical source identity와 연결한 뒤 exact/fuzzy mirrors, extracted/normalized descendants, token cache, packed samples, dataloader manifest, consumed UpdateID range로 전파한다. 이미 만들어진 model checkpoint에서 개별 문서 영향을 완전히 제거하는 것은 단순 shard 삭제와 다르다. retraining, selective unlearning, serving policy 같은 후속 선택을 명시한다.

reverse index가 없으면 모든 artifact를 scan해야 하고 scan coverage를 증명해야 한다. rejected document가 survivor의 mirror였다면 reject ID에서 survivor edge를 따라간다. paragraph가 다른 documents에 복제된 경우 span-level edge가 필요하다. tombstone은 loader가 다음 open에서 거부하도록 release registry와 결합한다.

deletion SLA에는 request validation, affected artifact discovery, quarantine, rebuild, release commit, downstream notification, verification 시간이 있다. “삭제 완료”는 storage object 하나의 삭제가 아니다. next training job의 sample fixture에서 tombstoned ID가 나오지 않고 old revision이 production selector에서 제외되어야 한다.

### 4.10.2 법적 판단을 기술 score로 숨기지 않는다

라이선스, 개인정보, 민감정보, opt-out, safety는 판단 권한과 이의 제기 경로가 서로 다르다. quality score가 높다고 권리 문제를 덮지 않고 toxicity score가 낮다고 개인정보가 없는 것은 아니다. policy decision과 detector output을 별도 fields로 저장한다.

redaction은 원문 좌표와 대체 방식, detector confidence, review state를 남긴다. 과도한 redaction이 문맥과 언어 품질을 훼손하는지 sample audit한다. false negative가 발견되면 detector revision만 올리는 데서 끝나지 않고 기존 descendants에 backfill scan을 수행한다.

**corpus release를 데이터베이스 commit처럼 검증한다.**

### 4.10.3 two-phase publish와 fail-closed loader를 사용한다

worker는 temporary namespace에 shard와 stats를 쓰고 checksum·schema·count reconciliation을 마친다. coordinator가 ordered shard manifest와 policy summary를 만들고 atomic commit marker를 게시한다. loader는 marker와 manifest checksum이 맞는 revision만 연다. directory에 파일이 있다는 이유로 최신 corpus로 선택하지 않는다.

speculative execution에서 같은 logical task의 두 attempt가 성공해도 manifest에는 하나만 들어간다. loser output은 garbage collection 대상이며 count에 중복 합산하지 않는다. partial write, truncated compression stream, wrong schema, stale marker를 고장 주입한다. build job이 성공 exit code를 냈다는 사실보다 loader가 불완전 revision을 거부하는지가 중요하다.

release diff에는 parent revision 대비 added/removed/modified documents, bytes/tokens, reason, domain/language/quality distribution, dedup cluster 이동, contamination hits를 둔다. 거대한 aggregate만 주지 않고 stratified sample locator를 연결한다. config 변경의 예상 영향 stage보다 앞에서 diff가 생기면 input 통제를 다시 본다.

**tokenizer·training 팀과의 인계.**

tokenizer 팀에는 normalized bytes, interval maps, special fragments, language/script/domain tags, DocumentID와 revision을 넘긴다. fertility와 fallback 분석 결과를 corpus filter에 feedback할 때 child revision으로 만든다. tokenizer가 바뀌면 corpus text는 같아도 emitted token 수와 mixture가 달라지므로 tokenized artifact revision을 분리한다.

training 팀에는 ordered shards, logical length, sampling weights, tombstone registry, contamination report, expected counts와 loader fixture를 넘긴다. consumed sample IDs와 UpdateID를 다시 corpus lineage에 기록할 callback을 제공한다. 데이터 제조와 학습이 일방향 파일 전달로 끝나지 않고 삭제·회귀 분석이 돌아오는 폐루프가 된다.

**corpus 공정의 최종 test matrix.**

**단위 fixture와 scale audit.**

단위 fixture에는 malformed encoding, nested boilerplate, code/table/math, multilingual mixed script, PII span, exact/fuzzy duplicate chain, benchmark paraphrase, rights tombstone를 둔다. 각 stage expected checksum, annotation, reason, survivor, interval을 사람이 승인한다. parser·filter·dedup upgrade마다 snapshot diff를 검토한다.

property test는 worker 수와 task order, retry 위치를 바꿔 survivor set과 release checksum의 결정성을 본다. Bloom capacity를 넘기고 false-positive detector가 경고하는지 본다. deletion request에서 모든 descendants가 발견되는지 graph traversal을 검증한다. partial publish에서는 loader가 parent revision을 계속 선택해야 한다.

scale audit는 counts, bytes, tokens, reason histogram, component distribution을 shard 합과 global 결과 사이에서 reconcile한다. sample-based quality estimate에는 sampling frame과 uncertainty를 붙인다. 실행하지 않은 full-corpus 검사를 완료로 표시하지 않는다.

**독자가 받아야 할 조사 도구.**

`explain-document DocumentID`는 raw locator부터 extraction intervals, tags, filter decisions, dedup component, survivor, token/packed descendants, consumption까지 보여줘야 한다. `diff-revision A B`는 stage별 영향과 reason을 보여준다. `trace-deletion RequestID`는 unresolved edge와 downstream acknowledgment를 보여준다.

이 도구가 없다면 장애 때 분산 logs와 object paths를 손으로 맞춰야 한다. 책의 설명은 특정 제품 명령을 강제하기보다 필요한 query contract와 expected output을 제시한다. 구현을 바꿔도 lineage와 상태 불변식은 유지된다.

최종 완료 조건은 크기나 공개 dataset 이름이 아니다. raw ancestor에서 accepted shard까지 모든 변환이 revisioned되고, reject와 survivor가 설명되며, contamination과 권리 정책의 검사 범위가 명시되고, tombstone이 loader와 consumption까지 전파되고, 불완전 publish가 선택되지 않아야 한다. 이 조건을 fixture와 release artifact로 반증하려 시도한 뒤 살아남은 corpus만 5장 tokenizer의 입력이 된다.

**비용 최적화가 선택 편향을 만들지 않는지 본다.**

**비싼 detector를 일부에만 적용하는 경우.**

PII, semantic contamination, model-based quality detector는 모든 문서에 적용하기 비쌀 수 있다. 먼저 싼 heuristic으로 후보를 줄이면 recall이 그 prefilter에 종속된다. cascade 각 단계의 conditional recall과 전체 estimated recall을 구분한다. 무작위 holdout에는 비싼 detector를 전부 적용해 prefilter가 놓친 population을 추정한다.

domain이나 language에 따라 detector 비용과 성능이 다르면 동일 budget에서 coverage가 달라진다. 처리량을 높이려고 긴 문서를 truncate하면 뒤쪽 PII나 benchmark span을 놓칠 수 있다. chunk overlap, maximum length, early-exit 조건을 config와 source anchor로 고정한다. 비용 보고에는 processed bytes와 fully scanned fraction을 둔다.

cache는 classifier 결과를 재사용하지만 key가 text checksum만이면 model revision 변경을 반영하지 못한다. cache key에 normalized checksum, detector/model/config revision을 포함한다. stale cache hit를 새 결과처럼 세지 않는다. cold/warm run 비용과 cache invalidation 범위를 별도로 측정한다.

**저장 비용과 감사 가능성 사이의 선택.**

모든 intermediate payload를 영구 보존하면 lineage는 쉽지만 비용과 개인정보 노출이 커진다. 모든 것을 버리면 재현과 삭제 검증이 불가능하다. raw locator/checksum, transform revision, interval map, decision record는 오래 보존하고 민감 payload는 접근 통제와 retention을 둔다. 재생 가능성은 source가 계속 존재한다는 가정까지 표시한다.

sample artifact는 stratified reservoir로 뽑아 흔한 domain만 남지 않게 한다. reject reason과 score boundary 주변을 oversample해 threshold review에 쓴다. 이 sample은 corpus 통계의 unbiased estimate와 다르므로 review sample과 measurement sample을 분리한다.

**세 공개 스택을 선택하는 판단표.**

**이름이 아니라 필요한 계약으로 고른다.**

DataTrove의 block/executor 구성이 필요한 변환 조립과 분산 task에 맞는지, Dolma의 tag/attribute와 mixer 표현이 span annotation과 mixture에 맞는지, DCLM의 standardized pool·training·evaluation이 data ablation 비교에 맞는지 묻는다. 세 프로젝트를 배타적 제품처럼 볼 필요는 없지만 revision과 artifact 경계를 흐린 채 조각을 섞어서는 안 된다.

선택표에는 input format, document schema, stage composition, task retry, exclusion evidence, exact/fuzzy dedup, classifier artifact, mixer semantics, atomic publish, reverse lineage, deletion, tests를 행으로 둔다. 공개 code에서 확인된 칸과 우리가 별도로 구현해야 하는 칸을 구분한다. 문서에 없는 기능을 추정으로 채우지 않는다.

통합 시 adapter boundary마다 DocumentID, text checksum, metadata schema, reason taxonomy를 검증한다. tool A의 span end가 exclusive이고 tool B가 inclusive면 한 글자씩 좌표가 틀어진다. 압축 JSONL에서 record order를 보장하지 않으면 shard checksum과 deterministic survivor에 영향을 준다. 작은 교차-tool fixture로 schema loss를 먼저 잡는다.

최종 handoff review에서는 corpus engineer뿐 아니라 tokenizer, training, evaluation, privacy 담당자가 같은 document dossier를 함께 검토한다. 각 담당자가 자기 stage 이후만 보는 구조에서는 contamination과 deletion의 끊긴 edge가 숨어든다. 하나의 문서를 raw에서 UpdateID까지, 하나의 삭제 요청을 source에서 다음 loader까지 추적할 수 있을 때 대규모 공정의 상호 연결이 실제로 닫힌다.

이 왕복 검사는 정상 accepted 문서, quality reject, duplicate loser, redacted 문서, tombstoned 문서를 각각 하나씩 골라 수행한다. 경로가 없는 상태와 접근 권한 때문에 payload를 볼 수 없는 상태를 구분하고, 후자는 승인된 resolver가 checksum과 필요한 span만 검증하도록 한다. 모든 사례의 예상 종착점과 실제 종착점이 일치해야 release 검토를 닫는다.

## 4.11 parse 결과에서 mixture·curriculum 질량까지 추적한다

개별 document의 accept/reject가 맞아도 전체 학습 분포는 달라질 수 있다. 이 절은 HTML 손실, split과 leakage, source별 availability를 document 수가 아니라 실제 tokenizer token과 loss-bearing token 질량으로 다시 계산한다. 그 결과가 6장의 sampler와 curriculum에 전달되는 입력 계약이다.

**URL은 identity가 아니다.** 같은 URL의 내용은 시간, 지역, cookie, user agent에 따라 달라질 수 있다. redirect와 canonical tag도 바뀐다. 문서 identity에는 crawl snapshot, fetch timestamp, resolved URL, HTTP status와 headers, payload checksum, WARC record locator를 둔다. URL normalization은 별도 파생 key이며 원 URL을 잃지 않는다.

WARC record는 HTTP request/response와 payload를 보존할 수 있지만 그 자체로 학습 document가 아니다. compressed shard, record offset, digest를 이용해 정확한 response를 다시 찾는다. content encoding과 transfer encoding을 해제한 bytes의 checksum, 원 wire payload checksum을 구분한다. parser가 읽은 bytes가 무엇인지 명시한다.

robots, 라이선스, 접근 정책은 crawler stage의 결정 상태다. 기술적으로 fetch 가능하다는 사실이 학습 사용 허가를 뜻하지 않는다. policy revision, decision reason, 적용 시각을 document lineage에 둔다. 나중의 삭제 요청과 정책 변경이 어떤 snapshots와 descendants에 영향을 주는지 역방향 index를 유지한다.

incremental crawl은 새 URL만 받는 과정이 아니다. 동일 URL의 content digest가 바뀌면 새 revision을 만들고, 동일 payload가 다른 URL에서 발견되면 provenance edge를 여러 개 둘 수 있다. redirect chain과 canonical relation을 dedup survivor 결정과 섞지 않는다. 최신 version 선호, 최초 수집 선호, 품질 score 선호는 서로 다른 정책이다.

fetch 실패를 빈 문서로 저장하지 않는다. DNS, timeout, TLS, 4xx, 5xx, oversized payload, unsupported MIME를 reason taxonomy로 나눈다. retryable 여부와 backoff, 최대 attempts를 기록한다. retry가 다른 payload를 받으면 attempt별 checksum을 보존하고 최종 선택 규칙을 둔다.

**partition은 처리량뿐 아니라 정확성 경계다.** URL hash, WARC shard, domain, time range로 tasks를 나눌 수 있다. partition key가 바뀌면 exact duplicate와 domain rate limit, deterministic ordering에 영향이 있다. task manifest에는 input shard range, expected record count와 bytes, executor revision을 둔다.

speculative execution으로 같은 task를 두 workers가 처리하면 output을 두 번 publish하지 않도록 attempt ID와 commit protocol이 필요하다. 먼저 완료한 attempt를 채택하더라도 둘의 output checksum이 다르면 nondeterministic transform을 조사한다. `skip_completed`는 output directory 존재 여부가 아니라 committed manifest와 input/config checksum 일치를 확인해야 안전하다.

task 성공 수만 보지 않고 input/output/reject/error count를 reconciliation한다. 한 record는 accepted, rejected, quarantined, error 중 정확히 하나의 terminal state를 가져야 한다. retry attempts를 document count에 중복 합산하지 않는다. bytes도 compression 전후 단위를 명시한다.

### 4.11.1 HTML→text 손실과 normalization 순서를 계측한다

**추출기는 DOM을 선택하고 순서를 만든다.** HTML parser는 invalid markup을 보정하고 DOM을 구성한다. boilerplate detector는 navigation, footer, cookie banner, 광고, code block, table을 선택하거나 제거한다. visible text order가 accessibility tree와 DOM order, 화면 layout 중 무엇을 따르는지 구현마다 다르다.

script/style 제거는 비교적 명확하지만 `alt`, title, aria label, hidden text는 정책이 필요하다. code와 mathematical notation을 일반 paragraph 규칙으로 처리하면 기호와 줄바꿈을 잃는다. table을 행렬 구조로 보존할지 평탄화할지 결정한다. Markdown conversion은 heading, list, link target, code fence를 새 syntax로 만든다.

추출 결과에는 raw byte interval에서 output character interval로 가는 span map을 둔다. 완전한 역함수가 불가능하면 transform event와 deleted spans를 남긴다. Unicode decoding replacement, entity unescape, whitespace collapse, hyphenation repair가 좌표를 바꾼다. 단순 offset delta 하나로는 중간 삭제와 병합을 표현하지 못한다.

parser revision을 바꾸면 동일 WARC sample을 old/new로 처리해 text checksum, length, language, filter decision, dedup signature 변화를 측정한다. 전체 acceptance rate가 비슷해도 특정 domain·language·content type에서 큰 차이가 있을 수 있다. stratified sample과 boundary cases를 비교한다.

**문서 경계와 paragraph 경계를 보존한다.** 여러 HTML pages를 한 document로 합치거나 pagination을 연결하면 provenance가 복잡해진다. 반대로 page 하나를 paragraphs로 쪼개면 dedup과 quality classifier 단위가 바뀐다. `DocumentID`와 `SegmentID`, parent relation을 구분한다.

문장 분리도 언어와 약어, code에 따라 오류가 난다. sentence dedup을 하기 전에 splitter revision과 offsets를 고정한다. normalized text에서 생성한 signature가 raw deletion request로 돌아갈 수 있도록 segment lineage를 둔다. 빈 paragraph와 반복 header 제거가 문서 길이 통계에 미치는 영향도 기록한다.

추출 quality fixture에는 nested tags, malformed HTML, multilingual text, RTL, emoji, entities, table, code, MathML, hidden elements, very long token, null byte를 넣는다. expected output을 사람이 검토하고 parser upgrade에서 diff한다. 웹의 실제 tail을 대표하지는 않으므로 production sample audit도 병행한다.

**정규화가 중복과 개인정보 검출 순서를 바꾼다.**

**하나의 canonical text로 모든 목적을 해결하지 않는다.** dedup에는 case folding과 whitespace collapse가 유용할 수 있지만 code, 고유명사, 비밀번호 패턴에는 정보를 잃는다. language identification은 normalization 전후 결과가 다를 수 있다. tokenizer용 text와 dedup signature용 view, PII detector용 view를 분리하고 공통 source span으로 연결한다.

Unicode NFC/NFKC 선택은 compatibility characters를 합친다. 전각 문자와 수학 기호, ligature가 달라질 수 있다. zero-width characters 제거는 obfuscation 탐지에 도움이 되지만 원문 증거를 잃을 수 있다. control character 정책과 line ending normalization을 config hash에 넣는다.

email, phone, IP, address, credential 같은 PII는 regex·NER·classifier가 서로 다른 recall/precision을 갖는다. redact, reject, quarantine 가운데 action을 분리한다. redaction replacement가 주변 tokenization과 dedup signature를 바꾸므로 PII stage 전후 identities를 연결한다. 동일 개인 정보가 duplicates에 퍼졌다면 한 문서 수정으로 끝나지 않는다.

secret scanner는 API key pattern과 entropy를 사용할 수 있지만 code corpus에서 false positives가 많다. 실제 secret 검증을 위해 외부 서비스에 값을 전송하면 더 큰 노출이 될 수 있다. detector output을 접근 통제하고 원문을 metric label에 넣지 않는다. false-positive appeal과 incident escalation 경로를 둔다.

normalization 순서의 test는 비가환성을 드러낸다. `normalize(redact(x))`와 `redact(normalize(x))`가 같은지 sample로 확인한다. 같지 않으면 어느 순서가 policy 목표에 맞는지 결정하고 pipeline graph에 고정한다. dedup 전 redaction과 dedup 후 redaction도 cluster 구조를 바꿀 수 있다.

**exact dedup을 hash table 이상의 계약으로 본다.**

**무엇을 hash하는지가 전부다.** raw payload, extracted text, normalized text, paragraph set은 서로 다른 exact duplicate를 정의한다. hash algorithm, encoding, newline, normalization revision을 signature metadata에 둔다. cryptographic collision 가능성은 작지만 hash만 저장한 상태에서 identity를 법적 증거처럼 과장하지 않는다.

exact duplicate group에서 survivor를 고르는 정책은 deterministic해야 한다. source quality, timestamp, license, domain, document length, stable DocumentID tie-break를 순서대로 둔다. 병렬 task 완료 순서로 첫 document를 선택하면 rerun에서 survivor가 바뀐다. downstream deletion과 mixture 통계가 흔들린다.

cluster에는 winner뿐 아니라 loser IDs와 reason, signature revision을 둔다. 저장 비용 때문에 모든 loser text를 보존하지 않더라도 provenance와 tombstone은 남긴다. winner가 삭제될 때 다음 loser를 승격할지 cluster 전체를 제거할지 정책을 정한다. license가 서로 다른 duplicates라면 가장 허용적인 source로 provenance를 바꾸는 식의 추정을 하지 않는다.

paragraph-level exact dedup은 반복 boilerplate를 줄이지만 document coherence를 훼손할 수 있다. paragraph를 제거한 뒤 빈 document와 문장 연결을 처리한다. 제거 비율이 높은 문서를 reject할지 남길지 결정한다. source span map은 삭제된 paragraph와 survivor source를 가리킨다.

incremental corpus에 새 shard를 추가하면 기존 index와 새 documents 사이, 새 documents 서로 사이를 모두 비교한다. index revision과 base corpus manifest를 고정한다. old winner 정책을 유지할지 새 고품질 document로 교체할지 release policy를 둔다. 교체는 downstream packed samples와 training lineage를 무효화할 수 있다.

**MinHash·LSH를 확률적 후보 생성기로 정확히 쓴다.**

**Jaccard와 signature를 연결한다.** document의 normalized shingles 집합을 A,B라 하면 Jaccard similarity는 `J=|A∩B|/|A∪B|`다. 무작위 permutation 아래 minimum hash가 같을 확률이 J라는 성질을 여러 independent hashes로 근사한다. 실제 구현은 permutation을 명시적으로 만들지 않고 universal hashing 계열을 사용한다.

signature 길이 `n`의 일치 비율은 J의 추정량이고 variance는 대략 `J(1-J)/n`이다. n을 늘리면 추정은 안정되지만 compute와 storage가 늘어난다. shingle size는 짧은 공통 표현과 긴 복제에 대한 민감도를 바꾼다. token shingle인지 word/character shingle인지 tokenizer와 normalization을 명시한다.

LSH banding에서 signature를 b개 bands, r개 rows로 나누면 similarity s인 pair가 적어도 한 band에서 모두 같아 candidate가 될 확률은 `1-(1-s^r)^b`다. 이것은 최종 duplicate 판정 확률이 아니라 candidate recall 곡선이다. candidate에 exact Jaccard 또는 다른 verifier를 적용할 수 있다. threshold라는 한 숫자로 n,b,r,verifier를 숨기지 않는다.

짧은 documents는 shingle 수가 적어 signature가 불안정하거나 빈 집합이 된다. min length, empty signature, repeated shingles를 별도 처리한다. boilerplate가 많은 pages는 높은 Jaccard로 묶이지만 핵심 content는 다를 수 있다. paragraph 또는 suffix-array 방법과 비교한다.

candidate edges가 만들어지면 connected components를 구할 수 있다. A-B, B-C가 threshold를 넘지만 A-C는 넘지 않아도 한 component가 된다. representative 하나를 제거하는 정책이 transitive closure를 얼마나 공격적으로 적용하는지 이해한다. edge threshold와 cluster survivor는 별도 실험 축이다.

**공개 구현의 좌표를 읽는다.** 로컬 DataTrove v0.10.0 snapshot은 commit `7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664`다. `src/datatrove/pipeline/dedup/minhash.py:41`의 `MinhashConfig`, `:124`의 signature 단계, `:324`의 bucket 단계, `:500`의 cluster 단계, `:599`의 filter 단계가 signature→candidate→component→survivor 공정을 함수 경계로 나눈다.

이 좌표는 현재 snapshot에서 확인한 시작점이다. 각 class의 constructor fields, input/output files, task partition, stats를 실제 source에서 읽는다. class 이름만 인용해 distributed determinism과 deletion lineage가 자동 보장된다고 말하지 않는다. 우리 pipeline manifest가 어떤 artifact를 추가해야 하는지 gap을 표에 남긴다.

MinHash test는 known Jaccard toy sets, identical/empty/short documents, permutation order, distributed shard reorder를 포함한다. 확률적 알고리즘이므로 random seed와 hash parameters를 고정한다. large-scale recall은 labeled duplicate pairs와 synthetic transformations로 추정하고 domain/language별로 분해한다.

**Bloom filter를 오염 제거와 중복 검출에 사용할 때의 손실.**

**false positive는 실제 삭제다.** m bits, k hash functions, n inserted elements인 Bloom filter의 false-positive 확률은 이상화하면 `(1-e^{-kn/m})^k` 근처다. false negative가 없다는 성질은 동일 hash/normalization과 손상 없는 structure를 전제로 한다. 구현 bug, version mismatch, incomplete build는 false negative를 만들 수 있다.

benchmark n-grams를 Bloom filter에 넣고 학습 document를 검사하면 false positive 때문에 무관한 문서나 span을 제거할 수 있다. corpus 규모가 매우 크면 작은 비율도 많은 tokens다. 예상 FP documents와 removed token mass를 계산하고 random negative sample로 관측한다. threshold와 number of matched n-grams 정책까지 포함한다.

filter를 shard별로 따로 만들고 결과를 합칠 때 capacity와 FP rate가 달라진다. union 가능한 bit arrays의 hash/config가 같은지 확인한다. incremental insertion으로 n이 설계치를 넘으면 FP가 증가한다. manifest에 capacity, actual insertions, bit checksum, hash seeds를 둔다.

DataTrove snapshot에서 `src/datatrove/pipeline/dedup/bloom_filter.py:24`의 `BloomFilterConfig`와 `:66`의 `SingleBloomFilter`, `pipeline/decont/n_grams.py:167`의 `NGramsDecontFilter`가 검토 출발점이다. filter build, match, remove action의 owner를 나눈다. source가 제공하는 통계와 corpus release에서 추가할 audit를 구분한다.

오염 제거는 exact string 한 종류가 아니다. benchmark prompt만, reference answer, prompt-answer pair, paraphrase, code solution, translated variant를 구분한다. n-gram exact matcher는 semantic paraphrase를 놓친다. embedding/classifier detector는 threshold와 false positives가 있다. detection evidence와 removal policy를 분리한다.

### 4.11.2 train/eval leakage를 계보와 영향으로 판정한다

**benchmark revision을 고정한다.** 같은 이름의 benchmark도 version, split, prompt template, few-shot examples가 달라진다. contamination index에는 dataset repository commit, file checksum, split, field selection, normalization, generated variants를 둔다. evaluation harness prompt가 원 dataset과 다른 문자열을 만들면 둘 다 검사한다.

학습 문서와 benchmark 사이 exact match가 발견되었다고 바로 memorization 영향이 증명되는 것은 아니다. 반대로 match가 없다고 독립성이 증명되지 않는다. 공개 웹에 정답 설명과 paraphrase가 있을 수 있다. contamination은 exposure evidence, similarity evidence, performance effect를 층으로 나눈다.

시간 기반 holdout은 평가 publication 이후 data를 제외하려는 방법이지만 web timestamp가 publication 시각과 같지 않다. crawl time, page publish/update time, repository commit time의 신뢰도를 표시한다. future benchmark를 만들 때 secret holdout과 access controls도 필요하다.

decontamination threshold를 지나치게 공격적으로 잡으면 benchmark domain과 유사한 합법적 학습 data를 제거해 distribution hole을 만든다. clean evaluation을 위해 training usefulness를 얼마나 잃는지 측정한다. removal count뿐 아니라 language/domain/length/token mass와 downstream effect를 본다.

benchmark가 red-team prompts나 harmful content를 포함하면 contamination scanner 자체가 민감 데이터를 처리한다. 접근 권한과 logging redaction을 둔다. secret test set을 모든 corpus workers에 복제하지 않고 privacy-preserving fingerprints 또는 제한된 service를 사용할 수 있지만 false-positive/negative와 threat model을 문서화한다.

**quality filter를 보이지 않는 데이터 목적함수로 읽는다.**

**heuristic은 값 판단을 코드로 고정한다.** 평균 문장 길이, alphabetic 비율, repeated line, terminal punctuation, stop-word 존재, symbol 비율 같은 rule은 빠르고 설명 가능하다. 그러나 언어·code·poetry·table에 따라 정상 분포가 다르다. 하나의 global threshold가 특정 writing style과 language를 체계적으로 제거할 수 있다.

DataTrove snapshot의 `src/datatrove/pipeline/filters/c4_filters.py:27` `C4QualityFilter`, `:139` `C4ParagraphFilter`, `:209` `C4BadWordsFilter`는 C4 계열 규칙을 함수 단위로 읽을 좌표다. `gopher_quality_filter.py:13`과 `gopher_repetition_filter.py:73`, `fineweb_quality_filter.py:8`은 서로 다른 rule families를 보여 준다. 각 filter의 exclusion reason과 thresholds를 추출해 표로 비교한다.

bad-word list는 단순 안전 filter가 아니다. 단어가 인용·교육·의학·피해 신고 맥락에 나타날 수 있고 소수자 표현을 과도하게 제거할 수 있다. substring과 token boundary, case normalization, inflection, multilingual coverage를 확인한다. document reject와 span redaction을 구분한다. list revision과 provenance를 보존한다.

repetition filter는 boilerplate와 generated spam을 줄일 수 있지만 refrain, legal template, code, tables도 제거할 수 있다. repeated n-gram 비율과 duplicate lines, top n-gram coverage가 length에 따라 어떻게 달라지는지 본다. threshold 주변 accepted/rejected pairs를 사람이 검토한다.

heuristic cascade는 순서에 따라 reason distribution과 비용이 달라진다. 앞 filter가 reject한 문서는 뒤 filter score가 없어 “뒤 조건을 통과했다”고 볼 수 없다. stage conditional pass rate와 common evaluation sample의 independent scores를 둘 다 보존한다. 하나의 primary reject reason만 저장해도 모든 matched reasons를 audit sample에서 유지한다.

**classifier score는 객관적 품질이 아니다.** high-quality reference와 web negatives로 classifier를 학습하면 reference selection과 feature가 score의 의미를 정한다. Wikipedia나 교과서 문체를 positive로 두면 대화체, 지역 언어, 실무 forum을 낮게 평가할 수 있다. model card에 training data, label protocol, metrics, calibration, known biases를 둔다.

threshold는 ROC나 PR curve와 corpus prevalence, compute budget, desired token mass를 함께 보고 정한다. balanced validation accuracy를 production precision으로 오인하지 않는다. domain shift에서 score calibration이 깨질 수 있다. language/domain별 score distribution과 manually labeled stratified sample을 본다.

hard threshold 대신 score-based sampling이나 mixture weight를 사용할 수 있다. 낮은 score를 모두 버리는 방식은 다양성을 잃고 경계를 날카롭게 만든다. temperature sampling과 score bins가 effective training distribution을 어떻게 바꾸는지 계산한다. 선택 정책 revision을 corpus manifest에 넣는다.

model-based filter는 자기 생성 텍스트나 특정 모델 선호를 강화할 수 있다. detector model revision, prompt/template, decoding config를 고정한다. nondeterministic API 결과를 cache할 때 raw response와 decision parser를 분리한다. provider가 model을 바꾸는 floating alias를 source evidence로 쓰지 않는다.

**language identification을 다국어 품질과 분리한다.**

**language label은 확률적 관측이다.** 짧은 text, code-switch, romanization, 이름, code에서는 불확실하다. top label만 저장하지 않고 score와 candidate labels, model revision, input view를 둔다. threshold 아래를 unknown으로 두는 정책과 multilingual documents 처리 방식을 정한다.

DataTrove snapshot의 `src/datatrove/pipeline/filters/language_filter.py:9` `LanguageFilter`와 `fasttext_filter.py:12` `FastTextClassifierFilter`는 현재 code path를 읽을 출발점이다. classifier model artifact checksum과 labels mapping이 source repository 밖에 있을 수 있으므로 별도 registry에 둔다.

language별 character set heuristic은 borrowed words, URLs, math, emoji를 고려한다. script와 language는 다르다. Serbian, Hindi-English code-switch처럼 하나의 script가 여러 language를 담거나 한 language가 여러 scripts를 쓴다. locale과 domain을 language label에서 추정하지 않는다.

language balance를 document count로 계산하면 평균 길이 차이를 숨긴다. raw bytes, characters, tokenizer tokens, valid training targets를 각각 센다. tokenizer fertility가 언어별로 다르면 token mixture와 byte mixture가 달라진다. 5장의 tokenizer audit와 연결해 desired sampling unit을 결정한다.

low-resource language에서 classifier error를 이유로 data를 더 제거하면 feedback loop가 생긴다. uncertainty sample을 native speakers와 검토하고 threshold를 다르게 둘 근거를 마련한다. synthetic translation과 original text를 provenance로 구분한다. 언어 다양성 지표를 overall quality score에 묻지 않는다.

**데이터 lineage를 관계형 원장으로 설계한다.**

**각 transform은 새 identity를 만든다.** raw WARC record에서 extracted document, normalized version, redacted version, dedup survivor, mixed corpus row, tokenized sample, packed sample로 갈 때 parent-child edge를 둔다. content가 같아도 stage와 revision이 다르면 다른 artifact다. mutable filename을 identity로 쓰지 않는다.

edge에는 transform symbol/version, config hash, execution attempt, input/output checksum, span map, decision reason이 있다. one-to-many split과 many-to-one merge를 표현한다. packed sample은 여러 document spans를 부모로 갖고, dedup cluster는 여러 members와 survivor를 가진다.

lineage graph는 모든 payload를 한 database에 복사하라는 뜻이 아니다. immutable manifests와 resolvable locators, checksums를 연결한다. storage retention으로 payload가 삭제되면 `PURGED` 상태와 정책 reason을 남긴다. `MISSING`과 구분한다. 접근 권한으로 볼 수 없는 artifact는 `RESTRICTED`이며 authorized resolver가 검증한다.

schema migration은 old IDs를 새 schema로 해석하는 방법을 버전 관리한다. field 이름을 바꾸며 의미까지 바뀌면 단순 alias가 아니다. inclusive/exclusive span, byte/character/token offset 단위를 type에 넣는다. timestamp timezone과 clock source도 명시한다.

**역방향 질문이 성능 requirement다.** 특정 UpdateID가 소비한 packed sample에서 raw documents로 돌아가야 한다. deletion request에서 영향을 받은 corpus releases와 tokenized shards, checkpoints를 찾을 수 있어야 한다. 모든 graph를 온라인 traversal하면 비싸므로 reverse indexes와 materialized impact tables를 만든다.

lineage completeness를 sample audit로 검증한다. accepted, rejected, dedup loser, redacted, tombstoned 상태에서 각각 forward와 reverse walk를 한다. terminal state가 없는 document, parent 없는 derived artifact, committed release가 참조하는 uncommitted shard를 오류로 본다.

graph 연결률 같은 요약 수치만으로 correctness를 보증하지 않는다. 잘못된 parent edge도 graph를 연결한다. content checksum과 interval conservation, count reconciliation을 함께 test한다. 고가치 benchmark와 deletion cases는 전수 검증한다.

**삭제 요청을 다음 학습 소비까지 전파한다.**

**요청 identity를 검증한다.** URL, domain, content fingerprint, author identity, legal request가 가리키는 범위는 서로 다르다. 요청자가 삭제 권한을 갖는지 판단하는 절차는 기술 score로 대체할 수 없다. request ID, scope, evidence, decision authority, effective date를 접근 통제 원장에 둔다.

source URL만 tombstone하면 mirrors와 exact/fuzzy duplicates가 남을 수 있다. content-derived cluster와 provenance relations를 이용해 후보 descendants를 찾고 사람이 승인할 수 있다. 너무 넓은 fuzzy cluster를 자동 삭제하면 무관한 문서를 제거할 수 있다. exact matches와 probable matches를 분리한다.

이미 tokenized/packed shards에 들어간 문서는 span-level mapping으로 영향을 찾는다. 새 corpus release에서 제외하고 loader가 old shard를 읽지 못하도록 release allowlist를 갱신한다. cache와 local copies, training workers의 prefetch도 고려한다. delete manifest revision이 loader startup gate에 포함되어야 한다.

이미 만들어진 checkpoint에서 특정 data의 영향을 제거하는 것은 단순 파일 삭제가 아니다. exact unlearning은 일반적으로 보장하기 어렵다. retraining, fine-tuned unlearning, access restriction 등 가능한 조치와 한계를 별도 정책으로 둔다. “데이터 삭제 완료”와 “모델 영향 제거 완료”를 구분한다.

deletion SLA는 request intake, corpus tombstone, next release exclusion, active job stop/restart, model action 단계로 나눈다. 각 단계의 evidence와 owner가 있다. 삭제 때문에 reproducibility artifact를 무기한 보존하지 않도록 retention과 legal hold 정책을 명시한다.

**corpus release를 재현 가능한 제품으로 출판한다.**

**release manifest를 중심에 둔다.** release ID, parent releases, source snapshots, pipeline/code commits, configs, schema, shard list/checksums, document/token/byte counts, language/domain mixture, filter/dedup/decontamination reports, known limitations를 포함한다. mutable `latest` 경로는 manifest resolver일 뿐 identity가 아니다.

각 shard는 임시 위치에서 쓰고 checksum·record count·schema validation 뒤 publish한다. 모든 shards가 준비되면 top manifest를 commit한다. loader는 commit marker와 checksum을 확인하고 fail-closed한다. partially uploaded release를 glob으로 읽지 않는다.

count reconciliation은 raw inputs에서 terminal states와 output을 맞춘다. `input=accepted+rejected+quarantined+errors`를 task와 global 수준에서 검사한다. dedup에서는 members, winners, losers, removed spans를 맞춘다. mixture에서는 source weights와 actual sampled token mass를 비교한다.

quality report는 headline token count만 보여주지 않는다. stage별 survival, reason, score distributions, language/domain/length, duplicates, PII, contamination, manual audit precision/recall 추정, uncertainty를 담는다. sample selection method와 confidence interval을 명시한다.

license와 usage restrictions는 document-level provenance에서 release 정책으로 집계한다. unknown license를 permissive로 추정하지 않는다. source attribution 요구, redistribution 제한, geographic or use constraints를 machine-readable policy와 human review로 연결한다.

release card에는 데이터를 쓰지 말아야 할 용도와 알려진 편향, 민감 content, opt-out 경로를 적는다. corpus가 크다는 이유로 모든 영역을 대표한다고 말하지 않는다. model training recipe는 실제 사용한 release ID와 mixture manifest를 인용한다.

**DataTrove executor의 재시도 의미를 source에서 읽는다.**

**local과 Slurm은 같은 pipeline을 다른 실행 환경에 배치한다.** DataTrove snapshot `7024ae...`의 `src/datatrove/executor/local.py:15` `LocalPipelineExecutor`와 `executor/slurm.py:48` `SlurmPipelineExecutor`를 비교한다. pipeline blocks가 같아도 task launch, dependencies, logs, retry, resource allocation이 다르다.

`PipelineStep`은 document stream을 읽고 쓰며 stats와 side effects를 가질 수 있다. retry-safe하려면 output path가 attempt-specific이고 최종 commit이 idempotent해야 한다. external classifier API나 mutable cache처럼 side effect가 있으면 정확히 한 번 의미를 별도로 설계한다.

stats는 단순 progress bar가 아니라 reconciliation evidence가 될 수 있다. 그러나 worker crash 전에 flush되지 않은 stats, retry 중복 stats, speculative attempts를 처리해야 한다. committed attempt의 stats만 global aggregate에 넣고 raw attempts도 debug용으로 보존한다.

resource request가 실제 stage의 memory/I/O/network profile에 맞는지 본다. MinHash bucket과 clustering은 skew가 크고, HTML extraction은 CPU, model classifier는 accelerator일 수 있다. 하나의 homogeneous job으로 묶으면 tail과 비용이 커진다. stage artifact 경계를 두어 resources와 retries를 독립시킨다.

FineWeb 로컬 snapshot `sources/training-datatrove-fineweb-current`의 commit은 `a649de79c14a550dc90f48a15c025f2dd3fd3b57`이다. examples와 configs를 읽을 때 DataTrove library revision과 조합을 기록한다. example path가 production release의 전체 private orchestration을 증명한다고 말하지 않는다.

**Dolma와 DCLM을 동일한 질문으로 비교한다.**

**Dolma는 annotation과 mixture 경계를 본다.** 로컬 `sources/training-dolma-toolkit` checkout의 commit을 manifest에서 확인하고 taggers, filters, mixers, CLI config와 tests를 source 좌표로 고정한다. annotation이 document bool인지 span attributes인지, filter가 payload를 제거하는지 tag만 남기는지 본다.

tag를 보존하면 threshold와 mixture policy를 나중에 바꿀 수 있지만 storage와 schema가 늘어난다. destructive filter는 단순하지만 재평가가 어렵다. 원 score와 model/config revision, decision을 분리해 저장한다. 여러 taggers가 같은 span에 다른 labels를 붙일 때 conflict policy를 둔다.

mixer는 source proportions를 documents, bytes, tokens 중 무엇으로 정의하는지 확인한다. oversampling은 동일 documents의 반복 exposure를 만든다. epoch 개념과 replacement sampling, seed, exhausted source 처리도 기록한다. planned weights와 실제 packed valid token weights를 비교한다.

**DCLM은 data intervention을 standardized training으로 평가한다.** 로컬 DCLM snapshot은 commit `361714bdd60bb9b7f4b2d8354cebbf0dec0c329e`다. pool, filtering, training recipe, evaluation harness의 경계를 source와 config로 고정한다. dataset score가 특정 model size와 compute recipe, benchmark set에 종속됨을 명시한다.

data ablation은 model/init/tokenizer/compute를 고정하고 corpus intervention만 바꾸려 한다. 그러나 token count를 고정해도 unique documents, duplication, language, sequence packing이 달라질 수 있다. run manifest로 실제 소비를 비교한다. 단일 aggregate benchmark 개선을 universal quality로 과장하지 않는다.

세 stack의 기능을 합칠 때 공통 schema adapter를 둔다. DocumentID, text, metadata, spans, tags, exclusion reason, stats 단위를 검증한다. adapter가 unknown fields를 버리는지 fail-closed한다. 각각의 source revision과 local extensions를 release manifest에 분리한다.

**대규모 corpus 공정의 장애와 관측성.**

**처리량은 records/sec 하나가 아니다.** fetched bytes, decompressed bytes, parsed documents, classifier tokens, signature pairs, output bytes를 stage별로 센다. CPU time, accelerator time, object-store requests, network bytes, cache hit를 연결한다. 병목을 다음 stage queue depth와 함께 본다.

data quality SLO에는 lineage completeness, checksum failure, unclassified errors, count reconciliation, duplicate recall estimate, PII incident, contamination removal lag가 있다. throughput이 높아도 unknown-error documents를 조용히 drop하면 release는 실패다.

skew는 domain giant, huge document, common MinHash bucket, language classifier slow path에서 생긴다. task duration p50/p95/max와 input size를 비교한다. hot keys를 split하되 deterministic component construction을 유지한다. straggler speculative retry가 duplicate output을 만들지 않게 commit을 둔다.

object storage eventual consistency와 transient 404를 permanent missing으로 처리하지 않는다. checksum mismatch, truncated read, decompression error를 retryable과 corruption으로 나눈다. 동일 shard가 반복 실패하면 quarantine하고 release commit을 막는다. silent skip하지 않는다.

metrics label에는 URL과 document ID를 넣지 않는다. stage, reason, language bucket, size bucket처럼 제한된 labels를 쓴다. 상세 sample은 access-controlled artifact에 correlation ID로 연결한다. PII detector match 자체가 민감할 수 있다.

alert 뒤 자동 재시도는 input/config checksum이 같은 attempt만 허용한다. code revision이 바뀌면 새 pipeline run이다. threshold를 운영 중 hot patch하면 앞뒤 documents가 다른 policy를 받으므로 release를 분할하거나 전량 재처리한다.

**corpus 공정의 수학적 편향을 추적한다.**

**filter는 selection probability를 만든다.** raw distribution `P(x)`에서 accept function `a(x)∈[0,1]`을 적용하면 retained distribution은 `P'(x)∝a(x)P(x)`다. deterministic threshold도 a가 0/1인 특수 경우다. model은 “웹”이 아니라 이 선택된 분포를 학습한다.

여러 filters를 cascade하면 acceptance는 조건부 곱으로 나타나지만 stages가 같은 features에 의존해 독립은 아니다. stage별 pass rate를 곱해 subgroup 최종 recall을 추정할 때 dependence를 고려한다. common holdout에서 end-to-end decision을 측정한다.

dedup은 repeated observations의 weight를 줄인다. duplicates가 실제 세계 빈도를 반영하는 경우와 배포 artifact의 복제를 반영하는 경우를 구분하기 어렵다. exact/fuzzy threshold와 survivor policy는 암묵적 reweighting이다. cluster size distribution과 domain별 removed mass를 본다.

mixture sampling weight `w_d`와 source token count `N_d`가 있을 때 expected exposure는 sampling scheme에 따라 달라진다. temperature sampling은 낮은 resource source를 올릴 수 있지만 반복률을 높인다. unique token coverage와 expected epochs를 함께 계산한다.

quality score를 importance weight처럼 사용한다면 score calibration과 normalization이 필요하다. hard removal보다 variance가 커질 수 있고 high-weight outlier가 gradient를 지배할 수 있다. 1장의 loss weight와 6장의 curriculum scheduler로 연결한다.

이 수학은 “편향을 제거한다”는 약속이 아니다. 어떤 선택이 어떤 population의 확률 질량을 바꾸는지 가시화한다. normative decision은 dataset goals와 governance에서 내리고, 기술 pipeline은 그 결정을 숨김없이 실행·측정한다.

**모델 생성 데이터를 웹 데이터와 섞을 때의 경계.**

**synthetic이라는 한 label로 충분하지 않다.** 생성 model과 revision, prompt source, system template, decoding parameters, rejection/filter, human edit, seed를 provenance에 둔다. 원 web document의 요약·재작성이라면 원본과 derived relation을 연결한다. fully synthetic prompt와 response, distillation output, self-play trajectory를 구분한다.

생성 데이터가 다시 웹에 공개되고 후속 crawl에 들어오면 model-generated text의 순환이 생긴다. watermark나 phrase detector는 불완전하다. known publication URLs, generation manifests, content fingerprints를 이용해 식별 가능한 부분을 표시한다. “합성 탐지 100%”를 주장하지 않는다.

teacher model의 오류와 style이 대량 복제될 수 있다. quality filter가 teacher와 유사한 model이면 자기 선호를 강화한다. independent verifier, rule checks, human audit, diversity measurement를 조합한다. accepted rate보다 factuality, task correctness, novelty, style concentration을 본다.

synthetic data의 duplication은 exact text뿐 아니라 prompt templates와 reasoning pattern에서 나타난다. instruction skeleton과 response n-grams, semantic clusters를 측정한다. 너무 강한 semantic dedup은 유효한 반복 연습을 제거할 수 있으므로 curriculum goal과 연결한다.

model license와 generated output usage terms, source data rights를 provenance에서 분리한다. human contributors의 consent와 compensation도 dataset card에 기록한다. 합성이라는 이유로 개인정보와 유해 content risk가 사라지지 않는다. prompt에 포함된 원문과 generated leak을 검사한다.

**code·수학·대화 데이터의 별도 공정을 설계한다.**

**code는 repository 구조를 잃으면 의미가 달라진다.** file text만 떼면 import relation, license, commit time, generated/vendor files를 잃는다. repository commit, path, blob hash, language, dependency metadata를 보존한다. fork와 vendored copy의 dedup은 file hash와 repository graph를 함께 본다.

secret, personal email, copyrighted header, binary blob, minified/generated code를 별도 detector로 처리한다. generic prose quality rule의 punctuation과 word-length threshold를 code에 적용하지 않는다. parser가 실패한 language와 notebook cell, mixed Markdown/code를 분리한다.

benchmark contamination에서 code problem statement, canonical solution, tests, translated variants를 검사한다. 함수 이름과 boilerplate만 일치하는 것은 contamination 증거가 약할 수 있다. AST/token fingerprints와 exact spans를 함께 보되 false positives를 사람이 검토한다.

**수학 문서는 symbol 보존이 핵심이다.** HTML MathML, LaTeX source, rendered alt text 중 어느 것을 선택하는지 기록한다. Unicode normalization이 수학 alphabet을 합치거나 backslash와 braces를 제거하지 않는지 본다. equation과 주변 prose span을 연결한다. OCR 수식은 confidence와 image provenance를 둔다.

**대화·forum은 speaker와 thread가 구조다.** quoted reply와 original post가 중복될 수 있다. user handle과 개인 정보 redaction, deleted posts, moderation labels를 처리한다. thread flattening 순서와 speaker boundaries를 보존한다. chat template를 미리 삽입하지 않고 structured roles로 넘겨 5장과 17장에서 revision을 선택할 수 있게 한다.

각 modality-like text family에는 별도 quality report와 sampling weight를 둔다. 하나의 classifier score로 통합하지 않는다. 최종 mixer에서 공통 token budget으로 만날 때 source family와 transform lineage를 유지한다.

**red-team 데이터가 corpus pipeline에 들어오는 방식.**

**공격 prompt와 일반 유해 텍스트를 구분한다.** red-team record에는 공격 목표, threat category, prompt, target behavior, model response, severity, annotator decision이 있을 수 있다. 웹 문서와 같은 free text schema로 평탄화하면 어느 부분을 학습해야 하는지 잃는다. structured artifact로 보존하고 SFT·preference·evaluation split을 명시한다.

red-team examples는 학습과 평가에 동시에 들어가면 leakage가 생긴다. campaign, attack family, semantic cluster 단위 split을 고려한다. exact string dedup만으로 paraphrased attacks의 leakage를 막지 못한다. secret evaluation prompts는 제한된 fingerprints로 corpus를 검사한다.

유해 prompt를 SFT refusal 데이터로 바꾸면 response quality와 over-refusal을 함께 평가해야 한다. preference pair라면 chosen/rejected provenance와 annotator policy가 필요하다. RL environment에서 생성된 trajectory면 policy/reference/reward revisions와 연결한다. 같은 원문이 단계별로 다른 학습 unit이 된다.

민감 content 접근 권한, annotator 보호, retention을 둔다. raw harmful examples를 일반 debug log와 책 본문에 불필요하게 복사하지 않는다. taxonomy와 detector 성능은 공개할 수 있어도 재현 가능한 공격 payload는 위험 평가를 거친다.

pipeline은 red-team record를 무조건 high-quality 또는 reject로 분류하지 않는다. 목표 corpus와 safety stage가 결정한다. foundation pretraining, safety SFT, reward model, eval은 서로 다른 inclusion policy를 갖는다. lineage graph가 동일 source에서 이 네 descendants를 구분한다.

**multimodal 웹 데이터의 공통 lineage를 준비한다.**

**텍스트만 남기면 정렬 정보를 잃는다.** image의 URL·bytes·MIME·dimensions·checksum과 caption/alt/OCR span, page context를 연결한다. audio/video에는 media checksum, segment timestamps, transcript spans, speaker/language metadata가 있다. 원 media에서 derived frames·clips·tokens로 가는 interval map을 둔다.

broken links와 hotlink replacement 때문에 media URL은 identity가 아니다. content digest와 crawl locator를 사용한다. perceptual hash는 near-duplicate 후보를 만들지만 crop, resize, watermark, re-encoding에 대한 threshold를 검증한다. exact byte duplicate와 perceptual cluster를 분리한다.

caption quality는 text fluency만으로 판단하지 않는다. image-content alignment, OCR hallucination, language, harmful content, personal identity를 본다. CLIP-like score threshold는 model bias와 domain shift를 갖는다. score distribution과 human audit를 domain별로 본다.

video frames를 균일 sampling하면 짧은 events를 놓치고 정적 frames를 반복한다. shot boundary, motion, audio alignment를 고려한다. transcript time offset과 subtitle provenance를 보존한다. deleted media의 derived frames와 embeddings를 역추적할 수 있어야 한다.

멀티모달 tokenization은 26장에서 다루지만 corpus 단계는 raw resolution, preprocessing revision, crop/resize, codec를 보존해야 한다. image patch/token 수와 audio duration이 최종 token budget과 batch packing을 바꾼다. mixer는 text token만으로 modality weight를 정의하지 않는다.

### 4.11.3 planned mixture를 realized token mass로 바꾼다

**난이도 score를 품질 score와 같게 두지 않는다.** 쉬운 문서가 저품질이고 어려운 문서가 고품질인 것은 아니다. length, perplexity, vocabulary rarity, reasoning annotation, domain, format complexity는 서로 다른 curriculum features다. score source와 revision, leakage risk를 기록한다.

model-based perplexity로 난이도를 계산하면 scorer model과 tokenizer의 편향이 들어간다. training model과 같은 family면 자기 familiarity를 측정할 수 있다. raw score와 language/length-normalized score를 함께 보고 curriculum의 목적을 명시한다.

curriculum scheduler가 stage별로 source weights를 바꾸려면 corpus release가 stable strata와 counts를 제공해야 한다. 각 stratum의 document/token/valid-target 예상량, duplication, quality, contamination status를 둔다. online loader가 실제 소비한 mixture와 planned mixture를 비교한다.

data ordering은 optimizer state와 결합한다. 같은 multiset도 쉬운→어려운 순서와 shuffle이 다른 trajectory를 만든다. sampler seed, epoch, temperature, replacement, buffer size를 checkpoint에 저장한다. corpus manifest와 6장의 schedule revision을 연결한다.

dynamic curriculum이 model loss로 sampling weight를 갱신하면 feedback loop와 distributed synchronization이 생긴다. 어느 checkpoint의 score인지, stale score를 얼마나 허용하는지, outlier가 oversample되지 않게 cap하는지 정한다. data pipeline이 training metric을 읽는 edge를 lineage에 표시한다.

**검증 corpus와 비용·증거 원장을 함께 설계한다.**

**test corpus를 계층적으로 만든다.**

**unit fixtures는 변환 하나를 겨냥한다.** HTML parser, normalization, language classifier, PII redaction, exact hash, MinHash signature, Bloom match마다 작은 expected input/output을 둔다. invalid bytes, empty, huge line, multilingual, code, duplicates와 boundary thresholds를 포함한다.

**integration fixture는 stage composition을 겨냥한다.** 동일 문서가 redaction 전후 dedup에서 어떻게 달라지는지, extract span이 tokenizer offset으로 이어지는지, reject writer가 reason과 payload locator를 보존하는지 본다. stage 순서를 의도적으로 바꾸어 output difference를 예상한다.

**distributed fixture는 partition과 retry를 겨냥한다.** 같은 input을 task count와 completion order를 바꿔 처리하고 committed output set과 survivor가 같은지 본다. task 하나를 crash·retry하고 stats가 중복되지 않는지 확인한다. speculative attempts의 output이 하나만 publish되는지 본다.

**scale audit는 tail을 겨냥한다.** production snapshot에서 stratified samples를 뽑아 domain, language, size, score boundary, reject reason을 사람이 검토한다. sample design과 confidence interval을 기록한다. unit PASS를 web 전체 precision 보증으로 확대하지 않는다.

**adversarial fixture는 우회를 겨냥한다.** zero-width characters로 PII를 숨기고, benchmark text에 punctuation을 삽입하고, duplicate 문서를 boilerplate로 둘러싼다. detector를 공격하는 예제를 공개할 때 위험을 평가하되 내부 regression suite에는 충분한 변형을 둔다.

expected artifacts는 framework upgrade 때 자동 overwrite하지 않는다. diff를 사람이 검토해 의도한 변화와 regression을 나눈다. source commit, config hash, detector model checksum을 fixture result에 넣는다. 실행하지 않은 대규모 recall 실험은 `NOT_RUN`으로 표시한다.

**비용 원장을 데이터 품질 원장과 연결한다.**

**비용 단위를 stage output과 묶는다.** fetch dollar/GB, extraction CPU-hour/TB, classifier GPU-second/million tokens, dedup storage/signature, object-store requests를 계산한다. accepted token당 비용과 rejected document 검사 비용을 분리한다. 단순 total bill은 어떤 filter가 비용을 만들고 어떤 가치가 있는지 말하지 않는다.

비싼 detector를 sampling하면 Horvitz–Thompson 같은 inclusion probability 기반 추정이 필요할 수 있다. review sample이 score boundary를 oversample했다면 raw 비율로 corpus prevalence를 추정하지 않는다. sample weight와 stratum size를 보존한다.

cache hit는 비용을 줄이지만 revision invalidation을 지켜야 한다. detector code/model/config, normalized input checksum이 key에 포함된다. cache corruption과 partial write를 checksum으로 잡는다. cache 없이 재실행한 sample과 비교해 stale result를 탐지한다.

비용 최적화 proposal마다 품질 영향 반경을 쓴다. early truncation은 긴 문서 tail의 PII/contamination recall을, lower MinHash permutations는 duplicate recall variance를, larger tasks는 retry waste와 straggler를 바꾼다. throughput 숫자와 함께 예상 selection bias를 제시한다.

carbon/energy report를 한다면 측정 경계, hardware, utilization, region electricity assumptions를 명시한다. 추정치를 정밀한 실측처럼 제시하지 않는다. 재처리를 줄이는 immutable artifacts와 cache는 비용뿐 아니라 재현성도 높일 수 있다.

**corpus review 회의에서 묻는 증거 질문.**

**수집.** source snapshots와 정책 revision이 고정되었는가. 모든 input records가 terminal state에 도달했는가. retries와 speculative attempts가 output을 중복시키지 않는가. WARC locator와 payload checksum으로 sample을 재생할 수 있는가.

**추출.** raw bytes에서 output spans로 돌아갈 수 있는가. parser upgrade의 stratified diff가 있는가. code, table, math, multilingual tail을 검토했는가. silent decoding replacement를 계측하는가.

**필터.** 각 threshold가 어떤 population을 제거하는지 아는가. classifier training data와 calibration, known bias가 있는가. reasons와 raw scores를 분리해 보존하는가. PII와 rights decisions에 human escalation이 있는가.

**중복.** signature view와 parameters, candidate probability, verifier, component, survivor가 분리되어 있는가. distributed order가 바뀌어도 survivor가 deterministic한가. winner 삭제 시 cluster descendants를 처리할 수 있는가.

**오염.** benchmark revision과 generated prompt variants가 고정되었는가. exact·n-gram·semantic evidence를 구분하는가. false-positive token mass와 semantic false negatives를 추정하는가. train/eval split lineage가 닫혔는가.

**출판.** 모든 shards가 committed manifest와 checksum으로 닫혔는가. count reconciliation과 quality report가 있는가. loader가 mutable glob 대신 release allowlist를 읽는가. deletion manifest와 active jobs가 연결되는가.

**학습 인계.** tokenizer가 읽을 text와 offsets, source metadata가 있는가. mixer/curriculum의 strata와 실제 token mass를 계산할 수 있는가. packed sample에서 raw document로 역추적할 수 있는가. UpdateID가 corpus release와 delete revision을 기록하는가.

회의는 모든 질문에 “예”라고 답하기 위한 의식이 아니다. 증거가 없는 항목은 risk와 owner, 완료 조건을 기록한다. release 규모가 크다는 사실이나 유명 pipeline 이름이 이 질문을 대신하지 않는다.

**한 문서의 전체 생애를 추적한다.**

웹 response 하나가 WARC shard의 byte range와 checksum을 받는다. parser revision이 DOM과 extracted text를 만들고 span map을 남긴다. normalization과 PII stage가 새 text revision과 decisions를 만든다. language·quality filters는 raw scores와 reasons를 붙인다.

exact·MinHash stages가 signatures와 candidate edges를 만들고 connected component와 deterministic survivor를 정한다. contamination scanner는 benchmark revision에 대한 match spans와 confidence를 붙인다. accepted document는 committed corpus shard와 release manifest에 들어간다.

tokenizer와 packer는 이 document의 spans를 token IDs와 packed indices로 옮긴다. loader consumption ledger는 UpdateID와 연결한다. 나중에 삭제 요청이 오면 source identity에서 document revisions, dedup cluster, corpus releases, tokenized/packed shards, active runs로 역방향 impact를 계산한다.

이 생애에서 payload가 사라진 edge에는 retention reason을, 볼 수 없는 edge에는 access policy를, 변환된 edge에는 revision과 checksum을 기록한다. 단순 URL 목록이나 최종 JSONL만으로는 이 왕복을 할 수 없다. lineage는 부록 문서가 아니라 데이터 공정의 control plane이다.

독자가 이 장을 이해했다는 기준은 FineWeb, Dolma, DCLM의 이름을 말하는 것이 아니다. 고정 source의 실제 block과 executor, signature·filter 경계를 읽고 자기 corpus의 빠진 계약을 찾아낼 수 있어야 한다. threshold 하나를 바꿨을 때 어느 documents와 token mass, benchmark contamination, training updates가 영향을 받는지 추적할 수 있어야 한다.

최종 인계물은 corpus bytes만이 아니다. release manifest, source registry, transform graph, decision records, quality/contamination report, deletion index, tokenizer handoff schema, tests와 `NOT_RUN` 목록이 함께 간다. 이 묶음이 닫혀야 다음 장의 tokenizer가 무엇을 어떤 근거로 분절하는지 설명할 수 있다.

**작은 corpus로 모든 보존식을 손검산한다.**

**다섯 문서를 만든다.** A는 정상 한국어 HTML, B는 A와 URL만 다른 exact text, C는 A의 문단 순서와 공백을 조금 바꾼 near duplicate, D는 benchmark 정답 span을 포함한 문서, E는 email과 code block이 섞인 문서다. 각 raw payload에 DocumentID와 byte checksum을 부여한다.

extractor가 A와 B에서 같은 normalized text를 만들면 exact signature가 같다. survivor 정책이 timestamp보다 license를 우선한다면 입력 순서가 바뀌어도 같은 winner여야 한다. B는 loser이지만 source provenance와 cluster membership을 유지한다. winner A가 삭제될 때 B를 자동 승격할지 cluster 전체 tombstone인지 test한다.

C는 shingle 집합과 Jaccard를 손으로 계산할 수 있게 짧게 만든다. MinHash signature를 작은 hash family로 계산하고 LSH bands의 candidate 여부를 확인한다. production hash 품질을 이 toy가 증명하지는 않지만 stage ordering과 component logic을 검산한다. C가 A와 edge를 만들고 B와도 연결되는 경우 component survivor를 확인한다.

D의 benchmark span을 normalization 전후에 추적한다. n-gram matcher가 exact hit를 만들고 Bloom false positive를 별도 synthetic negative에서 주입한다. removal policy가 전체 document reject인지 span removal인지에 따라 output count와 text checksum을 예측한다. benchmark ID와 match span이 evidence에 남는다.

E는 PII detector가 email을 redact하되 code block의 example pattern을 어떻게 처리하는지 보여 준다. redaction 전후 dedup signatures가 달라지는지 확인한다. raw payload는 제한된 artifact에 남고 공개 output에는 replacement와 decision record만 있다. code quality rule은 prose rule과 다른 branch를 탄다.

다섯 input은 terminal states로 reconciliation된다. exact loser, accepted survivor, fuzzy loser 또는 accepted C, contaminated reject D, redacted accepted E의 합이 5여야 한다. retries를 한 번 주입해도 committed outputs와 counts는 같아야 한다. task completion order를 바꿔도 release shard의 logical record set과 survivor가 같다.

**release를 두 번 만든다.** R1 뒤 filter threshold와 parser revision을 하나씩 바꾼 R2를 만든다. parser-only child run은 extraction 이후에서, threshold-only child run은 score decision 이후에서 first difference가 나야 한다. source input checksum이 달라지면 실험 통제가 실패한 것이다.

R1/R2 diff에는 added, removed, modified documents와 reasons, token mass, language/domain shifts가 있다. text checksum이 같은데 metadata만 바뀐 경우를 구분한다. tokenizer revision을 고정해 token count difference도 계산한다. downstream packed samples의 invalidation 범위를 산출한다.

마지막으로 A에 대한 삭제 요청을 넣는다. exact cluster와 R1/R2 descendants, tokenized fixture, consumption ledger까지 impact를 찾는다. 이미 존재하는 checkpoint를 자동으로 “정화됨”이라 표시하지 않는다. next release exclusion과 model action status를 별도 기록한다.

이 작은 corpus는 웹 전체의 quality를 대표하지 않는다. 대신 identity, transformation, dedup, contamination, redaction, publish, deletion의 control-flow를 완전히 검산한다. 실제 scale audit는 별도의 stratified samples와 통계가 담당한다.

**흔한 데이터 주장에 필요한 반증 조건.**

**“중복을 제거했다.”** exact인지 fuzzy인지, document·paragraph·sentence 중 어느 단위인지, normalization과 threshold, candidate recall, survivor policy를 제시해야 한다. 입력 순서와 shard 수를 바꿔도 logical result가 같은지 본다. 알려진 duplicate benchmark에서 recall/precision과 subgroup을 보고한다.

**“고품질 웹 데이터다.”** quality의 operational definition과 label source, classifier/heuristic revisions, threshold, manual audit가 필요하다. accepted sample만 보여 주지 않고 rejected와 boundary sample을 검토한다. language/domain별 selection rate와 uncertainty를 제시한다.

**“개인정보를 제거했다.”** PII taxonomy, detector coverage, precision/recall 추정, redaction/reject policy, incident handling을 제시한다. false negatives가 없다고 말하지 않는다. raw 접근 권한과 deletion propagation을 검증한다.

**“benchmark 오염이 없다.”** 검사한 benchmark revisions와 fields, normalization, exact/n-gram/semantic 방법, thresholds, false-positive/negative 한계를 제시한다. 공개 paraphrase와 hidden tests의 범위를 구분한다. removal이 downstream release와 실제 loader에 적용됐는지 확인한다.

**“재현 가능한 corpus다.”** source snapshots, code/config commits, immutable manifests, shard checksums, deterministic/seeded operations, external artifacts가 필요하다. raw source가 법적·운영상 더 이상 접근 불가할 수 있는 한계를 적는다. 같은 manifest로 sample을 재생해 checksum을 확인한다.

**“삭제를 지원한다.”** request scope에서 descendants와 active consumers로 가는 reverse lineage, tombstone enforcement, SLA와 model-level limitation을 제시한다. URL 하나를 blocklist에 넣는 것으로 완료하지 않는다.

주장마다 반례와 검증 방법을 붙이면 dataset card가 홍보 문서에서 engineering contract로 바뀐다. 독자는 이름이나 token 규모보다 어떤 증거가 빠졌는지 질문할 수 있다.

**5장으로 넘기는 정밀 인계서.**

tokenizer 팀은 committed release ID와 shard allowlist, document schema, text encoding, normalization 상태, source/segment IDs와 span maps를 받는다. special content flags, language/domain, code/math/dialogue structure, redaction markers가 있다. mutable URL이나 처리 중 directory를 받지 않는다.

각 document에는 content checksum과 length in bytes/characters가 있다. tokenizer output은 TokenizerRevision과 token IDs, offsets, error/fallback events를 새 descendants로 만든다. normalization을 tokenizer 내부에서 다시 수행한다면 corpus text와 tokenizer-normalized text의 diff를 lineage에 추가한다.

packing 팀은 document boundaries, EOS policy 후보, license/privacy grouping constraints를 받는다. 서로 섞으면 안 되는 restricted documents와 evaluation holdouts를 표시한다. dedup cluster ID와 source family를 전달해 exposure와 split leakage를 계측한다.

training 팀은 corpus release뿐 아니라 mixer/curriculum manifest를 사용한다. planned source weights와 actual valid target consumption을 비교한다. deletion revision이 바뀌면 loader startup 또는 checkpoint resume에서 허용 release를 다시 검증한다. 오래 열린 workers가 stale manifest를 계속 읽지 않게 한다.

evaluation 팀은 contamination report와 benchmark-specific match evidence를 받는다. match가 있는 items를 제외한 metric, contamination strata, temporal holdout을 계획할 수 있다. red-team secret set의 raw content는 권한 없이 노출하지 않고 승인된 검사 결과만 연결한다.

인계 승인은 임의 문서 하나를 raw WARC에서 tokenizer 입력까지 왕복하고, reject 하나가 release에 없음을 증명하며, delete request 하나가 모든 committed descendants에서 차단됨을 확인할 때 이루어진다. count와 checksum, source revisions가 없으면 prose 설명이 충분해도 승인하지 않는다.

이 인계서 덕분에 5장의 토크나이저는 출처 불명의 문자열을 분절하지 않는다. 어떤 선택과 손실을 거쳐 남은 text인지 알고, token 경계가 raw source와 어떻게 연결되는지 보존한다. 데이터 공정의 끝은 파일 전달이 아니라 다음 함수가 검증 가능한 입력을 받는 시점이다.

**FineWeb pipeline을 DataTrove block graph로 읽는다.**

FineWeb을 “Common Crawl을 깨끗하게 만든 데이터”라는 문장으로 줄이지 않는다. 어느 crawl dump를 어떤 reader가 읽고, URL·language·quality·repetition filter와 dedup을 어떤 순서·parameter로 적용해 어느 writer에 commit했는지가 재현 단위다.

고정 DataTrove revision `a649de79c14a550dc90f48a15c025f2dd3fd3b57`의 `examples/fineweb.py:26-47`은 dump metadata와 language/dump/rank를 output 구조에 연결하는 기준 좌표다. example은 block composition을 보여 주지만 특정 공개 release가 실제 사용한 모든 infrastructure·rights decision을 자동 증명하지 않는다.

PipelineStep은 document stream을 받아 text·metadata·stats와 terminal decision을 만든다. reader, extractor/filter, dedup signature/index/filter와 writer 사이에서 DocumentID와 source locator가 보존되는지 확인한다. worker-local enumeration을 durable ID로 오인하지 않는다.

executor option은 tasks, workers, retries, logging·stats와 output ordering을 바꾼다. retry가 같은 logical task output을 중복 commit하지 않도록 attempt와 idempotency를 둔다. speculative execution이 있으면 winner selection과 loser cleanup을 시험한다.

FineWeb식 dump-local dedup과 global dedup은 sampling distribution이 다르다. 다른 crawl 시점의 같은 page를 남길지 제거할지가 바뀐다. 같은 token budget에서 downstream small model/evaluation과 freshness·domain mass를 비교한다. dedup이 많을수록 무조건 좋다고 쓰지 않는다.

filter ablation은 accepted count뿐 아니라 language/domain/length·token mass와 evaluation을 함께 본다. 한 threshold가 다른 crawl과 language에서 같은 quality boundary라는 보장은 없다. raw score, calibration·manual boundary audit와 reason을 보존한다.

**C4 recipe에서 heuristic의 선택 편향을 추적한다.**

C4/T5 계열 공개 recipe는 Common Crawl WET text에서 language·heuristic cleaning, dedup와 blocklist를 적용하는 중요한 역사적 기준이다. TFDS revision `00a6c1cbe049634e1cfb823a910b83d6cb358ac2`의 builder/config와 generation path를 고정해 actual option을 읽는다.

line·document heuristic은 terminal punctuation, word count, boilerplate와 bad-word 등을 기준으로 text를 제거할 수 있다. 이것은 단순 noise reduction이 아니라 특정 문체, dialect, identity 표현과 domain의 selection probability를 바꿀 수 있다. accepted/rejected boundary sample을 strata별로 audit한다.

blocklist match는 context와 polysemy를 이해하지 못한다. precision을 높이려고 threshold를 바꾸면 false negative가 늘 수 있다. 법적·safety policy와 quality score를 한 detector에 숨기지 않는다. reason taxonomy와 raw evidence를 분리한다.

TFDS builder가 shard를 생성·download/cache하는 경로와 corpus semantic recipe를 구분한다. cache hit에도 raw/source revision과 prepared dataset fingerprint가 맞는지 본다. published cardinality나 split만으로 URL-level lineage와 deletion propagation을 추론하지 않는다.

C4의 유용한 교훈은 heuristic을 숨기지 않고 재현 가능한 code로 공개했다는 점과, 공개됐다는 사실이 편향·rights·삭제 문제를 자동 해결하지 않는다는 점을 함께 보는 것이다. 동일 recipe를 현재 crawl에 적용하면 input population 차이로 output distribution이 달라진다.

**Dolma의 annotation sidecar와 Bloom 경계를 검증한다.**

Dolma toolkit revision `669f534823b08d266a8fff01f8a1c916a5a56576`은 document text와 별도의 attributes/annotations를 통해 span·document decision을 표현할 수 있다. 원 text를 매 stage 재작성하는 방식과 달리 detector evidence를 sidecar로 보존하는 장점이 있다.

attribute는 source document identity, span coordinate unit, annotator revision, score/reason과 생성 시각을 가져야 한다. normalization 전 byte와 후 character coordinate가 섞이면 redaction·contamination span을 잘못 적용한다. overlap·boundary와 stale annotation을 fixture로 둔다.

Bloom filter를 dedup/decontamination에 쓰면 bit 수 \(m\), insert count \(n\), hash 수 \(k\)에서 false-positive 근사는 \((1-e^{-kn/m})^k\)다. 목표 FP와 estimated count로 sizing하더라도 실제 key distribution·count와 filter saturation을 관측한다.

일반 Bloom membership은 삽입한 key를 놓치지 않는 구조를 목표로 하지만 normalization/key segmentation이 다르면 이론 밖 false negative가 생긴다. false positive는 새로운 문서를 duplicate/contaminated로 제거해 data mass를 바꾼다. filter binary digest와 key recipe를 release evidence로 둔다.

read-only benchmark filter는 contamination candidate를 빠르게 찾을 수 있지만 어떤 benchmark item·span과 맞았는지 설명 가능한 verifier가 추가로 필요하다. Bloom hit 하나를 법적 삭제나 semantic contamination 최종 판정으로 쓰지 않는다.

Dolma release의 source mixture와 toolkit capability를 구분한다. tool이 attribute를 지원한다는 사실은 모든 release source가 동일 rights·deletion SLA를 가진다는 뜻이 아니다. dataset card와 exact release manifest를 함께 읽는다.

**공개 pipeline의 선택 편향과 재현성을 비교한다.**

**DCLM을 controlled data experiment로 이해한다.**

DCLM revision `361714bdd60bb9b7f4b2d8354cebbf0dec0c329e`의 `baselines/core/processor.py:54-101`은 input shard별 processed output과 stats sidecar를 만드는 경계다. `ray_processing/tokenize_shuffle.py:61-107`은 tokenizer·input/output과 shuffle workflow를 결합한다.

`tools/expdb.py:339-411`의 experiment database는 dataset/model/eval UUID를 merge해 비교 계보를 만드는 좌표다. UUID가 존재한다는 사실만으로 raw document→packed token→checkpoint의 universal lineage가 닫히는 것은 아니다. namespace crosswalk가 필요하다.

DCLM의 핵심 관점은 같은 model/compute/evaluation 조건에서 data curation 후보를 비교하는 controlled experiment다. filter score가 높다는 proxy만 보지 않고 downstream model metric으로 data recipe를 비교한다. 그러나 benchmark selection과 작은 proxy model 결과를 모든 model scale·domain에 일반화하지 않는다.

quality classifier threshold는 training label, raw pool과 calibration에 의존한다. 다른 corpus·language에 옮기면 score distribution과 FP/FN이 달라질 수 있다. threshold 전후 boundary sample, domain·language token mass와 small run을 새로 audit한다.

**분류기의 데이터 계보도 오염 검사 대상이다.** quality classifier의 학습 positive/negative, calibration split, threshold를 고른 downstream evaluation이 최종 corpus 후보와 겹치면 두 종류의 누수가 생긴다. 첫째, classifier가 이미 본 문서의 표면 특징을 높은 점수로 되돌려 주는 selection leakage다. 둘째, benchmark 점수로 threshold나 classifier checkpoint를 골라 그 benchmark에 맞는 corpus를 만드는 adaptive evaluation leakage다. 최종 model이 benchmark 원문을 직접 소비하지 않았더라도 선택 정책이 benchmark 피드백을 전달할 수 있다.

따라서 `ClassifierTrainingSetID → ClassifierRevision → ScoreArtifact → FilterDecision → CorpusRelease`와 `BenchmarkItemID → SelectionRunID → ThresholdRevision`을 별도 간선으로 남긴다. exact normalized hash, paragraph/n-gram overlap, semantic neighborhood를 차례로 검사하고, hit를 단순 삭제하지 말고 classifier train·calibration·selection-eval 중 어느 역할에서 겹쳤는지 표시한다.

paired ablation은 overlap cohort를 제외한 classifier와 원 classifier가 같은 고정 raw universe에서 만든 accepted token·domain·language 질량과 downstream 결과를 비교한다. 차이가 없다는 결과도 검사한 detector·threshold·모델 규모 밖의 보편적 무누수를 증명하지 않는다.

baseline recipe와 repository에 존재하는 optional processing path를 구분한다. 예컨대 exact dedup 구현이 있다고 실제 baseline이 그 path를 썼다고 추론하지 않는다. config, experiment record와 output stats에서 actual path를 확인한다.

**네 공개 pipeline을 같은 질문표로 비교한다.**

| 질문 | C4 | FineWeb/DataTrove | Dolma | DCLM |
|---|---|---|---|---|
| raw identity | TFDS download/source 계약 | dump·document metadata | document/source와 attributes | input shard·experiment material |
| 변환 표현 | builder/generator·heuristic | ordered blocks·stats | text+annotation sidecar | processor output+stats |
| dedup 핵심 | 공개 recipe의 exact/heuristic 범위 | dump/global MinHash 선택 | Bloom·dedup processors | baseline/optional path 구분 |
| 품질 검증 | heuristic·language audit | filter ablation·downstream | mixture/annotation audit | controlled model evaluation |
| 남는 gap | URL→삭제/학습 소비 | durable ID namespace 연결 | release별 rights·FP·소비 | row→token/update crosswalk |

이 표는 우승자를 고르지 않는다. 자기 pipeline의 요구에 맞는 component와 빠진 control-plane 계약을 찾는다. 어떤 stack을 사용해도 immutable source, coordinate, decision, release·delete와 consumption ledger는 별도 integration이 필요하다.

공개 dataset card의 token/document 수는 해당 release·계산법의 값이다. tokenizer·normalization과 denominator가 다르면 재계산 결과가 다를 수 있다. 숫자를 현재 pipeline 측정처럼 복사하지 않는다.

**contamination을 exact match에서 causal impact까지 확장한다.**

benchmark contamination scanner는 benchmark revision, fields, normalization과 query set을 고정한다. exact document/hash, substring·n-gram, paraphrase/semantic와 generated variant를 evidence level로 구분한다. 한 hit type의 precision·recall을 다른 type에 일반화하지 않는다.

exact/n-gram match는 span과 token overlap을 설명하기 쉽지만 paraphrase를 놓친다. semantic detector는 recall 후보를 넓히지만 domain-similar false positive가 많을 수 있다. candidate generation과 human/secondary verifier를 분리한다.

contamination removal은 evaluation item 자체뿐 아니라 solution, explanation, translated·reformatted descendant와 benchmark discussion을 고려한다. family lineage와 temporal cutoff를 사용한다. hidden test는 raw 접근이 제한되므로 공개 범위와 auditor protocol을 명시한다.

false positive로 유용한 domain text를 대량 제거하면 model capability와 evaluation 모두 달라진다. removed token mass, domain/language와 manual samples를 보고한다. “오염 0” 같은 절대 주장을 피한다.

model-level impact는 contaminated/clean strata, matched item exclusion과 temporal holdout을 비교할 수 있다. 그러나 score 차이를 한 training row의 causal effect로 단정하지 않는다. 완전한 row→update→score ledger가 공개 기본 pipeline에서 닫히지 않는 gap을 명시한다.

**삭제 요청을 identity·policy·model 상태로 나눈다.**

삭제 요청은 requester authentication과 scope resolution에서 시작한다. URL 문자열 하나가 canonical source identity와 같지 않을 수 있다. redirect, mirrors, content copies와 derived documents를 stable source/document IDs와 checksum으로 찾는다.

승인된 tombstone은 raw retention, processing eligibility와 training use authorization을 분리한다. 법적 조사 copy를 제한 보존하더라도 다음 release·loader에서 사용을 막을 수 있다. deletion과 revocation, redaction을 다른 action으로 기록한다.

reverse lineage는 document revisions, exact/fuzzy cluster, corpus releases, tokenized·packed shards, active runs와 checkpoints를 찾는다. dedup survivor가 삭제됐을 때 loser를 자동 승격할지 cluster 전체를 제외할지 policy를 시험한다. 조용한 승격은 삭제 의도를 위반할 수 있다.

새 corpus release에서 제외됐다고 이미 학습된 model에서 영향이 제거된 것은 아니다. affected checkpoint·adapter/export/deployment를 `affected`, `action-pending`, `retrained`, `restricted` 등으로 표시한다. machine unlearning/editing을 적용하면 새 artifact와 locality·retention evaluation이 필요하다.

loader는 startup/resume과 장기 worker에서 delete policy generation을 확인한다. stale manifest·cache와 already-prefetched records를 처리한다. tombstone 뒤 새 UpdateID가 해당 DocumentID descendant를 소비하지 않는지 fault test한다.

**분산 corpus 공정의 결정성을 증명한다.**

task partition과 completion order가 달라도 logical output set과 deterministic survivor가 같아야 하는 stage를 정의한다. output shard byte ordering까지 같은 강한 재현과 record-set 동일성을 구분한다. 정렬·compression metadata 때문에 byte가 달라질 수 있다.

각 task에는 input range/shard, code/config, attempt ID와 output digest·stats를 기록한다. retry와 speculative duplicate가 commit에서 한 번만 선택된다. partial output은 staging에 있고 release manifest에 들어가지 않는다.

global dedup은 worker-local candidate edges를 합쳐 connected components와 survivor를 결정한다. union/find merge order가 달라도 canonical component/survivor가 같게 tie-break를 정의한다. timestamp, quality, license와 source priority를 ordered key로 만든다.

distributed counters는 accepted/rejected/reason의 합이 input terminal count와 맞아야 한다. dropped task, double count와 late retry를 reconciliation으로 잡는다. document count 외에 bytes·characters·token estimate와 strata를 비교한다.

release rebuild는 clean cache와 worker count·task order를 바꿔 실행한다. claimed invariant에 맞는 record set, decisions와 stats를 비교한다. 실행하지 않은 scale result는 만들지 않고 small synthetic와 source-confirmed/hardware-pending을 구분한다.

**data attribution의 주장 수준을 구분한다.**

provenance attribution은 어느 source/document가 corpus와 run에 들어갔는지를 말한다. gradient influence는 특정 training example이 parameter/update에 미친 국소 효과를 추정한다. behavioral attribution은 model output이 어떤 source 때문에 나왔는지 묻는다. 세 수준을 하나의 “출처”로 합치지 않는다.

consumption ledger는 UpdateID가 어떤 packed samples와 source descendants를 읽었는지 증명할 수 있다. 그러나 shared parameter와 nonlinear training 때문에 한 document가 특정 문장을 생성하게 했다는 인과를 자동 증명하지 않는다. influence function, gradient similarity나 retrieval probe에는 가정·근사와 scale cost가 따른다.

data attribution 실험은 target output, candidate source set, checkpoint, tokenizer와 scoring method를 고정한다. exact memorization, paraphrase, domain influence와 generic language pattern을 구분한다. unrelated·same-domain controls와 source removal/retraining child가 강한 증거를 줄 수 있지만 비용이 크다.

attribution 결과를 삭제 완료 증거로 쓰지 않는다. probe가 기억을 찾지 못해도 영향 부재를 증명하지 못한다. 반대로 문장 overlap이 있다고 해당 source 하나가 유일 원인이라는 뜻도 아니다. policy action과 scientific inference를 분리한다.

dataset/model card에는 known source·mixture와 provenance 범위, 미확인 영역을 설명한다. source citation을 제공할 수 있는 product는 retrieval/source system과 model parametric memory를 구분한다. 근거 없는 자동 citation을 training lineage로 포장하지 않는다.

**span lineage를 byte에서 token까지 보존한다.**

raw WARC payload에는 byte offset과 checksum을, decoded text에는 encoding과 decoder revision을 기록한다. HTML extractor는 DOM/node와 text span map을 만든다. normalization, redaction과 segmentation은 parent interval→child interval relation을 기록한다.

Unicode normalization은 여러 code point를 합치거나 분해하고 HTML entity decoding도 byte/character 길이를 바꾼다. inclusive/exclusive end와 byte/codepoint/grapheme 단위를 schema로 명시한다. offset을 단순 정수 두 개로 저장해 단위를 잃지 않는다.

span이 삭제·치환되면 mapping이 일대일이 아닐 수 있다. redaction placeholder, boilerplate removal과 whitespace collapse를 event로 남긴다. original payload 접근이 제한·삭제돼도 transformation decision과 digest를 policy 범위에서 보존한다.

tokenizer offset mapping은 normalized text와 token ID를 연결한다. byte fallback과 special token은 원문 span이 없거나 여러 byte를 가질 수 있다. packed sample은 document/token range, separator와 loss mask를 mapping한다. 5·6장의 fixture와 같은 IDs를 쓴다.

임의 loss token에서 packed index→tokenized document→normalized span→raw locator로 역추적한다. 반대 방향으로 delete span이 영향을 주는 tokens, packed samples와 consumers를 찾는다. 양방향 test가 attribution·redaction·contamination의 기반이다.

**PII와 유해 콘텐츠 처리를 품질 filter와 분리한다.**

PII taxonomy에는 email·전화·주소, credentials, government ID와 context-dependent personal information이 있다. detector마다 precision·recall, language·format 범위가 다르다. 한 regex나 classifier로 “개인정보 제거 완료”라고 쓰지 않는다.

action은 redact span, reject document, restrict access, quarantine와 review로 나눈다. redaction placeholder가 tokenizer와 model에 새로운 반복 패턴을 만들 수 있다. 주변 context와 document utility, reconstruction risk를 평가한다.

detector output은 sensitive raw value를 log·metric에 복제하지 않는다. match type, span digest와 reason을 access-controlled evidence에 둔다. raw access, reviewer와 retention을 제한한다. scanner 자체의 secret leak을 시험한다.

유해 content는 안전 학습·red-team에서 필요할 수도 있다. 무조건 quality reject하거나 일반 corpus에 그대로 넣지 않는다. purpose, access, label와 mixture policy를 분리한다. red-team secret set이 training/evaluation에 누출되지 않게 lineage gate를 둔다.

false negative incident가 발견되면 source family와 transform descendants를 impact query하고 detector fixture를 추가한다. false positive는 language/domain selection bias와 removed token mass를 조사한다. policy threshold 변경은 새 corpus generation이다.

**corpus poisoning과 supply-chain 공격을 주입한다.**

공격자는 많은 near-duplicate 문서를 여러 domains에 배포하거나 quality classifier 특징을 모방하고, hidden trigger·target association을 삽입할 수 있다. 단순 URL count와 exact dedup만으로 잡히지 않을 수 있다.

source/domain/account burst, temporal synchronization, near-duplicate cluster, rare n-gram·language shift와 classifier score anomaly를 bounded statistics로 본다. high score가 신뢰할 수 있는 source라는 뜻은 아니다. source provenance와 content signals를 결합한다.

small synthetic poison family를 test namespace에 주입해 exact/fuzzy dedup, quality, contamination과 lineage detector가 어떻게 반응하는지 본다. production public corpus를 실제 오염시키지 않는다. trigger 원문과 exploit는 접근 통제한다.

poison이 release에 들어간 fault에서는 tombstone·rebuild와 affected packed/run query를 실행한다. 이미 trained model에는 targeted behavior, clean utility와 safety를 평가하고 restrict/retrain/edit decision을 기록한다. detector hit만으로 model 영향 제거를 주장하지 않는다.

pipeline dependency·container나 model-based filter 자체가 변조될 수도 있다. source/env digest, signature·SBOM와 runtime inventory를 27장 공급망 계약에 연결한다. 같은 input인데 decision이 달라지면 first different transform을 찾는다.

**공개 corpus 숫자를 재현할 때의 규칙.**

dataset card의 document·token·byte 수는 release, tokenizer, filtering과 counting convention에 의존한다. 현재 pipeline 결과처럼 복사하지 않는다. source의 exact revision·표와 denominator를 인용하고 필요한 경우 우리 manifest에서 다시 계산한다.

token 수를 비교하려면 tokenizer, special token, document separator, truncation과 count 위치를 고정한다. UTF-8 byte·character와 word count를 token으로 부르지 않는다. compressed storage size와 uncompressed content bytes도 구분한다.

filter retention은 input records, documents, bytes·characters와 tokens마다 다르다. 긴 documents를 주로 제거하면 document retention과 token retention이 다르다. strata와 reason별 numerator/denominator를 둔다.

dedup percentage는 candidate pairs, duplicate documents, clusters, removed survivors/losers 가운데 무엇인지 명시한다. transitive components 때문에 pair rate와 document removal이 같지 않다. Bloom false positive와 MinHash 확률을 uncertainty에 포함한다.

실제 full-scale job을 실행하지 않았다면 공개 source 결과와 synthetic fixture만 구분해 제시한다. hardware/runtime 처리량, 비용과 wall time을 만들어내지 않는다. 실행할 audit command와 expected manifest를 `scale-pending`으로 남긴다.

## 4.12 rights·lineage·realized mass로 release를 감사한다

release 감사는 shard 개수 확인으로 끝나지 않는다. content root, policy root와 deletion root를 고정하고, 권리 판단과 transform lineage가 실제 소비 질량까지 이어지는지 양방향으로 조회한다. 공개 corpus의 이름이나 문서 수는 이 세 폐루프를 대신하지 못한다.

### 4.12.1 corpus version diff를 원인별로 설명한다

R1과 R2의 diff는 added, removed, modified와 unchanged를 DocumentID·content revision으로 나눈다. removed에는 source disappearance, filter, dedup survivor, contamination, rights/delete와 parse failure reason이 있다. modified는 raw 변경과 parser/normalization 변경을 구분한다.

stage별 first difference를 계산한다. input digest가 같고 extracted text부터 다르면 parser, text는 같고 decision만 다르면 detector/config, decision은 같고 output만 다르면 writer/sharding 문제다. 모든 downstream file을 비교하기 전에 earliest edge를 찾는다.

document diff를 characters·tokens와 language/domain/quality strata로 집계한다. count가 작은 변경도 긴 documents나 희소 domain에서 token mass가 클 수 있다. planned mixture와 actual training consumption impact를 연결한다.

deterministic rebuild에서 logical set은 같고 shard ordering/compression만 다를 수 있다. claimed reproducibility grade에 맞게 record-set, ordered IDs 또는 bytes를 비교한다. meaningless timestamp/build path 차이는 제거하거나 metadata 차이로 설명한다.

R2 승인 전 R1 active loader와 checkpoints의 policy를 정한다. alias만 바꿔 오래 열린 workers가 자동 전환된다고 가정하지 않는다. new jobs는 R2, existing jobs는 pinned R1 또는 coordinated restart 중 하나를 명시한다. deletion urgency가 있으면 stale use를 차단한다.

**corpus loader를 fail-closed 소비자로 만든다.**

loader는 ReleaseID를 signed/approved manifest로 resolve하고 exact shard list, size·digest와 schema를 검사한다. directory glob으로 임시·partial·old shard를 섞지 않는다. cache hit에도 digest와 deletion/policy generation을 확인한다.

distributed workers는 같은 release·tokenizer/mixer digest를 startup all-gather한다. rank마다 다른 mirror snapshot을 읽지 않는다. worker-local retry가 row를 duplicate·skip하면 consumption ledger가 탐지한다. sample owner와 next cursor를 checkpoint한다.

manifest alias가 run 중 움직여도 pinned job은 same immutable release를 유지한다. urgent revoke는 coordinator가 모든 workers를 같은 boundary에서 중단·재시작한다. 일부 rank만 새 corpus로 전환하지 않는다.

partial download, checksum mismatch, stale delete revision, missing shard와 unauthorized source를 fault로 넣는다. fallback to previous release를 조용히 하지 않는다. fallback이 허용되는 availability policy면 exact predecessor, reason·duration과 data impact를 기록한다.

loader metric은 attempted/accepted documents·tokens, retries, skips, release/policy mismatch와 data wait를 드러낸다. 원문·URL을 label로 쓰지 않는다. UpdateID에서 consumed packed sample과 source lineage로 내려가는 artifact를 둔다.

**scale audit와 full validation의 범위를 정한다.**

모든 문서에 exact checksum·schema·terminal state 같은 cheap invariant를 적용한다. expensive manual quality, semantic contamination·PII와 decode inspection은 stratified·risk-based sample을 사용할 수 있다. sample frame과 weight를 기록한다.

sampling은 source, language, domain, length, score boundary, accepted/rejected와 dedup cluster를 포함한다. random accepted sample만 보면 tail과 rejection bias를 놓친다. known incident·new source는 oversample하되 population estimate에서 sampling weight를 보정한다.

confidence interval과 detector precision/recall을 report한다. zero observed failure가 zero population failure를 뜻하지 않는다. critical policy는 automated full scan과 manual sample을 결합한다.

full-scale execution이 필요한 record count, throughput, cost와 storage는 실제 run artifact 없이는 주장하지 않는다. synthetic correctness와 source behavior, scale-pending을 분리한다. 독자는 자기 infrastructure에서 executor·fault와 reconciliation을 실행한다.

### 4.12.2 rights·robots·consent를 기술 score와 분리한다

robots directive, terms, license, consent와 jurisdiction은 quality classifier score가 아니다. parser가 깨끗한 text를 만들고 model utility가 높아도 사용 authorization이 없을 수 있다. source acquisition policy와 legal review state를 별도 ledger에 둔다.

robots와 terms는 crawl·사용 시점과 agent·resource에 따라 해석 맥락이 있다. 현재 page만 보존하면 acquisition 당시 evidence를 잃는다. snapshot, retrieval time, policy revision과 reviewer를 기록한다. 기술 구현이 법적 결론을 자동 생성한다고 표현하지 않는다.

opt-out/delete 요청은 requester·scope 검증과 source identity resolution을 거친다. 요청 URL과 content copy·derived document를 graph에서 찾는다. 처리 SLA와 action status를 제공하되 민감 requester 정보를 corpus metric에 노출하지 않는다.

publicly accessible은 자유로운 학습·재배포 license와 같지 않다. dataset card의 license 문자열도 모든 underlying source rights를 자동 변경하지 않는다. unknown·mixed state를 permissive로 채우지 않는다.

policy가 바뀌면 corpus byte integrity는 같아도 authorization이 달라진다. active release·loader와 descendants를 query해 block/rebuild/restrict한다. 과거 decision과 현재 policy를 모두 보존한다.

**공개 pipeline 재현 보고서의 최소 형식.**

보고서는 upstream repository·commit, dataset card/release, raw source scope와 local adapter revision으로 시작한다. 사용한 config, executor, detector model·threshold와 output manifest digest를 둔다.

각 단계에는 input/output/reject count, byte/token 추정치, 사유와 소요 시간이 기록된다. 공개 출처의 수치와 로컬에서 실행해 얻은 수치를 구분한다. 실행하지 않은 full-scale 결과는 `scale-pending`이다.

quality report는 accepted/rejected/boundary samples의 strata와 sampling frame, contamination·PII/dedup의 precision/recall 한계를 적는다. rights/delete는 별도 policy status다. 평균 score 하나로 합치지 않는다.

reproducibility에는 clean cache rebuild, worker/task order 변화, retry·partial fault와 logical output comparison이 있다. deletion drill은 source→release→packed/consumer impact와 stale loader rejection을 포함한다.

독립 reviewer는 임의 accepted, rejected, duplicate loser와 tombstone을 raw locator에서 terminal state까지 왕복한다. report에는 불일치, unknown과 next action을 숨기지 않는다.

### 4.12.3 planned corpus와 realized training mass를 비교한다

release manifest의 document/token 비율은 training에서 실제 소비된 비율과 다를 수 있다. sampler weight, replacement, exhaustion, sequence packing, truncation, invalid row skip와 resume replay가 realized mass를 바꾼다.

consumption ledger에는 UpdateID, source/stratum, documents, raw·valid·packed tokens, dropped tail, repeated/replayed tokens, rank와 sampler revision을 둔다. per-row raw identity는 접근 통제 artifact에 두고 metric cardinality를 제한한다.

planned weight \(q_i\)와 realized valid-token fraction \(\hat q_i\) 차이를 step window와 full run에서 본다. 길이가 긴 source는 document sampling 비율보다 token mass가 커질 수 있다. domain-balanced objective weight가 추가되면 sampling과 loss weighting을 함께 계산한다.

deletion·corruption으로 source가 빠졌을 때 renormalization이 다른 source mass를 어떻게 늘리는지 기록한다. silent skip이 curriculum을 바꾸지 않게 coordinator policy를 둔다. affected UpdateID와 checkpoints를 query한다.

이 ledger는 6장의 mixture·packing, 13장의 scheduler/token clock, 16장의 cluster data placement와 28·29장의 resume에 연결된다. corpus가 좋다는 주장은 manifest만 아니라 실제 loss-bearing token distribution까지 내려가야 한다.

**source 좌표의 유효 범위를 다시 확인한다.**

DataTrove FineWeb example, Dolma processor, DCLM processor/tokenize/experiment와 TFDS C4 builder의 commit·path·symbol·line을 source ledger에 둔다. 각 좌표가 reader, transform, stats, executor 또는 release 중 무엇을 증명하는지 한 문장으로 제한한다.

example config가 published release의 실제 execution이라는 주장은 dataset card·manifest와 별도 evidence가 필요하다. optional repository code를 baseline path로 승격하지 않는다. toolkit capability와 corpus property를 분리한다.

upstream tests는 fixture, backend와 assertion을 기록한다. local integration은 durable DocumentID, span crosswalk, deletion과 training consumption처럼 공개 stack 사이에 빠진 계약을 보강한다. 한 source의 test가 cross-stack lineage를 보증하지 않는다.

upgrade 시 semantic anchor와 output schema, retry/idempotency·decision을 diff한다. old/new small corpus에서 first different stage를 확인한다. expected release를 먼저 재생성하지 않는다.

독립 검토자는 네 공개 pipeline에서 각각 한 좌표를 골라 local checkout의 내용 hash와 ledger를 대조한다. 이어 같은 input document가 각 stack의 어느 internal namespace를 받는지 확인하고 durable crosswalk가 없는 경계를 표시한다.

source가 이동하거나 option default가 달라졌으면 관련 fixture와 release diff를 stale로 전환한다. 문서 설명만 수정해 과거 실행을 새 code의 증거로 만들지 않는다. 재실행 전까지는 source-confirmed와 scale-pending을 명확히 구분한다.

마지막 report에는 source ledger digest, corpus ReleaseID, delete policy generation, consumption ledger 범위, reviewer와 실제 검사 시각을 기록한다. 이 조합이 5장의 tokenizer 입력을 고정하는 유일한 인계 키다.

모든 미실행 scale audit에는 정확한 재현 command, task·worker 수, 명시적 resource limit, 예상 count reconciliation과 output manifest를 반드시 명시한다. 실제 artifact가 생기기 전에는 공개 pipeline의 처리량·비용을 현재 환경의 결과처럼 절대로 쓰지 않으며, 실패도 별도의 독립 상태로 상세하게 보존한다.

예외는 없다.

**release 전수 감사의 불변식.**

**identity 불변식.** 모든 raw record에는 immutable locator와 checksum이 있다. 모든 derived document에는 정확한 transform revision과 parent가 연결된다. 동일 ID가 다른 content를 가리키지 않는다. mutable path는 resolver일 뿐 artifact identity가 아니다.

**terminal-state 불변식.** 각 input attempt와 logical record를 구분하고, logical record는 accepted, rejected, quarantined, error 가운데 정의된 terminal state 하나를 갖는다. retry와 speculative execution이 counts와 outputs를 중복시키지 않는다. unknown drop은 0이어야 한다.

**좌표 불변식.** raw byte, decoded character, normalized span, segment, token offset의 단위가 명시된다. 삭제·redaction·extraction으로 사라진 interval도 event로 설명된다. inclusive와 exclusive end를 혼용하지 않는다. 임의 accepted/rejected 문서에서 왕복한다.

**dedup 불변식.** signature config와 candidate generation, verified edge, component, survivor가 별도 artifacts다. winner는 partition과 completion order에 독립적인 policy로 결정된다. 모든 loser가 component와 reason을 가리킨다. incremental index의 base release가 고정된다.

**필터 불변식.** raw score, threshold, decision, reason, detector artifact를 구분한다. stage conditional missing score를 pass로 해석하지 않는다. threshold 변경의 impact set과 token mass를 계산할 수 있다. classifier의 unknown model alias를 허용하지 않는다.

**오염 불변식.** benchmark version과 prompt variants, matching view, detector parameters를 고정한다. match evidence는 source span과 연결된다. removal action이 release allowlist와 loader까지 전파된다. semantic 미검출 가능성과 false-positive 손실을 보고한다.

**출판 불변식.** top manifest가 가리키는 모든 shards는 committed, checksum-valid, schema-valid다. manifest 밖 shard는 loader가 읽지 않는다. global counts는 task manifests와 terminal states에서 재구성된다. release card의 수치는 같은 manifest query에서 나온다.

**삭제 불변식.** 승인된 tombstone이 다음 committed release와 active loader policy에 반영된다. reverse impact query가 dedup, tokenized, packed descendants를 찾는다. payload retention과 provenance retention을 구분한다. checkpoint에 남은 영향은 별도 status로 표시한다.

**권한 불변식.** 제한 payload는 일반 logs, metrics, debug dumps에 나타나지 않는다. resolver access가 감사된다. sample artifacts는 최소 필요 span만 노출한다. 접근 불가와 데이터 누락을 서로 다른 상태로 표현한다.

**재현 불변식.** source/code/config/model commits와 seeds가 있다. 외부 API와 mutable artifacts의 한계가 표시된다. task 수와 순서를 바꾼 rerun에서 deterministic parts가 같은 logical outputs를 만든다. 확률적 audits는 sample design과 uncertainty를 보존한다.

**학습 연결 불변식.** corpus release, tokenizer, mixer, packer, loader consumption, UpdateID 사이에 끊기지 않은 edge가 있다. 실제 consumed valid targets를 source mixture로 집계할 수 있다. planned weights와 divergence를 설명한다.

감사는 파일 수나 관계 레코드 수를 세는 것으로 끝나지 않는다. 각 불변식에 반례 fixture와 query를 둔다. 관계망이 연결되어도 잘못된 edge가 있을 수 있으므로 checksum, span, count를 함께 검증한다. PASS는 현재 revision과 sampled/full scope를 적는다.

**웹 코퍼스 제조를 한 문장으로 다시 정의한다.**

웹 코퍼스 제조는 인터넷에서 문장을 많이 모으는 일이 아니다. 변화하는 source를 immutable record로 고정하고, 손실 있는 추출과 정규화를 좌표화하며, 품질·권리·개인정보·중복·오염 결정을 증거로 남기고, 분산 재시도 속에서도 하나의 committed release를 만드는 작업이다.

좋은 corpus는 accepted text만 깨끗해 보이는 corpus가 아니다. 왜 남았고 왜 버려졌는지, 어떤 population의 확률 질량이 줄었는지, 어떤 source와 detector가 결정했는지, 삭제 요청이 어디까지 전파되는지를 설명할 수 있다. 이 설명은 dataset card와 source code, manifests, fixtures에서 서로 일치해야 한다.

공개 stack은 이 공정의 중요한 부품과 recipe를 제공한다. DataTrove의 block·executor와 dedup stages, FineWeb examples, Dolma의 annotations·mixture, DCLM의 controlled data experiments를 고정 revision에서 읽는다. 그러나 우리 release의 rights policy, deletion graph, integration adapters, 실행 환경은 별도 증거가 필요하다.

독자는 이 장을 바탕으로 pipeline 이름을 묻는 데서 멈추지 않는다. 입력과 출력 identity, 선택 편향, 확률적 detector의 오류, retry 원자성, source-to-update lineage를 묻는다. 그 질문에 답할 수 있을 때 corpus 규모와 model loss 사이의 인과를 실제로 조사할 수 있다.

마지막 승인은 정상 문서뿐 아니라 duplicate loser, contamination reject, PII redaction, tombstone을 왕복한 뒤 내려진다. 성공 경로만 있는 시스템은 삭제와 장애가 왔을 때 무너진다. 모든 terminal state가 설명되고 다음 tokenizer가 committed input만 읽을 때 4장의 계약이 닫힌다.

**인계 직전의 독립 재현.**

작성자와 다른 검토자가 release manifest에서 shard 하나와 문서 다섯 개를 무작위·층화 방식으로 고른다. raw locator와 checksum을 확인하고 고정 parser·normalizer·filters로 다시 처리한다. accepted text와 decision records, dedup signature가 committed artifact와 맞는지 본다. 외부 source가 사라져 재생할 수 없으면 보존한 raw record로 검증하고 그 retention 경계를 명시한다.

검토자는 threshold 주변 accepted/rejected pair도 고른다. raw score와 decision이 config대로인지, language·domain에 따라 예외 branch가 적용됐는지 확인한다. classifier가 floating model alias를 사용하지 않았는지 artifact checksum을 본다. 사람 판단이 개입했다면 decision authority와 reason이 있어야 한다.

dedup component에서는 winner와 loser 하나를 원문까지 되짚는다. partition order를 바꾼 작은 rerun에서 winner가 유지되는지 확인한다. Bloom과 MinHash는 parameters와 seeds를 복원하고 toy oracle 및 production sample audit의 범위를 구분한다. 확률적 detector의 한계를 PASS 문구에서 지우지 않는다.

release loader는 manifest 밖 임시 shard와 checksum mismatch shard를 거부해야 한다. delete revision을 하나 늦춘 loader를 의도적으로 실행해 startup gate가 실패하는지 본다. stale cache와 partial upload도 고장 주입한다. detector가 울리지 않으면 출판 control plane은 아직 완성되지 않았다.

마지막에는 selected packed sample에서 source documents로 역추적하고, source tombstone에서 affected packed samples로 정추적한다. 두 query의 결과가 서로 대응하고 UpdateID consumption record와 연결되어야 한다. payload 권한 때문에 직접 볼 수 없는 경우 authorized checksum verifier의 attestation을 사용하되 누락으로 처리하지 않는다.

이 독립 재현 보고서에는 검사 범위, source revisions, observed checksums, 불일치와 미실행 항목이 있다. 통과한 뒤에도 corpus가 영구히 완전하다고 선언하지 않는다. 새 crawl, detector, benchmark, 삭제 요청마다 child release와 diff audit를 반복한다. 반복 가능한 검증 절차 자체가 대규모 데이터 공정의 가장 오래 남는 자산이다.

검토가 끝나면 corpus release와 보고서를 함께 봉인한다. 본문 text, manifest, source registry, decision ledger 중 하나라도 서로 다른 revision을 가리키면 승인을 보류한다. tokenizer 팀은 이 봉인된 조합만 입력으로 사용한다. 임시 directory나 사람이 복사한 일부 JSONL은 실험 convenience일 수 있어도 정식 계보의 시작점이 될 수 없다.

이 규칙은 속도를 늦추기 위한 절차가 아니다. 수십억 문서에서 사후에 한 source와 한 삭제 요청을 찾는 비용을 줄이고, parser나 threshold 변경의 실제 영향 반경을 계산하게 한다. 초기 identity와 좌표를 잃으면 나중에 더 비싼 재크롤링과 재학습으로도 완전히 복구하기 어렵다.

결국 좋은 데이터 공정의 속도는 초당 처리량과 함께 잘못된 release를 얼마나 빨리 반증하고 안전하게 다시 만들 수 있는가로 측정된다. 4장은 그 재생·반증·삭제 능력을 다음 단계의 기본 입력 계약으로 넘긴다.

인계자는 마지막으로 release ID를 입력한 단일 명령이나 query가 정확한 manifest와 품질 보고서, 삭제 revision, tokenizer 입력 shard를 반환하는지 확인한다. 사람이 여러 경로에서 최신 파일을 추정해야 한다면 control plane이 아니다. resolver 결과도 checksum과 schema를 검증하고, 실패하면 오래된 release로 조용히 fallback하지 않는다. 이 fail-closed 동작까지 증명한 뒤 release를 승인한다.

승인 기록은 다음 release의 비교 기준선이자 삭제 감사의 출발점으로 영구 식별된다.

**수집에서 release까지 상태 기계를 종단 검증한다.**

**대규모 수집을 요청·응답·원본 보존의 세 상태로 나눈다.**

웹 corpus 수집기는 URL 목록을 text로 바꾸는 함수가 아니다. crawl request, network response, raw object와 fetch decision을 각각 식별한다. request에는 canonical URL 후보, timestamp, user agent/policy profile, retry ordinal과 scheduler partition이 있다. response에는 status, redirect chain, headers, content encoding, payload byte length와 transport checksum이 있다. raw object에는 immutable locator, content checksum, capture metadata와 retention class가 있다. parser가 만든 text는 이 raw object의 child다.

옵션은 concurrency, timeout, retry/backoff, redirect limit, maximum byte, accepted MIME과 compression 정책이다. 상태는 outstanding queue, host politeness clock, retry set와 partial download다. 효과는 coverage, source 편향, 중복 fetch, 비용과 재현성이다. timeout을 짧게 하면 느린 지역·소형 사이트가 과소 대표될 수 있다. maximum byte를 낮추면 긴 문서 domain이 사라진다. 이를 단순 성능 tuning이라고 기록하지 않는다.

fetch result state는 `Fetched`, `NotModified`, `RejectedByPolicy`, `HTTPFailure`, `TransportFailure`, `TooLarge`, `UnsupportedType`, `RetryPending`처럼 terminal과 transient를 구분한다. 실패 응답을 빈 문서로 parser에 보내지 않는다. retry가 성공하면 이전 attempt와 같은 RequestID 계보 아래 새 ordinal을 갖는다. 중복 success가 생겨도 raw object checksum과 canonical selection이 하나의 accepted capture를 정한다.

**redirect와 압축 폭탄 반례**

redirect loop, scheme 변경, 다른 registrable domain 이동과 상대 URL 오류를 test server로 만든다. final URL만 저장하면 원래 seed와 policy decision을 잃는다. chain 전체를 제한 길이로 저장하고 각 hop의 status와 timestamp를 기록한다. robots·rights 판단을 어느 URL 기준으로 했는지도 명시한다.

압축 payload는 wire byte가 작아도 해제 뒤 매우 클 수 있다. compressed와 decoded byte limit, expansion ratio와 streaming abort를 따로 둔다. test는 정상 gzip, truncated stream, checksum mismatch, nested archive와 높은 expansion payload를 포함한다. 실패 뒤 partial raw object를 committed prefix에 넣지 않는다. quarantine과 diagnostic metadata만 남긴다.

content encoding과 declared charset가 틀린 페이지를 넣는다. decoder는 선택한 encoding과 replacement count를 decision record에 남긴다. replacement가 많으면 reject 또는 raw-only 상태로 보낸다. silent replacement가 언어·품질 filter로 넘어가면 source별 편향이 생긴다.

**parser와 normalizer를 byte span 보존 함수로 검증한다.**

HTML parser는 raw bytes를 DOM 또는 event stream으로 바꾸고 boilerplate extractor는 visible text span을 선택한다. normalizer는 Unicode, whitespace, control, line boundary와 URL/email 정책을 적용한다. 각 단계는 text만 반환하지 않고 parent span map과 decision annotations를 반환해야 한다. 삭제·오염 조사에서 normalized substring을 raw response까지 되짚을 수 있어야 한다.

source function 좌표는 사용하는 parser wrapper, extractor block와 normalizer chain에 고정한다. library 이름만 적으면 default option 변경을 놓친다. parser recovery mode, maximum depth, script/style/comment 처리, link text와 alt text, table/list boundary를 config digest에 넣는다. generated text의 paragraph boundary가 dedup shingle과 tokenizer sequence에 영향을 준다.

normalization은 idempotent해야 한다. `N(N(x)) = N(x)`를 property test한다. 그러나 raw checksum과 normalized checksum을 혼동하지 않는다. Unicode NFC/NFKC 선택은 호환 문자와 일부 의미를 합칠 수 있다. case folding, digit 또는 punctuation 제거는 dedup recall을 높일 수 있지만 code·수식·고유명사를 훼손한다. canonicalization용 view와 training text view를 분리할 수 있다.

**parser differential test**

작은 HTML corpus에 malformed tag, nested table, hidden CSS, script에 포함된 closing tag, entity, bidi control, iframe과 template을 넣는다. current parser와 candidate parser의 selected spans를 DocumentID와 raw byte range로 diff한다. 단순 text checksum mismatch가 아니라 added, removed, moved와 normalized-only 변화로 분류한다.

parser upgrade가 전체 corpus에 미치는 영향을 층화 표본으로 추정한다. domain, language, MIME, size와 parse error class별 sample을 뽑는다. threshold를 넘는 class는 full replay 또는 release 분리를 요구한다. 새 parser가 더 많은 text를 만든다는 사실을 자동 품질 향상으로 보지 않는다. navigation·cookie와 hidden text 증가일 수 있다.

failure fixture는 process 종료, worker retry와 shard 재실행이다. output writer는 DocumentID별 deterministic child와 attempt record를 만들고 final manifest는 성공 generation만 가리킨다. 동일 input이 두 worker에서 처리되어도 winner rule과 checksum이 같아야 한다. nondeterministic parser dependency가 있으면 환경과 seed를 고정하거나 variation을 release metric으로 노출한다.

**품질 정제를 score, policy와 decision으로 분리한다.**

language detector, quality classifier, heuristic와 safety detector는 score 또는 annotation을 만든다. accept/reject는 release policy가 이 annotation에 적용한 decision이다. score 생성과 decision을 한 함수에 숨기면 threshold 변경 때 expensive model inference를 다시 하거나 과거 결과를 설명하기 어렵다. annotation에는 detector ID, artifact checksum, input view, raw score, calibration version과 timestamp가 있다.

heuristic은 character count, line statistics, symbol ratio, repetition, stopword와 structural feature를 사용할 수 있다. 각 feature의 denominator와 missing policy를 기록한다. empty text에서 division by zero, 매우 짧은 문서, CJK처럼 whitespace token이 적은 언어, code와 수식 domain을 test한다. global threshold가 특정 language를 제거하는지 acceptance rate와 score distribution을 층화한다.

classifier threshold option은 accepted set, class balance와 비용을 바꾼다. state는 raw annotation이 그대로이고 policy revision과 decision edge가 새로 생긴다. effect를 재현하려면 threshold 주변 문서, false positive/negative review와 downstream token mass를 본다. document count만 보면 긴 문서 한 class의 영향이 숨는다.

**fail-open과 fail-closed를 detector별로 정한다**

quality classifier timeout을 reject로 처리하면 outage가 특정 partition을 대량 삭제할 수 있다. accept로 처리하면 unscored content가 release에 섞인다. `NeedsReview` 또는 `Unscored` terminal을 두고 정식 release policy가 허용하는지 명시한다. PII·rights detector는 위험 성격 때문에 더 보수적인 정책을 가질 수 있다. 모든 detector에 같은 fallback을 적용하지 않는다.

failure experiment는 classifier artifact missing, wrong checksum, OOM, batch 하나의 NaN score와 schema field 누락을 각각 넣는다. worker는 이미 처리한 document까지 다른 model alias로 재시도하지 않는다. retry는 같은 DetectorID를 요구하거나 새 attempt와 generation을 만든다. mixed detector generation이 한 annotation shard에 들어가면 manifest가 이를 드러내야 한다.

human review는 detector truth를 덮는 boolean이 아니다. reviewed span, reviewer authority, rubric version, reason과 conflict resolution을 저장한다. privacy 때문에 reviewer identity를 제한하더라도 accountable role과 audit ID는 필요하다. review sample이 random인지 threshold·risk strata인지 기록해야 error estimate를 해석할 수 있다.

**dedup을 후보 생성, 확인과 winner 선택으로 해부한다.**

exact dedup은 normalized content checksum으로 같은 문서를 묶을 수 있다. near dedup은 shingles, MinHash 또는 embedding 등으로 후보를 만들고 similarity rule로 edge를 확인한다. 후보 생성은 false negative를, 확인 threshold는 precision/recall을, connected component와 winner selection은 어떤 source가 남는지를 바꾼다. “dedup 30%” 한 숫자로 이 세 단계를 설명할 수 없다.

DocumentID와 ContentID를 분리한다. 같은 bytes가 여러 URL/time에서 수집되면 DocumentID는 다르고 ContentID는 같을 수 있다. normalized view가 같아도 raw payload와 rights 상태가 다를 수 있다. winner가 다른 source의 policy를 상속하지 않는다. 삭제 요청이 loser URL에만 해당하는지 shared content 전체에 해당하는지도 policy가 정한다.

MinHash option은 shingle unit과 size, number of permutations, banding, seed와 normalization view다. state는 signature와 bucket assignment다. effect는 candidate recall, memory, shuffle byte와 component topology다. seed를 checkpoint하지 않으면 repartition rerun에서 후보가 달라질 수 있다. toy documents의 exact Jaccard와 estimated similarity를 비교해 threshold 주변 오차를 측정한다.

**transitive component의 반례**

A와 B, B와 C가 threshold 이상인데 A와 C는 미만일 수 있다. connected component 방식은 셋을 하나로 묶는다. pair-only drop은 다른 결과를 낸다. 세 문서 fixture로 component rule을 명시한다. 긴 boilerplate가 서로 다른 본문을 연결하는 bridge가 되지 않도록 shingle view와 per-domain 검사를 한다.

winner option은 earliest capture, highest quality, rights status, canonical source, length와 stable hash tie-break를 조합할 수 있다. state는 component membership과 winner reason이다. effect는 retained domain·language·time distribution이다. distributed partition order나 worker completion time을 tie-break로 쓰지 않는다. input 순서를 shuffle한 반복에서 같은 winner와 loser map이 나와야 한다.

global dedup은 shard-local dedup보다 shuffle 비용이 크지만 cross-shard duplicates를 찾는다. two-level 전략이면 local 단계에서 제거한 정보가 global 후보 생성에 필요한지 확인한다. component ID와 loser→winner edge를 release artifact로 저장한다. loser payload retention policy가 있더라도 identity와 deletion propagation은 유지한다.

**benchmark 오염을 문자열 검출과 영향 추정으로 나눈다.**

오염 registry는 benchmark 이름만 아니라 exact prompt, answer, normalized variants, translations, structured template와 release timestamp를 가진다. detector는 exact substring, n-gram, fuzzy, semantic candidate를 사용할 수 있다. 각 method의 search view, threshold와 false-positive review를 저장한다. 짧은 common phrase hit를 benchmark contamination으로 단정하지 않는다.

문서 수준 reject는 contamination span이 작은 경우 많은 clean text를 제거할 수 있다. span redaction, paragraph reject, document reject와 component propagation 정책을 비교한다. dedup winner만 검사하고 loser에서 오염을 놓치면 다른 release에서 winner가 바뀔 때 되살아날 수 있다. component 모든 member의 annotations를 winner decision에 연결한다.

오염은 data cutoff와도 관련된다. benchmark가 공개되기 전 capture에는 같은 문자열이 자연 source에서 있었을 수 있다. provenance time, source type와 answer-specific evidence를 함께 본다. release timestamp가 HTTP capture time과 같은지 archive import time인지 구분한다. 불확실한 time을 정확한 날짜처럼 쓰지 않는다.

**contamination failure suite**

fixture에는 exact prompt+answer, prompt only, paraphrase, translated version, template with different values, common sentence와 code snippet을 넣는다. detector별 expected candidate와 final policy를 정한다. normalization이 punctuation을 지워 false match를 만들거나 Unicode trick으로 miss하는 경우를 포함한다.

poisoned benchmark registry도 시험한다. 공격자가 지나치게 일반적인 phrase를 넣어 대규모 corpus를 제거할 수 있다. registry artifact는 review, checksum과 version 권한을 가져야 한다. 새 pattern의 estimated impact를 commit 전에 계산하고 removal mass가 bound를 넘으면 승인 단계를 요구한다.

영향 추정은 contaminated document count를 넘어 realized token exposure를 본다. curriculum과 sampling 때문에 retained contamination이 여러 epoch 소비될 수 있다. 6장의 mixture sampler와 UpdateID ledger를 통해 benchmark span이 실제 batch에 들어갔는지 연결한다. 발견됐다는 사실과 model이 학습했다는 사실을 구분한다.

**curriculum과 mixture를 planned probability에서 realized mass까지 검산한다.**

corpus release가 여러 source/domain/language bucket을 가지면 sampler는 step 또는 token horizon에 따른 weight를 사용한다. planned weight `w_k(t)`는 실제 소비 mass가 아니다. shard availability, document length, packing efficiency, exhausted iterator, retry와 worker straggler가 realized distribution을 바꾼다. 매 UpdateID에 source counts와 valid token mass를 집계한다.

curriculum option은 schedule knot, interpolation, temperature, cap/floor와 replacement policy다. state는 current horizon, bucket iterator cursor와 RNG다. effect는 gradient에 들어간 source별 token mass와 data reuse다. checkpoint에 horizon만 저장하고 bucket cursor를 빼면 resume 뒤 같은 weight여도 다른 document를 소비한다.

예를 들어 planned A:B가 70:30인데 A 문서가 짧아 packing waste가 크면 sample count 70:30과 valid token 70:30이 다르다. objective가 token mean이면 valid token mass를 기준으로 비교한다. document-balanced 목적이라면 document weight가 loss에 어떻게 반영되는지 명시한다. sampler probability와 loss weight를 혼동하지 않는다.

**curriculum 실패 주입**

bucket shard 하나를 지연시키거나 checksum 실패로 격리한다. sampler가 B를 자동 대체해 throughput을 유지하면 realized distribution이 달라진다. 허용 deviation과 pause/fail 정책을 정한다. 조용한 fallback은 금지한다. dashboard가 planned, available와 consumed mass를 같은 시간축에서 보여야 한다.

resume 직전 schedule knot에서 checkpoint를 만든다. uninterrupted와 resumed run의 next bucket IDs, sample IDs와 horizon을 비교한다. world size가 바뀌면 global sampler cursor를 새 ranks로 재배치한다. rank-local seed만 저장했다면 exact curriculum resume를 주장하지 않는다. 15장의 elastic topology와 연결한다.

long-tail bucket이 replacement로 반복되면 unique documents와 exposure count를 본다. 높은 weight가 데이터 다양성 증가가 아니라 소수 document 재사용일 수 있다. per-document exposure를 모두 저장하기 어렵다면 count-min sketch 또는 층화 audit를 사용하되 오차 bound와 seed를 기록한다.

**provenance와 소비 원장을 양방향 query로 설계한다.**

forward query는 raw capture에서 parser child, normalized document, annotations, dedup component, accepted release shard, tokenizer span, packed sample와 UpdateID로 간다. reverse query는 model update 또는 삭제 대상에서 upstream documents와 raw locators로 돌아간다. 모든 payload를 한 database에 넣을 필요는 없지만 stable IDs와 signed manifests로 edge를 연결해야 한다.

edge에는 parent/child IDs, transformation ID와 config digest, input/output byte 또는 span map, decision reason, generation과 artifact checksum이 있다. 단순 file path는 compaction과 object relocation 뒤 안정적이지 않다. locator와 content identity를 나눈다. 개인정보 접근 제한 때문에 text를 볼 수 없어도 authorized checksum verifier와 aggregate query가 lineage completeness를 증명할 수 있다.

소비 원장은 release에서 dataloader로 넘겼다는 기록과 optimizer commit에 기여했다는 기록을 구분한다. prefetched 후 process crash로 버려진 batch, all-ignored batch, overflow skip과 partial accumulation이 있기 때문이다. `Loaded`, `BackwardContributed`, `CommittedUpdate` 사건을 나누고 UpdateID와 denominator를 붙인다. 삭제 영향 범위를 계산할 때 원하는 의미에 맞는 사건을 선택한다.

**양방향 보존식**

accepted DocumentID 하나의 token spans를 모두 모으면 tokenizer가 소비한 normalized range와 padding/특수 token을 제외하고 대응해야 한다. packed sample의 source spans를 역추적하면 release manifest 안의 committed documents만 나와야 한다. tombstoned source에서 forward query한 affected artifacts와 각 artifact에서 reverse query한 source가 서로 포함 관계를 만족해야 한다.

property test는 작은 corpus에서 모든 edge를 전수 검사한다. 대규모 release에서는 manifest count·checksum 보존, 층화 sample과 deletion-specific full query를 조합한다. edge 누락을 빈 결과로 정상 처리하지 않는다. query engine timeout과 truly no consumption을 다른 status로 반환한다.

schema migration은 old IDs를 새 IDs로 다시 발급하지 않는다. mapping artifact와 migration generation을 만들고 양방향 lookup을 시험한다. checksum algorithm을 바꿔도 ContentID namespace에 algorithm을 포함한다. 서로 다른 digest를 문자열만 같다고 합치지 않는다.

**corpus supply chain과 삭제 실패를 한 release state machine으로 묶는다.**

release 상태는 `Building`, `Validated`, `Committed`, `Deprecated`, `Tombstoned`처럼 전이한다. shard가 모두 써졌다는 사실은 committed가 아니다. root manifest가 expected inputs, transformation artifacts, decision policy, shard checksums, deletion revision과 validation report를 묶은 뒤 publish되어야 한다. loader는 `Committed`와 허용 deletion revision만 읽는다.

supply-chain 공격은 parser wheel 교체, classifier alias 이동, config injection, raw shard 바꿔치기, manifest rollback과 cache poisoning으로 나눈다. dependency lock과 content hash, signed or access-controlled manifest, isolated build와 runtime loader verification을 사용한다. 27장의 공급망 재현성과 연결하되 corpus-specific 영향은 accepted document diff와 token mass로 계산한다.

**공격과 삭제 failure matrix**

parser artifact checksum을 바꾸고 같은 version label을 유지한다. build는 expected hash mismatch로 멈춰야 한다. classifier output shard 하나를 old generation으로 바꾼다. manifest component generation 검사가 잡아야 한다. dedup loser map을 삭제한다. coverage invariant가 component count mismatch를 보여야 한다. release root를 과거 deletion revision으로 rollback한다. loader admission이 거절해야 한다.

삭제 요청은 identity resolve, policy scope, affected graph query, tombstone generation, derived artifact rebuild와 consumer admission 순서다. URL 문자열만으로 identity를 찾으면 redirect, mirror와 normalized duplicate를 놓친다. requester authority와 legal basis는 기술 quality score와 별 state다. uncertain match는 review queue에 넣고 무관한 대량 삭제를 자동 실행하지 않는다.

deletion rebuild 중 old release를 계속 소비할지 즉시 pause할지는 정책이다. 요청 severity와 SLA, checkpoint lineage를 고려한다. 어느 쪽이든 어떤 UpdateID까지 old data가 소비됐는지 기록한다. model state에서 이미 학습된 효과를 제거하는 문제는 23장의 unlearning 범위이며 corpus tombstone과 같지 않다.

failure experiment는 deletion graph query timeout, affected shard rebuild 실패, new root publish 전 coordinator crash와 stale worker cache다. old root는 원자적으로 유지되거나 policy에 따라 tombstoned되어야 하고 half-new mixture가 보이면 안 된다. worker는 heartbeat에서 deletion revision을 확인하고 mismatch면 다음 batch admission 전에 멈춘다.

**대규모 corpus release의 종단 재구성 실험.**

인수 실험은 수십 개 문서의 synthetic corpus에서 시작한다. exact duplicate, near duplicate chain, multilingual short text, long boilerplate, malformed HTML, PII span, benchmark contamination, rights reject, poison marker와 deletion target을 의도적으로 넣는다. 각 문서는 stable expected terminal state와 lineage를 가진다. random text만으로 detector 분기를 우연에 맡기지 않는다.

먼저 fetch와 raw commit 경계를 검증한다. retry, redirect, truncated compression과 duplicate success를 주입하고 expected RequestID, response attempt, raw checksum이 유지되는지 본다. 다음에는 parser/normalizer에서 byte span, idempotence, malformed recovery와 candidate upgrade diff를 검사한다. 마지막으로 annotation/policy 단계에 detector artifact, threshold boundary, timeout과 mixed generation을 넣어 판정 계보가 흔들리지 않는지 확인한다.

그다음 dedup 단계에서 exact·near duplicate와 transitive component를 만들고, partition을 섞어도 stable winner가 유지되는지 확인한다. contamination·PII·rights 단계에서는 detector score와 policy decision을 분리하고 scope와 false-positive control을 각각 검증한다. 마지막 curriculum release 시험에서는 작은 sampler로 planned weight, available shard와 consumed valid-token mass를 재생한다.

일곱째는 provenance query다. raw→packed→UpdateID forward와 UpdateID→source reverse를 전수한다. prefetched-but-not-committed batch가 `Loaded`에는 있지만 `CommittedUpdate`에는 없는지 확인한다. 여덟째는 checkpoint/resume와 elastic worker 변화다. sampler cursor, schedule horizon과 realized mass를 비교한다.

아홉째는 deletion이다. 한 raw source, dedup loser와 shared component를 각각 tombstone해 영향 범위가 policy와 맞는지 본다. new root 이전 worker가 stale cache를 읽지 못해야 한다. 열째는 공급망 공격이다. artifact와 manifest rollback, shard corruption을 loader가 fail-closed로 잡는지 확인한다.

synthetic 전수 시험 뒤 production 규모에서는 count·byte·token 보존식, 층화 sample와 anomaly class를 감사한다. 모든 문서를 사람이 보지 못해도 transformation별 input/output count, terminal state 합, component coverage, accepted shard checksum과 consumed mass를 맞출 수 있다. 확률적 detector는 calibration sample과 confidence interval을 보고한다.

인수 보고서는 corpus 크기 숫자로 시작하지 않는다. source scope와 fetch 성공/실패, parse yield, decision별 count/token, dedup component, contamination·PII·rights 상태, release checksum, deletion revision과 consumption horizon을 순서대로 제시한다. 각 수치는 query와 artifact 좌표를 가진다.

비용도 같은 계보로 연결한다. fetch network byte, raw storage, parser CPU, detector accelerator, dedup shuffle, lineage index와 deletion rebuild를 transformation별로 계산한다. 비용 절감 option이 coverage와 provenance를 어떻게 바꾸는지 적는다. 원본 retention을 줄이면 storage는 줄지만 parser 회귀와 삭제 증명 능력이 약해질 수 있다.

마지막 검토자는 accepted document, rejected document, duplicate loser, contaminated span과 tombstoned source를 하나씩 고른다. 각각 raw bytes, transformation decisions, release 포함 여부와 소비 사건을 양방향으로 재생한다. terminal state가 없는 입력, parent가 없는 output와 root 밖 shard가 하나라도 있으면 승인하지 않는다.

이 장의 corpus는 text 파일 모음이 아니라 수집 요청에서 optimizer update까지 이어지는 versioned state machine이다. source code의 고정 함수가 각 변환을 수행하고, 실패 실험이 원자성과 편향을 반증하며, provenance와 소비 원장이 결과를 되짚게 한다. 이 조건을 만족한 committed release만 5장의 tokenizer와 6장의 mixture에 전달한다.

**source code를 stage contract와 failure hook로 읽는다.**

대규모 pipeline의 source를 읽을 때 executor의 최상위 `run` 함수만 고정하지 않는다. reader가 manifest와 shard를 여는 함수, block이 document를 변환하는 함수, writer가 temporary artifact와 commit marker를 만드는 함수, retry가 exception을 분류하는 함수, metrics가 terminal state를 세는 함수를 각각 좌표화한다. 고정 revision, path, symbol, caller와 config schema를 한 카드에 넣는다. 동일 이름 함수가 worker와 coordinator에서 다른 책임을 가질 수 있으므로 process role도 적는다.

reader contract는 input release ID, expected schema와 checksum, partition assignment를 소비한다. 반환하는 record에는 DocumentID와 parent artifact가 있어야 한다. 순수 payload iterator만 넘기면 downstream failure에서 어느 input이 사라졌는지 알 수 없다. writer contract는 output child ID, transformation generation과 checksum을 만들고 성공한 partition만 manifest 후보로 올린다. coordinator는 모든 expected partition의 terminal report를 검증한 뒤 root를 publish한다.

DataTrove류 block graph를 분석할 때 block 순서가 의미다. URL/metadata filter를 parser 뒤로 옮기거나 dedup signature를 normalization 전후 어디서 만드는지에 따라 결과가 달라진다. executor option이 task 병렬도만 바꾸는지 partition/winner order까지 바꾸는지 확인한다. source card에는 `reads`, `emits`, `drops`, `side_outputs`, `mutable state`, `retry boundary`를 적는다.

**stage별 최소 code fixture**

reader fixture는 valid shard, checksum mismatch, truncated final line, unknown schema와 duplicate DocumentID를 넣는다. fail-closed reader는 manifest 밖 file을 glob으로 추가하지 않는다. malformed record 하나를 skip하도록 허용하면 reject side output과 count가 있어야 한다. 전체 shard를 abort하는 정책과 document quarantine 정책을 config로 구분한다.

parser/filter fixture는 input record를 deep copy하거나 immutable하게 취급하는지 본다. in-place mutation 뒤 exception이 나고 retry하면 normalization이 두 번 적용될 수 있다. idempotence로 충분하지 않은 stage는 original parent에서 새 child를 만들어야 한다. test hook가 첫 write 뒤 exception을 내고 retry 결과를 clean run과 비교한다.

writer fixture는 partition 파일을 절반 쓴 뒤 process를 종료한다. partial object는 final namespace와 root manifest에 나타나지 않아야 한다. retry가 같은 partition을 완성하면 content checksum과 record order 정책을 확인한다. 두 worker가 같은 lease를 처리했을 때 stable attempt winner를 정하고 loser artifact를 garbage-collection 대상으로 표시한다.

metrics fixture는 input count가 accepted, rejected, quarantined와 failed 합으로 보존되는지 검사한다. child를 여러 개 만드는 split stage와 여러 parent를 하나로 묶는 dedup stage는 별 보존식을 쓴다. 단순 input=output count를 모든 stage에 강제하지 않는다. count와 byte/token mass를 함께 저장해 empty 또는 비정상적으로 긴 child를 탐지한다.

## 4.13 packed sample과 장기 drift를 training commit에 잇는다

출판된 corpus가 trainer에서 소비되지 않았다면 품질 변화가 model update에 미친 영향을 말할 수 없다. `DocumentID→TokenSpan→PackedSampleID→UpdateID`를 이어 계획한 mixture와 실제 contribution을 비교하고, 장기 drift는 기존 release의 수치를 덮는 대신 새 generation의 원인으로 기록한다.

### 4.13.1 packed sample 소비를 UpdateID까지 연결한다

tokenizer는 accepted DocumentID의 normalized character 또는 byte spans를 token spans로 변환한다. packer는 여러 token spans를 fixed-length sample에 넣고 boundary, special token, label-ignore와 position map을 만든다. dataloader는 packed sample을 rank와 update에 배정한다. 소비 원장은 이 세 변환을 분리한다. PackedSampleID만 저장하면 어느 document가 어느 loss 위치에 들어갔는지 알 수 없다.

packed record의 최소 필드는 `PackedSampleID`, tokenizer digest, sequence length, ordered segment list, 각 segment의 DocumentID·source span·token offset, inserted special-token positions, ignored label positions와 packer generation이다. payload checksum은 token IDs와 attention/segment metadata를 함께 덮는다. token만 같고 document boundary mask가 다른 sample은 같은 학습 입력이 아니다.

loader 사건은 `Assigned`, `Materialized`, `Forwarded`, `BackwardContributed`, `Committed`로 나눈다. GPU prefetch까지 갔지만 OOM으로 update가 abort된 sample은 materialized였어도 committed가 아니다. gradient accumulation에서는 여러 PackedSampleID가 하나의 UpdateID에 연결된다. valid token denominator와 loss weight를 edge에 두어 realized training mass를 계산한다.

**packer와 consumer failure experiment**

두 document A와 B를 한 sequence에 pack하고 boundary mask 한 칸을 의도적으로 제거한다. token payload checksum만 보는 validator는 통과하지만 segment-aware attention fixture는 실패해야 한다. target shift가 A 마지막에서 B 첫 token으로 넘어갈지 ignore할지는 packing policy가 정한다. 3장의 GoldenBatch pair 목록과 6장의 packing 계약으로 검증한다.

source span이 token normalization 때문에 일대일이 아닐 수 있다. tokenizer가 한 Unicode sequence를 합치거나 byte fallback을 쓰면 exact char offset 대신 mapping relation과 uncertainty를 저장한다. deletion query가 token 하나를 다른 document에 잘못 귀속하지 않게 tokenizer의 offset API와 독립 round trip을 시험한다.

prefetch queue에 sample 세 개가 있고 첫 update 뒤 crash가 나는 fixture를 만든다. exact resume이면 committed UpdateID 다음의 logical sampler state에서 다시 배정하고, queue의 uncommitted sample 처리 규칙을 적용한다. sample exposure를 “loader가 읽음”으로 셀지 “optimizer commit”으로 셀지 report에 명시한다. contamination 영향에는 둘을 별 열로 제공한다.

corpus tombstone 뒤 stale packed cache를 주입한다. packer artifact의 parent release와 deletion revision이 loader admission과 다르면 거절해야 한다. cache file mtime이나 directory 이름으로 최신을 추정하지 않는다. packed manifest가 source release root checksum을 직접 참조하고 resolver가 chain 전체를 검증한다.

curriculum weight가 바뀌는 knot에서 packer queue가 이전 mixture sample을 들고 있을 수 있다. strict boundary는 queue를 비우거나 sample에 planned horizon을 붙여 소비 허용 범위를 정한다. throughput을 위해 carry-over를 허용하면 realized mass에 old/new generation을 정확히 센다. schedule 변경이 즉시 적용됐다는 거짓 기록을 피한다.

### 4.13.2 변경·삭제·소비 보존식을 다시 계산한다

parser option 하나가 바뀌면 normalized ContentID에서 시작해 quality feature, dedup signature, contamination match, token span과 packed sample까지 연쇄적으로 달라질 수 있다. 따라서 변경 전에 affected graph query로 후보 범위를 계산한다. 전수 rebuild가 필요 없다고 판단했다면 unaffected edge의 함수가 실제로 동일하다는 사실을 source contract와 checksum으로 증명해야 한다. file timestamp가 같다는 이유만으로 기존 결과를 재사용해서는 안 된다.

quality threshold 하나를 바꾸면 raw annotation은 재사용할 수 있지만 decision, accepted release, dedup winner와 curriculum available mass가 바뀔 수 있다. winner가 reject되면 component의 다음 후보를 선택하는지 component 전체를 다시 계산하는지 정책을 확인한다. downstream packed cache는 parent release ID가 달라져 재검증 대상이다.

삭제 revision 하나를 올리면 tombstone 대상과 dedup/component descendants, accepted shard, tokenizer와 packed sample의 영향 집합을 계산한다. 이미 committed UpdateID는 별 exposure ledger에 남긴다. 새 corpus release를 만드는 것과 model에서 영향을 제거하는 것은 다른 작업이다. 23장의 unlearning job은 이 소비 원장을 입력 근거로 사용할 수 있다.

**최종 숫자 네 개**

첫 번째 검산은 input terminal 보존이다. fetched raw records 전체가 parse accepted, rejected, quarantined와 failed terminal로 빠짐없이 분해되어야 한다. 이어 dedup coverage에서는 eligible DocumentID가 component winner나 loser 가운데 하나로 정확히 한 번 분류되는지 본다. release-to-packed coverage는 accepted token mass를 packed, policy-excluded, tail-buffer와 error의 합으로 설명한다. 마지막 packed-to-update 검산에서는 assigned sample 전체가 committed, aborted, pending과 replay 상태로 닫히는지 확인한다.

각 숫자는 document count와 byte 또는 valid token mass를 함께 가진다. count가 맞아도 큰 shard 하나가 누락될 수 있고, byte가 맞아도 identity가 중복될 수 있다. set/hash coverage와 aggregate sum을 함께 사용한다. probabilistic sketch를 쓰면 collision과 error bound를 보고하고 작은 release의 exact validator로 구현을 교차 검사한다.

**삭제·재개 경계의 negative run.**

한 production-like 축소 release에 여섯 오류를 한 번씩 독립 주입한다. raw checksum mismatch는 reader, parser artifact 교체는 build admission, unstable dedup winner는 shuffled replay, contamination registry rollback은 policy gate, stale deletion revision은 loader, packed boundary 오류는 GoldenBatch가 각각 최초 검출해야 한다. 한 run에 오류를 합치지 않는다.

모든 detector가 예상 경계에서 실패한 뒤 clean run을 다시 만든다. root manifest, decision count, dedup map, contamination report, deletion revision, packed manifest와 소비 원장 checksum을 보존한다. 작성자와 다른 검토자가 accepted·rejected·duplicate·contaminated·deleted·committed 표본을 하나씩 재생한다.

최종 release note에는 검증한 source 범위, capture time 의미, parser/detector revisions, dedup parameters, contamination registry, curriculum과 deletion revision을 정확히 쓴다. 미수집 domain, 지원하지 않는 MIME, 미실행 언어와 알려진 detector blind spot도 같은 비중으로 적는다. “clean corpus”처럼 범위를 지우는 표현을 쓰지 않는다.

이 검산이 통과하면 corpus의 품질 주장은 문서 수나 평균 score에 머물지 않는다. 어떤 source가 어떤 code와 policy를 지나 어느 release와 packed sample이 되었고, 실제 어느 committed update에 얼마나 기여했는지 답할 수 있다. 실패와 삭제도 같은 graph에서 재생된다. 그때 비로소 대규모 corpus가 재현 가능한 학습 입력이 된다.

### 4.13.3 drift를 새 release 근거로 바꾼다

웹 source와 detector는 시간이 지나며 변한다. 같은 seed URL도 payload, redirect와 policy가 달라지고 language·quality classifier의 새 artifact는 score 분포를 바꾼다. 운영자는 최신 crawl을 과거 release에 덮어쓰지 않고 child generation으로 만든다. parent와 child의 fetch status, normalized content, decision, dedup component와 accepted token mass를 원인별로 diff한다.

drift dashboard는 전체 평균만 보여 주지 않는다. domain, language, MIME, size, capture cohort와 decision reason별 count·byte·token 변화를 보여 준다. source coverage 감소와 stricter threshold 효과를 분리한다. parser yield가 급락하면 network 실패, content encoding, markup 변화와 extractor regression을 순서대로 조사한다. classifier acceptance 변화는 input composition과 model score shift를 paired sample로 나눈다.

옵션은 drift threshold, sample size, baseline window와 automatic pause다. 상태는 candidate release, paired-document sample과 open anomaly다. 효과는 publish latency와 잘못된 release의 blast radius다. threshold를 너무 넓히면 systematic loss를 놓치고 너무 좁히면 정상 계절 변화를 장애로 만든다. 과거 분포와 high-risk source에 다른 bound를 둘 수 있지만 policy revision을 기록한다.

**canary release와 promotion**

전체 rebuild 전에 source와 language를 층화한 canary partition을 같은 pipeline으로 처리한다. canary는 별 코드 경로가 아니라 production graph의 작은 입력이어야 한다. source functions, artifacts와 policy가 동일한지 digest로 확인한다. canary에서 parser differential, threshold pairs, dedup stability, contamination·PII와 lineage query를 실행한다.

canary 통과는 full release 성공을 보장하지 않는다. rare archive format, huge component와 cross-partition duplicate는 규모에서만 나타날 수 있다. full build 동안 stage 보존식과 anomaly bound를 streaming으로 검사하고 breach면 root publish를 막는다. 부분 결과는 diagnostic generation이며 loader가 소비하지 않는다.

promotion은 candidate root, validation report, deletion revision과 curriculum availability를 원자적으로 활성화한다. 이미 실행 중인 trainer가 언제 새 release로 전환하는지 명시한다. update 중간에 shard resolver가 바뀌지 않도록 run 또는 curriculum boundary를 사용한다. old release를 읽은 UpdateID와 new release를 읽은 UpdateID가 consumption ledger에서 구분되어야 한다.

rollback은 old root를 다시 가리키는 것뿐 아니라 old deletion policy를 되살리지 않아야 한다. 새 release가 품질 문제로 rollback되어도 이후 접수된 tombstone은 유지해야 한다. corpus content generation과 deletion revision을 독립 축으로 관리하는 이유다. rollback fixture는 old content/new deletion 조합이 loader와 lineage query에서 일관적인지 확인한다.

**운영 감사의 종료 조건**

한 release window가 끝나면 planned와 realized source token mass, loader failure, aborted/committed exposure, deletion SLA와 drift incidents를 정산한다. 예상과 실제의 차이를 다음 sampler와 capacity 계획에 반영한다. 문서 count가 아니라 optimizer에 도달한 valid token을 최종 소비량으로 보고한다.

발견된 새 failure class는 synthetic corpus에 최소 반례로 추가한다. source payload를 그대로 복제하기 어려우면 구조를 보존한 비식별 fixture를 만든다. detector 수정은 그 fixture뿐 아니라 기존 정상·negative suite를 모두 통과해야 한다. incident 해결을 위해 임시 skip한 validator에는 만료 조건을 둔다.

장기 운영의 성공은 anomaly가 없었다는 선언이 아니다. 새 source 변화와 공격을 빠르게 격리하고, 어느 release와 update가 영향을 받았는지 계산하며, complete parent에서 안전한 child를 다시 만드는 능력이다. 원본·결정·dedup·packed·소비 원장이 같은 generation 계보를 유지할 때 이 능력을 반복해서 증명할 수 있다.

**crawl snapshot을 request·response·WARC record로 고정한다.**

crawl snapshot은 URL 목록이나 다운로드 폴더가 아니다. SeedID, frontier revision, normalized URL, request time, redirect chain, request headers, response status와 headers, payload bytes, transport error, retry attempt와 capture policy를 연결한 immutable generation이다. 동일 URL도 시간과 negotiation header에 따라 다른 표현을 반환하므로 URL을 문서 identity로 쓰지 않는다. CaptureID는 request와 받은 byte의 digest, 시각과 attempt를 묶는다.

WARC writer는 response만 저장하지 않고 request, response, metadata와 revisit relation을 구분한다. payload digest가 기존 capture와 같을 때 revisit record를 만들더라도 원 capture가 retention 범위에 있고 reader가 relation을 해석할 수 있어야 한다. compressed WARC의 record offset, compressed length, uncompressed payload length와 block/payload digest를 index에 둔다. 파일 순번과 offset만으로 identity를 만들면 repack 뒤 계보가 끊긴다.

frontier state는 discovered URL, source link, depth, priority, next eligible time, robots decision, attempt count와 terminal reason을 가진다. distributed crawler가 lease를 얻고 fetch하며 결과를 조건부 commit한다. worker가 timeout 뒤 늦게 응답해도 만료된 lease generation으로 성공을 덮어쓰지 못한다. duplicate fetch는 허용할 수 있지만 동일 attempt의 conflicting terminal state는 허용하지 않는다.

snapshot cutoff는 wall-clock 하나가 아니라 accepted frontier events의 high-water mark다. cutoff 뒤 늦게 도착한 response는 다음 generation으로 보내거나 explicit late partition으로 둔다. root manifest는 WARC object 목록, index digest, frontier checkpoint, policy artifacts, capture counts와 실패 reason 분포를 가진다. 완전하지 않은 upload는 root에 포함하지 않는다.

수집 보존식은 enqueued=leased+eligible+terminal+deferred 같은 frontier count와 request attempts=responses+transport failures+active leases를 비교한다. redirect loop, oversized response, unsupported MIME와 policy block도 terminal reason으로 센다. 성공 response 비율 하나가 높아도 특정 domain이나 language의 systematic loss를 숨길 수 있으므로 source cohort별로 계산한다.

**robots·license·privacy를 capture와 use policy로 분리한다.**

robots 판단은 fetch 시점의 user-agent, robots URL, fetched body digest, status, parsed rule, crawl delay와 evaluated path를 기록한다. robots.txt가 나중에 바뀌어도 과거 판단을 재구성할 수 있어야 한다. network failure, 404, 5xx와 parse error를 같은 allow로 합치지 않고 policy가 정한 state로 처리한다. cached robots rule에는 expiry와 source CaptureID를 둔다.

capture 허용과 training use 허용은 다른 결정이다. 보존·감사 목적의 원본 capture가 허용되어도 license, terms, privacy 또는 source policy 때문에 text release에서 제외될 수 있다. CaptureDecision과 UseDecision을 별도 revision으로 만들고 reason, evidence, reviewer와 expiry를 기록한다. use policy가 바뀌면 WARC를 다시 받지 않고 기존 immutable capture에 새 결정을 적용할 수 있다.

license detector는 page footer 문자열만으로 확정하지 않는다. HTTP header, embedded metadata, linked license page, site policy와 document-level notice를 evidence로 모으고 conflict를 드러낸다. unknown을 permissive license로 치환하지 않는다. domain-wide rule과 page-specific override의 precedence를 명시하고 evidence가 사라질 때를 대비해 허용 범위 안에서 snapshot을 보존한다.

privacy gate는 public accessibility를 무조건 consent로 보지 않는다. account wall 우회, session identifier, personalized page, private share link와 access token이 있는 URL을 차단한다. query parameter normalization 전에 sensitive token을 마스킹하되 raw request의 접근은 제한된 vault relation로 남긴다. 로그와 error message에도 credential과 personal query가 복사되지 않게 한다.

정책 변경은 affected CaptureID set과 descendant normalized document, dedup winner, packed shard와 training exposure를 계산한다. deletion만 실행하고 영향 보고를 생략하지 않는다. dispute 상태에서는 새 release와 loader 접근을 정지하고 보존 의무가 있는 audit artifact는 별 access class로 격리한다.

**byte에서 text로 가는 extraction을 span relation로 검증한다.**

extractor 입력은 payload bytes, declared content type/charset, detected encoding, parser revision과 resource limits다. output text의 각 block은 source byte 또는 DOM node relation, block role, visibility와 extraction reason을 가진다. 완벽한 byte-to-character 역함수가 불가능해도 최소한 title, paragraph, code, table cell과 alt text가 어디서 왔는지 추적한다.

encoding 선택은 header, BOM, meta charset와 detector confidence를 precedence rule로 처리한다. replacement character 비율, undecodable byte span과 fallback encoding을 기록한다. UTF-8로 강제 decode해 글자가 깨진 문서가 language filter에서 탈락하는 연쇄를 막는다. 동일 payload를 parser revision 전후에 paired diff해 text yield와 replacement 변화량을 본다.

HTML extraction은 script/style/navigation/footer/form, hidden node, repeated menu와 main content를 구분한다. CSS를 완전히 실행하지 않는 parser는 visibility 판단의 한계를 reason으로 남긴다. boilerplate 제거가 짧은 Q&A, poetry, code와 표를 과도하게 지우지 않는지 document genre별 fixture를 둔다. DOM bomb, deeply nested markup와 huge attribute에는 node/depth/time bound가 필요하다.

PDF, office, plain text, JSON, forum archive 같은 MIME은 별 adapter와 sandbox를 사용한다. claimed MIME과 magic byte가 다르면 quarantine한다. OCR을 쓴 경우 engine/model, page, bounding box와 confidence를 남겨 native text와 섞지 않는다. extraction 실패를 empty document로 publish하지 않고 unsupported, corrupt, timeout, encrypted와 policy blocked로 분류한다.

table과 code는 단순 줄바꿈으로 평탄화할 때 구조 손실이 생긴다. cell boundary, header relation, code fence language와 line order를 보존할 serialization을 정한다. 모델 소비 형식으로 변환한 text와 원 extraction block을 별 artifact로 두면 template 변경을 원 payload 재파싱 없이 검증할 수 있다.

**Unicode normalization과 canonical text를 손실 원장으로 만든다.**

Unicode 처리는 `NFC` 또는 `NFKC` 한 줄로 끝나지 않는다. input code point, normalization form, case folding, whitespace, control/format character, confusable과 invalid sequence 규칙을 stage별로 둔다. display text와 dedup canonical text를 분리한다. 학습 text에서 의미 있는 수학 기호나 전각 문자를 보존하면서 dedup key에서는 정책에 따라 동등성을 넓힐 수 있다.

line ending, non-breaking space, zero-width character와 repeated whitespace 변환은 edit event로 기록한다. 각 event는 input/output span, rule ID와 count를 가진다. 전체 원문을 edit log에 중복 저장하지 않아도 transform digest와 aggregate reason으로 differential을 재현한다. control character 제거가 source code indentation이나 bidirectional text 의미를 깨뜨리는 fixture를 둔다.

homoglyph과 bidirectional override는 security detector 입력이지만 자동 치환이 항상 옳지 않다. 의심 span을 flag하고 language/script context와 함께 policy가 결정한다. URL, email, code identifier와 natural text에 다른 rule을 적용한다. normalization 전후 token count, script distribution와 hash를 보존해 급격한 손실을 gate한다.

canonicalizer revision이 바뀌면 exact dedup hash와 near-dedup shingles가 모두 달라질 수 있다. old/new canonical text의 paired sample에서 edit distance, token delta, duplicate component split/merge를 측정한다. hash만 다시 계산하고 dedup winner lineage를 유지하면 잘못된 relation이 남으므로 child release에서 component를 재생성한다.

**language identification을 segment posterior와 routing decision으로 나눈다.**

document 하나에 language label 하나를 붙이면 navigation, quoted text, code와 다국어 문서를 잃는다. 먼저 block 또는 일정 token window별 posterior를 계산하고 document aggregation rule로 primary language, secondary languages와 mixed/unknown 비율을 만든다. classifier model digest, label set, preprocessing, minimum length와 confidence threshold를 기록한다.

짧은 text, transliteration, closely related languages와 code-heavy page에는 confidence가 낮다. low confidence를 dominant language로 강제하지 않고 unknown 또는 mixed queue로 보낸다. script heuristic은 보조 evidence이며 script와 language를 동일시하지 않는다. URL TLD나 domain prior를 쓰면 content posterior와 분리해 sampling bias를 관측한다.

평가는 macro accuracy 하나가 아니라 language별 precision/recall, calibration, confusion pair, length·domain·script slice로 한다. training mixture에서 작은 언어의 false negative가 큰 손실이므로 source prevalence를 반영한 expected retained tokens도 계산한다. human-reviewed stratified sample과 adversarial code-switch fixture를 보존한다.

language routing은 posterior, policy threshold, allowed set와 quota state를 입력으로 accept, reject, mixed split 또는 review를 출력한다. classifier score와 최종 decision을 분리하면 threshold를 바꾸어 재실행할 수 있다. threshold 변경의 효과는 accepted document 수보다 unique canonical token과 dedup 이후 token mass로 본다.

**quality filtering을 feature·score·policy의 세 revision으로 운영한다.**

quality feature에는 text length, alphabetic ratio, line repetition, stopword/function-word, punctuation, entropy, boilerplate ratio, link density, code/table fraction, perplexity proxy와 learned classifier embedding이 들어갈 수 있다. feature extractor revision과 missing-value rule을 고정한다. parser 실패로 text가 짧아진 문서를 low quality로 오분류하지 않게 upstream status를 입력으로 포함한다.

score model은 feature vector를 scalar 또는 class posterior로 바꾸지만 score 자체가 release 결정은 아니다. policy가 language, genre, source cohort별 threshold, quota와 allow/deny rule을 적용한다. global threshold 하나는 low-resource language와 specialist document를 체계적으로 제거할 수 있다. cohort-specific rule에는 근거, 표본 수와 expiry가 필요하다.

filter 평가 set은 accepted/rejected 경계 주변, high-volume domain, rare language, code/math, conversation, spam과 adversarial SEO를 층화한다. reviewer agreement와 uncertainty를 보존한다. precision만 높이려다 valuable recall을 잃거나 반대가 되는 tradeoff를 token mass와 downstream proxy로 보고한다.

여러 filter를 직렬 적용할 때 최종 reject reason 하나만 남기지 않는다. 모든 score와 최초 terminal policy reason, counterfactual로 다른 gate도 실패했는지를 기록한다. stage order를 바꾸면 compute 비용과 observable reason이 달라지므로 같은 sample에서 commutativity를 가정하지 않는다.

quality model 학습 data가 benchmark나 target evaluation을 포함하지 않는지 계보를 확인한다. learned classifier가 특정 문체, 길이 또는 domain 명칭을 품질 proxy로 쓰는지 counterfactual을 만든다. 문장 순서 유지/섞기, harmless footer 추가, 압축/상세화와 domain marker 제거로 score 민감도를 본다.

**exact dedup을 content identity와 winner policy로 분리한다.**

exact dedup key는 raw byte hash, extracted text hash, canonical text hash 중 목적에 따라 다르다. 세 hash를 모두 보존하면 transport duplicate, parser-equivalent duplicate와 normalized duplicate를 구분할 수 있다. empty text와 error placeholder는 같은 hash component로 합치지 않고 upstream failure로 제외한다.

distributed hash partition은 `(canonical_hash, DocumentID)` record를 owner partition에 모아 component를 만든다. partition count가 바뀌어도 hash function과 seed가 같으면 identity가 안정적이어야 한다. spill, retry와 speculative task가 record를 중복 출력할 수 있으므로 idempotent DocumentID로 reduce한다. input count, unique hash, component size 합과 output winner+loser 수를 보존식으로 검사한다.

winner는 먼저 본 문서가 아니라 명시적 policy로 고른다. license/use 허용, extraction completeness, source authority, capture freshness, text quality, stable DocumentID 순의 deterministic ordering을 둘 수 있다. policy revision이 바뀌면 winner가 달라져 descendant lineage도 갱신한다. loser는 삭제하지 않고 winner relation와 reason을 남겨 source별 중복 기여도를 계산한다.

exact component가 거대하면 templated error page나 crawler block page일 수 있다. size distribution의 tail과 representative text를 검사한다. 동일 legal boilerplate를 포함하지만 본문이 다른 문서가 canonicalizer 오류로 합쳐지는 negative fixture를 둔다. hash collision은 드물어도 length와 secondary digest를 확인한다.

**dedup·오염·PII·안전 판단의 증거 경계를 세운다.**

**MinHash·LSH near dedup을 candidate와 verified edge로 나눈다.**

near dedup은 canonical token 또는 character shingle set에서 Jaccard similarity를 근사한다. shingle size, tokenization, minimum document length와 boilerplate 처리에 따라 의미가 달라진다. MinHash signature 길이 m이면 estimator variance가 대략 `J(1-J)/m`이므로 threshold 근처 오차를 예상하고 exact verification band를 둔다.

LSH가 signature를 b개 band와 r개 row로 나누면 similarity s인 pair가 적어도 한 bucket에서 만날 확률은 `1-(1-s^r)^b`다. 목표 threshold 주변 candidate recall과 low-similarity candidate volume을 계산해 b,r을 정한다. configuration만 저장하지 않고 synthetic similarity grid에서 observed recall을 검산한다.

bucket skew는 distributed failure 원인이다. empty/짧은 document, common template와 adversarial repeated text가 giant bucket을 만든다. bucket size cap에서 조용히 candidate를 버리지 않고 heavy-bucket fallback, secondary partition 또는 template stripping으로 보낸다. cap으로 놓친 pair 상한과 affected documents를 보고한다.

candidate edge는 final duplicate relation이 아니다. exact Jaccard, containment 또는 aligned span overlap으로 확인하고 threshold/policy를 적용한다. 짧은 문서가 긴 문서에 완전히 포함될 때 Jaccard는 낮을 수 있으므로 containment를 별도로 본다. quotation, syndicated article와 updated version을 동일 처리할지 use case별 edge type으로 구분한다.

verified edge의 connected component를 만들면 transitive chain에서 양 끝 문서 similarity가 낮을 수 있다. single-link component 전체를 하나로 제거할지 representative-centered cluster를 쓸지 명시한다. component size, diameter sample와 winner-to-member similarity를 audit한다. graph algorithm의 partition/iteration checkpoint와 convergence 상태를 보존한다.

**benchmark contamination을 item lineage와 span evidence로 판정한다.**

benchmark registry는 이름과 test 문자열만 담지 않는다. BenchmarkID, revision, split, ItemID, prompt, answer, rationale, source document, publication 시각, license와 known mirrors를 가진다. training corpus capture time과 benchmark 공개 시점 관계도 기록한다. 공개 전 원 source가 corpus에 있는 것과 benchmark answer가 그대로 유출된 상황을 구분한다.

detector는 exact normalized match, n-gram overlap, MinHash candidate, long common substring, paraphrase retrieval와 code/test signature를 단계적으로 사용한다. 각 hit는 corpus DocumentID와 benchmark ItemID, matched spans, normalization, score와 detector revision을 가진다. score만 남기면 reviewer가 common phrase인지 실제 leakage인지 판단할 수 없다.

prompt-only, answer-only, prompt+answer, rationale와 generated solution은 위험이 다르다. executable benchmark는 hidden test, function signature와 canonical solution overlap을 따로 본다. multilingual translation과 formatting 변형 fixture를 둔다. false positive가 많은 legal/common template는 allow evidence를 두되 blanket domain allow를 피한다.

split contamination은 corpus release뿐 아니라 quality classifier, language model scorer, dedup winner와 synthetic generator의 training data까지 추적한다. benchmark text를 직접 제거해도 filter model이 답을 학습했을 수 있다. artifact lineage에서 affected score와 decisions를 찾아 재계산 범위를 정한다.

영향 추정은 matched document count가 아니라 retained token, duplicate component, packed sample와 consumed UpdateID로 이어진다. 제거 전후 benchmark score 변화만으로 인과를 확정하지 않고 clean-room rebuild와 matched-control set을 사용한다. uncertain hit는 별 quarantine과 sensitivity bound를 둔다.

**contamination release gate를 시간·유사도·소비량으로 닫는다.**

release gate는 benchmark별 허용 규칙을 가진다. exact prompt+answer hit는 즉시 차단할 수 있고, source passage overlap은 publication chronology와 task 목적을 검토할 수 있다. threshold와 reviewer decision, exception reason, expiry를 manifest에 둔다. 새 benchmark revision이 등록되면 기존 corpus generation에 detector를 replay한다.

canary는 known-positive exact/near/paraphrase, known-negative common phrase, translated item, truncated answer와 adversarial Unicode 변형을 포함한다. detector recall과 false positive를 slice별로 측정한다. index build가 일부 partition을 놓쳐도 aggregate score가 좋아 보일 수 있으므로 expected ItemID/DocumentID coverage를 검사한다.

발견 뒤 삭제는 raw capture를 무조건 파기하는 단계와 training use를 차단하는 단계를 정책에 따라 나눈다. dedup winner가 제거되면 valid loser를 승격할지 component 전체를 차단할지 결정한다. 새 winner도 benchmark hit가 없는지 재검사한다. packed shard는 tombstone-aware loader 또는 새 generation rebuild로 차단하고 consumption ledger에서 이미 노출된 범위를 계산한다.

사후 보고서는 detector version, hit 유형, chronology, reviewer verdict, affected release/shard/update, mitigation과 residual uncertainty를 가진다. benchmark score를 다시 계산할 때 동일 checkpoint에 clean evaluation만 적용한 결과와 corpus를 재학습한 결과를 구분한다. contamination 경고를 숫자 하나로 지우지 않는다.

**PII 탐지를 span·confidence·action으로 분리한다.**

PII detector는 document label 하나보다 span record를 출력한다. DocumentID, character와 byte span, category, detector/model revision, confidence, context hash와 proposed action을 가진다. email, phone, address, government identifier, financial account, precise location, health 정보와 사람 이름은 위험과 허용 규칙이 다르다. 공개된 조직 연락처와 사적 연락처도 context만으로 완벽히 구분되지 않으므로 uncertainty를 남긴다.

정규식은 구조화 identifier에 강하지만 Unicode separator, OCR 오류와 분할 token에 취약하다. learned NER는 문맥을 쓰지만 language/domain bias와 false positive가 있다. rule, model, checksum validator와 allowlist evidence를 ensemble하되 어느 detector가 span을 제안했는지 보존한다. checksum을 통과하지 않았다고 민감하지 않다고 결론내리지 않는다.

action은 keep, mask, remove span, remove document, quarantine와 review로 나뉜다. masking은 길이와 위치를 유지할지 typed placeholder를 쓸지 tokenizer 영향과 함께 정한다. overlapping span은 deterministic precedence로 합치고 replacement가 원 offset을 바꾸므로 pre/post span relation를 만든다. mask 뒤 원 text가 nearby context나 duplicate document에서 복원되지 않는지 검사한다.

평가는 category·language·source별 precision/recall과 residual exposed token을 본다. synthetic identifier만으로 실제 OCR, obfuscation과 table context를 대표하지 못하므로 승인된 비식별 fixture와 제한된 human audit를 사용한다. detector가 사람 이름을 과도하게 지워 역사·문학 corpus를 훼손하는 recall/utility tradeoff도 보고한다.

**secret scanner를 credential lifecycle과 연결한다.**

secret은 API key pattern, private key block, access token, password-bearing URL, cloud credential, database connection string와 session cookie를 포함한다. entropy만 높은 문자열은 hash, compressed data와 ID에서 false positive가 많다. prefix/format rule, entropy, context keyword와 provider validator를 조합한다. 외부 validation은 실제 credential을 로그나 제3자 endpoint에 노출하지 않는 승인된 방식만 쓴다.

발견 즉시 corpus 차단만으로 incident가 끝나지 않는다. source owner 또는 provider에 안전한 channel로 통지하고 가능하면 revoke/rotate 상태를 추적한다. corpus record에는 raw secret 대신 keyed digest와 span, category, detector revision, action을 남긴다. 운영 dashboard와 exception trace에 원문이 출력되지 않게 한다.

secret이 exact/near duplicate component에 퍼졌는지 canonical span 또는 keyed fingerprint로 찾는다. code archive의 fork와 quoted incident report도 descendant다. masking 뒤 dedup hash가 바뀌므로 component를 재계산하고 packed shard와 cache를 무효화한다. 이미 학습에 소비된 경우 affected release와 UpdateID 범위를 incident에 포함한다.

negative fixture는 high-entropy benign hash, example key, redacted token과 invalid checksum을 포함하고 positive fixture는 separator 삽입, multiline, URL encoding, Unicode confusable와 archive 내부 secret을 포함한다. scanner update가 known secret을 찾으면서 code corpus를 광범위하게 제거하지 않는지 paired report를 만든다.

**toxicity와 safety filtering을 context·target·use case로 평가한다.**

toxicity score 하나로 문서를 제거하면 인용, 반박, 교육, 피해자 서술과 표적 공격을 구분하지 못한다. detector는 span 또는 segment, category, target, speaker/quote relation, context confidence와 model revision을 출력한다. corpus use policy는 모델 안전 목표, 연구 허용 범위와 법적 요구에 따라 keep, downweight, quarantine 또는 remove를 결정한다.

language와 dialect bias를 반드시 slice한다. 특정 집단 정체성 용어가 존재한다는 이유만으로 toxicity가 높아지는 counterfactual을 만든다. identity term을 교체하고 공격 구조를 유지한 pair, 동일 term을 중립 문장에 둔 pair, quoted speech와 condemnation pair를 비교한다. threshold는 전체 accuracy가 아니라 집단별 false positive/negative와 retained token mass로 검토한다.

극단적 유해 content를 reviewer가 직접 반복 노출되지 않게 access control, sampling limit와 지원 절차를 둔다. raw text를 일반 log나 report에 복사하지 않고 category·digest와 승인된 redacted excerpt를 사용한다. human review decision과 disagreement도 보존해 model score를 확정 사실로 만들지 않는다.

safety filter와 quality filter의 순서가 결과를 바꿀 수 있다. spam으로 먼저 제거된 document에서도 secret이나 심각한 privacy incident를 발견해야 할 수 있다. training use decision과 incident detection pipeline을 분리해 reject된 text도 제한 환경에서 필요한 scan을 수행한다. 삭제 retention policy는 category에 맞게 적용한다.

**multimodal corpus를 asset·relation·rights의 집합으로 만든다.**

multimodal document는 HTML text와 image URL 목록이 아니다. AssetID별 original bytes digest, MIME, dimensions/duration, capture relation, alt text, caption, surrounding block, link target와 rights evidence를 가진다. page의 어떤 text span이 어떤 image, audio 또는 video segment를 설명하는지 relation type과 confidence를 기록한다.

image decode는 claimed MIME, magic byte, dimensions, color mode, orientation metadata와 corruption을 검사한다. decompression bomb와 huge dimension에는 pixel/resource bound를 둔다. perceptual hash와 embedding near duplicate는 resized, cropped, recompressed asset을 찾지만 threshold와 false merge를 audit한다. exact byte duplicate와 visual duplicate를 구분한다.

caption quality는 alt text, filename, nearby text, OCR와 generated caption의 provenance를 분리한다. generated caption을 source human caption으로 재표지하지 않는다. image-text similarity가 높아도 watermark, meme text, unsafe content나 개인정보가 있을 수 있다. OCR span에는 bounding box와 detector revision을 둬 PII/secret scan과 연결한다.

video/audio는 segment boundary, transcript source, timestamp alignment, codec와 sampled frames를 기록한다. 전체 asset license가 segment에도 적용되는지 evidence를 유지한다. 얼굴, voice, biometric signal과 location metadata는 별 privacy gate를 거친다. EXIF를 training payload에서 제거하더라도 audit relation와 deletion 대상은 관리한다.

multimodal dedup winner가 바뀌면 text-image pair relation도 함께 갱신한다. image는 같고 caption이 다른 경우를 duplicate pair로 무조건 합치지 않는다. asset component, caption component와 pair component를 따로 만들고 sampling unit이 어느 component인지 명시한다.

**multimodal 샘플링 편향과 검산을 token·pixel 비용으로 계산한다.**

image 한 장을 document 하나와 같은 weight로 세면 resolution, caption length와 patch/token 비용을 반영하지 못한다. mixture planner는 sample probability 외에 expected text tokens, image patches, decoded pixels, audio seconds와 preprocessing FLOPs를 추정한다. realized ledger는 실제 valid units와 dropped decode를 기록한다.

aspect ratio, resolution, language, source, content category와 caption origin 분포를 joint slice로 본다. 고해상도 stock image가 storage byte를 지배하거나, 짧은 English alt text가 multilingual corpus를 대표하는 것처럼 보일 수 있다. dedup 이후 unique asset와 unique relation 수를 함께 보고한다.

negative fixture는 caption-image mismatch, swapped asset, blank/corrupt image, duplicated caption, wrong orientation, OCR-only leakage와 asset deletion 뒤 dangling relation을 포함한다. loader가 mismatch를 조용히 text-only sample로 바꾸는지 확인한다. 허용 fallback은 explicit modality mask와 reason을 가져야 한다.

**dataset version을 content root·policy root·deletion root로 분리한다.**

하나의 version 문자열에 모든 변화를 숨기지 않는다. content root는 immutable input과 transform artifact의 Merkle-style digest, policy root는 license/privacy/quality/contamination decision revision, deletion root는 현재 유효 tombstone set을 나타낸다. release ID는 세 root와 schema, tokenizer handoff contract를 묶는다.

stage manifest는 parent partitions, code/build digest, configuration, model artifacts, input/output counts·bytes·tokens, rejection reasons, failure partitions와 completion marker를 가진다. child stage는 complete parent만 참조한다. retry가 같은 logical partition을 두 번 만들면 deterministic output digest가 같아야 하며 다르면 nondeterminism incident다.

schema evolution은 field add, semantic change와 representation migration을 구분한다. optional field 추가는 reader compatibility를 유지할 수 있지만 language score 의미나 DocumentID 생성 규칙 변경은 새 major contract다. reader는 모르는 required field를 무시하지 않고 supported schema range를 선언한다.

release diff는 added/removed/changed documents, decision changes, dedup component split/merge, token mass와 source/language distribution을 원인별로 보여 준다. 전체 token delta가 작아도 rare cohort가 사라질 수 있다. paired DocumentID sample로 transform 차이를 검토한다.

**deletion request를 tombstone 전파와 검증으로 완료한다.**

deletion request는 RequestID, requester/evidence, scope selector, legal/policy basis, received time, SLA, reviewer와 status를 가진다. URL만 주어졌을 때 redirect, normalized URL, CaptureID, content hash와 known mirrors를 찾아 scope를 확장하되 unrelated near duplicate를 자동 삭제하지 않는다. exact target과 inferred candidates를 구분해 검토한다.

tombstone은 raw capture, normalized document, dedup component, index/cache, packed shard, dataset release와 training consumer에 전파된다. immutable artifact를 물리적으로 즉시 수정할 수 없다면 access layer에서 차단하고 rebuild/retention purge를 예약한다. 어느 layer가 logical block, physical deletion 또는 pending인지 상태를 낸다.

dedup winner 삭제 뒤 loser 승격은 정책을 다시 실행하는 새 generation이다. 새 winner의 license, PII, contamination와 quality를 재검사한다. tombstoned winner relation을 단순히 제거하면 과거 release의 설명 가능성이 사라지므로 restricted audit record에는 relation와 reason을 보존한다.

삭제 완료 검증은 public loader, internal resolver, search index, cache, backup와 disaster replica에서 target identifier가 접근 불가능한지 probe한다. keyed content fingerprint로 변형 복사본을 찾되 원 text를 report에 노출하지 않는다. already consumed training exposure는 삭제되었다고 거짓말하지 않고 UpdateID 범위와 후속 정책을 기록한다.

## 4.14 분산 실행·streaming·resume의 보존식을 증명한다

단일 stage의 결정성을 확인한 뒤 executor를 여러 worker와 shard로 확장한다. retry, speculative execution, backpressure와 worker loss가 있어도 accepted·rejected·duplicate·committed 합이 보존되고, resume가 마지막 complete partition generation에서 이어지는지를 증명한다.

### 4.14.1 실행 graph를 fingerprint와 shard로 추적한다

Hugging Face Datasets를 사용할 때 `load_dataset`, builder/config, split generation, `map`, `filter`, `select`, `shuffle`, `shard`와 `save_to_disk` 호출을 stage contract에 연결한다. dataset fingerprint가 함수 source, arguments와 parent fingerprint를 어떻게 반영하는지 확인하고 외부 model file이나 environment variable처럼 자동 포착되지 않는 dependency를 explicit digest로 넣는다.

batched `map`은 input row 수와 output row 수가 다를 수 있다. remove_columns, with_indices, num_proc, batch_size와 writer_batch_size를 effective config에 기록한다. worker exception이 partial cache를 남기는지, retry가 동일 fingerprint를 재사용하는지 시험한다. 비결정적 network call을 map 함수 안에 숨기지 않는다.

Arrow shard의 schema, row count, null, large string/list offset와 file digest를 manifest에 둔다. memory mapping success만으로 content를 승인하지 않고 DocumentID uniqueness와 stage 보존식을 검사한다. `shuffle`의 indices mapping과 seed, epoch별 sampler relation을 training handoff에 보존한다.

streaming mode는 random access dataset과 상태 계약이 다르다. source shard order, buffer shuffle size, seed, worker/rank split와 resume cursor를 기록한다. world size 변경에서 정확한 next document 재현이 불가능하면 recovery grade를 낮추고 duplicate/skip bound를 계산한다.

**고정 소스 워크스루: streaming cursor가 문서를 보존해도 shuffle buffer는 보존하지 않는다.**

여기서는 Hugging Face Datasets commit `836b82e0544cabf6474b25ade131b4d21e570373`의 `src/datasets/iterable_dataset.py`를 고정한다. 웹 코퍼스의 레코드를 다음처럼 생각하자. 논리 문서 `D=(key, example)`에서 `example`은 적어도 `DocumentID`, `text`, `source_uri`, `capture_id`를 가진다. 물리 입력은 `S=[S0,S1,...]`인 shard 열이고, 각 shard는 generator에 넘길 keyword 묶음 하나다. 이 구현이 checkpoint에 담는 핵심 상태는 문서 전체나 byte offset이 아니라 `C={shard_idx, shard_example_idx, type}`이다. `shard_idx`는 다음에 열 shard의 위치, `shard_example_idx`는 그 shard 안에서 이미 내보낸 레코드 수다.

원문의 핵심은 `ExamplesIterable._init_state_dict`와 `__iter__`의 15줄이다(`iterable_dataset.py:307-321`). 이해에 필요한 부분만 줄이면 다음과 같다.

```python
self._state_dict = {"shard_idx": 0, "shard_example_idx": 0, "type": self.__class__.__name__}
for gen_kwargs in islice(_split_gen_kwargs(...), shard_idx_start, None):
    for key_example in islice(generate_examples_fn(**gen_kwargs), shard_example_idx_start, None):
        self._state_dict["shard_example_idx"] += 1
        yield key_example
    self._state_dict["shard_idx"] += 1
    self._state_dict["shard_example_idx"] = 0
```

첫 줄은 상태의 **형태**를 고정한다. `DocumentID`나 원문은 들어 있지 않다. 따라서 같은 shard 목록과 같은 generator가 같은 순서를 낸다는 외부 불변식이 깨지면 커서 숫자가 같아도 다른 문서를 가리킨다. 바깥 `islice`는 완료한 shard를 건너뛰고, 안쪽 `islice`는 현재 shard에서 이미 산출한 문서를 건너뛴다. 증가가 `yield` 직전에 일어나므로 소비자가 세 번째 문서를 받은 뒤 저장한 값은 “세 번째를 다시 내라”가 아니라 “다음 문서부터 내라”는 의미다. shard가 끝난 뒤에만 `shard_idx`를 올리고 내부 좌표를 0으로 돌린다. 이 두 갱신을 하나의 `global_row`로 합치지 않은 이유는 shard별 generator를 처음부터 다시 열어도 현재 shard 안에서만 선형 skip하면 되기 때문이다. 반대로 shard 경계나 shard 내부 순서가 바뀌면 이 압축 상태는 의미를 잃는다.

여기에 `shuffle(seed, buffer_size=B)`를 붙이면 상태 도형이 달라진다. `BufferShuffledExamplesIterable.__iter__`(`:1948-1963`)는 upstream 문서를 최대 B개짜리 `mem_buffer`에 채운다. 가득 찬 뒤 난수 인덱스 i의 문서를 내보내고 그 자리를 새 문서로 바꾸며, 입력이 끝나면 남은 버퍼를 섞어 비운다. 즉 관측 상태는 `(C, RNG, M)`이어야 sample-exact하다.

여기서 C는 upstream shard cursor, RNG는 난수 상태, M은 아직 내보내지 않은 문서 배열이다. 그러나 이 wrapper의 저장 상태는 upstream C를 전달할 뿐 M의 내용을 저장하지 않는다. `load_state_dict`가 “buffer content 없이 상태를 읽으면 다시 채운다”고 경고하는 이유다. `state_dict` 문서(`:2560-2620`)도 shuffle buffer의 예제가 재개 때 유실될 수 있음을 명시한다. seed를 저장했다는 사실만으로 M을 복원할 수는 없다. C는 이미 버퍼에 들어간 문서 뒤까지 전진했기 때문이다.

작은 변형 fixture로 이 차이를 손으로 확인해 보자. 두 shard가 `[A,B,C,D]`, `[E,F,G,H]`, `B=3`이고, 첫 buffer가 `[A,B,C]`라고 하자. 난수 인덱스가 1이면 B를 내보내고 그 자리를 D로 바꾸므로 M은 `[A,D,C]`, upstream C는 첫 shard 끝을 가리킨다. 이 직후 checkpoint가 C만 저장하고 process가 죽으면 재개한 wrapper는 E부터 새 buffer를 채운다. 아직 소비하지 않은 A·C·D는 checkpoint에 없고 다시 읽을 범위에도 없다. 따라서 첫 divergence는 모델 loss가 아니라 **재개 직후 첫 SampleID 열**에서 나타난다. 문서 multiset 차이는 그 다음, packed token 경계와 optimizer trajectory 차이는 더 뒤에 나타난다.

검사는 두 갈래로 나눈다. shuffle을 끈 fixture는 `take(k)→state_dict→load→rest`와 uninterrupted 결과가 DocumentID 순서까지 같아야 한다. shuffle fixture는 같은 seed와 epoch가 같은 첫 문서를 만든다는 upstream test(`tests/test_iterable_dataset.py:2187-2212`)를 유지하되, buffer가 찬 직후 checkpoint를 추가한다.

재개 열과 기준 열을 `SampleID, DocumentID, shard_idx, shard_example_idx`로 outer join해 최초 불일치 위치를 보고한다. `buffer_size=1`은 경계 대조군, `buffer_size=3`은 유실 노출군, shard 순서를 뒤집은 경우는 cursor identity 파괴군이다. 후자의 실패를 shuffle RNG 탓으로 분류하면 안 된다. cursor가 참조하는 shard manifest 자체가 바뀐 오류다.

운영 선택은 세 가지다. sample-exact 재개가 필요하면 M과 RNG까지 checkpoint하거나, durable materialized shuffle order를 만들어 그 위치를 저장한다. 중복·누락의 제한된 허용이 가능하면 공식 예외를 recovery grade에 명시하고 재개 때 잃을 수 있는 최대 문서·토큰을 `workers × buffer_size`와 prefetch 상태를 포함해 상한 낸다. 어느 쪽이든 단순히 “dataset state를 저장했다”고 쓰지 않는다. 저장된 것은 shard cursor인지, buffer와 RNG를 포함한 consumer state인지, 이미 optimizer update에 반영된 SampleID ledger인지 각각 이름을 붙인다. 이 구분이 24장의 checkpoint 복원과 25장의 장애 분석으로 이어지는 연결점이다.

**Datatrove pipeline을 task·executor·stats state로 해부한다.**

Datatrove류 pipeline에서는 reader, extractor, filter, dedup stage, writer와 stats collector의 순서를 실제 configuration과 code revision으로 고정한다. 각 block이 기대하는 document fields와 추가·변경·삭제하는 fields를 표로 만든다. `id`, text, metadata와 dump/source relation이 stage 사이에서 안정적인지 tiny fixture로 확인한다.

local, Slurm 또는 다른 executor가 task를 partition하는 기준과 task ID를 기록한다. task별 input shard, output, stats, log와 completion marker를 manifest에 모은다. failed task만 재실행할 때 다른 random seed나 external model revision을 쓰지 않게 run root가 artifact digest를 고정한다.

MinHash signature 생성, bucket 단계와 cluster 단계가 별 job이라면 signature schema, band/hash seed, bucket partition와 component output relation을 보존한다. giant bucket cap, spill과 duplicate task output을 failure injection한다. stats JSON 합계가 실제 writer row/token count와 맞는지 독립 reduce로 검산한다.

**NeMo Curator를 classifier·GPU batch·dedup artifact로 검증한다.**

NeMo Curator류 도구에서는 document dataset representation, classifier/filter modules, fuzzy/exact dedup, PII 또는 quality model과 distributed backend의 실제 revision을 기록한다. GPU classifier의 preprocessing, tokenizer/max length, batch padding, dtype와 score output column을 고정한다. OOM adaptive batch가 row를 skip하지 않는지 DocumentID coverage로 확인한다.

모델 artifact는 이름이 아니라 digest, label mapping, threshold와 calibration report를 가진다. 여러 GPU worker가 output partition을 쓰면 rank와 input partition ownership, retry와 merge rule을 기록한다. zero-row partition과 corrupt input이 전체 job success 속에 숨지 않게 terminal reason을 집계한다.

dedup 또는 classifier 결과를 Parquet/Arrow로 교환할 때 row order를 identity로 사용하지 않는다. DocumentID join의 uniqueness, missing/extra와 schema를 검사한다. tool version 변경 전후 golden corpus를 실행해 score, decision, component와 resource usage diff를 만든다.

**Spark corpus job을 partition·shuffle·commit protocol로 읽는다.**

Spark DataFrame pipeline에서는 input format, schema, partitioning, `mapPartitions`, UDF, join, groupBy, window와 write를 logical/physical plan과 연결한다. Python UDF와 native expression은 serialization, null 처리와 optimizer visibility가 다르다. explain plan, Spark/application revision, executor image와 configuration digest를 release artifact에 둔다.

dedup join이나 LSH bucket group은 shuffle skew를 만든다. partition별 input/output row·bytes, spill, fetch wait, peak memory와 task retry를 수집한다. giant domain/hash bucket을 salting할 때 salt seed와 final merge 보존식을 기록한다. skew partition을 조용히 cap하거나 샘플링하지 않는다.

speculative execution과 task retry는 같은 output partition을 여러 attempt가 쓸 수 있다. object-store committer 또는 staging+manifest가 하나의 successful attempt만 publish하게 한다. filename에 task attempt ID가 들어가도 root manifest가 logical partition과 chosen attempt를 연결한다. partial directory listing으로 success를 판단하지 않는다.

UDF failure를 null/empty text로 바꾸면 downstream quality filter가 원인을 숨긴다. success output과 quarantine/error output을 tagged union으로 만들고 exception class, bounded message, DocumentID와 stage revision을 기록한다. error rate, source 집중도와 retry 가능성을 gate한다.

join 전후 count만 같다고 correctness가 보장되지 않는다. unique DocumentID, multiplicity histogram, unmatched left/right, key null와 sampled row digest를 확인한다. many-to-many join이 accidental row explosion을 만드는 fixture와 duplicate partition retry fixture를 둔다.

**Ray pipeline을 object ownership·backpressure·retry로 검증한다.**

Ray Data 또는 task/actor pipeline에서는 read blocks, map batches, repartition/shuffle, model actor pool, write tasks와 driver coordinator의 ownership을 그린다. block metadata는 row count, bytes, schema, source partition와 digest를 가진다. object reference가 존재한다는 사실과 durable output commit을 구분한다.

GPU model inference actor는 model digest, device, dtype, batch policy와 warmup state를 가진다. actor restart 뒤 동일 artifact와 config를 load하는지 probe한다. stateful tokenizer/cache가 이전 batch 결과를 섞지 않게 request ID와 DocumentID coverage를 확인한다. batch OOM을 반으로 나누는 fallback은 모든 row의 terminal state를 남긴다.

backpressure는 pending blocks, object-store memory, spill disk, actor queue와 output writer throughput을 함께 본다. producer가 consumer보다 빠를 때 unbounded object를 만들지 않도록 concurrency와 buffer limit를 둔다. spill path full, slow writer와 dead actor를 주입해 driver가 hang하지 않고 affected partitions를 재시도하는지 본다.

Ray task retry는 함수가 idempotent해야 한다. external counter, random generator와 output append를 task body에 숨기지 않는다. logical PartitionID와 attempt generation으로 immutable output을 쓰고 coordinator가 하나를 선택한다. driver loss 뒤 catalog checkpoint에서 미완료 partition만 복원한다.

**stage마다 보존식과 failure envelope를 선언한다.**

corpus pipeline의 각 stage는 입력 item, 출력 item, reject/quarantine/error와 expansion relation을 선언한다. one-to-one normalize는 input=success+terminal failure를, extraction처럼 한 capture가 여러 document를 만들 수 있는 stage는 parent-child edge coverage를, dedup은 input winners+losers=eligible documents를 검사한다. row count가 변할 수 있다는 말로 검산을 포기하지 않는다.

bytes와 tokens도 보존식이 있다. extraction output bytes가 payload보다 클 수 있지만 expansion ratio의 cohort bound를 둔다. normalization edit count, PII mask span, quality reject token, dedup removed token과 packed emitted token을 stage별로 정산한다. unexplained token loss를 기타 reason으로 묻지 않는다.

failure envelope는 expected exception, timeout, corrupt input, resource exhaustion와 dependency unavailable일 때 retry, quarantine 또는 abort를 정한다. retryable network failure와 deterministic parser crash를 같은 횟수로 재시도하지 않는다. circuit breaker는 source cohort와 dependency scope를 좁게 적용하고 자동 해제 조건을 가진다.

worker가 정상 종료했다는 사실만으로 stage가 성공한 것은 아니다. 모든 logical partition에 terminal record가 있고 manifest가 닫혔으며 validation report까지 존재해야 성공으로 판정한다. 따라서 `NOT_RUN`, `FAILED`, `QUARANTINED`, `PASSED`를 구분해 기록한다. validation을 건너뛴 상태를 성공으로 표시해서는 안 된다.

### 4.14.2 partition generation에서 resume한다

수일 걸리는 build는 stage 전체를 처음부터 재실행하지 않도록 logical partition checkpoint를 가진다. partition key는 input root, stage code/config/artifact digest와 partition selector로 만든다. output manifest와 stats, error artifact가 commit되면 complete다. 같은 key의 retry output digest가 다르면 nondeterminism을 조사한다.

coordinator state는 expected partitions, leases, attempts, committed outputs, failed/quarantine counts와 downstream readiness를 저장한다. coordinator restart가 worker의 늦은 commit을 중복 수락하지 않도록 lease generation과 compare-and-swap을 쓴다. worker heartbeat loss와 실제 task failure를 구분하고 speculative retry의 winner를 명시한다.

dependency model이나 tokenizer가 중간에 바뀌면 remaining partitions만 새 artifact로 처리하지 않는다. run root가 dependency digest를 고정하고 mismatch worker를 admission에서 거부한다. 긴급 patch가 필요하면 새 child run으로 만들고 이미 완료된 partition을 호환성 검증 뒤 명시적으로 import한다.

복구 시험은 driver kill, executor loss, object-store timeout, full scratch disk, corrupt input shard, partial output upload와 stats commit 전후 kill을 포함한다. 이전 complete partition은 다시 계산하지 않으며 incomplete partition은 하나의 valid output으로 닫혀야 한다. root publish 전 missing/duplicate partition을 독립 auditor가 찾는다.

**source sampling bias를 inclusion probability로 계산한다.**

crawler와 filter를 거친 corpus는 웹에서 단순 무작위로 뽑은 표본이 아니다. URL discovery, robots, fetch success, MIME support, extraction, language, quality, dedup와 policy gate를 지날 때마다 inclusion probability가 달라지기 때문이다. 측정 가능한 단계에서는 cohort별 attempted/accepted count와 token을 기록해 어느 gate가 representation을 바꾸었는지 확인한다.

link-based frontier는 많이 연결된 site와 빠른 server를 과대표집한다. crawl delay와 per-domain cap은 대형 site를 줄이지만 작은 site discovery를 보장하지 않는다. seed source, depth, domain, fetch latency와 terminal reason의 joint distribution을 본다. late snapshot cutoff가 느린 region/language를 체계적으로 제외하는지 분석한다.

dedup은 반복이 많은 source의 mass를 줄이지만 winner policy가 특정 domain을 대표로 만들 수 있다. raw capture token, normalized token, dedup unique token와 winner token을 source별로 비교한다. quality threshold도 language와 genre의 retained probability를 바꾼다. downstream mixture weight가 upstream loss를 완전히 복구할 수 없는 경우를 명시한다.

inverse-probability weighting은 inclusion probability가 알려지고 positive일 때만 가능하며 분산을 크게 키울 수 있다. 모르는 웹 전체를 정확히 복원한다는 주장을 피하고 target mixture라는 의사결정을 명시한다. sensitivity analysis로 threshold, cap와 source prior 변화가 realized mass에 미치는 범위를 계산한다.

### 4.14.3 document·token probability를 분리해 인계한다

dataset mixture에서 source weight `w_s`가 document 선택 확률인지 token budget 비율인지 구분한다. document 길이가 다르면 두 정의는 다른 realized token mass를 만든다. source s에서 document를 균등 선택하면 token inclusion은 길이에 비례하고, token-balanced sampler는 document sampling을 길이로 보정해야 한다. packing이 cross-document를 허용하는지도 영향을 준다.

handoff manifest는 source/language/quality bins, eligible documents/tokens, sampling weight, temperature exponent, cap/floor, replacement 여부, epoch 정의와 RNG contract를 가진다. planned probability 합이 1인지, empty source와 exhausted source를 어떻게 처리하는지 명시한다. fallback source로 조용히 mass를 옮기지 않는다.

realized ledger는 requested source, resolved DocumentID, valid tokens, truncated/dropped tokens, packed SampleID와 committed UpdateID를 연결한다. aborted update의 token을 consumed로 셀지 정책을 정한다. planned와 realized token mass 차이를 confidence interval 및 reason으로 보고한다.

dedup component를 sampling unit으로 삼으면 winner length와 source attribution을 어떻게 처리할지 결정한다. multiple source가 같은 component에 기여했어도 winner domain에 모든 mass를 귀속하면 source 분석이 왜곡된다. contribution relation와 training payload owner를 별도 field로 둔다.

**curriculum을 immutable phase와 transition oracle로 만든다.**

curriculum은 시간에 따라 weight를 바꾸는 함수다. phase는 시작/종료 기준을 UpdateID, consumed token 또는 metric으로 정의하고 source weight vector, data release, tokenizer/template와 sampler revision을 가진다. wall-clock 기준만 쓰면 failure와 pause가 training exposure를 바꾼다.

transition은 update boundary에서 원자적으로 일어난다. 모든 rank가 같은 CurriculumGeneration을 받아야 하며 prefetch queue의 old-phase samples를 drain할지 discard할지 명시한다. phase 전환 뒤 first batch의 SampleID, source와 weight를 event로 남긴다. rank 일부만 새 weight를 쓰는 failure를 digest agreement에서 잡는다.

adaptive curriculum은 metric 입력, smoothing, lag, bounds와 controller state를 checkpoint한다. evaluation 지연이나 missing metric에서 이전 weight 유지, pause 또는 fallback 중 하나를 정한다. noisy metric이 weight oscillation을 만들지 않게 cooldown과 maximum change를 둔다. controller 변경은 새 experiment branch다.

resume은 checkpoint의 committed UpdateID와 curriculum state, sampler RNG와 prefetch cursor를 함께 복원한다. data release가 삭제 revision으로 바뀌면 exact sample replay가 불가능할 수 있으므로 replacement relation와 recovery grade를 기록한다. 삭제된 sample을 old cache에서 되살리지 않는다.

## 4.15 blind rehearsal과 release certificate로 출판을 닫는다

마지막 절에서는 새 threshold를 더 조정하지 않는다. 앞 절의 raw snapshot, policy decision, lineage, token contribution과 resume evidence만 받아 독립 검토자가 corpus를 재구성하고 삭제를 재연한다. 성공 summary가 아니라 100개 문서의 정방향·역방향 재계산과 negative fixture가 certificate의 근거다.

### 4.15.1 tokenizer·packing 인계를 corpus contract로 검증한다

normalized document는 tokenizer 입력 경계, metadata serialization, special token policy와 allowed length를 handoff한다. tokenizer digest와 normalization이 달라지면 document token count, quality feature, near dedup와 mixture budget이 달라질 수 있다. corpus release의 estimated token과 actual tokenizer token을 stratified sample에서 비교한다.

packing은 DocumentID, byte/character/token span, SampleID, loss mask와 boundary token relation을 보존한다. 여러 document가 한 sample에 들어갈 때 PII deletion이나 contamination hit가 어느 packed shard와 token range에 있는지 역질의할 수 있어야 한다. padding과 truncated suffix도 reason을 가진다.

long document split은 overlap, sentence boundary와 minimum segment 규칙을 명시한다. overlap token이 training mass에 중복 계산되는지 ledger에 표시한다. short document concatenation이 서로 다른 privacy/license class를 섞지 않게 compatibility policy를 둔다.

golden handoff fixture는 Unicode, code, table, multilingual, masked span, empty/long document와 deletion target을 포함한다. corpus text에서 token IDs, packed spans, loader batch와 consumption event까지 digest를 비교한다. tokenizer upgrade는 child release와 새 packed generation을 요구한다.

**비용 모델을 useful token과 재처리 증폭으로 계산한다.**

stage 비용은 input bytes, documents, output valid tokens, CPU/GPU seconds, network read/write, shuffle bytes, peak memory와 storage-day로 나눈다. cost per captured byte만 보면 expensive extraction이 높은-quality token을 만드는 가치를 놓치고, cost per accepted token만 보면 reject detector 비용을 숨긴다. stage와 cohort별 단가를 함께 본다.

retry amplification은 logical input 대비 총 attempt bytes/compute 비율이다. flaky source, giant bucket와 OOM adaptive batch가 비용 tail을 만든다. p50/p95 partition duration, speculative waste와 orphan storage를 측정한다. 작은 파일 compaction은 request 비용을 줄이지만 lineage granularity와 selective deletion 비용을 키울 수 있다.

incremental rebuild는 changed partitions만 계산하지만 canonicalizer, dedup hash, policy와 tokenizer 같은 global dependency 변경은 영향 범위가 넓다. dependency graph로 invalidation closure를 계산한다. 비용 때문에 stale child를 재사용하지 않고 exact reuse condition을 manifest에 둔다.

**운영 playbook을 최초 불변식 위반에서 시작한다.**

parser yield 급락 사건에서는 source fetch success, MIME/encoding distribution, payload size, parser timeout, extracted block와 normalized token 순으로 first divergence를 찾는다. quality threshold를 낮춰 증상을 숨기지 않는다. affected capture cohort를 quarantine하고 previous parser로 paired replay한다.

dedup component 폭증에서는 canonicalizer digest, empty/error text, shingle distribution, giant LSH bucket와 graph convergence를 본다. root publish를 멈추되 unaffected completed stage를 삭제하지 않는다. representative component의 source spans와 similarity를 확인하고 hash collision, template 또는 code regression을 분류한다.

PII/secret incident에서는 새 loader access를 차단하고 keyed fingerprint로 descendants를 찾으며 logs/cache를 함께 조사한다. raw value를 incident channel에 복사하지 않는다. detector patch, tombstone propagation, physical purge와 exposure report를 독립 owner가 검증한다.

distribution drift에서는 crawl composition, fetch failure, language score, quality policy, dedup와 mixture를 차례로 decompose한다. 전체 accepted token만 맞추기 위해 다른 source를 자동 oversample하지 않는다. rare cohort loss와 downstream planned/realized mass를 함께 보고 promotion 또는 rollback을 결정한다.

### 4.15.2 blind reconstruction으로 release를 rehearsal한다

첫 reviewer는 release root와 stage manifests만 받아 임의 DocumentID의 capture request/response, extraction span, normalization edits, language/quality decision, dedup component, contamination·PII verdict, packed sample와 consumption을 재구성한다. source 파일 경로나 담당자 기억에 의존하면 실패다.

둘째 reviewer는 deletion RequestID에서 반대 방향으로 raw capture, duplicate descendants, indexes, packed shards, active releases와 consumed UpdateID를 찾는다. expected unavailable probe와 physical purge evidence를 확인한다. content rollback 뒤에도 최신 deletion root가 유지되는지 시험한다.

failure rehearsal은 WARC upload 중 kill, parser timeout storm, Spark speculative duplicate, Ray actor restart, giant LSH bucket, stale classifier artifact, partial shard commit, mixed curriculum generation과 loader cache의 tombstone 무시를 포함한다. 각 실패는 예상 gate에서 root publish를 막거나 안전한 retry로 닫혀야 한다.

최종 certificate는 crawl cutoff와 WARC closure, policy evidence, extraction/Unicode/language/quality metrics, exact/near dedup graph, contamination와 privacy, multimodal relations, distributed partition closure, dataset/deletion roots, mixture/curriculum handoff, 비용과 blind rehearsal을 잇는다. 독립 reviewer가 같은 artifact로 동일 release와 rollback 결론을 내릴 때 production corpus가 승인된다.

**domain cap과 host politeness의 통계 효과를 검산한다.**

per-host concurrency, crawl delay, daily byte/page cap와 retry backoff는 운영 안전 장치이면서 sampling operator다. host가 느리거나 rate limit을 걸면 snapshot cutoff 전에 포함될 확률이 낮아진다. host별 discovered, eligible, attempted, success, deferred와 cutoff-late URL 수를 기록하고 latency quantile과 retained token을 연결한다.

domain cap은 한 대형 source의 지배를 줄이지만 URL order가 어떤 page를 남기는지 결정한다. lexical order, discovery order, random priority와 content-aware priority는 서로 다른 표본이다. seed와 priority function revision을 보존하고 동일 frontier에서 cap을 바꾼 counterfactual로 retained path depth, MIME, language와 unique token을 비교한다.

politeness 변경은 fetch throughput만으로 승인하지 않는다. robots 준수, error/429 rate, server retry-after, host diversity, late cohort와 downstream quality를 함께 본다. global worker를 늘려도 per-host limit 때문에 useful capture가 늘지 않을 수 있다. capacity planner는 frontier의 eligible host 수와 host별 service rate를 사용한다.

**archive와 compressed container를 bounded expansion으로 처리한다.**

ZIP, tar, gzip과 nested archive는 원 container AssetID, member path, compressed/uncompressed size, CRC/hash, nesting depth와 extraction decision을 가진다. path traversal, symlink, absolute path와 duplicate member name을 차단한다. member path를 filesystem에 그대로 쓰지 않고 logical identity로만 사용한다.

zip bomb 방지는 single member, total expansion ratio, member count, depth, CPU time와 output byte budget을 함께 둔다. budget 초과는 partial success가 아니라 explicit truncated/quarantine state다. 앞부분 member만 우연히 publish되지 않게 container completion과 child closure를 검사한다.

archive 안 code, document와 media는 각 MIME adapter로 보내되 parent rights/privacy policy를 상속하고 member-specific evidence를 추가한다. password-protected archive를 빈 corpus로 처리하지 않는다. extractor upgrade 뒤 member ordering이 달라도 stable member identity와 output digest가 유지되는지 시험한다.

**adversarial web 문서를 parser·filter 경계에서 격리한다.**

adversarial fixture에는 deeply nested HTML, repeated entity expansion, huge script literal, CSS-hidden spam, Unicode confusable, bidirectional override, prompt injection 문구, malformed charset, endless redirect와 content-type mismatch를 넣는다. 각 fixture는 expected terminal stage, resource bound와 sanitized evidence를 가진다.

extractor가 security sandbox를 벗어나 network나 local file에 접근하지 못하게 process/container permission을 제한한다. external image, iframe과 stylesheet fetch는 original crawl policy 없이 parser가 임의 수행하지 않는다. crash dump와 timeout log에 raw sensitive payload가 남지 않게 bounded diagnostic을 사용한다.

quality classifier와 judge prompt에 document text를 넣을 때 document 내부 instruction이 control prompt를 바꾸지 못하도록 data channel과 output schema를 고정한다. structured output parse failure, label injection과 excessive output을 negative fixture로 둔다. model response는 evidence이지 권한 있는 policy 변경이 아니다.

**metadata schema를 최소 권한과 선택적 공개로 설계한다.**

training loader에 필요한 metadata와 audit vault에 필요한 metadata를 분리한다. loader에는 DocumentID, text, language/source bin, sampling fields와 deletion check token만 제공하고 raw URL query, IP, request header, reviewer identity와 sensitive spans는 제한한다. relation query는 opaque ID로 수행한다.

Parquet/Arrow column별 classification, encryption, retention와 access role을 schema registry에 둔다. schema projection으로 민감 column을 제외했는지 실제 file metadata와 sample read로 확인한다. partition path나 filename에도 domain, user ID와 secret fragment가 노출되지 않게 한다.

aggregate report는 small cohort count와 rare source를 통해 개인을 재식별할 수 있다. minimum group size, redaction와 approved reviewer path를 둔다. privacy 때문에 운영 보존식을 완전히 숨기지 않고 제한된 auditor가 exact count를 검증하고 공개 report는 안전한 aggregate를 사용한다.

**index를 primary data가 아닌 재생성 가능한 가속기로 다룬다.**

URL index, content-hash index, MinHash bucket, contamination search, PII fingerprint와 lineage graph index는 canonical manifests에서 재생성 가능해야 한다. index root는 source data root, builder revision, schema, partition count, hash seed, item coverage와 completion marker를 가진다. index가 존재한다고 source closure를 가정하지 않는다.

incremental index update는 add/delete 모두 처리한다. tombstone을 query-time filter로 적용하더라도 compaction 뒤 물리 제거를 검증한다. stale replica가 삭제된 DocumentID를 반환하면 resolver가 최신 deletion root에서 차단해야 한다. cache key에는 deletion generation을 포함한다.

index audit는 known present/absent probe, random source-to-index coverage, index-to-source dangling record, duplicate key와 partition checksum을 본다. approximate index는 recall sample과 failure bound를 가진다. rebuild 전후 candidate set 차이를 threshold 주변 sample에서 분석한다.

**cross-release differential을 document·component·token 세 층에서 수행한다.**

release A와 B 비교는 DocumentID retained decision, normalized text digest, score와 reason을 먼저 본다. 같은 capture가 parser 변경으로 여러 child document로 나뉘면 parent relation로 align한다. added/removed만 집계하면 ID generation 변경을 content 변화로 오인할 수 있다.

dedup 층에서는 component split/merge, winner change, member count와 similarity distribution을 비교한다. canonicalizer나 threshold 변경이 giant merge를 만들면 accepted token이 급감한다. winner만 비교하지 않고 loser source와 rights evidence 이동을 확인한다.

token 층에서는 tokenizer를 고정해 retained token mass, source/language/quality bin과 repeated n-gram을 비교한다. 새 tokenizer까지 함께 바뀌면 text delta와 tokenization delta를 별 실험으로 분해한다. training mixture가 실제로 받는 probability와 expected unique token exposure를 계산한다.

differential sample은 random뿐 아니라 high delta domain, threshold boundary, rare language, long document와 changed component를 층화한다. reviewer verdict와 원인 code/policy revision을 연결한다. 설명되지 않은 delta가 bound를 넘으면 promotion을 막는다.

**quality threshold를 ROC보다 token utility frontier로 정한다.**

document-level precision/recall은 길이와 downstream 가치 차이를 무시한다. threshold별 retained human-good tokens, retained bad tokens, unique n-gram, language/source coverage와 processing 비용을 계산한다. 매우 긴 good document 하나가 metric을 지배하지 않도록 document-weighted와 token-weighted 결과를 함께 본다.

classifier score calibration이 cohort별로 다르면 동일 threshold의 오류율이 다르다. language/genre별 reliability curve와 minimum review sample을 둔다. cohort threshold는 utility frontier를 개선할 수 있지만 작은 표본에 과적합되지 않게 holdout과 expiry를 사용한다.

downstream proxy는 작은 model loss, retrieval quality 또는 human preference가 될 수 있지만 corpus filter와 같은 scorer로 평가하지 않는다. proxy 개선이 benchmark contamination이나 style shortcut 때문인지 clean slice와 counterfactual로 확인한다. threshold 선택의 불확실성을 release note에 남긴다.

**threshold·삭제·causal audit의 독립 승인 조건.**

**near-dedup threshold를 recall·removal risk와 함께 선택한다.**

threshold가 낮을수록 candidate와 제거율은 늘지만 관련 문서를 잘못 합칠 위험이 커진다. labeled pair를 exact duplicate, formatting variant, updated article, quotation, same template/different content와 unrelated로 나누고 threshold별 precision/recall을 본다. component-level over-removal token도 계산한다.

LSH candidate recall과 verifier precision을 분리한다. verifier threshold가 좋아도 candidate generator가 pair를 놓칠 수 있다. similarity grid와 real labeled pairs에서 candidate recall을 측정하고 giant bucket cap의 missed pairs를 추정한다. seed 변경 variance도 본다.

winner policy와 threshold는 상호작용한다. 업데이트된 문서를 old version과 합친 뒤 freshness winner를 고르면 history가 사라지고, authority winner는 syndication source를 제거할 수 있다. training 목표에 맞는 representative와 member metadata 보존을 구분한다.

**deletion 재난 복구를 backup·replica·key 폐기까지 시험한다.**

backup은 content generation뿐 아니라 deletion log와 current tombstone root를 함께 복원해야 한다. 오래된 backup을 복구한 뒤 최신 deletion journal을 재적용하지 않으면 삭제 content가 되살아난다. restore 순서, journal high-water mark와 resolver admission gate를 runbook에 둔다.

cross-region replica가 content는 최신인데 tombstone은 지연될 수 있다. replication readiness는 두 root의 closure로 판단한다. failover rehearsal에서 deleted DocumentID probe가 모든 loader와 index에서 실패하는지 확인한다. replica lag SLO는 content와 deletion에 별도로 둔다.

encryption key destruction을 physical deletion 수단으로 쓸 때 key scope가 unrelated artifact를 포함하지 않는지 확인한다. shared key를 폐기해 전체 release를 잃거나, copy가 다른 key로 남아 있는 상태를 피한다. key inventory와 artifact relation, backup key escrow policy를 audit한다.

**함수-level 구현 좌표를 golden corpus로 고정한다.**

각 도구에서 reader 함수, extraction mapper, normalizer, language/quality inference, exact/MinHash writer, cluster reducer, PII redactor, shard writer와 manifest committer의 repository revision과 body fingerprint를 기록한다. public API 이름만 같아도 내부 default와 failure handling이 바뀔 수 있다.

golden corpus는 각 stage마다 expected DocumentID, text digest, spans, score, decision, signature, component, mask와 output shard를 가진다. 단일 end-to-end digest만 비교하지 않고 first divergent stage를 찾는다. CPU/GPU, single/distributed와 supported tool version 조합을 matrix로 실행한다.

implementation upgrade는 source diff에서 affected contract를 먼저 예측하고 golden differential로 확인한다. tokenizer/model artifact download가 floating revision을 가리키지 않게 immutable digest를 사용한다. environment image, native library와 locale도 capture한다.

**corpus 수학을 conservation·estimation·uncertainty로 정산한다.**

각 stage s의 retained token ratio를 `r_s=N_s/N_{s-1}`로 두더라도 전체 ratio의 원인을 독립으로 가정하지 않는다. 동일 document가 여러 gate에 실패하고 upstream filtering이 downstream 분포를 바꾸기 때문이다. ordered pipeline의 observed conditional ratio와 counterfactual paired replay를 구분한다.

MinHash Jaccard estimate, language/quality posterior, PII probability와 mixture realized mass에는 uncertainty가 있다. point estimate만으로 threshold를 승인하지 않고 bootstrap 또는 analytic interval, labeled sample design과 effective sample size를 기록한다. component와 domain correlation을 무시한 row bootstrap을 피한다.

sampling audit에서 planned probability p와 n번 draw의 observed count는 binomial 기준을 참고할 수 있지만 replacement, packing, exhaustion와 distributed prefetch가 독립성을 깨뜨린다. simulator와 event ledger로 expected range를 만들고 persistent deviation을 sampler bug 또는 availability 변화로 분류한다.

### 4.15.3 100개 문서를 수기로 재구성해 certificate를 검산한다

전체 자동 검사를 통과한 뒤 source, language, MIME, quality decile, dedup component size, safety decision와 multimodal 여부를 층화해 100개 DocumentID를 고른다. reviewer는 WARC byte에서 extraction, normalization, decisions, duplicate relation, packed span까지 재구성한다. 편한 accepted HTML만 뽑지 않는다.

별도 negative 100개는 robots/license block, corrupt extraction, low-confidence language, quality reject, PII/secret mask, contamination, deleted winner, giant bucket와 failed partition을 포함한다. terminal reason이 원 evidence와 맞고 loader에서 접근되지 않는지 확인한다. `NOT_RUN`을 reject 성공으로 세지 않는다.

검산표는 expected/actual, first divergence, owner, correction generation과 regression fixture를 가진다. 오류 하나를 수동 예외로 덮지 않고 같은 failure class의 affected set을 query한다. 수정 뒤 positive와 negative sample을 새 seed로 다시 뽑는다.

최종 승인자는 네 가지를 확인한다. capture와 use가 정책 증거로 닫혔는가, accepted token이 transform과 dedup을 거쳐 재현되는가, 삭제·오염·privacy target이 모든 descendant에서 차단되는가, mixture consumer가 planned/realized mass와 release identity를 보고하는가. 네 답이 독립 artifact로 일치해야 corpus release를 승인한다.

**contamination causal audit를 matched rebuild로 수행한다.**

오염 hit를 제거한 뒤 benchmark score가 낮아졌다는 사실만으로 원 hit가 성능을 만들었다고 단정할 수 없다. 제거 과정에서 source, language, topic, length와 quality 분포가 함께 바뀌기 때문이다. hit document를 제거한 treatment corpus와 동일 source·capture cohort·length·quality·dedup component size를 가진 non-hit document를 같은 token mass만큼 제거한 matched-control corpus를 만든다.

baseline, treatment와 matched-control은 같은 parent capture, tokenizer, mixture, seed family와 training budget을 사용한다. stochastic training variance를 보기 위해 여러 seed를 실행하고 benchmark item별 delta, clean capability, held-out source와 loss curve를 비교한다. treatment만 특정 item에서 변하고 control은 변하지 않는지 보되 confidence interval과 multiple comparison을 기록한다.

exact hit, prompt-only, answer-only, rationale, paraphrase와 source passage를 별 cohort로 분리한다. 한 번에 모두 제거하면 어떤 leakage type이 영향을 만들었는지 알 수 없다. 다만 cohort 간 overlap이 있으면 factorial effect를 단순 합산하지 않고 DocumentID와 BenchmarkItemID graph에서 joint removal set을 만든다.

training 전 영향 예측도 남긴다. matched span token 수, repeated exposure, curriculum phase, first/last consumed UpdateID와 benchmark item별 proximity를 계산한다. training 뒤 결과와 비교해 detector priority를 조정한다. 이미 알려진 benchmark만 최적화하지 않도록 새 clean canary와 time-split benchmark를 둔다.

causal report는 오염이 없었다는 이진 결론 대신 detector coverage, uncertain hits, consumed exposure, treatment-control delta와 residual alternative explanation을 제시한다. score가 유지되어도 removal과 gate가 correctness를 위해 필요한 사실은 변하지 않는다. 성능 영향이 작다는 이유로 policy 위반 content를 되살리지 않는다.

**Spark·Ray 교차 실행에서 deterministic partition 결과를 검증한다.**

같은 logical stage를 Spark와 Ray로 실행할 수 있다면 row order가 아니라 DocumentID별 output digest와 terminal state를 비교한다. partition 수, batch boundary와 task scheduling은 달라도 pure transform 결과는 같아야 한다. floating reduction, GPU classifier와 approximate dedup처럼 허용 차이가 있는 stage는 tolerance와 expected component stability를 사전에 둔다.

cross-engine golden run은 empty partition, skewed giant key, corrupt row, retryable timeout, worker kill와 speculative duplicate를 포함한다. 두 engine 모두 input coverage, quarantine count와 one-commit-per-partition 불변식을 만족해야 한다. error class 이름이 달라도 canonical terminal reason으로 mapping하고 원 exception을 보존한다.

shuffle 뒤 output file 수나 ordering 차이는 정상일 수 있지만 release root가 content-equivalent한지 canonical manifest를 만든다. logical partition별 sorted DocumentID/digest accumulator, schema와 count를 비교한다. accelerator library나 locale가 text 결과를 바꾸면 environment digest에서 드러나야 한다.

실행 engine을 바꾸면 비용뿐 아니라 실패 조건도 달라진다. engine마다 retry semantics, object commit, actor/UDF state와 backpressure가 다르기 때문이다. 따라서 throughput과 useful tokens per compute만 비교하지 말고 retry amplification, tail partition time와 correctness fixture를 함께 승인한다. 일부 stage만 새 engine으로 옮겼다면 경계의 interchange schema와 lineage relation을 end-to-end로 시험해야 한다.

**mixture 소비 오차를 availability와 sampler 오류로 분해한다.**

planned source weight와 realized mass가 다를 때 먼저 eligible token inventory, tombstone, loader decode failure, exhausted shard와 prefetch cancellation을 측정한다. availability가 줄어든 것과 sampler가 잘못된 확률을 적용한 것을 분리하지 않으면 weight를 조정해 소프트웨어 오류를 숨길 수 있다.

sampler event는 requested bin, random draw 또는 counter, selected shard/DocumentID, eligibility result, fallback, valid token와 UpdateID를 가진다. rank별 event를 전역 ledger에 합쳐 duplicate SampleID, skipped counter와 divergent mixture generation을 찾는다. logging sampling을 쓰면 inclusion probability를 기록해 realized estimate를 보정한다.

finite dataset을 replacement 없이 소비하면 phase 후반 확률이 달라진다. source exhaustion에서 renormalize, stop 또는 replacement 전환 중 무엇을 하는지 contract로 둔다. renormalization은 남은 source mass를 바꾸므로 event와 dashboard에 표시한다. curriculum boundary와 deletion이 동시에 일어나는 fixture를 둔다.

통계 검산은 source count뿐 아니라 valid tokens, unique DocumentID, dedup component와 committed optimizer exposure를 본다. packing efficiency 차이가 sample count를 token mass로 바꾼다. aborted step과 gradient accumulation 중 replay를 일관되게 처리한다. deviation alert는 minimum expected count와 correlation-aware window를 사용한다.

**incident rebuild를 patch가 아닌 새 generation으로 실행한다.**

incident가 확인되면 affected release root를 freeze하고 새 training run의 admission을 막는다. 이미 실행 중인 consumer는 위험과 policy에 따라 stop, drain 또는 continue-with-blocklist를 명시적으로 선택한다. 담당자가 임시 파일 목록을 loader에 복사해 조용히 제외하지 않는다. containment selector와 expiry를 artifact로 만든다.

영향 graph는 trigger CaptureID/DocumentID에서 normalized children, decisions, dedup components, winners, indexes, shards, mixtures와 consumed UpdateID로 확장한다. 반대로 공통 parser/model/policy revision에서 처리된 모든 document를 찾는다. direct hit와 possibly affected를 구분해 rebuild scope와 uncertainty를 정한다.

수정은 code/config/model/policy의 새 revision과 regression fixture를 만든다. complete parent stage 중 reuse condition을 만족하는 partition만 import하고 affected stage부터 child generation으로 재실행한다. global canonicalization이나 dedup 변경은 local patch로 끝내지 않는다. imported output과 recomputed output의 root를 manifest에 표시한다.

rebuild 중에는 old/new paired metrics, stage conservation, error reason, component 변화와 deletion closure를 streaming으로 본다. 목표 count를 맞추기 위해 실패 partition을 제외하고 root를 닫지 않는다. completion 뒤 independent validator가 expected partition set과 dependency digest를 재계산한다.

promotion 전에 old release와 child release를 canary loader에서 동시에 읽어 affected fixture, random clean sample와 downstream packing을 비교한다. incident target은 접근 불가능하고 unrelated corpus의 예상 밖 delta는 bound 안이어야 한다. promotion 후 cache invalidation과 active consumer switch를 확인하며 rollback해도 최신 tombstone은 유지한다.

**cold rebuild와 cold delete로 release 의미를 검증한다.**

cold rebuild는 기존 worker cache, local model copy, intermediate index와 담당자 shell history가 없는 새 환경에서 시작한다. release manifest가 가리키는 immutable code image, model/tokenizer artifact, WARC roots와 policy/deletion roots만 사용한다. 필요한 dependency가 floating URL이거나 사라진 credential에 의존하면 재현 실패다.

작은 stratified partition set을 capture reader부터 extraction, normalization, classification, dedup, safety, packing과 mixture handoff까지 실행한다. original output과 DocumentID별 digest, decision, component, token span을 비교한다. approximate 또는 nondeterministic stage는 사전 tolerance, seed와 first divergence report를 사용한다.

cold delete는 rebuild 환경에 새 RequestID를 주고 target discovery, tombstone commit, index/cache 차단, dedup winner 재선정, shard rebuild와 loader probe를 문서 없이 수행한다. backup replica를 복원한 뒤에도 deletion journal이 재적용되는지 확인한다. 삭제 SLA는 request 접수부터 모든 required layer 검증까지 phase별로 잰다.

마지막 failure set은 coordinator kill, stale classifier cache, mixed schema, giant LSH bucket, corrupt WARC index, Spark duplicate commit, Ray actor OOM, mixture generation mismatch와 deletion replica lag다. 각 failure가 expected gate에서 중단되며 complete parent와 최신 tombstone을 손상하지 않아야 한다. recovery 뒤 새 root를 다시 cold read한다.

인수 bundle은 명령 자체보다 input/output roots, environment digest, partition plan, stage metrics, failure events, reviewer queries와 final decision을 보존한다. 별도 운영자가 bundle로 같은 재사용 범위, rebuild 결과와 삭제 결론을 얻으면 4장은 수집량 중심의 설명을 넘어 반복 가능한 corpus 제조·감사·복구 계약으로 완성된다.

**classifier calibration drift를 label shift와 model shift로 나눈다.**

새 crawl에서 quality, language, toxicity와 PII score 분포가 바뀌면 model이 나빠졌다고 즉시 결론내리지 않는다. source composition, document length, MIME, parser yield와 language mix가 달라진 label/input shift와 동일 paired document의 score가 달라진 model/preprocessing shift를 분리한다. old/new model을 old/new sample에 교차 적용하는 네 칸 실험을 만든다.

calibration audit sample은 score bin, source, language, genre와 high-risk category를 층화한다. reviewer는 model score를 보지 않은 채 label과 uncertainty를 기록한다. bin별 observed positive rate, Brier score, expected calibration error와 precision/recall을 계산하되 component/domain correlation을 반영한 bootstrap interval을 사용한다.

threshold를 재조정하면 기존 score cache만으로 decision을 다시 만들 수 있지만 preprocessing이나 model artifact가 바뀌면 score부터 재계산한다. threshold revision, calibration set lineage와 effective date를 manifest에 둔다. rare cohort에서 표본이 부족하면 broad threshold를 강제하지 않고 review/quarantine와 uncertainty bound를 사용한다.

drift alarm은 aggregate score mean보다 threshold 주변 mass, cohort false-negative risk, accepted token delta와 downstream mixture 변화를 본다. automatic pause는 affected classifier와 cohort로 범위를 좁히되 shared preprocessing drift이면 전체 publish를 막는다. recalibration 뒤 paired clean/negative fixture와 full canary를 다시 실행한다.

**release 승인 증거를 claim별 최소 artifact로 정리한다.**

수집 완전성 주장은 frontier cutoff, WARC/index closure와 terminal reason 보존식으로 증명한다. 정책 준수 주장은 robots snapshot, license/use decision와 privacy review relation으로 증명한다. text 품질 주장은 extraction span, Unicode edits, language/quality calibration과 stratified human audit로 증명한다. duplicate·오염 주장은 candidate recall, verified edge, component와 benchmark span evidence로 증명한다.

분산 정확성 주장은 expected logical partitions, chosen attempts, input/output conservation, cross-engine golden result와 cold rebuild로 증명한다. 삭제 주장은 current tombstone root, descendant query, loader/index/backup probe와 SLA phase로 증명한다. mixture 인계 주장은 planned weights, actual eligible inventory, sampler events, packed spans와 committed UpdateID mass로 증명한다.

하나의 dashboard screenshot이나 총 document 수로 여러 주장을 대신하지 않는다. 각 artifact는 generation, producer revision, creation time, digest, retention와 reader schema를 가진다. evidence가 없으면 `NOT_RUN` 또는 unresolved로 남기며 담당자의 구두 설명을 PASS로 변환하지 않는다.

최종 reviewer는 임의 claim에서 원 artifact까지, 임의 DocumentID에서 소비 또는 삭제 결과까지 양방향으로 이동한다. query 결과가 다른 root나 stale index를 섞으면 release를 거부한다. 이 최소 증거표가 모두 닫혀야 코퍼스의 규모와 품질에 관한 설명이 실제 release를 신뢰할 근거가 된다.

## 4.16 DCLM quality filter 한 줄이 학습 모집단을 바꾸는 순간

DCLM의 `quality_filter`는 복잡한 모델을 실행하지 않는다. 고정 소스의 핵심은 `quality_score = page.get(key, missing_score)`와 `quality_score >= threshold` 또는 `<= threshold`라는 반환 분기다. 입력은 metadata를 가진 document dictionary 하나이고 출력은 `[page]` 또는 빈 목록이다. 이 작은 shape 변화가 downstream에서는 “문서 한 건 유지”와 “표본 공간에서 제거”라는 되돌리기 어려운 상태 전이가 된다.

`lower_better`가 필요한 까닭은 perplexity처럼 작은 값이 좋은 score와 classifier probability처럼 큰 값이 좋은 score를 같은 함수가 받기 때문이다. 더 위험한 것은 score가 없을 때다. `key_must_exist=False`이면 큰 쪽이 좋은 score에는 `-inf`, 작은 쪽이 좋은 score에는 `+inf`를 넣어 누락 문서를 확실히 버린다. 이 sentinel을 반대로 두면 누락률이 높은 source가 오히려 전량 통과한다. 함수는 정상 종료하므로 총 문서 수만 보면 원인을 놓친다.

고정 fixture로 score `0.79`, `0.80`, `0.81`, missing 네 문서를 threshold `0.80`에 넣어 보자. 최초 불일치는 model loss가 아니라 filter 직후 survivor ID 집합이어야 한다. 비교 연산을 `>`로 바꾸면 경계 문서에서, `lower_better`를 뒤집으면 첫 문서에서, missing sentinel을 바꾸면 missing 행에서 갈라진다. 이어 source·language별 survivor mass와 token mass를 따로 계산한다. 문서 수가 같아도 긴 문서가 탈락하면 학습 기여도는 달라진다.

실무 진단은 score schema와 producer revision, 결측률, 방향, threshold, 경계값 tie, survivor DocumentID, downstream token 수 순서로 내려간다. 수정 뒤에는 4.9의 split·dedup 정책과 6장의 realized mixture까지 추적한다. filter의 boolean parity만 맞추고 최종 token mass가 달라진다면 다음 디깅 지점은 normalization·dedup·packer이지 classifier가 아니다.

### 4.16.1 FineWeb·RefinedWeb·C4를 이름이 아니라 연산자 열로 비교한다

세 이름을 “깨끗한 웹 데이터”라는 한 칸에 넣으면 재현에 실패한다. 비교 단위는 `crawl snapshot → URL/robots 정책 → WARC 추출기 → 문서·행 정규화 → 언어 점수 → 휴리스틱/모델 품질 점수 → 유해·PII 정책 → 문서 내부·문서 간 dedup → shard`라는 순서 있는 연산자 열이다. 같은 Common Crawl을 출발점으로 삼아도 extractor가 보존한 목록·코드·문단 경계가 다르면 뒤의 길이와 품질 feature가 이미 다른 모집단을 본다. filter 집합이 같아도 dedup을 먼저 하느냐 나중에 하느냐에 따라 component winner와 source별 잔존 질량이 달라진다.

DCLM의 고정 테스트가 특히 유용한 까닭은 이 차이를 작은 경계값으로 드러내기 때문이다. fastText 계열 확률은 큰 값이 좋고 KenLM perplexity는 작은 값이 좋다. 테스트는 threshold와 같은 값의 포함, 그 위·아래, metadata key 부재를 따로 고정한다. 따라서 recipe manifest에는 모델 이름뿐 아니라 score key, 방향, 비교 연산자, threshold, 결측 sentinel과 producer revision이 들어가야 한다. RefinedWeb banlist를 가져왔다는 사실도 URL 정규화, substring/exact-domain 모드와 적용 시점을 함께 기록하지 않으면 같은 정책이 아니다.

FineWeb의 DataTrove pipeline에서는 각 stage의 `total`, `forwarded`, `dropped_reason` 통계가 pipeline 병목과 탈락 원인을 보여 준다. 그러나 이 통계만으로 학습 기여도를 알 수는 없다. 문서 하나가 수십 token일 수도 수만 token일 수도 있고, tokenizer·truncation·packing·label mask 뒤 유효 정답 수가 다시 바뀐다. corpus release 표에는 stage별 document conservation과 함께 `survivor_bytes`, `emitted_tokens`, `valid_target_tokens`를 source·language·policy revision별로 보존한다. 이 계측이 없으면 FineWeb·RefinedWeb·C4의 총 token 수 비교는 어느 정책이 어느 모집단을 만들었는지 설명하지 못한다.

검증 fixture는 같은 raw document 묶음을 세 recipe adapter에 통과시켜 최초로 달라진 `(DocumentID, operator, decision, reason, score)`를 낸다. 최종 survivor만 비교하면 extractor 차이를 quality filter 차이로 오진한다. 경계 점수, metadata 누락, URL Unicode/port/subdomain, 반복 문단, boilerplate와 서로 다른 길이의 duplicate component를 반드시 포함한다. 결과 보고서는 어느 corpus가 보편적으로 우월하다고 선언하는 대신, 주어진 모델·token budget·평가 slice에서 어떤 연산자가 어떤 질량과 위험을 바꿨는지를 답해야 한다.
## 4.17 `license` 한 칸으로 권리 상태를 표현할 수 없는 이유

데이터셋 카드에 라이선스 이름이 적혀 있으면 거버넌스가 끝난 것처럼 보이기 쉽다. 그러나 Hugging Face Datasets의 `DatasetInfo`를 코드까지 내려가 읽으면 권리 관련 계약은 자유 형식 `license: str` 하나다. `write_to_directory`는 이 문자열을 `LICENSE` 파일로 쓰고, `from_merge`는 여러 데이터셋의 서로 다른 문자열을 빈 줄로 이어 붙인다. 이는 메타데이터를 보존하는 유용한 동작이지만, 어느 행이 어느 원천의 조건을 물려받았는지 판정하거나 상충하는 조건을 해결하는 집행기는 아니다.

실무에서는 적어도 네 상태를 분리해야 한다. `LicenseAssertion`은 카드 작성자가 무엇이라고 표기했는지, `CollectionDecision`은 수집 당시 URL과 robots 응답·정책 revision으로 왜 가져왔는지, `ConsentBasis`는 사람 데이터의 동의 범위와 철회 조건이 무엇인지, `DeletionEvent`는 어떤 안정 ID가 어느 파생물에서 언제 제외됐는지를 나타낸다. 이들을 문자열 하나로 합치면 “표기는 남았지만 집행 상태는 모르는” 상황과 “검증된 허가”를 구별할 수 없다.

robots도 현재 파일만 저장하면 부족하다. 수집 시각, 요청 URL, redirect chain, user-agent, 응답 코드, 원문 bytes의 digest와 parser revision을 함께 보존해야 당시 판정을 재생할 수 있다. `robots.txt` 허용은 저작권·개인정보·계약상 동의를 자동으로 부여하지 않으며, 반대로 접근 실패를 허용으로 간주한 fallback도 명시해야 한다. 그래서 수집기는 `allowed/denied/unknown/error`를 구분하고 `unknown`을 조용히 허용으로 바꾸지 않아야 한다.

삭제 요청은 URL blocklist 한 줄을 추가하는 것으로 닫히지 않는다. `SourceID → normalized document → dedup survivor → token offsets → packed sequence → shard generation → consumed RunID → checkpoint`의 역색인을 따라 영향 범위를 계산해야 한다. 새 corpus release에서 제외하는 것과 이미 학습된 checkpoint의 영향을 제거하는 것은 별도 명제다. 후자는 unlearning이나 재학습 증거가 없으면 `NotRemovedFromCheckpoint`로 남겨야 한다.

검증 fixture는 단순하다. 같은 원문이 URL 두 개와 redirect로 들어오고 dedup 뒤 하나만 살아남게 만든다. 그중 한 source에 tombstone을 넣은 뒤 survivor 선택, token shard, mixture manifest와 다음 run이 모두 갱신되는지 확인한다. 이미 만들어진 checkpoint에는 삭제 완료를 선언하지 못해야 한다. 이 음성 fixture가 통과하지 않으면 시스템은 삭제 요청을 받는 UI만 있을 뿐 삭제 lineage는 없다.

## Filter decision을 UpdateID까지 잇는 원장

필터가 문서를 버렸다는 기록만으로는 학습 데이터 계보가 닫히지 않는다. source row UUID에서 normalization hash, dedup cluster, quality classifier revision·score, threshold와 decision reason을 지나 tokenizer offset, PackID·segment, supervised mask와 실제 UpdateID까지 같은 행을 따라갈 수 있어야 한다. 파일명이나 row number만 join key로 쓰면 재수집·reshard에서 다른 문서가 같은 위치를 차지한다.

DCLM의 직접 테스트는 세 seed에서 probability 0.1 random filter를 10,000회 적용해 survivor가 900~1,100인지, 범위 밖 probability를 거부하는지 검사한다. 이것은 분포의 거친 sanity check다. 특정 DocumentUUID가 중단 전후 같은 decision을 받는지, classifier input과 score가 같은지는 증명하지 않는다. 재개 fixture는 RNG state뿐 아니라 UUID·transform revision·decision tuple을 비교해야 한다.

dedup과 quality classifier는 leakage graph로 함께 본다. 같은 duplicate cluster의 한 문서가 classifier train에, 다른 문서가 selection benchmark나 최종 holdout에 있으면 문자열이 달라도 누출이다. classifier가 만든 score를 다시 classifier 성능 평가에 쓰거나 proxy selection과 final evaluation이 같은 관측 집합을 공유해도 adaptive leakage가 생긴다. cluster ID, split role과 관측 횟수를 보존하고 교차 edge를 release gate에서 거부한다.

## 4.18 opt-out을 packed token과 소비한 update까지 역추적한다

삭제 파이프라인의 출발점은 URL blocklist가 아니라 인증된 `RequestID`다. 요청자 identity와 권한 범위, 대상 URL·snapshot digest, 수집 당시 robots·consent·license revision을 고정한다. 현재 robots.txt로 과거 판단을 덮어쓰지 않는다. robots는 접근 정책의 관측이지 consent·저작권·계약의 대체물이 아니다.

그다음은 원문 한 건이 아니라 파생물 폐쇄를 찾는다. canonical URL, alias와 redirect를 normalized document에 연결하고 exact·near dedup cluster의 모든 member와 survivor를 찾는다. 이어 token/index pair, packed segment offset, shard generation, mixture manifest와 실제 소비한 RunID·UpdateID를 역검색한다.

shard는 제자리 수정하지 않는다. tombstone set과 parent manifest digest로 새 immutable generation을 만들고 payload·index·통계가 준비된 뒤 complete marker를 발행한다. catalog, cache, streaming worker와 resume cursor가 구 generation을 읽지 않는지도 확인한다.

새 corpus에서 제외한 사실은 checkpoint나 adapter에서 영향이 제거됐다는 뜻이 아니다. 재학습이나 사전 정의된 unlearning·회귀 oracle이 없다면 `NotRemovedFromCheckpoint`다. 공개 HF Datasets·DataTrove·Dolma 구현에는 이 종단 상태 기계를 직접 고정한 시험이 없으므로 redirect·dedup·packing 합성 fixture를 deployment gate에서 실행해야 한다.

## 4.19 코드·수학·다국어 데이터는 파일 형식이 아니라 경계 보존 문제다

코드, 수학, 다국어 코퍼스를 각각 “전문 데이터”라는 비율로만 적으면 실제 학습 입력을 설명하지 못한다. 코드에서는 들여쓰기·줄바꿈·파일 의존성과 라이선스가, 수학에서는 문제와 풀이의 경계·수식 정규화·정답 검산이, 다국어에서는 문자 정규화·언어 판별 오차·언어별 token 비용이 모집단을 바꾼다. 세 영역의 공통 질문은 원문에서 loss를 받는 token까지 의미 있는 경계가 얼마나 살아남았느냐이다.

DeepSeek-Coder의 고정 파인튜닝 예제를 따라가 보자. `train_tokenize_function`은 `instruction`을 prompt로 만들고 `output` 뒤에 EOT를 붙인다. `preprocess`는 `source+target` 전체와 source만을 따로 토큰화하고, source token 수만큼 label을 `-100`으로 바꾼다. 의도는 분명하다. 모델이 사용자 지시를 복사하도록 보상하지 않고 정답에만 loss를 주려는 것이다. 하지만 구현에는 숨은 전제가 있다. source를 단독 토큰화한 결과가 source와 target을 붙여 토큰화한 결과의 정확한 prefix여야 한다.

### 4.19.1 prompt와 target 사이의 BPE merge가 label을 한 칸 옮길 수 있다

subword tokenizer는 문자열 연결에 대해 일반적으로 분배법칙을 만족하지 않는다. `T(s+t)`가 언제나 `T(s) || T(t)`인 것은 아니다. prompt 끝 공백, code fence, Unicode 결합문자나 수식 기호가 target 첫 문자와 새 merge를 만들면 독립 source 길이로 자른 mask가 실제 경계를 한 token 침범하거나 덜 가릴 수 있다. 이 오류는 tensor shape를 깨뜨리지 않는다. loss도 정상적으로 감소하지만 첫 정답 token이 사라지거나 prompt 말미가 학습되어 목적함수가 조용히 바뀐다.

검증은 문자 위치가 아니라 token offset으로 한다. tokenizer가 offset mapping을 제공하면 결합 문자열 한 번에서 prompt의 마지막 character boundary를 덮는 token을 찾아 정책을 명시한다. 경계 token 전체를 prompt로 가릴지, separator를 강제해 merge를 막을지 결정한다. slow tokenizer라 offset이 없다면 sentinel이 포함된 synthetic pair를 만들어 mask 전이 주변 decode를 기록한다. 코드에서는 fence·들여쓰기·comment prefix를, 수학에서는 LaTeX delimiter·Unicode minus를, 다국어에서는 NFC/NFD와 결합문자를 경계값으로 넣는다.

### 4.19.2 padding과 truncation은 attention mask와 loss 분모를 함께 바꾼다

같은 예제의 collator는 input을 pad token으로 채우고 label은 `-100`으로 채운 뒤 `input_ids != pad_token_id`로 attention mask를 만든다. 보통은 padding이 attention과 loss에서 함께 빠지는 올바른 계약이다. 그러나 pad token을 EOS와 공유하는 tokenizer에서는 문장 안의 진짜 EOS도 값만 보면 padding처럼 보인다. “pad를 EOS로 지정해도 된다”는 설정 편의와 “token id 비교로 mask를 만든다”는 구현은 반드시 함께 검토한다.

truncation도 source와 전체 문자열에 각각 적용된다. prompt만으로 최대 길이를 채우면 target valid token 수가 0이 될 수 있다. 이 행을 batch에 남기면 sample 수는 늘지만 gradient 기여는 없다. target 뒤가 잘리면 EOT가 사라져 종료 행동을 학습하지 못한다. 데이터 보고서에는 row 수와 input token 수뿐 아니라 `valid_target_tokens`, zero-target row 수, EOT survival, 언어·domain별 truncation 질량을 적는다. 분산 학습에서는 이 수가 전역 loss denominator와 gradient scale로 이어진다.

### 4.19.3 병렬 전처리 cache는 속도 기능이 아니라 데이터 revision이다

공개 recipe는 JSON train split에 batch 3,000, process 32로 map하고 cache reuse를 켠다. tokenizer revision·prompt builder·normalization code·입력 digest가 cache identity에 정확히 들어가지 않으면 코드를 고쳐도 오래된 token과 label을 읽을 수 있다. 로그에 새 옵션이 찍힌다는 사실은 실제 cached payload가 새 규칙으로 만들어졌다는 증거가 아니다.

cold/warm 두 경로를 같은 release gate에 둔다. cache가 없는 상태와 기존 cache가 있는 상태에서 `SampleID → input_ids digest → label-mask digest → valid-target count`를 비교한다. worker 수 1과 32에서도 결과와 행 순서가 같아야 한다. 다르면 먼저 최초 SampleID와 변환 stage를 찾고 곧바로 학습 loss 차이로 뛰어가지 않는다. cache fingerprint와 dataset manifest는 checkpoint가 참조하는 불변 artifact여야 한다.

**공통 원장을 쓰되 domain별 품질 oracle은 분리한다.**

공통 원장은 `SourceID → normalization → unit boundary → dedup cluster → tokenizer revision → PackID/offset → label mask → UpdateID`다. 하지만 품질 oracle은 다르다. 코드는 parse와 dependency closure, compile/test 실행, secret·license scan을 본다. 수학은 symbolic equivalence, numerical substitution, proof step validity와 answer leakage를 본다. 다국어는 language identification confusion, 번역 중복, script·locale coverage와 언어별 token inflation을 본다. 한 domain의 점수를 다른 domain에 그대로 적용하지 않는다.

작은 인수 fixture에는 Python 들여쓰기와 cross-file import, LaTeX와 Unicode가 섞인 동일 수식, 한국어 NFC/NFD와 code-switch 문장을 넣는다. 각 항목에 prompt-target 경계 merge, pad==eos, 최대 길이 바로 전후, 빈 target을 교차한다. 원문 의미가 같다는 gold relation과 token/label이 같아야 한다는 구현 invariant는 구분한다. tokenizer가 다르면 token 배열은 달라도 되지만 어느 의미 단위가 loss에서 빠졌는지와 valid-target 질량은 설명할 수 있어야 한다.

[설정한 mixture가 실제 손실 질량이 되기까지](../labs/06-mixture-realized-mass-lab.md)는 중복 행과 평가 split 누수 행을 서로 다른 이유로 허용 universe에서 제외한 뒤, 그 선택이 source별 문서·토큰·손실 분모에 미치는 영향을 고정 수치로 되짚는다. dedup과 오염 검출 결과를 sampler 뒤의 retry 통계로만 남기지 말아야 하는 이유를 가장 작은 반사실로 확인할 수 있다.

실전 디깅 순서는 raw SampleID와 bytes, normalization과 unit boundary, 결합 tokenization과 mask transition, padding·truncation 뒤 denominator, packing과 cache 순이다. 마지막에야 optimizer step과 benchmark를 비교한다. 그러면 “코드 데이터 비율을 늘렸더니 수학 성능이 떨어졌다”는 현상을 domain 이름으로 설명하는 대신 실제로 어느 모집단과 어느 loss 질량이 바뀌었는지 말할 수 있다.

## 4.20 중복 제거를 `hash 한 번`이 아니라 검출 그래프로 읽는다

중복 제거 결과를 제대로 설명하려면 `정규화 → 특징 → 후보 → 검증 edge → component → survivor → rewrite`를 끊어 읽어야 한다. 첫 단계에서 공백·Unicode·숫자·구두점을 어떻게 바꾸느냐가 동치류를 정한다. exact hash는 이 표면이 완전히 같은 문서를 찾는다. MinHash는 shingle 집합의 Jaccard를 유한한 signature로 추정하고 LSH는 그중 비교할 후보를 싸게 고른다. 어느 단계도 홀로 “중복”이라는 사실을 완성하지 않는다.

DataTrove의 고정 소스는 이 구분을 코드 구조로 드러낸다. `MinhashDedupSignature.get_signature`가 signature를 만들고, bucket 단계가 candidate pair를 쓰며, `MinhashDedupCluster.run`이 union-find로 component를 닫는다. 직접 시험 `test_signatures`는 같은 설정의 두 실행이 같은 signature를 내는지, `test_buckets_and_cluster`는 작은 문서 묶음이 기대한 pair와 cluster를 만드는지 검사한다. 이는 좋은 지역 보증이지만 새 corpus에서 threshold 주변 pair를 얼마나 놓치는지까지 말하지 않는다. 운영 release에는 labeled pair 표본에서 candidate recall과 verifier precision을 따로 계산해야 한다.

**4.20.1 LSH의 S-곡선은 임계값이 아니다.**

두 문서의 Jaccard가 `s`이고 band가 `b`, band당 row가 `r`이면 후보가 될 확률은 `1-(1-s^r)^b`다. 예컨대 목표 경계 바로 아래에서도 후보가 될 수 있고, 바로 위에서도 놓칠 수 있다. 따라서 보고서에 `threshold=0.8` 하나만 쓰면 잘못이다. shingle n, hash family와 seed, signature 길이, `b/r`, giant bucket cap, 후속 exact-Jaccard verifier와 그 threshold를 함께 적는다.

NeMo Curator의 `GPUMinHash.compute_minhashes`와 `test_minhash_approximation`은 이 근사를 코드와 합성 oracle로 잇는다. 시험의 허용 오차는 근사기가 exact Jaccard 자체가 아님을 명시한다. 여기서 false negative는 LSH가 비교해야 할 pair를 후보로 올리지 못한 경우이고, false positive는 후보가 됐지만 verifier에서 중복이 아니거나 정책상 보존해야 하는 경우다. candidate false positive는 주로 비용을 늘리지만, verifier 결과를 확인하지 않고 삭제하면 곧 데이터 손실이 된다.

**4.20.2 연결요소는 추이적 동일성을 새로 만든다.**

`A≈B`, `B≈C`라는 두 edge가 있으면 약연결요소는 셋을 묶는다. 하지만 `A≈C`일 필요는 없다. template hub, 약관 boilerplate, 반복되는 코드 header처럼 degree가 큰 문서가 있으면 이 chaining이 빠르게 커진다. DataTrove의 union-find와 NeMo Curator의 `ConnectedComponentsStage.weakly_connected_components`는 바로 이 graph closure를 수행한다. 작은 canonical 시험이 label mapping을 고정해도 giant component의 의미적 응집도나 survivor 품질은 보증하지 않는다.

component 감사에서는 edge 수와 component 크기만 보지 않는다. 각 member의 representative similarity, minimum spanning edge, source·domain·시간 분포, hub degree와 제거 token 질량을 본다. survivor를 최초 crawl, 최고 품질, 가장 긴 문서 중 무엇으로 고르는지도 별도 정책이다. 대표가 삭제 요청을 받았을 때 loser를 자동 승격한다면 새 대표의 권리·PII·오염 판정을 다시 수행한다.

**4.20.3 반사실 fixture가 detector의 손실을 보이게 한다.**

작은 인수 표에는 다음 pair를 함께 넣는다. bytes만 같은 exact pair, NFC/NFD만 다른 pair, 숫자 하나가 중요한 수식·코드 pair, Jaccard 경계 바로 아래와 위, A-B-C chain, 공통 지시 template를 공유하지만 답은 다른 깨끗한 pair다. 각 행에는 gold relation과 `must-keep/must-drop/review` 정책을 분리해 둔다. detector가 낸 후보, verifier score, component와 survivor를 모두 보존하면 최초 오판이 어느 단계인지 찾을 수 있다.

threshold sweep은 삭제 문서 수가 아니라 pair-level recall·precision, component-level overmerge·undermerge, source별 제거 token 질량과 downstream clean loss를 함께 낸다. normalization을 강하게 만들었더니 duplicate recall은 올랐지만 코드 상수와 수식 기호가 사라졌다면 그것은 공짜 개선이 아니다. 같은 raw universe와 token budget에서 detector 하나만 바꾼 matched rebuild가 필요하다.

## 4.21 오염 검출과 실제 학습 노출을 분리한다

DataTrove의 `NGramsDecontFilter.filter`와 lm-evaluation-harness의 `get_train_overlap`은 특정 정규화와 n-gram corpus에서 lexical hit를 찾는다. 전자는 query·label·overlap 모드를 구분하고, 후자는 평가 문서별 query n-gram을 정렬된 학습 n-gram stream과 대조한다. 이 결과가 증명하는 것은 `CorpusRevision C` 안에 `DetectorRevision D`가 찾은 문자열 흔적이 있다는 사실까지다.

그 문서가 dedup에서 제거됐는지, tokenizer 뒤 잘렸는지, pack의 context로만 들어갔는지, label mask 아래 loss를 받았는지, 어느 UpdateID에서 소비됐는지는 별도 lineage다. 더 나아가 checkpoint 점수가 그 노출 때문에 올랐다는 인과 주장에는 matched rebuild나 적절한 counterfactual이 필요하다. `hit → exposure → behavior effect` 세 등급을 하나의 contamination boolean으로 압축하지 않는다.

canonical decontamination 시험 자체도 상태를 읽어야 한다. DataTrove의 query·label·overlap 시험은 현재 dependency 제약으로 skip 표지가 붙어 있다. 소스에 시험 함수가 존재한다는 사실은 실행 성공과 같지 않다. release certificate에는 `LOCATED`, `EXECUTED_PASS`, `SKIPPED(reason)`, `NOT_RUN`을 구분하고, 이 책의 감사는 대규모 corpus나 GPU pipeline을 실행했다는 주장을 하지 않는다.

## 4.22 다국어·코드·수학 데이터는 한 개의 품질 점수로 합쳐지지 않는다

웹 문서를 영어냐 아니냐로 가르는 일, 저장소를 재배포할 수 있는지 판단하는 일, 수식이 실제 추론을 담는지 찾는 일은 서로 다른 판정 문제다. `quality_score > τ` 하나로 셋을 합치면 탈락 이유와 분모가 사라진다. release row에는 최소한 `language_model_revision`, 언어별 score, license detector와 정책 revision, code repository·commit·path, math extractor·classifier revision, 각 판정의 reason을 독립 열로 둔다. 그래야 threshold 하나를 바꿨을 때 언어·라이선스·도메인 중 어느 질량이 움직였는지 설명할 수 있다.

**언어 식별은 라벨링과 선택을 분리한다.**

DataTrove `LanguageFilter.filter`는 고정 revision에서 최상위 언어와 점수를 metadata에 먼저 기록한 뒤, 허용 언어와 threshold로 통과 여부를 정한다. canonical test는 영어·이탈리아어 문서를 통과시키고 프랑스어·포르투갈어 문서를 거부하며 metadata label도 확인한다. 이 좁은 시험이 보증하는 것은 fixture의 함수 의미다. code switching, 짧은 수식, 로마자 표기 비영어 문장, OCR 잡음에서의 실제 오류율은 별도 언어·길이·도메인 층화 표본으로 측정해야 한다.

특히 multilingual mixture에서 문서 수 균형을 token 균형으로 부르지 않는다. 언어별 평균 byte/token 비율과 tokenizer fertility가 다르므로 원문 byte, emitted token, valid-target token, committed-loss mass를 모두 집계한다. 낮은 자원 언어를 oversample했다면 duplicate exposure와 source exhaustion도 같이 보고한다.

**라이선스 필드는 법적 결론이 아니라 정책 입력이다.**

The Stack·StarCoder 계열의 license metadata, permissive-license 선택과 opt-out 절차는 중요한 공개 provenance다. 그러나 detector label이 임의 snapshot의 사용 권리를 자동 증명하지는 않는다. repository identity와 commit, detected license, confidence, policy가 허용한 SPDX expression, 파일별 예외, opt-out/deletion generation을 보존한다. unknown·conflict·generated/vendor subtree는 조용히 permissive로 합치지 않고 quarantine한다.

실패 fixture에는 repository license와 파일 header가 충돌하는 경우, dual license, submodule, vendored dependency, license 파일이 뒤늦게 바뀐 commit, 삭제 요청 뒤 파생 shard가 남은 경우를 넣는다. 합격 조건은 분류 정확도 하나가 아니라 원본 row에서 모든 파생 shard와 checkpoint exposure까지 deletion graph가 닫히는가다.

**코드 오염은 문자열 hit와 시간·저장소 계보를 함께 본다.**

code benchmark의 함수 본문이 학습 corpus에 그대로 있지 않아도 같은 저장소의 test, README 해설, fork와 번역이 답을 누설할 수 있다. exact hash, token n-gram, AST-normalized fingerprint, repository/fork component와 commit cutoff를 층별로 기록한다. benchmark와 같은 component가 classifier train·selection·최종 corpus split을 가로지르면 문서별 random split보다 먼저 component를 분리한다.

DataTrove `NGramsDecontFilter.filter`는 정규화한 word n-gram hash가 평가 색인에 있으면 task와 오염 n-gram을 metadata에 남기고 문서를 거부한다. query 모드와 overlap 모드의 canonical survivor 집합은 서로 다르다. 다만 현재 upstream 시험은 dependency 제약으로 skip되어 있다. 따라서 이 책은 `test located`를 `executed pass`로 승격하지 않는다. release 환경에서 dependency lock과 실제 실행 결과를 새 증거로 붙여야 한다.

**수학 추출기는 classifier 자체의 데이터 누설부터 감사한다.**

OpenWebMath·MathPile류 파이프라인은 LaTeX 흔적, 수식 밀도, 문맥 품질 또는 학습된 classifier로 후보를 좁힐 수 있다. 중요한 질문은 “수학 점수가 높은가”보다 그 classifier가 무엇을 양성·음성으로 배웠고 threshold가 어느 split에서 정해졌는가다. OpenWebMath 표본으로 classifier를 만들고 같은 계보의 문서를 최종 평가에 다시 쓰면 selector가 benchmark 문체를 미리 본다.

canonical split fixture는 classifier train, threshold tuning, corpus selection audit, final benchmark를 duplicate component 단위로 나눈다. Hugging Face Datasets의 `train_test_split(..., stratify_by_column=...)` canonical test는 ClassLabel support와 비율 보존을 고정하고, 열이 없거나 ClassLabel이 아니거나 class별 표본과 split 크기가 부족하면 실패시킨다. 그러나 stratification은 leakage 방지가 아니다. 같은 문제의 해설·번역·합성 변형이 양쪽에 있으면 class 비율은 완벽해도 독립성은 무너진다.

이 네 domain gate를 통과한 뒤에도 최종 승인표는 언어×라이선스 상태×코드/수학 domain의 교차 셀을 본다. 각 셀의 input, survivor, emitted token, valid target과 benchmark-neighbor hit를 표시한다. 주변부 언어의 수학 문서가 전부 제거되거나 permissive 코드가 특정 언어에만 몰리는 현상은 전체 평균 하나로 보이지 않는다.

[SourceRow에서 committed UpdateID까지](../labs/06-source-to-commit-golden-lab.md)는 raw hash와 normalized hash, filter reason과 duplicate parent를 같은 row에 보존한다. 작은 fixture이지만 “탈락 문서 수”가 아니라 어느 SourceRow가 이후 PackID와 UpdateID에 도달했는지 묻는 이 장의 release 계약을 끝까지 잇는다.
평가 집합의 exact·near duplicate와 가중 분모가 어떻게 갈라지는지는 [평가·오염·불확실성 결정적 실습](../labs/24-eval-contamination-uncertainty-lab.md)에서 모델 실행 없이 검산한다.
