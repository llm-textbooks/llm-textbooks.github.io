# Playbook 10. 평가 contamination

## 실행 순서

### 오염 경로 역추적
1. eval row의 exact/fuzzy/n-gram/semantic match를 단계별로 실행한다.
2. threshold별 false-positive control을 검사한다.
3. match를 raw `DocumentID→token shard→packed sample→CheckpointID`로 역추적한다.
4. 영향 checkpoint를 uncontaminated/private split에서 재평가한다.

## 분기

### 판정
- boilerplate match는 별도 class로 분류한다. answer-bearing span이 일치하면 고위험으로 보고, paraphrase는 수동 검토와 semantic detector의 일치가 있어야 판정한다.
- row를 평가에서 빼 점수만 다시 내는 것은 이미 학습된 영향을 제거하지 않는다.

### 오염을 하나의 문자열 match로 정의하지 않는다

입력 prompt만 있는 노출, 정답·reference·rationale까지 있는 노출, 번역·paraphrase·format 변형, benchmark 생성 소스와 가까운 파생 자료, evaluation 실행 중 생성된 output의 후속 학습을 나눈다. prompt 노출은 task familiarity를 높일 수 있지만 answer-bearing span 노출보다 위험이 낮을 수 있다. 대화 형식·license·boilerplate의 공통 문구는 내용 오염과 분리한다.

오염 단위는 `EvalRowID`, prompt/reference span, 출처 revision, 공개 시점, contamination policy version으로 고정한다. training 원장에는 corpus release, raw DocumentID·byte span, normalized/tokenized span, dedup cluster, packed sample, consumption UpdateID와 CheckpointID를 기록한다. 두 원장이 이어져야 검출 match가 실제 학습에 들어갔는지 판정할 수 있다.

### detector를 계층화하고 반례로 보정한다

exact byte/hash는 빠르고 명확하지만 normalization·format 변형을 놓친다. token n-gram/MinHash·suffix array는 부분 복사를 찾지만 짧은 공통 문구에 false positive가 많다. fuzzy edit·semantic retrieval은 paraphrase 후보를 늘리지만 근거가 되는 exact span을 대신하지 못한다. 따라서 저렴한 detector로 candidate를 좁힌 뒤 answer-bearing 교차·수동 판정으로 마감한다.

threshold는 검출 결과를 보고 정하지 않는다. 알려진 positive pair, 같은 주제지만 독립적인 hard negative, boilerplate, 짧은 answer string, 번역·paraphrase를 포함한 calibration set에서 precision/recall을 본다. 문자열 길이, 언어, code/math, task family별 오탐지를 나눈다. detector끼리 판정이 다르면 자동으로 무죄나 유죄를 정하지 말고 수동 검토 queue로 보낸다.

**작성된 n-gram test와 실제 실행된 test를 구분한다**

lm-evaluation-harness 고정 revision에는 `tests/test_janitor.py`가 있지만 파일 첫머리에서 module-level `pytest.skip(..., allow_module_level=True)`를 호출한다. 아래 `test_word_ngrams`에는 `n=1,2,3,5,13`의 sliding window 결과를 길이·순서·문자열까지 비교하는 assertion이 적혀 있다. 그러나 작성된 assertion은 일반 test collection에서 실행된 oracle이 아니다. test 이름과 코드가 존재한다는 이유로 contamination 제거가 검증됐다고 표시하면 안 된다.

설령 그 단위 test를 실행해 통과해도 증명 범위는 `word_ngrams`의 국소 변환뿐이다. corpus index가 같은 normalization을 썼는지, lookup hit가 threshold를 넘었는지, hit row가 contribution에서 제외됐는지, 제외 뒤 분모가 재계산됐는지는 남는다. 따라서 작은 fixture를 `clean`, `exact answer-bearing overlap`, `boilerplate`, `threshold 바로 아래`, `threshold 바로 위` 다섯 행으로 만들고 다음 상태를 순서대로 저장한다.

`EvalRowID → normalized tokens → n-gram signatures → index candidates → reviewed span class → disposition → committed contribution → numerator/denominator`

exact hit가 나왔는데 원문 span이 boilerplate라면 detector 성공이지 오염 확정이 아니다. semantic candidate가 나왔지만 answer-bearing 대응 구간을 제시하지 못하면 수동 검토 상태다. 반대로 disposition이 contaminated인데 contribution이 남아 있으면 metric 경계 오류다. row를 빼고 분모를 이전 값으로 두면 점수 변화는 모델이 아니라 산술 결함이다. 이 fixture의 모든 전이가 직접 실행되기 전에는 `clean score` 대신 `decontamination 미검증`, `candidate count`, `reviewed precision`을 따로 보고한다.

음성 근거도 함께 남긴다. module-level skip은 “test가 없다”는 보편적 증명이 아니라 선택한 revision의 기본 collection에서 실행되지 않는다는 증거다. n-gram false positive control은 semantic paraphrase recall을 증명하지 않고, 평가 row 제외는 이미 그 자료를 소비한 checkpoint에서 지식을 제거하지 않는다. detector PASS, lineage PASS, metric recompute PASS, model 영향 해소 PASS를 네 칸으로 나눠야 false positive와 부분 복구를 한 성공 상태로 섞지 않는다.

### 첫 30분에는 점수를 고치지 말고 증거를 동결한다

의심이 제기되면 먼저 평가 보고서, evaluator 입력과 출력, decoding 설정, benchmark revision, corpus manifest와 학습 cursor를 읽기 전용 묶음으로 고정한다. 운영 중인 결과 파일을 다시 생성해 덮어쓰면 최초 점수와 후속 정정 점수를 구분할 수 없다. 사건 식별자에 `EvalID`, `ModelRevision`, `DatasetRevision`, `TokenizerRevision`, 실행 시각과 담당자를 붙이고, 원본의 내용 digest와 저장 위치를 기록한다. 공개가 예정돼 있다면 자동 게시를 멈추되 원본 artifact는 삭제하지 않는다.

다음으로 의심의 출발점을 분류한다. 비정상적으로 높은 점수, 특정 문항의 완전 일치 출력, train/eval 파일명 충돌, 검색 detector 경보, 외부 제보는 서로 다른 증거다. 높은 점수만으로 오염을 확정하지 않고, match 하나만으로 실제 소비를 확정하지 않는다. 최초 관측과 아직 추론에 불과한 설명을 사건 기록에서 분리해야 이후 검토자가 결론을 되짚을 수 있다.

| 질문 | 필요한 증거 | 오판하기 쉬운 경우 |
|---|---|---|
| 같은 문자열이 존재하는가 | 양쪽 원문의 byte·token span | license·문제 지시문 같은 상용구 |
| 정답 정보까지 겹치는가 | prompt/reference/rationale 교차 구간 | 짧고 흔한 정답 문자열 |
| 해당 자료를 실제로 읽었는가 | sampler·packed-sample·UpdateID 원장 | corpus에만 있고 소비 전인 자료 |
| 어느 상태부터 영향받았는가 | checkpoint cursor와 descendant 계보 | 파일 생성 시각만 비교하는 경우 |
| 점수 상승이 암기인가 | 동형·시간 분할·비공개 대조군 | 같은 benchmark에서 행만 제외한 재평가 |

### 정규화 파이프라인을 detector의 일부로 봉인한다

비교 전에 Unicode 정규화, 대소문자, 공백, HTML·Markdown 제거, 코드 주석, 수식 표기, 숫자와 보기 순서를 어떻게 처리했는지 고정한다. 정규화를 강하게 하면 회수율은 오르지만 서로 무관한 짧은 문장이 합쳐진다. 약하게 하면 줄바꿈이나 특수문자 하나로 복사를 놓친다. 따라서 원문 byte match와 정규화 match를 별도 열로 남기며, 정규화된 결과만 보존해서는 안 된다.

n-gram detector는 `n`, stride, tokenizer, 최소 겹침 길이와 문서 길이 보정을 함께 기록한다. 동일한 overlap 수라도 20-token 문항의 15-token 일치와 2,000-token 문서의 15-token 일치는 증거력이 다르다. MinHash나 locality-sensitive hashing을 쓰면 signature 생성 revision과 band 설정을 남긴다. semantic retrieval은 후보 생성기로만 사용하고, 최종 판정에는 어느 원문 구간이 어떤 평가 정보와 대응하는지 사람이 확인할 수 있는 근거를 붙인다.

코드와 수학 문제는 자연어와 별도 calibration이 필요하다. 표준 함수 서명, import 문, 정리 이름은 널리 반복된다. 반면 테스트 입력·상수·주석·버그까지 같은 순서로 겹치면 강한 신호다. 객관식은 문제 본문만 아니라 보기의 값과 순서, 정답 index, 해설을 각각 비교한다. 보기 순서를 섞었는데도 동일한 오답 패턴을 재현하는지는 암기와 추론을 가르는 유용한 반례가 된다.

### 시점과 소비 사실을 확인한다

학습 자료에 평가 row가 있어도 해당 checkpoint가 그 자료를 소비하기 전이면 영향은 없을 수 있다. 반대로 raw corpus에서 삭제했어도 tokenized cache·packed shard가 남아 loader가 읽었을 수 있다. source lineage에 그치지 말고 실제로 소비한 sample/span 원장과 UpdateID 범위를 확인한다.

공개 시점도 중요하다. model cutoff 후 공개된 benchmark라도 synthetic data generator·retrieval API·annotator tool을 통해 간접 유입될 수 있다. 공개 전이어도 생성 소스와 학습 corpus가 같은 원문을 공유할 수 있다. 날짜 하나로 무오염을 증명하지 않는다.

### packed sample에서 원문까지 역추적한다

학습 loader가 읽은 것은 대개 원문 문서가 아니라 token shard와 packed sequence다. match된 원문 span을 `DocumentID`와 byte offset으로 고정한 뒤 normalization map을 통해 token offset으로 옮긴다. 그 token이 어느 shard, 어느 packed sample의 어느 구간에 들어갔는지 찾고, sampler epoch·rank·local position을 거쳐 실제 UpdateID에 도달했는지 확인한다. packing 경계에서 문항과 정답이 서로 다른 문서 조각으로 나뉘었는지도 본다. 문항만 소비한 경우와 문항·정답을 연속 문맥으로 소비한 경우는 위험 등급이 다르다.

gradient accumulation과 재시작이 있는 실행에서는 sample cursor만으로 충분하지 않다. 해당 microbatch가 forward만 하고 폐기됐는지, backward에 기여했는지, optimizer effect가 commit됐는지 구분한다. overflow로 step이 건너뛰었거나 장애 직전 accumulation이 checkpoint에 반영되지 않았다면 `read`와 `learned-from` 사이에 간격이 생긴다. 반대로 sampler 재시작으로 같은 sample을 여러 번 소비했을 수 있으므로 effect count를 계산한다.

SFT 이후에도 계보를 이어간다. 오염된 base checkpoint에서 만든 synthetic response, 그 response로 만든 preference pair, reward model, adapter와 merge는 간접 descendant다. 원문 span이 후속 데이터에 그대로 남지 않아도 teacher의 암기가 증류됐을 수 있다. 직접 문자열 match와 모델 계보에 의한 잠재 영향을 별도 등급으로 표시하고, 후자를 무조건 무죄로 처리하지 않는다.

## 영향 측정과 복구

### score sensitivity를 세 계층에서 본다

첫째, 확정·의심·무관 match를 단계별로 제외하면서 평가 점수가 움직이는 범위를 계산한다. 둘째, contamination risk가 다른 row에 inverse-probability 또는 보수적 bound를 적용해 uncertainty를 보고한다. 셋째, sealed private·time-split·counterfactual benchmark에서 같은 capability를 재측정한다. contaminated row만 빼고 나머지 점수를 원래 점수처럼 발표하지 않는다.

표면 암기와 capability generalization을 나누는 반례를 만든다. 이름·숫자·보기 순서·표현을 바꾼 동형 item, 같은 풀이 규칙의 새 item, 정답 문자열은 같지만 근거가 다른 item을 쓴다. 원 row에서만 갑자기 높고 동형에서 무너지면 노출 영향 가설이 강해진다.

### 영향 artifact를 격리한다

확정 contamination이 answer-bearing span으로 학습에 소비됐다면 그 UpdateID 이후 checkpoint, adapter, merge·quantization, preference/reward data, evaluation report·model card를 영향 후보로 표시한다. 해당 상태를 이용해 만든 synthetic data도 역추적한다. 오염 row를 corpus에서 지운 것은 기존 checkpoint의 영향을 지우지 않는다.

복구 선택은 영향 범위와 발행 목적에 따라 다르다. 오염 이전 checkpoint에서 정제 corpus로 재학습, 신뢰할 수 있는 unlearning/editing 절차와 재학습 테스트, benchmark 무효화 및 새 sealed evaluation을 검토한다. 단순 score 수정은 model artifact를 복구하지 않는다. 재학습하지 않는 결정을 내리면 제한·불확실성·무효 점수를 model card에 표시한다.

### 복구안을 비용이 아니라 주장 범위로 선택한다

오염 이전 checkpoint가 있고 소비 구간이 좁다면 그 지점에서 정제 shard와 새 sampler generation으로 재개하는 것이 가장 해석하기 쉽다. 이때 오염 문서만 삭제하고 token budget을 줄이지 않는다. replacement distribution과 curriculum 위치가 달라지므로 새 run은 별도의 revision이다. 첫 update의 sample IDs, loss denominator와 optimizer state를 비교해 재개의 의미가 유지됐는지 확인한다.

오염 이전 상태가 없거나 영향이 오래 누적됐다면 재학습, 검증된 unlearning, 평가 주장 축소 가운데 선택한다. unlearning을 택할 때는 forget set의 loss 변화만 보지 않는다. 동형 문항의 제거, retain set 성능, 근접하지만 정당한 지식의 손상, 재학습 공격으로 기억이 돌아오는지까지 본다. 삭제 기법을 실행했다는 사실은 영향 제거의 증거가 아니다.

평가만 새로 만들 수 있는 경우에는 기존 점수를 폐기하고 독립된 population을 정의한다. 같은 template에서 명사와 숫자만 바꾼 문항은 leakage channel을 그대로 공유할 수 있다. 생성자, 원천 자료, 작성 시점과 검토자를 분리하고, 모델 개발자가 반복 조회할 수 없는 sealed split을 둔다. 새 점수에는 이전 점수와 직접 비교할 수 없는 이유와 uncertainty를 함께 쓴다.

영향받은 공개 artifact가 있으면 배포 상태도 관리한다. model card와 benchmark 표에 정정 표시를 남기고, registry에서는 affected descendant를 격리한다. 이미 내려받은 사용자가 digest로 상태를 식별할 수 있도록 문제 revision과 대체 revision을 동시에 명시한다. 조용히 파일을 교체하면 제3자가 어느 모델로 실험했는지 복원할 수 없다.

### 종료 전 반증 실험

복구된 모델에는 최소 네 종류의 대조군을 건다. 첫째는 원 문항의 표현만 바꾼 근접 변형, 둘째는 같은 풀이 규칙이지만 새 내용인 동형 문항, 셋째는 정답 문자열만 같고 근거가 다른 hard negative, 넷째는 완전히 독립된 비공개 문항이다. 원 문항 성능만 떨어지고 동형 능력까지 무너지면 과도한 제거일 수 있다. 원 문항과 표현 변형만 비정상적으로 높다면 암기가 남았을 가능성이 있다.

detector 자체에도 실패를 주입한다. Unicode homoglyph, 공백·주석 변형, 번역, 보기 순서 변경, 긴 상용구와 짧은 answer-bearing span을 넣어 기대 등급을 확인한다. threshold를 바꾸었을 때 영향 문서와 checkpoint 범위가 얼마나 흔들리는지 민감도 표를 만든다. 결론이 좁은 threshold 구간에서만 성립하면 확정 판정보다 불확실성으로 보고한다.

### 재발 방지 gate

corpus release 전에 public benchmark의 prompt/reference/rationale 지문을 독립된 프로세스로 scan하고 detector/policy revision을 manifest에 고정한다. dedup 전 raw, normalized, tokenized, packed 각 계층에서 match lineage를 유지한다. SFT·preference·RL 데이터와 synthetic generator prompt도 같은 gate를 통과한다. evaluation output이 후속 학습 cache에 자동 유입되지 않게 권한·namespace·retention을 나눈다.

sealed/private evaluation은 무조건 안전한 것이 아니다. 접근 log, export 기록, evaluator prompt, model output cache, 사람 annotation 환경을 감사한다. item을 반복 재사용하면 수동 hyperparameter tuning을 통한 adaptive contamination이 생길 수 있으므로 query budget·holdout rotation·decision log를 둔다.

## 종료 조건

### 통과
영향 범위, score sensitivity, 재학습/무효화 결정과 새 `EvalID`가 기록돼야 한다.

종료 묶음은 match detector·threshold calibration, row별 관측 증거 span, training lineage와 consumption UpdateID, 영향 CheckpointID·PolicyVersion, score sensitivity·private split·counterfactual 결과, 격리·복구 결정을 포함한다. false-positive control과 수동 adjudication 기록도 남긴다. 새 `EvalID`는 contaminated row를 조용히 뺀 숫자가 아니라 바뀐 population, 추정량, uncertainty와 이전 점수와의 비교 제한을 명시해야 한다.
