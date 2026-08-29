# 27장 공급망과 재현성: 같은 이름의 다른 모델을 막는다

재현성은 seed 하나를 적는 것으로 끝나지 않는다. 데이터셋과 코드가 같아 보여도 wheel, CUDA 라이브러리, 컨테이너가 달라지면 다른 커널과 변환 경로가 선택된다. 같은 `checkpoint-final`이라는 이름 아래 shard가 교체될 수도 있다. 공격자는 바로 이 틈, 즉 사람이 같은 것이라고 믿지만 기계가 동일성을 증명하지 못하는 연결을 노린다.

따라서 이 장의 질문은 “파일을 어디에서 받았는가”가 아니다. **어떤 원본과 정책에서 출발한 바이트가, 어느 빌더와 실행 환경을 거쳐, 어떤 학습 상태와 변환을 품은 산출물이 되었으며, 지금 이 배포에서 왜 사용해도 되는가**이다. 답은 설명문 하나가 아니라 검산 가능한 상태 사슬이어야 한다.

`source revision → build environment/CUDA dependency → dataset·model·tokenizer·effective config → training run → checkpoint → adapter·merge·quantized export → manifest·SBOM·signature → registry promotion → deployment actual state → rollback·revocation`

이 사슬에서 노드의 정체성은 이름이 아니라 content digest이고, 간선의 정체성은 “대략 여기서 만들었다”가 아니라 입력 digest, 변환 코드, effective config, builder identity와 출력 digest다. 한 고리가 비면 뒤의 서명과 평가가 아무리 좋아도 계보 전체는 닫히지 않는다. 이 장은 먼저 이 사슬을 읽는 법을 세우고, 뒤에서 로더·SLSA·in-toto·Sigstore·SBOM·CUDA·체크포인트·캐시·철회를 차례로 그 위에 올린다.

### 독자가 먼저 붙잡아야 할 세 가지 분리

첫째, **identity와 authorization은 다르다.** SHA-256이 같다는 사실은 바이트가 같다는 뜻이지, 그 바이트를 production에서 써도 된다는 뜻이 아니다. 서명이 유효하다는 사실도 signer가 이 repository와 release channel에 권한이 있다는 뜻은 아니다. `content digest → signature authenticity → signer authorization → current policy·revocation`을 별 gate로 검사한다.

둘째, **artifact 완전성과 학습 재개 완전성은 다르다.** 모델 weight만 온전하면 추론은 가능할 수 있다. 그러나 같은 다음 update를 만들려면 optimizer의 moment와 parameter-group mapping, scheduler와 scaler, 각 RNG namespace, sampler·data cursor, gradient accumulation phase, world-size와 sharding metadata가 필요하다. 이것이 빠진 checkpoint를 로드해 학습이 시작됐다고 해서 “resume에 성공했다”고 말하면 안 된다. 그것은 같은 run의 연속이 아니라 일부 상태를 새로 만든 fork다.

셋째, **bitwise identity와 semantic parity는 다르다.** archive timestamp만 달라도 파일 digest는 달라질 수 있고, 반대로 파일 digest가 같아도 잘못된 tokenizer나 chat template와 결합하면 모델 함수는 달라진다. 그래서 바이트, 수치, 의미와 행동을 각각 검증한다. 낮은 등급의 통과는 높은 등급을 대신하지 않으며, 높은 바이트 동일성도 라이선스·데이터 동의·현재 사용 허가를 대신하지 않는다.

### 공급망 상태 사슬을 한 번에 읽는 표

| 경계 | 고정해야 하는 상태 | 처음 비교할 증거 | 최초 불일치가 뜻하는 것 | 다음 행동 |
|---|---|---|---|---|
| source·revision | commit, dirty tree, submodule, generated source | tree digest와 source manifest | 서로 다른 코드를 같은 release로 부름 | 빌드를 중단하고 source resolution부터 고친다 |
| build·CUDA | container, lock, compiler, flags, SM target, 실제 load된 `.so` | build attestation, SBOM, runtime inventory | dependency drift 또는 non-hermetic fetch | clean builder와 network 차단 fixture로 숨은 material을 찾는다 |
| data·tokenizer·config | shard, transform, token-ID map, template, resolved defaults | BundleID와 canonical effective config | 같은 문장이 다른 token·label·sample이 됨 | token trace와 SampleID에서 첫 차이를 찾는다 |
| training run | BatchID, RNG, precision, topology, code path | RunAttestation과 첫 update trace | 입력은 같지만 실행 상태가 달라짐 | forward→backward→collective→optimizer 순서로 좁힌다 |
| checkpoint | weight, optimizer, scheduler, scaler, RNG, cursor, shard generation | generation manifest와 next-update oracle | load 성공과 exact resume를 혼동함 | 누락 상태를 자동 초기화하지 말고 새 fork로 표시한다 |
| adapter·export | base digest, merge order, quantization/calibration, runtime kernel | derivation statement와 schema | 유효한 부품을 잘못된 조합으로 묶음 | parent compatibility와 golden logits를 재검사한다 |
| sign·registry | exact subject set, signer identity, policy, revocation | signature bundle과 DecisionRecord | 악성 치환 또는 권한 없는 정상 서명 | quarantine하고 alias를 움직이지 않는다 |
| deploy·rollback | desired BundleID, actual-loaded digest, cache generation | replica heartbeat와 canary | registry는 맞지만 fleet가 stale함 | traffic을 격리하고 승인된 exact digest로 rollback한다 |

표의 순서는 곧 디버깅 순서다. 최종 평가 점수가 다르다고 곧바로 CUDA nondeterminism을 의심하지 않는다. source tree가 같은지, 실제 데이터와 token ID가 같은지, 다음 batch와 RNG가 같은지부터 본다. 반대로 digest mismatch가 났는데 모델이 “정상적으로 답한다”는 이유로 통과시키지도 않는다. 의미가 비슷한 악성 치환도 공급망 사고이기 때문이다.

### 하나의 체크포인트가 남겨야 할 최소 상태

학습 checkpoint의 root manifest는 적어도 다음 논리 구조를 표현해야 한다. 이것은 특정 직렬화 포맷을 강제하는 예제가 아니라, 무엇이 빠졌는지 기계가 판정하기 위한 schema다.

```text
CheckpointGeneration {
  identity:      RunID, AttemptID, UpdateID, parent_generation
  materials:     SourceID, DatasetRevision, TokenizerBundleID,
                 ModelConfigID, EnvironmentID
  model_state:   exact shard set, tensor schema, shard digests
  update_state:  optimizer class, parameter-group mapping, moments,
                 scheduler, grad scaler, accumulation phase
  stream_state:  RNG namespaces, sampler epoch/cursor, worker state,
                 packed-sample lineage
  distributed:   world size, mesh, rank-to-shard map, completion generation
  evidence:      writer attestations, read-back result, next-update oracle
}
```

여기서 중요한 것은 파일이 존재한다는 사실이 아니라 관계가 닫혔다는 사실이다. 예를 들어 optimizer shard 64개가 있어도 parameter 이름 또는 stable ID에서 group과 moment로 가는 mapping이 없으면 상태를 올바른 parameter에 붙였다고 증명할 수 없다. tokenizer 파일이 있어도 special-token ID와 chat template가 학습 때 사용한 것과 같은지 모르면 labels의 위치를 재현할 수 없다. distributed writer가 모두 성공했다고 보고해도 generation manifest가 모든 필수 shard digest를 원자적으로 commit하지 않았다면 reader는 부분 세대를 완전한 checkpoint로 오인할 수 있다.

복구 검증은 새 process와 빈 application cache에서 수행한다. 직전 process에 남은 tokenizer, RNG, open dataloader와 JIT cache가 누락 상태를 우연히 보충할 수 있기 때문이다. 복구 branch와 uninterrupted branch에서 다음 `BatchID → token IDs·labels → loss numerator·denominator → selected gradient → parameter delta → optimizer state → scheduler LR`을 순서대로 비교한다. 첫 차이가 바로 소유자와 다음 실험을 정한다.

## 27.1 provenance DAG에서 산출물의 부모와 소비자를 잇는다

공급망은 파일 목록이 아니라 변경의 인과관계다. source, data, environment, training invocation, checkpoint와 release가 어떤 digest와 승인으로 이어지는지 DAG로 고정한다.

### 이름 대신 digest

`main`, `latest`, `model-final`은 identity가 아니라 사람이 읽는 가변 별칭이다. Git commit, dataset revision, container digest, wheel hash, checkpoint shard hash와 index hash를 manifest에 둔다. `RunID`는 이 입력 digest 집합과 canonical effective-config checksum을 가리킨다. 요청한 별칭과 실제 resolve된 digest를 모두 남겨야 나중에 “무엇을 요구했고 무엇을 받았는가”를 비교할 수 있다.

### build와 runtime을 분리한다

CUDA toolkit으로 compile한 버전, wheel이 포함한 실행 환경 library, host driver, GPU architecture는 별도 축이다. import 성공은 커널이 기대 architecture로 compile됐다는 증거가 아니다. extension build log와 loaded shared object 목록을 보존한다. `LD_LIBRARY_PATH`, host mount, JIT cache와 자동 backend 선택 때문에 선언한 dependency와 실제 소비한 dependency가 달라질 수 있으므로 build-time inventory와 runtime inventory를 대조한다.

### derivation statement

artifact마다 subject digest, builder identity, 소스/material digest, command/config, output digest를 연결한다. 동일 source라도 compiler flag와 CUDA arch list가 다르면 다른 artifact다. training checkpoint는 dataset, tokenizer, code, container, canonical config와 parent checkpoint를 materials로 가진다. adapter는 base model을, merge는 base와 adapter를, quantized export는 source weight와 calibration data·quantizer·runtime contract를 각각 parent로 가져야 한다.

**재빌드 결정 트리**

output hash가 다르면 먼저 nondeterministic metadata, timestamp, absolute build path와 archive order를 분리한다. tensor bytes가 다르면 compiler, kernel, RNG와 reduction order를 본다. tensor는 같은데 package hash만 다르면 packaging metadata를 본다. 이때 canonicalized digest를 실제 배포 digest인 것처럼 바꾸지 않는다. 실제 byte identity와 차이의 의미를 나란히 보고하고 reproducible build를 주장할 범위를 명시한다.

## 27.2 artifact 형식과 loader의 실행 권한을 분리한다

형식의 안전성은 확장자가 아니라 parser가 실행할 수 있는 동작, tensor metadata와 외부 코드 의존성의 경계에서 결정된다.

### safetensors와 pickle

safetensors는 tensor metadata와 bytes에 집중해 임의 Python object 실행면을 줄인다. pickle checkpoint는 load 자체가 코드 실행일 수 있다. 신뢰하지 않는 artifact를 production credential이 있는 process에서 열지 않는다. optimizer state가 pickle만 지원되면 격리된 변환 단계와 allowlist를 둔다.

### remote code와 tokenizer

`trust_remote_code`는 model implementation뿐 아니라 import side effect를 허용할 수 있다. revision을 고정하고 코드를 검토한다. tokenizer JSON, added tokens, chat template도 모델 weights와 함께 서명해야 한다. weight만 같아도 tokenizer가 바뀌면 다른 함수다.

### load 전 검증 순서

다운로드 임시 경로에서 size/hash/signature를 확인하고 expected manifest와 tensor index를 비교한 뒤 격리 process에서 parse한다. 그 뒤에만 production model store로 promote한다. hash 검증 전에 pickle/remote code를 import하면 검증 순서가 뒤집힌다.

**tensor schema 검사**

shard index의 tensor name→file mapping, dtype, shape, total bytes를 config와 비교한다. tied embedding/head가 실제로 공유되는지 export format에 따라 확인한다. unexpected/missing key를 경고로 흘리고 serving하면 silent fallback이 생길 수 있다.

**정상 파일·악성 파일·잘못 묶인 파일을 구분한다**

검증 실패를 모두 “hash 오류”로 뭉개면 대응이 틀어진다. 전송 중 손상이나 mirror 치환은 실제 byte가 expected digest와 다른 **identity 실패**다. 공격자가 훔친 승인 키로 악성 artifact를 정확히 서명했다면 digest와 signature는 모두 맞을 수 있지만 builder·workflow 권한, source materials 또는 행동 gate에서 실패해야 한다. 정상 weight를 다른 tokenizer나 config와 묶은 경우에는 각 파일의 digest와 서명이 모두 유효해도 **bundle 의미 계약**에서 실패한다.

따라서 download gate는 `exact file set·digest`, attestation gate는 `subject binding·materials·builder`, authorization gate는 `signer·workflow·channel`, loader gate는 `format·tensor schema·remote-code capability`, integration gate는 `tokenizer·config·runtime 조합`, 행동 관문는 `golden trace`를 맡는다. 한 gate가 나머지를 대신하지 않는다. 정상 파일을 잘못 묶은 사고와 권한 있는 계정이 만든 악성 파일을 구분할 수 있어야 조사 범위와 철회 반경도 정확해진다.

dependency drift 역시 세 부류로 나눈다. lockfile이 달라진 **선언 drift**, lockfile은 같은데 wheel이나 container digest가 달라진 **resolution drift**, 선언 artifact는 같지만 host mount·loader search path·JIT가 다른 코드를 선택한 **실행 환경 drift**다. 각각 manifest diff, resolved-material diff, actual-loaded inventory diff에서 처음 드러난다. `pip freeze`가 같다는 이유로 세 번째 부류를 닫았다고 말하지 않는다.

## 27.3 poisoning·secret·substitution의 위협 모델을 세운다

공격자를 하나로 뭉개지 않고 데이터 오염, 비밀 유출, artifact 치환과 transitive dependency 변조를 서로 다른 탐지 경로에 둔다.

### 데이터 공급망

수집 URL, extraction result, filter reason, dedup survivor를 lineage로 남긴다. secret scanner와 PII detector의 version 및 false-negative 범위를 기록한다. 삭제 요청은 23장의 `RevocationID`로 후손 artifact를 무효화한다.

### checkpoint substitution

분산 shard 하나가 바뀌어도 index가 같은 파일명을 가리킬 수 있다. shard별 hash와 전체 manifest signature를 검증한 뒤 load한다. object storage의 partial upload는 temporary generation에 쓰고 completion marker가 모든 shard digest를 포함하게 한다.

### poisoning 탐지의 한계

keyword/secret scanner, near-duplicate, anomaly score는 알려진 pattern을 잡는다. clean score가 안전 증거는 아니다. 출처 신뢰도, reviewer sampling, canary behavior, gradient/outlier 분석을 겹친다. detector version과 threshold를 corpus manifest에 둔다.

**credential과 log**

training config, environment dump, traceback, W&B artifact에 token이 들어갈 수 있다. secret을 metric label이나 command line에 넣지 않고 short-lived credential과 redaction을 쓴다. 이미 노출되면 삭제뿐 아니라 rotation과 descendant log/artifact revocation을 수행한다.

## 27.4 release verification을 실패 우선 gate로 실행한다

release 승인은 서명 존재 여부 하나가 아니라 identity, provenance, policy, 수치 parity와 폐기 가능성을 함께 검사하는 transaction이다.

**SBOM과 서명.**

SBOM은 포함된 package를 보여주지만 안전을 보장하지 않는다. provenance attestation은 누가 어떤 builder에서 어떤 입력으로 만들었는지 연결한다. verify 단계가 실패하면 fallback artifact를 조용히 쓰지 않는다.

**재현성 등급.**

“재현됐다”는 말 대신 검증한 경계를 적는다.

| 등급 | 요구하는 동일성 | 대표 oracle | 이 등급이 증명하지 않는 것 |
|---|---|---|---|
| artifact-identical | manifest와 파일 byte가 동일 | bundle·shard digest | 올바른 tokenizer 조합, 현재 사용 허가 |
| state-restorable | checkpoint의 다음 update 상태가 복원됨 | next BatchID, LR, parameter delta, moment checksum | 처음부터 다시 학습한 결과의 동일성 |
| sample-exact | 소비 sample·token·label 순서가 동일 | SampleID와 token trace prefix | kernel과 reduction의 bitwise 동일성 |
| numerically equivalent | 정한 tensor·연산이 사전 tolerance를 만족 | logits, loss, gradient, update의 first-difference/ULP | task 수준의 행동·안전 동등성 |
| semantically/behaviorally comparable | 동일 계약의 평가 분포가 허용 구간을 만족 | frozen evaluation, repeated generation, CI | artifact identity와 계보 무결성 |

대규모 GPU run에서 bitwise training identity가 현실적인 목표가 아닐 수 있다. 그렇더라도 manifest, source archive, dataset shard처럼 가능한 경계의 byte identity까지 포기할 이유는 없다. 반대로 평가 점수가 비슷하다는 semantic parity를 artifact substitution 허가로 쓰면 안 된다. 요구 등급, hardware·topology, 반복 수와 tolerance는 결과를 보기 전에 선언한다.

**verify policy.**

개발 build는 unsigned artifact를 허용할 수 있어도 production promotion은 signature, trusted builder, 소스 리비전, vulnerability/license policy를 요구할 수 있다. 예외에는 소유자와 만료일이 있어야 한다. 검증 실패 시 이전 artifact로 조용히 fallback하지 않고 release를 중단한다.

**source/test와 실험.**

safetensors spec 고정 revision에서 header와 tensor byte contract를, framework loader에서 missing/unexpected key 처리를 읽는다. in-toto/SLSA류 attestation은 provenance 구조를 제공하지만 모델 안전을 증명하지 않는다. 독자 실험은 shard 한 byte, tokenizer 파일, index mapping을 각각 변조해 어느 gate에서 거부되는지 확인한다.

**provenance를 그래프로 읽는다.**

provenance의 최소 단위는 `material→builder invocation→subject`다. source commit, lockfile, dataset manifest, parent checkpoint와 compiler image가 material이고, 격리된 builder identity와 command/config가 invocation이며, wheel·container·checkpoint가 subject다. 각 edge는 이름이 아니라 digest로 식별한다. 증명 문서 자체도 서명되고 어디에 보관됐는지 추적한다.

training run은 단일 build보다 길다. pretraining checkpoint에서 adapter를 학습하고 merge한 뒤 quantize하면 네 개의 서로 다른 artifact와 derivation edge가 생긴다. merge script revision, base와 adapter digest, dtype, shard 정책을 보존한다. 최종 파일만 서명하면 어떤 조상에서 문제가 들어왔는지 찾을 수 없다.

검증기는 주장된 source repository가 존재하는지만 보지 않는다. 승인된 builder가 정확한 material digest를 받았는지, subject digest가 실제 다운로드 bytes와 같은지, 정책이 요구한 step이 생략되지 않았는지 확인한다. provenance는 JSON이 있다는 사실이 아니라 검증 가능한 관계다.

**Hugging Face artifact를 고정한다.**

Hub의 model ID와 branch 이름은 가변 별칭이다. 다운로드에는 immutable revision을 사용하고 실제 resolve된 commit을 manifest에 남긴다. `config.json`, generation config, shard index, 모든 weight shard, tokenizer JSON/model, special-token map, chat template와 custom Python 파일을 하나의 artifact 집합으로 취급한다. 일부 파일만 revision을 고정하고 나머지를 최신으로 받지 않는다.

model card는 라이선스, intended use, base model과 평가 조건을 설명하지만 weight bytes의 증명서는 아니다. card의 `base_model`, dataset, library version 주장을 graph edge 후보로 가져오되 실제 digest와 build record로 확인한다. 누락된 정보는 추정 provenance로 표시하고 검증된 계보와 섞지 않는다.

cache 경로가 이미 존재할 때도 revision과 hash를 다시 확인한다. offline mode가 오래된 mutable snapshot을 조용히 재사용할 수 있다. snapshot directory의 symlink 대상, local-files-only 동작과 cache eviction을 fixture로 시험한다. production promotion은 네트워크에서 직접 load하지 않고 검증된 내부 store에서 수행한다.

**safetensors의 경계.**

safetensors 파일은 앞부분의 header 길이, JSON metadata와 tensor별 dtype·shape·data offset, 뒤의 연속 byte 영역으로 구성된다. loader는 offset의 범위와 겹침, 예상 file size, dtype와 shape가 요구하는 byte 수를 검증해야 한다. 형식이 임의 Python object 실행을 피한다고 해서 악의적인 거대 shape나 자원 고갈 위험까지 사라지는 것은 아니다.

sharded model은 index JSON이 tensor 이름을 shard 파일에 매핑한다. index와 shard 모두 hash해야 한다. index만 바꾸어 tensor를 다른 shard로 가리키거나 shard 한 개를 이전 버전으로 치환하는 실험을 한다. loader가 missing key를 자동 초기화하거나 unexpected key를 무시한다면 strict production policy로 승격한다.

metadata에 저장된 format이나 framework 문자열을 신뢰해 코드를 실행하지 않는다. load 전에 전체 manifest를 검증하고 격리된 process에 memory/CPU 제한을 둔다. 형식 안전성과 모델 의미의 정확성은 별개이므로 config의 layer 수, hidden size, vocabulary와 tensor schema를 교차 검사한다.

**pickle과 remote code의 신뢰 경계.**

pickle은 객체 복원 과정에서 callable을 실행할 수 있으므로 출처가 불명확한 checkpoint를 `torch.load`하는 행위 자체가 실행이다. “weights only” 제한이 있는 loader도 지원 type과 framework version을 확인한다. 불가피한 optimizer state 변환은 credential과 network가 없는 disposable container에서 하고 결과 tensor를 새 manifest로 promote한다.

remote model code는 architecture 정의뿐 아니라 import 시점의 side effect, dynamic import와 dependency 설치를 포함할 수 있다. commit 고정 뒤 파일 목록과 import graph를 검토하고 network/file/process access를 sandbox에서 관측한다. `trust_remote_code=True`를 편의 설정으로 전역 활성화하지 않는다. 승인된 custom code digest를 model manifest와 묶는다.

tokenizer의 custom normalizer나 decoder도 입력 bytes를 바꾸는 실행 경계다. added token ID 충돌, special token 재배치, chat template 변경을 golden strings와 token IDs로 검사한다. weights가 동일해도 이 계약이 다르면 학습과 추론 함수가 달라진다.

**SBOM이 말하는 것과 말하지 않는 것.**

SBOM은 package 이름·version·license·dependency를 열거해 알려진 취약점 조회와 영향 분석을 돕는다. 그러나 동적으로 load된 shared object, JIT로 만든 kernel, vendored source, host driver가 빠질 수 있다. Python lockfile만으로 CUDA 실행 환경의 SBOM이 완성되지 않는다.

container layer, OS package, Python wheel, wheel 내부 native library, NCCL/CUDA runtime, compiler와 firmware/driver를 계층별 inventory로 남긴다. 실행 시 `/proc` mapping이나 loader trace로 실제 load된 library digest를 수집해 선언 SBOM과 비교한다. build dependency와 runtime dependency를 구분한다.

취약점 scanner가 clean이라고 artifact가 안전한 것은 아니다. 데이터 poisoning, 모델 backdoor, 잘못된 license와 계정 탈취 서명은 package CVE 목록에 나타나지 않는다. SBOM은 provenance·서명·행동 평가와 결합되는 색인이지 승인 점수 하나가 아니다.

**서명과 trust root.**

hash는 bytes가 같음을 알려주지만 누가 승인했는지는 말하지 않는다. signature는 key identity를 subject digest와 policy statement에 결합한다. 검증기는 신뢰 root, 허용 signer, 용도, 발급·만료 시간과 transparency record를 확인한다. 저장소 옆에 public key를 같이 둔다고 신뢰가 생기지 않는다.

builder key와 human release key의 역할을 나눈다. 자동 build가 provenance를 서명하고, release approver는 검증 결과와 promotion policy를 서명한다. 장기 key 대신 workload identity와 짧은 수명의 certificate를 쓰면 탈취 범위를 줄일 수 있다. offline 환경에서는 root rotation과 폐기 목록 배포 방법을 미리 설계한다.

서명 성공 뒤에도 subject가 요청한 모델인지 확인한다. 공격자가 유효하게 서명된 다른 checkpoint를 치환할 수 있기 때문이다. deployment manifest가 기대하는 artifact ID와 digest, environment, policy version을 함께 검증한다.

**폐기는 후손으로 전파된다.**

dataset 문서, dependency, builder key, base checkpoint 중 하나가 폐기되면 그것을 material로 사용한 모든 descendant를 찾아야 한다. graph query는 direct child만이 아니라 adapter, merge, quantized export, distilled model과 deployment bundle까지 순회한다. 영향이 불확실하면 보수적으로 quarantine하고 재검증한다.

RevocationID에는 원인, affected digest set, 판정 시간, 대체 artifact와 owner를 둔다. object를 삭제하는 것과 serving에서 비활성화하는 것은 별도 상태다. CDN, node cache와 장기 실행 process가 옛 bytes를 계속 쓸 수 있으므로 rollout inventory와 runtime heartbeat의 loaded digest를 확인한다.

서명 key 폐기는 그 key로 만든 artifact가 모두 악성이라는 뜻은 아니지만 신뢰 증거가 약해졌다는 뜻이다. 별도 신뢰 경로로 재증명하거나 rebuild한다. 단순히 새 key로 옛 bytes에 재서명하면 침해 기간의 builder integrity 문제를 해결하지 못한다.

### CUDA와 native extension 재현성

같은 Python package version이라도 GPU architecture list, compiler, optimization flag와 CUDA toolkit이 다르면 wheel 내부 kernel이 다르다. PTX만 포함했는지 특정 SM cubin을 포함했는지, runtime JIT가 발생했는지를 기록한다. host driver가 선택한 code path와 실제 device capability도 RunID에 붙인다.

NCCL, cuDNN, cuBLAS와 custom attention extension은 system library를 쓰거나 wheel에 vendoring할 수 있다. version 문자열뿐 아니라 loaded path와 digest를 수집한다. `LD_LIBRARY_PATH` 순서가 다른 library를 먼저 선택하는 실험으로 startup gate가 차이를 잡는지 본다. import test만으로 collective와 kernel 정확성을 증명하지 않는다.

재현 실험은 먼저 artifact-identical build를 시도하고, 실패하면 archive timestamp와 file order 같은 packaging 차이를 tensor/kernel 차이와 분리한다. 다음으로 golden kernel input의 output tolerance, collective smoke test와 짧은 training trace를 비교한다. 어느 층까지 같았는지 등급으로 보고한다.

### 데이터와 tokenizer provenance

dataset manifest는 source URI만이 아니라 fetched digest, license snapshot, extraction tool, filter decision, dedup group과 split assignment를 가진다. 웹 원문이 바뀌거나 사라져도 사용한 bytes를 식별할 수 있어야 한다. 삭제 권한이 필요한 원문은 접근 제한 저장소에 두고 책이나 공개 manifest에는 digest와 허용 metadata만 남긴다.

tokenizer 학습은 corpus selection, normalization, trainer algorithm, vocabulary size, reserved tokens와 seed를 material로 가진다. export 뒤에는 vocabulary/token ID mapping, normalizer와 pre-tokenizer graph를 hash한다. tokenizer를 바꾸고 weight embedding을 resize했다면 parent model에서 새 artifact로 이어지는 derivation을 만든다.

data filter나 dedup 코드를 수정했는데 output digest가 우연히 같더라도 새 builder invocation은 보존한다. 반대로 output이 달라졌는데 config diff가 없다면 nondeterminism이나 숨은 dependency를 찾는다. provenance는 성공 실행뿐 아니라 예상 밖 divergence를 조사하는 도구다.

### poisoning과 secret 대응 결정 트리

scanner alert가 나면 먼저 bytes와 detector revision을 고정하고 artifact promotion을 멈춘다. secret이면 유효 credential인지 판단하기 전에도 노출 가능성을 가정해 rotation하고 log·cache·dataset descendant를 찾는다. PII면 법적·정책적 처리 경로와 학습 소비 범위를 확인한다. poisoning 의심이면 출처 신뢰도와 비정상 pattern, canary behavior를 별도로 조사한다.

false positive로 결론 내릴 때도 근거 span, reviewer와 예외 만료를 남긴다. allowlist가 glob 하나로 전체 source를 통과시키지 않게 최소 범위로 둔다. detector가 놓친 사례는 이미 만든 artifact를 그냥 유지하지 말고 소비 checkpoint와 파생 export까지 영향 분석한다.

모델 행동 이상에서 시작한 경우에는 training row로 역추적할 수 없는 가능성도 남긴다. code compromise, base checkpoint substitution, reward corruption을 병렬 가설로 둔다. 한 개의 의심 문서를 찾았다고 원인을 확정하지 않는다.

**변조 실험 묶음**

golden artifact 복사본에서 shard 한 byte, safetensors header offset, shard index mapping, config hidden size, tokenizer special token과 chat template를 각각 하나씩 바꾼다. 모든 변조는 download 직후 hash 또는 schema gate에서 load 전에 거부돼야 한다. 여러 항목을 동시에 바꾸면 어느 gate가 효과적이었는지 알 수 없으므로 한 번에 한 fault만 넣는다.

다음 묶음은 실행 경계다. unsigned remote code, 폐기된 signer, 만료 certificate, 승인되지 않은 builder와 잘못된 transparency entry를 넣는다. network가 차단된 sandbox에서 remote code가 외부 접속을 시도하는지도 관측한다. 실패 때 이전 cache로 조용히 fallback하면 테스트 실패다.

마지막은 운영 경계다. object-store partial upload, node의 오래된 cache, running process의 stale digest와 revocation event 누락을 주입한다. control plane의 desired digest와 각 replica의 loaded digest가 일치할 때만 완료로 판정한다. 복구 뒤 동일 golden inference와 tokenizer trace를 다시 실행한다.

**promotion 상태 기계**

artifact 상태는 `ingested`, `hashed`, `scanned`, `schema-verified`, `sandbox-tested`, `signed`, `approved`, `promoted`, `revoked`로 나눈다. 각 전이는 필요한 증거와 actor를 명시한다. 하나의 관리자 권한으로 build·approve·promote를 모두 수행하지 않는다. 실패 artifact는 격리하되 조사 증거가 사라지지 않게 retention을 둔다.

promotion은 immutable digest를 environment manifest에 기록하는 행위다. staging에서 production으로 파일을 다시 package하면 새 artifact이므로 재검증한다. 배포 도중 일부 replica만 새 digest를 load하면 혼합 상태를 metric과 log에 노출하고 허용 시간 이후 자동 중단한다.

rollback도 승인된 이전 digest로의 새 deployment다. “last good” 별칭을 해석하지 않고 정확한 digest와 당시 호환 config를 사용한다. 취약점 때문에 폐기된 artifact를 가용성 복구용으로 되살리지 못하도록 verify policy가 rollback에도 적용돼야 한다.

**출처와 실험의 증거 등급**

형식의 보안 성질은 safetensors specification과 loader source의 고정 revision에서 확인한다. 실제 missing key 처리, remote code path, cache resolution은 사용하는 Transformers와 Hub client source 및 test에서 따라간다. 서명과 provenance 구조는 in-toto·SLSA 계열 공식 규격에서 가져오되, 그 규격 준수가 모델 행동 안전을 보장한다고 확대 해석하지 않는다.

공식 문서는 계약을, source는 현재 구현을, upstream test는 저자가 고정한 사례를 보여준다. 우리 artifact의 검증 결과는 로컬 변조 실험과 release log가 보여준다. 네 증거를 같은 문장에 섞지 않고 출처·commit·실행 환경을 붙인다. 실행하지 않은 CUDA 조합이나 remote-code 모델은 지원된다고 단정하지 않는다.

재현성 보고에는 최초 불일치 byte와 계층, 허용 tolerance, 반복 횟수와 환경 digest를 남긴다. statistically comparable만 확인한 run을 sample-exact라고 부르지 않는다. 요구 등급에 못 미치면 원인과 잔여 위험을 release exception으로 올린다.

**공급망 완료 체크리스트**

완료 조건은 모든 입력과 출력이 immutable digest를 갖고 builder invocation으로 연결되는 것이다. model weights뿐 아니라 tokenizer, template, custom code, dataset, config, container, native library와 driver가 포함돼야 한다. load 전에 hash·signature·schema가 검증되고, remote code와 pickle은 격리 정책을 통과해야 한다.

SBOM은 runtime load inventory와 맞아야 하며 signer의 trust root, 만료·rotation·revocation 절차가 시험돼야 한다. data나 key 폐기가 adapter와 quantized export, node cache까지 전파되는지 장애 주입으로 확인한다. production replica는 자신이 실제 load한 digest를 지속적으로 보고해야 한다.

마지막으로 재빌드와 golden run을 수행해 요구 재현성 등급을 판정한다. 모든 assertion의 log와 증명 digest를 release record에 묶는다. 이 질문에 답하지 못하면 같은 모델 이름을 다시 받았을 뿐, 같은 시스템을 재현한 것이 아니다.

**키 침해 훈련**

서명 key가 침해됐다고 가정하면 먼저 발급과 사용 log에서 영향 시간창을 정한다. 그 기간의 모든 attestation과 artifact를 graph로 찾고 production inventory와 교차한다. key를 폐기하는 것만으로 이미 load된 artifact가 내려가지는 않는다. control plane 차단, replica drain, cache purge와 대체 digest promotion을 순서대로 수행한다.

재검증은 유효한 signature가 있다는 이유로 끝나지 않는다. 침해되지 않은 source mirror, builder log와 transparency record를 이용해 material과 subject를 다시 확인하거나 clean builder에서 rebuild한다. 동일 bytes가 나오면 별도 approver가 새 provenance를 발급한다. 나오지 않으면 최초 divergence를 조사하고 quarantine을 유지한다.

훈련의 측정값은 detection time, affected descendant 탐색 시간, stale replica 수, rotation 완료 시간과 재승인 근거다. 연락망과 수동 예외가 실제로 작동했는지도 기록한다. 훈련용 key와 production key를 혼동하지 않도록 환경 trust root를 분리한다.

**cache와 mirror의 함정**

artifact mirror는 외부 장애와 삭제에 대비하지만 오래된 취약 bytes를 보존하는 장소가 될 수 있다. mirror entry는 upstream revision, fetched time, content digest, scanner 결과와 revocation state를 가진다. revalidation 주기를 두고 upstream alias를 다시 해석하지 않는다. 삭제 요청과 license 변화도 metadata event로 처리한다.

node cache key에 model 이름만 쓰면 같은 이름의 새 revision과 충돌한다. 산출물 digest와 loader/config compatibility를 key에 넣는다. cache hit 때도 signature와 revocation 상태를 확인하고, process가 mmap한 파일은 디스크 삭제 뒤에도 살아 있음을 고려한다. loaded digest heartbeat로 실제 상태를 본다.

오프라인 cluster에는 폐기 목록과 새 trust bundle이 늦게 도달할 수 있다. 만료 후 fail-open할지 fail-closed할지를 위험 등급별로 정하고, bundle age를 metric으로 경보한다. 비상 artifact도 사전에 서명·검증된 digest 목록에서만 선택한다.

**재현 실패를 디버깅한다**

재빌드 결과가 다르면 먼저 file tree와 manifest를 diff한다. 파일 목록이 같으면 header·archive timestamp·JSON key order 같은 비의미 차이를 분리하고, tensor별 hash와 native binary section을 비교한다. tensor가 처음 달라진 training step을 찾기 위해 작은 deterministic golden run과 checkpoint ladder를 사용한다.

동일 입력인데 step 0부터 다르면 initialization seed, parameter order, tokenizer/data bytes와 kernel 선택을 본다. 여러 step 뒤 갈라지면 data cursor, collective order, nondeterministic operation과 fault recovery를 확인한다. final metric만 비슷하다고 divergence를 무시하지 않고 요구 등급에 따라 허용 여부를 결정한다.

환경 비교는 package version 표보다 실제 load evidence를 우선한다. driver, GPU SKU, clock/precision 설정, shared object digest와 environment variable의 allowlisted subset을 비교한다. secret이 든 전체 environment dump를 provenance에 넣지 않는다.

**소비자 관점의 검증**

artifact 생산자가 올바른 증명을 만들었어도 serving과 fine-tuning consumer가 검증하지 않으면 보호가 없다. loader wrapper는 expected digest와 policy를 입력받고, 검증 완료 전 GPU memory allocation이나 custom import를 시작하지 않는다. audit log에는 요청 artifact, 검증 규칙, 결과와 loaded digest를 남긴다.

fine-tuning job은 base checkpoint뿐 아니라 dataset과 training container의 승인을 확인한다. serving job은 quantized export, tokenizer, runtime extension과 generation config의 호환 bundle을 확인한다. 구성 요소를 개별 승인했어도 승인되지 않은 조합이면 integration golden test를 요구한다.

개발자는 로컬 파일을 실험할 수 있지만 production promotion과 같은 이름 공간을 쓰지 않는다. 예외 artifact에는 시각적 경고보다 machine-enforced environment restriction과 만료가 필요하다. 검증 실패를 무시하는 flag는 production role에서 권한 자체를 제거한다.

**최종 release 기록**

release record에는 subject digest 집합, provenance와 signature 검증 결과, SBOM과 vulnerability snapshot, schema·sandbox·golden test, 재현성 등급과 승인자가 들어간다. 알려진 예외에는 위험, 보완 통제, owner와 만료 날짜를 기록한다. 모델 카드와 배포 문서는 이 record를 가리키되 같은 역할을 대신하지 않는다.

release 뒤에는 runtime digest inventory, 새 취약점·폐기 event와 data deletion 요청을 감시한다. 영향 graph가 descendant를 찾으면 자동으로 owner에게 case를 열고 필요한 경우 rollout을 중단한다. release는 공급망 검증의 끝이 아니라 관측 가능한 수명 주기의 시작이다.

독자는 이 기록에서 “어디서 받았는가”를 넘어 “어떤 bytes가 누구의 어떤 builder에서 무엇으로 만들어졌고, 현재 왜 신뢰되는가”에 답할 수 있어야 한다. 답이 alias나 구두 확인에 머물면 promotion gate를 통과시키지 않는다.

**라이선스와 사용 조건도 계보다**

code, dataset, base model과 adapter의 라이선스는 서로 다른 제약을 가질 수 있다. artifact manifest에는 사용한 license text의 digest와 확인 시점을 넣는다. 웹 페이지의 현재 문구만 가리키면 학습 당시 조건을 복원할 수 없다. model card의 license tag는 시작점이며 실제 배포·재배포·상업 이용 조건을 별도로 검토한다.

dataset의 source별 조건이 다르면 aggregate 이름 하나로 덮지 않는다. 삭제·귀속·비상업 조건을 document lineage와 연결해 descendant 영향 분석에 사용한다. 법적 판단은 담당 검토자가 내리되 기술 시스템은 어떤 material이 어디에 쓰였는지 답할 수 있어야 한다.

조건이 바뀌거나 잘못 분류된 source가 발견되면 보안 취약점과 마찬가지로 case를 열고 promotion을 중단한다. 대체 데이터로 재학습할지, 특정 배포를 제한할지, artifact를 폐기할지는 검토 결과에 따른다. license scanner의 통과를 최종 법적 승인으로 오해하지 않는다.

**백업과 재난 복구**

재현 가능한 manifest가 있어도 material과 builder가 모두 사라지면 rebuild할 수 없다. source mirror, dataset snapshot, container와 toolchain, 서명된 provenance를 서로 다른 failure domain에 백업한다. 암호화 key와 복구 권한은 artifact와 분리하고 정기적으로 restore drill을 한다.

restore는 파일 수가 맞는지만 보지 않는다. digest, signature, graph edge와 access policy를 검증하고 격리 환경에서 golden build/load를 수행한다. 오래된 backup이 폐기된 artifact나 compromised trust root를 되살리지 않도록 최신 revocation bundle을 먼저 적용한다.

재난 복구 목표에는 bytes 복원 시간뿐 아니라 승인된 serving 상태로 돌아가는 시간과 허용 provenance 손실을 포함한다. drill 결과에서 수동 단계, 누락 credential과 stale cache를 기록해 runbook을 수정한다. 복구 편의를 위해 검증 단계를 생략하는 경로는 만들지 않는다.

## 27.5 loader 분기를 고정 소스와 음성 fixture로 감사한다

문서의 안전 주장보다 고정 revision에서 실제 loader가 어떤 branch를 타고 무엇을 실행하는지 확인한다.

### safetensors는 실행 방지와 구조 검증을 분리한다

safetensors의 장점은 “파일이면 안전하다”는 인증이 아니라 역직렬화할 임의 객체 그래프를 tensor metadata와 연속 byte 영역으로 제한한 데 있다. revision `6eb4dc9a28ebce297606e0f4836bbf28839cacef`의 [형식 설명](https://github.com/huggingface/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/README.md#L76-L94)은 첫 8 byte의 header 길이, JSON header, `dtype`, `shape`, 상대 `data_offsets`를 구분한다. 검증기는 확장자나 MIME type이 아니라 이 관계를 검사해야 한다.

같은 revision의 [`SafeTensors::read_metadata`](https://github.com/huggingface/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L390-L426)는 header 길이가 buffer 범위 안인지 확인하고 JSON을 `Metadata`로 변환한 뒤 파일 전체가 metadata가 주장한 구간을 포함하는지 검사한다. 이어지는 [`Metadata::new`](https://github.com/huggingface/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L599-L649)는 offset 순서와 빈틈·겹침, shape와 dtype에서 계산한 byte 수를 대조한다. 두 단계는 구분해야 한다.

JSON 파싱 성공은 tensor 구간의 구조적 일관성을 증명하지 않으며, 구조적 일관성은 해당 tensor가 기대한 모델의 weight라는 사실을 증명하지 않는다.

production loader 앞에는 세 겹의 gate가 필요하다. 첫째, 외부 manifest의 digest와 signature로 받은 bytes의 identity를 확인한다. 둘째, 형식 parser로 header와 tensor interval을 검증한다. 셋째, model config에서 유도한 expected key·shape·dtype·tying contract와 비교한다. 예를 들어 hidden size를 바꾼 config와 원래 weight를 각각 유효한 파일로 제공해도 세 번째 gate에서 거부돼야 한다.

**실패 주입 27-A.** header length를 실제 JSON보다 크게 만들고, offset이 겹치는 두 tensor와 shape product가 data length와 다른 tensor를 각각 만든다. 어느 오류가 digest, parser, schema gate 중 어디서 검출됐는지 기록한다. 여러 결함을 한 파일에 넣으면 가장 앞선 오류만 보여 방어층의 독립성을 증명하지 못한다.

**코드 워크스루—유효한 JSON을 유효한 tensor로 착각하지 않는다.** revision `6eb4dc9a28ebce297606e0f4836bbf28839cacef`, `safetensors/src/tensor.rs`의 [`SafeTensors::read_metadata`](https://github.com/huggingface/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L390-L425)는 `buffer:[u8; B]`의 첫 8 byte를 little-endian header 길이 `n`으로 읽고 `buffer[8:8+n]`만 UTF-8 JSON으로 parse한다.

출력은 `(n, Metadata)`이지만, 그 전에 `Metadata::validate`가 반환한 data 끝이 실제 `B-8-n`과 정확히 같은지 본다. 따라서 뒤에 붙인 polyglot payload와 잘린 tensor data는 둘 다 `MetadataIncompleteBuffer`로 차단된다.

```rust
let metadata: Metadata = metadata.try_into()?;
let buffer_end = metadata.validate()?;
if buffer_end + N_LEN + n != buffer_len {
    return Err(SafeTensorError::MetadataIncompleteBuffer);
}
```

shape·dtype 계약은 다음 층이다. [`Metadata::validate`](https://github.com/huggingface/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L627-L664)는 각 `TensorInfo {dtype, shape, data_offsets:(s,e)}`에 대해 첫 tensor는 `s=0`, 뒤 tensor는 `s=이전 e`인지 검사한다. 이어 overflow를 검출하며 `∏ shape × dtype.bitsize()/8 == e-s`를 확인한다.

`I32`, shape `[2,2]`의 예상 data는 16 byte다. offset `[0,4]`를 주면 JSON과 buffer가 parse되더라도 `TensorInvalidInfo`가 첫 불일치가 된다. offset `[0,16]`인 tensor 열 개가 같은 16 byte를 가리키면 첫 tensor 다음의 `s != start`에서 `InvalidOffset`이 된다. JSON map 순서 때문에 어느 tensor 이름이 먼저 보고될지는 변할 수 있으므로 error class를 oracle로 삼고 이름은 고정 oracle로 삼지 않는다.

이 변형은 upstream test와 직접 연결된다.

- [`test_json_attack`](https://github.com/huggingface/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L1412-L1453)은 중첩 offset을 `InvalidOffset`으로 판정한다.
- [`test_metadata_incomplete_buffer`](https://github.com/huggingface/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L1455-L1473)는 추가·누락 byte를 `MetadataIncompleteBuffer`로 판정한다.
- [`test_invalid_info`](https://github.com/huggingface/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L1563-L1571)는 `[2,2] I32`와 4-byte interval의 불일치를 `TensorInvalidInfo`로 assertion한다.

현장 verifier의 변형 fixture는 한 파일에 세 결함을 섞지 말고 정상 16 byte→14 byte truncation→16 byte+suffix→offset overlap→shape `[2,2]`/interval 4 byte를 각각 독립 케이스로 둔다. 예상 출력은 tensor value가 아니라 최초 거부 단계와 error class다.

이 test들은 parser 계약을 실행한 근거지만 외부 digest·서명이나 model key·shape schema를 검증하지는 않는다. parser test 통과를 artifact manifest 검증 완료로 확대하지 않아야 하는 이유가 여기에 있다.

### `weights_only`는 pickle을 safetensors로 바꾸지 않는다

PyTorch revision `3691693263d2b66a68867e39b7449876844e06cf`의 [`torch.load`](https://github.com/pytorch/pytorch/blob/3691693263d2b66a68867e39b7449876844e06cf/torch/serialization.py#L1320-L1418)는 `weights_only`일 때 허용된 tensor·primitive·등록 global로 unpickler의 능력을 제한한다. 같은 파일의 [`add_safe_globals`](https://github.com/pytorch/pytorch/blob/3691693263d2b66a68867e39b7449876844e06cf/torch/serialization.py#L284-L317)는 사용자가 허용 대상을 확장할 수 있음을 보여 준다.

`weights_only=True`는 모든 checkpoint를 무조건 안전하게 만드는 암호학적 검증이 아니며, allowlist 변경도 실행 정책의 일부다.

optimizer checkpoint를 복원할 때 custom tensor subclass나 scheduler 객체 때문에 allowlist를 추가하고 싶은 유혹이 생긴다. 이때 오류 메시지에 나온 global을 자동 등록해서는 안 된다. 먼저 격리된 환경에서 파일 digest를 고정하고, 필요한 상태를 primitive와 tensor schema로 변환할 수 있는지 본다. 불가피하게 global을 허용하면 정확한 module·symbol·source digest, 호출 가능한 constructor의 side effect, 허용 종료 시점을 manifest에 기록한다.

`weights_only=False` fallback은 편의 기능이 아니라 신뢰 경계 변경이다. production credential과 network가 있는 학습 process에서 재시도하지 않는다.

disposable converter가 원본을 읽고, 제한된 tensor artifact를 새 subject로 만들며, 변환 코드·원본 digest·출력 digest를 provenance edge에 남긴다. 변환 성공 뒤에도 tensor schema와 golden state-restore 시험을 별도로 수행한다.

**실패 주입 27-B.** 허용되지 않은 global을 포함한 checkpoint를 넣고 제한 loader가 거부하는지 본다. 다음에는 test 전용 allowlist로만 통과시킨 뒤 process·network·filesystem syscall을 관찰한다. 마지막으로 allowlist가 다음 load에 남지 않는지 확인한다. 이 실험의 목적은 악성 코드를 실행하는 것이 아니라 정책 수명이 예상보다 길어지는 결함을 찾는 것이다.

### remote code는 설정값이 아니라 import 상태 전이다

Transformers revision `550d7b3834670483a4df436541272c055dc364bf`의 [`resolve_trust_remote_code`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/dynamic_module_utils.py#L712-L794)는 local implementation의 존재, remote implementation의 존재, 사용자 선택을 조합해 최종 boolean을 만든다.

[`AutoConfig.from_pretrained`](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/models/auto/configuration_auto.py#L371-L418)는 config의 `auto_map`을 읽고 이 결정을 실제 dynamic class load 분기로 연결한다.

manifest에는 단순히 `trust_remote_code=true`가 아니라 어떤 repository revision의 어떤 Python 파일과 class가 선택됐는지를 남겨야 한다.

native implementation이 새 라이브러리 version에 추가되면 같은 model repository도 이전에는 remote path, 이후에는 local path로 갈 수 있다. weight와 config digest가 같아도 실행 함수가 달라지는 셈이다. 라이브러리 upgrade 검증에서 resolved class의 module path와 source digest를 golden record와 비교하고, 변경됐으면 logits·gradient·save/load parity를 다시 확인한다.

tokenizer와 image processor도 같은 동적 경로를 가질 수 있다. model class만 검토하고 tokenizer remote code를 놓치면 normalization과 token ID가 달라질 수 있다. artifact bundle의 모든 `auto_map` entry를 열거하고 import 전에 source tree digest, dependency allowlist와 sandbox 정책을 적용한다.

**실패 주입 27-C.** local code가 있는 환경과 없는 환경에서 같은 config를 resolve해 선택 class가 바뀌는지 기록한다. remote revision을 고정하지 않은 fixture는 의도적으로 거부한다. 승인된 remote code가 network에 접근하거나 작업 디렉터리 밖에 쓰기를 시도하면 sandbox event와 함께 promotion을 중단한다.

**세 로더를 하나의 승인 상태 기계로 묶는다**

다운로드 완료는 `received`일 뿐 `loadable`이 아니다. `digest-verified→signature-verified→format-verified→schema-verified→code-resolved→sandbox-tested→golden-tested` 순서를 manifest에 기록한다. 앞 단계의 결과와 policy version이 없으면 뒤 단계를 실행하지 않는다. cache hit도 이 상태 기계를 우회하지 않으며, revocation list와 trust bundle의 최신성을 다시 확인한다.

디버깅할 때도 최초 불일치 층을 지킨다. digest가 다르면 parser 오류를 분석하기 전에 substitution·partial download를 조사한다. digest는 맞고 parser가 실패하면 expected manifest 자체가 잘못 승인됐을 가능성을 연다. parser는 맞고 schema가 다르면 config·index·shard 세대를 대조한다. 모든 정적 gate를 통과했지만 golden output이 다르면 resolved class, tokenizer graph, native library와 device kernel을 본다.

특정 형식을 맹신하지 말아야 한다. safetensors는 객체 실행면을 줄이고, `weights_only`는 pickle unpickler의 허용 능력을 줄이며, remote-code opt-in은 dynamic import를 명시한다. 세 기능은 서로 다른 위험을 다루므로 어느 하나로 나머지를 대체할 수 없다. release 승인자는 각 gate가 실제 bytes와 실제 실행 경로에 적용됐다는 증거를 요구해야 한다.

승인 회의에서는 다음 질문에 답한다. 왜 이 digest가 요청한 모델의 완전한 bundle임을 믿는가. 왜 서명자는 이 환경과 용도에 대한 권한을 갖는가. 왜 parser 통과가 schema와 의미의 일치로 확대 해석되지 않았는가. 왜 remote class가 native class 대신 선택됐는가. 왜 allowlist에 추가한 global이 꼭 필요하며 언제 제거되는가. 왜 golden test가 실제 배포의 tokenizer·library·GPU 경로를 대표하는가. 답은 사람의 기억이 아니라 소스 좌표, policy result와 산출물 digest를 가리켜야 한다.

**이 장이 넘기는 것.** 서명된 run/artifact manifest, SBOM, `RevocationID`, 허용 재현성 등급과 loader 상태 전이를 28장의 golden run에 넘긴다. 29장의 failure injection에는 cache substitution·stale trust bundle 결함을 인계하고, 30장의 출시 관문에는 각 상태 전이의 증명 digest를 넘긴다.

## 27.6 provenance와 서명을 training invocation에 묶는다

SLSA와 Sigstore를 이름표처럼 붙이지 않고 어느 invocation이 어떤 input으로 artifact를 만들었는지 검증 가능한 statement로 구체화한다.

공급망 증명은 “누가 모델을 만들었다”는 서술문이 아니다. 검증기가 계산할 수 있는 subject, builder, build definition, external parameters, resolved dependencies와 실행 결과의 관계다. SLSA provenance v1의 핵심을 학습에 옮기면 subject는 최종 weight 한 파일만이 아니라 shard index·config·tokenizer·chat template·adapter를 묶은 release bundle digest가 된다. builder는 사람 이름이 아니라 격리된 pipeline identity이고, build definition은 training entrypoint, config와 허용 입력을 가리킨다.

external parameter와 resolved dependency를 구분해야 한다. 사용자가 `dataset=corpus-v3`, `base_model=org/model`이라고 요청한 문자열은 external parameter다. 실제 실행이 해석한 dataset snapshot digest, Git commit, wheel과 container digest는 resolved dependency다. 둘을 함께 남겨야 “무엇을 요청했는가”와 “무엇이 실제 사용됐는가”의 차이를 찾을 수 있다. branch 이름이나 floating model alias만 남기면 당시 resolution을 재현할 수 없다.

local mirror나 cache를 거친 입력도 resolved dependency에서 사라지지 않는다. URI는 위치 힌트일 뿐 identity가 아니므로 digest를 중심에 둔다. 같은 digest가 서로 다른 mirror에서 왔다면 origin과 retrieval event를 별도로 기록한다. 반대로 같은 URI가 다른 digest를 반환하면 새 material이다. 이 원칙을 dataset shard, tokenizer JSON, CUDA wheel, custom extension source 모두에 똑같이 적용한다.

### in-toto statement는 봉투와 내용물을 분리한다

in-toto attestation framework는 statement의 `subject`와 `predicateType`, predicate를 구분한다. 이 저장소에 고정한 [`in-toto Attestation Framework`](https://github.com/in-toto/attestation/tree/051624ce466deaed4c5a66e66877f69b471fccbe/spec/v1)는 statement가 무엇에 대한 주장인지 subject digest로 지정하고, 주장의 schema를 predicate type으로 식별한다. 서명은 이 statement를 DSSE 같은 envelope에 담아 보호한다. 서명된 JSON이 있다고 끝나는 것이 아니라 verifier가 이해하고 허용한 predicate type인지 확인해야 한다.

envelope signature가 유효해도 subject digest가 배포하려는 artifact와 다르면 아무것도 증명하지 않는다. predicate가 SLSA provenance라고 주장하지만 알 수 없는 major version이면 의미를 임의로 추측하지 않는다. builder identity가 유효해도 해당 repository·branch·release channel을 만들 권한이 있는지는 별도 authorization policy가 판단한다. authenticity, semantic validity, authorization은 서로 다른 gate다.

학습 run에는 여러 attestation을 붙일 수 있다. dataset 정제 pipeline은 정제 snapshot을 subject로 하고 원 raw shard와 filtering code를 material로 삼는다. tokenizer build는 tokenizer bundle을 subject로 하고 corpus sample·normalizer code·trainer config를 material로 삼는다. training provenance는 weight bundle을 subject로 하고 그 dataset·tokenizer attestation subject를 dependency로 가리킨다. 평가 attestation은 metric report를 subject로 하되 평가한 model digest와 harness digest를 predicate에 고정한다.

그래프가 이렇게 층을 이뤄야 한 증명을 거대한 자유 형식 문서로 만들지 않고도 end-to-end lineage를 검증할 수 있다.

### build service의 신뢰 수준을 과장하지 않는다

재현 가능한 명령을 적었다고 builder가 신뢰되는 것은 아니다. 개발자 노트북에서 실행한 shell script는 입력과 출력 digest를 남길 수 있지만 credential 탈취, 실행 중 source 변경, 결과 교체를 막았다는 증거는 약하다. 반면 ephemeral runner, hermetic input mount, network 차단, workload identity와 immutable log를 가진 builder는 더 강한 주장을 할 수 있다. release policy는 provenance 존재 여부만 보지 말고 builder class를 허용 목록과 대조한다.

GPU training은 완전한 hermetic build가 어렵다. scheduler, driver, firmware, NCCL topology와 object store가 외부 환경이다. 이 사실을 숨기기보다 environment dependency로 명시하고 관찰된 version과 digest를 남긴다. deterministic reproduction을 약속할 수 없다면 sample identity·config·code·산출물 계보가 같은 provenance reproduction, metric tolerance 안에 드는 statistical reproduction 등 등급을 나눈다. 약한 등급을 강한 표현으로 포장하지 않는 것이 공급망 정직성의 일부다.

### Sigstore 검증은 서명 하나를 확인하는 절차가 아니다

Sigstore 계열의 keyless signing은 짧은 수명의 certificate에 OIDC identity를 결합하고 transparency log에 signing event를 남긴다. 검증자는 artifact signature의 수학적 유효성뿐 아니라 certificate issuer, subject 또는 workflow identity, 유효 시간, transparency-log inclusion과 trust root를 확인한다. “GitHub Actions가 서명했다”가 아니라 어느 repository의 어느 workflow가 어느 ref 조건에서 서명할 권한이 있는지를 policy로 적는다.

certificate의 identity가 email 또는 URI로 표현될 때 문자열 prefix만 비교하면 유사 repository나 다른 workflow를 허용할 수 있다. issuer와 subject의 정확한 조합, workflow path, protected tag 또는 branch 조건을 검증한다. 재사용 가능한 workflow가 실제 caller identity를 어떻게 전달하는지도 확인한다. release job과 nightly job이 같은 keyless issuer를 쓴다는 이유로 같은 권한을 주지 않는다.

**online verification과 bundle verification**

항상 외부 transparency service에 접근할 수 없는 학습 클러스터에서는 서명 bundle을 artifact와 함께 보존한다. bundle에는 검증에 필요한 certificate chain, log entry와 inclusion evidence가 포함되어야 한다. 단, offline이라고 trust root를 영구 고정해서는 안 된다. 신뢰 root version과 갱신 시각을 manifest에 기록하고, 연결이 회복되면 최신 revocation·root 정책으로 재검증한다.

air-gapped mirror에 artifact를 반입할 때 경계 밖에서 한 verification 결과만 복사하지 않는다. 반입 bundle과 bytes를 내부 gate가 다시 검증하고 내부 catalog에 새 reception attestation을 만든다. 외부 signature는 artifact의 출처를, 내부 attestation은 검증 시점과 적용한 policy를 증명한다. 둘을 하나로 합치면 어느 신뢰 경계에서 어떤 판단을 했는지 사라진다.

transparency log inclusion은 artifact가 안전하다는 보증이 아니다. 특정 identity가 특정 digest에 서명했다는 사실을 감사 가능하게 만든다. compromised workflow가 악성 artifact를 정상적으로 서명할 수 있으므로 source review, builder isolation, schema·golden test가 여전히 필요하다. transparency는 사고 후 영향 범위와 서명 event를 찾는 능력을 높이는 장치다.

**키와 identity가 침해됐을 때**

침해 대응은 서명을 삭제하는 것으로 끝나지 않는다. 침해 window와 signer identity를 기준으로 transparency log와 internal catalog에서 모든 subject를 찾고 `RevocationID`를 발행한다. 그 artifact를 material로 사용한 checkpoint, adapter, quantized derivative와 serving image를 provenance DAG의 후손 탐색으로 구한다. 찾은 후손을 `QUARANTINED`로 전이하고 load cache와 배포 node에서 차단한다.

재서명은 복구가 아니다. 같은 의심 byte에 새 서명을 붙이면 lineage만 세탁한다. 신뢰할 수 있는 source commit과 material snapshot에서 깨끗한 builder로 재빌드하고, 새 subject digest와 새 provenance를 만든다. 재빌드 결과가 bitwise 같더라도 새 verification event와 builder identity를 남긴다. 결과가 다르면 차이를 dependency·toolchain·nondeterminism 층에서 설명한다.

## 27.7 cache·SBOM·bundle에서 실제 적재 identity를 증명한다

다운로드 cache, package lock과 model bundle은 각기 다른 질문에 답한다. 실행된 byte와 의미를 확인하려면 세 원장을 결합해야 한다.

Hub에서 `revision`을 생략한 `from_pretrained` 호출은 편리하지만 release 입력으로는 불충분하다. branch나 tag는 움직일 수 있고, repository snapshot 안에서도 weight, config, tokenizer와 remote code가 함께 해석된다. 다운로드 단계에서 commit SHA를 resolve하고 허용 파일 목록과 각 digest를 bundle manifest로 만든다. 학습 process에는 network resolution 권한을 주지 않고 검증된 local snapshot path만 전달하는 편이 경계를 분명하게 만든다.

cache는 blob을 content-addressed 형태로 중복 제거할 수 있지만, cache hit가 승인 상태를 뜻하지 않는다. 과거에 받은 blob이 현재 폐기됐거나 당시에는 다른 policy로 허용됐을 수 있다. loader는 cache에서 byte를 찾은 뒤에도 요청한 repository commit의 snapshot edge와 digest가 맞는지, 현재 revocation과 signature policy를 통과하는지 확인한다. cache directory의 파일명이나 symlink만 신뢰하지 않는다.

### snapshot 완전성

sharded weight는 index JSON이 tensor key를 여러 shard에 매핑한다. index digest만 고정하거나 shard만 따로 고정하면 완전성을 증명하지 못한다. index가 참조하는 shard 집합이 manifest와 정확히 같고 각 shard digest가 맞으며, 예상하지 않은 executable Python이나 binary가 snapshot에 추가되지 않았는지 검사한다. config의 `auto_map`, tokenizer class와 custom processor 선언도 code dependency 목록에 반영한다.

Git LFS pointer text를 실제 대형 blob으로 오인하는 실패도 방지한다. 허용 형식 parser, 최소·예상 byte size와 digest를 사용하고, download 중단으로 생긴 partial file이 final cache namespace에 나타나지 않게 임시 경로와 atomic promotion을 쓴다. object store mirror는 upstream commit SHA에서 내부 snapshot ID로 가는 signed mapping을 제공해야 한다.

local-files-only 모드는 network fetch를 막는 데 유용하지만 그 자체가 정확한 revision 선택을 보장하지 않는다. cache에 여러 revision이 있거나 requested ref mapping이 stale할 수 있다. 실행 manifest에 resolved snapshot directory와 commit, blob digest를 기록하고 다음 run에서 같은 path가 아니라 같은 identity를 검증한다. cache GC 뒤 path가 달라도 동일 digest bundle이면 같은 material일 수 있다.

### 캐시 오염 실패 주입

첫 시험은 snapshot symlink가 다른 유효한 blob을 가리키도록 바꿔 digest gate가 잡는지 본다. 둘째는 index에 새 shard를 추가하되 manifest에는 없는 상태를 만들어 exact-set 검사 여부를 본다. 셋째는 같은 branch ref를 새 commit으로 움직여 floating revision을 release policy가 거부하는지 본다. 넷째는 폐기된 digest가 cache에 남은 상태에서 offline loader가 stale trust bundle로 허용하지 않는지 본다.

다섯째는 두 process가 같은 blob을 동시에 다운로드하다 하나가 죽는 race다. 완료 marker 전에 partial bytes가 snapshot에서 발견되지 않아야 한다. 여섯째는 mirror가 404를 반환해 public Hub로 자동 fallback하는 설정이다. production에서는 이 fallback이 새로운 신뢰 경계와 데이터 유출 경로가 되므로 명시적으로 실패시키고 operator 승인 뒤 별도 retrieval job을 실행한다.

### SBOM은 Python package 목록보다 넓어야 한다

학습 container의 `pip freeze`는 Python 배포판의 이름과 version을 보여 주지만 실제 실행된 native code를 충분히 설명하지 못한다. PyTorch wheel 안의 CUDA library, system glibc와 libstdc++, NCCL, driver interface, JIT 또는 build된 custom extension, Triton cache, compiler와 linker가 결과와 보안면에 영향을 준다. SBOM은 package URL 또는 CPE, file digest, dependency relationship과 license를 포함하고 container layer와 host-mounted library를 구분한다.

SPDX나 CycloneDX 형식을 고르는 일보다 실제 배포 byte를 빠뜨리지 않는 일이 중요하다. build-time SBOM과 runtime inventory를 대조한다. container image에는 없지만 Kubernetes volume이나 host path에서 주입된 library, job 시작 뒤 컴파일된 `.so`, 다운로드된 kernel binary를 runtime supplement에 넣는다. 단순히 base image의 SBOM을 상속하면 학습 job이 만든 실행물을 놓친다.

**CUDA native extension의 재현 단위**

native extension identity에는 source commit, generated source, compiler와 nvcc version, compile flags, target compute capabilities, linked library digest와 최종 `.so` digest가 필요하다. `TORCH_CUDA_ARCH_LIST`, fast-math, line info, ABI flag와 C++ standard가 달라지면 같은 Python version에서도 binary가 달라진다. PTX만 포함했는지 특정 SM의 cubin을 포함했는지도 기록한다. 새 driver가 PTX를 JIT하면 driver version과 JIT cache가 실제 실행 경로에 들어온다.

reproducible build를 원하면 timestamp, absolute build path, nondeterministic archive ordering 같은 비의미 차이를 제거한다. 그래도 binary digest가 다르면 symbol table과 section 단위로 비교하고, kernel SASS 또는 PTX 차이가 수치 경로에 영향을 주는지 golden test로 간다. digest 불일치를 “컴파일은 원래 다르다”로 무시하면 source substitution과 benign nondeterminism을 구분할 수 없다.

prebuilt wheel의 filename은 identity가 아니다. wheel 자체 digest, 내부 RECORD, platform tag와 서명을 확인한다. runtime GPU가 wheel이 포함하지 않은 architecture라 JIT fallback 또는 다른 kernel path를 선택한다면 environment manifest와 실행 trace에 남긴다. 같은 wheel을 썼다는 사실만으로 같은 kernel이 실행됐다고 결론 내리지 않는다.

**취약점과 실행 가능성**

SBOM scanner가 CVE를 찾으면 package 이름 일치만으로 즉시 위험도를 정하지 않는다. 해당 component와 version이 실제 포함됐는지, 취약 symbol 또는 code path가 reachable한지, untrusted input이 그 경로에 닿는지 분석한다. 하지만 “Python에서 직접 import하지 않았다”는 이유만으로 native transitive library를 제외하지 않는다. 동적 linker와 plugin loader가 런타임에 불러올 수 있다.

완화 patch를 적용하면 새 image와 wheel digest, 재빌드 provenance를 만든다. 기존 모델 weight가 취약 library로 만들어졌다는 사실이 weight byte를 반드시 오염시키는 것은 아니지만, build integrity가 침해됐을 가능성과 serving image의 직접 취약성은 별도로 판정한다. 영향 범위는 artifact 종류와 provenance edge type에 따라 전파해야 한다.

**safetensors bundle의 완전성을 증명한다**

safetensors parser가 한 파일의 offset과 shape를 검증해도 여러 shard로 나뉜 모델의 의미적 완전성은 외부 index와 config에 달려 있다. verifier는 index의 `weight_map` key 집합, 각 shard 내부 key 집합, model architecture가 기대하는 state dict key 집합을 비교한다. `missing`, `unexpected`, duplicate key를 각각 분리하고 tied weight처럼 의도적으로 저장을 생략한 경우는 config에서 유도한 규칙으로만 허용한다.

tensor 총 byte 수는 dtype별 element count로 다시 계산한다. quantized model이라면 packed weight, scale, zero point와 group-size metadata의 관계를 확인한다. adapter라면 base model digest, target module, rank와 scaling을 manifest에 넣는다. 형식이 안전해도 잘못된 base에 adapter를 붙이면 의미가 깨진다. loader가 이름이 비슷한 최신 base를 자동 선택하게 두지 않는다.

**dtype와 shape만으로 부족한 이유**

같은 `[hidden, hidden]` shape의 attention projection 네 개는 서로 교체돼도 parser와 shape gate를 통과한다. 의미 검증에는 key identity, architecture mapping과 선택한 tensor의 통계·golden output이 필요하다. positional embedding과 token embedding도 우연히 shape가 같을 수 있다. conversion pipeline은 source key→target key 변환표와 transpose·reshape 연산을 machine-readable하게 남긴다.

checkpoint conversion 뒤에는 round-trip 가능성을 검사한다. source를 target format으로 바꾸고 가능한 범위에서 다시 canonical representation으로 변환해 tensor digest 또는 tolerance를 비교한다. lossy quantization이면 bitwise round-trip을 요구하지 않고 dequantized error bound와 task golden metric을 선언한다. 어떤 보장을 하지 않는지도 conversion attestation에 적는다.

**`weights_only` 정책을 운영 가능한 allowlist로 만든다**

PyTorch의 제한 unpickler는 위험한 객체 생성 능력을 줄이지만 allowlist를 넓히는 순간 그 symbol의 import와 reduce 경로를 신뢰하게 된다. 중앙 policy는 `(module, qualified name, source distribution digest)`를 승인 단위로 삼고, job 코드가 오류 메시지를 읽어 자동 추가하지 못하게 한다. 한 번의 conversion job에 필요한 symbol은 context가 끝나면 제거하고 장기 worker 전역에 남기지 않는다.

legacy optimizer checkpoint 때문에 객체를 읽어야 한다면 network·credential·GPU device가 없는 converter에서 실행한다. filesystem은 입력 read-only와 새 출력 경로만 제공하고 process·syscall을 관찰한다. 출력은 tensor·primitive 중심 schema로 바꾸며 원본 digest와 converter image digest를 새 provenance에 남긴다. 이후 production loader는 변환된 artifact만 읽는다.

**제한 로더의 음성 시험**

정상 tensor checkpoint만 load하는 시험 외에 예상치 못한 global, 동적 import, oversized allocation, 깊은 object graph와 손상된 zip entry를 넣는다. 목표는 실제 payload를 실행하는 것이 아니라 policy가 load 이전에 어떤 상태에서 거부하는지 확인하는 것이다. 오류가 친절하다는 이유로 fallback을 자동화하지 않는다. 제한 mode 실패는 quarantine event다.

allowlist update는 code review, expiry, owner와 사용 artifact 범위를 가진다. source package가 upgrade되면 같은 qualified name이라도 구현 digest가 바뀌므로 재승인한다. 폐기된 allowlist로 이미 변환한 output은 변환 로그와 영향 분석을 거쳐 유지 또는 재변환을 결정한다.

**Transformers remote code의 transitive closure를 계산한다**

`trust_remote_code=True`가 허용하는 것은 한 class 파일만이 아니다. 그 파일의 상대 import, package dependency, import-time side effect와 런타임 download까지 이어질 수 있다. 승인기는 repository revision의 `auto_map` entry에서 시작해 정적 import closure를 만들고, 동적 import와 subprocess·network 사용을 별도 위험으로 표시한다. model, config, tokenizer, processor 각각의 resolved class를 기록한다.

native implementation이 Transformers에 들어온 뒤 remote class 대신 local class가 선택되는 변화도 공급망 사건이다. source가 더 신뢰할 만해졌을 수 있지만 numerical behavior와 state-dict mapping이 같다는 뜻은 아니다. upgrade gate는 resolved class, config normalization, attention implementation, generation config와 save/load 결과를 비교한다. 어느 경로를 택할지 명시하고 자동 우선순위에 맡기지 않는다.

**sandbox는 검토를 대체하지 않는다**

remote code를 container에서 실행하면 host 피해를 줄일 수 있지만 training data와 credential이 같은 sandbox 안에 있으면 정보 유출은 여전히 가능하다. import 검증 단계는 dummy config와 최소 권한으로 수행하고 network를 차단한다. 승인 뒤 학습 단계에서도 dependency lock과 egress policy를 유지한다. runtime에 새 package를 설치하거나 Hub에서 추가 파일을 받으려 하면 중단한다.

정적 분석은 `exec`, reflection, native extension을 완전히 해석하지 못한다. 그래서 source review, sandbox 관찰과 golden behavior를 결합한다. “스캐너가 경고하지 않았다”는 승인 사유가 아니다. 미해석 동적 경로를 위험 목록에 남기고 필요하면 remote implementation 대신 감사된 vendor copy 또는 upstream native implementation을 쓴다.

## 27.8 폐기·재빌드·재현 실패를 그래프에서 처리한다

취약한 부모를 찾은 뒤 어떤 child를 폐기하고 무엇을 재빌드할지 provenance edge를 따라 계산한다.

artifact 폐기는 파일 이름 목록이 아니라 graph query다. compromised material digest, builder identity, vulnerable dependency 또는 잘못된 dataset snapshot을 시작점으로 `derivedFrom`, `builtWith`, `convertedFrom`, `quantizedFrom`, `evaluatedWith`, `deployedAs` edge를 따라 영향을 계산한다. 모든 edge가 같은 전파 의미를 갖지는 않는다. 평가 harness 취약점은 metric report를 폐기할 수 있지만 model weight까지 자동 폐기할지는 별도 판정이다.

영향 결과에는 직접 후손, 간접 후손, 이미 삭제됐지만 배포 기록에 남은 subject를 포함한다. registry, object store, node cache와 사용자 다운로드를 가능한 범위에서 추적한다. load gate는 revocation bundle version을 확인하고 offline 허용 기간을 제한한다. 오래 offline이었던 worker는 serving 또는 training 전에 최신 bundle로 재검증한다.

### clean rebuild의 조건

깨끗한 재빌드는 마지막 artifact를 입력으로 복사하는 일이 아니다. 신뢰할 수 있는 source와 material까지 rollback하고 새 builder에서 pipeline을 다시 실행한다. 침해된 compiler로 만든 중간 dataset filter binary나 tokenizer를 재사용하면 최종 weight만 새로 만들어도 오염 가능성이 남는다. graph에서 최초 신뢰 가능한 cut을 정하고 그 뒤 모든 derived subject를 재생성한다.

대규모 pretraining처럼 완전 재학습 비용이 막대할 때는 기술적 사실과 사업 결정을 분리한다. 재현이 비싸다는 이유로 lineage를 정상으로 표시하지 않는다. 위험 분석, 제한된 배포, 추가 검증 또는 retirement를 명시하고 예외 승인에 만료 시각을 둔다. 이후 fine-tune과 adapter가 compromised base를 상속했는지도 사용자에게 전달한다.

### 재빌드 차이를 설명한다

새 결과가 이전과 다르면 먼저 material digest, code와 config, toolchain, topology·kernel, RNG·sample order 순으로 비교한다. 각 층의 첫 차이를 찾기 전에 metric만 비교하면 우연히 비슷한 결과를 같은 build로 오인한다. bitwise 불가능한 학습에서는 predeclared statistical envelope와 golden probes를 사용하고, 같은 계보라는 주장과 같은 품질이라는 주장을 분리한다.

재빌드 보고서는 old/new subject, 교체한 trust root 또는 dependency, 동일한 material, 달라진 environment, 재현 등급과 잔여 위험을 담는다. 새 artifact가 promotion되면 old digest의 revocation을 유지한다. alias를 새 digest로 바꿔도 과거 audit record를 덮어쓰지 않는다.

### 출시 관문를 실패 우선으로 설계한다

공급망 release는 다음 순서로 진행한다. 요청한 source·dataset·base revision을 immutable digest로 resolve한다. 격리 builder가 resolved dependency와 environment를 기록한다. output bundle의 exact file set과 tensor schema를 검증한다. provenance statement를 만들고 authorized identity로 서명한다. SBOM과 runtime inventory를 붙인다. loader sandbox와 golden test를 수행한다. 마지막으로 catalog가 digest alias를 promotion한다.

각 단계는 실패하면 artifact를 quarantine하고 뒤 단계를 멈춘다. signature 실패 뒤 parser를 돌려 “파일은 정상”이라는 인상을 만들지 않는다. schema 실패 뒤 golden test tolerance를 넓히지 않는다. remote code review 실패 뒤 `trust_remote_code=False`로 우연히 local fallback되는 결과를 같은 artifact 승인으로 처리하지 않는다. 실패 원인을 보존하고 수정된 입력은 새 attempt와 새 provenance를 만든다.

**최종 인수용 열다섯 질문**

인수자는 실제 digest를 가리키며 답해야 한다. release bundle의 exact subject 집합은 무엇인가. 모든 floating ref는 어느 commit으로 resolve됐는가. dataset과 tokenizer의 parent attestation은 무엇인가. builder identity는 이 repository와 channel에 권한이 있는가. signature와 transparency evidence는 어느 trust root로 검증됐는가. cache hit에도 revocation 검사가 실행됐는가. safetensors의 file integrity와 model schema를 각각 무엇이 확인했는가. pickle allowlist에는 무엇이 왜 들어갔는가. remote code의 import closure와 resolved class는 무엇인가. SBOM에 JIT native extension이 포함됐는가.

CUDA binary는 어느 SM과 flag로 만들어졌는가. runtime inventory가 build SBOM과 어디서 다른가. 한 signer 또는 material을 폐기하면 어떤 후손이 격리되는가. clean rebuild는 어느 신뢰 가능한 cut에서 시작하는가. 재현 등급을 어떤 oracle로 판정했는가.

이 질문을 통과하면 공급망은 체크리스트 장식이 아니라 실행 가능한 검증 시스템이 된다. 독자는 artifact 이름 대신 digest와 lineage를 보고, loader 편의 옵션을 신뢰 경계 변화로 읽고, cache와 서명을 서로 다른 방어층으로 다룬다. “같은 모델”은 repository 이름이 아니라 동일한 material·builder·상태 전이와 검증 결과를 가진 subject라는 기술적 의미를 얻는다.

**재현 실패를 층별로 디버깅한다**

두 run이 다르다는 보고를 받으면 최종 benchmark 점수부터 비교하지 않는다. 먼저 subject bundle과 material digest, source·config, data order, environment, 실행 경로, 수치 결과 순서로 최초 차이를 찾는다. 앞선 identity가 다른데 뒤 결과가 비슷한 것은 재현 성공이 아니다. 반대로 identity가 같고 일부 GPU reduction 순서만 달라 bit가 다른 경우에는 선언한 statistical 등급으로 판정할 수 있다.

첫 층은 artifact set이다. weight shard 수, index, config, tokenizer와 template의 exact set을 비교한다. 한쪽에만 generation config 또는 custom processor가 있으면 모델 weight가 같아도 실행 bundle은 다르다. 각 파일의 digest와 역할을 비교하고 path·mtime 차이는 identity에서 제외한다. symlink가 가리키는 blob과 Git LFS materialization 상태도 실제 byte로 확인한다.

둘째 층은 provenance resolution이다. external parameter가 같아도 resolved commit이나 dataset snapshot이 달라질 수 있다. builder가 material digest를 남기지 않았다면 그 run은 강한 재현 판정을 받을 수 없다. cache log와 mirror mapping에서 당시 resolution을 복구하되, 추정한 값을 원래 attestation처럼 표시하지 않는다. missing evidence 자체를 결과에 남긴다.

셋째 층은 데이터와 tokenizer다. dataset shard digest가 같아도 filtering order, shuffle seed, packing과 worker partition이 다르면 sample stream이 달라진다. 최초 N개 global sample ID와 token·mask digest를 비교한다. token이 다르면 tokenizer bundle, normalizer와 remote tokenizer code를 본다. token은 같고 batch grouping만 다르면 sampler·packing config를 본다.

넷째 층은 Python과 native dependency다. lockfile만 비교하지 말고 installed wheel digest와 runtime-loaded shared object를 비교한다. `LD_DEBUG` 또는 process mapping을 제한된 재현 환경에서 수집하면 예상하지 않은 host NCCL·libstdc++ 주입을 찾을 수 있다. JIT extension의 cache key, compile command, `.so` digest와 target SM도 대조한다.

다섯째 층은 실행 경로다. Transformers가 native class와 remote class 중 무엇을 resolve했는지, attention backend와 fused optimizer가 무엇이었는지, CUDA kernel이 prebuilt cubin과 PTX JIT 중 어느 쪽이었는지 기록한다. config 문자열이 같아도 capability 검사와 library availability 때문에 다른 branch가 선택될 수 있다. resolved function과 module digest를 남겨야 한다.

여섯째 층에서야 수치를 본다. 첫 batch logits 표본, loss numerator·denominator, gradient norm, optimizer delta를 순서대로 비교한다. 첫 차이가 forward라면 weight·kernel·model code를, backward라면 graph·reduction을, update라면 optimizer state와 scheduler를 조사한다. 장기 metric만 비교하면 초기의 작은 차이가 증폭된 위치를 잃는다.

**재현 실험 27-A: material substitution.** 같은 repository 이름과 tag를 유지한 채 test mirror가 다른 commit을 반환하게 한다. resolved digest gate가 build 시작 전에 거부해야 한다. cache에 예전 blob이 있어 우연히 성공하는 경우도 따로 실행한다. 결과에는 requested ref, resolved commit, mirror identity와 cache source를 남긴다.

**재현 실험 27-B: 서명 subject 불일치.** 유효하게 서명된 attestation을 다른 safetensors 파일 옆에 둔다. signature math는 통과해도 statement subject와 실제 bundle digest 비교에서 실패해야 한다. 다음에는 authorized되지 않은 workflow identity가 올바른 subject에 서명하게 해 authorization gate가 실패하는지 본다. authenticity와 authorization의 독립성을 증명하는 실험이다.

**재현 실험 27-C: native binary 교체.** Python package version을 유지한 채 test extension `.so`만 다른 build flag의 binary로 교체한다. runtime SBOM 또는 loaded-object inventory가 차이를 검출해야 한다. golden output이 우연히 같아도 provenance identity는 다르다. 반대로 digest 차이가 timestamp section뿐이라면 reproducible-build 설정을 고쳐 의미 없는 변동을 제거한다.

**재현 실험 27-D: remote-code branch 변화.** 같은 Hub snapshot을 Transformers 두 revision에서 resolve해 local implementation 추가 전후의 class path를 기록한다. 선택 class가 바뀌면 config·state mapping·forward·save/load parity gate를 다시 실행한다. 결과가 같다는 이유로 source transition을 숨기지 않는다.

**재현 실험 27-E: stale revocation.** 폐기된 digest를 offline node cache에 남기고 오래된 trust bundle로 load를 시도한다. 허용 offline age를 넘으면 fail closed해야 한다. 최신 bundle을 받은 뒤 cache byte가 물리적으로 남아 있어도 load gate가 거부하는지 확인한다. purge와 policy denial을 별도 event로 기록한다.

이 디버깅 절차를 실행하려면 모든 run이 동일한 비교 key를 내야 한다. `BundleID`, material-set digest, builder identity, code/config digest, sample-stream prefix digest, environment/SBOM digest, resolved-class set, loaded-native-set과 golden trace를 저장한다. 민감한 dataset content는 남기지 않더라도 stable record ID와 salted digest, access-controlled mapping을 사용할 수 있다. 증거가 없으면 비슷한 metric을 근거로 재현됐다고 선언하지 않는다.

## 27.9 정책·데이터 공급망·운영 관측을 하나의 통제면에 둔다

정책 선언, 데이터 lineage, runtime 관측과 독립 검증 패키지를 분리해 저장하되 같은 ArtifactID와 ReleaseID로 조회할 수 있게 한다.

검증 프로그램에 허용 repository, signer와 exception을 하드코딩하면 정책 변경 때 verifier binary까지 다시 배포해야 하고 과거 판정을 재연하기 어렵다. verifier는 statement·signature·bundle·SBOM을 해석하는 결정적 engine으로 두고, 허용 identity, predicate version, required gate와 expiry는 versioned policy document로 둔다. 모든 판정 결과에 policy digest를 붙인다.

정책에는 deny가 allow보다 우선한다. 특정 digest나 signer가 폐기되면 넓은 repository allow rule이 다시 허용하지 못하게 한다. exception은 artifact·용도·환경 범위와 만료 시각, 승인자를 가진다. “연구용”처럼 해석이 넓은 label 대신 network 없는 conversion job, 특정 input digest와 output schema처럼 기계가 검사할 수 있는 범위를 쓴다.

policy update의 negative control은 매우 중요하다. signer allowlist에 새 workflow를 추가할 때 유사한 다른 repository, pull-request context, fork와 unprotected branch가 거부되는지 시험한다. remote code exception을 추가하면 다른 repository revision에는 전파되지 않는지 본다. `weights_only` global allowlist는 다른 worker와 다음 job에서 비어 있는지 확인한다.

판정 로그는 `received subject`, `verified digest`, `statement predicate`, `signer identity`, `authorization rule`, `schema result`, `revocation bundle`, `decision`과 이유를 가진다. secret이나 전체 certificate를 무분별하게 log하지 않되 audit에 필요한 digest와 public identity는 보존한다. deny 결과도 retention해야 공격자가 반복해 경계를 탐색하는 패턴을 찾을 수 있다.

### 데이터 공급망까지 같은 규칙을 적용한다

모델 공급망만 엄격하고 학습 dataset은 mutable directory로 받으면 provenance graph의 가장 큰 입력이 비어 있다. raw acquisition, license·consent 분류, dedup, quality filter, PII 처리, red-team 주입, curriculum split과 packing output을 각각 immutable snapshot과 transformation attestation으로 만든다. 각 단계는 input subject와 code/config digest, output subject, row-count·drop-reason 통계를 남긴다.

row content를 공개할 수 없는 경우에도 shard digest와 access-controlled record lineage를 보존한다. 삭제 요구가 오면 record ID에서 포함된 dataset snapshot, tokenizer와 model run으로 영향 범위를 계산한다. digest는 삭제된 개인 데이터를 영구 재식별하는 수단이 되지 않도록 threat model에 맞춰 salted 또는 keyed mapping과 접근 분리를 고려한다. 공급망 추적과 개인정보 최소화는 함께 설계해야 한다.

### poisoning 대응은 provenance와 통계를 결합한다

provenance가 정상이라고 데이터가 무해한 것은 아니다. 승인된 crawler와 pipeline이 공격 content를 정직하게 수집할 수 있다. source concentration, near-duplicate cluster, rare trigger, label·instruction anomaly와 canary benchmark를 별도 분석한다. anomaly가 발견되면 어느 raw shard와 transform version에서 유입됐는지 lineage가 조사 속도를 높인다.

filter update 뒤에는 제거된 row만 보지 않고 distribution shift와 downstream metric을 확인한다. 과도한 제거가 특정 언어·도메인을 훼손할 수 있다. clean rebuild는 새 filtered snapshot에서 tokenizer 재학습이 필요한지, 기존 tokenizer를 유지할지 명시한다. dataset 변화가 curriculum과 sample count, scheduler horizon에 미친 영향도 새 training provenance에 반영한다.

### 공급망 운영 대시보드와 경보

좋은 대시보드는 artifact 개수보다 상태 전이를 보여 준다. `received→digest-verified→signature-authorized→schema-verified→sandbox-tested→golden-tested→promoted`의 체류 시간과 실패율, quarantine backlog, 가장 오래된 trust bundle, unsigned artifact attempt, floating revision 거부, remote-code request와 exception expiry를 관측한다.

cache 지표에는 hit rate뿐 아니라 verified hit 비율, digest mismatch, partial download, stale-ref resolution과 revoked-blob access를 넣는다. SBOM 지표에는 inventory coverage, runtime-only native object, 알려지지 않은 component와 critical vulnerability age를 넣는다. signer 지표에는 identity별 release 수, 평소와 다른 workflow·시간대와 transparency inclusion 지연을 둔다.

경보는 operator가 행동할 수 있어야 한다. digest mismatch는 즉시 quarantine과 mirror 조사로, trust bundle age 초과는 offline load 중단과 sync로, exception 만료 임박은 owner review로 연결한다. 단순 signature failure와 authorization failure에는 서로 다른 runbook을 적용한다. 전자는 byte·bundle 손상 또는 공격을, 후자는 잘못된 actor나 policy drift를 시사한다.

**독립 검증 패키지의 구성**

release team이 제출한 manifest를 그대로 믿지 않고 독립 verifier가 clean workspace에서 bundle을 받는다. network를 차단한 상태에서 exact file set과 digest, signature bundle, predicate schema, signer authorization, revocation, safetensors 구조·state schema, SBOM과 remote-code closure를 재검사한다. golden load는 최소 권한 sandbox에서 수행한다.

패키지에는 source와 material resolution 표, SLSA provenance, in-toto statement와 signature bundle, policy digest와 판정 로그, SPDX 또는 CycloneDX SBOM, runtime native inventory, loader 상태 전이, failure-injection 결과, revocation 영향 query와 clean-rebuild 절차가 들어간다. 문서의 이름보다 내부 subject digest가 서로 일치하는지 자동 검사한다.

검증자는 upstream specification과 고정 소스 좌표를 설계 근거로 사용하고 실제 artifact에 대한 판정은 자신의 실행 결과로 남긴다. 모델 카드의 “safe format” 문구나 CI badge를 내부 schema·authorization 검사의 대체물로 쓰지 않는다. 반대로 모든 위험을 한 개의 점수로 압축하지 않는다. identity, authenticity, authorization, format, semantic schema, execution behavior와 reproducibility 등급을 따로 보여 준다.

최종 승인 뒤에도 검증은 끝나지 않는다. trust root, signer policy, dependency vulnerability나 dataset 권리가 바뀌면 기존 artifact를 새 정책으로 재평가한다. `RevocationID`와 후손 graph를 이용해 영향 범위를 계산하고, 필요하면 quarantine·rebuild·재평가를 실행한다. 지속적으로 재검증해야 release 시점의 스냅샷 증거가 장기 운영의 신뢰로 이어진다.

독립 검증의 마지막 단계에서는 세 가지 서로 다른 실패를 반드시 재연한다. 올바른 byte에 권한 없는 서명이 붙은 경우, 권한 있는 서명이 다른 subject를 가리키는 경우, 모든 서명이 맞지만 tensor schema나 remote-code policy가 틀린 경우다. 세 경우가 각기 authorization, subject binding, semantic·execution gate에서 멈춰야 한다. 하나의 “signature failed”로 합치면 운영자는 trust root 침해와 단순 bundle 교체, model compatibility 오류를 구분하지 못한다.

release alias를 새 digest로 이동한 뒤 이전 digest를 요청하는 재현 run도 수행한다. 명시한 digest는 계속 동일 byte로 해석되되 revoked 상태라면 정책이 거부해야 하고, floating alias는 새 subject와 새 provenance를 반환해야 한다. cache, mirror와 air-gapped node 모두 같은 판정을 내리는지 비교한다. 이 시험이 통과해야 이름 변경, byte identity와 사용 허가가 서로 독립적으로 관리된다는 사실이 입증된다.

최종 evidence package의 각 JSON·SBOM·signature bundle에는 자신이 설명하는 subject digest가 들어가야 하며 package 전체를 다시 묶는 index도 서명한다. index만 서명하고 child digest를 빠뜨리거나, child는 고정했지만 policy와 revocation version을 기록하지 않으면 사후 판정을 재연할 수 없다. 검증 시각, trust root와 policy digest를 보존해 미래의 조사자가 당시에는 왜 허용됐고 지금은 왜 거부되는지를 설명할 수 있게 한다.

**재현성을 단계별 등급으로 선언한다**

**byte 재현과 학습 재현은 같은 주장이 아니다.**

같은 source와 environment에서 container·extension·bundle이 byte-for-byte 같아지는 것은 build 재현성이다. 같은 checkpoint에서 다음 batch·loss·update가 같아지는 것은 resume 재현성이다. 처음부터 다시 학습해 최종 weight가 같은 것은 훨씬 강한 bitwise training 재현성이고, 분산 reduction과 kernel nondeterminism 때문에 현실적으로 비용이 클 수 있다. 독립 run이 정한 평가 분포 안에 들어오는 것은 statistical reproducibility다.

release record는 어느 등급을 주장하는지 명시한다. 낮은 등급을 높은 등급처럼 표현하지 않는다. bitwise가 불가능하다고 모든 검증을 포기하지도 않는다. sample-stream prefix, 첫 N update의 loss·parameter checksum, evaluation confidence interval처럼 가능한 oracle을 층별로 둔다. 차이가 생기면 최초 divergence를 data, environment, kernel, collective, checkpoint 상태로 좁힌다.

결정론 옵션은 상태와 비용을 바꾼다. deterministic algorithm 강제는 일부 kernel을 다른 구현으로 바꾸거나 지원되지 않는 연산에서 실패하게 할 수 있다. benchmark와 autotune 비활성화는 algorithm 선택 변동을 줄이는 대신 성능을 낮출 수 있다. seed 하나를 고정하는 것은 Python, NumPy, framework, device generator, dataloader worker와 augmentation generator를 모두 복원하는 것과 다르다.

재현 보고에는 hardware model·SM, driver, CUDA runtime, cuDNN·NCCL, compiler flag, world size와 topology를 포함한다. 같은 Python lockfile이 native 실행을 고정하지 않는다. statistical oracle은 metric, dataset revision, sampling, 반복 수, 허용 구간과 다중 비교 정책을 미리 정한다. 결과를 본 뒤 허용 오차를 넓히지 않는다.

**산출물 DAG에서 영향 반경을 계산한다**

**파일 목록을 parent-child 관계로 바꾼다.**

파인튜닝 release는 base model, tokenizer, chat template, dataset shards, filtering code, trainer code, environment, checkpoint, adapter merge, quantization, evaluation과 model card의 후손이다. 각 node는 digest와 type을 가지고 edge는 `derived-from`, `loaded-by`, `evaluated-on`, `documented-by`, `promoted-as`처럼 의미를 가진다. 같은 파일이 폴더에 있다는 이유만으로 dependency라고 추정하지 않는다.

dataset shard 하나가 철회되면 그것을 실제 소비한 run과 checkpoint, 그 checkpoint에서 만든 adapter·merged weight·quantized export, evaluation과 alias를 역방향으로 찾는다. 영향 query는 filename substring이 아니라 digest edge를 따른다. mix manifest에 shard digest 대신 mutable URI만 있으면 정확한 영향 반경을 계산할 수 없다.

base model license나 usage condition이 바뀌는 사건은 byte revocation과 다를 수 있다. artifact byte는 그대로지만 사용 허가 policy가 변한다. catalog는 identity, integrity와 authorization state를 분리한다. revoked blob을 cache에서 즉시 삭제할지 조사용으로 격리할지는 retention policy가 정하지만 production load는 거부해야 한다.

graph completeness도 측정한다. release subject에서 모든 leaf가 trusted source 또는 승인된 manual input에 도달하는지, digest 없는 edge와 unknown builder가 없는지 검사한다. orphan artifact는 삭제 대상이 아니라 먼저 계보 누락인지 조사한다. graph query와 결과는 policy revision과 함께 evidence package에 들어간다.

**safetensors를 tensor bundle 검증 절차로 읽는다**

**parser 통과 뒤 model 의미를 검증한다.**

header 길이와 JSON, offset, shape·dtype byte 수가 일치하면 파일 구조는 유효하다. 하지만 tensor 이름이 기대 모델과 맞는지, shard가 모두 있는지, config의 hidden size·head 수와 shape가 맞는지는 별도 문제다. index의 `weight_map`, shard별 실제 key, expected state schema를 세 방향으로 비교한다.

duplicate key와 overlapping tensor storage를 명시적으로 거부한다. tied weight의 생략은 임의 allowlist가 아니라 architecture config와 loader 계약에서 유도한다. unexpected key를 경고만 하고 계속 로드하는 기본 동작이 출시 관문에 적합한지 검토한다. adapter key prefix, quantization scale·zero point와 tensor-parallel shard 규칙도 bundle type별 schema에 넣는다.

`metadata`에는 임의 문자열이 들어갈 수 있으므로 신뢰 경계 밖의 값으로 취급한다. metadata의 framework·format 주장을 실제 file과 config 검증 대신 쓰지 않는다. 거대한 header, 비정상 dimension, integer overflow 경계, 잘린 shard와 trailing bytes fixture를 parser와 wrapper 양쪽에 넣는다. low-level parser가 거부해도 상위 loader가 예외를 삼키거나 fallback하는지 확인한다.

memory mapping은 로드 효율을 높이지만 원격 파일이 검증 도중 바뀌어서는 안 된다. immutable local blob을 digest 검증한 뒤 map한다. network filesystem이나 cache path를 직접 신뢰하지 않는다. 검증한 file descriptor와 실제 load한 bytes가 같은 객체인지 보장하고, symlink 교체와 partial download rename을 주입한다.

**`trust_remote_code`의 실행 폐쇄를 감사한다**

**한 Python 파일이 아니라 import graph를 본다.**

remote configuration은 `auto_map`을 통해 config, model, tokenizer와 processor class를 동적으로 선택할 수 있다. entry file만 검토해도 상대 import, package import, runtime download와 subprocess가 남을 수 있다. resolver가 선택한 exact revision과 class, 다운로드한 file set, import closure와 각 digest를 기록한다.

`trust_remote_code=False`는 remote code가 필요한 model을 안전한 내장 구현으로 자동 변환한다는 뜻이 아니다. 보통 동적 class 선택을 거부하거나 내장 mapping이 있는 경우 그것을 사용한다. 반대로 `True`는 코드가 안전하다는 판정이 아니라 실행을 허용하는 상태 전이다. 승인 policy는 repository·revision·class별 review evidence를 요구한다.

sandbox에는 network, filesystem write, device, environment secret와 subprocess 권한을 최소화한다. import time side effect와 model construction side effect를 구분해 관측한다. sandbox 성공은 악성 코드가 없다는 증명이 아니라 정한 capability 안에서 golden load가 끝났다는 증거다. source review와 behavior test, provenance·signature를 함께 쓴다.

dependency confusion도 closure에 포함한다. remote file이 일반 package 이름을 import할 때 environment의 어느 distribution으로 resolve됐는지 lock한다. local directory가 package shadowing을 만들지 검사한다. import된 `.so`와 JIT output은 runtime SBOM supplement로 들어간다.

**SBOM을 실제 실행 byte의 목록으로 확장한다**

**build inventory와 runtime inventory를 대조한다.**

container build 시점의 SBOM은 image layer에 들어간 package와 file을 보여 준다. 학습 시작 뒤 host mount에서 주입된 driver library, device plugin, NCCL, JIT compile된 Triton·CUDA extension, 다운로드된 remote code와 tokenizer library는 빠질 수 있다. process가 실제로 map한 native object와 생성한 executable cache를 runtime supplement로 수집한다.

Linux process의 memory map과 dynamic loader 정보를 수집할 때 path만 믿지 않는다. 같은 path의 file이 실행 중 교체될 수 있고 overlay filesystem의 layer가 다를 수 있다. device·inode, file digest와 load 시점을 기록한다. deleted-but-mapped object도 놓치지 않는다. collection agent의 권한은 read-only로 제한하고 training secret과 tensor memory를 읽지 않게 한다.

JIT artifact는 source, compiler, flags, target architecture와 parent environment를 가진다. CUDA extension의 결과가 같으려면 `-gencode`, fast-math, debug·line info, ABI, compiler와 linker가 같아야 한다. Triton cache key가 이 모든 의미를 포괄한다고 가정하지 않고 실제 output digest와 metadata를 기록한다. cache hit에도 artifact가 승인된 parent에서 왔는지 검사한다.

SBOM component에는 package URL이나 식별자, version, digest, license와 dependency relationship을 둔다. Python distribution과 import module 이름이 다를 수 있으므로 둘을 연결한다. wheel 안에 vendored된 native library와 system library를 중복 제거하되 어느 parent가 포함했는지는 보존한다. base image SBOM을 복사하는 대신 최종 layer와 runtime 차이를 계산한다.

**취약점 판정은 포함·도달·노출을 나눈다.**

scanner가 package 이름과 version으로 CVE를 매치하면 우선 실제 byte가 포함됐는지 확인한다. 다음으로 취약 symbol이나 code path가 로드·호출 가능한지 본다. 마지막으로 untrusted model, dataset, network input이 그 경로에 닿는지 분석한다. 이 세 단계는 위험 우선순위를 정하지만, 정보가 없다는 이유로 자동 면제하지 않는다.

native transitive dependency는 Python에서 직접 import하지 않아도 loader가 사용할 수 있다. plugin, codec, image/audio parser와 archive reader는 데이터 공급망의 입력면이다. 멀티모달 데이터셋은 텍스트-only pipeline보다 decoder 취약점 면이 넓다. 샘플 parser를 sandbox하고 resource limit, decompression ratio, dimension과 duration 한도를 둔다.

vulnerability snapshot에는 scanner database revision과 scan time을 넣는다. 어제 통과한 artifact가 오늘 새 advisory로 위험해질 수 있다. release catalog는 새 advisory가 들어오면 affected component digest에서 후손 모델과 deployment로 영향 query를 실행한다. 발견과 실제 완화 사이에는 owner, SLA, exception expiry가 필요하다.

**SLSA와 in-toto를 검증 순서로 구현한다**

**statement schema부터 거부 우선으로 읽는다.**

in-toto statement는 `_type`, `subject`, `predicateType`, `predicate`를 구분한다. verifier는 알지 못하는 major schema나 predicate를 임의로 해석하지 않는다. subject 배열의 각 name은 사람이 읽는 힌트이고 identity는 digest다. 동일 name에 다른 digest가 있거나 배포 bundle의 필수 child가 subject에서 빠지면 실패한다.

SLSA provenance predicate의 build definition은 build type, external parameters와 resolved dependencies를 담는다. external parameter에는 사람이 요청한 floating ref가 있을 수 있지만 resolved dependency에는 실제 commit·dataset·base image digest가 있어야 한다. 둘을 구분해야 “main branch를 요청했고 commit X를 사용했다”는 사실을 보존한다.

run details는 builder와 invocation, 시작·종료, metadata와 byproducts를 연결한다. builder identity가 서명됐다는 사실은 해당 repository와 release channel에 권한이 있다는 뜻이 아니다. verifier는 cryptographic authenticity를 확인한 뒤 authorization policy를 별도로 적용한다. policy에는 repository, workflow path, ref 조건, reusable workflow와 environment approval를 포함한다.

**검증 순서를 고정한다.**

첫째, envelope와 payload encoding을 parse하고 허용 media/schema인지 본다. 둘째, signature와 certificate·transparency evidence를 정한 trust root와 검증 시각에서 확인한다. 셋째, payload statement의 subject digest를 실제 bundle bytes와 비교한다. 넷째, predicate schema와 builder·materials를 semantic validation한다. 다섯째, authorization, freshness, revocation과 project policy를 적용한다.

signature만 먼저 녹색 표시하고 뒤의 subject mismatch를 경고로 낮추면 안 된다. 반대로 subject digest가 맞아도 unsigned statement는 provenance 주장으로 승인하지 않는다. 각 gate는 독립 결과와 failure reason을 내며 최종 decision은 모두를 결합한다. UI의 단일 checkmark보다 machine-readable verification log가 authoritative하다.

**hermetic이라는 주장을 network 차단으로 시험한다.**

provenance가 resolved dependency를 완전히 열거한다고 주장한다면 clean builder에서 cache를 비우고 선언한 material만 제공한 채 network를 차단해 rebuild한다. 숨은 package download, mutable apt index, time service, license server나 remote tokenizer fetch가 있으면 실패한다. 필요한 network input은 사전에 digest material로 승격한다.

완전한 bitwise rebuild가 안 되면 차이의 원인을 분류한다. archive timestamp, file ordering과 build path는 reproducible-build 설정으로 제거할 수 있다. compiler·GPU codegen 차이는 environment identity를 더 좁혀야 한다. 학습 자체의 stochastic 차이는 statistical oracle로 넘어가되 build artifact 재현 실패와 섞지 않는다.

**Sigstore identity와 키 침해에 대응한다**

**서명 주체를 email 문자열 하나로 축약하지 않는다.**

keyless signing에서는 certificate의 issuer, subject와 workflow-related claim, 유효 시간, transparency inclusion을 검증한다. 같은 repository라도 pull request와 protected release workflow는 권한이 다르다. reusable workflow를 썼다면 호출자와 실제 signer workflow의 관계를 policy가 이해해야 한다.

bundle verification은 offline 환경에서 certificate chain, signature와 transparency proof를 함께 검증할 수 있게 자료를 보존한다. 단순히 public log URL만 저장하면 air-gapped node나 미래 조사에서 재연하기 어렵다. 당시 trust root와 policy digest, verification time을 evidence에 넣는다. 현재 policy로 과거 결정을 다시 평가한 결과와 당시 policy 판정을 구분한다.

키나 identity가 침해되면 certificate 만료만 기다리지 않는다. compromise interval, affected identity·workflow·repository를 정의하고 그 구간에 서명된 subject를 graph query로 찾는다. artifact를 `suspect`로 전이하고 production promotion과 cache load를 차단한다. 조사 결과에 따라 `revoked` 또는 재승인 상태로 이동한다.

서명 rotation은 old signature를 무조건 무효로 만들지 않는다. artifact가 승인됐던 시점의 trust state와 현재 사용 허가를 분리한다. 새 release는 새 identity만 허용하되 과거 artifact는 보존된 transparency·policy evidence로 역사적 진위를 검증할 수 있다. 현재 deployment 허가는 별도 revocation policy가 결정한다.

**철회가 cache와 mirror까지 도달하는지 시험한다.**

중앙 catalog에서 revoked 표시를 바꿔도 node cache가 digest blob을 그대로 load하면 통제가 실패한다. loader는 cache hit 전에 policy와 revocation snapshot을 확인한다. offline node는 서명된 최신 trust bundle의 최대 age를 검사하고 너무 오래되면 fail closed하거나 사전 정의한 제한 모드로 들어간다.

mirror는 blob, provenance, signature, policy와 revocation metadata를 원자적으로 갱신해야 한다. 일부만 복제되면 integrity는 맞지만 authorization이 오래된 상태가 된다. fault test는 metadata sync를 지연하고 revoked blob 요청이 거부되는지 본다. 거부 log에는 산출물 digest, policy generation과 reason code를 남기되 민감 identity를 과다 노출하지 않는다.

**데이터셋 provenance를 row와 변환 단계까지 내린다**

**원본 보존과 학습 사용 허가를 분리한다.**

raw object를 조사 목적으로 보존한다고 그것을 학습에 사용할 권한이 지속되는 것은 아니다. source acquisition, license·consent, integrity, privacy review와 training eligibility를 별도 상태로 둔다. 한 상태가 revoked되면 해당 row·shard를 소비한 mixture와 checkpoint 후손을 찾는다.

row마다 모든 원문과 개인 식별자를 중앙 graph에 노출하지 않는다. stable record ID, source batch, content digest, policy class와 접근 통제된 mapping을 사용한다. dedup·filter·redaction·quality scoring·tokenization 변환은 code/config revision과 input/output manifest digest를 가진다. 어떤 row가 왜 제거됐는지 reason taxonomy를 bounded하게 유지한다.

streaming dataset은 전체 byte snapshot이 없을 수 있다. 그 경우 endpoint 이름만 기록해서는 재현할 수 없다. object version·ETag의 신뢰 계약, shard list, range, sampling seed와 소비 순서 prefix를 보존한다. source가 immutable version을 제공하지 않으면 materialize하거나 재현 등급을 낮춘다.

curriculum과 mixture scheduler는 시간에 따라 data distribution을 바꾼다. 단순 dataset list 외에 optimizer-step별 mixture weight, exhausted/replacement 정책, shard cursor와 retry·skip 결정을 기록한다. resume 뒤 같은 data를 소비한다는 주장은 이 상태가 복원될 때만 가능하다.

poisoning 대응은 provenance만으로 끝나지 않는다. 신뢰 가능한 source도 계정 탈취나 pipeline 오류로 오염될 수 있다. 신규 source·distribution shift, near-duplicate burst, rare token·language 변화와 label anomaly를 통계적으로 감시한다. 이상 cluster에서 source lineage를 따라 원 acquisition batch와 후손을 격리한다.

**환경 잠금은 package version보다 깊어야 한다**

**Python lock과 native ABI를 연결한다.**

`requirements.txt`나 lockfile은 dependency resolution의 출발점이다. wheel tag, index URL, 산출물 hash와 marker가 없으면 같은 version 이름이 다른 byte로 해석될 수 있다. editable install과 local path dependency는 source commit·dirty state를 기록한다. VCS dependency는 tag가 아니라 commit digest로 resolve한다.

PyTorch와 CUDA extension은 Python API가 같아도 C++ ABI, CUDA architecture와 linked library에 의존한다. wheel의 bundled library, system driver와 host-mounted NCCL을 inventory한다. `LD_LIBRARY_PATH`, preload, plugin path와 dynamic loader resolution 결과를 manifest에 넣는다. path 문자열보다 실제 loaded object digest가 강한 증거다.

driver는 container 밖에 있으므로 image digest만으로 재현되지 않는다. GPU model, VBIOS·firmware가 필요한 범위, driver version과 exposed capability를 기록한다. CUDA runtime과 driver compatibility가 허용 범위라는 사실은 동일 kernel selection과 수치 결과를 보장하지 않는다. golden run은 실제 environment identity에서 수행한다.

compiler 환경에는 Python, C/C++ compiler, NVCC 또는 JIT compiler, linker, build system과 flags가 들어간다. environment variable이 code generation이나 thread count를 바꾸면 allowlist로 manifest에 포함한다. 전체 환경 dump는 secret 위험이 있으므로 의미 있는 변수만 schema로 관리한다.

locale, timezone, filesystem ordering과 hash randomization도 preprocessing과 file discovery를 바꿀 수 있다. shard glob 결과는 정렬해 manifest로 고정한다. time-dependent default와 임시 directory path가 output에 들어가지 않게 한다. 재현 build는 서로 다른 workspace path와 시간에서 실행해 숨은 입력을 찾는다.

**checkpoint를 불완전한 상태 묶음으로 취급한다**

**weight 저장 성공과 재개 가능성을 분리한다.**

모델 weight만 있으면 inference artifact일 수 있지만 training resume artifact는 아니다. optimizer state, scheduler, scaler, RNG, sampler·data cursor, gradient accumulation policy, global optimizer generation과 distributed topology metadata가 필요하다. sharded optimizer라면 모든 rank shard와 index의 완전성을 검증한다.

checkpoint writer는 임시 generation에 각 component를 쓰고 digest와 size를 확인한 뒤 complete marker 또는 catalog alias를 원자적으로 승격한다. directory가 존재한다는 사실이나 rank 0 파일 하나로 완료를 판정하지 않는다. object store는 rename semantics가 다를 수 있으므로 immutable objects와 마지막 manifest commit을 사용한다.

저장 도중 rank가 죽으면 partial generation은 격리된다. retry가 같은 generation name에 다른 byte를 덮어쓰지 않게 attempt ID를 둔다. garbage collector는 active writer lease와 retention을 확인한 뒤 partial을 삭제한다. 조사 중 incident artifact와 마지막 known-good generation은 보호한다.

loader는 manifest에 열거된 exact file set과 digest를 먼저 검증한다. extra file도 provenance 혼동을 만들 수 있으므로 정책에 따라 거부하거나 명시적으로 무시한다. schema version, world-size compatibility와 optimizer class·parameter group mapping을 확인한다. config가 바뀐 상태로 `strict=False` load가 성공해도 동일 run resume로 승인하지 않는다.

**다음 update가 복구 oracle이다.**

load가 예외 없이 끝났다는 것은 약한 증거다. 고정 fixture에서 다음 batch identity, loss numerator/denominator, gradient 또는 update checksum, LR와 scheduler state를 기준 실행과 비교한다. stochastic kernel 때문에 bitwise가 어려우면 허용 오차와 반복 분포를 사전에 정한다. divergence가 나타나면 component별 state digest와 소비 순서를 비교한다.

accumulation 중간 checkpoint는 정책을 명시해야 한다. partial gradient와 microstep cursor를 저장하지 않으면 마지막 update boundary로 되돌아가 replay할 수 있다. 이때 중복 소비 token과 데이터 cursor 조정이 필요하다. 조용히 다음 batch로 건너뛰면 optimization trajectory가 달라진다.

**공급망 실패를 공격자 관점에서 주입한다**

**이름은 같고 byte만 바꾼다.**

mutable model tag, dataset URI, package version과 container tag가 다른 digest를 가리키게 만든다. resolver가 실행 시작 전에 immutable digest로 고정하고 provenance external parameter와 resolved material을 모두 기록해야 한다. cache가 이전 byte를 반환해도 요청한 digest 검사가 결과를 결정한다.

**byte는 맞고 증명만 바꾼다.**

올바른 model blob 옆에 다른 subject를 위한 유효 signature·provenance를 둔다. cryptographic signature는 유효하지만 subject binding에서 실패해야 한다. 이어 올바른 subject를 unauthorized workflow가 서명하게 해 authorization gate를 시험한다. 두 실패 reason이 구분되어야 incident triage가 가능하다.

**증명은 맞고 입력 하나를 숨긴다.**

builder가 network에서 undeclared tokenizer file을 받게 한다. warm cache에서는 build가 성공할 수 있으므로 clean environment와 network denial에서 실행한다. resolved material에 없는 access가 탐지되어야 한다. 로그에 secret URL query를 남기지 않으면서 dependency identity를 보고한다.

**cache가 검증 뒤 교체되게 한다.**

검증기가 path의 digest를 계산한 직후 symlink 또는 file을 교체하고 loader가 path를 다시 열게 한다. 검증한 immutable descriptor나 content-addressed blob을 실제 load하지 않으면 TOCTOU가 재현된다. partial download가 final name으로 보이는 경우와 concurrent writer도 함께 시험한다.

**SBOM에서 native object 하나를 누락한다.**

approved image에 host path의 다른 `.so`를 주입한다. build SBOM은 변하지 않지만 runtime loaded-object inventory가 차이를 검출해야 한다. 취약점 scanner와 revocation query가 runtime-only component에서도 후손 job을 찾는지 확인한다.

**stale revocation bundle을 제공한다.**

artifact 서명과 digest는 모두 유효하지만 offline node가 철회 전 trust snapshot을 가진 상태를 만든다. max-age와 policy generation 검사로 load가 거부되어야 한다. availability를 위해 제한적으로 허용하는 정책이라면 사용 가능한 artifact class, 시간과 감사 로그를 사전에 정의한다.

**remote code closure를 늦게 확장한다.**

승인된 entry file이 runtime 조건에서만 추가 module이나 URL을 import하게 한다. static closure와 sandbox network denial, runtime import log 가운데 어느 gate가 잡는지 본다. 조건이 발생하지 않은 golden test 하나만으로 안전을 주장하지 않는다. behavior fixture는 config branch와 optional dependency를 교차한다.

## 27.10 재현 실패를 최초 byte와 상태 차이에서 양분한다

최종 metric만 비교하지 않고 source, build, data order, kernel, checkpoint와 serving 단계의 최초 불일치를 찾는다.

두 run의 최종 metric만 다르다고 곧바로 nondeterminism이라 부르지 않는다. 먼저 material set, code·config, environment, sample-stream prefix가 같은지 비교한다. 어느 identity가 다르면 입력 재현 실패다. 같다면 첫 optimizer generation부터 loss·update checksum을 비교해 최초 수치 divergence를 찾는다.

첫 batch 전에 차이가 나면 model load, parameter initialization, tokenizer·collator와 data ordering을 본다. forward부터 다르면 kernel selection, dtype, device와 stochastic layer RNG를 본다. backward에서 처음 다르면 reduction order, gradient accumulation, checkpointing과 custom autograd를 본다. optimizer 이후 다르면 parameter group ordering, state restore, fused optimizer와 scheduler 순서를 본다.

여러 rank에서 divergence 시각이 다르면 collective 전 rank-local 값과 collective 후 값을 비교한다. 동일 rank가 먼저 갈라지면 그 rank의 data와 kernel path를 좁힌다. 모든 rank가 같은 순간 같은 방향으로 갈라지면 공통 config·code 또는 deterministic algorithm 선택을 의심한다.

binary search checkpoint를 짧은 간격으로 무한 저장하지 않는다. golden fixture에서 selected generation의 compact checksum과 state summary를 기록한다. divergence window가 좁혀지면 상세 trace를 켠다. 계측이 ordering을 바꿀 수 있으므로 관측 on/off 결과를 비교한다.

차이가 허용 오차 안이라는 판정에도 근거가 필요하다. 절대·상대 오차, tensor별 scale, evaluation variance와 downstream risk를 고려한다. 작은 parameter 차이가 generation 경로를 크게 바꿀 수 있으므로 weight 근접성만으로 행동 재현을 주장하지 않는다. 반대로 생성 text exact match만 요구하면 sampling 모델의 합법적 변동을 실패로 오판할 수 있다.

### policy engine과 verifier를 분리한다

**기술적 사실과 조직의 허용 결정을 나눈다.**

verifier는 signature가 유효한지, subject digest가 실제 byte와 맞는지, predicate schema가 해석 가능한지, SBOM component가 존재하는지를 결정적으로 계산한다. policy는 어느 builder identity와 repository가 production channel을 만들 수 있는지, 어떤 재현 등급과 vulnerability 예외를 허용하는지 결정한다. 둘을 한 코드의 조건문으로 섞으면 policy 변경 때 과거 판정을 재연하기 어렵다.

verification result에는 입력 bundle digest, engine version과 개별 gate 결과를 넣는다. policy decision에는 policy digest, evaluation time, requested action, principal, allow/deny와 reason을 넣는다. 같은 verification result를 staging과 production policy가 다르게 판단할 수 있다. artifact가 바뀌지 않아도 policy update로 허용 상태가 바뀔 수 있다.

fail-open 예외는 기본이 아니다. policy backend가 unavailable일 때 production load를 계속 허용하면 철회가 전파되지 않는다. availability 요구가 높은 환경은 짧은 수명의 signed decision 또는 policy bundle을 cache하고 max-age 뒤 fail closed한다. emergency override에는 좁은 artifact·cluster·시간 범위, 다중 승인과 사후 review가 필요하다.

정책 언어의 기능이 많을수록 분석이 어려워진다. network call이나 현재 시각 같은 비결정 입력을 제한하고 evaluation input을 artifact metadata, verified claims, requested environment로 명시한다. test fixture는 allow, deny, unknown schema, expired exception, revoked signer와 stale policy를 포함한다. policy coverage는 rule line보다 위험 시나리오로 측정한다.

### promotion을 alias 변경이 아닌 트랜잭션으로 만든다

**candidate에서 production까지 증거가 누적된다.**

artifact가 생성되면 `ingested` 상태에서 digest와 exact file set을 고정한다. schema, malware·secret, loader sandbox와 numerical golden test를 통과하면 `verified`가 된다. provenance·signature·SBOM과 policy decision이 붙으면 `approved` 후보가 된다. production alias 변경은 이 모든 parent evidence가 같은 subject를 가리킬 때만 실행된다.

두 승인자가 서로 다른 artifact를 보고 서명하지 않도록 approval request 자체가 subject digest와 evidence-index digest를 가진다. review 도중 새 evaluation이나 SBOM이 붙으면 index가 바뀌므로 이전 approval의 유효성을 policy가 판단한다. mutable web page의 체크 상태를 승인 기록으로 쓰지 않는다.

promotion transaction은 catalog alias, deployment manifest와 audit event 사이의 부분 실패를 고려한다. immutable subject를 먼저 준비하고 alias generation을 compare-and-swap으로 바꾼다. deployment는 alias 이름보다 resolved digest를 받는다. audit event가 durable하지 않으면 alias 변경을 완료로 간주하지 않는다.

rollback은 이전 alias를 되돌리는 것 이상이다. 이전 subject가 현재 policy와 revocation에서 여전히 허용되는지 다시 확인한다. 데이터 철회나 key compromise 때문에 rollback target도 금지됐을 수 있다. viable predecessor가 없으면 serving 중단·기능 제한 같은 별도 안전 정책을 적용한다.

canary는 candidate digest를 직접 로드하고 runtime attestation으로 실제 digest, loader class, SBOM delta와 config를 보고한다. 평가가 통과하면 fleet rollout이 같은 digest를 사용한다. node별 cache가 alias를 다시 resolve하게 두면 rollout 중 서로 다른 byte가 섞일 수 있다.

### 라이선스와 데이터 사용 조건을 실행 가능한 제약으로 만든다

**model card 서술과 machine policy를 연결한다.**

base model, dataset, tokenizer, code와 dependency는 서로 다른 license와 use restriction을 가질 수 있다. 최종 model card에 이름만 나열해서는 어느 artifact에 어떤 조건이 적용되는지 알 수 없다. material node별 license expression, 출처, notice requirement, redistribution·commercial·field restriction과 검토 상태를 기록한다.

license scanner의 자동 추정은 증거 후보이지 최종 법적 판단이 아니다. repository license file, model card와 dataset terms가 충돌하거나 누락되면 `unknown`으로 두고 승인 담당자에게 보낸다. 추정 결과를 임의로 permissive license로 채우지 않는다. 수동 판정에는 근거 URL·snapshot digest, reviewer와 적용 범위를 남긴다.

adapter만 배포해도 base model dependency가 사라진다고 자동 판단하지 않는다. merged weight, delta, distilled model과 synthetic data가 어떤 parent condition을 계승하는지는 정책·법적 분석 대상이다. 기술 graph는 derivation edge와 실제 bytes 포함 관계를 정확히 제공하고, 허용 결정은 별도 policy가 한다.

사용 조건이 바뀌면 byte integrity와 무관하게 authorization이 바뀐다. 변경 시각과 source snapshot을 기록하고 affected descendants를 query한다. 이미 배포된 artifact의 notice·access·deletion 의무와 새 release 금지를 구분한다. 모델 카드와 release catalog를 동시에 갱신하되 authoritative machine decision을 문서 문구로 대체하지 않는다.

**비밀과 자격 증명이 artifact에 섞이지 않게 한다**

**학습 config도 배포 산출물이다.**

experiment tracker와 provenance에 config를 남길 때 API token, signed URL, private dataset path와 사용자 식별자가 유출될 수 있다. 전체 환경·CLI를 그대로 직렬화하지 않고 schema allowlist를 사용한다. secret 값은 runtime secret manager에서 주입하고 manifest에는 secret identifier와 version 또는 정책상 허용된 digest만 기록한다.

checkpoint, tokenizer, generated sample, profiler trace와 log를 secret scanner로 검사한다. binary tensor에 우연히 token 문자열이 나타나는 단순 검색은 false positive가 많으므로 file type과 metadata·config 영역을 우선 검사하고 정책을 조정한다. 검출된 secret 원문을 scan report에 다시 복사하지 않는다.

secret이 발견되면 artifact 삭제만으로 끝나지 않는다. credential을 revoke·rotate하고 해당 secret으로 접근 가능한 system의 audit를 확인한다. 이미 mirror·cache·tracker에 복제된 descendant를 찾는다. 조사 증거는 접근 제한된 격리 저장소에 두고 production catalog에서는 load를 막는다.

빌드 log의 shell trace, exception과 package index URL도 점검한다. dependency resolver가 인증 URL을 provenance material에 그대로 넣지 않게 정규화한다. remote code sandbox에 production credential을 제공하지 않는다. 필요한 model download는 사전 materialization된 content-addressed input으로 전달한다.

**분산 checkpoint와 adapter 변환의 계보를 보존한다**

**shard layout과 논리 tensor를 둘 다 식별한다.**

FSDP나 tensor parallel checkpoint는 world size와 partition rule에 따라 물리 file set이 달라질 수 있다. 논리 state가 같아도 shard byte는 다르다. manifest에는 logical tensor schema와 physical shard mapping, process group topology와 planner version을 함께 둔다. resharding은 새 artifact를 만드는 변환으로 기록한다.

full-state export는 모든 shard를 모으는 과정에서 dtype cast, device move와 memory pressure를 만들 수 있다. 변환 code revision, option, output schema와 source checkpoint generation을 기록한다. source와 output의 tensor별 checksum을 canonical order로 비교한다. floating serialization 차이가 있으면 허용 규칙을 명시한다.

LoRA adapter merge는 \(W' = W + sBA\)라는 계산을 수행하므로 base weight digest, adapter tensors, scale와 dtype·accumulation precision이 결과 identity에 들어간다. merge 후 adapter metadata를 버리기 전에 parent edge를 보존한다. 여러 adapter의 순차 merge는 일반적으로 순서가 결과와 동작 의미에 영향을 줄 수 있으므로 ordered recipe를 기록한다.

quantization export는 calibration data, algorithm, group size, scale/zero-point dtype, kernel target과 outlier policy를 가진 파생 artifact다. 원 fp weight의 signature가 quantized output을 자동 인증하지 않는다. output bundle에 새 provenance·SBOM·evaluation을 붙인다. loader가 실제 선택한 quantized kernel과 fallback도 runtime inventory에 기록한다.

conversion tool이 unknown key를 무시하거나 tensor를 transpose·rename하는 경로를 test한다. 작은 synthetic state dict에 distinctive values를 넣어 mapping을 exact 검증한다. shape만 같은 tensor가 잘못 연결되는 오류는 일반적인 load 성공이나 aggregate checksum만으로 놓칠 수 있다.

**독립 검증자가 수행할 clean-room 절차**

검증자는 release pipeline의 workspace와 cache를 공유하지 않는다. subject digest를 요청하고 허용 mirror에서 immutable blob과 evidence index를 가져온다. 다운로드 중에는 partial name을 쓰고 digest 검증 뒤 content-addressed 위치로 원자 승격한다. network access log와 trust bundle version을 보존한다.

먼저 exact file set, size와 digest를 검사한다. 이어 signature·certificate·transparency, subject binding, predicate schema, builder authorization과 revocation을 순서대로 검증한다. SBOM을 parse하고 runtime sandbox inventory와 대조한다. safetensors 구조·state schema, tokenizer/config/chat template의 상호 호환성을 확인한다.

remote code가 있으면 import closure를 materialize하고 network 없는 최소 권한 sandbox에서 load한다. pickle 계열이 필요하면 `weights_only` allowlist와 negative fixture를 실행한다. golden inference 또는 작은 training next-step oracle은 고정 hardware compatibility class에서 실행하고 결과·환경을 기록한다. 대규모 학습을 재실행하지 않아도 release 경계의 결정적 상태는 검증할 수 있다.

검증자는 제출된 summary를 그대로 복사하지 않고 raw evidence에서 결론을 계산한다. pipeline과 verifier가 같은 library bug를 공유할 위험을 줄이기 위해 critical digest·schema 검사를 독립 구현이나 교차 도구로 수행할 수 있다. 차이가 나면 어느 구현이 authoritative한지 specification과 fixture로 판단한다.

최종 report는 pass뿐 아니라 claimed reproducibility grade, 검증하지 못한 영역, exception과 expiry를 포함한다. report와 모든 child evidence digest를 index로 묶고 독립 verifier identity로 서명한다. promotion service는 이 report의 subject와 policy를 다시 확인하며, report 파일이 존재한다는 사실만 보지 않는다.

**소스 저장소의 revision을 실행 입력으로 고정한다**

**commit hash만 기록하면 충분하지 않은 경우를 찾는다.**

Git commit은 tracked tree와 parent·metadata를 고정하지만 submodule, Git LFS object, generated file, untracked patch와 external build input까지 자동으로 담지 않는다. 학습 job 시작 시 repository commit, submodule commit, LFS object ID와 dirty diff 상태를 기록한다. dirty tree를 허용한다면 patch digest와 content bundle을 material로 승격한다.

tag와 branch는 사람이 읽는 provenance의 요청값으로 보존할 수 있지만 실행 identity는 resolved commit이다. shallow clone에서 필요한 history나 signed tag 검증이 누락되지 않게 한다. commit signature는 authoring provenance의 일부일 수 있으나 CI builder가 실제 그 tree를 사용했다는 build provenance를 대신하지 않는다.

generated source와 vendored dependency는 generator revision과 입력을 가진다. repository에 생성 결과만 commit되어 있다면 byte identity는 고정되지만 수정·감사 가능성을 위해 generator relation을 기록한다. build가 생성 파일을 다시 만드는지 기존 파일을 쓰는지 명확히 하고 두 경로의 차이를 test한다.

patch queue와 hotfix는 base commit 위 ordered patch digest로 표현한다. 실행 container에서 직접 file을 수정한 뒤 commit만 기록하는 관행을 금지한다. build context의 exact tree digest를 계산해 provenance subject 또는 material에 넣는다. `.gitignore` 밖의 config와 secret이 build context로 들어가지 않는지 검사한다.

**Hugging Face Hub와 모델 cache의 경계를 검증한다**

**repository revision과 snapshot directory를 구분한다.**

Hub repository의 commit은 file pointer 집합을 고정할 수 있지만 local snapshot이 완전하게 materialize됐는지는 별도다. snapshot directory의 symlink와 blob store를 따라 실제 file digest를 검사한다. requested allow/ignore pattern이 tokenizer·config·custom code 또는 shard를 빠뜨리지 않았는지 bundle schema와 비교한다.

`local_files_only`는 network를 막지만 cache byte의 provenance와 완전성을 자동 검증하지 않는다. 이전 partial download, manual file copy와 서로 다른 endpoint에서 온 blob이 섞일 수 있다. content digest와 source commit mapping을 검증한 뒤 offline bundle로 승격한다. cache directory를 출시 산출물처럼 서명하지 않고 필요한 exact file set을 새 manifest로 묶는다.

floating revision으로 online load한 뒤 cache path만 기록하면 나중에 같은 이름이 다른 commit을 가리킬 수 있다. resolve 단계에서 commit과 file metadata를 저장하고 이후 단계는 commit digest만 받는다. redirect, mirror와 CDN이 반환한 byte도 최종 content digest로 확인한다.

cache eviction은 재현성 정책과 연결된다. blob을 지워도 provenance record는 남기고 재취득 가능성과 source retention을 확인한다. upstream 삭제 가능성이 있는 중요한 base·dataset material은 라이선스와 정책 범위 안에서 내부 immutable archive에 보존한다. 보존 권한이 없으면 재현 등급과 위험을 명시한다.

**offline export를 독립 product로 본다.**

air-gapped cluster로 옮길 bundle에는 model shards, index, config, tokenizer, chat template, custom code, license·notice, provenance, signature, SBOM, trust/policy bundle을 포함한다. 전송 archive 자체의 digest와 child manifest를 모두 검증한다. archive extraction의 path traversal, symlink와 압축 폭탄을 방어한다.

반입 scanner와 cluster-side verifier가 같은 결과를 내는지 교차한다. 반입 뒤 cluster mirror가 file metadata를 바꾸거나 line ending을 변환하지 않게 content-addressed storage를 사용한다. offline trust bundle expiry와 revocation 갱신 절차를 운영한다. network가 없다는 사실은 공급망이 안전하다는 뜻이 아니다.

**컨테이너와 호스트의 경계를 provenance에 넣는다**

**image digest는 실행 rootfs의 시작점이다.**

container image digest는 layer 구성을 고정하지만 entrypoint가 시작하며 mount하는 config, secret, dataset, `/dev`와 host library는 포함하지 않는다. orchestrator spec, image pull policy, command·args, environment allowlist, volume source digest와 device allocation을 invocation record로 남긴다.

mutable image tag는 요청값일 뿐이다. node가 실제 pull한 manifest digest와 platform-specific image digest를 보고해야 한다. multi-architecture manifest에서 같은 tag가 다른 child image를 선택한다. CUDA 학습은 target architecture를 제한하더라도 platform resolution을 명시한다.

privileged container, host network·PID, broad volume mount는 remote code와 parser의 영향 범위를 키운다. golden sandbox와 production job의 capability 차이를 기록한다. sandbox에서 안전했던 loader가 production credential과 host path를 가진 상태에서도 동일 위험이라고 가정하지 않는다. 최소 권한을 production에도 적용한다.

Kubernetes admission이나 scheduler가 spec을 mutate할 수 있다. 요청 manifest와 실제 admitted pod spec의 digest를 모두 보존하고 sidecar, init container, injected volume과 environment 차이를 SBOM·policy에 넣는다. runtime node agent가 실제 process tree와 loaded objects를 report한다.

**데이터 삭제와 모델 철회의 현실적 경계를 기록한다**

**삭제 요청을 lineage query로 전환한다.**

특정 source batch나 record가 삭제 대상이 되면 raw·processed shard, cache, tokenizer training input, checkpoint와 파생 model을 graph에서 찾는다. 기술적으로 한 row가 weight에 미친 영향을 정확히 되돌리는 것은 일반적으로 어렵다. 따라서 삭제 가능성, retraining, model editing과 service restriction 가운데 정책상 요구되는 대응을 구분한다.

“데이터에서 삭제했다”는 말은 원본 object만 지웠는지, materialized shard·backup·cache와 후손 artifact까지 처리했는지 모호하다. deletion manifest에는 대상 digest, storage class, action, verification, exception과 완료 시각을 둔다. 법적 보존이 필요한 조사 copy와 production 사용 금지를 분리한다.

machine unlearning이나 knowledge editing을 사용한다면 새 파생 artifact와 평가가 필요하다. 원 모델을 조용히 덮어쓰지 않는다. 제거 대상 기억의 평가뿐 아니라 일반 능력, 인접 개념과 재학습 공격에 대한 검증을 기록한다. 이 기법의 성공 주장을 완전한 데이터 lineage 삭제와 동일시하지 않는다.

철회 통지는 model consumer가 이해할 수 있는 stable ArtifactID, affected version·digest, severity, action과 deadline을 가진다. downstream이 자체 merge·quantization을 만들었을 수 있으므로 parent digest를 보고하도록 계약한다. 공개 alias만 바꾸면 이미 pin된 deployment는 계속 실행된다.

**재현성과 보안의 긴장을 명시적으로 다룬다**

완전한 재현을 위해 raw data, environment와 debug artifact를 오래 보존하면 privacy·secret·license 위험이 커진다. 반대로 모두 빠르게 삭제하면 incident와 결과를 검증할 수 없다. artifact class별 최소 필요 정보, 접근 권한, retention과 암호화·파기 정책을 정한다.

민감 dataset은 원문 대신 immutable manifest, transformation code, aggregate statistics와 접근 통제된 escrow를 사용할 수 있다. 이것은 동일 data 재실행 가능성을 제한하므로 재현 등급에 반영한다. 숨겨진 data를 사용하면서 완전 공개 재현이라고 주장하지 않는다. 독립 감사자는 승인된 enclave나 clean room에서 digest와 pipeline을 검증할 수 있다.

deterministic seed와 sample order를 공개하면 개인 record의 membership 추론 위험이 커질 수 있다. 외부 공개 evidence와 내부 restricted evidence를 계층화한다. 공개 report에는 cryptographic commitment와 검증 결과를, 제한 영역에는 실제 mapping을 둔다. commitment가 존재한다는 사실만으로 내용이 적법하거나 올바르다는 보장은 없다.

보안 patch는 environment를 바꿔 bitwise 재현성을 깨뜨릴 수 있다. 취약 환경을 영구 실행 가능하게 보존하지 않고 image·SBOM·trace 같은 비실행 증거를 남긴다. 재현이 필요하면 격리된 제한 환경과 승인 절차를 사용한다. production에서는 보안 policy가 과거 byte 재현보다 우선할 수 있음을 명시한다.

**공급망 운영 체크리스트를 역할별로 나눈다**

**연구자는** source·base·dataset·tokenizer를 floating 이름이 아니라 resolved digest로 제출한다. seed와 data order만 아니라 environment, precision, distributed topology와 checkpoint policy를 config에 넣는다. manual notebook 변경을 patch artifact로 남긴다.

**builder는** clean workspace, declared materials와 최소 network로 실행한다. output exact file set, SBOM와 provenance를 생성하고 자신이 만든 subject만 서명한다. build service identity와 human approver를 분리한다. hidden cache와 nondeterministic metadata를 reproducibility test로 찾는다.

**검증자는** 제출 summary가 아니라 실제 bytes에서 digest와 schema를 다시 계산한다. signature authenticity, subject binding, builder authorization, policy와 revocation을 독립 gate로 평가한다. loader negative fixture와 next-step oracle을 수행한다.

**승인자는** evaluation과 red-team 결과가 같은 subject·config를 가리키는지 본다. exception의 owner·expiry와 남은 위험을 검토한다. alias promotion 뒤 runtime이 실제 digest를 load했는지 확인한다. 승인과 배포 권한을 한 identity에 집중하지 않는다.

**운영자는** cache hit에도 revocation과 policy freshness를 확인한다. runtime SBOM delta, loaded digest와 telemetry를 보고한다. incident에서 artifact를 임의 삭제하지 않고 격리와 evidence 보존을 수행한다. rollback target도 현재 허용되는지 재검증한다.

**데이터 관리자는** acquisition·license·consent·privacy·quality와 training eligibility를 분리한다. 삭제·철회 요청을 lineage graph에서 후손으로 전파한다. filter·dedup·tokenization 변환과 sample-stream 상태를 manifest로 보존한다.

각 역할의 handoff는 문서 링크가 아니라 subject digest, evidence index와 policy decision으로 이루어진다. 사람이 읽는 model card는 이 관계를 설명하지만 machine gate를 대신하지 않는다.

## 27.11 공급망을 승인·commit·revocation 상태 기계로 실행한다

각 단계의 권위 상태와 commit 증거를 명시하면 부분 성공이나 stale approval이 release로 승격되는 일을 막을 수 있다.

artifact의 상태를 폴더 이름이나 Slack 승인으로 표현하지 않는다. `discovered`, `resolved`, `fetched`, `integrity-verified`, `schema-verified`, `behavior-tested`, `attested`, `authorized`, `promoted`, `deployed`, `suspect`, `revoked`, `retired`처럼 machine-readable 상태를 둔다. 모든 전이는 이전 상태, actor, policy generation과 evidence digest를 요구한다.

`resolved`는 floating reference가 immutable digest로 바뀐 상태다. 아직 byte가 로컬에 있거나 안전한 것은 아니다. `fetched`는 byte를 받았다는 뜻이고 integrity 검증 전에는 quarantine에 둔다. `integrity-verified`는 요청 digest와 byte가 맞다는 뜻이지 schema·행동·license가 승인됐다는 뜻은 아니다.

`schema-verified`는 file format과 model bundle 계약을 통과한 상태다. safetensors offset, expected key, config/tokenizer 호환성과 exact file set을 포함한다. `behavior-tested`는 sandbox load와 golden fixture를 통과한 상태다. remote code의 악성 부재를 증명한다고 확대하지 않는다.

`attested`는 provenance와 signature가 subject에 결합된 상태다. `authorized`는 현재 policy가 builder, material, vulnerability, license와 release channel을 허용한 상태다. policy가 바뀌면 integrity나 attestation은 그대로여도 authorization을 다시 계산한다. `promoted`는 alias가 subject digest로 원자 변경됐고 audit event가 durable한 상태다.

`deployed`는 control plane의 의도가 아니라 runtime agent가 실제 load digest와 config를 보고해 일치한 상태다. 일부 replica가 다른 digest를 load하면 fleet 전체를 하나의 deployed 상태로 표시하지 않는다. rollout cohort별 상태와 mismatch를 보존한다.

`suspect`는 compromise interval, 새 advisory, 데이터 철회나 검증기 결함 때문에 조사가 필요한 상태다. production 신규 load를 막되 evidence를 보존한다. 조사 결과 잘못된 경보면 새 policy/evidence로 재승인하고, 실제 영향이면 `revoked`로 간다. revoked에서 promoted로 직접 돌아가는 전이는 금지하고 clean rebuild 또는 명시적 재검증 경로를 거친다.

`retired`는 정상 수명 종료다. revoked와 다르다. historical reproducibility와 audit를 위해 metadata·digest·증거를 보존하되 byte retention은 license·privacy policy에 따른다. alias에서 사라졌다는 이유로 graph node를 삭제하지 않는다.

**동시성과 fencing.** 두 pipeline이 같은 alias를 승격하려 하면 expected generation을 가진 compare-and-swap을 사용한다. approval은 특정 evidence-index generation에 묶인다. 철회와 promotion이 경쟁하면 revocation generation을 검사해 오래된 승인 transaction이 이기지 못하게 한다. cache loader도 policy·revocation generation을 decision에 넣는다.

**idempotency.** 재시도된 attestation upload, scan과 promotion이 중복 event나 다른 결과를 만들지 않게 subject·operation·policy generation으로 idempotency key를 만든다. 외부 scanner가 시간이 지나 다른 database로 결과를 내면 같은 operation 재시도가 아니라 새 scan generation이다.

### 실제 함수 경계를 따라 검토하는 방법

공급망 감사를 repository README에서 끝내지 않는다. resolver의 public API에서 시작해 reference를 commit으로 바꾸는 함수, cache path를 선택하는 함수, download 완료를 원자 승격하는 함수와 digest 검사를 따라간다. 실패·retry·offline 분기가 같은 검증을 통과하는지 확인한다. happy path에서만 digest를 검사하고 cache hit에서 생략하는 버그가 자주 위험하다.

loader에서는 configuration resolution, dynamic class selection, file dispatch, deserializer와 `load_state_dict` 경로를 따라간다. 옵션이 어느 boolean·enum state를 바꾸고 이후 어떤 branch를 여는지 적는다. `trust_remote_code`, `local_files_only`, `revision`, `subfolder`, `use_safetensors`, `weights_only`, `strict`는 서로 다른 경계를 바꾸며 하나가 나머지를 암시하지 않는다.

serialization에서는 writer가 tensor view, shared storage, dtype와 metadata를 어떻게 처리하는지 보고 reader의 검증과 대칭인지 확인한다. shard index 생성 함수와 shard writer 사이 partial failure를 찾는다. test에는 empty tensor, zero dimension, non-contiguous input, shared/tied weight, huge header, truncated body와 duplicate mapping을 넣는다.

provenance verifier에서는 payload decoding, signature verification, certificate identity extraction, transparency proof, subject comparison, predicate validation과 authorization 호출을 구분한다. error가 하나의 `verification failed`로 뭉개지면 운영자가 잘못된 복구를 선택한다. reason code는 공격자에게 과도한 내부 정보를 주지 않는 범위에서 안정적으로 정의한다.

policy engine에서는 input normalization, rule evaluation, exception와 default decision을 본다. unknown field나 schema가 무시되는지 거부되는지 확인한다. time·network 같은 비결정 입력을 통제한다. decision cache key가 artifact뿐 아니라 policy, revocation과 principal/action을 포함하는지 시험한다.

catalog에서는 immutable object create, alias compare-and-swap, state transition, audit append와 rollback의 transaction 경계를 본다. database commit 뒤 object upload가 실패하거나 그 반대인 경우를 주입한다. reconciler가 incomplete transaction을 안전하게 수습하는지 확인한다.

runtime에서는 deployment spec resolution, node cache, loader, process inventory와 health report를 잇는다. control plane의 desired digest와 process가 mmap한 actual digest를 비교한다. sidecar report만 믿지 않고 가능한 경우 process namespace와 file descriptor에서 교차 검증한다.

### 공급망 회귀 시험 묶음을 유지한다

fixture repository에는 작고 결정적인 정상 bundle과 변조 variant를 둔다. 정상 bundle은 tiny safetensors shards, index, config, tokenizer, 최소 remote-code 예제, provenance, signature test key와 SBOM을 포함한다. production secret이나 실제 대형 model은 필요하지 않다.

format fixture는 header length overflow, malformed JSON, overlapping·gapped offset, shape-byte mismatch, missing·extra·duplicate key, shard 누락과 잘린 download를 다룬다. loader fixture는 unsafe pickle global, allowlist 경계, remote import, optional dependency, symlink 교체와 offline cache를 다룬다.

provenance fixture는 wrong subject, unknown predicate, unauthorized builder, expired certificate, stale trust root, invalid transparency proof와 undeclared dependency를 포함한다. policy fixture는 exception expiry, channel mismatch, revoked signer, vulnerability threshold와 license unknown을 다룬다.

catalog fixture는 concurrent promotion, promotion 중 revocation, audit append 실패, mirror metadata 지연과 stale node cache를 주입한다. checkpoint fixture는 missing rank shard, incomplete manifest, wrong optimizer mapping과 one-step-old sampler를 포함한다. 각 fixture는 expected gate, reason code와 state transition을 exact 지정한다.

회귀 시험은 verifier library upgrade, policy change, model format·Transformers upgrade와 storage backend 변경 때 실행한다. 결과가 달라졌다면 더 안전해졌다고 직감하지 않고 semantic diff를 검토한다. 새 parser가 이전 malformed file을 허용하거나 정상 tied-weight bundle을 거부할 수 있다.

fixture 자체도 versioned artifact다. test key와 trust root는 production과 분리한다. known-bad sample이 실수로 release catalog에 promotion되지 않게 namespace와 policy를 분리한다. 예상 결과는 코드와 같은 변경에서 무심코 갱신하지 않고 독립 검토자가 위험 시나리오를 확인한다.

### 28장으로 넘길 재현 계약

단일 GPU golden run은 27장에서 만든 고정 bundle을 입력으로 받는다. bundle에는 base model·tokenizer·dataset fixture·trainer source·environment의 digest, loader policy, checkpoint schema와 claimed reproducibility grade가 있다. 28장은 이 identity를 바꾸지 않고 작은 학습 실행의 다음-step oracle과 artifact 생성을 검증한다.

golden run이 새 checkpoint를 만들면 parent material, exact config, sample-stream prefix, RNG와 environment를 provenance로 남긴다. tracker run ID는 보조 계보다. authoritative identity는 checkpoint generation과 subject digest다. loss curve screenshot만 27장의 검증 패키지로 돌아오지 못한다.

29장의 multi-node fault campaign에는 stale cache, partial checkpoint, signer·policy bundle 만료, runtime SBOM delta와 wrong-rank shard fixture를 넘긴다. fault가 발생하면 artifact state가 `suspect`로 전이되고 last known-good generation이 현재 policy에서 허용되는지 확인한다.

30장의 end-to-end release는 evidence DAG의 모든 leaf가 trusted material에 도달하는지 검사한다. SFT, adapter merge, quantization과 evaluation이 같은 parent를 가리켜야 한다. promotion 뒤 runtime actual digest와 model card subject가 일치해야 한다. 데이터나 signer 철회 query가 전체 후손을 찾는지도 release drill에 포함한다.

이 인계 계약 덕분에 재현성은 별도의 문서 장식이 아니다. golden run의 실행 입력, failure injection의 검증 oracle, release promotion의 승인 조건으로 소비된다. 공급망 정보가 후속 단계의 branch를 실제로 바꾸지 않는다면 아무리 자세한 manifest도 살아 있는 통제가 아니다.

**모델 카드와 기계 판독 manifest의 역할을 구분한다**

모델 카드는 사람이 목적, 구조, 학습 데이터의 범주, 평가, 한계와 사용 조건을 이해하게 한다. manifest는 도구가 exact artifact, revision, dependency, policy와 검증 결과를 계산하게 한다. 모델 카드에 hash를 몇 개 적었다고 machine manifest가 되지 않고, manifest가 완전하다고 독자가 위험을 이해하는 것도 아니다.

모델 카드의 base model, dataset, tokenizer, training framework와 evaluation 이름은 manifest의 stable ArtifactID와 연결한다. 사람이 읽는 floating link만 두지 않는다. 공개할 수 없는 dataset은 접근 제한 이유, 공개 가능한 aggregate와 검증 절차를 설명하고 내부 manifest commitment를 연결한다. 숨은 정보를 상상으로 채우지 않는다.

성능 표는 평가 artifact의 subject, harness revision, prompt/template, decoding, seed와 sample set을 가리킨다. best result만 골라 쓴 표는 release evidence가 아니다. 여러 seed나 checkpoint를 비교해 선택했다면 selection procedure와 전체 후보 계보를 남긴다. test set을 선택에 반복 사용한 사실도 기록한다.

한계와 안전 절은 red-team result, known failure slice와 mitigation state를 연결한다. 정책 문구가 바뀌었는데 model byte가 그대로라면 card revision과 artifact revision을 분리한다. 독자는 어느 card가 어느 subject에 적용되는지 알 수 있어야 한다. 오래된 card가 최신 alias를 설명하는 것처럼 보이지 않게 한다.

license·citation·attribution은 material graph에서 생성하되 사람 검토를 거친다. 자동 생성이 누락이나 충돌을 숨기지 않게 unknown 상태를 표시한다. model card 자체도 소스 리비전, reviewer, generated fields와 manual edits를 가진 artifact로 서명·보존한다.

**최종 승인자가 실행할 증거 순회**

승인자는 production alias가 가리킬 subject digest에서 시작한다. evidence index의 child digest를 모두 다시 계산하고 누락·extra file을 찾는다. provenance subject가 같은 bundle인지, builder와 materials가 authorization policy에 맞는지, signature·transparency와 revocation이 현재 시각에 유효한지 확인한다.

다음으로 SBOM과 runtime inventory의 예상 차이를 검토한다. critical vulnerability exception은 owner·완화·expiry가 있어야 한다. safetensors와 state schema, tokenizer/config, remote code closure와 pickle policy의 negative fixture가 통과했는지 본다. format parser 통과를 model 의미 검증으로 대체하지 않는다.

재현성 등급과 oracle을 읽는다. build byte 재현, checkpoint next-step, statistical evaluation 가운데 무엇이 실제 실행됐는지 확인한다. hardware·driver·CUDA·NCCL과 distributed topology가 비교 가능한지 본다. 결과가 다를 때 최초 divergence와 설명이 있는지, 단순히 허용 오차를 늘리지 않았는지 검토한다.

데이터 graph에서 source·license·consent·filter·mixture와 sample-stream manifest를 순회한다. 철회된 material과 unknown edge가 없는지 query한다. base model, adapter, merge, quantization과 evaluation의 parent가 같은지 본다. 모델 카드의 주장과 기계 manifest subject를 대조한다.

promotion transaction을 dry-run하고 concurrent update와 revocation generation을 확인한다. canary는 digest를 직접 load하고 actual runtime identity를 보고해야 한다. rollback predecessor도 현재 policy에서 허용되는지 확인한다. telemetry와 checkpoint가 끊긴 canary의 성공 응답만으로 승격하지 않는다.

승인 결과는 주관적 코멘트가 아니라 gate별 pass·fail·not-applicable, evidence digest, policy, reviewer와 시각으로 남긴다. not-applicable에는 이유와 적용 범위가 필요하다. exception은 자동 만료되며 만료 뒤 alias와 deployment가 어떤 상태로 전이하는지 시험한다.

**공급망과 재현성의 최종 판정**

이 장의 핵심은 “파일을 믿을 수 있는가”라는 하나의 질문을 여러 독립 질문으로 분해하는 데 있다. 받은 byte가 요청한 digest와 같은가. 형식과 model schema가 유효한가. 실행 경로가 허용 capability 안에 있는가. 누가 어떤 입력으로 만들었다는 증명이 유효한가. 그 주체가 해당 release에 권한이 있는가. 현재 철회·취약점·license policy가 사용을 허용하는가. 독립 환경에서 주장한 재현 등급을 확인했는가.

이 질문 가운데 하나의 성공으로 다른 성공을 추론하지 않는다. safetensors는 임의 객체 실행면을 줄이지만 올바른 model이나 승인된 출처를 보장하지 않는다. signature는 authenticity를 보이지만 subject completeness와 authorization을 보장하지 않는다. SBOM은 구성 요소를 열거하지만 취약 경로의 도달성과 학습 결과의 안전성을 보장하지 않는다. seed 고정은 sample stream, environment와 checkpoint 상태를 자동 복원하지 않는다.

반대로 이 경계를 정확히 지키면 실패는 막연한 “재현이 안 된다”에서 구체적인 상태로 바뀐다. material resolution mismatch, cache integrity failure, loader schema failure, hidden runtime dependency, RNG·sampler divergence, unauthorized builder, stale revocation과 incomplete descendant closure처럼 owner와 다음 시험이 있는 문제로 좁혀진다.

최종 산출물은 모델 파일 하나가 아니다. immutable bundle, 산출물 DAG, provenance·signature, SBOM·runtime inventory, loader 검증, 재현 oracle, policy decision, revocation·clean rebuild 절차와 사람이 읽는 모델 카드가 같은 subject를 중심으로 결합된 evidence package다. 이 package를 28장의 golden run이 실제로 소비하고 29장의 장애 주입이 깨뜨려 보며 30장의 release transaction이 검증할 때 공급망은 문서가 아니라 실행 가능한 신뢰 경계가 된다.

**혼동하기 쉬운 용어를 불변량으로 정리한다**

`identity`는 artifact를 다른 artifact와 구분하는 digest다. `integrity`는 관측한 byte가 그 identity와 일치한다는 주장이다. `authenticity`는 정한 주체가 statement에 서명했다는 주장이고, `authorization`은 그 주체와 입력이 특정 행동에 허용된다는 현재 정책 판정이다. 네 단어를 “신뢰됨” 하나로 합치지 않는다.

`provenance`는 subject가 어떤 builder·definition·material에서 나왔다는 계산 가능한 계보다. `reproducibility`는 정한 입력과 환경에서 정한 oracle을 다시 만족하는 성질이다. provenance가 자세해도 숨은 입력이 있으면 재현에 실패할 수 있고, 우연히 결과가 같아도 provenance가 없으면 그 결과가 어떤 과정에서 나왔는지 증명되지 않는다.

`SBOM`은 구성 요소와 관계의 inventory다. vulnerability report는 특정 advisory database와 시각에서 SBOM을 평가한 결과다. runtime reachability와 exploitability는 추가 분석이다. SBOM에 없다는 사실은 component 부재의 증거가 아니라 inventory coverage 실패일 수도 있다.

`signature validity`는 cryptographic 계산, certificate와 transparency proof를 통과한 결과다. `subject binding`은 그 서명이 실제 배포 byte를 가리키는지 확인한다. `signer authorization`은 signer가 해당 repository·workflow·channel에 권한이 있는지 묻는다. 세 gate의 failure reason을 분리한다.

`cache hit`는 재사용 가능한 local entry를 찾았다는 뜻일 뿐이다. digest verification, completeness, current authorization과 revocation freshness가 뒤따라야 한다. `offline`은 network 접근이 없다는 실행 조건이지 trust가 고정됐다는 뜻이 아니다. offline policy bundle에도 generation과 만료가 있다.

`checkpoint complete`는 writer가 종료됐다는 뜻이 아니라 manifest에 열거된 모든 component와 shard가 digest 검증되고 complete generation으로 원자 승격됐다는 뜻이다. `load success`는 deserializer가 반환했다는 뜻이고, `resume success`는 다음 batch·loss·update·scheduler가 정한 oracle을 만족한다는 뜻이다.

`deterministic`은 정한 환경과 입력에서 실행 선택을 제한하는 성질이다. `bitwise identical`은 output byte가 같은 강한 결과다. `statistically reproducible`은 미리 정한 반복·metric·허용 구간을 만족한다. 어느 수준인지 말하지 않은 “재현 가능”은 검증할 수 없는 문장이다.

`revoked`는 현재 사용이 금지된 상태이고 `retired`는 정상 수명 종료다. `deleted`는 특정 storage에서 byte가 제거된 사실이며 graph·backup·cache와 후손 artifact 전체의 처리 완료를 자동 의미하지 않는다. `clean rebuild`는 단순 재실행이 아니라 신뢰 가능한 graph cut에서 오염된 material과 cache를 제거한 새 계보를 만드는 절차다.

코드 리뷰, incident, model card와 release 회의에서는 이 불변량을 같은 언어로 쓴다. 누군가 “서명됐으니 안전하다”, “seed가 같으니 재현됐다”, “cache라서 같은 파일이다”라고 말하면 어떤 gate가 생략됐는지 즉시 물을 수 있다. 명확한 용어는 문체의 문제가 아니라 공급망 상태를 잘못 승격하지 않게 하는 correctness 장치다.

마지막 검토에서는 임의의 production replica 하나를 선택해 역추적한다. process가 실제 load한 tensor bundle과 config digest에서 deployment, promotion decision, 독립 검증 report, provenance, builder와 모든 material까지 도달해야 한다. 이어 임의의 dataset shard나 native library에서 정방향으로 출발해 그것을 소비한 run, checkpoint, adapter, export와 replica를 찾는다. 두 방향의 결과가 일치해야 graph가 단순 기록 목록을 넘어 영향 분석에 사용될 수 있다.

검토자는 그 경로 중 하나의 signature bundle을 다른 artifact와 바꾸고, revocation metadata를 오래된 generation으로 낮추며, cache blob 하나를 같은 이름의 다른 byte로 교체한다. 각 변조가 기대한 독립 gate에서 거부되고 다른 gate의 성공으로 덮이지 않는지 확인한다. 실패 reason, artifact state와 IncidentID가 일관되어야 한다.

정상 경로도 다시 실행한다. clean cache와 offline bundle에서 동일 subject가 해석되고 golden next-step oracle이 주장한 재현 등급을 만족해야 한다. policy와 trust bundle, runtime inventory가 evidence index에 고정되고 model card가 같은 subject를 설명하는지 확인한다. 이 양방향 graph 순회, 변조 시험과 clean 검증이 모두 통과하면 27장의 공급망 계약을 다음 장의 실행 입력으로 넘길 수 있다.

인계 기록에는 검사 도구의 revision, fixture bundle digest, 실행 환경, 시작·종료 시각과 gate별 원시 결과를 넣는다. 요약 보고서만 남기지 않는다. 다음 장에서 차이가 생기면 동일 입력을 다시 검증해 공급망 변화인지 학습 실행 변화인지 즉시 분리할 수 있어야 한다. 이 분리 가능성이 재현성 자료의 실질적인 가치다.

모든 판정은 자동 재실행 가능해야 하며 수동 예외는 책임자와 만료 조건을 가져야 한다.

**공급망 장부를 실제 자료 구조로 설계한다**

공급망 검증이 문서 검토에 머무는 가장 흔한 이유는 장부의 기본 키가 파일 이름이기 때문이다. `model.safetensors`, `config.json`, `train.py`는 사람이 알아보기 좋은 별칭이지 identity가 아니다. 같은 이름의 byte가 계속 바뀔 수 있고, 한 byte 묶음이 여러 이름으로 복제될 수도 있다. 장부의 기본 키는 `algorithm:digest`이고 이름, URI, repository, branch, tag와 release channel은 모두 관측 가능한 별도 속성으로 둔다. 디렉터리도 파일 목록을 canonical order로 정렬해 각 경로, mode, size와 content digest를 묶은 tree digest로 식별한다. 그래야 빈 파일 추가, 대소문자 충돌, symlink 변경과 실행 bit 변경도 새 artifact로 드러난다.

최소 record에는 `artifact_id`, `media_type`, `byte_size`, `created_at`, `observed_at`, `storage_locations`, `producer_run`, `materials`, `policy_generation`, `classification`, `retention`, `verification_state`가 필요하다. `created_at`은 producer의 주장이고 `observed_at`은 verifier가 byte를 본 시각이다. 두 값을 합치면 clock skew나 backfill을 놓친다. location은 identity가 아니므로 mirror가 늘어도 artifact record는 새로 만들지 않는다. 반대로 동일 location의 byte가 바뀌면 반드시 새 identity가 생기고 이전 record의 관측 이력을 지우지 않는다.

관계도 동사까지 고정한다. `consumed`는 실행이 입력으로 읽었다는 뜻이고, `declared-material`은 provenance가 입력이라고 주장한다는 뜻이며, `derived-from`은 의미적 부모다. `packaged-with`, `evaluated-by`, `signed-by`, `authorized-by`, `revokes`, `supersedes`를 한 종류의 `related_to`로 뭉치면 영향 반경을 계산할 수 없다. 실행 trace에서 관측한 `consumed`와 제출 provenance의 `declared-material` 차집합은 숨은 입력과 과잉 선언을 찾는 핵심 질의다.

예를 들어 run `R7`이 source tree `S`, dataset manifest `D`, base checkpoint `B`, image `I`를 선언했는데 runtime file trace가 cache blob `C`와 host library `H`도 읽었다고 하자. 선언 집합은 `{S,D,B,I}`, 관측 집합은 `{S,D,B,I,C,H}`다. `observed - declared={C,H}`이므로 재현 등급을 승인할 수 없다. 반대로 `declared - observed={B}`라면 base checkpoint가 실제로 사용되지 않았거나 trace coverage가 불완전하다. 어느 쪽도 “대부분 같다”로 통과시키지 않는다.

장부 쓰기는 append-only event와 현재 projection을 분리한다. `DISCOVERED`, `HASH_VERIFIED`, `SCHEMA_VERIFIED`, `PROVENANCE_VERIFIED`, `AUTHORIZED`, `QUARANTINED`, `REVOKED`, `PROMOTED` event를 순서대로 보존하고 현재 상태는 event를 fold해 계산한다. 운영자가 잘못된 승인을 취소해도 과거 event를 삭제하지 않고 `DECISION_REVOKED`를 추가한다. incident 조사에서 누가 언제 무엇을 알고도 승격했는지가 남아야 하기 때문이다.

동시 승인에는 compare-and-swap을 쓴다. 승인자가 읽은 alias generation이 41이고 새 subject `sha256:a`를 승격하려 할 때 transaction은 `expected_generation=41`을 조건으로 건다. 그사이 다른 승격이 generation 42를 만들었다면 쓰기를 거부하고 policy, revocation과 canary를 다시 평가한다. 마지막 writer가 무조건 이기는 registry는 검증을 통과한 subject를 오래된 판정으로 덮어쓸 수 있다.

손으로 작은 tree digest도 검산해 보자. 여기서는 설명을 위해 실제 SHA-256 대신 `H(x)`를 쓰고, 파일 leaf를 `H(path || 0x00 || size || 0x00 || content_digest)`로 정의한다. `config.json` 100 byte의 digest가 `c1`, `model.safetensors` 900 byte의 digest가 `c2`라면 leaf는 각각 `l1=H("config.json\0 100\0 c1")`, `l2=H("model.safetensors\0 900\0 c2")`다. bundle identity는 정렬된 `H(l1 || l2)`다. model byte가 같아도 config 한 byte가 바뀌면 `c1`, `l1`, bundle identity가 모두 바뀐다. 파일별 hash만 나열하고 bundle root를 만들지 않으면 서로 다른 release의 파일을 섞은 Frankenstein bundle을 탐지하기 어렵다.

이 구조는 28장의 golden run에도 직접 쓰인다. resolved config, GoldenBatch, checkpoint와 profiler trace가 각각 artifact이고 run은 이들을 생산·소비하는 node다. 29장의 rank shard는 동일 checkpoint generation 아래 component edge로 묶인다. 30장의 promotion은 새 byte 생성이 아니라 특정 evidence graph와 policy generation에 대한 승인 event다. 네 장이 같은 identity와 event 언어를 써야 사고 때 역방향과 정방향 질의가 끊기지 않는다.

**Transformers 저장·로딩 경계에서 manifest를 붙인다**

고정 revision `550d7b3834670483a4df436541272c055dc364bf`의 `src/transformers/modeling_utils.py:3278`에 있는 `PreTrainedModel.save_pretrained`는 state dict, safe serialization, shard 크기, variant와 push 여부를 받아 실제 파일 묶음을 만든다. 같은 파일 `modeling_utils.py:3859`의 `from_pretrained`는 config, cache, revision, dtype, device map과 loading option을 해석한다. 두 함수 이름이 대칭적으로 보여도 저장 시점의 Python 객체와 로딩 시점의 실행 상태가 자동으로 동일해지는 것은 아니다. 공급망 manifest는 두 호출 사이의 의미 손실을 메워야 한다.

저장 전에는 model class의 fully qualified name, config canonical digest, state-dict key 집합, key별 shape·dtype·stride 정책, tied-weight relation과 serialization option을 기록한다. 저장 뒤에는 생성된 index와 shard 전체를 다시 읽어 key coverage와 digest를 계산한다. “함수가 예외 없이 반환했다”는 writer 성공이지 durable bundle 검증이 아니다. remote filesystem이나 async uploader가 있다면 local close, upload complete, remote read-back과 manifest publish를 서로 다른 event로 남긴다.

로딩 전에는 요청한 repository와 revision을 immutable commit으로 resolve하고 allow pattern과 deny pattern을 기록한다. `src/transformers/utils/hub.py:166` 부근의 snapshot resolution과 같은 파일 `:453` 부근의 `snapshot_download` 호출은 cache가 경로를 제공하는 지점이지 모델 의미를 승인하는 지점이 아니다. 반환된 snapshot path에서 실제로 열린 파일의 digest를 수집해 요청 manifest와 대조한다. cache hit 여부는 성능 정보이고 integrity 판정과 분리한다.

로딩 뒤에는 missing key, unexpected key, mismatched shape, dtype cast, device placement와 offload map을 artifact로 남긴다. `strict=False`와 유사한 관대한 경로는 개발에는 편하지만 출시 관문에서는 차이 목록이 비어 있거나 승인된 migration과 정확히 일치해야 한다. warning log를 잃으면 관대한 로드가 정상 로드처럼 보인다. warning 문자열 파싱에 의존하지 말고 가능하면 loader가 반환하는 structured loading info를 보존한다.

간단한 예를 들자. manifest에는 `embed.weight [32000,4096] bf16`, `lm_head.weight`가 embed와 tied라고 적혀 있다. 소비자 config가 vocab을 32008로 늘리고 자동 resize를 수행하면 loader는 성공할 수 있지만 마지막 8행은 새로 초기화되고 tie가 재설정될 수 있다. 원래 tensor byte digest만 확인하면 이 runtime 변형을 놓친다. runtime model manifest는 `[32008,4096]`, 초기화된 row 범위, 최종 tie relation과 parameter digest를 별도로 가져야 한다. 이 모델을 base로 adapter를 학습하면 resize event가 새 parent edge가 된다.

반례로 shard index만 서명하고 shard body를 서명하지 않는 절차를 생각하자. 공격자가 index의 파일 이름은 유지한 채 shard 하나를 교체해도 JSON signature는 유효하다. 반대로 shard만 개별 서명하고 index를 보호하지 않으면 key-to-shard mapping을 바꾸거나 일부 shard를 빼는 공격이 가능하다. bundle root가 index와 모든 shard leaf를 함께 commit해야 한다. 소비자는 index 선언 key 집합, shard 내부 key 집합과 config 기대 key 집합의 삼자 일치를 확인한다.

복구는 “다시 다운로드”보다 구체적이어야 한다. cache blob digest가 틀리면 해당 immutable identity만 quarantine하고 같은 cache root의 다른 검증된 blob은 유지할 수 있다. snapshot symlink가 잘못됐으면 ref와 snapshot projection을 재생성하되 content-addressed blob은 다시 hash한다. upstream revision 자체가 철회됐으면 mirror를 바꿔 같은 byte를 얻는 것으로 해결되지 않는다. 승인된 새 parent에서 clean rebuild하고 모든 후손을 새 계보로 만든다.

**SBOM을 build-time, load-time, kernel-time으로 검산한다**

Python lockfile은 import 후보의 목록이지 GPU가 실행한 코드의 목록이 아니다. 한 학습 process에는 Python distribution, pure Python module, ELF shared object, CUDA runtime·driver API, JIT 또는 ahead-of-time extension, 커널 image, system library와 외부 executable이 겹친다. 따라서 SBOM을 세 층으로 나눈다. build SBOM은 image와 wheel을 만든 재료, load SBOM은 process가 실제 mmap·dlopen한 binary, kernel inventory는 CUDA context가 실행한 cubin·PTX와 JIT 산출물이다.

각 component에는 package URL이나 CPE만 쓰지 말고 content digest, build ID, compiler, target architecture, ABI, 소스 리비전과 build flags를 넣는다. CUDA extension이라면 `TORCH_CUDA_ARCH_LIST`, nvcc version, host compiler, `_GLIBCXX_USE_CXX11_ABI`, debug/fast-math flag가 결과를 바꾼다. PTX가 driver에서 JIT되면 driver version과 JIT cache artifact도 실행 계보에 들어간다. 같은 wheel hash라도 다른 GPU architecture에서 선택되는 kernel image가 다를 수 있다.

차집합 검산은 네 집합으로 한다. 선언 dependency `D`, image scan `I`, process load `L`, kernel observation `K`가 있을 때 `L-I`는 host 주입이나 scanner 누락, `I-D`는 transitive 또는 잔여 package, `D-I`는 build 누락, `K-L`은 JIT나 instrumentation coverage 문제를 의심한다. 예컨대 `|D|=120`, `|I|=138`, `|L|=67`, `|K|=14`이고 `L-I`에 host의 `libcuda.so` 하나가 있다면 67개가 적다는 이유로 통과시키지 않는다. 그 한 component가 실행의 핵심 신뢰 경계다.

취약점 판정도 단계적이다. advisory가 component version과 맞는지, 해당 code path가 build에 포함됐는지, process에서 load됐는지, 공격자 입력이 취약 symbol에 도달하는지, 완화가 실제 적용됐는지를 구분한다. reachability가 낮아도 release policy가 금지할 수 있고, CVE가 없어도 출처 불명 binary는 거부할 수 있다. “취약점 0개”는 scanner DB 시각과 coverage를 함께 쓰지 않으면 의미가 없다.

native extension을 source에서 다시 만들 때 bitwise equality가 안 나면 즉시 실패로만 끝내지 않는다. archive member timestamp, section ordering, build path embedding, compiler nondeterminism을 먼저 분리하고 normalize 가능한 metadata와 executable section을 따로 비교한다. 단, normalize한 digest를 배포 identity로 몰래 대체하지 않는다. 실제 byte identity와 semantic comparison 결과를 둘 다 보존한다.

커널 시간의 관측은 28장 profiler와 연결된다. golden run이 실행한 operator, selected kernel, input shape와 dtype을 trace로 남기면 SBOM component가 실제 workload에 도달했는지 판단할 수 있다. 29장에서는 rank마다 loaded library와 kernel inventory가 같아야 하며, heterogeneous node가 의도됐다면 rank별 차이가 topology manifest와 일치해야 한다. 30장 canary는 build SBOM만 받지 말고 runtime delta를 release evidence로 되돌려야 한다.

**데이터 공급망의 수치 보존을 검증한다**

데이터 provenance는 URL과 license 목록만으로 부족하다. 학습 의미를 바꾸는 것은 row 선택, parser, normalization, filtering, deduplication, contamination removal, mixture와 sample ordering이다. 각 stage를 content-addressed transform으로 보고 input manifest, code revision, canonical config, random state, output manifest와 rejection ledger를 연결한다. raw text를 공개할 수 없어도 cryptographic commitment와 aggregate, 검증 가능한 restricted report를 남길 수 있다.

변환마다 보존식이 있다. parser 단계는 `input_objects = parsed_ok + rejected_corrupt + rejected_policy`처럼 상호 배타적인 count를 가져야 한다. filter chain은 각 row가 최초 탈락 이유 하나와 추가 진단 태그를 갖게 해 이중 집계를 막는다. dedup은 row 수뿐 아니라 cluster 수, representative 선택 규칙과 제거된 token mass를 기록한다. mixture는 문서 비율과 token 비율을 구분한다.

손계산 예에서 A corpus는 100문서, 문서당 평균 1,000 token이고 B는 1,000문서, 평균 100 token이다. 문서 수 비율은 A:B=1:10이지만 token mass는 100,000:100,000으로 1:1이다. 문서 기반 50:50 sampler를 쓰면 기대 token 기여는 A가 약 90.9%, B가 9.1%가 된다. 반대로 token 기반 50:50을 원하면 sample probability는 길이와 packing을 고려해야 한다. manifest에 “A 50%, B 50%”만 쓰면 재현과 모델 해석이 모두 불가능하다.

dedup의 경계도 수치로 남긴다. 10,000 row 중 exact duplicate 500개, near-duplicate cluster에서 추가 700개를 제거했다면 output 8,800이라는 count만으로 부족하다. exact hash algorithm, text normalization, MinHash tokenizer, n-gram 크기, signature 수, LSH threshold, cluster union 순서와 representative rule이 필요하다. tie-break가 input order라면 shard 병렬성이 결과를 바꾼다. stable RowID 순으로 tie를 깨거나 결과 차이를 새 dataset generation으로 인정한다.

삭제 요청은 raw row만 지우는 작업이 아니다. RowID에서 normalized row, packed sample, training run과 checkpoint로 향하는 graph를 질의한다. 이미 학습된 weight에서 개별 row의 영향을 완전히 제거했다는 과장된 주장을 하지 않는다. 대신 접근 차단, future dataset generation 제외, affected model 식별, 재학습·unlearning 여부와 평가를 명시한다. backup과 offline cache도 별도 삭제 상태를 가진다.

poisoning 탐지는 provenance와 통계가 만나야 한다. 새 source에서 특정 rare token, 반복 n-gram, label pattern이나 gradient influence가 급증했을 때 어느 ingestion batch와 transform이 만들었는지 역추적한다. 통계 이상만으로 악성이라고 확정하지 않고, 출처가 승인됐다는 이유로 이상을 무시하지 않는다. quarantine dataset generation으로 작은 golden training을 다시 실행해 loss와 gradient 최초 차이를 확인한다.

**서명 검증의 실패를 단계별 수학으로 읽는다**

서명 검증은 `Verify(pk, signature, message)=true` 한 줄보다 넓다. 먼저 canonical statement byte를 만들고 그 digest가 서명 message와 같은지 본다. statement subject digest가 다운로드 bundle root와 같은지 확인한다. certificate chain과 유효 시간, transparency inclusion·checkpoint, identity claim을 검증한다. 마지막으로 policy가 그 identity에게 repository, workflow, branch, environment와 release channel에 대한 권한을 주는지 평가한다.

여섯 boolean을 `C` canonical parse, `S` cryptographic signature, `B` subject binding, `T` transparency·time, `I` expected identity, `A` authorization이라고 두면 promotion은 `C∧S∧B∧T∧I∧A`다. 하나라도 false면 실패다. `S=true`만 dashboard에 크게 표시하면 나머지 다섯 실패를 숨긴다. `unknown`도 true로 간주하지 않는다. offline에서 transparency freshness를 확인할 수 없다면 policy에 따라 `pending` 또는 제한된 이전 generation 사용으로 전이한다.

예를 들어 signature와 certificate는 유효하지만 certificate identity가 `pull_request` workflow이고 production policy는 protected `release` workflow만 허용한다고 하자. `C,S,B,T,I=true`, `A=false`다. “정상 GitHub identity가 서명했다”는 설명으로 승격하면 안 된다. 반대로 identity와 authorization이 맞아도 subject가 다른 digest면 `B=false`다. 같은 release 이름이라는 이유로 대체하지 않는다.

키 침해 시점 `t_c`와 signature integrated time `t_i`가 있을 때 단순히 `t_i<t_c`인 서명을 모두 안전하다고 보기도 어렵다. 공격자가 과거 시각을 주장할 수 있으므로 trusted transparency log의 inclusion time과 certificate validity를 본다. incident 범위는 compromised identity, workflow, repository와 시간 구간을 graph predicate로 만들고 모든 subject와 후손을 찾는다. old signature를 새 key로 재서명하는 것만으로 오염된 builder 결과가 깨끗해지지 않는다.

복구 훈련에서는 정상 bundle 네 개를 준비한다. statement byte 변조, subject 교체, 허가되지 않은 identity, 오래된 trust bundle을 각각 하나씩 넣는다. 네 fixture가 cryptographic, binding, authorization, freshness라는 서로 다른 reason code에서 실패해야 한다. verifier가 모든 경우를 `invalid signature`로 뭉치면 운영자는 cache 재다운로드, policy 수정, credential incident 중 무엇을 해야 하는지 알 수 없다.

## 27.12 재현성 oracle과 CUDA binary closure를 함께 검증한다

byte exact, tensor tolerant, behavior bounded라는 서로 다른 재현성 수준을 선언하고 Python lock에서 실제 CUDA binary까지 환경 closure를 닫는다.

재현성은 입력 고정의 선언이 아니라 비교 함수다. 첫째 byte oracle은 산출물 전체 digest가 같은지 본다. 둘째 state oracle은 tensor key·shape·dtype와 선택 값, optimizer·sampler·RNG 상태가 정한 허용 오차 안인지 본다. 셋째 behavioral oracle은 고정 evaluation과 통계적 반복에서 metric 분포가 사전 구간을 만족하는지 본다. 낮은 수준 실패가 높은 수준 성공으로 자동 면제되지는 않으며 목적에 따라 설명과 승인이 필요하다.

부동소수점 tensor 비교에서 `|x-y| <= atol + rtol|y|`를 쓰되 zero 부근과 큰 값의 성질이 다름을 기억한다. 기준 값 `y=0.001`, `atol=1e-5`, `rtol=1e-3`이면 허용치는 `0.000011`이다. 실제 `x=0.001012`의 차이 `0.000012`는 실패다. tensor 전체 max만 보면 한 outlier와 광범위한 작은 drift를 구분하지 못하므로 max, quantile, norm, cosine, nonfinite count와 위치를 함께 기록한다.

통계 oracle은 결과를 본 뒤 범위를 정하지 않는다. seed 5개에서 metric 차이를 보고 “대략 비슷하다”고 쓰지 말고 반복 수, primary metric, equivalence margin과 multiple comparison 처리를 먼저 정한다. 평균 차이가 작아도 variance가 두 배가 되면 운영 안정성이 달라질 수 있다. distribution shift가 큰 slice를 aggregate가 숨길 수 있으므로 safety·language·length slice를 별도 판정한다.

최초 차이 양분은 artifact chain을 따라간다. source tree와 input manifest가 같고 rendered batch digest가 다르면 tokenizer·template·sampler 경계다. batch가 같고 first forward activation이 다르면 model construction, dtype, kernel 또는 parameter load다. activation과 loss가 같고 gradient가 다르면 backward·recompute·scaler다. gradient가 같고 parameter delta가 다르면 clipping·optimizer·scheduler다. checkpoint 직전 state가 같고 resume 뒤 다르면 serializer·reader·RNG 복원이다.

28장의 golden run에서 이 순서를 구체적인 값으로 채운다. 27장은 어떤 artifact가 같아야 비교 자격이 생기는지 보장하고, 28장은 token·loss·gradient·delta의 oracle을 제공한다. 29장은 rank별 oracle을 추가하며 30장은 재현 결과를 promotion policy에 연결한다. 재현 실패를 “CUDA가 비결정적”이라는 포괄적 설명으로 닫지 않는다.

### 공급망 사고를 네 개의 종단 사례로 복구한다

첫 사례는 tokenizer 파일의 조용한 교체다. model tensor digest와 config 이름은 같지만 `tokenizer.json` 한 byte가 달라졌다. bundle root 검증이 즉시 실패하고 cache entry를 quarantine한다. 이미 실행된 run은 consumed artifact graph에서 찾고 checkpoint를 `suspect`로 전이한다. 승인된 tokenizer로 28장 GoldenBatch를 다시 encode해 최초 TokenID 차이를 보존한다. tensor가 변하지 않았으니 괜찮다는 결론은 잘못이다.

둘째는 취약한 native extension이다. advisory가 발표됐지만 package version 문자열은 vendor patch 때문에 모호하다. SBOM content digest와 소스 리비전, loaded library build ID를 대조하고 vulnerable symbol 포함 여부를 확인한다. 영향을 받는 run과 배포를 graph에서 찾고 차단 우선순위를 정한다. 새 extension을 clean build한 뒤 28장의 correctness와 profiler kernel inventory를 비교한다. 성능만 같고 수치 oracle이 다르면 승격하지 않는다.

셋째는 dataset source의 사용 권한 철회다. source node에서 정방향으로 normalized shard, mixtures, runs, checkpoints, adapters와 exports를 찾는다. public alias를 즉시 제한할지 법무·정책과 결정하되 기술 장부는 affected 범위를 축소하거나 과장하지 않는다. future mixture에서는 source를 제외하고 token mass를 재계산한다. 단순 제거가 curriculum과 optimizer step 수를 바꾸므로 replacement와 재학습 config가 새 provenance를 가진다.

넷째는 signer credential 침해다. trust policy generation을 올리고 해당 identity·시간의 attestation을 재평가한다. signature가 있던 artifact를 모두 삭제하지 않고 provenance materials, builder isolation과 independent reproduction으로 위험을 분류한다. 그러나 조사 중인 artifact를 last-known-good라 부르지 않는다. rollback 후보도 현재 policy에서 재검증한다. 새 key로 동일 byte에 도장만 다시 찍는 것은 builder compromise를 복구하지 못한다.

각 사례의 종료 조건은 서비스가 다시 뜨는 것이 아니다. 탐지 rule이 fixture를 잡고, 영향 graph가 독립 검산됐고, quarantine과 alias가 일치하며, clean rebuild 또는 명시적 risk acceptance가 새 evidence generation으로 승인돼야 한다. incident에서 배운 detector를 회귀 suite에 넣고 정상 control도 함께 실행한다.

### 독립 검토자가 상태 계보를 수치로 검산한다

검토자는 manifest 없이 파일 이름만 제공받았을 때 승격을 거부해야 한다. 이어 bundle root와 leaf 12개를 받아 임의 leaf 두 개의 경로·size·digest를 다시 계산한다. signed statement의 subject와 bundle root를 비교하고, signer identity가 해당 channel policy에서 허용되는지 별도로 판정한다. signature success와 authorization success를 한 칸에 쓰지 않는다.

산출물 DAG의 작은 표본도 손으로 푼다. dataset `D1`을 run `R1,R2`가 소비하고, `R1`이 checkpoint `C1`, `R2`가 `C2`, `C1`에서 adapter `A1`, `A1`에서 merged model `M1`이 나왔다면 `D1` 철회의 descendant closure는 `{R1,R2,C1,C2,A1,M1}`다. evaluation `E1`이 `M1`을 평가했다면 evidence로는 보존하되 현재 승인 상태는 parent 철회를 반영한다. edge 방향과 의미가 없으면 이 계산을 할 수 없다.

환경 비교에서는 package 이름이 같은지보다 실행 identity를 본다. builder A와 B의 Python lock이 같아도 A가 host `libstdc++`를, B가 image library를 load했다면 hidden material 차이다. loaded object digest, symbol version과 compiler ABI를 비교한다. 차이가 output에 영향 없다는 주장은 28장의 oracle로 시험하고 재현 등급에 명시한다.

마지막으로 partial checkpoint, stale policy, cache substitution과 wrong remote code revision을 차례로 주입한다. 기대 detector가 각각 completeness, freshness, integrity, execution-closure gate인지 확인한다. 한 fixture가 우연히 앞선 gate에서 막혀 뒤 gate가 시험되지 않으면 fixture를 조정한다. 모든 음성 시험 뒤 untouched control이 통과해야 verifier 자체의 오염을 배제할 수 있다.

이 계산을 독립 검토자가 clean workspace와 고정 trust bundle에서 반복할 수 있을 때 공급망 설명은 운영 계약이 된다. 결과물에는 command, tool revision, raw output, decision record와 생성된 모든 digest를 남긴다. 다음 장은 이 계약을 받아 한 GPU의 매 step이 동일한 input·state·output 계보를 만드는지 확인한다.

### checkpoint가 담아야 할 상태의 폐쇄를 증명한다

checkpoint를 parameter 파일로 정의하면 resume 재현은 구조적으로 불가능하다. 학습 상태의 폐쇄는 “다음 committed update를 계산하는 데 필요한 모든 값”으로 정의한다. model parameter와 buffer, optimizer moment와 internal step, scheduler state, gradient scaler, global·micro step, gradient accumulation 위치, data sampler cursor, packing buffer, Python·NumPy·framework·device generator, curriculum phase, callback state와 resolved config가 포함된다. adapter 학습이면 base 산출물 identity와 trainable mapping도 필요하다.

필수 상태를 찾는 실용적 방법은 다음 update의 데이터 흐름을 역으로 걷는 것이다. 다음 batch는 sampler cursor와 dataset generation에 의존한다. dropout mask는 device RNG에 의존한다. AdamW delta는 parameter, gradient, first·second moment, optimizer step, learning rate와 hyperparameter에 의존한다. learning rate는 scheduler state와 committed step에 의존한다. loss scaling의 commit 여부는 scaler state에 의존한다. 이 dependency closure에 포함되는데 serialize되지 않은 값은 hidden state다.

AdamW의 한 scalar를 계산해 누락 효과를 보자. checkpoint 시점에 `m=0.2`, `v=0.04`, 다음 gradient `g=0.4`, `β1=0.9`, `β2=0.99`라면 새 moment는 `m'=0.9×0.2+0.1×0.4=0.22`, `v'=0.99×0.04+0.01×0.16=0.0412`다. moment를 누락해 0에서 재시작하면 `m'=0.04`, `v'=0.0016`이다. loss와 batch가 같아도 update 방향의 정규화가 달라진다. parameter만 정상 load됐다는 사실은 resume 성공이 아니다.

sampler cursor도 단순 row index가 아닐 수 있다. shuffled permutation의 seed와 epoch, rank, consumed sample count, prefetch queue와 packing 잔여 token이 다음 batch를 결정한다. checkpoint writer가 `global_step=100`만 저장하고 `gradient_accumulation=8`이라는 설정에서 micro-step 3에 멈췄다면 이미 누적된 gradient를 저장할지 버릴지 정책이 필요하다. 가장 단순한 정책은 committed update 경계에서만 durable checkpoint를 publish하는 것이다. 중간 snapshot을 허용한다면 gradient buffer와 micro-step을 함께 저장한다.

generation manifest에는 component마다 required 여부, digest, logical schema와 writer completion을 넣는다. reader는 required component가 모두 있고 digest와 schema가 맞을 때만 generation을 complete로 본다. `latest` pointer를 먼저 쓰고 shard를 나중에 올리는 순서를 금지한다. staging generation을 모두 쓰고 read-back 검증한 뒤 complete manifest를 원자 publish하고 마지막에 alias를 바꾼다.

복구 시험은 같은 process에서 load하지 않는다. in-memory tokenizer, kernel cache, RNG와 open dataloader가 누락 상태를 보충할 수 있기 때문이다. 새 process, 빈 application cache와 명시한 offline material에서 resume한다. uninterrupted branch와 next BatchID, loss numerator·denominator, selected gradient, parameter delta, optimizer state와 scheduler LR을 비교한다. byte equality를 요구하지 않는 항목도 tolerance와 이유를 사전에 둔다.

**adapter·merge·quantization 계보를 손실 없이 잇는다**

adapter는 작다고 독립 model이 아니다. 의미는 base model identity, target module mapping, rank·alpha·dropout, tokenizer·template와 adapter tensor의 조합이다. `adapter_model.safetensors`만 옮기면 같은 이름의 다른 base에 붙을 수 있다. manifest는 base bundle root와 config digest, target parameter의 원래 shape·dtype, insertion rule과 trainable set을 parent relation으로 고정한다.

LoRA에서 base weight `W`와 adapter `A,B`, scale `α/r`가 있을 때 merged weight는 `W' = W + (α/r)BA`다. 작은 예로 `r=1`, `α=2`, `A=[1, -1]`, `B=[0.5, 1]^T`라면 `BA=[[0.5,-0.5],[1,-1]]`, update는 `[[1,-1],[2,-2]]`다. merge artifact 검증기는 parent `W,A,B`와 scale로 선택 원소를 다시 계산할 수 있다. merged file digest만 서명하면 어떤 base와 adapter가 결합됐는지 잃는다.

merge 순서도 material이다. adapter 두 개의 단순 가산은 선형처럼 보여도 dtype cast, quantized base, weight tying과 conflict resolution 때문에 byte 결과가 달라질 수 있다. `merge(A1,A2)`와 `merge(A2,A1)`의 정책, accumulator dtype와 rounding을 기록한다. target key가 겹치면 reject인지 합산인지 명시한다. missing target을 warning으로 넘기지 않는다.

quantization은 원본을 압축한 storage 변형 이상이다. calibration dataset과 preprocessing, observer, group size, axis, scale·zero-point dtype, clipping, kernel layout와 library revision이 새 모델의 행동을 결정한다. quantized artifact는 float parent, calibration data와 quantizer run을 모두 가리킨다. dequantized tensor의 오차와 고정 evaluation을 보존한다. 같은 `int4` 라벨로 서로 다른 group·packing을 호환된다고 보지 않는다.

export 과정이 tokenizer나 generation config를 복사한다면 복사본 digest가 원 parent와 같은지 확인한다. runtime format이 tensor 이름을 바꾸면 mapping table을 artifact로 보존한다. 일부 parameter를 fold하거나 transpose하면 shape만 비교해선 부족하다. 선택 tensor를 역변환하거나 reference output을 계산한다. 28장의 GoldenBatch를 export 전후에 실행하면 token에서 logit까지 의미 보존을 확인할 수 있다.

**policy를 버전된 순수 함수로 시험한다**

검증기와 정책 엔진을 나누어 사실과 결정을 분리한다. 검증기는 `digest_match=true`, `signature_identity=x`, `critical_cve=1`, `provenance_level=build` 같은 관측을 만든다. 정책은 관측, requested action, environment, 시각과 policy generation을 입력받아 `allow`, `deny`, `pending`과 reason을 반환한다. 검증기가 production 여부를 몰래 판단하면 같은 evidence를 staging과 production에 다르게 적용하기 어렵다.

정책은 순수 함수에 가깝게 만든다. 동일 입력과 generation이면 동일 결과가 나와야 하고 외부 network 조회는 미리 고정한 trust·revocation·advisory snapshot으로 materialize한다. 현재 시각이 필요하면 input으로 명시한다. “오늘은 됐는데 내일 안 됨”을 재현하려면 evaluation time과 만료 조건이 decision record에 있어야 한다.

정책 test table에는 최소 정상, missing evidence, unknown predicate, unauthorized builder, expired signature bundle, revoked material, critical vulnerability exception, nonproduction channel과 rollback을 둔다. unknown field를 무시하는 parser와 unknown predicate를 허용하는 정책을 피한다. schema가 앞으로 확장됐을 때 오래된 verifier가 이해하지 못한 보안 주장을 성공으로 해석할 수 있기 때문이다.

exception은 rule 삭제가 아니다. subject 범위, environment, 허용 action, owner, justification, compensating control과 expiry를 가진 별도 artifact다. wildcard subject나 무기한 exception을 거부한다. policy 결과에는 적용된 exception digest를 넣고 expiry 뒤 자동으로 deny가 되는 fixture를 실행한다. exception을 갱신하면 새 generation이며 과거 결정을 다시 쓰지 않는다.

rollback은 과거 alias로 돌아가는 단순 동작이 아니다. 과거 subject가 현재 revocation, vulnerability, license와 runtime compatibility policy를 통과해야 한다. 현재는 금지된 remote code나 취약 library를 포함한 last-known-good라면 새 deployment에 사용할 수 없다. rollback 후보가 없다는 사실을 사고 전에 drill로 발견해야 한다.

**mirror·cache·offline 실행의 신뢰 경계를 분리한다**

mirror는 availability와 locality를 제공하지만 원본 authenticity를 대체하지 않는다. upstream에서 받아 mirror에 넣을 때 digest와 provenance를 검증하고, mirror에서 소비할 때도 같은 immutable identity를 검증한다. mirror operator가 파일을 바꿀 수 있다는 threat model을 포함한다. TLS와 접근 제어는 전송 경로를 보호할 뿐 잘못 저장된 byte를 올바르게 만들지 않는다.

content-addressed cache는 blob key가 digest여야 하고 write 뒤 read-back hash를 확인한다. human-readable snapshot은 immutable blob으로 향하는 projection이다. symlink를 지원하지 않는 filesystem에서 copy를 쓰면 projection과 blob의 중복 byte가 생기므로 각각 검증한다. eviction은 identity record를 지우지 않고 local location만 제거한다. 다시 내려받은 같은 digest는 새 observation event를 갖는다.

offline bundle에는 model byte만 아니라 trust root, transparency proof, revocation snapshot, policy, schema와 verifier binary가 포함된다. 각 항목의 generation과 expiry를 기록한다. network가 없다는 이유로 freshness를 무기한 면제하지 않는다. mission 환경이 일정 기간 stale trust를 허용한다면 최대 age, 제한되는 action과 재연결 시 재평가 절차를 정책에 쓴다.

cache substitution fixture는 이름이 같은 다른 byte를 넣는 단순 시험과 digest key 아래 잘못된 byte를 넣는 강한 시험을 나눈다. 전자는 alias resolution 검증, 후자는 content-addressed store의 불변량을 시험한다. partial blob, zero-length blob, valid JSON이지만 다른 model config, valid safetensors이지만 wrong key set도 별도 fixture다. parser success가 identity success로 승격되지 않아야 한다.

복구 시 전체 cache를 무조건 지우면 서비스 복구가 느리고 forensic evidence가 사라진다. 오염 identity와 그것을 참조한 snapshot을 quarantine namespace로 원자 이동하고 읽기 금지한다. 남은 blob은 background 재검증하되 current request에 필요한 것은 synchronous gate를 통과한다. quarantine byte를 분석 도구가 production loader로 열지 않도록 capability를 분리한다.

**두 방향 계보 질의가 맞물리는지 시험한다**

역방향 질의는 production process에서 시작해 실제 loaded model, deployment manifest, promotion, evaluation, checkpoint, training run, dataset과 source로 간다. 정방향 질의는 source, library, signer나 builder에서 시작해 모든 run, artifact와 deployment 후손을 찾는다. 두 질의는 같은 typed edge를 반대 방향으로 걸어야 한다. 별도 CMDB와 실험 tracker를 수동 조인하면 누락이 정상 상태가 된다.

작은 그래프에서 `D→R→C→A→M→P`가 각각 dataset, run, checkpoint, adapter, merged model, production이고 `L→R`, `L→P`가 library consumption이라고 하자. `ancestors(P)`는 `{M,A,C,R,D,L}`을 포함해야 하고 `descendants(D)`는 `{R,C,A,M,P}`다. `P`의 ancestor에 `D`가 있는데 `D`의 descendant에 `P`가 없다면 edge index나 권한 필터가 일관되지 않다. nightly invariant로 양방향성을 검사한다.

접근 제한 node도 graph에서 사라지면 안 된다. 권한 없는 사용자는 raw URI나 민감 metadata 대신 opaque ArtifactID와 제한 이유를 보되, 영향받는 후손 수와 approval state는 확인할 수 있어야 한다. 그렇지 않으면 보안 때문에 provenance가 끊겨 안전 결정을 못 하는 역설이 생긴다. 상세 공개와 존재·관계 증명을 분리한다.

graph snapshot도 artifact다. 질의 결과에는 graph generation과 policy 시각을 넣는다. incident 도중 새 edge가 ingestion되면 첫 결과와 다음 결과가 다른 이유를 설명할 수 있다. closure 계산이 완료되기 전 affected 상태를 성급히 `complete`로 만들지 않고 `expanding`을 둔다. 늦게 도착한 offline run도 재평가 queue에 들어간다.

최종 drill은 임의 dataset row, compiler binary, signer identity와 tokenizer를 각각 seed로 선택한다. forward closure의 production replica와 backward ancestry의 교집합을 비교하고, 누락 edge를 owner별 queue로 보낸다. 네 seed가 각각 data, build, trust, semantic preprocessing 축을 덮는다. 이 drill을 통과해야 30장의 release dashboard가 실제 위험 반경을 보여 준다고 말할 수 있다.

**공급망 인수표를 증거 중심으로 판정한다**

인수자는 먼저 모든 artifact가 content identity를 가지며 alias가 generation 조건으로 갱신되는지 본다. bundle root가 config, tokenizer, tensor index, shard, code와 card를 빠짐없이 묶는지 확인한다. 저장소 commit, container, dataset과 native binary가 floating reference가 아닌 immutable material인지 검산한다.

다음으로 provenance statement의 subject binding, builder identity, material completeness와 policy authorization을 각각 판정한다. signature, transparency, revocation과 trust freshness를 별도 칸에 쓴다. SBOM은 build·runtime·kernel inventory 차집합과 연결하고 critical finding의 reachability·exception·expiry를 확인한다.

loader gate에서는 safetensors 구조, expected key·shape·dtype, remote code closure, pickle allowlist와 runtime mutation을 본다. checkpoint gate에서는 next-update dependency closure와 clean-process resume를 확인한다. data gate에서는 row lineage, count 보존식, dedup·mixture, consent·license와 deletion graph를 본다.

재현 gate에는 주장한 oracle 수준, exact environment, independent executor, first divergence와 실패 설명이 있어야 한다. byte 불일치가 허용됐다면 semantic·behavioral oracle이 왜 충분한지 적용 범위를 명시한다. hardware나 외부 서비스 때문에 실행하지 못한 것은 `NOT-RUN`이며 과거 upstream 결과를 local pass로 바꾸지 않는다.

운영 gate는 cache substitution, partial checkpoint, signer compromise, vulnerable library와 data withdrawal drill을 포함한다. detector가 기대 reason code를 내고 graph closure, quarantine, clean rebuild와 재승인이 이어져야 한다. 정상 control이 마지막에 통과해야 fault hook이나 stale state가 남지 않았음을 보인다.

최종 decision은 subject, action, environment, policy·trust·graph generation, evidence index, reviewer와 expiry를 가진다. 사람이 읽는 요약은 이 record에서 생성하되 원시 결과를 대체하지 않는다. 하나라도 unknown이면 승인자가 위험을 명시적으로 수락할 권한과 근거가 있는지 확인하고, 기본값으로 unknown을 success로 바꾸지 않는다.

이 인수표가 통과하면 28장에 넘기는 것은 model 경로가 아니다. 검증된 bundle root, resolved runtime inventory, canonical config, dataset generation, checkpoint state schema, trust·policy generation과 재현 oracle이다. 단일 GPU runner는 이 입력을 소비하고 새 checkpoint와 trace를 같은 graph에 돌려준다. 이 왕복이 성립해야 공급망이 학습 실행 바깥의 서류가 아니라 실행의 일부가 된다.

**provenance graph의 완전성을 표본이 아니라 불변량으로 감시한다**

graph가 커지면 사람이 모든 edge를 읽을 수 없으므로 type별 불변량을 실행한다. 모든 `Checkpoint`는 정확히 하나의 producer run을 가져야 하고, 모든 run은 source, resolved config, environment와 dataset generation을 가져야 한다. adapter는 base subject와 target mapping이 있어야 하며 merged model은 base와 모든 adapter parent를 가져야 한다. production deployment는 하나의 immutable model subject와 하나의 promotion decision을 가져야 한다.

cardinality만 맞아도 순환이 생길 수 있다. `derived-from`과 `produced-by/consumed`의 시간 방향 graph는 DAG여야 한다. adapter가 자기 merged output을 base로 가리키거나 checkpoint가 자신을 만든 run의 입력으로 들어가면 lineage가 모순이다. 단, `evaluated-by`나 `supersedes`처럼 별도 의미의 edge는 순환 가능성을 독립 schema로 다룬다. 모든 관계를 한 DAG에 강제로 넣지 않는다.

completeness score 하나로 승인하지 않는다. required edge가 하나 빠진 artifact와 optional 설명 edge가 여러 개 빠진 artifact의 위험이 다르다. type별 required predicate를 boolean gate로 검사하고 optional coverage는 별도 metric으로 본다. unknown 산출물 유형은 permissive default가 아니라 quarantine schema queue로 보낸다. 새 quantization format이나 processor가 들어왔는데 오래된 graph validator가 필수 parent를 모른 채 통과시키면 안 된다.

ingestion 지연도 측정한다. run 종료 뒤 provenance가 graph에 보일 때까지의 p50·p95·max, deployment 뒤 runtime actual identity가 연결될 때까지의 지연, revocation event가 affected replica를 차단할 때까지의 시간을 기록한다. graph가 논리적으로 완전해도 24시간 늦으면 즉시 incident 대응에는 쓸 수 없다. freshness SLO를 넘긴 node는 `unknown-current-state`로 표시한다.

orphan 탐지는 양쪽에서 한다. storage에는 있지만 graph에 없는 blob은 unauthorized upload나 ingestion failure일 수 있다. graph에는 있지만 어느 storage에서도 관측되지 않는 artifact는 retention, 삭제나 잘못된 location projection일 수 있다. expected retired artifact와 unexpected missing을 구분한다. orphan을 자동 삭제하기 전에 quarantine과 owner 확인을 거친다.

수치 예로 하루에 run 1,000개가 생기고 각 run이 평균 artifact 12개, edge 25개를 만든다고 하자. 0.1% edge 누락은 하루 25개이며 한 달이면 약 750개다. 99.9% completeness라는 aggregate는 좋아 보이지만 production ancestry 한 경로의 required dataset edge 누락이면 영향 질의가 실패한다. 따라서 random sample뿐 아니라 production에서 모든 ancestor required invariant를 전수 검사한다.

graph migration도 provenance를 가진다. schema v3에서 v4로 edge를 분리할 때 migration source, code revision, input snapshot, output snapshot과 rejected row를 기록한다. old graph를 덮어쓰지 않고 동일 질의의 old/new 결과를 비교한다. descendant count, root count와 cycle count가 예상 보존식을 만족해야 한다. migration 뒤 policy decision을 재평가해야 하는지 명시한다.

**build 재현에서 시간·경로·병렬성 숨은 입력을 제거한다**

같은 source와 compiler version으로 binary가 다를 때 source bug로 단정하지 않는다. archive timestamp, filesystem enumeration, absolute build path, locale, timezone, umask, hostname, user, CPU feature detection과 parallel link order가 byte에 들어갈 수 있다. builder manifest는 이 값을 고정하거나 결과에서 제거한 방법을 기록한다. 결과가 같았다는 한 번의 우연보다 숨은 입력을 의도적으로 바꾼 교차 build가 강한 시험이다.

builder A는 `/work/a`, UTC, locale C, parallelism 1을 쓰고 builder B는 `/different/path`, 다른 hostname, parallelism 8을 쓴다. 재현 가능한 build라 주장하려면 output byte가 같거나 차이가 허용된 metadata에만 있고 executable semantics가 같다는 정한 oracle을 통과해야 한다. path가 debug section에만 남는다면 실제 digest는 다르므로 byte 재현 실패를 숨기지 않고 normalized comparison을 보조 결과로 둔다.

container tag를 builder identity로 쓰지 않는다. image digest와 각 layer, build frontend·definition, base image, package repository snapshot과 secret mount 정책을 기록한다. build secret은 output이나 log에 들어가면 안 되지만 어떤 secret class가 사용됐는지는 provenance policy가 요구할 수 있다. network access는 domain 목록이 아니라 내려받은 material digest로 닫는다.

병렬 build nondeterminism은 file order뿐 아니라 generated code와 autotuning에 들어갈 수 있다. 같은 builder에서 parallelism 1과 8, clean cache와 warm cache를 교차 실행한다. 네 결과의 pairwise digest matrix를 만들고 최초 다른 file·section을 찾는다. cache가 결과를 바꾸면 cache key의 material closure가 불완전하다. cache를 성능 최적화로만 보고 provenance에서 제외하지 않는다.

source archive도 Git commit 문자열만 믿지 않는다. commit object와 submodule, LFS object, generated vendored source를 모두 material로 resolve한다. dirty working tree, untracked generated header와 patch 적용 순서를 기록한다. `git describe`가 같아도 worktree가 다를 수 있다. build 시작 전 tree digest와 diff artifact를 만들고 dirty build를 허용할지 policy로 결정한다.

교차 build 결과는 28장의 golden run으로 이어진다. binary byte가 다르지만 허용하려면 동일 GoldenBatch의 forward, gradient와 optimizer delta, profiler의 selected kernel identity를 비교한다. 수치가 같더라도 SBOM actual component는 별도 identity로 남는다. behavioral equality를 byte identity처럼 표현하지 않는다.

**source·test 좌표를 upgrade diff의 기준으로 유지한다**

소스 원장의 좌표는 설명 장식이 아니라 semantic anchor다. Transformers 고정 revision의 `src/transformers/modeling_utils.py:3278` `save_pretrained`, `:3859` `from_pretrained`, `src/transformers/utils/hub.py:166`과 `:453`의 snapshot 경계처럼 path, symbol, revision을 함께 쓴다. line number는 편의를 위한 것이며 upgrade에서는 symbol과 주변 invariant로 다시 찾는다. line만 그대로인데 함수 의미가 바뀔 수도 있다.

upgrade bot은 old/new source에서 signature, default, branch와 호출 closure를 diff한다. loader의 `trust_remote_code`, safe serialization, shard size, dtype와 cache option처럼 공급망 상태를 바꾸는 값을 우선 검토한다. upstream changelog만 읽고 끝내지 않고 관련 test의 positive·negative fixture가 어떻게 바뀌었는지 본다. 삭제된 test는 기능 제거인지 coverage 회귀인지 판단한다.

local contract test는 upstream 내부 구현을 그대로 복제하지 않는다. 우리 불변량인 immutable revision resolution, bundle completeness, remote-code closure, wrong-key 거부, partial checkpoint 거부와 exact resume를 시험한다. upstream refactor가 있어도 이 의미가 유지되면 local test가 통과한다. 의미가 바뀌면 소스 원장, RunSpec과 baseline을 함께 review한다.

좌표 검토의 결과는 `source-confirmed`, `upstream-test-confirmed`, `local-fixture-executed`, `hardware-executed`를 구분한다. source에서 분기를 봤다는 사실로 실제 artifact 검증을 실행했다고 쓰지 않는다. 반대로 local fixture가 통과해도 검토하지 않은 option과 format으로 범위를 넓히지 않는다.

upgrade 뒤에는 old/new 두 verifier를 동일 정상·악성 fixture bundle에 실행한다. 새 verifier가 old invalid를 valid로 바꾸면 정책 변경 근거가 필요하다. old valid를 invalid로 바꾸면 기존 registry 전체의 재평가와 migration 계획이 필요하다. 결과 차이를 단순 test snapshot 갱신으로 없애지 않는다.

이 source 좌표와 실행 evidence의 결합이 공급망 설명을 현재 revision에 묶는다. 독자는 어느 함수가 상태를 바꾸고 어느 fixture가 그 불변량을 검증했는지 따라갈 수 있으며, 다음 upgrade에서 무엇을 다시 읽고 실행해야 하는지도 알 수 있다.

검토 결과에는 확인 시각, 검토자, 사용한 local source tree digest와 실행하지 않은 경로도 함께 남긴다. 좌표가 있다는 사실만으로 모든 option을 확인했다고 과장하지 않으며, 새 branch가 추가되면 영향받는 fixture와 승인 상태를 자동으로 재평가한다.

재평가 결과와 이전 판정의 차이도 별도 불변 기록으로 보존한다.

운영 handoff에는 정상 release 절차뿐 아니라 mirror compromise, signer compromise, vulnerable native library, dataset 삭제 요구와 remote-code 취약점의 runbook을 포함한다. 각 runbook은 탐지 signal, 즉시 차단할 subject 범위, provenance graph query, cache purge, clean rebuild 시작점과 재승인 조건을 가진다. 담당자 연락처만 적고 기술적 판정 기준을 생략하면 야간 사고에서 과잉 폐기와 누락 폐기가 동시에 생긴다.

재현성 보고서에는 실패도 자산으로 남긴다. 서로 다른 builder에서 bitwise mismatch가 난 binary, cache substitution fixture, 거부된 attestation과 schema mismatch를 보존하면 verifier upgrade 뒤 음성 시험으로 다시 사용할 수 있다. 단, 악성 또는 민감한 fixture는 격리 저장소와 최소 권한으로 관리한다. 실패 표본을 정상 registry에 두거나 production loader로 직접 열지 않는다.

마지막 판정은 “서명이 있다”가 아니라 다음 문장으로 표현한다. 이 subject bundle은 명시한 material과 builder에서 생성됐고, 현재 policy와 trust root 아래 권한이 확인됐으며, 형식·schema·실행 gate와 선언한 재현 등급을 통과했고, 현재 revocation graph에서 유효하다. 이 문장의 각 구절이 독립 evidence로 연결될 때만 promotion한다.

**Python package lock에서 실제 CUDA binary까지 닫는다**

Python dependency 이름과 version만으로 실행 환경을 재현할 수 없다. wheel tag, platform, Python ABI, CUDA runtime linkage, bundled shared object와 compiled GPU architecture가 실제 kernel 경로를 결정한다. lock에는 index URL과 resolved 산출물 digest를 넣고, 설치 뒤 import된 module path와 native library digest를 inventory한다. 같은 version 문자열의 wheel이 mirror에서 바뀌거나 CPU build가 선택되는 상황을 막는다.

dynamic loader 관점에서는 driver, CUDA runtime, cuBLAS·cuDNN·NCCL, compiler runtime과 framework extension의 의존성을 확인한다. container 안 package만 고정하고 host driver·device firmware와 runtime mount를 무시하지 않는다. 실행 시작 시 loaded library와 device capability를 기록하되 mutable path만 저장하지 않고 content·build identity를 남긴다.

custom CUDA·Triton extension은 소스 리비전, compiler flags, target architecture, toolkit와 generated binary를 연결한다. build cache key에 숨은 environment가 빠지면 다른 source가 같은 cache를 재사용할 수 있다. clean build와 cached build의 binary·semantic fixture를 비교하고 지원하지 않은 architecture의 JIT fallback을 명시한다.

**데이터 snapshot은 row count가 아니라 변환 closure다**

dataset manifest에는 source objects, retrieval time, license·policy state, schema와 partition을 넣는다. 이후 decode, normalization, language·quality filtering, dedup, split, tokenizer/template와 packing transform을 순서가 있는 edge로 기록한다. 최종 shard hash만 있으면 같은 결과를 보존할 수는 있어도 어떤 source와 정책으로 만들었는지 설명하거나 삭제 영향을 계산하기 어렵다.

각 transform은 code revision, config, input·output count, rejection disposition과 deterministic 여부를 가진다. nondeterministic sampling·model filter는 seed뿐 아니라 모델 산출물와 batching·runtime 조건을 남긴다. distributed map에서 worker 수와 completion order가 output ordering이나 shard boundary를 바꾸는지 검사한다. bitwise ordering과 sample-set equality를 다른 재현 등급으로 구분한다.

snapshot verifier는 row IDs uniqueness, family split disjointness, declared counts·bytes와 shard digest를 확인한다. packed sample에서도 parent row IDs와 token span을 보존한다. training ledger가 manifest에 없는 sample을 소비하거나 허용된 cutoff 밖 source를 읽으면 run을 차단한다.

**remote code와 model repository를 실행 closure로 취급한다**

`trust_remote_code`는 편의 option이 아니라 외부 Python을 import하는 권한 변경이다. model revision만 고정하고 repository가 참조하는 다른 module, dynamic import, package dependency와 download URL을 놓치지 않는다. 실행 전 source tree를 materialize해 digest·license·static review와 sandbox test를 거친다. production에서 mutable branch를 다시 resolve하지 않는다.

custom modeling code는 weight schema, config parser, forward, generation과 save/load를 바꿀 수 있다. 같은 safetensors를 표준 class와 remote class가 다르게 해석할 수 있으므로 class identity와 code revision이 model bundle의 일부다. tokenizer·processor remote code도 bytes→IDs·tensor 의미를 바꾼다.

격리 fixture는 network·filesystem·environment 접근, unsafe deserialization과 import-time side effect를 검사한다. 허용된 capability를 최소화하고 remote code 없이 export한 대안이 있는지 본다. review 통과를 영구 보증으로 쓰지 않고 upstream diff와 dependency 취약점에서 재평가한다.

**checkpoint format의 안전과 의미를 동시에 검증한다**

safetensors 같은 data-only format은 arbitrary object deserialization 위험을 줄이지만 tensor가 올바른 logical parameter에 매핑됐다는 뜻은 아니다. manifest는 key, shape, dtype, shard range, content digest와 logical owner를 가진다. duplicate key, overlapping range, missing tensor, oversized shape와 metadata mismatch를 load 전 검증한다.

pickle 기반 state가 필요한 경우 신뢰 경계와 isolated loader를 명시한다. 출처를 모르는 checkpoint를 production credential이 있는 process에서 열지 않는다. format converter도 공격 표면이므로 input bounds, output manifest와 round-trip fixture를 가진다. 변환 성공과 model semantic parity를 분리한다.

optimizer·scheduler·scaler·RNG·data cursor는 weight와 다른 schema를 가진다. inference release에는 불필요한 training state를 포함하지 않지만 resume bundle은 state completeness를 증명해야 한다. partial state를 조용히 0으로 초기화하면 연속 run 주장을 철회하고 새 generation으로 기록한다.

**서명은 digest와 권한·의도를 함께 묶는다**

signature가 유효하다는 것은 해당 key가 bytes digest에 서명했다는 뜻이다. signer가 그 산출물 유형과 environment로 promotion할 권한이 있었는지, key가 당시 유효했고 revocation되지 않았는지 별도로 확인한다. artifact, provenance statement와 policy decision의 subject digest가 동일한지 검증한다.

key rotation은 old release를 무효화하지 않으면서 새 서명을 제한해야 한다. compromise가 의심되면 affected time window·artifact descendants를 query하고 cache·replica의 실제 loaded subject를 확인한다. 새 key로 같은 bytes를 다시 서명하는 것만으로 compromised build provenance가 정화되지 않는다.

offline verification을 지원하려면 trust root, revocation snapshot과 필요한 transparency evidence를 bundle한다. 검증 시각과 policy revision을 남겨 나중에 당시 결정을 재구성한다. “서명됨” badge 하나가 build correctness, license, vulnerability와 model quality를 대체하지 않는다.

**재현성은 bitwise·numerical·behavioral 등급을 나눈다**

bitwise reproducibility는 같은 입력에서 동일 bytes를 요구한다. timestamp, archive ordering, path, compiler nondeterminism과 parallel scheduling을 통제해야 한다. 모든 GPU training run에서 현실적인 요구는 아닐 수 있지만 manifest·data shard·package처럼 가능한 artifact에는 강한 가치가 있다.

numerical reproducibility는 tensor별 tolerance, first divergence와 누적 drift를 정의한다. hardware·kernel 변경에서 bitwise가 달라도 수식 의미를 보존할 수 있다. 단일 final loss `allclose`로 충분하지 않고 canonical forward, gradient, optimizer next step과 checkpoint round-trip을 경계별로 본다.

behavioral reproducibility는 frozen evaluation과 serving sentinel에서 의사결정 수준의 동등성을 본다. stochastic generation은 repeat distribution과 seed protocol을 사용한다. 낮은 등급이 높은 등급을 자동으로 증명하지 않는다. behavioral score가 같아도 supply-chain substitution을 허용할 수 없고, bitwise artifact가 같아도 잘못된 dataset 정책을 정당화하지 않는다.

**cache와 mirror는 성능 계층이 아니라 신뢰 계층이다**

package·model·dataset cache는 upstream 부하와 startup 시간을 줄이지만 stale·poisoned artifact를 오래 유지할 수 있다. cache key에는 immutable identity와 expected digest를 넣고 read 시 검증한다. mutable URL이나 filename만 key로 쓰지 않는다. partial download는 atomic completion marker 없이 visible하게 만들지 않는다.

mirror sync는 source authentication, freshness, deletion·revocation과 quarantine을 처리한다. upstream에서 제거된 취약 artifact가 내부 mirror에 남을 수 있으므로 denylist와 descendant query를 전파한다. air-gapped environment는 외부 연결이 없다는 이유로 안전한 것이 아니라 import bundle과 trust-root update 절차가 핵심이다.

cache purge rehearsal에서는 control plane alias뿐 아니라 node local disk, object store, container layer와 running process의 loaded mapping을 확인한다. purge 뒤 같은 digest를 다시 가져오는 경로가 승인된 mirror로 제한되는지 본다. 성능 cache와 evidence archive의 retention 정책을 혼동하지 않는다.

**공급망 사고의 영향 범위를 양방향으로 계산한다**

취약한 library나 오염 dataset이 발견되면 material에서 runs, checkpoints, transforms, evaluations와 deployments로 descendants를 탐색한다. production incident에서 출발하면 loaded bundle에서 source·builder·data와 signer로 ancestors를 올라간다. 두 query가 같은 release component에서 만나야 graph가 닫힌다.

영향 판정은 dependency가 manifest에 존재했다는 사실과 실제 실행되었다는 사실을 구분한다. 취약 code path가 호출되지 않았더라도 정책상 교체가 필요할 수 있다. native library는 runtime loaded inventory와 kernel trace가 더 강한 evidence다. 불확실하면 안전한 범위로 quarantine하되 unknown을 confirmed exploit로 과장하지 않는다.

복구는 clean material resolution, isolated rebuild, semantic fixtures, 재평가·재서명과 replica 교체를 순서대로 수행한다. 기존 cache와 mixed revision replica를 제거하고 rollback target도 같은 취약 lineage인지 확인한다. incident 종료 뒤 새 negative fixture와 revocation propagation test를 남긴다.

**27장의 독립 공급망 인수**

검토자는 release 하나를 선택해 source, lock, native binary, dataset snapshot, run, checkpoint, transform, evaluation, signature와 deployment를 모두 content identity로 잇는다. mutable alias만 있는 edge, digest가 있지만 schema·owner가 없는 node, 서명됐지만 권한이 없는 decision을 실패로 표시한다.

그다음 wrong digest, expired signer, partial checkpoint, poisoned cache, undeclared remote code와 deleted dataset family를 하나씩 주입한다. verifier가 load·promotion 전에 차단하고 affected descendants를 정확히 보고하는지 본다. 정상 fixture도 계속 통과해야 한다. 모든 것을 거부하는 verifier는 안전한 것이 아니라 쓸 수 없는 것이다.

마지막으로 clean-room builder와 기존 builder가 선언한 재현 등급을 비교한다. 차이가 나면 timestamp부터 compiler·parallel order·GPU kernel까지 first divergence를 찾는다. 실행하지 않은 hardware·format·policy cell은 공개한다. 이 증거가 있을 때 공급망은 파일 보관 규칙이 아니라 학습 결과를 설명하고 철회하며 안전하게 다시 만들 수 있는 실행 체계가 된다.

**container image는 시작점이지 실행 환경 전체가 아니다**

image digest는 root filesystem과 metadata를 고정하지만 host driver, mounted library·volume, device plugin, environment variable, secret과 runtime flag는 실행 때 추가된다. 따라서 RunEnvironment는 image digest와 runtime spec, node identity, GPU·driver, mounted artifact와 effective environment를 함께 가진다. secret 값은 저장하지 않고 credential identity·scope·revision만 기록한다.

privileged mode, host network, writable mount와 device access는 capability 변경이다. 편의를 위해 넓게 열지 않고 trainer·data reader·checkpoint writer가 필요한 권한을 분리한다. container entrypoint가 package를 업데이트하거나 remote code를 내려받지 않게 offline fixture를 실행한다. read-only root에서도 필요한 cache·temporary path가 명시적으로 작동하는지 본다.

orchestrator mutation도 추적한다. admission webhook, init container, sidecar와 injected environment가 submitted spec을 바꿀 수 있다. requested manifest와 admitted·running spec을 diff하고 실제 pod의 image·mount·device identity를 evidence에 넣는다. YAML 원본만으로 실행을 증명하지 않는다.

**configuration은 parse 결과와 effective state를 모두 보존한다**

CLI, YAML, environment, library default와 model config가 합쳐지면 같은 option이 여러 위치에서 override된다. parser 입력을 저장하는 것만으로 부족하다. type conversion, deprecated alias, auto-detection과 runtime clamp 뒤 canonical effective config를 만든다. source, precedence와 최종 consumer function을 option별로 연결한다.

unknown option을 조용히 무시하지 않고 fail하거나 명시적 disposition을 남긴다. boolean 문자열, null·missing, list ordering과 단위 변환을 property test로 검증한다. GPU 수에 따라 자동 계산되는 batch, scheduler total steps, shard와 dtype도 resolved value로 저장한다. requested와 effective가 다르면 이유와 승인 rule이 필요하다.

resume에서는 checkpoint config와 새 invocation의 diff를 분류한다. 관측 cadence처럼 의미를 보존하는 변경, LR·data·world size처럼 학습 의미를 바꾸는 변경, state mapping을 깨뜨리는 금지 변경을 구분한다. override가 허용돼도 새 run generation과 DecisionEvent를 만든다.

**데이터·모델 라이선스 판단을 artifact 계보와 연결한다**

라이선스 이름 한 칸은 사용 가능성을 자동으로 판정하지 않는다. source별 license text revision, attribution, 용도·재배포·파생물 조건과 관할 정책을 기록하고 eligibility decision을 versioned rule로 남긴다. 불명확한 source를 permissive로 추정하지 않고 quarantine한다. 법적 판단과 기술적 provenance 사실을 구분한다.

mixture dataset과 fine-tuned model은 여러 parent 조건을 가진다. 어느 조건이 final weight·adapter·evaluation example와 공개 문서에 적용되는지 policy owner가 판정한다. model card와 release package의 attribution이 dataset manifest와 일치하는지 검사한다. upstream license 변경은 이미 생성된 artifact의 당시 evidence와 현재 배포 정책을 각각 평가한다.

삭제·철회·접근 제한이 생기면 descendants를 찾고 data 제거, 재학습·언러닝, serving 제한과 문서 변경을 별도 action으로 기록한다. 기술 graph가 법적 결론을 대신하지 않지만 영향 범위를 누락 없이 제시해야 책임 있는 판단이 가능하다.

**evaluation artifact도 공급망 검증 대상이다**

benchmark score는 model만의 속성이 아니다. dataset·renderer, tokenizer, runtime, generation config, parser, judge와 aggregation code의 조합이다. evaluation bundle에도 소스 리비전, dependency lock, input manifest, raw response, disposition과 report digest를 둔다. mutable API judge와 live web environment는 timestamp·provider revision과 replay 한계를 명시한다.

judge model이나 parser upgrade는 같은 frozen responses를 old·new로 dual-score한다. benchmark data 수정은 item identity와 contamination disposition을 바꾼다. 과거 report를 덮어쓰지 않고 새 EvalID와 migration evidence를 만든다. report cell에서 raw response까지 역추적되지 않으면 release decision의 공급망도 끊긴다.

평가 cache는 model·protocol·item·runtime identity 전체를 key로 사용한다. stale cache substitution과 partial result를 negative fixture로 둔다. score가 같아도 wrong subject response를 재사용하면 실패다. 24장의 측정 계약이 27장의 artifact 무결성과 만나는 지점이다.

**철회와 재승격을 실제 fleet에서 rehearsal한다**

registry에서 artifact를 revoked로 표시하는 것만으로 실행 중 replica가 멈추지 않는다. control plane, scheduler cache, node local files, model server memory와 adapter slot에서 affected digest를 찾아 격리한다. 새로운 request 배정을 막고 in-flight 처리 정책을 적용한 뒤 approved parent로 rollback한다. 실제 loaded identity를 sentinel로 확인한다.

clean rebuild는 compromised material·builder·cache를 재사용하지 않는다. 새 provenance와 semantic evaluation을 통과한 artifact만 재승격한다. 같은 model name·version을 재사용하지 않고 generation을 올린다. downstream evaluation·model card와 deployment decision도 새 subject에 연결한다.

rehearsal은 detection, impact query, quarantine, rollback, rebuild, re-evaluation과 promotion 시간을 나누어 측정한다. 누락 descendant, stale replica와 재오염 cache를 failure로 기록한다. 이 훈련이 정기적으로 통과할 때 revocation은 문서상의 약속이 아니라 실제 서비스에서 작동하는 복구 메커니즘이 된다.

**최종 evidence bundle을 빈 환경에서 재검증한다**

인수자는 production registry를 신뢰 목록으로 사용하지 않는 빈 verifier 환경을 준비한다. 승인된 trust root와 policy snapshot, release bundle만 가져와 source·builder provenance, 산출물 digest, schema, signature, revocation과 dependency closure를 순서대로 검증한다. network가 없어도 필요한 evidence가 완결되는지 확인하고, 외부 조회가 필수라면 endpoint·freshness와 실패 정책을 명시한다.

첫 번째 negative fixture는 weight shard 한 byte를 바꾸고 manifest는 그대로 둔다. 두 번째는 올바른 artifact에 권한 없는 key로 서명한다. 세 번째는 유효한 checkpoint와 다른 tokenizer·remote code revision을 묶는다. 네 번째는 revoked dataset family의 descendant를 release에 남긴다. verifier는 각각 content, authorization, bundle compatibility와 policy lineage의 다른 단계에서 거부해야 한다. 모두 같은 “invalid” 메시지로 끝내지 않고 operator가 안전한 다음 행동을 선택할 만큼 구체적인 bounded error를 낸다.

정상 bundle은 검증 후 canonical fixture를 실행한다. tokenizer bytes→IDs, model first logits, adapter merge 또는 quantization parity, evaluation raw response→score와 serving sentinel을 가능한 범위에서 재생한다. 서명과 hash가 맞아도 semantic mapping이 틀릴 수 있기 때문이다. hardware가 없어 실행하지 못한 kernel·collective cell은 source-confirmed나 pending으로 남기고 실행됨으로 표시하지 않는다.

마지막에는 production catalog의 alias가 exact subject를 가리키는지, running replica가 그 digest를 실제로 load했는지 확인한다. rollback target도 동일한 verification을 통과하고 cache·tokenizer·template namespace를 포함해야 한다. bundle 생성 도구와 verifier가 같은 결함을 공유할 수 있으므로 독립 구현이나 수동 표본 재계산으로 critical digest·signature·edge를 삼각 측량한다.

이 검증 결과에는 pass뿐 아니라 거부된 fixture, 미검증 조건, policy exception, owner와 expiry를 포함한다. 다음 dependency·CUDA·framework·dataset·signer 변경에서 어떤 fixture를 다시 실행할지 impact rule을 둔다. 누구든 같은 evidence에서 artifact의 출처, 실행 의미, 허용된 용도, 현재 유효성과 안전한 철회 경로를 재구성할 수 있을 때 27장의 공급망은 닫힌다.

최종 reviewer는 bundle의 무작위 node 세 개를 선택해 선언된 parent와 실제 bytes를 양방향으로 검산한다. data shard에서는 source·transform·split을, optimizer checkpoint에서는 logical parameter·state generation을, serving artifact에서는 merge·quantization·runtime parent를 확인한다. edge가 content digest만 갖고 변환 함수·option·검증 상태를 잃었다면 영향 분석이 불가능하므로 인수하지 않는다. 이어 signer와 builder 하나를 철회한 모의 policy snapshot을 적용해 descendants가 빠짐없이 차단되는지 확인한다. 정상 sibling artifact까지 과잉 차단한다면 graph 또는 policy granularity를 고친다.

정확한 거부 범위와 clean rebuild 시작점을 제시할 수 있어야 사고 중 속도와 안전을 함께 확보한다. 이 결과를 다음 교대자가 동일한 입력과 정책에서 재현하고, 미검증 cell을 실행됨으로 오인하지 않는지 마지막으로 확인한다. 재현 명령, 예상 failure stage, 필요한 trust root와 복구 acceptance까지 적어야 공급망 지식이 담당자의 기억을 넘어 지속된다.

## 27.13 산출물 identity와 SLSA statement의 의미를 고정한다

같은 이름, 같은 digest, 같은 실행 의미는 서로 다른 주장이다. 세 층을 분리해 provenance가 무엇을 보증하고 무엇을 보증하지 않는지 한정한다.

artifact name은 사람이 찾는 label이고 digest는 byte identity이며 meaning은 schema와 parent relation이다. `model-final`, semantic version과 Hub repository는 mutable alias가 될 수 있다. digest가 같으면 bytes는 같지만 해당 bytes가 tokenizer인지 weight인지, 어느 architecture와 연결되는지는 별 manifest가 말한다.

BundleID는 canonical manifest의 digest로 만들고 exact file set, 각 content digest, media type, schema와 required relation을 포함한다. directory mtime, tar order나 storage URI를 identity로 쓰지 않는다. manifest 자체의 canonical serialization과 version을 고정한다. unknown required field는 거부한다.

content-addressing은 authenticity를 자동 제공하지 않는다. 공격자도 악성 bytes의 올바른 hash를 계산할 수 있다. trusted provenance와 signature가 누가 어떤 입력으로 만들었는지 연결하고 policy가 그 builder·source가 production channel에 권한이 있는지 판정한다. integrity, authenticity와 authorization을 분리한다.

meaning compatibility는 config, tokenizer/template, state schema와 runtime capability를 검사한다. weight shard hash가 맞아도 다른 index, hidden size나 special token과 묶이면 잘못된 bundle이다. tied weight, adapter base와 quantization scale relation을 invariant로 둔다.

alias promotion은 verified BundleID를 가리키는 작은 transactional record다. reader는 alias를 한 번 resolve한 뒤 exact digest를 사용하고 중간에 재해석하지 않는다. rollback도 old alias string이 아니라 이전 승인 BundleID로 이동한다. running replica는 실제 loaded digest를 보고한다.

### source provenance를 commit과 dirty workspace까지 닫는다

Git commit은 tracked file snapshot을 식별하지만 submodule, LFS blob, generated file, untracked patch와 build environment를 모두 설명하지 않는다. RunManifest에는 repository URL, commit, tree, submodule commit, LFS object digest, applied patch series와 dirty status를 넣는다. dirty run은 금지하거나 diff artifact를 material로 승격한다.

branch·tag는 요청 parameter로 남길 수 있지만 resolved commit이 실행 identity다. shallow clone과 force-pushed ref에서도 당시 resolution을 증명하려면 signed mapping 또는 fetch event를 보존한다. source archive를 받을 때는 archive digest와 extracted file set을 기록한다. Git metadata가 없다고 임의 commit을 추정하지 않는다.

generated source에는 generator binary, template, input schema, flags와 output digest가 필요하다. protobuf, CUDA generated kernel, tokenizer code와 compiled graph가 여기에 속한다. generated output만 commit돼 있어도 재생성 path를 별 provenance edge로 둔다. generator 변경이 output을 안 바꿔도 invocation identity는 다르다.

build script가 network에서 dependency나 code를 가져오면 hidden material이다. hermetic build는 network를 차단하고 선언된 material만 제공하거나 모든 fetch를 digest resolution event로 기록한다. mutable package index, current date와 host path가 output에 들어가는지 reproducible-build fixture로 찾는다.

source audit는 entrypoint에서 imported local module, plugin, remote code와 native extension source까지 closure를 만든다. Python module name만으로 distribution을 식별하지 않는다. editable install과 `PYTHONPATH` shadowing을 runtime resolved-module inventory로 검출한다.

### dataset provenance를 row·license·transformation edge로 만든다

raw dataset identity는 URL과 row count가 아니다. fetched object digest, acquisition time, source authority, license·consent snapshot, access condition와 retrieval tool을 가진다. source가 mutable이면 당시 bytes를 approved storage에 보존하거나 재취득 불가를 명시한다. sensitive source는 digest와 secure mapping을 사용한다.

transformation edge는 input shard/row set, code·config digest, output set, included·dropped count와 reason distribution을 가진다. decode, language ID, PII, quality, dedup, contamination, split, mixture와 packing을 분리한다. 한 huge script로 처리해도 logical stages와 evidence를 남긴다.

row identity는 source record와 byte/content digest를 결합하고 normalization 전후를 구분한다. exact duplicate group, semantic cluster와 document ancestor를 보존한다. split은 row hash뿐 아니라 group relation을 따라야 한다. packing output은 component row IDs, span, separator/tokenizer와 truncation을 가진다.

license는 artifact byte의 integrity와 별 authorization state다. license text digest, 확인 시각, jurisdiction·use restriction과 reviewer decision을 둔다. model card tag 하나를 최종 법적 결론으로 쓰지 않는다. condition이 바뀌면 bytes는 같아도 promotion policy가 달라질 수 있다.

deletion request는 raw row에서 transformed shard, mixture, packed sample, checkpoint, adapter·quantized export와 evaluation으로 descendant query를 실행한다. 완전 제거 가능성을 과장하지 않고 inaccessible backup, aggregate와 already-distributed artifact 범위를 기록한다. tombstone과 RevocationID를 유지한다.

### model card와 dataset card를 기계 manifest의 해설로 둔다

card는 intended use, data scope, limitations, evaluation, license와 risk를 사람이 읽게 설명한다. 그러나 mutable Markdown만으로 exact bytes와 dependency를 고정할 수 없다. card revision과 digest를 BundleID에 연결하고 machine manifest에는 source·weight·tokenizer·runtime identity를 넣는다.

dataset card는 collection, demographics/domain, annotation, filtering, PII, license와 known gaps를 설명한다. 통계는 어느 DatasetRevision에서 계산됐는지 갖는다. 최신 card가 과거 snapshot의 조건을 자동 덮어쓰지 않는다. correction은 old card를 수정하기보다 successor와 withdrawal annotation을 연결한다.

model card는 base, fine-tuning method, adapter/merge, quantization, tokenizer/template, context, evaluation protocol와 runtime requirement를 적는다. reported benchmark에 EvalID와 reproducibility grade를 붙인다. source command·raw ledger가 없으면 reproduced라고 쓰지 않는다.

card 생성 pipeline도 source commit, template와 input manifest를 가진다. 수동 표에 잘못된 model digest가 들어가는 fixture를 provenance validator가 잡아야 한다. card의 license·architecture field와 manifest가 충돌하면 promotion을 막는다. narrative가 machine truth보다 우선하지 않는다.

private data·security detail은 공개 card에 원문을 쓰지 않고 secure evidence 위치와 reviewer 절차를 적는다. transparency와 privacy를 함께 설계한다. 미검증·unsupported range를 숨기지 않는다. card는 승인 광고가 아니라 evidence index다.

**safetensors file safety와 bundle completeness를 분리한다**

safetensors commit `6eb4dc9a28ebce297606e0f4836bbf28839cacef`의 format은 header length, JSON metadata, dtype·shape·offset과 contiguous data bytes를 정의한다. `read_metadata`와 `Metadata::new`는 buffer range, ordering, overlap·hole와 required byte length를 확인한다. 확장자만 보고 safe loader를 선택하지 않는다.

한 file이 구조적으로 valid해도 model bundle이 complete하다는 뜻은 아니다. shard index의 weight map, actual shard exact set, file 내부 tensor keys와 architecture expected keys를 대조한다. duplicate, missing, unexpected와 tied omission을 구분한다. extra file도 supply-chain attack surface이므로 allowlist한다.

resource exhaustion을 고려한다. header size, tensor count, shape product overflow, total mapped bytes와 device allocation을 제한한다. parse는 credential·network가 없는 quarantine process에서 수행하고 production load 전에 digest·signature를 검증한다. valid giant shape가 OOM을 유발할 수 있다.

metadata에 arbitrary string이 있어도 실행되지 않지만 policy와 secret scan 대상일 수 있다. tensor name 길이, Unicode와 collision을 검증한다. framework converter가 name을 rename하거나 tied mapping을 바꿀 수 있어 semantic manifest를 별로 둔다.

negative fixture는 header length, overlapping offset, dtype-size mismatch, shard substitution, index duplicate, config shape mismatch와 tokenizer swap을 각각 넣는다. hash gate, format gate, bundle gate와 semantic load gate가 다른 error class로 막아야 한다. golden first logits까지 실행한다.

**pickle·weights_only·remote code의 능력 경계를 좁힌다**

pickle은 object graph 복원 중 callable을 실행할 수 있다. PyTorch commit `3691693263d2b66a68867e39b7449876844e06cf`의 `torch.load(weights_only=...)`는 허용된 tensor·primitive와 safe globals로 능력을 제한하지만 safetensors와 같은 format은 아니다. framework revision별 default와 allowed types를 확인한다.

`add_safe_globals`는 allowlist를 확장하므로 module·qualified name뿐 아니라 source distribution digest와 필요 이유를 policy에 둔다. error message를 읽어 job이 자동 allowlist하지 못하게 한다. conversion process scope가 끝나면 allowlist를 폐기한다. import side effect도 신뢰 경계다.

불가피한 optimizer pickle은 network, cloud credential와 production mount가 없는 disposable sandbox에서 load한다. resource/time limit와 syscall policy를 둔다. tensor-only canonical output을 새 manifest·provenance와 함께 만들고 원 input을 production loader가 직접 읽지 않게 한다.

`trust_remote_code`는 repository Python을 import·execute하는 권한이다. immutable commit, exact file set, import closure, dependency resolution과 side effect를 audit한다. dynamic download, subprocess, file/network access를 sandbox에서 관측한다. 승인 custom code digest를 model bundle에 묶는다.

remote code가 native extension을 JIT하면 source, compiler, flags와 output `.so`가 runtime SBOM에 들어간다. cache hit도 trusted parent provenance를 확인한다. native class와 remote class 중 실제 resolved class를 RunManifest에 남긴다. config 문자열만으로 실행 path를 추정하지 않는다.

**SBOM을 build·runtime·JIT 세 장부로 운영한다**

build SBOM은 container layer, OS package, Python distribution, wheel과 build tool을 기록한다. runtime SBOM은 host-mounted driver library, device plugin, preload, dynamically loaded `.so`와 actual process mapping을 더한다. JIT supplement는 Triton/CUDA extension, generated source와 executable cache를 기록한다.

component에는 package URL/CPE 등 식별자, version, file digest, license와 dependency relation이 있다. Python import module과 distribution name을 연결하고 vendored library의 parent를 보존한다. package version만 같고 wheel hash가 다르면 별 artifact다. index URL과 wheel tag도 material이다.

declared SBOM과 loaded-object inventory를 비교해 runtime-only, declared-but-unused와 unknown을 낸다. unused component도 attack surface나 license에 영향을 줄 수 있지만 reachability를 별 분석한다. “import하지 않았다”로 native transitive dependency를 제외하지 않는다.

취약점 scan은 database revision, timestamp, severity와 match evidence를 가진다. CVE match가 actual reachable risk인지 분석하되 false positive 가능성을 이유로 자동 무시하지 않는다. exception에는 owner, compensating control와 expiry가 있다. revocation query가 affected descendants를 찾는다.

fixture는 approved image에 host `.so`를 주입하고 JIT cache를 다른 build로 교체한다. runtime/JIT inventory가 detect해야 한다. SBOM generator와 scanner 자체도 pinned tool·provenance를 가진다. SBOM 존재를 안전 인증으로 쓰지 않는다.

**CUDA build identity를 compiler에서 loaded cubin까지 추적한다**

native extension provenance에는 source, generated source, nvcc/host compiler, C++ ABI, compile/link flags, target SM, PTX/cubin set과 output digest가 필요하다. `TORCH_CUDA_ARCH_LIST`, fast math, debug·line info와 deterministic flag가 결과를 바꿀 수 있다. build log와 response file을 보존한다.

fatbinary가 여러 cubin과 PTX를 포함하면 GPU capability와 driver가 code path를 선택한다. exact GPU, driver와 loaded image를 실행 증거에 둔다. PTX JIT가 발생하면 driver, JIT cache key·output과 cache origin을 기록한다. container digest만으로는 닫히지 않는다.

linked CUDA runtime, cuDNN, NCCL, math library와 libc++가 wheel 내부인지 host인지 resolve한다. `LD_LIBRARY_PATH`, rpath, preload와 plugin search path를 기록하고 actual mapped object digest를 수집한다. path 이름보다 bytes가 identity다.

CUDA toolkit-driver compatibility는 실행 가능 범위를 말하지만 numerical/algorithm equality를 보장하지 않는다. library heuristic, GPU architecture와 workspace가 kernel을 바꿀 수 있다. golden forward/backward/update와 performance fixture를 actual support cell에서 실행한다. 미실행 GPU에 성능 수치를 주장하지 않는다.

reproducible build는 timestamp, build path, archive order와 random seed를 제거하고 clean builder 두 번의 output digest를 비교한다. digest가 다르면 section diff로 원인을 찾는다. bitwise가 불가능하면 semantic binary inventory와 numerical oracle 등급을 명시한다.

**container digest와 host runtime의 경계를 명시한다**

OCI image digest는 image manifest와 layer를 고정하지만 runtime command, environment, secret, mount, device, kernel, driver와 orchestration policy를 고정하지 않는다. DeploymentManifest는 image digest, entrypoint/args, env allowlist, mounts, user/capability, network, resource와 node selector를 가진다.

mutable tag를 요청할 수 있어도 scheduler가 resolve한 digest를 admission·RunManifest에 기록한다. image pull policy와 node cache가 stale tag를 쓰지 않는지 확인한다. running process의 image ID와 catalog BundleID를 sentinel로 대조한다. digest-pinned image도 revoked 상태면 거부한다.

host mount는 config, data와 native library를 image 밖에서 주입한다. exact content digest, read/write mode와 valid policy를 기록한다. broad hostPath와 Docker socket은 신뢰 경계를 크게 넓힌다. training job에 불필요한 권한을 주지 않는다.

secret는 image layer·config·provenance에 포함하지 않는다. secret version/reference와 access policy만 남기고 runtime injection을 audit한다. debug dump와 environment capture에서 value를 redaction한다. secret rotation이 reproducibility를 깨뜨리는 external service behavior를 version한다.

container clean-room test는 network off, read-only root, declared mounts와 least privilege로 load·golden run을 실행한다. hidden network fetch, write path와 host dependency가 실패하면 material closure를 보완한다. production과 다른 permissive test만 통과시키지 않는다.

**dependency lock에서 actual import·load까지 검증한다**

lockfile은 requested dependency와 resolved package artifact를 연결한다. name/version뿐 아니라 index, marker, wheel tag와 hash가 필요하다. editable/local/VCS dependency는 source commit·dirty diff를 기록한다. platform marker가 build와 runtime에서 다른 resolution을 만들 수 있다.

dependency confusion은 internal package 이름을 public index에서 받거나 path shadowing으로 다른 module을 import할 때 생긴다. allowed index와 package origin policy를 둔다. runtime `module.__file__`, distribution metadata와 digest를 inventory한다. current working directory의 same-name file도 검사한다.

Python import 뒤 native loader가 다른 `.so`를 찾을 수 있다. wheel RECORD만 믿지 않고 loaded objects를 확인한다. plugin, tokenizer, attention backend와 optimizer provider의 resolved class/function을 기록한다. optional dependency availability가 branch를 바꿀 수 있다.

dependency update는 source API compatibility 외에 default·kernel·serialization 변화를 시험한다. old/new environment에서 same golden data의 token IDs, first logits, loss/gradient/update, checkpoint load와 export를 비교한다. observed numerical grade와 first difference를 기록한다.

mirror/cache는 artifact를 빠르게 제공하지만 approval을 대신하지 않는다. requested resolution, content digest, origin, fetched time, signature/revocation와 cache generation을 검증한다. offline mode에서도 stale trust bundle로 revoked package를 허용하지 않는다.

**reproducibility를 bitwise·numerical·behavioral로 계약한다**

bitwise reproducibility는 output/checkpoint bytes 또는 selected tensors가 동일하다는 가장 강한 등급이다. deterministic algorithm, same hardware/topology, compiler와 reduction order가 필요할 수 있다. file metadata·serialization order가 달라 byte만 다른 경우 model tensor bitwise와 artifact bitwise를 구분한다.

numerical reproducibility는 tensor별 absolute·relative/ULP tolerance와 first divergence를 사전 정의한다. topology·kernel reduction order 변화로 last bit가 달라질 수 있다. final loss 하나가 비슷하다는 기준은 부족하다. data IDs, logits, gradient, parameter·optimizer update를 단계별로 비교한다.

behavioral reproducibility는 fixed EvaluationCertificate에서 quality·safety distribution이 허용 interval에 있는지 본다. sampling, judge와 metric uncertainty를 포함한다. numerical mismatch를 behavioral pass로 숨기지 않고 서로 다른 grade로 보고한다. 반대로 bitwise 차이가 있어도 approved behavioral range일 수 있다.

reproduction manifest는 소스/material, environment, RNG, data order, topology와 oracle을 가진다. 결과를 본 뒤 tolerance를 넓히지 않는다. 미실행 hardware·scale에는 statistical 추정을 실행 증거로 쓰지 않는다. NotExecuted를 명시한다.

재현 실패는 source/data, environment, RNG/order, kernel, distributed reduction, checkpoint와 evaluation 순으로 first difference를 찾는다. 두 run의 final checkpoint만 diff하지 않는다. divergence certificate가 다음 build·training fix의 입력이다.

**RNG와 data order를 seed 목록보다 깊게 보존한다**

Python, NumPy, torch CPU, CUDA device, model-parallel tracker, dataloader worker와 augmentation이 서로 다른 RNG stream을 쓴다. generator namespace, algorithm, state, logical rank·worker와 next-draw oracle을 저장한다. seed 하나에서 모든 state를 재생성할 수 있다고 가정하지 않는다.

call order가 달라지면 같은 RNG state도 다른 op가 draw를 소비한다. model construction·compiler warm-up과 training RNG를 분리하고 restore 마지막에 training state를 주입한다. dropout, sampler, augmentation와 stochastic rounding mask를 golden fixture에서 비교한다.

data order는 dataset revision, global permutation, epoch, cursor, packing buffer와 worker prefetch를 포함한다. fetched, delivered, update-applied와 checkpoint-committed를 구분한다. rank-local offset만 저장하면 world-size change에서 duplicate·omission이 생긴다.

distributed sampler seed에 physical rank를 넣으면 topology reorder에서 logical order가 달라진다. exact resume가 목표면 global SampleID ledger와 update boundary를 사용한다. statistical data-order grade를 허용하면 replay·omission 범위를 report한다. curriculum scheduler와 mixture cursor도 state다.

fixture는 worker count, world size, prefetch, process restart와 checkpoint boundary를 바꿔 next N SampleIDs·tokens와 RNG draws를 비교한다. batch shape가 같다는 사실로 data equality를 대신하지 않는다. lineage가 없으면 data-reproducible 등급을 부여하지 않는다.

**kernel nondeterminism을 op·shape·topology별로 기록한다**

atomic update, parallel reduction, algorithm heuristic와 race는 동일 input에서도 bitwise 차이를 만들 수 있다. deterministic setting은 framework가 알고 있는 일부 op에 deterministic algorithm을 요구하거나 error를 낼 수 있지만 모든 custom/native kernel을 자동 증명하지 않는다. actual selected kernel을 기록한다.

op inventory는 model forward/backward, optimizer, collective와 preprocessing에서 nondeterministic 후보를 찾는다. shape, dtype, backend와 GPU별 fixture를 실행한다. 동일 process 반복, fresh process와 multi-rank 반복을 구분한다. output뿐 아니라 gradient·update를 본다.

cuDNN/BLAS workspace·heuristic, TF32, reduced-precision reduction와 fast math가 numerical path를 바꾼다. resolved flags와 library version을 RunManifest에 둔다. environment string이 같아도 capability와 workspace 때문에 branch가 달라질 수 있어 profiler/소스 근거를 사용한다.

collective reduction order는 world size·topology와 bucketization에 의존한다. same global state를 reshard한 run은 numerical grade일 수 있다. rank order와 process group generation을 기록한다. all-reduce 결과 tolerance를 first divergence에 포함한다.

성능과 determinism trade-off는 실제 support cell에서 측정한다. deterministic option의 속도 영향을 실행하지 않았다면 수치로 주장하지 않는다. release policy가 요구하는 grade와 cost를 별 decision으로 둔다. nondeterminism을 random seed 실패로만 쓰지 않는다.

**SLSA provenance를 training invocation에 맞게 구체화한다**

subject는 weight file 하나보다 release bundle manifest digest다. source commit, dataset snapshot, base model, tokenizer, container, lockfile와 parent checkpoint가 resolved dependency다. external parameter에는 사람이 요청한 branch·dataset alias와 hyperparameter가 있고 actual resolved values와 구분한다.

builder identity는 개인 이름이 아니라 controlled pipeline/workload identity다. build definition은 training type, entrypoint와 parameter schema를 고정한다. invocation은 exact config, run attempt와 result를 기록한다. distributed workers 전체가 같은 root manifest를 받았는지 handshake evidence를 둔다.

provenance statement가 서명됐어도 subject가 실제 bundle과 맞는지, predicate version을 이해하는지, resolved dependency가 complete한지 검증한다. unknown predicate major를 추측하지 않는다. builder가 해당 repository·channel에 authorization됐는지는 policy layer다.

training은 긴 실행이라 builder가 중간 checkpoint를 만든다. 각 durable generation은 parent UpdateID와 input root, code/environment를 참조한다. final export는 checkpoint subject를 material로 가진 별 builder invocation이다. evaluation, quantization과 merge도 독립 attestation을 만든다.

clean-room verifier는 attestation producer와 독립적으로 subject digest와 critical edge를 재계산한다. builder가 거짓 dependency를 적을 가능성, compromised cache와 signer를 threat model에 둔다. provenance 존재를 behavior safety 인증으로 과장하지 않는다.

**Sigstore 계열 서명을 identity·inclusion·authorization으로 나눈다**

keyless signing은 short-lived certificate와 workload identity를 signature에 연결할 수 있다. verifier는 산출물 digest의 signature, certificate chain, identity claim, validity time와 transparency inclusion evidence를 확인한다. cryptographic validity만으로 production 권한이 생기지 않는다.

policy는 repository, workflow, ref/environment와 release channel에 허용된 identity를 정한다. broad organization identity 하나를 모든 artifact에 허용하지 않는다. pull-request builder와 protected release builder를 구분한다. reusable workflow와 delegated builder relation을 명시한다.

transparency log는 detection과 audit를 돕지만 offline verifier의 freshness·checkpoint와 split-view threat를 고려한다. inclusion proof, signed tree head 또는 bundle이 무엇을 보존하는지 fixed format으로 기록한다. external log가 unreachable할 때 fail/open policy와 cached evidence age를 정한다.

key·identity compromise 시 RevocationID와 affected signing interval, subject와 descendants를 graph query한다. signature를 file에서 삭제하지 않고 revoked state를 policy에 반영한다. running fleet와 mirror cache에 전파한다. clean builder·trust root에서 재서명만 할지 rebuild할지 threat에 따라 결정한다.

fixture는 wrong subject, valid unauthorized identity, expired certificate, missing inclusion, stale trust root와 revoked signer를 각각 넣는다. verifier가 distinct bounded error를 내고 fallback unsigned artifact를 사용하지 않아야 한다. 정상 sibling을 과잉 차단하지 않는지도 본다.

**adapter·merge·quantization lineage를 function 변화로 기록한다**

LoRA artifact는 base ModelID, target module, rank, alpha, dropout, adapter tensor와 config를 필요로 한다. adapter bytes만으로 model function이 완성되지 않는다. base digest가 다른데 name이 같다는 이유로 load하지 않는다. tokenizer/template와 architecture schema도 parent다.

merge는 \(W'=W+sΔW\)를 materialize하는 transformation이다. merge tool commit, dtype, scale, order와 output digest를 기록한다. multiple adapter merge는 non-commutative dtype rounding과 module collision을 고려한다. unmerged runtime과 merged golden logits를 비교한다. repeated merge를 fixture로 막는다.

quantization은 source weight, method, calibration dataset, observer/range, group size, axis, scale·zero-point dtype, kernel layout와 toolchain을 가진다. 같은 4-bit 이름이라도 bytes와 runtime contract가 다르다. calibration data license·privacy와 lineage도 필요하다.

quantized export는 dequantized numerical parity, layer error, golden logits와 EvaluationCertificate를 가진 새 ArtifactID다. behavior pass가 source provenance를 대체하지 않는다. runtime kernel·GPU support와 loaded layout을 admission에서 확인한다. fallback dequant path를 기록한다.

descendant query는 base, adapter, merge, quantization, serving bundle와 deployment까지 잇는다. base나 calibration data가 revoked되면 모든 child를 찾는다. rollback은 compatible base·adapter/tokenizer/runtime bundle 전체를 복원한다.

**distributed checkpoint provenance를 global logical state로 만든다**

checkpoint shard file명과 rank는 identity가 아니다. manifest는 logical parameter, global shape, offset, dtype, optimizer slot, EMA, RNG·data cursor와 topology를 기록한다. FSDP·ZeRO·TP·PP·EP physical placement를 canonical global state와 연결한다.

writer는 immutable shard를 staging에 쓰고 digest·coverage를 검증한 뒤 manifest/commit record를 publish한다. partial generation은 candidate가 아니다. provenance subject는 committed manifest와 all required objects의 closure다. object URI가 바뀌어도 digest relation이 identity다.

reshard는 source checkpoint를 material로 target-layout generation을 만드는 transformation이다. planner/tool revision, source/target topology, byte-range plan, dtype conversion와 validation을 기록한다. optimizer·EMA를 model과 같은 stable parameter ID로 연결한다. rank order로 붙이지 않는다.

checkpoint load는 signature/hash뿐 아니라 next-update semantic oracle를 통과한다. parameter, optimizer/scheduler/scaler, RNG와 next SampleID를 비교한다. exact·numerical recovery grade를 provenance result에 둔다. load 성공을 training continuity 증거로 쓰지 않는다.

retention·GC는 delta/parent DAG와 active lease를 따른다. manifest가 참조하는 shard, encryption key와 dataset dependency를 먼저 지우지 않는다. revocation과 deletion은 operational restore ability를 주기적으로 cold test한다.

**export와 serving bundle을 training checkpoint에서 분기한다**

training checkpoint에는 optimizer·scheduler·scaler·RNG와 shard layout이 있고 serving export는 inference weight, config, tokenizer/template, generation defaults, runtime kernel과 optional adapter를 가진다. export는 단순 copy가 아니라 conversion invocation이다. parent CheckpointID와 selected raw/EMA variant를 기록한다.

state dict key rename, tied weight, transpose, shard gather, dtype cast와 padding이 export function을 바꾼다. tool/source commit과 exact option을 provenance에 둔다. parameter coverage, checksum와 golden first logits를 비교한다. missing/unexpected key warning을 release에서 무시하지 않는다.

serving engine이 weight를 다시 pack·quantize하거나 CUDA graph를 compile하면 runtime-derived artifact가 생긴다. source layout, engine·kernel, GPU capability와 output digest를 runtime supplement에 둔다. node cache hit에도 revoked parent와 policy를 재검증한다.

PipelineBundleID는 model, tokenizer/template, processor/VAE, adapter, quantization, runtime config와 code를 원자적으로 묶는다. replica가 loaded digest sentinel을 보고하고 mixed generation traffic을 막는다. alias update는 canary/evaluation certificate와 rollback parent를 요구한다.

serving incident에서 model name만으로 rollback하지 않는다. affected exact bundle과 descendants를 찾고 new request fence, in-flight policy와 cache eviction을 실행한다. rollback 뒤 golden request·response와 safety canary를 확인한다.

**산출물 DAG의 revocation을 identity·authorization·availability로 나눈다**

identity revocation은 bytes가 악성·잘못됐거나 signer/build가 compromise된 경우다. authorization revocation은 license·consent·policy로 사용 권한이 사라진 경우다. availability deletion은 storage에서 bytes를 지우는 운영 action이다. 세 상태를 하나의 deleted boolean으로 합치지 않는다.

revoked artifact는 조사·법적 보존 때문에 quarantine storage에 남을 수 있지만 production load는 거부해야 한다. verifier는 current policy·revocation snapshot을 검사한다. offline cache가 stale snapshot으로 허용하지 않게 freshness budget을 둔다. running replica에도 push/poll로 전파한다.

impact query는 revoked node의 descendants, deployments, evaluations와 cache를 찾는다. reverse query는 production bundle의 모든 ancestors와 current states를 찾는다. cycle 없는 DAG, exact edge type와 transformation digest를 invariant로 둔다. missing edge는 incident다.

normal sibling을 과잉 차단하지 않게 shared source·builder granularity를 명확히 한다. signer compromise time window, builder invocation와 material scope를 사용한다. 불확실하면 conservative quarantine과 prioritized verification을 한다. policy decision을 기록한다.

revocation rehearsal은 detection, graph query, catalog block, scheduler/node cache, running memory, rollback, clean rebuild와 re-promotion을 측정한다. stale replica와 orphan export를 failure로 잡는다. RTO와 affected requests/GPU time를 보고한다.

**deletion을 storage GC와 lineage tombstone으로 분리한다**

삭제 요청은 raw object, cached copy, transformed shard, checkpoint, backup와 deployment에 서로 다른 action을 요구한다. content bytes를 지워도 lineage tombstone, digest와 decision evidence는 남겨야 과거 사용과 재유입을 추적할 수 있다. tombstone 자체는 원문 복원을 가능하게 하지 않게 설계한다.

content-addressed dedup에서 한 blob을 여러 artifact가 참조할 수 있다. reference count만 믿지 않고 active manifests와 legal hold, reader lease를 mark한다. 한 lineage 삭제가 다른 authorized artifact의 shared byte를 지울 수 있다. logical authorization과 physical object를 분리한다.

backup·mirror·node cache와 running GPU memory는 main registry 삭제와 별 failure domain이다. deletion controller는 위치별 status, attempt와 verification을 기록한다. cache가 offline이면 pending을 성공으로 쓰지 않는다. key destruction을 deletion으로 쓸 때 key replica와 recovery policy를 검증한다.

model influence 제거는 byte deletion과 다르다. training data를 지워도 checkpoint weight에 영향이 남을 수 있다. retrain·unlearning과 behavior/privacy evaluation을 23·24장과 연결한다. 완전 제거 claim의 범위를 명시한다. 이미 배포된 downstream copy를 통제할 수 없는 경우 기록한다.

reintroduction 방지는 tombstone digest와 source lineage를 acquisition gate에서 검사한다. slightly transformed duplicate도 audit한다. deletion 후 same data가 새 URL로 다시 들어오는 fixture를 둔다. license update와 privacy request가 같은 RevocationID 체계를 사용하되 이유를 구분한다.

**policy engine을 verifier와 분리해 과거 판정을 재연한다**

verifier는 digest, signature, predicate schema, tensor/SBOM와 graph relation을 결정적으로 계산한다. policy engine은 trusted builder/signer, required provenance, vulnerability/license exception, reproducibility grade와 channel을 판정한다. code condition문에 현재 policy를 하드코딩하지 않는다.

PolicyRevision은 rules, trust roots, revocation snapshot, effective time와 exception을 가진다. DecisionRecord는 subject, evidence index, policy digest, result, bounded reason와 signer를 담는다. 미래 정책으로 과거 artifact를 재판정할 수 있지만 옛 결정을 덮어쓰지 않는다.

unknown evidence version은 fail closed 또는 explicit manual review다. optional field default를 임의 추정하지 않는다. policy가 unavailable할 때 cached decision freshness와 risk를 정한다. production load가 indefinitely stale authorization을 쓰지 않는다.

exception은 subject·scope, risk, compensating control, owner와 expiry를 가진다. broad package name 예외로 모든 future version을 허용하지 않는다. expiry test와 alert를 둔다. exception descendant도 impact query에 나타난다.

policy fixture는 valid authorized, valid unauthorized, revoked, expired exception, unknown predicate, missing SBOM와 wrong channel을 넣는다. reason code와 fallback action을 검증한다. policy update가 frozen evidence의 decisions를 어떻게 바꾸는지 dual-run한다.

**supply-chain incident를 최초 신뢰 경계에서 조사한다**

incident 시작에는 suspicious subject digest, 발견 source, affected channel와 first known verification event를 기록한다. alias를 즉시 freeze하고 new promotion을 막는다. running fleet의 actual loaded digest와 cache를 query한다. model name·version만으로 scope를 정하지 않는다.

가설은 material substitution, builder compromise, signer abuse, mirror/cache poisoning, loader bypass, 실행 환경 library injection과 data/license incident로 나눈다. 각 가설에 expected graph·log·signature evidence와 반증 query를 둔다. source, build와 runtime timestamp를 logical invocation ID로 맞춘다.

quarantine은 artifact를 삭제하기 전에 evidence를 보존하고 production authorization을 철회한다. access를 제한한다. clean rollback target도 current policy와 dependency closure를 검증한다. compromised trust root가 승인한 old artifact를 그대로 쓰지 않는다.

clean rebuild는 trusted source cut, fresh builder/cache/trust root와 pinned material에서 시작한다. same output hash라도 새 invocation과 verification을 기록한다. output이 다르면 nondeterminism·hidden dependency를 조사한다. behavioral evaluation과 golden load를 통과한다.

RCA는 entry vector, failed/absent gate, affected descendants, impact, containment, eradication, recovery와 fixture를 가진다. “signature invalid”가 아니라 왜 invalid artifact가 promotion/runtime까지 갔는지 쓴다. detection·quarantine·rollback·rebuild RTO를 분해한다.

**clean-room reproduction을 network·cache 차단으로 실행한다**

clean-room은 production registry와 developer home cache를 trusted input으로 쓰지 않는다. 빈 workspace에서 pinned verifier, trust root·policy snapshot와 evidence bundle만 trusted input으로 삼는다. network는 차단하거나 declared endpoints만 proxy해 모든 fetched digest를 기록한다. hidden dependency를 드러낸다.

첫 단계는 exact file set, digest, signature, provenance, revocation와 policy다. 둘째는 safetensors/schema, tokenizer/template와 remote code closure다. 셋째는 build/runtime SBOM와 loaded native inventory다. 넷째는 golden tokenization·logits/update 또는 serving canary다.

rebuild 재현은 소스/material에서 artifact를 다시 만들고 output digest·semantic oracle을 비교한다. training 전체를 반복하기 어려우면 small golden run과 checkpoint continuation을 구분해 grade를 낮춘다. “command가 실행됐다”를 full model reproduced로 쓰지 않는다.

secret·private data는 secure material service와 pseudonymous record를 사용한다. verifier가 원문을 export하지 않고 digest/aggregate를 계산하게 한다. 권한이 없어서 실행하지 못한 cell은 NotExecuted다. privacy를 무시해 재현하지 않는다.

clean-room report는 hidden network fetch, cache dependency, nondeterministic output, unavailable key/data와 actual oracle을 기록한다. fix 뒤 second independent reviewer가 반복한다. producer와 verifier가 같은 code defect를 공유하지 않게 critical digest·edge를 별 구현으로 삼각 측량한다.

**rollback을 artifact bundle과 trust snapshot으로 rehearsal한다**

rollback target은 이전 model weight만이 아니라 tokenizer/template, adapter/quantization, runtime image, data/config dependency, policy/trust와 evaluation certificate를 포함한 승인 BundleID다. current trust에서 revoked면 과거에 approved였어도 target이 아니다. safe ancestor를 graph에서 찾는다.

promotion controller는 alias compare-and-swap, running deployment generation과 fencing을 사용한다. stale controller가 새 bad artifact를 다시 promote하지 못하게 한다. node cache와 model server memory를 evict하고 actual loaded digest sentinel로 확인한다. in-flight request policy를 정한다.

rollback 뒤 golden tokenization, logits, response, capability·safety canary와 telemetry를 실행한다. training rollback이면 data cursor·optimizer checkpoint와 next update를 검증한다. serving rollback과 training lineage를 구분한다. 새 DecisionRecord와 incident relation을 만든다.

trust root rollback은 특히 조심한다. compromised root로 서명된 artifact를 old policy가 허용할 수 있다. trust snapshot과 revocation을 current safe state로 유지하면서 artifact만 safe parent로 돌린다. policy와 artifact version을 하나의 번호로 묶지 않는다.

rehearsal은 signer, builder, dataset, native library와 model bundle revocation을 각각 주입한다. impact scope, quarantine, rollback, clean rebuild와 re-promotion을 시간 측정한다. 정상 sibling 과잉 차단과 stale replica를 failure로 잡는다.

**audit query를 양방향으로 표준화한다**

forward query는 material에서 모든 transformation, checkpoint, adapter·quantized export, evaluation와 deployment descendants를 찾는다. reverse query는 running BundleID에서 source, data, builder, dependency, license, signature와 policy ancestors를 찾는다. 두 결과가 edge 역관계로 맞아야 한다.

node에는 type, digest, schema, created event, authorization·revocation와 storage location이 있다. edge에는 transformation type, invocation, config/tool digest와 validation result가 있다. URI와 display name은 property일 뿐 identity가 아니다. graph cycle과 orphan required node를 invariant로 막는다.

query는 time·policy snapshot을 받는다. artifact가 당시 approved였지만 현재 revoked된 상태를 설명할 수 있다. mutable current status만 보존하면 과거 decision을 재연할 수 없다. event sourcing과 successor relation을 사용한다.

large graph에서는 index 누락과 eventual consistency가 생길 수 있다. 출시 관문는 required closure를 transactional evidence index로 받고 async warehouse query와 reconcile한다. missing descendant가 revocation에 빠지는 fixture를 둔다. graph health metric과 owner를 운영한다.

감사 report는 node count보다 coverage를 말한다. expected file/dependency/row/checkpoint set과 observed closure를 비교한다. sample audit만으로 100%를 주장하지 않는다. private node는 digest와 secure access procedure로 존재를 증명한다.

## 27.14 golden suite와 fleet 관측으로 release 이후까지 검증한다

최종 suite는 정상 설치보다 치환·누락·오염·upgrade 실패를 먼저 심고, fleet가 실제로 적재한 identity를 지속적으로 대조한다.

suite는 floating ref substitution, cache symlink swap, partial download, safetensors corruption, pickle unsafe global, remote code network access, native `.so` injection, wrong signature subject와 revoked signer를 포함한다. 각 fault는 load/build 전에 expected gate에서 막혀야 한다.

data fixture는 license revoked row, transform config swap, split leakage, deleted record reintroduction와 packed lineage missing을 넣는다. checkpoint fixture는 partial shard, tokenizer mismatch, optimizer omission와 wrong parent를 넣는다. export fixture는 adapter wrong base, repeated merge와 quantization scale loss를 넣는다.

정상 path는 source resolve→hermetic build/training fixture→checkpoint→export→signature/SBOM→policy→promotion→load canary를 잇는다. 각 subject와 edge digest를 저장한다. clean-room에서 same evidence와 numerical/behavioral grade를 재현한다. actual large-scale performance는 실행한 support cell만 보고한다.

revocation path는 signer·dataset·native dependency와 model bundle을 각각 철회하고 impact query, fleet quarantine, rollback와 rebuild를 실행한다. stale cache, offline node와 running memory를 포함한다. 정상 sibling이 계속 서비스 가능한지도 본다.

suite 결과는 source/tool versions, input fixture, expected/actual stage, decision code, incident owner와 regression을 가진다. 새 dependency·CUDA·builder·policy 변경에서 affected tests를 자동 선택한다. green happy path만으로 공급망을 승인하지 않는다.

### 최종 supply-chain certificate의 최소 필드

certificate는 BundleID와 exact subject file set, 소스/material resolutions, builder invocation, checkpoint/export transformations, signature/provenance, SBOM/runtime inventory와 PolicyDecision을 가진다. dataset·model cards와 license/consent snapshot을 연결한다. alias는 별 promotion record다.

environment에는 container digest, command/effective config, OS/kernel, Python wheels, loaded native objects, GPU/driver/CUDA/cuDNN/NCCL, compiler/JIT artifact와 topology가 있다. RNG/data order와 deterministic flags, reproduction oracle·grade를 넣는다. unknown과 NotExecuted를 구분한다.

loader section은 safetensors format/bundle validation, pickle policy, remote code/import closure, cache/mirror resolution과 runtime resolved class를 가진다. distributed state는 checkpoint manifest, global coverage와 recovery certificate다. adapter/merge/quantization parent를 닫는다.

security·operations는 revocation snapshot, key/trust root, exception·expiry, deletion status, impact queries, rollback target와 rehearsal 결과를 포함한다. vulnerability scan database revision과 reachable analysis를 둔다. privacy evidence는 access 통제를 유지한다.

독립 검토자는 무작위 node/edge, signature subject와 actual loaded binary를 재계산하고 negative fixture를 실행한다. 어느 artifact가 어디서 왔고 어떤 코드·data·environment로 만들어졌으며 현재 왜 허용되고 어떻게 철회할지 답할 수 있어야 서명한다.

### 장간 supply-chain handoff를 artifact contract로 고정한다

4·6장에는 raw data, transform, mixture와 SampleID lineage가 있고 5장에는 tokenizer/template bundle이 있다. 10·18·19·20·22장은 base model, adapter, preference/reward, RL과 diffusion component transformation을 만든다. 27장은 이들을 digest·invocation·authorization edge로 묶는다.

14·15·16장은 CUDA kernel, distributed topology와 cluster environment를 다룬다. build SBOM과 runtime inventory가 실제 loaded code를 닫고 support matrix가 미실행 hardware를 표시한다. 17장에는 durable checkpoint generation과 next-update certificate가 있다.

23장은 editing/unlearning·deletion claim과 descendants를 다루고, 24·25장은 EvaluationCertificate와 red-team case를 다룬다. evaluation도 source/data/judge/sandbox supply chain을 가진다. 출시 산출물가 test firewall을 소비했는지 lineage로 확인한다.

26장은 actual loaded digest, metric/rule/incident와 revocation fleet 상태를 관측한다. 28·29장은 single-GPU golden과 multi-node failure injection으로 clean-room·rollback을 실행한다. 30장은 certificate와 policy decision을 출시 관문로 소비한다.

crosslink는 장 번호만 적지 않고 ArtifactID, edge, invariant, verifier와 failure owner를 넘긴다. 이 계약이 있어야 supply-chain graph가 책 전체의 data·code·model·operation을 하나의 감사 가능한 계보로 만든다.

### 인사이트가 드러나는 마지막 검토 질문

artifact를 볼 때 이름이 아닌 digest와 exact file set은 무엇인가. 누가 어떤 builder에서 어떤 source·data·base·environment로 만들었는가. external parameter와 actual resolved dependency가 모두 있는가. signature subject와 실제 bytes가 같은가. 현재 policy에서 signer·builder가 이 channel에 권한이 있는가.

load 전에 format, schema, bundle compatibility와 resource limit을 검증하는가. pickle과 remote code에 어떤 능력을 허용했는가. 실제 imported module, loaded native object와 JIT kernel이 SBOM에 있는가. CUDA architecture·driver가 어느 code path를 선택했는가.

training을 재현할 때 RNG streams, data order, topology, kernel nondeterminism과 checkpoint state가 닫혔는가. bitwise·numerical·behavioral 중 어느 grade를 어떤 oracle로 실행했는가. 실행하지 않은 GPU·scale을 supported로 쓰지 않았는가.

revocation이나 deletion이 오면 descendants와 running replicas를 찾을 수 있는가. offline cache와 stale trust, backup와 memory를 처리하는가. safe rollback target은 current policy를 통과하는가. clean rebuild가 compromised material·builder·cache를 재사용하지 않는가.

독립 verifier가 빈 환경에서 critical path와 negative fixtures를 재실행할 수 있는가. 답이 문서의 주장이나 mutable alias에만 의존한다면 edge가 비어 있다. 그 빈 edge가 다음 provenance, policy, test와 운영 작업의 우선순위다.

**reproducible build의 숨은 입력을 하나씩 제거한다**

같은 source와 lockfile인데 output digest가 다르면 먼저 timestamp, timezone, locale, hostname, absolute build path, file traversal order, archive metadata와 random seed를 본다. compiler가 build ID나 temporary path를 binary에 넣을 수 있다. clean builder 두 개의 environment diff와 binary section diff를 만든다.

parallel build는 dependency가 불완전하면 race에 따라 generated file order나 content가 달라질 수 있다. `-j1`과 parallel 결과를 비교하되 single-thread를 permanent 해결로 숨기지 않는다. build graph dependency와 deterministic sort를 고친다. generated code의 map iteration도 안정화한다.

package archive는 file mtime, uid/gid, permission, order와 compression library가 bytes를 바꾼다. canonical epoch, normalized metadata와 deterministic archiver를 사용한다. container layer도 instruction, package index와 file timestamp가 영향을 준다. semantic contents가 같아도 digest가 흔들리면 cache·signature와 audit 비용이 커진다.

compiler·linker nondeterminism은 debug section, symbol ordering와 LTO에서 나타날 수 있다. exact toolchain image와 flags를 고정하고 output section별 hash를 비교한다. 재현 불가능한 section을 무조건 strip하기 전에 runtime/debug·security 의미를 검토한다. strip tool도 provenance material이다.

bitwise reproducible build가 되지 않으면 원인을 기록하고 semantic fallback을 정의한다. exported symbol, cubin/PTX inventory, loaded behavior와 golden numerical oracle를 비교한다. 결과를 “동일 build”로 쓰지 않고 lower grade로 명시한다. 후속 toolchain에서 gap을 다시 줄인다.

**configuration provenance를 requested·parsed·effective로 분리한다**

사용자가 제출한 YAML/CLI는 requested config다. parser가 default, environment interpolation, type conversion과 deprecated alias를 적용한 결과가 parsed config다. framework가 hardware capability, world size와 optional library를 보고 선택한 branch가 effective state다. 세 층을 모두 보존한다.

환경 변수와 secret substitution은 value를 그대로 provenance에 넣지 않는다. non-secret resolved value와 secret reference/version·digest를 구분한다. `${VAR:-default}`가 host마다 다른 결과를 내는지 검증한다. unknown option과 typo를 warning으로 무시하지 않고 release config에서 거부한다.

Hydra/templating이나 include가 있으면 source files, composition order와 override history를 material로 둔다. rendered final config만 남기면 왜 값이 선택됐는지 잃고 requested files만 남기면 actual state를 잃는다. canonical effective digest를 RunID에 사용한다.

world size로 derived global batch, accumulation, scheduler steps와 sharding mesh를 계산한다. default attention backend, precision와 compiler mode도 resolved class/function과 함께 기록한다. `auto`라는 문자열은 재현 identity가 아니다. capability decision log를 둔다.

fixture는 missing env, extra unknown key, deprecated alias conflict, include order, `auto` backend와 resume override를 넣는다. same requested config가 다른 effective state가 되면 BundleID가 달라져야 한다. policy가 허용하지 않은 auto fallback을 fail closed한다.

**license·consent policy를 provenance graph에서 실행한다**

license text는 name tag보다 exact document digest, source, effective date와 scope가 필요하다. code, data, base model, adapter와 generated artifact가 서로 다른 conditions를 가진다. commercial use, redistribution, attribution, field-of-use와 share-alike를 구조화하되 최종 판단은 승인된 legal review와 연결한다.

data consent는 license와 별일 수 있다. 수집 목적, biometric/voice/face, jurisdiction, retention와 withdrawal을 기록한다. public access가 training consent를 자동 의미하지 않는다. row/shard EligibilityState를 integrity와 분리한다. policy가 바뀌면 same bytes의 authorization만 revoke할 수 있다.

derivative relation은 단순 file 포함과 model training influence에서 법적 해석이 복잡하다. graph는 사실 관계—어떤 data/base/code를 어떤 invocation이 사용했는지—를 정확히 제공하고 legal conclusion을 자동 과장하지 않는다. decision, counsel/owner와 scope를 별 policy artifact로 둔다.

model/dataset card의 tag와 repository license file이 충돌하면 자동 통과하지 않는다. fetched 당시 license snapshot과 current change event를 비교한다. source가 license를 소급 변경했다고 주장해도 policy가 어떻게 처리할지 기록한다. mutable web URL만 evidence로 쓰지 않는다.

license incident fixture는 wrong tag, missing attribution, revoked data source와 incompatible redistribution condition을 넣는다. promotion·export channel별 decision과 descendants를 확인한다. security revocation과 같은 operational machinery를 쓰되 reason·remedy를 구분한다.

**vulnerability scan을 component reachability와 운영 상태에 연결한다**

scanner result는 component identifier, version range, advisory database revision, severity와 evidence를 가진다. SBOM name mapping 오류와 vendored fork 때문에 false positive/negative가 생긴다. actual file/symbol digest와 loaded inventory를 확인한다. scan timestamp와 database freshness를 release certificate에 둔다.

reachability는 vulnerable code가 artifact에 포함됐는지, runtime에 load됐는지, untrusted input과 privilege에서 호출 가능한지를 분석한다. Python import가 없다고 native transitive library를 safe라 하지 않는다. JIT compiler와 build-only tool도 supply-chain compromise path가 될 수 있다.

severity만으로 decision하지 않고 exploitability, data/model confidentiality, cluster privilege와 compensating control을 본다. 그러나 reachability unknown을 unreachable로 쓰지 않는다. exception은 subject exact version, channel, owner, control와 expiry를 가진다. 다음 scan에서 자동 재검토한다.

critical advisory가 생기면 reverse dependency graph로 running job, checkpoint, image와 serving bundle을 찾는다. patched rebuild는 소스/material과 policy를 새로 만들고 golden numerical·behavior test를 실행한다. package version만 올리고 native loaded object가 old인지 확인한다.

fixture는 SBOM missing component, wrong purl, runtime-only vulnerable `.so`, stale advisory DB와 expired exception을 넣는다. scanner pass만으로 promotion되지 않고 evidence coverage와 policy가 함께 판정해야 한다.

**transparency evidence와 offline verification의 freshness를 설계한다**

서명 bundle에 certificate와 inclusion proof가 있어도 verifier가 어느 trust root, log key와 signed tree state를 사용했는지 알아야 한다. offline bundle은 당시 verification을 재연하는 데 유용하지만 current revocation·policy freshness도 필요하다. historical validity와 current authorization을 분리한다.

transparency log inclusion은 artifact가 공개 append-only record에 있었다는 증거이지 builder correctness나 model safety 증거가 아니다. identity certificate claim과 source workflow authorization을 별 검증한다. log unavailable 때 cached inclusion을 허용하는 최대 age와 channel risk를 정한다.

split-view나 compromised log threat에는 trusted checkpoint distribution과 monitor가 필요할 수 있다. 시스템이 실제 구현하지 않은 강도를 주장하지 않는다. 보존한 signed state와 independent monitor evidence 범위를 certificate에 적는다. unknown은 manual review다.

clock skew는 certificate validity와 build timestamp 판정에 영향을 준다. trusted timestamp source, signing event와 verification time를 기록한다. artifact metadata time을 security truth로 쓰지 않는다. expired certificate라도 signing 당시 valid한 bundle verification이 가능한 format인지 fixed spec에 따른다.

fixture는 missing inclusion, wrong log identity, stale checkpoint, not-yet-valid/expired certificate와 current revoked identity를 넣는다. cryptographic error, evidence freshness와 policy authorization을 distinct reason으로 낸다. operator가 unsigned fallback을 선택하지 못하게 한다.

**mirror·cache를 verified content store로 운영한다**

mirror는 upstream repository·package·model bytes를 내부에 보존하지만 origin과 immutable ref mapping을 잃으면 새로운 mutable source가 된다. entry는 requested origin/ref, resolved revision, content digest, fetched identity/time, signature·license와 verification state를 가진다. internal URI는 location일 뿐 identity가 아니다.

download는 temporary object에 쓰고 length/hash/signature를 확인한 뒤 immutable namespace로 promote한다. partial file, LFS pointer와 HTML error page를 expected format·size로 거부한다. same key overwrite를 금지한다. manifest commit 없이 shard 일부를 노출하지 않는다.

cache hit도 current revocation·policy와 requested snapshot edge를 확인한다. 과거 approved blob이 current production에 허용된다는 뜻은 아니다. offline loader는 signed trust/revocation snapshot의 freshness를 확인한다. stale일 때 risk policy에 따라 block한다.

cache GC는 active bundle closure, legal hold와 cold-rebuild anchor를 보존한다. content-addressed shared blob을 one artifact expiry로 삭제하지 않는다. 반대로 revoked blob을 normal cache에 남겨 accidental load하지 않게 quarantine namespace와 admission을 둔다.

fixture는 symlink swap, same URI different bytes, partial upload, stale ref, revoked offline hit와 mirror signer compromise를 넣는다. loader가 digest·policy gate 전에 parse/import하지 않는지 본다. origin 장애에서 승인되지 않은 fallback mirror를 조용히 쓰지 않는다.

**deterministic training 옵션의 보장 범위를 실제 op로 확인한다**

framework deterministic mode는 알고 있는 nondeterministic op에 deterministic alternative 또는 error를 요구할 수 있다. 설정이 성공했다고 custom CUDA extension, external library와 distributed collectives 전체가 결정적이라고 증명되지는 않는다. selected op/kernel inventory와 repeat test를 사용한다.

random seed, deterministic algorithms, benchmark/TF32, matmul precision, cuDNN과 environment 옵션을 effective config에 둔다. 동일 flag 이름의 version별 default와 support를 fixed source에서 확인한다. option 하나를 performance/correctness 만능 switch로 설명하지 않는다.

golden run은 same process repeat, fresh process, same node, different same-SKU node와 multi-rank를 단계적으로 실행한다. token IDs, RNG draws, selected activations, loss, gradient, parameter·optimizer update를 비교한다. 최초 divergence op, dtype와 ULP를 기록한다.

distributed data order와 collective tree가 다르면 bitwise target이 불가능할 수 있다. logical same-world-size와 elastic topology를 별 grade로 둔다. statistical behavior evaluation은 lower numerical grade를 숨기지 않는다. tolerance를 결과 후 넓히지 않는다.

deterministic mode의 overhead는 actual workload·hardware에서 profiler-off 반복으로 측정한 cell만 보고한다. 접근하지 않은 GPU에서는 예상 trade-off로 서술하지 않고 test plan을 둔다. reproducibility grade와 cost decision을 분리한다.

**build-to-run attestation을 long training job에 연결한다**

container/wheel build attestation만으로 long training invocation이 닫히지 않는다. scheduler가 approved image를 어떤 command/config, dataset/base/checkpoint와 실행했는지 RunAttestation을 만든다. membership generation과 worker root manifest handshake를 기록한다. 일부 rank가 다른 image/cache를 쓰면 중단한다.

training 중 code hot patch, package install와 remote fetch를 금지하거나 new material event로 기록한다. filesystem integrity monitor와 loaded module/native inventory를 시작·주기·종료에 비교한다. JIT artifact는 parent environment와 output digest를 추가한다. ephemeral generated file을 놓치지 않는다.

checkpoint generation마다 RunID, source/data/environment root와 UpdateID를 참조한다. job reschedule는 new AttemptID와 parent checkpoint를 만든다. W&B run name이나 scheduler job ID를 training identity로 대신하지 않는다. wall-clock interruption과 logical lineage를 분리한다.

final export builder는 exact checkpoint generation과 raw/EMA variant를 material로 받는다. evaluation builder는 export, task/judge/sandbox를 materials로 가진다. production promotion은 모든 evidence가 same subject bundle을 가리키는지 확인한다. pipeline stage alias를 다시 resolve하지 않는다.

RunAttestation producer가 compromise될 수 있어 scheduler event, storage commit와 independent runtime sentinel로 critical fields를 삼각 측량한다. self-reported digest만 믿지 않는다. mismatch는 supply-chain incident다.

**fleet에서 actual-loaded identity를 지속 검증한다**

catalog가 approved digest를 가리켜도 replica가 old cache, wrong mount나 partial rollout로 다른 bytes를 load할 수 있다. process는 model/tokenizer/code/runtime BundleID와 loaded native supplement를 signed 또는 authenticated sentinel로 보고한다. control plane은 desired와 actual을 비교한다.

sentinel은 model name·version이 아니라 canonical manifest digest와 component digests를 사용한다. sampling interval, startup admission와 periodic check를 둔다. runtime JIT나 adapter hot-swap이 있으면 generation을 올린다. high-cardinality replica detail은 operational store에 두고 aggregate compliance를 metric으로 낸다.

mixed generation traffic은 behavior와 evaluation을 오염시킨다. load balancer가 BundleID를 request trace에 넣고 canary/rollback이 exact cohort를 선택하게 한다. unknown actual identity replica는 traffic에서 격리한다. self-report가 불가능하면 node-level file/runtime verifier를 사용한다.

revocation 시 affected actual-loaded replicas, node cache와 in-flight request를 찾는다. new assignments를 fence하고 graceful/forced drain policy를 적용한다. cache eviction 뒤 stale process가 다시 materialize하지 못하게 catalog·policy generation을 검사한다.

fixture는 wrong local blob, stale adapter slot, old tokenizer, injected `.so`와 disconnected node를 넣는다. catalog dashboard가 green이어도 actual identity mismatch alert가 울리고 traffic이 차단돼야 한다. rollback 뒤 golden sentinel을 실행한다.

**공급망 운영 SLO를 검증·철회 시간으로 정의한다**

coverage SLI는 production bundle에서 required provenance, signature, SBOM, license, evaluation와 runtime identity closure가 완전한 비율이다. unknown을 pass 분모에 넣지 않는다. verification freshness와 exception age를 본다. 숫자만 높고 critical channel 하나가 비면 강제 관문다.

promotion SLO는 artifact ingestion부터 verified·approved·deployed까지 phase별 latency를 본다. 빠른 승격 때문에 scan·golden test를 skip하지 않는다. emergency release는 explicit exception과 후속 deadline을 가진다. verifier backend outage와 artifact rejection을 분리한다.

revocation SLO는 detection, impact query, new-load block, fleet quarantine, rollback와 clean rebuild로 나눈다. bytes·signer·dataset·license incident별 목표가 다를 수 있다. offline node와 mirror까지 closure를 확인한다. average보다 worst critical descendant를 본다.

reproducibility SLO는 cold bundle verification, rebuild success와 oracle grade를 주기적으로 측정한다. evidence가 storage에 있어도 key·tool·private data 접근이 만료되면 실패다. clean-room drill 결과를 metric으로 낸다. 실행하지 않은 full retraining은 grade에 포함하지 않는다.

error budget 소진 시 floating dependency, builder upgrade와 high-risk promotion을 제한한다. threshold를 최근 실패에 맞춰 낮추지 않는다. supply-chain monitoring과 26장의 alert/runbook을 연결한다. SLO 자체도 PolicyRevision이다.

**audit report를 주장·증거·결정으로 분리한다**

주장은 “이 bundle은 source X와 data Y로 만들어졌다”, “numerically reproducible하다”, “production 사용이 허용된다”처럼 쓴다. 각 주장에는 provenance edge, oracle result와 policy decision이 필요하다. signature valid 하나로 세 주장을 모두 지지하지 않는다.

증거는 raw manifest, subject digest, source/build log, SBOM, signature bundle, loader/golden output와 graph query다. evidence producer, schema, timestamp와 access를 기록한다. mutable dashboard screenshot이나 latest URL을 critical evidence로 쓰지 않는다. private evidence에는 secure reviewer 절차를 둔다.

결정은 특정 PolicyRevision에서 channel·scope와 expiry를 가진다. 같은 evidence도 research sandbox와 public serving에서 다른 결과일 수 있다. waiver와 compensating control, signer와 reversal trigger를 포함한다. 현재 revoked artifact의 historical approval을 설명할 수 있다.

report table의 각 cell은 node/edge와 DecisionRecord로 drill down한다. manual spreadsheet 수정과 copied hash를 금지한다. independent reviewer가 random node의 digest와 signature subject를 재계산한다. graph forward/reverse query count를 대조한다.

잘못된 주장 발견 시 old report를 덮지 않고 withdrawal와 successor를 연결한다. affected release·paper/model card를 찾는다. audit가 blame 문서가 아니라 executable correction mechanism이 되게 한다. 발견된 gap은 golden negative fixture로 변환한다.

**최종 release rehearsal을 공격자 역할과 함께 실행한다**

공격자 팀은 mutable ref, mirror substitution, dependency confusion, remote code side effect, stolen signer, SBOM omission, cache replay와 adapter wrong-base를 시도한다. 방어 팀은 expected gates와 bounded error를 관측한다. production secret나 실제 malicious payload 대신 safe synthetic fixture를 쓴다.

한 번에 fault 하나를 넣어 최초 방어선을 확인한 뒤 일부 복합 fault를 실행한다. valid signature+wrong subject, approved image+runtime `.so`, revoked data+offline cache처럼 단일 gate를 우회하는 조합을 만든다. defense-in-depth를 실제로 시험한다.

운영 팀은 alert에서 exact digest impact query, quarantine, rollback와 clean rebuild를 제한 시간 안에 수행한다. broad alias나 package name으로 정상 sibling을 과잉 차단하지 않는다. stale node, running GPU memory와 mirror를 포함한다. evidence를 보존한다.

독립 verifier는 정상 bundle과 공격 fixture를 network 없는 clean room에서 재검사한다. producer와 verifier output이 같은 결함을 공유하는지 critical digest를 수동/별 구현으로 확인한다. 미실행 CUDA/hardware cell은 과장하지 않는다.

rehearsal 결과는 missed edge, slow query, stale policy, unsafe command와 permission gap을 system backlog로 만든다. 수정 뒤 variant를 다시 실행한다. certificate와 runbook, policy·golden suite가 함께 갱신돼야 release가 재개된다.

**공급망 완성도를 판단하는 종합 인수표**

source·data 인수는 immutable resolution, dirty/generated closure, row transformation·license와 deletion lineage를 본다. build 인수는 hermetic materials, builder identity, reproducible output와 provenance를 본다. environment 인수는 container·host·CUDA/native/JIT actual inventory를 본다.

artifact 인수는 exact bundle, safetensors/schema, pickle/remote-code sandbox, checkpoint state, adapter/merge/quantization과 serving export를 본다. signature 인수는 subject, identity, inclusion, authorization·revocation와 policy를 본다. SBOM과 vulnerability exception을 포함한다.

reproducibility 인수는 RNG/data order, kernel·collective, checkpoint next update와 bitwise/numerical/behavioral oracle을 실제 support cell에서 실행한다. performance는 실제 profiler-off measurement가 있는 cell만 쓴다. NotExecuted를 PASS로 바꾸지 않는다.

운영 인수는 catalog transaction, actual-loaded identity, cache/mirror, impact query, deletion, fleet revocation, rollback·clean rebuild와 clean-room drill을 본다. security·privacy evidence와 access expiry를 확인한다. cards·evaluation·deployment decision까지 연결한다.

최종 질문은 누구나 BundleID 하나에서 source·data·builder·binary·policy와 behavior evidence를 거슬러 올라가고, ancestor 하나의 철회에서 모든 실제 descendants를 찾아 안전하게 rollback할 수 있는가다. 가능해야 공급망은 문서가 아니라 작동하는 신뢰·복구 system이 된다.

**source·spec upgrade를 semantic anchor로 승인한다**

safetensors, PyTorch serialization, Transformers loader, provenance verifier나 SBOM tool을 올릴 때 version number만 바꾸지 않는다. old/new fixed commit에서 selected symbol, input schema, default, validation order, error와 test를 diff한다. line 이동과 semantic change를 구분하기 위해 symbol·body digest를 사용한다.

golden artifact를 old/new parser·verifier에 넣어 accepted/rejected matrix를 만든다. 정상 bundle, corrupted offset, unsafe global, remote code, unknown predicate와 revoked signer를 포함한다. new version이 더 관대해진 row는 security review 없이 승인하지 않는다. 더 엄격해진 row는 affected existing artifact와 migration을 계산한다.

format/spec major version이 바뀌면 unknown field를 추측하지 않는다. converter는 source artifact를 덮어쓰지 않고 new subject와 provenance를 만든다. migration 전후 tensor/schema, golden logits와 policy decision을 비교한다. raw source를 rollback anchor로 보존한다.

dependency upgrade가 native binary를 바꾸면 Python API parity만으로 부족하다. SBOM/runtime inventory, CUDA kernel selection와 numerical oracle을 다시 실행한다. actual hardware가 없는 cell은 NotExecuted다. source-confirmed behavior를 performance evidence로 쓰지 않는다.

upgrade report에는 changed acceptance, BundleIDs, decision reversal, reproduction grade, measured cost와 rollback이 있다. overlap verifier를 운영하고 pointer를 transactional하게 이동한다. 과거 certificate를 새 tool 결과로 조용히 덮어쓰지 않는다.

**quantized serving artifact의 runtime closure를 검증한다**

quantized weight는 scale·zero-point와 packed layout만으로 완성되지 않는다. group size, axis, activation quantization, calibration, outlier path와 runtime kernel이 필요하다. loader가 unsupported GPU에서 dequant fallback을 쓰는지, 다른 kernel을 선택하는지 actual resolved path를 기록한다.

calibration dataset은 DatasetRevision, license·privacy와 sampling을 가진다. observer statistics, clipping percentile, excluded layer와 tool commit을 provenance edge에 둔다. 같은 source weight라도 calibration이 다르면 다른 ArtifactID다. random calibration order와 dtype을 고정한다.

export verifier는 exact parameter coverage, scale shape·range, packed bytes와 dequantized tensor error를 본다. golden tokens에서 logits, selected activation, generation와 EvaluationCertificate를 비교한다. acceptable behavior가 integrity·license를 대신하지 않는다. numerical tolerance를 결과 뒤 넓히지 않는다.

serving bundle은 quantized model, tokenizer/template, adapter merge, runtime engine, kernel library, GPU capability와 config를 묶는다. node가 compiled engine/JIT cache를 만들면 runtime-derived subject로 보고한다. stale cache와 wrong GPU layout을 admission에서 막는다.

rollback은 fp/bf source나 이전 quantized bundle 중 current policy와 hardware를 통과한 target을 고른다. model file만 바꾸고 old scales·engine cache를 남기지 않는다. actual-loaded sentinel과 golden request를 확인한다. runtime 성능은 실행한 support cell에서만 보고한다.

**deletion·revocation drill을 실제 fleet와 cache까지 확장한다**

drill은 revoked dataset row, compromised signer, vulnerable native library와 wrong model bundle을 각각 선택한다. graph forward query로 checkpoint, adapter·quantized export, evaluation, deployment와 node cache를 찾는다. expected descendant set과 actual query를 비교한다. missing edge와 overbroad sibling 차단을 모두 실패로 센다.

catalog는 new load를 막고 scheduler·serving control plane이 policy generation을 받는다. disconnected node와 offline mirror가 stale approval을 계속 쓰는지 시험한다. running process memory와 adapter slot의 actual digest를 확인하고 in-flight request policy를 적용한다. registry flag만 바꾸고 완료라 하지 않는다.

rollback target은 current trust, license, vulnerability와 EvaluationCertificate를 통과한다. alias compare-and-swap과 fencing으로 stale promoter를 막는다. node cache·compiled engine과 tokenizer namespace를 정리한다. canary 뒤 gradual traffic을 시작한다.

clean rebuild는 compromised 소스/material, builder, key와 cache를 격리한다. same bytes가 나와도 new invocation·signature와 decision을 만든다. 다른 bytes면 hidden dependency·nondeterminism을 조사한다. behavioral·security evaluation을 재실행한다.

drill report는 detection, graph query, block, fleet convergence, rollback와 rebuild 시간, affected requests/GPU jobs와 unresolved copies를 가진다. deletion은 tombstone과 reintroduction prevention까지 확인한다. 다음 교대자가 같은 evidence로 결론을 재현한다.

**독립 감사 certificate의 마지막 검산**

certificate index는 모든 manifest, provenance, signature, SBOM, policy, cards, EvaluationCertificate, revocation·rollback과 clean-room result의 digest를 가진다. index 자체도 authorized signer가 exact subject set에 서명한다. child evidence가 빠진 서명은 완전한 package를 증명하지 않는다.

auditor는 무작위 source, dataset row/shard, checkpoint parameter, adapter/quantization와 running replica를 선택한다. reverse lineage와 actual bytes를 확인하고 transformation tool·config·validation을 재계산한다. 이름·URI와 self-report만 믿지 않는다. private evidence는 secure procedure로 검증한다.

signature 하나는 actual 산출물 digest, identity certificate, transparency evidence와 channel authorization까지 확인한다. SBOM component 하나는 package source, file digest, loaded runtime와 vulnerability/license decision까지 추적한다. policy snapshot을 과거/current로 바꿔 decision 재연성을 본다.

negative fixture는 wrong subject, stale cache, partial bundle, unauthorized remote code, runtime `.so`, revoked ancestor와 missing graph edge를 포함한다. distinct gate가 expected reason으로 거부해야 한다. fallback artifact를 조용히 사용하거나 report만 warning이면 실패다.

최종 서명에는 tested hardware/topology, reproduction grade, NotExecuted, exception·expiry와 next trigger가 있다. certificate를 읽을 수 없거나 trust/key/data가 만료되면 support grade를 낮춘다. 이 독립 검산이 반복될 때 공급망은 장기적으로 감사 가능하고 실제 철회 가능한 상태를 유지한다.

**재현 실패를 first-difference matrix로 조사한다**

두 run의 final metric만 비교하지 않는다. requested·resolved 소스/material, effective config, module/native inventory, dataset SampleID·tokens, RNG draws, first logits, loss target·denominator, gradient, update와 checkpoint manifest를 순서대로 비교한다. 처음 다른 artifact/tensor가 다음 조사 owner를 정한다.

소스/material이 다르면 training numerical 분석을 중단하고 resolution·cache를 고친다. data IDs가 다르면 sampler·cursor·transform을, token이 다르면 tokenizer/template를 본다. forward부터 다르면 weight/runtime/kernel, backward부터 다르면 precision·activation checkpoint·collective, update부터 다르면 optimizer/scaler를 본다.

environment diff는 package version보다 actual imported module, loaded `.so`, CUDA kernel path와 topology를 본다. config가 같아도 optional dependency와 auto backend가 달라질 수 있다. resolved class/function과 capability decision을 대조한다. hidden JIT/cache artifact를 찾는다.

bitwise mismatch가 예상되는 topology에서는 numerical tolerance와 first ULP divergence를 사용한다. behavioral evaluation이 pass해도 provenance/numerical grade mismatch를 숨기지 않는다. 반대로 산출물 byte가 timestamp로만 달라도 reproducible-build 문제를 별로 고친다.

incident output은 source of first difference, affected BundleIDs, reproduction grade change, fix와 new fixture를 가진다. “seed가 달랐다”로 닫지 않고 어떤 RNG namespace·call order가 왜 달랐는지 적는다. missing evidence는 추정값이 아니라 supply-chain gap이다.

**release decision ledger를 append-only로 운영한다**

promotion, rejection, exception, revocation, rollback와 re-promotion은 DecisionEvent다. event는 subject/evidence index, PolicyRevision, actor/workload identity, result·reason, timestamp와 parent decision을 가진다. 과거 row를 수정하지 않고 successor로 상태를 바꾼다. 당시와 현재 판단을 모두 재연한다.

approval request가 review 중 evidence를 추가하면 evidence index digest가 바뀐다. 이전 서명이 새 package에 자동 적용되지 않게 한다. 여러 reviewer가 서로 다른 bytes를 보지 않도록 exact subject/index를 표시한다. web UI checkbox만 approval evidence로 쓰지 않는다.

exception은 subject, channel, risk, compensating control, owner와 expiry를 가진 decision이다. expiry 후 자동 promotion 지속을 막고 alert·quarantine policy를 둔다. broad name/tag나 future version을 허용하지 않는다. exception을 소비한 deployments와 descendants를 query한다.

rollback event는 target current verification, fleet convergence와 canary evidence를 포함한다. revocation 뒤 clean rebuild는 새 subject·provenance와 evaluation을 가진 re-promotion이다. 같은 semantic version을 재사용하지 않고 generation을 올린다. running actual digest와 ledger를 reconcile한다.

감사자는 임의 BundleID의 decision timeline과 당시 policy/trust snapshot을 재생한다. 누가 왜 허용했고 어떤 evidence로 철회·복구했는지 답할 수 있어야 한다. 이 ledger가 산출물 DAG와 결합될 때 공급망의 기술적 사실과 조직의 권한 결정이 끊기지 않는다.

**새 artifact를 받아들이는 마지막 질문 순서**

첫째, 요청한 이름이 어느 immutable source·data·base commit과 content digest로 resolve됐는가. 둘째, exact file set과 schema는 무엇이며 unexpected file·key가 없는가. 셋째, builder는 선언된 material만 사용했고 invocation·environment를 attestation에 남겼는가. 넷째, signature subject·identity·inclusion과 channel authorization이 모두 맞는가.

다섯째, safetensors·pickle·remote code의 load 능력과 resource limit은 무엇인가. 여섯째, build SBOM과 actual loaded native/JIT inventory가 맞는가. 일곱째, tokenizer/template, checkpoint·optimizer, adapter/merge/quantization와 서빙 실행 환경의 semantic parent가 닫혔는가. 여덟째, golden numerical·behavior oracle을 어떤 hardware·topology에서 실제 실행했는가.

아홉째, data/model cards의 license·consent·limitation과 machine manifest가 일치하는가. 열째, ancestor나 signer가 철회되면 graph가 모든 checkpoint·export·deployment를 찾는가. 열한째, current policy를 통과한 rollback과 clean rebuild 절차가 있는가. 열두째, empty cache·network-restricted clean room에서 independent verifier가 같은 판정을 내는가.

각 답은 문장이 아니라 digest, SourceCard, DecisionRecord, fixture와 certificate를 가리켜야 한다. `latest`, package version, 성공한 import와 signature-valid만으로 답하지 않는다. unknown·NotExecuted와 exception을 드러낸다. 이 순서로 새 framework, model format과 accelerator가 등장해도 무엇을 신뢰하고 무엇을 직접 검증해야 하는지 빠르게 가려낸다.

release 후에도 질문은 끝나지 않는다. runtime actual identity, revocation freshness, vulnerability/license update, evidence readability와 reproducibility drill을 감시한다. 새 incident와 삭제 요청은 DAG·decision ledger·golden suite를 갱신한다. 신뢰는 한 번의 승인 상태가 아니라 계속 검증되고 필요할 때 안전하게 철회되는 상태다.

최종 운영 지표에는 미검증 parent, stale signature·policy, unknown runtime component, expired exception, revocation fleet lag, deletion pending과 cold-reproduction failure가 포함된다. 단순 artifact 수보다 위험한 열린 edge의 age와 channel을 본다. 경보는 exact BundleID, owner, 안전한 격리·rollback과 재승인 acceptance를 제공해야 한다. 이 지표까지 artifact graph와 연결될 때 감사 결과가 실제 운영 행동으로 이어진다.
## 27.15 Ray와 MLflow 자체도 재현 대상에 넣는다

`ray==x`, `mlflow==y`만 기록해서는 실행을 복원할 수 없다. Ray Train의 v1/v2 API 경계, deprecated checkpoint 옵션, report upload mode와 controller 실패 정책은 revision에 따라 달라진다. MLflow의 active-run 해석, asynchronous logging 기본값과 autolog 지원 범위도 바뀐다. 그래서 release manifest에는 package version과 함께 source commit, Python lockfile, container digest, 활성 feature flag·환경 변수와 실제로 소비된 resolved config를 넣는다.

특히 환경 변수는 숨은 입력이다. MLflow는 `MLFLOW_RUN_ID`로 resume 대상을, `MLFLOW_ENABLE_ASYNC_LOGGING`으로 동기성을, system-metric 환경 변수로 monitor 시작 여부를 바꿀 수 있다. Ray는 runtime environment와 shared storage URI가 worker와 controller의 관측 가능한 파일계를 바꾼다. secret 값 자체를 기록할 필요는 없지만 변수 존재 여부, 비밀 참조 ID와 의미를 바꾸는 non-secret resolved value는 manifest에 남긴다.

재현성 검사는 API 호출 모양이 아니라 소비 지점까지 내려간다. Ray `RunConfig.storage_path`가 실제 공유 filesystem으로 resolve됐는지, `FailureConfig`의 preemption·ordinary failure budget이 무엇이었는지, MLflow `log_batch`가 synchronous로 끝났는지 확인한다. 원본 source span과 공식 실패 테스트를 함께 붙이면 문서 설명이 다음 버전에서 stale해졌을 때 diff가 드러난다.

## 27.16 서명된 artifact를 실제 training run identity까지 닫는다

서명, SBOM, container digest, secret과 RBAC를 각각 확인해도 실행 신원은 닫히지 않을 수 있다. 검증한 image digest와 kubelet이 실제로 시작한 digest가 다르거나, 올바른 image가 과도한 ServiceAccount로 다른 checkpoint를 읽으면 서명은 참이지만 run claim은 거짓이다. 폐루프는 `SourceCommit → BuildInvocation → ImageDigest → AttestationSubject → VerifiedIdentity → AdmissionDecision → PodUID → RuntimeImageID → ServiceAccount → SecretReferenceVersion → Dataset/CheckpointDigest → RunID`를 모두 연결해야 한다.

Cosign의 테스트가 경계를 구체화한다. `cmd/cosign/cli/attest/attest_blob_test.go:218-272`는 blob SHA-256을 계산하고 DSSE payload를 풀어 in-toto statement의 단일 subject digest와 predicate type을 확인한다. 즉 signature blob의 암호학적 성공만으로 subject가 기대 artifact라고 말하지 않는다. 같은 저장소의 `cmd/cosign/cli/attach.go:79-96`은 SBOM attach가 서명하지 않는다고 명시하고 attest 사용을 안내한다. “SBOM이 붙었다”와 “그 SBOM statement가 기대 subject에 대해 허용된 identity로 서명됐다”는 별도 주장이다.

KubeRay chart도 name과 runtime identity의 틈을 보인다. `ray-cluster/templates/raycluster-cluster.yaml:76-80`, `:209-213`은 repository와 tag로 head·worker image를 렌더링하며, `:161-162`, `:294-295`는 설정된 ServiceAccount 이름을 pod spec에 넣는다. 이 좌표는 chart render 사실만 증명한다. registry tag가 어느 digest로 resolve됐는지, admission 뒤 mutation이 있었는지, node runtime이 어느 image ID를 시작했는지는 별도 관측이 필요하다. 사실과 추론을 섞지 않는다.

**승인 전 체크리스트.** build material과 builder identity, attestation subject digest, signature trust root·policy revision, SBOM digest·format·생성기, vulnerability DB 시점, OCI manifest와 platform digest를 고정한다. admission에서는 tag를 digest로 resolve하거나 digest reference를 요구하고, verified subject와 admitted image를 비교한다. RBAC에는 실제 verb·resource·namespace를 저장하고 wildcard를 차단한다. secret 값은 로그에 쓰지 않되 secret object UID·resource version 또는 외부 secret version, mount 대상과 읽은 workload identity를 남긴다.

**실행 중 체크리스트.** PodUID, node, container runtime image ID, init/sidecar까지의 digest, ServiceAccount UID, token audience·expiry, effective environment key 목록, volume/source digest와 network policy generation을 RunID에 묶는다. checkpoint와 dataset은 URI가 아니라 소비 시점의 content digest로 기록한다. CUDA driver, runtime, NCCL·native extension과 JIT cache digest도 SBOM의 기대 목록과 runtime inventory로 나란히 둔다. secret 원문, 장기 bearer token과 개인 식별자는 provenance가 아니라 보호 대상이다.

**failure injection과 first divergence.** (1) 같은 tag가 다른 digest를 가리키게 하면 registry resolution에서, (2) attestation subject 한 nibble을 바꾸면 subject comparison에서, (3) 서명되지 않은 SBOM만 attach하면 evidence-policy에서, (4) worker ServiceAccount를 privileged account로 바꾸면 rendered pod/RBAC diff에서, (5) secret version을 회전하고 stale pod를 두면 consumed secret version에서, (6) sidecar image만 바꾸면 runtime inventory에서, (7) checkpoint URI 내용을 바꾸면 input digest에서 실패해야 한다. 마지막 loss가 같더라도 더 앞선 identity invariant가 깨지면 동일 run이 아니다.

promotion gate는 이 graph를 입력으로 받아 허용 또는 거부와 정확한 first-divergence edge를 낸다. verifier 오류와 policy 거부를 구분하고, network failure를 signature invalid로 바꾸지 않는다. waiver에는 빠진 edge, 위험 소유자, 범위, 만료와 철회 조건을 넣으며 영구 성공으로 캐시하지 않는다. 법률·라이선스 적합성은 기술 manifest만으로 확정하지 않는다. manifest는 검토할 사실과 계보를 제공하고, 법적 판단은 관할·계약·데이터 조건을 아는 담당자가 별도로 남긴다.

검토를 닫을 때에는 manifest의 첫 노드와 실제 실행의 마지막 노드를 서로 대조한다. 승인된 source·data·image digest에서 출발해 RunID와 checkpoint까지 정방향으로 걷고, 실행 중인 Pod의 image ID와 loaded checkpoint에서 다시 그 부모들까지 역방향으로 걷는다. 어느 방향에서든 tag·경로·사람의 기억으로만 건너뛰는 edge가 있으면 promotion을 보류한다. 그 빈칸을 채우는 resolver·attestation·runtime inventory가 이 장의 checklist가 요구하는 실제 조치다.
## 27.17 데이터 권리도 revisioned event graph로 재현한다

소스 commit과 container digest를 고정하면서 데이터 권리를 카드의 `license` 문자열로만 남기면 공급망은 절반만 닫힌다. 권리 판단은 원천별 주장, 수집 시점의 정책 관측, 조직의 허용 결정, 철회 요청과 파생물 적용 결과로 시간에 따라 변한다. 따라서 과거 문자열을 덮어쓰지 않고 `Assertion → Observation → Decision → Application` 이벤트를 append-only로 연결한다.

각 event에는 subject의 안정 ID, policy revision, 관측 bytes digest, actor·workload identity, 시각, 결과와 이전 event를 둔다. 데이터 혼합과 dedup은 parent 집합을 잃지 않아야 한다. 행 단위 추적이 너무 비싸면 shard manifest에 parent ID set의 content-addressed index를 두되, 실제 tombstone fixture로 false negative가 없음을 검증한다. Bloom filter만으로 삭제 완료를 증명해서는 안 된다. false positive는 과잉 삭제를 만들고 normalization 변화는 누락을 만들 수 있기 때문이다.

출시 관문는 `license-present`가 아니라 열린 권리 edge를 검사한다. robots 관측 없음, consent 범위 불명, request 인증 미완료, tombstone이 token shard까지 도달하지 않음, checkpoint 영향이 제거되지 않음은 서로 다른 상태다. 각각 owner와 만료·격리 정책을 갖게 해야 운영자가 “미상”을 “허용”으로 오독하지 않는다.

기존 참고문헌을 역으로 연결할 때도 같은 원칙을 따른다. 정확한 arXiv ID 목록에 수집 동의·robots·삭제 계보를 직접 다루는 논문이 없다면 억지로 인용을 붙이지 않는다. 일반적인 표현학습이나 최적화 논문을 주제가 가깝다는 이유만으로 연결하면 참고문헌 수는 늘지만 주장의 근거는 오히려 약해진다. 논문 ID와 본문의 실제 명제 사이에 전제 관계가 확인될 때만 인용하고, 그렇지 않으면 “직접 대응 논문을 확인하지 못함”이라는 조사 결과를 남긴다.

## 27.18 삭제 완료를 verification ledger의 양방향 탐색으로 증명한다

삭제 ledger는 요청 접수, 대상 식별, tombstone 적용, 새 generation 발행, downstream 차단, 모델 영향 판정을 서로 다른 상태로 보존한다. 각 전이는 actor, policy revision, 입력·출력 digest와 이전 event를 가진다. 인증 실패·범위 불명·법적 검토 대기·rewrite 실패를 성공으로 덮어쓰지 않는다.

검증은 양방향이다. RequestID에서 source·dedup cluster·packed token·shard·UpdateID·checkpoint까지 정방향으로 걷고, release 후보의 shard와 run manifest에서 parent source까지 역방향으로 걷는다. 두 탐색의 집합 차이가 0이어야 corpus 삭제가 닫힌다. Bloom filter는 후보 검색에는 쓸 수 있지만 삭제 완료의 단독 oracle은 아니다.

tombstone join, all-derivative search, immutable generation, downstream consumer의 구 세대 거부를 실제 assertion한 시험만 `testedBy`다. filter·token writer·reshard 시험은 component 경계를 설명하지만 종단 삭제 시험은 아니다. 공개 구현에서 해당 시험을 찾지 못했으므로 합성 fixture와 `NeedsReview`를 유지한다.

기술 ledger는 법률 의견서가 아니다. license와 robots snapshot을 보존해도 consent와 관할별 적합성이 자동 결정되지 않으며, 법적 승인만으로 파생 shard가 정리되지도 않는다. release는 법적 판단과 기술 집행 event를 모두 요구하고, 이미 학습한 parameter 영향은 재학습·unlearning 검증 없이 제거 완료로 승격하지 않는다.

## 27.19 서명 검증과 파일 파싱을 학습 admission의 한 경로로 묶는다

학습 공급망은 “서명이 맞다”에서 끝나지 않는다. 먼저 승인 manifest가 checkpoint·config·tokenizer·dataset shard·container·dependency lock의 digest를 하나의 RunID에 묶는다. verifier는 in-toto statement의 subject SHA-256이 지금 읽으려는 산출물 digest와 같은지 확인하고, transparency-log inclusion proof의 필수 필드·body type·trusted key를 검증한다. 그 뒤에야 파일 parser를 호출한다. 순서를 뒤집으면 신뢰하지 않은 거대 header나 잘못된 offset을 signature 결과보다 먼저 처리하게 된다.

safetensors의 고정 구현은 이 두 번째 경계를 구체적으로 보여 준다. 8-byte header length를 읽은 뒤 최대 크기와 정수 덧셈 overflow를 검사하고, 범위 안의 bytes만 UTF-8과 JSON으로 해석한다. metadata validation은 tensor offset이 앞 tensor의 끝에서 정확히 이어지는지, shape 원소 수와 dtype bit 수의 곱이 overflow하지 않는지, 선언 byte 수와 실제 slice가 같은지 검사한다. canonical negative test는 oversized·non-UTF8 header, trailing polyglot bytes, truncated payload, overlapping offsets와 거대 shape를 각각 거부한다. “안전한 포맷”이라는 형용사가 아니라 어느 분기에서 어떤 입력이 실패하는지가 admission 계약이다.

운영 순서는 `resolve digest → verify signature/identity → compare provenance subject → verify transparency evidence → parse bounded header → validate tensor layout → compare config/tokenizer/dataset digests → stage → train`이다. 각 단계는 독립된 failure code와 관측값을 남긴다. digest mismatch를 parser error로, registry timeout을 invalid signature로 합치면 최초 차이를 잃는다. dependency lock도 resolver 입력일 뿐이다. compiler, CUDA·driver, wheel ABI와 실제 loaded shared library까지 environment manifest로 관측하지 않으면 같은 lock에서 다른 실행이 생길 수 있다.

서명은 bytes의 출처와 변경 여부를 말하지만 내용의 진실성을 말하지 않는다. 서명된 dataset도 오염·권리 침해·중복을 포함할 수 있고, 서명된 checkpoint도 backdoor나 성능 퇴행을 품을 수 있다. SBOM signature 역시 제시된 SBOM이 바뀌지 않았음을 보일 뿐 누락 dependency가 없음을 증명하지 않는다. 따라서 content-quality, license, red-team, behavioral eval과 SBOM completeness oracle은 별도 gate로 둔다. 이 구분이 없으면 암호학적 성공을 의미적 승인으로 오독한다.

failure injection은 subject digest 한 nibble, null subject, 빠진 inclusion-proof field, non-string log body, header length overflow, invalid UTF-8, tensor offset overlap, truncated payload를 한 번에 하나씩 바꾼다. 기대 결과는 모두 “학습 시작 전 거부”지만 최초 실패 단계와 오류 유형은 달라야 한다. 실제 registry signing ceremony와 GPU run을 실행하지 않은 정적·canonical-test 근거는 runtime 검증으로 승격하지 않는다. production gate는 동일 mutation corpus를 실제 admission controller와 worker startup path에 통과시켜 process가 model bytes를 map하기 전에 멈추는지 확인해야 닫힌다.
