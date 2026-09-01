# 7장. 기억은 답변의 배경이 아니라 쓰기 가능한 데이터다

> 선수 지식: [4장](./04-context-assembly.md)의 prompt 입장 조건과 [6장](./06-context-compaction.md)의 세대 전환. 이 장에서는 저장·검색뿐 아니라 만료·삭제·파생 인덱스 회수까지 한 수명 주기로 다룬다.

한 에이전트가 어제의 장애를 기억한다는 말은 편리하지만 위험하다. 그 문장은 적어도 여섯 질문을 감춘다. 누가 그 기억을 썼는가, 어느 테넌트의 것인가, 언제까지 유효한가, 원래 관찰인가 아니면 모델의 요약인가, 프롬프트에 들어갈 자격은 누가 판정하는가, 그리고 삭제를 요청하면 무엇이 실제로 사라지는가. 이 질문에 답하지 못하면 장기 기억은 지능의 확장이 아니라 오래된 권한과 오염된 문장을 재유통하는 장치가 된다.

이 장의 관점은 단순하다. 기억을 문맥 창 바깥에 둔 텍스트 저장소로 보지 말고, **불완전하고 만료 가능하며 출처와 접근 범위를 가진 관찰 레코드**로 보자. 모델은 그 레코드를 읽을 수 있을 뿐, 진실성이나 권한을 부여하지 않는다.

## 7.1 첫 실패: 맞는 문장이 다른 사람에게 돌아온다

운영 지원 에이전트가 A사의 지난 장애 보고서에서 “인증서를 수동으로 갱신했다”는 문장을 찾았다. 질문은 B사의 인증서 오류였다. 의미 검색은 두 문장을 매우 가깝다고 판단했고, 모델은 그 절차를 답변에 넣었다. 기술적으로는 자연스러운 답이다. 그러나 A사의 내부 경로와 운영 습관이 B사에게 노출되었고, B사의 자동 갱신 정책과도 충돌한다.

임베딩 품질만 살펴서는 부족하다. 검색 결과가 프롬프트 후보가 되기 전에 **tenant, access scope, freshness, trust, tombstone**을 모두 통과했는지 확인해야 한다. 순서를 바꾸어 “먼저 top-k를 뽑고 나중에 애플리케이션에서 거르면 되지 않나?”라고 생각하기 쉽다. 하지만 후보 자체가 로그·메트릭·캐시에 남거나, 후처리 버그로 모델에 노출될 수 있다. 특히 권한 없는 결과가 상위 k를 차지해 허용된 결과를 밀어내면 post-filter는 recall도 잃는다. 권한은 점수 조정이 아니라 입장 조건이다.

```mermaid
flowchart LR
  Q[질문과 RunID] --> A[권한·tenant 범위 고정]
  A --> R[후보 검색]
  R --> G{prompt 입장 게이트}
  G -->|tenant/scope/freshness/trust/tombstone 통과| P[문맥 조립]
  G -->|하나라도 불통과| X[제외 사유 기록]
  P --> M[모델 proposal]
  W[새 관찰 또는 요약] --> V[검증·분류] --> S[기억 저장]
  D[삭제 요청] --> T[tombstone] --> U[파생 인덱스 삭제 확인]
```

## 7.2 기억 레코드의 최소 계약

`memory = text`라는 설계는 곧 막힌다. 최소한 다음 필드는 서로 다른 질문에 답한다.

| 필드 | 답하는 질문 | 빠졌을 때 생기는 거짓말 |
|---|---|---|
| `memory_id` | 정확히 어느 사실인가 | 같은 문장의 수정·삭제를 구분 못 한다 |
| `tenant_id`, `subject_scope` | 누가 읽을 수 있는가 | 유사도를 권한으로 오독한다 |
| `source_ref`, `observed_at` | 어디서 언제 관찰했는가 | 요약을 원전처럼 인용한다 |
| `valid_until`, `max_age` | 아직 쓸 수 있는가 | 과거의 정책을 현재 규칙으로 쓴다 |
| `trust_state` | prompt에 넣어도 되는가 | 미검증·오염 후보가 조용히 승격된다 |
| `tombstoned_at`, `deletion_state` | 삭제가 어느 단계인가 | 논리 삭제를 완전 삭제라 부른다 |
| `derivative_refs` | 어떤 embedding·요약이 파생됐는가 | 원문만 지우고 vector hit를 남긴다 |

이 표는 모든 저장소가 이 스키마를 제공한다는 주장이 아니다. 실제 제품별 API는 다르다. 다만 이 구분 없이는 결함을 관찰할 방법도 없다는 뜻이다. memory retrieval의 결과에는 텍스트뿐 아니라 왜 선택되었는지, 어떤 gate가 통과했는지, 어느 revision에서 보았는지를 내부 원장에 남겨야 한다. 사용자에게 모두 노출할 필요는 없지만, 나중에 “왜 이 문장이 모델에 들어갔나?”라는 질문에는 답할 수 있어야 한다.

## 7.3 코드에서 확인할 수 있는 좁은 사실

Jikji의 고정 공개 리비전에서 memory recall 경계는 `tenant_id`와 `MaxAge`를 포함한 요청을 storage 쪽으로 넘긴다. 이 짧은 경계는 중요하다. 호출자가 테넌트와 시간 제한을 전달하지 않는 구조보다 훨씬 낫다. [Jikji memory recall](https://github.com/jikji-labs/jikji/blob/d0cb4997e1882f9f5fc28b0b601ddf97317baf43/jikji/internal/compositor/memory_recall.go#L38-L139)

하지만 여기서 멈춰야 한다. 이 함수만으로 DB 쿼리가 `WHERE tenant_id = ?`를 실제 수행한다거나, 벡터 인덱스가 삭제를 전파한다거나, 백업과 replica에서 모든 사본을 지운다는 사실은 증명되지 않는다. 공개 테스트는 success, empty tenant, nil store 같은 호출 경계를 다룬다. [Jikji memory recall tests](https://github.com/jikji-labs/jikji/blob/d0cb4997e1882f9f5fc28b0b601ddf97317baf43/jikji/internal/compositor/memory_recall_test.go#L34-L153) 이 차이를 지키는 것이 코드 읽기의 출발점이다. **인자를 전달했다**와 **그 정책이 저장소 전체에서 강제된다**는 다른 주장이다.

실행기나 모델이 만든 요약도 자동으로 신뢰하면 안 된다. Jikji의 trajectory memory와 oracle poisoning 코드는 기억 후보와 오염 문제를 별도 관심사로 드러낸다. [trajectory memory](https://github.com/jikji-labs/jikji/blob/d0cb4997e1882f9f5fc28b0b601ddf97317baf43/jikji/internal/research/trajectory_memory.go#L122-L160) · [oracle poisoning](https://github.com/jikji-labs/jikji/blob/d0cb4997e1882f9f5fc28b0b601ddf97317baf43/jikji/internal/research/oracle_poisoning.go#L14-L125)

이것이 production-grade provenance·삭제를 보장한다는 뜻은 아니지만, ‘검색된 텍스트’와 ‘신뢰할 만한 기억’을 같은 상태로 두지 말아야 한다는 설계 신호다.

## 7.4 lifecycle은 저장과 검색 사이에 있다

기억의 생애는 보통 다음 상태를 가진다.

```mermaid
stateDiagram-v2
  [*] --> proposed: 관찰/요약 후보
  proposed --> admitted: 출처·범위 검증
  proposed --> quarantined: 오염·모호성
  admitted --> recalled: scope+freshness 통과
  recalled --> superseded: 더 새 revision
  admitted --> tombstoned: 삭제 또는 보존정책 만료
  quarantined --> tombstoned: 삭제 승인
  tombstoned --> purge_pending: 파생물 추적
  purge_pending --> physically_verified: 범위 내 삭제 영수증
```

`recalled`는 진실 판정이 아니다. 이번 prompt assembly에 들어갈 자격을 얻었다는 사건일 뿐이다. `admitted`도 영구 신뢰가 아니다. source가 철회되거나 policy가 바뀌면 quarantine·superseded로 전이할 수 있다. `physically_verified` 역시 저장소 범위를 반드시 붙여야 한다. primary SQLite table과 embedding table을 비웠다는 검증을 backup, WAL, export, 외부 vector store, 제공자 transcript의 완전 삭제로 확대하면 안 된다.

상태를 이렇게 나누면 “모델이 그 사실을 기억한다”는 문장을 더 정확히 읽을 수 있다. 모델 가중치가 바뀐 것이 아니라, 이번 step context에 특정 `memory_id`의 특정 revision이 들어갔다는 의미다. 그리고 그 입력은 model output의 정확성을 보증하지 않는다. 검색된 기억은 증거 후보이며, 고위험 의사결정에는 원전·현재 정책·별도 검증이 필요하다.

## 7.5 삭제: tombstone은 끝이 아니라 시작이다

삭제 요청은 보통 세 층으로 나뉜다.

| 단계 | 목적 | 필요한 증거 | 아직 보장하지 않는 것 |
|---|---|---|---|
| 논리 삭제 | 즉시 prompt 재노출 차단 | read gate가 tombstone을 거절 | 물리 사본 제거 |
| 파생물 정리 | embedding·요약·캐시를 식별 | `derivative_refs`와 delete job | 백업·외부 index 삭제 |
| 물리 삭제 검증 | 범위 내 row/blob 부재 확인 | 저장소별 receipt | 전 세계적 완전 삭제 |

삭제 전 prompt 후보가 이미 조립되어 있을 수 있다. 그러므로 `tombstoned_at`은 미래 검색을 막지만, 이미 시작한 step의 입력을 마술처럼 지우지 않는다. 고위험 action은 effect gate 직전에 다시 현재 revision을 검사해야 한다. 메모리 삭제와 외부 효과의 재검증은 같은 종류의 경쟁 조건이다.

또한 삭제 재시도는 idempotent해야 한다. `forget(memory_id)`가 두 번 도착해도 두 번째 호출은 이미 tombstoned/purged라는 receipt를 돌려야지, 예외를 던져 운영자를 오도해서는 안 된다. 반대로 다른 tenant의 동명 ID를 지우지 않도록 `tenant_id`와 deletion authority를 action digest에 묶어야 한다.

## 7.6 실습: 작은 SQLite gate로 큰 질문을 드러내기

아래 코드는 제품 구현 대신 검증 가능한 최소 gate만 담았다. 벡터 거리보다 policy predicate를 먼저 평가한다.

```sql
-- 실행 템플릿: memories schema와 named parameter binding은 애플리케이션이 제공한다.
SELECT memory_id, text, source_ref
FROM memories
WHERE tenant_id = :tenant
  AND tombstoned_at IS NULL
  AND trust_state = 'admitted'
  AND valid_until > :now
  AND created_at >= :min_created_at;
```

실습 데이터에는 다섯 row를 넣는다. (1) A tenant의 신선하고 admitted인 row, (2) B tenant의 완전 일치 row, (3) A tenant지만 만료된 row, (4) admitted였다가 quarantined가 된 row, (5) tombstoned row다. query 결과는 첫 row 하나여야 한다. 이어 첫 row를 forget한 뒤 같은 query가 빈 결과인지, primary row와 embedding row가 각각 어떤 상태인지 기록한다.

여기서 중요한 oracle은 “검색 결과가 그럴듯하다”가 아니다. `tenant B hit count = 0`, `expired hit count = 0`, `tombstoned prompt eligibility = false`, `primary delete receipt`, `derivative delete receipt`처럼 반증 가능한 값이다. 벡터 DB를 붙이기 전에도 이 계약을 테스트할 수 있다.

## 7.7 고장 주입

| 주입 | 기대 oracle | 금지된 결론 |
|---|---|---|
| tenant filter를 post-filter로 이동 | 허용 row가 top-k에서 사라지는 사례 | 후처리가 항상 안전하다 |
| `valid_until` 직후 검색 | 후보가 제외됨 | 저장소가 시간을 자동으로 정정한다 |
| poison verdict 뒤 동일 query | quarantine row가 prompt에 없음 | quarantine이 모델의 믿음을 삭제한다 |
| tombstone 뒤 crash | 새 read는 차단, purge는 pending/unknown | 물리 삭제가 끝났다 |
| embedding delete timeout | primary와 derivative 상태가 분리 기록 | 원문 삭제가 retrieval 삭제다 |
| source revision 철회 | 기존 memory가 superseded/quarantined | 임베딩 유사도가 신뢰를 회복한다 |

## 7.8 운영 체크리스트

- [ ] 모든 recall 요청이 tenant·principal·scope·time을 가진다.
- [ ] 후보 선택과 prompt admission의 경계가 로그에서 분리된다.
- [ ] memory source와 summary/embedding의 파생 관계를 추적한다.
- [ ] tombstone, purge pending, verified 범위를 같은 ‘deleted’로 뭉개지 않는다.
- [ ] 오래된 승인·정책·대상 revision을 effect 직전에 다시 확인한다.
- [ ] 권한 없는 후보의 존재 자체가 metrics label이나 debug dump에 새지 않게 한다.
- [ ] 평가 데이터는 cross-tenant near duplicate, stale truth, poisoned trusted-looking text를 포함한다.

## 7.9 이 장이 보장하지 않는 것

이 장의 lifecycle은 모델 파라미터 속 지식을 삭제하는 방법이 아니다. 여기서 말하는 기억은 AgentRun이 불러오는 외부 데이터다. 또한 tenant filter가 있다고 해서 모든 remote store·cache·backup의 격리를 증명하지 않는다. 특히 ANN 검색의 근사성, 분산 index의 deletion lag, 법적 보존 의무, 사람이 올린 문서의 진실성은 별도 설계와 별도 증거를 요구한다.

기억을 잘 다루는 에이전트는 많이 기억하는 에이전트가 아니다. 무엇을 아직 모르는지, 무엇을 더 이상 보여주면 안 되는지, 무엇이 이미 삭제되었는지와 무엇이 삭제 작업 중인지까지 말할 수 있는 에이전트다.

## 7.10 기억 평가를 검색 점수 밖으로 넓히기

memory 시스템은 answer accuracy만으로 평가하면 위험한 실패를 놓친다. 아래처럼 relevance와 governance를 같은 표에 두되, 하나의 점수로 합쳐 숨기지 않는다.

| 축 | 질문 | 예시 oracle |
|---|---|---|
| relevance | 필요한 관찰이 후보에 왔는가 | source-aware recall |
| authorization | 보이면 안 될 row가 0개인가 | cross-tenant exposure=0 |
| freshness | 만료 사실이 배제됐는가 | stale admission=0 |
| provenance | 원전과 요약을 구별하는가 | source_ref coverage |
| deletion | tombstone/purge 상태가 정직한가 | derivative receipt coverage |
| poisoning | 신뢰 흉내 row가 quarantine되는가 | adversarial recall pass rate |

숫자 하나를 최적화하면 부작용이 생긴다. recall만 올리면 허용되지 않은 후보가 늘 수 있고, 엄격한 filter만 올리면 필요한 자료가 모두 사라져 모델이 추측하게 된다. 그래서 query가 빈 결과일 때도 ‘모른다’ 또는 ‘현재 권한 범위에 증거가 없다’를 유효한 outcome으로 취급한다. 빈 결과를 피하려고 다른 tenant나 오래된 문서를 자동으로 보충하는 것은 친절함이 아니라 정책 위반이다.

## 7.11 기억과 대화 요약의 결합

대화 축약은 memory write처럼 보이지만 lifecycle이 다르다. 축약은 주로 한 run/thread의 context budget을 관리하며, 장기 memory는 후속 run의 retrieval 대상이 된다. 축약본을 장기 memory로 승격하려면 source span, scope, expiry, trust 검사가 다시 필요하다. ‘assistant가 앞에서 말했다’는 것은 원전이 아니다.

실무에서는 summary에 unresolved approval, in-flight logical call, current policy revision처럼 control 사실을 문장으로만 넣지 말고 structured ledger에서 다시 조립한다. 자연어 요약이 ‘배포 승인을 받았다’고 말해도 receipt가 만료됐으면 실행기에는 아무 권한이 없다. 이 원칙은 context compaction과 memory recall 모두에 적용된다.

마지막으로 memory write 자체도 도구 호출처럼 취급한다. 누가 어떤 source에서 어떤 claim을 추출해 어느 trust state로 넣었는지 write receipt가 있어야 한다. 모델이 만든 좋은 요약은 candidate가 될 수 있지만, source 없는 영구 기억이 되어서는 안 된다.

### 기억 장애를 만났을 때

운영자가 “에이전트가 예전 일을 잘못 기억했다”고 말하면 먼저 문장 품질을 보지 않는다. memory ID, source revision, admission gate, prompt generation, answer에 실제 인용된 observation을 차례로 따라간다. 이 경로가 없으면 모델이 틀린 것과 잘못된 기억이 들어간 것을 분리할 수 없다. 기억 문제의 해결은 종종 더 큰 context window가 아니라 더 짧고 검증 가능한 provenance chain이다.

그리고 그 chain을 찾을 수 없다면, 기억을 답에 사용하지 않는 것이 더 안전하다.

이 원칙은 recall quality가 낮다는 보고보다 더 구체적이다. 시스템은 어떤 memory가 빠졌는지뿐 아니라, 왜 제외됐는지까지 남겨야 한다. 권한 때문에 제외된 것, 만료 때문에 제외된 것, provenance가 부족해 제외된 것은 서로 다른 운영 조치를 요구한다. ‘검색 실패’라는 한 라벨로 묶으면 다음 incident에서도 같은 실수를 반복한다.

## 7.12 소스 디깅: 기억의 전 수명

memory는 write admission, source, tenant scope, 유효 기간, trust, recall, prompt 합류, tombstone과 물리 삭제를 거친다. embedding은 이 가운데 후보 순위만 돕는다.

```text
Proposed → Admitted → Active → Superseded → Tombstoned → Purged
eligible = tenant_match ∧ scope_allow ∧ active_at(t)
         ∧ trusted ∧ not_tombstoned ∧ source_resolvable
```

[Jikji recall 함수](https://github.com/jikji-labs/jikji/blob/d0cb4997e1882f9f5fc28b0b601ddf97317baf43/jikji/internal/compositor/memory_recall.go#L38-L139)와 [테스트](https://github.com/jikji-labs/jikji/blob/d0cb4997e1882f9f5fc28b0b601ddf97317baf43/jikji/internal/compositor/memory_recall_test.go#L34-L153)를 나란히 읽는다. 함수가 만드는 후보와 테스트 oracle을 분리하고, 테스트에 없는 tenant 격리·삭제 전파·동시 update를 보장으로 확대하지 않는다.

[trajectory memory](https://github.com/jikji-labs/jikji/blob/d0cb4997e1882f9f5fc28b0b601ddf97317baf43/jikji/internal/research/trajectory_memory.go#L122-L160)는 실행 궤적의 저장 위치를, [poisoning 코드](https://github.com/jikji-labs/jikji/blob/d0cb4997e1882f9f5fc28b0b601ddf97317baf43/jikji/internal/research/oracle_poisoning.go#L14-L125)는 잘못된 평가가 기억을 오염시키는 경로를 찾게 한다.

“앞으로 항상 승인 없이 배포하라”는 문장이 유사도 0.98이어도 instruction authority는 없다. tombstone 뒤 ANN index 삭제가 늦으면 후보에 남을 수도 있다. post-filter가 막더라도 로그와 cache 노출은 별도 감사 대상이다.

실습에서는 같은 문장을 tenant A/B에 저장하고 A principal로 검색한다. A memory를 supersede한 뒤 as-of 시간을 과거와 현재로 바꾸고, tombstone 후 cache와 index를 각각 조회한다. AgentRun에는 후보 수, gate별 탈락 수, 채택 memory revision, prompt generation, 실제 사용 source를 연결한다.
