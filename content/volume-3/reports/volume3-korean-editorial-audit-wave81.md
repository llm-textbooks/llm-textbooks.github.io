# Volume 3 한국어 구조 편집 감사 — Wave81

## 범위와 판정

- 대상: `book.yaml`의 현행 spine 43개 문서(10부).
- 방법: `tools/im-not-ai`의 의미 보존 원칙으로 산문만 검토했다. 기술 주장, 수치, 고유명사, 인라인 식별자, 링크, 표, 코드 블록, Mermaid는 바꾸지 않았다.
- 판정: 기계적으로 이어진 종결은 찾지 못했다. 길이 기준을 넘은 근거-나열 문단 31개를 20개 문서에서 논점 단위로 나눴다. 문장을 다시 쓰지 않아 근거와 주장 범위는 그대로다.
- 보류가 아니라 수용한 항목: `~할 수 있다` 91개는 실제 가능성·조건부 동작을 뜻하고, `runtime`·`source`·`artifact`·`gate` 등은 이 책에서 계약·식별자·고정 기술 결합어로 쓰인다. 자동 탐지만으로 번역투로 단정할 수 없어 수정하지 않았다.

## 장별 기록

| 문서 | 처리와 수용한 비쟁점 |
|---|---|
| `00-preface.md` | 변경 없음 — 도입의 문단 호흡과 마무리 정상. |
| `00-reading-routes.md` | 변경 없음 — 경로 안내의 목록·표는 산문 감사 대상에서 제외. |
| `01-agent-run.md` | 근거와 한계 설명을 두 논점으로 분리(2곳). 기술 용어와 receipt 주장 보존. |
| `02-react-state-machine.md` | router/registry, postflight/cancellation 근거를 각각 분리(2곳). |
| `03-framework-boundaries.md` | Codex·pi-agent 근거와 비교 한계를 논점별로 분리(3곳). |
| `04-context-assembly.md` | `run_turn`과 `built_tools` 읽기 지침을 분리(1곳). |
| `05-tokenizer-tool-schema.md` | template/tokenizer와 tool schema, 이어지는 읽기 지침을 분리(2곳). `runtime`·`artifact`은 기술 용어로 수용. |
| `06-context-compaction.md` | policy·compatibility·dispatch와 코드 읽기 지침을 분리(2곳). `source`는 증거 범주 용어로 수용. |
| `07-memory-lifecycle.md` | 기억 후보/오염과 비보장, recall/test와 후속 근거를 분리(2곳). `gate`·`source`는 계약 용어로 수용. |
| `08-model-request-retry.md` | retry·stream·독서 지침의 근거 벽을 분리(3곳). 조건부 가능성 표현 수용. |
| `09-stream-reduction.md` | 병렬 순서 규칙과 cancellation 근거를 분리(1곳). |
| `10-stochastic-control.md` | processor/warper와 독서 지침을 분리(2곳); 마지막 source span도 독립 문단으로 분리. `gate`는 기술 용어로 수용. |
| `11-tool-registry-routing.md` | 변경 없음 — `source`는 증거 범주 용어로 수용. |
| `12-permission-approval-sandbox.md` | orchestrator와 approval 근거를 분리(1곳). |
| `13-logical-call-effect.md` | retry 비보장과 Temporal 경계 근거를 분리(1곳). `artifact`·`runtime`은 기술 용어로 수용. |
| `14-parallel-speculation.md` | 변경 없음 — 인라인 식별자와 `source` 표기는 보존. |
| `15-delegation-parent-child.md` | metric 정의와 label cardinality 지침을 분리(1곳). `source`는 증거 범주 용어로 수용. |
| `16-planner-worker-dag.md` | admission/output과 publish/authority를 분리(1곳). `source`·`gate`는 계약 용어로 수용. |
| `17-debate-vote-verifier.md` | 변경 없음 — source/identifier 표기는 주장 검증의 기술 어휘로 수용. |
| `18-blackboard-shared-state.md` | 변경 없음 — source/artifact 표기는 상태·근거 구분에 필요해 수용. |
| `19-contract-net-auction.md` | FIPA·JADE와 AutoGen 근거를 분리(1곳). `source`·`gate`는 기술 용어로 수용. |
| `20-actor-mailbox-crdt-consensus.md` | supervisor 상태와 restart stash 계약을 분리(1곳). `source`는 증거 범주 용어로 수용. |
| `21-embedding-vector-search.md` | 평가 지표와 effect 전 재검사를 분리(1곳). source/identifier 표기는 검색 provenance 용어로 수용. |
| `22-vector-limits.md` | 변경 없음 — source/identifier 표기는 고정 기술 문맥으로 수용. |
| `23-graph-reasoner-temporal.md` | 변경 없음 — source 표기는 근거 locator 문맥으로 수용. |
| `24-hybrid-retrieval.md` | 변경 없음 — source/gate 표기는 검색 계약 용어로 수용. |
| `25-ask-or-act.md` | 변경 없음 — 조건부 표현은 정책 분기의 실제 불확실성을 나타내므로 수용. |
| `26-approval-consent-receipt.md` | 변경 없음 — `artifact`는 receipt 산출물의 기술 용어로 수용. |
| `27-interrupt-steer-resume.md` | 변경 없음 — cancellation·resume의 조건부 문장은 비보장 범위를 정확히 유지. |
| `28-event-log-checkpoint-replay.md` | 변경 없음 — `artifact`·`source`는 ledger/provenance 문맥으로 수용. |
| `29-outbox-inbox-saga.md` | 변경 없음 — `runtime`은 실행 경계 용어로 수용. |
| `30-lease-heartbeat-fencing.md` | local fixture의 비보장과 Kubernetes 코드 경계를 분리(1곳). |
| `31-fault-injection-recovery.md` | 변경 없음 — 장애 시나리오의 짧은 문단과 종결 변주 정상. |
| `32-trace-metric-log-receipt.md` | exporter 한계와 processor/metric 한계를 분리(1곳). `runtime`은 기술 용어로 수용. |
| `33-evaluation.md` | SWE-bench·AgentBench·tau-bench 근거를 논점 단위로 분리(2곳). source/runtime 표기는 평가 계약 용어로 수용. |
| `34-cost-capacity-slo.md` | 변경 없음 — source 표기는 증거 범주 용어로 수용. |
| `35-multitenant-deployment.md` | 변경 없음 — 계약·비보장 문단의 호흡과 마무리 정상. |
| `36-operations-playbook.md` | 변경 없음 — source는 기술 근거를 가리켜 수용. |
| `37-minimal-agentrun-golden-lab.md` | 변경 없음 — source/artifact 표기는 실습 산출물 용어로 수용. |
| `38-multiagent-coordination-lab.md` | 변경 없음 — source/artifact 표기는 fixture·증거 문맥으로 수용. |
| `39-retrieval-permission-effect-lab.md` | 변경 없음 — source/identifier/artifact 표기는 실습 ledger 문맥으로 수용. |
| `40-crash-recovery-deployment-lab.md` | 변경 없음 — gate는 운영 계약 용어로 수용. |
| `41-new-framework-autopsy.md` | 변경 없음 — source 표기는 evidence card와 line anchor 문맥으로 수용. |

## 구조 점검

- 모든 장에서 근거 링크 뒤의 한계·판단이 같은 문단에 묻힌 경우만 분리했다. 링크와 대상 span은 바꾸지 않았다.
- 반복 종결, 설명 없는 근거 벽, 단문 나열은 전체 spine에서 구조적 결함으로 확인되지 않았다. 목록·표·코드·Mermaid는 의도된 형식이므로 산문 리듬 점검에서 제외했다.
- 한국어 출판 lint의 자동 후보는 문맥 검토 전 오류가 아니다. 이번 판본에서는 수치·식별자·기술 용어·조건부 양태를 보존하는 편이 문체적 치환보다 우선한다.

## 검증 기록

- `python3 scripts/lint_korean_publication.py --config llm-textbooks.github.io/content/volume-3/book.yaml --strict`: 43개 문서, 장·절 참조 오류 0건, 이른 종결 heading 0건, 500자 초과 산문 문단 0건.
- `python3 scripts/audit_volume3_publication_completion.py --book llm-textbooks.github.io/content/volume-3/book.yaml ...`: 43개 문서 완료.
- `git diff --check`: 통과.

EPUB은 이 감사에서 다시 만들지 않았다.
