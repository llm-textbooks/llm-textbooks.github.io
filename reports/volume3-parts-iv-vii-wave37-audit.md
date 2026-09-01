# Volume 3 IV–VII부 구조 감사

11~27장 17개 파일을 대상으로 장 참조, 문단 중복, Mermaid와 표의 계약, code fence를 검사했다.

- 554개 장문 문단을 5-token shingle로 비교했다. Jaccard 0.45 이상인 장간 중복 문단은 없었다.
- 15→16, 21→22, 22→23, 23→24의 다음 장 안내는 실제 목차와 맞았다. 22장이 24장의 hybrid 조합을 미리 가리키는 문장도 명시적 예고라서 유지했다.
- cancellation request와 receiver rollback, task/tool completion과 effect receipt, retrieval candidate와 admission, selection과 authorization, trace와 durable state를 Mermaid·표·본문 사이에서 대조했다. 상반된 전이는 발견하지 못했다.
- code fence의 여는 표시는 모두 언어가 지정돼 있었다. 실행에 필요한 타입·저장소·receiver가 생략된 Python 블록에는 의사코드 또는 축약 예제임을 fence 첫 줄에서 밝혔다. 이미 의사코드 표지가 있던 블록은 중복 수정하지 않았다.
- 원전 링크, 수식, 수치, 상태 이름과 의미 앵커는 바꾸지 않았다.

기계 판독 결과는 같은 디렉터리의 `volume3-parts-iv-vii-wave37-audit.json`에 있다.
