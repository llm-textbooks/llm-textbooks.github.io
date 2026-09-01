# 검색·권한·효과 경계: 공개 검증 bundle

이 디렉터리는 39.9절이 가리키는 **공개 저장소 내부**의 재현 가능한 artifact-verification bundle이다. 세 명령은 네트워크, Docker, 부모 checkout을 사용하지 않는다. retained event ledger의 SHA-256과 event ordinal·핵심 outcome을 검증한다.

```bash
python3 labs/volume-3/retrieval-permission-effect/verify_generation_skew.py --verify-recorded
python3 labs/volume-3/retrieval-permission-effect/verify_lease_takeover.py --verify-recorded
python3 labs/volume-3/retrieval-permission-effect/verify_response_boundary.py --verify-recorded
```

## 무엇을 검증하는가

| harness | 확인하는 사건 | 보장하지 않는 것 |
|---|---|---|
| generation skew | Qdrant 후보 `g2`와 OpenFGA deny, allow와 빈 후보의 독립 관측 | 공통 watermark·cross-product transaction |
| lease takeover | epoch 1→2, stale owner 409, epoch 3 receipt/duplicate | multi-host lock·distributed exactly-once |
| response boundary | before-send의 receipt 부재와 post-commit response-loss 뒤 receipt/duplicate | TCP ACK·임의 partition·remote rollback |

## bounded live reproduction의 전제 조건

이 공개 bundle은 안전하고 결정적인 검증 모드만 제공한다. 원래의 live harness는 별도 연구 checkout에서 실행됐고, 다음 고정 조건을 요구한다: Qdrant `v1.19.0` binary, OpenFGA source commit `a7dfe8491dc7f9cd5905f4e9ae6c8e1d718c4bd9`를 빌드한 binary, loopback TCP port 네 개, Python 3 표준 라이브러리, disposable SQLite WAL scratch directory가 필요하다. lease/response arm은 이 디렉터리의 [receiver contract](./receiver-contract.md)를 구현한 disposable localhost receiver만 대상으로 해야 한다. production endpoint, customer tenant, 실제 결제·메일·배포 effect에는 실행하지 않는다.

Qdrant/OpenFGA arm의 fixed source coordinates는 [Qdrant query handler](https://github.com/qdrant/qdrant/blob/74f3e85b9473c62560006c043e13737ce6b48412/src/actix/api/query_api.rs#L31-L110), [OpenFGA Check cache path](https://github.com/openfga/openfga/blob/a7dfe8491dc7f9cd5905f4e9ae6c8e1d718c4bd9/internal/graph/cached_resolver.go#L136-L168), [OpenFGA Write command](https://github.com/openfga/openfga/blob/a7dfe8491dc7f9cd5905f4e9ae6c8e1d718c4bd9/pkg/server/commands/write.go#L80-L115)에서 확인한다. 검증 모드는 과거 실행의 증거 무결성을 확인할 뿐, 이 제품들을 새로 실행하거나 그 동작을 일반화하지 않는다.
