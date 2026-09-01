# 취소·관측·receipt 경계: 기록 검증 bundle

이 bundle은 13·27·32장의 **공개 저장소 안** 검증 자료다. 아래 명령은 네트워크·Docker·부모 연구 checkout을 사용하지 않는다. 보관된 ledger의 순서와 상태 oracle을 확인할 뿐, 과거 실행을 다시 실행하거나 live 서비스의 성질을 증명하지 않는다.

```bash
python3 labs/volume-3/runtime-cancellation-observability/verify_recorded_wave78.py --case rmcp-timeout-cancel --verify-recorded
python3 labs/volume-3/runtime-cancellation-observability/verify_recorded_wave78.py --case a2a-http-task-cancel --verify-recorded
python3 labs/volume-3/runtime-cancellation-observability/verify_recorded_wave78.py --case openfga-mcp-otel-receipt --verify-recorded
```

|기록|검증하는 관측|검증하지 않는 것|
|---|---|---|
|rmcp timeout→cancel|40 ms timeout 뒤 MCP cancellation notification, 협조적 handler가 effect 전 취소를 관측|이미 commit된 effect rollback, HTTP/A2A, durable retry|
|A2A HTTP task cancel|`WORKING` 재조회 뒤 cancel hook, cancel 응답과 후속 `GetTask`의 `CANCELED`|crash 뒤 state, multi-writer, 외부 효과 rollback|
|OpenFGA→MCP→OTel→receipt|allow arm만 tool/receipt로 진행, deny arm은 fail closed|권한 판정과 effect의 원자성, collector/Prometheus 전달 보장|

## 보관과 정제의 경계

ledger에는 loopback 주소, task ID, monotonic timestamp, credential, business payload를 넣지 않았다. 이 때문에 event ordering과 상태 전이는 확인할 수 있지만 지연 시간·실제 endpoint·원시 요청을 검증할 수는 없다. SHA-256 출력은 **배포된 정제본**의 무결성값이며, 원래 live harness의 실행 증명이 아니다.

## 원전 좌표

- [rmcp timeout 시 `notifications/cancelled` 전송](https://github.com/modelcontextprotocol/rust-sdk/blob/25220361d5540715294c501c289d79de4bec2bfc/crates/rmcp/src/service.rs#L400-L409)
- [rmcp server의 request-id cancellation token 취소](https://github.com/modelcontextprotocol/rust-sdk/blob/25220361d5540715294c501c289d79de4bec2bfc/crates/rmcp/src/service.rs#L1196-L1209)
- [A2A Python handler의 cancel 순서](https://github.com/a2aproject/a2a-python/blob/4b7b24293c55518e3f8b815b04fadf77ed488505/src/a2a/server/request_handlers/default_request_handler.py#L198-L258)
- [A2A Python JSON-RPC HTTP POST transport](https://github.com/a2aproject/a2a-python/blob/4b7b24293c55518e3f8b815b04fadf77ed488505/src/a2a/client/transports/jsonrpc.py#L339-L373)
- [OpenTelemetry Rust의 in-memory finished-span 조회](https://github.com/open-telemetry/opentelemetry-rust/blob/285dc925f98403ff426acc70968f104dc820d4f2/opentelemetry-sdk/src/trace/in_memory_exporter.rs#L117-L138)
- [SimpleSpanProcessor를 통한 exporter 연결](https://github.com/open-telemetry/opentelemetry-rust/blob/285dc925f98403ff426acc70968f104dc820d4f2/opentelemetry-sdk/src/trace/provider.rs#L306-L321)
