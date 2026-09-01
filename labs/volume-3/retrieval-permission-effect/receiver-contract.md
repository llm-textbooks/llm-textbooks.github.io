# Receiver fixture contract

이 lab의 retained ledger가 전제하는 수신자는 localhost의 단일 SQLite WAL 파일을 공유하는 두 HTTP process다. `PRAGMA journal_mode=WAL`, `PRAGMA synchronous=FULL`을 사용하며, 이 설정은 single-host local transaction 범위만 다룬다.

`acquire(owner, ttl)`은 current owner와 epoch를 반환하거나 409 busy를 반환한다. `apply(idempotency_key, payload_digest, epoch)`은 receiver transaction 안에서 current owner·epoch·expiry를 검사한다. 오래된 epoch는 409 `stale-owner-or-lease`, 같은 key와 같은 payload는 stable receipt를 가진 `duplicate`, 다른 payload는 conflict여야 한다. response가 유실되면 caller는 receipt 조회 전까지 `Unknown`을 유지한다.

이 계약은 외부 비가역 effect, cross-host lock, quorum, network partition, clock skew, distributed exactly-once를 보장하지 않는다.
