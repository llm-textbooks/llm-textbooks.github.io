#!/usr/bin/env python3
"""검색 시점 allow와 commit 시점 권한을 분리하는 fixture.

무엇을 검증하는가:
  policy generation p1에서 retrieval이 allow를 받고 후보를 얻은 뒤, p2에서
  같은 principal의 capability가 철회되는 순서를 in-process로 재현한다.
  effect-time recheck가 현재 generation(p2)으로 다시 판정해 deny하고,
  receiver에 receipt가 하나도 생기지 않는지 확인한다. 동시에 검색 시점
  allow를 generation 없는 key로 cache해 commit에 재사용하는 gate를 실제
  코드로 실행해, 그 gate가 철회 뒤에도 receipt를 만든다는 사실을
  "counterexample:" 접두 이벤트로 남긴다.

보장하지 않는 것:
  이것은 실제 policy engine(OpenFGA/OPA 등), 실제 SPARQL/SHACL 엔진,
  분산 저장소, 모델 provider의 동작을 관측한 결과가 아니다. 장이 정의한
  계약을 순수 표준 라이브러리로 만든 결정적 모델일 뿐이며, 여기서 통과한다고
  운영 시스템의 revocation 지연·복제 지연·cache 계층이 안전해지지 않는다.
  본 harness는 자기 event ledger 파일 외의 I/O·시간·난수를 쓰지 않는다.

관련 장:
  43장(캐시 공학: cache hit는 권한 증명이 아니다),
  45장(45.3.3 도구 실행 직전 권한 재판정, 45.8 반례 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

FIXTURE = "policy-revoke-between-retrieval-and-commit"
EXPECTED_SHA256 = "e63ee2577a5a8bc6c1037f9cb8a2e88d8400ff0ec42f779503f06057df981c84"

LEDGER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "recorded-events",
        FIXTURE + ".events.jsonl",
    )
)

RUN_ID = "run-ec-policy-revoke-001"
TURN_ID = "turn-2"
BRANCH_ID = "branch-main"
LOGICAL_CALL_ID = "call-refund-77"
IDEMPOTENCY_KEY = "idem-refund-77-a1"
PRINCIPAL = "principal:agent-desk-7"
TENANT = "tenant:acme"
RESOURCE = "order:77"
ACTION = "refund:create"
STATE_REVISION = "order:r17"
VECTOR_GENERATION = "v41"


class OracleError(AssertionError):
    """fixture가 주장하는 불변식이 깨졌을 때 올린다."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


class Ledger:
    """canonical event ledger. 한 줄에 하나의 정렬된 JSON row."""

    def __init__(self, fixture: str) -> None:
        self.fixture = fixture
        self.rows: list[dict] = []

    def record(self, stage: str, outcome: str, **fields: object) -> dict:
        row = dict(fields)
        row["fixture"] = self.fixture
        row["ordinal"] = len(self.rows) + 1
        row["stage"] = stage
        row["outcome"] = outcome
        self.rows.append(row)
        return row

    def render(self) -> str:
        return "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in self.rows
        )


class PolicyStore:
    """policy generation별 capability grant를 담는 결정적 모델."""

    def __init__(self) -> None:
        self._grants: dict[str, set[tuple[str, str, str]]] = {}
        self._order: list[str] = []
        self._decision_seq = 0

    def publish(self, generation: str, grants) -> None:
        self._grants[generation] = {tuple(grant) for grant in grants}
        self._order.append(generation)

    @property
    def current_generation(self) -> str:
        return self._order[-1]

    def decide(self, principal: str, action: str, resource: str, generation: str) -> dict:
        self._decision_seq += 1
        allowed = (principal, action, resource) in self._grants[generation]
        return {
            "decision_id": "pd-%s-%d" % (generation, self._decision_seq),
            "principal": principal,
            "action": action,
            "resource": resource,
            "policy_generation": generation,
            "effect": "allow" if allowed else "deny",
            "reason": "capability_granted" if allowed else "capability_revoked",
        }


class Receiver:
    """receipt를 발급하는 외부 receiver의 in-process 모델."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.receipts: list[dict] = []

    def commit(self, logical_call_id: str, idempotency_key: str, policy_generation: str) -> dict:
        for receipt in self.receipts:
            if receipt["idempotency_key"] == idempotency_key:
                return receipt
        receipt = {
            "receipt_id": "rcpt-%s-%d" % (self.name, len(self.receipts) + 1),
            "logical_call_id": logical_call_id,
            "idempotency_key": idempotency_key,
            "committed_revision": "order:r18",
            "authorizing_policy_generation": policy_generation,
        }
        self.receipts.append(receipt)
        return receipt


class GenerationFencedGate:
    """commit 시점에 현재 policy generation으로 다시 판정하는 gate."""

    def __init__(self, policy: PolicyStore) -> None:
        self.policy = policy
        self.decision_cache: dict[tuple, dict] = {}

    def cache_key(self, principal, action, resource, generation, state_revision) -> tuple:
        # 43.2: 재사용 시 결과의 해석을 바꾸는 변수는 key에 포함한다.
        return (principal, TENANT, action, resource, generation, state_revision)

    def authorize(self, generation: str) -> dict:
        key = self.cache_key(PRINCIPAL, ACTION, RESOURCE, generation, STATE_REVISION)
        cached = self.decision_cache.get(key)
        if cached is not None:
            return dict(cached, cache_state="hit")
        decision = self.policy.decide(PRINCIPAL, ACTION, RESOURCE, generation)
        self.decision_cache[key] = decision
        return dict(decision, cache_state="miss")


class GenerationBlindGate:
    """반증 대상: generation 없는 key로 검색 시점 allow를 cache해 재사용한다."""

    def __init__(self, policy: PolicyStore) -> None:
        self.policy = policy
        self.decision_cache: dict[tuple, dict] = {}

    def cache_key(self, principal, action, resource) -> tuple:
        return (principal, action, resource)

    def authorize(self, generation: str) -> dict:
        key = self.cache_key(PRINCIPAL, ACTION, RESOURCE)
        cached = self.decision_cache.get(key)
        if cached is not None:
            return dict(cached, cache_state="hit")
        decision = self.policy.decide(PRINCIPAL, ACTION, RESOURCE, generation)
        self.decision_cache[key] = decision
        return dict(decision, cache_state="miss")


def run_simulation() -> tuple[Ledger, list[str]]:
    ledger = Ledger(FIXTURE)
    policy = PolicyStore()
    policy.publish("p1", [(PRINCIPAL, ACTION, RESOURCE), (PRINCIPAL, "document:read", RESOURCE)])

    fenced_receiver = Receiver("fenced")
    blind_receiver = Receiver("blind")
    fenced_gate = GenerationFencedGate(policy)
    blind_gate = GenerationBlindGate(policy)

    ledger.record(
        "scope",
        "request_scoped",
        run_id=RUN_ID,
        turn_id=TURN_ID,
        branch_id=BRANCH_ID,
        principal=PRINCIPAL,
        tenant=TENANT,
        action=ACTION,
        resource=RESOURCE,
        state_revision=STATE_REVISION,
    )
    ledger.record(
        "policy",
        "generation_published",
        policy_generation="p1",
        grants=[[PRINCIPAL, ACTION, RESOURCE], [PRINCIPAL, "document:read", RESOURCE]],
    )

    retrieval_decision = fenced_gate.authorize("p1")
    blind_retrieval_decision = blind_gate.authorize("p1")
    expect(retrieval_decision["effect"] == "allow", "retrieval 시점 p1 판정이 allow가 아니다")
    expect(blind_retrieval_decision["effect"] == "allow", "반증 gate의 p1 판정이 allow가 아니다")

    ledger.record(
        "retrieval",
        "candidates_returned",
        vector_generation=VECTOR_GENERATION,
        candidate_ids=[RESOURCE],
        candidate_role="candidate_only_not_authorization",
    )
    ledger.record(
        "policy_decision",
        "allow",
        checkpoint="retrieval_time",
        policy_generation="p1",
        decision_id=retrieval_decision["decision_id"],
        cache_state=retrieval_decision["cache_state"],
        reason=retrieval_decision["reason"],
    )

    policy.publish("p2", [(PRINCIPAL, "document:read", RESOURCE)])
    ledger.record(
        "policy",
        "generation_published",
        policy_generation="p2",
        revoked=[[PRINCIPAL, ACTION, RESOURCE]],
        note="shift_handover_revoked_refund_capability",
    )

    commit_generation = policy.current_generation
    commit_decision = fenced_gate.authorize(commit_generation)
    expect(commit_generation == "p2", "commit 시점 generation이 p2가 아니다")
    expect(
        commit_decision["cache_state"] == "miss",
        "generation을 포함한 cache key가 p1 결정을 재사용했다",
    )
    expect(commit_decision["effect"] == "deny", "effect-time recheck가 deny하지 않았다")

    ledger.record(
        "admission",
        "deny",
        checkpoint="effect_time",
        policy_generation=commit_generation,
        decision_id=commit_decision["decision_id"],
        cache_state=commit_decision["cache_state"],
        reason=commit_decision["reason"],
        logical_call_id=LOGICAL_CALL_ID,
    )

    if commit_decision["effect"] == "allow":  # pragma: no cover - deny 경로만 도달한다
        fenced_receiver.commit(LOGICAL_CALL_ID, IDEMPOTENCY_KEY, commit_generation)
    ledger.record(
        "effect",
        "no_attempt",
        logical_call_id=LOGICAL_CALL_ID,
        idempotency_key=IDEMPOTENCY_KEY,
        attempts=0,
        receipts=len(fenced_receiver.receipts),
    )
    expect(fenced_receiver.receipts == [], "철회 뒤에 fenced receiver가 receipt를 발급했다")

    blind_commit_decision = blind_gate.authorize(policy.current_generation)
    ledger.record(
        "counterexample:admission",
        "allow",
        checkpoint="effect_time",
        gate="generation_blind_decision_cache",
        cache_state=blind_commit_decision["cache_state"],
        reused_decision_id=blind_commit_decision["decision_id"],
        reused_policy_generation=blind_commit_decision["policy_generation"],
        current_policy_generation=policy.current_generation,
    )
    expect(
        blind_commit_decision["cache_state"] == "hit"
        and blind_commit_decision["effect"] == "allow",
        "반증 gate가 p1 allow를 재사용하지 않아 대비가 성립하지 않는다",
    )

    blind_receipt = blind_receiver.commit(
        LOGICAL_CALL_ID, IDEMPOTENCY_KEY, blind_commit_decision["policy_generation"]
    )
    ledger.record(
        "counterexample:effect",
        "receipt_committed",
        gate="generation_blind_decision_cache",
        receipt_id=blind_receipt["receipt_id"],
        logical_call_id=LOGICAL_CALL_ID,
        idempotency_key=IDEMPOTENCY_KEY,
        authorizing_policy_generation=blind_receipt["authorizing_policy_generation"],
        current_policy_generation=policy.current_generation,
    )
    expect(len(blind_receiver.receipts) == 1, "반증 gate가 effect를 만들지 않았다")

    ledger.record(
        "oracle",
        "contrast_recorded",
        fenced_receipts=len(fenced_receiver.receipts),
        blind_receipts=len(blind_receiver.receipts),
        retrieval_trace_present=True,
        commit_time_decision="deny",
    )

    refuted = [
        "검색 시점 allow를 commit 시점 권한으로 재사용해도 된다 (generation 없는 decision cache는 철회 뒤에도 receipt를 만든다)",
        "retrieval trace가 남아 있으면 effect도 정당하다 (trace는 후보 기록일 뿐 승인 근거가 아니다)",
    ]
    return ledger, refuted


def validate_rows(rows: list[dict]) -> None:
    expect(bool(rows), "이벤트가 하나도 없다")
    for index, row in enumerate(rows, start=1):
        expect(row.get("ordinal") == index, "ordinal이 1..N 연속이 아니다: %r" % (row.get("ordinal"),))
        for field in ("fixture", "stage", "outcome"):
            expect(field in row, "필수 필드 %s가 없다: ordinal=%d" % (field, index))
        expect(row["fixture"] == FIXTURE, "fixture 이름이 다르다: ordinal=%d" % index)
        for key in row:
            expect(
                "timestamp" not in key and not key.endswith("_at"),
                "timestamp 필드는 금지된다: %s" % key,
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="%s fixture 검증" % FIXTURE)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="event ledger를 다시 쓰고 sha256을 출력한다",
    )
    args = parser.parse_args(argv)

    try:
        ledger, refuted = run_simulation()
        validate_rows(ledger.rows)
    except OracleError as exc:
        print("%s: oracle 실패: %s" % (FIXTURE, exc), file=sys.stderr)
        return 1

    payload = ledger.render().encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    if args.regenerate:
        os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
        with open(LEDGER_PATH, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload.decode("utf-8"))
        print(
            json.dumps(
                {
                    "fixture": FIXTURE,
                    "events": len(ledger.rows),
                    "sha256": digest,
                    "result": "regenerated",
                },
                ensure_ascii=False,
            )
        )
        return 0

    try:
        with open(LEDGER_PATH, "rb") as handle:
            committed = handle.read()
    except OSError as exc:
        print("%s: 커밋된 ledger를 읽을 수 없다: %s" % (FIXTURE, exc), file=sys.stderr)
        return 1

    if committed != payload:
        print(
            "%s: 재실행 결과가 커밋된 ledger와 다르다 (%d bytes != %d bytes)"
            % (FIXTURE, len(payload), len(committed)),
            file=sys.stderr,
        )
        return 1
    if digest != EXPECTED_SHA256:
        print(
            "%s: sha256 불일치: 계산 %s != 상수 %s" % (FIXTURE, digest, EXPECTED_SHA256),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "fixture": FIXTURE,
                "events": len(ledger.rows),
                "sha256": digest,
                "result": "pass",
                "refuted": refuted,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
