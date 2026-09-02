#!/usr/bin/env python3
"""dispatch 뒤 crash한 logical call의 resume reconciliation을 검증하는 fixture.

무엇을 검증하는가
  1. tool을 dispatch한 뒤 receipt를 받기 전에 process가 죽으면, durable run
     ledger만으로 상태를 재구성한 resume은 그 logical call을 `failed`가 아니라
     `unknown`으로 복원한다(44.6 "dispatch 후 process가 죽었다면 unknown이
     정직한 상태다").
  2. resume은 새 effect를 만들기 전에 같은 `effect_key`로 receiver receipt를
     조회한다. 두 분기를 모두 실행한다.
       - `applied`: 새 attempt를 dispatch하지 않고 receipt에서 local state를
         복구한다. receipt는 여전히 1건이다.
       - `not_seen`: 같은 effect_key로 안전하게 재시도한다. 그 결과도 receipt
         1건이다.
  3. receipt 조회 자체가 판정 불가(`indeterminate`)이면 dispatch도 종결도 하지
     않고 escalation한다. certainty를 꾸며 내지 않는다.
  4. 반증 oracle: `unknown`을 실패로 읽고 새 effect_key(`ek-7#resume`)로
     재시도하는 정책을 실제로 실행하면, 이미 적용된 receiver에서 receipt가
     2건 생긴다. 같은 업무가 외부 세계에 두 번 적용된 것이다.

이 fixture가 보장하지 않는 것
  - 실제 crash·실제 durable store·실제 결제 receiver를 재현하지 않는다.
    44장이 정의한 reconciliation 계약의 in-process 결정적 모델이다.
  - receiver가 effect_key로 receipt 조회를 제공한다는 것은 이 모델의 명세다.
    조회 API가 없는 receiver에서 `unknown`은 해소되지 않으며, 그 경우의 정답은
    자동 종결이 아니라 escalation이다(이 harness의 세 번째 분기).
  - resume이 복구하는 것은 logical call의 상태이지 model context나 사용자
    의도가 아니다. 재계획 정책은 이 fixture의 범위 밖이다.

관련 장: 44장(서브에이전트와 목표) 44.6 cancel/resume 상태 전이,
        44.7 completion proof의 effects 항목, 44.8 장면 C.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FIXTURE = "resume-pending-tool"
LEDGER_PATH = (
    Path(__file__).resolve().parent.parent / "recorded-events" / f"{FIXTURE}.events.jsonl"
)
EXPECTED_SHA256 = "92de8b63fb0a86884f29f43d2ab704f35328e9795a1a7a8fac09d7669037f4b8"

REFUTED = [
    "dispatch 뒤 crash를 실패로 읽고 새 effect_key로 재시도하면 이미 적용된 외부 효과가 두 번 적용된다",
    "local abort·process 종료는 receiver non-execution의 증거가 아니다. 증거는 effect_key로 조회한 receipt뿐이다",
]

LOGICAL_CALL_ID = "lc-7"
EFFECT_KEY = "ek-7"
NAIVE_EFFECT_KEY = "ek-7#resume"
TOOL = "payments.capture"
ARGS_DIGEST = "sha256:args-capture-order-77"


class OracleError(Exception):
    """계약 위반을 알리는 예외."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


class Ledger:
    """canonical event ledger. timestamp를 쓰지 않고 ordinal로만 순서를 고정한다."""

    def __init__(self, fixture: str) -> None:
        self.fixture = fixture
        self.rows: list[dict[str, object]] = []

    def emit(self, stage: str, outcome: str, **fields: object) -> dict[str, object]:
        row: dict[str, object] = dict(fields)
        row["fixture"] = self.fixture
        row["ordinal"] = len(self.rows) + 1
        row["stage"] = stage
        row["outcome"] = outcome
        self.rows.append(row)
        return row

    def canonical_bytes(self) -> bytes:
        return "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in self.rows
        ).encode("utf-8")


class DurableRunLedger:
    """crash를 넘겨 살아남는 append-only 행. in-memory projection은 살아남지 않는다."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def append(self, kind: str, **fields: object) -> dict[str, object]:
        row: dict[str, object] = dict(fields)
        row["kind"] = kind
        row["seq"] = len(self.rows) + 1
        self.rows.append(row)
        return row


class PaymentReceiver:
    """effect_key 하나당 최대 하나의 receipt를 남기고, 조회를 제공하는 모델."""

    def __init__(self, name: str, lookup_mode: str = "answers") -> None:
        self.name = name
        self.lookup_mode = lookup_mode
        self.receipts: dict[str, dict[str, object]] = {}
        self.apply_log: list[str] = []

    def submit(self, effect_key: str, attempt_id: str, amount: int) -> dict[str, object]:
        existing = self.receipts.get(effect_key)
        if existing is not None:
            return {"status": "duplicate_suppressed", "receipt": existing}
        receipt = {
            "receipt_id": f"r-{self.name}-{len(self.receipts) + 1}",
            "effect_key": effect_key,
            "applied_by_attempt": attempt_id,
            "captured_amount": amount,
        }
        self.receipts[effect_key] = receipt
        self.apply_log.append(effect_key)
        return {"status": "applied", "receipt": receipt}

    def lookup(self, effect_key: str) -> dict[str, object]:
        """44.6의 receipt query. 세 결과만 있다: applied / not_seen / indeterminate."""
        if self.lookup_mode == "no_query_api":
            return {"status": "indeterminate", "receipt": None}
        receipt = self.receipts.get(effect_key)
        if receipt is None:
            return {"status": "not_seen", "receipt": None}
        return {"status": "applied", "receipt": receipt}


def rebuild_call_states(durable_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """durable 행만 보고 logical call 상태를 재구성한다. 이것이 crash 모사의 전부다.

    dispatch 행이 있고 settle 행이 없으면 결론은 `unknown`이다. `failed`가 아니다.
    """
    states: dict[str, dict[str, object]] = {}
    for row in durable_rows:
        call_id = str(row.get("logical_call_id", ""))
        if not call_id:
            continue
        state = states.setdefault(
            call_id,
            {
                "logical_call_id": call_id,
                "status": "admitted",
                "effect_key": None,
                "attempts": [],
            },
        )
        if row["kind"] == "call_admitted":
            state["status"] = "admitted"
            state["effect_key"] = row.get("effect_key")
        elif row["kind"] == "attempt_dispatched":
            state["status"] = "in_flight"
            attempts = state["attempts"]
            assert isinstance(attempts, list)
            attempts.append(row.get("attempt_id"))
        elif row["kind"] == "attempt_settled":
            state["status"] = "settled"
    for state in states.values():
        if state["status"] == "in_flight":
            state["status"] = "unknown"
    return states


def resume_with_receipt_query(
    state: dict[str, object], receiver: PaymentReceiver, amount: int, retry_attempt_id: str
) -> dict[str, object]:
    """정상 resume 정책: 같은 effect_key로 조회하고, 조회 결과에만 근거해 행동한다."""
    effect_key = str(state["effect_key"])
    probe = receiver.lookup(effect_key)
    if probe["status"] == "applied":
        receipt = probe["receipt"]
        assert isinstance(receipt, dict)
        return {
            "action": "recovered_from_receipt",
            "dispatched": False,
            "effect_key": effect_key,
            "receipt": receipt,
        }
    if probe["status"] == "not_seen":
        result = receiver.submit(effect_key, retry_attempt_id, amount)
        return {
            "action": "safe_retry_same_key",
            "dispatched": True,
            "effect_key": effect_key,
            "receipt": result["receipt"],
        }
    return {
        "action": "escalate_unresolved",
        "dispatched": False,
        "effect_key": effect_key,
        "receipt": None,
    }


def resume_as_failure_with_new_key(
    state: dict[str, object], receiver: PaymentReceiver, amount: int, retry_attempt_id: str
) -> dict[str, object]:
    """반증 대상 정책: unknown을 실패로 읽고 조회 없이 새 effect_key로 재시도한다."""
    result = receiver.submit(NAIVE_EFFECT_KEY, retry_attempt_id, amount)
    return {
        "action": "retry_under_new_key",
        "dispatched": True,
        "effect_key": NAIVE_EFFECT_KEY,
        "receipt": result["receipt"],
    }


def build_pre_crash_durable_rows(ledger: Ledger, amount: int) -> DurableRunLedger:
    """crash 이전 구간: admission과 dispatch는 durable하게 남고 settle은 남지 않는다."""
    durable = DurableRunLedger()
    durable.append(
        "call_admitted",
        logical_call_id=LOGICAL_CALL_ID,
        tool=TOOL,
        effect_class="external_write",
        effect_key=EFFECT_KEY,
        args_digest=ARGS_DIGEST,
    )
    ledger.emit(
        "turn-admission",
        "call_admitted",
        logical_call_id=LOGICAL_CALL_ID,
        tool=TOOL,
        effect_class="external_write",
        effect_key=EFFECT_KEY,
        args_digest=ARGS_DIGEST,
        amount=amount,
    )
    durable.append(
        "attempt_dispatched",
        logical_call_id=LOGICAL_CALL_ID,
        attempt_id=f"{LOGICAL_CALL_ID}/at-1",
        effect_key=EFFECT_KEY,
    )
    ledger.emit(
        "dispatch",
        "attempt_dispatched",
        logical_call_id=LOGICAL_CALL_ID,
        attempt_id=f"{LOGICAL_CALL_ID}/at-1",
        effect_key=EFFECT_KEY,
        durable_before_dispatch=True,
    )
    return durable


def simulate() -> Ledger:
    ledger = Ledger(FIXTURE)
    amount = 4200

    durable = build_pre_crash_durable_rows(ledger, amount)

    # crash: in-memory future·scheduler·local verdict가 전부 사라진다.
    # durable ledger에는 attempt_settled 행이 없다.
    settled_rows = [row for row in durable.rows if row["kind"] == "attempt_settled"]
    check(settled_rows == [], "the crash must happen before any settle row is durable")
    ledger.emit(
        "crash",
        "process_lost_before_receipt",
        durable_rows=len(durable.rows),
        durable_kinds=sorted({str(row["kind"]) for row in durable.rows}),
        in_memory_state_lost=True,
        settle_rows=len(settled_rows),
    )

    rebuilt = rebuild_call_states(durable.rows)
    state = rebuilt[LOGICAL_CALL_ID]
    check(state["status"] == "unknown", f"resume must restore `unknown`, got {state['status']}")
    check(state["effect_key"] == EFFECT_KEY, "resume must restore the original effect_key")
    ledger.emit(
        "resume-rebuild",
        "logical_call_unknown",
        logical_call_id=LOGICAL_CALL_ID,
        restored_status=state["status"],
        effect_key=state["effect_key"],
        attempts_seen=state["attempts"],
        rebuilt_from="durable_run_ledger",
        note="unknown_is_not_failed",
    )

    # ---- 분기 A: receiver가 applied라고 답한다 ----
    applied_receiver = PaymentReceiver("applied")
    pre_crash = applied_receiver.submit(EFFECT_KEY, f"{LOGICAL_CALL_ID}/at-1", amount)
    check(pre_crash["status"] == "applied", "branch A requires the pre-crash attempt to have applied")
    branch_a = resume_with_receipt_query(
        state, applied_receiver, amount, f"{LOGICAL_CALL_ID}/at-2"
    )
    check(branch_a["action"] == "recovered_from_receipt", f"branch A action: {branch_a['action']}")
    check(branch_a["dispatched"] is False, "branch A must not dispatch a new attempt")
    check(len(applied_receiver.apply_log) == 1, "branch A must leave exactly one applied effect")
    receipt_a = branch_a["receipt"]
    assert isinstance(receipt_a, dict)
    ledger.emit(
        "resume-receipt-query",
        "applied",
        branch="A",
        logical_call_id=LOGICAL_CALL_ID,
        effect_key=EFFECT_KEY,
        receiver_status="applied",
        receipt_id=receipt_a["receipt_id"],
        applied_by_attempt=receipt_a["applied_by_attempt"],
    )
    ledger.emit(
        "resume-reconcile",
        "recovered_from_receipt",
        branch="A",
        logical_call_id=LOGICAL_CALL_ID,
        effect_key=EFFECT_KEY,
        dispatched=False,
        restored_status="applied",
        captured_amount=receipt_a["captured_amount"],
        receipts=len(applied_receiver.receipts),
    )

    # ---- 분기 B: receiver가 not_seen이라고 답한다 ----
    not_seen_receiver = PaymentReceiver("notseen")
    probe_b = not_seen_receiver.lookup(EFFECT_KEY)
    check(probe_b["status"] == "not_seen", "branch B requires the pre-crash attempt to have missed")
    ledger.emit(
        "resume-receipt-query",
        "not_seen",
        branch="B",
        logical_call_id=LOGICAL_CALL_ID,
        effect_key=EFFECT_KEY,
        receiver_status="not_seen",
        receipt_id=None,
    )
    branch_b = resume_with_receipt_query(
        state, not_seen_receiver, amount, f"{LOGICAL_CALL_ID}/at-2"
    )
    check(branch_b["action"] == "safe_retry_same_key", f"branch B action: {branch_b['action']}")
    check(branch_b["effect_key"] == EFFECT_KEY, "branch B must retry under the ORIGINAL effect_key")
    check(len(not_seen_receiver.apply_log) == 1, "branch B must leave exactly one applied effect")
    receipt_b = branch_b["receipt"]
    assert isinstance(receipt_b, dict)
    ledger.emit(
        "resume-reconcile",
        "safe_retry_same_effect_key",
        branch="B",
        logical_call_id=LOGICAL_CALL_ID,
        effect_key=EFFECT_KEY,
        dispatched=True,
        attempt_id=f"{LOGICAL_CALL_ID}/at-2",
        preserves_logical_identity=True,
        receipt_id=receipt_b["receipt_id"],
        receipts=len(not_seen_receiver.receipts),
    )

    # ---- 분기 C: receipt 조회 API가 없어 판정이 불가능하다 ----
    opaque_receiver = PaymentReceiver("opaque", lookup_mode="no_query_api")
    branch_c = resume_with_receipt_query(
        state, opaque_receiver, amount, f"{LOGICAL_CALL_ID}/at-2"
    )
    check(branch_c["action"] == "escalate_unresolved", f"branch C action: {branch_c['action']}")
    check(branch_c["dispatched"] is False, "an unresolved unknown must not be dispatched blindly")
    check(len(opaque_receiver.apply_log) == 0, "branch C must not create an effect")
    ledger.emit(
        "resume-receipt-query",
        "indeterminate",
        branch="C",
        logical_call_id=LOGICAL_CALL_ID,
        effect_key=EFFECT_KEY,
        receiver_status="indeterminate",
        receipt_query_api=False,
    )
    ledger.emit(
        "resume-escalation",
        "unresolved_escalated",
        branch="C",
        logical_call_id=LOGICAL_CALL_ID,
        effect_key=EFFECT_KEY,
        dispatched=False,
        goal_decision="not_complete",
        unresolved=[LOGICAL_CALL_ID],
    )

    # 두 해소 분기 모두 effect_key 하나에 receipt 하나다.
    check(
        sorted(applied_receiver.receipts) == [EFFECT_KEY],
        "branch A must not introduce a second effect_key",
    )
    check(
        sorted(not_seen_receiver.receipts) == [EFFECT_KEY],
        "branch B must not introduce a second effect_key",
    )
    ledger.emit(
        "receiver-audit",
        "one_receipt_per_effect_key",
        branch_a_receipts=len(applied_receiver.receipts),
        branch_b_receipts=len(not_seen_receiver.receipts),
        branch_c_receipts=len(opaque_receiver.receipts),
        effect_keys=[EFFECT_KEY],
    )

    # ---- 반증 경로: unknown을 실패로 읽고 새 effect_key로 재시도한다 ----
    naive_receiver = PaymentReceiver("naive")
    naive_receiver.submit(EFFECT_KEY, f"{LOGICAL_CALL_ID}/at-1", amount)
    naive = resume_as_failure_with_new_key(
        state, naive_receiver, amount, f"{LOGICAL_CALL_ID}/at-2"
    )
    check(naive["action"] == "retry_under_new_key", "the naive policy must retry under a new key")
    check(
        len(naive_receiver.apply_log) == 2,
        "the naive policy must produce a duplicate effect, otherwise nothing is refuted",
    )
    naive_receipt = naive["receipt"]
    assert isinstance(naive_receipt, dict)
    ledger.emit(
        "counterexample:unknown-treated-as-failed",
        "receipt_query_skipped",
        logical_call_id=LOGICAL_CALL_ID,
        restored_status="failed",
        actual_status="unknown",
        retry_effect_key=NAIVE_EFFECT_KEY,
        original_effect_key=EFFECT_KEY,
    )
    ledger.emit(
        "counterexample:receiver-commit",
        "duplicate_effect_applied",
        logical_call_id=LOGICAL_CALL_ID,
        effect_keys=sorted(naive_receiver.receipts),
        receipts=len(naive_receiver.receipts),
        applied_effects=len(naive_receiver.apply_log),
        duplicate_receipt_id=naive_receipt["receipt_id"],
        captured_amount_total=amount * 2,
    )

    ledger.emit(
        "oracle-refutation",
        "refuted",
        good_receipts_branch_a=len(applied_receiver.apply_log),
        good_receipts_branch_b=len(not_seen_receiver.apply_log),
        bad_receipts=len(naive_receiver.apply_log),
        refuted=REFUTED,
    )
    return ledger


def validate_rows(rows: list[dict[str, object]]) -> None:
    """공통 ledger 계약을 강제한다: 필수 필드, ordinal 1..N 연속, timestamp 금지."""
    check(bool(rows), "ledger is empty")
    for index, row in enumerate(rows, start=1):
        for field in ("fixture", "ordinal", "stage", "outcome"):
            check(field in row, f"required field {field} missing at ordinal {index}")
        check(row["fixture"] == FIXTURE, f"fixture name mismatch at ordinal {index}")
        check(
            row["ordinal"] == index,
            f"ordinal must be 1..N contiguous: got {row['ordinal']!r} at position {index}",
        )
        for key in row:
            check(
                "timestamp" not in key and not str(key).endswith("_at"),
                f"timestamp-like field is forbidden: {key}",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{FIXTURE} deterministic fixture")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="시뮬레이션을 다시 돌려 ledger 파일을 새로 쓰고 sha256을 출력한다",
    )
    args = parser.parse_args()

    try:
        ledger = simulate()
        validate_rows(ledger.rows)
    except OracleError as exc:
        sys.stderr.write(f"{FIXTURE}: oracle failed: {exc}\n")
        raise SystemExit(1)

    data = ledger.canonical_bytes()
    digest = hashlib.sha256(data).hexdigest()

    if args.regenerate:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEDGER_PATH.write_bytes(data)
        print(
            json.dumps(
                {
                    "fixture": FIXTURE,
                    "events": len(ledger.rows),
                    "sha256": digest,
                    "result": "regenerated",
                    "path": str(LEDGER_PATH),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return

    if not LEDGER_PATH.exists():
        sys.stderr.write(f"{FIXTURE}: missing ledger {LEDGER_PATH}; run --regenerate\n")
        raise SystemExit(1)
    committed = LEDGER_PATH.read_bytes()
    if committed != data:
        sys.stderr.write(
            f"{FIXTURE}: replayed ledger bytes differ from committed {LEDGER_PATH}\n"
            f"  replayed {len(data)} bytes sha256={digest}\n"
            f"  committed {len(committed)} bytes sha256={hashlib.sha256(committed).hexdigest()}\n"
        )
        raise SystemExit(1)
    if digest != EXPECTED_SHA256:
        sys.stderr.write(
            f"{FIXTURE}: sha256 mismatch: expected {EXPECTED_SHA256} got {digest}\n"
        )
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "fixture": FIXTURE,
                "events": len(ledger.rows),
                "sha256": digest,
                "result": "pass",
                "refuted": REFUTED,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
