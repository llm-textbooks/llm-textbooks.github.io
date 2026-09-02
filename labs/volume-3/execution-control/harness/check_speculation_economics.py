#!/usr/bin/env python3
"""hedge된 speculation의 비용 원장과 패자 effect 판정을 검증하는 fixture.

무엇을 검증하는가
  1. 같은 logical call에 대해 hedge attempt 2개를 dispatch하고 하나가 먼저
     settle해 승자가 된다. 패자는 dispatch 이후에 cancel된다.
  2. cancel은 비용 환불이 아니다. 패자가 이미 소비한 token은 cancel 뒤에도
     비용 원장에 settled로 남는다(42.3 "이미 끝난 loser의 비용은 cancel 뒤에도
     남으며", 44.7.1 budget ledger).
  3. 두 hedge attempt가 같은 effect_key를 공유하므로 receiver receipt는 1건이다.
     승자 attempt가 그 receipt의 소유자다.
  4. dispatch 후 cancel된 패자의 effect verdict는 rollback으로 얻지 않는다.
     같은 effect_key로 receipt를 조회해 "이미 승자 attempt가 적용했다"를 확인하는
     방식으로만 판정한다(42.2 "cancel은 effect verdict가 아니다", 44.8 장면 C).
  5. 비용 oracle: total_cost == winner_cost + loser_cost. cancel을 환불로 처리해
     패자 비용을 원장에서 지우는 회계를 실제로 실행하면, 남은 budget이 과대
     계상되어 44.7.1의 admission 부등식이 뒤집히고 hard cap을 넘는 child spawn을
     허용한다. 그 초과분을 코드로 계산해 event로 남긴다.

이 fixture가 보장하지 않는 것
  - 실제 provider의 hedge 요금, 실제 취소 API의 과금 시점, 실제 receiver의
     rollback 계약을 재현하지 않는다. 42·44장이 정의한 계약의 in-process
     결정적 모델이다.
  - "cancel 뒤 receipt가 1건"은 이 receiver가 effect_key로 중복을 억제하도록
    명세되었기 때문이다. 임의 receiver가 그렇다는 일반 법칙이 아니다.
  - 비용 단위는 이 fixture 안에서 정의한 정규화 단위(input 1, output 5)이며,
    특정 provider의 가격표가 아니다. token 비용만 세고 receiver 비용·human
    review는 세지 않는다.

관련 장: 42장(루프 엔지니어링) 42.2 cancel의 네 지점, 42.3 sequential/parallel 절,
        44장(서브에이전트와 목표) 44.7.1 budget ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FIXTURE = "speculation-economics"
LEDGER_PATH = (
    Path(__file__).resolve().parent.parent / "recorded-events" / f"{FIXTURE}.events.jsonl"
)
EXPECTED_SHA256 = "e4dec4340b218dd35f03619323558140e42ab2a6175e27f5621829292792d85e"

REFUTED = [
    "cancel을 환불로 회계하면 패자 비용이 사라져 남은 budget이 과대 계상되고 hard cap을 넘는 spawn이 승인된다",
    "cancel 이벤트는 effect verdict가 아니다. dispatch 후 취소된 attempt의 판정 근거는 rollback이 아니라 effect_key receipt 조회다",
    "hedge attempt마다 다른 effect_key를 쓰면 같은 logical call이 외부 세계에 두 번 적용된다",
]

LOGICAL_CALL_ID = "lc-9"
EFFECT_KEY = "ek-9"
TOOL = "search.index_rebuild"

# 정규화된 비용 단위. provider 가격표가 아니라 이 fixture의 정의다.
INPUT_UNIT_COST = 1
OUTPUT_UNIT_COST = 5

HARD_CAP = 6000
RECONCILIATION_RESERVE = 400
NEXT_CHILD_ESTIMATE = 900
NEXT_JOIN_ESTIMATE = 200


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


def cost_units(input_tokens: int, output_tokens: int) -> int:
    return input_tokens * INPUT_UNIT_COST + output_tokens * OUTPUT_UNIT_COST


class CostLedger:
    """attempt별 소비 비용을 settled로 기록하는 append-only 원장."""

    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def settle(
        self, attempt_id: str, role: str, input_tokens: int, output_tokens: int, disposition: str
    ) -> dict[str, object]:
        entry = {
            "attempt_id": attempt_id,
            "role": role,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_units": cost_units(input_tokens, output_tokens),
            "disposition": disposition,
            "state": "settled",
        }
        self.entries.append(entry)
        return entry

    def total(self) -> int:
        return sum(int(entry["cost_units"]) for entry in self.entries)

    def total_excluding_cancelled(self) -> int:
        """반증 대상 회계: cancel된 attempt의 비용을 원장에서 지운다."""
        return sum(
            int(entry["cost_units"])
            for entry in self.entries
            if entry["disposition"] != "cancelled_after_dispatch"
        )


class IndexReceiver:
    """effect_key 하나당 최대 하나의 receipt만 남기는 결정적 receiver 모델."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.receipts: dict[str, dict[str, object]] = {}
        self.apply_log: list[str] = []

    def submit(self, effect_key: str, attempt_id: str) -> dict[str, object]:
        existing = self.receipts.get(effect_key)
        if existing is not None:
            return {"status": "duplicate_suppressed", "receipt": existing}
        receipt = {
            "receipt_id": f"r-{self.name}-{len(self.receipts) + 1}",
            "effect_key": effect_key,
            "applied_by_attempt": attempt_id,
        }
        self.receipts[effect_key] = receipt
        self.apply_log.append(effect_key)
        return {"status": "applied", "receipt": receipt}

    def lookup(self, effect_key: str) -> dict[str, object]:
        receipt = self.receipts.get(effect_key)
        if receipt is None:
            return {"status": "not_seen", "receipt": None}
        return {"status": "applied", "receipt": receipt}


def available_budget(settled: int, reserved: int) -> int:
    """44.7.1: available = hard_cap - settled - reserved - reconciliation_reserve."""
    return HARD_CAP - settled - reserved - RECONCILIATION_RESERVE


def spawn_admitted(available: int) -> bool:
    """44.7.1: estimated_child + estimated_join <= available 일 때만 spawn을 허용한다."""
    return NEXT_CHILD_ESTIMATE + NEXT_JOIN_ESTIMATE <= available


def simulate() -> Ledger:
    ledger = Ledger(FIXTURE)
    costs = CostLedger()
    receiver = IndexReceiver("hedged")

    hedges = [
        {"attempt_id": f"{LOGICAL_CALL_ID}/at-a", "route": "provider-a", "input": 1200, "output": 180},
        {"attempt_id": f"{LOGICAL_CALL_ID}/at-b", "route": "provider-b", "input": 1200, "output": 300},
    ]

    ledger.emit(
        "turn-admission",
        "call_admitted",
        logical_call_id=LOGICAL_CALL_ID,
        tool=TOOL,
        effect_class="external_write",
        effect_key=EFFECT_KEY,
        hedge_degree=len(hedges),
        hedge_shares_effect_key=True,
    )
    for hedge in hedges:
        ledger.emit(
            "speculation-dispatch",
            "hedge_attempt_dispatched",
            logical_call_id=LOGICAL_CALL_ID,
            attempt_id=hedge["attempt_id"],
            route=hedge["route"],
            effect_key=EFFECT_KEY,
        )

    # at-b가 먼저 settle해 승자가 된다.
    winner, loser = hedges[1], hedges[0]
    submitted = receiver.submit(EFFECT_KEY, str(winner["attempt_id"]))
    check(submitted["status"] == "applied", "the winner must apply the effect")
    winner_receipt = submitted["receipt"]
    assert isinstance(winner_receipt, dict)
    winner_entry = costs.settle(
        str(winner["attempt_id"]), "winner", int(winner["input"]), int(winner["output"]), "settled_first"
    )
    ledger.emit(
        "speculation-settle",
        "winner_selected",
        logical_call_id=LOGICAL_CALL_ID,
        attempt_id=winner["attempt_id"],
        route=winner["route"],
        effect_key=EFFECT_KEY,
        receipt_id=winner_receipt["receipt_id"],
        cost_units=winner_entry["cost_units"],
    )

    # 패자를 cancel한다. cancel 시점은 이미 dispatch 이후다.
    loser_entry = costs.settle(
        str(loser["attempt_id"]),
        "loser",
        int(loser["input"]),
        int(loser["output"]),
        "cancelled_after_dispatch",
    )
    ledger.emit(
        "speculation-cancel",
        "loser_cancelled_after_dispatch",
        logical_call_id=LOGICAL_CALL_ID,
        attempt_id=loser["attempt_id"],
        route=loser["route"],
        cancel_point="handler_after_dispatch_before_response",
        cost_refunded=False,
        cost_units=loser_entry["cost_units"],
    )
    ledger.emit(
        "cost-ledger",
        "consumed_cost_retained_after_cancel",
        winner_attempt=winner["attempt_id"],
        winner_cost_units=winner_entry["cost_units"],
        loser_attempt=loser["attempt_id"],
        loser_cost_units=loser_entry["cost_units"],
        loser_state=loser_entry["state"],
        total_cost_units=costs.total(),
    )

    expected_total = int(winner_entry["cost_units"]) + int(loser_entry["cost_units"])
    check(
        costs.total() == expected_total,
        f"total_cost must equal winner + loser: {costs.total()} != {expected_total}",
    )
    check(int(loser_entry["cost_units"]) > 0, "the loser must have consumed real tokens")

    # dispatch 후 cancel된 패자의 effect 판정: rollback이 아니라 receipt 조회다.
    loser_probe = receiver.lookup(EFFECT_KEY)
    check(loser_probe["status"] == "applied", "the shared effect_key must resolve to one receipt")
    probe_receipt = loser_probe["receipt"]
    assert isinstance(probe_receipt, dict)
    check(
        probe_receipt["applied_by_attempt"] == winner["attempt_id"],
        "the receipt owner must be the winner attempt",
    )
    ledger.emit(
        "loser-effect-verdict",
        "resolved_by_receipt_query",
        logical_call_id=LOGICAL_CALL_ID,
        cancelled_attempt=loser["attempt_id"],
        effect_key=EFFECT_KEY,
        receiver_status=loser_probe["status"],
        applied_by_attempt=probe_receipt["applied_by_attempt"],
        rollback_attempted=False,
        cancel_is_effect_verdict=False,
    )

    check(len(receiver.apply_log) == 1, "hedging must not create a second external effect")
    check(sorted(receiver.receipts) == [EFFECT_KEY], "hedge attempts must share one effect_key")
    ledger.emit(
        "receiver-audit",
        "one_receipt_per_effect_key",
        effect_key=EFFECT_KEY,
        attempts_dispatched=len(hedges),
        receipts=len(receiver.receipts),
        applied_effects=len(receiver.apply_log),
        applied_by_attempt=winner_receipt["applied_by_attempt"],
    )

    honest_available = available_budget(costs.total(), reserved=0)
    honest_spawn = spawn_admitted(honest_available)
    check(honest_spawn is False, "the honest ledger must deny the next spawn for this fixture")
    ledger.emit(
        "budget-admission",
        "spawn_denied",
        hard_cap=HARD_CAP,
        settled=costs.total(),
        reserved=0,
        reconciliation_reserve=RECONCILIATION_RESERVE,
        available=honest_available,
        estimated_child=NEXT_CHILD_ESTIMATE,
        estimated_join=NEXT_JOIN_ESTIMATE,
        admitted=False,
    )

    # ---- 반증 경로 1: cancel을 환불로 회계한다 ----
    refund_settled = costs.total_excluding_cancelled()
    refund_available = available_budget(refund_settled, reserved=0)
    refund_spawn = spawn_admitted(refund_available)
    check(
        refund_settled == int(winner_entry["cost_units"]),
        "the refund accounting must drop exactly the loser cost",
    )
    check(
        refund_spawn is True,
        "the refund accounting must admit the spawn, otherwise nothing is refuted",
    )
    overshoot = NEXT_CHILD_ESTIMATE + NEXT_JOIN_ESTIMATE - honest_available
    check(overshoot > 0, "the admitted spawn must actually exceed the honest budget")
    ledger.emit(
        "counterexample:cancel-as-refund",
        "loser_cost_erased",
        erased_attempt=loser["attempt_id"],
        erased_cost_units=loser_entry["cost_units"],
        reported_settled=refund_settled,
        actual_settled=costs.total(),
    )
    ledger.emit(
        "counterexample:budget-admission",
        "spawn_admitted_over_hard_cap",
        reported_available=refund_available,
        actual_available=honest_available,
        estimated_child=NEXT_CHILD_ESTIMATE,
        estimated_join=NEXT_JOIN_ESTIMATE,
        admitted=True,
        hard_cap_overshoot_units=overshoot,
    )

    # ---- 반증 경로 2: hedge attempt마다 다른 effect_key를 쓴다 ----
    naive_receiver = IndexReceiver("perattempt")
    naive_keys = [f"{EFFECT_KEY}#{hedge['route']}" for hedge in hedges]
    for hedge, key in zip(hedges, naive_keys):
        naive_receiver.submit(key, str(hedge["attempt_id"]))
    check(
        len(naive_receiver.apply_log) == 2,
        "per-attempt effect keys must produce a duplicate effect, otherwise nothing is refuted",
    )
    ledger.emit(
        "counterexample:per-attempt-effect-key",
        "duplicate_effect_applied",
        logical_call_id=LOGICAL_CALL_ID,
        effect_keys=sorted(naive_receiver.receipts),
        receipts=len(naive_receiver.receipts),
        applied_effects=len(naive_receiver.apply_log),
        cancelled_attempt_still_applied=loser["attempt_id"],
    )

    ledger.emit(
        "oracle-refutation",
        "refuted",
        total_cost_units=costs.total(),
        winner_cost_units=winner_entry["cost_units"],
        loser_cost_units=loser_entry["cost_units"],
        refund_accounting_cost_units=refund_settled,
        good_receipts=len(receiver.apply_log),
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
