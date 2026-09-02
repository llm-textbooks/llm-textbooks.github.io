#!/usr/bin/env python3
"""병렬 tool 실행의 reducer 순서와 effect_key 중복 억제를 검증하는 fixture.

무엇을 검증하는가
  1. 두 tool call이 병렬로 dispatch되고 ordinal 2의 call이 ordinal 1보다 먼저
     settle해도, model-visible reducer는 admission 때 정한 call ordinal 순서로
     결과를 배열한다(42.3의 "실행 완료 순서와 대화 순서는 다른 계약이다").
  2. 같은 logical call의 두 attempt(원 시도 + response 유실 뒤 timeout retry)가
     같은 effect_key로 dispatch돼도 receiver ledger에는 receipt가 하나만 남는다
     (42.1의 "재시도는 logical identity를 보존한다").
  3. 반증 oracle: 완료 순서를 그대로 transcript 순서로 쓰는 reducer는 같은
     시뮬레이션에서 다른 transcript를 만들며, 이 harness는 그 차이를 코드로
     실행해 event로 남긴다.

이 fixture가 보장하지 않는 것
  - 실제 provider·실제 receiver·실제 분산 환경의 동작을 재현하지 않는다.
    여기서 도는 것은 42장이 정의한 계약의 in-process 결정적 모델이며,
    어떤 제품을 실행했다는 증거가 아니다.
  - receiver의 exactly-once는 이 모델이 그렇게 명세된 결과이지, 임의 receiver가
    effect_key만으로 중복을 억제한다는 일반 법칙이 아니다.
  - 병렬 결과 배열의 순서는 transcript 순서일 뿐 외부 세계의 commit 순서가 아니다.

관련 장: 42장(루프 엔지니어링) 42.1 불변식 표, 42.3 sequential/parallel 절.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FIXTURE = "parallel-order-and-effect-key"
LEDGER_PATH = (
    Path(__file__).resolve().parent.parent / "recorded-events" / f"{FIXTURE}.events.jsonl"
)
EXPECTED_SHA256 = "dd7b3f71cf0e9ec68758dd3b31b15306a636ecf470528494f28263754b66110c"

REFUTED = [
    "완료 순서를 transcript 순서로 쓰면 scheduler 타이밍이 다음 model request의 입력을 바꾼다",
    "timeout 뒤 재시도를 새 작업으로 보면 같은 logical call이 receiver에 두 번 적용된다",
]


class OracleError(Exception):
    """계약 위반을 알리는 예외."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


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


class Receiver:
    """effect_key 하나당 최대 하나의 receipt만 남기는 결정적 receiver 모델."""

    def __init__(self) -> None:
        self.receipts: dict[str, dict[str, object]] = {}
        self.apply_log: list[str] = []

    def submit(self, effect_key: str, attempt_id: str, payload: str) -> dict[str, object]:
        existing = self.receipts.get(effect_key)
        if existing is not None:
            return {"status": "duplicate_suppressed", "receipt": existing}
        receipt = {
            "receipt_id": f"r-{len(self.receipts) + 1}",
            "effect_key": effect_key,
            "applied_by_attempt": attempt_id,
            "payload_digest": sha256_text(payload),
        }
        self.receipts[effect_key] = receipt
        self.apply_log.append(effect_key)
        return {"status": "applied", "receipt": receipt}

    def lookup(self, effect_key: str) -> dict[str, object]:
        receipt = self.receipts.get(effect_key)
        if receipt is None:
            return {"status": "not_seen", "receipt": None}
        return {"status": "applied", "receipt": receipt}


def reduce_by_call_ordinal(settled: list[dict[str, object]]) -> list[str]:
    """model-visible reducer: admission 때 정한 call ordinal 순서로 배열한다."""
    ordered = sorted(settled, key=lambda entry: int(entry["call_ordinal"]))
    return [str(entry["logical_call_id"]) for entry in ordered]


def reduce_by_completion_order(settled: list[dict[str, object]]) -> list[str]:
    """반증 대상 reducer: settle된 순서를 그대로 transcript 순서로 쓴다."""
    ordered = sorted(settled, key=lambda entry: int(entry["completion_index"]))
    return [str(entry["logical_call_id"]) for entry in ordered]


def simulate() -> Ledger:
    ledger = Ledger(FIXTURE)
    receiver = Receiver()

    turn_id = "turn-3"
    admitted = [
        {
            "logical_call_id": "lc-1",
            "call_ordinal": 1,
            "tool": "repo.search",
            "effect_class": "read_only",
            "effect_key": None,
        },
        {
            "logical_call_id": "lc-2",
            "call_ordinal": 2,
            "tool": "ledger.append",
            "effect_class": "external_write",
            "effect_key": "ek-lc-2",
        },
    ]
    for call in admitted:
        ledger.emit(
            "turn-admission",
            "call_admitted",
            turn_id=turn_id,
            logical_call_id=call["logical_call_id"],
            call_ordinal=call["call_ordinal"],
            tool=call["tool"],
            effect_class=call["effect_class"],
            effect_key=call["effect_key"],
        )

    ledger.emit(
        "parallel-dispatch",
        "attempt_dispatched",
        logical_call_id="lc-1",
        call_ordinal=1,
        attempt_id="lc-1/at-1",
        effect_key=None,
    )
    ledger.emit(
        "parallel-dispatch",
        "attempt_dispatched",
        logical_call_id="lc-2",
        call_ordinal=2,
        attempt_id="lc-2/at-1",
        effect_key="ek-lc-2",
    )

    # lc-2의 첫 attempt는 receiver에 도달해 적용됐지만 응답이 유실된다.
    first = receiver.submit("ek-lc-2", "lc-2/at-1", "append(row=42)")
    check(first["status"] == "applied", "first attempt must apply on a fresh effect_key")
    ledger.emit(
        "receiver-commit",
        "applied",
        logical_call_id="lc-2",
        attempt_id="lc-2/at-1",
        effect_key="ek-lc-2",
        receipt_id=first["receipt"]["receipt_id"],
    )
    ledger.emit(
        "response-loss",
        "response_unknown",
        logical_call_id="lc-2",
        attempt_id="lc-2/at-1",
        effect_key="ek-lc-2",
        local_verdict="timeout",
        receiver_verdict="unknown_to_caller",
    )

    # 같은 logical call, 같은 effect_key로 재전송한다. 새 업무가 아니다.
    second = receiver.submit("ek-lc-2", "lc-2/at-2", "append(row=42)")
    check(
        second["status"] == "duplicate_suppressed",
        "retry under the same effect_key must not create a second effect",
    )
    ledger.emit(
        "retry-dispatch",
        "attempt_dispatched",
        logical_call_id="lc-2",
        attempt_id="lc-2/at-2",
        effect_key="ek-lc-2",
        preserves_logical_identity=True,
    )
    ledger.emit(
        "receiver-dedup",
        "duplicate_suppressed",
        logical_call_id="lc-2",
        attempt_id="lc-2/at-2",
        effect_key="ek-lc-2",
        receipt_id=second["receipt"]["receipt_id"],
        applied_by_attempt=second["receipt"]["applied_by_attempt"],
    )

    # settle 순서는 admission 순서와 반대다: ordinal 2가 먼저 끝난다.
    settled = [
        {
            "logical_call_id": "lc-2",
            "call_ordinal": 2,
            "completion_index": 1,
            "settled_attempt": "lc-2/at-2",
        },
        {
            "logical_call_id": "lc-1",
            "call_ordinal": 1,
            "completion_index": 2,
            "settled_attempt": "lc-1/at-1",
        },
    ]
    for entry in sorted(settled, key=lambda item: int(item["completion_index"])):
        ledger.emit(
            "future-settle",
            "settled",
            logical_call_id=entry["logical_call_id"],
            call_ordinal=entry["call_ordinal"],
            completion_index=entry["completion_index"],
            settled_attempt=entry["settled_attempt"],
        )

    transcript = reduce_by_call_ordinal(settled)
    completion_transcript = reduce_by_completion_order(settled)
    check(transcript == ["lc-1", "lc-2"], f"ordered reducer must follow call ordinal: {transcript}")
    check(
        completion_transcript == ["lc-2", "lc-1"],
        "the simulation must actually settle out of admission order",
    )
    check(
        transcript != completion_transcript,
        "the two reducers must disagree, otherwise the fixture proves nothing",
    )
    ledger.emit(
        "ordered-reduce",
        "call_ordinal_order_preserved",
        transcript_order=transcript,
        completion_order=completion_transcript,
        reducer="reduce_by_call_ordinal",
    )

    receipts_for_key = [r for r in receiver.receipts.values() if r["effect_key"] == "ek-lc-2"]
    check(len(receipts_for_key) == 1, f"expected one receipt, got {len(receipts_for_key)}")
    check(len(receiver.apply_log) == 1, "receiver must have applied the effect exactly once")
    ledger.emit(
        "receiver-audit",
        "one_receipt_per_effect_key",
        effect_key="ek-lc-2",
        receipts=len(receipts_for_key),
        attempts_dispatched=2,
        applied_by_attempt=receipts_for_key[0]["applied_by_attempt"],
    )

    # ---- 반증 경로 1: 완료 순서를 transcript 순서로 쓰는 reducer ----
    ledger.emit(
        "counterexample:completion-order-reduce",
        "transcript_reordered_by_scheduler_timing",
        transcript_order=completion_transcript,
        expected_order=transcript,
        reducer="reduce_by_completion_order",
    )

    # ---- 반증 경로 2: timeout 재시도를 새 effect_key로 보내는 정책 ----
    naive_receiver = Receiver()
    naive_first = naive_receiver.submit("ek-lc-2", "lc-2/at-1", "append(row=42)")
    naive_second = naive_receiver.submit("ek-lc-2#retry", "lc-2/at-2", "append(row=42)")
    check(naive_first["status"] == "applied", "naive first attempt applies")
    check(
        naive_second["status"] == "applied",
        "a fresh effect_key must slip past dedup, which is the defect being shown",
    )
    check(len(naive_receiver.apply_log) == 2, "naive policy must produce a duplicate effect")
    ledger.emit(
        "counterexample:new-key-retry",
        "duplicate_effect_applied",
        effect_keys=sorted(naive_receiver.receipts),
        receipts=len(naive_receiver.receipts),
        applied_effects=len(naive_receiver.apply_log),
    )

    ledger.emit(
        "oracle-refutation",
        "refuted",
        good_transcript=transcript,
        bad_transcript=completion_transcript,
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
