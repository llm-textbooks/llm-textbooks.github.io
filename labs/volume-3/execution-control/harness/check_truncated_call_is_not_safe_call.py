#!/usr/bin/env python3
"""truncated tool call fence를 검증하는 fixture.

무엇을 검증하는가
  1. provider stream이 finish_reason=length로 끝났고 부분 argument 문자열이
     우연히 JSON으로 parse되는 경우(`{"path":"/prod","recursive":false}`),
     fence는 그 proposal을 handler에 도달시키지 않는다. dispatch가 없으므로
     receiver receipt도 0건이다.
  2. fence event에는 original call id, finish_reason, partial arguments digest,
     dispatched=false가 남는다(42.3 "fence가 제공하는 것은 안전한 admission이다").
  3. 대조군: 같은 argument라도 terminal disposition이 complete이면 gate가
     통과시킨다. 즉 fence의 판정 축은 parse 가능성이 아니라 완결성이다.
  4. 반증 oracle: `json.loads` 성공을 dispatch 근거로 쓰는 gate를 실제로 실행하면
     같은 stream에서 위험한 외부 effect가 1건 만들어진다. 두 경로의 receiver
     ledger 차이를 event로 남긴다.

이 fixture가 보장하지 않는 것
  - 실제 provider의 truncation 동작이나 특정 framework의 fence 구현을 재현하지
    않는다. 42장이 정의한 admission 계약의 in-process 결정적 모델일 뿐이다.
  - fence는 admission만 막는다. 사용자가 이어서 재시도할 때 그 도구의 receiver가
    idempotent하다는 보장은 이 fixture 어디에도 없다.
  - partial arguments digest는 조사 상관관계용 식별자이지 인수 복원 수단이 아니다.

관련 장: 42장(루프 엔지니어링) 42.3 truncated-call fence 절, 42.1 불변식 표.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FIXTURE = "truncated-call-is-not-safe-call"
LEDGER_PATH = (
    Path(__file__).resolve().parent.parent / "recorded-events" / f"{FIXTURE}.events.jsonl"
)
EXPECTED_SHA256 = "63f0209867058e2d038bd5c7c3b1d4b076013a9eacb9d148b6eb73e303e3cbf7"

REFUTED = [
    "JSON.parse 성공을 완결된 action의 증거로 읽으면 잘린 인수가 그대로 파일 시스템에 투영된다",
]

TRUNCATED_ARGUMENTS = '{"path":"/prod","recursive":false}'


class OracleError(Exception):
    """계약 위반을 알리는 예외."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class Ledger:
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


class FileSystemReceiver:
    """dispatch된 destructive effect만 기록하는 결정적 receiver 모델."""

    def __init__(self) -> None:
        self.receipts: list[dict[str, object]] = []

    def apply_delete(self, effect_key: str, path: str, recursive: bool) -> dict[str, object]:
        receipt = {
            "receipt_id": f"r-{len(self.receipts) + 1}",
            "effect_key": effect_key,
            "operation": "fs.delete",
            "path": path,
            "recursive": recursive,
        }
        self.receipts.append(receipt)
        return receipt

    def receipts_for(self, effect_key: str) -> list[dict[str, object]]:
        return [r for r in self.receipts if r["effect_key"] == effect_key]


class StreamItem:
    """reducer가 조립한 tool proposal 하나."""

    def __init__(self, call_id: str, tool: str, arguments: str, finish_reason: str) -> None:
        self.call_id = call_id
        self.tool = tool
        self.arguments = arguments
        self.finish_reason = finish_reason

    @property
    def terminal_disposition(self) -> str:
        return "complete" if self.finish_reason == "tool_calls" else "incomplete"

    def parses(self) -> bool:
        try:
            json.loads(self.arguments)
        except ValueError:
            return False
        return True


def fence_gate(item: StreamItem) -> dict[str, object]:
    """정상 gate: terminal disposition이 complete일 때만 dispatch를 허용한다."""
    if item.terminal_disposition != "complete":
        return {
            "dispatched": False,
            "reason": "truncated_proposal",
            "finish_reason": item.finish_reason,
        }
    if not item.parses():
        return {"dispatched": False, "reason": "schema_invalid", "finish_reason": item.finish_reason}
    return {"dispatched": True, "reason": "complete_and_valid", "finish_reason": item.finish_reason}


def parse_only_gate(item: StreamItem) -> dict[str, object]:
    """반증 대상 gate: parse 성공만 보고 dispatch를 허용한다."""
    if not item.parses():
        return {"dispatched": False, "reason": "parse_failed", "finish_reason": item.finish_reason}
    return {"dispatched": True, "reason": "parse_succeeded", "finish_reason": item.finish_reason}


def simulate() -> Ledger:
    ledger = Ledger(FIXTURE)

    truncated = StreamItem(
        call_id="call_a1",
        tool="fs.delete",
        arguments=TRUNCATED_ARGUMENTS,
        finish_reason="length",
    )
    partial_digest = sha256_text(truncated.arguments)

    ledger.emit(
        "stream-assembly",
        "partial_arguments_assembled",
        original_call_id=truncated.call_id,
        tool=truncated.tool,
        partial_args_digest=partial_digest,
        partial_args_chars=len(truncated.arguments),
    )
    ledger.emit(
        "stream-terminal",
        "finish_reason_length",
        original_call_id=truncated.call_id,
        finish_reason=truncated.finish_reason,
        terminal_disposition=truncated.terminal_disposition,
    )

    parses = truncated.parses()
    check(parses, "the fixture requires the partial string to parse by coincidence")
    ledger.emit(
        "json-probe",
        "parse_succeeded",
        original_call_id=truncated.call_id,
        json_parse_ok=parses,
        note="parse_success_is_not_completion_evidence",
    )

    receiver = FileSystemReceiver()
    verdict = fence_gate(truncated)
    check(verdict["dispatched"] is False, "the fence must not dispatch a truncated proposal")
    ledger.emit(
        "truncated-call-fence",
        "fenced_no_dispatch",
        original_call_id=truncated.call_id,
        tool=truncated.tool,
        finish_reason=truncated.finish_reason,
        partial_args_digest=partial_digest,
        json_parse_ok=parses,
        dispatched=False,
        fence_reason=verdict["reason"],
    )
    ledger.emit(
        "tool-result-synthesis",
        "synthetic_error_returned_to_model",
        original_call_id=truncated.call_id,
        result_kind="tool_error",
        replans_next_turn=True,
        dispatched=False,
    )

    fenced_receipts = receiver.receipts_for("ek-call_a1")
    check(len(fenced_receipts) == 0, "a fenced call must leave no receipt")
    ledger.emit(
        "receiver-audit",
        "no_effect_created",
        effect_key="ek-call_a1",
        receipts=len(fenced_receipts),
        destructive_effects=len(receiver.receipts),
    )

    # ---- 대조군: 같은 인수, complete disposition ----
    complete = StreamItem(
        call_id="call_b7",
        tool="fs.delete",
        arguments=TRUNCATED_ARGUMENTS,
        finish_reason="tool_calls",
    )
    control_verdict = fence_gate(complete)
    check(control_verdict["dispatched"] is True, "a complete proposal must be admitted")
    control_args = json.loads(complete.arguments)
    control_receipt = receiver.apply_delete(
        "ek-call_b7", str(control_args["path"]), bool(control_args["recursive"])
    )
    ledger.emit(
        "control-complete-call",
        "dispatched_after_complete_disposition",
        original_call_id=complete.call_id,
        finish_reason=complete.finish_reason,
        terminal_disposition=complete.terminal_disposition,
        dispatched=True,
        effect_key="ek-call_b7",
        receipt_id=control_receipt["receipt_id"],
    )
    ledger.emit(
        "gate-axis-audit",
        "gate_keys_on_disposition_not_parseability",
        parse_ok_both=True,
        dispatched_truncated=False,
        dispatched_complete=True,
    )

    # ---- 반증 경로: parse 성공을 dispatch 근거로 쓰는 gate ----
    naive_receiver = FileSystemReceiver()
    naive_verdict = parse_only_gate(truncated)
    check(
        naive_verdict["dispatched"] is True,
        "the parse-only gate must actually dispatch, otherwise nothing is refuted",
    )
    naive_args = json.loads(truncated.arguments)
    naive_receipt = naive_receiver.apply_delete(
        "ek-call_a1", str(naive_args["path"]), bool(naive_args["recursive"])
    )
    ledger.emit(
        "counterexample:parse-only-gate",
        "dispatched_on_parse_success",
        original_call_id=truncated.call_id,
        finish_reason=truncated.finish_reason,
        gate_reason=naive_verdict["reason"],
        dispatched=True,
    )
    ledger.emit(
        "counterexample:receiver-commit",
        "destructive_effect_created",
        original_call_id=truncated.call_id,
        effect_key="ek-call_a1",
        receipt_id=naive_receipt["receipt_id"],
        path=naive_receipt["path"],
        recursive=naive_receipt["recursive"],
    )

    naive_count = len(naive_receiver.receipts_for("ek-call_a1"))
    check(naive_count == 1, "the parse-only gate must create exactly one dangerous effect")
    ledger.emit(
        "oracle-refutation",
        "refuted",
        effect_key="ek-call_a1",
        fenced_receipts=len(fenced_receipts),
        parse_only_receipts=naive_count,
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
