#!/usr/bin/env python3
"""Verify retained, sanitized Wave77/78 ledgers; it never starts a service."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = {
    "rmcp-timeout-cancel": {
        "file": "recorded-events/rmcp-timeout-cancel.events.jsonl",
        "outcomes": ["initialized", "tools_call_sent_with_timeout_40ms", "tools_call_received", "timeout_and_cancel_notification_sent", "request_context_cancelled_before_effect", "connection_closed_after_observation"],
    },
    "a2a-http-task-cancel": {
        "file": "recorded-events/a2a-http-task-cancel.events.jsonl",
        "kinds": ["task_working_emitted", "get_task_working", "cancel_entered", "task_canceled_emitted", "cancel_task_returned", "get_task_canceled"],
    },
    "openfga-mcp-otel-receipt": {
        "file": "recorded-events/openfga-mcp-otel-receipt.events.jsonl",
        "arms": {
            "allow_tool_receipt": ["started", "openfga_decision_allow_or_deny", "tools_call_sent", "tool_received", "effect_applied_receipt_committed", "receipt_joined"],
            "deny_no_tool_receipt": ["started", "openfga_decision_allow_or_deny", "not_invoked_fail_closed"],
        },
    },
}

def ordered_subset(observed: list[str], expected: list[str], label: str) -> None:
    cursor = 0
    for item in observed:
        if cursor < len(expected) and item == expected[cursor]:
            cursor += 1
    if cursor != len(expected):
        raise SystemExit(f"missing ordered event for {label}: {expected[cursor]}")

def verify(case: str) -> dict[str, object]:
    spec = CASES[case]
    path = ROOT / str(spec["file"])
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode().splitlines() if line]
    ordinals = [row["ordinal"] for row in rows]
    if ordinals != list(range(1, len(rows) + 1)) and case != "openfga-mcp-otel-receipt":
        raise SystemExit(f"non-contiguous ordinals: {case}")
    if "outcomes" in spec:
        ordered_subset([row["outcome"] for row in rows], spec["outcomes"], case)
    if "kinds" in spec:
        ordered_subset([row["kind"] for row in rows], spec["kinds"], case)
        states = {row.get("kind"): row.get("state") for row in rows}
        if states["get_task_working"] != "TASK_STATE_WORKING" or states["cancel_task_returned"] != "TASK_STATE_CANCELED" or states["get_task_canceled"] != "TASK_STATE_CANCELED":
            raise SystemExit("A2A state oracle failed")
    if "arms" in spec:
        for arm, expected in spec["arms"].items():
            arm_rows = [row for row in rows if row["arm"] == arm]
            if [row["ordinal"] for row in arm_rows] != list(range(1, len(arm_rows) + 1)):
                raise SystemExit(f"non-contiguous ordinals: {arm}")
            ordered_subset([row["outcome"] for row in arm_rows], expected, arm)
        deny = [row["outcome"] for row in rows if row["arm"] == "deny_no_tool_receipt"]
        if "tools_call_sent" in deny or "effect_applied_receipt_committed" in deny:
            raise SystemExit("deny arm invoked a tool or committed a receipt")
    return {"case": case, "events": len(rows), "sha256": hashlib.sha256(data).hexdigest(), "result": "pass", "mode": "recorded-verification-only"}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--verify-recorded", action="store_true", help="verify shipped ledger only; no process or network is started")
    args = parser.parse_args()
    if not args.verify_recorded:
        parser.error("this bundle intentionally supports --verify-recorded only")
    print(json.dumps(verify(args.case), sort_keys=True))

if __name__ == "__main__":
    main()
