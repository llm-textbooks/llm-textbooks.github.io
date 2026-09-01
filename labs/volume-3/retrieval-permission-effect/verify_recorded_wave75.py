#!/usr/bin/env python3
"""Verify the retained, sanitized Wave75 event ledgers without parent paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = {
    "generation-skew": {
        "file": "recorded-events/generation-skew.events.jsonl",
        "sha256": "348fed538d05714a483e0433008400ac2d35b9ca78c2f0dfdebd75b558021ffc",
        "runs": {
            "agent-run-wave75-vector_publish_then_auth_revoke-001": ["point_upserted", "viewer_tuple_deleted", "candidates_returned", "deny"],
            "agent-run-wave75-auth_revoke_then_vector_publish-001": ["viewer_tuple_deleted", "point_upserted", "candidates_returned", "deny"],
            "agent-run-wave75-auth_grant_then_vector_delete-001": ["viewer_tuple_written", "point_deleted", "candidates_returned", "allow"],
        },
    },
    "lease-takeover": {
        "file": "recorded-events/lease-takeover.events.jsonl",
        "sha256": "e1c51b90d98f1f281758dd58ec7822c9e8e1bae9568769e87d1622d0c6f56f99",
        "runs": {"agent-run-wave75-two-receiver-001": ["owner-a-epoch-1", "owner-b-busy", "owner-b-epoch-2", "stale-owner-rejected", "owner-b-applied", "owner-b-connection-lost-after-commit", "owner-b-epoch-3", "applied-receipt-read", "duplicate-reconciled"]},
    },
    "response-boundary": {
        "file": "recorded-events/response-boundary.events.jsonl",
        "sha256": "94f8266a7fbe75c06aab00847be2a88b53a2f1a6183e65bca0aa018d8909e315",
        "runs": {
            "agent-run-wave75-before_send_response_unknown-001": ["response_unknown", "absent_after_before_send", "absent", "applied", "one_local_applied_receipt"],
            "agent-run-wave75-after_receiver_commit_response_loss_retry_new_owner-001": ["response_unknown", "applied_after_client_response_unknown", "applied", "ready_new_process_shared_ledger", "duplicate", "one_local_applied_receipt"],
        },
    },
}

def verify(case: str) -> dict[str, object]:
    spec = CASES[case]
    path = ROOT / str(spec["file"])
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec["sha256"]:
        raise SystemExit(f"hash mismatch: {path} expected {spec['sha256']} got {digest}")
    rows = [json.loads(line) for line in data.decode().splitlines() if line]
    for run, expected in spec["runs"].items():
        observed = [row["outcome"] for row in rows if row["agent_run_id"] == run]
        cursor = 0
        for outcome in observed:
            if cursor < len(expected) and outcome == expected[cursor]:
                cursor += 1
        if cursor != len(expected):
            raise SystemExit(f"missing ordered outcome in {run}: {expected[cursor]}")
        ordinals = [row["ordinal"] for row in rows if row["agent_run_id"] == run]
        if ordinals != list(range(1, len(ordinals) + 1)):
            raise SystemExit(f"non-contiguous ordinal sequence: {run}")
    return {"case": case, "events": len(rows), "sha256": digest, "result": "pass"}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--verify-recorded", action="store_true", help="verify only the shipped event ledger; no process or network is started")
    args = parser.parse_args()
    if not args.verify_recorded:
        parser.error("this public harness intentionally supports --verify-recorded only; see README for bounded live-reproduction prerequisites")
    print(json.dumps(verify(args.case), sort_keys=True))

if __name__ == "__main__":
    main()
