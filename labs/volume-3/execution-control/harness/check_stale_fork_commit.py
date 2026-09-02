#!/usr/bin/env python3
"""stale fork의 join quarantine과 receiver compare-and-set 거절을 검증하는 fixture.

무엇을 검증하는가
  1. parent가 state_revision=s70, policy_generation=p12에서 child를 fork하고
     그 사이 parent가 s71/p13으로 진행하면, child 결과의 base_revision과 read set이
     parent가 바꾼 field와 겹치므로 join verifier는 promote하지 않고
     quarantine(stale_branch)한다(44.2의 join 의사코드 순서를 그대로 따른다).
  2. child가 join을 우회해 receiver에 직접 commit을 시도하면 receiver-side
     compare-and-set이 expected_revision=s70 vs current_revision=s71로 거절하며
     receipt를 만들지 않는다. 즉 방어선은 join 하나가 아니다.
  3. parent state_revision은 s71에 머무르고, 승격된 사실은 0건이다.
  4. 반증 oracle: 텍스트 유사도(Jaccard)를 merge key로 쓰는 join을 실제로 실행하면
     같은 child 결과가 s72로 승격되고, 승격된 사실이 현재 상태와 모순된다는 것을
     코드로 계산해 event로 남긴다.

이 fixture가 보장하지 않는 것
  - 실제 제품의 fork 구현, prompt cache 상속, 분산 branch merge를 재현하지 않는다.
    44장이 정의한 join 계약의 in-process 결정적 모델이다.
  - compare-and-set이 성공했다는 사실은 이 receiver 모델의 명세일 뿐, 임의 receiver가
    revision fence를 제공한다는 뜻이 아니다.
  - quarantine은 결과를 버리는 것이 아니라 승격을 막는 것이다. 재계획 정책은 범위 밖이다.

관련 장: 44장(서브에이전트와 목표) 44.2 fork/join 절, 44.8 장면 A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FIXTURE = "stale-fork-commit"
LEDGER_PATH = (
    Path(__file__).resolve().parent.parent / "recorded-events" / f"{FIXTURE}.events.jsonl"
)
EXPECTED_SHA256 = "58f2effd0d63a85ec971993f2d93ae17ceb134553c5d95d88bded3df557806ac"

REFUTED = [
    "text 유사도를 merge key로 쓰면 stale branch의 결론이 최신 상태를 덮어쓴다",
    "join을 우회한 child의 직접 commit은 receiver revision fence가 없으면 그대로 적용된다",
]

SIMILARITY_THRESHOLD = 0.5


class OracleError(Exception):
    """계약 위반을 알리는 예외."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


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


class DeploymentReceiver:
    """expected_revision을 compare-and-set으로 검사하는 결정적 receiver 모델."""

    def __init__(self, current_revision: str) -> None:
        self.current_revision = current_revision
        self.receipts: list[dict[str, object]] = []

    def commit(
        self, effect_key: str, expected_revision: str, payload: dict[str, object]
    ) -> dict[str, object]:
        if expected_revision != self.current_revision:
            return {
                "status": "cas_rejected",
                "expected_revision": expected_revision,
                "current_revision": self.current_revision,
                "receipt": None,
            }
        receipt = {
            "receipt_id": f"r-{len(self.receipts) + 1}",
            "effect_key": effect_key,
            "committed_revision": self.current_revision,
            "payload": payload,
        }
        self.receipts.append(receipt)
        return {"status": "applied", "receipt": receipt}


def tokens(text: str) -> frozenset[str]:
    return frozenset(word for word in text.lower().replace(",", " ").split() if word)


def jaccard(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compatible(base_revision: str, current_revision: str, read_set: list[str], changed: list[str]) -> bool:
    """44.2 join predicate: base가 같거나, parent 변경 field가 read set과 겹치지 않아야 한다."""
    if base_revision == current_revision:
        return True
    return not (set(read_set) & set(changed))


def join_child(parent: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    """44.2의 join_child 의사코드를 그대로 옮긴 판정기. 순서가 곧 quarantine reason이다."""
    if result["parent_run_id"] != parent["run_id"]:
        return {"decision": "quarantine", "reason": "foreign_parent"}
    if not compatible(
        str(result["base_state_revision"]),
        str(parent["state_revision"]),
        list(result["read_set"]),  # type: ignore[arg-type]
        list(parent["changed_fields"]),  # type: ignore[arg-type]
    ):
        return {"decision": "quarantine", "reason": "stale_branch"}
    if result["policy_generation"] != parent["policy_generation"]:
        return {"decision": "quarantine", "reason": "policy_changed"}
    return {"decision": "promote", "reason": "compatible"}


def join_by_text_similarity(parent_goal_text: str, result: dict[str, object]) -> dict[str, object]:
    """반증 대상 join: base revision을 보지 않고 문장 유사도만 본다.

    44.8 장면 A의 "text similarity는 freshness 판정이 아니다"를 코드로 실행한다.
    """
    score = jaccard(parent_goal_text, str(result["claim_text"]))
    if score >= SIMILARITY_THRESHOLD:
        return {"decision": "promote", "reason": "similarity_above_threshold", "similarity": score}
    return {"decision": "quarantine", "reason": "similarity_below_threshold", "similarity": score}


def simulate() -> Ledger:
    ledger = Ledger(FIXTURE)

    parent = {
        "run_id": "run-main-7",
        "goal_id": "g-scale-canary",
        "state_revision": "s70",
        "policy_generation": "p12",
        "state": {"replicas": 3, "canary_percent": 10},
        "changed_fields": [],
    }
    ledger.emit(
        "fork-admission",
        "child_forked",
        root_run_id=parent["run_id"],
        branch_id="branch-research-1",
        parent_run_id=parent["run_id"],
        base_state_revision=parent["state_revision"],
        policy_generation=parent["policy_generation"],
        scope="read_only",
        read_set=["replicas"],
    )

    # parent가 자기 branch에서 진행한다. child는 이 변경을 보지 못한다.
    parent = {
        **parent,
        "state_revision": "s71",
        "policy_generation": "p13",
        "state": {"replicas": 5, "canary_percent": 10},
        "changed_fields": ["replicas"],
    }
    ledger.emit(
        "parent-advance",
        "parent_advanced",
        run_id=parent["run_id"],
        state_revision=parent["state_revision"],
        policy_generation=parent["policy_generation"],
        changed_fields=parent["changed_fields"],
        replicas=parent["state"]["replicas"],
    )

    child_result = {
        "parent_run_id": "run-main-7",
        "branch_id": "branch-research-1",
        "base_state_revision": "s70",
        "policy_generation": "p12",
        "read_set": ["replicas"],
        "observed_replicas": 3,
        "claim_text": "scale replicas to 4 is safe because current replicas is 3",
    }
    ledger.emit(
        "child-terminal",
        "candidate_returned",
        branch_id=child_result["branch_id"],
        base_state_revision=child_result["base_state_revision"],
        policy_generation=child_result["policy_generation"],
        read_set=child_result["read_set"],
        observed_replicas=child_result["observed_replicas"],
        promoted=False,
    )

    overlap = sorted(set(child_result["read_set"]) & set(parent["changed_fields"]))
    check(overlap == ["replicas"], f"the fixture requires an overlapping read set: {overlap}")
    ledger.emit(
        "join-compatibility",
        "incompatible_read_set",
        base_state_revision=child_result["base_state_revision"],
        current_state_revision=parent["state_revision"],
        read_set=child_result["read_set"],
        changed_fields=parent["changed_fields"],
        overlap=overlap,
    )

    verdict = join_child(parent, child_result)
    check(verdict["decision"] == "quarantine", "a stale child result must not be promoted")
    check(
        verdict["reason"] == "stale_branch",
        f"the first failing predicate must be staleness, got {verdict['reason']}",
    )
    ledger.emit(
        "join-quarantine",
        "stale_branch",
        branch_id=child_result["branch_id"],
        quarantine_reason=verdict["reason"],
        promoted=False,
    )
    ledger.emit(
        "join-policy-observation",
        "policy_generation_mismatch",
        child_policy_generation=child_result["policy_generation"],
        parent_policy_generation=parent["policy_generation"],
        note="second_failing_predicate_not_the_quarantine_reason",
        promoted=False,
    )

    # child가 join을 우회해 receiver에 직접 commit을 시도한다.
    receiver = DeploymentReceiver(current_revision=str(parent["state_revision"]))
    direct = receiver.commit(
        effect_key="ek-branch-research-1",
        expected_revision=str(child_result["base_state_revision"]),
        payload={"replicas": 4},
    )
    check(direct["status"] == "cas_rejected", "receiver CAS must reject a stale expected_revision")
    check(direct["receipt"] is None, "a rejected commit must not produce a receipt")
    ledger.emit(
        "receiver-compare-and-set",
        "cas_rejected",
        effect_key="ek-branch-research-1",
        expected_revision=direct["expected_revision"],
        current_revision=direct["current_revision"],
        bypassed_join=True,
        receipt_id=None,
    )
    ledger.emit(
        "receiver-audit",
        "no_receipt_written",
        effect_key="ek-branch-research-1",
        receipts=len(receiver.receipts),
    )
    ledger.emit(
        "parent-state-audit",
        "parent_revision_unchanged",
        state_revision=parent["state_revision"],
        promoted_facts=0,
        replicas=parent["state"]["replicas"],
    )

    # ---- 반증 경로: text 유사도 기반 merge ----
    parent_goal_text = "is it safe to scale replicas to 4 given current replicas"
    naive = join_by_text_similarity(parent_goal_text, child_result)
    check(
        naive["decision"] == "promote",
        f"the similarity join must promote, otherwise nothing is refuted: {naive}",
    )
    ledger.emit(
        "counterexample:text-similarity-join",
        "similarity_above_threshold",
        similarity=round(float(naive["similarity"]), 6),
        threshold=SIMILARITY_THRESHOLD,
        parent_goal_text=parent_goal_text,
        claim_text=child_result["claim_text"],
        inspected_base_revision=False,
        inspected_read_set=False,
    )
    promoted_state_revision = "s72"
    ledger.emit(
        "counterexample:promote",
        "stale_conclusion_promoted",
        branch_id=child_result["branch_id"],
        promoted=True,
        state_revision=promoted_state_revision,
        promoted_observed_replicas=child_result["observed_replicas"],
    )
    contradiction = child_result["observed_replicas"] != parent["state"]["replicas"]
    check(contradiction, "the promoted stale fact must contradict the current state")
    ledger.emit(
        "counterexample:contradiction",
        "promoted_fact_contradicts_current_state",
        promoted_observed_replicas=child_result["observed_replicas"],
        current_replicas=parent["state"]["replicas"],
        current_state_revision=parent["state_revision"],
    )

    naive_receiver = DeploymentReceiver(current_revision=str(parent["state_revision"]))
    unfenced = {
        "receipt_id": "r-unfenced-1",
        "effect_key": "ek-branch-research-1",
        "payload": {"replicas": 4},
    }
    naive_receiver.receipts.append(unfenced)
    ledger.emit(
        "counterexample:unfenced-receiver",
        "stale_write_applied",
        effect_key="ek-branch-research-1",
        receipt_id=unfenced["receipt_id"],
        written_replicas=4,
        current_replicas=parent["state"]["replicas"],
        compare_and_set=False,
    )

    ledger.emit(
        "oracle-refutation",
        "refuted",
        fenced_receipts=len(receiver.receipts),
        unfenced_receipts=len(naive_receiver.receipts),
        join_decision=verdict["reason"],
        similarity_join_decision=naive["reason"],
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
