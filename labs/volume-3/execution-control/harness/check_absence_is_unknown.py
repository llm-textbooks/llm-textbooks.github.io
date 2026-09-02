#!/usr/bin/env python3
"""빈 결과가 FALSE가 아니라 UNKNOWN인 경계를 세우는 fixture.

무엇을 검증하는가:
  `NOT EXISTS` 형태의 부정 질의가 빈 결과를 돌려줬을 때, inventory
  completeness를 선언하지 않은 collection에서는 결론이 UNKNOWN이고,
  owner·coverage window·predicate·tenant·source revision을 선언한
  collection에서만 FALSE(부정 결론 허용)가 되는지를 in-process로 재현한다.
  선언이 있어도 질의 predicate가 선언 범위 밖이면 다시 UNKNOWN이 된다는
  분기까지 확인한다. UNKNOWN은 실패 종료가 아니라 분기 입력이므로,
  권위 있는 2차 조회로 해소하는 경로와 조회처가 없어 보류하는 경로를
  각각 이벤트로 남긴다. 반증 대상으로 빈 결과를 무조건 FALSE로 읽는
  closed-world gate를 실제로 실행해, 아직 적재되지 않은 항목에 대한
  부정 주장으로 effect가 commit된다는 사실을 "counterexample:" 접두
  이벤트로 남긴다.

보장하지 않는 것:
  실제 SPARQL 엔진, 실제 SHACL 엔진, 실제 triple store나 분산 색인의
  동작을 관측한 결과가 아니다. RDF의 open-world 의미론 전체를 구현하지도
  않는다. 45장이 서술한 범위(선언한 shape, inventory completeness,
  generation fence)만 순수 표준 라이브러리로 모델링한 결정적 시뮬레이션이며,
  여기서 통과한다고 운영 graph의 적재 누락·watermark 지연이 안전해지지 않는다.
  자기 event ledger 파일 외의 I/O·시간·난수를 쓰지 않는다.

관련 장:
  45장(45.5 부정 질의는 데이터 완전성 계약을 요구한다, 45.8 반례 3),
  43장(43.6 검색 cache는 답 cache가 아니라 후보 cache다).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

FIXTURE = "absence-is-unknown"
EXPECTED_SHA256 = "ebe8eb29a2f43d9a56c937b29667ac2b7c678f948cf6bed6ab099953e18a0481"

LEDGER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "recorded-events",
        FIXTURE + ".events.jsonl",
    )
)

RUN_ID = "run-ec-absence-unknown-001"
TURN_ID = "turn-3"
PRINCIPAL = "principal:agent-desk-7"
TENANT = "tenant:acme"
ACTION = "purchase_order:auto_approve"
GRAPH_GENERATION = "g7"

REGISTRY = "col-vendor-registry"
CONTRACTS = "col-partner-contracts"

# collection에 실제로 적재된 triple.
LOADED_TRIPLES = {
    REGISTRY: [
        ("vendor:v-12", "vendor:approvedFor", TENANT),
        ("vendor:v-88", "vendor:approvedFor", TENANT),
        ("vendor:v-40", "vendor:unapprovedDependency", "lib:agpl-codec"),
    ],
    CONTRACTS: [
        ("vendor:v-12", "contract:signedWith", TENANT),
    ],
}

# 세상에는 있으나 이 collection에 적재된 적 없는 사실. 질의는 이것을 볼 수 없다.
# closed-world gate의 부정 결론이 왜 사실과 어긋나는지 보이기 위한 ground truth일 뿐,
# 어떤 질의 경로도 이 표를 조회 근거로 쓰지 않는다.
UNINGESTED_REALITY = {
    CONTRACTS: [
        ("vendor:v-88", "vendor:unapprovedDependency", "lib:gpl-tool"),
    ],
}

# inventory completeness 선언. 선언이 없는 collection은 키 자체가 없다.
COMPLETENESS_DECLARATIONS = {
    REGISTRY: {
        "inventory_owner": "team:procurement-ops",
        "coverage_window": "2026-01-01/2026-08-31",
        "complete_predicates": ["vendor:approvedFor", "vendor:unapprovedDependency"],
        "complete_for_tenant": TENANT,
        "source_revision": "reg:r91",
        "graph_generation": GRAPH_GENERATION,
        "ingestion_watermark": "reg:r91",
        "tombstones_applied": True,
    },
}

# UNKNOWN을 해소할 수 있는 권위 있는 2차 조회처. 선언이 있는 것만 신뢰한다.
AUTHORITATIVE_PROBES = {
    CONTRACTS: {
        "probe_id": "probe:partner-contract-registry",
        "inventory_owner": "team:legal-ops",
        "complete_predicates": ["vendor:unapprovedDependency"],
        "complete_for_tenant": TENANT,
        "source_revision": "pcr:r14",
        "facts": [
            ("vendor:v-88", "vendor:unapprovedDependency", "lib:gpl-tool"),
        ],
    },
}


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


class GraphStore:
    """collection별 적재 triple과 completeness 선언을 담는 결정적 모델."""

    def __init__(self, loaded, declarations, uningested) -> None:
        self.loaded = {name: list(rows) for name, rows in loaded.items()}
        self.declarations = {name: dict(d) for name, d in declarations.items()}
        self.uningested = {name: list(rows) for name, rows in uningested.items()}

    def match(self, collection: str, subject: str, predicate: str) -> list:
        return [
            list(triple)
            for triple in self.loaded[collection]
            if triple[0] == subject and triple[1] == predicate
        ]

    def declaration(self, collection: str):
        return self.declarations.get(collection)

    def covers(self, collection: str, predicate: str, tenant: str):
        """선언이 이 predicate와 tenant를 완전하다고 덮는지 판정한다."""
        declaration = self.declaration(collection)
        if declaration is None:
            return False, "inventory_completeness_undeclared"
        if declaration["complete_for_tenant"] != tenant:
            return False, "tenant_outside_declared_completeness"
        if predicate not in declaration["complete_predicates"]:
            return False, "predicate_outside_declared_completeness"
        return True, "declared_complete_for_predicate_and_tenant"

    def truly_exists(self, collection: str, subject: str, predicate: str) -> bool:
        """ground truth. 적재된 것과 적재되지 않은 것을 합쳐서 본다."""
        rows = self.loaded[collection] + self.uningested.get(collection, [])
        return any(t[0] == subject and t[1] == predicate for t in rows)


def open_world_existence(store: GraphStore, collection: str, subject: str,
                         predicate: str, tenant: str) -> dict:
    """45.5의 규칙. 빈 결과는 완전성 선언이 덮을 때만 FALSE가 된다."""
    bindings = store.match(collection, subject, predicate)
    if bindings:
        return {
            "truth_value": "TRUE",
            "reason": "matching_triples_present",
            "binding_count": len(bindings),
        }
    covered, reason = store.covers(collection, predicate, tenant)
    if not covered:
        return {"truth_value": "UNKNOWN", "reason": reason, "binding_count": 0}
    return {"truth_value": "FALSE", "reason": reason, "binding_count": 0}


def closed_world_existence(store: GraphStore, collection: str, subject: str,
                           predicate: str) -> dict:
    """반증 대상: 완전성 선언을 보지 않고 빈 결과를 곧바로 FALSE로 읽는다."""
    bindings = store.match(collection, subject, predicate)
    if bindings:
        return {"truth_value": "TRUE", "reason": "matching_triples_present"}
    return {"truth_value": "FALSE", "reason": "empty_result_treated_as_negation"}


class Receiver:
    """effect commit과 receipt 발급의 in-process 모델."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.receipts: list[dict] = []

    def commit(self, subject: str, negative_claim: str, evidence: dict) -> dict:
        receipt = {
            "receipt_id": "rcpt-%s-%d" % (self.name, len(self.receipts) + 1),
            "logical_call_id": "call-approve-%s" % subject.split(":")[-1],
            "subject": subject,
            "negative_claim": negative_claim,
            "evidence": evidence,
        }
        self.receipts.append(receipt)
        return receipt


CASES = [
    {
        "case_id": "case-1-undeclared-collection",
        "collection": CONTRACTS,
        "subject": "vendor:v-88",
        "predicate": "vendor:unapprovedDependency",
        "branch_action": "probe_authoritative_source",
    },
    {
        "case_id": "case-2-declared-collection",
        "collection": REGISTRY,
        "subject": "vendor:v-12",
        "predicate": "vendor:unapprovedDependency",
        "branch_action": "none",
    },
    {
        "case_id": "case-3-predicate-outside-declaration",
        "collection": REGISTRY,
        "subject": "vendor:v-12",
        "predicate": "vendor:sanctionedBy",
        "branch_action": "hold_no_authoritative_source",
    },
]

NEGATIVE_CLAIM = "no_unapproved_dependency_for_subject"


def run_simulation() -> tuple[Ledger, list[str]]:
    ledger = Ledger(FIXTURE)
    store = GraphStore(LOADED_TRIPLES, COMPLETENESS_DECLARATIONS, UNINGESTED_REALITY)
    fenced_receiver = Receiver("fenced")
    blind_receiver = Receiver("blind")

    ledger.record(
        "scope",
        "request_scoped",
        run_id=RUN_ID,
        turn_id=TURN_ID,
        principal=PRINCIPAL,
        tenant=TENANT,
        action=ACTION,
        graph_generation=GRAPH_GENERATION,
        negative_claim=NEGATIVE_CLAIM,
    )
    for collection in (REGISTRY, CONTRACTS):
        declaration = store.declaration(collection)
        if declaration is None:
            ledger.record(
                "inventory",
                "completeness_undeclared",
                collection=collection,
                loaded_triples=len(store.loaded[collection]),
                note="open_world_default_no_negative_conclusion",
            )
        else:
            ledger.record(
                "inventory",
                "completeness_declared",
                collection=collection,
                loaded_triples=len(store.loaded[collection]),
                inventory_owner=declaration["inventory_owner"],
                coverage_window=declaration["coverage_window"],
                complete_predicates=declaration["complete_predicates"],
                complete_for_tenant=declaration["complete_for_tenant"],
                source_revision=declaration["source_revision"],
                ingestion_watermark=declaration["ingestion_watermark"],
                tombstones_applied=declaration["tombstones_applied"],
            )

    truth_values = []
    unknown_reasons = []
    for case in CASES:
        verdict = open_world_existence(
            store, case["collection"], case["subject"], case["predicate"], TENANT
        )
        truth_values.append(verdict["truth_value"])
        ledger.record(
            "query",
            "not_exists_evaluated",
            case_id=case["case_id"],
            collection=case["collection"],
            subject=case["subject"],
            predicate=case["predicate"],
            binding_count=verdict["binding_count"],
            existence_truth_value=verdict["truth_value"],
            reason=verdict["reason"],
            negative_conclusion_permitted=verdict["truth_value"] == "FALSE",
        )

        if verdict["truth_value"] == "FALSE":
            ledger.record(
                "admission",
                "allow",
                case_id=case["case_id"],
                negative_claim=NEGATIVE_CLAIM,
                basis="declared_inventory_completeness",
                declaring_owner=store.declaration(case["collection"])["inventory_owner"],
                source_revision=store.declaration(case["collection"])["source_revision"],
            )
            receipt = fenced_receiver.commit(
                case["subject"],
                NEGATIVE_CLAIM,
                {
                    "collection": case["collection"],
                    "completeness_declared": True,
                    "graph_generation": GRAPH_GENERATION,
                },
            )
            ledger.record(
                "effect",
                "receipt_committed",
                case_id=case["case_id"],
                receipt_id=receipt["receipt_id"],
                logical_call_id=receipt["logical_call_id"],
                subject=case["subject"],
            )
            continue

        unknown_reasons.append(verdict["reason"])
        ledger.record(
            "branch",
            "unknown_routed",
            case_id=case["case_id"],
            unknown_reason=verdict["reason"],
            unknown_is_failure=False,
            branch_action=case["branch_action"],
            note="UNKNOWN_is_a_branch_input_not_an_error_exit",
        )

        probe = AUTHORITATIVE_PROBES.get(case["collection"])
        if case["branch_action"] == "probe_authoritative_source" and probe is not None:
            hits = [
                list(fact)
                for fact in probe["facts"]
                if fact[0] == case["subject"] and fact[1] == case["predicate"]
            ]
            ledger.record(
                "probe",
                "authoritative_source_answered",
                case_id=case["case_id"],
                probe_id=probe["probe_id"],
                inventory_owner=probe["inventory_owner"],
                source_revision=probe["source_revision"],
                binding_count=len(hits),
                existence_truth_value="TRUE" if hits else "FALSE",
                resolves_unknown=True,
            )
            ledger.record(
                "admission",
                "deny",
                case_id=case["case_id"],
                negative_claim=NEGATIVE_CLAIM,
                reason="authoritative_probe_found_unapproved_dependency"
                if hits
                else "authoritative_probe_confirmed_absence",
                probe_id=probe["probe_id"],
            )
        else:
            ledger.record(
                "admission",
                "hold",
                case_id=case["case_id"],
                negative_claim=NEGATIVE_CLAIM,
                disposition="UNKNOWN",
                reason=verdict["reason"],
                next_action="declare_completeness_or_escalate_to_human",
            )

        ledger.record(
            "effect",
            "no_attempt",
            case_id=case["case_id"],
            attempts=0,
            receipts=len(fenced_receiver.receipts),
        )

    expect(
        truth_values == ["UNKNOWN", "FALSE", "UNKNOWN"],
        "빈 결과의 판정이 UNKNOWN/FALSE/UNKNOWN이 아니다: %r" % (truth_values,),
    )
    expect(
        unknown_reasons
        == ["inventory_completeness_undeclared", "predicate_outside_declared_completeness"],
        "UNKNOWN 사유가 두 분기로 구분되지 않았다: %r" % (unknown_reasons,),
    )
    expect(
        [r["subject"] for r in fenced_receiver.receipts] == ["vendor:v-12"],
        "선언된 collection의 부정 결론 하나만 commit되지 않았다: %r"
        % ([r["subject"] for r in fenced_receiver.receipts],),
    )

    blind_truth_values = []
    contradicted = []
    for case in CASES:
        verdict = closed_world_existence(
            store, case["collection"], case["subject"], case["predicate"]
        )
        blind_truth_values.append(verdict["truth_value"])
        if verdict["truth_value"] != "FALSE":
            continue
        really_exists = store.truly_exists(
            case["collection"], case["subject"], case["predicate"]
        )
        if really_exists:
            contradicted.append(case["case_id"])
        ledger.record(
            "counterexample:admission",
            "allow",
            gate="closed_world_empty_result_is_false",
            case_id=case["case_id"],
            collection=case["collection"],
            subject=case["subject"],
            predicate=case["predicate"],
            reason=verdict["reason"],
            completeness_declared=store.declaration(case["collection"]) is not None,
            open_world_truth_value=open_world_existence(
                store, case["collection"], case["subject"], case["predicate"], TENANT
            )["truth_value"],
            contradicts_uningested_fact=really_exists,
        )
        receipt = blind_receiver.commit(
            case["subject"],
            NEGATIVE_CLAIM,
            {"collection": case["collection"], "completeness_declared": False},
        )
        ledger.record(
            "counterexample:effect",
            "receipt_committed",
            gate="closed_world_empty_result_is_false",
            case_id=case["case_id"],
            receipt_id=receipt["receipt_id"],
            subject=case["subject"],
            negative_claim=NEGATIVE_CLAIM,
            contradicts_uningested_fact=really_exists,
        )

    expect(
        blind_truth_values == ["FALSE", "FALSE", "FALSE"],
        "closed-world gate가 세 경우 모두 FALSE로 읽지 않아 대비가 성립하지 않는다: %r"
        % (blind_truth_values,),
    )
    expect(
        contradicted == ["case-1-undeclared-collection"],
        "적재되지 않은 사실과 충돌한 commit이 case-1 하나가 아니다: %r" % (contradicted,),
    )
    expect(
        len(blind_receiver.receipts) == 3 and len(fenced_receiver.receipts) == 1,
        "receipt 수 대비가 3 대 1이 아니다",
    )

    ledger.record(
        "oracle",
        "contrast_recorded",
        open_world_truth_values=truth_values,
        closed_world_truth_values=blind_truth_values,
        fenced_receipts=len(fenced_receiver.receipts),
        blind_receipts=len(blind_receiver.receipts),
        blind_receipts_contradicting_reality=contradicted,
        unknown_reasons=unknown_reasons,
    )

    refuted = [
        "질의 결과가 비었으면 그 사실은 존재하지 않는다 (완전성 선언이 없는 collection에서는 UNKNOWN이며, 적재되지 않은 dependency가 실제로 존재한다)",
        "completeness를 한 번 선언하면 그 collection의 모든 부정 질의가 FALSE다 (선언 범위 밖 predicate는 다시 UNKNOWN이다)",
        "UNKNOWN은 검색 실패이므로 무시하고 진행해도 된다 (UNKNOWN은 2차 조회나 보류로 가는 분기 입력이다)",
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
