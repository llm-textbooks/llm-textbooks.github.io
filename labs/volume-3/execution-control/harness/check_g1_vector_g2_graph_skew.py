#!/usr/bin/env python3
"""vector generation과 graph generation이 어긋난 상태의 admission fixture.

무엇을 검증하는가:
  vector projection이 g1에 머문 채 graph/ACL이 g2로 교체된 혼합 상태를
  in-process로 재현한다. 요청 generation과 후보·ACL의 generation이 다르면
  admission이 allow도 deny도 아닌 GENERATION_SKEW 보류를 반환하고 effect가
  하나도 생기지 않는지 확인한다. 이어서 같은 generation(g2)으로 다시 조회하면
  정상 판정(doc-2 allow)이 남는지 확인한다. 반증 대상으로, generation label을
  무시하고 "어느 세대에서든 vector_hit이고 어느 세대에서든 graph_allow"만 보는
  gate를 실제로 실행해 (a) g2에서 철회된 doc-1을 승격시키고 (b) 어떤 단일
  snapshot에서도 성립한 적 없는 vector g1 + graph g2 조합으로 doc-3을
  commit한다는 사실을 "counterexample:" 이벤트로 남긴다.

보장하지 않는 것:
  실제 vector store(HNSW/exact), 실제 SPARQL/SHACL 엔진, 실제 graph store의
  transaction 의미론, 분산 색인 교체나 provider 동작을 관측한 결과가 아니다.
  45장이 서술한 범위(generation fence, inventory completeness, 선언한 shape)
  안에서만 계약을 모델링한 결정적 시뮬레이션이다. 자기 event ledger 파일 외의
  I/O·시간·난수를 쓰지 않는다.

관련 장:
  45장(45.4 generation 정합성, 45.8 반례 2),
  43장(43.7 graph generation과 vector generation을 한 숫자로 합치지 않는다).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

FIXTURE = "g1-vector-g2-graph-skew"
EXPECTED_SHA256 = "b2b2d1284540e14e202de537727912d0ba798c780d4bea66d13256250d10e8c1"

LEDGER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "recorded-events",
        FIXTURE + ".events.jsonl",
    )
)

RUN_ID = "run-ec-generation-skew-001"
TURN_ID = "turn-1"
PRINCIPAL = "principal:agent-desk-7"
TENANT = "tenant:acme"
ACTION = "document:cite"
REQUESTED_GENERATION = "g2"

VECTOR_INDEX = {
    "g1": ["doc-1", "doc-3"],
    "g2": ["doc-2"],
}
GRAPH_ACL = {
    "g1": {"doc-1": True, "doc-3": False},
    "g2": {"doc-1": False, "doc-2": True, "doc-3": True},
}
INVENTORY_COMPLETE = {"g1": True, "g2": True}


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


class KnowledgePlane:
    """vector projection과 graph ACL을 세대별로 분리해 두는 결정적 모델."""

    def __init__(self, vector_index, graph_acl, inventory_complete) -> None:
        self.vector_index = {gen: list(docs) for gen, docs in vector_index.items()}
        self.graph_acl = {gen: dict(acl) for gen, acl in graph_acl.items()}
        self.inventory_complete = dict(inventory_complete)

    def search(self, generation: str) -> list[str]:
        return list(self.vector_index[generation])

    def vector_hit(self, doc_id: str, generation: str) -> bool:
        return doc_id in self.vector_index[generation]

    def graph_allow(self, doc_id: str, generation: str) -> bool:
        return bool(self.graph_acl[generation].get(doc_id, False))

    def any_vector_hit(self, doc_id: str):
        for generation in sorted(self.vector_index):
            if self.vector_hit(doc_id, generation):
                return generation
        return None

    def any_graph_allow(self, doc_id: str):
        for generation in sorted(self.graph_acl):
            if self.graph_allow(doc_id, generation):
                return generation
        return None


class Receiver:
    """effect commit과 receipt 발급의 in-process 모델."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.receipts: list[dict] = []

    def commit(self, doc_id: str, logical_call_id: str, evidence: dict) -> dict:
        receipt = {
            "receipt_id": "rcpt-%s-%d" % (self.name, len(self.receipts) + 1),
            "logical_call_id": logical_call_id,
            "doc_id": doc_id,
            "evidence": evidence,
        }
        self.receipts.append(receipt)
        return receipt


def fenced_admission(plane: KnowledgePlane, doc_id: str, candidate_generation: str,
                     graph_generation: str, requested_generation: str) -> dict:
    """45.4의 술어를 그대로 구현한다. 세대가 어긋나면 판정하지 않고 보류한다."""
    if candidate_generation != requested_generation or graph_generation != requested_generation:
        return {
            "disposition": "GENERATION_SKEW",
            "reason": "requested_generation_not_equal_to_evidence_generation",
            "vector_generation": candidate_generation,
            "graph_generation": graph_generation,
            "requested_generation": requested_generation,
        }
    if not plane.inventory_complete[requested_generation]:
        return {"disposition": "UNKNOWN", "reason": "inventory_completeness_undeclared"}
    if not plane.vector_hit(doc_id, requested_generation):
        return {"disposition": "DENY", "reason": "not_a_candidate_at_requested_generation"}
    if not plane.graph_allow(doc_id, requested_generation):
        return {"disposition": "DENY", "reason": "graph_acl_denies_at_requested_generation"}
    return {"disposition": "ALLOW", "reason": "vector_hit_and_graph_allow_at_same_generation"}


def generation_blind_admission(plane: KnowledgePlane, doc_id: str) -> dict:
    """반증 대상: generation label을 버리고 두 술어의 존재만 확인한다."""
    hit_generation = plane.any_vector_hit(doc_id)
    allow_generation = plane.any_graph_allow(doc_id)
    if hit_generation is None or allow_generation is None:
        return {"disposition": "DENY", "reason": "predicate_missing_in_every_generation"}
    return {
        "disposition": "ALLOW",
        "reason": "vector_hit_and_graph_allow_ignoring_generation",
        "vector_generation": hit_generation,
        "graph_generation": allow_generation,
        "snapshot_ever_existed": hit_generation == allow_generation,
    }


def run_simulation() -> tuple[Ledger, list[str]]:
    ledger = Ledger(FIXTURE)
    plane = KnowledgePlane(VECTOR_INDEX, GRAPH_ACL, INVENTORY_COMPLETE)
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
        requested_generation=REQUESTED_GENERATION,
    )
    for generation in ("g1", "g2"):
        ledger.record(
            "index",
            "vector_generation_published",
            vector_generation=generation,
            indexed_docs=plane.vector_index[generation],
        )
    for generation in ("g1", "g2"):
        ledger.record(
            "graph",
            "graph_generation_published",
            graph_generation=generation,
            acl={doc: plane.graph_allow(doc, generation) for doc in sorted(plane.graph_acl[generation])},
            inventory_complete=plane.inventory_complete[generation],
        )

    stale_candidates = plane.search("g1")
    ledger.record(
        "retrieval",
        "candidates_returned",
        vector_generation="g1",
        graph_generation="g2",
        requested_generation=REQUESTED_GENERATION,
        candidate_ids=stale_candidates,
        note="vector_projection_lags_behind_graph",
    )

    skew_dispositions = []
    for doc_id in stale_candidates:
        verdict = fenced_admission(plane, doc_id, "g1", "g2", REQUESTED_GENERATION)
        skew_dispositions.append(verdict["disposition"])
        ledger.record(
            "admission",
            "generation_skew_hold",
            doc_id=doc_id,
            disposition=verdict["disposition"],
            reason=verdict["reason"],
            vector_generation=verdict["vector_generation"],
            graph_generation=verdict["graph_generation"],
            requested_generation=verdict["requested_generation"],
            next_action="refetch_at_same_generation_or_hold",
        )
    expect(
        skew_dispositions == ["GENERATION_SKEW"] * len(stale_candidates),
        "혼합 상태에서 admission이 GENERATION_SKEW로 보류하지 않았다: %r" % (skew_dispositions,),
    )

    ledger.record(
        "effect",
        "no_attempt",
        phase="skewed_candidates",
        attempts=0,
        receipts=len(fenced_receiver.receipts),
    )
    expect(fenced_receiver.receipts == [], "보류 상태에서 receipt가 발급됐다")

    same_generation_candidates = plane.search(REQUESTED_GENERATION)
    ledger.record(
        "retrieval",
        "candidates_returned",
        vector_generation=REQUESTED_GENERATION,
        graph_generation=REQUESTED_GENERATION,
        requested_generation=REQUESTED_GENERATION,
        candidate_ids=same_generation_candidates,
        note="refetched_at_same_generation",
    )

    admitted = []
    for doc_id in same_generation_candidates:
        verdict = fenced_admission(
            plane, doc_id, REQUESTED_GENERATION, REQUESTED_GENERATION, REQUESTED_GENERATION
        )
        ledger.record(
            "admission",
            verdict["disposition"].lower(),
            doc_id=doc_id,
            disposition=verdict["disposition"],
            reason=verdict["reason"],
            vector_generation=REQUESTED_GENERATION,
            graph_generation=REQUESTED_GENERATION,
            requested_generation=REQUESTED_GENERATION,
        )
        if verdict["disposition"] == "ALLOW":
            admitted.append(doc_id)
    expect(admitted == ["doc-2"], "동일 generation 재조회에서 doc-2만 허용되지 않았다: %r" % (admitted,))

    for doc_id in admitted:
        receipt = fenced_receiver.commit(
            doc_id,
            "call-cite-%s" % doc_id,
            {"vector_generation": REQUESTED_GENERATION, "graph_generation": REQUESTED_GENERATION},
        )
        ledger.record(
            "effect",
            "receipt_committed",
            doc_id=doc_id,
            receipt_id=receipt["receipt_id"],
            logical_call_id=receipt["logical_call_id"],
            vector_generation=REQUESTED_GENERATION,
            graph_generation=REQUESTED_GENERATION,
        )
    expect(len(fenced_receiver.receipts) == 1, "동일 generation 판정에서 receipt가 하나가 아니다")

    blind_allowed = []
    for doc_id in ("doc-1", "doc-2", "doc-3"):
        verdict = generation_blind_admission(plane, doc_id)
        if verdict["disposition"] != "ALLOW":
            continue
        blind_allowed.append(doc_id)
        ledger.record(
            "counterexample:admission",
            "allow",
            gate="generation_blind_conjunction",
            doc_id=doc_id,
            reason=verdict["reason"],
            vector_generation=verdict["vector_generation"],
            graph_generation=verdict["graph_generation"],
            requested_generation=REQUESTED_GENERATION,
            snapshot_ever_existed=verdict["snapshot_ever_existed"],
            fenced_disposition=fenced_admission(
                plane, doc_id, verdict["vector_generation"], verdict["graph_generation"],
                REQUESTED_GENERATION,
            )["disposition"],
        )
        receipt = blind_receiver.commit(
            doc_id,
            "call-cite-%s" % doc_id,
            {
                "vector_generation": verdict["vector_generation"],
                "graph_generation": verdict["graph_generation"],
            },
        )
        ledger.record(
            "counterexample:effect",
            "receipt_committed",
            gate="generation_blind_conjunction",
            doc_id=doc_id,
            receipt_id=receipt["receipt_id"],
            evidence_vector_generation=verdict["vector_generation"],
            evidence_graph_generation=verdict["graph_generation"],
            snapshot_ever_existed=verdict["snapshot_ever_existed"],
        )

    expect(
        blind_allowed == ["doc-1", "doc-2", "doc-3"],
        "반증 gate가 doc-1/doc-3을 승격시키지 않아 대비가 성립하지 않는다: %r" % (blind_allowed,),
    )
    mixed = [
        receipt["doc_id"]
        for receipt in blind_receiver.receipts
        if receipt["evidence"]["vector_generation"] != receipt["evidence"]["graph_generation"]
    ]
    expect(mixed == ["doc-3"], "혼합 snapshot commit이 doc-3 하나가 아니다: %r" % (mixed,))
    expect(
        not plane.graph_allow("doc-1", REQUESTED_GENERATION),
        "doc-1이 g2에서 철회된 상태가 아니다",
    )
    expect(
        not plane.vector_hit("doc-3", REQUESTED_GENERATION)
        and not plane.graph_allow("doc-3", "g1"),
        "doc-3이 어떤 단일 generation에서도 admissible하지 않은 상태가 아니다",
    )

    ledger.record(
        "oracle",
        "contrast_recorded",
        fenced_receipts=len(fenced_receiver.receipts),
        fenced_admitted=admitted,
        blind_receipts=len(blind_receiver.receipts),
        blind_admitted=blind_allowed,
        blind_mixed_snapshot_docs=mixed,
        skew_holds=len(stale_candidates),
    )

    refuted = [
        "vector_hit과 graph_allow가 각각 참이면 세대가 달라도 승인해도 된다 (g2에서 철회된 doc-1이 g1 ACL로 승격된다)",
        "generation을 섞어 판정해도 결국 같은 그래프다 (vector g1 + graph g2 조합은 존재한 적 없는 snapshot이며 doc-3을 commit시킨다)",
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
