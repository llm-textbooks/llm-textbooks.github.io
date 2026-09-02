#!/usr/bin/env python3
"""구조적 합치와 사실적 지지를 분리하는 fixture.

무엇을 검증하는가:
  선언한 shape의 constraint를 모두 통과해 conforms=true를 받은 claim이,
  원문에서 정정·철회됐다는 이유로 admission의 provenance/freshness 단계에서
  거절되는 순서를 in-process로 재현한다. 같은 이벤트 줄에 conforms=true와
  admission deny가 공존하도록 기록해 둘이 다른 판정임을 남긴다. 또한 shape에
  선언되지 않은 predicate는 애초에 검사 대상이 아니므로 conforms 판정에
  포함되지 않는다는 점(45장 "선언한 shape 범위" 원칙)을 unvalidated_paths로
  드러낸다. 대비를 위해 conform하면서 provenance/freshness도 통과하는 claim은
  실제로 commit되고, 구조적으로 어긋난 claim은 shape 단계에서 거절된다.
  반증 대상으로 SHACL pass를 사실성 증명으로 확대하는 gate를 실제로 실행해,
  철회된 claim이 commit된다는 사실을 "counterexample:" 접두 이벤트로 남긴다.

보장하지 않는 것:
  실제 SHACL 엔진(sh:node, sh:sparql, path 표현식 전체)이나 실제 SPARQL
  엔진, 실제 RDF store를 실행한 결과가 아니다. 여기서 구현한 것은 property
  shape의 min/max count, datatype, sh:in, sh:pattern에 해당하는 극히 일부이며,
  45장이 서술한 범위(선언한 shape, open world, inventory completeness)만
  모델링한 결정적 시뮬레이션이다. 여기서 통과한다고 운영 graph의 사실성이
  검증되지 않는다. 자기 event ledger 파일 외의 I/O·시간·난수를 쓰지 않는다.

관련 장:
  45장(45.5 SHACL의 경계, 45.8 반례 6),
  43장(43.5 도구와 스키마 cache: JSON Schema가 허가증은 아니다).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

FIXTURE = "shacl-pass-is-not-truth"
EXPECTED_SHA256 = "2c41e2e0fc3c5f81685a37e90ec4a0d12e0d5f65955e1b533c2968a0904e6d17"

LEDGER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "recorded-events",
        FIXTURE + ".events.jsonl",
    )
)

RUN_ID = "run-ec-shacl-not-truth-001"
TURN_ID = "turn-6"
PRINCIPAL = "principal:agent-desk-7"
TENANT = "tenant:acme"
ACTION = "vendor_certification:cite"
DOCUMENT = "doc:vendor-31"
REVISION_R1 = "rev:vendor-31:r1"
REVISION_R2 = "rev:vendor-31:r2"

SHAPE = {
    "shape_id": "ex:VendorClaimShape",
    "target_class": "ex:VendorClaim",
    "properties": [
        {"path": "ex:subject", "min_count": 1, "max_count": 1, "datatype": "xsd:anyURI"},
        {
            "path": "ex:predicate",
            "min_count": 1,
            "max_count": 1,
            "datatype": "xsd:string",
            "in": ["ex:isCertified", "ex:hasAuditFinding"],
        },
        {"path": "ex:object", "min_count": 1, "max_count": 1, "datatype": "xsd:string"},
        {
            "path": "ex:sourceRevision",
            "min_count": 1,
            "max_count": 1,
            "datatype": "xsd:string",
            "pattern": r"^rev:[a-z0-9\-]+:r[0-9]+$",
        },
        {"path": "ex:sourceSpan", "min_count": 1, "max_count": 1, "datatype": "xsd:string"},
    ],
}

# 원문 revision과 span 본문. r2가 r1의 인증 문장을 정정한다.
SOURCE_REVISIONS = {
    REVISION_R1: {
        "document": DOCUMENT,
        "superseded_by": REVISION_R2,
        "spans": {
            "sp-7": "vendor-31의 보안 인증은 유효하다.",
            "sp-9": "vendor-31은 2026년 1분기 감사에서 지적 사항 2건을 받았다.",
        },
    },
    REVISION_R2: {
        "document": DOCUMENT,
        "superseded_by": None,
        "spans": {
            "sp-7": "vendor-31의 보안 인증은 취소되었다. 이전 기재는 정정한다.",
            "sp-9": "vendor-31은 2026년 1분기 감사에서 지적 사항 2건을 받았다.",
        },
    },
}
HEAD_REVISION = REVISION_R2

CLAIMS = [
    {
        "node_id": "claim:c-501",
        "rdf_type": "ex:VendorClaim",
        "properties": {
            "ex:subject": ["vendor:v-31"],
            "ex:predicate": ["ex:isCertified"],
            "ex:object": ["true"],
            "ex:sourceRevision": [REVISION_R1],
            "ex:sourceSpan": ["sp-7"],
            # shape에 선언되지 않은 predicate. 검사 대상이 아니다.
            "ex:certifiedUntil": ["2027-12-31"],
            "ex:confidenceNote": ["retrieval score 0.91"],
        },
        "supporting_text": "보안 인증은 유효하다",
    },
    {
        "node_id": "claim:c-502",
        "rdf_type": "ex:VendorClaim",
        "properties": {
            "ex:subject": ["vendor:v-31"],
            "ex:predicate": ["ex:hasAuditFinding"],
            "ex:object": ["2"],
            "ex:sourceRevision": [HEAD_REVISION],
            "ex:sourceSpan": ["sp-9"],
        },
        "supporting_text": "지적 사항 2건",
    },
    {
        "node_id": "claim:c-503",
        "rdf_type": "ex:VendorClaim",
        "properties": {
            "ex:subject": ["vendor:v-31"],
            "ex:predicate": ["ex:isCertified"],
            "ex:object": ["true"],
            "ex:sourceSpan": ["sp-7"],
        },
        "supporting_text": "보안 인증은 유효하다",
    },
]


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


def validate_shape(shape: dict, node: dict) -> dict:
    """선언한 property shape만 검사하는 최소 validator.

    선언되지 않은 path는 검사하지 않고 unvalidated_paths로 보고한다. 이것이
    open world에서 conforms=true가 뜻하는 범위다.
    """
    declared_paths = [prop["path"] for prop in shape["properties"]]
    results = []
    checked = 0
    for prop in shape["properties"]:
        path = prop["path"]
        values = node["properties"].get(path, [])
        if "min_count" in prop:
            checked += 1
            if len(values) < prop["min_count"]:
                results.append(
                    {
                        "path": path,
                        "constraint": "sh:minCount",
                        "severity": "sh:Violation",
                        "detail": "value_count_below_min_count",
                    }
                )
        if "max_count" in prop:
            checked += 1
            if len(values) > prop["max_count"]:
                results.append(
                    {
                        "path": path,
                        "constraint": "sh:maxCount",
                        "severity": "sh:Violation",
                        "detail": "value_count_above_max_count",
                    }
                )
        for value in values:
            if "in" in prop:
                checked += 1
                if value not in prop["in"]:
                    results.append(
                        {
                            "path": path,
                            "constraint": "sh:in",
                            "severity": "sh:Violation",
                            "detail": "value_not_in_declared_enumeration",
                        }
                    )
            if "pattern" in prop:
                checked += 1
                if re.match(prop["pattern"], value) is None:
                    results.append(
                        {
                            "path": path,
                            "constraint": "sh:pattern",
                            "severity": "sh:Violation",
                            "detail": "value_does_not_match_declared_pattern",
                        }
                    )
    unvalidated = sorted(
        path for path in node["properties"] if path not in declared_paths
    )
    return {
        "conforms": not results,
        "results": results,
        "validated_paths": declared_paths,
        "constraints_checked": checked,
        "unvalidated_paths": unvalidated,
        "scope_note": "declared_shape_and_focus_node_scope_only",
    }


def provenance_check(node: dict, supporting_text: str) -> dict:
    """claim이 인용한 revision·span 본문이 지금도 그 claim을 지지하는지 본다."""
    revisions = node["properties"].get("ex:sourceRevision", [])
    spans = node["properties"].get("ex:sourceSpan", [])
    if not revisions or not spans:
        return {"disposition": "DENY", "reason": "citation_incomplete"}
    cited_revision = revisions[0]
    cited_span = spans[0]
    head_text = SOURCE_REVISIONS[HEAD_REVISION]["spans"].get(cited_span)
    if head_text is None:
        return {"disposition": "DENY", "reason": "cited_span_absent_from_head_revision"}
    if supporting_text not in head_text:
        return {
            "disposition": "DENY",
            "reason": "claim_corrected_or_retracted_in_head_revision",
            "cited_revision": cited_revision,
            "head_revision": HEAD_REVISION,
        }
    return {"disposition": "ALLOW", "reason": "span_text_still_supports_claim"}


def freshness_check(node: dict) -> dict:
    """인용한 revision이 supersede됐는지 본다."""
    cited_revision = node["properties"]["ex:sourceRevision"][0]
    if SOURCE_REVISIONS[cited_revision]["superseded_by"] is not None:
        return {
            "disposition": "DENY",
            "reason": "cited_revision_superseded",
            "superseded_by": SOURCE_REVISIONS[cited_revision]["superseded_by"],
        }
    return {"disposition": "ALLOW", "reason": "cited_revision_is_head"}


def layered_admission(node: dict, supporting_text: str) -> dict:
    """shape → provenance → freshness 순서로 각 층을 따로 판정한다."""
    report = validate_shape(SHAPE, node)
    if not report["conforms"]:
        return {
            "disposition": "DENY",
            "denied_by": ["shacl"],
            "reasons": sorted({r["constraint"] for r in report["results"]}),
            "shacl": report,
        }
    provenance = provenance_check(node, supporting_text)
    freshness = freshness_check(node)
    denied_by = []
    reasons = []
    if provenance["disposition"] == "DENY":
        denied_by.append("provenance")
        reasons.append(provenance["reason"])
    if freshness["disposition"] == "DENY":
        denied_by.append("freshness")
        reasons.append(freshness["reason"])
    if denied_by:
        return {
            "disposition": "DENY",
            "denied_by": denied_by,
            "reasons": reasons,
            "shacl": report,
        }
    return {
        "disposition": "ALLOW",
        "denied_by": [],
        "reasons": ["shape_conforms_and_citation_supported_by_head_revision"],
        "shacl": report,
    }


def shacl_is_truth_admission(node: dict) -> dict:
    """반증 대상: conforms=true를 사실성 증명으로 확대한다."""
    report = validate_shape(SHAPE, node)
    if report["conforms"]:
        return {
            "disposition": "ALLOW",
            "reason": "shacl_conforms_treated_as_factual_support",
            "shacl": report,
        }
    return {"disposition": "DENY", "reason": "shacl_violation", "shacl": report}


class Receiver:
    """effect commit과 receipt 발급의 in-process 모델."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.receipts: list[dict] = []

    def commit(self, node_id: str, evidence: dict) -> dict:
        receipt = {
            "receipt_id": "rcpt-%s-%d" % (self.name, len(self.receipts) + 1),
            "logical_call_id": "call-cite-%s" % node_id.split(":")[-1],
            "node_id": node_id,
            "evidence": evidence,
        }
        self.receipts.append(receipt)
        return receipt


def run_simulation() -> tuple[Ledger, list[str]]:
    ledger = Ledger(FIXTURE)
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
        document=DOCUMENT,
        head_revision=HEAD_REVISION,
    )
    ledger.record(
        "shape",
        "shape_declared",
        shape_id=SHAPE["shape_id"],
        target_class=SHAPE["target_class"],
        declared_paths=[prop["path"] for prop in SHAPE["properties"]],
        note="constraints_outside_declared_paths_are_never_evaluated",
    )
    ledger.record(
        "source",
        "revision_superseded",
        document=DOCUMENT,
        superseded_revision=REVISION_R1,
        head_revision=HEAD_REVISION,
        corrected_span="sp-7",
        correction="certification_revoked_in_head_revision",
    )

    conforms_flags = []
    dispositions = []
    for claim in CLAIMS:
        report = validate_shape(SHAPE, claim)
        conforms_flags.append(report["conforms"])
        ledger.record(
            "shacl",
            "conforms" if report["conforms"] else "violation",
            node_id=claim["node_id"],
            shape_id=SHAPE["shape_id"],
            conforms=report["conforms"],
            constraints_checked=report["constraints_checked"],
            violation_count=len(report["results"]),
            violated_constraints=sorted({r["constraint"] for r in report["results"]}),
            validated_paths=report["validated_paths"],
            unvalidated_paths=report["unvalidated_paths"],
            scope_note=report["scope_note"],
        )

        verdict = layered_admission(claim, claim["supporting_text"])
        dispositions.append(verdict["disposition"])
        ledger.record(
            "admission",
            verdict["disposition"].lower(),
            node_id=claim["node_id"],
            disposition=verdict["disposition"],
            shacl_conforms=report["conforms"],
            admission_denied=verdict["disposition"] == "DENY",
            denied_by=verdict["denied_by"],
            reasons=verdict["reasons"],
            unvalidated_paths=report["unvalidated_paths"],
            cited_revision=claim["properties"].get("ex:sourceRevision", [None])[0],
            head_revision=HEAD_REVISION,
        )

        if verdict["disposition"] == "ALLOW":
            receipt = fenced_receiver.commit(
                claim["node_id"],
                {
                    "shape_id": SHAPE["shape_id"],
                    "cited_revision": claim["properties"]["ex:sourceRevision"][0],
                    "cited_span": claim["properties"]["ex:sourceSpan"][0],
                },
            )
            ledger.record(
                "effect",
                "receipt_committed",
                node_id=claim["node_id"],
                receipt_id=receipt["receipt_id"],
                logical_call_id=receipt["logical_call_id"],
            )
        else:
            ledger.record(
                "effect",
                "no_attempt",
                node_id=claim["node_id"],
                attempts=0,
                receipts=len(fenced_receiver.receipts),
            )

    expect(
        conforms_flags == [True, True, False],
        "shape 판정이 conform/conform/violation이 아니다: %r" % (conforms_flags,),
    )
    expect(
        dispositions == ["DENY", "ALLOW", "DENY"],
        "admission 판정이 deny/allow/deny가 아니다: %r" % (dispositions,),
    )
    expect(
        [r["node_id"] for r in fenced_receiver.receipts] == ["claim:c-502"],
        "provenance까지 통과한 claim 하나만 commit되지 않았다: %r"
        % ([r["node_id"] for r in fenced_receiver.receipts],),
    )

    c501_report = validate_shape(SHAPE, CLAIMS[0])
    expect(
        c501_report["conforms"] and "ex:certifiedUntil" in c501_report["unvalidated_paths"],
        "선언되지 않은 predicate가 unvalidated_paths로 분리되지 않았다",
    )
    ledger.record(
        "oracle",
        "conforms_true_and_admission_denied",
        node_id="claim:c-501",
        shacl_conforms=True,
        constraints_checked=c501_report["constraints_checked"],
        unvalidated_paths=c501_report["unvalidated_paths"],
        admission_disposition="DENY",
        denied_by=["provenance", "freshness"],
        note="structural_conformance_is_not_factual_support",
    )

    blind_allowed = []
    for claim in CLAIMS:
        verdict = shacl_is_truth_admission(claim)
        if verdict["disposition"] != "ALLOW":
            continue
        blind_allowed.append(claim["node_id"])
        fenced_verdict = layered_admission(claim, claim["supporting_text"])
        ledger.record(
            "counterexample:admission",
            "allow",
            gate="shacl_conforms_is_factual_truth",
            node_id=claim["node_id"],
            reason=verdict["reason"],
            unvalidated_paths=verdict["shacl"]["unvalidated_paths"],
            fenced_disposition=fenced_verdict["disposition"],
            fenced_denied_by=fenced_verdict["denied_by"],
            cited_revision=claim["properties"]["ex:sourceRevision"][0],
            head_revision=HEAD_REVISION,
        )
        receipt = blind_receiver.commit(
            claim["node_id"],
            {
                "shape_id": SHAPE["shape_id"],
                "cited_revision": claim["properties"]["ex:sourceRevision"][0],
                "provenance_checked": False,
                "freshness_checked": False,
            },
        )
        ledger.record(
            "counterexample:effect",
            "receipt_committed",
            gate="shacl_conforms_is_factual_truth",
            node_id=claim["node_id"],
            receipt_id=receipt["receipt_id"],
            cited_revision=claim["properties"]["ex:sourceRevision"][0],
            head_revision=HEAD_REVISION,
            retracted_claim_committed=fenced_verdict["disposition"] == "DENY",
        )

    expect(
        blind_allowed == ["claim:c-501", "claim:c-502"],
        "반증 gate가 conform한 두 claim을 승인하지 않아 대비가 성립하지 않는다: %r"
        % (blind_allowed,),
    )
    retracted_commits = [
        receipt["node_id"]
        for receipt in blind_receiver.receipts
        if receipt["evidence"]["cited_revision"] != HEAD_REVISION
    ]
    expect(
        retracted_commits == ["claim:c-501"],
        "철회된 claim commit이 c-501 하나가 아니다: %r" % (retracted_commits,),
    )

    ledger.record(
        "oracle",
        "contrast_recorded",
        fenced_admitted=[r["node_id"] for r in fenced_receiver.receipts],
        blind_admitted=blind_allowed,
        fenced_receipts=len(fenced_receiver.receipts),
        blind_receipts=len(blind_receiver.receipts),
        blind_retracted_commits=retracted_commits,
        conforms_flags=conforms_flags,
    )

    refuted = [
        "SHACL conforms=true는 claim이 사실이라는 증명이다 (원문에서 정정된 claim도 구조적으로는 통과한다)",
        "validation이 통과했으니 인용한 revision도 유효하다 (conforms 판정은 supersession을 보지 않는다)",
        "shape가 통과했으니 claim의 모든 predicate가 검증됐다 (선언하지 않은 predicate는 검사 대상 자체가 아니다)",
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
