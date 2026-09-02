#!/usr/bin/env python3
"""원문 revision이 교체된 뒤에도 살아 있는 요약을 다루는 fixture.

무엇을 검증하는가:
  model summary에서 파생된 claim이 source revision r1을 근거로 만들어진 뒤,
  r1이 r2로 supersede되는 순서를 in-process로 재현한다. derivation edge를
  따라 tombstone이 전파되어 summary와 파생 claim이 stale로 표시되는지,
  freshness gate가 재검증 전에는 그 claim의 사용을 차단하는지, 재검증 뒤에
  현재 revision 기준 값으로만 effect가 commit되는지 확인한다. 요약 문장이
  memory에 문자열 그대로 남아 있다는 사실과 그 문장이 참이라는 사실이
  다르다는 점을 별도 이벤트로 드러낸다. 반증 대상으로 요약 본문에 문장이
  있으면 참으로 읽는 gate를 실제로 실행해, 철회된 값으로 effect가
  commit된다는 사실을 "counterexample:" 접두 이벤트로 남긴다.

보장하지 않는 것:
  실제 memory store, 실제 SPARQL/SHACL 엔진, 실제 vector store나 분산
  저장소의 tombstone 전파를 관측한 결과가 아니다. compaction 요약을 만드는
  모델도, 실제 문서 diff도 없다. 45장이 서술한 memory class 구분과 derivation
  edge 전파 계약만 순수 표준 라이브러리로 모델링한 결정적 시뮬레이션이며,
  여기서 통과한다고 운영 시스템의 삭제 전파·index 정합성이 보장되지 않는다.
  자기 event ledger 파일 외의 I/O·시간·난수를 쓰지 않는다.

관련 장:
  45장(45.6 memory를 사실 저장소 하나로 만들지 않는다, 45.8 반례 5),
  43장(43.4 compaction은 cache eviction이 아니라 의미 보존 변환이다).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

FIXTURE = "memory-provenance-decay"
EXPECTED_SHA256 = "ee93c7fcc9b4ff888cf88437fc01a2cc6068f08719e35d7670d5b6eae2d0d853"

LEDGER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "recorded-events",
        FIXTURE + ".events.jsonl",
    )
)

RUN_ID = "run-ec-memory-decay-001"
TURN_ID = "turn-5"
PRINCIPAL = "principal:agent-desk-7"
TENANT = "tenant:acme"
ACTION = "contract_notice:schedule"
DOCUMENT = "doc:contract-9"
FACT_KEY = "termination_notice_days"

REVISION_R1 = "rev:contract-9:r1"
REVISION_R2 = "rev:contract-9:r2"
SPAN_ID = "span-14"

SUMMARY_ID = "mem:summary:s1"
SOURCE_FACT_ID = "mem:source-fact:c1"
DERIVED_CLAIM_ID = "mem:claim:k1"
REVALIDATED_CLAIM_ID = "mem:claim:k2"

SUMMARY_SENTENCE = "해지 통보는 30일 전에 하면 된다."


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


class SourceStore:
    """문서 revision과 span 본문을 담는 결정적 모델."""

    def __init__(self) -> None:
        self.revisions: dict[str, dict] = {}
        self.head: dict[str, str] = {}

    def publish(self, document: str, revision_id: str, spans: dict, facts: dict) -> None:
        previous = self.head.get(document)
        self.revisions[revision_id] = {
            "revision_id": revision_id,
            "document": document,
            "spans": dict(spans),
            "facts": dict(facts),
            "supersedes": previous,
            "superseded_by": None,
        }
        if previous is not None:
            self.revisions[previous]["superseded_by"] = revision_id
        self.head[document] = revision_id

    def current_revision(self, document: str) -> str:
        return self.head[document]

    def is_superseded(self, revision_id: str) -> bool:
        return self.revisions[revision_id]["superseded_by"] is not None

    def fact(self, revision_id: str, key: str):
        return self.revisions[revision_id]["facts"].get(key)


class MemoryStore:
    """memory item과 derivedFrom edge를 담는 결정적 모델."""

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.order: list[str] = []

    def put(self, item_id: str, memory_class: str, derived_from, **fields) -> dict:
        item = dict(fields)
        item["item_id"] = item_id
        item["memory_class"] = memory_class
        item["derived_from"] = derived_from
        item["status"] = "fresh"
        self.items[item_id] = item
        self.order.append(item_id)
        return item

    def descendants(self, root_id: str) -> list[str]:
        """derivedFrom edge를 따라 root에서 파생된 item을 결정적 순서로 모은다."""
        found: list[str] = []
        frontier = [root_id]
        while frontier:
            current = frontier.pop(0)
            for item_id in self.order:
                item = self.items[item_id]
                if item["derived_from"] == current and item_id not in found:
                    found.append(item_id)
                    frontier.append(item_id)
        return found

    def propagate_stale(self, root_revision: str, superseded_by: str) -> list[str]:
        marked = []
        for item_id in self.descendants(root_revision):
            item = self.items[item_id]
            item["status"] = "stale"
            item["stale_reason"] = "source_revision_superseded"
            item["superseded_by"] = superseded_by
            marked.append(item_id)
        return marked


class Receiver:
    """effect commit과 receipt 발급의 in-process 모델."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.receipts: list[dict] = []

    def commit(self, notice_days: int, evidence: dict) -> dict:
        receipt = {
            "receipt_id": "rcpt-%s-%d" % (self.name, len(self.receipts) + 1),
            "logical_call_id": "call-notice-9",
            "notice_days": notice_days,
            "evidence": evidence,
        }
        self.receipts.append(receipt)
        return receipt


def freshness_gate(memory: MemoryStore, source: SourceStore, item_id: str) -> dict:
    """45.6의 계약. stale 표시와 revision 일치를 둘 다 요구한다."""
    item = memory.items[item_id]
    if item["memory_class"] == "model_summary":
        return {
            "disposition": "DENY",
            "reason": "model_summary_is_not_source_provenance",
        }
    if item["status"] == "stale":
        return {
            "disposition": "REVALIDATION_REQUIRED",
            "reason": item.get("stale_reason", "marked_stale"),
        }
    cited = item["source_revision"]
    if source.is_superseded(cited):
        return {"disposition": "REVALIDATION_REQUIRED", "reason": "cited_revision_superseded"}
    if cited != source.current_revision(item["document"]):
        return {"disposition": "REVALIDATION_REQUIRED", "reason": "cited_revision_not_head"}
    return {"disposition": "ALLOW", "reason": "claim_matches_current_source_revision"}


def summary_as_truth_gate(memory: MemoryStore, sentence_marker: str) -> dict:
    """반증 대상: 요약 본문에 문장이 남아 있으면 참으로 읽는다."""
    for item_id in memory.order:
        item = memory.items[item_id]
        if item["memory_class"] != "model_summary":
            continue
        if sentence_marker in item["text"]:
            return {
                "disposition": "ALLOW",
                "reason": "sentence_present_in_summary_text",
                "item_id": item_id,
                "notice_days": item["asserted_notice_days"],
            }
    return {"disposition": "DENY", "reason": "sentence_absent_from_summary_text"}


def run_simulation() -> tuple[Ledger, list[str]]:
    ledger = Ledger(FIXTURE)
    source = SourceStore()
    memory = MemoryStore()
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
        fact_key=FACT_KEY,
    )

    source.publish(
        DOCUMENT,
        REVISION_R1,
        {SPAN_ID: "본 계약의 해지 통보 기간은 30일로 한다."},
        {FACT_KEY: 30},
    )
    ledger.record(
        "source",
        "revision_published",
        document=DOCUMENT,
        revision=REVISION_R1,
        span_id=SPAN_ID,
        fact_key=FACT_KEY,
        fact_value=source.fact(REVISION_R1, FACT_KEY),
    )

    memory.put(
        SOURCE_FACT_ID,
        "source_grounded_fact",
        REVISION_R1,
        document=DOCUMENT,
        source_revision=REVISION_R1,
        source_span=SPAN_ID,
        value=30,
    )
    memory.put(
        SUMMARY_ID,
        "model_summary",
        REVISION_R1,
        document=DOCUMENT,
        source_revision=REVISION_R1,
        producer_model="compactor-model",
        source_message_range="turn-1..turn-4",
        text="계약 요약: " + SUMMARY_SENTENCE,
        asserted_notice_days=30,
    )
    memory.put(
        DERIVED_CLAIM_ID,
        "source_grounded_fact",
        SUMMARY_ID,
        document=DOCUMENT,
        source_revision=REVISION_R1,
        source_span=SPAN_ID,
        value=30,
        derivation_path=[REVISION_R1, SUMMARY_ID],
    )
    for item_id in (SOURCE_FACT_ID, SUMMARY_ID, DERIVED_CLAIM_ID):
        item = memory.items[item_id]
        ledger.record(
            "memory",
            "item_stored",
            item_id=item_id,
            memory_class=item["memory_class"],
            derived_from=item["derived_from"],
            source_revision=item["source_revision"],
            status=item["status"],
        )

    pre_supersede = freshness_gate(memory, source, DERIVED_CLAIM_ID)
    expect(
        pre_supersede["disposition"] == "ALLOW",
        "supersede 이전에 파생 claim이 통과하지 못했다: %r" % (pre_supersede,),
    )
    ledger.record(
        "freshness",
        "allow",
        phase="before_supersede",
        item_id=DERIVED_CLAIM_ID,
        source_revision=REVISION_R1,
        head_revision=source.current_revision(DOCUMENT),
        reason=pre_supersede["reason"],
    )

    source.publish(
        DOCUMENT,
        REVISION_R2,
        {SPAN_ID: "본 계약의 해지 통보 기간은 60일로 한다."},
        {FACT_KEY: 60},
    )
    ledger.record(
        "source",
        "revision_superseded",
        document=DOCUMENT,
        superseded_revision=REVISION_R1,
        head_revision=REVISION_R2,
        fact_key=FACT_KEY,
        old_value=source.fact(REVISION_R1, FACT_KEY),
        new_value=source.fact(REVISION_R2, FACT_KEY),
    )

    marked = memory.propagate_stale(REVISION_R1, REVISION_R2)
    ledger.record(
        "memory",
        "tombstone_propagated",
        root_revision=REVISION_R1,
        superseded_by=REVISION_R2,
        marked_stale=marked,
        propagation_edge="derivedFrom",
    )
    expect(
        marked == [SOURCE_FACT_ID, SUMMARY_ID, DERIVED_CLAIM_ID],
        "derivation edge를 따라 세 item이 stale로 표시되지 않았다: %r" % (marked,),
    )

    ledger.record(
        "memory",
        "summary_text_retained",
        item_id=SUMMARY_ID,
        sentence=SUMMARY_SENTENCE,
        sentence_still_present=SUMMARY_SENTENCE in memory.items[SUMMARY_ID]["text"],
        summary_status=memory.items[SUMMARY_ID]["status"],
        derived_claim_id=DERIVED_CLAIM_ID,
        derived_claim_status=memory.items[DERIVED_CLAIM_ID]["status"],
        derived_claim_value=memory.items[DERIVED_CLAIM_ID]["value"],
        current_source_value=source.fact(REVISION_R2, FACT_KEY),
        note="sentence_presence_is_not_truth",
    )
    expect(
        SUMMARY_SENTENCE in memory.items[SUMMARY_ID]["text"],
        "요약 본문에서 문장이 사라져 대비가 성립하지 않는다",
    )

    post_supersede = freshness_gate(memory, source, DERIVED_CLAIM_ID)
    expect(
        post_supersede["disposition"] == "REVALIDATION_REQUIRED",
        "supersede 뒤에도 freshness gate가 차단하지 않았다: %r" % (post_supersede,),
    )
    ledger.record(
        "admission",
        "deny",
        phase="after_supersede",
        item_id=DERIVED_CLAIM_ID,
        disposition=post_supersede["disposition"],
        reason=post_supersede["reason"],
        cited_revision=REVISION_R1,
        head_revision=source.current_revision(DOCUMENT),
    )
    ledger.record(
        "effect",
        "no_attempt",
        phase="after_supersede",
        item_id=DERIVED_CLAIM_ID,
        attempts=0,
        receipts=len(fenced_receiver.receipts),
    )
    expect(fenced_receiver.receipts == [], "재검증 전에 receipt가 발급됐다")

    summary_gate_before_revalidation = freshness_gate(memory, source, SUMMARY_ID)
    ledger.record(
        "admission",
        "deny",
        phase="summary_direct_use",
        item_id=SUMMARY_ID,
        disposition=summary_gate_before_revalidation["disposition"],
        reason=summary_gate_before_revalidation["reason"],
    )

    head = source.current_revision(DOCUMENT)
    memory.put(
        REVALIDATED_CLAIM_ID,
        "source_grounded_fact",
        head,
        document=DOCUMENT,
        source_revision=head,
        source_span=SPAN_ID,
        value=source.fact(head, FACT_KEY),
        derivation_path=[head],
    )
    ledger.record(
        "memory",
        "claim_revalidated",
        item_id=REVALIDATED_CLAIM_ID,
        replaces=DERIVED_CLAIM_ID,
        source_revision=head,
        source_span=SPAN_ID,
        value=source.fact(head, FACT_KEY),
        superseded_value=memory.items[DERIVED_CLAIM_ID]["value"],
    )

    revalidated = freshness_gate(memory, source, REVALIDATED_CLAIM_ID)
    expect(
        revalidated["disposition"] == "ALLOW",
        "재검증한 claim이 통과하지 못했다: %r" % (revalidated,),
    )
    ledger.record(
        "admission",
        "allow",
        phase="after_revalidation",
        item_id=REVALIDATED_CLAIM_ID,
        disposition=revalidated["disposition"],
        reason=revalidated["reason"],
        source_revision=head,
    )
    receipt = fenced_receiver.commit(
        memory.items[REVALIDATED_CLAIM_ID]["value"],
        {"source_revision": head, "source_span": SPAN_ID, "item_id": REVALIDATED_CLAIM_ID},
    )
    ledger.record(
        "effect",
        "receipt_committed",
        receipt_id=receipt["receipt_id"],
        logical_call_id=receipt["logical_call_id"],
        notice_days=receipt["notice_days"],
        source_revision=head,
    )
    expect(
        [r["notice_days"] for r in fenced_receiver.receipts] == [60],
        "재검증 뒤 commit된 값이 현재 revision 값(60)이 아니다",
    )

    blind_verdict = summary_as_truth_gate(memory, "30일")
    expect(
        blind_verdict["disposition"] == "ALLOW",
        "반증 gate가 요약 문장을 참으로 읽지 않아 대비가 성립하지 않는다",
    )
    ledger.record(
        "counterexample:admission",
        "allow",
        gate="summary_text_presence_is_truth",
        item_id=blind_verdict["item_id"],
        reason=blind_verdict["reason"],
        asserted_notice_days=blind_verdict["notice_days"],
        item_status=memory.items[blind_verdict["item_id"]]["status"],
        cited_revision=memory.items[blind_verdict["item_id"]]["source_revision"],
        head_revision=head,
        current_source_value=source.fact(head, FACT_KEY),
    )
    blind_receipt = blind_receiver.commit(
        blind_verdict["notice_days"],
        {
            "source_revision": memory.items[blind_verdict["item_id"]]["source_revision"],
            "item_id": blind_verdict["item_id"],
            "provenance": "model_summary_text",
        },
    )
    ledger.record(
        "counterexample:effect",
        "receipt_committed",
        gate="summary_text_presence_is_truth",
        receipt_id=blind_receipt["receipt_id"],
        notice_days=blind_receipt["notice_days"],
        authorizing_revision=REVISION_R1,
        head_revision=head,
        retracted_value_committed=blind_receipt["notice_days"] != source.fact(head, FACT_KEY),
    )
    expect(
        blind_receipt["notice_days"] == 30 and source.fact(head, FACT_KEY) == 60,
        "반증 gate가 철회된 값으로 commit하지 않았다",
    )

    ledger.record(
        "oracle",
        "contrast_recorded",
        fenced_receipt_values=[r["notice_days"] for r in fenced_receiver.receipts],
        blind_receipt_values=[r["notice_days"] for r in blind_receiver.receipts],
        stale_items=marked,
        summary_sentence_still_present=True,
        head_revision=head,
        head_value=source.fact(head, FACT_KEY),
    )

    refuted = [
        "요약에 문장이 남아 있으면 그 문장은 참이다 (원문 revision이 supersede된 뒤에도 문자열은 남지만 claim은 stale이다)",
        "summary 파생 claim은 원문 provenance를 대신할 수 있다 (freshness gate는 재검증 전 사용을 차단한다)",
        "source revision 교체는 새 조회에만 영향을 준다 (derivedFrom edge를 따라 기존 memory item까지 stale이 전파되어야 한다)",
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
