"""Bind authored RAG Gold locators to stable chunks in one persisted index.

This utility is deliberately deterministic: it never calls an embedding, LLM,
or reranker service.  It turns the human-authored source keywords in a Gold
draft into candidate evidence anchors for later review.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def load_units(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)["chunk"]
        by_doc.setdefault(str(chunk["doc_id"]), []).append(chunk)
    return by_doc


def bind(case: dict[str, Any], units: dict[str, list[dict[str, Any]]], top_n: int) -> dict[str, Any]:
    result = dict(case)
    if case.get("expected_outcome") == "insufficient":
        result["evidence"] = []
        result["evidence_binding_status"] = "not_applicable_no_answer"
        return result
    locator = case.get("source_locator") or {}
    keywords = [normalize(str(item)) for item in locator.get("keywords", []) if normalize(str(item))]
    candidates = [chunk for doc_id in case.get("scope_doc_ids", []) for chunk in units.get(str(doc_id), [])]
    ranked: list[tuple[int, dict[str, Any]]] = []
    for chunk in candidates:
        haystack = normalize(" ".join([chunk.get("text", ""), *chunk.get("headings", [])]))
        score = sum(keyword in haystack for keyword in keywords)
        if score:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: (-item[0], str(item[1]["chunk_id"])))
    if not ranked:
        raise ValueError(f"{case['case_id']}: no chunk matched locator keywords {keywords!r}")
    selected = ranked[:top_n]
    result["evidence"] = [
        {
            "evidence_group": "g1",
            "retrieval_unit_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "pdf_page": int(chunk["pages"][0]),
            "printed_page": None,
            "relevance": 3 if score == selected[0][0] else 2,
            "locator_match_count": score,
        }
        for score, chunk in selected
    ]
    result["evidence_binding_status"] = "auto_bound_pending_manual_review"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be positive")
    cases = json.loads(args.gold.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        parser.error("--gold must contain a JSON list")
    units = load_units(args.metadata)
    bound = [bind(case, units, args.top_n) for case in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bound, ensure_ascii=False, indent=2), encoding="utf-8")
    answerable = sum(bool(item.get("evidence")) for item in bound)
    print(json.dumps({"case_count": len(bound), "answerable_bound": answerable, "no_answer": len(bound) - answerable}, ensure_ascii=False))


if __name__ == "__main__":
    main()
