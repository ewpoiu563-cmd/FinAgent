"""Render a compact, human-readable judgement sheet for manual RAG qrels review.

Reads the frozen review pool (union of Top-5 candidates per retrieval route),
the answer key, and writes a text sheet where every candidate chunk is shown
with the window that best overlaps the reference answer. The sheet is the input
for a manual per-case relevance judgement; it does not compute any metric and
never modifies the frozen inputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")
NUMBER = re.compile(r"\d[\d,\.]*\s*(?:%|％|万|亿|元|股|吨|年|月|日)?")


def _keys(*texts: str) -> set[str]:
    """Answer key terms: numeric tokens and longer CJK n-grams from reference texts."""
    numbers: set[str] = set()
    grams: set[str] = set()
    for text in texts:
        if not text:
            continue
        for token in NUMBER.findall(text):
            token = token.strip()
            if len(token) >= 2:
                numbers.add(token)
        cjk_only = "".join(char if CJK.match(char) else " " for char in text)
        for segment in cjk_only.split():
            for size in (3, 4, 5, 6):
                for index in range(len(segment) - size + 1):
                    grams.add(segment[index : index + size])
    return numbers | grams


def _score(text: str, keys: set[str]) -> tuple[float, list[str]]:
    hits = [key for key in keys if key in text]
    score = sum(len(key) for key in hits)
    numbers = sum(len(key) for key in hits if key[0].isdigit())
    return score + 2 * numbers, sorted(hits, key=len, reverse=True)[:6]


def _best_window(text: str, keys: set[str], width: int = 150, step: int = 20) -> tuple[str, list[str], float]:
    if not text:
        return "", [], 0.0
    best_score = -1
    best_start = 0
    for start in range(0, max(1, len(text) - width + 1), step):
        window = text[start : start + width]
        score, _ = _score(window, keys)
        if score > best_score:
            best_score, best_start = score, start
    window = text[best_start : best_start + width]
    score, hits = _score(window, keys)
    return ("…" + window if best_start else window), hits, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-chars", type=int, default=45)
    parser.add_argument("--llm-chars", type=int, default=260)
    parser.add_argument("--min-score", type=float, default=14.0)
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    key_rows = {
        row["case_id"]: row for row in json.loads(args.answer_key.read_text(encoding="utf-8"))
    }

    blocks: list[str] = []
    for case in pool:
        key = key_rows.get(case["case_id"], {})
        canonical = key.get("canonical_answer", "") or ""
        seed_ids = set(re.findall(r"chk_[0-9a-f]+", canonical))
        keys = _keys(case.get("reference_answer", ""), " ".join(case.get("expected_claims") or []))
        lines = [
            "=" * 100,
            f"{case['case_id']} | {case.get('test_type')} | {case.get('difficulty')} | "
            f"outcome={case.get('expected_outcome')}",
            f"Q: {case.get('question')}",
            f"REF: {case.get('reference_answer')}",
        ]
        if case.get("expected_claims") and case.get("test_type") in {
            "multi_chunk_evidence",
            "cross_chapter_implicit",
            "no_answer_boundary",
            "concept_disambiguation",
        }:
            lines.append("CLAIMS: " + " / ".join(case["expected_claims"]))
        llm = (case.get("llm_answer") or "").replace("\n", " ")
        if len(llm) > args.llm_chars:
            llm = llm[: args.llm_chars] + "…"
        lines.append(f"LLM: {llm}")
        hits_seen = 0
        for cand in case["candidates"]:
            text = cand.get("text") or ""
            window, hits, score = _best_window(text, keys)
            if score >= args.min_score:
                hits_seen += 1
            else:
                window = text[: args.head_chars]
                hits = []
            ranks = cand.get("ranks") or {}
            rank_text = ",".join(
                f"{route[:2]}{ranks[route]}" for route in ("dense", "dense_rerank", "hybrid_rerank") if route in ranks
            )
            seed = " SEED" if cand["retrieval_unit_id"] in seed_ids else ""
            marker = f" [score={score:.0f}]" if hits else " [low-overlap]"
            alias = cand["retrieval_unit_id"][-6:]
            lines.append(
                f"  - [{alias}] ({rank_text}){seed}{marker} p{cand.get('pages')} :: "
                f"{window.replace(chr(10), ' ')}"
            )
        lines.append(f"  # candidates={len(case['candidates'])} keyword_overlap={hits_seen}")
        blocks.append("\n".join(lines))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"cases={len(blocks)} output={args.output}")


if __name__ == "__main__":
    main()
