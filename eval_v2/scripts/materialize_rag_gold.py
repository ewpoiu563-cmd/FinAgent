"""Bind reviewed RAG evidence anchors to immutable index/PDF identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_units(metadata_path: Path) -> dict[str, dict[str, Any]]:
    units = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        chunk = record["chunk"]
        units[str(chunk["chunk_id"])] = chunk
    return units


def materialize(seed_path: Path, metadata_path: Path, pdf_dir: Path) -> list[dict[str, Any]]:
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    units = load_units(metadata_path)
    index_hash = sha256(metadata_path)
    output = []
    for seed in seeds:
        evidence = []
        for anchor in seed["evidence"]:
            unit_id = str(anchor["retrieval_unit_id"])
            if unit_id not in units:
                raise ValueError(f"unknown retrieval unit {unit_id} in {seed['case_id']}")
            chunk = units[unit_id]
            if chunk.get("doc_id") != anchor["doc_id"]:
                raise ValueError(f"document mismatch for {unit_id}")
            if int(anchor["pdf_page"]) not in chunk.get("pages", []):
                raise ValueError(f"page mismatch for {unit_id}")
            source_file = str(chunk["file_name"])
            pdf_path = pdf_dir / source_file
            if not pdf_path.exists() and source_file.endswith(".pdf"):
                pdf_path = pdf_dir / f"{source_file[:-4]}.PDF"
            if not pdf_path.exists():
                raise ValueError(f"missing source PDF {source_file}")
            bound = dict(anchor)
            bound.update(
                source_file=pdf_path.name,
                source_pdf_sha256=sha256(pdf_path),
                evidence_text=chunk["text"],
                evidence_text_sha256=hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
            )
            evidence.append(bound)
        record = dict(seed)
        record.update(
            gold_status="needs_pool_review",
            answer_gold_status="verified",
            retrieval_judgment_status="seed_verified_needs_pooling",
            index_metadata_sha256=index_hash,
            source_review="PDF pages visually inspected against expected claims",
            materialized_at=datetime.now(timezone.utc).isoformat(),
            evidence=evidence,
        )
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = materialize(args.seed, args.metadata, args.pdf_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
