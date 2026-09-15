"""CLI for Multi-document RAG Stage B: cached document embeddings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rag.multi_document import embed_chunks_to_artifact, load_embedding_artifact
from scripts.build_index import _cli_batch_size, _non_negative_int, _positive_float
from rag.embedding import DEFAULT_BATCH_SIZE, DEFAULT_TIMEOUT

def main() -> int:
    parser = argparse.ArgumentParser(description="Stage B: embed one chunks JSONL once and persist a reusable artifact; no FAISS build.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, default=Path("data/documents/multi_doc_v1.json"))
    parser.add_argument("--batch-size", type=_cli_batch_size, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout", type=_positive_float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=_non_negative_int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate an existing completed artifact only; never call the API")
    args = parser.parse_args()
    try:
        if args.dry_run:
            manifest, _, _ = load_embedding_artifact(args.output_dir)
            print(json.dumps({"dry_run": True, "artifact": manifest}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(embed_chunks_to_artifact(args.input, args.output_dir, batch_size=args.batch_size, timeout=args.timeout, max_retries=args.max_retries, resume=args.resume, selection_manifest=args.selection_manifest), ensure_ascii=False, indent=2))
    except Exception as error:
        parser.exit(1, f"error: {error}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
