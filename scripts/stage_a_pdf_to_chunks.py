"""CLI for Multi-document RAG Stage A: one selected PDF to chunks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rag.multi_document import selected_document, stage_a_pdf_to_chunks

def main() -> int:
    parser = argparse.ArgumentParser(description="Stage A: parse exactly one manifest-selected PDF with frozen Docling/HybridChunker.")
    parser.add_argument("--input", type=Path, required=True, help="one explicit PDF path")
    parser.add_argument("--output", type=Path, required=True, help="one output chunks JSONL")
    parser.add_argument("--selection-manifest", type=Path, default=Path("data/documents/multi_doc_v1.json"))
    parser.add_argument("--force", action="store_true", help="allow replacing this chunk JSONL")
    parser.add_argument("--dry-run", action="store_true", help="validate selection only; do not parse PDF")
    args = parser.parse_args()
    try:
        if args.dry_run:
            print(json.dumps({"dry_run": True, "selected": selected_document(args.selection_manifest, args.input), "input": str(args.input), "output": str(args.output)}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(stage_a_pdf_to_chunks(args.input, args.output, selection_manifest=args.selection_manifest, force=args.force), ensure_ascii=False, indent=2))
    except Exception as error:
        parser.exit(1, f"error: {error}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
