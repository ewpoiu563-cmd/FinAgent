"""CLI for Multi-document RAG Stage C: offline cached-vector index build."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rag.multi_document import build_multi_document_index, load_embedding_artifact
from rag.vector_store import FaissVectorStore

def main() -> int:
    parser = argparse.ArgumentParser(description="Stage C: merge frozen base index and cached embedding artifacts without API calls.")
    parser.add_argument("--base-index", type=Path, default=Path("outputs/index/full_doc1"))
    parser.add_argument("--artifact", type=Path, required=True, action="append", help="repeat once per cached document artifact")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate base/artifacts only; do not write FAISS")
    args = parser.parse_args()
    try:
        if args.dry_run:
            base = FaissVectorStore.load(args.base_index)
            manifests = [load_embedding_artifact(path)[0] for path in args.artifact]
            print(json.dumps({"dry_run": True, "base_vectors": base.vector_count, "artifact_doc_ids": [m["doc_id"] for m in manifests], "output": str(args.output_dir)}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(build_multi_document_index(args.base_index, args.artifact, args.output_dir), ensure_ascii=False, indent=2))
    except Exception as error:
        parser.exit(1, f"error: {error}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
