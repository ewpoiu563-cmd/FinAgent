"""Search a persisted FinAgent dense FAISS index."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rag.retriever import DenseRetriever


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search a FinAgent dense FAISS index.")
    parser.add_argument("--index-dir", type=Path, required=True, help="index directory")
    parser.add_argument("--query", required=True, help="natural-language query")
    parser.add_argument("--top-k", type=_positive_int, default=5, help="result count")
    return parser


def _print_result(result: dict) -> None:
    print(f"rank: {result['rank']}")
    print(f"score: {result['score']:.6f}")
    print(f"file_name: {result['file_name']}")
    print(f"pages: {result['pages']}")
    print(f"chunk_type: {result['chunk_type']}")
    print(f"headings: {result['headings']}")
    print(f"text: {result['text'][:300]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        results = DenseRetriever(args.index_dir).retrieve(args.query, top_k=args.top_k)
    except Exception as error:
        parser.exit(1, f"error: {error}\n")

    for position, result in enumerate(results):
        if position:
            print()
        _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
