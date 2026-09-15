"""Docling-backed PDF processing and FinAgent chunk normalization.

PyMuPDF remains available in :mod:`rag.document_loader` as the page-level
baseline/fallback.  This module is the production structured-PDF path:

PDF -> DoclingDocument -> HybridChunker -> FinAgent unified chunks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TypedDict

from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import PictureItem, TableItem


TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"
MAX_TOKENS = 1024

logger = logging.getLogger(__name__)


class UnifiedChunk(TypedDict):
    """The stable chunk contract consumed by later FinAgent RAG stages."""

    chunk_id: str
    doc_id: str
    file_name: str
    pages: list[int]
    chunk_type: str
    headings: list[str]
    text: str
    embedding_text: str


class PictureMetadata(TypedDict):
    """Non-embedding metadata retained for one Docling PictureItem."""

    self_ref: str
    pages: list[int]
    caption: str


class ProcessingStats(TypedDict):
    page_count: int
    picture_count: int
    skipped_picture_chunks: int
    emitted_chunks: int
    oversized_chunk_count: int


class ProcessingTiming(TypedDict):
    """Wall-clock timings for the parse and Docling chunking stages."""

    parse_seconds: float
    chunk_seconds: float
    fallback_split_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class ProcessingResult:
    """Unified chunks plus document-level metadata excluded from embeddings."""

    chunks: list[UnifiedChunk]
    pictures: list[PictureMetadata]
    stats: ProcessingStats
    timing: ProcessingTiming | None = None


def build_converter() -> DocumentConverter:
    """Build the validated CPU-only Docling PDF converter (no OCR or VLM)."""

    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        do_picture_classification=False,
        do_picture_description=False,
        do_chart_extraction=False,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
        enable_remote_services=False,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU,
            num_threads=4,
        ),
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


def build_chunker(
    *,
    tokenizer_name: str = TOKENIZER_NAME,
    max_tokens: int = MAX_TOKENS,
) -> HybridChunker:
    """Build the validated table-aware HybridChunker configuration.

    The public Qwen3 Embedding tokenizer is loaded only as a local token-count
    proxy. No local embedding weights are loaded, and exact token parity with
    the hosted qwen3.7 embedding model is not assumed.
    """

    tokenizer = HuggingFaceTokenizer.from_pretrained(
        tokenizer_name,
        max_tokens=max_tokens,
    )
    return HybridChunker(
        tokenizer=tokenizer,
        repeat_table_header=True,
        merge_peers=False,
    )


def _validate_pdf(path: str | Path) -> Path:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {pdf_path}")
    return pdf_path


def _content_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_doc_id(path: Path) -> str:
    return f"{path.stem}-{_content_digest(path)[:12]}"


def _doc_items(value: Any) -> Sequence[Any]:
    meta = getattr(value, "meta", None)
    return tuple(getattr(meta, "doc_items", None) or ())


def _label(item: Any) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label)).lower()


def _is_picture(item: Any) -> bool:
    return isinstance(item, PictureItem) or _label(item) == "picture"


def _is_table(item: Any) -> bool:
    return isinstance(item, TableItem) or _label(item) == "table"


def _pages(items: Iterable[Any]) -> list[int]:
    page_numbers: set[int] = set()
    for item in items:
        for provenance in getattr(item, "prov", None) or ():
            page_number = getattr(provenance, "page_no", None)
            if page_number is not None:
                page_numbers.add(int(page_number))
    return sorted(page_numbers)


def _headings(chunk: Any) -> list[str]:
    headings = getattr(getattr(chunk, "meta", None), "headings", None) or ()
    return [str(heading) for heading in headings if str(heading).strip()]


def _picture_metadata(picture: Any, document: Any) -> PictureMetadata:
    try:
        caption = str(picture.caption_text(document) or "")
    except Exception:
        caption = ""
    return {
        "self_ref": str(getattr(picture, "self_ref", "")),
        "pages": _pages((picture,)),
        "caption": caption,
    }


def _chunk_fingerprint(
    *,
    doc_id: str,
    pages: list[int],
    chunk_type: str,
    headings: list[str],
    text: str,
    embedding_text: str,
) -> str:
    payload = {
        "doc_id": doc_id,
        "pages": pages,
        "chunk_type": chunk_type,
        "headings": headings,
        "text": text,
        "embedding_text": embedding_text,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_document(
    document: Any,
    *,
    file_name: str,
    doc_id: str,
    chunker: Any,
) -> ProcessingResult:
    """Chunk a DoclingDocument and normalize it into the FinAgent contract.

    PictureItem metadata is retained at document level.  Any chunk containing a
    picture is deliberately excluded from embedding output in this phase.
    Table text is accepted exactly as serialized by Docling/HybridChunker.
    """

    pictures = [
        _picture_metadata(picture, document)
        for picture in (getattr(document, "pictures", None) or ())
    ]
    chunks: list[UnifiedChunk] = []
    fingerprint_occurrences: dict[str, int] = {}
    skipped_picture_chunks = 0
    oversized_chunk_count = 0

    for source_chunk in chunker.chunk(document):
        items = _doc_items(source_chunk)
        if any(_is_picture(item) for item in items):
            skipped_picture_chunks += 1
            continue

        text = str(getattr(source_chunk, "text", ""))
        if not text.strip():
            continue
        pages = _pages(items)
        headings = _headings(source_chunk)
        chunk_type = "table" if any(_is_table(item) for item in items) else "text"
        embedding_text = str(chunker.contextualize(source_chunk))

        tokenizer = getattr(chunker, "tokenizer", None)
        if tokenizer is not None:
            token_count = int(tokenizer.count_tokens(embedding_text))
            token_budget = int(tokenizer.get_max_tokens())
            if token_count > token_budget:
                oversized_chunk_count += 1
                logger.warning(
                    "Docling emitted an indivisible oversized chunk: "
                    "doc_id=%s pages=%s tokens=%d budget=%d",
                    doc_id,
                    pages,
                    token_count,
                    token_budget,
                )

        fingerprint = _chunk_fingerprint(
            doc_id=doc_id,
            pages=pages,
            chunk_type=chunk_type,
            headings=headings,
            text=text,
            embedding_text=embedding_text,
        )
        occurrence = fingerprint_occurrences.get(fingerprint, 0)
        fingerprint_occurrences[fingerprint] = occurrence + 1
        unique_material = f"{fingerprint}:{occurrence}".encode("ascii")
        chunk_hash = hashlib.sha256(unique_material).hexdigest()[:24]

        chunks.append(
            {
                "chunk_id": f"chk_{chunk_hash}",
                "doc_id": doc_id,
                "file_name": file_name,
                "pages": pages,
                "chunk_type": chunk_type,
                "headings": headings,
                "text": text,
                "embedding_text": embedding_text,
            }
        )

    page_count = len(getattr(document, "pages", None) or ())
    stats: ProcessingStats = {
        "page_count": page_count,
        "picture_count": len(pictures),
        "skipped_picture_chunks": skipped_picture_chunks,
        "emitted_chunks": len(chunks),
        "oversized_chunk_count": oversized_chunk_count,
    }
    return ProcessingResult(chunks=chunks, pictures=pictures, stats=stats)


def process_pdf(
    path: str | Path,
    *,
    doc_id: str | None = None,
    converter: Any | None = None,
    chunker: Any | None = None,
) -> ProcessingResult:
    """Convert one PDF and return normalized chunks plus non-embedding metadata."""

    started_at = time.perf_counter()
    pdf_path = _validate_pdf(path)
    resolved_doc_id = doc_id or _default_doc_id(pdf_path)
    active_converter = converter or build_converter()
    active_chunker = chunker or build_chunker()
    parse_started_at = time.perf_counter()
    conversion_result = active_converter.convert(pdf_path)
    parse_seconds = time.perf_counter() - parse_started_at
    chunk_started_at = time.perf_counter()
    result = normalize_document(
        conversion_result.document,
        file_name=pdf_path.name,
        doc_id=resolved_doc_id,
        chunker=active_chunker,
    )
    chunk_seconds = time.perf_counter() - chunk_started_at
    # Semantic fallback is deliberately not attempted here.  It is an
    # embedding-error recovery step and is timed by Stage B when it occurs.
    return ProcessingResult(
        chunks=result.chunks,
        pictures=result.pictures,
        stats=result.stats,
        timing={
            "parse_seconds": round(parse_seconds, 3),
            "chunk_seconds": round(chunk_seconds, 3),
            "fallback_split_seconds": 0.0,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        },
    )


def write_jsonl(chunks: Iterable[UnifiedChunk], output_path: str | Path) -> int:
    """Write one UTF-8 JSON object per line and return the row count."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
            count += 1
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one PDF into FinAgent unified JSONL chunks with Docling."
    )
    parser.add_argument("input", type=Path, help="input PDF path")
    parser.add_argument("--output", type=Path, required=True, help="output JSONL path")
    parser.add_argument("--doc-id", help="optional explicit stable document ID")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = process_pdf(args.input, doc_id=args.doc_id)
    write_jsonl(result.chunks, args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                **result.stats,
                **(result.timing or {}),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
