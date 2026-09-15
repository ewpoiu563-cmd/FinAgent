"""Page-level PDF text extraction for the Financial RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import pymupdf


class PDFOpenError(RuntimeError):
    """Raised when a PDF cannot be opened."""


class PDFPageExtractionError(RuntimeError):
    """Raised when text extraction fails for a specific PDF page."""


class PDFPage(TypedDict, total=False):
    """Text and source metadata for one physical PDF page."""

    doc_id: str
    file_name: str
    page: int
    text: str
    extraction_error: str


def load_pdf(
    path: str | Path,
    *,
    on_page_error: Literal["raise", "record"] = "raise",
) -> list[PDFPage]:
    """Extract a PDF into one record per physical page.

    Text is returned as provided by PyMuPDF without additional cleaning. Blank or
    image-only pages are retained with an empty ``text`` value. By default, a
    page extraction failure raises :class:`PDFPageExtractionError`. The
    ``record`` policy is intended for audits: it retains the failed page with an
    empty text value and an ``extraction_error`` message.
    """

    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {pdf_path}")
    if on_page_error not in {"raise", "record"}:
        raise ValueError("on_page_error must be either 'raise' or 'record'")

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise PDFOpenError(f"Unable to open PDF '{pdf_path}': {exc}") from exc

    pages: list[PDFPage] = []
    try:
        for page_index in range(document.page_count):
            page_number = page_index + 1
            record: PDFPage = {
                "doc_id": pdf_path.stem,
                "file_name": pdf_path.name,
                "page": page_number,
                "text": "",
            }
            try:
                record["text"] = document.load_page(page_index).get_text()
            except Exception as exc:
                message = (
                    f"Failed to extract page {page_number} from "
                    f"'{pdf_path}': {exc}"
                )
                if on_page_error == "raise":
                    raise PDFPageExtractionError(message) from exc
                record["extraction_error"] = message
            pages.append(record)
    finally:
        document.close()

    return pages
