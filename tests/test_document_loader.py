"""Independent tests for page-level PDF loading."""

from pathlib import Path

import pymupdf
import pytest

from rag.document_loader import (
    PDFOpenError,
    PDFPageExtractionError,
    load_pdf,
)


def _make_pdf(path: Path, page_texts: list[str]) -> None:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_load_pdf_returns_one_record_per_page(tmp_path):
    path = tmp_path / "sample.pdf"
    _make_pdf(path, ["first page", ""])

    pages = load_pdf(path)

    assert len(pages) == 2
    assert pages[0] == {
        "doc_id": "sample",
        "file_name": "sample.pdf",
        "page": 1,
        "text": "first page\n",
    }
    assert pages[1]["page"] == 2
    assert pages[1]["text"] == ""


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_pdf(tmp_path / "missing.pdf")


def test_non_pdf_file_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Expected a \.pdf"):
        load_pdf(path)


def test_unopenable_pdf_has_clear_error(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a PDF")
    with pytest.raises(PDFOpenError, match="Unable to open PDF"):
        load_pdf(path)


class _BrokenPage:
    def get_text(self):
        raise RuntimeError("synthetic extraction failure")


class _BrokenDocument:
    page_count = 1

    def load_page(self, page_index):
        return _BrokenPage()

    def close(self):
        pass


def test_page_failure_raises_with_page_number(tmp_path, monkeypatch):
    path = tmp_path / "valid-name.pdf"
    path.touch()
    monkeypatch.setattr("rag.document_loader.pymupdf.open", lambda _: _BrokenDocument())

    with pytest.raises(PDFPageExtractionError, match="page 1"):
        load_pdf(path)


def test_page_failure_can_be_recorded_for_audit(tmp_path, monkeypatch):
    path = tmp_path / "valid-name.pdf"
    path.touch()
    monkeypatch.setattr("rag.document_loader.pymupdf.open", lambda _: _BrokenDocument())

    pages = load_pdf(path, on_page_error="record")

    assert pages[0]["text"] == ""
    assert "page 1" in pages[0]["extraction_error"]


def test_invalid_page_error_policy_is_rejected(tmp_path):
    path = tmp_path / "sample.pdf"
    _make_pdf(path, ["content"])
    with pytest.raises(ValueError, match="on_page_error"):
        load_pdf(path, on_page_error="ignore")  # type: ignore[arg-type]
