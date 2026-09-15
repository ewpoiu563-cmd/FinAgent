"""Unit tests for Docling-to-FinAgent chunk normalization (no PDF parsing)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import rag.document_processor as document_processor
from rag.document_processor import (
    MAX_TOKENS,
    TOKENIZER_NAME,
    build_chunker,
    normalize_document,
    process_pdf,
    write_jsonl,
)


def _item(label: str, pages: list[int], self_ref: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        label=label,
        prov=[SimpleNamespace(page_no=page) for page in pages],
        self_ref=self_ref,
    )


def _chunk(
    text: str,
    items: list[SimpleNamespace],
    headings: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        meta=SimpleNamespace(doc_items=items, headings=headings or []),
    )


class _FakeChunker:
    def __init__(self, chunks):
        self.chunks = chunks

    def chunk(self, document):
        return iter(self.chunks)

    def contextualize(self, chunk):
        headings = " > ".join(chunk.meta.headings)
        return f"{headings}\n{chunk.text}" if headings else chunk.text


class _FakePicture(SimpleNamespace):
    def caption_text(self, document):
        return self.caption


def _document(*, pictures=None, page_count=3):
    return SimpleNamespace(
        pages={page: object() for page in range(1, page_count + 1)},
        pictures=pictures or [],
    )


def test_normalizes_text_with_pages_headings_and_context():
    source = _chunk(
        "主营业务收入同比增长。",
        [_item("text", [3, 2])],
        ["财务会计信息", "营业收入"],
    )

    result = normalize_document(
        _document(),
        file_name="report.pdf",
        doc_id="report-abc123",
        chunker=_FakeChunker([source]),
    )

    chunk = result.chunks[0]
    assert chunk.keys() == {
        "chunk_id",
        "doc_id",
        "file_name",
        "pages",
        "chunk_type",
        "headings",
        "text",
        "embedding_text",
    }
    assert chunk["pages"] == [2, 3]
    assert chunk["headings"] == ["财务会计信息", "营业收入"]
    assert chunk["embedding_text"] == (
        "财务会计信息 > 营业收入\n主营业务收入同比增长。"
    )


def test_table_uses_docling_text_without_rechunking():
    serialized_table = "| 项目 | 2025年 |\n|---|---:|\n| 营业收入 | 100 |"
    source = _chunk(serialized_table, [_item("table", [8])], ["主要财务数据"])

    result = normalize_document(
        _document(),
        file_name="report.pdf",
        doc_id="report-abc123",
        chunker=_FakeChunker([source]),
    )

    assert len(result.chunks) == 1
    assert result.chunks[0]["chunk_type"] == "table"
    assert result.chunks[0]["text"] == serialized_table


def test_picture_metadata_is_retained_but_picture_chunk_is_skipped():
    picture = _FakePicture(
        label="picture",
        prov=[SimpleNamespace(page_no=4)],
        self_ref="#/pictures/0",
        caption="业务流程图",
    )
    source = _chunk("业务流程图", [picture], ["业务与技术"])

    result = normalize_document(
        _document(pictures=[picture], page_count=4),
        file_name="report.pdf",
        doc_id="report-abc123",
        chunker=_FakeChunker([source]),
    )

    assert result.chunks == []
    assert result.pictures == [
        {"self_ref": "#/pictures/0", "pages": [4], "caption": "业务流程图"}
    ]
    assert result.stats == {
        "page_count": 4,
        "picture_count": 1,
        "skipped_picture_chunks": 1,
        "emitted_chunks": 0,
        "oversized_chunk_count": 0,
    }


def test_chunk_ids_are_stable_and_duplicate_chunks_remain_unique():
    source_a = _chunk("same", [_item("text", [1])], ["heading"])
    source_b = _chunk("same", [_item("text", [1])], ["heading"])
    chunker = _FakeChunker([source_a, source_b])

    first = normalize_document(
        _document(), file_name="a.pdf", doc_id="doc", chunker=chunker
    )
    second = normalize_document(
        _document(), file_name="a.pdf", doc_id="doc", chunker=chunker
    )

    first_ids = [chunk["chunk_id"] for chunk in first.chunks]
    second_ids = [chunk["chunk_id"] for chunk in second.chunks]
    assert first_ids == second_ids
    assert len(set(first_ids)) == 2


def test_process_pdf_uses_content_based_default_doc_id(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"small fake PDF; converter is mocked")
    document = _document()
    converter = SimpleNamespace(convert=lambda path: SimpleNamespace(document=document))
    chunker = _FakeChunker([_chunk("text", [_item("text", [1])])])

    first = process_pdf(pdf, converter=converter, chunker=chunker)
    second = process_pdf(pdf, converter=converter, chunker=chunker)

    assert first.chunks[0]["doc_id"].startswith("sample-")
    assert first.chunks[0]["doc_id"] == second.chunks[0]["doc_id"]
    assert first.chunks[0]["chunk_id"] == second.chunks[0]["chunk_id"]


def test_write_jsonl_emits_one_chunk_per_line(tmp_path):
    source = _chunk("正文", [_item("text", [1])])
    result = normalize_document(
        _document(), file_name="a.pdf", doc_id="doc", chunker=_FakeChunker([source])
    )
    output = tmp_path / "nested" / "chunks.jsonl"

    assert write_jsonl(result.chunks, output) == 1
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == result.chunks[0]


def test_process_pdf_rejects_non_pdf(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Expected a \.pdf"):
        process_pdf(path, converter=object(), chunker=object())


class _FakeTokenizer:
    def __init__(self, max_tokens=1024):
        self.max_tokens = max_tokens

    def count_tokens(self, text):
        return len(text.split())

    def get_max_tokens(self):
        return self.max_tokens


def test_build_chunker_uses_qwen_proxy_and_1024_budget(monkeypatch):
    tokenizer = object()
    from_pretrained = Mock(return_value=tokenizer)
    hybrid_chunker = Mock(return_value=object())
    monkeypatch.setattr(
        document_processor.HuggingFaceTokenizer,
        "from_pretrained",
        from_pretrained,
    )
    monkeypatch.setattr(document_processor, "HybridChunker", hybrid_chunker)

    build_chunker()

    assert TOKENIZER_NAME == "Qwen/Qwen3-Embedding-0.6B"
    assert MAX_TOKENS == 1024
    from_pretrained.assert_called_once_with(TOKENIZER_NAME, max_tokens=MAX_TOKENS)
    hybrid_chunker.assert_called_once_with(
        tokenizer=tokenizer,
        repeat_table_header=True,
        merge_peers=False,
    )


def test_retrieval_chunks_respect_configured_token_budget():
    chunker = _FakeChunker(
        [_chunk("one two three", [_item("text", [1])], ["heading"])]
    )
    chunker.tokenizer = _FakeTokenizer(max_tokens=8)

    result = normalize_document(
        _document(), file_name="a.pdf", doc_id="doc", chunker=chunker
    )

    assert result.stats["oversized_chunk_count"] == 0
    assert all(
        chunker.tokenizer.count_tokens(chunk["embedding_text"])
        <= chunker.tokenizer.get_max_tokens()
        for chunk in result.chunks
    )


def test_indivisible_oversized_chunk_is_reported(caplog):
    chunker = _FakeChunker(
        [_chunk("one two three four", [_item("table", [7])])]
    )
    chunker.tokenizer = _FakeTokenizer(max_tokens=3)

    with caplog.at_level("WARNING"):
        result = normalize_document(
            _document(), file_name="a.pdf", doc_id="doc", chunker=chunker
        )

    assert result.stats["oversized_chunk_count"] == 1
    assert "indivisible oversized chunk" in caplog.text
