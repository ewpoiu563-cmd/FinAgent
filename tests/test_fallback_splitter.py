"""Unit tests for deterministic pathological-chunk fallback splitting."""

from __future__ import annotations

from rag.fallback_splitter import split_chunk


class CharTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(character) for character in text]

    def decode(self, tokens, skip_special_tokens: bool = True):
        return "".join(chr(token) for token in tokens)


def _table_chunk() -> dict:
    rows = ["术语|含义"] + [f"词汇{i:02d}|" + "释义内容" * 8 for i in range(8)]
    text = "\n".join(rows)
    return {
        "chunk_id": "parent-table",
        "doc_id": "doc-1",
        "file_name": "report.pdf",
        "pages": [14],
        "chunk_type": "table",
        "headings": ["本招股意向书中，下列词汇具有如下含义："],
        "text": text,
        "embedding_text": text,
    }


def test_table_chunk_splits_into_traceable_children():
    parent = _table_chunk()

    children = split_chunk(parent, CharTokenizer(), max_tokens=160)

    assert len(children) > 1
    assert [child["chunk_id"] for child in children] == [
        f"parent-table__part_{index:03d}"
        for index in range(1, len(children) + 1)
    ]
    assert len({child["chunk_id"] for child in children}) == len(children)
    for index, child in enumerate(children, start=1):
        assert child["parent_chunk_id"] == "parent-table"
        assert child["headings"] == parent["headings"]
        assert child["pages"] == [14]
        assert child["chunk_type"] == "table"
        assert child["fallback_split"] is True
        assert child["part_index"] == index
        assert child["part_count"] == len(children)
        assert child["text"].startswith("术语|含义\n")
        assert len(child["embedding_text"]) <= 160


def test_text_chunk_uses_paragraph_then_sentence_boundaries():
    parent = _table_chunk()
    parent["chunk_id"] = "parent-text"
    parent["chunk_type"] = "text"
    parent["text"] = "第一段。" * 20 + "\n\n" + "第二段。" * 20
    parent["embedding_text"] = parent["text"]

    children = split_chunk(parent, CharTokenizer(), max_tokens=100)

    assert len(children) > 1
    assert all(len(child["embedding_text"]) <= 100 for child in children)
    assert "".join(child["text"] for child in children).replace("\n", "") == (
        parent["text"].replace("\n", "")
    )
