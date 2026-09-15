"""Unit tests for the local BM25 retriever."""

from __future__ import annotations

import json

import pytest

from rag.bm25_retriever import BM25Retriever, tokenize
from eval.evaluate_bm25_retrieval import aggregate_metrics, retrieval_unit_id, score_case


def _chunk(index: int, *, heading: str = "", text: str = "", fallback=False) -> dict:
    chunk = {
        "chunk_id": f"chunk-{index}",
        "doc_id": "doc-1",
        "file_name": "report.pdf",
        "pages": [index + 1],
        "chunk_type": "text",
        "headings": [heading] if heading else [],
        "text": text or f"普通内容 {index}",
        "embedding_text": "must not be used by BM25",
    }
    if fallback:
        chunk.update(
            parent_chunk_id="parent-1",
            fallback_split=True,
            part_index=1,
            part_count=2,
        )
    return chunk


def _write_metadata(path, chunks) -> None:
    with (path / "metadata.jsonl").open("w", encoding="utf-8") as output:
        for position, chunk in enumerate(chunks):
            output.write(
                json.dumps({"position": position, "chunk": chunk}, ensure_ascii=False)
                + "\n"
            )


def test_tokenizer_segments_chinese_without_whitespace_split_and_keeps_entities():
    tokens = tokenize("HA 树脂与DNA产品在2013年至2015年增长67.63%，收入50,890.52万元")

    assert "ha" in tokens
    assert "dna" in tokens
    assert "2013" in tokens and "2015" in tokens
    assert "67.63%" in tokens
    assert "50,890.52" in tokens
    assert any(token in tokens for token in ("树脂", "产品", "增长", "收入"))


def test_searchable_text_uses_headings_plus_text_but_not_embedding_text(tmp_path):
    _write_metadata(
        tmp_path,
        [
            _chunk(0, heading="核心技术泄密风险", text="一般说明"),
            _chunk(1, heading="其他", text="血液灌流器销售收入 94.80%"),
            _chunk(2, heading="无关", text="其他内容"),
        ],
    )
    retriever = BM25Retriever(tmp_path)

    assert retriever.retrieve("核心技术泄密", top_k=1)[0]["chunk_id"] == "chunk-0"
    assert retriever.retrieve("94.80% 销售收入", top_k=1)[0]["chunk_id"] == "chunk-1"
    assert retriever.retrieve("must", top_k=1)[0]["score"] == pytest.approx(0.0)


def test_result_schema_and_fallback_metadata_match_dense_shape(tmp_path):
    _write_metadata(
        tmp_path,
        [
            _chunk(0, text="完全无关"),
            _chunk(1, text="HA 树脂 特殊采购", fallback=True),
            _chunk(2, text="另一段无关内容"),
        ],
    )

    result = BM25Retriever(tmp_path).retrieve("HA 树脂", top_k=1)[0]

    assert result == {
        "rank": 1,
        "score": pytest.approx(result["score"]),
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
        "file_name": "report.pdf",
        "pages": [2],
        "chunk_type": "text",
        "headings": [],
        "text": "HA 树脂 特殊采购",
        "parent_chunk_id": "parent-1",
        "fallback_split": True,
        "part_index": 1,
        "part_count": 2,
    }


def test_raw_top_k_keeps_sibling_chunks_and_caps_at_corpus_size(tmp_path):
    first = _chunk(0, text="采购模式", fallback=True)
    second = _chunk(1, text="采购模式", fallback=True)
    second["chunk_id"] = "parent-1__part_002"
    second["part_index"] = 2
    _write_metadata(tmp_path, [first, second])

    results = BM25Retriever(tmp_path).retrieve("采购模式", top_k=99)

    assert len(results) == 2
    assert [result["rank"] for result in results] == [1, 2]
    assert [result["parent_chunk_id"] for result in results] == ["parent-1", "parent-1"]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "2", None])
def test_invalid_top_k_is_rejected(tmp_path, top_k):
    _write_metadata(tmp_path, [_chunk(0)])
    with pytest.raises(ValueError, match="top_k"):
        BM25Retriever(tmp_path).retrieve("风险", top_k=top_k)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(tmp_path, query):
    _write_metadata(tmp_path, [_chunk(0)])
    with pytest.raises(ValueError, match="query must not be empty"):
        BM25Retriever(tmp_path).retrieve(query)


def test_non_string_query_and_punctuation_only_query_are_rejected(tmp_path):
    _write_metadata(tmp_path, [_chunk(0)])
    retriever = BM25Retriever(tmp_path)
    with pytest.raises(TypeError, match="query must be a string"):
        retriever.retrieve(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="searchable token"):
        retriever.retrieve("？！……")


def test_missing_metadata_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata file does not exist"):
        BM25Retriever(tmp_path)


@pytest.mark.parametrize(
    "line, message",
    [
        ("not-json\n", "invalid JSON"),
        (json.dumps({"position": 2, "chunk": _chunk(0)}) + "\n", "position mismatch"),
        (json.dumps({"position": 0, "chunk": {}}) + "\n", "missing required fields"),
    ],
)
def test_malformed_metadata_is_explicit(tmp_path, line, message):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "metadata.jsonl").write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        BM25Retriever(tmp_path)


def test_bm25_eval_uses_parent_relevance_boundary_and_distinct_recall():
    case = {
        "query_id": "RAG-TEST",
        "question": "question",
        "query_type": "operating_model",
        "gold_evidence": [
            {"retrieval_unit_id": "parent-1", "evidence_group": "group-1"},
            {"retrieval_unit_id": "source-2", "evidence_group": "group-2"},
        ],
    }
    results = [
        {
            "rank": rank,
            "score": 1.0 / rank,
            "chunk_id": chunk_id,
            "doc_id": "doc-1",
            "file_name": "report.pdf",
            "pages": [rank],
            "chunk_type": "text",
            "headings": [],
            "text": "text",
            **({"parent_chunk_id": parent} if parent else {}),
        }
        for rank, chunk_id, parent in [
            (1, "irrelevant", None),
            (2, "parent-1__part_001", "parent-1"),
            (3, "parent-1__part_002", "parent-1"),
            (4, "other", None),
            (5, "last", None),
        ]
    ]

    record = score_case(case, results)

    assert retrieval_unit_id(results[1]) == "parent-1"
    assert record["retrieved_top5_ids"] == [
        "irrelevant",
        "parent-1",
        "parent-1",
        "other",
        "last",
    ]
    assert record["first_relevant_rank"] == 2
    assert record["recall_at_5"] == pytest.approx(0.5)
    assert record["mrr_at_5"] == pytest.approx(0.5)


def test_bm25_eval_equivalent_units_in_one_group_use_or_semantics():
    case = {
        "query_id": "RAG-TEST",
        "question": "question",
        "query_type": "financial_operating_metrics",
        "gold_evidence": [
            {"retrieval_unit_id": "equivalent-a", "evidence_group": "answer"},
            {"retrieval_unit_id": "equivalent-b", "evidence_group": "answer"},
        ],
    }
    results = [
        {
            "rank": 1,
            "score": 1.0,
            "chunk_id": "equivalent-b",
            "doc_id": "doc-1",
            "file_name": "report.pdf",
            "pages": [1],
            "chunk_type": "text",
            "headings": [],
            "text": "text",
        }
    ]

    record = score_case(case, results)

    assert record["recall_at_5"] == pytest.approx(1.0)
    assert record["covered_evidence_groups"] == ["answer"]
    assert record["missing_evidence_groups"] == []


def test_bm25_eval_aggregate_metrics_average_queries():
    records = [
        dict(hit_at_1=True, hit_at_3=True, hit_at_5=True, recall_at_5=1.0, mrr_at_5=1.0),
        dict(hit_at_1=False, hit_at_3=False, hit_at_5=False, recall_at_5=0.0, mrr_at_5=0.0),
    ]
    assert aggregate_metrics(records) == {
        "Hit@1": 0.5,
        "Hit@3": 0.5,
        "Hit@5": 0.5,
        "Recall@5": 0.5,
        "MRR@5": 0.5,
    }
