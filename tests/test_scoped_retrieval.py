"""Offline scoped Dense/BM25/Hybrid/reranker/tool tests."""

import json
import logging
from unittest.mock import Mock

from rag.bm25_retriever import BM25Retriever
from rag.dense_retriever import DenseRetriever
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import RerankerRetriever
from tools.document_retrieval_tool import DocumentRetrievalTool


JIANFAN = "6c93724343cef6b9da70d51daa739e8a1c607a22-5d11f946465d"
OTHER_DOCS = {"doc_xinlitai", "doc_yaxia", "doc_sinocera"}


def _chunk(doc_id: str, index: int) -> dict:
    return {
        "chunk_id": f"{doc_id}-chunk-{index}",
        "doc_id": doc_id,
        "file_name": "6c93724343cef6b9da70d51daa739e8a1c607a22.pdf" if doc_id == JIANFAN else f"{doc_id}.pdf",
        "pages": [index],
        "chunk_type": "text",
        "headings": ["风险因素"],
        "text": f"{doc_id} evidence {index}",
    }


def _ranked(doc_id: str, index: int, rank: int | None = None) -> dict:
    result = _chunk(doc_id, index)
    result.update(
        rank=rank or index,
        score=1.0 / index,
        rrf_score=1.0 / (60 + index),
        retrieval_unit_id=result["chunk_id"],
        dense_rank=rank or index,
        bm25_rank=None,
        contributing_retrievers=["dense"],
    )
    return result


class _FakeVectorStore:
    dimension = 2

    def __init__(self, chunks):
        self.matches = [
            {"score": 1.0 - position / 100, "chunk": chunk}
            for position, chunk in enumerate(chunks)
        ]
        self.vector_count = len(self.matches)
        self.calls = []

    def search(self, vector, top_k=5):
        self.calls.append(top_k)
        return self.matches[:top_k]


def test_dense_scope_overretrieves_once_embedded_and_returns_only_allowed_ids():
    chunks = [_chunk("doc_sinocera", index) for index in range(1, 6)]
    chunks += [_chunk(JIANFAN, 6), _chunk(JIANFAN, 7)]
    store = _FakeVectorStore(chunks)
    embedding = Mock()
    embedding.embed_texts.return_value = [[1.0, 0.0]]
    retriever = object.__new__(DenseRetriever)
    retriever.vector_store = store
    retriever.embedding_client = embedding

    results = retriever.retrieve("经营风险", top_k=2, allowed_doc_ids={JIANFAN})

    assert len(results) == 2
    assert {item["doc_id"] for item in results} == {JIANFAN}
    assert [item["rank"] for item in results] == [1, 2]
    embedding.embed_texts.assert_called_once_with(["经营风险"], text_type="query")
    assert store.calls == [7]


class _FakeBM25:
    def get_scores(self, tokens):
        return [100.0, 90.0, 80.0, 70.0]


def test_bm25_scope_filters_metadata_positions_before_top_k_selection(caplog):
    caplog.set_level(logging.DEBUG, logger="rag.bm25_retriever")
    retriever = object.__new__(BM25Retriever)
    retriever._chunks = [
        _chunk("doc_sinocera", 1),
        _chunk(JIANFAN, 2),
        _chunk("doc_xinlitai", 3),
        _chunk(JIANFAN, 4),
    ]
    retriever._positions_by_doc_id = {
        "doc_sinocera": [0], JIANFAN: [1, 3], "doc_xinlitai": [2]
    }
    retriever._bm25 = _FakeBM25()

    results = retriever.retrieve("经营风险", top_k=2, allowed_doc_ids={JIANFAN})

    assert [item["chunk_id"] for item in results] == [f"{JIANFAN}-chunk-2", f"{JIANFAN}-chunk-4"]
    assert {item["doc_id"] for item in results} == {JIANFAN}
    record = next(item for item in caplog.records if item.message.startswith("SOURCE_SCOPE_FILTER:"))
    payload = json.loads(record.message.split(": ", 1)[1])
    assert payload["candidate_count_before_scope"] == 4
    assert payload["candidate_count_after_scope"] == 2


class _ScopedRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k=5, allowed_doc_ids=None):
        self.calls.append((query, top_k, allowed_doc_ids))
        # Include one bad result to verify Hybrid's defensive boundary.
        return self.results[:top_k] + [_ranked("doc_sinocera", 99, rank=99)]


def test_hybrid_scope_filters_both_candidate_lists_before_rrf():
    dense = _ScopedRetriever([_ranked(JIANFAN, 1)])
    bm25 = _ScopedRetriever([_ranked(JIANFAN, 2)])
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results = hybrid.retrieve("风险", top_k=2, candidate_k=20, allowed_doc_ids={JIANFAN})

    assert results
    assert {item["doc_id"] for item in results} == {JIANFAN}
    assert dense.calls[0][2] == frozenset({JIANFAN})
    assert bm25.calls[0][2] == frozenset({JIANFAN})


def test_reranker_never_sends_or_returns_out_of_scope_candidates():
    client = Mock()
    client.rerank.side_effect = lambda query, documents: [float(i) for i in range(len(documents))]
    retriever = RerankerRetriever(Mock(), reranker_client=client)
    candidates = [
        _ranked("doc_sinocera", 1),
        _ranked(JIANFAN, 2),
        _ranked("doc_xinlitai", 3),
        _ranked(JIANFAN, 4),
    ]

    results = retriever.rerank_candidates(
        "风险", candidates, final_k=5, allowed_doc_ids={JIANFAN}
    )

    sent_documents = client.rerank.call_args.args[1]
    assert len(sent_documents) == 2
    assert all(JIANFAN in text for text in sent_documents)
    assert {item["doc_id"] for item in results} == {JIANFAN}


def test_document_tool_scoped_to_jianfan_filters_before_rerank_and_final_top5():
    pool = [
        _ranked("doc_sinocera", 1),
        _ranked(JIANFAN, 2),
        _ranked("doc_xinlitai", 3),
        _ranked(JIANFAN, 4),
        _ranked("doc_yaxia", 5),
        _ranked(JIANFAN, 6),
    ]
    hybrid = Mock()
    hybrid.retrieve.return_value = pool
    reranker = Mock()

    def rerank(query, candidates, *, final_k, allowed_doc_ids):
        assert {item["doc_id"] for item in candidates} == {JIANFAN}
        assert allowed_doc_ids == frozenset({JIANFAN})
        return [dict(item, rerank_score=0.9) for item in candidates[:final_k]]

    reranker.rerank_candidates.side_effect = rerank
    tool = DocumentRetrievalTool(
        hybrid_retriever=hybrid,
        reranker_retriever=reranker,
        timeout_runner=lambda call, timeout: call(),
    )

    result = tool.retrieve("健帆经营风险", allowed_doc_ids={JIANFAN})

    assert result["success"] is True
    assert result["allowed_doc_ids"] == [JIANFAN]
    assert len(result["evidence"]) == 3
    assert {item["doc_id"] for item in result["evidence"]} == {JIANFAN}
    assert not ({item["doc_id"] for item in result["evidence"]} & OTHER_DOCS)
    hybrid.retrieve.assert_called_once_with(
        "健帆经营风险", top_k=20, candidate_k=20, allowed_doc_ids=frozenset({JIANFAN})
    )
