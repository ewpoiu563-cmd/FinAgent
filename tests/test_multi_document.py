"""Small mocked tests for independently runnable multi-document stages."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag.document_processor import ProcessingResult
from rag.multi_document import (
    build_multi_document_index,
    embed_chunks_to_artifact,
    load_embedding_artifact,
    stage_a_pdf_to_chunks,
)
from rag.vector_store import FaissVectorStore


def _chunk(doc_id: str, index: int, file_name: str) -> dict:
    return {
        "chunk_id": f"{doc_id}-{index}", "doc_id": doc_id, "file_name": file_name,
        "pages": [1], "chunk_type": "text", "headings": ["正文"],
        "text": f"text {index}", "embedding_text": f"context {index}",
    }


def _write_jsonl(path: Path, chunks: list[dict]) -> None:
    path.write_text("".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks), encoding="utf-8")


def _selection(path: Path, doc_id: str, file_name: str) -> None:
    path.write_text(json.dumps({"schema_version": 1, "documents": [{"doc_id": doc_id, "source_file": file_name}]}), encoding="utf-8")


class _CharTokenizer:
    def count_tokens(self, text):
        return len(text)

    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, tokens, **kwargs):
        return "".join(tokens)


def test_stage_a_requires_selected_single_pdf_and_writes_summary(tmp_path):
    pdf, selection, output = tmp_path / "chosen.pdf", tmp_path / "selection.json", tmp_path / "chunks.jsonl"
    pdf.write_bytes(b"not parsed by this test")
    _selection(selection, "doc-new", pdf.name)

    def fake_processor(path, *, doc_id):
        assert Path(path) == pdf
        assert doc_id == "doc-new"
        normal = _chunk(doc_id, 0, pdf.name)
        toc = _chunk(doc_id, 1, pdf.name)
        toc["headings"] = ["目录"]
        oversized = _chunk(doc_id, 2, pdf.name)
        oversized["text"] = oversized["embedding_text"] = "a" * 800
        return ProcessingResult(
            chunks=[normal, toc, oversized], pictures=[],
            stats={"page_count": 2, "picture_count": 1, "skipped_picture_chunks": 0, "emitted_chunks": 3, "oversized_chunk_count": 1},
            timing={"parse_seconds": 1.25, "chunk_seconds": 0.5, "fallback_split_seconds": 0.0, "elapsed_seconds": 1.75},
        )

    summary = stage_a_pdf_to_chunks(pdf, output, selection_manifest=selection, processor=fake_processor, fallback_tokenizer_factory=_CharTokenizer)
    assert summary["doc_id"] == "doc-new"
    assert summary["source_chunk_count"] == 3
    assert summary["skipped_toc_chunks"] == 1
    assert summary["skipped_other_chunks"] == 0
    assert summary["fallback_parent_count"] == 1
    assert summary["fallback_child_count"] == 2
    assert summary["final_chunk_count"] == 3
    assert summary["parse_seconds"] == 1.25
    final_chunks = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [chunk["chunk_id"] for chunk in final_chunks] == ["doc-new-0", "doc-new-2__part_001", "doc-new-2__part_002"]
    assert json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))["total_seconds"] >= 0
    with pytest.raises(FileExistsError, match="--force"):
        stage_a_pdf_to_chunks(pdf, output, selection_manifest=selection, processor=fake_processor, fallback_tokenizer_factory=_CharTokenizer)


class _FakeEmbeddingClient:
    model = "qwen3.7-text-embedding-flash"
    dimension = 1024
    instances: list["_FakeEmbeddingClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.__class__.instances.append(self)

    def embed_texts(self, texts, text_type="document"):
        self.calls.append((texts, text_type))
        result = []
        for offset, _ in enumerate(texts):
            vector = [0.0] * self.dimension
            vector[offset] = 1.0
            result.append(vector)
        return result


def test_stage_b_persists_reusable_artifact_and_refuses_repeat_api(tmp_path):
    chunks_path, artifact, selection = tmp_path / "chunks.jsonl", tmp_path / "artifact", tmp_path / "selection.json"
    _write_jsonl(chunks_path, [_chunk("doc-new", 0, "chosen.pdf"), _chunk("doc-new", 1, "chosen.pdf")])
    _selection(selection, "doc-new", "chosen.pdf")
    _FakeEmbeddingClient.instances.clear()
    manifest = embed_chunks_to_artifact(chunks_path, artifact, batch_size=1, client_factory=_FakeEmbeddingClient, selection_manifest=selection)
    assert manifest["vector_count"] == 2
    assert _FakeEmbeddingClient.instances[0].calls == [(["context 0"], "document"), (["context 1"], "document")]
    loaded_manifest, vectors, chunks = load_embedding_artifact(artifact)
    assert loaded_manifest["doc_id"] == "doc-new"
    assert loaded_manifest["chunk_count"] == loaded_manifest["vector_count"] == 2
    assert loaded_manifest["completed_batches"] == 2
    assert vectors.shape == (2, 1024)
    assert [chunk["chunk_id"] for chunk in chunks] == ["doc-new-0", "doc-new-1"]
    with pytest.raises(FileExistsError, match="refusing API reuse"):
        embed_chunks_to_artifact(chunks_path, artifact, client_factory=_FakeEmbeddingClient, selection_manifest=selection)


def test_stage_b_rejects_unfinalized_toc_chunks_without_filtering(tmp_path):
    chunks_path, artifact, selection = tmp_path / "chunks.jsonl", tmp_path / "artifact", tmp_path / "selection.json"
    toc = _chunk("doc-new", 0, "chosen.pdf")
    toc["headings"] = ["目录"]
    _write_jsonl(chunks_path, [toc])
    _selection(selection, "doc-new", "chosen.pdf")
    with pytest.raises(ValueError, match="already be final"):
        embed_chunks_to_artifact(chunks_path, artifact, client_factory=_FakeEmbeddingClient, selection_manifest=selection)


def test_stage_c_offline_merges_cached_artifact_and_rejects_duplicate_doc_id(tmp_path):
    base_dir, artifact_dir, output = tmp_path / "base", tmp_path / "artifact", tmp_path / "multi"
    base = FaissVectorStore()
    base.add(np.eye(1, 1024, dtype=np.float32), [_chunk("doc-base", 0, "base.pdf")])
    base.save(base_dir)
    artifact_dir.mkdir()
    vector = np.zeros((1, 1024), dtype=np.float32)
    vector[0, 1] = 1.0
    np.save(artifact_dir / "vectors.npy", vector, allow_pickle=False)
    (artifact_dir / "records.jsonl").write_text(json.dumps({"position": 0, "input_position": 0, "chunk": _chunk("doc-new", 0, "new.pdf")}) + "\n", encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(json.dumps({"schema_version": 1, "artifact_type": "document_embeddings", "doc_id": "doc-new", "source_file": "new.pdf", "source_chunks_path": "C:/mock/new.jsonl", "source_chunks_sha256": "mock", "chunk_count": 1, "embedding_model": "qwen3.7-text-embedding-flash", "dimension": 1024, "text_type": "document", "vector_count": 1, "completed_batches": 1}), encoding="utf-8")
    result = build_multi_document_index(base_dir, [artifact_dir], output)
    assert result["document_count"] == 2
    assert result["vector_count"] == 2
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["doc_ids"] == ["doc-base", "doc-new"]
    with pytest.raises(ValueError, match="duplicate doc_id"):
        build_multi_document_index(base_dir, [artifact_dir, artifact_dir], tmp_path / "duplicate")
