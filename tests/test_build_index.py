"""Tests for the JSONL-to-embedding-to-FAISS indexing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import faiss
import numpy as np
import pytest
import requests

from rag.embedding import EmbeddingClient
from rag.vector_store import FaissVectorStore
from scripts.build_index import CHECKPOINT_SCHEMA_VERSION, build_index, is_toc_chunk, read_chunks
from tests.test_fallback_splitter import CharTokenizer


DIMENSION = 1024


def _chunk(index: int) -> dict:
    return {
        "chunk_id": f"chunk-{index}",
        "doc_id": "doc-1",
        "file_name": "report.pdf",
        "pages": [index + 1],
        "chunk_type": "text",
        "headings": [f"Section {index}"],
        "text": f"raw text {index}",
        "embedding_text": f"contextual text {index}",
    }


def _write_jsonl(path: Path, chunks: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )


class FakeEmbeddingClient:
    dimension = DIMENSION
    model = "fake-embedding-model"
    instances: list["FakeEmbeddingClient"] = []

    def __init__(self, *, batch_size: int, timeout: float, max_retries: int):
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls: list[tuple[list[str], str]] = []
        type(self).instances.append(self)

    def embed_texts(self, texts: list[str], text_type: str = "document"):
        self.calls.append((texts, text_type))
        vectors = []
        for text in texts:
            index = int(text.rsplit(" ", 1)[1])
            vector = [0.0] * DIMENSION
            vector[index % DIMENSION] = 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture(autouse=True)
def _clear_fake_instances():
    FakeEmbeddingClient.instances.clear()


def test_read_chunks_preserves_jsonl_order(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    _write_jsonl(input_path, [_chunk(0), _chunk(1), _chunk(2)])

    chunks = read_chunks(input_path)

    assert [chunk["chunk_id"] for chunk in chunks] == ["chunk-0", "chunk-1", "chunk-2"]


def test_limit_controls_processed_chunk_count(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "index"
    _write_jsonl(input_path, [_chunk(i) for i in range(5)])

    stats = build_index(
        input_path, output_dir, limit=2, client_factory=FakeEmbeddingClient
    )

    assert stats["input_chunks"] == 2
    assert stats["embedded_chunks"] == 2
    assert FaissVectorStore.load(output_dir).vector_count == 2


def test_pipeline_batches_and_uses_document_text_type(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    _write_jsonl(input_path, [_chunk(i) for i in range(5)])

    build_index(
        input_path,
        tmp_path / "index",
        batch_size=2,
        client_factory=FakeEmbeddingClient,
    )

    client = FakeEmbeddingClient.instances[0]
    assert client.batch_size == 2
    assert [len(texts) for texts, _ in client.calls] == [2, 2, 1]
    assert [text_type for _, text_type in client.calls] == ["document"] * 3
    assert client.calls[0][0] == ["contextual text 0", "contextual text 1"]


def test_missing_embedding_text_is_rejected_before_client_creation(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    chunk = _chunk(0)
    del chunk["embedding_text"]
    _write_jsonl(input_path, [chunk])

    with pytest.raises(ValueError, match="missing required fields.*embedding_text"):
        build_index(input_path, tmp_path / "index", client_factory=FakeEmbeddingClient)

    assert FakeEmbeddingClient.instances == []


def test_duplicate_chunk_id_is_rejected_before_embedding(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    _write_jsonl(input_path, [_chunk(0), _chunk(0)])

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        build_index(input_path, tmp_path / "index", client_factory=FakeEmbeddingClient)

    assert FakeEmbeddingClient.instances == []


def test_embedding_failure_does_not_publish_partial_index(tmp_path):
    class FailingClient(FakeEmbeddingClient):
        def embed_texts(self, texts, text_type="document"):
            if self.calls:
                raise RuntimeError("mock API failure")
            return super().embed_texts(texts, text_type=text_type)

    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "index"
    _write_jsonl(input_path, [_chunk(i) for i in range(3)])

    with pytest.raises(RuntimeError, match="mock API failure"):
        build_index(
            input_path,
            output_dir,
            batch_size=2,
            client_factory=FailingClient,
        )

    assert (output_dir / ".build_tmp" / "progress.json").is_file()
    assert not any((output_dir / name).exists() for name in (
        "financial.faiss", "metadata.jsonl", "manifest.json"
    ))


def test_build_creates_expected_output_files_without_vectors_in_metadata(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "nested" / "index"
    _write_jsonl(input_path, [_chunk(0), _chunk(1)])

    stats = build_index(input_path, output_dir, client_factory=FakeEmbeddingClient)

    assert {path.name for path in output_dir.iterdir()} == {
        "financial.faiss",
        "metadata.jsonl",
        "manifest.json",
    }
    metadata_text = (output_dir / "metadata.jsonl").read_text(encoding="utf-8")
    assert '"embedding"' not in metadata_text
    assert '"embedding_text":"contextual text 0"' in metadata_text
    assert stats["output_dir"] == str(output_dir.resolve())


def test_existing_index_is_protected_without_overwrite(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "index"
    _write_jsonl(input_path, [_chunk(0)])
    build_index(input_path, output_dir, client_factory=FakeEmbeddingClient)
    FakeEmbeddingClient.instances.clear()

    with pytest.raises(FileExistsError, match="--overwrite"):
        build_index(input_path, output_dir, client_factory=FakeEmbeddingClient)

    assert FakeEmbeddingClient.instances == []


def test_overwrite_replaces_existing_index(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "index"
    _write_jsonl(input_path, [_chunk(0)])
    build_index(input_path, output_dir, client_factory=FakeEmbeddingClient)
    _write_jsonl(input_path, [_chunk(3), _chunk(4)])

    stats = build_index(
        input_path,
        output_dir,
        overwrite=True,
        client_factory=FakeEmbeddingClient,
    )

    assert stats["vector_count"] == 2
    loaded = FaissVectorStore.load(output_dir)
    assert loaded.search([0.0, 0.0, 0.0, 1.0] + [0.0] * 1020, top_k=1)[0][
        "chunk"
    ]["chunk_id"] == "chunk-3"


def test_vector_and_metadata_counts_match(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "index"
    _write_jsonl(input_path, [_chunk(i) for i in range(7)])

    stats = build_index(
        input_path,
        output_dir,
        batch_size=3,
        client_factory=FakeEmbeddingClient,
    )

    metadata_count = len(
        (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stats["input_chunks"] == 7
    assert stats["embedded_chunks"] == 7
    assert stats["vector_count"] == 7
    assert manifest["vector_count"] == 7
    assert metadata_count == 7


def test_embedding_count_mismatch_does_not_publish_index(tmp_path):
    class ShortClient(FakeEmbeddingClient):
        def embed_texts(self, texts, text_type="document"):
            return super().embed_texts(texts, text_type=text_type)[:-1]

    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "index"
    _write_jsonl(input_path, [_chunk(0), _chunk(1)])

    with pytest.raises(ValueError, match="embedding count mismatch"):
        build_index(input_path, output_dir, client_factory=ShortClient)

    assert not any((output_dir / name).exists() for name in (
        "financial.faiss", "metadata.jsonl", "manifest.json"
    ))


def test_failed_batch_checkpoint_resumes_without_reembedding_completed_chunks(tmp_path):
    class FailsSecondBatch(FakeEmbeddingClient):
        def embed_texts(self, texts, text_type="document"):
            if self.calls:
                self.calls.append((texts, text_type))
                raise RuntimeError("mock API failure")
            return super().embed_texts(texts, text_type=text_type)

    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "index"
    _write_jsonl(input_path, [_chunk(i) for i in range(5)])

    with pytest.raises(RuntimeError, match="next_input_position=2"):
        build_index(input_path, output_dir, batch_size=2, client_factory=FailsSecondBatch)

    progress = json.loads(
        (output_dir / ".build_tmp" / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["next_input_position"] == 2
    assert progress["embedded_chunks"] == 2
    assert progress["batches"][0]["vector_start"] == 0
    assert progress["batches"][0]["vector_end"] == 2

    stats = build_index(
        input_path, output_dir, batch_size=2, resume=True,
        client_factory=FakeEmbeddingClient,
    )

    resume_client = FakeEmbeddingClient.instances[-1]
    assert [texts for texts, _ in resume_client.calls] == [
        ["contextual text 2", "contextual text 3"], ["contextual text 4"]
    ]
    assert stats["vector_count"] == 5
    assert not (output_dir / ".build_tmp").exists()


def test_resumed_result_matches_one_shot_build(tmp_path):
    class FailsSecondBatch(FakeEmbeddingClient):
        def embed_texts(self, texts, text_type="document"):
            if self.calls:
                raise RuntimeError("mock API failure")
            return super().embed_texts(texts, text_type=text_type)

    input_path = tmp_path / "chunks.jsonl"
    resumed_dir = tmp_path / "resumed"
    one_shot_dir = tmp_path / "one-shot"
    _write_jsonl(input_path, [_chunk(i) for i in range(5)])
    with pytest.raises(RuntimeError):
        build_index(input_path, resumed_dir, batch_size=2, client_factory=FailsSecondBatch)
    build_index(input_path, resumed_dir, batch_size=2, resume=True,
                client_factory=FakeEmbeddingClient)
    build_index(input_path, one_shot_dir, batch_size=2,
                client_factory=FakeEmbeddingClient)

    assert (resumed_dir / "metadata.jsonl").read_text(encoding="utf-8") == (
        one_shot_dir / "metadata.jsonl").read_text(encoding="utf-8")
    resumed_index = faiss.read_index(str(resumed_dir / "financial.faiss"))
    one_shot_index = faiss.read_index(str(one_shot_dir / "financial.faiss"))
    np.testing.assert_array_equal(
        resumed_index.reconstruct_n(0, resumed_index.ntotal),
        one_shot_index.reconstruct_n(0, one_shot_index.ntotal),
    )


def _create_failed_checkpoint(input_path, output_dir):
    class FailsSecondBatch(FakeEmbeddingClient):
        def embed_texts(self, texts, text_type="document"):
            if self.calls:
                raise RuntimeError("mock API failure")
            return super().embed_texts(texts, text_type=text_type)

    with pytest.raises(RuntimeError):
        build_index(input_path, output_dir, batch_size=2,
                    client_factory=FailsSecondBatch)


def test_input_file_change_rejects_resume_before_embedding(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "index"
    _write_jsonl(input_path, [_chunk(i) for i in range(3)])
    _create_failed_checkpoint(input_path, output_dir)
    changed = [_chunk(i) for i in range(3)]
    changed[0]["embedding_text"] = "changed content 0"
    _write_jsonl(input_path, changed)
    FakeEmbeddingClient.instances.clear()

    with pytest.raises(ValueError, match="input_sha256 mismatch"):
        build_index(input_path, output_dir, batch_size=2, resume=True,
                    client_factory=FakeEmbeddingClient)
    assert FakeEmbeddingClient.instances[0].calls == []


@pytest.mark.parametrize("changed_field", ["model", "dimension"])
def test_model_or_dimension_change_rejects_resume(tmp_path, changed_field):
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "index"
    _write_jsonl(input_path, [_chunk(i) for i in range(3)])
    _create_failed_checkpoint(input_path, output_dir)

    class ChangedClient(FakeEmbeddingClient):
        model = "different-model" if changed_field == "model" else FakeEmbeddingClient.model
        dimension = 512 if changed_field == "dimension" else FakeEmbeddingClient.dimension

    with pytest.raises(ValueError, match=f"{changed_field}.*mismatch"):
        build_index(input_path, output_dir, batch_size=2, resume=True,
                    client_factory=ChangedClient)
    assert ChangedClient.instances[-1].calls == []


def test_batch_size_cannot_exceed_synchronous_api_limit(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    _write_jsonl(input_path, [_chunk(0)])

    with pytest.raises(ValueError, match="must not exceed.*20"):
        build_index(input_path, tmp_path / "index", batch_size=21,
                    client_factory=FakeEmbeddingClient)
    assert FakeEmbeddingClient.instances == []


def test_timeout_and_max_retries_are_passed_to_embedding_client(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    _write_jsonl(input_path, [_chunk(0)])

    build_index(
        input_path,
        tmp_path / "index",
        timeout=20,
        max_retries=0,
        client_factory=FakeEmbeddingClient,
    )

    client = FakeEmbeddingClient.instances[0]
    assert client.timeout == 20.0
    assert client.max_retries == 0


def test_zero_retries_exits_after_first_failure_and_keeps_checkpoint(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    output_dir = tmp_path / "index"
    _write_jsonl(input_path, [_chunk(0)])
    request_post = Mock(side_effect=requests.ReadTimeout("mock timeout"))

    def client_factory(**kwargs):
        return EmbeddingClient(
            api_key="mock-key",
            request_post=request_post,
            sleep=Mock(),
            **kwargs,
        )

    with pytest.raises(RuntimeError, match="mock timeout"):
        build_index(
            input_path,
            output_dir,
            timeout=20,
            max_retries=0,
            client_factory=client_factory,
            fallback_tokenizer_factory=CharTokenizer,
        )

    assert request_post.call_count == 1
    assert request_post.call_args.kwargs["timeout"] == (5.0, 20.0)
    progress = json.loads(
        (output_dir / ".build_tmp" / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["next_input_position"] == 0
    assert progress["embedded_chunks"] == 0
    assert progress["skipped_toc_chunks"] == 0
    assert progress["batches"] == []
    assert not any((output_dir / name).exists() for name in (
        "financial.faiss", "metadata.jsonl", "manifest.json"
    ))


@pytest.mark.parametrize("heading", ["目录", "目 录"])
def test_toc_heading_is_skipped(heading):
    chunk = _chunk(0)
    chunk["headings"] = [heading]
    assert is_toc_chunk(chunk)


@pytest.mark.parametrize("heading", ["第一章 目录说明", "产品目录管理", "目录页说明", "目\u3000录"])
def test_normal_heading_is_not_skipped(heading):
    chunk = _chunk(0)
    chunk["headings"] = [heading]
    assert not is_toc_chunk(chunk)


def test_skipped_toc_is_not_embedded_and_stats_are_persisted(tmp_path):
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    chunks[1]["headings"] = ["目 录"]
    input_path, output_dir = tmp_path / "chunks.jsonl", tmp_path / "index"
    _write_jsonl(input_path, chunks)

    stats = build_index(input_path, output_dir, batch_size=2,
                        client_factory=FakeEmbeddingClient)

    client = FakeEmbeddingClient.instances[0]
    assert client.calls == [(["contextual text 0", "contextual text 2"], "document")]
    assert stats["source_chunks"] == 3
    assert stats["indexed_chunks"] == stats["embedded_chunks"] == 2
    assert stats["skipped_toc_chunks"] == 1
    assert stats["skipped_other_chunks"] == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert {key: manifest[key] for key in (
        "source_chunks", "indexed_chunks", "skipped_toc_chunks", "skipped_other_chunks"
    )} == {"source_chunks": 3, "indexed_chunks": 2,
           "skipped_toc_chunks": 1, "skipped_other_chunks": 0}


def test_checkpoint_resume_crosses_skipped_chunk(tmp_path):
    class FailsAfterFirstBatch(FakeEmbeddingClient):
        def embed_texts(self, texts, text_type="document"):
            if self.calls:
                self.calls.append((texts, text_type))
                raise RuntimeError("mock API failure")
            return super().embed_texts(texts, text_type=text_type)

    chunks = [_chunk(i) for i in range(5)]
    chunks[1]["headings"] = ["目录"]
    input_path, output_dir = tmp_path / "chunks.jsonl", tmp_path / "index"
    _write_jsonl(input_path, chunks)
    with pytest.raises(RuntimeError):
        build_index(input_path, output_dir, batch_size=2,
                    client_factory=FailsAfterFirstBatch)

    progress = json.loads((output_dir / ".build_tmp" / "progress.json").read_text(
        encoding="utf-8"))
    assert progress["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert progress["next_input_position"] == 3
    assert progress["embedded_chunks"] == 2
    assert progress["skipped_toc_chunks"] == 1

    stats = build_index(input_path, output_dir, batch_size=2, resume=True,
                        client_factory=FakeEmbeddingClient)
    assert FakeEmbeddingClient.instances[-1].calls == [
        (["contextual text 3", "contextual text 4"], "document")
    ]
    assert stats["indexed_chunks"] == 4
    assert stats["skipped_toc_chunks"] == 1


def test_old_checkpoint_schema_is_explicitly_rejected(tmp_path):
    input_path, output_dir = tmp_path / "chunks.jsonl", tmp_path / "index"
    _write_jsonl(input_path, [_chunk(0)])
    checkpoint_dir = output_dir / ".build_tmp"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "progress.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported checkpoint schema_version.*--overwrite"):
        build_index(input_path, output_dir, resume=True,
                    client_factory=FakeEmbeddingClient)


def _pathological_table(index: int = 0) -> dict:
    chunk = _chunk(index)
    chunk["chunk_type"] = "table"
    chunk["pages"] = [14]
    chunk["headings"] = ["本招股意向书中，下列词汇具有如下含义："]
    rows = ["术语|含义"] + [f"词汇{row:02d}|" + "释义内容" * 30 for row in range(12)]
    chunk["text"] = "\n".join(rows)
    chunk["embedding_text"] = chunk["text"]
    return chunk


class TimeoutLargeClient(FakeEmbeddingClient):
    def embed_texts(self, texts, text_type="document"):
        self.calls.append((texts, text_type))
        if any(len(text) > 800 for text in texts):
            raise requests.ReadTimeout("pathological input")
        vectors = []
        for call_index, _ in enumerate(texts):
            vector = [0.0] * DIMENSION
            vector[(len(self.calls) + call_index) % DIMENSION] = 1.0
            vectors.append(vector)
        return vectors


def test_normal_success_never_loads_or_calls_fallback(tmp_path):
    input_path = tmp_path / "chunks.jsonl"
    _write_jsonl(input_path, [_chunk(0)])
    fallback_factory = Mock(side_effect=AssertionError("fallback must stay lazy"))

    build_index(input_path, tmp_path / "index",
                client_factory=FakeEmbeddingClient,
                fallback_tokenizer_factory=fallback_factory)

    fallback_factory.assert_not_called()


def test_transient_batch_failure_narrows_to_singletons_before_fallback(tmp_path):
    class BatchOnlyTimeout(FakeEmbeddingClient):
        def embed_texts(self, texts, text_type="document"):
            if len(texts) > 1:
                self.calls.append((texts, text_type))
                raise requests.ReadTimeout("batch-only timeout")
            return super().embed_texts(texts, text_type=text_type)

    input_path = tmp_path / "chunks.jsonl"
    _write_jsonl(input_path, [_chunk(0), _chunk(1)])
    fallback_factory = Mock(side_effect=AssertionError("content fallback not needed"))

    stats = build_index(
        input_path, tmp_path / "index", batch_size=2,
        client_factory=BatchOnlyTimeout,
        fallback_tokenizer_factory=fallback_factory,
    )

    client = BatchOnlyTimeout.instances[-1]
    assert [len(texts) for texts, _ in client.calls] == [2, 1, 1]
    assert stats["indexed_chunks"] == 2
    fallback_factory.assert_not_called()


def test_single_timeout_indexes_children_not_parent_and_restores_relation(tmp_path):
    parent = _pathological_table()
    input_path, output_dir = tmp_path / "chunks.jsonl", tmp_path / "index"
    _write_jsonl(input_path, [parent])

    stats = build_index(
        input_path, output_dir, batch_size=1,
        client_factory=TimeoutLargeClient,
        fallback_tokenizer_factory=CharTokenizer,
    )

    client = TimeoutLargeClient.instances[-1]
    assert len(client.calls[0][0]) == 1
    assert client.calls[0][0][0] == parent["embedding_text"]
    assert stats["indexed_chunks"] > 1
    metadata = [json.loads(line)["chunk"] for line in (
        output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert parent["chunk_id"] not in {chunk["chunk_id"] for chunk in metadata}
    assert all(chunk["parent_chunk_id"] == parent["chunk_id"] for chunk in metadata)
    assert all(chunk["fallback_split"] is True for chunk in metadata)
    loaded = FaissVectorStore.load(output_dir)
    assert loaded.vector_count == len(metadata)
    result = loaded.search([0.0, 1.0] + [0.0] * 1022, top_k=1)[0]["chunk"]
    assert result["parent_chunk_id"] == parent["chunk_id"]


def test_sub_768_token_timeout_uses_second_level_budget(tmp_path):
    class TimeoutOver500(FakeEmbeddingClient):
        def embed_texts(self, texts, text_type="document"):
            self.calls.append((texts, text_type))
            if any(len(text) > 500 for text in texts):
                raise requests.ReadTimeout("sub-768 pathological input")
            vectors = []
            for offset, _ in enumerate(texts):
                vector = [0.0] * DIMENSION
                vector[(len(self.calls) + offset) % DIMENSION] = 1.0
                vectors.append(vector)
            return vectors

    parent = _chunk(0)
    parent["chunk_type"] = "table"
    parent["text"] = "。".join(["释义内容" * 10] * 14) + "。"
    parent["embedding_text"] = parent["text"]
    assert 384 < len(parent["embedding_text"]) < 768
    input_path, output_dir = tmp_path / "chunks.jsonl", tmp_path / "index"
    _write_jsonl(input_path, [parent])

    stats = build_index(
        input_path, output_dir, batch_size=1,
        client_factory=TimeoutOver500,
        fallback_tokenizer_factory=CharTokenizer,
    )

    assert stats["indexed_chunks"] > 1
    metadata = [json.loads(line)["chunk"] for line in (
        output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(len(chunk["embedding_text"]) <= 384 for chunk in metadata)
    assert all(chunk["parent_chunk_id"] == parent["chunk_id"] for chunk in metadata)


def test_children_complete_before_checkpoint_advances_parent(tmp_path):
    class FailsNextSource(TimeoutLargeClient):
        def embed_texts(self, texts, text_type="document"):
            if texts == ["contextual text 1"]:
                raise RuntimeError("terminal next-source failure")
            return super().embed_texts(texts, text_type=text_type)

    input_path, output_dir = tmp_path / "chunks.jsonl", tmp_path / "index"
    _write_jsonl(input_path, [_pathological_table(), _chunk(1)])

    with pytest.raises(RuntimeError, match="terminal next-source failure"):
        build_index(
            input_path, output_dir, batch_size=1,
            client_factory=FailsNextSource,
            fallback_tokenizer_factory=CharTokenizer,
        )

    progress = json.loads((output_dir / ".build_tmp" / "progress.json").read_text(
        encoding="utf-8"))
    assert progress["next_input_position"] == 1
    assert progress["embedded_chunks"] > 1
    assert progress["batches"][0]["input_positions"] == [
        0
    ] * progress["embedded_chunks"]
    records = [json.loads(line) for line in (
        output_dir / ".build_tmp" / progress["batches"][0]["records_file"]
    ).read_text(encoding="utf-8").splitlines()]
    assert all(record["chunk"]["parent_chunk_id"] == "chunk-0" for record in records)


def test_fallback_depth_limit_fails_without_silent_skip(tmp_path):
    class AlwaysTimeout(FakeEmbeddingClient):
        def embed_texts(self, texts, text_type="document"):
            self.calls.append((texts, text_type))
            raise requests.ReadTimeout("still pathological")

    input_path, output_dir = tmp_path / "chunks.jsonl", tmp_path / "index"
    _write_jsonl(input_path, [_pathological_table()])

    with pytest.raises(RuntimeError, match="max_fallback_depth=2"):
        build_index(
            input_path, output_dir, batch_size=1,
            client_factory=AlwaysTimeout,
            fallback_tokenizer_factory=CharTokenizer,
        )

    progress = json.loads((output_dir / ".build_tmp" / "progress.json").read_text(
        encoding="utf-8"))
    assert progress["next_input_position"] == 0
    assert progress["embedded_chunks"] == 0
    assert progress["batches"] == []
    assert not (output_dir / "financial.faiss").exists()
