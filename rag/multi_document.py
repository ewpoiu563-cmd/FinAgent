"""Independent, cacheable stages for the manually selected PDF corpus.

This module never enumerates a PDF directory.  Stage A accepts exactly one
path and authorizes it against a small selection manifest.  Stage B persists
vectors for the final chunks unchanged.  Stage C only
reads those artifacts; it has no embedding client dependency.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rag.document_processor import ProcessingResult, process_pdf, write_jsonl
from rag.embedding import DEFAULT_BATCH_SIZE, DEFAULT_DIMENSION, DEFAULT_MODEL, DEFAULT_TIMEOUT, EmbeddingClient
from rag.fallback_splitter import (
    DEFAULT_FALLBACK_MAX_TOKENS,
    build_fallback_tokenizer,
    finalize_children,
    split_chunk,
)
from rag.vector_store import INDEX_FILE_NAME, MANIFEST_FILE_NAME, METADATA_FILE_NAME, FaissVectorStore
from scripts.build_index import (
    CHECKPOINT_FILE_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_BUILD_MAX_RETRIES,
    _atomic_write_batch,
    _atomic_write_json,
    _input_identity,
    _load_progress,
    _validate_batch_size,
    _validate_request_settings,
    is_toc_chunk,
    read_chunks,
)

SELECTION_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST = "manifest.json"
ARTIFACT_VECTORS = "vectors.npy"
ARTIFACT_RECORDS = "records.jsonl"
ARTIFACT_CHECKPOINT_DIR = ".embedding_tmp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected a JSON object")
    return value


def selected_document(manifest_path: str | Path, pdf_path: str | Path) -> dict[str, str]:
    """Return the one manifest entry matching ``pdf_path`` without directory scans."""

    manifest = _load_json(Path(manifest_path), "document selection manifest")
    if manifest.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported document selection manifest schema_version")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError("document selection manifest documents must be a list")
    candidate = Path(pdf_path)
    matches: list[dict[str, str]] = []
    for index, entry in enumerate(documents):
        if not isinstance(entry, Mapping):
            raise ValueError(f"manifest documents[{index}] must be an object")
        doc_id, source_file = entry.get("doc_id"), entry.get("source_file")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"manifest documents[{index}].doc_id must be a non-empty string")
        if not isinstance(source_file, str) or not source_file.strip():
            raise ValueError(f"manifest documents[{index}].source_file must be a non-empty string")
        if Path(source_file).name == candidate.name:
            matches.append({"doc_id": doc_id, "source_file": Path(source_file).name})
    if len(matches) != 1:
        raise ValueError(
            "the explicit PDF must appear exactly once in the document selection "
            f"manifest by filename: {candidate.name}"
        )
    return matches[0]


def _authorize_chunk_identity(manifest_path: str | Path, *, doc_id: str, source_file: str) -> None:
    """Require Stage-B chunks to identify one explicitly selected source."""

    manifest = _load_json(Path(manifest_path), "document selection manifest")
    documents = manifest.get("documents")
    if manifest.get("schema_version") != SELECTION_SCHEMA_VERSION or not isinstance(documents, list):
        raise ValueError("unsupported document selection manifest")
    matches = [
        entry for entry in documents if isinstance(entry, Mapping)
        and entry.get("doc_id") == doc_id
        and Path(str(entry.get("source_file", ""))).name == source_file
    ]
    if len(matches) != 1:
        raise ValueError("chunks doc_id/source_file must appear exactly once in the document selection manifest")


def _summary_path(chunk_path: Path) -> Path:
    return chunk_path.with_suffix(".summary.json")


def _finalize_stage_a_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    tokenizer_provider: Callable[[], Any],
) -> tuple[list[dict[str, Any]], int, int, int, float]:
    """Run all deterministic pre-embedding work and return final chunks."""

    final_chunks: list[dict[str, Any]] = []
    skipped_toc = 0
    fallback_parent_count = 0
    fallback_child_count = 0
    fallback_seconds = 0.0
    tokenizer: Any | None = None

    def get_tokenizer() -> Any:
        nonlocal tokenizer
        if tokenizer is None:
            tokenizer = tokenizer_provider()
        return tokenizer

    for chunk in chunks:
        if is_toc_chunk(chunk):
            skipped_toc += 1
            continue
        if int(get_tokenizer().count_tokens(str(chunk["embedding_text"]))) <= DEFAULT_FALLBACK_MAX_TOKENS:
            final_chunks.append(dict(chunk))
            continue
        split_started_at = time.perf_counter()
        children = split_chunk(chunk, get_tokenizer(), max_tokens=DEFAULT_FALLBACK_MAX_TOKENS)
        fallback_seconds += time.perf_counter() - split_started_at
        if len(children) <= 1:
            raise RuntimeError(f"fallback split did not reduce pathological chunk: {chunk['chunk_id']}")
        if any(int(get_tokenizer().count_tokens(str(child["embedding_text"]))) > DEFAULT_FALLBACK_MAX_TOKENS for child in children):
            raise RuntimeError(f"fallback split left an oversized chunk: {chunk['chunk_id']}")
        fallback_parent_count += 1
        fallback_child_count += len(children)
        final_chunks.extend(finalize_children(chunk, children))
    chunk_ids = [chunk["chunk_id"] for chunk in final_chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Stage A preprocessing produced duplicate chunk_id values")
    return final_chunks, skipped_toc, fallback_parent_count, fallback_child_count, fallback_seconds


def stage_a_pdf_to_chunks(
    pdf_path: str | Path,
    output_path: str | Path,
    *,
    selection_manifest: str | Path,
    force: bool = False,
    processor: Callable[..., ProcessingResult] = process_pdf,
    fallback_tokenizer_factory: Callable[[], Any] = build_fallback_tokenizer,
) -> dict[str, Any]:
    """Parse one selected PDF and emit final directly-embeddable chunks."""

    started_at = time.perf_counter()
    source, destination = Path(pdf_path), Path(output_path)
    selected = selected_document(selection_manifest, source)
    if destination.exists() and not force:
        raise FileExistsError(f"chunk output already exists: {destination}; use --force")
    result = processor(source, doc_id=selected["doc_id"])
    final_chunks, skipped_toc, fallback_parent_count, fallback_child_count, fallback_seconds = _finalize_stage_a_chunks(
        result.chunks, tokenizer_provider=fallback_tokenizer_factory)
    summary = {
        "doc_id": selected["doc_id"],
        "source_filename": source.name,
        "page_count": result.stats["page_count"],
        "source_chunk_count": len(result.chunks),
        "skipped_toc_chunks": skipped_toc,
        "skipped_other_chunks": 0,
        "fallback_parent_count": fallback_parent_count,
        "fallback_child_count": fallback_child_count,
        "final_chunk_count": len(final_chunks),
        "parse_seconds": (result.timing or {}).get("parse_seconds", 0.0),
        "chunk_seconds": (result.timing or {}).get("chunk_seconds", 0.0),
        "fallback_split_seconds": round(fallback_seconds, 3),
        "total_seconds": round(time.perf_counter() - started_at, 3),
    }
    # Do not touch an existing output until parsing has completed successfully.
    write_jsonl(final_chunks, destination)
    _atomic_write_json(_summary_path(destination), summary)
    return summary


def _artifact_files(destination: Path) -> list[Path]:
    return [destination / name for name in (ARTIFACT_MANIFEST, ARTIFACT_VECTORS, ARTIFACT_RECORDS)]


def _artifact_doc_identity(chunks: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    doc_ids = {str(chunk.get("doc_id", "")) for chunk in chunks}
    files = {str(chunk.get("file_name", "")) for chunk in chunks}
    if len(doc_ids) != 1 or "" in doc_ids:
        raise ValueError("chunks artifact input must contain exactly one non-empty doc_id")
    if len(files) != 1 or "" in files:
        raise ValueError("chunks artifact input must contain exactly one non-empty file_name")
    return next(iter(doc_ids)), next(iter(files))


def _read_checkpoint_records(checkpoint_dir: Path, progress: Mapping[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrices: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for batch in progress["batches"]:
        vectors = np.load(checkpoint_dir / batch["vectors_file"], allow_pickle=False)
        lines = (checkpoint_dir / batch["records_file"]).read_text(encoding="utf-8").splitlines()
        batch_records = [json.loads(line) for line in lines]
        if len(batch_records) != len(vectors):
            raise ValueError("checkpoint record/vector count mismatch")
        matrices.append(vectors)
        records.extend(batch_records)
    if not matrices:
        raise ValueError("embedding artifact contains no vectors")
    return np.concatenate(matrices, axis=0), records


def _new_embedding_progress(identity: Mapping[str, Any], *, model: str, dimension: int) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        **identity,
        "embedding_model": model,
        "dimension": dimension,
        "next_input_position": 0,
        "embedded_chunks": 0,
        "batches": [],
    }


def _validate_embedding_resume(
    progress: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    model: str,
    dimension: int,
    chunks: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path,
) -> None:
    expected = {"schema_version": CHECKPOINT_SCHEMA_VERSION, **identity, "embedding_model": model, "dimension": dimension}
    for field, expected_value in expected.items():
        if progress.get(field) != expected_value:
            raise ValueError(f"checkpoint {field} mismatch: expected {expected_value!r}, got {progress.get(field)!r}")
    next_position, embedded, batches = progress.get("next_input_position"), progress.get("embedded_chunks"), progress.get("batches")
    if not isinstance(next_position, int) or not isinstance(embedded, int) or not isinstance(batches, list):
        raise ValueError("invalid embedding checkpoint counters")
    if not 0 <= next_position <= len(chunks) or embedded != next_position:
        raise ValueError("embedding checkpoint must preserve one vector for every input chunk")
    expected_start = 0
    for batch_number, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise ValueError(f"invalid embedding checkpoint batch {batch_number}")
        start, end, positions = batch.get("vector_start"), batch.get("vector_end"), batch.get("input_positions")
        if not isinstance(end, int) or start != expected_start or end <= start or positions != list(range(start, end)):
            raise ValueError(f"invalid embedding checkpoint batch range at batch {batch_number}")
        vectors = np.load(checkpoint_dir / str(batch.get("vectors_file", "")), allow_pickle=False)
        records = [json.loads(line) for line in (checkpoint_dir / str(batch.get("records_file", ""))).read_text(encoding="utf-8").splitlines()]
        if vectors.shape != (end - start, dimension) or len(records) != end - start:
            raise ValueError(f"embedding checkpoint vector/record mismatch at batch {batch_number}")
        for offset, record in enumerate(records):
            position = start + offset
            if not isinstance(record, Mapping) or record.get("vector_position") != position or record.get("input_position") != position or record.get("chunk") != chunks[position]:
                raise ValueError(f"embedding checkpoint altered chunk data at batch {batch_number}")
        expected_start = end
    if expected_start != embedded:
        raise ValueError("embedding checkpoint batches do not cover embedded chunks")


def embed_chunks_to_artifact(
    chunks_path: str | Path,
    artifact_dir: str | Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_BUILD_MAX_RETRIES,
    resume: bool = False,
    selection_manifest: str | Path = Path("data/documents/multi_doc_v1.json"),
    client_factory: Callable[..., Any] = EmbeddingClient,
) -> dict[str, Any]:
    """Embed final Stage-A chunks unchanged and persist reusable vectors."""

    started_at = time.perf_counter()
    _validate_batch_size(batch_size)
    _validate_request_settings(timeout, max_retries)
    source, destination = Path(chunks_path), Path(artifact_dir)
    final_files = _artifact_files(destination)
    if any(path.exists() for path in final_files):
        raise FileExistsError(f"embedding artifact already exists: {destination}; refusing API reuse")
    checkpoint_dir = destination / ARTIFACT_CHECKPOINT_DIR
    progress_path = checkpoint_dir / CHECKPOINT_FILE_NAME
    if progress_path.exists() and not resume:
        raise FileExistsError("embedding checkpoint already exists; use --resume")
    chunks = read_chunks(source)
    doc_id, source_file = _artifact_doc_identity(chunks)
    _authorize_chunk_identity(selection_manifest, doc_id=doc_id, source_file=source_file)
    if any(is_toc_chunk(chunk) for chunk in chunks):
        raise ValueError("Stage B input must already be final chunks; TOC chunks are not allowed")
    expected_vector_count = len(chunks)
    identity = _input_identity(source, chunk_count=len(chunks), limit=None)
    progress = _load_progress(checkpoint_dir) if progress_path.exists() and resume else None
    client = client_factory(batch_size=batch_size, timeout=float(timeout), max_retries=max_retries)
    if client.model != DEFAULT_MODEL or client.dimension != DEFAULT_DIMENSION:
        raise ValueError(
            f"Stage B is frozen to {DEFAULT_MODEL} dimension={DEFAULT_DIMENSION}; "
            f"got model={client.model!r}, dimension={client.dimension!r}"
        )
    if progress is not None:
        _validate_embedding_resume(progress, identity, model=client.model, dimension=client.dimension,
                                   chunks=chunks, checkpoint_dir=checkpoint_dir)
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        progress = _new_embedding_progress(identity, model=client.model, dimension=client.dimension)
        _atomic_write_json(progress_path, progress)

    next_position, embedded = progress["next_input_position"], progress["embedded_chunks"]
    while next_position < len(chunks):
        batch = chunks[next_position : next_position + batch_size]
        vectors = client.embed_texts([str(chunk["embedding_text"]) for chunk in batch], text_type="document")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(batch), client.dimension) or not np.isfinite(matrix).all():
            raise ValueError("embedding response has invalid vector count, dimension, or numeric values")
        positions = list(range(next_position, next_position + len(batch)))
        progress["batches"].append(_atomic_write_batch(
            checkpoint_dir, vector_start=embedded, vectors=matrix,
            chunks=batch, input_positions=positions))
        embedded += len(batch)
        next_position += len(batch)
        progress["embedded_chunks"] = embedded
        progress["next_input_position"] = next_position
        _atomic_write_json(progress_path, progress)

    vectors, records = _read_checkpoint_records(checkpoint_dir, progress)
    if (next_position != expected_vector_count or embedded != expected_vector_count
            or len(records) != expected_vector_count
            or vectors.shape != (expected_vector_count, client.dimension)):
        raise RuntimeError("embedding artifact checkpoint count mismatch")
    final_chunks = [record["chunk"] for record in records]
    if final_chunks != chunks:
        raise RuntimeError("Stage B must not add, delete, split, filter, or alter chunks")
    destination.mkdir(parents=True, exist_ok=True)
    vectors_path, records_path = destination / ARTIFACT_VECTORS, destination / ARTIFACT_RECORDS
    with vectors_path.open("xb") as output:
        np.save(output, vectors, allow_pickle=False)
    with records_path.open("x", encoding="utf-8", newline="\n") as output:
        for position, record in enumerate(records):
            output.write(json.dumps({"position": position, "input_position": record["input_position"], "chunk": record["chunk"]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "document_embeddings",
        "doc_id": doc_id,
        "source_file": source_file,
        "source_chunks_path": identity["input_path"],
        "source_chunks_sha256": identity["input_sha256"],
        "chunk_count": len(chunks),
        "embedding_model": client.model,
        "dimension": client.dimension,
        "text_type": "document",
        "vector_count": embedded,
        "completed_batches": len(progress["batches"]),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    _atomic_write_json(destination / ARTIFACT_MANIFEST, manifest)
    shutil.rmtree(checkpoint_dir)
    return manifest


def load_embedding_artifact(artifact_dir: str | Path) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """Load an already-complete artifact and prove vector/metadata alignment."""

    source = Path(artifact_dir)
    manifest = _load_json(source / ARTIFACT_MANIFEST, "embedding artifact manifest")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION or manifest.get("artifact_type") != "document_embeddings":
        raise ValueError("unsupported embedding artifact manifest")
    if manifest.get("embedding_model") != DEFAULT_MODEL or manifest.get("dimension") != DEFAULT_DIMENSION or manifest.get("text_type") != "document":
        raise ValueError("embedding artifact does not use the frozen document embedding configuration")
    vectors = np.load(source / ARTIFACT_VECTORS, allow_pickle=False)
    lines = (source / ARTIFACT_RECORDS).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    count = manifest.get("vector_count")
    if (not isinstance(count, int) or manifest.get("chunk_count") != count
            or not isinstance(manifest.get("source_chunks_path"), str)
            or not isinstance(manifest.get("source_chunks_sha256"), str)
            or not isinstance(manifest.get("completed_batches"), int)
            or vectors.shape != (count, DEFAULT_DIMENSION) or len(records) != count):
        raise ValueError("embedding artifact vector/metadata count or dimension mismatch")
    chunks: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("position") != position or not isinstance(record.get("chunk"), Mapping):
            raise ValueError("embedding artifact metadata position mismatch")
        chunks.append(dict(record["chunk"]))
    if {chunk.get("doc_id") for chunk in chunks} != {manifest.get("doc_id")}:
        raise ValueError("embedding artifact doc_id does not match metadata")
    if len({chunk.get("chunk_id") for chunk in chunks}) != len(chunks):
        raise ValueError("embedding artifact has duplicate chunk_id values")
    if not np.isfinite(vectors).all():
        raise ValueError("embedding artifact contains NaN or infinity")
    return manifest, vectors, chunks


def build_multi_document_index(
    base_index_dir: str | Path,
    artifact_dirs: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Offline append of cached document vectors to the frozen base index."""

    destination = Path(output_dir)
    existing = [destination / name for name in (INDEX_FILE_NAME, METADATA_FILE_NAME, MANIFEST_FILE_NAME) if (destination / name).exists()]
    if existing:
        raise FileExistsError(f"output index already exists: {destination}; refusing overwrite")
    if not artifact_dirs:
        raise ValueError("at least one cached embedding artifact is required")
    base = FaissVectorStore.load(base_index_dir)
    if base.embedding_model != DEFAULT_MODEL or base.dimension != DEFAULT_DIMENSION:
        raise ValueError("base index does not use the frozen embedding configuration")
    base_doc_ids = {chunk["doc_id"] for chunk in base._chunks}
    artifacts = [load_embedding_artifact(path) for path in artifact_dirs]
    artifact_doc_ids = [manifest["doc_id"] for manifest, _, _ in artifacts]
    if len(set(artifact_doc_ids)) != len(artifact_doc_ids) or base_doc_ids.intersection(artifact_doc_ids):
        raise ValueError("duplicate doc_id append is forbidden")
    store = FaissVectorStore(dimension=base.dimension, embedding_model=base.embedding_model)
    store.add(base._index.reconstruct_n(0, base.vector_count), base._chunks)
    source_files: dict[str, str] = {}
    for doc_id in base_doc_ids:
        source_files[doc_id] = next(chunk["file_name"] for chunk in base._chunks if chunk["doc_id"] == doc_id)
    for manifest, vectors, chunks in artifacts:
        store.add(vectors, chunks)
        source_files[manifest["doc_id"]] = manifest["source_file"]
    doc_ids = sorted(source_files)
    stats: dict[str, Any] = {
        "document_count": len(doc_ids), "doc_ids": doc_ids, "source_files": source_files,
        "base_vector_count": base.vector_count, "cached_artifact_vector_count": sum(len(vectors) for _, vectors, _ in artifacts),
    }
    store.save(destination, ingestion_stats=stats)
    loaded = FaissVectorStore.load(destination)
    if loaded.vector_count != store.vector_count:
        raise RuntimeError("published multi-document index count mismatch")
    return {**stats, "vector_count": loaded.vector_count, "dimension": loaded.dimension, "embedding_model": loaded.embedding_model, "output_dir": str(destination.resolve())}
