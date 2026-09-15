"""Build a local FAISS index from FinAgent unified chunk JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypedDict

import numpy as np
import requests

from rag.embedding import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TIMEOUT,
    EmbeddingAPIError,
    EmbeddingClient,
)
from rag.fallback_splitter import (
    DEFAULT_FALLBACK_MAX_TOKENS,
    build_fallback_tokenizer,
    finalize_children,
    split_chunk,
)
from rag.vector_store import INDEX_FILE_NAME, MANIFEST_FILE_NAME, METADATA_FILE_NAME, FaissVectorStore

REQUIRED_INPUT_FIELDS = ("chunk_id", "embedding_text", "text")
OUTPUT_FILE_NAMES = (INDEX_FILE_NAME, METADATA_FILE_NAME, MANIFEST_FILE_NAME)
CHECKPOINT_DIR_NAME = ".build_tmp"
CHECKPOINT_FILE_NAME = "progress.json"
CHECKPOINT_SCHEMA_VERSION = 2
MAX_BATCH_SIZE = 20
PROGRESS_EVERY_BATCHES = 10
DEFAULT_BUILD_MAX_RETRIES = 2
MAX_FALLBACK_DEPTH = 2


class IndexBuildStats(TypedDict):
    input_chunks: int
    source_chunks: int
    indexed_chunks: int
    skipped_toc_chunks: int
    skipped_other_chunks: int
    embedded_chunks: int
    vector_count: int
    dimension: int
    embedding_model: str
    elapsed_time: float
    output_dir: str


class BatchEmbeddingError(RuntimeError):
    """A terminal batch failure with enough context for a safe resume."""

    def __init__(self, cause: Exception, *, next_input_position: int,
                 embedded_chunks: int, chunk_count: int,
                 failed_batch_start: int, failed_batch_end: int) -> None:
        self.next_input_position = next_input_position
        self.embedded_chunks = embedded_chunks
        self.remaining_chunks = chunk_count - next_input_position
        self.failed_batch_start = failed_batch_start
        self.failed_batch_end = failed_batch_end
        super().__init__(
            f"embedding batch failed: {cause}\n"
            f"next_input_position={self.next_input_position}\n"
            f"embedded_chunks={self.embedded_chunks}\n"
            f"remaining_chunks={self.remaining_chunks}\n"
            f"failed_batch_start={self.failed_batch_start}\n"
            f"failed_batch_end={self.failed_batch_end}"
        )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _cli_batch_size(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > MAX_BATCH_SIZE:
        raise argparse.ArgumentTypeError(
            f"must not exceed the synchronous API limit of {MAX_BATCH_SIZE}"
        )
    return parsed


def _validate_limit(limit: int | None) -> None:
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit must be a positive integer or None")


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must not exceed the synchronous API limit of {MAX_BATCH_SIZE}")


def _validate_request_settings(timeout: float, max_retries: int) -> None:
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("timeout must be positive")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")


def _validate_chunk(value: Any, *, line_number: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"line {line_number}: chunk must be a JSON object")
    missing = [field for field in REQUIRED_INPUT_FIELDS if field not in value]
    if missing:
        raise ValueError(f"line {line_number}: missing required fields: {missing}")
    chunk = dict(value)
    for field in REQUIRED_INPUT_FIELDS:
        if not isinstance(chunk[field], str) or not chunk[field].strip():
            raise ValueError(f"line {line_number}: {field} must be a non-empty string")
    return chunk


def read_chunks(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read and validate unified chunks, preserving JSONL order."""
    _validate_limit(limit)
    chunks: list[dict[str, Any]] = []
    chunk_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank lines are not allowed")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
            chunk = _validate_chunk(value, line_number=line_number)
            chunk_id = chunk["chunk_id"]
            if chunk_id in chunk_ids:
                raise ValueError(f"line {line_number}: duplicate chunk_id: {chunk_id}")
            chunk_ids.add(chunk_id)
            chunks.append(chunk)
            if limit is not None and len(chunks) >= limit:
                break
    if not chunks:
        raise ValueError("input JSONL contains no chunks")
    return chunks


def is_toc_chunk(chunk: Mapping[str, Any]) -> bool:
    """Return whether any heading is exactly ``目录`` after conservative normalization."""
    headings = chunk.get("headings")
    if not isinstance(headings, list):
        return False
    return any(
        isinstance(heading, str) and heading.strip().replace(" ", "") == "目录"
        for heading in headings
    )


def _existing_output_files(output_dir: Path) -> list[Path]:
    return [output_dir / name for name in OUTPUT_FILE_NAMES if (output_dir / name).exists()]


def _input_identity(path: Path, *, chunk_count: int, limit: int | None) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return {"input_path": str(path.resolve()), "input_sha256": digest.hexdigest(),
            "input_size": path.stat().st_size, "chunk_count": chunk_count, "limit": limit}


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_batch(checkpoint_dir: Path, *, vector_start: int,
                        vectors: Sequence[Sequence[float]],
                        chunks: Sequence[Mapping[str, Any]],
                        input_positions: Sequence[int]) -> dict[str, Any]:
    vector_end = vector_start + len(chunks)
    stem = f"batch_{vector_start:09d}_{vector_end:09d}"
    vectors_name, records_name = f"{stem}.npy", f"{stem}.jsonl"
    vectors_path, records_path = checkpoint_dir / vectors_name, checkpoint_dir / records_name
    temporary_vectors = vectors_path.with_name(f".{vectors_name}.{os.getpid()}.tmp")
    temporary_records = records_path.with_name(f".{records_name}.{os.getpid()}.tmp")
    try:
        with temporary_vectors.open("wb") as output:
            np.save(output, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        with temporary_records.open("w", encoding="utf-8", newline="\n") as output:
            for offset, (chunk, input_position) in enumerate(zip(chunks, input_positions)):
                output.write(json.dumps(
                    {"vector_position": vector_start + offset,
                     "input_position": input_position, "chunk_id": chunk["chunk_id"],
                     "chunk": chunk},
                    ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_vectors.replace(vectors_path)
        temporary_records.replace(records_path)
    finally:
        temporary_vectors.unlink(missing_ok=True)
        temporary_records.unlink(missing_ok=True)
    return {"vector_start": vector_start, "vector_end": vector_end,
            "input_positions": list(input_positions),
            "vectors_file": vectors_name, "records_file": records_name}


def _new_progress(identity: Mapping[str, Any], *, model: str, dimension: int) -> dict[str, Any]:
    return {"schema_version": CHECKPOINT_SCHEMA_VERSION, **identity,
            "embedding_model": model, "dimension": dimension,
            "next_input_position": 0, "embedded_chunks": 0,
            "skipped_toc_chunks": 0, "batches": []}


def _load_progress(checkpoint_dir: Path) -> dict[str, Any]:
    try:
        with (checkpoint_dir / CHECKPOINT_FILE_NAME).open(encoding="utf-8") as input_file:
            progress = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint progress file: {error}") from error
    if not isinstance(progress, dict):
        raise ValueError("invalid checkpoint: progress must be a JSON object")
    schema_version = progress.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported checkpoint schema_version: "
            f"expected {CHECKPOINT_SCHEMA_VERSION}, got {schema_version!r}; "
            "restart this index build with --overwrite"
        )
    return progress


def _validate_resume(progress: Mapping[str, Any], identity: Mapping[str, Any], *,
                     model: str, dimension: int, chunks: Sequence[Mapping[str, Any]],
                     checkpoint_dir: Path) -> None:
    expected = {"schema_version": CHECKPOINT_SCHEMA_VERSION, **identity,
                "embedding_model": model, "dimension": dimension}
    for field, expected_value in expected.items():
        if progress.get(field) != expected_value:
            raise ValueError(
                f"checkpoint {field} mismatch: expected {expected_value!r}, got {progress.get(field)!r}"
            )
    next_position = progress.get("next_input_position")
    embedded = progress.get("embedded_chunks")
    skipped_toc = progress.get("skipped_toc_chunks")
    batches = progress.get("batches")
    counters = (next_position, embedded, skipped_toc)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in counters):
        raise ValueError("invalid checkpoint counters")
    if (not 0 <= next_position <= len(chunks) or embedded < 0 or skipped_toc < 0
            or not isinstance(batches, list)):
        raise ValueError("invalid checkpoint progress range")
    expected_vector_start = 0
    seen_input_positions: list[int] = []
    for batch_number, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise ValueError(f"invalid checkpoint batch {batch_number}")
        start, end = batch.get("vector_start"), batch.get("vector_end")
        input_positions = batch.get("input_positions")
        if (start != expected_vector_start or not isinstance(end, int)
                or isinstance(end, bool) or end <= start
                or not isinstance(input_positions, list) or len(input_positions) != end - start):
            raise ValueError(f"invalid checkpoint batch range at batch {batch_number}")
        if end > embedded:
            raise ValueError(f"checkpoint batch {batch_number} exceeds completed progress")
        vector_file = checkpoint_dir / str(batch.get("vectors_file", ""))
        records_file = checkpoint_dir / str(batch.get("records_file", ""))
        try:
            vectors = np.load(vector_file, allow_pickle=False)
            records = [json.loads(line) for line in records_file.read_text(encoding="utf-8").splitlines()]
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid checkpoint batch {batch_number}: {error}") from error
        if vectors.shape != (end - start, dimension):
            raise ValueError(f"checkpoint vector shape mismatch at batch {batch_number}")
        if any(not isinstance(position, int) or isinstance(position, bool)
               or not 0 <= position < next_position or is_toc_chunk(chunks[position])
               for position in input_positions):
            raise ValueError(f"invalid checkpoint input positions at batch {batch_number}")
        if len(records) != end - start:
            raise ValueError(f"checkpoint record count mismatch at batch {batch_number}")
        for offset, (record, position) in enumerate(zip(records, input_positions)):
            if not isinstance(record, Mapping) or (
                record.get("vector_position") != start + offset
                or record.get("input_position") != position
            ):
                raise ValueError(f"checkpoint chunk mapping mismatch at batch {batch_number}")
            checkpoint_chunk = record.get("chunk")
            if checkpoint_chunk is None:
                expected_id = chunks[position]["chunk_id"]
            elif isinstance(checkpoint_chunk, Mapping):
                expected_id = checkpoint_chunk.get("chunk_id")
            else:
                raise ValueError(f"invalid checkpoint chunk at batch {batch_number}")
            if record.get("chunk_id") != expected_id:
                raise ValueError(f"checkpoint chunk mapping mismatch at batch {batch_number}")
        seen_input_positions.extend(input_positions)
        expected_vector_start = end
    if expected_vector_start != embedded or seen_input_positions != sorted(seen_input_positions):
        raise ValueError("checkpoint batches do not cover embedded_chunks")
    expected_eligible = [position for position in range(next_position)
                         if not is_toc_chunk(chunks[position])]
    if sorted(set(seen_input_positions)) != expected_eligible:
        raise ValueError("checkpoint input positions do not cover processed chunks")
    if next_position - len(expected_eligible) != skipped_toc:
        raise ValueError("checkpoint skipped_toc_chunks mismatch")


def _publish_store(store: FaissVectorStore, output_dir: Path,
                   ingestion_stats: Mapping[str, int]) -> None:
    """Stage complete files and atomically replace each one, manifest last."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-building-", dir=output_dir.parent) as temporary_directory:
        staging_dir = Path(temporary_directory)
        store.save(staging_dir, ingestion_stats=ingestion_stats)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (INDEX_FILE_NAME, METADATA_FILE_NAME, MANIFEST_FILE_NAME):
            (staging_dir / name).replace(output_dir / name)


def _build_store_from_checkpoint(progress: Mapping[str, Any], checkpoint_dir: Path,
                                 chunks: Sequence[Mapping[str, Any]], *, dimension: int,
                                 model: str) -> FaissVectorStore:
    store = FaissVectorStore(dimension=dimension, embedding_model=model)
    for batch in progress["batches"]:
        vectors = np.load(checkpoint_dir / batch["vectors_file"], allow_pickle=False)
        records_path = checkpoint_dir / batch["records_file"]
        records = [json.loads(line) for line in records_path.read_text(
            encoding="utf-8").splitlines()]
        indexed_chunks = [
            record.get("chunk") or chunks[record["input_position"]]
            for record in records
        ]
        store.add(vectors, indexed_chunks)
    return store


def _is_fallback_error(error: Exception) -> bool:
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(error, EmbeddingAPIError):
        status_code = error.status_code
        return status_code == 429 or (
            isinstance(status_code, int) and status_code >= 500
        )
    return False


def _fallback_failure(chunk: Mapping[str, Any], depth: int, cause: Exception) -> RuntimeError:
    return RuntimeError(
        "pathological chunk embedding failed after fallback: "
        f"chunk_id={chunk['chunk_id']}, fallback_depth={depth}, "
        f"max_fallback_depth={MAX_FALLBACK_DEPTH}, cause={cause}"
    )


def _embed_with_fallback(
    client: Any,
    chunks: Sequence[Mapping[str, Any]],
    *,
    tokenizer_provider: Callable[[], Any],
    fallback_timing: dict[str, float] | None = None,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Embed a batch, narrowing transient failures before content splitting."""
    try:
        vectors = client.embed_texts(
            [str(chunk["embedding_text"]) for chunk in chunks],
            text_type="document",
        )
        return vectors, [dict(chunk) for chunk in chunks]
    except Exception as error:
        if not _is_fallback_error(error):
            raise
        if len(chunks) > 1:
            middle = len(chunks) // 2
            left_vectors, left_chunks = _embed_with_fallback(
                client, chunks[:middle], tokenizer_provider=tokenizer_provider,
                fallback_timing=fallback_timing,
            )
            right_vectors, right_chunks = _embed_with_fallback(
                client, chunks[middle:], tokenizer_provider=tokenizer_provider,
                fallback_timing=fallback_timing,
            )
            return left_vectors + right_vectors, left_chunks + right_chunks
        return _embed_single_fallback(
            client,
            chunks[0],
            cause=error,
            depth=1,
            tokenizer_provider=tokenizer_provider,
            fallback_timing=fallback_timing,
        )


def _embed_single_fallback(
    client: Any,
    chunk: Mapping[str, Any],
    *,
    cause: Exception,
    depth: int,
    tokenizer_provider: Callable[[], Any],
    fallback_timing: dict[str, float] | None = None,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    if depth > MAX_FALLBACK_DEPTH:
        raise _fallback_failure(chunk, depth - 1, cause) from cause
    budget = DEFAULT_FALLBACK_MAX_TOKENS // (2 ** (depth - 1))
    split_started_at = time.perf_counter()
    children = split_chunk(chunk, tokenizer_provider(), max_tokens=budget)
    if fallback_timing is not None:
        fallback_timing["fallback_split_seconds"] = (
            fallback_timing.get("fallback_split_seconds", 0.0)
            + time.perf_counter() - split_started_at
        )
    if len(children) <= 1:
        if depth >= MAX_FALLBACK_DEPTH:
            raise _fallback_failure(chunk, depth, cause) from cause
        return _embed_single_fallback(
            client,
            chunk,
            cause=cause,
            depth=depth + 1,
            tokenizer_provider=tokenizer_provider,
            fallback_timing=fallback_timing,
        )

    vectors: list[list[float]] = []
    leaves: list[dict[str, Any]] = []
    for child in children:
        try:
            child_vectors = client.embed_texts(
                [child["embedding_text"]], text_type="document"
            )
            vectors.extend(child_vectors)
            leaves.append(child)
        except Exception as error:
            if not _is_fallback_error(error):
                raise
            if depth >= MAX_FALLBACK_DEPTH:
                raise _fallback_failure(child, depth, error) from error
            nested_vectors, nested_leaves = _embed_single_fallback(
                client,
                child,
                cause=error,
                depth=depth + 1,
                tokenizer_provider=tokenizer_provider,
                fallback_timing=fallback_timing,
            )
            vectors.extend(nested_vectors)
            leaves.extend(nested_leaves)
    return vectors, finalize_children(chunk, leaves)


def build_index(input_path: str | Path, output_dir: str | Path, *, limit: int | None = None,
                batch_size: int = DEFAULT_BATCH_SIZE, overwrite: bool = False,
                resume: bool = False, timeout: float = DEFAULT_TIMEOUT,
                max_retries: int = DEFAULT_BUILD_MAX_RETRIES,
                client_factory: Callable[..., Any] = EmbeddingClient,
                fallback_tokenizer_factory: Callable[[], Any] = build_fallback_tokenizer,
                ) -> IndexBuildStats:
    """Embed chunks with a durable checkpoint, then publish one complete index."""
    started_at = time.perf_counter()
    _validate_limit(limit)
    _validate_batch_size(batch_size)
    _validate_request_settings(timeout, max_retries)
    source, destination = Path(input_path), Path(output_dir)
    checkpoint_dir = destination / CHECKPOINT_DIR_NAME
    progress_path = checkpoint_dir / CHECKPOINT_FILE_NAME
    existing_files = _existing_output_files(destination)
    if existing_files and not overwrite:
        names = ", ".join(path.name for path in existing_files)
        raise FileExistsError(f"output directory already contains index files ({names}); use --overwrite")
    if progress_path.exists() and not resume and not overwrite:
        raise FileExistsError("checkpoint already exists; use --resume or --overwrite")

    chunks = read_chunks(source, limit=limit)
    identity = _input_identity(source, chunk_count=len(chunks), limit=limit)
    progress = _load_progress(checkpoint_dir) if progress_path.exists() and resume else None
    client = client_factory(
        batch_size=batch_size,
        timeout=float(timeout),
        max_retries=max_retries,
    )
    model, dimension = client.model, client.dimension

    if progress is not None:
        _validate_resume(progress, identity, model=model, dimension=dimension,
                         chunks=chunks, checkpoint_dir=checkpoint_dir)
    else:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        progress = _new_progress(identity, model=model, dimension=dimension)
        _atomic_write_json(progress_path, progress)

    next_position = progress["next_input_position"]
    embedded = progress["embedded_chunks"]
    fallback_tokenizer: Any = None

    def tokenizer_provider() -> Any:
        nonlocal fallback_tokenizer
        if fallback_tokenizer is None:
            fallback_tokenizer = fallback_tokenizer_factory()
        return fallback_tokenizer

    batch_number = len(progress["batches"])
    while next_position < len(chunks):
        while next_position < len(chunks) and is_toc_chunk(chunks[next_position]):
            next_position += 1
            progress["next_input_position"] = next_position
            progress["skipped_toc_chunks"] += 1
            _atomic_write_json(progress_path, progress)
        if next_position == len(chunks):
            break
        batch_positions: list[int] = []
        scan_position = next_position
        skipped_in_scan = 0
        while scan_position < len(chunks) and len(batch_positions) < batch_size:
            if is_toc_chunk(chunks[scan_position]):
                skipped_in_scan += 1
            else:
                batch_positions.append(scan_position)
            scan_position += 1
        batch = [chunks[position] for position in batch_positions]
        try:
            vectors, indexed_batch = _embed_with_fallback(
                client, batch, tokenizer_provider=tokenizer_provider
            )
        except Exception as error:
            raise BatchEmbeddingError(
                error, next_input_position=next_position, embedded_chunks=embedded,
                chunk_count=len(chunks), failed_batch_start=next_position,
                failed_batch_end=scan_position) from error
        if len(vectors) != len(indexed_batch):
            raise ValueError(
                f"embedding count mismatch: expected {len(indexed_batch)}, got {len(vectors)}"
            )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(indexed_batch), dimension):
            raise ValueError(
                "embedding dimension mismatch: "
                f"expected {(len(indexed_batch), dimension)}, got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("embedding contains NaN or infinity")
        position_by_root_id = {
            source_chunk["chunk_id"]: position
            for position, source_chunk in zip(batch_positions, batch)
        }
        indexed_input_positions = [
            position_by_root_id[
                indexed_chunk.get("parent_chunk_id") or indexed_chunk["chunk_id"]
            ]
            for indexed_chunk in indexed_batch
        ]
        batch_record = _atomic_write_batch(
            checkpoint_dir, vector_start=embedded, vectors=matrix,
            chunks=indexed_batch,
            input_positions=indexed_input_positions)
        progress["batches"].append(batch_record)
        embedded += len(indexed_batch)
        next_position = scan_position
        progress["embedded_chunks"] = embedded
        progress["next_input_position"] = next_position
        progress["skipped_toc_chunks"] += skipped_in_scan
        _atomic_write_json(progress_path, progress)
        batch_number += 1
        if batch_number % PROGRESS_EVERY_BATCHES == 0 or next_position == len(chunks):
            print(f"processed {next_position}/{len(chunks)}; embedded {embedded}")

    store = _build_store_from_checkpoint(progress, checkpoint_dir, chunks,
                                         dimension=dimension, model=model)
    skipped_toc = progress["skipped_toc_chunks"]
    skipped_other = len(chunks) - next_position
    if next_position != len(chunks) or store.vector_count != embedded or skipped_other != 0:
        raise RuntimeError(
            f"index build count mismatch: source={len(chunks)}, input_position={next_position}, "
            f"embedded={embedded}, skipped_toc={skipped_toc}, vectors={store.vector_count}")
    ingestion_stats = {"source_chunks": len(chunks), "indexed_chunks": embedded,
                       "skipped_toc_chunks": skipped_toc,
                       "skipped_other_chunks": skipped_other}
    _publish_store(store, destination, ingestion_stats)
    loaded = FaissVectorStore.load(destination)
    if loaded.vector_count != embedded:
        raise RuntimeError(f"published index count mismatch: embedded={embedded}, vectors={loaded.vector_count}")
    shutil.rmtree(checkpoint_dir)
    return {"input_chunks": len(chunks), **ingestion_stats, "embedded_chunks": embedded,
            "vector_count": store.vector_count, "dimension": store.dimension,
            "embedding_model": store.embedding_model,
            "elapsed_time": round(time.perf_counter() - started_at, 3),
            "output_dir": str(destination.resolve())}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed FinAgent unified chunks and build a local FAISS index.")
    parser.add_argument("--input", type=Path, required=True, help="input chunk JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="index directory")
    parser.add_argument("--limit", type=_positive_int, help="optional maximum chunk count")
    parser.add_argument("--batch-size", type=_cli_batch_size, default=DEFAULT_BATCH_SIZE,
                        help=f"embedding batch size (default: {DEFAULT_BATCH_SIZE}, maximum: {MAX_BATCH_SIZE})")
    parser.add_argument("--timeout", type=_positive_float, default=DEFAULT_TIMEOUT,
                        help=f"embedding read timeout in seconds (default: {DEFAULT_TIMEOUT:g})")
    parser.add_argument("--max-retries", type=_non_negative_int,
                        default=DEFAULT_BUILD_MAX_RETRIES,
                        help=f"retries after the initial request (default: {DEFAULT_BUILD_MAX_RETRIES})")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing index files and discard any checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="continue from a compatible checkpoint when one exists")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        stats = build_index(args.input, args.output_dir, limit=args.limit,
                            batch_size=args.batch_size, overwrite=args.overwrite,
                            resume=args.resume, timeout=args.timeout,
                            max_retries=args.max_retries)
    except Exception as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
