"""Catalog-to-index consistency checks and business-to-physical scope resolution."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .document_catalog import DocumentCatalog


logger = logging.getLogger(__name__)
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[1] / "outputs" / "index" / "multi_doc_v2"


@dataclass(frozen=True)
class CatalogIndexConsistencyReport:
    index_dir: str
    vector_count: int
    metadata_count: int
    catalog_to_index_doc_ids: Mapping[str, tuple[str, ...]]
    index_doc_id_to_source_file: Mapping[str, str]


def validate_catalog_index_consistency(
    catalog: DocumentCatalog,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> CatalogIndexConsistencyReport:
    """Validate every catalog physical ID and source file against persisted index data."""

    source = Path(index_dir)
    manifest_path = source / "manifest.json"
    metadata_path = source / "metadata.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"index manifest does not exist: {manifest_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"index metadata does not exist: {metadata_path}")

    with manifest_path.open(encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if not isinstance(manifest, dict):
        raise ValueError("index manifest must be a JSON object")
    manifest_doc_ids = manifest.get("doc_ids")
    manifest_source_files = manifest.get("source_files")
    if not isinstance(manifest_doc_ids, list) or any(not isinstance(item, str) for item in manifest_doc_ids):
        raise ValueError("index manifest doc_ids must be a list of strings")
    if not isinstance(manifest_source_files, dict):
        raise ValueError("index manifest source_files must be an object")

    metadata_files: dict[str, set[str]] = {}
    metadata_count = 0
    with metadata_path.open(encoding="utf-8") as input_file:
        for expected_position, line in enumerate(input_file):
            line_number = expected_position + 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid index metadata JSON at line {line_number}") from error
            if not isinstance(record, dict) or record.get("position") != expected_position:
                raise ValueError(f"index metadata position mismatch at line {line_number}")
            chunk = record.get("chunk")
            if not isinstance(chunk, dict):
                raise ValueError(f"index metadata chunk must be an object at line {line_number}")
            index_doc_id = chunk.get("doc_id")
            source_file = chunk.get("file_name")
            if not isinstance(index_doc_id, str) or not index_doc_id:
                raise ValueError(f"index metadata doc_id is invalid at line {line_number}")
            if not isinstance(source_file, str) or not source_file:
                raise ValueError(f"index metadata file_name is invalid at line {line_number}")
            metadata_files.setdefault(index_doc_id, set()).add(source_file)
            metadata_count += 1

    vector_count = manifest.get("vector_count")
    if not isinstance(vector_count, int) or isinstance(vector_count, bool):
        raise ValueError("index manifest vector_count must be an integer")
    if vector_count != metadata_count:
        raise ValueError(
            f"index vector/metadata count mismatch: manifest={vector_count}, metadata={metadata_count}"
        )

    manifest_doc_id_set = set(manifest_doc_ids)
    catalog_mapping: dict[str, tuple[str, ...]] = {}
    physical_mapping: dict[str, str] = {}
    for document in catalog.records:
        catalog_mapping[document.catalog_doc_id] = document.index_doc_ids
        for index_doc_id in document.index_doc_ids:
            if index_doc_id not in manifest_doc_id_set:
                raise ValueError(
                    f"catalog {document.catalog_doc_id} index_doc_id is absent from manifest: {index_doc_id}"
                )
            if index_doc_id not in metadata_files:
                raise ValueError(
                    f"catalog {document.catalog_doc_id} index_doc_id is absent from metadata: {index_doc_id}"
                )
            files = metadata_files[index_doc_id]
            if files != {document.source_file}:
                raise ValueError(
                    f"catalog {document.catalog_doc_id} source_file mismatch for {index_doc_id}: "
                    f"catalog={document.source_file!r}, metadata={sorted(files)!r}"
                )
            manifest_file = manifest_source_files.get(index_doc_id)
            if manifest_file != document.source_file:
                raise ValueError(
                    f"catalog {document.catalog_doc_id} source_file mismatch for {index_doc_id}: "
                    f"catalog={document.source_file!r}, manifest={manifest_file!r}"
                )
            physical_mapping[index_doc_id] = document.source_file

    report = CatalogIndexConsistencyReport(
        index_dir=str(source),
        vector_count=vector_count,
        metadata_count=metadata_count,
        catalog_to_index_doc_ids=catalog_mapping,
        index_doc_id_to_source_file=physical_mapping,
    )
    logger.debug(
        "CATALOG_INDEX_CONSISTENCY: %s",
        json.dumps(
            {
                "index_dir": report.index_dir,
                "vector_count": report.vector_count,
                "metadata_count": report.metadata_count,
                "catalog_to_index_doc_ids": report.catalog_to_index_doc_ids,
            },
            ensure_ascii=False,
            default=list,
        ),
    )
    return report


class SourceScopeResolver:
    """Translate business catalog IDs to physical IDs at the execution boundary."""

    def __init__(self, catalog: DocumentCatalog) -> None:
        self.catalog = catalog

    def resolve(self, catalog_doc_ids: Sequence[str]) -> tuple[str, ...]:
        if not isinstance(catalog_doc_ids, Sequence) or isinstance(catalog_doc_ids, (str, bytes)):
            raise TypeError("catalog_doc_ids must be a sequence of strings")
        requested = tuple(catalog_doc_ids)
        if not requested:
            raise ValueError("catalog_doc_ids must not be empty")
        if any(not isinstance(item, str) or not item.strip() for item in requested):
            raise ValueError("catalog_doc_ids must contain non-empty strings")
        if len(set(requested)) != len(requested):
            raise ValueError("catalog_doc_ids must not contain duplicates")

        allowed_index_doc_ids = self.catalog.index_doc_ids_for(requested)
        logger.debug(
            "SOURCE_SCOPE_RESOLUTION: %s",
            json.dumps(
                {
                    "catalog_doc_ids": list(requested),
                    "allowed_index_doc_ids": list(allowed_index_doc_ids),
                },
                ensure_ascii=False,
            ),
        )
        return allowed_index_doc_ids

    def resolve_validated(
        self,
        catalog_doc_ids: Sequence[str],
        index_dir: str | Path = DEFAULT_INDEX_DIR,
    ) -> tuple[str, ...]:
        validate_catalog_index_consistency(self.catalog, index_dir)
        return self.resolve(catalog_doc_ids)
