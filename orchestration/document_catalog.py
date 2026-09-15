"""Business document catalog, deliberately separate from index manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "catalog" / "document_catalog.json"


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class DocumentRecord:
    catalog_doc_id: str
    index_doc_ids: tuple[str, ...]
    source_file: str
    company_name: str
    aliases: tuple[str, ...]
    document_type: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], position: int) -> "DocumentRecord":
        label = f"documents[{position}]"
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object")

        index_doc_ids = value.get("index_doc_ids")
        aliases = value.get("aliases")
        if not isinstance(index_doc_ids, Sequence) or isinstance(index_doc_ids, (str, bytes)):
            raise ValueError(f"{label}.index_doc_ids must be a non-empty list of strings")
        if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
            raise ValueError(f"{label}.aliases must be a non-empty list of strings")

        normalized_index_ids = tuple(
            _non_empty_string(item, f"{label}.index_doc_ids") for item in index_doc_ids
        )
        normalized_aliases = tuple(
            _non_empty_string(item, f"{label}.aliases") for item in aliases
        )
        if not normalized_index_ids:
            raise ValueError(f"{label}.index_doc_ids must not be empty")
        if not normalized_aliases:
            raise ValueError(f"{label}.aliases must not be empty")
        if len(set(normalized_index_ids)) != len(normalized_index_ids):
            raise ValueError(f"{label}.index_doc_ids contains duplicates")
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise ValueError(f"{label}.aliases contains duplicates")

        return cls(
            catalog_doc_id=_non_empty_string(value.get("catalog_doc_id"), f"{label}.catalog_doc_id"),
            index_doc_ids=normalized_index_ids,
            source_file=_non_empty_string(value.get("source_file"), f"{label}.source_file"),
            company_name=_non_empty_string(value.get("company_name"), f"{label}.company_name"),
            aliases=normalized_aliases,
            document_type=_non_empty_string(value.get("document_type"), f"{label}.document_type"),
        )


class DocumentCatalog:
    """Validated business catalog with forward and reverse ID lookup."""

    def __init__(self, records: Sequence[DocumentRecord]) -> None:
        self._records = tuple(records)
        if not self._records:
            raise ValueError("document catalog must contain at least one document")

        self._by_catalog_id: dict[str, DocumentRecord] = {}
        self._by_index_doc_id: dict[str, DocumentRecord] = {}
        self._by_source_file: dict[str, DocumentRecord] = {}
        for record in self._records:
            if record.catalog_doc_id in self._by_catalog_id:
                raise ValueError(f"duplicate catalog_doc_id: {record.catalog_doc_id}")
            source_key = record.source_file.casefold()
            if source_key in self._by_source_file:
                raise ValueError(f"duplicate source_file: {record.source_file}")
            self._by_catalog_id[record.catalog_doc_id] = record
            self._by_source_file[source_key] = record
            for index_doc_id in record.index_doc_ids:
                if index_doc_id in self._by_index_doc_id:
                    raise ValueError(f"duplicate index_doc_id: {index_doc_id}")
                self._by_index_doc_id[index_doc_id] = record

    @property
    def records(self) -> tuple[DocumentRecord, ...]:
        return self._records

    def get(self, catalog_doc_id: str) -> DocumentRecord:
        try:
            return self._by_catalog_id[catalog_doc_id]
        except KeyError as error:
            raise KeyError(f"unknown catalog_doc_id: {catalog_doc_id}") from error

    def get_by_index_doc_id(self, index_doc_id: str) -> DocumentRecord | None:
        return self._by_index_doc_id.get(index_doc_id)

    def get_by_source_file(self, source_file: str) -> DocumentRecord | None:
        return self._by_source_file.get(source_file.casefold())

    def index_doc_ids_for(self, catalog_doc_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            index_doc_id
            for catalog_doc_id in catalog_doc_ids
            for index_doc_id in self.get(catalog_doc_id).index_doc_ids
        )

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DocumentCatalog":
        if not isinstance(data, Mapping):
            raise ValueError("document catalog root must be an object")
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported document catalog schema_version: {data.get('schema_version')!r}")
        documents = data.get("documents")
        if not isinstance(documents, list):
            raise ValueError("document catalog documents must be a list")
        return cls([DocumentRecord.from_mapping(item, position) for position, item in enumerate(documents)])

    @classmethod
    def load(cls, path: str | Path) -> "DocumentCatalog":
        source = Path(path)
        with source.open(encoding="utf-8") as input_file:
            data = json.load(input_file)
        return cls.from_data(data)


def load_default_catalog() -> DocumentCatalog:
    return DocumentCatalog.load(DEFAULT_CATALOG_PATH)
