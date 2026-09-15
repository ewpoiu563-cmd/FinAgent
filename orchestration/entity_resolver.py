"""Deterministic first-version company and document entity resolution."""

from __future__ import annotations

import unicodedata

from .document_catalog import DocumentCatalog, DocumentRecord
from .models import EntityResolution


def normalize_entity_text(value: str) -> str:
    """Normalize width/case and remove separators without fuzzy matching."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z", "C"))
    )


class EntityResolver:
    """Resolve an exact normalized alias using deterministic longest-match wins."""

    def __init__(self, catalog: DocumentCatalog) -> None:
        self.catalog = catalog
        aliases: list[tuple[str, str, DocumentRecord]] = []
        owners: dict[str, str] = {}
        for record in catalog.records:
            for alias in (*record.aliases, record.company_name):
                normalized = normalize_entity_text(alias)
                owner = owners.get(normalized)
                if owner is not None and owner != record.catalog_doc_id:
                    raise ValueError(
                        f"normalized alias {alias!r} belongs to both {owner} and {record.catalog_doc_id}"
                    )
                owners[normalized] = record.catalog_doc_id
                aliases.append((normalized, alias, record))
        self._aliases = tuple(aliases)

    def resolve(self, question: str) -> EntityResolution:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        normalized_question = normalize_entity_text(question)
        matches = [
            candidate
            for candidate in self._aliases
            if candidate[0] and candidate[0] in normalized_question
        ]
        if not matches:
            return EntityResolution(document_type=_document_type_from_question(normalized_question))

        # Stable catalog order breaks same-document alias ties; different-document
        # ties are rejected instead of silently choosing the wrong company.
        longest = max(len(candidate[0]) for candidate in matches)
        best = [candidate for candidate in matches if len(candidate[0]) == longest]
        owners = {candidate[2].catalog_doc_id for candidate in best}
        if len(owners) != 1:
            return EntityResolution(document_type=_document_type_from_question(normalized_question))

        _, matched_alias, record = best[0]
        return EntityResolution(
            company_name=record.company_name,
            matched_alias=matched_alias,
            catalog_doc_id=record.catalog_doc_id,
            document_type=record.document_type,
        )


def _document_type_from_question(normalized_question: str) -> str | None:
    if "招股说明书" in normalized_question or "招股书" in normalized_question:
        return "prospectus"
    if "年报" in normalized_question or "年度报告" in normalized_question:
        return "annual_report"
    if "财报" in normalized_question or "财务报告" in normalized_question:
        return "financial_report"
    return None
