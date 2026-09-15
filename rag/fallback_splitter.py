"""Deterministic retrieval-only splitting for pathological embedding inputs."""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Protocol, Sequence


DEFAULT_FALLBACK_MAX_TOKENS = 768
MIN_FALLBACK_TOKENS = 64
FALLBACK_TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


def build_fallback_tokenizer() -> Tokenizer:
    """Load the same local tokenizer proxy used by document processing."""
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    return HuggingFaceTokenizer.from_pretrained(
        FALLBACK_TOKENIZER_NAME,
        max_tokens=DEFAULT_FALLBACK_MAX_TOKENS,
        local_files_only=True,
    )


def _count(tokenizer: Tokenizer, text: str) -> int:
    return int(tokenizer.count_tokens(text))


def _base_tokenizer(tokenizer: Tokenizer) -> Any:
    getter = getattr(tokenizer, "get_tokenizer", None)
    return getter() if callable(getter) else tokenizer


def _encode(tokenizer: Tokenizer, text: str) -> list[Any]:
    encoder = getattr(_base_tokenizer(tokenizer), "encode", None)
    if not callable(encoder):
        raise TypeError("fallback tokenizer must provide encode/decode for hard splitting")
    try:
        return list(encoder(text, add_special_tokens=False))
    except TypeError:
        return list(encoder(text))


def _decode(tokenizer: Tokenizer, tokens: Sequence[Any]) -> str:
    decoder = getattr(_base_tokenizer(tokenizer), "decode", None)
    if not callable(decoder):
        raise TypeError("fallback tokenizer must provide encode/decode for hard splitting")
    try:
        return str(decoder(list(tokens), skip_special_tokens=True))
    except TypeError:
        return str(decoder(list(tokens)))


def _split_keep_delimiter(text: str, pattern: str) -> list[str]:
    return [piece.strip() for piece in re.findall(pattern, text, flags=re.S) if piece.strip()]


def _paragraphs(text: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"(?:\r?\n){2,}", text) if piece.strip()]


def _sentences(text: str) -> list[str]:
    pieces = _split_keep_delimiter(text, r".*?(?:[。！？；;!?]+|\r?\n|$)")
    return pieces or ([text.strip()] if text.strip() else [])


def _table_units(text: str) -> tuple[str | None, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return None, _sentences(text)

    header: str | None = None
    first = lines[0]
    for delimiter in ("|", "\t"):
        if delimiter in first and sum(delimiter in line for line in lines[1:]) >= 1:
            header = first
            break
    return header, lines[1:] if header else lines


def _hard_split(text: str, tokenizer: Tokenizer, budget: int) -> list[str]:
    tokens = _encode(tokenizer, text)
    parts = []
    for start in range(0, len(tokens), budget):
        part = _decode(tokenizer, tokens[start : start + budget]).strip()
        if part:
            parts.append(part)
    return parts


def _fit_unit(text: str, tokenizer: Tokenizer, budget: int) -> list[str]:
    if _count(tokenizer, text) <= budget:
        return [text]
    sentences = _sentences(text)
    if len(sentences) > 1:
        fitted: list[str] = []
        for sentence in sentences:
            fitted.extend(_fit_unit(sentence, tokenizer, budget))
        return fitted
    return _hard_split(text, tokenizer, budget)


def _pack(
    units: Sequence[str],
    tokenizer: Tokenizer,
    budget: int,
    *,
    header: str | None = None,
) -> list[str]:
    fitted: list[str] = []
    unit_budget = budget
    if header:
        unit_budget -= _count(tokenizer, header + "\n")
        if unit_budget < MIN_FALLBACK_TOKENS:
            header = None
            unit_budget = budget
    for unit in units:
        fitted.extend(_fit_unit(unit, tokenizer, unit_budget))

    parts: list[str] = []
    current: list[str] = []
    for unit in fitted:
        candidate_units = [*current, unit]
        body = "\n".join(candidate_units)
        candidate = f"{header}\n{body}" if header else body
        if current and _count(tokenizer, candidate) > budget:
            body = "\n".join(current)
            parts.append(f"{header}\n{body}" if header else body)
            current = [unit]
        else:
            current = candidate_units
    if current:
        body = "\n".join(current)
        parts.append(f"{header}\n{body}" if header else body)

    if len(parts) > 1 and _count(tokenizer, parts[-1]) < MIN_FALLBACK_TOKENS:
        merged = f"{parts[-2]}\n{parts[-1]}"
        if _count(tokenizer, merged) <= budget:
            parts[-2:] = [merged]
    return parts


def _embedding_context(chunk: Mapping[str, Any]) -> tuple[str, str]:
    text = str(chunk["text"])
    embedding_text = str(chunk["embedding_text"])
    offset = embedding_text.find(text)
    if offset >= 0:
        return embedding_text[:offset], embedding_text[offset + len(text) :]
    headings = chunk.get("headings")
    prefix = "\n".join(value for value in headings or [] if isinstance(value, str))
    return (f"{prefix}\n" if prefix else ""), ""


def split_chunk(
    chunk: Mapping[str, Any],
    tokenizer: Tokenizer,
    max_tokens: int = DEFAULT_FALLBACK_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Split one unified chunk into deterministic retrieval children."""
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if max_tokens < MIN_FALLBACK_TOKENS:
        raise ValueError(f"max_tokens must be at least {MIN_FALLBACK_TOKENS}")
    for field in ("chunk_id", "text", "embedding_text"):
        if not isinstance(chunk.get(field), str) or not str(chunk[field]).strip():
            raise ValueError(f"chunk.{field} must be a non-empty string")

    prefix, suffix = _embedding_context(chunk)
    context_tokens = _count(tokenizer, prefix) + _count(tokenizer, suffix)
    content_budget = max_tokens - context_tokens
    if content_budget < MIN_FALLBACK_TOKENS:
        prefix, suffix = "", ""
        content_budget = max_tokens

    text = str(chunk["text"])
    if chunk.get("chunk_type") == "table":
        header, units = _table_units(text)
        parts = _pack(units, tokenizer, content_budget, header=header)
    else:
        paragraphs = _paragraphs(text)
        units = paragraphs if len(paragraphs) > 1 else _sentences(text)
        parts = _pack(units, tokenizer, content_budget)

    root_id = str(chunk.get("parent_chunk_id") or chunk["chunk_id"])
    children: list[dict[str, Any]] = []
    for part_index, part in enumerate(parts, start=1):
        child = copy.deepcopy(dict(chunk))
        child["chunk_id"] = f"{chunk['chunk_id']}__part_{part_index:03d}"
        child["parent_chunk_id"] = root_id
        child["text"] = part
        child["embedding_text"] = f"{prefix}{part}{suffix}"
        child["fallback_split"] = True
        child["part_index"] = part_index
        child["part_count"] = len(parts)
        children.append(child)
    return children


def finalize_children(
    parent: Mapping[str, Any], leaves: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Give recursively split leaves one stable, flat parent-relative identity."""
    root_id = str(parent.get("parent_chunk_id") or parent["chunk_id"])
    finalized: list[dict[str, Any]] = []
    for part_index, leaf in enumerate(leaves, start=1):
        child = copy.deepcopy(dict(leaf))
        child["chunk_id"] = f"{root_id}__part_{part_index:03d}"
        child["parent_chunk_id"] = root_id
        child["fallback_split"] = True
        child["part_index"] = part_index
        child["part_count"] = len(leaves)
        finalized.append(child)
    return finalized
