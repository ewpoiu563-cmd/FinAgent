"""Deterministic deictic-time resolution for requirement planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date


_ANCHORED_PREVIOUS_YEAR = re.compile(
    r"(?P<year>(?:19|20)\d{2})年(?:的)?(?P<relative>上一年|前一年)"
)
_AMBIGUOUS_PREVIOUS_YEAR = re.compile(r"上一年|前一年")


@dataclass(frozen=True)
class TemporalReplacement:
    expression: str
    resolved_value: str
    anchor: str

    def to_dict(self) -> dict[str, str]:
        return {
            "expression": self.expression,
            "resolved_value": self.resolved_value,
            "anchor": self.anchor,
        }


@dataclass(frozen=True)
class TemporalResolution:
    original_text: str
    resolved_text: str
    reference_date: date
    replacements: tuple[TemporalReplacement, ...] = ()
    clarification_needed: bool = False
    clarification_question: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "resolved_text": self.resolved_text,
            "reference_date": self.reference_date.isoformat(),
            "replacements": [replacement.to_dict() for replacement in self.replacements],
            "clarification_needed": self.clarification_needed,
            "clarification_question": self.clarification_question,
        }


class TemporalResolver:
    """Resolve relative years without asking an LLM to infer an anchor."""

    def __init__(self, reference_date: date | None = None) -> None:
        self.reference_date = reference_date or date.today()

    def resolve(
        self,
        text: str,
        *,
        reference_date: date | None = None,
    ) -> TemporalResolution:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("temporal input must be a non-empty string")
        anchor_date = reference_date or self.reference_date
        resolved = text.strip()
        replacements: list[TemporalReplacement] = []

        def replace_anchored(match: re.Match[str]) -> str:
            year = int(match.group("year")) - 1
            value = f"{year}年"
            replacements.append(
                TemporalReplacement(match.group(0), value, f"{match.group('year')}年")
            )
            return value

        resolved = _ANCHORED_PREVIOUS_YEAR.sub(replace_anchored, resolved)
        if _AMBIGUOUS_PREVIOUS_YEAR.search(resolved):
            return TemporalResolution(
                original_text=text,
                resolved_text=resolved,
                reference_date=anchor_date,
                replacements=tuple(replacements),
                clarification_needed=True,
                clarification_question="“上一年/前一年”缺少明确年份锚点，请说明它相对于哪一年。",
            )

        for expression, offset in (("前年", -2), ("去年", -1), ("今年", 0)):
            if expression not in resolved:
                continue
            value = f"{anchor_date.year + offset}年"
            resolved = resolved.replace(expression, value)
            replacements.append(
                TemporalReplacement(expression, value, anchor_date.isoformat())
            )

        return TemporalResolution(
            original_text=text,
            resolved_text=resolved,
            reference_date=anchor_date,
            replacements=tuple(replacements),
        )
