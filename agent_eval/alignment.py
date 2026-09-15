"""Safe Gold-to-runtime requirement alignment without positional guessing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Protocol

from .models import Alignment, EvaluationRunArtifact


class RequirementAligner(Protocol):
    def align(self, gold_case: Mapping[str, Any], run: EvaluationRunArtifact) -> tuple[Alignment, ...]: ...


class DeterministicRequirementAligner:
    def align(self, gold_case: Mapping[str, Any], run: EvaluationRunArtifact) -> tuple[Alignment, ...]:
        results: list[Alignment] = []
        requirements = gold_case["requirements"]
        candidates = [task for task in run.planned_requirements if task.planned_source is not None]
        for requirement in gold_case["requirements"]:
            rid = requirement["requirement_id"]
            explicit = [task for task in candidates if rid in task.stable_origin_requirement_ids]
            if explicit:
                results.append(Alignment(rid, "mapped_explicit", tuple(task.runtime_task_id for task in explicit)))
                continue
            target = _normalize(requirement["question"])
            matches = [task for task in candidates if _normalize(task.atomic_question) == target]
            if len(matches) == 1:
                results.append(Alignment(rid, "mapped_semantic_exact", (matches[0].runtime_task_id,)))
            elif len(matches) > 1:
                results.append(Alignment(rid, "alignment_ambiguous", reason="multiple unique-scope candidates"))
            elif len(requirements) == 1 and len(candidates) == 1:
                results.append(Alignment(
                    rid,
                    "mapped_singleton_scope",
                    (candidates[0].runtime_task_id,),
                    "one Gold requirement and one executable runtime scope",
                ))
            else:
                results.append(Alignment(rid, "unmapped", reason="no explicit origin or unique exact atomic-scope match"))
        return tuple(results)


class ReviewedMappingAligner:
    """Apply evaluator-owned reviewed overrides only after deterministic mapping."""

    def __init__(self, mapping: Mapping[str, Any], *, fallback: RequirementAligner | None = None) -> None:
        self.mapping = mapping
        self.fallback = fallback or DeterministicRequirementAligner()

    @classmethod
    def from_json(cls, path: str | Path) -> "ReviewedMappingAligner":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("runtime mapping must be a JSON object")
        return cls(value)

    def align(self, gold_case: Mapping[str, Any], run: EvaluationRunArtifact) -> tuple[Alignment, ...]:
        automatic = {item.gold_requirement_id: item for item in self.fallback.align(gold_case, run)}
        case_mapping = self.mapping.get("cases", {}).get(gold_case["case_id"], {})
        known_tasks = {task.runtime_task_id for task in run.planned_requirements if task.planned_source is not None}
        results = []
        for requirement in gold_case["requirements"]:
            rid = requirement["requirement_id"]
            current = automatic[rid]
            override = case_mapping.get(rid) if isinstance(case_mapping, Mapping) else None
            if current.status.startswith("mapped") or not isinstance(override, Mapping):
                results.append(current)
                continue
            alternatives = override.get("runtime_task_id_sets")
            if isinstance(alternatives, list):
                task_ids = next(
                    (tuple(item) for item in alternatives if isinstance(item, list) and item and all(task_id in known_tasks for task_id in item)),
                    (),
                )
            else:
                task_ids = tuple(override.get("runtime_task_ids", []))
            reason = override.get("reason")
            if not task_ids or any(task_id not in known_tasks for task_id in task_ids) or not isinstance(reason, str) or not reason.strip():
                results.append(Alignment(rid, "mapping_override_invalid", reason="reviewed mapping is missing valid tasks or reason"))
                continue
            results.append(Alignment(rid, "mapped_reviewed_override", task_ids, reason.strip()))
        return tuple(results)


def _normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())
