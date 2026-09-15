"""Fail-open, evaluator-owned copies of ephemeral production runtime objects."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CapturedEvidenceBundle:
    payload: Mapping[str, Any]
    run_id: str | None
    session_id: str | None
    runtime_task_id: str | None
    source: str | None


@dataclass
class RuntimeCapture:
    """Observer implementation whose failures never affect the Agent run."""

    run_id: str | None = None
    session_id: str | None = None
    initial_task_plan: Mapping[str, Any] | None = None
    _task_plan_object: Any = field(default=None, repr=False)
    evidence_bundles: list[CapturedEvidenceBundle] = field(default_factory=list)
    dropped_artifacts: list[str] = field(default_factory=list)
    legacy_path_captured: bool = False

    def observe(self, kind: str, value: Any, metadata: Mapping[str, Any]) -> None:
        try:
            self.run_id = self.run_id or _text(metadata.get("run_id"))
            self.session_id = self.session_id or _text(metadata.get("session_id"))
            if kind == "task_plan.planned":
                self._task_plan_object = value
                self.initial_task_plan = copy.deepcopy(value.to_dict())
            elif kind == "legacy.run.started":
                question = _text(value.get("question")) if isinstance(value, Mapping) else None
                if question is None:
                    raise ValueError("legacy run question missing")
                self.legacy_path_captured = True
                self.initial_task_plan = _legacy_plan(question)
            elif kind == "evidence.bundle":
                payload = copy.deepcopy(value.to_dict(include_raw=True))
                self.evidence_bundles.append(
                    CapturedEvidenceBundle(
                        payload=payload,
                        run_id=_text(metadata.get("run_id")),
                        session_id=_text(metadata.get("session_id")),
                        runtime_task_id=_text(metadata.get("task_id")),
                        source=_text(metadata.get("source")),
                    )
                )
            elif kind == "legacy.run.completed":
                if not isinstance(value, Mapping):
                    raise ValueError("legacy run result must be a mapping")
                answer = _text(value.get("answer"))
                if self.initial_task_plan is None or not self.legacy_path_captured:
                    raise ValueError("legacy run start missing")
                terminal = copy.deepcopy(self.initial_task_plan)
                terminal["tasks"][0]["status"] = "success" if answer else "generation_failure"
                terminal["tasks"][0]["answer_fragment"] = answer
                terminal["completion_status"] = terminal["tasks"][0]["status"]
                self._task_plan_object = _CapturedPlan(terminal)
            elif kind == "legacy.tool_result":
                if not self.legacy_path_captured or not isinstance(value, Mapping):
                    raise ValueError("legacy tool result outside captured legacy run")
                source, payload = _legacy_bundle(value)
                plan = self.initial_task_plan
                if isinstance(plan, dict):
                    plan["source_plan"]["primary_source"] = source
                    plan["source_plan"]["required_sources"] = [source]
                    plan["tasks"][0]["source"] = source
                self.evidence_bundles.append(
                    CapturedEvidenceBundle(
                        payload=payload,
                        run_id=_text(metadata.get("run_id")),
                        session_id=_text(metadata.get("session_id")),
                        runtime_task_id="LEGACY",
                        source=source,
                    )
                )
        except Exception as error:  # pragma: no cover - outer hook is fail-open too
            self.dropped_artifacts.append(f"{kind}:{type(error).__name__}")

    @property
    def terminal_task_plan(self) -> Mapping[str, Any] | None:
        if self._task_plan_object is None:
            return None
        try:
            return copy.deepcopy(self._task_plan_object.to_dict())
        except Exception as error:
            self.dropped_artifacts.append(f"task_plan.terminal:{type(error).__name__}")
            return None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True, slots=True)
class _CapturedPlan:
    value: Mapping[str, Any]

    def to_dict(self) -> Mapping[str, Any]:
        return self.value


def _legacy_plan(question: str) -> dict[str, Any]:
    """Minimal observer-owned projection for a legacy fallback execution."""
    return {
        "original_question": question,
        "resolved_question": question,
        "source_plan": {
            "primary_source": "legacy_react",
            "required_sources": ["legacy_react"],
            "routing_method": "legacy_runtime_capture",
        },
        "tasks": [{
            "id": "LEGACY",
            "question": question,
            "source": "legacy_react",
            "required": True,
            "depends_on": [],
            "status": "pending",
            "evidence_ids": [],
            "answer_fragment": None,
            "missing_information": [],
            "tool_decision": {"mode": "tool_required"},
        }],
        "completion_status": "pending",
        "capture_kind": "legacy_runtime_capture",
    }


def _legacy_bundle(value: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    tool = _text(value.get("tool"))
    query = _text(value.get("query")) or ""
    result = value.get("result") if isinstance(value.get("result"), Mapping) else {}
    if tool == "retrieve_document":
        items = [
            {
                "source_type": "rag",
                "retrieval_unit_id": item.get("retrieval_unit_id"),
                "index_doc_id": item.get("doc_id"),
                "pages": item.get("page") or item.get("pages") or [],
                "content": item.get("text"),
                "raw": item,
            }
            for item in result.get("evidence", [])
            if isinstance(item, Mapping)
        ]
        return "rag", {"source_type": "rag", "items": items, "raw_tool_result": dict(result)}
    if tool == "query_financial_db":
        item = {
            "source_type": "sql", "retrieval_unit_id": "sql-result-1",
            "generated_sql": result.get("generated_sql"), "columns": result.get("columns", []),
            "rows": result.get("rows", []), "row_count": result.get("row_count", 0),
            "truncated": result.get("truncated", False), "raw": {"question": query},
        }
        return "sql", {"source_type": "sql", "items": [item], "raw_tool_result": dict(result)}
    raise ValueError(f"unsupported captured legacy tool: {tool}")


__all__ = ["CapturedEvidenceBundle", "RuntimeCapture"]
