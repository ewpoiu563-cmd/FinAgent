"""Build evaluator artifacts from captured runtime objects and Phase 4 telemetry."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .models import EvaluationRunArtifact, FallbackTransition, ObservedEvidence, PlannedRequirement
from .runtime_capture import CapturedEvidenceBundle, RuntimeCapture


@dataclass(frozen=True, slots=True)
class TraceBinding:
    run_id: str
    session_id: str | None
    events: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any] | None
    trace_complete: bool
    malformed_trace_lines: int
    invalid_reasons: tuple[str, ...]


class EvaluationRunArtifactBuilder:
    """Read runtime facts without reconstructing them from Gold or the answer."""

    def __init__(self, *, database_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path) if database_path is not None else None

    def build(
        self,
        *,
        case_id: str,
        reference_date: str,
        final_answer: str | None,
        capture: RuntimeCapture,
        trace_path: str | Path,
        run_summary_path: str | Path,
        system_snapshot: Mapping[str, Any],
        final_status: str | None = None,
    ) -> EvaluationRunArtifact:
        run_id = capture.run_id
        if not run_id:
            raise ValueError("runtime capture did not observe a run_id")
        binding = bind_trace_and_summary(trace_path, run_summary_path, run_id)
        initial = capture.initial_task_plan
        terminal = capture.terminal_task_plan
        invalid = list(binding.invalid_reasons)
        if initial is None:
            invalid.append("task_plan_capture_missing")
        if terminal is None:
            invalid.append("terminal_task_plan_missing")
        if capture.dropped_artifacts:
            invalid.append("runtime_capture_incomplete")

        conflicts = _plan_conflicts(initial, binding.events)
        evidence = self._evidence_registry(capture.evidence_bundles, terminal)
        requirements = _planned_requirements(initial, terminal, binding.events, evidence)
        status = final_status or _final_status(terminal, final_answer)
        snapshot = dict(system_snapshot)
        snapshot_id = hashlib.sha256(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return EvaluationRunArtifact(
            case_id=case_id,
            run_id=run_id,
            reference_date=reference_date,
            system_snapshot_id=snapshot_id,
            trace_ref=str(Path(trace_path)),
            run_summary_ref=str(Path(run_summary_path)) if binding.summary is not None else None,
            final_status=status,
            final_answer=final_answer,
            trace_complete=binding.trace_complete,
            planned_requirements=requirements,
            evidence_registry=evidence,
            plan_artifact_conflicts=conflicts,
            invalid_reasons=tuple(dict.fromkeys(invalid)),
            session_id=binding.session_id or capture.session_id,
            system_snapshot=snapshot,
            serialized_task_plan=initial,
            terminal_task_plan=terminal,
            run_summary=binding.summary,
            trace_malformed_line_count=binding.malformed_trace_lines,
        )

    def _evidence_registry(
        self,
        bundles: Sequence[CapturedEvidenceBundle],
        terminal_plan: Mapping[str, Any] | None,
    ) -> tuple[ObservedEvidence, ...]:
        execution_ids = [
            str(item["id"])
            for item in (terminal_plan or {}).get("tasks", [])
            if item.get("source") is not None
        ]
        database_identity = self._database_identity()
        observed: list[ObservedEvidence] = []
        seen: set[tuple[Any, ...]] = set()
        for captured in bundles:
            source = str(captured.payload.get("source_type") or captured.source or "")
            owner = captured.runtime_task_id
            if owner is None and len(execution_ids) == 1:
                owner = execution_ids[0]
            raw_bundle = captured.payload.get("raw_tool_result")
            for position, item in enumerate(captured.payload.get("items", []), start=1):
                if not isinstance(item, Mapping):
                    continue
                source = str(item.get("source_type") or source)
                retrieval_id = _text(item.get("retrieval_unit_id")) or f"{source}-{position}"
                runtime_evidence_id = f"{owner}:{retrieval_id}" if owner else retrieval_id
                identity, content, fact_as_of, publication_date = _evidence_identity(
                    source, item, raw_bundle, database_identity
                )
                key = (owner, source, retrieval_id, identity.get("url"), identity.get("doc_id"))
                if key in seen:
                    continue
                seen.add(key)
                observed.append(
                    ObservedEvidence(
                        evidence_id=runtime_evidence_id,
                        source=source,
                        identity=identity,
                        content=content,
                        runtime_task_id=owner,
                        fact_as_of=fact_as_of,
                        publication_date=publication_date,
                    )
                )
        return tuple(observed)

    def _database_identity(self) -> Mapping[str, Any]:
        path = self.database_path
        if path is None:
            try:
                from text2sql.schema_provider import DEFAULT_DB_PATH

                path = Path(DEFAULT_DB_PATH)
            except Exception:
                return {}
        identity: dict[str, Any] = {"database_file": str(path)}
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            identity["database_sha256"] = digest.hexdigest()
        except OSError:
            pass
        return identity


def bind_trace_and_summary(
    trace_path: str | Path,
    summary_path: str | Path,
    run_id: str,
) -> TraceBinding:
    trace_file = Path(trace_path)
    invalid: list[str] = []
    malformed = 0
    events: list[Mapping[str, Any]] = []
    if not trace_file.exists():
        invalid.append("raw_trace_missing")
    else:
        with trace_file.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    malformed += 1
                    continue
                if not isinstance(value, Mapping):
                    malformed += 1
                    continue
                if value.get("run_id") == run_id:
                    events.append(dict(value))
    if malformed:
        invalid.append("raw_trace_malformed")
    if not events:
        invalid.append("run_trace_missing")

    summary = _read_summary(summary_path, run_id, invalid)
    trace_run_ids = {str(event.get("run_id")) for event in events}
    if trace_run_ids and trace_run_ids != {run_id}:
        invalid.append("trace_run_id_mismatch")
    if summary is not None and summary.get("run_id") != run_id:
        invalid.append("run_summary_run_id_mismatch")
    session_ids = {event.get("session_id") for event in events if event.get("session_id") is not None}
    if len(session_ids) > 1:
        invalid.append("trace_session_id_conflict")
    session_id = next(iter(session_ids), None)
    if summary is not None and summary.get("session_id") not in {None, session_id}:
        invalid.append("run_summary_session_id_mismatch")
    has_start = any(event.get("event_type") == "run.started" for event in events)
    has_terminal = any(event.get("event_type") == "run.completed" for event in events)
    complete = bool(
        has_start
        and has_terminal
        and malformed == 0
        and summary is not None
        and summary.get("trace_complete") is True
        and not invalid
    )
    return TraceBinding(
        run_id,
        _text(session_id),
        tuple(events),
        summary,
        complete,
        malformed,
        tuple(dict.fromkeys(invalid)),
    )


def _read_summary(path: str | Path, run_id: str, invalid: list[str]) -> Mapping[str, Any] | None:
    source = Path(path)
    if not source.exists():
        invalid.append("run_summary_missing")
        return None
    matches: list[Mapping[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("summary line is not an object")
                if value.get("run_id") == run_id:
                    matches.append(dict(value))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        invalid.append("run_summary_malformed")
        return None
    if not matches:
        invalid.append("run_summary_missing")
        return None
    if len(matches) > 1:
        invalid.append("duplicate_run_summary")
    return matches[-1]


def _planned_requirements(initial, terminal, events, evidence) -> tuple[PlannedRequirement, ...]:
    initial_tasks = {str(item.get("id")): item for item in (initial or {}).get("tasks", [])}
    terminal_tasks = {str(item.get("id")): item for item in (terminal or {}).get("tasks", [])}
    identifiers = list(dict.fromkeys([*initial_tasks, *terminal_tasks]))
    evidence_ids_by_task: dict[str, list[str]] = {}
    for item in evidence:
        if item.runtime_task_id:
            evidence_ids_by_task.setdefault(item.runtime_task_id, []).append(item.evidence_id)
    results: list[PlannedRequirement] = []
    for task_id in identifiers:
        planned = initial_tasks.get(task_id, {})
        ended = terminal_tasks.get(task_id, planned)
        source = _text(planned.get("source"))
        decision = planned.get("tool_decision") if isinstance(planned.get("tool_decision"), Mapping) else {}
        mode = _text(decision.get("mode")) or (
            "no_tool" if source in {"direct", "local_compute"} else "tool_required" if source else "not_applicable"
        )
        actual, outcomes = _source_execution(task_id, events)
        if initial and initial.get("capture_kind") == "legacy_runtime_capture" and source:
            actual = (source,)
            outcomes = {source: str(ended.get("status") or "pending")}
        transitions = _fallbacks(task_id, events, outcomes)
        missing = [str(item) for item in ended.get("missing_information", [])]
        results.append(
            PlannedRequirement(
                runtime_task_id=task_id,
                atomic_question=str(planned.get("question") or ended.get("question") or ""),
                required=bool(planned.get("required", ended.get("required", True))),
                depends_on=tuple(str(item) for item in planned.get("depends_on", [])),
                tool_necessity_mode=mode,
                planned_source=source,
                terminal_status=str(ended.get("status") or "pending"),
                planned_status=str(planned.get("status") or "pending"),
                answer_fragment=_text(ended.get("answer_fragment")),
                evidence_ids=tuple(dict.fromkeys([*ended.get("evidence_ids", []), *evidence_ids_by_task.get(task_id, [])])),
                actual_sources=actual,
                fallback_transitions=transitions,
                source_outcomes=outcomes,
                dependency_failure=any("dependency" in item.casefold() for item in missing),
            )
        )
    return tuple(results)


def _source_execution(task_id: str, events: Sequence[Mapping[str, Any]]):
    relevant = [event for event in events if event.get("task_id") == task_id and str(event.get("event_type", "")).startswith("source.execution.")]
    if not relevant:
        relevant = [
            event
            for event in events
            if event.get("task_id") == task_id
            and event.get("source") in {"direct", "local_compute"}
            and str(event.get("event_type", "")).startswith("task.execution.")
        ]
    actual = tuple(dict.fromkeys(str(event.get("source")) for event in relevant if _text(event.get("source"))))
    outcomes: dict[str, str] = {}
    for event in relevant:
        if event.get("event_type") not in {"source.execution.completed", "source.execution.failed", "task.execution.completed", "task.execution.failed"}:
            continue
        source = _text(event.get("source"))
        if source:
            attributes = event.get("attributes") if isinstance(event.get("attributes"), Mapping) else {}
            outcomes[source] = str(attributes.get("source_result_status") or event.get("status") or "unknown")
    return actual, outcomes


def _fallbacks(task_id: str, events, outcomes) -> tuple[FallbackTransition, ...]:
    transitions: list[FallbackTransition] = []
    for event in events:
        if event.get("event_type") != "fallback.dispatched" or event.get("task_id") not in {None, task_id}:
            continue
        attributes = event.get("attributes") if isinstance(event.get("attributes"), Mapping) else {}
        source_from = _text(attributes.get("from_source")) or _text(event.get("source")) or _text(attributes.get("from_dispatch"))
        source_to = _text(attributes.get("to_source")) or _text(attributes.get("target"))
        if not source_from or not source_to or source_to == "legacy_react":
            continue
        reason = str(attributes.get("reason") or "")
        original = str(attributes.get("original_status") or outcomes.get(source_from) or "")
        trigger = "zero_result" if source_from == "sql" and original == "insufficient" else _canonical_trigger(reason, original)
        transitions.append(FallbackTransition(source_from, source_to, trigger, task_id))
    return tuple(transitions)


def _canonical_trigger(reason: str, status: str) -> str | None:
    text = f"{reason} {status}".casefold()
    for value in ("provider_content_block", "generation_failure", "source_failure", "insufficient"):
        if value in text:
            return value
    return status or reason or None


def _plan_conflicts(initial, events) -> tuple[str, ...]:
    if initial is None:
        return ()
    if initial.get("capture_kind") == "legacy_runtime_capture":
        return ()
    source_plan = initial.get("source_plan") if isinstance(initial.get("source_plan"), Mapping) else {}
    candidates = [event for event in events if event.get("event_type") == "planning.source_plan.completed"]
    if not candidates:
        return ("planning.source_plan.completed event missing",)
    attributes = candidates[-1].get("attributes") if isinstance(candidates[-1].get("attributes"), Mapping) else {}
    conflicts = []
    planned_primary = source_plan.get("primary_source")
    event_primary = attributes.get("primary_source")
    if planned_primary != event_primary:
        conflicts.append(f"primary_source: TaskPlan={planned_primary!r}, trace={event_primary!r}")
    planned_sources = tuple(source_plan.get("required_sources", []))
    event_sources = tuple(attributes.get("required_sources", []))
    if planned_sources != event_sources:
        conflicts.append(f"required_sources: TaskPlan={planned_sources!r}, trace={event_sources!r}")
    return tuple(conflicts)


def _evidence_identity(source, item, raw_bundle, database_identity):
    raw = item.get("raw") if isinstance(item.get("raw"), Mapping) else {}
    content = _text(item.get("content")) or _text(item.get("text")) or _text(item.get("snippet"))
    fact_as_of = _text(raw.get("fact_as_of")) or _text(raw.get("as_of")) or _text(raw.get("reference_date"))
    publication = _text(raw.get("publication_date")) or _text(raw.get("published_at")) or _text(raw.get("date"))
    if source == "sql":
        canonical_sql = _canonical_sql(item.get("generated_sql"))
        tables = re.findall(r"\b(?:FROM|JOIN)\s+\[([^\]]+)\]", canonical_sql or "", re.IGNORECASE)
        identity = dict(database_identity)
        identity.update(
            generated_sql=item.get("generated_sql"),
            canonical_sql=canonical_sql,
            query_artifact_id=(f"runtime-sql-sha256:{hashlib.sha256(canonical_sql.encode('utf-8')).hexdigest()}" if canonical_sql else None),
            table=tables[0] if tables else None,
            joined_table=tables[1] if len(tables) > 1 else None,
            columns=list(item.get("columns", [])),
            rows=list(item.get("rows", [])),
            row_count=item.get("row_count"),
            truncated=item.get("truncated"),
            parameters=(raw_bundle or {}).get("parameters") if isinstance(raw_bundle, Mapping) else None,
        )
    elif source == "rag":
        identity = {
            "catalog_doc_id": item.get("catalog_doc_id"),
            "doc_id": item.get("index_doc_id"),
            "retrieval_unit_id": item.get("retrieval_unit_id"),
            "pages": list(item.get("page", [])),
            "headings": list(item.get("headings", [])),
            "source_file": item.get("source_file"),
        }
    elif source == "web":
        url = _canonical_url(str(item.get("url") or ""))
        title = _text(item.get("title"))
        publication = publication or _date_from_text(title or "")
        identity = {
            "url": url,
            "publisher": raw.get("publisher"),
            "publisher_domain": _text(raw.get("publisher_domain")) or (urlsplit(url).hostname if url else None),
            "publication_date": publication,
            "page_title": item.get("title"),
            "content_sha256": hashlib.sha256((content or "").encode("utf-8")).hexdigest() if content else None,
        }
    else:
        identity = dict(raw)
    return {key: value for key, value in identity.items() if value is not None}, content, fact_as_of, publication


def _canonical_sql(value: Any) -> str | None:
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) and value.strip() else None


def _canonical_url(value: str) -> str:
    if not value:
        return value
    parts = urlsplit(value)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), path, parts.query, ""))


def _date_from_text(value: str) -> str | None:
    match = re.search(r"((?:19|20)\d{2})[年./-]?(\d{1,2})[月./-]?(\d{1,2})日?", value)
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}" if match else None


def _final_status(terminal, answer) -> str:
    if terminal and _text(terminal.get("completion_status")):
        status = str(terminal["completion_status"])
        return "source_failure" if status == "insufficient" else status
    text = answer or ""
    for status in ("partial_success", "clarification_needed", "source_failure", "generation_failure", "provider_content_block"):
        if text.startswith(status):
            return status
    return "success" if text.strip() else "source_failure"


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["EvaluationRunArtifactBuilder", "TraceBinding", "bind_trace_and_summary"]
